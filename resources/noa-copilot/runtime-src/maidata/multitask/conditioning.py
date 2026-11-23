"""Chart metadata and optional beat phase; the encoder remains audio-only."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .grid import BeatGrid


def phase_from_beats(beat, numerator=None, meters=None, *, metadata=None, grid=None):
    beat = torch.as_tensor(beat, dtype=torch.float32)
    if grid is None:
        meta = dict(metadata) if metadata is not None else {}
        if numerator is not None and "numerator" not in meta:
            meta["numerator"] = numerator
        if meters is not None and "meter_segments" not in meta:
            meta["meter_segments"] = meters
        grid = BeatGrid(meta)
    result = beat.new_zeros((*beat.shape, 32))
    harmonics = beat.new_tensor([1, 2, 3, 4, 6, 8], dtype=torch.float32)
    angle = beat.remainder(1)[..., None] * harmonics * (2 * math.pi)
    result[..., :12] = torch.stack((angle.sin(), angle.cos()), -1).flatten(-2)

    b_info = grid.bar_info(beat)
    t_info = grid.tempo_info(beat)
    if b_info["has_meter"]:
        bar_angle = b_info["bar_phase"] * (2 * math.pi)
        result[..., 12] = bar_angle.sin()
        result[..., 13] = bar_angle.cos()
        result[..., 15] = 1.0
        result[..., 16] = b_info["is_downbeat"]
        bib_angle = (b_info["beat_in_bar"].float() / 16.0) * (2 * math.pi)
        result[..., 17] = bib_angle.sin()
        result[..., 18] = bib_angle.cos()
        result[..., 19] = torch.log2(b_info["bar_beats"].clamp_min(0.25) / 4.0)
        result[..., 20] = b_info["num"].float() / 16.0
        log2_d = torch.log2(b_info["den"].float().clamp_min(1.0))
        result[..., 21] = log2_d / 4.0
        result[..., 22] = (log2_d * (math.pi / 2.0)).sin()
        result[..., 23] = (log2_d * (math.pi / 2.0)).cos()
    result[..., 14] = 1.0

    if t_info["has_tempo"]:
        u_bpm = torch.log2(t_info["bpm"] / 120.0)
        u_rel = torch.log2(t_info["bpm"] / t_info["base_bpm"])
        result[..., 24] = u_bpm
        result[..., 25] = u_rel
        result[..., 26] = (math.pi * u_bpm).sin()
        result[..., 27] = (math.pi * u_bpm).cos()
        result[..., 28] = (4.0 * math.pi * u_bpm).sin()
        result[..., 29] = (4.0 * math.pi * u_bpm).cos()
        result[..., 30] = torch.tanh(t_info["slope"] / 60.0)
        result[..., 31] = t_info["is_stop"]
    return result


def sample_history_seconds(count: int, device, training: bool = True) -> Tensor:
    """Per-chart prefix truncation, with the exact three probability strata."""
    if not training:
        return torch.full((count,), 8, device=device, dtype=torch.long)
    stratum = torch.rand(count, device=device)
    middle = .15 + torch.rand(count, device=device) * .10
    high = .5 + torch.rand(count, device=device) * .5
    ratio = torch.where(stratum < .75, torch.zeros_like(stratum),
                        torch.where(stratum < .95, middle, high))
    return (8 * (1 - ratio)).round().long()


@dataclass
class TaskRouting:
    chart_rows: tuple[Tensor, ...]
    token_rows: tuple[Tensor, ...]
    role_rows: tuple[tuple[Tensor, Tensor], ...]
    task_ids: tuple[int, ...]
    branch: str = "main"


def prepare_routing(task_ids, lengths, roles, sequence_length, tasks, device, branch="main"):
    """Static integer gathers prepared from CPU metadata once per forward."""
    ids = tuple(int(x) for x in torch.as_tensor(task_ids).tolist())
    lens = torch.as_tensor(lengths).tolist()
    roles = torch.as_tensor(roles).tolist()
    charts, tokens, by_role = [], [], []
    pieces = []
    specifications = []
    for task in range(tasks):
        cr = [b for b, value in enumerate(ids) if value == task]
        tr = [b * sequence_length + i for b in cr for i in range(lens[b])]
        rr = [[b * sequence_length + i for b in cr for i in range(lens[b])
               if roles[b][i] == role] for role in (0, 1)]
        arrays = [torch.tensor(v, dtype=torch.long) for v in (cr, tr, *rr)]
        specifications.append([len(v) for v in arrays])
        pieces.extend(arrays)
    packed = torch.cat(pieces).to(device, non_blocking=True)
    cursor = 0
    for sizes in specifications:
        views = []
        for size in sizes:
            views.append(packed[cursor:cursor + size])
            cursor += size
        charts.append(views[0]); tokens.append(views[1]); by_role.append(tuple(views[2:]))
    return TaskRouting(tuple(charts), tuple(tokens), tuple(by_role), ids, branch)


class TaskStateEmbedding(nn.Module):
    """Task-isolated embeddings for discrete slot/category pairs, never magnitudes."""
    def __init__(self, config):
        super().__init__()
        self.tasks = tuple(config.tasks)
        self.embeddings = nn.ModuleDict({task: nn.Embedding(32 * 12, 16) for task in self.tasks})
        self.projections = nn.ModuleDict({task: nn.Sequential(nn.Linear(32 * 16, config.d_model),
            nn.GELU(), nn.Linear(config.d_model, config.d_model)) for task in self.tasks})
        self.register_buffer("slot_offsets", torch.arange(32) * 12, persistent=False)

    def forward(self, states, routing):
        output = None
        for task, rows in zip(self.tasks, routing.chart_rows):
            if rows.numel():
                values = states.index_select(0, rows).long() + self.slot_offsets
                embedded = self.embeddings[task](values).flatten(-2)
                projected = self.projections[task](embedded)
                if len(rows) == len(states):
                    return projected
                if output is None:
                    output = projected.new_empty((*states.shape[:2], projected.shape[-1]))
                output = output.index_copy_(0, rows, projected)
        return output


class ChartConditioner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tasks = tuple(config.tasks)
        dim = config.d_model
        self.task = nn.Embedding(len(self.tasks), dim)
        self.level_mlp = nn.Sequential(nn.Linear(16, dim), nn.GELU(), nn.Linear(dim, dim))
        self.null_level = nn.Parameter(torch.zeros(dim))
        self.difficulty = nn.ModuleDict({task: nn.Embedding(config.difficulty_vocab_size, dim)
                                         for task in self.tasks})
        self.designer = nn.ModuleDict({task: nn.Embedding(config.designer_vocab_size, dim)
                                       for task in self.tasks})
        self.style_tag_emb = nn.Embedding(16, dim, padding_idx=0)
        from .timbre_tags import NUM_COARSE_TAGS, NUM_FINE_TAGS
        self.query_coarse_emb = nn.Embedding(NUM_COARSE_TAGS + 1, dim, padding_idx=0)
        self.query_fine_emb = nn.Embedding(NUM_FINE_TAGS + 1, dim, padding_idx=0)
        self.bpm_present = nn.Embedding(2, dim)
        self.phase = nn.Sequential(nn.Linear(32, dim), nn.GELU(), nn.Linear(dim, dim))
        self.null_phase = nn.Parameter(torch.zeros(dim))
        self.state = TaskStateEmbedding(config)
        self.last_phase_active: Tensor | None = None

    def forward(self, task_ids: Tensor, beats: Tensor, metadata=None, stage: int = 1,
                *, force_unconditional: bool = False, task_ids_cpu=None):
        """Return one chart vector and per-token phase vectors.

        Metadata is a list of chart dicts. A tensor dictionary is accepted too.
        Optional ``bpm_segments``, ``offset_seconds``, ``numerator`` derive phase
        here, including the prescribed per-chart offset jitter. Explicit
        ``phase_features`` are useful for inference/cached preprocessing.
        """
        device, batch, length = task_ids.device, len(task_ids), beats.shape[1]
        if metadata is None:
            metadata = [{} for _ in range(batch)]
        elif isinstance(metadata, dict):
            metadata = [{key: (value[n] if isinstance(value, (Tensor, list, tuple)) else value)
                         for key, value in metadata.items()} for n in range(batch)]
        conditions, phases, phase_active = [], [], []
        ids = task_ids.detach().cpu().tolist() if task_ids_cpu is None else task_ids_cpu
        draws = torch.rand((batch, 9), device="cpu").tolist() if self.training else [[1.] * 9 for _ in ids]
        for n, (task_id, meta) in enumerate(zip(ids, metadata)):
            task = self.tasks[task_id]
            joint_drop = force_unconditional or draws[n][0] < .05
            drop_level = joint_drop or draws[n][1] < .1
            drop_difficulty = joint_drop or draws[n][2] < .1
            drop_designer = joint_drop or draws[n][3] < .25
            level = meta.get("level")
            if level is None or drop_level:
                lv = self.null_level
            else:
                low, high = self.config.level_ranges.get(task, (0., 1.))
                value = torch.as_tensor(level, device=device, dtype=torch.float32)
                # No clamping: rare out-of-range chart ratings remain distinct.
                angle = (value - low) / (high - low) * torch.arange(1, 9, device=device) * math.pi
                lv = self.level_mlp(torch.cat((angle.sin(), angle.cos())))
            difficulty = 0 if drop_difficulty else int(meta.get("difficulty", 0) or 0)
            designer = 0 if drop_designer else int(meta.get("designer", 0) or 0)
            raw_phase = meta.get("phase_features")
            segments = meta.get("bpm_segments")
            if not segments and meta.get("bpm"):
                segments = [(0., float(meta["bpm"]))]
            has_phase = bool(meta.get("phase_available", raw_phase is not None or bool(segments)
                                      or bool(meta.get("meter_segments")) or bool(meta.get("measure_multipliers"))
                                      or meta.get("numerator") is not None))
            if has_phase and draws[n][4] < .3:
                has_phase = False
            has_phase = has_phase and not force_unconditional
            phase_active.append(has_phase)
            if has_phase:
                if raw_phase is None:
                    grid = meta.get("_grid") or BeatGrid(meta)
                    accepted = beats[n]
                    if draws[n][5] < .5:
                        origin = grid.beat_to_second(0.)
                        accepted = accepted - grid.second_to_beat(origin + (draws[n][6]*.04-.02))
                    raw_phase = phase_from_beats(accepted, meta.get("numerator"), meta.get("meter_segments"),
                                                 metadata=meta, grid=grid)
                raw_phase = torch.as_tensor(raw_phase, device=device, dtype=torch.float32)
                if raw_phase.shape[-1] < 32:
                    raw_phase = F.pad(raw_phase, (0, 32 - raw_phase.shape[-1]))
                pv = self.phase(raw_phase[:length])
            else:
                pv = self.null_phase.expand(length, -1)
            present_id = torch.tensor(int(has_phase), device=device)
            condition = (self.task.weight[task_id] + lv + self.difficulty[task].weight[difficulty] +
                         self.designer[task].weight[designer] + self.bpm_present(present_id))
            from .styles import parse_style_tags
            tags = parse_style_tags(meta.get('style_tags', [meta.get('primary_tag',0),meta.get('secondary_tag',0)]))
            if stage == 0 or task == 'midi' or force_unconditional or draws[n][7] < .25:
                tags = [0,0]
            elif draws[n][8] < .25:
                tags[1] = 0
            style=self.style_tag_emb(torch.tensor(tags,device=device))
            from .timbre_tags import resolve_tag_query
            q_coarse, q_fine = (0, 0) if force_unconditional else resolve_tag_query(meta)
            q_emb = (self.query_coarse_emb(torch.tensor(q_coarse, device=device)) +
                     self.query_fine_emb(torch.tensor(q_fine, device=device)))
            condition = condition + style[0] + .5*style[1] + q_emb
            conditions.append(condition)
            phases.append(pv)
        self.last_phase_active = torch.tensor(phase_active, device=device, dtype=torch.bool)
        return torch.stack(conditions), torch.stack(phases)
