"""Single-stream RuleMMoE with independent main and minimal-history token states."""

from dataclasses import dataclass, field
import math

import torch
from torch import nn
from torch.nn import functional as F

from maidata.chart.routes import catalog
from . import ARCHITECTURE, ARCHITECTURE_V187, DRAFT_ARCHITECTURES, ARCHITECTURE_V20
from .drafter import ChartDrafter, BeatTEReadout, DecoderConditionAttention, ConditionAdapter, ConditionMemory, condition_rope
from .reconstruction import FlatReconstructionHead, reconstruction_loss, task_mean
from .emotion import EmotionCurvePlanner, Plan4Embedder, emit_features, planner_loss
from .heads import HeteroInputHeads, RMSNorm
from .audio import BeatGridEncoder, DualAudioFusion, STREAMS, audio_rotary
from maidata.decoder.projections import packed_projection_scope
from maidata.designer import DESIGNER_VOCAB
from maidata.multitask.attention import rotary
from maidata.spine.features import CLOCK_DIM
from maidata.streaming.network import RuleMMoE, TaskRuleMMoE

from .codec import FAMILY_IDS, NEW_PADS, SIZE, SIZE_V20, values
from .layouts import AttentionLayout, AuxRequest

SELF_GUIDE_NAMES = ("occ", "near", "slide")


@dataclass
class FlatConfigV187:
    architecture = ARCHITECTURE_V187
    stage: int = 0
    dim: int = 256
    layers: int = 6
    heads: int = 8
    experts: int = 6
    expert_dim: int = 512
    conv_heads: int = 4
    mert_heads: int = 4
    conv_kv_heads: int = 4
    mert_kv_heads: int = 2
    head_dim: int = 64
    beat_encoder_layers: int = 2
    spec_augment_bands: int = 12
    mel_gain_jitter: float = 0.08
    mert_dropout: float = 0.10
    no_history_enabled: bool = True
    plan4_enabled: bool = True
    emotion_slide_boost: float = 0.20
    plan_warmup_steps: int = 800
    plan_log_weight: float = 0.0
    plan_chart_dropout: float = 0.10
    plan_bar_dropout: float = 0.05
    planner_loss_weight: float = 0.10
    plan_lr_scale: float = 2.0
    conv_past_beats: float = 1.5
    conv_future_beats: float = 1.5
    mert_past_beats: float = 8.0
    mert_future_beats: float = 8.0
    self_rotary_dims: int = 24
    self_rope_scale: float = 8.0
    self_rope_base: float = 64.0
    conv_rope_scale: float = 2.0
    conv_rope_base: float = 64.0
    mert_rope_scale: float = 0.35
    mert_rope_base: float = 64.0
    dropout: float = 0.0
    history_query_block: int = 128
    history_backend: str = "auto"
    attn_guide_weight: float = 0.05
    timing_history_dropout: float = 0.15
    slide_icebreak_beats: float = 6.0
    slide_icebreak_weight: float = 2.0
    slide_lookahead_weight: float = 0.03
    attn_guide_layers: tuple[float, ...] = (0.0, 0.0, 0.10, 0.20, 0.30, 0.40)
    condition_dropout: float = 0.05
    level_dropout: float = 0.10
    difficulty_dropout: float = 0.10
    charter_dropout: float = 0.25
    designer_vocab: list[str] = field(default_factory=lambda: list(DESIGNER_VOCAB))

    @property
    def inline_plan(self):
        return self.plan4_enabled and self.architecture == ARCHITECTURE_V187

    def __post_init__(self):
        self.attn_guide_layers = tuple(self.attn_guide_layers)


@dataclass
class FlatConfig(FlatConfigV187):
    architecture = ARCHITECTURE
    draft_loss_weight: float = 0.30
    plan_warmup_steps: int = 0
    planner_loss_weight: float = 0.20
    plan_lr_scale: float = 1.0
    plan_log_weight: float = 1.0
    plan_chart_dropout: float = 0.30
    draft_layers: int = 6
    draft_dim: int = 256
    draft_heads: int = 8
    draft_experts: int = 1
    draft_expert_dim: int = 1024
    draft_dropout: float = 0.10


@dataclass
class FlatConfigV20(FlatConfig):
    architecture = ARCHITECTURE_V20
    dim: int = 320
    draft_dim: int = 320
    draft_experts: int = 3
    draft_expert_dim: int = 640
    expert_dim: int = 640
    experts: int = 8
    self_rotary_dims: int = 32
    no_history_enabled: bool = False
    taiko_loss_weight: float = 1.0


def model_config(recipe):
    cls = {ARCHITECTURE_V187: FlatConfigV187, ARCHITECTURE_V20: FlatConfigV20}.get(recipe.get("architecture"), FlatConfig)
    return cls(**{"stage": recipe.get("stage", 0), **recipe.get("model", {})})


def partial_rotary(x, positions, *, scale, base, rotary_dims=None):
    prefix = x if rotary_dims is None else x[..., :rotary_dims]
    rotated = rotary(prefix, positions * scale, base)
    if prefix.shape[-1] == x.shape[-1]:
        return rotated
    return torch.cat((rotated, x[..., rotary_dims:]), -1)


class Grid32TimingHead(nn.Module):
    """Factor P(next) into a binary emit decision and a conditional GRID32 slot."""

    _version = 2  # Version 1 checkpoints stored the absolute midpoint projection.

    def __init__(self, dim, history_dropout=0.15, plan4_enabled=True, multitask=False):
        super().__init__()
        if multitask:
            self.task_slot_bias = nn.Embedding(2, 32)
            self.task_emit_bias = nn.Embedding(2, 1)
            nn.init.zeros_(self.task_slot_bias.weight)
            nn.init.zeros_(self.task_emit_bias.weight)
        self.h_drop = nn.Dropout(history_dropout)
        self.sub64_proj = nn.Linear(2 * dim, dim, bias=False)
        nn.init.zeros_(self.sub64_proj.weight)
        self.slot_acoustic = nn.Linear(dim, 1, bias=False)
        self.slot_key = nn.Linear(dim, dim, bias=False)
        self.slot_query = nn.Linear(dim, dim, bias=False)
        self.slot_prior = nn.Linear(dim, 32)
        self.emit_gate = nn.Sequential(
            nn.Linear(3 * dim + 3 + 4 * plan4_enabled, dim), nn.GELU(), nn.Linear(dim, 1)
        )
        self.plan4_enabled = plan4_enabled
        if plan4_enabled:
            with torch.no_grad():
                self.emit_gate[0].weight[:, -4:].zero_()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, *args):
        key = prefix + "sub64_proj.weight"
        if local_metadata.get("version") == 1 and key in state_dict:
            weight = state_dict[key].clone()
            for half in weight.chunk(2, dim=1):
                half.diagonal().sub_(0.5)
            state_dict[key] = weight
        super()._load_from_state_dict(state_dict, prefix, local_metadata, *args)

    def evidence(self, feat):
        beats = feat.reshape(-1, 24, feat.size(-1))
        a, b, c = beats.unflatten(1, (8, 3)).unbind(2)
        mid = 0.5 * (b + c) + self.sub64_proj(torch.cat((b, c), -1))
        slots = torch.stack((a, b, mid, c), 2).flatten(1, 2)
        return (
            slots,
            self.slot_acoustic(slots).squeeze(-1),
            self.slot_key(slots),
            beats.mean(1),
        )

    def forward(self, logits, hidden, rows, beat, legal, evidence, plan_features=None, task_ids=None, terminal=None):
        if not len(rows):
            return logits.float()
        slots, acoustic, keys, mean = evidence
        h = hidden[rows]
        history = self.h_drop(h)
        q = self.slot_query(h)
        # bmm reduces over channels without constructing a row x slot x dim product.
        match = torch.bmm(keys[beat], q.unsqueeze(2)).squeeze(2) * q.size(-1) ** -0.5
        u = (
            acoustic[beat].float() + match.float() + self.slot_prior(history).float()
        )
        if hasattr(self, "task_slot_bias"):
            task_ids = torch.zeros_like(beat) if task_ids is None else task_ids
            u = u + self.task_slot_bias(task_ids)
        u = u.masked_fill(~legal, -torch.inf)
        lse = u.logsumexp(-1, keepdim=True)
        count = legal.sum(-1, keepdim=True).float()
        rem = torch.bmm(
            (legal / count).to(slots.dtype).unsqueeze(1), slots[beat]
        ).squeeze(1)
        stats = torch.cat(
            (u.amax(-1, keepdim=True), lse - count.log(), count.log()), -1
        )
        if self.plan4_enabled:
            if plan_features is None:
                plan_features = stats.new_zeros((len(rows), 4))
            stats = torch.cat((stats, plan_features), -1)
        eta = self.emit_gate(
            torch.cat((history, rem, mean[beat], stats.to(h.dtype)), -1)
        ).float()
        if hasattr(self, "task_emit_bias"):
            eta = eta + self.task_emit_bias(task_ids)
        result = logits.float()
        if terminal is not None:
            result[rows, 0] = torch.where(terminal, F.logsigmoid(-eta[:, 0]), result[rows, 0])
        beat_start = FAMILY_IDS["BEAT"][0]
        moment_start = FAMILY_IDS["MOMENT"][0]
        result[rows, beat_start : beat_start + len(FAMILY_IDS["BEAT"])] = F.logsigmoid(
            -eta
        )
        result[rows, moment_start : moment_start + 32] = F.logsigmoid(eta) + (u - lse)
        return result


class SlideLookaheadHead(nn.Module):
    """Two future onset bins, with a detached prefix prior."""

    def __init__(self, dim, head_dim):
        super().__init__()
        self.mert_proj = nn.Linear(head_dim, 2, bias=False)
        self.prior_proj = nn.Linear(dim, 2)

    def forward(self, mert_ctx, normed_x):
        return self.mert_proj(mert_ctx) + self.prior_proj(normed_x.detach())


class Block(nn.Module):
    def __init__(self, cfg, with_lookahead=False):
        super().__init__()
        d = cfg.dim
        self.heads = cfg.heads
        self.cfg = cfg
        self.self_norm = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.cross_norm = nn.LayerNorm(d)
        self.audio_query = nn.Linear(d, cfg.heads * cfg.head_dim, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, eps=1e-6)
        self.audio_out = nn.Linear(cfg.heads * cfg.head_dim, d, bias=False)
        self.slide_lookahead = (
            SlideLookaheadHead(d, cfg.head_dim) if with_lookahead else None
        )
        if cfg.architecture in DRAFT_ARCHITECTURES:
            self.draft_attention = DecoderConditionAttention(cfg)
        self.ffnorm = nn.LayerNorm(d)
        if cfg.architecture == ARCHITECTURE_V20:
            self.cond_mod = nn.Linear(d, 2 * d)
            nn.init.zeros_(self.cond_mod.weight)
            nn.init.zeros_(self.cond_mod.bias)
            self.experts = TaskRuleMMoE(d, cfg.expert_dim, cfg.dropout)
        else:
            self.experts = RuleMMoE(d, cfg.expert_dim, cfg.experts, cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)

    def split(self, x):
        return x.reshape(1, len(x), self.heads, -1).transpose(1, 2)

    def join(self, x):
        return x.transpose(1, 2).reshape(x.shape[2], -1)

    def rotated(self, x, ticks):
        return partial_rotary(
            self.split(x),
            ticks / 96,
            scale=self.cfg.self_rope_scale,
            base=self.cfg.self_rope_base,
            rotary_dims=self.cfg.self_rotary_dims,
        )

    def projections(self, x, ticks):
        q, k, v = self.qkv(self.self_norm(x)).chunk(3, -1)
        return (
            self.rotated(q, ticks),
            self.rotated(k, ticks),
            self.split(v),
        )

    def read_query(self, x, ticks):
        d = x.shape[-1]
        q = F.linear(self.self_norm(x), self.qkv.weight[:d], self.qkv.bias[:d])
        return self.rotated(q, ticks)

    def finish(
        self, x, self_context, acoustic, roles, audio_tick, zero=None, lookahead=None, draft=None, condition=None, mai_rows=None
    ):
        residual = self.out(self.join(self_context))
        if zero is not None:
            residual = residual.masked_fill(zero[:, None], 0)
        x = x + self.drop(residual)
        normed_x = self.cross_norm(x)
        aq = self.q_norm(self.split(self.audio_query(normed_x)))
        queries = aq.split((self.cfg.conv_heads, self.cfg.mert_heads), dim=1)
        ctx = torch.cat(
            [
                attend(audio_rotary(self.cfg, stream, q, audio_tick / 96))
                for stream, q, attend in zip(STREAMS, queries, acoustic)
            ],
            dim=1,
        )
        if lookahead is not None:
            target, alpha, losses = lookahead
            rows = target["rows"]  # Compact main rows precede both auxiliary streams.
            head = self.cfg.conv_heads + self.cfg.mert_heads // 2
            with torch.autocast(x.device.type, enabled=False):
                logits = self.slide_lookahead(
                    ctx[0, head, rows].float(), normed_x[rows].float()
                )
                bce = F.binary_cross_entropy_with_logits(logits, target["targets"])
            losses["mert_slide_2b"] = losses["mert_slide_2b"] + alpha * bce
        x = x + self.drop(self.audio_out(self.join(ctx)))
        if draft is not None:
            tables, attend = draft
            n = tables[0].size(2)
            if n == len(x):
                x = x + self.draft_attention(x, tables, attend)
            else:
                residual = self.draft_attention(x[:n].clone(), tables, attend)
                x[:n].add_(residual)
        norm = self.ffnorm(x)
        if hasattr(self, "cond_mod"):
            # Project once per chart; decode supplies a row mapping for packed training.
            cond, mapping = condition if isinstance(condition, tuple) else (condition, None)
            modulation = self.cond_mod(cond)
            if isinstance(mai_rows, tuple):
                spans, splits = mai_rows
                pieces = []
                for group, role, ids, split in zip(norm.split(spans), roles.split(spans), mapping.split(spans), splits):
                    gamma, beta = modulation[ids].chunk(2, -1)
                    if len(group):
                        pieces.append(self.experts(group * (1 + gamma) + beta, role, split))
                return x + self.drop(torch.cat(pieces))
            if mapping is not None:
                modulation = modulation[mapping]
            gamma, beta = modulation.chunk(2, -1)
            norm = norm * (1 + gamma) + beta
            residual = self.experts(norm, roles, len(x) if mai_rows is None else mai_rows)
        else:
            residual = self.experts(norm, roles)
        return x + self.drop(residual)

    def reset_history(self):
        nn.init.ones_(self.self_norm.weight)
        nn.init.zeros_(self.self_norm.bias)
        for w in self.qkv.weight.chunk(3, 0):
            nn.init.orthogonal_(w)
        nn.init.zeros_(self.qkv.bias)
        nn.init.orthogonal_(self.out.weight, gain=0.02)
        nn.init.zeros_(self.out.bias)


class ActiveStateReadout(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("offsets", torch.arange(32) * 8, persistent=False)
        self.embed = nn.Embedding(32 * 8, 16)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.proj = nn.Sequential(
            nn.LayerNorm(32 * 16),
            nn.Linear(32 * 16, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, states):
        x = self.embed(states.long().clamp(0, 7) + self.offsets).flatten(-2)
        return self.proj(x).masked_fill(~states.bool().any(-1, keepdim=True), 0)


def attention_guide(q, k, lse, target, head):
    """The denominator comes from the kernel. Gather only supervised compact keys."""
    rows, keys = target["rows"], target["keys"]
    if not rows.numel():
        return lse.new_zeros(())
    query = q[0, head, rows].float()
    key = k[0, head // (q.shape[1] // k.shape[1]), keys].float()
    normalizer = lse[0, head, rows]
    scale = q.shape[-1] ** -0.5
    if "weights" in target:
        scores = (query * key).sum(-1) * scale
        return (
            target["weights"] * (normalizer - scores + target["log_weights"])
        ).sum() / target["support"]
    if "cluster" in target:
        # Ragged palm / slide clusters can contain arbitrarily many slots.
        cluster = target["cluster"]
        scores = (query[cluster] * key).sum(-1) * scale
        peaks = torch.full_like(normalizer, -torch.inf).scatter_reduce_(
            0, cluster, scores.detach(), reduce="amax"
        )
        sums = torch.zeros_like(peaks).scatter_add_(
            0, cluster, (scores - peaks[cluster]).exp()
        )
        union = peaks + sums.log()
        return (
            (normalizer - union - target["log_count"]) * target["scale"]
        ).sum() / target["support"]
    scores = (query[:, None] * key).sum(-1) * scale
    return (normalizer - (scores + target["log_kernel"]).logsumexp(-1)).mean()


def main_nll(z, ids, union):
    pos = z.gather(1, ids[:, None]).squeeze(1)
    if union is not None and union["rows"].numel():
        rows = union["rows"]
        positives = torch.zeros(
            (len(rows), z.shape[1]), dtype=torch.bool, device=z.device
        )
        positives.scatter_(1, ids[rows, None], True)
        positives[union["inverse"], union["cols"]] = True
        pos = pos.index_copy(
            0, rows, z[rows].masked_fill(~positives, -torch.inf).logsumexp(-1)
        )
    return z.logsumexp(-1) - pos


class ChartAudio(tuple):
    """Feature/KV pair with the source grid needed to sample the raw draft envelope."""

    def __new__(cls, features, keys, bank, grids):
        pair = super().__new__(cls, (features, keys))
        pair.bank, pair.grids = bank, grids
        return pair


class FlatModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        cfg = self.config = config or FlatConfig()
        d = cfg.dim
        multitask = cfg.architecture == ARCHITECTURE_V20
        if multitask:
            self.task_embed = nn.Embedding(2, d)
            nn.init.normal_(self.task_embed.weight, std=0.1)
        self.fusion = DualAudioFusion(cfg)
        self.beat_encoder = BeatGridEncoder(cfg)
        self.timing_head = Grid32TimingHead(
            d, cfg.timing_history_dropout, cfg.inline_plan, multitask
        )
        if cfg.architecture in DRAFT_ARCHITECTURES:
            self.drafter = ChartDrafter(cfg)
            self.condition_adapter = ConditionAdapter(cfg)
            self.reconstruction = FlatReconstructionHead(cfg.draft_dim, multitask)
        if cfg.plan4_enabled:
            self.planner = (BeatTEReadout(cfg.draft_dim, multitask) if cfg.architecture in DRAFT_ARCHITECTURES
                            else EmotionCurvePlanner(d))
        if cfg.inline_plan:
            self.plan4_embed = Plan4Embedder(d)
        self.input_heads = HeteroInputHeads(d, multitask)
        self.phase = nn.Sequential(nn.Linear(CLOCK_DIM, d), nn.GELU(), nn.Linear(d, d))
        self.level_mlp = nn.Sequential(nn.Linear(16, d), nn.GELU(), nn.Linear(d, d))
        self.null_level = nn.Parameter(torch.zeros(d))
        self.register_buffer(
            "level_freq", torch.arange(1, 9) * math.pi / 15, persistent=False
        )
        self.difficulty = nn.Embedding(13 if multitask else 8, d, padding_idx=0)
        self.charter = nn.Embedding(len(cfg.designer_vocab) + 1, d, padding_idx=0)
        self.position = nn.Embedding(34, d, padding_idx=33)
        self.active_readout = ActiveStateReadout(d)
        with torch.no_grad():
            for embedding in (self.position, self.charter, self.difficulty):
                nn.init.normal_(embedding.weight, std=0.02)
                embedding.weight[embedding.padding_idx].zero_()
        self.blocks = nn.ModuleList(
            Block(
                cfg,
                layer < len(cfg.attn_guide_layers) and cfg.attn_guide_layers[layer] > 0,
            )
            for layer in range(cfg.layers)
        )
        self.norm = nn.LayerNorm(d)
        self.output = nn.Linear(d, SIZE_V20 if multitask else SIZE)
        groups = torch.arange(self.output.out_features)
        groups[list(NEW_PADS)] = NEW_PADS[0]
        first_route = {}
        for route, row in zip(values("ROUTE"), catalog()):
            groups[route] = first_route.setdefault(row["motion"], route)
        self.register_buffer("coarse", groups, persistent=False)

    def sample_condition(self, values, ids):
        if not self.training:
            return values, ids
        level, difficulty = values.unbind(-1)
        draws = torch.rand((4, *ids.shape), device=ids.device)
        missing = draws[0] < self.config.condition_dropout
        level = level.masked_fill(missing | (draws[1] < self.config.level_dropout), 0)
        difficulty = difficulty.masked_fill(missing | (draws[2] < self.config.difficulty_dropout), 0)
        ids = ids.masked_fill(missing | (draws[3] < self.config.charter_dropout), 0)
        return torch.stack((level, difficulty), -1), ids

    def conditioning(self, values, ids, task_ids=None, *, apply_dropout=True):
        if apply_dropout:
            values, ids = self.sample_condition(values, ids)
        level, difficulty = values.unbind(-1)
        angle = level[..., None] * self.level_freq
        level_vec = self.level_mlp(torch.cat((angle.sin(), angle.cos()), -1))
        return (
            torch.where(level[..., None] != 0, level_vec, self.null_level)
            + (self.task_embed(torch.zeros_like(ids) if task_ids is None else task_ids) if hasattr(self, 'task_embed') else 0)
            + self.difficulty(difficulty.long())
            + self.charter(ids).masked_fill((ids == 0)[..., None], 0)
        )

    def protected_modules(self):
        if self.config.architecture in DRAFT_ARCHITECTURES:
            yield self.drafter
            yield self.reconstruction
            yield self.condition_adapter
            if self.config.plan4_enabled:
                yield self.planner
            for block in self.blocks:
                yield block.draft_attention
        yield self.fusion
        yield self.beat_encoder
        yield from (
            self.timing_head.sub64_proj,
            self.timing_head.slot_key,
            self.timing_head.slot_acoustic,
        )
        for block in self.blocks:
            if block.slide_lookahead is not None:
                yield block.slide_lookahead
            yield from (
                block.cross_norm,
                block.audio_query,
                block.q_norm,
                block.audio_out,
            )

    def audio_keys(self, bank, song=0, *, chart):
        grids = (
            chart["grids"]
            if "grids" in chart
            else [(song, chart["grid_seconds"], chart["in_audio"])]
        )
        encoded = [
            self.beat_encoder(bank, owner, seconds, valid)
            for owner, seconds, valid in grids
        ]
        if len(encoded) == 1:
            feat, pairs = encoded[0]
        else:
            feat = torch.cat([x for x, _ in encoded])
            pairs = [tuple(torch.cat(parts, 2) for parts in zip(*stream))
                     for stream in zip(*(kv for _, kv in encoded))]
        return ChartAudio(feat, pairs, bank, grids) if self.config.architecture in DRAFT_ARCHITECTURES else (feat, pairs)

    def draft_keys(self, z, valid, plan_inputs=None, lengths=None):
        plan = known = None
        if plan_inputs is not None:
            plan, known = plan_inputs
            lengths = lengths or [len(plan)]
            plan, known = (nn.utils.rnn.pad_sequence(t.split(lengths), batch_first=True)
                           for t in (plan, known))
        mem = self.condition_adapter(z, valid, plan, known)
        tables = condition_rope(mem.beats, self.config)
        layers = [block.draft_attention.attention.project_memory(mem.values, mem.beats, mem.valid, tables)
                  for block in self.blocks]
        return ConditionMemory(layers, mem.valid, z.valid.size(1))

    def predict_plan(self, bank, features, grids, grid_ids, values):
        if self.config.architecture in DRAFT_ARCHITECTURES:
            z, valid = self.drafter(features, grids, grid_ids, values, bank)
            return self.planner(z, valid), valid
        summaries, masks = [], []
        offset = 0
        for owner, seconds, valid in grids:
            summary, audible = self.planner.summarize(
                features[offset : offset + len(seconds)],
                bank.envelope[owner],
                seconds,
                valid,
                bank.mel_lengths[owner],
            )
            summaries.append(summary)
            masks.append(audible)
            offset += len(seconds)
        pad = nn.utils.rnn.pad_sequence
        summary = pad([summaries[i] for i in grid_ids], batch_first=True)
        valid = pad([masks[i] for i in grid_ids], batch_first=True)
        with torch.no_grad():
            level, difficulty = values.unbind(-1)
            angle = level[:, None] * self.level_freq
            level_vec = self.level_mlp(torch.cat((angle.sin(), angle.cos()), -1))
            condition = torch.where(
                level[:, None] != 0, level_vec, self.null_level
            ) + self.difficulty(difficulty.long())
            if self.training:
                condition = condition.masked_fill(
                    torch.rand_like(level.float())[:, None] < 0.10, 0
                )
        return self.planner(summary, valid, condition), valid

    def plan_inputs(self, plan, chart_ids=None, step=0):
        cfg = self.config
        known = torch.ones(len(plan), dtype=torch.bool, device=plan.device)
        if self.training and step >= cfg.plan_warmup_steps and chart_ids is not None:
            ids, n = chart_ids
            chart_known = torch.rand(n, device=plan.device) >= cfg.plan_chart_dropout
            known = chart_known[ids] & (
                torch.rand(len(plan), device=plan.device) >= cfg.plan_bar_dropout
            )
        if cfg.architecture in DRAFT_ARCHITECTURES:
            return plan, known
        return self.plan4_embed(plan, known), emit_features(plan, known).reshape(-1, 4)

    def base(self, heads, clock, cond):
        return self.input_heads(heads, len(clock)) + self.phase(clock) + cond

    def minimal_timing_state(self, active, positions):
        return self.active_readout(active) + self.position(positions)

    def decode(
        self,
        c,
        audio,
        cond,
        *,
        auxiliary=False,
        chart_condition=None,
        save_kv=None,
        draft_keys=None,
        beat_evidence=None,
        plan_inputs=None,
        guides=None,
        cross_scale=1.0,
    ):
        cfg = self.config
        coords, timing = c["coords"], c["timing"]
        ticks = coords[:, 0]
        chart_audio = audio
        grid_feat, audio = chart_audio
        roles = (~timing).long()
        x = self.base(c["heads"], c["clock"], cond)
        masks = c["layouts"]
        selected = masks["special"]
        token_rows, read_rows = c.get("token_rows"), c.get("read_rows")

        # Editor prefill captures K/V once; use native SDPA on the first request.
        supervised = self.training and save_kv is None and guides is not None
        guide_enabled = supervised and cfg.attn_guide_weight > 0
        lookahead_enabled = (
            supervised
            and cfg.slide_lookahead_weight > 0
            and cross_scale > 0
            and c["slide_lookahead"]["support"] > 0
        )
        self_guides = SELF_GUIDE_NAMES[: cfg.heads]
        backend = (
            "sdpa"
            if save_kv is not None or (guide_enabled and AuxRequest is None)
            else cfg.history_backend
        )

        def layout(name, qr, kr):
            return AttentionLayout(masks[name], backend, qr, kr)

        def audio_layout(prefix, qr):
            return tuple(
                layout(prefix + stream, qr, c.get("grid_rows")).bind_audio(*pair)
                for stream, pair in zip(STREAMS, audio)
            )

        if cfg.architecture in DRAFT_ARCHITECTURES:
            if draft_keys is None:
                bank, grids = chart_audio.bank, chart_audio.grids
                if "grids" in c:
                    ids, values = c["chart_grid_ids"], c["condition"][0]
                else:
                    ids = [0]
                    values = torch.as_tensor(c["condition"][0], device=x.device)[None]
                z, valid = self.drafter(grid_feat, grids, ids, values, bank,
                                        c.get("draft_valid_indices"), c.get("draft_all_valid", False),
                                        designers=torch.as_tensor(c["condition"][1], device=x.device).reshape(-1),
                                        task_ids=c.get("task_ids"), mai_anchors=c.get("mai_anchors"))
                if plan_inputs is None and cfg.plan4_enabled:
                    plan_inputs = self.plan_inputs(c["plan4"])
                draft_keys = self.draft_keys(z, valid, plan_inputs, c.get("plan_lengths"))
            buckets = c["cond_buckets"]
            window = draft_keys.tile_window(buckets)
            query_tables = condition_rope(coords[None, :, 1] / 96, cfg)

        main = layout("main", token_rows, token_rows)
        acoustic = audio_layout("audio_", token_rows)
        if cfg.stage == 0 or auxiliary:
            minimal = self.minimal_timing_state(c["active_state"], coords[selected, 3])
        if cfg.stage == 0:
            x = x.index_add(0, selected, minimal.to(x.dtype))
        if auxiliary:
            clean, read = x, x[selected] + minimal
            fields = masks["active_fields"]
            field_inputs = x.index_select(0, fields)
            field_ticks = ticks.index_select(0, fields)
            clean_mask = layout("clean", token_rows, token_rows)
            read_mask = layout("read", read_rows, token_rows)
            read_audio = audio_layout("read_audio_", read_rows)
            stream_roles = torch.cat((roles, roles, roles[selected]))
            stream_ticks = torch.cat((coords[:, 1], coords[:, 1], coords[selected, 1]))
            token_ticks = ticks.repeat(2)

        gate_features = None
        if cfg.inline_plan:
            if plan_inputs is None:
                plan_inputs = self.plan_inputs(c["plan4"])
            anchors, gates = plan_inputs
            rows = c["plan4_rows"]
            x = x.index_add(0, rows, anchors[c["anchor_bars"]].to(x.dtype))
            gate_features = gates[c["plan_emit_beats"]]

        def guided_audio(stream, attend, key, alpha):
            def apply(q):
                out, lse = attend(q, lse_heads=1)
                guides[stream] = guides[stream] + alpha * attention_guide(
                    q, key, lse, c["guide"][stream], 0
                )
                return out

            return apply

        def joint_audio(primary, clean_attend, read_attend):
            def apply(q):
                qm, qc, qr = q.split((len(ticks), len(ticks), len(selected)), dim=2)
                return torch.cat(
                    (primary(qm), clean_attend(qc), read_attend(qr)), dim=2
                )

            return apply

        layer_condition = cond
        if chart_condition is not None:
            layer_condition = ((chart_condition, c["chart_ids"]) if "chart_ids" in c else chart_condition)
        if auxiliary and cfg.architecture == ARCHITECTURE_V20:
            if isinstance(layer_condition, tuple):
                chart_cond, mapping = layer_condition
            else:
                chart_cond = layer_condition.reshape(-1, cfg.dim)
                mapping = (torch.zeros_like(ticks) if len(chart_cond) == 1
                           else torch.arange(len(ticks), device=ticks.device))
            aux_condition = chart_cond, torch.cat((mapping, mapping, mapping[selected]))
        for layer, block in enumerate(self.blocks):
            draft = (query_tables, draft_keys.bind(layer, buckets, window)) if draft_keys is not None else None
            alpha = (
                cfg.attn_guide_layers[layer]
                if supervised and layer < len(cfg.attn_guide_layers)
                else 0
            )
            primary = acoustic
            lookahead = (
                (c["slide_lookahead"], alpha, guides)
                if lookahead_enabled and alpha > 0
                else None
            )
            if guide_enabled and alpha and cross_scale:
                primary = tuple(
                    guided_audio(stream, attend, k, alpha)
                    if c["guide"][stream]["rows"].numel()
                    else attend
                    for stream, attend, (k, _) in zip(STREAMS, acoustic, audio)
                )

            def main_context(q, k, v):
                if not (guide_enabled and alpha and cfg.stage == 1):
                    return main(q, k, v)
                out, lse = main(q, k, v, lse_heads=len(self_guides))
                for h, name in enumerate(self_guides):
                    guides[name] = guides[name] + alpha * attention_guide(
                        q, k, lse, c["guide"][name], h
                    )
                return out

            if auxiliary:
                states = torch.cat((x, clean))
                if layer == 0 and not cfg.inline_plan:
                    qm, km, vm = block.projections(x, ticks)
                    qc, kc, vc = qm, km, vm
                    read_k, read_v = kc, vc
                else:
                    q, k, v = block.projections(states, token_ticks)
                    qm, qc = q.chunk(2, dim=2)
                    km, kc = k.chunk(2, dim=2)
                    vm, vc = v.chunk(2, dim=2)
                    read_k, read_v = kc, vc
                if layer and fields.numel():
                    # An active Hold/Slide may follow a Tap in the same moment. Its
                    # contextual clean KV would carry that Tap back into EOM/BEAT.
                    d = x.shape[-1]
                    key, value = F.linear(
                        block.self_norm(field_inputs),
                        block.qkv.weight[d:],
                        block.qkv.bias[d:],
                    ).chunk(2, -1)
                    read_k = kc.index_copy(2, fields, block.rotated(key, field_ticks))
                    read_v = vc.index_copy(2, fields, block.split(value))
                qr = block.read_query(read, ticks[selected])
                ctx = torch.cat(
                    (
                        main_context(qm, km, vm),
                        clean_mask(qc, kc, vc),
                        read_mask(qr, read_k, read_v),
                    ),
                    dim=2,
                )
                combined = block.finish(
                    torch.cat((states, read)),
                    ctx,
                    tuple(joint_audio(*a) for a in zip(primary, acoustic, read_audio)),
                    stream_roles,
                    stream_ticks,
                    lookahead=lookahead, draft=draft,
                    condition=aux_condition if cfg.architecture == ARCHITECTURE_V20 else None,
                    mai_rows=((len(ticks), len(ticks), len(selected)), (c["mai_rows"], c["mai_rows"], c["mai_special"])) if "mai_rows" in c else None,
                )
                x, clean, read = combined.split((len(ticks), len(ticks), len(selected)))
            else:
                q, k, v = block.projections(x, ticks)
                if save_kv is not None:
                    save_kv(k, v)
                x = block.finish(
                    x,
                    main_context(q, k, v),
                    primary,
                    roles,
                    coords[:, 1],
                    timing if cfg.stage == 0 else None,
                    lookahead=lookahead, draft=draft, condition=layer_condition, mai_rows=c.get("mai_rows"),
                )
        rows, beats, legal = c["emit_rows"], c["emit_beats"], c["moment_legal"]
        if save_kv is not None:
            x = x[-1:]
            rows = torch.zeros_like(rows)
            # Prefill consumes only the last decision, never project the entire song here.
            if beat_evidence is None and len(rows):
                grid_feat = grid_feat.reshape(-1, 24, cfg.dim)[beats].flatten(0, 1)
            beats = torch.zeros_like(beats)
        if beat_evidence is None and len(rows):
            beat_evidence = self.timing_head.evidence(grid_feat)

        def readout(states, gate=None):
            h = self.norm(states)
            return self.timing_head(self.output(h), h, rows, beats, legal, beat_evidence, gate,
                                    c.get("emit_tasks"), c.get("emit_terminal"))

        logits = readout(x, gate_features)
        aux = readout(clean.index_copy(0, selected, read)) if auxiliary else None
        return logits, aux

    def losses(self, logits, aux, chart, metrics=None, *, physical_out=None, physical_category_weights=(1.0, 1.0, 4.0, 1.0)):
        legal, ids = chart["legal"], chart["targets"]
        if legal.dtype != torch.bool:
            if "token_rows" in chart:
                legal = legal.index_select(0, chart["token_rows"])
            shifts = torch.arange(7, -1, -1, device=legal.device, dtype=torch.uint8)
            legal = ((legal[..., None] >> shifts) & 1).flatten(1)[:, :logits.size(-1)].bool()
        main_rows = legal.sum(-1) > 1
        if "real_mask" in chart:
            main_rows = main_rows & chart["real_mask"]
        elif "n_real" in chart:
            main_rows = main_rows & (
                torch.arange(len(main_rows), device=main_rows.device) < chart["n_real"]
            )
        z = logits.float().masked_fill(~legal, -torch.inf)
        if physical_out is not None and "physical" in chart:
            from .physical import CAT_NAMES, physical_loss

            p_total, p_cats, p_sup = physical_loss(
                z, chart["physical"], physical_category_weights
            )
            physical_out["total"] = p_total
            if metrics is not None:
                metrics.update(
                    physical=p_total.detach(),
                    phys_support=p_sup,
                    **{
                        f"phys_{name}": p_cats[idx].detach()
                        for idx, name in enumerate(CAT_NAMES)
                    },
                )
        union = chart.get("union")
        nll = main_nll(z, ids, union).masked_fill(~main_rows, 0)
        weights = chart.get("weights", main_rows).masked_fill(~main_rows, 0)
        main, weight_sum = (nll * weights).sum(), weights.sum()
        if "mai_rows" in chart:
            split = chart["mai_rows"]
            sums, factors = [], []
            for name, section, factor in (("maimai", slice(0, split), 1.0),
                                          ("taiko", slice(split, None), self.config.taiko_loss_weight)):
                w = weights[section]
                if len(w):
                    value = (nll[section] * w).sum() / w.sum().clamp_min(1e-6)
                    sums.append(factor * value)
                    factors.append(factor)
                    if metrics is not None:
                        metrics["main_" + name] = value.detach()
            main, weight_sum = sum(sums), main.new_tensor(sum(factors))
        if metrics is not None:
            metrics.update(
                main_support=main_rows.sum(),
                nll_per_moment=nll.detach().sum() / max(1, chart["moments"]),
            )
        if aux is None:
            return main, weight_sum, main.detach() * 0, main_rows.sum() * 0
        z = aux.float().masked_fill(~legal, -torch.inf)
        target_group = self.coarse[ids]
        accept = self.coarse[None, :] == target_group[:, None]
        if union is not None and union["rows"].numel():
            # Collect every alternative group before expanding groups to token IDs.
            # Several alternatives may belong to the same row or motion group.
            rows = union["rows"]
            groups = torch.zeros((len(rows), z.size(-1)), dtype=torch.bool, device=z.device)
            groups[union["inverse"], self.coarse[union["cols"]]] = True
            accept[rows] |= groups[:, self.coarse]
        active = main_rows & (legal & ~accept).any(-1)
        loss = torch.logsumexp(z, -1) - torch.logsumexp(
            z.masked_fill(~accept, -torch.inf), -1
        )
        return main, weight_sum, loss.masked_fill(~active, 0).sum(), active.sum()

    def loss(
        self,
        charts,
        bank,
        indices=None,
        *,
        cross_scale=1.0,
        no_history_weight=0.5,
        step=0,
        physical_weight=0.0,
        physical_category_weights=(1.0, 1.0, 4.0, 1.0),
    ):
        # The normal DataLoader passes a single worker-packed logical batch.
        if isinstance(charts, list):
            from .data import pack_charts
            from torch.utils._pytree import tree_map_only

            charts = pack_charts(
                tree_map_only(torch.Tensor, lambda t: t.cpu(), charts), indices.tolist()
            )
            charts = tree_map_only(
                torch.Tensor, lambda t: t.to(bank.mel_feat.device), charts
            )
        values, designers = self.sample_condition(*charts["condition"])
        task_ids = charts.get("task_ids")
        chart_condition = self.conditioning(values, designers, task_ids, apply_dropout=False)
        cond = chart_condition.index_select(0, charts["chart_ids"])
        audio = self.audio_keys(bank, chart=charts)
        if (
            self.config.stage == 1
            and not self.beat_encoder.proj["conv"].weight.requires_grad
        ):
            cross_scale = 0.0
        names = (
            (*STREAMS, *SELF_GUIDE_NAMES[: self.config.heads])
            if self.config.stage == 1
            else STREAMS
        )
        guides = {s: cond.new_zeros((), dtype=torch.float32) for s in names}
        guides["mert_slide_2b"] = cond.new_zeros((), dtype=torch.float32)
        plan_inputs = None
        draft_keys = None
        draft_terms = {}
        grids, ids = charts["grids"], charts["chart_grid_ids"]
        if self.config.architecture in DRAFT_ARCHITECTURES:
            z, valid = self.drafter(audio[0], grids, ids, values, bank,
                                    charts.get("draft_valid_indices"), charts["draft_all_valid"],
                                    designers, task_ids, charts.get("mai_anchors"))
            draft_terms = reconstruction_loss(
                self.reconstruction, z.values, charts["draft_targets"],
                taiko_weight=getattr(self.config, "taiko_loss_weight", 1.0),
            )
        if self.config.plan4_enabled:
            if self.config.architecture not in DRAFT_ARCHITECTURES:
                pred, valid = self.predict_plan(bank, audio[0], grids, ids, charts["condition"][0])
            else:
                pred = self.planner(z, valid, task_ids)
            if task_ids is None:
                plan_term, mae, corr = planner_loss(
                    pred, charts["gt_beat_te"], valid, log_weight=self.config.plan_log_weight)
            else:
                terms = []
                boundary = charts["mai_charts"]
                for name, section in (("maimai", slice(0, boundary)), ("taiko", slice(boundary, None))):
                    if len(pred[section]):
                        value, task_mae, task_corr = planner_loss(pred[section], charts["gt_beat_te"][section], valid[section],
                                                   log_weight=self.config.plan_log_weight)
                        draft_terms["plan_" + name] = value
                        draft_terms["plan_mae_" + name] = task_mae
                        draft_terms["plan_corr_" + name] = task_corr
                        terms.append((name == "taiko", torch.stack((value, task_mae, task_corr))))
                plan_term, mae, corr = task_mean(terms, self.config.taiko_loss_weight).unbind()
            plan_inputs = self.plan_inputs(charts["plan4"], (charts["plan_chart_ids"], charts["chart_count"]), step)
        if self.config.architecture in DRAFT_ARCHITECTURES:
            draft_keys = self.draft_keys(z, valid, plan_inputs, charts.get("plan_lengths"))
        with packed_projection_scope():
            z, aux = self.decode(
                charts,
                audio,
                cond,
                auxiliary=self.config.stage == 1 and self.config.no_history_enabled,
                guides=guides,
                draft_keys=draft_keys,
                plan_inputs=plan_inputs,
                cross_scale=cross_scale,
                chart_condition=chart_condition,
            )
        metrics = {}
        phys_out = {} if "physical" in charts else None
        main, count, extra, support = self.losses(
            z,
            aux,
            charts,
            metrics,
            physical_out=phys_out,
            physical_category_weights=physical_category_weights,
        )
        moments = charts["moments"]
        m, a = main / count.clamp_min(1e-6), extra / support.clamp_min(1)
        lookahead_loss = guides.pop("mert_slide_2b")
        guide = sum(
            value * (cross_scale if name in STREAMS else 1.0)
            for name, value in guides.items()
        ) / len(names)
        metrics.update(
            attn_guide=guide.detach(),
            mert_slide_2b=lookahead_loss.detach(),
            **{"attn_" + k: v.detach() for k, v in guides.items()},
            main=m.detach(),
        )
        loss = m + self.config.attn_guide_weight * guide
        if draft_terms:
            loss = loss + self.config.draft_loss_weight * draft_terms["draft_loss"]
            metrics.update({k: v.detach() for k, v in draft_terms.items()})
        if phys_out and "total" in phys_out and physical_weight > 0.0:
            loss = loss + physical_weight * phys_out["total"]
        if self.config.plan4_enabled:
            loss = loss + self.config.planner_loss_weight * plan_term
            metrics.update(plan_loss=plan_term.detach(), plan_mae=mae.detach(), plan_corr=corr.detach())
        if self.config.no_history_enabled:
            metrics.update(no_history=a.detach(), no_history_support=support.detach())
            loss = loss + no_history_weight * a
        return (
            loss + self.config.slide_lookahead_weight * cross_scale * lookahead_loss,
            metrics,
            moments,
        )
