"""Incremental flat decoding. Forced editor prefixes and sampled tokens use the same cursor."""

from bisect import bisect_left

import numpy as np
import torch
from torch.nn import functional as F

from maidata.decoder.projections import packed_projection_scope
from maidata.spine.features import clocks, condition
from maidata.spine.generate import generation_cli, pick
from . import ARCHITECTURE
from .codec import (
    FAMILY_IDS,
    TIMING_INPUTS,
    Grammar,
    chart_tokens,
    snap_moments_grid32,
    field_horizon,
    resolve_version_profile,
)
from .descriptors import HeadRows, describe
from .audio import STREAMS, generation_audio
from .data import (
    active_spans,
    record_commit,
    grid_interval,
    beat_grid,
    valid_grid_bounds,
    legal_bits,
    prefix_info,
    prepare_chart,
)
from .model import FlatConfig, FlatModel


class Cursor:
    def __init__(self, model, bank, grid, meta, end, *, field_limit=None, start_tick=0):
        self.model, self.grid, self.end = model, grid, end
        self.last_query = None
        self.device = bank.mel_feat.device
        self.meta = meta
        self.start_tick = start_tick
        grid._device_tensors(self.device)
        self.length = int(bank.mel_lengths[0])
        v, c = condition(meta, model.config.designer_vocab)
        self.cond = model.conditioning(
            self.tensor([v], torch.float32), self.tensor([c])
        )[0]
        self.field_limit = end if field_limit is None else field_limit
        self.audio_grid = beat_grid(grid, self.field_limit, self.length)
        seconds, valid = self.audio_grid
        self.valid_bounds = valid_grid_bounds(valid)
        self.grid_feat, self.audio = model.beat_encoder(
            bank, 0, seconds.to(self.device), valid.to(self.device)
        )
        self.evidence_beat = None
        self.capacity = 512
        self.note_keys = torch.empty(
            self.capacity, dtype=torch.bool, device=self.device
        )
        self.keep_start = 1
        self.sink_slot = 0
        self.history = []
        self.history_rows = []
        self.layers = [None] * len(model.blocks)
        self.last_clock = None

    def tensor(self, x, dtype=torch.long):
        return torch.as_tensor(x, dtype=dtype).to(self.device, non_blocking=True)

    def allocate(self, value):
        return value.new_empty((*value.shape[:2], self.capacity, value.shape[-1]))

    def reserve(self):
        if len(self.history) < self.capacity:
            return
        # Keep the sink at zero. Compact expired entries only when storage fills.
        lo = self.keep_start
        if len(self.history) - lo + 1 >= self.capacity // 2:
            self.capacity *= 2
        flags = self.note_keys.new_empty(self.capacity)
        flags[0] = True
        flags[1 : 1 + len(self.history) - lo].copy_(
            self.note_keys[lo : len(self.history)]
        )
        self.note_keys = flags
        for i, pair in enumerate(self.layers):
            new = tuple(self.allocate(x) for x in pair)
            for old, dest in zip(pair, new):
                dest[:, :, :1].copy_(old[:, :, :1])
                dest[:, :, 1 : 1 + len(self.history) - lo].copy_(
                    old[:, :, lo : len(self.history)]
                )
            self.layers[i] = new
        self.history = [self.history[0], *self.history[lo:]]
        self.history_rows = [self.history_rows[0], *self.history_rows[lo:]]
        self.keep_start = 1
        self.sink_slot = 0

    def prefill(self, g, seq):
        rows, states, commits = [], [], []
        heads, evict, pages = HeadRows(), [], []
        for idx in seq:
            g.advance(idx)
            rows.append(prefix_info(g))
            pages.append(g.page_first)
            record_commit(g, commits)
            heads.append(g)
            if len(rows) > 1:
                evict.append(g.prev_evicted)
            if idx in TIMING_INPUTS and self.model.config.stage == 0:
                states.append(g.active_state())
        evict.append(False)
        arrays = dict(
            pages=np.asarray(pages),
            field_limit=self.field_limit,
            last_legal=legal_bits(tuple(g.legal(start_tick=self.start_tick))),
            **heads.arrays(),
            evict=np.asarray(evict),
            active_spans=active_spans(commits, np.asarray(rows)[:, 0]),
            tokens=np.array([*seq, 0]),
            rows=np.array(rows),
            active_state=np.array(states, np.uint8).reshape(-1, 32),
            protected=np.empty((0, 3), np.int32),
        )
        cfg = self.model.config
        c = prepare_chart(
            arrays,
            self.meta,
            self.end,
            cfg,
            self.length,
            8,
            grid=self.grid,
            prefill=True,
            audio_grid=self.audio_grid,
        )
        from torch.utils._pytree import tree_map_only

        # The device already has this grid and its encoded features.
        del c["grid_seconds"], c["in_audio"]
        c = tree_map_only(
            torch.Tensor, lambda x: x.to(self.device, non_blocking=True), c
        )
        low = g.tick - 96 if cfg.stage == 0 else (g.tick // 96 - 8) * 96
        live = (
            {slot for _, _, slots, *_ in g.active for slot in slots}
            if cfg.stage == 1
            else set()
        )
        keep = [
            i
            for i, row in enumerate(rows)
            if i in live
            or i == len(seq) - 1
            or (not evict[i] and (not i or row[0] >= low))
        ]
        self.history_rows = keep
        self.history = [(rows[i][0], seq[i]) for i in keep]
        self.keep_start, self.sink_slot = 1, 0
        self.capacity = max(512, len(keep) * 2)
        self.note_keys = self.note_keys.new_empty(self.capacity)
        self.note_keys[: len(keep)].copy_(
            self.tensor(
                [not i or seq[i] not in TIMING_INPUTS for i in keep], torch.bool
            )
        )
        positions = self.tensor(keep)

        def capture(k, v):
            pair = tuple(self.allocate(t) for t in (k, v))
            for dest, t in zip(pair, (k, v)):
                dest[:, :, : len(keep)].copy_(t.index_select(2, positions))
            caches.append(pair)

        caches = []
        scores, _ = self.model.decode(
            c, (self.grid_feat, self.audio), self.cond, save_kv=capture
        )
        self.layers = caches
        return scores[-1]

    def step(self, g):
        cfg = self.model.config
        row = prefix_info(g)
        tick, query_tick, _, timing_pos = row
        stage0 = cfg.stage == 0
        timing = g.last in TIMING_INPUTS
        ticks, audio_ticks, position, roles = self.tensor(
            [tick, query_tick, timing_pos, int(not timing)]
        ).split(1)
        if self.last_clock != tick:
            self.clock = clocks(self.grid, ticks, self.end)[:, :26]
            self.last_clock = tick
        query = (query_tick, g.page_first)
        if self.last_query != query:
            self.audio_slices = []
            for stream, pair in zip(STREAMS, self.audio):
                lo, hi = grid_interval(cfg, stream, *query, self.valid_bounds)
                self.audio_slices.append(
                    tuple(torch.cat((t[:, :, :1], t[:, :, lo:hi]), 2) for t in pair)
                )
            self.last_query = query
        h, record = describe(g)
        slide = {}
        if h == 4:
            record, refs, dist, before, total = record
            slide["slide_refs"] = self.tensor(refs)
            slide["slide_dist"] = self.tensor(dist, torch.float32)
            slide["slide_arc"] = self.tensor(
                [[len(refs), before, total]], torch.float32
            )
            slide["slide_slots"] = torch.arange(len(refs), device=self.device)
        x = self.model.input_heads.project(h, self.tensor([record], torch.int32), slide)
        x = x + self.model.phase(self.clock) + self.cond
        if stage0 and timing:
            x = x + self.model.minimal_timing_state(
                self.tensor([g.active_state()]), position
            )
        low = tick - 96 if stage0 else (tick // 96 - 8) * 96
        if self.history and g.prev_evicted:
            pos = len(self.history) - 1
            self.history[pos] = (tick, g.last)
            self.history_rows[pos] = g.index
        else:
            self.reserve()
            pos = len(self.history)
            self.history.append((tick, g.last))
            self.history_rows.append(g.index)
        win_start = max(
            1, bisect_left(self.history, (low, -1), lo=min(self.keep_start, pos))
        )
        # Grammar rows survive cache compaction. Only translate exceptional old slots.
        old_slots = []
        if not stage0 and win_start > 1:
            cutoff = self.history_rows[win_start]
            old_slots = [
                bisect_left(self.history_rows, slot)
                for _, _, slots, *_ in g.active
                for slot in slots
                if slot < cutoff
            ]
        self.keep_start = min(old_slots, default=win_start)
        # The expired prefix may remain in storage until compaction. Move just the
        # sink into the slot preceding live keys to obtain a contiguous view.
        start = self.keep_start - 1
        allowed = None
        if stage0:
            self.note_keys[pos] = not timing or not pos
            if start:
                self.note_keys[start] = True
            if not timing:
                allowed = self.note_keys[None, start : pos + 1]

        if old_slots:
            visible = np.zeros(pos - start + 1, bool)
            visible[0] = True
            visible[win_start - start :] = True
            visible[np.asarray(old_slots) - start] = True
            allowed = self.tensor(visible, torch.bool)[None]

        def attend_slice(k, v):
            def apply(q):
                if not k.shape[2]:
                    return q * 0
                return F.scaled_dot_product_attention(
                    q, k, v, enable_gqa=q.shape[1] != k.shape[1]
                )

            return apply

        acoustic = tuple(attend_slice(*pair) for pair in self.audio_slices)
        zero = (roles == 0) if timing and stage0 else None
        for i, block in enumerate(self.model.blocks):
            q, k, v = block.projections(x, ticks)
            if self.layers[i] is None:
                self.layers[i] = self.allocate(k), self.allocate(v)
            pk, pv = self.layers[i]
            pk[:, :, pos : pos + 1].copy_(k)
            pv[:, :, pos : pos + 1].copy_(v)
            if start and self.sink_slot != start:
                pk[:, :, start : start + 1].copy_(pk[:, :, :1])
                pv[:, :, start : start + 1].copy_(pv[:, :, :1])
            k, v = pk[:, :, start : pos + 1], pv[:, :, start : pos + 1]
            ctx = (
                q * 0
                if timing and stage0
                else F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=allowed,
                )
            )
            x = block.finish(
                x,
                ctx,
                acoustic,
                roles,
                audio_ticks,
                zero,
            )
        self.sink_slot = start
        h = self.model.norm(x)
        logits = self.model.output(h)
        if g.phase == "time":
            # The last choice is the unique next BEAT.
            choices = g.time_choices(self.start_tick)[:-1]
            if choices:
                beat = min(max(0, g.beat), len(self.grid_feat) // 24 - 1)
                if self.evidence_beat != beat:
                    self.evidence = self.model.timing_head.evidence(
                        self.grid_feat[beat * 24 : (beat + 1) * 24]
                    )
                    self.evidence_beat = beat
                lo, hi = (
                    choices[0] - FAMILY_IDS["MOMENT"][0],
                    choices[-1] - FAMILY_IDS["MOMENT"][0],
                )
                slots = torch.arange(32, device=self.device)
                legal = ((slots >= lo) & (slots <= hi))[None]
                row = ticks.new_zeros(1)
                logits = self.model.timing_head(
                    logits, h, row, row, legal, self.evidence
                )
        return logits[0]


def pick_time(scores, temperature, choices):
    moments, beat = choices[:-1], choices[-1]
    if not moments:
        return beat
    branches = torch.stack((scores[beat], scores[moments].float().logsumexp(0)))
    if int(pick(branches, temperature, [0, 1])) == 0:
        return beat
    return int(pick(scores, temperature, moments))


@torch.inference_mode()
def generate(
    model,
    bank,
    grid,
    metadata,
    end_tick,
    *,
    temperature=0.0,
    slide_bias=0.0,
    max_words=None,
    max_pseudo_groups=8,
    max_slot_events=8,
    max_tail_fields=128,
    collision_retries=None,  # Obsolete keyword accepted for existing callers.
    prefix_actions=(),
    start_tick=0,
    stop_tick=None,
    on_step=None,
):
    prefix = []
    beat = -1
    for dt, notes in prefix_actions:
        if notes:
            prefix.append((beat * 96 + dt, notes))
        else:
            beat += 1
    limit = min(end_tick, stop_tick - 1) if stop_tick is not None else end_tick
    prefix = snap_moments_grid32(prefix)
    g = Grammar(
        max(end_tick, field_horizon(prefix, 0)),
        onset_limit=limit,
        max_groups=max_pseudo_groups,
        max_slot_events=max_slot_events,
        max_fields=max_tail_fields,
        profile=resolve_version_profile(metadata),
    )
    cur = Cursor(
        model,
        bank,
        grid,
        metadata,
        end_tick,
        field_limit=g.limit,
        start_tick=start_tick,
    )
    slide_heads = FAMILY_IDS["SLIDE_HEAD"]
    with packed_projection_scope():
        seq = list(chart_tokens(prefix, song=metadata.get("song_id") or 0, seed=None))
        if prefix:
            scores = cur.prefill(g, seq)
        else:
            for i in seq:
                g.advance(i)
                scores = cur.step(g)
        first = len(g.moments)
        while g.tick <= limit:
            if on_step:
                on_step(g.tick, g.moments[first:])
            if (
                g.phase == "time"
                and max_words is not None
                and len(g.moments) - first >= max_words
            ):
                break
            choices = g.legal(start_tick=start_tick)
            if not choices:
                break
            if slide_bias and slide_heads[0] in choices:
                # The grammar offers all three head variants together.
                scores = scores.float()
                scores[slide_heads[0] : slide_heads[-1] + 1] += slide_bias
            idx = (
                choices[0]
                if len(choices) == 1
                else pick_time(scores, temperature, choices)
                if g.phase == "time"
                else int(pick(scores, temperature, choices))
            )
            g.advance(idx)
            scores = cur.step(g)
        return g.moments[first:]


def load_checkpoint(path, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["architecture"] != ARCHITECTURE:
        raise ValueError(f"Expected checkpoint architecture {ARCHITECTURE}")
    cfg = state["config"]
    model = FlatModel(FlatConfig(**cfg["model"]))
    model.load_state_dict(state["model"])
    return (
        model.eval().to(device),
        cfg,
        {
            k: state.get(k)
            for k in ("architecture", "epoch", "stage", "step", "source", "binding")
        },
    )


def main(argv=None):
    return generation_cli(
        argv,
        loader=load_checkpoint,
        sampler=generate,
        name="maimai-flat320",
        audio_loader=generation_audio,
    )
