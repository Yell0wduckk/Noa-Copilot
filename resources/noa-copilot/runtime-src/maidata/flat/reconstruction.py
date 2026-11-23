"""Four-beat rhythm queries and centered two-beat hold/slide targets for draft latents."""

from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from maidata.nn.position import sinusoidal
from .codec import HOLD, SLIDE, TICK_TO_GRID32


SIZES = dict(onset=1, moment=32)
ONSET, MOMENT = 0, 1


def build_draft_targets(moments, beat_valid):
    """Each beat has two rhythm owners and belongs to one centered span window."""
    def events():
        for tick, notes in moments:
            if not notes:
                continue
            intervals = []
            for _, _, kind, _, end, _, branches in notes:
                if kind == HOLD:
                    intervals.append((0, end))
                elif kind == SLIDE:
                    intervals.append((1, max((b[1] for b in branches), default=tick + 1)))
            yield tick, intervals
    return rhythm_targets(events(), beat_valid)


def rhythm_targets(events, beat_valid):
    n_anchors = (len(beat_valid) + 1) // 2 + 1
    spans = np.zeros((n_anchors, 2), dtype=bool)
    by_beat = defaultdict(list)
    previous = None
    for tick, intervals in events:
        by_beat[tick // 96].append((tick, previous))
        previous = tick
        for channel, finish in intervals:
            first = max(0, (tick + 96) // 192)
            last = (max(tick, finish - 1) + 96) // 192
            spans[first:last + 1, channel] = True

    queries, scalars = [], []
    tasks = {name: [] for name in SIZES}
    valid_beats = np.flatnonzero(beat_valid)
    span_valid = np.zeros(n_anchors, dtype=bool)
    span_valid[(valid_beats + 1) // 2] = True
    anchors = np.flatnonzero(span_valid)
    for beat in valid_beats:
        events = by_beat[beat]
        for owner in (beat // 2, beat // 2 + 1):
            offset = beat - 2 * owner + 2
            tasks["onset"].append((len(queries) // 4, bool(events)))
            queries.extend((owner, offset, ONSET, -1))
            scalars.extend((0.0, False))
            for ordinal, (tick, prev) in enumerate(events):
                tasks["moment"].append((len(queries) // 4, TICK_TO_GRID32[tick % 96]))
                queries.extend((owner, offset, MOMENT, ordinal))
                gap = (prev - beat * 96) / 96 if prev is not None else 0.0
                scalars.extend((gap, prev is not None))

    def matrix(rows, width, dtype=np.int64):
        return torch.from_numpy(np.asarray(rows, dtype=dtype).reshape(-1, width))

    return dict(
        query=matrix(queries, 4),
        scalars=matrix(scalars, 2, np.float32),
        span=matrix(np.column_stack((anchors, spans[anchors])), 3),
        tasks={k: matrix(v, 2) for k, v in tasks.items()},
    )


def pack_targets(charts, stride):
    queries, scalars, spans = [], [], []
    tasks = {name: [] for name in SIZES}
    start = 0
    for chart, target in enumerate(charts):
        q = target["query"].clone()
        q[:, 0] += chart * stride
        queries.append(q)
        scalars.append(target["scalars"])
        span = target["span"].clone()
        span[:, 0] += chart * stride
        spans.append(span)
        for name, entries in target["tasks"].items():
            t = entries.clone()
            t[:, 0] += start
            tasks[name].append(t)
        start += len(q)
    return dict(
        query=torch.cat(queries),
        scalars=torch.cat(scalars),
        span=torch.cat(spans),
        tasks={k: torch.cat(v) for k, v in tasks.items()},
    )


class FlatReconstructionHead(nn.Module):
    def __init__(self, width, multitask=False):
        super().__init__()
        self.offset = nn.Embedding(4, width)
        self.role = nn.Embedding(2, width)
        self.scalars = nn.Linear(2, width, bias=False)
        positions = sinusoidal(torch.arange(128), width, 10000.0, 1.0)
        self.register_buffer("ordinal_table", F.pad(positions, (0, 0, 1, 0)), persistent=False)
        self.ffn = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 640 if multitask else 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(640 if multitask else 512, width),
        )
        self.norm = nn.LayerNorm(width)
        self.heads = nn.ModuleDict(
            {name: nn.Linear(width, size) for name, size in SIZES.items()}
        )
        if multitask:
            self.span_head_maimai = nn.Linear(width, 2)
            self.span_head_taiko = nn.Linear(width, 2)
        else:
            self.span_head = nn.Linear(width, 2)

    def forward(self, z, target):
        flat_z = z.flatten(0, 1)
        out = {}
        q = target["query"]
        if len(q):
            owner, offset, role, ordinal = q.unbind(-1)
            scalars = target["scalars"]
            x = (
                flat_z[owner] + self.offset(offset) + self.role(role)
                + self.ordinal_table[(ordinal + 1).clamp(0, 128)].to(flat_z.dtype)
                + self.scalars(scalars.sign() * scalars.abs().log1p())
            )
            h = self.norm(x + self.ffn(x))
            for name, entries in target["tasks"].items():
                if len(entries):
                    out[name] = self.heads[name](h[entries[:, 0]]).float()
        span = target["span"]
        if len(span):
            span_head = self.span_head if hasattr(self, "span_head") else (
                self.span_head_taiko if target.get("task_id") else self.span_head_maimai)
            out["span"] = span_head(flat_z[span[:, 0]]).float()
        return out


def task_mean(parts, taiko_weight):
    """Average per-game objectives over the games present in this batch."""
    weights = [taiko_weight if task else 1.0 for task, _ in parts]
    return sum(w * value for w, (_, value) in zip(weights, parts)) / max(sum(weights), 1e-6)


def reconstruction_loss(head, z, target, taiko_weight=1.0):
    if isinstance(target, list):
        groups = []
        metrics = {}
        for part in target:
            terms = reconstruction_loss(head, z, part)
            name = "taiko" if part["task_id"] else "maimai"
            metrics.update({key + "_" + name: terms[key] for key in ("draft_loss", "draft_onset", "draft_moment")})
            a, b = ("draft_roll", "draft_balloon") if part["task_id"] else ("draft_hold", "draft_slide")
            metrics[a], metrics[b] = terms["draft_hold"], terms["draft_slide"]
            groups.append((part["task_id"], terms))
        for key in ("draft_loss", "draft_onset", "draft_moment", "draft_span"):
            metrics[key] = task_mean([(task, g[key]) for task, g in groups], taiko_weight)
        return metrics
    logits = head(z, target)
    zero = z.flatten()[:0].sum()
    onset = moment = hold = slide = span = zero
    if "onset" in logits:
        onset = F.binary_cross_entropy_with_logits(
            logits["onset"][:, 0], target["tasks"]["onset"][:, 1].float()
        )
    if "moment" in logits:
        moment = F.cross_entropy(logits["moment"], target["tasks"]["moment"][:, 1])
    if "span" in logits:
        bce = F.binary_cross_entropy_with_logits(
            logits["span"], target["span"][:, 1:].float(), reduction="none"
        )
        hold, slide = bce.mean(0).unbind()
        span = 0.5 * (hold + slide)
    return dict(
        draft_loss=onset + moment + span,
        draft_onset=onset, draft_moment=moment, draft_span=span,
        draft_hold=hold, draft_slide=slide,
    )
