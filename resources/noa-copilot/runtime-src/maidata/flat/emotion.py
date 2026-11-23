"""TE v3 chart workload, fixed four-beat conditioning and the acoustic planner."""

import math
from functools import lru_cache
from itertools import groupby

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from maidata.multitask.attention import rotary
from maidata.spine.codec import HOLD, SLIDE
from .geometry import DISK_DISTANCE_SQ, ROUTE_STEP_LEN


@lru_cache(maxsize=64)
def _hann(bins):
    kernel = np.hanning(bins)
    return kernel / kernel.sum()


def stabilize_high_plateau_preserve_drops(hist, beat_sec):
    """20 Hz TE v3 filter used by the chart listening tool."""
    base = np.convolve(hist, _hann(71) * 20, "same")
    active = base[base > 0.30]
    if len(active) < 12:
        return base
    fast_bins = round(np.clip(3 * beat_sec, 0.80, 1.40) * 20) | 1
    fast = np.convolve(hist, _hann(fast_bins) * 20, "same")
    p80 = np.percentile(active, 80)
    base = base.astype(np.float32)
    radius = 64
    neighbors = np.lib.stride_tricks.sliding_window_view(
        np.pad(base, radius, mode="edge"), 2 * radius + 1
    )
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    weights = neighbors - base[:, None]
    weights *= 1 / max(0.80, 0.28 * p80)
    np.square(weights, out=weights)
    weights += (offsets / 36) ** 2
    weights *= -0.5
    np.exp(weights, out=weights)
    plateau = np.einsum("ij,ij->i", weights, neighbors) / weights.sum(axis=1)
    high = np.clip((base - 0.45 * p80) / max(0.30, 0.25 * p80), 0, 1)
    ratio = fast / np.maximum(plateau, 0.50)
    flat = base + high * np.clip((ratio - 0.38) / 0.27, 0, 1) * (plateau - base)
    quarter = _hann((max(3, round(beat_sec * 20) | 1)) + 2)[1:-1]
    plunge = np.convolve(np.clip((0.42 - ratio) / 0.25, 0, 0.60), quarter, "same")
    return np.convolve(flat + plunge * (np.minimum(flat, fast) - flat), quarter, "same")


def compute_beat_te_v3(moments, grid, grid_seconds, in_audio, emotion_slide_boost=0.20):
    seconds = np.asarray(grid_seconds)
    valid = np.asarray(in_audio)
    if not moments or not valid.any():
        return np.zeros(len(seconds) // 24, np.float32)
    # Use the actual score clock for note and span endpoints, including STOPs.
    if len(grid.bpms) == 1 and not grid.stop_times:
        origin = grid.second(0)
        quarter_sec = 60 / (grid.bpms[0] * grid.speed)

        def second(tick):
            return origin + tick * (quarter_sec / 96)

        beat_sec = min(0.65, max(0.18, quarter_sec))
    else:
        second = lru_cache(maxsize=None)(grid.second)
        # The reference uses the median local tempo at chart subgroups.
        ticks = np.array([t for t, notes in moments for _ in {n[1] for n in notes}])
        beats = ticks / 96
        segments = np.searchsorted(grid.beats, beats, side="right").clip(1) - 1
        base = np.asarray(grid.bpms)[segments]
        delta = np.maximum(0.0, beats - np.asarray(grid.beats)[segments])
        bpm = np.sqrt(
            np.maximum(1.0, base**2 + 120 * np.asarray(grid.slopes)[segments] * delta)
        )
        beat_sec = float(np.clip(60 / (np.median(bpm) * grid.speed), 0.18, 0.65))
    times = [second(t) for t, _ in moments]
    n = max(101, math.ceil(max(0.0, float(seconds[-1])) * 20) + 21)
    impulse = np.zeros(n)
    spans = np.zeros(n)
    slide_gate = np.zeros(n)
    branches = [
        (s, second(a), second(b), routes)
        for s, (_, notes) in zip(times, moments)
        for *_, paths in notes
        for a, b, _, routes in paths
    ]
    hits = np.array([s for s, *_ in branches])
    doubles = (
        hits.searchsorted(hits + 0.025, side="right") - hits.searchsorted(hits - 0.025)
        >= 2
    )

    def add_span(start, finish, work):
        # Exact overlap of each 50 ms cell with the physical holding/sliding interval.
        left, right = max(0.0, start * 20), min(n - 1.0, finish * 20)
        if finish <= start:
            if 0 <= start * 20 < n:
                spans[int(left)] += work
        elif right > left:
            i, j = int(left), int(right)
            unit = work / ((finish - start) * 20)
            if i == j:
                spans[i] += unit * (right - left)
            else:
                spans[i] += unit * (i + 1 - left)
                spans[i + 1 : j] += unit
                spans[j] += unit * (right - j)

    bridge = min(2.0, max(1.0, 4.5 * beat_sec))
    last_end = -math.inf
    for (hit, launch, finish, routes), double in zip(branches, doubles):
        count = sum(ROUTE_STEP_LEN[r] for r in routes) - len(routes) + 1
        u = min(1.0, max(0.0, (count - 3) / 9))
        body = 1 + 0.5 * u + 0.5 * u * u * (3 - 2 * u)
        add_span(launch, finish, body * (0.80 if double else 1.0))
        start = min(hit, last_end) if hit - last_end <= bridge else hit
        slide_gate[
            max(0, math.floor(20 * start)) : max(0, math.ceil(20 * finish) + 1)
        ] = 1
        last_end = max(last_end, finish)

    recent = []
    prev_time, prev_pad, prev_dir = -math.inf, None, 0
    bins = np.rint(np.asarray(times) * 20).astype(np.intp).clip(0, n - 1)
    for (tick, notes), t, idx in zip(moments, times, bins):
        if len(notes) == 1:
            groups = (notes,)
        else:
            groups = (
                members
                for _, members in groupby(
                    sorted(notes, key=lambda r: r[1]), key=lambda r: r[1]
                )
            )
        for members in groups:
            outer, touches, ex = set(), set(), False
            for pad, _, kind, flags, end, head, _ in members:
                ex |= bool(flags & 2)
                if pad < 8 and (kind != SLIDE or head == 0):
                    outer.add(pad)
                if kind == HOLD:
                    impulse[idx] += float(pad >= 8)
                    if end <= tick:
                        impulse[idx] += 0.2
                    else:
                        add_span(t, second(end), 0.6)
                elif pad >= 8 and kind != SLIDE:
                    touches.add(pad - 8)
            if outer:
                pad = next(iter(outer)) if len(outer) == 1 else None
                dt = t - prev_time
                direction = (
                    (pad - prev_pad) % 8
                    if pad is not None and prev_pad is not None and 0.004 < dt <= 0.090
                    else 0
                )
                spin = direction in (1, 7) and (
                    prev_dir == direction or ex or dt <= 0.055
                )
                impulse[idx] += (
                    (0.8 if spin else 1.0)
                    if pad is not None
                    else 1.6 + 0.6 * (len(outer) - 2)
                )
                prev_time, prev_pad, prev_dir = t, pad, direction
            if touches:
                recent = [(s, z) for s, z in recent if t - s <= 0.40]
                wipe = trill = 0
                for zone in touches:
                    near = any(
                        0.002 <= t - s <= 0.22
                        and 0 < DISK_DISTANCE_SQ[zone, z] <= 0.8**2
                        for s, z in recent
                    )
                    if any(
                        z == zone and 0.002 <= t - s <= (0.40 if near else 0.25)
                        for s, z in recent
                    ):
                        trill += 1
                    elif near:
                        wipe += 1
                k = len(touches)
                base = (0.75, 1.0, 1.25, 1.5, 1.5, 1.5, 1.75)[min(k, 7) - 1]
                impulse[idx] += base * (
                    0.4 / 0.75
                    if wipe and wipe >= trill
                    else 0.6 / 0.75
                    if trill
                    else 1.0
                )
                recent.extend((t, z) for z in touches)
    bins = max(3, round(beat_sec * 20) | 1)
    hist = np.convolve(impulse, np.full(bins, 1 / bins), "same") + spans
    hist *= 1 + emotion_slide_boost * np.convolve(slide_gate, _hann(31), "same")
    curve = stabilize_high_plateau_preserve_drops(hist, beat_sec)
    samples = np.interp(seconds * 20, np.arange(n), curve) * valid
    return samples.reshape(-1, 24).mean(1).astype(np.float32)


def quantize_plan4(beat_te, training=False, generator=None):
    """Shared worker / inference bridge. Return [bar, mean plus four deviations]."""
    beats = torch.as_tensor(beat_te).float()
    blocks = F.pad(beats, (0, -beats.numel() % 4)).reshape(-1, 4)
    mean = blocks.mean(-1, keepdim=True)
    delta = blocks - mean
    if training:
        draws = torch.rand((len(blocks), 6), device=beats.device, generator=generator)
        jitter = draws[:, :1] < 0.5
        mean = mean + jitter * (draws[:, 1:2] - 0.5) * 0.4
        noise = draws[:, 2:]
        delta = delta + jitter * (noise - noise.mean(-1, keepdim=True)) * 0.3
    step = torch.where(mean <= 4.0, 0.25, 0.5)
    mean = (mean.clamp(0, 16) / step).round() * step
    step = torch.where(mean <= 4.0, 0.25, 0.5)
    delta = ((delta.clamp(-2.5, 2.5) / step).round() * step).masked_fill(
        delta.abs() < 0.25, 0
    )
    return torch.cat((mean, delta), -1)


def emit_features(plan, known):
    mean = plan[:, :1]
    beat = (mean + plan[:, 1:]).clamp(0, 16)
    mask = known[:, None].expand_as(beat)
    return torch.stack(
        (beat.log1p() * mask, (beat == 0) * mask, (beat - mean) / 2.5 * mask, mask), -1
    )


def plan4_basis(plan, base_freq, delta_freq):
    base = plan[..., :1] * base_freq
    delta = (plan[..., 1:, None] * delta_freq).flatten(-2)
    return torch.cat((base.sin(), base.cos(), delta.sin(), delta.cos()), -1)


class Plan4Embedder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer(
            "base_freq", torch.arange(1, 5) * (math.pi / 16), persistent=False
        )
        self.register_buffer(
            "delta_freq", torch.arange(1, 4) * (math.pi / 5), persistent=False
        )
        self.proj = nn.Sequential(
            nn.Linear(32, 64), nn.GELU(), nn.Linear(64, dim, bias=False)
        )
        nn.init.normal_(self.proj[0].weight, std=math.sqrt(max(1, dim / 64) / 8))
        nn.init.orthogonal_(self.proj[-1].weight, gain=0.15)
        self.null_plan = nn.Parameter(torch.zeros(dim))

    def forward(self, plan, known):
        basis = plan4_basis(plan, self.base_freq, self.delta_freq)
        return torch.where(known[:, None], self.proj(basis), self.null_plan)


class PlannerRoFormerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )

    def forward(self, x, positions, valid):
        q, k, v = (
            t.unflatten(-1, (4, -1)).transpose(1, 2)
            for t in self.qkv(self.norm(x)).chunk(3, -1)
        )
        ctx = F.scaled_dot_product_attention(
            rotary(q, positions),
            rotary(k, positions),
            v,
            attn_mask=valid[:, None, None, :],
        )
        x = x + self.out(ctx.transpose(1, 2).flatten(2))
        return x + self.ffn(x)


class EmotionCurvePlanner(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.grid_proj = nn.Linear(2 * dim, dim)
        self.dynamics = nn.Sequential(nn.Linear(8, 64), nn.GELU(), nn.Linear(64, dim))
        self.clock = nn.Linear(5, dim)
        self.blocks = nn.ModuleList(PlannerRoFormerBlock(dim) for _ in range(3))
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def summarize(self, grid_feat, envelope, seconds, in_audio, length):
        grid = grid_feat.detach().reshape(-1, 24, grid_feat.size(-1))
        valid = in_audio.view(-1, 24)
        beat_valid = valid.any(1)
        pos = (seconds * 125 - 0.5).clamp_min(0).minimum((length - 1).clamp_min(0))
        lo = pos.long()
        frames = torch.lerp(
            envelope[lo],
            envelope[(lo + 1).minimum((length - 1).clamp_min(0))],
            (pos - lo)[:, None],
        ).reshape(-1, 24, 4)
        energy, flux = frames.split(2, -1)
        count = valid.sum(1)

        def avg(x):
            return (x * valid[:, :, None]).sum(1) / count.clamp_min(1)[:, None]

        # Linear P90 over valid grid cells, including partially audible beats.
        rank = (count - 1).clamp_min(0) * 0.9
        ordered = energy.masked_fill(~valid[:, :, None], torch.inf).sort(dim=1).values
        rows = torch.arange(len(grid), device=grid.device)
        peak = torch.lerp(
            ordered[rows, rank.long()],
            ordered[rows, rank.ceil().long()],
            rank.frac()[:, None],
        ).masked_fill(~beat_valid[:, None], 0)
        raw = torch.stack(
            (avg(energy), peak, avg(flux), (count / 24)[:, None].expand(-1, 2)), -1
        ).flatten(1)
        first = rows.masked_fill(~beat_valid, len(rows)).amin()
        last = rows.masked_fill(~beat_valid, -1).amax()
        angle = ((rows - first) / (last - first).clamp_min(1)).clamp(0, 1) * math.pi
        # Median inter-cell duration excludes a STOP discontinuity within a beat.
        duration = seconds.view(-1, 24).diff(dim=1).median(1).values * 24
        clock = torch.stack(
            (
                angle.sin(),
                angle.cos(),
                (2 * angle).sin(),
                (2 * angle).cos(),
                duration.clamp_min(1e-5).log(),
            ),
            -1,
        )
        summary = self.grid_proj(torch.cat((grid.mean(1), grid.amax(1)), -1))
        return summary + self.dynamics(raw) + self.clock(clock), beat_valid

    def forward(self, summary, valid, condition):
        x = summary + condition.detach()[:, None]
        positions = torch.arange(x.size(1), device=x.device)
        for block in self.blocks:
            x = block(x, positions, valid)
        return F.softplus(self.head(x).squeeze(-1).float()).masked_fill(~valid, 0)


def planner_loss(pred, target, valid, *, log_weight=0.0):
    """Per-chart reductions, including charts with fewer than four audible beats."""
    pred, target = pred.float(), target.float()

    def mean(x, mask):
        return (x * mask).sum(-1) / mask.sum(-1).clamp_min(1)

    error = pred - target
    absolute = mean(F.smooth_l1_loss(pred, target, reduction="none", beta=0.5), valid)
    slope = sum(
        mean((error[:, lag:] - error[:, :-lag]).abs(), valid[:, lag:] & valid[:, :-lag])
        for lag in (1, 4)
    )
    centered = [
        (x - mean(x, valid)[:, None]).masked_fill(~valid, 0) for x in (pred, target)
    ]
    stds = [
        torch.linalg.vector_norm(x, dim=1) / valid.sum(1).clamp_min(1).sqrt()
        for x in centered
    ]
    total = absolute + 0.5 * slope + 0.25 * (stds[0] - stds[1]).abs()
    if log_weight:
        log_error = F.smooth_l1_loss(
            pred.clamp_min(0).log1p(), target.clamp_min(0).log1p(),
            reduction="none", beta=0.1,
        )
        total = total + log_weight * mean(log_error, valid)
    loss = total.mean()
    corr = mean(centered[0] * centered[1], valid) / (stds[0] * stds[1]).clamp_min(1e-8)
    return loss, mean(error.abs(), valid).mean().detach(), corr.mean().detach()
