"""Whole-song augmentations on the calibrator's physical and feature clocks."""

import math

import numpy as np
import torch
from torch.nn import functional as F


def _clamped_speed_range(metadata, raw_samples, cfg):
    duration = raw_samples / 44100
    bpm = metadata.get("bpm", 120.)
    rows = metadata.get("bpm_segments") or ()
    if rows:
        starts, bpms = np.array([(r["start_seconds"], r["bpm"]) if isinstance(r, dict)
                                 else r[:2] for r in rows], dtype=float).T
        weights = np.diff(np.r_[starts.clip(0, duration), duration])
        order = np.argsort(bpms)
        if weights.sum() > 0:
            bpm = bpms[order[np.searchsorted(weights[order].cumsum(), weights.sum() / 2)]]
    low, high = cfg["speed_range"]
    min_bpm, max_bpm = cfg.get("speed_bpm_bounds", (55., 280.))
    low = max(low, min_bpm / bpm, duration / cfg.get("speed_max_seconds", 350.))
    high = min(high, max_bpm / bpm)
    return (low, high) if low < high else (1., 1.)


def _tempo_sections(grid, start, end):
    """BPM extrema on this source interval, including linear tempo ramps."""
    starts, bpms, slopes = map(np.asarray, (grid.starts, grid.bpms, grid.slopes))
    edges = np.r_[start, starts[(starts > start) & (starts < end)], end]
    idx = (np.searchsorted(starts, edges[:-1], side="right") - 1).clip(0)
    a = bpms[idx] + slopes[idx] * np.maximum(0., edges[:-1] - starts[idx])
    b = bpms[idx] + slopes[idx] * np.maximum(0., edges[1:] - starts[idx])
    return np.minimum(a, b).min(), np.maximum(a, b).max()


def tempo_meter_flags(metadata, duration):
    from .grid import BeatGrid

    grid = BeatGrid(metadata)
    start = max(0., metadata.get("active_start_seconds", metadata.get("offset_seconds", 0.)))
    end = min(duration, metadata.get("active_end_seconds", duration))
    if end <= start:
        return False, False
    if len(grid.bar_beats_explicit) >= 2:
        start = max(start, grid.beat_to_second(grid.bar_beats_explicit[0]))
        end = min(end, grid.beat_to_second(grid.bar_beats_explicit[-1]))
        if end <= start:
            return False, False
    lo, hi = _tempo_sections(grid, grid._running(start), grid._running(end))
    tempo = hi > lo * 1.01 or any(a < end and b > start for a, b in zip(grid.stop_times, grid.stop_ends))
    first, last = grid.second_to_beat(start), grid.second_to_beat(end)
    meter = any(a < last and b > first and (n, d) != (4, 4)
                for (a, n, d), b in zip(grid.meter_segments, [*grid.meter_starts[1:], math.inf]))
    bars = np.asarray(grid.bar_beats_explicit)
    if len(bars) >= 2:
        meter |= bool(np.any((bars[:-1] < last) & (bars[1:] > first) & (abs(np.diff(bars) - 4.) > 1e-5)))
    return bool(tempo), bool(meter)


def piecewise_eligible(metadata, duration):
    if metadata.get("stops"):
        return False
    from .grid import BeatGrid
    grid = BeatGrid(metadata)
    first = max(0., grid.second_to_beat(0.))
    last = grid.second_to_beat(duration)
    bars = grid.bar_beats_explicit
    if bars:
        return np.searchsorted(bars, last) - np.searchsorted(bars, first) >= 16
    _, bars = grid.pulse_and_bar_beats(first, last)
    return len(bars) >= 16


def _piecewise_schedule(grid, duration, cfg, rng):
    if grid.stop_times:
        return None
    marks = grid.bar_beats_explicit
    if not marks:
        _, marks = grid.pulse_and_bar_beats(max(0., grid.second_to_beat(0)), grid.second_to_beat(duration))
    bars = np.array([grid.beat_to_second(b) for b in marks])
    bars = bars[(bars >= 0.) & (bars < duration)]
    if len(bars) < 16:
        return None
    staircase = rng.random() < .5
    low, high = cfg.get("speed_bpm_bounds", (55., 280.))
    for _ in range(16):
        if staircase:
            count = int(rng.integers(4, 7))
            start = int(rng.integers(2, len(bars) - count))
            cuts = list(bars[start:start + count])
            steps = np.cumsum(rng.integers(1, 3, size=count)) * rng.choice([-1, 1])
            shifts = [0, *steps]
            if rng.random() < .5:
                cuts.append(bars[start + count])
                shifts.append(0)
        else:
            count = int(rng.integers(1, 4))
            candidates = list(range(2, len(bars) - 2))
            chosen = []
            while candidates and len(chosen) < count:
                k = int(rng.choice(candidates))
                chosen.append(k)
                candidates = [j for j in candidates if abs(k-j) >= 4]
            cuts = list(bars[sorted(chosen)])
            shifts = [0]
            for cut in cuts:
                bpm = _tempo_sections(grid, cut, cut + .000001)[0]
                prev = shifts[-1]
                legal = [m for m in range(-6, 7) if abs(m-prev) >= 2
                         and bpm * abs(2**(m/24) - 2**(prev/24)) >= 6.]
                if not legal:
                    break
                shifts.append(int(rng.choice(legal)))
            if len(shifts) != len(cuts) + 1:
                continue
        edges = np.r_[0., cuts, duration]
        rates = np.exp2(np.asarray(shifts) / 24)
        if np.sum(np.diff(edges) / rates) > cfg.get("speed_max_seconds", 350.):
            continue
        legal = True
        for i, (a, b, rate) in enumerate(zip(edges, edges[1:], rates)):
            slow, fast = _tempo_sections(grid, a, b)
            limit = max(high, 300.) if staircase and shifts[i] == max(shifts) and shifts[i] > 0 else high
            if slow * rate < low or fast * rate > limit:
                legal = False
                break
        if legal:
            return edges, rates
    return None


def _warp_tempo(wave, metadata, grid, edges, speeds):
    import soxr

    sr = 44100
    target = np.r_[0., np.cumsum(np.diff(edges) / speeds)]

    def warp(t):
        k = np.clip(np.searchsorted(edges[1:], t, side="right"), 0, len(speeds)-1)
        return target[k] + (t - edges[k]) / speeds[k]

    samples = np.rint(target * sr).astype(int)
    output = np.empty(samples[-1], dtype=np.float32)
    for i, (a, b, speed) in enumerate(zip(edges, edges[1:], speeds)):
        p, q = samples[i:i+2]
        pre = min(p, round(.016 * sr)) if i else 0
        left = max(0, round(a * sr - pre * speed))
        # Include filter support beyond the join, then crop to absolute output anchors.
        right = min(len(wave), math.ceil(b * sr + .016 * speed * sr))
        row = wave[left:right]
        if speed != 1.:
            row = soxr.resample(row, sr * speed, sr)
        need = q - p + pre
        if len(row) < need:
            row = np.pad(row, (0, need-len(row)))
        if pre:
            idx = np.arange(-pre, 0)
            alpha = np.sin(.5 * np.pi * np.clip((idx / sr + .014) / .008, 0, 1)) ** 2
            output[p-pre:p] += alpha * (row[:pre] - output[p-pre:p])
        output[p:q] = row[pre:need]

    # Split original tempo ramps at every real cut. Warping time by 1/s and
    # multiplying BPM by s preserves the score integral and every bar coordinate.
    starts = np.unique(np.r_[grid.starts, edges[1:-1]])
    segments = []
    for i, a in enumerate(starts):
        j = max(0, np.searchsorted(grid.starts, a, side="right")-1)
        speed = speeds[np.searchsorted(edges[1:-1], a, side="right")]
        bpm = grid.bpms[j] + grid.slopes[j] * max(0., a - grid.starts[j])
        seg = dict(start_seconds=float(warp(a)), bpm=float(bpm*speed))
        if grid.slopes[j] and i+1 < len(starts):
            seg["end_bpm"] = float((bpm + grid.slopes[j] * (starts[i+1]-a)) * speed)
        segments.append(seg)
    meta = dict(metadata, offset_seconds=float(warp(metadata.get("offset_seconds", 0.))),
                bpm=segments[0]["bpm"], bpm_segments=segments)
    table = [(float(a), float(b), float(speed), float(-12 * math.log2(speed)))
             for a, b, speed in zip(target, target[1:], speeds)]
    return output, meta, table


def apply_piecewise_tempo_augmentation(wave, metadata, cfg, rng):
    from .grid import BeatGrid

    grid = BeatGrid(metadata)
    schedule = _piecewise_schedule(grid, len(wave)/44100, cfg, rng)
    return (wave, metadata, []) if schedule is None else _warp_tempo(wave, metadata, grid, *schedule)


def mel_pitch_shift(speed, cfg, rng):
    speed_pitch = 12 * math.log2(speed)
    if rng.random() < cfg["pitch_probability"]:
        shift = rng.uniform(*cfg["pitch_semitone_range"]) - speed_pitch
    elif speed != 1. and rng.random() < .5:
        shift = -speed_pitch
    else:
        return 0.
    return float(np.clip(round(shift * 2) / 2, -6., 6.))


def mask_aug_sources(targets, cfg):
    """Plan permutations on the worker-provided CPU labels once per logical batch.

    A negative index leaves a frame alone. The same plan follows each song
    through OOM retries, including retries with a different microbatch order.
    """
    danger = (targets["change"] > .05) | (targets["stop"] > .5)
    # Exact +-63 frame dilation via prefix sum: O(B*T) instead of O(B*T*127) CPU max_pool1d.
    prefix = F.pad(danger.to(torch.int32), (64, 63)).cumsum(dim=-1)
    safe = targets["active"] & ~((prefix[:, 127:] - prefix[:, :-127]) > 0)
    sources = torch.full(safe.shape, -1, dtype=torch.long)
    min_span, max_span = (round(s * 125) for s in cfg["mask_aug_span_seconds"])
    min_chunks, max_chunks = cfg["mask_aug_chunks"]
    min_span = max(min_span, 4 * min_chunks)
    for row, allowed in enumerate(safe):
        if torch.rand(()).item() >= cfg["mask_aug_probability"]:
            continue
        count = int(allowed.sum())
        if count < max(63, min_span):
            continue
        remaining = int(count * torch.empty(()).uniform_(*cfg["mask_aug_ratio_range"]).item())
        # Track eligible continuous runs so short islands and padding do not
        # consume the 32 attempts or turn into unsafe partial spans.
        edges = torch.where(F.pad(allowed, (1, 0)) != F.pad(allowed, (0, 1)))[0].tolist()
        runs = list(zip(edges[::2], edges[1::2]))
        for _ in range(32):
            limit = min(max_span, remaining, max((b-a for a, b in runs), default=0))
            if limit < min_span:
                break
            span = int(torch.randint(min_span, limit + 1, ()))
            counts = torch.tensor([max(0, b-a-span+1) for a, b in runs]).cumsum(0)
            ticket = torch.randint(int(counts[-1]), ())
            j = int(torch.searchsorted(counts, ticket, right=True))
            a, b = runs.pop(j)
            lo = a + int(ticket) - (int(counts[j-1]) if j else 0)
            hi = lo + span
            runs.extend((u, v) for u, v in ((a, lo-63), (hi+63, b)) if v-u >= min_span)
            k = int(torch.randint(min_chunks, min(max_chunks, span // 4) + 1, ()))
            # Distinct cuts in the compressed interval, then restore the
            # reserved three frames per chunk (at least four in total).
            cuts = torch.randperm(span - 3*k - 1)[:k-1].sort().values + 1 + 3 * torch.arange(1, k)
            chunks = torch.tensor_split(torch.arange(lo, hi), cuts)
            perm = torch.randperm(k)
            if torch.equal(perm, torch.arange(k)):
                perm = perm.roll(1)
            sources[row, lo:hi] = torch.cat([chunks[j] for j in perm.tolist()])
            remaining -= span
    return sources
