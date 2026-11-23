"""Sparse physical anchors shared by the known-clock and audio-only decoders."""
from bisect import bisect_left
import math

import numpy as np

from .grid import BeatGrid, multiplier_to_meter


def _snap_span(span, seconds_per_beat):
    for denominator in (1, 2, 4, 3, 6):
        snapped = round(span * denominator) / denominator
        error = abs(snapped - span)
        if error <= 0.03 and error * seconds_per_beat <= 0.015:
            return snapped
    return None


def leading_bar_origin(first_seconds, bpm, bar_phase, frame_rate):
    """A silent lead-in alone cannot distinguish an offset from a first bar.

    Use audio zero only when the phase head independently describes one
    complete irregular bar before the first audible downbeat.
    """
    span = _snap_span(first_seconds * bpm / 60, 60 / bpm)
    count = int(first_seconds * frame_rate)
    if span is None or not 0 < span <= 4.5 or span * 2 == round(span * 2) or count < 4:
        return False
    phase = bar_phase[:count].numpy()
    expected = 2 * np.pi * (np.arange(count) + 0.5) / (first_seconds * frame_rate)
    error = np.hypot(phase[:, 0] - np.sin(expected), phase[:, 1] - np.cos(expected))
    return float(np.mean(error)) < 0.06


def meter_gaps(grid, seconds, strength, meters, duration, *, allow_leading, preferred_span):
    """Find short nonstandard spans between independently measured bar runs."""
    if len(seconds) < 3:
        return []
    beats = [grid.second_to_beat(t) for t in seconds]
    end = grid.second_to_beat(duration)
    seconds = list(seconds)
    strength = list(strength)
    if allow_leading and 0.03 < beats[0] <= 4.53 and grid.beat_to_second(0) >= -1e-6:
        beats.insert(0, 0.0)
        seconds.insert(0, grid.beat_to_second(0))
        strength.insert(0, 1.0)
    spans = {4 * n / d for n, d in meters}

    def supported(index, direction):
        neighbor = index + direction
        if not 0 <= neighbor < len(beats):
            return None
        observed = direction * (beats[neighbor] - beats[index])
        period = abs((seconds[neighbor] - seconds[index]) / max(observed, 1e-6))
        step = _snap_span(observed, period)
        if step not in spans or strength[neighbor] < 0.65:
            return None
        target = beats[index] + direction * 2 * step
        where = bisect_left(beats, target)
        for j in range(max(0, where - 1), min(len(beats), where + 1)):
            if strength[j] >= 0.65 and abs(target - beats[j]) <= min(0.03, 0.015 / period):
                return step
        # At EOF the third downbeat can lie beyond the observed recording.
        if (direction == 1 and neighbor == len(beats) - 1 and beats[neighbor] < end < target
                and preferred_span((seconds[index] + seconds[neighbor]) / 2) == step):
            return step
        return None

    gaps = []
    origin = 0.0
    for i, (a, b) in enumerate(zip(beats, beats[1:])):
        if not 0 < b - a <= 4.53 or strength[i+1] < 0.65:
            continue
        period = (seconds[i+1] - seconds[i]) / (b - a)
        span = _snap_span(b - a, period)
        if span is None or span <= 0:
            continue
        preceding, following = supported(i, -1), supported(i+1, 1)
        if following is None or span in (preceding, following):
            continue
        if preceding is None and not (i == 0 and abs(a) < 0.03):
            continue
        # Native edges already account for a well-classified change, including
        # distinct signatures of the same length, such as 3/4 and 6/8.
        if span in spans and preferred_span((seconds[i] + seconds[i+1]) / 2) == span:
            continue
        aligned = _snap_span(a - origin, period)
        if aligned is not None:
            left = origin + aligned
            origin = left + span
            gaps.append((left, origin))
    return gaps


def decode_bar_sections(grid, duration, gaps, decode_section, *, return_bars=False):
    """Decode sparse half-beat lattices on their own phases and join exact edges.

    decode_section returns absolute beat/meter rows, observed bars (optional),
    and the selected preset. A gap endpoint is an actual barline even when
    the meter on both sides has the same spelling.
    """
    if not gaps:
        return decode_section(None, None)
    segments, bars, presets = [], [], []
    bridges = []
    for left, right in gaps:
        span = right - left
        num, den = multiplier_to_meter(span / 4)
        signature = (num, den) if abs(4 * num / den - span) < 1e-5 else None
        return_bars |= signature is None
        bridges.append((left, right, signature))
    start = None
    for left, right, signature in [*bridges, (None, None, None)]:
        if left is None or left > (grid.second_to_beat(0) if start is None else start) + 1e-5:
            rows, observed, preset = decode_section(start, left)
            if preset is not None:
                presets.append(preset)
            segments.extend(row for row in rows if left is None or row[0] < left - 1e-5)
            if return_bars and observed is None:
                local = BeatGrid(dict(meter_segments=rows))
                lo = rows[0][0] if start is None else start
                hi = grid.second_to_beat(duration) if left is None else left
                observed = local.pulse_and_bar_beats(lo, hi)[1]
            if return_bars:
                bars.extend(b for b in observed if left is None or b < left - 1e-5)
        if left is not None:
            if signature is None:
                signature = segments[-1][1:] if segments else (4, 4)
            segments.append((left, *signature))
            bars.extend((left, right))
            start = right
    # Collapse identical adjacent meters only if their barline phases agree.
    reanchors = {right for _, right, signature in bridges if signature is None}
    compact = []
    for row in segments:
        if compact and row[0] not in reanchors and row[1:] == compact[-1][1:]:
            cycles = (row[0] - compact[-1][0]) / (4 * row[1] / row[2])
            if abs(cycles - round(cycles)) < 1e-6:
                continue
        compact.append(row)
    preset = max(presets, default=None, key=("simple", "medium", "complex").index)
    return compact, sorted(set(bars)) if return_bars else None, preset


def _pulse_islands(times, frames, bpm_track, down, frame_rate):
    """Fit each regular run separately, keeping its own phase intercept."""
    intervals = np.diff(times)
    islands = []
    i = 0
    while i < len(intervals):
        # Intersect slope intervals from the first anchor. This tolerates
        # adjacent peaks rounding to opposite sides of the 125 Hz frame
        # while preventing a gradual tempo ramp from accumulating drift.
        low, high = 0., np.inf
        j = i
        while j < len(intervals):
            count = j + 1 - i
            elapsed = times[j+1] - times[i]
            low = max(low, (elapsed - 1.25/frame_rate) / count)
            high = min(high, (elapsed + 1.25/frame_rate) / count)
            if low > high:
                break
            if j > i and abs(intervals[j] - intervals[i]) > .012 * intervals[i]:
                # Subframe slope and the local BPM head both changed. End at
                # the attack on the change, even if a single beat still fits.
                head = bpm_track[(frames[j] + frames[j+1]) // 2]
                base_head = bpm_track[(frames[i] + frames[i+1]) // 2]
                if abs(head - base_head) > .012 * base_head:
                    break
            j += 1
        count = j - i
        if count >= 4:
            x = np.arange(count + 1) - count / 2
            slope = float(x @ (times[i:j+1] - times[i:j+1].mean()) / (x @ x))
            coarse = float(np.median(bpm_track[frames[i]:frames[j]+1]))
            step = min((.25, .5, 1.), key=lambda q: abs(math.log(60 * q / (slope * coarse))))
            qbeats = count * step
            # A recording can end before eight post-gap beats are audible.
            # Still require two independent bar anchors for that partial tail.
            tail = (j == len(intervals) and qbeats >= 4
                    and len(down)/frame_rate - times[i] < 8*slope/step
                    and np.count_nonzero(down[frames[i:j+1]] >= .65) >= 2)
            if qbeats >= 8 or tail:
                islands.append((i, j, slope / step, float((x @ x) * step**2)))
        i = j
    return islands


def _periodic_residual(times, weights, rate):
    period = 60 / rate
    cycles = times / period
    origin = np.angle(np.dot(weights, np.exp(2j * np.pi * cycles))) / (2 * np.pi)
    return ((cycles - origin + 0.5) % 1 - 0.5) * period


def _snap_rate(rate, times, weights, drift):
    nearest = round(rate)
    if not len(times) or nearest <= 0 or abs(rate - nearest) > max(0.35, 0.0025 * rate):
        return rate

    def errors(bpm):
        ms = 1000 * _periodic_residual(times, weights, bpm)
        q = max(1, len(ms) // 4)
        return (
            abs(np.median(ms[-q:]) - np.median(ms[:q])),
            np.average(abs(ms), weights=weights),
            np.median(abs(ms)),
        )

    raw_drift, raw_mae, _ = errors(rate)
    int_drift, int_mae, med = errors(nearest)
    if (
        int_drift <= max(20.0, drift, 2 * max(raw_drift, 8.0))
        and int_mae <= max(raw_mae + 3.0, 1.15 * raw_mae)
        and med <= 22.0
    ):
        return float(nearest)
    return rate


def _phase_hold(phase, first, last, frame_rate):
    unit = phase[first:last+1]
    if len(unit) < 3:
        return None
    still = ((np.linalg.norm(np.diff(unit, axis=0), axis=1) < .008)
             & (np.linalg.norm(unit[:-1], axis=1) >= .15))
    edges = np.flatnonzero(np.diff(np.r_[False, still, False]))
    if not len(edges):
        return None
    a, b = max(zip(edges[::2], edges[1::2]), key=lambda pair: pair[1] - pair[0])
    if b - a >= max(3, round(.06 * frame_rate)):
        return (first + a + .5) / frame_rate, (b - a) / frame_rate
    return None


def reconcile_tempo_islands(tempos, stops, frames, times, bpm_track, beat, down, phase, frame_rate, snap_drift_ms,
                            *, transition_mask=None, islands=None):
    """Resolve only short phase slips flanked by independently equal slopes."""
    boundary_clock = islands is not None
    if islands is None:
        islands = _pulse_islands(times, frames, bpm_track, down, frame_rate)
    tempos = list(tempos)
    stops = list(stops)
    changed = False
    for (i, a, pa, wa), (b, j, pb, wb) in zip(islands, islands[1:]):
        elapsed = times[b] - times[a]
        if not 0 < elapsed <= 4 or abs(pa - pb) > .012 * max(pa, pb):
            continue
        if transition_mask is not None and np.any(transition_mask[frames[i]:frames[j]+1]):
            continue
        if any(s["time"] <= times[b] and s["time"] + s["duration"] >= times[a] - .008 for s in stops):
            continue
        period = (pa * wa + pb * wb) / (wa + wb)
        bpm = 60 / period
        drift_ms = 1000 * abs(bpm - round(bpm)) * (times[j] - times[i]) / bpm
        if boundary_clock and snap_drift_ms is not None:
            # Each island has its own phase; remove the gap before checking the
            # combined integer slope. Keep true quarter marks in subdivided runs.
            snapped = []
            for first, last in ((i, a), (b, j)):
                local = times[first:last+1] - times[first]
                aligned = abs(local / period - np.round(local / period)) * period <= .025
                weights = beat[frames[first:last+1]][aligned]
                snapped.append(_snap_rate(bpm, local[aligned], weights, snap_drift_ms))
            if snapped[0] == snapped[1]:
                bpm = snapped[0]
            period = 60 / bpm
        elif snap_drift_ms is not None and drift_ms <= snap_drift_ms:
            bpm = float(round(bpm))
            period = 60 / bpm
        span = elapsed / period
        residual = min(abs(span - round(span*d)/d) for d in (4, 6)) * period
        if abs(span - round(span)) * period <= .015:
            continue
        # Subdivision attacks inside a short rational bar belong to the
        # same clock. Unaligned intervening attacks can be a tempo ramp.
        if b > a + 1 and (residual > .015 or any(
            _snap_span((t - times[a]) / period, period) is None for t in times[a+1:b]
        )):
            continue
        hold = _phase_hold(phase, frames[a], frames[b], frame_rate) if residual > .5/frame_rate else None
        if hold is not None:
            at, held = hold
            steps = max(0, round((elapsed - held) / period))
            pause = elapsed - steps * period
            if pause > 0 and abs(pause - held) <= .025:
                # A measured stationary phase resolves even a near-grid tie.
                at = times[a] if abs(at - times[a]) < .015 else at
                stops.append(dict(time=float(at), duration=float(pause)))
            elif residual > .015:
                continue
        elif residual > .015:
            # Sparse attacks alone are ambiguous. A silent trough supports a
            # pause; coherent advancing phase at a different pace supports tempo.
            interval = slice(frames[a], frames[b]+1)
            if np.min(beat[interval]) >= .2:
                continue
            vectors = phase[interval]
            if np.all(np.linalg.norm(vectors, axis=1) >= .15):
                angles = np.unwrap(np.arctan2(vectors[:, 0], vectors[:, 1])) / (2*np.pi)
                winding = angles[-1] - angles[0]
                if winding >= .75 and abs(winding-round(winding)) < .04 and np.all(np.diff(angles) > 0):
                    continue
            steps = math.floor(span)
            stops.append(dict(time=float(times[a]), duration=float(elapsed - steps * period)))
        # Retain real changes on either side. The two fits share a slope,
        # never an intercept across the measured gap.
        lo, hi = times[i], times[j]
        onset = tempos[0]["start_seconds"]
        if i == 0:
            lo = min(lo, onset)
        before = [s for s in tempos if s["start_seconds"] < lo - .015]
        after = [s for s in tempos if s["start_seconds"] >= hi - .015]
        if not before or abs(before[-1]["bpm"] - bpm) >= .001:
            anchor = times[i] if i == 0 else lo
            before.append(dict(start_seconds=float(anchor), bpm=bpm))
        tempos = before + after
        changed = True
    if changed:
        stops.sort(key=lambda s: s["time"])
        clock = BeatGrid(dict(offset_seconds=tempos[0]["start_seconds"], bpm_segments=tempos, stops=stops))
        tempos = [dict(s, beat=beat) for s, beat in zip(tempos, clock.beats)]
    return tempos, stops
