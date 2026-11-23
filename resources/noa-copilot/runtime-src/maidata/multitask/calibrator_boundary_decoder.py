"""Physical tempo estimation and learned bar decoding for the nested-boundary model."""

from collections import Counter

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.special import expit, softmax, log_softmax
from scipy.signal import find_peaks
import torch

from .calibrator_islands import (
    _periodic_residual,
    _pulse_islands,
    _snap_rate,
    reconcile_tempo_islands,
)
from .calibrator_semicrf import (
    boundary_peaks,
    candidate_nodes,
    observed_phase_grids,
    score_lattice,
)
from .grid import BeatGrid, multiplier_to_meter
from .calibrator_audibility import inside_audible_bounds

HZ = 125


from .calibrator_partition import require_numba, tempo_dp_scan


def _running(seconds, stops):
    result = np.array(seconds, dtype=np.float64, copy=True)
    for stop in stops:
        result -= np.clip(seconds - stop["time"], 0, stop["duration"])
    return result


def _fit_pulse_clock(times, is_bar, prior):
    """IRLS on cumulative quarter counts, allowing a fractional last step into a bar."""
    if len(times) < 2:
        return prior, times, np.zeros(len(times))
    elapsed = np.diff(times)
    rough = elapsed * prior / 60
    regular = (rough > 0.72) & (rough < 1.28)
    period = np.median(elapsed[regular]) if regular.any() else 60 / prior
    # The head predicts quarters in every meter. Multiple equal interior
    # intervals give a direct physical period even if log_bpm is an octave off.
    interior = elapsed[~is_bar[1:] & (elapsed > 0.08)]
    if len(interior) >= 3:
        measured = np.median(interior)
        if np.mean(abs(interior - measured) <= max(0.012, 0.04 * measured)) >= 0.6:
            period = measured
    rough = elapsed / period
    steps = np.maximum(1, np.round(rough))
    options = np.stack([np.round(rough * d) / d for d in (4, 6)])
    rational = options[np.abs(options - rough).argmin(0), np.arange(len(rough))]
    steps = np.where(
        is_bar[1:] & (abs(rough - rational) * period <= 0.015), rational, steps
    )
    # Discard spurious peaks inside a quarter beat, retaining real short barlines.
    accepted = np.r_[True, (rough > 0.64) | (is_bar[1:] & (rational >= 1 / 6))]
    if not accepted.all():
        return _fit_pulse_clock(times[accepted], is_bar[accepted], prior)
    x = np.r_[0, steps.cumsum()]
    w = np.ones(len(times))
    for _ in range(4):
        xc = x - np.average(x, weights=w)
        yc = times - np.average(times, weights=w)
        denom = np.dot(w * xc, xc)
        if denom < 1e-9:
            return prior, times, x
        period = np.dot(w * xc, yc) / denom
        w = np.minimum(1.0, 0.008 / np.maximum(1e-7, abs(yc - period * xc)))
    return (float(60 / period) if period > 0 else prior), times, x


def _lock_transition_phases(
    tempos, stops, origin, end, ceilings, stable, quarter_anchors=()
):
    """Keep each stable clock fixed and adjust intervening tempo spans from beat zero outwards."""
    grid = BeatGrid(dict(offset_seconds=origin, bpm_segments=tempos, stops=stops))
    starts = np.array([row["start_seconds"] for row in tempos])
    knots = np.unique(np.r_[starts, origin, end])
    segments = np.maximum(0, np.searchsorted(starts, knots[:-1], side="right") - 1)
    rates = np.array(grid.bpms)[segments]
    # Splitting at beat zero also handles a first downbeat *inside* a ramp.
    limits = np.array([ceilings.get(t, 0.0) for t in starts])
    limits = limits[segments]
    duration = np.diff(_running(knots, stops))
    local_grids = []
    for direction in (1, -1):
        previous, correction = origin, 0.0
        use_local = False
        ordered = stable if direction == 1 else reversed(stable)
        for a, b, first, last in ordered:
            anchor = first if direction == 1 else last
            if direction * (anchor - previous) <= 0:
                continue
            lo, hi = sorted((previous, anchor))
            left, right = np.searchsorted(knots, [lo - 1e-8, hi + 1e-8])
            section = slice(left, max(left, right - 1))
            bridge = left + np.flatnonzero(
                (limits[section] > 0) & (duration[section] > 0)
            )
            previous = anchor
            distance = direction * grid.second_to_beat(anchor) + correction
            if len(bridge):
                span = np.dot(duration[bridge], rates[bridge]) / 60
                # A bridge must advance score time, even for a very short pickup.
                # A recovered four-quarter island whose measured pulses include
                # the bar anchor returns to the full quarter grid.
                division = 1 if anchor in quarter_anchors else 2
                first_tick = (
                    np.floor(division * (distance - span) + 1e-9) + 1
                ) / division
                target = max(first_tick, round(division * distance) / division)
                # Keep the fitted stair shape and respect every braking segment's ceiling.
                max_scale = np.min(limits[bridge] / rates[bridge])
                if np.isfinite(max_scale):
                    reachable = distance + span * (max_scale - 1)
                    target = min(
                        target, np.floor(division * reachable + 1e-9) / division
                    )
                delta = target - distance
                use_local = not 0.80 <= (span + delta) / span <= 1.25
                if not use_local:
                    rates[bridge] *= (span + delta) / span
                    correction += delta
            elif (
                abs(distance - round(2 * distance) / 2) * 60 / grid.bpms[segments[left]]
                > 0.020
            ):
                use_local = True
            if use_local:
                # A short or unobserved bridge can leave a stable island off-grid.
                coord = direction * distance
                local_grids.append((a, b, coord - round(2 * coord) / 2))

    rows = []
    for t, rate in zip(knots, rates):
        if not rows or abs(rows[-1]["bpm"] - rate) > 1e-9:
            rows.append(dict(start_seconds=float(t), bpm=float(rate)))
    return rows, local_grids


def _runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return zip(edges[::2], edges[1::2])


def _bar_clocks(bars, seconds, beat, eighth, bpm, snap_drift=None):
    """Follow repeated equal-length bars from an independently observed quarter/eighth clock."""
    clocks = []
    i = 0
    while i + 3 < len(bars):
        anchor = bars[i]
        best = None
        for following in bars[i + 1 : i + 4]:
            period = following - anchor
            if period <= 0:
                continue
            rate = 240 / period
            prior = bpm[int(np.clip(anchor * HZ, 0, len(bpm) - 1))]
            ratio = rate / prior
            if not (0.9 < ratio < 1.12 or 1.23 < ratio < 1.43):
                continue
            chain = [anchor, following]
            while True:
                expected = chain[-1] + np.median(np.diff(chain[-5:]))
                j = np.searchsorted(bars, expected)
                near = bars[max(0, j - 1) : j + 1]
                if not len(near) or np.min(abs(near - expected)) > 0.022:
                    break
                chain.append(near[np.argmin(abs(near - expected))])
            if len(chain) < 4:
                continue
            period = np.median(np.diff(chain))
            # The first bars must establish this interpretation independently of
            # the scalar prior. Later 3+3+2 accents may follow the triplet alias.
            starts = np.array(chain[: min(3, len(chain) - 1)])[:, None]

            def support(signal, offsets):
                return np.interp(
                    (starts + period * offsets).ravel(), seconds, signal
                ).mean()

            quarters = support(beat, np.array([0.25, 0.5, 0.75]))
            eighths = support(eighth, np.arange(1, 8, 2) / 8)
            triplets = support(eighth, np.array([1, 2, 4, 5]) / 6)
            if quarters < 0.40 or eighths < 0.38 or eighths < triplets + 0.06:
                continue
            x = np.arange(len(chain), dtype=float)
            slope = np.dot(x - x.mean(), chain - np.mean(chain)) / np.sum(
                (x - x.mean()) ** 2
            )
            if best is None or chain[-1] > best[1]:
                fitted = 240 / slope
                if snap_drift is not None:
                    fitted = _snap_rate(
                        fitted, np.asarray(chain), np.ones(len(chain)), snap_drift
                    )
                best = (anchor, chain[-1], fitted)
        if best is None:
            i += 1
        else:
            clocks.append(best)
            i = np.searchsorted(bars, best[1])
    return clocks


def _tempo_edges(
    seconds, pulses, bars, bpm, down, regime, change, threshold, bar_clocks, islands
):
    end = len(seconds) / HZ
    peaks = boundary_peaks(change, threshold)
    moving = np.isin(regime, (1, 2))
    candidates = [
        (seconds[k], 2 + 2 * expit(change[k]))
        for k in peaks
        if not moving[k] or expit(change[k]) >= 0.70
    ]
    # Compare one-second neighborhoods on the median-filtered log-tempo axis.
    smooth = median_filter(np.log(bpm), size=HZ, mode="nearest")
    half = HZ // 2
    diff = np.zeros(len(bpm))
    diff[half:-half] = abs(smooth[2 * half :] - smooth[: -2 * half])
    for lo, hi in _runs(diff > np.log(1.03)):
        local = diff[lo:hi]
        best = np.flatnonzero(local >= local.max() * 0.99)
        candidates.append((seconds[lo + best[len(best) // 2]], 1.0))
    for (_, a, pa, _), (b, _, pb, _) in zip(islands, islands[1:]):
        gap = pulses[b] - pulses[a]
        if (
            abs(pa - pb) > 0.012 * min(pa, pb)
            or abs(gap - round(gap / pa) * pa) > 0.020
        ):
            candidates.extend([(pulses[a], 1.5), (pulses[b], 1.5)])
    # Look for changes in the raw attack spacing as well as the octave-normalized
    # pulse islands (which deliberately have identical slopes across a doubling).
    for marks in (pulses, bars):
        intervals = np.diff(marks)
        for j in range(4, len(intervals) - 3):
            left, right = intervals[j - 4 : j], intervals[j : j + 4]
            lmed, rmed = np.median(left), np.median(right)
            if (
                abs(lmed - rmed) > max(0.015, 0.035 * lmed)
                and abs(left[-1] - lmed) < 0.016
                and abs(right[0] - rmed) < 0.016
                and np.count_nonzero(abs(left - lmed) < 0.016) >= 3
                and np.count_nonzero(abs(right - rmed) < 0.016) >= 3
            ):
                candidates.append((marks[j], 1.6))
    for a, b, _ in bar_clocks:
        candidates.extend([(a, 1.9), (b, 1.9)])
    # Only confirmed downbeats, confident changes and the two region endpoints
    # subdivide a moving interval. The slope/level proposals serve stable regions.
    candidates = [
        (t, weight)
        for t, weight in candidates
        if weight >= 2 or not moving[int(np.clip(t * HZ, 0, len(bpm) - 1))]
    ]
    for lo, hi in _runs(moving):
        candidates.extend([(lo / HZ, 3.0), (hi / HZ, 3.0)])
    for t in bars:
        k = int(np.clip(t * HZ, 0, len(bpm) - 1))
        if moving[k] and down[max(0, k - 1) : k + 2].max(initial=0) >= 0.45:
            candidates.append((t, 1.8))
    # Suppress competing proposals near the same attack, including tiny tails.
    selected = [0.0, end]
    for t, _ in sorted(candidates, key=lambda row: -row[1]):
        k = int(np.clip(t * HZ, 0, len(bpm) - 1))
        for marks in (bars, pulses):
            near = marks[abs(marks - t) <= 0.060]
            if len(near):
                t = float(near[np.argmin(abs(near - t))])
                break
        i = np.searchsorted(selected, t)
        if i == 0 or i == len(selected):
            continue
        left, right = selected[i - 1 : i + 1]
        if min(t - left, right - t) < 0.45:
            continue
        if moving[k] and min(np.diff(np.searchsorted(pulses, [left, t, right]))) < 2:
            continue
        selected.insert(i, t)
    return selected


def _coherent_rate(times, weights, seed):
    """Recover the common slope without counting a missed or doubled attack as a beat."""
    if len(times) < 4 or times[-1] - times[0] < 1.0:
        return seed
    x = times - times[0]
    # Sample the frequency likelihood at four samples per Fourier lobe.
    width = 0.035 * seed
    count = max(9, int(np.ceil(2 * width * x[-1] / 15)) + 1)
    rates = np.linspace(seed - width, seed + width, count)
    evidence = abs(np.exp((2j * np.pi / 60) * np.outer(rates, x)) @ weights)
    rate = rates[evidence.argmax()]
    for _ in range(3):
        residual = _periodic_residual(x, weights, rate)
        w = weights / np.maximum(1.0, (residual / 0.018) ** 2)
        centered = x - np.average(x, weights=w)
        rate *= 1 - np.dot(w * centered, residual) / max(
            1e-9, np.dot(w * centered, centered)
        )
    return rate


def _subdivision_rate(quarters, eighths, prior):
    """Estimate slope from local IOIs even when a ramp outruns the scalar BPM head."""
    q, e = np.diff(quarters), np.diff(eighths)
    q, e = q[q > 0.06], e[e > 0.035]
    if len(q) < 2 or len(e) < 3:
        return prior
    # A dropped event doubles an IOI, and a syncopation can halve it. Median
    # absolute error across these assignments tolerates either onset stream.
    rates = np.linspace(0.80 * prior, 1.25 * prior, 91)

    def matches(intervals, factor):
        step = (factor * 60 / rates)[:, None]
        count = np.maximum(0.5, np.round(2 * intervals / step) / 2)
        error = abs(intervals - count * step)
        return (
            np.exp(-0.5 * (error / 0.010) ** 2) / (1 + 0.18 * abs(np.log2(count)))
        ).mean(axis=1)

    evidence = matches(q, 1.0) + matches(e, 0.5) - 0.10 * abs(np.log2(rates / prior))
    return rates[evidence.argmax()]


def _choose_rate(clock, weights, is_bar, prior, bar_times, neighbor, eighth_times):
    fitted, observed, counts = _fit_pulse_clock(clock, is_bar, prior)
    if len(clock) < 3:
        return fitted if 0.80 * prior <= fitted <= 1.25 * prior else prior
    intervals = np.diff(clock)
    regular = intervals[intervals > 0.08]
    seed = 60 / np.median(regular) if len(regular) else prior
    ratios = seed / prior
    if min(abs(np.log2(ratios / np.array([0.5, 2.0])))) < 0.14:
        periods = np.diff(bar_times)
        measured = _coherent_rate(clock, weights, seed)
        harmonic = measured * (2 if seed < prior else 0.5)

        def support(rate):
            if not len(periods):
                return 0.0
            spans = periods * rate / 60
            typical = np.minimum.reduce(
                [abs(spans - 3), abs(spans - 4), abs(spans - 3.5)]
            )
            # Short apparent measures alternating in quarter phase support the
            # faster clock. A coherent downbeat train at the slower rate does not.
            coherence = abs(
                np.mean(np.exp(2j * np.pi * (bar_times - bar_times[0]) * rate / 60))
            )
            return np.mean(np.exp(-((typical / 0.18) ** 2))) + 0.75 * coherence

        evidence = support(measured) - support(harmonic)
        if neighbor is not None:
            evidence += 0.15 * (
                abs(np.log2(harmonic / neighbor)) - abs(np.log2(measured / neighbor))
            )
        seed = measured if len(periods) >= 3 and evidence > 0.20 else harmonic
    elif abs(np.log2(seed / prior)) > 0.20:
        # Direct support from several consecutive intervals outranks an erroneous
        # scalar trajectory; isolated drum fills do not.
        if (
            not 0.65 < seed / prior < 1.5
            or np.mean(abs(regular - 60 / seed) < 0.015) < 0.65
        ):
            seed = prior
    if 0.75 < seed / prior < 1.5:
        subdivision = _subdivision_rate(clock, eighth_times, prior)
        if 0.805 < subdivision / prior < 1.245:
            seed = subdivision
    if abs(fitted / seed - 1) < 0.035 and len(observed) >= 0.8 * len(clock):
        differences = observed - counts * 60 / fitted
        if np.quantile(abs(differences - np.median(differences)), 0.8) < 0.008:
            return fitted
    return _coherent_rate(clock, weights, seed)


def _merge_clocks(left, right, strong):
    a, wa, rate_a = left
    b, wb, rate_b = right
    if not len(a) or not len(b):
        return abs(rate_a / rate_b - 1) <= 0.03
    if abs(rate_a / rate_b - 1) > 0.035:
        return False
    times, weights = np.r_[a, b], np.r_[wa, wb]
    rate = _coherent_rate(times, weights, rate_a if len(a) >= len(b) else rate_b)
    residual = 1000 * _periodic_residual(times, weights, rate)
    ra, rb = residual[: len(a)], residual[len(a) :]
    separate = np.r_[
        _periodic_residual(a, wa, rate_a), _periodic_residual(b, wb, rate_b)
    ]
    return (
        abs(np.median(ra) - np.median(rb)) <= (18 if strong else 28)
        and max(np.median(abs(ra)), np.median(abs(rb))) <= 25
        and np.average(abs(residual) - 1000 * abs(separate), weights=weights)
        <= (7 if strong else 14)
    )


def _tempo_metadata(
    read, n, snap, drift, stop_threshold, min_stop_frames, change_threshold
):
    seconds = (np.arange(n) + 0.5) / HZ
    sub = read("subframe_offset_ms") * 0.001
    beat = read("beat_logit")
    bar = read("downbeat_logit")
    pulse_ids = boundary_peaks(beat)
    bar_ids = boundary_peaks(bar)
    pulses, bars = (seconds[idx] + sub[idx] for idx in (pulse_ids, bar_ids))
    measured_origin = float(bars[0] if len(bars) else pulses[0] if len(pulses) else 0.0)
    origin = measured_origin
    bpm = 120 * np.exp2(read("log_bpm").astype(np.float64))
    posterior = softmax(read("regime_logits"), axis=1)
    smoothed = uniform_filter1d(posterior, size=HZ, axis=0, mode="nearest")
    regime = np.zeros(n, dtype=int)
    for a, b in _runs(smoothed[:, 1:3].sum(1) >= 0.50):
        if (b - a) / HZ >= 0.75 and posterior[a:b, 1:3].sum(1).mean() >= 0.5:
            regime[a:b] = 1 + smoothed[a:b, 1:3].argmax(1)
    regime[posterior.argmax(1) == 3] = 3
    prob = expit(read("stop_logit"))
    mask = (prob >= stop_threshold) | ((regime == 3) & (prob >= 0.5))
    runs = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    stops = []
    for a, b in zip(runs[::2], runs[1::2]):
        if b - a < min_stop_frames:
            continue
        lo, hi = np.searchsorted(pulses, [a / HZ, b / HZ])
        # When both sides keep quarter phase, a short STOP classification alone
        # cannot introduce elapsed score time. Compare measured clocks locally.
        if 0 < lo and hi < len(pulses):
            gap = pulses[hi] - pulses[lo - 1]
            period = 60 / bpm[a]
            raw_error = abs(gap / period - round(gap / period)) * period
            paused = (gap - (b - a) / HZ) / period
            if (
                raw_error <= 0.015
                and raw_error + 0.005 < abs(paused - round(paused)) * period
            ):
                continue
        stops.append(dict(time=float(a / HZ), duration=float((b - a) / HZ)))
    change = read("change_logit")
    eighth = expit(read("eighth_logit"))
    eighth_ids = boundary_peaks(read("eighth_logit"))
    eighth_times = seconds[eighth_ids] + sub[eighth_ids]
    bar_clocks = _bar_clocks(
        bars, seconds, expit(beat), eighth, bpm, drift if snap else None
    )
    islands = _pulse_islands(pulses, pulse_ids, bpm, expit(bar), HZ)
    edges = _tempo_edges(
        seconds,
        pulses,
        bars,
        bpm,
        expit(bar),
        regime,
        change,
        change_threshold,
        bar_clocks,
        islands,
    )
    clock = _running(pulses, stops)
    weights = expit(beat[pulse_ids]).astype(float)
    valid = np.ones(len(clock), dtype=bool)
    for stop in stops:
        valid &= ~(
            (pulses > stop["time"] + 0.008)
            & (pulses < stop["time"] + stop["duration"] - 0.008)
        )
    is_bar = np.isin(pulse_ids, bar_ids)

    def measure(a, b, previous=None):
        lo, hi = np.searchsorted(seconds, [a - 1e-7, b - 1e-7])
        region = slice(min(n - 1, lo), max(lo + 1, hi))
        prior = float(np.median(bpm[region]))
        keep = valid & (pulses >= a - 0.008) & (pulses <= b + 0.008)
        marks = _running(bars[(bars >= a - 0.008) & (bars <= b + 0.008)], stops)
        eighth_clock = _running(
            eighth_times[(eighth_times >= a - 0.008) & (eighth_times <= b + 0.008)],
            stops,
        )
        rate = _choose_rate(
            clock[keep],
            weights[keep],
            is_bar[keep],
            prior,
            marks,
            previous,
            eighth_clock,
        )
        state = np.bincount(regime[region], minlength=4)[:3].argmax()
        for left, right, bar_rate in bar_clocks:
            if left - 0.060 <= (a + b) / 2 <= right + 0.060 and not state:
                # The head may follow every third eighth across equal bars.
                if (
                    0.88 < bar_rate / rate < 1.13
                    or 1.22 < bar_rate / rate < 1.47
                    or 1.9 < bar_rate / rate < 2.1
                ):
                    rate = bar_rate
                    break
        return keep, rate, state, region

    sections = []
    for a, b in zip(edges, edges[1:]):
        keep, rate, state, region = measure(a, b, sections[-1][3] if sections else None)
        if sections and state == sections[-1][4] == 0:
            left, _, old, old_rate, _, _ = sections[-1]
            k = round(a * HZ)
            strong = expit(change[max(0, k - 8) : k + 9]).max(initial=0) >= 0.70
            sparse_intro = left == 0.0 and a - left < 1.0 and old.sum() < 4
            if (
                sparse_intro
                or old.sum() <= 1
                or keep.sum() <= 1
                or _merge_clocks(
                    (clock[old], weights[old], old_rate),
                    (clock[keep], weights[keep], rate),
                    strong,
                )
            ):
                a = left
                sections.pop()
                keep, combined_rate, state, region = measure(a, b)
                # Combining the attacks must preserve the clock that passed
                # the two-sided residual test, including its octave.
                seed = old_rate if old.sum() >= 3 and not sparse_intro else rate
                rate = (
                    combined_rate
                    if abs(combined_rate / seed - 1) < 0.035
                    else _coherent_rate(clock[keep], weights[keep], seed)
                )
        sections.append((a, b, keep, rate, state, region))
    edges = [s[0] for s in sections] + [n / HZ]
    corrected = list(edges)
    stable, rates, moving = [], [], {}
    quarter_anchors = set()
    for i, (a, b, keep, fitted, state, region) in enumerate(sections):
        if state in (1, 2):
            prior = np.median(bpm[region])
            if not 0.65 * prior <= fitted <= 1.55 * prior:
                fitted = prior
            moving[i] = float(np.max(bpm[region])) + 1 if state == 2 else np.inf
            rates.append(min(fitted, moving[i]))
            continue
        marks = pulses[keep]
        observed, w = clock[keep], weights[keep]
        # Equal bar durations can hide a 3+3+2 rhythm. Check the neighboring
        # four-quarter clock against the bar intervals and the independent eighth head.
        bar_times = bars[(bars >= a - 0.008) & (bars <= b + 0.008)]
        for j in (i - 1, i + 1):
            if not 0 <= j < len(sections) or len(bar_times) < 3:
                continue
            neighbor = sections[j][3]
            if abs(neighbor / fitted - 4 / 3) > 0.055:
                continue
            period = 60 / neighbor
            gaps = np.diff(_running(bar_times, stops))
            regular = abs(gaps - 4 * period) < 0.04
            if regular.sum() < 2 or regular.mean() < 0.40:
                continue

            def eighth_support(rate):
                ticks = (
                    bar_times[:-1][regular, None] + np.arange(1, 6)[None, :] * 30 / rate
                )
                return np.mean(np.interp(ticks.ravel(), seconds, eighth))

            if eighth_support(neighbor) > eighth_support(fitted) + 0.05:
                fitted = neighbor
                break
        residual = _periodic_residual(observed, w, fitted) if len(w) else np.empty(0)
        aligned = abs(residual) <= 0.035
        if snap:
            fitted = _snap_rate(fitted, observed, w, drift)
        rates.append(fitted)
        if not aligned.any():
            continue
        marks = marks[aligned]
        residual = -_periodic_residual(observed[aligned], w[aligned], fitted)
        local = marks + residual
        if len(local) >= 4 and b - a >= 4 * 60 / fitted:
            stable.append((i, float(local[0]), float(local[-1])))
            for left, right, bar_rate in bar_clocks:
                if (
                    min(right, b) - max(left, a) < 4 * 60 / fitted
                    or abs(bar_rate / fitted - 1) > 0.012
                ):
                    continue
                for anchor in (local[0], local[-1]):
                    phase = (anchor - left) * fitted / 60
                    if abs(phase - round(phase)) * 60 / fitted < 0.020:
                        quarter_anchors.add(anchor)
        # Denoise shared cut times only when a boundary actually observed that pulse.
        for j in (i, i + 1):
            if 0 < j < len(edges) - 1:
                k = np.argmin(abs(marks - edges[j]))
                shift = min(
                    0.015, (edges[j] - edges[j - 1]) / 2, (edges[j + 1] - edges[j]) / 2
                )
                if (
                    abs(marks[k] - edges[j]) <= 0.008
                    and abs(local[k] - edges[j]) < shift
                ):
                    corrected[j] = float(local[k])
        if a <= measured_origin < b:
            k = np.argmin(abs(marks - measured_origin))
            if abs(marks[k] - measured_origin) < 1e-6 and abs(residual[k]) <= 0.015:
                origin = float(local[k])

    tempos = [dict(start_seconds=t, bpm=rate) for t, rate in zip(corrected, rates)]
    ceilings = {corrected[i]: ceiling for i, ceiling in moving.items()}
    stable = [
        (corrected[i], corrected[i + 1], first, last) for i, first, last in stable
    ]

    tempos, stops = reconcile_tempo_islands(
        tempos,
        stops,
        pulse_ids,
        pulses,
        bpm,
        expit(beat),
        expit(bar),
        read("pulse_phase"),
        HZ,
        drift if snap else None,
        transition_mask=np.isin(regime, (1, 2)),
        islands=islands,
    )
    tempos, local = _lock_transition_phases(
        tempos, stops, origin, n / HZ, ceilings, stable, quarter_anchors
    )
    # The first tempo also applies before its anchor (pickup/intro).
    if len(tempos) == 1 or origin < tempos[1]["start_seconds"]:
        tempos[0]["start_seconds"] = origin
    meta = dict(
        offset_seconds=origin, bpm=tempos[0]["bpm"], bpm_segments=tempos, stops=stops
    )
    grid = BeatGrid(meta)
    for row, coord in zip(tempos, grid.beats):
        row["beat"] = coord
    for row, coord in zip(stops, grid.stop_beats):
        row["beat"] = coord
    local_grids = [
        (grid.second_to_beat(a), grid.second_to_beat(b), p) for a, b, p in local
    ]
    return meta, (float(pulses[0]) if len(pulses) else origin), local_grids


@torch.no_grad()
def decode_boundary_grid(
    outputs,
    batch_index=0,
    metadata=None,
    *,
    snap=True,
    drift=2.0,
    stop_threshold=0.85,
    min_stop_frames=3,
    change_threshold=0.32,
):
    n = int(outputs["lengths"][batch_index])

    def read(key):
        return outputs[key][batch_index, :n].detach().float().cpu().numpy()

    local_grids = []
    if metadata is None:
        meta, first, local_grids = _tempo_metadata(
            read, n, snap, drift, stop_threshold, min_stop_frames, change_threshold
        )
    else:
        meta = dict(metadata)
        first = meta.get("offset_seconds", 0.0)
    clock = BeatGrid(meta)
    seconds = (np.arange(n) + 0.5) / HZ
    axis = np.array([clock.second_to_beat(t) for t in seconds])
    logits, sub = read("downbeat_logit"), read("subframe_offset_ms")
    peaks = boundary_peaks(logits)
    times = seconds[peaks] + sub[peaks] * 0.001
    start = 0.0
    if metadata is not None and (
        clock.beat_to_second(0) < -0.015 or clock.beat_to_second(0) > n / HZ
    ):
        first_seen = clock.second_to_beat(times[0]) if len(times) else axis[0]
        start = min(
            (round(first_seen * d) / d for d in (4, 6)),
            key=lambda b: abs(first_seen - b),
        )
    # Anchor the last fully observed interval; audio can end midway through its successor.
    last = (
        clock.second_to_beat(times[-1]) if len(times) else clock.second_to_beat(n / HZ)
    )
    local_grids += observed_phase_grids(
        axis, seconds, logits, sub, read("beat_logit"), start, last
    )
    phase = next((p for a, b, p in reversed(local_grids) if a <= last <= b + 1e-6), 0.0)
    end = phase + round(2 * (last - phase)) / 2
    bars, meters = [start], []
    if end - start >= 0.25:
        nodes = candidate_nodes(
            axis,
            seconds,
            logits,
            sub,
            start,
            end,
            local_grids=local_grids,
            inference=True,
        )
        graph, score, change = score_lattice(
            outputs,
            batch_index,
            axis,
            nodes,
            logits,
            inference=True,
            recoveries=[t for a, b, _ in local_grids for t in (a, b)],
        )
        bars = outputs["semicrf"].decode(graph, score, change).tolist()
        compound = expit(read("compound_logit"))
        for a, b in zip(bars, bars[1:]):
            span = b - a
            num, den = multiplier_to_meter(span / 4)
            if abs(span - 3) < 1e-5 or abs(span - 6) < 1e-5:
                region = compound[np.searchsorted(axis, a) : np.searchsorted(axis, b)]
                if len(region) and region.mean() >= 0.5:
                    num, den = round(2 * span), 8
            if not meters or (num, den) != meters[-1][1:]:
                meters.append((a, num, den))
    if not meters:
        meters = [(start, 4, 4)]
    meta.update(
        meter_segments=meters,
        bar_beats=bars,
        numerator=meters[0][1],
        denominator=meters[0][2],
    )
    if metadata is not None:
        return meta
    grid = BeatGrid(meta)
    first_beat = round(grid.second_to_beat(first) * 12) / 12
    _, pulse, down = grid.calibrator_boundary_beats(
        first_beat - 1e-6, grid.second_to_beat(n / HZ)
    )
    return dict(
        meta,
        grid=grid,
        first_beat=first_beat,
        beat_times=[grid.beat_to_second(b) for b in pulse],
        downbeat_times=[grid.beat_to_second(b) for b in down],
        decoder_preset="boundary-semicrf",
        meter_segment_dicts=[
            dict(beat=b, numerator=num, denominator=den) for b, num, den in meters
        ],
    )


def _tail_support(times, beats, changes, steps, accent, valid, end, reference_rate):
    tail = times >= end - 8
    explicit = tail & (changes >= 0.45) & (accent >= 0.35) & (np.abs(steps) >= 0.04)
    confident = np.where(tail & (valid >= 0.65))[0]
    before = np.flatnonzero(~tail)
    if len(confident) >= 4 and len(before):
        if len(before) >= 2:
            recent = before[-17:]
            positive = np.diff(times[recent]) > 1e-6
            if positive.any():
                reference_rate = np.median(
                    60 * np.diff(beats[recent])[positive] / np.diff(times[recent])[positive]
                )
        last = confident[-4:]
        dt = np.diff(times[last])
        if np.all(dt > 0) and reference_rate > 0:
            ratios = 60 * np.diff(beats[last]) / (dt * reference_rate)
            if np.all(ratios >= 1.05) or np.all(ratios <= 0.95):
                return True
    return bool(explicit.any())


def _tempo_dp_numpy(x, y, penalty):
    count = len(x)
    sums = np.vstack([np.r_[0.0, v.cumsum()] for v in (x, y, x * x, x * y, y * y)])
    best = np.full(count, np.inf)
    parent = np.zeros(count, dtype=int)
    best[0] = 0.0
    for b in range(1, count):
        a = np.arange(b)
        m = b - a + 1
        sx, sy, sxx, sxy, syy = sums[:, b + 1 : b + 2] - sums[:, :b]
        xx = (sxx - sx * sx / m).clip(1e-12)
        xy = sxy - sx * sy / m
        error = (syy - sy * sy / m - xy * xy / xx).clip(0)
        candidate = best[:b] + penalty[:b] + error
        # Discourage one-interval segments whose slopes have no independent support.
        if b > 1:
            candidate[-1] += 0.012
        parent[b] = int(np.argmin(candidate))
        best[b] = candidate[parent[b]]
    return parent


def _snap_linear_bpm(rate, pos, obs, weights, raw_residual, *, drift=2.0):
    """Snap pseudo-fractional fitted BPMs to integers first, then genuine half-integers."""
    span = float(pos[-1] - pos[0]) if len(pos) >= 2 else 0.0
    if span <= 1e-9 or rate <= 0.0 or not np.isfinite(rate):
        return float(rate)
    w_sum = max(1e-12, float(weights.sum()))
    p_bar = float(np.dot(weights, pos)) / w_sum
    dp = pos - p_bar
    denom = max(1e-12, float(np.dot(weights * dp, dp)))
    raw_wmae = float(np.average(raw_residual, weights=weights)) if len(raw_residual) else 0.0
    raw_drift_ms = float(np.quantile(raw_residual, 0.75) * 1000.0) if len(raw_residual) >= 4 else 0.0

    def accepts(candidate, *, max_diff, max_drift_ms, max_mae_add_ms):
        if candidate < 1.0 or abs(rate - candidate) > max_diff:
            return False
        c_period = 60.0 / candidate
        c_intercept = float(np.dot(weights, obs - pos * c_period)) / w_sum
        res_c = obs - (c_intercept + pos * c_period)
        abs_c = np.abs(res_c)
        wmae_c = float(np.average(abs_c, weights=weights))
        med_c = float(np.median(abs_c))
        trend_c = float(np.dot(weights * dp, res_c)) / denom
        drift_ms = max(abs(trend_c * span), abs(span * (c_period - 60.0 / rate))) * 1000.0
        return (
            med_c <= 0.022
            and wmae_c <= max(raw_wmae + max_mae_add_ms / 1000.0, 1.15 * raw_wmae)
            and drift_ms <= max_drift_ms
        )

    nearest_int = float(max(1, round(rate)))
    if span <= 12.0:
        int_diff = max(0.50, 0.004 * rate)
        int_drift = max(12.0, float(drift))
    else:
        int_diff = max(0.35, 0.0025 * rate)
        int_drift = min(22.0, max(16.0, float(drift), 2.0 * max(raw_drift_ms, 6.0)))
    if accepts(nearest_int, max_diff=int_diff, max_drift_ms=int_drift, max_mae_add_ms=3.0):
        return nearest_int

    if span >= 8.0:
        nearest_half = float(max(0.5, round(rate * 2.0) / 2.0))
        if abs(nearest_half - round(nearest_half)) > 0.25 and accepts(
            nearest_half, max_diff=0.15, max_drift_ms=max(6.0, float(drift)), max_mae_add_ms=1.5
        ):
            return nearest_half
    return float(rate)


def _fit_linear_bpm(pos, obs, fallback_bpm, *, snap=True, drift=2.0):
    """Fit weighted free-intercept slope on (beat, running-time) pairs."""
    pos = np.asarray(pos, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    n = len(pos)
    span = float(pos[-1] - pos[0]) if n >= 2 else 0.0
    if n < 2 or span <= 1e-9:
        return float(fallback_bpm), np.zeros(n, dtype=np.float64), np.ones(n, dtype=np.float64)
    weights = np.ones(n, dtype=np.float64)
    period = 60.0 / max(float(fallback_bpm), 1.0)
    if n == 2:
        period = float(obs[1] - obs[0]) / span
        residual = np.zeros(2, dtype=np.float64)
    else:
        residual = np.zeros(n, dtype=np.float64)
        for _ in range(5):
            w_sum = max(1e-12, float(weights.sum()))
            p_bar = float(np.dot(weights, pos)) / w_sum
            o_bar = float(np.dot(weights, obs)) / w_sum
            dp = pos - p_bar
            denom = float(np.dot(weights * dp, dp))
            if denom <= 1e-12:
                break
            cand_period = float(np.dot(weights * dp, obs - o_bar)) / denom
            if cand_period <= 0.01:
                break
            period = cand_period
            intercept = o_bar - period * p_bar
            residual = np.abs(obs - (intercept + pos * period))
            weights = np.minimum(1.0, 0.010 / np.maximum(1e-5, residual))
    rate = 60.0 / period if period > 0.01 else float(fallback_bpm)
    if snap and period > 0.01:
        rate = _snap_linear_bpm(rate, pos, obs, weights, residual, drift=drift)
        w_sum = max(1e-12, float(weights.sum()))
        p_final = 60.0 / rate
        intercept = float(np.dot(weights, obs - pos * p_final)) / w_sum
        residual = np.abs(obs - (intercept + pos * p_final))
    return float(rate), residual, weights


def _learned_tempo_segments(
    times,
    beats,
    prior,
    changes,
    steps,
    accent,
    audible_end,
    *,
    snap,
    drift,
    valid_prob=None,
    tail_supported=None,
    bar_prob=None,
):
    """Global least-squares segmentation, followed by robust continuous-clock fits."""
    count = len(times)
    if count < 2:
        return [(0, float(prior[0]) if count else 120.0)]
    if valid_prob is None:
        valid_prob = accent
    tail = times >= audible_end - 8
    if tail_supported is None:
        tail_supported = _tail_support(
            times, beats, changes, steps, accent, valid_prob, audible_end, prior[0]
        )
    # Keep original node indices for exported segment onsets after filtering.
    keep = np.arange(count)
    if not tail_supported:
        usable = ~(tail & (valid_prob < 0.65) & (accent < 0.35))
        usable[0] = True
        keep = keep[usable]
        times, beats, prior, changes, steps, accent, tail = (
            v[keep] for v in (times, beats, prior, changes, steps, accent, tail)
        )
        if bar_prob is not None:
            bar_prob = bar_prob[keep]
        count = len(keep)
        if count < 2:
            return [(0, float(prior[0]))]
    x = beats - beats[0]
    y = times - times[0]
    # A predicted step and the octave-resolved prior on either side provide
    # independent evidence for a cut in addition to timing residuals.
    neighbors = np.arange(count)
    prior_jump = np.log2(prior[np.minimum(neighbors + 2, count - 1)] / prior[np.maximum(neighbors - 2, 0)])
    consistency = np.exp(-np.abs(prior_jump - steps[:count]) / 0.12)
    support = changes * (0.75 + 0.25 * consistency)
    penalty = 0.0004 + 0.038 * (1 - support) ** 2
    if bar_prob is not None:
        # Tempo changes almost always occur on barlines or integer beats.
        non_bar = bar_prob < 0.40
        penalty = np.where(non_bar, penalty * 1.8, penalty)
        off_beat = np.abs(beats - np.rint(beats)) > 1e-4
        penalty = np.where(off_beat & (bar_prob < 0.55), penalty * 3.0, penalty)
    if not tail_supported:
        penalty[tail] = np.inf
    penalty[0] = 0
    parent = (tempo_dp_scan if require_numba("tempo_dp_scan") else _tempo_dp_numpy)(x, y, penalty)
    cuts = [count - 1]
    while cuts[-1]:
        cuts.append(parent[cuts[-1]])
    cuts.reverse()
    result = []
    for a, b in zip(cuts, cuts[1:]):
        pos, obs = x[a : b + 1] - x[a], times[a : b + 1] - times[a]
        rate, _, _ = _fit_linear_bpm(pos, obs, prior[a], snap=snap, drift=drift)
        if b - a == 1 and result and changes[a] > 0.65:
            predicted = result[-1][1] * 2 ** np.clip(steps[a], -2, 2)
            if abs(predicted / rate - 1) < 0.06:
                rate = float(predicted)
                if snap:
                    rate, _, _ = _fit_linear_bpm(pos, obs, rate, snap=True, drift=drift)
        result.append((int(keep[a]), float(rate)))
    return result


def _tempo_islands(rate, changes, separated, steps=None, running=None, *, quarter=False):
    n = len(rate)
    if n < 2:
        return [(0, n)]
    # Count the same musical spans on either candidate grid. Use a centered
    # odd median window so a sharp change stays on its observed node.
    nodes_per_beat = 1 if quarter else 2
    median_width = 5 if quarter else 11
    margin = 3 * nodes_per_beat
    bar_width = 4 * nodes_per_beat
    min_intervals = 2 if quarter else 3
    smooth = median_filter(np.log2(np.maximum(rate, 1.0)), size=min(median_width, n), mode="nearest")
    jumps = np.abs(np.diff(smooth)) >= 0.25
    gates = find_peaks(changes, height=0.55, prominence=0.12, distance=margin)[0]
    if steps is not None and running is not None:
        dt = np.diff(running)
        confirmed = []
        for g in gates:
            if g < margin or g > n - margin:
                continue
            lo, hi = max(0, g - bar_width), min(n, g + bar_width)
            rate_diff = abs(np.median(smooth[g:hi]) - np.median(smooth[lo:g]))
            left_dt, right_dt = dt[lo:g], dt[g : hi - 1]
            ioi_diff = (
                abs(np.log2(max(1e-4, np.median(right_dt)) / max(1e-4, np.median(left_dt))))
                if len(left_dt) >= min_intervals and len(right_dt) >= min_intervals
                else 0.0
            )
            if abs(steps[g]) >= 0.12 or rate_diff >= 0.10 or ioi_diff >= 0.12:
                confirmed.append(int(g))
        gates = np.asarray(confirmed, dtype=int)
    cuts = np.unique(np.r_[0, np.where(separated | jumps)[0] + 1, gates, n])
    # Remove a soft boundary of each short fragment, never a true long gap.
    hard = {0, n, *(np.flatnonzero(separated) + 1)}
    merged = [0]
    for right in cuts[1:]:
        if right - merged[-1] < bar_width:
            if right not in hard:
                continue
            if merged[-1] not in hard:
                merged.pop()
        merged.append(right)
    return list(zip(merged, merged[1:]))


def _physical_anchors(running, fit_rates, valid, accent, eighth, bar, *, pulse_seconds=30.):
    reliable = (valid >= .55) & ((eighth >= .38) | (accent >= .35) | (bar >= .45))
    if reliable.sum() < max(4, .3 * len(running)):
        reliable = valid >= .5
    # Octave voting needs the rejected, physically strong half-beats as well.
    fit_ids = np.flatnonzero(reliable | (eighth >= .55))
    if len(fit_ids) < 2:
        fit_ids = np.arange(len(running))
    ta, ra = running[fit_ids], fit_rates[fit_ids]
    gaps = np.diff(ta)
    close = gaps < .62 * pulse_seconds / ra[:-1]
    if pulse_seconds == 60.:
        # An observed odd-eighth bar can legitimately interrupt two quarter pulses.
        half = abs(gaps * ra[:-1] / 60 - .5) <= .045
        close &= ~(half & (np.maximum(bar[fit_ids[:-1]], bar[fit_ids[1:]]) >= .65))
    if len(gaps) >= 2:
        beats = gaps * ra[:-1]/60
        sixteenth = abs(beats - .25) < .045
        on_grid = abs(beats - np.rint(beats*4)/4) < .045
        transition = (bar[fit_ids[1:-1]] >= .65) & on_grid[:-1] & on_grid[1:]
        transition &= sixteenth[:-1] | sixteenth[1:]
        close[:-1] &= ~transition
        close[1:] &= ~transition
    if not close.any():
        return fit_ids
    confidence = valid[fit_ids] * (.5 + .5 * eighth[fit_ids])
    # Non-maximum suppression removes a weak splitter without merging both
    # surrounding true pulses into one transitive close-gap component.
    retain = np.ones(len(fit_ids), dtype=bool)
    for k in np.argsort(-confidence, kind="stable"):
        if retain[k]:
            if k and close[k-1]:
                retain[k-1] = False
            if k < len(close) and close[k]:
                retain[k+1] = False
    return fit_ids[retain]



def _merge_stable_segments(segments, t, beats, change, *, snap=True, drift=2.0):
    merged = list(segments)
    # First absorb isolated <2-beat half/double-time transition glitches.
    changed = True
    while changed and len(merged) > 1:
        changed = False
        for j in range(len(merged)):
            start, rate = merged[j]
            end = merged[j + 1][0] if j + 1 < len(merged) else len(t) - 1
            if beats[end] - beats[start] >= 2.0 - 1e-6:
                continue
            candidates = []
            if j > 0:
                a, lr = merged[j - 1]
                pos, obs = beats[a : end + 1] - beats[a], t[a : end + 1] - t[a]
                mr, res, _ = _fit_linear_bpm(pos, obs, lr, snap=snap, drift=drift)
                med = float(np.median(res)) if len(res) else 0.0
                lim = 0.035 if abs(np.log2(max(rate, 1.0) / max(lr, 1.0))) >= 0.70 else 0.022
                if med <= lim:
                    candidates.append((med, "left", mr))
            if j + 1 < len(merged):
                _, rr = merged[j + 1]
                end2 = merged[j + 2][0] if j + 2 < len(merged) else len(t) - 1
                pos, obs = beats[start : end2 + 1] - beats[start], t[start : end2 + 1] - t[start]
                mr, res, _ = _fit_linear_bpm(pos, obs, rr, snap=snap, drift=drift)
                med = float(np.median(res)) if len(res) else 0.0
                lim = 0.035 if abs(np.log2(max(rate, 1.0) / max(rr, 1.0))) >= 0.70 else 0.022
                if med <= lim:
                    candidates.append((med, "right", mr))
            if candidates:
                candidates.sort(key=lambda x: x[0])
                _, side, mr = candidates[0]
                if side == "left":
                    merged[j - 1] = (merged[j - 1][0], mr)
                    merged.pop(j)
                else:
                    merged[j] = (start, mr)
                    merged.pop(j + 1)
                changed = True
                break

    out = [merged[0]]
    for j, (start, rate) in enumerate(merged[1:], 2):
        a, last_rate = out[-1]
        b = merged[j][0] if j < len(merged) else len(t) - 1
        near_same = abs(rate / last_rate - 1.0) <= 0.015
        if near_same:
            pos, obs = beats[a : b + 1] - beats[a], t[a : b + 1] - t[a]
            mr, residual, _ = _fit_linear_bpm(pos, obs, last_rate, snap=snap, drift=drift)
            med = float(np.median(residual)) if len(residual) else 0.0
            p90 = float(np.quantile(residual, 0.90)) if len(residual) else 0.0
            flagged = change[max(0, start - 1) : start + 2].max() >= 0.50
            if med <= (0.010 if flagged else 0.015) and (not flagged or p90 <= 0.022):
                out[-1] = (a, float(mr))
                continue
        out.append((start, rate))
    return out


def _fit_island(running, rates, changes, steps, accent, valid, end, snap, drift, tail_supported,
                *, eighth=None, bar=None, pulse_seconds=30.):
    from .calibrator_corrector import observed_beat_steps

    quarter = pulse_seconds == 60.0

    def representatives(distinct, confidence):
        if distinct.all():
            return np.arange(len(confidence))
        starts = np.r_[0, np.flatnonzero(distinct)+1]
        lengths = np.diff(np.r_[starts, len(confidence)])
        best = np.flatnonzero(confidence == np.repeat(np.maximum.reduceat(confidence, starts), lengths))
        return best[np.r_[True, np.diff(np.r_[0, distinct.cumsum()][best]) > 0]]

    distance = np.diff(running) * (rates[:-1] + rates[1:]) / 120
    keep = representatives(distance >= 0.125, valid if eighth is None else np.maximum(valid,eighth))
    running, rates, changes, steps, accent, valid = (
        v[keep] for v in (running, rates, changes, steps, accent, valid)
    )
    # An isolated octave/tempo glitch cannot identify a tempo segment. Preserve
    # sustained steps and the rates at explicit change onsets.
    fit_rates = np.where(changes >= 0.5, rates, median_filter(rates, size=3, mode="nearest"))
    if eighth is None:
        # Original anchor selection for clean-v1 checkpoint comparisons.
        increments = observed_beat_steps(running, fit_rates, changes, quarter=quarter)
        fit_ids = np.arange(len(running))
        if len(running) > 2:
            # A semantic proposal may divide a clean eighth into two individually
            # off-grid intervals. Anchor the fit on the enclosing attacks in that
            # case and project the interior proposal with the fitted clock below.
            seconds = running[2:] - running[:-2]
            rate = fit_rates[:-2]
            period = 30 / rate
            extra = (
                (abs(seconds - period) <= np.minimum(0.015, 3 / rate))
                & (increments[:-1] + increments[1:] > 0.5)
                & (abs(fit_rates[2:] / rate - 1) < 0.04)
                & (changes[1:-1] < 0.5)
            )
            fit_ids = fit_ids[~np.r_[False, extra, False]]
    else:
        eighth, bar = eighth[keep], bar[keep]
        fit_ids = _physical_anchors(running, fit_rates, valid, accent, eighth, bar, pulse_seconds=pulse_seconds)
    t = running[fit_ids]
    seed = fit_rates[fit_ids].copy()
    change = changes[fit_ids]
    bar_fit = bar[fit_ids] if bar is not None else None
    if eighth is not None and len(t) >= 3:
        windows = np.lib.stride_tricks.sliding_window_view(np.pad(np.diff(t), (2, 3), mode="edge"), 5)
        physical_rate = pulse_seconds / np.median(windows, axis=1)
        ratio = physical_rate / seed
        stable = (change < .5) & (ratio >= .68) & (ratio <= 1.48) & (windows.std(1) < .12 * windows.mean(1))
        seed[stable] = physical_rate[stable]

    def fit(prior):
        beats = np.r_[0., observed_beat_steps(t, prior, change, quarter=quarter, bar_prob=bar_fit).cumsum()]
        segments = _learned_tempo_segments(t, beats, prior, change, steps[fit_ids], accent[fit_ids], end,
                                           snap=snap, drift=drift, valid_prob=valid[fit_ids],
                                           tail_supported=tail_supported, bar_prob=bar_fit)
        return beats, segments
    beats, segments = fit(seed)
    if eighth is not None and len(t) > 1:
        starts, rates1 = map(np.asarray, zip(*segments))
        fitted1 = rates1[np.searchsorted(starts, np.arange(len(t)), side="right")-1]
        if np.any(abs(fitted1 / seed - 1) > .06):
            beats, segments = fit(fitted1)
        segments = _merge_stable_segments(segments, t, beats, change, snap=snap, drift=drift)
    indices, fitted = map(np.asarray, zip(*segments))
    indices = fit_ids[indices]
    instant = fitted[(np.searchsorted(indices, np.arange(len(rates)), side="right") - 1).clip(0)]
    physical = np.r_[0.0, (np.diff(running) * instant[:-1] / 60).cumsum()]
    clean_steps = observed_beat_steps(running[fit_ids], instant[fit_ids], changes[fit_ids], quarter=quarter, bar_prob=bar_fit)
    if eighth is not None:
        physical -= physical[fit_ids[0]]
    if quarter:
        # Quarter-pulse grids keep integer/half-beat coordinates (never stray
        # 0.25 sixteenths) while preserving odd-eighth bars and 0.5x vs 1x
        # octave competition.
        coords = np.rint(physical * 2) / 2
    else:
        # A genuine odd sixteenth changes the phase of subsequent eighths.
        quarter_phase = np.r_[0.0, np.cumsum(np.mod(np.rint(clean_steps * 4), 2))] % 2 / 4
        quarter_phase = quarter_phase[(np.searchsorted(fit_ids, np.arange(len(rates)), side="right") - 1).clip(0)]
        coords = quarter_phase + np.rint((physical - quarter_phase) * 2) / 2
    coords = np.maximum.accumulate(coords)
    confidence = valid if eighth is None else np.maximum(valid, eighth)
    unique = representatives(np.diff(coords) > 1e-5, confidence)
    if eighth is not None:
        coords -= coords[unique[0]]
        indices[0] = 0
    return keep[unique], coords[unique], [(int(k), float(r)) for k, r in zip(keep[indices], fitted)]


def _debiased_bar_potentials(graph, potentials, bar):
    """Counteract positive per-edge 4/4 bias so 5/4 and 6/4 spans stay competitive."""
    score, switch, compound = potentials
    bar_f = bar.float().to(score.device)
    prefix = torch.nn.functional.pad(torch.nn.functional.logsigmoid(-bar_f).cumsum(0), (1, 0))
    a = torch.as_tensor(graph.starts, device=score.device, dtype=torch.long)
    b = torch.as_tensor(graph.ends, device=score.device, dtype=torch.long)
    bernoulli = torch.nn.functional.logsigmoid(bar_f[b]) + prefix[b] - prefix[a + 1]
    edge = score - bernoulli
    span = torch.as_tensor(graph.span, device=edge.device, dtype=edge.dtype)
    boost = 1.0 + 0.75 * (span / 4.0 - 1.0)
    edge = torch.where((edge > 0) & (span > 4.0) & (span <= 6.5), edge * boost, edge)
    return bernoulli + edge, switch.clamp_min(-1.5), compound


def _decode_bar_lattice(head, coords, song, *, debias=False):
    """Score all observations, including an incomplete pickup or tail."""
    from .calibrator_semicrf import BarLattice

    bar = head.bar_logits(song)
    if len(coords) < 2 or coords[-1] - coords[0] < 0.25:
        score = torch.nn.functional.logsigmoid(-bar).sum() + bar.max().clamp_min(0)
        path = np.array([int(bar.argmax())], dtype=int) if bar.max() >= 0 else np.empty(0, int)
        return path, [], float(score)
    graph = BarLattice(coords)
    potentials = head.potentials(graph, song)
    if debias and not getattr(head, "v1", False):
        potentials = _debiased_bar_potentials(graph, potentials, bar)
    path, score = head.decode(graph, potentials, boundary_logits=bar, return_score=True)
    meters = []
    if len(path) > 1:
        ends = path[1:]
        edges = graph.in_offsets[ends] + path[:-1] - graph.starts[graph.in_offsets[ends]]
        compound = potentials[2][edges].detach().cpu().numpy()
        spans = graph.span[edges]
        can_be_compound = np.isclose(spans, 3, atol=1e-4, rtol=0) | np.isclose(spans, 6, atol=1e-4, rtol=0)
        for k, span, use in zip(path, spans, can_be_compound & (compound >= 0)):
            meter = (round(2 * span), 8) if use else multiplier_to_meter(span / 4)
            meters.append((int(k), *meter))
    return path, meters, score


def _bar_path(head, coords, song):
    return _decode_bar_lattice(head, coords, song, debias=False)


def _final_bar_path(head, coords, song):
    """Decode final bars/meters after clock selection, applying v2 span debiasing."""
    return _decode_bar_lattice(head, coords, song, debias=True)


def _merge_cross_island_segments(segments, running, selected, coords, breaks, *, snap, drift):
    """Merge adjacent tempo segments split only by a soft island boundary (or in-phase break)."""
    if len(segments) <= 1 or not len(selected):
        return segments
    merged = [segments[0]]
    n_sel = len(selected)
    for j in range(1, len(segments)):
        start, rate = segments[j]
        a, last_rate = merged[-1]
        a_sel = min(n_sel - 1, int(np.searchsorted(selected, a, side="left")))
        start_sel = min(n_sel - 1, int(np.searchsorted(selected, start, side="left")))
        end_sel = (
            min(n_sel - 1, int(np.searchsorted(selected, segments[j + 1][0], side="left")))
            if j + 1 < len(segments)
            else n_sel - 1
        )
        cross_break = [br for br in breaks if a_sel < br <= start_sel]
        tol_ratio = 0.012 if cross_break else 0.020
        if abs(rate / last_rate - 1.0) <= tol_ratio and end_sel - a_sel >= 2:
            pos = coords[a_sel : end_sel + 1] - coords[a_sel]
            obs = running[selected[a_sel : end_sel + 1]] - running[selected[a_sel]]
            if pos[-1] > 0:
                new_rate, residual, _ = _fit_linear_bpm(pos, obs, last_rate, snap=snap, drift=drift)
                med = float(np.median(residual)) if len(residual) else 0.0
                p90 = float(np.quantile(residual, 0.90)) if len(residual) else 0.0
                if med <= 0.015 and (not cross_break or (p90 <= 0.025 and abs(new_rate / last_rate - 1.0) <= 0.004)):
                    gap_ok = True
                    for br in cross_break:
                        if br > 0:
                            gap_obs = running[selected[br]] - running[selected[br - 1]]
                            gap_exp_old = (coords[br] - coords[br - 1]) * 60.0 / last_rate
                            gap_exp_new = (coords[br] - coords[br - 1]) * 60.0 / new_rate
                            if abs(gap_obs - gap_exp_old) > 0.020 or abs(gap_obs - gap_exp_new) > 0.020:
                                gap_ok = False
                                break
                    if gap_ok:
                        merged[-1] = (a, float(new_rate))
                        continue
        merged.append((start, rate))
    return merged


def _decode_clock_islands(head, song, running, end, accent, snap, drift):
    raw = 120 * np.exp2(song["bpm"].float().cpu().numpy())
    changes, valid = (song[k].float().sigmoid().cpu().numpy() for k in ("change", "valid"))
    steps = song["step"].float().cpu().numpy()
    octaves = song["octave"].float().cpu().numpy()
    log_probs = log_softmax(octaves, axis=1)
    weights = 0.5 + song["bar"].float().sigmoid().cpu().numpy()
    coords = []
    selected = []
    bars, meters, segments, breaks = [], [], [], []
    from .calibrator_corrector import observed_beat_steps

    scales = np.array([0.5, 1., 2.])
    rates = raw[:, None] * scales
    legal_rates = (rates >= 45) & (rates <= 320)
    votes = np.where(legal_rates | ~legal_rates.any(1, keepdims=True), octaves, -np.inf)
    preliminary = rates[np.arange(len(raw)), votes.argmax(1)]
    quarter = "beat_logit" in song
    bar_all = weights - 0.5
    rough = np.r_[0.0, observed_beat_steps(running, preliminary, changes, quarter=quarter, bar_prob=bar_all).cumsum()]
    tail_ok = _tail_support(running, rough, changes, steps, accent, valid, end, preliminary[0])
    gaps = np.diff(running)
    separated = (gaps > 2.5) | (gaps * preliminary[:-1] / 60 > 8)
    cuts = _tempo_islands(preliminary, changes, separated, steps=steps, running=running, quarter=quarter)
    if not tail_ok:
        # A weak tail change must not bypass the DP hold by opening a new island.
        starts = [
            a
            for a, _ in cuts
            if a == 0
            or running[a] < end - 8
            or separated[a - 1]
        ]
        cuts = list(zip(starts, starts[1:] + [len(raw)]))
    previous_class = None
    previous_rate = None
    for start, stop in cuts:
        part = slice(start, stop)
        piece = {key: value[part] for key, value in song.items()}
        t = running[part]
        probs = softmax(np.average(octaves[part], axis=0, weights=weights[part]))
        top = int(probs.argmax())
        long_gap = start > 0 and separated[start - 1]
        choices = [top]
        raw_center = np.median(raw[part])
        centers = raw_center * scales
        octave_jump = previous_rate is not None and abs(np.log2(centers[top] / previous_rate)) >= 0.75
        if (probs[top] < .85 or not 45 <= centers[top] <= 320 or long_gap or octave_jump
                or previous_class not in (None, top)):
            order = np.argsort(-probs)
            legal = [k for k in order if 45 <= centers[k] <= 320]
            choices = [k for k in legal if k in order[:2] or probs[k] >= .08] or legal[:1] or choices
        eighth = None
        pulse_seconds = 60. if quarter else 30.
        pulse = piece.get("beat_logit", piece.get("eighth_logit"))
        if pulse is not None:
            eighth = pulse.float().sigmoid().cpu().numpy()
            strong = t[eighth >= .45]
            if len(strong) >= 4:
                ioi_center = pulse_seconds / np.median(np.diff(strong))
                if 45 <= ioi_center <= 320 and abs(np.log2(ioi_center / centers[top])) >= .65:
                    nearest = int(np.argmin(abs(np.log2(centers / ioi_center))))
                    if nearest not in choices:
                        choices.append(nearest)
        winner = None
        skip = None
        for k in choices:
            rate = raw[part] * 2.0 ** (k - 1)
            fold_ratio = rate / max(centers[k], 1.0)
            rate = np.where(fold_ratio > 1.65, rate * 0.5, np.where(fold_ratio < 0.60, rate * 2.0, rate))
            kept, local, fitted = _fit_island(
                t, rate, changes[part], steps[part], accent[part], valid[part], end, snap, drift, tail_ok,
                eighth=eighth, bar=weights[part]-.5, pulse_seconds=pulse_seconds,
            )
            nodes, skip_score = piece, 0.0
            if len(kept) != len(t):
                nodes = {key: value[kept] for key, value in piece.items()}
                if skip is None:
                    skip = torch.nn.functional.logsigmoid(-head.bar_logits(piece)).detach().cpu().numpy()
                dropped = np.ones(len(t), dtype=bool)
                dropped[kept] = False
                skip_score = skip[dropped].sum(dtype=np.float64)
            path, meter, score = _bar_path(head, local, nodes)
            score += skip_score
            # Neighboring eighths share octave evidence. Count roughly one vote per bar.
            score += 0.5 * max(1.0, len(t) / (4 if pulse_seconds == 60. else 8)) * np.average(log_probs[part, k], weights=weights[part])
            if previous_rate is not None and abs(fitted[0][1] / previous_rate - 1) <= 0.15:
                score += 1.5
            if winner is None or score > winner[0]:
                winner = (score, k, kept, local, fitted, path, meter)
        _, previous_class, kept, local, fitted, path, meter = winner
        base = len(coords)
        if start:
            gap = t[kept[0]] - running[selected[-1]]
            distance = gap * previous_rate / 60
            if long_gap:
                breaks.append(base)
                anchor = coords[-1] + distance
                grid_step = 1 if quarter else 2
                if (
                    bars
                    and len(path)
                    and abs(fitted[0][1] / previous_rate - 1) <= 0.015
                    and changes[start] < 0.5
                ):
                    first = path[0]
                    anchor = (
                        coords[bars[-1]]
                        + round((anchor + local[first] - coords[bars[-1]]) * grid_step) / grid_step
                        - local[first]
                    )
                else:
                    anchor = round(anchor * grid_step) / grid_step
            else:
                step = observed_beat_steps(
                    np.array([0.0, gap]),
                    np.array([previous_rate, fitted[0][1]]),
                    np.array([0.0, changes[start]]),
                    quarter=quarter,
                    bar_prob=np.array([bar_all[selected[-1]], bar_all[start + kept[0]]]),
                )[0]
                anchor = coords[-1] + step
            local = local + anchor
        coords.extend(local)
        selected.extend(start + kept)
        bars.extend((path + base).tolist())
        meters.extend((base + k, num, den) for k, num, den in meter)
        for k, rate in fitted:
            if not segments or abs(segments[-1][1] - rate) > 1e-6:
                segments.append((start + k, rate))
        previous_rate = fitted[-1][1]
    # A tempo boundary can fall inside a measure. Decode across adjacent clock
    # islands after selecting their scales; a true long rest reopens the path.
    # For clean-v2 heads, always run the final span-debiased pass after octave
    # selection so octave competition stays unbiased while final meters benefit.
    coords = np.asarray(coords)
    selected = np.asarray(selected)
    if len(coords) and (len(cuts) > len(breaks) + 1 or not getattr(head, "v1", False)):
        bars, meters = [], []
        compact = {key: value[selected] for key, value in song.items()}
        for ids in np.split(np.arange(len(coords)), breaks):
            path, meter, _ = _final_bar_path(head, coords[ids], {k: v[ids] for k, v in compact.items()})
            bars.extend(ids[path].tolist())
            meters.extend((int(ids[k]), num, den) for k, num, den in meter)
    segments = _merge_cross_island_segments(
        segments, running, selected, coords, breaks, snap=snap, drift=drift
    )
    return selected, coords, segments, bars, meters, breaks


def _physical_time(run_t, stops):
    """Invert BeatGrid._running on sorted STOP intervals."""
    t = float(run_t)
    elapsed = 0.0
    for st in sorted(stops, key=lambda s: s["time"]):
        pos = float(st["time"]) - elapsed
        if pos < t - 1e-9:
            elapsed += float(st["duration"])
        else:
            break
    return t + elapsed


def _reconcile_and_export_clean_segments(
    running,
    export,
    raw_segments,
    selected,
    bars,
    breaks,
    stops,
    *,
    bar_prob=None,
    quarter=False,
    snap=True,
    drift=2.0,
):
    """Snap tempo boundaries onto barlines/discrete beats and export beat-synchronized BPM rows."""
    n = len(running)
    if not n:
        return 0.0, [dict(start_seconds=0.0, beat=0.0, bpm=120.0)]
    if n == 1:
        bpm0 = float(raw_segments[0][1]) if raw_segments else 120.0
        t0 = _physical_time(float(running[0]) - float(export[0]) * 60.0 / bpm0, stops)
        return float(t0), [dict(start_seconds=float(t0), beat=0.0, bpm=bpm0)]

    # Map raw candidate indices onto selected indices and ensure monotone unique starts.
    mapped = []
    for k, bpm in raw_segments:
        sel_idx = min(n - 1, int(np.searchsorted(selected, k, side="left")))
        if not mapped or sel_idx > mapped[-1][0]:
            mapped.append([sel_idx, float(bpm)])
        else:
            mapped[-1][1] = float(bpm)
    mapped[0][0] = 0
    # Drop trailing degenerate boundary on the very last observation.
    while len(mapped) > 1 and mapped[-1][0] >= n - 1:
        mapped.pop()

    bar_set = set(int(b) for b in bars)
    break_set = set(int(b) for b in breaks)
    # Seed explicit break boundaries into mapped so equal-BPM out-of-phase rests
    # are tested separately before optional in-phase merging in step 3.
    existing = {row[0] for row in mapped}
    for br in sorted(break_set):
        if 0 < br < n - 1 and br not in existing:
            prev_bpm = next((r[1] for r in reversed(mapped) if r[0] < br), mapped[0][1])
            mapped.append([br, prev_bpm])
            existing.add(br)
    mapped.sort(key=lambda r: r[0])

    def seg_end(j):
        if j + 1 >= len(mapped):
            return n - 1
        nxt = mapped[j + 1][0]
        return max(mapped[j][0], nxt - 1) if nxt in break_set else nxt

    def segment_fit(a, b, fallback):
        pos = export[a : b + 1] - export[a]
        obs = running[a : b + 1] - running[a]
        return _fit_linear_bpm(pos, obs, fallback, snap=snap, drift=drift)

    def split_cost(lo, cut, hi, fb_l, fb_r):
        _, res_l, _ = segment_fit(lo, cut, fb_l)
        _, res_r, _ = segment_fit(cut, hi, fb_r)
        return float(np.minimum(res_l, 0.040).sum() + np.minimum(res_r, 0.040).sum())

    # 1. Snap internal non-barline tempo cuts to nearby decoded barlines (or integer beats).
    for j in range(1, len(mapped)):
        cur, r_cur = mapped[j]
        if cur in break_set or cur in bar_set:
            continue
        lo = mapped[j - 1][0]
        hi = mapped[j + 1][0] if j + 1 < len(mapped) else n - 1
        r_prev = mapped[j - 1][1]
        if hi - lo < 2:
            continue
        base_cost = split_cost(lo, cur, hi, r_prev, r_cur)
        best_cut = cur
        best_adj = base_cost
        max_span = 4.5 if (bar_prob is not None and bar_prob[cur] < 0.35) else 2.5
        cand_bars = [
            b for b in bars
            if lo < b < hi and abs(export[b] - export[cur]) <= max_span
            and (abs(export[b] - export[cur]) <= 2.5 or (bar_prob is not None and bar_prob[b] >= 0.50))
        ]
        on_int_beat = abs(export[cur] - round(export[cur])) <= 1e-4
        for b in cand_bars:
            cost_b = split_cost(lo, b, hi, r_prev, r_cur)
            # Prefer barlines over off-beat cuts, but stay conservative if cur
            # is already on an exact integer beat so we don't absorb a real ramp.
            if on_int_beat:
                margin = min(0.012, 0.005 * max(1.0, abs(export[b] - export[cur])))
            else:
                margin = min(0.030, 0.012 * max(1.0, abs(export[b] - export[cur])))
            if cost_b <= base_cost + margin and cost_b < best_adj + margin:
                best_adj = cost_b - margin
                best_cut = int(b)
        if best_cut == cur and abs(export[cur] - round(export[cur])) > 1e-4:
            # Never leave a non-barline tempo change off an integer beat when an integer beat is adjacent.
            near_int = [
                c for c in range(max(lo + 1, cur - 2), min(hi, cur + 3))
                if abs(export[c] - round(export[c])) <= 1e-4 and abs(export[c] - export[cur]) <= 1.0
            ]
            if near_int:
                best_cut = min(near_int, key=lambda c: split_cost(lo, c, hi, r_prev, r_cur))
        mapped[j][0] = best_cut

    # 2. Re-fit all segments with free-intercept IRLS + integer-first snap.
    for j in range(len(mapped)):
        a = mapped[j][0]
        b = seg_end(j)
        mapped[j][1] = segment_fit(a, b, mapped[j][1])[0]

    # 3. Prune <2-beat phantom/octave glitches and merge adjacent same-BPM segments.
    changed = True
    while changed and len(mapped) > 1:
        changed = False
        # (a) Drop isolated <2-beat glitches between un-broken segments.
        for j in range(1, len(mapped)):
            a, rate = mapped[j]
            b = mapped[j + 1][0] if j + 1 < len(mapped) else n - 1
            if a in break_set or (j + 1 < len(mapped) and mapped[j + 1][0] in break_set):
                continue
            if export[b] - export[a] >= 2.0 - 1e-6:
                continue
            cands = []
            # Absorb into left
            la, lr = mapped[j - 1]
            mr, res, _ = segment_fit(la, b, lr)
            med = float(np.median(res)) if len(res) else 0.0
            lim = 0.035 if abs(np.log2(max(rate, 1.0) / max(lr, 1.0))) >= 0.70 else 0.022
            if med <= lim:
                cands.append((med, "left", mr))
            # Absorb into right
            if j + 1 < len(mapped):
                _, rr = mapped[j + 1]
                end2 = mapped[j + 2][0] if j + 2 < len(mapped) else n - 1
                mr, res, _ = segment_fit(a, end2, rr)
                med = float(np.median(res)) if len(res) else 0.0
                lim = 0.035 if abs(np.log2(max(rate, 1.0) / max(rr, 1.0))) >= 0.70 else 0.022
                if med <= lim:
                    cands.append((med, "right", mr))
            if cands:
                cands.sort(key=lambda x: x[0])
                _, side, mr = cands[0]
                if side == "left":
                    mapped[j - 1][1] = mr
                    mapped.pop(j)
                else:
                    mapped[j][1] = mr
                    mapped.pop(j + 1)
                changed = True
                break
        if changed:
            continue
        # (b) Merge adjacent segments with near-identical BPM (including phase-aligned breaks).
        for j in range(1, len(mapped)):
            a, lr = mapped[j - 1]
            cut, rr = mapped[j]
            end_r = seg_end(j)
            cross_br = [br for br in breaks if a < br <= cut]
            tol_ratio = 0.012 if cross_br else 0.016
            if abs(rr / lr - 1.0) > tol_ratio:
                continue
            mr, res, _ = segment_fit(a, end_r, lr)
            med = float(np.median(res)) if len(res) else 0.0
            p90 = float(np.quantile(res, 0.90)) if len(res) else 0.0
            if med <= 0.015 and p90 <= 0.028 and (not cross_br or abs(mr / lr - 1.0) <= 0.004):
                gap_ok = True
                for br in cross_br:
                    if br > 0:
                        gap_obs = running[br] - running[br - 1]
                        gap_exp_old = (export[br] - export[br - 1]) * 60.0 / lr
                        gap_exp_new = (export[br] - export[br - 1]) * 60.0 / mr
                        if abs(gap_obs - gap_exp_old) > 0.020 or abs(gap_obs - gap_exp_new) > 0.020:
                            gap_ok = False
                            break
                if gap_ok:
                    mapped[j - 1][1] = mr
                    mapped.pop(j)
                    changed = True
                    break

    # 4. Build beat-synchronized (run_start, exact_beat, bpm) rows so BeatGrid matches export exactly.
    m = len(mapped)
    anchors = []
    for j in range(m):
        s_j = mapped[j][0]
        e_j = seg_end(j)
        bpm_j = mapped[j][1]
        p_j = 60.0 / bpm_j
        seg_beats = export[s_j : e_j + 1] - export[s_j]
        seg_times = running[s_j : e_j + 1]
        anchors.append(float(np.median(seg_times - seg_beats * p_j)))

    plan = []  # list of (run_start, beat, bpm)
    cur_run = anchors[0]
    for j in range(m - 1):
        s_j, bpm_j = mapped[j]
        s_next = mapped[j + 1][0]
        b_j, b_next = float(export[s_j]), float(export[s_next])
        db = b_next - b_j
        if db <= 1e-6:
            continue
        pred_next = cur_run + db * (60.0 / bpm_j)
        target_next = anchors[j + 1]
        err = target_next - pred_next
        is_break = s_next in break_set
        is_int = abs(bpm_j - round(bpm_j)) <= 1e-6
        next_end = seg_end(j + 1)
        next_db = float(export[next_end] - export[s_next])
        next_is_long_int = next_db >= 24.0 and abs(mapped[j + 1][1] - round(mapped[j + 1][1])) <= 1e-6
        if is_break:
            tol_close = 1e-6
        elif next_is_long_int and db < 24.0:
            # Close a short transition tightly onto a long constant island's phase anchor.
            tol_close = 0.003
        else:
            tol_close = 0.015 if is_int else 0.008

        is_long_int = is_int and db >= 20.0 and not is_break
        if abs(err) <= tol_close or (is_long_int and (not next_is_long_int or abs(err) <= 0.080)):
            # Never warp a long steady integer-BPM island to chase a neighboring ramp anchor.
            plan.append((cur_run, b_j, float(bpm_j)))
            cur_run = pred_next
        elif (is_break or (is_long_int and next_is_long_int)) and s_next > s_j:
            # Confine any silent-gap or large inter-island phase adjustment strictly
            # to the boundary interval [s_next-1, s_next], leaving the long island unwarped.
            br = s_next
            left_idx = br - 1
            b_left = float(export[left_idx])
            b_right = float(export[br])
            run_left = cur_run + (b_left - b_j) * (60.0 / bpm_j)
            gap_db = b_right - b_left
            gap_dt = target_next - run_left
            if gap_db > 1e-6 and gap_dt > 1e-3:
                bridge = float(np.clip(60.0 * gap_db / gap_dt, 10.0, 1000.0))
                if b_left > b_j + 1e-6:
                    plan.append((cur_run, b_j, float(bpm_j)))
                plan.append((run_left, b_left, bridge))
                cur_run = run_left + gap_db * (60.0 / bridge)
            else:
                plan.append((cur_run, b_j, float(bpm_j)))
                cur_run = pred_next
        else:
            dt_close = target_next - cur_run
            bpm_use = float(bpm_j)
            max_stretch = 0.02 if (is_int and db >= 12.0 and not next_is_long_int) else 0.35
            if dt_close > 1e-3:
                bpm_close = 60.0 * db / dt_close
                if 10.0 <= bpm_close <= 1000.0 and abs(bpm_close / bpm_j - 1.0) <= max_stretch:
                    if snap:
                        ni = float(max(1, round(bpm_close)))
                        snap_tol = 0.003 if (next_is_long_int and db < 24.0) else 0.012
                        if abs(target_next - (cur_run + db * 60.0 / ni)) <= snap_tol:
                            bpm_close = ni
                    bpm_use = float(bpm_close)
            plan.append((cur_run, b_j, bpm_use))
            cur_run = cur_run + db * (60.0 / bpm_use)
    plan.append((cur_run, float(export[mapped[-1][0]]), float(mapped[-1][1])))

    # Shift row 0 to exact beat 0.0 (offset_seconds) when a pickup precedes bar 0 inside segment 0.
    if plan[0][1] < -1e-6 and (len(plan) == 1 or plan[1][1] > 1e-6):
        r0, b0, bpm0 = plan[0]
        offset_run = r0 - b0 * (60.0 / bpm0)
        plan[0] = (offset_run, 0.0, bpm0)
    else:
        # Locate beat 0.0 along the continuous plan.
        offset_run = plan[0][0] - plan[0][1] * (60.0 / plan[0][2])
        for idx in range(len(plan)):
            r_k, b_k, bpm_k = plan[idx]
            b_hi = plan[idx + 1][1] if idx + 1 < len(plan) else float("inf")
            if b_k - 1e-6 <= 0.0 < b_hi:
                offset_run = r_k + (0.0 - b_k) * (60.0 / bpm_k)
                break

    # Deduplicate consecutive identical BPMs.
    rows = []
    for run_s, beat_s, bpm_s in plan:
        if not rows or abs(bpm_s - rows[-1]["bpm"]) > 1e-6:
            rows.append(
                dict(
                    start_seconds=float(_physical_time(run_s, stops)),
                    beat=float(beat_s),
                    bpm=float(bpm_s),
                )
            )
    offset_sec = float(_physical_time(offset_run, stops))
    return offset_sec, rows


def _bridge_clock_gaps(running, segments, selected, coords, breaks, origin):
    """Integrate each silent bridge to the right island's exact exported beat."""
    if not breaks:
        return segments
    starts, rates = map(np.asarray, zip(*segments))
    instant = rates[(np.searchsorted(starts, np.arange(len(running)), side="right") - 1).clip(0)]
    physical = np.r_[0.0, np.cumsum(np.diff(running) * instant[:-1] / 60)]
    # The component containing the first decoded bar defines beat zero. Use
    # its actual offset even when the first few pickup observations have jitter.
    anchors = np.r_[0, breaks]
    anchors[np.searchsorted(breaks, origin, side="right")] = origin
    for cut, a, b in zip(breaks, anchors, anchors[1:]):
        left, right = selected[cut - 1], selected[cut]
        dt = running[right] - running[left]
        if dt <= 1e-4:
            continue
        voiced = (physical[left] - physical[selected[a]]) + (physical[selected[b]] - physical[right])
        beats = coords[b] - coords[a] - voiced
        bridge = 60 * beats / dt
        if not 10.0 <= bridge <= 1000.0:
            local_beats = max(0.25, float(coords[cut] - coords[cut - 1]))
            bridge = float(np.clip(60 * local_beats / dt, 10.0, 1000.0))
        instant[left:right] = bridge
    starts = np.r_[0, np.flatnonzero(abs(np.diff(instant)) > 1e-6) + 1]
    return [(int(k), float(instant[k])) for k in starts]


@torch.no_grad()
def decode_clean_grid(
    outputs,
    batch_index,
    metadata=None,
    *,
    snap=True,
    drift=2.0,
    stop_threshold=0.85,
    min_stop_frames=3,
    graph_observations=None,
):
    from .bms_calibrator import _decode_stops
    from .calibrator_corrector import candidate_times, observed_beat_steps, refined_song

    i = batch_index
    n = int(outputs["lengths"][i])

    def read(key):
        return outputs[key][i, :n].detach().float().cpu().numpy()

    audible = read("audible_mask").astype(bool)
    stopped, stops = _decode_stops(
        torch.from_numpy(expit(read("stop_logit"))) * inside_audible_bounds(torch.from_numpy(audible)),
        stop_threshold,
        min_stop_frames,
    )
    supplied = BeatGrid(metadata) if metadata is not None else None
    if supplied is not None:
        stops = metadata.get("stops", [])
    head = outputs.get("correction")
    if "log_bpm" in outputs:
        log_bpm = read("log_bpm")
    else:
        phase = read("eighth_phase" if "eighth_phase" in outputs else "pulse_phase")
        cycles = np.unwrap(np.arctan2(phase[:, 0], phase[:, 1]).astype(np.float64)) / (2 * np.pi)
        bpm = np.gradient(cycles) * HZ * (30 if "eighth_logit" in outputs else 60) if n > 1 else np.zeros(n)
        bpm = median_filter(bpm, size=9, mode="nearest")
        log_bpm = np.where((bpm > 0) & np.isfinite(bpm), np.log2(bpm.clip(50, 300) / 120), 0.)
    change_prob = expit(read("change_logit")) if "change_logit" in outputs else np.zeros(n)
    quarter_mode = "eighth_logit" not in outputs
    if head is not None:
        song, coords = refined_song(
            outputs["songs"][i],
            clock=supplied,
            stops=stops,
            audible=outputs["audible_mask"][i, :n],
            fit_clock=False,
        )
        times = song["refined"].cpu().numpy() if song is not None else np.empty(0)
        if song is not None:
            quarter_mode = "beat_logit" in song
    else:
        times = candidate_times(
            (
                read("eighth_logit" if "eighth_logit" in outputs else "beat_logit"),
                read("beat_logit" if "eighth_logit" in outputs else "downbeat_logit"),
                read("eighth_phase" if "eighth_phase" in outputs else "pulse_phase")[:, 1],
                read("subframe_offset_ms"),
                audible & ~np.r_[False, stopped.numpy()[1:] & stopped.numpy()[:-1]],
                log_bpm,
                *((read("downbeat_logit"),) if "eighth_logit" not in outputs else ()),
            ),
            legacy="log_bpm" in outputs,
        )
        frames = (times * HZ).astype(int).clip(0, n - 1)
        prior = 120 * 2 ** log_bpm[frames]
        if supplied is None:
            steps = observed_beat_steps(_running(times, stops), prior, change_prob[frames], quarter=quarter_mode)
            coords = np.r_[0.0, steps.cumsum()][: len(times)]
        else:
            coords = np.round(np.array([supplied.second_to_beat(t) for t in times]) * 4) / 4
    frames = (times * HZ).astype(int).clip(0, n - 1)
    keep = np.flatnonzero((times >= 0) & (times < n / HZ) & audible[frames])
    times, coords, frames = times[keep], coords[keep], frames[keep]
    prior = 120 * 2 ** log_bpm[frames]
    accent = expit(read("beat_logit")[frames])
    changes = change_prob[frames]
    steps = np.zeros_like(times)
    bars, meters = [], []
    island_segments = None
    selected = np.arange(len(times), dtype=int)
    breaks = []
    bar_prob_sel = None
    if head is not None and len(times):
        song = {k: v[keep] for k, v in song.items()}
        if supplied is None:
            running = _running(times, stops)
            end = _running(np.array([(np.flatnonzero(audible)[-1] + 1) / HZ]), stops)[0]
            selected, coords, island_segments, bars, meters, breaks = _decode_clock_islands(
                head.bars, song, running, end, accent, snap, drift
            )
            bar_prob_sel = song["bar"][selected].float().sigmoid().cpu().numpy()
            times = times[selected]
        else:
            breaks = (np.flatnonzero(np.diff(coords) > 12) + 1).tolist()
            for ids in np.split(np.arange(len(coords)), breaks):
                path, meter, _ = _final_bar_path(head.bars, coords[ids], {k: v[ids] for k, v in song.items()})
                bars.extend(ids[path].tolist())
                meters.extend((int(ids[k]), num, den) for k, num, den in meter)
    elif len(times):
        bar_prob_sel = expit(read("downbeat_logit")[frames])
        bars = np.flatnonzero(bar_prob_sel >= 0.5).tolist()
        meters = [(a, *multiplier_to_meter((coords[b] - coords[a]) / 4)) for a, b in zip(bars, bars[1:])]
    if graph_observations is not None:
        graph_observations.update(
            times=times,
            coords=coords,
            component=np.searchsorted(breaks, np.arange(len(times)), side="right"),
        )
    origin = bars[0] if bars else 0
    if supplied is None:
        export = coords - (coords[origin] if len(coords) else 0.0)
        if len(times):
            running = _running(times, stops)
            end = _running(np.array([(np.flatnonzero(audible)[-1] + 1) / HZ]), stops)[0]
            segments = island_segments or _learned_tempo_segments(
                running, coords, prior, changes, steps, accent, end,
                snap=snap, drift=drift, bar_prob=bar_prob_sel,
            )
            offset_sec, rows = _reconcile_and_export_clean_segments(
                running,
                export,
                segments,
                selected,
                bars,
                breaks,
                stops,
                bar_prob=bar_prob_sel,
                quarter=quarter_mode,
                snap=snap,
                drift=drift,
            )
        else:
            offset_sec = 0.0
            rows = [dict(start_seconds=0.0, beat=0.0, bpm=120.0)]
        meta = dict(
            offset_seconds=float(offset_sec),
            bpm=float(rows[0]["bpm"]),
            bpm_segments=rows,
            stops=stops,
        )
        clock = BeatGrid(meta)
        for row in rows:
            row["beat"] = float(clock.second_to_beat(row["start_seconds"]))
    else:
        meta, export = dict(metadata), coords
    meter_segments = []
    for k, num, den in meters:
        if not meter_segments or meter_segments[-1][1:] != (num, den):
            meter_segments.append((float(export[k]), num, den))
    if not meter_segments:
        meter_segments = [(0.0, 4, 4)]
    observed_bars = [float(export[k]) for k in bars]
    structural_bars = []
    for j, beat in enumerate(observed_bars):
        if j and (
            coords[bars[j]] - coords[bars[j - 1]] > 12
            or any(bars[j - 1] < cut <= bars[j] for cut in breaks)
        ):
            # BeatGrid uses the distance between explicit lines as a measure.
            # Carry the previous meter through a long rest for downstream tick
            # quantization; only measured lines are exported as audible clicks.
            left = observed_bars[j - 1]
            _, num, den = next((m for m in reversed(meter_segments) if m[0] <= left), meter_segments[0])
            span = 4 * num / den
            structural_bars.extend(np.arange(left + span, beat - 1e-5, span).tolist())
        structural_bars.append(beat)
    meta.update(
        bar_beats=structural_bars,
        meter_segments=meter_segments,
        numerator=meter_segments[0][1],
        denominator=meter_segments[0][2],
    )
    if supplied is not None:
        return meta
    grid = BeatGrid(meta)
    first = min(float(export[0]), float(grid.second_to_beat(float(times[0])))) if len(times) else 0.0
    tactus = []
    if len(times):
        quarter = "beat_logit" in song if head is not None else "eighth_logit" not in outputs
        last = max(float(export[-1]), float(grid.second_to_beat(float(times[-1]))))
        tactus = grid.calibrator_tactus_beats(first - 1e-5, last + 1e-5)[1]
        if quarter:
            # Odd-eighth interior groups are acoustically observed rather than
            # forced onto equally spaced quarters in training or listening output.
            def odd_eighth(beat):
                meter = max(0, np.searchsorted(grid.meter_starts, beat + 1e-6, side="right") - 1)
                return grid.meter_dens[meter] >= 8 and grid.meter_nums[meter] % 2
            if any(d >= 8 and n % 2 for _, n, d in meter_segments):
                strength = read("beat_logit")[(times * HZ).astype(int).clip(0, n - 1)]
                observed_odd = [b for b, logit in zip(export, strength) if odd_eighth(b) and logit >= -.4]
                tactus = sorted(set(b for b in tactus if not odd_eighth(b)) | set(observed_odd) | set(observed_bars))

    def observed(marks):
        result = [grid.beat_to_second(b) for b in marks]
        kept = []
        for s in result:
            if 0 <= s < n / HZ:
                f = int(s * HZ)
                if audible[f] or (f + 1 < n and audible[f + 1]) or (f > 0 and audible[f - 1]):
                    kept.append(s)
        return kept

    return dict(
        meta,
        grid=grid,
        first_beat=first,
        beat_times=observed(tactus),
        downbeat_times=observed(observed_bars),
        decoder_preset="clean-lattice" if head else "clean-acoustic",
        meter_segment_dicts=[
            dict(beat=b, numerator=num, denominator=den) for b, num, den in meter_segments
        ],
    )


_METER_BUCKETS = ("regular_4_4", "non_4_4", "tempo_or_stop")


def _nearest_times(seconds, reference):
    if not len(seconds):
        return np.zeros(len(reference), dtype=int), np.full(len(reference), np.inf)
    right = np.searchsorted(seconds, reference).clip(0, len(seconds) - 1)
    left = np.maximum(right - 1, 0)
    ids = np.where(abs(seconds[left] - reference) <= abs(seconds[right] - reference), left, right)
    return ids, abs(seconds[ids] - reference)


def _bar_matches(predicted, reference, tol):
    pairs = []
    j = 0
    for i, mark in enumerate(reference):
        while j < len(predicted) and predicted[j] < mark - tol:
            j += 1
        if j < len(predicted) and predicted[j] <= mark + tol:
            pairs.append((i, j))
            j += 1
    return np.asarray(pairs, dtype=int).reshape(-1, 2)


class _ValidationBarHead:
    """CPU decoder requests edge scores from the thread owning the live GPU head."""
    def __init__(self, jobs, song_index, shutdown, *, v1=False):
        self.shutdown = shutdown
        self.jobs = jobs
        self.song_index = song_index
        self.v1 = v1

    @staticmethod
    def bar_logits(song):
        return song["boundary_logits"]

    def potentials(self, graph, song):
        from concurrent.futures import FIRST_COMPLETED, Future, wait
        response = Future()
        self.jobs.put((self.song_index, graph, song["source"], response))
        # If the training thread exits mid-validation, release waiting workers before joining them.
        done, _ = wait((response, self.shutdown), return_when=FIRST_COMPLETED)
        return next(iter(done)).result()

    @staticmethod
    def decode(graph, potentials, **kwargs):
        from .calibrator_semicrf import ContextualBarSemiCRF
        return ContextualBarSemiCRF.decode(graph, potentials, **kwargs)


@torch.no_grad()
def clean_validation_counts(outputs, targets):
    """Parallel CPU clocks, scans and metrics with live-device edge scoring."""
    from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
    from contextlib import ExitStack
    from queue import SimpleQueue
    from types import SimpleNamespace

    lengths = outputs["lengths"].tolist()
    if len(lengths) < 2:
        return _clean_song_counts(outputs, targets, 0, lengths[0]) if lengths else Counter()

    head = outputs["correction"].bars
    device = next(head.parameters()).device
    jobs = SimpleQueue()
    counts = Counter()
    with ThreadPoolExecutor(max_workers=min(8, len(lengths))) as pool, ExitStack() as cleanup:
        if device.type == "cpu":
            def cpu_song(i):
                return _clean_song_counts(outputs, targets, i, lengths[i])
            for result in pool.map(cpu_song, range(len(lengths))):
                counts.update(result)
            return counts

        keys = ("lengths", "audible_mask", "stop_logit", "log_bpm", "beat_logit", "change_logit")
        cpu = {k: outputs[k].detach().cpu() for k in keys}
        keys = ("gold_bar", "nearest_delta_ms", "meter_num", "meter_den", "change", "stop", "log_bpm", "bar_span")
        labels = {k: targets[k].detach().cpu() for k in keys}
        songs = outputs["songs"]
        cpu["songs"] = [None if s is None else {"source": torch.arange(len(s["times"]))} for s in songs]
        active = [s for s in songs if s is not None]
        if active:
            sizes = [len(s["times"]) for s in active]
            fields = ["times", "delta", "valid", "bpm", "change", "bar", "step", "tactus"]
            if "eighth_logit" in active[0]:
                fields.append("eighth_logit")
            elif "beat_logit" in active[0]:
                fields.append("beat_logit")
            packed = [torch.cat((torch.stack([s[k] for k in fields]+[head.bar_logits(s)],1),s["octave"]),1)
                      for s in active]
            fields.append("boundary_logits")
            rows = torch.cat(packed).float().cpu().split(sizes)
            for song, row in zip((s for s in cpu["songs"] if s is not None), rows):
                song.update(zip(fields,row[:, :len(fields)].unbind(1)))
                song["octave"] = row[:, len(fields):]

        shutdown = Future()
        cleanup.callback(shutdown.set_exception, CancelledError())

        def decode(i):
            local = dict(
                cpu,
                correction=SimpleNamespace(
                    bars=_ValidationBarHead(jobs, i, shutdown, v1=getattr(head, "v1", False))
                ),
            )
            return _clean_song_counts(local, labels, i, lengths[i])

        futures = [pool.submit(decode, i) for i in range(len(lengths))]
        for future in futures:
            future.add_done_callback(lambda _: jobs.put(None))
        pending = len(futures)
        while pending:
            job = jobs.get()
            if job is None:
                pending -= 1
                continue
            i, graph, source, response = job
            try:
                ids = source.to(device, non_blocking=True)
                keys = ("z", "frames", "sem_integral", "tactus", "bar_phase", "bar")
                piece = {k: songs[i][k][ids] for k in keys}
                edge, switch, compound = head.potentials(graph, piece)
                response.set_result(torch.cat((edge,switch,compound)).cpu().split(
                    [len(edge),len(switch),len(compound)]))
            except Exception as exc:
                response.set_exception(exc)
        for future in futures:
            counts.update(future.result())
    return counts


@torch.no_grad()
def _clean_song_counts(outputs, targets, i, n):
    counts = Counter()
    observed = {}
    decoded = decode_clean_grid(outputs, i, graph_observations=observed)
    grid = decoded["grid"]
    predicted = np.asarray(decoded["downbeat_times"])

    def read(key):
        return targets[key][i, :n].detach().cpu().numpy()

    gold = read("gold_bar")
    frames = np.flatnonzero(np.isfinite(gold))
    reference = (frames + 0.5) / HZ + 0.001 * read("nearest_delta_ms")[frames]
    _, offset = _nearest_times(predicted, reference)
    counts["offset_ms"] += np.minimum(offset, 0.5).sum() * 1000
    candidates = outputs["songs"][i]
    if candidates is not None:
        _, distances = _nearest_times(candidates["times"].detach().cpu().numpy(), reference)
        counts["covered"] += np.count_nonzero(distances <= 0.020)
    counts["matched20"] += len(_bar_matches(predicted, reference, 0.020))
    matches = _bar_matches(predicted, reference, 0.040)
    probe = np.minimum(frames + 1, n - 1)
    num, den = read("meter_num")[probe], read("meter_den")[probe]
    buckets = np.where((num == 4) & (den == 4), 0, 1)
    end = np.r_[frames[1:], n][: len(frames)]
    change = (read("change") > 0.5) | (read("stop") > 0.5)
    change |= np.r_[False, abs(np.diff(read("log_bpm"))) > 1e-6]
    cumulative = np.r_[0, change.cumsum()]
    buckets[cumulative[end] > cumulative[frames]] = 2
    times, coords = observed["times"], observed["coords"]
    span = np.diff(gold[frames])
    consecutive = np.isclose(span, read("bar_span")[probe[:-1]])
    # A missing stretch of reference bars marks both adjoining measures.
    gaps = np.flatnonzero(~consecutive)
    buckets[gaps] = buckets[gaps + 1] = 2
    predicted_buckets = np.full(len(predicted), -1, dtype=int)
    if len(frames):
        predicted_buckets = buckets[(np.searchsorted(reference, predicted, side="right") - 1).clip(0)]
        predicted_buckets[matches[:, 1]] = buckets[matches[:, 0]]
    ids, delta = _nearest_times(times, reference)
    reachable = np.zeros(len(span), dtype=bool)
    if len(times):
        reachable = (
            (delta[:-1] <= 0.040)
            & (delta[1:] <= 0.040)
            & (span >= 0.25)
            & (span <= 12)
            & np.isclose(np.diff(coords[ids]), span)
            & (np.diff(observed["component"][ids]) == 0)
        )
    meter_correct = np.zeros(len(span), dtype=bool)
    if len(span):
        info = grid.bar_info(grid.second_to_beat_tensor(torch.tensor(reference[:-1] + 0.04)))
        meter_correct = (info["num"].numpy() == num[:-1]) & (info["den"].numpy() == den[:-1])
        start = [grid.second_to_beat(t) for t in reference[:-1]]
        elapsed = np.array([grid.beat_to_second(b + d) for b, d in zip(start, span)])
        counts["clock_ms"] += abs(elapsed - reference[1:])[consecutive].sum() * 1000
    for j, prefix in enumerate(("", *[name + "_" for name in _METER_BUCKETS])):
        mask = np.ones(len(frames), dtype=bool) if not prefix else buckets == j - 1
        edges = mask[:-1] & consecutive
        counts[prefix + "reference"] += int(mask.sum())
        counts[prefix + "predicted"] += (
            len(predicted) if not prefix else int((predicted_buckets == j - 1).sum())
        )
        counts[prefix + "matched"] += int(mask[matches[:, 0]].sum())
        counts[prefix + "meters"] += int(mask[:-1].sum())
        counts[prefix + "correct"] += int((mask[:-1] & meter_correct).sum())
        counts[prefix + "edges"] += int(edges.sum())
        counts[prefix + "reachable"] += int((edges & reachable).sum())
    return counts


def clean_validation_metrics(counts):
    result = {}
    for prefix in ("", *[name + "_" for name in _METER_BUCKETS]):

        def value(key):
            return counts[prefix + key]

        total = max(1, value("predicted") + value("reference"))
        result[prefix + "bar_f1_40ms"] = 2 * value("matched") / total
        result[prefix + "meter_accuracy"] = value("correct") / max(1, value("meters"))
        result[prefix + "lattice_edge_reachability"] = value("reachable") / max(1, value("edges"))
        result[prefix + "reference_edges"] = value("edges")
    result.update(
        bar_f1_20ms=2 * counts["matched20"] / max(1, counts["predicted"] + counts["reference"]),
        candidate_bar_recall_20ms=counts["covered"] / max(1, counts["reference"]),
        bar_clock_mae_ms=counts["clock_ms"] / max(1, counts["edges"]),
        downbeat_offset_mae_ms=counts["offset_ms"] / max(1, counts["reference"]),
    )
    return result
