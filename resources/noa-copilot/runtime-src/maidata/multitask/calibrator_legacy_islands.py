"""Historical meter-head checkpoints only: V5 tempo and meter alias resolution."""
from collections import Counter
from dataclasses import dataclass
import math

import numpy as np

from .grid import BeatGrid


@dataclass
class BarEvidence:
    """Physical peaks before any STOP subtraction or quarter-beat interpolation."""

    bars: np.ndarray
    strength: np.ndarray
    pulses: np.ndarray
    beat: np.ndarray
    down: np.ndarray
    change: np.ndarray
    bpm: np.ndarray
    scores: np.ndarray
    meters: tuple
    hz: float

    @property
    def end(self):
        return len(self.beat) / self.hz

    def window(self, start, end):
        left = min(len(self.beat) - 1, max(0, int(start * self.hz)))
        return slice(left, min(len(self.beat), max(left + 1, math.ceil(end * self.hz))))

    def sample(self, signal, times, radius=0.022):
        frames = np.rint(np.asarray(times) * self.hz - 0.5).astype(int)
        offsets = np.arange(-round(radius * self.hz), round(radius * self.hz) + 1)
        return signal[np.clip(frames[..., None] + offsets, 0, len(signal) - 1)].max(axis=-1)

    def strongest_meter(self, start, end):
        scores = self.scores[self.window(start, end)].mean(axis=0)
        return self.meters[int(scores[: len(self.meters)].argmax())]

    def nearest_bar(self, time, tolerance=0.025):
        i = np.searchsorted(self.bars, time)
        lo, hi = max(0, i - 1), min(len(self.bars), i + 1)
        if lo < hi:
            j = lo + int(np.argmin(abs(self.bars[lo:hi] - time)))
            if abs(self.bars[j] - time) <= tolerance:
                return j
        return None


def _span(meter):
    return 4 * meter[0] / meter[1]


def _rows(tempos):
    # Beat coordinates are integrated once after every physical-time edit.
    return [(float(s["start_seconds"]), float(s["bpm"])) for s in tempos]


def _compact(rows):
    result = []
    for t, bpm in sorted(dict(rows).items()):
        if not result or abs(bpm - result[-1][1]) >= 0.01:
            result.append((t, bpm))
    return result


def _replace(rows, left, right, bpm):
    """Overwrite an interval; resume the original tempo at its right edge."""
    resume = next((b for t, b in reversed(rows) if t <= right), rows[0][1])
    kept = [(t, b) for t, b in rows if t < left or t > right]
    return _compact([*kept, (left, bpm), (right, resume)])


def filter_stutter_stops(stops, ev):
    """Compare independent downbeat anchors with and without the proposed freezes."""
    if len(stops) < 4:
        return stops
    clusters = []
    for stop in stops:
        t = stop["time"]
        if not clusters or t - clusters[-1][0]["time"] > 10:
            clusters.append([])
        clusters[-1].append(stop)
    kept = []
    for group in clusters:
        if sum(s["duration"] < 0.20 for s in group) < 4:
            kept.extend(group)
            continue
        lo, hi = group[0]["time"], group[-1]["time"] + group[-1]["duration"]
        use = (ev.bars >= lo - 2.5) & (ev.bars <= hi + 2.5)
        bars = ev.bars[use]
        if len(bars) < 4:
            kept.extend(group)
            continue
        running = bars.copy()
        for stop in group:
            running -= np.clip(bars - stop["time"], 0, stop["duration"])

        # Fit a quarter period to repeated bar spacings, using the same search
        # for either clock. Allow half-bars and syncopated accents between bars.
        def support(times):
            best = 0
            for i, anchor in enumerate(times):
                if bars[i] > lo:
                    break
                for dt in times[i + 1 :] - anchor:
                    for beats in (2, 3, 4, 6, 7, 8):
                        period = dt / beats
                        if not 0.19 <= period <= 0.8:
                            continue
                        ticks = (times - anchor) / period
                        hit = abs(ticks - np.round(ticks)) * period <= 0.025
                        if (hit & (bars >= hi)).any() and (hit & (bars < (lo + hi) / 2)).sum() >= 2:
                            best = max(best, int(hit.sum()))
            return best

        physical, paused = support(bars), support(running)
        if physical < 4 or physical <= paused:
            kept.extend(group)
    return kept


def halve_superfast_octave_chains(rows, ev, snap):
    """Fold only the coherent side of an octave discontinuity, retaining its times."""
    out = list(rows)
    used = set()
    for i, (_, peak) in sorted(enumerate(rows), key=lambda pair: -pair[1][1]):
        if peak <= 305 or i in used or peak > 400:
            continue
        left = right = i
        # Walk down either shoulder. A short noisy step may reverse by up to
        # 7%, but never cross the physical jump to the half-time neighbour.
        while left > 0:
            prev, cur = rows[left - 1][1], rows[left][1]
            if not (190 <= prev <= 1.07 * cur and cur <= 1.25 * prev):
                break
            left -= 1
        while right + 1 < len(rows):
            cur, nxt = rows[right][1], rows[right + 1][1]
            if not (190 <= nxt <= 1.07 * cur and cur <= 1.25 * nxt):
                break
            right += 1
        lo = rows[left][0]
        hi = rows[right + 1][0] if right + 1 < len(rows) else ev.end
        neighbours = []
        if left:
            neighbours.append((rows[left][1] / 2, rows[left - 1][1]))
        if right + 1 < len(rows):
            neighbours.append((rows[right][1] / 2, rows[right + 1][1]))
        continuity = any(abs(a / b - 1) <= 0.075 for a, b in neighbours)
        rates = [b for _, b in rows[left : right + 1]]
        bars = ev.bars[(ev.bars >= lo - 0.025) & (ev.bars < hi + 0.025) & (ev.strength >= 0.55)]
        if len(bars) > 2:
            times, speeds = np.array(rows).T
            local = speeds[
                np.clip(
                    np.searchsorted(times, (bars[:-1] + bars[1:]) / 2, side="right") - 1, 0, len(rows) - 1
                )
            ]
            spacious = np.mean(np.diff(bars) * local / 60) >= 7.5
        else:
            spacious = False
        if max(rates) > 400 or not (continuity or spacious):
            continue
        for j in range(left, right + 1):
            t, rate = rows[j]
            bpm = rate / 2
            if j + 1 < len(rows):
                end = rows[j + 1][0]
                bar_a, bar_b = ev.nearest_bar(t, 0.025), ev.nearest_bar(end, 0.025)
                if bar_a is not None and bar_b is not None and bar_b > bar_a:
                    dt = ev.bars[bar_b] - ev.bars[bar_a]
                    meter = ev.strongest_meter(t, end)
                    span = _span(meter)
                    fitted = 60 * span / dt
                    if 0.84 < bpm / fitted < 1.16 and 95 <= fitted <= 200:
                        bpm = float(round(fitted) if snap else fitted)
            if snap and bpm != round(bpm):
                end = rows[j + 1][0] if j + 1 < len(rows) else ev.end
                hits = ev.pulses[(ev.pulses >= t - 0.022) & (ev.pulses < end)]
                # Real high-rate onsets can subdivide the folded pulse.
                candidates = (math.floor(bpm), math.ceil(bpm))
                if len(hits) >= 3:

                    def residual(speed):
                        phase = (hits - t) * speed / 30
                        return np.median(abs(phase - np.round(phase))) * 30 / speed

                    nearest = min(candidates, key=residual)
                    if residual(nearest) <= residual(bpm) + 0.002:
                        bpm = float(nearest)
            out[j] = (t, bpm)
            used.add(j)
    for j in sorted(used):
        if not 0 < j < len(out) - 1:
            continue
        t, bpm = out[j]
        end, following = out[j + 1]
        previous = out[j - 1][1]
        if abs(previous / following - 1) > 0.012 or end - t > 4 or abs(bpm / previous - 1) < 0.025:
            continue
        period = 60 / previous
        pulses = np.arange(t, end + 0.022, period)
        if (
            len(pulses) >= 4
            and abs((end - t) / period - round((end - t) / period)) * period <= 0.025
            and ev.sample(ev.beat, pulses).mean() >= 0.55
            and abs(float(np.median(ev.bpm[ev.window(t, end)])) / previous - 1) < 0.075
        ):
            out[j] = (t, previous)
    return _compact(out)


@dataclass(frozen=True)
class TempoBarIsland:
    start: float
    end: float
    bpm: float
    meter: tuple


def _bar_runs(ev, lo, hi, bpm, meter):
    """Find independent phase runs, skipping extra accents and occasional missing peaks."""
    period = 60 / bpm
    step = period * _span(meter)
    taken = set()
    runs = []
    first, last = np.searchsorted(ev.bars, [lo - 0.025, hi + 0.025])
    for i in range(first, last):
        if i in taken:
            continue
        origin = ev.bars[i]
        run = [i]
        k = 1
        while origin + k * step <= hi + 0.027:
            j = ev.nearest_bar(origin + k * step)
            if j is None:
                j = ev.nearest_bar(origin + (k + 1) * step)
                if j is None or ev.bars[j] > hi + 0.025:
                    break
                k += 1
            run.append(j)
            k += 1
        if len(run) < 2:
            continue
        measured = ev.bars[run]
        length = (measured[-1] - origin) / period
        tail = hi >= ev.end - step and measured[-1] >= ev.end - 2 * step
        if (len(run) < 3 and not tail) or length + 0.12 < (8 if tail else max(16, 4 * _span(meter))):
            continue
        phase = measured - np.round((measured - origin) / step) * step
        if np.median(abs(phase - np.median(phase))) > 0.018:
            continue
        win = ev.window(origin, measured[-1])
        if np.quantile(ev.change[win], 0.9) >= 0.35:
            continue
        pulses = ev.pulses[(ev.pulses >= origin - 0.025) & (ev.pulses <= measured[-1] + 0.025)]
        delta = (pulses - np.median(phase)) / period
        residual = (delta - np.round(delta)) * period
        inliers = residual[abs(residual) <= 0.022]
        if len(inliers) < max(4, 0.40 * length) or np.mean(inliers**2) > 0.012**2:
            continue
        taken.update(run)
        runs.append(TempoBarIsland(float(origin), float(measured[-1]), bpm, meter))
    return runs


def _dominant(islands):
    durations = Counter()
    ends = {}
    for island in sorted(islands, key=lambda r: r.start):
        last = ends.get(island.meter, island.start)
        durations[island.meter] += max(0, island.end - max(last, island.start))
        ends[island.meter] = max(last, island.end)
    if durations:
        meter, seconds = durations.most_common(1)[0]
        if seconds > 0 and seconds >= 0.60 * sum(durations.values()):
            return meter
    return None


def _boundary(ev, island, sign):
    """Search up to four bars outside a fitted run for an observed change of clock."""
    p = 60 / island.bpm
    span = _span(island.meter)
    step = p * span
    edge = island.start if sign < 0 else island.end
    ids = np.where(
        (ev.bars >= min(edge, edge + sign * 4 * step) - 0.025)
        & (ev.bars <= max(edge, edge + sign * 4 * step) + 0.025)
    )[0]
    best = edge
    for j in ids[::-1] if sign < 0 else ids:
        at = ev.bars[j]
        near = abs(at - edge - round((at - edge) / step) * step) <= 0.025
        next_bar = ev.nearest_bar(at - sign * step)
        joint = ev.strength[j] >= 0.60 and ev.sample(ev.change, at, 0.08) >= 0.45
        if next_bar is None and not (joint and ev.sample(ev.down, at - sign * step) >= 0.32):
            continue
        offsets = np.arange(1, math.ceil(span)) * p
        if len(offsets) and np.mean(ev.sample(ev.beat, at - sign * offsets)) < 0.38:
            continue
        inside = ev.window(min(at, at - sign * step) + 0.08, max(at, at - sign * step) - 0.08)
        if ev.change[inside].max() >= 0.55:
            continue
        k = j + sign
        kink = (
            ev.strength[j] >= 0.65
            and 0 <= k < len(ev.bars)
            and abs(abs(at - ev.bars[k]) - step) > 0.035
            and ev.nearest_bar(at - sign * 2 * step) is not None
        )
        # A subdivided pickup can displace the first pair from the long-run
        # lattice by half a pulse. Require its own measured bar and joint peak.
        if (near and (joint or kink)) or (
            joint
            and abs(at - edge) <= 2.1 * step
            and abs((at - edge) / p * 2 - round((at - edge) / p * 2)) * p / 2 <= 0.025
        ):
            best = float(at)
    return best


def _pulse_support(ev, origin, bar, count, fractions):
    points = origin + (np.arange(count)[:, None] + np.asarray(fractions)) * bar
    return float(ev.sample(ev.beat, points).mean())


def resolve_bar_harmonic_aliases(rows, ev, islands, dom):
    if dom is None:
        return rows, []
    original = rows
    span = _span(dom)
    steps = np.array(sorted({_span(m) for m in ev.meters if m != dom}))
    confirmed = []
    for i in range(1, len(original)):
        left, cur = original[i]
        end = original[i + 1][0] if i + 1 < len(original) else ev.end
        prev = original[i - 1][1]
        following = original[i + 1][1] if i + 1 < len(original) else None
        for neighbour, sign in ((prev, 1), (following, -1)):
            if neighbour is None or not 60 <= neighbour <= 300 or abs(cur / neighbour - 1) < 0.05:
                continue
            other = min(steps, key=lambda s: abs(cur * span / s / neighbour - 1))
            if abs(cur * span / other / neighbour - 1) > 0.025:
                continue
            bar = 60 * span / neighbour
            boundary = left if sign == 1 else end
            seed = ev.nearest_bar(boundary, 0.09)
            if seed is None:
                continue
            origin = ev.bars[seed]
            lines = [origin]
            while True:
                j = ev.nearest_bar(lines[-1] + sign * bar)
                if j is None or not left - 0.09 <= ev.bars[j] <= end + 0.09:
                    break
                lines.append(ev.bars[j])
            count = len(lines) - 1
            if count < 3:
                continue
            a, b = sorted([lines[0], lines[-1]])
            meter = ev.strongest_meter(a, b)
            native = _pulse_support(ev, a, bar, count, np.arange(max(1, round(other))) / other)
            regular = _pulse_support(ev, a, bar, count, np.arange(max(1, round(span))) / span)
            dotted = _pulse_support(ev, a, bar, count, [0, 3 / 8, 6 / 8])
            # An unbroken physical bar period can outlast a 4:3 change of
            # attack grouping. Compare both grids at the point it enters, too.
            earlier = _pulse_support(ev, origin - sign * bar, bar, 1, np.arange(round(span)) / span)
            next_period = 60 * span / cur
            far = b if sign == 1 else a
            returns = all(ev.nearest_bar(far + sign * k * next_period) is not None for k in (1, 2))
            # Some heads track dotted beats as equally spaced thirds. Require
            # evidence at both ends: the original quarter pulse at entry and
            # a measured change of bar period at exit.
            if max(regular, dotted) + 0.025 < native and (earlier < 0.38 or not returns):
                continue
            if meter != dom and not returns and earlier < 0.38:
                continue
            near = origin
            limit = (
                original[i - 1][0] if sign == 1 else (original[i + 2][0] if i + 2 < len(original) else ev.end)
            )
            while True:
                j = ev.nearest_bar(near - sign * bar)
                if j is None or sign * (ev.bars[j] - limit) < -0.025:
                    break
                near = ev.bars[j]
            confirmed.append((min(near, a), max(near, b)))
            if sign == 1:
                rows = _replace(rows, left, min(end, b), neighbour)
            else:
                rows = _replace(rows, max(left, a), end, neighbour)
            break
    # A returning theme can share only every second or third bar with its
    # alias. Score both physical lattices from a common downbeat.
    references = {r.bpm for r in islands if r.end < 0.75 * ev.end and r.meter == dom}
    # An opening theme may end before sixteen beats. Two measured bars can
    # identify its tempo as a reference, never as a hard island by themselves.
    t0, first_bpm = rows[0]
    opening_bar = 60 * span / first_bpm
    if (
        len(rows) > 1
        and rows[1][0] - t0 >= 2 * opening_bar
        and all(ev.nearest_bar(t0 + k * opening_bar) is not None for k in (0, 1, 2))
    ):
        references.add(first_bpm)
    for i, (left, cur) in enumerate(list(rows)):
        if left < 0.75 * ev.end or (left, cur) not in rows:
            continue
        j = rows.index((left, cur))
        hi = rows[j + 1][0] if j + 1 < len(rows) else ev.end
        matches = []
        for ref in sorted(references):
            if not any(r.end < left and abs(r.bpm / ref - 1) > 0.05 for r in islands):
                continue
            if not any(abs(cur / ref / r - 1) <= 0.012 for r in (5 / 4, 4 / 3, 3 / 2, 4 / 5, 2 / 3)):
                continue
            bar = 60 * span / ref
            for at in ev.bars[(ev.bars >= left - 0.025) & (ev.bars < hi) & (ev.strength >= 0.60)]:
                if max(at - left, 0) > 4 * bar:
                    break
                points = np.arange(at + bar, min(ev.end, at + 6.02 * bar), bar)
                if len(points) < 2 or ev.sample(ev.down, points[1]) < 0.62:
                    continue
                strengths = ev.sample(ev.down, points)
                good = np.flatnonzero(strengths >= 0.45)
                if len(good) < 2 or good[1] != 1:
                    continue
                last = points[good[-1]]
                points = points[: good[-1] + 1]
                alt = np.arange(at + 60 * span / cur, last + 0.025, 60 * span / cur)
                gain = ev.sample(ev.down, points).mean() - ev.sample(ev.down, alt).mean() if len(alt) else 0
                if gain > 0.05 and ev.sample(ev.beat, points).mean() >= 0.40:
                    matches.append((len(good), gain, -at, ref, last))
        if matches:
            _, _, neg_at, ref, last = max(matches)
            at = -neg_at
            tail_end = next((t for t, _ in rows if t > last + 0.5 * 60 * span / ref), ev.end)
            rows = _replace(rows, max(left, at), tail_end, ref)
    return [(t, b) for t, b in rows if t < ev.end], confirmed


def anchor_stable_tempo_bar_islands(tempos, stops, ev, snap):
    original = _rows(tempos)
    rows = halve_superfast_octave_chains(original, ev, snap)
    old_rates = dict(original)
    protected = {t for t, bpm in rows if t in old_rates and bpm < 0.65 * old_rates[t]}
    if len(rows) < 2 or len(ev.bars) < 3:
        return rows, None, [], []
    islands = []
    for i, (start, bpm) in enumerate(rows):
        end = rows[i + 1][0] if i + 1 < len(rows) else ev.end
        if (end - start) * bpm / 60 < (8 if i + 1 == len(rows) else 15.8):
            continue
        meter = ev.strongest_meter(start, end)
        islands.extend(_bar_runs(ev, start, end, bpm, meter))
    dom = _dominant(islands)
    rows, harmonic = resolve_bar_harmonic_aliases(rows, ev, islands, dom)
    locked = []
    exits = []
    reset = set()
    for i, (start, bpm) in enumerate(rows):
        end = rows[i + 1][0] if i + 1 < len(rows) else ev.end
        meter = ev.strongest_meter(start, end)
        runs = [r for r in islands if r.bpm == bpm and start - 0.05 <= r.start <= end and r.end <= end + 0.05]
        if not runs:
            continue
        first, last = runs[0], max(runs, key=lambda r: r.end)
        step = _span(first.meter) * 60 / bpm
        a, b = start, end
        if i and first.start - start < 4.1 * step:
            at = _boundary(ev, first, -1)
            if abs(at - start) <= 4 * step and (
                abs(at - start) <= 0.03
                or ev.sample(ev.change, at, 0.08) >= 0.45
                or (ev.nearest_bar(at + step) is not None and ev.nearest_bar(at - step) is None)
            ):
                a = at
        if a < start - 0.025:
            previous = next((rate for t, rate in reversed(rows[:i]) if t <= a + 0.025), rows[0][1])

            # Two close steps can share a bar within the 25 ms capture
            # radius. Do not overwrite the one that better fits the attacks.
            def attack_error(speed):
                q = np.arange(1, min(4, max(1, round((start - a) * speed / 60))) + 1)
                targets = a + q * 60 / speed
                indices = np.searchsorted(ev.pulses, targets)
                if not len(ev.pulses):
                    return np.inf
                left = ev.pulses[np.maximum(0, indices - 1).clip(max=len(ev.pulses) - 1)]
                right = ev.pulses[indices.clip(max=len(ev.pulses) - 1)]
                return np.median(np.minimum(abs(targets - left), abs(targets - right)))

            if attack_error(previous) + 0.003 < attack_error(bpm):
                a = start
        if i + 1 < len(rows) and abs(last.end - end) < 1.05 * step:
            at = _boundary(ev, last, 1)
            if abs(at - end) < 0.030 or (end < at <= end + step and ev.sample(ev.change, at, 0.08) >= 0.45):
                b = at
        if any(s["time"] < b and s["time"] + s["duration"] > a for s in stops):
            continue
        # Keep a measured entrance when only the first few bars of the same
        # tempo carried a different meter or weaker downbeats.
        if a > start + 0.025 and ev.nearest_bar(start, 0.025) is not None:
            first_pulses = start + np.arange(1, 5) * 60 / bpm
            if ev.sample(ev.beat, first_pulses).mean() >= 0.50:
                a = start
        if abs(a - start) > 0.025:
            reset.add(a)
        if i + 1 < len(rows) and abs(b - end) > 0.005:
            exits.append((end, b, bpm, rows[i + 1][1]))
        locked.append(TempoBarIsland(a, b, bpm, meter))
    # Work in seconds until every edge is final. Competing sections can share
    # a boundary, but cannot subsequently move the start of the next island.
    for old_end, new_end, before, after in exits:
        if new_end > old_end:
            rows = _replace(rows, old_end, new_end, before)
        else:
            rows = _replace(rows, new_end, old_end, after)
    for island in locked:
        i = min(
            (j for j, r in enumerate(rows) if r[1] == island.bpm),
            key=lambda j: abs(rows[j][0] - island.start),
            default=None,
        )
        if i is None:
            continue
        old = rows[i][0]
        if i and abs(old - island.start) > 0.025:
            previous = rows[i - 1][1]
            if i > 1 and abs(rows[i - 1][0] - island.start) < 0.025 and island.start < old:
                rows.pop(i - 1)
            if island.start > old:
                rows = _replace(rows, old, island.start, previous)
            else:
                rows = _replace(rows, island.start, old, island.bpm)
    rows = _compact(rows)
    for before, after in zip(locked, locked[1:]):
        lo, hi = before.end, after.start
        if not 0.08 < hi - lo <= 10 or any(s["time"] < hi and s["time"] + s["duration"] > lo for s in stops):
            continue
        win = ev.window(lo, hi)
        cap = max(before.bpm, after.bpm)
        measured = ev.pulses[(ev.pulses >= lo) & (ev.pulses < hi)]
        expected = np.arange(lo, hi, 60 / before.bpm)
        sparse = len(measured) < 0.45 * len(expected)
        if np.quantile(ev.bpm[win], 0.25) >= 0.85 * before.bpm and not sparse:
            continue
        for j, (t, speed) in enumerate(rows):
            if lo - 0.025 <= t < hi - 0.025 and speed > cap:
                end = min(hi, rows[j + 1][0]) if j + 1 < len(rows) else hi
                anchors = measured[(measured >= t - 0.025) & (measured <= end + 0.025)]
                base = float(np.median(ev.bpm[ev.window(t, end)]))
                gaps = np.diff(anchors)
                consistent = gaps[(gaps >= 60 / cap) & (gaps < 1.35 * 60 / base)]
                fitted = 60 / np.median(consistent) if len(consistent) >= 2 else base
                rows[j] = (t, min(cap, float(fitted)))
    # A slowing fill can terminate in two well-measured long bars even
    # when the attack tracker doubled the intervening subdivisions.
    for island in locked:
        if island.start not in reset or island.meter != dom:
            continue
        end = island.start
        step = 60 * _span(dom) / island.bpm
        times = [end]
        for _ in range(3):
            candidates = ev.bars[
                (ev.bars >= end - 1.5 * step) & (ev.bars <= end - 0.70 * step) & (ev.strength >= 0.60)
            ]
            if len(candidates) != 1:
                break
            prior = float(candidates[0])
            times.append(prior)
            end = prior
        if len(times) < 3:
            continue
        first_gap, second_gap = times[0] - times[1], times[1] - times[2]
        if not (1.08 * step < first_gap and 1.05 * second_gap < first_gap < 1.40 * second_gap):
            continue
        for b, a in zip(times[:2], times[1:3]):
            bpm = 60 * _span(dom) / (b - a)
            rows = _replace(rows, a, b, bpm)
            protected.add(a)
    reset.update(b for _, b in harmonic)
    if dom and not stops and (reset or any(abs(bpm - round(bpm)) > 1e-3 for _, bpm in rows)):
        rows = _settle_transition_counts(rows, locked, reset, protected, dom)
    return _compact(rows), dom, harmonic, sorted(reset)


def _settle_transition_counts(rows, islands, reset, protected, dom):
    """Settle the moving clock between fixed physical downbeats, keeping integer beat boundaries."""
    span = _span(dom)
    rows = list(rows)
    # A short lead-in fit can be biased by the first partial pulse. Extend a
    # corroborated slope backwards when it lands on the same initial attack.
    if (
        len(rows) > 2
        and rows[1][0] - rows[0][0] < 60 * span / rows[1][1]
        and abs(rows[0][1] / rows[1][1] - 1) < 0.025
    ):
        period = 60 / rows[1][1]
        first = rows[1][0] - round((rows[1][0] - rows[0][0]) / period) * period
        if abs(first - rows[0][0]) <= 0.025 and first >= 0:
            rows = [(first, rows[1][1]), *rows[2:]]
    fixed = sorted(reset)
    stable = [(isl.start, isl.end) for isl in islands]

    # Close any unanchored fractional-BPM transition bridge onto an exact integer
    # beat count so downstream locked islands never inherit a fractional beat offset.
    for j in range(len(rows) - 1):
        a, bpm = rows[j]
        b = rows[j + 1][0]
        dt = b - a
        if dt <= 1e-4 or a in protected or abs(bpm - round(bpm)) <= 1e-3:
            continue
        if any(lo - 0.001 <= a < hi - 0.025 for lo, hi in stable):
            continue
        raw_beats = dt * bpm / 60.0
        target_beats = max(1, round(raw_beats))
        if abs(raw_beats - target_beats) <= 1e-3:
            continue
        bpm_new = 60.0 * target_beats / dt
        if abs(bpm_new / bpm - 1.0) <= 0.35:
            rows[j] = (a, float(bpm_new))

    # Do not change long constant sections to amortize a gap's rounding error.
    # The preceding transition pays the residual before the next island begins.
    last_anchor = rows[0][0]
    for t in fixed:
        if t <= last_anchor + 0.025:
            continue
        grid = BeatGrid(dict(bpm_segments=rows, offset_seconds=rows[0][0]))
        endbeat = grid.second_to_beat(t)
        delta = round(endbeat / span) * span - endbeat
        if abs(delta) > 1e-6:
            indices = [
                i
                for i, (a, _) in enumerate(rows[:-1])
                if i
                and a not in protected
                and last_anchor - 0.03 <= a < t - 0.02
                and rows[i + 1][0] - a <= 8 * 60 / rows[i][1]
                and rows[i + 1][0] <= t + 0.025
                and not any(lo - 0.001 <= a < hi - 0.025 for lo, hi in stable)
            ]
            if indices:
                use = indices[-1:]
                while len(use) < len(indices):
                    total = sum((rows[j + 1][0] - rows[j][0]) * rows[j][1] / 60 for j in use)
                    if abs(delta) <= 0.04 * total:
                        break
                    use = indices[-len(use) - 1 :]
                total = sum((rows[j + 1][0] - rows[j][0]) * rows[j][1] / 60 for j in use)
                if total > 0 and abs(delta) <= 0.35 * total:
                    # Keep intermediate transition boundaries on integer beats by
                    # assigning any fractional part only to the last bridge segment
                    # and distributing whole-beat steps across earlier segments.
                    int_delta = int(round(delta))
                    rem = delta - int_delta
                    per_seg = {j: 0.0 for j in use}
                    last_j = use[-1]
                    last_beats = (rows[last_j + 1][0] - rows[last_j][0]) * rows[last_j][1] / 60.0
                    if abs(rem) <= 0.35 * max(last_beats, 1e-6):
                        per_seg[last_j] += rem
                        step_dir = 1.0 if int_delta > 0 else -1.0
                        left_steps = abs(int_delta)
                        for j in reversed(use):
                            if left_steps <= 0:
                                break
                            seg_b = (rows[j + 1][0] - rows[j][0]) * rows[j][1] / 60.0
                            room = max(0, int(math.floor(0.35 * seg_b - abs(per_seg[j]))))
                            take = min(left_steps, max(1 if j == last_j else 0, room))
                            if seg_b + per_seg[j] + step_dir * take >= 1.0 and abs(per_seg[j] + step_dir * take) <= 0.35 * seg_b:
                                per_seg[j] += step_dir * take
                                left_steps -= take
                        if left_steps == 0:
                            for j in use:
                                if abs(per_seg[j]) > 1e-9:
                                    a, bpm = rows[j]
                                    dt = rows[j + 1][0] - a
                                    new_beats = dt * bpm / 60.0 + per_seg[j]
                                    rows[j] = (a, float(60.0 * new_beats / dt))
                        else:
                            for j in use:
                                a, bpm = rows[j]
                                rows[j] = (a, bpm * (1 + delta / total))
                    else:
                        for j in use:
                            a, bpm = rows[j]
                            rows[j] = (a, bpm * (1 + delta / total))
        last_anchor = t
    return rows
