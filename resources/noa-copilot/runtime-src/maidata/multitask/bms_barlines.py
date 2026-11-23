"""Recover structural BMS barlines on the accepted quarter-beat clock."""
from bisect import bisect_left, bisect_right
from itertools import groupby

import numpy as np

from .grid import BeatGrid, multiplier_to_meter


EPS = 1e-5


def _visual_bars(bars):
    """Collapse complete subdivisions, retaining original boundaries and unmatched fragments."""
    spans = np.diff(bars)
    result = [bars[0]]
    i = 0
    while i < len(spans):
        start = i
        # A staged 2 -> 1 -> sub-quarter subdivision is visual. Ordinary 2/4 stays intact.
        while i < len(spans) and abs(spans[i] - 2) < EPS:
            i += 1
        two_end = i
        while i < len(spans) and spans[i] <= 1 + EPS:
            i += 1
        staircase = (two_end > start and i > two_end and min(spans[two_end:i]) < 1 - EPS
                     and abs(spans[two_end] - 1) < EPS)
        if not staircase:
            result.extend(bars[start + 1:two_end + 1])
            start = two_end
        if i > start:
            before = result[-1] - result[-2] if len(result) > 1 else 4
            after = spans[i] if i < len(spans) else before
            parent = next((round(x) for x in (before, after) if round(x) in (2, 3, 4)
                           and abs(x - round(x)) < EPS), 4)
            # First coalesce existing microbars to an exact beat. Only these whole units
            # can be grouped into a parent bar. Irregular fragments stay available to the filter.
            units = []
            j = start
            while j < i:
                k = j + 1
                while k < i and bars[k] - bars[j] < 1 - EPS:
                    k += 1
                if abs(bars[k] - bars[j] - 1) < EPS or (staircase and abs(bars[k] - bars[j] - 2) < EPS):
                    units.append(bars[k])
                    j = k
                else:
                    if units:
                        result.append(units[-1])
                        units = []
                    j += 1
                    result.append(bars[j])
                    continue
                if units[-1] - result[-1] >= parent - EPS:
                    result.append(units[-1])
                    units = []
            if units:
                result.append(units[-1])
        if i < len(spans) and spans[i] > 1 + EPS and abs(spans[i] - 2) >= EPS:
            i += 1
            result.append(bars[i])
    return result


def _split_multi_bars(bars, trigger_beats):
    """Restore hidden boundaries only when the span, context or attacks resolve the parent."""
    spans = np.diff(bars)
    result = [bars[0]]
    for i, raw_span in enumerate(spans):
        start = bars[i]
        span = round(2 * raw_span) / 2
        if abs(raw_span - span) >= EPS or span not in (9, 10, 10.5, 12, 16, 20, 24, 28, 32):
            result.append(bars[i + 1])
            continue
        before = round(spans[i - 1], 5) if i else None
        after = round(spans[i + 1], 5) if i + 1 < len(spans) else None
        neighbours = (before, after)
        unit = span
        left, right = trigger_beats.searchsorted([start - EPS, bars[i + 1] - EPS])
        pos = trigger_beats[left:right] - start
        slots = np.rint(pos).astype(int)
        hit = np.bincount(slots[(abs(pos - slots) <= .04) & (slots < span)],
                          minlength=int(np.ceil(span)))
        if span == 10.5:
            # The three structural attacks must align. Interior ornamentation
            # can include sixteenths and triplets on the same chart.
            aligned = all(np.any(abs(pos - beat) <= .04) for beat in (0, 3.5, 7))
            if 3.5 in neighbours or aligned:
                unit = 3.5
        elif span == 9:
            accent = min(hit[[0, 3, 6]])
            triple_context = 3 in neighbours and not any(n in (4, 6) for n in neighbours)
            # Repeated three-beat attacks may end in a dense two-beat fill.
            triple_fill = (9 in neighbours and accent >= 2
                           and max(hit[1], hit[4]) < .65 * min(hit[0], hit[3])
                           and max(hit[2], hit[5]) <= .25 * accent
                           and min(hit[7], hit[8]) >= accent)
            if ((triple_context and hit[4] <= max(hit[3], hit[6])) or triple_fill
                    or (accent and max(hit[[1, 2, 4, 5, 7, 8]]) <= accent * .5)):
                unit = 3
        elif span == 10:
            # Compare normalized half-phrases so that a thicker second downbeat
            # does not mask an otherwise repeated five-beat rhythm.
            first, second = hit[:5], hit[5:]
            norm = np.linalg.norm(first) * np.linalg.norm(second)
            symmetric = norm > 0 and np.dot(first, second) >= .85 * norm
            accents = hit[0] > np.mean(first[1:]) and hit[5] > np.mean(second[1:])
            competing = max(hit[4], hit[8]) > max(hit[0], hit[5])
            # In a repeated five-beat phrase a loud midpoint can dominate
            # the cosine distance; compare the four inner beats separately.
            inner_norm = np.linalg.norm(first[1:]) * np.linalg.norm(second[1:])
            repeated = (10 in neighbours and hit[0] > 0 and hit[5] > max(second[1:])
                        and inner_norm > 0 and np.dot(first[1:], second[1:]) >= .85 * inner_norm)
            if neighbours == (5, 5) or ((symmetric and accents or repeated) and not competing):
                unit = 5
        elif span == 12:
            triple_hits = hit[::3]
            # A short irregular pickup can separate a three-beat phrase from
            # the first hidden bar (Psalms). Require its four attacks as well.
            left = i - 1
            while left >= 0 and spans[left] < 2 - EPS:
                left -= 1
            triple_pickup = (left < i - 1 and left >= 0 and abs(spans[left] - 3) < EPS
                             and after in (12, 24) and min(triple_hits) > 0)
            if neighbours == (6, 6):
                unit = 6
            elif (triple_pickup or (3 in neighbours and 4 not in neighbours)
                  or (min(triple_hits) > 0 and sum(triple_hits) >= 1.35 * max(sum(hit[::4]), 1))):
                unit = 3
            elif any(n in (None, 2, 4, 8) for n in neighbours):
                unit = 4
            elif 12 in neighbours:
                # Consecutive hidden bars inherit the nearest external parent.
                l, r = i, i
                while l and abs(spans[l - 1] - 12) < EPS:
                    l -= 1
                while r + 1 < len(spans) and abs(spans[r + 1] - 12) < EPS:
                    r += 1
                context = (spans[l - 1] if l else None, spans[r + 1] if r + 1 < len(spans) else None)
                if context == (6, 6):
                    unit = 6
                elif 3 in context and 4 not in context:
                    unit = 3
                elif any(n in (None, 2, 4, 8) for n in context):
                    unit = 4
        elif span % 3 == 0 and all(n in (3, 6) for n in neighbours) and sum(hit[::3]) > sum(hit[::4]):
            unit = 3
        else:
            unit = 4
        result.extend(start + j * unit for j in range(1, round(span / unit)))
        result.append(bars[i + 1])
    return result


def _compound(slots, short=False):
    if len(slots) == 6:
        return (slots[3] >= (4 if short else 6) and slots[3] >= (1.6 if short else 1.35) * max((slots[2] + slots[4]) / 2, 1)
                and slots[1] + slots[5] >= .20 * max(slots[0], 1))
    accent = slots[3] + slots[9]
    return (accent >= 5 and accent / 2 >= 1.35 * max(sum(slots[[2, 4, 8, 10]]) / 4, 1)
            and sum(slots[[1, 5, 7, 11]]) >= .20 * max(slots[0] + slots[6], 1))


def _meter_segments(bars, grid, triggers, ambiguous):
    spans = np.diff(bars)
    simple_evidence = set()
    meters = [multiplier_to_meter(x / 4) for x in spans]
    if triggers is None:
        marked = grid.ambiguous_compound_bars
        if marked is not None:
            marked = {round(beat * 1000) for beat in marked}
        for i, beat in enumerate(bars[:-1]):
            idx = max(0, bisect_right(grid.meter_starts, beat + EPS) - 1)
            prior = grid.meter_segments[idx][1:]
            if marked is not None and round(beat * 1000) not in marked and prior == meters[i]:
                simple_evidence.add(i)
            if prior in ((6, 8), (12, 8)) and abs(4 * prior[0] / prior[1] - spans[i]) < EPS:
                meters[i] = prior
    elif triggers and any(m in ((3, 4), (6, 4)) for m in meters):
        times = np.sort([t.time for t in triggers])
        # Anchor each cluster at its first attack; successive near hits cannot chain.
        cluster = []
        j = 0
        while j < len(times):
            k = times.searchsorted(times[j] + .005, side="right")
            cluster.append((grid.second_to_beat(times[j]), min(k - j, 3)))
            j = k
        beats, weights = np.asarray(cluster).T
        def region(i):
            seg = max(0, bisect_right(grid.beats, bars[i] + EPS) - 1)
            stop = bisect_right(grid.stop_beats, bars[i] + EPS)
            if ((seg + 1 < len(grid.beats) and grid.beats[seg + 1] < bars[i + 1] - EPS)
                    or (stop < len(grid.stop_beats) and grid.stop_beats[stop] < bars[i + 1] - EPS)):
                return None
            return (meters[i], seg, stop)
        for key, indices in groupby(range(len(spans)), key=region):
            group = list(indices)
            if key is None:
                continue
            (num, den), seg, _ = key
            short = num == 3 and len(group) in (2, 3)
            width = len(group) if short else (4 if num == 3 else 2)
            if den != 4 or num not in (3, 6):
                continue
            hist = []
            for i in group:
                lo, hi = beats.searchsorted([bars[i] - EPS, bars[i + 1] - EPS])
                pos = beats[lo:hi] - bars[i]
                slot = np.rint(2 * pos).astype(int)
                ok = abs(pos - slot / 2) <= .04
                hist.append(np.bincount(slot[ok] % (2 * num), weights=weights[lo:hi][ok], minlength=2*num))
            hist = np.asarray(hist)
            simple = [2, 4] if num == 3 else [2, 4, 8, 10]
            dotted = [3] if num == 3 else [3, 9]
            for j, h in enumerate(hist):
                if h[simple].sum() >= 2 and h[simple].mean() >= 1.35 * max(.5, h[dotted].mean()):
                    simple_evidence.add(group[j])
            if grid.bpms[seg] > 280 or len(group) < width:
                continue
            mask = np.zeros(len(group), dtype=bool)
            window = width if len(group) >= 2 * width else len(group)
            for j in range(len(group) - window + 1):
                if _compound(hist[j:j + window].sum(axis=0), short):
                    mask[j:j + window] = True
            for active, indices in groupby(range(len(mask)), key=mask.__getitem__):
                run = list(indices)
                if active:
                    # Trim unsupported edges of mixed windows to the actual bar boundary.
                    accent_slots = [3] if num == 3 else [3, 9]
                    accents = np.flatnonzero(hist[run][:, accent_slots].sum(axis=1))
                    if len(accents) and accents[-1] - accents[0] + 1 >= width:
                        for local in run[accents[0]:accents[-1] + 1]:
                            meters[group[local]] = (2 * num, 8)
    ambiguous.extend(bars[i] for i, meter in enumerate(meters)
                     if meter in ((3, 4), (6, 4)) and i not in simple_evidence)
    return [[bars[i], *meter] for i, meter in enumerate(meters) if i == 0 or meter != meters[i - 1]]


def sanitize_bms_barlines_and_meters(clean_meta, triggers=None):
    """Return (metadata, rejection reason). All clock edits preserve subsequent wall times."""
    bars = sorted(set(clean_meta.get("bar_beats") or ()))
    if len(bars) < 2:
        return clean_meta, None
    grid = BeatGrid(clean_meta)
    spans = np.diff(bars)
    if np.all(np.abs(spans - 4) < EPS):
        return clean_meta, None
    attacks = sorted(t.time for t in triggers) if triggers else []
    if attacks:
        begin = grid.second_to_beat(attacks[0])
        performance_end = max(t.end if t.end is not None else t.time for t in triggers)
        end = grid.second_to_beat(performance_end)
    meta = dict(clean_meta)
    clock_edges = sorted(set(grid.beats[1:] + grid.stop_beats))
    first_struct = next((i for i, span in enumerate(spans) if span >= 2 - EPS), 0)
    if (first_struct and bars[first_struct] - bars[0] < 2 - EPS
            and np.all(spans[:first_struct] < 1 - EPS)
            and bisect_right(clock_edges, bars[0] - EPS) == bisect_left(clock_edges, bars[first_struct] + EPS)):
        meta["offset_seconds"] = grid.beat_to_second(bars[first_struct])
        bars = bars[first_struct:]

    # Never combine across a tempo change or an existing STOP, including one inside a bar.
    cuts = {0, len(bars) - 1}
    for edge in clock_edges:
        i = bisect_left(bars, edge - EPS)
        if 0 < i < len(bars):
            cuts.add(i)
            if abs(edge - bars[i]) > EPS:
                cuts.add(i - 1)
    cuts = sorted(cuts)
    parts = [_visual_bars(bars[a:b + 1]) for a, b in zip(cuts, cuts[1:])]
    cleaned = [bars[0], *(beat for part in parts for beat in part[1:])]
    spans = np.diff(cleaned)
    if attacks:
        overlap = np.maximum(0, np.minimum(cleaned[1:], end) - np.maximum(cleaned[:-1], begin))
        odd = np.abs(spans * 2 - np.rint(spans * 2)) > 2 * EPS
        wall = np.asarray([grid.beat_to_second(b) for b in cleaned])
        duration = np.maximum(0, np.minimum(wall[1:], grid.beat_to_second(end))
                              - np.maximum(wall[:-1], attacks[0]))
        if (duration[odd].sum() > 0.40 * duration.sum()
                or any(k and len(list(g)) >= 16 for k, g in groupby(odd[overlap > EPS]))):
            return None, "reject_fake_subbeat_meter"
        active = spans[(np.asarray(cleaned[1:]) > begin + EPS) & (np.asarray(cleaned[:-1]) < end - EPS)]
        if sum(abs(2 * active - np.rint(2 * active)) > 2 * EPS) >= 6 or sum(active < 1 - EPS) > 8:
            return None, "reject_microbar_gimmick"

    beats = np.asarray([grid.second_to_beat(t) for t in attacks])
    cleaned = [bars[0]]
    for part in parts:
        edge = bisect_right(clock_edges, part[0] + EPS)
        if edge == len(clock_edges) or clock_edges[edge] >= part[-1] - EPS:
            part = _split_multi_bars(part, beats)
        cleaned.extend(part[1:])
    spans = np.diff(cleaned)
    times = [grid.beat_to_second(b) for b in cleaned]
    stops = [dict(time=s["time"], duration=s["duration"]) for s in meta.get("stops", ())]
    held = sorted((t.time, t.end) for t in triggers or () if t.end is not None)
    held_starts = [start for start, _ in held]
    held_ends = np.maximum.accumulate([end for _, end in held])
    kept_times = [times[0]]
    merged_next = False
    for i, (a, b) in enumerate(zip(cleaned, cleaned[1:])):
        t0, t1 = times[i:i + 2]
        movable = (0 < i < len(cleaned) - 2 and b - a < 1 - EPS and attacks
                   and spans[i - 1] >= 1 - EPS and spans[i + 1] >= 1 - EPS
                   and attacks[0] <= t0 + .005 and t1 <= performance_end + .005
                   and bisect_right(clock_edges, b + EPS) == bisect_left(clock_edges, a - EPS))
        if merged_next:
            kept_times.append(t1)
            merged_next = False
            continue
        if movable and abs(b - a - .5) < EPS:
            if any(abs(spans[i + 1] + .5 - n) < EPS for n in (2, 3, 4)) and (
                    bisect_left(clock_edges, cleaned[i + 2] - EPS) == bisect_right(clock_edges, b + EPS)):
                merged_next = True
                continue
            if any(abs(spans[i - 1] + .5 - n) < EPS for n in (2, 3, 4)) and (
                    bisect_left(clock_edges, a - EPS) == bisect_right(clock_edges, cleaned[i - 1] + EPS)):
                kept_times[-1] = t1
                continue
        last_held = bisect_right(held_starts, t1 - .005) - 1
        sounding = last_held >= 0 and held_ends[last_held] >= t0 + .005
        if movable and not sounding and bisect_right(attacks, t0 + .005) == bisect_left(attacks, t1 - .005):
            stops.append(dict(time=t0, duration=t1 - t0))
            continue
        if (movable and abs(b - a - .5) < EPS and len(kept_times) > 1
                and bisect_left(clock_edges, a - EPS) == bisect_right(clock_edges, cleaned[i - 1] + EPS)):
            kept_times.pop()
        kept_times.append(t1)
    meta["stops"] = sorted(stops, key=lambda s: s["time"])
    revised = BeatGrid(meta)
    for stop, beat in zip(meta["stops"], revised.stop_beats):
        stop["beat"] = beat
    bars = [round(revised.second_to_beat(t), 8) for t in kept_times]
    # Meter segment positions from the input refer to the original clock.
    if meta.get("meter_segments"):
        meta["meter_segments"] = [[revised.second_to_beat(grid.beat_to_second(b)), n, d]
                                  for b, n, d in meta["meter_segments"]]
    meta["bar_beats"] = bars
    meta.pop("measure_multipliers", None)
    ambiguous = []
    segments = _meter_segments(bars, BeatGrid(meta), triggers, ambiguous)
    meta["ambiguous_compound_bars"] = ambiguous
    meta.update(meter_segments=segments, numerator=segments[0][1], denominator=segments[0][2])
    return meta, None
