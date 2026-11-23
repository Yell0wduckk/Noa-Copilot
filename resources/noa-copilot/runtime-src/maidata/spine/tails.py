"""Variable-length tail fields and route occupancy in physical sensor coordinates."""

from functools import lru_cache
from importlib.resources import files
import json
import math
import numpy as np
import torch
from maidata.chart.routes import route_shape_end, mirror_route
from maidata.multitask.grid import MIN_HOLD_TICKS, hold_end_tick
from .codec import HOLD, SLIDE, PADS, PAD_ID, PAD_MAP

# A timestamp addresses one of the next 16 beats from the onset beat.
# PAGE moves this local planning window forward by 16 without advancing the chart.
PAGE = 16
REM = PAGE + 1
ROUTE = REM + 96
ROUTE_END = ROUTE + 71
HEAD = ROUTE_END + 1
TRACK = HEAD + 3
MORE = TRACK + 2
END = MORE + 1
WAIT = END + 1
START = WAIT + 1
FIELD_COUNT = START + 1


def endpoint_tokens(onset, tick):
    beat, offset = divmod(int(tick), 96)
    return [PAGE] * ((beat - int(onset) // 96) // 16) + [beat % 16, REM + offset]


def _timestamp(onset, ranges):
    first = onset // 96
    size = 2
    while True:
        beats = sorted(
            {
                b % 16
                for lo, hi in ranges
                for b in range(max(first, lo // 96), min(first + 15, hi // 96) + 1)
            }
        )
        if ranges[-1][1] // 96 >= first + 16:
            beats.append(PAGE)
        choice = yield beats, None
        if choice != PAGE:
            break
        first += 16
        size += 1
    target_beat = first + (choice - first) % 16
    base = target_beat * 96
    phases = [
        REM + r
        for lo, hi in ranges
        for r in range(max(0, lo - base), min(95, hi - base) + 1)
    ]
    offset = yield phases, target_beat
    return base + offset - REM, size


def grammar(kind, onset, limit, max_fields=None):
    """Absolute chart ticks with inclusive bounds, shared by packing and sampling."""
    onset, limit = int(onset), int(limit)
    if kind == HOLD:
        ranges = [(onset, onset)]
        if onset + MIN_HOLD_TICKS <= limit:
            ranges.append((onset + MIN_HOLD_TICKS, limit))
        yield from _timestamp(onset, ranges)
        return
    yield range(HEAD, HEAD + 3), None
    used = 1
    while True:
        route = yield range(ROUTE, ROUTE_END), None
        used += 1
        while route != ROUTE_END:
            choices = (
                [ROUTE_END]
                if route == ROUTE + 70 or (max_fields and used >= max_fields)
                else [*range(ROUTE, ROUTE + 70), ROUTE_END]
            )
            route = yield choices, None
            used += 1
        custom = [
            (lo, hi)
            for lo, hi in ((onset, min(onset + 95, limit - 1)), (onset + 97, limit - 1))
            if lo <= hi
        ]
        default = (
            _timestamp(onset, [(onset + 97, limit)]) if onset + 97 <= limit else None
        )
        first = list(next(default)[0]) if default is not None else []
        if custom and (not first or not max_fields or used + 7 <= max_fields):
            first.append(WAIT)
        choice = yield first, None
        if choice == WAIT:
            launch, launch_fields = yield from _timestamp(onset, custom)
            launch_fields += 1
            _, end_fields = yield from _timestamp(onset, [(launch + 1, limit)])
        else:
            launch_fields = 0
            step = default.send(choice)
            while True:
                token = yield step
                try:
                    step = default.send(token)
                except StopIteration as done:
                    _, end_fields = done.value
                    break
        yield (TRACK, TRACK + 1), None
        used += launch_fields + end_fields + 1
        again = yield (
            ((END,) if max_fields and used >= max_fields else (MORE, END)),
            None,
        )
        used += 1
        if again == END:
            return


def field_plan(kind, onset, limit, tokens):
    state = grammar(kind, onset, limit)
    choices, beat = next(state)
    allowed = np.zeros((len(tokens), FIELD_COUNT), dtype=bool)
    beats = np.full(len(tokens), -1, np.int32)
    for i, token in enumerate(tokens):
        if beat is not None:
            beats[i] = beat
        allowed[i, list(choices)] = True
        try:
            choices, beat = state.send(int(token))
        except StopIteration:
            break
    return allowed, beats


def tail_fields(tick, note):
    pad, group, kind, flags, end, head, branches = note
    if kind == HOLD:
        return endpoint_tokens(tick, hold_end_tick(tick, end))
    out = [HEAD + head]
    for j, (launch, stop, brk, routes) in enumerate(branches):
        out.extend(ROUTE + r for r in routes)
        out.append(ROUTE_END)
        if launch != tick + 96:
            out.append(WAIT)
            out.extend(endpoint_tokens(tick, launch))
        out.extend(endpoint_tokens(tick, stop))
        out.extend((TRACK + brk, MORE if j + 1 < len(branches) else END))
    return out


def read_tail(tick, pad, group, kind, tokens):
    stream = map(int, tokens)

    def number(first_token=None):
        first = int(tick) // 96
        phase = next(stream) if first_token is None else first_token
        while phase == PAGE:
            first += 16
            phase = next(stream)
        return (first + (phase - first) % 16) * 96 + next(stream) - REM

    if kind == HOLD:
        return (pad, group, HOLD, 0, hold_end_tick(tick, number()), 0, [])
    head = next(stream) - HEAD
    branches = []
    while True:
        routes = []
        for v in stream:
            if v == ROUTE_END:
                break
            routes.append(v - ROUTE)
        first = next(stream)
        launch = number() if first == WAIT else tick + 96
        stop = number(None if first == WAIT else first)
        branches.append((launch, stop, next(stream) - TRACK, routes))
        if next(stream) == END:
            break
    return (pad, group, SLIDE, 0, 0, head, branches)


@lru_cache(maxsize=1)
def geometry():
    return json.loads(
        files("maidata.multitask")
        .joinpath("assets/spine-route-contacts.json")
        .read_text()
    )


HAZ_DIM = 33 * 4
HAZ_OFFSETS = (0, 24, 48, 72, 96, 144, 192, 288, 384)
PAD_FEEDBACK = 12
FEEDBACK_DIM = 41 * PAD_FEEDBACK
DETAIL_DIM = 14


def detail(tick, note):
    # Current surface flags are predicted AFTER these fields.
    _, _, kind, _, end, head, branches = note
    x = np.zeros(DETAIL_DIM, np.float32)
    if kind == HOLD:
        x[:2] = (1, math.log1p(end - tick) / 6)
    elif kind == SLIDE:
        x[2:5] = (1, head / 2, math.log1p(len(branches)))
        measures = [
            (
                math.log1p(a - tick) / 6,
                math.log1p(z - a) / 6,
                brk,
                math.log1p(len(routes)),
            )
            for a, z, brk, routes in branches
        ]
        stats = np.asarray(measures)
        x[5:9] = stats.mean(0)
        x[9:12] = stats.max(0)[[0, 1, 3]]
        # Paired route summaries exchange under physical mirror.
        shapes = []
        for _, _, _, routes in branches:
            start = note[0]
            shape = np.zeros(2)
            for rid in routes:
                key = "A" + PADS[start]
                shape = (
                    0.5 * shape + np.array([rid + 1, mirror_route(key, rid) + 1]) / 72
                )
                _, start = route_shape_end(key, rid)
            shapes.append(shape)
        x[12:] = np.mean(shapes, axis=0)
    return x


def end_pads(note):
    pad, _, kind, _, _, _, branches = note
    out = np.zeros(41, np.uint8)
    if kind == HOLD:
        out[pad] = 1
    for _, _, _, routes in branches:
        target = pad
        for rid in routes:
            _, target = route_shape_end("A" + PADS[target], rid)
        out[target] = 1
    return out


def release_tick(note):
    return max((b[1] for b in note[6]), default=note[4])


def commit_pad(summary, pad, kind, flags):
    summary[pad, 0] = 1
    if kind:
        summary[pad, kind] = 1
    for bit in range(3):
        if flags & (1 << bit):
            summary[pad, 3 + bit] = 1


def feedback(first, tick, note, out):
    pad, group, kind, flags, *_ = note
    if not (kind or flags or group):
        return
    d = detail(tick, note)
    values = [
        kind == HOLD,
        kind == SLIDE,
        flags & 1,
        (flags >> 1) & 1,
        (flags >> 2) & 1,
        group / (group + 1),
        d[5],
        d[3],
        d[7],
        d[1] + d[6],
        d[4],
        1,
    ]
    view = out.reshape(41, PAD_FEEDBACK)[pad]
    np.maximum(view, values, out=view)


@lru_cache(maxsize=8 * 71)
def physical_segment(start, rid):
    # Use one measured template for both reflected spellings, including the
    # exact floating-point contact boundaries used by hard collision checks.
    reflected = start > PAD_MAP[8][start]
    anchor = PAD_MAP[8][start] if reflected else start
    template = mirror_route("A" + PADS[start], rid) if reflected else rid
    length, contacts = geometry()[template]
    pads = PAD_MAP[anchor]

    def sensor(region):
        p = pads[PAD_ID[region]]
        return (PAD_MAP[8][p] if reflected else p) - 8

    _, end = route_shape_end("A" + PADS[start], rid)
    return length, [(sensor(region), lo, hi) for region, lo, hi in contacts], end


@lru_cache(maxsize=8)
def first_route_contacts(pad):
    return tuple(
        sum(
            1 << sensor
            for sensor in {p for p, lo, hi in physical_segment(pad, r)[1] if lo < hi}
        )
        for r in range(71)
    )


class Hazard:
    """Only committed notes add intervals. Querying never reveals a future label."""

    def __init__(self, device=None):
        self.intervals = []
        self.latest_release = -1
        self.device = device
        if device is not None:
            self.cached_query = None
            self.offsets = torch.tensor(HAZ_OFFSETS, device=device)
            self.empty = torch.zeros(
                (len(HAZ_OFFSETS), 4 * 33), dtype=torch.uint8, device=device
            )
            self.device_intervals = None

    def refresh(self):
        if self.device is not None:
            self.cached_query = None
            self.device_intervals = None
            if self.intervals:
                times = np.array(self.intervals, dtype=np.float64)
                self.device_intervals = (
                    torch.as_tensor(times[:, :2].T.copy(), device=self.device),
                    torch.as_tensor(
                        (times[:, 2] * 33 + times[:, 3]).astype(np.int64),
                        device=self.device,
                    ),
                )

    def commit(self, tick, note):
        pad, _, kind, _, end, _, branches = note
        if self.device is not None:
            self.intervals = [item for item in self.intervals if item[1] > tick]
        sensor = pad if pad < 8 else pad - 8
        self.latest_release = max(self.latest_release, release_tick(note))
        if kind == HOLD:
            self.intervals.append((tick, end, 3, sensor))
        if kind != SLIDE:
            self.refresh()
            return
        for launch, z, _, routes in branches:
            start = pad
            segments = []
            for rid in routes:
                length, contacts, start = physical_segment(start, rid)
                segments.append((length, contacts))
            total = sum(length for length, _ in segments)
            left = launch
            for length, contacts in segments:
                duration = (z - launch) * length / total
                for region_id, lo, hi in contacts:
                    begin = left + duration * lo
                    self.intervals.append((tick, begin, 0, region_id))
                    self.intervals.append((begin, left + duration * hi, 1, region_id))
                left += duration
            self.intervals.append((z, z + 24, 2, start))

        self.refresh()

    def query(self, tick):
        if self.device is not None:
            if self.device_intervals is None:
                return self.empty.flatten()
            if self.cached_query is not None and self.cached_query[0] == tick:
                return self.cached_query[1]
            (a, b), idx = self.device_intervals
            t = (self.offsets + tick)[:, None]
            hit = ((a <= t) & (t < b)).to(torch.uint8)
            value = self.empty.scatter_reduce(
                1, idx.expand_as(hit), hit, "amax"
            ).flatten()
            self.cached_query = tick, value
            return value

        self.intervals = [item for item in self.intervals if item[1] > tick]
        result = np.zeros((len(HAZ_OFFSETS), 4, 33), np.uint8)
        for a, b, channel, pad in self.intervals:
            for i, delta in enumerate(HAZ_OFFSETS):
                if a <= tick + delta < b:
                    result[i, channel, pad] = 1
        return result.reshape(-1)

    def at_sensor(self, tick, sensor):
        if self.device is not None:
            return self.query(tick).view(-1, 4, 33)[0, :, sensor]
        out = np.zeros(4, np.uint8)
        for a, b, ch, p in self.intervals:
            if p == sensor and a <= tick < b:
                out[ch] = 1
        return out

    def slide_room(self, tick, pad, limit):
        if pad >= 8 or tick >= limit:
            return False
        if self.latest_release < limit:
            return True
        # An occupied sensor covering the entire remaining horizon excludes every
        # route that traverses it. If this rules out all 71 initial routes, a chain
        # or multi-branch slide cannot rescue the repeat. Partial overlaps still go
        # through the actual per-route collision test after sampling.
        blocked = 0
        for a, b, ch, p in self.intervals:
            if ch in (1, 3) and a <= tick and b >= limit:
                blocked |= 1 << p
        return not blocked or any(
            (mask & blocked) == 0 for mask in first_route_contacts(pad)
        )

    def conflicts(self, tick, note, future):
        candidate = Hazard()
        candidate.commit(tick, note)
        # Only occupied intervals are hard conflicts. The anticipatory channel
        # is a soft training feature, not a reservation of the whole route.
        occupied = [v for v in candidate.intervals if v[2] in (1, 3)]
        for start, stop, _, sensor in occupied:
            if any(
                ch in (1, 3) and p == sensor and max(start, a) < min(stop, b)
                for a, b, ch, p in self.intervals
            ):
                return True
            if any(
                start <= t < stop and (p if p < 8 else p - 8) == sensor
                for t, p in future
            ):
                return True
        return False
