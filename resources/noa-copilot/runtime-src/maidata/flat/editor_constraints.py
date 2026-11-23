"""Incremental masks for editor generation. The user's committed prefix is replayed verbatim."""

from bisect import bisect_left
from copy import copy
from heapq import heappush, heappop
import math

from .codec import CHORD, FAMILIES, GRID32_TICKS, GROUP, NEW_PADS, note_tokens
from maidata.spine.codec import HOLD, SLIDE

PAD = {idx: pad for pad, idx in enumerate(NEW_PADS)}
OVERLAY = 6 / 180
EPS = 1e-9


class Physical:
    def __init__(self, grid, prefix, full=False):
        self.grid, self.full = grid, full
        self.ends = [-math.inf] * 33
        self.touch = [-math.inf] * 33
        self.events = []
        self.busy = [0] * 11  # slides, touch holds, outer holds, guard counts by lane
        for tick, notes in prefix:
            self.commit(tick, notes)

    def fork(self):
        result = copy(self)
        result.ends, result.touch = self.ends.copy(), self.touch.copy()
        result.events, result.busy = self.events.copy(), self.busy.copy()
        return result

    def advance(self, sec):
        slot = math.floor(sec * 100 + 1e-7)
        while self.events and self.events[0][0] <= slot:
            _, index, delta = heappop(self.events)
            self.busy[index] += delta
        return self.busy

    def interval(self, index, start, stop):
        heappush(self.events, (start, index, 1))
        heappush(self.events, (stop, index, -1))

    def commit(self, tick, notes):
        sec = self.grid.second(tick)
        self.advance(sec)
        for pad, _, kind, _, end, _, branches in notes:
            sensor = pad if pad < 8 else pad - 8
            finish = self.grid.second(end) if kind == HOLD else sec
            self.ends[sensor] = max(self.ends[sensor], finish)
            if pad >= 8:
                self.touch[sensor] = max(self.touch[sensor], finish)
            if self.full:
                continue
            if kind == HOLD and finish > sec:
                head = math.floor(sec * 100 + 1e-7)
                stop = head + max(1, math.ceil((finish - sec) * 100 - 1e-9)) + 1
                self.interval(1 if pad >= 8 else 2, head, stop)
            for launch, end, _, _ in branches:
                begin = math.floor(self.grid.second(launch) * 100 + 0.5 + 1e-8)
                stop = math.floor(self.grid.second(end) * 100 + 0.5 + 1e-8)
                # At launch+1 slot the guard and moving hand overlap. Keep that
                # slot in the lane-aware guard so a same-lane press cancels it once.
                self.interval(3 + pad, begin - 5, begin + 2)
                self.interval(0, begin + 2, max(begin + 1, stop) + 2)

    def allowed(self, pad, sec, sensors, outer, touch, busy):
        sensor = pad if pad < 8 else pad - 8
        if sensors & (1 << sensor) or sec - self.ends[sensor] <= OVERLAY + EPS:
            return False
        if pad >= 8 and sec - self.touch[sensor] < 0.150 - EPS:
            return False
        outer |= (1 << pad) if pad < 8 else 0
        hands = outer.bit_count() + bool(touch or pad >= 8 or busy[1])
        if self.full:
            return hands <= 3
        guards = sum(
            max(0, count - bool(outer & (1 << lane)))
            for lane, count in enumerate(busy[3:])
        )
        return hands + guards + busy[0] + busy[2] <= 2

    def mask(self, g, choices):
        if g.phase == "time":
            # Inspect future slots on a scratch sweep; the sampled onset may be earlier.
            future = self.fork()
            out = []
            for idx in choices:
                family, value = FAMILIES[idx]
                if family != "MOMENT":
                    out.append(idx)
                    continue
                sec = self.grid.second(g.beat * 96 + GRID32_TICKS[value])
                busy = future.advance(sec)
                if any(
                    self.allowed(p, sec, 0, 0, False, busy)
                    for p in range(41)
                    if p < 8 or g.touch_tap_ok[p - 8] or g.touch_hold_ok[p - 8]
                ):
                    out.append(idx)
            return out
        if not any(i in PAD or i in (GROUP, CHORD) for i in choices):
            return choices
        sec = self.grid.second(g.tick)
        busy = self.advance(sec)
        sensors = outer = 0
        touch = False
        for note in g.notes:
            pad = note[0]
            sensors |= 1 << (pad if pad < 8 else pad - 8)
            if pad < 8:
                outer |= 1 << pad
            else:
                touch = True

        def valid(p):
            return self.allowed(p, sec, sensors, outer, touch, busy)

        can_add = any(valid(PAD[i]) for i in g.legal_pads())
        return [
            i
            for i in choices
            if (i not in PAD or valid(PAD[i])) and (i not in (GROUP, CHORD) or can_add)
        ]


class Skeleton:
    """Slots are consumed by kind, allowing the model to order pads within a chord."""

    def __init__(self, rows, *, reserve_slides=False):
        self.rows = rows
        self.ticks = sorted(rows)
        # Once a selection reserves Slides for starred slots, stepping past
        # its last star must not change the meaning of the remaining x slots.
        self.reserve_slides = reserve_slides or any(
            slot.get("branches", 0) > 0 for slots in rows.values() for slot in slots
        )

    def mask(self, g, choices, physical=None):
        def generated(ids):
            return physical.mask(g, ids) if physical is not None else ids

        if g.phase in ("first", "boundary"):
            return choices
        if g.phase == "time":
            at = bisect_left(self.ticks, max(g.beat * 96, g.last_onset + 1))
            target = self.ticks[at] if at < len(self.ticks) else None
            choices = [
                i
                for i in choices
                if (
                    g.beat * 96 + GRID32_TICKS[FAMILIES[i][1]] == target
                    if FAMILIES[i][0] == "MOMENT"
                    else target is None or target >= (g.beat + 1) * 96
                )
            ]
            if any("tokens" in slot for slot in self.rows.get(target, ())):
                return choices
            return generated(choices)
        slots = self.rows.get(g.tick, ())
        remaining = list(slots)
        # Replay fixed items first, then consume the selected placeholders.
        active = g.notes[-1] if g.phase != "start_note" and g.notes else None
        finished = g.notes[:-1] if active is not None else g.notes
        for note in finished:
            slot = self.match(note, remaining)
            if slot is not None:
                remaining.remove(slot)
        locked = [s for s in remaining if "tokens" in s]
        free = [s["branches"] for s in remaining if "tokens" not in s]
        wildcard = not self.reserve_slides and 0 in free
        if g.phase == "start_note":
            if locked:
                return [i for i in choices if i == locked[0]["tokens"][0]]
            return generated(
                [i for i in choices if i in PAD and (PAD[i] < 8 or 0 in free)]
            )
        if locked:
            seq = locked[0]["tokens"]
            offset = g.index - g.note_start + 1
            if offset < len(seq):
                return [i for i in choices if i == seq[offset]]
        is_slide = active is not None and active[2] == SLIDE
        branches = len(active[6]) if is_slide else 0
        done = bool(locked) or wildcard or branches in free
        more = len(remaining) > 1
        out = []
        for i in choices:
            family = FAMILIES[i][0]
            if family in ("CHORD", "EOM"):
                if done and (family == "CHORD") == more:
                    out.append(i)
            elif locked or family in ("PSEUDO_EACH", "TOUCH_PAD"):
                continue
            elif family == "SLIDE_BRANCH":
                if wildcard or any(k > branches for k in free):
                    out.append(i)
            elif family == "SLIDE_HEAD":
                if wildcard or any(k > 0 for k in free):
                    out.append(i)
            elif family != "HOLD" or 0 in free:
                out.append(i)
        # A separator leading to another handwritten note is part of its replay.
        return out if len(locked) > 1 else generated(out)

    def match(self, note, slots):
        for slot in slots:
            if "tokens" in slot and slot["tokens"] == list(
                note_tokens(slot["tick"], note)
            ):
                return slot
        return next(
            (s for s in slots if "tokens" not in s
             and (not self.reserve_slides or s["branches"] == len(note[6]))),
            None,
        )
