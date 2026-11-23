"""Accepted-prefix grammar shared by packing, teacher forcing and streaming."""

from collections import Counter
from itertools import groupby
from zlib import crc32

import numpy as np

from maidata.chart.routes import catalog
from maidata.chart.types import TOUCH_PADS
from maidata.multitask.grammar import bucket
from maidata.version_mask import (
    VERSION_ORDER,
    VersionSupervisionProfile,
    profile_for_chart,
)
from .geometry import (
    ROUTE_ARC,
    ROUTE_RATIOS,
    ROUTE_STEP_LEN,
    MAX_SLIDE_STEPS,
    active_hand_clusters,
)
from maidata.spine.codec import HOLD, PLAIN, SLIDE, PADS, PAD_MAP, mirror_notes
from maidata.spine.packing import beat_actions, slot_order
from maidata.spine.tails import endpoint_tokens, REM

GRID32_TICKS = tuple(12 * k + r for k in range(8) for r in (0, 4, 6, 8))
TICK_TO_GRID32 = {t: i for i, t in enumerate(GRID32_TICKS)}
BEAT_RHYTHM_MASKS = (
    1, 65537, 0, 65536, 16843009, 16842753, 16777217, 65793,
    2097153, 2099201, 65792, 16777473, 16842752, 286331153, 256, 136382497,
    16777472, 2048, 2097152, 134285313, 16843008, 257, 67617, 136380417,
    134283265, 67585, 134217729, 16777216, 136380416, 4369, 17, 69905,
    273, 2049, 67584, 286326785, 134219777, 2081, 286326784, 134219776,
    134285312, 131074, 33620481, 690563369, 2099200, 2, 17891329, 134217728,
    131072, 16843025, 2164769, 16844833, 33, 136314881, 297, 17891328,
    33619969, 3, 5, 2686976, 66049, 41, 1048576, 65553,
    1048577, 33554433, 2162688, 65568, 16847105, 9, 16, 17895697,
    33554944, 285212672, 687865857, 134283264, 286327041, 66048, 33554434, 4368,
    687866153, 2164737, 33554432, 33554945, 285212945, 85, 4096, 134283296,
    67881, 16781313, 33685506, 2345, 17891601, 272, 512, 65809,
    268435456, 33619968, 134217760, 136314880, 286265617, 2097408, 16847121, 136382464,
    286261249, 136380673, 32, 341, 69888, 76073, 285212673, 2147516416,
    2162689, 134225921, 136382465, 1114112, 1118464, 285217041, 8, 1118481,
    16779264, 17891585, 19464193, 690552832, 2155905152, 131584, 196608, 1114113,
)
BEAT_RHYTHM_OTHER = len(BEAT_RHYTHM_MASKS)  # 128
BEAT_RHYTHM_CLASSES = BEAT_RHYTHM_OTHER + 1  # 129
MASK_TO_BEAT_RHYTHM = {mask: i for i, mask in enumerate(BEAT_RHYTHM_MASKS)}

MOD12_SNAP = (0, 0, 0, 4, 4, 4, 6, 8, 8, 8, 12, 12)
WIDTHS = dict(
    BOUNDARY=1,
    BEAT=16,
    MOMENT=32,
    EOM=1,
    CHORD=1,
    PSEUDO_EACH=1,
    KEY_POS=8,
    TOUCH_PAD=33,
    KEY_BREAK=1,
    KEY_EX=1,
    TOUCH_FIREWORK=1,
    HOLD=1,
    HOLD_END_BEAT=16,
    HOLD_END_OFFSET=32,
    SLIDE_HEAD=3,
    SLIDE_BRANCH=1,
    ROUTE=71,
    TRACK_BREAK=1,
    SLIDE_WAIT=1,
    SLIDE_WAIT_BEAT=16,
    SLIDE_WAIT_OFFSET=32,
    SLIDE_END=1,
    SLIDE_END_BEAT=16,
    SLIDE_END_OFFSET=32,
    LONG_SPAN=1,
)
ROUTE_END_DELTA = tuple(row["endpoint_index"] - 1 for row in catalog())
_GRID = frozenset((0, 4, 6, 8))
_CEIL = tuple(next(d for d in range(4) if (i + d) % 12 in _GRID) for i in range(12))
_FLOOR = tuple(next(d for d in range(4) if (i - d) % 12 in _GRID) for i in range(12))


def ceil_grid(t):
    return t + _CEIL[t % 12]


def floor_grid(t):
    return t - _FLOOR[t % 12]


def snap_tick_grid32(tick, lo=0, hi=None):
    q, r = divmod(tick, 12)
    if hi is None:
        return max(ceil_grid(lo), q * 12 + MOD12_SNAP[r])
    # Bounded nearest point: on a tie choose the earlier legal tick.
    return min(
        max(
            min((floor_grid(tick), ceil_grid(tick)), key=lambda v: abs(v - tick)),
            ceil_grid(lo),
        ),
        floor_grid(hi),
    )


def snap_moments_grid32(moments, end_tick=None):
    """Snap durations relative to the new head; keep tails beyond the audio horizon."""
    merged = []
    for t, notes in sorted(moments, key=lambda row: row[0]):
        if not notes:
            continue
        onset = snap_tick_grid32(t)
        if not merged or merged[-1][0] != onset:
            merged.append((onset, []))
        dest = merged[-1][1]
        first = max((n[1] for n in dest), default=-1) + 1
        groups = {g: first + i for i, g in enumerate(sorted({n[1] for n in notes}))}
        for pad, group, kind, flags, end, head, branches in notes:
            tail = 0
            if kind == HOLD:
                tail = (
                    onset if end <= t else snap_tick_grid32(onset + end - t, onset + 6)
                )
            snapped = []
            for launch, finish, brk, routes in branches:
                start = (
                    onset + 96
                    if launch == t + 96
                    else snap_tick_grid32(onset + max(0, launch - t), onset)
                )
                stop = snap_tick_grid32(start + max(1, finish - launch), start + 1)
                snapped.append((start, stop, brk, tuple(routes)))
            dest.append([pad, groups[group], kind, flags, tail, head, snapped])
    return merged


def field_horizon(moments, end):
    limit = max(0, end)
    for t, notes in moments:
        limit = max(limit, t)
        for n in notes:
            limit = max(limit, n[4], *(b[1] for b in n[6]))
    return ceil_grid(limit)


FAMILIES = [(family, i) for family, width in WIDTHS.items() for i in range(width)]
FAMILY_OFFSET = {}
FAMILY_IDS = {}
for index, (family, value) in enumerate(FAMILIES):
    if value == 0:
        FAMILY_OFFSET[family] = index
        FAMILY_IDS[family] = tuple(range(index, index + WIDTHS[family]))
SIZE = len(FAMILIES)


def token(family, value=0):
    return FAMILY_OFFSET[family] + value


def values(family, choices=None):
    ids = FAMILY_IDS[family]
    return list(ids) if choices is None else [ids[i] for i in choices]


BOUNDARY, EOM, CHORD, GROUP, PAGE_ID = map(
    token, ("BOUNDARY", "EOM", "CHORD", "PSEUDO_EACH", "LONG_SPAN")
)
TIMING_INPUTS = frozenset([BOUNDARY, EOM, *values("BEAT")])
NEW_PADS = tuple(values("KEY_POS") + values("TOUCH_PAD"))
_TOUCH_TO_VERSION_IDX = tuple(TOUCH_PADS.index(pad) for pad in PADS[8:])


def resolve_version_profile(meta):
    version = meta.get("version")
    cabinet = meta.get("cabinet")
    song = meta.get("song_id")
    if not version:
        if cabinet not in ("SD", "DX"):
            return None
        version = "maimai FiNALE" if cabinet == "SD" else VERSION_ORDER[-1]
        song = 0 if cabinet == "SD" else 10000
    elif song in (None, ""):
        # Custom editor charts can specify a release without an official song ID.
        dx = (
            cabinet == "DX"
            if cabinet in ("SD", "DX")
            else version.startswith("maimai DX")
        )
        song = 10000 if dx else 0
    return profile_for_chart(song, meta.get("difficulty"), version)


def timestamp(family, onset, end):
    fields = endpoint_tokens(onset, end)
    return [
        *[PAGE_ID] * (len(fields) - 2),
        token(family + "_BEAT", fields[-2]),
        token(family + "_OFFSET", TICK_TO_GRID32[fields[-1] - REM]),
    ]


def branch_key(branch):
    launch, end, brk, routes = branch
    return launch, end, brk, tuple(routes)


def branch_order(
    branches, song, tick, group, pad, mirrored=False, *, seed=None, epoch=0
):
    if len(branches) <= 1:
        return branches
    base = (
        mirror_notes([(pad, group, SLIDE, 0, 0, 0, branches)])[0][6]
        if mirrored
        else branches
    )
    order = sorted(range(len(branches)), key=lambda i: branch_key(base[i]))
    if seed is not None:
        song = str(song)
        song_seed = int(song) if song.isdecimal() else crc32(song.encode("utf8"))
        base_pad = PAD_MAP[8][pad] if mirrored else pad
        np.random.default_rng([seed, epoch, song_seed, tick, group, base_pad]).shuffle(
            order
        )
    return [branches[i] for i in order]


def note_tokens(tick, note, song=0, mirrored=False, *, seed=None, epoch=0):
    pad, group, kind, flags, end, head, branches = note
    yield NEW_PADS[pad]
    if flags & 1:
        yield token("KEY_BREAK")
    if flags & 2:
        yield token("KEY_EX")
    if flags & 4:
        yield token("TOUCH_FIREWORK")
    if kind == HOLD:
        yield token("HOLD")
        yield from timestamp("HOLD_END", tick, end)
    if kind == SLIDE:
        yield token("SLIDE_HEAD", head)
        branches = branch_order(
            branches, song, tick, group, pad, mirrored, seed=seed, epoch=epoch
        )
        for b, (launch, finish, brk, routes) in enumerate(branches):
            if b:
                yield token("SLIDE_BRANCH")
            for index, rid in enumerate(routes):
                yield token("ROUTE", rid)
                if index == 0 and brk:
                    yield token("TRACK_BREAK")
            if launch != tick + 96:
                yield token("SLIDE_WAIT")
                yield from timestamp("SLIDE_WAIT", tick, launch)
            else:
                yield token("SLIDE_END")
            yield from timestamp("SLIDE_END", tick, finish)


def subgroup_items(members, song, tick, group, mirrored, seed, epoch):
    """In shuffled pad order, emit ordinary touches together at their first pad."""
    by_pad = {}
    for note in members:
        by_pad.setdefault(note[0], []).append(note)
    order = slot_order(
        by_pad.keys(), song, tick, group, mirrored, seed=seed, epoch=epoch
    )

    def same_pad_key(n):
        branches = mirror_notes([n])[0][6] if mirrored else n[6]
        return (*n[2:6], sorted(map(branch_key, branches)))

    touches = [n for p in order if p >= 8 for n in by_pad[p] if n[2] == PLAIN]
    for pad in order:
        notes = by_pad[pad]
        if len(notes) > 1:
            notes.sort(key=same_pad_key)
        for n in notes:
            if pad >= 8 and n[2] == PLAIN:
                if touches:
                    yield touches
                    touches = []
            else:
                yield (n,)


def chart_tokens(moments, horizon=None, song=0, mirrored=False, *, seed=0, epoch=0):
    yield BOUNDARY
    beat = -1
    for action, notes in beat_actions(moments, horizon):
        if not notes:
            beat += 1
            yield token("BEAT", beat % 16)
            continue
        tick = beat * 96 + action
        yield token("MOMENT", TICK_TO_GRID32[action])
        for j, (group, members) in enumerate(
            groupby(sorted(notes, key=lambda n: n[1]), key=lambda n: n[1])
        ):
            if j:
                yield GROUP
            items = subgroup_items(members, song, tick, group, mirrored, seed, epoch)
            for index, item in enumerate(items):
                if index:
                    yield CHORD
                for note in item:
                    yield from note_tokens(
                        tick, note, song, mirrored, seed=seed, epoch=epoch
                    )
        yield EOM


class Grammar:
    def __init__(
        self,
        limit,
        *,
        onset_limit=None,
        max_groups=None,
        max_slot_events=None,
        max_fields=None,
        profile: VersionSupervisionProfile | None = None,
    ):
        if profile is None:
            self.touch_tap_ok = self.touch_hold_ok = (True,) * 33
            self.route_ids = FAMILY_IDS["ROUTE"]
        else:
            taps = profile.touch_pad33_mask.tolist()
            holds = profile.touch_hold_pad33_mask.tolist()
            self.touch_tap_ok = tuple(taps[i] for i in _TOUCH_TO_VERSION_IDX)
            self.touch_hold_ok = tuple(holds[i] for i in _TOUCH_TO_VERSION_IDX)
            self.route_ids = tuple(
                i
                for i, ok in zip(FAMILY_IDS["ROUTE"], profile.route71_mask.tolist())
                if ok
            )
        self.chain_ids = tuple(i for i in self.route_ids if i != token("ROUTE", 70))
        self.ex_ok = profile is None or bool(profile.note_flag_mask[0, 0])
        self.firework_ok = profile is None or bool(profile.note_flag_mask[2, 2])
        self.short_hold_ok = profile is None or bool(profile.note_flag_mask[0, 3])
        self.short_touch_hold_ok = profile is None or bool(profile.note_flag_mask[2, 4])
        self.hold_break_ok = profile is None or bool(profile.note_flag_mask[1, 1])
        self.slide_break_ok = profile is None or bool(profile.note_flag_mask[4, 1])
        self.chained_route_ok = profile is None or profile.chained_route_enabled
        self.track_break_ok = profile is None or profile.track_break_enabled
        self.multi_track_ok = profile is None or profile.multi_track_enabled
        self.limit = limit
        self.onset_limit = limit if onset_limit is None else onset_limit
        self.max_groups, self.max_slot_events, self.max_fields = (
            max_groups,
            max_slot_events,
            max_fields,
        )
        self.phase = "boundary"
        self.tick, self.beat, self.index, self.moment = 0, -1, -1, 0
        self.beat_mask = self.beat_checked_lo = 0
        self.last_onset = -96
        self.prev_evicted = False
        self.pages = self.group = 0
        self.notes, self.moments, self.active = [], [], []
        self.note = None
        self.last = BOUNDARY
        self.counts = Counter()
        self.non_slide_pads = set()
        self.open_touch_mask = self.open_firework_mask = 0
        self.plain_touch_closed = False
        self.last_onset_index = -1

    @property
    def query_tick(self):
        if self.phase == "offset":
            return self.target_beat * 96
        if self.phase == "end_beat":
            return max(self.launch, (self.beat + self.pages * 16) * 96)
        if self.phase in ("wait_beat", "hold_beat"):
            return (self.beat + self.pages * 16) * 96
        return self.tick

    @property
    def page_first(self):
        if self.phase in ("hold_beat", "end_beat", "wait_beat"):
            return self.beat + 16 * self.pages
        return -1

    def time_choices(self, start_tick=0):
        advance = token("BEAT", (self.beat + 1) % 16)
        if self.phase == "first":
            return [advance]
        end = min(self.beat * 96 + 95, self.onset_limit)
        base = self.beat * 96
        first = max(self.last_onset + 1, start_tick)
        return [
            i
            for i, off in zip(FAMILY_IDS["MOMENT"], GRID32_TICKS)
            if first <= base + off <= end
        ] + [advance]

    def ranges(self, family):
        t, z = self.tick, floor_grid(self.limit)
        if family == "HOLD_END":
            short = self.short_touch_hold_ok if self.pad >= 8 else self.short_hold_ok
            return ([(t, t)] if short else []) + [(t + 6, z)]
        if family == "SLIDE_WAIT":
            return [(t, min(t + 95, floor_grid(z - 1))), (t + 97, floor_grid(z - 1))]
        return [(self.launch + 1, z)]

    def times(self, families):
        first = self.beat + self.pages * 16
        out, more = [], False
        for family in families:
            ids = FAMILY_IDS[family + "_BEAT"] * 2
            previous = first - 1
            for lo, hi in self.ranges(family):
                lo, hi = ceil_grid(lo), floor_grid(hi)
                if lo > hi:
                    continue
                begin = max(first, lo // 96, previous + 1)
                last = min(first + 15, hi // 96)
                if begin <= last:
                    offset = begin % 16
                    out.extend(ids[offset : offset + last - begin + 1])
                    previous = last
                more |= hi // 96 >= first + 16
        return out + ([PAGE_ID] if more else [])

    def has_hold_time(self, touch):
        short = self.short_touch_hold_ok if touch else self.short_hold_ok
        return (self.tick if short else ceil_grid(self.tick + 6)) <= self.limit

    def legal_pads(self, closing_plain_pad=None):
        can_slide = ceil_grid(self.tick) < floor_grid(self.limit)
        must_hold = self.plain_touch_closed or (
            closing_plain_pad is not None and closing_plain_pad >= 8
        )
        has_touch_time = self.has_hold_time(True)
        return [
            idx
            for pad, idx in enumerate(NEW_PADS)
            if (self.max_slot_events is None or self.counts[pad] < self.max_slot_events)
            and (
                pad < 8
                or (not must_hold and self.touch_tap_ok[pad - 8])
                or (has_touch_time and self.touch_hold_ok[pad - 8])
            )
            and (
                (pad < 8 and can_slide)
                or (pad not in self.non_slide_pads and pad != closing_plain_pad)
            )
        ]

    def legal(self, *, start_tick=0):
        phase = self.phase
        if phase == "boundary":
            return [BOUNDARY]
        if phase in ("first", "time"):
            return self.time_choices(start_tick)
        if phase == "offset":
            base = self.target_beat * 96
            ids = FAMILY_IDS[self.endpoint + "_OFFSET"]
            ranges = self.ranges(self.endpoint)
            return [
                ids[i]
                for i, off in enumerate(GRID32_TICKS)
                if any(a <= base + off <= z for a, z in ranges)
            ]
        if phase in ("hold_beat", "wait_beat", "end_beat"):
            return self.times((self.endpoint,))
        if phase == "start_note":
            return self.legal_pads()
        if phase == "typed":
            return list(self.route_ids)
        room = self.max_fields is None or self.index - self.note_start < self.max_fields
        if phase == "route":
            out = []
            if self.track_break_ok and len(self.routes) == 1 and not self.track_break:
                out.append(token("TRACK_BREAK"))
            if self.chained_route_ok and room and self.routes[-1] != 70:
                out.extend(self.chain_ids)
            if ceil_grid(self.tick + 97) <= floor_grid(self.limit):
                out.append(token("SLIDE_END"))
            if self.times(("SLIDE_WAIT",)):
                out.append(token("SLIDE_WAIT"))
            return out
        opened = phase in ("key", "break", "ex", "firework")
        can_slide = ceil_grid(self.tick) < floor_grid(self.limit)
        out = []
        if opened:
            if self.pad < 8:
                if phase == "key" and (
                    self.pad not in self.non_slide_pads or self.slide_break_ok
                ):
                    out.append(token("KEY_BREAK"))
                if self.ex_ok and phase != "ex":
                    out.append(token("KEY_EX"))
                if can_slide and (self.slide_break_ok or not self.note[3] & 1):
                    out.extend(values("SLIDE_HEAD"))
            elif phase == "key" and self.firework_ok:
                out.append(token("TOUCH_FIREWORK"))
            if self.pad in self.non_slide_pads:
                return out
            if self.pad >= 8:
                if (
                    self.open_touch_mask.bit_count() == 1
                    and self.touch_hold_ok[self.pad - 8]
                    and self.has_hold_time(True)
                ):
                    out.append(token("HOLD"))
                if self.plain_touch_closed or not self.touch_tap_ok[self.pad - 8]:
                    return out
                if room:
                    out.extend(
                        NEW_PADS[p]
                        for p in range(8, len(NEW_PADS))
                        if p != self.pad
                        and self.touch_tap_ok[p - 8]
                        and p not in self.non_slide_pads
                        and (
                            self.max_slot_events is None
                            or self.counts[p] < self.max_slot_events
                        )
                    )
            elif (self.hold_break_ok or not self.note[3] & 1) and self.has_hold_time(
                False
            ):
                out.append(token("HOLD"))
        elif phase == "slide_done" and self.multi_track_ok and room and can_slide:
            out.append(token("SLIDE_BRANCH"))
        if self.legal_pads(self.pad if opened else None):
            out.append(CHORD)
        if self.max_groups is None or self.group + 1 < self.max_groups:
            out.append(GROUP)
        return out + [EOM]

    def commit_long(self, end):
        kind = self.note[2]
        if end > self.tick:
            self.active.append(
                (
                    end,
                    self.moment,
                    (*self.milestones, self.index) if kind == SLIDE else (self.index,),
                    kind,
                    self.tick,
                    self.pad,
                    self.segment_end if kind == SLIDE else self.pad,
                    self.launch if kind == SLIDE else self.tick,
                )
            )

    def active_state(self):
        # The v15 readout uses the prefix grammar's 32-component occupancy vector.
        state = [0] * 32
        slides, touches = [], []
        for entry in self.active:
            end, _, _, kind, start, pad, target, launch = entry
            if kind == HOLD and pad >= 8:
                touches.append(entry)
            if kind == SLIDE:
                state[11 + pad] = state[19 + target] = 1
                slides.append((end, launch))
            elif pad < 8 or pad == 24:  # outer button or central Touch
                state[27 if pad == 24 else pad] = bucket(end - self.tick)
            else:
                state[29] = max(state[29], bucket(self.tick - start))
                state[30] = max(state[30], bucket(end - self.tick))
        if touches:
            centers = {s for entry in touches if entry[5] == 24 for s in entry[2]}
            state[28] = min(
                3, sum(centers.isdisjoint(c) for c in active_hand_clusters(touches))
            )
        if slides:
            end, launch = min(slides)
            state[8:11] = (
                min(3, len(slides)),
                bucket(launch - self.tick),
                bucket(end - max(self.tick, launch)),
            )
        return state

    def evicts(self, next_idx):
        """Does the newly observed token replace the previous input slot?"""
        family = FAMILIES[self.last][0]
        nxt = FAMILIES[next_idx][0]
        if family in (
            "BOUNDARY",
            "HOLD_END_OFFSET",
            "SLIDE_END_OFFSET",
        ):
            return False
        if family in ("TOUCH_PAD", "TOUCH_FIREWORK"):
            return nxt in ("TOUCH_PAD", "TOUCH_FIREWORK", "HOLD")
        if family in ("KEY_POS", "KEY_EX", "KEY_BREAK"):
            return nxt in (
                "KEY_BREAK",
                "KEY_EX",
                "SLIDE_HEAD",
                "TOUCH_FIREWORK",
                "HOLD",
            )
        if family == "ROUTE" and nxt == "ROUTE":
            length = len(self.chunk_steps)
            return (
                length < 16
                and length - 1 + ROUTE_STEP_LEN[FAMILIES[next_idx][1]]
                <= MAX_SLIDE_STEPS
            )
        return True

    def advance(self, idx):
        family, value = FAMILIES[idx]
        old = self.phase
        if self.index >= 0:
            self.prev_evicted = self.evicts(idx)
        self.index += 1
        self.last = idx
        if family == "BOUNDARY":
            self.phase = "first"
        elif family in ("BEAT", "MOMENT"):
            if family == "BEAT":
                self.beat_mask = self.beat_checked_lo = 0
                self.beat += 1
                self.tick = self.beat * 96
                self.phase = "time"
            else:
                self.beat_mask |= 1 << value
                self.beat_checked_lo = value + 1
                self.tick = self.beat * 96 + GRID32_TICKS[value]
                self.last_onset_index = self.index
                self.last_onset = self.tick
                self.phase, self.group, self.notes = "start_note", 0, []
                self.counts.clear()
                self.non_slide_pads.clear()
                self.open_touch_mask = self.open_firework_mask = 0
                self.plain_touch_closed = False
            self.active = [row for row in self.active if row[0] > self.tick]
            self.moment = self.index
        elif family in ("CHORD", "EOM", "PSEUDO_EACH"):
            if self.note[2] == PLAIN:
                self.non_slide_pads.add(self.pad)
                if self.pad >= 8:
                    self.plain_touch_closed = True
            self.open_touch_mask = self.open_firework_mask = 0
            self.note = None
            if family == "EOM":
                self.moments.append((self.tick, self.notes))
                self.phase = "time"
            else:
                if family == "PSEUDO_EACH":
                    self.plain_touch_closed = False
                    self.group += 1
                    self.counts.clear()
                    self.non_slide_pads.clear()
                self.phase = "start_note"
        elif family in ("KEY_POS", "TOUCH_PAD"):
            if family == "TOUCH_PAD" and old in ("key", "firework"):
                self.non_slide_pads.add(self.pad)
            else:
                self.note_start = self.index
            if family == "TOUCH_PAD":
                self.open_touch_mask |= 1 << value
            self.pad = value + (8 if family == "TOUCH_PAD" else 0)
            self.counts[self.pad] += 1
            self.note = [self.pad, self.group, PLAIN, 0, 0, 0, []]
            self.notes.append(self.note)
            self.phase = "key"
        elif family in ("KEY_BREAK", "KEY_EX", "TOUCH_FIREWORK"):
            bit, self.phase = {
                "KEY_BREAK": (1, "break"),
                "KEY_EX": (2, "ex"),
                "TOUCH_FIREWORK": (4, "firework"),
            }[family]
            self.note[3] |= bit
            if family == "TOUCH_FIREWORK":
                self.open_firework_mask |= 1 << (self.pad - 8)
        elif family == "HOLD":
            self.open_touch_mask = self.open_firework_mask = 0
            self.phase, self.endpoint = "hold_beat", "HOLD_END"
        elif family in ("SLIDE_HEAD", "SLIDE_BRANCH"):
            if family == "SLIDE_HEAD":
                self.note[5] = value
            self.phase = "typed"
        elif family == "ROUTE":
            if old == "route":
                self.segment_start = self.segment_end
                if not self.prev_evicted:
                    self.milestones.append(self.index - 1)
                    self.arc_before_chunk = self.arc
                    self.chunk_steps = []
            else:
                self.segment_start, self.routes = self.pad, []
                self.branch_idx = len(self.note[6])
                self.launch, self.track_break = self.tick + 96, False
                self.explicit_wait = False
                self.arc = self.arc_before_chunk = 0.0
                self.chunk_steps, self.milestones = [], []
            self.routes.append(value)
            offset = (self.segment_start * 71 + value) * 12
            first = int(bool(self.chunk_steps))
            if first:
                ref, arc, _ = self.chunk_steps[-1]
                self.chunk_steps[-1] = (ref, arc, True)
            self.chunk_steps.extend(
                (
                    offset + j,
                    self.arc + ratio * ROUTE_ARC[value],
                    j == 0 and bool(self.milestones),
                )
                for j, ratio in enumerate(ROUTE_RATIOS[value])
                if j >= first
            )
            self.arc += ROUTE_ARC[value]
            self.segment_end = (self.segment_start + ROUTE_END_DELTA[value]) % 8
            self.phase = "route"
        elif family == "TRACK_BREAK":
            self.track_break = True
        elif family == "SLIDE_WAIT":
            self.phase, self.endpoint = "wait_beat", "SLIDE_WAIT"
            self.explicit_wait = True
        elif family == "SLIDE_END":
            self.phase, self.endpoint = "end_beat", "SLIDE_END"
            self.launch = self.tick + 96
        elif family == "LONG_SPAN":
            self.pages += 1
        elif family.endswith("_BEAT"):
            self.target_beat = self.beat + (value - self.beat) % 16 + self.pages * 16
            self.endpoint, self.pages, self.phase = family[:-5], 0, "offset"
        elif family.endswith("_OFFSET"):
            end = self.target_beat * 96 + GRID32_TICKS[value]
            if self.endpoint == "HOLD_END":
                self.non_slide_pads.add(self.pad)
                self.note[2], self.note[4], self.phase = HOLD, end, "done"
                self.commit_long(end)
            elif self.endpoint == "SLIDE_WAIT":
                self.launch, self.phase, self.endpoint = end, "end_beat", "SLIDE_END"
                self.explicit_wait = True
            else:
                branch = (self.launch, end, int(self.track_break), tuple(self.routes))
                self.note[2], self.phase = SLIDE, "slide_done"
                self.note[6].append(branch)
                self.commit_long(end)
