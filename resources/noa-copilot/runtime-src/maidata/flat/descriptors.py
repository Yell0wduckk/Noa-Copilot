"""Compact, prefix-only snapshots. Rows are grouped before host-to-device transfer."""

import numpy as np
import torch

from .codec import FAMILIES
from .geometry import MAX_SLIDE_STEPS

SYNC = ("BOUNDARY", "BEAT", "MOMENT", "CHORD", "PSEUDO_EACH", "EOM")
KEY = ("KEY_POS", "KEY_BREAK", "KEY_EX", "SLIDE_HEAD", "SLIDE_BRANCH")
TOUCH = ("TOUCH_PAD", "TOUCH_FIREWORK")
HOLD_PHASE = {"HOLD": 0, "LONG_SPAN": 1, "HOLD_END_BEAT": 2, "HOLD_END_OFFSET": 3}
# Slide integer columns precede its step references and physical arc lengths.
WIDTH = (3, 5, 7, 8, 15)


def active_spans(commits, ticks):
    if not commits:
        return np.empty((0, 3), np.int32)
    starts, ends, slots = np.asarray(commits, np.int32).T
    return np.column_stack((starts + 1, np.searchsorted(ticks, ends), slots)).astype(
        np.int32
    )


def describe(g):
    f = FAMILIES[g.last][0]
    if f in SYNC:
        return 0, (
            g.tick,
            SYNC.index(f),
            g.group if f in ("CHORD", "PSEUDO_EACH") else 0,
        )
    flags, head = g.note[3], g.note[5]
    if f in KEY:
        return 1, (g.pad, flags, head, g.group, KEY.index(f))
    zone = g.pad if g.pad < 8 else g.pad - 8
    if f in TOUCH:
        # Split each 33-zone bitmap into nonnegative int32 descriptor columns.
        mask, fw = g.open_touch_mask, g.open_firework_mask
        return 2, (
            mask & 0x1FFFF,
            mask >> 17,
            fw & 0x1FFFF,
            fw >> 17,
            g.tick,
            g.group,
            TOUCH.index(f),
        )
    if f.startswith("HOLD") or g.phase == "hold_beat":
        phase = HOLD_PHASE[f]
        target = g.note[4] if phase == 3 else g.query_tick if phase == 2 else 0
        return 3, (
            zone,
            int(g.pad >= 8),
            flags,
            g.tick,
            target,
            g.pages,
            g.group,
            phase,
        )

    closed = g.phase == "slide_done"
    selecting = g.phase == "offset"
    wait_beat = selecting and g.endpoint == "SLIDE_WAIT"
    end_beat = selecting and g.endpoint == "SLIDE_END"
    launch = (
        g.query_tick
        if wait_beat
        else g.launch
        if g.phase == "end_beat" or end_beat or closed
        else -1
    )
    end = g.note[6][-1][1] if closed else g.query_tick if end_beat else -1
    # Route, wait beat/offset, then end beat/offset (including the closed branch).
    if wait_beat:
        phase = 2
    elif end_beat or closed:
        phase = 4
    elif g.phase == "end_beat":
        phase = 3
    else:
        phase = int(g.phase == "wait_beat")
    ints = (
        g.pad,
        g.segment_end,
        g.tick,
        launch,
        end,
        len(g.routes),
        g.branch_idx,
        int(g.track_break),
        flags,
        head,
        g.pages,
        int(g.explicit_wait or wait_beat),
        phase,
        int(closed),
        g.group,
    )
    # Reference 0 is padding; the sign stores the FESTiVAL joint without another array.
    refs = [(ref + 1) * (-1 if joint else 1) for ref, _, joint in g.chunk_steps]
    dist = [arc for _, arc, _ in g.chunk_steps]
    return 4, (ints, refs, dist, g.arc_before_chunk, g.arc)


class HeadRows:
    def __init__(self):
        self.ids = []
        self.records = [[] for _ in WIDTH]

    def append(self, g):
        h, record = describe(g)
        self.ids.append(h)
        self.records[h].append(record)

    def arrays(self):
        result = {}
        ids = np.asarray(self.ids, np.int8)
        for h, records in enumerate(self.records):
            rows = np.flatnonzero(ids == h)
            if h == 4:
                refs, dist, slots, spans = [], [], [], []
                for i, (_, route, arc, before, total) in enumerate(records):
                    refs.extend(route)
                    dist.extend(arc)
                    slots.extend(
                        range(i * MAX_SLIDE_STEPS, i * MAX_SLIDE_STEPS + len(route))
                    )
                    spans.append((len(route), before, total))
                result["slide_refs"] = np.asarray(refs, np.int16)
                result["slide_dist"] = np.asarray(dist, np.float32)
                result["slide_slots"] = np.asarray(slots, np.int64)
                result["slide_arc"] = np.asarray(spans, np.float32).reshape(-1, 3)
                records = [r[0] for r in records]
            result[f"rows{h}"] = rows
            result[f"args{h}"] = np.asarray(records, np.int32).reshape(-1, WIDTH[h])
        return dict(head_ids=ids, heads=result)


def pack_heads(charts, offsets):
    slide_offsets = np.cumsum(
        [0, *(len(c["heads"]["rows4"]) * MAX_SLIDE_STEPS for c in charts[:-1])]
    )
    merged = {}
    for k in charts[0]["heads"]:
        parts = [c["heads"][k] for c in charts]
        off = (
            offsets
            if k.startswith("rows")
            else slide_offsets
            if k == "slide_slots"
            else None
        )
        merged[k] = torch.cat(
            parts if off is None else [x + int(i) for x, i in zip(parts, off)]
        )
    return merged
