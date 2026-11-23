"""Note ownership, normalization and physical mirror transforms."""

from collections import defaultdict
from functools import lru_cache
from importlib.resources import files
import json

from maidata.chart.routes import mirror_route, shape_route_table, route_shape_end
from maidata.multitask.grid import hold_end_tick
from maidata.multitask.patterns import pad_action

PADS = (
    tuple(map(str, range(1, 9)))
    + tuple(region + str(i) for region in "AB" for i in range(1, 9))
    + ("C",)
    + tuple(region + str(i) for region in "DE" for i in range(1, 9))
)
PAD_ID = {p: i for i, p in enumerate(PADS)}
PLAIN, HOLD, SLIDE = range(3)
PAD_MAP = tuple(
    tuple(
        PAD_ID[pad_action(g, "A" + p)[1:] if i < 8 else pad_action(g, p)]
        for i, p in enumerate(PADS)
    )
    for g in range(16)
)


def grouped_geometry(notes):
    return tuple(sorted({n[0] for n in notes}))


def normalize(events, grid):
    """Build absolute tick, ordered pseudo subgroups, pad, kind and owned endpoints."""
    by_tick = defaultdict(list)
    for event in events:
        d = event.data
        tick = int(d["start_tick"]) if "start_tick" in d else grid.tick(event.time)
        if tick < 0:
            continue
        pad = PAD_ID[d["pad"] if event.kind.startswith("touch") else d["pad"][1:]]
        group = int(d.get("pseudo_group", 0)) or int(bool(d.get("pseudo_each")))
        flags = (
            int(bool(d.get("break_", d.get("break"))))
            | (int(bool(d.get("ex"))) << 1)
            | (int(bool(d.get("firework"))) << 2)
        )
        kind = (
            HOLD
            if event.kind in ("hold", "touch_hold")
            else SLIDE
            if event.kind == "slide"
            else PLAIN
        )
        end = 0
        if kind == HOLD:
            end = (
                int(d["end_tick"])
                if "end_tick" in d
                else grid.tick(event.time + d.get("duration", 0))
            )
            end = hold_end_tick(tick, end)
        branches = []
        if kind == SLIDE:
            for branch in d.get("branches", [d]):
                launch = (
                    int(branch["launch_tick"])
                    if "launch_tick" in branch
                    else grid.tick(event.time + branch["wait"])
                    if "wait" in branch
                    else tick + 96
                )
                finish = (
                    int(branch["end_tick"])
                    if "end_tick" in branch
                    else grid.tick(grid.second(launch) + branch["duration"])
                )
                routes = branch.get("route_ids")
                if routes is None:
                    routes = []
                    start = pad
                    for route in branch["routes"]:
                        target = route["end"]
                        routes.append(
                            shape_route_table()[route["shape"]][start][target]
                        )
                        start = target
                branches.append(
                    (
                        launch,
                        max(launch + 1, finish),
                        int(bool(branch.get("track_break"))),
                        tuple(routes),
                    )
                )
        by_tick[tick].append(
            (pad, group, kind, flags, end, int(d.get("head_type", 0)), tuple(branches))
        )
    ticks = sorted(by_tick)
    result = defaultdict(list)
    for i, tick in enumerate(ticks):
        prev = ticks[i - 1] if i else tick
        micro = 0 < tick - prev <= 2
        isolated = (
            micro
            and (i < 2 or prev - ticks[i - 2] > 2)
            and (i + 1 == len(ticks) or ticks[i + 1] - tick > 2)
        )
        anchor = prev if isolated else tick
        group_offset = 0
        if isolated:
            next_group = 1 + max(n[1] for n in result[prev])
            # A source may already mark the late onset as delayed. Shift its
            # groups only as far as necessary to append them in source order.
            group_offset = max(0, next_group - min(n[1] for n in by_tick[tick]))
        for pad, group, kind, flags, end, head, branches in by_tick[tick]:
            if isolated and kind == HOLD and end == tick:
                end = anchor
            if isolated and kind == SLIDE:
                branches = tuple(
                    (anchor + 96 if a == tick + 96 else a, z, b, r)
                    for a, z, b, r in branches
                )
            result[anchor].append(
                (pad, group + group_offset, kind, flags, end, head, branches)
            )
    return [
        (tick, sorted(notes, key=lambda n: (n[1], n[0])))
        for tick, notes in sorted(result.items())
        if notes
    ]


def to_score(moments):
    """Rows accepted by the tick96 Simai writer, retaining ordered branch and group IDs."""
    result = []
    for tick, notes in moments:
        for pad, group, kind, flags, end, head, branches in notes:
            touch = pad >= 8
            position = PADS[pad] if touch else "A" + PADS[pad]
            name = (
                ("touch" if touch else "tap"),
                ("touch_hold" if touch else "hold"),
                "slide",
            )[kind]
            row = dict(
                start_tick=tick,
                pad=position,
                kind=name,
                pseudo_group=group,
                pseudo_each=bool(group),
                break_=bool(flags & 1),
                ex=bool(flags & 2),
                firework=bool(flags & 4),
                head_type=head,
            )
            if kind == HOLD:
                row["end_tick"] = end
            elif kind == SLIDE:
                row["branches"] = []
                for launch, finish, brk, ids in branches:
                    routes = []
                    start = position
                    for rid in ids:
                        shape, target = route_shape_end(start, rid)
                        routes.append(dict(shape=shape, end=target))
                        start = f"A{target + 1}"
                    row["branches"].append(
                        dict(
                            launch_tick=launch,
                            end_tick=finish,
                            track_break=bool(brk),
                            routes=routes,
                        )
                    )
            result.append(row)
    return result


def mirror_notes(notes):
    mirrored = []
    for pad, group, kind, flags, end, head, branches in notes:
        paths = []
        for a, z, b, ids in branches:
            start = "A" + PADS[pad]
            mapped = []
            for rid in ids:
                mapped.append(mirror_route(start, rid))
                _, target = route_shape_end(start, rid)
                start = f"A{target + 1}"
            paths.append((a, z, b, mapped))
        mirrored.append((PAD_MAP[8][pad], group, kind, flags, end, head, paths))
    return sorted(mirrored, key=lambda n: (n[1], n[0]))


def transform(word, g):
    return tuple((t, tuple(sorted(PAD_MAP[g][p] for p in chord))) for t, chord in word)


class Vocabulary:
    def __init__(self):
        data = json.loads(
            files("maidata.multitask")
            .joinpath("assets/spine-vocab.json")
            .read_text("utf8")
        )
        self.words = [
            tuple((t, tuple(sorted(PAD_ID[p] for p in chord))) for t, chord in rows)
            for rows in data
        ]
        self.lookup = {}
        self.surfaces = []
        self.mirrors = []
        self.valid = []
        for i, word in enumerate(self.words):
            orbit = [transform(word, g) for g in range(16)]
            slots = {}
            valid = []
            for g, surface in enumerate(orbit):
                valid.append(surface not in slots)
                slots.setdefault(surface, g)
                if i:
                    self.lookup[surface] = (i, slots[surface])
            self.valid.append(valid)
            self.surfaces.append(orbit)
            self.mirrors.append([slots[transform(surface, 8)] for surface in orbit])

    @property
    def size(self):
        return len(self.words)


@lru_cache(maxsize=1)
def vocabulary():
    return Vocabulary()
