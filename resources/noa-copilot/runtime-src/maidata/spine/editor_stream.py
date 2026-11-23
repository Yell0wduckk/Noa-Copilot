"""Structured editor RPC. Timing and note semantics cross the wire; MiaCode owns the layout."""

from bisect import bisect_left
import math

import torch

from maidata.chart.routes import path_for, route_shape_end
from maidata.flat.codec import note_tokens, snap_moments_grid32
from maidata.flat.editor_constraints import Skeleton
from maidata.flat.editor_decode import EditorDecoder, MAX_CANDIDATE_ATTEMPTS, canonical
from maidata.spine.codec import HOLD, PADS, SLIDE
from .editor import EditorChart, _leading_tags


def notes_at(tick, notes, grid, skeleton=None):
    remaining = list(skeleton.rows.get(tick, ())) if skeleton else []
    rows = []
    for note in notes:
        pad, group, kind, flags, end, head, branches = note
        row = dict(pad=PADS[pad], group=group, flags=flags, kind=kind, head=head)
        if kind == HOLD:
            row.update(end=end, seconds=grid.second(end) - grid.second(tick))
        if kind == SLIDE:
            row["branches"] = []
            for launch, end, brk, routes in branches:
                pieces = []
                start = "A" + str(pad + 1)
                for rid in routes:
                    pieces.append(path_for(start, int(rid))[1:])
                    _, target = route_shape_end(start, int(rid))
                    start = "A" + str(target + 1)
                row["branches"].append(
                    dict(
                        path=pieces,
                        launch=launch,
                        end=end,
                        brk=bool(brk),
                        wait_seconds=grid.second(launch) - grid.second(tick),
                        seconds=grid.second(end) - grid.second(launch),
                    )
                )
        if remaining:
            slot = skeleton.match(note, remaining)
            if slot is not None:
                remaining.remove(slot)
                if "tokens" not in slot:
                    row.update(slot_start=slot["start"], slot_end=slot["stop"])
        rows.append(row)
    return rows


def targets(chart, span, grid, *, reserve_slides=False):
    if not span.skeleton:
        return None
    native, _, _ = chart.history(span.stop)
    by_tick = dict(snap_moments_grid32(native))
    targets = {}
    from maidata.flat.codec import snap_tick_grid32

    for idx in range(span.cell_start, span.cell_stop):
        cell = chart.cells[idx]
        tick = snap_tick_grid32(cell.tick)
        slots = targets.setdefault(tick, [])
        for note in by_tick.pop(tick, ()):
            slots.append(dict(tokens=list(note_tokens(tick, note)), tick=tick))
        for slot in span.skeleton.get(idx, ()):
            if slot["branches"] >= 0:
                slots.append(slot)
        if not slots:
            del targets[tick]
    return Skeleton(targets, reserve_slides=reserve_slides)


def boundary(chart, grid, span, stop, full=False):
    def suffix(i, pos):
        if i >= len(chart.cells):
            return {}
        cell = chart.cells[i]
        bpm, div = _leading_tags(chart.text[pos : cell.end])
        return dict(
            suffix_bpm=0 if bpm else cell.bpm, suffix_div="" if div else cell.division
        )

    out = dict(
        start=span.start,
        stop=stop,
        left=chart.u16[span.left],
        right=chart.u16[span.right],
        origin=span.line_origin,
        division=span.prev_div or 0,
        lead=span.lead,
        start_second=grid.second(span.start),
        end_second=grid.second(stop),
        initial_bpm=span.prev_bpm or 0,
        skeleton=bool(span.skeleton),
        tempos=[[int(t), float(v)] for t, v in sorted(chart.tempos.items())],
    )
    if span.skeleton:
        from maidata.flat.codec import snap_tick_grid32

        cells = []
        for cell in chart.cells[span.cell_start : span.cell_stop]:
            if cell.note:
                at = chart.clean.rfind(cell.note, cell.pos, cell.end)
                if at >= span.left:
                    cells.append(
                        [
                            snap_tick_grid32(cell.tick),
                            chart.u16[at],
                            chart.u16[at + len(cell.note)],
                        ]
                    )
        out["skeleton_cells"] = cells
    if full:
        return out
    out.update(suffix(span.cell_stop, span.right))
    if span.cell_stop > span.cell_start and span.cell_stop < len(chart.cells):
        out["tail_tick"] = max(stop, chart.cells[span.cell_stop].tick)
    ticks = [c.tick for c in chart.cells]
    cuts = []
    for t in range((span.start // 96 + 1) * 96, stop, 96):
        k = bisect_left(ticks, t, span.cell_start, span.cell_stop)
        pos = chart.cells[k].pos if k < span.cell_stop else span.right
        next_tick = (
            chart.cells[k].tick
            if k < span.cell_stop
            else (
                span.stop
                if span.cell_stop > span.cell_start and chart.end_tick >= span.stop
                else t
            )
        )
        cuts.append(
            dict(
                tick=t,
                position=chart.u16[pos],
                next_tick=next_tick,
                second=grid.second(t),
                **suffix(k, pos),
            )
        )
    out["cuts"] = cuts
    return out


class Stream:
    def __init__(self, service):
        self.service = service
        self.trunk = {}
        self.decoder = None
        self.number = 0
        self.root_started = False
        self.anchor_id = None

    def clear(self, trunk=False):
        self.decoder = None
        self.number = 0
        self.root_started = False
        self.anchor_id = None
        self.settings = None
        self.layout = None
        self.grid = None
        if trunk:
            self.trunk.clear()

    def context(self, req):
        s = self.service
        chart = EditorChart(
            req.get("text", ""), req.get("bpm") or 120, req.get("first", 0)
        )
        _, grid, meta = chart.history(0)
        end = math.floor(grid.second_to_beat(s.duration) * 96)
        meta.update(
            level=req.get("level", 13),
            difficulty=req.get("difficulty", 5),
            charter=req.get("designer", ""),
        )
        meta.update(
            (k, req[k])
            for k in ("song_id", "cabinet", "version", "plan_beat_te")
            if req.get(k) is not None
        )
        return chart, grid, meta, end

    def warm(self, req):
        s = self.service
        chart, grid, meta, end = self.context(req)
        with s.autocast():
            s.project.prepared(chart, grid, meta, end, req)

    def run(self, req, cancelled):
        s = self.service
        rid, rev = req["id"], req.get("revision", 0)

        def send(event, **data):
            s.emit(
                dict(id=rid, revision=rev, anchor=self.anchor_id, event=event, **data)
            )

        mode = "full" if req.get("mode") == "full" else "continue"
        full = mode == "full"
        op = req["op"]
        more = op == "more"
        if (
            op in ("more", "refill", "step_anchor")
            and req.get("anchor", self.anchor_id) != self.anchor_id
        ):
            return dict(event="pool_end", exhausted=True, count=0)
        if op not in ("more", "refill"):
            chart, grid, meta, end = self.context(req)
            span = chart.span(req, end + 1)
            stepped = req.get("step_tick") is not None and self.decoder is not None
            if stepped:
                span.start, span.stop = req["step_tick"], self.layout["stop"]
                span.left, span.right = (
                    bisect_left(chart.u16, req[k])
                    for k in ("replace_start", "replace_end")
                )
                ticks = [c.tick for c in chart.cells]
                span.cell_start = bisect_left(ticks, span.start)
                span.cell_stop = bisect_left(ticks, span.stop)
                span.lead = ""
                span.line_origin = self.layout["origin"]
                span.prev_div = req.get("division") or None
                span.prev_bpm = next(
                    v for t, v in reversed(self.layout["tempos"]) if t <= span.start
                )
                span.skeleton = (
                    chart.skeleton(span.left, span.right, span.left)
                    if self.decoder.skeleton
                    else None
                )
            reserve_slides = bool(stepped and self.decoder.skeleton and self.decoder.skeleton.reserve_slides)
            skeleton = targets(chart, span, grid, reserve_slides=reserve_slides)

            def decoder(prefix_tokens=None):
                raw_prefix, _, _ = chart.history(span.start)
                prefix = snap_moments_grid32(raw_prefix)
                # Audio conditioning always covers the audio grid. Long handwritten
                # releases extend the grammar field horizon inside EditorDecoder only.
                prepared = s.project.prepared(chart, grid, meta, end, req)
                return EditorDecoder(
                    s.model,
                    prepared,
                    grid,
                    meta,
                    prefix,
                    end,
                    span,
                    full,
                    self.trunk,
                    (s.project.audio_hash, s.project.grid_key, s.project.cond_key),
                    skeleton,
                    raw_prefix,
                    prefix_tokens,
                )

            if stepped:
                self.decoder = self.decoder.step_anchor(
                    req["candidate"], span.start, recondition=decoder
                )
                self.decoder.skeleton = skeleton
            else:
                self.decoder = decoder()
            self.grid = grid
            self.layout = boundary(chart, grid, span, span.stop, full)
            self.number = 1 if stepped else 0
            self.root_started = stepped
            self.anchor_id = rid
            self.settings = req
            seed = req.get("seed", -1)
            if seed >= 0:
                torch.manual_seed(seed)
            else:
                torch.seed()
        if op == "step_anchor":
            return None
        if self.decoder is None:
            return dict(event="pool_end", exhausted=True)
        d = self.decoder
        mode = "full" if self.settings.get("mode") == "full" else "continue"
        full = mode == "full"
        target = (
            max(self.number + 1, int(req.get("target", 0)))
            if more
            else (1 if full else max(1, int(self.settings.get("candidate_count", 3))))
        )
        temperature = self.settings.get("temperature", 0.5)
        attempt = 0
        exhausted = False
        while self.number < target and attempt < MAX_CANDIDATE_ATTEMPTS:
            if cancelled.is_set():
                raise InterruptedError()
            number = self.number + 1
            branch = d.branch(cancelled) if self.root_started else None
            if self.root_started and branch is None:
                exhausted = True
                break
            self.root_started = True
            attempt += 1
            # A repeated begin replaces the entire provisional path in this slot.
            send(
                "candidate_begin",
                candidate=number,
                attempt=attempt,
                mode=mode,
                **self.layout,
            )

            def publish(t, notes, advance):
                if advance:
                    send("beat", tick=t, candidate=number, attempt=attempt)
                else:
                    send(
                        "moment",
                        candidate=number,
                        attempt=attempt,
                        tick=t,
                        notes=notes_at(t, notes, self.grid, d.skeleton),
                    )

            track = d.sample(
                temperature,
                self.settings.get("slide_probability_multiplier", 1.0),
                publish,
                cancelled,
                branch=branch,
                keep=number == 1,
            )
            if track is None or canonical(track.moments) in d.seen:
                send("candidate_reset", candidate=number, attempt=attempt)
                continue
            self.number = number
            attempt = 0
            if not full:
                d.remember(number, track)
            send("candidate_end", candidate=number, mode=mode)
        count = self.number
        if full:
            self.clear(True)
        return dict(
            event="pool_end",
            exhausted=exhausted,
            count=count,
            pending=not exhausted and count < target,
        )
