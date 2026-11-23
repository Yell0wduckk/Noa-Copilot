"""Map MiaCode UTF-16 selections to complete Simai cells on the model's tick96 grid."""

from dataclasses import dataclass
from fractions import Fraction
import re

from .codec import normalize, to_score
from .simai import beat_events
from maidata.multitask.grid import BeatGrid
from maidata.multitask.simai_writer import SimaiLineLayout, _note


def utf16_size(text):
    return len(text.encode("utf-16-le")) // 2


@dataclass
class Cell:
    pos: int
    end: int
    tick: int
    after: int
    bpm: float
    division: str
    note: str


def _parse_division_int(raw):
    try:
        div = int(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return div if div > 0 and 384 % div == 0 else None


def _leading_tags(text):
    """Return (has_bpm, has_div) for tags immediately at the start of text."""
    s = re.sub(r"\|\|[^\n]*", "", text).lstrip()
    has_bpm = has_div = False
    while s.startswith(("(", "{")):
        tag, delim = s[0], ")" if s[0] == "(" else "}"
        _, sep, tail = s[1:].partition(delim)
        if not sep:
            break
        if tag == "(":
            has_bpm = True
        else:
            has_div = True
        s = tail.lstrip()
    return has_bpm, has_div


def _scan_tail_tags(text):
    """Scan uncommitted text before the insertion point for BPM, division, and newlines."""
    clean = re.sub(r"\|\|[^\n]*", "", text)
    bpm = None
    for m in re.finditer(r"\(([^()\n]+)\)", clean):
        try:
            bpm = float(m.group(1).strip())
        except ValueError:
            pass
    saw_nl = "\n" in clean
    after_nl = clean.rsplit("\n", 1)[-1] if saw_nl else clean
    line_div = None
    for m in re.finditer(r"\{([^{}\n]+)\}", after_nl):
        line_div = m.group(1).strip()
    return bpm, line_div, saw_nl, bool(after_nl.strip())


@dataclass
class Span:
    start: int
    stop: int
    left: int
    right: int
    restore: str
    lead: str = ""
    cell_start: int = 0
    cell_stop: int = 0
    prev_bpm: float | None = None
    prev_div: int | None = None
    line_origin: int = 0
    skeleton: dict | None = None


class EditorChart:
    def __init__(self, text, bpm, first):
        self.text = text
        # Space out comments without changing any offsets, including their commas.
        clean = re.sub(r"\|\|[^\n]*", lambda m: " " * len(m[0]), text)
        self.clean = clean
        self.cells = []
        pos = 0
        tick, step = Fraction(0), Fraction(96)
        div = "4"
        self.end_pos = len(text)
        self.base_bpm, self.first = bpm, first
        self.tempos = {0: bpm}
        for match in re.finditer(r"([^,]*)(,|$)", clean):
            raw, comma = match.groups()
            cell = raw.strip()
            while cell.startswith(("(", "{")):
                tag, delim = cell[0], ")" if cell[0] == "(" else "}"
                value, _, tail = cell[1:].partition(delim)
                if not _:
                    break
                cell = tail.lstrip()
                if tag == "(":
                    bpm = float(value)
                    self.tempos[round(tick)] = bpm
                else:
                    div = value
                    step = (
                        Fraction(384) / Fraction(div)
                        if not div.startswith("#")
                        else Fraction(div[1:]) * Fraction(str(bpm)) * 96 / 60
                    )
            if cell == "E":
                self.end_pos = match.start() + raw.rfind("E")
                break
            if not comma and not cell:
                break
            self.cells.append(
                Cell(pos, match.end(), round(tick), round(tick + step), bpm, div, cell)
            )
            tick += step
            pos = match.end()
        self.end_tick = round(tick)
        self.u16 = [0]
        for char in text:
            self.u16.append(self.u16[-1] + (2 if ord(char) > 0xFFFF else 1))

    def span(self, req, audio_stop):
        from bisect import bisect_left

        if req.get("mode") == "full":
            return Span(0, audio_stop, 0, len(self.text), "")

        def pos(key):
            return min(len(self.text), bisect_left(self.u16, max(0, req.get(key, 0))))

        cursor = pos("cursor_offset")
        left, right = sorted((pos("sel_start"), pos("sel_end")))
        # Older plain-text clients request an explicit duration in beats.
        aligned = "line_align_beats" in req or "continuation_lines" in req or "span_beats" not in req
        width = max(1, int(req.get("line_align_beats", 4))) * 96
        skeleton = self.skeleton(left, right, cursor) if aligned else {}
        left = self.cells[min(skeleton)].pos if skeleton else cursor
        cells = self.cells
        i = next((j for j, c in enumerate(cells) if c.end > left), len(cells))
        inside_note = False
        if not skeleton and i < len(cells) and cells[i].note:
            cell = cells[i]
            note_pos = self.clean.rfind(cell.note, cell.pos, cell.end)
            inside_note = (
                aligned
                and note_pos < cursor < note_pos + len(cell.note)
            )
            if not inside_note and (
                left > note_pos if aligned else self.text[cell.pos : left].strip()
            ):
                i += 1
        # A cell may start on the preceding line (its leading whitespace includes the newline).
        # At an exact line ending, keep the anchor on the user's side of the newline.
        start = cells[i].tick if i < len(cells) else self.end_tick
        a = cells[i].pos if i < len(cells) else self.end_pos
        if i > 0 and i < len(cells):
            # Keep a preceding bar line break outside the replaced range.
            leading_nl = re.match(r"[ \t]*\r?\n", self.text[a : cells[i].end])
            if leading_nl and (not aligned or skeleton or left >= a + leading_nl.end()):
                a += leading_nl.end()
        if aligned and not skeleton and i < len(cells):
            # Keep comments, blank lines and complete timing tags on the cursor's side.
            upto = min(
                left,
                (
                    self.clean.rfind(cells[i].note, cells[i].pos, cells[i].end)
                    if cells[i].note
                    else cells[i].end - 1
                ),
            )
            tail = self.clean[cells[i].pos : upto]
            if tail and re.fullmatch(r"(?:\s|\([^()]*\)|\{[^{}]*\})*", tail):
                a = max(a, upto)
        if aligned and i == len(cells):
            a = max(cells[-1].end if cells else 0, min(cursor, self.end_pos))
        prev_end = cells[i - 1].end if i > 0 else 0
        tail_bpm, tail_div, tail_saw_nl, tail_has_content = _scan_tail_tags(
            self.text[prev_end:a]
        )
        if i > 0:
            prev_bpm = tail_bpm if tail_bpm is not None else cells[i - 1].bpm
        else:
            prev_bpm = tail_bpm
        at_line_start = (i == 0 and not tail_has_content) or (
            tail_saw_nl and not tail_has_content
        )
        if at_line_start:
            prev_div = None
            line_origin = start
        else:
            raw_div = (
                tail_div
                if tail_div is not None
                else (
                    None if tail_saw_nl else (cells[i - 1].division if i > 0 else None)
                )
            )
            prev_div = _parse_division_int(raw_div)
            if tail_saw_nl or i == 0:
                line_origin = start
            else:
                line_cell_idx = 0
                for k in range(i - 1, -1, -1):
                    if "\n" in self.text[cells[k].pos : cells[k].end]:
                        line_cell_idx = k
                        break
                line_origin = cells[line_cell_idx].tick

        lead = "," if i and not self.text[:a].rstrip().endswith(",") else ""
        if aligned and i and self.text[cells[i - 1].end : a].strip():
            lead = ""
        span = Span(
            start=start, stop=start, left=a, right=a, restore="", lead=lead,
            cell_start=i, cell_stop=i, prev_bpm=prev_bpm, prev_div=prev_div,
            line_origin=line_origin, skeleton=skeleton or None,
        )
        if inside_note:
            self.rewind_moment(span, i)

        if skeleton:
            j = max(skeleton) + 1
            span.stop = cells[j - 1].after
        elif aligned:
            # A handwritten overlong line ends here. Its excess never counts toward the next line.
            if span.start - span.line_origin >= width:
                newline = re.match(r"[ \t]*\r?\n", self.text[span.left:])
                if newline and not span.lead:
                    span.left += newline.end()
                else:
                    span.lead += "\n"
                span.line_origin = span.start
                span.prev_div = None
            lines = max(1, int(req.get("continuation_lines", 4)))
            span.stop = max(span.start, min(audio_stop, span.line_origin + lines * width))
            j = bisect_left(cells, span.stop, lo=span.cell_start, key=lambda cell: cell.tick)
        else:
            target = min(audio_stop, start + max(1, int(req["span_beats"])) * 96)
            j = bisect_left(cells, target, lo=i, key=lambda cell: cell.tick)
            span.stop = cells[j].tick if j < len(cells) else max(self.end_tick, target)
            if start > line_origin and (start - line_origin) % width == 0:
                span.lead += "\n"
                span.prev_div = None
                span.line_origin = start

        span.cell_stop = j
        span.right = cells[j - 1].end if j > span.cell_start else span.left
        if not skeleton and span.stop > span.start and (span.stop - span.line_origin) % width == 0:
            # The generated last comma supplies this line break in the replacement.
            newline = re.match(r"[ \t]*\r?\n", self.text[span.right:])
            if newline:
                span.right += newline.end()
        span.restore = self.suffix_restore(j, None, None, suffix_pos=span.right)
        return span

    def rewind_moment(self, span, index):
        from bisect import bisect_right
        from math import gcd
        from maidata.flat.codec import GRID32_TICKS

        cells = self.cells
        cell = cells[index]
        beat = cell.tick // 96 * 96
        earliest = beat
        line = (
            self.text.rfind("\n", 0, self.clean.rfind(cell.note, cell.pos, cell.end))
            + 1
        )
        for j in range(index - 1, -1, -1):
            previous = cells[j]
            if previous.tick < beat:
                break
            if previous.end <= line:
                earliest = max(earliest, previous.after)
                break
            if previous.note:
                earliest = max(earliest, previous.tick + 1)
                break
        start = next(
            (beat + t for t in GRID32_TICKS if beat + t >= earliest), cell.tick
        )
        if start >= cell.tick:
            return
        slot = bisect_right(cells, start, 0, index, key=lambda c: c.tick) - 1
        if slot < 0:
            return
        source = cells[slot]
        # Move the textual cut independently of the decoder cutoff. An onset can
        # fall inside the previous comma's duration; carry its literal note once.
        raw = self.clean[source.pos : source.end].rstrip(",")
        head = re.match(r"(?:\s|\([^()]*\)|\{[^{}]*\})*", raw).end()
        span.left = source.pos + head
        span.lead = ""
        span.prev_bpm = source.bpm
        span.prev_div = _parse_division_int(source.division)
        if start > source.tick:
            gap = start - source.tick
            unit = gcd(gap, 384)
            div = 384 // unit
            span.lead = (
                ("" if div == span.prev_div else f"{{{div}}}")
                + source.note
                + "," * (gap // unit)
            )
            span.prev_div = div
        span.cell_start = slot
        span.start = start

    def skeleton(self, left, right, cursor):
        """Match complete note items only, in the editor's current difficulty body."""
        rows = {}
        line_starts = [0] + [m.end() for m in re.finditer("\n", self.text)]
        from bisect import bisect_right

        active_line = bisect_right(line_starts, cursor) - 1
        has_slots = set()
        for i, cell in enumerate(self.cells):
            if not any(
                re.fullmatch(r"x|x?\*+", item.strip(), re.I)
                for item in cell.note.split("/")
            ):
                continue
            clean = re.sub(
                r"\|\|[^\n]*", lambda m: " " * len(m[0]), self.text[cell.pos : cell.end]
            )
            start = clean.rfind(cell.note)
            if start < 0:
                continue
            items = []
            for match in re.finditer(r"[^/]+", cell.note):
                item = match[0].strip()
                a = (
                    cell.pos
                    + start
                    + match.start()
                    + len(match[0])
                    - len(match[0].lstrip())
                )
                b = a + len(item)
                placeholder = bool(re.fullmatch(r"x|x?\*+", item, re.I))
                line = bisect_right(line_starts, a) - 1
                if placeholder:
                    has_slots.add(line)
                selected = placeholder and (left == right or (a < right and b > left))
                items.append(
                    dict(
                        text=item,
                        branches=item.count("*") if selected else -1,
                        placeholder=placeholder,
                        start=self.u16[a],
                        stop=self.u16[b],
                    )
                )
            if any(item["branches"] >= 0 for item in items):
                rows[i] = (line, items)
        if left == right:
            if active_line not in has_slots:
                return {}
            lo = hi = active_line
            while lo - 1 in has_slots:
                lo -= 1
            while hi + 1 in has_slots:
                hi += 1
            rows = {i: row for i, row in rows.items() if lo <= row[0] <= hi}
        return {i: items for i, (_, items) in rows.items()}

    def suffix_restore(self, cell_index, end_bpm, end_division, suffix_pos=None):
        """Return minimal (bpm){div} tags needed to preserve cells[cell_index] timing."""
        if cell_index >= len(self.cells):
            return ""
        cell = self.cells[cell_index]
        pos = cell.pos if suffix_pos is None else suffix_pos
        has_bpm, has_div = _leading_tags(self.text[pos : cell.end])
        need_bpm = (not has_bpm) and (end_bpm is None or abs(end_bpm - cell.bpm) > 1e-7)
        need_div = (not has_div) and (
            end_division is None or str(end_division) != str(cell.division)
        )
        bpm_tag = f"({cell.bpm:.12g})" if need_bpm else ""
        div_tag = f"{{{cell.division}}}" if need_div else ""
        return bpm_tag + div_tag

    def history(self, cutoff):
        # Parse only committed notes; all tempo controls affect long-note releases.
        parts = []
        previous_bpm = self.base_bpm
        for cell in self.cells:
            change = f"({cell.bpm:.12g})" if previous_bpm != cell.bpm else ""
            parts.append(
                f"{change}{{{cell.division}}}"
                + (
                    "/".join(
                        item
                        for item in cell.note.split("/")
                        if not re.fullmatch(r"x|x?\*+", item.strip(), re.I)
                    )
                    if cell.tick < cutoff
                    else ""
                )
            )
            previous_bpm = cell.bpm
        events, meta = beat_events(",".join(parts), self.base_bpm, self.first)
        grid = BeatGrid(meta)
        return normalize(events, grid), grid, meta


def fragment_chunks(
    moments,
    grid,
    start,
    stop,
    *,
    initial_tempo=True,
    initial_bpm=None,
    initial_division=None,
    line_origin=0,
    layout=None,
    quarter_tail=False,
):
    """Yield (stop, inherited subdivision, text) for each acceptance step."""
    if stop <= start:
        return
    if layout is None:
        layout = SimaiLineLayout(division=initial_division, line_origin=line_origin)
    groups = {}
    for note in to_score(moments):
        groups.setdefault(note["start_tick"], {}).setdefault(
            note["pseudo_group"], []
        ).append(note)
    tempos = grid.step_tempos()
    bpm = next(
        (v for t, v in sorted(tempos.items(), reverse=True) if t <= start), grid.bpms[0]
    )
    prev_grid_bpm = next(
        (v for t, v in sorted(tempos.items(), reverse=True) if t < start), grid.bpms[0]
    )
    changes = {t: v for t, v in tempos.items() if start <= t < stop}
    if initial_tempo:
        changes[start] = bpm
    cur_bpm = (
        initial_bpm
        if initial_bpm is not None
        else (None if initial_tempo else prev_grid_bpm)
    )
    onsets = {t for t in groups if start <= t < stop}
    points = sorted({start, *changes, *onsets, stop})
    inherited = layout.division
    parts = []
    for t, nxt in zip(points, points[1:]):
        tempo = None
        if t in changes:
            bpm = changes[t]
            if t == 0 and initial_tempo and cur_bpm is None:
                parts.append(layout.begin_chart(bpm))
                cur_bpm = bpm
            elif cur_bpm is None or abs(bpm - cur_bpm) > 1e-7:
                tempo = bpm
                cur_bpm = bpm
        chord = groups.get(t, {})
        notes = chr(96).join(
            "/".join(_note(n, grid, bpm) for n in chord[g]) for g in sorted(chord)
        )
        parts.append(
            layout.interval(
                t, nxt, notes, tempo=tempo, quarter_tail=quarter_tail and nxt == stop
            )
        )
        if nxt == stop or nxt in onsets:
            yield nxt, inherited, "".join(parts)
            parts.clear()
            inherited = layout.division


def render_fragment(moments, grid, start, stop, **kwargs):
    return "".join(
        text for _, _, text in fragment_chunks(moments, grid, start, stop, **kwargs)
    )
