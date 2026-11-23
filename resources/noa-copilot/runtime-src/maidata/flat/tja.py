"""Parse single-player ESE .tja courses into beat, tick96 and GRID32 events."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from maidata.flat.codec import GRID32_TICKS, snap_tick_grid32
from maidata.multitask.grid import BeatGrid

COURSE_MAP = {
    "0": "easy",
    "easy": "easy",
    "1": "normal",
    "normal": "normal",
    "2": "hard",
    "hard": "hard",
    "3": "oni",
    "oni": "oni",
    "4": "edit",
    "edit": "edit",
    "ura": "edit",
    "5": "tower",
    "tower": "tower",
    "6": "dan",
    "dan": "dan",
}

COURSE_ID = {
    "easy": 0,
    "normal": 1,
    "hard": 2,
    "oni": 3,
    "edit": 4,
    "tower": 5,
    "dan": 6,
}

# Taiko note kind codes (mirroring .tja digits).
DON = 1
KA = 2
BIG_DON = 3
BIG_KA = 4
ROLL = 5
BIG_ROLL = 6
BALLOON = 7
KUSUDAMA = 9  # Big balloon / yam

HIT_KINDS = {DON, KA, BIG_DON, BIG_KA}
ROLL_KINDS = {ROLL, BIG_ROLL}
BALLOON_KINDS = {BALLOON, KUSUDAMA}
SUSTAIN_KINDS = ROLL_KINDS | BALLOON_KINDS

_GRID32_SET = frozenset(GRID32_TICKS)


@dataclass
class TaikoEvent:
    kind: int
    beat: float
    second: float
    tick96: int
    grid32_tick: int
    end_beat: float | None = None
    end_second: float | None = None
    end_tick96: int | None = None
    end_grid32_tick: int | None = None
    balloon_hits: int | None = None
    gogo: bool = False


@dataclass
class ParsedTaikoChart:
    path: str
    title: str
    title_ja: str
    subtitle: str
    wave: str
    course: str
    course_id: int
    star_level: int
    style: str
    bpm_initial: float
    offset_tja: float
    offset_seconds: float
    bpm_segments: list[list[float]]
    meter_segments: list[list[float]]
    stops: list[dict[str, float]]
    gogo_intervals: list[list[float]]
    events: list[TaikoEvent]
    total_beats: float
    duration_seconds: float
    has_bpmchange: bool
    has_branch: bool
    has_scroll: bool
    has_delay: bool
    warnings: list[str] = field(default_factory=list)

    def beat_grid_metadata(self) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "bpm": self.bpm_initial,
            "offset_seconds": self.offset_seconds,
            "bpm_segments": self.bpm_segments,
            "meter_segments": self.meter_segments,
        }
        if self.stops:
            meta["stops"] = self.stops
        return meta

    def stats(self) -> dict[str, Any]:
        hits = [e for e in self.events if e.kind in HIT_KINDS]
        rolls = [e for e in self.events if e.kind in ROLL_KINDS]
        balloons = [e for e in self.events if e.kind in BALLOON_KINDS]
        exact_g32 = sum(1 for e in hits if e.tick96 == e.grid32_tick)
        active_dur = (
            max(1.0, hits[-1].second - hits[0].second)
            if len(hits) >= 2
            else max(1.0, self.duration_seconds)
        )
        return {
            "hit_count": len(hits),
            "don_count": sum(1 for e in hits if e.kind in (DON, BIG_DON)),
            "ka_count": sum(1 for e in hits if e.kind in (KA, BIG_KA)),
            "big_count": sum(1 for e in hits if e.kind in (BIG_DON, BIG_KA)),
            "roll_count": len(rolls),
            "balloon_count": len(balloons),
            "exact_grid32_ratio": exact_g32 / max(1, len(hits)),
            "duration_seconds": round(self.duration_seconds, 3),
            "active_nps": round(len(hits) / active_dur, 3),
            "total_beats": round(self.total_beats, 3),
            "warnings": len(self.warnings),
        }


def decode_tja_bytes(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp932", "shift_jis", "latin1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _strip_comment(line: str) -> str:
    idx = line.find("//")
    if idx != -1:
        line = line[:idx]
    return line.strip()


def _parse_int_list(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        s = part.strip()
        if not s:
            continue
        try:
            out.append(int(float(s)))
        except ValueError:
            continue
    return out


def _select_branch_lines(
    raw_lines: list[str], preferred_order: tuple[str, ...] = ("M", "E", "N")
) -> tuple[list[str], bool]:
    """Resolve #BRANCHSTART blocks by selecting the highest available branch.

    In ESE, consecutive #BRANCHSTART blocks often start without an explicit
    #BRANCHEND between them; a new #BRANCHSTART implicitly closes the prior one.
    """
    out: list[str] = []
    has_branch = False
    i = 0
    n = len(raw_lines)
    while i < n:
        line = raw_lines[i]
        up = line.upper()
        if not up.startswith("#BRANCHSTART"):
            out.append(line)
            i += 1
            continue
        has_branch = True
        # Collect lines until next #BRANCHSTART or #BRANCHEND or end of chart.
        i += 1
        sections: dict[str, list[str]] = {}
        pre_branch: list[str] = []
        active: str | None = None
        while i < n:
            sub = raw_lines[i]
            sub_u = sub.upper()
            if sub_u.startswith("#BRANCHSTART"):
                break
            if sub_u.startswith("#BRANCHEND"):
                i += 1
                break
            if sub_u in ("#N", "#E", "#M"):
                active = sub_u[1]
                sections.setdefault(active, [])
                i += 1
                continue
            if active is None:
                pre_branch.append(sub)
            else:
                sections[active].append(sub)
            i += 1
        out.extend(pre_branch)
        chosen = next((b for b in preferred_order if b in sections), None)
        if chosen is not None:
            out.extend(sections[chosen])
    return out, has_branch


def _parse_single_course(
    path: Path,
    rel_path: str,
    meta: dict[str, str],
    raw_chart_lines: list[str],
    style: str,
    branch_order: tuple[str, ...] = ("M", "E", "N"),
) -> ParsedTaikoChart:
    warnings: list[str] = []
    raw_course = meta.get("COURSE", "Oni").strip().lower()
    course = COURSE_MAP.get(raw_course, raw_course)
    course_id = COURSE_ID.get(course, -1)
    try:
        star_level = int(float(meta.get("LEVEL", "0") or "0"))
    except ValueError:
        star_level = 0
    try:
        bpm_initial = float(meta.get("BPM", "120") or "120")
    except ValueError:
        bpm_initial = 120.0
        warnings.append("invalid_bpm_fallback_120")
    if bpm_initial <= 0:
        bpm_initial = 120.0
        warnings.append("nonpositive_bpm_fallback_120")

    try:
        offset_tja = float(meta.get("OFFSET", "0") or "0")
    except ValueError:
        offset_tja = 0.0
        warnings.append("invalid_offset_fallback_0")
    # In TJA, beat 0 occurs at physical audio second = -OFFSET.
    offset_seconds = -offset_tja

    balloon_key = {"M": "BALLOONMAS", "E": "BALLOONEXP", "N": "BALLOONNOR"}.get(
        branch_order[0], "BALLOON"
    )
    balloon_counts = _parse_int_list(meta.get(balloon_key) or meta.get("BALLOON", ""))
    balloon_idx = 0

    chart_lines, has_branch = _select_branch_lines(raw_chart_lines, branch_order)

    cur_bpm = bpm_initial
    cur_num, cur_den = 4, 4
    cur_beat = 0.0
    cur_sec = offset_seconds

    bpm_segments: list[list[float]] = [[cur_sec, cur_bpm]]
    meter_segments: list[list[float]] = [[0.0, cur_num, cur_den]]
    stops: list[dict[str, float]] = []
    gogo_intervals: list[list[float]] = []

    has_bpmchange = False
    has_scroll = False
    has_delay = False
    in_gogo = False
    gogo_start_beat: float | None = None

    events: list[TaikoEvent] = []
    open_sustain: TaikoEvent | None = None

    # Measure buffer holds ordered ("cmd", line) and ("note", char) entries.
    measure_items: list[tuple[str, str]] = []

    def close_sustain(end_beat: float, end_sec: float) -> None:
        nonlocal open_sustain
        if open_sustain is None:
            return
        et96 = round(end_beat * 96.0)
        eg32 = snap_tick_grid32(et96, lo=open_sustain.grid32_tick)
        open_sustain.end_beat = end_beat
        open_sustain.end_second = end_sec
        open_sustain.end_tick96 = et96
        open_sustain.end_grid32_tick = eg32
        open_sustain = None

    def apply_cmd(cmd_line: str, beat_at: float) -> None:
        nonlocal cur_bpm, cur_sec, cur_num, cur_den
        nonlocal has_bpmchange, has_scroll, has_delay, in_gogo, gogo_start_beat
        parts = cmd_line.split()
        head = parts[0].upper()
        if head == "#BPMCHANGE" and len(parts) >= 2:
            try:
                new_bpm = float(parts[1])
            except ValueError:
                warnings.append(f"bad_bpmchange:{cmd_line}")
                return
            if new_bpm <= 0:
                warnings.append(f"nonpositive_bpmchange:{cmd_line}")
                return
            has_bpmchange = True
            if abs(new_bpm - cur_bpm) > 1e-9:
                cur_bpm = new_bpm
                if abs(bpm_segments[-1][0] - cur_sec) < 1e-9:
                    bpm_segments[-1][1] = cur_bpm
                else:
                    bpm_segments.append([cur_sec, cur_bpm])
        elif head == "#MEASURE" and len(parts) >= 2:
            try:
                a_s, b_s = parts[1].split("/", 1)
                a_f, b_f = float(a_s), float(b_s)
                if a_f > 0 and b_f > 0:
                    cur_num, cur_den = a_f, b_f
                    if abs(meter_segments[-1][0] - beat_at) < 1e-9:
                        meter_segments[-1][1:] = [cur_num, cur_den]
                    else:
                        meter_segments.append([beat_at, cur_num, cur_den])
            except ValueError:
                warnings.append(f"bad_measure:{cmd_line}")
        elif head == "#DELAY" and len(parts) >= 2:
            try:
                delay = float(parts[1])
            except ValueError:
                warnings.append(f"bad_delay:{cmd_line}")
                return
            has_delay = True
            if delay > 1e-9:
                stops.append({"time": cur_sec, "duration": delay, "beat": beat_at})
                cur_sec += delay
            elif delay < -1e-9:
                # Negative #DELAY shifts timeline backwards; BeatGrid only supports positive stops.
                warnings.append(f"negative_delay:{delay}")
                cur_sec += delay
        elif head == "#GOGOSTART":
            if not in_gogo:
                in_gogo = True
                gogo_start_beat = beat_at
        elif head == "#GOGOEND":
            if in_gogo:
                in_gogo = False
                if gogo_start_beat is not None and beat_at > gogo_start_beat:
                    gogo_intervals.append([gogo_start_beat, beat_at])
                gogo_start_beat = None
        elif head == "#SCROLL":
            has_scroll = True

    def flush_measure() -> None:
        nonlocal cur_beat, cur_sec, balloon_idx, open_sustain
        note_count = sum(1 for tag, _ in measure_items if tag == "note")
        # Apply pre-note commands first (including #MEASURE before computing bar beats).
        idx = 0
        while idx < len(measure_items) and measure_items[idx][0] == "cmd":
            apply_cmd(measure_items[idx][1], cur_beat)
            idx += 1

        bar_beats = 4.0 * (cur_num / cur_den)
        if note_count == 0:
            cur_beat += bar_beats
            cur_sec += bar_beats * (60.0 / cur_bpm)
            measure_items.clear()
            return

        step_beats = bar_beats / note_count
        slot = 0
        bar_start_beat = cur_beat
        for tag, payload in measure_items[idx:]:
            slot_beat = bar_start_beat + slot * step_beats
            if tag == "cmd":
                apply_cmd(payload, slot_beat)
                continue
            ch = payload
            if ch != "0":
                t96 = round(slot_beat * 96.0)
                g32 = snap_tick_grid32(t96)
                if ch == "8":
                    if open_sustain is not None:
                        close_sustain(slot_beat, cur_sec)
                    else:
                        warnings.append(f"orphan_tail_8_at_beat_{slot_beat:.3f}")
                elif ch in "1234":
                    if open_sustain is not None:
                        warnings.append(f"unclosed_sustain_before_hit_at_{slot_beat:.3f}")
                        close_sustain(slot_beat, cur_sec)
                    events.append(
                        TaikoEvent(
                            kind=int(ch),
                            beat=slot_beat,
                            second=cur_sec,
                            tick96=t96,
                            grid32_tick=g32,
                            gogo=in_gogo,
                        )
                    )
                elif ch in "5679":
                    kind = int(ch)
                    if open_sustain is not None:
                        if open_sustain.kind == kind:
                            # TJA continuation marker (e.g. Kusudama 9...9...8 or repeated roll head).
                            slot += 1
                            cur_sec += step_beats * (60.0 / cur_bpm)
                            continue
                        warnings.append(f"unclosed_sustain_before_new_at_{slot_beat:.3f}")
                        close_sustain(slot_beat, cur_sec)
                    bhits = None
                    if kind in BALLOON_KINDS:
                        if balloon_idx < len(balloon_counts):
                            bhits = balloon_counts[balloon_idx]
                        balloon_idx += 1
                    ev = TaikoEvent(
                        kind=kind,
                        beat=slot_beat,
                        second=cur_sec,
                        tick96=t96,
                        grid32_tick=g32,
                        balloon_hits=bhits,
                        gogo=in_gogo,
                    )
                    events.append(ev)
                    open_sustain = ev
                else:
                    warnings.append(f"unknown_note_char:{ch}")
            slot += 1
            cur_sec += step_beats * (60.0 / cur_bpm)
        cur_beat = bar_start_beat + bar_beats
        measure_items.clear()

    for line in chart_lines:
        if line.startswith("#"):
            measure_items.append(("cmd", line))
            continue
        parts = line.split(",")
        for pi, part in enumerate(parts):
            for ch in part:
                if not ch.isspace():
                    measure_items.append(("note", ch))
            if pi < len(parts) - 1:
                flush_measure()

    # Handle trailing commands or unclosed sustains/gogo cleanly.
    if measure_items:
        has_notes = any(tag == "note" for tag, _ in measure_items)
        if has_notes:
            warnings.append("trailing_notes_without_comma")
            flush_measure()
        else:
            for _, payload in measure_items:
                apply_cmd(payload, cur_beat)
            measure_items.clear()

    if open_sustain is not None:
        warnings.append("unclosed_sustain_at_eof")
        close_sustain(cur_beat, cur_sec)
    if in_gogo and gogo_start_beat is not None and cur_beat > gogo_start_beat:
        gogo_intervals.append([gogo_start_beat, cur_beat])

    return ParsedTaikoChart(
        path=rel_path,
        title=meta.get("TITLE", path.stem),
        title_ja=meta.get("TITLEJA", ""),
        subtitle=meta.get("SUBTITLE", ""),
        wave=meta.get("WAVE", ""),
        course=course,
        course_id=course_id,
        star_level=star_level,
        style=style,
        bpm_initial=bpm_initial,
        offset_tja=offset_tja,
        offset_seconds=offset_seconds,
        bpm_segments=bpm_segments,
        meter_segments=meter_segments,
        stops=stops,
        gogo_intervals=gogo_intervals,
        events=events,
        total_beats=cur_beat,
        duration_seconds=max(0.0, cur_sec - offset_seconds),
        has_bpmchange=has_bpmchange,
        has_branch=has_branch,
        has_scroll=has_scroll,
        has_delay=has_delay,
        warnings=warnings,
    )


def parse_tja_file(
    path: Path,
    *,
    root: Path | None = None,
    target_only: bool = True,
    branch_order: tuple[str, ...] = ("M", "E", "N"),
) -> list[ParsedTaikoChart]:
    """Parse all single-player courses from a .tja file.

    When target_only=True, keeps Oni, Edit (Ura), and Hard ★7–8.
    """
    text = decode_tja_bytes(path.read_bytes())
    rel_path = path.relative_to(root).as_posix() if root is not None else path.name

    global_meta: dict[str, str] = {}
    course_meta: dict[str, str] = {}
    seen_any_course = False
    in_chart = False
    chart_style = "SINGLE"
    chart_lines: list[str] = []
    parsed: list[ParsedTaikoChart] = []

    course_scoped_keys = {
        "COURSE",
        "LEVEL",
        "BALLOON",
        "BALLOONNOR",
        "BALLOONEXP",
        "BALLOONMAS",
        "SCOREINIT",
        "SCOREDIFF",
        "STYLE",
        "LIFE",
    }

    for raw_line in text.splitlines():
        line = _strip_comment(raw_line)
        if not line:
            continue
        up = line.upper()
        if up.startswith("#START"):
            in_chart = True
            chart_lines = []
            parts = up.split()
            chart_style = parts[1] if len(parts) > 1 else "SINGLE"
            continue
        if up.startswith("#END"):
            if in_chart:
                merged = dict(global_meta)
                merged.update(course_meta)
                raw_c = merged.get("COURSE", "Oni").strip().lower()
                cname = COURSE_MAP.get(raw_c, raw_c)
                try:
                    lvl = int(float(merged.get("LEVEL", "0") or "0"))
                except ValueError:
                    lvl = 0
                is_single = chart_style not in ("P1", "P2")
                keep = is_single and (
                    not target_only
                    or cname in ("oni", "edit")
                    or (cname == "hard" and lvl in (7, 8))
                )
                if keep:
                    parsed.append(
                        _parse_single_course(
                            path, rel_path, merged, chart_lines, chart_style, branch_order
                        )
                    )
            in_chart = False
            chart_lines = []
            continue
        if in_chart:
            chart_lines.append(line)
            continue
        if ":" in line and not line.startswith("#"):
            k, v = line.split(":", 1)
            ku = k.strip().upper()
            vu = v.strip()
            if ku == "COURSE":
                seen_any_course = True
                course_meta = {"COURSE": vu}
            elif ku in course_scoped_keys or seen_any_course:
                course_meta[ku] = vu
            else:
                global_meta[ku] = vu

    return parsed


def verify_beatgrid_roundtrip(chart: ParsedTaikoChart) -> float:
    """Return max |BeatGrid.beat_to_second(beat) - parser_second| in milliseconds."""
    if not chart.events:
        return 0.0
    grid = BeatGrid(chart.beat_grid_metadata())
    max_err_ms = 0.0
    for ev in chart.events:
        err = abs(grid.beat_to_second(ev.beat) - ev.second) * 1000.0
        if err > max_err_ms:
            max_err_ms = err
        if ev.end_beat is not None and ev.end_second is not None:
            err_e = abs(grid.beat_to_second(ev.end_beat) - ev.end_second) * 1000.0
            if err_e > max_err_ms:
                max_err_ms = err_e
    return max_err_ms


def build_taiko_span_targets(chart: ParsedTaikoChart, n_beats: int) -> np.ndarray:
    """Centered 2-beat window [has_roll, has_balloon] matching v19 FlatReconstructionHead."""
    n_anchors = (n_beats + 1) // 2 + 1
    spans = np.zeros((n_anchors, 2), dtype=bool)
    for ev in chart.events:
        if ev.kind in ROLL_KINDS:
            ch = 0
        elif ev.kind in BALLOON_KINDS:
            ch = 1
        else:
            continue
        start = ev.grid32_tick
        finish = max(start + 1, ev.end_grid32_tick or (start + 1))
        k0 = max(0, (start + 96) // 192)
        k1 = min(n_anchors - 1, (finish - 1 + 96) // 192)
        if k0 <= k1:
            spans[k0 : k1 + 1, ch] = True
    return spans
