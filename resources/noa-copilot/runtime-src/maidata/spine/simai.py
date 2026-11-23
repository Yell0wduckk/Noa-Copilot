"""Read tick96 Simai note groups, including both spellings of pseudo-each.

The official corpus producer uses the accepted IR/cache readers. This reader
also accepts the standard step-grid and timing spellings produced by the model.
"""

from fractions import Fraction
from functools import lru_cache
from pathlib import Path
import re
from maidata.chart.routes import route_shape_end, route_spellings
from maidata.multitask.events import Event
from maidata.multitask.grid import BeatGrid


@lru_cache(maxsize=8)
def paths(start):
    return sorted(
        ((path[1:], i) for i in range(71) for path in route_spellings(start, i)),
        key=lambda x: -len(x[0]),
    )


def beat_events(body, bpm=120.0, offset=0.0):
    cells = []
    tempos = {0: float(bpm)}
    pos = Fraction(0)
    step = Fraction(96)
    for raw in re.sub(r"\|\|[^\n]*", "", body).split(","):
        text = "".join(raw.split())
        while text.startswith(("(", "{")):
            symbol = text[0]
            end = text.index(")" if symbol == "(" else "}")
            value = text[1:end]
            if symbol == "(":
                bpm = float(value)
                tempos[pos] = bpm
            else:
                step = (
                    Fraction(384) / Fraction(value)
                    if not value.startswith("#")
                    else Fraction(value[1:]) * Fraction(str(bpm)) * 96 / 60
                )
            text = text[end + 1 :]
        if text == "E":
            break
        if text:
            cells.append((round(pos), bpm, text))
        pos += step
    clock = []
    last = 0
    seconds = offset
    tempo = tempos[0]
    for tick, value in sorted(tempos.items()):
        seconds += float(tick - last) * 60 / (tempo * 96)
        clock.append((seconds, value))
        last, tempo = tick, value
    meta = dict(bpm_segments=clock, offset_seconds=offset)
    grid = BeatGrid(meta)
    events = []
    for tick, bpm, text in cells:

        def duration(sig, tempo):
            if "#" in sig:
                alt, sig = sig.split("#", 1)
                tempo = float(alt) if alt else tempo
            if ":" not in sig:
                return float(sig)
            div, count = map(float, sig.split(":"))
            return 240 * count / (tempo * div)

        sec = grid.second(tick)
        for group, part in enumerate(re.split(r"['\x60]", text)):
            for atom in part.split("/"):
                if not atom:
                    continue
                pieces = list(atom) if re.fullmatch("[1-8]+", atom) else [atom]
                for note in pieces:
                    match = re.match(r"([ABDE][1-8]|C[12]?|[1-8])([bxf]*)", note)
                    pad, flags = match.groups()
                    touch = not pad.isdigit()
                    pad = ("C" if pad.startswith("C") else pad) if touch else "A" + pad
                    rest = note[match.end() :]
                    d = dict(
                        start_tick=tick,
                        pad=pad,
                        pseudo_group=group,
                        pseudo_each=bool(group),
                        break_="b" in flags,
                        ex="x" in flags,
                        firework="f" in flags,
                    )
                    kind = "touch" if touch else "tap"
                    if rest.startswith("h"):
                        kind = "touch_hold" if touch else "hold"
                        sig = re.search(r"\[([^]]+)\]", rest)
                        d["end_tick"] = (
                            grid.tick(sec + duration(sig[1], bpm)) if sig else tick
                        )
                    elif rest:
                        kind = "slide"
                        head = 0
                        for prefix, index in (("$$", 1), ("!", 2), ("?", 2)):
                            if rest.startswith(prefix):
                                head = index
                                rest = rest[len(prefix) :]
                                break
                        d.update(head_type=head, branches=[])
                        for branch in rest.split("*"):
                            trajectory, sig = branch.split("[", 1)
                            trailing_break = sig.endswith("b")
                            if trailing_break:
                                sig = sig[:-1]
                            sig = sig.rstrip("]")
                            brk = trajectory.endswith("b") or trailing_break
                            if trajectory.endswith("b"):
                                trajectory = trajectory[:-1]
                            routes = []
                            start = pad
                            while trajectory:
                                path, rid = next(
                                    (p, i)
                                    for p, i in paths(start)
                                    if trajectory.startswith(p)
                                )
                                shape, end = route_shape_end(start, rid)
                                routes.append(dict(shape=shape, end=end))
                                start = f"A{end + 1}"
                                trajectory = trajectory[len(path) :]
                            if "##" in sig:
                                wait, sig = sig.split("##", 1)
                                wait = float(wait)
                            elif "#" in sig:
                                wait = 60 / float(sig.split("#", 1)[0])
                            else:
                                wait = 60 / bpm
                            launch = grid.tick(sec + wait)
                            end = grid.tick(sec + wait + duration(sig, bpm))
                            d["branches"].append(
                                dict(
                                    launch_tick=launch,
                                    end_tick=end,
                                    routes=routes,
                                    track_break=brk,
                                )
                            )
                    events.append(Event(sec, kind, d))
    return events, meta


def read_simai(path, difficulty=5):
    text = Path(path).read_text(encoding="utf-8-sig")
    markers = list(re.finditer(r"^&([^=\r\n]+)=", text, re.M))
    fields = {
        m[1]: text[
            m.end() : markers[i + 1].start() if i + 1 < len(markers) else len(text)
        ].strip()
        for i, m in enumerate(markers)
    }
    return beat_events(
        fields[f"inote_{difficulty}"],
        float(fields.get("wholebpm") or 120),
        float(fields.get(f"first_{difficulty}") or fields.get("first") or 0),
    )
