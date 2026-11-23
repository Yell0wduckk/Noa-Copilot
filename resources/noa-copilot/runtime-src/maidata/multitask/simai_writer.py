"""Tick96-native Simai export. Only the BPM spelling step consults physical time."""
from collections import defaultdict
from fractions import Fraction
from math import gcd
from .events import Event
from .grid import BeatGrid, hold_end_tick
from maidata.chart.routes import path_for, route_shape_end, shape_route_table


def tick_score(events, *, bpm=120., offset=0., segments=None, pseudo_each='backtick'):
    grid = BeatGrid(dict(bpm=bpm, offset_seconds=offset, bpm_segments=segments))
    notes = []
    for event in events:
        # Public score rows need only integer ticks. Older Event records are
        # converted at this boundary; the renderer never consumes their seconds.
        if isinstance(event, dict) and 'start_tick' in event:
            d, kind, source_time = event, event['kind'], None
        else:
            e = Event(**event) if isinstance(event, dict) else event
            d, kind, source_time = e.data, e.kind, e.time
        base = int(d['start_tick']) if 'start_tick' in d else grid.tick(source_time)
        delayed = bool(d.get('pseudo_each'))
        t = base + int(delayed and pseudo_each == 'tick')
        note = dict(kind=kind, pad=d['pad'], start_tick=t, pseudo_each=delayed and t==base, pseudo_group=(int(d.get('pseudo_group',delayed)) if t==base else 0),
                    break_=bool(d.get('break_',d.get('break'))), ex=bool(d.get('ex')),
                    firework=bool(d.get('firework')), head_type=d.get('head_type',0))
        if kind in ('hold','touch_hold'):
            if 'end_tick' in d:
                end = int(d['end_tick'])
            elif 'duration_ticks' in d:
                end = base + int(d['duration_ticks'])
            else:
                end = grid.tick(source_time+d.get('duration',0.)) if source_time is not None else base
            note['end_tick'] = max(t, hold_end_tick(base, end))
        if kind == 'slide':
            note['branches'] = []
            for b in d.get('branches',[d]):
                if 'launch_tick' in b:
                    launch = int(b['launch_tick'])
                elif 'wait_ticks' in b or 'wait' not in b:
                    launch = base + int(b.get('wait_ticks',96))
                else:
                    launch = grid.tick(source_time+b['wait'])
                if 'end_tick' in b:
                    end = int(b['end_tick'])
                elif 'duration_ticks' in b:
                    end = launch + int(b['duration_ticks'])
                else:
                    motion = source_time+b['wait'] if source_time is not None and 'wait' in b else grid.second(launch)
                    end = grid.tick(motion+b['duration'])
                launch = max(t,launch)
                routes = b.get('routes')
                if routes is None:
                    routes, start = [], d['pad']
                    for rid in b['route_ids']:
                        shape, target = route_shape_end(start,rid)
                        routes.append(dict(shape=shape,end=target))
                        start = f'A{target+1}'
                note['branches'].append(dict(launch_tick=launch,end_tick=max(launch+1,end),
                                             routes=routes,track_break=b.get('track_break',False)))
        notes.append(note)
    return dict(grid=grid, notes=notes, separator="'" if pseudo_each=='apostrophe' else chr(96))


def _fraction(ticks, whole=384):
    ratio = Fraction(ticks,whole)
    return f'{ratio.denominator}:{ratio.numerator}'


def _duration(grid, start, end, written_bpm):
    ticks = end-start
    seconds = grid.second(end)-grid.second(start)
    effective = ticks*60/(96*seconds)
    sig = _fraction(ticks)
    return sig if abs(effective-written_bpm)<1e-7 else f'{effective:.12g}#{sig}'


def _note(n, grid, bpm):
    t, kind = n['start_tick'], n['kind']
    text = (n['pad'] if kind.startswith('touch') else n['pad'][1:]) + ''.join(
        sign for key,sign in (('break_','b'),('ex','x'),('firework','f')) if n[key])
    if kind in ('hold','touch_hold'):
        return text+'h'+(f"[{_duration(grid,t,n['end_tick'],bpm)}]" if n['end_tick']!=t else '')
    if kind != 'slide':
        return text
    parts = []
    for b in n['branches']:
        start, path = n['pad'], ''
        for r in b['routes']:
            rid = shape_route_table()[r['shape']][int(start[1])-1][r['end']]
            path += path_for(start,rid)[1:]
            start = f"A{r['end']+1}"
        a,z = b['launch_tick'],b['end_tick']
        if a == t:
            sig = '0##'+_duration(grid,a,z,bpm)
        else:
            # Simai's temporary BPM specifies one beat of wait. Express the
            # motion as an exact tick ratio on that clock; a tempo change inside
            # the wait/body uses the equivalent ratio of integrated intervals.
            wait = grid.second(a)-grid.second(t)
            motion = grid.second(z)-grid.second(a)
            ratio = Fraction(z-a,4*(a-t))
            if abs(4*float(ratio)*wait-motion)>1e-9:
                ratio = Fraction(motion/(4*wait)).limit_denominator(1_000_000)
            temp = 60/wait
            sig = ('' if abs(temp-bpm)<1e-7 else f'{temp:.12g}#') + f'{ratio.denominator}:{ratio.numerator}'
        parts.append(path+('b' if b['track_break'] else '')+f'[{sig}]')
    return text + ('','$$','!')[n['head_type']] + '*'.join(parts)


class SimaiLineLayout:
    """Spell exact tick intervals on beat-aligned lines; the next interval owns its tag."""
    def __init__(self, division=None, *, line_origin=0, line_beats=4):
        self.division = division
        self.line_origin = line_origin
        self.line_ticks = max(1, int(line_beats)) * 96

    def begin_chart(self, bpm):
        self.division = 4
        return f"({bpm:.12g}){{4}}"

    def interval(self, tick, stop, body, *, tempo=None, quarter_tail=False):
        if quarter_tail:
            # The onset stays at its exact tick. Round the remaining duration
            # to quarter notes and use whole beats only for line wrapping.
            beats = max(1, (stop - tick + 48) // 96)
            tick -= tick % 96
            stop = tick + beats * 96
        parts = []
        while tick < stop:
            boundary = self.line_origin + ((tick - self.line_origin) // self.line_ticks + 1) * self.line_ticks
            size = min(stop, boundary) - tick
            unit = 96 if quarter_tail or size % 96 == 0 else gcd(size, 384)
            # Keep a compatible grid for mixed rhythms within the same line.
            if self.division is not None and not quarter_tail:
                current = 384 // self.division
                if size % current == 0 and size // current <= 16:
                    unit = current
            division = 384 // unit
            if tempo is not None:
                parts.append(f"({tempo:.12g})")
                tempo = None
            if division != self.division:
                parts.append(f"{{{division}}}")
            parts.extend((body, "," * (size // unit)))
            self.division = division
            body = ""
            tick += size
            if tick == boundary:
                parts.append("\n")
                self.division = None
        return "".join(parts)


def end_chart(fragment):
    return fragment.rstrip("\n") + "\nE" if fragment else "E"


def render_score(score, *, stop_tick=None):
    grid = score['grid']
    groups = defaultdict(lambda: defaultdict(list))
    # Simai has step BPM controls. During a ramp use the equivalent BPM of
    # each tick interval; every tick boundary therefore stays sample-aligned.
    controls = grid.step_tempos()
    first = min(0,min((n['start_tick'] for n in score['notes']),default=0))
    controls = {t:v for t,v in controls.items() if t>=first}
    controls.setdefault(first,60/(96*(grid.second(first+1)-grid.second(first))))
    for n in score['notes']:
        groups[n['start_tick']][n.get('pseudo_group',int(n['pseudo_each']))].append(n)
    if stop_tick is not None:
        stop_tick = max(stop_tick, max(groups, default=first) + 1)
        controls = {t: v for t, v in controls.items() if t < stop_tick}
    points = sorted(groups.keys()|controls.keys())
    layout = SimaiLineLayout()
    bpm = controls[first]
    parts = [layout.begin_chart(bpm)]
    if stop_tick is None:
        stop_tick = points[-1] + 96
    for i,t in enumerate(points):
        tempo = None
        if i and t in controls:
            bpm = controls[t]
            tempo = bpm
        text = score.get('separator',chr(96)).join('/'.join(_note(n,grid,bpm) for n in group) for group in (groups[t][g] for g in range(max(groups[t], default=-1) + 1)))
        last = i + 1 == len(points)
        nxt = stop_tick if last else points[i + 1]
        parts.append(layout.interval(t - first, nxt - first, text, tempo=tempo, quarter_tail=last))
    return end_chart(''.join(parts)), grid.second(first)
