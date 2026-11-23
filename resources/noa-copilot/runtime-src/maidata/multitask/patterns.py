"""Learn a D8 quotient vocabulary, relative placements and joint acoustic phrases."""
from collections import Counter, defaultdict
from functools import lru_cache
import json
from pathlib import Path

from maidata.chart.routes import mirror_pad, mirror_route, route_shape_end, shape_route_table
from .events import Event
from .grid import BeatGrid
from .slots import event_slot


def compose(g, h):
    """Apply h, then g: rotation follows the cabinet's actual mirror."""
    return ((g%8 + (-1 if g>=8 else 1)*(h%8))%8) + 8*((g//8)^(h//8))


def inverse(g):
    return g if g>=8 else (-g)%8


@lru_cache(maxsize=None)
def pad_action(g, pad):
    if pad == 'C':
        return pad
    p = mirror_pad(pad) if g>=8 else pad
    return p[0] + str((int(p[1])-1+g%8)%8+1)


@lru_cache(maxsize=None)
def route_action(g, start, shape, end):
    if g>=8:
        rid = shape_route_table()[shape][int(start[1])-1][end]
        shape, _ = route_shape_end(mirror_pad(start), mirror_route(start,rid))
    return shape, int(pad_action(g,f'A{end+1}')[1])-1


def transform(notes, g):
    out = []
    for t, kind, pad, flags, head, hold, branches in notes:
        bodies = []
        for wait, length, brk, routes in branches:
            start, steps = pad, []
            for shape, end in routes:
                steps.append(route_action(g,start,shape,end))
                start = f'A{end+1}'
            bodies.append((wait,length,brk,tuple(steps)))
        out.append((t,kind,pad_action(g,pad),flags,head,hold,tuple(sorted(bodies))))
    return tuple(sorted(out))


def canonical(notes):
    key, undo = min((transform(notes, g), g) for g in range(16))
    return key, inverse(undo)


def chart_phrases(events, grid, width):
    phrases = defaultdict(list)
    for e in events:
        d = e.data
        t = d['start_tick'] if 'start_tick' in d else grid.tick(e.time)
        if t < 0:
            continue
        flags = (int(bool(d.get('break_',d.get('break')))) | (int(bool(d.get('ex'))) << 1) |
                 (int(bool(d.get('firework'))) << 2) | (int(bool(d.get('pseudo_each'))) << 3))
        hold = 0
        if e.kind in ('hold','touch_hold'):
            z = d['end_tick'] if 'end_tick' in d else grid.tick(e.time+d.get('duration',0))
            hold = z-t if z-t>6 else 0
        branches = []
        if e.kind == 'slide':
            for b in d.get('branches',[d]):
                launch = e.time+b['wait'] if 'wait' in b else grid.second(t+96)
                routes = b.get('routes')
                if routes is None:
                    routes, start = [], d['pad']
                    for r in b['route_ids']:
                        shape,end = route_shape_end(start,r)
                        routes.append(dict(shape=shape,end=end))
                        start = f'A{end+1}'
                a = b['launch_tick'] if 'launch_tick' in b else grid.tick(launch)
                z = b['end_tick'] if 'end_tick' in b else grid.tick(launch+b['duration'])
                branches.append((a-t, max(1,z-a),
                                 bool(b.get('track_break')), tuple((r['shape'],r['end']) for r in routes)))
        phrases[t//width].append((t%width,e.kind,d['pad'],flags,d.get('head_type',0),hold,tuple(branches)))
    return phrases


def acoustic_phrases(events, grid, width):
    """Joint slot presence and per-slot melodic direction at each score tick."""
    by_tick = defaultdict(dict)
    for e in events:
        if e.kind == 'note_off':
            continue
        slot = event_slot(e.data,e.kind)
        row = by_tick[e.data['start_tick'] if 'start_tick' in e.data else grid.tick(e.time)]
        row[slot] = max(row.get(slot,-1), e.data.get('pitch',-1))
    result, previous = defaultdict(list), {}
    for t,row in sorted(by_tick.items()):
        sounds = []
        for slot,pitch in sorted(row.items()):
            delta = pitch-previous.get(slot,pitch)
            sounds.append((slot, (delta>0)-(delta<0)))
            previous[slot] = pitch
        result[t//width].append((t%width,tuple(sounds)))
    return result


def materialize(notes, base, grid, stop=None):
    out = []
    for t,kind,pad,f,head,hold,bodies in notes:
        t += base
        if stop is not None and t>=stop:
            continue
        sec = grid.second(t)
        d = dict(start_tick=t,pad=pad,break_=bool(f&1),ex=bool(f&2),firework=bool(f&4),pseudo_each=bool(f&8))
        if kind in ('hold','touch_hold'):
            z = min(t+hold,stop) if stop is not None else t+hold
            z = t if z-t<=6 else z
            d.update(end_tick=z,duration=grid.second(z)-sec)
        if kind == 'slide':
            d.update(head_type=head,branches=[])
            for w, length, brk, routes in bodies:
                a = t+w
                z = min(a+length,stop) if stop is not None else a+length
                if z<=a:
                    continue
                launch = grid.second(a)
                d['branches'].append(dict(launch_tick=a,end_tick=z,wait=launch-sec,duration=grid.second(z)-launch,
                    track_break=brk,routes=[dict(shape=s,end=e) for s,e in routes]))
            if not d['branches']:
                if head==2:
                    continue
                kind = 'tap'  # an audible star head can still finish the phrase
                del d['branches'], d['head_type']
        out.append(Event(sec,kind,d))
    return out


def mine(charts, *, width=96, min_count=1, max_patterns=None):
    ids, counts, transitions, acoustics = {}, Counter(), defaultdict(Counter), defaultdict(Counter)
    for events, midi, metadata in charts:
        grid = BeatGrid(metadata)
        chart = chart_phrases(events,grid,width)
        sounds = acoustic_phrases(midi,grid,width)
        previous = None
        for block in range(min(chart,default=0),max(chart,default=-1)+1):
            notes = chart.get(block,())
            key, g = canonical(notes)
            if not notes and previous is not None:
                g = previous[1]
            pid = ids.setdefault(key,len(ids))
            counts[pid] += 1
            if previous is not None:
                last, h = previous
                transitions[last][pid,compose(inverse(h),g)] += 1
            if block in sounds:
                acoustics[pid][tuple(sounds[block])] += 1
            previous = pid,g
    patterns = list(ids)
    kept = [k for k,n in counts.most_common(max_patterns) if n>=min_count]
    index = {p:i for i,p in enumerate(kept)}
    marginal = Counter()
    for p in kept:
        for phrase, count in acoustics[p].items():
            for _, sounds in phrase:
                marginal[sounds] += count
    return dict(width=width,slot_prior=list(marginal.items()),patterns=[patterns[p] for p in kept],counts=[counts[p] for p in kept],
        transitions=[[(index[q],dg,n) for (q,dg),n in transitions[p].items() if q in index] for p in kept],
        acoustic=[list(acoustics[p].items()) for p in kept])


def mine_manifest(path, **options):
    from .data import read_manifest, load_events
    rows,_ = read_manifest(path)
    train = [r for r in rows if r.get('split','train')=='train' and r.get('dataset')!='vita']
    def audio_key(row):
        return row.get('audio_id') or row.get('audio') or row.get('song_id')
    partners = {audio_key(r):r for r in train if r['task']=='midi'}
    def pairs():
        for row in train:
            if row['task'] == 'maimai' and not row.get('pseudo',False):
                midi = partners.get(audio_key(row))
                yield load_events(row['events']), load_events(midi['events']) if midi else (), row.get('metadata',{})
    return mine(pairs(),**options)


def draw_phrases(bank, rng, end_tick, grid):
    ids = range(len(bank['patterns']))
    pick = lambda: rng.choices(ids,weights=bank['counts'])[0]
    pid,g = pick(),rng.randrange(16)
    chart, accompaniment = [], []
    for start in range(0,end_tick,bank['width']):
        if start:
            rows = bank['transitions'][pid]
            if rows:
                pid,dg,_ = rng.choices(rows,weights=[r[2] for r in rows])[0]
                g = compose(g,dg)
            else:
                pid,g = pick(),rng.randrange(16)
        notes = transform(bank['patterns'][pid],g)
        chart.extend(materialize(notes,start,grid,stop=end_tick))
        sounds = bank['acoustic'][pid]
        if sounds:
            phrase,_ = rng.choices(sounds,weights=[r[1] for r in sounds])[0]
            accompaniment.extend((start+t,slots) for t,slots in phrase if start+t<end_tick)
        else:
            # Unpaired charts still provide a rhythm, never a fictitious
            # observed instrument label. Place audible attacks on their heads.
            prior = bank.get('slot_prior',())
            for t in sorted({n[0] for n in notes if start+n[0]<end_tick}):
                slots = (rng.choices(prior,weights=[r[1] for r in prior])[0][0] if prior else
                         [(rng.randrange(8),rng.choice((-1,0,1)))])
                accompaniment.append((start+t,slots))
    return chart,accompaniment
