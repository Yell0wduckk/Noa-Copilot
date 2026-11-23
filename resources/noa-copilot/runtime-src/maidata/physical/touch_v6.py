"""v6 inward Touch sharing and persistent post-Hold handoffs.

Expansion is one step from exact contacts, never a radial transitive closure.
"""
from dataclasses import dataclass, replace
from itertools import combinations
from functools import lru_cache
from .geometry import pad_bit
from .rules import EPS, handoff_releases

CONTRACT = 'inward-touch-late50-early15-hold-handoff-micro4.v2'


@lru_cache(maxsize=4096)
def inward(mask):
    result = mask
    for i in range(1,9):
        following, previous = i % 8+1, (i-2) % 8+1
        if mask & pad_bit(f'D{i}'): result |= pad_bit(f'E{i}')
        if mask & pad_bit(f'B{i}'):
            result |= pad_bit('C') | pad_bit(f'E{i}') | pad_bit(f'E{following}')
        if mask & pad_bit(f'A{i}'): result |= pad_bit(f'B{i}')
        if all(mask & pad_bit(f'A{j}') for j in (previous,i)):
            result |= pad_bit(f'D{i}') | pad_bit(f'E{i}')
        if all(mask & pad_bit(f'B{j}') for j in (previous,i)):
            result |= pad_bit(f'E{i}') | pad_bit('C')
    return result


@lru_cache(maxsize=16)
def button_neighborhood(pad, *, center=False):
    i = int(pad[1]); following = i % 8+1
    return sum(pad_bit(p) for p in (f'A{i}',f'B{i}',f'D{i}',f'D{following}',f'E{i}',f'E{following}')) | (pad_bit('C') if center else 0)


def hold_releases(actions):
    """Effective press ends, including already committed handoff history."""
    result = {id(a): min(a.press_end, a.press_release) if a.press_release is not None else a.press_end for a in actions}
    if not any(a.physical_v6 and a.kind=='hold' for a in actions): return result
    for hold in actions:
        if not hold.physical_v6 or hold.kind != 'hold' or hold.actual_end < 0: continue
        for a in actions:
            if not a.head or not hold.actual_end-EPS <= a.press_start < result[id(hold)]-EPS: continue
            if a.touch:
                nearby = bool(button_neighborhood(hold.pad,center=True) & pad_bit(a.pad))
            else:
                distance = abs(int(a.pad[1])-int(hold.pad[1]))
                nearby = min(distance,8-distance)==1
            if nearby: result[id(hold)] = a.press_start
    return result


def motion_windows(m):
    if m.contacts: return m.contacts
    # Small synthetic geometries expose judge entries only.
    return tuple((t, m.entries[i+1][0] if i+1<len(m.entries) else m.end, mask)
                 for i,(t,mask) in enumerate(m.entries))


@dataclass(frozen=True)
class Resource:
    key: tuple
    cost: int
    edges: frozenset = frozenset()
    sources: tuple = ()
    wide: tuple = ()
    direct: int = 0


def resources_at(actions, lo, releases, press_ends):
    from .conflicts import press_key, track_key
    presses = [a for a in actions if a.head and a.press_start<=lo+EPS and press_ends[id(a)]>lo+EPS]
    tracks = [a for a in actions if a.track and a.launch<=lo+EPS and releases[id(a)]>lo+EPS]
    track_ids={id(a) for a in tracks}
    touches = [a for a in presses if a.touch]
    simultaneous={a.press_start:sum(abs(b.press_start-a.press_start)<=EPS for b in presses if not b.touch) for a in presses if not a.touch}
    resources = []
    for a in tracks:
        sources, wide, direct = [], [], 0
        for i,t in enumerate(touches):
            local = whole = 0
            if t.short and a.launch-EPS<=t.press_start<=a.actual_end+EPS:
                for m in a.motions:
                    for start,end,mask in motion_windows(m):
                        whole |= mask
                        if start-.050-EPS<=t.press_start<=end+.015+EPS: local |= mask
                # At launch a hand can also cover the full inward route area.
                if abs(t.press_start-a.launch)<=EPS:
                    direct |= (1<<i) if (button_neighborhood(a.pad) | inward(whole)) & pad_bit(t.pad) else 0
            sources.append(local); wide.append(whole)
        resources.append(Resource(track_key(a),2 if a.wifi else 1,sources=tuple(sources),wide=tuple(wide),direct=direct))
    for a in (a for a in presses if not a.touch):
        edges = frozenset(track_key(t) for t in tracks if a.short and a.pad==t.pad and abs(a.press_start-t.launch)<=EPS)
        whole=0
        for m in a.motions:
            for _,_,mask in motion_windows(m): whole |= mask
        inner=bool(whole & (pad_bit('C') | sum(pad_bit(f'B{i}') for i in range(1,9))))
        neighborhood=button_neighborhood(a.pad,center=simultaneous[a.press_start]>=2 or inner)
        if a.track: neighborhood |= inward(whole)
        direct = sum(1<<i for i,t in enumerate(touches) if t.short and abs(a.press_start-t.press_start)<=EPS and neighborhood&pad_bit(t.pad))
        resources.append(Resource(press_key(a),1,edges,direct=direct))
    # Launch/press corridors are a property of all simultaneous paths, including
    # secondary branches whose own head=False but whose owner's head is down.
    owners={a.owner for a in presses if not a.touch}
    for i,t in enumerate(touches):
        if not t.short:continue
        whole=0
        for a in actions:
            if a.track and ((id(a) in track_ids and abs(a.launch-t.press_start)<=EPS) or
                            (a.owner in owners and abs(a.press_start-t.press_start)<=EPS)):
                for m in a.motions:
                    for _,_,mask in motion_windows(m):whole |= mask
        if inward(whole)&pad_bit(t.pad):
            for j,r in enumerate(resources):
                # All hosts are present here; subset witness enumeration below
                # freezes absorption before removing any of these hosts.
                if r.cost:
                    resources[j]=replace(r,direct=r.direct|(1<<i));break
    if touches and not covered(resources,touches): resources.append(Resource(('touch_pool',),1))
    return resources,touches


def covered(hosts,touches):
    local=[0]*len(touches); whole=[0]*len(touches); covered=0
    for r in hosts:
        covered |= r.direct
        for i,mask in enumerate(r.sources): local[i] |= mask
        for i,mask in enumerate(r.wide): whole[i] |= mask
    for i,t in enumerate(touches):
        if inward(local[i]) & pad_bit(t.pad): covered |= 1<<i
    for i,t in enumerate(touches):
        same=[j for j,u in enumerate(touches) if abs(u.press_start-t.press_start)<=EPS]
        if len(same)>=3 and all(inward(whole[j])&pad_bit(touches[j].pad) for j in same):
            for j in same: covered |= 1<<j
    return covered == (1<<len(touches))-1


def demand(resources,touches):
    from .conflicts import _demand
    hosts=[r for r in resources if r.key!=('touch_pool',)]
    return _demand([(r.key,r.cost,r.edges) for r in hosts]) + (len(hosts)!=len(resources))


def segments(actions, *, press_ends=None, spans=None):
    releases=handoff_releases(actions)
    if press_ends is None: press_ends=hold_releases(actions)
    points=sorted({t for a in actions for t in (a.press_start,press_ends[id(a)],*((a.launch,releases[id(a)],a.actual_end) if a.track else ()))})
    for lo,hi in zip(points,points[1:]):
        if hi<=lo+EPS: continue
        if spans is not None and not any(lo<end-EPS and hi>start+EPS for start,end in spans): continue
        resources,touches=resources_at(actions,lo,releases,press_ends)
        yield lo,hi,resources,touches


def capacity_conflicts(actions,spans, *, press_ends=None):
    from .conflicts import Conflict
    result=set()
    for lo,hi,resources,touches in segments(actions,press_ends=press_ends,spans=spans):
        if demand(resources,touches)<=2: continue
        cores=[]
        for width in range(1,min(4,len(resources))+1):
            for subset in combinations(resources,width):
                keys=frozenset(r.key for r in subset)
                if any(core<=keys for core in cores) or demand(subset,touches)<=2: continue
                cores.append(keys)
                result.add(Conflict('multitouch','capacity',tuple(sorted(keys))))
    return frozenset(result)
