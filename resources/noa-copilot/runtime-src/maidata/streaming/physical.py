"""One prefix-only oracle shared by offline supervision and live generation."""
import math
from copy import copy
from dataclasses import replace
from maidata.chart.types import (Moment, Note, Route, HoldDuration, SlideDuration, ChartError,
                                Completion, BUTTON_PADS, TOUCH_PADS)
from maidata.chart.routes import shape_route_table
from maidata.physical.inference import InferencePhysicalState
from maidata.physical.attribution import AttributionState, Evaluation, Evidence, project
from maidata.physical.conflicts import Conflict, branch_conflicts, first_route_coverage, slide_launch_capacity
from maidata.physical.support import CATEGORIES, duration_roots, _partition, _cuts, _times
from maidata.physical.geometry import pad_bit
from maidata.physical.rules import EPS
from maidata.inference.hold_tail import contacts, final_area_roots
from .sequence import EventToken, CONT_FAMILY, ZONES, EVENT_STEP, EVENT_CONT, HOLD_CONT, SLIDE_CONT, EVENT_STEPS
from .codec import EventMaterializer


class StreamPhysicalState(InferencePhysicalState):
    minimum_hold_ticks = 1


class StreamAttribution(AttributionState):
    def _fork(self,grammar):
        p = self.physical
        other = StreamPhysicalState(p.timing,p.geometry,audio_end_seconds=p.audio_end_seconds)
        other.grammar,other.actions = grammar,p.actions
        result = StreamAttribution(other); result.charged = self.charged
        return result

    def query(self,token,**kwargs):
        value = super().query(token,**kwargs)
        collisions = contacts(self.physical,token)
        if not collisions: return value
        # The existing four-category ledger owns this duration/path conflict.
        witnesses = frozenset(Conflict('slide_too_fast','hold_slide_final_area',
            (('press',c['hold_owner'],c['hold_tick'],c['hold_pad']),
             ('track',c['slide_owner'],c['track_index'],c['slide_tick']))) for c in collisions)
        return Evaluation(replace(value.raw,conflicts=value.raw.conflicts|witnesses),
                          replace(value.penalty,conflicts=value.penalty.conflicts|(witnesses-self.charged)))

    def _duration_values(self):
        p = self.physical
        lower,upper,roots = duration_roots(p,end_seconds=self.duration_end_seconds)
        if upper>=lower: roots.extend(final_area_roots(p,lower))
        constructor = SlideDuration if p.grammar.state.phase == 'chain' else HoldDuration
        rows = _partition(roots,lower,upper,lambda n:self.query(constructor(n)),lambda e:e) if upper>=lower else []
        return dict(domain=[lower,upper],intervals=rows,outside_basic='duration_outside_audio')

    def moment_value(self, tick, *, free_only=False):
        """Same Moment projection, evaluating equivalent free Touch pads once.

        Keep their multiplicity in project(): its stable witness-cover selection
        counts completions, so deduplicating the projection itself is unsafe.
        """
        p = self.physical
        try: grammar,_ = p._trial(Moment(p.timing.delta_ms(p.grammar.state.cursor,tick)))
        except ValueError as exc: return outside(str(exc))
        s = grammar.state
        if s.phase=='closed': return Evaluation()
        free = free_touch_pads(p,s.cursor,touch_repeat=False)
        def query(kind,pad):
            return self._action(Completion(s.owner+1,s.moment,s.cursor,Note.make(kind,pad)),touch_repeat=False)
        shared = query('touch',next(pad for pad in TOUCH_PADS if pad in free)) if free else None
        if shared is not None and not shared.raw.basic and not shared.raw.conflicts:
            return Evaluation()
        if free_only: return None
        return project([query('tap',pad) for pad in BUTTON_PADS]+
                       [shared if pad in free else query('touch',pad) for pad in TOUCH_PADS])

    def moment_values(self, lower, upper):
        """Partition only this beat; path-entry roots matter only under conflict."""
        p,s = self.physical,self.physical.grammar.state
        end = math.ceil(p.timing.second_to_beat(p.audio_end_seconds)*96)-1
        while p.timing.second(end)>=p.audio_end_seconds: end-=1
        upper = min(upper,end)
        lower = max(lower,s.cursor+1)
        rows = []
        if lower<=upper:
            def roots(entries):
                times = _times(p.actions,entries=entries)|{0.}
                if s.window_end is not None: times.add(p.timing.second(s.window_end))
                ticks = (p.timing.second_to_beat(t)*96 for t in times)
                return [x for tick in ticks for x in (tick,tick-2)]
            coarse = _cuts(roots(False),lower,upper)
            fine = None
            for a,b in zip(coarse,coarse[1:]):
                clean = self.moment_value((a+b-1)//2,free_only=True)
                if clean is not None:
                    rows.append([a,b-1,clean])
                else:
                    if fine is None: fine = roots(True)
                    rows.extend(_partition(fine,a,b-1,self.moment_value,lambda e:e))
        return dict(domain=[lower,upper],intervals=rows,outside_basic='onset_outside_audio')


def free_touch_pads(physical, tick, *, touch_repeat=True):
    """Pads with neither overlap, repeat, nor launch-specific handoff edges."""
    t = physical.timing.second(tick)
    occupied = 0
    excluded = set()
    for a in physical.actions:
        if a.press_start<=t+EPS and t<a.press_end-EPS: occupied |= pad_bit(a.pad)
        if a.track and abs(a.launch-t)<=EPS: excluded.add(a.pad)
        if touch_repeat and a.touch and -EPS<=t-a.press_start<=.25+EPS: excluded.add(a.pad)
    return {pad for pad in TOUCH_PADS if pad not in excluded and not occupied&pad_bit(pad)}


def rebind_onset(value, owner, tick):
    """Interval representatives must not leak their tick into charged identities."""
    if not value.raw.conflicts and not value.penalty.conflicts: return value
    def evidence(e):
        return replace(e,conflicts=frozenset(replace(c,participants=tuple(
            (p[0],p[1],tick,*p[3:]) if p[0]=='press' and p[1]==owner else p
            for p in c.participants)) for c in e.conflicts))
    return Evaluation(evidence(value.raw),evidence(value.penalty))


def clone_materializer(source,commit=None):
    result = copy(source)
    result.grammar = copy(source.grammar)
    result.grammar.events = list(source.grammar.events)
    # Speculative candidates need only current content, never an entire song.
    result.groups = []
    result.commit = commit
    return result


class Oracle:
    def __init__(self,timing,geometry,duration,start,end,*,notes_only=False,markers=False,profile=None,physical_enabled=True):
        self.physical_enabled = physical_enabled
        self.timing = timing
        self.profile = profile
        self.physical = StreamPhysicalState(timing,geometry,audio_end_seconds=duration)
        self.physical.inject_boundary(Moment(0),start_tick=start*96)
        self.attribution = StreamAttribution(self.physical)
        self.state = EventMaterializer(start,end,self._commit,notes_only=notes_only,markers=markers)
        self._supports = {}
        self._marker_followups = {}
        self._duration_support = None

    def _commit(self,tick,token):
        if tick != self.physical.grammar.state.cursor:
            self.physical.accept_source(Moment(self.timing.delta_ms(self.physical.grammar.state.cursor,tick)))
        if not self.physical_enabled and not getattr(self.state.grammar,'physical_v6',False):
            self.physical.grammar,_=self.physical._trial(token)
            self.physical._refresh()
        else:self.physical.accept_source(token)

    def fork(self,*,preserve_support=False):
        result = copy(self)
        p = self.physical
        result.physical = StreamPhysicalState(p.timing,p.geometry,audio_end_seconds=p.audio_end_seconds)
        result.physical.restore(p.snapshot())
        result.attribution = StreamAttribution(result.physical)
        result.attribution.charged = self.attribution.charged
        result.state = clone_materializer(self.state,result._commit)
        # A speculative fork can mutate its materializer directly. Never share
        # prefix caches across those mutations or back into its parent.
        result._supports = dict(self._supports) if preserve_support else {}
        result._marker_followups = dict(self._marker_followups) if preserve_support else {}
        result._duration_support = self._duration_support if preserve_support else None
        return result

    def _at_onset(self):
        tick = self.state.grammar.tick
        cursor = self.physical.grammar.state.cursor
        if tick != cursor:
            self.physical.accept_source(Moment(self.timing.delta_ms(cursor,tick)))

    def _position(self,pad,touch):
        self._at_onset()
        return self.attribution.query(Note.make('touch' if touch else 'tap',pad))

    def evaluate(self,token):
        """Candidate validity never conditions on a future teacher endpoint."""
        if not self.physical_enabled:
            return self.basic_support(CONT_FAMILY.get(token.type,token.type)).get(token.value,outside('grammar'))
        try: return self._evaluate(token)
        except (ChartError,ValueError) as exc: return Evaluation(Evidence((str(exc),)),Evidence((str(exc),)))

    def _evaluate(self,token):
        t,v = token.type,token.value
        family = CONT_FAMILY.get(t,t)
        g = self.state.grammar
        if v not in g.allowed_values(family): return Evaluation(Evidence(('grammar',)),Evidence(('grammar',)))
        if family == 1: return Evaluation()
        if family in (2,3,EVENT_STEP):
            cap = round(self.timing.second_to_beat(self.physical.audio_end_seconds)*96)-g.tick
            end = (2*EVENT_STEPS[0] if family == 2 else
                   2*EVENT_STEPS[v] if family == EVENT_STEP else
                   g.elapsed + (g.pending_step if v == EVENT_CONT else 0))
            if end>cap:
                return Evaluation(Evidence(('event_outside_audio',)),Evidence(('event_outside_audio',)))
            return Evaluation()
        if family in (17,18): return self.support(family)[0]
        trial = self.fork()
        evaluations = []
        def commit(tick,content):
            cursor = trial.physical.grammar.state.cursor
            if tick != cursor:
                moment = Moment(self.timing.delta_ms(cursor,tick))
                trial.physical.accept_source(moment)
            # A delayed flush advances accepted history. Only this candidate's
            # upgrade/completion is scored; EOM/EOR/new notes cannot inherit an
            # already accepted action's raw rejection a second time.
            own = (family==7 or family==9 and g.phase in ('after_pad','after_flags') or
                   family==4 and g.phase=='route_pad')
            if own: evaluations.append(trial.attribution.query(content))
            trial.physical.accept_source(content)
        trial.state.commit = commit
        if family == 0:
            if v == 96: return Evaluation()
            tick = g.beat*96+v
            if tick == trial.physical.grammar.state.cursor:
                return project(trial.attribution.query_kinds().values())
            return trial.attribution.query(Moment(self.timing.delta_ms(trial.physical.grammar.state.cursor,tick)))
        if family == 9:
            trial.state.accept(token)
            # Resolve first-route coverage over all geometrically valid ends.
            values = [trial.evaluate(EventToken(4,p)) for p in trial.state.grammar.allowed_values(4)]
            return combine(evaluations+[project_shape(values)])
        if family == 4 and g.phase == 'route_pad':
            trial.state.accept(token); trial.state.flush()
        elif family in (4,5):
            trial.state.accept(token)
            if family == 5 and v != 2:
                return combine(evaluations+[project(trial._position(ZONES[v]+str(p+1),True) for p in range(8))])
            return combine(evaluations+[trial._position(trial.state.pending.pad,trial.state.grammar.zone is not None)])
        elif family in (8,11):
            constructor = HoldDuration if family == 8 else SlideDuration
            if v != 96: return trial.attribution.query(constructor(g.duration+v+1))
            support = trial.duration_support()
            candidates = [e for lo,hi,e in support['intervals'] if hi >= g.duration+97]
            return project(candidates) if candidates else Evaluation(Evidence(('duration_outside_audio',)),Evidence(('duration_outside_audio',)))
        else:
            trial.state.accept(token)
            if family == 6 and isinstance(trial.state.pending,Route): trial.state.flush()
        return combine(evaluations)

    def duration_support(self):
        if self._duration_support is None:
            self._duration_support = self.attribution._duration_values()
        support = self._duration_support
        charged = self.attribution.charged
        if not charged: return support
        return dict(support,intervals=[(lo,hi,uncharge(e,charged)) for lo,hi,e in support['intervals']])

    def support(self,family):
        if family not in self._supports:
            self._supports[family] = self._support(family)
        values = self._supports[family]
        charged = self.attribution.charged
        return {v:uncharge(e,charged) for v,e in values.items()} if charged else values

    def basic_support(self,family):
        """Grammar/version consumers retain audio bounds without Physical queries."""
        from maidata.time import duration_end
        g=self.state.grammar; end=self.physical.audio_end_seconds
        v6=getattr(g,'physical_v6',False)
        def value(v):
            if family==0 and v!=96:
                t=self.timing.second(g.beat*96+v)
                return outside('onset_outside_audio') if t<0 or t>=end else Evaluation()
            if family in (8,11):
                ticks=g.duration+(97 if v==96 else v+1)
                kwargs=dict(wait=g.wait_ticks) if v6 and family==11 else {}
                t=duration_end(self.timing,g.tick,ticks,slide=family==11,**kwargs)
                return outside('duration_outside_audio') if t>end+EPS else Evaluation()
            if v6 and family==23:
                tick=g.tick+g.wait_elapsed+v+2
                return outside('wait_outside_audio') if self.timing.second(tick)>end+EPS else Evaluation()
            if family in (7,9,10,17) or family==4 and g.phase=='route_pad':
                t=(self.timing.second(g.tick+(1 if family==7 else 2)) if v6 else
                   duration_end(self.timing,g.tick,1,slide=family!=7))
                return outside('duration_outside_audio') if t>end+EPS else Evaluation()
            if not v6 and family in (2,3,EVENT_STEP):
                cap=round(self.timing.second_to_beat(end)*96)-g.tick
                ticks=2*EVENT_STEPS[0] if family==2 else 2*EVENT_STEPS[v] if family==EVENT_STEP else g.elapsed+(g.pending_step if v==EVENT_CONT else 0)
                if ticks>cap:return outside('event_outside_audio')
            return Evaluation()
        return {v:value(v) for v in g.allowed_values(family)}

    def _support(self,family):
        if not self.physical_enabled:return self.basic_support(family)
        g = self.state.grammar
        values = g.allowed_values(family)
        if not values: return {}
        if family in (17,18):
            trial=self.fork()
            head=Evaluation()
            if family==17 and g.phase in ('after_pad','after_flags'):
                trial._at_onset(); n=trial.state.pending
                head=trial.attribution.query(Note.make('slide',n.pad,ex=n.ex,break_=n.break_))
            trial.state.accept(EventToken(family))
            from .version import values as version_values
            following={}; choices=[]
            for f in ((9,) if family==17 else (4,5)):
                allowed=set(version_values(trial.state.grammar,f,self.profile))
                if allowed:
                    following[f]=trial.support(f)
                    choices.extend(e for v,e in following[f].items() if v in allowed)
            self._marker_followups[family] = following
            return {0:combine((head,project(choices)))}
        if family==0:
            if values==(96,): return {96:Evaluation()}
            support = self.attribution.moment_values(g.beat*96+g.offset+1,g.beat*96+95)
            result = {}
            for v in values:
                tick = g.beat*96+v
                if v==96: result[v] = Evaluation()
                elif tick==self.physical.grammar.state.cursor: result[v] = self.evaluate(EventToken(0,v))
                else: result[v] = rebind_onset(interval_value(support,tick),self.physical.grammar.state.owner+1,tick)
            return result
        if family in (8,11):
            support = self.duration_support()
            result = {v:interval_value(support,g.duration+v+1) for v in values if v!=96}
            if 96 in values:
                tail = [e for lo,hi,e in support['intervals'] if hi>=g.duration+97]
                result[96] = project(tail) if tail else outside(support['outside_basic'])
            return result
        if family in (4,5) and g.phase!='route_pad':
            trial = self.fork(); trial.state.flush(); trial._at_onset()
            if family==4:
                zone = g.zone if g.phase=='touch_pad' else 'A'
                return {v:trial._position(f'{zone}{v+1}',g.phase=='touch_pad') for v in values}
            free = free_touch_pads(trial.physical,g.tick)
            shared = None
            def position(pad):
                nonlocal shared
                if pad not in free: return trial._position(pad,True)
                if shared is None: shared = trial._position(pad,True)
                return shared
            return {v:position('C') if v==2 else
                    project(position(f'{ZONES[v]}{p+1}') for p in range(8)) for v in values}
        if family==9:
            trial = self.fork()
            head = Evaluation()
            if g.phase in ('after_pad','after_flags'):
                trial._at_onset()
                n = trial.state.pending
                head = trial.attribution.query(Note.make('slide',n.pad,ex=n.ex,break_=n.break_))
            # Flush the slide head or preceding route once for all 19 shapes.
            trial.state.accept(EventToken(9,values[0]))
            p = trial.physical
            capacities = {wifi:slide_launch_capacity(p.actions,p.grammar.state,p.timing,wifi=wifi)
                          for wifi in (False,True)}
            start = trial.state.grammar.pad
            paths = {r for s in values for r in shape_route_table()[s][start] if r>=0}
            routes = {r:route_value(trial.attribution,Route(r,0 if trial.state.grammar.first_route else None),capacities)
                      for r in paths}
            return {s:combine((head,project_shape(routes[r] for r in shape_route_table()[s][start] if r>=0)))
                    for s in values}
        return {v:self.evaluate(EventToken({8:HOLD_CONT,11:SLIDE_CONT}.get(family,family) if v==96 else family,v))
                for v in values}

    def accept(self,token,evaluation=None):
        g = self.state.grammar
        if token.type == 0 and token.value != 96: self.attribution.charged = frozenset()
        if evaluation is not None: self.attribution.charged |= evaluation.penalty.conflicts
        followup = self._marker_followups.get(token.type,{})
        self.state.accept(token)
        self._supports = followup
        self._marker_followups = {}
        if token.type not in (HOLD_CONT,SLIDE_CONT): self._duration_support = None


def uncharge(value, charged):
    conflicts = value.penalty.conflicts
    if not conflicts or conflicts.isdisjoint(charged): return value
    return replace(value,penalty=replace(value.penalty,conflicts=conflicts-charged))


def outside(reason):
    return Evaluation(Evidence((reason,)),Evidence((reason,)))


def interval_value(support,tick):
    return next((e for lo,hi,e in support['intervals'] if lo<=tick<=hi),outside(support['outside_basic']))


def route_value(attribution,token,capacities):
    """Same Route witnesses as query(), sharing the prefix launch checks."""
    p = attribution.physical
    try: grammar,_ = p._trial(token)
    except ValueError as exc: return outside(str(exc))
    prior,s = p.grammar.state,grammar.state
    branch = branch_conflicts(p.actions,prior.owner,prior.track_count,prior.cursor,wifi=token.route_id==70)
    if branch: return attribution._leaf(conflicts=branch)
    coverage = first_route_coverage(p.actions,s,p.timing,p.geometry)
    capacity,inherited = capacities[s.routes==(70,)]
    if coverage or capacity:
        return attribution._leaf(conflicts=coverage|capacity,inherited=inherited,deferred=not coverage)
    if s.routes==(70,): return attribution._fork(grammar)._duration_projection()
    return attribution._leaf(deferred=True)


def combine(values):
    values = tuple(values)
    def merged(which):
        rows = [getattr(v,which) for v in values]
        return Evidence(tuple(sorted({b for r in rows for b in r.basic})),
                        frozenset(c for r in rows for c in r.conflicts),any(r.deferred for r in rows))
    return Evaluation(merged('raw'),merged('penalty'))


def project_shape(values):
    """A category belongs to SHAPE only when every basic endpoint has it."""
    values = tuple(values)
    raw = project(values).raw
    supported = [v.penalty for v in values if not v.raw.basic]
    if not supported: return Evaluation(raw,raw)
    mandatory = set.intersection(*({c.category for c in v.conflicts} for v in supported))
    witnesses = frozenset(c for v in supported for c in v.conflicts if c.category in mandatory)
    return Evaluation(raw,Evidence(conflicts=witnesses,deferred=all(v.deferred for v in supported)))


def penalty(value):
    categories = value.penalty.decision().physical
    return [c in categories for c in CATEGORIES]
