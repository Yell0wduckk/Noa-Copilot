"""Global-tick v6/v8 materialization and prefix-only inference support."""
from copy import copy
from dataclasses import replace
import math
from maidata.chart.types import Note, Route, Moment, HoldDuration, SlideDuration, Completion, BUTTON_PADS
from maidata.chart.routes import shape_route_table
from maidata.physical.actions import materialize
from maidata.physical.attribution import Evidence, Evaluation, project
from maidata.physical.conflicts import action_conflicts, Conflict, capacity_conflicts, branch_conflicts, track_key
from maidata.physical.rules import prune_history, EPS
from maidata.physical.state import Decision
from maidata.physical.support import _partition, _times, _cuts
from maidata.inference.hold_tail import contacts
from .physical import Oracle, StreamPhysicalState, outside, interval_value, rebind_onset
from .codec import EventMaterializer
from .sequence import EventToken, CONT_FAMILY
from .physical_schema import *


def timing_v6(timing):
    result=copy(timing); result.global_ticks=True
    return result


class State(StreamPhysicalState):
    def _trial(self,token,window_end=None):
        if isinstance(token,SlideDuration):
            token=replace(token,wait_ticks=self.grammar.state.wait_ticks)
        return super()._trial(token,window_end)

    def accept_source(self,token):
        grammar,completion=self._trial(token)
        if completion is not None:
            self.actions+= (materialize(completion,self.timing,self.geometry),)
        if isinstance(token,Moment): self.actions=prune_history(self.actions,self.timing.second(grammar.state.cursor))
        self.grammar=grammar
        self._refresh()
        return completion,Decision()


class Materializer(EventMaterializer):
    def __init__(self,start,end,commit=None,*,no_placeholders=True):
        super().__init__(start,end,commit,markers=True)
        if no_placeholders:
            from .compact_schema import PhysicalGrammar as Grammar
        else: Grammar=PhysicalGrammar
        self.grammar=Grammar(start,end)

    def start_hold(self):
        g=self.grammar; n=self.pending
        self.pending=Note.make('touch_hold' if g.zone is not None else 'hold',n.pad,
            **({'firework':n.firework} if g.zone is not None else {'ex':n.ex,'break_':n.break_}))
        self.flush()

    def accept(self,token):
        t,v=token.type,token.value
        g=self.grammar
        if getattr(g,'no_placeholders',False):
            f=SLIDE_WAIT if t==SLIDE_WAIT_CONT else CONT_FAMILY.get(t,t)
            if v not in g.allowed_values(f):
                from maidata.chart.types import ChartError
                raise ChartError(f'illegal compact token {token} in {g.phase}')
            if f==8:
                if g.phase in ('after_pad','after_flags'): self.start_hold()
                if v!=96: self.emit(HoldDuration(g.duration+v+1))
                g.accept(token); return
            if t==TOUCH_PAD: self.flush()
            if t in (SLIDE_WAIT,SLIDE_WAIT_CONT) and g.phase=='route_chain': self.flush()
        if t in (*STATES,SLIDE_WAIT,SLIDE_WAIT_CONT):
            g.accept(token); return
        if t==TOUCH:
            self.flush(); g.accept(token); return
        if t==TOUCH_PAD:
            g.accept(token); self.pending=Note.make('touch',TOUCH_PADS[v]); return
        if t==11 and v!=96:
            self.emit(SlideDuration(g.duration+v+1,g.wait_ticks)); g.accept(token); return
        super().accept(token)


def action_value(state,completion):
    basic=state._basic(completion)
    if basic: return outside(basic[0])
    action=materialize(completion,state.timing,state.geometry)
    conflicts=action_conflicts(state.actions,action)
    spans=[(action.press_start,action.press_end)]+([(action.launch,action.release)] if action.track else [])
    inherited=capacity_conflicts(state.actions,spans) if any(c.rule=='capacity' for c in conflicts) else frozenset()
    return Evaluation(Evidence(conflicts=conflicts),Evidence(conflicts=conflicts-inherited))


class PhysicalOracle(Oracle):
    def __init__(self,timing,geometry,duration,start,end,**kwargs):
        super().__init__(timing_v6(timing),geometry,duration,start,end,markers=True,profile=kwargs.get('profile'))
        self.physical=State(self.timing,geometry,audio_end_seconds=duration)
        self.physical.inject_boundary(Moment(0),start_tick=start*96)
        self.state=Materializer(start,end,self._commit,no_placeholders=kwargs.get('no_placeholders',True))
        self._duration_support=None
        self.touch_context=()
        self.teacher_mode=bool(kwargs.get('teacher_mode',False))
        self.physical_enabled=kwargs.get('physical_enabled',True)
        self.state_tokens=False
        self._position_prefix=None
        self._position_cache={}
        self._flushed_prefix=None
        self._hold_prefix=None

    def fork(self,*,preserve_support=False):
        result=super().fork(preserve_support=preserve_support)
        state=State(self.timing,self.physical.geometry,audio_end_seconds=self.physical.audio_end_seconds)
        state.restore(self.physical.snapshot()); result.physical=state
        result.state.commit=result._commit
        result._position_prefix=None
        result._position_cache={}
        result._flushed_prefix=None
        result._hold_prefix=None
        return result

    def _flushed(self):
        if self._flushed_prefix is None:
            self._flushed_prefix=self.fork()
            self._flushed_prefix.state.flush()
        return self._flushed_prefix

    def _commit(self,tick,token):
        if isinstance(token,SlideDuration):
            self.physical.grammar.state=replace(self.physical.grammar.state,wait_ticks=token.wait_ticks)
        super()._commit(tick,token)

    def position(self,pad,touch=False,tick=None,*,pool_touch=False):
        p=self.physical; g=self.state.grammar
        tick=g.tick if tick is None else tick
        context=self.touch_context if tick==g.tick and (touch or g.phase=='prefix' or pool_touch) else ()
        prefix=self._position_prefix
        identity=(p.grammar.state.owner,p.grammar.state.moment)
        if prefix is None or prefix[0] is not p.actions or prefix[1] is not context or prefix[2]!=identity:
            self._position_prefix=(p.actions,context,identity)
            self._position_cache={}
            from maidata.physical.touch_v6 import inward, motion_windows, button_neighborhood
            from maidata.physical.geometry import pad_bit
            affected=whole=0
            for a in (*p.actions,*context):
                affected |= pad_bit(a.pad)
                if not a.touch: affected |= button_neighborhood(a.pad,center=True)
                for m in a.motions:
                    for _,_,bits in motion_windows(m): whole |= bits
            self._touch_affected=affected|inward(whole)
        from maidata.physical.geometry import pad_bit
        key=(tick,touch,None if touch and not self._touch_affected&pad_bit(pad) else pad,self.teacher_mode)
        if key in self._position_cache:return self._position_cache[key]
        if context:
            p=copy(p)
            known={(a.owner,a.tick) for a in p.actions}
            p.actions+=tuple(a for a in context if (a.owner,a.tick) not in known)
        completion=Completion(p.grammar.state.owner+1,p.grammar.state.moment+1,tick,Note.make('touch' if touch else 'tap',pad))
        result=action_value(p,completion)
        if touch and not self.teacher_mode and not result.raw.basic:
            result=self._touch_completion(p,completion,result)
        self._position_cache[key]=result
        return result

    def _touch_completion(self,state,completion,result):
        """Existential row completion; EOM evaluates the actual accepted row."""
        capacity={c for c in result.raw.conflicts if c.rule=='capacity' and ('touch_pool',) in c.participants}
        if not capacity: return result
        from maidata.physical.touch_v6 import inward, motion_windows
        from maidata.physical.geometry import pad_bit
        from maidata.chart.types import TOUCH_PADS as PROFILE_PADS
        current=materialize(completion,state.timing,state.geometry)
        actions=(*state.actions,current); t=current.press_start
        row=[a for a in actions if a.touch and abs(a.press_start-t)<=EPS]
        if len(row)>=3 or not all(a.short for a in row): return result
        mask=0
        for a in actions:
            if a.track and a.launch-EPS<=t<=a.actual_end+EPS:
                for m in a.motions:
                    for _,_,bits in motion_windows(m): mask |= bits
        mask=inward(mask)
        if any(not mask&pad_bit(a.pad) for a in row): return result
        extra=[]; used={a.pad for a in row}; owner=max(a.owner for a in actions)+1
        for pad in TOUCH_PADS:
            if pad in used or not mask&pad_bit(pad): continue
            if self.profile is not None and not self.profile.touch_pad33_mask[PROFILE_PADS.index(pad)]: continue
            candidate=replace(current,owner=owner+len(extra),pad=pad)
            if action_conflicts((*actions,*extra),candidate,check_capacity=False): continue
            extra.append(candidate)
            if len(row)+len(extra)==3: break
        if len(row)+len(extra)<3: return result
        remaining=capacity_conflicts((*actions,*extra),[(t,current.press_end)])
        cleared=capacity-remaining
        if not cleared: return result
        return Evaluation(replace(result.raw,conflicts=result.raw.conflicts-cleared,deferred=True),
                          replace(result.penalty,conflicts=result.penalty.conflicts-cleared,deferred=True))

    def duration_value(self,ticks,family=None):
        g=self.state.grammar
        token=HoldDuration(ticks) if (family==8 or family is None and g.phase=='hold') else SlideDuration(ticks,g.wait_ticks)
        _,c=self.physical._trial(token)
        result=action_value(self.physical,c)
        witnesses=frozenset(Conflict('slide_too_fast','hold_slide_final_area',
            (('press',v['hold_owner'],v['hold_tick'],v['hold_pad']),('track',v['slide_owner'],v['track_index'],v['slide_tick'])))
            for v in contacts(self.physical,token))
        return Evaluation(replace(result.raw,conflicts=result.raw.conflicts|witnesses),replace(result.penalty,conflicts=result.penalty.conflicts|witnesses))

    def duration_support(self,family=None):
        if self._duration_support is not None: return self._duration_support
        p=self.physical; g=self.state.grammar; timing=self.timing
        hold=family==8 or family is None and g.phase=='hold'
        origin=g.tick+(0 if hold else g.wait_ticks)
        upper=math.floor(timing.second_to_beat(p.audio_end_seconds)*96-origin+1e-7)
        roots=[]
        fixed=_times(p.actions)|{timing.second(g.tick),timing.second(origin),timing.second(g.tick+2)}
        if upper>=1:
            constructor=lambda n: HoldDuration(n) if hold else SlideDuration(n,g.wait_ticks)
            _,c1=p._trial(constructor(1)); _,c2=p._trial(constructor(2))
            a=materialize(c1,timing,p.geometry); b=materialize(c2,timing,p.geometry)
            end1=timing.second(origin+1); end2=timing.second(origin+2)
            moving=[(x,y) for m,n in zip(a.motions,b.motions) for x,y in ((m.start,n.start),(m.end,n.end),*((u[0],v[0]) for u,v in zip(m.entries,n.entries)))]
            moving.append((end1,end2))
            for x,y in moving:
                slope=(y-x)/(end2-end1)
                if slope:
                    for t in fixed:
                        end=end1+(t-x)/slope
                        roots.append(timing.second_to_beat(end)*96-origin)
            for t in fixed: roots.append(timing.second_to_beat(t)*96-22-origin)
            if hold:
                for old in p.actions:
                    if old.track and old.motions[-1].entries:
                        roots.append(timing.second_to_beat(old.motions[-1].entries[-1][0])*96-96-origin)
            elif a.motions[-1].entries:
                x=a.motions[-1].entries[-1][0]; y=b.motions[-1].entries[-1][0]
                slope=(y-x)/(end2-end1)
                if slope:
                    for hold in p.actions:
                        if hold.kind in ('hold','touch_hold'):
                            for t in (hold.press_start,timing.second(timing.second_to_beat(hold.press_end)*96-22+96)):
                                roots.append(timing.second_to_beat(end1+(t-x)/slope)*96-origin)
        query=self.duration_value if family is None else lambda n:self.duration_value(n,family)
        rows=_partition(roots,1,upper,query,lambda e:e) if upper>=1 else []
        self._duration_support=dict(domain=[1,upper],intervals=rows,outside_basic='duration_outside_audio')
        return self._duration_support

    def _route_value(self,routes,*,sealed=False):
        from maidata.chart.routes import chain_paths
        p=self.physical; g=self.state.grammar; s=p.grammar.state
        pending=isinstance(self.state.pending,Note)
        owner=s.owner+int(pending); index=0 if pending else s.track_count
        start='A'+str(g.head_pad+1)
        conflicts=set(branch_conflicts(p.actions,owner,index,g.tick,wifi=routes==(70,)))
        active_routes=() if g.phase=='after_slide' and not routes else (routes or g.routes)
        conflicts.update(self._first_path_conflicts(active_routes,1))
        if routes and (sealed or routes==(70,)):
            paths=chain_paths(start,routes); t=self.timing.second(g.tick)
            for a in p.actions:
                if a.track and a.pad==start and a.path_key==paths and a.press_start-EPS<=t<=a.launch+EPS:
                    conflicts.add(Conflict('slide_too_fast','same_path',(('track',owner,index,g.tick),track_key(a))))
        evidence=Evidence(conflicts=frozenset(conflicts))
        return Evaluation(evidence,evidence)

    def _first_path_conflicts(self,routes,wait):
        from maidata.chart.routes import chain_paths
        from maidata.physical.geometry import Geometry
        if not routes:return frozenset()
        g=self.state.grammar; p=self.physical; s=p.grammar.state
        pending=isinstance(self.state.pending,Note)
        owner=s.owner+int(pending); index=0 if pending else s.track_count
        first=chain_paths('A'+str(g.head_pad+1),routes)[0]; path=p.geometry.paths[first]
        t=self.timing.second(g.tick); launch=self.timing.second(g.tick+wait)
        return frozenset(Conflict('slide_too_fast','path_coverage',
            (('track',owner,index,g.tick),track_key(a)),(0,j))
            for a in p.actions for j,m in enumerate(a.motions)
            if not path.exempt and not m.exempt and Geometry.covers(m.path,first) and m.start<=launch+EPS and m.end>=t-EPS)

    def _wait_value(self,wait,sealed=frozenset()):
        g=self.state.grammar
        conflicts=self._first_path_conflicts(g.routes,wait)
        conflicts |= sealed
        evidence=Evidence(conflicts=conflicts)
        return Evaluation(evidence,evidence)

    def _touch_pads(self):
        if self.profile is None:return TOUCH_PADS
        from maidata.chart.types import TOUCH_PADS as PROFILE_PADS
        return tuple(p for p in TOUCH_PADS if self.profile.touch_pad33_mask[PROFILE_PADS.index(p)])

    def branch_value(self,family):
        g=self.state.grammar
        if family in (8,9):
            if self.timing.second(g.tick+(1 if family==8 else 2))>self.physical.audio_end_seconds+EPS:
                return outside('duration_outside_audio')
            return self._route_value(()) if family==9 and self.physical_enabled else Evaluation()
        if self.physical_enabled: return self._route_value(g.routes,sealed=True)
        return project(self.basic_support(SLIDE_WAIT).values())

    def _support(self,family):
        if not self.physical_enabled:
            return self.basic_support(family)
        g=self.state.grammar; values=g.allowed_values(family)
        if family in STATES: return {v:Evaluation() for v in values}
        if family==12:
            trial=self.fork(); trial.state.flush()
            actions=trial.physical.actions; t=self.timing.second(g.tick)
            touches=[a for a in actions if a.touch and abs(a.press_start-t)<=EPS]
            if not touches: return {0:Evaluation()}
            spans=[(a.press_start,a.press_end) for a in touches]
            conflicts=capacity_conflicts(actions,spans)
            inherited=capacity_conflicts(tuple(a for a in actions if not (a.touch and abs(a.press_start-t)<=EPS)),spans)
            conflicts=conflicts-inherited
            return {0:Evaluation(Evidence(conflicts=conflicts),Evidence())}
        if family==0:
            lower=g.beat*96+g.offset+1
            upper=min(g.beat*96+95,math.ceil(self.timing.second_to_beat(self.physical.audio_end_seconds)*96)-1)
            def roots(touch_windows):
                fixed=_times(self.physical.actions,touch_windows=touch_windows)|{0.}
                return [x for t in fixed for tick in (self.timing.second_to_beat(t)*96,) for x in (tick,tick-2)]
            def onset(tick,buttons_only=False):
                button=[]
                for pad in BUTTON_PADS:
                    value=self.position(pad,tick=tick); button.append(value)
                    if not value.raw.basic and not value.raw.conflicts:return Evaluation()
                if buttons_only:return None
                penalty=project(button).penalty
                touch=[]
                for pad in self._touch_pads():
                    value=self.position(pad,True,tick=tick); touch.append(value)
                    if not value.raw.basic and not value.raw.conflicts:return Evaluation(Evidence(),penalty)
                return Evaluation(project(button+touch).raw,penalty)
            rows=[]; fine=None
            coarse=_cuts(roots(False),lower,upper) if upper>=lower else []
            for a,b in zip(coarse,coarse[1:]):
                clean=onset((a+b-1)//2,buttons_only=True)
                if clean is not None: rows.append([a,b-1,clean])
                else:
                    if fine is None:fine=roots(True)
                    rows.extend(_partition(fine,a,b-1,onset,lambda e:e))
            support=dict(intervals=rows,outside_basic='onset_outside_audio')
            return {v:Evaluation() if v==96 else rebind_onset(interval_value(support,g.beat*96+v),self.physical.grammar.state.owner+1,g.beat*96+v) for v in values}
        if family==18:
            trial=self.fork(); trial.state.flush()
            button=[trial.position(p) for p in BUTTON_PADS]
            penalty=project(button).penalty
            if any(e==Evaluation() for e in button):return {0:Evaluation(Evidence(),penalty)}
            touch=[]
            for pad in trial._touch_pads():
                value=trial.position(pad,True);touch.append(value)
                if value==Evaluation():return {0:Evaluation(Evidence(),penalty)}
            return {0:Evaluation(project(button+touch).raw,project(button).penalty)}
        if family==5:
            touch=[]
            for pad in self._touch_pads():
                value=self.position(pad,True);touch.append(value)
                if value==Evaluation():return {0:value}
            return {0:project(touch)}
        position=self._flushed() if getattr(g,'no_placeholders',False) and g.phase in ('after_pad','after_flags','after_note','after_slide') and family in (4,TOUCH_PAD) else self
        if family==TOUCH_PAD: return {v:position.position(TOUCH_PADS[v],True) for v in values}
        if family==4 and g.phase!='route_pad': return {v:position.position(BUTTON_PADS[v]) for v in values}
        if family in (7,17):
            if self.timing.second(g.tick+(1 if family==7 else 2))>self.physical.audio_end_seconds+EPS:
                return {v:outside('duration_outside_audio') for v in values}
            return {v:(self._route_value(()) if family==17 else Evaluation()) for v in values}
        if family==9:
            if getattr(g,'no_placeholders',False) and self.timing.second(g.tick+2)>self.physical.audio_end_seconds+EPS:
                return {v:outside('duration_outside_audio') for v in values}
            return {v:project(self._route_value(g.routes+(r,)) for r in shape_route_table()[v][g.pad] if r>=0) for v in values}
        if family==4 and g.phase=='route_pad':
            return {v:self._route_value(g.routes+(shape_route_table()[g.shape][g.pad][v],)) for v in values}
        if family==10:return {v:self._route_value(g.routes,sealed=True) for v in values}
        if family in (8,11):
            duration_oracle=self
            if family==8 and g.phase in ('after_pad','after_flags'):
                if self.timing.second(g.tick+1)>self.physical.audio_end_seconds+EPS:
                    return {v:outside('duration_outside_audio') for v in values}
                if self._hold_prefix is None:
                    self._hold_prefix=self.fork()
                    self._hold_prefix.state.start_hold()
                duration_oracle=self._hold_prefix
            support=duration_oracle.duration_support(family if getattr(g,'no_placeholders',False) else None)
            self._duration_support=support
            return {v:(interval_value(support,g.duration+v+1) if v!=96 else
                project(e for lo,hi,e in support['intervals'] if hi>=g.duration+97)) for v in values}
        if family==SLIDE_WAIT:
            sealed=self._route_value(g.routes,sealed=True).raw.conflicts if getattr(g,'no_placeholders',False) else frozenset()
            cap=math.floor(self.timing.second_to_beat(self.physical.audio_end_seconds)*96-g.tick)-1
            return {v:self._wait_value(g.wait_elapsed+v+1,sealed) if g.wait_elapsed+v+1<=cap else outside('wait_outside_audio') for v in values}
        return {v:Evaluation() for v in values}

    def support(self,family):
        if family not in self._supports: self._supports[family]=self._support(family)
        return self._supports[family]

    def evaluate(self,token):
        family=SLIDE_WAIT if token.type==SLIDE_WAIT_CONT else CONT_FAMILY.get(token.type,token.type)
        return self.support(family).get(token.value,outside('grammar'))

    def accept(self,token,evaluation=None):
        self.state.accept(token)
        g=self.state.grammar
        if token.type==SLIDE_WAIT:
            self.physical.grammar.state=replace(self.physical.grammar.state,wait_ticks=g.wait_ticks)
        self._supports={}
        self._position_prefix=None
        self._position_cache={}
        self.touch_context=()
        self._flushed_prefix=None
        self._hold_prefix=None
        if token.type not in (14,15): self._duration_support=None
