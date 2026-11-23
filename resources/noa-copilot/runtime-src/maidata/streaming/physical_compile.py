"""v8 supervision with explicit per-category fractional field responsibility."""
from collections import defaultdict
from dataclasses import replace
import math
import torch
from maidata.chart.types import Note, Route, HoldDuration, SlideDuration, Moment, Completion, BUTTON_PADS
from maidata.chart.grammar import Grammar
from maidata.chart.routes import shape_route_table
from maidata.physical.actions import materialize
from maidata.physical.conflicts import action_conflicts
from maidata.physical.attribution import Evaluation, Evidence, project
from maidata.physical.rules import CATEGORIES, EPS, prune_history
from maidata.physical.geometry import pad_bit
from .physical_schema import *
from .physical_codec import encode, note_blocks
from .v6_oracle import PhysicalOracle
from .sequence import BOUNDARY, CONT_FAMILY
from .codec import input_row
from .attention import layouts
from .version import values as version_values, probe_heads

ATTRIBUTION_CONTRACT='whole-moment-causal-budget.v6.2'


def compile_chart(chart,geometry,duration,*,start=None,end=None,audio_beats=None,audio_valid=None,
                  weighted_spans=(),multiplier=1.,state_ce_weight=.25,physical_enabled=True,state_tokens=False,audio_before=8.,audio_after=None,no_placeholders=True):
    from . import PHYSICAL_V7_CACHE_SCHEMA, PHYSICAL_V8_CACHE_SCHEMA, PHYSICAL_ARCHITECTURE, COMPACT_ARCHITECTURE
    from maidata.physical.touch_v6 import CONTRACT
    start=min(0,chart.groups[0].tick//96) if start is None and chart.groups else (start or 0)
    end=max(start+1,math.ceil(chart.timing.second_to_beat(duration))) if end is None else end
    audio_after=8. if audio_after is None else audio_after
    audio_beats=torch.arange(start-math.ceil(audio_before),end+math.ceil(audio_after),1/8)+1/16 if audio_beats is None else audio_beats
    tokens,chart=encode(chart,geometry,duration,start,end,physical_enabled=physical_enabled,state_tokens=False,no_placeholders=no_placeholders)
    from maidata.version_mask import profile_for_chart
    profile=profile_for_chart(chart.song_id,chart.difficulty,chart.version)
    oracle=PhysicalOracle(chart.timing,geometry,duration,start,end,profile=profile,teacher_mode=True,physical_enabled=physical_enabled,state_tokens=False,no_placeholders=no_placeholders)
    inputs=[input_row(EventToken(BOUNDARY),oracle.state.grammar,chart.timing)]
    labels=defaultdict(list); probes=[]; notes=[]; current=None; onset=None; chord=None
    ledger=set(); field_rows=defaultdict(list); field_id=0; interval=None; time_field=('time',0); position_field=None
    grammar=Grammar(chart.timing); grammar.inject_boundary(Moment(0),start_tick=start*96)
    completions=[]
    for group in chart.groups if physical_enabled else ():
        if group.tick<start*96 or group.tick>=end*96: continue
        if grammar.state.cursor!=group.tick: grammar.accept(Moment(chart.timing.delta_ms(grammar.state.cursor,group.tick)))
        for token in group.tokens:
            c=grammar.accept(token)
            if c: completions.append(c)
    actions=[materialize(c,chart.timing,geometry) for c in completions]
    by_owner=defaultdict(list)
    for c,a in zip(completions,actions): by_owner[c.owner].append((c,a))
    touches_by_tick=defaultdict(list)
    for a in actions:
        if a.touch: touches_by_tick[a.tick].append(a)
    counts_by_tick=defaultdict(int)
    for group in chart.groups: counts_by_tick[group.tick]=len(note_blocks(group))

    def add(key,width,target,allowed,index,weight,*,main=True,physical=False,values=None,identity=None):
        physical=physical and physical_enabled
        mask=torch.zeros(width,dtype=torch.bool); mask[list(allowed)]=True; mask[target]=True
        row=dict(indices=index,target=target,allowed=mask,main=weight if main and int(mask.sum())>1 else 0.,
                 no_history=True,_field=identity,_diagnostic=True)
        if physical_enabled:
            row.update(penalty=torch.zeros(width,4,dtype=torch.bool),category_scale=torch.zeros(4),
                 _contact=torch.tensor([any(c.rule=='hold_slide_final_area' for c in (values or {}).get(v,Evaluation()).penalty.conflicts) for v in range(width)]),
                 _coverage=torch.tensor([any(c.rule=='path_coverage' for c in (values or {}).get(v,Evaluation()).penalty.conflicts) for v in range(width)]),
                 _tail=torch.tensor([any(c.rule=='path_guard' for c in (values or {}).get(v,Evaluation()).penalty.conflicts) for v in range(width)]))
        if values and physical and int(mask.sum())>1:
            for v,e in values.items():
                for c in e.penalty.conflicts:
                    if c in ledger or c.rule in ('path_guard','launch_guard','hold_slide_final_area'): continue
                    row['penalty'][v,CATEGORIES.index(c.category)]=True
            effective=row['penalty'][mask].any(0)&~row['penalty'][mask].all(0)
            row['penalty'][:,~effective]=False
            row['category_scale'][effective]=weight
        labels[key].append(row); ref=(key,len(labels[key])-1)
        if physical and int(mask.sum())>1 and (key!=SLIDE_WAIT or row['penalty'].any()): field_rows[identity].append(ref)
        return ref

    for i,token in enumerate(tokens):
        g=oracle.state.grammar; family=SLIDE_WAIT if token.type==SLIDE_WAIT_CONT else CONT_FAMILY.get(token.type,token.type)
        weight=multiplier if any(a<=g.tick<b for a,b in weighted_spans) else 1.
        note_start=family in (4,TOUCH_PAD) and g.phase!='route_pad'
        previous_current=current
        if family in (TOUCH,TOUCH_PAD) and (no_placeholders or g.phase in ('prefix','touch_pad')):
            oracle.touch_context=tuple(a for a in touches_by_tick[g.tick] if a.owner!=len(notes))
        available={f:set(version_values(g,f,profile)) for f in g.allowed_types()}
        supports={f:{v:e for v,e in oracle.support(f).items() if v in available[f] or (f==family and v==token.value)}
                  for f in available if not (no_placeholders and f!=family and f in (8,9,SLIDE_WAIT))}
        supports={f:s for f,s in supports.items() if s}
        if family in (8,11,SLIDE_WAIT):
            if interval is None: field_id+=1; interval=field_id
            identity=interval
        elif family==0:identity=time_field
        elif family==4 and g.phase=='route_pad':identity=position_field
        else:
            field_id+=1; identity=field_id
            if family==9:position_field=identity
        type_values={f:project(s.values()) for f,s in supports.items()}
        if no_placeholders:
            for f in (8,9,SLIDE_WAIT):
                if f in available and (available[f] or f==family):
                    type_values[f]=oracle.branch_value(f)
            if TOUCH_PAD in type_values and g.phase!='prefix':
                type_values[TOUCH_PAD]=replace(type_values[TOUCH_PAD],penalty=Evidence())
        type_allowed=[f for f,e in type_values.items() if not e.raw.basic]
        type_ref=None
        if len(set(type_allowed)|{family})>1:
            type_ref=add('type',len(FAMILIES),family,type_allowed,i,weight,physical=g.phase!='prefix',values=type_values,identity=('type',i))
            if physical_enabled and counts_by_tick[g.tick]>=2 and 4 in supports:
                r=labels['type'][type_ref[1]]
                touch_family=TOUCH_PAD if no_placeholders else TOUCH
                prefix=g.phase=='prefix' and touch_family in supports
                compact_touch=no_placeholders and family==TOUCH_PAD and g.phase!='prefix' and not r['penalty'][4,0]
                if prefix or compact_touch:
                    button=project(supports[4].values()) if prefix else project(
                        oracle._flushed().position(BUTTON_PADS[v],pool_touch=True) for v in supports[4])
                    touch=project(supports[touch_family].values())
                    if any(c.category=='multitouch' and c.rule=='capacity' for c in button.penalty.conflicts) and not any(c.category=='multitouch' for c in touch.penalty.conflicts):
                        r['penalty'][4,0]=True; r['category_scale'][0]=weight
                        if prefix: field_rows[('type',i)].append(type_ref)
        values=supports[family]
        ref=None
        if family==6:
            heads=(0,1) if g.flag_context=='button' else (2,) if g.flag_context=='touch' else (3,)
            for bit,head in enumerate(heads):
                ref=add('flag'+str(head),2,(token.value>>bit)&1,(0,1),i,weight)
                if current is not None and head==0: current['ex']=ref
        elif WIDTHS[family]>1:
            allowed=[v for v,e in values.items() if not e.raw.basic]
            ref=add(family,WIDTHS[family],token.value,allowed,i,weight,
                    physical=family in (0,4,TOUCH_PAD,8,9,11) or (family==SLIDE_WAIT and any(e.penalty.conflicts for e in values.values())),values=values,identity=identity)
        if family==0 and token.value!=96:
            onset=ref; chord=None; ledger.clear()
        if family==18: chord=type_ref
        if note_start and (no_placeholders or g.phase in ('prefix','touch_pad')):
            if no_placeholders: chord=type_ref if g.phase!='prefix' else None
            current=dict(owner=len(notes),tick=g.tick,pad=ref,onset=chord or onset,weight=weight,tracks=[],hold=[],ex=None)
            notes.append(current)
        elif family==9 and current is not None:
            if (g.phase!='route_chain' if no_placeholders else g.first_route): current['tracks'].append(dict(segments=[],duration=[],wait=[]))
            current['tracks'][-1]['segments'].append(dict(shape=ref,pad=None))
        elif family==4 and g.phase=='route_pad': current['tracks'][-1]['segments'][-1]['pad']=ref
        elif family==8: current['hold'].append(ref)
        elif family==11: current['tracks'][-1]['duration'].append(ref)
        elif family==SLIDE_WAIT: current['tracks'][-1]['wait'].append(ref)
        if 6 in supports and family!=6:
            for head in probe_heads(g,profile): probes.append((i,head,weight))
            probe_current=previous_current if no_placeholders and note_start else current
            if physical_enabled and probe_current is not None and g.flag_context=='button' and probe_current['ex'] is None:
                probe_current['ex']=add('flag0',2,0,(0,1),i,weight,main=False)
        for label_ref,evidence in ((ref,values[token.value]),(type_ref,type_values[family])):
            if not physical_enabled or label_ref is None: continue
            r=labels[label_ref[0]][label_ref[1]]
            charged={CATEGORIES[c] for c in range(4) if r['category_scale'][c]>0 and r['penalty'][r['target'],c]}
            ledger.update(c for c in evidence.penalty.conflicts if c.category in charged)
        context=g.flag_context; oracle.accept(token)
        inputs.append(input_row(token,oracle.state.grammar,chart.timing,context=context))
        if family in (8,11,SLIDE_WAIT) and token.value!=96: interval=None
        if family==0 and token.value!=96:time_field=('time',i+1)
    for refs in field_rows.values():
        if refs[0][0] in (0,8,11,SLIDE_WAIT):
            for key,row in refs: labels[key][row]['category_scale']/=len(refs)
    physical_count=sum(max(labels[k][i]['main'] for k,i in refs) for refs in field_rows.values())
    if physical_enabled:redistribute(notes,by_owner,actions,labels,chart.timing,geometry)
    budgets={identity:max(labels[k][i]['main'] for k,i in refs) for identity,refs in field_rows.items()}
    totals=defaultdict(lambda:torch.zeros(4))
    for rows in labels.values() if physical_enabled else ():
        for r in rows:totals[r['_field']]+=r['category_scale']*r['penalty'].any(0)
    for rows in labels.values() if physical_enabled else ():
        for r in rows:
            if r['_field'] in budgets:
                limit=budgets[r['_field']]; total=totals[r['_field']]
                r['category_scale']*=torch.where(total>limit,limit/total.clamp_min(1e-12),1.)
    packed={}
    for key,rows in labels.items():
        if key=='flag0':
            rows=[r for r in rows if r['main'] or (physical_enabled and r['category_scale'].any())]
            if not rows: continue
        if key in (0,8,11):
            diagnostic=[r for r in rows if r['_diagnostic']]
            ends={id(r):j==len(diagnostic)-1 or r['_field']!=diagnostic[j+1]['_field'] for j,r in enumerate(diagnostic)}
            for r in rows:
                r['time_rows']=r['_diagnostic']; r['time_ends']=ends.get(id(r),False)
        for r in rows:
            for name in ('_contact','_coverage','_tail','_field','_diagnostic'):r.pop(name,None)
        packed[key]={name:torch.stack([r[name] for r in rows]) if isinstance(rows[0][name],torch.Tensor) else
                     torch.tensor([r[name] for r in rows],dtype=torch.bool if name in ('no_history','time_rows','time_ends') else torch.float32 if name=='main' else torch.long)
                     for name in rows[0]}
    tensor_inputs={k:torch.tensor([r[k] for r in inputs[:-1]],dtype=torch.long if k in ('types','vocab_ids') else torch.float32) for k in inputs[0]}
    return dict(schema=PHYSICAL_V8_CACHE_SCHEMA if no_placeholders else PHYSICAL_V7_CACHE_SCHEMA,architecture=COMPACT_ARCHITECTURE if no_placeholders else PHYSICAL_ARCHITECTURE,tokens=tensor_inputs,labels=packed,
        negative_probes=dict(indices=torch.tensor([p[0] for p in probes],dtype=torch.long),heads=torch.tensor([p[1] for p in probes],dtype=torch.long),weights=torch.tensor([p[2] for p in probes],dtype=torch.float32)),
        counts=dict(physical=physical_count),**layouts(tensor_inputs['causal_beats'],tensor_inputs['slide_beats'],audio_beats,audio_valid,audio_before=audio_before,audio_after=audio_after),
        metadata=dict(**(dict(placeholders=False) if no_placeholders else {}),physical_enabled=physical_enabled,start=start,end=end,tokens=len(tokens),audio_first=float(audio_beats[0]),audio_length=len(audio_beats),event_count=0,duration_contract='global-tick96.v1',physical_attribution_contract=ATTRIBUTION_CONTRACT,physical_touch_contract=CONTRACT,state_tokens=False))


def redistribute(notes,by_owner,actions,labels,timing,geometry):
    """Offline counterfactual masks; no teacher content enters runtime inputs."""
    def row(ref): return labels[ref[0]][ref[1]]
    assigned={}
    def assign(ref,category,bad,weight):
        if ref is None: return
        original=row(ref); c=CATEGORIES.index(category)
        bad=torch.as_tensor(bad,dtype=torch.bool)&original['allowed']
        if not bad.any() or bad[original['allowed']].all() or weight<=0:return
        identity=(ref,c)
        if identity not in assigned:
            assigned[identity]=[]
            if original['penalty'][:,c].any() and original['category_scale'][c]>0:
                assigned[identity].append((original['penalty'][:,c].clone(),float(original['category_scale'][c])))
        assigned[identity].append((bad,float(torch.tensor(weight,dtype=torch.float32))))
    def target(ref,category,weight):
        if ref is None:return
        r=row(ref); bad=torch.zeros_like(r['allowed']); bad[r['target']]=True
        assign(ref,category,bad,weight)
    def endpoint(note,track,segment=-1):
        r=note['tracks'][track]['segments'][segment]
        return r['pad'] if int(row(r['pad'])['allowed'].sum())>1 else r['shape']
    def final_contact(hold,slide):
        if hold.kind not in ('hold','touch_hold') or not slide.track or not slide.motions[-1].entries:return False
        entered,mask=slide.motions[-1].entries[-1]
        return bool(mask&pad_bit(hold.pad) and hold.press_start<=entered+EPS and timing.second_to_beat(entered)*96-96<timing.second_to_beat(hold.press_end)*96-22-EPS)
    def alternate(c,pad):
        return materialize(replace(c,note=replace(c.note,pad=pad)),timing,geometry)
    def endpoint_mask(completion,ref,test,segment=-1):
        from maidata.chart.routes import chain_paths, route_shape_end
        routes=completion.track.routes
        segment=segment if segment>=0 else len(routes)-1
        start=completion.note.pad if segment==0 else 'A'+chain_paths(completion.note.pad,routes[:segment])[-1][-1]
        shape,_=route_shape_end(start,routes[segment]); start_index=int(start[1])-1
        result=[]
        for v in range(len(row(ref)['allowed'])):
            possible=([shape_route_table()[shape][start_index][v]] if ref[0]==4 else
                      [r for r in shape_route_table()[v][start_index] if r>=0])
            possible=[r for r in possible if r>=0]
            result.append(bool(possible) and all(test(materialize(replace(completion,track=replace(completion.track,routes=routes[:segment]+(r,)+routes[segment+1:])),timing,geometry)) for r in possible))
        return result
    history=(); previous=()
    for note in notes:
        own=by_owner[note['owner']]; c,a=own[0]
        history=prune_history((*history,*previous),timing.second(a.tick))
        previous=tuple(action for _,action in own)
        padrow=row(note['pad'])
        pads=TOUCH_PADS if note['pad'][0]==TOUCH_PAD else tuple('A'+str(i+1) for i in range(8))
        candidates=[]; trials=[]
        tail_possible=not a.touch and any(
            entered<a.press_start-EPS and a.press_start<=entered+.2+EPS
            for old in history for segment,motion in enumerate(old.motions)
            for entry,(entered,_) in enumerate(motion.entries)
            if not (old.physical_v6 and segment==0 and entry==0))
        for pad in pads if tail_possible else ():
            trial=materialize(Completion(a.owner,a.moment,a.tick,Note.make('touch' if a.touch else 'tap',pad)),timing,geometry)
            trials.append(trial)
            candidates.append(action_conflicts(history,trial,check_capacity=False))
        if any(c.rule=='path_guard' for values in candidates for c in values):
            candidates=[action_conflicts(history,trial) for trial in trials]
        else:candidates=[]
        free=any(bool(padrow['allowed'][i]) and not any(c.category in ('multitouch','overlap','tap_on_slide') for c in conflicts) for i,conflicts in enumerate(candidates))
        if not a.touch and free:
            assign(note['pad'],'tap_on_slide',[any(c.rule=='path_guard' for c in v) for v in candidates],note['weight'])
        elif candidates:
            conflicts=action_conflicts(history,replace(a,ex=False),check_capacity=False)
            tails=[v for v in conflicts if v.rule=='path_guard']
            evidence=[]
            for v in tails:
                track=v.participants[1]; old=next(x for x in history if x.owner==track[1] and x.track_index==track[2] and x.track)
                entered=old.motions[v.detail[0]].entries[v.detail[1]][0]
                evidence.append((v,old,a.press_start-entered>.150+EPS))
            if evidence:
                wide=all(e[2] for e in evidence)
                share=note['weight']/(3 if wide else 1)
                onset_ref=note['onset']
                if onset_ref[0]==0:
                    assign(onset_ref,'tap_on_slide',row(onset_ref)['_tail'],share)
                else: target(onset_ref,'tap_on_slide',share)
                if wide:
                    targets=defaultdict(dict)
                    for v,old,_ in evidence:
                        ref=endpoint(notes[old.owner],old.track_index,v.detail[0])
                        prior=next(c for c,x in by_owner[old.owner] if x.track_index==old.track_index)
                        mask=torch.tensor(endpoint_mask(prior,ref,lambda slide:any(v.rule=='path_guard' for v in action_conflicts([slide],replace(a,ex=False),check_capacity=False)),segment=v.detail[0]))
                        track_targets=targets[old.owner,old.track_index]
                        track_targets[ref]=track_targets.get(ref,torch.zeros_like(mask))|mask
                    for track_targets in targets.values():
                        for ref,mask in track_targets.items():assign(ref,'tap_on_slide',mask,share/(len(targets)*len(track_targets)))
                    assign(note['ex'],'tap_on_slide',(True,False),share)
        for completion,action in own:
            refs=(note['tracks'][action.track_index]['duration'] if action.track else note['hold'])
            contact=False
            if action.track:
                holds=[old for old in history if old.kind in ('hold','touch_hold')]
                contact=any(final_contact(h,action) for h in holds)
                if contact:
                    ref=endpoint(note,action.track_index)
                    assign(ref,'slide_too_fast',endpoint_mask(completion,ref,lambda trial:any(final_contact(h,trial) for h in holds)),note['weight']/2)
                conflicts=action_conflicts(history,action,check_capacity=False)
                repeated=any(c.rule=='same_path' for c in conflicts)
                coverage=any(c.rule=='path_coverage' and c.detail[0]==0 for c in conflicts)
                wait_refs=note['tracks'][action.track_index]['wait']
                wait_locked=bool(wait_refs) and all(bool(row(r)['_coverage'][row(r)['allowed']].all()) for r in wait_refs)
                if repeated or (coverage and wait_locked):
                    segment=-1 if repeated else 0; ref=endpoint(note,action.track_index,segment)
                    rule='same_path' if repeated else 'path_coverage'
                    assign(ref,'slide_too_fast',endpoint_mask(completion,ref,lambda trial:any(c.rule==rule and (rule=='same_path' or c.detail[0]==0) for c in action_conflicts(history,trial,check_capacity=False)),segment=segment),note['weight'])
            elif action.kind in ('hold','touch_hold'):
                slides=[old for old in history if old.track]
                contact=any(final_contact(action,old) for old in slides)
                if contact:assign(note['pad'],'slide_too_fast',[any(final_contact(alternate(completion,pad),old) for old in slides) for pad in pads],note['weight']/2)
            for ref in refs:
                assign(ref,'slide_too_fast',row(ref)['_contact'],note['weight']/((2 if contact else 1)*len(refs)))
    for (ref,c),contributions in assigned.items():
        original=row(ref); layers=[]; previous=0.
        for level in sorted({w for _,w in contributions}):
            mask=torch.stack([m for m,w in contributions if w>=level]).any(0)
            delta=level-previous; previous=level
            if layers and torch.equal(layers[-1][0],mask):layers[-1]=(mask,layers[-1][1]+delta)
            else:layers.append((mask,delta))
        original['penalty'][:,c]=False;original['category_scale'][c]=0.
        layers=[(mask,weight) for mask,weight in layers if not mask[original['allowed']].all()]
        for j,(mask,weight) in enumerate(layers):
            r=original if j==0 else dict(original,main=0.,no_history=False,_diagnostic=False,penalty=torch.zeros_like(original['penalty']),category_scale=torch.zeros(4))
            r['penalty'][:,c]=mask; r['category_scale'][c]=weight
            if j:labels[ref[0]].append(r)
