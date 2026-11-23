"""v8 chart ingestion and canonical onset order."""
from dataclasses import replace
import math
from maidata.chart.types import Note, Route, HoldDuration, SlideDuration, Moment
from maidata.chart.codec import EncodedChart, encode_chart
from maidata.chart.routes import route_shape_end
from maidata.draft.chart_data import OnsetGroup, ChartData
from maidata.physical.actions import materialize
from maidata.physical.conflicts import action_conflicts
from maidata.chart.grammar import Grammar
from .sequence import EventToken, duration_tokens
from .codec import canonical_groups
from .physical_schema import TOUCH_PADS, TOUCH_PAD, wait_tokens
from .v6_oracle import PhysicalOracle, timing_v6


def snap(value):
    return math.floor(value+.5)


def global_chart(chart):
    """Convert accepted written-BPM durations once; never reinterpret v6 data."""
    if getattr(chart.timing,'global_ticks',False): return chart
    timing=chart.timing; groups=[]
    for group in canonical_groups(chart.groups):
        content=[]
        for token in group.tokens:
            if isinstance(token,(HoldDuration,SlideDuration)):
                slide=isinstance(token,SlideDuration)
                wait=token.wait_ticks if slide else 0
                launch=timing.second(group.tick+wait)
                end=launch+token.ticks*timing.written_tick_seconds(group.tick)
                ticks=max(1,snap(timing.second_to_beat(end)*96)-group.tick-wait)
                token=SlideDuration(ticks,wait) if slide else HoldDuration(ticks)
            content.append(token)
        groups.append(replace(group,tokens=tuple(content)))
    return replace(chart,groups=tuple(groups),timing=timing_v6(timing))


def from_ir(ir):
    """Use original physical seconds before snapping, including custom waits."""
    from copy import deepcopy
    from maidata.time import TimingMap
    from maidata.chart.codec import _chart_tick
    source=deepcopy(ir); timing=TimingMap.from_ir(ir)
    durations=[]
    for moment in source['moments']:
        tick=_chart_tick(moment['position_beats'],position=True)
        for group in moment['groups']:
            for note in group['notes']:
                slide=note['kind'] in ('slide','wifi')
                if slide:
                    wait=note['wait']
                    launch=(timing.second(tick+96) if wait.get('basis')=='beat' and wait.get('beat_num')==wait.get('beat_den') else
                            timing.second(tick)+float(wait['seconds']))
                    note_launch=launch
                    note['wait'].update(basis='beat',beat_num=1,beat_den=1)
                    for branch in note['branches']:
                        launch=note_launch
                        if 'wait' in branch:
                            launch=timing.second(tick)+float(branch['wait']['seconds'])
                        launch_tick=max(tick+1,snap(timing.second_to_beat(launch)*96))
                        d=branch['duration']
                        seconds=(d['beat_num']/d['beat_den']*96*timing.written_tick_seconds(tick) if d.get('basis')=='beat' else float(d['seconds']))
                        end=max(launch_tick+1,snap(timing.second_to_beat(launch+seconds)*96))
                        durations.append(SlideDuration(end-launch_tick,launch_tick-tick))
                        d.update(basis='beat',beat_num=1,beat_den=1)
                elif note['kind'] in ('hold','touch_hold'):
                    d=note['duration']
                    seconds=(d['beat_num']/d['beat_den']*96*timing.written_tick_seconds(tick) if d.get('basis')=='beat' else float(d['seconds']))
                    durations.append(HoldDuration(max(1,snap(timing.second_to_beat(timing.second(tick)+seconds)*96)-tick)))
                    d.update(basis='beat',beat_num=1,beat_den=1)
    encoded=encode_chart(source); values=iter(durations)
    tokens=tuple(next(values) if isinstance(t,(HoldDuration,SlideDuration)) else t for t in encoded.tokens)
    metadata=dict(encoded.metadata,duration_contract='global-tick96.v1')
    timing=timing_v6(timing)
    grammar=Grammar(timing); grammar.inject_boundary(tokens[0],start_tick=encoded.start_tick)
    rows=[]
    for t in tokens[1:]:
        grammar.accept(t)
        if isinstance(t,Moment): continue
        tick=grammar.state.cursor
        if not rows or rows[-1][0]!=tick: rows.append((tick,[]))
        rows[-1][1].append(t)
    return ChartData(tuple(OnsetGroup(t,tuple(c)) for t,c in rows),(),timing,None),EncodedChart(tokens,metadata,encoded.start_tick)


def note_blocks(group):
    blocks=[]
    for t in group.tokens:
        if isinstance(t,Note): blocks.append([])
        blocks[-1].append(t)
    return blocks


def normalize_touch_ticks(chart):
    """Normalize isolated micro-delays once, preserving regular Touch streams."""
    groups=list(chart.groups); launches=set()
    for group in groups:
        launches.update(group.tick+t.wait_ticks for t in group.tokens if isinstance(t,SlideDuration))
    def pure_touch(group):
        notes=[t for t in group.tokens if isinstance(t,Note)]
        return bool(notes) and all(t.kind=='touch' and not t.short_touch_hold for t in notes)
    output=[]; i=0
    while i<len(groups):
        group=groups[i]
        if not pure_touch(group): output.append(group); i+=1; continue
        prev=groups[i-1] if i else None
        anchor=max((t for t in launches if t<group.tick),default=None)
        if prev is not None and (anchor is None or prev.tick>=anchor):
            anchor=prev.tick if not any(isinstance(t,Note) and t.kind in ('touch','touch_hold') for t in prev.tokens) else None
        delta=group.tick-anchor if anchor is not None else None
        stop=i+1
        if delta in (1,2,3,4):
            while stop<len(groups) and groups[stop].tick<=anchor+4 and pure_touch(groups[stop]): stop+=1
        following=groups[stop] if stop<len(groups) else None
        if (delta in (1,2,3,4,6,8) and following is not None and following.tick>anchor+4
                and following.tick-groups[stop-1].tick!=delta):
            parts=groups[i:stop]
            content=[]; indices=[]; seen={}
            for part in parts:
                offset=0
                for block in note_blocks(part):
                    note=block[0]
                    if stop>i+1 and note.pad in seen:
                        j=seen[note.pad]
                        content[j]=replace(content[j],firework=int(bool(content[j].firework or note.firework)))
                    else:
                        seen[note.pad]=len(content);content.extend(block)
                        indices.extend(part.source_indices[offset:offset+len(block)])
                    offset+=len(block)
            output.append(replace(group,tick=anchor+4,tokens=tuple(content),source_indices=tuple(indices)))
            i=stop
        else: output.append(group); i+=1
    return replace(chart,groups=tuple(output))


def canonical_order(chart,geometry,*,physical_enabled=True):
    """Rank against the immutable previous-onset history, then commit in order."""
    from maidata.physical.rules import EPS, prune_history
    chart=normalize_touch_ticks(global_chart(chart)); timing=chart.timing
    history=(); owner=0; groups=[]
    for group in canonical_groups(chart.groups):
        history=prune_history(history,timing.second(group.tick))
        slide_pads={t.pad for t in group.tokens if isinstance(t,Note) and t.kind=='slide'}
        def actions(block):
            g=Grammar(timing); g.inject_boundary(Moment(0),start_tick=group.tick)
            g.state=replace(g.state,owner=owner-1)
            result=[]
            for t in block:
                c=g.accept(t)
                if c: result.append(materialize(c,timing,geometry))
            return result
        def rank(block):
            n=block[0]
            star_head=n.kind=='tap' and n.pad in slide_pads
            if not physical_enabled:return n.kind in ('touch','touch_hold'),n.kind=='slide',star_head,n.pad,n.kind
            candidate=actions(block)
            bad=any(c.category in ('tap_on_slide','overlap','slide_too_fast') for a in candidate for c in action_conflicts(history,a,check_capacity=False))
            from maidata.physical.geometry import pad_bit
            for a in candidate:
                for old in history:
                    hold,slide=(old,a) if a.track else (a,old)
                    if hold.kind not in ('hold','touch_hold') or not slide.track or not slide.motions[-1].entries: continue
                    entered,mask=slide.motions[-1].entries[-1]
                    bad |= bool(mask&pad_bit(hold.pad) and hold.press_start<=entered+EPS and timing.second_to_beat(entered)*96-96<timing.second_to_beat(hold.press_end)*96-22-EPS)
            locked=any(a.track and a.pad==n.pad and abs(a.launch-timing.second(group.tick))<=EPS for a in history)
            return n.kind in ('touch','touch_hold'),bad,locked,n.kind=='slide',star_head,n.pad,n.kind
        blocks=note_blocks(group)
        offsets=[]; offset=0
        for block in blocks: offsets.append(offset); offset+=len(block)
        content=[]; indices=[]
        order=range(len(blocks)) if len(blocks)<=1 else sorted(range(len(blocks)),key=lambda i:rank(blocks[i]))
        for index in order:
            block=blocks[index]
            if physical_enabled:history+=tuple(actions(block))
            owner+=1; content.extend(block)
            if group.source_indices: indices.extend(group.source_indices[offsets[index]:offsets[index]+len(block)])
        groups.append(replace(group,tokens=tuple(content),source_indices=tuple(indices)))
    return replace(chart,groups=tuple(groups))


def encode(chart,geometry,duration,start,end,*,ordered=False,physical_enabled=True,state_tokens=False,no_placeholders=True):
    chart=global_chart(chart) if ordered else canonical_order(chart,geometry,physical_enabled=physical_enabled)
    oracle=PhysicalOracle(chart.timing,geometry,duration,start,end,physical_enabled=physical_enabled,state_tokens=False,no_placeholders=no_placeholders)
    by_beat={}
    for g in chart.groups: by_beat.setdefault(g.tick//96,[]).append(g)
    output=[]
    def append(token):
        if no_placeholders and token.type in (5,7,10,17,18): return
        output.append(token); oracle.accept(token)
    for beat in range(start,end):
        for group in by_beat.get(beat,()):
            append(EventToken(0,group.tick%96))
            current=head=None
            for j,block in enumerate(note_blocks(group)):
                if j: append(EventToken(18))
                for t in block:
                    if isinstance(t,Note):
                        current=head=t.pad; touch=t.kind in ('touch','touch_hold')
                        if touch: append(EventToken(5)); append(EventToken(TOUCH_PAD,TOUCH_PADS.index(t.pad)))
                        else: append(EventToken(4,int(t.pad[1])-1))
                        flags=int(t.firework or 0) if touch else int(t.ex or 0)+2*int(t.break_ or 0)
                        if flags: append(EventToken(6,flags))
                        if t.kind in ('hold','touch_hold'): append(EventToken(7))
                    elif isinstance(t,Route):
                        if t.track_break is not None: current=head
                        shape,pad=route_shape_end(current,t.route_id)
                        append(EventToken(17)); append(EventToken(9,shape)); append(EventToken(4,pad))
                        if t.track_break: append(EventToken(6,1))
                        current='A'+str(pad+1)
                    elif isinstance(t,HoldDuration):
                        for token in duration_tokens(8,t.ticks): append(token)
                    elif isinstance(t,SlideDuration):
                        append(EventToken(10))
                        for token in wait_tokens(t.wait_ticks): append(token)
                        for token in duration_tokens(11,t.ticks): append(token)
            append(EventToken(12))
        append(EventToken(0,96))
    oracle.state.finish()
    return tuple(output),chart
