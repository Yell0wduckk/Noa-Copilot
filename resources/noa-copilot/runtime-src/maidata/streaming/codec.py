"""Canonical hold normalization, EventMaterializer and token input rows."""
from dataclasses import replace
from maidata.chart.types import Note, Route, HoldDuration, SlideDuration
from maidata.chart.routes import route_shape_end, shape_route_table
from maidata.draft.chart_data import OnsetGroup
from .sequence import EventToken, EventGrammar, duration_tokens, ZONES, vocab_id, BOUNDARY, HOLD_CONT, SLIDE_CONT


def canonical_groups(groups):
    result = []
    for group in groups:
        content = []
        for t in group.tokens:
            if isinstance(t, Note) and (t.short_hold or t.short_touch_hold):
                kind = 'hold' if t.short_hold else 'touch_hold'
                flags = {f:getattr(t,f) for f in (('ex','break_') if kind == 'hold' else ('firework',))}
                content.extend((Note.make(kind,t.pad,**flags), HoldDuration(1)))
            else: content.append(t)
        result.append(OnsetGroup(group.tick, tuple(content)))
    return tuple(result)


def encode_events(chart, start_beat, end_beat, *, notes_only=True, markers=False):
    groups = canonical_groups(chart.groups)
    by_beat = {}
    for group in groups: by_beat.setdefault(group.tick//96,[]).append(group)
    output = []
    for beat in range(start_beat,end_beat):
        row = by_beat.get(beat,())
        for group in row:
            output.append(EventToken(0,group.tick%96))
            head = current = None
            note_count = 0
            for t in group.tokens:
                if isinstance(t,Note):
                    if markers and note_count: output.append(EventToken(18))
                    note_count += 1
                    head = current = t.pad
                    touch = t.kind in ('touch','touch_hold')
                    if touch: output.append(EventToken(5,ZONES.index(t.pad[0])))
                    if t.pad != 'C': output.append(EventToken(4,int(t.pad[1])-1))
                    flags = int(t.firework or 0) if touch else int(t.ex or 0)+2*int(t.break_ or 0)
                    if flags: output.append(EventToken(6,flags))
                    if t.kind in ('hold','touch_hold'): output.append(EventToken(7))
                elif isinstance(t,Route):
                    if t.track_break is not None: current = head
                    shape,end = route_shape_end(current,t.route_id)
                    if markers: output.append(EventToken(17))
                    output.extend((EventToken(9,shape),EventToken(4,end)))
                    if t.track_break: output.append(EventToken(6,1))
                    current = 'A'+str(end+1)
                elif isinstance(t,HoldDuration): output.extend(duration_tokens(8,t.ticks))
                elif isinstance(t,SlideDuration):
                    output.append(EventToken(10)); output.extend(duration_tokens(11,t.ticks))
            output.append(EventToken(12))
        output.append(EventToken(0,96))
    return tuple(output)


class EventMaterializer:
    def __init__(self,start_beat,end_beat,commit=None,*,notes_only=True,markers=False):
        from .note_schema import NoteGrammar
        from .marker_schema import MarkerGrammar
        self.grammar = (MarkerGrammar if markers else NoteGrammar if notes_only else EventGrammar)(start_beat,end_beat)
        self.pending = None
        self.groups = []
        self.commit = commit

    def emit(self,token):
        tick = self.grammar.tick
        if self.commit is not None: self.commit(tick,token)
        if not self.groups or self.groups[-1][0] != tick: self.groups.append([tick,[]])
        self.groups[-1][1].append(token)

    def flush(self):
        if self.pending is not None:
            self.emit(self.pending); self.pending = None

    def accept(self,token):
        g = self.grammar
        t,v = token.type,token.value
        from .sequence import CONT_FAMILY
        if v not in g.allowed_values(CONT_FAMILY.get(t,t)):
            from maidata.chart.types import ChartError
            raise ChartError(f'illegal Simai token {token} in {g.phase}')
        if t in (4,5) and g.phase not in ('touch_pad','route_pad') or t in (12,18):
            self.flush()
        if t == 5 and v == 2: self.pending = Note.make('touch','C')
        elif t == 4:
            if g.phase == 'route_pad':
                self.pending = Route(shape_route_table()[g.shape][g.pad][v], 0 if g.first_route else None)
            else:
                touch = g.phase == 'touch_pad'
                self.pending = Note.make('touch' if touch else 'tap',(g.zone if touch else 'A')+str(v+1))
        elif t == 6:
            if isinstance(self.pending,Route): self.pending = replace(self.pending,track_break=1)
            elif g.zone is not None: self.pending = replace(self.pending,firework=1)
            else: self.pending = replace(self.pending,ex=v&1,break_=(v>>1)&1)
        elif t == 7:
            n = self.pending
            self.pending = Note.make('touch_hold' if g.zone is not None else 'hold',n.pad,
                **({'firework':n.firework} if g.zone is not None else {'ex':n.ex,'break_':n.break_}))
            self.flush()
        elif t in (9,17):
            if g.phase in ('after_pad','after_flags'):
                n = self.pending; self.pending = Note.make('slide',n.pad,ex=n.ex,break_=n.break_)
            self.flush()
        elif t == 10: self.flush()
        elif t in (8,11) and v != 96:
            self.emit((HoldDuration if t == 8 else SlideDuration)(g.duration+v+1))
        g.accept(token)

    def finish(self):
        self.flush(); self.grammar.finish()
        return tuple(OnsetGroup(t,tuple(tokens)) for t,tokens in self.groups)


def decode_events(tokens,start_beat,end_beat,*,notes_only=True,markers=False):
    out = EventMaterializer(start_beat,end_beat,notes_only=notes_only,markers=markers)
    for t in tokens: out.accept(t)
    return out.finish(),tuple(out.grammar.events)


def input_row(token, grammar, timing, *, context='button'):
    """Called after accept. Query addressing depends only on accepted prefix."""
    if getattr(grammar,'physical_v6',False):
        from .physical_schema import vocab_id as new_vocab, SLIDE_WAIT, SLIDE_WAIT_CONT
        beat=cross=grammar.tick/96
        if token.type in (10,SLIDE_WAIT_CONT): beat=cross=(grammar.tick+grammar.wait_elapsed)/96
        elif token.type in (SLIDE_WAIT,SLIDE_CONT): beat=cross=(grammar.tick+grammar.wait_ticks+grammar.duration)/96
        elif token.type in (7,HOLD_CONT): beat=cross=(grammar.tick+grammar.duration)/96
        return dict(types=token.type,vocab_ids=new_vocab(token,context),beats=float(beat),slide_beats=float(cross),causal_beats=float(grammar.tick/96))
    t = token.type
    causal = grammar.tick/96
    beat = causal
    cross = grammar.tick/96
    if grammar.phase == 'event_dur':
        beat = cross = (grammar.tick+grammar.elapsed)/96
    elif t in (7,10,HOLD_CONT,SLIDE_CONT):
        from maidata.time import duration_end
        ticks = grammar.duration
        cross = timing.second_to_beat(duration_end(timing,grammar.tick,ticks,slide=t in (10,SLIDE_CONT)))
        if t in (HOLD_CONT,SLIDE_CONT): beat = timing.second_to_beat(duration_end(timing,grammar.tick,grammar.duration,slide=t==SLIDE_CONT))
    return dict(types=t,vocab_ids=vocab_id(token,context),beats=float(beat),
                slide_beats=float(cross),causal_beats=float(causal))
