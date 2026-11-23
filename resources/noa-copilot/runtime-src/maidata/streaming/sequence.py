"""Simai postfix grammar. All numeric steps are determined by accepted inputs."""
from dataclasses import dataclass
from maidata.chart.types import ChartError
from maidata.chart.routes import shape_route_table

EVENT_TAGS = ('trill', 'jack', 'sweep', 'scatter', 'tap_slide', 'alternation',
              'button_touch_alt', 'chord_stream', 'sync_pair', 'fast_burst')
EVENT_STEPS = (6, 8, 12, 16, 24, 32, 48)

FAMILIES = ('MOMENT', 'BEAT_SUBDIV', 'EVENT_TAG', 'EVENT_DUR', 'PAD',
    'TOUCH_ZONE', 'FLAGS', 'HOLD', 'HOLD_DURATION', 'SLIDE_SHAPE', 'EOR',
    'SLIDE_DURATION', 'EOM', 'EVENT_STEP')
WIDTHS = (97, 9, 10, 2, 8, 5, 4, 1, 97, 19, 1, 97, 1, 7)
EVENT_STEP, HOLD_CONT, SLIDE_CONT = 13, 14, 15
EVENT_END, EVENT_CONT = 0, 1
CONT_FAMILY = {HOLD_CONT:8, SLIDE_CONT:11}
BOUNDARY = 16
ZONES = 'ABCDE'
FLAGS = ('ex', 'break_', 'firework')
VOCAB_OFFSETS = tuple(sum(WIDTHS[:i]) for i in range(len(FAMILIES)))
CONT_OFFSET = sum(WIDTHS)
FLAGS_TOUCH_OFFSET = CONT_OFFSET + 3
FLAGS_ROUTE_OFFSET = FLAGS_TOUCH_OFFSET + 1
VOCAB_SIZE = FLAGS_ROUTE_OFFSET + 1


@dataclass(frozen=True)
class EventToken:
    type: int
    value: int = 0


def vocab_id(token, context='button'):
    if token.type in (17,18): return VOCAB_SIZE + token.type - 17
    if token.type >= len(FAMILIES):
        return CONT_OFFSET + token.type - len(FAMILIES)
    if token.type == 6 and context != 'button':
        return FLAGS_TOUCH_OFFSET if context == 'touch' else FLAGS_ROUTE_OFFSET
    return VOCAB_OFFSETS[token.type] + token.value


def duration_tokens(family, ticks):
    cont = {8:HOLD_CONT, 11:SLIDE_CONT}[family]
    while ticks > 96:
        yield EventToken(cont, 96)
        ticks -= 96
    yield EventToken(family, max(1, ticks)-1)


class EventGrammar:
    def __init__(self, start_beat, end_beat):
        self.beat, self.end = start_beat, end_beat
        self.offset, self.phase = -1, 'subdiv'
        self.zone, self.pad, self.head_pad = None, None, None
        self.shape, self.routes, self.first_route = None, (), False
        self.tag_rank, self.pending_tag, self.elapsed = -1, None, 0
        self.pending_step = None
        self.duration = 0
        self.events = []

    @property
    def tick(self): return self.beat*96 + max(0, self.offset)

    @property
    def flag_context(self):
        return 'route' if self.phase == 'route_chain' else 'touch' if self.zone is not None else 'button'

    def allowed_types(self):
        fixed = {'subdiv':(1,), 'moment':(0,), 'event_step':(EVENT_STEP,), 'event_dur':(3,), 'touch_pad':(4,),
                 'route_pad':(4,), 'hold':(8,), 'slide':(11,), 'finished':()}
        if self.phase in fixed: return fixed[self.phase]
        if self.phase == 'prefix': return ((2,) if self.tag_rank < 1 else ()) + (4,5)
        if self.phase == 'route_chain':
            return ((6,) if self.first_route else ()) + (() if 70 in self.routes else (9,)) + (10,)
        if self.phase in ('after_pad', 'after_flags'):
            return ((6,) if self.phase == 'after_pad' else ()) + (7,) + (() if self.zone is not None else (9,)) + (4,5,12)
        if self.phase == 'after_slide': return (9,4,5,12)
        return (4,5,12)

    def allowed_values(self, family):
        if family not in self.allowed_types(): return ()
        if family == 0: return tuple(range(self.offset+1,97))
        if family == 2: return tuple(i for i in range(len(EVENT_TAGS)) if self.rank(i)>self.tag_rank)
        if family == EVENT_STEP:
            return tuple(range(7 if EVENT_TAGS[self.pending_tag] == 'chord_stream' else 5))
        if family == 4 and self.phase == 'route_pad':
            return tuple(i for i,r in enumerate(shape_route_table()[self.shape][self.pad]) if r >= 0)
        if family == 6: return (1,2,3) if self.flag_context == 'button' else (1,)
        if family == 9 and self.phase == 'route_chain': return tuple(range(18))
        return tuple(range(WIDTHS[family]))

    @staticmethod
    def rank(tag): return 1 if tag<7 else 0

    def accept(self, token):
        t, v = token.type, token.value
        family = CONT_FAMILY.get(t,t)
        if v not in self.allowed_values(family):
            raise ChartError(f'illegal Simai event {token} in {self.phase} at {self.tick}')
        if family == 1: self.phase = 'moment'
        elif family == 0:
            if v == 96:
                self.beat += 1; self.offset = -1
                self.phase = 'finished' if self.beat >= self.end else 'subdiv'
            else:
                self.offset = v; self.tag_rank = -1; self.phase = 'prefix'
        elif family == 2:
            self.pending_tag = v; self.tag_rank = self.rank(v); self.elapsed = 0; self.phase = 'event_step'
        elif family == EVENT_STEP:
            self.pending_step = EVENT_STEPS[v]; self.elapsed = 2*self.pending_step; self.phase = 'event_dur'
        elif family == 3:
            if v == EVENT_CONT: self.elapsed += self.pending_step
            else:
                self.events.append((self.pending_tag,self.pending_step,self.tick,self.tick+self.elapsed))
                self.elapsed = 0; self.pending_tag = self.pending_step = None; self.phase = 'prefix'
        elif family == 5:
            self.zone = ZONES[v]; self.pad = None; self.routes = ()
            self.phase = 'after_pad' if self.zone == 'C' else 'touch_pad'
        elif family == 4:
            if self.phase == 'route_pad':
                r = shape_route_table()[self.shape][self.pad][v]
                self.routes += (r,); self.pad = v; self.phase = 'route_chain'
            else:
                if self.phase != 'touch_pad': self.zone = None
                self.pad = self.head_pad = v; self.routes = (); self.phase = 'after_pad'
        elif family == 6:
            if self.phase == 'route_chain': self.first_route = False
            else: self.phase = 'after_flags'
        elif family == 7: self.duration = 0; self.phase = 'hold'
        elif family == 9:
            self.first_route = self.phase != 'route_chain'
            if self.first_route: self.routes = (); self.pad = self.head_pad
            self.shape = v; self.phase = 'route_pad'
        elif family == 10: self.duration = 0; self.phase = 'slide'
        elif family in (8,11):
            if v == 96: self.duration += 96
            else:
                self.duration = 0; self.phase = 'after_note' if family == 8 else 'after_slide'
        elif family == 12: self.phase = 'moment'

    def finish(self):
        if self.phase != 'finished': raise ChartError('incomplete Simai sequence')
