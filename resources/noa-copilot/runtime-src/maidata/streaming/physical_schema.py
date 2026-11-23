"""v6/v8 base vocabulary: independent Touch positions and slide waits."""
import torch
from . import marker_schema as old
from .sequence import EventToken, vocab_id as legacy_vocab_id, VOCAB_OFFSETS as LEGACY_OFFSETS

TOUCH, TOUCH_PAD = 5, 19
HOLD_CARRY, SLIDE_WAIT_STATE, SLIDE_MOVE = 20, 21, 22
SLIDE_WAIT, SLIDE_WAIT_CONT = 23, 24
STATES = (HOLD_CARRY, SLIDE_WAIT_STATE, SLIDE_MOVE)
TOUCH_PADS = tuple(f'{z}{i}' for z in 'AB' for i in range(1,9)) + ('C',) + tuple(f'{z}{i}' for z in 'DE' for i in range(1,9))
FAMILIES = old.FAMILIES[:5] + ('TOUCH',) + old.FAMILIES[6:] + ('TOUCH_PAD','HOLD_CARRY','SLIDE_WAIT_STATE','SLIDE_MOVE','SLIDE_WAIT','SLIDE_WAIT_CONT')
WIDTHS = old.WIDTHS[:5] + (1,) + old.WIDTHS[6:] + (33,97,97,97,97,0)
VOCAB_OFFSETS = LEGACY_OFFSETS + (0,0,0,0,0) + tuple(old.VOCAB_SIZE + sum(WIDTHS[19:i]) for i in range(19,25))
VOCAB_SIZE = VOCAB_OFFSETS[24] + 1
ACTIVE_FAMILIES = old.ACTIVE_FAMILIES + (19,20,21,22,23)
INPUT_IDS = tuple(i for i in old.INPUT_IDS if i not in range(LEGACY_OFFSETS[5]+1,LEGACY_OFFSETS[5]+5)) + tuple(range(old.VOCAB_SIZE,VOCAB_SIZE))
GATES = old.GATES


def vocab_id(token, context='button'):
    return VOCAB_OFFSETS[token.type] + (0 if token.type==SLIDE_WAIT_CONT else token.value) if token.type>=19 else legacy_vocab_id(token,context)


def vocab_gate_map():
    table=torch.zeros(VOCAB_SIZE,dtype=torch.long)
    table[:old.VOCAB_SIZE]=old.vocab_gate_map()
    table[LEGACY_OFFSETS[5]:LEGACY_OFFSETS[5]+5]=1
    table[VOCAB_OFFSETS[TOUCH_PAD]:VOCAB_OFFSETS[HOLD_CARRY]]=2
    table[VOCAB_OFFSETS[SLIDE_WAIT]:]=5
    return table


class PhysicalGrammar(old.MarkerGrammar):
    physical_v6=True
    widths=WIDTHS

    def __init__(self,start_beat,end_beat):
        super().__init__(start_beat,end_beat)
        self.wait_ticks=96
        self.wait_elapsed=0
        self.state_queue=()

    def allowed_types(self):
        if self.phase=='touch_pad': return (TOUCH_PAD,)
        if self.phase=='wait': return (SLIDE_WAIT,)
        return super().allowed_types()

    def allowed_values(self,family):
        if family not in self.allowed_types(): return ()
        if family==TOUCH: return (0,)
        if family in (TOUCH_PAD,SLIDE_WAIT): return tuple(range(WIDTHS[family]))
        return super().allowed_values(family)

    def accept(self,token):
        t,v=token.type,token.value
        f=SLIDE_WAIT if t==SLIDE_WAIT_CONT else t
        if f in (TOUCH,TOUCH_PAD,SLIDE_WAIT):
            if v not in self.allowed_values(f):
                from maidata.chart.types import ChartError
                raise ChartError(f'illegal v6 token {token} in {self.phase}')
            if f==TOUCH:
                self.zone='A'; self.pad=None; self.routes=(); self.phase='touch_pad'
            elif f==TOUCH_PAD:
                pad=TOUCH_PADS[v]; self.zone=pad[0]
                self.pad=self.head_pad=0 if pad=='C' else int(pad[1])-1
                self.routes=(); self.phase='after_pad'
            elif v==96: self.wait_elapsed+=96
            else:
                self.wait_ticks=self.wait_elapsed+v+1
                self.duration=0; self.phase='slide'
            return
        super().accept(token)
        if t==10: self.wait_elapsed=0; self.wait_ticks=0; self.phase='wait'


def wait_tokens(ticks):
    while ticks>96:
        yield EventToken(SLIDE_WAIT_CONT,96); ticks-=96
    yield EventToken(SLIDE_WAIT,ticks-1)
