"""v5 structural tokens. Legacy semantic IDs and input rows stay unchanged."""
import torch
from maidata.chart.types import ChartError
from .sequence import (FAMILIES as OLD_FAMILIES, WIDTHS as OLD_WIDTHS,
    VOCAB_SIZE as OLD_VOCAB_SIZE, VOCAB_OFFSETS, CONT_OFFSET,
    FLAGS_TOUCH_OFFSET, FLAGS_ROUTE_OFFSET, HOLD_CONT, SLIDE_CONT, BOUNDARY)
from .note_schema import NoteGrammar, NOTE_FAMILIES, NOTE_INPUT_IDS

SLIDE, CHORD = 17, 18
FAMILIES = OLD_FAMILIES + ('HOLD_CONT', 'SLIDE_CONT', 'BOUNDARY', 'SLIDE', 'CHORD')
WIDTHS = OLD_WIDTHS + (0, 0, 0, 1, 1)
ACTIVE_FAMILIES = NOTE_FAMILIES + (SLIDE, CHORD)
INPUT_IDS = NOTE_INPUT_IDS + (OLD_VOCAB_SIZE, OLD_VOCAB_SIZE + 1)
VOCAB_SIZE = OLD_VOCAB_SIZE + 2
GATES = ('moment', 'pad', 'note_structure', 'route', 'hold_duration', 'slide_duration')


def vocab_gate_map():
    table = torch.zeros(VOCAB_SIZE, dtype=torch.long)
    for f, gate in ((0,1),(4,2),(5,1),(6,2),(7,4),(8,2),(9,3),(10,5),(11,2),(12,0)):
        table[VOCAB_OFFSETS[f]:VOCAB_OFFSETS[f]+OLD_WIDTHS[f]] = gate
    table[VOCAB_OFFSETS[0]+96] = 0
    table[VOCAB_OFFSETS[5]+2] = 2
    table[VOCAB_OFFSETS[8]+96] = 4
    table[VOCAB_OFFSETS[11]+96] = 5
    table[CONT_OFFSET:CONT_OFFSET+3] = torch.tensor([4,5,0])
    table[FLAGS_TOUCH_OFFSET:FLAGS_ROUTE_OFFSET+1] = 2
    table[OLD_VOCAB_SIZE:] = torch.tensor([3,1])
    return table


class MarkerGrammar(NoteGrammar):
    """Only accepted input tokens select a gate; no target/state routing input."""
    markers = True
    widths = WIDTHS

    def allowed_types(self):
        if self.phase == 'slide_shape': return (9,)
        old = super().allowed_types()
        if self.phase in ('after_pad','after_flags','after_note','after_slide'):
            return tuple(SLIDE if t==9 else CHORD if t==4 else t for t in old if t!=5)
        if self.phase == 'route_chain': return tuple(SLIDE if t==9 else t for t in old)
        return old

    def allowed_values(self, family):
        if family not in self.allowed_types(): return ()
        if family in (SLIDE,CHORD): return (0,)
        if family==9 and self.phase=='slide_shape':
            return tuple(range(18 if self.slide_origin=='route_chain' else 19))
        return super().allowed_values(family)

    def accept(self, token):
        t = token.type
        if t in (SLIDE,CHORD) or t==9:
            if token.value not in self.allowed_values(t):
                raise ChartError(f'illegal marker token {token} in {self.phase} at {self.tick}')
            if t==CHORD: self.phase='prefix'
            elif t==SLIDE:
                self.slide_origin=self.phase
                self.first_route=self.phase!='route_chain'
                if self.first_route: self.routes=(); self.pad=self.head_pad
                self.phase='slide_shape'
            else: self.shape=token.value; self.phase='route_pad'
        else: super().accept(token)
