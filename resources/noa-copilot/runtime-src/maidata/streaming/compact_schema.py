"""Physical v8: stable semantic/vocabulary IDs, compact payload-only syntax."""
import torch
from maidata.chart.types import ChartError
from .physical_schema import (FAMILIES, WIDTHS, VOCAB_OFFSETS, VOCAB_SIZE,
    INPUT_IDS as LEGACY_INPUT_IDS, PhysicalGrammar as LegacyGrammar,
    TOUCH_PAD, TOUCH_PADS, SLIDE_WAIT, SLIDE_WAIT_CONT, STATES, vocab_id)
from .sequence import CONT_FAMILY, EventToken

PLACEHOLDERS = (5, 7, 10, 17, 18)
ACTIVE_FAMILIES = (0, 4, 6, 8, 9, 11, 12, 19, 23)
ACTIVE_STATE_FAMILIES = ACTIVE_FAMILIES + STATES
INPUT_IDS = tuple(i for i in LEGACY_INPUT_IDS if i not in {vocab_id(EventToken(f)) for f in PLACEHOLDERS})
INPUT_FAMILIES = (0, 4, 6, 8, 9, 11, 12, 14, 15, 16, 19, 23, 24)
GATES = ('moment', 'note')


def vocab_gate_map():
    table = torch.ones(VOCAB_SIZE, dtype=torch.long)
    table[VOCAB_OFFSETS[0] + 96] = 0
    table[VOCAB_OFFSETS[12]] = 0
    table[vocab_id(EventToken(16))] = 0
    return table


class PhysicalGrammar(LegacyGrammar):
    no_placeholders = True

    def allowed_types(self):
        if self.state_queue: return (self.state_queue[0].type,)
        fixed = {'moment': (0,), 'prefix': (4, 19), 'hold': (8,),
                 'route_pad': (4,), 'wait': (23,), 'slide': (11,), 'finished': ()}
        if self.phase in fixed: return fixed[self.phase]
        if self.phase in ('after_pad', 'after_flags'):
            return ((6,) if self.phase == 'after_pad' else ()) + ((8,) if self.zone in (None, 'C') else ()) + ((9,) if self.zone is None else ()) + (4, 19, 12)
        if self.phase == 'route_chain':
            return ((6,) if self.first_route else ()) + (() if 70 in self.routes else (9,)) + (23,)
        if self.phase == 'after_slide': return (9, 4, 19, 12)
        return (4, 19, 12)

    def allowed_values(self, family):
        if family not in self.allowed_types(): return ()
        if self.state_queue: return (self.state_queue[0].value,)
        if family == 0: return tuple(range(self.offset + 1, 97))
        if family == 4 and self.phase == 'route_pad':
            from maidata.chart.routes import shape_route_table
            return tuple(i for i, r in enumerate(shape_route_table()[self.shape][self.pad]) if r >= 0)
        if family == 6: return (1, 2, 3) if self.flag_context == 'button' else (1,)
        if family == 9 and self.phase == 'route_chain': return tuple(range(18))
        return tuple(range(WIDTHS[family]))

    def _reset_note(self):
        self.pad = self.head_pad = self.zone = None
        self.routes = (); self.first_route = True
        self.duration = self.wait_elapsed = 0; self.wait_ticks = 96

    def accept(self, token):
        t, v = token.type, token.value
        f = SLIDE_WAIT if t == SLIDE_WAIT_CONT else CONT_FAMILY.get(t, t)
        if v not in self.allowed_values(f):
            raise ChartError(f'illegal compact token {token} in {self.phase} at {self.tick}')
        if self.state_queue:
            self.state_queue = self.state_queue[1:]; return
        if f == 0:
            if v == 96:
                self.beat += 1; self.offset = -1
                self.phase = 'finished' if self.beat >= self.end else 'moment'
            else: self.offset = v; self.phase = 'prefix'
        elif f in (4, 19) and self.phase != 'route_pad':
            self._reset_note()
            if f == 19:
                pad = TOUCH_PADS[v]; self.zone = pad[0]
                self.pad = self.head_pad = 0 if pad == 'C' else int(pad[1]) - 1
            else: self.pad = self.head_pad = v
            self.phase = 'after_pad'
        elif f == 4:
            from maidata.chart.routes import shape_route_table
            self.routes += (shape_route_table()[self.shape][self.pad][v],)
            self.pad = v; self.phase = 'route_chain'
        elif f == 6:
            if self.phase == 'route_chain': self.first_route = False
            else: self.phase = 'after_flags'
        elif f == 9:
            self.first_route = self.phase != 'route_chain'
            if self.first_route: self.routes = (); self.pad = self.head_pad
            self.shape = v; self.phase = 'route_pad'
        elif f == 23:
            if self.phase == 'route_chain': self.wait_elapsed = 0
            if v == 96: self.wait_elapsed += 96; self.phase = 'wait'
            else:
                self.wait_ticks = self.wait_elapsed + v + 1
                self.duration = 0; self.phase = 'slide'
        elif f in (8, 11):
            if v == 96: self.duration += 96; self.phase = 'hold' if f == 8 else 'slide'
            else:
                self.duration = 0
                if f == 11:
                    self.routes = (); self.first_route = True; self.pad = self.head_pad
                self.phase = 'after_note' if f == 8 else 'after_slide'
        elif f == 12: self._reset_note(); self.phase = 'moment'
