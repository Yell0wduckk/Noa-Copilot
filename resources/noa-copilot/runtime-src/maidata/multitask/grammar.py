"""One accepted-prefix grammar for cached decoding and flat masked teacher forcing."""
from __future__ import annotations
from dataclasses import dataclass, field
from functools import lru_cache
import math

from maidata.chart.routes import shape_route_table
from .events import Event
from .grid import BeatGrid

COMMON = (('BOUNDARY', 1), ('BEAT', 16), ('MOMENT', 96), ('EOM', 1))
SHARED_ROWS = sum(width for _, width in COMMON)
FIELDS = {
    'maimai': (('KEY_POS', 8), ('TOUCH_PAD', 33), ('KEY_BREAK', 1), ('KEY_EX', 1),
               ('PSEUDO_EACH', 1), ('KEY_TYPE', 2), ('TOUCH_FIREWORK', 1), ('TRACK_BREAK', 1),
               ('HOLD_END_BEAT', 16), ('HOLD_END_OFFSET', 96), ('ROUTE', 19),
               ('SLIDE_WAIT_BEAT', 16), ('SLIDE_WAIT_OFFSET', 96),
               ('SLIDE_END_BEAT', 16), ('SLIDE_END_OFFSET', 96), ('LONG_SPAN', 1)),
    'midi': (('SLOT_PITCHED', 4), ('NOTE_ON', 128), ('SLOT_UNPITCHED', 4), ('VELOCITY', 5),
             ('MIDI_END_BEAT', 16), ('MIDI_END_OFFSET', 96), ('LONG_SPAN', 1)),
}
TOUCH_PADS = tuple(p for zone in 'ABCDE' for p in (
    ('C',) if zone == 'C' else tuple(f'{zone}{n}' for n in range(1, 9))))


@dataclass(frozen=True, slots=True)
class Token:
    family: str
    value: int = 0
    pseudo: bool = field(default=False, compare=False)  # acoustic teacher annotation
    slot_group: int = field(default=0, compare=False)
    velocity_db: float = field(default=float('nan'), compare=False)


@lru_cache(maxsize=2)
class TaskVocabulary:
    def __init__(self, task):
        self.task = task
        self.widths = dict(COMMON + FIELDS[task])
        self.offsets, self.tokens, self._ids = {}, [], {}
        for name, width in self.widths.items():
            base = self.offsets[name] = len(self.tokens)
            self.tokens.extend(Token(name, i) for i in range(width))
            self._ids[name] = tuple(range(base, base + width))
        self.size = len(self.tokens)
        self.boundary_id, self.eom_id = 0, self.offsets['EOM']

    def encode(self, family, value=0):
        if isinstance(family, Token):
            family, value = family.family, family.value
        return self._ids[family][value]

    def decode(self, token_id):
        return self.tokens[int(token_id)]

    def ids(self, family, values=None):
        ids = self._ids[family]
        return ids if values is None else tuple(ids[i] for i in values)


@lru_cache(maxsize=128)
def route_options(enabled, chained):
    tables = shape_route_table()
    ends = tuple(tuple(tuple(e for e, r in enumerate(shape[start])
                             if r >= 0 and (enabled is None or enabled[r]) and not (chained and k == 18))
                       for start in range(8)) for k, shape in enumerate(tables))
    shapes = tuple(tuple(k for k, row in enumerate(ends) if row[start]) for start in range(8))
    return shapes, ends


def bucket(ticks):
    return 0 if ticks <= 0 else min(7, max(1, math.ceil(math.log2(ticks/8)) + 1))


class Grammar:
    def __init__(self, task, *, start_beat=None, metadata=None, version_profile=None, grid=None):
        self.task, self.vocab = task, TaskVocabulary(task)
        self.grid = grid or BeatGrid(metadata)
        self.first = self.grid.first_beat() if start_beat is None else start_beat
        self.beat = self.first - 1
        self.time = self.query_tick = self.first * 96
        self.offset = -1
        self.phase = 'boundary'
        self.index = self.moment_index = -1
        self.last_token = Token('BOUNDARY')
        self.events = []
        self.active = {}  # identity -> (onset tick, end, moment row, first row, last row, payload)
        self._state, self._spans = None, None
        self.slot = self.pitch = self.drum = -1
        self.previous_pitches, self.chord_pitches, self.pitch_times = {}, {}, {}
        self.pending = None
        self.pages = 0
        self.short_ticks = int((metadata or {}).get('midi_short_ticks', 12))
        self.moment_only = bool((metadata or {}).get('moment_only', False))
        self.pure_no_history = bool((metadata or {}).get('no_history', False))
        if task == 'maimai':
            from maidata.version_mask import profile_for_chart
            from maidata.chart.types import TOUCH_PADS as version_pads
            m = metadata or {}
            p = version_profile or profile_for_chart(m.get('song_id'), m.get('difficulty'), m.get('version'))
            self.touch_ids = tuple(range(33)) if p is None or p.kind5_mask[2] else ()
            self.touch_taps = set(self.touch_ids) if p is None else {
                i for i, pad in enumerate(TOUCH_PADS) if p.touch_pad33_mask[version_pads.index(pad)]}
            self.chain = p is None or p.chained_route_enabled
            self.multi = p is None or p.multi_track_enabled
            self.track_break = p is None or p.track_break_enabled
            self.ex = p is None or bool(p.note_any_flag_mask[0])
            self.hold_break = p is None or bool(p.note_flag_mask[1, 1])
            self.slide_break = p is None or bool(p.note_flag_mask[4, 1])
            routes = None if p is None else tuple(p.route71_mask.tolist())
            self.routes_by_start, self.route_ends = route_options(routes, False)
            self.chains_by_start, _ = route_options(routes, True)

    @property
    def previous_pitch(self):
        return self.previous_pitches.get(self.slot, -1)

    def _clock(self, tick):
        self.time = tick
        if self.active:
            expired = [k for k, v in self.active.items() if v[1] <= tick]
            for k in expired:
                del self.active[k]
            if expired:
                self._spans = None
        self._state = None

    def _activate(self, key, end, payload):
        if end > self.time:
            self.active[key] = (self.time, end, self.moment_index, self.note_start, self.index + 1, payload)
            self._spans = self._state = None

    def _new_keys(self):
        v = self.vocab
        out = [] if self.touches else [v.offsets['KEY_POS'] + i for i in range(8)
            if i not in self.pseudos or (not self.in_pseudo and i not in self.keys)]
        out.extend(v.offsets['TOUCH_PAD'] + i for i in self.touch_ids if i not in self.touches)
        if self.content:
            out.append(v.eom_id)
        return out

    def _beat_choices(self, family):
        # The 16-beat page is anchored at the onset beat, including for slide
        # ends with a custom wait. Both twin slides therefore spell the same
        # absolute pair even when their starts/launches differ.
        low = self._minimum(family)
        first = self.beat + 16*self.pages
        return self.vocab.ids(family+'_BEAT', (b % 16 for b in range(max(first, low//96), first+16)))

    def _minimum(self, family):
        if family == 'SLIDE_END':
            return self.launch + 1
        return self.time + (25 if family == 'MIDI_END' else 0)

    def _end_options(self, *families):
        opts = []
        for family in families:
            opts.extend(self._beat_choices(family))
        if self.pages < 16 or not opts:
            opts.insert(0, self.vocab.offsets['LONG_SPAN'])
        return opts

    def legal_tokens(self):
        v, phase = self.vocab, self.phase
        if phase == 'boundary':
            return (0,)
        next_beat = v.offsets['BEAT'] + (self.beat+1) % 16
        if phase == 'first':
            return (next_beat,)
        if phase == 'moment':
            return (next_beat,) + v.ids('MOMENT')[self.offset+1:]
        if phase == 'moment_only_eom':
            return (v.eom_id,)
        if phase == 'end_offset':
            low = max(0, self._minimum(self.endpoint) - self.target_beat*96)
            high = 95 if self.endpoint == 'SLIDE_WAIT' and self.target_beat >= self.beat + 16*16 + 15 else 96
            ids = v.ids(self.endpoint+'_OFFSET')[low:high]
            if self.endpoint == 'SLIDE_WAIT' and self.target_beat == (self.time+96)//96:
                default = v.offsets['SLIDE_WAIT_OFFSET'] + (self.time+96)%96
                return tuple(i for i in ids if i != default)
            return ids
        if phase == 'pages':
            return tuple(self._end_options(*self.end_families))
        if self.task == 'midi':
            if phase == 'pitch':
                return v.ids('NOTE_ON')
            if phase == 'velocity':
                return v.ids('VELOCITY')
            opts = []
            if self.drum < 0:
                if self.slot >= 4:
                    opts.extend(v.ids('NOTE_ON')[self.pitch+1:])
                opts.extend(v.ids('SLOT_PITCHED')[max(0, self.slot-3):])
            opts.extend(v.ids('SLOT_UNPITCHED')[self.drum+1:])
            if self.content:
                opts.append(v.eom_id)
            if phase == 'pitched_tail':
                opts.extend(self._end_options('MIDI_END'))
            return tuple(opts)
        if phase == 'route_pad':
            return v.ids('KEY_POS', self.route_ends[self.shape][self.route_start])
        if phase == 'typed':
            return v.ids('ROUTE', self.routes_by_start[self.head])
        if phase == 'route_tail':
            opts = self._end_options('SLIDE_WAIT', 'SLIDE_END')
            if self.chain and self.route[0]['shape'] != 18:
                opts.extend(v.ids('ROUTE', self.chains_by_start[self.route[-1]['end']]))
            if len(self.route) == 1 and self.track_break and not self.branch_break:
                opts.append(v.offsets['TRACK_BREAK'])
            return tuple(opts)
        if phase == 'waited':
            return tuple(self._end_options('SLIDE_END'))
        if phase == 'key' and self.force_pseudo:
            return (v.offsets['PSEUDO_EACH'],)
        opts = self._new_keys()
        if phase == 'slide_done' and self.multi:
            opts.extend(v.ids('ROUTE', self.routes_by_start[self.head]))
        if phase in ('key', 'pseudo', 'break', 'ex'):
            if phase == 'key' and self.can_pseudo:
                opts.append(v.offsets['PSEUDO_EACH'])
            if phase in ('key', 'pseudo'):
                opts.append(v.offsets['KEY_BREAK'])
            if phase != 'ex' and self.ex:
                opts.append(v.offsets['KEY_EX'])
            if self.hold_break or not self.pending.get('break_'):
                opts.extend(self._end_options('HOLD_END'))
            if self.slide_break or not self.pending.get('break_'):
                if self.routes_by_start[self.head]:
                    opts.extend(v.ids('KEY_TYPE'))
                    opts.extend(v.ids('ROUTE', self.routes_by_start[self.head]))
        elif phase in ('touch', 'firework'):
            if self.touch not in self.touch_taps:
                opts = []  # this version allows this pad as a Hold only
            if phase == 'touch':
                opts.append(v.offsets['TOUCH_FIREWORK'])
            opts.extend(self._end_options('HOLD_END'))
        return tuple(opts)

    def _flush(self):
        if self.pending is not None:
            kind = 'touch' if self.pending['touch'] else 'tap'
            data = dict(self.pending)
            del data['touch']
            self.events.append(Event(self.grid.second(self.time), kind, data))
            self.pending = None

    def _complete(self, family, end):
        sec, end_sec = self.grid.second(self.time), self.grid.second(end)
        if family == 'HOLD_END':
            data = dict(self.pending)
            touch = data.pop('touch')
            data.update(end_tick=end, duration=end_sec-sec)
            self.events.append(Event(sec, 'touch_hold' if touch else 'hold', data))
            self._activate(('hold', data['pad'], touch), end, ('hold', data['pad'], touch))
            self.pending = None
            self.phase = 'notes'
        elif family == 'SLIDE_WAIT':
            self.launch, self.phase = end, 'waited'
        elif family == 'SLIDE_END':
            launch_sec = self.grid.second(self.launch)
            branch = dict(routes=self.route, track_break=self.branch_break, launch_tick=self.launch, end_tick=end,
                          wait=launch_sec-sec, duration=end_sec-launch_sec)
            if self.pending is not None:
                data = dict(self.pending)
                del data['touch']
                data['branches'] = []
                self.slide = Event(sec, 'slide', data)
                self.events.append(self.slide)
                self.pending = None
            self.slide.data['branches'].append(branch)
            self._activate(('slide', self.note_start, len(self.slide.data['branches'])), end,
                           ('slide', self.head, self.route[-1]['end'], self.launch))
            self.phase = 'slide_done'
        else:
            self.events[-1].data.update(end_tick=end, duration=end_sec-sec)
            self._activate((self.slot, self.pitch), end, ('midi', self.pitch))
            self.phase = 'notes'

    def advance(self, token_id, *, validate=True):
        v = self.vocab
        token_id = v.encode(token_id) if isinstance(token_id, Token) else int(token_id)
        if validate and token_id not in self.legal_tokens():
            raise ValueError(f'illegal {v.decode(token_id)} in {self.task}/{self.phase}')
        t = self.last_token = v.decode(token_id)
        f, val, old = t.family, t.value, self.phase
        self.index += 1
        if f == 'BOUNDARY':
            self.phase = 'first'
        elif f == 'BEAT':
            self.beat += 1
            self.offset = -1
            self._clock(self.beat*96)
            self.moment_index, self.phase = self.index, 'moment'
        elif f == 'MOMENT':
            self.offset = val
            self._clock(self.beat*96 + val)
            if self.moment_only:
                self.moment_index, self.phase, self.content = self.index, 'moment_only_eom', True
                self.events.append(Event(self.grid.second(self.time), 'moment', dict(start_tick=self.time)))
            else:
                self.moment_index, self.phase, self.content = self.index, 'notes', False
                self.keys, self.pseudos, self.touches = set(), set(), set()
                self.in_pseudo = False
                self.slot = self.pitch = self.drum = -1
        elif f == 'EOM':
            self._flush()
            self.pitch_times.update(dict.fromkeys(self.chord_pitches,self.time))
            self.previous_pitches.update(self.chord_pitches)
            self.chord_pitches.clear()
            self.phase = 'moment'
        elif f == 'LONG_SPAN':
            if old != 'pages':
                self.end_families = (('MIDI_END',) if self.task == 'midi' else
                    ('SLIDE_END',) if old == 'waited' else
                    ('SLIDE_WAIT', 'SLIDE_END') if old == 'route_tail' else ('HOLD_END',))
            self.pages += 1
            self.phase = 'pages'
        elif f.endswith('_BEAT'):
            self.endpoint = f[:-5]
            self.target_beat = self.beat + (val-self.beat)%16 + 16*self.pages
            self.pages, self.phase = 0, 'end_offset'
        elif f.endswith('_OFFSET'):
            self._complete(self.endpoint, self.target_beat*96+val)
        elif f == 'SLOT_PITCHED':
            self.slot, self.pitch, self.phase = val+4, -1, 'pitch'
        elif f == 'SLOT_UNPITCHED':
            self.drum, self.slot, self.phase = val, val, 'velocity'
        elif f == 'NOTE_ON':
            self.note_start, self.pitch, self.phase = self.index, val, 'velocity'
            self.chord_pitches[self.slot] = val  # sorted chords: last is the highest
        elif f == 'VELOCITY':
            self.content = True
            sec = self.grid.second(self.time)
            data = dict(slot=self.slot, velocity_bin=val, start_tick=self.time)
            if self.drum < 0:
                data.update(pitch=self.pitch, end_tick=self.time+self.short_ticks, duration=self.grid.second(self.time+self.short_ticks)-sec)
                self.events.append(Event(sec, 'note', data))
                self.phase = 'pitched_tail'
            else:
                self.events.append(Event(sec, 'unpitched', data))
                self.phase = 'notes'
        elif f == 'KEY_POS' and old == 'route_pad':
            self.route.append(dict(shape=self.shape, end=val))
            self.phase = 'route_tail'
        elif f in ('KEY_POS', 'TOUCH_PAD'):
            self._flush()
            self.content, self.note_start = True, self.index
            if f == 'KEY_POS':
                self.head = val
                self.can_pseudo = bool(self.keys) and val not in self.pseudos
                self.force_pseudo = self.in_pseudo or val in self.keys
                if not self.force_pseudo:
                    self.keys.add(val)
                self.pending, self.phase = dict(pad=f'A{val+1}', start_tick=self.time, touch=False), 'key'
            else:
                self.touch = val
                self.touches.add(val)
                self.pending, self.phase = dict(pad=TOUCH_PADS[val], start_tick=self.time, touch=True), 'touch'
        elif f == 'PSEUDO_EACH':
            if not self.force_pseudo:
                self.keys.remove(self.head)
            self.pseudos.add(self.head)
            self.in_pseudo = self.pending['pseudo_each'] = True
            self.phase = 'pseudo'
        elif f in ('KEY_BREAK', 'KEY_EX', 'KEY_TYPE', 'TOUCH_FIREWORK'):
            key, value, self.phase = {'KEY_BREAK': ('break_', True, 'break'), 'KEY_EX': ('ex', True, 'ex'),
                'KEY_TYPE': ('head_type', val+1, 'typed'), 'TOUCH_FIREWORK': ('firework', True, 'firework')}[f]
            self.pending[key] = value
        elif f == 'ROUTE':
            if old == 'route_tail':
                self.route_start = self.route[-1]['end']
            else:
                self.route_start, self.route, self.branch_break = self.head, [], False
                self.launch = self.time + 96
            self.shape, self.phase = val, 'route_pad'
        elif f == 'TRACK_BREAK':
            self.branch_break = True
        self.query_tick = self.target_beat*96 if self.phase == 'end_offset' else self.time

    def spans(self):
        if self._spans is None:
            notes = self.active.values()
            if self.task == 'midi' and len(self.active) > 32:
                notes = sorted(notes, key=lambda n: n[3], reverse=True)[:32]
            self._spans = frozenset(span for n in notes for span in ((n[2], n[2]+1), (n[3], n[4])))
        return self._spans

    def state_features(self):
        if self._state is not None:
            return self._state
        out = [0.] * 32
        slides = []
        for start, end, _, _, _, obj in self.active.values():
            if obj[0] == 'midi':
                pc = obj[1] % 12
                out[pc] = min(2, out[pc]+1)
            elif obj[0] == 'slide':
                out[11+obj[1]] = out[19+obj[2]] = 1
                slides.append((end, obj[3]))
            elif not obj[2] or obj[1] == 'C':
                out[27 if obj[2] else int(obj[1][1])-1] = bucket(end-self.time)
            else:
                out[28] = min(3, out[28]+1)
                out[29] = max(out[29], bucket(self.time-start))
                out[30] = max(out[30], bucket(end-self.time))
        if self.task == 'midi' and self.active:
            n = len(self.active)
            out[12] = n if n <= 2 else 3 if n <= 4 else 4 if n <= 8 else 5
            ages = [self.time-note[0] for note in self.active.values()]
            out[13:15] = [bucket(max(ages)), bucket(min(ages))]
        if slides:
            end, launch = min(slides)
            out[8:11] = [min(3, len(slides)), bucket(launch-self.time), bucket(end-max(self.time, launch))]
        self._state = out
        return out

    def input_features(self, *, legal_ids=None):
        role = int(self.last_token.family not in ('BOUNDARY', 'BEAT', 'EOM'))
        visibility = 0 if self.pure_no_history else 1 - role
        protected = sorted({i for a,b in self.spans() for i in range(a,b)})
        retained = protected if self.task != 'midi' or len(self.active) <= 32 else sorted({
            i for _,_,m,a,b,_ in self.active.values() for i in (m, *range(a,b))})
        return dict(causal_beats=self.time/96, query_beats=self.query_tick/96,
                    audio_positions=125*self.grid.second(self.query_tick), onset_seconds=self.grid.second(self.time),
                    token_steps=self.index, moment_ids=self.moment_index, roles=role, history_visibility=visibility,
                    state_features=self.state_features(), protected_token_indices=protected,
                    retain_active_token_indices=retained)

    def finish(self, **unused):
        if self.phase in ('key','pseudo','break','ex','touch','firework') and not (
                self.phase == 'key' and self.force_pseudo):
            self._flush()
        return list(self.events)
