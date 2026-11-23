"""Flat tick96 teacher forcing. Each row describes only its accepted prefix."""
from __future__ import annotations
import math
import json
from collections import defaultdict
import random
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
import torch

from .events import Event, MidiCropHistory, crop_events, mirror_events, quantize_time, stretch_events
from .grid import BeatGrid
from .grammar import Grammar, TaskVocabulary, Token, TOUCH_PADS
from .slots import PITCHED_SLOTS, event_slot, velocity_bin, supervision_group

@dataclass
class EncodedChart:
    task: str
    token_ids: torch.Tensor
    causal_beats: torch.Tensor
    query_beats: torch.Tensor
    audio_positions: torch.Tensor
    onset_seconds: torch.Tensor
    token_steps: torch.Tensor
    moment_ids: torch.Tensor
    roles: torch.Tensor
    history_visibility: torch.Tensor
    state_features: torch.Tensor
    loss_mask: torch.Tensor
    legal_mask: torch.Tensor
    protected_spans: torch.Tensor
    current_slots: torch.Tensor
    previous_pitches: torch.Tensor
    target_pseudo: torch.Tensor
    target_groups: torch.Tensor
    target_velocity_db: torch.Tensor
    residual_neg_mask: torch.Tensor | None = None
    moment_type_targets: torch.Tensor | None = None
    moment_type_weights: torch.Tensor | None = None
    moment_slot6_union: torch.Tensor | None = None

    def model_inputs(self):
        result = {name: getattr(self, name)[:-1] for name in
                  ('token_ids', 'causal_beats', 'query_beats', 'audio_positions', 'onset_seconds', 'token_steps',
                   'moment_ids', 'roles', 'history_visibility', 'state_features', 'legal_mask',
                   'current_slots', 'previous_pitches')}
        result.update(targets=self.token_ids[1:], loss_mask=self.loss_mask[1:],
                      protected_spans=self.protected_spans,
                      target_pseudo=self.target_pseudo[1:], target_groups=self.target_groups[1:],
                      target_velocity_db=self.target_velocity_db[1:])
        if self.residual_neg_mask is not None:
            result['residual_neg_mask'] = self.residual_neg_mask[1:]
        if self.moment_type_targets is not None:
            result['moment_type_targets'] = self.moment_type_targets[:-1]
            result['moment_type_weights'] = self.moment_type_weights[:-1]
            result['moment_slot6_union'] = self.moment_slot6_union[:-1]
        return result


def _canonical_maimai_events(events, clock):
    """One press per quantized onset/pad; collapse duplicate source objects.

    A co-located tap is the head of an existing hold/slide. Multiple slides at
    the same head become branches, preserving their paths and timing.
    """
    grouped = {}
    rank = {'tap': 0, 'touch': 0, 'hold': 1, 'touch_hold': 1, 'slide': 2}
    for e in events:
        frame = e.data['start_tick'] if 'start_tick' in e.data else clock(e.time)
        key = (frame, e.kind in ('touch', 'touch_hold'), e.data.get('pad', 'A1'), bool(e.data.get('pseudo_each')))
        d = dict(e.data)
        current = Event(e.time, e.kind, d)
        previous = grouped.get(key)
        if previous is None:
            grouped[key] = current
            continue
        chosen = current if rank[current.kind] > rank[previous.kind] else previous
        merged = dict(chosen.data)
        merged['break_'] = bool(d.get('break_', d.get('break')) or previous.data.get('break_', previous.data.get('break')))
        merged['ex'] = bool(d.get('ex') or previous.data.get('ex'))
        merged['firework'] = bool(d.get('firework') or previous.data.get('firework'))
        if current.kind == previous.kind == 'slide':
            branches = previous.data.get('branches', [previous.data]) + d.get('branches', [d])
            unique = {}
            for branch in branches:
                body = {k: v for k, v in branch.items() if k in ('routes', 'route_ids', 'launch_tick', 'end_tick', 'wait', 'duration', 'track_break')}
                unique.setdefault(json.dumps(body, sort_keys=True), body)
            merged['branches'] = list(unique.values())
        elif current.kind == previous.kind and current.kind in ('hold', 'touch_hold'):
            merged['end_tick'] = max(note.data['end_tick'] if 'end_tick' in note.data else
                                     clock(note.time+note.data.get('duration',0.)) for note in (current,previous))
            merged['duration'] = max(note.time+note.data.get('duration',0.) for note in (current,previous))-chosen.time
        grouped[key] = Event(chosen.time, chosen.kind, merged)
    return [(key[0], e) for key, e in sorted(grouped.items(), key=lambda item: item[0])]


def endpoint_tokens(family, onset, end):
    beats, offset = divmod(end, 96)
    return ([Token('LONG_SPAN')] * ((beats-onset//96)//16) +
            [Token(family+'_BEAT', beats % 16), Token(family+'_OFFSET', offset)])


def _maimai_tokens(events, grid, rng):
    by_tick = defaultdict(list)
    for tick, event in _canonical_maimai_events(events, grid.tick):
        by_tick[tick].append(event)
    ticks = sorted(by_tick)
    key_ticks = [t for t in ticks if any(e.kind not in ('touch', 'touch_hold') for e in by_tick[t])]
    key_gaps = [b-a for a,b in zip(key_ticks, key_ticks[1:])]
    key_index = {t: k for k, t in enumerate(key_ticks)}
    layers = defaultdict(lambda: [[], [], []])
    last_main = None
    for tick in ticks:
        # Three or more regularly spaced attacks are a real stream, including
        # its first and last pairs. Do not collapse a delayed group's touches.
        k = key_index.get(tick)
        gap = key_gaps[k-1] if k else None
        micro = gap in (1, 2) and not ((k >= 2 and key_gaps[k-2] in (1, 2)) or
                                       (k < len(key_gaps) and key_gaps[k] in (1, 2)))
        anchor = last_main
        local_keys = any(e.kind not in ('touch','touch_hold') and not e.data.get('pseudo_each')
                         for e in by_tick[tick])
        for e in by_tick[tick]:
            d, pad = e.data, e.data['pad']
            touch = e.kind in ('touch', 'touch_hold')
            explicit = bool(d.get('pseudo_each'))
            pseudo = not touch and ((explicit and local_keys) or
                (anchor is not None and ((explicit and not local_keys) or
                 (micro and anchor == key_ticks[k-1] and not explicit))))
            onset = (tick if explicit and local_keys else anchor) if pseudo else tick
            layer = 2 if touch else int(pseudo)
            if touch:
                tokens = [Token('TOUCH_PAD', TOUCH_PADS.index(pad))]
                if d.get('firework'):
                    tokens.append(Token('TOUCH_FIREWORK'))
            else:
                tokens = [Token('KEY_POS', int(pad[1])-1)]
                if pseudo:
                    tokens.append(Token('PSEUDO_EACH'))
                if d.get('break_', d.get('break')):
                    tokens.append(Token('KEY_BREAK'))
                if d.get('ex'):
                    tokens.append(Token('KEY_EX'))
            if e.kind in ('hold','touch_hold'):
                end = d['end_tick'] if 'end_tick' in d else grid.tick(e.time + d.get('duration', 0.))
                tokens += endpoint_tokens('HOLD_END', onset, onset if end-tick <= 6 else end)
            if e.kind == 'slide':
                if d.get('head_type'):
                    tokens.append(Token('KEY_TYPE', int(d['head_type'])-1))
                for branch in d.get('branches', [d]):
                    routes = branch.get('routes')
                    if routes is None:
                        from maidata.chart.routes import route_shape_end
                        routes, start = [], pad
                        for rid in branch['route_ids']:
                            shape, target = route_shape_end(start, rid)
                            routes.append(dict(shape=shape, end=target))
                            start = f'A{target+1}'
                    for k, route in enumerate(routes):
                        tokens += [Token('ROUTE', route['shape']), Token('KEY_POS', route['end'])]
                        if k == 0 and branch.get('track_break'):
                            tokens.append(Token('TRACK_BREAK'))
                    raw_wait = branch.get('wait', d.get('wait'))
                    launch_sec = grid.second(tick+96) if raw_wait is None else e.time+raw_wait
                    launch = max(onset, branch['launch_tick'] if 'launch_tick' in branch else grid.tick(launch_sec))
                    raw_end = (branch['end_tick'] if 'end_tick' in branch else
                               grid.tick(launch_sec+branch.get('duration', d.get('duration', 1.))))
                    if pseudo and not explicit and launch == tick+96 and raw_end > onset+96:
                        launch = onset+96
                    if launch-onset != 96:
                        tokens += endpoint_tokens('SLIDE_WAIT', onset, launch)
                    end = max(launch+1, raw_end)
                    tokens += endpoint_tokens('SLIDE_END', onset, end)
            layers[onset][layer].append(tokens)
        if layers.get(tick, ([],))[0]:
            last_main = tick
    for tick, (main, pseudo, touch) in sorted(layers.items()):
        if rng is not None:
            rng.shuffle(main)
            rng.shuffle(pseudo)
        yield tick, [t for note in (*main, *pseudo, *touch) for t in note]


def _midi_tokens(events, grid):
    from .midi import events_to_notes
    if any(e.kind in ('note_on', 'note_off') for e in events):
        events = events_to_notes(events, keep_open=True)
    pitched, hits = defaultdict(list), {}
    for e in events:
        slot = event_slot(e.data, e.kind)
        tick = e.data['start_tick'] if 'start_tick' in e.data else grid.tick(e.time)
        data = e.data
        if slot in PITCHED_SLOTS:
            end = max(tick, data['end_tick'] if 'end_tick' in data else grid.tick(e.time + data.get('duration', 0.)))
            pitched[slot, data['pitch']].append((tick, end, data))
        elif e.kind != 'note_off':
            key = tick, slot
            hits[key] = _merge_labels(hits[key], data) if key in hits else data
    by_tick = defaultdict(lambda: defaultdict(list))
    for (slot, pitch), values in pitched.items():
        # Unison from different GM tracks is one sounding interval. Touching
        # endpoints are separate attacks; only overlapping intervals merge.
        current = None
        for start, end, data in sorted(values, key=lambda x: x[:2]):
            if current and (start < current[1] or start == current[0]):
                current = current[0], max(end, current[1]), _merge_labels(current[2], data)
            else:
                if current:
                    by_tick[current[0]][slot].append((pitch, *current[1:]))
                current = start, end, data
        by_tick[current[0]][slot].append((pitch, *current[1:]))
    for (tick, slot), data in hits.items():
        by_tick[tick][slot].append((-1, tick, data))
    for tick, slots in sorted(by_tick.items()):
        tokens = []
        for slot in sorted(slots, key=lambda s: (s < 4, s)):
            values = sorted(slots[slot], key=lambda x: x[0])
            group = 0
            for _, _, d in values:
                group |= supervision_group(d, slot)
            tokens.append(Token('SLOT_PITCHED' if slot >= 4 else 'SLOT_UNPITCHED', slot%4,
                                slot_group=group, pseudo=any(d.get('pseudo') for _,_,d in values)))
            for pitch, end, data in values:
                if pitch >= 0:
                    tokens.append(Token('NOTE_ON', pitch))
                db = data.get('velocity_db')
                tokens.append(Token('VELOCITY', velocity_bin(data), velocity_db=float('nan') if db is None else db))
                if pitch >= 0 and end-tick > 24:
                    tokens += endpoint_tokens('MIDI_END', tick, end)
        yield tick, tokens


def _merge_labels(a, b):
    db = max((v for v in (a.get('velocity_db'), b.get('velocity_db')) if v is not None), default=None)
    return {**a, 'velocity_bin': max(velocity_bin(a), velocity_bin(b)), 'velocity_db': db,
            'pseudo': bool(a.get('pseudo') or b.get('pseudo'))}


def tokenize_chart(task, events, *, duration_seconds=None, seed=0, metadata=None, shuffle_chords=False, grid=None):
    grid = grid or BeatGrid(metadata)
    events = [Event(**e) if isinstance(e, dict) else e for e in events]
    groups = list(_midi_tokens(events, grid) if task == 'midi' else
                  _maimai_tokens(events, grid, random.Random(seed) if shuffle_chords else None))
    first = min(grid.first_beat(), groups[0][0]//96) if groups else grid.first_beat()
    # Audio bounds are half open: a beat exactly at EOF is not a dangling suffix.
    limit = None if duration_seconds is None else grid.second_to_beat(duration_seconds)*96
    last = ((groups[-1][0]//96 if groups else first) if limit is None else math.ceil(limit/96 - 1e-10)-1)
    sequence = [Token('BOUNDARY')]
    beat = first-1
    for tick, payload in groups:
        if limit is not None and tick >= limit-1e-8:
            break
        while beat < tick//96:
            beat += 1
            sequence.append(Token('BEAT', beat%16))
        sequence.extend((Token('MOMENT', tick%96), *payload, Token('EOM')))
    while beat < max(first, last):
        beat += 1
        sequence.append(Token('BEAT', beat%16))
    return sequence, first


@lru_cache(maxsize=4096)
def _mask(task, legal):
    row = np.zeros(TaskVocabulary(task).size, dtype=bool)
    row[list(legal)] = True
    return row


def encode_tokens(task, tokens, *, start_beat=None, metadata=None, grid=None, prior_onsets=()):
    grid = grid or BeatGrid(metadata)
    g = Grammar(task, start_beat=start_beat, metadata=metadata, grid=grid)
    v = g.vocab
    tokens = [v.decode(t) if not isinstance(t, Token) else t for t in tokens]
    ids = [v.encode(t) for t in tokens]
    n = len(ids)
    legal = np.empty((n, v.size), dtype=bool)
    times, queries, moments, states, slots, prevs, roles, supervise = [], [], [], [], [], [], [], []
    rectangles, live, last_spans = [], {}, None
    prior = iter(prior_onsets)
    earlier = next(prior, None)
    history = {}
    candidates = (v.boundary_id,)
    for i, (tid, t) in enumerate(zip(ids, tokens)):
        if tid not in candidates:
            raise ValueError(f'illegal {t} in {task}/{g.phase}')
        g.advance(tid, validate=False)
        candidates = g.legal_tokens()
        legal[i] = _mask(task, candidates)
        times.append(g.time)
        queries.append(g.query_tick)
        moments.append(g.moment_index)
        states.append(g.state_features())
        slots.append(g.slot)
        while earlier is not None and earlier[0] < g.time:
            _, slot, pitch = earlier
            history[slot] = earlier[0], pitch
            earlier = next(prior, None)
        reference = g.previous_pitch
        if history and g.slot in history:
            omitted_t, omitted_pitch = history[g.slot]
            known_t = g.pitch_times.get(g.slot, -math.inf)
            if omitted_t >= known_t:
                reference = omitted_pitch if omitted_t > known_t else max(reference, omitted_pitch)
        prevs.append(reference)
        roles.append(int(t.family not in ('BOUNDARY','BEAT','EOM')))
        is_singleton = g.moment_only and (t.family == 'EOM' or i == 1)
        supervise.append(t.family != 'BOUNDARY' and not is_singleton and not (t.family == 'VELOCITY' and t.value == 0))
        spans = g.spans()
        if spans is not last_spans:
            for span in live.keys() - spans:
                rectangles.append((*span, live.pop(span), i))
            for span in spans:
                live.setdefault(span, i)
            last_spans = spans
    rectangles.extend((*span, start, n) for span,start in live.items())
    # Map distinct score ticks once per chart, not once per chord member.
    physical = {tick: grid.second(tick) for tick in set(times).union(queries)}
    causal = torch.tensor([physical[t] for t in times], dtype=torch.float32)
    mask = torch.tensor(supervise) & (causal >= -1e-6)
    roles_tensor = torch.tensor(roles)
    visibility = torch.zeros_like(roles_tensor) if g.pure_no_history else 1 - roles_tensor
    return EncodedChart(task=task, token_ids=torch.tensor(ids), causal_beats=torch.tensor(times)/96,
        query_beats=torch.tensor(queries)/96, audio_positions=torch.tensor([125*physical[t] for t in queries]),
        onset_seconds=causal, token_steps=torch.arange(n), moment_ids=torch.tensor(moments),
        roles=roles_tensor, history_visibility=visibility, state_features=torch.tensor(states),
        loss_mask=mask, legal_mask=torch.from_numpy(legal), protected_spans=torch.tensor(rectangles,dtype=torch.long).reshape(-1,4),
        current_slots=torch.tensor(slots), previous_pitches=torch.tensor(prevs),
        target_pseudo=torch.tensor([bool(t.pseudo) for t in tokens]),
        target_groups=torch.tensor([t.slot_group for t in tokens]), target_velocity_db=torch.tensor([t.velocity_db for t in tokens]))


def encode_moment_only_chart(task, events, *, duration_seconds=None, metadata=None, residual_intervals=()):
    """Encode a pure SHARED_ROWS timing sequence (BOUNDARY/BEAT/MOMENT/EOM)."""
    from .timbre_tags import MOMENT_TYPE_DIM, merge_intervals
    meta = {**(metadata or {}), 'moment_only': True}
    grid = BeatGrid(meta)
    events = [Event(**e) if isinstance(e, dict) else e for e in events]
    ticks = sorted({int(e.data['start_tick'] if 'start_tick' in e.data else grid.tick(e.time))
                    for e in events if e.kind != 'note_off'})
    first = min(grid.first_beat(), ticks[0] // 96) if ticks else grid.first_beat()
    limit = None if duration_seconds is None else grid.second_to_beat(duration_seconds) * 96
    last = ((ticks[-1] // 96 if ticks else first) if limit is None else math.ceil(limit / 96 - 1e-10) - 1)
    sequence = [Token('BOUNDARY')]
    beat = first - 1
    for tick in ticks:
        if limit is not None and tick >= limit - 1e-8:
            break
        while beat < tick // 96:
            beat += 1
            sequence.append(Token('BEAT', beat % 16))
        sequence.extend((Token('MOMENT', tick % 96), Token('EOM')))
    while beat < max(first, last):
        beat += 1
        sequence.append(Token('BEAT', beat % 16))
    encoded = encode_tokens(task, sequence, start_beat=first, metadata=meta, grid=grid)
    n = len(sequence)
    res_mask = torch.zeros(n, dtype=torch.bool)
    merged = merge_intervals(residual_intervals)
    if merged and n > 2:
        prev_sec = encoded.onset_seconds[:-1].numpy()
        next_sec = encoded.onset_seconds[1:].numpy()
        is_beat_target = np.array([t.family == 'BEAT' for t in sequence[1:]], dtype=bool)
        overlaps = np.zeros(n - 1, dtype=bool)
        for a, b in merged:
            overlaps |= (np.maximum(prev_sec, a) <= np.minimum(np.maximum(next_sec, prev_sec), b) + 1e-4)
        res_mask[1:] = torch.from_numpy(is_beat_target & overlaps) & encoded.loss_mask[1:]
    encoded.residual_neg_mask = res_mask
    encoded.moment_type_targets = torch.zeros((n, MOMENT_TYPE_DIM), dtype=torch.float32)
    encoded.moment_type_weights = torch.zeros((n, MOMENT_TYPE_DIM), dtype=torch.float32)
    encoded.moment_slot6_union = torch.zeros(n, dtype=torch.bool)
    return encoded


def encode_chart(task, events, *, duration_seconds=None, seed=0, metadata=None, midi_history=None, shuffle_chords=False):
    grid = BeatGrid(metadata)
    tokens, first = tokenize_chart(task, events, duration_seconds=duration_seconds, seed=seed, grid=grid,
                                   shuffle_chords=shuffle_chords)
    prior = sorted((grid.tick(start), slot, pitch) for (slot,pitch), intervals in
                   (midi_history.intervals if midi_history else {}).items() for start,_ in intervals)
    return encode_tokens(task, tokens, start_beat=first, metadata=metadata, grid=grid, prior_onsets=prior)


def count_chart_tokens(task, events, *, duration_seconds=None, seed=0, metadata=None, diagnostics=None):
    grid = BeatGrid(metadata)
    tokens, first = tokenize_chart(task, events, duration_seconds=duration_seconds, seed=seed, grid=grid)
    beat, tick, count = first-1, first*96, 0
    first_tick = math.ceil(grid.second_to_beat(0.)*96-1e-8)
    for t in tokens:
        if t.family == 'BEAT':
            beat += 1
            tick = beat*96
        elif t.family == 'MOMENT':
            tick = beat*96+t.value
        count += tick >= first_tick and t.family != 'BOUNDARY' and not (t.family == 'VELOCITY' and t.value == 0)
    return count


def decode_tokens(task, tokens, *, start_beat=None, metadata=None):
    g = Grammar(task, start_beat=start_beat, metadata=metadata)
    for t in tokens:
        g.advance(t)
    return g.finish()
