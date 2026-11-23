"""Physical-second event interchange and waveform-aligned augmentations."""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import random

@dataclass(frozen=True)
class Event:
    time: float
    kind: str
    data: dict = field(default_factory=dict)


@dataclass
class MidiCropHistory:
    """Cropped-out pitched intervals in the same stretched seconds as the window."""
    speed: float
    intervals: dict[tuple[int, int], list[tuple[float, float]]] = field(default_factory=dict)

    def omit(self, note: Event, crop_start: float):
        key = (note.data['slot'], int(note.data['pitch']))
        start = (note.time - crop_start) / self.speed
        end = start + note.data['duration'] / self.speed
        self.intervals.setdefault(key, []).append((start, end))


def quantize_time(seconds: float, rng: random.Random | None = None) -> int:
    """Nearest 10 ms, with unbiased midpoint choice (seedable by the producer)."""
    value = float(seconds) * 100
    low = math.floor(value)
    if math.isclose(value - low, .5, rel_tol=0, abs_tol=1e-9):
        return low + (rng or random).randrange(2)
    return math.floor(value + .5)


def crop_events(events, start: float, duration: float, *, truncate_end=False,
                midi_history: MidiCropHistory | None = None):
    """Crop with unsupervised negative-onset carry notes at the left boundary.

    Training preserves true right-edge durations; only diagnostic excerpts set
    ``truncate_end=True``. Optional loss context records ended pitched intervals during
    this same pass. Legacy explicit MIDI streams are paired before cutting intervals.
    """
    end, result = start + duration, []
    explicit_notes = []
    for value in events:
        e = Event(**value) if isinstance(value, dict) else value
        if e.kind in ('note', 'note_on', 'note_off', 'drum', 'unpitched'):
            from .midi import normalize_event
            e = normalize_event(e)
            if e is None:
                continue
            if e.kind in ('note_on', 'note_off'):
                explicit_notes.append(e)
                continue
        d = dict(e.data)
        note_end = e.time + d.get('duration', 0)
        if 'duration' in d and e.kind in ('note', 'hold', 'touch_hold', 'unpitched'):
            if e.time < start and note_end <= start:
                if midi_history is not None and e.kind == 'note':
                    midi_history.omit(e, start)
                continue
            if e.time >= end:
                continue
            if truncate_end and note_end>end:
                d['duration'] = end-e.time
                d.pop('end_tick',None)  # re-map the shortened interval on its crop grid
            result.append(Event(e.time - start, e.kind, d))
        elif e.kind == 'slide' and e.time < end:
            branches = []
            for branch in d.get('branches', [d]):
                wait = branch.get('wait', d.get('wait', .5))
                duration_ = branch.get('duration', d.get('duration', 1))
                if e.time + wait + duration_ <= start:
                    continue
                # Preserve the original start/route and its elapsed state. The
                # carry prefix is unsupervised; its initial full path is never
                # reinterpreted as a fresh path starting at the crop boundary.
                body = dict(branch, wait=wait, duration=duration_)
                if truncate_end and e.time+wait+duration_>end:
                    body['duration'] = max(0., end-e.time-wait)
                    body.pop('end_tick',None)
                branches.append(body)
            if branches:
                d['branches'] = branches
                result.append(Event(e.time - start, e.kind, d))
        elif start <= e.time < end:
            result.append(Event(e.time - start, e.kind, d))
    if explicit_notes:
        from .midi import events_to_notes
        explicit_notes.sort(key=lambda e: (e.time, e.kind != 'note_off'))
        closed = []
        for e in events_to_notes(explicit_notes, keep_open=True):
            if e.kind == 'note':
                closed.append(e)
            elif e.time < end:
                # Still sounding at the end of the source: keep the observed
                # attack without inventing a release at the crop boundary.
                result.append(Event(e.time - start, e.kind, e.data))
        result.extend(crop_events(closed, start, duration, truncate_end=truncate_end, midi_history=midi_history))
        result.sort(key=lambda e: e.time)
    return result


def stretch_events(events, rate: float):
    result = []
    for value in events:
        e = Event(**value) if isinstance(value, dict) else value
        if e.kind in ('note', 'note_on', 'note_off', 'drum', 'unpitched'):
            from .midi import normalize_event
            e = normalize_event(e)
            if e is None:
                continue
        d = dict(e.data)
        for key in ('duration', 'wait'):
            if d.get(key) is not None:
                d[key] /= rate
        if 'branches' in d:
            d['branches'] = [{**b, **{k: b[k] / rate for k in ('duration', 'wait') if k in b}}
                             for b in d['branches']]
        result.append(Event(e.time / rate, e.kind, d))
    return result


def mirror_events(events):
    from maidata.chart.routes import mirror_pad, mirror_route, route_shape_end, shape_route_table, path_for
    result = []
    for value in events:
        e = Event(**value) if isinstance(value, dict) else value
        d = dict(e.data)
        if 'pad' not in d:
            result.append(e)
            continue
        original_pad = d['pad']
        d['pad'] = mirror_pad(original_pad)
        if e.kind == 'slide':
            branches = []
            for branch in d.get('branches', [d]):
                original, transformed, routes = original_pad, d['pad'], []
                source_routes = branch.get('routes')
                if source_routes is None:
                    source_routes = []
                    source_start = original
                    for rid in branch.get('route_ids', []):
                        shape, target = route_shape_end(source_start, rid)
                        source_routes.append({'shape': shape, 'end': target})
                        source_start = 'A' + str(target + 1)
                for r in source_routes:
                    rid = shape_route_table()[r['shape']][int(original[1]) - 1][r['end']]
                    mirrored = mirror_route(original, rid)
                    shape, target = route_shape_end(transformed, mirrored)
                    routes.append({'shape': shape, 'end': target})
                    original, transformed = 'A' + str(r['end'] + 1), 'A' + str(target + 1)
                branches.append({**branch, 'routes': routes})
            if 'branches' in d:
                d['branches'] = branches
            elif branches:
                d.update(branches[0])
                d['pad'] = mirror_pad(original_pad)
        result.append(Event(e.time, e.kind, d))
    return result
