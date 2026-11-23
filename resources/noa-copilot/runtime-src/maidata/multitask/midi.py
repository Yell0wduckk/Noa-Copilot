"""Standard MIDI ingestion/export, sustain folding and effective velocity.

The small SMF reader has no optional package dependency. It handles format 0/1,
running status, tempo maps, program changes, CC7/11 and CC64 sustain pedal.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math
from operator import attrgetter
from pathlib import Path
import struct

from .codec import Event
from .slots import (PITCHED_SLOTS, UNPITCHED_SLOTS, METALLIC_DRUM_PITCHES,
                    program_slot, drum_slot, event_slot, velocity_bin, velocity_amplitude)


def program_class(program: int | None, *, channel=None, vocal=False, teacher=False, pitch=None):
    """Compatibility name for callers mapping raw MIDI messages to slots."""
    return drum_slot(pitch) if channel == 9 else program_slot(program, teacher=teacher, vocal=vocal)


def quantize_velocities(values, *, reliable=True):
    from .loudness import loudness_targets
    levels = [40 * math.log10(v / 127) if reliable and v is not None and v > 0 else None for v in values]
    return [band for band, _ in loudness_targets(levels)]

@dataclass(frozen=True)
class MidiMessage:
    time: float
    kind: str
    channel: int = 0
    value: int = 0
    value2: int = 0
    track: int = 0


def _shared_program_changes(messages):
    """Older files can put a channel's program on a separate setup track."""
    changes, sounding = {}, set()
    for m in messages:
        if m.kind == 'program_change':
            changes.setdefault(m.channel, {}).setdefault(m.track, []).append(m)
        elif m.kind == 'note_on' and m.value2:
            sounding.add((m.track, m.channel))
    shared = {}
    for channel, tracks in changes.items():
        if -1 in tracks:
            shared[channel] = tracks[-1]
        else:
            setup = [values for track, values in tracks.items() if (track, channel) not in sounding]
            if len(setup) == 1:
                shared[channel] = setup[0]
    return shared


def _note_observations(messages, *, track_gains=None, vocal_tracks=(), teacher=False,
                       default_program=None, force_drum_channel=False, force_slot=None):
    """Fold track-local controls and return events with unquantized amplitudes."""
    gains = track_gains or {}
    programs, volume, expression, pedal = {}, {}, {}, {}
    pressed, sustained = {}, {}
    notes, drums, last_time = [], [], 0.0
    ordered = sorted(messages, key=attrgetter('time'))
    shared = _shared_program_changes(ordered)

    def close(note, time):
        start, cls, pitch, effective, supervision = note
        if time > start:
            data = dict(slot=cls, duration=time - start, pseudo=bool(teacher),
                        slot_supervision=supervision)
            if cls in PITCHED_SLOTS:
                data['pitch'] = pitch
                notes.append((Event(start, 'note', data), effective))
            else:
                drums.append((Event(start, 'unpitched', data), effective))

    for m in ordered:
        last_time = max(last_time, m.time)
        part = (m.track, m.channel)
        key = (part, m.value)
        if m.kind == 'program_change':
            programs[part] = m.value
        elif m.kind == 'control_change':
            if m.value == 7:
                volume[part] = m.value2 / 127
            elif m.value == 11:
                expression[part] = m.value2 / 127
            elif m.value == 64:
                enabled = m.value2 >= 64
                if pedal.get(part, False) and not enabled:
                    for k in list(sustained):
                        if k[0] == part:
                            for note in sustained.pop(k):
                                close(note, m.time)
                pedal[part] = enabled
            elif m.value in (120, 123):
                for collection in (pressed, sustained):
                    for k in list(collection):
                        if k[0] == part:
                            value = collection.pop(k)
                            for note in (value if collection is sustained else [value]):
                                close(note, m.time)
        elif m.kind == 'note_on' and m.value2 > 0:
            effective = m.value2 * volume.get(part, 1) * expression.get(part, 1) * gains.get(m.track, 1)
            if force_drum_channel or (m.channel == 9 and default_program is None and force_slot is None):
                drums.append((Event(m.time, 'unpitched', {'slot': drum_slot(m.value),
                    'metallic': m.value in METALLIC_DRUM_PITCHES, 'pseudo': bool(teacher)}), effective))
                continue
            program = default_program if default_program is not None else programs.get(part)
            if program is None:
                changes = shared.get(m.channel)
                # Track order in the SMF must not decide which program applies
                # when a conductor and a note have the same absolute tick.
                index = (bisect_right(changes, m.time, key=attrgetter('time'))
                         if changes and changes[0].track != m.track else 0)
                program = changes[index - 1].value if index else None
            cls = force_slot if force_slot is not None else program_slot(
                program, teacher=teacher, vocal=default_program is None and m.track in vocal_tracks)
            # Same-pitch retriggers end the preceding pedal tail; this makes
            # (instrument class,pitch) a representable open-note identity.
            if key in pressed:
                close(pressed.pop(key), m.time)
            for note in sustained.pop(key, []):
                close(note, m.time)
            supervision=('pitched' if program is None and force_slot is None and m.track not in vocal_tracks
                         else 'env' if teacher and cls in (6,7) else 'hard')
            pressed[key] = (m.time, cls, m.value, effective, supervision)
        elif m.kind == 'note_off' or m.kind == 'note_on' and m.value2 == 0:
            note = pressed.pop(key, None)
            if note is not None:
                if pedal.get(part, False):
                    sustained.setdefault(key, []).append(note)
                else:
                    close(note, m.time)
    for note in pressed.values():
        close(note, max(last_time, note[0] + .01))
    for row in sustained.values():
        for note in row:
            close(note, max(last_time, note[0] + .01))
    return notes, drums


def _finalize_song(notes, drums, reliable_velocity):
    observations = notes + drums
    from .loudness import loudness_targets
    levels = [e.data.get('loudness_db', 40 * math.log10(max(v, 1e-10) / 127))
              if reliable_velocity else None for e, v in observations]
    targets = loudness_targets(levels)
    return merge_slot_events(Event(e.time, e.kind, {**e.data, 'velocity_bin': band, 'velocity_db': db})
                             for (e, _), (band, db) in zip(observations, targets))



def merge_slot_events(events):
    """Union overlapping pitched intervals and coalesce simultaneous same-slot hits."""
    # Slot merging can overlap the same pitch from separate instruments.
    # Collapse those unison intervals to a single sounding slot/pitch union.
    merged = []
    groups, drums = {}, []
    for e in events:
        if e.kind == 'unpitched':
            drums.append(e)
        else:
            groups.setdefault((e.data['slot'], e.data['pitch']), []).append(e)
    for group in groups.values():
        current = None
        for e in sorted(group, key=lambda x: x.time):
            if current is None:
                current = e
            elif e.time < current.time + current.data['duration']:
                end = max(e.time + e.data['duration'], current.time + current.data['duration'])
                current = Event(current.time, 'note', {**current.data, 'duration': end - current.time,
                                'velocity_bin': max(current.data.get('velocity_bin', 0), e.data.get('velocity_bin', 0)),
                                'velocity_db': max((v for v in (current.data.get('velocity_db'),e.data.get('velocity_db')) if v is not None),default=None)})
            else:
                merged.append(current)
                current = e
        if current is not None:
            merged.append(current)
    # Concurrent transient/FX voices share a slot token, retaining acoustic
    # duration and target annotations for the encoder's auxiliary heads.
    unique = {}
    for e in drums:
        key = (e.time, e.data['slot'])
        old = unique.get(key)
        if old is not None:
            data = dict(e.data, metallic=bool(e.data.get('metallic') or old.data.get('metallic')),
                        pseudo=bool(e.data.get('pseudo') or old.data.get('pseudo')),
                        velocity_bin=max(e.data.get('velocity_bin', 0), old.data.get('velocity_bin', 0)))
            if 'duration' in old.data or 'duration' in e.data:
                data['duration'] = max(e.data.get('duration', 0), old.data.get('duration', 0))
            e = Event(e.time, e.kind, data)
        unique[key] = e
    drums = list(unique.values())
    return sorted(merged + drums, key=lambda e: (e.time, e.data['slot'], e.data.get('pitch', -1)))


def messages_to_events(messages, *, reliable_velocity=True, track_gains=None, vocal_tracks=(),
                       teacher=False, default_program=None, force_drum_channel=False, force_slot=None):
    """Convert timed messages, honoring authoritative dataset instrument labels."""
    notes, drums = _note_observations(messages, track_gains=track_gains, vocal_tracks=vocal_tracks,
                                      teacher=teacher, default_program=default_program,
                                      force_drum_channel=force_drum_channel, force_slot=force_slot)
    return _finalize_song(notes, drums, reliable_velocity)


def _read_var(data, pos):
    result = 0
    while True:
        value = data[pos]
        pos += 1
        result = (result << 7) | (value & 127)
        if value < 128:
            return result, pos


def _write_var(value):
    out = [int(value) & 127]
    value = int(value) >> 7
    while value:
        out.append(128 | (value & 127))
        value >>= 7
    return bytes(reversed(out))


def _read_smf(path, vocal_tracks=()):
    data = Path(path).read_bytes()
    if data[:4] != b'MThd':
        raise ValueError('expected Standard MIDI file')
    length = struct.unpack('>I', data[4:8])[0]
    format_, tracks, division = struct.unpack('>HHH', data[8:14])
    if format_ == 2 or division & 0x8000:
        raise ValueError('independent-pattern/SMPTE MIDI needs conversion to a shared musical clock')
    pos, messages, tempos = 8 + length, [], [(0, 500000)]
    meters = {}
    if isinstance(vocal_tracks, (int, str)):
        vocal_tracks = (vocal_tracks,)
    inferred_vocals = {track for track in vocal_tracks if isinstance(track, int)}
    vocal_names = {name.strip().casefold() for name in vocal_tracks if isinstance(name, str)}
    for track in range(tracks):
        if data[pos:pos + 4] != b'MTrk':
            raise ValueError('missing MIDI track chunk')
        size = struct.unpack('>I', data[pos + 4:pos + 8])[0]
        raw, pos = data[pos + 8:pos + 8 + size], pos + 8 + size
        i, tick, running = 0, 0, None
        while i < len(raw):
            delta, i = _read_var(raw, i)
            tick += delta
            status = raw[i]
            if status < 128:
                status = running
            else:
                i += 1
            if status == 0xFF:
                kind = raw[i]
                size, i = _read_var(raw, i + 1)
                value, i = raw[i:i + size], i + size
                if kind == 0x51:
                    tempos.append((tick, int.from_bytes(value, 'big')))
                elif kind == 0x58 and len(value)>=2:
                    meters[tick/division] = (int(value[0]),1 << value[1])
                elif kind == 3:
                    name = value.decode('utf8', 'ignore').strip().casefold()
                    if name in vocal_names or any(word in name for word in ('vocal', 'voice', 'utau', 'choir')):
                        inferred_vocals.add(track)
                elif kind == 0x2F:
                    messages.append((tick, 'end', 0, 0, 0, track))
                    break
                continue
            if status in (0xF0, 0xF7):
                size, i = _read_var(raw, i)
                i += size
                running = None
                continue
            running = status
            family, channel = status >> 4, status & 15
            count = 1 if family in (0xC, 0xD) else 2
            values = raw[i:i + count]
            i += count
            kind = {0x8: 'note_off', 0x9: 'note_on', 0xB: 'control_change', 0xC: 'program_change'}.get(family)
            if kind:
                messages.append((tick, kind, channel, values[0], values[1] if count == 2 else 0, track))
    tempo_map = sorted(dict(tempos).items())
    controls = []
    last_tick, second, tempo = 0, 0.0, 500000
    for tick, value in tempo_map:
        second += (tick - last_tick) * tempo / division / 1e6
        controls.append((tick, second, value))
        last_tick, tempo = tick, value
    cursor, timed = 0, []
    for tick, kind, channel, v1, v2, track in sorted(messages, key=lambda row: row[0]):
        while cursor + 1 < len(controls) and controls[cursor + 1][0] <= tick:
            cursor += 1
        base, second, tempo = controls[cursor]
        timed.append(MidiMessage(second + (tick - base) * tempo / division / 1e6,
                                 kind, channel, v1, v2, track))
    metadata = dict(bpm_segments=[(second, 60e6 / value) for _, second, value in controls],
                    offset_seconds=0.)
    if meters:
        from .grid import normalize_meter
        metadata['meter_segments'] = [(b, *normalize_meter(*v)) for b, v in sorted(meters.items())]
        metadata['numerator'] = metadata['meter_segments'][0][1]
        metadata['denominator'] = metadata['meter_segments'][0][2]
    return timed, metadata, inferred_vocals


def read_midi(path, *, reliable_velocity=True, track_gains=None, vocal_tracks=(), return_metadata=False,
              teacher=False, pseudo=False, default_program=None, force_drum_channel=False, force_slot=None):
    """Read a song, optionally overriding embedded instruments with known source labels."""
    timed, metadata, inferred_vocals = _read_smf(path, vocal_tracks)
    events = messages_to_events(timed, reliable_velocity=reliable_velocity, track_gains=track_gains,
                                vocal_tracks=inferred_vocals, teacher=teacher or pseudo,
                                default_program=default_program, force_drum_channel=force_drum_channel,
                                force_slot=force_slot)
    return (events, metadata) if return_metadata else events


def read_midi_stems(stems, *, reliable_velocity=True, return_metadata=False, teacher=False, pseudo=False):
    """Combine aligned Slakh stems using each file's tempo map and one song-wide velocity scale."""
    notes, drums, metadata = [], [], {}
    for stem in stems:
        timed, timing, vocals = _read_smf(stem['midi'])
        for key, value in timing.items():
            metadata.setdefault(key, value)
        stem_notes, stem_drums = _note_observations(
            timed, vocal_tracks=vocals, teacher=teacher or pseudo,
            default_program=stem.get('program_num'), force_drum_channel=stem.get('is_drum', False))
        observations = stem_notes + stem_drums
        if reliable_velocity and stem.get('audio'):
            from .audio import read_waveform
            from .loudness import k_weight
            import numpy as np
            waveform = k_weight(read_waveform(stem['audio']).numpy())
            adjusted = []
            for event, velocity in observations:
                span = min(event.data.get('duration', .2), .2) if event.kind == 'note' else .1
                left, right = round(event.time*44100), round((event.time+span)*44100)
                local = waveform[left:right]
                level = -.691 + 10 * math.log10(max(1e-20, float(np.mean(local*local)) if len(local) else 0))
                maximum = max((v for e, v in observations if event.time <= e.time < event.time+span), default=velocity)
                level += 40 * math.log10(max(velocity,1e-10)/max(maximum,1e-10))
                adjusted.append((Event(event.time,event.kind,dict(event.data,loudness_db=level)),velocity))
            stem_notes = [(e,v) for e,v in adjusted if e.kind == 'note']
            stem_drums = [(e,v) for e,v in adjusted if e.kind == 'unpitched']
        elif reliable_velocity:
            stem_notes = [(Event(e.time,e.kind,dict(e.data,loudness_db=None)),v) for e,v in stem_notes]
            stem_drums = [(Event(e.time,e.kind,dict(e.data,loudness_db=None)),v) for e,v in stem_drums]
        notes.extend(stem_notes)
        drums.extend(stem_drums)
    events = _finalize_song(notes, drums, reliable_velocity)
    return (events, metadata) if return_metadata else events


def normalize_event(value):
    """Adapt old raw event JSON to compact slots without changing time."""
    e = Event(**value) if isinstance(value, dict) else value
    slot = event_slot(e.data, e.kind)
    data = {key: value for key, value in e.data.items() if key != 'program'}
    data.update(slot=slot, velocity_bin=velocity_bin(e.data))
    kind = e.kind
    if slot in UNPITCHED_SLOTS:
        if kind == 'note_off':
            return None
        data['metallic'] = bool(data.get('metallic', False) or
                                kind == 'drum' and data.get('pitch') in METALLIC_DRUM_PITCHES)
        data.pop('pitch', None)
        kind = 'unpitched'
    return Event(e.time, kind, data)


def events_to_notes(events, *, end_time=None, keep_open=False):
    """Pair generated true-pitch note_on/off events; normalize legacy JSON."""
    result, active = [], {}
    for value in events:
        e = normalize_event(value)
        if e is None:
            continue
        if e.kind == 'unpitched':
            result.append(e)
            continue
        key = (e.data['slot'], e.data['pitch'])
        if e.kind == 'note_on':
            onset = active.pop(key, None)
            if onset is not None and e.time > onset.time:
                result.append(Event(onset.time, 'note', {**onset.data, 'duration': e.time - onset.time}))
            active[key] = e
        elif e.kind == 'note_off':
            onset = active.pop(key, None)
            if onset is not None:
                result.append(Event(onset.time, 'note', {**onset.data, 'duration': max(.01, e.time - onset.time)}))
        else:
            result.append(e)
    if end_time is not None:
        for onset in active.values():
            result.append(Event(onset.time, 'note', {**onset.data, 'duration': max(.01, end_time - onset.time)}))
    elif keep_open:
        result.extend(active.values())
    return sorted(result, key=lambda e: (e.time, e.data['slot'], e.data.get('pitch', -1)))


def write_midi(events, path, *, end_time=None, metadata=None):
    """Write a 96 PPQ score with its own tempo and meter, preserving note endpoints."""
    from .grid import BeatGrid
    meta = metadata or {}
    grid = BeatGrid(meta)
    # SMF tick zero starts playback. Preserve a positive audio intro as
    # pre-roll, leaving all musical intervals in their original tick units.
    origin = grid.tick(0.)
    events = events_to_notes(events, end_time=end_time)
    representatives = (53, 33, 0, 48)
    channels = {slot: channel for channel, slot in enumerate(PITCHED_SLOTS)}
    tick_tempos = grid.step_tempos()
    tick_tempos.setdefault(origin, 60/(96*(grid.second(origin+1)-grid.second(origin))))
    messages = [(t-origin,-3,b'\xff\x51\x03'+round(60e6/bpm).to_bytes(3,'big'))
                for t,bpm in tick_tempos.items() if t>=origin]
    for beat,n,d in meta.get('meter_segments',()):
        messages.append((max(0,round(beat*96)-origin),-2,
                         b'\xff\x58\x04'+bytes((int(n),int(d).bit_length()-1,24,8))))
    for channel, program in enumerate(representatives):
        messages.append((0, -1, bytes((0xC0 | channel, program))))
    for e in events:
        slot = e.data['slot']
        if slot in PITCHED_SLOTS:
            channel, pitch = channels[slot], int(e.data['pitch'])
        else:
            channel, pitch = 9, {0: 36, 1: 38, 2: 42, 3: 49}[slot]
        value = int(e.data.get('velocity_bin', 0))
        velocity = 64 if value == 0 else max(1, round(velocity_amplitude(value)))
        a = e.data['start_tick'] if 'start_tick' in e.data else grid.tick(e.time)
        z = e.data['end_tick'] if 'end_tick' in e.data else grid.tick(e.time + e.data.get('duration', .01))
        onset = max(0,a-origin)
        stop = max(onset+1,z-origin)
        messages.append((onset, 1, bytes((0x90 | channel, pitch, velocity))))
        messages.append((stop, 0, bytes((0x80 | channel, pitch, 0))))
    raw, previous = bytearray(), 0
    for tick, _, payload in sorted(messages, key=lambda x: x[:2]):
        raw.extend(_write_var(tick - previous))
        raw.extend(payload)
        previous = tick
    raw.extend(b'\x00\xff\x2f\x00')
    Path(path).write_bytes(b'MThd' + struct.pack('>IHHH', 6, 0, 1, 96) + b'MTrk' + struct.pack('>I', len(raw)) + raw)
    return Path(path)
