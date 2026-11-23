"""BMS/BME/BML/PMS and BMSON autoplay audio plus compact acoustic Events.

The parser follows the authors' BMS/BML specifications and BMSON 1.0:
https://bm98.yaneu.com/bm98/bmsformat.html
https://nvyu.net/rdm/rby_ex.php
https://github.com/lifthrasiir/angolmois/blob/master/INTERNALS.md
https://github.com/bemusic/bmson-spec/blob/master/doc/index.rst

WAV IDs, not key lanes or acoustic slots, have polyphony one. Separate IDs
referencing the same file remain separate voices. BMSON duplicate notes share
one slice; continuation slices retain their source offsets. At a stop position,
notes sound first, BPM changes next, and STOP uses the resulting BPM.

Rendering includes every selected autoplay voice. Pitch/slot estimation is an
acoustic heuristic, not a claim that a mixed sample has lossless MIDI labels.
No teachers are loaded here: composite BGM transcription is an explicit callable.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass, field, replace
from fractions import Fraction
from functools import cache
from itertools import groupby
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess

import numpy as np
import torch

from .codec import Event
from .slots import PITCHED_SLOTS, UNPITCHED_SLOTS

EXTENSIONS = {'.bms', '.bme', '.bml', '.pms', '.bmson'}
AUDIO_EXTENSIONS = {'.wav', '.ogg', '.flac', '.mp3', '.m4a', '.aac', '.opus'}
SAMPLE_RATE = 44100


@dataclass
class Trigger:
    sound: str
    time: float
    bgm: bool
    lane: str
    end: float | None = None
    source_offset: float = 0.
    source_duration: float | None = None
    voice: str | None = None
    hold_duration: float | None = None  # BMSON label only; independent of the slice.
    seamless_end: bool = False


@dataclass
class Chart:
    path: Path
    samples: dict[str, str]
    triggers: list[Trigger]
    metadata: dict = field(default_factory=dict)


class BeatClock:
    """Piecewise physical clock; an event at a STOP starts before the pause."""
    def __init__(self, bpm, changes=(), stops=()):
        tempo = {Fraction(beat): float(value) for beat, value in changes}
        pause = defaultdict(float)
        for beat, duration in stops:
            pause[Fraction(beat)] += float(duration)
        self.beats, self.before, self.after, self.bpms = [], [], [], []
        self.segments, self.stops = [[0., float(bpm)]], []
        last, elapsed, current = Fraction(0), 0., float(bpm)
        for beat in sorted({Fraction(0), *tempo, *pause}):
            elapsed += float(beat - last) * 60. / current
            current = tempo.get(beat, current)
            if current <= 0:
                raise ValueError(f'Cannot render nonpositive BMS tempo at beat {beat}: {current}')
            self.beats.append(beat)
            self.before.append(elapsed)
            self.bpms.append(current)
            if beat in tempo:
                if beat == 0:
                    self.segments[0] = [elapsed, current]
                else:
                    self.segments.append([elapsed, current])
            seconds = pause[beat] * 60. / current
            if seconds:
                self.stops.append(dict(beat=float(beat), time=elapsed, duration=seconds))
            elapsed += seconds
            self.after.append(elapsed)
            last = beat

    def seconds(self, beat):
        beat = Fraction(beat)
        i = bisect_right(self.beats, beat) - 1
        if beat == self.beats[i]:
            return self.before[i]
        return self.after[i] + float(beat - self.beats[i]) * 60. / self.bpms[i]


def _read_text(path, encoding=None, payload=None):
    if payload is None:
        payload = path.read_bytes()
    if encoding:
        return payload.decode(encoding)
    if payload.startswith((b'\xff\xfe', b'\xfe\xff')):
        return payload.decode('utf-16')
    # Some legacy editors prepend a UTF-8 BOM to a still-CP932 body.
    payload = payload.removeprefix(b'\xef\xbb\xbf')
    try:
        return payload.decode('utf-8')
    except UnicodeDecodeError:
        try:
            return payload.decode('cp932')
        except UnicodeDecodeError:
            return payload.decode('gb18030')


def _fixed_lines(text, random_seed=0, random_value=1):
    """Resolve nested RANDOM and fall-through SWITCH blocks reproducibly."""
    rng = random.Random(random_seed)
    randoms, conditions, switches, choices, lines = [], [], [], [], []
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith('#'):
            continue
        match = re.match(r'#(RANDOM|SETRANDOM|ENDRANDOM|IF|ELSEIF|ELSE|ENDIF|'
                         r'SWITCH|SETSWITCH|CASE|DEF|SKIP|ENDSW)\b\s*(.*)', line, re.I)
        active = all(f['active'] for f in conditions) and all(s['active'] for s in switches)
        if not match:
            if active:
                lines.append(line)
            continue
        command, argument = match.group(1).upper(), match.group(2).split('//')[0].strip()
        if command in ('RANDOM', 'SETRANDOM', 'SWITCH', 'SETSWITCH'):
            n = int(argument)
            choice = (n if command.startswith('SET') else
                      rng.randint(1, n) if random_value is None else 1 + (int(random_value) - 1) % n)
            if command.endswith('SWITCH'):
                switches.append(dict(choice=choice, active=False, skipped=False))
            else:
                randoms.append(choice)
            if active:
                choices.append(dict(command=command, range=n, selected=choice))
        elif command == 'ENDRANDOM':
            if randoms:
                randoms.pop()
        elif command == 'IF':
            matched = bool(randoms and randoms[-1] == int(argument))
            conditions.append(dict(parent=active, active=active and matched, matched=matched))
        elif command in ('ELSE', 'ELSEIF') and conditions:
            frame = conditions[-1]
            matched = command == 'ELSE' or bool(randoms and randoms[-1] == int(argument))
            frame['active'] = frame['parent'] and not frame['matched'] and matched
            frame['matched'] |= matched
        elif command == 'ENDIF' and conditions:
            conditions.pop()
        elif command in ('CASE', 'DEF') and switches:
            switch = switches[-1]
            if not switch['skipped'] and (command == 'DEF' or int(argument) == switch['choice']):
                switch['active'] = True
        elif command == 'SKIP' and switches and active:
            switches[-1].update(active=False, skipped=True)
        elif command == 'ENDSW' and switches:
            switches.pop()
    return lines, choices


def parse_bms(source, *, random_seed=0, random_value=1, encoding=None, payload=None):
    """Read timing and sample triggers only; no audio is opened here."""
    path = Path(source)
    if path.suffix.lower() == '.bmson':
        return _parse_bmson(path, payload=payload)
    lines, choices = _fixed_lines(_read_text(path, encoding, payload=payload), random_seed, random_value)
    # Keep the original two-character keys for provenance. In base 62 each
    # uppercase/lowercase pair identifies different voices and timing objects.
    base62 = any(re.match(r'#BASE\s+62\b', line, re.I) for line in lines)
    object_key = str if base62 else str.upper
    headers, samples, bpms, stops, measures, rows = {}, {}, {}, {}, {}, []
    for line in lines:
        message = re.match(r'#(\d{3})([0-9A-Z]{2})\s*:\s*(.*)', line, re.I)
        if message:
            measure, channel, value = int(message[1]), message[2].upper(), message[3].split('//')[0].strip()
            if channel == '02':
                measures[measure] = Fraction(value)
            else:
                rows.append((measure, channel, object_key(''.join(value.split()))))
            continue
        header = re.match(r'#([^\s]+)\s+(.*)', line)
        if not header:
            continue
        name, value = header[1].upper(), header[2].split('//')[0].strip().strip('"')
        if re.fullmatch(r'WAV[0-9A-Z]{2}', name):
            samples[object_key(header[1][-2:])] = value
        elif re.fullmatch(r'(?:EX)?BPM[0-9A-Z]{2}', name):
            bpms[object_key(header[1][-2:])] = float(value)
        elif re.fullmatch(r'STOP[0-9A-Z]{2}', name):
            stops[object_key(header[1][-2:])] = float(value) / 48.
        else:
            headers[name] = value
    from .grid import multiplier_to_meter
    max_measure = max([m for m, _, _ in rows] + list(measures) + [0])
    starts = [Fraction(0)]
    meter_segments = []
    for m in range(max_measure + 1):
        mult = measures.get(m, Fraction(1))
        num, den = multiplier_to_meter(float(mult))
        start_beat = float(starts[-1])
        if not meter_segments or meter_segments[-1][1:] != [num, den] or abs(4. * num / den - 4. * float(mult)) > 1e-6:
            meter_segments.append([start_beat, num, den])
        starts.append(starts[-1] + 4 * mult)
    lane_rows, bgm_rows, timing, ln_cells = defaultdict(dict), [], {}, defaultdict(dict)
    warnings = []
    lntype = int(headers.get('LNTYPE', 1))
    for measure, channel, payload in rows:
        count = len(payload) // 2
        span = starts[measure + 1] - starts[measure]
        for i in range(count):
            value = payload[2*i:2*i+2]
            beat = starts[measure] + span * Fraction(i, count)
            if value == '00':
                continue
            if lntype == 2 and channel[0] in '56' and channel[1] in '123456789':
                # Each nonempty cell covers its own row's subdivision. Empty
                # cells in other rows cannot remove any part of that interval.
                end = beat + span / count
                previous = ln_cells[channel].get(beat)
                if previous is not None:
                    end = max(end, previous[1])
                ln_cells[channel][beat] = (value, end)
            if channel == '01':
                bgm_rows.append((beat, value))
            elif channel in ('03', '08', '09'):
                timing[(beat, channel)] = value
            elif channel[0] in '1256' and channel[1] in '123456789':
                lane_rows[channel][beat] = value
    changes, pauses = {}, []
    for (beat, channel), value in sorted(timing.items()):
        if channel == '03':
            changes[beat] = int(value, 16)
        elif channel == '08':
            changes[beat] = bpms[value]
        elif value in stops:
            pauses.append((beat, stops[value]))
        else:
            warnings.append(f'Ignored undefined STOP{value} at beat {beat}')
    raw_triggers = [(beat, value, True, '01', None) for beat, value in bgm_rows]
    lnobj = object_key(headers.get('LNOBJ', ''))
    for lane, objects in lane_rows.items():
        ordered = sorted(objects.items())
        if lane[0] in '12':
            pending = None
            for beat, value in ordered:
                if value == lnobj and pending is not None:
                    old = raw_triggers[pending]
                    raw_triggers[pending] = (*old[:4], beat)
                    # RDM's LNOBJ extension permits a defined end marker WAV
                    # as BGM. Undefined markers are purely structural.
                    if value in samples:
                        raw_triggers.append((beat, value, True, lane + ':release', None))
                    pending = None
                elif value != lnobj:
                    pending = len(raw_triggers)
                    raw_triggers.append((beat, value, False, lane, None))
        elif lntype == 2:
            opened, end = None, None
            for beat, (value, following) in sorted(ln_cells[lane].items()):
                if opened is not None and beat > end:
                    raw_triggers.append((opened[0], opened[1], False, lane, end))
                    opened = None
                if opened is None:
                    opened, end = (beat, value), following
                else:
                    end = max(end, following)
            if opened is not None:
                raw_triggers.append((opened[0], opened[1], False, lane, end))
        else:
            for i in range(0, len(ordered), 2):
                beat, value = ordered[i]
                end = ordered[i+1][0] if i+1 < len(ordered) else None
                if end is None:
                    warnings.append(f'Unclosed LN in channel {lane} at beat {beat}; retaining its sample tail')
                raw_triggers.append((beat, value, False, lane, end))
    # A beat-zero change sets the clock before any elapsed time, even when
    # the display BPM in the header is a range rather than a single number.
    reverse_scroll = None
    unsupported = [beat for beat, bpm in changes.items() if bpm <= 0]
    if unsupported:
        first = min(unsupported)
        if changes[first] < 0:
            # Angolmois/LR2-style reverse scrolling cannot reach later beats.
            # Already playing WAVs continue, including holds whose release
            # lies beyond the reverse point. Notes at the boundary sound first.
            reverse_scroll = dict(beat=float(first), bpm=changes[first],
                                  unreachable_triggers=sum(beat > first for beat, *_ in raw_triggers),
                                  unreachable_releases=sum(beat <= first < end for beat, _, _, _, end
                                                           in raw_triggers if end is not None))
            raw_triggers = [(beat, value, bgm, lane, end if end is None or end <= first else None)
                            for beat, value, bgm, lane, end in raw_triggers if beat <= first]
            warnings.append(f'Reverse scroll at beat {first}: {reverse_scroll["unreachable_triggers"]} '
                            'later sound events are unreachable; retaining already triggered sample tails')
        last_sound = max((end if end is not None else beat
                          for beat, _, _, _, end in raw_triggers), default=0)
        if first >= last_sound:
            # Onsets/releases precede a tempo change at the same beat. A
            # terminal visual effect therefore leaves all sample tails intact.
            changes = {beat: bpm for beat, bpm in changes.items() if beat < first}
            pauses = [(beat, duration) for beat, duration in pauses if beat < first]
            warnings.append(f'Omitted timing tail from beat {first} after last sound at {last_sound}')
    initial_bpm = changes[0] if 0 in changes else float(headers.get('BPM', 130))
    clock = BeatClock(initial_bpm, changes.items(), pauses)
    triggers = [Trigger(value, clock.seconds(beat), bgm, lane,
                        None if end is None else clock.seconds(end), voice=value)
                for beat, value, bgm, lane, end in raw_triggers]
    if 'MIDIFILE' in headers:
        warnings.append('Legacy #MIDIFILE synthesis is not represented by WAV samples')
    metadata = dict(format=path.suffix.lower()[1:], title=headers.get('TITLE'), artist=headers.get('ARTIST'),
                    genre=headers.get('GENRE'),
                    bpm_segments=clock.segments, stops=clock.stops, measure_multipliers={str(k): float(v) for k,v in measures.items()},
                    meter_segments=meter_segments, numerator=meter_segments[0][1], denominator=meter_segments[0][2],
                    bar_beats=[float(b) for b in starts],
                    offset_seconds=0., random_choices=choices, random_seed=random_seed,
                    random_value=random_value, volume=float(headers.get('VOLWAV', 100))/100.,
                    sample_root=headers.get('PATH_WAV', ''), warnings=warnings,
                    ln_policy='RDM endpoints silent; defined LNOBJ release as BGM; sample/label truncated at LN end')
    if reverse_scroll is not None:
        metadata['reverse_scroll'] = reverse_scroll
    return Chart(path, samples, sorted(triggers, key=lambda t: (t.time, t.sound, t.lane)), metadata)


def _parse_bmson(path, payload=None):
    from .grid import multiplier_to_meter
    text = path.read_text(encoding='utf-8-sig') if payload is None else payload.decode('utf-8-sig')
    document = json.loads(text)
    info = document.get('info', {})
    resolution = abs(int(info.get('resolution') or 240))
    def beat(value):
        return Fraction(value) / resolution
    clock = BeatClock(info.get('init_bpm', info.get('initBPM', 130)),
        [(beat(e['y']), e.get('bpm', e.get('v'))) for e in document.get('bpm_events', document.get('bpmNotes', []))],
        [(beat(e['y']), beat(e.get('duration', e.get('v')))) for e in document.get('stop_events', document.get('stopNotes', []))])
    raw_lines = document.get('lines') or document.get('barLines') or []
    bar_beats = sorted({float(beat(item['y'])) for item in raw_lines
                        if isinstance(item, dict) and 'y' in item and beat(item['y']) >= 0})
    if bar_beats and bar_beats[0] > 0.:
        bar_beats.insert(0, 0.)
    meter_segments = []
    for b0, b1 in zip(bar_beats, bar_beats[1:]):
        if b1 > b0:
            num, den = multiplier_to_meter((b1 - b0) / 4.)
            if not meter_segments or meter_segments[-1][1:] != [num, den] or abs(4. * num / den - (b1 - b0)) > 1e-6:
                meter_segments.append([b0, num, den])
    if not meter_segments:
        meter_segments = [[0., 4, 4]]
    samples, triggers = {}, []
    for channel_index, channel in enumerate(document.get('sound_channels', document.get('soundChannel', []))):
        sound = str(channel_index)
        samples[sound] = channel['name']
        positions = defaultdict(list)
        for note in channel['notes']:
            positions[note['y']].append(note)
        ordered = sorted(positions)
        offset, previous = 0., 0.
        for i, position in enumerate(ordered):
            notes = positions[position]
            time = clock.seconds(beat(position))
            offset = offset + time - previous if i and all(n.get('c', False) for n in notes) else 0.
            playable = [n for n in notes if n.get('x')]
            used = playable or notes
            lengths = [n.get('l', 0) for n in used]
            hold_duration = clock.seconds(beat(position + max(lengths))) - time if max(lengths) > 0 else None
            duration = clock.seconds(beat(ordered[i+1])) - time if i+1 < len(ordered) else None
            seamless_end = i+1 < len(ordered) and all(n.get('c', False) for n in positions[ordered[i+1]])
            triggers.append(Trigger(sound, time, not playable, ','.join(str(n.get('x', 0)) for n in used),
                                    source_offset=offset, source_duration=duration,
                                    voice=f'{sound}:{position}', hold_duration=hold_duration, seamless_end=seamless_end))
            previous = time
    return Chart(path, samples, sorted(triggers, key=lambda t: (t.time, t.sound)),
                 dict(format='bmson', title=info.get('title'), artist=info.get('artist'),
                      genre=info.get('genre'),
                      bpm_segments=clock.segments, stops=clock.stops, meter_segments=meter_segments,
                      numerator=meter_segments[0][1], denominator=meter_segments[0][2],
                      bar_beats=bar_beats, offset_seconds=0.,
                      resolution=resolution, random_choices=[], volume=1., warnings=[]))


def is_appledouble(path):
    if not path.name.startswith('._') or not path.is_file():
        return False
    with path.open('rb') as stream:
        return stream.read(4) == b'\x00\x05\x16\x07'


def _directory_layout(files):
    suffixes = {Path(name).suffix.lower() for name in files}
    return bool(suffixes & EXTENSIONS), bool(suffixes & AUDIO_EXTENSIONS)


def _declared_samples(path):
    # Project discovery needs filenames only, without rendering, audio reads,
    # or requiring a chart's timing/control flow to be parseable.
    try:
        if path.suffix.lower() == '.bmson':
            chart = json.loads(path.read_text(encoding='utf-8-sig'))
            channels = chart.get('sound_channels', chart.get('soundChannel', []))
            return '', [c['name'] for c in channels]
        lines = _read_text(path)
    except (UnicodeError, ValueError, KeyError, TypeError, AttributeError):
        return '', []
    root = re.search(r'^\s*#PATH_WAV\s+([^\r\n]+)', lines, re.I | re.M)
    names = re.findall(r'^\s*#WAV[0-9a-z]{2}[ \t]+([^\r\n]+)', lines, re.I | re.M)
    def value(text):
        return text.split('//')[0].strip().strip('"').replace('\\', '/')
    return stored_bms_path(value(root[1])) if root else '', [value(n) for n in names]


def _uses_parent_samples(directory, index):
    for script in directory.iterdir():
        if script.suffix.lower() not in EXTENSIONS or not script.is_file() or is_appledouble(script):
            continue
        root, names = _declared_samples(script)
        for name in names:
            if (_find_sample(directory / root, name, index) is None and
                    _find_sample(directory.parent / root, name, index) is not None):
                return True
    return False


def _song_root(directory, layout, boundary, index):
    while directory != boundary and directory.parent != directory:
        # Common archives keep alternate charts in song/sabun[/author].
        alias = next((p for p in (directory, *directory.parents)
                      if p != boundary and boundary in p.parents
                      and re.fullmatch(r'(?:sabun|差分|difficulties)(?:[-_ ].*)?', p.name, re.I)), None)
        # A complete song inside a library itself called "sabun" still owns
        # its local audio. Stop at any intervening independent chart project.
        between = (p for p in (directory, *directory.parents) if alias is not None and alias in p.parents)
        if alias is not None and not any(layout(p)[1] or (p != directory and layout(p)[0]) for p in between):
            directory = alias.parent
            continue
        charts, audio = layout(directory)
        if not charts or audio or not _uses_parent_samples(directory, index):
            break
        directory = directory.parent
    return directory


def bms_song_directory(source):
    """Resolve a single chart/directory for manual source-index deduplication."""
    path = Path(source).resolve()
    directory = path.parent if path.suffix.lower() in EXTENSIONS and not path.is_dir() else path
    @cache
    def layout(folder):
        return _directory_layout(p.name for p in folder.iterdir() if p.is_file() and not is_appledouble(p))
    return _song_root(directory, layout, Path(directory.anchor), cache(_sample_directory_index))


def _project_scripts(root):
    layouts, charts = {}, {}
    for current, _, files in os.walk(root):
        directory = Path(current)
        files = [f for f in files if not (f.startswith('._') and is_appledouble(directory / f))]
        layouts[directory] = _directory_layout(files)
        if layouts[directory][0]:
            charts[directory] = [directory / f for f in files if Path(f).suffix.lower() in EXTENSIONS]
    result = defaultdict(list)
    index = cache(_sample_directory_index)
    for directory, scripts in charts.items():
        owner = _song_root(directory, layouts.__getitem__, root, index)
        result[owner].extend(scripts)
    return result


def discover_bms_projects(root):
    """Return one directory per song, including BMS and BMSON difficulty aliases."""
    return sorted(_project_scripts(Path(root).resolve()))


def select_script(source, **options):
    """Prefer deterministic scripts, then most sounding triggers, then name."""
    source = Path(source)
    if not source.is_dir():
        return parse_bms(source, **options)
    source = source.resolve()
    candidates = [parse_bms(path, **options) for path in sorted(_project_scripts(source).get(source, []))]
    if not candidates:
        raise FileNotFoundError(f'No BMS/BMSON script in {source}')
    def score(chart):
        deterministic = all(c['command'].startswith('SET') or c['range'] == 1 for c in chart.metadata['random_choices'])
        count = len({(t.time, t.voice) for t in chart.triggers})
        return not deterministic, -count, chart.path.name.casefold()
    chosen = min(candidates, key=score)
    chosen.metadata['representatives_considered'] = [dict(path=str(c.path), triggers=len(c.triggers)) for c in candidates]
    return chosen


def _sample_directory_index(folder):
    by_name, by_stem = {}, {}
    for path in sorted(folder.iterdir()):
        by_name[path.name.casefold()] = path
        if path.suffix.lower() in AUDIO_EXTENSIONS and path.is_file():
            by_stem.setdefault(path.stem.casefold(), path)
    return by_name, by_stem


def stored_bms_path(name):
    """Shorten archive components exceeding Linux NAME_MAX, retaining normal names."""
    if len(name) <= 63 or len(name.encode('utf-8')) <= 255:
        return name
    parts = name.split('/')
    for i, part in enumerate(parts):
        if len(part.encode('utf-8')) <= 255:
            continue
        stem, ext = os.path.splitext(part)
        if len(ext.encode('utf-8')) > 32:
            stem, ext = part, ''
        prefix = stem.encode('utf-8')[:72].decode('utf-8', errors='ignore')
        digest = hashlib.sha256(stem.encode('utf-8')).hexdigest()
        parts[i] = prefix + '~' + digest + ext
    return '/'.join(parts)


def _find_sample(root, filename, index):
    filename = stored_bms_path(filename.replace('\\', '/'))
    path = root / filename
    if path.is_file():
        return path
    # BMS archives commonly declare .wav while distributing .ogg, and were
    # authored on case-insensitive systems. Resolve only the requested folder.
    folder = path.parent
    if not folder.is_dir():
        current = root
        for part in Path(filename.replace('\\', '/')).parts[:-1]:
            if not current.is_dir():
                return None
            current = index(current)[0].get(part.casefold(), current / part)
        folder = current
    if folder.is_dir():
        by_name, by_stem = index(folder)
        exact = by_name.get(path.name.casefold())
        if exact and exact.is_file():
            return exact
        return by_stem.get(path.stem.casefold())
    return None


def _resolve_sample(script_dir, song_parent, sample_root, filename, index):
    relative = stored_bms_path(sample_root.replace('\\', '/'))
    bases = [script_dir]
    if song_parent in script_dir.parents:
        for ancestor in script_dir.parents:
            bases.append(ancestor)
            if ancestor == song_parent:
                break
    for base in bases:
        path = _find_sample(base / relative, filename, index)
        if path is not None:
            return path.resolve()
    raise FileNotFoundError(f'BMS referenced audio slice is missing: {script_dir / relative / filename}')


def _read_sample(path, sample_rate):
    import soundfile as sf
    from .audio import read_waveform
    try:
        return read_waveform(path, sample_rate=sample_rate).numpy()
    except sf.LibsndfileError:
        if path.suffix.lower() not in ('.m4a', '.aac', '.opus'):
            raise
    # libsndfile builds vary in codec support. Use FFmpeg only on demand.
    import io
    import subprocess
    result = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(path),
                             '-map', '0:a:0', '-c:a', 'pcm_f32le', '-f', 'wav', '-'],
                            capture_output=True, check=True)
    return read_waveform(io.BytesIO(result.stdout), sample_rate=sample_rate).numpy()


_EXPLICIT_NAME_GROUPS = (
    (3, ('choir_fx',), ()),
    (0, ('bassdrum', 'deepkick', 'subkick', 'hardkick', 'kickroll', 'fxkick', '808kick', '909kick', 'kick'),
        ('bd', 'kck', 'kik', 'bd1', 'bd2')),
    (1, ('snareroll', 'snare', 'handclap', 'handcrap', 'clap', 'rimshot', 'sidestick', 'fingersnap'),
        ('sd', 'sn', 'snr', 'clp', 'crap', 'rim', 'snap')),
    (2, ('hihat', 'hi_hat', 'openhat', 'closehat', 'closedhat', 'clhat', 'ophat', 'openhh', 'closehh',
         'ridebell', 'tambourine', 'shaker', 'conga', 'bongo', 'cowbell', 'claves', 'woodblock', 'taiko',
         'percussion', 'agogo', 'cabasa', 'maracas', 'timbale', 'djembe', 'cajon', 'castanet', 'guiro',
         'cuica', 'whistle', 'floortom', 'flortom', 'midtom', 'hitom', 'lowtom'),
        ('hat', 'chh', 'ohh', 'hho', 'hhc', 'hh', 'oh', 'ch', 'ride', 'rd', 'tom', 'perc', 'tamb', 'tam', 'triangle')),
    (6, ('harpsichord', 'glockenspiel', 'glock', 'vibraphone', 'marimba', 'xylophone', 'xylo', 'kalimba',
         'timpani', 'celesta', 'epiano', 'elepi', 'piano', 'rhodes', 'wurlitzer', 'wurli', 'clavinet', 'clav',
         'musicbox', 'orgel', 'steeldrum', 'dulcimer', 'zither', 'mallet', 'pizzicato', 'pizz', 'arpeggio',
         'arrpegio', 'pluck', 'guitar', 'guiter', 'acogi', 'shamisen', 'koto', 'sitar', 'banjo', 'mandolin',
         'ukulele', 'harp', 'chime', 'bell'),
        ('pf', 'pia', 'pi', 'ep', 'gt', 'gtr', 'guit', 'aco', 'ag', 'eg', 'distgt', 'arp', 'pl', 'plk', 'vib', 'vibe', 'keys')),
    (4, ('vocaloid', 'vocal', 'voice', 'utau', 'whisper', 'scream', 'laugh', 'breath'),
        ('vo', 'vox', 'voc', 'sing', 'rap', 'uta', 'koe')),
    (5, ('synthbass', 'subbass', 'acidbass', 'neurobass', 'fmbass', 'ebass', 'pickbass', 'fingerbass',
         '808bass', 'wobble', 'reese', 'tb303', 'growl', 'donk', 'fretless', 'bass', 'acid'),
        ('bs', '808', 'slap')),
    (7, ('supersaw', 'synthlead', 'sawlead', 'squarelead', 'violoncello', 'contrabass', 'violin', 'viola',
         'cello', 'strings', 'string', 'orchestra', 'trumpet', 'trombone', 'frenchhorn', 'saxophone', 'brass',
         'piccolo', 'clarinet', 'bassoon', 'recorder', 'ocarina', 'flute', 'oboe', 'woodwind', 'accordion',
         'harmonica', 'bagpipe', 'organ', 'choir_pad', 'choir', 'hoover', 'lead'),
        ('ld', 'pd', 'pad', 'str', 'vln', 'vio', 'orc', 'fl', 'br', 'tp', 'tb', 'tuba', 'sax', 'horn', 'org', 'wind', 'saw', 'mainst', 'backst')),
    (3, ('downlifter', 'uplifter', 'tapestop', 'scratch', 'reverse', 'cymbal', 'symbal', 'revcym', 'crash',
         'clash', 'china', 'splash', 'riser', 'sweep', 'impact', 'whoosh', 'noise', 'noize', 'noiz', 'laser',
         'siren', 'alarm', 'glitch', 'vinyl', 'boom', 'sfx'),
        ('cym', 'fx', 'se', 'scr', 'rev', 'cc')),
    (7, (), ('chord', 'poly', 'solo')),
    (5, (), ('sub',)),
)

_NAMES = tuple((slot, compounds + tokens) for slot, compounds, tokens in _EXPLICIT_NAME_GROUPS)


def _compile_group_pattern(compounds, tokens):
    parts = [ '|'.join(map(re.escape, compounds)) ] if compounds else []
    if tokens:
        parts.append(r'(?<![a-z])(?:' + '|'.join(map(re.escape, tokens)) + r')s?(?![a-z])')
    return re.compile('|'.join(parts))


_NAME_PATTERNS = [(slot, _compile_group_pattern(compounds, tokens))
                  for slot, compounds, tokens in _EXPLICIT_NAME_GROUPS]
_TRAILING_NOTE_SUFFIX = re.compile(r'_(?:\d{3,5}_[a-g][a-gp#b]*|(?:v\d+)?l[\d-]+o[0-8][a-g](?:p|s|#)?)$', re.I)


def _name_slot(filename):
    stem = _TRAILING_NOTE_SUFFIX.sub('', Path(filename).stem)
    stem = re.sub(r'([a-z])([A-Z])', r'\1 \2', stem)
    stem = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', stem).casefold()
    for slot, pattern in _NAME_PATTERNS:
        if pattern.search(stem):
            return slot
    return None


_NOTE_NAME = re.compile(r'(?:^|_)(?:v\d+)?l[\d-]+o([0-8])([a-g])(p|s|#)?(?=$|_|\.)', re.I)


def _filename_pitch(filename):
    match = _NOTE_NAME.search(Path(filename).name)
    if match:
        octave, note, sharp = match.groups()
        semitone = {'c': 0, 'd': 2, 'e': 4, 'f': 5, 'g': 7, 'a': 9, 'b': 11}[note.lower()]
        return (int(octave) + 1) * 12 + semitone + bool(sharp)
    return None


def keysound_root_pitch(named, f0):
    """MML octave numbering differs across producers. Retain the name's pitch class."""
    if f0:
        measured = 69 + 12 * math.log2(f0 / 440)
        octave = round((measured - named) / 12)
        candidate = named + 12 * octave
        if abs(octave)==1 and 0 <= candidate <= 127 and abs(candidate - measured) < .75:
            return candidate
    return named


def _pitch_features(waveform, sample_rate):
    """YIN cumulative difference, evaluated on a bounded set of dry frames."""
    from .audio import fast_resample
    core = waveform[:round(.35 * sample_rate)]
    signal = fast_resample(torch.as_tensor(core), sample_rate, 11025).numpy().astype(np.float64)
    size = min(2048, len(signal))
    if size < 96:
        return None, 0., False, 0., 0.
    low, high = 5, min(size//2, 368)
    pitches, confidences, residuals, centroids = [], [], [], []
    frequency = np.fft.rfftfreq(size, 1/11025)
    window = np.hanning(size)
    for start in np.linspace(0, max(0, len(signal)-size), min(16, max(1, len(signal)//512)), dtype=int):
        frame = signal[start:start+size]
        frame = frame - frame.mean()
        if np.max(np.abs(frame)) < 1e-8:
            continue
        nfft = 1 << (2*len(frame)-1).bit_length()
        power = np.abs(np.fft.rfft(frame, nfft))**2
        corr = np.fft.irfft(power, nfft)
        prefix = np.r_[0., np.cumsum(frame**2)]
        lag = np.arange(high+1)
        difference = np.maximum(0., prefix[size-lag] + prefix[size] - prefix[lag] - 2*corr[:high+1])
        cmnd = np.ones(high+1)
        cmnd[1:] = difference[1:] * np.arange(1, high+1) / np.maximum(np.cumsum(difference[1:]), 1e-18)
        candidates = np.flatnonzero(cmnd[low:high+1] < .2) + low
        if len(candidates):
            tau = int(candidates[0])
            while tau < high and cmnd[tau+1] < cmnd[tau]:
                tau += 1
        else:
            tau = int(np.argmin(cmnd[low:high+1])) + low
        confidence = float(max(0., 1.-cmnd[tau]))
        refined = float(tau)
        if 1 <= tau < high:
            denominator = cmnd[tau-1]-2*cmnd[tau]+cmnd[tau+1]
            if abs(denominator) > 1e-12:
                refined += float(np.clip(.5*(cmnd[tau-1]-cmnd[tau+1])/denominator, -.5, .5))
        spectrum = np.abs(np.fft.rfft(frame * window))**2
        centroids.append(float(np.dot(spectrum, frequency)/max(1e-18, spectrum.sum())))
        if confidence < .8:
            continue
        f_frame = 11025 / refined
        peaks = np.flatnonzero((spectrum[1:-1] > spectrum[:-2]) & (spectrum[1:-1] > spectrum[2:])) + 1
        peak_power = spectrum.max()
        peaks = peaks[spectrum[peaks] > peak_power * .04]
        significant = peaks[spectrum[peaks] >= peak_power * .15]
        # A 4:5:6 chord repeats at a frequency below any of its actual notes.
        # Replace that virtual YIN fundamental with the lowest observed peak.
        if len(significant) and frequency[significant[0]] >= 2.5 * f_frame:
            f_frame = float(frequency[significant[0]])
            residuals.append(1.)
        elif len(peaks):
            ratios = frequency[peaks] / max(f_frame, 1e-6)
            nonharmonic = np.abs(ratios - np.round(ratios)) > .15
            residuals.append(float(spectrum[peaks[nonharmonic]].sum() / spectrum[peaks].sum()))
        else:
            residuals.append(0.)
        pitches.append(f_frame)
        confidences.append(confidence)
    f0 = float(np.median(pitches)) if pitches else None
    confidence = float(np.median(confidences)) if confidences else 0.
    pitch_iqr = 0.
    if pitches:
        q25, q75 = np.percentile(12 * np.log2(pitches), [25, 75])
        pitch_iqr = float(q75 - q25)
    multiple = bool(residuals and np.median(residuals) > .22)
    centroid = float(np.median(centroids)) if centroids else 0.
    return f0, confidence, multiple, centroid, pitch_iqr


def analyze_sample(waveform, sample_rate=SAMPLE_RATE, *, filename=''):
    """Measure one dry keysound; only a background trigger can request transcription."""
    signal = np.asarray(waveform, dtype=np.float32).reshape(-1)
    block = max(1, round(sample_rate * .005))
    peak = float(np.max(np.abs(signal))) if len(signal) else 0.
    if peak <= 1e-10:
        return dict(silent=True, onset_offset=0., end_offset=0., slot=3, pitch=None,
                    effective_duration=0., peak=peak, attack_peak=0., pitch_iqr_semitones=0., composite=False,
                    coarse_id=4, fine_id=13, coarse_tag='fx_cymbal', fine_tag='fx_cymbal.impact_hit_se',
                    tag_supervision='coarse_only')
    absolute = np.abs(signal)
    first = int(np.flatnonzero(absolute >= peak * 10**(-40/20))[0])
    last = int(np.flatnonzero(absolute >= peak * 10**(-35/20))[-1]) + 1
    active = signal[first:last]
    padded = np.pad(active, (0, (-len(active)) % block))
    envelope = np.sqrt(np.mean(padded.reshape(-1, block)**2, axis=1))
    attack_peak = float(np.max(envelope))
    attack = float(np.flatnonzero(envelope >= .9 * attack_peak)[0]) * block / sample_rate
    quarter = max(1, len(envelope)//4)
    sustain = float(np.mean(envelope[-quarter:]) / max(1e-12, attack_peak))
    f0, confidence, multiple, centroid, pitch_iqr = _pitch_features(active, sample_rate)
    named = _name_slot(filename)
    duration = (last-first)/sample_rate
    slot_method = 'waveform'
    if named in UNPITCHED_SLOTS:
        slot = named
        slot_method = 'filename'
    elif named is not None:
        slot = named
        slot_method = 'filename'
    elif f0 is not None:
        slot = 5 if f0 < 130 and not multiple else 7 if multiple or attack > .08 or sustain > .65 else 6
    else:
        slot = 3 if duration > 1. or attack > .08 else 0 if centroid < 220 else 2 if centroid > 3500 else 1
    # Named polyphonic harmony still needs a representative pitch; use its
    # dominant spectral peak rather than inventing a default MIDI note.
    pitch_method = 'yin'
    named_pitch = _filename_pitch(filename) if slot in PITCHED_SLOTS else None
    if slot in PITCHED_SLOTS and named_pitch is None and f0 is None:
        n = min(len(active), sample_rate)
        power = np.abs(np.fft.rfft(active[:n] * np.hanning(n)))**2
        frequency = np.fft.rfftfreq(n, 1/sample_rate)
        power[(frequency < 40) | (frequency > 4000)] = 0
        f0 = float(frequency[np.argmax(power)]) if power.max() > 0 else None
        pitch_method = 'dominant-peak-low-confidence'
    pitch = min(127, max(0, round(69+12*math.log2(f0/440)))) if f0 else None
    if named_pitch is not None:
        pitch = keysound_root_pitch(named_pitch,f0)
        pitch_method = 'filename' if pitch==named_pitch else 'filename/yin-octave'
    from .loudness import k_loudness
    from .timbre_tags import classify_dry_waveform
    window_seconds = min(duration, .2) if slot in PITCHED_SLOTS else .1
    level = k_loudness(active[:max(1, round(window_seconds * sample_rate))], sample_rate)
    base_result = dict(
        k_loudness_db=level, composite=False, silent=False, onset_offset=first/sample_rate, end_offset=last/sample_rate,
        effective_duration=duration, attack_seconds=attack, attack_peak=attack_peak, sustain_ratio=sustain,
        periodicity_confidence=confidence, fundamental_hz=f0, pitch=pitch, slot=slot,
        pitch_method=pitch_method, slot_method=slot_method, pitch_iqr_semitones=pitch_iqr,
        metallic=slot == 3 and bool(re.search(r'crash|china|splash|cymbal|cym', Path(filename).stem.casefold())),
        polyphonic=multiple, centroid_hz=centroid, peak=peak,
        rms=float(np.sqrt(np.mean(active.astype(np.float64)**2))),
    )
    tag_info = classify_dry_waveform(active, sample_rate, filename=filename, base_analysis=base_result)
    base_result.update(
        coarse_id=tag_info['coarse_id'],
        fine_id=tag_info['fine_id'],
        coarse_tag=tag_info['coarse_tag'],
        fine_tag=tag_info['fine_tag'],
        tag_supervision=tag_info['tag_supervision'],
    )
    return base_result


def _label_sources(data):
    return data.get('label_sources') or (data['label_source'],)


def _merged_label_data(first, second):
    from .timbre_tags import classify_event_tags
    sources = sorted(set(_label_sources(first)) | set(_label_sources(second)))
    data = {**second, **first, 'pseudo': first.get('pseudo', False) or second.get('pseudo', False)}
    triples = list(dict.fromkeys(classify_event_tags(first) + classify_event_tags(second)))
    if len(triples) > 1:
        data['merged_tag_triples'] = [list(item) for item in triples]
    if len(sources) > 1:
        data.update(label_source='mixed', label_sources=sources)
    return data


def _merge_events(events, metadata):
    """Union unison voices and coalesce drum hits on the song's score grid."""
    from .grid import BeatGrid
    grid = BeatGrid(metadata)
    notes, transients = defaultdict(list), {}
    for event in events:
        if event.kind == 'note':
            notes[(event.data['slot'], event.data['pitch'])].append(event)
        else:
            key = (grid.tick(event.time), event.data['slot'])
            previous = transients.get(key)
            if previous is None:
                transients[key] = event
            else:
                onset = min(previous.time, event.time)
                end = max(previous.time+previous.data.get('duration', 0.),
                          event.time+event.data.get('duration', 0.))
                data = _merged_label_data(previous.data, event.data)
                data.update(duration=end-onset,
                            metallic=previous.data.get('metallic', False) or event.data.get('metallic', False))
                transients[key] = Event(onset, 'unpitched', data)
    result = list(transients.values())
    for group in notes.values():
        current = None
        for event in sorted(group, key=lambda e: e.time):
            if current is not None and event.time < current.time + current.data['duration']:
                end = max(event.time+event.data['duration'], current.time+current.data['duration'])
                data = _merged_label_data(current.data, event.data)
                data.update(duration=end-current.time, velocity_bin=max(
                    current.data.get('velocity_bin', 0), event.data.get('velocity_bin', 0)))
                current = Event(current.time, 'note', data)
            else:
                if current is not None:
                    result.append(current)
                current = event
        if current is not None:
            result.append(current)
    return sorted(result, key=lambda e: (e.time, e.data['slot'], e.data.get('pitch', -1)))


def _bmson_label_regions(render):
    """Analyze seamless BGM as one signal. Playable slices keep their own attacks."""
    runs, active = [], {}
    for i, (trigger, _, _, _) in enumerate(render):
        if trigger.sound not in active:
            active[trigger.sound] = []
            runs.append(active[trigger.sound])
        active[trigger.sound].append(i)
        if not trigger.seamless_end:
            del active[trigger.sound]

    regions = {}
    def combine(indices):
        if len(indices) > 1:
            # Continuation markers can occur after the source has gone silent.
            stop = max((render[i][2] + render[i][3] for i in indices if render[i][3]),
                       default=render[indices[0]][2])
            regions.update(dict.fromkeys(indices, (indices[0], stop)))

    for run in runs:
        if len(run) == 1:
            continue
        for bgm, indices in groupby(run, key=lambda i: render[i][0].bgm):
            indices = list(indices)
            if bgm:
                combine(indices)
    return regions


def _index_teacher_events(events):
    events = sorted(events, key=lambda event: event.time)
    starts, ends = [], []
    latest_end = -math.inf
    for event in events:
        starts.append(event.time)
        latest_end = max(latest_end, event.time + event.data.get('duration', 0.))
        ends.append(latest_end)
    return events, starts, ends


def _teacher_slice(index, start, stop, trigger):
    events, starts, ends = index
    # Prefix maximum ends include notes already active at the source offset;
    # the onset index excludes all notes from later slices.
    for i in range(bisect_left(ends, start), bisect_left(starts, stop)):
        event = events[i]
        local_start = max(event.time, start)
        source_end = event.time + event.data.get('duration', 0.)
        local_end = min(source_end, stop)
        data = dict(event.data, velocity_bin=0, pseudo=True, label_source='yourmt3_plus', bms_sample=trigger.sound)
        data.pop('label_sources', None)
        if event.kind == 'note' and local_end > local_start:
            data['duration'] = local_end - local_start
        elif event.kind == 'unpitched' and start <= event.time:
            if 'duration' in data:
                data['duration'] = max(0., local_end - event.time)
        else:
            continue
        yield i, source_end > stop, Event(trigger.time + (local_start - start), event.kind, data)


def _local_bms_event(trigger, estimate, duration, source, pseudo, volume):
    onset = trigger.time + estimate['onset_offset']
    slot = estimate['slot']
    data = dict(slot=slot, duration=duration, label_source=source, pseudo=pseudo,
                bms_sample=trigger.sound, acoustic_source=estimate['slot_method'])
    if estimate.get('coarse_id'):
        data.update(
            coarse_id=estimate['coarse_id'],
            fine_id=estimate.get('fine_id', 0),
            coarse_tag=estimate.get('coarse_tag'),
            fine_tag=estimate.get('fine_tag'),
            tag_supervision=estimate.get('tag_supervision', 'coarse_only'),
        )
    data['slot_supervision'] = ('hard' if estimate['slot_method'] == 'filename' else
                                'env' if slot in PITCHED_SLOTS else 'unpitched')
    level = estimate['k_loudness_db'] + 20 * math.log10(max(volume, 1e-10))
    if slot in PITCHED_SLOTS and estimate['pitch'] is not None:
        data['pitch'] = estimate['pitch']
        return Event(onset, 'note', data), level
    if slot in PITCHED_SLOTS:
        data.update(coarse_id=4, fine_id=13, coarse_tag='fx_cymbal',
                    fine_tag='fx_cymbal.impact_hit_se', tag_supervision='coarse_only')
    data.update(slot=3 if slot in PITCHED_SLOTS else slot, metallic=estimate.get('metallic', False))
    return Event(onset, 'unpitched', data), level


def render_bms(chart, output_audio):
    """Render one selected chart with the standard full-mix autoplay renderer."""
    _, metadata = convert_bms(chart, output_audio, audio_only=True)
    return metadata


def convert_bms(source, output_audio, *, random_seed=0, random_value=1,
                yourmt3=None, sample_rate=SAMPLE_RATE, encoding=None, allow_partial_audio=False, audio_only=False,
                balance_options=None):
    """Return ``(events, metadata)`` and write one 24-bit mono FLAC mix.

    ``source`` is a script or one song directory. ``yourmt3(audio, output_mid)``
    is optional and is called once per distinct composite source file. In audio-only mode the mix is rendered without any acoustic analysis or labels.
    """
    import soundfile as sf
    from .midi import quantize_velocities, read_midi
    chart = source if isinstance(source, Chart) else select_script(
        source, random_seed=random_seed, random_value=random_value, encoding=encoding)
    output = Path(output_audio)
    if isinstance(source, Chart):
        source = chart.path
    song_parent = Path(source).resolve() if Path(source).is_dir() else bms_song_directory(source)
    directory_index = cache(_sample_directory_index)
    script_dir = chart.path.resolve().parent
    paths, waveforms, unavailable = {}, {}, {}
    for key in dict.fromkeys(t.sound for t in chart.triggers):
        filename = chart.samples.get(key)
        try:
            if filename is None:
                raise KeyError(key)
            path = _resolve_sample(script_dir, song_parent, chart.metadata.get('sample_root', ''),
                                   filename, directory_index)
            if allow_partial_audio and not path.is_relative_to(song_parent):
                raise FileNotFoundError(filename)
            if path not in waveforms:
                waveforms[path] = _read_sample(path, sample_rate)
            paths[key] = path
        except (KeyError, FileNotFoundError, sf.LibsndfileError, subprocess.CalledProcessError) as exc:
            if not allow_partial_audio:
                raise
            unavailable[key] = dict(filename=filename, reason=type(exc).__name__)
    bmson = chart.metadata['format'] == 'bmson'
    @cache
    def analyze_interval(path, offset, length):
        return analyze_sample(waveforms[path][offset:offset+length], sample_rate, filename=path.name)
    # A simultaneous trigger of one WAV/slice is a single voice, even if the
    # same ID appears in several BGM columns and keys at that instant.
    unique = {}
    for trigger in chart.triggers:
        if trigger.sound in unavailable:
            continue
        key = (round(trigger.time*sample_rate), trigger.voice)
        if key in unique:
            old = unique[key]
            ends = [t.end for t in (old, trigger) if t.end is not None]
            unique[key] = replace(old, bgm=old.bgm or trigger.bgm,
                                  end=max(ends) if len(ends) == 2 else None)
        else:
            unique[key] = trigger
    triggers = sorted(unique.values(), key=lambda t: (t.time, t.sound))
    next_time, render = {}, []
    for trigger in reversed(triggers):
        path = paths[trigger.sound]
        waveform = waveforms[path]
        offset = round(trigger.source_offset*sample_rate)
        length = max(0, len(waveform)-offset)
        for bound in (trigger.source_duration,
                      None if trigger.end is None else trigger.end-trigger.time,
                      None if trigger.voice not in next_time else next_time[trigger.voice]-trigger.time):
            if bound is not None:
                length = min(length, max(0, round(bound*sample_rate)))
        next_time[trigger.voice] = trigger.time
        render.append((trigger, path, offset, length))
    render.reverse()
    # A final BMSON release may outlast its keysound. Retain the silent frames
    # so the training tokenizer can see the release on the same audio timeline.
    total = max([max(round(t.time*sample_rate)+n, round((t.time+(t.hold_duration or 0.))*sample_rate))
                 for t, _, _, n in render] + [1])
    mix = np.zeros(total, dtype=np.float32)
    events, amplitudes, fallback_files, details = [], [], [], []
    warnings = list(chart.metadata.get('warnings', []))

    @cache
    def teacher_events(path):
        folder = output.parent / (output.stem + '-bms-teachers')
        folder.mkdir(parents=True, exist_ok=True)
        destination = folder / f'{len(fallback_files):04d}.mid'
        result = yourmt3(path, destination)
        if isinstance(result, (list, tuple)) and (not result or isinstance(result[0], Event)):
            extracted, midi = result, None
        else:
            midi = str(result or destination)
            extracted = read_midi(midi, reliable_velocity=False, teacher=True)
        fallback_files.append(dict(audio=str(path), midi=midi, event_count=len(extracted)))
        return _index_teacher_events(extracted)

    @cache
    def rendered_analysis(path, offset, length):
        if bmson:
            return analyze_interval(path, offset, length)
        dry = analyze_interval(path, 0, len(waveforms[path]))
        if length == len(waveforms[path]):
            return dry
        end = min(dry['end_offset'], length / sample_rate)
        onset = dry['onset_offset']
        return dict(dry, end_offset=end, dry_effective_duration=dry['effective_duration'],
                    effective_duration=max(0., end-onset),
                    silent=dry['silent'] or end <= onset)

    def add_local(trigger, estimate, duration, source, pseudo=False):
        event, amplitude = _local_bms_event(trigger, estimate, duration, source, pseudo, chart.metadata['volume'])
        events.append(event)
        amplitudes.append(amplitude)

    balance_stems = {} if balance_options is not None else None
    pending, warned_paths = [], set()
    regions = _bmson_label_regions(render) if bmson and not audio_only else {}
    teacher_tails = {}
    for i, (trigger, path, offset, length) in enumerate(render):
        part = waveforms[path][offset:offset+length]
        at = round(trigger.time*sample_rate)
        audible = part * chart.metadata['volume']
        if offset+length < len(waveforms[path]) and not trigger.seamless_end:
            fade = min(round(.0025*sample_rate), length//4)
            if fade:
                audible[-fade:] *= np.linspace(0., 1., fade, dtype=np.float32)[::-1]
        mix[at:at+len(part)] += audible
        if audio_only:
            continue
        leader, stop = regions.get(i, (i, offset + length))
        detail = dict(sound=trigger.sound, path=str(path), time=trigger.time, bgm=trigger.bgm,
                      lane=trigger.lane, source_offset=offset/sample_rate, rendered_duration=length/sample_rate,
                      hold_duration=trigger.hold_duration)
        details.append(detail)
        if i in regions:
            detail['label_region_first_trigger'] = leader
            physical = analyze_interval(path, offset, length)
            detail['analysis'] = (dict(physical, composite=True) if trigger.bgm and
                                  physical['effective_duration'] > 12. and physical['slot'] != 3 else physical)
        if balance_stems is not None:
            # Same keysound slice, cut, volume and onset as the producer's mix.
            analysis = detail.get('analysis') or rendered_analysis(path, offset, offset + length)
            if length and not analysis.get('silent') and not analysis.get('composite') and not (
                    trigger.bgm and analysis.get('dry_effective_duration', analysis['effective_duration']) > 12
                    and analysis['slot'] != 3):
                slot = analysis['slot']
                if slot not in balance_stems:
                    balance_stems[slot] = np.zeros_like(mix)
                balance_stems[slot][at:at+length] += audible
        if i != leader:
            continue  # The leading trigger's analysis covers this continuation.
        analysis = rendered_analysis(path, offset, stop - offset)
        duration = analysis['effective_duration']
        # A legacy WAV ID restarts the same source on each hit. Its class
        # and teacher policy are properties of the dry file, even on a short cut.
        dry_duration = analysis.get('dry_effective_duration', duration)
        if trigger.bgm and dry_duration > 12. and analysis['slot'] != 3:
            analysis = dict(analysis, composite=True)
        detail['label_region_analysis' if i in regions else 'analysis'] = analysis
        if analysis['silent']:
            continue
        if not analysis['composite']:
            if trigger.hold_duration is not None:
                duration = max(0., trigger.hold_duration - analysis['onset_offset'])
            add_local(trigger, analysis, duration, 'bms_acoustic')
            continue
        if yourmt3 is not None:
            count = 0
            for note_id, continues, event in _teacher_slice(teacher_events(path), offset/sample_rate,
                                                            stop/sample_rate, trigger):
                count += 1
                position = len(events)
                if bmson and event.kind == 'note':
                    # A role change can split one c:true play into two label
                    # regions. Only reunite pieces of the very same source note.
                    play_start = round((trigger.time - trigger.source_offset) * sample_rate)
                    key = (trigger.sound, play_start, note_id)
                    tail = teacher_tails.pop(key, None)
                    if tail is not None and abs(tail[0] - offset) <= 1:
                        position = tail[1]
                        previous = events[position]
                        duration = event.time + event.data['duration'] - previous.time
                        events[position] = Event(previous.time, 'note', dict(previous.data, duration=duration))
                    if continues:
                        teacher_tails[key] = (stop, position)
                if position == len(events):
                    events.append(event)
                    amplitudes.append(None)
            detail['yourmt3_event_count'] = count
            if count:
                continue
        reason = 'teacher_unavailable' if yourmt3 is None else 'empty_teacher_interval'
        pending.append(dict(path=str(path), sound=trigger.sound, time=trigger.time, lane=trigger.lane,
                            source_offset=offset/sample_rate, duration=(stop-offset)/sample_rate, reason=reason))
        if path not in warned_paths:
            warnings.append(f'Unresolved composite BMS audio ({reason}): {path}')
            warned_paths.add(path)
    peak = float(np.max(np.abs(mix)))
    gain = min(1., .999 / peak) if peak else 1.
    # FLAC stores bounded PCM. A documented fixed gain prevents clipping and
    # preserves all relative amplitudes; it is not per-frame normalization.
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, mix*gain, sample_rate, format='FLAC', subtype='PCM_24')
    metadata = {**chart.metadata, 'selected_script': str(chart.path.resolve()), 'audio': str(output.resolve()),
                'sample_rate': sample_rate, 'duration_seconds': total/sample_rate,
                'trigger_count': len(triggers), 'raw_trigger_count': len(chart.triggers),
                'sample_count': len(waveforms), 'mix_peak_before_gain': peak, 'mix_gain': gain,
                'audio_encoding': 'FLAC/PCM_24/mono', 'polyphony_cut': 'per-WAV-ID-or-BMSON-slice',
                'unavailable_triggers': [dict(sound=t.sound, time=t.time, lane=t.lane, **unavailable[t.sound])
                                         for t in chart.triggers if t.sound in unavailable],
                'audio_complete': not unavailable and not any('MIDIFILE' in w for w in warnings),
                'warnings': warnings}
    if balance_stems is not None:
        from .studio_mix import measured_balance
        metadata['balance_db'] = measured_balance(balance_stems, mix, sample_rate, balance_options)
    if audio_only:
        return [], metadata
    from .loudness import loudness_targets
    targets = loudness_targets(amplitudes)
    events = [Event(e.time, e.kind, {**e.data, 'velocity_bin': band, 'velocity_db': db})
              for e, (band, db) in zip(events, targets)]
    events = _merge_events(events, metadata)
    from .timbre_tags import merge_intervals
    residual_mixed = merge_intervals(
        [item['time'], item['time'] + item['duration']] for item in pending if item.get('duration', 0.) > 0.
    )
    metadata.update({
        'samples': details, 'teacher_calls': len(fallback_files), 'teacher_sources': fallback_files,
        'yourmt3_event_count': sum('yourmt3_plus' in _label_sources(e.data) for e in events),
        'pending_teacher_regions': pending,
        'residual_mixed_intervals': residual_mixed,
        'acoustic_labels_complete': not pending and not unavailable
            and not any('MIDIFILE' in warning for warning in warnings),
        'label_estimator': 'isolated-onset-yin-filename-long-bgm.v5',
    })
    return events, metadata
