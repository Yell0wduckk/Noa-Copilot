"""Read aligned singing notes from GTSinger and M4Singer."""
from dataclasses import asdict
import json
import os
from pathlib import Path

from .codec import Event


_INITIALS = set('b p m f d t n l g k h j q x zh ch sh r z c s y w'.split())
_RESTS = {'AP', 'SP', 'SIL', 'PAU'}
# Held-out singer/song pairs in the authors' code/usr/configs/m4singer/base.yaml.
_M4_TEST = {'Alto-2#岁月神偷', 'Alto-2#奇妙能力歌', 'Tenor-1#一千年以后', 'Tenor-1#童话',
            'Tenor-2#消愁', 'Tenor-2#一荤一素', 'Soprano-1#念奴娇赤壁怀古', 'Soprano-1#问春'}


def _note(start, end, pitch):
    return Event(start, 'note', dict(slot=4, pitch=int(pitch), duration=end-start, velocity_bin=0))


def _m4_events(item):
    # The score repeats a syllable's note for its initial and final phonemes.
    # Their aligned durations partition one note; a slur starts a new note.
    notes, phones = item['notes'], item.get('phs', [])
    slurs = item.get('is_slur', [])
    aligned = 'ph_dur' in item
    durations = item['ph_dur'] if aligned else item['notes_dur']
    result = []
    start = 0.
    previous_initial = False
    for i, (raw_pitch, duration) in enumerate(zip(notes, durations)):
        end = start + float(duration)
        pitch = int(raw_pitch)
        phone = str(phones[i]).lower() if phones else ''
        voiced = pitch > 0 and phone.strip('<>').upper() not in _RESTS and end > start
        if voiced:
            same_syllable = (previous_initial and phone not in _INITIALS and int(notes[i-1]) == pitch
                             and not (slurs and int(slurs[i])))
            if same_syllable:
                # Without phone boundaries, notes_dur repeats the full score
                # duration on both phones. The initial already advanced time.
                if not aligned:
                    end = start
                result[-1] = _note(result[-1].time, end, pitch)
            else:
                result.append(_note(start, end, pitch))
        start, previous_initial = end, voiced and phone in _INITIALS
    return result


def _events(document):
    if isinstance(document, dict):
        if 'ph_dur' in document or 'notes_dur' in document:
            return _m4_events(document)
        document = document.get('words', document.get('notes', [document]))
    events = []
    for row in document:
        if 'ph_dur' in row or 'notes_dur' in row:
            events.extend(_m4_events(row))
        elif 'note_start' in row:
            # Raw GTSinger JSON gives each word's individual note times in seconds.
            for pitch, start, end in zip(row['note'], row['note_start'], row['note_end']):
                start, end = float(start), float(end)
                if int(pitch) > 0 and end > start:
                    events.append(_note(start, end, pitch))
        elif 'pitch' in row and 'start' in row:
            start = float(row['start'])
            end = float(row['end']) if 'end' in row else start + float(row['duration'])
            if int(row['pitch']) > 0 and end > start:
                events.append(_note(start, end, row['pitch']))
    return events


def read_vocal_json(path):
    return _events(json.loads(Path(path).read_text(encoding='utf-8-sig')))


def scan_vocals(root: Path, dataset: str):
    """Yield each recording with its preferred MIDI, sidecar or aggregate labels."""
    root = root.resolve()
    gtsinger = dataset.casefold() == 'gtsinger'
    if gtsinger and 'paired_speech_group' in (s.casefold() for s in root.parts):
        return
    rows, by_stem, metadata = {}, {}, []
    for directory, folders, files in os.walk(root):
        folders[:] = sorted(d for d in folders if not (gtsinger and d.casefold() == 'paired_speech_group'))
        folder = Path(directory)
        names = {name.casefold(): folder / name for name in files}
        clips = []
        for audio in sorted(names.values()):
            if audio.suffix.casefold() not in ('.wav', '.flac'):
                continue
            stem = audio.stem.casefold()
            if audio.suffix.casefold() == '.flac' and stem + '.wav' in names:
                continue
            midi = names.get(stem + '.mid') or names.get(stem + '.midi')
            if midi:
                options = dict(midi=midi, force_slot=4, reliable_velocity=False)
            elif (sidecar := names.get(stem + '.json')) is not None:
                options = dict(vocal_json=sidecar)
            else:
                options = None
            rows[audio] = options
            clips.append(audio)
            by_stem[folder, stem] = audio
        if 'meta.json' in names:
            metadata.append((names['meta.json'], clips))

    for path, clips in metadata:
        document = json.loads(path.read_text(encoding='utf-8-sig'))
        if isinstance(document, list) and any('item_name' in item for item in document):
            for item in document:
                name = item['item_name']
                song, _, clip = name.rpartition('#')
                audio = by_stem.get((path.parent / song, clip.casefold())) or by_stem.get((path.parent, clip.casefold()))
                if audio is not None:
                    if rows[audio] is None:
                        rows[audio] = dict(event_values=[asdict(event) for event in _events(item)])
                    if item.get('split'):
                        rows[audio]['split'] = item['split']
        elif len(clips) == 1:
            if rows[clips[0]] is None:
                rows[clips[0]] = dict(vocal_json=path)
            if isinstance(document, dict) and document.get('split'):
                rows[clips[0]]['split'] = document['split']

    for audio, options in rows.items():
        if options is not None:
            if not gtsinger and audio.parent.name in _M4_TEST:
                options['default_split'] = 'test'
            yield audio, options
