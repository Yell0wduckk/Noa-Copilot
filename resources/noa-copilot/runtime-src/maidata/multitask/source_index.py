"""Index extracted Stage 0 corpora for maidata.multitask.prepare."""
import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile


DATASETS = ('bms', 'e-gmd', 'slakh', 'rwc-pop', 'maestro', 'guitarset')
MIDI = ('.mid', '.midi')
AUDIO = ('.wav', '.flac')
SPLITS = {
    'train': 'train', 'train_data': 'train', 'train_labels': 'train',
    'validation': 'validation', 'valid': 'validation', 'val': 'validation',
    'test': 'test', 'test_data': 'test', 'test_labels': 'test',
    'omitted': 'omitted',
}


def _split(path, root, default):
    for part in reversed((root.name, *path.relative_to(root).parts[:-1])):
        if part.lower() in SPLITS:
            return SPLITS[part.lower()]
    return default


def _sibling(path, extensions):
    return next((p for ext in extensions if (p := path.with_suffix(ext)).is_file()), None)


def _official_table(path):
    if path.suffix == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as stream:
            yield from csv.DictReader(stream)
        return
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if isinstance(data, list):
        yield from data
    else:
        # MAESTRO v3 JSON is a pandas column-oriented table keyed by row ID.
        audio_column = data['audio_filename']
        for i in (audio_column if isinstance(audio_column, dict) else range(len(audio_column))):
            yield {field: data[field][i] for field in ('audio_filename', 'midi_filename', 'split')}


def _piano_or_drums(root, dataset, egmd_kits_per_sequence=2):
    if dataset == 'maestro':
        name, options = 'maestro-v3.0.0.json', dict(default_program=0)
    else:
        name, options = 'e-gmd-v1.0.0.csv', dict(force_drum_channel=True)
    tables = [root / name] if (root / name).is_file() else sorted(root.rglob(name))
    if not tables and dataset == 'maestro':
        tables = sorted(root.rglob('maestro-v3.0.0.csv'))
    if tables:
        selected_kits = defaultdict(set)
        for table in tables:
            for row in _official_table(table):
                if not row.get('audio_filename') or not row.get('midi_filename'):
                    continue
                audio = table.parent / row['audio_filename']
                midi = table.parent / row['midi_filename']
                if not audio.is_file():
                    audio = _sibling(audio, AUDIO)
                if audio and midi.is_file():
                    if dataset == 'e-gmd' and egmd_kits_per_sequence > 0:
                        # Official E-GMD repeats each performance on 43 kits.
                        # Keep the sequence's official fold and choose the
                        # requested number of available timbres in CSV order.
                        sequence = (row.get('drummer'), row.get('session'),
                                    row.get('id') or row['midi_filename'], row['split'])
                        kit = row.get('kit_name') or row['audio_filename']
                        chosen = selected_kits[sequence]
                        if kit in chosen or len(chosen) >= egmd_kits_per_sequence:
                            continue
                        chosen.add(kit)
                    yield audio, dict(options, midi=midi, split=row['split'])
        return
    for midi in sorted(p for p in root.rglob('*') if p.suffix.lower() in MIDI):
        audio = _sibling(midi, AUDIO)
        if audio:
            yield audio, dict(options, midi=midi)


def _slakh(root):
    for audio in sorted(root.rglob('mix.flac')):
        if 'omitted' in (part.lower() for part in (root.name, *audio.relative_to(root).parts[:-1])):
            continue
        song = audio.parent
        combined = song / 'all_src.mid'
        metadata = song / 'metadata.yaml'
        midi_dir = song / 'MIDI'
        if midi_dir.is_dir() and metadata.is_file():
            import yaml
            instruments = yaml.safe_load(metadata.read_text(encoding='utf-8-sig'))['stems']
            # all_src is the original Lakh score. Prefer the MIDI that actually
            # rendered each audible stem, including Slakh's instrument edits.
            sources = [dict(midi=midi_dir / (stem + '.mid'), program_num=info['program_num'],
                            is_drum=info['is_drum'], audio=song/'stems'/(stem+'.flac'))
                       for stem, info in sorted(instruments.items()) if info.get('audio_rendered', True)]
            if sources and all(s['midi'].is_file() for s in sources):
                yield audio, dict(midi_stems=sources)
                continue
        if combined.is_file():
            yield audio, dict(midi=combined, reliable_velocity=False)


def _rwc_pop(root):
    labels = defaultdict(list)
    vocals = defaultdict(list)
    audio_files = []
    for p in sorted(root.rglob('*')):
        if p.name.startswith('._') or '__MACOSX' in p.parts:
            continue
        if p.suffix.lower() in MIDI:
            labels[p.stem.casefold()].append(p)
        elif p.suffix.lower() in AUDIO:
            if not re.match(r'rwc_[cgjr]\d', p.stem, re.I):
                audio_files.append(p)
        elif p.name == 'metadata_mapping_vocals.csv':
            with p.open(encoding='utf-8-sig', newline='') as stream:
                for row in csv.DictReader(stream, delimiter=';'):
                    if row['MIDITrackName']:
                        vocals[row['RWCID'].casefold()].append(row['MIDITrackName'])
    for audio in audio_files:
        key = audio.stem.casefold()
        candidates = labels.get(key, [])
        if candidates:
            # Prefer the curated aligned release if other MIDI copies coexist.
            midi = min(candidates, key=lambda p: (
                'midi_aligned' not in p.as_posix().lower().split('/'), p.parent != audio.parent, p))
            yield audio, dict(midi=midi, vocal_tracks=vocals[key], reliable_velocity=False)


def _musicnet(root):
    labels = {}
    audio_files = []
    for p in sorted(root.rglob('*')):
        if p.suffix.lower() in AUDIO:
            audio_files.append(p)
        elif p.suffix.lower() == '.csv':
            labels[_split(p, root, None), p.stem] = p
    for audio in audio_files:
        label = labels.get((_split(audio, root, None), audio.stem))
        if label:
            yield audio, dict(musicnet_csv=label)
        else:
            # Only explicitly paired MIDI. The separate reference scores in
            # the official release predate alignment to the recording.
            midi = _sibling(audio, MIDI)
            if midi:
                yield audio, dict(midi=midi)


def read_musicnet_notes(path):
    from .codec import Event
    from .slots import program_slot
    from .midi import merge_slot_events
    notes = []
    with path.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            # Aligned labels are sample offsets on MusicNet's 44.1 kHz clock;
            # its GM instrument numbers are one-based.
            start, end = int(row['start_time']), int(row['end_time'])
            slot = program_slot(int(row['instrument']) - 1)
            notes.append(Event(start / 44100, 'note', dict(
                duration=(end - start) / 44100, pitch=int(row['note']), slot=slot, velocity_bin=0)))
    return merge_slot_events(notes)


def solo_beat_grid(beat_times, downbeats=(), meters=None):
    beats = sorted(set(float(t) for t in beat_times))
    if len(beats) < 2:
        return {}
    offset = min(downbeats,default=beats[0])
    segments, changes = [], []
    quarter, zero = 0., 0.
    for i,t in enumerate(beats):
        meter = meters.get(t) if meters else None
        if meter and (not changes or meter != changes[-1][1]):
            changes.append((quarter,meter))
        if t == offset:
            zero = quarter
        span = 4 / meter[1] if meter else 1.
        if i+1 < len(beats):
            segments.append([t,60*span/(beats[i+1]-t)])
        quarter += span
    grid = dict(bpm_segments=segments,offset_seconds=offset)
    if changes:
        grid['meter_segments'] = [[q-zero,*m] for q,m in changes]
    return grid


def guitarset_grid(path):
    annotations = json.loads(path.read_text(encoding='utf-8-sig')).get('annotations',[])
    for ann in sorted(annotations,key=lambda a:a.get('namespace')!='beat_position'):
        if ann.get('namespace') not in ('beat_position','beat'):
            continue
        rows = ann.get('data',[])
        down = [r['time'] for r in rows if isinstance(r.get('value'),dict) and r['value'].get('position')==1]
        meters = {r['time']:(v['num_beats'],v['beat_units']) for r in rows if
                  isinstance(v:=r.get('value'),dict) and v.get('num_beats') and v.get('beat_units')}
        grid = solo_beat_grid([r['time'] for r in rows],down,meters)
        if grid:
            return dict(grid,solo_beat_source='GuitarSet JAMS')
    return {}


def _guitarset(root):
    audio={}
    for path in sorted(root.rglob('*.wav')):
        key=re.sub(r'_(mic|mix|hex)$','',path.stem)
        if key not in audio or path.stem.endswith('_mic'):
            audio[key]=path
    for label in sorted(root.rglob('*.jams')):
        if label.stem in audio:
            # Hold out entire performers; this map is independent of excerpts.
            performer=label.stem.split('_')[0]
            fold={'04':'validation','05':'test'}.get(performer,'train')
            yield audio[label.stem],dict(guitarset_jams=label,split=fold,reliable_velocity=False,metadata=guitarset_grid(label))

def read_guitarset_notes(path):
    from .codec import Event
    document=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    notes=[]
    for annotation in document.get('annotations',[]):
        if annotation.get('namespace')!='note_midi':
            continue
        for row in annotation['data']:
            notes.append(Event(float(row['time']),'note',dict(slot=6,pitch=round(float(row['value'])),
                duration=float(row['duration']),velocity_bin=0)))
    return notes

def _sources(root, dataset, egmd_kits_per_sequence=2):
    if dataset in ('maestro', 'e-gmd'):
        return _piano_or_drums(root, dataset, egmd_kits_per_sequence)
    if dataset in ('gtsinger', 'm4singer'):
        from .vocal_sources import scan_vocals
        return scan_vocals(root, dataset)
    if dataset == 'guitarset':
        return _guitarset(root)
    if dataset == 'bms':
        from .bms import discover_bms_projects
        return ((p, dict(bms_dir=p, split='train')) for p in discover_bms_projects(root))
    return {'slakh': _slakh, 'rwc-pop': _rwc_pop, 'musicnet': _musicnet}[dataset](root)


def build_source_index(roots, output, *, egmd_kits_per_sequence=2, solo_grids=None, solo_beat_command=None):
    """Write one source-index.json from the requested dataset roots."""
    output = Path(output).resolve()

    def index_path(path):
        # Windows datasets and index may reside on different drives.
        if path.drive.casefold() != output.drive.casefold():
            return path.as_posix()
        return Path(os.path.relpath(path, output.parent)).as_posix()

    records = {}
    for dataset, directory in roots.items():
        if dataset not in DATASETS:
            continue
        root = Path(directory).resolve()
        for path, options in _sources(root, dataset, egmd_kits_per_sequence):
            relative = path.relative_to(root)
            identifier = dataset + '/' + (relative.as_posix() if relative.parts else root.name)
            if dataset in ('maestro','guitarset'):
                grid = (solo_grids or {}).get(identifier, options.get('metadata',{}))
                if grid.get('beats'):
                    grid = solo_beat_grid(grid['beats'],grid.get('downbeats',()))
                if not grid and solo_beat_command:
                    from .quantizer import beat_this_segments
                    segments = beat_this_segments(solo_beat_command,path)
                    if segments:
                        grid = dict(bpm_segments=[[g.start_seconds,g.bpm] for g in segments],
                                    offset_seconds=segments[0].offset_seconds,solo_beat_source='beat tracker')
                if not grid.get('bpm_segments'):
                    continue
                options['metadata'] = {**grid,'solo_beat_source':grid.get('solo_beat_source','aligned annotation')}
            fold = 'unspecified' if dataset in ('maestro', 'e-gmd', 'slakh', 'musicnet') else 'train'
            fold = options.pop('default_split', fold)
            row = dict(id=identifier, song_id=identifier, dataset=dataset, task='midi',
                       source_family='bms' if dataset == 'bms' else 'real_midi',
                       split=_split(path, root, fold))
            if dataset != 'bms':
                row['audio'] = index_path(path)
            for key, value in options.items():
                if key == 'midi_stems':
                    value = [dict(stem, **{name:index_path(stem[name]) for name in ('midi','audio') if stem.get(name)}) for stem in value]
                elif isinstance(value, Path):
                    value = index_path(value)
                row[key] = value
            records[identifier] = row
    output.parent.mkdir(parents=True, exist_ok=True)
    stream = NamedTemporaryFile('w', dir=output.parent, encoding='utf-8', delete=False)
    temporary = Path(stream.name)
    try:
        with stream:
            json.dump(dict(records=[records[k] for k in sorted(records)]), stream, ensure_ascii=False, indent=2)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return Counter(row['dataset'] for row in records.values())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in DATASETS:
        parser.add_argument('--' + name, dest=name, type=Path, metavar='DIR')
    parser.add_argument('--egmd-kits-per-sequence',type=int,default=2,
                        help='Available timbres per official E-GMD performance (default 2; 0 keeps all)')
    parser.add_argument('--solo-beat-grids',type=Path,help='Aligned beat times or BPM segments by recording ID')
    parser.add_argument('--solo-beat-command',nargs='+',help='Optional beat tracker for solo recordings without annotations')
    parser.add_argument('--output', type=Path, required=True, help='Destination source-index.json')
    args = vars(parser.parse_args(argv))
    roots = {name: args[name] for name in DATASETS if args[name] is not None}
    grids = json.loads(args['solo_beat_grids'].read_text(encoding='utf-8-sig')) if args['solo_beat_grids'] else None
    counts = build_source_index(roots, args['output'],egmd_kits_per_sequence=args['egmd_kits_per_sequence'],
                                solo_grids=grids,solo_beat_command=args['solo_beat_command'])
    print(args['output'], {name: counts[name] for name in roots})


if __name__ == '__main__':
    main()
