"""Produce only requested tick96 event assets from an explicit source index.

The index contains records with id/task/song_id/dataset/split and audio plus
events (JSON), MIDI or maimai IR. Official dataset splits are retained.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import torch
from . import AUDIO_SCHEMA,CACHE_SCHEMA

_BMS_BACKENDS={}


def _bms_backend(options):
    if options is None:return None
    key=json.dumps(options,sort_keys=True)
    if key not in _BMS_BACKENDS:
        import atexit
        from .teacher_backends import make_yourmt3
        backend=make_yourmt3(**options)
        _BMS_BACKENDS[key]=backend
        if hasattr(backend,'close'):atexit.register(backend.close)
    return _BMS_BACKENDS[key]


def _produce_group(job):
    rows,root,output,seed,bms_teacher_options=job
    root,output=Path(root),Path(output)
    from .codec import Event,count_chart_tokens
    torch.set_num_threads(1)
    bms_events,bms_metadata=None,{}
    source=rows[0]
    if source.get('bms'):
        from .bms import convert_bms
        script=(root/source['bms']).resolve()
        key=hashlib.sha256(str(script).encode()).hexdigest()[:20]
        audio=output/'audio'/f'{key}.flac'
        teacher = None

        def transcribe(path, destination):
            nonlocal teacher
            if teacher is None:
                teacher = _bms_backend(bms_teacher_options)
            return teacher(path, destination)

        bms_events,bms_metadata=convert_bms(script,audio,random_seed=seed,
            yourmt3=transcribe if bms_teacher_options is not None else None)
        if getattr(teacher, 'provenance', None):
            bms_metadata['yourmt3_provenance'] = teacher.provenance
    else:
        audio=(root/source['audio']).resolve() if source.get('audio') else None
    if audio is not None:
        import soundfile as sf
        info=sf.info(audio)
        duration=info.frames/info.samplerate
    else:
        duration=source.get('duration_seconds')
        if duration is None:
            saved=torch.load(root/source['features'],map_location='cpu',weights_only=True)
            duration=len(saved['features'] if isinstance(saved,dict) else saved)/100.
    records=[]
    for row in rows:
        metadata={**{key:row[key] for key in ('song_id','difficulty','version') if key in row},
                  **row.get('metadata',{})}
        metadata['pseudo']=bool(row.get('pseudo',metadata.get('pseudo',False)))
        if row.get('bms'):
            events=bms_events
            # A new render owns its measured metadata. The optional teacher
            # identity can also disappear when only local labels are produced.
            metadata.pop('yourmt3_provenance', None)
            metadata.update(bms_metadata)
        elif row.get('midi') or row.get('midi_stems'):
            from .midi import read_midi, read_midi_stems
            options = dict(reliable_velocity=row.get('reliable_velocity', True) and not metadata['pseudo'],
                           teacher=metadata['pseudo'], return_metadata=True)
            if row.get('midi'):
                events, timing_metadata = read_midi(
                    root / row['midi'], track_gains=row.get('track_gains'),
                    vocal_tracks=row.get('vocal_tracks', ()), default_program=row.get('default_program'),
                    force_drum_channel=row.get('force_drum_channel', False),
                    force_slot=row.get('force_slot'), **options)
            else:
                stems = [dict(stem, midi=root / stem['midi'],
                              **({'audio': root / stem['audio']} if stem.get('audio') else {}))
                         for stem in row['midi_stems']]
                events, timing_metadata = read_midi_stems(stems, **options)
            metadata={**timing_metadata,**metadata}
        elif row.get('guitarset_jams'):
            from .source_index import read_guitarset_notes
            events=read_guitarset_notes(root/row['guitarset_jams'])
        elif row.get('musicnet_csv'):
            from .source_index import read_musicnet_notes
            events = read_musicnet_notes(root / row['musicnet_csv'])
        elif row.get('vocal_json'):
            from .vocal_sources import read_vocal_json
            events = read_vocal_json(root / row['vocal_json'])
        elif row.get('chart_cache'):
            import gzip
            from .v7_events import events_from_record, metadata_from_record
            path = root / row['chart_cache']
            with (gzip.open(path,'rt',encoding='utf-8') if path.suffix=='.gz' else path.open(encoding='utf-8')) as f:
                record = json.load(f)
            events = events_from_record(record)
            metadata = {**metadata_from_record(record), **metadata}
        elif row.get('chart'):
            from .maimai import events_from_ir
            ir=json.loads((root/row['chart']).read_text(encoding='utf-8-sig'))
            events=events_from_ir(ir)
            metadata={**metadata_from_ir(ir),**metadata}
        else:
            values=row.get('event_values')
            if values is None:
                values=json.loads((root/row['events']).read_text(encoding='utf-8-sig'))
            if isinstance(values,dict): values=values['events']
            events=[Event(**v) for v in values]
        token_count=count_chart_tokens(row['task'],events,duration_seconds=duration,seed=seed,metadata=metadata)
        name=hashlib.sha256(str(row['id']).encode()).hexdigest()[:20]
        event_path=output/'events'/f'{name}.json'
        event_path.parent.mkdir(parents=True,exist_ok=True)
        event_path.write_text(json.dumps([asdict(e) for e in events],ensure_ascii=False),encoding='utf-8')
        records.append({k:v for k,v in row.items() if k not in (
            'audio','chart','midi','midi_stems','musicnet_csv','guitarset_jams','vocal_json','events','event_values','features','bms','bms_dir')}|
            dict(events=event_path.relative_to(output).as_posix(),
                 token_count=token_count,duration_seconds=duration,metadata=metadata))
        if audio is not None:
            records[-1]['audio']=str(audio)
        if row.get('features'):
            records[-1]['features']=str((root/row['features']).resolve())
        if row.get('bms'):
            records[-1]['bms_source']=str((root/row['bms']).resolve())
        if row.get('macro_features'):
            records[-1]['macro_features']=str((root/row['macro_features']).resolve())
    return records


def metadata_from_ir(ir):
    """Preserve the existing score's timing; explicit index metadata overrides."""
    from maidata.time import TimingMap
    timing=TimingMap.from_ir(ir)
    result=dict(bpm_segments=[(float(second),float(bpm))
                             for second,(_,bpm) in zip(timing.seconds,timing.points)],
                offset_seconds=float(ir.get('first_seconds',0.)))
    for source in (ir.get('source',{}),ir):
        for key in ('song_id','version'):
            if isinstance(source,dict) and source.get(key) is not None:
                result[key]=source[key]
    if ir.get('default_meter'):
        from .grid import normalize_meter
        meter = ir['default_meter']
        n0, d0 = normalize_meter(meter['numerator'], meter['denominator'])
        result['numerator'] = n0
        result['denominator'] = d0
        meters = {0.: (n0, d0)}
        for c in ir.get('controls', ()):
            if c['kind']=='meter':
                beat = c['position_beats']
                meters[beat['num']/beat['den']] = normalize_meter(c['numerator'],c['denominator'])
        result['meter_segments'] = [(b,*v) for b,v in sorted(meters.items())]
    difficulty=ir.get('difficulty')
    if isinstance(difficulty,dict):
        for source,target in (('index','difficulty'),('level','level'),('level_value','level'),('designer_id','designer')):
            if isinstance(difficulty.get(source),(int,float)):
                result[target]=difficulty[source]
    elif isinstance(difficulty,int):
        result['difficulty']=difficulty
    elif isinstance(difficulty,str):
        # Match the simai difficulty slots used by the exporter; zero is null.
        value={'easy':1,'basic':2,'advanced':3,'expert':4,'master':5,'re:master':6,'remaster':6}.get(difficulty.lower())
        if value is not None:
            result['difficulty']=value
    for source,target in (('level','level'),('level_value','level'),('designer_id','designer')):
        if isinstance(ir.get(source),(int,float)):
            result[target]=ir[source]
    return result


def produce(index,output,workers=12,seed=20261104,bms_yourmt3=None):
    from collections import defaultdict
    index,output=Path(index).resolve(),Path(output).resolve()
    source=json.loads(index.read_text(encoding='utf-8-sig'))
    groups=defaultdict(list)
    from .data import excluded_song, grid_corpus
    for row in source['records']:
        if excluded_song(row) or (row['task']=='midi' and not grid_corpus(row)):
            continue
        if row.get('features'):
            # Preparation copies a reference. Preserve the original frequency
            # and silence convention even when the new manifest describes a
            # different online frontend. A payload's own tag wins at load time.
            if 'features_schema' not in row:
                row = dict(row, features_schema=row.get('audio_schema') or source.get('audio_schema'))
        bms = row.get('bms_dir') or row.get('bms')
        if bms:
            from .bms import bms_song_directory
            path = (index.parent / bms).resolve()
            directory = path if row.get('bms_dir') else bms_song_directory(path)
            groups[('bms',str(directory))].append(dict(row, bms=bms))
        else:
            groups[('audio',row.get('audio_id') or row.get('audio') or row.get('features'))].append(row)
    for (kind,identity),rows in list(groups.items()):
        if kind=='bms':
            # Difficulty aliases describe one acoustic recording, not
            # independent training examples. The converter selects its most
            # complete deterministic script from this song directory.
            groups[(kind,identity)]=[dict(rows[0],bms=identity,
                bms_source_records=[dict(id=r['id'],bms=r['bms']) for r in rows])]
    output.mkdir(parents=True,exist_ok=True)
    jobs=[(rows,str(index.parent),str(output),seed,bms_yourmt3) for rows in groups.values()]
    if workers:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            records=[r for rows in pool.map(_produce_group,jobs) for r in rows]
    else:
        records=[r for job in jobs for r in _produce_group(job)]
    from maidata.joint_training.checkpoint import source_identity
    manifest=dict(schema=CACHE_SCHEMA,audio_schema=AUDIO_SCHEMA,audio_storage='waveform',producer=source_identity(),
                  source_index=str(index),source_metadata=source.get('metadata',{}),records=records)
    target=output/'manifest.json'
    target.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return target


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--bms-yourmt3',type=Path,help='Explicit local YourMT3 options for composite BMS samples')
    args=parser.parse_args(argv)
    options=json.loads(args.bms_yourmt3.read_text(encoding='utf-8-sig')) if args.bms_yourmt3 else None
    print(produce(args.index,args.output,args.workers,bms_yourmt3=options))


if __name__=='__main__':
    main()
