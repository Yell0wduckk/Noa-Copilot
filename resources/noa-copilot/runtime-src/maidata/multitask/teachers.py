"""Explicit YourMT3+ pseudo-MIDI production; no teacher runs during training."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from .data import excluded_song

def extract(index,output,backend_options,*,limit=None,factories=None,scope='games'):
    from .midi import read_midi
    from .teacher_backends import make_yourmt3
    index,output=Path(index).resolve(),Path(output).resolve()
    source=json.loads(index.read_text(encoding='utf-8-sig'))
    rows=[]
    for value in source['records']:
        if excluded_song(value):
            continue
        row=dict(value)
        for key in ('audio','midi','events','chart','bms','bms_dir','musicnet_csv','vocal_json','guitarset_jams','macro_features'):
            if row.get(key):
                row[key]=str((index.parent/row[key]).resolve())
        if row.get('midi_stems'):
            row['midi_stems']=[dict(stem,**{key:str((index.parent/stem[key]).resolve())
                for key in ('audio','midi') if stem.get(key)}) for stem in row['midi_stems']]
        rows.append(row)
    unique={}
    for row in rows:
        if row.get('split','train')=='train' and row.get('audio') and row['task']!='midi':
            unique.setdefault(row['audio'],row)
    output.mkdir(parents=True,exist_ok=True)
    factory=(factories or {}).get('yourmt3',make_yourmt3)
    options=backend_options.get('yourmt3',backend_options)
    backend=factory(**options)
    try:
        for i,(audio,row) in enumerate(unique.items()):
            if limit is not None and i>=limit:
                break
            midi=output/f'{i:06d}.mid'
            result=backend(Path(audio),midi)
            events=read_midi(result or midi,reliable_velocity=False,teacher=True)
            path=output/f'{i:06d}.events.json'
            path.write_text(json.dumps([asdict(event) for event in events],ensure_ascii=False),encoding='utf-8')
            metadata=dict(row.get('metadata',{}),pseudo=True)
            rows.append(dict(row,id=str(row['id'])+'/pseudo-midi',task='midi',
                events=str(path),source_family='pseudo_midi',pseudo=True,metadata=metadata))
    finally:
        if hasattr(backend,'close'):
            backend.close()
    target=output/'source-index.json'
    target.write_text(json.dumps(dict(source.get('metadata',{}),records=rows),ensure_ascii=False,indent=2),encoding='utf-8')
    return target

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--backend-options',type=Path,required=True)
    parser.add_argument('--limit',type=int)
    args=parser.parse_args(argv)
    print(extract(args.index,args.output,json.loads(args.backend_options.read_text(encoding='utf-8-sig')),limit=args.limit))
if __name__=='__main__':
    main()
