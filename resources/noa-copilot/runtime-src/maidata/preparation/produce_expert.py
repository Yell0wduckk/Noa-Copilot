"""Incremental Expert assets and reference-only training unions.

Explicit offline command; training never invokes this producer automatically.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from functools import partial
from pathlib import Path
import hashlib
import json

from maidata.chart_cache.cache import read_json, write_json, digest, load_chart
from maidata.chart_cache import produce as charts
from maidata.preparation.timing import fields, chart_timing
from maidata.preparation.produce import token
from maidata.sampling.plan import chart_windows, _coverage, _assemble, write_plan
from maidata.streaming.cache import pack_path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def discover(plan, audio, source_root):
    """Read only newly requested Expert source bodies; no old chart payloads."""
    songs = {str(s['song_id']):s for s in audio['songs']}
    result = []
    for song_id in sorted({str(c['song_id']) for c in plan['charts']}):
        song = songs[song_id]
        path = Path(source_root)/song_id/'maidata.txt'
        text = path.read_text(encoding='utf-8-sig'); f = fields(text)
        body = f.get('inote_4','')
        if not body.strip(): continue
        timing = chart_timing(body,f.get('wholebpm') or 120,f.get('first_4',f.get('first',0)) or 0)
        variant = token(dict(asset=song['asset'],timing=timing,samples_per_beat=24))
        result.append(dict(chart_id=f'{song_id}:4',song_id=song_id,difficulty='4',version=f.get('version',''),
            source_path=str(path.resolve()),source_sha256=sha(path),body_sha256=hashlib.sha256(body.encode()).hexdigest(),
            source_header_line=next(i+1 for i,line in enumerate(text.splitlines()) if line.startswith('&inote_4=')),
            audio_variant=variant,audio_asset=song['asset'],timing=timing,
            audio_end_seconds=audio['assets'][song['asset']]['duration_seconds'],
            path=f'charts/{song_id}/4.json.gz'))
    return result


def extend_plan(base, windows, rows, records, chart_root, old_chart_root):
    """Reuse accepted windows and their existing complex flags/weights."""
    chart_meta = [dict(c,chart_cache_root=c.get('chart_cache_root',str(old_chart_root))) for c in base['charts']]
    groups = {g['audio_variant']:deepcopy(g) for g in base['audio_groups']}
    windows = list(windows)
    for row,record in zip(rows,records):
        local = chart_windows(record,config=base['config']);coverage = _coverage(row,base['config'])
        key = row['chart_id'];variant = row['audio_variant']
        if variant not in groups:
            groups[variant] = dict(audio_variant=variant,audio_asset=row['audio_asset'],
                audio_end_seconds=row['audio_end_seconds'],timing=row['timing'],
                **{k:v for k,v in coverage.items() if k!='target_starts_beats'},chart_ids=[],song_ids=[])
        group = groups[variant]
        group['chart_ids'].append(key)
        if row['song_id'] not in group['song_ids']: group['song_ids'].append(row['song_id'])
        chart_meta.append(dict(chart_id=key,song_id=row['song_id'],difficulty='4',audio_variant=variant,
            chart_cache_root=str(chart_root),chart_cache_path=row['path'],source_path=row['source_path'],
            version=row['version'],chart_index=len(chart_meta)+1,first_beat=coverage['first_beat'],
            base_windows=len(local),complex_windows=sum(w['complex'] for w in local)))
        windows.extend(local)
    header = {k:deepcopy(v) for k,v in base.items() if k not in ('charts','windows','occurrences','audio_groups','totals','by_difficulty','files')}
    header['name'] = str(base.get('name','train'))+'-expert'
    header['policy'].update(chart_sampling='master-remaster-expert.v1',song_unit='song_id')
    return _assemble(header,windows,chart_meta,groups)


def _produce(args):
    return charts.produce(*args)


def _parallel(fn, jobs, workers, initializer=None, initargs=()):
    if workers == 1:
        if initializer: initializer(*initargs)
        return [fn(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers,initializer=initializer,initargs=initargs) as pool:
        return list(pool.map(fn,jobs))


def reference_manifest(old_root, new_root, new_ids, output, producer):
    original = read_json(Path(old_root)/'manifest.json')
    additions = read_json(Path(new_root)/'manifest.json')
    union = deepcopy(original)
    old_sha,new_sha=sha(Path(old_root)/'manifest.json'),sha(Path(new_root)/'manifest.json')
    refs = {key:dict(root=str(Path(old_root).resolve()),path=pack_path('.',key).as_posix(),
        producer=original.get('producer'),manifest_sha256=old_sha) for key in original['charts']}
    refs.update({key:dict(root=str(Path(new_root).resolve()),path=pack_path('.',key).as_posix(),
        producer=additions.get('producer'),manifest_sha256=new_sha) for key in new_ids})
    union.update(charts=list(refs),references=refs,producer=producer,
                 source_manifests=[dict(root=str(Path(r).resolve()),sha256=sha(Path(r)/'manifest.json')) for r in (old_root,new_root)])
    write_json(Path(output)/'manifest.json',union)
    return union


def build(config, output, *, parser, workers=12, source_root=None, wav_root=None,
          allow_unrepresentable=False):
    from maidata.training_common.checkpoint import source_identity
    from maidata.preparation.produce_compactspec import produce_job
    output = Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    old_chart_root = Path(config['pretrain_chart_cache']).resolve()
    audio_root = Path(config['pretrain_audio_cache']).resolve()
    audio = read_json(audio_root/'catalog.json')
    source_root = Path(source_root or config['pretrain_source_root']).resolve()
    source = source_identity();producer = dict(kind='incremental-expert.v1',source=source)
    plans, inputs, rows = {}, {}, {}
    for side in ('train','validation'):
        root = Path(config[f'pretrain_{side}_plan']).resolve()
        base = read_json(root/'plan.json')
        windows = [json.loads(s) for s in (root/base['files']['windows']).read_text(encoding='utf-8').splitlines() if s]
        inputs[side] = (base,windows)
        rows[side] = discover(base,audio,source_root)
    all_rows = [row for side in rows.values() for row in side]
    chart_root = output/'charts';recipe = digest(producer)
    results = _parallel(_produce,[(r,str(chart_root),str(parser),recipe) for r in all_rows],workers,
        charts.initialize,(config['decoder_train_geometry'],config['decoder_train_slide_data']))
    write_json(chart_root/'production-summary.json',dict(recipe=recipe,results=results))
    excluded = [r for r in results if r['status']!='success']
    if excluded and (not allow_unrepresentable or any(r['status']!='unrepresentable' for r in excluded)):
        # Report the producer's existing per-chart failures; do not publish a usable union.
        return dict(complete=False,results=results)
    accepted = {r['chart_id'] for r in results if r['status']=='success'}
    candidates = len(all_rows)
    rows = {side:[r for r in values if r['chart_id'] in accepted] for side,values in rows.items()}
    all_rows = [row for values in rows.values() for row in values]
    old_catalog = read_json(old_chart_root/'catalog.json')
    refs = {r['chart_id']:dict(root=str(old_chart_root),path=r['path'],producer=old_catalog.get('producer_recipe')) for r in old_catalog['charts']}
    refs.update({r['chart_id']:dict(root=str(chart_root),path=r['path'],producer=recipe) for r in all_rows})
    write_json(chart_root/'catalog.json',dict(old_catalog,charts=old_catalog['charts']+all_rows,
        references=refs,producer_recipe=recipe,source_catalog=str(old_chart_root/'catalog.json')))
    records = {r['chart_id']:load_chart(chart_root/r['path'],validate_input=False) for r in all_rows}
    for side,(base,windows) in inputs.items():
        plans[side] = extend_plan(base,windows,rows[side],[records[r['chart_id']] for r in rows[side]],chart_root,old_chart_root)
        plans[side]['sources'].update(chart_cache=str(chart_root),chart_catalog=str(chart_root/'catalog.json'),
            chart_production_summary=str(chart_root/'production-summary.json'),base_plan=str(Path(config[f'pretrain_{side}_plan']).resolve()/'plan.json'))
        write_plan(plans[side],output/'plans'/side)
    # Only newly required timing variants are extracted, with the accepted fixed stats.
    joined_audio = deepcopy(audio);joined_audio['grid_references'] = {
        key:str(audio_root/'grids'/f'{key}.npz') for key in audio['variants']}
    missing = defaultdict(list)
    for row in all_rows:
        key,asset = row['audio_variant'],row['audio_asset']
        if key in joined_audio['variants']: continue
        variant = dict(asset=asset,timing=row['timing'],samples_per_beat=24,charts=[row['chart_id']])
        joined_audio['variants'][key] = variant;missing[asset].append((key,variant))
    new_audio = output/'audio'
    jobs=[]
    for asset,variants in missing.items():
        value = dict(audio['assets'][asset])
        if wav_root: value['wav_path']=str(Path(wav_root)/Path(value['wav_path']).parent.name/Path(value['wav_path']).name)
        jobs.append((str(new_audio),asset,value,variants,audio['stats']))
    _parallel(partial(produce_job,representation='compactspec'),jobs,workers)
    write_json(new_audio/'catalog.json',joined_audio)
    write_json(new_audio/'stats.json',read_json(audio_root/'stats.json'))
    # Reuse the established exporter, including its resume recipe binding.
    from maidata.streaming.cache import main as export_cache
    cache_root=output/'expert-v5'
    export_plans={}
    for side,(base,_) in inputs.items():
        empty=dict(base,charts=[],audio_groups=[])
        expert_plan=extend_plan(empty,[],rows[side],[records[r['chart_id']] for r in rows[side]],chart_root,old_chart_root)
        root=output/'expert-plans'/side
        write_plan(expert_plan,root);export_plans[side]=str(root/'plan.json')
    write_json(output/'export-plans.json',export_plans)
    write_json(output/'producer-config.json',config)
    export_cache(['--config',str(output/'producer-config.json'),'--plans',str(output/'export-plans.json'),
                  '--chart-cache',str(chart_root),'--output',str(cache_root),'--workers',str(workers)])
    reference_manifest(config['whole_chart_event_cache'],cache_root,[r['chart_id'] for r in all_rows],output/'whole-chart-v5',producer)
    ready=dict(config,pretrain_chart_cache=str(chart_root),decoder_train_chart_cache=str(chart_root),
        pretrain_audio_cache=str(new_audio),decoder_train_audio_cache=str(new_audio),whole_chart_event_cache=str(output/'whole-chart-v5'),
        pretrain_source_root=str(source_root),decoder_train_source_root=str(source_root))
    for side in plans:
        for prefix in ('pretrain','decoder_train'): ready[f'{prefix}_{side}_plan']=str(output/'plans'/side)
    write_json(output/'train-config.json',ready)
    report=dict(complete=True,expert_charts=len(all_rows),candidate_expert_charts=candidates,
        allow_unrepresentable=allow_unrepresentable,excluded_expert_charts=excluded,
        existing_audio_variants=len({r['audio_variant'] for r in all_rows if r['audio_variant'] in audio['variants']}),
        missing_audio_variants=sum(map(len,missing.values())),
        expert_by_partition={side:len(rs) for side,rs in rows.items()},config=str(output/'train-config.json'))
    write_json(output/'production-summary.json',report)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--base-config',type=Path,required=True,help='Accepted source asset/plan configuration')
    p.add_argument('--output',type=Path,default=Path('E:/01YD/0wd-maidata/global-cross-expert'))
    p.add_argument('--parser',type=Path,required=True)
    p.add_argument('--workers',type=int,default=12)
    p.add_argument('--source-root',type=Path)
    p.add_argument('--wav-root',type=Path)
    p.add_argument('--allow-unrepresentable',action='store_true',
                   help='Explicitly publish representable charts and retain per-chart exclusions')
    args=p.parse_args(argv)
    from maidata.entry import CONFIG
    from maidata.joint_training.recipe import resolve
    config=resolve(dict(CONFIG,**read_json(args.config)))
    base=resolve(dict(CONFIG,**read_json(args.base_config)))
    from maidata.joint_training.recipe import PATH_KEYS
    config['whole_chart_event_cache']=base['whole_chart_event_cache']
    config.update({key:base[key] for key in PATH_KEYS if key!='joint_train_output' and key in base})
    print(json.dumps(build(config,args.output,parser=args.parser.resolve(),workers=args.workers,
        source_root=args.source_root,wav_root=args.wav_root,
        allow_unrepresentable=args.allow_unrepresentable),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
