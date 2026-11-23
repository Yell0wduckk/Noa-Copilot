"""Offline whole-chart compilation. Workers load packs without grammar replay."""
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import torch
from maidata.chart_cache.cache import load_chart, write_json
from maidata.draft.chart_data import from_record
from maidata.physical import Geometry
from maidata.training_common.decoder_window_cache import tensor_pack, tensor_unpack
from . import PHYSICAL_V7_CACHE_SCHEMA, PHYSICAL_V8_CACHE_SCHEMA, PHYSICAL_ARCHITECTURE, COMPACT_ARCHITECTURE


def compile_chart(chart,geometry,duration,*,start=None,end=None,audio_beats=None,audio_valid=None,
                  weighted_spans=(),multiplier=1.,state_ce_weight=.25,physical_enabled=True,state_tokens=False,audio_before=8.,audio_after=None,no_placeholders=True,**unused):
    from .physical_compile import compile_chart as compile_v6
    return compile_v6(chart,geometry,duration,start=start,end=end,audio_beats=audio_beats,audio_valid=audio_valid,
                      weighted_spans=weighted_spans,multiplier=multiplier,state_ce_weight=state_ce_weight,
                      physical_enabled=physical_enabled,state_tokens=False,audio_before=audio_before,
                      audio_after=audio_after,no_placeholders=no_placeholders)


def pack_path(root,key):
    return Path(root)/'charts'/(hashlib.sha256(str(key).encode()).hexdigest()+'.pt')


class WholeChartCache:
    def __init__(self,root,keys=(),*,notes_only=True,markers=True,physical_v6=True,physical_enabled=True,state_tokens=False,no_placeholders=True):
        self.root = Path(root)
        self.manifest = json.loads((self.root/'manifest.json').read_text('utf-8'))
        schema,architecture = (PHYSICAL_V8_CACHE_SCHEMA,COMPACT_ARCHITECTURE) if no_placeholders else (PHYSICAL_V7_CACHE_SCHEMA,PHYSICAL_ARCHITECTURE)
        if self.manifest.get('schema')!=schema or self.manifest.get('architecture')!=architecture:
            raise ValueError('whole-chart cache schema or architecture differs')
        if self.manifest.get('compiler_config',{}).get('decoder_physical_enabled',True)!=physical_enabled:
            raise ValueError('whole-chart cache Physical mode differs from training config')
        self.keys = frozenset(self.manifest['charts'])
        if not set(keys)<=self.keys: raise ValueError('whole-chart cache lacks requested chart keys')

    def read(self,key):
        if key not in self.keys: raise KeyError(f'chart is not in whole-chart manifest: {key}')
        reference = self.manifest.get('references',{}).get(key)
        path = pack_path(self.root,key) if reference is None else Path(reference['root'])/reference['path']
        return tensor_unpack(torch.load(path,map_location='cpu',weights_only=True))


def _export(task):
    key,path,duration,start,end,audio_first,audio_end,spans,multiplier,config,output = task
    torch.set_num_threads(1)
    destination = pack_path(output,key)
    if destination.exists():
        return key
    geometry = _geometry(config['decoder_train_geometry'],config['decoder_train_slide_data'])
    ir_path=config.get('_physical_ir_index',{}).get(key)
    if ir_path is not None:
        from .physical_codec import from_ir
        from dataclasses import replace
        raw=json.loads(Path(ir_path).read_text('utf-8-sig'))
        chart,_=from_ir(raw)
        song,difficulty=str(key).rsplit(':',1)
        chart=replace(chart,song_id=song,difficulty=difficulty,version=raw.get('version'))
    else:
        chart = from_record(load_chart(path,validate_input=False),validate_input=False)
    from . import decoder_windows, compact
    windows=decoder_windows(config)
    pack = compile_chart(chart,geometry,duration,start=start,end=end,
        audio_beats=torch.arange(audio_first,audio_end,1/8)+1/16,weighted_spans=spans,multiplier=multiplier,
        no_placeholders=compact(config),physical_enabled=config.get("decoder_physical_enabled",True),
        state_tokens=False,audio_before=windows.get("audio_before",8.),audio_after=windows.get("audio_after",8.))
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    torch.save(tensor_pack(pack),temporary); temporary.replace(destination)
    return key


@lru_cache(maxsize=1)
def _geometry(root,slides):
    return Geometry.from_resources(root,slides)


def resume_manifest(previous,current,output,*,allow_source_change=False):
    fields=('schema','architecture','plans','charts','compiler_config','partition')
    active=previous.get('active_producer',previous.get('producer'))
    if (any(previous.get(k)!=current.get(k) for k in fields) or
            active!=current['producer'] and not allow_source_change):
        raise ValueError('whole-chart export identity differs; use a new output directory')
    if active==current['producer']:
        return dict(previous,workers=current['workers'])
    producers=previous.get('chart_producers',{})
    return dict(current,producer=previous['producer'],active_producer=current['producer'],
        chart_producers={key:(producers.get(key,previous['producer']) if pack_path(output,key).is_file()
                              else current['producer']) for key in current['charts']})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--plans',type=Path,required=True,help='JSON mapping plan names to plan.json files')
    parser.add_argument('--chart-cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--resume-source-change',action='store_true')
    parser.add_argument('--physical',action=argparse.BooleanOptionalAction,default=None)
    parser.add_argument('--ir-index',type=Path)
    parser.add_argument('--shard-count',type=int,default=1)
    parser.add_argument('--shard-index',type=int,default=0)
    args = parser.parse_args(argv)
    if not 0 <= args.shard_index < args.shard_count:
        parser.error('shard-index must be within shard-count')
    from maidata.entry import CONFIG
    from maidata.training_common.checkpoint import source_identity
    config = dict(CONFIG,**json.loads(args.config.read_text('utf-8-sig')))
    if args.physical is not None:config['decoder_physical_enabled']=args.physical
    if args.ir_index is not None:
        config['_physical_ir_index']={k:str((args.ir_index.parent/Path(v)).resolve()) for k,v in json.loads(args.ir_index.read_text('utf-8-sig')).items()}
    from maidata.preparation.audio_padding import audio_padding
    before,after=audio_padding(config)
    tasks,plan_hashes = {},{}
    for name,file in json.loads(args.plans.read_text('utf-8-sig')).items():
        file = Path(file); raw = file.read_bytes(); plan = json.loads(raw)
        plan_hashes[name] = hashlib.sha256(raw).hexdigest()
        windows = [json.loads(s) for s in (file.parent/plan['files']['windows']).read_text('utf-8').splitlines()]
        charts = {c['chart_id']:c for c in plan['charts']}
        for group in plan['audio_groups']:
            from maidata.time import TimingMap
            from .coverage import coverage as whole_coverage
            timing = TimingMap.from_audio_timing(group['timing'],validate_input=False)
            first = min(w['start_beat'] for w in windows if w['audio_variant']==group['audio_variant'])
            coverage = whole_coverage(timing,group['audio_end_seconds'],first_beat=first)
            end = coverage.first_beat+coverage.draft_beats
            for key in group['chart_ids']:
                spans = [(w['start_tick'],w['end_tick']) for w in windows if w['chart_id']==key and w['complex']]
                tasks[key] = (key,Path(charts[key].get('chart_cache_root',args.chart_cache))/charts[key]['chart_cache_path'],group['audio_end_seconds'],
                    first,end,first-before,end+after,spans,plan['config']['complex_rhythm_multiplier'],config,args.output)
    scope = sorted(tasks)
    tasks = {key:tasks[key] for key in scope[args.shard_index::args.shard_count]}
    args.output.mkdir(parents=True,exist_ok=True)
    from . import cache_identity
    from maidata.physical.touch_v6 import CONTRACT
    from .physical_compile import ATTRIBUTION_CONTRACT
    schema,architecture = cache_identity(config)
    manifest = dict(schema=schema,architecture=architecture,plans=plan_hashes,
                    producer=source_identity(),charts=sorted(tasks),workers=args.workers,
                    compiler_config={k:config[k] for k in ('fpn_rates','decoder_train_geometry','decoder_train_slide_data')})
    manifest['compiler_config'].update(
        decoder_physical_enabled=config.get('decoder_physical_enabled',True),
        duration_contract='global-tick96.v1',
        physical_attribution_contract=ATTRIBUTION_CONTRACT,
        physical_touch_contract=CONTRACT,
        decoder_state_tokens=False,
    )
    if args.ir_index is not None: manifest['compiler_config']['ir_index_sha256']=hashlib.sha256(args.ir_index.read_bytes()).hexdigest()
    manifest = json.loads(json.dumps(manifest))
    if args.shard_count != 1:
        manifest['partition'] = dict(index=args.shard_index,count=args.shard_count,charts=scope)
    identity = args.output/'manifest.json'
    recipe = args.output/'export.json'
    prior = identity if identity.exists() else recipe
    if prior.exists():
        previous = json.loads(prior.read_text('utf-8'))
        manifest=resume_manifest(previous,manifest,args.output,allow_source_change=args.resume_source_change)
    atomic_manifest(recipe,manifest)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for key in pool.map(_export,tasks.values()): print(key,flush=True)
    if not all(pack_path(args.output,key).is_file() for key in tasks):
        raise ValueError('whole-chart export is incomplete')
    atomic_manifest(identity,manifest)
    return 0


def merge_shards(roots,output):
    roots = [Path(root) for root in roots]; output = Path(output)
    manifests = [json.loads((root/'manifest.json').read_text('utf-8')) for root in roots]
    first = manifests[0]
    scope = first['partition']['charts']; count = first['partition']['count']
    seen = set(); indices = set()
    for root,manifest in zip(roots,manifests):
        partition = manifest['partition']
        if any(manifest[k]!=first[k] for k in ('schema','architecture','plans','producer','compiler_config')):
            raise ValueError('whole-chart shard identity differs')
        index = partition['index']
        if partition['charts']!=scope or partition['count']!=count or index in indices:
            raise ValueError('whole-chart shard partition differs or repeats')
        if manifest['charts']!=scope[index::count] or seen.intersection(manifest['charts']):
            raise ValueError('whole-chart shard keys differ or overlap')
        if not all(pack_path(root,key).is_file() for key in manifest['charts']):
            raise ValueError('whole-chart shard is incomplete')
        indices.add(index); seen.update(manifest['charts'])
    if indices!=set(range(count)) or seen!=set(scope):
        raise ValueError('whole-chart shards do not cover complete scope')
    merged = {k:v for k,v in first.items() if k!='partition'}
    merged.update(charts=scope,workers=[m['workers'] for m in manifests],
                  shards=[dict(partition=m['partition']['index'],charts=len(m['charts']),workers=m['workers'])
                          for m in manifests])
    output.mkdir(parents=True,exist_ok=True)
    recipe = output/'export.json'
    if recipe.exists() and json.loads(recipe.read_text('utf-8'))!=merged:
        raise ValueError('whole-chart merge identity differs; use a new output directory')
    atomic_manifest(recipe,merged)
    for root,manifest in zip(roots,manifests):
        for key in manifest['charts']:
            destination = pack_path(output,key)
            if destination.exists(): continue
            destination.parent.mkdir(parents=True,exist_ok=True)
            temporary = destination.with_suffix('.tmp')
            try: os.link(pack_path(root,key),temporary)
            except OSError: shutil.copyfile(pack_path(root,key),temporary)
            temporary.replace(destination)
    atomic_manifest(output/'manifest.json',merged)
    return merged


def atomic_manifest(path,value):
    temporary = path.with_suffix('.tmp')
    write_json(temporary,value)
    temporary.replace(path)


if __name__ == '__main__': raise SystemExit(main())
