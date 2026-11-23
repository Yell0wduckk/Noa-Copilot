"""Produce independent CompactSpec-TOS assets; never reuse old spectra as T/O/S.

python -m maidata.preparation.produce_compactspec --catalog OLD/catalog.json
    --train-plan TRAIN/plan.json --output E:/01YD/0wd-maidata/compactspec-audio-DATE
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from functools import partial
from importlib import import_module
import json
from pathlib import Path
import time

import numpy as np
import torch

from .features import grid_coordinates
from .compactspec import read_stereo, resample_stereo, mid_side
from .produce import read_json, write_json, token, save_arrays
from .timing import TempoMap


def statistics_job(job, *, representation="compactspec"):
    features = import_module("." + representation, __package__)
    Statistics, raw_at_centers = features.Statistics, features.raw_at_centers
    asset, variants = job
    torch.set_num_threads(1)
    y,sr = read_stereo(asset['wav_path'])
    pcm,duration = resample_stereo(y,sr),len(y)/sr
    stats = Statistics()
    waves = mid_side(pcm)
    previous_centers = previous_valid = previous_stats = None
    for variant in variants:
        grid = grid_coordinates(TempoMap(variant['timing']),duration,len(pcm),24)
        centers,valid = grid['sample_indices'].reshape(-1),grid['sample_valid'].reshape(-1)
        if (previous_stats is not None and np.array_equal(centers,previous_centers)
                and np.array_equal(valid,previous_valid)):
            increment = previous_stats
        else:
            increment = Statistics()
            for start in range(0,len(centers),4096):
                mask = valid[start:start+4096]
                increment.add(raw_at_centers(pcm,centers[start:start+4096],mask,waves=waves),mask,1/len(variants))
        stats.total += increment.total
        stats.square += increment.square
        stats.weight += increment.weight
        previous_centers,previous_valid,previous_stats = centers,valid,increment

    return stats.total,stats.square,stats.weight


def produce_job(job, *, representation="compactspec"):
    features = import_module("." + representation, __package__)
    RECIPE, extract_grid = features.RECIPE, features.extract_grid
    output,asset_id,asset,variants,stats = job
    torch.set_num_threads(1)
    y,sr = read_stereo(asset['wav_path'])
    pcm,duration = resample_stereo(y,sr),len(y)/sr
    records = []
    for variant_id,variant in variants:
        path = Path(output)/'grids'/f'{variant_id}.npz'
        key = token(dict(asset=asset_id,timing=variant['timing'],recipe=RECIPE,stats=stats))
        meta_path = path.with_suffix('.json')
        if meta_path.exists():
            meta = read_json(meta_path)
            if meta['input_key'] != key or not path.is_file():
                raise ValueError(f'existing variant mismatch {variant_id}')
        else:
            grid = extract_grid(pcm,variant['timing'],duration,stats)
            grid['timing_json'] = np.array(json.dumps(grid.pop('timing'),sort_keys=True))
            grid['duration_seconds'] = np.array(grid['duration_seconds'],np.float64)
            meta = dict(input_key=key,variant=variant_id,asset=asset_id,representation=RECIPE['representation'],
                        **save_arrays(path,grid,grid=True))
            write_json(meta_path,meta)
        records.append(meta)
    return dict(asset=asset_id,grids=records)


def build(catalog_path, train_plan, output, *, wav_root=None, limit=None, workers=4, representation="compactspec"):
    features = import_module("." + representation, __package__)
    RECIPE, Statistics = features.RECIPE, features.Statistics
    catalog_path,output = Path(catalog_path),Path(output)
    original = read_json(catalog_path)
    if output.resolve() == catalog_path.parent.resolve():
        raise ValueError('Spectral output must be independent of the source catalog')
    catalog = deepcopy(original)
    if wav_root is not None:
        from .produce import jsonl
        rows = {r['output_sha256']:r for r in jsonl(Path(wav_root)/'audio-manifest.jsonl')}
        for key,asset in catalog['assets'].items():
            asset['wav_path'] = str(Path(wav_root)/rows[key]['output_relative_path'])
    plan_path = Path(train_plan)
    if plan_path.is_dir(): plan_path = plan_path/'plan.json'
    plan = read_json(plan_path)
    train_variants = {g['audio_variant'] for g in plan['audio_groups']}
    grouped = {key:[] for key in catalog['assets']}
    for key,variant in catalog['variants'].items():
        grouped[variant['asset']].append((key,variant))
    assets = list(catalog['assets'].items())
    if limit is not None: assets = assets[:limit]
    jobs = [(asset,[v for key,v in grouped[asset_id] if key in train_variants]) for asset_id,asset in assets]
    jobs = [(asset,variants) for asset,variants in jobs if variants]
    started = time.perf_counter()
    provenance = dict(source_catalog=str(catalog_path.resolve()),
        statistics_train_plan=str(plan_path.resolve()),spec_recipe=RECIPE,
        source_catalog_identity=token(original),statistics_train_plan_identity=token(plan),
        statistics_assets_identity=token(assets))
    stats_path = output/'stats.json'
    saved = read_json(stats_path) if stats_path.exists() else None
    if saved is not None and saved.get('provenance') == provenance and saved.get('recipe') == RECIPE:
        stats = {k:v for k,v in saved.items() if k != 'provenance'}
    else:
        aggregate = Statistics()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for total,square,weight in pool.map(partial(statistics_job,representation=representation),jobs):
                aggregate.total += total
                aggregate.square += square
                aggregate.weight += weight
        stats = aggregate.finish()
    for key in ('mel_recipe','aux_recipe','aux_keys','old_full_reuse','pilot_reuse','fixed_stats_path','fixed_stats_sha256'):
        catalog.pop(key,None)
    for variant in catalog['variants'].values(): variant.pop('reused_mel',None)
    catalog.update(spec_recipe=RECIPE,stats=stats,stats_identity=token(stats),
                   source_catalog=str(catalog_path.resolve()),statistics_train_plan=str(plan_path.resolve()))
    if (output/'catalog.json').exists() and read_json(output/'catalog.json') != catalog:
        raise ValueError('existing spectral catalog identity differs')
    write_json(output/'catalog.json',catalog)
    write_json(output/'stats.json',dict(stats,provenance=provenance))
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [(str(output),key,asset,grouped[key],stats) for key,asset in assets]
        for record in pool.map(partial(produce_job,representation=representation),jobs):
            records.append(record)
            print(f"DONE {len(records)}/{len(assets)} {record['asset'][:12]}",flush=True)
    summary = dict(complete=limit is None,assets=len(records),variants=sum(len(r['grids']) for r in records),
                   output_bytes=sum(g['bytes'] for r in records for g in r['grids']),
                   wall_seconds=time.perf_counter()-started,workers=workers,representation=RECIPE['representation'])
    write_json(output/('sample-summary.json' if limit is not None else 'production-summary.json'),summary)
    write_json(output/'production.json',records)
    return summary


def main(argv=None, *, representation="compactspec"):
    parser = argparse.ArgumentParser(description=f'Produce independent {representation} assets with weighted statistics and BF16 grids.')
    parser.add_argument('--catalog',required=True,type=Path)
    parser.add_argument('--train-plan',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--wav-root',type=Path)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--workers',type=int,default=4,help='offline production processes, not training DataLoader workers')
    args = parser.parse_args(argv)
    print(json.dumps(build(args.catalog,args.train_plan,args.output,wav_root=args.wav_root,
                           limit=args.limit,workers=args.workers,representation=representation),ensure_ascii=False))


if __name__ == '__main__':
    main()
