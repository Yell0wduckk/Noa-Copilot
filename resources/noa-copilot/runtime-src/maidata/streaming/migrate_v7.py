"""Strip v7 structural placeholders into v8 without charts, audio or Oracle replay."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import argparse
import json
import time
import torch
from . import PHYSICAL_V7_CACHE_SCHEMA, PHYSICAL_V8_CACHE_SCHEMA, PHYSICAL_ARCHITECTURE, COMPACT_ARCHITECTURE
from .compact_schema import PLACEHOLDERS, TOUCH_PADS, VOCAB_OFFSETS
from .attention import layouts
from .cache import pack_path, atomic_manifest
from maidata.training_common.decoder_window_cache import tensor_pack, tensor_unpack

METHOD = 'strip-v7-structural-placeholders.v1'


def _worker_init():
    torch.set_num_threads(1)


def _payloads(root, source, keys):
    references = source.get('references', {})
    for key in keys:
        ref = references.get(key)
        yield key, pack_path(root, key) if ref is None else Path(ref['root'])/ref['path']


def migrate_pack(pack):
    types=pack['tokens']['types']
    keep=~torch.isin(types,torch.tensor(PLACEHOLDERS,dtype=types.dtype))
    mapping=keep.long().cumsum(0)-1
    tokens={k:v[keep] for k,v in pack['tokens'].items()}
    labels={}; physical_delta=0.
    for key,group in pack['labels'].items():
        if key!='type':
            labels[key]=dict(group,indices=mapping[group['indices']]); continue
        idx=group['indices']; chord_prefix=types[idx]==18
        target=group['target'].clone(); allowed=group['allowed'].clone()
        touch_available=bool(allowed[:,5].any())
        penalty=group['penalty'].clone() if 'penalty' in group else None
        scale=group['category_scale'].clone() if penalty is not None else None
        if penalty is not None:
            positions={int(pos):r for r,pos in enumerate(idx.tolist())}
            for r in chord_prefix.nonzero(as_tuple=True)[0].tolist():
                if not bool(scale[r].any()): continue
                previous=positions.get(int(idx[r])-1)
                if previous is not None and scale[previous,0]==0 and penalty[r,4,0]:
                    penalty[previous,4,0]=True; scale[previous,0]=scale[r,0]
                # The removed prefix already owns a separate Physical vote,
                # even when its capacity mask is transferred to a kept row.
                physical_delta-=float(group['main'][r])
        retain=~chord_prefix
        idx=idx[retain]; target=target[retain]; allowed=allowed[retain]
        if penalty is not None: penalty=penalty[retain]; scale=scale[retain]
        for old,new in ((5,19),(7,8),(10,23),(17,9)):
            target[target==old]=new
            allowed[:,new] |= allowed[:,old]
            if penalty is not None: penalty[:,new] |= penalty[:,old]
        chord=target==18
        # The next retained input is the true chord PAD/TOUCH_PAD payload.
        if chord.any():
            next_pos=idx[chord]+1
            while torch.isin(types[next_pos],torch.tensor(PLACEHOLDERS)).any():
                next_pos += torch.isin(types[next_pos],torch.tensor(PLACEHOLDERS)).long()
            target[chord]=types[next_pos]
        # Compact Touch holds are available only at the central sensor.
        source_pos=idx.clone()
        flags=types[source_pos]==6
        source_pos[flags]-=1
        touch=types[source_pos]==19
        central=pack['tokens']['vocab_ids'][source_pos]==VOCAB_OFFSETS[19]+TOUCH_PADS.index('C')
        allowed[touch & ~central,8]=False
        allowed[:,4] |= allowed[:,18]
        allowed[:,19] |= allowed[:,18] & touch_available
        allowed[:,list(PLACEHOLDERS)]=False
        # Existing accepted charts include non-central one-tick Touch holds.
        # Preserve the compiler's GT-positive floor after candidate narrowing.
        allowed.scatter_(1,target[:,None],True)
        if penalty is not None:
            penalty[:,4] |= penalty[:,18]
            penalty[:,list(PLACEHOLDERS)]=False
            effective=(penalty & allowed[:,:,None]).any(1) & ~(penalty | ~allowed[:,:,None]).all(1)
            penalty=torch.where(effective[:,None,:],penalty,False)
            scale=torch.where(effective,scale,0.)
        row={k:v[retain] for k,v in group.items() if k not in ('indices','target','allowed','penalty','category_scale')}
        row.update(indices=mapping[idx],target=target,allowed=allowed)
        if penalty is not None: row.update(penalty=penalty,category_scale=scale)
        labels[key]=row
    probes=dict(pack['negative_probes'],indices=mapping[pack['negative_probes']['indices']])
    meta=dict(pack['metadata'],tokens=int(keep.sum()),placeholders=False)
    audio=meta['audio_first']+torch.arange(meta['audio_length'])/8
    valid=torch.zeros(len(audio),dtype=torch.bool)
    source=pack['audio_layout']; valid[source['indices'][source['allowed'].any(1)]]=True
    addressing=layouts(tokens['causal_beats'],tokens['slide_beats'],audio,valid,
                       query_width=pack['self_layout']['query_width'])
    return dict(pack,schema=PHYSICAL_V8_CACHE_SCHEMA,architecture=COMPACT_ARCHITECTURE,
                tokens=tokens,labels=labels,negative_probes=probes,
                counts=dict(pack['counts'],physical=float(pack['counts']['physical'])+physical_delta),
                metadata=meta,**addressing,
                migration=dict(method=METHOD,source_schema=pack['schema'],source_tokens=len(keep),
                               removed_tokens=int((~keep).sum())))


def _convert(key, payload, output):
    from io import BytesIO
    began = time.perf_counter()
    pack = tensor_unpack(torch.load(BytesIO(payload) if isinstance(payload, bytes) else payload,
                                    map_location='cpu', weights_only=True))
    result = migrate_pack(pack)
    path = pack_path(output, key)
    temporary = path.with_suffix('.tmp')
    torch.save(tensor_pack(result), temporary)
    temporary.replace(path)
    return dict(chart=key, **result['migration'], tokens=result['metadata']['tokens'],
                source_physical=pack['counts']['physical'], physical=result['counts']['physical'],
                bytes=path.stat().st_size, seconds=time.perf_counter()-began)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args(argv)
    source = json.loads((args.source/'manifest.json').read_text('utf-8-sig'))
    if source['schema'] != PHYSICAL_V7_CACHE_SCHEMA or source['architecture'] != PHYSICAL_ARCHITECTURE:
        parser.error('source must be a Physical v7 cache')
    from maidata.training_common.checkpoint import source_identity
    compiler = dict(source['compiler_config'], decoder_state_tokens=False)
    compiler.pop('decoder_state_ce_weight', None)
    lineage = {k: source[k] for k in ('producer', 'active_producer', 'chart_producers') if k in source}
    manifest = dict(source, schema=PHYSICAL_V8_CACHE_SCHEMA, architecture=COMPACT_ARCHITECTURE, compiler_config=compiler,
                    producer=source_identity(), workers=args.workers,
                    migration=dict(method=METHOD, source=str(args.source.resolve()),
                                   source_schema=source['schema'], **lineage))
    for key in ('references', 'active_producer', 'chart_producers'):
        manifest.pop(key, None)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'charts').mkdir(exist_ok=True)
    recipe = args.output/'export.json'
    prior = args.output/'manifest.json'
    if not prior.exists():
        prior = recipe
    if prior.exists():
        previous = json.loads(prior.read_text('utf-8'))
        if any(previous.get(k) != manifest.get(k) for k in
               ('schema', 'architecture', 'plans', 'charts', 'compiler_config', 'partition', 'migration')):
            parser.error('output belongs to a different export; use a new directory')
        manifest['producer'] = previous['producer']
    atomic_manifest(recipe, manifest)
    remaining = [k for k in source['charts'] if not pack_path(args.output, k).exists()]
    completed = len(source['charts'])-len(remaining)
    began = time.perf_counter()
    if remaining:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as pool:
            pending = set()

            def collect(done):
                nonlocal completed
                for future in done:
                    row = future.result()
                    completed += 1
                    print(json.dumps(dict(completed=completed, total=len(source['charts']), **row)), flush=True)

            for key, payload in _payloads(args.source, source, remaining):
                pending.add(pool.submit(_convert, key, payload, args.output))
                if len(pending) >= args.workers*2:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    collect(done)
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                collect(done)
    if not all(pack_path(args.output, k).is_file() for k in source['charts']):
        raise ValueError('migration is incomplete')
    atomic_manifest(args.output/'manifest.json', manifest)
    print(f'Complete: {completed} charts; {time.perf_counter()-began:.1f}s; {args.output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
