"""Deterministic song-owned logical batches; workers only load selected charts."""
from collections import defaultdict
from dataclasses import dataclass
from itertools import islice
import random

POLICY = 'master-remaster-expert.v1'


def enabled(config):
    return config.get('joint_train_chart_sampling') == POLICY


@dataclass(frozen=True)
class Selection:
    variant: str
    chart_ids: tuple


def variant_key(value):
    return value.variant if isinstance(value, Selection) else value


def request_cost(plan, request, config):
    from .batching import group_token_cost
    if not isinstance(request, Selection): return group_token_cost(plan.groups[request],config)
    group = dict(plan.groups[request.variant])
    group['occurrences'] = sum(plan.charts[key].get('occurrences',0) for key in request.chart_ids)
    scheduling = dict(config,**{key:plan.manifest['config'][key]
        for key in ('audio_before_beats','target_beats','audio_after_beats')})
    return group_token_cost(group,scheduling)


def batches(plan, epoch, config):
    from maidata.preparation.heldout import excluded_song
    songs = defaultdict(list)
    for key, row in plan.charts.items():
        if excluded_song(row['song_id'],row.get('title','')):
            continue
        if str(row['difficulty']) in ('4','5','6'): songs[str(row['song_id'])].append(key)
    order = sorted(songs)
    seed = plan.manifest['seed']
    random.Random(f'{seed}:{epoch}:songs').shuffle(order)
    size = config.get('pretrain_songs_per_batch',4)
    k = config.get('joint_train_charts_per_song',2)
    limit = config.get('joint_train_max_charts_per_update')
    # Keep each song's Master/Re:MASTER charts together. A chart cap may move
    # a whole song to the next update, but never drops its preferred charts.
    song_batches, members, preferred = [], [], 0
    for song in order:
        count = sum(str(plan.charts[key]['difficulty']) in ('5','6') for key in songs[song])
        if members and (len(members) == size or (limit is not None and preferred + count > limit)):
            song_batches.append(members)
            members, preferred = [], 0
        members.append(song)
        preferred += count
    if members:
        song_batches.append(members)
    steps, chart_counts = [], []
    for members in song_batches:
        base, experts = [], []
        for song in members:
            keys = sorted(songs[song])
            base.extend(key for key in keys if str(plan.charts[key]['difficulty']) in ('5','6'))
            candidates = [key for key in keys if str(plan.charts[key]['difficulty']) == '4']
            if candidates: experts.append(candidates[0])
        target = k*len(members)
        if limit is not None:
            target = min(target,limit)
        budget = min(len(experts),max(0,target-len(base)))
        rng = random.Random(f'{seed}:{epoch}:expert:{",".join(members)}')
        chosen = base + rng.sample(experts,budget)
        by_variant = defaultdict(list)
        for key in chosen: by_variant[plan.charts[key]['audio_variant']].append(key)
        requests = [Selection(variant,tuple(keys)) for variant,keys in by_variant.items()]
        # Physical subdivisions preserve this logical song batch and its draw.
        from .batching import optimizer_batches
        costs = [request_cost(plan,request,config) for request in requests]
        physical = optimizer_batches(requests,dict(config,pretrain_gradient_accumulation_steps=1),token_costs=costs)
        steps.append(tuple(batch for step in physical for batch in step))
        chart_counts.append(len(chosen))
    accumulation = config['pretrain_gradient_accumulation_steps']
    updates, pending, count = [], [], 0
    for step, charts in zip(steps,chart_counts):
        if pending and (len(pending) == accumulation or (limit is not None and count + charts > limit)):
            updates.append(tuple(batch for part in pending for batch in part))
            pending, count = [], 0
        pending.append(step)
        count += charts
    if pending:
        updates.append(tuple(batch for part in pending for batch in part))
    return tuple(updates)


def logical_updates(loader, restored_rng=None):
    from .audio_data import epoch_iterator
    iterator = epoch_iterator(loader,restored_rng)
    for step in loader.batch_sampler.steps():
        rows = list(islice(iterator,len(step)))
        if rows: yield rows
