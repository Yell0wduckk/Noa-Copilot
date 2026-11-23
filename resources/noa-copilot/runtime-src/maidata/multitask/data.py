"""Metadata-only epoch planning, shared-song loading and time augmentation."""
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from . import AUDIO_SCHEMA, CACHE_SCHEMA


def excluded_song(row):
    meta = row.get('metadata', {})
    song_id = str(row.get('song_id', meta.get('song_id', ''))).lstrip('0')
    title = str(row.get('title', meta.get('title', '')))
    return song_id == '854' or '全世界共通リズム感テスト' in title


def read_manifest(path):
    path=Path(path)
    value=json.loads(path.read_text(encoding='utf-8-sig'))
    # Only the manifest is read for input binding; accepted payloads are not scanned.
    binding=dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 schema=value.get('schema'),producer=value.get('producer'),audio_schema=value.get('audio_schema'))
    style_tags=value.get('song_style_tags',{})
    if isinstance(style_tags,str):
        style_tags=json.loads((path.parent/style_tags).read_text(encoding='utf-8-sig'))
    rows=[]
    for row in value['records']:
        if excluded_song(row):
            continue
        row=dict(row)
        if row.get('features') and 'features_schema' not in row:
            row['features_schema'] = row.get('audio_schema') or value.get('audio_schema')
        for key in ('features','events','audio','midi','chart','macro_features'):
            if row.get(key):
                row[key]=str((path.parent/row[key]).resolve())
        tags=style_tags.get(str(row.get('song_id',row.get('id',''))))
        if tags:
            row['metadata']={**tags,**row.get('metadata',{})}
        rows.append(row)
    return rows,binding


def grid_corpus(row):
    dataset = str(row.get('dataset','')).casefold()
    if dataset in ('maestro','guitarset'):
        meta = row.get('metadata',{})
        return bool(meta.get('bpm_segments') and 'offset_seconds' in meta)
    return dataset not in {'musicnet','gtsinger','m4singer'}


def stage_sources(rows, stage):
    if stage != 0:
        return rows
    # Custom prepared grid-aligned corpora can still be named locally.
    return [r for r in rows if r['task'] == 'midi' and
            grid_corpus(r) and
            r.get('source_family', 'real_midi') in ('real_midi','bms','vita','studio')]


class EpochPlanner:
    """Stage 0 traverses acoustic labels; Stage 1 is anchored to game charts."""
    def __init__(self,rows,config,state=None):
        self.rows=rows
        self.config=config
        self.state=deepcopy(state or dict(epoch=0,cursors={},cycles={}))

    def next_epoch(self):
        epoch=self.state['epoch']+1
        rng=random.Random(self.config['seed']+epoch)
        acoustic = self.config['stage'] == 0
        anchor = 'midi' if acoustic else 'maimai'
        base=[i for i,r in enumerate(self.rows) if r['task']==anchor and r.get('split','train')=='train']
        rng.shuffle(base)
        synthetic = self.config.get('synthetic',{})
        # Share is by anchor chart occurrences, independent of audio length.
        # Round cumulatively: even small corpora realize the requested share.
        share = synthetic.get('fraction',0.)
        draws = synthetic.get('samples_per_epoch')
        def mixed_quota(real_count):
            quota = real_count*share/(1-share) if share else 0.
            return int(epoch*quota+.5)-int((epoch-1)*quota+.5)
        synthetic_only = share == 1.
        if synthetic_only:
            draws = max(1,len(base)) if draws is None else draws
            base = []
        elif draws is None:
            draws = mixed_quota(len(base))
        midi_cost = synthetic.get('estimated_midi_tokens',1024)
        game_cost = synthetic.get('estimated_maimai_tokens',256)
        selected=list(base)
        base_tokens=sum(self.rows[i]['token_count'] for i in base)
        proportions={'midi':1.0} if acoustic else self.config['task_proportions']
        # A pure acoustic epoch needs no maimai record or invented maimai
        # budget. Without an explicit source mixture every training item is
        # visited once. An explicit mixture preserves the total token budget.
        weighted_acoustic = acoustic and bool(self.config.get('dataset_proportions'))
        if weighted_acoustic:
            selected=[]
        for task,task_share in proportions.items():
            if synthetic_only or (task==anchor and not weighted_acoustic) or task_share<=0:
                continue
            pools=defaultdict(list)
            for i,row in enumerate(self.rows):
                if row['task']==task and row.get('split','train')=='train':
                    pools[row.get('dataset',task)].append(i)
            if not pools:
                continue
            names=sorted(pools)
            explicit=self.config.get('dataset_proportions',{})
            weights=[max(0.,explicit.get(n,0.)) if explicit else
                sum(self.rows[i]['token_count'] for i in pools[n])**self.config['dataset_temperature'] for n in names]
            if not sum(weights):continue
            target=(sum(self.rows[i]['token_count'] for n,w in zip(names,weights) if w>0 for i in pools[n])
                    if acoustic else max(0,(base_tokens+draws*game_cost)*task_share/proportions['maimai']
                                           - (draws*midi_cost if task=='midi' else 0)))
            consumed=0
            remaining=[target*w/sum(weights) for w in weights]
            orders = {}
            salts = {name:int.from_bytes(hashlib.sha256(f'{task}:{name}'.encode()).digest()[:4],'little')
                     for name in names}
            while consumed<target:
                # Dataset mixing, like task mixing, is in tokens rather than
                # chart counts (long MIDI files must not silently dominate).
                chosen=rng.choices(range(len(names)),weights=[max(0.,v) for v in remaining])[0]
                name=names[chosen]
                key=f'{task}:{name}'
                cursor=self.state['cursors'].get(key,0)
                cycle=self.state['cycles'].get(key,0)
                if name not in orders or orders[name][0]!=cycle:
                    order=list(pools[name])
                    random.Random(self.config['seed']+salts[name]+cycle).shuffle(order)
                    orders[name]=(cycle,order)
                else:
                    order=orders[name][1]
                item=order[cursor]
                selected.append(item)
                item_tokens=max(1,self.rows[item]['token_count'])
                consumed+=item_tokens
                remaining[chosen]-=item_tokens
                cursor+=1
                if cursor==len(order):
                    cursor=0
                    cycle+=1
                self.state['cursors'][key]=cursor
                self.state['cycles'][key]=cycle
        if weighted_acoustic and not synthetic_only and synthetic.get('samples_per_epoch') is None:
            draws = mixed_quota(len(selected))
        # Keep all selected charts of one occurrence's audio together. Repeated
        # external draws remain distinct occurrences rather than duplicate tokens.
        groups=[]
        first_audio_groups={}
        drawn=set()
        for i in selected:
            row=self.rows[i]
            if i not in drawn:
                # The accepted feature asset is the audio identity across
                # datasets/tasks (maimai + its pseudo MIDI share one encode).
                key=(('audio_id',row['audio_id']) if row.get('audio_id') else
                     ('audio',row['audio']) if row.get('audio') else
                     ('features',row['features']) if row.get('features') else ('record',i))
                if key not in first_audio_groups:
                    first_audio_groups[key]=len(groups)
                    groups.append([])
                groups[first_audio_groups[key]].append(i)
                drawn.add(i)
            else:
                # A repeated external draw is a new occurrence, not another
                # copy of its labels in the original occurrence's song group.
                groups.append([i])
        groups.extend([-(i+1)] for i in range(draws))
        rng.shuffle(groups)
        batches=[]
        batch=[]
        tokens=0
        for group in groups:
            size=(sum(self.rows[i]['token_count'] for i in group) if group[0]>=0 else
                  midi_cost+(0 if acoustic else game_cost))
            if batch and (len(batch)>=self.config['audio_groups_per_batch'] or
                          tokens+size>self.config['token_budget']):
                batches.append(batch)
                batch=[]
                tokens=0
            batch.append(group)
            tokens+=size
        if batch:
            batches.append(batch)
        self.state['epoch']=epoch
        return batches


def schedule_plans(rows,config):
    planner=EpochPlanner(rows,config)
    plans=[]
    states=[]
    for _ in range(config['scheduler_epochs']):
        plans.append(planner.next_epoch())
        states.append(deepcopy(planner.state))
    return plans,states


def load_events(path):
    from .codec import Event
    values=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if isinstance(values,dict):
        values=values['events']
    return [Event(**value) for value in values]




class UpdateDataset(torch.utils.data.Dataset):
    def __init__(self,rows,batches,config,epoch,*,training=True):
        self.rows,self.batches,self.config,self.epoch=rows,batches,config,epoch
        self.training=training

    def __len__(self):
        return len(self.batches)

    def __getitem__(self,index):
        from .codec import MidiCropHistory,encode_chart,encode_moment_only_chart,crop_events,stretch_events,mirror_events
        from .timbre_tags import attach_unconditioned_moment_types,crop_and_stretch_intervals,sample_tag_moment_charts
        rng=random.Random(self.config['seed']+self.epoch*1000003+index)
        tm_cfg=self.config.get('tag_moment') or {}
        tm_enabled=bool(tm_cfg.get('enabled',False))
        result=[]
        for group in self.batches[index]:
            if group[0] < 0:
                from .synthetic import example, RATE
                occurrence = -group[0]-1
                # Unique seed per occurrence, independent of worker scheduling,
                # batch order and total number of draws in previous epochs.
                uid = (self.config['seed'] << 64) + (self.epoch << 32) + occurrence
                opt = self.config['synthetic']
                game = self.config['stage']==1
                wave,midi,chart,meta = example(uid,opt,with_chart=game)
                # Same downmix as read_waveform for real stereo recordings.
                wave = wave.mean(dim=1) if wave.ndim == 2 else wave
                duration = len(wave)/RATE
                charts = []
                for task,events in [('midi',midi)] + ([('maimai',chart)] if game else []):
                    if not tm_enabled or task!='midi' or tm_cfg.get('include_unconditioned_chart',True):
                        if tm_enabled and task=='midi' and tm_cfg.get('unconditioned_moment_only',False):
                            encoded = encode_moment_only_chart(task,events,duration_seconds=duration,metadata=meta)
                        else:
                            encoded = encode_chart(task,events,duration_seconds=duration,seed=uid,metadata=meta,
                                                   shuffle_chords=self.training)
                        if tm_enabled and task=='midi':
                            encoded.residual_neg_mask = torch.zeros(len(encoded.token_ids), dtype=torch.bool)
                            attach_unconditioned_moment_types(encoded, events, meta,
                                                              residual_neg_weight=tm_cfg.get('residual_neg_weight', .25))
                        charts.append(dict(encoded=encoded,task=task,metadata=meta,events=events,
                                           record=dict(id=f'studio/{self.epoch}/{occurrence}/{task}',dataset='studio')))
                    if self.training and tm_enabled and task=='midi':
                        charts.extend(sample_tag_moment_charts(
                            events, meta, duration_seconds=duration,
                            max_queries=int(tm_cfg.get('max_queries_per_song', 4)),
                            coarse_prob=float(tm_cfg.get('coarse_condition_prob', .5)),
                            max_coarse_subtypes=int(tm_cfg.get('max_coarse_subtypes', 2)),
                            max_collision_rate=float(tm_cfg.get('max_collision_rate', .05)),
                            min_events=int(tm_cfg.get('min_events_per_query', 1)),
                            rng=rng,
                            record_prefix=f'studio/{self.epoch}/{occurrence}/tag',
                        ))
                result.append(dict(waveform=wave,speed=1.,audio_frames=math.ceil(duration*125),
                                   charts=charts,synthetic=True))
                continue
            row = self.rows[group[0]]
            from .audio import audio_duration_seconds, frame_lengths, read_waveform
            duration = float(row.get('duration_seconds') or audio_duration_seconds(row['audio']))
            speed=(rng.uniform(*self.config['speed_range']) if self.training and
                   rng.random()<self.config['speed_probability'] else 1.)
            raw_span=self.config['max_audio_seconds']*speed
            start=rng.uniform(0,max(0.,duration-raw_span)) if self.training else 0.
            start=round(start*100)/100
            end=min(duration,start+raw_span)
            waveform = read_waveform(row['audio'], start_seconds=start,
                                     duration_seconds=None if start == 0. and end >= duration else end - start)
            frames = int(frame_lengths([len(waveform)], [speed])[0])
            loaded = dict(waveform=waveform, speed=speed, audio_frames=frames)
            macro_path = next((self.rows[i]['macro_features'] for i in group if self.rows[i].get('macro_features')), None)
            if macro_path:
                native = torch.load(macro_path, map_location='cpu', weights_only=True, mmap=True)
                origins = []
                for key, rate in (('mert',75.),('muq',25.)):
                    base_origin = float(native.get(f'{key}_origin', .5 / rate))
                    begin = max(0, math.floor(start * rate - base_origin * rate))
                    stop = min(len(native[key]), math.ceil(end * rate + base_origin * rate) + 1)
                    loaded[key] = native[key][begin:stop].clone()
                    origins.append(((begin + base_origin * rate) / rate - start) / speed)
                loaded['macro_origins'] = origins
            charts=[]
            eff_duration=(end-start)/speed
            for i in group:
                item=self.rows[i]
                history=MidiCropHistory(speed) if start>0 and item['task']=='midi' else None
                cropped=crop_events(load_events(item['events']),start,end-start,midi_history=history)
                events=stretch_events(cropped,speed) if speed!=1. else cropped
                if self.training and item['task']=='maimai' and rng.random()<self.config['mirror_probability']:
                    events=mirror_events(events)
                metadata={**{key:item[key] for key in ('song_id','difficulty','version') if key in item},
                          **item.get('metadata',{})}
                metadata['pseudo']=bool(item.get('pseudo',metadata.get('pseudo',False)))
                if speed!=1.:
                    metadata['level']=None
                metadata['crop_start']=start
                metadata['speed']=speed
                if metadata.get('residual_mixed_intervals'):
                    metadata['residual_mixed_intervals']=crop_and_stretch_intervals(
                        metadata['residual_mixed_intervals'], start, end-start, speed
                    )
                include_base=(not tm_enabled or item['task']!='midi'
                              or not self.training or tm_cfg.get('include_unconditioned_chart',True))
                tag_extras=[]
                if self.training and tm_enabled and item['task']=='midi':
                    tag_extras=sample_tag_moment_charts(
                        events, metadata, duration_seconds=eff_duration,
                        max_queries=int(tm_cfg.get('max_queries_per_song', 4)),
                        coarse_prob=float(tm_cfg.get('coarse_condition_prob', .5)),
                        max_coarse_subtypes=int(tm_cfg.get('max_coarse_subtypes', 2)),
                        max_collision_rate=float(tm_cfg.get('max_collision_rate', .05)),
                        min_events=int(tm_cfg.get('min_events_per_query', 1)),
                        rng=rng,
                        record_prefix=f"{item.get('id', i)}/tag",
                    )
                if include_base or not tag_extras:
                    if tm_enabled and item['task']=='midi' and self.training and tm_cfg.get('unconditioned_moment_only',False):
                        encoded=encode_moment_only_chart(item['task'],events,duration_seconds=eff_duration,metadata=metadata)
                    else:
                        encoded=encode_chart(item['task'],events,duration_seconds=eff_duration,
                                             seed=rng.randrange(2**31),metadata=metadata,midi_history=history,
                                             shuffle_chords=self.training)
                    if tm_enabled and item['task']=='midi':
                        encoded.residual_neg_mask=torch.zeros(len(encoded.token_ids),dtype=torch.bool)
                        attach_unconditioned_moment_types(encoded,events,metadata,
                                                          residual_neg_weight=tm_cfg.get('residual_neg_weight',.25))
                    charts.append(dict(encoded=encoded,task=item['task'],metadata=metadata,events=events,record=item))
                charts.extend(tag_extras)
            loaded['charts']=charts
            result.append(loaded)
        return result


def identity_collate(value):
    return value


def loader(dataset,workers,seed,generator=None):
    generator=generator if generator is not None else torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(dataset,batch_size=None,num_workers=workers,
        collate_fn=identity_collate,generator=generator,persistent_workers=workers>0,pin_memory=torch.cuda.is_available())
