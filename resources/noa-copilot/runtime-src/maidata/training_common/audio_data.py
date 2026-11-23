"""Whole-audio plan ordering and DataLoader helpers for joint training."""
from collections import defaultdict
from dataclasses import fields, is_dataclass
import json
from pathlib import Path
import random

import torch
from torch.utils.data import Sampler
from maidata.training_common.batching import optimizer_batches, group_token_cost


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


class AudioPlan:
    def __init__(self, root, config):
        self.root = Path(root)
        self.manifest = read_json(self.root / 'plan.json')
        p = self.manifest
        self.windows = [json.loads(line) for line in
                        (self.root / p['files']['windows']).read_text(encoding='utf-8').splitlines()]
        self.by_audio = defaultdict(list)
        for w in self.windows:
            self.by_audio[w['audio_variant']].append(w)
        self.groups = {g['audio_variant']: g for g in p['audio_groups']}
        self.charts = {c['chart_id']: c for c in p['charts']}

    def order(self, epoch):
        result = list(self.by_audio)
        random.Random(f'{self.manifest["seed"]}:{epoch}:groups').shuffle(result)
        return result

    def rows(self, variant, epoch):
        result = list(self.by_audio[variant])
        random.Random(f'{self.manifest["seed"]}:{epoch}:within:{variant}').shuffle(result)
        return result

    def token_cost(self, variant, config):
        scheduling = dict(config)
        for key in ('audio_before_beats', 'target_beats', 'audio_after_beats'):
            scheduling[key] = self.manifest['config'][key]
        return group_token_cost(self.groups[variant], scheduling)

    def batches(self, epoch, config):
        from .expert_sampling import enabled, batches
        if enabled(config) and getattr(self, 'training', True):
            return batches(self, epoch, config)
        order = self.order(epoch)
        return optimizer_batches(order, config, token_costs=[self.token_cost(k, config) for k in order])


def map_tensors(value, fn):
    if isinstance(value, torch.Tensor):
        return fn(value)
    if is_dataclass(value):
        return type(value)(**{f.name: map_tensors(getattr(value, f.name), fn) for f in fields(value)})
    if isinstance(value, dict):
        return {k: map_tensors(v, fn) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(map_tensors(v, fn) for v in value)
    return value


class AudioBatchSampler(Sampler):
    def __init__(self, plan, config):
        self.plan, self.config, self.epoch = plan, config, plan.manifest['epoch']
        self.selected_step = None

    def steps(self):
        steps = self.plan.batches(self.epoch, self.config)
        return steps if self.selected_step is None else (steps[self.selected_step],)

    def __iter__(self):
        for step in self.steps():
            for batch in step:
                yield [(variant, self.epoch) for variant in batch]

    def __len__(self):
        return sum(len(s) for s in self.steps())


def worker_init(_):
    torch.set_num_threads(1)


def epoch_iterator(loader, restored_rng=None):
    iterator = iter(loader)
    if restored_rng is not None and loader.persistent_workers:
        loader.generator.set_state(restored_rng.cpu())
    return iterator
