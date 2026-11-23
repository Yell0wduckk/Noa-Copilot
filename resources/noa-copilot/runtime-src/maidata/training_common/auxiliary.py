"""Seed, scheduler, and autocast helpers for joint training."""
from contextlib import nullcontext
import math
import random
import numpy as np
import torch


def seed_scratch(config):
    """Call once before fresh model construction; resume must restore RNG instead."""
    if config['pretrain_initialization'] != 'scratch':
        raise ValueError('fresh trainable components require scratch initialization')
    seed = config['pretrain_seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def epoch_range(completed_epochs, config):
    stop, horizon = config['pretrain_stop_after_epochs'], config['pretrain_scheduler_epochs']
    if (any(type(n) is not int for n in (completed_epochs, stop, horizon))
            or not 0 <= completed_epochs <= stop <= horizon or stop < 1):
        raise ValueError('require 0 <= completed epochs <= absolute stop <= scheduler horizon')
    return range(completed_epochs + 1, stop + 1)


def make_scheduler(optimizer, steps_by_epoch, config):
    """Linear warmup + cosine decay using the full horizon's optimizer steps."""
    epoch_range(0, config)
    horizon, warmup = config['pretrain_scheduler_epochs'], config['pretrain_warmup_steps']
    if (len(steps_by_epoch) != horizon
            or any(type(n) is not int or n < 1 for n in steps_by_epoch)
            or type(warmup) is not int or warmup < 0):
        raise ValueError('scheduler needs positive update counts for every horizon epoch and nonnegative warmup')
    total = sum(steps_by_epoch)

    def factor(step):
        if warmup and step < warmup:
            return min(1., max(1e-6, (step + 1) / warmup))
        progress = (step - warmup) / max(1, total - warmup)
        return .5 * (1. + math.cos(math.pi * min(1., max(0., progress))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def autocast(device, config):
    from maidata.nn.precision import enable_tf32
    enable_tf32(device)
    precision = config['pretrain_precision']
    if precision not in ('bf16', 'fp32'):
        raise ValueError('pretraining precision must be bf16 or fp32')
    return (torch.autocast(torch.device(device).type, dtype=torch.bfloat16)
            if precision == 'bf16' else nullcontext())
