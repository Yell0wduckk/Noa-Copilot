"""Causal grammar for cyclic beats, increasing MOMENT phases and micro-runs."""

import torch
from torch import nn
from torch.nn import functional as F
from .packing import BEAT, CYCLE, BOS, MOMENT, TIME_COUNT


class MomentTiming(nn.Module):
    @staticmethod
    def history(steps):
        """Last actual onset and pending micro-run at each row, across beat tokens."""
        real = steps[..., 0] == MOMENT
        tick = steps[..., 2]
        rows = torch.arange(1, tick.shape[1] + 1, device=tick.device)
        index = torch.where(real, rows, 0).cummax(1).values
        ticks = F.pad(tick, (1, 0), value=-96)
        last = ticks.gather(1, index)
        # Rows are one-based and a moment cannot be its own predecessor.
        before = F.pad(index[:, :-1], (2, 0))
        prior_index = before.gather(1, index)
        prior = ticks.gather(1, prior_index)
        earlier = ticks.gather(1, before.gather(1, prior_index))
        gap = last - prior
        owed = (gap > 0) & (gap <= 2) & (prior - earlier > 2)
        return last, owed, (gap > 0) & (gap <= 2)

    @staticmethod
    def times(logits, prev, cursor, last_onset, owed, limit_tick, last_small):
        ids = torch.arange(TIME_COUNT, device=logits.device)
        beat = cursor // 96
        next_beat = torch.where(prev == BOS, 0, beat + 1)
        at = beat[..., None] * 96 + ids
        moment = (ids < BEAT) & (prev[..., None] != BOS) & (at > last_onset[..., None])
        limit = torch.as_tensor(limit_tick, device=at.device)
        moment &= at <= limit[..., None]
        # An isolated micro-gap cannot be the last onset: one more onset is owed.
        moment &= ~(
            (at == limit[..., None])
            & (at - last_onset[..., None] <= 2)
            & ~last_small[..., None]
        )
        moment &= ~owed[..., None] | (at <= last_onset[..., None] + 2)
        advance = (ids == BEAT + next_beat[..., None] % CYCLE) & (
            ~owed | (next_beat * 96 <= torch.minimum(last_onset + 2, limit))
        )[..., None]
        return logits.masked_fill(~(moment | advance), -torch.inf)
