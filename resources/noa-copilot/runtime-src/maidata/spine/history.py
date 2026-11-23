"""Ordered committed word audio and integer-beat local self-attention."""

from collections import deque
from functools import lru_cache
from typing import NamedTuple

import numpy as np
import torch
from torch.nn import functional as F


def history_layout(
    first_ticks,
    query_ticks,
    lengths,
    beats=16,
    block=32,
    *,
    strict=False,
    release_ticks=None,
):
    """Monotone left edges and potentially visible KV blocks, built once on workers."""
    n = first_ticks.shape[1]
    blocks = (n + block - 1) // block
    left = np.zeros((len(lengths), n), np.int32)
    counts = np.zeros((len(lengths), 1, blocks), np.int32)
    indices = np.zeros((*counts.shape, blocks), np.int32)
    for b, size in enumerate(lengths):
        keys = np.asarray(first_ticks[b, :size])
        left[b, :size] = np.searchsorted(
            keys, (np.asarray(query_ticks[b, :size]) // 96 - beats) * 96
        )
        for j in range((size + block - 1) // block):
            start = left[b, j * block] // block
            chosen = ([0] if not strict and start else []) + list(range(start, j + 1))
            if release_ticks is not None and start:
                # Earlier blocks are needed only if some record still sounds at the first query.
                active = np.flatnonzero(
                    np.asarray(release_ticks[b, : start * block])
                    > int(query_ticks[b, j * block])
                )
                chosen = sorted(set(chosen).union(active // block))
            counts[b, 0, j] = len(chosen)
            indices[b, 0, j, : len(chosen)] = chosen
    return dict(
        left=torch.from_numpy(left),
        lengths=torch.as_tensor(lengths),
        strict=strict,
        **(
            {"release": release_ticks, "query": query_ticks}
            if release_ticks is not None
            else {}
        ),
        block=block,
        counts=torch.from_numpy(counts),
        indices=torch.from_numpy(indices),
    )


@lru_cache(maxsize=1)
def compiled_attention():
    from torch.nn.attention.flex_attention import flex_attention

    return torch.compile(flex_attention, dynamic=True)


class DenseHistory(NamedTuple):
    mask: torch.Tensor
    has_history: torch.Tensor


def materialize(layout, backend, dropout=0):
    left, lengths, strict = layout["left"], layout["lengths"], layout["strict"]
    n = left.shape[1]

    def visible(b, h, q, k):
        qsafe = q.clamp_max(left.shape[1] - 1)
        lower = (k >= left[b, qsafe]) | ((k == 0) & (not strict))
        if "release" in layout:
            lower = lower | (
                layout["release"][b, k.clamp_max(n - 1)] > layout["query"][b, qsafe]
            )
        return (q < lengths[b]) & (k <= q - int(strict)) & lower

    if (
        (left.is_cuda or backend == "flex")
        and backend != "sdpa"
        and layout["block"] % 128 == 0
        and not dropout
    ):
        from torch.nn.attention.flex_attention import BlockMask

        return BlockMask.from_kv_blocks(
            layout["counts"],
            layout["indices"],
            BLOCK_SIZE=layout["block"],
            mask_mod=visible,
            seq_lengths=(n, n),
        )
    q = torch.arange(n, device=left.device)
    b = torch.arange(len(lengths), device=left.device)[:, None, None]
    return visible(b, 0, q[None, :, None], q[None, None, :])[:, None]


def prepare_attention(layout):
    if not isinstance(layout, torch.Tensor):
        return layout
    has_history = layout.any(-1, keepdim=True)
    # SDPA needs a finite softmax on empty rows too. Preserve the visibility mask.
    mask = torch.cat((layout[..., :1] | ~has_history, layout[..., 1:]), -1)
    return DenseHistory(mask, has_history)


def bucket_attention(q, k, v, layout, dropout):
    layout = prepare_attention(layout)
    if not isinstance(layout, DenseHistory):
        return compiled_attention()(q, k, v, block_mask=layout)
    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=layout.mask, dropout_p=dropout
    )
    return out.masked_fill(~layout.has_history, 0)


class HistoryCache:
    """One chronological cursor for all layers of a single generated chart."""

    def __init__(self, layers, beats, *, sink=False):
        self.keep_sink = sink
        self.sinks = [None] * layers
        self.evicted = False
        self.buffers = [None] * layers
        self.beats = beats
        self.ticks = deque()
        self.releases = deque()
        self.step_count = 0
        self.protected = []
        self.start = self.stop = self.capacity = 0

    def begin(self, query_tick):
        self.lower = (query_tick // 96 - self.beats) * 96
        if self.protected:
            self.protected = [
                (release, kv) for release, kv in self.protected if release > query_tick
            ]
        while self.ticks and self.ticks[0] < self.lower:
            self.evicted = True
            self.ticks.popleft()
            release = self.releases.popleft()
            if release > query_tick:
                self.protected.append(
                    (
                        release,
                        [
                            buf[..., self.start : self.start + 1, :].clone()
                            for buf in self.buffers
                        ],
                    )
                )
            self.start += 1
        if not self.ticks:
            self.start = self.stop = 0
        if self.stop < self.capacity:
            return
        used = len(self.ticks)
        self.capacity = max(32, self.capacity, 2 * (used + 1))
        for i, buf in enumerate(self.buffers):
            if buf is not None:
                shape = (*buf.shape[:-2], self.capacity, buf.shape[-1])
                new = buf.new_empty(shape)
                new[..., :used, :].copy_(buf[..., self.start : self.stop, :])
                self.buffers[i] = new
        self.start, self.stop = 0, used

    def write(self, layer, k, v):
        if self.keep_sink and self.sinks[layer] is None:
            self.sinks[layer] = torch.stack((k, v))
        buf = self.buffers[layer]
        if buf is None:
            buf = k.new_empty(2, *k.shape[:2], self.capacity, k.shape[-1])
            self.buffers[layer] = buf
        buf[0, ..., self.stop : self.stop + 1, :].copy_(k)
        buf[1, ..., self.stop : self.stop + 1, :].copy_(v)

    def attend(self, layer, q, include_current=False):
        end = self.stop + int(include_current)
        keep_bos = self.keep_sink and self.evicted
        if end == self.start and not self.protected:
            return self.sinks[layer][1] if keep_bos else torch.zeros_like(q)
        kv = self.buffers[layer][..., self.start : end, :]
        if self.protected or keep_bos:
            prefix = [self.sinks[layer]] if keep_bos else []
            kv = torch.cat(
                [*prefix, *(record[layer] for _, record in self.protected), kv], -2
            )
        k, v = kv.unbind()
        return F.scaled_dot_product_attention(q, k, v)

    def finish(self, key_tick, release_tick=-1):
        self.step_count += 1
        if key_tick >= self.lower:
            self.ticks.append(key_tick)
            self.releases.append(release_tick)
            self.stop += 1


def sample_bank(bank, owners, seconds):
    frames, lengths = bank
    pos = torch.minimum((seconds * 125 - 0.5).clamp_min(0), lengths[owners] - 1)
    lo = pos.long()
    hi = torch.minimum(lo + 1, lengths[owners] - 1)
    # A single row index also keeps gradients for repeated frame reads on the
    # index_select accumulation path when many phases land between two frames.
    rows = frames.flatten(0, 1)
    start = owners * frames.shape[1]

    def gather(frame):
        return rows.index_select(0, (start + frame).reshape(-1)).reshape(
            *pos.shape, frames.shape[-1]
        )

    a = gather(lo)
    return torch.lerp(a, gather(hi), (pos - lo).to(a.dtype)[..., None])
