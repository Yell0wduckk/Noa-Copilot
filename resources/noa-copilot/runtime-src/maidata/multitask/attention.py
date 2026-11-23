"""Shared RoPE and bounded, batched SDPA layouts.

Layouts are prepared once per forward and reused by every decoder layer.
The local encoder never allocates a song-length squared attention mask.
"""
from __future__ import annotations

from functools import lru_cache
from dataclasses import dataclass
from collections import defaultdict

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@lru_cache(maxsize=64)
def rotary_frequency(device, dims, base):
    return base ** (-torch.arange(0, dims, 2, device=device, dtype=torch.float32) / dims)


def rotary(x: Tensor, positions: Tensor, base: float = 10000.0) -> Tensor:
    """x [B,H,T,D], positions [B,T] or [T], interleaved rotary pairs."""
    dims = x.shape[-1]
    if not dims:
        return x
    freq = rotary_frequency(x.device, dims, base)
    angles = positions.float().unsqueeze(-1) * freq
    if angles.ndim == 2:
        angles = angles.unsqueeze(0)
    cos, sin = angles.cos().unsqueeze(1).to(x.dtype), angles.sin().unsqueeze(1).to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(-2)


def dual_rotary(x: Tensor, times: Tensor, steps: Tensor, step_dims: int) -> Tensor:
    split = x.shape[-1] - step_dims
    return torch.cat((rotary(x[..., :split], times),
                      rotary(x[..., split:], steps, 256.0)), dim=-1)


@dataclass
class LayoutGroup:
    batch: Tensor
    queries: Tensor
    keys: Tensor
    allowed: Tensor
    query_valid: Tensor
    take_output: Tensor
    write_output: Tensor


@dataclass
class SparseLayout:
    groups: list[LayoutGroup]
    query_length: int
    query_has_keys: Tensor

    def to(self, device):
        """One integer and one boolean transfer for the whole ragged layout."""
        if self.query_has_keys.device == torch.device(device):
            return self
        integer, boolean, specs = [], [self.query_has_keys], []
        for group in self.groups:
            values = [group.batch, group.queries, group.keys, group.take_output, group.write_output]
            specs.append(([v.shape for v in values], group.allowed.shape, group.query_valid.shape))
            integer.extend(values)
            boolean.extend((group.allowed, group.query_valid))
        longs = torch.cat([v.reshape(-1) for v in integer]).to(device, non_blocking=True) if integer else None
        bools = torch.cat([v.reshape(-1) for v in boolean]).to(device, non_blocking=True)
        offset_i, offset_b = 0, self.query_has_keys.numel()
        has_keys = bools[:offset_b].view_as(self.query_has_keys)
        result = []
        for shapes, allowed_shape, valid_shape in specs:
            values = []
            for shape in shapes:
                count = shape.numel()
                values.append(longs[offset_i:offset_i + count].view(shape))
                offset_i += count
            masks = []
            for shape in (allowed_shape, valid_shape):
                count = shape.numel()
                masks.append(bools[offset_b:offset_b + count].view(shape))
                offset_b += count
            result.append(LayoutGroup(*values[:3], *masks, *values[3:]))
        return SparseLayout(result, self.query_length, has_keys)


def _pack_blocks(blocks, query_length: int, device) -> SparseLayout:
    """Group similarly sized windows to batch GEMMs without global padding."""
    groups = defaultdict(list)
    batch_size = max((item[0] for item in blocks), default=-1) + 1
    has_keys = torch.zeros((batch_size, query_length), dtype=torch.bool)
    for batch, queries, keys, allowed in blocks:
        has_keys[batch, queries] = allowed.any(-1)
        if not len(keys):
            continue
        groups[((len(keys) + 63) // 64) * 64].append((batch, queries, keys, allowed))
    packed = []
    for key_size, members in groups.items():
        # Limit gathered working sets, not recomputation. Q/K/V projections are
        # made once and each query is evaluated exactly once.
        # Every equal-width bucket is one batched attention call. Do not split
        # a logical batch into serial 32-block microbatches by default.
        for group in (members,):
            qsize = max(len(m[1]) for m in group)
            bs = torch.tensor([m[0] for m in group], dtype=torch.long)
            qs = torch.zeros((len(group), qsize), dtype=torch.long)
            ks = torch.zeros((len(group), key_size), dtype=torch.long)
            mask = torch.zeros((len(group), qsize, key_size), dtype=torch.bool)
            valid = torch.zeros((len(group), qsize), dtype=torch.bool)
            for n, (_, qi, ki, allow) in enumerate(group):
                qs[n, :len(qi)] = qi
                ks[n, :len(ki)] = ki
                mask[n, :len(qi), :len(ki)] = allow
                valid[n, :len(qi)] = True
            take = torch.nonzero(valid.reshape(-1), as_tuple=False).flatten()
            write = (bs[:, None] * query_length + qs).reshape(-1).index_select(0, take)
            packed.append(LayoutGroup(bs, qs, ks, mask, valid, take, write))
    return SparseLayout(packed, query_length, has_keys).to(device)


def regular_window_layout(batch: int, frames: int, lengths: Tensor, left=128,
                          right=128, query_block=64, device=None) -> SparseLayout:
    """Vectorized regular acoustic grid; no searchsorted/CPU roundtrip."""
    device = lengths.device if device is None else device
    blocks = (frames + query_block - 1) // query_block
    owners = torch.arange(batch, device=device).repeat_interleave(blocks)
    start = torch.arange(blocks, device=device).repeat(batch) * query_block
    qr = start[:, None] + torch.arange(query_block, device=device)
    kr = start[:, None] - left + torch.arange(query_block + left + right - 1, device=device)
    qvalid = qr < frames
    allowed = ((kr[:, None] >= qr[:, :, None] - left) &
               (kr[:, None] < qr[:, :, None] + right) &
               (kr[:, None] >= 0) & (kr[:, None] < lengths[owners, None, None]) &
               (qr[:, :, None] < lengths[owners, None, None]))
    take = torch.arange(batch * blocks * query_block, device=device).view(batch, -1)[:, :frames].reshape(-1)
    write = torch.arange(batch * frames, device=device)
    group = LayoutGroup(owners, qr.clamp_max(frames - 1), kr.clamp(0, frames - 1),
                        allowed, qvalid, take, write)
    has_keys = torch.arange(frames, device=device)[None] < lengths[:, None]
    return SparseLayout([group], frames, has_keys)


def window_layout(query_times: Tensor, key_times: Tensor, query_lengths: Tensor,
                  key_lengths: Tensor, left: float, right: float,
                  query_block: int = 64) -> SparseLayout:
    """Half-open [query-left, query+right), native 125 Hz frame coordinates."""
    qcpu, kcpu = query_times.detach().cpu(), key_times.detach().cpu()
    qlens, klens = query_lengths.tolist(), key_lengths.tolist()
    blocks = []
    for b, (qlen, klen) in enumerate(zip(qlens, klens)):
        keys_all = kcpu[b, :klen]
        # Endpoint queries can jump many measures. Sort only the audio reads;
        # scatter their outputs back to token order. Skip gaps between disjoint
        # windows so that a distant END never gathers the intervening song.
        order = torch.argsort(qcpu[b, :qlen], stable=True)
        qt_sorted = qcpu[b, order].contiguous()
        starts = torch.searchsorted(keys_all, qt_sorted-left)
        ends = torch.searchsorted(keys_all, qt_sorted+right)
        for start in range(0, qlen, query_block):
            qi = order[start:start + query_block]
            qt = qt_sorted[start:start + query_block]
            lo, hi = starts[start:start+query_block], ends[start:start+query_block]
            breaks = torch.where(lo[1:] > hi[:-1])[0].tolist()
            bounds = [0, *(i+1 for i in breaks), len(qi)]
            ki = torch.cat([torch.arange(int(lo[i]), int(hi[j-1])) for i,j in zip(bounds,bounds[1:])])
            allowed = ((keys_all[ki][None] >= qt[:, None] - left) &
                       (keys_all[ki][None] < qt[:, None] + right))
            blocks.append((b, qi, ki, allowed))
    result = _pack_blocks(blocks, query_times.shape[1], query_times.device)
    if result.query_has_keys.shape[0] < len(query_lengths):
        result.query_has_keys = F.pad(result.query_has_keys, (0, 0, 0, len(query_lengths) - result.query_has_keys.shape[0]))
    return result


def self_layout(causal_times: Tensor, moment_ids: Tensor, lengths: Tensor,
                history_seconds: Tensor, no_history: Tensor,
                protected_spans=None, query_block: int = 64,
                protection_enabled=None, query_rows=None) -> SparseLayout:
    """Causal seconds-prefix truncation + sink + active-note protection.

    Protection rows are half-open token-index spans [key0,key1,query0,query1].
    The codec derives them from prefix state, including the MIDI 32-note cap.
    """
    times, moments = causal_times.detach().cpu(), moment_ids.detach().cpu()
    history, nohist = history_seconds.cpu().tolist(), no_history.cpu().tolist()
    spans_by_chart = protected_spans if protected_spans is not None else [[] for _ in lengths]
    enabled = torch.ones_like(times, dtype=torch.bool) if protection_enabled is None else protection_enabled.cpu()
    output_length = causal_times.shape[1] if query_rows is None else max((len(q) for q in query_rows), default=0)
    blocks = []
    for b, qlen in enumerate(lengths.tolist()):
        allkeys = torch.arange(qlen)
        spans = torch.as_tensor(spans_by_chart[b], dtype=torch.long).reshape(-1, 4).cpu()
        rows = torch.arange(qlen) if query_rows is None else torch.as_tensor(query_rows[b])
        for start in range(0, len(rows), query_block):
            qi = rows[start:start + query_block]
            # Determine the union before constructing a mask, so work scales
            # with a local history window rather than the complete song.
            if nohist[b]:
                visible = torch.isin(moments[b, :qlen], moments[b, qi]) & (allkeys <= qi[-1])
            else:
                cutoff = torch.minimum(times[b, qi].min() - history[b],
                                       times[b, qi].min().floor())
                visible = (times[b, :qlen] >= cutoff) & (allkeys <= qi[-1])
            visible[0] = True
            relevant = spans[(spans[:, 2] <= qi[-1]) & (spans[:, 3] > qi[0])]
            if not bool(enabled[b, qi].any()):
                relevant = relevant[:0]
            for key0, key1, _, _ in relevant.tolist():
                visible[max(0, key0):min(qlen, key1)] = True
            ki = allkeys[visible]
            if nohist[b]:
                allowed = moments[b, qi, None] == moments[b, ki][None]
            else:
                cutoff = torch.minimum(times[b, qi] - history[b],
                                       times[b, qi].floor())
                allowed = times[b, ki][None] >= cutoff[:, None]
            allowed |= ki[None] == 0
            for key0, key1, query0, query1 in relevant.tolist():
                allowed |= ((qi[:, None] >= query0) & (qi[:, None] < query1) &
                            (ki[None] >= key0) & (ki[None] < key1) & enabled[b, qi, None])
            allowed &= ki[None] <= qi[:, None]
            output_queries = qi if query_rows is None else torch.arange(start, start + len(qi))
            blocks.append((b, output_queries, ki, allowed))
    # Empty auxiliary rows still retain the full chart batch dimension.
    result = _pack_blocks(blocks, output_length, causal_times.device)
    if result.query_has_keys.shape[0] != len(lengths):
        result.query_has_keys = F.pad(result.query_has_keys, (0, 0, 0, len(lengths) - result.query_has_keys.shape[0]))
    return result


def sparse_sdpa(q: Tensor, k: Tensor, v: Tensor, layout: SparseLayout, memory_indices=None) -> Tensor:
    """Projected [B,H,T,D] inputs; no projection or layer recomputation."""
    batch, heads, _, width = q.shape
    result = q.new_zeros((batch * layout.query_length, heads, width))
    for group in layout.groups:
        qb = q[group.batch[:, None], :, group.queries].transpose(1, 2)
        owners = group.batch if memory_indices is None else memory_indices[group.batch]
        kb = k[owners[:, None], :, group.keys].transpose(1, 2)
        vb = v[owners[:, None], :, group.keys].transpose(1, 2)
        out = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=group.allowed[:, None])
        out = out.transpose(1, 2).reshape(-1, heads, width).index_select(0, group.take_output)
        result = result.index_copy(0, group.write_output, out)
    return result.view(batch, layout.query_length, heads, width).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, step_dims: int = 0):
        super().__init__()
        self.heads, self.head_dim, self.step_dims = heads, dim // heads, step_dims
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def project_query(self, x: Tensor, times: Tensor, steps: Tensor | None = None):
        q = self.q(x).view(*x.shape[:2], self.heads, self.head_dim).transpose(1, 2)
        return dual_rotary(q, times, steps, self.step_dims) if self.step_dims else rotary(q, times)

    def project_memory(self, x: Tensor, times: Tensor, steps: Tensor | None = None):
        k = self.k(x).view(*x.shape[:2], self.heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(*x.shape[:2], self.heads, self.head_dim).transpose(1, 2)
        k = dual_rotary(k, times, steps, self.step_dims) if self.step_dims else rotary(k, times)
        return k, v

    def finish(self, x: Tensor):
        return self.out(x.transpose(1, 2).flatten(-2))

    def forward(self, x: Tensor, memory: Tensor, query_times: Tensor, memory_times: Tensor,
                layout: SparseLayout | None = None, *, query_steps=None, memory_steps=None,
                memory_valid=None, memory_kv=None, memory_indices=None):
        q = self.project_query(x, query_times, query_steps)
        k, v = self.project_memory(memory, memory_times, memory_steps) if memory_kv is None else memory_kv
        if layout is None:
            mask = memory_valid[:, None, None, :] if memory_valid is not None else None
            result = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            result = sparse_sdpa(q, k, v, layout, memory_indices)
        result = self.finish(result)
        return result if layout is None else result * layout.query_has_keys[..., None]
