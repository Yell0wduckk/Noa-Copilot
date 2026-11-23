"""Consumer-owned projections and immutable K/V; no producer model imports."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from maidata.nn.position import apply_rope
from maidata.nn.varlen_attention import can_use_varlen_flash, varlen_flash_attention
from .types import KV
from .projections import linear_parameters


class Attention(nn.Module):
    def __init__(self, dim, memory_dim, heads, dropout, base, scale):
        super().__init__()
        self.dim, self.heads, self.dropout = dim, heads, dropout
        self.base, self.scale = base, scale
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(memory_dim, dim)
        self.value = nn.Linear(memory_dim, dim)
        self.output = nn.Linear(dim, dim)

    @property
    def q_proj(self):
        return self.query

    @property
    def k_proj(self):
        return self.key

    @property
    def v_proj(self):
        return self.value

    @property
    def out_proj(self):
        return self.output

    def split(self, x):
        return x.reshape(*x.shape[:2], self.heads, self.dim // self.heads).transpose(1, 2)

    def project_query(self, values):
        projected = self.query(values)
        if projected.ndim == 2:
            projected = projected[None]
        return self.split(projected)

    def project(self, memory, *, indices=None, tables=None, compact=False):
        if indices is not None:
            b, length = memory.valid.shape
            full = len(indices) == memory.valid.numel()
            x = memory.values.flatten(0, 1)
            beats = memory.beats.flatten()
            x = (x if full else x.index_select(0, indices))[None]
            beats = (beats if full else beats.index_select(0, indices))[None]
            raw = self.split(self.key(x))
            key = apply_rope(raw, beats, self.base, self.scale, tables=tables)
            value = self.split(self.value(x))
            flat_key, flat_value = key[0].transpose(0, 1), value[0].transpose(0, 1)
            if compact and can_use_varlen_flash(flat_key, flat_key, flat_value,
                    dropout_p=self.dropout if self.training else 0.):
                return KV(flat_key, flat_value, memory.valid, compact_indices=indices)
            def restore(t):
                flat = t[0].transpose(0, 1)
                if full:
                    return flat.reshape(b, length, self.heads, -1).transpose(1, 2)
                return flat.new_zeros((b*length, self.heads, self.dim//self.heads)).index_copy(
                    0, indices, flat).reshape(b, length, self.heads, -1).transpose(1, 2)
            return KV(restore(key), restore(value), memory.valid)
        x = memory.values.masked_fill(~memory.valid[..., None], 0)
        raw = self.split(self.key(x))
        k = apply_rope(raw, memory.beats.masked_fill(~memory.valid, 0), self.base, self.scale, tables=tables)
        return KV(k, self.split(self.value(x)), memory.valid)

    def forward(self, x, beats, valid, kv, *, causal_offset=None, tables=None):
        raw = self.project_query(x.masked_fill(~valid[..., None], 0))
        return self.attend(raw, beats, valid, kv, causal_offset=causal_offset, tables=tables)

    def attend(self, raw, beats, valid, kv, *, causal_offset=None, tables=None):
        kv = kv.padded()
        q = apply_rope(raw, beats.masked_fill(~valid, 0), self.base, self.scale, tables=tables)
        if kv.query_layout is not None:
            return self.mapped(q, valid, kv)
        allowed = valid[:, None, :, None] & kv.valid[:, None, None, :]
        if causal_offset is not None:
            causal = (torch.arange(kv.key.shape[2], device=raw.device)[None, :]
                      <= causal_offset + torch.arange(valid.shape[1], device=raw.device)[:, None])
            allowed = allowed & causal[None, None]
        has_visible = allowed.any(-1, keepdim=True)
        if kv.key.shape[2] == 0:
            y = kv.value.sum(2, keepdim=True).expand(-1, -1, q.shape[2], -1)
        else:
            safe = allowed.clone()
            safe[..., 0] |= ~has_visible.squeeze(-1)
            y = F.scaled_dot_product_attention(q, kv.key, kv.value, attn_mask=safe,
                                               dropout_p=self.dropout if self.training else 0.)
        y = y.masked_fill(~has_visible, 0)
        y = y.transpose(1, 2).reshape(*valid.shape, self.dim)
        return self.output(y).masked_fill(~has_visible.squeeze(1), 0)

    def mapped(self, q, valid, kv, *, packed=False):
        layout = kv.query_layout
        flat = q.transpose(1, 2).flatten(0, 1)
        from .packed_attention import valid_layout
        if (packed and layout.get('varlen_full_keys', False) and valid_layout(kv, layout)
                and len(flat) and can_use_varlen_flash(flat, kv.key[0].transpose(0, 1),
                    kv.value[0].transpose(0, 1), dropout_p=self.dropout if self.training else 0.)):
            query = flat.index_select(0, layout['varlen_q_gather'])
            key, value = kv.key, kv.value
            if not layout['varlen_all_songs']:
                key, value = (t.index_select(0, layout['varlen_songs']) for t in (key, value))
            key, value = (t.transpose(1, 2).flatten(0, 1) for t in (key, value))
            attended = varlen_flash_attention(query, key, value,
                layout['varlen_cu_seqlens_q'], layout['varlen_cu_seqlens_k'],
                layout['varlen_max_q'], layout['varlen_max_k'],
                dropout_p=self.dropout if self.training else 0., is_causal=False,
                scale=1/math.sqrt(self.dim//self.heads))
            output = attended.index_select(0, layout['varlen_q_restore'])
            return self.output(output.reshape(len(flat), self.dim))
        sizes = [g[0] for g in layout['groups']]
        queries = flat.index_select(0, layout['compact_gather'] if packed else layout['gather']).split(
            [g[0]*g[1] for g in layout['groups']])
        keys = kv.key.index_select(0, layout['songs']).split(sizes)
        values = kv.value.index_select(0, layout['songs']).split(sizes)
        masks = kv.valid.index_select(0, layout['songs']).split(sizes)
        pieces = []
        for query, key, value, mask, (count, width, lo, hi) in zip(queries, keys, values, masks, layout['groups']):
            hi = kv.valid.shape[1] if hi is None else hi
            query = query.reshape(count, width, self.heads, -1).transpose(1, 2)
            result = F.scaled_dot_product_attention(query, key.narrow(2, lo, hi-lo), value.narrow(2, lo, hi-lo),
                attn_mask=mask[:, None, None, lo:hi], dropout_p=self.dropout if self.training else 0.)
            pieces.append(result.transpose(1, 2).reshape(-1, self.heads, self.dim//self.heads))
        selected = torch.cat(pieces).index_select(0, layout['take']) if pieces else flat[:0]
        if packed and 'compact_restore' in layout:
            output = selected.index_select(0, layout['compact_restore'])
        else:
            output = torch.zeros_like(flat).index_copy(0, layout['compact_put'] if packed else layout['put'], selected)
        y = output.reshape(flat.shape[0], self.dim) if packed else output.reshape(*valid.shape, self.dim)
        active = valid & kv.valid.any(-1)[layout['mapping']][:, None]
        if packed:
            active = active.flatten().index_select(0, layout['valid_indices'])
        return self.output(y).masked_fill(~active[..., None], 0)

    def self_attention(self, x, beats, valid, previous, *, tables=None, need_cache=True, cache=None):
        layers = (self.query, self.key, self.value)
        parts = F.linear(x.masked_fill(~valid[..., None], 0), *linear_parameters(layers, cache)).chunk(3, -1)
        raw, key, value = map(self.split, parts)
        current = KV(apply_rope(key, beats.masked_fill(~valid, 0), self.base, self.scale, tables=tables), value, valid)
        offset = 0 if previous is None else previous.valid.shape[1]
        kv = current if previous is None else KV(torch.cat((previous.key, current.key), 2),
                    torch.cat((previous.value, current.value), 2), torch.cat((previous.valid, valid), 1))
        return self.attend(raw, beats, valid, kv, causal_offset=offset, tables=tables), kv if need_cache else None
