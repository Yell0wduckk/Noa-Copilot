"""Dense consumer-owned projections, optional beat RoPE, safe empty masks."""

import torch
from torch import nn
from torch.nn import functional as F

from maidata.nn.position import apply_rope


def safe_mask(mask):
    active = mask.any(-1, keepdim=True)
    first = torch.arange(mask.size(-1), device=mask.device) == 0
    return torch.where(active, mask, first), active


class Attention(nn.Module):
    def __init__(
        self,
        query_dim,
        memory_dim,
        width,
        heads,
        dropout,
        *,
        rotary,
        base,
        scale,
        qk_normalization=False,
        output_bias=True,
        fused_self=False,
    ):
        super().__init__()
        self.width, self.heads, self.dropout = width, heads, dropout
        self.rotary, self.base, self.scale = rotary, base, scale
        self.qk_normalization = qk_normalization
        if fused_self:
            self.qkv = nn.Linear(query_dim, 3 * width)
        else:
            self.query = nn.Linear(query_dim, width)
            self.key = nn.Linear(memory_dim, width)
            self.value = nn.Linear(memory_dim, width)
        self.output = nn.Linear(width, query_dim, bias=output_bias)

    def split(self, x):
        return x.reshape(*x.shape[:2], self.heads, self.width // self.heads).transpose(
            1, 2
        )

    def normalize_qk(self, value):
        if not self.qk_normalization:
            return value
        return F.rms_norm(value.float(), (value.size(-1),), eps=1e-6).to(value.dtype)

    def project_query(self, values):
        """Return normalized, unrotated heads for ordinary and pooled queries."""
        return self.normalize_qk(self.split(self.query(values)))

    def project_memory(self, values, beats, valid, tables=None):
        values = values.masked_fill(~valid[..., None], 0)
        k, v = (
            self.normalize_qk(self.split(self.key(values))),
            self.split(self.value(values)),
        )
        if self.rotary:
            k = apply_rope(
                k, beats.masked_fill(~valid, 0), self.base, self.scale, tables=tables
            )
        return k, v

    def forward(self, query, beats, valid, kv, memory_valid, tables=None, key_mask=None):
        values = query.masked_fill(~valid[..., None], 0)
        if kv is None:
            q, k, v = self.qkv(values).chunk(3, -1)
            q, k = (self.normalize_qk(self.split(t)) for t in (q, k))
            kv = (
                apply_rope(k, beats, self.base, self.scale, tables=tables),
                self.split(v),
            )
        else:
            q = self.project_query(values)
        if self.rotary:
            q = apply_rope(
                q, beats.masked_fill(~valid, 0), self.base, self.scale, tables=tables
            )
        if key_mask is None:
            key_mask = safe_mask(memory_valid[:, None, None, :])
        allowed, key_active = key_mask
        active = valid[:, None, :, None] & key_active
        h = F.scaled_dot_product_attention(
            q, *kv, attn_mask=allowed, dropout_p=self.dropout if self.training else 0.0
        )
        h = (
            h.masked_fill(~active, 0)
            .transpose(1, 2)
            .reshape(*query.shape[:2], self.width)
        )
        active = active[:, 0, :, 0]
        return self.output(h).masked_fill(~active[..., None], 0)
