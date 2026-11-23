"""Local 8-fps P0 Conformer blocks and FP32 parameter-free output LayerNorm."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from maidata.nn.position import apply_rope
from maidata.nn.varlen_attention import can_use_varlen_flash, varlen_flash_attention

from .blocks import MacaronFeedForward, TemporalConvolution
from .geometry import mask_values
from .types import ScaleMemory


class P0OutputNorm(nn.LayerNorm):
    """Normalize in FP32, preserving the representation's storage dtype."""

    def __init__(self, dim, eps=1e-5):
        super().__init__(dim, eps=eps, elementwise_affine=False)

    def forward(self, values):
        with torch.autocast(device_type=values.device.type, enabled=False):
            normalized = F.layer_norm(values.float(), self.normalized_shape,
                                      None, None, self.eps)
        return normalized.to(values.dtype)


class P0SelfAttention(nn.Module):
    def __init__(self, dim, heads, dropout, rope_base, rope_position_scale):
        super().__init__()
        if heads < 1 or dim < 1 or dim % heads or (dim // heads) % 2:
            raise ValueError('P0 attention requires positive, even-sized attention heads')
        self.heads, self.head_dim = heads, dim // heads
        self.dropout_p = dropout
        self.rope_base, self.rope_position_scale = rope_base, rope_position_scale
        self.qkv = nn.Linear(dim, 3 * dim)
        self.output = nn.Linear(dim, dim)

    def forward(self, values, beats, valid, *, tables, attention, masked_input=False):
        projected = mask_values(self.qkv(values if masked_input else mask_values(values, valid)), valid)
        query, key, content = (part.unflatten(-1, (self.heads, self.head_dim)).transpose(1, 2)
                               for part in projected.chunk(3, -1))
        query = apply_rope(query, beats, self.rope_base, self.rope_position_scale, tables=tables)
        key = apply_rope(key, beats, self.rope_base, self.rope_position_scale, tables=tables)
        dropout_p = self.dropout_p if self.training else 0.0
        if 'local' in attention:
            from .continuous_attention import joint_attention
            attended = joint_attention(query, ((key, content),), attention['local'], dropout_p)
            return mask_values(self.output(attended.transpose(1, 2).flatten(-2)), valid)
        if (not attention['all_valid'] and attention['max_length'] > 0
                and can_use_varlen_flash(query[:, :, 0], key[:, :, 0], content[:, :, 0],
                                         dropout_p=dropout_p)):
            indices = attention['gather']
            packed = [part.transpose(1, 2).flatten(0, 1).index_select(0, indices)
                      for part in (query, key, content)]
            attended = varlen_flash_attention(*packed,
                attention['cu_seqlens'], attention['cu_seqlens'],
                attention['max_length'], attention['max_length'],
                dropout_p=dropout_p, is_causal=False, scale=1/math.sqrt(self.head_dim))
            flat = attended.new_zeros((values.shape[0]*values.shape[1], self.heads, self.head_dim))
            hidden = flat.index_copy(0, indices, attended).unflatten(0, values.shape[:2])
            hidden = hidden.flatten(-2)
        elif values.shape[1]:
            attended = F.scaled_dot_product_attention(query, key, content,
                attn_mask=attention['mask'], dropout_p=dropout_p)
            hidden = attended.transpose(1, 2).flatten(-2)
        else:
            hidden = content.transpose(1, 2).flatten(-2)
        return mask_values(self.output(mask_values(hidden, valid)), valid)


class P0ConformerBlock(nn.Module):
    """Pre-LN local attention, temporal convolution, and residual FFN."""

    def __init__(self, dim, heads, ffn_dim, kernel_size, dropout,
                 rope_base, rope_position_scale):
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = P0SelfAttention(dim, heads, dropout, rope_base,
                                         rope_position_scale)
        self.attention_dropout = nn.Dropout(dropout)
        self.convolution = TemporalConvolution(dim, kernel_size, dropout)
        self.ffn = MacaronFeedForward(dim, ffn_dim, dropout)

    def forward(self, memory, *, tables, attention):
        value = mask_values(memory.values, memory.valid)
        normalized = mask_values(self.attention_norm(value), memory.valid)
        hidden = self.attention(normalized, memory.beats, memory.valid,
                                tables=tables, attention=attention, masked_input=True)
        value = value + self.attention_dropout(hidden)
        value = value + self.convolution(value, memory.valid, masked_input=True)
        value = value + self.ffn(value, memory.valid, masked_input=True)
        return ScaleMemory(value, memory.beats, memory.valid)
