"""E0 micro-Conformer blocks."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from maidata.nn.position import apply_rope
from maidata.audio.geometry import mask_values
from maidata.audio.types import ScaleMemory


class MacaronFeedForward(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor, valid: Tensor, *, masked_input: bool = False) -> Tensor:
        values = self.norm(values if masked_input else mask_values(values, valid))
        values = self.dropout(F.silu(self.linear1(values)))
        return mask_values(self.dropout(self.linear2(values)), valid)


class TemporalConvolution(nn.Module):
    """Local depthwise-separable temporal convolution."""

    def __init__(self, d_model: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError("temporal kernel size must be positive and odd")
        self.norm = nn.LayerNorm(d_model)
        self.expand = nn.Linear(d_model, 2 * d_model)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size, padding=kernel_size // 2, groups=d_model
        )
        self.project = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor, valid: Tensor, *, masked_input: bool = False) -> Tensor:
        values = self.norm(values if masked_input else mask_values(values, valid))
        values = mask_values(F.glu(self.expand(values), dim=-1), valid)
        values = self.depthwise(values.transpose(1, 2)).transpose(1, 2)
        values = mask_values(F.silu(values), valid)
        return mask_values(self.dropout(self.project(values)), valid)


class SharedConformerBlock(nn.Module):
    """Pre-LN single-FFN micro-Conformer block for E0."""

    def __init__(
        self,
        d_model: int,
        heads: int,
        ffn_dim: int,
        temporal_kernel_size: int,
        attention_radius: float,
        dropout: float,
        rope_base: float,
        rope_position_scale: float,
    ) -> None:
        super().__init__()
        if heads < 1 or d_model < 1 or d_model % heads or (d_model // heads) % 2:
            raise ValueError("d_model must split into positive, even-sized attention heads")
        self.heads = heads
        self.attention_radius = attention_radius
        self.dropout_p = dropout
        self.rope_base = rope_base
        self.rope_position_scale = rope_position_scale
        self.attention_norm = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.convolution = TemporalConvolution(d_model, temporal_kernel_size, dropout)
        # Named ffn2 to preserve state_dict compatibility with v18/v11 checkpoints.
        self.ffn2 = MacaronFeedForward(d_model, ffn_dim, dropout)
        self.final_norm = nn.Identity()

    def _heads(self, values: Tensor) -> Tensor:
        return values.unflatten(-1, (self.heads, -1)).transpose(1, 2)

    def _rotated(self, values: Tensor, memory: ScaleMemory, tables=None, *, masked_input=False) -> Tensor:
        return apply_rope(
            self._heads(values if masked_input else mask_values(values, memory.valid)),
            memory.beats if tables is not None else memory.beats.masked_fill(~memory.valid, 0.0),
            self.rope_base,
            self.rope_position_scale,
            tables=tables,
        )

    def _projection_parameters(self):
        layers = (self.q, self.k, self.v)
        return (torch.cat([layer.weight for layer in layers], dim=0),
                torch.cat([layer.bias for layer in layers], dim=0))

    def _attention_residual(self, attended: Tensor, usable: Tensor) -> Tensor:
        values = attended.transpose(1, 2).flatten(-2)
        return mask_values(self.attention_dropout(self.out(values)), usable)

    def forward(self, local: ScaleMemory, local_tables=None, joint_layout=None) -> ScaleMemory:
        return self.forward_local(local, local_tables, joint_layout['local'])

    def forward_local(self, local: ScaleMemory, tables, layout) -> ScaleMemory:
        from .continuous_attention import joint_attention
        value = mask_values(local.values, local.valid)
        normalized = mask_values(self.attention_norm(value), local.valid)
        q, k, v = mask_values(F.linear(normalized, *self._projection_parameters()), local.valid).chunk(3, -1)
        attended = joint_attention(
            self._rotated(q, local, tables, masked_input=True),
            ((self._rotated(k, local, tables, masked_input=True), self._heads(v)),),
            layout,
            self.dropout_p if self.training else 0.0,
        )
        value = value + self._attention_residual(attended, local.valid)
        value = value + self.convolution(value, local.valid, masked_input=True)
        value = value + self.ffn2(value, local.valid, masked_input=True)
        return ScaleMemory(mask_values(self.final_norm(value), local.valid), local.beats, local.valid)
