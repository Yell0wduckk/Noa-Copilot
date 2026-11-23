"""Trainable lightweight fused audio encoder for YMT3+ (8ms @ 512d) + MuQ Layer 10 (40ms @ 1024d)."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from maidata.nn.position import apply_rope, rope_tables
from maidata.ymt3_muq_contract import FRAME_SECONDS_8MS, MUQ_DIM, RATIO_40_TO_8, YMT3_DIM
from .geometry import mask_values
from .p0_conformer import P0OutputNorm
from .types import ContinuousSongEncoding, ScaleMemory


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor, valid: Tensor) -> Tensor:
        values = self.dropout(F.silu(self.linear1(values)))
        return mask_values(self.dropout(self.linear2(values)), valid)


class LocalSlidingWindowAttention(nn.Module):
    """Chunked block-local SDPA at 8ms without allocating an O(T^2) attention matrix."""

    def __init__(
        self,
        d_model: int,
        heads: int,
        window_frames: int = 128,
        q_block: int = 128,
        dropout: float = 0.1,
        rope_base: float = 2048.0,
        rope_position_scale: float = 3.14,
    ) -> None:
        super().__init__()
        if heads < 1 or d_model < 1 or d_model % heads or (d_model // heads) % 2:
            raise ValueError("d_model must split into positive, even-sized attention heads")
        if window_frames < 1 or q_block < 1:
            raise ValueError("window_frames and q_block must be positive integers")
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads
        self.window_frames = int(window_frames)
        self.q_block = int(q_block)
        self.dropout_p = float(dropout)
        self.rope_base = float(rope_base)
        self.rope_position_scale = float(rope_position_scale)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: Tensor) -> Tensor:
        return x.unflatten(-1, (self.heads, self.head_dim)).transpose(1, 2)

    def _qkv_parameters(self) -> tuple[Tensor, Tensor]:
        layers = (self.q_proj, self.k_proj, self.v_proj)
        return (
            torch.cat([layer.weight for layer in layers], dim=0),
            torch.cat([layer.bias for layer in layers], dim=0),
        )

    def forward(self, x: Tensor, valid: Tensor, tables: tuple[Tensor, Tensor]) -> Tensor:
        batch, length, _ = x.shape
        if length == 0:
            return x * 0.0
        q, k, v = mask_values(F.linear(x, *self._qkv_parameters()), valid).chunk(3, dim=-1)
        pos_dummy = x.new_zeros((batch, length))
        q = apply_rope(self._split_heads(q), pos_dummy, self.rope_base, self.rope_position_scale, tables=tables)
        k = apply_rope(self._split_heads(k), pos_dummy, self.rope_base, self.rope_position_scale, tables=tables)
        v = self._split_heads(v)

        qb = self.q_block
        radius = self.window_frames
        kb = qb + 2 * radius
        pad_q = (-length) % qb
        padded_len = length + pad_q
        num_blocks = padded_len // qb

        q_pad = F.pad(q, (0, 0, 0, pad_q))
        q_blk = (
            q_pad.reshape(batch, self.heads, num_blocks, qb, self.head_dim)
            .permute(0, 2, 1, 3, 4)
            .reshape(batch * num_blocks, self.heads, qb, self.head_dim)
        )

        k_pad = F.pad(k, (0, 0, radius, pad_q + radius))
        v_pad = F.pad(v, (0, 0, radius, pad_q + radius))
        k_blk = (
            k_pad.unfold(2, kb, qb)
            .permute(0, 2, 1, 4, 3)
            .reshape(batch * num_blocks, self.heads, kb, self.head_dim)
        )
        v_blk = (
            v_pad.unfold(2, kb, qb)
            .permute(0, 2, 1, 4, 3)
            .reshape(batch * num_blocks, self.heads, kb, self.head_dim)
        )

        q_valid = F.pad(valid, (0, pad_q), value=False).reshape(batch, num_blocks, qb)
        k_valid = F.pad(valid, (radius, pad_q + radius), value=False).unfold(1, kb, qb)
        q_rel = torch.arange(qb, device=x.device)[:, None]
        k_rel = torch.arange(kb, device=x.device)[None, :] - radius
        rel_mask = (q_rel - k_rel).abs() <= radius

        mask = (
            q_valid[:, :, :, None]
            & k_valid[:, :, None, :]
            & rel_mask[None, None, :, :]
        ).reshape(batch * num_blocks, 1, qb, kb)

        has_visible = mask.any(-1, keepdim=True)
        safe = mask.clone()
        safe[..., 0] |= ~has_visible.squeeze(-1)

        attended = F.scaled_dot_product_attention(
            q_blk,
            k_blk,
            v_blk,
            attn_mask=safe,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        attended = attended.masked_fill(~has_visible, 0.0)
        attended = (
            attended.reshape(batch, num_blocks, self.heads, qb, self.head_dim)
            .permute(0, 2, 1, 3, 4)
            .reshape(batch, self.heads, padded_len, self.head_dim)[:, :, :length, :]
            .transpose(1, 2)
            .flatten(-2)
        )
        return mask_values(self.dropout(self.out_proj(attended)), valid)


class YMT3MuQBlock(nn.Module):
    """Pre-LN Sliding-Window Transformer block at 8ms."""

    def __init__(
        self,
        d_model: int,
        heads: int,
        ffn_dim: int,
        window_frames: int = 128,
        dropout: float = 0.1,
        rope_base: float = 2048.0,
        rope_position_scale: float = 3.14,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = LocalSlidingWindowAttention(
            d_model=d_model,
            heads=heads,
            window_frames=window_frames,
            q_block=min(128, max(1, window_frames)),
            dropout=dropout,
            rope_base=rope_base,
            rope_position_scale=rope_position_scale,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_dim, dropout)

    def forward(self, x: Tensor, valid: Tensor, tables: tuple[Tensor, Tensor]) -> Tensor:
        x = mask_values(x, valid)
        normed = mask_values(self.norm1(x), valid)
        x = x + self.attn(normed, valid, tables)
        x = x + self.ffn(mask_values(self.norm2(x), valid), valid)
        return mask_values(x, valid)


class YMT3MuQEncoder(nn.Module):
    """Lightweight fused YMT3+ (8ms) + MuQ Layer 10 (40ms -> 8ms) encoder."""

    def __init__(
        self,
        *,
        d_model: int = 256,
        heads: int = 8,
        ffn_dim: int = 1024,
        encoder_layers: int = 2,
        encoder_window_frames: int = 128,
        ymt3_dim: int = YMT3_DIM,
        muq_dim: int = MUQ_DIM,
        dropout: float = 0.1,
        rope_base: float = 2048.0,
        rope_position_scale: float = 3.14,
        **unused,
    ) -> None:
        super().__init__()
        if d_model % heads or (d_model // heads) % 2:
            raise ValueError("d_model / heads must be an even integer for full-head RoPE")
        if type(encoder_layers) is not int or encoder_layers <= 0:
            raise ValueError("encoder_layers must be a positive integer")
        if type(encoder_window_frames) is not int or encoder_window_frames <= 0:
            raise ValueError("encoder_window_frames must be a positive integer")

        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads
        self.ffn_dim = ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_window_frames = encoder_window_frames
        self.ymt3_dim = ymt3_dim
        self.muq_dim = muq_dim
        self.rope_base = float(rope_base)
        self.rope_position_scale = float(rope_position_scale)

        self.ymt3_muq = True
        self.local_dual_stream = False
        self.global_audio = False

        self.ymt3_norm = nn.LayerNorm(ymt3_dim, elementwise_affine=False, eps=1e-5)
        self.ymt3_proj = nn.Linear(ymt3_dim, d_model)
        self.muq_norm = nn.LayerNorm(muq_dim, elementwise_affine=False, eps=1e-5)
        self.muq_proj = nn.Linear(muq_dim, d_model)

        self.layers = nn.ModuleList(
            YMT3MuQBlock(
                d_model=d_model,
                heads=heads,
                ffn_dim=ffn_dim,
                window_frames=encoder_window_frames,
                dropout=dropout,
                rope_base=rope_base,
                rope_position_scale=rope_position_scale,
            )
            for _ in range(encoder_layers)
        )
        self.output_norm = P0OutputNorm(d_model)
        self.reset_projection_parameters()

    def reset_projection_parameters(self) -> None:
        nn.init.orthogonal_(self.ymt3_proj.weight, gain=1.0)
        nn.init.zeros_(self.ymt3_proj.bias)
        nn.init.orthogonal_(self.muq_proj.weight, gain=1.0)
        nn.init.zeros_(self.muq_proj.bias)

    def encode_song(self, inputs) -> ContinuousSongEncoding:
        if (
            inputs.ymt3_frames is None
            or inputs.muq_frames is None
            or inputs.ymt3_frames.ndim != 3
            or inputs.muq_frames.ndim != 3
            or inputs.ymt3_frames.shape[:2] != inputs.beats.shape
            or inputs.ymt3_frames.shape[-1] != self.ymt3_dim
            or inputs.muq_frames.shape[0] != inputs.beats.shape[0]
            or inputs.muq_frames.shape[-1] != self.muq_dim
            or inputs.valid.shape != inputs.beats.shape
        ):
            raise ValueError("YMT3+MuQ continuous inputs must have aligned ymt3_frames/muq_frames/beats/valid axes")

        dtype = self.ymt3_proj.weight.dtype
        y = self.ymt3_proj(self.ymt3_norm(inputs.ymt3_frames.to(dtype)))
        if inputs.sample_valid is not None:
            y = y.masked_fill(~inputs.sample_valid[..., None], 0.0)

        m40 = self.muq_proj(self.muq_norm(inputs.muq_frames.to(dtype)))
        if inputs.muq_sample_valid is not None:
            m40 = m40.masked_fill(~inputs.muq_sample_valid[..., None], 0.0)

        t_8ms = y.shape[1]
        if t_8ms > 0:
            m8 = F.interpolate(
                m40.transpose(1, 2),
                size=t_8ms,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        else:
            m8 = y.new_zeros(y.shape)
        if inputs.sample_valid is not None:
            m8 = m8.masked_fill(~inputs.sample_valid[..., None], 0.0)

        x = (y + m8) * (0.5 ** 0.5)
        x = mask_values(x, inputs.valid)

        # Physical-time RoPE: 1 frame (8ms) = 0.04 units (5 frames = 40ms = 0.2, 25 frames = 200ms = 1.0)
        frame_step = FRAME_SECONDS_8MS * (RATIO_40_TO_8 / 1.0)  # 0.04
        positions = (
            torch.arange(t_8ms, device=x.device, dtype=torch.float32)[None, :].expand(x.shape[0], -1)
            * frame_step
        )
        tables = rope_tables(
            positions.masked_fill(~inputs.valid, 0.0),
            self.head_dim,
            self.rope_base,
            self.rope_position_scale,
        )
        for layer in self.layers:
            x = layer(x, inputs.valid, tables)

        x = mask_values(self.output_norm(x), inputs.valid)
        p0 = ScaleMemory(values=x, beats=inputs.beats.float(), valid=inputs.valid)
        return ContinuousSongEncoding(
            P0=p0,
            beat_memory=None,
            stage_beats=(inputs.beats.float(),),
            stage_valid=(inputs.valid,),
            layout=inputs.layout,
        )

    def forward(self, song_inputs, requests=None) -> ContinuousSongEncoding:
        return self.encode_song(song_inputs)
