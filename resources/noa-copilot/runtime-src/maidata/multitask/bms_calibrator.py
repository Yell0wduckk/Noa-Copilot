"""Pure-audio full-song BMS beat, downbeat, meter and bar-zero offset calibration on 125/25 Hz clocks."""

# 训练机使用说明（clean lattice）
# v3 通过 --stage1-config / --stage2-config 选择独立配方。用 train --cost-check
# --cost-warmup 3 --cost-steps 8 实测目标 GPU 上含数据等待的完整更新。
# numba 为默认强制依赖（与当前 NumPy 版本匹配），用于编译并行分区、Viterbi、速度 DP
# 与候选去重；可用下列命令核对编译版分区函数：
# python -c "from maidata.multitask.calibrator_semicrf import partition_scan; print(partition_scan)"
# 缺少兼容 Numba 时入口会直接报错，仅在显式传入 --allow-no-numba（或设置
# MAIDATA_ALLOW_NO_NUMBA=1）时才允许回退到 NumPy。首次调用包含 JIT 编译耗时。
# PRO 6000 串联训练可追加 --stage2-batch-size 16 --stage2-learning-rate 0.00028
# --stage2-warmup-steps 250。L4 上的 v3 batch 与学习率按整曲实测选取，可用 --stage-mert-first。
# 校正器的 SDPA 带有浮点注意力偏置；训练环境需允许 PyTorch 为此掩码选择
# memory-efficient 或 math 后端。以下用例在支持 BF16 的 GPU 上执行两阶段各一次
# batch=8、每曲 6 秒的前向、反向与参数更新；整曲显存与吞吐以训练机实测为准。
# python -m pytest -o addopts= -s tests/multitask/test_clean_lattice.py::test_clean_lattice_cuda_recipe_batch_update_and_metrics

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from contextlib import nullcontext
from copy import copy
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import random
import re
import shutil
import subprocess
import sys
import time
from tempfile import TemporaryDirectory
import unicodedata
import zipfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from maidata.nn.position import apply_rope, rope_tables
from . import bms
from .calibrator_augmentation import (
    _clamped_speed_range, apply_piecewise_tempo_augmentation, mel_pitch_shift, piecewise_eligible, tempo_meter_flags,
)
from .calibrator_audibility import audibility_masks, inside_audible_bounds, song_ref_and_audibility
from .calibrator_boundary_decoder import decode_boundary_grid, decode_clean_grid
from .calibrator_partition import allow_numba_fallback, require_numba
from .calibrator_semicrf import BarSemiCRFHead, sequence_loss
from .bms_barlines import sanitize_bms_barlines_and_meters
from .calibrator_islands import decode_bar_sections, leading_bar_origin, meter_gaps, reconcile_tempo_islands
from .calibrator_legacy_islands import BarEvidence, anchor_stable_tempo_bar_islands, filter_stutter_stops
from .audio import DEFAULT_MERT_V2_MODEL, MERT_V2_ORIGIN_SECONDS, fast_resample, sample_centers_batch
from .mel_frontend import (
    DualWindowMelFrontend, FRAME_RATE, SAMPLE_RATE, _k_sections, _shifted_mel_banks,
    frontend_segments, segment_frame_slices, song_reference_db,
)
from .calibrator_oom import (
    CALIBRATOR_OOM_EXIT_CODE,
    CalibratorBatch,
    CalibratorOOMRecovery,
    CalibratorRestartRequired,
    batch_needs_online_mert,
    calibrator_batch_key,
    padded_batch_seconds,
    precompute_batch_mert,
    slice_calibrator_batch,
    split_calibrator_batch,
    supervise_calibrator_train,
)
from .data import excluded_song
from .grid import BeatGrid, multiplier_to_meter

CALIBRATOR_ARCHITECTURE = "maidata.bms-beat-calibrator.unet-roformer-mertv2.v18"
CACHE_SCHEMA = "bms-beat-cache.v18"
CHECKPOINT_SCHEMA = "bms-beat-checkpoint.v18"
MERT_STRIDE_SAMPLES = SAMPLE_RATE // 25
REFERENCE_SONG_FRAMES = 180 * FRAME_RATE
METERS = ((4, 4), (3, 4), (2, 4), (6, 8), (12, 8), (5, 4), (7, 8), (9, 8))
OTHER = len(METERS)
STAGE2_EXTRA_METERS = (
    (1, 4), (6, 4), (7, 4), (5, 8), (8, 4), (3, 8), (11, 8),
    (11, 4), (13, 8), (15, 8), (9, 4), (10, 4),
)
FOCUSED_V2_METERS = METERS + STAGE2_EXTRA_METERS[:7]
TRANSITIONAL_METERS = METERS + STAGE2_EXTRA_METERS[:4]
EXTENDED_METERS = METERS + STAGE2_EXTRA_METERS
MIN_CALIBRATOR_BPM = 50.0
MAX_CALIBRATOR_BPM = 300.0
MIN_ANCHOR_SECONDS = 1.5
MAX_CALIBRATOR_SEGMENTS = 48
MAX_CALIBRATOR_DURATION_SECONDS = 300.0
DEFAULT_MAX_TRAIN_SECONDS = 240.0
DEFAULT_STAGE2_MERT_LAYERS = (17,)
EXTERNAL_CALIBRATOR_SONGS = (("11629", "熱異常"), ("10146", "39"), ("834", "PANDORA PARADOXXX"),
         ("799", "QZKago Requiem"), ("11662", "raputa"), ("11818", "World's end BLACKBOX"),
         ("11820", "Xaleid◆scopiX"))
HARMONIC_GIMMICK_RATIOS = (0.25, 1.0 / 3.0, 0.5, 2.0 / 3.0, 0.75, 4.0 / 3.0, 1.5, 2.0, 3.0, 4.0)
HARMONIC_LOSS_CANDIDATES = (
    (0.25, 0.10),
    (1.0 / 3.0, 0.08),
    (0.5, 0.05),
    (2.0 / 3.0, 0.08),
    (1.5, 0.08),
    (2.0, 0.05),
    (3.0, 0.08),
    (4.0, 0.10),
)


class RMSNorm(nn.RMSNorm):
    def forward(self, x):
        # Matching dtypes select the fused RMS kernel under BF16 autocast.
        return F.rms_norm(x, self.normalized_shape, self.weight.to(x.dtype), self.eps)


class SpectralBlock(nn.Module):
    def __init__(self, incoming, width):
        super().__init__()
        self.down = nn.Conv2d(incoming, width, (3, 5), (1, 2), (1, 2))
        self.conv = nn.Conv2d(width, width, 3, padding=1)
        self.skip = nn.Conv2d(incoming, width, 1, stride=(1, 2))
        self.norm = RMSNorm(width)

    def forward(self, x, valid):
        h = self.down(x)
        h = F.gelu(self.norm(h.movedim(1, -1)).movedim(-1, 1))
        h = self.conv(h * valid)
        return F.gelu(h + self.skip(x)) * valid


class TemporalBlock(nn.Module):
    def __init__(self, d, kernel, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv1d(
            d, d, kernel, dilation=dilation, padding=dilation * (kernel // 2), groups=d
        )
        self.pointwise = nn.Linear(d, d)

    def forward(self, x):
        return x + self.pointwise(F.gelu(self.depthwise(x.transpose(1, 2)).transpose(1, 2)))


class AcousticStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.spectral = nn.ModuleList(
            SpectralBlock(a, b) for a, b in zip((5, 32, 64, 96), (32, 64, 96, 128))
        )
        self.project = nn.Linear(128 * 8, 384)
        self.temporal = nn.ModuleList(TemporalBlock(384, 7, dilation) for dilation in (1, 2, 4))

    def forward(self, x, valid):
        mask = valid[:, None, :, None]
        # Windows CPU oneDNN can corrupt memory during channels-last convolution backward.
        windows_cpu = sys.platform == "win32" and x.device.type == "cpu"
        memory_format = torch.contiguous_format if windows_cpu else torch.channels_last
        x = (x * mask).contiguous(memory_format=memory_format)
        for layer in self.spectral:
            x = layer(x, mask)
        x = self.project(x.transpose(1, 2).flatten(2)) * valid[..., None]
        for layer in self.temporal:
            x = layer(x) * valid[..., None]
        return x


class LocalAttentionLayout:
    """Construct the window once per song batch and reuse it across six layers."""

    tile = 128

    def __init__(self, lengths, count, radius=375):
        from torch.utils._triton import has_triton

        self.radius = int(radius)
        self.block_mask = None
        self.count = count
        positions = torch.arange(count, device=lengths.device)
        if count <= self.radius + 1:
            self.mask = positions[None, None, None, :] < lengths[:, None, None, None]
        elif lengths.is_cuda and has_triton():
            self.block_mask = self.flex_mask(lengths, count, self.radius)
        else:
            # SDPA fallback: batch neighbouring key bands, linear in song duration.
            self.starts = torch.arange(0, count, self.tile, device=lengths.device)
            qs = self.starts[:, None] + torch.arange(self.tile, device=lengths.device)
            offsets = torch.arange(-self.radius, self.tile + self.radius, device=lengths.device)
            keys = self.starts[:, None] + offsets
            self.keys = keys.clamp(0, count - 1)
            self.mask = ((qs[:, :, None] - keys[:, None, :]).abs() <= self.radius)[None]
            self.mask = (
                self.mask
                & (keys >= 0)[None, :, None]
                & (keys[None, :, None] < lengths[:, None, None, None])
            )
            # Padding queries are cleared after the block. Give empty rows a
            # finite softmax even on SDPA backends that return NaN for them.
            empty = ~self.mask.any(-1, keepdim=True)
            self.mask = torch.cat((self.mask[..., :1] | empty, self.mask[..., 1:]), dim=-1)

    @staticmethod
    def _ordered_blocks(mask):
        counts = mask.sum(-1, dtype=torch.int32)
        # BlockMask uses the last index dimension as the logical block-grid
        # width. Kernels visit only each row's counted prefix.
        indices = torch.argsort(mask.to(torch.int8), dim=-1, descending=True, stable=True)
        return counts, indices.int().contiguous()

    @classmethod
    def flex_mask(cls, lengths, count, radius=375):
        from torch.nn.attention.flex_attention import BlockMask

        lo = torch.arange(0, count, cls.tile, device=lengths.device)
        limit = lengths[:, None, None]
        q_lo, k_lo = lo[None, :, None], lo[None, None, :]
        q_hi = torch.minimum(q_lo + cls.tile - 1, limit - 1)
        k_hi = torch.minimum(k_lo + cls.tile - 1, limit - 1)
        any_visible = (q_lo < limit) & (k_lo < limit) & (q_lo - k_hi <= radius) & (k_lo - q_hi <= radius)
        full = (
            (q_lo + cls.tile <= limit)
            & (k_lo + cls.tile <= limit)
            & ((q_lo - k_lo).abs() + cls.tile - 1 <= radius)
        )
        counts, indices = cls._ordered_blocks((any_visible & ~full)[:, None])
        full_counts, full_indices = cls._ordered_blocks(full[:, None])

        def visible(b, h, q, k):
            return (q < lengths[b]) & (k < lengths[b]) & ((q - k).abs() <= radius)

        return BlockMask(
            seq_lengths=(count, count),
            kv_num_blocks=counts,
            kv_indices=indices,
            q_num_blocks=counts,
            q_indices=indices,
            full_kv_num_blocks=full_counts,
            full_kv_indices=full_indices,
            full_q_num_blocks=full_counts,
            full_q_indices=full_indices,
            BLOCK_SIZE=(cls.tile, cls.tile),
            mask_mod=visible,
        )

    def attend(self, q, k, v):
        if self.block_mask is not None:
            from maidata.spine.history import compiled_attention

            return compiled_attention()(q, k, v, block_mask=self.block_mask)
        if self.count <= self.radius + 1:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=self.mask)
        b, h, n, d = q.shape
        tiles = len(self.starts)
        query = (
            F.pad(q, (0, 0, 0, tiles * self.tile - n)).reshape(b, h, tiles, self.tile, d).transpose(1, 2)
        )

        def gather(bank):
            return bank[:, :, self.keys].transpose(1, 2).reshape(b * tiles, h, -1, d)

        result = F.scaled_dot_product_attention(
            query.reshape(b * tiles, h, self.tile, d),
            gather(k),
            gather(v),
            attn_mask=self.mask.reshape(b * tiles, 1, self.tile, -1),
        )
        return result.reshape(b, tiles, h, self.tile, d).transpose(1, 2).reshape(b, h, -1, d)[:, :, :n]


class GroupedFFN(nn.Module):
    def __init__(self, dim=512, hidden=2048, groups=2):
        super().__init__()
        self.groups = groups
        width, inner = dim // groups, hidden // groups
        self.w1 = nn.Parameter(torch.empty(groups, width, inner))
        self.b1 = nn.Parameter(torch.empty(groups, 1, inner))
        self.w2 = nn.Parameter(torch.empty(groups, inner, width))
        self.b2 = nn.Parameter(torch.empty(groups, 1, width))
        # Linear's Kaiming-uniform initialization uses the per-group fan-in.
        for weight, bias in ((self.w1, self.b1), (self.w2, self.b2)):
            bound = 1 / math.sqrt(weight.shape[1])
            nn.init.uniform_(weight, -bound, bound)
            nn.init.uniform_(bias, -bound, bound)

    def forward(self, z):
        grouped = z.reshape(-1, self.groups, self.w1.shape[1]).transpose(0, 1)
        hidden = F.gelu(torch.baddbmm(self.b1, grouped, self.w1))
        return torch.baddbmm(self.b2, hidden, self.w2).transpose(0, 1).reshape_as(z)

    def _load_from_state_dict(self, state_dict, prefix, *args):
        # Early v18 weight exports stored the same two matrices as grouped 1x1 convolutions.
        if prefix + "0.weight" in state_dict:
            for i, conv in ((1, "0"), (2, "2")):
                weight = state_dict.pop(prefix + conv + ".weight").squeeze(-1)
                bias = state_dict.pop(prefix + conv + ".bias")
                state_dict[prefix + f"w{i}"] = weight.unflatten(0, (self.groups, -1)).transpose(1, 2)
                state_dict[prefix + f"b{i}"] = bias.reshape(self.groups, 1, -1)
        super()._load_from_state_dict(state_dict, prefix, *args)


class RoFormerBlock(nn.Module):
    def __init__(self, dim=512, heads=16, hidden=2048, groups=2):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.attention_norm = RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = GroupedFFN(dim, hidden, groups)

    def forward(self, x, layout, pos, tables, valid):
        q, k, v = self.qkv(self.attention_norm(x)).unflatten(-1, (3, self.heads, self.head_dim)).permute(2, 0, 3, 1, 4)
        h = layout.attend(apply_rope(q, pos, tables=tables), apply_rope(k, pos, tables=tables), v)
        x = x + self.out(h.transpose(1, 2).flatten(2))
        return (x + self.ffn(self.ffn_norm(x))) * valid[..., None]


class MERTGate(nn.Module):
    def __init__(self, layers, *, fixed_layer=False, dropout=0.35, gate_bias=-2.0):
        super().__init__()
        self.layer_logits = None if fixed_layer else nn.Parameter(torch.zeros(layers))
        self.norm = nn.LayerNorm(1024, bias=False)
        self.projection = nn.Linear(1024, 512, bias=False)
        self.gate = nn.Linear(1024, 512)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_bias)
        self.dropout = dropout

    def forward(self, h, mert):
        with torch.no_grad():
            normed = F.layer_norm(mert.to(h.dtype), self.norm.normalized_shape, eps=self.norm.eps)
        if normed.shape[-2] == 1:
            mixed = normed.squeeze(-2)
        else:
            mixed = self.layer_logits.softmax(0).to(h.dtype) @ normed
        mixed = mixed * self.norm.weight.to(h.dtype)
        m = self.projection(mixed)
        if self.training and self.dropout > 0:
            m = m * (torch.rand(len(h), 1, 1, device=h.device) >= self.dropout)
        return h + self.gate(torch.cat((h, m), -1)).sigmoid() * m


class SumHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.components = nn.Linear(384, 2)

    def forward(self, h):
        sub, down = self.components(h).float().unbind(-1)
        return torch.logaddexp(sub, down), down


class NestedBoundaryHead(nn.Linear):
    def __init__(self):
        super().__init__(384, 3)

    def forward(self, features):
        eighth, quarter, bar = super().forward(features).float().unbind(-1)
        beat = torch.logaddexp(quarter, bar)
        return torch.logaddexp(eighth, beat), beat, bar


class ExtendedMeterHead(nn.Linear):
    """Use the aligned CUDA head and preserve the original CPU hot-start accumulation order."""

    def forward(self, features):
        if features.is_cuda and self.out_features in (len(FOCUSED_V2_METERS) + 1, len(EXTENDED_METERS) + 1):
            return F.linear(features, self.weight, self.bias)
        legacy_weight = torch.cat((self.weight[:OTHER], self.weight[-1:]))
        legacy_bias = torch.cat((self.bias[:OTHER], self.bias[-1:]))
        legacy = F.linear(features, legacy_weight, legacy_bias)
        extra = F.linear(features, self.weight[OTHER:-1], self.bias[OTHER:-1])
        return torch.cat((legacy[..., :OTHER], extra, legacy[..., -1:]), dim=-1)


def _meter_table_from_logits_dim(num_logits):
    for meters in (METERS, TRANSITIONAL_METERS, FOCUSED_V2_METERS, EXTENDED_METERS):
        if num_logits == len(meters) + 1:
            return meters, len(meters)
    raise ValueError(f"Expected 9, 13, 16 or 21 meter logits, got {num_logits}")


@dataclass
class BMSCalibratorConfig:
    stage: int = 1
    mert_layers: int = 0
    extended_meters: bool = False
    unfreeze_grid_head: bool = False
    grid_lr_scale: float = 0.25
    boundary_model: bool = False
    clean_lattice: bool = False
    clean_v3: bool = False
    stage1_roformer_radius: int = 250
    stage2_macro_radius: int = 200
    stage2_mert_dropout: float = 0.0


class BMSBeatCalibrator(nn.Module):
    architecture = CALIBRATOR_ARCHITECTURE

    def __init__(self, config=None, *, legacy_clean=False, clean_v1=False):
        super().__init__()
        self.config = (
            config if isinstance(config, BMSCalibratorConfig) else BMSCalibratorConfig(**(config or {}))
        )
        if self.config.clean_v3:
            self.config.clean_lattice = True
        if self.config.boundary_model:
            self.config.stage = 2
            self.config.mert_layers = len(DEFAULT_STAGE2_MERT_LAYERS)
        if self.config.clean_lattice and self.config.stage == 2:
            self.config.mert_layers = 1
        self.frontend = DualWindowMelFrontend(pitch_banks=self.config.clean_lattice and (self.config.stage == 1 or self.config.clean_v3) and not legacy_clean)
        self.stage_a = AcousticStem()
        if self.config.clean_v3:
            self.blocks = nn.ModuleList(RoFormerBlock(384, 12, 1536) for _ in range(6))
            self.fusion = TemporalBlock(384, 5)
            self.sum_head = SumHead()
            self.grid_head = nn.Linear(384, 5)
            if self.config.stage == 2:
                from .calibrator_corrector import LatticeCorrection
                self.requires_grad_(False)
                self.correction = LatticeCorrection(v3_config=self.config)
                self.train(self.training)
            return
        self.downsample = nn.Conv1d(384, 512, 5, stride=5)
        self.blocks_pre = nn.ModuleList(RoFormerBlock() for _ in range(4))
        self.blocks_post = nn.ModuleList(RoFormerBlock() for _ in range(2))
        self.mert_gate = (MERTGate(self.config.mert_layers, fixed_layer=self.config.boundary_model)
                          if self.config.stage == 2 and not self.config.clean_lattice else None)
        self.up = nn.Linear(512, 384)
        self.fusion = TemporalBlock(384, 5)
        if self.config.clean_lattice:
            self.boundary_head = NestedBoundaryHead()
            self.grid_head = nn.Linear(384, 9 if legacy_clean else 5)
            if self.config.stage == 2:
                from .calibrator_corrector import LatticeCorrection
                self.requires_grad_(False)
                self.correction = LatticeCorrection(legacy=legacy_clean, v1=clean_v1)
                self.train(self.training)
            return
        if self.config.boundary_model:
            self.boundary_head = NestedBoundaryHead()
            self.kinematic_head = nn.Linear(384, 7)
            self.tempo_regime_head = nn.Linear(384, 5)
            self.compound_head = nn.Linear(384, 1)
            self.semicrf = BarSemiCRFHead()
            self.stage_a.requires_grad_(False)
            self.downsample.requires_grad_(False)
            self.blocks_pre[:3].requires_grad_(False)
            return
        self.sum_head = SumHead()
        self.grid_head = nn.Linear(384, 8)
        meters = EXTENDED_METERS if self.config.extended_meters else METERS
        head_type = ExtendedMeterHead if self.config.extended_meters else nn.Linear
        self.meter_head = head_type(384, len(meters) + 1)
        if self.config.stage == 2:
            self.requires_grad_(False)
            self.set_stage2_roformer_trainable(True)
            for module in (self.mert_gate, self.sum_head, self.meter_head):
                module.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "correction"):
            for name, module in self.named_children():
                if name != "correction":
                    module.eval()
        return self

    def set_stage2_roformer_trainable(self, trainable: bool):
        if self.config.stage == 2 and not self.config.clean_lattice:
            self.blocks_pre[3].requires_grad_(trainable)
            self.blocks_post.requires_grad_(trainable)
            if self.config.unfreeze_grid_head and not self.config.boundary_model:
                for module in (self.up, self.fusion, self.grid_head):
                    module.requires_grad_(trainable)

    def parameter_groups(self, lr, *, split_roformer=False):
        if self.config.stage == 1:
            return [dict(params=self.parameters(), lr=lr)]
        if self.config.clean_lattice:
            return [dict(params=self.correction.parameters(), lr=lr)]
        slow = dict(params=list(self.blocks_pre[3].parameters()), lr=0.1 * lr)
        post = list(self.blocks_post.parameters())
        if self.config.boundary_model:
            heads = (self.mert_gate, self.boundary_head, self.tempo_regime_head, self.compound_head, self.semicrf)
            bridge = (self.up, self.fusion, self.kinematic_head)
            return [slow, dict(params=post, lr=lr),
                    dict(params=[p for m in bridge for p in m.parameters()], lr=.25 * lr),
                    dict(params=[p for m in heads for p in m.parameters()], lr=lr)]
        heads = [
            p for module in (self.mert_gate, self.sum_head, self.meter_head) for p in module.parameters()
            if p.requires_grad
        ]
        grid_groups = []
        if self.config.unfreeze_grid_head:
            bridge = [p for module in (self.up, self.fusion, self.grid_head) for p in module.parameters()]
            grid_groups.append(dict(params=bridge, lr=self.config.grid_lr_scale * lr))
        if split_roformer:
            return [slow, dict(params=post, lr=lr), *grid_groups, dict(params=heads, lr=lr)]
        return [slow, *grid_groups, dict(params=[p for p in post if p.requires_grad] + heads, lr=lr)]

    def _resolve_online_mert_extractor(self, device, mert_model=None):
        """Lazily create and cache an OnlineMERTv2Extractor outside nn.Module state."""
        cache = self.__dict__.setdefault("_online_mert_cache", {})
        model_ref = DEFAULT_MERT_V2_MODEL if mert_model is None else mert_model
        key = (
            str(model_ref) if isinstance(model_ref, (str, Path)) else ("backbone", id(model_ref)),
            str(device),
            int(self.config.mert_layers),
        )
        extractor = cache.get(key)
        if extractor is None:
            layers = DEFAULT_STAGE2_MERT_LAYERS if self.config.boundary_model or self.config.clean_lattice else self.config.mert_layers
            if isinstance(model_ref, (str, Path)):
                extractor = OnlineMERTv2Extractor(
                    str(model_ref),
                    device=device,
                    layers=layers,
                )
            else:
                extractor = OnlineMERTv2Extractor(
                    device=device,
                    layers=layers,
                    backbone=model_ref,
                )
            cache[key] = extractor
        return extractor

    def forward_waveforms(
        self,
        waveforms,
        sample_lengths=None,
        song_ref_db=None,
        *,
        mert=None,
        mert_extractor=None,
        mert_model=None,
        pitch_semitones=None,
        mask_aug_sources=None,
        speed_segments=None,
        **kwargs,
    ):
        wave = waveforms.reshape(-1, waveforms.shape[-1])
        if sample_lengths is None:
            sample_lengths = [wave.shape[-1]] * len(wave)
        if song_ref_db is None:
            # Uncached inference only. Training always supplies the worker's scalar.
            song_ref_db = [
                song_reference_db(row[:n].float().detach().cpu().numpy())
                for row, n in zip(wave, sample_lengths)
            ]
        if self.config.stage == 2 and mert is None:
            extractor = (
                mert_extractor
                if mert_extractor is not None
                else self._resolve_online_mert_extractor(wave.device, mert_model=mert_model)
            )
            mert = (extractor(wave, sample_lengths, speed_segments=speed_segments) if speed_segments
                    else extractor(wave, sample_lengths))
        pitch_segments = None
        if speed_segments:
            pitch_segments = [[(a, b, pitch) for a, b, _, pitch in frontend_segments(row)] for row in speed_segments]
        features, lengths = self.frontend.batch(wave, sample_lengths, song_ref_db, pitch_semitones,
                                                **({"pitch_segments": pitch_segments} if pitch_segments else {}))
        if mask_aug_sources is not None:
            # CPU index plan, one gather/scatter for all selected frames on GPU.
            rows, frames = torch.where(mask_aug_sources >= 0)
            source = mask_aug_sources[rows, frames].to(wave.device)
            rows, frames = rows.to(wave.device), frames.to(wave.device)
            features[rows, :4, frames] = features[rows, :4, source]
        if self.config.clean_lattice and "audible_mask" not in kwargs:
            refs = torch.as_tensor(song_ref_db).reshape(-1).expand(len(wave)).tolist()
            samples = torch.as_tensor(sample_lengths).reshape(-1).expand(len(wave)).tolist()
            masks = [audibility_masks(row[:n].detach().float().cpu().numpy(), ref, _k_sections())[0]
                     for row, n, ref in zip(wave, samples, refs)]
            kwargs["audible_mask"] = nn.utils.rnn.pad_sequence(masks, batch_first=True).to(wave.device)
        out = self(features, lengths, mert=mert, **kwargs)
        if mask_aug_sources is not None:
            out["mask_aug_frames"] = (mask_aug_sources >= 0).to(wave.device)
        return out

    def forward(self, features, lengths=None, *, mert=None, audible_mask=None):
        if self.config.stage == 2 and mert is None:
            raise ValueError("Stage 2 BMSBeatCalibrator requires `mert` inputs in forward().")
        b, _, n, _ = features.shape
        frames = torch.arange(n, device=features.device)
        if lengths is None:
            lengths = frames.new_full((b,), n)
        lengths = lengths.to(frames.device)
        valid = frames < lengths[:, None]
        if self.config.clean_v3:
            frozen = self.config.stage == 2
            with torch.no_grad() if frozen else nullcontext():
                local = self.stage_a(features, valid)
                layout = LocalAttentionLayout(lengths, n, self.config.stage1_roformer_radius)
                pos = frames[None]
                tables = rope_tables(pos, 32)
                h = local
                for layer in self.blocks:
                    h = layer(h, layout, pos, tables, valid)
                u = self.fusion((local + h) * valid[..., None]) * valid[..., None]
                sub, down = self.sum_head.components(u).float().unbind(-1)
                g = self.grid_head(u).float()
                audible = (g[..., 4] >= 0 if audible_mask is None else
                           F.pad(audible_mask[:, :n].bool(), (0, max(0, n - audible_mask.shape[1]))))
                out = dict(beat_logit=torch.logaddexp(sub.contiguous(), down.contiguous()), downbeat_logit=down,
                           subframe_offset_ms=4 * g[..., 0].tanh(), pulse_phase=g[..., 1:3],
                           stop_logit=g[..., 3], audible_logit=g[..., 4], audible_mask=audible & valid,
                           valid_mask=valid, lengths=lengths, boundary_features=u)
            if frozen:
                self.correction.forward_quarter(out, mert, sub)
            return out
        frozen = self.config.clean_lattice and self.config.stage == 2
        with torch.no_grad() if frozen else nullcontext():
            local = self.stage_a(features, valid)
            h = self.downsample(F.pad(local.transpose(1, 2), (0, (-n) % 5))).transpose(1, 2)
            pos = torch.arange(h.shape[1], device=h.device)[None]
            coarse_lengths = (lengths + 4) // 5
            coarse_valid = pos < coarse_lengths[:, None]
            layout = LocalAttentionLayout(coarse_lengths, h.shape[1])
            tables = rope_tables(pos, 32)
            for layer in self.blocks_pre:
                h = layer(h, layout, pos, tables, coarse_valid)
            if self.mert_gate is not None and mert is not None:
                if b == 1 and mert.ndim == 3:
                    mert = mert.unsqueeze(0)
                mert = mert.to(device=h.device)
                h = self.mert_gate(h, _prepare_mert_batch(mert, h.shape[1])) * coarse_valid[..., None]
            h_pre = h
            for layer in self.blocks_post:
                h = layer(h, layout, pos, tables, coarse_valid)
            coarse = self.up(h)
            locations = torch.minimum(((frames + 0.5) / 5 - 0.5).clamp_min(0), coarse_lengths[:, None] - 1)
            left = locations.long()
            song = torch.arange(b, device=h.device)[:, None]
            interp = torch.lerp(
                coarse[song, left],
                coarse[song, torch.minimum(left + 1, coarse_lengths[:, None] - 1)],
                (locations - left).to(coarse.dtype)[..., None],
            )
            u = self.fusion((local + interp) * valid[..., None]) * valid[..., None]
            if self.config.clean_lattice:
                eighth, tactus, bar = self.boundary_head(u)
                g = self.grid_head(u).float()
                audible = ((g[..., -1] >= 0) if audible_mask is None else
                           F.pad(audible_mask[:, :n].bool(), (0, max(0, n-audible_mask.shape[1]))))
                out = dict(eighth_logit=eighth, beat_logit=tactus, downbeat_logit=bar,
                           subframe_offset_ms=4*g[..., 0].tanh(), pulse_phase=g[..., 1:3],
                           eighth_phase=g[..., 1:3], stop_logit=g[..., -2], audible_logit=g[..., -1],
                           audible_mask=audible & valid, valid_mask=valid, lengths=lengths,
                           boundary_features=u, hidden_40ms=h_pre)
                if self.grid_head.out_features == 9:  # Historical clean checkpoints.
                    out.update(bar_phase=g[..., 3:5], log_bpm=4*(g[..., 5]/4).tanh(), change_logit=g[..., 6])
        if self.config.clean_lattice:
            if self.config.stage == 2:
                aligned = _prepare_mert_batch(mert.unsqueeze(0) if b == 1 and mert.ndim == 3 else mert, h.shape[1])
                self.correction(out, aligned.to(h.device), layout, pos, tables, coarse_valid, locations)
            return out
        if self.config.boundary_model:
            eighth, beat, bar = self.boundary_head(u)
            kin = self.kinematic_head(u).float()
            tempo = self.tempo_regime_head(u).float()
            return dict(eighth_logit=eighth, beat_logit=beat, barbeat_logit=bar, downbeat_logit=bar,
                        subframe_offset_ms=4 * kin[..., 0].tanh(), pulse_phase=kin[..., 1:3],
                        bar_phase=kin[..., 3:5], log_bpm=4 * (kin[..., 5] / 4).tanh(), stop_logit=kin[..., 6],
                        tempo_change_logit=tempo[..., 0], change_logit=tempo[..., 0], regime_logits=tempo[..., 1:],
                        compound_logit=self.compound_head(u).squeeze(-1).float(), boundary_features=u,
                        semicrf=self.semicrf, valid_mask=valid, lengths=lengths, hidden_40ms=h)
        beat, down = self.sum_head(u)
        raw = self.grid_head(u).float()
        return dict(
            beat_logit=beat,
            downbeat_logit=down,
            subframe_offset_ms=4 * raw[..., 0].tanh(),
            pulse_phase=raw[..., 1:3],
            bar_phase=raw[..., 3:5],
            log_bpm=4 * (raw[..., 5] / 4).tanh(),
            change_logit=raw[..., 6],
            stop_logit=raw[..., 7],
            meter_logits=self.meter_head(u).float(),
            valid_mask=valid,
            lengths=lengths,
            hidden_40ms=h,
        )


def _event_times(grid, duration, start_beat=None):
    lo = grid.second_to_beat(0) if start_beat is None else float(start_beat)
    pulses, bars = grid.pulse_and_bar_beats(lo, grid.second_to_beat(duration))
    return tuple(grid.beat_to_second_tensor(torch.tensor(marks, dtype=torch.float32)) for marks in (pulses, bars))


def _nearest(events, centers):
    i = torch.searchsorted(events.contiguous(), centers.contiguous()).clamp_max(len(events) - 1)
    left, right = events[(i - 1).clamp_min(0)], events[i]
    return torch.where((right - centers).abs() < (centers - left).abs(), right, left)


def _meter_class_id(num, den, span, meter_table=METERS):
    for klass, (m_num, m_den) in enumerate(meter_table):
        if num == m_num and den == m_den and abs(span - 4.0 * m_num / m_den) < 1e-5:
            return klass
    return len(meter_table)


def _unverified_compound(grid, beats, info, duration):
    simple = (info["den"] == 4) & ((info["num"] == 3) | (info["num"] == 6))
    if not simple.any():
        return simple
    starts = beats - info["bar_phase"] * info["bar_beats"]
    # Old caches lack the accent-confidence field. Reconstruct the uninterrupted
    # run on the original score clock, independent of training speed.
    if grid.ambiguous_compound_bars is not None:
        marks = beats.new_tensor(grid.ambiguous_compound_bars)
        return simple & torch.isin((starts * 1000).round(), (marks * 1000).round())
    bars = np.asarray(grid.bar_beats_explicit)
    if len(bars) < 2:
        end = grid.second_to_beat(duration)
        _, marks = grid.pulse_and_bar_beats(min(grid.meter_starts[0], float(beats[0])), end)
        bars = np.unique([*marks, end])
    if len(bars) < 2:
        return simple
    widths = np.diff(bars)
    left = bars[:-1]
    tempo = np.searchsorted(grid.beats, left + 1e-6, side="right") - 1
    meter = np.searchsorted(grid.meter_starts, left + 1e-6, side="right") - 1
    stops = np.searchsorted(grid.stop_beats, left + 1e-6, side="right")
    # The original accent histogram never pooled across a STOP or a tempo cut,
    # nor did it classify the measure containing an interior cut.
    divided = (tempo != np.searchsorted(grid.beats, bars[1:] - 1e-6, side="right") - 1) | (
        stops != np.searchsorted(grid.stop_beats, bars[1:] - 1e-6, side="right")
    )
    same = (
        (np.diff(tempo) == 0)
        & (np.diff(meter) == 0)
        & (np.diff(stops) == 0)
        & np.isclose(widths[:-1], widths[1:])
        & ~divided[:-1]
        & ~divided[1:]
    )
    cuts = np.r_[0, np.flatnonzero(~same) + 1, len(left)]
    lengths = np.repeat(np.diff(cuts), np.diff(cuts))
    lengths[divided] = 0
    idx = torch.searchsorted(beats.new_tensor(left), beats, right=True).sub(1).clamp(0, len(left) - 1)
    run = beats.new_tensor(lengths, dtype=torch.long)[idx]
    tempo_idx = torch.searchsorted(beats.new_tensor(grid.beats), beats, right=True).sub(1).clamp_min(0)
    fast = beats.new_tensor(grid.bpms)[tempo_idx] > 210
    return simple & (fast | (run < torch.where(info["num"] == 3, 4, 2)))


def calibrator_targets_from_grid(grid, num_frames, *, active_start_seconds=None, extended_meters=False, boundary_model=False,
                                 clean_lattice=False, clean_v3=False, audio_masks=None):
    clean_lattice = clean_lattice or clean_v3
    boundary_model = boundary_model or clean_lattice
    t = (torch.arange(num_frames) + 0.5) / FRAME_RATE
    beats = grid.second_to_beat_tensor(t)
    info = grid.bar_info(beats)
    dur_sec = num_frames / FRAME_RATE
    if boundary_model:
        lower = grid.second_to_beat(-.05 if grid.crop > 0 else 0)
        marks = grid.calibrator_tactus_beats if clean_lattice and not clean_v3 else grid.calibrator_boundary_beats
        eighth_marks, quarter_marks, bar_marks = marks(lower, grid.second_to_beat(dur_sec + .05))
        eighth, pulse, bars = (grid.beat_to_second_tensor(t.new_tensor(marks))
                               for marks in (eighth_marks, quarter_marks, bar_marks))
    elif grid.crop > 0.0:
        pulse_marks, bar_marks = grid.pulse_and_bar_beats(
            grid.second_to_beat(-0.05), grid.second_to_beat(dur_sec + 0.05)
        )
        pulse, bars = (
            grid.beat_to_second_tensor(torch.tensor(marks, dtype=torch.float32))
            for marks in (pulse_marks, bar_marks)
        )
    else:
        pulse, bars = _event_times(grid, dur_sec)
    changes = grid.beats[1:] if boundary_model else sorted(set(grid.beats[1:]) | set(grid.meter_starts[1:]))
    if clean_v3:
        changes = []
        for i in range(1, len(grid.starts)):
            prev = grid.bpms[i-1] + grid.slopes[i-1] * (grid.starts[i] - grid.starts[i-1])
            if not math.isclose(prev, grid.bpms[i]) or not math.isclose(grid.slopes[i-1], grid.slopes[i]):
                changes.append(grid.beats[i])
    change_times = grid.beat_to_second_tensor(t.new_tensor(changes))
    if active_start_seconds is not None and active_start_seconds > 0:
        start_sec = float(active_start_seconds)
        event_guard = max(0.0, start_sec - 0.02)
        if boundary_model:
            eighth = eighth[eighth >= event_guard]
        pulse = pulse[pulse >= event_guard]
        bars = bars[bars >= event_guard]
        change_times = change_times[change_times > start_sec + 0.02]
        active = t >= start_sec + 1e-5
    else:
        active = torch.ones_like(t, dtype=torch.bool)

    def residual(events):
        return _nearest(events, t) - t if len(events) else torch.full_like(t, torch.inf)

    def gaussian(dt):
        return (-0.5 * (dt * FRAME_RATE / 0.8).square()).exp()

    def circle(phi):
        angle = 2 * math.pi * phi
        return torch.stack((angle.sin(), angle.cos()), -1)

    offset_events = eighth if boundary_model and not clean_v3 else pulse
    delta = residual(offset_events)
    peak = torch.zeros_like(t, dtype=torch.bool)
    # The decoder takes the first point of a flat maximum. Resolve
    # floating-point ties at exact frame edges to that same peak frame.
    values = gaussian(delta)
    if len(offset_events):
        p_win = offset_events
        if boundary_model or grid.crop > 0:
            p_win = p_win[(p_win >= -1e-6) & (p_win <= dur_sec + 1e-6)]
        if len(p_win):
            right = (p_win * FRAME_RATE).floor().long().clamp(0, num_frames - 1)
            left = (right - 1).clamp_min(0)
            frames = torch.where(values[left] >= values[right], left, right)
            if grid.crop > 0.0:
                frames = frames[(p_win - t[frames]).abs() <= 0.5 / FRAME_RATE + 1e-6]
            peak[frames] = True
    meter_table = EXTENDED_METERS if extended_meters or boundary_model else METERS
    meter = torch.full_like(beats, len(meter_table), dtype=torch.long)
    for klass, (num, den) in enumerate(meter_table):
        meter[
            (info["num"] == num) & (info["den"] == den) & ((info["bar_beats"] - 4 * num / den).abs() < 1e-5)
        ] = klass
    dt_bar = residual(bars)
    # A pause has constant beat coordinate even though physical seconds advance.
    stop = torch.zeros_like(peak)
    if grid.stop_times:
        starts, ends = t.new_tensor(grid.stop_times), t.new_tensor(grid.stop_ends)
        song_sec = grid.crop + t * grid.speed
        idx = torch.searchsorted(starts, song_sec, right=True) - 1
        stop = (idx >= 0) & (song_sec < ends[idx.clamp_min(0)]) & active
    change = gaussian(residual(change_times))
    # BeatGrid maps augmented seconds to source seconds with a speed multiplier.
    seg_beats_t = t.new_tensor(grid.beats)
    ids = torch.searchsorted(seg_beats_t, beats.contiguous(), right=True).sub_(1).clamp_min_(0)
    area = (beats - seg_beats_t[ids]) * 60
    base, slope = t.new_tensor(grid.bpms)[ids], t.new_tensor(grid.slopes)[ids]
    bpm = (base.square() + 2 * slope * area.clamp_min(0)).clamp_min(1.0).sqrt() * grid.speed

    # Continuous unwrapped segment coordinates for harmonic-ratio loss tolerance.
    m_starts_t = t.new_tensor(grid.meter_starts)
    m_ids = torch.searchsorted(m_starts_t, beats.contiguous(), right=True).sub_(1).clamp_min_(0)
    raw_seg = ids * max(1, len(grid.meter_starts)) + m_ids
    seg_id = torch.where(slope == 0, raw_seg, -1)
    if boundary_model:
        position = info["bar_phase"] * info["bar_beats"]
        signs = np.sign(np.diff(grid.bpms))
        regime = torch.zeros_like(ids)
        if len(signs) > 1:
            chain = (signs[:-1] == signs[1:]) & (np.diff(grid.beats)[1:] <= 16 + 1e-5)
            state = np.zeros(len(grid.beats), dtype=np.int64)
            state[1:-1] = np.where(chain, np.where(signs[:-1] > 0, 1, np.where(signs[:-1] < 0, 2, 0)), 0)
            regime = torch.as_tensor(state)[ids]
        regime = torch.where(slope > 0, 1, torch.where(slope < 0, 2, regime))
        gold_bar = torch.full((num_frames,), torch.nan, dtype=torch.float64)
        true_times = grid.beat_to_second_tensor(t.new_tensor(bar_marks))
        usable = (true_times >= max(0., (active_start_seconds or 0.) - .02)) & (true_times <= dur_sec)
        gold_bar[(true_times[usable] * FRAME_RATE).long().clamp_max(num_frames - 1)] = torch.tensor(
            bar_marks, dtype=torch.float64)[usable]
        target = dict(eighth=values, beat=gaussian(residual(pulse)), downbeat=gaussian(dt_bar), stop=stop.float(),
                    change=change, pulse_phase=circle(position.remainder(1)), bar_phase=circle(info["bar_phase"]),
                    bar_valid=~stop & active, log_bpm=torch.log2(bpm / 120), active=active, peak=peak,
                    subframe_offset_ms=torch.where(peak, delta * 1000, 0.), score_beat=beats, gold_bar=gold_bar,
                    compound=((info["den"] == 8) & ((info["num"] == 6) | (info["num"] == 12))).float(),
                    compound_valid=(meter != len(meter_table)) & ~stop & active,
                    regime=torch.where(stop, 3, regime), seg_id=seg_id, pulse_coord=position)
        if clean_lattice:
            audible, onset = audio_masks if audio_masks is not None else (active, active)
            score_active = active & audible
            ambiguous = ((info["den"] >= 8) & ~((info["den"] == 8) & (info["num"] % 3 == 0))) | (info["bar_beats"] < 1)
            unverified = _unverified_compound(grid, beats, info, dur_sec)
            ambiguous |= unverified
            target["compound_valid"] &= ~unverified
            target.update(audible=audible.float(), active=score_active, eighth_phase=circle(2*position),
                          pulse_phase=circle(2*position), tactus_valid=~ambiguous | (dt_bar.abs() < .024) | ~score_active,
                          meter_num=info["num"], meter_den=info["den"], bar_span=info["bar_beats"],
                          tactus_step=torch.where(info["den"] == 8, 1.5, 1.), grid_peak=peak & score_active,
                          nearest_delta_ms=delta.nan_to_num(posinf=1.)*1000,
                          has_onset=onset, peak=peak & score_active & onset, bar_valid=score_active & ~stop)
            if clean_v3:
                compound = (info["den"] == 8) & ((info["num"] == 6) | (info["num"] == 12))
                odd = (info["den"] >= 8) & (info["num"] % 2 == 1)
                ambiguous |= compound | odd
                bar_region = dt_bar.abs() < .024
                target.update(meter=meter, pulse_phase=circle(position.remainder(1)),
                              tactus_step=torch.ones_like(position),
                              tactus_valid=~ambiguous | bar_region | ~score_active)
                target["peak"] &= target["tactus_valid"]
                target["grid_peak"] &= target["tactus_valid"]
                del target["eighth_phase"]
            target["stop"] = target["stop"] * inside_audible_bounds(audible)
            for key in ("eighth", "beat", "downbeat", "change"):
                target[key] = target[key] * score_active
            stop_interior = stop & F.pad(stop[:-1], (1, 0))
            for key in ("eighth", "beat", "downbeat"):
                target[key].masked_fill_(stop_interior, 0)
            if clean_v3:
                del target["eighth"]
            target["gold_bar"] = gold_bar.masked_fill(~score_active, torch.nan)
        return target
    pulse_step = torch.where(info["den"] >= 8, 4.0 / info["den"].float(), 1.0)
    bar_span = info["bar_beats"].float().clamp_min(0.25)
    base_origin = m_starts_t[m_ids]
    if len(grid.bar_beats_explicit) > 1:
        bars_t = t.new_tensor(grid.bar_beats_explicit)
        b_idx = torch.searchsorted(bars_t, base_origin.contiguous()).clamp_max(len(bars_t) - 1)
        near_bar = bars_t[b_idx]
        base_origin = torch.where((near_bar - base_origin).abs() < bar_span, near_bar, base_origin)
    seg_entry = torch.where(ids > 0, seg_beats_t[ids], base_origin)
    pulse_anchor = base_origin + torch.round((seg_entry - base_origin) / pulse_step) * pulse_step
    bar_anchor = base_origin + torch.floor((seg_entry - base_origin + 1e-4) / bar_span) * bar_span
    bar_near = base_origin + torch.round((seg_entry - base_origin) / bar_span) * bar_span
    bar_anchor = torch.where((seg_entry - bar_near).abs() < 1e-3, bar_near, bar_anchor)
    pulse_coord = (beats - pulse_anchor) / pulse_step
    bar_coord = (beats - bar_anchor) / bar_span

    return dict(
        beat=values,
        downbeat=gaussian(dt_bar),
        stop=stop.float(),
        change=change,
        pulse_phase=circle(info["pulse_phase"]),
        bar_phase=circle(info["bar_phase"]),
        log_bpm=torch.log2(bpm / 120),
        meter=meter,
        active=active,
        bar_valid=(info["bar_beats"] > 1) & ~stop & active,
        peak=peak,
        subframe_offset_ms=torch.where(peak, delta * 1000, 0.0),
        seg_id=seg_id,
        pulse_coord=pulse_coord,
        bar_coord=bar_coord,
        pulse_step=pulse_step,
        bar_span=bar_span,
    )


@torch.no_grad()
def _boundary_harmonics(outputs, targets):
    """Allow coherent half/double pulse interpretations while retaining the annotated bar sequence."""
    moving = outputs["valid_mask"] & targets["active"] & ~targets["stop"].bool() & (targets["seg_id"] >= 0)
    quarter = "eighth_logit" not in outputs
    pulses = ("beat",) if quarter else ("eighth", "beat")
    zero = outputs["beat_logit"].new_zeros(())
    if "mask_aug_frames" in outputs:
        moving = moving & ~outputs["mask_aug_frames"]
    with_bpm = "log_bpm" in outputs
    if with_bpm and not ((outputs["log_bpm"] - targets["log_bpm"]).abs().gt(.35) & moving).any():
        return targets, zero
    keys = pulses + ("pulse_phase",) + (("log_bpm",) if with_bpm else ())
    pred = {k: outputs[k][moving].float() for k in ("eighth_logit", "beat_logit", "pulse_phase", "log_bpm") if k in outputs}
    batch, _ = moving.nonzero(as_tuple=True)
    groups, ids = torch.unique(torch.stack((batch, targets["seg_id"][moving]), 1), dim=0, return_inverse=True)
    count = torch.bincount(ids, minlength=len(groups)).float()

    def means(values):
        return values.new_zeros(len(groups)).scatter_add_(0, ids, values) / count

    def cost(y):
        total = F.smooth_l1_loss(pred["pulse_phase"], y["pulse_phase"], reduction="none").mean(-1)
        if with_bpm:
            total = total + F.smooth_l1_loss(pred["log_bpm"], y["log_bpm"], beta=.05, reduction="none")
        for key in pulses:
            p = y[key]
            entropy = torch.special.entr(p) + torch.special.entr(1 - p)
            total += (1 + 5 * p) * (F.binary_cross_entropy_with_logits(pred[key + "_logit"], p, reduction="none") - entropy)
        return means(total)

    base = {k: targets[k][moving] for k in keys}
    unit = F.normalize(pred["pulse_phase"], dim=-1)
    coherence = means((unit * base["pulse_phase"]).sum(-1))
    if not with_bpm and not (coherence < .70).any():
        return targets, zero
    copied = (*keys, "peak") if "tactus_valid" in targets else keys
    resolved = {k: v.clone() if k in copied else v for k, v in targets.items()}
    best = cost(base)
    truth_bpm = targets["log_bpm"][moving]
    sec = .5 * (-truth_bpm).exp2()
    slack = torch.zeros_like(count)
    for ratio in (.5, 2.):
        coord = targets["pulse_coord"][moving] * ratio
        clean = "tactus_valid" in targets
        angle = (4 if clean and not quarter else 2) * math.pi * coord
        phase = torch.stack((angle.sin(), angle.cos()), -1)
        def pulse(scale):
            dt = (coord * scale - (coord * scale).round()) * sec / (ratio * scale)
            return torch.maximum(targets["downbeat"][moving], (-.5 * (dt * FRAME_RATE / .8).square()).exp())
        y = dict(beat=pulse(1 / targets["tactus_step"][moving] if clean else 1),
                 pulse_phase=phase, log_bpm=truth_bpm + math.log2(ratio))
        if not quarter:
            y["eighth"] = pulse(2)
        fit = means((unit * phase).sum(-1))
        candidate = cost(y) + .05
        accept = ((candidate < best) & (fit > .55) & (fit > coherence + .25) & (count >= 50))
        accept &= (means(pred["log_bpm"] - y["log_bpm"]).abs() < .22) if with_bpm else (coherence < .70)
        if clean:
            # Keep ambiguous interiors unlabelled and the annotated high-speed clock
            # whenever a nearby acoustic attack supports either nested pulse.
            accept &= means((~targets["tactus_valid"][moving]).float()) == 0
            if ratio == .5:
                protected = ((truth_bpm >= math.log2(200/120)) & targets["has_onset"][moving]
                             & (base[pulses[0]] > .4))
                accept &= means(protected.float()) == 0
        chosen = accept[ids]
        if clean:
            resolved["peak"][moving] &= ~chosen
        best = torch.where(accept, candidate, best)
        slack = torch.where(accept, .05, slack)
        for key in keys:
            resolved[key][moving] = torch.where(chosen[:, None] if key == "pulse_phase" else chosen,
                                                y[key], resolved[key][moving])
    if "eighth_phase" in resolved:
        resolved["eighth_phase"] = resolved["pulse_phase"]
    return resolved, (slack * count).sum()


@torch.no_grad()
def _resolve_harmonic_targets(outputs, targets):
    if "eighth_logit" in outputs or "audible_logit" in outputs:
        return _boundary_harmonics(outputs, targets)
    zero_offset = outputs["log_bpm"].new_zeros(())
    if "seg_id" not in targets:
        return targets, zero_offset
    valid = outputs["valid_mask"]
    active = valid & targets["active"] if "active" in targets else valid
    moving = active & ~targets["stop"].bool()
    pred_log_bpm = outputs["log_bpm"].detach()
    bpm_diff = (pred_log_bpm - targets["log_bpm"]).abs()
    far = moving & (bpm_diff > 0.35)
    if not far.any():
        return targets, zero_offset

    def gaussian(dt):
        return (-0.5 * (dt * FRAME_RATE / 0.8).square()).exp()

    def circle_mod(u):
        phi = u.remainder(1.0)
        phi = torch.where(1.0 - phi < 1e-6, torch.zeros_like(phi), phi)
        angle = 2 * math.pi * phi
        return torch.stack((angle.sin(), angle.cos()), -1)

    def bce_term(logit, y):
        return (1 + 5 * y) * F.binary_cross_entropy_with_logits(logit, y, reduction="none")

    def entropy_term(y):
        return (1 + 5 * y) * (torch.special.entr(y) + torch.special.entr(1.0 - y))

    meter_table, other_idx = _meter_table_from_logits_dim(outputs["meter_logits"].shape[-1])
    resolved = None
    entropy_slack_offset = 0.0
    batch_size, total_frames = valid.shape
    for b in range(batch_size):
        if not far[b].any():
            continue
        mov_b = moving[b]
        seg_row = targets["seg_id"][b]
        unique_segs = torch.unique(seg_row[mov_b]).tolist()
        seg_records = []
        for sid in unique_segs:
            if sid < 0:
                continue
            mask = mov_b & (seg_row == sid)
            n_mov = int(mask.sum().item())
            idx = torch.where(mask)[0]
            f_start, f_end = int(idx[0].item()), int(idx[-1].item())
            base_log_bpm = float(targets["log_bpm"][b, mask].mean().item())
            base_bpm = 120.0 * (2.0**base_log_bpm)
            base_meter = int(torch.mode(targets["meter"][b, mask]).values.item())
            rec = dict(
                sid=sid,
                mask=mask,
                f_start=f_start,
                f_end=f_end,
                ratio=1.0,
                eff_log_bpm=base_log_bpm,
                eff_meter=base_meter,
                offset=0.0,
                payload=None,
            )
            if n_mov < 50:
                seg_records.append(rec)
                continue
            mean_pred_log = float(pred_log_bpm[b, mask].mean().item())
            eligible = []
            for r, slack in HARMONIC_LOSS_CANDIDATES:
                cand_bpm = r * base_bpm
                if not (45.0 <= cand_bpm <= 320.0):
                    continue
                if abs(mean_pred_log - (base_log_bpm + math.log2(r))) > 0.22:
                    continue
                if (n_mov / FRAME_RATE) * (cand_bpm / 60.0) < 1.5:
                    continue
                eligible.append((r, slack))
            if not eligible:
                seg_records.append(rec)
                continue

            v_pulse = outputs["pulse_phase"][b, mask].detach().float()
            norms = v_pulse.norm(dim=-1)
            if float(norms.mean().item()) < 0.40:
                seg_records.append(rec)
                continue
            v_unit = v_pulse / norms.clamp_min(1e-6).unsqueeze(-1)
            coh_1 = float((v_unit * targets["pulse_phase"][b, mask]).sum(-1).mean().item())

            beat_logit = outputs["beat_logit"][b, mask].detach().float()
            down_logit = outputs["downbeat_logit"][b, mask].detach().float()
            v_bar = outputs["bar_phase"][b, mask].detach().float()
            bar_val = targets["bar_valid"][b, mask]
            pred_log_seg = pred_log_bpm[b, mask].float()
            pred_sub = outputs["subframe_offset_ms"][b, mask].detach().float()
            pred_meter_logits = outputs["meter_logits"][b, mask].detach().float()

            def eval_segment_kl(y_beat, y_down, y_pphase, y_bphase, y_log_bpm, y_meter, y_peak, y_sub):
                ent = (entropy_term(y_beat) + entropy_term(y_down)).sum()
                kl = (bce_term(beat_logit, y_beat) + bce_term(down_logit, y_down)).sum() - ent
                total = kl + F.smooth_l1_loss(v_pulse, y_pphase, reduction="none").mean(-1).sum()
                if bar_val.any():
                    total = total + (
                        F.smooth_l1_loss(v_bar, y_bphase, reduction="none").mean(-1) * bar_val
                    ).sum()
                total = total + F.smooth_l1_loss(pred_log_seg, y_log_bpm, beta=0.05, reduction="none").sum()
                m_valid = y_meter != other_idx
                if m_valid.any():
                    total = total + (
                        F.cross_entropy(pred_meter_logits, y_meter, reduction="none") * m_valid
                    ).sum()
                if y_peak.any():
                    total = total + (
                        F.smooth_l1_loss(pred_sub, y_sub, reduction="none") * y_peak
                    ).sum()
                return float(total.item()), float(ent.item())

            kl_1 = ent_1 = None
            best_cost = math.inf
            u_pulse = targets["pulse_coord"][b, mask]
            u_bar = targets["bar_coord"][b, mask]
            p_step = targets["pulse_step"][b, mask]
            b_span = targets["bar_span"][b, mask]
            sec_per_quarter = 60.0 / (120.0 * (2.0 ** targets["log_bpm"][b, mask]))

            for r, slack in eligible:
                u_p_r = r * u_pulse
                y_pphase = circle_mod(u_p_r)
                coh_r = float((v_unit * y_pphase).sum(-1).mean().item())
                if coh_r < 0.55 or coh_r <= coh_1 + 0.25:
                    continue
                if kl_1 is None:
                    kl_1, ent_1 = eval_segment_kl(
                        targets["beat"][b, mask],
                        targets["downbeat"][b, mask],
                        targets["pulse_phase"][b, mask],
                        targets["bar_phase"][b, mask],
                        targets["log_bpm"][b, mask],
                        targets["meter"][b, mask],
                        targets["peak"][b, mask],
                        targets["subframe_offset_ms"][b, mask],
                    )
                    best_cost = kl_1

                dt_p = (torch.round(u_p_r) - u_p_r) * (p_step / r) * sec_per_quarter
                y_beat = gaussian(dt_p)
                m_idx = torch.round(u_p_r)
                near = dt_p.abs() <= (0.5 / FRAME_RATE + 1e-6)
                left_ok = torch.ones_like(near)
                right_ok = torch.ones_like(near)
                if n_mov > 1:
                    same_left = (m_idx[1:] == m_idx[:-1]) & ((idx[1:] - idx[:-1]) == 1)
                    left_ok[1:] = ~same_left | (y_beat[1:] > y_beat[:-1])
                    right_ok[:-1] = ~same_left | (y_beat[:-1] >= y_beat[1:])
                y_peak = near & left_ok & right_ok
                y_sub = torch.where(y_peak, dt_p * 1000.0, 0.0)

                r_bar = r
                y_bphase = circle_mod(r * u_bar)
                y_meter = targets["meter"][b, mask]
                pulses_per_bar = float((b_span[0] / p_step[0]).item())
                scaled_ppb = r * pulses_per_bar
                if bar_val.any() and abs(scaled_ppb - round(scaled_ppb)) < 1e-4 and 1 <= round(scaled_ppb) <= 16:
                    alt_bphase = targets["bar_phase"][b, mask]
                    err_r = F.smooth_l1_loss(v_bar[bar_val], y_bphase[bar_val], reduction="none").mean()
                    err_1 = F.smooth_l1_loss(v_bar[bar_val], alt_bphase[bar_val], reduction="none").mean()
                    if float(err_1.item()) + 0.02 < float(err_r.item()):
                        r_bar = 1.0
                        y_bphase = alt_bphase
                        eff_span = float(b_span[0].item()) * r
                        eff_num, eff_den = multiplier_to_meter(eff_span / 4.0)
                        klass = _meter_class_id(eff_num, eff_den, eff_span, meter_table)
                        y_meter = torch.full_like(y_meter, klass)

                u_b_r = r_bar * u_bar
                dt_b = (torch.round(u_b_r) - u_b_r) * (b_span / r_bar) * sec_per_quarter
                y_down = gaussian(dt_b)
                y_log = targets["log_bpm"][b, mask] + math.log2(r)

                cand_kl, _ = eval_segment_kl(
                    y_beat, y_down, y_pphase, y_bphase, y_log, y_meter, y_peak, y_sub
                )
                cand_cost = cand_kl + slack * n_mov
                if cand_cost < best_cost:
                    best_cost = cand_cost
                    eff_m = int(torch.mode(y_meter).values.item())
                    rec["ratio"] = r
                    rec["eff_log_bpm"] = base_log_bpm + math.log2(r)
                    rec["eff_meter"] = eff_m
                    rec["offset"] = slack * n_mov
                    rec["payload"] = (y_beat, y_down, y_pphase, y_bphase, y_log, y_meter, y_peak, y_sub)
            seg_records.append(rec)

        if not any(r["payload"] is not None for r in seg_records):
            continue
        if resolved is None:
            resolved = {
                k: v.clone()
                if k
                in (
                    "beat",
                    "downbeat",
                    "change",
                    "pulse_phase",
                    "bar_phase",
                    "log_bpm",
                    "meter",
                    "peak",
                    "subframe_offset_ms",
                )
                else v
                for k, v in targets.items()
            }
        for rec in seg_records:
            if rec["payload"] is None:
                continue
            entropy_slack_offset += rec["offset"]
            m = rec["mask"]
            (
                resolved["beat"][b, m],
                resolved["downbeat"][b, m],
                resolved["pulse_phase"][b, m],
                resolved["bar_phase"][b, m],
                resolved["log_bpm"][b, m],
                resolved["meter"][b, m],
                resolved["peak"][b, m],
                resolved["subframe_offset_ms"][b, m],
            ) = rec["payload"]

        seg_records.sort(key=lambda x: x["f_start"])
        for left_rec, right_rec in zip(seg_records, seg_records[1:]):
            if left_rec["ratio"] == 1.0 and right_rec["ratio"] == 1.0:
                continue
            if (
                abs(left_rec["eff_log_bpm"] - right_rec["eff_log_bpm"]) < 0.02
                and left_rec["eff_meter"] == right_rec["eff_meter"]
                and right_rec["f_start"] - left_rec["f_end"] <= 4
            ):
                lo = max(0, left_rec["f_end"] - 5)
                hi = min(total_frames, right_rec["f_start"] + 6)
                resolved["change"][b, lo:hi] = 0.0

    if resolved is None:
        return targets, zero_offset
    return resolved, outputs["log_bpm"].new_tensor(entropy_slack_offset)


def quarter_pulse_frame_sum(outputs, targets):
    # Select harmonic targets from the micro heads alone in both stages.
    keys = ("beat_logit", "downbeat_logit", "pulse_phase", "stop_logit", "valid_mask", "mask_aug_frames")
    micro = {k: outputs[k] for k in keys if k in outputs}
    truth, offset = _boundary_harmonics(micro, targets)
    valid = outputs["valid_mask"]
    use_beat = valid & truth["tactus_valid"]
    loss = offset
    for key in ("beat", "downbeat", "stop"):
        y = truth[key]
        entropy = torch.special.entr(y) + torch.special.entr(1 - y)
        kl = F.binary_cross_entropy_with_logits(outputs[key + "_logit"], y, reduction="none") - entropy
        loss = loss + (kl * (1 + 5*y) * (use_beat if key == "beat" else valid)).sum()
    phase = outputs["pulse_phase"]
    moving = use_beat & truth["active"] & ~truth["stop"].bool()
    phase_loss = F.smooth_l1_loss(phase, truth["pulse_phase"], reduction="none").mean(-1)
    loss = loss + ((phase_loss + .1*(phase.norm(dim=-1)-1).square()) * moving).sum()
    return loss, truth


def calibrator_rhythm_frame_sum(outputs, targets):
    if "correction" in outputs:
        return outputs["correction"].loss(outputs, targets)
    if "audible_logit" in outputs and "eighth_logit" not in outputs:
        loss, truth = quarter_pulse_frame_sum(outputs, targets)
        valid = outputs["valid_mask"]
        audible = F.binary_cross_entropy_with_logits(outputs["audible_logit"], truth["audible"], reduction="none")
        peak = valid & truth["peak"]
        if "mask_aug_frames" in outputs:
            peak &= ~outputs["mask_aug_frames"]
        offset = F.smooth_l1_loss(outputs["subframe_offset_ms"], truth["subframe_offset_ms"], reduction="none")
        return loss + .1*(audible * valid).sum() + (offset * peak).sum()
    targets, harmonic_offset = _resolve_harmonic_targets(outputs, targets)
    valid = outputs["valid_mask"]
    active = valid & targets["active"] if "active" in targets else valid
    moving = active & ~targets["stop"].bool()

    def bce_kl(key):
        y = targets[key]
        ent = (torch.special.entr(y) + torch.special.entr(1.0 - y)).detach()
        return (1 + 5 * y) * (
            F.binary_cross_entropy_with_logits(outputs[key + "_logit"], y, reduction="none") - ent
        )

    micro_only = "audible_logit" in outputs and "log_bpm" not in outputs
    binary_keys = ("downbeat", "stop") if micro_only else ("downbeat", "stop", "change")
    loss = sum(bce_kl(key) for key in binary_keys) * valid
    loss = loss + bce_kl("beat") * (valid & targets.get("tactus_valid", valid))
    phases = [("pulse_phase", moving)]
    if not micro_only:
        phases.append(("bar_phase", valid & targets["bar_valid"]))
    for key, mask in phases:
        v = outputs[key]
        phase = F.smooth_l1_loss(v, targets[key], reduction="none").mean(-1)
        loss = loss + (phase + 0.1 * (v.norm(dim=-1) - 1).square()) * mask
    if not micro_only:
        loss = loss + F.smooth_l1_loss(outputs["log_bpm"], targets["log_bpm"], beta=.05, reduction="none") * moving
    if "audible_logit" in outputs:
        loss = loss + bce_kl("eighth") * valid
        loss = loss + 0.1 * F.binary_cross_entropy_with_logits(
            outputs["audible_logit"], targets["audible"], reduction="none") * valid
    elif "eighth_logit" in outputs:
        loss = loss + bce_kl("eighth") * valid
        loss = loss + F.binary_cross_entropy_with_logits(outputs["compound_logit"], targets["compound"],
                                                          reduction="none") * (valid & targets["compound_valid"])
        loss = loss + F.cross_entropy(outputs["regime_logits"].transpose(1, 2), targets["regime"],
                                      reduction="none") * active
        harmonic_offset = harmonic_offset + 0.25 * sequence_loss(outputs, targets)
    else:
        meter = targets["meter"]
        _, other_idx = _meter_table_from_logits_dim(outputs["meter_logits"].shape[-1])
        loss = loss + F.cross_entropy(outputs["meter_logits"].transpose(1, 2), meter, reduction="none") * (
            moving & (meter != other_idx)
        )
    offset_mask = valid & targets["peak"]
    if "mask_aug_frames" in outputs:
        offset_mask &= ~outputs["mask_aug_frames"]
    loss = loss + F.smooth_l1_loss(
        outputs["subframe_offset_ms"], targets["subframe_offset_ms"], reduction="none"
    ) * offset_mask
    return loss.sum() + harmonic_offset


def calibrator_loss(outputs, targets, ref_batch_frames=None):
    rhythm = calibrator_rhythm_frame_sum(outputs, targets)
    denominator = outputs["valid_mask"].sum().clamp_min(1) if ref_batch_frames is None else ref_batch_frames
    return rhythm / denominator


def _peaks(value, threshold, radius):
    pooled = F.max_pool1d(value[None, None], 2 * radius + 1, 1, radius).flatten()
    keep = (value >= threshold) & (value == pooled)
    # Pick the leading frame of a tied peak.
    return torch.where(keep & ~F.pad(keep[:-1], (1, 0)))[0]


def _runs(mask):
    edges = F.pad(mask.to(torch.int8), (1, 1)).diff()
    return zip(torch.where(edges == 1)[0].tolist(), torch.where(edges == -1)[0].tolist())


def _circular_time(seconds, period, weights):
    angle = 2 * math.pi * seconds / period
    return (
        float(torch.atan2((weights * angle.sin()).sum(), (weights * angle.cos()).sum()))
        * period
        / (2 * math.pi)
    )


def _other_meter(theta, bar_theta, moving, bpm):
    """Recover less common signatures from circle winding per physical frame."""

    def winding(phase):
        delta = (phase.diff() + 0.5).remainder(1) - 0.5
        valid = moving[1:] & moving[:-1]
        return float(delta[valid].median()) if valid.any() else 0.0

    qps = bpm / (60 * FRAME_RATE)
    pulse_rate, bar_rate = winding(theta), winding(bar_theta)
    if pulse_rate <= 0 or bar_rate <= 0:
        return 4, 4
    # Quantize only to supported pulse spacings, including sixteenth-note pulses.
    den = min((4, 8, 16), key=lambda d: abs(qps / pulse_rate - 4 / d))
    num = max(1, min(16, round(qps / bar_rate * den / 4)))
    return num, den


LEGACY_VITERBI_METERS = (
    (4, 4),
    (3, 4),
    (2, 4),
    (1, 4),
    (6, 8),
    (12, 8),
    (5, 4),
    (6, 4),
    (7, 4),
    (7, 8),
    (9, 8),
)

VITERBI_METERS = LEGACY_VITERBI_METERS + tuple(m for m in EXTENDED_METERS if m not in LEGACY_VITERBI_METERS)

VALID_DECODER_PRESETS = ("auto", "simple", "medium", "complex")


def _decode_stops(stop_prob, stop_threshold=0.85, min_stop_frames=3):
    """Build stop mask and metadata list using positive-weight-calibrated runs."""
    stopped = torch.zeros_like(stop_prob, dtype=torch.bool)
    min_frames = max(1, int(min_stop_frames))
    for sa, sb in _runs(stop_prob >= stop_threshold):
        if sb - sa >= min_frames:
            stopped[sa:sb] = True
    stops = [dict(time=a / FRAME_RATE, duration=(b - a) / FRAME_RATE) for a, b in _runs(stopped)]
    return stopped, stops


def _first_music_frame(beat, down, stopped, pulse_norm):
    """Locate the earliest unstopped beat, downbeat, or high-confidence pulse frame."""
    peaks = _peaks(beat, 0.45, 2)
    peaks = peaks[~stopped[peaks]] if len(peaks) else peaks
    down_ids = _peaks(down, 0.5, 2)
    down_ids = down_ids[~stopped[down_ids]] if len(down_ids) else down_ids
    candidates = []
    if len(peaks):
        candidates.append(int(peaks[0]))
    if len(down_ids):
        candidates.append(int(down_ids[0]))
    if candidates:
        return min(candidates), peaks, down_ids
    active_norm = torch.where(~stopped & (pulse_norm >= 0.35))[0]
    if len(active_norm):
        return int(active_norm[0]), peaks, down_ids
    return None, peaks, down_ids


def _segment_peak_mask(bp, phase_mag, active_seg, thresh=0.35, min_norm=0.25):
    """Return local peak indices inside a segment passing confidence thresholds."""
    if len(bp) == 0:
        return torch.empty(0, dtype=torch.long)
    pks = _peaks(bp, thresh, 2)
    if len(pks) == 0:
        return pks
    keep = active_seg[pks] & (phase_mag[pks] >= min_norm)
    return pks[keep]


def _estimate_segment_bpm(
    a, b, theta, phase_mag, beat, sub, bpm_track, running, active, step=1.0, seed_bpm=None, half_beats=None
):
    """Estimate segment BPM via masked phase OLS, 1x/2x periodogram, and IRLS."""
    act_idx = torch.where(active[a:b])[0] + a
    if len(act_idx) < 4:
        med = float(seed_bpm) if seed_bpm is not None else (float(bpm_track[a:b].median()) if b > a else 120.0)
        return med, torch.empty(0, dtype=torch.float64), torch.empty(0, dtype=torch.float64)

    mag_a = phase_mag[act_idx]
    mag_thresh = max(0.25, float(mag_a.quantile(0.25)))
    good = act_idx[mag_a >= mag_thresh]
    if len(good) < 4:
        good = act_idx
    bpm_med = float(seed_bpm) if seed_bpm is not None else float(bpm_track[good].median())

    # Fit phase slope only across contiguous unstopped runs so holes cannot inject cycle slips.
    splits = torch.where((act_idx[1:] - act_idx[:-1]) > 1)[0] + 1
    num_sum, den_sum = 0.0, 0.0
    prev = 0
    for s in splits.tolist() + [len(act_idx)]:
        if s - prev >= 4:
            run = act_idx[prev:s]
            delta = (theta[run].double().diff() + 0.5).remainder(1.0) - 0.5
            angles = F.pad(delta.cumsum(0), (1, 0))
            tr = running[run]
            wr = phase_mag[run].double().clamp(0.1, 1.0)
            wr = wr / wr.sum()
            tm = (wr * tr).sum()
            am = (wr * angles).sum()
            num_sum += float((wr * (tr - tm) * (angles - am)).sum())
            den_sum += float((wr * (tr - tm) ** 2).sum())
        prev = s
    bpm_phase = (60.0 * step * num_sum / den_sum) if den_sum > 1e-6 and num_sum > 0 else bpm_med

    pk_local = _segment_peak_mask(beat[a:b], phase_mag[a:b], active[a:b], 0.35, 0.25)
    if len(pk_local) < 4:
        pk_local = _segment_peak_mask(beat[a:b], phase_mag[a:b], active[a:b], 0.15, 0.0)
    pk_global = pk_local + a
    peak_t = (running[pk_global] + sub[pk_global]).double()
    peak_w = (beat[pk_global] * phase_mag[pk_global].clamp(0.1, 1.0)).double()

    if len(peak_t) < 4 or float(peak_t[-1] - peak_t[0]) < 0.5:
        if (
            seed_bpm is None
            and 25.0 <= bpm_phase <= 480.0
            and abs(bpm_phase - bpm_med) < 0.15 * max(bpm_med, 1.0)
        ):
            return bpm_phase, peak_t, peak_w
        return bpm_med, peak_t, peak_w

    span = float(peak_t[-1] - peak_t[0])
    centers = [bpm_med]
    if (
        seed_bpm is None
        and 25.0 <= bpm_phase <= 480.0
        and abs(bpm_phase - bpm_med) / max(bpm_med, 1.0) < 0.25
    ):
        centers.append(bpm_phase)
    c_lo = max(25.0, min(centers) * 0.94)
    c_hi = min(480.0, max(centers) * 1.06)
    d_bpm = max(0.002, min(0.05, 0.12 * 60.0 * step / max(span, 1.0)))
    grid_bpm = torch.arange(c_lo, c_hi + 0.5 * d_bpm, d_bpm, dtype=torch.float64)
    pw64 = peak_w / peak_w.sum().clamp_min(1e-9)
    peak_shift = torch.zeros_like(peak_t)
    if half_beats is not None and len(half_beats[0]):
        hb_t, hb_mod = half_beats
        idx_r = np.clip(np.searchsorted(hb_t, peak_t.numpy()), 0, len(hb_t) - 1)
        idx_l = np.maximum(0, idx_r - 1)
        choose_l = np.abs(peak_t.numpy() - hb_t[idx_l]) <= np.abs(hb_t[idx_r] - peak_t.numpy())
        nearest = np.where(choose_l, idx_l, idx_r)
        peak_shift = torch.from_numpy(hb_mod[nearest]).to(peak_t.dtype)

    best_bpm = bpm_med
    best_score = -1.0
    best_phi0 = 0.0
    for chunk in grid_bpm.split(256):
        periods = (60.0 * step) / chunk[:, None]
        ph = (peak_t[None, :] / periods - peak_shift[None, :]).remainder(1.0)
        ang1 = 2.0 * math.pi * ph
        phi1 = (
            torch.atan2((pw64[None] * torch.sin(ang1)).sum(-1), (pw64[None] * torch.cos(ang1)).sum(-1))
            / (2.0 * math.pi)
        ).remainder(1.0)
        ang2 = 4.0 * math.pi * ph
        phi2 = (
            torch.atan2((pw64[None] * torch.sin(ang2)).sum(-1), (pw64[None] * torch.cos(ang2)).sum(-1))
            / (4.0 * math.pi)
        ).remainder(0.5)
        cands = torch.stack([phi1, phi2, (phi2 + 0.5).remainder(1.0)], dim=1)
        diff = (ph[:, None, :] - cands[:, :, None] + 0.5).remainder(1.0) - 0.5
        scores = (torch.exp(-0.5 * (diff / 0.08) ** 2) * pw64[None, None, :]).sum(-1)
        max_sc, idx_k = scores.max(dim=1)
        top_sc, top_i = max_sc.max(dim=0)
        if float(top_sc) > best_score:
            best_score = float(top_sc)
            best_bpm = float(chunk[top_i])
            best_phi0 = float(cands[top_i, idx_k[top_i]])

    bpm_cur = best_bpm
    phi0 = best_phi0
    for _ in range(5):
        period = 60.0 * step / bpm_cur
        n_idx = torch.round(peak_t / period - peak_shift - phi0) + peak_shift
        res_cycles = (peak_t / period - phi0) - n_idx
        w_irl = pw64 * torch.exp(-0.5 * (res_cycles / 0.10) ** 2)
        if float(w_irl.sum()) <= 1e-9:
            break
        w_irl = w_irl / w_irl.sum()
        nm = (w_irl * n_idx).sum()
        tm = (w_irl * peak_t).sum()
        denom = float((w_irl * (n_idx - nm) ** 2).sum())
        if denom < 1e-6:
            break
        period_new = float((w_irl * (n_idx - nm) * (peak_t - tm)).sum()) / denom
        if period_new <= 0:
            break
        bpm_new = 60.0 * step / period_new
        if abs(bpm_new - best_bpm) / best_bpm > 0.03:
            break
        bpm_cur = bpm_new
        t_anchor = float(tm - period_new * nm)
        phi0 = (t_anchor / period_new) % 1.0
    if half_beats is not None and len(peak_t):
        peak_t = peak_t - peak_shift * (60.0 * step / bpm_cur)
    return bpm_cur, peak_t, peak_w


def _closed_loop_peak_residuals(peak_t, peak_w, bpm, step=1.0):
    """Evaluate closed-loop peak alignment residuals and early-vs-late drift at bpm."""
    if len(peak_t) == 0:
        return None
    period = 60.0 * step / bpm
    ang = 2.0 * math.pi * (peak_t / period).remainder(1.0)
    pw = peak_w / peak_w.sum().clamp_min(1e-9)
    phi0 = (
        float(torch.atan2((pw * torch.sin(ang)).sum(), (pw * torch.cos(ang)).sum())) / (2.0 * math.pi)
    ) % 1.0
    res_cycles = ((peak_t / period - phi0 + 0.5).remainder(1.0)) - 0.5
    res_ms = res_cycles * period * 1000.0
    abs_ms = res_ms.abs()
    drift_ms = 0.0
    if len(peak_t) >= 8 and float(peak_t[-1] - peak_t[0]) > 2.0:
        q = max(2, len(peak_t) // 4)
        drift_ms = abs(float(res_ms[-q:].median()) - float(res_ms[:q].median()))
    return dict(
        phi0=phi0,
        res_ms=res_ms,
        wmae_ms=float((pw * abs_ms).sum()),
        median_abs_ms=float(abs_ms.median()),
        drift_ms=drift_ms,
    )


def _should_merge_segments(
    left, right, theta, phase_mag, beat, sub, bpm_track, running, active, change_prob
):
    """Decide whether adjacent tempo segments represent the same BPM without phase slip."""
    a1, b1, bpm1, pt1, pw1, step1 = left
    a2, b2, bpm2, pt2, pw2, step2 = right
    if abs(step1 - step2) > 1e-6:
        return False, None
    rel_diff = abs(bpm1 - bpm2) / max(min(bpm1, bpm2), 1.0)
    if rel_diff > 0.12 and min(len(pt1), len(pt2)) >= 6:
        return False, None
    bpm_c, pt_c, pw_c = _estimate_segment_bpm(
        a1, b2, theta, phase_mag, beat, sub, bpm_track, running, active, step=step1
    )
    if len(pt_c) < 6:
        if rel_diff < 0.02:
            return True, (a1, b2, bpm_c, pt_c, pw_c, step1)
        return False, None

    rc = _closed_loop_peak_residuals(pt_c, pw_c, bpm_c, step=step1)
    mid_t = float(running[b1])
    m1 = pt_c < mid_t
    m2 = pt_c >= mid_t
    if int(m1.sum()) < 3 or int(m2.sum()) < 3:
        if rc["wmae_ms"] <= 28.0 or rel_diff <= 0.03:
            return True, (a1, b2, bpm_c, pt_c, pw_c, step1)
        return False, None

    r1 = _closed_loop_peak_residuals(pt1, pw1, bpm1, step=step1)
    r2 = _closed_loop_peak_residuals(pt2, pw2, bpm2, step=step1)
    period_c = 60.0 * step1 / bpm_c
    cycles_c = ((pt_c / period_c - rc["phi0"] + 0.5).remainder(1.0)) - 0.5
    res_c_ms = cycles_c * period_c * 1000.0
    med_err_1 = float(res_c_ms[m1].abs().median())
    med_err_2 = float(res_c_ms[m2].abs().median())
    phase_jump_ms = abs(float(res_c_ms[m2].median()) - float(res_c_ms[m1].median()))

    w1, w2 = float(pw1.sum()), float(pw2.sum())
    sep_wmae = (w1 * r1["wmae_ms"] + w2 * r2["wmae_ms"]) / max(w1 + w2, 1e-9)
    b_lo, b_hi = max(0, b1 - 12), min(len(change_prob), b1 + 13)
    ch_max = float(change_prob[b_lo:b_hi].max())
    max_jump = 18.0 if ch_max >= 0.72 else 30.0
    max_extra_wmae = 7.0 if ch_max >= 0.72 else 14.0
    if (
        phase_jump_ms <= max_jump
        and med_err_1 <= 26.0
        and med_err_2 <= 26.0
        and (rc["wmae_ms"] - sep_wmae) <= max_extra_wmae
    ):
        return True, (a1, b2, bpm_c, pt_c, pw_c, step1)

    # Guard against eighth-note double-peak aliasing splitting an almost-identical tempo.
    frac_half_1 = float((cycles_c[m1].abs() > 0.30).float().mean())
    frac_half_2 = float((cycles_c[m2].abs() > 0.30).float().mean())
    if rel_diff <= 0.008 and max(frac_half_1, frac_half_2) >= 0.15:
        res_half = (((pt_c / (0.5 * period_c) - 2.0 * rc["phi0"] + 0.5).remainder(1.0)) - 0.5) * (0.5 * period_c) * 1000.0
        h_err_1 = float(res_half[m1].abs().median())
        h_err_2 = float(res_half[m2].abs().median())
        h_jump = abs(float(res_half[m2].median()) - float(res_half[m1].median()))
        if h_jump <= 20.0 and h_err_1 <= 34.0 and h_err_2 <= 34.0:
            return True, (a1, b2, bpm_c, pt_c, pw_c, step1)
    return False, None


def _merge_tempo_segments(segs, theta, phase_mag, beat, sub, bpm_track, running, active, change_prob):
    """Iteratively merge spurious tempo splits caused by meter changes or drum fills."""
    cur = list(segs)
    merged = True
    while merged and len(cur) > 1:
        merged = False
        nxt = []
        i = 0
        while i < len(cur):
            if i + 1 < len(cur):
                ok, combined = _should_merge_segments(
                    cur[i],
                    cur[i + 1],
                    theta,
                    phase_mag,
                    beat,
                    sub,
                    bpm_track,
                    running,
                    active,
                    change_prob,
                )
                if ok:
                    nxt.append(combined)
                    i += 2
                    merged = True
                    continue
            nxt.append(cur[i])
            i += 1
        cur = nxt
    return cur


def _snap_integer_bpm(bpm, peak_t, peak_w, active_frames, step=1.0, snap_max_drift_ms=2.0):
    """Snap to round(bpm) when closed-loop peak residuals show no systematic drift."""
    nearest = round(bpm)
    if nearest < 1 or abs(bpm - nearest) > max(0.35, 0.0025 * bpm):
        return bpm
    seg_seconds = max(1.0 / FRAME_RATE, float(active_frames) / FRAME_RATE)
    drift_ms = 1000.0 * seg_seconds * abs(bpm - nearest) / max(nearest, 1)
    if len(peak_t) < 6:
        return float(nearest) if (abs(bpm - nearest) < 0.15 and drift_ms <= snap_max_drift_ms) else bpm
    r_raw = _closed_loop_peak_residuals(peak_t, peak_w, bpm, step=step)
    r_int = _closed_loop_peak_residuals(peak_t, peak_w, float(nearest), step=step)
    if r_int["drift_ms"] > max(22.0, 2.2 * max(r_raw["drift_ms"], 8.0)):
        return bpm
    if (
        r_int["wmae_ms"] <= max(r_raw["wmae_ms"] + 3.5, 1.15 * r_raw["wmae_ms"])
        and r_int["median_abs_ms"] <= 22.0
    ):
        return float(nearest)
    return bpm


def _track_quarter_beats(running, beat, down, sub, bpm_track, meter_scores, stopped, first_frame):
    """Joint beat + downbeat pulse-tracking DP on unstopped running time."""
    n = len(running)
    not_stopped = ~stopped
    bp_ids = _peaks(beat, 0.14, 3)
    bp_ids = bp_ids[not_stopped[bp_ids]] if len(bp_ids) else bp_ids
    dp_ids = _peaks(down, 0.30, 4)
    dp_ids = dp_ids[not_stopped[dp_ids]] if len(dp_ids) else dp_ids

    cand_set = set(bp_ids.tolist())
    bp_arr_set = np.array(sorted(cand_set), dtype=np.int64)
    for d_idx in dp_ids.tolist():
        if float(beat[d_idx]) < 0.05:
            continue
        if len(bp_arr_set) == 0 or np.min(np.abs(bp_arr_set - d_idx)) > 3:
            cand_set.add(d_idx)
    cand_idx = np.array(sorted(cand_set), dtype=np.int64)
    if first_frame is not None:
        cand_idx = cand_idx[cand_idx >= max(0, int(first_frame) - 3)]
    if len(cand_idx) < 2:
        return None

    run_np = running.numpy()
    sub_np = sub.numpy()
    beat_np = beat.numpy()
    down_np = down.numpy()
    bpm_tr_np = bpm_track.numpy()
    m_probs_np = meter_scores.softmax(-1).numpy()
    meters, _ = _meter_table_from_logits_dim(meter_scores.shape[-1])
    eighth_cols = [k for k, (_, den) in enumerate(meters) if den == 8]
    p_eighth_np = m_probs_np[:, eighth_cols].sum(axis=-1)
    # Retain the released head's scoring outside sustained 2/4. The expanded
    # head supports local 2/4 changes as well.
    p_24_np = m_probs_np[:, 2]
    revised_tracker = len(meters) > len(METERS) or p_24_np.mean() >= 0.35
    p_34_np = m_probs_np[:, 1]
    p_54_np = m_probs_np[:, 5]

    v3 = meters == EXTENDED_METERS
    if (3, 8) in meters:
        odd_cols = [idx for idx, (num, den) in enumerate(meters) if den == 8 and num % 2]
        shift_prob = m_probs_np[:, odd_cols].sum(axis=1)

    cand_t = run_np[cand_idx] + sub_np[cand_idx]
    keep = np.ones(len(cand_t), dtype=bool)
    for idx in range(1, len(cand_t)):
        if cand_t[idx] <= cand_t[idx - 1] + 0.02:
            if max(beat_np[cand_idx[idx]], down_np[cand_idx[idx]]) > max(
                beat_np[cand_idx[idx - 1]], down_np[cand_idx[idx - 1]]
            ):
                keep[idx - 1] = False
            else:
                keep[idx] = False
    cand_idx = cand_idx[keep]
    cand_t = cand_t[keep]
    if len(cand_t) < 2:
        return None

    active_cands = np.where(
        (beat_np[cand_idx] >= 0.50) | ((down_np[cand_idx] >= 0.50) & (beat_np[cand_idx] >= 0.05))
    )[0]
    if len(active_cands) >= 2:
        last_active_t = cand_t[active_cands[-1]]
        cand_keep = cand_t <= last_active_t + 1.0
        if cand_keep.sum() >= 2:
            cand_idx = cand_idx[cand_keep]
            cand_t = cand_t[cand_keep]

    cand_bp = beat_np[cand_idx]
    cand_dp = down_np[cand_idx]
    cand_tr = bpm_tr_np[cand_idx]
    cand_p8 = p_eighth_np[cand_idx]

    dom_db_ids = _peaks(down, 0.68, 56)
    dom_db_ids = dom_db_ids[not_stopped[dom_db_ids]].numpy()
    strong_db_t = run_np[dom_db_ids] + sub_np[dom_db_ids]
    strong_db_p = down_np[dom_db_ids]

    c_len = len(cand_t)
    min_dt, max_dt, max_span = 0.165, 1.500, 2.600
    n_bins = 40
    log_min, log_max = math.log(min_dt), math.log(max_dt)

    def dt_bin(dt):
        pos = (math.log(dt) - log_min) / (log_max - log_min) * (n_bins - 1)
        return max(0, min(n_bins - 1, int(round(pos))))

    states = [{} for _ in range(c_len)]
    for j in range(min(c_len, 20)):
        missed_pen = 0.25 * float(cand_bp[:j].sum()) if j > 0 else 0.0
        for k in range(j + 1, min(c_len, j + 16)):
            span = float(cand_t[k] - cand_t[j])
            if span < min_dt:
                continue
            if span > max_span:
                break
            for m in (1, 2):
                dt = span / m
                if not (min_dt <= dt <= max_dt) or (m == 2 and cand_bp[j] < 0.30):
                    continue
                inst_bpm = 60.0 / dt
                tr_bpm = 0.5 * (cand_tr[j] + cand_tr[k])
                ratio = inst_bpm / max(tr_bpm, 25.0)
                p8_loc = 0.5 * (cand_p8[j] + cand_p8[k])
                p24_loc = 0.5 * (p_24_np[cand_idx[j]] + p_24_np[cand_idx[k]]) if revised_tracker else 0.0
                oct_pen = 0.0
                if ratio > 1.75:
                    if p8_loc >= 0.35 or p24_loc >= 0.35 or inst_bpm >= 260.0:
                        oct_pen = max(0.85, 2.4 * span)
                    elif tr_bpm >= 135.0:
                        oct_pen = 1.5 * span
                elif ratio < 0.58 and tr_bpm >= 130.0:
                    oct_pen = 0.8 * span
                anchor_pen = 0.02 * span * (math.log(max(ratio, 0.25)) ** 2)
                db_bonus = 0.45 * (cand_dp[j] + cand_dp[k]) if p8_loc < 0.45 else 0.0
                sc = (
                    (cand_bp[j] - 0.30)
                    + (cand_bp[k] - 0.30)
                    + 0.4 * span
                    + db_bonus
                    - missed_pen
                    - anchor_pen
                    - oct_pen
                    - 0.25 * (m - 1)
                )
                b = dt_bin(dt)
                cur = states[k].get(b)
                if cur is None or sc > cur[0]:
                    states[k][b] = (sc, j, -1, dt, m)

    for j in range(1, c_len):
        st_j = states[j]
        if not st_j:
            sc_re = 0.2 * float(cand_bp[j])
            for k in range(j + 1, min(c_len, j + 16)):
                span = float(cand_t[k] - cand_t[j])
                if span < min_dt:
                    continue
                if span > max_span:
                    break
                if min_dt <= span <= max_dt:
                    b = dt_bin(span)
                    sc = sc_re + (cand_bp[k] - 0.30) + 0.4 * span
                    cur = states[k].get(b)
                    if cur is None or sc > cur[0]:
                        states[k][b] = (sc, j, -1, span, 1)
            continue

        top_items = (
            sorted(st_j.items(), key=lambda kv: kv[1][0], reverse=True)[:6]
            if len(st_j) > 6
            else list(st_j.items())
        )
        t_j = float(cand_t[j])
        dp_j = float(cand_dp[j])
        f_j = int(cand_idx[j])

        for k in range(j + 1, min(c_len, j + 24)):
            t_k = float(cand_t[k])
            span = t_k - t_j
            if span < (min_dt * .5 if v3 else min_dt):
                continue
            if span > max_span:
                break
            f_k = int(cand_idx[k])
            bp_k = float(cand_bp[k])
            dp_k = float(cand_dp[k])
            tr_bpm = 0.5 * (float(cand_tr[j]) + float(cand_tr[k]))
            p8_loc = float(p_eighth_np[f_j : f_k + 1].mean()) if f_k > f_j else float(cand_p8[k])
            p24_loc = float(p_24_np[f_j : f_k + 1].mean()) if revised_tracker else 0.0
            allow_db = p8_loc < 0.45
            p34_loc = float(p_34_np[f_j : f_k + 1].mean()) if f_k > f_j else float(p_34_np[f_k])
            p54_loc = float(p_54_np[f_j : f_k + 1].mean()) if f_k > f_j else float(p_54_np[f_k])

            in_dbs = []
            if allow_db and len(strong_db_t):
                d_lo = int(np.searchsorted(strong_db_t, t_j + 0.035, side="right"))
                d_hi = int(np.searchsorted(strong_db_t, t_k - 0.035, side="left"))
                if d_hi > d_lo:
                    in_dbs = [(float(strong_db_t[u]), float(strong_db_p[u])) for u in range(d_lo, d_hi)]

            mid_bp = 0.0
            if span >= 0.42 and p8_loc < 0.40 and (f_k - f_j) >= 12:
                m_lo = f_j + int(0.38 * (f_k - f_j))
                m_hi = f_j + int(0.62 * (f_k - f_j)) + 1
                if m_hi > m_lo:
                    mid_bp = float(beat_np[m_lo:m_hi].max())

            shift_boundary = (v3 and max(dp_j, dp_k) >= .45
                              and shift_prob[max(0, f_j-4):min(n, f_k+5)].max() >= .30)
            both_db = allow_db and (dp_j >= 0.45) and (dp_k >= 0.45)
            both_db_7 = allow_db and (min(dp_j, dp_k) >= 0.36) and (max(dp_j, dp_k) >= 0.50)
            for m in (1, 2, 3, 4, 7):
                if m == 7 and not both_db_7:
                    continue
                dt = span / m
                regular_period = min_dt <= dt <= max_dt
                can_shift = m == 1 and shift_boundary
                if not regular_period and not can_shift:
                    continue
                inst_bpm = 60.0 / dt
                ratio = inst_bpm / max(tr_bpm, 25.0)
                oct_pen = 0.0
                if ratio > 1.75:
                    if p8_loc >= 0.35 or p24_loc >= 0.35 or inst_bpm >= 260.0:
                        oct_pen = max(0.85, 2.4 * span)
                    elif tr_bpm >= 135.0:
                        oct_pen = 1.8 * span
                elif ratio < 0.58 and tr_bpm >= 130.0:
                    oct_pen = 0.9 * span
                anchor_pen = 0.012 * span * (math.log(max(ratio, 0.25)) ** 2)

                db_skip_pen = 0.0
                if in_dbs:
                    for dt_db, dp_db in in_dbs:
                        if m == 1:
                            db_skip_pen += 1.8 * (dp_db - 0.50)
                        else:
                            r_near = max(1, min(m - 1, round((dt_db - t_j) / dt)))
                            if abs((t_j + r_near * dt) - dt_db) > 0.035:
                                db_skip_pen += 1.8 * (dp_db - 0.50)

                inter_sum = 0.0
                if m > 1:
                    for r in range(1, m):
                        t_r = t_j + r * dt
                        f_r = max(0, min(n - 1, int(np.searchsorted(run_np, t_r))))
                        lo = max(0, f_r - 2)
                        hi = min(n, f_r + 3)
                        inter_sum += float(beat_np[lo:hi].max()) if hi > lo else 0.0
                inter_mean = inter_sum / max(1, m - 1) if m > 1 else 1.0

                db_bonus = 0.0
                if allow_db:
                    db_bonus = 0.50 * dp_k if dp_k >= 0.45 else 0.15 * dp_k
                    if (both_db and m == 4 and (not revised_tracker or ratio <= 1.55)
                            and max(p24_loc, p34_loc, p54_loc) < 0.35):
                        if span <= 0.90 and inter_mean >= 0.25:
                            db_bonus += 1.55 * min(dp_j, dp_k)
                        elif span <= 1.00 and inter_mean >= 0.28:
                            db_bonus += 0.75 * min(dp_j, dp_k)
                    elif both_db_7 and m == 7 and span >= 1.35 and inter_mean >= 0.25:
                        db_bonus += 2.05 * max(0.45, min(dp_j, dp_k))
                    elif both_db and m == 2 and inter_mean >= 0.25:
                        db_bonus += (0.75 if p24_loc >= 0.35 else 0.45) * min(dp_j, dp_k)

                offgrid_pen = 0.0
                if m == 1:
                    half_pen = (
                        1.6 * span * max(0.0, mid_bp - 0.45) if (span >= 0.42 and p8_loc < 0.40) else 0.0
                    )
                    emit = (bp_k - 0.30) + 0.40 * span + db_bonus - half_pen - db_skip_pen
                    use_low_skip = False
                else:
                    use_low_skip = (
                        (both_db and m == 4 and span <= 1.00 and inter_mean >= 0.28)
                        or (both_db and m == 4 and span <= 0.90 and inter_mean >= 0.25)
                        or (both_db_7 and m == 7 and span >= 1.35 and inter_mean >= 0.25)
                        or (both_db and m == 2 and inter_mean >= 0.25)
                    )
                    if (
                        not use_low_skip
                        and not (both_db and m == 4 and span <= 0.845 and inter_mean >= 0.18)
                        and inter_mean < 0.24
                        and k - j > 1
                    ):
                        bp_cut = 0.45 if (m == 4 and span <= 1.00) else 0.60
                        for u in range(j + 1, k):
                            if cand_bp[u] >= bp_cut:
                                r_u = max(0, min(m, round((float(cand_t[u]) - t_j) / dt)))
                                if abs((t_j + r_u * dt) - float(cand_t[u])) > 0.035:
                                    offgrid_pen += 0.50 if bp_cut == 0.45 else 0.45
                    skip_unit = 0.04 if use_low_skip else 0.22
                    inter_wt = 0.42 if (both_db_7 and m == 7 and use_low_skip) else 0.28
                    emit = (
                        (bp_k - 0.30)
                        + 0.40 * span
                        + db_bonus
                        + inter_wt * inter_sum
                        - skip_unit * (m - 1)
                        - db_skip_pen
                        - offgrid_pen
                    )

                nominal_bin = dt_bin(dt)
                for _b_old, (sc_old, _, _, dt_old, _) in top_items:
                    unit, b_new, octave_cost = dt, nominal_bin, oct_pen
                    if can_shift:
                        relative = dt / dt_old
                        half = .5 if relative < 1 else 1.5
                        if abs(relative - half) <= .12:
                            unit = dt / half
                            b_new = dt_bin(unit)
                            octave_cost = 0.0
                    if not regular_period and not (min_dt <= unit <= max_dt):
                        continue
                    rel_log = abs(math.log(unit / dt_old))
                    jump_pen = 0.0
                    if rel_log > 0.12:
                        if (
                            dt_old <= 0.228
                            and unit > 0.240
                            and not (m == 4 and dp_j >= 0.45 and dp_k >= 0.45 and span <= 0.85)
                        ):
                            jump_pen = 1.6
                        elif bp_k < 0.15 or inter_mean < 0.15:
                            jump_pen = 1.25
                        else:
                            jump_pen = 0.35 if (dp_j >= 0.45 or dp_k >= 0.45) else 1.25
                    tempo_pen = 8.0 * min(rel_log, 0.35) ** 2 + jump_pen
                    hi_bpm_bonus = 0.0
                    if (
                        both_db
                        and m == 4
                        and span <= 0.845
                        and dt_old <= 0.228
                        and (not revised_tracker or ratio <= 1.55)
                        and p24_loc < 0.35
                        and p34_loc < 0.35
                        and p54_loc < 0.35
                    ):
                        # Continue high-BPM 4/4 bar-by-bar accelerando (>=284 BPM) across weak interior beats.
                        hi_bpm_bonus = (3.80 if use_low_skip else 4.20) * min(dp_j, dp_k)
                    sc_new = sc_old + emit + hi_bpm_bonus - anchor_pen - octave_cost - tempo_pen
                    cur = states[k].get(b_new)
                    if cur is None or sc_new > cur[0]:
                        states[k][b_new] = (sc_new, j, _b_old, unit, m)

    best_sc = -1e18
    best_node = (-1, -1)
    for k in range(max(0, c_len - 25), c_len):
        for b, tup in states[k].items():
            sc_tot = tup[0] + 0.05 * cand_t[k]
            if sc_tot > best_sc:
                best_sc = sc_tot
                best_node = (k, b)
    if best_node[0] < 0:
        return None

    path = []
    cur_k, cur_b = best_node
    while cur_k >= 0 and cur_b >= 0:
        tup = states[cur_k][cur_b]
        _, prev_k, prev_b, dt_val, m_step = tup
        path.append((cur_k, prev_k, dt_val, m_step))
        cur_k, cur_b = prev_k, prev_b
    first_k = path[-1][1] if path else best_node[0]
    path.reverse()

    b_times = [float(cand_t[first_k])]
    b_bp = [float(cand_bp[first_k])]
    b_dp = [float(cand_dp[first_k])]
    b_idx = [int(cand_idx[first_k])]
    b_interp = [False]
    for cur_k, prev_k, _dt, m_step in path:
        t0 = float(cand_t[prev_k])
        t1 = float(cand_t[cur_k])
        if m_step > 1:
            for r in range(1, m_step):
                tr = t0 + r * (t1 - t0) / m_step
                fr = max(0, min(n - 1, int(np.searchsorted(run_np, tr))))
                b_times.append(tr)
                b_bp.append(float(beat_np[max(0, fr - 2) : min(n, fr + 3)].max()))
                b_dp.append(float(down_np[max(0, fr - 2) : min(n, fr + 3)].max()))
                b_idx.append(fr)
                b_interp.append(True)
        b_times.append(t1)
        b_bp.append(float(cand_bp[cur_k]))
        b_dp.append(float(cand_dp[cur_k]))
        b_idx.append(int(cand_idx[cur_k]))
        b_interp.append(False)

    # An odd eighth-note bar may end halfway between two points on the
    # preceding quarter clock. Fit the observed 0.5/1.5-beat interval as such
    # instead of inventing a short tempo island or moving the next downbeat.
    positions = np.arange(len(b_times), dtype=np.float64)
    if (3, 8) in meters:
        steps = np.ones(len(b_times) - 1)
        intervals = np.diff(b_times)
        for i, (a, b) in enumerate(zip(b_idx, b_idx[1:])):
            delta = intervals[i] * (bpm_tr_np[a] + bpm_tr_np[b]) / 120
            half = round(2 * float(delta)) / 2
            if v3:
                neighbours = intervals[max(0, i-3):i].tolist() + intervals[i+1:i+4].tolist()
                stable = np.median(neighbours) if neighbours else intervals[i]
                observed = intervals[i] / stable
                local_half = round(2 * observed) / 2
                confident = (min(b_bp[i:i+2]) >= .65 and local_half in (.5, 1.5)
                             and abs(observed - local_half) <= .08
                             and sum(abs(x/stable - 1) < .08 for x in neighbours) >= 2)
                if confident:
                    steps[i] = local_half
                elif (half in (.5, 1.5) and abs(delta-half) <= .12
                      and shift_prob[max(0, a-3):min(n, b+4)].max() >= .30):
                    steps[i] = half
            elif half in (.5, 1.5) and abs(delta-half) <= .12 and shift_prob[a:b+1].max() >= .5:
                steps[i] = half
        positions[1:] = np.cumsum(steps)
    return dict(
        positions=positions,
        half_grid=v3 and np.any(positions % 1),
        times=np.array(b_times, dtype=np.float64),
        beat_p=np.array(b_bp, dtype=np.float64),
        down_p=np.array(b_dp, dtype=np.float64),
        frames=np.array(b_idx, dtype=np.int64),
        interp=np.array(b_interp, dtype=bool),
        running=run_np,
    )


def _partition_tempo_segments(tracked, change_prob, lambda_bar=80.0, lambda_mid=550.0):
    """Optimal piecewise-constant tempo segmentation on tracked quarter beats."""
    t = tracked["times"]
    n_beats = len(t) - 1
    if n_beats < 4:
        return [(0, n_beats)]
    w = np.where(tracked["interp"], 0.80, np.clip(0.5 + 0.5 * tracked["beat_p"], 0.4, 1.0))
    idx = tracked["positions"]
    s0 = np.r_[0.0, np.cumsum(w)]
    s1 = np.r_[0.0, np.cumsum(w * idx)]
    s2 = np.r_[0.0, np.cumsum(w * idx * idx)]
    t0 = np.r_[0.0, np.cumsum(w * t)]
    t1 = np.r_[0.0, np.cumsum(w * idx * t)]
    t2 = np.r_[0.0, np.cumsum(w * t * t)]

    frames = tracked["frames"]
    n = len(change_prob)
    cp_np = change_prob.numpy()
    ch_at = np.array([float(cp_np[max(0, f - 10) : min(n, f + 11)].max()) for f in frames])
    split_pen = np.full(len(t), lambda_mid, dtype=np.float64)
    is_bar = tracked["down_p"] >= 0.45
    # Infer a local lattice only from two measured, consistently spaced bars.
    # An isolated syncopation cannot seed a whole passage of cheap boundaries.
    anchors = np.flatnonzero(is_bar)
    is_half = np.zeros(len(t), dtype=bool)
    for left, right in zip(anchors, anchors[1:]):
        span = idx[right] - idx[left]
        if span in (2, 3, 3.5, 4, 5, 6, 7):
            is_half[left:right] |= abs(idx[left:right] - idx[left] - span / 2) < 0.1
    # Bridge one missing downbeat when the two surrounding measured bars
    # agree on the period. Odd meters retain their own local span.
    for a, b, c, d in zip(anchors, anchors[1:], anchors[2:], anchors[3:]):
        span = idx[b] - idx[a]
        if span in (2, 3, 3.5, 4, 5, 6, 7) and idx[c]-idx[b] == 2*span and idx[d]-idx[c] == span:
            offset = idx[b:c] - idx[b]
            is_bar[b:c] |= abs(offset-span) < .1
            is_half[b:c] |= (abs(offset-span/2)<.1) | (abs(offset-1.5*span)<.1)
    split_pen[is_half] = 210.
    split_pen[is_bar] = lambda_bar
    near_bar = np.zeros(len(t), dtype=bool)
    for anchor in np.flatnonzero(is_bar):
        near_bar |= abs(idx - idx[anchor]) <= 1
    if tracked.get("half_grid"):
        # At a half-quarter reanchor the previous regular bar lattice is no
        # longer a valid prior for the change itself. Keep the measured cut.
        half_clock_change = (idx % 1 != 0) & (ch_at >= 0.55)
        split_pen[half_clock_change] = 80.
    split_pen[0] = 0.0

    dp = np.full(len(t), 1e18, dtype=np.float64)
    back = np.full(len(t), -1, dtype=np.int64)
    dp[0] = 0.0
    min_beats = 4
    for j in range(min_beats, len(t)):
        starts = np.r_[0, np.arange(min_beats, j - min_beats + 1)]
        e = j + 1
        sw = s0[e] - s0[starts]
        sx = s1[e] - s1[starts]
        sxx = s2[e] - s2[starts]
        sy = t0[e] - t0[starts]
        sxy = t1[e] - t1[starts]
        syy = t2[e] - t2[starts]
        denom = sw * sxx - sx * sx
        slope = np.where(denom > 1e-9, (sw * sxy - sx * sy) / np.maximum(denom, 1e-9), 0.25)
        slope = np.clip(slope, 60.0 / 365.0, 60.0 / 40.0)
        intercept = (sy - slope * sx) / np.maximum(sw, 1e-9)
        sse_s2 = np.maximum(
            0.0,
            syy
            - 2.0 * intercept * sy
            - 2.0 * slope * sxy
            + intercept * intercept * sw
            + 2.0 * intercept * slope * sx
            + slope * slope * sxx,
        )
        fit = sse_s2 * 1e6
        penalty = split_pen[starts].copy()
        bars = is_bar[starts]
        if bars.any():
            mid_win = (~near_bar[starts]) & (ch_at[starts] >= 0.55) & (fit < 0.70 * fit[bars].min())
            penalty[mid_win] = lambda_bar
        cost = dp[starts] + fit + penalty
        best_pos = int(np.argmin(cost))
        dp[j] = float(cost[best_pos])
        back[j] = int(starts[best_pos])

    segs = []
    cur = len(t) - 1
    while cur > 0:
        prv = int(back[cur])
        segs.append((prv, cur))
        cur = prv
    segs.reverse()
    return segs


def _fit_beats_bpm(tracked, k0, k1):
    """Weighted linear slope BPM fit across tracked quarter beats [k0, k1]."""
    ts = tracked["times"][k0 : k1 + 1]
    ws = np.where(
        tracked["interp"][k0 : k1 + 1], 0.80, np.clip(0.5 + 0.5 * tracked["beat_p"][k0 : k1 + 1], 0.4, 1.0)
    )
    xs = tracked["positions"][k0:k1 + 1] - tracked["positions"][k0]
    xm = np.average(xs, weights=ws)
    tm = np.average(ts, weights=ws)
    denom = float(np.sum(ws * (xs - xm) ** 2))
    slope = (
        float(np.sum(ws * (xs - xm) * (ts - tm)) / denom)
        if denom > 1e-9
        else (ts[-1] - ts[0]) / max(1, k1 - k0)
    )
    return 60.0 / max(slope, 60.0 / 365.0)


def _merge_dp_segments(segs, tracked, change_prob, beat=None, down=None, bpm_track=None):
    """Merge adjacent false DP splits and bridge short phase-locked meter islands."""
    cur = list(segs)

    def seg_fit_err(k0, k1):
        ts = tracked["times"][k0 : k1 + 1]
        ws = np.where(tracked["interp"][k0 : k1 + 1], 0.80, 1.0)
        xs = tracked["positions"][k0:k1 + 1] - tracked["positions"][k0]
        xm = np.average(xs, weights=ws)
        tm = np.average(ts, weights=ws)
        slope = float(np.sum(ws * (xs - xm) * (ts - tm)) / max(np.sum(ws * (xs - xm) ** 2), 1e-9))
        pred = tm + slope * (xs - xm)
        res_ms = (ts - pred) * 1000.0
        w_real = ~tracked["interp"][k0 : k1 + 1]
        max_abs = float(np.max(np.abs(res_ms[w_real]))) if w_real.any() else float(np.max(np.abs(res_ms)))
        rms = float(np.sqrt(np.average(res_ms**2, weights=ws)))
        return 60.0 / slope, rms, max_abs, res_ms

    changed = True
    while changed and len(cur) > 1:
        changed = False
        nxt = []
        i = 0
        while i < len(cur):
            if i + 1 < len(cur):
                k0, km = cur[i]
                _, k1 = cur[i + 1]
                bpm1 = _fit_beats_bpm(tracked, k0, km)
                bpm2 = _fit_beats_bpm(tracked, km, k1)
                rel = abs(bpm1 - bpm2) / max(min(bpm1, bpm2), 1.0)
                _, rms_c, max_c, res_c = seg_fit_err(k0, k1)
                _, rms1, _, _ = seg_fit_err(k0, km)
                _, rms2, _, _ = seg_fit_err(km, k1)
                n1 = km - k0 + 1
                jump = abs(float(np.median(res_c[n1 - 1 :])) - float(np.median(res_c[:n1])))
                b_f = int(tracked["frames"][km])
                ch_m = float(change_prob[max(0, b_f - 12) : min(len(change_prob), b_f + 13)].max())
                short_wobble = min(km - k0, k1 - km) <= 8 and rel < 0.024 and rms_c <= 8.0 and max_c <= 18.0
                same_tempo = (
                    rel < 0.018
                    and jump <= (14.0 if ch_m >= 0.72 else 22.0)
                    and rms_c <= max(14.0, 1.5 * max(rms1, rms2))
                )
                stable_pulse = False
                if bpm_track is not None and rel <= 0.035 and rms_c <= 12 and max_c <= 25:
                    a, m, b = tracked["frames"][[k0, km, k1]]
                    head1 = float(bpm_track[a : m + 1].median())
                    head2 = float(bpm_track[m : b + 1].median())
                    bars = tracked["positions"][k0 : k1 + 1][tracked["down_p"][k0 : k1 + 1] >= 0.50]
                    spans, counts = np.unique(np.diff(bars), return_counts=True)
                    alternating = sum(counts[np.isin(spans, (3, 3.5, 5, 7))]) >= 2
                    stable_pulse = (
                        alternating
                        and sum(counts >= 2) >= 2
                        and jump <= 8
                        and abs(head2 / head1 - 1) <= 0.025
                    )
                if same_tempo or short_wobble or stable_pulse:
                    nxt.append((k0, k1))
                    i += 2
                    changed = True
                    continue
            nxt.append(cur[i])
            i += 1
        cur = nxt

    if len(cur) >= 3:
        beat_np = beat.numpy() if beat is not None else None
        down_np = down.numpy() if down is not None else None
        changed = True
        while changed and len(cur) >= 3:
            changed = False
            nxt = []
            i = 0
            while i < len(cur):
                merged_island = False
                if i + 2 < len(cur):
                    k0, ka = cur[i]
                    if (ka - k0) >= 16:
                        b1 = _fit_beats_bpm(tracked, k0, ka)
                        for j in range(i + 2, min(len(cur), i + 8)):
                            kb, k2 = cur[j]
                            if (k2 - kb) < 16:
                                continue
                            span_island = float(tracked["times"][kb]) - float(tracked["times"][ka])
                            if span_island > 12.0:
                                break
                            b3 = _fit_beats_bpm(tracked, kb, k2)
                            if abs(b1 - b3) / max(b1, 1.0) >= 0.012:
                                continue
                            p_avg = 60.0 / (0.5 * (b1 + b3))
                            dt_tot = float(tracked["times"][k2]) - float(tracked["times"][k0])
                            n_tot_f = dt_tot / p_avg
                            err_tot_ms = abs(n_tot_f - round(n_tot_f)) * p_avg * 1000.0
                            n_isl_f = span_island / p_avg
                            err_isl_ms = abs(n_isl_f - round(n_isl_f)) * p_avg * 1000.0
                            if err_tot_ms > 45.0 or err_isl_ms > 35.0:
                                continue
                            run_arr = tracked.get("running")
                            grid_ok = True
                            if beat_np is not None:
                                n_isl = max(1, round(n_isl_f))
                                t_start = float(tracked["times"][ka])
                                b_hits = []
                                for r in range(1, n_isl):
                                    t_r = t_start + r * p_avg
                                    fr = (
                                        int(np.searchsorted(run_arr, t_r))
                                        if run_arr is not None
                                        else int(round(t_r * FRAME_RATE))
                                    )
                                    b_hits.append(
                                        float(beat_np[max(0, fr - 3) : min(len(beat_np), fr + 4)].max())
                                    )
                                hit_lim = 0.28 if (err_tot_ms <= 20.0 and err_isl_ms <= 20.0) else 0.45
                                if b_hits and float(np.mean(b_hits)) < hit_lim:
                                    grid_ok = False
                            if grid_ok and down_np is not None and not (err_tot_ms <= 20.0 and err_isl_ms <= 20.0):
                                f_lo = int(tracked["frames"][ka]) + 4
                                f_hi = int(tracked["frames"][kb]) - 4
                                if f_hi > f_lo:
                                    dp_pks = _peaks(down[f_lo:f_hi], 0.55, 4) + f_lo
                                    if len(dp_pks):
                                        t_pks = (
                                            run_arr[dp_pks.numpy()]
                                            if run_arr is not None
                                            else (dp_pks.numpy() / FRAME_RATE + 0.5 / FRAME_RATE)
                                        )
                                        rel_p = (t_pks - float(tracked["times"][ka])) / p_avg
                                        phase_off = np.abs(rel_p - np.round(rel_p)) * p_avg * 1000.0
                                        if float(np.median(phase_off)) > 35.0:
                                            grid_ok = False
                            if grid_ok:
                                nxt.append((k0, k2))
                                i = j + 1
                                changed = True
                                merged_island = True
                                break
                if not merged_island:
                    nxt.append(cur[i])
                    i += 1
            cur = nxt
    return cur


def _downbeat_meter_support(
    down_prob_s, is_active_e, m_probs, lengths, cand_meters, cand_trained_idx, i0, i1, ch_arr=None
):
    """Score sustained odd/OTHER meters and exact run barlines from multi-bar downbeat periodicity."""
    num_e = len(down_prob_s)
    db_sup = np.zeros((num_e, len(lengths)), dtype=np.float64)
    db_sup_start = np.zeros((num_e, len(lengths)), dtype=np.float64)
    dp_act = np.where(is_active_e, down_prob_s, 0.0)
    p44 = m_probs[:, 0]
    for k, length in enumerate(lengths):
        if length in (2, 8):
            continue
        m_idx = cand_trained_idx[k]
        long_or_untrained = m_idx < 0 or cand_meters[k] in ((6, 4), (7, 4)) or length > 16
        if long_or_untrained:
            competing_indices = [col for col in range(1, m_probs.shape[1] - 1) if col != m_idx]
            competing_non44_p = m_probs[:, competing_indices].max(axis=1)
        n_bars_req = 3 if long_or_untrained else 2
        p44_lim = 0.65 if (long_or_untrained and length % 2 == 0) else (0.78 if long_or_untrained else 0.68)
        bar_ok = np.zeros(num_e, dtype=np.float64)
        for i in range(i0, max(i0, i1 - n_bars_req * length)):
            if not is_active_e[i] or dp_act[i] < 0.42:
                continue
            bar1_carryover = False
            if length == 14 and i + 8 < i1 and dp_act[i + 8] >= 0.50:
                bar1_carryover = (
                    i + 14 < i1
                    and dp_act[i + 14] >= 0.65
                    and (ch_arr is None or (ch_arr[i + 14] >= 0.45 and ch_arr[i + 8] < 0.30))
                    and all(
                        dp_act[i + r * 14 + 8] < 0.35
                        for r in range(1, n_bars_req)
                        if i + r * 14 + 8 < i1
                    )
                )
                if not bar1_carryover:
                    continue
            if float(p44[i : i + n_bars_req * length].mean()) >= p44_lim:
                continue
            if long_or_untrained and float(competing_non44_p[i : i + n_bars_req * length].mean()) >= 0.40:
                continue
            if (m_probs.shape[1] == len(EXTENDED_METERS) + 1 and length >= 16 and m_idx >= 0
                    and m_probs[i:i + n_bars_req * length, m_idx].mean() < .35):
                continue
            # 9/4 can also be three 3/4 bars with ordinary downbeats.
            if length == 18 and any(dp_act[i + r*length + s] >= .45
                                    for r in range(n_bars_req) for s in (6, 12)):
                continue
            pts = [i + r * length for r in range(n_bars_req + 1)]
            vals = dp_act[pts]
            if float(vals.mean()) < 0.58:
                continue
            if n_bars_req == 2:
                if min(vals) < 0.42:
                    continue
            elif min(vals) < 0.36 or int((vals >= 0.62).sum()) < 3:
                continue
            # Reject double-bar harmonics when L//2 also carries strong downbeats (e.g., 6/4 vs 3/4, 7/4 vs 7/8).
            if length >= 10 and length % 2 == 0 and (length // 2) in lengths:
                half_pts = [i + r * length + length // 2 for r in range(n_bars_req)]
                if cand_meters[k] == (8, 4):
                    half_harmonic = (dp_act[half_pts].mean() >= 0.45
                                     and m_probs[i : pts[-1], m_idx].mean() < 0.35)
                else:
                    half_harmonic = max(dp_act[half_pts]) >= 0.45
                if half_harmonic:
                    continue
            interior = [dp_act[i + r * length + s] for r in range(n_bars_req) for s in range(1, length)]
            int_mean = float(np.mean(interior)) if interior else 0.0
            pts44 = [i + r * 8 for r in range(1, (n_bars_req * length) // 8 + 1) if i + r * 8 < i1]
            if cand_meters[k] == (8, 4):
                # An 8/4 bar shares every other 4/4 boundary. Only the intervening
                # downbeats distinguish the shorter harmonic.
                pts44 = half_pts
            h44_loc = float(np.mean(dp_act[pts44])) if pts44 else 0.0
            adv_44 = float(vals.mean()) - max(h44_loc, int_mean + 0.15)
            h44_lim = (0.40 if bar1_carryover else 0.36) if long_or_untrained else 0.30
            # For 8/4 the half-bar test above already includes the trained-meter evidence.
            h44_ok = cand_meters[k] == (8, 4) or h44_loc < h44_lim
            if adv_44 >= 0.25 and (float(vals.mean()) - int_mean) >= 0.38 and h44_ok:
                bar_ok[i] = float(np.clip(adv_44 * 2.2, 0.0, 1.0))
        for i in range(i0, max(i0, i1 - n_bars_req * length)):
            if bar_ok[i] > 0.0:
                span_end = min(i1, i + n_bars_req * length)
                db_sup[i:span_end, k] = np.maximum(db_sup[i:span_end, k], bar_ok[i])
                for r in range(n_bars_req):
                    b_pos = i + r * length
                    if b_pos < i1:
                        db_sup_start[b_pos, k] = max(db_sup_start[b_pos, k], bar_ok[i])
        # Also detect sustained 5-bar 7/4 runs when 5/6 barlines hit cleanly even under flat 4/4 meter logits.
        if long_or_untrained and length == 14:
            n5 = 5
            for i in range(i0, max(i0, i1 - n5 * length)):
                if not is_active_e[i] or dp_act[i] < 0.42:
                    continue
                bar1_carryover = False
                if i + 8 < i1 and dp_act[i + 8] >= 0.50:
                    bar1_carryover = (
                        i + 14 < i1
                        and dp_act[i + 14] >= 0.65
                        and (ch_arr is None or (ch_arr[i + 14] >= 0.45 and ch_arr[i + 8] < 0.30))
                        and all(
                            dp_act[i + r * 14 + 8] < 0.35
                            for r in range(1, n5)
                            if i + r * 14 + 8 < i1
                        )
                    )
                    if not bar1_carryover:
                        continue
                if float(competing_non44_p[i : i + n5 * length].mean()) >= 0.40:
                    continue
                half_pts = [i + r * length + length // 2 for r in range(n5)]
                if float(np.mean(dp_act[half_pts])) >= 0.35:
                    continue
                pts = [i + r * length for r in range(n5 + 1)]
                vals = dp_act[pts]
                if float(vals.mean()) < 0.62 or int((vals >= 0.65).sum()) < 5:
                    continue
                interior = [dp_act[i + r * length + s] for r in range(n5) for s in range(1, length)]
                int_mean = float(np.mean(interior)) if interior else 0.0
                pts44 = [i + r * 8 for r in range(1, (n5 * length) // 8 + 1) if i + r * 8 < i1]
                h44_loc = float(np.mean(dp_act[pts44])) if pts44 else 0.0
                adv_44 = float(vals.mean()) - max(h44_loc, int_mean + 0.15)
                p44_mean = float(p44[i : i + n5 * length].mean())
                p44_ok = (p44_mean < 0.80) or (adv_44 >= 0.32 and h44_loc < 0.25 and float(vals.mean()) >= 0.72)
                h44_lim_n5 = 0.40 if bar1_carryover else 0.36
                if p44_ok and adv_44 >= 0.25 and (float(vals.mean()) - int_mean) >= 0.38 and h44_loc < h44_lim_n5:
                    sc = float(np.clip(adv_44 * 2.2, 0.0, 1.0))
                    db_sup[i : min(i1, i + n5 * length), k] = np.maximum(
                        db_sup[i : min(i1, i + n5 * length), k], sc
                    )
                    for r in range(n5):
                        b_pos = i + r * length
                        if b_pos < i1:
                            db_sup_start[b_pos, k] = max(db_sup_start[b_pos, k], sc)
        # Detect sustained 6-bar 7/4 runs backed by change_prob + downbeat periodicity even under 4/4 sub-accents.
        if long_or_untrained and length == 14 and ch_arr is not None:
            n_ch = 6
            for i in range(i0, max(i0, i1 - n_ch * length)):
                if not is_active_e[i]:
                    continue
                if float(competing_non44_p[i : i + n_ch * length].mean()) >= 0.40:
                    continue
                half_pts = [i + r * length + length // 2 for r in range(n_ch)]
                if float(np.mean(dp_act[half_pts])) >= 0.35:
                    continue
                pts = [i + r * length for r in range(n_ch + 1)]
                vals = dp_act[pts]
                ch_vals = ch_arr[pts]
                if float(vals[0]) < 0.65 or float(vals[-1]) < 0.65:
                    continue
                if int((vals >= 0.70).sum()) < 6 or float(vals.mean()) < 0.68:
                    continue
                if int((ch_vals >= 0.45).sum()) < 5 or float(ch_vals.mean()) < 0.52:
                    continue
                pts_set = set(pts)
                pts44 = [
                    i + r * 8
                    for r in range(1, (n_ch * length) // 8 + 1)
                    if i + r * 8 < i1 and (i + r * 8) not in pts_set
                ]
                if pts44 and (
                    float(np.max(ch_arr[pts44])) >= 0.25
                    or float(ch_vals.mean()) - float(np.mean(ch_arr[pts44])) < 0.35
                ):
                    continue
                sc = float(np.clip((float(ch_vals.mean()) - 0.15) * 1.8, 0.55, 1.0))
                db_sup[i : min(i1, i + n_ch * length), k] = np.maximum(
                    db_sup[i : min(i1, i + n_ch * length), k], sc
                )
                for r in range(n_ch):
                    b_pos = i + r * length
                    if b_pos < i1:
                        db_sup_start[b_pos, k] = max(db_sup_start[b_pos, k], sc)
    return db_sup, db_sup_start


def _classify_decoder_preset(
    down_prob_s, is_active_e, m_probs, ch_arr, db_sup, bpm_vals_e, cand_meters, cand_trained_idx, i0, i1
):
    """Classify a song into 'simple', 'medium', or 'complex' from raw prediction statistics."""
    act_mask = is_active_e[i0:i1]
    if int(act_mask.sum()) < 4:
        return "simple"

    non44_act = 1.0 - m_probs[i0:i1, 0]
    non44_vals = non44_act[act_mask]
    non44_mean = float(non44_vals.mean())
    non44_frac_030 = float((non44_vals > 0.30).mean())
    ch_q95 = float(np.quantile(ch_arr[i0:i1][act_mask], 0.95))

    bpm_vals = bpm_vals_e[i0:i1][act_mask]
    if len(bpm_vals) >= 4:
        med_bpm = max(float(np.median(bpm_vals)), 1.0)
        bpm_rel_exc = float(np.quantile(bpm_vals, 0.95) - np.quantile(bpm_vals, 0.05)) / med_bpm
    else:
        bpm_rel_exc = 0.0

    # 32-beat (64 eighth-note) sliding windows for local meter-probability and local mod-4 phase stability.
    win_len = 64
    step = 8
    span = i1 - i0
    if span <= win_len:
        starts = [i0]
    else:
        starts = list(range(i0, i1 - win_len + 1, step))
        if starts[-1] != i1 - win_len:
            starts.append(i1 - win_len)

    w_non44_list = []
    w_off_mod4_list = []
    for ws in starts:
        we = min(i1, ws + win_len)
        act_w = is_active_e[ws:we]
        if int(act_w.sum()) < 4 and len(starts) > 1:
            continue
        w_non44_list.append(float((1.0 - m_probs[ws:we, 0])[act_w].mean()) if act_w.any() else 0.0)
        dp_w = np.where(act_w & (down_prob_s[ws:we] >= 0.30), down_prob_s[ws:we], 0.0)
        idx_mod8 = np.arange(ws, we) % 8
        mass = np.bincount(idx_mod8, weights=dp_w, minlength=8)
        tot_mass = float(mass.sum())
        w_off_mod4_list.append(1.0 - float(mass.max()) / tot_mass if tot_mass > 1e-6 else 0.0)

    win32b_non44_max = max(w_non44_list) if w_non44_list else non44_mean
    win32b_non44_frac_025 = float(np.mean([v > 0.25 for v in w_non44_list])) if w_non44_list else 0.0
    win32b_off_local_mod4_mean = float(np.mean(w_off_mod4_list)) if w_off_mod4_list else 0.0

    odd_long_cols = [
        k for k, m_idx in enumerate(cand_trained_idx) if m_idx < 0 or cand_meters[k] in ((6, 4), (7, 4))
    ]
    has_sustained_odd_other = any(float(db_sup[i0:i1, k].max()) >= 0.35 for k in odd_long_cols)

    if (
        non44_mean >= 0.35
        or win32b_non44_frac_025 >= 0.50
        or (win32b_off_local_mod4_mean >= 0.35 and ch_q95 >= 0.30)
        or (bpm_rel_exc >= 0.22 and non44_mean >= 0.15)
    ):
        return "complex"
    if (
        win32b_non44_max < 0.15
        and non44_frac_030 < 0.03
        and win32b_off_local_mod4_mean < 0.08
        and ch_q95 < 0.15
        and not has_sustained_odd_other
    ):
        return "simple"
    return "medium"


def _physical_meter_gaps(grid, down, sub, pulse_norm, stopped, meter_scores, *, allow_leading, peaks=None):
    meters, _ = _meter_table_from_logits_dim(meter_scores.shape[-1])
    if peaks is None:
        peaks = _peaks(down, .55, 2)
    peaks = peaks[(down[peaks] >= .55) & ~stopped[peaks] & (pulse_norm[peaks] >= .15)]
    seconds = (peaks.double() + .5) / FRAME_RATE + sub[peaks]

    def preferred_span(second):
        index = min(len(down) - 1, max(0, int(second * FRAME_RATE)))
        num, den = meters[int(meter_scores[index, :len(meters)].argmax())]
        return 4 * num / den

    return meter_gaps(grid, seconds.numpy(), down[peaks].numpy(), meters, len(down) / FRAME_RATE,
                      allow_leading=allow_leading, preferred_span=preferred_span)


def _section_eighths(grid, n, start, end):
    origin = 0.0 if start is None else start
    first = math.ceil(2 * grid.second_to_beat(0) - 1e-6) if start is None else 0
    last_beat = grid.second_to_beat(n / FRAME_RATE) if end is None else end
    last = math.floor(2 * (last_beat - origin) + 1e-6)
    return first, origin + torch.arange(first, max(first, last) + 1, dtype=torch.float64) * .5


def _decode_meter_viterbi(
    prov_grid,
    n,
    meter_scores,
    down_logits,
    change_prob,
    bar_phase,
    pulse_norm,
    stopped,
    first_running,
    *,
    bpm_track=None,
    dominant_meter=None,
    harmonic_intervals=(),
    anchored_bars=(),
    preset="auto",
    start=None,
    end=None,
):
    """Jointly decode meter segments and barlines via Semi-Markov barline-interval DP on the 8th-note clock."""
    if preset not in VALID_DECODER_PRESETS:
        raise ValueError(f"Unknown decoder preset {preset!r}; expected one of {VALID_DECODER_PRESETS}.")

    first_e, e_beats = _section_eighths(prov_grid, n, start, end)
    e_secs = prov_grid.beat_to_second_tensor(e_beats)
    ix = (e_secs * FRAME_RATE - 0.5).round().long().clamp(0, n - 1)

    if start is None and first_running is not None and len(e_beats) > 1:
        i0 = max(0, min(len(e_beats) - 1, round(2.0 * prov_grid.second_to_beat(first_running)) - first_e))
    else:
        i0 = 0
    unstopped_e = torch.where(~stopped[ix])[0]
    i1 = int(unstopped_e[-1]) + 1 if len(unstopped_e) and int(unstopped_e[-1]) >= i0 else len(e_beats)
    if i1 - i0 < 2:
        return [(0.0, 4, 4)], float(e_beats[i0]), ("simple" if preset == "auto" else preset)

    down_pooled = F.max_pool1d(down_logits[None, None], 5, stride=1, padding=2)[0, 0, ix]
    ch_pooled = F.max_pool1d(change_prob[None, None], 11, stride=1, padding=5)[0, 0, ix]
    m_logp_raw = meter_scores[ix].log_softmax(-1).numpy()
    down_prob_s = down_pooled.sigmoid().numpy()
    ch_arr = ch_pooled.numpy()
    contrast = float(
        np.clip(np.quantile(down_prob_s[i0:i1], 0.98) - np.quantile(down_prob_s[i0:i1], 0.15), 0.0, 1.0)
    )
    unit_bar = bar_phase[ix].numpy()
    is_active_e = (~stopped[ix]).numpy() & (pulse_norm[ix].numpy() >= 0.15)

    meters, other_idx = _meter_table_from_logits_dim(meter_scores.shape[-1])
    cand_meters = [m for m in VITERBI_METERS if m in LEGACY_VITERBI_METERS or m in meters or m == (15, 8)]
    lengths = np.array([round(8 * num / den) for num, den in cand_meters], dtype=int)
    num_k = len(cand_meters)
    meter_to_idx = {m: k for k, m in enumerate(meters)}
    cand_to_idx = {m: k for k, m in enumerate(cand_meters)}
    k_44 = cand_to_idx[(4, 4)]
    k_34 = cand_to_idx[(3, 4)]
    k_54 = cand_to_idx[(5, 4)]
    k_78 = cand_to_idx[(7, 8)]
    k_14 = cand_to_idx[(1, 4)]
    additive_ks = frozenset(cand_to_idx[m] for m in ((3, 4), (4, 4), (5, 4)))
    sustained_ks = tuple(k for k in range(num_k) if k != k_14)

    # On a physically confirmed bar-preserving alias, the old meter-head
    # distribution describes the competing pulse count. Transfer that support
    # to the measured meter only over the confirmed interval.
    for left, right in harmonic_intervals:
        inside = (e_secs.numpy() >= left - 0.025) & (e_secs.numpy() < right - 0.025)
        dom_idx = meter_to_idx[dominant_meter]
        m_logp_raw[inside, dom_idx] = m_logp_raw[inside].max(axis=1)
        for j in range(m_logp_raw.shape[1]):
            if j != dom_idx:
                m_logp_raw[inside,j] = np.minimum(m_logp_raw[inside,j], m_logp_raw[inside,dom_idx]-3.5)
    m_probs = np.exp(m_logp_raw)
    p44 = m_probs[:, 0]
    cand_trained_idx = [meter_to_idx.get(meter, -1) for meter in cand_meters]
    db_sup, db_sup_start = _downbeat_meter_support(
        down_prob_s, is_active_e, m_probs, lengths, cand_meters, cand_trained_idx, i0, i1, ch_arr=ch_arr
    )

    compound = cand_to_idx[(15, 8)]
    # The older heads have no 15/8 column. A cycle of 7+8 eighths
    # can still be measured against its two competing 7/8 and 4/4 grids.
    accent = np.where(is_active_e, down_prob_s, 0.0)
    for i in range(i0, i1 - 45):
        points = i + np.arange(4) * 15
        heads = accent[points]
        if heads.min() < 0.64 or heads.mean() < 0.79:
            continue
        groups = np.maximum(accent[points[:-1] + 7], accent[points[:-1] + 8])
        if groups.min() < 0.42 or heads.mean() < groups.mean() + 0.075:
            continue
        if max(accent[i + np.arange(1, 5) * 7].mean(), accent[i + np.arange(1, 5) * 8].mean()) > 0.50:
            continue
        db_sup[i : i + 45, compound] = 1.0
        db_sup_start[points[:-1], compound] = 1.0

    # At the entrance to an additive cycle, a few real 7/8 bars
    # may precede the longer period on the same pulse lattice.
    short_entrance = np.zeros(len(accent), dtype=bool)
    for j in np.flatnonzero(db_sup_start[:, compound] == 1.0):
        if j < i0 + 14 or j + 7 >= i1:
            continue
        short = accent[j + np.array([-14, -7, 0, 7])]
        if short.min() >= 0.48 and short.mean() >= 0.62 and short.mean() - accent[j - 14 + 8] > 0.18:
            db_sup_start[[j - 14, j - 7], k_78] = 1.0
            db_sup[j - 14 : j, k_78] = 1.0
            short_entrance[j - 14 : j] = True

    if preset == "auto":
        bpm_vals_e = (
            bpm_track[ix].numpy()
            if bpm_track is not None
            else np.ones(len(e_beats), dtype=np.float64) * 120.0
        )
        resolved_preset = _classify_decoder_preset(
            down_prob_s, is_active_e, m_probs, ch_arr, db_sup, bpm_vals_e, cand_meters, cand_trained_idx, i0, i1
        )
    else:
        resolved_preset = preset

    if resolved_preset == "simple":
        # Suppress speculative odd/OTHER meter promotion in simple 4/4 songs.
        for k in range(num_k):
            if k != k_44:
                db_sup[:, k] = 0.0
                db_sup_start[:, k] = 0.0
        turnaround_ks = tuple(cand_to_idx[m] for m in ((1, 4), (2, 4), (3, 4)))
    else:
        turnaround_ks = tuple(cand_to_idx[m] for m in ((1, 4), (2, 4), (3, 4), (7, 8), (3, 8)) if m in cand_to_idx)

    m_logp_clip = np.maximum(m_logp_raw, -3.5)
    max_trained_p = m_probs[:, :other_idx].max(axis=1)
    top_trained_clip = m_logp_clip[:, :other_idx].max(axis=1)
    periodic_col = np.maximum(
        np.where(max_trained_p >= 0.55, -3.5, top_trained_clip - 0.12),
        top_trained_clip - 0.05,
    )
    num_e = len(e_beats)
    m_col = np.zeros((num_e, num_k), dtype=np.float64)
    for k, m_idx in enumerate(cand_trained_idx):
        if m_idx >= 0:
            trained_col = m_logp_clip[:, m_idx] + 1.25 * db_sup[:, k]
            if cand_meters[k] in ((6, 4), (7, 4), (15, 8)):
                # 6/4 and 7/4 warm-start with -4.0 bias in Stage 2; retain legacy periodic downbeat support.
                m_col[:, k] = np.where(
                    db_sup[:, k] >= 0.35,
                    np.maximum(trained_col, periodic_col + 1.35 * db_sup[:, k]),
                    trained_col,
                )
            else:
                m_col[:, k] = trained_col
        elif k == k_14:
            m_col[:, k] = -3.5
        else:
            # 6/4 and 7/4 are mapped to OTHER (masked during training); require strong downbeat spacing.
            m_col[:, k] = np.where(
                db_sup[:, k] >= 0.35,
                periodic_col + 1.35 * db_sup[:, k],
                -9.0,
            )

    # Competing 7/8 logits measure the inner accent group in this case.
    compound_run = db_sup[:, compound] >= 1.0
    m_col[compound_run, compound] = np.maximum(
        m_col[compound_run, compound], top_trained_clip[compound_run] + 1.5
    )

    m_col[short_entrance, k_78] = top_trained_clip[short_entrance] + 1.7
    m_col[short_entrance, compound] = -3.5

    tempo_chg_e = np.zeros(num_e, dtype=bool)
    for tb in prov_grid.beats[1:]:
        ti = round(2.0 * (float(tb) - float(e_beats[0])))
        for d in (-1, 0, 1):
            if 0 <= ti + d < num_e:
                tempo_chg_e[ti + d] = True

    near_tempo = np.zeros(num_e, dtype=bool)
    dom_k = cand_to_idx.get(dominant_meter)
    if dom_k is not None:
        changes = np.asarray(prov_grid.beats[1:])
        at = e_beats.numpy()
        near_tempo = np.searchsorted(changes, at + 8) - np.searchsorted(changes, at - 8) >= 2
        for boundary, before, after in zip(changes, prov_grid.bpms, prov_grid.bpms[1:]):
            if abs(after / before - 1) >= 0.075:
                near_tempo |= abs(at - boundary) <= lengths[dom_k] / 2

    is_db_peak = np.zeros(num_e, dtype=bool)
    for t in range(i0, i1):
        if is_active_e[t] and down_prob_s[t] >= 0.42:
            lo = max(i0, t - 2)
            hi = min(i1, t + 3)
            left_max = float(down_prob_s[lo:t].max()) if t > lo else -1.0
            right_max = float(down_prob_s[t + 1 : hi].max()) if t + 1 < hi else -1.0
            if down_prob_s[t] > left_max + 1e-6 and down_prob_s[t] >= right_max - 1e-6:
                is_db_peak[t] = True

    db_node = np.where(is_active_e, contrast * 3.0 * (down_prob_s - 0.35), 0.0)
    for second in anchored_bars:
        pos = round(2 * (prov_grid.second_to_beat(second) - float(e_beats[0])))
        if 0 <= pos < len(e_secs) and abs(float(e_secs[pos]) - second) <= 0.025:
            db_node[pos] += 60.
    skip_contrib = np.where(is_db_peak, np.maximum(0.0, down_prob_s - 0.38), 0.0)
    skip_74_contrib = np.where(ch_arr < 0.25, 0.25 * skip_contrib, skip_contrib)

    cum_act = np.r_[0, np.cumsum(is_active_e.astype(np.int32))]
    p44_cum = np.r_[0.0, np.cumsum(p44)]
    p54_cum = np.r_[0.0, np.cumsum(m_probs[:, 5])]
    p34_cum = np.r_[0.0, np.cumsum(m_probs[:, 1])]
    p78_cum = np.r_[0.0, np.cumsum(m_probs[:, 6])]
    cum_skip = np.r_[0.0, np.cumsum(skip_contrib)]
    cum_skip_74 = np.r_[0.0, np.cumsum(skip_74_contrib)]
    m_col_cum = np.vstack([np.zeros((1, num_k), dtype=np.float64), np.cumsum(m_col, axis=0)])

    phase_templates = []
    for L in lengths:
        ph_k = np.arange(L, dtype=np.float64) / L
        phase_templates.append(np.stack((np.sin(2.0 * np.pi * ph_k), np.cos(2.0 * np.pi * ph_k)), axis=1))

    # Vectorized bar_phase dot-products for full bars [i, i + L) across all meters.
    full_phase_align = np.zeros((num_e, num_k), dtype=np.float64)
    for k, L in enumerate(lengths):
        if i1 - i0 >= L:
            win = np.lib.stride_tricks.sliding_window_view(unit_bar[i0:i1], L, axis=0)
            full_phase_align[i0 : i1 - L + 1, k] = np.einsum("icl,lc->i", win, phase_templates[k]) / L

    def partial_phase_align(a, b, k, phase_start=0):
        seg_u = unit_bar[a:b]
        if len(seg_u) == 0:
            return 0.0
        L = lengths[k]
        ph = (np.arange(len(seg_u), dtype=np.float64) + phase_start) / L
        sc = seg_u[:, 0] * np.sin(2.0 * np.pi * ph) + seg_u[:, 1] * np.cos(2.0 * np.pi * ph)
        return float(sc.mean())

    def score_interval(a, b, k, *, phase_start=0, is_pickup=False):
        L = int(lengths[k])
        span = b - a
        act_cnt = int(cum_act[b] - cum_act[a])
        if act_cnt == 0:
            return 0.02 if k == k_44 else 0.0
        act_frac = act_cnt / L
        dur_wt = act_cnt / 8.0
        bar_p44 = (p44_cum[b] - p44_cum[a]) / span
        if is_pickup:
            skip_raw = float(cum_skip[b - 1] - cum_skip[a]) if b - 1 > a else 0.0
            if down_prob_s[a] >= 0.42 and not is_db_peak[a]:
                skip_raw += max(0.0, float(down_prob_s[a]) - 0.40)
        else:
            skip_raw = float(cum_skip[b - 1] - cum_skip[a + 2]) if (b - 1 > a + 2) else 0.0

        if is_pickup:
            skip_pen = contrast * 4.5 * skip_raw
        elif k == k_44 and bar_p44 >= 0.75 and (down_prob_s[a] >= 0.35 or (b < i1 and down_prob_s[b] >= 0.35)):
            skip_pen = 0.0
        elif cand_meters[k] == (7, 4) and db_sup[a, k] >= 0.35:
            if b - 1 > a + 4:
                if db_sup_start[a, k] >= 0.35:
                    # Tolerate low-change 4+3/mod-4 sub-accents inside verified 7/4 bars, keeping 7/8 midpoint full.
                    skip_late = float(cum_skip_74[b - 1] - cum_skip_74[a + 4])
                    if a + 7 < b - 1:
                        skip_late += 0.75 * float(skip_contrib[a + 7])
                else:
                    skip_late = float(cum_skip[b - 1] - cum_skip[a + 4])
            else:
                skip_late = 0.0
            skip_pen = contrast * 4.5 * skip_late
        else:
            skip_pen = contrast * 4.5 * skip_raw

        m_bar = (m_col_cum[b, k] - m_col_cum[a, k]) / span
        bar_p54 = (p54_cum[b] - p54_cum[a]) / span
        bar_p34 = (p34_cum[b] - p34_cum[a]) / span
        additive_54_mode = (
            (resolved_preset != "simple")
            and (not is_pickup)
            and (k in additive_ks)
            and (bar_p44 < 0.58)
            and (bar_p54 >= 0.28)
            and ((bar_p54 + bar_p34) >= 0.45)
        )
        if additive_54_mode:
            db_a_18 = (k == k_54) or (down_prob_s[a] >= 0.18)
            db_b_18 = (k == k_54) or (b >= i1) or (down_prob_s[b] >= 0.18)
            if db_a_18 and db_b_18:
                m54_bar = (m_col_cum[b, k_54] - m_col_cum[a, k_54]) / span
                m34_bar = (m_col_cum[b, k_34] - m_col_cum[a, k_34]) / span
                m_bar = max(m_bar, max(m54_bar, m34_bar))

        if span == L and phase_start == 0:
            p_align = float(full_phase_align[a, k])
        else:
            p_align = partial_phase_align(a, b, k, phase_start=phase_start)
        meter_sc = 4.5 * dur_wt * m_bar + (0.05 * dur_wt if k == k_44 else 0.0)
        phase_wt = 0.35 if additive_54_mode else 1.6
        phase_sc = phase_wt * act_frac * p_align
        return meter_sc + phase_sc - skip_pen

    # DP states at each barline i:
    #   mode 0 = sustained bar ending at i in meter k
    #   mode 1 = single-bar turnaround ending at i in meter k (must exit to sustained meter != k)
    NEG_INF = -1e12
    dp_s = np.full((num_e, num_k), NEG_INF, dtype=np.float64)
    dp_ta = np.full((num_e, num_k), NEG_INF, dtype=np.float64)
    back_s = np.full((num_e, num_k, 3), -1, dtype=np.int32)  # (prev_i, prev_k, prev_mode)
    back_ta = np.full((num_e, num_k, 3), -1, dtype=np.int32)

    for k in sustained_ks:
        L = int(lengths[k])
        init_prior = 1.5 if k == k_44 else 0.0
        for r in range(1 if start is not None else min(L, i1 - i0)):
            b0 = i0 + r
            tie = 0.5 if (k == k_44 and r == 0) else 0.0
            if r == 0:
                sc0 = init_prior + tie + float(db_node[b0])
            else:
                pickup_db_pen = 4.0 * max(0.0, float(down_prob_s[i0]) - 0.50)
                sc0 = (
                    init_prior
                    + score_interval(i0, b0, k, phase_start=L - r, is_pickup=True)
                    + float(db_node[b0])
                    - pickup_db_pen
                )
            if sc0 > dp_s[b0, k]:
                dp_s[b0, k] = sc0
                back_s[b0, k] = (-1, k, 0)

    # Precompute per-position lookahead cues used by transition penalties.
    p44_ahead_arr = np.zeros(num_e, dtype=np.float64)
    p54_ahead_arr = np.zeros(num_e, dtype=np.float64)
    p34_ahead_arr = np.zeros(num_e, dtype=np.float64)
    p78_ahead_arr = np.zeros(num_e, dtype=np.float64)
    for i in range(i0, i1):
        w12 = min(i1, i + 12)
        den_w = max(1, w12 - i)
        p44_ahead_arr[i] = (p44_cum[w12] - p44_cum[i]) / den_w
        p54_ahead_arr[i] = (p54_cum[w12] - p54_cum[i]) / den_w
        p34_ahead_arr[i] = (p34_cum[w12] - p34_cum[i]) / den_w
        p78_ahead_arr[i] = (p78_cum[w12] - p78_cum[i]) / den_w

    m_ahead_arr = np.zeros((num_e, num_k), dtype=np.float64)
    db_ahead_arr = np.zeros((num_e, num_k), dtype=np.float64)
    for k in sustained_ks:
        m_idx = cand_trained_idx[k]
        L = int(lengths[k])
        for i in range(i0, i1):
            w8 = min(i1, i + 8)
            w2L = min(i1, i + 2 * L)
            if m_idx >= 0:
                m_ahead_arr[i, k] = float(m_probs[i:w8, m_idx].mean())
            db_ahead_arr[i, k] = float(db_sup[i:w2L, k].mean())

    best_end_sc = NEG_INF
    best_end_term = (i0, k_44)
    best_end_src = (i0, k_44, 0)

    is_simple = resolved_preset == "simple"
    is_complex = resolved_preset == "complex"
    ch_thresh = 0.45 if is_complex else 0.72

    for i in range(i0, i1):
        s_row = dp_s[i]
        ta_row = dp_ta[i]
        if max(s_row.max(), ta_row.max()) <= NEG_INF * 0.5:
            continue

        best_s_k = int(np.argmax(s_row))
        best_s_v = float(s_row[best_s_k])

        best_add_k = -1
        best_add_v = NEG_INF
        for ak in additive_ks:
            if s_row[ak] > best_add_v:
                best_add_v = float(s_row[ak])
                best_add_k = ak

        has_change = ch_arr[i] >= ch_thresh
        p44_ahead = float(p44_ahead_arr[i])
        p54_ahead = float(p54_ahead_arr[i])
        p34_ahead = float(p34_ahead_arr[i])
        p78_ahead = float(p78_ahead_arr[i])
        allow_additive = (
            (not is_simple)
            and p44_ahead < 0.65
            and p54_ahead >= 0.20
            and p78_ahead < 0.40
            and not (p34_ahead >= 0.58 and p54_ahead < 0.25)
        )

        # 1) Sustained transitions i -> j = i + L_k (or song termination at i1).
        for k in sustained_ks:
            L = int(lengths[k])
            j = i + L
            is_term = j >= i1
            b_end = i1 if is_term else j
            bar_sc = score_interval(i, b_end, k)
            if not is_term:
                bar_sc += float(db_node[j])

            m_idx = cand_trained_idx[k]
            db_run_ok = (db_sup_start[i, k] >= 0.35) or (db_ahead_arr[i, k] >= 0.45)
            m_ahead = m_ahead_arr[i, k]
            bar_m_prob = float(m_probs[i:b_end, m_idx].mean()) if (m_idx >= 0 and b_end > i) else 0.0

            # Best incoming sustained state at i for target meter k:
            # a) Same meter (zero switch penalty)
            best_in_sc = float(s_row[k])
            best_in_src = (i, k, 0)

            # b) Switching from another sustained meter
            if best_s_v > NEG_INF * 0.5:
                if is_simple:
                    if k == k_44:
                        pen_sw = 7.0 if has_change else (13.0 if p44_ahead >= 0.70 else 14.0)
                    elif m_idx >= 0 and has_change and m_ahead >= 0.45 and p44_ahead < 0.50:
                        pen_sw = 8.0
                    else:
                        pen_sw = 80.0
                elif is_complex:
                    alt_78_44 = (
                        p78_ahead >= 0.18
                        and down_prob_s[i] >= 0.50
                        and (is_term or down_prob_s[b_end] >= 0.50)
                        and (has_change or (not is_term and ch_arr[b_end] >= 0.45))
                    )
                    if k == k_44:
                        if best_s_k == k_78 and alt_78_44:
                            pen_sw = 2.0 if has_change else 3.5
                        else:
                            pen_sw = 3.5 if has_change else (6.5 if p44_ahead >= 0.65 else 7.5)
                    elif k == k_78 and best_s_k == k_44 and alt_78_44:
                        pen_sw = 2.0 if has_change else 3.5
                    elif db_run_ok:
                        if db_sup_start[i, k] >= 0.35:
                            if cand_meters[k] == (7, 4):
                                pen_sw = 3.5 if has_change else 5.5
                            else:
                                pen_sw = 3.5 if has_change else 7.0
                        else:
                            pen_sw = 4.5 if has_change else 8.0
                    elif p44_ahead >= 0.85 and not has_change:
                        pen_sw = 40.0
                    elif m_idx >= 0 and (bar_m_prob >= 0.25 or m_ahead >= 0.25):
                        pen_sw = 3.5 if has_change else 6.0
                    elif m_idx >= 0 and has_change and down_prob_s[i] >= 0.55 and p44_ahead < 0.80:
                        pen_sw = 5.5
                    else:
                        pen_sw = 45.0
                else:
                    if k == k_44:
                        pen_sw = 7.0 if has_change else (13.0 if p44_ahead >= 0.70 else 14.0)
                    elif db_run_ok:
                        if db_sup_start[i, k] >= 0.35:
                            if cand_meters[k] == (7, 4):
                                pen_sw = 4.0 if ch_arr[i] >= 0.45 else 5.5
                            else:
                                pen_sw = 6.0 if (has_change and ch_arr[i] >= 0.60) else 13.5
                        else:
                            pen_sw = 7.5 if has_change else 14.0
                    elif p44_ahead >= 0.78:
                        pen_sw = 80.0
                    elif m_idx >= 0:
                        pen_sw = (7.0 if has_change else 14.0) if m_ahead >= 0.32 else 60.0
                    else:
                        pen_sw = 70.0
                cand_sw = best_s_v - pen_sw
                if cand_sw > best_in_sc:
                    best_in_sc = cand_sw
                    best_in_src = (i, best_s_k, 0)

            # c) Low-cost transition within additive meter family {3/4, 4/4, 5/4}
            p54_prev = (p54_cum[i] - p54_cum[i - 8]) / 8.0 if i >= i0 + 8 else 0.0
            exit_to_44 = (not is_simple) and (k == k_44) and (p44_ahead >= 0.65) and (p54_prev >= 0.30)
            if k in additive_ks and best_add_k >= 0 and (allow_additive or exit_to_44):
                skip_bar = float(cum_skip[b_end - 1] - cum_skip[i + 2]) if (b_end - 1 > i + 2) else 0.0
                if skip_bar < 0.50:
                    pen_add = 2.0 if exit_to_44 else (0.4 if ch_arr[i] >= 0.45 else 0.5)
                    cand_add = best_add_v - pen_add
                    if cand_add > best_in_sc:
                        best_in_sc = cand_add
                        best_in_src = (i, best_add_k, 0)

            # d) Exiting a 1-bar turnaround state at i into sustained meter k != ta_k
            for ta_k in turnaround_ks:
                if ta_k == k or ta_row[ta_k] <= NEG_INF * 0.5:
                    continue
                prev_s_k = int(back_ta[i, ta_k][1])
                if prev_s_k == k:
                    # Same-meter phase re-anchor (e.g., 4/4 -> ta -> 4/4): require genuine multi-bar phase win.
                    if k != k_44 and not (is_complex and k == k_34):
                        pen_ta_exit = 28.0
                    else:
                        prev_i = int(back_ta[i, ta_k][0])
                        step_L = int(lengths[k])
                        new_pts = [i + step_L * r for r in range(4) if i + step_L * r < i1]
                        old_pts = [prev_i + step_L * r for r in range(5) if prev_i + step_L * r < i1]
                        new_db_mean = float(down_prob_s[new_pts].mean()) if new_pts else float(down_prob_s[i])
                        old_db_mean = float(down_prob_s[old_pts].mean()) if old_pts else 0.0
                        phase_flip_ok = (len(new_pts) >= 2) and (
                            (new_db_mean - old_db_mean >= 0.25) or (old_db_mean < 0.35)
                        )
                        if not phase_flip_ok:
                            pen_ta_exit = 60.0
                        elif is_complex:
                            pen_ta_exit = 5.0 if (tempo_chg_e[i] or ch_arr[i] >= 0.45) else 10.0
                        elif tempo_chg_e[i] or (ch_arr[i] >= 0.65 and down_prob_s[i] >= 0.80):
                            pen_ta_exit = 12.0
                        elif (
                            cand_trained_idx[ta_k] >= 0
                            and ch_arr[i] >= 0.30
                            and float(
                                m_probs[max(i0, i - int(lengths[ta_k])) : i, cand_trained_idx[ta_k]].mean()
                            )
                            >= 0.25
                        ):
                            pen_ta_exit = 16.0
                        else:
                            pen_ta_exit = 24.0
                elif (cand_meters[k] == (7, 4) and db_sup_start[i, k] >= 0.35) or k == k_44:
                    pen_ta_exit = 0.0
                elif is_simple:
                    pen_ta_exit = 80.0
                elif is_complex and (db_sup_start[i, k] >= 0.35 or m_ahead >= 0.22 or bar_m_prob >= 0.22):
                    pen_ta_exit = 0.0
                elif p44_ahead < 0.68 and (db_sup_start[i, k] >= 0.35 or m_ahead >= 0.32):
                    pen_ta_exit = 8.0 if cand_meters[k] == (7, 8) else 0.0
                elif p44_ahead < 0.68 and db_run_ok:
                    pen_ta_exit = 8.0 if cand_meters[k] == (7, 8) else 5.0
                else:
                    pen_ta_exit = 80.0
                cand_ta = float(ta_row[ta_k]) - pen_ta_exit
                if cand_ta > best_in_sc:
                    best_in_sc = cand_ta
                    best_in_src = (i, ta_k, 1)

            if near_tempo[i] and k != dom_k and best_in_src[1] != k:
                best_in_sc -= 2.2
                if s_row[k] > best_in_sc:
                    best_in_sc, best_in_src = float(s_row[k]), (i,k,0)
            total_sc = best_in_sc + bar_sc
            if is_term:
                if total_sc > best_end_sc:
                    best_end_sc = total_sc
                    best_end_term = (i, k)
                    best_end_src = best_in_src
            elif total_sc > dp_s[j, k]:
                dp_s[j, k] = total_sc
                back_s[j, k] = best_in_src

        # 2) Single-bar turnaround transitions i -> j = i + L_ta (only from sustained states, j < i1).
        if contrast >= 0.15 and best_s_v > NEG_INF * 0.5 and (i == i0 or down_prob_s[i] >= 0.38):
            for ta_k in turnaround_ks:
                L_ta = int(lengths[ta_k])
                j = i + L_ta
                if j >= i1:
                    continue
                m_ta_idx = cand_trained_idx[ta_k]
                ta_m_mean = float(m_probs[i:j, m_ta_idx].mean()) if m_ta_idx >= 0 else 0.0
                ta_m_max = float(m_probs[i:j, m_ta_idx].max()) if m_ta_idx >= 0 else 0.0
                ta_ch_max = float(ch_arr[i : j + 1].max())
                ta_meter_backed = (m_ta_idx >= 0) and (
                    ta_m_mean >= 0.25 or (ta_m_max >= 0.50 and ta_ch_max >= 0.25)
                )
                min_j_db = 0.44 if ta_meter_backed else 0.50
                if not is_active_e[j] or down_prob_s[j] < min_j_db:
                    continue
                act_cnt = int(cum_act[j] - cum_act[i])
                if act_cnt == 0:
                    continue
                # Find best sustained predecessor != ta_k (same-meter belongs to sustained path).
                prev_k = -1
                prev_v = NEG_INF
                for pk in sustained_ks:
                    if pk != ta_k and s_row[pk] > prev_v:
                        prev_v = float(s_row[pk])
                        prev_k = pk
                if prev_k < 0 or prev_v <= NEG_INF * 0.5:
                    continue
                reanchor = (
                    (ch_arr[j] >= 0.60 and down_prob_s[j] >= 0.75)
                    or bool(tempo_chg_e[j])
                    or (float(db_sup_start[j].max()) >= 0.35)
                )
                if not reanchor and not ta_meter_backed and down_prob_s[j] < 0.68:
                    continue
                if is_complex:
                    ta_pen = 5.0 if reanchor else (7.0 if ta_meter_backed else 12.0)
                else:
                    ta_pen = 5.0 if reanchor else (8.0 if ta_meter_backed else 15.0)
                skip_raw = float(cum_skip[j - 1] - cum_skip[i + 2]) if (j - 1 > i + 2) else 0.0
                skip_pen = contrast * 4.5 * skip_raw
                p_align = float(full_phase_align[i, ta_k])
                ta_m_bonus = 6.0 * (act_cnt / 8.0) * ta_m_mean
                ta_ch_bonus = 3.0 * max(0.0, ta_ch_max - 0.20) if ta_meter_backed else 0.0
                ta_sc = (
                    prev_v
                    + 0.8 * (act_cnt / L_ta) * p_align
                    + ta_m_bonus
                    + ta_ch_bonus
                    - skip_pen
                    - ta_pen
                    - (2.2 if near_tempo[i] and ta_k != dom_k else 0.)
                    + float(db_node[j])
                )
                if ta_sc > dp_ta[j, ta_k]:
                    dp_ta[j, ta_k] = ta_sc
                    back_ta[j, ta_k] = (i, prev_k, 0)

    if best_end_sc <= NEG_INF * 0.5:
        return [(0.0, 4, 4)], float(e_beats[i0]), resolved_preset

    # Backtrack barlines (bar_start_idx, meter_idx).
    bars = [(int(best_end_term[0]), cand_meters[best_end_term[1]])]
    cur_i, cur_k, cur_mode = (int(best_end_src[0]), int(best_end_src[1]), int(best_end_src[2]))
    while cur_i >= 0:
        row = back_ta[cur_i, cur_k] if cur_mode == 1 else back_s[cur_i, cur_k]
        prev_i = int(row[0])
        prev_k = int(row[1])
        prev_mode = int(row[2])
        if prev_i < 0:
            break
        bars.append((prev_i, cand_meters[cur_k]))
        cur_i, cur_k, cur_mode = prev_i, prev_k, prev_mode
    bars.reverse()

    b0_e = float(e_beats[bars[0][0]])
    meter_segments: list[tuple[float, int, int]] = []
    prev_m: tuple[int, int] | None = None
    for b_idx, m in bars:
        if prev_m is None or m != prev_m:
            meter_segments.append((float(e_beats[b_idx]) - b0_e, int(m[0]), int(m[1])))
            prev_m = m
    if not meter_segments:
        return [(0.0, 4, 4)], float(e_beats[i0]), resolved_preset
    return meter_segments, b0_e, resolved_preset


def _apply_tail_tempo_hold(dp_segs, tracked, change_prob, bpm_track, running):
    """Hold pre-tail high-tempo segment through <=8.5s fadeout/white-noise drop lacking coherent reset."""
    if len(dp_segs) < 2:
        return dp_segs
    t_end = float(running[-1]) if len(running) else float(tracked["times"][-1])
    ch_np = change_prob.detach().cpu().numpy() if isinstance(change_prob, torch.Tensor) else np.asarray(change_prob)
    tr_np = bpm_track.detach().cpu().numpy() if isinstance(bpm_track, torch.Tensor) else np.asarray(bpm_track)

    seg_bpms = [_fit_beats_bpm(tracked, a, b) for a, b in dp_segs]
    for i_cut in range(1, len(dp_segs)):
        k_cut = int(dp_segs[i_cut][0])
        k_end = int(dp_segs[-1][1])
        tail_dur = t_end - float(tracked["times"][k_cut])
        if tail_dur > 8.5:
            continue

        bpm_prev = seg_bpms[i_cut - 1]
        earlier_bpms = seg_bpms[: i_cut - 1]
        if not earlier_bpms or bpm_prev < 1.05 * min(earlier_bpms):
            continue

        tail_bpms = seg_bpms[i_cut:]
        if bpm_prev < 1.08 * min(tail_bpms):
            continue

        a_tail = int(tracked["frames"][k_cut])
        b_tail = int(tracked["frames"][k_end]) + 1
        tr_slice = tr_np[a_tail:b_tail]
        tr_tail = float(np.median(tr_slice)) if len(tr_slice) else float(np.median(tr_np))
        if tr_tail > bpm_prev / 1.05:
            continue

        # Preserve genuine end-of-song tempo resets that have a change peak and sustained multi-bar lock.
        ch_peak = float(ch_np[max(0, a_tail - 6) : min(len(ch_np), a_tail + 7)].max()) if len(ch_np) else 0.0
        tail_beats = k_end - k_cut
        tail_span = float(tracked["times"][k_end] - tracked["times"][k_cut])
        real_mask = ~tracked["interp"][k_cut : k_end + 1]
        real_bp_mean = float(tracked["beat_p"][k_cut : k_end + 1][real_mask].mean()) if real_mask.any() else 0.0
        strong_bars = int(
            ((~tracked["interp"][k_cut : k_end + 1]) & (tracked["down_p"][k_cut : k_end + 1] >= 0.55)).sum()
        )
        coherent_tail = (len(tail_bpms) == 1) or (
            (max(tail_bpms) - min(tail_bpms)) / max(min(tail_bpms), 1.0) <= 0.05
        )
        genuine_reset = (
            ch_peak >= 0.45
            and coherent_tail
            and tail_beats >= 8
            and tail_span >= 2.5
            and real_bp_mean >= 0.65
            and strong_bars >= 2
        )
        if not genuine_reset:
            return dp_segs[:i_cut]
    return dp_segs


def _decode_tracked_tempo_segments(
    tracked,
    change_prob,
    theta,
    pulse_norm,
    beat,
    down,
    sub,
    bpm_track,
    running,
    active_global,
    stops,
    has_first_music,
    first_running,
    snap_integer_bpm,
    snap_max_drift_ms,
):
    """Segment tracked quarter-beats via DP, refine BPMs, and anchor continuous BeatGrid segments."""
    dp_segs = _partition_tempo_segments(tracked, change_prob)
    dp_segs = _merge_dp_segments(dp_segs, tracked, change_prob, beat=beat, down=down, bpm_track=bpm_track)
    held = _apply_tail_tempo_hold(dp_segs, tracked, change_prob, bpm_track, running)
    held_bpm = None
    if len(held) < len(dp_segs) and len(held) >= 3:
        # A final interpolated or slowing beat may be assigned to the last
        # staircase step. Measure its last complete bar before holding it.
        (a, b), (c, d), (lo, hi) = held[-3:]
        bar = b - a
        end = lo + bar
        if bar >= 3 and d - c == bar and hi == end + 1 and not tracked["interp"][[lo, end]].any():
            times = tracked["times"]
            beats = tracked["positions"][end] - tracked["positions"][lo]
            period = (times[end] - times[lo]) / beats
            last_dt = times[hi] - times[end]
            if tracked["beat_p"][end] >= 0.65 and last_dt - period >= 0.015:
                held_bpm = 60 / period
                if (
                    snap_integer_bpm
                    and abs(held_bpm - round(held_bpm)) * (times[end] - times[lo]) / held_bpm * 1000
                    <= snap_max_drift_ms
                ):
                    held_bpm = float(round(held_bpm))
                held = [*held[:-1], (lo, end)]
    dp_segs = held

    def run_to_phys(r_sec):
        phys = r_sec
        for st in stops:
            if st["time"] < phys - 1e-9:
                phys += st["duration"]
        return phys

    resolved = []
    subdivision = 2 if tracked.get("half_grid") else 1
    cur_run_start = 0.0
    for idx_s, (k0, k1) in enumerate(dp_segs):
        a = int(tracked["frames"][k0])
        b = int(tracked["frames"][k1]) + 1
        bpm_dp = _fit_beats_bpm(tracked, k0, k1)
        positions = tracked["positions"][k0:k1 + 1]
        has_half_steps = np.any(np.diff(positions) != 1)
        n_beats = float(positions[-1] - positions[0]) if has_half_steps else k1 - k0
        half_beats = (tracked["times"][k0:k1 + 1], positions % 1) if has_half_steps else None
        if idx_s == 0:
            bpm_e, pt_e, pw_e = _estimate_segment_bpm(
                a, b, theta, pulse_norm, beat, sub, bpm_track, running, active_global,
                step=1.0, seed_bpm=bpm_dp, half_beats=half_beats,
            )
            bpm_cand = bpm_e if len(pt_e) >= 6 else bpm_dp
            if snap_integer_bpm:
                fallback_ts = torch.tensor(tracked["times"][k0 : k1 + 1], dtype=torch.float64)
                fallback_ws = torch.ones_like(fallback_ts)
                if has_half_steps:
                    fallback_ts -= torch.from_numpy(positions % 1) * (60.0 / bpm_dp)
                bpm_cand = _snap_integer_bpm(
                    bpm_cand,
                    pt_e if len(pt_e) >= 6 else fallback_ts,
                    pw_e if len(pt_e) >= 6 else fallback_ws,
                    int(active_global[a:b].sum()),
                    step=1.0,
                    snap_max_drift_ms=snap_max_drift_ms,
                )
            p0 = 60.0 / bpm_cand
            ts0 = torch.tensor(tracked["times"][k0 : k1 + 1], dtype=torch.float64)
            ws0 = torch.tensor(np.where(tracked["interp"][k0 : k1 + 1], 0.35, 1.0), dtype=torch.float64)
            pulse_anchor = _circular_time(ts0 - torch.from_numpy(positions % 1) * p0, p0, ws0) % p0
            if p0 - pulse_anchor < 0.5 / FRAME_RATE:
                pulse_anchor = 0.0
            if has_first_music:
                kk = math.ceil((first_running - 0.5 * p0 - pulse_anchor) / p0 - 1e-9)
                anchor_sec = max(0.0, pulse_anchor + kk * p0)
            else:
                anchor_sec = pulse_anchor
            cur_run_start = anchor_sec
            bpm_f32 = float(torch.tensor(bpm_cand, dtype=torch.float32).item())
            elapsed = (tracked["times"][k1] - cur_run_start) * bpm_f32 / 60.0
            n_beats_actual = max(1, round(subdivision * elapsed) / subdivision)
            resolved.append((cur_run_start, 0.0, bpm_f32, n_beats_actual))
            cur_run_start = cur_run_start + n_beats_actual * (60.0 / bpm_f32)
        else:
            dt_target = max(0.2, float(tracked["times"][k1]) - cur_run_start)
            bpm_hit = 60.0 * n_beats / dt_target
            bpm_cand = bpm_hit if abs(bpm_hit - bpm_dp) / max(bpm_dp, 1.0) <= 0.04 else bpm_dp
            pt_e = torch.empty(0, dtype=torch.float64)
            pw_e = torch.empty(0, dtype=torch.float64)
            if n_beats >= 12:
                bpm_e, pt_e, pw_e = _estimate_segment_bpm(
                    a, b, theta, pulse_norm, beat, sub, bpm_track, running, active_global,
                    step=1.0, seed_bpm=bpm_dp, half_beats=half_beats,
                )
                if len(pt_e) >= 6:
                    bpm_cand = bpm_e
            if snap_integer_bpm:
                nearest = round(bpm_cand)
                if nearest >= 1 and abs(bpm_cand - nearest) <= max(0.45, 0.003 * bpm_cand):
                    if n_beats <= 12:
                        if abs(60.0 * n_beats / nearest - dt_target) * 1000.0 <= 15.0:
                            bpm_cand = float(nearest)
                    else:
                        ts_s = torch.tensor(tracked["times"][k0 : k1 + 1], dtype=torch.float64)
                        ws_s = torch.tensor(np.where(tracked["interp"][k0 : k1 + 1], 0.35, 1.0), dtype=torch.float64)
                        if has_half_steps:
                            ts_s = ts_s - torch.from_numpy(positions % 1) * (60.0 / bpm_cand)
                        bpm_cand = _snap_integer_bpm(
                            bpm_cand,
                            pt_e if (has_half_steps and len(pt_e) >= 6) else ts_s,
                            pw_e if (has_half_steps and len(pt_e) >= 6) else ws_s,
                            int(active_global[a:b].sum()),
                            step=1.0,
                            snap_max_drift_ms=snap_max_drift_ms,
                        )
            if held_bpm is not None and idx_s == len(dp_segs) - 1:
                bpm_cand = held_bpm
            bpm_f32 = float(torch.tensor(bpm_cand, dtype=torch.float32).item())
            # Anchor long segment boundaries directly to tracked["times"][k1] to prevent multi-bar drift.
            n_beats_actual = max(1, round(subdivision * dt_target * bpm_f32 / 60.0) / subdivision) if n_beats >= 12 else n_beats
            prev_beat = resolved[-1][1] + resolved[-1][3]
            resolved.append((cur_run_start, prev_beat, bpm_f32, n_beats_actual))
            cur_run_start = cur_run_start + n_beats_actual * (60.0 / bpm_f32)

    tempos_prov = []
    for run_s, beat_s, bpm_s, _nb in resolved:
        if not tempos_prov or abs(bpm_s - tempos_prov[-1]["bpm"]) >= 0.25:
            tempos_prov.append(dict(start_seconds=run_to_phys(run_s), beat=beat_s, bpm=bpm_s))
    return tempos_prov


def decode_full_song_grid(
    outputs,
    batch_index=0,
    metadata=None,
    *,
    decoder="auto",
    change_threshold=None,
    stop_threshold=0.85,
    min_stop_frames=3,
    min_segment_seconds=1.6,
    snap_integer_bpm=True,
    snap_max_drift_ms=2.0,
    preset="auto",
):
    if preset not in VALID_DECODER_PRESETS:
        raise ValueError(f"Unknown decoder preset {preset!r}; expected one of {VALID_DECODER_PRESETS}.")

    n = int(outputs["lengths"][batch_index])
    if n <= 0:
        resolved_preset = "simple" if preset == "auto" else preset
        meta = dict(
            offset_seconds=0.0,
            bpm=120.0,
            bpm_segments=[dict(start_seconds=0.0, beat=0.0, bpm=120.0)],
            numerator=4,
            denominator=4,
            meter_segments=[(0.0, 4, 4)],
            stops=[],
        )
        grid = BeatGrid(meta)
        return dict(
            meta,
            first_beat=0.0,
            grid=grid,
            beat_times=[],
            downbeat_times=[],
            meter_segment_dicts=[dict(beat=0.0, numerator=4, denominator=4)],
            decoder_preset=resolved_preset,
        )

    if decoder == "clean-lattice" or (decoder == "auto" and "audible_logit" in outputs):
        return decode_clean_grid(outputs, batch_index, metadata, snap=snap_integer_bpm, drift=snap_max_drift_ms,
                                 stop_threshold=stop_threshold, min_stop_frames=min_stop_frames)
    if change_threshold is None:
        change_threshold = 0.32 if "semicrf" in outputs else 0.72
    if "semicrf" in outputs:
        return decode_boundary_grid(outputs, batch_index, snap=snap_integer_bpm, drift=snap_max_drift_ms,
                                    stop_threshold=stop_threshold, min_stop_frames=min_stop_frames,
                                    change_threshold=change_threshold)
    def read(key):
        return outputs[key][batch_index, :n].detach().float().cpu()

    t = (torch.arange(n, dtype=torch.float64) + 0.5) / FRAME_RATE
    stop_prob = read("stop_logit").sigmoid()
    stopped, stops = _decode_stops(stop_prob, stop_threshold=stop_threshold, min_stop_frames=min_stop_frames)
    bpm_track = 120.0 * read("log_bpm").exp2()
    meter_scores = read("meter_logits")
    beat = read("beat_logit").sigmoid()
    down_logits = read("downbeat_logit")
    down = down_logits.sigmoid()
    sub = read("subframe_offset_ms") * 0.001
    pulse_phase = read("pulse_phase")
    pulse_norm = pulse_phase.norm(dim=-1)
    theta = torch.atan2(pulse_phase[:, 0], pulse_phase[:, 1]) / (2.0 * math.pi)
    bar_phase = read("bar_phase")
    bar_theta = torch.atan2(bar_phase[:, 0], bar_phase[:, 1]) / (2.0 * math.pi)
    change_prob = read("change_logit").sigmoid()

    bar_ids = _peaks(down, 0.50, 2)
    pulse_ids = _peaks(beat, 0.50, 2)
    meter_table, _ = _meter_table_from_logits_dim(meter_scores.shape[-1])
    evidence = BarEvidence(
        (t[bar_ids] + sub[bar_ids]).numpy(),
        down[bar_ids].numpy(),
        (t[pulse_ids] + sub[pulse_ids]).numpy(),
        beat.numpy(),
        down.numpy(),
        change_prob.numpy(),
        bpm_track.numpy(),
        meter_scores.numpy(),
        meter_table,
        FRAME_RATE,
    )
    retained = filter_stutter_stops(stops, evidence)
    if len(retained) != len(stops):
        stopped.zero_()
        for st in retained:
            stopped[round(st["time"] * FRAME_RATE) : round((st["time"] + st["duration"]) * FRAME_RATE)] = True
        stops = retained
    elapsed = (stopped.cumsum(0) - 0.5 * stopped.float()) / FRAME_RATE
    running = t - elapsed
    first_frame, beat_ids, down_ids = _first_music_frame(beat, down, stopped, pulse_norm)
    has_first_music = first_frame is not None
    first_running = float(running[first_frame] + sub[first_frame]) if has_first_music else None
    first_physical = float(t[first_frame] + sub[first_frame]) if has_first_music else None

    active_global = ~stopped
    if has_first_music and first_frame > 0 and active_global[first_frame:].sum() > 2:
        active_global = active_global.clone()
        active_global[:first_frame] = False

    # Preserve rare sub-eighth-note OTHER signatures (e.g., 12/16) directly on their native clock.
    other_sig = None
    weights_g = (active_global & (pulse_norm >= 0.2)).float()
    if float(weights_g.sum()) < 2:
        weights_g = (~stopped).float()
    if not weights_g.any():
        weights_g = torch.ones(n, dtype=torch.float32)
    _, other_idx = _meter_table_from_logits_dim(meter_scores.shape[-1])
    if int((meter_scores * weights_g[:, None]).sum(0).argmax()) == other_idx:
        vals_g = bpm_track[active_global]
        med_g = float((vals_g if len(vals_g) else bpm_track).median())
        num_o, den_o = _other_meter(theta, bar_theta, active_global if active_global.any() else ~stopped, med_g)
        if (num_o, den_o) not in VITERBI_METERS and den_o == 16:
            other_sig = (num_o, den_o)

    tracked = None
    if other_sig is None:
        tracked = _track_quarter_beats(
            running, beat, down, sub, bpm_track, meter_scores, stopped, first_frame
        )

    tempos_prov = None
    if tracked is not None and len(tracked["times"]) >= 2:
        tempos_prov = _decode_tracked_tempo_segments(
            tracked,
            change_prob,
            theta,
            pulse_norm,
            beat,
            down,
            sub,
            bpm_track,
            running,
            active_global,
            stops,
            has_first_music,
            first_running,
            snap_integer_bpm,
            snap_max_drift_ms,
        )
    else:
        gap = max(1, round(min_segment_seconds * FRAME_RATE))
        candidates = _peaks(change_prob, change_threshold, max(1, gap // 2))
        edges = [0]
        for idx in candidates.tolist():
            if idx - edges[-1] >= gap and n - idx >= gap:
                edges.append(idx)
        edges.append(n)
        raw_segs = []
        for a, b in zip(edges, edges[1:]):
            step = 4.0 / other_sig[1] if other_sig is not None else 1.0
            bpm_e, pt_e, pw_e = _estimate_segment_bpm(
                a, b, theta, pulse_norm, beat, sub, bpm_track, running, active_global, step=step
            )
            raw_segs.append((a, b, bpm_e, pt_e, pw_e, step))
        segs = _merge_tempo_segments(
            raw_segs, theta, pulse_norm, beat, sub, bpm_track, running, active_global, change_prob
        )
        resolved_segs = []
        for a, b, bpm_e, pt_e, pw_e, step in segs:
            if snap_integer_bpm:
                bpm_e = _snap_integer_bpm(
                    bpm_e, pt_e, pw_e, int(active_global[a:b].sum()), step=step, snap_max_drift_ms=snap_max_drift_ms
                )
            bpm_e = float(torch.tensor(bpm_e, dtype=torch.float32).item())
            resolved_segs.append((a, b, bpm_e, pt_e, pw_e, step))

        a0, b0, bpm0, pt0, pw0, step0 = resolved_segs[0]
        pulse_p0 = (60.0 / bpm0) * step0
        if len(pt0):
            pulse_anchor = _circular_time(pt0, pulse_p0, pw0) % pulse_p0
        else:
            w0 = (active_global[a0:b0] | ~stopped[a0:b0]).float()
            if not w0.any():
                w0 = torch.ones(b0 - a0, dtype=torch.float32)
            pulse_anchor = _circular_time(running[a0:b0] - theta[a0:b0] * pulse_p0, pulse_p0, w0) % pulse_p0
        if pulse_p0 - pulse_anchor < 0.5 / FRAME_RATE:
            pulse_anchor = 0.0

        if other_sig is not None and len(resolved_segs) == 1:
            num0, den0 = other_sig
            span = (60.0 / bpm0) * 4.0 * num0 / den0
            w0 = active_global.float() if active_global.any() else torch.ones(n, dtype=torch.float32)
            bar_anchor = _circular_time(running - bar_theta * span, span, w0)
            if len(down_ids):
                bar_anchor = _circular_time(running[down_ids] + sub[down_ids], span, down[down_ids])
            anchor = (pulse_anchor + round((bar_anchor - pulse_anchor) / pulse_p0) * pulse_p0) % span
            if span - anchor < 0.5 / FRAME_RATE:
                anchor = 0.0
            if has_first_music:
                k0 = math.ceil((first_running - 0.5 * pulse_p0 - anchor) / span - 1e-9)
                anchor = max(0.0, anchor + k0 * span)
            offset = anchor + sum(st["duration"] for st in stops if st["time"] < anchor)
            tempos = [dict(start_seconds=offset, beat=0.0, bpm=bpm0)]
            meters = [(0.0, num0, den0)]
            if preset == "auto":
                non44_g = 1.0 - meter_scores.softmax(-1)[:, 0]
                act_idx = active_global if active_global.any() else ~stopped
                non44_mean_g = float(non44_g[act_idx].mean()) if act_idx.any() else 0.0
                resolved_preset = "complex" if non44_mean_g >= 0.35 else "medium"
            else:
                resolved_preset = preset
        else:
            if has_first_music:
                k0 = math.ceil((first_running - 0.5 * pulse_p0 - pulse_anchor) / pulse_p0 - 1e-9)
                anchor_sec = max(0.0, pulse_anchor + k0 * pulse_p0)
            else:
                anchor_sec = pulse_anchor
            offset_prov = anchor_sec + sum(st["duration"] for st in stops if st["time"] < anchor_sec)
            tempos_prov = []
            for idx_s, (a, _b, bpm_s, _pt, _pw, step_s) in enumerate(resolved_segs):
                if idx_s == 0:
                    tempos_prov.append(dict(start_seconds=offset_prov, beat=0.0, bpm=bpm_s))
                else:
                    prev_grid = BeatGrid(
                        dict(offset_seconds=offset_prov, bpm_segments=tempos_prov, stops=stops, meter_segments=[(0.0, 4, 4)])
                    )
                    quantum = min(step_s, resolved_segs[idx_s - 1][5])
                    sb = round(prev_grid.second_to_beat(float(t[a])) / quantum) * quantum
                    sec_s = prev_grid.beat_to_second(sb)
                    if abs(bpm_s - tempos_prov[-1]["bpm"]) >= 0.25:
                        tempos_prov.append(dict(start_seconds=sec_s, beat=sb, bpm=bpm_s))
    explicit_bars = None
    if tempos_prov is not None:
        peaks = beat_ids[(beat[beat_ids] >= .65) & (pulse_norm[beat_ids] >= .15)]
        times = ((peaks.double() + .5) / FRAME_RATE + sub[peaks]).numpy()
        tempos_prov, stops = reconcile_tempo_islands(
            tempos_prov, stops, peaks.numpy(), times, bpm_track.numpy(), beat.numpy(), down.numpy(), pulse_phase.numpy(),
            FRAME_RATE, snap_max_drift_ms if snap_integer_bpm else None,
        )
        for stop in stops:
            left = max(0, math.ceil(stop["time"] * FRAME_RATE - .5))
            right = math.ceil((stop["time"] + stop["duration"]) * FRAME_RATE - .5)
            stopped[left:right] = True
        rows, dominant_meter, harmonic_intervals, anchored_bars = anchor_stable_tempo_bar_islands(
            tempos_prov, stops, evidence, snap_integer_bpm
        )
        tempos_prov = [dict(start_seconds=t, bpm=bpm) for t,bpm in rows]
        offset_prov = tempos_prov[0]["start_seconds"]
        leading = (has_first_music and float(down[first_frame]) >= .65
                   and leading_bar_origin(first_physical, tempos_prov[0]["bpm"], bar_phase, FRAME_RATE))
        if leading:
            offset_prov = 0.0
            tempos_prov[0] = dict(tempos_prov[0], start_seconds=0.0)
        prov_grid = BeatGrid(dict(offset_seconds=offset_prov, bpm_segments=tempos_prov, stops=stops))
        gaps = _physical_meter_gaps(prov_grid, down, sub, pulse_norm, stopped, meter_scores,
                                    allow_leading=leading, peaks=down_ids)

        def decode_section(start, end):
            rows, anchor, route = _decode_meter_viterbi(
                prov_grid,
                n,
                meter_scores,
                down_logits,
                change_prob,
                bar_phase,
                pulse_norm,
                stopped,
                first_physical,
                bpm_track=bpm_track,
                dominant_meter=dominant_meter,
                harmonic_intervals=harmonic_intervals,
                anchored_bars=anchored_bars,
                preset=preset,
                start=start,
                end=end,
            )
            return [(anchor + b, num, den) for b, num, den in rows], None, route

        absolute, explicit_bars, resolved_preset = decode_bar_sections(
            prov_grid, n / FRAME_RATE, gaps, decode_section
        )
        b0 = absolute[0][0]
        meters = [(b - b0, num, den) for b, num, den in absolute]
        if explicit_bars is not None:
            explicit_bars = [b - b0 for b in explicit_bars]
        offset = prov_grid.beat_to_second(b0)
        if abs(offset) < 0.5 / FRAME_RATE:
            offset = 0.0
        tempos = [dict(start_seconds=offset, beat=0.0, bpm=tempos_prov[0]["bpm"])]
        for segment in tempos_prov[1:]:
            tempos.append(dict(segment, beat=prov_grid.second_to_beat(segment["start_seconds"]) - b0))

    meta = dict(
        offset_seconds=offset,
        bpm=tempos[0]["bpm"],
        bpm_segments=tempos,
        numerator=meters[0][1],
        denominator=meters[0][2],
        meter_segments=meters,
        stops=stops,
    )
    if explicit_bars:
        meta["bar_beats"] = explicit_bars
    grid = BeatGrid(meta)
    for stop, beat_number in zip(stops, grid.stop_beats):
        stop["beat"] = beat_number
    first_beat = 0.0
    if has_first_music:
        pulse_q = 4.0 / meters[0][2] if meters[0][2] in (8, 16) else 1.0
        observed_beat = grid.second_to_beat(first_physical)
        first_beat = round(observed_beat / pulse_q) * pulse_q
        if explicit_bars:
            nearest = min(explicit_bars, key=lambda b: abs(b - observed_beat))
            if abs(grid.beat_to_second(nearest) - first_physical) <= .015:
                first_beat = nearest
        if abs(first_beat) < 1e-9:
            first_beat = 0.0
    pulses, bars = _event_times(
        grid, n / FRAME_RATE, start_beat=first_beat - 1e-6 if has_first_music else None
    )
    return dict(
        meta,
        first_beat=first_beat,
        grid=grid,
        beat_times=pulses.tolist(),
        downbeat_times=bars.tolist(),
        meter_segment_dicts=[dict(beat=b, numerator=num, denominator=den) for b, num, den in meters],
        decoder_preset=resolved_preset,
    )


def shift_timing_metadata(metadata, seconds):
    meta = dict(metadata)
    meta["offset_seconds"] = float(meta.get("offset_seconds", 0)) + seconds
    if "bpm_segments" in meta:
        meta["bpm_segments"] = [
            dict(s, start_seconds=s["start_seconds"] + seconds)
            if isinstance(s, dict)
            else [s[0] + seconds, *s[1:]]
            for s in meta["bpm_segments"]
        ]
    if "stops" in meta:
        meta["stops"] = [dict(s, time=s["time"] + seconds) for s in meta["stops"]]
    return meta


def annotate_known_grid_bars(outputs, metadata, *, batch_index=0, section_boundaries=()):
    """Infer barlines on the supplied clock, anchoring each stable bar phase independently."""
    grid = BeatGrid(metadata)
    n = int(outputs["lengths"][batch_index])
    if n == 0:
        return dict(metadata, meter_segments=[(0.0, 4, 4)], bar_beats=[])
    if "audible_logit" in outputs:
        return decode_clean_grid(outputs, batch_index, metadata)
    if "semicrf" in outputs:
        return decode_boundary_grid(outputs, batch_index, metadata)

    @lru_cache(None)
    def read(key):
        return outputs[key][batch_index, :n].detach().float().cpu()

    meters, _ = _meter_table_from_logits_dim(outputs["meter_logits"].shape[-1])
    stopped = torch.zeros(n, dtype=torch.bool)
    for a, b in zip(grid.stop_times, grid.stop_ends):
        left = max(0, math.ceil((a - grid.crop) / grid.speed * FRAME_RATE - .5))
        right = max(0, math.ceil((b - grid.crop) / grid.speed * FRAME_RATE - .5))
        stopped[left:right] = True
    sub = read("subframe_offset_ms") * .001 if "subframe_offset_ms" in outputs else torch.zeros(n)
    pulse_norm = read("pulse_phase").norm(dim=-1) if "pulse_phase" in outputs else torch.ones(n)
    gaps = _physical_meter_gaps(grid, read("downbeat_logit").sigmoid(), sub, pulse_norm, stopped,
                                read("meter_logits"), allow_leading=True)

    def decode_section(start, end):
        return _known_meter_section(grid, n, read, meters, section_boundaries, start, end)

    segments, bars, _ = decode_bar_sections(grid, n / FRAME_RATE, gaps, decode_section, return_bars=True)
    return dict(metadata, meter_segments=segments, bar_beats=bars)


def _known_meter_section(grid, n, read, active_meters, section_boundaries, start, end):
    first, beats = _section_eighths(grid, n, start, end)
    secs = grid.beat_to_second_tensor(beats)
    ix = (secs * FRAME_RATE - 0.5).round().long().clamp(0, n - 1)

    def sample(key):
        return read(key)[ix]

    candidates = list(range(len(active_meters)))
    lengths = np.array([int(8 * active_meters[k][0] / active_meters[k][1]) for k in candidates])
    ends = np.cumsum(lengths) - 1
    starts = np.r_[0, ends[:-1] + 1]
    klass = np.repeat(candidates, lengths)
    phase = np.concatenate([np.arange(k) / k for k in lengths])
    meter = sample("meter_logits").log_softmax(-1).numpy()
    logits = sample("downbeat_logit")
    down = torch.stack((F.logsigmoid(-logits), F.logsigmoid(logits)), -1).numpy()
    # A uniform downbeat floor carries no bar-position information. Weight
    # its likelihood by observed contrast so a silent song does not prefer
    # the longest meter merely because it contains fewer downbeats.
    prob = logits.sigmoid()
    contrast = float(prob.max() - prob.min())
    down *= contrast
    unit = sample("bar_phase").numpy()
    expected = np.stack((np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)))
    emission = down[:, (phase == 0).astype(int)] + unit @ expected + 0.5 * meter[:, klass]
    flat_meter = np.exp(meter.min(axis=1) - meter.max(axis=1))
    uncertainty = flat_meter * (1 - contrast) * (1 - np.linalg.norm(unit, axis=-1).clip(0, 1))
    emission[:, klass == 0] += 0.05 * uncertainty[:, None]
    prior = np.zeros(len(phase))
    prior[klass == 0] = 0.5
    # Weak or flat evidence defaults to 4/4, downbeat at score beat zero.
    prior[(klass == 0) & np.isclose(phase, (first % 8) / 8)] += 1
    scores = prior + emission[0]
    if start is not None:
        scores[phase != 0] = -np.inf
    change_logits = read("change_logit")
    change = F.max_pool1d(change_logits[None, None], 5, stride=1, padding=2)[0, 0, ix].numpy() > 0
    for boundary in section_boundaries:
        i = round(2 * (boundary - float(beats[0])))
        if 0 <= i < len(change):
            change[i] = True
    previous = np.roll(np.arange(len(phase)), 1)
    previous[starts] = ends
    back = np.empty((len(beats), len(phase)), dtype=np.int16)
    for i in range(1, len(beats)):
        pred = previous.copy()
        best = ends[np.argmax(scores[ends])]
        if change[i]:
            switch = scores[best] - 2 > scores[ends]
            pred[starts[switch]] = best
        penalty = (pred != previous) * 2
        scores = scores[pred] - penalty + emission[i]
        back[i] = pred
    path = np.empty(len(beats), dtype=int)
    path[-1] = scores.argmax()
    for i in range(len(beats) - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    meters = klass[path]
    changes = np.r_[0, np.flatnonzero(meters[1:] != meters[:-1]) + 1]
    bars = beats[phase[path] == 0].tolist()
    segments = [
        (
            float(beats[i]) - float(phase[path[i]] * 4 * active_meters[meters[i]][0] / active_meters[meters[i]][1]),
            *active_meters[meters[i]],
        )
        for i in changes
    ]
    return segments, bars, None


def _coalesce_sanitized_bars(raw_clean_bars, temp_grid, bridged_windows):
    """Remove sub-2-beat SV keyframe micro-measures while preserving structural downbeats and transition bars."""
    if len(raw_clean_bars) < 2:
        return list(raw_clean_bars), 0.0
    bars = [raw_clean_bars[0]]
    for b in raw_clean_bars[1:]:
        if b - bars[-1] >= 0.05:
            bars.append(b)
    if len(bars) < 2:
        return bars, 0.0

    expanded = [bars[0]]
    for b0, b1 in zip(bars, bars[1:]):
        t0, t1 = temp_grid.beat_to_second(b0), temp_grid.beat_to_second(b1)
        spanned_bridge = any(w0 < t1 - 0.003 and w1 > t0 + 0.003 for w0, w1 in bridged_windows)
        cur = b0
        if spanned_bridge and (b1 - b0) >= 6.0 - 1e-4:
            while b1 - cur >= 6.0 - 1e-4:
                cur = round(cur + 4.0, 6)
                expanded.append(cur)
        expanded.append(b1)
    bars = expanded

    is_struct = [(bars[i + 1] - bars[i]) >= 2.0 - 1e-4 for i in range(len(bars) - 1)]
    first_struct = next((i for i, ok in enumerate(is_struct) if ok), None)
    if first_struct is None:
        return bars, 0.0

    beat_shift = 0.0
    if first_struct > 0 and any(
        w0 <= temp_grid.beat_to_second(bars[first_struct]) and w1 >= temp_grid.beat_to_second(bars[0])
        for w0, w1 in bridged_windows
    ):
        lead_span = bars[first_struct] - bars[0]
        if lead_span < 2.0 - 1e-4:
            # A <2-beat #000 lead-in before the first real bar becomes a negative-beat weak pickup.
            beat_shift = bars[first_struct]
            bars = [round(b - beat_shift, 6) for b in bars[first_struct:]]
        else:
            kept_prefix = [bars[0]]
            cur = bars[0]
            while bars[first_struct] - cur >= 6.0 - 1e-4:
                cur = round(cur + 4.0, 6)
                kept_prefix.append(cur)
            bars = kept_prefix + bars[first_struct:]

    result = [bars[0]]
    i = 0
    m = len(bars) - 1
    while i < m:
        if (bars[i + 1] - result[-1]) >= 2.0 - 1e-4 and (i + 1 == m or (bars[i + 1] - bars[i]) >= 2.0 - 1e-4):
            result.append(bars[i + 1])
            i += 1
            continue
        j = i
        while j < m and (bars[j + 1] - bars[j]) < 2.0 - 1e-4:
            j += 1
        target = bars[j]
        # Outside the corrected SV, the general barline pass uses actual triggers.
        # Keep genuine short meters here, even on a chart containing a remote SV.
        t0, t1 = temp_grid.beat_to_second(bars[i]), temp_grid.beat_to_second(target)
        if not any(w0 < t1 - 0.003 and w1 > t0 + 0.003 for w0, w1 in bridged_windows):
            result.extend(bars[i + 1:j + 1])
            i = j
            continue
        span = target - bars[i]
        if span >= 2.0 - 1e-4:
            cur = bars[i]
            while target - cur >= 6.0 - 1e-4:
                cur = round(cur + 4.0, 6)
                result.append(cur)
            result.append(target)
        elif j < m and len(result) >= 2:
            prev_start = result[-2]
            result.pop()
            cur = prev_start
            while target - cur >= 6.0 - 1e-4:
                cur = round(cur + 4.0, 6)
                result.append(cur)
            result.append(target)
        i = j
    return result, beat_shift


def sanitize_bms_calibrator_metadata(
    metadata,
    *,
    triggers=None,
    waveform=None,
    sample_rate=SAMPLE_RATE,
    max_audible_seconds=MAX_CALIBRATOR_DURATION_SECONDS,
    return_status=False,
):
    """Filter unrecoverable BMS tempo gimmicks and bridge pure visual SV spikes.

    Sustained musical tempo sections in [50, 300] BPM (including genuine 0.5x half-time
    and <=300 BPM 2x sections) are preserved intact. Extreme (<50 or >300 BPM) or
    sub-2s bounce excursions that return cleanly to the surrounding musical beat grid
    are bridged on wall-clock time while re-projecting explicit BMS barlines and
    transition/fill bars (过门小节).
    """

    def finish(value, tag):
        if value is not None:
            value, rejection = sanitize_bms_barlines_and_meters(value, triggers)
            tag = rejection or tag
        return (value, tag) if return_status else value

    raw_segs = metadata.get("bpm_segments") or [
        [metadata.get("offset_seconds", 0.0), metadata.get("bpm", 120.0)]
    ]
    parsed = [
        (float(s["start_seconds"]), float(s["bpm"]))
        if isinstance(s, dict)
        else (float(s[0]), float(s[1]))
        for s in raw_segs
    ]
    if not parsed:
        return finish(None, "reject_empty_tempo")
    if any(not math.isfinite(t0) or not math.isfinite(b) or b <= 0.0 for t0, b in parsed):
        return finish(None, "reject_nonpositive_bpm")

    stops = []
    for s in metadata.get("stops", ()):
        st_t, st_d = float(s["time"]), float(s.get("duration", 0.0))
        if not (math.isfinite(st_t) and math.isfinite(st_d)) or st_d < 0.0:
            return finish(None, "reject_invalid_stop")
        if st_d > 0.0:
            stops.append(dict(time=st_t, duration=st_d, beat=s.get("beat")))
    stops.sort(key=lambda x: x["time"])

    wave_start = wave_end = wave_dur = None
    if waveform is not None:
        wave_arr = np.asarray(waveform)
        wave_dur = len(wave_arr) / float(sample_rate)
        audible = np.flatnonzero(np.abs(wave_arr) > 1e-4)
        if len(audible) == 0:
            return finish(None, "reject_silent")
        wave_start = float(audible[0]) / float(sample_rate)
        wave_end = float(audible[-1] + 1) / float(sample_rate)

    trig_start = trig_end = None
    if triggers is not None:
        if len(triggers) == 0:
            return finish(None, "reject_empty_triggers")
        trig_start = min(float(t.time) for t in triggers)
        trig_end = max(float(t.end if t.end is not None else t.time) for t in triggers)

    if wave_start is not None and trig_start is not None:
        audible_start = max(0.0, min(wave_start, trig_start))
        if wave_end - trig_end <= 15.0:
            audible_end = min(wave_end, max(audible_start + 0.5, trig_end + 1.0))
        else:
            audible_end = wave_end
        audio_duration = wave_dur
    elif wave_start is not None:
        audible_start, audible_end, audio_duration = wave_start, wave_end, wave_dur
    elif trig_start is not None:
        audible_start = max(0.0, trig_start)
        audible_end = max(audible_start + 0.5, trig_end)
        audio_duration = max(audible_end + 2.0, 1.0)
    else:
        audible_start = max(0.0, float(metadata.get("offset_seconds", 0.0)))
        audible_end = max(audible_start + 8.0, parsed[-1][0] + 8.0)
        audio_duration = audible_end + 1.0

    if max_audible_seconds is not None and (
        audible_end - audible_start > max_audible_seconds
        or audible_end > max_audible_seconds + 5.0
    ):
        return finish(None, "reject_excessive_duration")

    orig_grid = BeatGrid(metadata)
    t_horizon = max(audio_duration, audible_end + 0.5, parsed[-1][0] + 1.0)
    all_onsets = (
        sorted({round(float(t.time), 4) for t in triggers}) if triggers is not None else []
    )
    play_onsets = (
        sorted({round(float(t.time), 4) for t in triggers if not getattr(t, "bgm", False)})
        if triggers is not None
        else []
    )
    intervals = []
    for i, (t0, bpm) in enumerate(parsed):
        t1 = parsed[i + 1][0] if i + 1 < len(parsed) else t_horizon
        if t1 < t0 - 1e-9:
            return finish(None, "reject_nonmonotonic_segments")
        t1 = max(t0, t1)
        run_dur = max(0.0, orig_grid._running(t1) - orig_grid._running(t0))
        a0 = max(t0, audible_start - 0.05)
        a1 = min(t1, audible_end + 0.05)
        aud_wall = max(0.0, a1 - a0)
        aud_run = max(0.0, orig_grid._running(a1) - orig_grid._running(a0)) if a1 > a0 else 0.0
        if intervals and abs(intervals[-1]["bpm"] - bpm) <= 1e-4:
            intervals[-1]["t1"] = t1
            intervals[-1]["wall_dur"] += t1 - t0
            intervals[-1]["run_dur"] += run_dur
            intervals[-1]["aud_wall"] += aud_wall
            intervals[-1]["aud_run"] += aud_run
        else:
            intervals.append(
                dict(
                    t0=t0,
                    t1=t1,
                    bpm=bpm,
                    wall_dur=t1 - t0,
                    run_dur=run_dur,
                    aud_wall=aud_wall,
                    aud_run=aud_run,
                )
            )

    for seg in intervals:
        sa0 = max(seg["t0"], audible_start - 1e-4)
        sa1 = min(seg["t1"], audible_end + 1e-4)
        if sa1 > sa0 and all_onsets:
            seg["all_count"] = bisect_right(all_onsets, sa1 - 1e-4) - bisect_left(all_onsets, sa0 - 1e-4)
            seg["play_count"] = bisect_right(play_onsets, sa1 - 1e-4) - bisect_left(play_onsets, sa0 - 1e-4)
        else:
            seg["all_count"] = 0
            seg["play_count"] = 0

    def in_band(bpm):
        return MIN_CALIBRATOR_BPM <= bpm <= MAX_CALIBRATOR_BPM

    def match_harmonic_ratio(bpm, ref_bpm):
        ratio = bpm / ref_bpm
        for cand in HARMONIC_GIMMICK_RATIOS:
            if abs(ratio - cand) / cand <= 0.005:
                return cand
        return None

    bpm_weights = {}
    total_aud = 0.0
    weight_key = "aud_run" if sum(s["aud_run"] for s in intervals) > 0 else "aud_wall"
    for seg in intervals:
        w = seg[weight_key]
        if w > 0:
            key = round(seg["bpm"], 3)
            bpm_weights[key] = bpm_weights.get(key, 0.0) + w
            total_aud += w
    if not bpm_weights:
        return finish(None, "reject_no_audible_tempo")
    dominant_bpm = max(bpm_weights.items(), key=lambda kv: kv[1])[0]
    if not in_band(dominant_bpm):
        return finish(None, "reject_dominant_bpm_out_of_range")

    in_band_aud = sum(seg[weight_key] for seg in intervals if in_band(seg["bpm"]))
    if total_aud > 0 and in_band_aud / total_aud < 0.70:
        return finish(None, "reject_majority_out_of_band")

    if len(intervals) == 1 and all(st["duration"] <= 8.0 for st in stops):
        return finish(dict(metadata), "kept_unchanged_single")

    min_anchor_sec = min(MIN_ANCHOR_SECONDS, max(0.4, 0.2 * total_aud))
    dom_segs = [
        s for s in intervals if abs(s["bpm"] - dominant_bpm) <= 0.05 and s["aud_run"] >= 0.25
    ]
    dom_run = sum(s["aud_run"] for s in dom_segs)
    dom_all_nps = sum(s["all_count"] for s in dom_segs) / dom_run if dom_run >= 1.0 else 0.0
    dom_play_nps = sum(s["play_count"] for s in dom_segs) / dom_run if dom_run >= 1.0 else 0.0

    for idx, seg in enumerate(intervals):
        bpm = seg["bpm"]
        if not in_band(bpm):
            seg["is_gimmick"] = True
            continue
        r_dom = match_harmonic_ratio(bpm, dominant_bpm)
        if r_dom is None:
            seg["is_gimmick"] = False
            continue
        if seg["aud_run"] < min_anchor_sec:
            seg["is_gimmick"] = True
            continue
        bounded_by_dom = (
            0 < idx < len(intervals) - 1
            and abs(intervals[idx - 1]["bpm"] - dominant_bpm) <= 0.05
            and abs(intervals[idx + 1]["bpm"] - dominant_bpm) <= 0.05
            and seg["aud_run"] <= 16.0
        )
        if bounded_by_dom and dom_all_nps >= 2.0 and seg["aud_run"] >= 0.5:
            seg_all_nps = seg["all_count"] / seg["aud_run"]
            seg_play_nps = seg["play_count"] / seg["aud_run"]
            all_ratio = seg_all_nps / dom_all_nps
            play_ratio = seg_play_nps / dom_play_nps if dom_play_nps >= 1.0 else all_ratio
            if r_dom <= 0.75 and all_ratio >= max(0.80, r_dom + 0.30) and play_ratio >= max(0.52, r_dom * 1.05):
                seg["is_gimmick"] = True
                continue
            if r_dom >= 4.0 / 3.0 and all_ratio <= min(1.15, r_dom * 0.65) and play_ratio <= min(1.15, r_dom * 0.65):
                seg["is_gimmick"] = True
                continue
        seg["is_gimmick"] = False

    anchors = [
        idx
        for idx, seg in enumerate(intervals)
        if not seg["is_gimmick"]
        and (
            seg["aud_run"] >= min_anchor_sec
            or (abs(seg["bpm"] - dominant_bpm) <= 0.05 and seg["aud_run"] >= 0.40)
        )
    ]
    if not anchors:
        if (
            all(in_band(s["bpm"]) and not s["is_gimmick"] for s in intervals if s["aud_wall"] > 1e-3)
            and len(intervals) <= MAX_CALIBRATOR_SEGMENTS
            and all(st["duration"] <= 8.0 for st in stops)
        ):
            return finish(dict(metadata), "kept_unchanged_inband_multi")
        return finish(None, "reject_no_stable_anchor")

    has_gimmick = any(s["is_gimmick"] for s in intervals)
    has_micro_bounce = False
    for a_pos in range(len(anchors) - 1):
        i0, i1 = anchors[a_pos], anchors[a_pos + 1]
        if i1 > i0 + 1 and abs(intervals[i0]["bpm"] - intervals[i1]["bpm"]) <= 0.05:
            ref_b = intervals[i0]["bpm"]
            exc_slice = intervals[i0 + 1 : i1]
            exc_wall = sum(s["wall_dur"] for s in exc_slice)
            smooth_rit = all(
                in_band(s["bpm"])
                and not s["is_gimmick"]
                and s["aud_run"] >= 0.25
                and 0.76 <= s["bpm"] / ref_b <= 1.32
                for s in exc_slice
            )
            if not smooth_rit and (exc_wall < 2.0 or any(s["is_gimmick"] for s in exc_slice)):
                has_micro_bounce = True

    if (
        not has_gimmick
        and not has_micro_bounce
        and len(intervals) <= MAX_CALIBRATOR_SEGMENTS
        and all(st["duration"] <= 8.0 for st in stops)
    ):
        return finish(dict(metadata), "kept_unchanged_clean_multi")

    clean_bpm_segs = []
    kept_stops = []
    bridged_windows = []

    def collect_anchor_stops(t_a, t_b):
        for st in stops:
            if t_a - 1e-6 <= st["time"] < t_b - 1e-6:
                if st["duration"] > 8.0:
                    return False
                kept_stops.append(dict(time=st["time"], duration=st["duration"]))
        return True

    def preserve_inband_excursion(exc_segs, right_seg):
        for s in exc_segs:
            if abs(s["bpm"] - clean_bpm_segs[-1]["bpm"]) > 1e-4:
                clean_bpm_segs.append(dict(start_seconds=s["t0"], bpm=s["bpm"]))
            if not collect_anchor_stops(s["t0"], s["t1"]):
                return False
        if abs(right_seg["bpm"] - clean_bpm_segs[-1]["bpm"]) > 1e-4:
            clean_bpm_segs.append(dict(start_seconds=right_seg["t0"], bpm=right_seg["bpm"]))
        return True

    first_idx = anchors[0]
    first_anchor = intervals[first_idx]
    if first_idx > 0:
        bridged_windows.append((0.0, first_anchor["t0"]))
        start_t = 0.0
    else:
        start_t = first_anchor["t0"]
    clean_bpm_segs.append(dict(start_seconds=start_t, bpm=first_anchor["bpm"]))
    if not collect_anchor_stops(first_anchor["t0"], first_anchor["t1"]):
        return finish(None, "reject_extreme_stop")

    for a_pos in range(len(anchors) - 1):
        i0, i1 = anchors[a_pos], anchors[a_pos + 1]
        left_anc, right_anc = intervals[i0], intervals[i1]
        if i1 == i0 + 1:
            if abs(right_anc["bpm"] - clean_bpm_segs[-1]["bpm"]) > 1e-4:
                clean_bpm_segs.append(dict(start_seconds=right_anc["t0"], bpm=right_anc["bpm"]))
        else:
            exc = intervals[i0 + 1 : i1]
            t_in, t_out = left_anc["t1"], right_anc["t0"]
            dt = t_out - t_in
            exc_in_band = all(in_band(s["bpm"]) for s in exc)
            exc_has_gimmick = any(s["is_gimmick"] for s in exc)
            if (
                exc_in_band
                and not exc_has_gimmick
                and abs(left_anc["bpm"] - right_anc["bpm"]) > 0.05
                and all(s["run_dur"] >= 0.20 for s in exc)
            ):
                if not preserve_inband_excursion(exc, right_anc):
                    return finish(None, "reject_extreme_stop")
            elif abs(left_anc["bpm"] - right_anc["bpm"]) <= 0.05:
                smooth_return = (
                    exc_in_band
                    and not exc_has_gimmick
                    and all(
                        s["run_dur"] >= 0.25 and 0.76 <= s["bpm"] / left_anc["bpm"] <= 1.32
                        for s in exc
                    )
                )
                if smooth_return:
                    if not preserve_inband_excursion(exc, right_anc):
                        return finish(None, "reject_extreme_stop")
                else:
                    b0 = clean_bpm_segs[-1]["bpm"]
                    period0 = 60.0 / b0
                    dbeats = dt / period0
                    q_beats = round(dbeats * 4.0) / 4.0
                    err_ms = abs(dbeats - q_beats) * period0 * 1000.0
                    seg_origin = max(clean_bpm_segs[-1]["start_seconds"], first_anchor["t0"])
                    stops_since = sum(
                        st["duration"]
                        for st in kept_stops
                        if seg_origin - 1e-6 <= st["time"] < t_out - 1e-6
                    )
                    cum_beats = ((t_out - seg_origin) - stops_since) / period0
                    cum_err_ms = abs(cum_beats - round(cum_beats * 4.0) / 4.0) * period0 * 1000.0
                    q3_beats = round(dbeats * 3.0) / 3.0
                    err3_ms = abs(dbeats - q3_beats) * period0 * 1000.0
                    cum_err3_ms = abs(cum_beats - round(cum_beats * 3.0) / 3.0) * period0 * 1000.0
                    aligned_q4 = not (q_beats <= 0.0 and dt > 0.006) and err_ms <= 6.0 and cum_err_ms <= 6.0
                    aligned_q3 = not (q3_beats <= 0.0 and dt > 0.006) and err3_ms <= 3.0 and cum_err3_ms <= 3.0
                    if aligned_q4 or aligned_q3:
                        bridged_windows.append((t_in, t_out))
                    elif exc_in_band and all(s["run_dur"] >= 0.25 for s in exc):
                        if not preserve_inband_excursion(exc, right_anc):
                            return finish(None, "reject_extreme_stop")
                    else:
                        return finish(None, "reject_unaligned_return_excursion")
            else:
                b0, b1 = left_anc["bpm"], right_anc["bpm"]
                p0, p1 = 60.0 / b0, 60.0 / b1
                q0 = round((dt / p0) * 4.0) / 4.0
                err0_ms = abs(dt / p0 - q0) * p0 * 1000.0
                q1 = round((dt / p1) * 4.0) / 4.0
                err1_ms = abs(dt / p1 - q1) * p1 * 1000.0
                if dt <= 0.006 or (q0 > 0 and err0_ms <= 6.0):
                    bridged_windows.append((t_in, t_out))
                    if abs(b1 - clean_bpm_segs[-1]["bpm"]) > 1e-4:
                        clean_bpm_segs.append(dict(start_seconds=t_out, bpm=b1))
                elif q1 > 0 and err1_ms <= 6.0:
                    bridged_windows.append((t_in, t_out))
                    if abs(b1 - clean_bpm_segs[-1]["bpm"]) > 1e-4:
                        clean_bpm_segs.append(dict(start_seconds=t_in, bpm=b1))
                elif exc_in_band and all(s["run_dur"] >= 0.25 for s in exc):
                    if not preserve_inband_excursion(exc, right_anc):
                        return finish(None, "reject_extreme_stop")
                else:
                    return finish(None, "reject_unaligned_transition_excursion")

        if not collect_anchor_stops(right_anc["t0"], right_anc["t1"]):
            return finish(None, "reject_extreme_stop")

    last_idx = anchors[-1]
    if last_idx + 1 < len(intervals):
        aud_tail = sum(s["aud_wall"] for s in intervals[last_idx + 1 :] if not in_band(s["bpm"]))
        if aud_tail > 8.0:
            return finish(None, "reject_long_out_of_band_tail")
        bridged_windows.append((intervals[last_idx]["t1"], t_horizon))

    if len(clean_bpm_segs) > MAX_CALIBRATOR_SEGMENTS:
        return finish(None, "reject_too_many_segments")

    raw_bars = sorted(set(float(b) for b in (metadata.get("bar_beats") or ())))
    bar_times = [orig_grid.beat_to_second(b) for b in raw_bars]
    offset_sec = float(metadata.get("offset_seconds", 0.0))
    if first_idx > 0 and bar_times:
        post_anc_bars = [t for t in bar_times if t >= first_anchor["t0"] - 0.002]
        if post_anc_bars:
            t_ref = post_anc_bars[0]
            bar_span_sec = 4.0 * 60.0 / first_anchor["bpm"]
            k_bars = math.floor(t_ref / bar_span_sec + 1e-6)
            offset_sec = t_ref - k_bars * bar_span_sec
            clean_bpm_segs[0]["start_seconds"] = min(0.0, offset_sec)

    temp_grid = BeatGrid(
        dict(
            offset_seconds=offset_sec,
            bpm=clean_bpm_segs[0]["bpm"],
            bpm_segments=clean_bpm_segs,
            stops=kept_stops,
            numerator=4,
            denominator=4,
        )
    )

    def in_bridged_interior(t_val):
        return any(w0 + 0.003 < t_val < w1 - 0.003 for w0, w1 in bridged_windows)

    def snap_beat(b_val):
        q16 = round(b_val * 4.0) / 4.0
        if abs(b_val - q16) < 0.025:
            return float(q16)
        q24 = round(b_val * 6.0) / 6.0
        if abs(b_val - q24) < 0.015:
            return float(q24)
        return round(float(b_val), 6)

    clean_bars = []
    if bar_times:
        kept_bar_times = [t for t in bar_times if not in_bridged_interior(t) and t <= t_horizon + 5.0]
        raw_clean_bars = sorted({snap_beat(temp_grid.second_to_beat(t)) for t in kept_bar_times})
        clean_bars, beat_shift = _coalesce_sanitized_bars(raw_clean_bars, temp_grid, bridged_windows)
        if beat_shift != 0.0:
            offset_sec = temp_grid.beat_to_second(beat_shift)
            clean_bpm_segs[0]["start_seconds"] = min(clean_bpm_segs[0]["start_seconds"], offset_sec)

    meter_segments = []
    if len(clean_bars) >= 2:
        for b0, b1 in zip(clean_bars, clean_bars[1:]):
            if b1 > b0:
                n, d = multiplier_to_meter((b1 - b0) / 4.0)
                if not meter_segments or meter_segments[-1][1:] != [n, d] or abs(4.0 * n / d - (b1 - b0)) > 1e-6:
                    meter_segments.append([b0, n, d])
    if not meter_segments:
        meter_segments = [
            [0.0, int(metadata.get("numerator", 4) or 4), int(metadata.get("denominator", 4) or 4)]
        ]

    final_stops = [dict(time=st["time"], duration=st["duration"]) for st in kept_stops]
    final_grid = BeatGrid(
        dict(
            offset_seconds=offset_sec,
            bpm=clean_bpm_segs[0]["bpm"],
            bpm_segments=clean_bpm_segs,
            stops=final_stops,
            meter_segments=meter_segments,
            bar_beats=clean_bars,
        )
    )
    for st, b_num in zip(final_stops, final_grid.stop_beats):
        st["beat"] = b_num

    clean_meta = dict(metadata)
    clean_meta["offset_seconds"] = offset_sec
    clean_meta["bpm"] = clean_bpm_segs[0]["bpm"]
    clean_meta["bpm_segments"] = [[s["start_seconds"], s["bpm"]] for s in clean_bpm_segs]
    clean_meta["stops"] = final_stops
    clean_meta["bar_beats"] = clean_bars
    clean_meta["meter_segments"] = meter_segments
    clean_meta["numerator"] = meter_segments[0][1]
    clean_meta["denominator"] = meter_segments[0][2]
    clean_meta.pop("measure_multipliers", None)
    return finish(clean_meta, "sanitized")


def _select_calibrator_script(project):
    source = Path(project)
    if not source.is_dir():
        raw_candidates = [bms.parse_bms(source)]
    else:
        source = source.resolve()
        paths = sorted(bms._project_scripts(source).get(source, []))
        if not paths:
            raise FileNotFoundError(f"No BMS/BMSON script in {source}")
        raw_candidates = [bms.parse_bms(p) for p in paths]

    non_excluded = [
        c
        for c in raw_candidates
        if not excluded_song(dict(song_id=project.name, title=c.metadata.get("title") or c.path.stem))
        and not excluded_song(dict(title=str(c.path)))
    ]
    if not non_excluded:
        return None, "excluded"
    sounding = [c for c in non_excluded if c.triggers]
    if not sounding:
        return None, "empty"

    evaluated = []
    last_reason = "reject_gimmick"
    for c in sounding:
        clean_meta, status = sanitize_bms_calibrator_metadata(
            c.metadata, triggers=c.triggers, return_status=True
        )
        if clean_meta is None:
            last_reason = status
            continue
        deterministic = all(
            ch["command"].startswith("SET") or ch["range"] == 1
            for ch in c.metadata.get("random_choices", ())
        )
        count = len({(t.time, t.voice) for t in c.triggers})
        evaluated.append((c, status, deterministic, count))

    if not evaluated:
        return None, last_reason

    max_count = max(item[3] for item in evaluated)

    def rank(item):
        c, status, deterministic, count = item
        full_arrangement = count >= 0.85 * max_count
        was_sanitized = status == "sanitized"
        return (not full_arrangement, not deterministic, was_sanitized, -count, c.path.name.casefold())

    chosen = min(evaluated, key=rank)[0]
    chosen.metadata["representatives_considered"] = [
        dict(path=str(c.path), triggers=len(c.triggers)) for c in raw_candidates
    ]
    return chosen, "ok"


def _sampling_traits(metadata, samples):
    duration = samples / SAMPLE_RATE
    tempo, meter = tempo_meter_flags(metadata, duration)
    return dict(has_tempo_or_stop=tempo, has_non_4_4=meter,
                piecewise_eligible=bool(piecewise_eligible(metadata, duration)))


def _prepare_song(task):
    import soundfile as sf

    project, directory = task
    project = Path(project)
    slug = hashlib.sha256(project.as_posix().encode()).hexdigest()[:20]
    audio = Path(directory) / "songs" / (slug + ".flac")
    meta_file = audio.with_suffix(".pt")
    # Publish only completed renders. A failed retry must not delete an
    # existing song that an earlier manifest still references.
    with TemporaryDirectory(dir=audio.parent) as scratch:
        mix, labels = Path(scratch) / audio.name, Path(scratch) / meta_file.name
        try:
            chart, select_status = _select_calibrator_script(project)
            if chart is None:
                if select_status in ("excluded", "empty"):
                    return None
                return dict(project=str(project), filtered_gimmick=select_status)
            identity = dict(song_id=project.name, title=chart.metadata.get("title") or chart.path.stem)
            meta = bms.render_bms(chart, mix)
            if meta["sample_count"] == 0 or meta["mix_peak_before_gain"] <= 0:
                return None
            wave, _ = sf.read(mix, dtype="float32")
            clean_meta, post_status = sanitize_bms_calibrator_metadata(
                chart.metadata,
                triggers=chart.triggers,
                waveform=wave,
                sample_rate=SAMPLE_RATE,
                return_status=True,
            )
            if clean_meta is None:
                return dict(project=str(project), filtered_gimmick=post_status)
            max_cache_samples = round((MAX_CALIBRATOR_DURATION_SECONDS + 5.0) * SAMPLE_RATE)
            if len(wave) > max_cache_samples:
                wave = wave[:max_cache_samples]
                sf.write(mix, wave, SAMPLE_RATE)
        except (KeyError, ValueError, OSError, sf.LibsndfileError, subprocess.CalledProcessError) as exc:
            return dict(project=str(project), skipped_error=f"{type(exc).__name__}: {exc}")
        ref = song_reference_db(wave)
        torch.save(dict(metadata=clean_meta), labels)
        mix.replace(audio)
        labels.replace(meta_file)
    return dict(
        identity,
        **_sampling_traits(clean_meta, len(wave)),
        key=slug,
        audio="songs/" + audio.name,
        labels="songs/" + meta_file.name,
        sample_length=len(wave),
        song_ref_db=ref,
    )


def prepare_cache(roots, output, workers=12):
    from maidata.joint_training.checkpoint import source_identity

    output = Path(output)
    (output / "songs").mkdir(parents=True, exist_ok=True)
    projects = set()
    for root in roots:
        path = Path(root).resolve()
        found = [bms.bms_song_directory(path)] if path.is_file() else bms.discover_bms_projects(path)
        projects.update(p for p in found if not excluded_song(dict(song_id=p.name, title=str(p))))
    tasks = [(p, output) for p in sorted(projects)]
    if workers:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_prepare_song, tasks))
    else:
        results = list(map(_prepare_song, tasks))
    records = []
    skipped_errors = 0
    filtered_gimmicks = 0
    for row in results:
        if row is None:
            continue
        if "skipped_error" in row:
            skipped_errors += 1
            print(f"Skipping {row['project']}: {row['skipped_error']}", flush=True)
        elif "filtered_gimmick" in row:
            filtered_gimmicks += 1
        else:
            records.append(row)
    manifest = output / "manifest.json"
    tmp = manifest.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(
            dict(
                schema=CACHE_SCHEMA,
                producer=source_identity(),
                skipped_errors=skipped_errors,
                filtered_gimmicks=filtered_gimmicks,
                records=records,
            ),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    tmp.replace(manifest)
    return manifest


SPLIT_SCHEMA = "bms-beat-split.v18"
GENRE_TAXONOMY = (
    (
        "dnb_breakcore_jungle",
        r"drum\s*(?:['`n&\-\s]|and)*\s*bass|dnb|d'n'b|liquid\s*funk|neuro\s*funk|break\s*core|"
        r"jungle|amen|break\s*beats?|breaks|drum\s*step|tech\s*step|jump\s*up|ドラムンベース|ブレイク",
    ),
    (
        "artcore_orchestral_elec",
        r"art\s*core|symphoni|symphony|orchestra|gothic|renaissance|baroque|cinematic|soundtrack|"
        r"classical\s*crossover|fantas(?:y|ia)|requiem|elegy|rhapsody|dramatic|アートコア|オーケストラ|シンフォ|ファンタジ",
    ),
    (
        "hardcore_speedcore_gabber",
        r"hard\s*core|j-?core|uk\s*hard\s*core|ukhc|happy\s*hard\s*core|freeform|makina|french\s*core|"
        r"speed\s*core|gabba|gabber|terror|extratone|uptempo|crossbreed|hard\s*style|raw\s*style|rave|"
        r"hard\s*dance|hard\s*nrg|jump\s*style|tribe\s*core|dub\s*core|future\s*core|hybrid\s*core|"
        r"ハードコア|スピードコア|ガバ|ハードスタイル",
    ),
    (
        "trance_psy_eurobeat",
        r"trance|psy\s*trance|psychedelic|\bpsy\b|full\s*on|goa|uplifting|tech\s*trance|tech\s*dance|"
        r"euro\s*beat|euro\s*dance|j-?euro|hi-?nrg|para\s*para|hands?\s*up|handz\s*up|dance\s*speed|"
        r"トランス|ユーロ|サイケ",
    ),
    (
        "techno_schranz_minimal",
        r"techno|schranz|minimal|acid|industrial|hard\s*tek|tribal|テクノ|シュランツ|ミニマル|インダストリアル",
    ),
    (
        "house_edm_electro",
        r"house|electro(?!nica)|edm|progressive|future\s*bass|future\s*bounce|color\s*bass|dub\s*step|"
        r"brostep|riddim|glitch\s*hop|moombahton|complextro|garage|2-?step|ukg|disco|nu-?disco|funkot|"
        r"jersey|juke|footwork|trap|bass\s*music|big\s*beat|ハウス|ダブステップ|フューチャー",
    ),
    (
        "rock_metal_punk",
        r"rock|metal|punk|emo|screamo|djent|grunge|guitar|band|alternative|visual\s*kei|mixture|"
        r"ロック|メタル|パンク|ギター",
    ),
    (
        "jazz_funk_soul_lofi",
        r"jazz|swing|bebop|bossa|fusion|funk|soul|r&b|rnb|blues|hip[\s\-]*hop|trip\s*hop|rap|chill|"
        r"lo-?fi|city\s*pop|shibuya|lounge|easy\s*listening|shuffle|groove|skweee|"
        r"ジャズ|ファンク|ヒップホップ|ボッサ|ラウンジ",
    ),
    (
        "classical_piano_waltz",
        r"piano|classic|waltz|vals|sonata|etude|solo|chamber|acoustic|folk|celtic|irish|world|ethnic|"
        r"traditional|oriental|new\s*age|healing|ambient|ballad|contemporary|"
        r"ピアノ|クラシック|ワルツ|アコースティック|アンビエント|バラード|民族|和風",
    ),
    (
        "chiptune_game_8bit",
        r"chip\s*tune|8-?bit|famitracker|famicom|nes|vgm|falcom|game|arcade|retro|midi|チップ|ゲーム|レトロ",
    ),
    (
        "pop_vocal_kawaii",
        r"pop|kawaii|cute|idol|vocal|utau|vocaloid|anime|anison|denpa|happy|melodic|"
        r"ポップ|ボカロ|ボーカル|アイドル|電波|カワイイ",
    ),
    (
        "idm_electronica_experimental",
        r"\bidm\b|electronica|experimental|glitch|synth\s*wave|synth\s*beat|cyber|エレクトロニカ|エレクトロ",
    ),
)


def classify_bms_genre(raw_genre):
    text = unicodedata.normalize("NFKC", str(raw_genre or "")).strip().lower()
    if not text or not re.search(r"[a-z0-9\u3040-\u30ff\u4e00-\u9fff]", text):
        return "unspecified_other"
    for label, pattern in GENRE_TAXONOMY:
        if re.search(pattern, text):
            return label
    return "electronic_other"


def calibrator_bpm_band(bpm):
    for hi, label in (
        (95.0, "050_095"),
        (120.0, "095_120"),
        (135.0, "120_135"),
        (150.0, "135_150"),
        (168.0, "150_168"),
        (185.0, "168_185"),
        (210.0, "185_210"),
    ):
        if bpm < hi:
            return label
    return "210_300"


def _norm_dedup_token(text, *, strip_diff=False):
    if not text:
        return ""
    v = unicodedata.normalize("NFKC", str(text)).strip()
    if strip_diff:
        v = re.sub(
            r"\s*[\[\(\-～~](?:another|hyper|normal|beginner|light|7key|7keys|5key|5keys|9key|14key|"
            r"sp|dp|insane|black\s*another|leggendaria|ex|maniac|extra|hard|easy|standard|ogg|wav|"
            r"bms\s*edit|bms\s*ver|short\s*ver|cut\s*ver)[^\]\)]*[\]\)]?\s*$",
            "",
            v,
            flags=re.I,
        )
    return re.sub(r"[\s_\-\.\!\?\~\'\"`・／/\\:：;；,，\(\)\[\]\{\}†‡★☆♡♥]+", "", v.lower())


def _strip_charter_credit(artist):
    if not artist:
        return ""
    v = unicodedata.normalize("NFKC", str(artist)).strip()
    v = re.sub(r"(?i)\b(?:obj|bga|movie|illust|chart|pattern|diff|sp|dp)\s*[:：\.／/]\s*.*$", "", v)
    v = re.sub(r"\s*(?:/|／)\s*bms(?:sp)?\b.*$", "", v, flags=re.I)
    return _norm_dedup_token(v, strip_diff=False)


def summarize_calibrator_chart(chart, clean_meta, status):
    triggers = chart.triggers
    trig_start = min(float(t.time) for t in triggers)
    trig_end = max(float(t.end if t.end is not None else t.time) for t in triggers)
    trig_span = max(0.0, trig_end - trig_start)
    segs = [
        [float(s["start_seconds"]), float(s["bpm"])]
        if isinstance(s, dict)
        else [float(s[0]), float(s[1])]
        for s in (clean_meta.get("bpm_segments") or [[clean_meta.get("offset_seconds", 0.0), clean_meta["bpm"]]])
    ]
    bpm_durs = Counter()
    for si, (t0, bval) in enumerate(segs):
        t1 = segs[si + 1][0] if si + 1 < len(segs) else max(t0, trig_end)
        dur = max(0.0, min(t1, trig_end) - max(t0, trig_start))
        bpm_durs[round(bval, 3)] += dur
    dom_bpm = bpm_durs.most_common(1)[0][0] if bpm_durs else round(float(clean_meta["bpm"]), 3)
    n_bpm_segs = len(segs)
    has_stops = len(clean_meta.get("stops") or ()) > 0
    meters = {(int(m[1]), int(m[2])) for m in (clean_meta.get("meter_segments") or [[0.0, 4, 4]])}
    has_non44 = meters != {(4, 4)}
    if n_bpm_segs == 1 and status == "kept_unchanged_single":
        tempo_profile = "single_stop" if has_stops else "single_clean"
    elif n_bpm_segs == 1 and status == "sanitized":
        tempo_profile = "sanitized_to_single"
    elif n_bpm_segs > 1 and status == "sanitized":
        tempo_profile = "sanitized_multi"
    else:
        ratios = [s[1] / dom_bpm for s in segs if abs(s[1] - dom_bpm) > 0.05]
        is_harm = any(
            any(abs(r - c) / c <= 0.015 for c in HARMONIC_GIMMICK_RATIOS) for r in ratios
        )
        tempo_profile = "multi_harmonic" if is_harm else "multi_gradual_or_step"
    deterministic = all(
        ch["command"].startswith("SET") or ch["range"] == 1
        for ch in chart.metadata.get("random_choices", ())
    )
    uniq_trigs = len({(round(float(t.time), 4), t.voice) for t in triggers})
    raw_genre = str(chart.metadata.get("genre") or "").strip()
    return dict(
        raw_genre=raw_genre,
        genre_bucket=classify_bms_genre(raw_genre),
        status=status,
        dom_bpm=dom_bpm,
        bpm_band=calibrator_bpm_band(dom_bpm),
        trig_span=round(trig_span, 3),
        n_bpm_segs=n_bpm_segs,
        has_stops=has_stops,
        has_non44=has_non44,
        tempo_profile=tempo_profile,
        deterministic=deterministic,
        uniq_trigs=uniq_trigs,
    )


def _compact_supervision_metadata(clean_meta):
    segs = [
        [float(s["start_seconds"]), float(s["bpm"])]
        if isinstance(s, dict)
        else [float(s[0]), float(s[1])]
        for s in (clean_meta.get("bpm_segments") or [[clean_meta.get("offset_seconds", 0.0), clean_meta["bpm"]]])
    ]
    return dict(
        format=clean_meta.get("format"),
        title=clean_meta.get("title"),
        artist=clean_meta.get("artist"),
        genre=clean_meta.get("genre"),
        offset_seconds=float(clean_meta.get("offset_seconds", 0.0)),
        bpm=float(clean_meta.get("bpm", segs[0][1])),
        bpm_segments=segs,
        stops=[dict(s) for s in (clean_meta.get("stops") or ())],
        meter_segments=[list(m) for m in (clean_meta.get("meter_segments") or [[0.0, 4, 4]])],
        numerator=int(clean_meta.get("numerator", 4) or 4),
        denominator=int(clean_meta.get("denominator", 4) or 4),
        bar_beats=[float(b) for b in (clean_meta.get("bar_beats") or ())],
        **({"ambiguous_compound_bars": list(clean_meta["ambiguous_compound_bars"])}
           if "ambiguous_compound_bars" in clean_meta else {}),
    )


def _evaluate_zip_split_chunk(task):
    zip_path, rows = task
    out = []
    with zipfile.ZipFile(zip_path) as zf:
        for row in rows:
            chart_rel = row["chart"]
            info = zf.getinfo(chart_rel)
            if info.file_size == 0:
                out.append(("rejected", "empty_file", dict(project=row.get("project", chart_rel))))
                continue
            payload = zf.read(chart_rel)
            try:
                parsed = bms.parse_bms(chart_rel, payload=payload)
            except Exception as exc:
                out.append(
                    (
                        "rejected",
                        f"parse_error:{type(exc).__name__}",
                        dict(project=row.get("project", chart_rel), error=str(exc)),
                    )
                )
                continue
            project_name = row.get("project") or Path(chart_rel).stem
            collection = row.get("collection") or Path(chart_rel).parts[0]
            key = (
                Path(row["audio_member"]).parent.name
                if row.get("audio_member")
                else hashlib.sha256(f"{collection}/{project_name}".encode()).hexdigest()[:20]
            )
            title = parsed.metadata.get("title") or Path(project_name).name
            artist = parsed.metadata.get("artist") or ""
            if (
                excluded_song(dict(song_id=key, title=title))
                or excluded_song(dict(song_id=Path(project_name).name, title=project_name))
                or excluded_song(dict(title=chart_rel))
            ):
                out.append(("rejected", "excluded_song", dict(key=key, project=project_name, title=title)))
                continue
            if not parsed.triggers:
                out.append(("rejected", "empty_triggers", dict(key=key, project=project_name, title=title)))
                continue
            clean_meta, status = sanitize_bms_calibrator_metadata(
                parsed.metadata, triggers=parsed.triggers, return_status=True
            )
            if clean_meta is None:
                out.append(("rejected", status, dict(key=key, project=project_name, title=title)))
                continue
            summary = summarize_calibrator_chart(parsed, clean_meta, status)
            out.append(
                (
                    "ok",
                    status,
                    dict(
                        key=key,
                        song_id=Path(project_name).name,
                        collection=collection,
                        project=project_name,
                        chart=chart_rel,
                        chart_sha1=hashlib.sha1(payload).hexdigest(),
                        source_zip=str(zip_path),
                        source_archive=row.get("source_archive", ""),
                        source_chart=row.get("source_chart", ""),
                        audio_archive=row.get("audio_archive", ""),
                        audio_member=row.get("audio_member", ""),
                        title=title,
                        artist=artist,
                        norm_title=_norm_dedup_token(title, strip_diff=True),
                        norm_artist=_strip_charter_credit(artist),
                        **summary,
                        metadata=_compact_supervision_metadata(clean_meta),
                    ),
                )
            )
    return out


def _evaluate_dir_split_project(project_tuple):
    collection, project_str = project_tuple
    project = Path(project_str)
    key = hashlib.sha256(project.as_posix().encode()).hexdigest()[:20]
    try:
        chosen, select_status = _select_calibrator_script(project)
    except Exception as exc:
        return ("rejected", f"parse_error:{type(exc).__name__}", dict(project=project_str, error=str(exc)))
    if chosen is None:
        return ("rejected", select_status, dict(key=key, project=project_str))
    clean_meta, status = sanitize_bms_calibrator_metadata(
        chosen.metadata, triggers=chosen.triggers, return_status=True
    )
    if clean_meta is None:
        return ("rejected", status, dict(key=key, project=project_str))
    title = chosen.metadata.get("title") or project.name
    artist = chosen.metadata.get("artist") or ""
    summary = summarize_calibrator_chart(chosen, clean_meta, status)
    return (
        "ok",
        status,
        dict(
            key=key,
            song_id=project.name,
            collection=collection,
            project=project.as_posix(),
            chart=chosen.path.as_posix(),
            chart_sha1=hashlib.sha1(chosen.path.read_bytes()).hexdigest(),
            source_zip="",
            source_archive="",
            source_chart=chosen.path.name,
            audio_archive="",
            audio_member="",
            title=title,
            artist=artist,
            norm_title=_norm_dedup_token(title, strip_diff=True),
            norm_artist=_strip_charter_credit(artist),
            **summary,
            metadata=_compact_supervision_metadata(clean_meta),
        ),
    )


def _supervision_clusters(entries, *, reviewed_pairs=()):
    """Group identical source charts and musically matching song uploads."""
    parent = list(range(len(entries)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    by_sha1 = defaultdict(list)
    by_key = defaultdict(list)
    by_ta = defaultdict(list)
    by_t = defaultdict(list)
    for idx, item in enumerate(entries):
        by_key[item["key"]].append(idx)
        if item.get("chart_sha1"):
            by_sha1[item["chart_sha1"]].append(idx)
        nt, na = item["norm_title"], item["norm_artist"]
        if len(nt) >= 2 and len(na) >= 2:
            by_ta[(nt, na)].append(idx)
        if len(nt) >= 4 or (len(nt) >= 2 and len(na) == 1):
            by_t[nt].append(idx)

    for val_key, train_key in reviewed_pairs:
        if val_key in by_key and train_key in by_key:
            union(by_key[val_key][0], by_key[train_key][0])
    for group in (*by_key.values(), *by_ta.values(), *by_sha1.values()):
        for j in group[1:]:
            union(group[0], j)

    for idxs in by_t.values():
        if len(idxs) < 2:
            continue
        for p in range(len(idxs)):
            for q in range(p + 1, len(idxs)):
                i, j = idxs[p], idxs[q]
                if find(i) == find(j):
                    continue
                na_i, na_j = entries[i]["norm_artist"], entries[j]["norm_artist"]
                artist_sub = len(na_i) >= 3 and len(na_j) >= 3 and (na_i in na_j or na_j in na_i)
                same_timing = (
                    abs(entries[i]["dom_bpm"] - entries[j]["dom_bpm"]) <= 0.5
                    and abs(entries[i]["trig_span"] - entries[j]["trig_span"]) <= 5.0
                )
                if same_timing and (artist_sub or (len(na_i) == 1 and na_i == na_j)):
                    union(i, j)

    clusters = defaultdict(list)
    for idx, item in enumerate(entries):
        clusters[find(idx)].append(item)

    return clusters.values()


def _supervision_rank(row):
    return (row.get("status") == "sanitized", not row.get("deterministic"),
            row.get("collection") == "EventPackage", -row.get("uniq_trigs", 0), row["key"])


def _deduplicate_supervision_candidates(entries):
    kept, dropped = [], []
    for members in _supervision_clusters(entries):
        ranked = sorted(members, key=_supervision_rank)
        winner = dict(ranked[0])
        winner.pop("norm_title", None)
        winner.pop("norm_artist", None)
        if len(ranked) > 1:
            winner["dedup_cluster_size"] = len(ranked)
            for dup in ranked[1:]:
                dropped.append(
                    dict(
                        key=dup["key"],
                        collection=dup["collection"],
                        project=dup["project"],
                        title=dup["title"],
                        artist=dup["artist"],
                        kept_key=winner["key"],
                        kept_collection=winner["collection"],
                    )
                )
        else:
            winner["dedup_cluster_size"] = 1
        kept.append(winner)
    return kept, dropped


def _assign_stratified_splits(entries, *, val_ratio=0.10, seed=20261114):
    """Balance train/val jointly across tempo profile, meter, stops, genre, and continuous BPM."""
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("val_ratio must lie in [0.0, 1.0)")
    ordered = sorted(
        entries,
        key=lambda x: (
            x["tempo_profile"],
            x["has_non44"],
            x["has_stops"],
            x["genre_bucket"],
            round(float(x["dom_bpm"]), 3),
            x["collection"],
            x["key"],
        ),
    )
    n = len(ordered)
    target_val = round(n * val_ratio)
    if val_ratio > 0.0 and n >= 2 and target_val == 0:
        target_val = 1
    val_indices = set()
    if target_val > 0:
        rng = random.Random(seed)
        cycle = max(2, round(1.0 / val_ratio)) if val_ratio > 0 else 10
        perm = []
        for k in range(target_val):
            if k % cycle == 0:
                perm = list(range(cycle))
                rng.shuffle(perm)
            lo = round(k * n / target_val)
            hi = max(lo + 1, round((k + 1) * n / target_val))
            span = hi - lo
            decile = perm[k % cycle]
            pick = lo + min(span - 1, int((decile + rng.random()) * span / cycle))
            val_indices.add(pick)
    result = []
    for idx, item in enumerate(ordered):
        rec = dict(item)
        rec["split"] = "val" if idx in val_indices else "train"
        result.append(rec)
    return result


def _summarize_split_balance(records, rejected_counts, dropped_duplicates, *, val_ratio, seed):
    train_recs = [r for r in records if r["split"] == "train"]
    val_recs = [r for r in records if r["split"] == "val"]

    def breakdown(field):
        keys = sorted(
            {str(r[field]) for r in records},
            key=lambda k: (-sum(1 for r in records if str(r[field]) == k), k),
        )
        out = {}
        for k in keys:
            tr = sum(1 for r in train_recs if str(r[field]) == k)
            va = sum(1 for r in val_recs if str(r[field]) == k)
            tot = tr + va
            out[k] = dict(total=tot, train=tr, val=va, val_share=round(va / max(tot, 1), 4))
        return out

    def bpm_quantiles(sub):
        if not sub:
            return {}
        arr = np.array([float(r["dom_bpm"]) for r in sub], dtype=np.float64)
        qs = [0.05, 0.25, 0.50, 0.75, 0.95]
        vals = np.quantile(arr, qs)
        return {f"p{int(q * 100):02d}": round(float(v), 2) for q, v in zip(qs, vals)} | {
            "mean": round(float(arr.mean()), 2)
        }

    return dict(
        schema=SPLIT_SCHEMA,
        seed=seed,
        requested_val_ratio=val_ratio,
        total_candidates=len(records) + len(dropped_duplicates) + sum(rejected_counts.values()),
        rejected_total=sum(rejected_counts.values()),
        rejected_by_reason=dict(Counter(rejected_counts).most_common()),
        valid_before_dedup=len(records) + len(dropped_duplicates),
        cross_pack_duplicates_dropped=len(dropped_duplicates),
        retained_supervision_total=len(records),
        train_count=len(train_recs),
        val_count=len(val_recs),
        actual_val_share=round(len(val_recs) / max(len(records), 1), 4),
        bpm_quantiles=dict(train=bpm_quantiles(train_recs), val=bpm_quantiles(val_recs)),
        by_tempo_profile=breakdown("tempo_profile"),
        by_bpm_band=breakdown("bpm_band"),
        by_genre_bucket=breakdown("genre_bucket"),
        by_non44_meter=breakdown("has_non44"),
        by_has_stops=breakdown("has_stops"),
        by_collection=breakdown("collection"),
    )


def split_supervision_charts(
    *,
    charts_zips=(),
    bms_roots=(),
    output,
    val_ratio=0.10,
    seed=20261114,
    workers=12,
    write_charts_zip=None,
):
    """Filter unusable/duplicate BMS charts and produce a stratified train/val supervision split."""
    from maidata.joint_training.checkpoint import source_identity

    if not charts_zips and not bms_roots:
        raise ValueError("Provide at least one --charts-zip or --bms-root")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    rejected_counts = Counter()
    valid_entries = []

    for zpath in charts_zips:
        zpath = Path(zpath).resolve()
        with zipfile.ZipFile(zpath) as zf:
            if "index.csv" in zf.NameToInfo:
                idx_rows = list(csv.DictReader(io.StringIO(zf.read("index.csv").decode("utf-8-sig"))))
            else:
                idx_rows = [
                    dict(
                        collection=Path(name).parts[0] if len(Path(name).parts) > 1 else zpath.stem,
                        project=Path(name).stem,
                        chart=name,
                    )
                    for name in sorted(zf.namelist())
                    if Path(name).suffix.lower() in bms.EXTENSIONS
                ]
        chunks = [(zpath, idx_rows[i : i + 400]) for i in range(0, len(idx_rows), 400)]
        if workers and len(chunks) > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                batches = list(pool.map(_evaluate_zip_split_chunk, chunks))
        else:
            batches = list(map(_evaluate_zip_split_chunk, chunks))
        for batch in batches:
            for kind, reason, payload in batch:
                if kind == "ok":
                    valid_entries.append(payload)
                else:
                    rejected_counts[reason] += 1

    if bms_roots:
        dir_projects = []
        seen_dirs = set()
        for root in bms_roots:
            rpath = Path(root).resolve()
            found = [bms.bms_song_directory(rpath)] if rpath.is_file() else bms.discover_bms_projects(rpath)
            for p in sorted(found):
                if p not in seen_dirs:
                    seen_dirs.add(p)
                    dir_projects.append((rpath.name, str(p)))
        if workers and len(dir_projects) > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                dir_results = list(pool.map(_evaluate_dir_split_project, dir_projects))
        else:
            dir_results = list(map(_evaluate_dir_split_project, dir_projects))
        for kind, reason, payload in dir_results:
            if kind == "ok":
                valid_entries.append(payload)
            else:
                rejected_counts[reason] += 1

    deduped, dropped_duplicates = _deduplicate_supervision_candidates(valid_entries)
    assigned = _assign_stratified_splits(deduped, val_ratio=val_ratio, seed=seed)

    # Strip internal source_zip path from serialized records while keeping chart archive mapping.
    zip_lookup = {r["key"]: (r.pop("source_zip", ""), r["chart"]) for r in assigned}
    producer = source_identity()
    summary = _summarize_split_balance(
        assigned, rejected_counts, dropped_duplicates, val_ratio=val_ratio, seed=seed
    )
    summary["producer"] = producer

    train_records = [r for r in assigned if r["split"] == "train"]
    val_records = [r for r in assigned if r["split"] == "val"]

    def atomic_write_json(target, obj, *, indent=None):
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=indent), encoding="utf-8")
        tmp.replace(target)

    split_manifest_path = output / "split_manifest.json"
    train_split_path = output / "train_split.json"
    val_split_path = output / "val_split.json"
    summary_path = output / "split_summary.json"

    full_payload = dict(
        schema=SPLIT_SCHEMA,
        producer=producer,
        seed=seed,
        val_ratio=val_ratio,
        summary=summary,
        records=assigned,
        dropped_duplicates=dropped_duplicates,
    )
    atomic_write_json(split_manifest_path, full_payload)
    atomic_write_json(
        train_split_path,
        dict(schema=SPLIT_SCHEMA, producer=producer, split="train", count=len(train_records), records=train_records),
    )
    atomic_write_json(
        val_split_path,
        dict(schema=SPLIT_SCHEMA, producer=producer, split="val", count=len(val_records), records=val_records),
    )
    atomic_write_json(summary_path, summary, indent=2)

    if write_charts_zip:
        out_zip = Path(write_charts_zip)
        out_zip.parent.mkdir(parents=True, exist_ok=True)
        tmp_zip = out_zip.with_suffix(out_zip.suffix + ".tmp")
        by_src_zip = defaultdict(list)
        for r in assigned:
            src_z, member = zip_lookup[r["key"]]
            by_src_zip[src_z].append((r, member))
        with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as dst_zf:
            for src_z, items in sorted(by_src_zip.items()):
                if src_z:
                    with zipfile.ZipFile(src_z) as szf:
                        for _, member in items:
                            dst_zf.writestr(member, szf.read(member))
                else:
                    for r, member in items:
                        arc = f"charts/{r['collection']}/{r['key']}--{Path(member).name}"
                        dst_zf.write(member, arc)
            dst_zf.write(split_manifest_path, "split_manifest.json")
            dst_zf.write(train_split_path, "train_split.json")
            dst_zf.write(val_split_path, "val_split.json")
            dst_zf.write(summary_path, "split_summary.json")
        tmp_zip.replace(out_zip)

    return split_manifest_path


def _finalize_split_cache_song(task):
    import soundfile as sf

    row, staged_flac, chart_bytes, songs_dir = task
    key = row["key"]
    songs_dir = Path(songs_dir)
    staged_flac = Path(staged_flac)
    audio_dst = songs_dir / f"{key}.flac"
    labels_dst = songs_dir / f"{key}.pt"
    try:
        wave, sr = sf.read(staged_flac, dtype="float32")
        if sr != SAMPLE_RATE:
            import soxr

            wave = soxr.resample(wave, sr, SAMPLE_RATE)
        if wave.ndim > 1:
            wave = wave.mean(axis=-1)
        if len(wave) == 0 or not np.any(np.abs(wave) > 1e-4):
            staged_flac.unlink(missing_ok=True)
            return dict(key=key, split=row["split"], filtered_gimmick="reject_silent")
        if chart_bytes is not None:
            parsed = bms.parse_bms(row["chart"], payload=chart_bytes)
            clean_meta, post_status = sanitize_bms_calibrator_metadata(
                parsed.metadata,
                triggers=parsed.triggers,
                waveform=wave,
                sample_rate=SAMPLE_RATE,
                return_status=True,
            )
        else:
            clean_meta, post_status = sanitize_bms_calibrator_metadata(
                row["metadata"],
                waveform=wave,
                sample_rate=SAMPLE_RATE,
                return_status=True,
            )
        if clean_meta is None:
            staged_flac.unlink(missing_ok=True)
            return dict(key=key, split=row["split"], filtered_gimmick=post_status)
        max_cache_samples = round((MAX_CALIBRATOR_DURATION_SECONDS + 5.0) * SAMPLE_RATE)
        trimmed_tail = len(wave) > max_cache_samples
        if trimmed_tail:
            wave = wave[:max_cache_samples]
        ref = song_reference_db(wave)
        tmp_labels = songs_dir / f".{key}.pt.tmp"
        torch.save(dict(metadata=clean_meta), tmp_labels)
        if sr != SAMPLE_RATE or trimmed_tail:
            tmp_audio = songs_dir / f".{key}.flac.tmp"
            sf.write(tmp_audio, wave, SAMPLE_RATE, format="FLAC")
            staged_flac.unlink(missing_ok=True)
            tmp_audio.replace(audio_dst)
        else:
            staged_flac.replace(audio_dst)
        tmp_labels.replace(labels_dst)
        return dict(
            split=row["split"],
            record=dict(
                **_sampling_traits(clean_meta, len(wave)),
                song_id=row.get("song_id") or Path(row["project"]).name,
                title=row["title"],
                key=key,
                audio="songs/" + audio_dst.name,
                labels="songs/" + labels_dst.name,
                sample_length=len(wave),
                song_ref_db=ref,
            ),
        )
    except Exception as exc:
        staged_flac.unlink(missing_ok=True)
        return dict(key=key, split=row["split"], skipped_error=f"{type(exc).__name__}: {exc}")


def _focused_hard(row):
    metadata = row.get("metadata", {})
    bars = metadata.get("bar_beats") or ()
    return (
        row.get("tempo_profile", "single_clean") != "single_clean"
        or len(metadata.get("bpm_segments") or ()) > 1
        or row.get("has_stops", False)
        or bool(metadata.get("stops"))
        or row.get("has_non44", False)
        or (metadata.get("numerator", 4), metadata.get("denominator", 4)) != (4, 4)
        or any((num, den) != (4, 4) for _, num, den in metadata.get("meter_segments", ()))
        or any(
            not math.isclose(end - start, 4.0, rel_tol=0, abs_tol=1e-6) for start, end in zip(bars, bars[1:])
        )
    )


def _sample_focused_anchors(rows, count, fields, rng):
    """One seeded draw from each equally populated interval of the ordered strata."""
    ordered = sorted(rows, key=lambda r: (r[fields[0]], r[fields[1]], round(r["dom_bpm"], 2), r["key"]))
    count = min(count, len(ordered))
    return [
        ordered[rng.randrange(i * len(ordered) // count, (i + 1) * len(ordered) // count)]
        for i in range(count)
    ]


def _focused_split(rows, anchor_count, seed):
    rng = random.Random(seed)
    hard, fast, regular = [], [], []
    for row in rows:
        if excluded_song(row):
            continue
        if _focused_hard(row):
            hard.append(row)
            continue
        span = row.get("trig_span", 0)
        if (
            row.get("tempo_profile") == "single_clean"
            and row.get("deterministic")
            and span >= 45
            and row.get("uniq_trigs", 0) / max(span, 1) >= 0.35
        ):
            bucket = fast if row["bpm_band"] in ("185_210", "210_300") else regular
            bucket.append(row)

    # If the input is smaller than the full corpus, fill any shortfall from the other pool.
    quota = min(anchor_count, len(fast) + len(regular))
    fast_count = min(len(fast), max(round(quota * 0.4), quota - len(regular)))
    anchors = _sample_focused_anchors(fast, fast_count, ("bpm_band", "genre_bucket"), rng)
    anchors += _sample_focused_anchors(regular, quota - fast_count, ("genre_bucket", "bpm_band"), rng)
    retained = [dict(row, focused_role="hard") for row in hard]
    retained.extend(dict(row, focused_role="anchor") for row in anchors)
    retained.sort(key=lambda r: r["key"])
    summary = dict(input=len(rows), eligible_fast=len(fast), eligible_regular=len(regular))
    summary.update(_focused_selection_counts(retained))
    return retained, summary


def _focused_selection_counts(retained):
    hard = [r for r in retained if r["focused_role"] == "hard"]
    anchors = [r for r in retained if r["focused_role"] == "anchor"]
    fast_count = sum(r["bpm_band"] in ("185_210", "210_300") for r in anchors)
    summary = dict(hard=len(hard), anchors=len(anchors), total=len(retained),
                   fast_anchors=fast_count, regular_anchors=len(anchors) - fast_count)
    for field in ("tempo_profile", "bpm_band", "genre_bucket"):
        summary["by_" + field] = {
            name: dict(sorted(Counter(r.get(field, "unknown") for r in group).items()))
            for name, group in (("hard", hard), ("anchor", anchors))
        }
    return summary


def _repack_focused_cache(records, source_dirs, pack_dir):
    """Copy selected payloads in source order, keeping both files of each song on one shard."""
    import tarfile
    from contextlib import ExitStack

    member_keys = {row[field]: row["key"] for row in records for field in ("audio", "labels")}
    pack_dir = Path(pack_dir)
    pack_dir.mkdir(parents=True, exist_ok=True)
    song_shards = {}
    writers = {}
    completed_members = Counter()
    with TemporaryDirectory(dir=pack_dir, prefix=".focused-") as staging:
        staged = Path(staging)
        with ExitStack() as handles:
            source_archives = (
                archive for root in source_dirs for archive in sorted(root.glob("songs-[0-9]*.tar"))
            )
            for source in source_archives:
                with tarfile.open(source, "r|*", bufsize=1024 * 1024) as archive:
                    for member in archive:
                        name = member.name.removeprefix("./")
                        if name not in member_keys or not member.isfile():
                            continue
                        key = member_keys.pop(name)
                        shard = song_shards.setdefault(key, len(song_shards) // 256)
                        if shard not in writers:
                            path = staged / f"songs-focused-{shard:04d}.tar"
                            writers[shard] = handles.enter_context(tarfile.open(path, "w"))
                        with archive.extractfile(member) as payload:
                            writers[shard].addfile(member, payload)
                        completed_members[shard] += 1
                        if completed_members[shard] == 2 * 256:
                            writers[shard].close()
        if member_keys:
            raise FileNotFoundError(
                f"Source packs lack {len(member_keys)} selected cache files, e.g. {next(iter(member_keys))}"
            )
        names = []
        for path in sorted(staged.glob("songs-focused-*.tar")):
            names.append(path.name)
            path.replace(pack_dir / path.name)
        # A smaller replacement selection must not leave stale shards in this generated series.
        for old in pack_dir.glob("songs-focused-[0-9][0-9][0-9][0-9].tar"):
            if old.name not in names:
                old.unlink()
    return names


def _focused_cache_reference(source_root, path, new_root):
    absolute = Path(os.path.abspath(source_root / path))
    try:
        return Path(os.path.relpath(absolute, new_root)).as_posix()
    except ValueError:  # Different Windows volumes cannot share a relative path.
        return absolute.as_posix()


def select_stage2_subset(
    split_manifest,
    output,
    *,
    train_manifest=None,
    val_manifest=None,
    source_pack_dir=None,
    output_pack_dir=None,
    seed=20261114,
):
    """Select focused supervision and optionally reuse the corresponding accepted cache assets."""
    source = Path(split_manifest)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            split_doc = json.loads(archive.read("split_manifest.json"))
    else:
        split_doc = json.loads(source.read_bytes())
    if split_doc.get("schema") != SPLIT_SCHEMA:
        raise ValueError(f"Expected {SPLIT_SCHEMA} split manifest")
    cache_paths = {
        split: Path(path) for split, path in (("train", train_manifest), ("val", val_manifest))
        if path is not None
    }
    if source_pack_dir is not None or output_pack_dir is not None:
        output_pack_dir = Path(output_pack_dir or output)
        if source_pack_dir is not None:
            source_dirs = [Path(source_pack_dir)]
        else:
            source_dirs = list(dict.fromkeys(path.parent for path in cache_paths.values())) or [source.parent]
        if not cache_paths:
            # In the standard pack layout the accepted cache manifests live next to the source tars.
            root = source_dirs[0]
            defaults = {split: root / f"{split}.json" for split in ("train", "val")}
            cache_paths = {split: path for split, path in defaults.items() if path.is_file()} or defaults

    selected, stats, documents, cache_records = {}, {}, {}, []
    producer = split_doc.get("producer")
    for split, count in (("train", 500), ("val", 55)):
        rows = [row for row in split_doc["records"] if row["split"] == split]
        selected[split], stats[split] = _focused_split(rows, count, f"{seed}:{split}")
        records = selected[split]
        documents[f"focused_{split}_split.json"] = dict(
            schema=SPLIT_SCHEMA, producer=producer, split=split, count=len(records), records=records
        )
        if split in cache_paths:
            selected_keys = {r["key"] for r in records}
            cache_source = cache_paths[split]
            cache = json.loads(cache_source.read_bytes())
            if cache.get("schema") != CACHE_SCHEMA:
                raise ValueError(f"Expected {CACHE_SCHEMA} cache manifest at {cache_source}")
            kept = [r for r in cache["records"] if r["key"] in selected_keys and not excluded_song(r)]
            stats[split]["cache_records"] = len(kept)
            stats[split]["missing_cache_keys"] = sorted(selected_keys.difference(r["key"] for r in kept))
            if output_pack_dir is None:
                kept = [
                    dict(
                        r,
                        **{
                            field: _focused_cache_reference(cache_source.parent, r[field], output)
                            for field in ("audio", "labels")
                        },
                    )
                    for r in kept
                ]
            cache_records.extend(kept)
            documents[f"{split}.json"] = dict(cache, records=kept)
            if "count" in cache:
                documents[f"{split}.json"]["count"] = len(kept)

    summary = dict(seed=seed, producer=producer, splits=stats)
    if output_pack_dir is not None:
        summary["packs"] = _repack_focused_cache(cache_records, source_dirs, output_pack_dir)
    documents["focused_summary.json"] = summary
    documents["focused_split_manifest.json"] = dict(
        schema=SPLIT_SCHEMA,
        producer=producer,
        seed=seed,
        summary=summary,
        records=selected["train"] + selected["val"],
    )
    for name, doc in documents.items():
        path = output / name
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(doc, ensure_ascii=False, indent=2 if name == "focused_summary.json" else None),
            encoding="utf-8",
        )
        tmp.replace(path)
    return summary


def prepare_split_cache(
    split_manifest,
    output,
    *,
    flac_root=None,
    charts_zip=None,
    bms_roots=None,
    workers=12,
):
    """Materialize train.json, val.json, and songs/ for a pre-split supervision manifest."""
    from maidata.joint_training.checkpoint import source_identity

    if flac_root is None and not bms_roots:
        raise ValueError("prepare-split-cache requires --flac-root or --bms-root")
    manifest_path = Path(split_manifest)
    embedded_charts_zip = None
    if manifest_path.suffix.lower() == ".zip":
        embedded_charts_zip = manifest_path
        with zipfile.ZipFile(manifest_path) as zf:
            split_doc = json.loads(zf.read("split_manifest.json").decode("utf-8-sig"))
    else:
        split_doc = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if split_doc.get("schema") != SPLIT_SCHEMA:
        raise ValueError(f"Expected {SPLIT_SCHEMA} split manifest")

    chart_archive_path = Path(charts_zip) if charts_zip else embedded_charts_zip
    chart_payloads = {}
    if chart_archive_path is not None and chart_archive_path.is_file():
        with zipfile.ZipFile(chart_archive_path) as czf:
            names = czf.NameToInfo
            for r in split_doc["records"]:
                if r.get("chart") in names:
                    chart_payloads[r["key"]] = czf.read(r["chart"])

    output = Path(output)
    songs_dir = output / "songs"
    songs_dir.mkdir(parents=True, exist_ok=True)

    records = [r for r in split_doc["records"] if not excluded_song(r)]
    staged_tasks = []
    skipped_errors = {"train": 0, "val": 0}
    filtered_gimmicks = {"train": 0, "val": 0}

    bms_project_lookup = {}
    if bms_roots:
        for root in bms_roots:
            rpath = Path(root).resolve()
            found = [bms.bms_song_directory(rpath)] if rpath.is_file() else bms.discover_bms_projects(rpath)
            for p in found:
                bms_project_lookup[p.as_posix()] = p
                bms_project_lookup[(rpath.name + "/" + p.relative_to(rpath).as_posix())] = p
                bms_project_lookup[p.name] = p

    by_archive = defaultdict(list)
    direct_rows = []
    if flac_root is not None:
        flac_root = Path(flac_root)
        for r in records:
            arch = r.get("audio_archive") or ""
            if arch and (flac_root / arch).is_file():
                by_archive[arch].append(r)
            elif arch and (flac_root / Path(arch).name).is_file():
                by_archive[Path(arch).name].append(r)
            else:
                direct_rows.append(r)
    else:
        direct_rows = list(records)

    for arch_rel, arch_rows in sorted(by_archive.items()):
        with zipfile.ZipFile(flac_root / arch_rel) as azf:
            info_map = azf.NameToInfo
            present = []
            for r in arch_rows:
                member = r.get("audio_member") or f"songs/{r['key']}/mix.flac"
                if member not in info_map:
                    skipped_errors[r["split"]] += 1
                    print(f"Skipping {r['key']} ({r['project']}): missing {member} in {arch_rel}", flush=True)
                    continue
                present.append((info_map[member].header_offset, member, r))
            # Read in forward zip offset order to maximize sequential throughput on network/Drive mounts.
            for _, member, r in sorted(present, key=lambda x: x[0]):
                staged_path = songs_dir / f".stage-{r['key']}.flac"
                with azf.open(member) as src, staged_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
                staged_tasks.append((r, staged_path, chart_payloads.get(r["key"]), songs_dir))

    for r in direct_rows:
        key = r["key"]
        staged_path = songs_dir / f".stage-{key}.flac"
        src_file = None
        if flac_root is not None:
            candidates = [
                flac_root / r["audio_member"] if r.get("audio_member") else None,
                flac_root / "songs" / key / "mix.flac",
                flac_root / "songs" / f"{key}.flac",
                flac_root / f"{key}.flac",
            ]
            src_file = next((c for c in candidates if c is not None and c.is_file()), None)
        if src_file is not None:
            shutil.copyfile(src_file, staged_path)
            staged_tasks.append((r, staged_path, chart_payloads.get(key), songs_dir))
            continue
        proj_dir = (
            bms_project_lookup.get(r["project"])
            or bms_project_lookup.get(f"{r['collection']}/{r['project']}")
            or bms_project_lookup.get(Path(r["project"]).name)
        )
        if proj_dir is not None:
            try:
                chosen, _ = _select_calibrator_script(proj_dir)
                if chosen is None:
                    filtered_gimmicks[r["split"]] += 1
                    continue
                meta = bms.render_bms(chosen, staged_path)
                if meta["sample_count"] == 0 or meta["mix_peak_before_gain"] <= 0:
                    staged_path.unlink(missing_ok=True)
                    filtered_gimmicks[r["split"]] += 1
                    continue
                staged_tasks.append((r, staged_path, chosen.path.read_bytes(), songs_dir))
                continue
            except Exception as exc:
                staged_path.unlink(missing_ok=True)
                skipped_errors[r["split"]] += 1
                print(f"Skipping {key} ({r['project']}): {type(exc).__name__}: {exc}", flush=True)
                continue
        skipped_errors[r["split"]] += 1
        print(f"Skipping {key} ({r['project']}): audio source not found", flush=True)

    if workers and len(staged_tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_finalize_split_cache_song, staged_tasks))
    else:
        results = list(map(_finalize_split_cache_song, staged_tasks))

    split_records = {"train": [], "val": []}
    for res in results:
        sp = res["split"]
        if "skipped_error" in res:
            skipped_errors[sp] += 1
            print(f"Skipping {res['key']}: {res['skipped_error']}", flush=True)
        elif "filtered_gimmick" in res:
            filtered_gimmicks[sp] += 1
        else:
            split_records[sp].append(res["record"])

    producer = source_identity()
    written = {}
    for sp in ("train", "val"):
        manifest_file = output / f"{sp}.json"
        tmp = manifest_file.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(
                dict(
                    schema=CACHE_SCHEMA,
                    producer=producer,
                    split=sp,
                    skipped_errors=skipped_errors[sp],
                    filtered_gimmicks=filtered_gimmicks[sp],
                    records=split_records[sp],
                ),
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        tmp.replace(manifest_file)
        written[sp] = manifest_file
    return written


def load_mert_v2_backbone(model_name_or_path: str = DEFAULT_MERT_V2_MODEL, device="cuda"):
    from transformers import AutoModel

    dev = torch.device(device)
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    backbone = AutoModel.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    return backbone.eval().requires_grad_(False).to(device=dev, dtype=dtype)


class OnlineMERTv2Extractor:
    """Frozen online MERT-v2-FullSong extractor aligned to the 25 Hz (20 ms center) clock."""

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MERT_V2_MODEL,
        *,
        device="cuda",
        layers=None,
        backbone=None,
    ):
        self.model_id = str(model_name_or_path)
        self.device = torch.device(device)
        self.backbone = (
            load_mert_v2_backbone(self.model_id, self.device)
            if backbone is None
            else backbone.eval().requires_grad_(False).to(self.device)
        )
        cfg = getattr(self.backbone, "config", None)
        self.sampling_rate = int(getattr(cfg, "sampling_rate", 24000))
        self.ratio = int(getattr(cfg, "inputs_to_logits_ratio", 960))
        self.frame_rate = float(getattr(cfg, "frame_rate", self.sampling_rate / self.ratio))
        self.min_samples = max(int(getattr(cfg, "minimum_input_samples", 1025)), self.ratio)
        self.total_layers = int(getattr(cfg, "num_hidden_layers", 24))
        if layers is None:
            self.layer_indices = tuple(range(self.total_layers))
        elif isinstance(layers, int):
            self.layer_indices = tuple(range(int(layers)))
        else:
            self.layer_indices = tuple(int(i) for i in layers)
        if not self.layer_indices or any(i < 0 or i >= self.total_layers for i in self.layer_indices):
            raise ValueError(
                f"Invalid MERT layer indices {self.layer_indices} for {self.total_layers}-layer backbone"
            )
        self.num_layers = len(self.layer_indices)
        self.executed_layers = self.total_layers
        depth = max(self.layer_indices) + 1
        owner = getattr(self.backbone, "encoder", self.backbone)
        if depth < self.total_layers and hasattr(owner, "layers"):
            # Keep a caller-supplied full backbone usable by other extractors.
            truncated = copy(owner)
            truncated._modules = owner._modules.copy()
            truncated.layers = owner.layers[:depth]
            if owner is self.backbone:
                self.backbone = truncated
            else:
                self.backbone = copy(self.backbone)
                self.backbone._modules = self.backbone._modules.copy()
                self.backbone.encoder = truncated
            self.executed_layers = depth
        self.last_layer_only = self.layer_indices == (self.executed_layers - 1,) and hasattr(owner, "layers")
        self._on_device = True
        self._shift_banks = {}

    def _shifted_mel(self, wave, tables):
        front = self.backbone.feature_extractor
        signal = fast_resample(wave, SAMPLE_RATE, 48000)
        min_length = max(2 * self.min_samples, 2049)  # reflect padding for a centered 4096-point FFT
        if signal.shape[-1] < min_length:
            signal = F.pad(signal, (0, min_length - signal.shape[-1]))
        window = self._shift_banks.get("window")
        if window is None:
            window = self._shift_banks["window"] = torch.hann_window(4096, device=wave.device)
        power = torch.stft(signal, 4096, 480, window=window, return_complex=True).abs().square()
        spec = power[..., :-1].transpose(-1, -2)
        fb = front.mel_scale.fb
        mel = spec.new_empty((*spec.shape[:-1], fb.shape[1]))
        def bank_for(rate):
            bank = self._shift_banks.get(rate)
            if bank is None:
                source = torch.arange(spec.shape[-1], device=wave.device) / rate
                lo = source.long().clamp_max(len(fb)-1)
                hi = (lo+1).clamp_max(len(fb)-1)
                bank = torch.lerp(fb[lo], fb[hi], (source-lo).unsqueeze(-1)) * .25
                bank = bank.masked_fill((source >= len(fb))[:, None], 0.)
                self._shift_banks[rate] = bank
            return bank

        for i, segments in enumerate(tables):
            for left, right, (_, _, rate, _) in segment_frame_slices(frontend_segments(segments), spec.shape[1], 100):
                mel[i, left:right] = spec[i, left:right] @ bank_for(rate)
        return (front.amplitude_to_db(mel) - front.mel_mean) / front.mel_std.clamp_min(1e-5)

    def _encode_shifted(self, signal, tables):
        with torch.autocast(device_type=signal.device.type, enabled=False):
            mel = self._shifted_mel(signal, tables)
            subsample = self.backbone.subsampling_module
            h = subsample(mel.to(next(subsample.parameters()).dtype))
            positions = self.backbone.embed_positions(h)
            selected = {}
            for i, layer in enumerate(self.backbone.layers):
                h = layer(h, positions)
                if i in self.layer_indices:
                    selected[i] = h
            return torch.stack([selected[i] for i in self.layer_indices], 2)

    def _from_segments(self, wave, lengths, segments):
        # FullSong's global normalization and convolutions must exclude all
        # padding. Group equal lengths so the backbone still processes batches.
        groups = defaultdict(list)
        for i, (length, table) in enumerate(zip(lengths.tolist(), segments)):
            groups[length, any(rate != 1. for _, _, rate, _ in table)].append(i)
        rows = [None] * len(wave)
        dtype = torch.bfloat16 if wave.is_cuda else torch.float32
        for (length, shifted), ids in groups.items():
            signal = wave[:, :length] if len(ids) == len(wave) else wave[ids, :length]
            if shifted:
                states = self._encode_shifted(signal, [segments[i] for i in ids])
                count = math.ceil(length / MERT_STRIDE_SAMPLES)
                centers = (torch.arange(count, dtype=torch.float64, device=wave.device) + .5) / 25
                sizes = lengths.new_full((len(ids),), states.shape[1])
                values = sample_centers_batch(states.flatten(2).float(), sizes, self.frame_rate,
                                              centers, MERT_V2_ORIGIN_SECONDS).unflatten(-1, states.shape[2:])
            else:
                values = self(signal)
            for i, value in zip(ids, values.to(dtype)):
                rows[i] = value
        frames = max(1, math.ceil(wave.shape[-1] / MERT_STRIDE_SAMPLES))
        # Native alignment clamps beyond the last valid center to the last feature.
        return torch.stack([torch.cat((r, r[-1:].expand(frames-len(r), -1, -1))) for r in rows])

    def offload_to_cpu(self):
        """Move frozen MERT weights to CPU and release cached CUDA blocks."""
        if self.device.type != "cpu" and self._on_device:
            self._shift_banks.clear()
            self.backbone.to("cpu")
            self._on_device = False
            if self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def restore_to_device(self):
        """Move frozen MERT weights back to `self.device` before online extraction."""
        if self.device.type != "cpu" and not self._on_device:
            self.backbone.to(self.device)
            self._on_device = True

    @torch.no_grad()
    def __call__(self, waveforms: torch.Tensor, sample_lengths=None, *, speed_segments=None) -> torch.Tensor:
        if not self._on_device:
            self.restore_to_device()
        wave = waveforms.reshape(-1, waveforms.shape[-1]).to(device=self.device, dtype=torch.float32)
        if sample_lengths is None:
            raw_len = torch.full((len(wave),), wave.shape[-1], device=wave.device, dtype=torch.long)
        else:
            raw_len = torch.as_tensor(sample_lengths, device=wave.device, dtype=torch.long).reshape(-1)
        raw_len = raw_len.clamp(1, max(1, wave.shape[-1]))

        if speed_segments and any(rate != 1. for row in speed_segments for _, _, rate, _ in row):
            return self._from_segments(wave, raw_len, speed_segments)

        if len(wave) == 1 or bool((raw_len == raw_len[0]).all()):
            valid_samples = int(raw_len[0].item())
            wave_24k = fast_resample(wave[:, :valid_samples], SAMPLE_RATE, self.sampling_rate)
            if wave_24k.shape[-1] < self.min_samples:
                wave_24k = F.pad(wave_24k, (0, self.min_samples - wave_24k.shape[-1]))
            lengths_24k = torch.full(
                (len(wave_24k),), wave_24k.shape[-1], device=wave_24k.device, dtype=torch.long
            )
        else:
            rows_24k = []
            for row, n_44k in zip(wave, raw_len.tolist()):
                r24 = fast_resample(row[: max(1, int(n_44k))], SAMPLE_RATE, self.sampling_rate)
                if r24.shape[-1] < self.min_samples:
                    r24 = F.pad(r24, (0, self.min_samples - r24.shape[-1]))
                rows_24k.append(r24)
            lengths_24k = torch.tensor(
                [r.shape[-1] for r in rows_24k], device=wave.device, dtype=torch.long
            )
            wave_24k = nn.utils.rnn.pad_sequence(rows_24k, batch_first=True)

        attn_mask = torch.arange(wave_24k.shape[-1], device=wave_24k.device)[None, :] < lengths_24k[:, None]
        with torch.autocast(device_type=wave_24k.device.type, enabled=False):
            out = self.backbone(
                input_values=wave_24k,
                attention_mask=attn_mask,
                output_hidden_states=not self.last_layer_only,
                return_dict=True,
            )
        if self.last_layer_only:
            stacked = out.last_hidden_state.unsqueeze(2)
        else:
            states = out.hidden_states[-self.executed_layers:]
            stacked = torch.stack([states[i] for i in self.layer_indices], dim=2)
        if stacked.is_inference():
            stacked = stacked.clone()

        feat_mask = getattr(out, "feature_attention_mask", None)
        if feat_mask is not None:
            feat_lengths = feat_mask.sum(dim=1).to(dtype=torch.long).clamp(1, stacked.shape[1])
        else:
            feat_lengths = (lengths_24k // self.ratio).clamp(1, stacked.shape[1])

        n_125 = max(1, math.ceil(wave.shape[-1] * FRAME_RATE / SAMPLE_RATE))
        target_frames = (n_125 + 4) // 5
        centers = (torch.arange(target_frames, dtype=torch.float64, device=stacked.device) + 0.5) / 25.0
        aligned = sample_centers_batch(
            stacked.flatten(2).float(),
            feat_lengths,
            self.frame_rate,
            centers,
            MERT_V2_ORIGIN_SECONDS,
        )
        out_dtype = torch.bfloat16 if stacked.is_cuda else torch.float32
        result = aligned.unflatten(-1, stacked.shape[2:]).to(dtype=out_dtype)
        return result.clone() if result.is_inference() else result


def prepare_mert_cache(manifest, val_manifest, output, *, mert_model=DEFAULT_MERT_V2_MODEL,
                       layers=DEFAULT_STAGE2_MERT_LAYERS, batch_size=8, device="cuda"):
    import soundfile as sf

    root = Path(output)
    (root / "features").mkdir(parents=True, exist_ok=True)
    songs = {}
    for path in map(Path, (manifest, val_manifest)):
        for row in json.loads(path.read_bytes())["records"]:
            if not excluded_song(row):
                songs.setdefault(row["key"], (path.parent / row["audio"], row["sample_length"]))
    extractor = OnlineMERTv2Extractor(mert_model, device=device, layers=layers)
    ordered = sorted(songs, key=lambda key: songs[key][1])
    records = []
    for start in range(0, len(ordered), batch_size):
        keys = ordered[start:start + batch_size]
        waves = [torch.from_numpy(sf.read(songs[k][0], dtype="float32")[0]) for k in keys]
        lengths = [len(w) for w in waves]
        values = extractor(nn.utils.rnn.pad_sequence(waves, batch_first=True), lengths)
        for key, row, length in zip(keys, values, lengths):
            frames = math.ceil(length / MERT_STRIDE_SAMPLES)
            path = Path("features") / (key + ".pt")
            tmp = (root / path).with_suffix(".tmp")
            torch.save(dict(mert=row[:frames].to("cpu", torch.bfloat16).contiguous().clone(),
                            origin_seconds=0.02), tmp)
            tmp.replace(root / path)
            records.append(dict(key=key, features=path.as_posix()))
    destination = root / "mert_manifest.json"
    temp = destination.with_suffix(".tmp")
    temp.write_text(json.dumps(dict(model="MERT-v2-fullsong", model_source=str(mert_model),
                                    frame_rate=25, layers=list(layers), records=records)), encoding="utf-8")
    temp.replace(destination)
    return destination


def _prepare_mert_batch(values, frames, origin_seconds=0.02):
    """Align [B,T,L,D] to the acoustic 25 Hz clock without stretching a song."""
    length = values.shape[1]
    if origin_seconds == 0.02:
        if length >= frames:
            return values[:, :frames]
        tail = values[:, -1:].expand(-1, frames - length, -1, -1)
        return torch.cat((values, tail), dim=1)
    centers = (torch.arange(frames, dtype=torch.float64, device=values.device) + 0.5) / 25
    aligned = sample_centers_batch(values.flatten(2).float(), length, 25, centers, origin_seconds)
    return aligned.unflatten(-1, values.shape[2:]).to(values.dtype)


def _aligned_mert(path, frames, *, crop_mert_frames=0, speed=1.0, pad_sec=0.0, trim_sec=0.0):
    cached = torch.load(path, map_location="cpu", weights_only=True)
    return align_cached_mert(cached["mert"], frames, float(cached.get("origin_seconds", 0.02)),
                             crop_mert_frames=crop_mert_frames, speed=speed, pad_sec=pad_sec, trim_sec=trim_sec)


def align_cached_mert(mert, frames, origin=.02, *, crop_mert_frames=0, speed=1., pad_sec=0., trim_sec=0.):
    # Whole 25 Hz translations preserve the cached values exactly.
    pad, trim = round(pad_sec * 25), round(trim_sec * 25)
    if (speed == 1. and abs(origin - .02) < 1e-9 and (pad_sec or trim_sec)
            and abs(pad - pad_sec*25) < 1e-9 and abs(trim - trim_sec*25) < 1e-9):
        unpadded = torch.arange(frames, device=mert.device) + crop_mert_frames - pad
        return mert[(unpadded + trim).clamp(0, len(mert)-1)].masked_fill((unpadded < 0)[:, None, None], 0)
    if speed != 1.0 or pad_sec or trim_sec:
        centers = (torch.arange(frames, device=mert.device, dtype=torch.float64) + 0.5) / 25
        unpadded = centers + crop_mert_frames / 25 - pad_sec
        source = unpadded * speed + trim_sec
        values = sample_centers_batch(mert.flatten(1)[None].float(), len(mert), 25, source, origin)
        return values[0].unflatten(-1, mert.shape[1:]).to(mert.dtype).masked_fill(
            (unpadded < 0)[:, None, None], 0)
    if crop_mert_frames > 0:
        if abs(origin - 0.02) < 1e-9 and crop_mert_frames < len(mert):
            mert = mert[crop_mert_frames:]
        else:
            origin -= crop_mert_frames * 0.04
    return _prepare_mert_batch(mert[None], frames, origin)[0]


class _CalibratorItem(dict):
    song_key: str = ""


class BMSCalibratorDataset(torch.utils.data.Dataset):
    def __init__(self, manifest, config, *, training, mert_manifest=None):
        path = Path(manifest)
        self.root = path.parent
        data = path.read_bytes()
        cache = json.loads(data)
        if cache["schema"] != CACHE_SCHEMA:
            raise ValueError("BMS cache schema differs from v18")
        self.binding = dict(cache=hashlib.sha256(data).hexdigest(), producer=cache.get("producer"))
        max_song_sec = config.get("max_song_seconds", MAX_CALIBRATOR_DURATION_SECONDS)
        max_samples = (
            round((float(max_song_sec) + 5.0) * SAMPLE_RATE)
            if max_song_sec is not None and float(max_song_sec) > 0.0
            else None
        )
        holdout_titles = {_norm_dedup_token(title) for _, title in EXTERNAL_CALIBRATOR_SONGS}
        holdout_ids = {key for key, _ in EXTERNAL_CALIBRATOR_SONGS}
        def external_benchmark(row):
            return ((config.get("clean_v3") and str(row.get("song_id", "")) in holdout_ids) or
                    str(row.get("game", "")).casefold().startswith("maimai") or
                    _norm_dedup_token(row.get("title"), strip_diff=True) in holdout_titles)
        self.records = [
            row
            for row in cache["records"]
            if not excluded_song(row)
            and (not config.get("clean_lattice") or not external_benchmark(row))
            and (max_samples is None or int(row.get("sample_length", 0)) <= max_samples)
        ]
        self.cfg, self.training = config, training
        self.mert_files = {}
        self.mert_layers = 0
        self.repeats = [1] * len(self.records)
        self.piecewise_eligible = [r.get("piecewise_eligible") for r in self.records]
        tempo_mult = int(config.get("tempo_oversample", 1))
        meter_mult = int(config.get("non_4_4_oversample", 1))
        if training and max(tempo_mult, meter_mult) > 1:
            # Earlier train.json files carry keys but no score metadata. Their
            # original split manifest supplies the same keys without opening songs.
            lookup = {}
            source = config.get("oversample_manifest")
            if source is None:
                source = next((p for p in (self.root / "train_split.json", self.root / "split_manifest.json")
                               if p.is_file()), None)
            if source is not None:
                source = Path(source)
                if source.suffix.lower() == ".zip":
                    with zipfile.ZipFile(source) as archive:
                        data = archive.read("split_manifest.json")
                else:
                    data = source.read_bytes()
                self.binding["oversample_manifest"] = hashlib.sha256(data).hexdigest()
                lookup = {r["key"]: r for r in json.loads(data)["records"]}
            unknown = 0
            for i, row in enumerate(self.records):
                r = {**lookup.get(row["key"], {}), **row}
                if self.piecewise_eligible[i] is None and "metadata" in r:
                    self.piecewise_eligible[i] = piecewise_eligible(r["metadata"], r["sample_length"] / SAMPLE_RATE)
                if "has_tempo_or_stop" in r and "has_non_4_4" in r:
                    tempo, meter = r["has_tempo_or_stop"], r["has_non_4_4"]
                elif "metadata" in r:
                    tempo, meter = tempo_meter_flags(r["metadata"], r["sample_length"] / SAMPLE_RATE)
                else:
                    unknown += 1
                    continue
                self.repeats[i] = max(tempo_mult if tempo else 1, meter_mult if meter else 1)
            if unknown:
                print(f"{unknown} training rows lack tempo/meter metadata and have one occurrence; "
                      "pass --oversample-manifest with the original split manifest for these rows.", flush=True)
        self.hybrid = bool(config.get("clean_lattice") and mert_manifest is not None
                           and config.get("speed_probability", 0.) + config.get("piecewise_tempo_probability", 0.) > 0 and training)
        if config["stage"] == 2:
            if mert_manifest is not None:
                mert_path = Path(mert_manifest)
                data = mert_path.read_bytes()
                mert = json.loads(data)
                if (mert["model"] != "MERT-v2-fullsong" or mert["frame_rate"] != 25
                        or ((config.get("boundary_model") or config.get("clean_lattice")) and tuple(mert["layers"]) != DEFAULT_STAGE2_MERT_LAYERS)):
                    raise ValueError("Stage 2 requires its configured 25 Hz MERT-v2-fullsong cache (boundary model: L17)")
                self.mert_layers = len(mert["layers"])
                self.mert_files = {
                    r["key"]: mert_path.parent / r["features"] for r in mert["records"] if not excluded_song(r)
                }
                missing = [r["key"] for r in self.records if r["key"] not in self.mert_files]
                if missing:
                    raise ValueError(
                        f"Stage 2 MERT manifest lacks {len(missing)} song key(s), e.g. {missing[:3]}"
                    )
                self.binding["mert"] = hashlib.sha256(data).hexdigest()
            if mert_manifest is None or self.hybrid:
                self.binding["mert_model"] = str(config.get("mert_model", DEFAULT_MERT_V2_MODEL))
                layers = config.get("mert_layers", list(DEFAULT_STAGE2_MERT_LAYERS) if config.get("clean_lattice") else None)
                if layers is not None:
                    self.binding["mert_layers"] = layers

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import soundfile as sf

        force_speed = None
        occurrence = 0
        if isinstance(index, tuple):
            if len(index) == 4:
                epoch, index, force_speed, occurrence = index
            elif len(index) == 3:
                epoch, index, force_speed = index
            else:
                epoch, index = index
        else:
            epoch = 0
        row = self.records[index]
        seed = [self.cfg["seed"], epoch, index]
        if occurrence:
            seed.append(occurrence)
        rng = np.random.default_rng(seed)
        wave, _ = sf.read(self.root / row["audio"], dtype="float32")
        raw_samples = len(wave)
        ref = row["song_ref_db"]
        labels = torch.load(self.root / row["labels"], weights_only=True)
        v3 = self.cfg.get("clean_v3", False)
        metadata = labels["metadata"]
        speed_segments = []
        trim_floor = 1e-4
        if self.cfg.get("clean_lattice"):
            # Preserve a quiet attack in low-level masters before the worker
            # computes the relative loudness envelope on the augmented waveform.
            trim_floor = min(trim_floor, 10 ** ((ref - 48) / 20))
        audible = np.flatnonzero(np.abs(wave) > trim_floor)
        first_sound = int(audible[0]) if len(audible) else 0
        trim = 0
        pad_samples = 0
        speed = 1.0
        gain = 0.0
        max_silence = float(self.cfg.get("max_leading_silence_seconds", 0.0))
        if self.training and (self.cfg.get("boundary_model") or self.cfg.get("clean_lattice") or not self.mert_files):
            if v3:
                piece_prob = self.cfg.get("piecewise_tempo_probability", 0.)
                total = piece_prob + self.cfg["speed_probability"]
                draw = rng.random()
                use_speed = draw < total if force_speed is None else bool(force_speed)
                piecewise = draw < piece_prob
                if isinstance(force_speed, str):
                    piecewise = force_speed == "piecewise"
                elif force_speed:
                    piecewise = draw < piece_prob / max(total, 1e-12)
                if use_speed and piecewise:
                    wave, metadata, speed_segments = apply_piecewise_tempo_augmentation(wave, metadata, self.cfg, rng)
                if use_speed and not speed_segments:
                    lo, hi = _clamped_speed_range(metadata, raw_samples, self.cfg)
                    # Half-semitone speed grid makes frontend cancellation exact.
                    shifts = [m for m in range(math.ceil(24*math.log2(lo)), math.floor(24*math.log2(hi))+1) if m]
                    if shifts:
                        speed = 2 ** (int(rng.choice(shifts)) / 24)
                if speed_segments:
                    # Leading silence can span a cut; map the detected onset on the new clock.
                    original = BeatGrid(labels["metadata"])
                    first_sound = max(0, round(BeatGrid(metadata).beat_to_second(
                        original.second_to_beat(first_sound / SAMPLE_RATE)) * SAMPLE_RATE))
            else:
                use_speed = (rng.random() < self.cfg["speed_probability"]) if force_speed is None else bool(force_speed)
            if use_speed and not v3:
                bounds = (_clamped_speed_range(labels["metadata"], raw_samples, self.cfg)
                          if self.cfg.get("clean_lattice") else self.cfg["speed_range"])
                speed = float(rng.uniform(*bounds))
            cached_clock = self.mert_files and self.cfg.get("clean_lattice") and speed == 1. and not speed_segments
            if max_silence > 0.0 and first_sound > 1024:
                trim = first_sound - 1024
                if cached_clock:
                    trim -= trim % MERT_STRIDE_SAMPLES
                wave = wave[trim:]
            if speed != 1.:
                import soxr

                wave = soxr.resample(wave, SAMPLE_RATE * speed, SAMPLE_RATE)
            if max_silence > 0.0:
                pad_samples = round(float(rng.uniform(0.0, max_silence)) * SAMPLE_RATE)
                if cached_clock:
                    pad_samples -= pad_samples % MERT_STRIDE_SAMPLES
                if v3 and (speed_segments or speed != 1.):
                    cap = round(self.cfg.get("speed_max_seconds", 350.) * SAMPLE_RATE)
                    pad_samples = min(pad_samples, max(0, cap - len(wave)))
                if pad_samples > 0:
                    wave = np.concatenate((np.zeros(pad_samples, dtype=np.float32), wave))
            gain = float(rng.uniform(-self.cfg["gain_jitter_db"], self.cfg["gain_jitter_db"]))

        crop_mert_frames = 0
        recompute_ref = bool(speed_segments) or trim > 0 or speed != 1.0 or pad_samples > 0
        max_train_sec = self.cfg.get("max_train_seconds", DEFAULT_MAX_TRAIN_SECONDS)
        if self.training and max_train_sec is not None and float(max_train_sec) > 0.0:
            win_mert_frames = max(1, round(float(max_train_sec) * 25))
            win_samples = win_mert_frames * MERT_STRIDE_SAMPLES
            if len(wave) > win_samples:
                # Keep the t=0 intro / leading-silence / Bar-0 offset anchor 50% of the time,
                # and sample a 25 Hz-clock-aligned random window across the rest of the song 50% of the time.
                if rng.random() >= 0.5:
                    max_start_frames = (len(wave) - win_samples) // MERT_STRIDE_SAMPLES
                    crop_mert_frames = int(rng.integers(0, max_start_frames + 1))
                crop_samples = crop_mert_frames * MERT_STRIDE_SAMPLES
                wave = wave[crop_samples : crop_samples + win_samples]
                recompute_ref = True

        masks = None
        if recompute_ref:
            if self.cfg.get("clean_lattice"):
                ref, masks = song_ref_and_audibility(wave, _k_sections(), gain)
            else:
                ref = song_reference_db(wave)
        if gain != 0.0:
            wave = wave * np.float32(10 ** (gain / 20))
            if masks is None:
                ref += gain
        if masks is None and self.cfg.get("clean_lattice"):
            masks = audibility_masks(wave, ref, _k_sections())

        pad_sec = pad_samples / SAMPLE_RATE
        trim_sec = trim / SAMPLE_RATE
        crop_sec = crop_mert_frames * 0.04
        raw_active_start = pad_sec + max(0, first_sound - trim) / (SAMPLE_RATE * speed)
        active_start_sec = max(0.0, raw_active_start - crop_sec)
        meta = shift_timing_metadata(metadata, pad_sec * speed - trim_sec)
        meta["speed"] = speed
        if crop_sec > 0.0:
            meta["crop_start"] = float(meta.get("crop_start", 0.0)) + crop_sec * speed
        grid = BeatGrid(meta)
        n = max(1, math.ceil(len(wave) * FRAME_RATE / SAMPLE_RATE))
        targets = calibrator_targets_from_grid(
            grid, n, active_start_seconds=active_start_sec, extended_meters=self.cfg.get("extended_meters", False),
            boundary_model=self.cfg.get("boundary_model", False), clean_lattice=self.cfg.get("clean_lattice", False),
            clean_v3=v3, audio_masks=masks,
        )
        item = _CalibratorItem(
            waveform=torch.from_numpy(wave),
            targets=targets,
            song_ref_db=ref,
        )
        item.song_key = str(row.get("key", index))
        if self.cfg.get("clean_lattice"):
            item["pitch_semitones"] = None
            if v3:
                if speed_segments:
                    shift = pad_sec - trim_sec - crop_sec
                    end = len(wave) / SAMPLE_RATE
                    table = [(a+shift, b+shift, r, p) for a,b,r,p in speed_segments]
                    # Extend the first/last bank through leading/trailing padding.
                    table[0] = (min(0., table[0][0]), *table[0][1:])
                    table[-1] = (table[-1][0], max(end, table[-1][1]), *table[-1][2:])
                    item["speed_segments"] = [(max(0., a), min(end, b), r, p)
                                              for a,b,r,p in table if a < end and b > 0.]
                elif speed != 1.:
                    item["speed_segments"] = [(0., len(wave)/SAMPLE_RATE, speed, -12*math.log2(speed))]
                if self.training and self.cfg["stage"] == 1 and not speed_segments and speed == 1.:
                    item["pitch_semitones"] = mel_pitch_shift(1., self.cfg, rng)
            elif self.training and self.cfg["stage"] == 1 and "pitch_probability" in self.cfg:
                item["pitch_semitones"] = mel_pitch_shift(speed, self.cfg, rng)
        if self.cfg["stage"] == 2 and self.mert_files:
            if self.cfg.get("clean_lattice"):
                item["mert_needs_online"] = self.hybrid and (speed != 1. or bool(speed_segments))
                if item["mert_needs_online"]:
                    return item
            item["mert"] = _aligned_mert(
                self.mert_files[row["key"]], (n + 4) // 5, crop_mert_frames=crop_mert_frames,
                speed=speed, pad_sec=pad_sec, trim_sec=trim_sec
            )
        return item


class SongBatches(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset, self.size, self.epoch = dataset, batch_size, 0

    def __len__(self):
        return math.ceil(sum(getattr(self.dataset, "repeats", [1] * len(self.dataset))) / self.size)

    def __iter__(self):
        rng = random.Random(self.dataset.cfg["seed"] + self.epoch)
        max_train_sec = self.dataset.cfg.get("max_train_seconds", DEFAULT_MAX_TRAIN_SECONDS)
        cap_samples = (
            round(float(max_train_sec) * SAMPLE_RATE)
            if self.dataset.training and max_train_sec is not None and float(max_train_sec) > 0.0
            else None
        )

        def effective_length(i):
            raw_len = int(self.dataset.records[i]["sample_length"])
            eff = min(raw_len, cap_samples) if cap_samples is not None else raw_len
            # Bounded +-4% length jitter mixes neighboring songs across epochs while
            # strictly preventing >8.3% intra-batch duration mismatches or tail padding bloat.
            return eff * rng.uniform(0.96, 1.04) if self.dataset.training else eff

        repeats = getattr(self.dataset, "repeats", [1] * len(self.dataset))
        ordered = sorted(((i, occ) for i, copies in enumerate(repeats) for occ in range(copies)),
                         key=lambda row: effective_length(row[0]))
        batches = [ordered[i : i + self.size] for i in range(0, len(ordered), self.size)]
        if self.dataset.training:
            rng.shuffle(batches)
        if getattr(self.dataset, "hybrid", False):
            piece_prob = self.dataset.cfg.get("piecewise_tempo_probability", 0.)
            speed_prob = self.dataset.cfg.get("speed_probability", 0.) + piece_prob
            piece_carry = 0.
            speed_max_sec = self.dataset.cfg.get("speed_max_seconds", 350.0)
            speed_max_samples = (
                round(float(speed_max_sec) * SAMPLE_RATE)
                if speed_max_sec is not None and float(speed_max_sec) > 0.0
                else None
            )
            carry = 0.0
            for batch in batches:
                # Balance offline vs online MERT evenly inside each batch (e.g. 4/4 for B=8, p=0.5)
                # to bound peak online-MERT VRAM and eliminate high-variance tail batches.
                target = len(batch) * speed_prob + carry
                online_count = min(len(batch), max(0, int(math.floor(target + 1e-12))))
                carry = target - online_count
                if speed_max_samples is not None:
                    eligible = [
                        pos
                        for pos, (idx, _) in enumerate(batch)
                        if int(self.dataset.records[idx]["sample_length"]) <= speed_max_samples
                    ]
                else:
                    eligible = list(range(len(batch)))
                count_online = min(online_count, len(eligible))
                if piece_prob:
                    quota = count_online * piece_prob / speed_prob + piece_carry
                    count = min(count_online, math.floor(quota + 1e-12))
                    piece_carry = quota - count
                    flags = self.dataset.piecewise_eligible
                    known = [j for j in eligible if flags[batch[j][0]]]
                    unknown = [j for j in eligible if flags[batch[j][0]] is None]
                    pieces = set(rng.sample(known, min(len(known), count)))
                    pieces.update(rng.sample(unknown, min(len(unknown), count-len(pieces))))
                    remaining = [j for j in eligible if j not in pieces]
                    chosen = pieces | set(rng.sample(remaining, count_online-len(pieces)))
                    yield [(self.epoch, idx, "piecewise" if pos in pieces else "whole" if pos in chosen else False, occ)
                           for pos, (idx, occ) in enumerate(batch)]
                else:
                    chosen = set(rng.sample(eligible, count_online))
                    yield [(self.epoch, idx, pos in chosen, occ) if occ else (self.epoch, idx, pos in chosen)
                           for pos, (idx, occ) in enumerate(batch)]
            return
        for batch in batches:
            yield [(self.epoch, i, None, occ) if occ else (self.epoch, i) for i, occ in batch]


def collate_calibrator_batch(items):
    pad = nn.utils.rnn.pad_sequence
    targets = {k: pad([s["targets"][k] for s in items], batch_first=True) for k in items[0]["targets"]}
    out = CalibratorBatch(
        waveforms=pad([s["waveform"] for s in items], batch_first=True),
        targets=targets,
        sample_lengths=torch.tensor([len(s["waveform"]) for s in items]),
        song_ref_db=torch.tensor([s["song_ref_db"] for s in items]),
        mert=None,
    )
    if "pitch_semitones" in items[0]:
        out["pitch_semitones"] = None
        if any(s["pitch_semitones"] is not None for s in items):
            out["pitch_semitones"] = torch.tensor([s["pitch_semitones"] or 0. for s in items], dtype=torch.float32)
    if any(s.get("speed_segments") for s in items):
        out["speed_segments"] = [s.get("speed_segments", []) for s in items]
    if any("mert_needs_online" in s for s in items):
        needs = torch.tensor([s.get("mert_needs_online", False) for s in items])
        out["mert_needs_online"] = needs
        cached = [(i, s["mert"]) for i, s in enumerate(items) if "mert" in s]
        if cached:
            example = cached[0][1]
            frames = max(1, math.ceil(out["waveforms"].shape[1] / MERT_STRIDE_SAMPLES))
            out["mert"] = example.new_zeros(len(items), frames, *example.shape[1:])
            for i, value in cached:
                out["mert"][i, :len(value)] = value
    elif "mert" in items[0]:
        out["mert"] = pad([s["mert"] for s in items], batch_first=True)
    keys = [getattr(s, "song_key", "") for s in items]
    if any(keys):
        out.song_keys = tuple(keys)
    return out


class _Numpy1CheckpointPickle:
    # NumPy 2 records the RNG ndarray under numpy._core in complete checkpoints.
    # Resolve that name when warm-starting on machines still using NumPy 1.
    load = staticmethod(pickle.load)

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            return super().find_class(module.replace("numpy._core.", "numpy.core."), name)


def _read_calibrator_checkpoint(path, *, mmap=False):
    compat = _Numpy1CheckpointPickle if np.__version__.startswith("1.") else pickle
    return torch.load(path, map_location="cpu", weights_only=False, pickle_module=compat, mmap=mmap)


def load_calibrator_checkpoint(path, device="cpu"):
    saved = _read_calibrator_checkpoint(path)
    if saved["architecture"] != CALIBRATOR_ARCHITECTURE or saved["schema"] != CHECKPOINT_SCHEMA:
        raise ValueError("Expected a v18 beat calibrator checkpoint")
    legacy_clean = saved["model_config"].get("clean_lattice") and saved["model"]["grid_head.weight"].shape[0] == 9
    weights = saved["model"]
    clean_v1 = any(key in weights and weights[key].shape[1] == width for key, width in (
        ("correction.corrector.input.weight", 720), ("correction.bars.edge.0.weight", 128)))
    model = BMSBeatCalibrator(saved["model_config"], legacy_clean=legacy_clean, clean_v1=clean_v1)
    width = None if model.config.boundary_model or model.config.clean_lattice else saved["model"]["meter_head.weight"].shape[0]
    if width in (len(TRANSITIONAL_METERS) + 1, len(FOCUSED_V2_METERS) + 1):
        model.meter_head = ExtendedMeterHead(model.meter_head.in_features, width)
    model.load_state_dict(saved["model"])
    return model.to(device), saved


def initialize_stage2(model, path):
    stage1, _ = load_calibrator_checkpoint(path)
    if stage1.config.stage != 1:
        raise ValueError("Stage 2 initialization requires a Stage 1 checkpoint")
    if model.config.clean_lattice:
        if model.config.clean_v3:
            model.config.stage1_roformer_radius = stage1.config.stage1_roformer_radius
        for name, module in stage1.named_children():
            getattr(model, name).load_state_dict(module.state_dict())
        return
    if model.config.boundary_model:
        for name in ("stage_a", "downsample", "blocks_pre", "blocks_post", "up", "fusion"):
            getattr(model, name).load_state_dict(getattr(stage1, name).state_dict())
        with torch.no_grad():
            for field in ("weight", "bias"):
                grid = getattr(stage1.grid_head, field)
                getattr(model.kinematic_head, field).copy_(grid[[0, 1, 2, 3, 4, 5, 7]])
                getattr(model.tempo_regime_head, field)[0].copy_(grid[6])
                getattr(model.boundary_head, field)[1:].copy_(getattr(stage1.sum_head.components, field))
        return
    # Keep the legacy explicit rows in place and move OTHER to the last row.
    for name, module in stage1.named_children():
        dest = getattr(model, name)
        if name == "meter_head" and module.out_features == OTHER + 1 and dest.out_features == len(EXTENDED_METERS) + 1:
            with torch.no_grad():
                dest.weight[:OTHER].copy_(module.weight[:OTHER])
                dest.bias[:OTHER].copy_(module.bias[:OTHER])
                dest.weight[-1].copy_(module.weight[OTHER])
                dest.bias[-1].copy_(module.bias[OTHER])
                dest.weight[OTHER:-1].zero_()
                dest.bias[OTHER:-1].fill_(-4.0)
        else:
            dest.load_state_dict(module.state_dict())


def save_calibrator_checkpoint(path, model, **run):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        dict(
            schema=CHECKPOINT_SCHEMA,
            architecture=model.architecture,
            model_config=asdict(model.config),
            model=model.state_dict(),
            **run,
        ),
        tmp,
    )
    tmp.replace(path)
    return path


def train(
    config,
    manifest,
    val_manifest,
    output,
    *,
    stage1_checkpoint=None,
    resume=None,
    mert_manifest=None,
    val_mert_manifest=None,
    mert_model=None,
    device="cuda",
    allow_process_restart=False,
    allow_no_numba=False,
    cost_check=False, cost_warmup=3, cost_steps=8,
):
    from maidata.joint_training.checkpoint import rng_state, restore_rng, source_identity

    cfg = dict(config, mert_model=mert_model) if mert_model is not None else config
    if allow_no_numba:
        allow_numba_fallback(True)
    if cfg.get("boundary_model") or (cfg.get("clean_lattice") and cfg.get("stage") == 2):
        require_numba("calibrator training")
    if cfg["stage"] == 2 and not resume and stage1_checkpoint is None:
        raise ValueError("Stage 2 initialization requires a Stage 1 checkpoint")
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    generator = torch.Generator().manual_seed(cfg["seed"])
    if cfg.get("boundary_model") and mert_manifest is None and mert_model is None:
        mert_manifest = Path(manifest).parent / "mert_manifest.json"
    datasets = [
        BMSCalibratorDataset(p, cfg, training=training, mert_manifest=m)
        for p, training, m in [
            (manifest, True, mert_manifest),
            (val_manifest, False, val_mert_manifest or mert_manifest),
        ]
    ]
    loaders = [
        torch.utils.data.DataLoader(
            ds,
            batch_sampler=SongBatches(ds, cfg["batch_size"]),
            collate_fn=collate_calibrator_batch,
            generator=generator,
            num_workers=cfg["workers"],
            persistent_workers=cfg["workers"] > 0,
            prefetch_factor=(
                (1 if (cfg["stage"] == 2 and ds.mert_layers > 4) else 2)
                if cfg["workers"] > 0
                else None
            ),
            pin_memory=device.startswith("cuda"),
        )
        for ds in datasets
    ]
    binding = [ds.binding for ds in datasets]
    recipe = {
        k: v
        for k, v in cfg.items()
        if k not in (
            "stop_after_epochs",
            "max_microbatch_seconds",
            "stage_mert_first",
            "stage_mert_batches",
            "stage_mert_subbatch",
            "oversample_manifest",  # dataset binding records the manifest content hash
        )
    }
    source = source_identity()
    restored = None
    state = None
    if resume:
        restored, state = load_calibrator_checkpoint(resume)
        # Older checkpoints also stored the machine-local manifest path.
        saved_recipe = {k: v for k, v in state["recipe"].items() if k != "oversample_manifest"}
        if saved_recipe != recipe or state["binding"] != binding:
            raise ValueError("Complete-epoch resume recipe or cache binding differs")
        # 用户明确要求（2026-11-20）：禁止基于源码 commit / 源码哈希拒绝跨提交续训，
        # 不允许再加回任何 source commit / source hash 续训拦截校验；checkpoint 能否
        # 跨 commit 续训完全由用户自行判定。`source` 仅作为审计元数据写入，不作门禁。
        if state["scheduler"]["last_epoch"] != state["epoch"] * len(loaders[0]):
            raise ValueError("Complete-epoch resume scheduler step differs from epoch boundary")
    output = Path(output)
    output.mkdir(exist_ok=True, parents=True)
    recovery = CalibratorOOMRecovery(
        output,
        allow_process_restart=allow_process_restart,
        frame_rate=FRAME_RATE,
        sample_rate=SAMPLE_RATE,
    )
    mert_extractor = None
    mert_layers = datasets[0].mert_layers
    if cfg["stage"] == 2 and (mert_manifest is None or datasets[0].hybrid):
        mert_extractor = OnlineMERTv2Extractor(
            cfg.get("mert_model", DEFAULT_MERT_V2_MODEL),
            device=device,
            layers=cfg.get("mert_layers", DEFAULT_STAGE2_MERT_LAYERS if cfg.get("clean_lattice") else None),
        )
        mert_layers = mert_extractor.num_layers
    model = BMSBeatCalibrator(BMSCalibratorConfig(
        stage=cfg["stage"], clean_lattice=cfg.get("clean_lattice", False), boundary_model=cfg.get("boundary_model", False),
        clean_v3=cfg.get("clean_v3", False),
        stage1_roformer_radius=cfg.get("stage1_roformer_radius", 250),
        stage2_macro_radius=cfg.get("stage2_macro_radius", 200),
        stage2_mert_dropout=cfg.get("stage2_mert_dropout", 0.),
        mert_layers=mert_layers,
        extended_meters=cfg.get("extended_meters", False),
        unfreeze_grid_head=cfg.get("unfreeze_grid_head", False),
        grid_lr_scale=cfg.get("grid_lr_scale", 0.25),
    ))
    if resume and model.config.clean_v3:
        model.config.stage1_roformer_radius = restored.config.stage1_roformer_radius
    if cfg["stage"] == 2 and not resume:
        initialize_stage2(model, stage1_checkpoint)
    model.to(device)
    roformer_freeze = int(cfg.get("roformer_freeze_steps", 0)) if cfg["stage"] == 2 else 0
    roformer_warmup = (
        int(cfg.get("roformer_warmup_steps", cfg["warmup_steps"])) if cfg["stage"] == 2 else 0
    )
    split_roformer = cfg["stage"] == 2 and (
        roformer_freeze > 0 or roformer_warmup != cfg["warmup_steps"]
    )
    use_fused = device.startswith("cuda") and not recovery.state.get("optimizer_low_memory")
    optimizer = torch.optim.AdamW(
        model.parameter_groups(cfg["learning_rate"], split_roformer=split_roformer),
        fused=use_fused,
    )
    recovery.configure_optimizer(optimizer)
    steps = cfg["scheduler_epochs"] * len(loaders[0])
    warmup = cfg["warmup_steps"]

    def make_cosine_warmup(freeze_steps, warmup_steps):
        def scale_fn(i):
            if i < freeze_steps:
                return 0.0
            j = i - freeze_steps
            if j < warmup_steps:
                return (j + 1) / max(warmup_steps, 1)
            progress = min(1.0, (j - warmup_steps) / max(1, steps - freeze_steps - warmup_steps))
            return 0.5 * (1 + math.cos(math.pi * progress))

        return scale_fn

    head_scale = make_cosine_warmup(0, warmup)
    if split_roformer:
        rof_scale = make_cosine_warmup(roformer_freeze, roformer_warmup)
        # All groups before the heads share the delayed thaw and warmup.
        scales = ([rof_scale, rof_scale, head_scale, head_scale] if model.config.boundary_model else
                  [rof_scale] * (len(optimizer.param_groups) - 1) + [head_scale])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scales)
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, head_scale)
    start, best = 0, math.inf
    if resume:
        model.load_state_dict(restored.state_dict())
        del restored
        optimizer.load_state_dict(state["optimizer"])
        recovery.configure_optimizer(optimizer)
        scheduler.load_state_dict(state["scheduler"])
        restore_rng(state["rng"])
        generator.set_state(state["loader_rng"])
        start, best = state["epoch"], state["best"]
    stage_mert_first = bool(cfg.get("stage_mert_first", False)) and mert_extractor is not None
    stage_mert_batches = max(1, int(cfg.get("stage_mert_batches", 8)))
    stage_mert_subbatch = max(1, int(cfg.get("stage_mert_subbatch", 2)))
    if stage_mert_first:
        mert_extractor.offload_to_cpu()

    def iter_batches(loader):
        if not stage_mert_first:
            yield from loader
            return
        loader_iter = iter(loader)
        while True:
            chunk = []
            for _ in range(stage_mert_batches):
                try:
                    chunk.append(next(loader_iter))
                except StopIteration:
                    break
            if not chunk:
                break
            if any(batch_needs_online_mert(b) for b in chunk):
                mert_extractor.restore_to_device()
                try:
                    chunk = [
                        precompute_batch_mert(b, mert_extractor, subbatch=stage_mert_subbatch)
                        for b in chunk
                    ]
                finally:
                    mert_extractor.offload_to_cpu()
            yield from chunk

    step_mert_extractor = None if stage_mert_first else mert_extractor

    def update(batch):
        return recovery.run_train_step(
            model, batch, optimizer, scheduler, cfg=cfg, device=device,
            roformer_freeze=roformer_freeze, loss_fn=calibrator_loss,
            ref_song_frames=REFERENCE_SONG_FRAMES, mert_extractor=step_mert_extractor)

    if cost_check:
        loader = loaders[0]
        loader.batch_sampler.epoch = start + 1
        wait = 0.
        def stream():
            nonlocal wait
            while len(loader):
                iterator = iter(loader)
                for _ in range(len(loader)):
                    before = time.perf_counter()
                    batch = next(iterator)
                    wait += time.perf_counter() - before
                    yield batch
                loader.batch_sampler.epoch += 1
        cuda = str(device).startswith("cuda")
        elapsed, waits, sizes = [], [], []
        retries = 0
        # Chunked MERT staging (when requested) is included in end-to-end time.
        iterator = iter(iter_batches(stream()))
        try:
            for step in range(max(0, cost_warmup) + max(1, cost_steps)):
                if cuda:
                    torch.cuda.synchronize(device)
                    if step == max(0, cost_warmup):
                        torch.cuda.reset_peak_memory_stats(device)
                prev_wait, before = wait, time.perf_counter()
                batch = next(iterator)
                _, _, retried = update(batch)
                if cuda:
                    torch.cuda.synchronize(device)
                if step >= max(0, cost_warmup):
                    elapsed.append(time.perf_counter() - before)
                    waits.append(wait - prev_wait)
                    sizes.append(len(batch["sample_lengths"]))
                    retries += retried
        finally:
            iterator.close()
        report = dict(stage=cfg["stage"], steps=len(elapsed), batch_sizes=sizes, oom_retries=retries,
                      mean_step_ms=1000*np.mean(elapsed), mean_worker_wait_ms=1000*np.mean(waits),
                      peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20 if cuda else None,
                      peak_reserved_mib=torch.cuda.max_memory_reserved(device)/2**20 if cuda else None)
        print(json.dumps(report), flush=True)
        return report
    val_every = max(1, int(cfg.get("val_every_epochs", 5)))
    for epoch in range(start + 1, cfg["stop_after_epochs"] + 1):
        evaluate = epoch % val_every == 0 or epoch == cfg["stop_after_epochs"]
        save_numbered = evaluate if cfg.get("save_every_val") else (epoch % 5 == 0 or epoch == cfg["stop_after_epochs"])
        metrics = {}
        grid_counts = Counter() if cfg.get("clean_lattice") and cfg["stage"] == 2 else None
        for loader, training in zip(loaders, (True, False)):
            if not training and not evaluate:
                continue
            model.train(training)
            loader.batch_sampler.epoch = epoch if training else 0
            sums = torch.zeros(2, device=device, dtype=torch.float64)
            with torch.set_grad_enabled(training):
                for batch in iter_batches(loader):
                    if training:
                        frame_sum, valid_sum, _ = update(batch)
                    else:
                        frame_sum, valid_sum = recovery.run_eval_step(
                            model,
                            batch,
                            cfg=cfg,
                            device=device,
                            frame_sum_fn=calibrator_rhythm_frame_sum,
                            **({"grid_counts": grid_counts} if grid_counts is not None else {}),
                            mert_extractor=step_mert_extractor,
                        )
                    sums[0] += frame_sum
                    sums[1] += valid_sum
            if training and roformer_freeze > 0:
                model.set_stage2_roformer_trainable(True)
            metrics["train" if training else "validation"] = float(sums[0] / sums[1].clamp_min(1))
        if evaluate and grid_counts is not None:
            from .calibrator_boundary_decoder import clean_validation_metrics
            metrics.update(clean_validation_metrics(grid_counts))
        improved = evaluate and metrics["validation"] < best
        if improved:
            best = metrics["validation"]
        # Publish the resume cursor after all validation aliases. An interrupt
        # can then replay the epoch from the previous complete checkpoint.
        last = save_calibrator_checkpoint(
            output / ("epoch-complete.pending" if cfg.get("save_every_val") else "last.ckpt"),
            model,
            epoch=epoch,
            recipe=recipe,
            binding=binding,
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            rng=rng_state(),
            loader_rng=generator.get_state(),
            best=best,
            metrics=metrics,
            source=source,
        )
        for name in ([f"epoch-{epoch:03}.ckpt"] if save_numbered else []) + (
            ["best.ckpt"] if improved else []
        ):
            dst = output / name
            tmp = dst.with_suffix(".tmp")
            shutil.copyfile(last, tmp)
            tmp.replace(dst)
        if cfg.get("save_every_val"):
            last.replace(output / "last.ckpt")
        print(json.dumps(dict(epoch=epoch, **metrics)), flush=True)


def train_chain(*, stage1_config, stage2_config, stage1_output, stage2_output, stage1_select,
                manifest, val_manifest, device, mert_manifest=None, val_mert_manifest=None, mert_model=None,
                workers=None, batch_size=None, max_microbatch_seconds=None, grad_clip_norm=None,
                stage_mert_first=None, stage_mert_batches=None, stage_mert_subbatch=None,
                stage2_batch_size=None, stage2_learning_rate=None, stage2_warmup_steps=None,
                allow_no_numba=False, oversample_manifest=None):
    if allow_no_numba:
        allow_numba_fallback(True)
    require_numba("clean lattice train-chain")
    shared = ["--manifest", manifest, "--val-manifest", val_manifest, "--device", device]
    if allow_no_numba:
        shared.append("--allow-no-numba")
    for name, value in (("oversample-manifest", oversample_manifest), ("workers", workers),
                        ("max-microbatch-seconds", max_microbatch_seconds),
                        ("grad-clip-norm", grad_clip_norm)):
        if value is not None:
            shared.extend(["--" + name, str(value)])
    for stage, (config, output) in enumerate(((stage1_config, stage1_output), (stage2_config, stage2_output)), 1):
        last = Path(output) / "last.ckpt"
        stop = json.loads(Path(config).read_text(encoding="utf-8-sig"))["stop_after_epochs"]
        if last.is_file() and (last.parent / f"epoch-{stop:03}.ckpt").is_file():
            epoch = _read_calibrator_checkpoint(last, mmap=True)["epoch"]
            if epoch >= stop:
                print(f"Stage {stage} complete at epoch {epoch}, skipping.", flush=True)
                continue
        args = [sys.executable, "-m", "maidata.multitask.bms_calibrator", "train", "--config", config,
                "--output", output, *shared]
        size = stage2_batch_size if stage == 2 and stage2_batch_size is not None else batch_size
        if size is not None:
            args.extend(["--batch-size", str(size)])
        if last.is_file():
            args.extend(["--resume", str(last)])
        elif stage == 2:
            args.extend(["--stage1-checkpoint", str(Path(stage1_output) / (stage1_select + ".ckpt"))])
        if stage == 2:
            for name, value in (("mert-manifest", mert_manifest), ("val-mert-manifest", val_mert_manifest),
                                ("mert-model", mert_model)):
                if value is not None:
                    args.extend(["--" + name, value])
            if stage_mert_first is not None:
                args.append("--stage-mert-first" if stage_mert_first else "--no-stage-mert-first")
            for name, value in (("learning-rate", stage2_learning_rate),
                                ("warmup-steps", stage2_warmup_steps), ("stage-mert-batches", stage_mert_batches),
                                ("stage-mert-subbatch", stage_mert_subbatch)):
                if value is not None:
                    args.extend(["--" + name, str(value)])
        result = subprocess.run(args)
        if result.returncode:
            return result.returncode
    return 0


def main(argv=None):
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    split_cmd = commands.add_parser(
        "split-charts",
        help="filter unusable/duplicate BMS charts and create a stratified train/val supervision split",
    )
    split_cmd.add_argument("--charts-zip", action="append", default=[])
    split_cmd.add_argument("--bms-root", action="append", default=[])
    split_cmd.add_argument("--output", required=True)
    split_cmd.add_argument("--val-ratio", type=float, default=0.10)
    split_cmd.add_argument("--seed", type=int, default=20261114)
    split_cmd.add_argument("--workers", type=int, default=12)
    split_cmd.add_argument("--write-charts-zip")

    focused = commands.add_parser(
        "select-stage2-subset", help="retain complex supervision and stratified clean 4/4 anchors"
    )
    focused.add_argument("--split-manifest", required=True)
    focused.add_argument("--output", required=True)
    for key in ("train-manifest", "val-manifest", "source-pack-dir", "output-pack-dir"):
        focused.add_argument("--" + key)
    focused.add_argument("--seed", type=int, default=20261114)

    prep_split = commands.add_parser(
        "prepare-split-cache",
        help="assemble train.json, val.json, and songs/ for a stratified split from rendered FLAC archives",
    )
    prep_split.add_argument("--split-manifest", required=True)
    prep_split.add_argument("--flac-root")
    prep_split.add_argument("--charts-zip")
    prep_split.add_argument("--bms-root", action="append", default=[])
    prep_split.add_argument("--output", required=True)
    prep_split.add_argument("--workers", type=int, default=12)

    cache = commands.add_parser("prepare-mert-cache", help="extract frozen MERT layers once per BMS song")
    for key in ("manifest", "val-manifest", "output"):
        cache.add_argument("--" + key, required=True)
    cache.add_argument("--mert-model", default=DEFAULT_MERT_V2_MODEL)
    cache.add_argument("--layers", nargs="+", type=int, default=DEFAULT_STAGE2_MERT_LAYERS)
    cache.add_argument("--batch-size", type=int, default=8)
    cache.add_argument("--device", default="cuda")
    prep = commands.add_parser("prepare-cache", help="render one standard mono mix per BMS project")
    prep.add_argument("--bms-root", action="append", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--workers", type=int, default=12)
    fit = commands.add_parser("train", help="train or resume at a full epoch boundary")
    for key in ("config", "manifest", "val-manifest", "output"):
        fit.add_argument("--" + key, required=True)
    for key in ("mert-manifest", "val-mert-manifest", "mert-model", "stage1-checkpoint", "resume"):
        fit.add_argument("--" + key)
    fit.add_argument("--cost-check", action="store_true", help="Measure complete updates without saving checkpoints")
    fit.add_argument("--cost-warmup", type=int, default=3)
    fit.add_argument("--cost-steps", type=int, default=8)
    fit.add_argument("--oversample-manifest", help="Original supervision split JSON/ZIP for legacy cache manifests")
    fit.add_argument("--batch-size", type=int)
    fit.add_argument("--learning-rate", type=float)
    fit.add_argument("--warmup-steps", type=int)
    fit.add_argument("--extended-meters", action=argparse.BooleanOptionalAction)
    fit.add_argument("--unfreeze-grid-head", action=argparse.BooleanOptionalAction)
    fit.add_argument("--grid-lr-scale", type=float)
    fit.add_argument("--workers", type=int)
    fit.add_argument("--scheduler-epochs", type=int)
    fit.add_argument("--stop-after-epochs", type=int)
    fit.add_argument("--val-every-epochs", type=int)
    fit.add_argument("--max-song-seconds", type=float)
    fit.add_argument("--max-train-seconds", type=float)
    fit.add_argument("--max-microbatch-seconds", type=float)
    fit.add_argument("--grad-clip-norm", type=float)
    fit.add_argument(
        "--stage-mert-first",
        action=argparse.BooleanOptionalAction,
        help="Optional low-VRAM Stage 2 mode (default off): precompute online MERT in chunks and offload MERT to CPU before updates.",
    )
    fit.add_argument("--stage-mert-batches", type=int)
    fit.add_argument("--stage-mert-subbatch", type=int)
    fit.add_argument(
        "--allow-no-numba",
        action="store_true",
        help="Explicitly allow the slower NumPy fallback when compatible Numba is unavailable.",
    )
    fit.add_argument("--device", default="cuda")
    chain = commands.add_parser("train-chain", help="train or resume the two clean stages in sequence")
    chain.add_argument("--stage1-config", default="configs/v18-bms-calibrator-clean-stage1.json")
    chain.add_argument("--stage2-config", default="configs/v18-bms-calibrator-lattice-mertl17.json")
    for key in ("manifest", "val-manifest", "stage1-output", "stage2-output"):
        chain.add_argument("--" + key, required=True)
    for key in ("mert-manifest", "val-mert-manifest", "mert-model"):
        chain.add_argument("--" + key)
    chain.add_argument("--oversample-manifest")
    chain.add_argument("--stage1-select", choices=("best", "last"), default="best")
    chain.add_argument("--device", default="cuda")
    chain.add_argument("--workers", type=int)
    chain.add_argument("--batch-size", type=int)
    chain.add_argument("--stage2-batch-size", type=int)
    chain.add_argument("--stage2-learning-rate", type=float)
    chain.add_argument("--stage2-warmup-steps", type=int)
    chain.add_argument("--max-microbatch-seconds", type=float)
    chain.add_argument("--grad-clip-norm", type=float)
    chain.add_argument(
        "--stage-mert-first",
        action=argparse.BooleanOptionalAction,
        help="Optional low-VRAM Stage 2 mode (default off): precompute online MERT in chunks and offload MERT to CPU before updates.",
    )
    chain.add_argument("--stage-mert-batches", type=int)
    chain.add_argument("--stage-mert-subbatch", type=int)
    chain.add_argument(
        "--allow-no-numba",
        action="store_true",
        help="Explicitly allow the slower NumPy fallback when compatible Numba is unavailable.",
    )
    args = vars(parser.parse_args(argv))
    cmd = args.pop("command")
    if cmd == "train-chain":
        return train_chain(**args)
    if cmd == "prepare-mert-cache":
        print(prepare_mert_cache(**args))
        return 0
    if cmd == "select-stage2-subset":
        print(json.dumps(select_stage2_subset(**args), ensure_ascii=False))
        return 0
    if cmd == "split-charts":
        print(
            split_supervision_charts(
                charts_zips=args["charts_zip"],
                bms_roots=args["bms_root"],
                output=args["output"],
                val_ratio=args["val_ratio"],
                seed=args["seed"],
                workers=args["workers"],
                write_charts_zip=args["write_charts_zip"],
            )
        )
        return 0
    if cmd == "prepare-split-cache":
        written = prepare_split_cache(
            args["split_manifest"],
            args["output"],
            flac_root=args["flac_root"],
            charts_zip=args["charts_zip"],
            bms_roots=args["bms_root"],
            workers=args["workers"],
        )
        print(json.dumps({k: str(v) for k, v in written.items()}, ensure_ascii=False))
        return 0
    if cmd == "prepare-cache":
        print(prepare_cache(args["bms_root"], args["output"], args["workers"]))
        return 0

    if args["allow_no_numba"]:
        allow_numba_fallback(True)
    else:
        peek = json.loads(Path(args["config"]).read_text(encoding="utf-8-sig"))
        if peek.get("boundary_model") or (peek.get("clean_lattice") and peek.get("stage") == 2):
            require_numba("calibrator training")

    protected = args["device"].startswith("cuda") or os.environ.get("MAIDATA_CALIBRATOR_OOM_SUPERVISE") == "1"
    protected = protected and not args["cost_check"]
    if protected and os.environ.get("MAIDATA_CALIBRATOR_OOM_WORKER") != "1":
        return supervise_calibrator_train(raw_argv, args["output"], initial_resume=args["resume"])

    cfg = json.loads(Path(args.pop("config")).read_text(encoding="utf-8-sig"))
    for override_key in (
        "oversample_manifest",
        "extended_meters",
        "unfreeze_grid_head",
        "grid_lr_scale",
        "batch_size",
        "learning_rate",
        "warmup_steps",
        "workers",
        "scheduler_epochs",
        "stop_after_epochs",
        "val_every_epochs",
        "max_song_seconds",
        "max_train_seconds",
        "max_microbatch_seconds",
        "grad_clip_norm",
        "stage_mert_first",
        "stage_mert_batches",
        "stage_mert_subbatch",
    ):
        val = args.pop(override_key)
        if val is not None:
            cfg[override_key] = val
    if protected and os.environ.get("MAIDATA_CALIBRATOR_OOM_RESUME"):
        args["resume"] = os.environ["MAIDATA_CALIBRATOR_OOM_RESUME"]
    try:
        if args["cost_check"]:
            path = Path(args["output"])
            path.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(prefix="cost-", dir=path) as directory:
                train(cfg, **dict(args, output=directory))
        else:
            train(cfg, **args, allow_process_restart=protected)
    except CalibratorRestartRequired:
        return CALIBRATOR_OOM_EXIT_CODE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
