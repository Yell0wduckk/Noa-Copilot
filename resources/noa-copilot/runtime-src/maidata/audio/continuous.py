"""Continuous local dual-stream audio encoder (`CompactSpec-TOS` + `E0` + `Fold3` `P0` / `G_local`)."""

import math
from dataclasses import replace

import torch
from torch import nn

from maidata.nn.position import rope_tables

from .blocks import SharedConformerBlock
from .compactspec import CompactSpecFrontend
from .continuous_attention import with_joint_mask
from .continuous_layout import prepare_continuous_layout, to_device
from .freqconv import AudioContext
from .geometry import mask_values
from .p0_conformer import P0ConformerBlock, P0OutputNorm
from .pyramid import P0Bank
from .types import EncoderOutput, ScaleMemory, SongEncoding


class ContinuousAudioEncoder(nn.Module):
    """CompactSpec E0 micro-Conformer + Fold3 local dual-stream (P0 & G_local) encoder."""

    def __init__(
        self,
        *,
        samples_per_beat=24,
        d_model=320,
        heads=8,
        ffn_dim=1280,
        stage_depths=(1, 1, 1),
        stage_attention_radii=None,
        stage_attention_radius=1,
        stage_kernel_sizes=(5, 5, 3),
        fpn_rates=(8,),
        target_beats=16,
        audio_before_beats=8,
        audio_after_beats=8,
        dropout=0.1,
        rope_base=2048.0,
        rope_position_scale=3.14,
        fpn_p0_normalization=True,
        slot_attention_radius=None,
        p0_conformer_layers=2,
        global_audio=True,
        beat_p0_modulation=False,
        beat_use_convolution=True,
        encoder_local_dual_stream=True,
        encoder_e0_fold3=True,
        stage_transformer=False,
        **frontend_config,
    ):
        super().__init__()
        if stage_attention_radius is not None:
            stage_attention_radii = (stage_attention_radius,) * 3
        if d_model % heads or (d_model // heads) % 2:
            raise ValueError("d_model / heads must be an even integer for full-head RoPE")
        stage_kernel_sizes = tuple(stage_kernel_sizes)
        if len(stage_depths) != 3 or len(stage_attention_radii) != 3 or len(stage_kernel_sizes) != 3:
            raise ValueError("the encoder requires three stages, radii, and kernel sizes")
        if tuple(fpn_rates) != (8,):
            raise ValueError("E0/Fold3 encoder requires fpn_rates == (8,)")
        if fpn_p0_normalization is not True:
            raise ValueError("E0/Fold3 encoder requires fpn_p0_normalization == True")
        if type(p0_conformer_layers) is not int or p0_conformer_layers <= 0:
            raise ValueError("p0_conformer_layers must be a positive integer")

        context = audio_before_beats + target_beats + audio_after_beats
        self.encoder_e0_fold3 = True
        self.local_dual_stream = True
        self.global_audio = True
        self.compactspec = True
        self.logspec = True
        self.stage_kernel_sizes = stage_kernel_sizes
        self.stage_attention_radii = tuple(stage_attention_radii)
        self.samples_per_beat = samples_per_beat
        self.context_beats = context
        self.audio_before_beats = audio_before_beats

        radius = context / 2 if slot_attention_radius is None else slot_attention_radius
        if isinstance(radius, bool) or not isinstance(radius, (int, float)) or not math.isfinite(radius) or radius < 0:
            raise ValueError("slot attention radius must be finite and nonnegative")
        self.slot_attention_radius = float(radius)

        self.spectral = CompactSpecFrontend(d_model, frontend_config)
        self.feature_fusion = AudioContext(d_model)
        self.stages = nn.ModuleList([
            nn.ModuleList([
                SharedConformerBlock(
                    d_model, heads, ffn_dim, kernel_size, attn_radius,
                    dropout, rope_base, rope_position_scale,
                ) for _ in range(depth)
            ]) for depth, attn_radius, kernel_size in zip(stage_depths, stage_attention_radii, stage_kernel_sizes)
        ])
        self.transitions = nn.ModuleList()
        self.pyramid = P0Bank(d_model, fpn_rates, context, audio_before_beats,
                              p0_normalization=fpn_p0_normalization)
        self.p0_conformer = nn.ModuleList()
        self.beat_conformer = nn.ModuleList(
            P0ConformerBlock(
                d_model, heads, ffn_dim, 3, dropout,
                rope_base, rope_position_scale,
            ) for _ in range(p0_conformer_layers)
        )
        self.beat_norm = P0OutputNorm(d_model)

    def encode_song(self, inputs):
        """Encode the fixed song domain once; all retained values remain differentiable."""
        feature_shape = (249,)
        if (inputs.spec is None or inputs.spec.shape != (*inputs.beats.shape, *feature_shape)
                or inputs.silence_spec.shape != feature_shape
                or inputs.context.shape != (*inputs.beats.shape, 2)
                or inputs.sample_valid.shape != inputs.beats.shape
                or inputs.valid.shape != inputs.beats.shape):
            raise ValueError("continuous CompactSpec inputs must have aligned spectrum/context/coordinate axes")
        layout = inputs.layout
        if layout is None:
            layout = prepare_continuous_layout(
                replace(inputs, song=None),
                radii=self.stage_attention_radii,
                rates=self.pyramid.fpn_rates,
                slot_radius=self.slot_attention_radius,
                e0_only=True,
                samples_per_beat=self.samples_per_beat,
                local_dual_stream=True,
            )
        layout = to_device(layout, inputs.spec.device)
        values = self.spectral(inputs.spec, inputs.sample_valid, inputs.valid, inputs.silence_spec)
        values = self.feature_fusion(values, inputs.context, inputs.beats, inputs.valid)
        local = ScaleMemory(values, inputs.beats.float(), inputs.valid)
        stages = []
        local_tables = None
        e0_joint_by_radius = {}
        for index, stage in enumerate(self.stages):
            first = stage[0]
            if local_tables is None:
                local_tables = rope_tables(
                    local.beats.masked_fill(~local.valid, 0),
                    first.q.out_features // first.heads,
                    first.rope_base,
                    first.rope_position_scale,
                )
            attention = layout['attention'][index]
            radius = self.stage_attention_radii[index]
            if radius not in e0_joint_by_radius:
                e0_joint_by_radius[radius] = dict(local=with_joint_mask(
                    attention['local'], local.beats, (local.beats,)))
            joint = e0_joint_by_radius[radius]
            for block in stage:
                local = block(local, local_tables=local_tables, joint_layout=joint)
            stages.append(local)

        e0 = ScaleMemory(mask_values(self.pyramid.p0_norm(local.values), local.valid), local.beats, local.valid)
        padding = (-local.values.shape[1]) % 3
        folded = torch.nn.functional.pad(e0.values, (0, 0, 0, padding)).unflatten(1, (-1, 3)).flatten(-2)
        p0 = mask_values(self.pyramid.fold3(folded), layout['grid_valid'][0])
        result = [ScaleMemory(p0, layout['grids'][0], layout['grid_valid'][0])]
        encoding = SongEncoding(
            EncoderOutput(*result),
            tuple(s.beats for s in stages),
            tuple(s.valid for s in stages),
            layout,
            self.pyramid.p0_norm,
        )
        source = encoding.P0
        beat_memory = result[0]
        beat_attention = dict(local=with_joint_mask(
            layout['local_beat_attention'], source.beats, (source.beats,)))
        first = self.beat_conformer[0].attention
        tables = rope_tables(
            beat_memory.beats.masked_fill(~beat_memory.valid, 0),
            first.head_dim, first.rope_base, first.rope_position_scale,
        )
        for block in self.beat_conformer:
            beat_memory = block(beat_memory, tables=tables, attention=beat_attention)
        beat_memory = replace(
            beat_memory,
            values=mask_values(self.beat_norm(beat_memory.values), beat_memory.valid),
        )
        encoding.beat_memory = beat_memory
        return encoding

    def forward(self, song_inputs):
        return self.encode_song(song_inputs)
