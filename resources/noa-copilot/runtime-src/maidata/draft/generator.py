"""Full-song, non-AR beat latents; independent of Encoder and Decoder classes."""

import torch
from torch import nn
from torch.nn import functional as F

from maidata.nn.experts import dense_expert_values
from maidata.nn.position import rope_tables, sinusoidal
from .attention import Attention, safe_mask
from .types import BeatMemory, DraftInputs, ProjectedAudio


class DenseMoE(nn.Module):
    def __init__(self, width, experts, hidden, dropout):
        super().__init__()
        self.gate = nn.Linear(width, experts)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(width, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, width),
            )
            for _ in range(experts)
        )

    def forward(self, x, indices=None):
        shape = x.shape
        if indices is not None:
            x = x.flatten(0, 1).index_select(0, indices)
        weights = self.gate(x).float().softmax(-1).to(x.dtype)
        mixed = (dense_expert_values(self.experts, x) * weights[..., None]).sum(-2)
        if indices is None:
            return mixed
        return (
            mixed.new_zeros(shape[0] * shape[1], shape[2])
            .index_copy(0, indices, mixed)
            .reshape(shape)
        )


class TaskDraftMMoE(nn.Module):
    def __init__(self, width, hidden, dropout):
        super().__init__()
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, width))
            for _ in range(3)
        )
        self.gates = nn.ModuleList(nn.Linear(width, 2) for _ in range(2))
        for gate in self.gates:
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)

    def forward(self, x, indices, mai_anchors):
        flat = x.flatten(0, 1)
        rows = flat if indices is None else flat.index_select(0, indices)
        if mai_anchors in (0, len(rows)):
            task = int(mai_anchors == 0)
            values = dense_expert_values((self.experts[0], self.experts[task + 1]), rows)
            weights = self.gates[task](rows).float().softmax(-1).to(values.dtype)
            result = torch.bmm(weights[:, None], values).squeeze(1)
        else:
            shared = self.experts[0](rows)
            parts = []
            for task, section in enumerate((slice(0, mai_anchors), slice(mai_anchors, None))):
                group = rows[section]
                weights = self.gates[task](group).float().softmax(-1).to(shared.dtype)
                parts.append(shared[section] * weights[:, :1] + self.experts[task + 1](group) * weights[:, 1:])
            result = torch.cat(parts)
        if indices is not None:
            result = result.new_zeros(flat.shape).index_copy(0, indices, result)
        return result.reshape_as(x)


class SingleFFN(nn.Module):
    def __init__(self, width, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, width)
        )

    def forward(self, x, indices=None):
        if indices is None:
            return self.net(x)
        flat = x.flatten(0, 1)
        out = self.net(flat.index_select(0, indices))
        return out.new_zeros(flat.shape).index_copy(0, indices, out).reshape(x.shape)


class DraftBlock(nn.Module):
    def __init__(
        self,
        audio_dim,
        width,
        heads,
        experts,
        hidden,
        dropout,
        eps,
        base,
        scale,
        qk_normalization=False,
        task_routing=False,
    ):
        super().__init__()
        self.norms = nn.ModuleList(nn.LayerNorm(width, eps=eps) for _ in range(3))
        self.self_attention = Attention(
            width,
            width,
            width,
            heads,
            dropout,
            rotary=True,
            base=base,
            scale=scale,
            qk_normalization=qk_normalization,
            fused_self=True,
        )
        self.cross_attention = Attention(
            width,
            audio_dim,
            width,
            heads,
            dropout,
            rotary=True,
            base=base,
            scale=scale,
            qk_normalization=qk_normalization,
        )
        self.modulation = nn.Linear(width, 2 * width)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)
        self.experts = (
            TaskDraftMMoE(width, hidden, dropout) if task_routing else
            SingleFFN(width, hidden, dropout) if experts == 1
            else DenseMoE(width, experts, hidden, dropout)
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x,
        beats,
        valid,
        condition,
        audio_kv,
        audio_valid,
        tables,
        valid_indices=None,
        self_mask=None,
        audio_mask=None,
        mai_anchors=None,
    ):
        scale, shift = self.modulation(condition).chunk(2, -1)
        h = self.norms[0](x) * (1 + scale[:, None]) + shift[:, None]
        attended = self.self_attention(h, beats, valid, None, valid, tables, self_mask)
        x = (x + self.drop(attended)).masked_fill(~valid[..., None], 0)
        attended = self.cross_attention(
            self.norms[1](x), beats, valid, audio_kv, audio_valid, tables, audio_mask
        )
        x = (x + self.drop(attended)).masked_fill(~valid[..., None], 0)
        normalized = self.norms[2](x)
        residual = (self.experts(normalized, valid_indices, mai_anchors)
                    if isinstance(self.experts, TaskDraftMMoE) else self.experts(normalized, valid_indices))
        return (x + self.drop(residual)).masked_fill(
            ~valid[..., None], 0
        )


class DraftGenerator(nn.Module):
    def __init__(
        self,
        *,
        audio_dim,
        layers,
        dim,
        heads,
        experts,
        expert_ffn_dim,
        dropout,
        norm_eps,
        difficulty_vocab_size,
        level_center,
        level_scale,
        span_scale,
        rope_base,
        rope_position_scale,
        qk_normalization=False,
        task_routing=False,
    ):
        super().__init__()
        self.audio_dim, self.dim = audio_dim, dim
        self.base, self.scale = rope_base, rope_position_scale
        self.head_dim = dim // heads
        self.level_center, self.level_scale, self.span_scale = (
            level_center,
            level_scale,
            span_scale,
        )
        self.query = nn.Parameter(torch.zeros(dim))
        self.difficulty = nn.Embedding(difficulty_vocab_size, dim, padding_idx=0)
        with torch.no_grad():
            nn.init.normal_(self.difficulty.weight, std=0.1)
            self.difficulty.weight[0].zero_()
        self.null_level = nn.Parameter(torch.zeros(dim))
        self.cond_norm = nn.LayerNorm(dim, eps=norm_eps)
        self.level = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        self.span = nn.Linear(1, dim)
        nn.init.normal_(self.span.weight, std=0.1)
        nn.init.zeros_(self.span.bias)
        self.blocks = nn.ModuleList(
            DraftBlock(
                audio_dim,
                dim,
                heads,
                experts,
                expert_ffn_dim,
                dropout,
                norm_eps,
                rope_base,
                rope_position_scale,
                qk_normalization=qk_normalization, task_routing=task_routing,
            )
            for _ in range(layers)
        )
        self.output_norm = nn.LayerNorm(dim, eps=norm_eps)

    def project_audio(self, audio) -> ProjectedAudio:
        """One projection per layer and unique audio, reusable within this graph only."""
        tables = rope_tables(audio.beats, self.head_dim, self.base, self.scale)
        return ProjectedAudio(
            tuple(
                block.cross_attention.project_memory(
                    audio.values, audio.beats, audio.valid, tables
                )
                for block in self.blocks
            ),
            audio.valid,
        )

    def forward(self, audio, inputs: DraftInputs) -> BeatMemory:
        return self.from_projected(self.project_audio(audio), inputs)

    def from_projected(self, audio: ProjectedAudio, inputs: DraftInputs) -> BeatMemory:
        beats, valid, mapping = (
            inputs.query_beats,
            inputs.draft_valid,
            inputs.audio_mapping,
        )
        difficulty, level, span = inputs.difficulty_id, inputs.level, inputs.span_beats
        av = audio.valid.index_select(0, mapping)
        positions = beats.masked_fill(~valid, 0)
        level_vec = self.level(
            (level[:, None].to(self.query) - self.level_center) / self.level_scale
        )
        level_vec = torch.where((level > 0)[:, None], level_vec, self.null_level.to(level_vec))
        conditioning = self.difficulty(difficulty)
        if inputs.task_id is not None:
            conditioning = conditioning + self.task_embed(inputs.task_id) + self.charter(inputs.charter_id)
        condition = self.cond_norm(
            conditioning
            + level_vec
            + self.span(torch.log1p(span[:, None].to(self.query) / self.span_scale))
        )
        tables = rope_tables(positions, self.head_dim, self.base, self.scale)
        x = self.query + sinusoidal(positions, self.dim, self.base, self.scale).to(
            condition
        )
        if inputs.task_id is not None:
            x = x + self.task_query(inputs.task_id)[:, None]
        x = x.masked_fill(~valid[..., None], 0)
        valid_indices = inputs.valid_indices
        if valid_indices is None and not inputs.all_valid and not valid.all():
            valid_indices = valid.flatten().nonzero().flatten()
        mai_anchors = inputs.mai_anchors
        if inputs.task_id is not None and mai_anchors is None:
            # Standalone callers may supply charts in any order. Pack active anchors
            # once, then let each MoE scatter its residual back to the original rows.
            tasks = inputs.task_id[:, None].expand_as(valid).flatten()
            if valid_indices is None:
                valid_indices = torch.arange(tasks.numel(), device=tasks.device)
            tasks = tasks[valid_indices]
            mai_anchors = int((tasks == 0).sum())
            if 0 < mai_anchors < len(tasks):
                valid_indices = valid_indices[tasks.argsort(stable=True)]
        self_mask = safe_mask(valid[:, None, None, :])
        audio_mask = safe_mask(av[:, None, None, :])
        for block, kv in zip(self.blocks, audio.key_values):
            x = block(
                x,
                positions,
                valid,
                condition,
                tuple(value.index_select(0, mapping) for value in kv),
                av,
                tables,
                valid_indices,
                self_mask,
                audio_mask,
                mai_anchors,
            )
        # Explicit FP32 parameters also support an otherwise bf16 model.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = F.layer_norm(
                x.float(),
                self.output_norm.normalized_shape,
                self.output_norm.weight.float(),
                self.output_norm.bias.float(),
                self.output_norm.eps,
            )
        return BeatMemory(x.masked_fill(~valid[..., None], 0), beats, valid)
