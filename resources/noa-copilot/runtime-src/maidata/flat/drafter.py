"""Full-song draft latents and their two consumers in Flat320 v19."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from maidata.draft.attention import Attention, safe_mask
from maidata.draft.generator import DraftGenerator
from . import ARCHITECTURE_V20
from maidata.draft.types import BeatMemory, DraftInputs
from maidata.nn.position import apply_rope, rope_tables
from .emotion import plan4_basis


def draft_anchor_valid(beat_valid):
    """Each audible beat pair contributes to both adjacent even anchors."""
    pairs = F.pad(beat_valid, (0, beat_valid.size(-1) % 2)).unflatten(-1, (-1, 2)).any(-1)
    return F.pad(pairs, (1, 0)) | F.pad(pairs, (0, 1))


class ChartDrafter(DraftGenerator):
    def __init__(self, cfg):
        super().__init__(
            audio_dim=cfg.dim,
            layers=cfg.draft_layers,
            dim=cfg.draft_dim,
            heads=cfg.draft_heads,
            experts=cfg.draft_experts,
            task_routing=cfg.architecture == ARCHITECTURE_V20,
            expert_ffn_dim=cfg.draft_expert_dim,
            dropout=cfg.draft_dropout,
            norm_eps=1e-5,
            difficulty_vocab_size=13 if cfg.architecture == ARCHITECTURE_V20 else 8,
            level_center=10.0,
            level_scale=5.0,
            span_scale=64.0,
            rope_base=10000.0,
            rope_position_scale=1.0,
            qk_normalization=True,
        )

        if cfg.architecture == ARCHITECTURE_V20:
            self.task_embed = nn.Embedding(2, cfg.draft_dim)
            nn.init.normal_(self.task_embed.weight, std=0.1)
            self.task_query = nn.Embedding(2, cfg.draft_dim)
            nn.init.normal_(self.task_query.weight, std=0.1)
            self.charter = nn.Embedding(len(cfg.designer_vocab) + 1, cfg.draft_dim, padding_idx=0)
            with torch.no_grad():
                nn.init.normal_(self.charter.weight, std=0.02)
                self.charter.weight[0].zero_()
        self.envelope_proj = nn.Sequential(
            nn.Linear(9, cfg.dim), nn.GELU(), nn.Dropout(0.1), nn.Linear(cfg.dim, cfg.dim, bias=False)
        )
        nn.init.normal_(self.envelope_proj[-1].weight, std=0.02)

    def dynamics(self, env, mel_length, seconds, valid, weights):
        last_frame = (mel_length - 1).clamp_min(0)
        pos = (125 * seconds - 0.5).clamp_min(0).minimum(last_frame)
        lo = pos.long()
        frames = torch.lerp(env[lo], env[(lo + 1).minimum(last_frame)], pos.frac()[:, None])
        env8 = (frames.view(-1, 3, 4) * weights).sum(1)
        if self.training:
            scale = env8.new_empty(1, 4).uniform_(0.95, 1.05)
            env8 = env8 * scale + 0.015 * torch.randn_like(env8)

        beat_valid = valid.view(-1, 24).any(1)
        rows = torch.arange(len(beat_valid), device=seconds.device)
        first = rows.masked_fill(~beat_valid, len(rows)).amin()
        last = rows.masked_fill(~beat_valid, -1).amax()
        centers = (3 * torch.arange(len(env8), device=seconds.device) + 1) / 24
        angle = ((centers - first) / (last - first).clamp_min(1)).clamp(0, 1) * torch.pi
        duration = seconds.view(-1, 24).diff(dim=1).median(1).values * 24
        log_duration = (duration.clamp_min(1e-5) / 0.5).log().repeat_interleave(8)
        clock = torch.stack((angle.sin(), angle.cos(), (2 * angle).sin(), (2 * angle).cos(), log_duration), -1)
        return torch.cat((env8, clock), -1).masked_fill(~valid.view(-1, 3).any(1)[:, None], 0)

    def forward(self, features, grids, grid_ids, values, bank, valid_indices=None, all_valid=False,
                designers=None, task_ids=None, mai_anchors=None):
        pooled, audible = [], []
        offset = 0
        for song, seconds, valid in grids:
            n = len(seconds)
            cells = features[offset : offset + n].reshape(-1, 3, features.size(-1))
            mask = valid.view(-1, 3)
            weights = mask[..., None] / mask.sum(1).clamp_min(1)[:, None, None]
            pooled_cells = (cells * weights.to(cells)).sum(1)
            dynamics = self.dynamics(bank.envelope[song], bank.mel_lengths[song], seconds, valid, weights)
            pooled_cells = pooled_cells + self.envelope_proj(dynamics.to(cells)).to(cells)
            pooled.append(pooled_cells.masked_fill(~mask.any(1)[:, None], 0))
            audible.append(mask.any(1))
            offset += n
        audio = pad_sequence(pooled, batch_first=True)
        valid = pad_sequence(audible, batch_first=True)
        positions = (3 * torch.arange(audio.size(1), device=audio.device) + 1) / 24
        memory = BeatMemory(audio, positions.expand_as(valid), valid)
        ids = torch.as_tensor(grid_ids).to(audio.device, non_blocking=True)
        beat_valid = valid.reshape(len(audio), -1, 8).any(-1).index_select(0, ids)
        anchors = draft_anchor_valid(beat_valid)
        beats = (2 * torch.arange(anchors.size(1), device=audio.device)).expand_as(
            anchors
        )
        span = torch.stack([item[2].sum() for item in grids]).float() / 24
        if hasattr(self, "task_query") and task_ids is None:
            task_ids = torch.zeros_like(ids)
            if all_valid or valid_indices is not None:
                mai_anchors = anchors.numel() if valid_indices is None else len(valid_indices)
        if task_ids is not None and designers is None:
            designers = torch.zeros_like(ids)
        inputs = DraftInputs(
            beats,
            anchors,
            ids,
            values[:, 1].long(),
            values[:, 0],
            span.index_select(0, ids),
            valid_indices,
            all_valid, designers, task_ids, mai_anchors,
        )
        return super().forward(memory, inputs), beat_valid


class BeatTEReadout(nn.Module):
    """One positional query per beat, over the even anchors within eight beats."""

    def __init__(self, latent_dim, multitask=False):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(128))
        if multitask:
            self.task_query = nn.Embedding(2, 128)
            nn.init.normal_(self.task_query.weight, std=0.1)
        self.pool = Attention(
            128,
            latent_dim,
            128,
            4,
            0.1,
            rotary=True,
            base=64.0,
            scale=0.35,
            qk_normalization=True,
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, z, beat_valid, task_ids=None):
        batch, count = beat_valid.shape
        beats = torch.arange(count, device=z.values.device)
        query = self.query.view(1, 1, -1)
        if hasattr(self, "task_query"):
            if task_ids is None:
                task_ids = beats.new_zeros(batch)
            query = query + self.task_query(task_ids)[:, None]
        q = self.pool.project_query(query)
        q = apply_rope(q, beats[None], self.pool.base, self.pool.scale).expand(batch, -1, -1, -1)
        k, v = self.pool.project_memory(z.values, z.beats, z.valid)
        # Tile queries to gather a bounded neighborhood once per sixteen beats.
        tile = 16
        pad = (-count) % tile
        query_beats = torch.arange(count + pad, device=beats.device).view(-1, tile)
        keys = query_beats[:, :1] // 2 + torch.arange(-4, 12, device=beats.device)
        indices = keys.clamp(0, z.valid.size(1) - 1)
        mask = (query_beats[:, :, None] - 2 * keys[:, None]).abs() <= 8
        mask = mask & ((keys >= 0) & (keys < z.valid.size(1)))[:, None]
        mask = mask[None] & z.valid[:, indices][:, :, None, :]

        mask = mask & F.pad(beat_valid, (0, pad)).view(batch, -1, tile, 1)
        safe, active = safe_mask(mask)

        def tiles(tensor):
            return tensor[:, :, indices].transpose(1, 2).flatten(0, 1)

        q = F.pad(q, (0, 0, 0, pad)).unflatten(2, (-1, tile)).transpose(1, 2)
        h = F.scaled_dot_product_attention(
            q.flatten(0, 1),
            tiles(k),
            tiles(v),
            attn_mask=safe.flatten(0, 1)[:, None],
            dropout_p=self.pool.dropout if self.training else 0.0,
        )
        h = h.transpose(1, 2).reshape(batch, count + pad, 128)[:, :count]
        active = active.reshape(batch, count + pad)[:, :count]
        h = self.pool.output(h.masked_fill(~active[..., None], 0)).masked_fill(
            ~active[..., None], 0
        )
        return F.softplus(self.mlp(h).squeeze(-1).float()).masked_fill(~active, 0)


class ConditionAdapter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.z_proj = nn.Linear(cfg.draft_dim, cfg.dim)
        self.z_norm = nn.LayerNorm(cfg.dim, elementwise_affine=False)
        self.plan_proj = nn.Sequential(
            nn.Linear(32, 128), nn.GELU(), nn.Linear(128, cfg.dim)
        )
        self.plan_norm = nn.LayerNorm(cfg.dim, elementwise_affine=False)
        self.source = nn.Embedding(2, cfg.dim)
        nn.init.normal_(self.source.weight, std=0.02)
        self.register_buffer(
            "base_freq", torch.arange(1, 5) * (torch.pi / 16), persistent=False
        )
        self.register_buffer(
            "delta_freq", torch.arange(1, 4) * (torch.pi / 5), persistent=False
        )

    def forward(self, z, beat_valid, plan=None, known=None):
        a = self.z_norm(self.z_proj(z.values) + self.source.weight[0])
        bar_valid = (
            F.pad(beat_valid, (0, -beat_valid.size(1) % 4))
            .unflatten(1, (-1, 4))
            .any(-1)
        )
        if plan is None:
            b = a.new_zeros(len(a), 1, a.size(-1))
            bar_valid = torch.zeros_like(bar_valid[:, :1])
        else:
            b = self.plan_norm(
                self.plan_proj(plan4_basis(plan, self.base_freq, self.delta_freq))
                + self.source.weight[1]
            )
            bar_valid = bar_valid & known
        centers = 4 * torch.arange(b.size(1), device=b.device) + 2
        return BeatMemory(
            torch.cat((a, b), 1),
            torch.cat((z.beats, centers.expand_as(bar_valid)), 1),
            torch.cat((z.valid, bar_valid), 1),
        )


def condition_buckets(anchors_list, width=16):
    """Group probe rows by chart and two-beat anchor before tiling."""
    owners, anchors, tiles, groups, reverse = [], [], [], [], []
    offset = 0
    for chart, chart_anchors in enumerate(anchors_list):
        probe = np.asarray(chart_anchors)
        order = np.argsort(probe, kind="stable")
        inverse = np.empty(len(order), dtype=np.int64)
        for group in np.split(order, np.flatnonzero(np.diff(probe[order])) + 1):
            if not len(group):
                continue
            owners.append(chart)
            anchors.append(probe[group[0]])
            base = len(tiles) * width
            inverse[group] = base + np.arange(len(group))
            padded = np.pad(group + offset, (0, -len(group) % width), mode="edge")
            groups.extend([len(anchors) - 1] * (len(padded) // width))
            tiles.extend(padded.reshape(-1, width))
        reverse.append(inverse)
        offset += len(probe)
    # The key gather is per query tile; all rows within a tile share its twelve keys.
    return dict(
        charts=torch.tensor(owners),
        anchors=torch.tensor(anchors),
        tile_group=torch.tensor(groups, dtype=torch.long),
        rows=torch.from_numpy(np.asarray(tiles, dtype=np.int64).reshape(-1, width)),
        inverse=torch.from_numpy(np.concatenate(reverse)),
    )


@dataclass
class ConditionMemory:
    layers: list
    valid: torch.Tensor
    z_count: int

    def window(self, charts, anchors):
        nz = self.z_count
        np4 = self.valid.size(1) - nz
        z = anchors[:, None] + torch.arange(-3, 4, device=anchors.device)
        p = anchors[:, None] // 2 + torch.arange(-2, 3, device=anchors.device)
        indices = torch.cat((z.clamp(0, nz - 1), nz + p.clamp(0, np4 - 1)), -1)
        inside = torch.cat(((z >= 0) & (z < nz), (p >= 0) & (p < np4)), -1)
        safe, active = safe_mask(
            (inside & self.valid[charts[:, None], indices])[:, None, None]
        )
        return indices, safe, active

    def tile_window(self, buckets):
        group = buckets["tile_group"]
        charts = buckets["charts"][group]
        indices, safe, active = self.window(buckets["charts"], buckets["anchors"])
        return charts, indices[group], safe[group], active[group]

    def slices(self, layer, charts, indices):
        # [chart, key, head, d] advanced indexing gathers only the requested keys.
        return tuple(
            t.transpose(1, 2)[charts[:, None], indices].transpose(1, 2)
            for t in self.layers[layer]
        )

    def bind(self, layer, buckets, window):
        charts, indices, safe, active = window
        k, v = self.slices(layer, charts, indices)

        def attend(q):
            queries = q[0].transpose(0, 1)[buckets["rows"]].transpose(1, 2)
            h = F.scaled_dot_product_attention(queries, k, v, attn_mask=safe)
            h = h.masked_fill(~active, 0).transpose(1, 2).flatten(0, 1)
            return h[buckets["inverse"]].transpose(0, 1)[None]

        return attend


class DecoderConditionAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm = nn.LayerNorm(cfg.dim)
        self.attention = Attention(
            cfg.dim,
            cfg.dim,
            cfg.dim,
            cfg.heads,
            0.0,
            rotary=True,
            base=64.0,
            scale=0.35,
            qk_normalization=True,
            output_bias=False,
        )
        self.gate = nn.Parameter(torch.tensor(0.001))

    def forward(self, x, tables, attend):
        q = self.attention.project_query(self.norm(x)[None])
        ctx = attend(apply_rope(q, None, tables=tables))
        return self.gate * self.attention.output(ctx.transpose(1, 2).reshape_as(x))


def condition_rope(beats, cfg):
    return rope_tables(beats, cfg.dim // cfg.heads, 64.0, 0.35)
