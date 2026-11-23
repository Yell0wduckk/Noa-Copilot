"""Single-stream RuleMMoE with independent main and minimal-history token states."""

from dataclasses import dataclass, field
import math

import torch
from torch import nn
from torch.nn import functional as F

from maidata.chart.routes import catalog
from .heads import HeteroInputHeads, RMSNorm
from .audio import BeatGridEncoder, DualAudioFusion, STREAMS, audio_rotary
from maidata.decoder.projections import packed_projection_scope
from maidata.designer import DESIGNER_VOCAB
from maidata.multitask.attention import rotary
from maidata.spine.features import CLOCK_DIM
from maidata.spine.model import chart_condition
from maidata.streaming.network import RuleMMoE

from .codec import FAMILY_IDS, NEW_PADS, SIZE, values
from .layouts import AttentionLayout, AuxRequest

SELF_GUIDE_NAMES = ("occ", "near", "slide")


@dataclass
class FlatConfig:
    stage: int = 0
    dim: int = 256
    layers: int = 6
    heads: int = 8
    experts: int = 6
    expert_dim: int = 512
    conv_heads: int = 4
    mert_heads: int = 4
    conv_kv_heads: int = 4
    mert_kv_heads: int = 2
    head_dim: int = 64
    beat_encoder_layers: int = 2
    spec_augment_bands: int = 12
    mel_gain_jitter: float = 0.08
    mert_dropout: float = 0.10
    no_history_enabled: bool = True
    conv_past_beats: float = 1.5
    conv_future_beats: float = 1.5
    mert_past_beats: float = 8.0
    mert_future_beats: float = 8.0
    self_rotary_dims: int = 24
    self_rope_scale: float = 8.0
    self_rope_base: float = 64.0
    conv_rope_scale: float = 2.0
    conv_rope_base: float = 64.0
    mert_rope_scale: float = 0.35
    mert_rope_base: float = 64.0
    dropout: float = 0.0
    history_query_block: int = 128
    history_backend: str = "auto"
    attn_guide_weight: float = 0.05
    slide_lookahead_weight: float = 0.03
    attn_guide_layers: tuple[float, ...] = (0.0, 0.0, 0.10, 0.20, 0.30, 0.40)
    condition_dropout: float = 0.05
    level_dropout: float = 0.10
    difficulty_dropout: float = 0.10
    charter_dropout: float = 0.25
    designer_vocab: list[str] = field(default_factory=lambda: list(DESIGNER_VOCAB))

    def __post_init__(self):
        self.attn_guide_layers = tuple(self.attn_guide_layers)


def partial_rotary(x, positions, *, scale, base, rotary_dims=None):
    prefix = x if rotary_dims is None else x[..., :rotary_dims]
    rotated = rotary(prefix, positions * scale, base)
    if prefix.shape[-1] == x.shape[-1]:
        return rotated
    return torch.cat((rotated, x[..., rotary_dims:]), -1)


class Grid32TimingHead(nn.Module):
    """Factor P(next) into a binary emit decision and a conditional GRID32 slot."""

    _version = 2  # Version 1 checkpoints stored the absolute midpoint projection.

    def __init__(self, dim):
        super().__init__()
        self.sub64_proj = nn.Linear(2 * dim, dim, bias=False)
        nn.init.zeros_(self.sub64_proj.weight)
        self.slot_acoustic = nn.Linear(dim, 1, bias=False)
        self.slot_key = nn.Linear(dim, dim, bias=False)
        self.slot_query = nn.Linear(dim, dim, bias=False)
        self.slot_prior = nn.Linear(dim, 32)
        self.emit_gate = nn.Sequential(
            nn.Linear(3 * dim + 3, dim), nn.GELU(), nn.Linear(dim, 1)
        )

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, *args):
        key = prefix + "sub64_proj.weight"
        if local_metadata.get("version") == 1 and key in state_dict:
            weight = state_dict[key].clone()
            for half in weight.chunk(2, dim=1):
                half.diagonal().sub_(0.5)
            state_dict[key] = weight
        super()._load_from_state_dict(state_dict, prefix, local_metadata, *args)

    def evidence(self, feat):
        beats = feat.reshape(-1, 24, feat.size(-1))
        a, b, c = beats.unflatten(1, (8, 3)).unbind(2)
        mid = 0.5 * (b + c) + self.sub64_proj(torch.cat((b, c), -1))
        slots = torch.stack((a, b, mid, c), 2).flatten(1, 2)
        return (
            slots,
            self.slot_acoustic(slots).squeeze(-1),
            self.slot_key(slots),
            beats.mean(1),
        )

    def forward(self, logits, hidden, rows, beat, legal, evidence):
        if not len(rows):
            return logits.float()
        slots, acoustic, keys, mean = evidence
        h = hidden[rows]
        q = self.slot_query(h)
        # bmm reduces over channels without constructing a row x slot x dim product.
        match = torch.bmm(keys[beat], q.unsqueeze(2)).squeeze(2) * q.size(-1) ** -0.5
        u = (
            acoustic[beat].float() + match.float() + self.slot_prior(h).float()
        ).masked_fill(~legal, -torch.inf)
        lse = u.logsumexp(-1, keepdim=True)
        count = legal.sum(-1, keepdim=True).float()
        rem = torch.bmm(
            (legal / count).to(slots.dtype).unsqueeze(1), slots[beat]
        ).squeeze(1)
        stats = torch.cat(
            (u.amax(-1, keepdim=True), lse - count.log(), count.log()), -1
        )
        eta = self.emit_gate(
            torch.cat((h, rem, mean[beat], stats.to(h.dtype)), -1)
        ).float()
        result = logits.float()
        beat_start = FAMILY_IDS["BEAT"][0]
        moment_start = FAMILY_IDS["MOMENT"][0]
        result[rows, beat_start : beat_start + len(FAMILY_IDS["BEAT"])] = F.logsigmoid(
            -eta
        )
        result[rows, moment_start : moment_start + 32] = F.logsigmoid(eta) + (u - lse)
        return result


class SlideLookaheadHead(nn.Module):
    """Two future onset bins, with a detached prefix prior."""

    def __init__(self, dim, head_dim):
        super().__init__()
        self.mert_proj = nn.Linear(head_dim, 2, bias=False)
        self.prior_proj = nn.Linear(dim, 2)

    def forward(self, mert_ctx, normed_x):
        return self.mert_proj(mert_ctx) + self.prior_proj(normed_x.detach())


class Block(nn.Module):
    def __init__(self, cfg, with_lookahead=False):
        super().__init__()
        d = cfg.dim
        self.heads = cfg.heads
        self.cfg = cfg
        self.self_norm = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.cross_norm = nn.LayerNorm(d)
        self.audio_query = nn.Linear(d, cfg.heads * cfg.head_dim, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, eps=1e-6)
        self.audio_out = nn.Linear(cfg.heads * cfg.head_dim, d, bias=False)
        self.slide_lookahead = (
            SlideLookaheadHead(d, cfg.head_dim) if with_lookahead else None
        )
        self.ffnorm = nn.LayerNorm(d)
        self.experts = RuleMMoE(d, cfg.expert_dim, cfg.experts, cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)

    def split(self, x):
        return x.reshape(1, len(x), self.heads, -1).transpose(1, 2)

    def join(self, x):
        return x.transpose(1, 2).reshape(x.shape[2], -1)

    def rotated(self, x, ticks):
        return partial_rotary(
            self.split(x),
            ticks / 96,
            scale=self.cfg.self_rope_scale,
            base=self.cfg.self_rope_base,
            rotary_dims=self.cfg.self_rotary_dims,
        )

    def projections(self, x, ticks):
        q, k, v = self.qkv(self.self_norm(x)).chunk(3, -1)
        return (
            self.rotated(q, ticks),
            self.rotated(k, ticks),
            self.split(v),
        )

    def read_query(self, x, ticks):
        d = x.shape[-1]
        q = F.linear(self.self_norm(x), self.qkv.weight[:d], self.qkv.bias[:d])
        return self.rotated(q, ticks)

    def finish(
        self, x, self_context, acoustic, roles, audio_tick, zero=None, lookahead=None
    ):
        residual = self.out(self.join(self_context))
        if zero is not None:
            residual = residual.masked_fill(zero[:, None], 0)
        x = x + self.drop(residual)
        normed_x = self.cross_norm(x)
        aq = self.q_norm(self.split(self.audio_query(normed_x)))
        queries = aq.split((self.cfg.conv_heads, self.cfg.mert_heads), dim=1)
        ctx = torch.cat(
            [
                attend(audio_rotary(self.cfg, stream, q, audio_tick / 96))
                for stream, q, attend in zip(STREAMS, queries, acoustic)
            ],
            dim=1,
        )
        if lookahead is not None:
            target, alpha, losses = lookahead
            rows = target["rows"]  # Compact main rows precede both auxiliary streams.
            head = self.cfg.conv_heads + self.cfg.mert_heads // 2
            with torch.autocast(x.device.type, enabled=False):
                logits = self.slide_lookahead(
                    ctx[0, head, rows].float(), normed_x[rows].float()
                )
                bce = F.binary_cross_entropy_with_logits(logits, target["targets"])
            losses["mert_slide_2b"] = losses["mert_slide_2b"] + alpha * bce
        x = x + self.drop(self.audio_out(self.join(ctx)))
        return x + self.drop(self.experts(self.ffnorm(x), roles))

    def reset_history(self):
        nn.init.ones_(self.self_norm.weight)
        nn.init.zeros_(self.self_norm.bias)
        for w in self.qkv.weight.chunk(3, 0):
            nn.init.orthogonal_(w)
        nn.init.zeros_(self.qkv.bias)
        nn.init.orthogonal_(self.out.weight, gain=0.02)
        nn.init.zeros_(self.out.bias)


class ActiveStateReadout(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("offsets", torch.arange(32) * 8, persistent=False)
        self.embed = nn.Embedding(32 * 8, 16)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.proj = nn.Sequential(
            nn.LayerNorm(32 * 16),
            nn.Linear(32 * 16, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, states):
        x = self.embed(states.long().clamp(0, 7) + self.offsets).flatten(-2)
        return self.proj(x).masked_fill(~states.bool().any(-1, keepdim=True), 0)


def attention_guide(q, k, lse, target, head):
    """The denominator comes from the kernel. Gather only supervised compact keys."""
    rows, keys = target["rows"], target["keys"]
    if not rows.numel():
        return lse.new_zeros(())
    query = q[0, head, rows].float()
    key = k[0, head // (q.shape[1] // k.shape[1]), keys].float()
    normalizer = lse[0, head, rows]
    scale = q.shape[-1] ** -0.5
    if "weights" in target:
        scores = (query * key).sum(-1) * scale
        return (
            target["weights"] * (normalizer - scores + target["log_weights"])
        ).sum() / target["support"]
    if "cluster" in target:
        # Ragged palm / slide clusters can contain arbitrarily many slots.
        cluster = target["cluster"]
        scores = (query[cluster] * key).sum(-1) * scale
        peaks = torch.full_like(normalizer, -torch.inf).scatter_reduce_(
            0, cluster, scores.detach(), reduce="amax"
        )
        sums = torch.zeros_like(peaks).scatter_add_(
            0, cluster, (scores - peaks[cluster]).exp()
        )
        union = peaks + sums.log()
        return (
            (normalizer - union - target["log_count"]) * target["scale"]
        ).sum() / target["support"]
    scores = (query[:, None] * key).sum(-1) * scale
    return (normalizer - (scores + target["log_kernel"]).logsumexp(-1)).mean()


def main_nll(z, ids, union):
    pos = z.gather(1, ids[:, None]).squeeze(1)
    if union is not None and union["rows"].numel():
        rows = union["rows"]
        positives = torch.zeros(
            (len(rows), z.shape[1]), dtype=torch.bool, device=z.device
        )
        positives.scatter_(1, ids[rows, None], True)
        positives[union["inverse"], union["cols"]] = True
        pos = pos.index_copy(
            0, rows, z[rows].masked_fill(~positives, -torch.inf).logsumexp(-1)
        )
    return z.logsumexp(-1) - pos


class FlatModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        cfg = self.config = config or FlatConfig()
        d = cfg.dim
        self.fusion = DualAudioFusion(cfg)
        self.beat_encoder = BeatGridEncoder(cfg)
        self.timing_head = Grid32TimingHead(d)
        self.input_heads = HeteroInputHeads(d)
        self.phase = nn.Sequential(nn.Linear(CLOCK_DIM, d), nn.GELU(), nn.Linear(d, d))
        self.level_mlp = nn.Sequential(nn.Linear(16, d), nn.GELU(), nn.Linear(d, d))
        self.null_level = nn.Parameter(torch.zeros(d))
        self.register_buffer(
            "level_freq", torch.arange(1, 9) * math.pi / 15, persistent=False
        )
        self.difficulty = nn.Embedding(8, d, padding_idx=0)
        self.charter = nn.Embedding(len(cfg.designer_vocab) + 1, d, padding_idx=0)
        self.position = nn.Embedding(34, d, padding_idx=33)
        self.active_readout = ActiveStateReadout(d)
        with torch.no_grad():
            for embedding in (self.position, self.charter, self.difficulty):
                nn.init.normal_(embedding.weight, std=0.02)
                embedding.weight[embedding.padding_idx].zero_()
        self.blocks = nn.ModuleList(
            Block(
                cfg,
                layer < len(cfg.attn_guide_layers) and cfg.attn_guide_layers[layer] > 0,
            )
            for layer in range(cfg.layers)
        )
        self.norm = nn.LayerNorm(d)
        self.output = nn.Linear(d, SIZE)
        groups = torch.arange(SIZE)
        groups[list(NEW_PADS)] = NEW_PADS[0]
        first_route = {}
        for route, row in zip(values("ROUTE"), catalog()):
            groups[route] = first_route.setdefault(row["motion"], route)
        self.register_buffer("coarse", groups, persistent=False)

    conditioning = chart_condition

    def protected_modules(self):
        yield self.fusion
        yield self.beat_encoder
        yield from (
            self.timing_head.sub64_proj,
            self.timing_head.slot_key,
            self.timing_head.slot_acoustic,
        )
        for block in self.blocks:
            if block.slide_lookahead is not None:
                yield block.slide_lookahead
            yield from (
                block.cross_norm,
                block.audio_query,
                block.q_norm,
                block.audio_out,
            )

    def audio_keys(self, bank, song=0, *, chart):
        grids = (
            chart["grids"]
            if "grids" in chart
            else [(song, chart["grid_seconds"], chart["in_audio"])]
        )
        encoded = [
            self.beat_encoder(bank, owner, seconds, valid)
            for owner, seconds, valid in grids
        ]
        if len(encoded) == 1:
            return encoded[0]
        feat = torch.cat([x for x, _ in encoded])
        pairs = [
            tuple(torch.cat(parts, 2) for parts in zip(*stream))
            for stream in zip(*(kv for _, kv in encoded))
        ]
        return feat, pairs

    def base(self, heads, clock, cond):
        return self.input_heads(heads, len(clock)) + self.phase(clock) + cond

    def minimal_timing_state(self, active, positions):
        return self.active_readout(active) + self.position(positions)

    def decode(
        self,
        c,
        audio,
        cond,
        *,
        auxiliary=False,
        save_kv=None,
        guides=None,
        cross_scale=1.0,
    ):
        cfg = self.config
        coords, timing = c["coords"], c["timing"]
        ticks = coords[:, 0]
        grid_feat, audio = audio
        roles = (~timing).long()
        x = self.base(c["heads"], c["clock"], cond)
        masks = c["layouts"]
        selected = masks["special"]
        token_rows, read_rows = c.get("token_rows"), c.get("read_rows")

        # Editor prefill captures K/V once; use native SDPA on the first request.
        supervised = self.training and save_kv is None and guides is not None
        guide_enabled = supervised and cfg.attn_guide_weight > 0
        lookahead_enabled = (
            supervised
            and cfg.slide_lookahead_weight > 0
            and cross_scale > 0
            and c["slide_lookahead"]["support"] > 0
        )
        self_guides = SELF_GUIDE_NAMES[: cfg.heads]
        backend = (
            "sdpa"
            if save_kv is not None or (guide_enabled and AuxRequest is None)
            else cfg.history_backend
        )

        def layout(name, qr, kr):
            return AttentionLayout(masks[name], backend, qr, kr)

        def audio_layout(prefix, qr):
            return tuple(
                layout(prefix + stream, qr, c.get("grid_rows")).bind_audio(*pair)
                for stream, pair in zip(STREAMS, audio)
            )

        main = layout("main", token_rows, token_rows)
        acoustic = audio_layout("audio_", token_rows)
        if cfg.stage == 0 or auxiliary:
            minimal = self.minimal_timing_state(c["active_state"], coords[selected, 3])
        if cfg.stage == 0:
            x = x.index_add(0, selected, minimal.to(x.dtype))
        if auxiliary:
            clean, read = x, x[selected] + minimal
            fields = masks["active_fields"]
            field_inputs = x.index_select(0, fields)
            field_ticks = ticks.index_select(0, fields)
            clean_mask = layout("clean", token_rows, token_rows)
            read_mask = layout("read", read_rows, token_rows)
            read_audio = audio_layout("read_audio_", read_rows)
            stream_roles = torch.cat((roles, roles, roles[selected]))
            stream_ticks = torch.cat((coords[:, 1], coords[:, 1], coords[selected, 1]))
            token_ticks = ticks.repeat(2)

        def guided_audio(stream, attend, key, alpha):
            def apply(q):
                out, lse = attend(q, lse_heads=1)
                guides[stream] = guides[stream] + alpha * attention_guide(
                    q, key, lse, c["guide"][stream], 0
                )
                return out

            return apply

        def joint_audio(primary, clean_attend, read_attend):
            def apply(q):
                qm, qc, qr = q.split((len(ticks), len(ticks), len(selected)), dim=2)
                return torch.cat(
                    (primary(qm), clean_attend(qc), read_attend(qr)), dim=2
                )

            return apply

        for layer, block in enumerate(self.blocks):
            alpha = (
                cfg.attn_guide_layers[layer]
                if supervised and layer < len(cfg.attn_guide_layers)
                else 0
            )
            primary = acoustic
            lookahead = (
                (c["slide_lookahead"], alpha, guides)
                if lookahead_enabled and alpha > 0
                else None
            )
            if guide_enabled and alpha and cross_scale:
                primary = tuple(
                    guided_audio(stream, attend, k, alpha)
                    if c["guide"][stream]["rows"].numel()
                    else attend
                    for stream, attend, (k, _) in zip(STREAMS, acoustic, audio)
                )

            def main_context(q, k, v):
                if not (guide_enabled and alpha and cfg.stage == 1):
                    return main(q, k, v)
                out, lse = main(q, k, v, lse_heads=len(self_guides))
                for h, name in enumerate(self_guides):
                    guides[name] = guides[name] + alpha * attention_guide(
                        q, k, lse, c["guide"][name], h
                    )
                return out

            if auxiliary:
                states = torch.cat((x, clean))
                if layer == 0:
                    qm, km, vm = block.projections(x, ticks)
                    qc, kc, vc = qm, km, vm
                    read_k, read_v = kc, vc
                else:
                    q, k, v = block.projections(states, token_ticks)
                    qm, qc = q.chunk(2, dim=2)
                    km, kc = k.chunk(2, dim=2)
                    vm, vc = v.chunk(2, dim=2)
                    read_k, read_v = kc, vc
                if layer and fields.numel():
                    # An active Hold/Slide may follow a Tap in the same moment. Its
                    # contextual clean KV would carry that Tap back into EOM/BEAT.
                    d = x.shape[-1]
                    key, value = F.linear(
                        block.self_norm(field_inputs),
                        block.qkv.weight[d:],
                        block.qkv.bias[d:],
                    ).chunk(2, -1)
                    read_k = kc.index_copy(2, fields, block.rotated(key, field_ticks))
                    read_v = vc.index_copy(2, fields, block.split(value))
                qr = block.read_query(read, ticks[selected])
                ctx = torch.cat(
                    (
                        main_context(qm, km, vm),
                        clean_mask(qc, kc, vc),
                        read_mask(qr, read_k, read_v),
                    ),
                    dim=2,
                )
                combined = block.finish(
                    torch.cat((states, read)),
                    ctx,
                    tuple(joint_audio(*a) for a in zip(primary, acoustic, read_audio)),
                    stream_roles,
                    stream_ticks,
                    lookahead=lookahead,
                )
                x, clean, read = combined.split((len(ticks), len(ticks), len(selected)))
            else:
                q, k, v = block.projections(x, ticks)
                if save_kv is not None:
                    save_kv(k, v)
                x = block.finish(
                    x,
                    main_context(q, k, v),
                    primary,
                    roles,
                    coords[:, 1],
                    timing if cfg.stage == 0 else None,
                    lookahead=lookahead,
                )
        rows, beats, legal = c["emit_rows"], c["emit_beats"], c["moment_legal"]
        if save_kv is not None:
            x = x[-1:]
            rows = torch.zeros_like(rows)
            # Prefill consumes only the last decision, never project the entire song here.
            grid_feat = grid_feat.reshape(-1, 24, cfg.dim)[beats].flatten(0, 1)
            beats = torch.zeros_like(beats)
        evidence = self.timing_head.evidence(grid_feat) if len(rows) else None

        def readout(states):
            h = self.norm(states)
            return self.timing_head(self.output(h), h, rows, beats, legal, evidence)

        logits = readout(x)
        aux = readout(clean.index_copy(0, selected, read)) if auxiliary else None
        return logits, aux

    def losses(self, logits, aux, chart, metrics=None):
        legal, ids = chart["legal"], chart["targets"]
        if legal.dtype != torch.bool:
            if "token_rows" in chart:
                legal = legal.index_select(0, chart["token_rows"])
            shifts = torch.arange(7, -1, -1, device=legal.device, dtype=torch.uint8)
            legal = ((legal[..., None] >> shifts) & 1).flatten(1)[:, :SIZE].bool()
        main_rows = legal.sum(-1) > 1
        z = logits.float().masked_fill(~legal, -torch.inf)
        union = chart.get("union")
        nll = main_nll(z, ids, union).masked_fill(~main_rows, 0)
        weights = chart.get("weights", main_rows).masked_fill(~main_rows, 0)
        main, weight_sum = (nll * weights).sum(), weights.sum()
        if metrics is not None:
            metrics.update(
                main_support=main_rows.sum(),
                nll_per_moment=nll.detach().sum() / max(1, chart["moments"]),
            )
        if aux is None:
            return main, weight_sum, main.detach() * 0, main_rows.sum() * 0
        z = aux.float().masked_fill(~legal, -torch.inf)
        target_group = self.coarse[ids]
        accept = self.coarse[None, :] == target_group[:, None]
        if union is not None and union["rows"].numel():
            # Collect every alternative group before expanding groups to token IDs.
            # Several alternatives may belong to the same row or motion group.
            rows = union["rows"]
            groups = torch.zeros((len(rows), SIZE), dtype=torch.bool, device=z.device)
            groups[union["inverse"], self.coarse[union["cols"]]] = True
            accept[rows] |= groups[:, self.coarse]
        active = main_rows & (legal & ~accept).any(-1)
        loss = torch.logsumexp(z, -1) - torch.logsumexp(
            z.masked_fill(~accept, -torch.inf), -1
        )
        return main, weight_sum, loss.masked_fill(~active, 0).sum(), active.sum()

    def loss(
        self, charts, bank, indices=None, *, cross_scale=1.0, no_history_weight=0.5
    ):
        # The normal DataLoader passes a single worker-packed logical batch.
        if isinstance(charts, list):
            from .data import pack_charts
            from torch.utils._pytree import tree_map_only

            charts = pack_charts(
                tree_map_only(torch.Tensor, lambda t: t.cpu(), charts), indices.tolist()
            )
            charts = tree_map_only(
                torch.Tensor, lambda t: t.to(bank.mel_feat.device), charts
            )
        cond = self.conditioning(*charts["condition"]).index_select(
            0, charts["chart_ids"]
        )
        audio = self.audio_keys(bank, chart=charts)
        if (
            self.config.stage == 1
            and not self.beat_encoder.proj["conv"].weight.requires_grad
        ):
            cross_scale = 0.0
        names = (
            (*STREAMS, *SELF_GUIDE_NAMES[: self.config.heads])
            if self.config.stage == 1
            else STREAMS
        )
        guides = {s: cond.new_zeros((), dtype=torch.float32) for s in names}
        guides["mert_slide_2b"] = cond.new_zeros((), dtype=torch.float32)
        with packed_projection_scope():
            z, aux = self.decode(
                charts,
                audio,
                cond,
                auxiliary=self.config.stage == 1 and self.config.no_history_enabled,
                guides=guides,
                cross_scale=cross_scale,
            )
        metrics = {}
        main, count, extra, support = self.losses(z, aux, charts, metrics)
        moments = charts["moments"]
        m, a = main / count.clamp_min(1e-6), extra / support.clamp_min(1)
        lookahead_loss = guides.pop("mert_slide_2b")
        guide = sum(
            value * (cross_scale if name in STREAMS else 1.0)
            for name, value in guides.items()
        ) / len(names)
        metrics.update(
            attn_guide=guide.detach(),
            mert_slide_2b=lookahead_loss.detach(),
            **{"attn_" + k: v.detach() for k, v in guides.items()},
            main=m.detach(),
        )
        loss = m + self.config.attn_guide_weight * guide
        if self.config.no_history_enabled:
            metrics.update(no_history=a.detach(), no_history_support=support.detach())
            loss = loss + no_history_weight * a
        return (
            loss + self.config.slide_lookahead_weight * cross_scale * lookahead_loss,
            metrics,
            moments,
        )
