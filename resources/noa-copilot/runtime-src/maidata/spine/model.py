"""MOMENT205. Reflection exchanges two tied spatial channels in each layer."""

from dataclasses import dataclass, field
from typing import NamedTuple
import math
import torch
from torch import nn
from torch.nn import functional as F
from maidata.designer import DESIGNER_VOCAB

from .history import bucket_attention, materialize, prepare_attention, sample_bank
from maidata.chart.routes import mirror_route
from .codec import PAD_MAP
from .pieces import GeometryHead
from .features import CLOCK_DIM
from .tails import REM, ROUTE
from .timing import MomentTiming
from .packing import BEAT, BEAT_WORD, MOMENT, TIME_COUNT
from .tails import PAD_FEEDBACK, DETAIL_DIM, HAZ_DIM, HAZ_OFFSETS, FIELD_COUNT, START
from maidata.multitask.audio import sample_centers_batch
from maidata.multitask.attention import dual_rotary


@dataclass(frozen=True)
class SpineConfig:
    spine_dim: int = 512
    spine_layers: int = 4
    tail_dim: int = 512
    tail_layers: int = 2
    history_beats: int = 16
    history_query_block: int = 128
    history_backend: str = "auto"
    head_dim: int = 64
    rotary_step_dims: int = 8
    audio_kv_heads: int = 2
    ffn_multiplier: int = 4
    piece_dim: int = 256
    endpoint_audio_dim: int = 32
    field_dim: int = 128
    flag_dim: int = 128
    adapter_layers: int = 2
    adapter_kernel: int = 5
    delta_gain: float = 16.0
    dropout: float = 0.0
    charter_dropout: float = 0.25
    designer_vocab: list[str] = field(default_factory=lambda: list(DESIGNER_VOCAB))
    audio_offsets: list[int] = field(default_factory=lambda: list(range(-16, 49, 4)))
    tail_past_seconds: float = 0.256
    tail_future_seconds: float = 2.048
    condition_dropout: float = 0.05
    level_dropout: float = 0.1
    difficulty_dropout: float = 0.1
    tail_audio_stride_frames: int = 4
    beat_probes: list[float] = field(
        default_factory=lambda: [-1, -0.5, 0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4]
    )
    tail_probes: list[float] = field(
        default_factory=lambda: [
            -1,
            -0.5,
            0.25,
            0.5,
            0.75,
            *[i / 2 for i in range(2, 17)],
        ]
    )

    @property
    def tail_audio_offsets(self):
        def side(seconds):
            n = round(seconds * 125)
            return (
                [
                    *range(
                        self.tail_audio_stride_frames, n, self.tail_audio_stride_frames
                    ),
                    n,
                ]
                if n
                else []
            )

        return (
            [-i for i in reversed(side(self.tail_past_seconds))]
            + [0]
            + side(self.tail_future_seconds)
        )


class Bank(NamedTuple):
    frames: torch.Tensor
    lengths: torch.Tensor
    kv: torch.Tensor


class Adapter(nn.Module):
    def __init__(self, dim, kernel):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel, padding="same", groups=dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, 1),
        )

        nn.init.zeros_(self.conv[-1].weight)
        nn.init.zeros_(self.conv[-1].bias)

    def forward(self, x, pad):
        # Per-frame channel RMS in channel-first layout. Accumulate in FP32.
        scale = x.float().square().mean(1, keepdim=True).add(1e-6).rsqrt().to(x.dtype)
        z = x * scale * self.norm.weight.to(x.dtype)[None, :, None]
        return x + self.conv(z.masked_fill(pad, 0))


class AudioFusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim = cfg.audio_dim if hasattr(cfg, "audio_dim") else cfg.spine_dim
        self.delta_gain = cfg.delta_gain
        self.mel = nn.Linear(256, dim)
        self.delta = nn.Linear(256, dim, bias=False)
        self.mert_layer_logits = nn.Parameter(torch.zeros(2))
        self.mert = nn.Linear(1024, dim, bias=False)
        self.muq = nn.Sequential(
            nn.LayerNorm(1024, elementwise_affine=False),
            nn.Linear(1024, dim, bias=False),
        )
        self.macro_gate = nn.Linear(3 * dim, 2 * dim)
        nn.init.zeros_(self.macro_gate.weight)
        nn.init.constant_(self.macro_gate.bias, -2.0)
        self.adapters = nn.ModuleList(
            Adapter(dim, cfg.adapter_kernel) for _ in range(cfg.adapter_layers)
        )
        self.kv = nn.Linear(dim, 2 * cfg.audio_kv_heads * cfg.head_dim, bias=False)

    def forward(self, audio):
        mel = audio["mel"].to(self.mel.weight.dtype)
        m = self.mel(mel[..., :256]) + self.delta(mel[..., 256:] * self.delta_gain)
        lengths = audio["mel_lengths"]
        frames = torch.arange(m.shape[1], device=m.device)
        seconds = (frames.double() + 0.5) / 125

        def align(key, x, rate):
            return sample_centers_batch(
                x, audio[key + "_lengths"], rate, seconds, audio[key + "_origins"]
            )

        a, b = [
            F.layer_norm(audio[k].to(self.mert.weight.dtype), (1024,))
            for k in ("mert_l6", "mert")
        ]
        c = self.mert(
            torch.lerp(a, b, self.mert_layer_logits.softmax(0)[1].to(a.dtype))
        )
        u = self.muq(audio["muq"].to(self.mert.weight.dtype))

        c, u = align("mert", c, 75), align("muq", u, 25)
        gc, gu = self.macro_gate(torch.cat((m, c, u), -1)).sigmoid().chunk(2, -1)
        x = (m + gc * c + gu * u).transpose(1, 2).contiguous()
        pad = frames[None, None, :] >= lengths[:, None, None]
        for adapter in self.adapters:
            x = adapter(x, pad)
        x = x.masked_fill(pad, 0).transpose(1, 2).contiguous()
        return Bank(x, lengths, self.kv(x))


class ProjectionRMSNorm(nn.RMSNorm):
    """Apply affine RMSNorm in the projection dtype, including under autocast."""

    def forward(self, x):
        return F.rms_norm(x, self.normalized_shape, self.weight.to(x.dtype), self.eps)


class LocalContext(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.hazard = nn.Linear(
            HAZ_DIM, 2 * cfg.audio_kv_heads * cfg.head_dim, bias=False
        )
        self.register_buffer(
            "mirror", torch.tensor(PAD_MAP[8][8:]) - 8, persistent=False
        )
        self.register_buffer(
            "offsets", torch.tensor(cfg.audio_offsets) / 125, persistent=False
        )
        self.register_buffer(
            "tail_offsets", torch.tensor(cfg.tail_audio_offsets) / 125, persistent=False
        )
        self.tail_past = -cfg.tail_audio_offsets[0] / 125
        self.tail_future = cfg.tail_audio_offsets[-1] / 125
        self.hazard_norm = ProjectionRMSNorm(
            2 * cfg.audio_kv_heads * cfg.head_dim, eps=1e-6
        )
        self.register_buffer(
            "lane",
            torch.arange(2).repeat_interleave(len(HAZ_OFFSETS)),
            persistent=False,
        )

    def forward(self, bank, owners, clock, hazard, *, tail=False):
        sec = clock[..., CLOCK_DIM : CLOCK_DIM + 1]
        offsets = self.tail_offsets if tail else self.offsets
        probes = clock[..., CLOCK_DIM + 1 :]
        if tail:
            in_window = (probes >= sec - self.tail_past) & (
                probes <= sec + self.tail_future
            )
            probes = torch.where(in_window, probes, sec)
        times = torch.cat((sec + offsets, probes), -1)
        kv = sample_bank((bank.kv, bank.lengths), owners[:, None, None], times)
        z = hazard.unflatten(-1, (len(HAZ_OFFSETS), 4, 33)).to(kv.dtype)
        pair = (
            torch.stack((z, z.index_select(-1, self.mirror)), -4)
            .flatten(-4, -3)
            .flatten(-2)
        )
        # [B,T,2*offsets,4*33]. Each query lane sees its own hazard spelling.
        active = pair.any(-1)
        hk = self.hazard_norm(self.hazard(pair)).masked_fill(~active[..., None], 0)
        allowed = active.unsqueeze(-2) & (
            self.lane == torch.arange(2, device=z.device)[:, None]
        )
        if tail:
            # Beat probes are optional extra keys, never an extension of the time window.
            audio_allowed = F.pad(in_window, (len(offsets), 0), value=True)
            allowed = torch.cat(
                (audio_allowed.unsqueeze(-2).expand(*allowed.shape[:-1], -1), allowed),
                -1,
            )
        else:
            allowed = F.pad(allowed, (kv.shape[-2], 0), value=True)
        # Efficient CUDA SDPA accepts an additive mask with aligned row strides.
        # Pad only the mask storage; attention still sees exactly the real keys.
        allowed = F.pad(allowed, (0, -allowed.shape[-1] % 8), value=False)
        return torch.cat((kv, hk), -2), allowed


class LocalAudio(nn.Module):
    def __init__(self, width, cfg, probes, offsets):
        super().__init__()
        self.d = cfg.head_dim
        self.kvh = cfg.audio_kv_heads
        self.groups = max(1, math.ceil(width / (self.d * self.kvh)))
        self.q = nn.Linear(width, self.kvh * self.groups * self.d, bias=False)
        self.out = nn.Linear(self.kvh * self.groups * self.d, width, bias=False)
        self.offset_bias = nn.Parameter(
            torch.zeros(
                self.kvh,
                self.groups,
                len(offsets) + len(probes) + len(HAZ_OFFSETS),
            )
        )

    def forward(self, x, local):
        kv, allowed = local
        b, t = x.shape[:2]
        k, v = kv.view(b * t, -1, self.kvh, 2 * self.d).transpose(1, 2).chunk(2, -1)
        q = (
            self.q(x)
            .view(b * t, 2, self.kvh, self.groups, self.d)
            .transpose(1, 2)
            .reshape(b * t, self.kvh, 2 * self.groups, self.d)
        )
        bias = torch.cat(
            (self.offset_bias, self.offset_bias[..., -len(HAZ_OFFSETS) :]), -1
        )
        bias = F.pad(bias, (0, -bias.shape[-1] % 8))
        bias = (
            bias[None, :, None]
            .to(q.dtype)
            .masked_fill(~allowed.reshape(b * t, 1, 2, 1, -1), -torch.inf)
            .flatten(2, 3)
        )
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias[..., : k.shape[-2]]
        )
        return self.out(
            out.reshape(b * t, self.kvh, 2, self.groups, self.d)
            .transpose(1, 2)
            .reshape(b, t, 2, -1)
        )


class Block(nn.Module):
    def __init__(self, dim, cfg, probes, offsets):
        super().__init__()
        width = dim // 2
        self.d = cfg.head_dim
        self.step_dims = min(cfg.rotary_step_dims // 2 * 2, max(0, self.d - 2))
        self.heads = max(1, width // self.d)
        self.attn_dim = self.heads * self.d
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * self.attn_dim, bias=False)
        self.out = nn.Linear(self.attn_dim, width, bias=False)
        self.cross_norm = nn.LayerNorm(width)
        self.cross = LocalAudio(width, cfg, probes, offsets)
        self.dropout = cfg.dropout
        self.ffnorm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(2 * width, cfg.ffn_multiplier * width),
            nn.GELU(),
            nn.Linear(cfg.ffn_multiplier * width, width),
        )

    def heads_view(self, v):
        return v.reshape(*v.shape[:2], 2 * self.heads, self.d).transpose(1, 2)

    def rotate(self, heads, ticks, steps):
        return dual_rotary(heads, ticks / 96, steps, self.step_dims)

    def record_kv(self, z, record, ticks, steps, grid=lambda x: x):
        k, v = F.linear(z + record, self.qkv.weight[self.attn_dim :]).chunk(2, -1)
        return self.rotate(self.heads_view(grid(k)), ticks, steps), self.heads_view(
            grid(v)
        )

    def forward(
        self,
        x,
        first,
        last,
        local,
        layout,
        cache,
        layer,
        record=None,
        pending=None,
        include=True,
        valid=None,
    ):
        # Pointwise work stays on [real rows,1,2,d]. Only self-attention uses the padded grid.
        def grid(v):
            if valid is None:
                return v
            out = v.new_zeros((*first.shape, *v.shape[2:]))
            out[valid] = v[:, 0]
            return out

        steps = (
            torch.arange(first.shape[1], device=x.device)
            if cache is None
            else first.new_full((1,), cache.step_count)
        )
        z = self.norm(x)
        if record is not None or pending is not None:
            q = F.linear(z, self.qkv.weight[: self.attn_dim])
            if pending is None:
                k, v = self.record_kv(z, record, first, steps, grid)
            else:
                pending.append(z)
        else:
            q, k, v = self.qkv(z).chunk(3, -1)
            k, v = (
                self.rotate(self.heads_view(grid(k)), first, steps),
                self.heads_view(grid(v)),
            )
        q = self.rotate(self.heads_view(grid(q)), last, steps)
        if cache is None:
            attn = bucket_attention(q, k, v, layout, 0.0)
        else:
            if pending is None:
                cache.write(layer, k, v)
            attn = cache.attend(layer, q, pending is None and include)
        attended = attn.transpose(1, 2).reshape(*first.shape, 2, -1)
        if valid is not None:
            attended = attended[valid][:, None]
        x = x + self.out(attended)
        x = x + self.cross(self.cross_norm(x), local)
        z = self.ffnorm(x)
        y = self.ffn(torch.cat((z, z.flip((-2,))), -1))
        if self.dropout and self.training:
            # Shared dropout mask preserves reflection for each realization.
            y = y * F.dropout(torch.ones_like(y[..., :1, :]), self.dropout, True)
        return x + y


def mlp(inp, out, hidden=None):
    return nn.Sequential(
        nn.LayerNorm(inp),
        nn.Linear(inp, hidden or out),
        nn.GELU(),
        nn.Linear(hidden or out, out),
    )


def chart_condition(self, values, ids):
    level, difficulty = values.unbind(-1)
    if self.training:
        draws = torch.rand((4, *ids.shape), device=ids.device)
        missing = draws[0] < self.config.condition_dropout
        level = level.masked_fill(
            missing | (draws[1] < self.config.level_dropout), 0
        )
        difficulty = difficulty.masked_fill(
            missing | (draws[2] < self.config.difficulty_dropout), 0
        )
        ids = ids.masked_fill(missing | (draws[3] < self.config.charter_dropout), 0)
    angle = level[..., None] * self.level_freq
    l = self.level_mlp(torch.cat((angle.sin(), angle.cos()), -1))
    return (
        torch.where(level[..., None] != 0, l, self.null_level)
        + self.difficulty(difficulty.long())
        + self.charter(ids).masked_fill((ids == 0)[..., None], 0)
    )


class SpineModel(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        cfg = self.config = cfg or SpineConfig()
        dim = cfg.spine_dim
        d = dim // 2
        self.register_buffer("mirror_pad", torch.tensor(PAD_MAP[8]), persistent=False)
        self.timing = MomentTiming()
        self.fusion = AudioFusion(cfg)
        self.local = LocalContext(cfg)
        self.action = nn.Embedding(BEAT_WORD + 1, d)
        self.dt_embed = nn.Embedding(TIME_COUNT, d)
        self.feedback = nn.Linear(41 * PAD_FEEDBACK, d, bias=False)
        self.feedback_norm = ProjectionRMSNorm(d, eps=1e-6)
        self.level_mlp = nn.Sequential(nn.Linear(16, d), nn.GELU(), nn.Linear(d, d))
        self.register_buffer(
            "level_freq", torch.arange(1, 9) * math.pi / 15, persistent=False
        )
        self.null_level = nn.Parameter(torch.zeros(d))
        self.difficulty = nn.Embedding(8, d, padding_idx=0)
        self.charter = nn.Embedding(len(cfg.designer_vocab) + 1, d, padding_idx=0)
        self.phase = nn.Sequential(nn.Linear(CLOCK_DIM, d), nn.GELU(), nn.Linear(d, d))
        self.spine = nn.ModuleList(
            Block(dim, cfg, cfg.beat_probes, cfg.audio_offsets)
            for _ in range(cfg.spine_layers)
        )
        self.norm = nn.LayerNorm(d)
        self.dt = nn.Linear(d, TIME_COUNT)
        self.pieces = GeometryHead(d, cfg.piece_dim)
        self.onset = nn.Linear(dim, d)
        self.word_mix = mlp(4 * d, d)
        self.slot_mix = mlp(6 * d + 4, d)
        self.pad = nn.Embedding(41, d)
        self.offset = nn.Linear(2, d)
        self.slot_norm = nn.LayerNorm(d)
        self.members = nn.Linear(d + DETAIL_DIM, 2)
        self.kind_head = nn.Linear(d, 3)
        self.type_embed = nn.Embedding(3, d)
        self.previous_flags = nn.Embedding(8, d)
        self.slot_history = nn.Sequential(
            nn.Linear(41 * 6, d, bias=False), ProjectionRMSNorm(d, eps=1e-6)
        )
        self.geometry = nn.Linear(41, d, bias=False)
        td = cfg.tail_dim // 2
        self.tail_in = mlp(2 * d, td)
        self.tail_phase = nn.Sequential(
            nn.Linear(CLOCK_DIM, td), nn.GELU(), nn.Linear(td, td)
        )
        self.tail_start_pad = nn.Embedding(41, td)
        self.tail_descriptor = nn.Linear(DETAIL_DIM + 41, td, bias=False)
        self.tail = nn.ModuleList(
            Block(cfg.tail_dim, cfg, cfg.tail_probes, cfg.tail_audio_offsets)
            for _ in range(cfg.tail_layers)
        )
        self.tail_norm = nn.LayerNorm(td)
        self.field_embed = nn.Embedding(FIELD_COUNT, cfg.field_dim)
        self.fields = nn.GRU(cfg.field_dim, td, batch_first=True)
        self.field_out = nn.Linear(td, FIELD_COUNT)
        self.endpoint_keys = nn.Linear(dim, cfg.endpoint_audio_dim, bias=False)
        self.endpoint_query = nn.Sequential(
            nn.LayerNorm(td), nn.Linear(td, cfg.endpoint_audio_dim)
        )
        field_mirror = torch.arange(FIELD_COUNT)
        field_mirror[ROUTE : ROUTE + 71] = torch.tensor(
            [ROUTE + mirror_route("A1", r) for r in range(71)]
        )
        self.register_buffer("field_mirror", field_mirror, persistent=False)
        self.flags = mlp(d + DETAIL_DIM, 8, cfg.flag_dim)

    conditioning = chart_condition

    def spine_states(
        self,
        steps,
        feedback,
        hazard,
        condition,
        bank,
        indices,
        onset,
        clock,
        *,
        layout=None,
        cache=None,
        cursor=None,
        valid=None,
        footprint=None,
    ):
        ticks = steps[..., 2]
        if valid is not None:
            owners = valid[0]
            condition, indices = condition[owners], indices[owners]
            steps, feedback, hazard, onset, clock = (
                v[valid][:, None] for v in (steps, feedback, hazard, onset, clock)
            )
            if footprint is not None:
                footprint = footprint[valid][:, None]
        z = feedback.unflatten(-1, (41, PAD_FEEDBACK))
        fb = torch.stack((z, z.index_select(-2, self.mirror_pad)), -3).flatten(-2)
        x = self.action(steps[..., 0].long()).unsqueeze(-2) + self.feedback_norm(
            self.feedback(fb)
        ).masked_fill(~fb.any(-1, keepdim=True), 0)
        if footprint is not None:
            x = x + self.footprint(
                ((footprint[..., None] & self.pieces.bits) != 0).to(x.dtype)
            )
        x = x + (
            condition[:, None]
            + self.dt_embed(steps[..., 1].long())
            + self.phase(clock[..., :CLOCK_DIM])
        ).unsqueeze(-2)
        x = x + onset.masked_fill((steps[..., 0] != MOMENT)[..., None], 0).unsqueeze(-2)
        local = self.local(bank, indices, clock, hazard)
        if cache is not None:
            cache.begin(cursor[1])
        else:
            layout = prepare_attention(layout)
        for j, layer in enumerate(self.spine):
            x = layer(
                x,
                ticks,
                ticks,
                local,
                layout,
                cache,
                j,
                valid=valid,
                include=cache is None or cursor[0] >= cache.lower,
            )
        if cache is not None:
            cache.finish(cursor[0])
        return self.norm(x) if valid is None else self.norm(x[:, 0])

    def word_state(self, h, dt, clock, a):
        cond = torch.cat((self.dt_embed(dt), a, self.phase(clock[..., :CLOCK_DIM])), -1)
        return h + self.word_mix(
            torch.cat((h, cond.unsqueeze(-2).expand(*h.shape[:-1], -1)), -1)
        )

    def pad_pair(self, pad):
        return torch.stack((pad, self.mirror_pad[pad]), -1)

    def slot_state(self, wh, geom, pad, group, occ, onset, hazard, committed):
        x = wh
        prior = torch.stack(
            (committed, committed[..., self.mirror_pad, :]), -3
        ).flatten(-2)
        prior = self.slot_history(prior.to(x.dtype)).masked_fill(
            (committed.any(dim=(-2, -1)) == 0)[..., None, None], 0
        )
        y = self.offset(torch.stack((group / 4, occ.float()), -1))
        shared = (
            torch.cat((y, onset, hazard.float()), -1)
            .unsqueeze(-2)
            .expand(*x.shape[:-1], -1)
        )
        return x + self.slot_mix(
            torch.cat((x, geom, prior, self.pad(self.pad_pair(pad)), shared), -1)
        )

    def footprint(self, pads):
        return self.geometry(torch.stack((pads, pads[..., self.mirror_pad]), -2))

    def tail_record(self, pad, descriptor, ends):
        flipped = torch.cat(
            (descriptor[..., :-2], descriptor[..., -2:].flip((-1,))), -1
        )
        pair = torch.stack(
            (
                torch.cat((descriptor, ends), -1),
                torch.cat((flipped, ends[..., self.mirror_pad]), -1),
            ),
            -2,
        )
        return self.tail_start_pad(self.pad_pair(pad)) + self.tail_descriptor(pair)

    def tail_states(
        self,
        x,
        kind,
        times,
        clock,
        bank,
        indices,
        hazard,
        geometry,
        *,
        layout=None,
        record=None,
        cache=None,
        tick=None,
        pending=None,
        valid=None,
    ):
        if valid is not None:
            indices = indices[valid[0]]
            kind, clock, hazard = (v[valid][:, None] for v in (kind, clock, hazard))
            x, geometry, record = (v[:, None] for v in (x, geometry, record))
        phase = self.tail_phase(clock[..., :CLOCK_DIM])
        x = self.tail_in(
            torch.cat((x + self.type_embed(kind).unsqueeze(-2), geometry), -1)
        ) + phase.unsqueeze(-2)
        local = self.local(bank, indices, clock, hazard, tail=True)
        if cache is not None:
            cache.begin(tick)
        else:
            layout = prepare_attention(layout)
        for j, layer in enumerate(self.tail):
            x = layer(
                x, times, times, local, layout, cache, j, record, pending, valid=valid
            )
        return self.tail_norm(x) if valid is None else self.tail_norm(x[:, 0])

    def commit_tail(
        self, cache, pending, tick, times, pad, descriptor, ends, release_tick
    ):
        record = self.tail_record(pad, descriptor, ends)
        for j, (layer, z) in enumerate(zip(self.tail, pending)):
            cache.write(
                j,
                *layer.record_kv(
                    z, record, times, times.new_full((1,), cache.step_count)
                ),
            )
        cache.finish(tick, release_tick)

    def field_sequence(self, tokens, hidden):
        """Run both physical spellings in one GRU batch with shared weights."""
        paired = torch.stack((tokens, self.field_mirror[tokens]), 1)
        n, _, length = paired.shape
        seq, hidden = self.fields(
            self.field_embed(paired.reshape(n * 2, length)), hidden
        )
        return seq.reshape(n, 2, length, -1).transpose(1, 2), hidden

    def field_scores(self, state):
        left, right = self.field_out(state).unbind(-2)
        return (left + right[..., self.field_mirror]) * 0.5

    def endpoint_scores(self, state, keys):
        """One score per candidate phase of the beat selected by the field prefix."""
        query = (
            self.endpoint_query(state.mean(-2)) * self.config.endpoint_audio_dim**-0.5
        )
        return (keys @ query.unsqueeze(-1)).squeeze(-1)

    def field_hidden(self, h):
        return h.reshape(1, -1, self.config.tail_dim // 2).contiguous()

    def flag_scores(self, x, pads):
        logits = self.flags(x).mean(-2)
        codes = torch.arange(8, device=x.device)
        allowed = torch.where(
            pads[..., None] < 8, codes < 4, (codes == 0) | (codes == 4)
        )
        return logits.masked_fill(~allowed, -torch.inf)

    def flags_input(self, x, kind, detail):
        flipped = torch.cat((detail[..., :-2], detail[..., -2:].flip((-1,))), -1)
        return torch.cat(
            (
                x + self.type_embed(kind).unsqueeze(-2),
                torch.stack((detail, flipped), -2),
            ),
            -1,
        )

    def repeat_scores(self, features, flags, can_repeat):
        x, detail = features.split([self.config.spine_dim // 2, DETAIL_DIM], -1)
        logits = self.members(
            torch.cat((x + self.previous_flags(flags).unsqueeze(-2), detail), -1)
        ).mean(-2)
        return torch.stack(
            (logits[..., 0], logits[..., 1].masked_fill(~can_repeat, -torch.inf)), -1
        )

    def kind_scores(self, sn, pad, prior_non_slide, has_time):
        logits = self.kind_head(sn).mean(-2)
        return logits.masked_fill(
            torch.stack((prior_non_slide, prior_non_slide, (pad >= 8) | ~has_time), -1),
            -torch.inf,
        )

    def loss(self, batch, bank, indices, loss_weights=None):
        steps = batch["steps"]
        layout = materialize(batch["spine_layout"], self.config.history_backend)
        chart, row = batch["valid_steps"]
        actual = batch["real_pos"]
        ci, u = chart[actual], row[actual]
        moment_audio = self.onset(
            sample_bank(bank[:2], indices[ci], batch["word_clock"][ci, u, CLOCK_DIM])
        )
        # Row u predicts this MOMENT. Its committed frame enters the spine at u+1.
        onset_audio = moment_audio.new_zeros((*steps.shape[:2], moment_audio.shape[-1]))
        onset_audio[ci, u + 1] = moment_audio
        h = self.spine_states(
            steps,
            batch["feedback"],
            batch["hazard"],
            self.conditioning(batch["conditions"], batch["charter"]),
            bank,
            indices,
            onset_audio,
            batch["spine_clock"],
            layout=layout,
            valid=(chart, row),
            footprint=batch["footprint"],
        )
        target = steps[chart, row].long()
        dt = target[:, 3]
        totals, means = {}, {}

        def ce(name, logits, truth):
            if truth.numel():
                totals[name] = F.cross_entropy(logits, truth, reduction="sum")
                means[name] = totals[name].detach() / truth.numel()

        last_onset, owed, last_small = self.timing.history(steps)
        time_nll = F.cross_entropy(
            self.timing.times(
                self.dt(h).mean(-2),
                target[:, 0],
                target[:, 2],
                last_onset[chart, row],
                owed[chart, row],
                batch["limits"][chart],
                last_small[chart, row],
            ),
            dt,
            reduction="none",
        )
        totals["time"] = time_nll.sum()
        means["time"] = totals["time"].detach() / len(dt)
        wh = self.word_state(
            h[actual], dt[actual], batch["word_clock"][ci, u], moment_audio
        )
        moment_index = torch.zeros(
            steps.shape[:2], dtype=torch.long, device=steps.device
        )
        moment_index[ci, u] = torch.arange(len(ci), device=steps.device)
        piece_state = self.pieces.state(self.pieces.norm(wh))
        pc = batch["piece_chart"]
        pu, earlier, used, subgroup, target_piece = batch["pieces"].long().T
        ce(
            "piece",
            self.pieces(piece_state[moment_index[pc, pu]], earlier, used, subgroup),
            target_piece,
        )
        # The complete footprint becomes visible after the current geometry is committed.
        pads = ((batch["footprint"][ci, u + 1, None] & self.pieces.bits) != 0).to(
            wh.dtype
        )
        geo = self.footprint(pads)
        ci = batch["slot_chart"]
        u, pad, group, membership, kind, flags = batch["slots"].long().T
        owner = moment_index[ci, u]
        x = self.slot_state(
            wh[owner],
            geo[owner],
            pad,
            group,
            batch["occurrence"],
            moment_audio[owner],
            batch["slot_haz"],
            batch["committed_pads"],
        )
        sn = self.slot_norm(x)
        has_time = (
            steps[ci, u, 2] // 96 * 96 + steps[ci, u, 3] < batch["audio_limits"][ci]
        )
        ty = self.kind_scores(sn, pad, batch["prior_non_slide"], has_time)
        ce("type", ty, kind)
        details = batch["detail"]
        if batch["tail_slots"]:
            valid = batch["tail_valid"]
            p = batch["tail_pos"][valid]
            geom = geo[owner[p]]
            th = self.tail_states(
                x[p],
                batch["tail_kind"],
                batch["tail_tick"],
                batch["tail_clock"],
                bank,
                indices[batch["tail_chart"]],
                batch["tailhaz"],
                geom,
                layout=materialize(batch["tail_layout"], self.config.history_backend),
                record=self.tail_record(
                    pad[p], details[p], batch["tail_ends"][valid].to(details.dtype)
                ),
                valid=valid,
            )
            target = batch["field_target"]
            inp = F.pad(target[:, :-1], (1, 0), value=START)
            out, _ = self.field_sequence(inp, self.field_hidden(th))
            ix = batch["field_valid"]
            states = out[ix]
            logits = self.field_scores(states)
            rows = batch["endpoint_rows"]
            keys = sample_bank(
                (self.endpoint_keys(bank.frames), bank.lengths),
                indices[batch["endpoint_chart"]][:, None],
                batch["endpoint_seconds"],
            )
            logits[rows, REM:ROUTE] += self.endpoint_scores(
                states[rows], keys.index_select(0, batch["endpoint_index"])
            )
            ce(
                "tail",
                logits.masked_fill(~batch["field_allowed"], -torch.inf),
                target[ix],
            )
        features = self.flags_input(sn, kind, details)
        ce("flags", self.flag_scores(features, pad), flags)
        ce(
            "repeat",
            self.repeat_scores(
                features, flags, (pad < 8) & has_time & batch["repeat_room"]
            ),
            membership,
        )
        weights = loss_weights or {}
        denom = max(1, batch["moments"])
        time_weight = weights.get("time", 1.0)
        onset = time_nll.index_select(0, actual).sum() * time_weight / denom
        upper = (time_nll * (dt >= BEAT)).sum() * time_weight / denom
        if "piece" in totals:
            upper = upper + totals["piece"] * weights.get("piece", 1.0) / denom
        lower = onset.new_zeros(())
        for name in ("repeat", "type", "tail", "flags"):
            if name in totals:
                lower = lower + totals[name] * weights.get(name, 1.0) / denom
        means["nll_per_moment"] = (onset + upper + lower).detach()
        total = onset + upper + lower
        means.update(
            moment_loss=onset.detach(),
            upper_other_loss=upper.detach(),
            lower_loss=lower.detach(),
        )
        return (
            total,
            means,
            dict(
                words=len(actual),
                pieces=batch["geometry_pieces"],
                steps=len(dt),
                advance=len(dt) - len(actual),
                moments=batch["moments"],
                slots=batch["present_slots"],
                tail_slots=batch["tail_slots"],
            ),
        )
