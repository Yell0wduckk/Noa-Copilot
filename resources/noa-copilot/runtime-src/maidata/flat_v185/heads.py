"""One input projection per token; the expensive path only sees Slide rows."""

import math

import torch
from torch import nn

from .geometry import (
    MAX_SLIDE_STEPS,
    ROUTE_STEP_META,
    ROUTE_STEP_WEIGHTS,
    build_uob_proj,
    build_z_base,
    mod16_clock,
)


# The A ring has radius 240 in the path samples and 0.90 in Z_base.
DISK_UNITS_PER_PIXEL = 0.90 / 240.0


def hot(x, width):
    return x[:, None] == torch.arange(width, device=x.device)


def bit(flags, mask):
    return flags.bitwise_and(mask) != 0


def columns(*xs):
    return torch.cat([x[:, None] if x.ndim == 1 else x for x in xs], dim=1).float()


def masked_clock(tick, mask):
    return mod16_clock(tick) * mask[..., None]


class RMSNorm(nn.RMSNorm):
    def forward(self, x):
        # Match BF16 activations for the fused CUDA kernel; keep the master weight FP32.
        return nn.functional.rms_norm(
            x, self.normalized_shape, self.weight.to(x.dtype), self.eps
        )


def refresh_uob_bf16(module, _incompatible_keys):
    # The checkpoint owns the FP32 projection. Derive the inference copy from it.
    module._uob_bf16 = module.uob_proj.bfloat16()


class HeteroInputHeads(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("z_base", build_z_base())
        self.register_buffer("uob_proj", build_uob_proj())
        self.register_buffer("_uob_bf16", self.uob_proj.bfloat16(), persistent=False)
        self.register_load_state_dict_post_hook(refresh_uob_bf16)
        # Fixed per-step geometry; no repeated dense 33-zone products per chart.
        z = ROUTE_STEP_WEIGHTS @ self.z_base
        meta = ROUTE_STEP_META[None, :, :, :3].expand(8, -1, -1, -1)
        self.register_buffer(
            "step_space", torch.cat((z, meta), -1).reshape(-1, 19), persistent=False
        )
        self.register_buffer("zone_bits", 1 << torch.arange(33), persistent=False)
        self.hold_modality_emb = nn.Parameter(torch.zeros(2, 16))
        self.slide_step_bind = nn.Sequential(nn.Linear(32, 32), nn.SiLU())
        self.projections = nn.ModuleList(
            [
                *(nn.Linear(n, dim, bias=False) for n in (17, 26, 101, 64)),
                nn.Sequential(
                    nn.Linear(449, 384, bias=False),
                    nn.GELU(),
                    nn.Linear(384, dim, bias=False),
                ),
            ]
        )
        nn.init.orthogonal_(self.slide_step_bind[0].weight, gain=math.sqrt(2))
        nn.init.zeros_(self.slide_step_bind[0].bias)
        nn.init.orthogonal_(self.projections[4][0].weight, gain=math.sqrt(2))
        # Match the other heads' pre-normalization scale after GELU.
        nn.init.orthogonal_(self.projections[4][2].weight, gain=1.6)
        self.norms = nn.ModuleList(RMSNorm(dim, eps=1e-6) for _ in range(5))
        self.head_type_bias = nn.Parameter(torch.zeros(5, dim))

    def slide_atoms(self, args, heads):
        refs, slots, dist = (
            heads[k] for k in ("slide_refs", "slide_slots", "slide_dist")
        )
        owners = slots // MAX_SLIDE_STEPS
        first = slots.remainder(MAX_SLIDE_STEPS) == 0
        _, before, total = heads["slide_arc"][owners].T
        hit, launch, end, closed = args[:, [2, 3, 4, 13]][owners].T
        geom = self.step_space[refs.long().abs() - 1]
        tau = torch.where(
            closed.bool(), dist / total, (dist - before) / (total - before)
        )
        arrival = launch + tau * (end - launch)
        previous = torch.where(first, before, dist.roll(1))
        delta = torch.where(
            first & (before == 0),
            launch - hit,
            (dist - previous) / total * (end - launch),
        )
        time = columns(
            mod16_clock(arrival),
            ((arrival - hit).clamp_min(0) / 96).log1p(),
            (delta.clamp_min(0) / 96).log1p(),
        ).masked_fill(~closed.bool()[:, None], 0)
        return columns(geom[:, :18], refs < 0, geom[:, 18], time, tau, closed)

    def features(self, h, a, heads):
        if h == 0:
            tick, kind, group = a.T
            off = tick % 96
            return columns(
                mod16_clock(tick), hot(kind, 6), group.log1p(), off / 96, off == 0
            )
        if h == 2:
            lo, hi, fw_lo, fw_hi, tick, group, phase = a.T

            def unpack(low, high):
                mask = (high.long() << 17) | low
                return bit(mask[:, None], self.zone_bits).float()

            active, fw = unpack(lo, hi), unpack(fw_lo, fw_hi)
            count = active.sum(-1, keepdim=True).clamp_min(1)
            fw_count = fw.sum(-1, keepdim=True)
            return columns(
                active * count.rsqrt(),
                fw * fw_count.clamp_min(1).rsqrt(),
                fw_count > 0,
                (active.to(self.z_base.dtype) @ self.z_base) / count,
                active[:, :8].mean(-1),
                active[:, 8:16].mean(-1),
                active[:, 16],
                active[:, 17:25].mean(-1),
                active[:, 25:].mean(-1),
                count.log1p(),
                count == 1,
                mod16_clock(tick),
                group.log1p(),
                hot(phase, 2),
            )
        z = self.z_base[a[:, 0].long()]
        if h == 1:
            _, flags, head, group, phase = a.T
            return columns(
                z,
                bit(flags, 1),
                bit(flags, 2),
                head == 1,
                head == 2,
                hot(phase, 5),
                group.log1p(),
            )
        if h == 3:
            _, touch, flags, hit, target, pages, group, phase = a.T
            known = phase == 3
            return columns(
                z,
                self.hold_modality_emb[touch.long()],
                mod16_clock(hit),
                masked_clock(target, phase >= 2),
                touch,
                bit(flags, 1),
                bit(flags, 2),
                bit(flags, 4),
                known,
                ((target - hit).clamp_min(0) / 96).log1p() * known,
                (hit == target) & known,
                pages / 8,
                group.log1p(),
                hot(phase, 4),
                z.new_zeros(len(z), 3),
            )
        (
            _,
            dest,
            _,
            launch,
            end,
            segments,
            branch,
            track,
            flags,
            head,
            pages,
            wait,
            phase,
            _,
            group,
        ) = a.T
        arcs = heads["slide_arc"]
        # Bind only real judge steps. Scatter *after* the biased affine and SiLU:
        # unused UOB coordinates stay exactly zero and perform no binding work.
        atoms = self.slide_atoms(a, heads)
        bound = self.slide_step_bind(atoms.to(self.slide_step_bind[0].weight.dtype))
        padded = bound.new_zeros(len(a) * MAX_SLIDE_STEPS, 32)
        padded.index_copy_(0, heads["slide_slots"], bound)
        projection = self.uob_proj
        if padded.dtype == torch.bfloat16 and self._uob_bf16.dtype == padded.dtype:
            projection = self._uob_bf16
        uob = padded.reshape(len(a), -1) @ projection.to(padded.dtype)
        uob = uob * (2.0 * arcs[:, :1].rsqrt())
        return columns(
            uob,
            z,
            self.z_base[dest.long()],
            masked_clock(launch, launch >= 0),
            masked_clock(end, end >= 0),
            (arcs[:, -1] * DISK_UNITS_PER_PIXEL).log1p(),
            segments.log1p(),
            branch.log1p(),
            track,
            bit(flags, 1),
            bit(flags, 2),
            head == 1,
            head == 2,
            pages / 8,
            wait,
            arcs[:, -2] > 0,
            hot(phase, 5),
            group.log1p(),
        )

    def project(self, h, args, heads):
        feat = self.features(h, args, heads).to(self.head_type_bias.dtype)
        return self.norms[h](self.projections[h](feat)) + self.head_type_bias[h]

    def forward(self, heads, length):
        out = self.head_type_bias.new_empty(length, self.head_type_bias.shape[1])
        for h in range(5):
            rows = heads[f"rows{h}"]
            if rows.numel():
                e = self.project(h, heads[f"args{h}"], heads)
                out.index_copy_(0, rows, e.to(out.dtype))
        return out
