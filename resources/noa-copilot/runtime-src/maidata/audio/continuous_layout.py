"""CPU geometry for E0 Fold3 and local dual-stream G_local audio encoding."""
import torch

from .continuous_attention import prepare_joint


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(to_device(v, device) for v in value)
    return value


def prepare_p0_attention_layout(valid):
    """Prepare whole-song packing and the safe SDPA key mask on the CPU."""
    if valid.device.type != 'cpu' or valid.ndim != 2 or valid.dtype != torch.bool:
        raise ValueError('P0 attention layout requires a CPU boolean song mask')
    if bool(valid.all()):
        return dict(all_valid=True, max_length=valid.shape[-1], mask=None)
    lengths = valid.sum(-1, dtype=torch.int32)
    nonempty = lengths[lengths > 0]
    safe = valid.clone()
    if safe.shape[-1]:
        safe[:, 0] |= ~safe.any(-1)
    return dict(all_valid=False, gather=valid.flatten().nonzero().flatten(),
                cu_seqlens=torch.cat((torch.zeros(1, dtype=torch.int32),
                                      nonempty.cumsum(0, dtype=torch.int32))),
                max_length=int(nonempty.max()) if len(nonempty) else 0,
                mask=safe[:, None, None, :])


def prepare_continuous_layout(inputs, *, radii, rates, slot_radius=None, e0_only=True,
                              samples_per_beat=24, local_dual_stream=True):
    return prepare_e0_layout(
        inputs, radii=radii, samples_per_beat=samples_per_beat,
        p0_rate=rates[0], local_dual_stream=local_dual_stream)


def prepare_e0_layout(inputs, *, radii, samples_per_beat, p0_rate, local_dual_stream=True):
    """E0 trunk with Fold3 P0 and local dual-stream G_local attention layout."""
    beats, valid = inputs.beats.detach().cpu(), inputs.valid.cpu()
    starts, ends = inputs.domain_start.cpu().float(), inputs.domain_end.cpu().float()
    stages = tuple(dict(beats=beats, valid=valid) for _ in radii)
    by_radius = {}
    for radius in radii:
        if radius not in by_radius:
            by_radius[radius] = dict(local=prepare_joint(
                beats, valid, (beats,), (valid,), (radius,), query_width=32))
    attention = tuple(by_radius[radius] for radius in radii)
    p0_valid = torch.nn.functional.pad(valid, (0, (-valid.shape[1]) % 3)).unflatten(1, (-1, 3)).any(-1)
    p0_beats = starts[:, None] + (torch.arange(p0_valid.shape[1]).float()[None] + .5) / p0_rate
    extra = dict(local_beat_attention=prepare_joint(
        p0_beats, p0_valid, (p0_beats,), (p0_valid,), (4.0,), query_width=32))
    return dict(**extra, stages=stages, attention=attention, grids=(p0_beats,), grid_valid=(p0_valid,),
                resample=(), upsample=(), domain_start=starts, domain_end=ends,
                p0_attention=None, beat_upsample=None)
