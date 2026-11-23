"""Continuous positions in beats, with FP32 phases and adjacent pairs."""
import torch
from torch import Tensor


def rope_tables(beats: Tensor, dim: int, base: float = 10000., scale: float = 1.) -> tuple[Tensor, Tensor]:
    """Build FP32 phases once per forward and shared beat grid/head width.

    Callers own the returned graph-bearing tables. They may share them across
    layers in this forward, but must rebuild them for the next forward. This
    also preserves gradients when beat coordinates themselves are learned.
    """
    with torch.autocast(device_type=beats.device.type, enabled=False):
        frequencies = base ** (
            -torch.arange(0, dim, 2, device=beats.device, dtype=torch.float32) / dim
        )
        phase = beats.float()[:, None, :, None] * scale * frequencies
        return phase.cos(), phase.sin()


def dual_rope_tables(
    beats: Tensor,
    steps: Tensor,
    dim: int,
    step_dim: int,
    base: float = 10000.,
    scale: float = 1.,
    step_base: float = 256.,
    step_scale: float = 1.,
) -> tuple[Tensor, Tensor]:
    """Channel-split 2D RoPE tables: leading (dim - step_dim) for beats, trailing step_dim for token steps."""
    if step_dim <= 0:
        return rope_tables(beats, dim, base, scale)
    if step_dim >= dim or dim % 2 or step_dim % 2:
        raise ValueError(f'invalid dual RoPE dimensions: dim={dim}, step_dim={step_dim}')
    beat_cos, beat_sin = rope_tables(beats, dim - step_dim, base, scale)
    step_cos, step_sin = rope_tables(steps, step_dim, step_base, step_scale)
    return torch.cat((beat_cos, step_cos), dim=-1), torch.cat((beat_sin, step_sin), dim=-1)


def _rotate(value: Tensor, cosine: Tensor, sine: Tensor) -> Tensor:
    """FP32 adjacent-pair rotation, shared by forward and its transpose."""
    with torch.autocast(device_type=value.device.type, enabled=False):
        # Promote the activation once. Keep FP32 rotation (including the sums)
        # so low-precision tables/elementwise rounding cannot change positions.
        floating = value.float()
        even, odd = floating.unflatten(-1, (-1, 2)).unbind(-1)
        rotated = torch.stack((even * cosine - odd * sine,
                               even * sine + odd * cosine), dim=-1).flatten(-2)
    return rotated.to(value.dtype)


class _ConstantRoPE(torch.autograd.Function):
    """Constant-position rotation needs its tables, never the input activation."""
    @staticmethod
    def forward(ctx, value, cosine, sine):
        ctx.save_for_backward(cosine, sine)
        ctx.input_shape = value.shape
        ctx.input_dtype = value.dtype
        return _rotate(value, cosine, sine)

    @staticmethod
    def backward(ctx, grad_output):
        cosine, sine = ctx.saved_tensors
        gradient = _rotate(grad_output.float(), cosine, -sine)
        # Preserve the expression's broadcasting contract for public callers.
        if gradient.shape != ctx.input_shape:
            gradient = gradient.sum_to_size(ctx.input_shape)
        return gradient.to(ctx.input_dtype), None, None


def apply_rope(value: Tensor, beats: Tensor, base: float = 10000., scale: float = 1., *,
               tables: tuple[Tensor, Tensor] | None = None) -> Tensor:
    """Rotate adjacent pairs; optional tables must describe this exact grid."""
    if tables is None:
        tables = rope_tables(beats, value.shape[-1], base, scale)
    cosine, sine = tables
    if cosine.requires_grad or sine.requires_grad:
        return _rotate(value, cosine, sine)
    return _ConstantRoPE.apply(value, cosine, sine)


def sinusoidal(beats: Tensor, width: int, base: float, scale: float) -> Tensor:
    with torch.autocast(device_type=beats.device.type, enabled=False):
        frequency = base ** (-torch.arange(0, width, 2, device=beats.device,
                                         dtype=torch.float32) / width)
        phase = beats.float()[..., None] * scale * frequency
        return torch.stack((phase.sin(), phase.cos()), -1).flatten(-2)[..., :width]
