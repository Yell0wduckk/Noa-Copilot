"""Masking and pooling helpers for continuous audio tensors."""
from torch import Tensor
from torch.nn import functional as F

from .types import ScaleMemory


def mask_values(values: Tensor, valid: Tensor) -> Tensor:
    """Clear padding, including arbitrary non-finite padding payloads."""
    return values.masked_fill(~valid.unsqueeze(-1), 0)


def pool_stride(values: Tensor, beats: Tensor, valid: Tensor) -> ScaleMemory:
    """Mask-weighted k=3/s=2/p=1 pooling, keeping valid sampling centers."""
    if values.shape[1] == 0:
        return ScaleMemory(values, beats.float(), valid)
    weights = valid.float().unsqueeze(1)
    clean = mask_values(values, valid).float().transpose(1, 2)
    clean_beats = beats.float().masked_fill(~valid, 0).unsqueeze(1)

    def pooled(value):
        return F.avg_pool1d(value, 3, stride=2, padding=1, count_include_pad=False)

    mass = pooled(weights)
    pooled_values = (pooled(clean) / mass.clamp_min(1e-12)).transpose(1, 2)
    pooled_beats = (pooled(clean_beats) / mass.clamp_min(1e-12)).squeeze(1)
    pooled_valid = mass.squeeze(1).gt(0) & valid[:, ::2]
    return ScaleMemory(
        mask_values(pooled_values.to(values.dtype), pooled_valid),
        pooled_beats.masked_fill(~pooled_valid, 0),
        pooled_valid,
    )
