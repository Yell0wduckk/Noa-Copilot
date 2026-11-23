"""Fold3 projection bank from E0 (24 fps) to P0 (8 fps)."""

from torch import nn


class P0Bank(nn.Module):
    """Fold three E0 frames into P0; no multiscale pyramid projections."""

    def __init__(self, d_model, fpn_rates, context_beats, audio_before_beats, *, p0_normalization):
        super().__init__()
        from .p0_conformer import P0OutputNorm
        self.fpn_rates = tuple(fpn_rates)
        self.fold3 = nn.Linear(3 * d_model, d_model)
        self.p0_norm = P0OutputNorm(d_model) if p0_normalization else nn.Identity()
