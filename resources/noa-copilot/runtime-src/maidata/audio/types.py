"""Tensor containers; boolean masks use True for valid audio."""

from dataclasses import dataclass, field
from typing import Callable

from torch import Tensor


@dataclass
class SongInputs:
    mel_patches: Tensor
    sample_valid: Tensor
    beat_centers: Tensor
    aux32_centers: Tensor


@dataclass
class WindowInputs:
    mel: Tensor
    aux32: Tensor
    beats: Tensor
    valid: Tensor
    target_start_beats: Tensor
    window_to_song: Tensor
    layout: dict | None = None


@dataclass
class ScaleMemory:
    values: Tensor
    beats: Tensor
    valid: Tensor


@dataclass
class SongMemory:
    g0: ScaleMemory
    block_kv: tuple[tuple[Tensor, Tensor] | None, ...]


@dataclass
class EncoderOutput:
    P0: ScaleMemory
    P1: ScaleMemory | None = None
    P2: ScaleMemory | None = None


@dataclass
class ContinuousSongInputs:
    """Physically padded L domain; historical architectures also carry G0 input."""
    song: SongInputs | None
    mel: Tensor | None
    aux32: Tensor | None
    beats: Tensor
    valid: Tensor
    domain_start: Tensor
    domain_end: Tensor
    layout: dict | None = None
    spec: Tensor | None = None  # BF16 [B,T,4,204] LogSpec or [B,T,249] CompactSpec
    context: Tensor | None = None
    sample_valid: Tensor | None = None
    silence_spec: Tensor | None = None
    ymt3_frames: Tensor | None = None  # BF16 [B,T_8ms,512]
    muq_frames: Tensor | None = None  # BF16 [B,T_40ms,1024]
    muq_sample_valid: Tensor | None = None  # bool [B,T_40ms]
    muq_valid: Tensor | None = None  # bool [B,T_40ms]
    silence_ymt3: Tensor | None = None  # float32 [512]
    silence_muq: Tensor | None = None  # float32 [1024]


@dataclass
class ContinuousSongEncoding:
    """Normalized P0 output for single-stream continuous encoders (e.g. YMT3MuQEncoder)."""
    P0: ScaleMemory
    beat_memory: ScaleMemory | None = None
    stage_beats: tuple[Tensor, ...] = ()
    stage_valid: tuple[Tensor, ...] = ()
    layout: dict | None = None


@dataclass
class WindowRequests:
    target_start_beats: Tensor
    window_to_song: Tensor
    layout: dict | None = None


@dataclass
class SongEncoding:
    """One song graph. Public P0 is normalized; interpolation uses private U0."""
    _pyramid: EncoderOutput
    stage_beats: tuple[Tensor, ...]
    stage_valid: tuple[Tensor, ...]
    layout: dict
    _p0_norm: Callable[[Tensor], Tensor] = field(repr=False)
    beat_memory: ScaleMemory | None = None
    _p0: ScaleMemory | None = field(default=None, init=False, repr=False)

    @property
    def P0(self) -> ScaleMemory:
        """Normalize once when a whole-song or aligned readout consumes P0.

        Fractional readouts can use U0 without computing an unused normalized
        bank. Like the rest of this object, the cached result owns this forward's
        autograd graph and must not outlive a parameter update.
        """
        if self._p0 is None:
            source = self._pyramid.P0
            values = self._p0_norm(source.values).masked_fill(~source.valid[..., None], 0)
            self._p0 = ScaleMemory(values, source.beats, source.valid)
        return self._p0
