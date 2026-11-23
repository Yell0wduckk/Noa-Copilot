"""Waveform-side labels on the same fractional 125 Hz clock as the frontend."""

import numpy as np
from scipy import fft, ndimage, signal
import torch


def _runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return edges.reshape(-1, 2)


def song_ref_and_audibility(wave, sections, gain_db=0.0):
    from .mel_frontend import _weighted_reference_db

    # Reference gating uses double-precision filter coefficients in the original frontend.
    weighted = signal.sosfilt(sections, wave) if len(wave) else np.asarray(wave, dtype=float)
    mask_wave = weighted.astype(np.float32)
    ref = _weighted_reference_db(weighted) + gain_db
    if gain_db:
        mask_wave *= np.float32(10 ** (gain_db / 20))
    return ref, audibility_masks(wave, ref, sections, weighted=mask_wave)


def audibility_masks(wave, ref_db, sections, sample_rate=44100, rate=125, *, weighted=None):
    wave = np.asarray(wave, dtype=np.float32)
    count = max(1, int(np.ceil(len(wave) * rate / sample_rate)))
    centers = ((np.arange(count) + 0.5) * sample_rate / rate).astype(int)
    if weighted is None:
        weighted = signal.sosfilt(sections.astype(np.float32), wave) if wave.size else wave
    size = 512
    padded = np.pad(weighted, size // 2)
    windows = np.lib.stride_tricks.sliding_window_view(padded, size)[centers.clip(max=len(wave))]
    power = np.einsum("ij,ij->i", windows, windows) / size
    db = 10 * np.log10(power.clip(1e-20)) - ref_db
    envelope = np.zeros(count, dtype=bool)
    sustained = [(a, b) for a, b in _runs((db >= -48) & (power > 1e-18)) if b - a >= 3]
    if sustained:
        envelope[max(0, sustained[0][0] - 3) : sustained[-1][1] + 3] = True
    magnitude = np.abs(fft.rfft(windows * np.hanning(size).astype(np.float32)))
    flux = np.maximum(np.diff(magnitude, axis=0, prepend=np.zeros_like(magnitude[:1])), 0).sum(1)
    floor = ndimage.median_filter(flux, size=9, mode="nearest")
    onset = (db >= -44) & (flux > 1.5 * floor + 0.025 * magnitude.sum(1))
    return (
        torch.from_numpy(envelope),
        torch.from_numpy(ndimage.maximum_filter1d(onset, size=7) & envelope),
    )


def inside_audible_bounds(audible):
    """Keep internal quiet measures (including true STOPs) inside the song."""
    seen = audible.long().cumsum(-1)
    return (seen > 0) & ((seen[..., -1:] - seen + audible.long()) > 0)
