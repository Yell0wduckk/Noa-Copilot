"""Physical dual-window Mel features on the shared 125 Hz clock."""

from functools import lru_cache
import math

import numpy as np
from scipy import signal
import torch
from torch import nn
from torch.nn import functional as F

from .audio import _mel_filter

FRAME_RATE = 125
SAMPLE_RATE = 44100


@lru_cache
def _k_sections():
    k = math.tan(math.pi * 1681.974450955533 / SAMPLE_RATE)
    q = 0.7071752369554196
    vh = 10 ** (3.999843853973347 / 20)
    vb = vh**0.4996667741545416
    a = 1 + k / q + k * k
    shelf = [
        (vh + vb * k / q + k * k) / a,
        2 * (k * k - vh) / a,
        (vh - vb * k / q + k * k) / a,
        1,
        2 * (k * k - 1) / a,
        (1 - k / q + k * k) / a,
    ]
    k = math.tan(math.pi * 38.13547087602444 / SAMPLE_RATE)
    q = 0.5003270373238773
    a = 1 + k / q + k * k
    return np.array([shelf, [1, -2, 1, 1, 2 * (k * k - 1) / a, (1 - k / q + k * k) / a]])


def song_reference_db(waveform):
    """Two-pass gated K-weighted power in dB, computed once on the mono mix."""
    weighted = signal.sosfilt(_k_sections(), np.asarray(waveform))
    return _weighted_reference_db(weighted)


def _weighted_reference_db(weighted):
    """Consume a private K-weighted waveform to compute gated whole-song power."""
    block, step = 17640, 4410
    power = np.square(weighted, out=weighted)
    starts = np.arange(0, len(power) - block + 1, step)
    if len(starts):
        prefix = np.concatenate(([0.0], np.cumsum(power)))
        blocks = (prefix[starts + block] - prefix[starts]) / block
        audible = blocks[blocks >= np.quantile(blocks, 0.98) * 1e-4]
        level = audible[audible >= 0.1 * audible.mean()].mean()
    else:
        level = power.mean() if len(power) else 0.0
    return float(10 * np.log10(max(level, 1e-18)))


@lru_cache(maxsize=2)
def _base_mel_bank(size):
    window = torch.hann_window(size)
    # Integrate the same physical triangles over native FFT bins. Even
    # narrow low bands on the short FFT retain their physical location.
    oversample = 16
    fine = _mel_filter(SAMPLE_RATE, size * oversample, 128)
    bank = F.avg_pool1d(
        F.pad(fine, (oversample // 2, oversample // 2 - 1))[None], oversample, oversample
    )[0]
    bank = bank / bank.sum(1, keepdim=True).clamp_min(1e-12)
    frequencies = np.arange(size // 2 + 1) * (2 * math.pi / size)
    _, response = signal.sosfreqz(_k_sections(), worN=frequencies)
    bank *= torch.tensor(abs(response) ** 2, dtype=torch.float32) / window.square().sum()
    return bank


@lru_cache(maxsize=2)
def _shifted_mel_banks(size):
    # Integrate only each triangle's support on the same 16x FFT grid as
    # the original bank. Normalize against its full, untruncated area.
    df = SAMPLE_RATE / size
    oversample = 16
    fine_df = df / oversample
    nyquist = SAMPLE_RATE / 2
    edges = 700 * np.expm1(np.linspace(np.log1p(30 / 700), np.log1p(nyquist / 700), 130))
    bins = np.arange(size // 2 + 1)
    banks = np.zeros((25, 128, len(bins)), dtype=np.float32)
    window_power = torch.hann_window(size).square().sum().item()
    for index in range(25):
        if index == 12:
            continue  # Insert the original bank bit for bit below.
        alpha = 2 ** (-(index - 12) / 24)
        freq = np.minimum(bins * df / alpha, nyquist)
        _, response = signal.sosfreqz(_k_sections(), worN=2 * np.pi * freq / SAMPLE_RATE)
        shifted = edges * alpha / fine_df
        for row, (a, b, c) in enumerate(zip(shifted, shifted[1:], shifted[2:])):
            fine = np.arange(math.ceil(a), min(math.floor(c), size * oversample // 2) + 1)
            native = (fine + oversample // 2) // oversample
            triangle = np.minimum((fine - a) / (b - a), (c - fine) / (c - b))
            banks[index, row] = np.bincount(native, triangle, minlength=len(bins)) * (2 / (c - a))
        banks[index] *= np.abs(response) ** 2 / window_power
    shifted = torch.from_numpy(banks)
    base = _base_mel_bank(size)
    base_sum = base.sum(1, keepdim=True).clamp_min(1e-20)
    shifted[12].copy_(base)
    for index in range(12):
        cur = shifted[index]
        alpha = 2 ** ((12-index)/24)
        cut = torch.from_numpy(edges[2:] * alpha > nyquist)[:, None]
        support = cur.sum(1, keepdim=True) / base_sum
        # Normalize partial triangles and continuously revert to the calibrated
        # native row over the final 20% of available support.
        repaired = torch.lerp(base, cur / support.clamp_min(1e-12), (support / .2).clamp(0, 1))
        cur.copy_(torch.where(cut, repaired, cur))
    return shifted


def segment_frame_slices(segments, frames, frame_rate):
    """Half-open STFT frame slices; frame j is centered at j / frame_rate."""
    left = 0
    for i, segment in enumerate(segments):
        right = (min(frames, max(left, math.ceil(segments[i+1][0] * frame_rate - 1e-9)))
                 if i+1 < len(segments) else frames)
        if right > left:
            yield left, right, segment
        left = right


def frontend_segments(speed_segments):
    """Switch analysis banks at the midpoint of the pre-downbeat crossfade."""
    return [(max(0., a - .010), b, rate, pitch) for a, b, rate, pitch in speed_segments]


def _power_eager(spec_ri):
    return spec_ri[..., 0].square() + spec_ri[..., 1].square()


def _post_mel_eager(mel, samples, ref, centers, hop: int):
    db = 10.0 * mel.clamp_min(1e-30).log10() - ref[:, None, None]
    m = (db.clamp(-60.0, 36.0) + 48.0) * (1.0 / 72.0)
    cap = samples[:, None] // hop
    pos = torch.minimum(centers / hop, cap.double())
    left = pos.long()
    right = torch.minimum(left + 1, cap)
    frac = (pos - left).to(m.dtype).unsqueeze(-1)
    idx_l = left.unsqueeze(-1).expand(-1, -1, m.shape[-1])
    idx_r = right.unsqueeze(-1).expand(-1, -1, m.shape[-1])
    val = torch.lerp(m.gather(1, idx_l), m.gather(1, idx_r), frac)
    delta = 16.0 * F.pad(val[:, 1:] - val[:, :-1], (0, 0, 1, 0))
    return val, delta


@lru_cache(maxsize=1)
def _compiled_mel_kernels():
    try:
        from torch.utils._triton import has_triton
    except ImportError:
        return None
    if not has_triton():
        return None
    return (
        torch.compile(_power_eager, dynamic=True),
        torch.compile(_post_mel_eager, dynamic=True),
    )


class DualWindowMelFrontend(nn.Module):
    def __init__(self, pitch_banks=True):
        super().__init__()
        self.register_buffer("frequency", torch.linspace(-1, 1, 128), persistent=False)
        for size in (2048, 512):
            window = torch.hann_window(size)
            bank = _base_mel_bank(size).clone()
            self.register_buffer(f"window_{size}", window, persistent=False)
            self.register_buffer(f"bank_{size}", bank, persistent=False)

            if pitch_banks:
                shifted = _shifted_mel_banks(size).clone()
                self.register_buffer(f"shifted_bank_{size}", shifted, persistent=False)

    def _window(self, wave, samples, ref, centers, size, hop, pitch_indices=None, pitch_segments=None):
        kernels = _compiled_mel_kernels() if wave.device.type == "cuda" else None
        power_fn, post_fn = kernels[:2] if kernels is not None else (_power_eager, _post_mel_eager)
        spectrum = torch.stft(
            wave,
            size,
            hop,
            window=getattr(self, f"window_{size}"),
            return_complex=True,
            pad_mode="constant",
        )
        # torch.stft returns [B, K, T] transposed from contiguous [B, T, K].
        power = power_fn(torch.view_as_real(spectrum.transpose(1, 2)))
        if pitch_segments is not None:
            bank = getattr(self, f"shifted_bank_{size}")
            mel = power.new_empty((*power.shape[:2], bank.shape[1]))
            for i, segments in enumerate(pitch_segments):
                if not segments:
                    idx = 12 if pitch_indices is None else pitch_indices[i]
                    mel[i] = power[i] @ bank[idx].T
                    continue
                for a, b, (_, _, pitch) in segment_frame_slices(segments, power.shape[1], SAMPLE_RATE / hop):
                    idx = max(0, min(24, 12 + round(pitch * 2)))
                    mel[i, a:b] = power[i, a:b] @ bank[idx].T
        elif pitch_indices is None:
            mel = power @ getattr(self, f"bank_{size}").transpose(0, 1)
        else:
            banks = getattr(self, f"shifted_bank_{size}")[pitch_indices]
            mel = torch.bmm(power, banks.transpose(1, 2))
        return post_fn(mel, samples, ref, centers, hop)

    @torch.no_grad()
    def batch(self, waveforms, sample_lengths, song_ref_db, pitch_semitones=None, pitch_segments=None):
        wave = waveforms.reshape(-1, waveforms.shape[-1]).float()
        samples = torch.as_tensor(sample_lengths, dtype=torch.long, device=wave.device).reshape(-1)
        ref = torch.as_tensor(song_ref_db, dtype=torch.float32, device=wave.device).reshape(-1)
        lengths = ((samples * FRAME_RATE + SAMPLE_RATE - 1) // SAMPLE_RATE).clamp_min(1)
        count = max(1, math.ceil(wave.shape[-1] * FRAME_RATE / SAMPLE_RATE))
        frames = torch.arange(count, device=wave.device)
        centers = (frames.double() + 0.5) * (SAMPLE_RATE / FRAME_RATE)
        wave = wave.masked_fill(torch.arange(wave.shape[-1], device=wave.device) >= samples[:, None], 0)
        pitch_segments = pitch_segments or None
        if pitch_segments and not any(len(row) > 1 for row in pitch_segments):
            uniform = [row[0][-1] if row else 0. for row in pitch_segments]
            if pitch_semitones is None:
                pitch_semitones = torch.tensor(uniform)
            else:
                pitch_semitones = torch.where(torch.tensor([bool(row) for row in pitch_segments],
                                                           device=pitch_semitones.device),
                                              pitch_semitones.new_tensor(uniform), pitch_semitones)
            pitch_segments = None
        indices = None
        if pitch_semitones is not None and pitch_semitones.any():
            indices = (pitch_semitones.to(wave.device) * 2).round().long().clamp(-12, 12) + 12
        with torch.autocast(device_type=wave.device.type, enabled=False):
            long, dl = self._window(wave, samples, ref, centers, 2048, 353, indices, pitch_segments)
            short, ds = self._window(wave, samples, ref, centers, 512, 176, indices, pitch_segments)
        features = torch.stack((long, dl, short, ds, self.frequency.expand_as(long)), 1)
        return features.masked_fill(frames[None, None, :, None] >= lengths[:, None, None, None], 0), lengths
