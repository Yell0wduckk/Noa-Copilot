"""Amplitude-preserving online spectra with exact 8ms physical centers."""
import math
from functools import lru_cache
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

POWER_FLOOR = 1e-10
LOG_POWER_FLOOR = math.log(POWER_FLOOR)

DEFAULT_MERT_V2_MODEL = "m-a-p/MERT-v2-FullSong"
MERT_V2_ORIGIN_SECONDS = 0.015


def _mel_filter(sample_rate=44100, n_fft=2048, n_mels=256, f_min=30.):
    high = 2595. * math.log10(1. + sample_rate / 1400.)
    low = 2595. * math.log10(1. + f_min / 700.)
    edges = 700 * (10 ** (torch.linspace(low, high, n_mels + 2) / 2595) - 1)
    hz = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1)
    left = (hz[None] - edges[:-2, None]) / (edges[1:-1] - edges[:-2])[:, None]
    right = (edges[2:, None] - hz[None]) / (edges[2:] - edges[1:-1])[:, None]
    return torch.minimum(left, right).clamp_min(0)

@lru_cache(maxsize=16)
def _resampler(source, target, device):
    try:
        from torchaudio.transforms import Resample
        return Resample(source, target, dtype=torch.float32).to(device)
    except (ImportError, OSError):
        from scipy.signal import resample_poly
        g = math.gcd(int(source), int(target))
        up, down = int(target) // g, int(source) // g
        def _scipy_resample(x: torch.Tensor) -> torch.Tensor:
            arr = x.detach().cpu().numpy()
            out = resample_poly(arr, up, down, axis=-1).astype('float32', copy=False)
            return torch.from_numpy(out.copy()).to(device=x.device, dtype=torch.float32)
        return _scipy_resample

@torch.no_grad()
def fast_resample(waveform, source_rate, target_rate=44100):
    waveform = waveform.float()
    if source_rate == target_rate or waveform.shape[-1] == 0:
        return waveform
    if waveform.device.type == 'cpu':
        try:
            import soxr
        except ImportError:
            pass
        else:
            shape = waveform.shape
            value = soxr.resample(waveform.reshape(-1, shape[-1]).T.contiguous().numpy(),
                                  source_rate, target_rate, quality='HQ')
            return torch.from_numpy(value.T.copy()).reshape(*shape[:-1], -1)
    return _resampler(source_rate, target_rate, str(waveform.device))(waveform)

def audio_duration_seconds(path):
    try:
        import soundfile as sf
    except ImportError:
        from scipy.io import wavfile
        rate, data = wavfile.read(path, mmap=True)
        return data.shape[0] / float(rate)
    info = sf.info(path)
    return info.frames / float(info.samplerate)

def _to_mono_tensor(samples):
    wave = torch.from_numpy(samples)
    if wave.ndim == 2:
        channels = wave.shape[1]
        if channels == 1:
            return wave.reshape(-1)
        if channels == 2:
            return wave[:, 0].add(wave[:, 1]).mul_(0.5)
        return wave.mean(1)
    return wave.contiguous()

def read_mono(path, *, start_seconds=0., duration_seconds=None):
    try:
        import soundfile as sf
    except ImportError:
        import numpy as np
        from scipy.io import wavfile
        rate, raw = wavfile.read(path, mmap=isinstance(path, (str, Path)))
        start = max(0, round(start_seconds * rate))
        stop = raw.shape[0] if duration_seconds is None else min(raw.shape[0], start + max(0, round(duration_seconds * rate)))
        chunk = np.asarray(raw[start:stop])
        if chunk.dtype.kind in ('i', 'u'):
            bits = 8 * chunk.dtype.itemsize
            offset = float(1 << (bits - 1)) if chunk.dtype.kind == 'u' else 0.
            scale = float(1 << (bits - 1))
            samples = (chunk.astype('float32') - offset) / scale
        else:
            samples = chunk.astype('float32')
        return _to_mono_tensor(samples), rate
    if (start_seconds > 0. or duration_seconds is not None) and isinstance(path, (str, Path)):
        info = sf.info(path)
        start = max(0, round(start_seconds * info.samplerate))
        frames = None if duration_seconds is None else max(0, round(duration_seconds * info.samplerate))
        samples, rate = sf.read(path, start=start, frames=frames, dtype='float32', always_2d=True)
    else:
        samples, rate = sf.read(path, dtype='float32', always_2d=True)
    return _to_mono_tensor(samples), rate

def read_waveform(path, sample_rate=44100, *, start_seconds=0., duration_seconds=None):
    wave, rate = read_mono(path, start_seconds=start_seconds, duration_seconds=duration_seconds)
    return fast_resample(wave, rate, sample_rate)

def frame_lengths(sample_lengths, speeds=None, frame_seconds=.008):
    samples = torch.as_tensor(sample_lengths).cpu().double()
    speed = 1. if speeds is None else torch.as_tensor(speeds).cpu().double()
    return torch.ceil(samples / (44100 * frame_seconds * speed)).long().clamp_min(1)

def sample_centers(values, source_rate, target_rate, length, *, source_origin=None, target_origin=None):
    """Native-rate interpolation; tail length never changes the physical grid."""
    source_origin = .5 / source_rate if source_origin is None else source_origin
    target_origin = .5 / target_rate if target_origin is None else target_origin
    position = ((torch.arange(length, device=values.device).double() / target_rate +
                 target_origin - source_origin) * source_rate).clamp(0, values.shape[-2] - 1)
    left = position.floor().long()
    right = (left + 1).clamp_max(values.shape[-2] - 1)
    fraction = (position - left).to(values.dtype)
    return values.index_select(-2, left) + (values.index_select(-2, right) -
                                            values.index_select(-2, left)) * fraction[:, None]

def sample_centers_batch(values, lengths, source_rate, target_seconds, source_origins):
    """Interpolate a padded [songs, frames, channels] tensor on each song's own clock."""
    device = values.device
    origins = torch.as_tensor(source_origins, dtype=torch.float64, device=device).reshape(-1, 1)
    lengths = torch.as_tensor(lengths, device=device).reshape(-1, 1)
    position = ((target_seconds - origins) * source_rate).clamp_min(0)
    position = torch.minimum(position, lengths - 1)
    left = position.long()
    right = torch.minimum(left + 1, lengths - 1)
    song = torch.arange(len(values), device=device)[:, None]
    a, b = values[song, left], values[song, right]
    return a + (b - a) * (position - left).to(values.dtype).unsqueeze(-1)


def signed_delta(mel):
    return torch.cat((mel, torch.cat((torch.zeros_like(mel[..., :1, :]),
                                     mel[..., 1:, :] - mel[..., :-1, :]), -2)), -1)

class LogMelFrontend(nn.Module):
    sample_rate, n_fft, n_mels, f_min, feature_dim = 44100, 2048, 256, 30., 512
    # The nearest integer STFT hop is sampled at exact physical 8ms centers;
    # this removes the 0.2-sample/frame drift over long songs.
    hop_length = 353

    def __init__(self, *, persistent=True, frame_hz=125.):
        super().__init__()
        self.frame_hz = frame_hz
        self.register_buffer('window', torch.hann_window(self.n_fft), persistent=False)
        self.register_buffer('mel_filter', _mel_filter(), persistent=persistent)

    def _log_mel(self, waveform):
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            spectrum = torch.stft(waveform.float(), self.n_fft, self.hop_length,
                window=self.window.float(), center=True, pad_mode='constant', return_complex=True)
            power = spectrum.abs().square()
            return (torch.matmul(self.mel_filter.float(), power).clamp_min_(POWER_FLOOR).log_()
                    - LOG_POWER_FLOOR).div_(-LOG_POWER_FLOOR).transpose(-1, -2)

    @torch.no_grad()
    def batch(self, waveforms, sample_lengths, speeds=None):
        samples = torch.as_tensor(sample_lengths).cpu().long()
        rates = torch.ones(len(samples), dtype=torch.float64) if speeds is None else torch.as_tensor(speeds).cpu().double()
        sample_valid = torch.arange(waveforms.shape[-1], device=waveforms.device)[None] < samples.to(waveforms.device)[:, None]
        mel = self._log_mel(waveforms.masked_fill(~sample_valid, 0))
        lengths = frame_lengths(samples, rates, 1/self.frame_hz)
        frames = torch.arange(int(lengths.max()), device=mel.device, dtype=torch.float64)
        centers = ((frames + .5) / self.frame_hz) * rates.to(mel.device)[:, None]
        aligned = sample_centers_batch(mel, samples // self.hop_length + 1,
                                       44100. / self.hop_length, centers, 0.)
        valid = frames < lengths.to(mel.device)[:, None]
        features = signed_delta(aligned).masked_fill(~valid[..., None], 0)
        return features, lengths

    @torch.no_grad()
    def forward(self, waveform):
        single = waveform.ndim == 1
        if single:
            waveform = waveform[None]
        features, _ = self.batch(waveform, [waveform.shape[-1]] * len(waveform))
        return features[0] if single else features

    @torch.no_grad()
    def from_file(self, path):
        return self(read_waveform(path).to(self.mel_filter.device))

def stretch_features(features, rate):
    mel = sample_centers(features[..., :256], 125., 125. / rate,
                         max(1, math.ceil(features.shape[-2] / rate)))
    return signed_delta(mel)

def unpack_feature_payload(payload, schema=None):
    return (payload['features'], payload.get('audio_schema',schema)) if isinstance(payload,dict) else (payload,schema)

def adapt_legacy_features(features, schema=None):
    # No legacy main-model feature conversion is performed.
    return features
