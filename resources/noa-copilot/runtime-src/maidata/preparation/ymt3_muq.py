"""YMT3+ (8ms @ 512d) and MuQ Layer 10 (40ms @ 1024d) windowed feature extraction and continuous domain."""

import json
import math
from math import gcd
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from maidata.audio.types import ContinuousSongInputs
from maidata.ymt3_muq_contract import (
    FRAME_SECONDS_8MS,
    FRAME_SECONDS_40MS,
    MUQ_DIM,
    RATIO_40_TO_8,
    REPRESENTATION,
    YMT3_DIM,
)
from .compactspec import bf16_bits, from_bits, read_stereo
from .timing import TempoMap

RECIPE = dict(
    representation=REPRESENTATION,
    ymt3=dict(
        model="T5-YourMT3-Encoder-8L",
        sample_rate=16000,
        n_fft=2048,
        hop_length=128,
        n_mels=512,
        frame_ms=8.0,
        dim=YMT3_DIM,
        window_samples=32768,
        window_frames=256,
        stride_samples=16384,
        stride_frames=128,
        trim_frames=32,
        crossfade_frames=64,
    ),
    muq=dict(
        model="OpenMuQ/MuQ-large-msd-iter",
        layer=10,
        sample_rate=24000,
        frame_ms=40.0,
        dim=MUQ_DIM,
        window_samples=720000,
        window_frames=750,
        stride_samples=576000,
        stride_frames=600,
        trim_frames=50,
        crossfade_frames=50,
    ),
    storage="bf16-rne-uint16-bits",
)


def resample_mono(y: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, 0] if arr.shape[1] == 1 else (arr[:, 0] + arr[:, 1]) * 0.5
    if sr == target_sr:
        return arr.astype(np.float32, copy=False)
    from scipy.signal import resample_poly

    common = gcd(int(sr), int(target_sr))
    count = max(1, round(len(arr) * target_sr / sr))
    resampled = resample_poly(
        arr,
        target_sr // common,
        sr // common,
        axis=0,
        window=("kaiser", 5.0),
        padtype="constant",
        cval=0,
    )[:count]
    return resampled.astype(np.float32)


def _htk_mel_filterbank(sample_rate: int, n_fft: int, n_mels: int, f_min: float, f_max: float) -> Tensor:
    """Match torchaudio.transforms.MelSpectrogram default (mel_scale='htk', norm=None)."""
    def hz_to_mel(freq):
        return 2595.0 * np.log10(1.0 + np.asarray(freq, dtype=np.float64) / 700.0)

    def mel_to_hz(mels):
        return 700.0 * (10.0 ** (np.asarray(mels, dtype=np.float64) / 2595.0) - 1.0)

    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1, dtype=np.float64)
    mel_pts = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2, dtype=np.float64)
    hz_pts = mel_to_hz(mel_pts)
    fdiff = np.diff(hz_pts)
    ramps = np.subtract.outer(hz_pts, fft_freqs)
    weights = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper)).astype(np.float32)
    return torch.from_numpy(weights.T.copy())


class YMT3MelStem(nn.Module):
    """Standalone PyTorch Mel spectrogram frontend matching YourMT3+ Melspectrogram (16kHz -> 8ms @ 512d)."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 128,
        n_mels: int = 512,
        f_min: float = 50.0,
        f_max: float = 8000.0,
        eps: float = 1e-5,
        norm_eps: float = 0.008,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.eps = eps
        self.norm_eps = norm_eps
        self.register_buffer("window", torch.hann_window(n_fft, periodic=True), persistent=False)
        self.register_buffer(
            "mel_basis",
            _htk_mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max),
            persistent=False,
        )

    def forward(self, waveform: Tensor) -> Tensor:
        if waveform.ndim == 1:
            waveform = waveform[None, :]
        elif waveform.ndim == 3 and waveform.shape[1] == 1:
            waveform = waveform[:, 0, :]
        # Official YourMT3 slices 32767 samples so center=True STFT with hop=128 yields 256 frames.
        if waveform.shape[-1] == 32768:
            waveform = waveform[:, :32767]
        spec = torch.stft(
            waveform.float(),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window.float(),
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        power_spec = spec.abs().pow(2.0).transpose(1, 2)  # [B, T_frames, n_fft // 2 + 1]
        mel = torch.matmul(power_spec, self.mel_basis.float())
        log_mel = torch.log(mel + self.eps)
        lo = log_mel.amin(dim=(-2, -1), keepdim=True)
        hi = log_mel.amax(dim=(-2, -1), keepdim=True)
        return torch.nan_to_num((log_mel - lo) / (hi - lo + self.norm_eps))


class _T5LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.variance_epsilon = eps

    def forward(self, x: Tensor) -> Tensor:
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(self.weight.dtype)


class _T5Attention(nn.Module):
    def __init__(self, d_model: int = 512, num_heads: int = 6, d_kv: int = 64) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = num_heads
        self.key_value_proj_dim = d_kv
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.q = nn.Linear(d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, d_model, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        b, t, _ = hidden_states.shape
        q = self.q(hidden_states).view(b, t, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        k = self.k(hidden_states).view(b, t, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        v = self.v(hidden_states).view(b, t, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1))
        attn = F.softmax(scores.float(), dim=-1).to(q.dtype)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, t, self.inner_dim)
        return self.o(out)


class _T5LayerSelfAttention(nn.Module):
    def __init__(self, d_model: int = 512, num_heads: int = 6, d_kv: int = 64, eps: float = 1e-6) -> None:
        super().__init__()
        self.SelfAttention = _T5Attention(d_model, num_heads, d_kv)
        self.layer_norm = _T5LayerNorm(d_model, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.SelfAttention(self.layer_norm(x))


class _T5GatedMLP(nn.Module):
    def __init__(self, d_model: int = 512, d_ff: int = 1024) -> None:
        super().__init__()
        self.wi_0 = nn.Linear(d_model, d_ff, bias=False)
        self.wi_1 = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.wo(F.gelu(self.wi_0(x), approximate="tanh") * self.wi_1(x))


class _T5LayerFF(nn.Module):
    def __init__(self, d_model: int = 512, d_ff: int = 1024, eps: float = 1e-6) -> None:
        super().__init__()
        self.DenseReluDense = _T5GatedMLP(d_model, d_ff)
        self.layer_norm = _T5LayerNorm(d_model, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.DenseReluDense(self.layer_norm(x))


class _T5Block(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 6,
        d_kv: int = 64,
        d_ff: int = 1024,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.layer = nn.ModuleList(
            [
                _T5LayerSelfAttention(d_model, num_heads, d_kv, eps=eps),
                _T5LayerFF(d_model, d_ff, eps=eps),
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.layer[0](x)
        x = self.layer[1](x)
        return x


class _FixedSinusoidalPositionalEmbedding(nn.Embedding):
    """Match YourMT3 FixedSinusoidalPositionalEmbedding (first half sin, second half cos)."""

    def __init__(self, num_positions: int, embedding_dim: int) -> None:
        super().__init__(num_positions, embedding_dim)
        self._init_sinusoidal_weight(self.weight)

    @staticmethod
    def _init_sinusoidal_weight(out: nn.Parameter) -> None:
        n_pos, dim = out.shape
        pos = np.arange(n_pos, dtype=np.float64)[:, None]
        div = np.power(10000.0, 2.0 * (np.arange(dim, dtype=np.float64) // 2) / dim)[None, :]
        angles = pos / div
        out.requires_grad_(False)
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        with torch.no_grad():
            out[:, :sentinel] = torch.from_numpy(np.sin(angles[:, 0::2]).astype(np.float32))
            out[:, sentinel:] = torch.from_numpy(np.cos(angles[:, 1::2]).astype(np.float32))

    @torch.no_grad()
    def forward(self, seq_len: int) -> Tensor:
        if seq_len > self.weight.shape[0]:
            temp = nn.Parameter(torch.empty((seq_len, self.weight.shape[1]), dtype=torch.float32), requires_grad=False)
            self._init_sinusoidal_weight(temp)
            return temp.to(device=self.weight.device, dtype=self.weight.dtype)[None, :, :]
        positions = torch.arange(seq_len, dtype=torch.long, device=self.weight.device)
        return super().forward(positions)[None, :, :]


class YMT3T5Encoder(nn.Module):
    """Standalone PyTorch T5 YourMT3+ 8-layer encoder (16kHz waveform -> [B, 256, 512]).

    Retains exact official T5EncoderYMT3 parameter names and shapes (d_model=512,
    num_heads=6, d_kv=64 -> inner_dim=384, d_ff=1024, num_layers=8).
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 6,
        d_kv: int = 64,
        num_layers: int = 8,
        d_ff: int = 1024,
        max_length: int = 1036,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.mel_stem = YMT3MelStem(n_mels=d_model)
        self.embed_positions = _FixedSinusoidalPositionalEmbedding(max_length, d_model)
        self.block = nn.ModuleList(_T5Block(d_model, num_heads, d_kv, d_ff) for _ in range(num_layers))
        self.final_layer_norm = _T5LayerNorm(d_model)

    @property
    def blocks(self) -> nn.ModuleList:
        return self.block

    def forward(self, waveform_16k: Tensor) -> Tensor:
        x = self.mel_stem(waveform_16k)
        pe = self.embed_positions(x.shape[1]).to(x.dtype)
        x = x + pe
        for blk in self.block:
            x = blk(x)
        return self.final_layer_norm(x)


def extract_ymt3_windowed(
    waveform_16k: np.ndarray | Tensor,
    model,
    duration_seconds: float | None = None,
    *,
    batch_size: int = 16,
    device: str | torch.device | None = None,
) -> Tensor:
    """Extract 8ms @ 512d YMT3+ features using 2.048s windows, 1.024s stride, 32-frame trim + 64-frame crossfade."""
    wave = torch.as_tensor(waveform_16k, dtype=torch.float32).flatten()
    if duration_seconds is None:
        duration_seconds = float(wave.numel()) / 16000.0
    n_8ms = max(1, math.ceil(float(duration_seconds) / FRAME_SECONDS_8MS))

    win_samples = 32768
    stride_samples = 16384
    win_frames = 256
    stride_frames = 128

    if n_8ms <= win_frames:
        num_windows = 1
    else:
        num_windows = 1 + math.ceil((n_8ms - win_frames) / stride_frames)

    needed_samples = (num_windows - 1) * stride_samples + win_samples
    if wave.numel() < needed_samples:
        wave = F.pad(wave, (0, needed_samples - wave.numel()))

    windows = wave.unfold(0, win_samples, stride_samples)[:num_windows]
    if device is not None:
        windows = windows.to(device)

    outputs = []
    with torch.no_grad():
        for start in range(0, num_windows, max(1, batch_size)):
            batch = windows[start : start + max(1, batch_size)]
            out = model(batch)
            if isinstance(out, dict):
                out = out.get("last_hidden_state", out.get("hidden_states"))
            if out.shape[1] > win_frames:
                out = out[:, :win_frames, :]
            elif out.shape[1] < win_frames:
                out = F.pad(out, (0, 0, 0, win_frames - out.shape[1]))
            outputs.append(out.detach().cpu().to(torch.float32))
    win_feats = torch.cat(outputs, dim=0)  # [W, 256, 512]

    if num_windows == 1:
        return win_feats[0, :n_8ms].contiguous()

    total_frames = (num_windows - 1) * stride_frames + win_frames
    stitched = torch.empty((total_frames, win_feats.shape[-1]), dtype=torch.float32)
    alpha = (torch.arange(64, dtype=torch.float32)[:, None] + 0.5) / 64.0

    # Window 0: untrimmed left start + left overlap + center [0..159]
    stitched[0:160] = win_feats[0, 0:160]
    for w in range(num_windows - 1):
        start_cf = w * stride_frames + 160
        end_cf = start_cf + 64
        # Trim interior 32 frames (w's 224..255 and (w+1)'s 0..31) and crossfade 64 frames:
        stitched[start_cf:end_cf] = (1.0 - alpha) * win_feats[w, 160:224] + alpha * win_feats[w + 1, 32:96]
        # Center of window w+1 [96..159]:
        center_start = (w + 1) * stride_frames + 96
        stitched[center_start : center_start + 64] = win_feats[w + 1, 96:160]

    # Final window right tail [160..255]
    last_tail_start = (num_windows - 1) * stride_frames + 160
    stitched[last_tail_start : last_tail_start + 96] = win_feats[num_windows - 1, 160:256]
    return stitched[:n_8ms].contiguous()


def extract_muq_windowed(
    waveform_24k: np.ndarray | Tensor,
    model,
    n_40ms: int | None = None,
    *,
    duration_seconds: float | None = None,
    layer: int = 10,
    batch_size: int = 4,
    device: str | torch.device | None = None,
) -> Tensor:
    """Extract native 40ms @ 1024d MuQ Layer 10 features using 30s windows, 24s stride, 50-frame trim + 50-frame crossfade."""
    wave = torch.as_tensor(waveform_24k, dtype=torch.float32).flatten()
    if n_40ms is None:
        if duration_seconds is None:
            duration_seconds = float(wave.numel()) / 24000.0
        n_8ms = max(1, math.ceil(float(duration_seconds) / FRAME_SECONDS_8MS))
        n_40ms = max(1, math.ceil(n_8ms / RATIO_40_TO_8))

    win_samples = 720000
    stride_samples = 576000
    win_frames = 750
    stride_frames = 600

    if n_40ms <= win_frames:
        num_windows = 1
    else:
        num_windows = 1 + math.ceil((n_40ms - win_frames) / stride_frames)

    needed_samples = (num_windows - 1) * stride_samples + win_samples
    if wave.numel() < needed_samples:
        wave = F.pad(wave, (0, needed_samples - wave.numel()))

    windows = wave.unfold(0, win_samples, stride_samples)[:num_windows]
    if device is not None:
        windows = windows.to(device)

    outputs = []
    with torch.no_grad():
        for start in range(0, num_windows, max(1, batch_size)):
            batch = windows[start : start + max(1, batch_size)]
            out = model(batch)
            if hasattr(out, "hidden_states") and out.hidden_states is not None:
                out = out.hidden_states[layer]
            elif isinstance(out, dict) and "hidden_states" in out:
                out = out["hidden_states"][layer]
            if out.shape[1] > win_frames:
                out = out[:, :win_frames, :]
            elif out.shape[1] < win_frames:
                out = F.pad(out, (0, 0, 0, win_frames - out.shape[1]))
            outputs.append(out.detach().cpu().to(torch.float32))
    win_feats = torch.cat(outputs, dim=0)  # [W, 750, 1024]

    if num_windows == 1:
        return win_feats[0, :n_40ms].contiguous()

    total_frames = (num_windows - 1) * stride_frames + win_frames
    stitched = torch.empty((total_frames, win_feats.shape[-1]), dtype=torch.float32)
    alpha = (torch.arange(50, dtype=torch.float32)[:, None] + 0.5) / 50.0

    # Window 0: untrimmed left start + left overlap + center [0..649]
    stitched[0:650] = win_feats[0, 0:650]
    for w in range(num_windows - 1):
        start_cf = w * stride_frames + 650
        end_cf = start_cf + 50
        # Trim interior 50 frames (w's 700..749 and (w+1)'s 0..49) and crossfade 50 frames:
        stitched[start_cf:end_cf] = (1.0 - alpha) * win_feats[w, 650:700] + alpha * win_feats[w + 1, 50:100]
        # Center of window w+1 [100..649]:
        center_start = (w + 1) * stride_frames + 100
        stitched[center_start : center_start + 550] = win_feats[w + 1, 100:650]

    # Final window right tail [650..749]
    last_tail_start = (num_windows - 1) * stride_frames + 650
    stitched[last_tail_start : last_tail_start + 100] = win_feats[num_windows - 1, 650:750]
    return stitched[:n_40ms].contiguous()


def extract_grid(
    pcm_or_path,
    timing: dict,
    duration: float,
    *,
    ymt3_model,
    muq_model,
    silence_ymt3: np.ndarray | None = None,
    silence_muq: np.ndarray | None = None,
    first_beat: int | None = None,
    sr: int = 24000,
    device: str | torch.device | None = None,
) -> dict:
    if isinstance(pcm_or_path, (str, Path)):
        y, sr = read_stereo(pcm_or_path)
        duration = len(y) / sr
    else:
        y = np.asarray(pcm_or_path, dtype=np.float32)

    wave_16k = resample_mono(y, sr, 16000)
    wave_24k = resample_mono(y, sr, 24000)

    ymt3_feat = extract_ymt3_windowed(wave_16k, ymt3_model, duration_seconds=duration, device=device)
    n_8ms = ymt3_feat.shape[0]
    n_40ms = max(1, math.ceil(n_8ms / RATIO_40_TO_8))
    muq_feat = extract_muq_windowed(wave_24k, muq_model, n_40ms=n_40ms, duration_seconds=duration, device=device)

    if silence_ymt3 is None:
        silence_ymt3 = np.zeros((YMT3_DIM,), dtype=np.float32)
    else:
        silence_ymt3 = np.asarray(silence_ymt3, dtype=np.float32)
    if silence_muq is None:
        silence_muq = np.zeros((MUQ_DIM,), dtype=np.float32)
    else:
        silence_muq = np.asarray(silence_muq, dtype=np.float32)

    if first_beat is None:
        tempo = TempoMap(timing)
        first_beat = min(0, math.floor(float(tempo.to_beats(0.0))))

    return dict(
        ymt3_frames=bf16_bits(ymt3_feat.numpy()),
        muq_frames=bf16_bits(muq_feat.numpy()),
        silence_ymt3=silence_ymt3,
        silence_muq=silence_muq,
        first_beat=np.array(int(first_beat), dtype=np.int64),
        duration_seconds=np.array(float(duration), dtype=np.float64),
        timing=timing,
    )


def validate_grid(grid: dict) -> None:
    if "ymt3_frames" not in grid or "muq_frames" not in grid:
        raise ValueError("YMT3+MuQ grid must contain ymt3_frames and muq_frames")
    ymt3 = grid["ymt3_frames"]
    muq = grid["muq_frames"]
    if ymt3.ndim != 2 or ymt3.shape[1] != YMT3_DIM or ymt3.dtype != np.uint16 or ymt3.shape[0] < 1:
        raise ValueError(f"ymt3_frames must be uint16 BF16 bits [N_8ms, {YMT3_DIM}]")
    expected_40ms = math.ceil(ymt3.shape[0] / RATIO_40_TO_8)
    if muq.ndim != 2 or muq.shape != (expected_40ms, MUQ_DIM) or muq.dtype != np.uint16:
        raise ValueError(f"muq_frames must be uint16 BF16 bits [{expected_40ms}, {MUQ_DIM}]")
    if not torch.isfinite(from_bits(ymt3)).all() or not torch.isfinite(from_bits(muq)).all():
        raise ValueError("nonfinite YMT3+MuQ grid values")
    for key, dim in (("silence_ymt3", YMT3_DIM), ("silence_muq", MUQ_DIM)):
        if key in grid:
            s = np.asarray(grid[key])
            if s.shape != (dim,) or s.dtype != np.float32 or not np.isfinite(s).all():
                raise ValueError(f"{key} must be finite float32 [{dim}]")


def continuous_inputs(grids: list[dict], coverages: list, config: dict, *, validate_input: bool = False) -> ContinuousSongInputs:
    if not grids or len(grids) != len(coverages):
        raise ValueError("continuous YMT3+MuQ audio requires one coverage per grid")
    from .audio_padding import add_global_bounds, audio_padding

    before, after = audio_padding(config)
    starts, ends = [], []
    k8_bounds, k40_bounds = [], []

    for grid, coverage in zip(grids, coverages, strict=True):
        if validate_input:
            validate_grid(grid)
        tempo = TempoMap(grid["timing"])
        first = int(grid.get("first_beat", 0))
        duration = float(grid["duration_seconds"])
        start_beat = float(min(first, coverage.first_beat - before))
        end_beat = float(
            max(
                math.ceil(float(tempo.to_beats(duration))),
                coverage.first_beat + coverage.draft_beats + after,
            )
        )
        starts.append(start_beat)
        ends.append(end_beat)

        t_start = float(tempo.to_seconds(start_beat))
        t_end = float(tempo.to_seconds(end_beat))
        k40_start = math.floor(t_start / FRAME_SECONDS_40MS)
        k40_end = math.ceil(t_end / FRAME_SECONDS_40MS)
        if k40_end <= k40_start:
            k40_end = k40_start + 1
        k8_start = RATIO_40_TO_8 * k40_start
        k8_end = RATIO_40_TO_8 * k40_end
        k40_bounds.append((k40_start, k40_end))
        k8_bounds.append((k8_start, k8_end))

    lengths_40ms = [b - a for a, b in k40_bounds]
    lengths_8ms = [b - a for a, b in k8_bounds]
    max_40ms = max(lengths_40ms)
    max_8ms = max(lengths_8ms)  # Identically 5 * max_40ms

    batch_size = len(grids)
    ymt3_frames = torch.zeros((batch_size, max_8ms, YMT3_DIM), dtype=torch.bfloat16)
    muq_frames = torch.zeros((batch_size, max_40ms, MUQ_DIM), dtype=torch.bfloat16)
    beats = torch.zeros((batch_size, max_8ms), dtype=torch.float32)
    valid = torch.zeros((batch_size, max_8ms), dtype=torch.bool)
    sample_valid = torch.zeros((batch_size, max_8ms), dtype=torch.bool)
    muq_valid = torch.zeros((batch_size, max_40ms), dtype=torch.bool)
    muq_sample_valid = torch.zeros((batch_size, max_40ms), dtype=torch.bool)

    silence_ymt3_arr = np.asarray(grids[0].get("silence_ymt3", np.zeros(YMT3_DIM, np.float32)), dtype=np.float32)
    silence_muq_arr = np.asarray(grids[0].get("silence_muq", np.zeros(MUQ_DIM, np.float32)), dtype=np.float32)
    silence_ymt3 = torch.from_numpy(silence_ymt3_arr)
    silence_muq = torch.from_numpy(silence_muq_arr)

    for i, (grid, (k8_start, k8_end), (k40_start, k40_end)) in enumerate(zip(grids, k8_bounds, k40_bounds)):
        l8 = k8_end - k8_start
        l40 = k40_end - k40_start
        s_y = torch.from_numpy(np.asarray(grid.get("silence_ymt3", silence_ymt3_arr), dtype=np.float32)).to(torch.bfloat16)
        s_m = torch.from_numpy(np.asarray(grid.get("silence_muq", silence_muq_arr), dtype=np.float32)).to(torch.bfloat16)

        ymt3_frames[i, :l8] = s_y
        muq_frames[i, :l40] = s_m
        valid[i, :l8] = True
        muq_valid[i, :l40] = True

        y_vals = from_bits(grid["ymt3_frames"])
        n8 = y_vals.shape[0]
        src8_lo = max(0, k8_start)
        src8_hi = min(n8, k8_end)
        if src8_hi > src8_lo:
            dst8_lo = src8_lo - k8_start
            dst8_hi = src8_hi - k8_start
            ymt3_frames[i, dst8_lo:dst8_hi] = y_vals[src8_lo:src8_hi]
            sample_valid[i, dst8_lo:dst8_hi] = True

        m_vals = from_bits(grid["muq_frames"])
        n40 = m_vals.shape[0]
        src40_lo = max(0, k40_start)
        src40_hi = min(n40, k40_end)
        if src40_hi > src40_lo:
            dst40_lo = src40_lo - k40_start
            dst40_hi = src40_hi - k40_start
            muq_frames[i, dst40_lo:dst40_hi] = m_vals[src40_lo:src40_hi]
            muq_sample_valid[i, dst40_lo:dst40_hi] = True

        tempo = TempoMap(grid["timing"])
        times_8ms = (k8_start + np.arange(l8, dtype=np.float64)) * FRAME_SECONDS_8MS
        beats_8ms = np.asarray(tempo.to_beats(times_8ms), dtype=np.float32)
        beats[i, :l8] = torch.from_numpy(beats_8ms)

    result = ContinuousSongInputs(
        song=None,
        mel=None,
        aux32=None,
        beats=beats,
        valid=valid,
        domain_start=torch.tensor(starts, dtype=torch.float32),
        domain_end=torch.tensor(ends, dtype=torch.float32),
        ymt3_frames=ymt3_frames,
        muq_frames=muq_frames,
        sample_valid=sample_valid,
        muq_sample_valid=muq_sample_valid,
        muq_valid=muq_valid,
        silence_ymt3=silence_ymt3,
        silence_muq=silence_muq,
    )
    result.layout = dict(
        grids=(beats,),
        grid_valid=(valid,),
        muq_valid=muq_valid,
        k8_start=torch.tensor([a for a, _ in k8_bounds], dtype=torch.long),
        k40_start=torch.tensor([a for a, _ in k40_bounds], dtype=torch.long),
    )
    add_global_bounds(result, coverages, config)
    return result
