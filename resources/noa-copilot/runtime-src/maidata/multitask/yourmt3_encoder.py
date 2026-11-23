"""YourMT3+ T5: native 8-layer pure Transformer, 8ms / 125Hz / 512d.

Tensor names match the official T5EncoderYMT3 checkpoint. The lower prefix
runs once under no_grad; the selected top layers and final norm fine-tune.
"""
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from torch import Tensor,nn
from torch.nn import functional as F

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
        # T5 deliberately has no 1/sqrt(d_k) score scaling.
        out = F.scaled_dot_product_attention(q,k,v,dropout_p=0.,scale=1.)
        out = out.transpose(1,2).contiguous().view(b,t,self.inner_dim)
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


class TrainableYMT3T5Encoder(YMT3T5Encoder):
    frame_hop = 128
    sample_rate = 16000

    def __init__(self, unfreeze_layers=2):
        super().__init__()
        self.unfreeze_layers = max(0,min(len(self.block),int(unfreeze_layers)))
        self.freeze_top_block = True
        self.set_top_trainable(False)

    @property
    def frozen_prefix(self):
        return len(self.block)-self.unfreeze_layers

    def top_parameters(self):
        if not self.unfreeze_layers:
            return iter(())
        return (p for module in (*self.block[self.frozen_prefix:],self.final_layer_norm)
                for p in module.parameters())

    def set_top_trainable(self, enabled):
        self.requires_grad_(False)
        self.freeze_top_block = not enabled
        for parameter in self.top_parameters():
            parameter.requires_grad_(enabled)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        self.mel_stem.eval()
        self.embed_positions.eval()
        for block in self.block[:self.frozen_prefix]:
            block.eval()
        for block in self.block[self.frozen_prefix:]:
            block.train(mode and not self.freeze_top_block)
        self.final_layer_norm.train(mode and not self.freeze_top_block)
        return self

    @torch.no_grad()
    def forward_frozen_prefix(self,waveform):
        x=self.mel_stem(waveform)
        x=x+self.embed_positions(x.shape[1]).to(x.dtype)
        for block in self.block[:self.frozen_prefix]:
            x=block(x)
        return x

    def forward(self,waveform):
        x=self.forward_frozen_prefix(waveform)
        with torch.no_grad() if self.freeze_top_block else nullcontext():
            for block in self.block[self.frozen_prefix:]:
                x=block(x)
            return self.final_layer_norm(x)


def read_yourmt3_checkpoint(path):
    return torch.load(path,map_location='cpu',weights_only=False,mmap=True)


def load_yourmt3_encoder_weights(module,path):
    payload=read_yourmt3_checkpoint(path)
    state=(payload['frozen_frontends']['ymt3_state'] if 'frozen_frontends' in payload else
           payload.get('state_dict',payload.get('model_state',payload)))
    selected={}
    for key,value in state.items():
        if key.startswith('encoder.encoder.'):
            key=key[len('encoder.encoder.'):]
        elif key.startswith('encoder.'):
            key=key[len('encoder.'):]
        if key=='additive_pe.weight':
            key='embed_positions.weight'
        if key.startswith(('block.','final_layer_norm.','embed_positions.')):
            selected[key]=value
    if 'embed_positions.weight' in selected:
        # Official releases store 1034 or 1036 fixed sinusoidal positions.
        # Only the first 256 are used by a native window. Keep a stable model
        # buffer shape for saved training checkpoints and retain every used row.
        positions=module.embed_positions.weight.detach().clone()
        count=min(len(positions),len(selected['embed_positions.weight']))
        positions[:count].copy_(selected['embed_positions.weight'][:count])
        selected['embed_positions.weight']=positions
    else:
        selected['embed_positions.weight']=module.embed_positions.weight.detach()
    module.load_state_dict(selected,strict=True)
    return dict(checkpoint=str(path),tensors=len(selected),encoder='YourMT3+ T5 8L / 125Hz')
