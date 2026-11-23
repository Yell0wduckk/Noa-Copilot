"""Flat320: physical Mel/L8 song features and the shared 24-step beat encoder."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, partial
import json
import math
from pathlib import Path
import random
import sys
import tempfile
from typing import NamedTuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from maidata.multitask.audio import (
    DEFAULT_MERT_V2_MODEL, MERT_V2_ORIGIN_SECONDS, fast_resample, read_waveform, sample_centers_batch,
)
from maidata.multitask.mel_frontend import DualWindowMelFrontend, song_reference_db
from maidata.spine import official
from . import DRAFT_ARCHITECTURES
from maidata.spine.features import pad_audio
from . import AUDIO_PACK_SCHEMA
from maidata.multitask.attention import rotary

GRID_STEPS_PER_BEAT = 24

FLAT_RATES = {"mel": 125, "mert": 25}
STREAMS = ("conv", "mert")
AUDIO_MODELS = (DEFAULT_MERT_V2_MODEL + "#L8", "DualWindowMel-125Hz")


class DualBank(NamedTuple):
    mel_feat: torch.Tensor
    mel_lengths: torch.Tensor
    mert_feat: torch.Tensor
    mert_lengths: torch.Tensor
    envelope: torch.Tensor | None = None


class DampedRMSNorm(nn.Module):
    def __init__(self, dim, floor=0.25):
        super().__init__()
        self.floor_sq = floor**2
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        scale = (x.float().square().mean(-1, keepdim=True) + self.floor_sq).rsqrt()
        return x * scale.to(x.dtype) * self.weight.to(x.dtype)


class MertTemporalAdapter(nn.Module):
    """25 Hz residual convolution on raw projected L8 features."""

    def __init__(self, dim):
        super().__init__()
        self.depthwise = nn.Conv1d(dim, dim, 5, padding=2, groups=dim, bias=False)
        self.pointwise = nn.Conv1d(dim, dim, 1, bias=False)
        nn.init.zeros_(self.pointwise.weight)

    def forward(self, x, padding):
        residual = self.pointwise(F.gelu(self.depthwise(x)))
        return (x + residual).masked_fill(padding, 0)


class FlatTemporalBlock(nn.Module):
    """125 Hz depthwise-separable residual convolution in NCT layout."""

    def __init__(self, dim: int, kernel: int = 7, dilation: int = 1):
        super().__init__()
        self.depthwise = nn.Conv1d(
            dim,
            dim,
            kernel,
            padding=dilation * (kernel // 2),
            dilation=dilation,
            groups=dim,
            bias=False,
        )
        self.pointwise = nn.Conv1d(dim, dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pointwise(F.gelu(self.depthwise(x)))


class FlatMelStem(nn.Module):
    """Encode each 125 Hz Mel frame with spectral convolutions and a sub-band shortcut."""

    def __init__(
        self,
        dim: int = 256,
        widths: tuple[int, ...] = (24, 48, 64),
        *,
        spec_augment_bands=12,
        mel_gain_jitter=0.08,
    ):
        super().__init__()
        self.spec_augment_bands = spec_augment_bands
        self.mel_gain_jitter = mel_gain_jitter
        self.spectral = nn.ModuleList(
            nn.Conv2d(a, b, (3, 5), stride=(1, 2), padding=(1, 2), bias=False)
            for a, b in zip((5, *widths), widths)
        )
        for conv in self.spectral:
            nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
        # Start with acoustic channels only; the frequency coordinate can learn.
        nn.init.zeros_(self.spectral[0].weight[:, 4])
        self.band_width = 2 ** len(widths)
        bands = 128 // self.band_width
        self.project = nn.Linear(widths[-1] * bands, dim, bias=False)
        self.shortcut = nn.Linear(4 * bands, dim, bias=False)
        self.temporal = nn.Sequential(FlatTemporalBlock(dim, kernel=5, dilation=1))
        self._sync_memory_format()

    @staticmethod
    def _memory_format_for(device):
        # Windows CPU oneDNN requires contiguous weights and activations for backward.
        if sys.platform == "win32" and device.type == "cpu":
            return torch.contiguous_format
        return torch.channels_last

    def _sync_memory_format(self):
        fmt = self._memory_format_for(self.spectral[0].weight.device)
        for conv in self.spectral:
            conv.weight.data = conv.weight.data.contiguous(memory_format=fmt)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        self._sync_memory_format()
        return self

    def forward_song(
        self, mel_song: torch.Tensor, frequency: torch.Tensor
    ) -> torch.Tensor:
        """Map an unpadded [1, frames, 512] song to [1, frames, dim]."""
        if not torch.is_autocast_enabled(mel_song.device.type):
            mel_song = mel_song.to(self.project.weight.dtype)
        frames = mel_song.shape[1]
        if frames == 0:
            return mel_song.new_empty((1, 0, self.project.out_features))
        channels = mel_song.unflatten(-1, (4, 128)).transpose(1, 2)
        if self.training:
            if self.mel_gain_jitter:
                channels = channels * channels.new_empty(()).uniform_(
                    1 - self.mel_gain_jitter, 1 + self.mel_gain_jitter
                )
            if self.spec_augment_bands:
                width = torch.randint(
                    min(128, self.spec_augment_bands) + 1, (), device=channels.device
                )
                start = (torch.rand((), device=channels.device) * (129 - width)).long()
                bands = torch.arange(128, device=channels.device)
                channels = channels.masked_fill(
                    (bands >= start) & (bands < start + width), 0
                )
        # Frequency is a coordinate of present sound; silence has no static carrier.
        freq = frequency.to(channels.dtype) * channels.abs().mean(1, keepdim=True)
        x = torch.cat((channels, freq), dim=1).contiguous(
            memory_format=self._memory_format_for(channels.device)
        )
        for conv in self.spectral:
            x = F.gelu(conv(x))
        h = self.project(x.transpose(1, 2).flatten(2))
        pooled = F.avg_pool2d(channels, (1, self.band_width)).transpose(1, 2).flatten(2)
        h = (h + self.shortcut(pooled)).transpose(1, 2).contiguous()
        return self.temporal(h).transpose(1, 2).contiguous()


class DualAudioFusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.stem = FlatMelStem(
            cfg.dim,
            spec_augment_bands=cfg.spec_augment_bands,
            mel_gain_jitter=cfg.mel_gain_jitter,
        )
        self.register_buffer("frequency", torch.linspace(-1, 1, 128), persistent=False)
        self.mert_proj = nn.Linear(1024, cfg.dim, bias=False)
        nn.init.normal_(
            self.mert_proj.weight, std=0.20 / math.sqrt(self.mert_proj.in_features)
        )
        self.adapters = nn.ModuleList(MertTemporalAdapter(cfg.dim) for _ in range(2))
        self.mert_dropout = cfg.mert_dropout
        self.interleave_envelope = cfg.architecture in DRAFT_ARCHITECTURES
        self.envelope_enabled = cfg.plan4_enabled or self.interleave_envelope

    def forward(self, audio):
        mel, mert = audio["mel"], audio["mert"]
        if mert.dtype == torch.int8:
            dtype = self.mert_proj.weight.dtype
            mert = mert.to(dtype) * audio["mert_scale"][:, None].to(dtype)
        lengths, z_lengths = audio["mel_lengths"], audio["mert_lengths"]
        padding = torch.arange(mert.shape[1], device=mert.device) >= z_lengths[:, None]
        m = self.mert_proj(
            mert.masked_fill(padding[..., None], 0).to(self.mert_proj.weight.dtype)
        )
        m = m.transpose(1, 2)
        for adapter in self.adapters:
            m = adapter(m, padding[:, None])
        if self.training and self.mert_dropout:
            m = m * (torch.rand(len(m), 1, 1, device=m.device) >= self.mert_dropout)
        envelope = None
        if self.envelope_enabled:
            # Native channels are long Mel, long delta x16, short Mel, short delta x16.
            windows = mel.unflatten(-1, (4, 128))
            energy = windows[..., ::2, :].mean(-1, dtype=torch.float32)
            flux = windows[..., 1::2, :].clamp_min(0).mean(-1, dtype=torch.float32)
            envelope = (torch.stack((energy, flux), -1).flatten(-2) if self.interleave_envelope
                        else torch.cat((energy, flux / 16), -1))
        parts = []
        song_lengths = audio.get("mel_lengths_cpu") or lengths.tolist()
        for i, n in enumerate(song_lengths):
            h = self.stem.forward_song(mel[i : i + 1, :n], self.frequency)
            parts.append(F.pad(h, (0, 0, 0, mel.shape[1] - n)))
        return DualBank(torch.cat(parts), lengths, m.transpose(1, 2), z_lengths, envelope)


class LocalBeatRoFormerBlock(nn.Module):
    """Four-beat query tiles, each with a two-beat halo on either side."""

    width, halo, heads = 96, 48, 4

    def __init__(self, dim):
        super().__init__()
        self.attn_norm = DampedRMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.sink = nn.Parameter(torch.zeros(self.heads, 1, dim // self.heads))
        self.ffn_norm = DampedRMSNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 2 * dim, bias=False),
            nn.GELU(),
            nn.Linear(2 * dim, dim, bias=False),
        )

    def forward(self, x, positions, valid):
        w, h, n = self.width, self.halo, len(x)
        pad = (-n) % w
        q, k, v = (
            t.reshape(n, self.heads, -1).transpose(0, 1)[None]
            for t in self.qkv(self.attn_norm(x)).chunk(3, -1)
        )
        q, k = (rotary(t, positions * 2, 64.0)[0] for t in (q, k))
        q = F.pad(q, (0, 0, 0, pad)).unflatten(1, (-1, w)).transpose(0, 1)

        def halo(t, sink):
            tiles = (
                F.pad(t, (0, 0, h, h + pad)).unfold(1, w + 2 * h, w).permute(1, 0, 3, 2)
            )
            return torch.cat(
                (tiles, sink.to(t.dtype).expand(len(tiles), -1, -1, -1)), 2
            )

        keys = halo(k, self.sink)
        vals = halo(v[0], torch.zeros_like(self.sink))
        j = torch.arange(w + 2 * h, device=x.device) - h
        i = torch.arange(w, device=x.device)
        key_valid = F.pad(valid, (h, h + pad)).unfold(0, w + 2 * h, w)
        mask = ((i[:, None] - j).abs() <= h) & key_valid[:, None, :]
        ctx = F.scaled_dot_product_attention(
            q, keys, vals, attn_mask=F.pad(mask, (0, 1), value=True)[:, None]
        )
        ctx = ctx.transpose(1, 2).reshape(-1, x.shape[-1])[:n]
        x = x + self.out(ctx)
        return (x + self.ffn(self.ffn_norm(x))).masked_fill(~valid[:, None], 0)


def audio_rotary(cfg, stream, x, beats):
    return rotary(
        x,
        beats * getattr(cfg, stream + "_rope_scale"),
        getattr(cfg, stream + "_rope_base"),
    )


class BeatGridEncoder(nn.Module):
    """One encoding per (song, beat map); both views share the resulting features."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.dim
        self.cfg = cfg
        self.head_dim = cfg.head_dim
        self.subdiv_conv = nn.ModuleList(FlatTemporalBlock(d, 5, r) for r in (1, 3, 4))
        self.blocks = nn.ModuleList(
            LocalBeatRoFormerBlock(d) for _ in range(cfg.beat_encoder_layers)
        )
        self.proj = nn.ModuleDict()
        self.norm = nn.ModuleDict()
        self.sinks = nn.ParameterDict()
        for name in STREAMS:
            heads = getattr(cfg, name + "_kv_heads")
            self.proj[name] = nn.Linear(d, 2 * heads * cfg.head_dim, bias=False)
            self.norm[name] = DampedRMSNorm(cfg.head_dim)
            self.sinks[name] = nn.Parameter(torch.zeros(heads, cfg.head_dim))

    def forward(self, bank, song, seconds, valid):
        def sample(frames, length, rate):
            # Clamp to valid frame centers, then hard-mask physical time on both sides.
            pos = ((seconds * rate - 0.5).clamp_min(0)).minimum(
                (length - 1).clamp_min(0)
            )
            lo = pos.long()
            right = (lo + 1).minimum((length - 1).clamp_min(0))
            return torch.lerp(
                frames[song, lo],
                frames[song, right],
                (pos - lo)[:, None].to(frames.dtype),
            )

        x = sample(bank.mel_feat, bank.mel_lengths[song], 125) + sample(
            bank.mert_feat, bank.mert_lengths[song], 25
        )
        x = x.masked_fill(~valid[:, None], 0).T[None].contiguous()
        for conv in self.subdiv_conv:
            x = conv(x).masked_fill(~valid, 0)
        x = x[0].T.contiguous()
        beats = torch.arange(len(x), device=x.device) / GRID_STEPS_PER_BEAT
        for block in self.blocks:
            x = block(x, beats, valid)
        return x, self.project_audio(x, beats)

    def project_audio(self, x, beats=None):
        if beats is None:
            beats = torch.arange(len(x), device=x.device) / GRID_STEPS_PER_BEAT
        pairs = []
        for name in STREAMS:
            k, v = (
                self.proj[name](x).unflatten(-1, (-1, 2 * self.head_dim)).chunk(2, -1)
            )
            k = audio_rotary(
                self.cfg, name, self.norm[name](k).transpose(0, 1)[None], beats
            )
            sink = self.sinks[name].to(k.dtype)[None, :, None]
            pairs.append(
                (
                    torch.cat((sink, k), 2),
                    torch.cat((torch.zeros_like(sink), v.transpose(0, 1)[None]), 2),
                )
            )
        return pairs


@lru_cache(maxsize=1)
def frozen_mert(model_path, device, bundled=False):
    py_state, np_state = random.getstate(), np.random.get_state()
    dev = torch.device(device)
    try:
        with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):
            if bundled:
                # Load the shipped Python package in place. AutoModel's remote-code
                # loader otherwise copies these local modules into the user's HF cache.
                from importlib import import_module

                folder = Path(model_path).resolve()
                sys.path.insert(0, str(folder.parent))
                try:
                    module = import_module(folder.name + ".modeling_mert2")
                finally:
                    sys.path.pop(0)
                config_type, model_type = module.MERT2Config, module.MERT2Model
                options = {"local_files_only": True}
            else:
                from transformers import AutoConfig, AutoModel

                config_type, model_type = AutoConfig, AutoModel
                options = {"trust_remote_code": True}

            cfg = config_type.from_pretrained(model_path, **options)
            # FullSong exposes Conformer blocks directly as .layers. Construct/load
            # only L1..L8 so last_hidden_state is L8 even for unequal song lengths.
            cfg.num_hidden_layers = 8
            model = model_type.from_pretrained(
                model_path,
                config=cfg,
                **options,
                attn_implementation="sdpa",
            )
            return (
                model.eval()
                .requires_grad_(False)
                .to(
                    device=dev,
                    dtype=torch.bfloat16 if dev.type == "cuda" else torch.float32,
                )
            )
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


@lru_cache(maxsize=1)
def flat_mel_frontend(device):
    return DualWindowMelFrontend().to(device)


def load_wave(path):
    wave = read_waveform(path, 44100)
    return wave, song_reference_db(wave.numpy())


@torch.inference_mode()
def extract_flat_audio_batch(
    paths, device, mert_model=DEFAULT_MERT_V2_MODEL, *, pool=None, bundled=False,
    feature_dtype=torch.bfloat16,
):
    rows = list(map(load_wave, paths) if pool is None else pool.map(load_wave, paths))
    waves, refs = zip(*rows)
    device = torch.device(device)
    lengths = torch.tensor([len(w) for w in waves], device=device)
    wave = nn.utils.rnn.pad_sequence(waves, batch_first=True).to(device)
    if wave.shape[1] == 0:
        wave = F.pad(wave, (0, 1))
    features, mel_lengths = flat_mel_frontend(device).batch(wave, lengths, refs)
    mel = features[:, :4].transpose(1, 2).flatten(2).to(feature_dtype)
    mert_lengths = (mel_lengths + 4) // 5
    mert = extract_mert_batch(waves, device, mert_model, (mel.shape[1] + 4) // 5, pool=pool, bundled=bundled)
    return dict(
        mel=mel,
        mert=mert.to(feature_dtype),
        sample_lengths=lengths,
        mel_lengths=mel_lengths,
        mert_lengths=mert_lengths,
    )


@torch.inference_mode()
def extract_mert_batch(waves, device, mert_model, frames, *, pool=None, bundled=False):
    backbone = frozen_mert(str(mert_model), device, bundled)
    resample = partial(fast_resample, source_rate=44100, target_rate=backbone.config.sampling_rate)
    native = list(map(resample, waves) if pool is None else pool.map(resample, waves))
    return extract_native_mert_batch(native, device, backbone, frames)


@torch.inference_mode()
def extract_native_mert_batch(native, device, backbone, frames):
    cfg = backbone.config
    native = [F.pad(w, (0, cfg.minimum_input_samples - len(w))) if len(w) < cfg.minimum_input_samples
              else w for w in native]
    sizes = torch.tensor([len(w) for w in native], device=device)
    inputs = nn.utils.rnn.pad_sequence(native, batch_first=True).to(device)
    mask = torch.arange(inputs.shape[1], device=device) < sizes[:, None]
    with torch.autocast(device.type, enabled=False):
        out = backbone(
            inputs, attention_mask=mask, output_hidden_states=False, return_dict=True
        )
    times = (
        torch.arange(frames, device=device, dtype=torch.float64) + 0.5
    ) / 25
    return sample_centers_batch(
        out.last_hidden_state.float(),
        out.feature_attention_mask.sum(1),
        25,
        times,
        MERT_V2_ORIGIN_SECONDS,
    )


def write_flat_audio(
    folder, features, i, *, int8_mert=False, fp16=False, mel_uint16_2ch=False
):
    folder.mkdir(parents=True, exist_ok=True)
    samples = int(features["sample_lengths"][i])
    mel_length = max(1, (samples * 125 + 44099) // 44100)
    frames = {"mel": mel_length, "mert": (mel_length + 4) // 5}
    dtype = torch.float16 if fp16 else torch.bfloat16
    # Validate the newly produced arrays once, before publishing either file.
    with tempfile.TemporaryDirectory(dir=folder) as temp:
        for key, width in (("mel", 512), ("mert", 1024)):
            n = int(features[key + "_lengths"][i])
            quantize = (key == "mel" and mel_uint16_2ch) or (
                key == "mert" and int8_mert
            )
            value = (
                features[key][i, :n]
                .to(device="cpu", dtype=torch.float32 if quantize else dtype)
                .contiguous()
            )
            if (
                n != frames[key]
                or value.shape != (n, width)
                or not value.isfinite().all()
            ):
                raise ValueError(
                    f"Invalid {key} features for {folder.name}: expected {frames[key]} x {width} finite values"
                )
            if key == "mel" and mel_uint16_2ch:
                np.save(Path(temp) / "mel.npy", quantize_mel_2ch(value).numpy())
            elif key == "mert" and int8_mert:
                quantized, scale = quantize_mert(value)
                np.save(Path(temp) / "mert_scale.npy", scale.numpy())
                np.save(Path(temp) / "mert.npy", quantized.numpy())
            elif fp16:
                np.save(Path(temp) / (key + ".npy"), value.numpy())
            else:
                np.save(Path(temp) / (key + ".npy"), value.view(torch.uint16).numpy())
        if int8_mert:
            (Path(temp) / "mert_scale.npy").replace(folder / "mert_scale.npy")
        for key in frames:
            (Path(temp) / (key + ".npy")).replace(folder / (key + ".npy"))
    return dict(
        **({"mel_dtype": "uint16_2ch"} if mel_uint16_2ch else {}),
        **({"mert_dtype": "int8"} if int8_mert else {}),
        **({"storage_dtype": "float16"} if fp16 else {}),
        samples=samples,
        frames=frames,
        origins={k: 0.5 / rate for k, rate in FLAT_RATES.items()},
    )


def read_flat_audio(folder, info):
    out = {"sample_lengths": info["samples"]}
    for k in FLAT_RATES:
        values = torch.from_numpy(np.load(folder / (k + ".npy"), mmap_mode="c"))
        if k == "mert" and info.get("mert_dtype") == "int8":
            scale = torch.from_numpy(np.load(folder / "mert_scale.npy"))
            out[k] = (values.float() * scale.float()).bfloat16()
        elif k == "mel" and info.get("mel_dtype") == "uint16_2ch":
            out[k] = dequantize_mel_2ch(values)
        elif info.get("storage_dtype") == "float16":
            out[k] = values.bfloat16()
        else:
            out[k] = values.view(torch.bfloat16)
        out[k + "_lengths"] = info["frames"][k]
    return out


def flat_audio_manifest(doc):
    if doc["schema"] != AUDIO_PACK_SCHEMA:
        raise ValueError(f"Expected audio pack {AUDIO_PACK_SCHEMA}")
    return doc


def generation_audio(args, cfg, device):
    if not args.compact_manifest:
        return extract_flat_audio_batch(
            [args.audio], device, args.mert_model or DEFAULT_MERT_V2_MODEL
        )
    manifest = args.compact_manifest
    doc = json.loads(manifest.read_bytes())
    if "audio_manifest" in doc:
        manifest = manifest.parent / doc["audio_manifest"]
        doc = json.loads(manifest.read_bytes())
    info = flat_audio_manifest(doc)["audio"][args.audio_id]
    feat = pad_flat_audio(
        [read_flat_audio(manifest.parent / "audio" / args.audio_id, info)]
    )
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in feat.items()
    }


MEL_MIN = -1.0 / 6.0  # (-60 dB + 48) / 72
MEL_RANGE = 4.0 / 3.0  # (36 - (-60)) / 72
MEL_UINT16_SCALE = 65535.0 / MEL_RANGE
MEL_UINT16_INV_SCALE = MEL_RANGE / 65535.0


def quantize_mel_2ch(values):
    """Quantize bounded [long, short] Mel channels over [-1/6, 7/6] to uint16."""
    base = values.float().unflatten(-1, (4, 128))[:, (0, 2)].flatten(1)
    scaled = (base - MEL_MIN) * MEL_UINT16_SCALE
    return scaled.clamp_(0, 65535).round().to(torch.uint16)


def dequantize_mel_2ch(values):
    """Restore [long, dl, short, ds] BF16 Mel from 2-channel uint16 storage."""
    if values.ndim != 2 or values.shape[1] != 256 or values.dtype != torch.uint16:
        raise ValueError(
            f"Expected 2-channel uint16 Mel of shape (T, 256), got {tuple(values.shape)} {values.dtype}"
        )
    n = len(values)
    out = torch.empty((n, 4, 128), dtype=torch.bfloat16)
    if n == 0:
        return out.flatten(1)
    base = (
        values.unflatten(-1, (2, 128))
        .float()
        .mul_(MEL_UINT16_INV_SCALE)
        .add_(MEL_MIN)
    )
    out[:, 0] = base[:, 0]
    out[:, 2] = base[:, 1]
    out[0, 1] = 0
    out[0, 3] = 0
    if n > 1:
        diff = (base[1:] - base[:-1]).mul_(16.0)
        out[1:, 1] = diff[:, 0]
        out[1:, 3] = diff[:, 1]
    return out.flatten(1)


def quantize_mert(values):
    values = values.float()
    # The scale used for quantization must equal the stored FP16 scale, even
    # for silent channels (1e-6 / 127 alone rounds to zero in FP16).
    scale = (values.abs().amax(0) / 127).clamp_min(2.0**-24).half()
    quantized = (values / scale.float()).round().clamp_(-127, 127).to(torch.int8)
    return quantized, scale


def pad_flat_audio(songs):
    audio = pad_audio(songs, FLAT_RATES)
    audio["mel_lengths_cpu"] = tuple(audio["mel_lengths"].tolist())
    return audio


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mert-model", default=DEFAULT_MERT_V2_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args(argv)
    normalized = json.loads(args.manifest.read_bytes())
    if normalized["schema"] != "moment205.normalized.v2":
        raise ValueError("Expected moment205.normalized.v2 chart records")
    songs = {
        r["audio_id"]: args.manifest.parent / r["audio"]
        for r in normalized["records"]
        if official(r) and r["split"] == "fit"
    }
    keys = list(songs)
    args.output.mkdir(parents=True, exist_ok=True)
    audio = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for start in range(0, len(keys), args.batch_size):
            group = keys[start : start + args.batch_size]
            features = extract_flat_audio_batch(
                [songs[k] for k in group], args.device, args.mert_model, pool=pool
            )
            for i, key in enumerate(group):
                audio[key] = write_flat_audio(args.output / "audio" / key, features, i)
    with tempfile.NamedTemporaryFile(
        "w", dir=args.output, encoding="utf8", delete=False
    ) as fp:
        json.dump(
            dict(
                schema=AUDIO_PACK_SCHEMA,
                audio_models=AUDIO_MODELS,
                mert_model=args.mert_model,
                audio=audio,
            ),
            fp,
        )
    Path(fp.name).replace(args.output / "manifest.json")


if __name__ == "__main__":
    main()
