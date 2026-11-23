"""Feature-only MERT L6/L14 and MuQ L8 for offline desktop inference."""

import json
from pathlib import Path

from torch import nn
from safetensors.torch import load_file


class MuQFeatures(nn.Module):
    frame_origin = 0.0

    def __init__(self, cfg):
        super().__init__()
        from muq.muq.modules.conv import Conv2dSubsampling
        from muq.muq.modules.features import MelSTFT
        from transformers import Wav2Vec2ConformerConfig
        from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
            Wav2Vec2ConformerEncoderLayer,
            Wav2Vec2ConformerRotaryPositionalEmbedding,
        )

        self.mel = MelSTFT(n_fft=2048, hop_length=cfg["hop_length"], is_db=True)
        self.mean = cfg["stat"]["melspec_2048_mean"]
        self.std = cfg["stat"]["melspec_2048_std"]
        self.conv = Conv2dSubsampling(
            1,
            cfg["conv_dim"],
            cfg["encoder_dim"],
            strides=[2, 2],
            n_bands=cfg["n_mels"],
        )
        config = Wav2Vec2ConformerConfig.from_dict(cfg["w2v2_config"])
        self.positions = Wav2Vec2ConformerRotaryPositionalEmbedding(config)
        self.layers = nn.ModuleList(
            Wav2Vec2ConformerEncoderLayer(config) for _ in range(cfg["encoder_depth"])
        )

    def forward(self, wave):
        mel = self.mel.to(wave.device)(wave.float())[..., :-1]
        x = self.conv((mel - self.mean) / self.std)
        positions = self.positions(x)
        for layer in self.layers:
            x = layer(x, relative_position_embeddings=positions)[0]
        return x


def compact_mert(folder, device, paired):
    from transformers import AutoConfig, AutoFeatureExtractor, AutoModel
    from .frontends import FrozenLayer, FrozenMertPair

    folder = Path(folder)
    cfg = AutoConfig.from_pretrained(
        str(folder), trust_remote_code=True, local_files_only=True
    )
    net = AutoModel.from_config(cfg, trust_remote_code=True)
    net.encoder.layer_norm = nn.Identity()  # final-only norm lies after the L14 tap
    net.load_state_dict(load_file(str(folder / "frontend.safetensors")))
    processor = AutoFeatureExtractor.from_pretrained(str(folder), local_files_only=True)
    receptive, jump = 1, 1
    for kernel, stride in zip(cfg.conv_kernel, cfg.conv_stride):
        receptive += (kernel - 1) * jump
        jump *= stride
    opts = dict(
        normalize_waveform=processor.do_normalize, frame_origin=(receptive - 1) / 48000
    )
    wrapper = (
        FrozenMertPair(net, **opts)
        if paired
        else FrozenLayer(net, "encoder.layers", 14, **opts)
    )
    return wrapper.to(device)


def compact_muq(folder, device):
    folder = Path(folder)
    model = MuQFeatures(json.loads((folder / "config.json").read_text("utf8")))
    model.load_state_dict(load_file(str(folder / "frontend.safetensors")))
    return model.eval().requires_grad_(False).to(device)
