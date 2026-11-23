"""Frame-local dense / two-layer T-convolution frontends, without normalization."""
import torch
from torch import nn

from maidata.compactspec_contract import TCONV


class CompactSpecFrontend(nn.Module):
    def __init__(self, d_model, config):
        super().__init__()
        self.bins = 249
        self.dense = nn.Linear(311, d_model)
        self.energy = nn.Sequential(nn.Linear(4, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.convolution = self.project = None
        if config['audio_frontend'] == TCONV:
            self.convolution = nn.Sequential(nn.Conv1d(3, 16, 9, padding=4), nn.GELU(),
                nn.Conv1d(16, 16, 5, stride=2, padding=2), nn.GELU())
            self.project = nn.Linear(944, d_model, bias=False)
            indices = torch.arange(117)[None] + torch.tensor([0, 24, 38])[:, None]
            self.register_buffer('harmonic_indices', indices.clamp_max(116), persistent=False)
            self.register_buffer('harmonic_valid', indices < 117, persistent=False)

    @staticmethod
    def energy_features(spec, silence):
        delta = spec - silence.to(device=spec.device, dtype=spec.dtype)
        return torch.stack([delta[..., start:stop].mean(-1)
                            for start, stop in ((0,149), (32,149), (149,211), (211,249))], -1)

    @staticmethod
    def assemble(spec, previous_o, sample_valid, previous_valid):
        difference = torch.relu(spec[...,149:211] - previous_o)
        difference = difference.masked_fill(~(sample_valid & previous_valid)[...,None], 0)
        return torch.cat((spec[...,:211], difference, spec[...,211:]), -1)

    def harmonics(self, spec, silence):
        silence = silence.to(device=spec.device, dtype=spec.dtype)
        return torch.where(self.harmonic_valid, spec[...,32:149][:,self.harmonic_indices], silence[148])

    def forward(self, spec, sample_valid, valid, silence_spec):
        # Establish predecessors on original song coordinates BEFORE compaction.
        previous_valid = torch.cat((torch.zeros_like(sample_valid[:,:1]), sample_valid[:,:-1]), 1)
        flat = spec.flatten(0, 1)
        selected = valid.flatten().nonzero().flatten()
        values = flat[selected]
        previous_o = flat[(selected-1).clamp_min(0),149:211]
        dense = self.assemble(values, previous_o, sample_valid.flatten()[selected], previous_valid.flatten()[selected])
        energy = self.energy_features(values, silence_spec)
        # Keep BF16 operands intact under autocast; only cast when running without autocast.
        dtype = spec.dtype if torch.is_autocast_enabled(spec.device.type) else self.dense.weight.dtype
        output = self.dense(dense.to(dtype)) + self.energy(energy.to(dtype))
        if self.convolution is not None:
            conv = self.convolution(self.harmonics(values, silence_spec).to(dtype))
            output = output + self.project(conv.flatten(1))
        result = output.new_zeros((len(flat), self.dense.out_features)).index_copy(0, selected, output)
        return result.reshape(*valid.shape, -1)
