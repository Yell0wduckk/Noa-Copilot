"""Audio BPM and song-progress context injection (`bpm-progress.v1`)."""

from torch import nn


class AudioContext(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.bpm = nn.Sequential(nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.progress = nn.Linear(1, d_model)

    def forward(self, spectral, context, beats, valid):
        context = context.to(self.bpm[0].weight.dtype)
        value = spectral + self.bpm(context[..., :1]) + self.progress(context[..., 1:])
        return value.masked_fill(~valid[..., None], 0)
