"""Three native-rate sources, amplitude bypass, zero-residual narrow adapters."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .audio import LogMelFrontend, fast_resample, sample_centers
from .yourmt3_encoder import TrainableYMT3T5Encoder

class NarrowAdapter(nn.Module):
    def __init__(self, dim, kernel=5):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.up = nn.Conv1d(dim, 2 * dim, 1)
        self.depthwise = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.down = nn.Conv1d(dim, dim, 1)
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x, valid):
        z = self.norm(x).masked_fill(~valid[..., None], 0).transpose(1, 2)
        z = F.glu(self.up(z), dim=1).masked_fill(~valid[:, None], 0)
        z = self.down(F.silu(self.depthwise(z))).transpose(1, 2)
        return (x + self.gamma * z).masked_fill(~valid[..., None], 0)

class GatedAudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.d_model
        self.frontend = LogMelFrontend()
        self.yourmt3 = TrainableYMT3T5Encoder(config.yourmt3_unfreeze_layers)
        self.ymt3_norm = nn.LayerNorm(512, elementwise_affine=False)
        self.ymt3_projection = nn.Linear(512, dim, bias=False)
        self.input_projection = nn.Linear(512, dim, bias=False)
        self.input_convolution = nn.Sequential(
            nn.Conv2d(2, 8, (5, 7), padding=(2, 3), bias=False), nn.GELU(),
            nn.Conv2d(8, 2, 1, bias=False))
        self.convolution_projection = nn.Linear(512, dim, bias=False)
        # Separate anchor and macro gate projections preserve the complete learned
        # Stage 0 spec gate when Stage 1 adds its two new sources.
        self.gate_anchor = nn.Linear(2 * dim + 8, dim)
        self.macro_enabled = config.macro_enabled
        if self.macro_enabled:
            self.mert_norm = nn.LayerNorm(1024, elementwise_affine=False)
            self.muq_norm = nn.LayerNorm(1024, elementwise_affine=False)
            self.mert_projection = nn.Linear(1024, dim, bias=False)
            self.muq_projection = nn.Linear(1024, dim, bias=False)
            self.gate_macro = nn.Linear(4 * dim + 8, 2 * dim)
            nn.init.orthogonal_(self.mert_projection.weight)
            nn.init.orthogonal_(self.muq_projection.weight)
            nn.init.zeros_(self.gate_macro.weight)
            nn.init.constant_(self.gate_macro.bias, -2.)
        self.adapters = nn.ModuleList([NarrowAdapter(dim, config.adapter_kernel)
                                       for _ in range(config.adapter_layers)])
        nn.init.orthogonal_(self.ymt3_projection.weight)

    def set_trainable(self, frozen=False, top_enabled=True):
        self.requires_grad_(not frozen)
        self.yourmt3.set_top_trainable(top_enabled and not frozen)
        if frozen:
            self.eval()
            # The Stage 0 fine-grained trunk stays fixed while the newly
            # introduced Stage 1 sources learn their projections and gates.
            if self.macro_enabled:
                for module in (self.mert_projection, self.muq_projection, self.gate_macro):
                    module.requires_grad_(True)

    @staticmethod
    def _prepare_ymt3_windows(waveform):
        waveform = fast_resample(waveform, 44100, 16000)
        frames = max(1, math.ceil(waveform.numel() / 128))
        count = max(1, math.ceil((frames - 256) / 192) + 1)
        # Shift by 64 samples (4ms at 16kHz) so native hop=128 STFT centers
        # land directly on (i + 0.5) * 8ms without half-frame interpolation.
        waveform = waveform[64:]
        waveform = F.pad(waveform, (0, (count - 1) * 24576 + 32768 - waveform.numel()))
        return waveform.unfold(0, 32768, 24576)[:count], frames, count

    @staticmethod
    def _stitch_ymt3(values, frames, speed=1.):
        count = len(values)
        if count == 1:
            stitched = values[0, :frames]
        else:
            alpha = ((torch.arange(32, device=values.device, dtype=values.dtype) + .5) / 32)[:, None]
            pieces = [values[0, :240]]
            for i in range(1, count):
                # 64 overlap frames = 16 trimmed at each edge + 32 crossfade.
                # Every output index stays at its absolute native frame.
                previous = pieces.pop()
                cross = previous[-32:] * (1 - alpha) + values[i, 16:48] * alpha
                pieces.extend([previous[:-32], cross,
                               values[i, 48:256 if i == count - 1 else 240]])
            stitched = torch.cat(pieces)[:frames]
        if speed != 1.:
            stitched = sample_centers(stitched, 125., 125. / speed,
                                      max(1, math.ceil(frames / speed)),
                                      source_origin=.004, target_origin=.004 * speed)
        return stitched

    def _ymt3_song(self, waveform, speed=1.):
        # One pass per physical window; overlapping context is mandated by the
        # pretrained frontend, not activation recomputation. No CPU detaches.
        windows, frames, count = self._prepare_ymt3_windows(waveform)
        parts = [self.yourmt3(windows[start:start + self.config.frontend_batch_windows])
                 for start in range(0, count, self.config.frontend_batch_windows)]
        return self._stitch_ymt3(torch.cat(parts), frames, speed)

    def _ymt3_batch(self, waveforms, sample_lengths, speeds=None):
        window_groups, meta = [], []
        for i, samples in enumerate(torch.as_tensor(sample_lengths).cpu().tolist()):
            rate = 1. if speeds is None else float(speeds[i])
            windows, frames, count = self._prepare_ymt3_windows(waveforms[i, :samples])
            window_groups.append(windows)
            meta.append((frames, count, rate))
        all_windows = torch.cat(window_groups, dim=0)
        step = self.config.frontend_batch_windows
        parts = [self.yourmt3(all_windows[start:start + step])
                 for start in range(0, len(all_windows), step)]
        all_values = torch.cat(parts, dim=0)
        rows, offset = [], 0
        for frames, count, rate in meta:
            rows.append(self._stitch_ymt3(all_values[offset:offset + count], frames, rate))
            offset += count
        return nn.utils.rnn.pad_sequence(rows, batch_first=True)

    def forward(self, features, lengths=None, position_offset=0, *, waveforms=None,
                sample_lengths=None, speeds=None, ymt3=None, mert=None, muq=None,
                macro_lengths=None, macro_origins=None):
        from .model import AudioStreams
        batch, frames, _ = features.shape
        lengths_cpu = (torch.full((batch,), frames) if lengths is None else torch.as_tensor(lengths).cpu().long())
        device = features.device
        valid = torch.arange(frames, device=device)[None] < lengths_cpu.to(device)[:, None]
        features = features.masked_fill(~valid[..., None], 0)
        if ymt3 is None:
            ymt3 = self._ymt3_batch(waveforms, sample_lengths, speeds)
        ymt3 = F.pad(ymt3[:, :frames], (0, 0, 0, max(0, frames - ymt3.shape[1])))
        y = self.ymt3_projection(self.ymt3_norm(ymt3))
        image = features.reshape(batch, frames, 2, 256).permute(0, 2, 1, 3)
        local = self.input_convolution(image).permute(0, 2, 1, 3).flatten(-2)
        spec = self.input_projection(features) + self.convolution_projection(local)
        mel, delta = features.chunk(2, -1)
        energy = torch.stack((mel.mean(-1), mel[..., 192:].mean(-1),
            delta.clamp_min(0).mean(-1), (-delta).clamp_min(0).mean(-1),
            mel[..., :32].mean(-1), mel[..., 32:96].mean(-1),
            mel[..., 96:192].mean(-1), delta[..., 192:].mean(-1)), -1)
        anchor = torch.cat((y, spec, energy), -1)
        gs = self.gate_anchor(anchor).sigmoid()
        c = m = gc = gm = 0.
        if self.macro_enabled:
            # Read native streams once per song. No backbone is constructed
            # here; frozen L14/L8 extraction is an explicit preparation tool.
            if device.type == 'cpu':
                mert, muq = mert.float(), muq.float()
            cs, ms = [], []
            for i in range(batch):
                c_length = mert.shape[1] if macro_lengths is None else macro_lengths[i][0]
                m_length = muq.shape[1] if macro_lengths is None else macro_lengths[i][1]
                cp = self.mert_projection(self.mert_norm(mert[i, :c_length]))
                mp = self.muq_projection(self.muq_norm(muq[i, :m_length]))
                speed = 1. if speeds is None else float(speeds[i])
                origins = (.5 / (75. * speed), .5 / (25. * speed)) if macro_origins is None else macro_origins[i]
                cs.append(sample_centers(cp, 75. * speed, 125., frames, source_origin=origins[0]))
                ms.append(sample_centers(mp, 25. * speed, 125., frames, source_origin=origins[1]))
            c, m = torch.stack(cs), torch.stack(ms)
            gc, gm = self.gate_macro(torch.cat((anchor, c, m), -1)).sigmoid().chunk(2, -1)
        x = (y + gc * c + gm * m + gs * spec).masked_fill(~valid[..., None], 0)
        for adapter in self.adapters:
            x = adapter(x, valid)
        times = ((torch.arange(frames, device=device).float() + .5) + position_offset).expand(batch, -1)
        return AudioStreams(x, times, lengths_cpu.to(device), lengths_cpu, position_offset)
