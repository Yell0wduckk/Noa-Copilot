"""Native MERT and MuQ extraction with batched overlapping windows."""
import argparse
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .audio import read_waveform

class FrozenLayer(nn.Module):
    """Retain the exact chosen block output, before any final-only layer norm."""
    def __init__(self, backbone, layers, depth, *, normalize_waveform=False, frame_origin=None):
        super().__init__()
        self.backbone = backbone.eval().requires_grad_(False)
        parent, name = layers.rsplit('.',1)
        owner = backbone.get_submodule(parent)
        blocks = getattr(owner,name)
        setattr(owner,name,nn.ModuleList(list(blocks)[:depth]))
        self.normalize_waveform = normalize_waveform
        self.frame_origin = frame_origin
        self.selected = None
        getattr(owner,name)[-1].register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self.selected = output[0] if isinstance(output,tuple) else output

    @torch.no_grad()
    def forward(self, waveform):
        if self.normalize_waveform:
            waveform = waveform.float()
            waveform = (waveform - waveform.mean(-1, keepdim=True)) / waveform.var(
                -1, unbiased=False, keepdim=True).clamp_min(1e-4).sqrt()
        self.backbone(waveform, output_hidden_states=False)
        return self.selected

class FrozenMertPair(FrozenLayer):
    """Capture L6 and L14 in one frozen encoder traversal."""
    paired_layers = True

    def __init__(self, backbone, **kwargs):
        super().__init__(backbone, 'encoder.layers', 14, **kwargs)
        self.early = None
        self.backbone.encoder.layers[5].register_forward_hook(self._capture_early)

    def _capture_early(self, module, inputs, output):
        self.early = output[0] if isinstance(output, tuple) else output

    @torch.no_grad()
    def forward(self, waveform):
        late = super().forward(waveform)
        return torch.cat((self.early, late), dim=-1)

    @torch.no_grad()
    def sixth(self, waveform):
        # Only used by the single extraction thread when all selected tracks
        # already have L14. Reuse the same parameters and normalization.
        encoder = self.backbone.encoder
        full = encoder.layers
        try:
            encoder.layers = full[:6]
            super().forward(waveform)
            return self.early
        finally:
            encoder.layers = full


def load_mert(path, device='cuda', *, paired=False):
    if (Path(path) / 'frontend.safetensors').is_file():
        from .compact_frontends import compact_mert
        return compact_mert(path, device, paired)
    from transformers import AutoModel, AutoFeatureExtractor
    model=AutoModel.from_pretrained(path,trust_remote_code=True)
    processor=AutoFeatureExtractor.from_pretrained(path,trust_remote_code=True)
    receptive, jump = 1, 1
    for kernel,stride in zip(model.config.conv_kernel,model.config.conv_stride):
        receptive += (kernel-1)*jump
        jump *= stride
    options = dict(normalize_waveform=processor.do_normalize, frame_origin=(receptive-1)/2/24000)
    if paired:
        return FrozenMertPair(model, **options).to(device)
    return FrozenLayer(model,'encoder.layers',14, **options).to(device)

def load_muq(path, device='cuda'):
    if (Path(path) / 'frontend.safetensors').is_file():
        from .compact_frontends import compact_muq
        return compact_muq(path, device)
    from muq import MuQ
    model=MuQ.from_pretrained(path)
    model.model.linear=nn.Identity()  # unused 4096-way quantizer head
    # MuQ's centered 10ms Mel STFT starts at t=0. Both time-stride-2
    # 3-tap convolutions use padding 1, preserving that first frame center.
    return FrozenLayer(model,'model.conformer.layers',8,frame_origin=0.0).to(device)

@torch.inference_mode()
def extract_windowed_batch(waveforms, model, *, rate, window_seconds, stride_seconds,
                           trim_frames, windows_per_song, sample_lengths=None):
    """Pool the next native windows of all active songs in each backbone call."""
    if sample_lengths is None:
        sample_lengths = [len(w) for w in waveforms]
        waveforms = nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
    window, stride = round(window_seconds * 24000), round(stride_seconds * 24000)
    frames, hop = round(window_seconds * rate), round(stride_seconds * rate)
    lengths = [max(1, math.ceil(n * rate / 24000)) for n in sample_lengths]
    counts = [max(1, math.ceil((n - frames) / hop) + 1) for n in lengths]
    device = waveforms.device
    padded = F.pad(waveforms, (0, (max(counts) - 1) * stride + window - waveforms.shape[1]))
    windows = padded.unfold(1, window, stride)
    counts_cpu = torch.tensor(counts)
    starts = (counts_cpu.cumsum(0) - counts_cpu).to(device)
    encoded = None
    for start in range(0, max(counts), windows_per_song):
        offsets = torch.arange(start, min(start + windows_per_song, max(counts)))
        song, local = (offsets < counts_cpu[:, None]).nonzero(as_tuple=True)
        song, position = song.to(device), offsets[local].to(device)
        values = model(windows[song, position])
        if values.shape[1] < frames:
            values = F.pad(values, (0, 0, 0, frames - values.shape[1]), mode='replicate')
        if encoded is None:
            encoded = values.new_empty(sum(counts), frames, values.shape[-1])
        encoded.index_copy_(0, starts[song] + position, values[:, :frames])

    # Stitch all songs on their own clocks. At most two native windows cover a frame.
    t = torch.arange(max(lengths), device=device)
    t = torch.minimum(t, torch.tensor(lengths, device=device)[:, None] - 1)
    current = torch.minimum(t // hop, counts_cpu.to(device)[:, None] - 1)
    offset = t - current * hop
    index = starts[:, None] + current
    output = encoded[index, offset]
    if max(counts) > 1:
        overlap = frames - hop
        crossfade = overlap - 2 * trim_frames
        ramp = ((torch.arange(crossfade, device=device, dtype=encoded.dtype) + .5) / crossfade)
        weight = F.pad(ramp, (trim_frames, 0))
        weight = F.pad(weight, (0, trim_frames), value=1)
        cross = (current > 0) & (offset < overlap)
        row, col = cross.nonzero(as_tuple=True)
        j = offset[row, col]
        first, second = weight.flip(0)[j, None], weight[j, None]
        old = encoded[index[row, col] - 1, hop + j]
        output[row, col] = (old * first + output[row, col] * second) / (first + second)
    return [row[:n] for row, n in zip(output, lengths)]


def extract_mert_batch(waveforms, mert, windows_per_song, sample_lengths, *, sixth_only=False,
                       output_device='cpu'):
    values = extract_windowed_batch(waveforms, mert.sixth if sixth_only else mert,
                                    rate=75., window_seconds=5., stride_seconds=4., trim_frames=15,
                                    windows_per_song=windows_per_song, sample_lengths=sample_lengths)
    origin = getattr(mert, 'frame_origin', None)
    result = []
    for value in values:
        row = dict(mert_origin=.5/75 if origin is None else float(origin))
        if sixth_only:
            row["mert_l6"] = value.to(device=output_device, dtype=torch.bfloat16)
        else:
            if getattr(mert, "paired_layers", False):
                early, value = value.chunk(2, -1)
                row["mert_l6"] = early.to(device=output_device, dtype=torch.bfloat16)
            row.update(mert=value.to(device=output_device, dtype=torch.bfloat16), mert_layer=14)
        result.append(row)
    return result


def extract_muq_batch(waveforms, muq, windows_per_song, sample_lengths, *, output_device='cpu'):
    values = extract_windowed_batch(waveforms, muq, rate=25., window_seconds=30.,
                                    stride_seconds=24., trim_frames=50,
                                    windows_per_song=windows_per_song, sample_lengths=sample_lengths)
    origin = getattr(muq, 'frame_origin', None)
    return [dict(muq=y.to(device=output_device, dtype=torch.bfloat16),
                 muq_origin=.5/25 if origin is None else float(origin),
                 muq_layer=8, frame_clock="native-centers-v2") for y in values]


def extract_macro_batch(waveforms, mert, muq, windows_per_song=4, *, output_device='cpu',
                        sample_lengths=None):
    if sample_lengths is None:
        sample_lengths = [len(w) for w in waveforms]
        waveforms = nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
    c = extract_mert_batch(waveforms, mert, windows_per_song, sample_lengths, output_device=output_device)
    m = extract_muq_batch(waveforms, muq, windows_per_song, sample_lengths, output_device=output_device)
    return [x | y for x, y in zip(c, m)]


def extract_windowed(waveform, model, *, rate, window_seconds, stride_seconds, trim_frames,
                     batch_size=4):
    return extract_windowed_batch([waveform], model, rate=rate, window_seconds=window_seconds,
                                  stride_seconds=stride_seconds, trim_frames=trim_frames,
                                  windows_per_song=batch_size)[0]


def extract_macro(waveform, mert, muq, batch_size=4, *, output_device='cpu'):
    return extract_macro_batch([waveform], mert, muq, batch_size, output_device=output_device)[0]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio',type=Path,required=True)
    parser.add_argument('--mert-model',required=True)
    parser.add_argument('--muq-model',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--batch-size',type=int,default=4)
    args=parser.parse_args(argv)
    wave=read_waveform(args.audio,24000).to(args.device)
    result=extract_macro(wave,load_mert(args.mert_model,args.device,paired=True),load_muq(args.muq_model,args.device),args.batch_size)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(result,args.output)
if __name__=='__main__':
    main()
