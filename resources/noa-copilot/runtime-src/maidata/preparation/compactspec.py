"""Compact T/O/S extraction; production and inference share this exact path."""
from functools import lru_cache
from math import gcd
import numpy as np
import torch

from maidata.compactspec_contract import REPRESENTATION, CHANNELS, SLICES
from .features import grid_coordinates
from .timing import TempoMap

EPS = 2.5e-10
LOG_EPS = np.log(np.float32(EPS))
RECIPE = dict(representation=REPRESENTATION, sample_rate=24000,
    windows=[2048,512,1024], channels=['T:M2048','O:M512','S:S1024'], bins=249,
    bins_per_octave=[24,12,6], minimum_hz=[30.,30.,120.], counts=[149,62,38],
    slices=[[0,149],[149,211],[211,249]], reference_hz=32.703, epsilon=EPS,
    power='abs(rfft)**2/sum(hann)**2;one-sided-no-double',
    centers='linear-fft-bins-below-switch;reference-log-grid-through-j203/24.v1',
    filter='adjacent-centers;first-minus-bin;last-log-step;min-one-bin-halves;peak-one;weighted-sum',
    resample='scipy-resample-poly-kaiser5-constant0-round-length.v1',
    phase='absolute-k/24', sample_rounding='half-up-float64', boundary='reflect',
    normalization='fp64-frame-weight-1/asset-variant-count;std-floor-1e-6',
    storage='bf16-rne-uint16-bits')


def read_stereo(path):
    import soundfile as sf
    y, sr = sf.read(path, dtype='float32', always_2d=True)
    if not len(y) or not np.isfinite(y).all():
        raise ValueError(f'empty/nonfinite WAV: {path}')
    return y, sr


def resample_stereo(y, sr):
    if sr == 24000:
        return y
    from scipy.signal import resample_poly
    common = gcd(sr, 24000)
    count = max(1, round(len(y)*24000/sr))
    return resample_poly(y, 24000//common, sr//common, axis=0,
                         window=('kaiser',5.0), padtype='constant', cval=0)[:count].astype(np.float32)


def mid_side(pcm):
    """Prepare each asset's waveforms once for repeated frame queries."""
    y = torch.from_numpy(np.asarray(pcm, np.float32))
    mid = y[:,0] if y.shape[1] == 1 else (y[:,0]+y[:,1])*.5
    side = torch.zeros_like(mid) if y.shape[1] == 1 else (y[:,0]-y[:,1])*.5
    return mid,side


def silence_from_stats(stats):
    return ((LOG_EPS-np.asarray(stats['mean'],np.float32))/np.asarray(stats['std'],np.float32)).astype(np.float32)


def bf16_bits(values):
    return torch.as_tensor(np.asarray(values,np.float32)).to(torch.bfloat16).view(torch.uint16).numpy()


def from_bits(values):
    return torch.from_numpy(values).view(torch.bfloat16)


def context_at(beats, valid, tempo, duration):
    beats = np.asarray(beats,np.float64)
    progress = np.clip(beats/max(float(tempo.to_beats(duration)),1),0,1)
    progress = np.where(valid,progress,0)
    return np.stack((np.log2(tempo.tempo(beats)/120),progress),axis=-1).astype(np.float32)


@lru_cache(maxsize=3)
def centers_for(channel):
    _, n, b, minimum, _ = CHANNELS[channel]
    spacing = 24000/n
    switch = spacing/(2**(1/b)-1)
    linear = np.arange(np.ceil(minimum/spacing), np.ceil(switch/spacing))*spacing
    # j/b <= 203/24 avoids boundary-rounding errors at the top T grid.
    j = np.arange(int(np.ceil(b*np.log2(switch/32.703))), int(np.floor(b*203/24))+1)
    return np.concatenate((linear, 32.703*2.**(j/b)))


@lru_cache(maxsize=3)
def filter_bank(channel):
    _, n, b, _, _ = CHANNELS[channel]
    centers = centers_for(channel)
    spacing = 24000/n
    left = np.r_[centers[0]-spacing, centers[:-1]]
    right = np.r_[centers[1:], centers[-1]*2**(1/b)]
    low, high = np.minimum(left, centers-spacing), np.maximum(right, centers+spacing)
    bins = np.arange(n//2+1)*spacing
    weights = np.maximum(0, np.minimum((bins[None]-low[:,None])/(centers-low)[:,None],
                                     (high[:,None]-bins[None])/(high-centers)[:,None]))
    return torch.from_numpy(weights.astype(np.float32)), torch.hann_window(n, periodic=True)


def raw_at_centers(pcm, centers, valid, *, waves=None):
    mid, side = mid_side(pcm) if waves is None else waves
    centers = np.asarray(centers,np.int64).reshape(-1)
    valid = np.asarray(valid,bool).reshape(-1)
    result = torch.full((len(centers),249), float(LOG_EPS), dtype=torch.float32)
    for start in range(0,len(centers),512):
        stop = min(start+512,len(centers))
        for channel, (wave, (_,n,_,_,_), (lo,hi)) in enumerate(zip((mid,mid,side),CHANNELS,SLICES)):
            bank, window = filter_bank(channel)
            indices = torch.from_numpy(centers[start:stop])[:,None]+torch.arange(-n//2,n//2)
            if len(mid) == 1:
                indices = torch.zeros_like(indices)
            else:
                period = 2*(len(mid)-1)
                indices = indices.remainder(period)
                indices = torch.minimum(indices,period-indices)
            power = torch.fft.rfft(wave[indices]*window,dim=-1).abs().square()/window.sum().square()
            band = power @ bank.T
            result[start:stop,lo:hi] = torch.where(band == 0, float(LOG_EPS), torch.log(band+EPS))
    result[~torch.from_numpy(valid)] = float(LOG_EPS)
    return result.numpy()


class Statistics:
    def __init__(self):
        self.total = np.zeros((249,),np.float64)
        self.square = np.zeros_like(self.total)
        self.weight = 0.

    def add(self, raw, valid, weight):
        values = raw.reshape(-1,249)[np.asarray(valid).reshape(-1)].astype(np.float64)
        self.total += values.sum(0)*weight
        self.square += np.square(values).sum(0)*weight
        self.weight += len(values)*weight

    def finish(self):
        mean = self.total/self.weight
        std = np.sqrt(np.maximum(self.square/self.weight-mean**2,0)).clip(1e-6)
        return dict(recipe=RECIPE,mean=mean.astype(np.float32).tolist(),
                    std=std.astype(np.float32).tolist(),effective_frames=self.weight)


def extract_grid(pcm, timing, duration, stats, *, first_beat=None):
    tempo = TempoMap(timing)
    coordinates = grid_coordinates(tempo,duration,len(pcm),24,first_beat=first_beat)
    raw = raw_at_centers(pcm,coordinates['sample_indices'],coordinates['sample_valid'])
    mean, std = (np.asarray(stats[k],np.float32) for k in ('mean','std'))
    spec = (raw-mean)/std
    shape = coordinates['grid_k'].shape
    return dict(coordinates, spec_frames=bf16_bits(spec.reshape(*shape,249)),
                context_grid=context_at(coordinates['grid_k']/24,coordinates['sample_valid'],tempo,duration),
                silence_spec=silence_from_stats(stats), timing=timing, duration_seconds=duration)


def audio_grid(path, timing, stats, *, first_beat=0):
    y,sr = read_stereo(path)
    duration = len(y)/sr
    return extract_grid(resample_stereo(y,sr),timing,duration,stats,first_beat=first_beat),duration


def validate_grid(grid):
    """The existing produced/read-input shape and finite contract for the new representation."""
    value = grid['spec_frames']
    j = len(grid['beat_centers'])
    if value.shape != (j,24,249) or value.dtype != np.uint16:
        raise ValueError('spec_frames must be BF16 uint16 bits [J,24,249]')
    if (grid['sample_valid'].shape != (j,24) or grid['sample_valid'].dtype != np.bool_
            or grid['context_grid'].shape != (j,24,2)):
        raise ValueError('CompactSpec mask/context axes differ')
    if not torch.isfinite(from_bits(value)).all() or not np.isfinite(grid['context_grid']).all():
        raise ValueError('nonfinite CompactSpec input')
    silence = grid['silence_spec']
    if silence.shape != (249,) or silence.dtype != np.float32 or not np.isfinite(silence).all():
        raise ValueError('silence_spec must be finite float32 [249]')
    invalid = ~grid['sample_valid']
    if not np.all(value[invalid] == bf16_bits(silence)):
        raise ValueError('invalid CompactSpec frames must contain BF16 silence_spec bits')
    if np.any(grid['context_grid'][invalid,1] != 0):
        raise ValueError('invalid CompactSpec frames must have zero progress')
