"""Offline BS.1770 K weighting and shared song-level loudness targets."""
import math
import numpy as np
from scipy.signal import lfilter

def k_weight(signal, sample_rate=44100):
    x = np.asarray(signal, dtype=np.float64)
    # BS.1770 prefilter constants, bilinear transform at the actual sample rate.
    k = math.tan(math.pi * 1681.974450955533 / sample_rate)
    q, vh = .7071752369554196, 10 ** (3.999843853973347 / 20)
    vb = vh ** .4996667741545416
    a0 = 1 + k / q + k * k
    b = np.array([vh + vb*k/q + k*k, 2*(k*k-vh), vh-vb*k/q+k*k]) / a0
    a = np.array([1., 2*(k*k-1)/a0, (1-k/q+k*k)/a0])
    x = lfilter(b, a, x, axis=0)
    k, q = math.tan(math.pi * 38.13547087602444 / sample_rate), .5003270373238773
    a0 = 1 + k/q + k*k
    return lfilter(np.array([1., -2., 1.]) / a0,
                   [1., 2*(k*k-1)/a0, (1-k/q+k*k)/a0], x, axis=0)

def k_loudness(signal, sample_rate=44100):
    filtered = k_weight(signal, sample_rate)
    energy = float(np.mean(filtered * filtered)) if len(filtered) else 0.
    return -.691 + 10 * math.log10(max(1e-20, energy))

def integrated_loudness(wave, sample_rate=44100):
    """BS.1770 stereo channel sum, 400 ms blocks, 75% overlap and both loudness gates."""
    filtered = k_weight(wave, sample_rate)
    power = (np.einsum('ij,ij->i', filtered,filtered) if filtered.ndim==2 else filtered**2)
    block = round(.4*sample_rate)
    step = round(.1*sample_rate)
    total = np.empty(len(power)+1)
    total[0] = 0.
    np.cumsum(power, out=total[1:])
    starts = np.arange(0,len(power)-block+1,step)
    energy = (total[starts+block]-total[starts])/block if len(starts) else total[-1:]/block
    # The high-pass in k_weight uses unity high-frequency gain. BS.1770's RLB
    # numerator is [1,-2,1] after normalising the denominator.
    k = math.tan(math.pi*38.13547087602444/sample_rate)
    energy *= (1+k/.5003270373238773+k*k)**2
    audible = energy[energy >= 10**((-70+.691)/10)]
    if not audible.size:
        return -math.inf
    gated = audible[audible >= audible.mean()*.1]
    return float(-.691+10*np.log10(gated.mean()))


def loudness_targets(levels):
    known = [float(x) for x in levels if x is not None and math.isfinite(x)]
    p95 = float(np.percentile(known, 95)) if known else 0.
    result = []
    for level in levels:
        if level is None or not math.isfinite(level):
            result.append((0, None))
        else:
            db = float(level) - p95
            result.append((1 + int(db >= -26) + int(db >= -16) + int(db >= -6), db))
    return result
