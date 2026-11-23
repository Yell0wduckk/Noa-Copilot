"""Full-song refinement on observed acoustic events, independent of score metadata."""

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import signal
from scipy.special import expit
import torch
from torch import nn
from torch.nn import functional as F

from maidata.nn.position import apply_rope, rope_tables
from .calibrator_semicrf import BarLattice, ContextualBarSemiCRF
from .calibrator_audibility import inside_audible_bounds
from .calibrator_partition import deduplicate_peaks, require_numba

_CANDIDATE_POOL = ThreadPoolExecutor(max_workers=8)


@torch.no_grad()
def phase_log_bpm_track(eighth_phase, valid_mask, *, beats_per_cycle=.5, min_bpm=45., max_bpm=360., kernel=15):
    phase = eighth_phase.float()
    sine, cosine = phase.unbind(-1)
    if phase.shape[1] < 2:
        return torch.zeros_like(sine), torch.zeros_like(sine)
    cross = sine[:, 1:] * cosine[:, :-1] - cosine[:, 1:] * sine[:, :-1]
    dot = cosine[:, 1:] * cosine[:, :-1] + sine[:, 1:] * sine[:, :-1]
    bpm = F.pad(torch.atan2(cross, dot), (1, 0), mode="replicate") * (3750 * beats_per_cycle / math.pi)
    # Replicate each song's own last sample before median filtering a padded batch.
    last = (valid_mask.long().sum(1, keepdim=True)-1).clamp_min(0)
    bpm = torch.where(valid_mask, bpm, bpm.gather(1, last))
    bpm = F.pad(bpm, (kernel//2, kernel//2), mode="replicate").unfold(1, kernel, 1).median(-1).values
    conf = ((phase.norm(dim=-1) - .30) / .45).clamp(0, 1) * valid_mask * (last > 0) * (bpm > 20)
    return conf * (bpm.clamp(min_bpm, max_bpm) / 120).log2(), conf


def _pulse_ioi_log_bpm(times, eighth_logit, beat_logit, fallback_log_bpm, *, min_bpm=43., max_bpm=400.,
                       strong_ids=None, beats_per_cycle=.5):
    if strong_ids is None:
        strong_ids = torch.where((eighth_logit >= -.2) | (beat_logit >= -.4))[0]
    if len(strong_ids) < 4:
        return fallback_log_bpm, fallback_log_bpm
    strong = times[strong_ids]
    dt = strong.diff()
    span = 60 * beats_per_cycle
    dt = dt.masked_fill((dt < span/max_bpm) | (dt > span/min_bpm), float("nan"))
    nearest = _nearest_indices(strong, times)
    nearby = (times - strong[nearest]).abs() <= 1.
    def estimate(width):
        intervals = F.pad(dt[None], (width//2, width//2 + 1), mode="replicate")
        median = intervals.unfold(1, width, 1).nanmedian(-1).values.flatten()[nearest]
        log = ((span/median).clamp(min_bpm, max_bpm)/120).log2()
        return torch.where(nearby & median.isfinite(), log, fallback_log_bpm)
    return estimate(5), estimate(17)


def _octave_augmentation(times, log_bpm, rng=np.random):
    """Choose one octave error in physical time, thinning the same interval in mode B."""
    size = len(times)
    if size <= 16:
        return times, None
    mode = rng.random()
    if mode < .55:
        width = min(size, rng.randint(16, 33))
    else:
        width = size if rng.random() < .5 else rng.randint(math.ceil(.3 * size), size + 1)
    lo = rng.randint(size - width + 1)
    hi = lo + width
    segment = times[lo:hi]
    first, last = float(segment[0]), float(segment[-1])
    if mode < .55:
        direction = rng.choice((-1., 1.))
    else:
        bpm = 120 * 2 ** np.median(log_bpm[(segment * 125).astype(int).clip(0, len(log_bpm) - 1)])
        if mode < .80:
            legal = [s for s in (-1., 1.) if 50 <= bpm * 2 ** s <= 300]
            if not legal:
                return times, None
            direction = rng.choice(legal)
        elif bpm >= 120:
            direction = -1.
            times = np.r_[times[:lo], segment[rng.randint(2)::2], times[hi:]]
        else:
            return times, None
    return times, (first, last, direction, mode >= .80)


def candidate_times(data, augment=False, *, with_octave_patch=False, legacy=False, asymmetric=True, rng=np.random):
    """Return physical seconds, with evidence from the nested pulse and phase heads."""
    eighth, tactus, phase, offset, audible, log_bpm, *semantic_bar = data
    strength = 0.7 * expit(eighth) + 0.2 * expit(tactus) + 0.1 * np.clip(phase, -1, 1)
    usable = audible.astype(bool)
    peaks = (
        signal.find_peaks(
            np.r_[-2, np.where(usable, strength, -2), -2], height=0.12, distance=4, prominence=0.04
        )[0]
        - 1
    )
    # A well-formed phase cycle can bridge an unaccented eighth even when
    # neither discrete head fires. A constant cosine cannot create events.
    phase_peaks = signal.find_peaks(
        np.where(usable, phase, -1.0), height=0.65, prominence=0.8, distance=4
    )[0]
    peaks = np.union1d(peaks, phase_peaks)
    if semantic_bar:
        combined = np.where(usable, semantic_bar[0], -20.0)
        bar_peaks = signal.find_peaks(np.r_[-20., combined, -20.], height=-0.5,
                                     distance=12, prominence=0.6)[0] - 1
        peaks = np.union1d(peaks, bar_peaks)
        strength = np.maximum(strength, expit(combined))
    # Suppress duplicate sub-peaks within a single attack using the local clock.
    radius = np.clip((12 if legacy else 10) * np.exp2(-log_bpm[peaks]), 4, 20 if legacy else 16)
    require_numba("candidate_times deduplicate_peaks")
    peaks = deduplicate_peaks(peaks, strength, radius)
    times = (peaks + 0.5) / 125 + 0.001 * offset[peaks]
    if augment and len(times) > 3:
        period = rng.uniform(2, 12)
        times += rng.uniform(0.006, 0.018) * np.sin(
            2 * math.pi * times / period + rng.uniform(0, 6)
        )
        extra = 0.5 * (times[1:] + times[:-1])
        add = (rng.random(len(extra)) < 0.07) & (np.diff(times) < 1.0)
        retained = rng.random(len(times)) > 0.06
        if len(times) >= 24:
            gap = rng.randint(1, len(times) - 5)
            retained[gap : gap + rng.randint(2, 6)] = False
        near = np.empty(0)
        if asymmetric and not legacy:
            dt = np.diff(times)
            pick = (dt > .085) & (rng.random(len(dt)) < .05)
            u = rng.uniform(.26, .42, size=pick.sum())
            u = np.where(rng.random(len(u)) < .5, u, 1-u)
            offset = u * dt[pick]
            good = np.minimum(offset, dt[pick]-offset) > .025
            near = (times[:-1][pick] + offset)[good]
        times = np.sort(np.r_[times[retained], extra[add], near])
    ix = np.floor(times * 125).astype(int)
    times = times[(ix >= 0) & (ix < len(usable)) & usable[ix.clip(0, len(usable) - 1)]]
    if len(times) > 4096:
        # Retain the strongest observation in each chronological bin across the whole song.
        index = np.floor(times * 125).astype(int)
        times = times[
            [part[strength[index[part]].argmax()] for part in np.array_split(np.arange(len(times)), 4096)]
        ]
    if with_octave_patch:
        return _octave_augmentation(times, log_bpm, rng) if augment and not legacy else (times, None)
    return times


class SemanticMERTBranch(nn.Module):
    def __init__(self, *, legacy=False):
        super().__init__()
        from .bms_calibrator import MERTGate, RoFormerBlock

        self.gate = MERTGate(1, fixed_layer=True, dropout=.35 if legacy else 0., gate_bias=-2. if legacy else 0.)
        self.layers = nn.ModuleList(RoFormerBlock() for _ in range(2))
        self.up = nn.Linear(512, 256)
        self.aux = nn.Linear(256, 3 if legacy else 6)

    def forward(self, h, mert, layout, pos, tables, valid, locations):
        h = self.gate(h.detach(), mert) * valid[..., None]
        for layer in self.layers:
            h = layer(h, layout, pos, tables, valid)
        coarse = self.up(h)
        i = locations.long()
        j = torch.minimum(i + 1, valid.sum(1)[:, None] - 1)
        batch = torch.arange(len(h), device=h.device)[:, None]
        sem = torch.lerp(coarse[batch, i], coarse[batch, j], (locations - i).to(coarse.dtype)[..., None])
        return sem, self.aux(sem).float()


class MacroMERTRefiner(nn.Module):
    """Fuse aligned L17 at 25 Hz and predict additive updates on the acoustic frame clock."""

    def __init__(self, config):
        super().__init__()
        from .bms_calibrator import EXTENDED_METERS, ExtendedMeterHead, MERTGate, RoFormerBlock, TemporalBlock

        self.radius = config.stage2_macro_radius
        self.down = nn.Conv1d(384, 512, 5, stride=5)
        self.gate = MERTGate(1, fixed_layer=True, dropout=config.stage2_mert_dropout, gate_bias=-1.)
        self.layers = nn.ModuleList(RoFormerBlock() for _ in range(2))
        self.up = nn.Linear(512, 384)
        self.fusion = TemporalBlock(384, 5)
        self.sem_proj = nn.Linear(384, 256)
        self.delta_sum = nn.Linear(384, 2)
        self.delta_grid = nn.Linear(384, 3)
        self.aux = nn.Linear(384, 5)  # change, compound, tempo residual, bar sin/cos
        self.meter = ExtendedMeterHead(384, len(EXTENDED_METERS) + 1)
        with torch.no_grad():
            for layer in (self.delta_sum, self.delta_grid):
                layer.weight.zero_()
                layer.bias.zero_()
            self.aux.weight[2].zero_()
            self.aux.bias[2].zero_()

    def forward(self, out, mert, sub):
        from .bms_calibrator import LocalAttentionLayout, _prepare_mert_batch

        local = out["boundary_features"].detach()
        b, n = local.shape[:2]
        lengths = (out["lengths"] + 4) // 5
        h = self.down(F.pad(local.transpose(1, 2), (0, -n % 5))).transpose(1, 2)
        pos = torch.arange(h.shape[1], device=h.device)[None]
        valid = pos < lengths[:, None]
        layout = LocalAttentionLayout(lengths, h.shape[1], self.radius)
        tables = rope_tables(pos, 32)
        if b == 1 and mert.ndim == 3:
            mert = mert.unsqueeze(0)
        h = self.gate(h, _prepare_mert_batch(mert.to(h.device), h.shape[1])) * valid[..., None]
        for layer in self.layers:
            h = layer(h, layout, pos, tables, valid)
        coarse = self.up(h)
        locations = ((torch.arange(n, device=h.device) + .5) / 5 - .5).clamp_min(0)
        locations = torch.minimum(locations, lengths[:, None] - 1)
        left = locations.long()
        right = torch.minimum(left + 1, lengths[:, None] - 1)
        rows = torch.arange(b, device=h.device)[:, None]
        interp = torch.lerp(coarse[rows, left], coarse[rows, right], (locations - left).to(coarse.dtype)[..., None])
        mask = out["valid_mask"][..., None]
        u = self.fusion((local + interp) * mask) * mask
        dsub, ddown = self.delta_sum(u).float().unbind(-1)
        dgrid = self.delta_grid(u).float()
        down = out["downbeat_logit"] + ddown
        pulse = out["pulse_phase"] + dgrid[..., :2]
        aux = self.aux(u).float()
        phase, confidence = phase_log_bpm_track(pulse, out["valid_mask"], beats_per_cycle=1.)
        tempo = (phase + 2 * (aux[..., 2] / 2).tanh()).clamp(-2, 2)
        out.update(beat_logit=torch.logaddexp(sub + dsub, down), downbeat_logit=down,
                   pulse_phase=pulse, stop_logit=out["stop_logit"] + dgrid[..., 2],
                   log_bpm=tempo, phase_log_bpm=phase, change_logit=aux[..., 0],
                   compound_logit=aux[..., 1], bar_phase=aux[..., 3:], meter_logits=self.meter(u).float(),
                   boundary_features=u)
        return self.sem_proj(u), torch.stack((down, aux[..., 0], aux[..., 1]), -1), confidence


class GridBlock(nn.Module):
    def __init__(self, *, legacy=False):
        super().__init__()
        slopes = torch.tensor([.02, .04, .08, .12, .18, .25, .35, .50])
        self.slope_raw = None if legacy else nn.Parameter(slopes.expm1().log())
        self.norm = nn.LayerNorm(256)
        self.qkv = nn.Linear(256, 768)
        self.out = nn.Linear(256, 256)
        self.ffn = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 1024), nn.GELU(), nn.Linear(1024, 256))

    def forward(self, x, pos, tables, log_dt, valid):
        q, k, v = self.qkv(self.norm(x)).reshape(*x.shape[:2], 3, 8, 32).permute(2, 0, 3, 1, 4)
        bias = log_dt[:, None]
        if self.slope_raw is not None:
            slopes = F.softplus(self.slope_raw).to(v.dtype)
            bias = -slopes[None, :, None, None] * bias
        else:
            bias = bias.clone()
        bias.masked_fill_(~valid[:, None, None, :], -torch.inf)
        bias.diagonal(dim1=-2, dim2=-1).masked_fill_(~valid[:, None, :], 0.)
        y = F.scaled_dot_product_attention(
            apply_rope(q, pos, tables=tables),
            apply_rope(k, pos, tables=tables),
            v,
            attn_mask=bias,
        )
        x = x + self.out(y.transpose(1, 2).reshape_as(x))
        return (x + self.ffn(x)) * valid[..., None]


def _interval_ratio(interval, width):
    padded = F.pad(interval[None, None], (width-1, width), mode="replicate").flatten()
    medians = padded.unfold(0, width, 1).median(dim=-1).values
    return (medians[width:] / medians[:-width]).log2().clamp(-2, 2)


def _tempo_targets(truth, bpm_in):
    diff = truth - bpm_in
    shift = diff.round().clamp(-1, 1)
    return shift.long() + 1, (diff - shift).clamp(-.5, .5)


class BeatGridCorrector(nn.Module):
    def __init__(self, *, legacy=False, v1=False):
        super().__init__()
        self.semantic_pool = nn.Linear(256, 32)
        self.legacy = legacy
        self.v2 = not (legacy or v1)
        if self.v2:
            self.norm_ac = nn.LayerNorm(384)
            self.norm_sem = nn.LayerNorm(256)
            self.norm_mean = nn.LayerNorm(35)
            self.norm_max = nn.LayerNorm(35)
            self.geom_proj = nn.Sequential(nn.Linear(15, 64), nn.SiLU(), nn.Linear(64, 64))
        self.input = nn.Linear(384 + 256 + 2 * 35 + (64 if self.v2 else 8 if legacy else 10), 256)
        self.layers = nn.ModuleList(GridBlock(legacy=legacy) for _ in range(4))
        self.head = nn.Linear(256, 7 if legacy else 8)
        with torch.no_grad():
            biases = [1., 0., -2., 0.] + ([] if legacy else [0.]) + [-1., 1., -1.]
            self.head.bias.copy_(torch.tensor(biases))
            if not legacy:
                self.head.weight[4].zero_()
            self.head.weight[1].zero_()

    def prepare(self, out, sem, aux, i, times, augment, *, octave_patch=None, length=None, acoustic=None):
        device = sem.device
        t = torch.as_tensor(times, device=device, dtype=torch.float32)
        n = int(out["lengths"][i]) if length is None else length
        frame_ids = (np.asarray(times, dtype=np.float32) * 125).astype(np.int64).clip(0, n-1)
        f = torch.from_numpy(frame_ids).to(device, non_blocking=True)
        if not len(t):
            return None

        def read(key):
            return out[key][i, :n].detach()

        quarter = "eighth_logit" not in out
        pulse_key = "beat_logit" if quarter else "eighth_logit"
        accent_key = "downbeat_logit" if quarter else "beat_logit"
        interval = t.diff(prepend=t[:1]).clamp_min(0.008)
        if len(t) > 1:
            interval[0] = interval[1]
        bpm = read("log_bpm")[f].clone()
        if self.v2:
            sem_bpm = bpm
            phase_bpm = out.get("phase_log_bpm", out["log_bpm"])[i, f].detach()
            strong = None
            if acoustic is not None:
                strong = torch.from_numpy(np.flatnonzero((acoustic[0, frame_ids] >= -.2) |
                                                         (acoustic[1, frame_ids] >= -.4))).to(device, non_blocking=True)
            bpm, wide = _pulse_ioi_log_bpm(t, read(pulse_key)[f], read(accent_key)[f], sem_bpm,
                                           strong_ids=strong, beats_per_cycle=1. if quarter else .5)
            bpm = bpm.clone()
        bar = aux[i, f, 0] if quarter else read("downbeat_logit")[f] + aux[i, f, 0]
        if augment and len(t) > 4:
            width = min(len(t), np.random.randint(16, 33))
            start = np.random.randint(len(t) - width + 1)
            end = start + width
            bar = bar.clone()
            bar_stride = 4 if quarter else 8
            bar[start:end:bar_stride] += 3
            if self.legacy:
                bpm[start:end] += np.random.choice([-1., 1.])
            elif octave_patch is not None:
                lo, hi, shift, thinned = octave_patch
                local_shift = ((t >= lo) & (t <= hi)) * shift
                if not (self.v2 and thinned):
                    bpm += local_shift
                if self.v2:
                    sem_bpm = sem_bpm + local_shift
                    phase_bpm = phase_bpm + local_shift
            if not self.legacy and len(t) >= 16 and np.random.random() < .3:
                width = min(len(t), np.random.randint(16, 65))
                lo = np.random.randint(len(t) - width + 1)
                bpm[lo:lo+width] += np.random.choice((-1., 1.)) * np.random.uniform(.12, .35)
        sig = torch.stack([read(key) for key in (pulse_key, accent_key, "audible_logit")], -1)
        # Keep semantic_pool in FP32 so BF16 autocast does not quantize frames
        # into artificial ties before prefix-sum and segment-max pooling.
        with torch.autocast(device_type=device.type, enabled=False):
            pooled_sem = F.linear(
                sem[i, :n].float(),
                self.semantic_pool.weight.float(),
                self.semantic_pool.bias.float() if self.semantic_pool.bias is not None else None,
            )
        values = torch.cat((pooled_sem, sig.float()), -1)
        left = torch.cat((f[:1], f[:-1] + 1)).minimum(f)
        length = f - left + 1
        prefix = F.pad(values.cumsum(0), (0, 0, 1, 0))
        mean = (prefix[f + 1] - prefix[left]) / length[:, None]
        # PyTorch's segment_reduce("max") backward omits dividing by tie multiplicity
        # when grad_output < 0, creating a one-way positive-drift ratchet. Use
        # scatter_reduce("amax") initialized to -inf (not 0, which also collides with
        # tie counts when max == 0) so ties split gradients symmetrically for both signs.
        unique, inv = np.unique(frame_ids, return_inverse=True)
        first, last = int(unique[0]), int(unique[-1])
        seg = values[first:last+1]
        seg_ids = np.repeat(np.arange(len(unique)), np.diff(unique, prepend=first-1))
        seg_ids = torch.from_numpy(seg_ids).to(device, non_blocking=True)
        maxima = seg.new_full((len(unique), seg.shape[1]), -torch.inf).scatter_reduce(
            0, seg_ids[:, None].expand_as(seg), seg, reduce="amax", include_self=False
        )[torch.from_numpy(inv).to(device, non_blocking=True)]
        geom = torch.stack(
            (
                interval.log2(),
                (interval.roll(-1) / interval)
                .log2()
                .masked_fill(torch.arange(len(t), device=device) == len(t) - 1, 0.0),
                ((0.5 if quarter else 0.25) / interval).log2() - bpm,
                read("subframe_offset_ms")[f] / 4,
                read("pulse_phase" if quarter else "eighth_phase")[f, 1],
                bar.clamp(-10, 10)/5 if self.v2 else bar,
                bpm,
                aux[i, f, 1].clamp(-10, 10)/5 if self.v2 else aux[i, f, 1],
                *(() if self.legacy else (_interval_ratio(interval, 4), _interval_ratio(interval, 8))),
                *((read(pulse_key)[f].clamp(-10,10)/5, read(accent_key)[f].clamp(-10,10)/5,
                   phase_bpm, sem_bpm, wide) if self.v2 else ()),
            ),
            -1,
        )
        acoustic_nodes, semantic_nodes = out["boundary_features"][i, f], sem[i, f]
        if not quarter:
            acoustic_nodes = acoustic_nodes.detach()
        if self.v2:
            parts = (self.norm_ac(acoustic_nodes), self.norm_sem(semantic_nodes), self.norm_mean(mean),
                     self.norm_max(maxima), self.geom_proj(geom))
        else:
            parts = (acoustic_nodes, semantic_nodes, mean, maxima, geom)
        return dict(z=self.input(torch.cat(parts, -1)), times=t, frames=f, bar=bar, bpm=bpm,
                    bar_phase=out["bar_phase"][i, f], tactus=read("beat_logit")[f],
                    **({pulse_key: read(pulse_key)[f]} if self.v2 else {}))

    def refine_batch(self, songs):
        active = [song for song in songs if song is not None]
        if not active:
            return songs
        pad = nn.utils.rnn.pad_sequence
        x = pad([song["z"] for song in active], batch_first=True)
        times = pad([song["times"] for song in active], batch_first=True)
        sizes = [len(song["times"]) for song in active]
        pos = torch.arange(x.shape[1], device=x.device)[None]
        valid = pos < torch.tensor(sizes, device=x.device)[:, None]
        log_dt = torch.log1p((times[:, :, None] - times[:, None, :]).abs()).to(x.dtype)
        if self.legacy:
            log_dt = -.1 * log_dt
        tables = rope_tables(pos, 32)
        for layer in self.layers:
            x = layer(x, pos, tables, log_dt, valid)
        heads = self.head(x).float()
        for i, (song, size) in enumerate(zip(active, sizes)):
            raw = heads[i, :size]
            song.update(z=x[i, :size], valid=raw[:, 0], delta=20 * raw[:, 1].tanh(),
                        change=raw[:, 2], step=raw[:, 3], octave=raw[:, -3:])
            if not self.legacy:
                resid = .5 * raw[:, 4].tanh()
                song.update(bpm_in=song["bpm"], bpm_resid=resid, bpm=song["bpm"] + resid)
        return songs

    def forward(self, out, sem, aux, i, times, augment, *, octave_patch=None):
        song = self.prepare(out, sem, aux, i, times, augment, octave_patch=octave_patch)
        return self.refine_batch([song])[0]


def octave_resolved_rates(song):
    """Pool octave votes between observed change peaks; retain local BPM detail."""
    raw = song["bpm"].detach().float().cpu().numpy()
    logits = song["octave"].detach().float().cpu().numpy()
    gates = song["change"].detach().float().sigmoid().cpu().numpy()
    bar = song["bar"].detach().float().sigmoid().cpu().numpy()
    votes = raw + logits.argmax(1) - 1
    result = np.empty_like(raw)
    cuts = signal.find_peaks(gates, height=0.65, prominence=0.1, distance=2)[0]
    ratios = np.arange(-1, 2)
    for part in np.split(np.arange(len(raw)), cuts):
        if not len(part):
            continue
        order = np.argsort(votes[part])
        weights = (0.5 + bar[part][order]).cumsum()
        center = votes[part[order[np.searchsorted(weights, weights[-1] / 2)]]]
        evidence = logits[part] - 1.25 * np.minimum(1.0, abs(raw[part, None] + ratios - center))
        result[part] = 120 * np.exp2(raw[part] + evidence.argmax(1) - 1)
    return result


def observed_beat_steps(running, rates, changes=None, *, quarter=False, bar_prob=None):
    """Quantize observed intervals onto the active pulse clock (eighth or quarter)."""
    rate = 0.5 * (rates[:-1] + rates[1:])
    jump = abs(np.log2(rates[1:] / np.maximum(rates[:-1], 1e-3))) > 0.14
    if changes is not None:
        jump |= changes[1:] >= 0.5
    rate = np.where(jump, rates[:-1], rate)
    distance = np.diff(running) * rate / 60
    if quarter:
        full = np.maximum(1.0, np.round(distance))
        half = np.maximum(0.5, np.round(distance * 2) / 2)
        tol = np.minimum(0.020, 3.5 / np.maximum(rate, 1.0))
        supports_half = (
            (distance < 2.25)
            & (abs(half - full) > 1e-4)
            & (abs(distance - half) * 60 / rate <= tol)
            & (abs(distance - full) * 60 / rate > 2.0 * tol)
        )
        return np.where(supports_half, half, full)
    eighth = np.maximum(0.5, np.round(distance * 2) / 2)
    sixteenth = np.maximum(0.25, np.round(distance * 4) / 4)
    tol = np.minimum(0.015, 3 / np.maximum(rate, 1.0))
    supports_short_transition = ((distance < 1.5) & (abs(sixteenth - eighth) > 1e-4)
                                 & (abs(distance - sixteenth) * 60 / rate <= tol)
                                 & (abs(distance - eighth) * 60 / rate > 2 * tol))
    return np.where(supports_short_transition, sixteenth, eighth)


def refined_song(song, *, clock=None, stops=(), audible=None, protected=None, fit_clock=True):
    """Place surviving events on a score axis inferred entirely from audio."""
    if song is None:
        return None, np.empty(0)
    dt = song["times"] + 0.001 * song["delta"]
    accepted = song["valid"] >= 0
    pulse = song.get("eighth_logit", song.get("beat_logit"))
    if not fit_clock and clock is None and pulse is not None:
        accepted |= (pulse >= 1.0) & (song["valid"] + pulse >= 0.5)
    if protected is not None:
        accepted |= protected
    if audible is not None:
        frames = (dt * 125).floor().long()
        accepted &= (frames >= 0) & (frames < len(audible)) & audible[frames.clamp(0, len(audible) - 1)]
    ids = torch.where(accepted)[0]
    ids = ids[dt[ids].argsort()]
    if not len(ids):
        return None, np.empty(0)
    result = {key: value[ids] for key, value in song.items()}
    times = dt[ids].detach().cpu().numpy().astype(float)
    if not fit_clock and clock is None:
        # Select physical observations before comparing octave hypotheses. Do not
        # discard a close pair using the very octave that inference must resolve.
        running = times.copy()
        for stop in stops:
            running -= np.clip(times - stop["time"], 0, stop["duration"])
        distinct = np.diff(running) > .001
        if not distinct.all():
            groups = np.split(np.arange(len(times)), np.flatnonzero(distinct)+1)
            confidence = result["valid"].detach().cpu().numpy()
            keep = [g[np.argmax(confidence[g])] for g in groups]
            result = {key: value[keep] for key, value in result.items()}
            ids, running = ids[keep], running[keep]
        result["refined"] = dt[ids]
        return result, running
    bpm = octave_resolved_rates(result)
    result["rate"] = torch.as_tensor(bpm, device=dt.device)
    if clock is None:
        running = times.copy()
        for stop in stops:
            running -= np.clip(times - stop["time"], 0, stop["duration"])
        dist = np.diff(running) * (bpm[:-1] + bpm[1:]) / 120
        # Two refined observations can land on the same attack (or the same
        # stopped beat). Keeping both would inject a permanent quarter-beat shift.
        if np.any(dist < 0.125):
            confidence = result["valid"].detach().float().cpu().numpy()
            if protected is not None:
                confidence = expit(confidence) + protected[ids].cpu().numpy()
            groups = np.split(np.arange(len(times)), np.flatnonzero(dist >= 0.125) + 1)
            keep = np.array([group[np.argmax(confidence[group])] for group in groups])
            indices = torch.as_tensor(keep, device=dt.device)
            result = {key: value[indices] for key, value in result.items()}
            ids = ids[indices]
            running, bpm = running[keep], bpm[keep]
        increments = observed_beat_steps(running, bpm, result["change"].detach().sigmoid().cpu().numpy())
        coords = np.r_[0.0, increments.cumsum()]
    else:
        coords = np.asarray([clock.second_to_beat(t) for t in times])
        coords = np.round(coords * 4) / 4
    unique = np.r_[True, np.diff(coords) > 1e-5]
    selected = torch.as_tensor(np.flatnonzero(unique), device=dt.device)
    result = {key: value[selected] for key, value in result.items()}
    result["refined"] = dt[ids][selected]
    return result, coords[unique]


def _nearest_indices(values, query):
    right = torch.searchsorted(values.contiguous(), query.contiguous()).clamp_max(len(values) - 1)
    left = (right - 1).clamp_min(0)
    return torch.where((values[left] - query).abs() <= (values[right] - query).abs(), left, right)


@torch.no_grad()
def _bar_clock_inputs(songs):
    """Download the detached geometry for all candidate sequences in one transfer."""
    keys = ("times", "delta", "valid", "bpm", "change", "bar")
    packed = [
        torch.cat((torch.stack([s[k] for k in keys], dim=1), s["octave"]), dim=1)
        for s in songs if s is not None
    ]
    if not packed:
        return [None] * len(songs)
    rows = iter(torch.cat(packed).float().cpu().split([len(p) for p in packed]))
    result = []
    for song in songs:
        if song is None:
            result.append(None)
            continue
        row = next(rows)
        clock = dict(zip(keys, row[:, :len(keys)].unbind(1)))
        clock.update(octave=row[:, len(keys):], source=torch.arange(len(row)))
        result.append(clock)
    return result


def _bar_frame_inputs(audible, stop_logit, offsets):
    device = stop_logit.device
    return torch.stack(
        (
            audible.to(device, non_blocking=True),
            stop_logit.detach().float().sigmoid(),
            offsets.to(device, non_blocking=True),
        ),
        dim=-2,
    ).float().cpu()


class LatticeCorrection(nn.Module):
    def __init__(self, *, legacy=False, v1=False, v3_config=None):
        super().__init__()
        self.legacy = legacy
        self.v2 = not (legacy or v1)
        self.quarter = v3_config is not None
        self.semantic = MacroMERTRefiner(v3_config) if self.quarter else SemanticMERTBranch(legacy=legacy)
        if self.v2 and not self.quarter:
            # A new model starts on the measured phase clock.
            with torch.no_grad():
                self.semantic.aux.weight[3].zero_()
                self.semantic.aux.bias[3].zero_()
        self.corrector = BeatGridCorrector(legacy=legacy, v1=v1)
        self.bars = ContextualBarSemiCRF(v1=not self.v2)

    def forward(self, out, mert, layout, pos, tables, valid, locations):
        sem, aux = self.semantic(out["hidden_40ms"], mert, layout, pos, tables, valid, locations)
        if not self.legacy:
            if self.v2:
                phase, confidence = phase_log_bpm_track(out["eighth_phase"], out["valid_mask"])
                tempo = (phase + 2 * (aux[..., 3] / 2).tanh()).clamp(-2, 2)
                out.update(phase_log_bpm=phase, sem_log_bpm=tempo)
            else:
                tempo = 4 * (aux[..., 3] / 4).tanh()
            out.update(log_bpm=tempo, bar_phase=aux[..., 4:6], change_logit=aux[..., 1])
            aux = aux[..., :3]
        self._refine_nodes(out, sem, aux, torch.where(confidence > .3, phase, tempo.detach())
                           if self.v2 else out["log_bpm"].detach())

    def forward_quarter(self, out, mert, sub):
        sem, aux, confidence = self.semantic(out, mert, sub)
        clock = torch.where(confidence > .3, out["phase_log_bpm"], out["log_bpm"].detach())
        self._refine_nodes(out, sem, aux, clock)

    def _refine_nodes(self, out, sem, aux, clock):
        stopped = out["stop_logit"].sigmoid() >= 0.85
        usable = out["audible_mask"] & ~(stopped & F.pad(stopped[:, :-1], (1, 0)))
        data = (
            torch.stack(
                (
                    out["beat_logit" if self.quarter else "eighth_logit"],
                    out["downbeat_logit" if self.quarter else "beat_logit"],
                    out["pulse_phase" if self.quarter else "eighth_phase"][..., 1],
                    out["subframe_offset_ms"],
                    usable,
                    clock,
                    torch.maximum(out["downbeat_logit"], aux[..., 0]),
                ),
                1,
            )
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        lengths = out["lengths"].tolist()
        parallel = len(lengths) >= 4 and (self.v2 or not self.training)
        seeds = np.random.randint(2**31 - 1, size=len(lengths)) if parallel and self.training else None

        def candidates(i):
            rng = np.random if seeds is None else np.random.RandomState(seeds[i])
            augment = self.training and rng.random() < .5
            times, patch = candidate_times(data[i, :, :lengths[i]], augment, with_octave_patch=True,
                                           legacy=self.legacy, asymmetric=self.v2, rng=rng)
            return times, patch, augment
        # Worker RNGs derive solely from checkpointed state, in batch order.
        rows = _CANDIDATE_POOL.map(candidates, range(len(lengths))) if parallel else map(candidates, range(len(lengths)))
        songs = []
        for i, (times, patch, augment) in enumerate(rows):
            length = lengths[i]
            song = self.corrector.prepare(out, sem, aux, i, times, augment, octave_patch=patch,
                                          length=length, acoustic=data[i])
            if song is not None:
                # Integrate real frame time once for all candidate bar spans.
                # Missed nodes, pruning and tempo changes do not bias the mean.
                projected = self.bars.semantic(sem[i, :length])
                song["sem_integral"] = projected.float().cumsum(0)[song["frames"]]
            songs.append(song)
        self.corrector.refine_batch(songs)
        out.update(correction=self, songs=songs, semantic_aux=aux)

    def loss(self, out, targets):
        aux = out["semantic_aux"]
        mask = out["valid_mask"]
        losses = aux.sum() * 0.0
        if self.quarter:
            from .bms_calibrator import quarter_pulse_frame_sum
            micro_loss, _ = quarter_pulse_frame_sum(out, targets)
            moving = mask & targets["active"] & ~targets["stop"].bool()
            meter = F.cross_entropy(out["meter_logits"].transpose(1, 2), targets["meter"], reduction="none")
            # An unverified 3/4 or 6/4 label cannot resolve its compound counterpart.
            ambiguous = (targets["meter_den"] == 4) & ((targets["meter_num"] == 3) | (targets["meter_num"] == 6))
            use_meter = moving & (~ambiguous | targets["compound_valid"])
            change = F.binary_cross_entropy_with_logits(out["change_logit"], targets["change"], reduction="none")
            compound = F.binary_cross_entropy_with_logits(out["compound_logit"], targets["compound"], reduction="none")
            losses = (.5 * micro_loss + .15 * (meter * use_meter).sum()
                      + .15 * (change * (1 + 4 * targets["change"]) * mask).sum()
                      + .15 * (compound * moving * targets["compound_valid"]).sum())
        else:
            for j, key in enumerate(("downbeat", "change", "compound")):
                y = targets[key]
                use = mask if j != 2 else mask & targets["compound_valid"] & targets["active"]
                bce = F.binary_cross_entropy_with_logits(aux[..., j], y, reduction="none")
                losses = losses + (bce * (1 + 5 * y) * use).sum() * .15
        if not self.legacy:
            moving = mask & targets["active"] & ~targets["stop"].bool()
            tempo = F.smooth_l1_loss(out["log_bpm"], targets["log_bpm"], beta=.05, reduction="none")
            phase = out["bar_phase"]
            phase_loss = F.smooth_l1_loss(phase, targets["bar_phase"], reduction="none").mean(-1)
            phase_loss = phase_loss + .1 * (phase.norm(dim=-1) - 1).square()
            losses += .5 * (tempo * moving).sum() + .25 * (phase_loss * mask * targets["bar_valid"]).sum()
        clocks = _bar_clock_inputs(out["songs"])
        # OOM recovery already keeps the two dense score annotations on CPU.
        gold_bars = targets["gold_bar"].detach().cpu()
        score_beats = targets["score_beat"].detach().cpu()
        frame_data = _bar_frame_inputs(out["audible_mask"], out["stop_logit"], targets["nearest_delta_ms"])
        all_paths = []
        lengths = out["lengths"].tolist()
        grid_peaks = targets["grid_peak"].detach().cpu()
        change = targets["change"].detach().cpu()
        local_max = F.max_pool1d(change[:, None], 5, 1, 2).squeeze(1)
        change_peaks = (change >= local_max) & (change > .5)
        rows = []
        for i, n in enumerate(lengths):
            peaks = grid_peaks[i, :n].nonzero().flatten()
            marks = (peaks + .5)/125 + .001 * frame_data[i, 2, peaks]
            changes = change_peaks[i, :n].nonzero().flatten()
            seconds = (changes + .5)/125
            if clocks[i] is not None and len(changes):
                times = clocks[i]["times"]
                idx = _nearest_indices(times, seconds)
                close = (times[idx] - seconds).abs() < .15
                idx, changes = idx[close], changes[close]
            else:
                idx = changes[:0]
                changes = changes[:0]
            bars = (gold_bars[i, :n].isfinite().nonzero().flatten() + .5)/125
            rows.extend((marks, idx, changes, bars))
        packed = torch.cat(rows).to(aux.device, non_blocking=True)
        indices = iter(packed.split([len(row) for row in rows]))
        for i, song in enumerate(out["songs"]):
            marks, idx, peaks, seconds = (next(indices) for _ in range(4))
            if song is None:
                continue
            n = lengths[i]

            def get(key):
                return targets[key][i, :n].to(aux.device)

            f = song["frames"]
            if not len(marks):
                losses += F.softplus(song["valid"]).mean() * n
                continue
            target = marks[_nearest_indices(marks, song["times"])]
            residual = (target - song["times"]) * 1000
            valid = residual.abs() <= 20

            def average(value, use):
                return (value * use).sum() / use.sum().clamp_min(1)

            if self.quarter:
                valid_loss = F.binary_cross_entropy_with_logits(song["valid"], valid.float(), reduction="none")
                node_loss = average(valid_loss, get("tactus_valid")[f] | valid)
            else:
                node_loss = F.binary_cross_entropy_with_logits(song["valid"], valid.float())
            node_loss += average(
                F.smooth_l1_loss(song["delta"] / 10, residual / 10, reduction="none"), valid
            )
            truth = get("log_bpm")
            octave, resid_target = _tempo_targets(truth[f], song.get("bpm_in", song["bpm"]))
            if not self.legacy:
                node_loss += (3. if self.v2 else 1.) * average(F.smooth_l1_loss(song["bpm_resid"], resid_target, beta=.05,
                                                      reduction="none"), valid)
            node_loss += average(F.cross_entropy(song["octave"], octave, reduction="none"), valid)
            # Assign each true change to its nearest candidate, including changes
            # occurring between eighths. The step regression sees the full jump.
            labels = torch.zeros_like(song["change"])
            steps = torch.zeros_like(labels)
            if len(peaks):
                idx, peaks = idx.long(), peaks.long()
                labels[idx] = 1
                steps[idx] = truth[(peaks + 2).clamp_max(n - 1)] - truth[(peaks - 2).clamp_min(0)]
            node_loss += (
                F.binary_cross_entropy_with_logits(song["change"], labels, reduction="none")
                * (1 + (6 if self.quarter else 5 if self.legacy else 15) * labels)
            ).mean()
            node_loss += average(F.smooth_l1_loss(song["step"], steps, beta=0.05, reduction="none"), labels)
            # A missing bar can break a complete CRF path. Isolated observed
            # downbeats still supervise the contextual boundary probability.
            if len(seconds):
                near = seconds[_nearest_indices(seconds, song["times"])]
                y = ((near - song["times"]).abs() <= 0.024).float()
                logit = self.bars.bar_logits(song)
                node_loss += 0.25 * F.binary_cross_entropy_with_logits(logit, y)
            paths, compound_loss, count = self.prepare_bar_paths(
                song,
                gold_bars[i, :n],
                get("compound"),
                get("compound_valid"),
                score_beats[i, :n],
                targets["nearest_delta_ms"][i, :n],
                out,
                i,
                clock=clocks[i],
                frame_data=frame_data[i, :, :n],
            )
            if paths:
                scale = n / count
                all_paths.extend((g, pot, gold, w * scale) for g, pot, gold, w in paths)
                losses = losses + scale * compound_loss
            losses = losses + n * node_loss
        return losses + self.bars.path_nll(all_paths) if all_paths else losses

    def bar_loss(self, *args, **kwargs):
        paths, compound, count = self.prepare_bar_paths(*args, **kwargs)
        return (self.bars.path_nll(paths) + compound) / count if paths else compound

    def prepare_bar_paths(self, candidates, gold_frames, compound, compound_valid, score_beat, offsets, out, i,
                 *, clock=None, frame_data=None):
        from .bms_calibrator import _decode_stops

        zero = candidates["z"].new_zeros(())
        gold_frames = gold_frames.detach().cpu()
        frames = torch.where(gold_frames.isfinite())[0]
        if len(frames) < 2:
            return [], zero, 0
        if clock is None:
            clock = _bar_clock_inputs([candidates])[0]
        if frame_data is None:
            n = len(gold_frames)
            frame_data = _bar_frame_inputs(out["audible_mask"][i, :n], out["stop_logit"][i, :n], offsets)
        score_beat = score_beat.detach().cpu()
        audible, stop_probability, offsets = frame_data
        audible = audible.bool()
        _, stops = _decode_stops(stop_probability * inside_audible_bounds(audible))
        target = (frames + 0.5) / 125 + 0.001 * offsets[frames]
        refined = clock["times"] + 0.001 * clock["delta"]
        order = refined.argsort()
        matched = order[_nearest_indices(refined[order], target)]
        protected = torch.zeros_like(refined, dtype=torch.bool)
        protected[matched[(refined[matched] - target).abs() <= 0.040]] = True
        song, coords = refined_song(clock, stops=stops, audible=audible, protected=protected)
        if song is None or len(coords) < 2:
            return [], zero, 0
        times = song["refined"].detach()
        nearest = _nearest_indices(times, target)
        ids = nearest.numpy()
        good = ((times[nearest] - target).abs() <= 0.040).numpy()
        distance = np.diff(coords[ids])
        gold_values = gold_frames[frames].numpy()
        annotated = np.diff(gold_values)
        pairs = good[1:] & good[:-1] & (np.diff(ids) > 0) & (annotated >= 0.25) & (annotated <= 12)
        # Normalize each contiguous observed gold subpath on its own endpoints.
        # Never insert a missing reference bar or assume the first event is a downbeat.
        edges = np.flatnonzero(np.diff(np.r_[False, pairs, False])).reshape(-1, 2)
        total = zero
        paths = []
        count = 0
        confidence = song["valid"].numpy()
        observed_frames = (times * 125).long().clamp(0, len(score_beat) - 1)
        label_coords = (score_beat[observed_frames] * 4).round().numpy() / 4
        for a, b in edges:
            # Assemble one connected gold subpath. Only intervals with an incompatible
            # predicted span use label coordinates, and only at existing observed nodes.
            selected = [ids[a]]
            local_coords = [0.0]
            gold = [0]
            for m in range(a, b):
                lo, hi = ids[m : m + 2]
                idx = np.arange(lo + 1, hi)
                if np.isclose(distance[m], annotated[m]):
                    positions = coords[idx] - coords[lo]
                else:
                    positions = label_coords[idx] - gold_values[m]
                    interior = (positions > 1e-5) & (positions < annotated[m] - 1e-5)
                    idx, positions = idx[interior], positions[interior]
                    # Keep the most credible observation at each quantized position.
                    if len(idx):
                        ranked = np.lexsort((-confidence[idx], positions))
                        unique = ranked[np.r_[True, np.diff(positions[ranked]) > 1e-5]]
                        idx, positions = idx[unique], positions[unique]
                base = local_coords[-1]
                selected.extend(idx.tolist() + [hi])
                local_coords.extend((base + positions).tolist() + [base + annotated[m]])
                gold.append(len(selected) - 1)
            source = song["source"][selected].to(candidates["z"].device, non_blocking=True)
            piece = {key: value[source] for key, value in candidates.items()}
            graph = BarLattice(local_coords)
            gold = np.asarray(gold)
            potentials = self.bars.potentials(graph, piece)
            span = np.diff(graph.nodes[gold])
            weight = 2. if self.v2 and np.any(abs(span - 4.) > 1e-4) else 1.
            paths.append((graph, potentials, gold, weight))
            ambiguous = np.flatnonzero(np.isclose(span, 3, atol=1e-4, rtol=0) |
                                       np.isclose(span, 6, atol=1e-4, rtol=0))
            if len(ambiguous):
                ix = graph.gold_edges(graph.nodes[gold])[ambiguous]
                inside = (frames[a:b] + frames[a + 1 : b + 1]) // 2
                use = compound_valid[inside[ambiguous]]
                y = compound[inside[ambiguous]]
                total = (
                    total
                    + weight * (
                        F.binary_cross_entropy_with_logits(potentials[2][ix], y, reduction="none") * use
                    ).sum()
                )
            count += b - a
        return paths, total, count
