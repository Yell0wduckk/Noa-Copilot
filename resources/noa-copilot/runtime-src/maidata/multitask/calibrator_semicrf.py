"""Exact first-order semi-Markov inference on a sparse, physical quarter-beat axis."""

import math

import numpy as np
from scipy.special import expit, logit, logsumexp
from scipy.signal import find_peaks
from scipy.ndimage import maximum_filter1d
import torch
from torch import nn
from torch.nn import functional as F

import os
from concurrent.futures import ThreadPoolExecutor

from .calibrator_partition import NUMBA_ERROR, partition_scan, require_numba, viterbi_scan

if NUMBA_ERROR is not None:
    partition_scan = None

_SCAN_POOL = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1))


def observed_phase_grids(axis, seconds, logits, sub, beat_logits, start, end):
    """Only a strong bar between two coherent quarter clocks can change the lattice phase."""
    events = boundary_peaks(beat_logits)
    physical = seconds[events] + 0.001 * sub[events]
    quarters = np.interp(physical, seconds, axis)
    grids = []
    phase = 0.0
    for peak in boundary_peaks(logits, 0.55):
        time = seconds[peak] + 0.001 * sub[peak]
        hit = np.interp(time, seconds, axis)
        where = np.searchsorted(quarters, hit - 0.05)
        before = quarters[max(0, where - 5) : where]
        after = quarters[np.searchsorted(quarters, hit + 0.05) :][:5]
        if len(before) < 3 or len(after) < 3 or not start < hit < end:
            continue
        # A confirmed reset can also return to the original half-beat phase.
        nearest = phase + round(2 * (hit - phase)) / 2
        if abs(_node_times(nearest, axis, seconds) - time) <= 0.020:
            continue
        snapped = min((round(hit * d) / d for d in (4, 6)), key=lambda q: abs(q - hit))
        if abs(_node_times(snapped, axis, seconds) - time) > 0.015:
            continue

        def supports(values, anchor):
            distance = values - anchor
            return np.count_nonzero(
                abs(distance - np.round(distance)) < 0.05
            ) >= 0.75 * len(values)

        old = before[-1]
        if abs(2 * (old - phase) - round(2 * (old - phase))) > 0.10:
            continue
        if (
            supports(before, old)
            and supports(after, snapped)
            and not supports(after, old)
        ):
            if grids:
                grids[-1] = (grids[-1][0], snapped, phase)
            phase = snapped - round(2 * snapped) / 2
            grids.append((snapped, end, phase))
    return grids


def candidate_nodes(
    axis,
    seconds,
    logits,
    offsets_ms,
    start,
    end,
    gold=(),
    *,
    local_grids=(),
    inference=False,
):
    """Regular eighths plus observed, time-qualified sixteenths/triplets and training anchors."""
    half = np.arange(math.ceil(2 * start - 1e-6), math.floor(2 * end + 1e-6) + 1) / 2
    regular = half
    peaks = boundary_peaks(logits)
    physical = seconds[peaks] + offsets_ms[peaks] * 0.001
    hits = np.interp(physical, seconds, axis)
    phase = np.zeros_like(hits)
    for a, b, offset in local_grids:
        lo, hi = max(start, a), min(end, b)
        ticks = np.arange(
            math.ceil(2 * (lo - offset) - 1e-8),
            math.floor(2 * (hi - offset) + 1e-8) + 1,
        )
        half = np.r_[half[(half < lo) | (half > hi)], offset + ticks / 2]
        phase[(hits >= a) & (hits <= b)] = offset
    if inference:
        # Retain the global clock across a local recovery so a weak or missing
        # bridge endpoint cannot disconnect the rest of the recording.
        return np.unique(np.round(np.r_[start, regular, half, end], 8))
    snaps = np.stack([phase + np.round(d * (hits - phase)) / d for d in (4, 6)])
    times = _node_times(snaps, axis, seconds)
    which = np.abs(times - physical).argmin(0)
    i = np.arange(len(peaks))
    good = np.abs(times[which, i] - physical) <= 0.015
    snapped = snaps[which, i][good]
    nodes = np.sort(
        np.r_[gold, start, end, half, snapped[(snapped > start) & (snapped < end)]]
    )
    return nodes[np.r_[True, np.diff(nodes) > 1e-5]]


def _node_times(nodes, axis, seconds):
    times = np.interp(nodes, axis, seconds)
    # The attack at a STOP precedes the freeze. A point just beyond it
    # interpolates from the *end* of the plateau using the plain inverse.
    first = np.searchsorted(axis, nodes).clip(0, len(axis) - 1)
    return np.where(abs(axis[first] - nodes) <= 1e-6, seconds[first], times)


def boundary_peaks(logits, probability=0.35):
    # Padding makes an attack at frame zero / the last frame observable.
    height = logit(np.clip(probability, 0., 1.))
    return find_peaks(np.r_[-np.inf, logits, -np.inf], height=height)[0] - 1


class BarLattice:
    def __init__(self, nodes):
        self.nodes = np.asarray(nodes, dtype=np.float64)
        first = np.searchsorted(self.nodes, self.nodes - 12.0 - 1e-5)
        last = np.searchsorted(self.nodes, self.nodes - 0.25 + 1e-5, side="right")
        count = last - first
        self.ends = np.repeat(np.arange(len(nodes)), count)
        self.starts = np.concatenate([np.arange(a, b) for a, b in zip(first, last)])
        self.span = self.nodes[self.ends] - self.nodes[self.starts]
        self.in_offsets = np.r_[0, count.cumsum()]
        self.out_order = np.argsort(self.starts, kind="stable")
        self.out_offsets = np.r_[
            0, np.bincount(self.starts, minlength=len(nodes)).cumsum()
        ]

    def incoming(self, i):
        return slice(self.in_offsets[i], self.in_offsets[i + 1])

    def outgoing(self, i):
        return self.out_order[self.out_offsets[i] : self.out_offsets[i + 1]]

    def gold_edges(self, gold):
        ids = np.abs(self.nodes[:, None] - np.asarray(gold)).argmin(0)
        return (
            self.in_offsets[ids[1:]] + ids[:-1] - self.starts[self.in_offsets[ids[1:]]]
        )


def _scan(graph, score, switch, same_reward, *, viterbi=False, gradients=False, open_ends=None, return_score=False):
    """Each edge is a state. Incoming and outgoing edges meet at exactly one node."""
    score = np.asarray(score, dtype=np.float64)
    alpha = np.full(len(score), -np.inf)
    initial = graph.outgoing(0)
    alpha[initial] = score[initial]
    if viterbi:
        if open_ends is not None:
            leading, trailing = open_ends
            prefix_span = graph.nodes[graph.starts] - graph.nodes[0]
            alpha = np.where(prefix_span < graph.span, score + leading[graph.starts], -np.inf)
        parent = np.full(len(score), -1, dtype=np.int64)
        for i in range(1, len(graph.nodes) - 1):
            inc, out = graph.incoming(i), graph.outgoing(i)
            if inc.start == inc.stop or not len(out):
                continue
            same = abs(graph.span[inc, None] - graph.span[out]) < 1e-4
            choices = alpha[inc, None] + np.where(same, same_reward, switch[i])
            best_idx = choices.argmax(0)
            continued = score[out] + choices[best_idx, np.arange(len(out))]
            better = continued > alpha[out]
            parent[out[better]] = inc.start + best_idx[better]
            alpha[out] = np.maximum(alpha[out], continued)
        # A short, weak final fragment may have no legal incoming edge.
        reachable = np.flatnonzero(np.isfinite(alpha))
        if not len(reachable):
            return ([0], -np.inf) if return_score else [0]
        if open_ends is None:
            final = graph.incoming(graph.ends[reachable[-1]])
            edge = final.start + int(alpha[final].argmax())
            best_score = float(alpha[edge])
        else:
            tail_span = graph.nodes[-1] - graph.nodes[graph.ends]
            terminal = alpha + trailing[graph.ends]
            eligible = reachable[tail_span[reachable] < graph.span[reachable]]
            if not len(eligible):
                eligible = reachable
            edge = int(eligible[terminal[eligible].argmax()])
            best_score = float(terminal[edge])
        path = [int(graph.ends[edge])]
        while edge >= 0:
            path.append(int(graph.starts[edge]))
            edge = int(parent[edge])
        return (path[::-1], best_score) if return_score else path[::-1]

    def sums(src, dst, values, cost):
        # Sources and destinations are sorted by span. Prefix/suffix sums
        # exclude the contiguous equal-span window without subtracting log masses.
        spans, query = graph.span[src], graph.span[dst]
        prefix = np.r_[-np.inf, np.logaddexp.accumulate(values)]
        suffix = np.r_[np.logaddexp.accumulate(values[::-1])[::-1], -np.inf]
        lo = np.searchsorted(spans, query - 1e-4, side="right")
        hi = np.maximum(lo, np.searchsorted(spans, query + 1e-4, side="left"))
        equal = np.full(len(dst), -np.inf)
        single = hi == lo + 1
        equal[single] = values[lo[single]]
        if np.any(hi > lo + 1):
            # Two-stack aggregate queue keeps even overlapping tolerance windows
            # linear, without cancellation from subtracting nearly equal masses.
            front, back = [], []
            left = right = 0
            for j in range(len(dst)):
                while right < hi[j]:
                    v = values[right]
                    back.append((v, np.logaddexp(v, back[-1][1]) if back else v))
                    right += 1
                while left < lo[j]:
                    if not front:
                        while back:
                            v = back.pop()[0]
                            front.append(np.logaddexp(v, front[-1]) if front else v)
                    front.pop()
                    left += 1
                equal[j] = np.logaddexp(front[-1] if front else -np.inf, back[-1][1] if back else -np.inf)
        equal += same_reward
        total = np.logaddexp(equal, np.logaddexp(prefix[lo], suffix[hi]) + cost)
        return total, equal

    same_mass = np.zeros(len(score)) if gradients else None
    for i in range(1, len(graph.nodes)-1):
        inc = np.arange(graph.in_offsets[i+1]-1, graph.in_offsets[i]-1, -1)
        out = graph.outgoing(i)
        if len(inc) and len(out):
            total, equal = sums(inc, out, alpha[inc], switch[i])
            alpha[out] = score[out] + total
            if gradients:
                ok = np.isfinite(total)
                same_mass[out[ok]] = np.exp(equal[ok]-total[ok])
    final = graph.incoming(len(graph.nodes) - 1)
    z = logsumexp(alpha[final])
    if not gradients:
        return z, None
    # Exact reverse pass: propagate posterior flow conditional on each outgoing state.
    marginal = np.zeros_like(alpha)
    marginal[final] = np.exp(alpha[final] - z)
    d_switch = np.zeros(len(graph.nodes))
    d_same = 0.0
    beta = np.full_like(alpha, -np.inf)
    beta[final] = score[final]
    for i in range(len(graph.nodes)-2, 0, -1):
        inc = np.arange(graph.in_offsets[i+1]-1, graph.in_offsets[i]-1, -1)
        out = graph.outgoing(i)
        if len(inc) and len(out):
            total, _ = sums(out, inc, beta[out], switch[i])
            marginal[inc] = np.exp(alpha[inc]-z+total)
            beta[inc] = total + score[inc]
        equal = marginal[out] * same_mass[out]
        d_same += equal.sum()
        d_switch[i] = (marginal[out]-equal).sum()
    return z, (marginal, d_switch, d_same)


class _LatticeLoss(torch.autograd.Function):
    """One score download and one loss/marginal upload across all song pieces."""

    @staticmethod
    def forward(ctx, packed, terms, gradients):
        values = packed.detach().cpu().double().numpy()
        response = np.zeros(len(values) + 1 if gradients else 1)
        same = values[0]
        cursor = 1
        use_compiled = require_numba("Semi-CRF partition_scan") and partition_scan is not None
        jobs = []
        for graph, gold_edges, weight in terms:
            edges = slice(cursor, cursor + len(graph.span))
            nodes = slice(edges.stop, edges.stop + len(graph.nodes))
            cursor = nodes.stop
            jobs.append((graph, gold_edges, weight, edges, nodes))
        def evaluate(job):
            graph, gold_edges, weight, edges, nodes = job
            score, switch = values[edges], values[nodes]
            marginal = d_switch = None
            d_same = 0.
            if not use_compiled:
                loss, grads = _scan(graph, score, switch, same, gradients=gradients)
                if gradients:
                    marginal, d_switch, d_same = grads
            else:
                loss, marginal, d_switch, d_same = partition_scan(
                    graph.span, graph.in_offsets, graph.out_offsets, graph.out_order,
                    score, switch, same, gradients,
                )
            if gold_edges is not None:
                equal = np.abs(np.diff(graph.span[gold_edges])) < 1e-4
                changed = graph.starts[gold_edges[1:][~equal]]
                loss -= score[gold_edges].sum() + equal.sum() * same + switch[changed].sum()
                if gradients:
                    marginal[gold_edges] -= 1
                    d_switch[changed] -= 1
                    d_same -= equal.sum()
            return loss, marginal, d_switch, d_same
        results = _SCAN_POOL.map(evaluate, jobs) if use_compiled and len(jobs) >= 2 else map(evaluate, jobs)
        for (_, _, weight, edges, nodes), (loss, marginal, d_switch, d_same) in zip(jobs, results):
            response[-1] += weight * loss
            if gradients:
                response[0] += weight * d_same
                response[edges] = weight * marginal
                response[nodes] = weight * d_switch
        result = torch.from_numpy(response).to(dtype=packed.dtype).to(packed.device, non_blocking=True)
        if gradients:
            ctx.save_for_backward(result[:-1])
        return result[-1]

    @staticmethod
    def backward(ctx, scale):
        return ctx.saved_tensors[0] * scale, None, None


def _device_array(values, device, dtype=None):
    return torch.as_tensor(values, dtype=dtype, device="cpu").to(device, non_blocking=True)


def _phase_cosine(graph, frames, axis, unit, length, *, interior=False):
    """Prefix sums over just the union of edge intervals at each frequency.

    Regular eighth grids reuse the same 24 frequencies throughout the song.
    With arbitrary BMS bar lengths, a frequency can instead occur on one edge.
    Flattening those unions keeps both cases in one vectorized computation.
    """
    n = len(axis)
    spans, group = np.unique(np.round(graph.span, 5), return_inverse=True)
    left = group * n + frames[graph.starts] + int(interior)
    right = group * n + frames[graph.ends]
    order = np.lexsort((left, group))
    a, b = left[order], right[order]
    frontier = np.r_[a[0], np.maximum.accumulate(b)[:-1]]
    gap = np.maximum(0, a - frontier)
    skipped = a[0] + gap.cumsum()
    # Separate ranges in this concatenation have the same added prefix at
    # both endpoints of an edge, so it cancels in the interval difference.
    run = np.r_[0, np.flatnonzero(gap[1:]) + 1]
    run_end = np.maximum.reduceat(b, run)
    size = run_end - a[run]
    offsets = np.r_[0, size.cumsum()[:-1]]
    device = axis.device

    def index(x):
        return torch.from_numpy(x).to(device, non_blocking=True)

    total = int(size.sum())
    packed = torch.arange(total, device=device) + torch.repeat_interleave(
        index(a[run] - offsets), index(size), output_size=total
    )
    times = packed % n
    freq = _device_array(2 * math.pi / spans, device, axis.dtype)
    angle = axis[times] * freq[packed // n]
    sin, cos = angle.sin(), angle.cos()
    u = unit[times]
    rotated = torch.stack(
        (u[:, 1] * cos + u[:, 0] * sin, u[:, 0] * cos - u[:, 1] * sin), -1
    )
    prefix = F.pad(rotated.cumsum(0), (0, 0, 1, 0))
    skip = np.empty_like(skipped)
    skip[order] = skipped
    sums = prefix[index(right - skip)] - prefix[index(left - skip)]
    origin = _device_array(graph.nodes[graph.starts], device, axis.dtype) * freq[index(group)]
    return (sums[:, 0] * origin.cos() - sums[:, 1] * origin.sin()) / length


class BarSemiCRFHead(nn.Module):
    def __init__(self, width=384):
        super().__init__()
        self.project = nn.Linear(width, 24)
        self.span_embedding = nn.Embedding(50, 8)
        self.span_score = nn.Sequential(
            nn.Linear(3 * 24 + 8 + 5, 48), nn.SiLU(), nn.Linear(48, 1)
        )
        self.endpoint_weight = nn.Parameter(torch.ones(()))
        self.same_reward = nn.Parameter(torch.zeros(()))
        self.switch_cost = nn.Parameter(torch.zeros(()))
        self.covar_cost = nn.Parameter(torch.zeros(()))

    def score(self, graph, axis, frames, features, bar_logit, phase, *, frame_indices=None):
        device = features.device
        start, end = (
            _device_array(v, device) for v in (graph.starts, graph.ends)
        )
        span = _device_array(graph.span, device, torch.float32)
        f = _device_array(frames, device) if frame_indices is None else frame_indices
        a, b = f[start], f[end]
        projected = self.project(features)
        prefix = F.pad(projected.float().cumsum(0), (0, 0, 1, 0))
        length = (b - a).clamp_min(1)
        mean = (prefix[b] - prefix[a]) / length[:, None]
        unit = F.normalize(phase.float(), dim=-1)
        mag = F.pad(phase.float().norm(dim=-1).cumsum(0), (1, 0))
        cosine = _phase_cosine(graph, frames, axis, unit, length)
        quarters = span * 4
        is_quarter = (quarters - quarters.round()).abs() < 1e-4
        is_half = (span * 2 - (span * 2).round()).abs() < 1e-4
        classes = torch.where(is_quarter, quarters.round().long().clamp(0, 48), 49)
        geometry = torch.stack(
            (cosine, (mag[b] - mag[a]) / length, span / 12, is_quarter, is_half), -1
        )
        vectors = torch.cat(
            (projected[a], projected[b], mean, self.span_embedding(classes), geometry),
            -1,
        )
        return (
            self.endpoint_weight * bar_logit[b].float()
            + self.span_score(vectors).squeeze(-1).float()
        )

    def partition(self, graph, score, change):
        return self.lattice_loss([(graph, score, change, None, 1.0)])

    def nll(self, graph, score, change, gold):
        edges = graph.gold_edges(gold)
        return self.lattice_loss([(graph, score, change, edges, 1.0 / len(edges))])

    def lattice_loss(self, lattices):
        tensors = [self.same_reward.reshape(1)]
        terms = []
        for graph, score, change, edges, weight in lattices:
            tensors.extend((score, -self.switch_cost - self.covar_cost * change))
            terms.append((graph, edges, weight))
        packed = torch.cat(tensors)
        return _LatticeLoss.apply(packed, terms, packed.requires_grad)

    @torch.no_grad()
    def decode(self, graph, score, change):
        switch = -self.switch_cost - self.covar_cost * change
        path = _scan(
            graph,
            score.double().cpu().numpy(),
            switch.double().cpu().numpy(),
            self.same_reward.item(),
            viterbi=True,
        )
        return graph.nodes[path]


def score_lattice(
    outputs, batch_index, axis, nodes, bar_logits, *, inference=False, recoveries=()
):
    graph = BarLattice(nodes)
    head = outputs["semicrf"]
    n = len(axis)
    device = outputs["boundary_features"].device
    seconds = (np.arange(n) + 0.5) / 125
    times = _node_times(nodes, axis, seconds)
    frames = np.floor(times * 125 + 1e-7).astype(int).clip(0, n - 1)
    # Choose the same frame as the event target on exact frame edges.
    logits = outputs["downbeat_logit"][batch_index, :n]
    prev = np.maximum(frames - 1, 0)
    tie = abs(times * 125 - frames) < 1e-4
    frames = np.where(tie & (bar_logits[prev] >= bar_logits[frames]), prev, frames)
    if inference:
        # 125 Hz quantization and subframe predictions leave a genuine peak
        # one or two frames from its clock tick. Use that same attack in the head.
        offsets = np.arange(-2, 3)
        window = (frames[:, None] + offsets).clip(0, n - 1)
        frames = window[np.arange(len(frames)), bar_logits[window].argmax(1)]
    frame_indices = _device_array(frames, device)
    score = head.score(
        graph,
        _device_array(axis, device, torch.float32),
        frames,
        outputs["boundary_features"][batch_index, :n],
        logits,
        outputs["bar_phase"][batch_index, :n],
        frame_indices=frame_indices,
    )
    change = outputs["change_logit"][batch_index, frame_indices].float().sigmoid()
    if inference:
        left, right = graph.starts, graph.ends
        bar = bar_logits[frames]
        # Prefix sums exclude both endpoints of the edge.
        # Count an observed peak once when a local and a global tick fall in
        # its frame window. Multiple nearby hypotheses describe the same event.
        unique, index = np.unique(frames, return_inverse=True)
        evidence = np.logaddexp(0., bar_logits[unique] + 1.)
        prefix = np.r_[0., evidence.astype(float).cumsum()]
        skipped = np.maximum(0., prefix[index[right]] - prefix[index[left]+1])
        beat = outputs["beat_logit"][batch_index, :n].detach().float().cpu().numpy()
        beat = maximum_filter1d(beat, size=5)

        def support(coords):
            return np.interp(coords, axis, beat).clip(-6, 3)

        total = support(nodes[left]) + support(nodes[right])
        count = np.full(len(left), 2)
        for step in range(1, 8):
            inside = step < graph.span - 1e-5
            total[inside] += support(nodes[left[inside]] + step)
            count[inside] += 1
        valid = np.isin(
            graph.span.round(5), (1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 7.5, 8)
        )
        accented = (expit(bar[left]) >= 0.55) & (expit(bar[right]) >= 0.55)
        valid |= accented & np.isin(graph.span.round(5), (0.25, 0.5, 0.75))
        phase_shift = abs(2*graph.span - np.round(2*graph.span)) > 1e-5
        for recovery in recoveries:
            # Enter or leave the shifted grid at an observed bar adjacent to
            # the recovery; the tempo boundary itself can lie inside a bar.
            crosses = (nodes[left] < recovery + 1e-5) & (
                nodes[right] >= recovery - 1e-5
            )
            valid |= accented & crosses & phase_shift & (graph.span <= 8)
        # A run of equally spaced, prominent downbeats supports its bar length
        # even if a syncopation inside one bar briefly looks like another boundary.
        # Use the weaker of two independent flanking observations. A single
        # coincident accent cannot give a span this continuity bonus.
        peak_bar = maximum_filter1d(bar_logits, size=5)
        previous = np.interp(
            nodes[left] - graph.span, axis, peak_bar, left=-12, right=-12
        )
        following = np.interp(
            nodes[right] + graph.span, axis, peak_bar, left=-12, right=-12
        )
        repeated = np.minimum.reduce([previous, bar[left], bar[right], following])
        adjustment = (
            1.3 - 0.75 * skipped + 0.35 * total / count + np.clip(repeated, 0, 3)
        )
        score = score + _device_array(
            np.where(valid, adjustment, -np.inf), device, score.dtype
        )
    return graph, score, change


def sequence_loss(outputs, targets):
    lattices = []
    predictions = torch.stack(
        (outputs["downbeat_logit"].detach(), outputs["subframe_offset_ms"].detach()), dim=1
    ).float().cpu().numpy()
    for batch, n in enumerate(outputs["lengths"].tolist()):
        labels = targets["gold_bar"][batch, :n].detach().cpu().numpy()
        gold = labels[np.isfinite(labels)].astype(np.float64)
        if len(gold) < 2:
            continue
        axis = targets["score_beat"][batch, :n].detach().cpu().double().numpy()
        seconds = (np.arange(n) + 0.5) / 125
        logits, sub = predictions[batch, :, :n]
        # A clipped window or an out-of-domain interval supplies no whole legal
        # gold edge there. Both neighboring runs still contribute exact NLL.
        span = np.diff(gold)
        pieces = np.split(
            gold, 1 + np.flatnonzero((span < 0.25 - 1e-5) | (span > 12 + 1e-5))
        )
        count = sum(len(p) - 1 for p in pieces)
        for piece in pieces:
            if len(piece) < 2:
                continue
            nodes = candidate_nodes(
                axis, seconds, logits, sub, piece[0], piece[-1], piece
            )
            graph, score, change = score_lattice(outputs, batch, axis, nodes, logits)
            lattices.append((graph, score, change, graph.gold_edges(piece), n / count))
    if not lattices:
        return outputs["log_bpm"].new_zeros(())
    return outputs["semicrf"].lattice_loss(lattices)


def _node_phase_cosine(graph, coords, bar_phase):
    if not len(graph.span):
        return coords.new_empty(0)
    count = _device_array((graph.ends - graph.starts - 1).clip(1), coords.device)
    return _phase_cosine(graph, np.arange(len(coords)), coords, F.normalize(bar_phase.float(), dim=-1),
                         count, interior=True)


class ContextualBarSemiCRF(nn.Module):
    """Node-conditioned segment potentials shared by partition and Viterbi."""

    def __init__(self, *, v1=False):
        super().__init__()
        self.v1 = v1
        self.node = nn.Linear(256, 32)
        self.semantic = nn.Linear(256, 16)
        self.span_embedding = nn.Embedding(49, 12)
        self.boundary = nn.Linear(256, 1)
        self.switch = nn.Sequential(nn.Linear(256, 32), nn.SiLU(), nn.Linear(32, 1))
        if not v1:
            nn.init.zeros_(self.switch[-1].weight)
            nn.init.zeros_(self.switch[-1].bias)
        self.edge = nn.Sequential(nn.Linear(128 if v1 else 64, 64), nn.SiLU(), nn.Linear(64, 2))

    def bar_logits(self, song):
        z = song["z"].to(self.boundary.weight.dtype)
        return self.boundary(z).squeeze(-1).float() + song["bar"]

    def potentials(self, graph, song):
        z = song["z"].to(self.node.weight.dtype)
        device = z.device
        a, b = (_device_array(x, device) for x in (graph.starts, graph.ends))
        coords = _device_array(graph.nodes, device, torch.float)
        span = coords[b] - coords[a]
        node = self.node(z)

        def mean(x):
            prefix = F.pad(x.float().cumsum(0), (0, 0, 1, 0))
            if not self.v1:
                return (prefix[b] - prefix[a+1]) / (b-a-1).clamp_min(1)[:, None]
            return (prefix[b + 1] - prefix[a]) / (b - a + 1)[:, None]

        # Gather internal simple/compound accents by score distance. Missing
        # pulses carry no support. Never use a frame or label from the gold grid.
        support = []
        for spacing in (1.0, 1.5):
            steps = torch.arange(spacing, 12, spacing, device=device)
            query = coords[a, None] + steps
            ix = torch.searchsorted(coords, query).clamp_max(len(coords) - 1)
            hit = (query < coords[b, None] - 0.01) & ((coords[ix] - query).abs() < 0.01)
            support.append((song["tactus"][ix].sigmoid() * hit).sum(1) / hit.sum(1).clamp_min(1))
        if self.v1:
            phase = F.normalize(song["bar_phase"].float(), dim=-1)
            cosine = (phase[a] * phase[b]).sum(-1)
        else:
            cosine = _node_phase_cosine(graph, coords, song["bar_phase"])
        geometry = torch.stack((*support, cosine, span / 12), -1)
        vec = torch.cat(
            (
                *((node[a], node[b]) if self.v1 else ()),
                mean(node),
                (song["sem_integral"][b] - song["sem_integral"][a])
                / (song["frames"][b] - song["frames"][a]).clamp_min(1)[:, None],
                self.span_embedding((span * 4).round().long().clamp(0, 48)),
                geometry,
            ),
            -1,
        )
        edge, compound = self.edge(vec).float().unbind(-1)
        if not self.v1:
            edge = 2 * (edge / 2).tanh()
        bar = self.bar_logits(song)
        prefix = F.pad(F.logsigmoid(-bar).cumsum(0), (1, 0))
        score = F.logsigmoid(bar[b]) + prefix[b] - prefix[a + 1] + edge
        raw = self.switch(z).squeeze(-1).float()
        switch = -F.softplus(raw) - .05 if self.v1 else -(.25 + 2.25 * raw.sigmoid())
        return score, switch, compound

    @staticmethod
    def nll(graph, potentials, gold_ids):
        return ContextualBarSemiCRF.path_nll([(graph, potentials, gold_ids)]) / (len(gold_ids) - 1)

    @staticmethod
    def path_nll(paths):
        tensors = [paths[0][1][0].new_zeros(1)]
        terms = []
        for graph, (scores, switch, _), gold_ids, *weight in paths:
            tensors.extend((scores, switch))
            terms.append((graph, graph.gold_edges(graph.nodes[gold_ids]), weight[0] if weight else 1.0))
        packed = torch.cat(tensors)
        return _LatticeLoss.apply(packed, terms, packed.requires_grad)

    @staticmethod
    def decode(graph, potentials, *, boundary_logits=None, return_score=False):
        score, switch, _ = potentials
        partial = None
        if boundary_logits is not None:
            logits = boundary_logits.detach().double().cpu().numpy()
            skip = -np.logaddexp(0, logits)
            # Free pickup and final fragment: charge the same Bernoulli skip
            # cost as complete edges and count the first endpoint exactly once.
            partial = (
                np.r_[0.0, skip.cumsum()[:-1]] - np.logaddexp(0, -logits),
                np.r_[skip[:0:-1].cumsum()[::-1], 0.0],
            )
        use_compiled = require_numba("Semi-CRF viterbi_scan")
        score_np = score.detach().double().cpu().numpy()
        switch_np = switch.detach().double().cpu().numpy()
        if not use_compiled:
            result = _scan(graph, score_np, switch_np, 0.0, viterbi=True, open_ends=partial, return_score=return_score)
            if return_score:
                path, value = result
                return np.asarray(path, dtype=np.int64), value
            return np.asarray(result, dtype=np.int64)
        ends = partial if partial is not None else (np.empty(0), np.empty(0))
        path, value = viterbi_scan(graph.span, graph.in_offsets, graph.out_offsets, graph.out_order,
                                   score_np, switch_np, 0., partial is not None, *ends, graph.nodes)
        return (path, value) if return_score else path
