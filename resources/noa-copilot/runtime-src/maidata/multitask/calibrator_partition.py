"""Compiled sum-product scan for equal-span versus changed-span transitions."""

import os
import warnings

import numpy as np
try:
    from numba import njit
except Exception as exc:  # Compiler may be absent or incompatible with the installed NumPy version.
    NUMBA_ERROR = str(exc)

    def njit(**kwargs):
        return lambda function: function
else:
    NUMBA_ERROR = None

_allow_fallback = False
_warned_fallback = False


def allow_numba_fallback(enabled=None):
    """Query or explicitly toggle permission to fall back when Numba is unavailable."""
    global _allow_fallback
    if enabled is not None:
        _allow_fallback = bool(enabled)
        return _allow_fallback
    return _allow_fallback or os.environ.get("MAIDATA_ALLOW_NO_NUMBA", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def require_numba(context="clean lattice"):
    """Fail fast unless Numba compiled cleanly or fallback was explicitly enabled."""
    global _warned_fallback
    if NUMBA_ERROR is None:
        return True
    if not allow_numba_fallback():
        raise RuntimeError(
            f"Numba is required for {context}, but could not be loaded ({NUMBA_ERROR}). "
            "Install a Numba version compatible with the active NumPy version, "
            "or explicitly allow the slower NumPy fallback with --allow-no-numba "
            "(or MAIDATA_ALLOW_NO_NUMBA=1)."
        )
    if not _warned_fallback:
        warnings.warn(
            f"Numba unavailable ({NUMBA_ERROR}); using explicit fallback for {context}.",
            RuntimeWarning,
        )
        _warned_fallback = True
    return False


@njit(nogil=True, cache=True)
def _transition_sums(span, sources, targets, values, same, switch, totals, same_prob):
    """Equal spans form a contiguous window in the sorted source edges.

    The prefix before and suffix after the window contain all changed spans.
    Source and target edges are disjoint, so totals may alias values.
    """
    prefix = np.empty(len(sources) + 1)
    suffix = np.empty_like(prefix)
    prefix[0] = suffix[-1] = -np.inf
    for index in range(len(sources)):
        prefix[index + 1] = np.logaddexp(prefix[index], values[sources[index]])
        reverse = len(sources) - 1 - index
        suffix[reverse] = np.logaddexp(suffix[reverse + 1], values[sources[reverse]])

    left = 0
    for edge in targets:
        target = span[edge]
        while left < len(sources) and target - span[sources[left]] >= 1e-4:
            left += 1
        right = left
        matched = -np.inf
        while right < len(sources) and abs(target - span[sources[right]]) < 1e-4:
            matched = np.logaddexp(matched, values[sources[right]])
            right += 1
        matched += same
        changed = np.logaddexp(prefix[left], suffix[right]) + switch
        total = np.logaddexp(matched, changed)
        totals[edge] = total
        if len(same_prob) and np.isfinite(total):
            same_prob[edge] = np.exp(matched - total)


@njit(nogil=True, cache=True)
def partition_scan(span, incoming, outgoing, order, score, switch, same, gradients):
    """Return log Z and its derivatives for one ragged lattice in float64."""
    edges = np.arange(len(span))
    alpha = np.full(len(edges), -np.inf)
    first = order[outgoing[0]:outgoing[1]]
    alpha[first] = score[first]
    same_prob = np.zeros(len(edges) if gradients else 0)
    for node in range(1, len(incoming) - 2):
        inc = edges[incoming[node]:incoming[node + 1]][::-1]
        out = order[outgoing[node]:outgoing[node + 1]]
        _transition_sums(span, inc, out, alpha, same, switch[node], alpha, same_prob)
        alpha[out] += score[out]

    final = edges[incoming[-2]:]
    log_z = -np.inf
    for edge in final:
        log_z = np.logaddexp(log_z, alpha[edge])
    marginal = np.empty(len(edges) if gradients else 0)
    d_switch = np.zeros(len(switch) if gradients else 0)
    d_same = 0.0
    if gradients:
        beta = np.full(len(edges), -np.inf)
        beta[final] = score[final]
        marginal[final] = np.exp(alpha[final] - log_z)
        empty = np.empty(0)
        for node in range(len(incoming) - 3, 0, -1):
            inc = edges[incoming[node]:incoming[node + 1]][::-1]
            out = order[outgoing[node]:outgoing[node + 1]]
            _transition_sums(span, out, inc, beta, same, switch[node], beta, empty)
            marginal[inc] = np.exp((alpha[inc] - log_z) + beta[inc])
            beta[inc] += score[inc]
        for node in range(1, len(incoming) - 2):
            for edge in order[outgoing[node]:outgoing[node + 1]]:
                mass = marginal[edge]
                d_same += mass * same_prob[edge]
                d_switch[node] += mass * (1.0 - same_prob[edge])
    return log_z, marginal, d_switch, d_same

@njit(nogil=True, cache=True)
def deduplicate_peaks(peaks, strength, radius):
    keep = np.ones(len(peaks), dtype=np.bool_)
    previous = -1
    for i in range(len(peaks)):
        f = peaks[i]
        if previous >= 0 and f - peaks[previous] < radius[i]:
            if strength[f] <= strength[peaks[previous]]:
                keep[i] = False
                continue
            keep[previous] = False
        previous = i
    return peaks[keep]


@njit(nogil=True, cache=True)
def _better(alpha, a, b):
    if b < 0 or (a >= 0 and (alpha[a] > alpha[b] or (alpha[a] == alpha[b] and a < b))):
        return a
    return b


@njit(nogil=True, cache=True)
def viterbi_scan(span, incoming, outgoing, order, score, switch, same, has_open, leading, trailing, nodes):
    """Linear transition maxima in span order, with first-edge tie breaking."""
    n = len(nodes)
    parent = np.full(len(span), -1, np.int64)
    alpha = np.full(len(span), -np.inf)
    starts = np.empty(len(span), np.int64)
    ends = np.empty_like(starts)
    for i in range(n):
        starts[order[outgoing[i]:outgoing[i+1]]] = i
        ends[incoming[i]:incoming[i+1]] = i
    for e in range(len(span)):
        a = starts[e]
        if has_open and nodes[a] - nodes[0] < span[e]:
            alpha[e] = score[e] + leading[a]
        elif not has_open and a == 0:
            alpha[e] = score[e]
    for i in range(1, n-1):
        first, last = incoming[i], incoming[i+1]
        size = last-first
        suffix = np.full(size+1, -1, np.int64)
        queue = np.empty(size, np.int64)
        for j in range(size-1, -1, -1):
            suffix[j] = _better(alpha, last-1-j, suffix[j+1])
        low = high = head = tail = 0
        prefix = -1
        for e in order[outgoing[i]:outgoing[i+1]]:
            while high < size and span[last-1-high] - span[e] < 1e-4:
                k = last-1-high
                while tail > head and _better(alpha, k, queue[tail-1]) == k:
                    tail -= 1
                queue[tail] = k
                tail += 1
                high += 1
            while low < high and span[e] - span[last-1-low] >= 1e-4:
                k = last-1-low
                prefix = _better(alpha, k, prefix)
                if head < tail and queue[head] == k:
                    head += 1
                low += 1
            other = _better(alpha, prefix, suffix[high])
            matched = queue[head] if head < tail else -1
            v = alpha[other] + switch[i] if other >= 0 else -np.inf
            w = alpha[matched] + same if matched >= 0 else -np.inf
            if w > v or (w == v and matched >= 0 and (other < 0 or matched < other)):
                other, v = matched, w
            if score[e] + v > alpha[e]:
                alpha[e] = score[e] + v
                parent[e] = other

    best = -1
    value = -np.inf
    last_reachable = -1
    eligible = False
    for e in range(len(span)):
        if not np.isfinite(alpha[e]):
            continue
        b = ends[e]
        ok = nodes[-1] - nodes[b] < span[e]
        candidate = alpha[e] + trailing[b] if has_open else alpha[e]
        if has_open:
            if eligible and not ok:
                continue
            reset = ok and not eligible
            eligible = eligible or ok
        else:
            reset = b != last_reachable
        if reset or candidate > value:
            best, value = e, candidate
        last_reachable = b
    if best < 0:
        return np.zeros(1, np.int64), -np.inf
    path = [ends[best]]
    while best >= 0:
        path.append(starts[best])
        best = parent[best]
    return np.array(path[::-1]), value


@njit(nogil=True, cache=True)
def tempo_dp_scan(x, y, penalty):
    n = len(x)
    sums = np.zeros((5, n+1))
    for j in range(n):
        values = (x[j], y[j], x[j]*x[j], x[j]*y[j], y[j]*y[j])
        for k in range(5):
            sums[k, j+1] = sums[k, j] + values[k]
    best = np.zeros(n)
    parent = np.zeros(n, np.int64)
    for b in range(1, n):
        best[b] = np.inf
        for a in range(b):
            m = b-a+1
            sx = sums[0,b+1] - sums[0,a]
            sy = sums[1,b+1] - sums[1,a]
            xx = max(1e-12, sums[2,b+1]-sums[2,a]-sx*sx/m)
            xy = sums[3,b+1]-sums[3,a]-sx*sy/m
            error = max(0., sums[4,b+1]-sums[4,a]-sy*sy/m-xy*xy/xx)
            value = best[a] + penalty[a] + error
            if b > 1 and a == b-1:
                value += .012
            if value < best[b]:
                best[b], parent[b] = value, a
    return parent
