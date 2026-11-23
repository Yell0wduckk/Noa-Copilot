"""Phase-constrained whole-chart segmentation. All clocks are integer tick96."""

from functools import lru_cache
import numpy as np
from .codec import PAD_MAP, grouped_geometry, vocabulary


@lru_cache(maxsize=8192)
def orbit(chord):
    return tuple(tuple(sorted(table[p] for p in chord)) for table in PAD_MAP)


@lru_cache(maxsize=65536)
def signature(a, b, c):
    return min(zip(orbit(a), orbit(b), orbit(c)))


def simpler(chords, shifts):
    for step in shifts:
        if any(
            all(orbit(a)[g] == b for a, b in zip(chords, chords[step:]))
            for g in range(16)
        ):
            return True
    return False


def phase_edges(ticks, chords):
    n = len(ticks)
    allowed = np.ones((n, 4), bool)

    def cut(point):
        for i in range(max(0, point - 3), point):
            allowed[i, point - i :] = False

    cut(n)
    i = 0
    while i + 1 < n:
        gap = ticks[i + 1] - ticks[i]
        end = i + 2
        while end < n and ticks[end] - ticks[end - 1] == gap:
            end += 1
        if gap in (8, 12, 16, 24, 32) and end - i >= 3:
            cut(i)
            cut(end)
        pos = i
        while gap in (12, 16, 24, 32) and pos + 6 <= end:
            pattern = signature(*chords[pos : pos + 3])
            stop = pos + 3
            while stop + 3 <= end and signature(*chords[stop : stop + 3]) == pattern:
                stop += 3
            run = chords[pos:stop]
            accept = len(run) >= 6 and not simpler(run, (1, 2))
            if accept and len(run) == 6:
                accept = ((pos - i) % 4 == 0 or ticks[pos] % 96 == 0) and not any(
                    simpler(run[j : j + 4], (1,)) for j in (0, 2)
                )
            elif accept:
                accept = not simpler(run[3:], (1,))
            if accept:
                cut(pos)
                cut(stop)
                for j in range(pos, stop):
                    triple = allowed[j, 2] and (j - pos) % 3 == 0
                    allowed[j, 1:] = False
                    allowed[j, 2] = triple
                pos = stop
            else:
                pos += 1
        i = end - 1
    return allowed


def lattice(moments):
    ticks = [t for t, _ in moments]
    chords = [grouped_geometry(notes) for _, notes in moments]
    allowed = phase_edges(ticks, chords)
    ids = np.full((len(ticks), 4), -1, np.int16)
    orientations = np.zeros_like(ids, dtype=np.int8)
    book = vocabulary()
    for i, t in enumerate(ticks):
        key = []
        for j in range(i, min(i + 4, len(ticks))):
            dt = ticks[j] - t
            if dt >= 96:
                break
            key.append((dt, chords[j]))
            k = j - i
            if allowed[i, k]:
                match = book.lookup.get(tuple(key))
                if match is not None:
                    ids[i, k], orientations[i, k] = match
                elif not k:
                    ids[i, k] = 0
    return ids, orientations


def gap_scores(ticks, prior):
    if not len(ticks):
        return np.zeros(0)
    gaps = np.diff(ticks, prepend=ticks[0])
    scores = np.array([prior.get(int(g), prior[-1]) for g in gaps])
    scores += 0.25 * (np.asarray(ticks) % 96 == 0)
    scores[0] = 0
    return scores


def viterbi(ids, orientation, ticks, weights, gaps, *, rng=None, dropout=0.0):
    n = len(ids)
    best = np.full(n + 1, -np.inf)
    best[0] = 0
    back = np.ones(n + 1, np.int8)
    score = weights[ids] + gaps[:, None]
    score[ids < 0] = -np.inf
    if rng is not None and dropout:
        score[:, 1:] = np.where(rng.random((n, 3)) < dropout, -np.inf, score[:, 1:])
    for j in range(1, n + 1):
        ks = np.arange(1, min(4, j) + 1)
        values = best[j - ks] + score[j - ks, ks - 1]
        k = int(np.argmax(values))
        best[j] = values[k]
        back[j] = ks[k]
    out = []
    j = n
    while j:
        k = int(back[j])
        j -= k
        out.append([j, k, int(ids[j, k - 1]), int(orientation[j, k - 1])])
    return out[::-1]


def estimate(records, rounds=6):
    """Vectorized forward/backward over similarly sized charts; fit split only."""
    from collections import Counter

    gap_count = Counter()
    size = vocabulary().size
    counts = np.zeros(size)
    for ticks, ids in records:
        gap_count.update(map(int, np.diff(ticks)))
        counts += np.bincount(ids[ids >= 0], minlength=size)
    denom = sum(gap_count.values()) + 1
    prior = {g: float(np.log(n / denom)) for g, n in gap_count.items()}
    prior[-1] = -float(np.log(denom))
    placement = -np.log(np.sum(vocabulary().valid, axis=1))
    logp = np.log((counts + 0.1) / (counts.sum() + 0.1 * size))
    buckets = []
    records = sorted(records, key=lambda x: len(x[0]))
    for first in range(0, len(records), 128):
        block = records[first : first + 128]
        lengths = np.array([len(t) for t, _ in block])
        ids = np.full((len(block), max(lengths), 4), -1, np.int16)
        gaps = np.zeros(ids.shape[:2])
        for i, (ticks, row) in enumerate(block):
            ids[i, : len(ticks)] = row
            gaps[i, : len(ticks)] = gap_scores(ticks, prior)
        buckets.append((ids, gaps, lengths))
    for _ in range(rounds):
        counts.fill(0)
        for ids, gaps, lengths in buckets:
            b, n, _ = ids.shape
            rows = np.arange(b)
            scores = np.where(
                ids >= 0, (logp + placement)[ids] + gaps[:, :, None], -np.inf
            )
            forward = np.full((b, n + 1), -np.inf)
            forward[:, 0] = 0
            for j in range(1, n + 1):
                k = np.arange(1, min(4, j) + 1)
                forward[:, j] = np.logaddexp.reduce(
                    forward[:, j - k] + scores[:, j - k, k - 1], axis=1
                )
            back = np.full_like(forward, -np.inf)
            back[rows, lengths] = 0
            for j in reversed(range(n)):
                k = np.arange(1, min(4, n - j) + 1)
                back[:, j] = np.where(
                    j == lengths,
                    0,
                    np.logaddexp.reduce(scores[:, j, k - 1] + back[:, j + k], axis=1),
                )
            total = forward[rows, lengths, None]
            for k in range(1, min(n, 4) + 1):
                stop = n - k + 1
                index = ids[:, :stop, k - 1]
                mask = index >= 0
                post = forward[:, :stop] + scores[:, :stop, k - 1] + back[:, k:] - total
                counts += np.bincount(
                    index[mask], weights=np.exp(post[mask]), minlength=size
                )
        logp = np.log((counts + 0.1) / (counts.sum() + 0.1 * size))
    return logp + placement, prior
