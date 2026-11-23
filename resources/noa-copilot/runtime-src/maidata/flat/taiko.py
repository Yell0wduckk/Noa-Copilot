"""GRID32 single-player Taiko events, grammar and acoustic supervision."""

import math
from functools import lru_cache

import numpy as np

from .codec import (
    GRID32_TICKS,
    TICK_TO_GRID32,
    SIZE_V20,
    TAIKO_EVENT,
    TAIKO_LONG_SPAN,
    TAIKO_END_BEAT,
    TAIKO_END_OFFSET,
)
from .descriptors import HeadRows, active_spans
from .emotion import stabilize_high_plateau_preserve_drops
from .reconstruction import rhythm_targets

KINDS = (1, 2, 3, 4, 5, 6, 7, 9)


def taiko_tokens(events, horizon):
    yield 0
    beat = -1
    for tick, kind, end, _ in events:
        while beat < tick // 96:
            beat += 1
            yield 1 + beat % 16
        yield 17 + TICK_TO_GRID32[tick % 96]
        yield TAIKO_EVENT + KINDS.index(kind)
        if kind >= 5:
            for _ in range((end // 96 - beat) // 16):
                yield TAIKO_LONG_SPAN
            yield TAIKO_END_BEAT + end // 96 % 16
            yield TAIKO_END_OFFSET + TICK_TO_GRID32[end % 96]
    while beat < horizon // 96:
        beat += 1
        yield 1 + beat % 16
    yield 0


class TaikoGrammar:
    def __init__(self, onset_limit, field_limit):
        self.onset_limit, self.field_limit = onset_limit, field_limit
        self.phase = "start"
        self.beat, self.last_tick = -1, -1
        self.tick = self.query_tick = self.minimum = self.pages = 0
        self.end_beat = self.kind = 0

    @property
    def page(self):
        return self.beat + self.pages * 16

    def legal(self):
        if self.phase == "start":
            return (0,)
        if self.phase == "done":
            return ()
        if self.phase == "timing":
            choices = [
                17 + i
                for i, offset in enumerate(GRID32_TICKS)
                if self.minimum <= self.beat * 96 + offset <= self.onset_limit
            ]
            choices.append(
                1 + (self.beat + 1) % 16 if self.beat < self.onset_limit // 96 else 0
            )
        elif self.phase == "event":
            last = self.beat * 96 + GRID32_TICKS[-1]
            next_tick = (
                (self.beat + 1) * 96
                if self.tick == last
                else self.beat * 96 + GRID32_TICKS[TICK_TO_GRID32[self.tick % 96] + 1]
            )
            choices = range(
                TAIKO_EVENT, TAIKO_EVENT + (8 if next_tick <= self.field_limit else 4)
            )
        elif self.phase == "end_beat":
            choices = [
                TAIKO_END_BEAT + b % 16
                for b in range(self.page, self.page + 16)
                if b * 96 <= self.field_limit
                and (
                    b > self.beat
                    or any(
                        self.tick < b * 96 + s <= self.field_limit for s in GRID32_TICKS
                    )
                )
            ]
            if (self.page + 16) * 96 <= self.field_limit:
                choices.append(TAIKO_LONG_SPAN)
        else:
            choices = [
                TAIKO_END_OFFSET + i
                for i, off in enumerate(GRID32_TICKS)
                if self.tick < self.end_beat * 96 + off <= self.field_limit
            ]
        return tuple(choices)

    def advance(self, token):
        if token == 0:
            self.phase = "timing" if self.phase == "start" else "done"
            return
        if token <= 16:
            self.beat += 1
            self.tick = self.query_tick = self.beat * 96
        elif token <= 48:
            self.tick = self.query_tick = self.beat * 96 + GRID32_TICKS[token - 17]
            self.last_tick, self.phase = self.tick, "event"
        elif token < TAIKO_LONG_SPAN:
            self.kind = token - TAIKO_EVENT
            self.pages = 0
            self.phase = "timing" if self.kind < 4 else "end_beat"
            self.minimum = self.tick + 1
        elif token == TAIKO_LONG_SPAN:
            self.pages += 1
            self.query_tick = self.page * 96
        elif token < TAIKO_END_OFFSET:
            self.end_beat = self.page + (token - TAIKO_END_BEAT - self.page) % 16
            self.query_tick = self.end_beat * 96
            self.phase = "end_offset"
        else:
            self.minimum = self.end_beat * 96 + GRID32_TICKS[token - TAIKO_END_OFFSET]
            self.query_tick = self.tick
            self.phase = "timing"


@lru_cache(maxsize=8192)
def legal_bits(tokens):
    mask = np.zeros(SIZE_V20, np.uint8)
    mask[list(tokens)] = 1
    return np.packbits(mask)


def encode_taiko(events, horizon, limit):
    seq = list(taiko_tokens(events, horizon))
    grammar = TaikoGrammar(horizon, limit)
    heads = HeadRows()
    records, indices, rows, allowed, pages, timing = [], [], [], [], [], []
    conv, mert, commits = [], [], []
    moment = -1
    for i, token in enumerate(seq[:-1]):
        grammar.advance(token)
        g = grammar
        is_time = g.phase == "timing"
        if token >= TAIKO_END_OFFSET:
            commits.append((i, g.minimum, i))
        if 17 <= token <= 48:
            moment = i
        position = (
            TICK_TO_GRID32[g.last_tick % 96]
            if 0 <= g.last_tick and g.tick - g.last_tick <= 96
            else 32
        )
        rows.append((g.tick, g.query_tick, moment, position if is_time else 33))
        if token < TAIKO_EVENT:
            heads.ids.append(0)
            heads.records[0].append(
                (g.tick, 0 if token == 0 else 1 if token <= 16 else 2, 0)
            )
        else:
            heads.ids.append(5)
            indices.append(i)
            phase = 2 if token >= TAIKO_END_OFFSET else int(token >= TAIKO_END_BEAT)
            end = g.minimum if phase == 2 else g.end_beat * 96 if phase == 1 else -1
            records.append((g.kind, phase, g.tick, end, g.pages))
        target = seq[i + 1]
        if 17 <= target <= 48:
            conv.append((i, g.beat * 96 + GRID32_TICKS[target - 17]))
        elif TAIKO_END_BEAT <= target < TAIKO_END_OFFSET:
            end_beat = g.page + (target - TAIKO_END_BEAT - g.page) % 16
            offset = GRID32_TICKS[seq[i + 2] - TAIKO_END_OFFSET]
            mert.append((i, end_beat * 96 + offset))
        elif target >= TAIKO_END_OFFSET:
            conv.append((i, g.end_beat * 96 + GRID32_TICKS[target - TAIKO_END_OFFSET]))
        allowed.append(legal_bits(g.legal()))
        timing.append(is_time)
        pages.append(g.page if g.phase == "end_beat" else -1)
    descriptors = heads.arrays()["heads"]
    descriptors.update(
        rows5=np.array(indices, np.int64),
        args5=np.array(records, np.int32).reshape(-1, 5),
    )
    rows = np.asarray(rows, np.int32)
    tokens = np.asarray(seq, np.int16)
    inputs = tokens[:-1]
    keep = (inputs == 0) | ((TAIKO_EVENT <= inputs) & (inputs < TAIKO_EVENT + 4)) | (inputs >= TAIKO_END_OFFSET)
    n = len(rows)
    return dict(
        taiko_events=events,
        heads=descriptors,
        field_limit=limit,
        rows=rows,
        tokens=tokens,
        legal=np.stack(allowed),
        timing=np.array(timing),
        pages=np.array(pages),
        active_state=np.zeros((sum(timing), 32), np.uint8),
        evict=~keep,
        active_spans=active_spans(commits, rows[:, 0]),
        protected=np.empty((0, 3), np.int32),
        conv_ticks=conv,
        mert_ticks=mert,
        weights=np.ones(n, np.float32),
        union_pairs=np.empty((0, 2), np.int32),
        slide_lookahead=(np.empty(0, np.int64), np.empty((0, 2), np.float32)),
    )


def taiko_draft_targets(events, beat_valid):
    return rhythm_targets(
        (
            (tick, [(int(kind >= 7), end)] if kind >= 5 else [])
            for tick, kind, end, _ in events
        ),
        beat_valid,
    )


def compute_taiko_beat_te_v3(events, grid, seconds, valid):
    seconds, valid = np.asarray(seconds), np.asarray(valid)
    n = max(101, math.ceil(max(0.0, seconds[-1]) * 20) + 21)
    impulse, spans = np.zeros(n), np.zeros(n)
    for tick, kind, end, hits in events:
        start = grid.second(tick)
        if kind < 5:
            impulse[int(np.clip(round(start * 20), 0, n - 1))] += (
                1.2 if kind >= 3 else 1.0
            )
        else:
            finish = grid.second(end)
            if finish <= start:
                continue
            rate = {5: 4.5, 6: 5.0}.get(
                kind, np.clip(hits / (finish - start), 3, 9) if hits > 0 else 5.0
            )
            left, right = np.clip([start * 20, finish * 20], 0, n)
            indices = np.arange(int(left), math.ceil(right))
            spans[indices] += (
                rate / 20 * (np.minimum(indices + 1, right) - np.maximum(indices, left))
            )
    beat_sec = float(
        np.clip(np.median(np.diff(seconds.reshape(-1, 24), axis=1)) * 24, 0.18, 0.65)
    )
    bins = max(3, round(beat_sec * 20) | 1)
    hist = np.convolve(impulse, np.full(bins, 1 / bins), "same") + spans
    curve = stabilize_high_plateau_preserve_drops(hist, beat_sec)
    return (
        (np.interp(seconds * 20, np.arange(n), curve) * valid)
        .reshape(-1, 24)
        .mean(1)
        .astype(np.float32)
    )
