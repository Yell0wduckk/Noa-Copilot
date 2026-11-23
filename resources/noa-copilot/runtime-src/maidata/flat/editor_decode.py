"""Single-worker editor decoding with bounded prefix and branch snapshots."""

from collections import OrderedDict, deque
from copy import deepcopy
from dataclasses import dataclass, field, replace
from itertools import groupby
from weakref import ref

import torch


from maidata.spine.generate import pick, scaled_slide_probabilities
from .codec import (
    EOM,
    FAMILIES,
    FAMILY_IDS,
    Grammar,
    TIMING_INPUTS,
    chart_tokens,
    field_horizon,
    resolve_version_profile,
)
from .editor_constraints import Physical
from .generate import Cursor, pick_time


def grammar_copy(g):
    # Completed moments never mutate. Only a partial chord shares mutable note objects.
    return deepcopy(g, {id(g.moments): list(g.moments)})


@dataclass
class Snapshot:
    grammar: object
    physical: object
    scores: object
    layers: list
    flags: object
    history: list
    rows: list
    last_plan_block: int

    @classmethod
    def capture(cls, c, g, physical, scores):
        indices = [0, *range(c.keep_start, len(c.history))]
        selected = c.tensor(indices)
        layers = [tuple(x.index_select(2, selected) for x in pair) for pair in c.layers]
        return cls(
            grammar_copy(g),
            physical.fork(),
            scores,
            layers,
            c.note_keys[selected],
            [c.history[i] for i in indices],
            [c.history_rows[i] for i in indices],
            c.last_plan_block,
        )

    def restore(self, c):
        c.last_plan_block = self.last_plan_block
        c.history, c.history_rows = list(self.history), list(self.rows)
        c.keep_start, c.sink_slot = 1, 0
        c.last_query = c.last_clock = c.evidence_beat = None
        c.capacity = max(512, len(c.history) + 128)
        c.layers = []
        for pair in self.layers:
            storage = tuple(c.allocate(t) for t in pair)
            for dst, src in zip(storage, pair):
                dst[:, :, : len(c.history)].copy_(src)
            c.layers.append(storage)
        c.note_keys = self.flags.new_empty(c.capacity)
        c.note_keys[: len(c.history)].copy_(self.flags)
        return grammar_copy(self.grammar), self.physical.fork(), self.scores


class PrefillCheckpoints:
    """Gather rollback rows from the same batched decoder pass as the final prefix."""

    def __init__(self, points, checkpoints, physical):
        self.points = set(points)
        self.checkpoints = checkpoints
        self.physical = physical
        self.layers = []

    def record(self, cursor, grammar, rows, evict, seq):
        positions = cursor.prefill_positions(grammar, rows, evict)
        indices = cursor.tensor(positions)
        self.physical.advance(cursor.grid.second(grammar.tick))
        saved = deepcopy(grammar, {id(grammar.moments): []})
        flags = cursor.tensor(
            [not i or seq[i] not in TIMING_INPUTS for i in positions], torch.bool
        )
        snapshot = Snapshot(
            saved,
            self.physical.fork(),
            None,
            [],
            flags,
            [(rows[i][0], seq[i]) for i in positions],
            positions,
            cursor.last_plan_block,
        )
        self.checkpoints[len(rows)] = snapshot
        self.layers.append((indices, snapshot.layers))

    def capture_layer(self, keys, values):
        for indices, layers in self.layers:
            layers.append(
                (keys.index_select(2, indices), values.index_select(2, indices))
            )


@dataclass
class Track:
    tokens: list = field(default_factory=list)
    decisions: list = field(default_factory=list)
    beats: dict = field(default_factory=dict)
    snapshots: dict = field(default_factory=dict)
    moments: list = field(default_factory=list)
    needs_rescore: bool = False

    def record(self, index, alternatives):
        for rank in (0, 1):
            ids = [
                idx
                for idx in alternatives
                if (FAMILIES[idx][0] in STRUCTURAL) == (rank == 0)
            ]
            if ids:
                self.decisions.append((rank, index, ids))


def canonical(moments):
    """Keys follow the playable Simai result, including the order of pseudo-each groups."""

    def note_key(note):
        pad, _, kind, flags, end, head, branches = note
        paths = tuple(
            sorted((a, b, br, tuple(routes)) for a, b, br, routes in branches)
        )
        return (
            pad,
            kind,
            flags,
            end if kind == 1 else 0,
            head if kind == 2 else 0,
            paths,
        )

    return tuple(
        (
            tick,
            tuple(
                tuple(sorted(map(note_key, group)))
                for _, group in groupby(notes, key=lambda n: n[1])
            ),
        )
        for tick, notes in moments
    )


# On 3 official Master charts, the 10th percentile of native top-1 was
# .9496 on free choices and .9960 with occupied hands (v19 physical epoch 60).
LOCK_PROBABILITY = 0.98
# Native runner-up P25 among unlocked decisions on the three calibration charts: .0304.
MIN_BRANCH_PROBABILITY = 0.03
MAX_CANDIDATE_ATTEMPTS = 8
STRUCTURAL = {
    "MOMENT",
    "BEAT",
    "KEY_POS",
    "TOUCH_PAD",
    "HOLD",
    "SLIDE_HEAD",
    "ROUTE",
    "SLIDE_BRANCH",
    "CHORD",
    "EOM",
}


def eligible_children(scores, choices, lock_probability=LOCK_PROBABILITY, slide_multiplier=1.0):
    probabilities = scaled_slide_probabilities(scores, choices, 1.0, slide_multiplier, FAMILY_IDS["SLIDE_HEAD"])
    if probabilities is None:
        probabilities = scores[choices].float().softmax(0)
    values = probabilities.tolist()
    best = max(range(len(choices)), key=values.__getitem__)
    if values[best] >= lock_probability:
        return [choices[best]]
    return [
        idx
        for i, idx in enumerate(choices)
        if i == best or values[i] >= MIN_BRANCH_PROBABILITY
    ]


class BranchFrontier:
    """Unvisited edges survive both a dead end and eviction of a finished candidate's KV."""

    def __init__(self):
        self.queues = (deque(), deque())

    def record(self, source, decisions):
        for priority, index, children in decisions:
            self.queues[priority].extend((source, index, child) for child in children)

    def pop(self):
        for queue in self.queues:
            if queue:
                (live, path), index, child = queue.popleft()
                return live() or path, index, child
        return None


def branch_source(track):
    # Share token/beat lists; only the bounded live tracks own GPU snapshots.
    return ref(track), Track(
        tokens=track.tokens, beats=track.beats, decisions=track.decisions
    )


class EditorDecoder:
    def __init__(
        self,
        model,
        prepared,
        grid,
        meta,
        prefix,
        end,
        span,
        full,
        trunk,
        binding,
        skeleton=None,
        raw_prefix=None,
        prefix_tokens=None,
    ):
        self.start, self.stop = span.start, span.stop
        self.limit = min(end, span.stop - 1)
        self.full, self.skeleton = full, skeleton
        self.tracks, self.seen = OrderedDict(), set()
        self.selected = 1
        self.frontier = BranchFrontier()
        horizon = max(end, field_horizon(prefix, 0))
        g = Grammar(
            horizon,
            onset_limit=self.limit,
            max_groups=8,
            max_slot_events=8,
            max_fields=128,
            profile=resolve_version_profile(meta),
            plan4_anchor=model.config.inline_plan,
        )
        c = self.cursor = Cursor(
            model,
            None,
            grid,
            meta,
            end,
            field_limit=horizon,
            prefix=prefix,
            start_tick=self.start,
            prepared=prepared,
            plan_beat_te=meta.get("plan_beat_te"),
        )
        self.seq = (
            list(chart_tokens(prefix, song=meta.get("song_id") or 0, seed=None))
            if prefix_tokens is None
            else prefix_tokens
        )
        physical = Physical(grid, [], full)
        self.trunk = trunk
        plan = getattr(c, "plan4", None)
        # Prefix truth may have changed underneath the same audio. Never reuse KV
        # conditioned on different Plan4 inputs. Reuse branches within a step directly.
        key = (binding, full, horizon)
        previous = trunk.get("seq", []) if trunk.get("key") == key else []
        common = next(
            (i for i, (a, b) in enumerate(zip(previous, self.seq)) if a != b),
            min(len(previous), len(self.seq)),
        )
        plan = plan.detach().cpu() if plan is not None else None
        old_plan = trunk.get("plan")
        first_change = float("inf")
        if plan is not None and old_plan is not None and old_plan.shape == plan.shape:
            changes = (plan != old_plan).any(-1).nonzero()
            if len(changes):
                first_change = int(changes[0])
        elif plan is not None or old_plan is not None:
            first_change = 0
        checkpoints = {
            k: v
            for k, v in trunk.get("checkpoints", {}).items()
            if k <= common
            and (k < len(self.seq) or v.scores is not None)
            and v.last_plan_block < first_change
            and previous
        }
        restored_prefix = bool(checkpoints)
        if restored_prefix:
            n = max(checkpoints)
            g, physical, scores = checkpoints[n].restore(c)
            g.onset_limit = self.limit
            g.moments.clear()
        else:
            n = len(self.seq)
            beats = [
                i + 1
                for i, idx in enumerate(self.seq[:-1])
                if FAMILIES[idx][0] == "BEAT"
            ]
            onsets = [
                i + 1
                for i, idx in enumerate(self.seq[:-1])
                if FAMILIES[idx][0] == "MOMENT"
            ]
            rollback = PrefillCheckpoints(
                beats[-16:-4:3] + beats[-4:] + onsets[-8:], checkpoints, physical
            )
            scores = c.prefill(g, self.seq, rollback=rollback)
        for i in range(n, len(self.seq)):
            idx = self.seq[i]
            g.advance(idx)
            if idx == EOM:
                physical.commit(g.tick, g.notes)
                g.moments.clear()
            scores = c.step(g)
            if FAMILIES[idx][0] in ("BEAT", "MOMENT"):
                checkpoints[i + 1] = Snapshot.capture(c, g, physical, scores)
        g.moments.clear()
        if raw_prefix is not None and (restored_prefix or raw_prefix != prefix):
            # Preserve real elapsed seconds for handwritten notes off the model grid.
            physical = Physical(grid, raw_prefix, full)
        self.anchor = Snapshot.capture(c, g, physical, scores)
        checkpoints[len(self.seq)] = self.anchor
        trunk.update(
            key=key,
            plan=plan,
            seq=self.seq,
            checkpoints=dict(sorted(checkpoints.items())[-17:]),
        )
        self.base_checkpoints = dict(trunk["checkpoints"])

    def replay(self, track, index):
        c = self.cursor
        start = max((k for k in track.snapshots if k <= index), default=0)
        g, p, scores = track.snapshots.get(start, self.anchor).restore(c)
        if start:
            # Retained beat snapshots can predate a local step on this same candidate.
            g.moments = [m for m in g.moments if m[0] >= self.start]
        for idx in track.tokens[start:index]:
            g.advance(idx)
            if idx == EOM:
                p.commit(g.tick, g.notes)
            scores = c.step(g)
        return g, p, scores

    def activate(self, number):
        if number in self.tracks:
            self.selected = number
            self.remember_prefix(self.tracks[number])

    def remember_prefix(self, track):
        checkpoints = dict(self.base_checkpoints)
        checkpoints.update(
            (len(self.seq) + k, snap)
            for k, snap in sorted(track.snapshots.items())[-4:]
        )
        self.trunk.update(
            seq=self.seq + track.tokens,
            checkpoints=dict(sorted(checkpoints.items())[-21:]),
        )

    def step_anchor(self, number, tick, *, recondition=None):
        track = self.tracks[number]
        index = track.beats[tick]
        seq = self.seq + track.tokens[:index]
        remaining = Track(
            tokens=track.tokens[index:],
            decisions=[
                (priority, k - index, alts)
                for priority, k, alts in track.decisions
                if k >= index
            ],
            beats={t: k - index for t, k in track.beats.items() if t > tick},
            snapshots={k - index: v for k, v in track.snapshots.items() if k > index},
            moments=[m for m in track.moments if m[0] >= tick],
            needs_rescore=track.needs_rescore,
        )
        decoder = self
        if recondition is not None and tick // 384 != self.start // 384:
            # The newly complete bars now use prefix truth. Reuse only KV that never read
            # a changed plan block. The accepted suffix stays available on screen.
            self.remember_prefix(track)
            old_plan = self.trunk.get("plan")
            decoder = recondition(seq)
            plan = decoder.trunk.get("plan")
            if plan is not None and (
                old_plan is None or not torch.equal(old_plan, plan)
            ):
                remaining.snapshots.clear()
                # Branch probabilities were measured under the old plan. Recalculate
                # them lazily only if this retained suffix is used to generate a sibling.
                remaining.decisions.clear()
                remaining.needs_rescore = True
        else:
            if index in track.snapshots:
                snap = track.snapshots[index]
                grammar = deepcopy(snap.grammar, {id(snap.grammar.moments): []})
                self.anchor = replace(snap, grammar=grammar)
            else:
                g, p, scores = self.replay(track, index)
                g.moments.clear()
                self.anchor = Snapshot.capture(self.cursor, g, p, scores)
            self.seq, self.start = seq, tick
            self.cursor.start_tick = tick
            self.base_checkpoints[len(seq)] = self.anchor
            self.base_checkpoints = dict(sorted(self.base_checkpoints.items())[-17:])
        decoder.tracks = OrderedDict([(1, remaining)])
        decoder.seen, decoder.selected = {canonical(remaining.moments)}, 1
        decoder.frontier = BranchFrontier()
        decoder.frontier.record(branch_source(remaining), remaining.decisions)
        decoder.remember_prefix(remaining)
        return decoder

    def choices(self, g, p):
        legal = g.legal(start_tick=self.start)
        return self.skeleton.mask(g, legal, p) if self.skeleton else p.mask(g, legal)

    def branch(self, cancelled):
        retained = self.tracks.get(1)
        if retained is not None and retained.needs_rescore:
            # A changed Plan4 can even put the retained suffix outside the new tree.
            # Explore the new root once; the retained completed chart is already in seen.
            retained.needs_rescore = False
            return retained, 0, None
        return self.frontier.pop()

    def sample(self, temperature, slide_multiplier, emit, cancelled, *, branch=None, keep=False):
        c = self.cursor
        track = Track()
        if branch:
            parent, index, forced = branch
            g, p, scores = self.replay(parent, index)
            track.tokens = parent.tokens[:index]
            track.beats = {t: k for t, k in parent.beats.items() if k <= index}
            track.decisions = [d for d in parent.decisions if d[1] < index]
            track.snapshots = {
                k: snap for k, snap in parent.snapshots.items() if k <= index
            }
            if len(track.snapshots) > 4:
                keys = sorted(track.snapshots)
                track.snapshots = {k: track.snapshots[k] for k in keys[:2] + keys[-2:]}
            for t, notes in g.moments:
                emit(t, notes, False)
            if g.beat * 96 > self.start:
                emit(g.beat * 96, [], True)
        else:
            g, p, scores = self.anchor.restore(c)
            forced = None
        source = branch_source(track)
        while g.tick <= self.limit:
            if cancelled.is_set():
                raise InterruptedError()
            choices = self.choices(g, p)
            if not choices:
                return None
            if not self.full and len(choices) > 1:
                choices = eligible_children(scores, choices, slide_multiplier=slide_multiplier)
            if len(choices) == 1:
                idx = choices[0]
            else:
                probs = scaled_slide_probabilities(scores, choices, temperature, slide_multiplier, FAMILY_IDS["SLIDE_HEAD"])
                if forced is not None:
                    idx = forced
                elif probs is not None:
                    i = torch.multinomial(probs, 1) if temperature else probs.argmax()
                    idx = choices[int(i)]
                elif g.phase == "time" and FAMILIES[choices[-1]][0] == "BEAT":
                    idx = pick_time(scores, temperature, choices)
                else:
                    idx = int(pick(scores, temperature, choices))
                if not self.full:
                    rest = [child for child in choices if child != idx]
                    weights = scores[rest].float() if probs is None else (
                        probs[[i for i, child in enumerate(choices) if child != idx]].log() * (temperature or 1)
                    )
                    if temperature > 0:
                        weights = (
                            weights
                            - temperature
                            * torch.empty_like(weights).exponential_().log()
                        )
                    order = weights.argsort(descending=True, stable=True).tolist()
                    first = len(track.decisions)
                    track.record(len(track.tokens), [rest[i] for i in order])
                    if forced is None:
                        self.frontier.record(source, track.decisions[first:])
            forced = None
            g.advance(idx)
            scores = c.step(g)
            track.tokens.append(idx)
            if idx == EOM:
                p.commit(g.tick, g.notes)
                emit(g.tick, g.notes, False)
            if FAMILIES[idx][0] == "BEAT" and g.tick > self.start:
                emit(g.tick, [], True)
                track.beats[g.tick] = len(track.tokens)
                # Keep rollback anchors while decoding, including compact anchors for
                # background candidates. Switching candidates never runs the model.
                if not self.full:
                    track.snapshots[len(track.tokens)] = Snapshot.capture(
                        c, g, p, scores
                    )
                    if len(track.snapshots) > (16 if keep else 4):
                        del track.snapshots[list(track.snapshots)[2]]
        track.moments = g.moments
        return track

    def remember(self, number, track):
        self.tracks[number] = track
        self.seen.add(canonical(track.moments))
        if number == self.selected:
            self.remember_prefix(track)
        while len(self.tracks) > 17:
            oldest = next(i for i in self.tracks if i not in (1, self.selected, number))
            del self.tracks[oldest]
