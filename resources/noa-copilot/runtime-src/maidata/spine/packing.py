"""Teacher-forced decisions in exactly the same commit order as free generation."""

from collections import defaultdict
from zlib import crc32
import numpy as np
from .pieces import piece_rows, pad_mask
from .codec import PAD_MAP
from .tails import (
    detail,
    commit_pad,
    release_tick,
    feedback,
    tail_fields,
    Hazard,
    FEEDBACK_DIM,
    HAZ_DIM,
    DETAIL_DIM,
    HAZ_OFFSETS,
)

# IDs 0..95 select a MOMENT within the current beat. The next 16 IDs open
# consecutive integer beats in cyclic order, including beat zero after BOS.
BEAT = 96
CYCLE = 16
TIME_COUNT = BEAT + CYCLE
MOMENT, BOS, BEAT_WORD = range(3)


def beat_actions(moments, horizon_tick=None):
    beat = 0
    yield BEAT, ()
    for tick, notes in moments:
        while beat < tick // 96:
            beat += 1
            yield BEAT + beat % CYCLE, ()
        yield tick % 96, notes
    if horizon_tick is not None:
        while beat <= horizon_tick // 96:
            beat += 1
            yield BEAT + beat % CYCLE, ()


def slot_order(pads, song_id, tick, group, mirrored=False, *, seed=None, epoch=0):
    mirror = PAD_MAP[8]
    base = sorted(mirror[p] if mirrored else p for p in pads)
    if len(base) > 1:
        song = str(song_id)
        song_seed = int(song) if song.isdecimal() else crc32(song.encode("utf8"))
        # Existing Spine labels and editor replay use the original seed sequence.
        entropy = [song_seed, tick, group]
        if seed is not None:
            entropy = [seed, epoch, *entropy]
        np.random.default_rng(entropy).shuffle(base)
    return [mirror[p] for p in base] if mirrored else base


def chart_pack(
    moments,
    grid,
    meta,
    horizon_tick=None,
    *,
    song_id=0,
    mirrored=False,
    field_limit=None,
):
    """Steps: previous action, previous time token, cursor tick, next time token.

    Slots: step, absolute pad, pseudo group, another same-pad event, kind, flags.
    """
    steps = []
    slots = []
    tails = []
    fields = []
    pieces = []
    hazards = []
    tailhaz = []
    summaries = []
    details = []
    slot_haz = []

    prior_states, releases, footprints, nonplain_used = [], [], [], []
    repeat_room = []
    footprint = 0
    step_seconds = []
    tail_seconds = []
    hazard = Hazard()
    previous = BOS
    last = 0
    prev_dt = 0
    history = np.zeros(FEEDBACK_DIM, np.float32)

    def step(dt):
        steps.append((previous, prev_dt, last, dt))
        step_seconds.append(grid.second(last))
        summaries.append(history.copy())
        footprints.append(footprint)
        hazards.append(hazard.query(last))

    horizon_tick = max(horizon_tick or 0, moments[-1][0] if moments else 0)
    for token, notes in beat_actions(moments, horizon_tick):
        step(token)
        history.fill(0)
        footprint = 0
        if token >= BEAT:
            last = 0 if previous == BOS else (last // 96 + 1) * 96
            previous, prev_dt = BEAT_WORD, token
            continue
        u = len(steps) - 1
        previous, prev_dt = MOMENT, token
        committed = np.zeros((41, 6), np.uint8)
        t = last = last // 96 * 96 + token
        by_group = defaultdict(lambda: defaultdict(list))
        for note in notes:
            by_group[note[1]][note[0]].append(note)
        group_ids = sorted(by_group)
        pieces.extend(piece_rows(u, [tuple(sorted(by_group[g])) for g in group_ids]))
        for group_index, group in enumerate(group_ids):
            for pad in slot_order(by_group[group], song_id, t, group, mirrored):
                occurrences = by_group[group][pad]
                non_slide = False
                for i, note in enumerate(occurrences):
                    membership = int(i + 1 < len(occurrences))
                    slot = len(slots)
                    prior_states.append(np.packbits(committed.reshape(-1)))
                    nonplain_used.append(non_slide)
                    sensor = pad if pad < 8 else pad - 8
                    slot_haz.append(hazard.at_sensor(t, sensor))
                    kind, flags = note[2:4]
                    slots.append((u, pad, group_index, membership, kind, flags))
                    details.append(detail(t, note))
                    commit_pad(committed, pad, kind, flags)
                    non_slide |= kind != 2
                    normalized_note = (pad, group_index, *note[2:])
                    feedback(t, t, normalized_note, history)
                    if kind:
                        releases.append(release_tick(note))
                        tails.append((slot, t, kind))
                        tail_seconds.append(grid.second(t))
                        tailhaz.append(hazard.query(t))
                        fields.append(tail_fields(t, note))
                        hazard.commit(t, note)
                    repeat_room.append(
                        field_limit is None or hazard.slide_room(t, pad, field_limit)
                    )
        footprint = pad_mask(set(n[0] for n in notes))
    return dict(
        steps=np.array(steps, np.int32),
        step_seconds=np.asarray(step_seconds, np.float32),
        slot_haz=np.asarray(slot_haz, np.uint8).reshape(-1, 4),
        committed_pads=np.asarray(prior_states, np.uint8).reshape(-1, 31),
        repeat_room=np.asarray(repeat_room, np.bool_),
        prior_non_slide=np.asarray(nonplain_used, np.bool_),
        tail_release=np.asarray(releases, np.int32),
        footprint=np.asarray(footprints, np.int64),
        pieces=np.asarray(pieces, np.int64).reshape(-1, 5),
        tail_seconds=np.asarray(tail_seconds, np.float32),
        feedback=np.array(summaries),
        hazard=np.array(hazards),
        slots=np.array(slots, np.int32).reshape(-1, 6),
        detail=np.array(details, np.float32).reshape(-1, DETAIL_DIM),
        tails=np.array(tails, np.int32).reshape(-1, 3),
        tailhaz=np.array(tailhaz, np.uint8).reshape(-1, len(HAZ_OFFSETS) * HAZ_DIM),
        fields=fields,
        meta=meta,
    )
