"""Tokenize the accepted chart prefix and commit its real notes into incremental KV."""

import torch

from .codec import PAD_MAP, mirror_notes
from .packing import beat_actions as prefix_actions  # noqa: F401
from .packing import slot_order
from .tails import commit_pad, release_tick, detail, end_pads, feedback


def _reflected_order(notes, group_pads, wh):
    """Choose one orientation for the already committed MOMENT and its reflection."""
    if all(len(pads) == 1 for pads in group_pads) or not any(n[2] for n in notes):
        return False  # Order cannot affect a Tail record.
    reflected = [tuple(sorted(PAD_MAP[8][p] for p in pads)) for pads in group_pads]
    if group_pads != reflected:
        return group_pads > reflected

    # Symmetric geometry can still have an asymmetric Tap, Hold or Slide route.
    def signature(events):
        return [
            (*n[:6], tuple((a, z, b, tuple(route)) for a, z, b, route in n[6]))
            for n in sorted(events, key=lambda note: (note[1], note[0]))
        ]

    direct, flipped = signature(notes), signature(mirror_notes(notes))
    if direct != flipped:
        return direct > flipped
    odd = wh[0] - wh[1]
    return bool(odd[odd.abs().argmax()] < 0)


def replay_moment(
    model, bank, wh, onset, tick, notes, history, hazard, tail_cache, clock, song_id=0
):
    device = wh.device

    def val(x):
        return torch.as_tensor(x, device=device)

    pads = sorted({note[0] for note in notes})
    history.fill(0)
    footprint = wh.new_zeros(41)
    footprint[pads] = 1
    geom = model.footprint(footprint)
    groups = {value: i for i, value in enumerate(sorted({n[1] for n in notes}))}
    group_pads = [tuple(sorted({n[0] for n in notes if n[1] == g})) for g in groups]
    mirrored = _reflected_order(notes, group_pads, wh)
    committed = wh.new_zeros(41, 6)
    occurrence = {}
    order = {
        (g, p): i
        for g, pads in zip(groups, group_pads)
        for i, p in enumerate(slot_order(pads, song_id, tick, g, mirrored))
    }
    for note in sorted(notes, key=lambda n: (n[1], order[n[1], n[0]])):
        pad, group, kind, flags = note[:4]
        group = groups[group]
        note = (pad, group, *note[2:])
        p = val(pad)
        key = group, pad
        if kind:
            state = model.slot_state(
                wh,
                geom,
                p,
                val(group),
                val(occurrence.get(key, 0)),
                onset,
                val(hazard.at_sensor(tick, pad if pad < 8 else pad - 8)),
                committed,
            )
            times, pending = val([[tick]]), []
            model.tail_states(
                state[None, None],
                val([[kind]]),
                times,
                clock([[tick]], model.config.tail_probes),
                bank,
                val([0]),
                val(hazard.query(tick))[None, None],
                geom[None, None],
                cache=tail_cache,
                tick=tick,
                pending=pending,
            )
            hazard.commit(tick, note)
            model.commit_tail(
                tail_cache,
                pending,
                tick,
                times,
                p,
                val(detail(tick, note)),
                val(end_pads(note)),
                release_tick(note),
            )
        feedback(tick, tick, note, history)
        commit_pad(committed, pad, kind, flags)
        occurrence[key] = occurrence.get(key, 0) + 1
