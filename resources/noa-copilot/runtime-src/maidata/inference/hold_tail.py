"""Inference-only repair helpers for Hold / final Slide contacts."""
from dataclasses import replace

from maidata.chart.types import HoldDuration, SlideDuration
from maidata.physical.actions import materialize
from maidata.physical.geometry import pad_bit
from maidata.physical.rules import EPS
from maidata.physical.support import _partition, decision_record
from maidata.time import advance_ticks


def contacts(state, token):
    """Only the final segment's final judge-area entry is relevant."""
    try:
        _, completion = state._trial(token)
    except ValueError:
        return ()
    if completion is None:
        return ()
    candidate = materialize(completion, state.timing, state.geometry)
    result = []
    for old in state.actions:
        hold, slide = ((old, candidate) if candidate.track else (candidate, old))
        if hold.kind not in ('hold', 'touch_hold') or not slide.track:
            continue
        if not slide.motions[-1].entries:
            continue
        entered, mask = slide.motions[-1].entries[-1]
        if not mask & pad_bit(hold.pad):
            continue
        checked = advance_ticks(state.timing, entered, -96)
        # Actions include a 22-tick release guard; recover the true hold end.
        end = advance_ticks(state.timing, hold.press_end, -22)
        if hold.press_start <= entered + EPS and checked < end - EPS:
            result.append(dict(hold_owner=hold.owner, hold_tick=hold.tick,
                hold_pad=hold.pad, hold_start=hold.press_start, hold_end=end,
                slide_owner=slide.owner, slide_tick=slide.tick,
                track_index=slide.track_index, path=slide.motions[-1].path,
                judge_area_mask=mask, entry_seconds=entered, checked_seconds=checked))
    return tuple(result)


def fork(state, snapshot=None):
    result = type(state)(state.timing, state.geometry, audio_end_seconds=state.audio_end_seconds)
    result.physical_enabled = getattr(state, 'physical_enabled', True)
    result.duration_end_seconds = state.duration_end_seconds
    result.restore(snapshot or state.snapshot())
    return result


def legal(state, token, sampler):
    return sampler.allowed(state.query(token)) and not contacts(state, token)


def strict_support(state):
    """Exact inference-only duration intervals, including new-rule crossings.

    Slide entry time is affine in written duration. Its preceding beat overlaps
    a hold when entry >= hold start and entry < hold end shifted forward a beat.
    Only the end bound shifts, including through BPM changes. No tick enumeration.
    """
    base = state.time_support('SLIDE_DURATION' if state.grammar.state.phase == 'chain' else 'HOLD_DURATION')
    lower, upper = base['domain']
    if upper < lower:
        return base
    roots = [v for lo, hi, _ in base['intervals'] for v in (lo, hi + 1)]
    roots.extend(final_area_roots(state, lower))
    constructor = SlideDuration if state.grammar.state.phase == 'chain' else HoldDuration

    def query(ticks):
        token = constructor(ticks)
        decision = state.query(token)
        conflict = bool(contacts(state, token))
        if conflict:
            decision = replace(decision, physical=decision.physical | {'slide_too_fast'})
        return dict(decision_record(decision), hold_slide_final_area=conflict)

    return dict(base, intervals=_partition(roots, lower, upper, query, record=lambda value: value))


def final_area_roots(state, lower):
    """Final-area duration crossings, without evaluating a support partition."""
    slide = state.grammar.state.phase == 'chain'
    constructor = SlideDuration if slide else HoldDuration
    _, first = state._trial(constructor(lower))
    a = materialize(first, state.timing, state.geometry)
    roots = []
    if slide and a.motions[-1].entries:
        _, second = state._trial(constructor(lower + 1))
        b = materialize(second, state.timing, state.geometry)
        t = a.motions[-1].entries[-1][0]
        slope = b.motions[-1].entries[-1][0] - t
        mask = a.motions[-1].entries[-1][1]
        if slope:
            for hold in state.actions:
                if hold.kind in ('hold', 'touch_hold') and mask & pad_bit(hold.pad):
                    end = advance_ticks(state.timing, hold.press_end, -22)
                    roots.extend(lower + (bound - t) / slope
                                 for bound in (hold.press_start, advance_ticks(state.timing, end, 96)))
    elif not slide:
        origin = a.press_start
        scale = state.timing.written_tick_seconds(state.grammar.state.cursor)
        for old in state.actions:
            if old.track and old.motions[-1].entries:
                entered, mask = old.motions[-1].entries[-1]
                checked = advance_ticks(state.timing, entered, -96)
                if mask & pad_bit(a.pad) and origin <= entered + EPS:
                    roots.append((checked - origin) / scale)

    return roots
