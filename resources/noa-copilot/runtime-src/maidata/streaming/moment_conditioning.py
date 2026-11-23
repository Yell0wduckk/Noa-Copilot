"""Within-beat moment position and active hold/slide occupancy conditioning for DenseEventDecoder."""

from dataclasses import dataclass
from fractions import Fraction
import torch
from torch import Tensor

from maidata.time import TimingMap, duration_end
from .physical_schema import VOCAB_OFFSETS


def _default_timing() -> TimingMap:
    return TimingMap(((Fraction(0, 1), 120.0),), 0.0, validate_input=False)


def _resolve_timing(timing, physical_v6: bool = True) -> TimingMap:
    if timing is None:
        timing = _default_timing()
    elif isinstance(timing, dict):
        timing = TimingMap.from_audio_timing(timing, validate_input=False)
    if physical_v6:
        from .v6_oracle import timing_v6

        timing = timing_v6(timing)
    return timing


@dataclass(frozen=True)
class MomentConditioningState:
    """Compact O(1) incremental state for streaming moment conditioning."""

    prev_onset_tick: int | None = None
    active_holds: tuple[float, ...] = ()
    active_slides: tuple[float, ...] = ()
    onset_tick: int | None = None
    span_tokens: tuple[tuple[int, int], ...] | None = None
    last_type: int | None = None
    global_duration: bool = True


def _finish_span_tokens(
    span_tokens: tuple[tuple[int, int], ...] | None,
    onset_tick: int | None,
    active_holds: tuple[float, ...],
    active_slides: tuple[float, ...],
    timing: TimingMap,
    global_duration: bool,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if not span_tokens or onset_tick is None:
        return active_holds, active_slides
    from .physical_schema import VOCAB_OFFSETS as PHYSICAL_OFFSETS

    hold_ends: list[float] = []
    slide_ends: list[float] = []
    continuation = 0
    wait = 96
    waiting = 0
    for ti, vi in span_tokens:
        if ti in (7, 10):
            continuation = 0
            waiting = 0
        elif ti == 24:
            waiting += 96
        elif ti == 23:
            wait = waiting + vi - PHYSICAL_OFFSETS[23] + 1
            waiting = 0
        elif ti in (14, 15):
            continuation += 96
        elif ti == 8:
            duration = continuation + vi - VOCAB_OFFSETS[8] + 1
            end_sec = duration_end(
                timing,
                onset_tick,
                duration,
                slide=False,
                **(dict(wait=wait) if global_duration else {}),
            )
            hold_ends.append(end_sec)
            continuation = 0
        elif ti == 11:
            duration = continuation + vi - VOCAB_OFFSETS[11] + 1
            end_sec = duration_end(
                timing,
                onset_tick,
                duration,
                slide=True,
                **(dict(wait=wait) if global_duration else {}),
            )
            slide_ends.append(end_sec)
            continuation = 0
    if hold_ends:
        active_holds = active_holds + (max(hold_ends),)
    if slide_ends:
        active_slides = active_slides + (max(slide_ends),)
    return active_holds, active_slides


def step_moment_conditioning(
    state: MomentConditioningState | None,
    types: Tensor,
    vocab_ids: Tensor,
    causal_beats: Tensor,
    timing=None,
    *,
    physical_v6: bool = True,
    no_placeholders: bool = True,
) -> tuple[dict[str, Tensor], MomentConditioningState]:
    """Incrementally advance moment conditioning by `step_len` tokens in O(step_len) time."""
    squeeze = types.ndim == 1
    if squeeze:
        types = types[None]
        vocab_ids = vocab_ids[None]
        causal_beats = causal_beats[None]

    device = types.device
    t_list = types[0].detach().cpu().tolist()
    v_list = vocab_ids[0].detach().cpu().tolist()
    ticks = (causal_beats[0].detach().cpu().double() * 96.0).round().long().tolist()
    resolved_timing = _resolve_timing(timing, physical_v6=physical_v6)

    if state is None:
        prev_onset_tick = None
        active_holds: tuple[float, ...] = ()
        active_slides: tuple[float, ...] = ()
        onset_tick: int | None = None
        span_tokens: tuple[tuple[int, int], ...] | None = None
        last_type: int | None = None
        global_duration = physical_v6 or any(t >= 19 for t in t_list)
    else:
        prev_onset_tick = state.prev_onset_tick
        active_holds = state.active_holds
        active_slides = state.active_slides
        onset_tick = state.onset_tick
        span_tokens = state.span_tokens
        last_type = state.last_type
        global_duration = state.global_duration or physical_v6 or any(t >= 19 for t in t_list)

    mq_out = [False] * len(t_list)
    pos_out = [0] * len(t_list)
    occ_out = [0] * len(t_list)

    for i, (t, v, tick) in enumerate(zip(t_list, v_list, ticks)):
        tick = int(tick)
        # 1. Moment position ID update
        if t == 16:
            prev_onset_tick = None
            pos_out[i] = 96
        elif t == 12:
            pos_out[i] = tick % 96
            prev_onset_tick = tick
        elif t == 0 and v == 96:
            if prev_onset_tick is not None and (tick - prev_onset_tick) <= 96:
                pos_out[i] = prev_onset_tick % 96
            else:
                pos_out[i] = 96
        else:
            pos_out[i] = tick % 96
            prev_onset_tick = tick

        # 2. Occupancy state update
        if t == 16:
            active_holds = ()
            active_slides = ()
            onset_tick = None
            span_tokens = None
        elif t == 0 and v < 96:
            active_holds, active_slides = _finish_span_tokens(
                span_tokens, onset_tick, active_holds, active_slides, resolved_timing, global_duration
            )
            onset_tick = tick
            span_tokens = ()
        elif t == 18 or (
            no_placeholders
            and (t == 19 or (t == 4 and last_type != 9))
            and onset_tick is not None
            and span_tokens is not None
            and len(span_tokens) >= 1
        ):
            active_holds, active_slides = _finish_span_tokens(
                span_tokens, onset_tick, active_holds, active_slides, resolved_timing, global_duration
            )
            span_tokens = ((t, v),)
        elif t == 12 or (t == 0 and v == 96):
            active_holds, active_slides = _finish_span_tokens(
                span_tokens, onset_tick, active_holds, active_slides, resolved_timing, global_duration
            )
            span_tokens = None
        else:
            if onset_tick is None:
                onset_tick = tick
                span_tokens = ((t, v),)
            elif span_tokens is not None:
                span_tokens = span_tokens + ((t, v),)

        is_mq = t in (16, 12) or (t == 0 and v == 96)
        mq_out[i] = is_mq
        if is_mq:
            now = resolved_timing.second(tick)
            active_holds = tuple(end for end in active_holds if end > now + 1e-9)
            active_slides = tuple(end for end in active_slides if end > now + 1e-9)
            occ_out[i] = min(3, len(active_holds)) * 4 + min(3, len(active_slides))
        else:
            occ_out[i] = 0
        last_type = t

    mq_t = torch.tensor(mq_out, dtype=torch.bool, device=device)
    pos_t = torch.tensor(pos_out, dtype=torch.long, device=device)
    occ_t = torch.tensor(occ_out, dtype=torch.long, device=device)
    if not squeeze:
        mq_t, pos_t, occ_t = mq_t[None], pos_t[None], occ_t[None]

    next_state = MomentConditioningState(
        prev_onset_tick=prev_onset_tick,
        active_holds=active_holds,
        active_slides=active_slides,
        onset_tick=onset_tick,
        span_tokens=span_tokens,
        last_type=last_type,
        global_duration=global_duration,
    )
    return dict(moment_query=mq_t, moment_pos_id=pos_t, occupancy_id=occ_t), next_state


def moment_query_mask(types: Tensor, vocab_ids: Tensor) -> Tensor:
    """True on BOUNDARY(16), EOM(12), and EOB(0, 96) tokens that predict the next moment."""
    return (types == 16) | (types == 12) | ((types == 0) & (vocab_ids == 96))


def moment_key_mask(types: Tensor, vocab_ids: Tensor) -> Tensor:
    """True on EOM(12) and EOB(0, 96) tokens (excluded as keys in Stage 1, keeping BOUNDARY sink 0)."""
    return (types == 12) | ((types == 0) & (vocab_ids == 96))


def compute_moment_pos_id(
    types: Tensor,
    vocab_ids: Tensor,
    causal_beats: Tensor,
    *,
    sinks: Tensor | None = None,
) -> Tensor:
    """Compute moment_pos_id in 0..96 for each token position.

    - BOUNDARY(16): 96
    - EOM(12): ticks[q] % 96 (0..95, preceding moment's within-beat position)
    - EOB(0, 96): T_prev % 96 (0..95) if most recent onset since BOUNDARY has ticks[q] - T_prev <= 96, else 96
    - All non-moment_query tokens: ticks[q] % 96 (0..95, current moment's within-beat position)
    """
    squeeze = types.ndim == 1
    if squeeze:
        types = types[None]
        vocab_ids = vocab_ids[None]
        causal_beats = causal_beats[None]

    device = types.device
    t_list = types[0].detach().cpu().tolist()
    v_list = vocab_ids[0].detach().cpu().tolist()
    ticks = (causal_beats[0].detach().cpu().double() * 96.0).round().long().tolist()
    sink_set = set(sinks.detach().cpu().tolist()) if sinks is not None else set()

    out = [0] * len(t_list)
    prev_onset_tick = None

    for q, (t, v, tick) in enumerate(zip(t_list, v_list, ticks)):
        if q in sink_set or t == 16:
            prev_onset_tick = None
            if t == 16:
                out[q] = 96
                continue
        if t == 12:
            out[q] = int(tick % 96)
            prev_onset_tick = int(tick)
        elif t == 0 and v == 96:
            if prev_onset_tick is not None and (int(tick) - prev_onset_tick) <= 96:
                out[q] = int(prev_onset_tick % 96)
            else:
                out[q] = 96
        else:
            out[q] = int(tick % 96)
            prev_onset_tick = int(tick)

    result = torch.tensor(out, dtype=torch.long, device=device)
    return result if squeeze else result[None]


def compute_occupancy_id(
    types: Tensor,
    vocab_ids: Tensor,
    causal_beats: Tensor,
    timing=None,
    *,
    sinks: Tensor | None = None,
    physical_v6: bool = True,
    no_placeholders: bool = True,
) -> Tensor:
    """Compute active hold/slide occupancy ID in 0..15 on moment_query positions (0 elsewhere).

    occupancy_id = min(3, active_holds) * 4 + min(3, active_slides)
    """
    squeeze = types.ndim == 1
    if squeeze:
        types = types[None]
        vocab_ids = vocab_ids[None]
        causal_beats = causal_beats[None]

    if timing is None:
        timing = _default_timing()
    elif isinstance(timing, dict):
        timing = TimingMap.from_audio_timing(timing, validate_input=False)
    if physical_v6:
        from .v6_oracle import timing_v6

        timing = timing_v6(timing)

    from .physical_schema import VOCAB_OFFSETS as PHYSICAL_OFFSETS

    device = types.device
    t_list = types[0].detach().cpu().tolist()
    v_list = vocab_ids[0].detach().cpu().tolist()
    ticks = (causal_beats[0].detach().cpu().double() * 96.0).round().long().tolist()
    sink_set = set(sinks.detach().cpu().tolist()) if sinks is not None else set()
    global_duration = physical_v6 or any(t >= 19 for t in t_list)

    out = [0] * len(t_list)
    active_holds: list[float] = []
    active_slides: list[float] = []
    onset: int | None = None
    span: int | None = None

    def finish_span(stop: int) -> None:
        if span is None or onset is None:
            return
        hold_ends: list[float] = []
        slide_ends: list[float] = []
        continuation = 0
        wait = 96
        waiting = 0
        for i in range(span, stop):
            ti = t_list[i]
            if ti in (7, 10):
                continuation = 0
                waiting = 0
            elif ti == 24:
                waiting += 96
            elif ti == 23:
                wait = waiting + v_list[i] - PHYSICAL_OFFSETS[23] + 1
                waiting = 0
            elif ti in (14, 15):
                continuation += 96
            elif ti == 8:
                duration = continuation + v_list[i] - VOCAB_OFFSETS[8] + 1
                end_sec = duration_end(
                    timing,
                    ticks[onset],
                    duration,
                    slide=False,
                    **(dict(wait=wait) if global_duration else {}),
                )
                hold_ends.append(end_sec)
                continuation = 0
            elif ti == 11:
                duration = continuation + v_list[i] - VOCAB_OFFSETS[11] + 1
                end_sec = duration_end(
                    timing,
                    ticks[onset],
                    duration,
                    slide=True,
                    **(dict(wait=wait) if global_duration else {}),
                )
                slide_ends.append(end_sec)
                continuation = 0
        if hold_ends:
            active_holds.append(max(hold_ends))
        if slide_ends:
            active_slides.append(max(slide_ends))

    for q, (t, v, tick) in enumerate(zip(t_list, v_list, ticks)):
        if q in sink_set or t == 16:
            active_holds = []
            active_slides = []
            onset = None
            span = None
        elif t == 0 and v < 96:
            finish_span(q)
            onset = q
            span = q + 1
        elif t == 18 or (
            no_placeholders
            and (t == 19 or (t == 4 and (q == 0 or t_list[q - 1] != 9)))
            and onset is not None
            and q > onset + 1
        ):
            finish_span(q)
            span = q
        elif t == 12:
            finish_span(q)
            span = None
        elif t == 0 and v == 96:
            finish_span(q)
            span = None
        else:
            if onset is None:
                onset = q
                span = q

        is_mq = t in (16, 12) or (t == 0 and v == 96)
        if is_mq:
            now = timing.second(tick)
            active_holds = [end for end in active_holds if end > now + 1e-9]
            active_slides = [end for end in active_slides if end > now + 1e-9]
            h_cnt = min(3, len(active_holds))
            s_cnt = min(3, len(active_slides))
            out[q] = h_cnt * 4 + s_cnt
        else:
            out[q] = 0

    result = torch.tensor(out, dtype=torch.long, device=device)
    return result if squeeze else result[None]


def prepare_moment_conditioning(
    tokens,
    timing=None,
    *,
    sinks: Tensor | None = None,
    physical_v6: bool = True,
    no_placeholders: bool = True,
) -> dict[str, Tensor]:
    """Compute moment_query, moment_pos_id, and occupancy_id tensors for a token sequence."""
    if isinstance(tokens, dict):
        types = tokens["types"]
        vocab_ids = tokens["vocab_ids"]
        causal_beats = tokens["causal_beats"]
    else:
        types = tokens.types
        vocab_ids = tokens.vocab_ids
        causal_beats = tokens.causal_beats

    mq = moment_query_mask(types, vocab_ids)
    pos_id = compute_moment_pos_id(types, vocab_ids, causal_beats, sinks=sinks)
    occ_id = compute_occupancy_id(
        types,
        vocab_ids,
        causal_beats,
        timing,
        sinks=sinks,
        physical_v6=physical_v6,
        no_placeholders=no_placeholders,
    )
    return dict(moment_query=mq, moment_pos_id=pos_id, occupancy_id=occ_id)
