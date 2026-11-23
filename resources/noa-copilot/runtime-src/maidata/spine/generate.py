"""Autoregressive spine sampling with persistent self-attention and committed tail state."""

import argparse
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
import json
import math
from pathlib import Path
import numpy as np
import torch
from . import ARCHITECTURE
from .codec import SLIDE, to_score
from .pieces import inventory, pad_mask, END_MOMENT, NEXT_GROUP
from .data import audio_features_batch
from .model import SpineConfig, SpineModel
from .packing import BOS, BEAT_WORD, BEAT, CYCLE, TIME_COUNT, MOMENT
from .history import HistoryCache, sample_bank
from .features import (
    CLOCK_DIM,
    clocks,
    condition,
    endpoint_seconds,
    read_audio,
    pad_audio,
)
from .codec import PAD_MAP
from .tails import (
    FEEDBACK_DIM,
    end_pads,
    START,
    REM,
    ROUTE,
    grammar,
    release_tick,
    commit_pad,
    detail,
    feedback,
    read_tail,
    Hazard,
)
from maidata.multitask.grid import BeatGrid
from maidata.multitask.maimai import write_simai


_slide_sampling = ContextVar("slide_sampling", default=None)


@contextmanager
def slide_sampling(multiplier):
    if multiplier is None:
        yield
        return
    from maidata.flat.codec import FAMILY_IDS

    token = _slide_sampling.set((multiplier, FAMILY_IDS["SLIDE_HEAD"]) if multiplier != 1 else None)
    try:
        yield
    finally:
        _slide_sampling.reset(token)


def scaled_slide_probabilities(logits, choices, temperature, multiplier, slide_heads):
    """Multiply legal slide-head probability, preserving proportions within both groups."""
    if multiplier == 1 or not any(token in slide_heads for token in choices):
        return None
    mask = torch.tensor([token in slide_heads for token in choices], device=logits.device)
    scores = logits[..., list(choices)].float() / (temperature or 1.0)
    probs = scores.softmax(-1)
    slide = probs.masked_fill(~mask, 0)
    other = probs.masked_fill(mask, 0)
    mass = slide.sum(-1, keepdim=True)
    target = (mass * multiplier).clamp(max=1)
    tiny = torch.finfo(probs.dtype).tiny
    return slide * (target / mass.clamp_min(tiny)) + other * ((1 - target) / other.sum(-1, keepdim=True).clamp_min(tiny))


def pick(logits, temperature, allowed=None):
    if allowed is not None:
        policy = _slide_sampling.get()
        if policy:
            probs = scaled_slide_probabilities(logits, allowed, temperature, *policy)
            if probs is not None:
                i = torch.multinomial(probs, 1).squeeze(-1) if temperature else probs.argmax(-1)
                return torch.as_tensor(allowed, device=logits.device)[i]
        choices = torch.as_tensor(list(allowed), device=logits.device, dtype=torch.long)
        logits = logits.index_select(-1, choices)
    else:
        choices = None
    if temperature:
        out = torch.multinomial((logits.float() / temperature).softmax(-1), 1).squeeze(
            -1
        )
    else:
        out = logits.argmax(-1)
    return choices[out] if choices is not None else out


def sample_tail(
    model,
    state,
    kind,
    tick,
    pad,
    group,
    limit,
    temperature,
    target_audio,
    max_tail_fields=128,
):
    hidden = model.field_hidden(state)
    prev = START
    tokens = []
    rules = grammar(kind, tick, limit, max_tail_fields)
    allowed, beat = next(rules)
    while True:
        x, hidden = model.field_sequence(
            torch.tensor([[prev]], device=state.device), hidden
        )
        state = x[0, 0]
        logits = model.field_scores(state)
        if beat is not None:
            logits[REM:ROUTE] += model.endpoint_scores(state, target_audio(beat))
        prev = int(pick(logits, temperature, allowed))
        tokens.append(prev)
        try:
            allowed, beat = rules.send(prev)
        except StopIteration:
            return read_tail(tick, pad, group, kind, tokens)


def sample_groups(model, wh, temperature, max_groups):
    words = inventory()[1]
    piece_state = model.pieces.state(model.pieces.norm(wh))
    groups = []
    earlier = 0
    orientation = None
    while True:
        pads = []
        used = 0
        while True:
            context = torch.tensor([earlier, used, len(groups)], device=wh.device)
            logits = model.pieces(piece_state, *context.unbind())
            if len(groups) + 1 >= max_groups:
                logits[NEXT_GROUP] = -torch.inf
            token = int(pick(logits, temperature))
            if token in (END_MOMENT, NEXT_GROUP):
                break
            piece = words[token]
            if not temperature and len(piece) > 1:
                # Conjugate lane states reverse the physical ordering. An exactly
                # symmetric state has no deterministic equivariant tie-break.
                if orientation is None:
                    odd = wh[0] - wh[1]
                    orientation = float(odd[odd.abs().argmax()])
                if orientation:
                    left = orientation > 0
                else:
                    conjugate = tuple(sorted(PAD_MAP[8][p] for p in piece))
                    left = (
                        piece < conjugate
                        if piece != conjugate
                        else bool(torch.randint(2, ()))
                    )
                piece = sorted(piece, key=lambda p: p if left else PAD_MAP[8][p])
            pads.extend(piece)
            used |= pad_mask(piece)
        if temperature and len(pads) > 1:
            pads = [pads[i] for i in torch.randperm(len(pads)).tolist()]
        groups.append(pads)
        if token == END_MOMENT:
            return groups
        earlier |= used


@torch.inference_mode()
def generate(
    model,
    bank,
    grid,
    metadata,
    end_tick,
    *,
    temperature=0.0,
    max_words=None,
    max_pseudo_groups=8,
    max_tail_fields=128,
    max_slot_events=8,
    collision_retries=2,
    prefix_actions=(),
    start_tick=0,
    stop_tick=None,
    on_step=None,
):
    device = bank.frames.device

    def val(a):
        return torch.as_tensor(a, device=device)

    def clock(t, probes=()):
        return clocks(grid, val(t), end_tick, probes)

    cfg = model.config
    idx = val([0])
    scalar = val(0)

    @lru_cache(maxsize=1)
    def audio_keys():
        return model.endpoint_keys(bank.frames), bank.lengths

    @lru_cache(maxsize=16)
    def target_audio(beat):
        seconds = endpoint_seconds(grid, val(beat))
        return sample_bank(audio_keys(), scalar, seconds)

    values, charter_id = condition(metadata, cfg.designer_vocab)
    cond = model.conditioning(val([values]), val([charter_id]))
    sc = HistoryCache(len(model.spine), cfg.history_beats, sink=True)
    tc = HistoryCache(len(model.tail), cfg.history_beats)
    haz = Hazard(device=device)
    history = np.zeros(FEEDBACK_DIM, np.float32)
    committed_audio = bank.frames.new_zeros((1, 1, cfg.spine_dim // 2))
    empty_feedback = bank.frames.new_zeros(1, 1, FEEDBACK_DIM)
    previous_feedback = empty_feedback
    footprint_bits = 0
    prev = BOS
    prevdt = first = last = word_count = 0
    result = []
    last_onset = -96
    last_small = owed = False
    time_ids = torch.arange(TIME_COUNT, device=device)
    forced_actions = iter(prefix_actions)
    onset_limit = end_tick if stop_tick is None else min(end_tick, stop_tick - 1)
    while last <= onset_limit and (max_words is None or word_count < max_words):
        if on_step is not None:
            on_step(last, result)
        forced = next(forced_actions, None)
        row = val([[[prev, prevdt, last, 0]]])
        x = model.spine_states(
            row,
            previous_feedback,
            val(haz.query(last))[None, None],
            cond,
            bank,
            idx,
            committed_audio,
            clock([[last]], cfg.beat_probes),
            cache=sc,
            cursor=(first, last),
            footprint=val([[footprint_bits]]),
        )[0, 0]
        beat = last // 96
        next_beat = 0 if prev == BOS else beat + 1
        if forced is not None:
            dt = forced[0]
        elif owed and max(start_tick, last_onset + 1) > min(
            onset_limit, last_onset + 2
        ):
            # An externally supplied prefix can already owe an onset outside the edit span.
            break
        elif prev == BOS or (not owed and next_beat * 96 <= start_tick):
            dt = BEAT + next_beat % CYCLE
        else:
            scores = model.timing.times(
                model.dt(x).mean(-2),
                val(prev),
                val(last),
                val(last_onset),
                val(owed),
                onset_limit,
                val(last_small),
            )
            ticks = beat * 96 + time_ids
            scores = scores.masked_fill(
                (ticks < start_tick) & (time_ids < BEAT), -torch.inf
            )
            if not last_small and max_words is not None and word_count + 1 == max_words:
                # Opening a micro-run consumes this slot and needs one more.
                small = (
                    (ticks > last_onset) & (ticks <= last_onset + 2) & (time_ids < BEAT)
                )
                scores = scores.masked_fill(small, -torch.inf)
            dt = int(pick(scores, temperature))
        prevdt = dt
        if dt >= BEAT:
            footprint_bits = 0
            previous_feedback = empty_feedback
            prev, first, last = BEAT_WORD, next_beat * 96, next_beat * 96
            continue
        first = beat * 96 + dt
        onset_clock = clock(first)
        onset = model.onset(sample_bank(bank[:2], scalar, onset_clock[CLOCK_DIM]))
        wh = model.word_state(x, val(dt), onset_clock, onset)
        small = first - last_onset in (1, 2)
        last_onset = first
        owed, last_small = small and not last_small, small
        prev, last = MOMENT, first
        committed_audio = onset[None, None]
        if forced is not None:
            from .editor_prefix import replay_moment

            footprint_bits = pad_mask(set(n[0] for n in forced[1]))
            replay_moment(
                model,
                bank,
                wh,
                onset,
                first,
                forced[1],
                history,
                haz,
                tc,
                clock,
                metadata.get("song_id") or 0,
            )
            previous_feedback = val(history)[None, None]
            continue
        groups = sample_groups(model, wh, temperature, max_pseudo_groups)
        pads = sorted({p for group in groups for p in group})
        history.fill(0)
        footprint_bits = pad_mask(pads)
        footprint = wh.new_zeros(41)
        footprint[pads] = 1
        geom = model.footprint(footprint)
        word_count += 1
        committed = wh.new_zeros(41, 6)
        tick = first
        notes = []
        for group, members in enumerate(groups):
            for pad in members:
                p = val(pad)
                occurrence = 0
                prior_non_slide = False
                while True:
                    sensor = pad if pad < 8 else pad - 8
                    sx = model.slot_state(
                        wh,
                        geom,
                        p,
                        val(group),
                        val(occurrence),
                        onset,
                        val(haz.at_sensor(tick, sensor)),
                        committed,
                    )
                    sn = model.slot_norm(sx)
                    kinds = model.kind_scores(
                        sn, p, val(prior_non_slide), val(tick < end_tick)
                    )
                    kind = int(pick(kinds, temperature))
                    exhausted = False
                    note = (pad, group, kind, 0, 0, 0, [])
                    if kind:
                        pending_tail = []
                        times = val([[tick]])
                        tx = model.tail_states(
                            sx[None, None],
                            val([[kind]]),
                            times,
                            clock([[tick]], cfg.tail_probes),
                            bank,
                            idx,
                            val(haz.query(tick))[None, None],
                            geom[None, None],
                            cache=tc,
                            tick=tick,
                            pending=pending_tail,
                        )[0, 0]
                        attempts = collision_retries + 1 if temperature else 1
                        contacts = [(tick, other) for other in pads if other != pad]
                        for _ in range(attempts):
                            note = sample_tail(
                                model,
                                tx,
                                kind,
                                tick,
                                pad,
                                group,
                                end_tick,
                                temperature,
                                target_audio,
                                max_tail_fields,
                            )
                            if not haz.conflicts(tick, note, contacts):
                                break
                        else:
                            exhausted = True
                            if prior_non_slide:
                                # No legal additional route: stop this pad without committing a duplicate.
                                break
                            kind = 0
                            note = (pad, group, 0, 0, 0, 0, [])
                    descriptor = val(detail(tick, note))
                    if kind:
                        haz.commit(tick, note)
                        model.commit_tail(
                            tc,
                            pending_tail,
                            tick,
                            times,
                            p,
                            descriptor,
                            val(end_pads(note)),
                            release_tick(note),
                        )
                    features = model.flags_input(sn, val(kind), descriptor)
                    flags = int(pick(model.flag_scores(features, p), temperature))
                    note = (*note[:3], flags, *note[4:])
                    notes.append(note)
                    feedback(first, tick, note, history)
                    commit_pad(committed, pad, kind, flags)
                    prior_non_slide |= kind != SLIDE
                    if (
                        exhausted
                        or not haz.slide_room(tick, pad, end_tick)
                        or occurrence + 1 >= max_slot_events
                    ):
                        break
                    if not int(
                        pick(
                            model.repeat_scores(features, val(flags), val(True)),
                            temperature,
                        )
                    ):
                        break
                    occurrence += 1
        previous_feedback = val(history)[None, None]
        result.append((tick, notes))
    return result


def load_checkpoint(path, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["architecture"] != ARCHITECTURE:
        raise ValueError("Expected a moment205 checkpoint")
    cfg = state.get("config") or {}
    model = SpineModel(SpineConfig(**cfg.get("model", {})))
    model.load_state_dict(state["model"])
    return (
        model.eval().to(device),
        cfg,
        {
            k: state.get(k)
            for k in ("architecture", "epoch", "step", "source", "binding")
        },
    )


def generation_audio(args, cfg, device):
    if args.compact_manifest:
        doc = json.loads(args.compact_manifest.read_text(encoding="utf8"))
        manifest = args.compact_manifest
        if "audio_manifest" in doc:
            manifest = manifest.parent / doc["audio_manifest"]
            doc = json.loads(manifest.read_text(encoding="utf8"))
        info = doc["audio"][args.audio_id]
        feat = pad_audio(
            [
                read_audio(
                    manifest.parent / "audio" / args.audio_id,
                    info,
                    info["samples"] / 44100,
                )
            ]
        )
    else:
        feat = audio_features_batch(
            [(args.audio, args.macro_features)],
            device,
            args.mert_model or cfg["audio_models"][0],
            args.muq_model or cfg["audio_models"][1],
            args.frontend_windows_per_song or 4,
        )
    feat = {k: torch.as_tensor(v, device=device) for k, v in feat.items()}
    for key in ("mert", "mert_l6", "muq", "mel"):
        feat[key] = feat[key].to(torch.bfloat16)
    return feat


def generation_cli(argv, *, loader, sampler, name, audio_loader=generation_audio):
    p = argparse.ArgumentParser(description=f"{name} full-song chart generation")
    p.add_argument("--joint-checkpoint", "--checkpoint", type=Path, required=True)
    p.add_argument("--audio")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--metadata", type=Path)
    p.add_argument("--charter", "--designer", dest="charter")
    p.add_argument("--compact-manifest", type=Path)
    p.add_argument("--audio-id")
    p.add_argument(
        "--collision-retries",
        type=int,
        default=2,
        help=None if sampler is generate else argparse.SUPPRESS,
    )
    legacy_help = None if audio_loader is generation_audio else argparse.SUPPRESS
    p.add_argument("--macro-features", type=Path, help=legacy_help)
    p.add_argument("--bpm", type=float)
    p.add_argument("--offset", type=float)
    p.add_argument("--level", type=float)
    p.add_argument("--difficulty", type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=20261111)
    p.add_argument("--max-moments", "--max-words", dest="max_words", type=int)
    p.add_argument("--mert-model")
    p.add_argument("--muq-model", help=legacy_help)
    p.add_argument("--frontend-windows-per-song", type=int, help=legacy_help)
    p.add_argument("--max-pseudo-groups", type=int, default=8)
    p.add_argument("--max-tail-fields", type=int, default=128)
    p.add_argument("--max-slot-events", type=int, default=8)
    args = p.parse_args(argv)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model, cfg, provenance = loader(args.joint_checkpoint, device)
    meta = (
        json.loads(args.metadata.read_text(encoding="utf-8-sig"))
        if args.metadata
        else {}
    )
    for field in ("level", "difficulty", "bpm", "charter"):
        if getattr(args, field) is not None:
            meta[field] = getattr(args, field)
    if args.offset is not None:
        meta["offset_seconds"] = args.offset
    if args.bpm is not None:
        meta.update(
            bpm_segments=[(meta.get("offset_seconds", 0.0), args.bpm)],
            grid_source="explicit_bpm",
        )
    elif meta.get("bpm_segments") or meta.get("bpm"):
        meta["grid_source"] = "metadata"
    else:
        meta.update(bpm=120.0, grid_source="default_120_bpm")
    meta.setdefault("level", 13)
    meta.setdefault("difficulty", 5)
    grid = BeatGrid(meta)
    with (
        torch.inference_mode(),
        torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=device.type == "cuda" and cfg.get("bf16", True),
        ),
    ):
        feat = audio_loader(args, cfg, device)
        bank = model.fusion(feat)
        duration = int(feat["sample_lengths"][0]) / 44100
        end = math.floor(grid.second_to_beat(duration) * 96)
        moments = sampler(
            model,
            bank,
            grid,
            meta,
            end,
            temperature=args.temperature,
            max_words=args.max_words,
            max_pseudo_groups=args.max_pseudo_groups,
            max_tail_fields=args.max_tail_fields,
            max_slot_events=args.max_slot_events,
            collision_retries=args.collision_retries,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_simai(
        to_score(moments),
        args.output,
        bpm=meta.get("bpm", 120),
        offset=meta.get("offset_seconds", 0),
        segments=meta.get("bpm_segments"),
        level=meta.get("level", 13),
        difficulty=meta.get("difficulty", 5),
        title=meta.get("title", ""),
        designer=name,
        pseudo_each="backtick",
        audio_duration=duration,
    )
    record = dict(
        checkpoint=str(args.joint_checkpoint),
        provenance=provenance,
        metadata=meta,
        seed=args.seed,
        temperature=args.temperature,
        moment_limit=args.max_words,
        max_pseudo_groups=args.max_pseudo_groups,
        max_slot_events=args.max_slot_events,
        max_tail_fields=args.max_tail_fields,
        moments=len(moments),
        duration_seconds=duration,
    )
    args.output.with_suffix(args.output.suffix + ".provenance.json").write_text(
        json.dumps(record, ensure_ascii=False), encoding="utf8"
    )
    print(
        json.dumps(
            dict(
                output=str(args.output),
                moments=len(moments),
                notes=sum(len(ns) for _, ns in moments),
            )
        )
    )


def main(argv=None):
    return generation_cli(argv, loader=load_checkpoint, sampler=generate, name="moment205")
