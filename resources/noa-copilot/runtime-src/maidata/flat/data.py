"""Pack only chart supervision. Reference a separate dual-rate Flat314 audio pack."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import tempfile
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import gzip
import hashlib
from itertools import chain
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from maidata.multitask.grid import BeatGrid
from maidata.spine import official
from maidata.spine.codec import HOLD, SLIDE, mirror_notes
from maidata.spine.features import clocks, condition
from . import ARCHITECTURE_V20, AUDIO_PACK_SCHEMA, DRAFT_ARCHITECTURES, PACK_SCHEMA, TAIKO_PACK_SCHEMA
from .audio import (
    FLAT_RATES,
    GRID_STEPS_PER_BEAT,
    STREAMS,
    flat_audio_manifest,
    pad_flat_audio,
    read_flat_audio,
)
from .taiko import encode_taiko, taiko_draft_targets, compute_taiko_beat_te_v3
from .codec import SIZE_V20
from .drafter import condition_buckets, draft_anchor_valid
from .reconstruction import build_draft_targets, pack_targets
from .emotion import compute_beat_te_v3, quantize_plan4
from .geometry import DISK_DISTANCE_SQ, active_hand_clusters
from .codec import (
    FAMILIES,
    GRID32_TICKS,
    PAGE_ID,
    Grammar,
    SIZE,
    TIMING_INPUTS,
    FAMILY_OFFSET,
    WIDTHS,
    TICK_TO_GRID32,
    chart_tokens,
    resolve_version_profile,
    snap_moments_grid32,
    field_horizon,
)
from .descriptors import HeadRows, active_spans, describe, pack_heads


@lru_cache(maxsize=8192)
def legal_bits(ids):
    mask = np.zeros(SIZE, np.uint8)
    mask[list(ids)] = 1
    return np.packbits(mask)


def prefix_info(g):
    recent = 0 <= g.last_onset and g.tick - g.last_onset <= 96
    pos = TICK_TO_GRID32[g.last_onset % 96] if recent else 32
    return (
        g.tick,
        g.query_tick,
        g.moment,
        pos if g.last in TIMING_INPUTS else 33,
    )


def union_pairs(seq):
    """Sweep backwards: the remaining set grows until the start of its owner."""
    pads, touch_chain, routes, pairs = set(), set(), set(), []
    for i in reversed(range(len(seq) - 1)):
        family = FAMILIES[seq[i]][0]
        target = seq[i + 1]
        if (
            family in ("TOUCH_PAD", "TOUCH_FIREWORK")
            and FAMILIES[target][0] == "TOUCH_PAD"
        ):
            remaining = touch_chain
            pads.add(target)
        elif family in ("CHORD", "MOMENT", "PSEUDO_EACH"):
            remaining = pads
            touch_chain.clear()
        elif family in ("SLIDE_BRANCH", "SLIDE_HEAD"):
            remaining = routes
        else:
            continue
        pairs.extend((i, alt) for alt in sorted(remaining) if alt != target)
        remaining.add(target)
        if family in ("MOMENT", "PSEUDO_EACH", "SLIDE_HEAD"):
            remaining.clear()
    return np.asarray(pairs, np.int32).reshape(-1, 2)


def union_indices(pairs):
    # Compute the ragged row map in the worker. A CUDA unique would force a host sync.
    rows, inverse = np.unique(pairs[:, 0], return_inverse=True)
    return dict(
        rows=torch.from_numpy(rows.astype(np.int64)),
        inverse=torch.from_numpy(inverse.reshape(-1).astype(np.int64)),
        cols=torch.from_numpy(pairs[:, 1].astype(np.int64)),
    )


def record_commit(g, commits):
    if g.active and g.active[-1][2][-1] == g.index:
        end, _, slots, *_ = g.active[-1]
        commits.extend((g.index, end, slot) for slot in slots)


LONG_PHASES = {
    "typed",
    "route",
    "hold_beat",
    "wait_beat",
    "end_beat",
    "offset",
    "done",
    "slide_done",
}


def decision_weight(g, family, clusters, touch, idle, speed):
    if g.last in TIMING_INPUTS:
        return 1.0
    if family == "MOMENT" and not clusters and idle > 0.50:
        return 0.40 + 0.60 * 2 ** (-(idle - 0.50) / 0.50)
    # Grammar appends a branch on its closing OFFSET. The first closed branch
    # alone therefore gives no bonus to the decision between EOM and SLIDE_BRANCH.
    later_branch = g.note is not None and len(g.note[6]) > int(g.phase == "slide_done")
    multi = family == "CHORD" or g.group > 0 or len(g.notes) > 1 or later_branch
    long_step = g.phase in LONG_PHASES
    complexity = 0.20 * min(2, len(clusters)) + 0.15 * (
        int(touch) + int(multi) + int(long_step)
    )
    return min(2.20, 1.0 + speed + complexity)


def encode(
    moments,
    grid,
    end_tick,
    field_limit,
    song=0,
    mirrored=False,
    *,
    seed=0,
    epoch=0,
    validate=True,
    slide_icebreak_beats=6.0,
    slide_icebreak_weight=1.0,
    profile=None,
    plan4_anchor=False,
    physical_plan=None,
):
    horizon = max(end_tick, moments[-1][0] if moments else 0)
    seq = (
        list(physical_plan["tokens"])
        if physical_plan is not None
        else list(
            chart_tokens(moments, horizon, song, mirrored, seed=seed, epoch=epoch)
        )
    )
    g = Grammar(field_limit, onset_limit=horizon, profile=profile, plan4_anchor=plan4_anchor)
    rows, allowed, protected, states = [], [], [], []
    heads, evict = HeadRows(), []
    grid = grid or BeatGrid()
    commits, weights, zones, occupancy = [], [], [], []
    zone_masks, slide_occupancy, lookahead_rows = [], [], []
    active_slides = []
    boundary_cluster = ((0,),)
    conv_ticks, mert_ticks, pages = [], [], []
    fork_requests = (
        physical_plan["fork_requests"] if physical_plan is not None else None
    )
    upairs = union_pairs(seq)
    fork_snapshots = {} if fork_requests else None
    fork_comb_rows = {} if fork_requests else None
    real_to_comb = [] if physical_plan is not None else None
    in_tokens = [] if physical_plan is not None else None
    tgt_tokens = [] if physical_plan is not None else None
    alt_map = None
    if fork_requests:
        from .physical import clone_grammar, should_fold_fork

        alt_map = defaultdict(set)
        for r_u, c_u in upairs.tolist():
            alt_map[r_u].add(c_u)
    slide_anchor = None  # First chart onset, then the most recently consumed star.
    icebreak_ticks = round(slide_icebreak_beats * 96)
    prev_onset = last_release = -math.inf
    idle, speed = math.inf, 0.0
    clusters, touch = [], False
    for i, idx in enumerate(seq[:-1]):
        family, _ = FAMILIES[idx]
        target, value = FAMILIES[seq[i + 1]]
        old_active = g.active
        g.advance(idx)
        pos = g.index
        if real_to_comb is not None:
            real_to_comb.append(pos)
            in_tokens.append(idx)
            tgt_tokens.append(seq[i + 1])
        record_commit(g, commits)
        if family in ("BEAT", "MOMENT"):
            for end, *_ in old_active:
                if end <= g.tick:
                    last_release = max(last_release, grid.second(end))
        if family == "SLIDE_HEAD":
            if slide_anchor is not None and g.tick - slide_anchor >= icebreak_ticks:
                # The star confirms the open key and its modifiers as positive labels.
                for row in range(g.note_start, pos):
                    weights[row] *= slide_icebreak_weight
            slide_anchor = g.tick
        if family == "MOMENT":
            if slide_anchor is None:
                slide_anchor = g.tick
            now = grid.second(g.tick)
            gap, idle = now - prev_onset, now - max(prev_onset, last_release)
            speed = 0.35 * math.log2(0.20 / max(gap, 0.04)) if gap < 0.20 else 0.0
            prev_onset = now
        if family in ("MOMENT", "BEAT", "HOLD_END_OFFSET", "SLIDE_END_OFFSET"):
            clusters = active_hand_clusters(g.active, g.tick)
            active_slides = [e[2] for e in g.active if e[3] == SLIDE]
            touch = any(e[3] == HOLD and e[5] >= 8 for e in g.active)
        weights.append(decision_weight(g, family, clusters, touch, idle, speed))
        zone = (
            -1
            if g.note is None or idx in TIMING_INPUTS
            else (g.pad if g.pad < 8 else g.pad - 8)
        )
        zones.append(zone)
        zone_masks.append((g.open_touch_mask or (1 << zone)) if zone >= 0 else 0)
        if i:
            slide_occupancy.append((pos, active_slides or boundary_cluster))
        if idx not in TIMING_INPUTS:
            occupancy.append((pos, clusters or boundary_cluster))
            if g.phase not in ("hold_beat", "wait_beat", "end_beat", "offset"):
                lookahead_rows.append(pos)
        pages.append(g.page_first)
        if idx in TIMING_INPUTS and target == "MOMENT":
            conv_ticks.append((pos, g.beat * 96 + GRID32_TICKS[value]))
        elif g.phase == "offset":
            conv_ticks.append((pos, g.target_beat * 96 + GRID32_TICKS[value]))
        elif (
            g.phase in ("wait_beat", "end_beat", "hold_beat") and seq[i + 1] != PAGE_ID
        ):
            # Beat and GRID32 offset are consecutive fields even after a page.
            beat = g.beat + (value - g.beat) % 16 + g.pages * 16
            mert_ticks.append((pos, beat * 96 + GRID32_TICKS[FAMILIES[seq[i + 2]][1]]))
        rows.append(prefix_info(g))
        heads.append(g)
        evict.append(g.evicts(seq[i + 1]))
        legal = tuple(g.legal())
        if validate and seq[i + 1] not in legal:
            raise ValueError(
                f"Unrepresentable flat target {seq[i + 1]} in {g.phase} at {g.tick} for {song}"
            )
        allowed.append(legal_bits(legal))
        if idx in TIMING_INPUTS:
            states.append(g.active_state())
            if g.last_onset_index >= 0 and g.tick - g.last_onset <= 96:
                protected.append((pos, g.last_onset_index, g.last_onset_index + 1))
            for _, moment, slots, *_ in g.active:
                protected.append((pos, moment, moment + 1))
                protected.extend((pos, j, j + 1) for j in slots)
        if fork_requests and i in fork_requests:
            parent_legal_set = frozenset(legal)
            positives_p = alt_map[i] | {int(seq[i + 1])}
            requests = fork_requests[i]
            prefixes = {fork[:-1] for fork in requests}
            for ftup in requests:
                if len(ftup) == 1:
                    if ftup[0] in positives_p or ftup[0] not in parent_legal_set:
                        continue
                else:
                    prior = fork_snapshots.get((i, ftup[:-1]))
                    if prior is None or ftup[-1] not in prior["legal_set"]:
                        continue
                gf = clone_grammar(g, pos + 1)
                for kf, ftok in enumerate(ftup):
                    gf.index = pos + kf
                    gf.advance(ftok)
                f_legal = tuple(gf.legal())
                f_legal_set = frozenset(f_legal)
                snap = dict(
                    tok=ftup[-1],
                    row=prefix_info(gf),
                    head=describe(gf),
                    page=gf.page_first,
                    legal_bits=legal_bits(f_legal),
                    legal_set=f_legal_set,
                )
                fork_snapshots[(i, ftup)] = snap
                if f_legal and not should_fold_fork(
                    physical_plan["grouped"].get((i, ftup)),
                    ftup,
                    f_legal_set,
                    ftup in prefixes,
                ):
                    q = len(rows)
                    g.index += 1
                    rt = snap["row"]
                    if FAMILIES[snap["tok"]][0] == "MOMENT":
                        rt = (rt[0], rt[1], q, rt[3])
                    rows.append(rt)
                    h_id, h_rec = snap["head"]
                    heads.ids.append(h_id)
                    heads.records[h_id].append(h_rec)
                    evict.append(True)
                    pages.append(snap["page"])
                    allowed.append(snap["legal_bits"])
                    weights.append(0.0)
                    zones.append(-1)
                    zone_masks.append(0)
                    in_tokens.append(snap["tok"])
                    tgt_tokens.append(0)
                    fork_comb_rows[(i, ftup)] = q

    weights_arr = np.asarray(weights, np.float32)
    allowed_arr = np.stack(allowed)
    extra = {}
    if physical_plan is not None:
        from .physical import finalize_physical_plan

        real_to_comb_arr = np.asarray(real_to_comb, np.int64)
        real_mask = np.zeros(len(rows), dtype=bool)
        real_mask[real_to_comb_arr] = True
        if physical_plan["gt_rows"] and physical_plan["gt_conflict_ce_scale"] != 1.0:
            scale = physical_plan["gt_conflict_ce_scale"]
            for r_gt in physical_plan["gt_rows"]:
                weights_arr[real_to_comb_arr[r_gt]] *= scale
        phys_tensors = finalize_physical_plan(
            physical_plan,
            seq,
            upairs,
            allowed_arr,
            fork_snapshots or {},
            real_to_comb_arr,
            fork_comb_rows or {},
        )
        if len(upairs):
            upairs = upairs.copy()
            upairs[:, 0] = real_to_comb_arr[upairs[:, 0]]
        extra.update(
            real_mask=real_mask,
            real_rows=real_to_comb_arr,
            input_tokens=np.asarray(in_tokens, np.int16),
            target_tokens=np.asarray(tgt_tokens, np.int16),
            physical=phys_tensors,
        )

    rows = np.asarray(rows, np.int32)
    lh_rows = np.asarray(lookahead_rows, np.int64)
    slide_ticks = np.asarray(
        [t for t, notes in moments if any(n[2] == SLIDE for n in notes)], np.int64
    )
    bounds = rows[lh_rows, 0, None] + np.array([0, 96, 192])
    future = np.searchsorted(slide_ticks, bounds, side="right")
    return dict(
        chart_moments=moments,
        union_pairs=upairs,
        weights=weights_arr,
        zones=np.asarray(zones),
        zone_masks=np.asarray(zone_masks, np.int64),
        slide_occupancy=slide_occupancy,
        slide_lookahead=(lh_rows, (np.diff(future, axis=1) > 0).astype(np.float32)),
        occupancy=occupancy,
        field_limit=field_limit,
        pages=np.asarray(pages),
        conv_ticks=conv_ticks,
        mert_ticks=mert_ticks,
        active_spans=active_spans(commits, rows[:, 0]),
        **heads.arrays(),
        evict=np.asarray(evict, bool),
        tokens=np.asarray(seq, np.int16),
        rows=rows,
        legal=allowed_arr,
        active_state=np.array(states, np.uint8),
        protected=np.asarray(protected, np.int32).reshape(-1, 3),
        **extra,
    )


def pack_events(moments):
    words = []
    for tick, notes in moments:
        words.extend((tick, len(notes)))
        for *fields, branches in notes:
            words.extend((*fields, len(branches)))
            for launch, end, brk, routes in branches:
                words.extend((launch, end, brk, len(routes), *routes))
    return np.asarray(words, np.int32)


def unpack_events(words):
    from itertools import islice

    it = iter(words.tolist())
    moments = []
    for tick in it:
        notes = []
        for _ in range(next(it)):
            fields = list(islice(it, 6))
            branches = []
            for _ in range(next(it)):
                launch, end, brk, n = islice(it, 4)
                branches.append((launch, end, brk, list(islice(it, n))))
            notes.append([*fields, branches])
        moments.append((tick, notes))
    return moments


def chart_metadata(folder, row, metadata):
    meta = dict(metadata)
    meta["song_id"] = str(row["song"])
    meta["difficulty"] = int(meta.get("difficulty") or row["id"].split(":")[1])
    if not meta.get("version") and row.get("audio"):
        maidata = (folder / row["audio"]).with_name("maidata.txt")
        if maidata.is_file():
            match = re.search(
                r"^&version=([^\r\n]+)", maidata.read_text("utf-8-sig"), re.M
            )
            if match:
                meta["version"] = match[1].strip()
    return meta


def bake_one(job):
    folder, row, samples = job
    with gzip.open(folder / row["labels"], "rt", encoding="utf8") as stream:
        chart = json.load(stream)
    meta = chart_metadata(folder, row, chart["metadata"])
    moments = chart["moments"]
    profile = resolve_version_profile(meta)
    grid = BeatGrid(meta)
    end = math.floor(grid.second_to_beat(samples / 44100) * 96)
    moments = snap_moments_grid32(moments)
    limit = field_horizon(moments, end)
    views = []
    for flip in (False, True):
        source = [(t, mirror_notes(notes)) for t, notes in moments] if flip else moments
        encode(source, grid, end, limit, row["song"], flip, profile=profile)
        views.append(pack_events(source))
    return dict(
        id=row["id"],
        song=row["song"],
        title=row.get("title", ""),
        split=row["split"],
        audio_id=row["audio_id"],
        moments=len(moments),
        meta=meta,
        end_tick=end,
        field_limit=limit,
    ), views


def bake(normalized, audio_manifest, output, workers=12):
    doc = json.loads(normalized.read_text("utf8"))
    native = json.loads(audio_manifest.read_text("utf8"))
    if (
        doc["schema"] != "moment205.normalized.v2"
        or native["schema"] != AUDIO_PACK_SCHEMA
    ):
        raise ValueError(
            "Flat314 needs normalized.v2 labels and a Flat314 dual-rate audio manifest"
        )
    output.mkdir(parents=True, exist_ok=True)
    from importlib.resources import files

    split = json.loads(
        files("maidata.multitask").joinpath("assets/spine-split.json").read_text("utf8")
    )
    with_fit = {r["id"] for r in split if r["split"] == "fit" and official(r)}
    jobs = [
        (normalized.parent, r, native["audio"][r["audio_id"]]["samples"])
        for r in doc["records"]
        if r["split"] == "fit" and official(r) and r["id"] in with_fit
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        records, chunks, total = [], [], 0
        for record, views in pool.map(bake_one, jobs):
            record["views"] = []
            for view in views:
                record["views"].append([total, total + len(view)])
                chunks.append(view)
                total += len(view)
            records.append(record)
    words = np.concatenate(chunks) if chunks else np.empty(0, np.int32)
    # Content-addressing keeps readers of the previous manifest valid during a rebake.
    archive = "charts-" + hashlib.sha256(words).hexdigest()[:20] + ".npz"
    if not (output / archive).exists():
        with tempfile.NamedTemporaryFile(
            dir=output, suffix=".npz", delete=False
        ) as stream:
            np.savez_compressed(stream, events=words)
        Path(stream.name).replace(output / archive)
    try:
        ref = os.path.relpath(audio_manifest.resolve(), output.resolve())
    except ValueError:  # separate Windows volumes
        ref = str(audio_manifest.resolve())
    result = dict(
        schema=PACK_SCHEMA,
        archive=archive,
        audio_manifest=Path(ref).as_posix(),
        audio_models=native["audio_models"],
        records=records,
    )
    with tempfile.NamedTemporaryFile(
        mode="w", dir=output, encoding="utf8", delete=False
    ) as stream:
        json.dump(result, stream, ensure_ascii=False)
    Path(stream.name).replace(output / "manifest.json")


def sample_history(rng):
    p = rng.random()
    if p < 0.75:
        return 8
    drop = rng.uniform(0.15, 0.25) if p < 0.95 else rng.uniform(0.5, 1.0)
    return round(8 * (1 - drop))


class ChartGroups(Dataset):
    def __init__(self, manifest, seed, config, *, physical_cfg=None):
        self.folder = manifest.parent
        doc = json.loads(manifest.read_text("utf8"))
        if doc["schema"] != PACK_SCHEMA:
            raise ValueError("Expected a Flat314 v2 event pack")
        self.archive = self.folder / doc["archive"]
        self.events = None
        asset = self.folder / doc["audio_manifest"]
        native_bytes = asset.read_bytes()
        self.native = flat_audio_manifest(json.loads(native_bytes))
        self.audio_binding = hashlib.sha256(native_bytes).digest()
        self.audio_folder = asset.parent / "audio"
        self.audio_models = self.native["audio_models"]
        self.seed, self.config = seed, config
        self.physical = None
        self.physical_cfg = physical_cfg
        if physical_cfg and physical_cfg.get("physical_cache"):
            from .physical import PhysicalPack

            self.physical = PhysicalPack.open(
                physical_cfg["physical_cache"],
                manifest,
                workers=physical_cfg.get("workers", 12),
            )
        groups = defaultdict(list)
        for row in doc["records"]:
            if official(row) and row["split"] == "fit":
                groups[row["audio_id"]].append(row)
        self.groups = list(groups.items())

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, item):
        epoch, index = item
        key, rows = self.groups[index]
        info = self.native["audio"][key]
        audio = read_flat_audio(self.audio_folder / key, info)
        charts = []
        for row in rows:
            song = int(row["song"])
            flip = int((song + epoch) % 3 == 0)
            rng = np.random.default_rng(
                [self.seed, epoch, song, int(row["meta"]["difficulty"])]
            )
            if self.events is None:
                with np.load(self.archive) as pack:
                    self.events = pack["events"]
            lo, hi = row["views"][flip]
            meta = dict(row["meta"], song_id=row["song"])
            grid = BeatGrid(meta)
            moments = unpack_events(self.events[lo:hi])
            plan = None
            if self.physical is not None:
                plan = self.physical.prepare_plan(
                    row["id"],
                    flip,
                    moments,
                    row["end_tick"],
                    song,
                    seed=self.seed,
                    epoch=epoch,
                    warning_weight=self.physical_cfg.get("physical_warning_weight", 0.0),
                    use_share=self.physical_cfg.get("physical_use_share", True),
                    gt_conflict_ce_scale=self.physical_cfg.get(
                        "gt_conflict_ce_scale", 0.25
                    ),
                )
            arrays = encode(
                moments,
                grid,
                row["end_tick"],
                row["field_limit"],
                song,
                flip,
                seed=self.seed,
                epoch=epoch,
                validate=False,
                slide_icebreak_beats=self.config.slide_icebreak_beats,
                slide_icebreak_weight=self.config.slide_icebreak_weight,
                profile=resolve_version_profile(meta),
                plan4_anchor=self.config.inline_plan,
                physical_plan=plan,
            )
            charts.append(
                prepare_chart(
                    arrays,
                    meta,
                    row["end_tick"],
                    self.config,
                    audio["mel_lengths"],
                    sample_history(rng),
                    grid=grid,
                    plan_rng=rng,
                    for_batch=True,
                )
            )
        return audio, charts


class TaikoGroups(Dataset):
    def __init__(self, manifest, seed, config):
        doc = json.loads(manifest.read_bytes())
        if doc["schema"] != TAIKO_PACK_SCHEMA:
            raise ValueError("Expected a packed Taiko manifest")
        self.seed, self.config = seed, config
        asset = manifest.parent / doc["audio_manifest"]
        native = asset.read_bytes()
        self.audio_binding = hashlib.sha256(native).digest()
        self.native = flat_audio_manifest(json.loads(native))
        self.audio_folder = asset.parent / "audio"
        self.archive = manifest.parent / doc["archive"]
        self.events = None
        groups = defaultdict(list)
        for row in doc["records"]:
            if str(row.get("song", "")) == "854" or "全世界共通リズム感テスト" in row["meta"].get("title", ""):
                continue
            groups[row["audio_id"]].append(row)
        self.groups = list(groups.items())

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, item):
        epoch, index = item
        key, rows = self.groups[index]
        audio = read_flat_audio(self.audio_folder / key, self.native["audio"][key])
        if self.events is None:
            self.events = np.load(self.archive, mmap_mode="r")
        charts = []
        for row in rows:
            lo, hi = row["events"]
            events = self.events[lo:hi].tolist()
            grid = BeatGrid(row["meta"])
            audio_end = math.floor(grid.second_to_beat(audio["sample_lengths"] / 44100) * 96)
            # A final roll may release after the audio ends.
            last_onset, _, last_release, _ = events[-1] if events else (0, 0, 0, 0)
            end = max(audio_end, last_onset, last_release)
            arrays = encode_taiko(events, end, end)
            rng = np.random.default_rng([self.seed, epoch, index, row["meta"]["difficulty"]])
            charts.append(prepare_chart(arrays, row["meta"], end, self.config,
                                        audio["mel_lengths"], sample_history(rng), grid=grid, plan_rng=rng, for_batch=True))
        return audio, charts


class MixedChartGroups(Dataset):
    def __init__(self, maimai, taiko):
        self.maimai, self.taiko = maimai, taiko

    def __len__(self):
        return len(self.maimai) + len(self.taiko)

    def __getitem__(self, index):
        epoch, task, song = index
        return (self.taiko if task else self.maimai)[epoch, song]


class MixedEpochSampler:
    """A full maimai permutation plus a contiguous slice of the persistent Taiko deck."""
    def __init__(self, n_maimai, n_taiko, seed, batch_size=4):
        if batch_size < 2 or batch_size % 2 or n_taiko < n_maimai:
            raise ValueError("1:1 no-replacement batches require an even batch and at least as many Taiko groups")
        self.n_maimai, self.n_taiko = n_maimai, n_taiko
        self.half, self.seed, self.epoch = batch_size // 2, seed, 1

    def __len__(self):
        return math.ceil(self.n_maimai / self.half)

    def cursor(self, epochs):
        return divmod(epochs * self.n_maimai, self.n_taiko)

    def deck(self, cycle):
        # A new deck can start mid-epoch. Keep that epoch's old tail out of its
        # new prefix, without losing those cards from the new cycle.
        previous = None
        for c in range(cycle + 1):
            deck = np.random.default_rng([self.seed, 0x5441494b, c]).permutation(self.n_taiko)
            used = c * self.n_taiko % self.n_maimai
            if used:
                repeat = np.isin(deck, previous[-used:])
                deck = np.concatenate((deck[~repeat], deck[repeat]))
            previous = deck
        return deck

    def __iter__(self):
        mai = np.random.default_rng(self.seed + self.epoch).permutation(self.n_maimai)
        cycle, offset = self.cursor(self.epoch - 1)
        taiko = self.deck(cycle)[offset:offset + self.n_maimai]
        remaining = self.n_maimai - len(taiko)
        if remaining:
            taiko = np.concatenate((taiko, self.deck(cycle + 1)[:remaining]))
        for start in range(0, len(mai), self.half):
            section = slice(start, start + self.half)
            yield [(self.epoch, task, int(i)) for task, group in enumerate((mai, taiko)) for i in group[section]]


def beat_grid(grid, field_limit, length):
    n = max(1, math.ceil(field_limit / 96) + 2) * GRID_STEPS_PER_BEAT
    seconds = grid.beat_to_second_tensor(torch.arange(n) / GRID_STEPS_PER_BEAT)
    return seconds, (seconds >= 0) & (seconds <= length / 125)


def valid_grid_bounds(in_audio):
    indices = np.flatnonzero(in_audio.numpy())
    return (int(indices[0]), int(indices[-1]) + 1) if len(indices) else (0, 0)


def grid_interval(cfg, stream, tick, page_first, valid_bounds):
    """Half-open real-key bounds, with index zero reserved for the sink."""
    page = (page_first >= 0) & (stream == "mert")
    center = np.where(page, (page_first + 8) * GRID_STEPS_PER_BEAT, tick // 4)
    left = round(getattr(cfg, stream + "_past_beats") * GRID_STEPS_PER_BEAT)
    right = round(getattr(cfg, stream + "_future_beats") * GRID_STEPS_PER_BEAT)
    return 1 + np.clip(center - left, *valid_bounds), 1 + np.clip(
        center + right + (1 - page), *valid_bounds
    )


def soft_targets(rows, keys, log_kernel, width):
    return dict(
        rows=torch.as_tensor(rows, dtype=torch.long),
        keys=torch.as_tensor(np.asarray(keys, dtype=np.int64).reshape(-1, width)),
        log_kernel=torch.as_tensor(
            np.asarray(log_kernel, dtype=np.float32).reshape(-1, width)
        ),
    )


def audio_targets(targets, bounds, sigma, tick_bounds):
    pairs = np.asarray(targets, np.int64).reshape(-1, 2)
    rows, ticks = pairs.T
    centers = 1 + np.rint(ticks / 4).astype(np.int64)
    lo, hi = bounds[rows].T
    valid = (
        (lo <= centers)
        & (centers < hi)
        & (tick_bounds[0] <= ticks)
        & (ticks <= tick_bounds[1])
    )
    delta = np.arange(-2, 3)
    keys = centers[valid, None] + delta
    inside = (lo[valid, None] <= keys) & (keys < hi[valid, None])
    return soft_targets(
        rows[valid],
        np.where(inside, keys, 0),
        np.where(inside, -(delta**2) / (2 * sigma**2), -np.inf),
        5,
    )


# Zone -1 denotes an unknown query position and contributes zero spatial cost.
NEAR_ZONE_COST = np.pad(DISK_DISTANCE_SQ / (2 * 0.65**2), ((0, 1), (0, 0)))


def cluster_targets(occupancy):
    occ_rows = np.fromiter(
        (row for row, _ in occupancy), np.int64, count=len(occupancy)
    )
    palms = [clusters for _, clusters in occupancy]
    counts = np.fromiter(map(len, palms), np.intp, count=len(palms))
    clusters = list(chain.from_iterable(palms))
    sizes = np.fromiter(map(len, clusters), np.intp, count=len(clusters))
    return dict(
        rows=torch.from_numpy(np.repeat(occ_rows, counts)),
        keys=torch.from_numpy(np.fromiter(chain.from_iterable(clusters), np.int64)),
        cluster=torch.from_numpy(
            np.repeat(np.arange(len(clusters), dtype=np.int64), sizes)
        ),
        scale=torch.from_numpy(np.repeat((1.0 / counts).astype(np.float32), counts)),
        log_count=torch.from_numpy(
            np.repeat(np.log(counts).astype(np.float32), counts)
        ),
        support=len(occ_rows),
    )


def near_targets(arrays, timing, history):
    ticks, zones, masks = arrays["rows"][:, 0], arrays["zones"], arrays["zone_masks"]
    real_mask = arrays.get("real_mask")
    rows = np.flatnonzero(~timing if real_mask is None else (~timing & real_mask))
    committed = np.flatnonzero(~arrays["evict"] & ~timing & (masks != 0))
    cticks = ticks[committed]
    hi = np.searchsorted(committed, rows)
    lo = hi
    if len(committed):
        # Find the starts of the latest and penultimate distinct ticks, then clip to the window.
        latest = np.searchsorted(cticks, cticks[(hi - 1).clip(min=0)])
        previous = np.searchsorted(cticks, cticks[(latest - 1).clip(min=0)])
        cutoff = np.maximum(ticks[rows] - 192, (ticks[rows] // 96 - history) * 96)
        lo = np.minimum(hi, np.maximum(previous, np.searchsorted(cticks, cutoff)))

    # One edge per visible slot, or one BOUNDARY edge for an empty row.
    counts = np.maximum(hi - lo, 1)
    starts = np.cumsum(counts) - counts
    owners = np.repeat(np.arange(len(rows)), counts)
    edge_rows = rows[owners]
    real = hi[owners] > lo[owners]
    positions = lo[owners] + np.arange(len(owners)) - starts[owners]
    keys = np.zeros(len(owners), np.int64)
    keys[real] = committed[positions[real]]
    spatial = NEAR_ZONE_COST[zones[edge_rows], zones[keys]]
    key_masks = masks[keys]
    multi = (key_masks & (key_masks - 1)) != 0
    if multi.any():
        active = ((key_masks[multi, None] >> np.arange(33)) & 1).astype(bool)
        distances = NEAR_ZONE_COST[zones[edge_rows[multi]]]
        spatial[multi] = np.where(active, distances, np.inf).min(1)
    cost = (ticks[edge_rows] - ticks[keys]) / 96.0 + spatial
    cost[~real] = 0
    weights = np.exp(-cost)
    weights /= np.add.reduceat(weights, starts)[owners]
    return dict(
        rows=torch.from_numpy(edge_rows),
        keys=torch.from_numpy(keys),
        weights=torch.from_numpy(weights.astype(np.float32)),
        log_weights=torch.from_numpy(np.log(weights).astype(np.float32)),
        support=len(rows),
    )


def self_targets(arrays, timing, history):
    return dict(
        occ=cluster_targets(arrays["occupancy"]),
        near=near_targets(arrays, timing, history),
        slide=cluster_targets(arrays["slide_occupancy"]),
    )


def prepare_chart(
    arrays,
    meta,
    end,
    cfg,
    length,
    history,
    *,
    grid=None,
    prefill=False,
    audio_grid=None,
    plan_rng=None,
    for_batch=False,
):
    from .layouts import layouts

    grid = grid or BeatGrid(meta)
    ts = arrays["rows"]
    ticks = ts[:, 0]
    seconds, in_audio = (
        audio_grid
        if audio_grid is not None
        else beat_grid(grid, arrays["field_limit"], length)
    )
    valid_bounds = valid_grid_bounds(in_audio)
    bounds = {
        stream: np.stack(
            grid_interval(cfg, stream, ts[:, 1], arrays["pages"], valid_bounds), axis=-1
        )
        for stream in STREAMS
    }
    unique, inv = np.unique(ticks, return_inverse=True)
    clock = clocks(grid, torch.from_numpy(unique), end)[:, :26][torch.from_numpy(inv)]
    seq_tokens = arrays["tokens"].astype(np.int64)
    in_tokens = (
        arrays["input_tokens"].astype(np.int64)
        if "input_tokens" in arrays
        else seq_tokens[:-1]
    )
    tgt_tokens = (
        arrays["target_tokens"].astype(np.int64)
        if "target_tokens" in arrays
        else seq_tokens[1:]
    )
    taiko = "taiko_events" in arrays
    timing = arrays["timing"] if taiko else np.isin(in_tokens, list(TIMING_INPUTS))
    packed_layouts = layouts(
        ts,
        in_tokens,
        timing,
        bounds,
        history,
        arrays["protected"],
        arrays["evict"],
        cfg.stage,
        cfg.history_query_block,
        auxiliary=not prefill and cfg.no_history_enabled,
        active_spans=arrays["active_spans"],
    )
    chart = dict(
        heads={k: torch.from_numpy(v) for k, v in arrays["heads"].items()},
        tokens=torch.from_numpy(in_tokens),
        targets=torch.from_numpy(tgt_tokens),
        active_state=torch.from_numpy(arrays["active_state"]),
        coords=torch.from_numpy(ts.astype(np.int64)),
        clock=clock,
        timing=torch.from_numpy(timing),
        condition=condition(meta, cfg.designer_vocab),
        layouts=packed_layouts,
        moments=int(
            (
                (seq_tokens >= FAMILY_OFFSET["MOMENT"])
                & (seq_tokens < FAMILY_OFFSET["MOMENT"] + WIDTHS["MOMENT"])
            ).sum()
        ),
    )
    if cfg.architecture == ARCHITECTURE_V20:
        chart["task_id"] = int(taiko)
        if not taiko:
            chart["heads"].update(rows5=torch.empty(0, dtype=torch.long), args5=torch.empty((0, 5), dtype=torch.int32))
    if cfg.architecture in DRAFT_ARCHITECTURES:
        beat_valid = in_audio.numpy().reshape(-1, 24).any(1)
        chart["anchor_valid"] = draft_anchor_valid(torch.from_numpy(beat_valid))
        pages = arrays["pages"].astype(np.int64, copy=False)
        center_ticks = np.where(pages >= 0, (pages + 8) * 96, ts[:, 1])
        chart["cond_anchors"] = torch.from_numpy(((center_ticks + 96) // 192).astype(np.int64))
        if not for_batch:
            chart["cond_buckets"] = condition_buckets([chart["cond_anchors"]])
        if not prefill:
            chart["draft_targets"] = (taiko_draft_targets(arrays["taiko_events"], beat_valid) if taiko
                                      else build_draft_targets(arrays["chart_moments"], beat_valid))
    if "real_mask" in arrays:
        chart["real_mask"] = torch.from_numpy(arrays["real_mask"])
    # Send only rows where the gate actually decides; no GPU nonzero or full-vocab unpack.
    timing_rows = np.flatnonzero(timing)
    if prefill:
        timing_rows = timing_rows[timing_rows == len(ticks) - 1]
    if "last_legal" in arrays:
        packed = arrays["last_legal"][None, :][: len(timing_rows)]
    else:
        packed = arrays["legal"][timing_rows]
    start = FAMILY_OFFSET["MOMENT"]
    allowed = np.unpackbits(packed[:, start // 8 : (start + 32 + 7) // 8], axis=1)[
        :, start % 8 : start % 8 + 32
    ].astype(bool)
    active = allowed.any(-1)
    emit_rows = timing_rows[active]
    chart.update(
        grid_seconds=seconds,
        in_audio=in_audio,
        emit_rows=torch.from_numpy(emit_rows),
        moment_legal=torch.from_numpy(allowed[active]),
        emit_beats=torch.from_numpy(
            (ticks[emit_rows] // 96).clip(0, len(seconds) // 24 - 1).astype(np.int64)
        ),
    )
    if cfg.architecture == ARCHITECTURE_V20 and not for_batch:
        task = int(taiko)
        chart.update(
            task_ids=torch.tensor([task]),
            mai_rows=0 if taiko else len(ticks),
            mai_special=0 if taiko else len(packed_layouts["special"]),
            mai_anchors=0 if taiko else int(chart["anchor_valid"].sum()),
            emit_tasks=torch.full((len(emit_rows),), task, dtype=torch.long),
            emit_terminal=torch.from_numpy((packed[active, 0] & 128) != 0),
        )
    if cfg.inline_plan:
        anchors = np.flatnonzero(
            np.isin(in_tokens, np.arange(0, 16, 4) + FAMILY_OFFSET["BEAT"])
        )
        chart["plan4_rows"] = torch.from_numpy(anchors)
        chart["anchor_bars"] = torch.from_numpy(ticks[anchors].astype(np.int64) // 384)
        chart["plan_emit_beats"] = chart["emit_beats"]
    if cfg.plan4_enabled and not prefill:
        te = compute_taiko_beat_te_v3(arrays["taiko_events"], grid, seconds, in_audio) if taiko else compute_beat_te_v3(
            arrays["chart_moments"],
            grid,
            seconds,
            in_audio,
            cfg.emotion_slide_boost,
        )
        chart["gt_beat_te"] = torch.from_numpy(te)
        generator = (
            None
            if plan_rng is None
            else torch.Generator().manual_seed(int(plan_rng.integers(2**63 - 1)))
        )
        chart["plan4"] = quantize_plan4(te, training=generator is not None, generator=generator)
    if not prefill:
        lh_rows, labels = arrays["slide_lookahead"]
        chart["slide_lookahead"] = dict(
            rows=torch.from_numpy(lh_rows),
            targets=torch.from_numpy(labels),
            support=len(lh_rows),
        )
        chart["union"] = union_indices(arrays["union_pairs"])
        chart["weights"] = torch.from_numpy(arrays["weights"])
        guide = chart["guide"] = {}
        if cfg.attn_guide_weight and any(cfg.attn_guide_layers[: cfg.layers]):
            # A GRID32 midpoint may round onto a valid key from just outside the song.
            first, last = 4 * valid_bounds[0], 4 * (valid_bounds[1] - 1)
            if grid.second(first - 2) >= 0:
                first -= 2
            if grid.second(last + 2) <= length / 125:
                last += 2
            for stream, sigma in zip(STREAMS, (1.5, 4.0)):
                guide[stream] = audio_targets(
                    arrays[stream + "_ticks"], bounds[stream], sigma, (first, last)
                )
            if cfg.stage == 1:
                if taiko:
                    guide.update(occ=cluster_targets([]), slide=cluster_targets([]),
                                 near=dict(rows=torch.empty(0, dtype=torch.long), keys=torch.empty(0, dtype=torch.long),
                                           weights=torch.empty(0), log_weights=torch.empty(0), support=0))
                else:
                    guide.update(self_targets(arrays, timing, history))
        chart["legal"] = torch.from_numpy(arrays["legal"])
        if cfg.architecture == ARCHITECTURE_V20 and not taiko:
            chart["legal"] = torch.nn.functional.pad(chart["legal"], (0, (SIZE_V20 + 7) // 8 - chart["legal"].size(1)))
        if "physical" in arrays:
            chart["physical"] = arrays["physical"]
    return chart


def aligned_rows(lengths, block):
    starts, valid, total = [], [], 0
    for n in lengths:
        starts.append(total)
        valid.append(torch.arange(total, total + n))
        total += ((n + block - 1) // block) * block
    return starts, torch.cat(valid), total


def pack_audio_grid(charts, owners, block):
    ids, grids, mappings = {}, [], []
    for chart, song in zip(charts, owners):
        seconds = chart["grid_seconds"]
        key = (song, seconds.numpy().tobytes())
        if key not in ids:
            ids[key] = len(grids)
            grids.append((song, seconds, chart["in_audio"]))
        mappings.append(ids[key])
    lengths = [1 + len(seconds) for _, seconds, _ in grids]
    starts, rows, size = aligned_rows(lengths, block)
    compact = np.cumsum([0, *lengths[:-1]])
    beat_offsets = np.cumsum([0, *((n - 1) // 24 for n in lengths[:-1])])
    # The v18.7 planner indexes Python lists with its host-side mapping.
    grid_ids = torch.tensor(mappings, dtype=torch.long) if "anchor_valid" in charts[0] else mappings
    return (
        dict(grids=grids, grid_rows=rows, chart_grid_ids=grid_ids),
        ([starts[g] for g in mappings], size),
        [compact[g] for g in mappings],
        [beat_offsets[g] for g in mappings],
    )


def pack_charts(charts, owners):
    from .layouts import merge_layout

    multitask = "task_id" in charts[0]
    if multitask:
        order = sorted(range(len(charts)), key=lambda i: charts[i]["task_id"])
        owners = [owners[i] for i in order]
        charts = [charts[i] for i in order]
    block = charts[0]["layouts"]["main"]["block"]
    sizes = [len(c["tokens"]) for c in charts]
    starts, token_rows, token_size = aligned_rows(sizes, block)
    specials = [c["layouts"]["special"] for c in charts]
    _, read_rows, _ = aligned_rows([len(v) for v in specials], block)
    audio, space, kv_offsets, beat_offsets = pack_audio_grid(charts, owners, block)
    compact_keys = dict.fromkeys(STREAMS, kv_offsets)
    key_spaces = {
        prefix + stream: space
        for stream in STREAMS
        for prefix in ("audio_", "read_audio_")
    }
    if "anchor_valid" in charts[0]:
        valid_z = torch.nn.utils.rnn.pad_sequence([c["anchor_valid"] for c in charts], batch_first=True)
        audio["draft_all_valid"] = bool(valid_z.all())
        if not audio["draft_all_valid"]:
            audio["draft_valid_indices"] = torch.where(valid_z.flatten())[0]
        stride = valid_z.size(1)
        if multitask:
            ids = torch.tensor([c["task_id"] for c in charts])
            nmai = int((ids == 0).sum())
            audio.update(task_ids=ids, mai_charts=nmai, mai_rows=sum(sizes[:nmai]),
                         mai_anchors=int(valid_z[:nmai].sum()), mai_special=sum(len(v) for v in specials[:nmai]),
                         emit_tasks=torch.cat([ids[i].expand(len(c["emit_rows"])) for i, c in enumerate(charts)]),
                         emit_terminal=torch.cat([c["legal"][c["emit_rows"], 0].bitwise_and(128).bool() for c in charts]))
            targets = []
            for task, (offset, group) in enumerate(((0, charts[:nmai]), (nmai, charts[nmai:]))):
                if group:
                    part = pack_targets([c["draft_targets"] for c in group], stride)
                    part["query"][:, 0] += offset * stride
                    part["span"][:, 0] += offset * stride
                    targets.append(dict(part, task_id=task))
            audio["draft_targets"] = targets
        else:
            audio["draft_targets"] = pack_targets([c["draft_targets"] for c in charts], stride)
        audio["cond_buckets"] = condition_buckets([c["cond_anchors"] for c in charts])
    packed_layouts = {}
    for name in charts[0]["layouts"]:
        if name not in ("special", "active_fields"):
            offsets, size = key_spaces.get(name, (starts, token_size))
            packed_layouts[name] = merge_layout(
                [c["layouts"][name] for c in charts], offsets, size
            )
    cumulative = np.cumsum([0, *sizes[:-1]])
    packed_layouts["special"] = torch.cat(
        [s + int(n) for s, n in zip(specials, cumulative)]
    )
    if "active_fields" in charts[0]["layouts"]:
        packed_layouts["active_fields"] = torch.cat(
            [
                c["layouts"]["active_fields"] + int(off)
                for c, off in zip(charts, cumulative)
            ]
        )
    chart_ids = torch.arange(len(charts)).repeat_interleave(torch.tensor(sizes))
    values, designers = zip(*(c["condition"] for c in charts))
    legal = torch.zeros((token_size, charts[0]["legal"].shape[1]), dtype=torch.uint8)
    legal.index_copy_(0, token_rows, torch.cat([c["legal"] for c in charts]))
    guide = {}
    for name in charts[0]["guide"]:
        parts = [c["guide"][name] for c in charts]
        cluster_offsets = np.cumsum([0, *(len(t["rows"]) for t in parts[:-1])])
        offsets = dict(
            rows=cumulative,
            keys=compact_keys.get(name, cumulative),
            cluster=cluster_offsets,
        )
        merged = {}
        for key in parts[0]:
            items = [t[key] for t in parts]
            if key == "support":
                merged[key] = sum(items)
            else:
                if key in offsets:
                    items = [item + int(off) for item, off in zip(items, offsets[key])]
                merged[key] = torch.cat(items)
        guide[name] = merged
    unions = [c["union"] for c in charts]
    union_offsets = np.cumsum([0, *(len(u["rows"]) for u in unions[:-1])])
    union = dict(
        rows=torch.cat([u["rows"] + int(off) for u, off in zip(unions, cumulative)]),
        inverse=torch.cat(
            [u["inverse"] + int(off) for u, off in zip(unions, union_offsets)]
        ),
        cols=torch.cat([u["cols"] for u in unions]),
    )
    plan = {}
    if "plan4" in charts[0]:
        bars = [len(c["plan4"]) for c in charts]
        offsets = np.cumsum([0, *bars[:-1]])
        plan = dict(
            plan4=torch.cat([c["plan4"] for c in charts]),
            plan_chart_ids=torch.arange(len(charts)).repeat_interleave(
                torch.tensor(bars)
            ),
            gt_beat_te=torch.nn.utils.rnn.pad_sequence(
                [c["gt_beat_te"] for c in charts], batch_first=True
            ),
        )
        if "plan4_rows" in charts[0]:
            plan.update(
                plan4_rows=torch.cat(
                    [c["plan4_rows"] + int(off) for c, off in zip(charts, cumulative)]
                ),
                anchor_bars=torch.cat(
                    [c["anchor_bars"] + int(off) for c, off in zip(charts, offsets)]
                ),
                plan_emit_beats=torch.cat(
                    [c["plan_emit_beats"] + 4 * int(off) for c, off in zip(charts, offsets)]
                ),
            )
        else:
            plan["plan_lengths"] = bars
    phys_pack = {}
    if any("physical" in c for c in charts):
        from .physical import pack_physical_batches

        phys_pack = dict(
            physical=pack_physical_batches([c for c in charts if "physical" in c],
                                           [off for c, off in zip(charts, cumulative) if "physical" in c]),
            real_mask=torch.cat(
                [
                    c.get("real_mask", torch.ones(len(c["tokens"]), dtype=torch.bool))
                    for c in charts
                ]
            ),
        )
    return dict(
        **plan,
        **phys_pack,
        legal=legal,
        emit_rows=torch.cat(
            [c["emit_rows"] + off for c, off in zip(charts, cumulative)]
        ),
        emit_beats=torch.cat(
            [c["emit_beats"] + off for c, off in zip(charts, beat_offsets)]
        ),
        moment_legal=torch.cat([c["moment_legal"] for c in charts]),
        slide_lookahead=dict(
            rows=torch.cat(
                [
                    c["slide_lookahead"]["rows"] + int(n)
                    for c, n in zip(charts, cumulative)
                ]
            ),
            targets=torch.cat([c["slide_lookahead"]["targets"] for c in charts]),
            support=sum(c["slide_lookahead"]["support"] for c in charts),
        ),
        union=union,
        guide=guide,
        **{
            k: torch.cat([c[k] for c in charts])
            for k in (
                "tokens",
                "targets",
                "weights",
                "coords",
                "timing",
                "active_state",
                "clock",
            )
        },
        heads=pack_heads(charts, cumulative),
        layouts=packed_layouts,
        token_rows=token_rows,
        read_rows=read_rows,
        **audio,
        condition=(torch.tensor(values), torch.tensor(designers)),
        chart_ids=chart_ids,
        moments=sum(c["moments"] for c in charts),
        chart_count=len(charts),
    )


@dataclass
class CompactBatch:
    buffers: dict
    views: list
    tree: object

    def pin_memory(self):
        return CompactBatch(
            {k: t.pin_memory() for k, t in self.buffers.items()}, self.views, self.tree
        )

    def to(self, device):
        buffers = {k: t.to(device, non_blocking=True) for k, t in self.buffers.items()}
        return expand_batch(buffers, self.views, self.tree)


def compact_batch(batch):
    """Keep mmap-to-IPC audio storages intact; coalesce small metadata by dtype."""
    from torch.utils._pytree import tree_flatten

    audio, charts = batch
    audio = dict(audio)
    large = {k: audio.pop(k) for k in FLAT_RATES}
    leaves, tree = tree_flatten((audio, charts))
    buffers = defaultdict(list)
    offsets = defaultdict(int)
    views = []
    for leaf in leaves:
        if isinstance(leaf, torch.Tensor):
            dtype = leaf.dtype
            views.append((dtype, offsets[dtype], leaf.numel(), leaf.shape))
            buffers[dtype].append(leaf.reshape(-1))
            offsets[dtype] += leaf.numel()
        else:
            views.append(leaf)
    return CompactBatch(
        {**large, **{dtype: torch.cat(parts) for dtype, parts in buffers.items()}},
        views,
        tree,
    )


def expand_batch(buffers, views, tree):
    from torch.utils._pytree import tree_unflatten

    def unpack(view):
        if not isinstance(view, tuple):
            return view
        dtype, start, size, shape = view
        return buffers[dtype].narrow(0, start, size).view(shape)

    audio, charts = tree_unflatten([unpack(v) for v in views], tree)
    audio.update((k, buffers[k]) for k in FLAT_RATES)
    return audio, charts


def collate(groups, *, config=None):
    audio, grouped = zip(*groups)
    charts = pack_charts(
        [c for charts in grouped for c in charts],
        [i for i, charts in enumerate(grouped) for _ in charts],
    )
    return compact_batch((pad_flat_audio(audio), charts))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("manifest", "audio-manifest", "output"):
        p.add_argument("--" + key, required=True, type=Path)
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args(argv)
    bake(a.manifest, a.audio_manifest, a.output, a.workers)


if __name__ == "__main__":
    main()
