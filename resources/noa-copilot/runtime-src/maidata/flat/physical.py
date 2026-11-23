"""Compact offline Physical sidecar packing, cross-epoch alignment, and GPU loss."""

import argparse
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from copy import copy
import gzip
import hashlib
from itertools import groupby
import json
import math
import os
from pathlib import Path
import tempfile
from zlib import crc32

import numpy as np
import torch

from maidata.spine import official
from maidata.spine.codec import HOLD, PAD_MAP, PLAIN, SLIDE, mirror_notes
from maidata.spine.packing import beat_actions

from . import PACK_SCHEMA
from .codec import (
    BOUNDARY,
    CHORD,
    EOM,
    GROUP,
    NEW_PADS,
    SIZE,
    TICK_TO_GRID32,
    branch_key,
    subgroup_items,
    timestamp,
    token,
)

PHYSICAL_SIDECAR_SCHEMA = "flat320.prefix-physical.prototype.v2"
PHYSICAL_PACK_SCHEMA = "maimai-flat320-physical.packed.v2"
LEGACY_PACK_SCHEMA = "maimai-flat320-physical.packed.v1"

ANCHOR_TIME = 0
ANCHOR_EOM = 1
ANCHOR_SLOT = 2
ANCHOR_CLOSE = 3
ANCHOR_ROUTE = 4
ANCHOR_ENDPOINT = 5

RULE_TO_CAT = {"multi": 0, "overlap": 1, "inner": 2, "path": 3}
CAT_NAMES = ("multi", "overlap", "inner", "path")
CHANNELS = (
    "native_muri",
    "touch_repeat",
    "path_guard",
    "touch_hold_occupancy",
    "native_warning",
)
CHANNEL_BIT = {name: 1 << i for i, name in enumerate(CHANNELS)}
STRONG_MASK = (
    CHANNEL_BIT["native_muri"]
    | CHANNEL_BIT["touch_repeat"]
    | CHANNEL_BIT["path_guard"]
    | CHANNEL_BIT["touch_hold_occupancy"]
)
WARNING_MASK = CHANNEL_BIT["native_warning"]

FLAG_PLAIN_TOUCH_CLOSED = 1
FLAG_GT_STRONG = 1
FLAG_GT_WARN = 2


def _ordered_branches_with_index(
    branches, song, tick, group, pad, mirrored=False, *, seed=None, epoch=0
):
    if len(branches) <= 1:
        return [(0, b) for b in branches]
    base = (
        mirror_notes([(pad, group, SLIDE, 0, 0, 0, branches)])[0][6]
        if mirrored
        else branches
    )
    order = sorted(range(len(branches)), key=lambda i: branch_key(base[i]))
    if seed is not None:
        song_str = str(song)
        song_seed = int(song_str) if song_str.isdecimal() else crc32(song_str.encode("utf8"))
        base_pad = PAD_MAP[8][pad] if mirrored else pad
        np.random.default_rng([seed, epoch, song_seed, tick, group, base_pad]).shuffle(
            order
        )
    return [(i, branches[i]) for i in order]


def trace_chart_anchors(moments, horizon=None, song=0, mirrored=False, *, seed=0, epoch=0):
    """Replay chart_tokens once while indexing structural anchors and committed sets."""
    horizon = max(horizon or 0, moments[-1][0] if moments else 0)
    tokens = [BOUNDARY]
    anchors = {}
    slot_meta = {}
    close_meta = {}
    route_meta = {}
    endpoint_meta = {}
    note_loc = {}

    beat = -1
    cur_tick = 0
    after_eom = 0
    note_base = 0

    for action, notes in beat_actions(moments, horizon):
        if not notes:
            if beat >= 0:
                anchors[(ANCHOR_TIME, cur_tick, after_eom, 0)] = len(tokens) - 1
            beat += 1
            tokens.append(token("BEAT", beat % 16))
            cur_tick = beat * 96
            after_eom = 0
            continue

        anchors[(ANCHOR_TIME, cur_tick, after_eom, 0)] = len(tokens) - 1
        tick = beat * 96 + action
        tokens.append(token("MOMENT", TICK_TO_GRID32[action]))

        for j, (group, pairs_iter) in enumerate(
            groupby(sorted(enumerate(notes), key=lambda p: p[1][1]), key=lambda p: p[1][1])
        ):
            if j:
                tokens.append(GROUP)
            owners = defaultdict(deque)
            members = []
            for orig_idx, n in pairs_iter:
                oid = note_base + orig_idx
                owners[id(n)].append(oid)
                note_loc[oid] = (tick, group)
                members.append(n)

            plain_touch_closed = False
            committed_group = []
            slot_idx = 0
            items = list(subgroup_items(members, song, tick, group, mirrored, seed, epoch))
            for item_idx, item in enumerate(items):
                if item_idx:
                    tokens.append(CHORD)
                for note in item:
                    oid = owners[id(note)].popleft()
                    pad, _, kind, flags, end, head, branches = note
                    headless = kind == SLIDE and head == 2
                    row_start = len(tokens) - 1
                    key_slot = (ANCHOR_SLOT, tick, group, slot_idx)
                    anchors[key_slot] = row_start
                    comm_frozen = frozenset(committed_group)
                    slot_meta[key_slot] = (
                        row_start,
                        oid,
                        headless,
                        plain_touch_closed,
                        comm_frozen,
                    )
                    slot_idx += 1

                    tokens.append(NEW_PADS[pad])
                    pad_row = len(tokens) - 1
                    mod_count = (
                        int(bool(flags & 1))
                        + int(bool(flags & 2))
                        + int(bool(flags & 4))
                    )
                    if flags & 1:
                        tokens.append(token("KEY_BREAK"))
                    if flags & 2:
                        tokens.append(token("KEY_EX"))
                    if flags & 4:
                        tokens.append(token("TOUCH_FIREWORK"))
                    for m in range(mod_count + 1):
                        key_close = (ANCHOR_CLOSE, oid, m, 0)
                        anchors[key_close] = pad_row + m
                        close_meta[key_close] = (pad_row + m, comm_frozen)

                    if kind == HOLD:
                        tokens.append(token("HOLD"))
                        for common, tok_ts in enumerate(timestamp("HOLD_END", tick, end)):
                            key_ep = (ANCHOR_ENDPOINT, oid, 0, common)
                            anchors[key_ep] = len(tokens) - 1
                            endpoint_meta[key_ep] = (len(tokens) - 1, comm_frozen, 0)
                            tokens.append(tok_ts)
                    elif kind == SLIDE:
                        tokens.append(token("SLIDE_HEAD", head))
                        ordered_br = _ordered_branches_with_index(
                            branches,
                            song,
                            tick,
                            group,
                            pad,
                            mirrored,
                            seed=seed,
                            epoch=epoch,
                        )
                        prior_bmask = 0
                        for b_pos, (orig_bidx, (launch, finish, brk, routes)) in enumerate(
                            ordered_br
                        ):
                            if b_pos:
                                tokens.append(token("SLIDE_BRANCH"))
                            for seg, rid in enumerate(routes):
                                key_rt = (ANCHOR_ROUTE, oid, orig_bidx, seg)
                                anchors[key_rt] = len(tokens) - 1
                                route_meta[key_rt] = (
                                    len(tokens) - 1,
                                    comm_frozen,
                                    prior_bmask,
                                )
                                tokens.append(token("ROUTE", rid))
                                if seg == 0 and brk:
                                    tokens.append(token("TRACK_BREAK"))
                            if launch != tick + 96:
                                tokens.append(token("SLIDE_WAIT"))
                                for common, tok_ts in enumerate(
                                    timestamp("SLIDE_WAIT", tick, launch)
                                ):
                                    key_w = (ANCHOR_ENDPOINT, oid, orig_bidx * 4 + 1, common)
                                    anchors[key_w] = len(tokens) - 1
                                    endpoint_meta[key_w] = (
                                        len(tokens) - 1,
                                        comm_frozen,
                                        prior_bmask,
                                    )
                                    tokens.append(tok_ts)
                            else:
                                tokens.append(token("SLIDE_END"))
                            for common, tok_ts in enumerate(
                                timestamp("SLIDE_END", tick, finish)
                            ):
                                key_e = (ANCHOR_ENDPOINT, oid, orig_bidx * 4 + 2, common)
                                anchors[key_e] = len(tokens) - 1
                                endpoint_meta[key_e] = (
                                    len(tokens) - 1,
                                    comm_frozen,
                                    prior_bmask,
                                )
                                tokens.append(tok_ts)
                            prior_bmask |= 1 << orig_bidx

                    if not headless:
                        committed_group.append(oid)
                if item[-1][0] >= 8 and item[-1][2] == PLAIN:
                    plain_touch_closed = True

        anchors[(ANCHOR_EOM, tick, 0, 0)] = len(tokens) - 1
        tokens.append(EOM)
        cur_tick = tick
        after_eom = 1
        note_base += len(notes)

    return dict(
        tokens=tokens,
        anchors=anchors,
        slot_meta=slot_meta,
        close_meta=close_meta,
        route_meta=route_meta,
        endpoint_meta=endpoint_meta,
        note_loc=note_loc,
    )


def _reverse_anchor_index(trace):
    by_kind_row = defaultdict(dict)
    for key, row in trace["anchors"].items():
        kind = key[0]
        if kind in (ANCHOR_TIME, ANCHOR_EOM, ANCHOR_SLOT):
            by_kind_row[(kind, -1)][row] = key
        else:
            by_kind_row[(kind, key[1])][row] = key
    return by_kind_row


def _resolve_canonical_key(family, row0, owner, rev):
    if family == "empty_moment":
        return rev[(ANCHOR_TIME, -1)][row0]
    if family == "extra_chord":
        return rev[(ANCHOR_EOM, -1)][row0]
    if family == "position":
        return rev[(ANCHOR_SLOT, -1)][row0]
    if family in ("ex_guard", "hold_open"):
        return rev[(ANCHOR_CLOSE, owner)][row0]
    if family.startswith("route"):
        return rev[(ANCHOR_ROUTE, owner)][row0]
    if family.endswith("_offset") or family.endswith("_offset_with_suffix"):
        return rev[(ANCHOR_ENDPOINT, owner)][row0]
    raise ValueError(f"Unsupported physical row family: {family}")


def compile_view_sidecar(doc, moments, end_tick, song, mirrored):
    """Compile one v2 sidecar dict into flat typed NumPy tables."""
    recipe = doc.get("recipe", {})
    if recipe.get("schema") != PHYSICAL_SIDECAR_SCHEMA:
        raise ValueError(
            f"Expected {PHYSICAL_SIDECAR_SCHEMA}, got {recipe.get('schema')}"
        )
    seed0 = int(recipe.get("seed", 0))
    epoch0 = int(recipe.get("epoch", 0))
    trace = trace_chart_anchors(
        moments, end_tick, song, mirrored, seed=seed0, epoch=epoch0
    )
    if len(trace["tokens"]) != int(doc["tokens"]):
        raise ValueError(
            f"Token count mismatch for {doc.get('id')} mirror={mirrored}: "
            f"expected {len(trace['tokens'])}, sidecar has {doc['tokens']}"
        )
    rev = _reverse_anchor_index(trace)
    note_loc = trace["note_loc"]

    row_meta = []
    fork_tokens = []
    row_neg_bounds = []
    ev_col, ev_cat, ev_ch, ev_share = [], [], [], []
    ev_role_bounds, role_ids = [], []

    for r in doc["rows"]:
        fam = r["family"]
        row0 = int(r["row"])
        owner = int(r["owner"])
        fork = r.get("fork") or ()
        key = _resolve_canonical_key(fam, row0, owner, rev)
        kind, a0, a1, a2 = key

        flags = 0
        prior_bmask = 0
        loc = (int(r["tick"]), int(r["group"]))
        if kind == ANCHOR_SLOT:
            _, _, _, ptc0, _ = trace["slot_meta"][key]
            if ptc0:
                flags |= FLAG_PLAIN_TOUCH_CLOSED
        elif kind == ANCHOR_ROUTE:
            _, _, prior_bmask = trace["route_meta"][key]
        elif kind == ANCHOR_ENDPOINT:
            _, _, prior_bmask = trace["endpoint_meta"][key]

        suffix_set = set(r.get("suffix") or ())
        real_owner = -1 if fam in ("extra_chord", "empty_moment") else owner

        neg_lo = len(ev_col)
        for col, _, evidence in r["negatives"]:
            col_int = int(col)
            for ev in evidence:
                rule = ev["rule"]
                if rule not in RULE_TO_CAT:
                    continue
                mask = 0
                for ch in ev.get("channels", ()):
                    mask |= CHANNEL_BIT.get(ch, 0)
                if not mask:
                    continue
                r_lo = len(role_ids)
                if kind not in (ANCHOR_TIME, ANCHOR_EOM):
                    for rid in ev.get("roles", ()):
                        rid = int(rid)
                        if (
                            rid != real_owner
                            and rid not in suffix_set
                            and note_loc.get(rid) == loc
                        ):
                            role_ids.append(rid)
                r_hi = len(role_ids)
                ev_col.append(col_int)
                ev_cat.append(RULE_TO_CAT[rule])
                ev_ch.append(mask)
                ev_share.append(float(ev.get("share", 1.0)))
                ev_role_bounds.append((r_lo, r_hi))
        neg_hi = len(ev_col)
        if neg_hi == neg_lo:
            continue

        fork_lo = len(fork_tokens)
        fork_tokens.extend(fork)
        row_meta.append(
            (
                kind,
                a0,
                a1,
                a2,
                real_owner,
                flags,
                prior_bmask,
                fork_lo,
                len(fork_tokens),
            )
        )
        row_neg_bounds.append((neg_lo, neg_hi))

    gt_meta = []
    for g in doc.get("gt_rows", ()):
        fam = g["family"]
        row0 = int(g["row"])
        owner = int(g["owner"])
        key = _resolve_canonical_key(fam, row0, owner, rev)
        kind, a0, a1, a2 = key
        sev = 0
        for ev in g.get("events", ()):
            if ev.get("muri") or any(
                c != "native_warning" for c in ev.get("channels", ())
            ):
                sev |= FLAG_GT_STRONG
            elif "native_warning" in ev.get("channels", ()):
                sev |= FLAG_GT_WARN
        if sev:
            gt_meta.append((kind, a0, a1, a2, int(g["target"]), sev))

    return dict(
        support=len(doc.get("queries", ())),
        row_meta=np.asarray(row_meta, np.int32).reshape(-1, 9),
        fork_tokens=np.asarray(fork_tokens, np.uint16),
        row_neg_bounds=np.asarray(row_neg_bounds, np.int32).reshape(-1, 2),
        ev_col=np.asarray(ev_col, np.uint16),
        ev_cat=np.asarray(ev_cat, np.uint8),
        ev_ch=np.asarray(ev_ch, np.uint8),
        ev_share=np.asarray(ev_share, np.float32),
        ev_role_bounds=np.asarray(ev_role_bounds, np.int32).reshape(-1, 2),
        role_ids=np.asarray(role_ids, np.int32),
        gt_meta=np.asarray(gt_meta, np.int32).reshape(-1, 6),
    )


def _compile_job(job):
    from .data import unpack_events

    idx, sidecar_path, words, end_tick, song, flip = job
    with gzip.open(sidecar_path, "rt", encoding="utf8") as stream:
        doc = json.load(stream)
    return idx, compile_view_sidecar(
        doc, unpack_events(words), end_tick, song, flip
    )


def _discover_sidecar_files(cache_dir):
    bench = cache_dir / "benchmark.json"
    mapping = {}
    if bench.is_file():
        info = json.loads(bench.read_text("utf8"))
        for r in info.get("records") or info.get("results") or ():
            path = cache_dir / r["file"]
            if path.is_file():
                mapping[(r["id"], int(bool(r["mirror"])))] = path
    if not mapping:
        for path in sorted(cache_dir.glob("*.json.gz")):
            with gzip.open(path, "rt", encoding="utf8") as stream:
                doc = json.load(stream)
            if doc.get("recipe", {}).get("schema") == PHYSICAL_SIDECAR_SCHEMA:
                mapping[(doc["id"], int(bool(doc["mirror"])))] = path
    return mapping


def pack_physical_cache(cache_dir, chart_manifest, output_dir=None, *, workers=12):
    """Pack v2 .json.gz sidecars into one content-addressed .npz + manifest."""
    cache_dir = Path(cache_dir)
    chart_manifest = Path(chart_manifest)
    output_dir = Path(output_dir) if output_dir is not None else cache_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = json.loads(chart_manifest.read_text("utf8"))
    if doc.get("schema") != PACK_SCHEMA:
        raise ValueError(f"Expected {PACK_SCHEMA} chart manifest")
    with np.load(chart_manifest.parent / doc["archive"]) as pack:
        events = pack["events"]

    sidecars = _discover_sidecar_files(cache_dir)
    jobs = []
    view_keys = []
    for row in doc["records"]:
        if not (official(row) and row.get("split") == "fit"):
            continue
        cid = row["id"]
        song = int(row["song"])
        for flip in (0, 1):
            path = sidecars.get((cid, flip))
            if path is None:
                continue
            lo, hi = row["views"][flip]
            idx = len(jobs)
            view_keys.append((cid, flip, path.name))
            jobs.append(
                (idx, path, events[lo:hi].copy(), row["end_tick"], song, bool(flip))
            )

    if not jobs:
        raise ValueError(f"No matching v2 physical sidecars found in {cache_dir}")

    pool_cls = (
        ThreadPoolExecutor
        if os.name == "nt" and len(jobs) < 64
        else ProcessPoolExecutor
    )
    compiled = [None] * len(jobs)
    max_w = max(1, min(int(workers), len(jobs)))
    if max_w == 1:
        for job in jobs:
            i, res = _compile_job(job)
            compiled[i] = res
    else:
        with pool_cls(max_workers=max_w) as pool:
            for i, res in pool.map(_compile_job, jobs):
                compiled[i] = res

    view_bounds = []
    row_meta_parts, row_neg_parts, fork_parts = [], [], []
    ev_col_parts, ev_cat_parts, ev_ch_parts, ev_share_parts = [], [], [], []
    ev_role_parts, role_parts, gt_parts = [], [], []
    r_off = e_off = role_off = gt_off = fork_off = 0

    records_index = {}
    for (cid, flip, fname), item in zip(view_keys, compiled):
        nr = len(item["row_meta"])
        ne = len(item["ev_col"])
        nrole = len(item["role_ids"])
        ngt = len(item["gt_meta"])
        v_idx = len(view_bounds)
        records_index.setdefault(cid, {})[str(flip)] = dict(
            index=v_idx, file=fname, support=int(item["support"]), rows=nr
        )
        view_bounds.append(
            (r_off, r_off + nr, gt_off, gt_off + ngt, int(item["support"]))
        )
        if nr:
            row_meta = item["row_meta"]
            row_meta[:, -2:] += fork_off
            row_meta_parts.append(row_meta)
            row_neg_parts.append(item["row_neg_bounds"] + e_off)
            fork_parts.append(item["fork_tokens"])
            fork_off += len(item["fork_tokens"])
        if ne:
            ev_col_parts.append(item["ev_col"])
            ev_cat_parts.append(item["ev_cat"])
            ev_ch_parts.append(item["ev_ch"])
            ev_share_parts.append(item["ev_share"])
            ev_role_parts.append(item["ev_role_bounds"] + role_off)
        if nrole:
            role_parts.append(item["role_ids"])
        if ngt:
            gt_parts.append(item["gt_meta"])
        r_off += nr
        e_off += ne
        role_off += nrole
        gt_off += ngt

    arrays = dict(
        view_bounds=np.asarray(view_bounds, np.int32).reshape(-1, 5),
        row_meta=np.concatenate(row_meta_parts)
        if row_meta_parts
        else np.empty((0, 9), np.int32),
        fork_tokens=np.concatenate(fork_parts)
        if fork_parts
        else np.empty(0, np.uint16),
        row_neg_bounds=np.concatenate(row_neg_parts)
        if row_neg_parts
        else np.empty((0, 2), np.int32),
        ev_col=np.concatenate(ev_col_parts)
        if ev_col_parts
        else np.empty(0, np.uint16),
        ev_cat=np.concatenate(ev_cat_parts) if ev_cat_parts else np.empty(0, np.uint8),
        ev_ch=np.concatenate(ev_ch_parts) if ev_ch_parts else np.empty(0, np.uint8),
        ev_share=np.concatenate(ev_share_parts)
        if ev_share_parts
        else np.empty(0, np.float32),
        ev_role_bounds=np.concatenate(ev_role_parts)
        if ev_role_parts
        else np.empty((0, 2), np.int32),
        role_ids=np.concatenate(role_parts) if role_parts else np.empty(0, np.int32),
        gt_meta=np.concatenate(gt_parts)
        if gt_parts
        else np.empty((0, 6), np.int32),
    )

    hasher = hashlib.sha256()
    for k in sorted(arrays):
        hasher.update(k.encode("utf8"))
        hasher.update(arrays[k].tobytes())
    digest = hasher.hexdigest()[:20]
    archive_name = f"physical-{digest}.npz"
    archive_path = output_dir / archive_name
    if not archive_path.exists():
        with tempfile.NamedTemporaryFile(
            dir=output_dir, suffix=".npz", delete=False
        ) as tmp:
            tmp_name = tmp.name
            np.savez(tmp, **arrays)
        os.replace(tmp_name, archive_path)

    manifest_doc = dict(
        schema=PHYSICAL_PACK_SCHEMA,
        source_schema=PHYSICAL_SIDECAR_SCHEMA,
        archive=archive_name,
        chart_archive=doc["archive"],
        views=len(view_bounds),
        records=records_index,
    )
    out_manifest = output_dir / "physical-manifest.json"
    with tempfile.NamedTemporaryFile(
        mode="w", dir=output_dir, encoding="utf8", delete=False
    ) as tmp:
        tmp_name = tmp.name
        json.dump(manifest_doc, tmp, ensure_ascii=False, indent=2)
    os.replace(tmp_name, out_manifest)
    return out_manifest


def clone_grammar(g, next_index):
    gf = copy(g)
    gf.counts = g.counts.copy()
    gf.non_slide_pads = g.non_slide_pads.copy()
    gf.notes = list(g.notes)
    gf.index = next_index - 1
    return gf


def _choquet_levels(col_weights, legal_set):
    """Exact level-set decomposition preserving boolean log-complement barriers."""
    filtered = {
        c: min(1.0, float(w))
        for c, w in col_weights.items()
        if c in legal_set and w > 1e-7
    }
    if not filtered:
        return ()
    levels = sorted(set(filtered.values()))
    out = []
    prev = 0.0
    for lv in levels:
        cols = tuple(sorted(c for c, w in filtered.items() if w >= lv - 1e-9))
        delta = lv - prev
        prev = lv
        if delta > 1e-7:
            out.append((cols, delta))
    return out


class PhysicalPack:
    """Worker-friendly reader over a packed Physical archive."""

    def __init__(self, manifest_path, *, strict=True):
        manifest_path = Path(manifest_path)
        raw_bytes = manifest_path.read_bytes()
        doc = json.loads(raw_bytes)
        if doc.get("schema") not in (PHYSICAL_PACK_SCHEMA, LEGACY_PACK_SCHEMA):
            raise ValueError(f"Expected {PHYSICAL_PACK_SCHEMA} at {manifest_path}")
        self.manifest_path = manifest_path
        self.archive_path = manifest_path.parent / doc["archive"]
        self.records = doc["records"]
        self.strict = strict
        self.binding = hashlib.sha256(raw_bytes).digest()
        self._data = None

    @classmethod
    def open(cls, path, chart_manifest, *, workers=12, strict=True):
        target = Path(path)
        if target.is_dir():
            packed = target / "physical-manifest.json"
            if not packed.is_file():
                packed = pack_physical_cache(
                    target, chart_manifest, target, workers=workers
                )
            return cls(packed, strict=strict)
        return cls(target, strict=strict)

    def _ensure_loaded(self):
        if self._data is None:
            with np.load(self.archive_path) as pack:
                self._data = {k: pack[k] for k in pack.files}
        return self._data

    def has_view(self, chart_id, flip):
        return str(int(flip)) in self.records.get(chart_id, {})

    def prepare_plan(
        self,
        chart_id,
        flip,
        moments,
        end_tick,
        song,
        *,
        seed,
        epoch,
        warning_weight=0.0,
        use_share=True,
        gt_conflict_ce_scale=0.25,
    ):
        rec = self.records.get(chart_id, {}).get(str(int(flip)))
        if rec is None:
            if self.strict:
                raise KeyError(
                    f"Missing Physical sidecar for chart {chart_id} mirror={int(flip)}"
                )
            return None

        data = self._ensure_loaded()
        v_idx = int(rec["index"])
        r_lo, r_hi, gt_lo, gt_hi, support = data["view_bounds"][v_idx].tolist()
        trace = trace_chart_anchors(
            moments, end_tick, song, bool(flip), seed=seed, epoch=epoch
        )
        anchors = trace["anchors"]
        slot_meta = trace["slot_meta"]
        close_meta = trace["close_meta"]
        route_meta = trace["route_meta"]
        endpoint_meta = trace["endpoint_meta"]

        gt_rows = set()
        if gt_conflict_ce_scale != 1.0 and gt_hi > gt_lo:
            for kind, a0, a1, a2, _, sev in data["gt_meta"][gt_lo:gt_hi].tolist():
                if not (sev & FLAG_GT_STRONG) and not (
                    warning_weight > 0.0 and (sev & FLAG_GT_WARN)
                ):
                    continue
                row_ep = anchors.get((kind, a0, a1, a2))
                if row_ep is not None:
                    gt_rows.add(int(row_ep))

        row_meta_list = data["row_meta"][r_lo:r_hi].tolist()
        row_neg_arr = data["row_neg_bounds"][r_lo:r_hi]
        if len(row_meta_list):
            e_base = int(row_neg_arr[0, 0])
            e_stop = int(row_neg_arr[-1, 1])
            row_neg_list = (row_neg_arr - e_base).tolist()
            ev_col_list = data["ev_col"][e_base:e_stop].tolist()
            ev_cat_list = data["ev_cat"][e_base:e_stop].tolist()
            ev_ch_list = data["ev_ch"][e_base:e_stop].tolist()
            ev_share_list = data["ev_share"][e_base:e_stop].tolist()
            ev_roles_arr = data["ev_role_bounds"][e_base:e_stop]
            has_any_roles = bool(len(ev_roles_arr) and ev_roles_arr[-1, 1] > ev_roles_arr[0, 0])
            if has_any_roles:
                rl_base = int(ev_roles_arr[0, 0])
                rl_stop = int(ev_roles_arr[-1, 1])
                ev_roles_list = (ev_roles_arr - rl_base).tolist()
                role_ids_list = data["role_ids"][rl_base:rl_stop].tolist()
            else:
                ev_roles_list = None
                role_ids_list = None
        else:
            row_neg_list = ()
            ev_col_list = ev_cat_list = ev_ch_list = ev_share_list = ()
            ev_roles_list = role_ids_list = None

        # Merge weights by (parent_row, fork_tokens) -> cat -> col -> max_weight.
        grouped = {}
        fork_requests = defaultdict(set)
        touch_start = NEW_PADS[8]

        fork_data = data.get("fork_tokens")
        if fork_data is not None and row_meta_list:
            fork_base = row_meta_list[0][-2]
            fork_data = fork_data[fork_base : row_meta_list[-1][-1]].tolist()
        for (kind, a0, a1, a2, _, flags, prior_bmask, f_lo, f_hi), (n_lo, n_hi) in zip(
            row_meta_list, row_neg_list
        ):
            key = (kind, a0, a1, a2)
            parent = anchors.get(key)
            if parent is None:
                continue

            committed = None
            forbid_touch_pad = False
            if kind == ANCHOR_SLOT:
                _, _, headless_ep, ptc_ep, committed = slot_meta[key]
                if headless_ep:
                    continue
                if (flags & FLAG_PLAIN_TOUCH_CLOSED) and not ptc_ep:
                    forbid_touch_pad = True
            elif kind == ANCHOR_CLOSE:
                _, committed = close_meta[key]
            elif kind == ANCHOR_ROUTE:
                _, committed, ep_bmask = route_meta[key]
                if (prior_bmask & ~ep_bmask) != 0:
                    continue
            elif kind == ANCHOR_ENDPOINT:
                _, committed, ep_bmask = endpoint_meta[key]
                if (prior_bmask & ~ep_bmask) != 0:
                    continue

            if fork_data is None:
                fork = tuple(tok for tok in (f_lo, f_hi) if tok >= 0)
            else:
                fork = tuple(fork_data[f_lo - fork_base : f_hi - fork_base])
            q_key = (parent, fork)
            cats = None

            for e_idx in range(n_lo, n_hi):
                ch_mask = ev_ch_list[e_idx]
                cw = (
                    1.0
                    if (ch_mask & STRONG_MASK)
                    else (warning_weight if (ch_mask & WARNING_MASK) else 0.0)
                )
                if cw <= 0.0:
                    continue
                col = ev_col_list[e_idx]
                if forbid_touch_pad and col >= touch_start:
                    continue
                if ev_roles_list is not None and committed is not None:
                    rl_lo, rl_hi = ev_roles_list[e_idx]
                    if rl_hi > rl_lo:
                        ok = True
                        for rid in role_ids_list[rl_lo:rl_hi]:
                            if rid not in committed:
                                ok = False
                                break
                        if not ok:
                            continue
                w = cw * (ev_share_list[e_idx] if use_share else 1.0)
                if w <= 1e-7:
                    continue
                if cats is None:
                    cats = grouped.setdefault(q_key, {})
                cat = ev_cat_list[e_idx]
                cm = cats.setdefault(cat, {})
                if w > cm.get(col, 0.0):
                    cm[col] = w

            if cats and fork:
                fork_requests[parent].update(fork[:i] for i in range(1, len(fork) + 1))

        return dict(
            tokens=trace["tokens"],
            support=max(1, int(support)),
            gt_rows=gt_rows,
            gt_conflict_ce_scale=float(gt_conflict_ce_scale),
            grouped=grouped,
            fork_requests={p: sorted(fs) for p, fs in fork_requests.items()},
        )


def should_fold_fork(cats, fork, legal_set, has_children):
    """Return True when an isolated one-token query folds into the parent."""
    if has_children or len(fork) != 1:
        return False
    if not cats:
        return True
    for cm in cats.values():
        valid_cm = {c: w for c, w in cm.items() if c in legal_set}
        if not valid_cm:
            continue
        if len(valid_cm) == len(legal_set):
            min_w = min(valid_cm.values())
            if all(w - min_w <= 1e-7 for w in valid_cm.values()):
                continue
            return False
        return False
    return True


def finalize_physical_plan(
    plan,
    seq,
    union_pairs_arr,
    main_legal_packed,
    fork_snapshots,
    real_to_comb,
    fork_comb_rows,
):
    """Build sparse Physical loss tensors over in-place combined row indices."""
    alt = defaultdict(set)
    for r, c in union_pairs_arr.tolist():
        alt[r].add(c)
    legal_bool = np.unpackbits(main_legal_packed, axis=1)[:, :SIZE].astype(bool)
    legal_counts = legal_bool.sum(axis=1).tolist()

    direct_extra = defaultdict(lambda: defaultdict(dict))
    pending_fork_queries = []

    for (parent, fork), cats in sorted(plan["grouped"].items()):
        positives_p = alt[parent] | {int(seq[parent + 1])}
        comb_p = int(real_to_comb[parent])
        row_legal = legal_bool[comb_p]
        if not fork:
            for cat, cm in cats.items():
                dest = direct_extra[parent][cat]
                for col, w in cm.items():
                    if col not in positives_p and row_legal[col]:
                        if w > dest.get(col, 0.0):
                            dest[col] = w
            continue

        snap = fork_snapshots.get((parent, fork))
        if snap is None:
            continue
        legal_f = snap["legal_set"]
        if not legal_f:
            continue
        rem_cats = {}
        for cat, cm in cats.items():
            valid_cm = {c: w for c, w in cm.items() if c in legal_f}
            if not valid_cm:
                continue
            if len(fork) == 1 and len(valid_cm) == len(legal_f):
                min_w = min(valid_cm.values())
                dest = direct_extra[parent][cat]
                f0 = fork[0]
                if min_w > dest.get(f0, 0.0):
                    dest[f0] = min_w
                residual = {
                    c: w - min_w for c, w in valid_cm.items() if w - min_w > 1e-7
                }
                if not residual:
                    continue
                valid_cm = residual
            rem_cats[cat] = valid_cm

        if rem_cats and (parent, fork) in fork_comb_rows:
            pending_fork_queries.append((parent, fork, legal_f, rem_cats))

    pair_eval_rows = []
    pair_cats = []
    pair_scales = []
    edge_pair = []
    edge_col = []
    fork_gates = []

    for parent, cats in sorted(direct_extra.items()):
        comb_p = int(real_to_comb[parent])
        row_legal = legal_bool[comb_p]
        n_leg = legal_counts[comb_p]
        for cat, cm in sorted(cats.items()):
            filtered = {
                c: min(1.0, float(w))
                for c, w in cm.items()
                if row_legal[c] and w > 1e-7
            }
            if not filtered:
                continue
            prev = 0.0
            for lv in sorted(set(filtered.values())):
                cols = tuple(sorted(c for c, w in filtered.items() if w >= lv - 1e-9))
                delta = lv - prev
                prev = lv
                if delta > 1e-7 and len(cols) < n_leg:
                    p_idx = len(pair_eval_rows)
                    pair_eval_rows.append(comb_p)
                    pair_cats.append(cat)
                    pair_scales.append(delta)
                    edge_pair.extend([p_idx] * len(cols))
                    edge_col.extend(cols)

    for parent, fork, legal_f, rem_cats in pending_fork_queries:
        eval_row = int(fork_comb_rows[(parent, fork)])
        parent_rows = [int(real_to_comb[parent])]
        parent_rows.extend(
            fork_comb_rows[(parent, fork[:i])] for i in range(1, len(fork))
        )
        while len(fork_gates) < len(fork):
            fork_gates.append(([], [], []))
        for cat, cm in sorted(rem_cats.items()):
            for cols, delta in _choquet_levels(cm, legal_f):
                p_idx = len(pair_eval_rows)
                pair_eval_rows.append(eval_row)
                pair_cats.append(cat)
                pair_scales.append(delta)
                edge_pair.extend([p_idx] * len(cols))
                edge_col.extend(cols)
                for (pairs, rows, tokens), row, col in zip(fork_gates, parent_rows, fork):
                    pairs.append(p_idx)
                    rows.append(row)
                    tokens.append(col)

    return dict(
        pair_eval_rows=torch.tensor(pair_eval_rows, dtype=torch.int64),
        pair_cats=torch.tensor(pair_cats, dtype=torch.int64),
        pair_scales=torch.tensor(pair_scales, dtype=torch.float32),
        edge_pair=torch.tensor(edge_pair, dtype=torch.int64),
        edge_col=torch.tensor(edge_col, dtype=torch.int64),
        fork_gates=[
            tuple(torch.tensor(values, dtype=torch.int64) for values in gate)
            for gate in fork_gates
            if gate[0]
        ],
        support=plan["support"],
    )


def pack_physical_batches(charts, cumulative):
    parts = [c["physical"] for c in charts]
    pair_offsets = np.cumsum([0, *(len(p["pair_eval_rows"]) for p in parts[:-1])])
    gates_by_depth = []
    for p, row_off, pair_off in zip(parts, cumulative, pair_offsets):
        for depth, (idx, rows, cols) in enumerate(p["fork_gates"]):
            if depth == len(gates_by_depth):
                gates_by_depth.append(([], [], []))
            indices, parent_rows, tokens = gates_by_depth[depth]
            indices.append(idx + int(pair_off))
            parent_rows.append(rows + int(row_off))
            tokens.append(cols)
    return dict(
        pair_eval_rows=torch.cat(
            [p["pair_eval_rows"] + int(off) for p, off in zip(parts, cumulative)]
        ),
        pair_cats=torch.cat([p["pair_cats"] for p in parts]),
        pair_scales=torch.cat([p["pair_scales"] for p in parts]),
        edge_pair=torch.cat(
            [p["edge_pair"] + int(off) for p, off in zip(parts, pair_offsets)]
        ),
        edge_col=torch.cat([p["edge_col"] for p in parts]),
        fork_gates=[tuple(torch.cat(values) for values in gate) for gate in gates_by_depth],
        support=sum(p["support"] for p in parts),
    )


def _fork_gate_log_safe(z, parent_rows, fork_cols, child_log_safe):
    zf = z.index_select(0, parent_rows)
    lse = zf.logsumexp(-1)
    log_pf = zf.gather(1, fork_cols[:, None]).squeeze(1) - lse
    zf_comp = zf.scatter(1, fork_cols[:, None], -torch.inf)
    has_comp = torch.isfinite(zf_comp).any(-1)
    safe_comp = torch.where(has_comp[:, None], zf_comp, torch.zeros_like(zf_comp))
    log_1m_pf = safe_comp.logsumexp(-1) - lse
    gated = torch.logaddexp(log_1m_pf, log_pf + child_log_safe)
    return torch.where(has_comp, gated, child_log_safe)


def physical_loss(z, phys, category_weights=(1.0, 1.0, 4.0, 1.0)):
    """Vectorized 4-category complement log-loss without host syncs or 1-p cancellation."""
    pair_rows = phys["pair_eval_rows"]
    support = max(1, int(phys["support"]))
    if not pair_rows.numel():
        zeros = z.new_zeros(4)
        return zeros.sum(), zeros, support

    zp = z.index_select(0, pair_rows)
    lse = zp.logsumexp(-1)
    safe_z = zp.clone()
    if phys["edge_pair"].numel():
        safe_z.index_put_(
            (phys["edge_pair"], phys["edge_col"]),
            safe_z.new_tensor(-torch.inf),
            accumulate=False,
        )
    has_safe = torch.isfinite(safe_z).any(-1)
    safe_finite = torch.where(has_safe[:, None], safe_z, torch.zeros_like(safe_z))
    log_safe = torch.where(has_safe, safe_finite.logsumexp(-1) - lse, -torch.inf)

    for idx, rows, cols in reversed(phys["fork_gates"]):
        gated = _fork_gate_log_safe(z, rows, cols, log_safe.index_select(0, idx))
        log_safe = log_safe.index_copy(0, idx, gated)

    floor = math.log(torch.finfo(torch.float32).eps)
    pair_loss = -log_safe.clamp(min=floor, max=0.0) * phys["pair_scales"]
    cat_sums = z.new_zeros(4).scatter_add_(0, phys["pair_cats"], pair_loss)
    cat_means = cat_sums / support
    weights = z.new_tensor(category_weights, dtype=torch.float32)
    total = (cat_means * weights).sum() / 4.0
    return total, cat_means, support


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args(argv)
    out = pack_physical_cache(
        args.cache_dir, args.manifest, args.output, workers=args.workers
    )
    print(json.dumps(dict(packed_manifest=str(out)), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
