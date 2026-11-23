"""Refresh a focused Stage 2 cache from its original split and BMS charts, reusing all audio.

Run python -m maidata.multitask.refresh_focused_labels --help for the input paths.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
import zipfile

import numpy as np
import torch

from .grid import BeatGrid
from . import bms
from .bms_calibrator import (
    EXTENDED_METERS, _norm_dedup_token, _strip_charter_credit, _supervision_clusters, _supervision_rank,
    _compact_supervision_metadata, _focused_hard, _focused_selection_counts, _focused_split, excluded_song,
    sanitize_bms_calibrator_metadata, summarize_calibrator_chart,
)


# Original validation and training chart keys matched by keysound names and relative onsets.
REVIEWED_SONG_PAIRS = {
    "e3002588591cd86214e7": "eb5459f17584d2665c60",  # Ascension to Heaven
    "30c9402cb373c46eb465": "ba34d4ab2c8da75c3ef4",  # サンバランド
    "e31e55a67c55f5d53832": "d2852d2d1b1fc9021238",  # Green Cherry
    "f50ada8a80073e155636": "08daf46d95a9366ac4b9",  # 紅月夜想
    "413c71ed2153c5919a47": "24b881a8a57e4cc4a07e",  # モリオンの姫
    "cd9a5088ba6e6dcf3390": "67415b1bdb5e9ac71c43",  # conflict
}


def _performed_meters(meta, triggers):
    grid = BeatGrid(meta)
    bars = np.asarray(meta.get("bar_beats", ()), dtype=np.float32)
    if len(bars) < 2 or not triggers:
        return []
    first = grid.second_to_beat(min(t.time for t in triggers))
    last = grid.second_to_beat(max(t.end if t.end is not None else t.time for t in triggers))
    middle = (bars[:-1] + bars[1:]) / 2
    segment = np.searchsorted(np.asarray(grid.meter_starts, dtype=np.float32), middle, side="right") - 1
    meters = set()
    for i in np.flatnonzero((first <= middle) & (middle <= last)):
        meter = grid.meter_segments[max(0, segment[i])][1:]
        if meter in EXTENDED_METERS and abs(bars[i + 1] - bars[i] - 4 * meter[0] / meter[1]) < 1e-5:
            meters.add(f"{meter[0]}/{meter[1]}")
    return sorted(meters)


def _clean_batch(task):
    charts_zip, rows = task
    accepted, rejected = [], []
    with zipfile.ZipFile(charts_zip) as archive:
        for row in rows:
            payload = archive.read(row["chart"])
            parsed = bms.parse_bms(row["chart"], payload=payload)
            row = dict(row, chart_sha1=hashlib.sha1(payload).hexdigest(),
                       norm_title=_norm_dedup_token(row["title"], strip_diff=True),
                       norm_artist=_strip_charter_credit(row.get("artist", "")))
            meta, status = sanitize_bms_calibrator_metadata(parsed.metadata, triggers=parsed.triggers, return_status=True)
            if meta is None:
                rejected.append(dict(row, reason=status))
                continue
            summary = summarize_calibrator_chart(parsed, meta, status)
            accepted.append(dict(row, **summary, performed_meters=_performed_meters(meta, parsed.triggers),
                                 metadata=_compact_supervision_metadata(meta)))
    return accepted, rejected


def _add_bytes(tar, name, data):
    member = tarfile.TarInfo(name)
    member.size = len(data)
    tar.addfile(member, io.BytesIO(data))


def _json(data):
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def refresh_focused_labels(split_manifest, charts_zip, train_manifest, val_manifest, output,
                           workers=12, seed=20261114, full_bms=False):
    if Path(split_manifest).suffix.lower() == ".zip":
        with zipfile.ZipFile(split_manifest) as archive:
            source = json.loads(archive.read("split_manifest.json"))
    else:
        source = json.loads(Path(split_manifest).read_bytes())
    cache = {split: json.loads(Path(path).read_bytes()) for split, path in
             (("train", train_manifest), ("val", val_manifest))}
    available = {split: {r["key"]: r for r in doc["records"] if not excluded_song(r)}
                 for split, doc in cache.items()}
    rows = [r for r in source["records"] if not excluded_song(r)]
    tasks = [(charts_zip, rows[i:i + 48]) for i in range(0, len(rows), 48)]
    clean, rejected = [], []
    with ProcessPoolExecutor(workers) if workers > 0 else nullcontext() as pool:
        for records, rejects in (pool.map(_clean_batch, tasks) if pool else map(_clean_batch, tasks)):
            clean.extend(records)
            rejected.extend(rejects)

    # A rejected or uncached original train copy still rules its group out of validation.
    all_available = {**available["train"], **available["val"]}
    duplicates, unique, reviewed_groups = [], [], []
    reviewed = REVIEWED_SONG_PAIRS.items() if full_bms else ()
    for group in _supervision_clusters([*clean, *rejected], reviewed_pairs=reviewed):
        original_train = any(r["split"] == "train" or (full_bms and r["key"] in REVIEWED_SONG_PAIRS) for r in group)
        ranked = sorted(group, key=lambda r: (original_train and r["split"] != "train",
                        "reason" in r, r["key"] not in all_available, _supervision_rank(r)))
        if full_bms:
            ranked.sort(key=lambda r: ("reason" in r, r["key"] not in all_available))
        winner = ranked[0]
        if full_bms and original_train:
            winner = dict(winner, split="train")
            for key in {r["key"] for r in group} & REVIEWED_SONG_PAIRS.keys():
                reviewed_groups.append(dict(val_key=key, original_train_key=REVIEWED_SONG_PAIRS[key],
                                           kept_train_key=winner["key"] if "reason" not in winner else None))
        if "reason" not in winner:
            unique.append(winner)
        for dup in ranked[1:]:
            a, b = (all_available.get(r["key"]) for r in (winner, dup))
            same_audio = None if not a or not b else (
                a["sample_length"] == b["sample_length"] and round(a["song_ref_db"], 6) == round(b["song_ref_db"], 6))
            duplicates.append(dict(key=dup["key"], kept_key=winner["key"], split=dup["split"],
                                   same_chart=dup["chart_sha1"] == winner["chart_sha1"], same_cache_fingerprint=same_audio))

    candidates = unique if full_bms else [r for r in unique if r["key"] in all_available or _focused_hard(r)]
    selections, stats = {}, {}
    for split, quota in (("train", 500), ("val", 55)):
        members = [r for r in candidates if r["split"] == split]
        if full_bms:
            retained, stats[split] = members, {}
        else:
            retained, stats[split] = _focused_split(members, quota, f"{seed}:{split}")
        stats[split]["original_split"] = sum(r["split"] == split for r in rows)
        for row in retained:
            row.pop("norm_title", None)
            row.pop("norm_artist", None)
        selections[split] = retained
    summary = dict(seed=seed, splits=stats, duplicates=duplicates, reviewed_same_song_pairs=reviewed_groups,
                   rejected=[{key: r[key] for key in ("key", "split", "title", "reason")} for r in rejected])
    return _write_focused_release(output, source, cache, selections, summary, full_bms=full_bms)


def _write_focused_release(output, source, cache, selections, summary, *, full_bms=False):
    """Assign the sampled charts by song, then emit their cached records and labels."""
    all_available = {r["key"]: r for doc in cache.values() for r in doc["records"] if not excluded_song(r)}
    train = {r["key"]: r for r in selections["train"]}
    val = {r["key"]: r for r in selections["val"]}
    reviewed = summary.get("reviewed_same_song_pairs", [])
    for val_key, train_key in REVIEWED_SONG_PAIRS.items():
        row = val.pop(val_key, None)
        if row is None:
            continue
        kept_key = train_key
        if train_key not in train or (train_key not in all_available and val_key in all_available):
            train.pop(train_key, None)
            train[val_key] = dict(row, split="train")
            kept_key = val_key
        reviewed.append(dict(val_key=val_key, original_train_key=train_key, kept_train_key=kept_key))
    summary["reviewed_same_song_pairs"] = reviewed

    def meter_counts(records):
        return Counter(m for r in records if r["key"] in all_available for m in r["performed_meters"])

    summary["val_meter_song_counts_before_migration"] = dict(meter_counts(val.values()))
    migrated = set()
    while val and not full_bms:
        rare = {m for m, count in meter_counts(val.values()).items() if count <= 2}
        move = [key for key, r in val.items() if key in all_available and rare.intersection(r["performed_meters"])]
        if not move:
            break
        migrated.update(move)
        for key in move:
            train[key] = dict(val.pop(key), split="train")
    summary["migrated_val_rare_keys"] = sorted(migrated)

    selected, docs = [], {}
    for split, members in (("train", train), ("val", val)):
        retained = [members[key] for key in sorted(members)]
        selected.extend(retained)
        present = [r for r in retained if r["key"] in all_available]
        stats = summary["splits"][split]
        if full_bms:
            stats["total"] = len(retained)
        else:
            stats.update(_focused_selection_counts(retained))
        stats["missing_cache_keys"] = [r["key"] for r in retained if r["key"] not in all_available]
        stats["cache_records"] = len(present)
        stats["meter_songs"] = dict(meter_counts(present))
        doc = dict(cache[split], records=[all_available[r["key"]] for r in present])
        if "count" in doc:
            doc["count"] = len(present)
        docs[split] = doc

    prefix = "full-bms" if full_bms else "focused"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as tmp:
        root = Path(tmp)
        archive = root / f"{prefix}-labels-patch.tar.gz"
        with archive.open("wb") as dest, gzip.GzipFile(filename="", mode="wb", fileobj=dest, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w|") as tar:
                for split, doc in docs.items():
                    digest = hashlib.sha256()
                    for row in selected:
                        if row["split"] != split or row["key"] not in all_available:
                            continue
                        data = io.BytesIO()
                        torch.save(dict(metadata=row["metadata"]), data)
                        payload = data.getvalue()
                        digest.update(payload)
                        _add_bytes(tar, all_available[row["key"]]["labels"], payload)
                    doc["labels_sha256"] = digest.hexdigest()
                    _add_bytes(tar, split + ".json", _json(doc))
        (root / f"{prefix}_summary_v2.json").write_bytes(_json(summary))
        (root / f"{prefix}_split_manifest_v2.json").write_bytes(_json(dict(
            schema=source["schema"], producer=source.get("producer"), seed=summary["seed"], records=selected)))
        for path in root.iterdir():
            path.replace(output / path.name)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("split-manifest", "charts-zip", "train-manifest", "val-manifest", "output"):
        parser.add_argument("--" + arg, required=True)
    parser.add_argument("--full-bms", action="store_true", help="retain all cleaned BMS songs in the original split, without focused sampling or rare-meter migration")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261114)
    summary = refresh_focused_labels(**vars(parser.parse_args()))
    for split, stats in summary["splits"].items():
        print(split, "selected", stats["total"], "cached", stats["cache_records"], "meters", stats["meter_songs"])


if __name__ == "__main__":
    main()
