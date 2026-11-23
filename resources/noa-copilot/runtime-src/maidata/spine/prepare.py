"""Prepare normalized whole charts for fixed within-MOMENT BPE."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
from importlib.resources import files
import json
import os
from pathlib import Path

from . import official
from .codec import normalize
from maidata.designer import designer_name
from maidata.multitask.grid import BeatGrid
from maidata.multitask.maimai import events_from_ir
from maidata.multitask.v7_events import events_from_record, metadata_from_record
from maidata.multitask.prepare import metadata_from_ir


def chart_name(row):
    return row["id"].replace(":", "-")


def save_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")), encoding="utf8"
    )


def _prepare(args):
    root, out, row = args
    path = root / row["source"]
    with gzip.open(path, "rb") as f:
        doc = json.load(f)
    if row["source"].endswith(".ir.json.gz"):
        events = events_from_ir(doc)
        meta = metadata_from_ir(doc)
    else:
        events = events_from_record(doc)
        meta = metadata_from_record(doc)
    meta.update(
        title=row["title"], song_id=row["song"], difficulty=int(row["id"].split(":")[1])
    )
    for fields in (doc.get("source", {}), doc.get("metadata", {}), doc):
        name = designer_name(fields)
        if name is not None:
            meta["charter"] = name
            break
    source = root / "sources" / row["song"] / "maidata.txt"
    if source.is_file():
        import re

        fields = dict(
            re.findall(
                r"^&(lv_\d+|des_\d+)=([^\r\n]*)",
                source.read_text(encoding="utf-8-sig"),
                re.M,
            )
        )
        value = fields.get("lv_" + str(meta["difficulty"]), "").strip()
        if value:
            try:
                meta["level"] = float(value.replace("+", ".7"))
            except ValueError:
                pass
        key = "des_" + str(meta["difficulty"])
        if key in fields:
            meta["charter"] = fields[key].strip()
    moments = normalize(events, BeatGrid(meta))
    name = chart_name(row)
    with gzip.open(
        out / (name + ".json.gz"), "wt", encoding="utf8", compresslevel=1
    ) as f:
        json.dump(
            dict(metadata=meta, moments=moments),
            f,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    row.update(labels=name + ".json.gz", moments=len(moments))
    return row


def prepare(dataset, out, workers=12):
    root = Path(dataset).resolve()
    out = Path(out).resolve()
    out.mkdir(exist_ok=True, parents=True)
    rows = json.loads(
        files("maidata.multitask").joinpath("assets/spine-split.json").read_text("utf8")
    )
    rows = [r for r in rows if official(r)]
    jobs = [(root, out, row) for row in rows]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_prepare, jobs, chunksize=4))
    audio_catalog = json.loads(
        (root / "audio/encoder-inputs/catalog.json").read_text(encoding="utf8")
    )
    audio_ids = {str(r["song_id"]): r["asset"] for r in audio_catalog["songs"]}
    for row in rows:
        recording = root / "sources" / row["song"] / "track.mp3"
        relative = (
            Path(os.path.relpath(recording, out))
            if recording.drive == out.drive
            else recording
        )
        row.update(
            audio_id=audio_ids[row["song"]],
            audio=relative.as_posix(),
        )
    save_json(
        out / "manifest.json", dict(schema="moment205.normalized.v2", records=rows)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    prepare(args.dataset, args.output, args.workers)
