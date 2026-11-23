"""Prepare canonical ESE chart packs and their independent v2 audio cache."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unicodedata

import numpy as np
import torch

from . import AUDIO_PACK_SCHEMA, TAIKO_PACK_SCHEMA
from .audio import (
    AUDIO_MODELS,
    DEFAULT_MERT_V2_MODEL,
    extract_flat_audio_batch,
    write_flat_audio,
)
from .tja import parse_tja_file

HERE = Path(__file__).parent
RIDGE = HERE / "taiko_ridge.json"
EXCLUSIONS = HERE / "taiko_exclusions.json"


def title_key(value):
    text = unicodedata.normalize("NFKC", value).casefold()
    text = re.sub(r"\[(?:sd|dx)\]", "", text)
    return "".join(c for c in text if c.isalnum())


def excluded_titles(normalized, test_set):
    titles = {
        "Calamity Fortune",
        "Oshama Scramble!",
        "Synthesis.",
        "全世界共通リズム感テスト",
    }
    song_titles = {}
    if normalized:
        for row in json.loads(normalized.read_bytes())["records"]:
            song_titles[str(row["song"])] = row["title"]
            if row["split"] != "fit" or str(row["song"]) == "854":
                titles.add(row["title"])
    if test_set:
        for file in test_set.rglob("maidata.txt"):
            if any(part.startswith(".") for part in file.relative_to(test_set).parts):
                continue
            folder = file.parent.name
            if folder in song_titles:
                titles.add(song_titles[folder])
            elif not folder.isdecimal():
                titles.add(folder)
            for line in file.read_text("utf-8-sig", errors="replace").splitlines():
                if line.startswith("&title="):
                    titles.add(line.partition("=")[2].strip())
                    break
    return sorted(t for t in titles if title_key(t))


def ridge_features(chart):
    hits = [e for e in chart.events if e.kind <= 4]
    seconds = np.array([e.second for e in hits])
    gaps = np.diff(seconds)
    n = len(hits)
    active = n / max(1.0, seconds[-1] - seconds[0]) if n else 0.0

    def rates(width):
        return (
            (np.searchsorted(seconds, seconds + width, side="left") - np.arange(n))
            / width
            if n
            else [0.0]
        )

    two = rates(2)
    changes = np.diff([e.kind % 2 for e in hits]) != 0

    def fraction(flags):
        return float(np.mean(flags)) if len(flags) else 0.0

    star = chart.star_level
    return [
        star,
        star**2,
        *(float(chart.course == c) for c in ("hard", "oni", "edit")),
        active,
        np.log1p(active),
        np.percentile(two, 95),
        np.max(two),
        np.percentile(rates(5), 95),
        fraction(gaps < 0.12),
        fraction(gaps < 0.09),
        fraction(changes),
        fraction(changes & (gaps < 0.12)),
        fraction([e.grid32_tick % 96 in (32, 64) for e in hits]),
        np.log1p(n),
        float(chart.has_bpmchange),
        fraction([e.kind in (3, 4) for e in hits]),
    ]


def fit_ridge(samples, alpha=1.0):
    features, truth = zip(*samples)
    x, y = np.asarray(features), np.asarray(truth)
    center, scale = x.mean(0), x.std(0)
    scale[scale == 0] = 1
    z = (x - center) / scale
    weights = (
        np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y - y.mean()))
        / scale
    )
    return dict(
        weights=weights.tolist(),
        intercept=float(y.mean() - center @ weights),
        exact_charts=len(y),
    )


@lru_cache(maxsize=4096)
def wave_oid(path):
    with path.open("rb") as fp:
        prefix = fp.read(256)
        if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
            return re.search(rb"oid sha256:([0-9a-f]{64})", prefix + fp.read())[1].decode()
        fp.seek(0)
        return hashlib.file_digest(fp, "sha256").hexdigest()


def collect(root, mapping, constants, excluded):
    excluded = excluded - {""}
    ids = {path.replace(chr(92), "/").casefold(): key for key, path in mapping.items()}
    by_audio, blocked, exact = {}, set(), []
    for path in sorted(root.rglob("*.tja")):
        choices = {}
        for chart in parse_tja_file(path, root=root):
            audio = path.parent / chart.wave
            oid = wave_oid(audio)
            if excluded.intersection(
                map(title_key, (chart.title, chart.title_ja, audio.stem, path.stem))
            ):
                blocked.add(oid)
            if any(w.startswith("negative_delay") for w in chart.warnings):
                continue
            key = ids.get(chart.path.casefold())
            constant = constants.get(key, {}).get(chart.course, {}).get("main_constant")
            features = ridge_features(chart)
            if constant is not None and oid not in blocked:
                exact.append((oid, features, constant))
            events = [
                (e.grid32_tick, e.kind, e.end_grid32_tick or 0, e.balloon_hits or 0)
                for e in chart.events
            ]
            meta = dict(
                chart.beat_grid_metadata(),
                difficulty=8 + chart.course_id,
                title=chart.title,
            )
            end = max(round(chart.total_beats * 96), events[-1][0] if events else 0)
            row = dict(
                audio_id=oid,
                id=chart.path + ":" + chart.course,
                source=chart.path,
                wave=audio.relative_to(root).as_posix(),
                course=chart.course,
                star_level=chart.star_level,
                end_tick=end,
                field_limit=max(end, max((e[2] for e in events), default=0)),
                meta=meta,
                constant=constant,
                features=features,
                events=events,
            )
            choices.setdefault(oid, {}).setdefault(chart.course, row)
        for oid, courses in choices.items():
            # Prefer a complete canonical chart, then the mapped chart, then lexical order.
            priority = (
                len(courses),
                sum(c["constant"] is not None for c in courses.values()),
            )
            if oid not in by_audio or priority > by_audio[oid][0]:
                by_audio[oid] = priority, courses
    return (
        [
            r
            for oid, (_, courses) in by_audio.items()
            if oid not in blocked
            for r in courses.values()
        ],
        [(f, g) for oid, f, g in exact if oid not in blocked],
    )


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf8", newline="\n", dir=path.parent, delete=False
    ) as out:
        json.dump(value, out, ensure_ascii=False, indent=2)
        out.write("\n")
    Path(out.name).replace(path)


def bake(args):
    exclusions = (
        set(json.loads(EXCLUSIONS.read_bytes())) if EXCLUSIONS.exists() else set()
    )
    exclusions.update(excluded_titles(args.maimai_normalized, args.test_set))
    rows, exact = collect(
        args.ese,
        json.loads(args.mapping.read_bytes()),
        json.loads(args.constants.read_bytes()),
        set(map(title_key, exclusions)),
    )
    if args.fit_ridge:
        ridge = fit_ridge(exact)
        write_json(args.fit_ridge, ridge)
    else:
        ridge = json.loads(RIDGE.read_bytes())
    chunks = []
    offset = 0
    for row in rows:
        constant, features = row.pop("constant"), row.pop("features")
        raw = (
            constant
            if constant is not None
            else ridge["intercept"] + np.dot(features, ridge["weights"])
        )
        g = float(np.clip(raw, 0.0, 15.5))
        row.update(
            gugu_constant=g,
            level_source="gugu_exact" if constant is not None else "gugu_imputed",
        )
        row["meta"]["level"] = 10.0 + 5.0 * (g / 15.5) ** 0.55
        events = np.asarray(row["events"], np.int32).reshape(-1, 4)
        chunks.append(events)
        row["events"] = [offset, offset + len(events)]
        offset += len(events)
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.output, delete=False) as tmp:
        np.save(tmp, np.concatenate(chunks) if chunks else np.empty((0, 4), np.int32))
    archive = Path(tmp.name)
    with archive.open("rb") as fp:
        name = "events-" + hashlib.file_digest(fp, "sha256").hexdigest()[:16] + ".npy"
    archive.replace(args.output / name)
    write_json(
        args.output / "manifest.json",
        dict(
            schema=TAIKO_PACK_SCHEMA,
            archive=name,
            audio_manifest=os.path.relpath(
                args.audio / "manifest.json", args.output
            ).replace(chr(92), "/"),
            exclusions=sorted(exclusions),
            records=rows,
        ),
    )
    print(
        f"Packed {len(rows)} charts, {len(set(r['audio_id'] for r in rows))} audio groups",
        flush=True,
    )


def materialize(path, root):
    with path.open("rb") as fp:
        prefix = fp.read(200)
        if not prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
            return path
        pointer = prefix + fp.read()

    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as out:
        temp = Path(out.name)
        try:
            subprocess.run(
                ["git", "lfs", "smudge", "--", path.relative_to(root).as_posix()],
                input=pointer,
                stdout=out,
                cwd=root,
                check=True,
                env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "0"},
            )
        except BaseException:
            out.close()
            temp.unlink()
            raise
    temp.replace(path)
    return path


STORAGE_MODES = {
    "uint16-int8": dict(mel_uint16_2ch=True, int8_mert=True, fp16=False),
    "bf16-int8": dict(mel_uint16_2ch=False, int8_mert=True, fp16=False),
    "int8": dict(mel_uint16_2ch=False, int8_mert=True, fp16=False),
    "fp16-int8": dict(mel_uint16_2ch=False, int8_mert=True, fp16=True),
    "fp16": dict(mel_uint16_2ch=False, int8_mert=False, fp16=True),
}


def extract(args):
    doc = json.loads((args.output / "manifest.json").read_bytes())
    songs = {r["audio_id"]: args.ese / r["wave"] for r in doc["records"]}
    manifest = args.audio / "manifest.json"
    audio = json.loads(manifest.read_bytes())["audio"] if manifest.exists() else {}
    keys = [k for k in songs if k not in audio]
    mode = STORAGE_MODES[args.storage_dtype]
    feature_dtype = (
        torch.float32
        if mode["mel_uint16_2ch"] or mode["int8_mert"]
        else torch.float16
    )
    with ThreadPoolExecutor(args.workers) as pool:
        for i in range(0, len(keys), args.batch_size):
            group = keys[i : i + args.batch_size]
            paths = list(pool.map(lambda k: materialize(songs[k], args.ese), group))
            features = extract_flat_audio_batch(
                paths, args.device, args.mert_model, pool=pool, bundled=args.bundled,
                feature_dtype=feature_dtype,
            )
            for index, key in enumerate(group):
                audio[key] = write_flat_audio(
                    args.audio / "audio" / key, features, index, **mode
                )
            write_json(
                manifest,
                dict(
                    schema=AUDIO_PACK_SCHEMA,
                    audio_models=AUDIO_MODELS,
                    mert_model=args.mert_model,
                    audio=audio,
                ),
            )
            print(f"Cached {len(audio)}/{len(songs)} songs", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("charts", "audio"))
    parser.add_argument("--ese", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("data/packed-taiko-ese-grid24")
    )
    parser.add_argument("--audio", type=Path, default=Path("data/taiko-ese-audio"))
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--constants", type=Path)
    parser.add_argument("--maimai-normalized", type=Path)
    parser.add_argument("--test-set", type=Path)
    parser.add_argument("--fit-ridge", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mert-model", default=DEFAULT_MERT_V2_MODEL)
    parser.add_argument("--bundled", action="store_true")
    parser.add_argument(
        "--storage-dtype",
        choices=tuple(STORAGE_MODES),
        default="uint16-int8",
        help=(
            "Taiko storage for Mel/MERT: 2ch UINT16/INT8 (default), "
            "BF16/INT8 (bf16-int8 or int8), FP16/INT8, or FP16/FP16."
        ),
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    (bake if args.action == "charts" else extract)(args)


if __name__ == "__main__":
    main()
