"""Prepared teacher-forcing tensors, per-audio feature cache and complete-epoch grouping."""

import argparse
from collections import defaultdict
from functools import lru_cache
import gzip
import hashlib
import json
import os
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import torch
from torch.utils.data import Dataset
from .codec import mirror_notes
from .pieces import SIZE, WORDS, pad_mask
from .packing import BEAT, chart_pack
from .tails import (
    read_tail,
    end_pads,
    field_plan,
    FIELD_COUNT,
    HAZ_DIM,
    HAZ_OFFSETS,
)
from .history import history_layout
from .tails import PAD_FEEDBACK
from . import official
from .features import io_pool
from .features import (
    chart_clocks,
    condition,
    write_audio,
    read_audio,
    pad_audio,
)
from maidata.multitask.grid import BeatGrid
from maidata.multitask.audio import (
    LogMelFrontend,
    read_mono,
    fast_resample,
    sample_centers_batch,
)
from maidata.multitask.frontends import (
    extract_mert_batch,
    extract_muq_batch,
    load_mert,
    load_muq,
)


DEFAULT_MERT = "m-a-p/MERT-v1-330M"
DEFAULT_MUQ = "OpenMuQ/MuQ-large-msd-iter"


@lru_cache(maxsize=2)
def frozen_frontend(name, path, device):
    # Upstream model construction initializes parameters and MuQ resets Python's RNG.
    # A cache miss must not advance the chart model's dropout or generation streams.
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    target = torch.device(device)
    try:
        with torch.random.fork_rng(devices=[target] if target.type == "cuda" else []):
            if name == "mert":
                return load_mert(path, device, paired=True)
            return load_muq(path, device)
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)


@lru_cache(maxsize=2)
def mel_frontend(device):
    return LogMelFrontend().to(device)


def load_audio_input(paths):
    audio, native = paths
    wave, rate = read_mono(audio)
    macro = (
        torch.load(native, map_location="cpu", weights_only=True)
        if native and native.is_file()
        else {}
    )
    for stored, key in (
        ("mert_l06", "mert_l6"),
        ("mert_l14", "mert"),
        ("muq_l08", "muq"),
    ):
        if stored in macro and key not in macro:
            macro[key] = macro.pop(stored)
    if macro.get("schema") == "maidata.full-mert6-mert14-muq8.v1":
        # This producer stored the standard MuQ convolution's 0 ms frame at 20 ms.
        # Correct its clock in memory while keeping the accepted cache unchanged.
        macro.update(mert_layer=14, muq_layer=8)
    missing = not {"mert", "mert_l6", "muq"}.issubset(macro)
    native_wave = fast_resample(wave, rate, 24000) if missing else None
    return fast_resample(wave, rate, 44100), macro, native_wave


def save_native_cache(dst, macro):
    dst.parent.mkdir(exist_ok=True, parents=True)
    tmp = dst.with_suffix(f".{os.getpid()}.tmp")
    torch.save(
        {
            k: v.to("cpu", copy=True).contiguous() if isinstance(v, torch.Tensor) else v
            for k, v in macro.items()
        },
        tmp,
    )
    tmp.replace(dst)


def merge_mert_layers(macros, owners, encoded):
    # A legacy L14 file may end one frame earlier/later or carry a custom
    # origin. Bring only the newly extracted L6 onto the stored physical grid.
    adjust = [
        j
        for j, i in enumerate(owners)
        if "mert" in macros[i]
        and (
            len(encoded[j]["mert_l6"]) != len(macros[i]["mert"])
            or encoded[j]["mert_origin"] != macros[i]["mert_origin"]
        )
    ]
    if adjust:
        early = [encoded[j]["mert_l6"] for j in adjust]
        values = torch.nn.utils.rnn.pad_sequence(early, batch_first=True)
        reference = [macros[owners[j]] for j in adjust]
        lengths = [len(m["mert"]) for m in reference]
        times = (
            torch.arange(max(lengths), device=values.device, dtype=torch.float64) / 75
        )
        origins = times.new_tensor([m["mert_origin"] for m in reference])
        aligned = sample_centers_batch(
            values,
            [len(x) for x in early],
            75,
            times + origins[:, None],
            [encoded[j]["mert_origin"] for j in adjust],
        )
        for j, x, n in zip(adjust, aligned, lengths):
            encoded[j]["mert_l6"] = x[:n]
    for i, part in zip(owners, encoded):
        if "mert" in macros[i]:
            macros[i]["mert_l6"] = part["mert_l6"]
        else:
            macros[i].update(part)


@torch.no_grad()
def audio_features_batch(
    paths, device, mert, muq, windows_per_song, *, save_native=False
):
    # Disk reads and decoding can overlap without passing whole tracks through DataLoader IPC.
    if len(paths) == 1:
        inputs = [load_audio_input(paths[0])]
    else:
        with ThreadPoolExecutor(max_workers=len(paths)) as pool:
            inputs = list(pool.map(load_audio_input, paths))
    waves, macros, native_waves = map(list, zip(*inputs))
    lengths = [len(w) for w in waves]
    batch = torch.nn.utils.rnn.pad_sequence(waves, batch_first=True).to(device)
    mel, mel_lengths = mel_frontend(str(device)).batch(batch, lengths)

    def waveform_batch(ids):
        cold = [native_waves[i] for i in ids]
        return torch.nn.utils.rnn.pad_sequence(cold, batch_first=True).to(device), [
            len(w) for w in cold
        ]

    need_mert = [
        i for i, m in enumerate(macros) if "mert_l6" not in m or "mert" not in m
    ]
    need_muq = [i for i, m in enumerate(macros) if "muq" not in m]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        if need_mert:
            waves24, samples = waveform_batch(need_mert)
            early_only = all("mert" in macros[i] for i in need_mert)
            encoded = extract_mert_batch(
                waves24,
                frozen_frontend("mert", mert, str(device)),
                windows_per_song,
                samples,
                sixth_only=early_only,
                output_device=device,
            )
            merge_mert_layers(macros, need_mert, encoded)
        if need_muq:
            if need_muq != need_mert:
                waves24, samples = waveform_batch(need_muq)
            encoded = extract_muq_batch(
                waves24,
                frozen_frontend("muq", muq, str(device)),
                windows_per_song,
                samples,
                output_device=device,
            )
            for i, part in zip(need_muq, encoded):
                macros[i].update(part)
    # The original L14/L8 cache writer labeled MuQ's centered convolution as
    # a half-frame grid. The stored vectors are native frames and need no rewrite.
    for m in macros:
        if (
            muq == DEFAULT_MUQ
            and "frame_clock" not in m
            and m.get("mert_layer") == 14
            and m.get("muq_layer") == 8
            and m["muq_origin"] == 0.5 / 25
        ):
            m["muq_origin"] = 0.0
    if save_native:
        writes = [
            (paths[i][1], macros[i])
            for i in sorted(set(need_mert) | set(need_muq))
            if paths[i][1]
        ]
        if writes:
            with ThreadPoolExecutor(max_workers=len(writes)) as pool:
                list(pool.map(save_native_cache, *zip(*writes)))
    result = dict(mel=mel, mel_lengths=mel_lengths, sample_lengths=lengths)
    for key in ("mert", "mert_l6", "muq"):
        native = [m[key] for m in macros]
        # Transfer once for an all-cache batch. Mixed hits and misses can live on different devices.
        if any(v.device != native[0].device for v in native):
            native = [v.to(device) for v in native]
        result[key] = torch.nn.utils.rnn.pad_sequence(native, batch_first=True).to(
            device
        )
        if key != "mert_l6":
            result[key + "_lengths"] = [len(v) for v in native]
            result[key + "_origins"] = [m[key + "_origin"] for m in macros]
    return result


def song_paths(row, manifest, cfg):
    audio = manifest.parent / row["audio"]
    cache = (
        Path(cfg["macro_dir"]) if cfg.get("macro_dir") else manifest.parent / "macro"
    )
    models = (cfg["mert_model"], cfg["muq_model"])
    if not cfg.get("macro_dir") and models != (DEFAULT_MERT, DEFAULT_MUQ):
        producer = hashlib.sha256(json.dumps(models).encode()).hexdigest()[:16]
        cache = cache / producer
    identity = cache / (row.get("audio_id", row["song"]) + ".pt")
    alias = cache / (row["song"] + ".pt")
    return audio, alias if not identity.is_file() and alias.is_file() else identity


def pack_arrays(pack, end_tick):
    fields = pack.pop("fields")
    pack.pop("meta")
    pack["field_len"] = np.array([len(f) for f in fields], np.int32)
    pack["field_tokens"] = np.array([v for f in fields for v in f], np.uint8)
    masks, beats = [], []
    ends = np.zeros((len(fields), 41), np.uint8)
    for i, ((slot, t, kind), tokens) in enumerate(zip(pack["tails"], fields)):
        pad = int(pack["slots"][slot, 1])
        note = read_tail(t, pad, 0, kind, tokens)
        ends[i] = end_pads(note)
        allowed, selected = field_plan(kind, t, end_tick, tokens)
        masks.append(allowed)
        beats.append(selected)
    pack["field_beats"] = np.concatenate(beats) if beats else np.empty(0, np.int32)
    pack["tail_ends"] = ends
    mask = np.concatenate(masks) if masks else np.empty((0, FIELD_COUNT), bool)
    pack["field_allowed"] = np.packbits(mask, axis=-1)
    for key in ("hazard", "tailhaz"):
        pack[key] = np.packbits(pack[key], axis=-1)
    feedback = pack.pop("feedback").reshape(-1, 41, PAD_FEEDBACK)
    rows, pads = np.nonzero(feedback.any(-1))
    pack["feedback_indices"] = np.column_stack((rows, pads)).astype(np.int32)
    pack["feedback_values"] = feedback[rows, pads]
    return pack


def unpack_chart(arrays):
    for key in ("hazard", "tailhaz"):
        arrays[key] = np.unpackbits(
            arrays[key], axis=-1, count=len(HAZ_OFFSETS) * HAZ_DIM
        )
    feedback = np.zeros((len(arrays["steps"]), 41, PAD_FEEDBACK), np.float32)
    rows, pads = arrays.pop("feedback_indices").T
    feedback[rows, pads] = arrays.pop("feedback_values")
    arrays["feedback"] = feedback.reshape(len(feedback), -1)


def append_arrays(stream, arrays):
    index = {}
    for name, value in arrays.items():
        padding = (-stream.tell()) % max(4, value.itemsize)
        stream.write(bytes(padding))
        index[name] = [stream.tell(), value.dtype.str, value.shape]
        stream.write(np.ascontiguousarray(value).tobytes())
    return index


def bake_one(args):
    source, out, row = args
    with gzip.open(source / row["labels"], "rt", encoding="utf8") as f:
        doc = json.load(f)
    meta, mom = doc["metadata"], doc["moments"]
    grid = BeatGrid(meta)
    last_onset = mom[-1][0] if mom else 0
    last_endpoint = max(
        (
            stop
            for _, notes in mom
            for n in notes
            if n[2]
            for stop in (n[4], *(branch[1] for branch in n[6]))
        ),
        default=None,
    )
    last = max(last_onset, last_endpoint or 0)
    audio_end = int(np.floor(grid.second_to_beat(row["duration"]) * 96))
    # Only the physical audio length conditions the network. The annotation
    # horizon also permits exact supervision of releases beyond the audio.
    end_tick = max(last, audio_end)
    views = []
    tmp = out / (row["id"].replace(":", "-") + ".arena.tmp")
    with tmp.open("wb") as f:
        for mirrored in range(2 if row["split"] == "fit" else 1):
            notes = [(t, mirror_notes(ns)) for t, ns in mom] if mirrored else mom
            views.append(
                append_arrays(
                    f,
                    pack_arrays(
                        chart_pack(
                            notes,
                            grid,
                            meta,
                            audio_end,
                            song_id=int(row["song"]),
                            mirrored=mirrored,
                            field_limit=end_tick,
                        ),
                        end_tick,
                    ),
                )
            )
    row.update(
        meta=meta,
        end_tick=audio_end,
        field_end_tick=end_tick,
        end_second=grid.second(last),
        views=views,
        last_endpoint=last_endpoint,
        last_onset=last_onset,
        last_tail=max((t for t, ns in mom if any(n[2] for n in ns)), default=None),
    )
    return row, tmp


def bake_audio(
    records,
    manifest,
    output,
    device,
    macro_dir,
    audio_groups,
    windows_per_song,
    mert,
    muq,
):
    by_audio = {}
    for r in records:
        if r["split"] == "fit":
            by_audio.setdefault(r.get("audio_id", r["song"]), r)
    pairs = list(by_audio.items())
    audio = {}
    paths_cfg = dict(macro_dir=macro_dir, mert_model=mert, muq_model=muq)
    for start in range(0, len(pairs), audio_groups):
        group = pairs[start : start + audio_groups]
        feat = audio_features_batch(
            [song_paths(r, manifest, paths_cfg) for _, r in group],
            torch.device(device),
            mert,
            muq,
            windows_per_song,
        )
        for i, (key, _) in enumerate(group):
            audio[key] = write_audio(output / "audio" / key, feat, i)
        del feat
    return audio


def bake(
    manifest,
    output,
    workers=12,
    songs=None,
    *,
    device="cuda",
    macro_dir=None,
    audio_groups=4,
    windows_per_song=4,
    margin=2.5,
    mert=DEFAULT_MERT,
    muq=DEFAULT_MUQ,
    restore_audio_only=False,
):
    """Build a new pack, or restore only its audio arrays from the bound manifest."""
    import shutil

    source = manifest.parent
    doc = json.loads(manifest.read_text(encoding="utf8"))
    if restore_audio_only:
        bake_audio(
            doc["records"],
            manifest,
            output,
            device,
            macro_dir,
            audio_groups,
            windows_per_song,
            *doc["audio_models"],
        )
        return
    if source.resolve() == output.resolve():
        output = source / "packed-moment205-v4"
    output.mkdir(parents=True, exist_ok=True)
    records = [
        r for r in doc["records"] if official(r) and (not songs or r["song"] in songs)
    ]
    audio = bake_audio(
        records,
        manifest,
        output,
        device,
        macro_dir,
        audio_groups,
        windows_per_song,
        mert,
        muq,
    )
    for row in records:
        # The packed manifest must resolve the original recording without PREPARED.
        original = (source / row["audio"]).absolute()
        row["audio"] = (
            Path(os.path.relpath(original, output))
            if original.drive == output.absolute().drive
            else original
        ).as_posix()
        info = audio.get(row.get("audio_id", row["song"]))
        row["duration"] = info["samples"] / 44100 if info else 0
    jobs = ((source, output, r) for r in records)
    with (
        (output / "charts.bin.tmp").open("wb") as arena,
        ProcessPoolExecutor(max_workers=workers) as pool,
    ):
        records = []
        for row, tmp in pool.map(bake_one, jobs, chunksize=1):
            arena.write(bytes((-arena.tell()) % 4))
            row["arena_offset"] = arena.tell()
            with tmp.open("rb") as part:
                shutil.copyfileobj(part, arena)
            tmp.unlink()
            records.append(row)
    (output / "charts.bin.tmp").replace(output / "charts.bin")
    result = dict(
        schema="moment205.packed.v4",
        records=records,
        audio=audio,
        margin=margin,
        audio_models=[mert, muq],
    )
    tmp = output / "manifest.json.tmp"
    tmp.write_text(
        json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf8"
    )
    tmp.replace(output / "manifest.json")


class ChartGroups(Dataset):
    def __init__(self, manifest, seed, config):
        from importlib.resources import files

        self.folder = manifest.parent
        doc = json.loads(manifest.read_text(encoding="utf8"))
        self.audio_models = doc["audio_models"]
        self.audio, self.margin = doc["audio"], doc["margin"]
        fit = {
            r["id"]
            for r in json.loads(
                files("maidata.multitask")
                .joinpath("assets/spine-split.json")
                .read_text("utf8")
            )
            if r["split"] == "fit"
        }
        groups = defaultdict(list)
        for r in doc["records"]:
            if official(r) and r["id"] in fit and r["split"] == "fit":
                r["condition"] = condition(r["meta"], config.designer_vocab)
                groups[r.get("audio_id", r["song"])].append(r)
        self.groups = list(groups.items())
        # Keep every actual probe and the convolution/interpolation halo. A
        # chart-derived cut inside that support would leak its ending and make
        # the same query on the full inference bank produce different features.
        self.until = {}
        halo = (config.adapter_layers * (config.adapter_kernel // 2) + 2) / 125
        tail_radius = config.tail_audio_offsets[-1] / 125
        for key, rows in self.groups:
            cut = max(r["end_second"] for r in rows) + self.margin
            for r in rows:
                grid = BeatGrid(r["meta"])
                if r["last_endpoint"] is not None:
                    beat_end = r["last_endpoint"] // 96 * 96 + 95
                    cut = max(cut, grid.second(beat_end) + halo)
                onset = max(r["last_onset"], r["end_tick"])
                cut = max(
                    cut,
                    grid.second(onset + 96 * max([0, *config.beat_probes])) + halo,
                    grid.second(onset) + max([0, *config.audio_offsets]) / 125 + halo,
                )
                if r["last_tail"] is not None:
                    cut = max(cut, grid.second(r["last_tail"]) + tail_radius + halo)
            self.until[key] = cut
        self.arena = None
        self.seed = seed

    def __len__(self):
        return len(self.groups)

    def __getitems__(self, items):
        if self.arena is None:
            self.arena = np.memmap(self.folder / "charts.bin", mode="c", dtype=np.uint8)
        if len(items) == 1:
            return [self[items[0]]]
        return list(io_pool(len(items)).map(self.__getitem__, items))

    def __getitem__(self, item):
        epoch, index = item
        key, rows = self.groups[index]
        if self.arena is None:
            self.arena = np.memmap(self.folder / "charts.bin", mode="c", dtype=np.uint8)
        charts = []
        for r in rows:
            seed = int(r["song"]) + epoch
            view = int(seed % 3 == 0)
            arrays = {
                k: np.ndarray(
                    shape,
                    np.dtype(dtype),
                    buffer=self.arena,
                    offset=r["arena_offset"] + pos,
                )
                for k, (pos, dtype, shape) in r["views"][view].items()
            }
            unpack_chart(arrays)
            shuffle_pieces(arrays, self.seed, epoch, int(r["song"]), view)
            charts.append(
                dict(
                    arrays,
                    meta=r["meta"],
                    end_tick=r["end_tick"],
                    field_end_tick=r["field_end_tick"],
                    moments_count=r["moments"],
                    condition=r["condition"],
                )
            )
        return read_audio(
            self.folder / "audio" / key, self.audio[key], self.until[key]
        ), charts


def shuffle_pieces(arrays, seed, epoch, song, mirrored):
    # Locate only multi-piece subgroups; copy the mapped labels before changing them.
    from .pieces import MIRROR_WORD

    pieces = arrays["pieces"]
    multi = np.flatnonzero((pieces[:, 4] < SIZE) & (pieces[:, 2] != 0))
    if not len(multi):
        return
    pieces = arrays["pieces"] = pieces.copy()
    for i in multi:
        if pieces[i - 1, 2] != 0:
            continue
        start = i - 1
        stop = i + 1
        while pieces[stop, 4] < SIZE:
            stop += 1
        row, _, _, group, _ = pieces[start]
        ids = sorted(
            pieces[start:stop, 4], key=lambda w: MIRROR_WORD[w] if mirrored else w
        )
        np.random.default_rng([seed, epoch, song, int(row), int(group)]).shuffle(ids)
        used = 0
        for j, w in enumerate(ids, start):
            pieces[j, 2], pieces[j, 4] = used, w
            used |= pad_mask(WORDS[w])


class EpochOrder:
    def __init__(self, dataset):
        self.dataset = dataset
        self.epoch = 1

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        order = np.random.default_rng(self.dataset.seed + self.epoch).permutation(
            len(self)
        )
        return iter((self.epoch, i) for i in order)


def collate(groups, *, config):
    songs, grouped = zip(*groups)
    charts = [chart for group in grouped for chart in group]
    group_ids = torch.tensor([i for i, g in enumerate(grouped) for _ in g])

    def pad(seq):
        return torch.nn.utils.rnn.pad_sequence(seq, batch_first=True)

    def padded(key, subset):
        return pad([torch.as_tensor(c[key]) for c in subset])

    def flat(key):
        return torch.from_numpy(np.concatenate([c[key] for c in charts]))

    def owners(key):
        return torch.repeat_interleave(
            torch.arange(len(charts)), torch.tensor([len(c[key]) for c in charts])
        )

    for c in charts:
        c.update(chart_clocks(c, config))
    labels = {
        k: padded(k, charts)
        for k in (
            "steps",
            "spine_clock",
            "word_clock",
            "feedback",
            "hazard",
            "footprint",
        )
    }
    sizes = [len(c["steps"]) for c in charts]
    u = flat("slots")
    starts = np.cumsum([0] + [len(c["slots"]) for c in charts[:-1]])
    length = torch.tensor(sizes)
    labels.update(
        conditions=torch.tensor([c["condition"][0] for c in charts]),
        charter=torch.tensor([c["condition"][1] for c in charts]),
        step_lengths=length,
        valid_steps=(torch.arange(max(sizes)) < length[:, None]).nonzero(as_tuple=True),
        slots=u,
        slot_chart=owners("slots"),
        detail=flat("detail"),
        pieces=flat("pieces"),
        slot_haz=flat("slot_haz"),
        piece_chart=owners("pieces"),
        committed_pads=torch.from_numpy(
            np.unpackbits(flat("committed_pads").numpy(), axis=-1, count=246).reshape(
                -1, 41, 6
            )
        ),
        repeat_room=flat("repeat_room"),
        prior_non_slide=flat("prior_non_slide"),
        limits=torch.tensor(
            [max(c["end_tick"], int(c["steps"][-1, 2])) for c in charts]
        ),
        audio_limits=torch.tensor(
            [c.get("field_end_tick", c["end_tick"]) for c in charts]
        ),
        moments=sum(c["moments_count"] for c in charts),
        present_slots=len(u),
    )
    labels["geometry_pieces"] = int((labels["pieces"][:, 4] < SIZE).sum())
    vchart, vrow = labels["valid_steps"]
    labels["real_pos"] = (labels["steps"][vchart, vrow, 3] < BEAT).nonzero().flatten()
    labels["spine_layout"] = history_layout(
        labels["steps"][..., 2],
        labels["steps"][..., 2],
        sizes,
        config.history_beats,
        config.history_query_block,
    )
    rows = torch.arange(len(u))
    repeats = torch.nn.functional.pad(u[:-1, 3] == 1, (1, 0))[: len(u)]
    labels["occurrence"] = rows - rows.masked_fill(repeats, 0).cummax(0).values
    active_ids = [i for i, c in enumerate(charts) if len(c["tails"])]
    active = [charts[i] for i in active_ids]
    labels["tail_slots"] = sum(len(c["tails"]) for c in active)
    if active:
        tails = padded("tails", active)
        lengths = torch.tensor([len(c["tails"]) for c in active])
        lens = flat("field_len").long()
        fm = torch.arange(int(lens.max())) < lens[:, None]
        target = torch.zeros(fm.shape, dtype=torch.long)
        target[fm] = flat("field_tokens").long()
        # Each selected beat is sampled once per chart, even if multiple branches end there.
        counts = [len(c["endpoint_seconds"]) for c in charts]
        offsets = np.cumsum([0, *counts[:-1]])
        labels.update(
            endpoint_seconds=torch.cat([c["endpoint_seconds"] for c in charts]),
            endpoint_chart=torch.repeat_interleave(
                torch.arange(len(charts)), torch.tensor(counts)
            ),
            endpoint_index=torch.from_numpy(
                np.concatenate(
                    [c["endpoint_index"] + off for c, off in zip(charts, offsets)]
                )
            ),
            endpoint_rows=(flat("field_beats") >= 0).nonzero().flatten(),
            tail_chart=torch.tensor(active_ids),
            tail_ends=padded("tail_ends", active),
            tail_layout=history_layout(
                tails[..., 1],
                tails[..., 1],
                lengths.tolist(),
                config.history_beats,
                config.history_query_block,
                strict=True,
                release_ticks=padded("tail_release", active),
            ),
            tail_pos=tails[..., 0].long()
            + torch.from_numpy(starts[active_ids])[:, None],
            tail_tick=tails[..., 1],
            tail_clock=padded("tail_clock", active),
            tail_kind=tails[..., 2].long(),
            tailhaz=padded("tailhaz", active),
            tail_valid=(torch.arange(tails.shape[1]) < lengths[:, None]).nonzero(
                as_tuple=True
            ),
            field_target=target,
            field_valid=fm.nonzero(as_tuple=True),
            field_allowed=torch.from_numpy(
                np.unpackbits(flat("field_allowed").numpy(), axis=-1, count=FIELD_COUNT)
            ).bool(),
        )
    return pad_audio(songs), labels, group_ids


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--restore-audio-only",
        action="store_true",
        help="Restore audio arrays from a packed manifest without rewriting chart data or bindings",
    )
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--macro-dir", type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--audio-groups", type=int, default=4)
    p.add_argument("--margin", type=float, default=2.5)
    p.add_argument("--windows-per-song", type=int, default=4)
    p.add_argument("--mert", default=DEFAULT_MERT)
    p.add_argument("--muq", default=DEFAULT_MUQ)
    p.add_argument("--songs", nargs="*")
    bake(**vars(p.parse_args()))
