"""Physical clocks and compact, memory-mapped input to the chart decoder."""

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import math
import os

import numpy as np
import torch

from maidata.multitask.grid import BeatGrid
from maidata.designer import DESIGNER_VOCAB, designer_id, designer_name


def condition(meta, vocabulary=DESIGNER_VOCAB):
    return [
        float(meta.get("level") or 0),
        float(meta.get("difficulty") or 0),
    ], designer_id(designer_name(meta), vocabulary)


CLOCK_DIM = 26


@lru_cache(maxsize=64)
def constants(values, device):
    return torch.tensor(values, device=device, dtype=torch.float32)


def clocks(grid, ticks, end_tick, probes=(), seconds=None):
    t = torch.as_tensor(ticks, dtype=torch.float32)
    beats = t / 96
    angles = (2 * math.pi) * torch.cat(
        (
            (t.remainder(96) / 96)[..., None] * constants((1, 2, 3, 4, 6, 8), t.device),
            (beats.floor().remainder(16) / 16)[..., None]
            * constants((1, 2, 4, 8), t.device),
        ),
        -1,
    )
    tempo = grid.tempo_info(beats)
    duration = max(1, end_tick)
    phase = torch.cat(
        (
            angles.sin(),
            angles.cos(),
            torch.stack(
                (
                    (tempo["bpm"] / 120).log2(),
                    (tempo["bpm"] / tempo["base_bpm"]).log2(),
                    (tempo["slope"] / 60).tanh(),
                    tempo["is_stop"],
                    t / duration,
                    torch.full_like(t, math.log2(duration / (96 * 256))),
                ),
                -1,
            ),
        ),
        -1,
    )
    if seconds is None:
        offsets = constants((0, *probes), t.device)
        seconds = grid.beat_to_second_tensor(beats[..., None] + offsets)
    else:
        seconds = torch.as_tensor(seconds, dtype=t.dtype)[..., None]
        if probes:
            extra = grid.beat_to_second_tensor(
                beats[..., None] + constants(tuple(probes), t.device)
            )
            seconds = torch.cat((seconds, extra), -1)
    return torch.cat((phase, seconds), -1)


RATES = {"mel": 125, "mert": 75, "mert_l6": 75, "muq": 25}


def write_audio(folder, features, song):
    folder.mkdir(parents=True, exist_ok=True)
    result = {
        "samples": int(features["sample_lengths"][song]),
        "frames": {},
        "origins": {},
    }
    for key in RATES:
        native = "mert" if key == "mert_l6" else key
        n = int(features[native + "_lengths"][song])
        origin = 0.004 if key == "mel" else float(features[native + "_origins"][song])
        result["origins"][key] = origin
        result["frames"][key] = n
        value = (
            features[key][song, :n].to(dtype=torch.bfloat16, device="cpu").contiguous()
        )
        np.save(folder / (key + ".npy"), value.view(torch.uint16).numpy())
    return result


def read_audio(folder, info, until):
    out = {"sample_lengths": info["samples"]}
    for key, rate in RATES.items():
        origin = info["origins"][key]
        n = min(info["frames"][key], max(1, math.ceil((until - origin) * rate) + 2))
        out[key] = torch.from_numpy(
            np.load(folder / (key + ".npy"), mmap_mode="c")[:n]
        ).view(torch.bfloat16)
        if key != "mert_l6":
            out[key + "_lengths"] = n
            if key != "mel":
                out[key + "_origins"] = origin
    return out


@lru_cache(maxsize=1)
def _worker_pool(pid, size):
    # One persistent pool per loader process; mmap page faults occur during copy.
    return ThreadPoolExecutor(max_workers=size)


def io_pool(size):
    return _worker_pool(os.getpid(), size)


def pad_audio(songs, rates=RATES):
    out = {k: torch.tensor([s[k] for s in songs]) for k in songs[0] if k not in rates}
    for k in rates:
        value = songs[0][k]
        shape = (len(songs), max(len(s[k]) for s in songs), value.shape[1])
        if torch.utils.data.get_worker_info() is not None:
            # As in PyTorch's default tensor collator, write directly into IPC storage.
            storage = value._typed_storage()._new_shared(
                math.prod(shape), device=value.device
            )
            out[k] = value.new(storage).resize_(shape)
        else:
            out[k] = value.new_empty(shape)

    def copy_song(i):
        for k in rates:
            v = songs[i][k]
            out[k][i, : len(v)].copy_(v)
            out[k][i, len(v) :].zero_()

    if len(songs) == 1:
        copy_song(0)
    else:
        list(io_pool(len(songs)).map(copy_song, range(len(songs))))
    return out


def endpoint_seconds(grid, beats):
    """All candidate phases of selected integer beats, including BPM changes inside a beat."""
    beats = torch.as_tensor(beats)
    phases = constants(tuple(i / 96 for i in range(96)), beats.device)
    return grid.beat_to_second_tensor(torch.as_tensor(beats)[..., None] + phases)


def chart_clocks(chart, cfg):
    grid = BeatGrid(chart["meta"])
    steps = chart["steps"]
    end = chart["end_tick"]
    first = steps[:, 2] // 96 * 96 + np.minimum(steps[:, 3], 95)
    beat = chart["field_beats"]
    unique, inverse = np.unique(beat[beat >= 0], return_inverse=True)
    return {
        "endpoint_seconds": endpoint_seconds(grid, unique),
        "endpoint_index": inverse,
        "spine_clock": clocks(
            grid, steps[:, 2], end, cfg.beat_probes, chart["step_seconds"]
        ),
        "word_clock": clocks(grid, first, end),
        "tail_clock": clocks(
            grid, chart["tails"][:, 1], end, cfg.tail_probes, chart["tail_seconds"]
        ),
    }
