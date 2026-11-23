"""Offline producer for YMT3+ (8ms @ 512d) and MuQ Layer 10 (40ms @ 1024d) BF16 grids.

python -m maidata.preparation.produce_ymt3_muq --catalog OLD/catalog.json
    --train-plan TRAIN/plan.json --output E:/01YD/0wd-maidata/ymt3-muq-audio
"""

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .compactspec import bf16_bits, read_stereo
from .produce import jsonl, read_json, save_arrays, token, write_json
from .timing import TempoMap
from .ymt3_muq import (
    MUQ_DIM,
    RATIO_40_TO_8,
    RECIPE,
    YMT3_DIM,
    YMT3T5Encoder,
    extract_muq_windowed,
    extract_ymt3_windowed,
    resample_mono,
)


class _MetadataOnly:
    """Inert stand-in for unused official Lightning hyperparameter objects."""


def _read_ymt3_checkpoint(ckpt_path: str | Path) -> dict:
    metadata = [
        "utils.task_manager.TaskManager",
        "utils.note_event_dataclasses.Event",
        "utils.note_event_dataclasses.EventRange",
        "utils.tokenizer.NoteEventTokenizer",
        "utils.event_codec.FastCodec",
    ]
    safe = [(_MetadataOnly, name) for name in metadata]
    safe += [
        (np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
        (np._core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
        np.ndarray,
        np.dtype,
        type(np.dtype("float64")),
        type(np.dtype("float32")),
        type(np.dtype("int64")),
    ]
    with torch.serialization.safe_globals(safe):
        return torch.load(Path(ckpt_path), map_location="cpu", weights_only=True)


def load_ymt3_model(checkpoint_path: str | Path | None = None, *, device: str = "cpu"):
    model = YMT3T5Encoder()
    if checkpoint_path is not None:
        payload = _read_ymt3_checkpoint(checkpoint_path)
        state = payload.get("state_dict", payload.get("model_state", payload))
        remapped = {}
        for k, v in state.items():
            # The official YMT3+ T5 release wraps its T5Stack as
            # encoder.encoder and names the fixed sinusoid additive_pe.
            if k.startswith("encoder.encoder."):
                k = k[len("encoder.encoder.") :]
            elif k.startswith("encoder."):
                k = k[len("encoder.") :]
            if k == "additive_pe.weight":
                k = "embed_positions.weight"
            if k.startswith(("block.", "final_layer_norm.", "embed_positions.")):
                remapped[k] = v
        if "embed_positions.weight" in remapped and remapped["embed_positions.weight"].shape != model.embed_positions.weight.shape:
            n_pos, dim = remapped["embed_positions.weight"].shape
            model.embed_positions = type(model.embed_positions)(n_pos, dim)
        elif "embed_positions.weight" not in remapped:
            remapped["embed_positions.weight"] = model.embed_positions.weight.detach().clone()
        model.load_state_dict(remapped, strict=True)
    return model.to(device).eval().requires_grad_(False)


def load_muq_model(model_name_or_path: str = "OpenMuQ/MuQ-large-msd-iter", *, layer: int = 10, device: str = "cpu"):
    from muq import MuQ

    muq = MuQ.from_pretrained(model_name_or_path).to(device).eval().requires_grad_(False)

    def forward(wave_batch: torch.Tensor) -> torch.Tensor:
        out = muq(wave_batch, output_hidden_states=True)
        return out.hidden_states[layer]

    return forward


def compute_silence_vectors(ymt3_model, muq_model, *, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        zero_16k = torch.zeros((1, 32768), dtype=torch.float32, device=device)
        y_sil = extract_ymt3_windowed(zero_16k[0], ymt3_model, duration_seconds=2.048, device=device)
        silence_ymt3 = y_sil[min(128, len(y_sil) - 1)].detach().cpu().to(torch.float32).numpy()

        zero_24k = torch.zeros((1, 720000), dtype=torch.float32, device=device)
        m_sil = extract_muq_windowed(zero_24k[0], muq_model, n_40ms=750, duration_seconds=30.0, device=device)
        silence_muq = m_sil[min(375, len(m_sil) - 1)].detach().cpu().to(torch.float32).numpy()
    return silence_ymt3.astype(np.float32), silence_muq.astype(np.float32)


def produce_asset(output, asset_id, asset, variants, stats, ymt3_model, muq_model, *, device):
    """Produce one independent asset and its existing grid metadata."""
    output = Path(output)
    silence_ymt3 = np.asarray(stats["silence_ymt3"], dtype=np.float32)
    silence_muq = np.asarray(stats["silence_muq"], dtype=np.float32)
    y, sr = read_stereo(asset["wav_path"])
    duration = len(y) / sr
    wave_16k = resample_mono(y, sr, 16000)
    wave_24k = resample_mono(y, sr, 24000)

    ymt3_feat = extract_ymt3_windowed(wave_16k, ymt3_model, duration_seconds=duration, device=device)
    n_8ms = ymt3_feat.shape[0]
    n_40ms = max(1, math.ceil(n_8ms / RATIO_40_TO_8))
    muq_feat = extract_muq_windowed(wave_24k, muq_model, n_40ms=n_40ms, duration_seconds=duration, device=device)

    ymt3_bits = bf16_bits(ymt3_feat.numpy())
    muq_bits = bf16_bits(muq_feat.numpy())

    asset_rel = f"assets/{asset_id}.npz"
    asset_path = output / asset_rel
    asset_arrays = dict(
        ymt3_frames=ymt3_bits,
        muq_frames=muq_bits,
        silence_ymt3=silence_ymt3,
        silence_muq=silence_muq,
    )
    asset_meta = save_arrays(asset_path, asset_arrays, grid=True)
    ymt3_muq_ref = dict(
        asset_path=asset_rel,
        ymt3_shape=list(ymt3_bits.shape),
        muq_shape=list(muq_bits.shape),
        sha256=asset_meta["sha256"],
    )

    grid_records = []
    for variant_id, variant in variants:
        path = output / "grids" / f"{variant_id}.npz"
        key = token(dict(asset=asset_id, timing=variant["timing"], recipe=RECIPE, stats=stats))
        meta_path = path.with_suffix(".json")
        tempo = TempoMap(variant["timing"])
        first_beat = min(0, math.floor(float(tempo.to_beats(0.0))))
        grid = dict(
            silence_ymt3=silence_ymt3,
            silence_muq=silence_muq,
            first_beat=np.array(int(first_beat), dtype=np.int64),
            duration_seconds=np.array(float(duration), dtype=np.float64),
            timing_json=np.array(json.dumps(variant["timing"], sort_keys=True)),
        )
        meta = dict(
            input_key=key,
            variant=variant_id,
            asset=asset_id,
            representation=RECIPE["representation"],
            ymt3_muq_reference=ymt3_muq_ref,
            **save_arrays(path, grid, grid=False),
        )
        write_json(meta_path, meta)
        grid_records.append(meta)
    return dict(asset=asset_id, asset_bytes=asset_meta["bytes"], grids=grid_records)


def build(
    catalog_path: str | Path,
    train_plan: str | Path,
    output: str | Path,
    *,
    wav_root: str | Path | None = None,
    limit: int | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ymt3_model=None,
    muq_model=None,
    ymt3_checkpoint: str | Path | None = None,
    muq_name: str = "OpenMuQ/MuQ-large-msd-iter",
) -> dict:
    catalog_path, output = Path(catalog_path), Path(output)
    original = read_json(catalog_path)
    if output.resolve() == catalog_path.parent.resolve():
        raise ValueError("YMT3+MuQ output must be independent of the source catalog")
    catalog = deepcopy(original)
    if wav_root is not None:
        rows = {r["output_sha256"]: r for r in jsonl(Path(wav_root) / "audio-manifest.jsonl")}
        for key, asset in catalog["assets"].items():
            asset["wav_path"] = str(Path(wav_root) / rows[key]["output_relative_path"])

    plan_path = Path(train_plan)
    if plan_path.is_dir():
        plan_path = plan_path / "plan.json"
    plan = read_json(plan_path)

    if ymt3_model is None:
        ymt3_model = load_ymt3_model(ymt3_checkpoint, device=device)
    if muq_model is None:
        muq_model = load_muq_model(muq_name, device=device)

    started = time.perf_counter()
    silence_ymt3, silence_muq = compute_silence_vectors(ymt3_model, muq_model, device=device)
    stats = dict(
        recipe=RECIPE,
        silence_ymt3=silence_ymt3.tolist(),
        silence_muq=silence_muq.tolist(),
    )

    grouped = {key: [] for key in catalog["assets"]}
    for key, variant in catalog["variants"].items():
        grouped[variant["asset"]].append((key, variant))

    assets = list(catalog["assets"].items())
    if limit is not None:
        assets = assets[:limit]

    for key in (
        "mel_recipe",
        "aux_recipe",
        "aux_keys",
        "old_full_reuse",
        "pilot_reuse",
        "fixed_stats_path",
        "fixed_stats_sha256",
    ):
        catalog.pop(key, None)
    for variant in catalog["variants"].values():
        variant.pop("reused_mel", None)

    catalog.update(
        spec_recipe=RECIPE,
        stats=stats,
        stats_identity=token(stats),
        source_catalog=str(catalog_path.resolve()),
        statistics_train_plan=str(plan_path.resolve()),
    )
    write_json(output / "catalog.json", catalog)
    write_json(output / "stats.json", stats)

    records = [
        produce_asset(output, asset_id, asset, grouped[asset_id], stats,
                      ymt3_model, muq_model, device=device)
        for asset_id, asset in assets
    ]

    summary = dict(
        complete=limit is None,
        assets=len(records),
        variants=sum(len(r["grids"]) for r in records),
        output_bytes=sum(r["asset_bytes"] + sum(g["bytes"] for g in r["grids"]) for r in records),
        wall_seconds=time.perf_counter() - started,
        representation=RECIPE["representation"],
    )
    write_json(output / ("sample-summary.json" if limit is not None else "production-summary.json"), summary)
    write_json(output / "production.json", records)
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Produce independent YMT3+MuQ BF16 audio grids.")
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--train-plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--wav-root", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ymt3-checkpoint", type=Path)
    parser.add_argument("--muq-model", default="OpenMuQ/MuQ-large-msd-iter")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            build(
                args.catalog,
                args.train_plan,
                args.output,
                wav_root=args.wav_root,
                limit=args.limit,
                device=args.device,
                ymt3_checkpoint=args.ymt3_checkpoint,
                muq_name=args.muq_model,
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
