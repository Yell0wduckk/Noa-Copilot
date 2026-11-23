"""Train the dedicated official maimai chart model; checkpoints represent complete epochs."""

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from functools import partial
from torch.utils._pytree import tree_map_only
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from maidata.joint_training.checkpoint import rng_state, restore_rng, source_identity
from . import ARCHITECTURE
from .data import (
    ChartGroups,
    collate,
    EpochOrder,
)
from .model import SpineConfig, SpineModel
from .tails import HAZ_OFFSETS

DEFAULTS = dict(
    seed=20261111,
    workers=12,
    audio_groups_per_batch=4,
    learning_rate=0.0006,
    warmup_steps=200,
    scheduler_epochs=120,
    stop_after_epochs=75,
    output="artifacts/moment205-v4-designer21",
    manifest="artifacts/packed-moment205-v4/manifest.json",
    device="cuda",
    model=asdict(SpineConfig()),
    loss_weights={k: 1.0 for k in ("time", "piece", "repeat", "type", "tail", "flags")},
    bf16=True,
    betas=[0.9, 0.987],
    weight_decay=0.01,
    grad_clip=1.0,
    min_lr_ratio=0.05,
    prefetch_factor=2,
    log_interval=10,
    cost_warmup=0,
    cost_steps=1,
)


def binding(manifest):
    from importlib.resources import files

    digest = hashlib.sha256()
    for path in (
        manifest,
        files("maidata.multitask").joinpath("assets/moment_bpe_v4.json"),
    ):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run(cfg, manifest, resume=None, cost=False):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    device = torch.device(cfg["device"])
    model_config = SpineConfig(**cfg["model"])
    dataset = ChartGroups(manifest, cfg["seed"], model_config)
    cfg = dict(
        cfg,
        model=asdict(model_config),
        audio_models=dataset.audio_models,
    )
    model = SpineModel(model_config).to(device)
    decay_ids = {
        id(p)
        for module in model.modules()
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.GRU))
        for name, p in module.named_parameters(recurse=False)
        if name.startswith("weight") and p.ndim >= 2
    }
    params = [
        {
            "params": [p for p in model.parameters() if (id(p) in decay_ids) == decay],
            "weight_decay": cfg["weight_decay"] if decay else 0.0,
        }
        for decay in (True, False)
    ]
    opt = torch.optim.AdamW(
        params, lr=cfg["learning_rate"], betas=cfg["betas"], fused=device.type == "cuda"
    )
    dest = Path(cfg["output"])
    dest.mkdir(parents=True, exist_ok=True)
    batch = cfg["audio_groups_per_batch"]
    updates = math.ceil(len(dataset) / batch)
    warm = cfg["warmup_steps"]
    horizon = updates * cfg["scheduler_epochs"]

    def factor(step):
        if step < warm:
            return (step + 1) / max(1, warm)
        t = min(1, (step - warm) / max(1, horizon - warm))
        floor = cfg["min_lr_ratio"]
        return floor + (1 - floor) * (1 + math.cos(math.pi * t)) / 2

    sched = torch.optim.lr_scheduler.LambdaLR(opt, factor)
    signature = binding(manifest)
    first = 1
    step = 0
    recipe = {
        k: cfg[k]
        for k in (
            "seed",
            "audio_groups_per_batch",
            "learning_rate",
            "warmup_steps",
            "scheduler_epochs",
            "model",
            "loss_weights",
            "bf16",
            "betas",
            "weight_decay",
            "grad_clip",
            "min_lr_ratio",
        )
    }
    generator = torch.Generator().manual_seed(cfg["seed"])
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        if (state["architecture"], state["binding"], state["recipe"]) != (
            ARCHITECTURE,
            signature,
            recipe,
        ):
            raise ValueError(
                "Checkpoint architecture, training inputs or update recipe differ"
            )
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        first = state["epoch"] + 1
        step = state["step"]
        generator.set_state(state["loader_rng"])
        restore_rng(state["rng"])
        del state
    order = EpochOrder(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch,
        sampler=order,
        num_workers=cfg["workers"],
        collate_fn=partial(collate, config=model.config),
        pin_memory=device.type == "cuda",
        persistent_workers=bool(cfg["workers"]),
        generator=generator,
        prefetch_factor=cfg["prefetch_factor"] if cfg["workers"] else None,
    )
    measured = []
    cost_updates = 0
    stop = (
        first + math.ceil((cfg["cost_warmup"] + cfg["cost_steps"]) / updates) - 1
        if cost
        else cfg["stop_after_epochs"]
    )
    for epoch in range(first, stop + 1):
        order.epoch = epoch
        interval_start = time.perf_counter()
        interval_moments = interval_steps = 0
        completed = interval_start
        # Persistent workers are bootstrapped once in an uninterrupted run.
        # Recreate them with that same seed without consuming an extra epoch's RNG.
        if resume and epoch == first and cfg["workers"]:
            cursor = generator.get_state()
            generator.manual_seed(cfg["seed"])
            batches = iter(loader)
            generator.set_state(cursor)
        else:
            batches = iter(loader)
        for audio, labels, indices in batches:
            load_seconds = time.perf_counter() - completed
            diagnosing = cost and cost_updates >= cfg["cost_warmup"]
            if diagnosing and device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            before = time.perf_counter()
            audio = tree_map_only(
                torch.Tensor, lambda t: t.to(device, non_blocking=True), audio
            )
            labels = tree_map_only(
                torch.Tensor, lambda t: t.to(device, non_blocking=True), labels
            )
            indices = indices.to(device, non_blocking=True)
            if diagnosing and device.type == "cuda":
                torch.cuda.synchronize()
            h2d = time.perf_counter() - before
            opt.zero_grad(set_to_none=True)
            with torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
                enabled=device.type == "cuda" and cfg["bf16"],
            ):
                bank = model.fusion(audio)
                loss, terms, counts = model.loss(
                    labels, bank, indices, cfg["loss_weights"]
                )
            loss.backward()
            gradients = {}
            if diagnosing:
                for name, module in [
                    ("fusion", model.fusion),
                    ("spine", model.spine),
                    ("tail", model.tail),
                    ("flags", model.flags),
                ]:
                    norms = [
                        p.grad.detach().float().norm()
                        for p in module.parameters()
                        if p.grad is not None
                    ]
                    gradients[name] = (
                        torch.stack(norms).norm() if norms else loss.new_zeros(())
                    )
            if cfg["grad_clip"] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            sched.step()
            step += 1
            if cost:
                cost_updates += 1
                if not diagnosing:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    completed = time.perf_counter()
                    interval_start = completed
                    interval_steps = interval_moments = 0
                    continue
            interval_moments += counts["moments"]
            interval_steps += 1
            if not cost and step % cfg["log_interval"]:
                continue
            if device.type == "cuda":
                torch.cuda.synchronize()
            finished = time.perf_counter()
            elapsed = finished - (completed if cost else interval_start)
            metrics = dict(
                epoch=epoch,
                step=step,
                loss=float(loss.detach()),
                seconds=elapsed / interval_steps,
                moments_per_second=interval_moments / elapsed,
                **counts,
                audio_groups=len(audio["mel_lengths"]),
                charts=len(indices),
                nll_per_moment=float(terms["nll_per_moment"]),
                **{
                    k: float(terms[k])
                    for k in ("moment_loss", "upper_other_loss", "lower_loss")
                },
                mean_ce={
                    k: float(v) for k, v in terms.items() if k in cfg["loss_weights"]
                },
            )
            print(json.dumps(metrics), flush=True)
            interval_start = time.perf_counter()
            interval_moments = interval_steps = 0
            if cost:
                spine_rows, tail_rows = counts["steps"], counts["tail_slots"]
                spine_audio = len(model_config.audio_offsets) + len(
                    model_config.beat_probes
                )
                tail_audio = len(model_config.tail_audio_offsets) + len(
                    model_config.tail_probes
                )
                metrics.update(
                    local_audio=dict(
                        spine_rows=spine_rows,
                        spine_rectangular_rows=labels["steps"].shape[0]
                        * labels["steps"].shape[1],
                        tail_rows=tail_rows,
                        tail_rectangular_rows=labels["tail_tick"].numel()
                        if tail_rows
                        else 0,
                        spine_sample_positions=spine_rows * spine_audio,
                        tail_sample_positions=tail_rows * tail_audio,
                        spine_kv_per_row=spine_audio + 2 * len(HAZ_OFFSETS),
                        tail_kv_per_row=tail_audio + 2 * len(HAZ_OFFSETS),
                    ),
                    h2d_seconds=h2d,
                    loader_seconds=load_seconds,
                    update_seconds=finished - before,
                    peak_allocated=torch.cuda.max_memory_allocated()
                    if device.type == "cuda"
                    else None,
                    gradients={k: float(v) for k, v in gradients.items()},
                    losses={k: float(v.detach()) for k, v in terms.items()},
                    workers=cfg["workers"],
                    parameters=sum(p.numel() for p in model.parameters()),
                    device=str(device),
                )
                measured.append(metrics)
                if len(measured) >= cfg["cost_steps"]:
                    seconds = sum(r["seconds"] for r in measured)
                    summary = dict(
                        metrics,
                        updates=measured,
                        mean_seconds=seconds / len(measured),
                        steady_moments_per_second=sum(r["moments"] for r in measured)
                        / seconds,
                    )
                    (dest / "cost.json").write_text(json.dumps(summary, indent=2))
                    return
                completed = time.perf_counter()
        if cost:
            continue
        state = dict(
            architecture=ARCHITECTURE,
            source=source_identity(),
            config=cfg,
            recipe=recipe,
            binding=signature,
            epoch=epoch,
            step=step,
            model=model.state_dict(),
            optimizer=opt.state_dict(),
            scheduler=sched.state_dict(),
            rng=rng_state(),
            loader_rng=generator.get_state(),
        )
        names = ["last.ckpt"]
        if epoch % 5 == 0 or epoch == cfg["stop_after_epochs"]:
            names.append(f"epoch-{epoch:03}.ckpt")
        for name in names:
            tmp = dest / (name + ".tmp")
            torch.save(state, tmp)
            tmp.replace(dest / name)
        print(json.dumps(dict(completed_epoch=epoch, checkpoints=names)), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--output")
    p.add_argument("--device")
    p.add_argument("--resume", type=Path)
    p.add_argument("--cost-check", action="store_true")
    p.add_argument("--stop-after-epochs", type=int)
    p.add_argument("--scheduler-epochs", type=int)
    p.add_argument("--show-config", action="store_true")
    p.add_argument("--history-backend", choices=("auto", "sdpa"))
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=None)
    for key, kind in (
        ("audio_groups_per_batch", int),
        ("workers", int),
        ("learning_rate", float),
        ("seed", int),
        ("warmup_steps", int),
        ("grad_clip", float),
        ("weight_decay", float),
        ("min_lr_ratio", float),
        ("prefetch_factor", int),
        ("log_interval", int),
        ("cost_steps", int),
        ("cost_warmup", int),
    ):
        p.add_argument("--" + key.replace("_", "-"), type=kind)
    args = p.parse_args(argv)
    cfg = dict(DEFAULTS)
    if args.config:
        supplied = json.loads(args.config.read_text(encoding="utf8"))
        for key in ("model", "loss_weights"):
            if key in supplied:
                supplied[key] = {**DEFAULTS[key], **supplied[key]}
        cfg.update(supplied)
    for key in DEFAULTS:
        value = getattr(args, key, None)
        if value is not None and key != "manifest":
            cfg[key] = value
    if args.history_backend:
        cfg["model"] = {**cfg["model"], "history_backend": args.history_backend}
    if args.manifest:
        cfg["manifest"] = str(args.manifest)
    if args.show_config:
        print(json.dumps(dict(cfg, architecture=ARCHITECTURE), indent=2))
        return
    manifest = Path(cfg["manifest"])
    run(cfg, manifest.resolve(), args.resume, args.cost_check)
