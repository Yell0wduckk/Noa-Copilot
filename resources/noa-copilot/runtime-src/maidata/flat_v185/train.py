"""Two-stage maimai training with complete-epoch checkpoints."""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from maidata.joint_training.checkpoint import rng_state, restore_rng, source_identity
from maidata.spine.data import EpochOrder
from maidata.spine.train import DEFAULTS as V16
from . import ARCHITECTURE
from .codec import WIDTHS
from .data import ChartGroups, collate
from .model import FlatConfig
from maidata.model import build_model

DEFAULTS = {
    k: v
    for k, v in V16.items()
    if k not in ("model", "loss_weights", "output", "manifest", "stop_after_epochs")
}


def recipe(stage, supplied):
    cfg = {
        **DEFAULTS,
        "architecture": ARCHITECTURE,
        "stage": stage,
        "manifest": "data/packed-flat320-grid24/manifest.json",
        "output": f"artifacts/v18.5-flat320/stage{stage}",
        "stop_after_epochs": 15 if stage == 0 else 75,
        "stage0_checkpoint": None,
        "keep_stage0_self_attention": False,
        "stage1_from_scratch": False,
        "no_history_weight": 0.5,
        **supplied,
    }
    if cfg["stage1_from_scratch"] and "no_history_weight" not in supplied:
        cfg["no_history_weight"] = 1.0
    cfg["model"] = asdict(FlatConfig(**{**cfg.get("model", {}), "stage": stage}))
    if cfg["model"]["no_history_enabled"]:
        del cfg["model"]["no_history_enabled"]
    return cfg


def stage_output(cfg, resume):
    output = cfg["output"]
    if cfg.get("stage1_from_scratch") and output == "artifacts/v18.5-flat320/stage1":
        return str(Path(resume).parent) if resume else output + "-from-scratch"
    if (
        cfg.get("keep_stage0_self_attention")
        and output == "artifacts/v18.5-flat320/stage1"
    ):
        return str(Path(resume).parent) if resume else output + "-keep-self-attention"
    return output


def initialize_stage(model, path, *, keep_stage0_self_attention=False):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["architecture"] != ARCHITECTURE or state["stage"] != 0:
        raise ValueError("Stage 1 initialization requires a Flat320 Stage 0 checkpoint")
    model.load_state_dict(state["model"])
    if not keep_stage0_self_attention:
        for block in model.blocks:
            block.reset_history()
    return dict(
        path=str(path),
        epoch=state["epoch"],
        source=state["source"],
        binding=state["binding"],
        keep_stage0_self_attention=keep_stage0_self_attention,
    )


def cross_thaw(stage, step, updates):
    return 1.0 if stage == 0 else min(1.0, max(0.0, (step - 3 * updates + 1) / 200))


def optimizer(model, cfg, updates):
    protected = {
        id(p) for module in model.protected_modules() for p in module.parameters()
    }
    decay = {
        id(p)
        for module in model.modules()
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d))
        and getattr(module, "groups", 1) == 1
        for name, p in module.named_parameters(recurse=False)
        if name == "weight" and p.ndim >= 2
    }
    groups = []
    for audio in (False, True):
        for regularize in (False, True):
            params = [
                p
                for p in model.parameters()
                if (id(p) in protected) == audio and (id(p) in decay) == regularize
            ]
            if params:
                groups.append(
                    dict(
                        params=params,
                        weight_decay=cfg["weight_decay"] if regularize else 0.0,
                        audio=audio,
                    )
                )
    opt = torch.optim.AdamW(
        groups,
        lr=cfg["learning_rate"],
        betas=cfg["betas"],
        fused=model.output.weight.is_cuda,
    )
    warm, horizon = cfg["warmup_steps"], cfg["scheduler_epochs"] * updates

    def normal(i):
        if i < warm:
            return (i + 1) / max(1, warm)
        u = min(1.0, (i - warm) / max(1, horizon - warm))
        floor = cfg["min_lr_ratio"]
        return floor + (1 - floor) * (1 + math.cos(math.pi * u)) * 0.5

    def thaw(i):
        return normal(i) * cross_thaw(cfg["stage"], i, updates)

    return opt, torch.optim.lr_scheduler.LambdaLR(
        opt,
        [
            thaw
            if g["audio"]
            and cfg["stage"] == 1
            and not cfg.get("stage1_from_scratch", False)
            else normal
            for g in groups
        ],
    )


def set_epoch(model, epoch, *, from_scratch=False):
    frozen = model.config.stage == 1 and epoch <= 3 and not from_scratch
    for mod in model.protected_modules():
        mod.requires_grad_(not frozen)
        mod.train(not frozen)


def run(cfg, manifest, resume=None, cost=False):
    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device(cfg["device"])
    conf = FlatConfig(**cfg["model"])
    dataset = ChartGroups(manifest, seed, conf)
    model = build_model(cfg).to(device)
    cfg = dict(cfg, audio_models=dataset.audio_models)
    order = EpochOrder(dataset)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=cfg["audio_groups_per_batch"],
        sampler=order,
        collate_fn=collate,
        num_workers=cfg["workers"],
        generator=gen,
        persistent_workers=bool(cfg["workers"]),
        pin_memory=device.type == "cuda",
        prefetch_factor=cfg["prefetch_factor"] if cfg["workers"] else None,
    )
    opt, sched = optimizer(model, cfg, len(loader))
    binding = hashlib.sha256(
        manifest.read_bytes() + dataset.audio_binding + json.dumps(WIDTHS).encode()
    ).hexdigest()
    signature = {
        k: cfg[k]
        for k in (
            "stage",
            "model",
            "seed",
            "audio_groups_per_batch",
            "learning_rate",
            "warmup_steps",
            "scheduler_epochs",
            "betas",
            "weight_decay",
            "grad_clip",
            "bf16",
            "min_lr_ratio",
        )
    }
    if cfg.get("stage1_from_scratch") or cfg.get("no_history_weight", 0.5) != 0.5:
        signature.update(
            stage1_from_scratch=cfg.get("stage1_from_scratch", False),
            no_history_weight=cfg.get("no_history_weight", 0.5),
        )
    step, first, init = 0, 1, None
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        if (state["architecture"], state["binding"], state["recipe"]) != (
            ARCHITECTURE,
            binding,
            signature,
        ):
            raise ValueError(
                "Checkpoint architecture, training inputs or update recipe differ"
            )
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        restore_rng(state["rng"])
        gen.set_state(state["loader_rng"])
        step, first, init = state["step"], state["epoch"] + 1, state["initialization"]
        if init is not None and "path" in init:
            # Initialization belongs to the original run. Older checkpoints used reset.
            cfg["keep_stage0_self_attention"] = init.get(
                "keep_stage0_self_attention", False
            )
            cfg["stage0_checkpoint"] = init["path"]
        del state
    elif cfg["stage"] == 1 and cfg.get("stage1_from_scratch", False):
        init = {"mode": "random"}
    elif cfg["stage"] == 1:
        init = initialize_stage(
            model,
            cfg["stage0_checkpoint"] or "artifacts/v18.5-flat320/stage0/epoch-015.ckpt",
            keep_stage0_self_attention=cfg.get("keep_stage0_self_attention", False),
        )
    cfg["output"] = stage_output(cfg, resume)
    dest = Path(cfg["output"])
    dest.mkdir(parents=True, exist_ok=True)
    measured = []
    ncost = cfg["cost_warmup"] + cfg["cost_steps"]
    stop = (
        first + math.ceil(ncost / len(loader)) - 1 if cost else cfg["stop_after_epochs"]
    )
    num = 0
    for epoch in range(first, stop + 1):
        set_epoch(model, epoch, from_scratch=cfg.get("stage1_from_scratch", False))
        order.epoch = epoch
        cursor = (
            gen.get_state() if resume and epoch == first and cfg["workers"] else None
        )
        if cursor is not None:
            gen.manual_seed(seed)
        batches = iter(loader)
        if cursor is not None:
            gen.set_state(cursor)
        start = time.perf_counter()
        interval_moments = 0
        for batch in batches:
            before = time.perf_counter()
            diagnosing = cost and num >= cfg["cost_warmup"]
            if diagnosing and device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            audio, charts = batch.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
                enabled=cfg["bf16"] and device.type == "cuda",
            ):
                bank = model.fusion(audio)
                loss, terms, moments = model.loss(
                    charts,
                    bank,
                    cross_scale=1.0
                    if cfg.get("stage1_from_scratch", False)
                    else cross_thaw(conf.stage, step, len(loader)),
                    no_history_weight=cfg.get("no_history_weight", 0.5),
                )
            loss.backward()
            grad = (
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                if cfg["grad_clip"] is not None
                else None
            )
            opt.step()
            sched.step()
            step += 1
            num += 1
            interval_moments += moments
            if diagnosing or step % cfg["log_interval"] == 0:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                now = time.perf_counter()
                row = dict(
                    epoch=epoch,
                    step=step,
                    loss=float(loss.detach()),
                    **{k: float(v) for k, v in terms.items()},
                    audio_groups=len(audio["mel_lengths"]),
                    charts=charts["chart_count"],
                    moments=moments,
                    moments_per_second=interval_moments / (now - start),
                )
                print(json.dumps(row), flush=True)
                interval_moments, start = 0, now
                if diagnosing:
                    measured.append(
                        dict(
                            row,
                            update_seconds=now - before,
                            grad_norm=float(grad) if grad is not None else None,
                            peak_allocated=torch.cuda.max_memory_allocated()
                            if device.type == "cuda"
                            else None,
                        )
                    )
                    if num == ncost:
                        (dest / "cost.json").write_text(
                            json.dumps(dict(updates=measured), indent=2)
                        )
                        return
        if cost:
            continue
        state = dict(
            architecture=ARCHITECTURE,
            source=source_identity(),
            stage=cfg["stage"],
            initialization=init,
            config=cfg,
            recipe=signature,
            binding=binding,
            epoch=epoch,
            step=step,
            model=model.state_dict(),
            optimizer=opt.state_dict(),
            scheduler=sched.state_dict(),
            rng=rng_state(),
            loader_rng=gen.get_state(),
        )
        names = ["last.ckpt"]
        if epoch % 5 == 0 or epoch == stop:
            names.append(f"epoch-{epoch:03}.ckpt")
        for name in names:
            tmp = dest / (name + ".tmp")
            torch.save(state, tmp)
            tmp.replace(dest / name)
        print(json.dumps(dict(completed_epoch=epoch, checkpoints=names)), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path)
    p.add_argument("--stage", type=int, choices=(0, 1))
    p.add_argument("--resume", type=Path)
    p.add_argument(
        "--stage1-from-scratch",
        action="store_true",
        default=None,
        help="Initialize Stage 1 randomly and warm up all trainable parameters from step one.",
    )
    p.add_argument(
        "--keep-stage0-self-attention",
        action="store_true",
        default=None,
        help="Keep Stage 0 self-attention and its input LayerNorm when initializing Stage 1.",
    )
    p.add_argument("--cost-check", action="store_true")
    p.add_argument("--show-config", action="store_true")
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--history-backend", choices=("auto", "sdpa"))
    for key in ("manifest", "output", "stage0_checkpoint", "device"):
        p.add_argument("--" + key.replace("_", "-"))
    for key in (
        "audio_groups_per_batch",
        "workers",
        "seed",
        "warmup_steps",
        "stop_after_epochs",
        "scheduler_epochs",
        "prefetch_factor",
        "log_interval",
        "cost_warmup",
        "cost_steps",
    ):
        p.add_argument("--" + key.replace("_", "-"), type=int)
    for key in (
        "learning_rate",
        "grad_clip",
        "weight_decay",
        "min_lr_ratio",
        "no_history_weight",
    ):
        p.add_argument("--" + key.replace("_", "-"), type=float)
    args = p.parse_args(argv)
    supplied = json.loads(args.config.read_text("utf8")) if args.config else {}
    supplied.update(
        {
            k: v
            for k, v in vars(args).items()
            if v is not None
            and k
            not in ("config", "resume", "cost_check", "show_config", "history_backend")
        }
    )
    cfg = recipe(supplied.pop("stage", 0), supplied)
    if args.history_backend:
        cfg["model"]["history_backend"] = args.history_backend
    if args.show_config:
        cfg["output"] = stage_output(cfg, args.resume)
        print(json.dumps(cfg, indent=2))
    else:
        run(cfg, Path(cfg["manifest"]).resolve(), args.resume, args.cost_check)


if __name__ == "__main__":
    main()
