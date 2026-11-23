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
from . import ARCHITECTURE, ARCHITECTURE_V187, DRAFT_ARCHITECTURES, ARCHITECTURE_V20
from .codec import WIDTHS
from .data import ChartGroups, MixedChartGroups, MixedEpochSampler, TaikoGroups, collate
from .model import model_config
from maidata.model import build_model

DEFAULTS = {
    k: v
    for k, v in V16.items()
    if k not in ("model", "loss_weights", "output", "manifest", "stop_after_epochs")
}


def default_output(stage, architecture):
    version = "v18.7" if architecture == ARCHITECTURE_V187 else "v20" if architecture == ARCHITECTURE_V20 else "v19"
    return f"artifacts/{version}-flat320/stage{stage}"


def recipe(stage, supplied):
    cfg = {
        **DEFAULTS,
        "architecture": ARCHITECTURE,
        "stage": stage,
        "manifest": "data/packed-flat320-grid24/manifest.json",
        "output": default_output(stage, supplied.get("architecture", ARCHITECTURE)),
        "stop_after_epochs": 15 if stage == 0 else 75,
        "stage0_checkpoint": None,
        "keep_stage0_self_attention": False,
        "stage1_from_scratch": False,
        "no_history_weight": 0.5,
        "physical_cache": None,
        "physical_weight": 0.0,
        "physical_warmup_steps": 0,
        "physical_warning_weight": 0.0,
        "physical_use_share": True,
        "physical_category_weights": [1.0, 1.0, 4.0, 1.0],
        "gt_conflict_ce_scale": 0.25,
        **supplied,
    }
    if cfg["physical_cache"] and "physical_weight" not in supplied:
        cfg["physical_weight"] = 1.0
    if cfg["physical_weight"] > 0.0 and not cfg["physical_cache"]:
        raise ValueError("physical_weight > 0 requires --physical-cache")
    cfg["physical_category_weights"] = [float(x) for x in cfg["physical_category_weights"]]
    if cfg["stage1_from_scratch"] and "no_history_weight" not in supplied:
        cfg["no_history_weight"] = 1.0
    cfg["model"] = asdict(model_config(cfg))
    if cfg["model"]["no_history_enabled"] and cfg["architecture"] != ARCHITECTURE_V20:
        del cfg["model"]["no_history_enabled"]
    return cfg


def stage_output(cfg, resume):
    output = cfg["output"]
    default_stage1 = default_output(1, cfg["architecture"])
    if cfg.get("stage1_from_scratch") and output == default_stage1:
        return str(Path(resume).parent) if resume else output + "-from-scratch"
    if (
        cfg.get("keep_stage0_self_attention")
        and output == default_stage1
    ):
        return str(Path(resume).parent) if resume else output + "-keep-self-attention"
    return output


def initialize_stage(model, path, *, keep_stage0_self_attention=False):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["architecture"] != model.config.architecture or state["stage"] != 0:
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
    plan_params = {id(param) for name, param in model.named_parameters()
                   if name.startswith(("planner.", "plan4_embed."))}
    groups = []
    for category in ("decoder", "plan", "audio", "protected_plan"):
        audio = category in ("audio", "protected_plan")
        planner = category in ("plan", "protected_plan")
        for regularize in (False, True):
            params = [
                p
                for p in model.parameters()
                if (id(p) in protected) == audio
                and (id(p) in plan_params) == planner
                and (id(p) in decay) == regularize
            ]
            if params:
                groups.append(
                    dict(
                        params=params,
                        weight_decay=cfg["weight_decay"] if regularize else 0.0,
                        audio=audio,
                        lr=cfg["learning_rate"] * (model.config.plan_lr_scale if planner else 1.0),
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


def module_grad_norms(model):
    groups = {"grad_drafter": [], "grad_audio": [], "grad_decoder": []}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if name.startswith(("drafter.", "planner.", "reconstruction.")):
            group = "grad_drafter"
        elif name.startswith(("fusion.", "beat_encoder.")):
            group = "grad_audio"
        else:
            group = "grad_decoder"
        groups[group].append(parameter.grad)
    return {
        key: torch.nn.utils.get_total_norm(grads)
        for key, grads in groups.items()
    }


def run(cfg, manifest, resume=None, cost=False):
    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device(cfg["device"])
    conf = model_config(cfg)
    physical_cfg = (
        dict(
            physical_cache=str(Path(cfg["physical_cache"]).resolve()),
            workers=cfg["workers"],
            physical_warning_weight=cfg.get("physical_warning_weight", 0.0),
            physical_use_share=cfg.get("physical_use_share", True),
            gt_conflict_ce_scale=cfg.get("gt_conflict_ce_scale", 0.25),
        )
        if cfg.get("physical_cache")
        else None
    )
    dataset = ChartGroups(manifest, seed, conf, physical_cfg=physical_cfg)
    model = build_model(cfg).to(device)
    cfg = dict(cfg, audio_models=dataset.audio_models)
    order = EpochOrder(dataset)
    sampling = dict(batch_size=cfg["audio_groups_per_batch"], sampler=order)
    extra_binding = b""
    training_data = dataset
    if conf.architecture == ARCHITECTURE_V20:
        path = Path(cfg["taiko_manifest"])
        taiko = TaikoGroups(path, seed, conf)
        training_data = MixedChartGroups(dataset, taiko)
        order = MixedEpochSampler(len(dataset), len(taiko), seed, cfg["audio_groups_per_batch"])
        sampling = dict(batch_sampler=order)
        extra_binding = path.read_bytes() + taiko.audio_binding
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        training_data,
        **sampling,
        collate_fn=collate,
        num_workers=cfg["workers"],
        generator=gen,
        persistent_workers=bool(cfg["workers"]),
        pin_memory=device.type == "cuda",
        prefetch_factor=cfg["prefetch_factor"] if cfg["workers"] else None,
    )
    opt, sched = optimizer(model, cfg, len(loader))
    phys_binding = dataset.physical.binding if dataset.physical is not None else b""
    binding = hashlib.sha256(
        manifest.read_bytes()
        + dataset.audio_binding
        + json.dumps(WIDTHS).encode()
        + extra_binding
        + phys_binding
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
    if conf.architecture == ARCHITECTURE_V187 and conf.plan_log_weight == 0:
        # Preserve the original v18.7 resume signature for its unchanged loss.
        signature["model"] = dict(signature["model"])
        del signature["model"]["plan_log_weight"]
    if cfg.get("stage1_from_scratch") or cfg.get("no_history_weight", 0.5) != 0.5:
        signature.update(
            stage1_from_scratch=cfg.get("stage1_from_scratch", False),
            no_history_weight=cfg.get("no_history_weight", 0.5),
        )
    if cfg.get("physical_cache"):
        signature.update(
            physical_weight=cfg["physical_weight"],
            physical_warmup_steps=cfg["physical_warmup_steps"],
            physical_warning_weight=cfg["physical_warning_weight"],
            physical_use_share=cfg["physical_use_share"],
            physical_category_weights=cfg["physical_category_weights"],
            gt_conflict_ce_scale=cfg["gt_conflict_ce_scale"],
        )
    step, first, init = 0, 1, None
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        if (state["architecture"], state["binding"], state["recipe"]) != (
            conf.architecture,
            binding,
            signature,
        ):
            raise ValueError(
                "Checkpoint architecture, training inputs or update recipe differ"
            )
        if isinstance(order, MixedEpochSampler) and tuple(state["taiko_cursor"]) != order.cursor(state["epoch"]):
            raise ValueError("Checkpoint Taiko deck cursor differs")
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
            cfg["stage0_checkpoint"] or str(Path(default_output(0, conf.architecture)) / "epoch-015.ckpt"),
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
            diagnosing = cost and num >= cfg["cost_warmup"]
            if diagnosing and device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            before = time.perf_counter()
            audio, charts = batch.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
                enabled=cfg["bf16"] and device.type == "cuda",
            ):
                bank = model.fusion(audio)
                pwarm = cfg.get("physical_warmup_steps", 0)
                phys_ramp = 1.0 if pwarm <= 0 else min(1.0, (step + 1) / pwarm)
                loss, terms, moments = model.loss(
                    charts,
                    bank,
                    step=step,
                    cross_scale=1.0
                    if cfg.get("stage1_from_scratch", False)
                    else cross_thaw(conf.stage, step, len(loader)),
                    no_history_weight=cfg.get("no_history_weight", 0.5),
                    physical_weight=cfg.get("physical_weight", 0.0) * phys_ramp,
                    physical_category_weights=cfg.get(
                        "physical_category_weights", (1.0, 1.0, 4.0, 1.0)
                    ),
                )
            loss.backward()
            logging_step = diagnosing or (step + 1) % cfg["log_interval"] == 0
            group_norms = module_grad_norms(model) if logging_step and conf.architecture in DRAFT_ARCHITECTURES else {}
            grad = (
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                if cfg["grad_clip"] is not None
                else None
            )
            if grad is None and logging_step:
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                grad = torch.nn.utils.get_total_norm(grads)
            opt.step()
            sched.step()
            step += 1
            num += 1
            interval_moments += moments
            if logging_step:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                now = time.perf_counter()
                row = dict(
                    epoch=epoch,
                    step=step,
                    loss=float(loss.detach()),
                    grad_norm=float(grad),
                    **{k: float(v) for k, v in group_norms.items()},
                    **{k: float(v) for k, v in terms.items()},
                    audio_groups=len(audio["mel_lengths"]),
                    charts=charts["chart_count"],
                    moments=moments,
                    moments_per_second=interval_moments / (now - start),
                )
                if conf.architecture in DRAFT_ARCHITECTURES:
                    gates = torch.stack([block.draft_attention.gate.detach() for block in model.blocks])
                    row["draft_gates"] = [round(g, 5) for g in gates.tolist()]
                print(json.dumps(row), flush=True)
                interval_moments, start = 0, now
                if diagnosing:
                    measured.append(
                        dict(
                            row,
                            update_seconds=now - before,
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
            **({"taiko_cursor": order.cursor(epoch)} if isinstance(order, MixedEpochSampler) else {}),
            architecture=conf.architecture,
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
    # Guided FlexAttention training needs a PyTorch build exposing
    # torch.nn.attention.flex_attention.AuxRequest(lse=True) and working GPU Triton.
    # Check both on the training host: builds without AuxRequest (e.g. 2.8) use
    # bucketed SDPA for guided attention, increasing step time and peak memory.
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
    p.add_argument(
        "--physical-use-share", action=argparse.BooleanOptionalAction, default=None
    )
    p.add_argument("--history-backend", choices=("auto", "sdpa"))
    for key in (
        "manifest",
        "output",
        "stage0_checkpoint",
        "device",
        "physical_cache",
    ):
        p.add_argument("--" + key.replace("_", "-"))
    for key in (
        "audio_groups_per_batch",
        "workers",
        "seed",
        "warmup_steps",
        "physical_warmup_steps",
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
        "physical_weight",
        "physical_warning_weight",
        "gt_conflict_ce_scale",
    ):
        p.add_argument("--" + key.replace("_", "-"), type=float)
    p.add_argument("--taiko-manifest")
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
