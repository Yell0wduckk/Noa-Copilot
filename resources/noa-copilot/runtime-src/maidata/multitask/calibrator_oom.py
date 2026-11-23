"""Bounded pure-microbatch OOM recovery for the v18 BMS Beat Calibrator.

Only execution partitioning (microbatch song count, trailing-padding trimming,
and single-tensor optimizer fallback after complete-epoch process recovery)
may change. Never uses activation checkpointing or forward recomputation.
"""

from __future__ import annotations

from collections import Counter
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import torch

from maidata.joint_training.checkpoint import restore_rng, rng_state
from .calibrator_augmentation import mask_aug_sources

CALIBRATOR_OOM_EXIT_CODE = 85


class CalibratorRestartRequired(RuntimeError):
    """Signal the CLI supervisor to discard the process and restore a complete epoch."""


class CalibratorBatch(dict):
    """Collated calibrator batch preserving exact dict keys while carrying song IDs."""

    song_keys: tuple[str, ...] = ()


def calibrator_batch_key(batch) -> str:
    keys = getattr(batch, "song_keys", ())
    if keys:
        payload = dict(keys=sorted(str(k) for k in keys))
    else:
        lengths = batch["sample_lengths"].tolist()
        refs = [round(float(x), 5) for x in batch["song_ref_db"].tolist()]
        payload = dict(samples=sorted(zip(lengths, refs)))
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def padded_batch_seconds(batch, sample_rate: int = 44100) -> float:
    count = int(len(batch["sample_lengths"]))
    width = int(batch["waveforms"].shape[-1]) if count else 0
    return count * (width / float(sample_rate))


def slice_calibrator_batch(
    batch,
    indices: list[int],
    *,
    frame_rate: int = 125,
    sample_rate: int = 44100,
) -> CalibratorBatch:
    idx = torch.as_tensor(indices, dtype=torch.long)
    sub_lengths = batch["sample_lengths"].index_select(0, idx)
    max_samples = max(1, int(sub_lengths.max().item()))
    max_frames = max(1, math.ceil(max_samples * frame_rate / sample_rate))
    max_mert = (max_frames + 4) // 5
    targets = {
        k: v.index_select(0, idx)[:, :max_frames] for k, v in batch["targets"].items()
    }
    needs = batch.get("mert_needs_online")
    if needs is not None:
        needs = needs.index_select(0, idx)
    mert = None
    if batch.get("mert") is not None and (needs is None or not needs.all()):
        mert = batch["mert"][:, :max_mert].index_select(0, idx)
    out = CalibratorBatch(
        waveforms=batch["waveforms"][:, :max_samples].index_select(0, idx),
        targets=targets,
        sample_lengths=sub_lengths,
        song_ref_db=batch["song_ref_db"].index_select(0, idx),
        mert=mert,
    )
    if "speed_segments" in batch:
        out["speed_segments"] = [batch["speed_segments"][i] for i in indices]
    if "pitch_semitones" in batch:
        pitch = batch["pitch_semitones"]
        out["pitch_semitones"] = None if pitch is None else pitch.index_select(0, idx)
    if "mask_aug_sources" in batch:
        out["mask_aug_sources"] = batch["mask_aug_sources"][:, :max_frames].index_select(0, idx)
    if needs is not None:
        out["mert_needs_online"] = needs
    src_keys = getattr(batch, "song_keys", ())
    if src_keys:
        out.song_keys = tuple(src_keys[i] for i in indices)
    return out


def split_calibrator_batch(
    batch,
    *,
    min_parts: int = 1,
    max_microbatch_seconds: float | None = None,
    frame_rate: int = 125,
    sample_rate: int = 44100,
) -> list[dict]:
    count = int(len(batch["sample_lengths"]))
    if count <= 1:
        return [batch]
    over_budget = (
        max_microbatch_seconds is not None
        and max_microbatch_seconds > 0.0
        and padded_batch_seconds(batch, sample_rate) > max_microbatch_seconds
    )
    if min_parts <= 1 and not over_budget:
        return [batch]

    # Group songs of closest sample_length together so each microbatch trims
    # trailing padding tightly to its own longest song.
    lengths = [int(x) for x in batch["sample_lengths"].tolist()]
    order = sorted(range(count), key=lambda i: lengths[i])

    if max_microbatch_seconds is not None and max_microbatch_seconds > 0.0:
        groups: list[list[int]] = []
        cur: list[int] = []
        for i in order:
            cand_len = len(cur) + 1
            cand_sec = cand_len * (lengths[i] / float(sample_rate))
            if cur and cand_sec > max_microbatch_seconds:
                groups.append(cur)
                cur = [i]
            else:
                cur.append(i)
        if cur:
            groups.append(cur)
    else:
        groups = [order]

    target_parts = min(count, max(1, int(min_parts)))
    while len(groups) < target_parts:
        splittable = [gi for gi, g in enumerate(groups) if len(g) > 1]
        if not splittable:
            break
        worst = max(splittable, key=lambda gi: len(groups[gi]) * lengths[groups[gi][-1]])
        g = groups[worst]
        mid = len(g) // 2
        groups[worst : worst + 1] = [g[:mid], g[mid:]]

    return [
        slice_calibrator_batch(batch, g, frame_rate=frame_rate, sample_rate=sample_rate)
        for g in groups
    ]


def _loss_targets(piece, model, device):
    cpu_keys = ("gold_bar", "score_beat", "grid_peak") if hasattr(model, "correction") else ("gold_bar", "score_beat")
    return {key: value if key in cpu_keys else value.to(device, non_blocking=True)
            for key, value in piece["targets"].items()}


def _forward_inputs(piece, targets, device, mert_extractor):
    kwargs = {k: v.to(device, non_blocking=True) for k, v in piece.items()
              if k not in ("targets", "mask_aug_sources", "mert_needs_online", "mert", "pitch_semitones", "speed_segments")
              and v is not None}
    if "speed_segments" in piece:
        kwargs["speed_segments"] = piece["speed_segments"]
    if piece.get("pitch_semitones") is not None:
        kwargs["pitch_semitones"] = piece["pitch_semitones"]
    if "mask_aug_sources" in piece:
        kwargs["mask_aug_sources"] = piece["mask_aug_sources"]
    if "audible" in targets:
        kwargs["audible_mask"] = targets["audible"].bool()
    needs = piece.get("mert_needs_online")
    if mert_extractor is not None and needs is not None and needs.any():
        idx = torch.where(needs)[0]
        width = int(piece["sample_lengths"][idx].max())
        tables = {"speed_segments": [piece["speed_segments"][j] for j in idx.tolist()]} if "speed_segments" in piece else {}
        idx = idx.to(device)
        extracted = mert_extractor(kwargs["waveforms"][:, :width].index_select(0, idx),
                                   kwargs["sample_lengths"][idx], **tables)
        cached = piece.get("mert")
        if cached is None:
            kwargs["mert"] = extracted
        else:
            frames = max(cached.shape[1], extracted.shape[1])
            merged = extracted.new_zeros(len(needs), frames, *extracted.shape[2:])
            merged[:, :cached.shape[1]].copy_(cached, non_blocking=True)
            merged[idx] = 0
            merged[idx, :extracted.shape[1]] = extracted
            kwargs["mert"] = merged
    elif piece.get("mert") is not None:
        kwargs["mert"] = piece["mert"].to(device, non_blocking=True)
    elif mert_extractor is not None and needs is None:
        kwargs["mert"] = mert_extractor(kwargs["waveforms"], kwargs["sample_lengths"],
                                        **({"speed_segments": piece["speed_segments"]} if "speed_segments" in piece else {}))
    return kwargs


def release_oom_state(optimizer=None) -> None:
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def batch_needs_online_mert(batch) -> bool:
    needs = batch.get("mert_needs_online")
    if needs is not None:
        return bool(needs.any())
    return batch.get("mert") is None


def precompute_batch_mert(batch, mert_extractor, *, subbatch: int = 2) -> CalibratorBatch:
    """Compute online MERT rows into a CPU batch before calibrator forward/backward.

    Used by the optional low-VRAM staged mode (`stage_mert_first`) so MERT weights
    and forward workspaces never coexist with calibrator backward graphs on GPU.
    """
    if mert_extractor is None:
        return batch
    needs = batch.get("mert_needs_online")
    cached = batch.get("mert")
    if needs is not None:
        if not bool(needs.any()):
            return batch
        online_idx = torch.where(needs)[0].tolist()
    elif cached is not None:
        return batch
    else:
        online_idx = list(range(len(batch["sample_lengths"])))
    if not online_idx:
        return batch

    lengths = batch["sample_lengths"]
    online_order = sorted(online_idx, key=lambda i: int(lengths[i]))
    step = max(1, int(subbatch))
    extracted_rows: dict[int, torch.Tensor] = {}
    pos = 0
    while pos < len(online_order):
        group = online_order[pos : pos + step]
        idx_t = torch.as_tensor(group, dtype=torch.long)
        sub_len = lengths.index_select(0, idx_t)
        width = max(1, int(sub_len.max().item()))
        sub_wave = batch["waveforms"][:, :width].index_select(0, idx_t)
        try:
            with torch.no_grad():
                tables = ({"speed_segments": [batch["speed_segments"][j] for j in group]}
                          if "speed_segments" in batch else {})
                feats = mert_extractor(sub_wave, sub_len, **tables).to("cpu", non_blocking=False)
        except torch.cuda.OutOfMemoryError:
            del sub_wave
            if step <= 1:
                raise
            release_oom_state()
            step = max(1, step // 2)
            continue
        for row_idx, feat in zip(group, feats):
            extracted_rows[row_idx] = feat
        del feats, sub_wave
        pos += len(group)

    example = next(iter(extracted_rows.values()))
    max_frames = max(
        max(feat.shape[0] for feat in extracted_rows.values()),
        0 if cached is None else cached.shape[1],
    )
    merged = example.new_zeros(len(lengths), max_frames, *example.shape[1:])
    if cached is not None:
        merged[:, : cached.shape[1]].copy_(cached.to(dtype=merged.dtype))
    for row_idx, feat in extracted_rows.items():
        merged[row_idx] = 0
        merged[row_idx, : feat.shape[0]] = feat

    out = CalibratorBatch(batch)
    out.song_keys = getattr(batch, "song_keys", ())
    out["mert"] = merged
    if needs is not None:
        out["mert_needs_online"] = torch.zeros_like(needs, dtype=torch.bool)
    return out


class CalibratorOOMRecovery:
    """Persist learned microbatch partitions and recover in-place from CUDA OOM."""

    def __init__(
        self,
        output: str | Path,
        *,
        allow_process_restart: bool = False,
        frame_rate: int = 125,
        sample_rate: int = 44100,
    ):
        self.path = Path(output) / "oom-recovery.json"
        if self.path.exists():
            self.state = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.state = dict(
                policies={},
                learned_max_batch_seconds=None,
                learned_max_val_seconds=None,
                optimizer_low_memory=False,
            )
        self.allow_process_restart = allow_process_restart
        self.frame_rate = frame_rate
        self.sample_rate = sample_rate
        self.phase = "start"
        self.index = 0

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    def mark(self, phase: str, index: int) -> None:
        self.phase = phase
        self.index = index

    def configure_optimizer(self, optimizer) -> None:
        if self.state.get("optimizer_low_memory"):
            for group in optimizer.param_groups:
                group.update(fused=False, foreach=False)

    def policy(self, batch) -> dict:
        return self.state["policies"].get(calibrator_batch_key(batch), {})

    def effective_limit(self, *, training: bool, configured_max_seconds: float | None = None) -> float | None:
        learned_key = "learned_max_batch_seconds" if training else "learned_max_val_seconds"
        learned = self.state.get(learned_key)
        candidates = [
            float(v)
            for v in (configured_max_seconds, learned)
            if v is not None and float(v) > 0.0
        ]
        return min(candidates) if candidates else None

    def _record_failure(self, batch, pieces: list[dict], message: str, *, training: bool) -> bool:
        key = calibrator_batch_key(batch)
        policy = dict(self.policy(batch))
        parts_field = "parts" if training else "val_parts"
        learned_field = "learned_max_batch_seconds" if training else "learned_max_val_seconds"
        total_count = int(len(batch["sample_lengths"]))
        restart = False

        if training and self.phase == "optimizer":
            if self.state.get("optimizer_low_memory") or not self.allow_process_restart:
                return False
            self.state["optimizer_low_memory"] = True
            restart = True
        else:
            fail_piece = pieces[min(self.index, len(pieces) - 1)]
            piece_count = int(len(fail_piece["sample_lengths"]))
            piece_sec = padded_batch_seconds(fail_piece, self.sample_rate)
            if len(pieces) < total_count:
                prev_parts = int(policy.get(parts_field, 1))
                policy[parts_field] = min(total_count, max(prev_parts * 2, len(pieces) + 1))
                if piece_count > 1 and piece_sec > 0.0:
                    safe_sec = round(0.92 * piece_sec, 3)
                    prev_safe = self.state.get(learned_field)
                    self.state[learned_field] = (
                        safe_sec if prev_safe is None else min(float(prev_safe), safe_sec)
                    )
            else:
                if policy.get("clean_restart") or not self.allow_process_restart:
                    return False
                policy["clean_restart"] = True
                restart = True

        self.state["policies"][key] = policy
        self.state["last_failure"] = dict(
            phase=self.phase,
            batch=key,
            songs=list(getattr(batch, "song_keys", ())),
            policy=policy,
            learned_max_batch_seconds=self.state.get("learned_max_batch_seconds"),
            learned_max_val_seconds=self.state.get("learned_max_val_seconds"),
            message=message,
            restart=restart,
        )
        self.save()
        print(
            json.dumps(
                dict(event="oom_restart" if restart else "oom_retry", **self.state["last_failure"]),
                ensure_ascii=False,
            ),
            flush=True,
        )
        if restart:
            raise CalibratorRestartRequired(message)
        return True

    def run_train_step(
        self,
        model,
        batch,
        optimizer,
        scheduler,
        *,
        cfg: dict,
        device: str,
        roformer_freeze: int,
        loss_fn,
        ref_song_frames: int,
        mert_extractor=None,
    ):
        if cfg.get("clean_lattice") and cfg["stage"] == 1 and cfg.get("mask_aug_probability", 0.) > 0:
            augmented = CalibratorBatch(batch)
            augmented.song_keys = getattr(batch, "song_keys", ())
            augmented["mask_aug_sources"] = mask_aug_sources(batch["targets"], cfg)
            batch = augmented
        rng = rng_state()
        use_cuda_amp = str(device).startswith("cuda")
        denom = float(cfg["batch_size"] * ref_song_frames)
        cfg_limit = cfg.get("max_microbatch_seconds")
        retries = 0

        while True:
            pol = self.policy(batch)
            limit = self.effective_limit(training=True, configured_max_seconds=cfg_limit)
            pieces = split_calibrator_batch(
                batch,
                min_parts=int(pol.get("parts", 1)),
                max_microbatch_seconds=limit,
                frame_rate=self.frame_rate,
                sample_rate=self.sample_rate,
            )
            self.mark("start", 0)
            try:
                if roformer_freeze > 0:
                    model.set_stage2_roformer_trainable(scheduler.last_epoch >= roformer_freeze)
                optimizer.zero_grad(set_to_none=True)
                total_frame_sum = torch.zeros((), device=device, dtype=torch.float64)
                total_valid = torch.zeros((), device=device, dtype=torch.float64)

                for idx, piece in enumerate(pieces):
                    self.mark("microbatch", idx)
                    sub_targets = _loss_targets(piece, model, device)
                    kwargs = _forward_inputs(piece, sub_targets, device, mert_extractor)
                    with torch.autocast(torch.device(device).type, torch.bfloat16, enabled=use_cuda_amp):
                        out = model.forward_waveforms(**kwargs)
                        frame_sum = loss_fn(out, sub_targets, ref_batch_frames=1.0)
                        sub_loss = frame_sum / denom
                    sub_loss.backward()
                    total_frame_sum = total_frame_sum + frame_sum.detach()
                    total_valid = total_valid + out["valid_mask"].sum()
                    del sub_loss, frame_sum, out, sub_targets, kwargs

                self.mark("optimizer", len(pieces) - 1)
                grad_clip = cfg.get("grad_clip_norm", 50.0)
                if grad_clip is not None and float(grad_clip) > 0.0:
                    params = [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
                    if params:
                        torch.nn.utils.clip_grad_norm_(params, float(grad_clip))
                optimizer.step()
                scheduler.step()
                if retries > 0:
                    self.state["last_success"] = dict(
                        retries=retries,
                        batch=calibrator_batch_key(batch),
                        pieces=len(pieces),
                        learned_max_batch_seconds=self.state.get("learned_max_batch_seconds"),
                    )
                    self.save()
                    print(
                        json.dumps(dict(event="oom_recovered", **self.state["last_success"]), ensure_ascii=False),
                        flush=True,
                    )
                return total_frame_sum, total_valid, retries
            except torch.cuda.OutOfMemoryError as exc:
                message = str(exc)
                exc.__traceback__ = None
                exc.__context__ = None
                exc.__cause__ = None

            release_oom_state(optimizer)
            restore_rng(rng)
            if not self._record_failure(batch, pieces, message, training=True):
                raise torch.cuda.OutOfMemoryError(
                    f"Calibrator OOM persists at the finest microbatch strategy: {message}"
                )
            retries += 1

    @torch.no_grad()
    def run_eval_step(
        self,
        model,
        batch,
        *,
        cfg: dict,
        device: str,
        frame_sum_fn,
        grid_counts=None,
        mert_extractor=None,
    ):
        rng = rng_state()
        use_cuda_amp = str(device).startswith("cuda")
        cfg_limit = cfg.get("max_microbatch_seconds")

        while True:
            pol = self.policy(batch)
            limit = self.effective_limit(training=False, configured_max_seconds=cfg_limit)
            pieces = split_calibrator_batch(
                batch,
                min_parts=int(pol.get("val_parts", 1)),
                max_microbatch_seconds=limit,
                frame_rate=self.frame_rate,
                sample_rate=self.sample_rate,
            )
            self.mark("validation", 0)
            batch_counts = Counter() if grid_counts is not None else None
            try:
                total_frame_sum = torch.zeros((), device=device, dtype=torch.float64)
                total_valid = torch.zeros((), device=device, dtype=torch.float64)
                for idx, piece in enumerate(pieces):
                    self.mark("validation_microbatch", idx)
                    sub_targets = _loss_targets(piece, model, device)
                    kwargs = _forward_inputs(piece, sub_targets, device, mert_extractor)
                    with torch.autocast(torch.device(device).type, torch.bfloat16, enabled=use_cuda_amp):
                        out = model.forward_waveforms(**kwargs)
                        frame_sum = frame_sum_fn(out, sub_targets)
                        if grid_counts is not None:
                            from .calibrator_boundary_decoder import clean_validation_counts
                            batch_counts.update(clean_validation_counts(out, sub_targets))
                    total_frame_sum = total_frame_sum + frame_sum.detach()
                    total_valid = total_valid + out["valid_mask"].sum()
                    del frame_sum, out, sub_targets, kwargs
                if grid_counts is not None:
                    grid_counts.update(batch_counts)
                return total_frame_sum, total_valid
            except torch.cuda.OutOfMemoryError as exc:
                message = str(exc)
                exc.__traceback__ = None
                exc.__context__ = None
                exc.__cause__ = None

            release_oom_state()
            restore_rng(rng)
            if not self._record_failure(batch, pieces, message, training=False):
                raise torch.cuda.OutOfMemoryError(
                    f"Calibrator validation OOM persists at the finest microbatch strategy: {message}"
                )


def supervise_calibrator_train(argv: list[str], output: str | Path, *, initial_resume=None) -> int:
    """Lightweight parent supervisor that restarts worker from complete-epoch last.ckpt on code 85."""
    env = dict(os.environ, MAIDATA_CALIBRATOR_OOM_WORKER="1")
    out_dir = Path(output)
    command = [sys.executable, "-m", "maidata.multitask.bms_calibrator", *argv]
    while True:
        result = subprocess.run(command, env=env)
        if result.returncode != CALIBRATOR_OOM_EXIT_CODE:
            return result.returncode
        latest = out_dir / "last.ckpt"
        resume = latest if latest.exists() else initial_resume
        if resume is not None:
            env.pop("MAIDATA_CALIBRATOR_OOM_FRESH_RETRY", None)
            env["MAIDATA_CALIBRATOR_OOM_RESUME"] = str(Path(resume).resolve())
        else:
            env["MAIDATA_CALIBRATOR_OOM_FRESH_RETRY"] = "1"
        print(
            json.dumps(
                dict(
                    event="oom_process_recovery",
                    resume=None if resume is None else str(resume),
                    action="restore_complete_epoch" if resume is not None else "repeat_initial_seed",
                ),
                ensure_ascii=False,
            ),
            flush=True,
        )
