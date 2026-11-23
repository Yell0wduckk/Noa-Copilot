"""Current-song NoaCopilot features. Files are published only after a complete write."""

import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import torch

from maidata.draft.types import BeatMemory
from maidata.flat import DRAFT_ARCHITECTURES
from maidata.flat.audio import (
    extract_mert_batch,
    flat_mel_frontend,
    frozen_mert,
    load_wave,
)
from maidata.flat.audio_pitch import quantize_mert
from maidata.flat.data import beat_grid
from maidata.flat.drafter import draft_anchor_valid
from maidata.spine.features import condition


def route_library_caches(folder):
    # Native and WSL workers resolve their own paths, including for library caches.
    for key, subdir in (("HF_HOME", "huggingface"), ("HF_HUB_CACHE", "huggingface/hub"),
                        ("HF_MODULES_CACHE", "modules"), ("TORCH_HOME", "torch"),
                        ("TORCHINDUCTOR_CACHE_DIR", "torch/inductor"),
                        ("TRITON_CACHE_DIR", "triton"), ("CUDA_CACHE_PATH", "cuda")):
        os.environ[key] = str(folder / subdir)
    os.environ["HF_HUB_OFFLINE"] = "1"
    # Older Transformers versions give these deprecated settings precedence over HF_HOME.
    for key in ("PYTORCH_PRETRAINED_BERT_CACHE", "PYTORCH_TRANSFORMERS_CACHE", "TRANSFORMERS_CACHE"):
        os.environ.pop(key, None)


def song_cache_folder(audio):
    folder = Path(audio).absolute().parent / ".miacode" / ".noacopilot"
    route_library_caches(folder)
    return folder


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def file_digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_write(path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        name = temp.name
    try:
        writer(Path(name))
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


@torch.inference_mode()
def extract_l8(path, model_path, device):
    device = torch.device(device)
    wave, _ = load_wave(path)
    # Match the trained feature clock even for a partial final frame.
    frames = max(1, (len(wave) * 125 + 44099) // 44100)

    def extract():
        mert = extract_mert_batch([wave], device, model_path, (frames + 4) // 5, bundled=True)[0]
        data, scale = quantize_mert(mert)
        return dict(
            mert=data.cpu()[None], mert_scale=scale.cpu()[None], samples=len(wave)
        )

    if device.type != "cuda":
        return extract()
    try:
        return extract()
    finally:
        frozen_mert.cache_clear()
        gc.collect()
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


class ProjectCache:
    def __init__(self, service, checkpoint):
        self.service = service
        self.checkpoint = checkpoint
        self.checkpoint_key = None
        self.path = self.stat = self.folder = None
        self.l1 = self.l2 = self.l3 = None
        self.grid_key = self.cond_key = None
        self.index = {}

    def publish(self, layer, name, payload):
        atomic_write(self.folder / name, lambda p: torch.save(payload, p))
        old = dict(self.index)
        self.index.update(
            version=1,
            audio_hash=self.audio_hash,
            architecture=self.service.model.config.architecture
            if self.service.model is not None
            else None,
        )
        if old.get("audio_hash") != self.audio_hash:
            self.index.pop("mert", None)
            self.index.pop("roformer", None)
            self.index.pop("drafter", None)
        if layer == "roformer" and old.get(layer) != name:
            self.index.pop("drafter", None)
        self.index[layer] = name
        atomic_write(
            self.folder / "manifest.json",
            lambda p: p.write_text(json.dumps(self.index, indent=2), encoding="utf-8"),
        )
        for key in ("mert", "roformer", "drafter"):
            previous = old.get(key, "")
            if (
                previous
                and previous != self.index.get(key)
                and Path(previous).name == previous
                and previous.startswith(key + "_")
                and previous.endswith(".pt")
            ):
                (self.folder / previous).unlink(missing_ok=True)

    def get(self, layer, name, produce):
        path = self.folder / name
        if path.is_file():
            try:
                return torch.load(path, map_location="cpu", weights_only=True)
            except (OSError, RuntimeError, EOFError, ValueError):
                pass
        data = produce()
        self.publish(layer, name, data)
        return data

    def bind(self, path, req):
        path = Path(path)
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
        if self.path != path or self.stat != signature:
            self.path, self.stat = path, signature
            self.folder = song_cache_folder(path)
            self.audio_hash = file_digest(path)
            self.grid_key = self.cond_key = None
            self.l1 = self.l2 = self.l3 = None
            try:
                self.index = json.loads(
                    (self.folder / "manifest.json").read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                self.index = {}
        if self.l1 is None:

            def extract():
                self.service.emit(
                    dict(
                        event="status",
                        state="encoding_audio",
                        message="正在分析音频...",
                    )
                )
                argv = self.service.mert_command
                if not argv:
                    return extract_l8(path, self.service.mert, self.service.mert_device)
                # stdout is reserved for JSON in the parent. Hand off tensors in a transient file.
                self.folder.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(dir=self.folder) as tmp:
                    out = Path(tmp) / "l8.pt"
                    source = str(Path(__file__).resolve().parents[2])
                    args = [
                        source,
                        str(path),
                        self.service.mert,
                        str(out),
                        self.service.mert_device,
                    ]
                    env = dict(os.environ)
                    command = list(argv)
                    if command[0].lower().endswith("wsl.exe"):
                        command += [
                            "--cd",
                            source,
                            "--exec",
                            "env",
                            "PYTHONIOENCODING=utf-8",
                            "OMP_NUM_THREADS=" + str(torch.get_num_threads()),
                            self.service.mert_python,
                        ]
                        args[0] = "."  # --cd translates the Windows mount for WSL.
                    elif os.name == "posix" and command[0][1:2] == ":":
                        from .copilot import native_path

                        command[0] = native_path(command[0])
                        args[:4] = [
                            subprocess.check_output(
                                ["wslpath", "-w", p], text=True
                            ).strip()
                            for p in args[:4]
                        ]
                    boot = "import sys,runpy;sys.path.insert(0,sys.argv.pop(1));runpy.run_module('maidata.spine.project_cache',run_name='__main__')"
                    subprocess.run(
                        command + ["-B", "-c", boot, *args],
                        check=True,
                        env=env,
                        stdout=sys.stderr,
                    )
                    return torch.load(out, weights_only=True, map_location="cpu")

            self.l1 = self.get(
                "mert",
                f"mert_{self.audio_hash}.pt",
                extract,
            )
        mert_name = f"mert_{self.audio_hash}.pt"
        if self.index.get("mert") != mert_name:
            self.publish("mert", mert_name, self.l1)
        return self.l1["samples"] / 44100

    def prepared(self, chart, grid, meta, limit, req):
        model, device = self.service.model, self.service.device
        if self.checkpoint_key is None:
            self.checkpoint_key = file_digest(self.checkpoint)
        ck = self.checkpoint_key[:24]
        grid_key = digest(
            [
                sorted(chart.tempos.items()),
                chart.first,
                limit,
                req.get("conv_enabled", True),
                req.get("mert_enabled", True),
            ]
        )
        name = f"{self.audio_hash}_{grid_key[:24]}"
        self.index.update(checkpoint_hash=self.checkpoint_key, grid_hash=grid_key)
        if self.grid_key != grid_key:

            def encode():
                wave, ref = load_wave(self.path)
                lengths = torch.tensor([len(wave)], device=device)
                features, lengths = flat_mel_frontend(device).batch(
                    wave[None].to(device), lengths, [ref]
                )
                mel = features[:, :4].transpose(1, 2).flatten(2).bfloat16()
                mert = self.l1["mert"].to(device)
                bank = model.fusion(
                    dict(
                        mel=mel,
                        mel_lengths=lengths,
                        mert=mert,
                        mert_scale=self.l1["mert_scale"].to(device),
                        mert_lengths=(lengths + 4) // 5,
                    )
                )
                if not req.get("conv_enabled", True):
                    bank = bank._replace(
                        mel_feat=torch.zeros_like(bank.mel_feat),
                        envelope=torch.zeros_like(bank.envelope),
                    )
                if not req.get("mert_enabled", True):
                    bank = bank._replace(mert_feat=torch.zeros_like(bank.mert_feat))
                times, valid = beat_grid(grid, limit, int(lengths[0]))
                cells, _ = model.beat_encoder(
                    bank, 0, times.to(device), valid.to(device)
                )
                return dict(
                    features=cells.half().cpu(),
                    length=int(lengths[0]),
                    envelope=bank.envelope.half().cpu(),
                )

            data = self.get("roformer", f"roformer_{name}_{ck}.pt", encode)
            dtype = next(model.parameters()).dtype
            cells = data["features"].to(device, dtype)
            self.l2 = dict(
                features=cells,
                length=data["length"],
                envelope=data["envelope"].to(device, dtype),
                audio_grid=beat_grid(grid, limit, data["length"]),
                audio=model.beat_encoder.project_audio(cells),
            )
            self.grid_key, self.cond_key = grid_key, None
        roformer_name = f"roformer_{name}_{ck}.pt"
        if self.index.get("roformer") != roformer_name:
            self.publish(
                "roformer",
                roformer_name,
                {
                    "length": self.l2["length"],
                    "features": self.l2["features"].half().cpu(),
                    "envelope": self.l2["envelope"].half().cpu(),
                },
            )
        cond_key = digest(
            [
                meta.get(k)
                for k in (
                    "level",
                    "difficulty",
                    "charter",
                    "song_id",
                    "version",
                    "cabinet",
                )
            ]
        )
        if self.cond_key != cond_key:
            self.index["condition_hash"] = cond_key

            def draft():
                layer = self.l2
                times, valid = (t.to(device) for t in layer["audio_grid"])
                grids = [(0, times, valid)]
                bank = SimpleNamespace(
                    envelope=layer["envelope"],
                    mel_lengths=torch.tensor([layer["length"]], device=device),
                )
                values = torch.tensor(
                    [condition(meta, model.config.designer_vocab)[0]], device=device
                )
                if model.config.architecture in DRAFT_ARCHITECTURES:
                    anchors = draft_anchor_valid(valid.view(-1, 24).any(1))
                    all_valid = bool(anchors.all())
                    z, beats = model.drafter(
                        layer["features"],
                        grids,
                        [0],
                        values,
                        bank,
                        None if all_valid else anchors.nonzero().flatten(),
                        all_valid, designers=torch.tensor([condition(meta, model.config.designer_vocab)[1]], device=device),
                    )
                    return dict(
                        z_values=z.values.half().cpu(),
                        z_beats=z.beats.cpu(),
                        z_valid=z.valid.cpu(),
                        beat_valid=beats.cpu(),
                        curve=model.planner(z, beats).half().cpu(),
                    )
                curve, _ = model.predict_plan(
                    bank, layer["features"], grids, [0], values
                )
                return dict(curve=curve.half().cpu())

            data = self.get(
                "drafter", f"drafter_{name}_{cond_key[:24]}_{ck}.pt", draft
            )
            self.l3 = {
                k: t.to(device, torch.float32 if t.is_floating_point() else t.dtype)
                for k, t in data.items()
            }
            if "z_values" in self.l3:
                self.l3["z"] = BeatMemory(
                    *(self.l3.pop(k) for k in ("z_values", "z_beats", "z_valid"))
                )
            self.cond_key = cond_key
        drafter_name = f"drafter_{name}_{cond_key[:24]}_{ck}.pt"
        if self.index.get("drafter") != drafter_name:
            payload = {"curve": self.l3["curve"].half().cpu()}
            if "z" in self.l3:
                z = self.l3["z"]
                payload.update(
                    z_values=z.values.half().cpu(),
                    z_beats=z.beats.cpu(),
                    z_valid=z.valid.cpu(),
                    beat_valid=self.l3["beat_valid"].cpu(),
                )
            self.publish("drafter", drafter_name, payload)
        return {**self.l2, **self.l3}


if __name__ == "__main__":
    from .copilot import native_path

    path, model, out, device = sys.argv[1:]
    song_cache_folder(native_path(path))
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "4"))))
    torch.save(
        extract_l8(native_path(path), native_path(model), device), native_path(out)
    )
