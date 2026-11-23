"""MiaCode JSON-lines service for full charts and cursor continuation."""

import gc
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
from queue import Queue
import subprocess
import sys
import threading
import time

import numpy as np
import torch

from maidata.designer import DESIGNER_VOCAB
from maidata.flat import ARCHITECTURE
from maidata.flat.audio import extract_flat_audio_batch, flat_mel_frontend, frozen_mert
from .data import save_native_cache
from maidata.multitask.simai_writer import SimaiLineLayout, end_chart
from .editor import EditorChart, fragment_chunks, render_fragment
from .editor_prefix import prefix_actions
from maidata.flat.generate import generate, model_from_checkpoint


class _Numpy1CheckpointPickle:
    load = staticmethod(pickle.load)

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # Colab's NumPy 2 RNG state uses the renamed private module path.
            if module.startswith("numpy._core."):
                module = module.replace("numpy._core.", "numpy.core.", 1)
            return super().find_class(module, name)


def native_path(value):
    if value and os.name == "posix" and (value[1:2] == ":" or value.startswith("\\\\")):
        return subprocess.check_output(
            ["wslpath", "-a", "-u", value], text=True
        ).rstrip("\n")
    return value


def _ensure_standalone_division(text, division):
    """Ensure a standalone preview snippet declares its starting division after any leading BPM."""
    if not text or division is None:
        return text
    idx = 0
    n = len(text)
    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx < n and text[idx] == "(":
            close = text.find(")", idx + 1)
            if close < 0:
                break
            idx = close + 1
            continue
        break
    if idx < n and text[idx] == "{":
        return text
    return text[:idx] + f"{{{division}}}" + text[idx:]


class Copilot:
    def __init__(self, emit):
        self.emit = emit
        self.model = self.bank = self.audio_key = None
        self.device = torch.device("cpu")
        from .editor_stream import Stream

        self.stream = Stream(self)
        self.project = None

    def configure(self, req):
        self.model = self.bank = self.audio_key = None
        self.stream.clear(True)
        self.project = None
        if req.get("cache_dir"):
            from .project_cache import route_library_caches

            route_library_caches(Path(native_path(req["cache_dir"])))
        threads = max(1, int(req.get("cpu_threads", 4)))
        os.environ["OMP_NUM_THREADS"] = str(threads)
        torch.set_num_threads(threads)
        self.mert_command = req.get("mert_command") or []
        self.mert_python = req.get("mert_python", "")
        self.mert_device = req.get("mert_device", req.get("device", "cpu"))
        name = req.get("device", "auto")
        self.device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else name
        )
        self.precision = (
            req.get("precision", "bf16") if self.device.type == "cuda" else "fp32"
        )
        root = Path(native_path(req["models"]))
        self.mert = str(root / "mert-v2")
        architecture = ARCHITECTURE
        self.generate = generate
        vocab = DESIGNER_VOCAB
        ckpt = native_path(req.get("checkpoint"))
        if ckpt:
            pickle_module = _Numpy1CheckpointPickle if np.__version__.startswith("1.") else None
            state = torch.load(
                ckpt, map_location="cpu", weights_only=False, pickle_module=pickle_module
            )
            architecture = state["architecture"]
            package = Path(__file__).resolve().parents[1]
            if (
                architecture == "maidata.maimai-flat320.grid24-mel2d-mertv2l8.beatpat129.hetero5-uob.rule6.v18.5"
                and (package / "flat_v185").is_dir()
            ):
                from maidata.flat_v185.generate import generate as generate_v185
                from maidata.flat_v185.generate import model_from_checkpoint as load_v185

                self.model, _, _ = load_v185(state, self.device)
                self.generate = generate_v185
            elif (architecture == "maidata.maimai-flat320.grid24-mel2d-mertv2l8.hetero5-uob.rule6.v18.4"
                    and (package / "flat_v184").is_dir()):
                # The editor bundle ships the original v18.4 decoder for existing charts.
                from maidata.flat_v184.generate import generate as generate_v184
                from maidata.flat_v184.model import FlatConfig, FlatModel

                self.model = FlatModel(FlatConfig(**state["config"]["model"]))
                self.model.load_state_dict(state["model"])
                self.model.eval().to(self.device)
                self.generate = generate_v184
            else:
                self.model, _, _ = model_from_checkpoint(state, self.device)
            vocab = self.model.config.designer_vocab
        if self.generate is generate:
            from .project_cache import ProjectCache

            self.project = ProjectCache(self, ckpt)
        return dict(
            event="configured",
            ok=True,
            checkpoint_loaded=bool(ckpt),
            device=str(self.device),
            precision=self.precision,
            representation=architecture,
            ordered_steps=self.project is not None,
            cache_barrier=True,
            sync_chart=False,
            designer_vocab=list(vocab),
        )

    def place_model(self):
        if self.model is not None:
            actual = next(self.model.parameters()).device
            if actual.type != self.device.type or (
                self.device.index is not None and actual != self.device
            ):
                self.model.to(self.device)

    def autocast(self):
        return torch.autocast(
            "cuda",
            dtype=torch.float16 if self.precision == "fp16" else torch.bfloat16,
            enabled=self.device.type == "cuda" and self.precision != "fp32",
        )

    def audio(self, req, *, fuse=True):
        raw_audio = req.get("audio")
        if not raw_audio:
            return False
        path = Path(native_path(raw_audio))
        if not path.is_file():
            return False
        stat = path.stat()
        key = hashlib.sha256(
            f"v18.4-mel2d-mertv2l8|{path}|{stat.st_size}|{stat.st_mtime_ns}".encode()
        ).hexdigest()
        from .project_cache import song_cache_folder

        cache = song_cache_folder(path) / (key + ".pt")
        need_extract = not cache.is_file()
        need_bank = fuse and self.model is not None and (self.bank is None or key != self.audio_key)
        if not need_extract and not need_bank:
            return False
        if need_extract:
            self.emit(
                dict(
                    event="status",
                    state="encoding_audio",
                    message="正在分析音频...",
                )
            )
            feat = extract_flat_audio_batch([path], self.device, self.mert, bundled=True)
            save_native_cache(cache, feat)
        else:
            feat = torch.load(cache, map_location=self.device, weights_only=True)
        self.duration = int(feat["sample_lengths"][0]) / 44100
        if fuse and self.model is not None:
            self.place_model()
            with self.autocast():
                self.bank = self.model.fusion(feat)
            self.audio_key = key
        return need_extract

    def suggest(self, req, cancelled):
        mode = "full" if req.get("mode") == "full" else "continue"
        if self.audio(req):
            msg = (
                "正在流式生成整曲..."
                if mode == "full"
                else "正在续写..."
            )
            self.emit(dict(event="status", state="generating", message=msg))
        chart = EditorChart(req["text"], req.get("bpm", 120.0), req.get("first", 0.0))
        # Full and partial requests share exactly the same physical song clock.
        prefix, grid, meta = chart.history(0)
        end = math.floor(grid.second_to_beat(self.duration) * 96)
        span = chart.span(req, end + 1)
        if span.start:
            prefix, _, _ = chart.history(span.start)
        meta.update(
            level=req.get("level", 13),
            difficulty=req.get("difficulty", 5),
            charter=req.get("designer", ""),
        )
        meta.update(
            (key, req[key])
            for key in ("song_id", "version", "cabinet")
            if req.get(key) is not None
        )
        line_beats = req.get("line_align_beats", 4)
        full = mode == "full"

        def bpm_at(tick):
            return next(
                (v for t, v in sorted(chart.tempos.items(), reverse=True) if t <= tick),
                chart.base_bpm,
            )

        stream_interval = 2.0 if full else 0.08
        next_publish = time.monotonic() + (2.0 if full else 0.0)
        stream_layout = SimaiLineLayout()
        rendered_count = 0
        rendered_until = span.start
        parts = []
        last_stream_sig = (-1, -1)

        def step(tick, moments):
            nonlocal next_publish, rendered_count, rendered_until, last_stream_sig
            if cancelled.is_set():
                raise InterruptedError()
            now = time.monotonic()
            if now < next_publish:
                return
            if full:
                next_publish = now + stream_interval
                # Rows before the newest completed onset have a definitive duration.
                # Retain that last onset for the next prefix (or the final result).
                stop_index = len(moments) - 1
                message = dict(
                    event="stream_chunk",
                    id=req["id"],
                    revision=req.get("revision", 0),
                    mode="full",
                    beat=tick // 96,
                    total_beats=max(1, math.ceil(end / 96)),
                    progress=tick / max(1, end),
                )
                if stop_index > rendered_count:
                    stop = moments[stop_index][0]
                    parts.append(
                        render_fragment(
                            moments[rendered_count:stop_index],
                            grid,
                            rendered_until,
                            stop,
                            initial_tempo=not parts,
                            layout=stream_layout,
                        )
                    )
                    rendered_count, rendered_until = stop_index, stop
                    message["chart_text"] = end_chart("".join(parts))
                self.emit(message)
                return

            beat_tick = (tick // 96) * 96
            last_onset_end = ((moments[-1][0] // 96) + 1) * 96 if moments else span.start
            preview_stop = min(span.stop, max(beat_tick, last_onset_end))
            if preview_stop <= span.start:
                return
            sig = (len(moments), preview_stop)
            if sig == last_stream_sig:
                return
            last_stream_sig = sig
            next_publish = now + stream_interval
            preview_layout = SimaiLineLayout(
                division=span.prev_div, line_origin=span.line_origin, line_beats=line_beats
            )
            preview_intervals = list(
                fragment_chunks(
                    moments,
                    grid,
                    span.start,
                    preview_stop,
                    initial_tempo=(span.prev_bpm is None),
                    initial_bpm=span.prev_bpm,
                    layout=preview_layout,
                    quarter_tail=False,
                )
            )
            preview_fragment = "".join(chunk for _, _, chunk in preview_intervals)
            if not preview_fragment:
                return
            preview_text = span.lead + preview_fragment
            preview_ghost = (
                _ensure_standalone_division(preview_fragment, preview_intervals[0][1])
                if preview_intervals
                else preview_fragment
            )
            total_span = max(1, span.stop - span.start)
            self.emit(
                dict(
                    event="stream_chunk",
                    id=req["id"],
                    revision=req.get("revision", 0),
                    mode=mode,
                    text=preview_text,
                    ghost_simai=end_chart(preview_ghost) if preview_ghost else "",
                    replace_start=chart.u16[span.left],
                    replace_end=chart.u16[span.right],
                    start_second=grid.second(span.start),
                    end_second=grid.second(span.stop),
                    active_bpm=bpm_at(span.start),
                    note_count=sum(len(n) for _, n in moments),
                    beat=max(0, (preview_stop - span.start) // 96),
                    total_beats=max(1, math.ceil(total_span / 96)),
                    progress=min(1.0, max(0.0, (preview_stop - span.start) / total_span)),
                )
            )

        seed = req.get("seed", -1)
        if seed >= 0:
            torch.manual_seed(seed)
        else:
            torch.seed()
        from .generate import slide_sampling

        with self.autocast(), slide_sampling(req.get("slide_probability_multiplier", 1.0)):
            bank = self.bank
            if not req.get("conv_enabled", True):
                bank = bank._replace(mel_feat=torch.zeros_like(bank.mel_feat))
                if getattr(bank, "envelope", None) is not None:
                    bank = bank._replace(envelope=torch.zeros_like(bank.envelope))
            if not req.get("mert_enabled", True):
                bank = bank._replace(mert_feat=torch.zeros_like(bank.mert_feat))
            moments = (
                self.generate(
                    self.model,
                    bank,
                    grid,
                    meta,
                    end,
                    start_tick=span.start,
                    stop_tick=span.stop,
                    temperature=req.get("temperature", 1.0),
                    prefix_actions=prefix_actions(prefix),
                    on_step=step,
                    **(
                        {"plan_beat_te": req["plan_beat_te"]}
                        if self.generate is generate
                        and req.get("plan_beat_te") is not None
                        else {}
                    ),
                )
                if span.start < span.stop
                else []
            )
        export_stop = span.stop
        if full:
            last_onset = max((tick for tick, _ in moments), default=span.start)
            export_stop = max(grid.tick(self.duration), last_onset + 1)

        render_layout = (
            SimaiLineLayout()
            if full
            else SimaiLineLayout(division=span.prev_div, line_origin=span.line_origin, line_beats=line_beats)
        )
        intervals = list(
            fragment_chunks(
                moments,
                grid,
                span.start,
                export_stop,
                initial_tempo=True if full else (span.prev_bpm is None),
                initial_bpm=None if full else span.prev_bpm,
                layout=render_layout,
                quarter_tail=full,
            )
        )
        chunks = [text for _, _, text in intervals]
        fragment = "".join(chunks)
        end_bpm = (
            bpm_at(max(span.start, export_stop - 1))
            if export_stop > span.start
            else span.prev_bpm
        )
        padding = ""
        if not full and span.cell_start < span.cell_stop < len(chart.cells):
            next_tick = chart.cells[span.cell_stop].tick
            if next_tick > export_stop:
                padding = render_fragment([], grid, export_stop, next_tick, initial_tempo=False,
                                          initial_bpm=end_bpm, layout=render_layout)
                end_bpm = bpm_at(next_tick - 1)
        restore = (
            ""
            if full
            else chart.suffix_restore(
                span.cell_stop, end_bpm, render_layout.division, suffix_pos=span.right
            )
        )
        if chunks and not full:
            chunks[0] = span.lead + chunks[0]
            chunks[-1] += padding + restore
        text = end_chart(fragment) if full else "".join(chunks)
        ghost_fragment = (
            _ensure_standalone_division(fragment, intervals[0][1])
            if (fragment and not full and intervals)
            else fragment
        )
        steps = []
        if not full:
            for j, (finish, _, _) in enumerate(intervals):
                rest = "".join(chunks[j + 1 :])
                ghost = (
                    _ensure_standalone_division(rest, intervals[j + 1][1])
                    if rest
                    else ""
                )
                if j < len(intervals) - 1 and span.right > span.left:
                    k = next(
                        (
                            idx
                            for idx in range(span.cell_start, span.cell_stop)
                            if chart.cells[idx].tick >= finish
                        ),
                        span.cell_stop,
                    )
                    orig_cut_pos = chart.cells[k].pos if k < span.cell_stop else span.right
                    orig_cut_tick = chart.cells[k].tick if k < span.cell_stop else span.stop
                    step_bpm = bpm_at(max(span.start, finish - 1))
                    pad_layout = SimaiLineLayout(
                        division=intervals[j + 1][1], line_origin=span.line_origin, line_beats=line_beats
                    )
                    if orig_cut_tick > finish:
                        pad = render_fragment(
                            [],
                            grid,
                            finish,
                            orig_cut_tick,
                            initial_tempo=False,
                            initial_bpm=step_bpm,
                            layout=pad_layout,
                        )
                        pad_end_bpm = bpm_at(orig_cut_tick - 1)
                    else:
                        pad = ""
                        pad_end_bpm = step_bpm
                    bridge_restore = chart.suffix_restore(
                        k if k < span.cell_stop else span.cell_stop,
                        pad_end_bpm,
                        pad_layout.division,
                        suffix_pos=orig_cut_pos,
                    )
                    escape_bridge = pad + bridge_restore
                    step_replace_end = chart.u16[orig_cut_pos]
                else:
                    escape_bridge = ""
                    step_replace_end = chart.u16[span.right]
                steps.append(
                    dict(
                        accept_text=chunks[j],
                        escape_bridge=escape_bridge,
                        replace_end_u16=step_replace_end,
                        preview_remaining_text=rest,
                        remaining_ghost_simai=end_chart(ghost) if rest else "",
                        remaining_start_second=grid.second(finish),
                        remaining_bpm=bpm_at(finish),
                    )
                )
        return dict(
            event="result",
            ok=True,
            mode=mode,
            text=text,
            steps=steps,
            ghost_simai=end_chart(ghost_fragment) if ghost_fragment else "",
            replace_start=chart.u16[span.left],
            replace_end=chart.u16[span.right],
            start_second=grid.second(span.start),
            end_second=grid.second(span.stop),
            active_bpm=bpm_at(span.start),
            note_count=sum(len(n) for _, n in moments),
        )

    @torch.inference_mode()
    def handle(self, req, cancelled):
        op = req["op"]
        if op == "configure":
            return self.configure(req)
        if op == "cache_barrier":
            # Prior disk writes are complete. Keep live features and candidate KV.
            if self.project is not None:
                self.project.index = {}
            return dict(event="cache_barrier")
        if op == "discard":
            self.stream.clear()
            return None
        if op == "select":
            if self.stream.decoder and req.get("anchor", self.stream.anchor_id) == self.stream.anchor_id:
                self.stream.decoder.activate(req["candidate"])
            return None
        if (
            self.project
            and req.get("protocol", 0) >= 2
            and op in ("suggest", "bind_song", "warm", "more", "step_anchor", "refill")
        ):
            if op not in ("more", "refill", "step_anchor"):
                path = native_path(req.get("audio", ""))
                if not path or not Path(path).is_file():
                    if op == "suggest":
                        return dict(
                            event="result",
                            ok=False,
                            mode=req.get("mode"),
                            error="请先打开歌曲音频",
                        )
                    return None
                previous = self.project.audio_hash if self.project.path else None
                self.duration = self.project.bind(path, req)
                if previous != self.project.audio_hash:
                    self.stream.clear(True)
            with self.autocast():
                if op == "bind_song":
                    return dict(event="song_bound")
                if op == "warm":
                    if self.model is not None and req.get("valid_bpm", True):
                        self.place_model()
                        try:
                            self.stream.warm(req)
                            return dict(event="cache_updated")
                        except (ValueError, ZeroDivisionError):
                            # A partially typed timing tag will be retried by the next debounced edit.
                            return None
                    return None
                from maidata.decoder.projections import packed_projection_scope

                self.place_model()
                with packed_projection_scope():
                    return self.stream.run(req, cancelled)
        if op == "suggest":
            return self.suggest(req, cancelled)
        if op == "bind_song":
            self.audio(req, fuse=self.model is not None)
            return dict(event="song_bound")
        if op in ("release_vram", "clear_cache"):
            self.bank = self.audio_key = None
            self.stream.clear(True)
            if self.project is not None:
                self.project.path = self.project.l1 = self.project.l2 = (
                    self.project.l3
                ) = None
            if op == "release_vram":
                if self.model is not None:
                    self.model.cpu()
                frozen_mert.cache_clear()
                flat_mel_frontend.cache_clear()
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        # The editor sends the authoritative current text with every suggestion.
        return None


def main():
    if os.name == "nt":
        # The Windows OpenBLAS DLL must initialize before the JSON worker and
        # its blocking pipe reader start. Transformers imports SciPy lazily.
        from scipy.linalg import blas  # noqa: F401
    output, sys.stdout = sys.stdout, sys.stderr

    def emit(message):
        print(json.dumps(message, ensure_ascii=False), file=output, flush=True)

    service = Copilot(emit)
    queue = Queue()
    pending = {}
    lock = threading.Lock()

    def worker():
        while (item := queue.get()) is not None:
            req, cancelled = item
            rid = req.get("id", 0)
            try:
                if cancelled.is_set():
                    raise InterruptedError()
                result = service.handle(req, cancelled)
                if cancelled.is_set():
                    raise InterruptedError()
            except InterruptedError:
                result = dict(event="cancelled")
            except Exception as error:
                import traceback

                traceback.print_exc()
                result = dict(
                    event={"configure": "configured", "suggest": "result"}.get(
                        req["op"], "status"
                    ),
                    state="error",
                    mode=req.get("mode"),
                    ok=False,
                    error=str(error),
                    message=str(error),
                )
            if result is not None:
                emit(dict(result, id=rid, revision=req.get("revision", 0)))
            with lock:
                pending.pop(rid, None)

    thread = threading.Thread(target=worker)
    thread.start()
    closing = False
    last_warm = None
    for line in sys.stdin:
        req = json.loads(line)
        if req["op"] == "close":
            closing = True
            break
        with lock:
            if req["op"] == "cancel":
                token = pending.get(req["id"])
                if token:
                    token.set()
            else:
                if (
                    req["op"] in ("warm", "suggest", "step_anchor", "refill", "bind_song", "configure")
                    and last_warm is not None
                ):
                    last_warm.set()
                    last_warm = None
                token = threading.Event()
                if req["op"] == "warm":
                    last_warm = token
                pending[req.get("id", 0)] = token
                queue.put((req, token))
    # EOF can follow a piped request batch. Explicit close cancels in-flight work.
    if closing:
        with lock:
            for token in pending.values():
                token.set()
    queue.put(None)
    thread.join()


if __name__ == "__main__":
    main()
