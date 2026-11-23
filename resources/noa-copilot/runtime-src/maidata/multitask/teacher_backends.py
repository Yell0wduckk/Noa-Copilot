"""Explicit local adapter for an official YourMT3+ transcription checkpoint.

Dependencies and weights are loaded only by the factories, once per job. No
Gradio application, downloader, training launcher, or remote API is invoked.
Interfaces follow the official sources:
https://huggingface.co/spaces/mimbres/YourMT3/blob/main/model_helper.py
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import importlib
import importlib.util
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import traceback

import torch


@contextmanager
def _local_imports(paths):
    previous = sys.path[:]
    keys = ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')
    environment = {key: os.environ.get(key) for key in keys}
    sys.path[:0] = [str(Path(path).resolve()) for path in paths]
    os.environ.update({key: '1' for key in keys})
    try:
        yield
    finally:
        sys.path[:] = previous
        for key, value in environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _load_audio(path, sample_rate):
    import librosa
    waveform, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    return torch.from_numpy(waveform)


def _backend_worker(connection, factory, options, paths):
    # Official projects use generic package names (utils/config/dataset).
    # Keep each in its own interpreter, including every deferred import.
    with _local_imports(paths):
        try:
            module, function = factory.split(':', 1)
            backend = getattr(importlib.import_module(module), function)(**options)
            connection.send((True, getattr(backend, 'provenance', {})))
        except Exception:
            connection.send((False, traceback.format_exc()))
            connection.close()
            return
        try:
            while True:
                request = connection.recv()
                if request is None:
                    break
                try:
                    backend(*request)
                    connection.send((True, request[1]))
                except Exception:
                    connection.send((False, traceback.format_exc()))
        except EOFError:
            pass
        finally:
            connection.close()


class _ProcessBackend:
    """One reusable model process; only filenames cross the process boundary."""
    def __init__(self, factory, options, paths, *, result_kind):
        context = multiprocessing.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_backend_worker,
            args=(child, factory, options, paths), daemon=True)
        self.result_kind = result_kind
        self.process.start()
        child.close()
        try:
            self.provenance = self._receive()
        except Exception:
            self.close()
            raise

    def _receive(self):
        success, value = self.connection.recv()
        if not success:
            raise RuntimeError(value)
        return value

    def __call__(self, audio_path, output_path):
        self.connection.send((str(Path(audio_path).resolve()), str(Path(output_path).resolve())))
        path = Path(self._receive())
        return torch.load(path, map_location='cpu', weights_only=True) if self.result_kind == 'structure' else path

    def close(self):
        if not self.connection.closed:
            try:
                self.connection.send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass
            self.connection.close()
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class YourMT3Backend:
    def __init__(self, model, helper, *, device, batch_size, provenance):
        self.model, self.helper = model, helper
        self.device, self.batch_size = device, batch_size
        self.provenance = provenance

    @torch.inference_mode()
    def __call__(self, audio_path, output_path):
        import torchaudio
        model, helper = self.model, self.helper
        waveform, sample_rate = torchaudio.load(str(audio_path))
        audio = torchaudio.functional.resample(waveform.mean(0, keepdim=True),
                                              sample_rate, model.audio_cfg['sample_rate'])
        window = model.audio_cfg['input_frames']
        segments = helper.slice_padded_array(audio.numpy(), window, window)
        segments = torch.as_tensor(segments, dtype=torch.float32, device=self.device).unsqueeze(1)
        predictions, _ = model.inference_file(bsz=self.batch_size, audio_segments=segments)
        starts = [window * i / model.audio_cfg['sample_rate'] for i in range(len(segments))]
        channels = []
        for channel in range(model.task_manager.num_decoding_channels):
            tokens = [batch[:, channel, :] for batch in predictions]
            zipped, _, _ = model.task_manager.detokenize_list_batches(tokens, starts, return_events=True)
            notes, _ = helper.merge_zipped_note_events_and_ties_to_notes(zipped)
            channels.append(notes)
        notes = helper.mix_notes(channels)
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='yourmt3-', dir=output_path.parent) as temporary:
            helper.write_model_output_as_midi(notes, temporary, output_path.stem, model.midi_output_inverse_vocab)
            (Path(temporary) / 'model_output' / (output_path.stem + '.mid')).replace(output_path)
        return output_path


def _load_yourmt3(*, repo, checkpoint, model_args=None, device='cuda', batch_size=8):
    """Load the official HF Space checkout, with an explicit local checkpoint.

    model_args contains architecture flags after exp_id. The default is the
    transcription model's arguments. Training encoder initialization uses the
    standalone 8-layer T5 loader and never depends on this transcription helper.
    """
    repo, checkpoint = Path(repo).resolve(), Path(checkpoint).resolve()
    flags = list(model_args or ())
    with _local_imports([repo / 'amt' / 'src', repo]):
        spec = importlib.util.spec_from_file_location('_maidata_official_yourmt3_helper', repo / 'model_helper.py')
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        default_shared = importlib.import_module('config.config').shared_cfg
        # The official helper otherwise constructs a Lightning trainer merely
        # to derive a checkpoint path. Supply that path directly, with no logs.
        original = helper.initialize_trainer
        def local_checkpoint(args, stage='test'):
            return None, None, dict(last_ckpt_path=str(checkpoint),
                                    lightning_dir=str(checkpoint.parent)), deepcopy(default_shared)
        helper.initialize_trainer = local_checkpoint
        try:
            model = helper.load_model_checkpoint(args=['local', *flags], device=device)
        finally:
            helper.initialize_trainer = original
        return YourMT3Backend(model, helper, device=device, batch_size=batch_size,
            provenance=dict(teacher='YourMT3+', repo=str(repo), checkpoint=str(checkpoint),
                            model_args=flags))


def make_yourmt3(*, repo, checkpoint, model_args=None, device='cuda', batch_size=8):
    """Load the official HF Space checkout once in its own interpreter."""
    options = dict(repo=repo, checkpoint=checkpoint, model_args=model_args,
                   device=device, batch_size=batch_size)
    repo = Path(repo).resolve()
    return _ProcessBackend(__name__ + ':_load_yourmt3', options,
                           [repo / 'amt' / 'src', repo], result_kind='midi')
