"""Independent audio/event BPM and tick quantizer.

The quantizer owns its audio weights and reads shared 512-column Log-Mel
cache. It never loads the main encoder. Known, fixed and variable BPM modes
share one network and the same three-candidate tick selector.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from copy import deepcopy
from pathlib import Path
import argparse
import hashlib
import json
import math
import random
import subprocess
from bisect import bisect_right

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import AUDIO_SCHEMA
from .audio import LOG_POWER_FLOOR, LogMelFrontend, adapt_legacy_features, unpack_feature_payload


@dataclass(frozen=True)
class TempoSegment:
    start_seconds: float
    bpm: float
    offset_seconds: float = 0.0
    beats_per_bar: float = 4


@dataclass
class QuantizerConfig:
    dim: int = 256
    heads: int = 8
    layers: int = 4
    ffn_dim: int = 1024
    dropout: float = 0.0
    bpm_min: float = 40.0
    bpm_max: float = 300.0
    bpm_bins: int = 512
    ticks_per_beat: int = 96
    segment_seconds: float = 8.0


def _position(values: torch.Tensor, width: int) -> torch.Tensor:
    scales = torch.exp(torch.arange(width // 2, device=values.device,
                                   dtype=torch.float32) * (-math.log(10000) / (width // 2)))
    angles = values.float()[..., None] * scales
    return torch.cat((angles.sin(), angles.cos()), -1)


def candidate_ticks(event_frames: torch.Tensor, segments: list[TempoSegment],
                    ticks_per_beat: int = 96) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return three adjacent tick candidates and the owning tempo segment.

    Inputs use the common 10ms index; outputs are integer tick IDs relative to
    each segment's beat-zero offset and their physical second coordinates.
    """
    seconds = event_frames.double() * .01
    starts = seconds.new_tensor([s.start_seconds for s in segments])
    owner = torch.searchsorted(starts, seconds, right=True).sub(1).clamp_min(0)
    if isinstance(segments, CalibratorSegments):
        grid = segments.grid
        centers = (grid.second_to_beat_tensor(seconds)*ticks_per_beat).round().long()
        ticks = centers[:, None] + torch.arange(-1, 2, device=seconds.device)
        return ticks, grid.beat_to_second_tensor(ticks/ticks_per_beat).double(), owner
    bpms = seconds.new_tensor([s.bpm for s in segments])[owner]
    offsets = seconds.new_tensor([s.offset_seconds for s in segments])[owner]
    centers = ((seconds - offsets) * bpms * ticks_per_beat / 60).round().long()
    ticks = centers[:, None] + torch.arange(-1, 2, device=seconds.device)
    candidates = offsets[:, None] + ticks.double() * 60 / (bpms[:, None] * ticks_per_beat)
    return ticks, candidates, owner


class TickQuantizer(nn.Module):
    def __init__(self, config: QuantizerConfig | dict | None = None):
        super().__init__()
        self.config = config if isinstance(config, QuantizerConfig) else QuantizerConfig(**(config or {}))
        c = self.config
        self.frontend = LogMelFrontend(persistent=False,frame_hz=100.)
        self.audio_projection = nn.Linear(512, c.dim)
        layer = nn.TransformerEncoderLayer(c.dim, c.heads, c.ffn_dim, c.dropout,
                                           activation='gelu', batch_first=True, norm_first=True)
        self.audio_encoder = nn.TransformerEncoder(layer, c.layers, norm=nn.LayerNorm(c.dim),
                                                   enable_nested_tensor=False)
        self.event_projection = nn.Linear(4, c.dim)
        self.event_attention = nn.MultiheadAttention(c.dim, c.heads, c.dropout, batch_first=True)
        self.event_norm = nn.LayerNorm(c.dim)
        self.tempo_head = nn.Linear(c.dim * 2, c.bpm_bins)
        self.offset_head = nn.Linear(c.dim * 2, 2)
        self.candidate_projection = nn.Sequential(nn.Linear(5, c.dim), nn.GELU(), nn.Linear(c.dim, 1))
        self.event_selector = nn.Linear(c.dim, 3)
        self.register_buffer('log_bpms', torch.linspace(math.log(c.bpm_min), math.log(c.bpm_max), c.bpm_bins))

    def encode(self, features: torch.Tensor, event_frames: torch.Tensor,
               event_durations: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # One unpadded song per call; no repeated main-encoder computation.
        features = adapt_legacy_features(features)
        extra = (-features.shape[0]) % 4
        audio = F.pad(features, (0, 0, 0, extra)).reshape(-1, 4, 512).sum(1)
        counts = features.new_full((len(audio), 1), 4)
        if extra:
            counts[-1] = 4 - extra
        audio = self.audio_projection(audio / counts)
        positions = torch.arange(len(audio), device=audio.device) * 4 + 1.5
        audio = self.audio_encoder((audio + _position(positions, self.config.dim))[None])[0]
        duration = torch.zeros_like(event_frames) if event_durations is None else event_durations
        gaps = F.pad(torch.diff(event_frames), (1, 0)) if len(event_frames) else event_frames
        event_data = torch.stack((event_frames / 100, duration / 100, gaps / 100,
                                  torch.ones_like(event_frames)), -1).to(audio.dtype)
        queries = self.event_projection(event_data) + _position(event_frames, self.config.dim)
        context, _ = self.event_attention(queries[None], audio[None], audio[None], need_weights=False)
        events = self.event_norm(queries + context[0])
        return audio, events

    def tempo_outputs(self, audio: torch.Tensor, events: torch.Tensor,
                      event_frames: torch.Tensor, mode: str = 'fixed') -> dict:
        duration = len(audio) * .04
        starts = ([0.0] if mode != 'variable' else
                  [i * self.config.segment_seconds for i in range(max(1, math.ceil(duration / self.config.segment_seconds)))])
        summaries = []
        for i, start in enumerate(starts):
            stop = starts[i + 1] if i + 1 < len(starts) else duration + 1
            lo, hi = int(start / .04), min(len(audio), math.ceil(stop / .04))
            am = audio[lo:hi].mean(0)
            selected = (event_frames >= start * 100) & (event_frames < stop * 100)
            em = events[selected].mean(0) if bool(selected.any()) else torch.zeros_like(am)
            summaries.append(torch.cat((am, em)))
        summaries = torch.stack(summaries)
        return dict(starts=starts, bpm_logits=self.tempo_head(summaries),
                    offset_vectors=self.offset_head(summaries))

    def predicted_segments(self, outputs: dict) -> list[TempoSegment]:
        bpms = self.log_bpms[outputs['bpm_logits'].argmax(-1)].exp().detach().cpu().tolist()
        vectors = outputs['offset_vectors'].detach().float()
        phases = torch.atan2(vectors[:, 0], vectors[:, 1]).remainder(2 * math.pi) / (2 * math.pi)
        return [TempoSegment(start, bpm, start + float(phase) * 60 / bpm)
                for start, bpm, phase in zip(outputs['starts'], bpms, phases.cpu())]

    def forward(self, features: torch.Tensor, event_frames: torch.Tensor, *,
                mode: str = 'known', segments: list[TempoSegment] | None = None,
                event_durations: torch.Tensor | None = None) -> dict:
        audio, events = self.encode(features, event_frames, event_durations)
        tempo = self.tempo_outputs(audio, events, event_frames, mode)
        if mode != 'known':
            segments = self.predicted_segments(tempo)
        return dict(**tempo, **self.select(events, event_frames, segments))

    def select(self, events, event_frames, segments):
        ticks, seconds, owner = candidate_ticks(event_frames, segments, self.config.ticks_per_beat)
        bpms = seconds.new_tensor([s.bpm for s in segments])[owner]
        beat = ticks.double() / self.config.ticks_per_beat
        candidate_features = torch.stack(((seconds-event_frames[:, None]*.01)*100,
                                         beat.remainder(1), (beat*2*math.pi).sin(),
                                         (beat*2*math.pi).cos(), bpms[:, None].expand_as(seconds)/300), -1)
        logits = self.event_selector(events) + self.candidate_projection(candidate_features.to(events.dtype)).squeeze(-1)
        selected = logits.argmax(-1)
        chosen_seconds = seconds.gather(-1, selected[:, None])[:, 0]
        return dict(segments=segments, candidate_ticks=ticks, candidate_seconds=seconds,
                    segment_indices=owner, tick_logits=logits, seconds=chosen_seconds,
                    ticks=ticks.gather(-1, selected[:, None])[:, 0])

    def loss(self, features: torch.Tensor, event_frames: torch.Tensor, original_seconds: torch.Tensor,
             segments: list[TempoSegment], *, variable: bool = False,
             event_durations: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        # Training supplies the true grid to the selector while training the
        # shared tempo heads independently; no discrete argmax gradient bridge.
        audio, events = self.encode(features, event_frames, event_durations)
        tempo = self.tempo_outputs(audio, events, event_frames, 'variable' if variable else 'fixed')
        ticks, seconds, owner = candidate_ticks(event_frames, segments, self.config.ticks_per_beat)
        bpms = seconds.new_tensor([s.bpm for s in segments])[owner]
        beat = ticks.double() / self.config.ticks_per_beat
        candidates = torch.stack(((seconds-event_frames[:, None]*.01)*100, beat.remainder(1),
                                  (beat*2*math.pi).sin(), (beat*2*math.pi).cos(),
                                  bpms[:, None].expand_as(seconds)/300), -1)
        logits = self.event_selector(events) + self.candidate_projection(candidates.to(events.dtype)).squeeze(-1)
        nearest = (seconds-original_seconds[:, None]).abs().argmin(-1)
        tick_loss = F.cross_entropy(logits.float(), nearest) if len(logits) else events.sum()*0
        targets, phases = [], []
        for start in tempo['starts']:
            segment = next((s for s in reversed(segments) if s.start_seconds <= start), segments[0])
            targets.append((self.log_bpms-math.log(segment.bpm)).abs().argmin())
            phase = ((segment.offset_seconds-start)*segment.bpm/60) % 1
            phases.append((math.sin(phase*2*math.pi), math.cos(phase*2*math.pi)))
        tempo_loss = F.cross_entropy(tempo['bpm_logits'].float(), torch.stack(targets))
        offset_loss = F.mse_loss(F.normalize(tempo['offset_vectors'].float(), dim=-1),
                                 tempo['offset_vectors'].new_tensor(phases).float())
        loss = tick_loss + tempo_loss + offset_loss
        return loss, dict(tick_ce=float(tick_loss.detach()), bpm_ce=float(tempo_loss.detach()),
                          offset_loss=float(offset_loss.detach()),
                          candidate_error_ms=float((seconds-original_seconds[:, None]).abs().min(-1).values.mean()*1000) if len(seconds) else 0.)


def beat_this_segments(command: list[str], audio_path: str | Path) -> list[TempoSegment]:
    """Optional external Beat This! adapter, emitting JSON beats or segments.

    No shell expansion is used. The executable receives the audio path as its
    final argument and returns {"segments": [...]} or {"beats": [seconds...]}.
    """
    output = subprocess.run([*command, str(audio_path)], check=True, capture_output=True, text=True)
    value = json.loads(output.stdout)
    if 'segments' in value:
        return [TempoSegment(**s) for s in value['segments']]
    beats = np.asarray(value['beats'], dtype=np.float64)
    periods = np.diff(beats)
    if len(periods) == 0:
        return []
    # Beat timestamps are the adapter's grid; preserve each observed tempo.
    return [TempoSegment(float(beat), float(60/period), float(beat))
            for beat, period in zip(beats[:-1], periods) if period > 0]


class CalibratorSegments(list):
    """TempoSegment sequence with the full (possibly STOP-bearing) decoded audio timing."""
    def __init__(self, segments, metadata, decoded_grid=None):
        super().__init__(segments)
        from .grid import BeatGrid
        self.corrected_metadata = metadata
        self.decoded_grid = decoded_grid
        self.grid = BeatGrid(metadata)


@torch.no_grad()
def bms_calibrator_segments(calibrator, features_or_waveform, *, is_waveform=False,
                            metadata=None, mert=None, song_ref_db=None,
                            mert_extractor=None, mert_model=None):
    from .bms_calibrator import (
        annotate_known_grid_bars, decode_full_song_grid, load_calibrator_checkpoint, shift_timing_metadata,
    )
    from .grid import BeatGrid
    if isinstance(calibrator, (str, Path)):
        calibrator, _ = load_calibrator_checkpoint(calibrator, device=features_or_waveform.device)
    device = next(calibrator.parameters()).device
    inputs = features_or_waveform.to(device)
    mert = mert.to(device) if mert is not None else None
    was_training = calibrator.training
    try:
        calibrator.eval()
        if is_waveform:
            waveform_kwargs = dict(mert=mert, song_ref_db=song_ref_db)
            if mert_extractor is not None:
                waveform_kwargs["mert_extractor"] = mert_extractor
            if mert_model is not None:
                waveform_kwargs["mert_model"] = mert_model
            out = calibrator.forward_waveforms(inputs, **waveform_kwargs)
        else:
            if getattr(getattr(calibrator, "config", None), "stage", 1) == 2 and mert is None:
                raise ValueError(
                    "Stage 2 calibrator requires `mert` when `is_waveform=False`."
                )
            out = calibrator(inputs.unsqueeze(0) if inputs.ndim == 3 else inputs, mert=mert)
    finally:
        calibrator.train(was_training)
    decoded = decode_full_song_grid(out)
    timing = {k: decoded[k] for k in (
        'bpm', 'offset_seconds', 'bpm_segments', 'stops',
        'meter_segments', 'numerator', 'denominator', 'bar_beats',
    ) if k in decoded}
    if metadata:
        has_clock = any(k in metadata for k in ('offset_seconds', 'bpm_segments', 'stops'))
        has_full_clock = has_clock and ('bpm' in metadata or 'bpm_segments' in metadata)
        has_meter = any(k in metadata for k in ('meter_segments', 'measure_multipliers', 'bar_beats', 'numerator'))
        if has_full_clock:
            timing = dict(metadata)
            if not has_meter:
                annotated = annotate_known_grid_bars(out, metadata)
                timing['meter_segments'] = annotated['meter_segments']
                _, timing['numerator'], timing['denominator'] = annotated['meter_segments'][0]
                meter_grid = BeatGrid(timing)
                bars = annotated['bar_beats']
                for a, b in zip(bars, bars[1:]):
                    index = max(0, bisect_right(meter_grid.meter_starts, a) - 1)
                    if abs(b - a - meter_grid.meter_spans[index]) > 1e-4:
                        timing['bar_beats'] = bars
                        break
        else:
            meter_hints = {k: metadata[k] for k in (
                'meter_segments', 'numerator', 'denominator', 'measure_multipliers', 'bar_beats'
            ) if k in metadata}
            if meter_hints:
                timing.pop('bar_beats', None)
            if meter_hints and 'meter_segments' not in meter_hints:
                timing.pop('meter_segments', None)
            if has_clock:
                if 'offset_seconds' in metadata:
                    timing = shift_timing_metadata(timing, metadata['offset_seconds'] - timing['offset_seconds'])
                timing.update(metadata)
            else:
                timing.update(meter_hints)
    grid = BeatGrid(timing)
    marks = set(grid.beats) | set(grid.meter_starts)
    bars = grid.bar_beats_explicit
    if len(bars) >= 2:
        previous = grid.meter_spans[max(0, bisect_right(grid.meter_starts, bars[0]) - 1)]
        for a, b in zip(bars, bars[1:]):
            if abs(b - a - previous) > 1e-4:
                marks.add(a)
            previous = b - a
        if abs(previous - grid.meter_spans[max(0, bisect_right(grid.meter_starts, bars[-1]) - 1)]) > 1e-4:
            marks.add(bars[-1])
    segments = []
    for beat in sorted(marks):
        sec = grid.beat_to_second(beat)
        bpm = float(grid.tempo_info(beat)['bpm'])
        span = float(grid.bar_info(beat)['bar_beats'])
        segments.append(TempoSegment(sec, bpm, sec-beat*60/bpm, span))
    return CalibratorSegments(segments, timing, decoded)


def continuous_segments(segments: list[TempoSegment]) -> list[TempoSegment]:
    """Resolve segment beat-zero offsets onto one continuous beat coordinate."""
    result = []
    for segment in segments:
        if not result:
            result.append(segment)
        else:
            prior = result[-1]
            beat = (segment.start_seconds-prior.offset_seconds)*prior.bpm/60
            result.append(TempoSegment(segment.start_seconds, segment.bpm,
                                       segment.start_seconds-beat*60/segment.bpm,
                                       segment.beats_per_bar))
    return result


def tempo_segments_from_metadata(raw, offset_seconds=0.) -> list[TempoSegment]:
    """Derive only missing offsets; explicit local phases remain authoritative."""
    entries = sorted(raw, key=lambda s: s.get('start_seconds', 0.) if isinstance(s, dict) else s[0])
    result = []
    for value in entries:
        if isinstance(value, dict):
            segment = TempoSegment(float(value.get('start_seconds', 0.)), float(value['bpm']),
                                   float(value.get('offset_seconds', offset_seconds)),
                                   int(value.get('beats_per_bar', 4)))
            explicit = 'offset_seconds' in value
        else:
            segment = TempoSegment(float(value[0]), float(value[1]), float(offset_seconds))
            explicit = False
        if result and not explicit:
            segment = continuous_segments([result[-1], segment])[-1]
        result.append(segment)
    return result


def event_queries(events):
    """Return seconds, remaining durations and bindings for all quantized points.

    An onset carries its complete span, a slide movement start carries its
    movement duration, and terminal endpoints carry zero. Both production and
    inference use this layout; tensor duration units are 10ms frames.
    """
    positions, durations, bindings = [], [], []
    for i, event in enumerate(events):
        branches = event.data.get('branches', [event.data]) if event.kind == 'slide' else []
        duration = max((b.get('wait', .5) + b.get('duration', 1.) for b in branches), default=0.) if branches else event.data.get('duration', 0.)
        positions.append(event.time); durations.append(float(duration or 0.)); bindings.append((i, 'onset', -1))
        if event.kind in ('hold', 'touch_hold', 'note') and event.data.get('duration') is not None:
            positions.append(event.time + event.data['duration']); durations.append(0.); bindings.append((i, 'end', -1))
        for b, branch in enumerate(branches):
            wait, movement = branch.get('wait', .5), branch.get('duration', 1.)
            positions.extend((event.time + wait, event.time + wait + movement))
            durations.extend((movement, 0.))
            bindings.extend(((i, 'wait', b), (i, 'slide_end', b)))
    return positions, durations, bindings


@torch.inference_mode()
def quantize_events(model: TickQuantizer, features: torch.Tensor, events, *,
                    mode='known', segments=None):
    """Quantize onsets and every hold/slide endpoint using one independent model."""
    from .codec import Event
    model.eval()
    device = next(model.parameters()).device
    source = [Event(e.time, e.kind, deepcopy(e.data)) for e in events]
    positions, durations, bindings = event_queries(source)
    frames = torch.tensor(positions, device=device, dtype=torch.float32) * 100
    durations = torch.tensor(durations, device=device, dtype=torch.float32) * 100
    order = frames.argsort(stable=True)
    sorted_frames = frames[order]
    features = features.to(device)
    audio, hidden = model.encode(features, sorted_frames, durations[order])
    if mode != 'known':
        tempo = model.tempo_outputs(audio, hidden, sorted_frames, mode)
        segments = model.predicted_segments(tempo)
    # Reuse this single audio/event pass for tempo estimation and tick selection.
    output = model.select(hidden, sorted_frames, segments)
    corrected = torch.empty_like(output['seconds'])
    corrected[order] = output['seconds']
    decoded = {}
    for binding, value in zip(bindings, corrected.cpu().tolist()):
        decoded[binding] = value
    result = []
    for i, event in enumerate(source):
        onset = decoded[i,'onset',-1]
        data = event.data
        if (i,'end',-1) in decoded:
            data['duration'] = max(.001, decoded[i,'end',-1]-onset)
        if event.kind == 'slide':
            for b, branch in enumerate(data.get('branches',[data])):
                waiting = decoded[i,'wait',b]
                ending = decoded[i,'slide_end',b]
                branch['wait'] = max(.001, waiting-onset)
                branch['duration'] = max(.001, ending-waiting)
        result.append(Event(onset,event.kind,data))
    return result, segments


def prepare_targets(source_manifest, output_manifest, *, seed=20261104):
    """Build quantizer supervision from original event seconds and BPM metadata.

    Feature payloads are referenced in place and never decoded or recomputed.
    Unknown grids and held-out rows are counted, not made into startup gates.
    """
    from .codec import Event, quantize_time
    from .data import read_manifest
    rows, binding = read_manifest(source_manifest)
    output_manifest = Path(output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    report = dict(source=binding, written=0, no_bpm=0, held_out=0, events=0)
    with output_manifest.open('w', encoding='utf-8') as stream:
        for row in rows:
            if row.get('split','train') != 'train':
                report['held_out'] += 1
                continue
            metadata = row.get('metadata', {})
            raw = metadata.get('bpm_segments')
            if not raw and metadata.get('bpm'):
                raw = [(0., metadata['bpm'])]
            if not raw:
                report['no_bpm'] += 1
                continue
            segments = tempo_segments_from_metadata(raw, metadata.get('offset_seconds', 0.))
            values = row.get('event_values')
            if values is None:
                values = json.loads(Path(row['events']).read_text('utf-8-sig'))
            if isinstance(values,dict):
                values = values['events']
            seconds, durations, _ = event_queries([Event(**value) for value in values])
            order = sorted(range(len(seconds)), key=seconds.__getitem__)
            seconds, durations = [seconds[i] for i in order], [durations[i] for i in order]
            target = dict(id=row['id'], task=row['task'],
                          original_seconds=seconds, event_frames=[quantize_time(s,rng) for s in seconds],
                          event_durations=[quantize_time(d,rng) for d in durations],
                          segments=[asdict(s) for s in segments])
            target.update({key: row[key] for key in ('features', 'features_schema', 'audio') if row.get(key)})
            stream.write(json.dumps(target,ensure_ascii=False)+'\n')
            report['written'] += 1
            report['events'] += len(seconds)
    output_manifest.with_suffix(output_manifest.suffix+'.report.json').write_text(
        json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


class QuantizerDataset(torch.utils.data.Dataset):
    def __init__(self, manifest: str | Path):
        self.path = Path(manifest)
        self.rows = [json.loads(line) for line in self.path.read_text('utf-8').splitlines() if line.strip()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = dict(self.rows[index])
        if row.get('audio'):
            from .audio import read_waveform
            path = Path(row['audio'])
            row['waveform'] = read_waveform(path if path.is_absolute() else self.path.parent/path)
            row.pop('features', None)
        else:
            path = Path(row['features'])
            path = path if path.is_absolute() else self.path.parent/path
            saved = torch.from_numpy(np.load(path).astype(np.float32)) if path.suffix=='.npy' else torch.load(path, map_location='cpu', weights_only=True)
            row['features'], row['features_schema'] = unpack_feature_payload(saved, row.get('features_schema'))
        row['event_frames'] = torch.tensor(row['event_frames'], dtype=torch.float32)
        row['event_durations'] = torch.tensor(row.get('event_durations', [0.] * len(row['event_frames'])), dtype=torch.float32)
        row['original_seconds'] = torch.tensor(row['original_seconds'], dtype=torch.float64)
        row['segments'] = [TempoSegment(**s) for s in row['segments']]
        return row


def _single_sample(samples):
    return samples[0]


def checkpoint_features(features, checkpoint):
    return features


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-source', type=Path)
    parser.add_argument('--output-manifest', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--epochs', type=int, default=75)
    parser.add_argument('--schedule-epochs', type=int, default=150)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--warmup-steps', type=int, default=2000)
    args = parser.parse_args(argv)
    if args.prepare_source:
        output = args.output_manifest or args.prepare_source.with_name('quantizer-targets.jsonl')
        print(json.dumps(prepare_targets(args.prepare_source,output,seed=args.seed),ensure_ascii=False))
        return 0
    if args.manifest is None or args.output is None:
        parser.print_help()
        return 0
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dataset = QuantizerDataset(args.manifest)
    model = TickQuantizer().to(args.device)
    from .optim import make_optimizer
    optimizer = make_optimizer(model, dict(optimizer='adamw', learning_rate=2e-4,
                                         betas=(.9,.95), weight_decay=.01))
    loader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=True,num_workers=args.workers,collate_fn=_single_sample)
    horizon=args.schedule_epochs*max(1,len(loader))
    def learning_rate(step):
        if step < args.warmup_steps:
            return (step+1)/max(1,args.warmup_steps)
        progress=min(1.,(step-args.warmup_steps)/max(1,horizon-args.warmup_steps))
        return .5*(1+math.cos(math.pi*progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,learning_rate)
    from maidata.joint_training.checkpoint import source_identity
    binding = dict(config=asdict(model.config),
                   manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                   schedule_epochs=args.schedule_epochs, steps_per_epoch=len(loader), seed=args.seed,
                   warmup_steps=args.warmup_steps)
    start = 0
    if args.resume:
        state = torch.load(args.resume,map_location=args.device,weights_only=False)
        if state['binding'] != binding:
            raise ValueError('quantizer complete-epoch inputs or training recipe differ')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        start = state['completed_epochs']
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'].cpu())
        if torch.cuda.is_available() and state['cuda_rng'] is not None: torch.cuda.set_rng_state_all(state['cuda_rng'])
    args.output.mkdir(parents=True,exist_ok=True)
    for epoch in range(start,args.epochs):
        model.train();metrics=[]
        for row in loader:
            frames=row['event_frames'].to(args.device)
            durations=row['event_durations'].to(args.device)
            features=(adapt_legacy_features(row['features'].to(args.device), row['features_schema']) if row.get('features') is not None else
                      model.frontend(row['waveform'].to(args.device)))
            originals=row['original_seconds'].to(args.device)
            shift=random.randint(0,100)
            if shift:
                features=F.pad(features,(0,0,shift,0))
                frames=frames+shift
                originals=originals+shift*.01
            segments=[TempoSegment(max(0,s.start_seconds+shift*.01),s.bpm,s.offset_seconds+shift*.01,s.beats_per_bar)
                      for s in row['segments']]
            # Simulate event timing error without changing original supervision.
            frames=frames+torch.randint(-2,3,frames.shape,device=frames.device)
            optimizer.zero_grad(set_to_none=True)
            loss,diagnostics=model.loss(features,frames,originals,segments,variable=len(segments)>1,
                                        event_durations=durations)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();scheduler.step();metrics.append(diagnostics)
        state=dict(schema='maidata.tick-quantizer.v1',audio_schema=AUDIO_SCHEMA,config=asdict(model.config),model=model.state_dict(),
                   optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),completed_epochs=epoch+1,
                   binding=binding,
                   manifest=str(args.manifest.resolve()),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                   torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        temporary=args.output/'last.tmp';torch.save(state,temporary);temporary.replace(args.output/'last.ckpt')
        if (epoch+1)%5==0 or epoch+1==args.epochs: torch.save(state,args.output/f'epoch-{epoch+1:03d}.ckpt')
        print(json.dumps(dict(epoch=epoch+1,metrics={k:sum(r[k] for r in metrics)/max(1,len(metrics)) for k in metrics[0]} if metrics else {})),flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
