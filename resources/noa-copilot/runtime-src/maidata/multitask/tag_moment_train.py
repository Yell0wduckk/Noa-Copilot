"""Stage 0 hierarchical timbre-tag historyless MOMENT + unconditioned Type CLI.

Completely decoupled from any external BMS sample bank: accepts any prepared
manifest (BMS, GM MIDI, Slakh, E-GMD, Studio) and shares one audio encode per
song across the unconditioned Type readout and K tag-conditioned MOMENT tracks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

from .config import load, read_studio_options, resolve
from .timbre_tags import (
    COARSE_TAGS,
    FINE_TAGS,
    FINE_TO_COARSE,
    analyze_song_tag_tracks,
)


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / 'configs/multitask-stage0-tag-moment.json'


def _build_train_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--config', type=Path, default=_default_config_path())
    parser.add_argument('--output', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--stage0-checkpoint', type=Path)
    parser.add_argument('--yourmt3-checkpoint', type=Path)
    parser.add_argument('--workers', type=int, help='DataLoader workers (contract default: 12)')
    parser.add_argument('--audio-groups-per-batch', type=int)
    parser.add_argument('--token-budget', type=int)
    parser.add_argument('--scheduler-epochs', type=int)
    parser.add_argument('--stop-after-epochs', type=int)
    parser.add_argument('--device', type=str)
    parser.add_argument('--precision', choices=['float32', 'bfloat16'])
    parser.add_argument('--synthetic-options', type=Path)
    parser.add_argument('--synthetic-fraction', type=float)
    parser.add_argument('--pattern-bank', type=Path)
    parser.add_argument('--sample-bank', type=Path, help='Optional Studio pluck multisample index; never required by tag taxonomy')
    # Hierarchical Tag-Moment & Type training controls
    parser.add_argument('--max-queries-per-song', type=int, help='Sampled non-interfering tag MOMENT tracks per song')
    parser.add_argument('--coarse-condition-prob', type=float, help='Probability of promoting low-mix families to Coarse Macro Text Condition')
    parser.add_argument('--max-coarse-subtypes', type=int, help='Max active fine subtypes allowed for Coarse promotion')
    parser.add_argument('--max-collision-rate', type=float, help='Max co-onset subtype collision rate allowed for Coarse promotion')
    parser.add_argument('--min-events-per-query', type=int)
    parser.add_argument('--moment-type-loss-weight', type=float, help='Weight on unconditioned MOMENT -> 9+31 Type BCE')
    parser.add_argument('--residual-neg-weight', type=float, help='Negative beat/type weight inside >12s residual_mixed_intervals')
    parser.add_argument('--ll-ignore-threshold', type=float, help='Post-warmup single-tick MOMENT peak prob threshold for PU negative ignore')
    parser.add_argument('--ll-ignore-weight', type=float, help='Downweight factor when Peak LL-Ignore triggers outside residual intervals')
    parser.add_argument('--unconditioned-moment-only', action='store_true', help='Use MOMENT-only tokens on the unconditioned Type branch too')
    parser.add_argument('--no-unconditioned-chart', action='store_true', help='Disable Branch B and train only conditioned Branch A MOMENT tracks')
    parser.add_argument('--show-config', action='store_true')
    parser.add_argument('--cost-check', action='store_true', help='Measure one full logical update on device without saving a checkpoint')
    return parser


def _run_inspect(args: argparse.Namespace) -> int:
    source = args.source.resolve()
    metadata: dict = {}
    if source.is_dir() or source.suffix.lower() in {'.bms', '.bme', '.bml', '.pms', '.bmson'}:
        from .bms import convert_bms
        with tempfile.TemporaryDirectory() as tmp:
            events, metadata = convert_bms(source, Path(tmp) / 'mix.flac', allow_partial_audio=True)
    elif source.suffix.lower() in {'.mid', '.midi'}:
        from .midi import read_midi, read_midi_tempo
        events = read_midi(source)
        segments, _ = read_midi_tempo(source)
        if segments:
            metadata['bpm_segments'] = segments
    else:
        from .data import load_events
        payload = json.loads(source.read_text(encoding='utf-8-sig'))
        if isinstance(payload, dict):
            metadata = dict(payload.get('metadata', {}))
        events = load_events(source)

    if args.bpm is not None:
        metadata['bpm'] = float(args.bpm)

    report = analyze_song_tag_tracks(
        events,
        metadata,
        max_coarse_subtypes=args.max_coarse_subtypes,
        max_collision_rate=args.max_collision_rate,
    )
    summary = dict(
        source=str(source),
        total_events=len(events),
        active_coarse_tracks=[
            dict(
                coarse_id=c_id,
                coarse_tag=COARSE_TAGS[c_id - 1],
                moment_count=len(ticks),
                promotable_to_text_coarse=report['coarse_eligibility'][c_id]['promotable'],
                active_fine_tags=report['coarse_eligibility'][c_id]['active_fine_tags'],
                collision_rate=report['coarse_eligibility'][c_id]['collision_rate'],
            )
            for c_id, ticks in report['coarse_ticks'].items()
        ],
        active_fine_tracks=[
            dict(
                fine_id=f_id,
                fine_tag=FINE_TAGS[f_id - 1],
                coarse_tag=COARSE_TAGS[FINE_TO_COARSE[f_id] - 1],
                moment_count=len(ticks),
            )
            for f_id, ticks in report['fine_ticks'].items()
        ],
        slot6_ambiguous_count=report['slot6_ambiguous_count'],
        residual_mixed_intervals=report['residual_mixed_intervals'],
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def _run_train(args: argparse.Namespace) -> int:
    config = load(args.config)
    config.setdefault('tag_moment', {})['enabled'] = True
    for cli_name, cfg_key in (
        ('output', 'output'),
        ('manifest', 'manifest'),
        ('stage0_checkpoint', 'stage0_checkpoint'),
        ('yourmt3_checkpoint', 'yourmt3_checkpoint'),
    ):
        val = getattr(args, cli_name)
        if val is not None:
            config[cfg_key] = str(val)
    for key in (
        'workers',
        'audio_groups_per_batch',
        'token_budget',
        'scheduler_epochs',
        'stop_after_epochs',
        'device',
        'precision',
    ):
        val = getattr(args, key)
        if val is not None:
            config[key] = val

    tm = dict(config.get('tag_moment', {}))
    tm['enabled'] = True
    for key in (
        'max_queries_per_song',
        'coarse_condition_prob',
        'max_coarse_subtypes',
        'max_collision_rate',
        'min_events_per_query',
        'moment_type_loss_weight',
        'residual_neg_weight',
        'll_ignore_threshold',
        'll_ignore_weight',
    ):
        val = getattr(args, key)
        if val is not None:
            tm[key] = val
    if args.unconditioned_moment_only:
        tm['unconditioned_moment_only'] = True
    if args.no_unconditioned_chart:
        tm['include_unconditioned_chart'] = False
    config['tag_moment'] = tm

    opt = config['synthetic'] = dict(config.get('synthetic', {}))
    if args.synthetic_options:
        root = Path(config['manifest']).resolve().parent if config.get('manifest') else None
        opt.update(read_studio_options(args.synthetic_options, root))
    if args.synthetic_fraction is not None:
        opt['fraction'] = args.synthetic_fraction
    for name in ('pattern_bank', 'sample_bank'):
        if getattr(args, name):
            opt[name] = str(getattr(args, name).resolve())
    if config.get('manifest'):
        for name in ('pattern_bank', 'sample_bank'):
            if opt.get(name):
                opt[name] = str((Path(config['manifest']).resolve().parent / opt[name]).resolve())

    config = resolve(config)
    if args.show_config:
        print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)
        return 0
    from maidata.joint_training.runner import run
    run(config, args.resume, cost_check=args.cost_check)
    return 0


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw or raw[0] not in ('train', 'inspect-tags', '-h', '--help'):
        raw = ['train', *raw]

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)

    train_p = sub.add_parser('train', help='Run Stage 0 hierarchical tag-moment + unconditioned type training')
    _build_train_parser(train_p)

    inspect_p = sub.add_parser('inspect-tags', help='Inspect 9x31 timbre tracks and Coarse promotion on a song/JSON/MIDI')
    inspect_p.add_argument('--source', type=Path, required=True)
    inspect_p.add_argument('--bpm', type=float, default=None)
    inspect_p.add_argument('--max-coarse-subtypes', type=int, default=2)
    inspect_p.add_argument('--max-collision-rate', type=float, default=0.05)

    args = parser.parse_args(raw)
    if args.command == 'inspect-tags':
        return _run_inspect(args)
    return _run_train(args)


if __name__ == '__main__':
    raise SystemExit(main())
