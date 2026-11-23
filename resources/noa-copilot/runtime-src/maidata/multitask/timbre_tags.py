"""Self-contained 9-coarse x 31-fine discrete/text timbre taxonomy.

Decoupled from any external or local BMS sample bank: text categories, GM MIDI
programs/drums, synthetic Studio tracks, and isolated BMS dry waveforms all map
deterministically into the same (coarse_id, fine_id) label space.
"""
from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
import random
import re
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from .grid import BeatGrid
from .slots import METALLIC_DRUM_PITCHES, PITCHED_SLOTS, UNPITCHED_SLOTS

COARSE_SPECS: tuple[tuple[str, int, tuple[str, ...], tuple[tuple[str, tuple[str, ...]], ...]], ...] = (
    (
        'kick',
        0,
        ('kick', 'bd', 'bassdrum', 'bass_drum', '底鼓', '大鼓', '地鼓'),
        (
            ('kick.punch', ('kick_punch', 'punch_kick', '909kick', 'tight_kick')),
            ('kick.sub_808', ('kick_sub_808', 'sub_808', '808kick', 'subkick', 'deepkick')),
            ('kick.distorted_hard', ('kick_distorted_hard', 'distorted_kick', 'hardkick', 'gabber_kick')),
        ),
    ),
    (
        'snare_clap',
        1,
        ('snare_clap', 'snare', 'clap', 'sd', '军鼓', '拍手', '小鼓'),
        (
            ('snare_clap.snare_tight', ('snare_tight', 'tight_snare', 'acoustic_snare', 'elec_snare')),
            ('snare_clap.snare_break_roll', ('snare_break_roll', 'snare_roll', 'break_snare', 'amen_snare')),
            ('snare_clap.clap_snap', ('clap_snap', 'handclap', 'fingersnap', 'snap')),
            ('snare_clap.rim_wood_tom', ('rim_wood_tom', 'rimshot', 'sidestick', 'tom', 'woodblock')),
        ),
    ),
    (
        'hat_perc',
        2,
        ('hat_perc', 'hat', 'hihat', 'perc', 'percussion', '踩镲', '打击乐'),
        (
            ('hat_perc.hat_closed_shaker', ('hat_closed_shaker', 'closed_hat', 'chh', 'shaker')),
            ('hat_perc.hat_open_ride', ('hat_open_ride', 'open_hat', 'ohh', 'ride')),
            ('hat_perc.perc_latin_metal', ('perc_latin_metal', 'cowbell', 'conga', 'bongo', 'tambourine', 'taiko')),
        ),
    ),
    (
        'fx_cymbal',
        3,
        ('fx_cymbal', 'fx', 'sfx', 'cymbal', 'crash', '镲片', '音效', '过门音效'),
        (
            ('fx_cymbal.cymbal_crash', ('cymbal_crash', 'china_cymbal', 'splash_cymbal')),
            ('fx_cymbal.riser_sweep', ('riser_sweep', 'riser', 'uplifter', 'downlifter', 'reverse_cymbal', 'sweep')),
            ('fx_cymbal.impact_hit_se', ('impact_hit_se', 'impact', 'boom_fx', 'laser', 'scratch', 'se')),
        ),
    ),
    (
        'vocal',
        4,
        ('vocal', 'voice', 'vox', 'singing', 'vocaloid', '人声', '主唱', '歌声'),
        (
            ('vocal.lead_singing', ('lead_singing', 'vocal_singing', 'vocal_lead', 'utau')),
            ('vocal.chop_shout', ('chop_shout', 'vocal_chop', 'vox_chop', 'shout', 'rap_vox')),
            ('vocal.choir_voice_pad', ('choir_voice_pad', 'vocal_choir', 'choir', 'aah_ooh')),
        ),
    ),
    (
        'bass',
        5,
        ('bass', 'sub', 'bs', '贝斯', '低音'),
        (
            ('bass.sub_808', ('bass_sub_808', 'subbass', '808bass', 'sine_bass')),
            ('bass.synth_acid_wobble', ('synth_acid_wobble', 'bass_acid_wobble', 'synthbass', 'acidbass', 'wobble_bass', 'reese_bass')),
            ('bass.electric_acoustic', ('electric_acoustic', 'bass_elec_acoustic', 'ebass', 'slap_bass', 'finger_bass', 'acoustic_bass')),
        ),
    ),
    (
        'piano_keys',
        6,
        ('piano_keys', 'piano', 'keys', 'keyboard', 'pf', '钢琴', '键盘', '电钢'),
        (
            ('piano_keys.acoustic_piano', ('acoustic_piano', 'grand_piano', 'upright_piano', 'bright_piano')),
            ('piano_keys.epiano_clav_harpsi', ('epiano_clav_harpsi', 'epiano', 'rhodes', 'wurlitzer', 'clavinet', 'harpsichord')),
        ),
    ),
    (
        'pluck_mallet',
        6,
        ('pluck_mallet', 'pluck', 'mallet', 'guitar', 'arp', 'bell', '弹拨', '敲击旋律', '吉他', '马林巴', '钟琴'),
        (
            ('pluck_mallet.bell_glock_celesta', ('bell_glock_celesta', 'bell_musicbox', 'glockenspiel', 'celesta', 'musicbox', 'chime')),
            ('pluck_mallet.mallet_marimba_vibe', ('mallet_marimba_vibe', 'mallet_marimba', 'marimba', 'xylophone', 'vibraphone', 'kalimba')),
            ('pluck_mallet.chip_8bit', ('chip_8bit', '8bit', 'chiptune', 'pikopiko', 'nes_pulse')),
            ('pluck_mallet.guitar_harp_acoustic', ('guitar_harp_acoustic', 'guitar_acoustic_harp', 'acoustic_guitar', 'harp', 'koto', 'shamisen', 'pizzicato')),
            ('pluck_mallet.guitar_electric', ('guitar_electric', 'electric_guitar', 'clean_guitar', 'dist_guitar', 'overdrive_guitar')),
            ('pluck_mallet.synth_pluck_arp', ('synth_pluck_arp', 'synth_pluck', 'pluck_arp')),
        ),
    ),
    (
        'synth_orch',
        7,
        ('synth_orch', 'synth', 'lead', 'pad', 'strings', 'brass', 'orchestra', '合成器', '管弦', '弦乐', '铜管'),
        (
            ('synth_orch.synth_lead', ('synth_lead', 'synth_lead_saw', 'supersaw', 'sawlead', 'squarelead')),
            ('synth_orch.synth_pad_organ', ('synth_pad_organ', 'synth_pad_chord', 'synth_pad', 'organ', 'accordion')),
            ('synth_orch.strings_ensemble', ('strings_ensemble', 'violin', 'viola', 'cello', 'orchestral_strings')),
            ('synth_orch.brass_woodwind', ('brass_woodwind', 'brass_wind', 'trumpet', 'trombone', 'saxophone', 'flute', 'woodwind')),
        ),
    ),
)

COARSE_TAGS: tuple[str, ...] = tuple(spec[0] for spec in COARSE_SPECS)
NUM_COARSE_TAGS: int = len(COARSE_TAGS)

_fine_names: list[str] = []
_fine_to_coarse: list[int] = [0]
_coarse_to_fine: dict[int, list[int]] = {}
_coarse_to_slot: list[int] = [-1]
_TEXT_ALIASES: dict[str, tuple[int, int]] = {
    '': (0, 0),
    'null': (0, 0),
    'none': (0, 0),
    'unconditional': (0, 0),
    '无条件': (0, 0),
}

for _c_idx, (_c_name, _slot, _c_aliases, _f_specs) in enumerate(COARSE_SPECS, start=1):
    _coarse_to_slot.append(_slot)
    _coarse_to_fine[_c_idx] = []
    for _alias in (_c_name, *_c_aliases):
        _TEXT_ALIASES[_alias.strip().casefold()] = (_c_idx, 0)
    for _f_name, _f_aliases in _f_specs:
        _f_idx = len(_fine_names) + 1
        _fine_names.append(_f_name)
        _fine_to_coarse.append(_c_idx)
        _coarse_to_fine[_c_idx].append(_f_idx)
        _short = _f_name.split('.', 1)[-1]
        for _alias in (_f_name, _short, *_f_aliases):
            _TEXT_ALIASES[_alias.strip().casefold()] = (_c_idx, _f_idx)

FINE_TAGS: tuple[str, ...] = tuple(_fine_names)
NUM_FINE_TAGS: int = len(FINE_TAGS)
FINE_TO_COARSE: tuple[int, ...] = tuple(_fine_to_coarse)
COARSE_TO_FINE_IDS: dict[int, tuple[int, ...]] = {k: tuple(v) for k, v in _coarse_to_fine.items()}
COARSE_TO_SLOT: tuple[int, ...] = tuple(_coarse_to_slot)
MOMENT_TYPE_DIM: int = NUM_COARSE_TAGS + NUM_FINE_TAGS

assert NUM_COARSE_TAGS == 9 and NUM_FINE_TAGS == 31 and MOMENT_TYPE_DIM == 40

FINE_INDEX: dict[str, int] = {name: idx for idx, name in enumerate(FINE_TAGS, start=1)}
COARSE_INDEX: dict[str, int] = {name: idx for idx, name in enumerate(COARSE_TAGS, start=1)}


def _normalize_tag_key(text: str) -> str:
    cleaned = text.strip().casefold().replace('-', '_').replace('/', '_').replace(' ', '_')
    return re.sub(r'_+', '_', cleaned)


_NORMALIZED_ALIASES: dict[str, tuple[int, int]] = {
    _normalize_tag_key(k): v for k, v in _TEXT_ALIASES.items()
}


def format_tag_label(coarse_id: int, fine_id: int = 0) -> str:
    if fine_id > 0:
        return FINE_TAGS[fine_id - 1]
    if coarse_id > 0:
        return COARSE_TAGS[coarse_id - 1]
    return 'null'


def resolve_tag_query(spec) -> tuple[int, int]:
    """Resolve None, a text alias, (coarse_id, fine_id), or metadata dict."""
    if spec is None:
        return (0, 0)
    if isinstance(spec, (tuple, list)) and len(spec) == 2:
        c, f = int(spec[0]), int(spec[1])
        if f < 0 or f > NUM_FINE_TAGS:
            raise ValueError(f'fine tag id out of range: {f}')
        if f > 0:
            expected = FINE_TO_COARSE[f]
            if c not in (0, expected):
                raise ValueError(f'fine tag {f} belongs to coarse {expected}, got {c}')
            return (expected, f)
        if c < 0 or c > NUM_COARSE_TAGS:
            raise ValueError(f'coarse tag id out of range: {c}')
        return (c, 0)
    if isinstance(spec, Mapping):
        if spec.get('query_fine_id') or spec.get('query_coarse_id'):
            return resolve_tag_query((spec.get('query_coarse_id', 0), spec.get('query_fine_id', 0)))
        if 'query_tag' in spec and spec['query_tag'] is not None:
            return resolve_tag_query(spec['query_tag'])
        return (0, 0)
    if isinstance(spec, str):
        raw = spec.strip()
        if not raw:
            return (0, 0)
        key = raw.casefold()
        if key in _TEXT_ALIASES:
            return _TEXT_ALIASES[key]
        norm = _normalize_tag_key(raw)
        if norm in _NORMALIZED_ALIASES:
            return _NORMALIZED_ALIASES[norm]
        if '.' in norm:
            left, right = norm.split('.', 1)
            if left in _NORMALIZED_ALIASES and right in _NORMALIZED_ALIASES:
                c_left, _ = _NORMALIZED_ALIASES[left]
                c_right, f_right = _NORMALIZED_ALIASES[right]
                if f_right > 0 and (c_left == 0 or c_left == c_right):
                    return (c_right, f_right)
        raise ValueError(f'unknown timbre tag query: {spec!r}')
    raise TypeError(f'unsupported timbre tag specification: {type(spec)!r}')


_TRAILING_NOTE_SUFFIX = re.compile(r'_(?:\d{3,5}_[a-g][a-gp#b]*|(?:v\d+)?l[\d-]+o[0-8][a-g](?:p|s|#)?)$', re.I)


def _compile_pattern(compounds: tuple[str, ...], tokens: tuple[str, ...]) -> re.Pattern[str]:
    parts = ['|'.join(map(re.escape, compounds))] if compounds else []
    if tokens:
        parts.append(r'(?<![a-z])(?:' + '|'.join(map(re.escape, tokens)) + r')s?(?![a-z])')
    return re.compile('|'.join(parts))


# Ordered specific-before-generic filename rules returning (coarse_id, fine_id, supervision).
_FILENAME_FINE_RULES: tuple[tuple[int, int, str, re.Pattern[str]], ...] = tuple(
    (c_id, f_id, sup, _compile_pattern(compounds, tokens))
    for c_id, f_id, sup, compounds, tokens in (
        (4, 13, 'hard', ('choir_fx',), ()),
        # Kick subtypes + generic kick
        (1, 2, 'hard', ('808kick', 'subkick', 'deepkick', 'boomkick'), ()),
        (1, 3, 'hard', ('hardkick', 'distkick', 'gabberkick', 'rawkick', 'schranz'), ()),
        (1, 1, 'hard', ('909kick', 'punchkick', 'tightkick', 'acokick'), ()),
        (1, 0, 'coarse_only', ('bassdrum', 'kickroll', 'fxkick', 'kick'), ('bd', 'kck', 'kik', 'bd1', 'bd2')),
        # Snare / Clap subtypes
        (2, 6, 'hard', ('handclap', 'handcrap', 'clap', 'fingersnap'), ('clp', 'crap', 'snap')),
        (2, 5, 'hard', ('snareroll', 'break_snare', 'amensnare', 'ghostsnare'), ('amen',)),
        (2, 7, 'hard', ('rimshot', 'sidestick', 'woodblock', 'claves', 'floortom', 'flortom', 'midtom', 'hitom', 'lowtom'),
         ('rim', 'tom')),
        (2, 4, 'hard', ('snare',), ('sd', 'sn', 'snr')),
        # Hat / Perc subtypes
        (3, 9, 'hard', ('openhat', 'ophat', 'openhh', 'ridebell', 'ridecym'), ('ohh', 'hho', 'oh', 'ride', 'rd')),
        (3, 8, 'hard', ('closehat', 'closedhat', 'clhat', 'closehh', 'shaker', 'cabasa', 'maracas'),
         ('chh', 'hhc', 'ch')),
        (3, 10, 'hard', ('tambourine', 'conga', 'bongo', 'cowbell', 'taiko', 'agogo', 'timbale', 'djembe', 'cajon', 'castanet', 'guiro', 'cuica', 'whistle'),
         ('perc', 'tamb', 'tam', 'triangle')),
        (3, 0, 'coarse_only', ('hihat', 'hi_hat', 'percussion'), ('hat', 'hh')),
        # FX / Cymbal subtypes
        (4, 12, 'hard', ('downlifter', 'uplifter', 'riser', 'sweep', 'revcym', 'reverse'), ('rev',)),
        (4, 11, 'hard', ('cymbal', 'symbal', 'crash', 'clash', 'china', 'splash'), ('cym', 'cc')),
        (4, 13, 'hard', ('tapestop', 'scratch', 'impact', 'whoosh', 'noise', 'noize', 'noiz', 'laser', 'siren', 'alarm', 'glitch', 'vinyl', 'boom', 'sfx'),
         ('fx', 'se', 'scr')),
        # Pluck / Mallet / Piano explicit subtypes (before generic keys/pluck)
        (8, 24, 'hard', ('pikopiko', 'piko', 'chiptune', '8bit', 'gameboy', 'famicom'), ('chip', 'nes')),
        (8, 22, 'hard', ('glockenspiel', 'glock', 'celesta', 'musicbox', 'orgel', 'tubular', 'chime', 'bell'), ()),
        (8, 23, 'hard', ('vibraphone', 'marimba', 'xylophone', 'xylo', 'kalimba', 'steeldrum', 'dulcimer', 'mallet', 'timpani'),
         ('vib', 'vibe')),
        (7, 21, 'hard', ('harpsichord', 'epiano', 'elepi', 'rhodes', 'wurlitzer', 'wurli', 'clavinet', 'clav'), ('ep',)),
        (7, 20, 'hard', ('piano',), ('pf', 'pia', 'pi')),
        (7, 0, 'coarse_only', (), ('keys',)),
        (8, 25, 'hard', ('acogi', 'shamisen', 'koto', 'sitar', 'banjo', 'mandolin', 'ukulele', 'harp', 'zither', 'pizzicato', 'pizz'),
         ('aco', 'ag')),
        (8, 26, 'hard', ('distgt', 'overdrive', 'eguitar', 'elecguitar'), ('eg',)),
        (8, 0, 'coarse_only', ('guitar', 'guiter'), ('gt', 'gtr', 'guit')),
        (8, 27, 'hard', ('arpeggio', 'arrpegio', 'synthpluck', 'pluck'), ('arp', 'pl', 'plk')),
        # Vocal subtypes
        (5, 16, 'hard', ('choir_pad', 'choir'), ()),
        (5, 15, 'hard', ('vox_chop', 'vocalchop', 'voxchop', 'scream', 'shout', 'laugh', 'breath'), ('rap', 'chop')),
        (5, 14, 'hard', ('vocaloid', 'vocal', 'voice', 'utau', 'whisper'), ('vo', 'vox', 'voc', 'sing', 'uta', 'koe')),
        # Bass subtypes
        (6, 17, 'hard', ('subbass', '808bass', 'sinebass'), ('808', 'sub')),
        (6, 18, 'hard', ('synthbass', 'acidbass', 'neurobass', 'fmbass', 'wobble', 'reese', 'tb303', 'growl', 'donk', 'acid'), ()),
        (6, 19, 'hard', ('ebass', 'pickbass', 'fingerbass', 'fretless', 'slapbass'), ('slap',)),
        (6, 0, 'coarse_only', ('bass',), ('bs',)),
        # Synth / Orchestral subtypes
        (9, 30, 'hard', ('violoncello', 'contrabass', 'violin', 'viola', 'cello', 'strings', 'string', 'orchestra'),
         ('str', 'vln', 'vio', 'orc', 'mainst', 'backst')),
        (9, 31, 'hard', ('trumpet', 'trombone', 'frenchhorn', 'saxophone', 'brass', 'piccolo', 'clarinet', 'bassoon',
                         'recorder', 'ocarina', 'flute', 'oboe', 'woodwind', 'harmonica', 'bagpipe'),
         ('fl', 'br', 'tp', 'tb', 'tuba', 'sax', 'horn', 'wind')),
        (9, 29, 'hard', ('organ', 'accordion'), ('pd', 'pad', 'org', 'chord', 'poly')),
        (9, 28, 'hard', ('supersaw', 'synthlead', 'sawlead', 'squarelead', 'hoover', 'lead'), ('ld', 'saw', 'solo')),
    )
)


def _match_filename_tag(filename: str) -> tuple[int, int, str] | None:
    if not filename:
        return None
    stem = _TRAILING_NOTE_SUFFIX.sub('', Path(filename).stem)
    stem = re.sub(r'([a-z])([A-Z])', r'\1 \2', stem)
    stem = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', stem).casefold()
    for c_id, f_id, sup, pattern in _FILENAME_FINE_RULES:
        if pattern.search(stem):
            return (c_id, f_id, sup)
    return None


def _spectral_profile(signal: np.ndarray, sample_rate: int, f0: float | None = None) -> dict[str, float]:
    """Fast dry-slice spectral energy ratios and harmonic symmetry."""
    n = min(len(signal), max(256, round(0.18 * sample_rate)))
    if n < 64:
        return dict(sub_110_ratio=0.0, mid_150_600_ratio=0.0, click_2k_6k_ratio=0.0,
                    crest_factor=1.0, odd_even_ratio=1.0)
    snippet = signal[:n].astype(np.float64)
    rms = float(np.sqrt(np.mean(snippet ** 2)))
    peak = float(np.max(np.abs(snippet)))
    crest = peak / max(rms, 1e-12)
    windowed = snippet * np.hanning(n)
    power = np.abs(np.fft.rfft(windowed)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    total = float(power[freqs >= 25.0].sum())
    if total <= 1e-18:
        return dict(sub_110_ratio=0.0, mid_150_600_ratio=0.0, click_2k_6k_ratio=0.0,
                    crest_factor=crest, odd_even_ratio=1.0)
    sub_110 = float(power[(freqs >= 25.0) & (freqs <= 115.0)].sum()) / total
    mid_150_600 = float(power[(freqs >= 145.0) & (freqs <= 620.0)].sum()) / total
    click_2k_6k = float(power[(freqs >= 2000.0) & (freqs <= 6500.0)].sum()) / total
    odd_even = 1.0
    if f0 is not None and 55.0 <= f0 <= sample_rate / 6.0:
        odd_e, even_e = 0.0, 0.0
        bw = max(18.0, f0 * 0.08)
        for h in range(1, 8):
            center = f0 * h
            if center >= sample_rate * 0.48:
                break
            band = power[(freqs >= center - bw) & (freqs <= center + bw)]
            val = float(band.sum())
            if h >= 3 and h % 2 == 1:
                odd_e += val
            elif h >= 2 and h % 2 == 0:
                even_e += val
        odd_even = (odd_e + 1e-12) / (even_e + 1e-12)
    return dict(
        sub_110_ratio=sub_110,
        mid_150_600_ratio=mid_150_600,
        click_2k_6k_ratio=click_2k_6k,
        crest_factor=crest,
        odd_even_ratio=odd_even,
    )


def _refine_fine_from_physics(coarse_id: int, duration: float, attack: float, sustain: float,
                              f0: float | None, centroid: float, multiple: bool,
                              spec: Mapping[str, float]) -> tuple[int, str]:
    sub = spec['sub_110_ratio']
    mid = spec['mid_150_600_ratio']
    click = spec['click_2k_6k_ratio']
    crest = spec['crest_factor']
    odd_even = spec['odd_even_ratio']
    if coarse_id == 1:  # kick
        if mid > 0.40 or (crest < 2.55 and mid > 0.26):
            return (3, 'hard')  # kick.distorted_hard
        if sub > 0.66 and duration > 0.20 and click < 0.05:
            return (2, 'hard')  # kick.sub_808
        return (1, 'hard')  # kick.punch
    if coarse_id == 2:  # snare_clap
        if click > 0.36 and sub < 0.08 and duration < 0.28:
            return (6, 'hard')  # clap_snap
        if sub > 0.28 or centroid < 950.0:
            return (7, 'hard')  # rim_wood_tom
        return (4, 'hard')  # snare_tight
    if coarse_id == 3:  # hat_perc
        if centroid > 4200.0:
            return (9 if (duration > 0.24 or sustain > 0.22) else 8, 'hard')
        return (10, 'hard')
    if coarse_id == 4:  # fx_cymbal
        if attack > 0.14:
            return (12, 'hard')  # riser_sweep
        if centroid > 3200.0 and duration > 0.35:
            return (11, 'hard')  # cymbal_crash
        return (13, 'hard')  # impact_hit_se
    if coarse_id == 5:  # vocal
        if duration < 0.32 and attack < 0.04:
            return (15, 'hard')  # vocal.chop_shout
        if multiple and attack > 0.09:
            return (16, 'hard')  # vocal.choir_voice_pad
        return (14, 'hard')  # vocal.lead_singing
    if coarse_id == 6:  # bass
        if sub > 0.72 and centroid < 320.0:
            return (17, 'hard')  # bass.sub_808
        if centroid > 620.0 or mid > 0.45:
            return (18, 'hard')  # bass.synth_acid_wobble
        return (19, 'coarse_only')
    if coarse_id == 7:  # piano_keys
        return (21 if centroid > 1850.0 else 20, 'coarse_only')
    if coarse_id == 8:  # pluck_mallet
        if (f0 is not None and f0 >= 520.0) or centroid >= 2400.0:
            return (22, 'hard')  # bell_glock_celesta (catches melody3 cleanly)
        if odd_even >= 2.4 and attack <= 0.018 and not multiple:
            return (24, 'hard')  # chip_8bit
        if not multiple and centroid < 1200.0 and duration < 0.45:
            return (23, 'coarse_only')  # mallet_marimba_vibe
        return (27, 'coarse_only')  # synth_pluck_arp
    if coarse_id == 9:  # synth_orch
        if multiple or attack > 0.09 or sustain > 0.68:
            return (29, 'coarse_only')  # synth_pad_organ
        return (28, 'coarse_only')  # synth_lead
    return (0, 'coarse_only')


def classify_dry_waveform(waveform, sample_rate: int = 44100, *, filename: str = '',
                          base_analysis: Mapping | None = None) -> dict:
    """Assign (coarse_id, fine_id, tag_supervision) to one dry audio slice."""
    signal = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if base_analysis is not None:
        if base_analysis.get('silent'):
            return dict(coarse_id=4, fine_id=13, coarse_tag='fx_cymbal', fine_tag='fx_cymbal.impact_hit_se',
                        tag_supervision='coarse_only', slot=3, sub_110_ratio=0.0,
                        mid_150_600_ratio=0.0, click_2k_6k_ratio=0.0)
        duration = float(base_analysis.get('effective_duration', len(signal) / max(sample_rate, 1)))
        attack = float(base_analysis.get('attack_seconds', 0.01))
        sustain = float(base_analysis.get('sustain_ratio', 0.2))
        f0 = base_analysis.get('fundamental_hz')
        centroid = float(base_analysis.get('centroid_hz', 0.0))
        multiple = bool(base_analysis.get('polyphonic', False))
        slot = int(base_analysis.get('slot', 7))
    else:
        peak = float(np.max(np.abs(signal))) if len(signal) else 0.0
        if peak <= 1e-10:
            return dict(coarse_id=4, fine_id=13, coarse_tag='fx_cymbal', fine_tag='fx_cymbal.impact_hit_se',
                        tag_supervision='coarse_only', slot=3, sub_110_ratio=0.0,
                        mid_150_600_ratio=0.0, click_2k_6k_ratio=0.0)
        duration = len(signal) / sample_rate
        block = max(1, round(sample_rate * 0.005))
        padded = np.pad(signal, (0, (-len(signal)) % block))
        env = np.sqrt(np.mean(padded.reshape(-1, block) ** 2, axis=1))
        env_peak = float(np.max(env))
        attack = float(np.flatnonzero(env >= 0.9 * env_peak)[0]) * block / sample_rate
        sustain = float(np.mean(env[-max(1, len(env) // 4):]) / max(env_peak, 1e-12))
        nfft = min(len(signal), 4096)
        spec_pow = np.abs(np.fft.rfft(signal[:nfft] * np.hanning(nfft))) ** 2
        freqs = np.fft.rfftfreq(nfft, 1.0 / sample_rate)
        centroid = float(np.dot(spec_pow, freqs) / max(spec_pow.sum(), 1e-18))
        f0 = None
        multiple = False
        slot = 0 if centroid < 220 else 2 if centroid > 3500 else 1

    spec = _spectral_profile(signal, sample_rate, f0=f0)
    matched = _match_filename_tag(filename)
    if matched is not None:
        coarse_id, fine_id, sup = matched
        if fine_id == 0:
            fine_id, phys_sup = _refine_fine_from_physics(
                coarse_id, duration, attack, sustain, f0, centroid, multiple, spec
            )
            # Kick/Snare/Hat/FX physical sub-splits on named drum stems are crisp & reliable.
            sup = 'hard' if coarse_id in (1, 2, 3, 4) else phys_sup
        elif coarse_id == 5 and fine_id == 14 and duration < 0.30 and attack < 0.035:
            fine_id = 15  # short vocal slice -> chop_shout
    else:
        # Pure waveform fallback: never assign Vocal (5) without human vocal metadata/name,
        # preventing high-pitched bells/synths (e.g. melody3) from contaminating Vocal.
        if slot == 0:
            coarse_id = 1
        elif slot == 1:
            coarse_id = 2
        elif slot == 2:
            coarse_id = 3
        elif slot == 3:
            coarse_id = 4
        elif slot == 5:
            coarse_id = 6
        elif slot == 6:
            # Distinguish warm chordal/low-mid piano vs high bell / odd-harmonic chip / pluck.
            if (f0 is not None and f0 >= 500.0) or centroid >= 2100.0 or spec['odd_even_ratio'] >= 2.3:
                coarse_id = 8
            elif multiple and centroid < 1450.0 and duration > 0.25:
                coarse_id = 7
            else:
                coarse_id = 8
        else:
            coarse_id = 9
        fine_id, sup = _refine_fine_from_physics(
            coarse_id, duration, attack, sustain, f0, centroid, multiple, spec
        )

    return dict(
        coarse_id=coarse_id,
        fine_id=fine_id,
        coarse_tag=COARSE_TAGS[coarse_id - 1],
        fine_tag=FINE_TAGS[fine_id - 1] if fine_id > 0 else None,
        tag_supervision=sup,
        slot=COARSE_TO_SLOT[coarse_id],
        sub_110_ratio=round(spec['sub_110_ratio'], 4),
        mid_150_600_ratio=round(spec['mid_150_600_ratio'], 4),
        click_2k_6k_ratio=round(spec['click_2k_6k_ratio'], 4),
    )


def classify_gm_program(program: int, *, vocal: bool = False, teacher: bool = False) -> tuple[int, int, str]:
    """Deterministic GM 0..127 program -> (coarse_id, fine_id, supervision)."""
    if vocal or (teacher and int(program) == 100):
        return (5, 14, 'coarse_only' if teacher else 'hard')
    p = int(program)
    if 0 <= p <= 3:
        return (7, 20, 'hard')
    if 4 <= p <= 7:
        return (7, 21, 'hard')
    if p in (8, 9, 10, 11, 14, 112):
        return (8, 22, 'hard')
    if p in (12, 13, 47, 108, 114):
        return (8, 23, 'hard')
    if 16 <= p <= 23:
        return (9, 29, 'hard')
    if p in (24, 25, 45, 46, 104, 105, 106, 107):
        return (8, 25, 'hard')
    if 26 <= p <= 31:
        return (8, 26, 'hard')
    if p in (32, 33, 34, 35, 36, 37):
        return (6, 19, 'hard')
    if p in (38, 39):
        return (6, 18, 'hard')
    if (40 <= p <= 44) or (48 <= p <= 51):
        return (9, 30, 'hard')
    if 52 <= p <= 54:
        return (5, 16, 'hard')
    if p == 55:
        return (4, 13, 'hard')
    if 56 <= p <= 79:
        return (9, 31, 'hard')
    if p == 80:
        return (8, 24, 'coarse_only')
    if 81 <= p <= 87:
        return (9, 28, 'hard')
    if 88 <= p <= 103:
        return (9, 29, 'hard')
    if p == 113 or (115 <= p <= 118):
        return (3, 10, 'hard')
    if p >= 119:
        return (4, 12 if p in (121, 122, 126) else 13, 'hard')
    return (8, 27, 'coarse_only')


def classify_gm_drum(pitch: int) -> tuple[int, int, str]:
    """Deterministic GM percussion key 0..127 -> (coarse_id, fine_id, supervision)."""
    p = int(pitch)
    if p == 35:
        return (1, 2, 'hard')
    if p == 36:
        return (1, 1, 'hard')
    if p in (38, 40):
        return (2, 4, 'hard')
    if p == 39:
        return (2, 6, 'hard')
    if p in (37, 41, 43, 45, 47, 48, 50):
        return (2, 7, 'hard')
    if p in (42, 44, 69, 70, 82):
        return (3, 8, 'hard')
    if p in (46, 51, 53, 59):
        return (3, 9, 'hard')
    if p in METALLIC_DRUM_PITCHES:
        return (4, 11, 'hard')
    return (3, 10, 'hard')


def _classify_studio_role(data: Mapping, studio_instruments: Mapping | None = None) -> tuple[int, int, str] | None:
    role = str(data.get('role', '')).strip().casefold()
    if not role:
        return None
    inst = (studio_instruments or {}).get(role)
    inst_text = str(inst).casefold() if inst is not None else ''
    art = str(data.get('articulation', '')).casefold()
    if role == 'kick':
        if '808' in inst_text or 'sub' in inst_text:
            return (1, 2, 'hard')
        if 'hard' in inst_text or 'dist' in inst_text:
            return (1, 3, 'hard')
        return (1, 1, 'hard')
    if role == 'snare':
        if 'clap' in inst_text or 'snap' in art:
            return (2, 6, 'hard')
        if 'roll' in art or 'break' in art:
            return (2, 5, 'hard')
        return (2, 4, 'hard')
    if role == 'hat':
        if 'open' in art or 'open' in inst_text or 'ride' in inst_text:
            return (3, 9, 'hard')
        return (3, 8, 'hard')
    if role in ('crash', 'fx'):
        if 'riser' in art or 'sweep' in art or 'reverse' in art or 'lifter' in art:
            return (4, 12, 'hard')
        if role == 'crash' or 'crash' in art:
            return (4, 11, 'hard')
        return (4, 13, 'hard')
    if role == 'vocal':
        if float(data.get('duration', 0.5)) < 0.25 and not data.get('lyric'):
            return (5, 15, 'hard')
        return (5, 14, 'hard')
    if role == 'bass':
        if '808' in inst_text or 'sub' in inst_text:
            return (6, 17, 'hard')
        if 'slap' in inst_text or 'finger' in inst_text or 'pick' in inst_text:
            return (6, 19, 'hard')
        return (6, 18, 'hard')
    if role == 'pluck':
        matched = _match_filename_tag(inst_text)
        if matched is not None and matched[0] in (7, 8):
            c_id, f_id, sup = matched
            return (c_id, f_id if f_id > 0 else (20 if c_id == 7 else 27), sup)
        return (8, 27, 'hard')
    if role == 'pad':
        return (9, 29, 'hard')
    if role == 'lead':
        matched = _match_filename_tag(inst_text)
        if matched is not None and matched[0] == 9 and matched[1] > 0:
            return matched
        return (9, 28, 'hard')
    return None


def classify_event_tags(event_or_data, kind: str = 'note', *,
                        studio_instruments: Mapping | None = None) -> list[tuple[int, int, str]]:
    """Return one or more (coarse_id, fine_id, supervision) tuples for an event."""
    if hasattr(event_or_data, 'data'):
        kind = event_or_data.kind
        data = event_or_data.data
    else:
        data = event_or_data

    if data.get('merged_tag_triples'):
        return [(int(c), int(f), str(s)) for c, f, s in data['merged_tag_triples']]
    if data.get('fine_id') or data.get('coarse_id'):
        c, f = resolve_tag_query((data.get('coarse_id', 0), data.get('fine_id', 0)))
        sup = str(data.get('tag_supervision', 'hard' if f > 0 else 'coarse_only'))
        return [(c, f, sup)]
    if data.get('fine_tag') or data.get('coarse_tag'):
        c, f = resolve_tag_query(data.get('fine_tag') or data.get('coarse_tag'))
        sup = str(data.get('tag_supervision', 'hard' if f > 0 else 'coarse_only'))
        return [(c, f, sup)]

    studio_hit = _classify_studio_role(data, studio_instruments)
    if studio_hit is not None:
        return [studio_hit]

    if kind == 'drum' or data.get('program') in (9, 'Drums'):
        return [classify_gm_drum(data.get('pitch', 36))]
    if 'gm_program' in data and data['gm_program'] is not None:
        return [classify_gm_program(int(data['gm_program']),
                                    vocal=bool(data.get('vocal', False)),
                                    teacher=bool(data.get('pseudo', False)))]

    from .slots import event_slot
    slot = event_slot(data, kind)
    slot_to_coarse = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 7: 9}
    if slot in slot_to_coarse:
        return [(slot_to_coarse[slot], 0, 'coarse_only')]
    # Legacy slot == 6 without program/tag spans both piano_keys (7) and pluck_mallet (8).
    return [(0, 0, 'slot6_ambiguous')]


def merge_intervals(intervals: Iterable[Sequence[float]] | None) -> list[list[float]]:
    if not intervals:
        return []
    cleaned = sorted((float(a), float(b)) for a, b in intervals if float(b) > float(a) + 1e-6)
    if not cleaned:
        return []
    merged = [[cleaned[0][0], cleaned[0][1]]]
    for start, end in cleaned[1:]:
        if start <= merged[-1][1] + 1e-3:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [[round(a, 4), round(b, 4)] for a, b in merged]


def crop_and_stretch_intervals(intervals: Iterable | None, crop_start: float,
                               crop_duration: float, speed: float = 1.0) -> list[list[float]]:
    if not intervals:
        return []
    crop_end = crop_start + crop_duration
    out = []
    for pair in intervals:
        a, b = max(float(pair[0]), crop_start), min(float(pair[1]), crop_end)
        if b > a + 1e-6:
            out.append([(a - crop_start) / speed, (b - crop_start) / speed])
    return merge_intervals(out)


def _in_intervals(seconds: np.ndarray, intervals: list[list[float]]) -> np.ndarray:
    if not intervals or len(seconds) == 0:
        return np.zeros(len(seconds), dtype=bool)
    active = np.zeros(len(seconds), dtype=bool)
    for start, end in intervals:
        active |= (seconds >= start - 1e-4) & (seconds <= end + 1e-4)
    return active


def attach_unconditioned_moment_types(encoded, events: list, metadata: Mapping | None = None, *,
                                      residual_neg_weight: float = 0.25):
    """Populate 40-dim (9 coarse + 31 fine) multi-label targets on MOMENT tokens."""
    meta = metadata or {}
    n_tokens = len(encoded.token_ids)
    targets = np.zeros((n_tokens, MOMENT_TYPE_DIM), dtype=np.float32)
    weights = np.zeros((n_tokens, MOMENT_TYPE_DIM), dtype=np.float32)
    slot6_union = np.zeros(n_tokens, dtype=bool)

    if meta.get('moment_only') or resolve_tag_query(meta) != (0, 0):
        encoded.moment_type_targets = torch.from_numpy(targets)
        encoded.moment_type_weights = torch.from_numpy(weights)
        encoded.moment_slot6_union = torch.from_numpy(slot6_union)
        return encoded

    grid = BeatGrid(meta)
    studio_inst = meta.get('studio_instruments')
    by_tick: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    for ev in events:
        if getattr(ev, 'kind', '') == 'note_off':
            continue
        tick = ev.data['start_tick'] if 'start_tick' in ev.data else grid.tick(ev.time)
        by_tick[int(tick)].extend(classify_event_tags(ev, studio_instruments=studio_inst))

    tok_np = encoded.token_ids.numpy()
    moment_rows = np.flatnonzero((tok_np >= 17) & (tok_np < 113))
    if len(moment_rows) == 0:
        encoded.moment_type_targets = torch.from_numpy(targets)
        encoded.moment_type_weights = torch.from_numpy(weights)
        encoded.moment_slot6_union = torch.from_numpy(slot6_union)
        return encoded

    moment_ticks = np.rint(encoded.causal_beats.numpy()[moment_rows] * 96.0).astype(np.int64)
    moment_secs = encoded.onset_seconds.numpy()[moment_rows]
    residual = merge_intervals(meta.get('residual_mixed_intervals'))
    in_res = _in_intervals(moment_secs, residual)

    for idx, row, tick, res_flag in zip(range(len(moment_rows)), moment_rows, moment_ticks, in_res):
        if moment_secs[idx] < -1e-6:
            continue
        triples = by_tick.get(int(tick))
        if not triples:
            continue
        neg_w = float(residual_neg_weight) if res_flag else 1.0
        w_row = np.full(MOMENT_TYPE_DIM, neg_w, dtype=np.float32)
        t_row = np.zeros(MOMENT_TYPE_DIM, dtype=np.float32)
        coarse_only_families: set[int] = set()
        has_slot6_ambiguous = False
        for c_id, f_id, sup in triples:
            if sup == 'slot6_ambiguous':
                has_slot6_ambiguous = True
                continue
            if 1 <= c_id <= NUM_COARSE_TAGS:
                t_row[c_id - 1] = 1.0
                w_row[c_id - 1] = 1.0
                if f_id > 0 and sup == 'hard':
                    t_row[NUM_COARSE_TAGS + f_id - 1] = 1.0
                    w_row[NUM_COARSE_TAGS + f_id - 1] = 1.0
                else:
                    coarse_only_families.add(c_id)
        for c_id in coarse_only_families:
            for sib_f in COARSE_TO_FINE_IDS[c_id]:
                col = NUM_COARSE_TAGS + sib_f - 1
                if t_row[col] < 0.5:
                    w_row[col] = 0.0
        if has_slot6_ambiguous:
            # Mask fine siblings in Piano (7) and Pluck (8) where unconfirmed.
            for c_id in (7, 8):
                for sib_f in COARSE_TO_FINE_IDS[c_id]:
                    col = NUM_COARSE_TAGS + sib_f - 1
                    if t_row[col] < 0.5:
                        w_row[col] = 0.0
            if t_row[6] < 0.5 and t_row[7] < 0.5:
                w_row[6] = 0.0
                w_row[7] = 0.0
                slot6_union[row] = True
        targets[row] = t_row
        weights[row] = w_row

    encoded.moment_type_targets = torch.from_numpy(targets)
    encoded.moment_type_weights = torch.from_numpy(weights)
    encoded.moment_slot6_union = torch.from_numpy(slot6_union)
    return encoded


def analyze_song_tag_tracks(events: list, metadata: Mapping | None = None, *,
                            max_coarse_subtypes: int = 2,
                            max_collision_rate: float = 0.05) -> dict:
    """Group song events into fine tracks and determine Coarse-promotion eligibility."""
    from .events import Event
    meta = metadata or {}
    grid = BeatGrid(meta)
    studio_inst = meta.get('studio_instruments')
    fine_ticks: dict[int, set[int]] = defaultdict(set)
    coarse_ticks: dict[int, set[int]] = defaultdict(set)
    coarse_has_ambiguous_subtypes: set[int] = set()
    slot6_ambiguous_count = 0

    for raw in events:
        ev = Event(**raw) if isinstance(raw, dict) else raw
        if ev.kind == 'note_off':
            continue
        tick = int(ev.data['start_tick'] if 'start_tick' in ev.data else grid.tick(ev.time))
        for c_id, f_id, sup in classify_event_tags(ev, studio_instruments=studio_inst):
            if sup == 'slot6_ambiguous' or c_id <= 0:
                slot6_ambiguous_count += 1
                continue
            coarse_ticks[c_id].add(tick)
            if f_id > 0 and sup == 'hard':
                fine_ticks[f_id].add(tick)
            else:
                coarse_has_ambiguous_subtypes.add(c_id)

    coarse_eligible: dict[int, dict] = {}
    for c_id, c_set in sorted(coarse_ticks.items()):
        sub_ids = [f_id for f_id in COARSE_TO_FINE_IDS[c_id] if fine_ticks.get(f_id)]
        total_sub_hits = sum(len(fine_ticks[f_id]) for f_id in sub_ids)
        union_sub_hits = len(set().union(*(fine_ticks[f_id] for f_id in sub_ids))) if sub_ids else 0
        collision_rate = (
            (total_sub_hits - union_sub_hits) / max(union_sub_hits, 1)
            if len(sub_ids) > 1 else 0.0
        )
        # Eligible for Coarse Macro Text Condition when:
        # 1) single/few subtypes without heavy co-onset interference, OR
        # 2) coarse_only track with zero conflicting fine tracks in the same macro family.
        low_mix = (
            len(sub_ids) <= max_coarse_subtypes
            and collision_rate <= max_collision_rate
            and not (c_id in coarse_has_ambiguous_subtypes and len(sub_ids) > 0)
        )
        coarse_eligible[c_id] = dict(
            coarse_id=c_id,
            coarse_tag=COARSE_TAGS[c_id - 1],
            moment_count=len(c_set),
            active_fine_ids=sub_ids,
            active_fine_tags=[FINE_TAGS[f - 1] for f in sub_ids],
            collision_rate=round(collision_rate, 4),
            promotable=bool(low_mix and not (c_id in (7, 8) and slot6_ambiguous_count > 0)),
        )

    return dict(
        fine_ticks={f: sorted(ts) for f, ts in sorted(fine_ticks.items())},
        coarse_ticks={c: sorted(ts) for c, ts in sorted(coarse_ticks.items())},
        coarse_eligibility=coarse_eligible,
        slot6_ambiguous_count=slot6_ambiguous_count,
        residual_mixed_intervals=merge_intervals(meta.get('residual_mixed_intervals')),
    )


def sample_tag_moment_charts(events: list, metadata: Mapping | None = None, *,
                             duration_seconds: float | None = None,
                             max_queries: int = 4,
                             coarse_prob: float = 0.5,
                             max_coarse_subtypes: int = 2,
                             max_collision_rate: float = 0.05,
                             min_events: int = 1,
                             rng: random.Random | None = None,
                             record_prefix: str = 'tag_moment') -> list[dict]:
    """Sample up to K non-interfering historyless MOMENT-only charts for Branch A."""
    from .codec import encode_moment_only_chart
    from .events import Event

    rng = rng or random.Random(0)
    meta = dict(metadata or {})
    analysis = analyze_song_tag_tracks(
        events, meta, max_coarse_subtypes=max_coarse_subtypes, max_collision_rate=max_collision_rate
    )
    fine_ticks = analysis['fine_ticks']
    coarse_ticks = analysis['coarse_ticks']
    eligibility = analysis['coarse_eligibility']
    residual = analysis['residual_mixed_intervals']

    # Build candidate pool: decide per Coarse family whether this draw promotes it
    # to a Coarse Macro Text Condition (C, 0) or keeps clean Fine tracks (C, f).
    candidates: list[tuple[int, int, list[int], float]] = []
    covered_coarse: set[int] = set()
    for c_id, info in eligibility.items():
        if info['moment_count'] < min_events:
            continue
        sub_ids = [f for f in info['active_fine_ids'] if len(fine_ticks.get(f, ())) >= min_events]
        if info['promotable'] and (not sub_ids or rng.random() < coarse_prob):
            ticks = coarse_ticks[c_id]
            candidates.append((c_id, 0, ticks, math.sqrt(len(ticks))))
            covered_coarse.add(c_id)
    for f_id, ticks in fine_ticks.items():
        c_id = FINE_TO_COARSE[f_id]
        if c_id in covered_coarse or len(ticks) < min_events:
            continue
        candidates.append((c_id, f_id, ticks, math.sqrt(len(ticks))))

    if not candidates or max_queries <= 0:
        return []

    chosen: list[tuple[int, int, list[int]]] = []
    pool = list(candidates)
    while pool and len(chosen) < max_queries:
        weights = [max(1e-3, item[3]) for item in pool]
        pick = rng.choices(range(len(pool)), weights=weights, k=1)[0]
        c_id, f_id, ticks, _ = pool.pop(pick)
        chosen.append((c_id, f_id, ticks))

    grid = BeatGrid(meta)
    charts: list[dict] = []
    for c_id, f_id, ticks in chosen:
        tag_label = format_tag_label(c_id, f_id)
        tag_events = [Event(grid.second(t), 'moment', dict(start_tick=int(t))) for t in ticks]
        chart_meta = {
            **meta,
            'query_tag': tag_label,
            'query_coarse_id': int(c_id),
            'query_fine_id': int(f_id),
            'moment_only': True,
            'no_history': True,
        }
        encoded = encode_moment_only_chart(
            'midi',
            tag_events,
            duration_seconds=duration_seconds,
            metadata=chart_meta,
            residual_intervals=residual,
        )
        charts.append(dict(
            encoded=encoded,
            task='midi',
            metadata=chart_meta,
            events=tag_events,
            record=dict(id=f'{record_prefix}/{tag_label}', dataset='tag_moment', pseudo=False),
        ))
    return charts
