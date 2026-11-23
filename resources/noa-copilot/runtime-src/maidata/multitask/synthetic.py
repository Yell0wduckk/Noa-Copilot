"""Seeded tick96 arrangements shared by studio rendering and MIDI/maimai supervision."""
from bisect import bisect_right
from collections import defaultdict
from dataclasses import asdict
from functools import lru_cache
import argparse
import json
import math
import re
from pathlib import Path
import random

import numpy as np
import torch
from .grid import BeatGrid
from .studio_mix import choice
from .config import read_studio_options, studio_options
from .studio_sources import PACKAGE, read_json
from .patterns import draw_phrases, mine_manifest

RATE = 44100
PITCHES = (36, 48, 93, 79, 67, 37, 72, 55)
ALL_METERS = tuple((n, d) for d in (2, 4, 8, 16) for n in range(1, 17))
_COMMON_METER_WEIGHTS = {
    (4, 4): 36.0, (3, 4): 10.0, (2, 4): 6.0, (6, 8): 8.0,
    (5, 4): 4.5, (7, 8): 4.5, (12, 8): 3.5, (9, 8): 3.0,
    (5, 8): 2.5, (7, 16): 1.5, (11, 8): 1.5, (15, 8): 1.5,
    (2, 2): 2.5, (3, 2): 1.5, (4, 2): 1.0, (3, 8): 2.0,
    (9, 16): 1.0, (13, 16): 1.0, (15, 16): 1.0, (5, 16): 1.0,
}
METER_WEIGHTS = tuple(_COMMON_METER_WEIGHTS.get(m, 0.25) for m in ALL_METERS)
TURNAROUND_METERS = ((2, 4), (1, 4), (7, 8), (3, 8), (5, 8), (3, 16), (5, 16))

GENRE_PROFILES = {
    "denpa_jcore": {
        "bpm_range": (164, 182),
        "scale": (0, 2, 4, 5, 7, 9, 11),
        "progressions": (
            (3, 4, 2, 5),  # IV - V - iii - vi (Royal Road)
            (5, 3, 0, 4),  # vi - IV - I - V
        ),
        "arp_pattern": (0, 1, 2, 3, 1, 2, 3, 2),
        "drum_style": "four_floor",
        "style_tags": "j_core,denpa",
    },
    "baroque_artcore": {
        "bpm_range": (172, 192),
        "scale": (0, 2, 3, 5, 7, 8, 11),  # Harmonic Minor
        "progressions": (
            (0, 3, 4, 0),  # i - iv - V7 - i
            (0, 5, 2, 4),  # i - VI - III - V7
        ),
        "arp_pattern": (0, 2, 1, 3, 2, 3, 1, 2),
        "drum_style": "artcore_break",
        "style_tags": "artcore,symphonic",
    },
    "cyberpunk_complextro": {
        "bpm_range": (168, 186),
        "scale": (0, 2, 3, 5, 7, 9, 10),  # Dorian
        "progressions": (
            (0, 3, 5, 4),  # i - IV7 - vi - v
            (0, 6, 5, 4),  # i - bVII - VI - v
        ),
        "arp_pattern": (0, 3, 1, 2, 3, 1, 2, 0),
        "drum_style": "dnb_complextro",
        "style_tags": "tech,trance",
    },
    "oriental_electro": {
        "bpm_range": (158, 176),
        "scale": (0, 1, 5, 7, 10, 12, 13),  # Japanese In-Sen / Pentatonic
        "progressions": (
            (4, 5, 0, 3),  # Quartal / Modal progression
            (2, 3, 4, 0),
        ),
        "arp_pattern": (0, 1, 3, 2, 0, 2, 3, 1),
        "drum_style": "four_floor",
        "style_tags": "oriental,j_core",
    },
}

@lru_cache(maxsize=8)
def asset(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig')) if path else None


def _sample_bpm(rng, bpm_range, integer_probability):
    low, high = float(bpm_range[0]), float(bpm_range[1])
    if math.ceil(low) <= math.floor(high) and rng.random() < integer_probability:
        return float(rng.randint(math.ceil(low), math.floor(high)))
    return round(rng.uniform(low, high), 2)


def _structural_bars(rng, opts):
    """Build the full-song tempo, meter and section timeline."""
    genre_name = opts.get("genre") or choice(rng, opts.get("genres", tuple(GENRE_PROFILES)))
    profile = GENRE_PROFILES[genre_name]
    bpm_range = opts.get("bpm_range", profile["bpm_range"])
    base_bpm = _sample_bpm(rng, bpm_range, opts['integer_bpm_probability'])
    bridge_bpm = round(max(82.0, base_bpm * rng.choice(opts['bridge_tempo_ratios'])), 2)
    climax_bpm = round(min(208.0, base_bpm + rng.choice(opts['climax_bpm_increments'])), 2)
    odd_meter_1 = rng.choice(((5, 4), (7, 8), (5, 8)))
    odd_meter_2 = rng.choice(((7, 4), (7, 8), (6, 8), (3, 4)))

    # Solo windows are drawn separately on the actual tempo map for both short and full songs.
    bars = []
    def add(count, section, tempo, meter=(4, 4), shift=0):
        bars.extend(dict(bpm=tempo, meter=meter, section=section, key_shift=shift) for _ in range(count))
    add(8, 'intro_duo', base_bpm)
    for section in ('verse_groove', 'verse_var'):
        for i in range(8):
            meter = ((3 if section=='verse_groove' else 5),4) if genre_name=='oriental_electro' and i%4==3 else (4,4)
            add(1, section, base_bpm, meter)
    add(6, 'prechorus_build_8th', base_bpm)
    add(5, 'prechorus_build_16th', base_bpm)
    add(1, 'lead_pickup', base_bpm)
    add(16, 'chorus_full', base_bpm)
    add(4, 'bridge_duo_bass_drums', base_bpm)
    add(8, 'verse_var', bridge_bpm, (3,4))
    for i in range(4):
        bars.append(dict(bpm=bridge_bpm+(climax_bpm-bridge_bpm)*i/4,
                         end_bpm=bridge_bpm+(climax_bpm-bridge_bpm)*(i+1)/4,
                         meter=(4,4), section='bridge_accel_build', key_shift=0))
    add(8, 'chorus_full', climax_bpm, shift=2)
    add(4, 'climax_odd_full', climax_bpm, odd_meter_1, 2)
    add(4, 'climax_odd_full', climax_bpm, odd_meter_2, 2)
    add(4, 'intro_duo', 118, shift=2)
    for bpm in (110,104,98,90):
        add(1, 'intro_duo', bpm, shift=2)
    return bars, genre_name, profile["style_tags"]


def arrangement(rng, opts):
    """BPM ramps are linear in seconds. Choose their lengths in score beats."""
    opts = studio_options(opts)
    bars = opts.get('layout')
    genre_name = opts.get('genre') or choice(rng, opts.get('genres', tuple(GENRE_PROFILES)))
    style_tags = GENRE_PROFILES[genre_name]['style_tags']
    if bars is None:
        explicit_bars = opts.get('bars')
        use_structural = opts.get('structural', explicit_bars is None or int(explicit_bars) >= 32)
        if use_structural and explicit_bars is None:
            bars, genre_name, style_tags = _structural_bars(rng, dict(opts, genre=genre_name))
        else:
            bpm_range = opts.get('bpm_range', (110, 190))
            bpm = _sample_bpm(rng, bpm_range, opts['integer_bpm_probability'])
            custom_meters = opts.get('meters')
            meter = (tuple(rng.choice(custom_meters)) if custom_meters else
                     (4, 4) if rng.random() < opts['initial_four_four_probability'] else rng.choices(ALL_METERS, weights=opts.get('meter_weights', METER_WEIGHTS), k=1)[0])
            bars = []
            num_bars = int(explicit_bars if explicit_bars is not None else 8)
            turnaround_restore = None
            for i in range(num_bars):
                if turnaround_restore is not None:
                    meter = turnaround_restore
                    turnaround_restore = None
                elif i and rng.random() < opts.get('meter_change_probability', .25):
                    if not custom_meters and i + 1 < num_bars and rng.random() < opts.get('turnaround_probability', .3):
                        turnaround_restore = meter
                        meter = rng.choice(TURNAROUND_METERS)
                    else:
                        meter = (tuple(rng.choice(custom_meters)) if custom_meters else
                                 rng.choices(ALL_METERS, weights=opts.get('meter_weights', METER_WEIGHTS), k=1)[0])
                if i and rng.random() < opts.get('tempo_change_probability', .4):
                    bpm = _sample_bpm(rng, bpm_range, opts['integer_bpm_probability'])
                end = (_sample_bpm(rng, bpm_range, opts['integer_bpm_probability']) if
                       rng.random() < opts.get('ramp_probability', .2) else bpm)
                bars.append(dict(bpm=bpm, end_bpm=end, meter=meter))
                bpm = end
    t, sec = 0, 0.
    tempos, meters, ranges = [], [], []
    sections, key_shifts = [], []
    for bar in bars:
        n, d = bar.get('meter', (4, 4))
        bpm = float(bar['bpm'])
        end = float(bar.get('end_bpm', bpm))
        ticks = max(24, round(384 * n / d))
        tempos.append(dict(start_seconds=sec, bpm=bpm, end_bpm=end))
        if not meters or meters[-1][1:] != [n, d]:
            meters.append([t / 96, n, d])
        ranges.append((t, t + ticks))
        sections.append(bar.get('section'))
        key_shifts.append(int(bar.get('key_shift', 0)))
        sec += (ticks / 96) * 120 / (bpm + end)
        t += ticks
    tempos.append(dict(start_seconds=sec, bpm=end))
    meta = dict(bpm_segments=tempos, offset_seconds=0., meter_segments=meters,
                numerator=meters[0][1], denominator=meters[0][2])
    if any(sections):
        meta['sections'] = sections
        meta['key_shifts'] = key_shifts
    if genre_name:
        meta['genre'] = genre_name
    if style_tags:
        meta.setdefault('style_tags', style_tags)
    return meta, ranges, t


def _structural_score(rng, bars, end, cooccurrence, opts, sections, key_shifts, genre_name):
    """Compose a sectioned ensemble from a seeded melodic template and chord progression."""
    profile = GENRE_PROFILES[genre_name]
    scale = tuple(opts.get("custom_scale") or profile["scale"])
    custom_progs = opts.get("custom_progressions")
    progs = tuple(tuple(p) for p in custom_progs) if custom_progs else profile["progressions"]
    arp_pat = tuple(opts.get("custom_arp_pattern") or profile["arp_pattern"])
    drum_style = opts.get("custom_drum_style") or profile["drum_style"]
    root_base = rng.choice(opts['tonics'])

    def deg_pitch(base_midi, degree, key_shift=0):
        octv, idx = divmod(int(degree), len(scale))
        return int(np.clip(base_midi + root_base + key_shift + octv * 12 + scale[idx], 28, 92))

    themes = read_json(PACKAGE/'studio-themes.json')[genre_name]
    custom_melody = opts.get("custom_melody")
    section_themes = opts.get('section_templates', {}) if not opts.get('template') else {}
    theme_names = {k:choice(rng,weights) for k, weights in section_themes.items()}
    # Transpose each studio theme as a unit, preserving its contour, rhythm and intervals.
    tonic = {'denpa_jcore':3,'baroque_artcore':0,'cyberpunk_complextro':6,'oriental_electro':2}[genre_name]
    transposition = (root_base-tonic+6)%12-6
    progress_start = rng.randrange(len(progs))

    score, chords = [], {}
    for b_idx, (a, b) in enumerate(bars):
        bar_ticks = b - a
        sec_type = sections[b_idx] or "chorus_full"
        k_shift = key_shifts[b_idx] if b_idx < len(key_shifts) else 0
        c_idx = b_idx % 4
        prog = progs[(progress_start + b_idx // 8) % len(progs)]
        chord_deg = prog[c_idx % len(prog)]
        chord_pitches = [deg_pitch(48, chord_deg + off, k_shift) for off in (0, 2, 4, 6)]
        chords[a] = chord_pitches
        bass_root = deg_pitch(36, chord_deg, k_shift)
        theme = custom_melody if custom_melody else themes[theme_names.get(sec_type,opts['_template'])]
        call_notes, resp_notes = ([(round(pos*96), int(pitch+transposition+k_shift), round(length*96))
                                   for pos,pitch,length in theme[part][c_idx % len(theme[part])]] for part in ('lead','resp'))

        if sec_type == "intro_duo":
            # The bass pedal supports the alternating pluck and lead outside designated solos.
            score.append([a,b,5,bass_root,opts['intro_bass_velocity']])
            for pos, deg, dur in call_notes:
                if a + pos < b:
                    score.append([a + pos, min(b, a + pos + dur), 6, deg, 0.88])
            for pos, deg, dur in resp_notes:
                if a + pos < b:
                    score.append([a + pos, min(b, a + pos + dur), 7, deg, 0.82])

        elif sec_type == "lead_pickup":
            # Lead Pickup (Slot 7) + 11-hit 16th Snare Roll Crescendo (Slot 1) + Reverse Cymbal (Slot 3)
            for pos, deg, dur in call_notes[:3]:
                if a + pos < b:
                    score.append([a + pos, min(b, a + pos + dur), 7, deg, 0.92])
            for s_i in range(11):
                t0 = a + s_i * 24
                if t0 < b:
                    vel = round(0.48 + 0.50 * (s_i / 10.0), 2)
                    score.append([t0, min(b, t0 + 20), 1, PITCHES[1], vel])
            if a + 264 < b:
                score.append([a + 264, min(b, a + 360), 3, PITCHES[3], 0.85])

        elif sec_type == "bridge_duo_bass_drums":
            # 2-Track Bass + Drum Groove
            for beat_t in range(a, b, 96):
                score.append([beat_t, min(b, beat_t + 18), 0, PITCHES[0], 0.88])
                if ((beat_t - a) // 96) % 2 == 1:
                    score.append([beat_t, min(b, beat_t + 20), 1, PITCHES[1], 0.82])
                for sub_i, off in ((1, 0), (2, 7), (3, 4)):
                    t_bs = beat_t + sub_i * 24
                    if t_bs < b:
                        score.append([t_bs, min(b, t_bs + 20), 5, deg_pitch(36, chord_deg + off, k_shift), 0.84])

        else:
            is_full = sec_type in ("chorus_full", "climax_odd_full")
            is_build = sec_type in ("prechorus_build_8th", "prechorus_build_16th", "bridge_accel_build")

            if c_idx == 0 and is_full:
                score.append([a, min(b, a + 60), 3, PITCHES[3], 0.88])

            # 1. Lead (Slot 4) + Synth Lead / Counter-Melody (Slot 7)
            for pos, deg, dur in call_notes:
                t0 = a + pos
                if t0 < b:
                    score.append([t0, min(b, t0 + dur), 4, deg, 0.94 if is_full else 0.84])
                    if is_full or is_build:
                        score.append([t0, min(b, t0 + dur), 7, deg, 0.86])
            for pos, deg, dur in resp_notes:
                t0 = a + pos
                if t0 < b:
                    score.append([t0, min(b, t0 + dur), 7 if is_full else 4, deg, 0.82])

            # Fill extra beats in odd meters (5/4, 7/4, 7/8)
            if bar_ticks > 384:
                for extra_t in range(a + 384, b, 48):
                    deg = rng.choice((4, 5, 6, 7))
                    score.append([extra_t, min(b, extra_t + 44), 7, deg_pitch(60, deg, k_shift), 0.88])

            # 2. Continuous 16th-Note Arpeggio (Slot 6)
            for step_i, t_arp in enumerate(range(a, b, 24)):
                if step_i and rng.random() > opts['backing_density']:
                    continue
                p_arp = chord_pitches[arp_pat[step_i % len(arp_pat)]] + 12
                vel_arp = 0.84 if step_i % 4 == 0 else (0.76 if step_i % 2 == 1 else 0.68)
                score.append([t_arp, min(b, t_arp + 20), 6, min(92, p_arp), vel_arp])

            # 3. Drums & Bass (Slots 0, 1, 2, 5)
            if is_build:
                step_ticks = 48 if sec_type == "prechorus_build_8th" else 24
                n_steps = max(1, bar_ticks // step_ticks)
                for s_i, t_sn in enumerate(range(a, b, step_ticks)):
                    vel_sn = round(0.48 + 0.48 * (s_i / max(1, n_steps - 1)), 2)
                    score.append([t_sn, min(b, t_sn + step_ticks - 4), 1, PITCHES[1], vel_sn])
                score.append([a, min(b, a + min(192, bar_ticks)), 5, bass_root, 0.78])
            elif drum_style == "dnb_complextro" and bar_ticks >= 384:
                # Two-Step DnB Breakbeat + 3+3+2 Syncopated Neuro Bass
                for k_off in (0, 240):
                    if a + k_off < b:
                        score.append([a + k_off, min(b, a + k_off + 18), 0, PITCHES[0], 0.94])
                for sn_off in (96, 288):
                    if a + sn_off < b:
                        score.append([a + sn_off, min(b, a + sn_off + 20), 1, PITCHES[1], 0.88])
                for t_hat in range(a, b, 24):
                    score.append([t_hat, min(b, t_hat + 8), 2, PITCHES[2], 0.58])
                for st_16, dur_16, off in ((0, 3, 0), (3, 3, 4), (6, 2, 7), (8, 3, 0), (11, 3, 6), (14, 2, 4)):
                    t_bs = a + st_16 * 24
                    if t_bs < b:
                        score.append([t_bs, min(b, t_bs + dur_16 * 22), 5, deg_pitch(36, chord_deg + off, k_shift), 0.88])
            else:
                for beat_i, beat_t in enumerate(range(a, b, 96)):
                    score.append([beat_t, min(b, beat_t + 18), 0, PITCHES[0], 0.92 if is_full else 0.80])
                    if beat_i % 2 == 1:
                        score.append([beat_t, min(b, beat_t + 20), 1, PITCHES[1], 0.84 if is_full else 0.72])
                    if beat_t + 48 < b:
                        score.append([beat_t + 48, min(b, beat_t + 56), 2, PITCHES[2], 0.64])
                    for sub_i, off in ((1, 0), (2, 7), (3, 0)):
                        t_bs = beat_t + sub_i * 24
                        if t_bs < b:
                            score.append([t_bs, min(b, t_bs + 20), 5, deg_pitch(36, chord_deg + off, k_shift), 0.86 if is_full else 0.74])

    # Preserve paired chart/MIDI rhythm and pitch motion in every arrangement length.
    pitches = list(PITCHES)
    starts = list(chords)
    present = {(r[0],r[2]) for r in score}
    for tick, slots in cooccurrence:
        if not 0 <= tick < end:
            continue
        bar = bisect_right(starts,tick)-1
        shift = key_shifts[bar] if key_shifts else 0
        classes = {(root_base+shift+degree)%12 for degree in scale}
        for original, direction in slots:
            if original==4 and not opts['cooccurrence_vocal']:
                continue
            slot = 2 if original==3 or (original==0 and (tick-starts[bar])%48) else original
            if (tick,slot) in present:
                continue
            if slot>=4:
                low,high = opts['cooccurrence_pitch_ranges'][str(slot)]
                pool = [p for p in range(low,high+1) if not opts['cooccurrence_in_key'] or p%12 in classes]
                index = min(range(len(pool)),key=lambda i:abs(pool[i]-pitches[slot]))
                pitches[slot] = pool[min(len(pool)-1,max(0,index+direction*rng.choice((1,2))))]
            velocity = opts['cooccurrence_crash_hat_velocity' if original==3 else 'cooccurrence_velocity']
            score.append([tick,min(end,tick+24),slot,pitches[slot],rng.uniform(*velocity),'cooccurrence'])
            present.add((tick,slot))
    return score, dict(chords=chords, root_base=root_base, scale=scale, sections=sections,
                       key_shifts=list(key_shifts) or [0]*len(bars), section_themes=theme_names)


def _score(rng, bars, end, cooccurrence, opts):
    sections = opts.get('_sections') or [rng.choice(('verse_groove','verse_var','chorus_full')) for _ in bars]
    return _structural_score(rng, bars, end, cooccurrence, opts, sections, opts.get('_key_shifts') or (),
                             opts['_genre'])


def compose(seed, options, with_chart=False):
    rng = random.Random(seed)
    opts = studio_options(options)
    meta, bars, end = arrangement(rng, opts)
    grid = BeatGrid(meta)
    bank = asset(opts.get('pattern_bank'))
    chart, co = draw_phrases(bank, rng, end, grid) if with_chart else ([], [])
    meta['studio_seed'] = seed
    meta['template'] = opts.get('template') or choice(rng, opts['templates'])
    score, harmony = _score(rng, bars, end, co, dict(opts, _sections=meta.get('sections'), _key_shifts=meta.get('key_shifts'),
                                            _genre=meta['genre'], _template=meta['template']))
    from .studio_daw import arrange_tracks, render
    meta.update(harmony)
    notes, cast = arrange_tracks(score, bars, grid, opts, rng, harmony)
    meta.update(cast)
    return render(notes, grid, end, meta, chart, opts, rng, keep_stems=options.get('export_stems', False))


def example(seed, options, *, with_chart=False):
    """Return stereo [samples,2]. The acoustic encoder downmixes at its usual input boundary."""
    song = compose(seed, options, with_chart)
    return torch.from_numpy(song.wave), song.events, song.chart, song.metadata


def sample_bank(manifest):
    from .bms import keysound_root_pitch
    from .data import read_manifest
    rows, _ = read_manifest(manifest)
    result = defaultdict(dict)
    for r in rows:
        if r.get('split', 'train') != 'train':
            continue
        for s in r.get('metadata', {}).get('samples', ()):
            a = s.get('analysis', {})
            if a.get('slot') is None or a.get('silent') or a.get('composite') or not s.get('rendered_duration'):
                continue
            slot = a['slot']
            if slot >= 4 and a.get('pitch') is None:
                continue
            start = s.get('source_offset', 0.) + a.get('onset_offset', 0.)
            # A keysound cut by its next BMS trigger still owns its entire dry one-shot.
            whole_hit = slot < 4 and r['metadata'].get('format') != 'bmson' and not s.get('source_offset')
            length = None if whole_hit else a['effective_duration']
            item = dict(path=s['path'], start=start, length=length, song=r['id'])
            if slot>=4:
                item.update(pitch=keysound_root_pitch(a['pitch'],a.get('fundamental_hz')),instrument=re.split(r'[_ -]v\d+|[_ -]\d+',Path(s['path']).stem)[0])
            if match := re.search(r'(?:^|_)v(\d{1,3})(?=[lo_])',Path(s['path']).stem,re.I):
                item['velocity']=min(1.,int(match[1])/127)
            key = item['path'], start
            old = result[str(slot)].get(key)
            if old is None or (length or 0) > (old.get('length') or 0):
                result[str(slot)][key] = item
    return {k: list(v.values()) for k, v in result.items()}


def export(output, seed, options):
    import soundfile as sf
    from .codec import count_chart_tokens
    from .midi import write_midi
    with_chart = bool(options.get('pattern_bank'))
    song = compose(seed, options, with_chart)
    output.mkdir(parents=True, exist_ok=True)
    sf.write(output/'mix.flac', song.wave, RATE, subtype='PCM_24')
    for role, stem in song.stems.items():
        # FLOAT preserves the pre-master amplitude of an individual stem, including peaks above 0 dBFS.
        sf.write(output/(role+'.wav'), stem, RATE, subtype='FLOAT')
    meta = song.metadata
    write_midi(song.events, output/'notes.mid', metadata=meta)
    if with_chart:
        from .maimai import write_simai
        write_simai(song.chart, output/'maidata.txt', bpm=meta['bpm_segments'][0]['bpm'],
                    segments=meta['bpm_segments'], title=f'Studio {seed}')
    rows = []
    duration = len(song.wave)/RATE
    for task, events in [('midi', song.events)] + ([('maimai', song.chart)] if with_chart else []):
        name = task+'.json'
        (output/name).write_text(json.dumps([asdict(e) for e in events]),encoding='utf-8')
        rows.append(dict(id=f'studio/{seed}/{task}', audio='mix.flac', events=name, duration_seconds=duration,
                         dataset='studio',task=task,split='train',source_family='studio',metadata=meta,
                         token_count=count_chart_tokens(task,events,metadata=meta,duration_seconds=duration)))
    (output/'manifest.json').write_text(json.dumps(dict(records=rows)),encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='cmd', required=True)
    mine = commands.add_parser('mine')
    mine.add_argument('--manifest', required=True)
    mine.add_argument('--phrase-ticks', type=int, default=96)
    mine.add_argument('--min-count', type=int, default=1)
    mine.add_argument('--max-patterns', type=int)
    mine.add_argument('--output', type=Path, required=True)
    samples = commands.add_parser('samples')
    samples.add_argument('--manifest', required=True)
    samples.add_argument('--output', type=Path, required=True)
    profile = commands.add_parser('profile', help='Fit the offline BMS balance and maimai spectral reference bank')
    profile.add_argument('--bms', nargs='*', default=[])
    profile.add_argument('--bms-manifest')
    profile.add_argument('--reference-manifest')
    profile.add_argument('--reference-root', type=Path)
    profile.add_argument('--base-profile', type=Path, help='Reuse prepared BMS balances when replacing only the references')
    profile.add_argument('--references', nargs='*', default=[])
    profile.add_argument('--limit', type=int)
    profile.add_argument('--seed', type=int, default=0)
    profile.add_argument('--options', type=Path)
    profile.add_argument('--output', type=Path, required=True)
    vet = commands.add_parser('vet-presets', help='Audition pitched Vital candidates and measure their sounding octave offline')
    vet.add_argument('--preset-index', type=Path)
    vet.add_argument('--probes', type=Path)
    vet.add_argument('--options', type=Path)
    vet.add_argument('--workers', type=int, default=4)
    vet.add_argument('--output', type=Path, required=True)
    for name in ('render','studio-render'):
        demo = commands.add_parser(name)
        demo.add_argument('--options', type=Path)
        demo.add_argument('--seed', type=int, default=0)
        demo.add_argument('--genre', choices=('all', *GENRE_PROFILES))
        demo.add_argument('--stems', action='store_true', help='Also export post-FX, post-fader stereo stems')
        demo.add_argument('--output', type=Path, required=name=='render', default=Path('artifacts/studio-preview'))
    args = parser.parse_args(argv)
    if args.cmd in ('render','studio-render'):
        opts = read_studio_options(args.options,Path.cwd()) if args.options else {}
        if args.stems:
            opts = {**opts,'export_stems':True}
        genres = tuple(GENRE_PROFILES) if args.genre=='all' else (args.genre or opts.get('genre'),)
        for i,genre in enumerate(genres):
            export(args.output/genre if len(genres)>1 else args.output, args.seed+i, {**opts,'genre':genre})
    elif args.cmd=='vet-presets':
        from .studio_presets import vet_presets
        vet_presets(args.output, args.preset_index, args.probes,
                    read_studio_options(args.options) if args.options else None, args.workers)
    elif args.cmd=='profile':
        from .studio_profile import profiles
        value = profiles(bms=args.bms,bms_manifest=args.bms_manifest,reference_manifest=args.reference_manifest,
                         references=args.references,reference_root=args.reference_root,base_profile=args.base_profile,
                         limit=args.limit,seed=args.seed,
                         options=read_studio_options(args.options) if args.options else None)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(value,ensure_ascii=False,indent=2), encoding='utf-8')
    else:
        value = (mine_manifest(args.manifest, width=args.phrase_ticks, min_count=args.min_count, max_patterns=args.max_patterns)
                 if args.cmd == 'mine' else sample_bank(args.manifest))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(value, separators=(',', ':')), encoding='utf-8')


if __name__ == '__main__':
    main()
