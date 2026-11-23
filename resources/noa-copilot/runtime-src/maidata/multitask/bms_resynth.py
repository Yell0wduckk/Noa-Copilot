"""Multi-layer DAW, TANO*C producer drum/FX pack, and external bus DSP resynthesis for BMS charts."""
from collections import defaultdict
import math
import os
from pathlib import Path
import re

import numpy as np
from scipy import ndimage, signal
import soundfile as sf
import soxr

from .bms import analyze_sample, parse_bms
from .loudness import integrated_loudness
from .sf2_synth import (
    extract_polyphonic_notes_from_waveform,
    render_hybrid_acoustic_stem,
    transcribe_long_bgm_stem,
)
from .studio_mix import FREQUENCIES, RATE, read_stereo, spectrum
from .studio_sources import ASSETS, Archive, drum_subtype, instrument, read_json, vital_preset, vst_path

MML_RE = re.compile(r"^(?:b_)?(.+?)_v(\d+)(?:l([\d-]+))?o([0-9])([a-g])([ps#m-]?)$", re.I)
NOTE_ONLY_RE = re.compile(r"^(?:b_)?(.+?)[_-]o?([0-9])([a-g])([ps#m-]?)$", re.I)
MID2BMS_NOTES_RE = re.compile(r"^(?:b_)?(.+?)_(\d{3,6})_([a-gps#m-]+)$", re.I)
NUMBERED_STEM_RE = re.compile(r"^(?:b_)?(.+?)_(?:\d{3}_)?(\d{3,6})$", re.I)
NOTE_TOKEN_RE = re.compile(r"([a-g])([ps#m-]?)", re.I)
SEMI_MAP = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}

DTS_MAP = {
    "d_ts_mainlead_1": ("mainlead", 73, 0.34),
    "d_ts_subleadlead_1": ("sublead", 80, 0.34),
    "d_ts_coverlead_1a": ("coverlead", 85, 0.34),
    "d_ts_coverlead_1b": ("coverlead", 68, 0.34),
    "d_ts_mainlead_2": ("mainlead", 66, 0.32),
    "d_ts_dublead_2": ("sublead", 78, 0.32),
    "d_ts_mainlead_3": ("mainlead", 64, 0.28),
    "d_ts_sublead_3": ("sublead", 76, 0.28),
}

NATIVE_P90_TARGETS_DB = {
    "kick": -20.50,
    "hardkick": -15.32,
    "snare": -21.02,
    "hat": -27.40,
    "amen_break": -23.00,
    "cymbal_fx": -23.10,
    "bass_sub": -23.30,
    "bass_mid": -28.60,
    "bass_hi": -27.50,
    "lead_main": -23.40,
    "lead_sub": -25.00,
    "lead_cover": -25.40,
    "lead_drop": -23.20,
    "lead_back": -26.20,
    "chord_stab": -25.60,
    "pad_chord": -25.60,
    "arp_stereo": -25.60,
    "arp_chip": -26.80,
    # Acoustic / Orchestral / Traditional instrument families
    "piano": -23.20,
    "epiano": -25.00,
    "organ": -25.50,
    "guitar_clean": -24.20,
    "guitar_drive": -23.40,
    "harp": -24.80,
    "violin": -23.80,
    "strings_stacc": -24.20,
    "strings_legato": -24.60,
    "brass": -24.00,
    "woodwinds": -24.20,
    "xylophone": -24.20,
    "vibraphone": -24.80,
    "musicbox": -25.20,
    "bell": -25.40,
    "koto": -23.80,
    "shamisen": -23.60,
    "taiko": -21.80,
    "timpani": -22.50,
    "orch_snare": -21.80,
}

RHYTHM_GROUPS = {
    "kick", "hardkick", "snare", "hat", "amen_break", "cymbal_fx",
    "bass_sub", "bass_mid", "bass_hi", "taiko", "timpani", "orch_snare",
}

ACOUSTIC_FAMILIES = (
    "piano", "epiano", "organ", "guitar_clean", "guitar_drive",
    "harp", "violin", "strings_stacc", "strings_legato",
    "brass", "woodwinds", "xylophone", "vibraphone",
    "musicbox", "bell", "koto", "shamisen",
    "taiko", "timpani", "orch_snare",
)

ACOUSTIC_STAGE_PAN = {
    "piano": 0.0,
    "epiano": -0.15,
    "organ": 0.12,
    "guitar_clean": -0.28,
    "guitar_drive": 0.26,
    "harp": 0.25,
    "violin": -0.20,
    "strings_stacc": -0.22,
    "strings_legato": 0.08,
    "brass": 0.22,
    "woodwinds": 0.15,
    "xylophone": -0.20,
    "vibraphone": 0.20,
    "musicbox": -0.16,
    "bell": 0.18,
    "koto": -0.25,
    "shamisen": 0.24,
    "taiko": 0.0,
    "timpani": -0.12,
    "orch_snare": 0.06,
}


def parse_mml_gate_sec(l_str: str | None, bar_sec: float) -> float:
    if not l_str:
        return 0.16
    parts = l_str.split("-")
    denom = float(parts[0])
    num = float(parts[1]) if len(parts) > 1 else 1.0
    if denom <= 0:
        return 0.16
    return max(0.025, (num / denom) * bar_sec)


def parse_mml_pitch(octv: str, note: str, acc: str, inst: str, l_str: str = "") -> int:
    semi = SEMI_MAP[note.lower()] + (1 if acc.lower() in ("p", "s", "#") else -1 if acc.lower() in ("m", "-") else 0)
    midi = int(octv) * 12 + semi
    if inst == "sublead" and l_str != "96-5":
        midi += 12
    return int(np.clip(midi, 12, 108))


def fold_synth_register(pitch: int, inst_grp: str) -> int:
    """Fold extreme octaves (e.g. o8..o9 or o3..o4) into each synth/lead family's audible melodic register."""
    p = int(pitch)
    if inst_grp in ("mainlead", "sublead", "coverlead", "coverlead_less", "droplead", "backlead"):
        while p < 60:
            p += 12
        while p > 84:
            p -= 12
    elif inst_grp in ("chip_loop", "chip_loop2", "backarp_l", "backarp_r"):
        while p < 55:
            p += 12
        while p > 86:
            p -= 12
    elif inst_grp in ("chordstab_l", "chordstab_r", "backstab_l", "backstab_r", "chordlong"):
        while p < 48:
            p += 12
        while p > 81:
            p -= 12
    elif inst_grp in ("lowbass", "lowbassroll"):
        while p > 47:
            p -= 12
        while p < 28:
            p += 12
    elif inst_grp in ("hibass", "hibass_lowcover"):
        while p > 55:
            p -= 12
        while p < 36:
            p += 12
    return int(np.clip(p, 12, 108))


def parse_mid2bms_note_tokens(raw_inst: str, note_str: str, gate_sec: float = 0.24) -> list[tuple[float, float, int, float]]:
    """Parse Mid2BMS multi-note/chord/phrase tokens like `b_11__fami_melo_00046_gp` or `b_06_piano_right_00072_gce`."""
    tokens = NOTE_TOKEN_RE.findall(note_str)
    if not tokens:
        return []
    s = raw_inst.lower()
    is_bass = any(k in s for k in ("bass", "bs", "reese", "sub_bass", "808"))
    is_lead = any(
        k in s
        for k in (
            "melo", "lead", "main", "solo", "slead", "violin", "vln",
            "dizi", "erhu", "flute", "synth_super", "synth_main", "synth_sub",
            "fami_riff", "riff_trible",
        )
    )
    base_oct = 3 if is_bass else (5 if is_lead else 4)
    pitches = []
    prev_midi = None
    for note_c, acc in tokens[:6]:
        semi = SEMI_MAP[note_c.lower()] + (1 if acc.lower() in ("p", "s", "#") else -1 if acc.lower() in ("m", "-") else 0)
        midi = base_oct * 12 + semi
        if prev_midi is not None and not is_lead and not is_bass:
            while midi <= prev_midi and midi + 12 <= (base_oct + 2) * 12:
                midi += 12
        pitches.append(int(np.clip(midi, 24, 96)))
        prev_midi = midi

    if is_lead or is_bass or len(pitches) == 1:
        step = max(0.05, gate_sec / max(1, len(pitches)))
        return [(idx * step, (idx + 1) * step, p, 0.86) for idx, p in enumerate(pitches[:4])]
    return [(0.0, max(0.20, gate_sec), p, 0.80) for p in pitches[:4]]


def lookahead_master_to_lufs(wave: np.ndarray, target_lufs: float = -9.5, ceiling_db: float = -1.0) -> np.ndarray:
    """Loudness-match any stereo mix to `target_lufs` with transparent true-peak lookahead limiting."""
    x = np.nan_to_num(wave.astype(np.float32, copy=True))
    sos_hp = signal.butter(2, 25.0, "highpass", fs=RATE, output="sos")
    x = signal.sosfilt(sos_hp, x, axis=0).astype(np.float32)

    cur_lufs = integrated_loudness(x, RATE)
    if math.isfinite(cur_lufs):
        x *= 10.0 ** ((target_lufs - cur_lufs) / 20.0)

    ceiling = 10.0 ** (ceiling_db / 20.0)
    for _ in range(3):
        up = signal.resample_poly(x, 2, 1, axis=0)
        peak_env = np.max(np.abs(up), axis=1)[::2][: len(x)]
        need_gain = np.minimum(1.0, ceiling / np.maximum(peak_env, 1e-9)).astype(np.float32)
        look = max(1, round(0.004 * RATE))
        rel = max(1, round(0.065 * RATE))
        g_min = ndimage.minimum_filter1d(need_gain, size=2 * look + 1, mode="nearest")
        alpha = math.exp(-1.0 / rel)
        g_smooth = signal.lfilter([1.0 - alpha], [1.0, -alpha], g_min).astype(np.float32)
        g_final = np.minimum(g_min, g_smooth)
        x = x * g_final[:, None]
        cur_lufs = integrated_loudness(x, RATE)
        if math.isfinite(cur_lufs) and abs(target_lufs - cur_lufs) > 0.15:
            x *= 10.0 ** ((target_lufs - cur_lufs) / 20.0)

    return np.clip(x, -ceiling, ceiling).astype(np.float32)


def pan_stereo(wave: np.ndarray, pan: float) -> np.ndarray:
    """Constant-power stereo panner (-1.0 = full left, 0.0 = center, +1.0 = full right)."""
    if abs(pan) < 1e-4:
        return wave
    angle = (pan + 1.0) * (math.pi / 4.0)
    gl = math.cos(angle)
    gr = math.sin(angle)
    out = wave.copy()
    if pan < 0:
        out[:, 0] = wave[:, 0] * gl + wave[:, 1] * (0.25 * abs(pan))
        out[:, 1] = wave[:, 1] * gr
    else:
        out[:, 0] = wave[:, 0] * gl
        out[:, 1] = wave[:, 1] * gr + wave[:, 0] * (0.25 * abs(pan))
    return out


def multiband_ott(wave: np.ndarray, mix_amount: float = 0.45, hp_hz: float = 215.0) -> np.ndarray:
    """External 3-Band Upward/Downward OTT-style Multiband Compressor for crystalline J-Core Lead & Chord sheen."""
    if np.max(np.abs(wave)) < 1e-6:
        return wave
    sos_hp = signal.butter(2, hp_hz, "highpass", fs=RATE, output="sos")
    clean_in = signal.sosfilt(sos_hp, wave, axis=0).astype(np.float32)

    sos_lowmid = signal.butter(2, [hp_hz, 880.0], "bandpass", fs=RATE, output="sos")
    sos_mid = signal.butter(2, [880.0, 3600.0], "bandpass", fs=RATE, output="sos")
    sos_hi = signal.butter(2, 3600.0, "highpass", fs=RATE, output="sos")

    b_low = signal.sosfilt(sos_lowmid, clean_in, axis=0)
    b_mid = signal.sosfilt(sos_mid, clean_in, axis=0)
    b_hi = signal.sosfilt(sos_hi, clean_in, axis=0)

    win = max(1, round(0.012 * RATE))
    kernel = np.ones(win, dtype=np.float32) / win

    def compress_band(band: np.ndarray, target_rms: float, up_ratio: float, down_ratio: float) -> np.ndarray:
        pwr = np.mean(band**2, axis=1)
        rms = np.sqrt(np.convolve(pwr, kernel, mode="same") + 1e-12)
        active = rms > (target_rms * 0.08)
        gain = np.ones_like(rms)
        over = rms > target_rms
        gain[over] = (target_rms / rms[over]) ** (1.0 - 1.0 / down_ratio)
        under = active & (rms < target_rms)
        gain[under] = np.minimum(2.1, (target_rms / rms[under]) ** (1.0 - 1.0 / up_ratio))
        return band * gain[:, None]

    c_low = compress_band(b_low, 0.068, 1.25, 3.8) * 0.86
    c_mid = compress_band(b_mid, 0.082, 1.95, 3.4) * 1.16
    c_hi = compress_band(b_hi, 0.056, 1.70, 3.6) * 1.22
    wet = np.tanh((c_low + c_mid + c_hi) * 1.06) / 1.06
    out = ((1.0 - mix_amount) * clean_in + mix_amount * wet).astype(np.float32)
    sos_air_soft = signal.butter(2, 11200.0, "lowpass", fs=RATE, output="sos")
    return (0.65 * out + 0.35 * signal.sosfilt(sos_air_soft, out, axis=0)).astype(np.float32)


def tempo_pingpong_delay(
    wave: np.ndarray,
    bpm: float = 185.0,
    beat_fraction: float = 0.75,
    feedback: float = 0.28,
    wet_level: float = 0.16,
    duck_under_dry: bool = True,
) -> np.ndarray:
    """Tempo-synced stereo ping-pong delay with bandpass filtering and strong dry-signal ducking for clarity."""
    if np.max(np.abs(wave)) < 1e-6:
        return wave
    delay_sec = (60.0 / bpm) * beat_fraction
    d_samples = max(1, round(delay_sec * RATE))
    sos_bp = signal.butter(2, [520.0, 5200.0], "bandpass", fs=RATE, output="sos")
    src = signal.sosfilt(sos_bp, wave, axis=0).astype(np.float32)
    mono_src = src.mean(axis=1)

    echo_l = np.zeros(len(wave), dtype=np.float32)
    echo_r = np.zeros(len(wave), dtype=np.float32)
    for tap in range(1, 5):
        offset = tap * d_samples
        if offset >= len(wave):
            break
        g = feedback ** (tap - 1)
        if tap % 2 == 1:
            echo_l[offset:] += mono_src[:-offset] * g
            echo_r[offset:] += mono_src[:-offset] * (g * 0.30)
        else:
            echo_r[offset:] += mono_src[:-offset] * g
            echo_l[offset:] += mono_src[:-offset] * (g * 0.30)

    wet = np.stack([echo_l, echo_r], axis=1)
    if duck_under_dry:
        win = max(1, round(0.020 * RATE))
        dry_env = np.sqrt(np.convolve(np.mean(wave**2, axis=1), np.ones(win, dtype=np.float32) / win, mode="same"))
        duck = 1.0 / (1.0 + (dry_env / 0.045) ** 1.8)
        wet *= (0.20 + 0.80 * duck)[:, None]
    return (wave + wet * wet_level).astype(np.float32)


def lush_stereo_reverb(wave: np.ndarray, rt60: float = 1.05, wet_level: float = 0.14, predelay_ms: float = 24.0) -> np.ndarray:
    """True stereo decorrelated plate reverb with 480 Hz high-pass and dynamic dry-ducking (keysound-choke style)."""
    if np.max(np.abs(wave)) < 1e-6:
        return wave
    rng_ir = np.random.default_rng(18500)
    ir_len = round(rt60 * RATE)
    t = np.arange(ir_len, dtype=np.float32) / RATE
    decay = np.exp(-6.908 * t / rt60)[:, None]
    noise_l = rng_ir.standard_normal(ir_len).astype(np.float32)
    noise_r = rng_ir.standard_normal(ir_len).astype(np.float32)
    ir = np.stack([noise_l, noise_r], axis=1) * decay
    sos_bp = signal.butter(2, [480.0, 7600.0], "bandpass", fs=RATE, output="sos")
    ir = signal.sosfilt(sos_bp, ir, axis=0).astype(np.float32)
    predelay = round(predelay_ms * RATE / 1000.0)
    ir = np.pad(ir, ((predelay, 0), (0, 0)))
    ir /= max(float(np.sqrt(np.sum(ir**2, axis=0).mean())), 1e-8)

    wet_l = signal.oaconvolve(wave[:, 0], ir[:, 0], mode="full")[: len(wave)]
    wet_r = signal.oaconvolve(wave[:, 1], ir[:, 1], mode="full")[: len(wave)]
    wet = np.stack([wet_l, wet_r], axis=1).astype(np.float32)
    win = max(1, round(0.025 * RATE))
    dry_env = np.sqrt(np.convolve(np.mean(wave**2, axis=1), np.ones(win, dtype=np.float32) / win, mode="same"))
    duck = 1.0 / (1.0 + (dry_env / 0.05) ** 1.6)
    wet *= (0.28 + 0.72 * duck)[:, None]
    return (wave + wet * wet_level).astype(np.float32)


def apply_tape_stop(snippet: np.ndarray) -> np.ndarray:
    """Physical decelerating tape-stop resampler (rate drops from 1.0 -> 0.10 over snippet duration)."""
    n = len(snippet)
    if n < 64:
        return snippet
    u = np.linspace(0.0, 1.0, n, dtype=np.float64)
    speed = np.maximum(0.08, 1.0 - 0.90 * (u**1.8))
    src_idx = np.cumsum(speed)
    src_idx = src_idx * (min(n - 1, int(n * 0.58)) / max(src_idx[-1], 1.0))
    idx0 = np.clip(np.floor(src_idx).astype(int), 0, n - 2)
    frac = (src_idx - idx0).astype(np.float32)[:, None]
    out = snippet[idx0] * (1.0 - frac) + snippet[idx0 + 1] * frac
    fade = min(round(0.03 * RATE), n // 4)
    if fade > 0:
        out[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)[:, None]
    return out.astype(np.float32)


def render_custom_vital_layer(
    rel_preset_path: str,
    pitch_offset: int,
    note_events: list,
    frames: int,
    *,
    vital_lib_root: Path | None = None,
    vital_vst: str | None = None,
    polyphony: int = 12,
    max_attack_sec: float = 0.008,
    release_sec: float = 0.16,
    min_sustain: float | None = None,
    min_decay_sec: float | None = None,
    cutoff_override: float | None = None,
    mono_choke: bool = False,
    open_dark_filters: bool = True,
    max_detune: float = 2.0,
    disable_chorus: bool = True,
    disable_sub_osc3: bool = False,
) -> np.ndarray:
    """Render a Vital preset with open filters, tight unison detune, and monophonic keysound-channel choke."""
    if not note_events:
        return np.zeros((frames, 2), dtype=np.float32)

    lib_root = vital_lib_root or (ASSETS / "vital-presets-library")
    vst_bin = vital_vst or vst_path(os.environ.get("VITAL_VST3"))
    preset_file = lib_root / rel_preset_path
    plugin, initial = instrument(vst_bin)
    patch = read_json(preset_file)
    settings = dict(patch["settings"])
    settings["delay_on"] = 0.0
    settings["reverb_on"] = 0.0
    if disable_chorus:
        settings["chorus_on"] = 0.0
    settings["polyphony"] = float(1 if mono_choke else polyphony)
    settings["legato"] = 0.0
    for env_idx in (1, 2, 3):
        atk_key = f"env_{env_idx}_attack"
        if atk_key in settings:
            settings[atk_key] = min(float(settings[atk_key]), max_attack_sec * (1.0 if env_idx == 1 else 1.8))
    if "env_1_release" in settings:
        settings["env_1_release"] = float(np.clip(settings["env_1_release"], 0.035 if mono_choke else 0.06, release_sec))
    if min_sustain is not None and "env_1_sustain" in settings:
        settings["env_1_sustain"] = max(float(settings["env_1_sustain"]), float(min_sustain))
    if min_decay_sec is not None and "env_1_decay" in settings:
        settings["env_1_decay"] = max(float(settings["env_1_decay"]), float(min_decay_sec))
    if open_dark_filters:
        for fi in (1, 2):
            if settings.get(f"filter_{fi}_on", 0.0) > 0.5 and float(settings.get(f"filter_{fi}_cutoff", 100.0)) < 82.0:
                settings[f"filter_{fi}_on"] = 0.0
    if cutoff_override is not None:
        settings["filter_1_on"] = 1.0
        settings["filter_1_cutoff"] = float(cutoff_override)
    if disable_sub_osc3 and int(settings.get("osc_3_transpose", 0)) <= -12:
        settings["osc_3_on"] = 0.0
    for i in (1, 2, 3):
        settings[f"osc_{i}_random_phase"] = 0.0
        det_k = f"osc_{i}_unison_detune"
        if det_k in settings:
            settings[det_k] = min(float(settings[det_k]), float(max_detune))

    plugin.preset_data = vital_preset(initial, dict(patch, settings=settings, preset_name=str(preset_file)))

    messages = []
    if mono_choke:
        evs_all = sorted(
            [
                (
                    float(start_s),
                    float(end_s),
                    int(np.clip(pitch - pitch_offset, 12, 115)),
                    int(np.clip(round(vel * 127), 8, 127)),
                )
                for start_s, end_s, pitch, vel in note_events
            ],
            key=lambda x: x[0],
        )
        dedup_evs = []
        for ev in evs_all:
            if dedup_evs and abs(ev[0] - dedup_evs[-1][0]) < 0.004:
                if ev[3] > dedup_evs[-1][3]:
                    dedup_evs[-1] = ev
            else:
                dedup_evs.append(ev)
        for idx, (s_sec, e_sec, key, vel_int) in enumerate(dedup_evs):
            if idx + 1 < len(dedup_evs):
                e_sec = min(e_sec, dedup_evs[idx + 1][0] - 0.002)
            e_sec = max(s_sec + 0.016, e_sec)
            messages.append((s_sec, 1, key, vel_int))
            messages.append((e_sec, 0, key, 0))
    else:
        by_pitch = defaultdict(list)
        for start_s, end_s, pitch, vel in note_events:
            key = int(np.clip(pitch - pitch_offset, 12, 115))
            by_pitch[key].append((float(start_s), float(end_s), int(np.clip(round(vel * 127), 8, 127))))

        for key, evs in by_pitch.items():
            evs.sort(key=lambda x: x[0])
            for idx, (s_sec, e_sec, vel_int) in enumerate(evs):
                if idx + 1 < len(evs):
                    e_sec = min(e_sec, evs[idx + 1][0] - 0.0025)
                e_sec = max(s_sec + 0.018, e_sec)
                messages.append((s_sec, 1, key, vel_int))
                messages.append((e_sec, 0, key, 0))

    messages.sort(key=lambda x: (x[0], x[1], x[2]))
    stream = [(bytes([0x90 if on else 0x80, k, v]), t_sec) for t_sec, on, k, v in messages]
    duration = float((frames + RATE) / RATE)
    out = plugin(stream, duration=duration, sample_rate=RATE, num_channels=2, buffer_size=512, reset=True)
    return np.ascontiguousarray(out.T[:frames], dtype=np.float32)


def classify_drum_waveform(w_raw: np.ndarray, stem_name: str) -> str:
    """Classify an individual drum-kit piece waveform by its spectral band distribution and duration."""
    mono = np.mean(w_raw, axis=1) if w_raw.ndim == 2 else w_raw
    dur = len(mono) / float(RATE)
    if len(mono) < 64:
        return "hihat"
    spec = np.abs(np.fft.rfft(mono))
    freqs = np.fft.rfftfreq(len(mono), 1.0 / RATE)
    low_e = float(np.sum(spec[(freqs >= 20.0) & (freqs < 180.0)] ** 2))
    mid_e = float(np.sum(spec[(freqs >= 180.0) & (freqs < 2500.0)] ** 2))
    hi_e = float(np.sum(spec[freqs >= 2500.0] ** 2))
    tot_e = low_e + mid_e + hi_e + 1e-12
    low_r = low_e / tot_e
    hi_r = hi_e / tot_e
    s = stem_name.lower()
    if "orch" in s:
        if low_r >= 0.45:
            return "taiko"
        if hi_r >= 0.50:
            return "crash"
        return "orch_snare"
    if low_r >= 0.42:
        return "kick"
    if hi_r >= 0.58:
        if dur >= 0.90:
            return "crash"
        if dur >= 0.28:
            return "hihatrenda"
        return "hihat"
    return "snare"


def classify_generic_inst_name(raw_inst: str, pitch: int, gate_sec: float) -> str:
    """Map any BMS MML or keysound instrument stem name into acoustic/orchestral/traditional or synth/drum groups."""
    s = raw_inst.lower().strip()
    if s in (
        "mainlead", "sublead", "coverlead", "coverlead_less", "droplead", "backlead",
        "chordstab_l", "chordstab_r", "backstab_l", "backstab_r", "chordlong",
        "backarp_l", "backarp_r", "chip_loop", "chip_loop2",
        "lowbass", "lowbassroll", "hibass_lowcover", "hibass",
        "kick", "hardkick", "hard kick cover", "snare", "hihat", "hihatrenda",
        "amen", "crash", "crashshort", "reversecym_long", "reversecym_short",
        "impact", "downfx",
    ):
        return s

    # 1. Traditional Oriental & Orchestral Percussion / Solo Strings & Winds
    if any(k in s for k in ("shamisen", "sami", "biwa")):
        return "shamisen"
    if any(k in s for k in ("koto", "guzheng", "chinapluck", "sitar")):
        return "koto"
    if any(k in s for k in ("taiko", "wadaiko", "grancassa")):
        return "taiko"
    if any(k in s for k in ("timpani", "timp", "tinp")):
        return "timpani"
    if any(k in s for k in ("orchestral snare", "orchsnare", "orch_snare", "snareroll", "marchingsnare")):
        return "orch_snare"
    if any(k in s for k in ("erhu", "niko", "kokyu")):
        return "violin"
    if any(
        k in s
        for k in (
            "dizi", "xiao", "shinobue", "shakuhachi", "fue",
            "woodwind", "flute", "piccolo", "oboe", "clarinet", "bassoon", "recorder",
        )
    ):
        return "woodwinds"

    # 2. Acoustic / Band / Orchestral Pitched Instruments
    if any(k in s for k in ("epiano", "elpiano", "introep", "rhodes", "wurli")):
        return "epiano"
    if any(k in s for k in ("organ", "hammond", "church")):
        return "organ"
    if any(k in s for k in ("piano", "pf", "steinway", "grand", "keys")) or s == "key":
        return "piano"
    if any(k in s for k in ("musicbox", "music_box", "orgel", "celesta", "glock")):
        return "musicbox"
    if any(k in s for k in ("xylophone", "xylo", "marimba")):
        return "xylophone"
    if any(k in s for k in ("vibraphone", "vibra", "vibes", "mallet")):
        return "vibraphone"
    if any(k in s for k in ("tubular", "bell", "chime", "kalimba", "ferrum", "metal")):
        return "bell"
    if "harp" in s:
        return "harp"
    if any(k in s for k in ("guitar", "guiter", "gt", "acogt", "elecgt", "accgit")):
        if any(k in s for k in ("clean", "aco", "nylon", "jazz", "accgit")):
            return "guitar_clean"
        return "guitar_drive" if (gate_sec >= 0.20 or pitch < 60) else "guitar_clean"
    if any(k in s for k in ("violin", "vln", "fiddle", "viola", "cello")):
        return "violin"
    if any(k in s for k in ("stacc", "pizz", "piz", "spicc", "shortstr", "tightstr", "st_sta")):
        return "strings_stacc"
    if any(k in s for k in ("strings", "str", "orche", "oke_pa")):
        return "strings_stacc" if gate_sec < 0.22 else "strings_legato"
    if any(k in s for k in ("brass", "horn", "trumpet", "trombone", "tuba", "sax")):
        return "brass"

    # 3. Electronic Drum / FX detection
    if any(k in s for k in ("revcym", "reverse")):
        return "reversecym_long" if gate_sec >= 0.9 else "reversecym_short"
    if any(k in s for k in ("impact", "boom", "hit", "explosion")):
        return "impact"
    if any(k in s for k in ("down", "fall", "subdrop")) and "lead" not in s:
        return "downfx"
    if any(k in s for k in ("amen", "break", "loop")) and "chip" not in s:
        return "amen"
    if any(k in s for k in ("shaker", "tamb", "noiz")):
        return "hihat"
    if any(k in s for k in ("fill", "tom")):
        return "snare"
    if any(k in s for k in ("drums", "drumkit", "krush_drum")):
        return "kick"
    d_sub = drum_subtype(s)
    if d_sub == "kick":
        return "hardkick" if any(k in s for k in ("hard", "dist", "gabber", "punch")) else "kick"
    if d_sub in ("snare", "clap"):
        return "snare"
    if d_sub in ("closed_hat", "open_hat", "shaker", "tamb", "perc"):
        return "hihatrenda" if d_sub == "open_hat" else "hihat"
    if d_sub in ("crash", "china", "ride"):
        return "crash"

    # 4. Bass detection (by explicit bass keywords, without hijacking low-octave leads/chords)
    if any(k in s for k in ("subbass", "lowbass", "sub_bass", "808")):
        return "lowbass"
    if any(k in s for k in ("bass", "bs", "reese", "donk", "acid")):
        return "hibass" if pitch >= 40 else "lowbass"

    # 5. Explicit Synth Melody / Lead detection MUST precede generic `"synth"` / `"riff"` / `"chord"`
    if any(k in s for k in ("droplead", "subdroplead", "screech", "supersaw", "synth_super")):
        return "droplead"
    if any(
        k in s
        for k in (
            "mainlead", "synth_main", "lead_main", "slead", "melo", "lead", "solo",
            "01_fami", "fami_riff", "riff_trible",
        )
    ):
        if any(k in s for k in ("sub", "2", "harm", "back")) and "main" not in s:
            return "sublead"
        return "mainlead"
    if any(k in s for k in ("sublead", "synth_sub")):
        return "sublead"
    if any(k in s for k in ("coverlead", "cover")):
        return "coverlead"
    if any(k in s for k in ("backlead", "counter", "call", "resp")):
        return "backlead"

    # 6. Synth Chiptune / Arp / Pad / Chord Stab detection
    if any(k in s for k in ("chip", "8bit", "nes", "gb", "square", "fami")) and not any(k in s for k in ("cord", "chord")):
        return "chip_loop" if (hash(s) % 2 == 0) else "chip_loop2"
    if any(k in s for k in ("arp", "seq", "pluck", "plk")) and not any(k in s for k in ("cord", "chord")):
        return "backarp_l" if ("_l" in s or hash(s) % 2 == 0) else "backarp_r"
    if any(k in s for k in ("pad", "choir", "atmos", "long")) or gate_sec >= 0.72:
        return "chordlong"
    if any(k in s for k in ("chord", "cord", "stab", "saw", "synth", "riff")):
        if "_r" in s or "2" in s:
            return "chordstab_r"
        if "back" in s or "sub" in s:
            return "backstab_l" if (hash(s) % 2 == 0) else "backstab_r"
        return "chordstab_l"
    if any(k in s for k in ("drop", "screech")):
        return "droplead"
    if any(k in s for k in ("sub", "harm", "high", "oct")) or pitch >= 76:
        return "sublead"
    return "mainlead"


def find_tanoc(pattern: str, tanoc_root: Path | None = None) -> Path:
    root = tanoc_root or (ASSETS / "vocal-and-tanoc-samples-library")
    matches = [p for p in root.rglob(pattern) if not p.name.startswith("._")]
    if not matches:
        raise FileNotFoundError(f"Could not find TANO*C sample for {pattern}")
    return matches[0]


def place_oneshot(stem_buf: np.ndarray, clip_wav: np.ndarray, t_sec: float, gain: float, max_dur_sec: float | None = None):
    at = round(t_sec * RATE)
    if at >= len(stem_buf):
        return
    src = clip_wav if max_dur_sec is None else clip_wav[: round(max_dur_sec * RATE)]
    n = min(len(src), len(stem_buf) - at)
    if n <= 0:
        return
    part = src[:n].copy()
    fade = min(round(0.004 * RATE), n // 4)
    if fade > 0:
        part[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)[:, None]
    stem_buf[at : at + n] += part * float(gain)


def render_native_bms(
    bms_path: Path,
    collection: str = "gdbg_bms_package_2022",
    song_like: str | None = None,
    target_lufs: float = -9.5,
    arch_root: Path | None = None,
) -> tuple[np.ndarray, dict]:
    """Render the original BMS audio from the archive for side-by-side reference comparison."""
    chart = parse_bms(Path(bms_path))
    total_duration_sec = max((t.time for t in chart.triggers), default=1.0) + 5.0
    total_frames = int(math.ceil(total_duration_sec * RATE))
    arch = Archive(arch_root or (ASSETS / "bms-samples-l4-20261106/final-download"))

    fmap = {}
    song_filter = song_like or Path(bms_path).stem.split("--")[-1]
    for path, base in arch.db.execute(
        "SELECT path, basename FROM files WHERE collection=? AND song LIKE ?",
        (collection, f"%{song_filter}%"),
    ):
        fmap[Path(base).stem.lower()] = path

    waveforms = {}
    for wid, wname in chart.samples.items():
        stem = Path(wname).stem.lower()
        if stem.startswith("c_vc_"):
            continue
        if stem in fmap:
            waveforms[wid] = arch.audio(collection, fmap[stem])

    unique_trig = {}
    for t in chart.triggers:
        if t.sound not in waveforms:
            continue
        key = (round(t.time * RATE), t.voice)
        if key not in unique_trig:
            unique_trig[key] = t

    sorted_trig = sorted(unique_trig.values(), key=lambda t: (t.time, t.sound))
    next_time_wid, native_render_list = {}, []
    for t in reversed(sorted_trig):
        w = waveforms[t.sound]
        length = len(w)
        if t.voice in next_time_wid:
            length = min(length, max(0, round((next_time_wid[t.voice] - t.time) * RATE)))
        next_time_wid[t.voice] = t.time
        native_render_list.append((t, length))
    native_render_list.reverse()

    native_mix = np.zeros((total_frames, 2), dtype=np.float32)
    for t, length in native_render_list:
        if length <= 0:
            continue
        part = waveforms[t.sound][:length].copy()
        if length < len(waveforms[t.sound]):
            fade = min(round(0.0025 * RATE), length // 4)
            if fade > 0:
                part[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)[:, None]
        at = round(t.time * RATE)
        end_at = min(total_frames, at + len(part))
        if end_at > at:
            native_mix[at:end_at] += part[: end_at - at]

    native_master = lookahead_master_to_lufs(native_mix, target_lufs)
    return native_master, {"chart": chart, "sorted_triggers": sorted_trig, "fmap": fmap, "waveforms": waveforms}


def render_bms_multilayer_resynth(
    bms_path: Path,
    *,
    bpm: float | None = None,
    target_lufs: float = -9.5,
    reference_wave: np.ndarray | None = None,
    analyze_non_mml_waveforms: dict | None = None,
) -> np.ndarray:
    """Resynthesize any BMS chart using the 32-layer Vital + TANO*C + External Bus DSP pipeline (0 original samples)."""
    chart = parse_bms(Path(bms_path))
    bpm_val = float(bpm or getattr(chart, "bpm", 185.0) or 185.0)
    bar_sec = 240.0 / bpm_val
    total_duration_sec = max((t.time for t in chart.triggers), default=1.0) + 5.0
    total_frames = int(math.ceil(total_duration_sec * RATE))

    unique_trig = {}
    for t in chart.triggers:
        if t.sound not in chart.samples:
            continue
        stem = Path(chart.samples[t.sound]).stem.lower()
        if stem.startswith("c_vc_"):
            continue
        key = (round(t.time * RATE), t.voice)
        if key not in unique_trig:
            unique_trig[key] = t
    sorted_trig = sorted(unique_trig.values(), key=lambda t: (t.time, t.sound))

    inst_notes = defaultdict(list)
    dts_events = []
    filter_events = []
    unlabeled_pitched_triggers = []
    analyzed_cache = {}
    drum_piece_cache = {}
    song_name = Path(bms_path).stem.split("--")[-1]

    for t in sorted_trig:
        wname = chart.samples[t.sound]
        stem = Path(wname).stem.lower()
        m = MML_RE.match(stem)
        if m:
            raw_inst, vel_s, l_str, octv, note_c, acc = m.groups()
            vel = float(np.clip(int(vel_s) / 100.0, 0.06, 1.0))
            gate_sec = parse_mml_gate_sec(l_str, bar_sec)
            pitch = parse_mml_pitch(octv, note_c, acc, raw_inst.lower(), l_str or "")
            raw_low = raw_inst.lower()
            if (
                any(k in raw_low for k in ("drums", "drumkit", "orchdrum", "acdrum", "krush_drum"))
                and analyze_non_mml_waveforms
                and t.sound in analyze_non_mml_waveforms
            ):
                if t.sound not in drum_piece_cache:
                    drum_piece_cache[t.sound] = classify_drum_waveform(analyze_non_mml_waveforms[t.sound], raw_low)
                inst = drum_piece_cache[t.sound]
            else:
                inst = classify_generic_inst_name(raw_inst, pitch, gate_sec)
                pitch = fold_synth_register(pitch, inst)
            tag = f"o{octv}{note_c}{acc}".lower()
            inst_notes[inst].append((t.time, t.time + gate_sec, pitch, vel, tag))
        elif stem in DTS_MAP:
            inst, pitch, dur_s = DTS_MAP[stem]
            dts_events.append((t.time, inst, pitch, dur_s))
        elif stem.startswith("c_filter_") or "riser" in stem or "sweep" in stem:
            filter_events.append((t.time, stem))
        else:
            m2 = NOTE_ONLY_RE.match(stem)
            if m2:
                raw_inst, octv, note_c, acc = m2.groups()
                pitch = parse_mml_pitch(octv, note_c, acc, raw_inst.lower(), "")
                raw_low = raw_inst.lower()
                if (
                    any(k in raw_low for k in ("drums", "drumkit", "orchdrum", "acdrum", "krush_drum"))
                    and analyze_non_mml_waveforms
                    and t.sound in analyze_non_mml_waveforms
                ):
                    if t.sound not in drum_piece_cache:
                        drum_piece_cache[t.sound] = classify_drum_waveform(analyze_non_mml_waveforms[t.sound], raw_low)
                    inst = drum_piece_cache[t.sound]
                else:
                    inst = classify_generic_inst_name(raw_inst, pitch, 0.22)
                    pitch = fold_synth_register(pitch, inst)
                tag = f"o{octv}{note_c}{acc}".lower()
                inst_notes[inst].append((t.time, t.time + 0.22, pitch, 0.85, tag))
            else:
                # Non-MML sample: check long BGM stems, multi-piece drum stems, percussion stems, Mid2BMS note tokens, and polyphonic pitched stems
                w_raw = analyze_non_mml_waveforms.get(t.sound) if analyze_non_mml_waveforms else None
                m3 = MID2BMS_NOTES_RE.match(stem)
                stem_inst_name = m3.group(1) if m3 else stem

                if w_raw is not None and "bgm" in stem and len(w_raw) > int(4.0 * RATE):
                    bgm_events = transcribe_long_bgm_stem(w_raw, t.time, bar_sec)
                    for s_b, e_b, p_b, v_b in bgm_events.get("bass", ()):
                        inst_notes["lowbass"].append((s_b, e_b, p_b, v_b, "bgm"))
                    for s_b, e_b, p_b, v_b in bgm_events.get("strings_legato", ()):
                        inst_notes["strings_legato"].append((s_b, e_b, p_b, v_b, "bgm"))
                    for s_b, e_b, p_b, v_b in bgm_events.get("piano", ()):
                        inst_notes["piano"].append((s_b, e_b, p_b, v_b, "bgm"))
                    continue

                if w_raw is not None and any(k in stem for k in ("drums", "drumkit", "orchdrum", "acdrum", "krush_drum")):
                    if t.sound not in drum_piece_cache:
                        drum_piece_cache[t.sound] = classify_drum_waveform(w_raw, stem)
                    d_inst = drum_piece_cache[t.sound]
                    inst_notes[d_inst].append((t.time, t.time + 0.22, 38, 0.85, "o5c"))
                    continue

                inst_guess = classify_generic_inst_name(stem_inst_name, 60, min(2.5, (len(w_raw) / RATE) if w_raw is not None else 0.22))
                if inst_guess in ("taiko", "timpani", "orch_snare"):
                    inst_notes[inst_guess].append((t.time, t.time + 0.35, 45, 0.88, "o5c"))
                    continue

                d_sub = drum_subtype(stem_inst_name)
                if d_sub is not None and inst_guess not in ACOUSTIC_FAMILIES:
                    inst = classify_generic_inst_name(stem_inst_name, 36, 0.18)
                    inst_notes[inst].append((t.time, t.time + 0.18, 36, 0.85, "o5c"))
                elif w_raw is not None:
                    if t.sound not in analyzed_cache:
                        dur_w = min(2.8, max(0.08, len(w_raw) / RATE))
                        inst_cand = classify_generic_inst_name(stem_inst_name, 60, dur_w)
                        if inst_cand in ACOUSTIC_FAMILIES:
                            poly_notes = extract_polyphonic_notes_from_waveform(w_raw, inst_cand, max_polyphony=3)
                            if poly_notes:
                                analyzed_cache[t.sound] = [(inst_cand, s_o, e_o, m_p, v_p) for s_o, e_o, m_p, v_p in poly_notes]
                            elif m3:
                                parsed_evs = parse_mid2bms_note_tokens(m3.group(1), m3.group(3), gate_sec=min(0.45, dur_w))
                                analyzed_cache[t.sound] = [(inst_cand, s_o, e_o, m_p, v_p) for s_o, e_o, m_p, v_p in parsed_evs]
                            else:
                                analyzed_cache[t.sound] = [(inst_cand, 0.0, min(0.5, dur_w), 60, 0.75)]
                        else:
                            fam_for_pitch = "bass" if ("bass" in inst_cand or "bs" in stem_inst_name) else "piano"
                            max_poly = 1 if ("lead" in inst_cand or fam_for_pitch == "bass") else 2
                            poly_notes = extract_polyphonic_notes_from_waveform(w_raw, fam_for_pitch, max_polyphony=max_poly)
                            if poly_notes:
                                analyzed_cache[t.sound] = [
                                    (
                                        classify_generic_inst_name(stem_inst_name, m_p, e_o - s_o),
                                        s_o,
                                        e_o,
                                        fold_synth_register(m_p, classify_generic_inst_name(stem_inst_name, m_p, e_o - s_o)),
                                        v_p,
                                    )
                                    for s_o, e_o, m_p, v_p in poly_notes
                                ]
                            elif m3:
                                parsed_evs = parse_mid2bms_note_tokens(m3.group(1), m3.group(3), gate_sec=min(0.35, dur_w))
                                analyzed_cache[t.sound] = [
                                    (
                                        classify_generic_inst_name(m3.group(1), m_p, e_o - s_o),
                                        s_o,
                                        e_o,
                                        fold_synth_register(m_p, classify_generic_inst_name(m3.group(1), m_p, e_o - s_o)),
                                        v_p,
                                    )
                                    for s_o, e_o, m_p, v_p in parsed_evs
                                ]
                            else:
                                info = analyze_sample(w_raw, filename=wname)
                                p_det = info.get("pitch")
                                if p_det is not None:
                                    inst_det = classify_generic_inst_name(stem_inst_name, int(p_det), min(1.5, dur_w))
                                    p_folded = fold_synth_register(int(p_det), inst_det)
                                    analyzed_cache[t.sound] = [(inst_det, 0.0, min(1.5, dur_w), p_folded, 0.82)]
                                else:
                                    d_cls = classify_drum_waveform(w_raw, stem)
                                    analyzed_cache[t.sound] = [(d_cls, 0.0, 0.18, 38, 0.82)]
                    for inst_det, s_o, e_o, p_det, v_det in analyzed_cache[t.sound]:
                        inst_notes[inst_det].append((t.time + s_o, t.time + e_o, p_det, v_det, "o5c"))
                elif m3:
                    # Mid2BMS note/chord token filename without archived audio waveform
                    raw_inst, _slice_id, note_tokens = m3.groups()
                    for s_o, e_o, m_p, v_p in parse_mid2bms_note_tokens(raw_inst, note_tokens, gate_sec=0.22):
                        inst_det = classify_generic_inst_name(raw_inst, m_p, e_o - s_o)
                        p_folded = fold_synth_register(m_p, inst_det)
                        inst_notes[inst_det].append((t.time + s_o, t.time + e_o, p_folded, v_p, "o5c"))
                else:
                    m4 = NUMBERED_STEM_RE.match(stem)
                    if m4:
                        raw_inst, num_s = m4.groups()
                        inst_det = classify_generic_inst_name(raw_inst, 67, 0.20)
                        if inst_det not in RHYTHM_GROUPS and inst_det not in ("impact", "downfx", "reversecym_long", "reversecym_short"):
                            unlabeled_pitched_triggers.append((t.time, inst_det, int(num_s)))

    # If the chart has unlabeled numbered melodic triggers whose waveforms were missing from the archive
    # (e.g. `01_fami_0004`..`01_fami_0289` in incomplete BMS packs), harmonize them against simultaneous song notes
    if unlabeled_pitched_triggers:
        ref_pool = []
        for grp_k, evs_k in inst_notes.items():
            if grp_k in RHYTHM_GROUPS:
                continue
            for s_k, _e_k, p_k, _v_k, _tag_k in evs_k:
                ref_pool.append((s_k, p_k % 12))
        ref_pool.sort(key=lambda x: x[0])
        ref_times = np.array([x[0] for x in ref_pool], dtype=np.float64) if ref_pool else np.zeros(0, dtype=np.float64)
        scale_pcs = [0, 2, 4, 5, 7, 9, 11]
        for idx_u, (t_u, inst_u, num_u) in enumerate(unlabeled_pitched_triggers):
            next_t = unlabeled_pitched_triggers[idx_u + 1][0] if idx_u + 1 < len(unlabeled_pitched_triggers) else t_u + 0.22
            gate_u = float(np.clip(next_t - t_u, 0.09, 0.32))
            if len(ref_times) > 0:
                pos = int(np.clip(np.searchsorted(ref_times, t_u), 0, len(ref_times) - 1))
                root_pc = ref_pool[pos][1]
            else:
                root_pc = 0
            deg = scale_pcs[(num_u * 3 + idx_u) % len(scale_pcs)]
            pitch_u = fold_synth_register(60 + ((root_pc + deg) % 12) + 12 * ((num_u % 2)), inst_u)
            inst_notes[inst_u].append((t_u, t_u + gate_u, pitch_u, 0.86, "o5c"))

    # Reinforce sparse synth lead stacks so single-stem BMS melodies get the full 3-layer unison body
    if not inst_notes["mainlead"] and inst_notes["sublead"]:
        inst_notes["mainlead"] = [
            (s, e, fold_synth_register(p, "mainlead"), v, tag)
            for s, e, p, v, tag in inst_notes["sublead"]
        ]
    if inst_notes["mainlead"] and len(inst_notes["sublead"]) < max(8, len(inst_notes["mainlead"]) // 3):
        inst_notes["sublead"].extend(
            (s, e, fold_synth_register(p + 12, "sublead"), v * 0.92, tag)
            for s, e, p, v, tag in inst_notes["mainlead"]
        )
    if inst_notes["mainlead"] and len(inst_notes["coverlead"]) < max(8, len(inst_notes["mainlead"]) // 3):
        inst_notes["coverlead"].extend(
            (s, e, fold_synth_register(p, "coverlead"), v * 0.90, tag)
            for s, e, p, v, tag in inst_notes["mainlead"]
        )

    # 1. 5-Layer Lead Stack with Monophonic Keysound Choke & Open Harmonic Sheen
    ev_mainlead = [(s, e, p, v) for s, e, p, v, _ in inst_notes["mainlead"]]
    w_mainlead_1 = render_custom_vital_layer(
        "repos/kawaii-core-bank-for-vital-vol.1/_unpacked_Kawaii Core Bank for Vital Vol.1/Kawaii Core Bank for Vital Vol.1/Presets/LD Clean.vital",
        0, ev_mainlead, total_frames, mono_choke=True, max_attack_sec=0.003, release_sec=0.045, max_detune=1.35,
    )
    w_mainlead_2 = render_custom_vital_layer(
        "repos/kawaii-core-bank-for-vital-vol.1/_unpacked_Kawaii Core Bank for Vital Vol.1/Kawaii Core Bank for Vital Vol.1/Presets/LD FM Lead.vital",
        0, ev_mainlead, total_frames, mono_choke=True, disable_sub_osc3=True, max_attack_sec=0.002, release_sec=0.045, max_detune=1.35,
    )
    w_mainlead = w_mainlead_1 * 0.68 + w_mainlead_2 * 0.42

    w_sublead = render_custom_vital_layer(
        "repos/kawaii-core-bank-for-vital-vol.1/_unpacked_Kawaii Core Bank for Vital Vol.1/Kawaii Core Bank for Vital Vol.1/Presets/LD FM Lead.vital",
        0, [(s, e, p, v) for s, e, p, v, _ in inst_notes["sublead"]], total_frames,
        mono_choke=True, disable_sub_osc3=True, max_attack_sec=0.003, release_sec=0.045, max_detune=1.40,
    )
    w_coverlead = render_custom_vital_layer(
        "repos/kawaii-core-bank-for-vital-vol.1/_unpacked_Kawaii Core Bank for Vital Vol.1/Kawaii Core Bank for Vital Vol.1/Presets/PD Kawaii Saw Chord.vital",
        0, [(s, e, p, v) for s, e, p, v, _ in inst_notes["coverlead"] + inst_notes["coverlead_less"]], total_frames,
        polyphony=12, max_attack_sec=0.004, release_sec=0.08, max_detune=1.80,
    )
    ev_droplead = [(s, e, p, v) for s, e, p, v, _ in inst_notes["droplead"]]
    w_droplead_1 = render_custom_vital_layer(
        "repos/kawaii-core-bank-for-vital-vol.1/_unpacked_Kawaii Core Bank for Vital Vol.1/Kawaii Core Bank for Vital Vol.1/Presets/LD shake leeeed.vital",
        0, ev_droplead, total_frames, mono_choke=True, max_attack_sec=0.003, release_sec=0.045, max_detune=1.20,
    )
    w_droplead_2 = render_custom_vital_layer(
        "repos/kawaii-core-bank-for-vital-vol.1/_unpacked_Kawaii Core Bank for Vital Vol.1/Kawaii Core Bank for Vital Vol.1/Presets/LD Chip Lead 2.vital",
        0, ev_droplead, total_frames, mono_choke=True, max_attack_sec=0.002, release_sec=0.045, max_detune=1.10,
    )
    w_droplead = w_droplead_1 * 0.65 + w_droplead_2 * 0.45

    w_backlead = render_custom_vital_layer(
        "repos/kawaii-core-bank-for-vital-vol.1/_unpacked_Kawaii Core Bank for Vital Vol.1/Kawaii Core Bank for Vital Vol.1/Presets/LD Pulse Lead.vital",
        0, [(s, e, p, v) for s, e, p, v, _ in inst_notes["backlead"]], total_frames,
        mono_choke=True, max_attack_sec=0.003, release_sec=0.050, max_detune=1.25,
    )

    if dts_events:
        dts_notes_by_inst = defaultdict(list)
        for t_sec, inst, pitch, dur_s in dts_events:
            dts_notes_by_inst[inst].append((t_sec, t_sec + dur_s, pitch, 0.95))
        w_dts_raw = render_custom_vital_layer(
            "repos/kawaii-core-bank-for-vital-vol.1/_unpacked_Kawaii Core Bank for Vital Vol.1/Kawaii Core Bank for Vital Vol.1/Presets/LD Clean.vital",
            0, [ev for evs in dts_notes_by_inst.values() for ev in evs], total_frames,
            polyphony=8, max_attack_sec=0.003, release_sec=0.08, max_detune=1.35,
        )
        for t_sec, _, _, dur_s in dts_events:
            a = round(t_sec * RATE)
            b = min(total_frames, a + round((dur_s + 0.06) * RATE))
            if b > a + 64:
                w_dts_raw[a:b] = apply_tape_stop(w_dts_raw[a:b].copy())
        w_mainlead += w_dts_raw

    # 2. L/R Dual-Tracked Chord Stabs & Sustained Pad
    ev_cstab_l = [(s, e, p, v) for s, e, p, v, _ in inst_notes["chordstab_l"]]
    ev_cstab_r = [(s, e, p, v) for s, e, p, v, _ in inst_notes["chordstab_r"]]
    ev_bstab_l = [(s, e, p, v) for s, e, p, v, _ in inst_notes["backstab_l"]]
    ev_bstab_r = [(s, e, p, v) for s, e, p, v, _ in inst_notes["backstab_r"]]

    w_cstab_l_saw = render_custom_vital_layer(
        "repos/kawaii-core-bank-for-vital-vol.1/_unpacked_Kawaii Core Bank for Vital Vol.1/Kawaii Core Bank for Vital Vol.1/Presets/PD Kawaii Saw Chord.vital",
        0, ev_cstab_l, total_frames, polyphony=16, max_attack_sec=0.004, release_sec=0.09, max_detune=1.85,
    )
    w_cstab_l_atk = render_custom_vital_layer(
        "repos/ArcerionDev__vital_discord_preset_archive/no_author/_unpacked_Rabbit_House_Otoge_Sound_Bank/Rabbit House Otoge Sound Bank/Presets/chord pluck.vital",
        0, ev_cstab_l, total_frames, polyphony=16, max_attack_sec=0.002, release_sec=0.08, open_dark_filters=True, max_detune=1.75,
    )
    w_cstab_l = pan_stereo(w_cstab_l_saw * 0.65 + w_cstab_l_atk * 0.52, -0.50)

    w_cstab_r_saw = render_custom_vital_layer(
        "repos/kawaii-core-bank-for-vital-vol.1/_unpacked_Kawaii Core Bank for Vital Vol.1/Kawaii Core Bank for Vital Vol.1/Presets/PD Kawaii Saw Chord.vital",
        0, ev_cstab_r, total_frames, polyphony=16, max_attack_sec=0.004, release_sec=0.09, min_sustain=0.75, max_detune=1.85,
    )
    w_cstab_r_atk = render_custom_vital_layer(
        "repos/kawaii-core-bank-for-vital-vol.1/_unpacked_Kawaii Core Bank for Vital Vol.1/Kawaii Core Bank for Vital Vol.1/Presets/PL Thick Pluck.vital",
        0, ev_cstab_r, total_frames, polyphony=16, max_attack_sec=0.002, release_sec=0.08, min_sustain=0.50, open_dark_filters=True, max_detune=1.75,
    )
    w_cstab_r = pan_stereo(w_cstab_r_saw * 0.65 + w_cstab_r_atk * 0.52, +0.48)

    w_bstab_l = pan_stereo(
        render_custom_vital_layer(
            "repos/kawaii-core-bank-for-vital-vol.1/_unpacked_Kawaii Core Bank for Vital Vol.1/Kawaii Core Bank for Vital Vol.1/Presets/PL Comb ChordPluck.vital",
            0, ev_bstab_l, total_frames, polyphony=12, max_attack_sec=0.003, release_sec=0.08, min_sustain=0.55, open_dark_filters=True, max_detune=1.75,
        ),
        -0.42,
    )
    w_bstab_r = pan_stereo(
        render_custom_vital_layer(
            "repos/kawaii-core-bank-for-vital-vol.1/_unpacked_Kawaii Core Bank for Vital Vol.1/Kawaii Core Bank for Vital Vol.1/Presets/PL Comb ChordPluck.vital",
            0, ev_bstab_r, total_frames, polyphony=12, max_attack_sec=0.003, release_sec=0.08, min_sustain=0.55, open_dark_filters=True, max_detune=1.75,
        ),
        +0.42,
    )
    w_chord_stab = w_cstab_l + w_cstab_r + w_bstab_l * 0.75 + w_bstab_r * 0.75

    ev_pad = [(s, e, p, v) for s, e, p, v, _ in inst_notes["chordlong"]]
    w_pad_1 = render_custom_vital_layer(
        "repos/kawaii-core-bank-for-vital-vol.1/_unpacked_Kawaii Core Bank for Vital Vol.1/Kawaii Core Bank for Vital Vol.1/Presets/PD Soft Super Saw Chord.vital",
        0, ev_pad, total_frames, polyphony=16, max_attack_sec=0.020, release_sec=0.18, min_sustain=0.80, open_dark_filters=True, max_detune=1.95,
    )
    w_pad_2 = render_custom_vital_layer(
        "repos/kawaii-core-bank-for-vital-vol.1/_unpacked_Kawaii Core Bank for Vital Vol.1/Kawaii Core Bank for Vital Vol.1/Presets/PD Kawaii Saw Chord.vital",
        0, ev_pad, total_frames, polyphony=16, max_attack_sec=0.020, release_sec=0.18, min_sustain=0.85, max_detune=1.95,
    )
    w_pad_chord = w_pad_1 * 0.60 + w_pad_2 * 0.60

    # 3. L/R Dual-Tracked Arpeggios & Chiptune Layers
    ev_arp_l = [(s, e, p, v) for s, e, p, v, _ in inst_notes["backarp_l"]]
    ev_arp_r = [(s, e, p, v) for s, e, p, v, _ in inst_notes["backarp_r"]]
    w_arp_l_1 = render_custom_vital_layer(
        "repos/kawaii-core-bank-for-vital-vol.1/_unpacked_Kawaii Core Bank for Vital Vol.1/Kawaii Core Bank for Vital Vol.1/Presets/PL Kawaii Space.vital",
        12, ev_arp_l, total_frames, polyphony=8, max_attack_sec=0.002, release_sec=0.07, min_sustain=0.55, min_decay_sec=0.22, open_dark_filters=True, max_detune=1.60,
    )
    w_arp_l_2 = render_custom_vital_layer(
        "repos/kawaii-core-bank-for-vital-vol.1/_unpacked_Kawaii Core Bank for Vital Vol.1/Kawaii Core Bank for Vital Vol.1/Presets/LD Pulse Lead.vital",
        0, ev_arp_l, total_frames, polyphony=8, max_attack_sec=0.003, release_sec=0.07, open_dark_filters=True, max_detune=1.40,
    )
    w_arp_l = pan_stereo(w_arp_l_1 * 0.68 + w_arp_l_2 * 0.52, -0.40)

    w_arp_r_1 = render_custom_vital_layer(
        "repos/kawaii-core-bank-for-vital-vol.1/_unpacked_Kawaii Core Bank for Vital Vol.1/Kawaii Core Bank for Vital Vol.1/Presets/PL Thick Pluck.vital",
        0, ev_arp_r, total_frames, polyphony=8, max_attack_sec=0.002, release_sec=0.07, min_sustain=0.55, min_decay_sec=0.22, open_dark_filters=True, max_detune=1.60,
    )
    w_arp_r_2 = render_custom_vital_layer(
        "repos/kawaii-core-bank-for-vital-vol.1/_unpacked_Kawaii Core Bank for Vital Vol.1/Kawaii Core Bank for Vital Vol.1/Presets/LD Clean.vital",
        0, ev_arp_r, total_frames, polyphony=8, max_attack_sec=0.003, release_sec=0.07, max_detune=1.40,
    )
    w_arp_r = pan_stereo(w_arp_r_1 * 0.68 + w_arp_r_2 * 0.48, +0.50)
    w_arp_stereo = lush_stereo_reverb(
        tempo_pingpong_delay(multiband_ott(w_arp_l + w_arp_r, 0.40, hp_hz=240.0), bpm_val, beat_fraction=0.5, feedback=0.24, wet_level=0.15),
        rt60=0.95, wet_level=0.12,
    )

    w_chip_l = pan_stereo(
        render_custom_vital_layer(
            "repos/kawaii-core-bank-for-vital-vol.1/_unpacked_Kawaii Core Bank for Vital Vol.1/Kawaii Core Bank for Vital Vol.1/Presets/LD Chip Lead 2.vital",
            0, [(s, e, p, v) for s, e, p, v, _ in inst_notes["chip_loop"]], total_frames, polyphony=8, max_attack_sec=0.002, release_sec=0.09,
        ),
        -0.55,
    )
    w_chip_r = pan_stereo(
        render_custom_vital_layer(
            "repos/ArcerionDev__vital_discord_preset_archive/no_author/_unpacked_Rabbit_House_Otoge_Sound_Bank/Rabbit House Otoge Sound Bank/Presets/8bit Lead.vital",
            12, [(s, e, p, v) for s, e, p, v, _ in inst_notes["chip_loop2"]], total_frames, polyphony=8, max_attack_sec=0.002, release_sec=0.09,
        ),
        +0.55,
    )
    w_arp_chip = tempo_pingpong_delay(w_chip_l + w_chip_r, bpm_val, beat_fraction=0.5, feedback=0.25, wet_level=0.22)

    # 4. 3-Layer Bass Stack (Sub + Low-Mid Cover + Gritty Mid-Bass)
    ev_sub = [(s, e, p, v) for s, e, p, v, _ in inst_notes["lowbass"] + inst_notes["lowbassroll"]]
    w_bass_sub_vital = render_custom_vital_layer(
        "repos/kawaii-core-bank-for-vital-vol.1/_unpacked_Kawaii Core Bank for Vital Vol.1/Kawaii Core Bank for Vital Vol.1/Presets/BS Short Sub Bass.vital",
        0, ev_sub, total_frames, polyphony=2, max_attack_sec=0.003, release_sec=0.08,
    )
    w_bass_sub_sine = np.zeros((total_frames, 2), dtype=np.float32)
    for s_sec, e_sec, pitch, vel in sorted(ev_sub, key=lambda x: x[0]):
        a = round(s_sec * RATE)
        b = min(total_frames, round((e_sec + 0.025) * RATE))
        if b <= a + 8:
            continue
        f0 = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
        t_arr = np.arange(b - a, dtype=np.float32) / RATE
        sig = (np.sin(2.0 * math.pi * f0 * t_arr) + 0.30 * np.sin(4.0 * math.pi * f0 * t_arr)) * vel
        atk = min(round(0.003 * RATE), len(sig) // 3)
        rel = min(round(0.015 * RATE), len(sig) // 3)
        if atk > 0:
            sig[:atk] *= np.linspace(0.0, 1.0, atk, dtype=np.float32)
        if rel > 0:
            sig[-rel:] *= np.linspace(1.0, 0.0, rel, dtype=np.float32)
        w_bass_sub_sine[a:b, 0] += sig
        w_bass_sub_sine[a:b, 1] += sig

    sos_sub_lp = signal.butter(2, 160.0, "lowpass", fs=RATE, output="sos")
    w_bass_sub = signal.sosfilt(sos_sub_lp, w_bass_sub_vital * 0.65 + w_bass_sub_sine * 0.55, axis=0).astype(np.float32)

    w_bass_mid = render_custom_vital_layer(
        "repos/kawaii-core-bank-for-vital-vol.1/_unpacked_Kawaii Core Bank for Vital Vol.1/Kawaii Core Bank for Vital Vol.1/Presets/BS Super Saw Bass.vital",
        0, [(s, e, p, v) for s, e, p, v, _ in inst_notes["hibass_lowcover"]], total_frames, polyphony=4, max_attack_sec=0.004, release_sec=0.10,
    )
    sos_bmid_bp = signal.butter(2, [85.0, 650.0], "bandpass", fs=RATE, output="sos")
    w_bass_mid = signal.sosfilt(sos_bmid_bp, w_bass_mid, axis=0).astype(np.float32)

    ev_hibass = [(s, e, p, v) for s, e, p, v, _ in inst_notes["hibass"]]
    w_bass_hi_1 = render_custom_vital_layer(
        "repos/kawaii-core-bank-for-vital-vol.1/_unpacked_Kawaii Core Bank for Vital Vol.1/Kawaii Core Bank for Vital Vol.1/Presets/BS Acid.vital",
        -12, ev_hibass, total_frames, polyphony=4, max_attack_sec=0.003, release_sec=0.10,
    )
    w_bass_hi_2 = render_custom_vital_layer(
        "repos/kawaii-core-bank-for-vital-vol.1/_unpacked_Kawaii Core Bank for Vital Vol.1/Kawaii Core Bank for Vital Vol.1/Presets/BS Solid Bass.vital",
        -12, ev_hibass, total_frames, polyphony=4, max_attack_sec=0.003, release_sec=0.10,
    )
    sos_bhi_bp = signal.butter(2, [95.0, 2200.0], "bandpass", fs=RATE, output="sos")
    w_bass_hi = np.tanh(signal.sosfilt(sos_bhi_bp, w_bass_hi_1 * 0.65 + w_bass_hi_2 * 0.55, axis=0) * 1.4).astype(np.float32)

    # 5. TANO*C Producer Drums, Sliced Breakbeat (`amen`), & FX
    wav_kick = read_stereo(find_tanoc("*Zekk_Drum_TK_Kick_01 - F#.wav"))[: round(0.28 * RATE)]
    wav_hkick_1 = read_stereo(find_tanoc("*LUHP_Hard_Kick_Gallery_01_G_Maximized.wav"))[: round(0.32 * RATE)]
    wav_hkick_2 = read_stereo(find_tanoc("*LUHP_Rawphoric_Kick_F#1_Maximized.wav"))[: round(0.32 * RATE)]
    wav_snare_1 = read_stereo(find_tanoc("*Zekk_Drum_TK_Snare_01 - F#.wav"))[: round(0.26 * RATE)]
    wav_snare_2 = read_stereo(find_tanoc("*LUHP_Hard_Snare_01.wav"))[: round(0.26 * RATE)]
    wav_snare = wav_snare_1 * 0.68 + wav_snare_2[: len(wav_snare_1)] * 0.48
    wav_hat_c = read_stereo(find_tanoc("*Zekk_Drum_TK_Closed Hihats - 1.wav"))[: round(0.14 * RATE)]
    wav_hat_o = read_stereo(find_tanoc("*Zekk_Drum_TK_Open Hihats - 1.wav"))[: round(0.32 * RATE)]
    wav_crash = read_stereo(find_tanoc("*SHC Crash 1.wav"))
    wav_impact = read_stereo(find_tanoc("*LUHP_FX_Impact_01.wav"))
    wav_down = read_stereo(find_tanoc("*YUKIYANAGI Sweep Down 01.wav"))
    wav_riser = read_stereo(find_tanoc("*170 Riser FX 01.wav"))

    wav_break_raw = read_stereo(find_tanoc("*SHC Drum Loop 1.wav"))
    wav_break_bpm = soxr.resample(wav_break_raw, round(RATE * (bpm_val / 170.0)), RATE, quality="HQ")
    sos_amen_hp = signal.butter(2, 60.0, "highpass", fs=RATE, output="sos")
    wav_break_bpm = signal.sosfilt(sos_amen_hp, wav_break_bpm, axis=0).astype(np.float32)
    step_16th = max(1, round((60.0 / bpm_val / 4.0) * RATE))
    break_slices = []
    for i in range(16):
        a = (i * step_16th) % max(1, len(wav_break_bpm) - step_16th)
        sl = wav_break_bpm[a : a + round(step_16th * 1.25)].copy()
        fade = min(round(0.008 * RATE), len(sl) // 3)
        if fade > 0:
            sl[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)[:, None]
        break_slices.append(sl)

    AMEN_SLICE_IDX = {
        "o5c": 0, "o5cp": 10, "o5g": 8, "o5d": 4, "o5dp": 12, "o5e": 4,
        "o5a": 12, "o5ap": 14, "o5f": 2, "o5fp": 6, "o5gp": 11, "o5b": 15,
    }

    w_kick = np.zeros((total_frames, 2), dtype=np.float32)
    w_hardkick = np.zeros((total_frames, 2), dtype=np.float32)
    w_snare = np.zeros((total_frames, 2), dtype=np.float32)
    w_hat = np.zeros((total_frames, 2), dtype=np.float32)
    w_amen = np.zeros((total_frames, 2), dtype=np.float32)
    w_cym_fx = np.zeros((total_frames, 2), dtype=np.float32)

    kick_times = []
    for s_sec, _, _, vel, _ in inst_notes["kick"]:
        place_oneshot(w_kick, wav_kick, s_sec, vel**1.1, max_dur_sec=0.24)
        kick_times.append(s_sec)
    for s_sec, _, _, vel, _ in inst_notes["hardkick"]:
        place_oneshot(w_hardkick, wav_hkick_1, s_sec, vel * 0.85, max_dur_sec=0.30)
        kick_times.append(s_sec)
    for s_sec, _, _, vel, _ in inst_notes["hard kick cover"]:
        place_oneshot(w_hardkick, wav_hkick_2, s_sec, vel * 0.55, max_dur_sec=0.28)
    for s_sec, _, _, vel, _ in inst_notes["snare"]:
        place_oneshot(w_snare, wav_snare, s_sec, vel**1.15, max_dur_sec=0.24)

    hat_events = sorted(
        [(s, v, False) for s, _, _, v, _ in inst_notes["hihat"]]
        + [(s, v, True) for s, _, _, v, _ in inst_notes["hihatrenda"]],
        key=lambda x: x[0],
    )
    for idx, (s_sec, vel, is_renda) in enumerate(hat_events):
        next_s = hat_events[idx + 1][0] if idx + 1 < len(hat_events) else s_sec + 0.5
        choke_dur = max(0.035, min(0.25, next_s - s_sec + 0.008))
        src_hat = wav_hat_o if (is_renda and vel >= 0.85 and choke_dur > 0.12) else wav_hat_c
        place_oneshot(w_hat, src_hat, s_sec, vel**1.1, max_dur_sec=choke_dur)

    amen_events = sorted(inst_notes["amen"], key=lambda x: x[0])
    for idx, (s_sec, e_sec, _, vel, tag) in enumerate(amen_events):
        next_s = amen_events[idx + 1][0] if idx + 1 < len(amen_events) else s_sec + 0.25
        choke_dur = max(0.04, min(e_sec - s_sec + 0.02, next_s - s_sec + 0.005))
        sl_idx = AMEN_SLICE_IDX.get(tag, 2)
        place_oneshot(w_amen, break_slices[sl_idx], s_sec, vel * 0.95, max_dur_sec=choke_dur)

    wav_revcym_long = wav_crash[: round(bar_sec * RATE)][::-1].copy()
    wav_revcym_short = wav_crash[: round(0.5 * bar_sec * RATE)][::-1].copy()

    riser_tail = wav_riser[-round(bar_sec * RATE) :].copy()
    n_flt = len(riser_tail)
    t_flt = np.arange(n_flt, dtype=np.float32) / RATE
    sweep_syn = np.zeros((n_flt, 2), dtype=np.float32)
    for midi_p, pan_v in ((54, -0.35), (57, 0.35), (61, -0.20), (66, 0.20)):
        f0 = 440.0 * (2.0 ** ((midi_p - 69) / 12.0))
        saw = 2.0 * ((t_flt * f0) % 1.0) - 1.0
        sweep_syn[:, 0] += saw * (0.18 * (1.0 - pan_v))
        sweep_syn[:, 1] += saw * (0.18 * (1.0 + pan_v))
    for bi in range(8):
        s0 = (bi * n_flt) // 8
        s1 = ((bi + 1) * n_flt) // 8
        fc = 260.0 * ((4800.0 / 260.0) ** (bi / 7.0))
        sos_sw = signal.butter(2, [max(90.0, fc * 0.45), min(8000.0, fc * 1.45)], "bandpass", fs=RATE, output="sos")
        sweep_syn[s0:s1] = signal.sosfilt(sos_sw, sweep_syn[s0:s1], axis=0)
    env_flt = np.linspace(0.25, 1.0, n_flt, dtype=np.float32)[:, None]
    wav_filter_turnaround = (riser_tail * 0.70 + sweep_syn * env_flt * 0.85).astype(np.float32)

    for s_sec, _, _, vel, _ in inst_notes["crash"]:
        place_oneshot(w_cym_fx, wav_crash, s_sec, vel * 0.85, max_dur_sec=3.2)
    for s_sec, _, _, vel, _ in inst_notes["crashshort"]:
        place_oneshot(w_cym_fx, wav_crash, s_sec, vel * 0.75, max_dur_sec=0.85)
    for s_sec, _, _, vel, _ in inst_notes["reversecym_long"]:
        place_oneshot(w_cym_fx, wav_revcym_long, s_sec, vel * 0.90, max_dur_sec=bar_sec)
    for s_sec, _, _, vel, _ in inst_notes["reversecym_short"]:
        place_oneshot(w_cym_fx, wav_revcym_short, s_sec, vel * 0.80, max_dur_sec=0.5 * bar_sec)
    for s_sec, _, _, vel, _ in inst_notes["impact"]:
        place_oneshot(w_cym_fx, wav_impact, s_sec, vel * 0.90, max_dur_sec=3.5)
    for s_sec, _, _, vel, _ in inst_notes["downfx"]:
        place_oneshot(w_cym_fx, wav_down, s_sec, vel * 0.80, max_dur_sec=3.5)
    for s_sec, _ in filter_events:
        place_oneshot(w_cym_fx, wav_filter_turnaround, s_sec, 0.95, max_dur_sec=bar_sec)

    # 6. External Bus Processing & Separate Rhythm/Synth Bus Clarity Sculpting
    sos_cym_tame = signal.butter(2, 5200.0, "lowpass", fs=RATE, output="sos")
    w_hat = (0.52 * w_hat + 0.48 * signal.sosfilt(sos_cym_tame, w_hat, axis=0)).astype(np.float32)
    w_cym_fx = (0.58 * w_cym_fx + 0.42 * signal.sosfilt(sos_cym_tame, w_cym_fx, axis=0)).astype(np.float32)

    w_mainlead = lush_stereo_reverb(
        tempo_pingpong_delay(multiband_ott(w_mainlead, 0.50, hp_hz=175.0), bpm_val, 0.75, 0.24, 0.12), rt60=0.85, wet_level=0.10
    )
    w_sublead = lush_stereo_reverb(
        tempo_pingpong_delay(multiband_ott(w_sublead, 0.46, hp_hz=210.0), bpm_val, 0.50, 0.24, 0.13), rt60=0.90, wet_level=0.11
    )
    w_coverlead = lush_stereo_reverb(multiband_ott(w_coverlead, 0.42, hp_hz=190.0), rt60=0.95, wet_level=0.12)
    w_droplead = lush_stereo_reverb(
        tempo_pingpong_delay(multiband_ott(w_droplead, 0.52, hp_hz=220.0), bpm_val, 0.75, 0.24, 0.13), rt60=0.85, wet_level=0.11
    )
    w_backlead = lush_stereo_reverb(
        tempo_pingpong_delay(multiband_ott(w_backlead, 0.42, hp_hz=210.0), bpm_val, 0.50, 0.24, 0.14), rt60=0.95, wet_level=0.12
    )
    w_chord_stab = lush_stereo_reverb(multiband_ott(w_chord_stab, 0.42, hp_hz=210.0), rt60=0.90, wet_level=0.12)
    sos_pad_hp = signal.butter(2, 175.0, "highpass", fs=RATE, output="sos")
    w_pad_chord = lush_stereo_reverb(signal.sosfilt(sos_pad_hp, w_pad_chord, axis=0).astype(np.float32), rt60=1.20, wet_level=0.18)
    w_snare = lush_stereo_reverb(w_snare, rt60=0.65, wet_level=0.10, predelay_ms=14.0)

    sc_env = np.ones(total_frames, dtype=np.float32)
    sc_len = round(0.095 * RATE)
    sc_curve = np.concatenate([
        np.linspace(1.0, 0.22, round(0.004 * RATE), dtype=np.float32),
        0.22 + 0.78 * (np.linspace(0.0, 1.0, sc_len - round(0.004 * RATE), dtype=np.float32) ** 1.6),
    ])
    for kt in kick_times:
        a = round(kt * RATE)
        b = min(total_frames, a + len(sc_curve))
        if b > a:
            sc_env[a:b] = np.minimum(sc_env[a:b], sc_curve[: b - a])

    w_bass_sub *= sc_env[:, None]
    w_bass_mid *= (0.15 + 0.85 * sc_env)[:, None]
    w_bass_hi *= (0.25 + 0.75 * sc_env)[:, None]
    w_pad_chord *= (0.25 + 0.75 * sc_env)[:, None]
    w_chord_stab *= (0.45 + 0.55 * sc_env)[:, None]
    w_amen *= (0.35 + 0.65 * sc_env)[:, None]

    v3_stems = {
        "kick": w_kick, "hardkick": w_hardkick, "snare": w_snare, "hat": w_hat,
        "amen_break": w_amen, "cymbal_fx": w_cym_fx,
        "bass_sub": w_bass_sub, "bass_mid": w_bass_mid, "bass_hi": w_bass_hi,
        "lead_main": w_mainlead, "lead_sub": w_sublead, "lead_cover": w_coverlead,
        "lead_drop": w_droplead, "lead_back": w_backlead,
        "chord_stab": w_chord_stab, "pad_chord": w_pad_chord,
        "arp_stereo": w_arp_stereo, "arp_chip": w_arp_chip,
    }

    # 7. Render & Spatialize Hybrid Acoustic / Orchestral / Traditional Stems (SF2 + Cross-Song Multisamples)
    sos_aco_hp = signal.butter(2, 105.0, "highpass", fs=RATE, output="sos")
    for fam in ACOUSTIC_FAMILIES:
        evs_fam = [(s, e, p, v) for s, e, p, v, _ in inst_notes.get(fam, ())]
        if not evs_fam:
            continue
        w_fam = render_hybrid_acoustic_stem(fam, evs_fam, total_frames, exclude_song=song_name)
        pan_v = ACOUSTIC_STAGE_PAN.get(fam, 0.0)
        if abs(pan_v) > 1e-4:
            w_fam = pan_stereo(w_fam, pan_v)
        if fam not in ("taiko", "timpani", "orch_snare", "piano", "guitar_drive"):
            w_fam = signal.sosfilt(sos_aco_hp, w_fam, axis=0).astype(np.float32)
        rev_rt = 1.25 if fam in ("strings_legato", "harp", "violin", "woodwinds", "brass", "bell") else 0.90
        rev_wet = 0.17 if fam in ("strings_legato", "harp", "violin", "musicbox", "bell", "vibraphone") else 0.12
        w_fam = lush_stereo_reverb(w_fam, rt60=rev_rt, wet_level=rev_wet, predelay_ms=20.0)
        if fam not in ("taiko", "timpani", "orch_snare"):
            w_fam *= (0.68 + 0.32 * sc_env)[:, None]
        v3_stems[fam] = w_fam

    win_200ms = int(0.2 * RATE)
    n_win = total_frames // win_200ms
    v3_rhythm_bus = np.zeros((total_frames, 2), dtype=np.float32)
    v3_synth_bus = np.zeros((total_frames, 2), dtype=np.float32)
    v3_acoustic_bus = np.zeros((total_frames, 2), dtype=np.float32)
    has_acoustic = False

    for grp, stem in v3_stems.items():
        blocks = np.sqrt(np.mean(stem[: n_win * win_200ms].reshape(n_win, win_200ms, 2) ** 2, axis=(1, 2)))
        active = blocks[blocks > 1e-4]
        if len(active) == 0:
            continue
        cur_p90_db = 20.0 * math.log10(float(np.percentile(active, 90)))
        target_db = NATIVE_P90_TARGETS_DB[grp]
        g = 10.0 ** ((target_db - cur_p90_db) / 20.0)
        if grp in RHYTHM_GROUPS:
            v3_rhythm_bus += stem * g
        elif grp in ACOUSTIC_FAMILIES:
            v3_acoustic_bus += stem * g
            has_acoustic = True
        else:
            v3_synth_bus += stem * g

    synth_eq_db = np.zeros_like(FREQUENCIES, dtype=np.float32)
    for idx, f_hz in enumerate(FREQUENCIES):
        if 180.0 <= f_hz <= 480.0:
            synth_eq_db[idx] = -2.2 * math.sin(math.pi * (f_hz - 180.0) / (480.0 - 180.0))
        elif 1400.0 <= f_hz <= 7200.0:
            synth_eq_db[idx] = +2.4 * math.sin(math.pi * math.log2(f_hz / 1400.0) / math.log2(7200.0 / 1400.0))

    taps_synth = signal.firwin2(
        1025,
        np.r_[0.0, FREQUENCIES, RATE / 2.0],
        10.0 ** (np.r_[synth_eq_db[0], synth_eq_db, synth_eq_db[-1]] / 20.0),
        fs=RATE,
    ).astype(np.float32)
    if np.max(np.abs(v3_synth_bus)) > 1e-6:
        v3_synth_bus = signal.oaconvolve(v3_synth_bus, taps_synth[:, None], mode="same", axes=0).astype(np.float32)

    if has_acoustic:
        aco_eq_db = np.zeros_like(FREQUENCIES, dtype=np.float32)
        for idx, f_hz in enumerate(FREQUENCIES):
            if 200.0 <= f_hz <= 450.0:
                aco_eq_db[idx] = -1.4 * math.sin(math.pi * (f_hz - 200.0) / (450.0 - 200.0))
            elif 1800.0 <= f_hz <= 8500.0:
                aco_eq_db[idx] = +1.8 * math.sin(math.pi * math.log2(f_hz / 1800.0) / math.log2(8500.0 / 1800.0))
        taps_aco = signal.firwin2(
            1025,
            np.r_[0.0, FREQUENCIES, RATE / 2.0],
            10.0 ** (np.r_[aco_eq_db[0], aco_eq_db, aco_eq_db[-1]] / 20.0),
            fs=RATE,
        ).astype(np.float32)
        v3_acoustic_bus = signal.oaconvolve(v3_acoustic_bus, taps_aco[:, None], mode="same", axes=0).astype(np.float32)

    v3_mix = v3_rhythm_bus + v3_synth_bus + v3_acoustic_bus

    if reference_wave is not None:
        ref_spec = spectrum(reference_wave, RATE, smoothing_oct=0.65)
        src_spec = spectrum(v3_mix, RATE, smoothing_oct=0.65)
        raw_diff = ref_spec - src_spec
        diff_db = np.zeros_like(raw_diff)
        for idx, f_hz in enumerate(FREQUENCIES):
            if f_hz <= 1400.0:
                diff_db[idx] = float(np.clip(raw_diff[idx] * 0.65, -2.5, 3.5))
            elif f_hz <= 6200.0:
                diff_db[idx] = float(np.clip(raw_diff[idx] * 0.40, 0.0, 2.5))
            else:
                diff_db[idx] = float(np.clip(raw_diff[idx] * 0.45, -2.2, 1.5))
        taps = signal.firwin2(
            1025,
            np.r_[0.0, FREQUENCIES, RATE / 2.0],
            10.0 ** (np.r_[diff_db[0], diff_db, diff_db[-1]] / 20.0),
            fs=RATE,
        ).astype(np.float32)
        v3_mix = signal.oaconvolve(v3_mix, taps[:, None], mode="same", axes=0).astype(np.float32)

    return lookahead_master_to_lufs(v3_mix, target_lufs)
