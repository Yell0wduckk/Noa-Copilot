"""Pure-Python SoundFont 2 (.sf2) polyphonic stereo sampler + Cross-Song BMS Acoustic Multisample Engine."""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import math
from pathlib import Path
import re
import struct

import numpy as np
from scipy import signal
import soxr

from .studio_mix import RATE
from .studio_sources import ASSETS, Archive

MML_PITCH_RE = re.compile(r"^b_(.+?)_v(\d+)l([\d-]+)o([0-9])([a-g])([ps#m-]?)$", re.I)
NOTE_SEMI = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}


def _mml_to_midi(octv_s: str, note_c: str, acc: str) -> int:
    semi = NOTE_SEMI[note_c.lower()]
    if acc.lower() in ("p", "s", "#"):
        semi += 1
    elif acc.lower() in ("m", "-"):
        semi -= 1
    return int(np.clip((int(octv_s) + 1) * 12 + semi, 21, 108))


class SF2SoundFont:
    """Pure-Python SoundFont 2.01 (.sf2) polyphonic stereo sample synthesizer."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        data = self.path.read_bytes()
        chunks = {}
        pos = 12
        while pos < len(data):
            cid = data[pos : pos + 4]
            csz = struct.unpack_from("<I", data, pos + 4)[0]
            if cid == b"LIST":
                sub = pos + 12
                end = pos + 8 + csz
                while sub < end:
                    scid = data[sub : sub + 4]
                    scsz = struct.unpack_from("<I", data, sub + 4)[0]
                    chunks[scid] = (sub + 8, scsz)
                    sub += 8 + scsz + (scsz & 1)
            pos += 8 + csz + (csz & 1)

        smpl_off, smpl_sz = chunks[b"smpl"]
        self.pcm = np.frombuffer(data, dtype="<i2", count=smpl_sz // 2, offset=smpl_off)

        shdr_off, shdr_sz = chunks[b"shdr"]
        self.samples = []
        for off in range(shdr_off, shdr_off + shdr_sz - 46, 46):
            name, start, end, sloop, eloop, sr, orig_pitch, pitch_corr, link, stype = struct.unpack_from(
                "<20sIIIIIBbHH", data, off
            )
            self.samples.append(
                {
                    "name": name.split(b"\0", 1)[0].decode("ascii", errors="ignore"),
                    "start": start,
                    "end": end,
                    "start_loop": sloop,
                    "end_loop": eloop,
                    "sample_rate": sr,
                    "orig_pitch": orig_pitch if orig_pitch <= 127 else 60,
                    "pitch_corr": pitch_corr,
                    "sample_type": stype,
                }
            )

        inst_off, inst_sz = chunks[b"inst"]
        insts = [struct.unpack_from("<20sH", data, off) for off in range(inst_off, inst_off + inst_sz, 22)]
        ibag_off, ibag_sz = chunks[b"ibag"]
        ibags = [struct.unpack_from("<HH", data, off) for off in range(ibag_off, ibag_off + ibag_sz, 4)]
        igen_off, igen_sz = chunks[b"igen"]
        igens = [struct.unpack_from("<Hh", data, off) for off in range(igen_off, igen_off + igen_sz, 4)]

        self.inst_zones = []
        for i in range(len(insts) - 1):
            b0, b1 = insts[i][1], insts[i + 1][1]
            global_gens = {}
            zones = []
            for b_idx in range(b0, b1):
                g0, g1 = ibags[b_idx][0], ibags[b_idx + 1][0]
                gens = dict(global_gens)
                has_sample = False
                for g_idx in range(g0, g1):
                    oper, amt = igens[g_idx]
                    gens[oper] = amt
                    if oper == 53:
                        has_sample = True
                if not has_sample and b_idx == b0:
                    global_gens = gens
                elif has_sample:
                    zones.append(gens)
            self.inst_zones.append(zones)

        phdr_off, phdr_sz = chunks[b"phdr"]
        phdrs = [struct.unpack_from("<20sHHHIII", data, off) for off in range(phdr_off, phdr_off + phdr_sz, 38)]
        pbag_off, pbag_sz = chunks[b"pbag"]
        pbags = [struct.unpack_from("<HH", data, off) for off in range(pbag_off, pbag_off + pbag_sz, 4)]
        pgen_off, pgen_sz = chunks[b"pgen"]
        pgens = [struct.unpack_from("<Hh", data, off) for off in range(pgen_off, pgen_off + pgen_sz, 4)]

        self.presets = {}
        for i in range(len(phdrs) - 1):
            name, prog, bank, b0, _, _, _ = phdrs[i]
            b1 = phdrs[i + 1][3]
            global_pgens = {}
            pzones = []
            for b_idx in range(b0, b1):
                g0, g1 = pbags[b_idx][0], pbags[b_idx + 1][0]
                gens = dict(global_pgens)
                has_inst = False
                for g_idx in range(g0, g1):
                    oper, amt = pgens[g_idx]
                    gens[oper] = amt
                    if oper == 41:
                        has_inst = True
                if not has_inst and b_idx == b0:
                    global_pgens = gens
                elif has_inst:
                    pzones.append(gens)
            self.presets[(bank, prog)] = pzones
        self._note_cache = {}

    @staticmethod
    def _range(val):
        if val is None:
            return 0, 127
        u = val & 0xFFFF
        return u & 0xFF, (u >> 8) & 0xFF

    @staticmethod
    def _timecents_to_sec(tc, default_sec):
        if tc is None:
            return default_sec
        if tc <= -11900:
            return 0.001
        return float(np.clip(2.0 ** (tc / 1200.0), 0.001, 8.0))

    def _matching_zones(self, bank: int, program: int, key: int, vel_127: int):
        pzones = self.presets.get((bank, program)) or self.presets.get((0, program)) or []
        matched = []
        for pg in pzones:
            pk_lo, pk_hi = self._range(pg.get(43))
            pv_lo, pv_hi = self._range(pg.get(44))
            if not (pk_lo <= key <= pk_hi and pv_lo <= vel_127 <= pv_hi):
                continue
            inst_id = pg[41] & 0xFFFF
            if inst_id >= len(self.inst_zones):
                continue
            for ig in self.inst_zones[inst_id]:
                ik_lo, ik_hi = self._range(ig.get(43))
                iv_lo, iv_hi = self._range(ig.get(44))
                if not (ik_lo <= key <= ik_hi and iv_lo <= vel_127 <= iv_hi):
                    continue
                sid = ig[53] & 0xFFFF
                if sid >= len(self.samples):
                    continue
                shdr = self.samples[sid]
                if shdr["sample_type"] & 0x8000:
                    continue
                matched.append((pg, ig, shdr))
        return matched

    def render_note_stereo(
        self,
        bank: int,
        program: int,
        key: int,
        vel_127: int,
        gate_sec: float,
        max_release_sec: float = 0.35,
    ) -> np.ndarray:
        key = int(np.clip(key, 12, 108))
        q_gate = round(max(0.03, min(6.0, gate_sec)) * 40.0) / 40.0
        q_vel = max(24, min(127, (int(vel_127) // 16) * 16))
        cache_key = (bank, program, key, q_vel, q_gate, round(max_release_sec, 2))
        if cache_key in self._note_cache:
            return self._note_cache[cache_key]

        zones = self._matching_zones(bank, program, key, q_vel)
        if not zones and bank == 0:
            # Try nearest velocity if exact velocity layer missed
            zones = self._matching_zones(bank, program, key, 100) or self._matching_zones(bank, program, key, 64)
        if not zones:
            return np.zeros((max(1, round(q_gate * RATE)), 2), dtype=np.float32)

        rendered_zones = []
        max_len = 1
        for pg, ig, shdr in zones[:4]:
            start = shdr["start"] + ig.get(0, 0) + ig.get(4, 0) * 32768
            end = shdr["end"] + ig.get(1, 0) + ig.get(12, 0) * 32768
            sloop = shdr["start_loop"] + ig.get(2, 0) + ig.get(45, 0) * 32768
            eloop = shdr["end_loop"] + ig.get(3, 0) + ig.get(50, 0) * 32768
            start = max(0, min(len(self.pcm) - 1, start))
            end = max(start + 1, min(len(self.pcm), end))
            if end - start < 8:
                continue

            root_key = ig.get(58, -1)
            if root_key is None or root_key < 0 or root_key > 127:
                root_key = shdr["orig_pitch"]
            scale_tune = ig.get(56, 100) + pg.get(56, 0)
            coarse = ig.get(51, 0) + pg.get(51, 0)
            fine = ig.get(52, 0) + pg.get(52, 0) + shdr["pitch_corr"]
            semitones = ((key - root_key) * (scale_tune / 100.0)) + coarse + (fine / 100.0)
            ratio = (shdr["sample_rate"] / float(RATE)) * (2.0 ** (semitones / 12.0))

            sample_mode = ig.get(54, 0) & 3
            atk_sec = min(0.08, self._timecents_to_sec(ig.get(34, pg.get(34)), 0.002))
            dec_sec = self._timecents_to_sec(ig.get(36, pg.get(36)), 1.2)
            sus_cb = max(0, ig.get(37, 0) + pg.get(37, 0))
            sus_level = float(10.0 ** (-min(sus_cb, 1000) / 200.0))
            rel_sec = min(max_release_sec, self._timecents_to_sec(ig.get(38, pg.get(38)), 0.16))

            total_sec = max(0.03, q_gate + rel_sec)
            n_out = max(8, round(total_sec * RATE))
            src_pos = np.arange(n_out, dtype=np.float64) * ratio

            raw_len = end - start
            loop_len = eloop - sloop
            if sample_mode in (1, 3) and sloop >= start and eloop <= end and loop_len >= 16:
                rel_start = sloop - start
                beyond = src_pos >= rel_start
                if np.any(beyond):
                    src_pos[beyond] = rel_start + ((src_pos[beyond] - rel_start) % loop_len)
            else:
                valid_mask = src_pos < (raw_len - 1)
                if not np.any(valid_mask):
                    continue
                n_out = int(np.argmax(~valid_mask)) if not np.all(valid_mask) else n_out
                src_pos = src_pos[:n_out]

            idx0 = np.clip(start + np.floor(src_pos).astype(np.int64), start, end - 2)
            frac = (src_pos - np.floor(src_pos)).astype(np.float32)
            s0 = self.pcm[idx0].astype(np.float32) * (1.0 / 32768.0)
            s1 = self.pcm[idx0 + 1].astype(np.float32) * (1.0 / 32768.0)
            mono = s0 + frac * (s1 - s0)

            t_arr = np.arange(n_out, dtype=np.float32) / RATE
            env = np.ones(n_out, dtype=np.float32)
            if atk_sec > 0.002:
                atk_n = min(n_out, round(atk_sec * RATE))
                if atk_n > 1:
                    env[:atk_n] = np.linspace(0.0, 1.0, atk_n, dtype=np.float32)
            gate_n = min(n_out, max(1, round(q_gate * RATE)))
            if dec_sec > 0.005 and sus_level < 0.99:
                atk_n = min(gate_n, round(atk_sec * RATE))
                if gate_n > atk_n:
                    dec_t = t_arr[atk_n:gate_n] - t_arr[atk_n]
                    env[atk_n:gate_n] = sus_level + (1.0 - sus_level) * np.exp(-3.0 * dec_t / max(dec_sec, 0.01))
            if n_out > gate_n:
                rel_t = t_arr[gate_n:] - t_arr[gate_n]
                start_rel_val = float(env[gate_n - 1])
                env[gate_n:] = start_rel_val * np.exp(-4.5 * rel_t / max(rel_sec, 0.01))
            fade_end = min(round(0.004 * RATE), n_out // 4)
            if fade_end > 0:
                env[-fade_end:] *= np.linspace(1.0, 0.0, fade_end, dtype=np.float32)

            atten_cb = max(0, ig.get(48, 0) + pg.get(48, 0))
            atten_gain = float(10.0 ** (-atten_cb * 0.04 / 20.0))
            mono *= env * atten_gain

            pan_raw = ig.get(17, pg.get(17, 0))
            stype = shdr["sample_type"]
            if stype == 4 and pan_raw == 0:
                pan_val = -0.65
            elif stype == 2 and pan_raw == 0:
                pan_val = 0.65
            else:
                pan_val = float(np.clip(pan_raw / 500.0, -1.0, 1.0))
            angle = (pan_val + 1.0) * (math.pi / 4.0)
            gl, gr = math.cos(angle), math.sin(angle)
            stereo = np.column_stack((mono * gl, mono * gr)).astype(np.float32)
            rendered_zones.append(stereo)
            max_len = max(max_len, len(stereo))

        if not rendered_zones:
            return np.zeros((max(1, round(q_gate * RATE)), 2), dtype=np.float32)
        out = np.zeros((max_len, 2), dtype=np.float32)
        for z in rendered_zones:
            out[: len(z)] += z
        peak = float(np.max(np.abs(out)))
        if peak > 1e-5:
            out *= min(2.2, 0.38 / peak)
        if len(self._note_cache) < 6000:
            self._note_cache[cache_key] = out
        return out


@lru_cache(maxsize=4)
def load_soundfonts() -> list[SF2SoundFont]:
    """Load FluidR3_GM.sf2, GeneralUser-GS.sf2, and TimGM6mb.sf2 if present."""
    sf_dirs = [
        ASSETS / "soundfonts",
        Path(r"D:\01YD\01code\codex\maimai-ai-charting\artifacts\soundfonts"),
        ASSETS / "music-gen-repos/layered_music_gen/sf/harmony",
        Path(r"D:\01YD\01code\codex\maimai-ai-charting\artifacts\music-gen-repos\layered_music_gen\sf\harmony"),
    ]
    loaded = []
    seen = set()
    for name in ("FluidR3_GM.sf2", "GeneralUser-GS.sf2", "TimGM6mb.sf2"):
        for d in sf_dirs:
            p = d / name
            if p.exists() and name not in seen:
                try:
                    loaded.append(SF2SoundFont(p))
                    seen.add(name)
                    break
                except Exception:
                    pass
    return loaded


# Cross-song studio multisample sources in bms-samples-l4-20261106/final-download
# Strictly filtered by `exclude_song` at render time so 0 target-song samples are ever used.
CROSS_SONG_MULTISAMPLE_SPECS = {
    "piano": [
        ("gdbg_bms_package_2022", "Roject_HAB", "piano_wet_1x"),
        ("gdbg_bms_package_2022", "Tsukasound_Song_ogg", "h_piano"),
        ("gdbg_bms_package_2023", "Hyouri_Crystallized[ogg]", "piano"),
    ],
    "epiano": [
        ("gdbg_bms_package_2022", "iemonogatari_o", "introep"),
        ("gdbg_bms_package_2022", "[utzbo]GaulungNeonCity_ogg", "elpiano"),
    ],
    "organ": [
        ("gdbg_bms_package_2022", "qfeileadh_Hybris", "organ"),
    ],
    "guitar_clean": [
        ("gdbg_bms_package_2022", "qfeileadh_Hybris", "guitarclean"),
        ("gdbg_bms_package_2022", "pillarsofash_ogg", "guitar"),
    ],
    "guitar_drive": [
        ("gdbg_bms_package_2022", "pillarsofash_ogg", "guitar"),
        ("gdbg_bms_package_2022", "qfeileadh_Hybris", "guitarclean"),
    ],
    "harp": [
        ("gdbg_bms_package_2022", "Tsukasound_Song_ogg", "harp"),
        ("gdbg_bms_package_2022", "[otoshi.b_qfeileadh]futarino_signal_ogg", "orche_harp"),
    ],
    "violin": [
        ("gdbg_bms_package_2022", "Roject_HAB", "synth_violinx"),
    ],
    "strings_stacc": [
        ("gdbg_bms_package_2022", "[otoshi.b_qfeileadh]futarino_signal_ogg", "pizzicato_l"),
        ("gdbg_bms_package_2022", "iemonogatari_o", "sabistrings"),
    ],
    "strings_legato": [
        ("gdbg_bms_package_2022", "Roject_HAB", "synth_violinx"),
        ("gdbg_bms_package_2022", "iemonogatari_o", "sabistrings"),
    ],
    "brass": [
        ("gdbg_bms_package_2022", "iemonogatari_o", "sabibrass03"),
        ("gdbg_bms_package_2022", "iemonogatari_o", "sabibrass01"),
        ("gdbg_bms_package_2022", "[otoshi.b_qfeileadh]futarino_signal_ogg", "orche_horn"),
    ],
    "woodwinds": [
        ("gdbg_bms_package_2022", "pillarsofash_ogg", "woodwinds"),
    ],
    "xylophone": [
        ("gdbg_bms_package_2022", "Tsukasound_Song_ogg", "xylophone"),
        ("gdbg_bms_package_2023", "TsukaRekusound_LIFE_ogg", "xylophone"),
    ],
    "vibraphone": [
        ("gdbg_bms_package_2023", "TsukaRekusound_LIFE_ogg", "vibraphone"),
        ("gdbg_bms_package_2022", "[otoshi.b_qfeileadh]futarino_signal_ogg", "glock_solo"),
    ],
    "musicbox": [
        ("gdbg_bms_package_2022", "[otoshi.b_qfeileadh]futarino_signal_ogg", "glock_solo"),
        ("gdbg_bms_package_2023", "Hyouri_Crystallized[ogg]", "glocken"),
    ],
    "bell": [
        ("gdbg_bms_package_2022", "[otoshi.b_qfeileadh]futarino_signal_ogg", "bell_solo"),
        ("gdbg_bms_package_2022", "Tsukasound_Song_ogg", "h tubular bells"),
    ],
    "koto": [
        ("gdbg_bms_package_2023", "TsukaRekusound_LIFE_ogg", "koto"),
        ("gdbg_bms_package_2022", "[utzbo]GaulungNeonCity_ogg", "chinapluck"),
    ],
    "shamisen": [
        ("gdbg_bms_package_2023", "TsukaRekusound_LIFE_ogg", "shamisen"),
        ("gdbg_bms_package_2022", "[utzbo]GaulungNeonCity_ogg", "chinapluck"),
    ],
}

# SF2 (bank, program) layer recipes for each acoustic/orchestral/traditional family
SF2_FAMILY_PATCHES = {
    "piano": [((0, 0), 0.72, 0.38), ((0, 1), 0.38, 0.32)],
    "epiano": [((0, 4), 0.70, 0.35), ((0, 5), 0.42, 0.30)],
    "organ": [((0, 16), 0.65, 0.25), ((0, 19), 0.48, 0.30)],
    "guitar_clean": [((0, 24), 0.58, 0.32), ((0, 25), 0.48, 0.32), ((0, 27), 0.35, 0.28)],
    "guitar_drive": [((0, 29), 0.68, 0.25), ((0, 30), 0.42, 0.25), ((0, 25), 0.28, 0.25)],
    "harp": [((0, 46), 0.85, 0.48), ((0, 24), 0.25, 0.35)],
    "violin": [((0, 40), 0.75, 0.30), ((0, 48), 0.40, 0.35)],
    "strings_stacc": [((0, 45), 0.58, 0.18), ((0, 48), 0.52, 0.18), ((0, 44), 0.32, 0.16)],
    "strings_legato": [((0, 48), 0.72, 0.45), ((0, 49), 0.45, 0.45), ((0, 40), 0.30, 0.35)],
    "brass": [((0, 61), 0.75, 0.32), ((0, 60), 0.45, 0.35), ((0, 56), 0.32, 0.28)],
    "woodwinds": [((0, 73), 0.72, 0.28), ((0, 68), 0.38, 0.28), ((0, 71), 0.35, 0.28)],
    "xylophone": [((0, 13), 0.78, 0.25), ((0, 12), 0.45, 0.28)],
    "vibraphone": [((0, 11), 0.82, 0.42), ((0, 12), 0.32, 0.32)],
    "musicbox": [((0, 10), 0.78, 0.42), ((0, 8), 0.42, 0.38), ((0, 9), 0.28, 0.35)],
    "bell": [((0, 14), 0.75, 0.50), ((0, 9), 0.45, 0.40)],
    "koto": [((0, 107), 0.82, 0.38), ((0, 46), 0.32, 0.35)],
    "shamisen": [((0, 106), 0.85, 0.28), ((0, 105), 0.35, 0.25)],
    "taiko": [((0, 116), 0.90, 0.45), ((0, 47), 0.40, 0.40)],
    "timpani": [((0, 47), 0.88, 0.50), ((0, 116), 0.35, 0.40)],
    "orch_snare": [((128, 48), 0.85, 0.30), ((128, 0), 0.45, 0.25)],
}


@lru_cache(maxsize=32)
def _load_cross_song_multisample_pool(col: str, song_sub: str, inst_name: str) -> dict[int, np.ndarray]:
    arch = Archive(ASSETS / "bms-samples-l4-20261106/final-download")
    rows = arch.db.execute(
        "SELECT path, basename FROM files WHERE collection=? AND song LIKE ?",
        (col, f"%{song_sub}%"),
    ).fetchall()
    by_pitch = {}
    for path, base in rows:
        stem = Path(base).stem.lower()
        m = MML_PITCH_RE.match(stem)
        if not m:
            continue
        raw_inst, vel_s, l_str, octv, note_c, acc = m.groups()
        if raw_inst.lower() != inst_name.lower():
            continue
        midi_p = _mml_to_midi(octv, note_c, acc)
        try:
            w = arch.audio(col, path)
        except Exception:
            continue
        if len(w) < 256:
            continue
        # Keep the longest/fullest velocity sample per pitch
        if midi_p not in by_pitch or len(w) > len(by_pitch[midi_p]):
            peak = float(np.max(np.abs(w)))
            if peak > 1e-4:
                w = (w * min(2.5, 0.36 / peak)).astype(np.float32)
                by_pitch[midi_p] = w
    return by_pitch


def _render_cross_song_multisample_note(
    family: str, pitch: int, gate_sec: float, exclude_song: str | None = None
) -> np.ndarray | None:
    specs = CROSS_SONG_MULTISAMPLE_SPECS.get(family, ())
    ex_low = (exclude_song or "").lower()
    pool = None
    for col, song_sub, inst_name in specs:
        if ex_low and (ex_low in song_sub.lower() or song_sub.lower() in ex_low):
            continue
        cand = _load_cross_song_multisample_pool(col, song_sub, inst_name)
        if len(cand) >= 5:
            pool = cand
            break
    if not pool:
        return None
    nearest_p = min(pool.keys(), key=lambda p: abs(p - pitch))
    semi_diff = pitch - nearest_p
    if abs(semi_diff) > 14:
        return None
    src = pool[nearest_p]
    if semi_diff != 0:
        target_sr = round(RATE * (2.0 ** (-semi_diff / 12.0)))
        src = soxr.resample(src, RATE, max(4000, min(192000, target_sr)), quality="HQ").astype(np.float32)
    rel_sec = 0.25 if family not in ("strings_stacc", "xylophone", "shamisen") else 0.10
    max_n = min(len(src), max(16, round((gate_sec + rel_sec) * RATE)))
    out = src[:max_n].copy()
    fade = min(round(rel_sec * RATE), len(out) // 3)
    if fade > 1:
        out[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)[:, None]
    return out


def render_hybrid_acoustic_stem(
    family: str,
    events: list[tuple[float, float, int, float]],
    total_frames: int,
    *,
    exclude_song: str | None = None,
) -> np.ndarray:
    """Render an acoustic/orchestral/traditional stem by layering dual SF2 SoundFonts + Cross-Song BMS Multisamples."""
    out = np.zeros((total_frames, 2), dtype=np.float32)
    if not events:
        return out
    sfs = load_soundfonts()
    recipe = SF2_FAMILY_PATCHES.get(family, [((0, 0), 0.8, 0.3)])
    note_unit_cache = {}

    for s_sec, e_sec, pitch, vel in events:
        at = round(s_sec * RATE)
        if at < 0 or at >= total_frames:
            continue
        gate_sec = max(0.04, min(5.0, e_sec - s_sec))
        if family == "strings_stacc":
            gate_sec = min(gate_sec, 0.24)
        q_gate = round(gate_sec * 32.0) / 32.0
        vel_127 = max(24, min(127, round(float(vel) * 127.0)))
        q_vel = (vel_127 // 16) * 16
        pitch_clamped = int(np.clip(pitch, 24, 103))
        if family in ("taiko", "timpani"):
            pitch_clamped = int(np.clip(pitch_clamped, 38, 58))
        elif family == "orch_snare":
            pitch_clamped = 38

        ukey = (family, pitch_clamped, q_vel, q_gate, exclude_song or "")
        if ukey not in note_unit_cache:
            layers = []
            max_l = 1
            for sf_idx, sf2 in enumerate(sfs[:2]):
                sf_weight = 0.68 if sf_idx == 0 else 0.44
                for (bank, prog), w_patch, rel_s in recipe:
                    w_sf = sf2.render_note_stereo(bank, prog, pitch_clamped, q_vel, q_gate, max_release_sec=rel_s)
                    if len(w_sf) > 1:
                        layers.append(w_sf * (w_patch * sf_weight))
                        max_l = max(max_l, len(w_sf))
            w_ms = _render_cross_song_multisample_note(family, pitch_clamped, q_gate, exclude_song=exclude_song)
            if w_ms is not None and len(w_ms) > 1:
                layers.append(w_ms * 0.55)
                max_l = max(max_l, len(w_ms))
            unit = np.zeros((max_l, 2), dtype=np.float32)
            for lyr in layers:
                unit[: len(lyr)] += lyr
            if family == "guitar_drive" and len(unit) > 8:
                # Warm overdrive saturation for rock/artcore guitar
                unit = (np.tanh(unit * 2.4) * 0.48).astype(np.float32)
            if len(note_unit_cache) < 4096:
                note_unit_cache[ukey] = unit
        else:
            unit = note_unit_cache[ukey]

        end_at = min(total_frames, at + len(unit))
        if end_at > at:
            out[at:end_at] += unit[: end_at - at] * float(vel)

    return out


# Register bounds for accurate polyphonic peak extraction on non-MML samples
FAMILY_PITCH_BOUNDS = {
    "piano": (40, 88),
    "epiano": (45, 86),
    "organ": (45, 86),
    "guitar_clean": (45, 84),
    "guitar_drive": (40, 82),
    "harp": (45, 88),
    "violin": (55, 92),
    "strings_stacc": (48, 86),
    "strings_legato": (43, 84),
    "brass": (45, 81),
    "woodwinds": (60, 92),
    "xylophone": (60, 92),
    "vibraphone": (55, 88),
    "musicbox": (64, 94),
    "bell": (60, 92),
    "koto": (52, 88),
    "shamisen": (50, 86),
    "bass": (28, 52),
}


def extract_polyphonic_notes_from_waveform(
    w_raw: np.ndarray,
    family: str,
    max_polyphony: int = 3,
) -> list[tuple[float, float, int, float]]:
    """Extract 1..max_polyphony chord notes (or multi-onset phrase notes) from a non-MML waveform."""
    if len(w_raw) < 256:
        return []
    mono = np.mean(w_raw, axis=1) if w_raw.ndim == 2 else w_raw
    dur = len(mono) / float(RATE)
    lo_midi, hi_midi = FAMILY_PITCH_BOUNDS.get(family, (48, 84))
    lo_hz = 440.0 * (2.0 ** ((lo_midi - 69) / 12.0))
    hi_hz = 440.0 * (2.0 ** ((hi_midi - 69) / 12.0))

    # Split into sub-windows if sample is a multi-note phrase (> 0.45s)
    if dur <= 0.55 or family in ("strings_legato", "brass", "organ"):
        segments = [(0.0, min(dur, 3.8))]
    else:
        step = 0.25 if dur <= 1.6 else 0.35
        n_seg = max(1, min(6, int(round(dur / step))))
        seg_dur = dur / n_seg
        segments = [(i * seg_dur, min(dur, (i + 1) * seg_dur)) for i in range(n_seg)]

    notes = []
    for s_off, e_off in segments:
        a = round(s_off * RATE)
        b = min(len(mono), round(e_off * RATE))
        part = mono[a:b]
        if len(part) < 256:
            continue
        rms = float(np.sqrt(np.mean(part * part)))
        if rms < 0.004:
            continue
        n_fft = max(4096, min(16384, 1 << int(math.ceil(math.log2(len(part))))))
        win = np.hanning(len(part))
        spec = np.abs(np.fft.rfft(part * win, n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, 1.0 / RATE)
        valid = (freqs >= lo_hz) & (freqs <= hi_hz)
        if not np.any(valid):
            continue
        f_sub = freqs[valid]
        s_sub = spec[valid].copy()
        peak_val = float(np.max(s_sub))
        if peak_val <= 1e-6:
            continue

        chosen_midis = []
        n_want = 1 if family in ("bass", "violin", "woodwinds", "shamisen") else max_polyphony
        for _ in range(n_want):
            idx = int(np.argmax(s_sub))
            val = float(s_sub[idx])
            if val < 0.28 * peak_val:
                break
            f0 = float(f_sub[idx])
            midi_p = int(np.clip(round(69.0 + 12.0 * math.log2(f0 / 440.0)), lo_midi, hi_midi))
            if all(abs(midi_p - m) >= 2 for m in chosen_midis):
                rel_vel = float(np.clip((val / peak_val) * min(1.0, rms * 5.5), 0.35, 0.98))
                chosen_midis.append(midi_p)
                notes.append((s_off, max(s_off + 0.08, e_off), midi_p, rel_vel))
            # Suppress fundamental and its overtones (1x, 2x, 3x, 4x) in s_sub
            for mult in (1.0, 2.0, 3.0, 4.0):
                harm_hz = f0 * mult
                mask_h = np.abs(f_sub - harm_hz) <= (harm_hz * 0.065)
                s_sub[mask_h] *= 0.05

    return notes


def transcribe_long_bgm_stem(
    w_raw: np.ndarray,
    trig_time: float,
    bar_sec: float,
) -> dict[str, list[tuple[float, float, int, float]]]:
    """Decompose a long BGM stem (> 4s) into bass, strings_legato, piano, and woodwinds events (0 original samples)."""
    result = defaultdict(list)
    mono = np.mean(w_raw, axis=1) if w_raw.ndim == 2 else w_raw
    dur = len(mono) / float(RATE)
    step_sec = max(0.18, min(0.35, bar_sec / 8.0))
    n_steps = int(dur / step_sec)
    for si in range(n_steps):
        s0 = si * step_sec
        s1 = min(dur, (si + 1) * step_sec)
        a = round(s0 * RATE)
        b = min(len(mono), round(s1 * RATE))
        part = mono[a:b]
        if len(part) < 512:
            continue
        rms = float(np.sqrt(np.mean(part * part)))
        if rms < 0.012:
            continue
        n_fft = 8192
        spec = np.abs(np.fft.rfft(part * np.hanning(len(part)), n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, 1.0 / RATE)

        # 1. Sub/Bass band (40..155 Hz)
        b_mask = (freqs >= 40.0) & (freqs <= 155.0)
        if np.any(b_mask):
            b_sub = spec[b_mask]
            b_idx = int(np.argmax(b_sub))
            if float(b_sub[b_idx]) > 0.15 * float(np.max(spec)):
                f_b = float(freqs[b_mask][b_idx])
                m_b = int(np.clip(round(69.0 + 12.0 * math.log2(f_b / 440.0)), 32, 52))
                result["bass"].append((trig_time + s0, trig_time + s1, m_b, float(np.clip(rms * 2.8, 0.30, 0.78))))

        # 2. Mid Harmony band (180..1100 Hz) -> strings_legato & piano (every 2 steps for smooth chords)
        if si % 2 == 0:
            h_end = min(dur, s0 + 2.0 * step_sec)
            m_mask = (freqs >= 180.0) & (freqs <= 1100.0)
            if np.any(m_mask):
                f_m = freqs[m_mask]
                s_m = spec[m_mask].copy()
                pk = float(np.max(s_m))
                if pk > 1e-5:
                    for c_i in range(2):
                        idx = int(np.argmax(s_m))
                        if float(s_m[idx]) < 0.35 * pk:
                            break
                        f0 = float(f_m[idx])
                        midi_p = int(np.clip(round(69.0 + 12.0 * math.log2(f0 / 440.0)), 48, 76))
                        vel_h = float(np.clip(rms * 2.4, 0.25, 0.68))
                        result["strings_legato"].append((trig_time + s0, trig_time + h_end, midi_p, vel_h))
                        if c_i == 0:
                            result["piano"].append((trig_time + s0, trig_time + s1, midi_p + 12, vel_h * 0.85))
                        for mult in (1.0, 2.0, 3.0):
                            s_m[np.abs(f_m - f0 * mult) <= f0 * mult * 0.06] *= 0.05

    return result
