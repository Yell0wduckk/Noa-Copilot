"""Worker-local instruments, BMS one-shots and VOICEVOX singing."""
from collections import defaultdict
import csv
from functools import lru_cache
import io
import json
import os
import random
from pathlib import Path
import re
import sqlite3
import struct
import sys
import zipfile

import numpy as np
from scipy import signal
import soxr

from .studio_mix import RATE, SLOTS, choice, draw, draw_int, read_stereo

# Local linked worktrees share the asset store. Installed distributions can point anywhere.
_PROJECT = Path(__file__).resolve()
LOCAL_ASSETS = next((p/'artifacts' for p in _PROJECT.parents if (p/'artifacts/vital-presets-library').is_dir()),
                    Path.cwd()/'artifacts')
ASSETS = Path(os.environ.get('MAIDATA_STUDIO_ASSETS', LOCAL_ASSETS))
PACKAGE = Path(__file__).with_name('assets')


@lru_cache(maxsize=16)
def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


class SplitFile(io.RawIOBase):
    """Seekable view of the stored collection ZIPs. Decode only selected keysounds."""
    def __init__(self, parts):
        self.parts = [p.open('rb') for p in parts]
        self.ends = np.cumsum([p.stat().st_size for p in parts]).tolist()
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = [0, self.pos, self.ends[-1]][whence] + offset
        return self.pos

    def readinto(self, buffer):
        from bisect import bisect_right
        i = bisect_right(self.ends, self.pos)
        if i == len(self.parts):
            return 0
        self.parts[i].seek(self.pos - (self.ends[i-1] if i else 0))
        count = self.parts[i].readinto(memoryview(buffer)[:self.ends[i] - self.pos])
        self.pos += count
        return count

    def close(self):
        for part in self.parts:
            part.close()
        super().close()


DRUM_TERMS = {
    'open_hat': r'open *(?:hi *)?hat|(?:hi *)?hat *(?:open|op\b)|\b(?:ohh?|hho|ohat|hh ?(?:o|op|open)|o hat)\b',
    'shaker': r'shaker|maraca', 'tamb': r'tamb', 'perc': r'perc',
    'clap': r'clap|handclap', 'snare': r'snare|\bsd\b|\bsnr\b',
    'closed_hat': r'hat|\bhh[ct]?\b|\bchh?\b', 'china': r'china', 'ride': r'ride',
    'crash': r'crash|cym|clash|splash', 'kick': r'kick|bass[ _-]*drum|\bbd\b',
}
DRUM_SUBTYPES = {'kick':('kick',), 'snare':('snare','clap'), 'hat':('closed_hat',),
                 'open_hat':('open_hat',), 'crash':('crash','china','ride')}
DRUM_SLOT = {t: s for s,types in enumerate((('kick',), ('snare','clap'),
             ('closed_hat','open_hat','shaker','tamb','perc'), ('crash','china','ride'))) for t in types}
PATTERNS = [(k,re.compile(v)) for k,v in DRUM_TERMS.items()]
VELOCITY_TAG = re.compile(r'(?:^|[_ -])v(\d{1,3})(?!\d)',re.I)


def drum_subtype(name):
    stem = re.sub(r'([a-z])([A-Z])', r'\1 \2', Path(name).stem)
    stem = re.sub(r'[\W_\d]+', ' ', stem.lower())
    return next((k for k,pat in PATTERNS if pat.search(stem)), None)


class SampleBank(dict):
    def __init__(self, rows):
        super().__init__(rows)
        self.sources = drum_sources(rows)
        self.plucks = pluck_sources(rows.get('6',()))


def drum_sources(bank):
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for slot, default in enumerate(('kick','snare','closed_hat','crash')):
        for item in bank.get(str(slot), ()):
            path = Path(item['path'])
            subtype = item.get('subtype') or drum_subtype(path.name) or default
            source = (item.get('collection',''), item.get('song') or str(path.parent))
            family = re.split(r'[_ -]v\d|[_ -](?:rr)?\d', path.stem.lower())[0]
            family = re.sub(r'\d+$','',family)
            groups[source][subtype][family].append(item)
    return groups


class Archive:
    def __init__(self, root):
        root = Path(root)
        self.db = sqlite3.connect((root/'index.sqlite').resolve().as_uri()+'?mode=ro', uri=True)
        self.zip = zipfile.ZipFile(io.BufferedReader(SplitFile(sorted(root.glob('all-samples.zip.part*')))))
        term_types = {}
        terms = {'bd','sd','hh','ch','oh','snr','cym','perc','cc'} | set(DRUM_TERMS)
        self.samples = defaultdict(list)
        seen = set()
        melodic_terms = {'piano','pluck','koto','harp','marimba','kalimba','pizz','guitar'}
        melodic_songs = {}

        with (root/'target-representatives.csv').open(encoding='utf-8-sig', newline='') as stream:
            for r in csv.DictReader(stream):
                if r['term'] in melodic_terms:
                    melodic_songs[(r['collection'],r['song'])] = True
                term = r['term']
                if term not in term_types:
                    term_types[term] = term in terms or drum_subtype(term) is not None
                if not term_types[term]:
                    continue
                name = Path(r['path']).stem
                if re.search('roll|loop|fill|reverse|revcym', name, re.I):
                    continue
                subtype = drum_subtype(name)
                key = (r['collection'], r['path'])
                if subtype is None or key in seen:
                    continue
                seen.add(key)
                item = dict(collection=r['collection'], song=r['song'], path=r['path'], subtype=subtype)
                if velocity := VELOCITY_TAG.search(name):
                    item['velocity'] = min(1., int(velocity[1])/127)
                self.samples[str(DRUM_SLOT[subtype])].append(item)
        # Named one-shot multisamples, grouped by song and instrument (indexed path prefix).
        mml = re.compile(r'^b_(.+?)_v(\d+)(?:l[\d-]+)?o(\d)([a-g])([ps#m-]?)\.', re.I)
        for collection, song in melodic_songs:
            prefix = song+'/'
            for path, name in self.db.execute('select path,basename from files where collection=? and path>=? and path<?',
                                               (collection,prefix,song+'0')):
                m = mml.match(name)
                if not m or not any(t in m[1].lower() for t in melodic_terms):
                    continue
                # MID2BMS o5c in these multisample banks is sounding MIDI 60.
                pitch = int(m[3])*12 + {'c':0,'d':2,'e':4,'f':5,'g':7,'a':9,'b':11}[m[4].lower()]
                pitch += (1 if m[5] in ('p','s','#') else -1 if m[5] in ('m','-') else 0)
                self.samples['6'].append(dict(collection=collection, path=path, pitch=pitch,
                                            instrument=song+'/'+m[1], velocity=int(m[2])/127))
        self.parts = {}

    def audio(self, collection, path):
        col, member = self.db.execute('select b.collection,b.member from files f join blobs b '
            'on f.raw_hash=b.hash where f.collection=? and f.path=?', (collection, path)).fetchone()
        if col not in self.parts:
            self.parts[col] = zipfile.ZipFile(self.zip.open(f'collections/{col}--samples.zip'))
        return read_stereo(io.BytesIO(self.parts[col].read(member)))


@lru_cache(maxsize=2)
def archive(directory):
    return Archive(directory)


@lru_cache(maxsize=128)
def clip(root, collection, path, start, duration):
    if collection:
        data = archive(root).audio(collection, path)
        a = round(start * RATE)
        return data[a:] if duration is None else data[a:a + round(duration * RATE)]
    return read_stereo(path, start, duration)


def sample_clip(item, root):
    return clip(root, item.get('collection'), item['path'], item.get('start', 0.), item.get('length'))


def load_bank(opts):
    directory = opts.get('bms_archive', str(ASSETS/'bms-samples-l4-20261106/final-download'))
    return _load_bank(opts.get('sample_bank'), directory)


@lru_cache(maxsize=4)
def _load_bank(path, directory):
    bank = {}
    if path:
        for slot,items in read_json(path).items():
            bank[slot] = [s if s.get('collection') else {**s, 'path':str(Path(path).parent/s['path'])} for s in items]
    if not all(bank.get(str(k)) for k in range(4)) and (Path(directory)/'index.sqlite').is_file():
        bank = {**archive(directory).samples, **bank}
    return directory, SampleBank(bank)


def trim_attack(wave, onset_db):
    magnitude = np.max(np.abs(wave),axis=1)
    sounding = np.flatnonzero(magnitude > max(1e-10, magnitude.max(initial=0)*10**(onset_db/20)))
    return wave[sounding[0]:] if len(sounding) else wave[:0]


def prepare_hit(item, root, role, opts):
    wave = trim_attack(sample_clip(item,root), opts['sample_onset_db'])
    if not len(wave):
        return None
    maximum = opts['drum_max_seconds'][role]
    wave = wave[:round(maximum*RATE)].copy()
    if role in opts['drum_lowband_max'] or role in opts['drum_lowband_min']:
        hz, energy = signal.welch(wave, RATE, nperseg=min(2048,len(wave)), axis=0, detrend=False)
        low = energy[hz < opts['drum_lowband_hz']].sum()/max(energy.sum(),1e-20)
        if not opts['drum_lowband_min'].get(role,0.) <= low <= opts['drum_lowband_max'].get(role,1.):
            return None
    fade = min(len(wave), round(opts['drum_end_fade_ms']*RATE/1000))
    if fade:
        wave[-fade:] *= np.linspace(1,0,fade,dtype=np.float32)[:,None]
    return wave


class DrumKit:
    """One source and one instrument family. Real velocity layers retain their native gain."""
    def __init__(self, candidates, opts, root, rng, role='snare'):
        self.layers = defaultdict(list)
        self.cursors = defaultdict(int)
        self.paths, self.variations = [], []
        self.opts = opts
        rr = draw_int(rng,opts['drum_round_robin'])
        groups = defaultdict(list)
        for item in candidates:
            tag = VELOCITY_TAG.search(Path(item['path']).stem)
            velocity = item.get('velocity', int(tag[1])/127 if tag else None)
            groups[velocity].append(item)
        # Do not invent velocity layers from loudness differences between unrelated recordings.
        if len(groups)>1:
            groups.pop(None,None)
        for level,items in groups.items():
            for item in rng.sample(items,len(items)):
                wave = prepare_hit(item,root,role,opts)
                if wave is None:
                    continue
                self.layers[level].append(wave)
                self.paths.append(item['path'])
                if len(self.layers[level])>=rr:
                    break
        if len(self.paths)==1:
            pool = next(iter(self.layers.values()))
            base = pool[0]
            for _ in range(max(0,opts['drum_variation_count']-1)):
                cents = draw(rng,opts['drum_variation_cents'])
                gain = draw(rng,opts['drum_variation_db'])
                shelf = draw(rng,opts['drum_variation_shelf_db'])
                varied = soxr.resample(base,RATE*2**(cents/1200),RATE,quality='HQ') if cents else base
                if shelf:
                    high = signal.sosfilt(signal.butter(2,5000,'high',fs=RATE,output='sos'),varied,axis=0)
                    varied = varied + (10**(shelf/20)-1)*high
                pool.append((varied[:round(opts['drum_max_seconds'][role]*RATE)]*10**(gain/20)).astype(np.float32))
                self.variations.append(dict(cents=cents,gain_db=gain,shelf_db=shelf))

    def hit(self, velocity):
        level = None if None in self.layers else min(self.layers,key=lambda v:abs(v-velocity))
        waves = self.layers[level]
        wave = waves[self.cursors[level] % len(waves)]
        self.cursors[level] += 1
        gain = (velocity**self.opts['drum_velocity_gamma'] if level is None else
                np.clip(velocity/max(level,1/127), *self.opts['drum_layer_gain_range']))
        return wave, float(gain)


def select_kit(bank, required, root, opts, rng):
    sources = bank.sources if isinstance(bank,SampleBank) else drum_sources(bank)
    if not sources or not required:
        return {}, {}
    complete = [s for s,g in sources.items() if 'kick' in g and ('snare' in g or 'clap' in g) and 'closed_hat' in g]
    primary = rng.choice(complete or list(sources))
    kits, details = {}, {}
    for role in required:
        types = DRUM_SUBTYPES[role]
        alternatives = [s for s,g in sources.items() if s!=primary and any(t in g for t in types)]
        rng.shuffle(alternatives)
        order = [primary,*alternatives]
        if not opts['drum_kit_single_source']:
            rng.shuffle(order)
        for source in order:
            subtypes = [t for t in types if t in sources[source]]
            rng.shuffle(subtypes)
            for subtype in subtypes:
                families = list(sources[source][subtype].values())
                rng.shuffle(families)
                for family in families:
                    kit = DrumKit(family,opts,root,rng,'hat' if role=='open_hat' else role)
                    if kit.paths:
                        kits[role] = kit
                        details[role] = dict(source=source,subtype=subtype,paths=kit.paths,
                                             fallback=source!=primary,variations=kit.variations)
                        break
                if role in kits:
                    break
            if role in kits:
                break
    return kits, dict(kit_source=primary, roles=details)


def pluck_sources(items):
    banks = defaultdict(lambda:defaultdict(list))
    for item in items:
        if 'pitch' in item:
            key = (item.get('collection',''),item.get('song') or str(Path(item['path']).parent),
                   item.get('instrument','pluck'))
            banks[key][item['pitch']].append(item)
    return banks


def choose_pluck(bank, notes, opts, rng):
    banks = bank.plucks if isinstance(bank,SampleBank) else pluck_sources(bank.get('6',()))
    pitches = {n.pitch for n in notes}
    keys = sorted(banks)
    for key in rng.sample(keys,min(len(keys),opts['pluck_instrument_attempts'])):
        roots = banks[key]
        distances = {p:min(abs(p-k) for k in roots) for p in pitches}
        for radius in (opts['max_transpose_semitones'],opts['pluck_extended_transpose_semitones']):
            if all(d<=radius for d in distances.values()):
                pool = {p:[s for k,items in roots.items() if abs(p-k)==distances[p] for s in items] for p in pitches}
                return key,radius,pool
    return None


def nearest_sample(bank, pitch, velocity, radius):
    candidates = [s for s in bank if 'pitch' in s and abs(s['pitch']-pitch) <= radius]
    return min(candidates, key=lambda s:(abs(s['pitch']-pitch), abs(s.get('velocity', velocity)-velocity)),
               default=None)


@lru_cache(maxsize=160)
def transposed(root, collection, path, start, length, semitones, onset_db=-60):
    data = trim_attack(clip(root, collection, path, start, length), onset_db)
    if semitones:
        data = soxr.resample(data, RATE * 2 ** (semitones/12), RATE, quality='HQ')
    return data


@lru_cache(maxsize=2)
def vst_path(configured):
    from pedalboard import VST3Plugin
    if configured:
        return str(configured)
    for local in (
        Path.cwd() / 'artifacts/studio-instruments/Vial.vst3',
        ASSETS / 'studio-instruments/Vial.vst3',
        *(p / 'artifacts/studio-instruments/Vial.vst3' for p in _PROJECT.parents),
    ):
        if local.exists():
            return str(local.resolve())
    return next((p for p in VST3Plugin.installed_plugins if Path(p).stem.lower() in ('vital','vial','vitalium')),
                'Vital.vst3')


def vital_preset(class_header, settings):
    # VST3 preset header + component stream + chunk list, using the hosted instrument's class ID.
    chunk = json.dumps(settings, separators=(',', ':')).encode() + b'\0'
    return (class_header[:40] + struct.pack('<Q', 48+len(chunk)) + chunk +
            struct.pack('<4sI4sQQ', b'List', 1, b'Comp', 48, len(chunk)))


@lru_cache(maxsize=2)
def instrument(path):
    from pedalboard import load_plugin
    plugin = load_plugin(path)
    return plugin, plugin.preset_data


def component(data):
    directory = struct.unpack_from('<Q', data, 40)[0]
    chunks = struct.iter_unpack('<4sQQ', data[directory+8:])
    offset, size = next((o,s) for kind,o,s in chunks if kind == b'Comp')
    return json.loads(data[offset:offset+size].split(b'\0',1)[0])


def preset_role(row):
    if row['slot'] == 7:
        return 'pad' if re.search(r'pad|atmos|strings', row.get('preset_style','')+' '+row.get('preset_name',''), re.I) else 'lead'
    return {5:'bass', 6:'pluck'}.get(row['slot'])


@lru_cache(maxsize=8)
def preset_library(index, library_root):
    data = read_json(index)
    rows = data['presets'] if isinstance(data, dict) else data
    groups = defaultdict(dict)
    for row in rows:
        role = row.get('role') or preset_role(row)
        if role and not row.get('reason_excluded'):
            path = os.path.abspath(Path(library_root)/row['rel_path'])
            groups[role][path] = row.get('pitch_offset')
    return groups


def preset_candidates(role, opts):
    library = ASSETS/'vital-presets-library'
    vetted = opts.get('preset_vetted') or 'vital-preset-vetted.json'
    bundled = PACKAGE/vetted if not Path(vetted).is_absolute() else Path(vetted)
    index = opts.get('preset_index')
    root = opts.get('preset_root') or (Path(index).parent if index else library)
    known = preset_library(str(bundled), str(root)).get(role, {}) if bundled.is_file() and Path(root).is_dir() else {}
    bank = preset_library(str(index), str(root)).get(role,{}) if index else known
    if index:
        known = {**known, **{p:v for p,v in bank.items() if v is not None}}
    paths = opts.get('presets', {})
    explicit = paths.get(role, paths.get(str(SLOTS[role]))) if isinstance(paths, dict) else paths
    weights = (dict(explicit) if isinstance(explicit, dict) else dict.fromkeys(explicit, 1.)) if explicit is not None else dict.fromkeys(bank, 1.)
    allowed = (opts.get('preset_allowlist') or {}).get(role)
    if allowed is not None:
        weights = {p:w for p,w in weights.items() if p in allowed or Path(p).name in allowed or Path(p).stem in allowed}
    return weights, known


# Cache a load verdict per host/preset/phase. Never use the previous role's state after a failed load.
_load_verdict = {}


def load_patch(path, preset, phase):
    key = path, preset, phase
    if _load_verdict.get(key) is False:
        return None
    for attempt in range(2):
        plugin, initial = instrument(path)
        try:
            patch = component(Path(preset).read_bytes()) if Path(preset).suffix.lower()=='.vstpreset' else read_json(preset)
            settings = dict(patch['settings'], delay_on=0., reverb_on=0.)
            settings.update({f'osc_{i}_random_phase':phase for i in (1,2,3)})
            # The component loads save_info only after installing controls, wavetables and modulation.
            # A unique name survives version migrations and detects a silently rejected component.
            name = str(preset)
            plugin.preset_data = vital_preset(initial, dict(patch, settings=settings, preset_name=name))
            if key in _load_verdict:
                return plugin
            actual = component(plugin.preset_data)
            if (actual.get('preset_name') == name and
                    all(abs(actual['settings'][k]-settings[k]) < 1e-5 for k in ('delay_on','reverb_on'))):
                _load_verdict[key] = True
                return plugin
        except (OSError, ValueError, KeyError, RuntimeError, StopIteration, struct.error):
            pass
        if not attempt:
            instrument.cache_clear()
    _load_verdict[key] = False
    return None


@lru_cache(maxsize=1024)
def override_pitch(path, preset, phase, role, buffer_size, tolerance):
    from .studio_presets import probe_pitch
    plugin = load_patch(path, preset, phase)
    return probe_pitch(plugin, role, buffer_size, tolerance) if plugin is not None else None


def preset_for(role, opts, rng, pitch_range=None):
    path = vst_path(opts.get('vital_vst3') or os.getenv('VITAL_VST3'))
    weights, known = preset_candidates(role, opts)
    failures, outside = [], []
    while weights:
        preset = choice(rng, weights)
        plugin = load_patch(path, preset, opts['vst_random_phase'])
        offset = known.get(os.path.abspath(preset))
        if plugin is not None and offset is None:
            offset = override_pitch(path, preset, opts['vst_random_phase'], role, opts['vst_buffer_size'],
                                    opts['preset_vet']['pitch_tolerance_semitones'])
        if plugin is None or offset is None:
            failures.append(preset)
        elif pitch_range and (pitch_range[0]-offset<0 or pitch_range[1]-offset>127):
            outside.append(preset)
        else:
            return plugin, dict(preset=preset,pitch_offset=offset,preset_load_failures=failures,pitch_range_skips=outside)
        del weights[preset]
    # A fresh host init patch has a concert-pitch oscillator, including when all overrides failed.
    plugin, initial = instrument(path)
    patch = component(initial)
    settings = dict(patch['settings'], osc_2_on=0., osc_1_transpose=0., osc_1_tune=0.,
                    osc_1_random_phase=opts['vst_random_phase'], reverb_on=0., delay_on=0.,
                    filter_1_on=1., filter_1_cutoff=rng.uniform(56,88),
                    env_1_attack=.18 if role=='pad' else .02, env_1_sustain=.18 if role=='pluck' else .78)
    plugin.preset_data = vital_preset(initial, dict(patch, settings=settings))
    return plugin, dict(preset='init-variation', pitch_offset=0, cutoff=settings['filter_1_cutoff'],
                        preset_load_failures=failures,pitch_range_skips=outside)


def _pan_layer(wave, pan):
    if abs(pan) < 1e-4:
        return wave
    angle = (pan + 1.0) * (np.pi / 4.0)
    gl, gr = float(np.cos(angle)), float(np.sin(angle))
    out = wave.copy()
    if pan < 0:
        out[:, 0] = wave[:, 0] * gl + wave[:, 1] * (0.25 * abs(pan))
        out[:, 1] = wave[:, 1] * gr
    else:
        out[:, 0] = wave[:, 0] * gl
        out[:, 1] = wave[:, 1] * gr + wave[:, 0] * (0.25 * abs(pan))
    return out


def _apply_tape_stop(snippet):
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


def _sculpt_and_render_vital_layer(
    path,
    preset_path,
    pitch_offset,
    notes,
    grid,
    frames,
    opts,
    *,
    role="lead",
    mono_choke=False,
    max_attack_sec=0.004,
    release_sec=0.08,
    min_sustain=None,
    min_decay_sec=None,
    max_detune=1.65,
    open_dark_filters=True,
    disable_sub_osc3=False,
):
    plugin, initial = instrument(path)
    if preset_path != "init-variation" and Path(preset_path).exists():
        patch = component(Path(preset_path).read_bytes()) if Path(preset_path).suffix.lower() == ".vstpreset" else read_json(preset_path)
    else:
        patch = component(initial)
    settings = dict(patch["settings"])
    settings["delay_on"] = 0.0
    settings["reverb_on"] = 0.0
    settings["chorus_on"] = 0.0
    settings["legato"] = 0.0
    settings["polyphony"] = float(1 if mono_choke else (16 if role in ("pluck", "pad") else 8))
    for env_idx in (1, 2, 3):
        atk_k = f"env_{env_idx}_attack"
        if atk_k in settings:
            settings[atk_k] = min(float(settings[atk_k]), max_attack_sec * (1.0 if env_idx == 1 else 1.8))
    if "env_1_release" in settings:
        settings["env_1_release"] = float(np.clip(settings["env_1_release"], 0.035 if mono_choke else 0.055, release_sec))
    if min_sustain is not None and "env_1_sustain" in settings:
        settings["env_1_sustain"] = max(float(settings["env_1_sustain"]), float(min_sustain))
    if min_decay_sec is not None and "env_1_decay" in settings:
        settings["env_1_decay"] = max(float(settings["env_1_decay"]), float(min_decay_sec))
    if open_dark_filters:
        for fi in (1, 2):
            if settings.get(f"filter_{fi}_on", 0.0) > 0.5 and float(settings.get(f"filter_{fi}_cutoff", 100.0)) < 82.0:
                settings[f"filter_{fi}_on"] = 0.0
    if disable_sub_osc3 and int(settings.get("osc_3_transpose", 0)) <= -12:
        settings["osc_3_on"] = 0.0
    for i in (1, 2, 3):
        settings[f"osc_{i}_random_phase"] = float(opts.get("vst_random_phase", 0.0))
        det_k = f"osc_{i}_unison_detune"
        if det_k in settings:
            settings[det_k] = min(float(settings[det_k]), float(max_detune))

    plugin.preset_data = vital_preset(initial, dict(patch, settings=settings, preset_name=str(preset_path)))

    messages = []
    if mono_choke:
        evs_all = sorted(
            [
                (
                    grid.second(n.start),
                    grid.second(n.end),
                    int(np.clip(n.pitch - pitch_offset, 12, 115)),
                    max(8, min(127, round(127 * n.velocity))),
                )
                for n in notes
            ],
            key=lambda x: x[0],
        )
        dedup = []
        for ev in evs_all:
            if dedup and abs(ev[0] - dedup[-1][0]) < 0.004:
                if ev[3] > dedup[-1][3]:
                    dedup[-1] = ev
            else:
                dedup.append(ev)
        for idx, (s_sec, e_sec, key, vel_int) in enumerate(dedup):
            if idx + 1 < len(dedup):
                e_sec = min(e_sec, dedup[idx + 1][0] - 0.002)
            e_sec = max(s_sec + 0.016, e_sec)
            messages.append((s_sec, 1, key, vel_int))
            messages.append((e_sec, 0, key, 0))
    else:
        by_pitch = defaultdict(list)
        for n in notes:
            key = int(np.clip(n.pitch - pitch_offset, 12, 115))
            by_pitch[key].append((grid.second(n.start), grid.second(n.end), max(8, min(127, round(127 * n.velocity)))))
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
    duration = float(np.nextafter(np.float32((frames + 1) / RATE), np.float32(np.inf)))
    wave = plugin(stream, duration=duration, sample_rate=RATE, num_channels=2, buffer_size=opts["vst_buffer_size"], reset=True)
    return np.ascontiguousarray(wave.T[:frames], dtype=np.float32)


def _render_hybrid_acoustic_role_layer(role, notes, grid, frames, opts):
    if not opts.get("hybrid_acoustic", True) or role == "bass":
        return None, ()
    try:
        from .sf2_synth import render_hybrid_acoustic_stem
    except Exception:
        return None, ()
    default_fams = {
        "lead": ("violin", "piano"),
        "pluck": ("piano", "harp"),
        "pad": ("strings_legato", "brass"),
    }
    fams = opts.get(f"acoustic_{role}_families") or default_fams.get(role, ())
    if not fams:
        return None, ()
    evs = [(grid.second(n.start), grid.second(n.end), int(n.pitch), float(n.velocity)) for n in notes]
    aco_mix = np.zeros((frames, 2), dtype=np.float32)
    pans = (-0.24, +0.24, 0.0)
    weights = (0.65, 0.45, 0.35)
    for idx, fam in enumerate(fams[:3]):
        w_fam = render_hybrid_acoustic_stem(fam, evs, frames)
        aco_mix += _pan_layer(w_fam, pans[idx]) * weights[idx]
    return aco_mix, tuple(fams[:3])


def render_vital(role, notes, grid, frames, opts, rng):
    keys = [n.pitch for n in notes]
    p_range = (min(keys), max(keys))
    plugin, info = preset_for(role, opts, rng, p_range)
    # Preserve exact stream behavior for mocked/unit-test instruments or when multilayer_synth is disabled
    if "preset" not in info or not opts.get("multilayer_synth", True) or info["preset"] == "init-variation":
        messages = []
        for n in notes:
            key = n.pitch - info["pitch_offset"]
            messages.extend([(n.start, 1, key, max(1, round(127 * n.velocity))), (n.end, 0, key, 0)])
        stream = [(bytes([0x90 if on else 0x80, pitch, vel]), grid.second(t)) for t, on, pitch, vel in sorted(messages)]
        duration = float(np.nextafter(np.float32((frames + 1) / RATE), np.float32(np.inf)))
        wave = plugin(stream, duration=duration, sample_rate=RATE, num_channels=2, buffer_size=opts["vst_buffer_size"], reset=True)
        return np.ascontiguousarray(wave.T[:frames]), info

    path = vst_path(opts.get("vital_vst3") or os.getenv("VITAL_VST3"))
    layer_rng = random.Random(f"{role}:{info['preset']}:{len(notes)}:{p_range[0]}:{p_range[1]}")
    _, info2 = preset_for(role, opts, layer_rng, p_range)
    has_layer2 = "preset" in info2 and info2["preset"] != "init-variation"
    if has_layer2:
        info["secondary_preset"] = info2["preset"]
    aco_wave, aco_fams = _render_hybrid_acoustic_role_layer(role, notes, grid, frames, opts)
    if aco_fams:
        info["acoustic_families"] = list(aco_fams)
    aco_dom = bool(opts.get("acoustic_dominant", False))

    if role == "lead":
        # Check if lead notes are mostly monophonic (single melody line) so we apply monophonic keysound choke
        start_counts = defaultdict(int)
        for n in notes:
            start_counts[n.start] += 1
        is_mono = sum(1 for c in start_counts.values() if c == 1) >= 0.75 * max(1, len(start_counts))
        w1 = _sculpt_and_render_vital_layer(
            path, info["preset"], info["pitch_offset"], notes, grid, frames, opts,
            role="lead", mono_choke=is_mono, max_attack_sec=0.003, release_sec=0.048,
            max_detune=1.35, open_dark_filters=True, disable_sub_osc3=True,
        )
        if has_layer2:
            w2 = _sculpt_and_render_vital_layer(
                path, info2["preset"], info2["pitch_offset"], notes, grid, frames, opts,
                role="lead", mono_choke=is_mono, max_attack_sec=0.002, release_sec=0.048,
                max_detune=1.35, open_dark_filters=True, disable_sub_osc3=True,
            )
            wave = w1 * 0.68 + w2 * 0.42
        else:
            wave = w1
        if aco_wave is not None:
            wave = wave * (0.38 if aco_dom else 0.74) + aco_wave * (0.72 if aco_dom else 0.34)
        sorted_lead = sorted(notes, key=lambda x: x.start)
        for idx, n in enumerate(sorted_lead):
            is_phrase_end = (
                opts.get("auto_tape_stop", True)
                and (n.end - n.start) >= 96
                and idx + 1 < len(sorted_lead)
                and (sorted_lead[idx + 1].start - n.end) >= 192
            )
            if "tape_stop" in (n.articulation or "") or is_phrase_end:
                a = round(grid.second(n.start) * RATE)
                b = min(frames, round((grid.second(n.end) + 0.05) * RATE))
                if b > a + 64:
                    wave[a:b] = _apply_tape_stop(wave[a:b].copy())
        return wave.astype(np.float32), info

    if role == "bass":
        w1 = _sculpt_and_render_vital_layer(
            path, info["preset"], info["pitch_offset"], notes, grid, frames, opts,
            role="bass", mono_choke=True, max_attack_sec=0.003, release_sec=0.075,
            max_detune=1.15, open_dark_filters=False,
        )
        # Phase-locked fundamental + 2nd-harmonic Sub Sine reinforcement (35-155 Hz)
        sub_sine = np.zeros((frames, 2), dtype=np.float32)
        for n in notes:
            a = round(grid.second(n.start) * RATE)
            b = min(frames, round((grid.second(n.end) + 0.02) * RATE))
            if b <= a + 8:
                continue
            f0 = 440.0 * (2.0 ** ((n.pitch - 69) / 12.0))
            t_arr = np.arange(b - a, dtype=np.float32) / RATE
            sig = (np.sin(2.0 * np.pi * f0 * t_arr) + 0.30 * np.sin(4.0 * np.pi * f0 * t_arr)) * float(n.velocity) * 0.22
            atk = min(round(0.003 * RATE), len(sig) // 3)
            rel = min(round(0.015 * RATE), len(sig) // 3)
            if atk > 0:
                sig[:atk] *= np.linspace(0.0, 1.0, atk, dtype=np.float32)
            if rel > 0:
                sig[-rel:] *= np.linspace(1.0, 0.0, rel, dtype=np.float32)
            sub_sine[a:b, 0] += sig
            sub_sine[a:b, 1] += sig
        sos_sub = signal.butter(2, 160.0, "lowpass", fs=RATE, output="sos")
        sub_sine = signal.sosfilt(sos_sub, sub_sine, axis=0).astype(np.float32)
        if has_layer2:
            w2 = _sculpt_and_render_vital_layer(
                path, info2["preset"], info2["pitch_offset"], notes, grid, frames, opts,
                role="bass", mono_choke=True, max_attack_sec=0.003, release_sec=0.08,
                max_detune=1.35, open_dark_filters=True,
            )
            sos_mid = signal.butter(2, [95.0, 2200.0], "bandpass", fs=RATE, output="sos")
            w2_mid = np.tanh(signal.sosfilt(sos_mid, w2, axis=0) * 1.35).astype(np.float32)
            wave = w1 * 0.62 + sub_sine + w2_mid * 0.42
        else:
            wave = w1 * 0.75 + sub_sine
        return wave.astype(np.float32), info

    if role == "pluck":
        w1 = _sculpt_and_render_vital_layer(
            path, info["preset"], info["pitch_offset"], notes, grid, frames, opts,
            role="pluck", mono_choke=False, max_attack_sec=0.003, release_sec=0.08,
            min_sustain=0.55, min_decay_sec=0.22, max_detune=1.65, open_dark_filters=True,
        )
        if has_layer2:
            w2 = _sculpt_and_render_vital_layer(
                path, info2["preset"], info2["pitch_offset"], notes, grid, frames, opts,
                role="pluck", mono_choke=False, max_attack_sec=0.003, release_sec=0.08,
                min_sustain=0.55, min_decay_sec=0.22, max_detune=1.65, open_dark_filters=True,
            )
            wave = _pan_layer(w1, -0.38) * 0.65 + _pan_layer(w2, +0.38) * 0.58
        else:
            wave = w1
        if aco_wave is not None:
            wave = wave * (0.35 if aco_dom else 0.72) + aco_wave * (0.75 if aco_dom else 0.38)
        return wave.astype(np.float32), info

    # role == "pad"
    w1 = _sculpt_and_render_vital_layer(
        path, info["preset"], info["pitch_offset"], notes, grid, frames, opts,
        role="pad", mono_choke=False, max_attack_sec=0.022, release_sec=0.18,
        min_sustain=0.82, max_detune=1.90, open_dark_filters=True,
    )
    if has_layer2:
        w2 = _sculpt_and_render_vital_layer(
            path, info2["preset"], info2["pitch_offset"], notes, grid, frames, opts,
            role="pad", mono_choke=False, max_attack_sec=0.025, release_sec=0.18,
            min_sustain=0.82, max_detune=1.90, open_dark_filters=True,
        )
        wave = _pan_layer(w1, -0.25) * 0.60 + _pan_layer(w2, +0.25) * 0.55
    else:
        wave = w1
    if aco_wave is not None:
        wave = wave * (0.35 if aco_dom else 0.68) + aco_wave * (0.75 if aco_dom else 0.42)
    return wave.astype(np.float32), info


@lru_cache(maxsize=2)
def voice_engine(directory, threads, models):
    from voicevox_core.blocking import Onnxruntime, OpenJtalk, Synthesizer, VoiceModelFile
    root = Path(directory)
    # VOICEVOX's binary release ships platform-specific runtime filenames.
    pattern = 'voicevox_onnxruntime.dll' if sys.platform == 'win32' else (
        '*voicevox_onnxruntime*.dylib' if sys.platform == 'darwin' else '*voicevox_onnxruntime.so*')
    lib = next((root/'onnxruntime/lib').glob(pattern))
    engine = Synthesizer(Onnxruntime.load_once(filename=str(lib)), OpenJtalk(str(next((root/'dict').iterdir()))),
                         acceleration_mode='CPU', cpu_num_threads=threads)
    for model in models:
        with VoiceModelFile.open(str(root/'models/vvms'/model)) as f:
            engine.load_voice_model(f)
    return engine


def render_voice(notes, grid, frames, opts, rng):
    from voicevox_core import Note, Score
    from pedalboard import Compressor
    root = opts.get('voicevox_dir', ASSETS/'svs-and-vocaloid-library/voicevox/voicevox_core')
    engine = voice_engine(str(root), opts['voicevox_threads'], tuple(opts.get('voicevox_models', ['s0.vvm'])))
    voices = opts.get('singers') or [s.id for speaker in engine.metas() for s in speaker.styles
                                     if s.type in ('frame_decode','sing')]
    singer = int(choice(rng,voices))
    double = None
    if rng.random()<opts['vocal_double_probability']:
        double = dict(level_db=draw(rng,opts['vocal_double_db']),detune_cents=draw(rng,opts['vocal_double_cents']),
                      lfo_hz=draw(rng,opts['vocal_double_lfo_hz']),lfo_cents=draw(rng,opts['vocal_double_lfo_cents']),
                      phrase_seeds=[])
    wave = np.zeros((frames,2), np.float32)
    # Absolute VOICEVOX frame boundaries prevent syllable rounding drift across BPM changes.
    phrases = []
    for n in notes:
        if not phrases or grid.second(n.start)-grid.second(phrases[-1][-1].end) > opts.get('vocal_phrase_gap', .25) or len(phrases[-1]) >= opts.get('vocal_phrase_notes',32):
            phrases.append([])
        phrases[-1].append(n)
    fps, pad = 93.75, 12
    for phrase in phrases:
        origin = grid.second(phrase[0].start)
        bounds = [(pad+round((grid.second(n.start)-origin)*fps),pad+round((grid.second(n.end)-origin)*fps))
                  for n in phrase]
        for take in range(2 if double else 1):
            shifts = {}
            if take:
                seed = rng.getrandbits(64)
                double['phrase_seeds'].append(seed)
                local = random.Random(seed)
                for t in sorted({t for pair in bounds for t in pair}):
                    shifts[t] = local.choice((-1,1))*draw_int(local,opts['vocal_double_boundary_frames'])
                phase = local.uniform(0,2*np.pi)
            score, spans, cursor = [], [], 0
            for n,(a,b) in zip(phrase,bounds):
                a = max(cursor,a+shifts.get(a,0))
                b = max(a+1,b+shifts.get(b,0))
                if a>cursor:
                    score.append(Note(a-cursor,''))
                score.append(Note(b-a,n.lyric,key=n.pitch))
                spans.append((a,b,n.velocity))
                cursor = b
            query = engine.create_sing_frame_audio_query(Score([*score,Note(pad,'')]),opts['voicevox_teacher'])
            volume = np.asarray(query.volume)
            for a,b,velocity in spans:
                volume[a:b] *= velocity
            query.volume = volume.tolist()
            if take:
                time = np.arange(len(query.f0))/fps
                cents = double['detune_cents'] + double['lfo_cents']*np.sin(2*np.pi*double['lfo_hz']*time+phase)
                query.f0 = (np.asarray(query.f0)*2**(cents/1200)).tolist()
            audio = read_stereo(io.BytesIO(engine.frame_synthesis(query,singer)))
            place = round((origin-pad/fps)*RATE)
            src,dst = max(0,-place),max(0,place)
            count = min(len(audio)-src,frames-dst)
            if count>0:
                wave[dst:dst+count] += audio[src:src+count]*(10**(double['level_db']/20) if take else 1.)
    # Split-band de-essing before broadband vocal compression.
    hp = signal.butter(2, opts['vocal_deess_hz'], 'highpass', fs=RATE, output='sos')
    high = signal.sosfilt(hp, wave, axis=0).astype(np.float32)
    def compress(data, threshold, ratio):
        fx = Compressor(threshold_db=threshold, ratio=ratio, attack_ms=opts['vocal_attack_ms'],
                        release_ms=opts['vocal_release_ms'])
        return fx(np.ascontiguousarray(data.T), RATE).T
    wave += compress(high, opts['vocal_deess_threshold_db'], opts['vocal_deess_ratio']) - high
    return compress(wave, opts['vocal_compressor_threshold_db'], opts['vocal_compressor_ratio']), dict(singer=singer,double=double)
