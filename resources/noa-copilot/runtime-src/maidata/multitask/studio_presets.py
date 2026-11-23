"""Offline Vital audition bank. Only the selected preset is opened during synthesis."""
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
from pathlib import Path

import numpy as np
from scipy import signal
import soundfile as sf

from .config import studio_options
from .studio_mix import RATE
from .studio_sources import ASSETS, component, instrument, load_patch, preset_role, read_json, vst_path

PROBES = {'bass': (36, 43), 'lead': (60, 72), 'pluck': (60, 72), 'pad': (48, 55, 60)}


def audition(plugin, pitches, gate, buffer):
    stream = [(bytes((144, k, 100)), 0.) for k in pitches]
    stream += [(bytes((128, k, 0)), gate) for k in pitches]
    return plugin(stream, duration=gate + .6, sample_rate=RATE, num_channels=2,
                  buffer_size=buffer, reset=True).T


def fundamental(wave):
    """YIN on three active frames of the strongest channel (also works on wide stereo)."""
    x = wave[:, int(np.argmax(np.sum(wave**2, axis=0)))]
    x = signal.resample_poly(x, 1, 4)
    rate = RATE / 4
    size = min(2048, len(x))
    pitches = []
    for start in np.linspace(0, len(x)-size, 5, dtype=int):
        y = x[start:start+size].astype(float)
        y -= y.mean()
        if y.std() < 1e-6:
            continue
        high = min(size//2, int(rate/20))
        lag = np.arange(high+1)
        corr = signal.correlate(y, y, method='fft')[size-1:size+high]
        energy = np.r_[0., np.cumsum(y*y)]
        d = np.maximum(0., energy[size-lag]+energy[-1]-energy[lag]-2*corr)
        cmnd = np.r_[1., d[1:]*lag[1:]/np.maximum(np.cumsum(d[1:]), 1e-16)]
        troughs = signal.find_peaks(-cmnd)[0]
        troughs = troughs[(troughs >= 3) & (cmnd[troughs] < .20)]
        if troughs.size:
            i = troughs[0]
            curvature = cmnd[i-1] - 2*cmnd[i] + cmnd[i+1]
            shift = (cmnd[i-1]-cmnd[i+1])/(2*max(curvature, 1e-12))
            pitches.append(69+12*np.log2(rate/(i+shift)/440))
    return float(np.median(pitches)) if pitches else None


def offset_from_probes(probes, tolerance):
    errors = []
    for wave, key in probes:
        midi = fundamental(wave[:RATE//2])
        if midi is None:
            return None, errors
        errors.append(round(midi-key, 3))
    offset = 12*round(float(np.median(errors))/12)
    return (offset if all(abs(v-offset)<=tolerance for v in errors) else None), errors


def pitch_probes(plugin, role, buffer):
    # The pad chord measures its colour. Isolated notes disambiguate its voicing and octave.
    return [(audition(plugin, [k], .5, buffer), k) for k in PROBES[role][:2]]


def probe_pitch(plugin, role, buffer, tolerance):
    offset, _ = offset_from_probes(pitch_probes(plugin, role, buffer), tolerance)
    return offset


def describe(waves, gate):
    active = np.concatenate([w[:round(gate*RATE)] for w in waves])
    energy = np.mean(active**2, axis=1)
    hz, power = signal.welch(active, RATE, nperseg=4096, axis=0)
    power = power.mean(axis=1).astype(float)
    total = max(power.sum(), 1e-20)
    step = RATE//200
    envelopes = [np.sqrt(np.mean(w[:len(w)//step*step].reshape(-1,step,2)**2, axis=(1,2))) for w in waves]
    attack, sustain, tail = [], [], []
    for env in envelopes:
        gate_frame = round(gate*RATE/step)
        peak = max(env[:gate_frame].max(), 1e-9)
        above = np.flatnonzero(env[:gate_frame] >= peak*.8)
        attack.append(above[0]*step/RATE*1000 if above.size else 0.)
        sustain.append(env[round(.3*RATE/step):gate_frame].mean()/peak)
        sounding = np.flatnonzero(env[gate_frame:] > peak*.01)
        tail.append((sounding[-1]+1)*step/RATE if sounding.size else 0.)
    l, r = active.T.astype(float)
    features = dict(rms_db=10*np.log10(max(energy.mean(),1e-20)), peak=np.abs(active).max(),
                    centroid_hz=hz@power/total, low_fraction=power[hz<120].sum()/total,
                    lr_correlation=np.dot(l,r)/max(np.linalg.norm(l)*np.linalg.norm(r),1e-20),
                    flatness=np.exp(np.mean(np.log(np.maximum(power,1e-20))))/max(power.mean(),1e-20),
                    attack_ms=np.median(attack), sustain_ratio=np.median(sustain), release_seconds=np.median(tail))
    return {k:round(float(v),5) for k,v in features.items()}


def exclusion(role, f, offset, rules):
    if f['rms_db'] < rules['silent_rms_db']:
        return 'silent'
    if f['flatness'] > rules['max_flatness'] or offset is None:
        return 'atonal_or_pitch'
    eligible = {'bass': f['low_fraction'] >= rules['bass_low_min'],
                'pluck': f['sustain_ratio'] <= rules['pluck_sustain_max'],
                'pad': f['low_fraction'] <= rules['pad_low_max'] and
                       (f['attack_ms'] >= rules['pad_attack_ms'] or f['sustain_ratio'] >= rules['pad_sustain_min']),
                'lead': f['low_fraction'] <= rules['lead_low_max'] and f['lr_correlation'] >= rules['lead_corr_min']}
    return None if eligible[role] else 'role'


def vet_one(row, *, root, path, host_version, opts, directory):
    role = preset_role(row)
    record = dict(id=row['id'], role=role, rel_path=row['rel_path'])
    preset = str(root/row['rel_path'])
    # Vital compares major.minor and returns before applying any settings for newer banks.
    # Avoid reconstructing the host thousands of times for this known native rejection offline.
    try:
        patch_version = read_json(preset)['synth_version']
    except (OSError, ValueError, KeyError):
        return dict(record, reason_excluded='unreadable')
    if tuple(map(int,patch_version.split('.')[:2])) > host_version:
        return dict(record, reason_excluded='newer_feature_version')
    plugin = load_patch(path, preset, opts['vst_random_phase'])
    if plugin is None:
        return dict(record, reason_excluded='load')
    buffer = opts['vst_buffer_size']
    probes = pitch_probes(plugin, role, buffer)
    tolerance = opts['preset_vet']['pitch_tolerance_semitones']
    offset, errors = offset_from_probes(probes, tolerance)
    if offset:
        # Recheck actual sounding pitch when sending the compensated MIDI keys.
        compensated = [(audition(plugin,[k-offset],.5,buffer), k) for _,k in probes]
        corrected, _ = offset_from_probes(compensated, tolerance)
        if corrected != 0:
            offset = None
    gate = 2. if role=='pad' else .5
    waves = [audition(plugin, PROBES[role], gate, buffer)] if role=='pad' else [w for w,_ in probes]
    features = describe(waves, gate)
    record.update(pitch_offset=offset, features=dict(features, f0_error_semitones=errors))
    reason = exclusion(role, features, offset, opts['preset_vet'])
    if reason:
        record['reason_excluded'] = reason
    dest = directory/role/(row['id']+'.flac')
    dest.parent.mkdir(exist_ok=True, parents=True)
    audio = np.concatenate(waves)
    sf.write(dest, audio / max(1., float(np.abs(audio).max())), RATE, subtype='PCM_16')
    return record


def vet_presets(output, index=None, probes=None, options=None, workers=4):
    opts = studio_options(options or {})
    index = Path(index or opts.get('preset_index') or ASSETS/'vital-presets-library/preset-index.json').resolve()
    directory = Path(probes or output.parent/'vital-probes')
    path = vst_path(opts.get('vital_vst3'))
    version = component(instrument(path)[1])['synth_version']
    rows = [r for r in read_json(index) if preset_role(r)]
    task = partial(vet_one, root=index.parent, path=path, host_version=tuple(map(int, version.split('.')[:2])),
                   opts=opts, directory=directory)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(task, rows, chunksize=8))
    output.parent.mkdir(parents=True, exist_ok=True)
    write_bank(output,version,opts['preset_vet'],records)


def write_bank(path, version, rules, records):
    # One measured preset per line keeps revisions to the bank reviewable.
    with path.open('w',encoding='utf8',newline='\n') as out:
        out.write('{\n  "host_version": '+json.dumps(version)+',\n  "rules": '+json.dumps(rules)+',\n  "presets": [\n')
        out.write(',\n'.join('    '+json.dumps(r,ensure_ascii=True,separators=(',',':')) for r in records))
        out.write('\n  ]\n}\n')
