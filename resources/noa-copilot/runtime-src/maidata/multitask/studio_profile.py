"""Offline per-song BMS fader observations and maimai reference spectra."""
from collections import defaultdict
from pathlib import Path
import random
from tempfile import TemporaryDirectory

from .data import read_manifest, excluded_song
import numpy as np
import soundfile as sf

from .loudness import integrated_loudness
from .studio_mix import spectrum
from .config import studio_options


def profiles(*, bms=(), bms_manifest=None, reference_manifest=None, references=(), reference_root=None,
             base_profile=None, limit=None, seed=0, options=None):
    from .bms import convert_bms
    opts = studio_options(options or {})
    rng = random.Random(seed)
    def selected(items):
        items = list(items)
        return rng.sample(items, limit) if limit is not None and len(items) > limit else items
    rows = [(Path(p).name, p, None) for p in bms]
    if bms_manifest:
        for r in read_manifest(bms_manifest)[0]:
            m = r.get('metadata', {})
            if r.get('split', 'train')=='train' and (r.get('source_family')=='bms' or r.get('dataset')=='bms'):
                rows.append((r['id'], r.get('bms_source') or m.get('selected_script'), m.get('balance_db')))
    balance = []
    with TemporaryDirectory() as tmp:
        for key, source, stats in selected(rows):
            if stats is None and source:
                _, m = convert_bms(source, Path(tmp)/'mix.flac', balance_options=opts)
                stats = m['balance_db']
                key = m.get('title') or key
            if stats:
                balance.append(dict(id=key, db={k:round(v,3) for k,v in stats.items()}))
    balance_keys = ('balance_window_seconds','balance_active_range_db','balance_percentile')
    measurement = {k:opts[k] for k in balance_keys}
    if not rows and base_profile:
        from .studio_sources import read_json
        base = read_json(base_profile)
        balance = base.get('balance',[])
        measurement.update({k:base['measurement'][k] for k in balance_keys if k in base.get('measurement',{})})
    measurement['reference_eq_smoothing_oct'] = opts['reference_eq_smoothing_oct']
    audio = {}
    def add(path, song_id, title='', version=''):
        if not excluded_song(dict(song_id=song_id,title=title)):
            audio.setdefault(str(path),dict(id=str(song_id),version=version))
    for path in references:
        add(path,Path(path).parent.name,str(path))
    if reference_root:
        for chart in sorted(Path(reference_root).glob('*/maidata.txt')):
            fields = {}
            with chart.open(encoding='utf-8-sig',errors='replace') as stream:
                for line in stream:
                    if line.startswith('&inote'):
                        break
                    if line.startswith('&') and '=' in line:
                        k,v=line[1:].strip().split('=',1)
                        if k in ('title','version','shortid'):
                            fields[k]=v
                    if len(fields)==3:
                        break
            path=chart.parent/'track.mp3'
            if path.is_file():
                add(path,fields.get('shortid',chart.parent.name),fields.get('title',''),fields.get('version',''))
    if reference_manifest:
        for r in read_manifest(reference_manifest)[0]:
            m=r.get('metadata',{})
            if (r['task']=='maimai' and r.get('split','train')=='train' and r.get('audio')
                    and not r.get('pseudo',m.get('pseudo',False)) and r.get('source_family') not in ('vita','studio')):
                add(r['audio'],r.get('song_id') or m.get('song_id') or r['id'],r.get('title',m.get('title','')),
                    r.get('version',m.get('version','')))
    # Sample releases in rotation so a large recent cabinet version cannot dominate the reference.
    groups = defaultdict(list)
    for path,row in audio.items():
        groups[row['version']].append(path)
    pools = list(groups.values())
    rng.shuffle(pools)
    for pool in pools:
        rng.shuffle(pool)
    paths = []
    while pools and (limit is None or len(paths)<limit):
        for pool in pools:
            if limit is not None and len(paths)>=limit:
                break
            paths.append(pool.pop())
        pools = [pool for pool in pools if pool]
    refs, loudness = [], []
    for path in paths:
        wave,rate = sf.read(path,dtype='float32',always_2d=True)
        envelope = spectrum(wave,rate,opts['reference_eq_smoothing_oct'])
        level = integrated_loudness(wave,rate)
        if np.isfinite(level):
            peak_db = 20*np.log10(max(float(np.abs(wave).max()),1e-10))
            loudness.append(dict(audio[path],lufs=round(level,3),plr=round(peak_db-level,3)))
            refs.append(dict(audio[path],spectrum=envelope.round(3).tolist()))
    mean = np.mean([row['spectrum'] for row in refs],axis=0).round(3).tolist() if refs else []
    return dict(measurement=measurement, balance=balance, references=refs,
                loudness=loudness,reference_mean_spectrum=mean)
