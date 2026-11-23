"""Render the very same tick96 performances that supervise the MIDI decoder."""
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np
from scipy import signal

from .events import Event
from .loudness import loudness_targets
from .studio_mix import RATE, SLOTS, DRUM_ROLES, balance, choice, draw, draw_int, master, spatial_mix
from .studio_sources import (PACKAGE, select_kit, load_bank, choose_pluck, read_json,
                             render_vital, render_voice, transposed, _render_hybrid_acoustic_role_layer)


@dataclass
class Note:
    start: int
    end: int
    role: str
    pitch: int
    velocity: float
    articulation: str = ''
    lyric: str = ''


@dataclass
class Song:
    wave: np.ndarray
    events: list
    chart: list
    metadata: dict
    stems: dict


def solo_windows(bars, grid, fraction, opts, rng, vocal, sections=None):
    budget = grid.second(bars[-1][1])*fraction
    roles = {r:w for r,w in opts['solo_roles'].items() if (vocal or r!='vocal') and w>0}
    sections = sections or ['']*len(bars)
    starts = [i for i,sec in enumerate(sections) if i%opts['solo_phrase_bars']==0 or sec!=sections[i-1]]
    rng.shuffle(starts)
    windows = []
    for i in starts:
        section = sections[i] or ''
        if not roles or any(section.startswith(s) for s in opts['solo_exclude_sections']):
            continue
        count = draw_int(rng,opts['solo_window_bars'])
        if i+count>len(bars) or any(s!=sections[i] for s in sections[i:i+count]):
            continue
        a,b = bars[i][0],bars[i+count-1][1]
        seconds = grid.second(b)-grid.second(a)
        if seconds<=budget and not any(a<y and b>x for x,y,_ in windows):
            windows.append((a,b,choice(rng,roles)))
            budget -= seconds
    return sorted(windows)


def pad_voicing(chord, previous, count, limits):
    low,high = limits
    classes = {p%12 for p in chord}
    candidates = [k for k in range(low,high+1) if k%12 in classes]
    voicings = [v for v in combinations(candidates,count) if len({p%12 for p in v})==count]
    center = (low+high)/2
    target = previous or tuple(center + 4*(i-(count-1)/2) for i in range(count))
    return min(voicings, key=lambda v:sum(abs(a-b) for a,b in zip(v,target)), default=())


def arrange_tracks(rows, bars, grid, opts, rng, harmony):
    vocal = rng.random() < opts['vocal_probability']
    pad = rng.random() < opts['pad_probability']
    sections = harmony.get('sections') or ['']*len(bars)
    starts = [a for a,_ in bars]
    fraction = draw(rng,opts['solo_fraction'])
    windows = solo_windows(bars,grid,fraction,opts,rng,vocal,sections)
    names = (*DRUM_ROLES, 'vocal' if vocal else 'lead', 'bass', 'pluck','lead')
    notes = [Note(start,stop,names[slot],pitch,vel,*mark) for start,stop,slot,pitch,vel,*mark in rows]
    count = draw_int(rng,opts['pad_voices'])
    accompanied = {starts[bisect_right(starts,n.start)-1] for n in notes if n.role=='pluck' and not n.articulation}
    previous = ()
    split = Counter()
    if vocal:
        sung = {(n.start,n.pitch) for n in notes if n.role=='vocal'}
        arranged = []
        for n in notes:
            if n.role=='lead' and (n.start,n.pitch) in sung:
                action = choice(rng,opts['lead_vocal_split'])
                split[action] += 1
                if action=='drop':
                    continue
                if action=='octave_up':
                    n.pitch += 12
                else:
                    bar = bisect_right(starts,n.start)-1
                    tonic = harmony['root_base']+harmony['key_shifts'][bar]
                    scale = list(harmony['scale'])
                    below = [p for p in range(n.pitch-24,n.pitch) if (p-tonic)%12 in scale]
                    n.pitch = below[-rng.choice(opts['lead_harmony_degrees'])]
                if (n.start,n.pitch) in sung:
                    continue
                n.articulation = 'lead_'+action
            arranged.append(n)
        notes = arranged
    ghosts = []
    for n in notes:
        bar = bisect_right(starts,n.start)-1
        if (SLOTS[n.role] not in opts['ghost_slots'] or n.role not in ('snare','hat') or
                not any(s in (sections[bar] or '') for s in opts['ghost_sections']) or
                rng.random()>=opts['ghost_probability']):
            continue
        offsets = opts['ghost_offsets_ticks']
        t = n.start+rng.choice(offsets[n.role] if isinstance(offsets,dict) else offsets)
        if t<bars[bar][1]:
            ghosts.append(Note(t,min(t+8,bars[bar][1]),n.role,n.pitch,draw(rng,opts['ghost_velocity']),'ghost'))
    notes.extend(ghosts)
    for i,(a,b) in enumerate(bars):
        if pad and a in accompanied and a in harmony['chords']:
            previous = pad_voicing(harmony['chords'][a],previous,count,opts['pad_pitch_range'])
            notes.extend(Note(a,b,'pad',p,draw(rng,opts['pad_velocity'])) for p in previous)
        if (i+1)%opts['fx_every_bars']==0 and rng.random()<opts['fx_probability']:
            kind = choice(rng,opts['fx_types'])
            length = round(rng.choice(opts['fx_beats'])*96)
            start = a if kind == 'impact' else max(a,b-length)
            stop = min(b,start+length)
            if kind=='impact':
                limit = grid.second(start)+opts['impact_max_seconds']
                cap = grid.tick(limit)
                stop = min(stop,cap-int(grid.second(cap)>limit))
            if stop>start:
                notes.append(Note(start,stop,'fx',49,draw(rng,opts['fx_velocity']),kind))
    kept = []
    for a,b,role in windows:
        present = {n.role for n in notes if a<=n.start<b}
        drum_roles = {'kick','snare','hat'}
        available = {r:w for r,w in opts['solo_roles'].items() if w>0 and
                     (r in present or r=='drums' and drum_roles & present)}
        if not available:
            continue
        if role not in available:
            role = choice(rng,available)
        keep = drum_roles if role=='drums' else {role}
        notes = [n for n in notes if not (a<=n.start<b and n.role not in keep)]
        for n in notes:
            if n.start<a<n.end and n.role not in keep:
                n.end = a
        kept.append((a,b,role))
    windows = kept
    # The MIDI vocabulary has one voice per (slot,pitch) and one simultaneous drum per slot.
    # Resolve unisons/re-attacks before a single sample or MIDI message is rendered.
    unique = {}
    for n in notes:
        slot = SLOTS[n.role]
        key = (n.start,slot,n.pitch if slot>=4 else -1)
        old = unique.get(key)
        if old is None or (n.role=='fx', n.velocity) > (old.role=='fx',old.velocity):
            unique[key]=n
    notes = sorted(unique.values(), key=lambda n:(n.start,SLOTS[n.role],n.pitch))
    following = {}
    for n in reversed(notes):
        key = (SLOTS[n.role], n.pitch) if n.role!='vocal' else ('vocal',)
        if key in following:
            n.end = min(n.end,following[key])
        following[key]=n.start
        if n.role=='vocal':
            n.lyric = rng.choice(opts['vocal_syllables'])
    notes = [n for n in notes if n.end>n.start]
    return notes, dict(solo_windows=windows, solo_requested_fraction=fraction, lead_vocal_actions=dict(split),
                       solo_fraction=sum(grid.second(b)-grid.second(a) for a,b,_ in windows)
                       /grid.second(bars[-1][1]), vocal_enabled=vocal, pad_enabled=pad)


def add_clip(stem, wave, at, gain):
    count = min(len(wave), len(stem)-at)
    if count>0:
        stem[at:at+count] += wave[:count] * gain


def fx_sound(kind, count, crash, rng, opts):
    t = np.arange(count, dtype=np.float32)/RATE
    phase = np.arange(count,dtype=np.float32)/max(1,count-1)
    seed = rng.getrandbits(64)
    rand = np.random.default_rng(seed)
    noise = rand.standard_normal((count,2),dtype=np.float32)
    common = draw(rng,opts['fx_noise_correlation'])
    noise *= math.sqrt(1-common**2)
    noise += common*rand.standard_normal((count,1),dtype=np.float32)
    # A bank of rising band-pass filters makes a true frequency sweep at fixed cost.
    if kind=='reverse_cymbal' and crash is not None:
        src = crash[::-1]
        if len(src) != count:
            import soxr
            src = soxr.resample(src,len(src),count,quality='HQ')
        result = src[:count]
    elif kind=='impact':
        low = np.sin(2*np.pi*(48*t+2*(1-np.exp(-30*t))))*np.exp(-5*phase)
        if opts['impact_hp_hz']:
            low = signal.sosfilt(signal.butter(2,opts['impact_hp_hz'],'highpass',fs=RATE,output='sos'),low)
        result = .40*noise*np.exp(-8*phase)[:,None]+low[:,None]
    else:
        result = np.zeros_like(noise)
        for j,f in enumerate((250,720,2000,5800,12500)):
            band = signal.iirpeak(f, .9, fs=RATE)
            weight = np.maximum(0,1-np.abs(phase*4-j))
            result += signal.lfilter(*band,noise,axis=0)*weight[:,None]
        if kind=='riser':
            result += .3*signal.chirp(t,110,max(t[-1],1/RATE),1760,method='logarithmic')[:,None]
        if kind=='filter_build':
            saw_stack = np.zeros_like(t)
            for f_hz in (130.81, 196.00, 261.63, 329.63, 392.00):
                saw_stack += signal.sawtooth(2*np.pi*f_hz*t) * 0.14
            sweep_out = np.zeros_like(t)
            n_seg = 12
            seg_len = max(1, count // n_seg)
            for si in range(n_seg):
                sa = si * seg_len
                sb = count if si == n_seg - 1 else (si + 1) * seg_len
                u = (si + 0.5) / n_seg
                fc = 260.0 * ((4800.0 / 260.0) ** (u ** 1.6))
                sos_bp = signal.butter(2, [max(80.0, fc * 0.72), min(16000.0, fc * 1.38)], 'bandpass', fs=RATE, output='sos')
                sweep_out[sa:sb] = np.tanh(signal.sosfilt(sos_bp, saw_stack[sa:sb]) * 2.2) * (0.28 + 0.72 * u)
            result = result * (.45 + .55*np.sin(2*np.pi*(phase*4+phase**2*8))**2)[:,None] + 0.48 * sweep_out[:,None]
        result *= phase[:,None]
    return result.astype(np.float32,copy=False), dict(seed=seed,common=common)


def render(notes, grid, end, meta, chart, opts, rng, *, keep_stems=False):
    root, bank = load_bank(opts)
    tracks = defaultdict(list)
    for n in notes:
        tracks[n.role].append(n)
    required = [r for r in DRUM_ROLES if r in tracks or
                (r=='crash' and any(n.articulation=='reverse_cymbal' for n in tracks.get('fx',())))]
    if 'hat' in required and opts['open_hat_probability']>0:
        required.append('open_hat')
    kits, meta['drum_kit'] = select_kit(bank,required,root,opts,rng)
    # A source may supply no usable cymbal. Remove that voice before rendering and emitting labels.
    notes = [n for n in notes if (n.role not in DRUM_ROLES or n.role in kits)
             and (n.articulation!='reverse_cymbal' or 'crash' in kits)]
    tracks = defaultdict(list)
    for n in notes:
        tracks[n.role].append(n)
    frames = max(1, math.ceil((grid.second(end)+opts['tail_seconds'])*RATE))
    hits = {}
    fade_frames = round(opts['drum_choke_fade_ms']*RATE/1000)
    for role in DRUM_ROLES:
        group = tracks.get(role,())
        for i,n in enumerate(group):
            kit_role = role
            if (role=='hat' and 'open_hat' in kits and n.start%96==48 and not n.articulation and
                    rng.random()<opts['open_hat_probability']):
                kit_role = n.articulation = 'open_hat'
            wave, gain = kits[kit_role].hit(n.velocity)
            start = round(grid.second(n.start)*RATE)
            stop = start + len(wave)
            fade_at = stop
            if role in opts['drum_choke_roles'] and i+1<len(group):
                fade_at = min(stop,round(grid.second(group[i+1].start)*RATE))
                stop = min(stop,fade_at+fade_frames)
            # Solo cuts apply to sounding one-shots as well as the pitched score.
            for a,b,solo in meta.get('solo_windows', ()):
                if n.start<a and role!=solo and not (solo=='drums' and role in ('kick','snare','hat')):
                    cut = round(grid.second(a)*RATE)
                    if cut<stop:
                        stop = cut
                        fade_at = min(fade_at,max(start,cut-fade_frames))
            hits[id(n)] = (wave[:stop-start],gain,fade_at-start)
            frames = max(frames,stop)
            n.end = max(n.start+1,grid.tick(stop/RATE))
    if not tracks:
        return Song(np.zeros((frames,2),np.float32), [], chart, meta, {})
    stems, instruments = {}, {}
    for role, group in tracks.items():
        if role in ('bass','lead','pad'):
            wave, instruments[role] = render_vital(role, group, grid, frames, opts, rng)
        elif role=='vocal':
            wave, instruments[role] = render_voice(group, grid, frames, opts, rng)
        else:
            wave = np.zeros((frames,2),dtype=np.float32)
            if role in DRUM_ROLES:
                kit = kits[role]
                instruments[role] = kit.paths
                for n in group:
                    src, gain, fade_at = hits[id(n)]
                    if fade_at<len(src):
                        src = src.copy()
                        src[fade_at:] *= np.linspace(1,0,len(src)-fade_at,dtype=np.float32)[:,None]
                    add_clip(wave,src,round(grid.second(n.start)*RATE),gain)
            elif role=='pluck':
                selected = choose_pluck(bank,group,opts,rng) if rng.random()<opts['sample_probability'] else None
                pieces = []
                if selected:
                    source,radius,pool = selected
                    for n in group:
                        sample = min(pool[n.pitch],key=lambda s:abs(s.get('velocity',n.velocity)-n.velocity))
                        src = transposed(root,sample.get('collection'),sample['path'],sample.get('start',0.),
                                         sample.get('length'),n.pitch-sample['pitch'],opts['sample_onset_db'])
                        if not len(src):
                            pieces.clear()
                            break
                        pieces.append(src)
                if not pieces:
                    wave,preset = render_vital(role,group,grid,frames,opts,rng)
                    instruments[role] = {'vital':preset}
                else:
                    instruments[role] = {'multisample':source[-1]}
                    meta['pluck_sample_source'] = dict(collection=source[0],song=source[1],max_transpose=radius)
                    for n,src in zip(group,pieces):
                        start = grid.second(n.start)
                        gate = max(1,round((grid.second(n.end)-start)*RATE))
                        if len(src)<gate:
                            n.end = max(n.start+1,min(n.end,grid.tick(start+len(src)/RATE)))
                        piece = src[:gate+round(opts['pluck_release_seconds']*RATE)]
                        if len(piece)>gate:
                            piece = piece.copy()
                            piece[gate:] *= np.linspace(1,0,len(piece)-gate,dtype=np.float32)[:,None]
                        add_clip(wave,piece,round(start*RATE),n.velocity)
                    aco_w, aco_fams = _render_hybrid_acoustic_role_layer('pluck', group, grid, frames, opts)
                    if aco_w is not None:
                        aco_dom = bool(opts.get('acoustic_dominant', False))
                        wave = (wave * (0.45 if aco_dom else 0.72) + aco_w * (0.68 if aco_dom else 0.35)).astype(np.float32)
                        instruments[role]['acoustic_families'] = list(aco_fams)
            else:
                for n in group:
                    count = max(1, round((grid.second(n.end)-grid.second(n.start))*RATE))
                    crash = kits['crash'].hit(n.velocity)[0] if n.articulation=='reverse_cymbal' else None
                    fx,setting = fx_sound(n.articulation,count,crash,rng,opts)
                    meta.setdefault('fx_settings',[]).append(dict(tick=n.start,kind=n.articulation,**setting))
                    add_clip(wave,fx,round(grid.second(n.start)*RATE),n.velocity)
        stems[role] = wave
    profile = read_json(opts.get('studio_profile') or PACKAGE/'studio-profile.json')
    meta['preset_load_failures'] = {}
    for role,item in instruments.items():
        if isinstance(item,dict):
            patch = item.get('vital',item)
            if isinstance(patch,dict) and patch.get('preset_load_failures'):
                meta['preset_load_failures'][role] = patch['preset_load_failures']
    meta.update(studio_instruments=instruments, **spatial_mix(stems,tracks,grid,opts,rng))
    mix, levels, faders = balance(stems,tracks,grid,opts,rng,profile)
    wave, mastering = master(mix,opts,rng,profile)
    meta.update(studio_balance=faders, studio_master=mastering)
    events = []
    for n,(band,db) in zip(notes,loudness_targets([levels[id(n)] for n in notes])):
        sec = grid.second(n.start)
        slot = SLOTS[n.role]
        data = dict(slot=slot, start_tick=n.start, end_tick=n.end, duration=grid.second(n.end)-sec,
                    velocity_bin=band, velocity_db=db, performance_velocity=n.velocity, role=n.role)
        if slot>=4:
            data['pitch']=n.pitch
        if n.articulation:
            data['articulation']=n.articulation
        if n.lyric:
            data['lyric']=n.lyric
        events.append(Event(sec,'note' if slot>=4 else 'unpitched',data))
    exports = {}
    if keep_stems:
        exports = {role:stem * np.float32(10**(faders['fader_db'][role]/20)) for role,stem in stems.items()}
    return Song(wave, events, chart, meta, exports)
