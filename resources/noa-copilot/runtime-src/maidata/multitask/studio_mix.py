"""Stereo studio mix and the same K-weighted balance statistic for BMS and synthesis."""
from functools import lru_cache
import math
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
from scipy import ndimage, signal
import soundfile as sf

from .loudness import integrated_loudness, k_weight

RATE = 44100
# One smoothed spectral envelope per reference, independent of track length.
FREQUENCIES = np.geomspace(40, 20000, 33)
DRUM_ROLES = ('kick', 'snare', 'hat', 'crash')
SLOTS = {name: i for i, name in enumerate((*DRUM_ROLES, 'vocal', 'bass', 'pluck', 'lead'))}
SLOTS.update(pad=7, fx=3)


def draw(rng, value):
    return rng.uniform(*value) if isinstance(value, (list, tuple)) else value


def draw_int(rng, value):
    return rng.randint(*value) if isinstance(value,(list,tuple)) else int(value)


def choice(rng, values):
    if isinstance(values, dict):
        return rng.choices(list(values), weights=list(values.values()), k=1)[0]
    return rng.choice(values)


def db(power):
    return 10 * np.log10(np.maximum(power, 1e-20))


def sample_power(wave, rate):
    weighted = k_weight(wave, rate)
    return np.einsum('ij,ij->i', weighted, weighted) if weighted.ndim == 2 else weighted * weighted


def block_power(power, size):
    starts = np.arange(0, len(power), size)
    return np.add.reduceat(power, starts) / np.minimum(size, len(power) - starts)


def relative_level(stem, mix, opts):
    # Use simultaneous windows. Long rests and a silent track must not determine the fader.
    active = (stem > 1e-16) & (stem >= stem.max() * 10 ** (-opts['balance_active_range_db'] / 10))
    if not active.any():
        return None
    return float(np.percentile(db(stem[active]) - db(mix[active]), opts['balance_percentile']))


def measured_balance(stems, mix, rate, opts):
    n = max(1, round(rate * opts['balance_window_seconds']))
    total = block_power(sample_power(mix, rate), n)
    return {str(slot): level for slot, wave in stems.items()
            if (level := relative_level(block_power(sample_power(wave, rate), n), total, opts)) is not None}


def balance(stems, notes, grid, opts, rng, profile):
    n = max(1, round(RATE * opts['balance_window_seconds']))
    windows, levels = {}, {}
    for role, stem in stems.items():
        power = sample_power(stem, RATE)
        windows[role] = block_power(power, n)
        integral = np.empty(len(power) + 1)
        integral[0] = 0
        np.cumsum(power, out=integral[1:])
        pitched = SLOTS[role] >= 4
        cap = opts['pitched_velocity_window_seconds' if pitched else 'drum_velocity_window_seconds']
        for note in notes[role]:
            sec = grid.second(note.start)
            a = round(sec * RATE)
            z = max(a + 1, min(len(stem), round(min(grid.second(note.end), sec + cap) * RATE)))
            levels[id(note)] = float(-0.691 + db((integral[z] - integral[a]) / (z - a)))
    rows = profile.get('balance', ())
    row = rng.choice(rows) if rows and rng.random()<opts['balance_profile_probability'] else {}
    targets = {}
    for role in stems:
        key = 'lead_with_vocal' if role=='lead' and 'vocal' in stems else role
        low,high = opts['role_balance_db'][key]
        observed = row.get('db',{}).get(str(SLOTS[role]))
        if observed is None:
            targets[role] = rng.uniform(low,high)+draw(rng,opts['balance_jitter_db'])
        else:
            targets[role] = float(np.clip(observed,low,high))
    if 'kick' in targets:
        targets['kick'] = min(targets['kick'],opts['kick_max_db'])
    if 'vocal' in targets:
        for role in ('lead','pluck'):
            if role in targets:
                targets[role] = min(targets[role],targets['vocal']-opts['vocal_priority_db'])
    gains = {role: 0. for role in stems}
    achieved = {}
    # Only the mix changes in these iterations. Stem measurements and all instruments are reused.
    for i in range(opts['balance_iterations'] + 1):
        mix = sum(stems[role] * np.float32(10 ** (gains[role] / 20)) for role in stems)
        mix_energy = block_power(sample_power(mix, RATE), n)
        for role, energy in windows.items():
            level = relative_level(energy, mix_energy, opts)
            if level is not None:
                achieved[role] = level + gains[role]
        if i == 0:
            initial = dict(achieved)
        if i < opts['balance_iterations'] and achieved:
            delta = {r: targets[r] - measured for r, measured in achieved.items()}
            # Relative targets have arbitrary common gain. Anchor it for stable headroom.
            common = np.mean(list(delta.values()))
            for role in delta:
                gains[role] += delta[role] - common
    for role, group in notes.items():
        for note in group:
            levels[id(note)] += gains[role]
    return mix, levels, dict(balance_mode='profile_clamped' if row else 'range', balance_source=row.get('id'), target_db=targets, before_db=initial, achieved_db=achieved,
                             fader_db=gains)


def spectrum(wave, rate, smoothing_oct=.5):
    # Uniformly spaced, constant-cost Welch frames cover the whole track.
    x = wave[:, None] if wave.ndim == 1 else wave
    size = 4096
    if len(x) < size:
        x = np.pad(x, ((0, size - len(x)), (0, 0)))
    starts = np.linspace(0, len(x) - size, min(128, 1 + (len(x)-size)//(size//2)), dtype=int)
    frames = x[starts[:, None] + np.arange(size)] * np.hanning(size)[None, :, None]
    power = np.mean(np.abs(np.fft.rfft(frames, axis=1)) ** 2, axis=(0, 2))
    hz = np.fft.rfftfreq(size, 1 / rate)
    # Smooth energy on the same logarithmic frequency grid as the references.
    offsets = np.log2(np.maximum(hz, 1)[None, :] / FREQUENCIES[:, None])
    weights = np.exp(-0.5 * (offsets / smoothing_oct) ** 2)
    envelope = db((weights @ power) / weights.sum(axis=1))
    envelope -= np.median(envelope[(FREQUENCIES >= 150) & (FREQUENCIES <= 5500)])
    return envelope


@lru_cache(maxsize=32)
def reference_spectrum(path, smoothing_oct=.5):
    audio, sr = sf.read(path, dtype='float32', always_2d=True)
    return spectrum(audio, sr, smoothing_oct).round(3).tolist()


def reference_eq(wave, reference, opts, rng):
    source = spectrum(wave, RATE, opts['reference_eq_smoothing_oct'])
    limit = opts['reference_eq_limit_db']
    diff = np.asarray(reference) - source
    diff -= np.median(diff[(FREQUENCIES>=150)&(FREQUENCIES<=5500)])
    gain = np.clip(diff * draw(rng,opts['reference_eq_strength']),-limit,limit)
    taps = signal.firwin2(int(opts['eq_taps']) | 1, np.r_[0, FREQUENCIES, RATE/2],
                          10 ** (np.r_[gain[0], gain, gain[-1]] / 20), fs=RATE).astype(np.float32)
    # Center the linear-phase filter so MIDI onsets are unchanged.
    wet = signal.oaconvolve(wave, taps[:, None], mode='same', axes=0)
    return wet, gain.round(3).tolist()


def read_stereo(path, start=0., length=None):
    with sf.SoundFile(path) as f:
        rate = f.samplerate
        f.seek(min(len(f), round(start * rate)))
        wave = f.read(-1 if length is None else round(length * rate), dtype='float32', always_2d=True)
    if rate != RATE:
        import soxr
        wave = soxr.resample(wave, rate, RATE)
    return np.repeat(wave, 2, axis=1) if wave.shape[1] == 1 else wave[:, :2]


def filtered(wave, hz, kind, order=2):
    sos = signal.butter(order,hz,kind,fs=RATE,output='sos').astype(np.float32)
    return signal.sosfilt(sos,wave,axis=0)


def impulse(rng, paths, decay_seconds, opts):
    if paths:
        path = rng.choice(paths)
        ir = read_stereo(path)
        info = dict(source=str(path))
    else:
        duration = draw(rng,decay_seconds)
        seed = rng.getrandbits(64)
        rand = np.random.default_rng(seed)
        time = np.arange(max(2,round(RATE*duration)),dtype=np.float32)/RATE
        ir = np.zeros((len(time),2),np.float32)
        edges = [0,*opts['ir_band_edges_hz'],RATE/2]
        for low,high,factor in zip(edges,edges[1:],opts['ir_rt60_factors']):
            noise = rand.standard_normal(ir.shape,dtype=np.float32)
            if low:
                noise = filtered(noise,low,'highpass')
            if high<RATE/2:
                noise = filtered(noise,high,'lowpass')
            ir += noise * (.08*np.exp(-6.908*time/(duration*factor)))[:,None]
        early = np.zeros_like(ir)
        offsets = []
        for _ in range(draw_int(rng,opts['ir_early_reflections'])):
            ms = draw(rng,opts['ir_early_window_ms'])
            offsets.append(ms)
            early[min(len(ir)-1,round(ms*RATE/1000))] += rand.uniform(-1,1,2)
        cutoff = draw(rng,opts['ir_lowpass_hz'])
        ir = filtered(ir+filtered(early,opts['ir_early_lowpass_hz'],'lowpass'),cutoff,'lowpass')
        # Orthogonalise the diffuse pair before applying the sampled L/R correlation.
        left,right = ir.T
        right -= left*(np.dot(left,right)/max(np.dot(left,left),1e-12))
        right *= np.linalg.norm(left)/max(np.linalg.norm(right),1e-10)
        corr = draw(rng,opts['reverb_lr_correlation'])
        right[:] = corr*left + math.sqrt(1-corr*corr)*right
        predelay = draw(rng,opts['reverb_predelay_ms'])
        ir = np.pad(ir,((round(predelay*RATE/1000),0),(0,0)))
        info = dict(source='generated',seed=seed,rt60_seconds=duration,lowpass_hz=cutoff,
                    predelay_ms=predelay,correlation=corr,early_ms=offsets)
    energy = np.sqrt(np.sum(ir*ir,axis=0).mean())
    ir = ir/max(float(energy),1e-8)
    # Filtering the impulse once gives the same linear return EQ for every stem on this bus.
    for key,kind in [('reverb_return_hp','highpass'),('reverb_return_lp','lowpass')]:
        hz = draw(rng,opts[key])
        info[key] = hz
        if hz:
            ir = filtered(ir,hz,kind)
    return ir,info


def correlation(wave):
    left,right = wave.T
    ll = np.dot(left,left)
    rr = np.dot(right,right)
    return float(np.dot(left,right)/max(math.sqrt(float(ll)*float(rr)),1e-20))


def widen(wave, role, width, opts, rng):
    mid,side = (wave[:,0]+wave[:,1])*.5,(wave[:,0]-wave[:,1])*.5
    side *= width
    info = dict(width=width)
    if (role in opts['synthetic_width_roles'] and rng.random()<opts['synthetic_width_probability']
            and correlation(wave)>opts['synthetic_width_mono_correlation']):
        decorrelated = mid
        stages = []
        for _ in range(draw_int(rng,opts['allpass_stages'])):
            delay = max(1,round(draw(rng,opts['allpass_delay_ms'])*RATE/1000))
            feedback = draw(rng,opts['allpass_feedback'])
            # One first-order allpass per delay phase, O(samples) independent of delay length.
            x = np.pad(decorrelated,(0,(-len(mid))%delay)).reshape(-1,delay)
            decorrelated = signal.lfilter(np.array([feedback,1],np.float32),np.array([1,feedback],np.float32),
                                          x,axis=0).ravel()[:len(mid)]
            stages.append([delay,feedback])
        amount = draw(rng,opts['synthetic_width_amount'])
        side += amount*filtered(decorrelated,opts['width_highpass_hz'],'highpass')
        info.update(allpass=stages,synthetic_width_amount=amount)
    return np.column_stack((mid+side,mid-side)),info


def delayed(wave, frames, beats, grid):
    # Constant-tempo regions need only a fractional-sample shift. Refine the map
    # inside ramps, including the interval a delayed tap takes to cross a tempo change.
    last = grid.second_to_beat(frames/RATE)
    knots = list(grid.beats)
    for i,slope in enumerate(grid.slopes[:-1]):
        if slope:
            knots.extend(np.arange(grid.beats[i],grid.beats[i+1],1/16))
    boundaries = np.unique(np.clip([0., last, *knots, *(b+beats for b in knots)],0,last))
    target = np.array([grid.beat_to_second(b)*RATE for b in boundaries])
    source = np.array([grid.beat_to_second(b-beats)*RATE for b in boundaries])
    result = np.zeros_like(wave)
    for i in range(len(target)-1):
        a = max(0, math.ceil(target[i]-1e-8))
        b = min(frames, math.ceil(target[i+1]-1e-8))
        left = source[i]-target[i]
        if a>=b:
            continue
        if abs(source[i+1]-target[i+1]-left)<1e-6:
            offset = math.floor(left+1e-8)
            frac = max(0.,left-offset)
            a = max(a,-offset)
            b = min(b,frames-offset-1)
            if a<b:
                result[a:b] = ((1-frac)*wave[a+offset:b+offset] + frac*wave[a+offset+1:b+offset+1])
        else:
            pos = source[i]+(np.arange(a,b)-target[i])*(source[i+1]-source[i])/(target[i+1]-target[i])
            valid = (pos>=0)&(pos<frames-1)
            index = pos[valid].astype(np.intp)
            frac = (pos[valid]-index).astype(np.float32)[:,None]
            result[a:b][valid] = wave[index]+frac*(wave[index+1]-wave[index])
    return result


def role_filter(wave, role, opts, rng):
    eq = {band:draw(rng,hz) for band,hz in opts['role_eq'].get(role,{}).items() if hz}
    for band,cutoff in eq.items():
        if cutoff:
            sos = signal.butter(opts['role_eq_order'],cutoff,'highpass' if band=='hp' else 'lowpass',
                                fs=RATE,output='sos').astype(np.float32)
            wave = signal.sosfiltfilt(sos,wave,axis=0,padlen=min(len(wave)-1,3*(2*len(sos)+1)))
    return wave,eq


def multiband_ott(wave, mix_amount=0.44, hp_hz=215.0):
    """High-passed 3-band upward/downward OTT compression for bright, clean synth leads and chords."""
    if len(wave) < 64 or float(np.max(np.abs(wave))) < 1e-6:
        return wave
    work = filtered(wave, hp_hz, 'highpass', order=2) if hp_hz and hp_hz > 0 else wave
    b_low = filtered(work, 215.0, 'lowpass', order=2)
    b_mid = filtered(filtered(work, 215.0, 'highpass', order=2), 880.0, 'lowpass', order=2)
    b_hmid = filtered(filtered(work, 880.0, 'highpass', order=2), 3600.0, 'lowpass', order=2)
    b_high = filtered(work, 3600.0, 'highpass', order=2)
    win = max(1, round(0.012 * RATE))
    smooth_win = max(1, round(0.006 * RATE))
    wet = np.zeros_like(work, dtype=np.float32)
    for band, target_rms, gain_scale in (
        (b_low, 0.11, 0.90),
        (b_mid, 0.14, 0.86),
        (b_hmid, 0.13, 1.16),
        (b_high, 0.10, 1.22),
    ):
        env = np.sqrt(ndimage.uniform_filter1d(np.mean(band * band, axis=1), size=win, mode='nearest') + 1e-12)
        g_down = (target_rms / np.maximum(env, 1e-6)) ** 0.58
        g_up = np.minimum(2.6, (target_rms / np.maximum(env, 1e-6)) ** 0.36)
        g = np.where(env > target_rms, g_down, np.where(env > 0.0025, g_up, 1.0)).astype(np.float32)
        g = ndimage.uniform_filter1d(g, size=smooth_win, mode='nearest')
        wet += (np.tanh(band * g[:, None] * 1.15) / 1.15) * np.float32(gain_scale)
    return ((1.0 - mix_amount) * work + mix_amount * wet).astype(np.float32)


def synth_clarity_sculpt(wave):
    """Carve 180-480Hz mud and boost 1.45-7.2kHz articulation presence on pitched synths."""
    if len(wave) < 64 or float(np.max(np.abs(wave))) < 1e-6:
        return wave
    mud = filtered(filtered(wave, 180.0, 'highpass', order=2), 480.0, 'lowpass', order=2)
    pres = filtered(filtered(wave, 1450.0, 'highpass', order=2), 7200.0, 'lowpass', order=2)
    air = filtered(wave, 7800.0, 'highpass', order=2)
    return (wave - 0.22 * mud + 0.30 * pres + 0.14 * air).astype(np.float32)


def spatial_mix(stems, notes, grid, opts, rng):
    frames = len(next(iter(stems.values())))
    room, room_name = impulse(rng,opts.get('room_irs'),opts['room_seconds'],opts)
    hall, hall_name = impulse(rng,opts.get('hall_irs'),opts['hall_seconds'],opts)
    envelope = np.ones(frames, dtype=np.float32)
    length = max(1, round(opts['duck_release_seconds'] * RATE))
    duck = 1 - opts['duck_depth'] * np.exp(-4 * np.arange(length, dtype=np.float32) / length)
    for hit in notes.get('kick', ()):
        a = round(grid.second(hit.start) * RATE)
        region = envelope[a:a + length]
        np.minimum(region, duck[:len(region)], out=region)
    clarity_dsp = opts.get('clarity_dsp', True)
    settings = {}
    for role, wave in stems.items():
        wave, eq = role_filter(wave,role,opts,rng)
        if clarity_dsp:
            if role in ('lead', 'pluck', 'pad'):
                hp_hz = 230.0 if role == 'lead' else (210.0 if role == 'pluck' else 175.0)
                ott_mix = 0.48 if role == 'lead' else (0.38 if role == 'pluck' else 0.28)
                wave = multiband_ott(wave, mix_amount=ott_mix, hp_hz=hp_hz)
                wave = synth_clarity_sculpt(wave)
            elif role in ('hat', 'crash'):
                wave = (0.55 * wave + 0.45 * filtered(wave, 5200.0, 'lowpass', order=2)).astype(np.float32)
        if role in opts['duck_roles']:
            wave *= envelope[:, None]
        width = draw(rng, opts['width'].get(role, 1.))
        width_info = dict(width=width)
        if width!=1. or role in opts['synthetic_width_roles']:
            wave,width_info = widen(wave,role,width,opts,rng)
        pan = draw(rng, opts['pan'].get(role, 0.))
        angle = (1 + pan) * np.pi / 4
        wave *= (np.sqrt(2.) * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32))
        drum = role in DRUM_ROLES
        bus = role+'_send' if role in ('kick','hat','bass') else 'room_send' if drum else 'hall_send'
        send = draw(rng,opts[bus])
        delay_beats = rng.choice(opts['delay_beats'])
        send_delay = draw(rng, opts['delay_send']) if role in opts['delay_roles'] else 0.
        dry = wave
        wet = dry.copy()
        duck_wet = None
        if clarity_dsp and not drum and role in ('lead', 'pluck', 'pad', 'vocal') and (send or send_delay):
            dry_env = np.sqrt(ndimage.uniform_filter1d(np.mean(dry * dry, axis=1), size=max(1, round(0.032 * RATE)), mode='nearest') + 1e-12)
            env_peak = float(np.percentile(dry_env, 92)) if len(dry_env) else 0.0
            if env_peak > 1e-5:
                duck_wet = np.clip(1.0 - 0.65 * (dry_env / max(env_peak, 1e-4)), 0.25, 1.0).astype(np.float32)
        if send:
            rev_ret = signal.oaconvolve(dry, room if drum else hall, axes=0)[:frames]
            if duck_wet is not None:
                rev_ret = rev_ret * (0.28 + 0.72 * duck_wet)[:, None]
            wet += send * rev_ret
        if send_delay:
            gain = send_delay
            feedback = draw(rng, opts['delay_feedback'])
            del_ret = np.zeros_like(dry)
            for repeat in range(1, opts['delay_repeats'] + 1):
                del_ret += gain * delayed(dry[:, ::-1] if repeat % 2 else dry, frames, delay_beats * repeat, grid)
                gain *= feedback
            if duck_wet is not None:
                del_ret = del_ret * (0.22 + 0.78 * duck_wet)[:, None]
            wet += del_ret
        wet = mono_bass(wet,opts['stem_mono_hz'])
        corr = correlation(filtered(wet,opts['width_highpass_hz'],'highpass'))
        if corr<0:
            mid,side = (wet[:,0]+wet[:,1])*.5,(wet[:,0]-wet[:,1])*.5
            recovery = max(0.,corr+opts['side_recovery_margin'])
            wet = np.column_stack((mid+recovery*side,mid-recovery*side))
            width_info.update(side_recovery=recovery)
        stems[role] = wet
        settings[role] = dict(**width_info, role_eq=eq, pan=pan, correlation_before_recovery=corr, reverb_send=send,
                              delay_beats=delay_beats,delay_send=send_delay,delay_feedback=feedback if send_delay else 0.)
    return dict(room_ir=room_name['source'],hall_ir=hall_name['source'],room_settings=room_name,
                hall_settings=hall_name, spatial=settings)


def mono_bass(wave, hz):
    if hz:
        mid, side = (wave[:, 0] + wave[:, 1]) * .5, (wave[:, 0] - wave[:, 1]) * .5
        hp = signal.butter(4, hz, 'highpass', fs=RATE, output='sos')
        side = signal.sosfilt(hp, side).astype(np.float32)
        wave = np.column_stack((mid + side, mid - side))
    return wave


def encode_domain(wave, opts, rng):
    codec = choice(rng, opts['codec_weights'])
    ffmpeg = shutil.which(opts.get('ffmpeg', 'ffmpeg'))
    if codec == 'lowpass' or (codec in ('mp3', 'ogg') and not ffmpeg):
        hz = draw(rng, opts['lowpass_hz'])
        sos = signal.butter(8, hz, fs=RATE, output='sos')
        return signal.sosfilt(sos, wave, axis=0).astype(np.float32), dict(codec='lowpass', cutoff_hz=hz)
    if codec == 'none':
        return wave, dict(codec='none')
    bitrate = rng.choice(opts['codec_bitrates_kbps'])
    with TemporaryDirectory() as tmp:
        dest = str(Path(tmp) / ('mix.' + codec))
        # A seekable container retains encoder delay/padding information (especially MP3 Xing).
        subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-f', 'f32le', '-ar', str(RATE),
                        '-ac', '2', '-i', 'pipe:0', '-c:a', {'mp3':'libmp3lame', 'ogg':'libvorbis'}[codec],
                        '-b:a', f'{bitrate}k', dest], input=wave.astype('<f4', copy=False).tobytes(), check=True)
        result, _ = sf.read(dest, dtype='float32', always_2d=True)
    if len(result) < len(wave):
        result = np.pad(result, ((0, len(wave)-len(result)), (0,0)))
    return result[:len(wave)], dict(codec=codec, bitrate_kbps=bitrate)


def lookahead_limit(wave, ceiling, lookahead_ms, release_ms):
    look = max(1,round(lookahead_ms*.001*RATE))
    raw = np.minimum(1.,ceiling/np.maximum(np.max(np.abs(wave),axis=1),1e-10))
    ahead = ndimage.minimum_filter1d(raw,2*look-1,mode='nearest')
    smooth = np.minimum(raw,ndimage.uniform_filter1d(ahead,look,mode='nearest'))
    # g[t] = 1 - max(1-smooth[t], (1-g[t-1])*decay). Local exponent scales
    # keep the exact peak-hold/release recurrence finite on arbitrarily long songs.
    tau = max(1.,release_ms*.001*RATE)
    block = max(1,min(8192,int(8*tau)))
    exponent = np.exp(np.arange(block)/tau)
    previous = 0.
    for i in range(0,len(smooth),block):
        part = smooth[i:i+block]
        held = (1-part)*exponent[:len(part)]
        held[0] = max(held[0], previous*math.exp(-1/tau))
        np.maximum.accumulate(held,out=held)
        attenuation = held/exponent[:len(part)]
        previous = attenuation[-1]
        part[:] = 1-attenuation
    out = wave*smooth[:,None]
    return np.clip(out,-ceiling,ceiling,out=out)


def master_dynamics(mix, opts, rng, profile):
    glue = None
    if rng.random()<opts['master_glue_probability']:
        from pedalboard import Compressor
        rms_db = float(db(np.mean(mix**2,dtype=np.float64)))
        glue = dict(threshold_db=rms_db+draw(rng,opts['master_glue_threshold_above_rms_db']),
                    ratio=draw(rng,opts['master_glue_ratio']), attack_ms=draw(rng,opts['master_glue_attack_ms']),
                    release_ms=draw(rng,opts['master_glue_release_ms']))
        mix = Compressor(**glue)(np.ascontiguousarray(mix.T),RATE).T
    drive_db = None
    if rng.random()<opts['master_clip_probability']:
        drive_db = draw(rng,opts['master_drive_db'])
        drive = 10**(drive_db/20)
        mix = np.tanh(mix*np.float32(drive/max(float(np.max(np.abs(mix))),1e-8)))/np.float32(np.tanh(drive))
    look = draw(rng,opts['limiter_lookahead_ms'])
    release = draw(rng,opts['limiter_release_ms'])
    ceiling = 10**(opts['master_ceiling_db']/20)
    limits = opts['target_lufs']
    low,high = limits if isinstance(limits,(tuple,list)) else (limits,limits)
    examples = [s for s in profile.get('loudness',()) if low<=s['lufs']<=high]
    ref = rng.choice(examples) if examples else None
    target = ref['lufs'] if ref else rng.uniform(low,high)
    source = integrated_loudness(mix,RATE)
    gain = target-source if math.isfinite(source) else 0.
    best, best_lufs, best_gain = mix,None,gain
    previous = None
    for _ in range(max(1,opts['loudness_iterations'])):
        result = lookahead_limit(mix*np.float32(10**(gain/20)),ceiling,look,release)
        measured = integrated_loudness(result,RATE)
        if best_lufs is None or abs(measured-target)<abs(best_lufs-target):
            best,best_lufs,best_gain = result,measured,gain
        if not math.isfinite(measured) or abs(measured-target)<=opts['loudness_tolerance_lu']:
            break
        slope = ((measured-previous[1])/(gain-previous[0])) if previous and gain!=previous[0] else .65
        previous = (gain,measured)
        gain += float(np.clip((target-measured)/max(.1,slope),-12.,12.))
    return best, dict(target_lufs=target, loudness_reference=ref['id'] if ref else None, glue=glue,
                      clip_drive_db=drive_db, limiter_lookahead_ms=look, limiter_release_ms=release,
                      limiter_gain_db=best_gain, limiter_lufs=best_lufs if math.isfinite(best_lufs) else None)


def master(mix, opts, rng, profile):
    info = {}
    if rng.random() < opts['reference_eq_probability']:
        paths = opts.get('reference_audio') or ()
        refs = profile.get('references') or ()
        mean = profile.get('reference_mean_spectrum')
        if not paths and mean and rng.random()<opts['reference_mean_probability']:
            target = mean
            info['reference'] = 'profile_mean'
        elif paths or refs:
            count = draw_int(rng,opts['reference_count'])
            pool = paths or refs
            subset = rng.sample(pool,min(len(pool),count))
            if paths:
                subset = [dict(id=str(p),spectrum=reference_spectrum(str(p),opts['reference_eq_smoothing_oct'])) for p in subset]
            target = np.mean([s['spectrum'] for s in subset],axis=0)
            info['reference'] = [s['id'] for s in subset]
        else:
            target = None
        if target is not None:
            strength = draw(rng,opts['reference_eq_strength'])
            info.update(eq_strength=strength,eq_smoothing_oct=opts['reference_eq_smoothing_oct'])
            mix,info['eq_db'] = reference_eq(mix,target,dict(opts,reference_eq_strength=strength),rng)
    ceiling = 10 ** (opts['master_ceiling_db'] / 20)
    mix, dynamics = master_dynamics(mix,opts,rng,profile)
    mix, encoding = encode_domain(mix, opts, rng)
    mix = mono_bass(mix, opts['mono_bass_hz'])
    # Lossy overshoot must not clip only the exported PCM relative to online training.
    mix *= min(1., ceiling / max(float(np.abs(mix).max()), 1e-8))
    measured = integrated_loudness(mix,RATE)
    return mix.astype(np.float32, copy=False), {**info, **dynamics, **encoding,
                                                'achieved_lufs':measured if math.isfinite(measured) else None}
