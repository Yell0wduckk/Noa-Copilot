"""Continuous score beats <-> local audio seconds, including crop, speed and BMS STOP.

Step BPM metadata uses (audio second, BPM). A segment can additionally give an
end_bpm: its BPM varies linearly in physical time up to the next segment.
"""
from bisect import bisect_left, bisect_right
import math

import torch

METER_DENOMINATORS = (2, 4, 8, 16)
_PRIORITY_DENOMINATORS = (4, 8, 16, 2)
_DENOMINATOR_TO_ID = {2: 0, 4: 1, 8: 2, 16: 3}

MIN_HOLD_TICKS = 6
MAX_GRID_PULSES = 200_000


def hold_end_tick(onset, end):
    """A tick96 Hold shorter than six ticks releases at its onset."""
    return onset if end - onset < MIN_HOLD_TICKS else end


def multiplier_to_meter(multiplier: float) -> tuple[int, int]:
    """Map a BMS/BMSON bar length multiplier (1.0 = 4 quarter beats) to (N, D)."""
    span = max(0.25, float(multiplier) * 4.0)
    best_err = math.inf
    best = (4, 4)
    for den in _PRIORITY_DENOMINATORS:
        num = min(16, max(1, round(span * den / 4.0)))
        err = abs(4.0 * num / den - span)
        if err < best_err - 1e-6:
            best_err = err
            best = (int(num), int(den))
    return best


def normalize_meter(num: int, den: int) -> tuple[int, int]:
    """Map any time signature (num, den) into N in 1..16 and D in (2, 4, 8, 16)."""
    n = int(round(float(num)))
    d = int(round(float(den)))
    if d in METER_DENOMINATORS and 1 <= n <= 16:
        return n, d
    if d <= 0 or n <= 0:
        return 4, 4
    return multiplier_to_meter(float(n) / float(d))


class BeatGrid:
    def __init__(self, metadata=None):
        m = metadata or {}
        self.crop = float(m.get('crop_start', 0.))
        self.speed = float(m.get('speed', 1.))
        offset = float(m.get('offset_seconds', 0.))
        self.has_tempo = bool(m.get('bpm_segments') or m.get('bpm') is not None or m.get('stops'))
        raw = m.get('bpm_segments') or [(offset, m.get('bpm', 120.))]
        # Remove elapsed STOP time before integrating the tempo. At the exact
        # stop beat, the attack is heard before the pause, as in the BMS mixer.
        stops = sorted((s['time'], s['duration']) for s in m.get('stops', ()))
        self.stop_times, self.stop_ends, self.stop_offsets = [], [], [0.]
        self.stop_positions = []
        for t, duration in stops:
            self.stop_times.append(float(t))
            self.stop_ends.append(float(t + duration))
            self.stop_positions.append(float(t - self.stop_offsets[-1]))
            self.stop_offsets.append(float(self.stop_offsets[-1] + duration))
        rows = [(r['start_seconds'], r['bpm'], r.get('end_bpm', r['bpm'])) if isinstance(r, dict)
                else (r[0], r[1], r[1]) for r in raw]
        self.starts = [self._running(r[0]) for r in rows]
        self.bpms = [float(r[1]) for r in rows]
        self.slopes = [(r[2] - r[1]) / (self.starts[i+1] - self.starts[i])
                       if i+1 < len(rows) and r[2] != r[1] else 0. for i, r in enumerate(rows)]
        self.beats = [0.]
        for i, dt in enumerate(b-a for a, b in zip(self.starts, self.starts[1:])):
            self.beats.append(self.beats[-1] + dt * (self.bpms[i] + .5*self.slopes[i]*dt) / 60)
        zero = self._beat(self._running(offset))
        self.beats = [v-zero for v in self.beats]
        self.stop_beats = [float(s.get('beat', self._beat(pos)))
                           for s, pos in zip(sorted(m.get('stops', ()), key=lambda x: x['time']), self.stop_positions)]
        self._tensor_cache = {}

        raw_meters = m.get('meter_segments')
        multipliers = m.get('measure_multipliers')
        bar_beats = m.get('bar_beats')
        self.ambiguous_compound_bars = m.get("ambiguous_compound_bars")
        self.bar_beats_explicit = sorted(set(map(float, bar_beats or ())))
        self.has_meter = bool(raw_meters or multipliers or m.get('numerator') is not None or bar_beats)
        if raw_meters:
            segments = []
            for row in raw_meters:
                n, d = normalize_meter(row[1], row[2])
                segments.append((float(row[0]), n, d))
        elif multipliers:
            segments = []
            max_m = max((int(k) for k in multipliers), default=0)
            cursor = 0.0
            for idx in range(max_m + 2):
                mult = float(multipliers.get(str(idx), multipliers.get(idx, 1.0)))
                n, d = multiplier_to_meter(mult)
                if not segments or segments[-1][1:] != (n, d) or abs(4.0 * n / d - 4.0 * mult) > 1e-6:
                    segments.append((cursor, n, d))
                cursor += 4.0 * mult
        elif self.bar_beats_explicit and len(self.bar_beats_explicit) >= 2:
            segments = []
            bars_sorted = sorted(self.bar_beats_explicit)
            for b0, b1 in zip(bars_sorted, bars_sorted[1:]):
                if b1 > b0:
                    n, d = multiplier_to_meter((b1 - b0) / 4.0)
                    if not segments or segments[-1][1:] != (n, d) or abs(4.0 * n / d - (b1 - b0)) > 1e-6:
                        segments.append((b0, n, d))
            if not segments:
                segments = [(0.0, 4, 4)]
        else:
            n, d = normalize_meter(m.get('numerator', 4) or 4, m.get('denominator', 4) or 4)
            segments = [(0.0, n, d)]
        self.meter_segments = segments
        self.meter_starts = [s[0] for s in segments]
        self.meter_nums = [s[1] for s in segments]
        self.meter_dens = [s[2] for s in segments]
        self.meter_spans = [4.0 * s[1] / s[2] for s in segments]

    def _running(self, song_second):
        i = bisect_right(self.stop_ends, song_second)
        if i < len(self.stop_times):
            song_second = min(song_second, self.stop_times[i])
        return song_second - self.stop_offsets[i]

    def _beat(self, running):
        i = max(0, bisect_right(self.starts, running) - 1)
        dt = running - self.starts[i]
        return self.beats[i] + dt * (self.bpms[i] + .5*self.slopes[i]*max(0., dt)) / 60

    def second_to_beat(self, second):
        return self._beat(self._running(self.crop + second * self.speed))

    def beat_to_second(self, beat):
        i = max(0, bisect_right(self.beats, beat) - 1)
        area = (beat - self.beats[i]) * 60
        bpm, slope = self.bpms[i], self.slopes[i]
        dt = 2*area/(bpm + math.sqrt(max(0., bpm*bpm + 2*slope*area))) if slope and area > 0 else area/bpm
        running = self.starts[i] + dt
        stops = self.stop_offsets[bisect_left(self.stop_positions, running - 1e-10)]
        return (running + stops - self.crop) / self.speed

    def second_to_beat_tensor(self, seconds: torch.Tensor) -> torch.Tensor:
        s = torch.as_tensor(seconds, dtype=torch.float32)
        song_sec = self.crop + s * self.speed
        if self.stop_ends:
            stop_times = s.new_tensor(self.stop_times)
            stop_ends = s.new_tensor(self.stop_ends)
            stop_offsets = s.new_tensor(self.stop_offsets)
            idx = torch.searchsorted(stop_ends, song_sec.contiguous(), right=True)
            in_range = idx < len(self.stop_times)
            clamped_idx = idx.clamp_max(max(0, len(self.stop_times) - 1))
            clamped_sec = torch.where(in_range, torch.minimum(song_sec, stop_times[clamped_idx]), song_sec)
            running = clamped_sec - stop_offsets[idx]
        else:
            running = song_sec
        starts = s.new_tensor(self.starts)
        beats = s.new_tensor(self.beats)
        bpms = s.new_tensor(self.bpms)
        slopes = s.new_tensor(self.slopes)
        seg_idx = torch.searchsorted(starts, running.contiguous(), right=True).sub_(1).clamp_min_(0)
        dt = running - starts[seg_idx]
        return beats[seg_idx] + dt * (bpms[seg_idx] + 0.5 * slopes[seg_idx] * dt.clamp_min(0.0)) / 60.0

    def beat_to_second_tensor(self, beats: torch.Tensor) -> torch.Tensor:
        b = torch.as_tensor(beats, dtype=torch.float32)
        tc = self._device_tensors(b.device)
        seg_beats, seg_starts, seg_bpms, seg_slopes = (tc[k] for k in ('seg_beats','seg_starts','seg_bpms','seg_slopes'))
        ids = torch.searchsorted(seg_beats, b.contiguous(), right=True).sub_(1).clamp_min_(0)
        area = (b - seg_beats[ids]) * 60.0
        bpm = seg_bpms[ids]
        slope = seg_slopes[ids]
        dt = torch.where(
            (slope != 0.0) & (area > 0.0),
            2.0 * area / (bpm + torch.sqrt((bpm * bpm + 2.0 * slope * area).clamp_min(0.0))),
            area / bpm,
        )
        running = seg_starts[ids] + dt
        if self.stop_positions:
            # Compare in beat coordinates to keep an attack on its STOP on
            # the pre-pause side, even if tempo integration rounded by one ULP.
            s_idx = torch.searchsorted(tc["stop_b"], b.contiguous(), right=False)
            stops = tc["stop_off"][s_idx]
        else:
            stops = 0.0
        return (running + stops - self.crop) / self.speed

    def _device_tensors(self, device):
        cached = self._tensor_cache.get(device)
        if cached is None:
            cached = dict(
                bars=torch.tensor(self.bar_beats_explicit, dtype=torch.float32, device=device),
                m_starts=torch.tensor(self.meter_starts, dtype=torch.float32, device=device),
                m_nums=torch.tensor(self.meter_nums, dtype=torch.long, device=device),
                m_dens=torch.tensor(self.meter_dens, dtype=torch.long, device=device),
                m_spans=torch.tensor(self.meter_spans, dtype=torch.float32, device=device),
                m_den_ids=torch.tensor([_DENOMINATOR_TO_ID.get(int(d), 1) for d in self.meter_dens],
                                       dtype=torch.long, device=device),
                seg_beats=torch.tensor(self.beats, dtype=torch.float32, device=device),
                seg_starts=torch.tensor(self.starts, dtype=torch.float32, device=device),
                seg_bpms=torch.tensor(self.bpms, dtype=torch.float32, device=device),
                seg_slopes=torch.tensor(self.slopes, dtype=torch.float32, device=device),
                stop_off=torch.tensor(self.stop_offsets,device=device,dtype=torch.float32),
                stop_b=torch.tensor(self.stop_beats, dtype=torch.float32, device=device) if self.stop_beats else None,
            )
            self._tensor_cache[device] = cached
        return cached

    def bar_info(self, beats):
        b = torch.as_tensor(beats, dtype=torch.float32)
        tc = self._device_tensors(b.device)
        starts, nums, dens, spans, den_ids = tc['m_starts'], tc['m_nums'], tc['m_dens'], tc['m_spans'], tc['m_den_ids']

        ids = torch.searchsorted(starts, b.contiguous(), right=True).sub_(1).clamp_min_(0)
        active_start = starts[ids]
        active_span = spans[ids]
        active_num = nums[ids]
        active_den = dens[ids]
        active_den_id = den_ids[ids]

        bars = tc['bars']
        if bars.numel() > 1:
            i = torch.searchsorted(bars, b.contiguous(), right=True) - 1
            inside = (i >= 0) & (i < len(bars) - 1)
            j = i.clamp(0, len(bars) - 2)
            active_start = torch.where(inside, bars[j], active_start)
            active_start = torch.where(b >= bars[-1], torch.maximum(active_start, bars[-1]), active_start)
            active_span = torch.where(inside, bars[j + 1] - bars[j], active_span)

        bar_offset = (b - active_start).remainder(active_span)
        bar_offset = torch.where(active_span - bar_offset < 1e-4, torch.zeros_like(bar_offset), bar_offset)
        bar_phase = (bar_offset / active_span).clamp(0.0, 1.0 - 1e-7)
        dist_to_bar = torch.minimum(bar_offset, (active_span - bar_offset).clamp_min(0.0))
        is_downbeat = (dist_to_bar < 0.005).float()
        pulse_scale = active_den.float() / 4.0
        beat_in_bar = torch.floor(bar_offset * pulse_scale + 1e-4).long().remainder(16)
        num_id = (active_num.clamp(1, 16) - 1).long()
        pulse_step = torch.where(active_den >= 8, 4.0 / active_den, 1.0)
        pulse_phase = ((b - active_start) / pulse_step).remainder(1)
        return dict(
            pulse_phase=torch.where(1.0 - pulse_phase < 1e-6, 0.0, pulse_phase),
            bar_phase=bar_phase,
            is_downbeat=is_downbeat,
            beat_in_bar=beat_in_bar,
            num=active_num,
            den=active_den,
            num_id=num_id,
            den_id=active_den_id,
            bar_beats=active_span,
            has_meter=self.has_meter,
        )

    def tempo_info(self, beats):
        b = torch.as_tensor(beats, dtype=torch.float32)
        tc = self._device_tensors(b.device)
        seg_beats, seg_bpms, seg_slopes, stop_b = tc['seg_beats'], tc['seg_bpms'], tc['seg_slopes'], tc['stop_b']

        ids = torch.searchsorted(seg_beats, b.contiguous(), right=True).sub_(1).clamp_min_(0)
        b0 = seg_beats[ids]
        bpm0 = seg_bpms[ids]
        slope = seg_slopes[ids]
        area = (b - b0) * 60.0
        instant_bpm = torch.where(
            (slope != 0.0) & (area > 0.0),
            torch.sqrt((bpm0.square() + 2.0 * slope * area).clamp_min(1.0)),
            bpm0,
        )
        bpm = (instant_bpm / self.speed).clamp_min(1.0)
        base_id = max(0, bisect_right(self.beats, 0.0) - 1)
        base_bpm = max(1.0, float(self.bpms[base_id]) / self.speed)
        if stop_b is not None:
            dist = (b[..., None] - stop_b).abs().min(dim=-1).values
            is_stop = (dist < 0.005).float()
        else:
            is_stop = torch.zeros_like(b)
        return dict(
            bpm=bpm,
            base_bpm=base_bpm,
            slope=slope / self.speed,
            is_stop=is_stop,
            has_tempo=self.has_tempo,
        )

    def pulse_and_bar_beats(self, start, end):
        """Return physical pulse and bar coordinates in quarter beats."""
        def marks(origin, spacing, left, right):
            first = math.ceil((left - origin) / spacing - 1e-8)
            last = math.floor((right - origin) / spacing + 1e-8)
            if last - first > MAX_GRID_PULSES:
                raise ValueError(
                    f"BeatGrid pulse span [{left}, {right}] requires {last - first + 1} marks "
                    f"(exceeds {MAX_GRID_PULSES}); sanitize extreme BPM gimmicks first"
                )
            return (origin + k * spacing for k in range(first, last + 1))

        # Exact BMS lines own the whole interval between the first and last
        # line. Extrapolate only in the uncovered head and tail of the audio.
        explicit = self.bar_beats_explicit
        down = {b for b in explicit if start <= b <= end}
        pulses = set(down)
        for i, (anchor, num, den) in enumerate(self.meter_segments):
            lo = max(start, anchor) if i else start
            limit = self.meter_starts[i + 1] if i + 1 < len(self.meter_starts) else math.inf
            hi = min(end, limit)
            if hi < lo:
                continue
            span = 4.0 * num / den
            step = 4.0 / den if den in (8, 16) else 1.0
            intervals = [(anchor, lo, hi)]
            if len(explicit) >= 2:
                intervals = [(anchor, lo, min(hi, explicit[0])),
                             (max(anchor, explicit[-1]), max(lo, explicit[-1]), hi)]
                first = max(0, bisect_right(explicit, lo) - 1)
                last = min(len(explicit) - 1, bisect_right(explicit, hi))
                for j in range(first, last):
                    pulses.update(b for b in marks(explicit[j], step, max(lo, explicit[j]), min(hi, explicit[j+1]))
                                  if b < explicit[j+1] and b < limit)
            for origin, left, right in intervals:
                pulses.update(b for b in marks(origin, step, left, right) if b < limit)
                if self.has_meter:
                    down.update(b for b in marks(origin, span, left, right) if b < limit)
        return sorted(pulses | down), sorted(down)

    def calibrator_boundary_beats(self, start, end):
        """Eighth, quarter and bar marks; reset the two pulse clocks at every bar."""
        return self._calibrator_marks(start, end, tactus=False)

    def calibrator_tactus_beats(self, start, end):
        """Return eighth, audible tactus and bar coordinates on the score clock."""
        return self._calibrator_marks(start, end, tactus=True)

    def _calibrator_marks(self, start, end, *, tactus):
        # Ask for a preceding bar as a crop can start inside an odd-length bar.
        margin = max(self.meter_spans)
        _, bars = self.pulse_and_bar_beats(start - margin, end + margin)
        if not bars:
            bars = [4.0 * k for k in range(math.floor(start / 4), math.ceil(end / 4) + 1)]
        explicit = self.bar_beats_explicit
        if len(explicit) >= 2:
            left = bisect_right(explicit, start) - 1
            if left >= 0:
                bars = sorted(set(bars) | {explicit[left]})
        quarters, eighths = set(), set()
        for i, anchor in enumerate(bars):
            limit = bars[i + 1] if i + 1 < len(bars) else math.inf
            meter = max(0, bisect_right(self.meter_starts, anchor + 1e-7) - 1)
            num, den = self.meter_nums[meter], self.meter_dens[meter]
            spacing = 2
            if tactus and (den >= 8 or limit - anchor < 1 - 1e-5):
                spacing = 3 if den == 8 and num % 3 == 0 and limit-anchor >= 1 else None
            first = max(0, math.ceil((start - anchor) * 2 - 1e-7))
            last = math.floor((end - anchor) * 2 + 1e-7)
            for k in range(first, last + 1):
                beat = anchor + k / 2
                if beat >= limit - 1e-5:
                    break
                eighths.add(beat)
                if k == 0 or (spacing and k % spacing == 0):
                    quarters.add(beat)
        return sorted(eighths), sorted(quarters), [b for b in bars if start <= b <= end]

    def frame_rhythm_targets(self, num_frames: int, device=None):
        n = int(num_frames)
        if n <= 0:
            empty_f = torch.zeros(0, dtype=torch.float32, device=device)
            empty_l = torch.zeros(0, dtype=torch.long, device=device)
            return dict(beat_target=empty_f, downbeat_target=empty_f, log_bpm_target=empty_f,
                        stop_target=empty_f, num_target=empty_l, den_target=empty_l,
                        has_meter=self.has_meter, has_tempo=self.has_tempo)
        sec = (torch.arange(n, dtype=torch.float32, device=device) + 0.5) / 125.0
        song_sec = self.crop + sec * self.speed
        if self.stop_ends:
            stop_times = sec.new_tensor(self.stop_times)
            stop_ends = sec.new_tensor(self.stop_ends)
            stop_offsets = sec.new_tensor(self.stop_offsets)
            idx = torch.searchsorted(stop_ends, song_sec.contiguous(), right=True)
            in_range = idx < len(self.stop_times)
            clamped_idx = idx.clamp_max(max(0, len(self.stop_times) - 1))
            clamped_sec = torch.where(in_range, torch.minimum(song_sec, stop_times[clamped_idx]), song_sec)
            running = clamped_sec - stop_offsets[idx]
            stop_target = (in_range & (song_sec >= stop_times[clamped_idx]) & (song_sec < stop_ends[clamped_idx])).float()
        else:
            running = song_sec
            stop_target = torch.zeros(n, dtype=torch.float32, device=device)

        tc = self._device_tensors(sec.device)
        starts, beats, bpms, slopes = tc['seg_starts'], tc['seg_beats'], tc['seg_bpms'], tc['seg_slopes']
        seg_idx = torch.searchsorted(starts, running.contiguous(), right=True).sub_(1).clamp_min_(0)
        dt = running - starts[seg_idx]
        frame_beats = beats[seg_idx] + dt * (bpms[seg_idx] + 0.5 * slopes[seg_idx] * dt.clamp_min(0.0)) / 60.0
        instant_bpm = (bpms[seg_idx] + slopes[seg_idx] * dt.clamp_min(0.0)).clamp_min(1.0) / self.speed
        log_bpm_target = torch.log2(instant_bpm.clamp_min(1.0) / 120.0)
        b_info = self.bar_info(frame_beats)

        duration = n / 125.0
        b_start = self.second_to_beat(0.0)
        b_end = self.second_to_beat(duration)
        pulses_list, downbeats_list = self.pulse_and_bar_beats(b_start, b_end)

        beat_target = torch.zeros(n, dtype=torch.float32, device=device)
        downbeat_target = torch.zeros(n, dtype=torch.float32, device=device)
        if pulses_list:
            p_sec = self.beat_to_second_tensor(torch.tensor(pulses_list, dtype=torch.float32, device=device))
            p_frames = torch.round(p_sec * 125.0 - 0.5).long()
            valid_p = (p_frames >= 0) & (p_frames < n) & (p_sec >= -1e-6) & (p_sec < duration + 1e-6)
            beat_target[p_frames[valid_p]] = 1.0
        if downbeats_list:
            d_sec = self.beat_to_second_tensor(torch.tensor(downbeats_list, dtype=torch.float32, device=device))
            d_frames = torch.round(d_sec * 125.0 - 0.5).long()
            valid_d = (d_frames >= 0) & (d_frames < n) & (d_sec >= -1e-6) & (d_sec < duration + 1e-6)
            downbeat_target[d_frames[valid_d]] = 1.0
            beat_target[d_frames[valid_d]] = 1.0

        return dict(
            beat_target=beat_target,
            downbeat_target=downbeat_target,
            log_bpm_target=log_bpm_target,
            stop_target=stop_target,
            num_target=b_info['num_id'],
            den_target=b_info['den_id'],
            has_meter=self.has_meter,
            has_tempo=self.has_tempo,
        )

    def tick(self, second):
        return math.floor(self.second_to_beat(second)*96 + .5)

    def second(self, tick):
        return self.beat_to_second(tick/96)

    def step_tempos(self):
        """Step-clock spellings for score exports; ramps retain every tick boundary."""
        marks = {round(b*96):v for b,v in zip(self.beats,self.bpms)}
        for i,slope in enumerate(self.slopes):
            if slope:
                for t in range(round(self.beats[i]*96),round(self.beats[i+1]*96)):
                    marks[t] = 60/(96*(self.second(t+1)-self.second(t)))
        return marks

    def first_beat(self):
        return max(0, math.floor(self.second_to_beat(0.) + 1e-9))
