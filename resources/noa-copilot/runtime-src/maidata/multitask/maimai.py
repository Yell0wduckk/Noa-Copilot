"""Adapt original chart IR and export Simai in tick96 score coordinates."""
from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path

from .codec import Event
from .prepare import metadata_from_ir


def events_from_ir(document):
    """Read existing accepted Chart IR directly, including absolute durations.

    Resolve written-BPM and absolute-time source notation once at import.
    Keep score ticks for the Decoder and audio seconds for crops/augmentation.
    """
    from maidata.time import TimingMap
    from maidata.chart.routes import route_for, route_shape_end
    timing = TimingMap.from_ir(document)
    events = []

    def seconds(value, start_beat):
        if 'seconds' in value:
            return float(value['seconds'])
        if value.get('basis') == 'beat':
            beats = Fraction(value['beat_num'], value['beat_den'])
            return timing.beat_to_second(start_beat + beats) - timing.beat_to_second(start_beat)
        return float(value.get('resolved_ticks', 0)) / document.get('tick_rate', 180)

    for moment in document['moments']:
        position = moment['position_beats']
        beat = Fraction(position['num'], position['den'])
        onset = float(moment.get('second', timing.beat_to_second(beat)))
        for group_index, group in enumerate(moment['groups']):
            for note in group['notes']:
                kind = note['kind']
                flags = note.get('head', {}).get('flags', note.get('flags', []))
                data = dict(start_tick=round(beat*96), pad=note.get('pad', note.get('start_pad')),
                            pseudo_group=group_index, ex='ex' in flags, pseudo_each=(group_index > 0 or bool(note.get('pseudo_each'))),
                            break_='break' in flags,
                            firework='firework' in flags)
                if kind in ('tap','touch') and any(f in flags for f in ('short_hold','short_touch_hold')):
                    kind = 'hold' if kind == 'tap' else 'touch_hold'
                if kind in ('hold', 'touch_hold'):
                    data['duration'] = seconds(note.get('duration', {}), beat)
                    data['end_tick'] = round(96*timing.second_to_beat(onset+data['duration']))
                elif kind in ('slide', 'wifi'):
                    kind = 'slide'
                    data['head_type'] = {'star': 0, 'tap': 1, 'none_immediate': 2,
                                         'none_fade': 2}.get(note['head']['kind'], 0)
                    if 'star_material_double' in flags:
                        data['head_type'] = 1
                    branches = []
                    for branch in note['branches']:
                        wait = seconds(branch.get('wait', note['wait']), beat)
                        start, routes = data['pad'], []
                        for segment in branch['segments']:
                            shape, target = route_shape_end(start, route_for(segment['path']))
                            routes.append({'shape': shape, 'end': target})
                            start = segment['end_pad']
                        if branch['timing_mode'] == 'total':
                            duration = seconds(branch['duration'], timing.second_to_beat(onset + wait))
                        else:
                            cursor = onset + wait
                            for segment in branch['segments']:
                                cursor += seconds(segment['duration'], timing.second_to_beat(cursor))
                            duration = cursor-(onset+wait)
                        branches.append(dict(routes=routes, wait=wait, duration=duration,
                                             launch_tick=round(96*timing.second_to_beat(onset+wait)),
                                             end_tick=round(96*timing.second_to_beat(onset+wait+duration)),
                                             track_break='break' in branch.get('flags', [])))
                    data['branches'] = branches
                events.append(Event(onset, kind, data))
    return sorted(events, key=lambda e: e.time)


def read_maimai(path, difficulty=5):
    """Load accepted Chart IR JSON or an already-normalized Event JSON list."""
    if Path(path).suffix.lower()=='.txt':
        from maidata.spine.simai import read_simai
        return read_simai(path,difficulty)[0]
    document = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if isinstance(document, list):
        return [Event(**row) for row in document]
    return events_from_ir(document)


def write_simai(events, path, *, bpm=120.0, offset=0.0, segments=None,
                difficulty=5, level=13.0, title='', artist='', designer='maidata',
                pseudo_each='backtick', audio_duration=None):
    """Write tick96 notes (start_tick, end_tick, per-branch launch_tick/end_tick).

    Relative duration_ticks/wait_ticks and legacy Event inputs are accepted.
    """
    from .simai_writer import tick_score, render_score
    score = tick_score(events, bpm=bpm, offset=offset, segments=segments, pseudo_each=pseudo_each)
    if pseudo_each=='backtick':
        from maidata.spine.codec import normalize,to_score
        normalized=normalize([Event(score['grid'].second(n['start_tick']),n['kind'],n) for n in score['notes']],score['grid'])
        score['notes']=to_score(normalized)
    stop_tick = score['grid'].tick(audio_duration) if audio_duration is not None else None
    chart, first = render_score(score, stop_tick=stop_tick)
    clean = lambda text: str(text).replace('\n',' ').replace('\r',' ').replace('&',' ')
    text = '\n'.join((f'&title={clean(title)}', f'&artist={clean(artist)}',
                      f'&first={first:.12g}', f'&wholebpm={bpm:.12g}', '&cabinet=DX',
                      f'&lv_{difficulty}={level:.4g}', f'&des_{difficulty}={clean(designer)}',
                      f'&inote_{difficulty}={chart}', ''))
    Path(path).write_text(text, encoding='utf-8', newline='\n')
    return Path(path)
