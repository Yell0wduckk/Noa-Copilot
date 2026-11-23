"""Recover physical-second events from accepted v7 source chart records.

The v7 model packs contain beat-coordinate training tensors, not original
timing. Their manifest selects the charts; the corresponding accepted
``full-chart-original.v2`` records supply tokens, timing and token ownership.
No geometry, Physical replay or original simai parsing is needed here.
"""
from maidata.chart.routes import route_shape_end
from maidata.time import TimingMap, duration_end

from .codec import Event
from .prepare import metadata_from_ir


def events_from_record(record):
    """Decode an accepted chart record without quantizing its second values.

    ``token_map.second`` is the onset authority. Slide actions retain their
    physical launch/end seconds; press/release guard endpoints are never used
    as musical durations. Records without materialized actions use their
    original written-BPM or global-tick duration contract.
    """
    timing = TimingMap.from_ir(record['metadata'], validate_input=False)
    actions = {row['token_index']: row['action'] for row in record.get('actions', ())}
    events = []
    current = None
    routes, track_break = [], False
    for index, (token, position) in enumerate(zip(record['tokens'], record['token_map'])):
        family = token['type']
        if family == 'MOMENT':
            continue
        tick = position['tick']
        if family == 'NOTE':
            kind = token['kind']
            data = dict(pad=token['pad'], start_tick=tick, ex=bool(token.get('ex')),
                        pseudo_group=int((position.get('source') or {}).get('group_index',0)),
                        pseudo_each=bool(token.get('pseudo_each') or (position.get('source') or {}).get('group_index',0)),
                        break_=bool(token.get('break')), firework=bool(token.get('firework')))
            if token.get('short_hold') or token.get('short_touch_hold'):
                kind = 'hold' if token.get('short_hold') else 'touch_hold'
                data['duration'] = 0.
                data['end_tick'] = tick
            if kind == 'slide':
                data.update(head_type=token.get('head_type', 0), branches=[])
            current = Event(float(position['second']), kind, data)
            events.append(current)
            routes, track_break = [], False
        elif family == 'HOLD_DURATION':
            release = duration_end(timing, tick, token['ticks'])
            current.data.update(duration=release-timing.second(tick), end_tick=round(96*timing.second_to_beat(release)))
        elif family == 'ROUTE':
            start = current.data['pad'] if not routes else 'A' + str(routes[-1]['end'] + 1)
            shape, end = route_shape_end(start, token['route_id'])
            if not routes:
                track_break = bool(token.get('track_break'))
            routes.append(dict(shape=shape, end=end))
        elif family == 'SLIDE_DURATION':
            action = actions.get(index)
            if action is not None:
                launch, end = float(action['launch']), float(action['actual_end'])
            else:
                wait_ticks = token.get('wait_ticks', 96)
                launch = timing.second(tick + wait_ticks)
                end = (timing.second(tick + wait_ticks + token['ticks']) if timing.global_ticks
                       else launch + token['ticks'] * timing.written_tick_seconds(tick))
            current.data['branches'].append(dict(routes=routes, track_break=track_break,
                launch_tick=round(96*timing.second_to_beat(launch)), end_tick=round(96*timing.second_to_beat(end)),
                wait=launch - current.time, duration=end - launch))
            routes, track_break = [], False
    return events


def metadata_from_record(record):
    """Preserve accepted timing and source identity in current metadata names."""
    result = metadata_from_ir(record['metadata'])
    source = record.get('source', {})
    for key in ('song_id', 'version'):
        if source.get(key) is not None:
            result[key] = source[key]
    if source.get('difficulty') is not None:
        result['difficulty'] = int(source['difficulty'])
    for key in ('level', 'designer'):
        if isinstance(source.get(key), (int, float)):
            result[key] = source[key]
    return result
