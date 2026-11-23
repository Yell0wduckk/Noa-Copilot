"""Route-71 and spatial transforms; no model or Physical imports."""
from functools import lru_cache
from importlib.resources import files
import json
import re

from .types import ChartError, BUTTON_PADS, TOUCH_PADS


@lru_cache(maxsize=1)
def catalog():
    return tuple(json.loads(files('maidata.chart').joinpath('assets/slide_variant_catalog.json').read_text('utf-8'))['variants'])


def path_for(start_pad, route_id):
    if start_pad not in BUTTON_PADS or type(route_id) is not int or not 0 <= route_id < 71:
        raise ChartError('invalid start/Route-71 pair')
    return catalog()[route_id]['paths_by_start'][int(start_pad[1])-1]


def route_spellings(start_pad, route_id):
    path = path_for(start_pad, route_id)
    if SLIDE_SHAPES[ROUTE_SHAPES[route_id]] in ('arc_ccw_short', 'arc_cw_short'):
        return path, path[0] + '^' + path[-1]
    return (path,)


@lru_cache(maxsize=1)
def _inverse():
    return {path: route for start in BUTTON_PADS for route in range(71)
            for path in route_spellings(start, route)}


def route_for(path):
    try:
        return _inverse()[path]
    except KeyError as e:
        raise ChartError(f'unsupported Route-71 path: {path}') from e


def chain_paths(start_pad, routes):
    if 70 in routes and len(routes) != 1:
        raise ChartError('WiFi cannot be a segment of a continuous chain')
    result = []
    for route in routes:
        path = path_for(start_pad, route)
        result.append(path)
        start_pad = 'A'+path[-1]
    return tuple(result)


def mirror_pad(pad):
    if pad == 'C':
        return pad
    if pad not in set(BUTTON_PADS+TOUCH_PADS):
        raise ChartError(f'invalid pad: {pad}')
    lane = int(pad[1])
    lane = (1-lane) % 8+1 if pad[0] in 'DE' else 9-lane
    return pad[0]+str(lane)


def mirror_route(start_pad, route):
    path = path_for(start_pad, route)
    # Route-71's < and > are absolute rotation directions; p/q and s/z swap.
    path = path.translate(str.maketrans('<>pqsz', '><qpzs'))
    path = re.sub(r'[1-8]', lambda m: str(9-int(m[0])), path)
    return route_for(path)


# Rotation-invariant shape families. Endpoint remains an absolute button.
SLIDE_SHAPES = ('straight', 'arc_ccw_short', 'arc_cw_short', 'arc_ccw_long',
    'arc_cw_long', 'p_short', 'p_long', 'pp_short', 'pp_long', 'q_short',
    'q_long', 'qq_short', 'qq_long', 's', 'z', 'v', 'V_cw', 'V_ccw', 'w')
_SHAPE_ROUTES = (range(0,5), range(5,8), range(8,11), range(11,16),
    range(16,21), range(21,26), range(37,40), range(26,29), range(40,45),
    range(29,34), range(45,48), range(34,37), range(48,53), (53,), (54,),
    range(55,62), range(62,66), range(66,70), (70,))
ROUTE_SHAPES = tuple(next(s for s, routes in enumerate(_SHAPE_ROUTES) if r in routes)
                     for r in range(71))


@lru_cache(maxsize=1)
def slide_motion_groups():
    """The catalog's nine motion families, expressed as existing shape IDs."""
    motions = tuple(dict.fromkeys(v['motion'] for v in catalog()))
    shape_motions = tuple(catalog()[next(iter(routes))]['motion'] for routes in _SHAPE_ROUTES)
    return tuple(tuple(s for s, motion in enumerate(shape_motions) if motion == name)
                 for name in motions)


@lru_cache(maxsize=1)
def shape_route_table():
    """[shape, start, absolute end] -> Route-71, -1 for impossible pairs."""
    table = [[[-1]*8 for _ in range(8)] for _ in SLIDE_SHAPES]
    for r, shape in enumerate(ROUTE_SHAPES):
        for start in range(8):
            end = int(path_for(BUTTON_PADS[start], r)[-1])-1
            table[shape][start][end] = r
    return tuple(tuple(tuple(row) for row in shape) for shape in table)


def route_shape_end(start_pad, route_id):
    return ROUTE_SHAPES[route_id], int(path_for(start_pad, route_id)[-1])-1
