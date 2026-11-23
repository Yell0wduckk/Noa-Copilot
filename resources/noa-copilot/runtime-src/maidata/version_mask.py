"""CPU supervision availability from chart release metadata, never chart usage.

Profiles are shared read-only values. Applying one returns new label containers
and changed tensors, leaving the accepted cache payload and token stream intact.
"""
from dataclasses import dataclass, replace

import torch

from maidata.chart.types import APPLICABLE, FLAGS, KINDS, TOUCH_PADS

AXIS15_GROUPS = (
    (0, 4), (1, 3), (2,), (5, 10), (6, 9), (7, 8),
    (11, 16), (12, 20), (13, 19), (14, 18), (15, 17),
    (21, 29), (22, 33), (23, 32), (24, 31), (25, 30),
    (26, 34), (27, 36), (28, 35),
    (37, 47), (38, 46), (39, 45), (40, 52), (41, 51),
    (42, 50), (43, 49), (44, 48), (53, 54),
    (55,), (56, 61), (57, 60), (58, 59),
    (62, 69), (63, 68), (64, 67), (65, 66), (70,),
)
ROUTE71_TO_AXIS15 = tuple(next(i for i, group in enumerate(AXIS15_GROUPS)
                             if route in group) for route in range(71))

VERSION_SUPERVISION = 'maidata.cabinet-version-supervision.v1'
VERSION_ORDER = (
    'maimai', 'maimai PLUS', 'maimai GreeN', 'maimai GreeN PLUS',
    'maimai ORANGE', 'maimai ORANGE PLUS', 'maimai PiNK', 'maimai PiNK PLUS',
    'maimai MURASAKi', 'maimai MURASAKi PLUS', 'maimai MiLK', 'maimai MiLK PLUS',
    'maimai FiNALE', 'maimai DX', 'maimai DX PLUS', 'maimai DX Splash',
    'maimai DX Splash PLUS', 'maimai DX UNiVERSE', 'maimai DX UNiVERSE PLUS',
    'maimai DX FESTiVAL', 'maimai DX FESTiVAL PLUS', 'maimai DX BUDDiES',
    'maimai DX BUDDiES PLUS', 'maimai DX PRiSM', 'maimai DX PRiSM PLUS',
    'maimai DX CiRCLE', 'maimai DX CiRCLE PLUS',
)
_VERSION_INDEX = {name: i for i, name in enumerate(VERSION_ORDER)}
NON_C_TOUCH_HOLD_EXCEPTION_SONG_IDS = frozenset(('11820', '11821', '11879'))


@dataclass(frozen=True)
class VersionSupervisionProfile:
    cabinet: str
    effective_version: str
    beat_kind_mask: tuple[bool, ...]
    beat_flag_mask: tuple[bool, ...]
    axis15_mask: tuple[bool, ...]
    beat_kind_tensor: torch.Tensor
    axis15_tensor: torch.Tensor
    kind5_mask: torch.Tensor
    touch_pad33_mask: torch.Tensor
    touch_hold_pad33_mask: torch.Tensor
    note_flag_mask: torch.Tensor
    note_any_flag_mask: torch.Tensor
    route71_mask: torch.Tensor
    track_break_enabled: bool
    chained_route_enabled: bool
    multi_track_enabled: bool


def _compile(cabinet, version):
    era = _VERSION_INDEX[version]
    since = lambda name: era >= _VERSION_INDEX[name]
    dx = cabinet == 'DX'
    festival = dx and since('maimai DX FESTiVAL')
    short_hold = since('maimai ORANGE')
    short_touch_hold = dx and since('maimai DX Splash PLUS')
    pink = since('maimai PiNK')
    axes = {1, 2, 3, 9, 10}
    if since('maimai GreeN'):
        axes.add(4)
    if since('maimai GreeN PLUS'):
        axes.update((0, 7, 8, *range(11, 16), *range(19, 22), 27, *range(28, 32)))
    if since('maimai ORANGE'):
        axes.add(6)
    if pink:
        axes.update((*range(16, 19), *range(22, 27), *range(32, 37)))
    if dx and since('maimai DX UNiVERSE'):
        axes.add(5)
    kind = (True, True, dx, dx)
    beat_flags = (True, True, dx, dx, short_hold, short_touch_hold)
    axis = tuple(i in axes for i in range(37))
    available = {
        'tap': dict(ex=dx, break_=True, short_hold=short_hold),
        'hold': dict(ex=dx, break_=festival),
        'touch': dict(firework=dx, short_touch_hold=short_touch_hold),
        'touch_hold': dict(firework=dx),
        'slide': dict(ex=dx, break_=pink),
    }
    note_flags = torch.tensor([[f in APPLICABLE[k] and available[k][f] for f in FLAGS]
                               for k in KINDS], dtype=torch.bool)
    tensor = lambda values: torch.tensor(values, dtype=torch.bool)
    return VersionSupervisionProfile(cabinet, version, kind, beat_flags, axis,
        tensor(kind), tensor(axis), tensor((True, True, dx, dx, True)),
        tensor(tuple(dx and (festival or pad[0] in 'BEC') for pad in TOUCH_PADS)),
        tensor(tuple(dx and (since('maimai DX CiRCLE') or pad == 'C') for pad in TOUCH_PADS)),
        note_flags, note_flags.any(-1), tensor(tuple(axis[i] for i in ROUTE71_TO_AXIS15)),
        festival, festival, pink)


_PROFILES = {('SD' if i < 13 else 'DX', v): _compile('SD' if i < 13 else 'DX', v)
             for i, v in enumerate(VERSION_ORDER)}
_PRISM_EXCEPTION = replace(_PROFILES['DX', 'maimai DX PRiSM PLUS'],
    touch_hold_pad33_mask=torch.ones(33, dtype=torch.bool))


def profile_for_chart(song_id, difficulty, version):
    """Return a precompiled profile; unversioned legacy callers stay unmasked."""
    if song_id is None or version in (None, ''):
        return None
    number = int(song_id)
    if not 0 <= number < 100000:
        raise ValueError(f'unsupported chart song_id: {song_id!r}')
    cabinet = 'DX' if number >= 10000 else 'SD'
    if str(difficulty) == '6':
        version = 'maimai FiNALE' if cabinet == 'SD' else VERSION_ORDER[-1]
    if version not in _VERSION_INDEX:
        raise ValueError(f'unsupported chart version: {version!r}')
    if cabinet == 'SD' and _VERSION_INDEX[version] >= 13:
        version = 'maimai FiNALE'
    if cabinet == 'DX' and _VERSION_INDEX[version] < 13:
        raise ValueError(f'DX chart has pre-DX release: {version!r}')
    if str(song_id) in NON_C_TOUCH_HOLD_EXCEPTION_SONG_IDS and version == 'maimai DX PRiSM PLUS':
        return _PRISM_EXCEPTION
    return _PROFILES[cabinet, version]
