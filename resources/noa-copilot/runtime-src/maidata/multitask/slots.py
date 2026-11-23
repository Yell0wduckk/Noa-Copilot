"""Eight acoustic classes: event IDs 0..3 unpitched, 4..7 pitched.

Each token family uses local IDs 0..3; explicit event slots use the disjoint
eight-class space so kicks and vocals never collide.
"""
SLOT_NAMES = ('Kick', 'Snare / Clap', 'Hat / Perc', 'Crash / FX',
              'Vocal', 'Bass', 'Pitched-Percussive', 'Pitched-Sustained')
UNPITCHED_SLOTS = (0, 1, 2, 3)
PITCHED_SLOTS = (4, 5, 6, 7)
METALLIC_DRUM_PITCHES = frozenset((49, 52, 55, 57))
LEGACY_PROGRAM_CLASSES = ('Keys', 'Guitar', 'Bass', 'Strings', 'Winds', 'Synth Lead',
                          'Synth Pad', 'Vocal', 'Other', 'Drums', 'Unknown')
LEGACY_CLASS_SLOTS = (6, 6, 5, 7, 7, 7, 7, 4, 3, 2, 7)

def program_slot(program, *, teacher=False, vocal=False):
    if vocal or teacher and program == 100:
        return 4
    if program is None:
        return 7
    program = int(program)
    if 32 <= program <= 39:
        return 5
    if program == 55 or program >= 119:
        return 3
    if program == 113 or 115 <= program <= 118:
        return 2
    if (0 <= program <= 15 or 24 <= program <= 28 or program == 31 or
            45 <= program <= 47 or 104 <= program <= 108 or program in (112, 114)):
        return 6
    return 7

def drum_slot(pitch):
    if pitch in (35, 36):
        return 0
    if pitch in (37, 38, 39, 40):
        return 1
    return 3 if pitch in METALLIC_DRUM_PITCHES else 2

def event_slot(data, kind='note'):
    if 'slot' in data:
        return int(data['slot'])
    if any(data.get(k) for k in ('fine_id', 'coarse_id', 'fine_tag', 'coarse_tag')):
        from .timbre_tags import COARSE_TO_SLOT, resolve_tag_query
        c_id, _ = resolve_tag_query({
            'query_coarse_id': data.get('coarse_id', 0),
            'query_fine_id': data.get('fine_id', 0),
            'query_tag': data.get('fine_tag') or data.get('coarse_tag'),
        })
        if c_id > 0:
            return COARSE_TO_SLOT[c_id]
    if kind == 'drum' or data.get('program') in (9, 'Drums'):
        return drum_slot(data.get('pitch', 0))
    if 'gm_program' in data:
        return program_slot(data['gm_program'], teacher=data.get('pseudo', False), vocal=data.get('vocal', False))
    program = data.get('program', 10)
    if isinstance(program, str):
        program = LEGACY_PROGRAM_CLASSES.index(program)
    return LEGACY_CLASS_SLOTS[program] if 0 <= program < 11 else program_slot(program)

def supervision_group(data, slot):
    """Target-family bitset. Zero denotes a reliable hard class."""
    group = data.get('slot_supervision', 'hard')
    if 'slot' not in data and 'gm_program' not in data and data.get('program',10) in (10,'Unknown'):
        group='pitched'
    if group in ('pitched', 'unpitched'):
        return 15
    if group == 'env' or data.get('pseudo', False) and slot in (6, 7):
        return 12
    return 0

def velocity_bin(data):
    return int(data.get('velocity_bin', 0))

def velocity_amplitude(value):
    return (0., 127 * 10 ** (-31 / 20), 127 * 10 ** (-21 / 20),
            127 * 10 ** (-11 / 20), 127.)[int(value)]
