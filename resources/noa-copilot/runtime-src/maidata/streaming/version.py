"""Project existing cabinet availability onto accepted Simai prefixes."""
from maidata.chart.types import TOUCH_PADS
from maidata.chart.routes import shape_route_table


def values(grammar,family,profile):
    result = grammar.allowed_values(family)
    if profile is None: return result
    g = grammar
    if getattr(g,'physical_v6',False):
        from .physical_schema import TOUCH_PADS as PADS, TOUCH_PAD, SLIDE_WAIT, STATES
        if family==5: return result if profile.touch_pad33_mask.any() else ()
        if family==TOUCH_PAD: return tuple(v for v in result if profile.touch_pad33_mask[TOUCH_PADS.index(PADS[v])])
        if family in (*STATES,SLIDE_WAIT): return result
    if family == 5:
        from .sequence import ZONES
        return tuple(v for v in result if any(bool(profile.touch_pad33_mask[i])
            for i,pad in enumerate(TOUCH_PADS) if pad[0]==ZONES[v]))
    if family == 4 and g.phase == 'touch_pad':
        return tuple(v for v in result if bool(profile.touch_pad33_mask[TOUCH_PADS.index(g.zone+str(v+1))]))
    if (family == 7 or family == 8 and getattr(g,'no_placeholders',False)) and g.zone is not None:
        pad = 'C' if g.zone=='C' else g.zone+str(g.pad+1)
        if not profile.touch_hold_pad33_mask[TOUCH_PADS.index(pad)]: return ()
        if family==7: return result
    if family == 17:
        from copy import copy
        from .sequence import EventToken
        trial=copy(g); trial.accept(EventToken(17))
        return result if values(trial,9,profile) else ()
    if family == 18:
        from copy import copy
        from .sequence import EventToken
        trial=copy(g); trial.accept(EventToken(18))
        return result if values(trial,4,profile) or values(trial,5,profile) else ()
    if family == 9:
        phase = getattr(g,'slide_origin',g.phase) if g.phase=='slide_shape' else g.phase
        if phase == 'route_chain' and not profile.chained_route_enabled: return ()
        if phase == 'after_slide' and not profile.multi_track_enabled: return ()
        return tuple(s for s in result if any(r>=0 and profile.route71_mask[r]
            for r in shape_route_table()[s][g.pad if phase=='route_chain' else g.head_pad]))
    if family == 4 and g.phase == 'route_pad':
        return tuple(v for v in result if profile.route71_mask[shape_route_table()[g.shape][g.pad][v]])
    if family == 6:
        if g.flag_context == 'route': return result if profile.track_break_enabled else ()
        if g.flag_context == 'touch': return result if profile.note_flag_mask[2,2] else ()
        ex = bool(profile.note_flag_mask[:,0].any())
        return tuple(v for v in result if not v&1 or ex)
    if family == 8 and g.duration == 0:
        short = profile.beat_flag_mask[5 if g.zone is not None else 4]
        return tuple(v for v in result if v!=0 or short)
    return result


def probe_heads(grammar,profile):
    if not values(grammar,6,profile): return ()
    if grammar.flag_context == 'route': return (3,)
    if grammar.flag_context == 'touch': return (2,)
    return (0,1) if profile is None or profile.note_flag_mask[:,0].any() else (1,)
