"""MIDI diagnostics use mir_eval matching; no score is an acceptance gate."""
from collections import defaultdict
import numpy as np
from .slots import PITCHED_SLOTS, UNPITCHED_SLOTS, velocity_amplitude
from .midi import events_to_notes


def note_arrays(events,program=None):
    events=events_to_notes(events)
    notes=[e for e in events if e.kind=='note' and e.data.get('duration',0)>0
           and (program is None or e.data['slot']==program)]
    intervals=np.asarray([[e.time,e.time+e.data['duration']] for e in notes],dtype=float).reshape(-1,2)
    pitches=np.asarray([440*2**((e.data['pitch']-69)/12) for e in notes])
    velocities=np.asarray([velocity_amplitude(e.data.get('velocity_bin',0)) for e in notes])
    return intervals,pitches,velocities


def counts(reference,estimate,*,offset=True,program=None,velocity=False):
    from mir_eval.transcription import match_notes
    from mir_eval.transcription_velocity import match_notes as match_velocity
    r,p,v=note_arrays(reference,program)
    e,q,w=note_arrays(estimate,program)
    estimated_count=len(e)
    if velocity:
        # Unknown pseudo-label velocity never becomes a validation target.
        r,p,v=r[v>0],p[v>0],v[v>0]
        if not len(r):
            return (0,0,0)
        # UNKNOWN is not a measured loudness. Including zeros in mir_eval's
        # affine velocity fit can falsely turn an all-unknown prediction into
        # a perfect match when the reference has one constant velocity.
        known=w>0
        e,q,w=e[known],q[known],w[known]
    if not len(r) or not len(e):
        return (0,len(r),estimated_count)
    kwargs=dict(onset_tolerance=.05,offset_ratio=.2 if offset else None,offset_min_tolerance=.05)
    pairs=match_velocity(r,p,v,e,q,w,**kwargs) if velocity else match_notes(r,p,e,q,**kwargs)
    return len(pairs),len(r),estimated_count


def midi_counts(reference,estimate):
    result={'onset':counts(reference,estimate,offset=False),
            'onset_offset':counts(reference,estimate),
            'velocity':counts(reference,estimate,velocity=True)}
    result['instrument_onset']=tuple(sum(v) for v in zip(
        *(counts(reference,estimate,offset=False,program=p) for p in PITCHED_SLOTS)))
    # Compact unpitched events are onset-only and matched by acoustic slot.
    reference,estimate=events_to_notes(reference),events_to_notes(estimate)
    from mir_eval.util import match_events
    drum=[0,0,0]
    for slot in UNPITCHED_SLOTS:
        r=np.asarray([e.time for e in reference if e.kind=='unpitched' and e.data['slot']==slot])
        e=np.asarray([e.time for e in estimate if e.kind=='unpitched' and e.data['slot']==slot])
        drum[0]+=len(match_events(r,e,.05)) if len(r) and len(e) else 0
        drum[1]+=len(r)
        drum[2]+=len(e)
    result['unpitched_onset']=tuple(drum)
    for slot in PITCHED_SLOTS:
        result[f'class_{slot}_onset']=counts(reference,estimate,offset=False,program=slot)
    for slot in UNPITCHED_SLOTS:
        r=np.asarray([e.time for e in reference if e.kind=='unpitched' and e.data['slot']==slot])
        e=np.asarray([e.time for e in estimate if e.kind=='unpitched' and e.data['slot']==slot])
        result[f'class_{slot}_onset']=(len(match_events(r,e,.05)) if len(r) and len(e) else 0,len(r),len(e))
    result['eight_class_onset']=tuple(a+b for a,b in zip(result['instrument_onset'],drum))
    result['drum_onset']=tuple(drum)  # Existing validation report key.
    return result


def scores(counts_by_name):
    result={}
    for name,(matched,reference,estimate) in counts_by_name.items():
        precision=matched/estimate if estimate else 0.
        recall=matched/reference if reference else 0.
        result[name]=dict(precision=precision,recall=recall,
                         f1=2*matched/(reference+estimate) if reference+estimate else 0.,
                         reference_notes=reference,estimated_notes=estimate)
    return result


def envelope_confusion(reference, estimate):
    from mir_eval.transcription import match_notes
    r=[e for e in events_to_notes(reference) if e.kind=='note' and e.data['slot'] in (6,7)]
    e=[e for e in events_to_notes(estimate) if e.kind=='note' and e.data['slot'] in (6,7)]
    matrix=[[0,0],[0,0]]
    if r and e:
        def arrays(notes):
            return (np.asarray([[n.time,n.time+n.data['duration']] for n in notes]),
                    np.asarray([440*2**((n.data['pitch']-69)/12) for n in notes]))
        ri,rp=arrays(r);ei,ep=arrays(e)
        for i,j in match_notes(ri,rp,ei,ep,onset_tolerance=.05,offset_ratio=None):
            matrix[r[i].data['slot']-6][e[j].data['slot']-6]+=1
    return matrix
