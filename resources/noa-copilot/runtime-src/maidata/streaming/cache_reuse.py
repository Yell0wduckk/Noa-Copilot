"""Conservative v4 row reuse; ambiguous witness projections use the compiler.

Active rows certify clean selected decisions within an onset. Deterministic
value rows are additionally checked against their concrete accepted candidate.
Only then can its labels be split and moved; ambiguous marker projections are
recomputed, not guessed from lossy v4 penalty bitsets. MOMENT always uses the
compiler because v4 could charge a projection to an inactive type head.
"""
import torch
from .note_schema import NoteGrammar
from .sequence import CONT_FAMILY
from .marker_schema import SLIDE, CHORD
from .version import values as version_values


def prepare(pack,tokens,profile):
    meta=pack['metadata']; grammar=NoteGrammar(meta['start'],meta['end'])
    groups={}
    for key,group in pack['labels'].items():
        for n,i in enumerate(group['indices'].tolist()):
            groups.setdefault(i,{})[key]={k:v[n] for k,v in group.items()}
    probes={}
    for i,h,w in zip(*(pack['negative_probes'][k].tolist() for k in ('indices','heads','weights'))):
        probes.setdefault(i,[]).append((h,w))
    mapping=[]; starts=[]; inserted=[]; phases=[]; eligible=[]; offset=0
    for i,token in enumerate(tokens):
        phases.append(grammar.phase)
        marker=(SLIDE if token.type==9 else CHORD if token.type in (4,5) and
                grammar.phase in ('after_pad','after_flags','after_note','after_slide') else None)
        starts.append(i+offset); inserted.append(marker)
        offset+=marker is not None; mapping.append(i+offset)
        row=groups.get(i,{})
        family=CONT_FAMILY.get(token.type,token.type)
        clean=True
        if family in (4,5,9):
            # Missing rows may have hidden an unavoidable projection. Keep
            # those positions on the exact path rather than assume zero.
            clean &= family in row and (grammar.phase in ('prefix','touch_pad','route_pad') or 'type' in row)
        for key,group in row.items():
            if 'penalty' in group and bool(group['penalty'][int(group['target'])].any()): clean=False
        if grammar.phase in ('after_pad','after_flags','after_note','after_slide','route_chain'):
            t=row.get('type')
            if t is None: clean=False
            else:
                # CHORD is the union of PAD and TOUCH. A supported zero-penalty
                # member proves the projection has no charged witness.
                # Ambiguous alternative projections are recovered individually
                # by the live compiler; they do not invalidate clean targets.
                if 9 in grammar.allowed_types():
                    # Restricted shapes can change the common head projection.
                    # Keep that case on the exact compiler path.
                    domain=version_values(grammar,9,profile)
                    if domain and domain!=grammar.allowed_values(9): clean=False

        eligible.append(clean); grammar.accept(token)
    reused={}; onset=None; count=0
    for i,token in enumerate(tokens):
        if token.type==0:
            if token.value!=96: onset=i
        if token.type!=12 or onset is None: continue
        if all(eligible[onset:i+1]):
            count+=1
            for j in range(onset+1,i+1):
                row=groups.get(j,{})
                content={k:v for k,v in row.items() if k!='type'}
                t=row.get('type')
                outer={}; checks=[]
                if t is not None:
                    transform=phases[j] in ('after_pad','after_flags','after_note','after_slide','route_chain')
                    outer_type=_type_row(t,transform=transform)
                    if outer_type is not None: outer['type']=outer_type
                    if transform:
                        p=t.get('penalty',torch.zeros(14,4,dtype=torch.bool))
                        if phases[j] in ('after_pad','after_flags') and bool(p[9].any()): checks.append(SLIDE)
                        if phases[j]!='route_chain' and not any(bool(t['allowed'][f]) and not bool(p[f].any()) for f in (4,5)):
                            checks.append(CHORD)
                if inserted[j] is not None:
                    reused[starts[j]]=(outer,probes.get(j,()),checks)
                    prefix=_prefix_row(t) if inserted[j]==CHORD else None
                    if prefix is not None: content['type']=prefix
                    reused[mapping[j]]=(content,(),())
                else:
                    reused[mapping[j]]=({**outer,**content},probes.get(j,()),checks)
        onset=None
    return reused,dict(reuse_onset_candidates=count,reuse_decision_candidates=len(reused))


def _type_row(row,*,transform):
    allowed=torch.zeros(19,dtype=torch.bool); penalties=torch.zeros(19,4,dtype=torch.bool)
    old=row['allowed']; penalty=row.get('penalty',torch.zeros(14,4,dtype=torch.bool))
    allowed[:14]=old; penalties[:14]=penalty
    target=int(row['target'])
    if transform:
        allowed[SLIDE]=old[9]; penalties[SLIDE]=penalty[9]
        allowed[CHORD]=old[4] | old[5]
        allowed[[4,5,9]]=False; penalties[[4,5,9]]=False
        target=SLIDE if target==9 else CHORD if target in (4,5) else target
    return dict(target=target,allowed=allowed,penalty=penalties,main=row['main'])


def _prefix_row(row):
    if row is None or int(row['allowed'][4:6].sum())<=1: return None
    result=_type_row(row,transform=False)
    result['allowed'][:4]=False; result['allowed'][6:]=False
    result['penalty'][:4]=False; result['penalty'][6:]=False
    return result
