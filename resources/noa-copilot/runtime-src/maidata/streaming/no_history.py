"""Worker-side minimal-state visibility and audio-only supervision reductions."""
from collections import defaultdict
from functools import lru_cache
import numpy as np
import torch
from maidata.chart.routes import slide_motion_groups
from maidata.decoder.losses import Reduction
from maidata.time import duration_end
from .sequence import VOCAB_OFFSETS
from .bucket_layout import _pack_buckets


def enabled(config):
    from . import markers, pure_e_main
    return bool(config.get('decoder_no_history_supervision', False) and markers(config) and pure_e_main(config))


def self_layout(tokens, timing, *, self_before=8., physical_v6=True, no_placeholders=True):
    """Build sparse Q32 blocks, never a chart-length squared matrix.

    EOM/EOB states are never keys of later queries. Note states can only read
    their own onset, so active notes cannot relay earlier onset history.
    Duration IDs are full vocabulary IDs, not field-local duration values.
    """
    if physical_v6:
        from .v6_oracle import timing_v6
        timing=timing_v6(timing)
    types = tokens['types'].tolist()
    global_duration = physical_v6 or any(t>=19 for t in types)
    ids = tokens['vocab_ids'].tolist()
    ticks = (tokens['causal_beats'].double()*96).round().long().tolist()
    rows, active, onset, sink = [], [], None, 0
    span = None
    def finish_span(stop):
        if span is None or onset is None: return
        ends, continuation, wait, waiting = [], 0, 96, 0
        for i in range(span, stop):
            t = types[i]
            if t in (7, 10): continuation = 0; waiting=0
            elif t==24: waiting+=96
            elif t==23:
                from .physical_schema import VOCAB_OFFSETS as OFFSETS
                wait=waiting+ids[i]-OFFSETS[23]+1
                waiting=0
            elif t in (14, 15): continuation += 96
            elif t in (8, 11):
                duration = continuation + ids[i] - VOCAB_OFFSETS[t] + 1
                ends.append(duration_end(timing, ticks[onset], duration, slide=t == 11, **(dict(wait=wait) if global_duration else {})))
                continuation = 0
        if ends: active.append((max(ends), onset, tuple(range(span, stop))))

    for q, t in enumerate(types):
        if t == 16:
            active, onset, span, sink = [], None, None, q
        elif t == 0 and ids[q] < 96:
            onset, span = q, q+1
        elif t == 18 or (no_placeholders and (t==19 or t==4 and (q==0 or types[q-1]!=9)) and onset is not None and q>onset+1):
            finish_span(q)
            span = q
        elif t == 12:
            finish_span(q)
            span = None
        moment_query = t in (16, 12) or (t == 0 and ids[q] == 96)
        if moment_query:
            keys = {sink, q}
            if onset is not None: keys.add(onset)
            now = timing.second(ticks[q])
            active = [record for record in active if record[0] > now + 1e-9]
            for _, origin, note in active:
                keys.add(origin)
                keys.update(note)
        else:
            keys = {sink, *range(onset if onset is not None else q, q+1)}
        lower = (ticks[q]//96)*96 - round(self_before*96)
        rows.append(sorted(k for k in keys if k in (sink,q) or (k<=q and lower<=ticks[k]<=ticks[q])))
    groups = defaultdict(list)
    for lo in range(0, len(rows), 32):
        part = rows[lo:lo+32]
        keys = np.unique(np.concatenate(part))
        mask = np.zeros((len(part), len(keys)), dtype=bool)
        for i, row in enumerate(part): mask[i, np.searchsorted(keys, row)] = True
        groups[32, max(32, (len(keys)+31)//32*32)].append((np.arange(lo,lo+len(part)),keys,mask))
    return _pack_buckets(groups, len(rows))


def selected_groups(groups):
    for key, group in groups.items():
        if 'no_history' in group:
            keep=group['no_history']
            group={k:v[keep] for k,v in group.items()}
        if key == 'type':
            keep = ~(group['allowed'][:, 4] & ~group['allowed'][:, 12])
            yield key, {k:v[keep] for k,v in group.items() if k in ('indices','allowed','target','main')}
        elif key in (0, 8, 9, 11, 23, 'flag0', 'flag1', 'flag2', 'flag3'):
            yield key, group


def support(groups, probes):
    return sum(float(g['main'].sum()) for _,g in selected_groups(groups)) + float(probes['weights'].sum())


@lru_cache(maxsize=1)
def motion_membership():
    membership = torch.zeros(19,19,dtype=torch.bool)
    for members in slide_motion_groups():
        for shape in members: membership[shape,list(members)] = True
    return membership


def loss(decoder, hidden, labels):
    total = hidden[..., :0].sum().float()
    for key, group in selected_groups(labels.groups):
        h = hidden.index_select(0, group['indices'])
        logits = (decoder.type_logits(h) if key == 'type' else
                  decoder.flag_heads[int(key[4:])](h) if isinstance(key, str) else decoder.field_logits(h, key))
        lp = logits.float().masked_fill(~group['allowed'], -torch.inf).log_softmax(-1)
        if key == 9:
            target = motion_membership().to(lp.device)[group['target']]
            nll = -lp.masked_fill(~target, -torch.inf).logsumexp(-1)
        else:
            nll = -lp.gather(1, group['target'][:,None])[:,0]
        total = total + (nll*group['main']).sum()
    probes = labels.negative_probes
    for head in range(4):
        selected = probes['heads'] == head
        lp = decoder.flag_heads[head](hidden.index_select(0,probes['indices'][selected])).float().log_softmax(-1)
        total = total - (lp[:,0]*probes['weights'][selected]).sum()
    return Reduction(total, labels.counts['no_history'])
