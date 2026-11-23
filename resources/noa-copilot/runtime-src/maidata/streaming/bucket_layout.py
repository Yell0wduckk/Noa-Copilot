"""Worker-side views of v2 masks; no cache rewrite or new visibility rules."""
from collections import defaultdict
import numpy as np
import torch


def _tensor(value):
    return torch.from_numpy(np.asarray(value, dtype=np.int64))


def _pack_buckets(groups, length):
    queries, keys, buckets, real, positions = [], [], [], [], []
    offset = 0
    for (qw, kw), rows in sorted(groups.items()):
        q = np.zeros((len(rows), qw), dtype=np.int64)
        k = np.zeros((len(rows), kw), dtype=np.int64)
        masks = np.zeros((len(rows), qw, kw), dtype=bool)
        for i, (qi, ki, mask) in enumerate(rows):
            q[i, :len(qi)] = qi
            k[i, :len(ki)] = ki
            masks[i, :len(qi), :len(ki)] = mask
            real.extend(qi)
            positions.extend(range(offset+i*qw, offset+i*qw+len(qi)))
        queries.append(q.ravel()); keys.append(k.ravel())
        buckets.append(dict(count=len(rows), query_width=qw, key_width=kw,
                            allowed=torch.from_numpy(masks)))
        offset += q.size
    restore = np.zeros(length, dtype=np.int64)
    if real: restore[np.asarray(real)] = positions
    return dict(query_indices=_tensor(np.concatenate(queries) if queries else []),
                indices=_tensor(np.concatenate(keys) if keys else []),
                output_indices=_tensor(restore), buckets=buckets, length=length)


def optimized_layouts(charts, starts, length, audio_length, audio_valid=None):
    """Split Self into Q32; coalesce Audio rows with identical visible keys.

    Hash exact key sets, not rounded beats: this preserves cache float32
    boundaries and interior validity holes. Song offsets isolate audio banks.
    Empty audio rows are restored as zero attention (before output projection).
    """
    self_groups, audio_groups = defaultdict(list), {}
    valid = None if audio_valid is None else audio_valid.numpy().reshape(-1)
    for chart, start in zip(charts, starts):
        for name in ('self_layout', 'audio_layout'):
            b = chart.layout[name]
            ids, masks = b['indices'].numpy(), b['allowed'].numpy()
            qw = b['query_width']
            for block in range(len(ids)):
                count = min(qw, chart.cost-block*qw)
                qi = np.arange(start+block*qw, start+block*qw+count)
                if name == 'self_layout':
                    for lo in range(0, count, 32):
                        mask = masks[block, lo:min(lo+32, count)]
                        used = mask.any(0)
                        keys = ids[block, used]+start
                        width = max(32, (len(keys)+31)//32*32)
                        self_groups[32, width].append((qi[lo:lo+32], keys, mask[:, used]))
                else:
                    keys = ids[block]+chart.song*audio_length
                    mask = masks[block, :count]
                    if valid is not None: mask = mask & valid[keys][None]
                    # Group repeated masks in C, rather than hash 96 keys per token.
                    unique, inverse = np.unique(mask, axis=0, return_inverse=True)
                    for i, row in enumerate(unique):
                        selected = keys[row]
                        if not len(selected): continue
                        identity = selected.tobytes()
                        if identity not in audio_groups: audio_groups[identity] = (selected, [])
                        audio_groups[identity][1].extend(qi[inverse == i])
    self_layout = _pack_buckets(self_groups, length)
    # Every nonempty exact key set is a ragged segment, even for a single
    # query. The CPU/device fallback uses these same segments and does not
    # prebuild another padded Q/K mask bank.
    audio_layout = ragged_audio_layout(
        sorted(audio_groups.values(), key=lambda row: (row[0][0], row[0][-1])),
        length, verified_valid=valid is not None)
    result = dict(self_layout=self_layout, audio_layout=audio_layout, sinks=torch.tensor(starts))
    if 'no_history_self_layout' in charts[0].layout:
        groups = defaultdict(list)
        for chart,start in zip(charts,starts):
            layout = chart.layout['no_history_self_layout']
            qo = ko = 0
            for b in layout['buckets']:
                count,qw,kw = (b[k] for k in ('count','query_width','key_width'))
                queries = layout['query_indices'][qo:qo+count*qw].numpy().reshape(count,qw)
                keys = layout['indices'][ko:ko+count*kw].numpy().reshape(count,kw)
                for i,(qi,ki,mask) in enumerate(zip(queries,keys,b['allowed'].numpy())):
                    real = layout['output_indices'].numpy()[qi] == np.arange(qo+i*qw,qo+(i+1)*qw)
                    used = mask[real].any(0)
                    groups[qw,kw].append((qi[real]+start,ki[used]+start,mask[real][:,used]))
                qo += count*qw; ko += count*kw
        result['no_history_self_layout'] = _pack_buckets(groups,length)
    return result


def ragged_audio_layout(groups, length, *, verified_valid=False):
    """Compact (keys, queries) segments shared across charts of the same song."""
    groups = [(np.asarray(k,dtype=np.int64),np.asarray(q,dtype=np.int64))
              for k,q in groups if len(k) and len(q)]
    q_lengths = [len(q) for _,q in groups]
    k_lengths = [len(k) for k,_ in groups]
    qi = _tensor(np.concatenate([q for _,q in groups]) if groups else [])
    ki = _tensor(np.concatenate([k for k,_ in groups]) if groups else [])
    return dict(ragged_audio=True,length=length,varlen_queries=qi,varlen_keys=ki,
        q_lengths=q_lengths,k_lengths=k_lengths,
        cu_q=torch.tensor([0,*np.cumsum(q_lengths)],dtype=torch.int32),
        cu_k=torch.tensor([0,*np.cumsum(k_lengths)],dtype=torch.int32),
        max_q=max(q_lengths,default=0),max_k=max(k_lengths,default=0),
        visible_queries=qi,verified_valid=verified_valid)
