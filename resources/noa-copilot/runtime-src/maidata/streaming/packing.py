"""Whole-chart bin packing and isolated, padding-free Decoder chunk layouts."""
from functools import lru_cache
import torch


def partition(costs, budget):
    """Minimum bin count, then improve squared load imbalance deterministically.

    Oversized charts own a singleton bin. Exact feasibility searches start at
    the volume lower bound, eliminating symmetric load states. Batch membership
    is unchanged; original chart order is retained within each resulting bin.
    """
    oversized = [[i] for i,c in enumerate(costs) if c>budget]
    order = sorted((i for i,c in enumerate(costs) if c<=budget),key=lambda i:(-costs[i],i))
    if not order: return oversized
    weights = [costs[i] for i in order]
    total = sum(weights)
    # Best-fit decreasing supplies a finite upper bound.
    greedy,loads = [],[]
    for i,w in zip(order,weights):
        fit = [b for b,v in enumerate(loads) if v+w<=budget]
        b = max(fit,key=lambda b:loads[b]) if fit else len(loads)
        if b==len(loads): greedy.append([]); loads.append(0)
        greedy[b].append(i); loads[b]+=w
    bins = greedy
    for count in range(max(1,(total+budget-1)//budget),len(greedy)):
        @lru_cache(None)
        def feasible(at, state):
            if at==len(weights): return ()
            w = weights[at]; seen = set()
            for b in sorted(range(count),key=lambda b:-state[b]):
                if state[b] in seen or state[b]+w>budget: continue
                seen.add(state[b])
                next_state = list(state); next_state[b]+=w
                # Canonicalize equivalent bins for the failure memo.
                suffix = feasible(at+1,tuple(sorted(next_state)))
                if suffix is not None: return (state[b],)+suffix
            return None
        choices = feasible(0,(0,)*count)
        if choices is not None:
            bins=[[] for _ in range(count)]; loads=[0]*count
            for i,load in zip(order,choices):
                b=loads.index(load); bins[b].append(i); loads[b]+=costs[i]
            break
    loads = [sum(costs[i] for i in b) for b in bins]
    # Fixed minimum count: best improving move/swap, with no budget overflow.
    while True:
        best = None; improvement = 0
        for a in range(len(bins)):
            for b in range(a+1,len(bins)):
                for i in [None,*bins[a]]:
                    for j in [None,*bins[b]]:
                        if i is None and j is None: continue
                        if i is None and len(bins[b])==1 or j is None and len(bins[a])==1: continue
                        delta = (costs[i] if i is not None else 0)-(costs[j] if j is not None else 0)
                        x,y = loads[a]-delta,loads[b]+delta
                        gain = loads[a]**2+loads[b]**2-x*x-y*y
                        if max(x,y)<=budget and gain>improvement:
                            best=(a,b,i,j,x,y); improvement=gain
        if best is None: break
        a,b,i,j,x,y=best
        loads[a],loads[b]=x,y
        if i is not None: bins[a].remove(i); bins[b].append(i)
        if j is not None: bins[b].remove(j); bins[a].append(j)
    return sorted((sorted(b) for b in bins+oversized),key=lambda b:b[0])


def merge(charts, audio_length, audio_valid=None):
    """Concatenate real tokens; retain per-chart block tails and Sink identities."""
    from .training import Chunk,Labels
    from .network import StreamTokens
    lengths=[c.cost for c in charts]
    starts=[]; length=0
    for size in lengths: starts.append(length); length+=size
    tokens=StreamTokens(**{k:torch.cat([getattr(c.tokens,k) for c in charts],1) for k in vars(charts[0].tokens)})
    from .bucket_layout import optimized_layouts
    layout=optimized_layouts(charts,starts,length,audio_length,audio_valid)
    layout['song_indices'] = torch.cat([torch.full_like(c.tokens.types,c.song) for c in charts],1)
    layout['chart_indices'] = torch.cat([torch.full_like(c.tokens.types,int(c.indices[0])) for c in charts],1)
    layout['chart_counts'] = {int(c.indices[0]):dict(c.labels.counts) for c in charts}
    for key in ('moment_query', 'moment_pos_id', 'occupancy_id'):
        if key in charts[0].layout:
            layout[key] = torch.cat([c.layout[key].reshape(1, -1) for c in charts], 1)
    groups={}
    for key in dict.fromkeys(k for c in charts for k in c.labels.groups):
        rows=[(c.labels.groups[key],start) for c,start in zip(charts,starts) if key in c.labels.groups]
        physical=any('scale' in g for g,_ in rows)
        names=list(rows[0][0])
        if physical: names=list(dict.fromkeys([*names,'scale','penalty']))
        def field(g,start,name):
            if name=='indices': return g[name]+start
            if name=='scale' and name not in g: return torch.zeros_like(g['main'])
            if name=='penalty' and name not in g: return torch.zeros((*g['allowed'].shape,4),dtype=torch.bool)
            return g[name]
        groups[key]={name:torch.cat([field(g,start,name) for g,start in rows]) for name in names}
    probes={k:torch.cat([c.labels.negative_probes[k]+(start if k=='indices' else 0)
                        for c,start in zip(charts,starts)]) for k in charts[0].labels.negative_probes}
    counts={key:sum(c.labels.counts.get(key,0) for c in charts) for key in charts[0].labels.counts}
    return Chunk(tokens,Labels(groups,probes,counts),layout,None,torch.cat([c.indices for c in charts]))
