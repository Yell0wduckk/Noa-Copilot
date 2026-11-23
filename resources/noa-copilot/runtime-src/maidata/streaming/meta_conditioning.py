"""Chart-level metadata and causal history dropout for the pure E+M recipe."""
from dataclasses import dataclass
from pathlib import Path
import re
import torch
from torch import nn
from maidata.designer import (DESIGNER_ALIASES as DESIGNER_ALIASES, DESIGNER_VOCAB,
    designer_id, normalize_designer as normalize_designer)
from maidata.nn.position import sinusoidal
from .sequence import BOUNDARY


@dataclass(frozen=True)
class ChartMetadata:
    level: torch.Tensor
    difficulty: torch.Tensor
    designer: torch.Tensor


def metadata(rows, device='cpu'):
    return ChartMetadata(torch.tensor([r[0] for r in rows],dtype=torch.float32,device=device),
        torch.tensor([r[1] for r in rows],dtype=torch.long,device=device),
        torch.tensor([r[2] for r in rows],dtype=torch.long,device=device))


def load_item(item, plan, config, cache):
    from maidata.chart_cache.cache import load_chart
    from maidata.draft.conditions import read_condition
    rows, layouts = {}, {}
    for key, pack in item['packs']:
        identity = ('chart-meta', key)
        if identity not in cache:
            source = dict(pack['metadata'], **plan.charts[key])
            if source.get('chart_cache_path') and config.get('pretrain_chart_cache'):
                record = load_chart(Path(source.get('chart_cache_root',config['pretrain_chart_cache']))/source['chart_cache_path'],validate_input=False)
                source.update(record['source'])
            if config.get('pretrain_source_root'):
                source['source_path'] = str(Path(config['pretrain_source_root'])/source['song_id']/'maidata.txt')
            condition = read_condition(source, numeric_level=source.get('level'), validate_input=False)
            name = source.get('designer', '')
            path = Path(source['source_path']) if source.get('source_path') else None
            if path is not None and path.is_file():
                match = re.search(rf'^&des_{condition.difficulty_id}=([^\r\n]*)',path.read_text(encoding='utf-8-sig'),re.M)
                if match:
                    name = match[1]
            cache[identity] = (condition.level,condition.difficulty_id,
                designer_id(name,config.get('decoder_designer_vocab',DESIGNER_VOCAB)))
        rows[key] = cache[identity]
        layouts[key] = cache.setdefault(('meta-integer-attention',key,item['coverage']),{})
    return dict(item,chart_meta=rows,decoder_layouts=layouts)


class ChartMetaConditioner(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config['decoder_dim']
        self.dropout = config.get('decoder_designer_dropout',.25)
        self.level = nn.Sequential(nn.Linear(64,d),nn.SiLU(),nn.Linear(d,d))
        self.difficulty = nn.Embedding(8,d)
        self.norm = nn.LayerNorm(d,eps=config['decoder_norm_eps'])
        self.designer = nn.Embedding(len(config.get('decoder_designer_vocab',DESIGNER_VOCAB))+1,d,padding_idx=0)

    def forward(self, meta):
        ids = meta.designer
        if self.training and self.dropout:
            ids = ids.masked_fill(torch.rand(ids.shape,device=ids.device)<self.dropout,0)
        level = sinusoidal((meta.level-10.)/5.,64,10000.,1.)
        value = self.norm(self.level(level)+self.difficulty(meta.difficulty))
        # Enforce null semantics even for a loaded or externally edited zero row.
        designer = self.designer(ids).masked_fill((ids==0)[...,None],0)
        return value+designer.to(value.dtype)


def history_drop_tokens(tokens, owners, chart_count):
    """One rate per chart and one decision per integer beat; no CUDA readback."""
    device = tokens.types.device
    mode = torch.rand(chart_count,device=device)
    rate = torch.rand(chart_count,device=device)
    rate = torch.where(mode<.75,0.,torch.where(mode<.95,.15+.10*rate,.5+.5*rate))
    beats = tokens.causal_beats[0].floor().long()
    first = torch.cat((torch.ones(1,dtype=torch.bool,device=device),
        (beats[1:]!=beats[:-1]) | (owners[1:]!=owners[:-1])))
    indices = torch.arange(len(beats),device=device)
    starts = torch.where(first,indices,0).cummax(0).values
    draw = torch.rand(beats.shape,device=device)[starts]
    return draw<rate[owners]


def apply_self_history_dropout(layout, tokens, chart_count, *, dropped=None):
    """Clone visibility only, keeping sink/current prefix and worker indices intact."""
    owners = layout.get('chart_indices',torch.zeros_like(tokens.types))[0]
    dropped = history_drop_tokens(tokens,owners,chart_count) if dropped is None else dropped
    beats = tokens.causal_beats[0].floor().long()
    types = tokens.types[0]
    block = layout['self_layout']
    def mask(q, k, allowed):
        removed = dropped[k][:,None,:] & (beats[k][:,None,:]<beats[q][...,None])
        removed &= types[k][:,None,:]!=BOUNDARY
        return allowed & ~removed
    if 'buckets' in block:
        buckets=[]
        qo=ko=0
        for bucket in block['buckets']:
            count,qw,kw = (bucket[n] for n in ('count','query_width','key_width'))
            q = block['query_indices'][qo:qo+count*qw].reshape(count,qw)
            k = block['indices'][ko:ko+count*kw].reshape(count,kw)
            buckets.append(dict(bucket,allowed=mask(q,k,bucket['allowed'])))
            qo+=count*qw
            ko+=count*kw
        changed = dict(block,buckets=buckets)
    else:
        count,qw = len(block['indices']),block['query_width']
        q = torch.arange(count*qw,device=beats.device).clamp_max(len(beats)-1).reshape(count,qw)
        changed = dict(block,allowed=mask(q,block['indices'],block['allowed']))
    return dict(layout,self_layout=changed)
