"""Whole-chart LR augmentation of accepted packs, without grammar replay."""
import random
import torch
from .sequence import VOCAB_OFFSETS

PAD_AB = (7, 6, 5, 4, 3, 2, 1, 0)
PAD_DE = (0, 7, 6, 5, 4, 3, 2, 1)
SHAPE_MIRROR = (0, 2, 1, 4, 3, 9, 10, 11, 12, 5, 6, 7, 8, 14, 13, 15, 17, 16, 18)


def selected(seed, epoch, chart_id, probability):
    return random.Random(f'mirror-lr:{seed}:{epoch}:{chart_id}').random() < probability


def mirror_pack(pack):
    """Copy only spatial inputs/labels; retain time, layout and loss measures."""
    if pack.get('schema') in ('maidata.whole-chart-physical-cache.v6','maidata.whole-chart-physical-cache.v7','maidata.whole-chart-physical-cache.v8'):
        return mirror_v6(pack)
    tokens = dict(pack['tokens'])
    types, original = tokens['types'], tokens['vocab_ids']
    vids = original.clone()
    ab, de, shape = (torch.tensor(p,device=vids.device) for p in (PAD_AB,PAD_DE,SHAPE_MIRROR))
    pad_off, zone_off, shape_off = (VOCAB_OFFSETS[f] for f in (4,5,9))
    zone = original-zone_off
    is_de = (types == 5) & ((zone == 3) | (zone == 4))
    previous_de = torch.zeros_like(is_de)
    previous_de[1:] = is_de[:-1]
    for mask, perm, offset in ((types.eq(4) & ~previous_de,ab,pad_off),
                               (types.eq(4) & previous_de,de,pad_off),
                               (types.eq(9),shape,shape_off)):
        vids[mask] = offset + perm[original[mask]-offset]
    tokens['vocab_ids'] = vids
    labels = dict(pack['labels'])
    for family in (4,9):
        if family not in labels: continue
        group = dict(labels[family])
        perm = (torch.where(is_de[group['indices']][:,None],de,ab) if family==4
                else shape.expand(len(group['indices']),-1))
        group['target'] = perm.gather(1,group['target'][:,None]).squeeze(1)
        group['allowed'] = group['allowed'].gather(1,perm)
        if 'penalty' in group:
            group['penalty'] = group['penalty'].gather(1,perm[...,None].expand_as(group['penalty']))
        labels[family] = group
    return dict(pack,tokens=tokens,labels=labels)


def mirror_v6(pack):
    from .physical_schema import TOUCH_PADS, VOCAB_OFFSETS as OFFSETS
    from maidata.chart.routes import mirror_pad
    permutations={4:PAD_AB,9:SHAPE_MIRROR,19:tuple(TOUCH_PADS.index(mirror_pad(p)) for p in TOUCH_PADS)}
    tokens=dict(pack['tokens']); ids=tokens['vocab_ids'].clone(); types=tokens['types']
    labels=dict(pack['labels'])
    for family,values in permutations.items():
        perm=torch.tensor(values,device=ids.device); offset=OFFSETS[family]
        mask=types==family; ids[mask]=offset+perm[ids[mask]-offset]
        if family in labels:
            group=dict(labels[family]); group['target']=perm[group['target']]
            group['allowed']=group['allowed'][:,perm]
            if 'penalty' in group: group['penalty']=group['penalty'][:,perm]
            labels[family]=group
    tokens['vocab_ids']=ids
    return dict(pack,tokens=tokens,labels=labels)
