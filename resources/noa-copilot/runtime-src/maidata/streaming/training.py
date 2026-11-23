"""Retained tick96 Physical loss reference; not the current training entry."""
from dataclasses import dataclass
import torch
from maidata.decoder.losses import Reduction,physical_categories
KEYS=('main','physical')

@dataclass
class StreamLoss:
    main: Reduction
    physical: Reduction
    no_history: Reduction | None = None
    main_moment: torch.Tensor | None = None  # detached logging numerator only

    def objective(self,physical_weight,no_history_weight=.5,**unused):
        auxiliary = 0 if self.no_history is None else no_history_weight*self.no_history.total/(self.no_history.count or 1)
        return self.main.mean()+auxiliary+physical_weight*(self.physical.mean()*self.physical.total.new_tensor((1,1,4,1))).sum()/4


def _loss(model, output, chunk, diagnostics, time_diagnostics, *, physical_enabled=True):
    hidden = output.hidden[0]
    zero = hidden[...,:0].sum().float()
    totals = dict(main=zero,physical=zero.expand(4))
    main_moment = zero.detach()
    for key,group in chunk.labels.groups.items():
        h = hidden.index_select(0,group['indices'])
        logits = ((model.decoder.type_logits(h) if output.type_logits is None else
                   output.type_logits[0].index_select(0,group['indices'])) if key == 'type' else
                  model.decoder.flag_heads[int(key[4:])](h) if isinstance(key,str) and key.startswith('flag') else
                  model.decoder.field_logits(h,key))
        lp = logits.float().masked_fill(~group['allowed'],-torch.inf).log_softmax(-1)
        nll = -lp.gather(1,group['target'][:,None])[:,0]
        main = (nll*group['main']).sum()
        totals['main'] = totals['main']+main
        if key == 0:  # MOMENT family, including the end-of-beat value
            main_moment = main_moment+main.detach()
        if physical_enabled and 'category_scale' in group:
            totals['physical'] = totals['physical']+(physical_categories(lp,group['penalty'],validate=False)*group['category_scale']).sum(0)
        elif physical_enabled and 'scale' in group:
            totals['physical'] = totals['physical']+(physical_categories(lp,group['penalty'],validate=False)*group['scale'][:,None]).sum(0)
        if time_diagnostics is not None and 'time_ends' in group:
            from .sequence import FAMILIES
            row = time_diagnostics.setdefault(FAMILIES[key],dict(probability_sum=0.,nll_sum=0.,count=0))
            selected=group.get('time_rows',torch.ones_like(group['time_ends']))
            values=nll.detach()[selected]; flags=group['time_ends'][selected]
            sums=values.cumsum(0)+(row.pop('_pending_nll',0.) if len(values) else 0.)
            ends = sums[flags]
            field_nll = torch.diff(ends,prepend=ends.new_zeros(1))
            row['probability_sum'] += float((-field_nll).exp().sum())
            completed=float(ends[-1]) if len(ends) else 0.
            row['nll_sum'] += completed
            row['count'] += len(ends)
            if len(values) and not bool(flags[-1]):row['_pending_nll']=float(sums[-1])-completed
        if diagnostics is not None:
            row = diagnostics.setdefault(str(key),dict(correct=0,count=0))
            row['correct'] += int(((lp.argmax(-1)==group['target']) & (group['main']>0)).sum())
            row['count'] += int((group['main']>0).sum())
    probes = chunk.labels.negative_probes
    for head in range(4):
        selected = probes['heads']==head
        h = hidden.index_select(0,probes['indices'][selected])
        lp = model.decoder.flag_heads[head](h).float().log_softmax(-1)
        totals['main'] = totals['main']-(lp[:,0]*probes['weights'][selected]).sum()
    return StreamLoss(*(Reduction(totals[k],chunk.labels.counts[k]) for k in KEYS),
                      main_moment=main_moment),output
