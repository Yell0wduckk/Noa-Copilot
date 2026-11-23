"""Decoder chunk loss and metrics shared by joint training."""
import torch

from maidata.training_common.decoder_supervision import KEYS


class Metrics:
    def __init__(self):
        self.totals = None
        self.counts = dict.fromkeys(KEYS, 0)

    def add(self, terms):
        keys = tuple(k for k in (*KEYS, 'no_history') if getattr(terms, k, None) is not None)
        if self.totals is None:
            self.counts = dict.fromkeys(keys, 0)
        moment = getattr(terms, 'main_moment', None)
        moment = terms.main.total.detach().new_zeros(()) if moment is None else moment.detach()
        values = torch.cat([*[getattr(terms, k).total.detach().double().reshape(-1) for k in keys],
                            moment.double().reshape(-1)])
        self.totals = values if self.totals is None else self.totals + values
        for k in keys:
            self.counts[k] += getattr(terms, k).count

    def merge(self, other):
        if self.totals is None:
            self.counts = dict.fromkeys(other.counts, 0)
        if other.totals is not None:
            self.totals = other.totals if self.totals is None else self.totals + other.totals
        for k in other.counts:
            self.counts[k] += other.counts[k]

    def report(self, config):
        values = [0.] * 6 if self.totals is None else self.totals.cpu().tolist()
        sums = dict(main=values[0], physical=values[1:5])
        categories = [v / max(1, self.counts['physical']) for v in sums['physical']]
        means = dict(main=sums['main'] / max(1, self.counts['main']),
                     physical=sum(v * w for v, w in zip(categories, (1, 1, 4, 1))) / 4)
        # Common main support makes Moment + other exactly reconstruct main.
        # Other includes type decisions, note fields/flags and negative probes.
        sums.update(main_moment=values[-1], main_other=sums['main']-values[-1])
        means.update({key:sums[key] / max(1, self.counts['main'])
                      for key in ('main_moment','main_other')})
        offset = 5
        extra = 0.
        if 'no_history' in self.counts:
            sums['no_history'] = values[offset] if len(values) > offset else 0.
            means['no_history'] = sums['no_history'] / (self.counts['no_history'] or 1)
            extra += means['no_history'] * config.get('decoder_no_history_loss_weight', .5)
        return dict(loss=means['main'] + extra + means['physical'] * config['decoder_train_physical_weight'],
                    means=means, physical_categories=categories, counts=self.counts, sums=sums)


def chunk_loss(model, conditions, audio_kv, chunk, config, geometry=None, *, diagnostics=None, time_diagnostics=None):
    from maidata.streaming.training import chunk_loss as stream_loss
    return stream_loss(model, conditions, audio_kv, chunk, config, diagnostics=diagnostics, time_diagnostics=time_diagnostics)
