"""Logical-batch objective normalization for streaming whole-chart supervision."""
from maidata.decoder.losses import Reduction

KEYS = ('main', 'physical')


def objective(terms, counts, config):
    """All chunks and accumulated groups use the same logical-batch denominators."""
    from maidata.streaming.training import StreamLoss
    extra = getattr(terms, 'no_history', None)
    normalized = StreamLoss(*(Reduction(getattr(terms, k).total, counts[k]) for k in ('main', 'physical')),
        no_history=None if extra is None else Reduction(extra.total, counts['no_history']))
    return normalized.objective(
        physical_weight=config['decoder_train_physical_weight'],
        no_history_weight=config.get('decoder_no_history_loss_weight', .5),
    )
