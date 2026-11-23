"""Zero-initialized AdaLN modulation head (`zero_modulation`)."""

from torch import nn


def zero_modulation(source_dim, target_dim):
    head = nn.Linear(source_dim, 2 * target_dim)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    return head
