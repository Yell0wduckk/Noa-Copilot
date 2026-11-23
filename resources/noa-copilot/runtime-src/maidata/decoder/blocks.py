"""MLP feed-forward blocks for the streaming decoder."""
from torch import nn


def ffn(dim, middle, dropout):
    return nn.Sequential(
        nn.Linear(dim, middle),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(middle, dim),
    )
