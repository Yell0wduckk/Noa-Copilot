"""Batched dense experts without changing their checkpoint parameter layout."""
import torch
from torch.nn import functional as F
from maidata.decoder.projections import scoped_parameter_pack


def dense_expert_values(experts, x):
    """Evaluate Linear/GELU/Dropout/Linear experts on the same input rows.

    The ModuleList owns the parameters; concatenation remains in the graph so
    each original parameter receives its gradient. Dropout is still independent
    per element, but packing experts changes its random-number consumption order.
    """
    count = len(experts)
    middle = experts[0][0].out_features
    width = experts[0][3].out_features
    parameters = tuple(p for expert in experts
                       for layer in (expert[0], expert[3]) for p in (layer.weight, layer.bias))
    def pack():
        return (torch.cat([expert[0].weight for expert in experts], dim=0),
                torch.cat([expert[0].bias for expert in experts], dim=0),
                torch.stack([expert[3].weight.t() for expert in experts], dim=0),
                torch.stack([expert[3].bias for expert in experts], dim=0))
    first_weight, first_bias, second_weight, second_bias = scoped_parameter_pack(
        parameters, 'dense_experts', pack)
    hidden = F.gelu(F.linear(x, first_weight, first_bias))
    hidden = experts[0][2](hidden)
    hidden = hidden.reshape(-1, count, middle).transpose(0, 1)
    values = torch.bmm(hidden, second_weight)
    # Match Linear's autocast output dtype instead of promoting BF16 to FP32.
    values = values + second_bias.to(values.dtype)[:, None, :]
    return values.transpose(0, 1).reshape(*x.shape[:-1], count, width)
