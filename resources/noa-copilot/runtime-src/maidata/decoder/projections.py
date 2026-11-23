"""Forward-local parameter packing, retaining the registered parameter keys."""
import torch
from contextlib import contextmanager
from contextvars import ContextVar

_forward_packing=ContextVar("decoder_forward_packing",default=None)


@contextmanager
def packed_projection_scope():
    """Reuse packed weights only for this forward/rollout, never across updates."""
    token=_forward_packing.set({})
    try:yield
    finally:_forward_packing.reset(token)



def linear_parameters(layers, cache=None):
    scoped=cache is None and _forward_packing.get() is not None
    if cache is None:cache=_forward_packing.get()
    parameters = tuple(p for layer in layers for p in (layer.weight, layer.bias))
    try:
        versions = tuple(p._version for p in parameters)
    except RuntimeError:
        versions = None
    key = (tuple(id(p) for p in parameters), versions, torch.is_grad_enabled())
    # Explicit scopes own packed copies for one forward/rollout. No packed
    # parameter graph is retained in token caches or across optimizer updates.
    packed = None if cache is None or (not torch.is_grad_enabled() and not scoped) else cache.setdefault('projections', {})
    if packed is not None and versions is not None and key in packed:
        return packed[key]
    result = torch.cat([layer.weight for layer in layers]), torch.cat([layer.bias for layer in layers])
    if packed is not None and versions is not None:
        packed[key] = result
    return result


def scoped_parameter_pack(parameters, namespace, build):
    """Share a differentiable parameter pack within the explicit forward scope."""
    cache = _forward_packing.get()
    if cache is None:
        return build()
    try:
        versions = tuple(p._version for p in parameters)
    except RuntimeError:
        return build()
    key = (namespace, tuple(id(p) for p in parameters), versions, torch.is_grad_enabled())
    if key not in cache:
        cache[key] = build()
    return cache[key]
