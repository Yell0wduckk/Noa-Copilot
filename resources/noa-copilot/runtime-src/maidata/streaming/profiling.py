"""Opt-in nested CUDA event scopes, shared by the existing cost diagnostic."""
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
import torch

_recorder = ContextVar('streaming_recorder', default=None)


def section(name):
    recorder = _recorder.get()
    return nullcontext() if recorder is None else _section(recorder, name)


@contextmanager
def _section(recorder, name):
    with torch.profiler.record_function('streaming/' + name), recorder.section(name):
        yield


def recorded_update(function):
    @wraps(function)
    def run(*args, **kwargs):
        token = _recorder.set(kwargs.get('recorder'))
        try:
            return function(*args, **kwargs)
        finally:
            _recorder.reset(token)
    return run


class _BackwardBoundary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, timing, start, *values):
        ctx.timing, ctx.start = timing, start
        return tuple(values)

    @staticmethod
    def backward(ctx, *gradients):
        first, last, recorder, name = ctx.timing
        if ctx.start:
            first.record()
        else:
            last.record()
            recorder.pairs.setdefault(name+'_backward', []).append((first, last))
        return None, None, *gradients


def measured_call(name, function, *args):
    """CUDA event windows on both sides of a differentiable tensor operation.

    Timing is diagnostic-only. Nested backward windows overlap and must not
    be summed with the encompassing M/shared backward scopes.
    """
    recorder = _recorder.get()
    if recorder is None:
        return function(*args)
    positions = [i for i, a in enumerate(args) if isinstance(a, torch.Tensor) and a.requires_grad]
    timing = None
    if positions and torch.is_grad_enabled():
        timing = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True), recorder, name)
        wrapped = _BackwardBoundary.apply(timing, False, *(args[i] for i in positions))
        args = list(args)
        for i, value in zip(positions, wrapped): args[i] = value
    with section(name):
        result = function(*args)
    if timing is not None:
        values = result if isinstance(result, tuple) else (result,)
        values = _BackwardBoundary.apply(timing, True, *values)
        return values if isinstance(result, tuple) else values[0]
    return result
