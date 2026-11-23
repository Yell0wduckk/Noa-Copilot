"""Read-only tensor pack/unpack helpers for whole-chart supervision caches."""
import torch


def tensor_pack(value):
    """Plain container metadata plus one contiguous tensor per dtype; weights_only safe."""
    tensors, sizes = {}, {}
    def describe(v):
        if isinstance(v, torch.Tensor):
            offset = sizes.get(v.dtype, 0)
            sizes[v.dtype] = offset + v.numel()
            tensors.setdefault(v.dtype, []).append(v.contiguous().flatten())
            return ('tensor', v.dtype, offset, v.numel(), tuple(v.shape))
        if isinstance(v, dict):
            return ('dict', {k: describe(x) for k, x in v.items()})
        if isinstance(v, (tuple, list)):
            return ('tuple' if isinstance(v, tuple) else 'list', [describe(x) for x in v])
        return ('value', v)
    layout = describe(value)
    return dict(layout=layout, buffers={k: torch.cat(v) for k, v in tensors.items()})


def tensor_unpack(payload):
    def restore(spec):
        kind = spec[0]
        if kind == 'tensor':
            _, dtype, offset, count, shape = spec
            return payload['buffers'][dtype].narrow(0, offset, count).view(shape)
        if kind == 'dict':
            return {k: restore(v) for k, v in spec[1].items()}
        if kind == 'tuple':
            return tuple(restore(v) for v in spec[1])
        if kind == 'list':
            return [restore(v) for v in spec[1]]
        return spec[1]
    return restore(payload['layout'])
