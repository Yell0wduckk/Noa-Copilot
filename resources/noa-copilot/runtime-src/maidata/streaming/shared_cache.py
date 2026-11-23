"""One immutable, slab-packed host tensor arena shared by DataLoader workers."""
import torch
from maidata.training_common.transport import _restore


class SharedChartCache:
    def __init__(self, slab_bytes=32 * 1024**2):
        self.slab_bytes = slab_bytes
        self.buffers, self.used, self.current, self.entries = [], [], {}, {}
        self.tensor_bytes = 0

    def add(self, key, value):
        # A mirrored pack shares most tensors with its original. Preserve that
        # aliasing, without retaining the source tensors after this call.
        seen = {}
        def describe(value):
            if isinstance(value, torch.Tensor):
                if id(value) in seen:
                    return seen[id(value)]
                count, dtype = value.numel(), value.dtype
                index = self.current.get(dtype)
                if index is None or self.used[index] + count > self.buffers[index].numel():
                    index = len(self.buffers)
                    capacity = max(count, self.slab_bytes // value.element_size())
                    self.buffers.append(torch.empty(capacity, dtype=dtype).share_memory_())
                    self.used.append(0)
                    self.current[dtype] = index
                offset = self.used[index]
                self.buffers[index][offset:offset+count].copy_(value.reshape(-1))
                self.used[index] += count
                self.tensor_bytes += count * value.element_size()
                spec = ('tensor', index, offset, count, tuple(value.shape))
                seen[id(value)] = spec
                return spec
            if isinstance(value, dict):
                return ('dict', {k: describe(v) for k, v in value.items()})
            if isinstance(value, (list, tuple)):
                return ('sequence', type(value), [describe(v) for v in value])
            return ('scalar', value)
        self.entries[key] = describe(value)

    def read(self, key):
        # Views are cheap and short lived. No per-worker tensor/layout cache.
        return _restore(self.entries[key], self.buffers)

    def extend(self, other):
        """Copy a bounded worker result, preserving original/mirror aliases."""
        from maidata.training_common.audio_data import map_tensors
        for key in other.entries:
            aliases = {}
            def reuse(tensor):
                identity = (tensor.dtype, tensor.data_ptr(), tuple(tensor.shape))
                return aliases.setdefault(identity, tensor)
            self.add(key, map_tensors(other.read(key), reuse))

    @property
    def allocated_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.buffers)
