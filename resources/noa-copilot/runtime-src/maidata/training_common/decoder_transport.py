"""Dtype-packed IPC/H2D for shared audio/context and separate Decoder chunks."""
from dataclasses import dataclass, replace
from maidata.training_common.transport import pack_batch


@dataclass
class Packet:
    context: object
    chunks: list
    metadata: dict | None = None

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return getattr(self.context, name)

    def pin_memory(self):
        return Packet(self.context.pin_memory(), [c.pin_memory() for c in self.chunks], self.metadata)

    def to(self, device):
        return replace(self.context.to(device), chunks=self.chunks)


def pack(batch, *, metadata=None):
    return Packet(pack_batch(replace(batch, chunks=None)), [pack_batch(c) for c in batch.chunks], metadata)
