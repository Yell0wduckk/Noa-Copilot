"""Ordered merge rules on unordered absolute pads within one pseudo group."""

import json
from functools import lru_cache
from importlib.resources import files

import torch
from torch import nn

from .codec import PAD_MAP


@lru_cache(maxsize=1)
def inventory():
    table = json.loads(
        files("maidata.multitask")
        .joinpath("assets/moment_bpe_v4.json")
        .read_text("utf8")
    )
    return (
        table["ranks"],
        tuple(map(tuple, table["words"])),
        tuple(table["mirror_word"]),
    )


RANKS, WORDS, MIRROR_WORD = inventory()
SIZE = len(WORDS)
NEXT_GROUP, END_MOMENT = SIZE, SIZE + 1


@lru_cache(maxsize=8192)
def encode_group(pads):
    parts = set(pads)
    for rank in RANKS:
        matches = [(a, b, w) for a, b, w in rank if a in parts and b in parts]
        if len(matches) == 2 and len({v for a, b, _ in matches for v in (a, b)}) < 4:
            continue
        for a, b, w in matches:
            parts.difference_update((a, b))
            parts.add(w)
    return tuple(sorted(parts))


def pad_mask(pads):
    return sum(1 << p for p in pads)


def piece_rows(row, groups):
    words = inventory()[1]
    earlier = 0
    for g, pads in enumerate(groups):
        used = 0
        for w in encode_group(pads):
            yield (row, earlier, used, g, w)
            used |= pad_mask(words[w])
        yield (
            row,
            earlier,
            used,
            g,
            NEXT_GROUP if g + 1 < len(groups) else END_MOMENT,
        )
        earlier |= used


class GeometryHead(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        words = inventory()[1]
        self.register_buffer(
            "masks", torch.tensor([pad_mask(w) for w in words]), persistent=False
        )
        self.register_buffer("mirror_pad", torch.tensor(PAD_MAP[8]), persistent=False)
        self.register_buffer(
            "bits", torch.tensor([1 << p for p in range(41)]), persistent=False
        )
        self.register_buffer(
            "mirror_piece",
            torch.tensor((*MIRROR_WORD, NEXT_GROUP, END_MOMENT)),
            persistent=False,
        )
        self.norm = nn.LayerNorm(width)
        self.context = nn.Linear(83, hidden, bias=False)
        self.state = nn.Linear(width, hidden)
        self.output = nn.Linear(hidden, SIZE + 2)

    def forward(self, state, earlier, used, group):
        past = (earlier[..., None] & self.bits) != 0
        present = (used[..., None] & self.bits) != 0

        def lanes(bits):
            return torch.stack((bits, bits[..., self.mirror_pad]), -2)

        info = torch.cat(
            (
                lanes(past),
                lanes(present),
                (group / 4)[..., None, None].expand(*group.shape, 2, 1),
            ),
            -1,
        ).to(state.dtype)
        raw = self.output(torch.nn.functional.gelu(state + self.context(info)))
        logits = (raw[..., 0, :] + raw[..., 1, self.mirror_piece]) * 0.5
        legal = (self.masks & used[..., None]) == 0
        nonempty = (used != 0)[..., None]
        return logits.masked_fill(
            ~torch.cat((legal, nonempty, nonempty), -1), -torch.inf
        )
