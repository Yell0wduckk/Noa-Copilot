"""Local attention tiles. Workers build bitmaps; both kernels consume the same rows."""

from functools import lru_cache
from types import FunctionType

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention.flex_attention import BlockMask, flex_attention

try:
    from torch.nn.attention.flex_attention import AuxRequest
except ImportError:  # Earlier supported Torch builds use the SDPA training fallback.
    AuxRequest = None
from torch.utils._triton import has_triton
from .codec import BOUNDARY, values

MOMENT_IDS = frozenset(values("MOMENT"))


@lru_cache(maxsize=8)
def compiled_attention(head_dim, gqa, with_lse=False):
    # Dynamo caches by code object, so two closures around the same code are insufficient.
    def attend(q, k, v, block_mask):
        if with_lse:
            out, aux = flex_attention(
                q,
                k,
                v,
                block_mask=block_mask,
                enable_gqa=gqa,
                return_aux=AuxRequest(lse=True),
            )
            return out, aux.lse
        return flex_attention(q, k, v, block_mask=block_mask, enable_gqa=gqa)

    name = f"flat_attention_d{head_dim}_gqa{int(gqa)}_lse{int(with_lse)}"
    fn = FunctionType(
        attend.__code__.replace(co_name=name),
        attend.__globals__,
        name,
        closure=attend.__closure__,
    )
    return torch.compile(fn, dynamic=True)


def build(
    rows,
    starts,
    stops,
    length,
    block,
    *,
    sink=False,
    excluded=None,
    valid_keys=None,
    extras=None,
):
    """Sparse tile list; bits[0] is the single shared all-visible tile."""
    partial = [np.full((block, (block + 7) // 8), 255, np.uint8)]
    tables, flags, active = [], [], []
    extras = extras or {}
    lanes = np.arange(block)
    for base in range(0, len(rows), block):
        r = rows[base : base + block]
        spans = [(int(starts[q]), int(stops[q])) for q in r]
        spans.extend(span for q in r for span in extras.get(q, ()))
        used = {
            j
            for a, z in spans
            if z > a
            for j in range(a // block, (z - 1) // block + 1)
        }
        if sink or not used or np.any(stops[r] <= starts[r]):
            used.add(0)
        col = np.array(sorted(used), np.int32)
        keys = (col[:, None] * block + lanes).reshape(-1)
        visible = (starts[r, None] <= keys) & (keys < stops[r, None])
        if valid_keys is not None:
            visible &= valid_keys[np.minimum(keys, length - 1)]
        if excluded is not None:
            visible &= (keys == r[:, None]) | ~excluded[np.minimum(keys, length - 1)]
        if sink:
            visible |= keys == 0
        for i, q in enumerate(r):
            for a, z in extras.get(q, ()):
                visible[i] |= (a <= keys) & (keys < z)
        visible &= keys < length
        has = visible.any(axis=1)
        active.append(has)
        # Exclusions can also empty a row. With sink=True the sink is always real.
        if not has.all() and col[0] != 0:
            col = np.insert(col, 0, 0)
            visible = np.pad(visible, ((0, 0), (block, 0)))
        tiles = np.zeros((len(col), block, block), bool)
        tiles[:, : len(r)] = visible.reshape(len(r), -1, block).transpose(1, 0, 2)
        if not has.all():
            tiles[0, np.flatnonzero(~has), 0] = True
        tiles[0, len(r) :, 0] = (
            True  # physical padding lanes are discarded after attention
        )
        keep = tiles.any(axis=(1, 2))
        tiles, col = tiles[keep], col[keep]
        full = tiles.all(axis=(1, 2))
        ids = np.zeros(len(col), np.int32)
        ids[~full] = np.arange(len(partial), len(partial) + sum(~full))
        partial.extend(np.packbits(tiles[~full], axis=-1))
        tables.append(col)
        flags.append(ids)
    return tile_layout(tables, flags, partial, np.concatenate(active), length, block)


def tile_layout(columns, tile_ids, bits, has_key, keys, block):
    nq, nk = len(columns), max(1, (keys + block - 1) // block)
    lookup = np.zeros((max(2, nq), max(2, nk)), np.int32)
    indices = np.zeros((nq, nk), np.int32)
    counts, full_counts = [], []
    for q, (col, ids) in enumerate(zip(columns, tile_ids)):
        part = ids != 0
        indices[q, : len(col)] = np.concatenate([col[part], col[~part]])
        counts.append(len(col))
        full_counts.append(int((~part).sum()))
        lookup[q, col] = ids
    bits = np.asarray(bits, np.uint8)
    if len(bits) == 1:
        bits = np.repeat(bits, 2, axis=0)
    return dict(
        indices=torch.from_numpy(indices),
        counts_host=tuple(counts),
        full_host=tuple(full_counts),
        lookup=torch.from_numpy(lookup),
        bits=torch.from_numpy(bits),
        has_key=torch.from_numpy(has_key),
        rows=len(has_key),
        keys=keys,
        block=block,
    )


def merge_layout(items, key_offsets, key_size):
    """Place chart-local query blocks consecutively and translate their key blocks."""
    block = items[0]["block"]
    columns, ids, active = [], [], []
    bit_parts = [items[0]["bits"][:1].numpy()]
    bit_offset = 0
    for item, offset in zip(items, key_offsets):
        lut = item["lookup"].numpy()
        for q, count in enumerate(item["counts_host"]):
            col = item["indices"][q, :count].numpy()
            idx = lut[q, col]
            columns.append(col + offset // block)
            ids.append(np.where(idx, idx + bit_offset, 0))
        bits = item["bits"][1:].numpy()
        bit_parts.append(bits)
        bit_offset += len(bits)
        active.append(np.pad(item["has_key"].numpy(), (0, (-item["rows"]) % block)))
    return tile_layout(
        columns, ids, np.concatenate(bit_parts), np.concatenate(active), key_size, block
    )


def layouts(
    coords,
    tokens,
    timing,
    bounds,
    history,
    protected,
    evict,
    stage,
    block,
    *,
    auxiliary=True,
    active_spans=(),
):
    ticks, _, moments, _ = coords.T
    n = len(tokens)
    rows = np.arange(n)
    left = np.searchsorted(
        ticks, ticks - 96 if stage == 0 else (ticks // 96 - history) * 96
    )
    excluded = evict | (timing & (tokens != BOUNDARY)) if stage == 0 else evict

    main_extra = {}
    if stage == 1:
        for lo, hi, slot in active_spans:
            if lo < hi and left[hi - 1] > slot:
                first = lo + np.searchsorted(left[lo:hi], slot, side="right")
                for row in range(first, hi):
                    main_extra.setdefault(row, []).append((slot, slot + 1))
    result = dict(
        main=build(
            rows,
            left,
            rows + 1,
            n,
            block,
            sink=True,
            excluded=excluded,
            extras=main_extra,
        ),
        special=torch.from_numpy(np.flatnonzero(timing)),
    )
    for stream, intervals in bounds.items():
        lo, hi = intervals.T
        result["audio_" + stream] = build(
            rows, lo, hi, max(1, int(hi.max())), block, sink=True
        )
    if stage == 1 and auxiliary:
        special = np.flatnonzero(timing)
        extra = {}
        for q, a, z in protected.tolist():
            extra.setdefault(q, []).append((a, z))
        # Keys for long-note fields must carry only their own committed input when
        # accessed by the timing readout; contextual ordinary rows can contain taps.
        active_fields = np.zeros(n, dtype=bool)
        for a, z in np.unique(protected[:, 1:], axis=0):
            if tokens[a] not in MOMENT_IDS:
                active_fields[a:z] = True
        # A timing row can become a key in later layers, including its own readout.
        result.update(
            active_fields=torch.from_numpy(np.flatnonzero(active_fields)),
            clean=build(
                rows,
                np.where(timing, rows, moments),
                rows + 1,
                n,
                block,
                sink=True,
                excluded=evict,
            ),
            read=build(special, rows, rows + 1, n, block, sink=True, extras=extra),
        )
        for stream, intervals in bounds.items():
            lo, hi = intervals.T
            result["read_audio_" + stream] = build(
                special, lo, hi, result["audio_" + stream]["keys"], block, sink=True
            )
    return result


def flex_mask(data):
    bits, lut, cols = data["bits"], data["lookup"], data["indices"]
    b, device = data["block"], bits.device
    counts, full = data["counts_host"], data["full_host"]

    def mask(batch, head, q, k):
        byte = bits[lut[q // b, k // b], q % b, (k % b) // 8].to(torch.int32)
        return ((byte >> (7 - k % 8)) & 1) != 0

    total = torch.tensor(counts, device=device, dtype=torch.int32)
    fc = torch.tensor(full, device=device, dtype=torch.int32)
    pc = total - fc
    slots = torch.arange(cols.shape[1], device=device)
    full_cols = cols.gather(1, (slots + pc[:, None]).clamp_max(cols.shape[1] - 1))
    return BlockMask.from_kv_blocks(
        pc[None, None],
        cols[None, None],
        fc[None, None],
        full_cols[None, None],
        BLOCK_SIZE=b,
        mask_mod=mask,
        seq_lengths=(data["rows"], data["keys"]),
    )


class AttentionLayout:
    def __init__(self, data, backend, query_rows=None, key_rows=None):
        self.data, self.query_rows, self.key_rows = data, query_rows, key_rows
        self.flex = None
        self.groups = []
        b = data["block"]
        bits, lut, cols = data["bits"], data["lookup"], data["indices"]
        counts = data["counts_host"]
        device = bits.device
        self.has = (
            data["has_key"]
            if query_rows is None
            else data["has_key"].index_select(0, query_rows)
        )
        if (
            device.type == "cuda"
            and backend != "sdpa"
            and b % 128 == 0
            and has_triton()
        ):
            self.flex = flex_mask(data)
            return
        lanes = torch.arange(b, device=device)
        buckets = {}
        for q, count in enumerate(counts):
            buckets.setdefault(1 << (max(1, count) - 1).bit_length(), []).append(q)
        for width, blocks in buckets.items():
            qr = torch.tensor(blocks, device=device)
            n = torch.tensor([counts[i] for i in blocks], device=device)
            cc = F.pad(
                cols.index_select(0, qr)[:, :width], (0, max(0, width - cols.shape[1]))
            )
            tile = bits[lut[qr[:, None], cc]]
            allowed = ((tile[..., lanes // 8] >> (7 - lanes % 8)) & 1).bool()
            allowed &= (torch.arange(width, device=device) < n[:, None])[
                :, :, None, None
            ]
            allowed = allowed.transpose(1, 2).reshape(len(blocks), b, width * b)
            queries = (qr[:, None] * b + lanes).flatten()
            keys = (
                (cc[:, :, None] * b + lanes)
                .reshape(len(blocks), -1)
                .clamp_max(data["keys"] - 1)
            )
            self.groups.append((queries, keys, allowed[:, None]))
        # Unpermute bucket output; padding rows remain outside the compact decoder.
        self.order = torch.argsort(torch.cat([q for q, _, _ in self.groups]))[
            : data["rows"]
        ]

    @staticmethod
    def expand(x, rows, length):
        if rows is None:
            return x[:, :, :length]
        return x.new_zeros((*x.shape[:2], length, x.shape[-1])).index_copy(2, rows, x)

    def bind_audio(self, k, v):
        k = self.expand(k, self.key_rows, self.data["keys"])
        v = self.expand(v, self.key_rows, self.data["keys"])
        if self.flex is not None:
            return lambda q, lse_heads=0: self.attend(q, k, v, None, lse_heads)

        def gather(x, keys):
            return x[0, :, keys, :].transpose(0, 1)

        batches = [(gather(k, keys), gather(v, keys)) for _, keys, _ in self.groups]
        return lambda q, lse_heads=0: self.attend(q, k, v, batches, lse_heads)

    def attend(self, q, k, v, batches, lse_heads: int = 0):
        q = self.expand(q, self.query_rows, self.data["rows"])
        gqa = q.shape[1] != k.shape[1]
        if self.flex is not None:
            result = compiled_attention(q.shape[-1], gqa, bool(lse_heads))(
                q, k, v, self.flex
            )
            if lse_heads:
                out, lse = result
                lse = lse[:, :lse_heads]
            else:
                out = result
        else:
            parts, normalizers = [], []
            for (rows, _, mask), (kg, vg) in zip(self.groups, batches):
                query = (
                    q[0, :, rows.clamp_max(q.shape[2] - 1), :]
                    .reshape(q.shape[1], kg.shape[0], -1, q.shape[-1])
                    .transpose(0, 1)
                )
                out = F.scaled_dot_product_attention(
                    query, kg, vg, attn_mask=mask, enable_gqa=gqa
                )
                if lse_heads:
                    # Only bucket-local scores. Keep the normalizer in FP32 under BF16 training.
                    with torch.autocast(query.device.type, enabled=False):
                        guided = query[:, :lse_heads].float()
                        group_size = query.shape[1] // kg.shape[1]
                        kv_heads = (lse_heads + group_size - 1) // group_size
                        keys = kg[:, :kv_heads].float()
                        if gqa and kv_heads > 1:
                            keys = keys.repeat_interleave(group_size, dim=1)[
                                :, :lse_heads
                            ]
                        scale = query.shape[-1] ** -0.5
                        scores = (guided @ keys.transpose(-1, -2)) * scale
                        local = scores.masked_fill(~mask, -torch.inf).logsumexp(-1)
                    normalizers.append(local.transpose(0, 1).flatten(1, 2))
                parts.append(out.transpose(0, 1).flatten(1, 2))
            out = torch.cat(parts, dim=1).index_select(1, self.order)[None]
            if lse_heads:
                lse = torch.cat(normalizers, dim=1).index_select(1, self.order)[None]
        if self.query_rows is not None:
            if lse_heads:
                lse = lse.index_select(2, self.query_rows)
            out = out.index_select(2, self.query_rows)
        out = out.masked_fill(~self.has[None, None, :, None], 0)
        return (out, lse.masked_fill(~self.has[None, None, :], 0)) if lse_heads else out

    def __call__(self, q, k, v, lse_heads: int = 0):
        return self.bind_audio(k, v)(q, lse_heads)
