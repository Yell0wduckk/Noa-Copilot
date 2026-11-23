"""Frozen physical coordinates and rotationally equivalent arcade judge paths."""

import json
import math
from importlib.resources import files

import numpy as np
import torch


from maidata.spine.codec import HOLD

MAX_SLIDE_STEPS = 24
ROUTES = json.loads(
    files("maidata.multitask")
    .joinpath("assets/route71_judge_steps.json")
    .read_text("utf8")
)
ROUTE_STEP_LEN = tuple(row["step_count"] for row in ROUTES)
ROUTE_ARC = tuple(row["total_arc"] for row in ROUTES)
ROUTE_RATIOS = tuple(tuple(s["arc_ratio"] for s in row["steps"]) for row in ROUTES)
RINGS = (
    ("A", 0.90, 0.40),
    ("B", 0.42, 0.28),
    ("C", 0.0, 0.36),
    ("D", 0.96, 0.20),
    ("E", 0.68, 0.22),
)


def build_z_base():
    rows = []
    for j, (ring, r, area) in enumerate(RINGS):
        for lane in range(1 if ring == "C" else 8):
            a = (lane + 0.5 * (ring in "AB")) * math.pi / 4
            rows.append(
                [
                    r * math.sin(a),
                    r * math.cos(a),
                    *(
                        v if r else 0.0
                        for k in (2, 3, 4)
                        for v in (math.sin(k * a), math.cos(k * a))
                    ),
                    r,
                    r * r,
                    *(float(i == j) for i in range(5)),
                    area,
                ]
            )
    return torch.tensor(rows, dtype=torch.float32)


Z_BASE_XY = build_z_base()[:, :2].numpy().copy()
DISK_DISTANCE_SQ = ((Z_BASE_XY[:, None] - Z_BASE_XY[None, :]) ** 2).sum(-1)
TOUCH_PALM_RADIUS = 0.50
TOUCH_PALM_ADJ = DISK_DISTANCE_SQ <= TOUCH_PALM_RADIUS**2


def active_hand_clusters(active, tick=None):
    clusters, touches = [], {}
    for _, _, slots, kind, _, pad, _, launch in active:
        if tick is not None and kind != HOLD and tick < launch:
            continue
        if kind == HOLD and pad >= 8:
            touches.setdefault(pad - 8, []).extend(slots)
        else:
            clusters.append(slots)
    while touches:
        zone, slots = touches.popitem()
        pending = [zone]
        while pending:
            adjacent = TOUCH_PALM_ADJ[pending.pop()]
            for neighbor in list(touches):
                if adjacent[neighbor]:
                    slots.extend(touches.pop(neighbor))
                    pending.append(neighbor)
        clusters.append(tuple(slots))
    return clusters


def build_uob_proj(seed=20261115):
    gen = torch.Generator().manual_seed(seed)
    blocks = []
    for _ in range(2):
        q, r = torch.linalg.qr(torch.randn(384, 384, generator=gen))
        blocks.append(q * r.diagonal().sign())
    return torch.cat(blocks)


def mod16_clock(tick):
    beat = (tick.float() / 96).remainder(16)
    angle = beat[..., None] * tick.new_tensor(
        [2 * math.pi / p for p in (16, 4, 1, 0.25)], dtype=torch.float32
    )
    return torch.stack((angle.sin(), angle.cos()), dim=-1).flatten(-2)


def _tables():
    weights = np.zeros((8, 71, 12, 33), np.float32)
    meta = np.zeros((71, 12, 4), np.float32)
    for r in ROUTES:
        rid = r["id"]
        for i, step in enumerate(r["steps"]):
            for ring, lane, w in step["zones"]:
                for start in range(8):
                    zone = (
                        16
                        if ring == "C"
                        else {"A": 0, "B": 8, "D": 17, "E": 25}[ring]
                        + (lane + start) % 8
                    )
                    weights[start, rid, i, zone] += w
            meta[rid, i] = (
                weights[0, rid, i].sum(),
                step["is_or"],
                step["is_hard_corner"],
                step["arc_ratio"],
            )
    return torch.from_numpy(weights), torch.from_numpy(meta)


ROUTE_STEP_WEIGHTS, ROUTE_STEP_META = _tables()
