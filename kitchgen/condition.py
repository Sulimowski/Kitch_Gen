from dataclasses import asdict, fields

import numpy as np

from .shapes import KINDS, PARAM_CLASSES, RANGES, shape_bbox, warp_radius

SCALE = 0.25
CLAMP = 0.03
N_INPUTS = 4

COND_SLOTS = [(kind, f.name) for kind in KINDS for f in fields(PARAM_CLASSES[kind])]
SLOT_INDEX = {slot: i for i, slot in enumerate(COND_SLOTS)}
TINE_COUNTS = list(range(RANGES["fork"]["n_tines"][0], RANGES["fork"]["n_tines"][1] + 1))
COND_DIM = len(KINDS) + len(COND_SLOTS) + len(TINE_COUNTS)


def condition_vector(kind, params):
    c = np.zeros(COND_DIM, dtype=np.float32)
    c[KINDS.index(kind)] = 1.0
    for name, value in asdict(params).items():
        lo, hi = RANGES[kind][name]
        c[len(KINDS) + SLOT_INDEX[(kind, name)]] = (value - lo) / (hi - lo)
    if kind == "fork":
        c[len(KINDS) + len(COND_SLOTS) + TINE_COUNTS.index(int(round(params.n_tines)))] = 1.0
    return c


def shape_center(kind, params):
    lo, hi = shape_bbox(kind, params)
    return (lo + hi) / 2


def network_inputs(kind, params, p):
    out = np.empty((len(p), N_INPUTS), dtype=np.float32)
    out[:, :3] = (p - shape_center(kind, params)) / SCALE
    out[:, 3] = warp_radius(kind, params, p) / SCALE
    return out
