# utils.py — fungsi utilitas

import numpy as np
from config import DEFAULT_PARAMS


def round_to_bps(value):
    return round(value * 4) / 4


def clip_state(s, r_min=None, r_max=None):
    if r_min is None:
        r_min = DEFAULT_PARAMS["r_min"]
    if r_max is None:
        r_max = DEFAULT_PARAMS["r_max"]
    return [round_to_bps(max(r_min, min(r_max, rt))) for rt in s]


def generate_initial_state(T=8, r_min=None, r_max=None, rng=None):
    if r_min is None:
        r_min = DEFAULT_PARAMS["r_min"]
    if r_max is None:
        r_max = DEFAULT_PARAMS["r_max"]
    if rng is None:
        rng = np.random.default_rng()

    r0 = DEFAULT_PARAMS["r0"]
    deltas = [-0.50, -0.25, 0.00, 0.25, 0.50]

    s = []
    prev = r0
    for _ in range(T):
        r = prev + rng.choice(deltas)
        r = round_to_bps(max(r_min, min(r_max, r)))
        s.append(r)
        prev = r

    return s
