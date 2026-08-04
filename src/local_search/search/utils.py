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

    # no C2, soft constraint via w_r in objective
    s = []
    for _ in range(T):
        r = round_to_bps(rng.uniform(r_min, r_max))
        s.append(r)

    return s
