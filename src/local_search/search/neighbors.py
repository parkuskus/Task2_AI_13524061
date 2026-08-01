# search.neighbors — successor function (Move 1, Move 2, Move 3)

import numpy as np
from config import DEFAULT_PARAMS
from search.utils import round_to_bps, clip_state


def generate_neighbor(s, move_type=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    r_min = DEFAULT_PARAMS["r_min"]
    r_max = DEFAULT_PARAMS["r_max"]
    T = len(s)

    if move_type is None:
        move_type = rng.integers(1, 3)

    s_new = list(s)

    if move_type == 1:
        t = rng.integers(0, T)
        delta = rng.choice([-0.25, 0.25])
        r_prev = DEFAULT_PARAMS["r0"] if t == 0 else s[t - 1]
        r_next = s[t + 1] if t + 1 < T else None

        new_val = round_to_bps(s[t] + delta)
        new_val = max(r_min, min(r_max, new_val))

        if abs(new_val - r_prev) > 0.50:
            new_val = r_prev + 0.50 if new_val > r_prev else r_prev - 0.50
            new_val = round_to_bps(new_val)
            new_val = max(r_min, min(r_max, new_val))

        if r_next is not None and abs(r_next - new_val) > 0.50:
            if new_val > r_next:
                new_val = r_next + 0.50
            else:
                new_val = r_next - 0.50
            new_val = round_to_bps(new_val)
            new_val = max(r_min, min(r_max, new_val))

        s_new[t] = new_val

    elif move_type == 2:
        t = rng.integers(0, T)
        delta = rng.choice([-0.25, 0.25])

        r_prev = DEFAULT_PARAMS["r0"] if t == 0 else s[t - 1]
        new_val = round_to_bps(s[t] + delta)
        new_val = max(r_min, min(r_max, new_val))

        if abs(new_val - r_prev) > 0.50:
            delta = 0.0

        if delta != 0.0:
            for i in range(t, T):
                s_new[i] = round_to_bps(max(r_min, min(r_max, s_new[i] + delta)))

    return clip_state(s_new, r_min, r_max)


def generate_neighbor_crossover(s1, s2, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    T = len(s1)
    k = rng.integers(1, T)
    c1 = s1[:k] + s2[k:]
    c2 = s2[:k] + s1[k:]

    return clip_state(c1), clip_state(c2)
