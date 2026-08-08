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
        new_val = round_to_bps(s[t] + delta)
        new_val = max(r_min, min(r_max, new_val))
        s_new[t] = new_val

    elif move_type == 2:
        t = rng.integers(0, T)
        delta = rng.choice([-0.25, 0.25])
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


def generate_all_neighbors(s, r_min=None, r_max=None):
    if r_min is None:
        r_min = DEFAULT_PARAMS["r_min"]
    if r_max is None:
        r_max = DEFAULT_PARAMS["r_max"]

    T = len(s)
    neighbors = []

    for t in range(T):
        for delta in [-0.25, 0.25]:
            s_new = list(s)
            new_val = round_to_bps(s[t] + delta)
            if r_min <= new_val <= r_max:
                s_new[t] = new_val
                neighbors.append(s_new)

    for t in range(T):
        for delta in [-0.25, 0.25]:
            s_new = list(s)
            valid = True
            for i in range(t, T):
                new_val = round_to_bps(s_new[i] + delta)
                if r_min <= new_val <= r_max:
                    s_new[i] = new_val
                else:
                    valid = False
                    break
            if valid:
                neighbors.append(s_new)

    unique = []
    seen = set()
    for n in neighbors:
        key = tuple(n)
        if key not in seen:
            seen.add(key)
            unique.append(n)

    return unique
