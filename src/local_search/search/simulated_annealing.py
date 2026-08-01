# search.simulated_annealing — Simulated Annealing

import numpy as np
import math
from search.neighbors import generate_neighbor
from evaluation.objective import compute_objective


def simulated_annealing(s0, max_iter=1000, T0=10.0, cooling_rate=0.995,
                         shocks=None, params=None):
    rng = np.random.default_rng()

    current = list(s0)
    score, _, _ = compute_objective(current, shocks, params)

    best = list(current)
    best_score = score

    T = T0
    history = [score]

    for _ in range(1, max_iter + 1):
        neighbor = generate_neighbor(current, rng=rng)
        n_score, _, _ = compute_objective(neighbor, shocks, params)

        delta = n_score - score
        if delta > 0 or rng.random() < math.exp(delta / T):
            current = neighbor
            score = n_score

        if score > best_score:
            best = list(current)
            best_score = score

        T *= cooling_rate
        history.append(score)

    return {
        "best_state": best,
        "best_score": best_score,
        "history": history,
        "iterations": len(history),
    }
