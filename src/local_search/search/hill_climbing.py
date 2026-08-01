# search.hill_climbing — Hill-Climbing

import numpy as np
from search.neighbors import generate_neighbor
from evaluation.objective import compute_objective


def hill_climbing(s0, max_iter=1000, shocks=None, params=None, patience=200):
    rng = np.random.default_rng()

    current = list(s0)
    best = list(s0)
    score, _, _ = compute_objective(current, shocks, params)
    best_score = score

    history = [score]
    no_improve = 0

    for _ in range(1, max_iter + 1):
        neighbor = generate_neighbor(current, rng=rng)
        n_score, _, _ = compute_objective(neighbor, shocks, params)

        if n_score > score:
            current = neighbor
            score = n_score
            no_improve = 0
            if n_score > best_score:
                best = list(neighbor)
                best_score = n_score
        else:
            no_improve += 1

        history.append(score)

        if no_improve >= patience:
            break

    return {
        "best_state": best,
        "best_score": best_score,
        "history": history,
        "iterations": len(history),
    }
