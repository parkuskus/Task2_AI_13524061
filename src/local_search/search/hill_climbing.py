import numpy as np
from search.neighbors import generate_neighbor, generate_all_neighbors
from search.utils import generate_initial_state
from evaluation.objective import compute_objective


def hill_climbing(s0, max_iter=1000, shocks=None, params=None,
                  variant="steepest_ascent", patience=200,
                  sideways_limit=100, restarts=10, seed=None,
                  on_iteration=None):
    if variant == "random_restart":
        return _random_restart(s0, max_iter, shocks, params,
                               patience, restarts, seed, on_iteration)

    variants = {
        "steepest_ascent": _steepest_ascent,
        "sideways": _sideways,
        "stochastic": _stochastic,
    }
    if variant not in variants:
        raise ValueError(f"Unknown variant '{variant}'. "
                         f"Choose from: {list(variants.keys())} + 'random_restart'")

    return variants[variant](s0, max_iter, shocks, params, patience, sideways_limit, seed, on_iteration)


def _make_result(best_state, best_score, history, initial_state, initial_score):
    return {
        "best_state": best_state,
        "best_score": best_score,
        "history": history,
        "iterations": len(history),
        "initial_state": initial_state,
        "initial_score": initial_score,
    }


def _steepest_ascent(s0, max_iter, shocks, params, patience, _sideways, seed, on_iteration):
    current = list(s0)
    best = list(s0)
    initial_state = list(s0)
    score, _, _ = compute_objective(current, shocks, params)
    best_score = score
    initial_score = score
    history = [score]
    no_improve = 0

    if on_iteration:
        on_iteration(list(current), score, 0)

    for it in range(1, max_iter + 1):
        neighbors = generate_all_neighbors(current)
        best_n = None
        best_n_score = float("-inf")

        for n in neighbors:
            n_score, _, _ = compute_objective(n, shocks, params)
            if n_score > best_n_score:
                best_n_score = n_score
                best_n = n

        if best_n_score > score:
            current = best_n
            score = best_n_score
            no_improve = 0
            if score > best_score:
                best = list(best_n)
                best_score = score
        else:
            no_improve += 1

        history.append(score)
        if on_iteration:
            on_iteration(list(current), score, it)

        if no_improve >= patience:
            break

    return _make_result(best, best_score, history, initial_state, initial_score)


def _sideways(s0, max_iter, shocks, params, patience, sideways_limit, seed, on_iteration):
    current = list(s0)
    best = list(s0)
    initial_state = list(s0)
    score, _, _ = compute_objective(current, shocks, params)
    best_score = score
    initial_score = score
    history = [score]
    sideways_count = 0
    rng = np.random.default_rng(seed)

    if on_iteration:
        on_iteration(list(current), score, 0)

    for it in range(1, max_iter + 1):
        neighbors = generate_all_neighbors(current)
        scored = []
        best_n = None
        best_n_score = float("-inf")

        for n in neighbors:
            n_score, _, _ = compute_objective(n, shocks, params)
            scored.append((n, n_score))
            if n_score > best_n_score:
                best_n_score = n_score
                best_n = n

        if best_n_score > score:
            current = best_n
            score = best_n_score
            sideways_count = 0
            if score > best_score:
                best = list(best_n)
                best_score = score
        elif best_n_score == score and sideways_count < sideways_limit:
            sideways_neighbors = [sn for sn, sn_sc in scored if sn_sc == score and sn != current]
            if sideways_neighbors:
                current = sideways_neighbors[rng.integers(0, len(sideways_neighbors))]
            sideways_count += 1
        else:
            break

        history.append(score)
        if on_iteration:
            on_iteration(list(current), score, it)

    return _make_result(best, best_score, history, initial_state, initial_score)


def _stochastic(s0, max_iter, shocks, params, patience, _sideways, seed, on_iteration):
    rng = np.random.default_rng(seed)

    current = list(s0)
    best = list(s0)
    initial_state = list(s0)
    score, _, _ = compute_objective(current, shocks, params)
    best_score = score
    initial_score = score
    history = [score]
    no_improve = 0

    if on_iteration:
        on_iteration(list(current), score, 0)

    for it in range(1, max_iter + 1):
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
        if on_iteration:
            on_iteration(list(current), score, it)

        if no_improve >= patience:
            break

    return _make_result(best, best_score, history, initial_state, initial_score)


def _random_restart(s0, max_iter, shocks, params, patience, restarts, seed, on_iteration):
    init_state = list(s0)

    res = _steepest_ascent(s0, max_iter, shocks, params, patience, None, seed, None)
    best_state = list(res["best_state"])
    best_score = res["best_score"]
    combined_history = list(res["history"])

    for r in range(restarts):
        new_s0 = generate_initial_state(
            T=len(s0),
            rng=np.random.default_rng(None if seed is None else seed + r + 100)
        )
        res_r = _steepest_ascent(new_s0, max_iter, shocks, params, patience, None, seed, None)
        combined_history.extend(res_r["history"])

        if res_r["best_score"] > best_score:
            best_state = list(res_r["best_state"])
            best_score = res_r["best_score"]

    return {
        "best_state": best_state,
        "best_score": best_score,
        "history": combined_history,
        "iterations": len(combined_history),
        "initial_state": init_state,
        "initial_score": compute_objective(init_state, shocks, params)[0],
    }