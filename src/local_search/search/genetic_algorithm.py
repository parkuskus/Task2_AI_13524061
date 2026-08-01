# search.genetic_algorithm — Genetic Algorithm

import numpy as np
from config import DEFAULT_PARAMS
from search.utils import generate_initial_state
from search.neighbors import generate_neighbor, generate_neighbor_crossover
from evaluation.objective import compute_objective


def _tournament_select(population, scores, k=3, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    n = len(population)
    indices = rng.choice(n, size=min(k, n), replace=False)
    best_idx = indices[0]
    for idx in indices[1:]:
        if scores[idx] > scores[best_idx]:
            best_idx = idx
    return list(population[best_idx])


def genetic_algorithm(pop_size=50, generations=200, mutation_rate=0.15,
                       shocks=None, params=None):
    if params is None:
        params = DEFAULT_PARAMS
    T = params["T"]
    rng = np.random.default_rng()

    population = [generate_initial_state(T=T, rng=rng) for _ in range(pop_size)]
    scores = [compute_objective(ind, shocks, params)[0] for ind in population]

    best_state = list(population[0])
    best_score = scores[0]
    history = []

    for gen in range(generations):
        idx_best = int(np.argmax(scores))
        elite = list(population[idx_best])

        if scores[idx_best] > best_score:
            best_state = list(population[idx_best])
            best_score = scores[idx_best]

        history.append(scores[idx_best])

        new_population = [elite]

        while len(new_population) < pop_size:
            p1 = _tournament_select(population, scores, k=3, rng=rng)
            p2 = _tournament_select(population, scores, k=3, rng=rng)
            c1, c2 = generate_neighbor_crossover(p1, p2, rng=rng)

            if rng.random() < mutation_rate:
                c1 = generate_neighbor(c1, move_type=1, rng=rng)
            if rng.random() < mutation_rate:
                c2 = generate_neighbor(c2, move_type=1, rng=rng)

            new_population.append(c1)
            if len(new_population) < pop_size:
                new_population.append(c2)

        population = new_population[:pop_size]
        scores = [compute_objective(ind, shocks, params)[0] for ind in population]

    idx_best = int(np.argmax(scores))
    if scores[idx_best] > best_score:
        best_state = list(population[idx_best])
        best_score = scores[idx_best]

    return {
        "best_state": best_state,
        "best_score": best_score,
        "history": history,
        "iterations": generations,
    }
