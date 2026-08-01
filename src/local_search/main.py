import numpy as np
from config import DEFAULT_PARAMS
from economy.shocks import generate_shocks
from economy.simulation import simulate_economy
from evaluation.constraints import check_constraints
from evaluation.objective import compute_objective
from evaluation.metrics import compute_rmsd
from search.utils import generate_initial_state
from search.hill_climbing import hill_climbing
from search.simulated_annealing import simulated_annealing
from search.genetic_algorithm import genetic_algorithm


def main():
    params = DEFAULT_PARAMS
    T = params["T"]
    shocks = generate_shocks(T=T, seed=42)
    rng = np.random.default_rng(42)

    s0 = generate_initial_state(T=T, rng=rng)
    print(f"HC   start: {s0}")
    hc_single = hill_climbing(s0, max_iter=2000, shocks=shocks,
                               params=params, patience=500)

    best_hc = hill_climbing(s0, max_iter=2000, shocks=shocks,
                             params=params, patience=500)
    for restart in range(10):
        s_r = generate_initial_state(T=T,
                                      rng=np.random.default_rng(100 + restart))
        hc_r = hill_climbing(s_r, max_iter=2000, shocks=shocks,
                              params=params, patience=500)
        if hc_r["best_score"] > best_hc["best_score"]:
            best_hc = hc_r
    hc = best_hc

    print(f"SA   start: {s0}")
    sa = simulated_annealing(s0, max_iter=5000, T0=50.0,
                              cooling_rate=0.998, shocks=shocks,
                              params=params)

    ga = genetic_algorithm(pop_size=100, generations=300,
                            mutation_rate=0.25, shocks=shocks,
                            params=params)

    def _print_snapshots(history, max_show=15):
        n = len(history)
        if n <= max_show:
            idxs = list(range(n))
        else:
            idxs = [int(round(i * (n - 1) / (max_show - 1))) for i in range(max_show)]
            idxs = sorted(set(idxs))
        for i in idxs:
            label = "Init" if i == 0 else ("Final" if i == n - 1 else f"Iter {i}")
            print(f"  {label:>6}: J(s) = {history[i]:.4f}")

    results = []
    for name, res in [
        ("HC-single", hc_single),
        ("HC-10 restart", hc),
        ("SA", sa),
        ("GA", ga),
    ]:
        traj = simulate_economy(res["best_state"], shocks, params)
        cons = check_constraints(res["best_state"], traj, params)
        rmsd = compute_rmsd(traj)
        results.append((name, res, traj, cons, rmsd))
        print("\n") 
        print(f"=== {name} ===")
        print(f"State awal : {res['initial_state']}")
        print(f"J(s) awal  : {res['initial_score']:.4f}")
        print(f"State akhir: {res['best_state']}")
        print(f"J(s) akhir : {res['best_score']:.4f}")
        print(f"RMSD_pi    : {rmsd:.4f}")
        print(f"Feasible   : {cons['feasible']}")
        print(f"Iterations : {res['iterations']}")
        print(f"pi_T       : {traj['pi'][-1]:.2f}%")
        if not cons["feasible"]:
            v_str = {k: f"{v:.4f}" for k, v in cons["violations"].items()}
            print(f"Violations : {v_str}")
        print(f"J(s) per iterasi:")
        _print_snapshots(res["history"])

    print("\n" + "=" * 75)
    print(f"{'':<20} {'J(s)':>10} {'RMSD_pi':>10} {'Feasible':>10} {'pi_T':>8}")
    print("-" * 75)
    for name, res, traj, cons, rmsd in results:
        print(
            f"{name:<20} {res['best_score']:>10.4f} {rmsd:>10.4f} "
            f"{'YES' if cons['feasible'] else 'NO':>10} {traj['pi'][-1]:>7.2f}%"
        )
    print("-" * 75)


if __name__ == "__main__":
    main()
