import numpy as np
import matplotlib.pyplot as plt
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

    print(f"Initial state  : {s0}")
    score0, _, _ = compute_objective(s0, shocks, params)
    print(f"Initial J(s)   : {score0:.4f}")

    hc = hill_climbing(s0, max_iter=2000, shocks=shocks,
                        params=params, patience=500)

    sa = simulated_annealing(s0, max_iter=5000, T0=50.0,
                              cooling_rate=0.998, shocks=shocks,
                              params=params)

    ga = genetic_algorithm(pop_size=100, generations=300,
                            mutation_rate=0.25, shocks=shocks,
                            params=params)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(hc["history"], color="#1f77b4", linewidth=0.8)
    ax.axhline(y=hc["best_score"], color="red", linestyle="--",
               label=f"Best = {hc['best_score']:.2f}")
    ax.set_title(f"Hill-Climbing ({hc['iterations']} evals)")
    ax.set_xlabel("Evaluasi")
    ax.set_ylabel("J(s)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(sa["history"], color="#ff7f0e", linewidth=0.5, alpha=0.7)
    best_sofar = np.maximum.accumulate(sa["history"])
    ax.plot(best_sofar, color="red", linewidth=1.0,
            label=f"Best = {sa['best_score']:.2f}")
    ax.set_title(f"Simulated Annealing ({sa['iterations']} evals)")
    ax.set_xlabel("Evaluasi")
    ax.set_ylabel("J(s)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(ga["history"], color="#2ca02c", linewidth=1.0)
    ax.axhline(y=ga["best_score"], color="red", linestyle="--",
               label=f"Best = {ga['best_score']:.2f}")
    ax.set_title(f"Genetic Algorithm ({ga['iterations']} gens)")
    ax.set_xlabel("Generasi")
    ax.set_ylabel("Best J(s) per Generasi")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(hc["history"], color="#1f77b4", alpha=0.4, linewidth=0.5,
            label="Hill-Climbing")
    ax.plot(np.maximum.accumulate(sa["history"]),
            color="#ff7f0e", linewidth=1.2, label="SA (best-so-far)")
    ax.plot(ga["history"], color="#2ca02c", linewidth=1.2,
            label="GA (best per gen)")
    ax.set_title("Perbandingan Konvergensi")
    ax.set_xlabel("Evaluasi / Generasi")
    ax.set_ylabel("J(s)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("convergence.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\n" + "=" * 70)
    print(f"{'Algoritma':<20} {'J(s)':>10} {'RMSD_pi':>10} {'Feasible':>10} {'pi_T':>8} {'Evals':>8}")
    print("-" * 70)
    for name, res in [("HC", hc), ("SA", sa), ("GA", ga)]:
        traj = simulate_economy(res["best_state"], shocks, params)
        cons = check_constraints(res["best_state"], traj, params)
        rmsd = compute_rmsd(traj)
        print(
            f"{name:<20} {res['best_score']:>10.4f} {rmsd:>10.4f} "
            f"{'YES' if cons['feasible'] else 'NO':>10} "
            f"{traj['pi'][-1]:>7.2f}% {res['iterations']:>8}"
        )
    print("-" * 70)
    print("\nPlot saved: convergence.png")


if __name__ == "__main__":
    main()
