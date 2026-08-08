import sys
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


HC_VARIANTS = ["steepest_ascent", "sideways", "stochastic", "random_restart"]

HARD_LABELS = {"C1": "Batasan/diskritisasi r_t", "C2": "Defisit CA",
               "C3": "Spread SBN", "C4": "Konvergensi pi_T"}
SOFT_LABEL = "C5 (gradualisme/smoothing) -- BOLEH dilanggar, kena penalti di J(s)"


def print_usage():
    print("Usage: python main.py [--hc <variant>]")
    print(f"  Hill-Climbing variants: {', '.join(HC_VARIANTS)}")
    print("  Default variant: steepest_ascent")
    sys.exit(0)


def _print_constraints(cons, params):
    """Print per-constraint status with hard/soft distinction."""
    print("Constraints:")
    for cid in ["C1", "C2", "C3", "C4"]:
        label = HARD_LABELS.get(cid, cid)
        ok = cons[cid]
        status_icon = "PASS" if ok else "FAIL"
        print(f"  [{status_icon}] {cid}: {label}")
    c5_viol = cons["violations"]["C5"]
    soft_status = f"ok (viol={c5_viol:.4f})" if c5_viol == 0.0 else f"PENALIZED! violation = +{c5_viol:.4f}"
    print(f"  [SOFT] C5: {SOFT_LABEL}  -->  {soft_status}")


def _print_feasible(cons, params):
    """Print feasible conclusion with explanation."""
    if cons["feasible"]:
        print("Feasible   : YES -- seluruh hard constraint (C1-C4) terpenuhi.")
    else:
        print(f"Feasible   : NO  -- {len(cons['hard_failed'])} hard constraint gagal:")
        for f in cons["hard_failed"]:
            print(f"               - {f}")
    print(f"  * C5 adalah SOFT constraint (penalized) -- TIDAK mempengaruhi status feasible.")


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


def main():
    hc_variant = "steepest_ascent"
    args = sys.argv[1:]

    i = 0
    while i < len(args):
        if args[i] in ("--hc", "--hc-variant"):
            i += 1
            if i >= len(args) or args[i] not in HC_VARIANTS:
                print(f"Invalid variant: {args[i] if i < len(args) else '<missing>'}")
                print_usage()
            hc_variant = args[i]
        elif args[i] in ("-h", "--help"):
            print_usage()
        else:
            print(f"Unknown argument: {args[i]}")
            print_usage()
        i += 1

    print(f"Hill-Climbing variant: {hc_variant}")
    print("Use --hc <variant> to change.\n")

    params = DEFAULT_PARAMS
    T = params["T"]
    pi_star = params["pi_star"]
    shocks = generate_shocks(T=T, seed=42)
    rng = np.random.default_rng(42)

    s0 = generate_initial_state(T=T, rng=rng)
    print(f"HC   start: {s0}")
    hc = hill_climbing(s0, max_iter=2000, shocks=shocks,
                       params=params, variant=hc_variant,
                       patience=500, sideways_limit=100,
                       restarts=10)

    print(f"SA   start: {s0}")
    sa = simulated_annealing(s0, max_iter=5000, T0=50.0,
                              cooling_rate=0.998, shocks=shocks,
                              params=params)

    ga = genetic_algorithm(pop_size=100, generations=300,
                            mutation_rate=0.25, shocks=shocks,
                            params=params)

    results = []
    for name, res in [
        (f"HC-{hc_variant}", hc),
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
        print(f"Iterations : {res['iterations']}")

        pi_terminal = traj["pi"][-1]
        print(f"pi_T       : {pi_terminal:.2f}%  "
              f"(target: {pi_star}% +/- 1%, C4: |pi_T - {pi_star}| <= 1%)")

        _print_constraints(cons, params)
        _print_feasible(cons, params)

        print(f"J(s) per iterasi:")
        _print_snapshots(res["history"])

    print("\n" + "=" * 75)
    print("Keterangan kolom:")
    print("  J(s)     = nilai objective function (semakin besar/mendekati 0 = semakin baik)")
    print(f"  RMSD_pi  = root-mean-square deviation inflasi terhadap target {pi_star}%")
    print("  Feasible = apakah C1--C4 terpenuhi? (C5 adalah soft constraint)")
    print(f"  pi_T     = inflasi terminal di kuartal T={T}; harus mendekati {pi_star}% (lihat C4)")
    print()
    print(f"{'':<25} {'J(s)':>10} {'RMSD_pi':>10} {'Feasible':>10} {'pi_T':>8}")
    print("-" * 75)
    for name, res, traj, cons, rmsd in results:
        print(
            f"{name:<25} {res['best_score']:>10.4f} {rmsd:>10.4f} "
            f"{'YES' if cons['feasible'] else 'NO':>10} {traj['pi'][-1]:>7.2f}%"
        )
    print("-" * 75)


if __name__ == "__main__":
    main()