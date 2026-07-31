# main.py — demo Fase 1 + Fase 2

import numpy as np
from config import DEFAULT_PARAMS
from shocks import generate_shocks
from simulation import simulate_economy
from evaluation import (
    check_constraints,
    compute_objective,
    compute_rmsd,
)


def main():
    params = DEFAULT_PARAMS
    T = params["T"]
    shocks = generate_shocks(T=T, seed=42)

    s = [6.00, 6.25, 6.25, 6.00, 5.75, 5.50, 5.25, 5.25]

    print("=== Fase 1: Simulasi Ekonomi ===")
    traj = simulate_economy(s, shocks, params)
    header = f"{'t':>3} {'r':>7} {'y':>7} {'pi':>7} {'de':>7} {'e':>9} {'PP':>7} {'CA':>7} {'sbn':>7}"
    print(header)
    print("-" * len(header))
    for t in range(T):
        print(
            f"{t+1:>3} {s[t]:>6.2f}% "
            f"{traj['y'][t]:>+6.2f} "
            f"{traj['pi'][t]:>6.2f}% "
            f"{traj['de'][t]:>+6.2f}% "
            f"{traj['e'][t]:>8.0f} "
            f"{traj['PP'][t]:>+6.2f} "
            f"{traj['CA'][t]:>+6.2f}% "
            f"{traj['sbn'][t]:>6.2f}%"
        )

    print(f"\n=== Fase 2: Evaluasi State ===")
    cons = check_constraints(s, traj, params)
    print(f"Feasible : {cons['feasible']}")
    print(f"C1 (diskritisasi)  : {'OK' if cons['C1'] else 'FAIL'}")
    print(f"C2 (smoothing)     : {'OK' if cons['C2'] else 'FAIL'}")
    print(f"C3 (diff vs Fed)   : {'OK' if cons['C3'] else 'FAIL'}")
    print(f"C4 (CA/PDB)        : {'OK' if cons['C4'] else 'FAIL'}")
    print(f"C5 (spread SBN)    : {'OK' if cons['C5'] else 'FAIL'}")
    print(f"C6 (inflasi akhir) : {'OK' if cons['C6'] else 'FAIL'}")
    print(f"Violations: {cons['violations']}")

    score, traj2, cons2 = compute_objective(s, shocks, params)
    rmsd = compute_rmsd(traj2, params["pi_star"])
    print(f"J_penalized(s) : {score:.4f}")
    print(f"RMSD_pi        : {rmsd:.4f}")
    print(f"Interpretasi   : {'SANGAT BAIK' if rmsd < 0.5 else 'BAIK' if rmsd < 1.0 else 'BURUK'}")


if __name__ == "__main__":
    main()
