# main.py — demo Fase 1: Persiapan dan Simulasi Ekonomi

import numpy as np
from config import DEFAULT_PARAMS
from shocks import generate_shocks
from simulation import simulate_economy


def main():
    T = DEFAULT_PARAMS["T"]
    shocks = generate_shocks(T=T, seed=42)

    s = [6.00, 6.25, 6.25, 6.00, 5.75, 5.50, 5.25, 5.25]
    traj = simulate_economy(s, shocks, DEFAULT_PARAMS)

    print("=== Fase 1: Simulasi Ekonomi ===")
    print(f"{'t':>3} {'r':>7} {'y':>7} {'pi':>7} {'de':>7} {'e':>9} {'PP':>7} {'CA':>7} {'sbn':>7}")
    print("-" * 68)
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

    print(f"\n=== Shocks (seed=42) ===")
    for key, arr in shocks.items():
        print(f"eps_{key:3s}: {np.round(arr, 4)}")


if __name__ == "__main__":
    main()
