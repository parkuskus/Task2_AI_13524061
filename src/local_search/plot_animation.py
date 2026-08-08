import sys
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DEFAULT_PARAMS
from economy.shocks import generate_shocks
from search.utils import generate_initial_state
from search.hill_climbing import hill_climbing
from search.simulated_annealing import simulated_annealing
from search.genetic_algorithm import genetic_algorithm

HC_VARIANTS = ["steepest_ascent", "sideways", "stochastic", "random_restart"]
ALGOS = ["HC", "SA", "GA"]

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualization")


def record_search(algo, variant, max_iter, params, shocks, s0):
    recorded_states = []
    recorded_scores = []

    def on_iter(state, score, it):
        recorded_states.append(state)
        recorded_scores.append(score)

    if algo == "HC":
        res = hill_climbing(s0, max_iter=max_iter, shocks=shocks, params=params,
                            variant=variant, patience=500, on_iteration=on_iter)
    elif algo == "SA":
        res = simulated_annealing(s0, max_iter=max_iter, T0=50.0,
                                   cooling_rate=0.998, shocks=shocks,
                                   params=params, on_iteration=on_iter)
    else:
        res = genetic_algorithm(pop_size=50, generations=max_iter,
                                 mutation_rate=0.25, shocks=shocks,
                                 params=params, on_iteration=on_iter)

    return res, recorded_states, recorded_scores


def create_animation(algo, variant, recorded_states, recorded_scores,
                     params, shocks, max_iter, output_path):
    T = params["T"]
    quarters = list(range(1, T + 1))
    r_min = params["r_min"]
    r_max = params["r_max"]
    r0 = params["r0"]
    pi_star = params["pi_star"]

    n_frames = len(recorded_states)

    fig, (ax_j, ax_s) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(f"Local Search Animation: {algo}" +
                 (f" ({variant})" if algo == "HC" and variant else ""),
                 fontsize=13, fontweight="bold")

    ax_j.set_title("Convergence: J(s) per Iteration")
    ax_j.set_xlabel("Iteration")
    ax_j.set_ylabel("J(s)")
    ax_j.grid(alpha=0.3)

    ax_s.set_title("Current State: BI-Rate Trajectory")
    ax_s.set_xlabel("Quarter")
    ax_s.set_ylabel("BI-Rate (%)")
    ax_s.set_xticks(quarters)
    ax_s.set_ylim(r_min - 0.25, r_max + 0.25)
    ax_s.axhline(y=r0, color="gray", linestyle=":", alpha=0.5, label=f"r0={r0}%")
    ax_s.fill_between(quarters, r_min, r_max, alpha=0.05, color="green")
    ax_s.legend(fontsize=8, loc="lower left")
    ax_s.grid(alpha=0.3)

    def animate(frame_idx):
        ax_j.clear()
        ax_j.set_title("Convergence: J(s) per Iteration")
        ax_j.set_xlabel("Iteration")
        ax_j.set_ylabel("J(s)")
        ax_j.grid(alpha=0.3)

        ax_s.clear()
        ax_s.set_title(f"Current State: BI-Rate Trajectory (iter {frame_idx})")
        ax_s.set_xlabel("Quarter")
        ax_s.set_ylabel("BI-Rate (%)")
        ax_s.set_xticks(quarters)
        ax_s.set_ylim(r_min - 0.25, r_max + 0.25)
        ax_s.axhline(y=r0, color="gray", linestyle=":", alpha=0.5, label=f"r0={r0}%")
        ax_s.axhline(y=pi_star, color="green", linestyle="--", alpha=0.3, label=f"pi*={pi_star}%")
        ax_s.fill_between(quarters, r_min, r_max, alpha=0.05, color="green")
        ax_s.legend(fontsize=7, loc="lower left")
        ax_s.grid(alpha=0.3)

        scores_so_far = recorded_scores[:frame_idx + 1]
        xs = list(range(len(scores_so_far)))

        color = {"HC": "#1f77b4", "SA": "#ff7f0e", "GA": "#2ca02c"}.get(algo, "#1f77b4")

        if len(scores_so_far) > 0:
            ax_j.plot(xs, scores_so_far, color=color, linewidth=0.8)
            best_sofar = np.maximum.accumulate(scores_so_far)
            ax_j.plot(xs, best_sofar, color="red", linewidth=0.6, linestyle="--", alpha=0.6)
            ax_j.scatter([frame_idx], [scores_so_far[-1]], color=color, s=30, zorder=5)
            ax_j.set_xlim(0, max(n_frames - 1, 1))

        state = recorded_states[frame_idx]
        ax_s.plot(quarters, state, "o-", color=color, linewidth=2, markersize=6)
        ax_s.scatter([quarters[-1]], [state[-1]], color="red", s=40, zorder=5)

    ani = animation.FuncAnimation(fig, animate, frames=n_frames,
                                   interval=150, blit=False, repeat=True)

    ani.save(output_path, writer="pillow", fps=6, dpi=100)
    plt.close(fig)
    print(f"Animation saved: {output_path}")


def main():
    algo = "SA"
    variant = None
    max_iter = 200

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--algo":
            i += 1
            if i >= len(args) or args[i] not in ALGOS:
                print(f"Invalid algo: {args[i] if i < len(args) else '<missing>'}")
                print(f"Valid: {ALGOS}")
                sys.exit(1)
            algo = args[i]
        elif args[i] == "--variant":
            i += 1
            if i >= len(args) or args[i] not in HC_VARIANTS:
                print(f"Invalid variant: {args[i] if i < len(args) else '<missing>'}")
                print(f"Valid: {HC_VARIANTS}")
                sys.exit(1)
            variant = args[i]
        elif args[i] == "--max-iter":
            i += 1
            if i >= len(args):
                print("--max-iter requires a value")
                sys.exit(1)
            max_iter = int(args[i])
        elif args[i] in ("-h", "--help"):
            print("Usage: python plot_animation.py --algo <HC|SA|GA> [--variant <variant>] [--max-iter N]")
            sys.exit(0)
        else:
            print(f"Unknown: {args[i]}")
            sys.exit(1)
        i += 1

    if algo == "HC" and variant is None:
        variant = "steepest_ascent"
    if variant and algo != "HC":
        print("Warning: --variant hanya berlaku untuk HC, diabaikan.")
        variant = None

    params = DEFAULT_PARAMS
    shocks = generate_shocks(T=params["T"], seed=42)
    rng = np.random.default_rng(42)
    s0 = generate_initial_state(T=params["T"], rng=rng)

    print(f"Algo: {algo}" + (f" ({variant})" if variant else ""))
    print(f"Max iter: {max_iter}")
    print(f"Initial state: {s0}")
    print("Recording search...")

    res, recorded_states, recorded_scores = record_search(
        algo, variant, max_iter, params, shocks, s0
    )

    print(f"Recorded {len(recorded_states)} frames")
    print(f"Best J(s): {res['best_score']:.4f}")

    algo_label = algo + (f"-{variant}" if variant else "")
    os.makedirs(OUT_DIR, exist_ok=True)
    output_path = os.path.join(OUT_DIR, f"animation_{algo_label}.gif")

    create_animation(algo, variant, recorded_states, recorded_scores,
                     params, shocks, max_iter, output_path)


if __name__ == "__main__":
    main()
