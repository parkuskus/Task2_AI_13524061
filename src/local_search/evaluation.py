# evaluation.py — Fase 2: Evaluasi State

import numpy as np
from config import DEFAULT_PARAMS
from simulation import simulate_economy


# ─── Fungsi Proksi ─────────────────────────────────────────────────────

def compute_purchasing_power(y, pi):
    return np.asarray(y) - np.asarray(pi)


def compute_current_account(de, y, eps_ca, alpha1=0.20, alpha2=0.15):
    return alpha1 * np.asarray(de) - alpha2 * np.asarray(y) + np.asarray(eps_ca)


def compute_sbn_yield(r, eps_sbn, rho_fiskal=1.50):
    return np.asarray(r) + rho_fiskal + np.asarray(eps_sbn)


# ─── Constraint Validation ─────────────────────────────────────────────

def check_constraints(s, trajectory, params=None):
    if params is None:
        params = DEFAULT_PARAMS

    p = params
    T = len(s)

    c1_ok = all(p["r_min"] <= rt <= p["r_max"] and rt % 0.25 == 0 for rt in s)

    c2_ok = True
    for t in range(T):
        r_prev = p["r0"] if t == 0 else s[t - 1]
        if abs(s[t] - r_prev) > 0.50:
            c2_ok = False
            break

    CA = trajectory["CA"]
    c3_ok = all(ca >= -p["theta"] for ca in CA)
    violation_3 = max(0.0, max(-p["theta"] - ca for ca in CA))

    sbn = trajectory["sbn"]
    c4_ok = all(sbn[t] - s[t] <= p["sigma_max"] for t in range(T))
    violation_4 = max(0.0, max((sbn[t] - s[t]) - p["sigma_max"] for t in range(T)))

    pi_T = trajectory["pi"][-1]
    c5_ok = abs(pi_T - p["pi_star"]) <= 1.0
    violation_5 = max(0.0, abs(pi_T - p["pi_star"]) - 1.0)

    feasible = c1_ok and c2_ok and c3_ok and c4_ok and c5_ok
    violations = {"C3": violation_3, "C4": violation_4, "C5": violation_5}

    return {
        "feasible": feasible,
        "C1": c1_ok,
        "C2": c2_ok,
        "C3": c3_ok,
        "C4": c4_ok,
        "C5": c5_ok,
        "violations": violations,
    }


# ─── Objective Function ────────────────────────────────────────────────

def compute_objective(s, shocks=None, params=None, mu=100):
    if params is None:
        params = DEFAULT_PARAMS

    traj = simulate_economy(s, shocks, params)
    cons = check_constraints(s, traj, params)

    p = params
    T = len(s)

    L = 0.0
    for t in range(T):
        dt = p["delta"] ** (t + 1)
        r_prev = p["r0"] if t == 0 else s[t - 1]
        L += dt * (
            p["w_pi"] * (traj["pi"][t] - p["pi_star"]) ** 2
            + p["w_y"] * (traj["y"][t] ** 2)
            + p["w_pp"] * (traj["PP"][t] - p["pp_star"]) ** 2
            + p["w_r"] * ((s[t] - r_prev) ** 2)
        )

    penalty = sum(mu * (cons["violations"][k] ** 2) for k in cons["violations"])
    return -L - penalty, traj, cons


# ─── Model Evaluation Metric ───────────────────────────────────────────

def compute_rmsd(trajectory, pi_star=2.5):
    pi = np.asarray(trajectory["pi"])
    return float(np.sqrt(np.mean((pi - pi_star) ** 2)))
