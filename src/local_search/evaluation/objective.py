from config import DEFAULT_PARAMS
from economy.simulation import simulate_economy
from evaluation.constraints import check_constraints


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
