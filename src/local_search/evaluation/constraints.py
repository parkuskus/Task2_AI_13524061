# evaluation.constraints — validasi constraint C1-C5

from config import DEFAULT_PARAMS


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
        "C1": c1_ok, "C2": c2_ok, "C3": c3_ok,
        "C4": c4_ok, "C5": c5_ok,
        "violations": violations,
    }
