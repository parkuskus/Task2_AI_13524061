from config import DEFAULT_PARAMS


def check_constraints(s, trajectory, params=None):
    if params is None:
        params = DEFAULT_PARAMS

    p = params
    T = len(s)

    c1_ok = all(p["r_min"] <= rt <= p["r_max"] and rt % 0.25 == 0 for rt in s)

    CA = trajectory["CA"]
    c2_ok = all(ca >= -p["theta"] for ca in CA)
    violation_2 = max(0.0, max(-p["theta"] - ca for ca in CA))

    sbn = trajectory["sbn"]
    c3_ok = all(sbn[t] - s[t] <= p["sigma_max"] for t in range(T))
    violation_3 = max(0.0, max((sbn[t] - s[t]) - p["sigma_max"] for t in range(T)))

    pi_T = trajectory["pi"][-1]
    c4_ok = abs(pi_T - p["pi_star"]) <= 1.0
    violation_4 = max(0.0, abs(pi_T - p["pi_star"]) - 1.0)

    violation_5 = max(
        0.0,
        max(
            abs(s[t] - (p["r0"] if t == 0 else s[t - 1])) - 0.50
            for t in range(T)
        ),
    )
    c5_ok = violation_5 == 0.0

    feasible = c1_ok and c2_ok and c3_ok and c4_ok
    hard_failed = []
    if not c1_ok:
        hard_failed.append("C1 (batas/diskritisasi suku bunga)")
    if not c2_ok:
        hard_failed.append("C2 (defisit transaksi berjalan)")
    if not c3_ok:
        hard_failed.append("C3 (spread SBN)")
    if not c4_ok:
        hard_failed.append("C4 (konvergensi inflasi)")

    violations = {"C2": violation_2, "C3": violation_3,
                  "C4": violation_4, "C5": violation_5}

    return {
        "feasible": feasible,
        "hard_failed": hard_failed,
        "C1": c1_ok,
        "C2": c2_ok,
        "C3": c3_ok,
        "C4": c4_ok,
        "C5": c5_ok,
        "violations": violations,
    }
