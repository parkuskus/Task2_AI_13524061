import numpy as np
from config import DEFAULT_PARAMS
from economy.shocks import generate_shocks


def simulate_economy(s, shocks=None, params=None):
    if shocks is None:
        shocks = generate_shocks(T=len(s), seed=42)
    if params is None:
        params = DEFAULT_PARAMS

    T = len(s)
    p = params

    y   = np.zeros(T)
    pi  = np.zeros(T)
    de  = np.zeros(T)
    e   = np.zeros(T)
    PP  = np.zeros(T)
    CA  = np.zeros(T)
    sbn = np.zeros(T)

    y_prev = p["y0"]
    pi_prev = p["pi0"]
    e_prev = p["e0"]

    for t in range(T):
        r_prev = p["r0"] if t == 0 else s[t - 1]

        de[t] = -(s[t] - p["r_US"][t]) + p["rho_e"] + shocks["FX"][t]

        y[t] = (
            p["rho_y"] * y_prev
            - p["beta"] * (r_prev - pi_prev - p["r_star"])
            + shocks["IS"][t]
        )

        pi[t] = (
            p["rho_pi"] * pi_prev
            + p["kappa"] * y[t]
            + p["phi"] * de[t]
            + shocks["PC"][t]
        )

        e[t] = e_prev * (1.0 + de[t] / 100.0)

        PP[t] = y[t] - pi[t]
        CA[t] = p["alpha1"] * de[t] - p["alpha2"] * y[t] + shocks["CA"][t]
        sbn[t] = s[t] + p["rho_fiskal"] + shocks["SBN"][t]

        y_prev = y[t]
        pi_prev = pi[t]
        e_prev = e[t]

    return {"y": y, "pi": pi, "de": de, "e": e, "PP": PP, "CA": CA, "sbn": sbn}
