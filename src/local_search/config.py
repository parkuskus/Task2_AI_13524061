DEFAULT_PARAMS = {
    # kondisi awal
    "r0": 5.75,
    "pi0": 3.34,
    "y0": 0.00,
    "e0": 17885,
    # r_US: time-varying Fed Funds Rate
    "r_US": [3.375, 3.125, 3.125, 3.125, 3.375, 3.625, 3.875, 4.125],

    # target
    "pi_star": 2.5,
    "pp_star": 0.0,

    # parameter struktural
    "rho_y": 0.60,
    "beta": 0.30,
    "kappa": 0.11,
    "phi": 0.15,
    "rho_pi": 0.60,
    "rho_e": 0.00,
    "r_star": 2.5,
    "alpha1": 0.20,
    "alpha2": 0.15,
    "rho_fiskal": 1.50,

    # constraint
    "r_min": 3.00,
    "r_max": 8.00,
    "delta_min": 1.50,
    "theta": 3.00,
    "sigma_max": 2.50,

    # objective
    "delta": 0.95,
    "mu": 100,
    "w_pi": 1.0,
    "w_y": 0.5,
    "w_pp": 0.3,
    "w_r": 0.2,

    # horizon
    "T": 8,
}
