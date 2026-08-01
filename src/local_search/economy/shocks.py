import numpy as np


def generate_shocks(T=8, seed=42):
    rng = np.random.default_rng(seed)
    return {
        "IS":  rng.normal(0, 0.15, size=T),
        "PC":  rng.normal(0, 0.10, size=T),
        "FX":  rng.normal(0, 1.50, size=T),
        "CA":  rng.normal(0, 0.20, size=T),
        "SBN": rng.normal(0, 0.10, size=T),
    }
