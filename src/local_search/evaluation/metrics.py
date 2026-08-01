import numpy as np


def compute_rmsd(trajectory, pi_star=2.5):
    pi = np.asarray(trajectory["pi"])
    return float(np.sqrt(np.mean((pi - pi_star) ** 2)))
