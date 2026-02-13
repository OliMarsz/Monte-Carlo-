import numpy as np


def simulate_ST(S0, r, sigma, T, n_sims, seed=None):

    if seed is not None:
        np.random.seed(seed)

    Z = np.random.normal(0.0, 1.0, size=n_sims)
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * Z
    ST = S0 * np.exp(drift + diffusion)
    return ST

def simulate_ST_from_Z(S0, r, sigma, T, Z):
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * (T**0.5) * Z
    return S0 * np.exp(drift + diffusion)

def simulate_paths(S0, r, sigma, T, n_steps, n_sims, seed=None):
  #Simple simulate_paths. Returns n_sims x n_steps+1 array

    if n_steps <= 0 or n_sims <= 0 or S0 <= 0 or sigma < 0 or T <= 0:
        raise ValueError("Require S0>0, sigma>=0, T>0, n_steps>0, n_sims>0.")

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    Z = rng.standard_normal((n_sims, n_steps))

    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z

    log_increments = drift + diffusion
    log_paths = np.cumsum(log_increments, axis=1)

    paths = np.empty((n_sims, n_steps + 1), dtype=float)
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(log_paths)
    return paths
