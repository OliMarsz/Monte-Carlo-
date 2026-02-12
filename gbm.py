import numpy as np

def simulate_ST(S0, r, sigma, T, n_sims, seed=None):

    if seed is not None:
        np.random.seed(seed)

    Z = np.random.normal(0.0, 1.0, size=n_sims)
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * Z
    ST = S0 * np.exp(drift + diffusion)
    return ST
