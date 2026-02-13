import numpy as np
import matplotlib.pyplot as plt
import gbm

def plot_gbm_paths(S0, r, sigma, T, n_steps=252, n_sims=50, seed=1, title=None):
    paths = gbm.simulate_paths(S0, r, sigma, T, n_steps, n_sims, seed=seed)
    t = np.linspace(0, T, n_steps + 1)

    plt.figure()
    plt.plot(t, paths.T)  
    plt.xlabel("Time (years)")
    plt.ylabel("Price")
    plt.title(title or f"GBM paths (n_sims={n_sims}, steps={n_steps})")
    plt.show()

def plot_paths_with_barrier(S0, r, sigma, T, B, n_steps=252, n_sims=30, seed=1):
    paths = gbm.simulate_paths(S0, r, sigma, T, n_steps, n_sims, seed=seed)
    t = np.linspace(0, T, n_steps + 1)

    hit = (paths >= B).any(axis=1)

    plt.figure()
    for i in range(n_sims):
        plt.plot(t, paths[i], alpha=0.8, linestyle="--" if hit[i] else "-")
    plt.axhline(B)  
    plt.xlabel("Time (years)")
    plt.ylabel("Price")
    plt.title(f"GBM paths with barrier B={B} (dashed = hit barrier)")
    plt.show()

plot_paths_with_barrier(100, 0.05, 0.2, 1.0, B=130, n_steps=252, n_sims=40, seed=3)