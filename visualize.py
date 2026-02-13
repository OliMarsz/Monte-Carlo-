import numpy as np
import matplotlib.pyplot as plt
import gbm
from matplotlib.colors import LogNorm

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

def plot_path_density_heatmap( S0, r, sigma, T,  n_steps=252, n_sims=200_000, seed=1,
    time_bins=80, price_bins=140, clip_q=0.999):

    paths = gbm.simulate_paths(S0, r, sigma, T, n_steps, n_sims, seed=seed)
    t = np.linspace(0, T, n_steps + 1)

    tt = np.repeat(t, n_sims)
    ss = paths.T.reshape(-1)
    logS = np.log(ss)

    lo, hi = np.quantile(logS, [1-clip_q, clip_q])
    m = (logS >= lo) & (logS <= hi)
    tt, logS = tt[m], logS[m]

    H, xedges, yedges = np.histogram2d(tt, logS, bins=[time_bins, price_bins])

    plt.figure()
    plt.imshow(
        H.T + 1,  # +1 to avoid zeros for logs 
        origin="lower",
        aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        norm=LogNorm()
    )
    plt.colorbar(label="count (log scale)")
    plt.xlabel("Time (years)")
    plt.ylabel("log(Price)")
    plt.title(f"Path density (log-price, log-color), n={n_sims}")
    plt.show()

plot_path_density_heatmap(S0=100, r=0.05, sigma=0.2, T=1, n_sims=200_000)