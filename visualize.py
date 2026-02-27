import numpy as np
import matplotlib.pyplot as plt
import gbm
from matplotlib.colors import LogNorm

"""
Visualization functions for plotting GBM paths, 
path density heatmaps, strategy equity density, and more.
"""

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

def plot_hist(x, bins=80, title="Histogram"):
    x = np.asarray(x)

    plt.figure()
    plt.hist(x, bins=bins)
    plt.axvline(x.mean())
    plt.title(f"{title} | mean={x.mean():.6f}")
    plt.xlabel("value")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()

def plot_equity_density_heatmap(equity, *, bins_y=120, use_log_value=True,
                                y_clip=(0.01, 0.99),  
                                log_color=True,
                                title="Strategy value density"):
    
    ###equity: (n_sims, n_steps+1)
    
    eq = np.asarray(equity, dtype=float)
    if use_log_value:
        y = np.log(np.maximum(eq, 1e-12))
        y_label = "log(strategy value)"
    else:
        y = eq
        y_label = "strategy value"

    n_sims, n_pts = y.shape

    lo, hi = np.quantile(y, y_clip)
    y = np.clip(y, lo, hi)

    t = np.tile(np.arange(n_pts), n_sims)
    y_flat = y.reshape(-1)

    H, xedges, yedges = np.histogram2d(t, y_flat, bins=[n_pts, bins_y])

    plt.figure()
    norm = LogNorm(vmin=1, vmax=H.max()) if log_color else None

    plt.imshow(H.T, aspect="auto", origin="lower",
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
               norm=norm)

    plt.colorbar(label="count" + (" (log scale)" if log_color else ""))
    plt.xlabel("time step")
    plt.ylabel(y_label)
    plt.title(title + f" | y_clip={y_clip}, bins_y={bins_y}")
    plt.tight_layout()
    plt.show()

def plot_position_prob(position, title="P(position != 0) over time"):
    pos = np.asarray(position)
    active = (pos != 0).mean(axis=0)    
    longp = (pos > 0).mean(axis=0)
    shortp = (pos < 0).mean(axis=0)

    plt.figure()
    plt.plot(active, label="|pos|>0")
    plt.plot(longp, label="pos>0")
    plt.plot(shortp, label="pos<0")
    plt.ylim(0, 1)
    plt.xlabel("time step")
    plt.ylabel("probability")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()