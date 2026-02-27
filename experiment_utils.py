import numpy as np
import mc_pricer as mpc 
import black_scholes as bs
from scipy.stats import norm
import visualize as vz 

"""
Utility functions for running experiments, including confidence interval analysis and delta hedging.
"""

def price_mc(discounted_payoffs_fn, params: dict, 
             n_sims: int, seed=None, ci=True,):
    #recreating function for easier use 
    disc = discounted_payoffs_fn(n_sims=n_sims, seed=seed, **params)

    if not ci:
        return disc.mean()

    mean, lo, hi = mpc.mc_price_CI(disc)
    return {
        "mean": mean,
        "ci_low": lo,
        "ci_high": hi,
        "ci_width": hi - lo,
        "n": len(disc),
    }


def sweep_mc(discounted_payoffs_fn, params: dict, n_list, seed=None, benchmark=None):
    out = []
    for n in n_list:
        res = price_mc(discounted_payoffs_fn, params, n_sims=n, seed=seed, ci=True)
        res["n_sims_requested"] = n
        if benchmark is not None:
            res["benchmark"] = benchmark
            res["in_range"] = (res["ci_low"] <= benchmark <= res["ci_high"])
        out.append(res)
    return out


def run_bot_mc_and_plot(*, runner, paths, initial_capital=10000,
                        entry_kwargs=None,
                        returns_key="Strategy_Returns",
                        position_key="Position",
                        bins_y=90,
                        equity_title="Strategy value density across MC sims",
                        position_title="P(in trade) over time"):
   

    entry_kwargs = entry_kwargs or {}
    outs = runner.run_paths(paths, entry_kwargs=entry_kwargs)

    n_sims, n_pts = paths.shape
    n_steps = n_pts - 1

    equity = np.zeros((n_sims, n_steps + 1))
    equity[:, 0] = initial_capital
    pos = np.zeros((n_sims, n_steps))

    for i, out in enumerate(outs):
        sr = out[returns_key]
        rets = np.asarray(sr.fillna(0.0), dtype=float)[:n_steps]
        equity[i, 1:] = initial_capital * np.cumprod(1.0 + rets)

        p = np.asarray(out[position_key].fillna(0.0), dtype=float)[:n_steps]
        pos[i] = p

    vz.plot_equity_density_heatmap(equity, bins_y=bins_y, title=equity_title)
    vz.plot_position_prob(pos, title=position_title)

    return {"outs": outs, "equity": equity, "position": pos}