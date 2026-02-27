import numpy as np
import mc_pricer as mpc 
import black_scholes as bs
from scipy.stats import norm

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
