import numpy as np
import mc_pricer as mpc 

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

def backtest_positions(*, paths, position_fn, cost=0.001, **params):
   
    close = paths
    rets = close[:, 1:] / close[:, :-1] - 1.0       

    pos = np.asarray(position_fn(close=close, rets=rets, **params), dtype=float)
    if pos.shape != rets.shape:
        raise ValueError(f"position_fn must return shape {rets.shape}, got {pos.shape}")

    turnover = np.zeros_like(pos)
    turnover[:, 1:] = np.abs(pos[:, 1:] - pos[:, :-1])

    strat_rets = rets * pos - cost * turnover
    equity = np.cumprod(1.0 + strat_rets, axis=1)

    return {"returns": strat_rets, "equity": equity, "final_return": equity[:, -1] - 1.0, "position": pos}