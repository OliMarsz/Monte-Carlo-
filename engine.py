import numpy as np
import mc_pricer as mpc 
import black_scholes as bs
from scipy.stats import norm

"""
Utility functions for running experiments, including confidence interval analysis and delta hedging.
"""

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

def bs_delta_call_vec(S, K, r, sigma, T):
    if T <= 0 or sigma <= 0:
        return np.where(S > K, 1.0, 0.0)

    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    return norm.cdf(d1)

import black_scholes as bs


def delta_hedge_short_call(*, paths, K, r, sigma_hedge, T, cost):

    n_sims, n_pts = paths.shape
    n_steps = n_pts - 1
    dt = T / n_steps

    S0 = paths[:, 0]

    premium = bs.bs_call_price(S0[0], K, r, sigma_hedge, T)

    delta = bs_delta_call_vec(S0, K, r, sigma_hedge, T)
    shares = delta.copy()

    cash = premium - shares * S0

    for i in range(n_steps):
        cash *= np.exp(r * dt)
        S = paths[:, i + 1]

        if i < n_steps - 1:
            tau = T - (i + 1) * dt
            new_delta = bs_delta_call_vec(S, K, r, sigma_hedge, tau)

            d_shares = new_delta - shares
            cash -= d_shares * S + cost * np.abs(d_shares) * S
            shares = new_delta

    ST = paths[:, -1]
    payoff = np.maximum(ST - K, 0.0)
    portfolio_value = cash + shares * ST
    hedge_error = portfolio_value - payoff

    return hedge_error