import gbm
import payoffs as pf
import numpy as np


def discounted_payoffs_call(*, S0, K, r, sigma, T, n_sims, seed=None):
    ST = gbm.simulate_ST(S0, r, sigma, T, n_sims, seed)
    payoff = pf.call_payoff(ST, K)
    return np.exp(-r * T) * payoff

def price_european_call_mc(*,S0, K, r, sigma, T, n_sims, seed=None):
    disc = discounted_payoffs_call(S0, K, r, sigma, T, n_sims, seed)
    return disc.mean()

def mc_price_CI(discounted_payoffs):
    mean = discounted_payoffs.mean()
    std = discounted_payoffs.std(ddof=1)
    n = len(discounted_payoffs)

    half_width = 1.96 * std / np.sqrt(n)
    return (float(mean), float(mean - half_width), float(mean + half_width))

def discounted_payoffs_call_antithetic(S0, K, r, sigma, T, n_sims, seed=None):
    #similiar to original function. Just mirroring generated z scores 
    half = n_sims // 2
    Z = np.random.default_rng(seed).standard_normal(half)

    ST1 = gbm.simulate_ST_from_Z(S0, r, sigma, T, Z)
    ST2 = gbm.simulate_ST_from_Z(S0, r, sigma, T, -Z)

    payoff = 0.5 * (pf.call_payoff(ST1, K) + pf.call_payoff(ST2, K))
    return np.exp(-r * T) * payoff

def discounted_payoffs_asian_call(*, S0, K, r, sigma, T, n_sims, n_steps, seed=None):
    paths = gbm.simulate_paths(S0, r, sigma, T, n_steps, n_sims, seed)
    payoff = pf.asian_call_payoff(paths, K)
    return np.exp(-r * T) * payoff

def discounted_payoffs_up_and_out_call(*, S0, K, B, r, sigma, T, n_sims, n_steps=252, seed=None):
    paths = gbm.simulate_paths(S0, r, sigma, T, n_steps, n_sims, seed=seed)
    payoff = pf.up_and_out_call_payoff(paths, K, B)
    return np.exp(-r * T) * payoff

def DPC_control_variate(S0, K, r, sigma, T, n_sims, seed=None):

    ST = gbm.simulate_ST(S0, r, sigma, T, n_sims, seed=seed)
    X = np.exp(-r * T) * pf.call_payoff(ST, K)
    Y = np.exp(-r * T) * ST
    Expected_Y = S0  

    #optimal regression co at Cov(X,Y)/Var(Y)
    Yc = Y - Y.mean()
    Xc = X - X.mean()
    regression_co = (Xc @ Yc) / (Yc @ Yc)

    X_cv = X - regression_co * (Y - Expected_Y)
    return X_cv