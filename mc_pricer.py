import gbm
import payoffs as pf
import numpy as np

def price_european_call_mc(S0, K, r, sigma, T, n_sims, seed=None):
    ST = gbm.simulate_ST(S0, r, sigma, T, n_sims, seed)
    payoffs = pf.call_payoff(ST, K)
    price = np.exp(-r * T) * payoffs.mean()
    return price

def mc_price_CI(discounted_payoffs):
    mean = discounted_payoffs.mean()
    std = discounted_payoffs.std(ddof=1)
    n = len(discounted_payoffs)

    half_width = 1.96 * std / np.sqrt(n)
    return mean, mean - half_width, mean + half_width