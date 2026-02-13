import gbm
import mc_pricer as mpc 
import black_scholes as bs 
import numpy as np



def experiment_nomral(S0, K, r, sigma, T, n_sims,
                          seed=None, lst=[1000, 5000, 10000, 50000, 100000]):
    bs_price = bs.bs_call_price(S0, K, r, sigma, T, n_sims, seed=None)
    print(bs_price)
    for n in lst:
        mc_vector = mpc.discounted_payoffs_call(S0, K,
                                                r, sigma, T, n, seed=seed)
        mc_CI = mpc.mc_price_CI(mc_vector)
        print(mc_CI)
        if bs_price > mc_CI[1] and bs_price < mc_CI[2]:
            print("In range")
        else:
            print("Out of range")

def experiment_antithetic(S0, K, r, sigma, T,
                          seed=None, lst=[1000, 5000, 10000, 50000, 100000]):
    bs_price = bs.bs_call_price(S0, K, r, sigma, T)
    print(bs_price)
    for n in lst:
        mc_vector = mpc.discounted_payoffs_call_antithetic(S0, K,
                                                            r, sigma, T, n, seed=seed)
        mc_CI = mpc.mc_price_CI(mc_vector)
        print(mc_CI)
        if bs_price > mc_CI[1] and bs_price < mc_CI[2]:
            print("In range")
        else:
            print("Out of range")

experiment_antithetic(100, 100, .05, .2, 1)


