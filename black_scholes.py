import gbm
import payoffs as pf
import numpy as np
import math
from scipy.stats import norm

'''
Black-Scholes formulas for European call and put options, as well as delta and gamma.
Note: these functions are not vectorized and are meant for single inputs. For vectorized versions, see engine.py.
'''

def norm_cdf(x):
    #Computes standard normal with error function implementation
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def d1(S0, K, r, sigma , T):
    if S0 <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        raise ValueError("Require S0>0, K>0, sigma>0, T>0.")
    return (math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))


def d2(S0, K, r, sigma, T):
    return d1(S0, K, r, sigma, T) - sigma * math.sqrt(T)

def bs_call_price(S0, K, r, sigma, T):
    D1 = d1(S0, K, r, sigma, T)
    D2 = D1 - sigma * math.sqrt(T)
    return S0 * norm_cdf(D1) - K * math.exp(-r * T) * norm_cdf(D2)


def bs_put_price(S0, K, r, sigma, T):
    D1 = d1(S0, K, r, sigma, T)
    D2 = D1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm_cdf(-D2) - S0 * norm_cdf(-D1)

def bs_delta_call(S0, K, r, sigma, T):
    D1 = d1(S0, K, r, sigma, T)
    return norm_cdf(D1)

def bs_gamma_call(*, S0, K, r, sigma, T):
    if T <= 0 or sigma <= 0:
        return 0.0

    sqrtT = np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrtT)

    pdf_d1 = norm.pdf(d1)

    return pdf_d1 / (S0 * sigma * sqrtT)