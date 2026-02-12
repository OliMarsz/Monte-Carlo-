import numpy as np

def call_payoff(ST, K):
    return np.maximum(ST - K, 0.0)
