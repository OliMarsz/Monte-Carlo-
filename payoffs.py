import numpy as np

def call_payoff(ST, K):
    return np.maximum(ST - K, 0.0)

def asian_call_payoff(paths, K):
    avg = paths[:, 1:].mean(axis=1)
    return np.maximum(avg - K, 0.0)

def up_and_out_call_payoff(paths, K, B):
   
    hit = (paths >= B).any(axis=1)          
    ST = paths[:, -1]
    vanilla = np.maximum(ST - K, 0.0)
    return np.where(hit, 0.0, vanilla)