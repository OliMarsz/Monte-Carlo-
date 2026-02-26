import numpy as np

def pos_momentum(*, close, lookback=20, threshold=0.02, **_):
    n_sims, n_pts = close.shape
    n_steps = n_pts - 1

    mom = np.full((n_sims, n_steps), np.nan)
    if lookback < n_steps:
        mom[:, lookback:] = close[:, 1+lookback:] / close[:, 1:-lookback] - 1.0

    sig = np.zeros((n_sims, n_steps))
    sig[mom > threshold] = 1.0
    sig[mom < -threshold] = -1.0

    pos = np.zeros_like(sig)
    pos[:, 1:] = sig[:, :-1]   
    return pos