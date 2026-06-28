import numpy as np

"""
Example strategy implementations for testing and demonstration purposes."""

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

##Test strategy
def momentum_strategy(data, lookback=20, threshold=0.02, cost=0.001):
    returns = data["Close"].pct_change().fillna(0)
    momentum = data["Close"].pct_change(periods=lookback)

    signal = pd.Series(0, index=data.index)
    signal[momentum > threshold] = 1
    signal[momentum < -threshold] = -1

    position = signal.shift(1).fillna(0)

    transaction_cost = cost * position.diff().abs().fillna(0)
    strategy_returns = returns * position - transaction_cost

    equity = (1 + strategy_returns).cumprod()

    return {
        "returns": strategy_returns,
        "equity": equity,
        "position": position,
    }



class MomentumBot:
    def run(self, data, lookback=20, threshold=0.02, cost=0.001):
        returns = data["Close"].pct_change().fillna(0)
        momentum = data["Close"].pct_change(periods=lookback)

        signal = pd.Series(0, index=data.index)
        signal[momentum > threshold] = 1
        signal[momentum < -threshold] = -1

        position = signal.shift(1).fillna(0)

        transaction_cost = cost * position.diff().abs().fillna(0)
        strategy_returns = returns * position - transaction_cost

        equity = (1 + strategy_returns).cumprod()

        return {
            "returns": strategy_returns,
            "equity": equity,
            "position": position,
        }
