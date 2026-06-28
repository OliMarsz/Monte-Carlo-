from app import MomentumBot, momentum_strategy

###Choose the strategy or bot to test
STRATEGY = MomentumBot()

PARAMS = {
    "lookback": 20,
    "threshold": 0.02,
    "cost": 0.001,
}

MARKET = {
    "S0": 100,
    "r": 0.05,
    "sigma": 0.20,
    "T": 1.0,
    "n_steps": 252,
    "n_sims": 1000,
    "seed": 1,
}

INITIAL_CAPITAL = 10000