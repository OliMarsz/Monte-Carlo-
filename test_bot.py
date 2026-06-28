import pandas as pd


class TestBot:
    def run(self, data, cost=0.001):
        returns = data["Close"].pct_change().fillna(0)

        position = pd.Series(1, index=data.index)
        strategy_returns = returns * position - cost * position.diff().abs().fillna(0)
        equity = (1 + strategy_returns).cumprod()

        return {
            "returns": strategy_returns,
            "equity": equity,
            "position": position,
        }


def test_strategy(data, leverage=1.0):
    returns = data["Close"].pct_change().fillna(0)
    strategy_returns = leverage * returns
    equity = (1 + strategy_returns).cumprod()

    return {
        "returns": strategy_returns,
        "equity": equity,
    }
        
    