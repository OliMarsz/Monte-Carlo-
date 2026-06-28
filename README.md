# Monte Carlo Simulation Engine

This project provides a Monte Carlo simulation framework for two use cases:

1. Testing trading bots across simulated market paths
2. Pricing options using Monte Carlo payoff simulation

The main idea is to simulate many possible future asset-price paths, run a strategy or payoff calculation on each path, and summarize the distribution of outcomes.

## Features

* Simulates stock price paths using Geometric Brownian Motion
* Runs user-provided trading bots across many simulated markets
* Supports command-line bot loading from an external Python file
* Prints basic Monte Carlo statistics such as mean, median, percentiles, and probability of loss
* Displays basic plots for simulated paths and final portfolio values
* Includes Monte Carlo option-pricing support for European, Asian, barrier, antithetic, and control-variate examples

## Installation

Clone the repository:

```bash
git clone https://github.com/OliMarsz/Monte-Carlo-.git
cd Monte-Carlo-
```

Install dependencies:

```bash
pip install numpy pandas matplotlib scipy yfinance
```

## Trading Bot Mode

A user can run a Monte Carlo simulation on their own trading bot by providing a Python file path and either a bot class or strategy function.

### Required Bot Format

Your bot must be a class with a `.run()` method:

```python
class MyBot:
    def run(self, data, lookback=20, threshold=0.02, cost=0.001):
        # data is a pandas DataFrame with a "Close" column

        return {
            "returns": strategy_returns,
            "equity": equity,
            "position": position,
        }
```

Or a function:

```python
def my_strategy(data, lookback=20, threshold=0.02, cost=0.001):
    # data is a pandas DataFrame with a "Close" column

    return {
        "returns": strategy_returns,
        "equity": equity,
        "position": position,
    }
```

The returned dictionary must include:

* `returns`: strategy returns over time
* `equity`: cumulative strategy equity over time

Optional:

* `position`: position over time

### Example Bot File

Create a file called `test_bot.py`:

```python
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
```

Run the Monte Carlo simulation:

```bash
python app.py strategy --bot-file test_bot.py --bot-class TestBot --n-sims 1000 --params cost=0.001
```

Run without plots:

```bash
python app.py strategy --bot-file test_bot.py --bot-class TestBot --n-sims 1000 --no-plots --params cost=0.001
```

Example with custom market assumptions:

```bash
python app.py strategy --bot-file test_bot.py --bot-class TestBot --S0 100 --r 0.05 --sigma 0.3 --T 1 --n-steps 252 --n-sims 5000 --params cost=0.001
```

## Option Pricing Mode

The project can also price options using Monte Carlo simulation.

European call:

```bash
python app.py option --option-type european_call --params S0=100 K=100 r=0.05 sigma=0.2 T=1 n_sims=10000
```

Asian call:

```bash
python app.py option --option-type asian_call --params S0=100 K=100 r=0.05 sigma=0.2 T=1 n_sims=10000 n_steps=252
```

Up-and-out barrier call:

```bash
python app.py option --option-type up_and_out_call --params S0=100 K=100 B=130 r=0.05 sigma=0.2 T=1 n_sims=10000 n_steps=252
```

Other supported option types may include:

* `european_call`
* `asian_call`
* `up_and_out_call`
* `european_call_antithetic`
* `european_call_control_variate`
* `asian_call_control_variate`

## Command Help

Show all commands:

```bash
python app.py --help
```

Show strategy-mode help:

```bash
python app.py strategy --help
```

Show option-mode help:

```bash
python app.py option --help
```

## Project Structure

```text
Monte-Carlo-/
├── app.py              # command-line entry point
├── gbm.py              # Geometric Brownian Motion simulation
├── data_providers.py   # converts simulated paths into DataFrames
├── mc_pricer.py        # Monte Carlo option-pricing functions
├── payoffs.py          # payoff functions
├── option_runner.py    # option-pricing command wrapper
├── strategies.py       # example strategy helpers
└── README.md
```

## Notes

This project is intended as an educational Monte Carlo simulation framework. It is not financial advice and should not be used for live trading without additional validation, testing, and risk controls.
