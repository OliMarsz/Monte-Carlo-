import numpy as np
import matplotlib.pyplot as plt
from option_runner import price_option

import argparse
import importlib.util
from pathlib import Path

import gbm
from data_providers import path_to_df
from strategies import MomentumBot, momentum_strategy




def summarize_results(final_values, initial_capital=10000):
    returns = final_values / initial_capital - 1

    return {
        "mean_final_value": final_values.mean(),
        "median_final_value": np.median(final_values),
        "p5_final_value": np.quantile(final_values, 0.05),
        "p95_final_value": np.quantile(final_values, 0.95),
        "mean_return": returns.mean(),
        "median_return": np.median(returns),
        "probability_of_loss": np.mean(final_values < initial_capital),
    }



def plot_final_values(final_values):
    plt.figure(figsize=(10, 6))
    plt.hist(final_values, bins=60)
    plt.axvline(final_values.mean(), linestyle="--", label="Mean")
    plt.axvline(np.median(final_values), linestyle="--", label="Median")
    plt.title("Final Portfolio Value Across Monte Carlo Simulations")
    plt.xlabel("Final portfolio value")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_sample_paths(paths, n=25):
    plt.figure(figsize=(10, 6))

    for path in paths[:n]:
        plt.plot(path, alpha=0.5)

    plt.title(f"Sample Simulated Price Paths")
    plt.xlabel("Time step")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.show()


# -----------------------------

def run_monte_carlo(
    *,
    strategy,
    params=None,
    S0=100,
    r=0.05,
    sigma=0.20,
    T=1.0,
    n_steps=252,
    n_sims=1000,
    seed=1,
    initial_capital=10000,
    show_plots=True,
):
    params = params or {}

    paths = gbm.simulate_paths(
        S0=S0,
        r=r,
        sigma=sigma,
        T=T,
        n_steps=n_steps,
        n_sims=n_sims,
        seed=seed,
    )

    all_equity = []
    all_positions = []
    final_values = []

    for path in paths:
        data = path_to_df(path)

      
        if hasattr(strategy, "run"):
            out = strategy.run(data, **params)
        else:
            out = strategy(data, **params)

        equity = initial_capital * out["equity"]

        all_equity.append(equity.values)
        final_values.append(equity.iloc[-1])

        if "position" in out:
            all_positions.append(out["position"].values)

    all_equity = np.array(all_equity)
    final_values = np.array(final_values)
    all_positions = np.array(all_positions) if all_positions else None

    stats = summarize_results(final_values, initial_capital)

    if show_plots:
        plot_sample_paths(paths)
        plot_final_values(final_values)

    return {
        "paths": paths,
        "equity": all_equity,
        "positions": all_positions,
        "final_values": final_values,
        "stats": stats,
    }

def load_strategy_from_file(bot_file, bot_class=None, bot_function=None):
    bot_path = Path(bot_file)

    if not bot_path.exists():
        raise FileNotFoundError(f"Could not find bot file: {bot_file}")

    spec = importlib.util.spec_from_file_location(bot_path.stem, bot_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if bot_class:
        cls = getattr(module, bot_class)
        return cls()

    if bot_function:
        return getattr(module, bot_function)

    raise ValueError("Provide either --bot-class or --bot-function.")

def parse_strategy_params(param_list):
    params = {}

    for item in param_list:
        if "=" not in item:
            raise ValueError(f"Invalid param '{item}'. Use key=value format.")

        key, value = item.split("=", 1)

        try:
            if "." in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False

        params[key] = value

    return params

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Monte Carlo simulations for trading bots or option pricing."
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Strategy mode
    strategy_parser = subparsers.add_parser(
        "strategy",
        help="Run Monte Carlo simulation on a user-provided trading bot.",
    )

    strategy_parser.add_argument("--bot-file", type=str, required=True)
    strategy_parser.add_argument("--bot-class", type=str, default=None)
    strategy_parser.add_argument("--bot-function", type=str, default=None)
    strategy_parser.add_argument("--params", nargs="*", default=[])

    strategy_parser.add_argument("--S0", type=float, default=100)
    strategy_parser.add_argument("--r", type=float, default=0.05)
    strategy_parser.add_argument("--sigma", type=float, default=0.20)
    strategy_parser.add_argument("--T", type=float, default=1.0)
    strategy_parser.add_argument("--n-steps", type=int, default=252)
    strategy_parser.add_argument("--n-sims", type=int, default=1000)
    strategy_parser.add_argument("--seed", type=int, default=1)
    strategy_parser.add_argument("--initial-capital", type=float, default=10000)

    strategy_parser.add_argument("--no-plots", action="store_true")

    # Option mode
    option_parser = subparsers.add_parser(
        "option",
        help="Run Monte Carlo option pricing.",
    )

    option_parser.add_argument("--option-type", type=str, required=True)
    option_parser.add_argument("--params", nargs="*", default=[])

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "strategy":
        strategy = load_strategy_from_file(
            bot_file=args.bot_file,
            bot_class=args.bot_class,
            bot_function=args.bot_function,
        )

        params = parse_strategy_params(args.params)

        results = run_monte_carlo(
            strategy=strategy,
            params=params,
            S0=args.S0,
            r=args.r,
            sigma=args.sigma,
            T=args.T,
            n_steps=args.n_steps,
            n_sims=args.n_sims,
            seed=args.seed,
            initial_capital=args.initial_capital,
            show_plots=False,
        )

        print("\n=== Monte Carlo Strategy Results ===")
        print(f"Bot file: {args.bot_file}")
        print(f"Strategy parameters: {params}")

        for key, value in results["stats"].items():
            print(f"{key}: {value:.4f}")

        if not args.no_plots:
            plot_sample_paths(results["paths"])
            plot_final_values(results["final_values"])

    elif args.mode == "option":
        params = parse_strategy_params(args.params)

        results = price_option(
            option_type=args.option_type,
            params=params,
        )

        print("\n=== Monte Carlo Option Pricing Results ===")
        print(f"Option type: {results['option_type']}")
        print(f"Price estimate: {results['price']:.6f}")
        print(f"95% CI: [{results['ci_low']:.6f}, {results['ci_high']:.6f}]")
        print(f"CI width: {results['ci_width']:.6f}")
        print(f"Samples used: {results['n_samples']}")