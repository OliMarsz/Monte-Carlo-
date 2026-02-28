import gbm
import mc_pricer as mpc 
import black_scholes as bs 
import numpy as np
import engine as eng
import strategies as strat
import visualize as vz 
from Experiment_Bot.MTbot import MT_Model
from mc_runner import MCRunner
from experiment_utils import run_bot_mc_and_plot

### standard variable numbers for testing 
S0, K, r, sigma, T, n_sims = 100, 100, 0.05, 0.2, 1.0, 2000
n_steps, lookback, threshold, cost, seed = 252, 20, 0.02, 0.0, 1
sigma_hedge, sigma_true, steps_list = 0.2, 0.2, [12, 52, 252]
initial_capital = 10000
vol_list = [0.1, 0.2, 0.4]
symbol, start_date, end_date = "FAKE", "2020-01-01", "2021-01-01"
lookback_period, threshold = 20, 0.02
entrypoint, apis, quiet = "run_complete_backtest", ["yfinance.download"], True

def run_experiment_CI(discounted_payoffs_fn, S0,
                    K, r, sigma, T, 
                   seed=None, lst=None, benchmark_fn=None,
                     label = "Confidence Interval",**kwargs):
    
    #Use discounted_payoffs_fn to input func such as european normal or antithetic version
    
    #Turn on benchmark for asian call options 
    
    if lst is None:
        lst = [1000, 5000, 10000, 50000, 100000]

    print(f"\n=== {label} ===")

    bench = None
    if benchmark_fn is not None:
        bench = benchmark_fn(S0, K, r, sigma, T)
        print("Benchmark:", bench)

    for n in lst:
        disc = discounted_payoffs_fn(
            S0=S0,
            K=K,
            r=r,
            sigma=sigma,
            T=T,
            n_sims=n,
            seed=seed,
            **kwargs
                )
        mean, lo, hi = mpc.mc_price_CI(disc)

        width = hi - lo
        print(f"n={n:<7d}  mean={mean:.6f}  CI=[{lo:.6f}, {hi:.6f}]  width={width:.6f}")

        if bench is not None:
            print("  ->", "In range" if (lo <= bench <= hi) else "Out of range")
            

def experiment_delta(S0, K, r, sigma, T, n, h):
    print("\n==== DELTA COMPARISON ====\n")

    bs_delta = bs.bs_delta_call(S0=S0, K=K, r=r, sigma=sigma, T=T)

    mc_fd = mpc.delta_fd_crn_call(S0=S0, K=K, r=r, sigma=sigma,
                                 T=T, h=h, n_sims=n, seed=42)

    mc_pw = mpc.delta_pathwise_call(S0=S0, K=K, r=r, sigma=sigma,
                                   T=T, n_sims=n, seed=42)

    print(f"BS Delta      = {bs_delta:.8f}")
    print(f"MC FD Delta   = {mc_fd:.8f} | Error = {abs(mc_fd-bs_delta):.8f}")
    print(f"MC PW Delta   = {mc_pw:.8f} | Error = {abs(mc_pw-bs_delta):.8f}")


def experiment_gamma(S0, K, r, sigma, T, n, h):
    print("\n==== GAMMA COMPARISON ====\n")

    bs_gamma = bs.bs_gamma_call(S0=S0, K=K, r=r, sigma=sigma, T=T)

    mc_gamma = mpc.gamma_fd_crn_call(S0=S0, K=K, r=r, sigma=sigma,
                                    T=T, h=h, n_sims=n, seed=42)

    print(f"BS Gamma      = {bs_gamma:.8f}")
    print(f"MC FD Gamma   = {mc_gamma:.8f} | Error = {abs(mc_gamma-bs_gamma):.8f}")

def experiment_delta_hedge_sweep(S0_=S0, K_=K, r_=r, sigma_true_=sigma_true, T_=T, n_=n_sims,
                                 sigma_hedge_=sigma_hedge, cost_=cost, steps_list_=steps_list, seed_=seed,
                                 plot_steps=(12, 252)):

    print("\n==== DELTA HEDGE: REBALANCE FREQUENCY SWEEP ====\n")
    for steps in steps_list_:
        paths = gbm.simulate_paths(S0_, r_, sigma_true_, T_, steps, n_, seed=seed_)
        err = eng.delta_hedge_short_call(paths=paths, K=K_, r=r_,
                                         sigma_hedge=sigma_hedge_, T=T_, cost=cost_)

        print(f"steps={steps:>4} | mean={err.mean(): .6f} | std={err.std(ddof=1): .6f} | P(loss)={(err<0).mean(): .4f}")

        if steps in plot_steps:
            vz.plot_hist(err, title=f"Hedge error (short call) | steps={steps}, cost={cost_}")

    m = err.mean()
    sd = err.std(ddof=1)
    p = (err < 0).mean()
    q05, q50, q95 = np.quantile(err, [0.05, 0.50, 0.95])

    print(f"mean={m: .6f} | sd={sd: .6f} | P(loss)={p: .4f} | q05={q05: .6f} | q50={q50: .6f} | q95={q95: .6f}")


def experiment_vol_sweep(*,
                         runner,
                         S0_=S0, r_=r, T_=T,
                         n_steps_=n_steps, n_sims_=n_sims,
                         seed_=seed,
                         initial_capital_=initial_capital,
                         vol_list_=vol_list,
                         bins_y=120):

    for sigma_ in vol_list_:
        paths = gbm.simulate_paths(S0_, r_, sigma_, T_,
                                   n_steps_, n_sims_, seed=seed_)

        run_bot_mc_and_plot(
            runner=runner,
            paths=paths,
            initial_capital=initial_capital_,
            entry_kwargs={"initial_capital": initial_capital_, "plot": False},
            returns_key="Strategy_Returns",
            position_key="Position",
            bins_y=bins_y
        )

if __name__ == "__main__":
    runner = MCRunner(
        bot_cls=MT_Model,
        bot_kwargs=dict(symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        lookback_period=lookback_period,
                        threshold=threshold),
        entrypoint=entrypoint,
        apis=apis,
        quiet=quiet
    )

    experiment_vol_sweep(runner=runner)

