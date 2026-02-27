import gbm
import mc_pricer as mpc 
import black_scholes as bs 
import numpy as np
import engine as eng
import strategies as strat
import visualize as vz 
from Experiment_Bot.MTbot import MT_Model
from mc_runner import MCRunner

### standard variable numbers for testing 
S0, K, r, sigma, T, n_sims = 100, 100, 0.05, 0.2, 1.0, 2000
n_steps, lookback, threshold, cost, seed = 252, 20, 0.02, 0.0, 1
sigma_hedge, sigma_true, steps_list = 0.2, 0.2, [12, 52, 252]

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

paths = gbm.simulate_paths(S0, r, sigma, T, n_steps, n_sims, seed=1)
initial_capital = 10000

runner = MCRunner(
    bot_cls=MT_Model,
    bot_kwargs=dict(symbol="FAKE", start_date="2020-01-01", end_date="2021-01-01",
                    lookback_period=20, threshold=0.02),
    entrypoint="run_complete_backtest",
    apis=["yfinance.download"],
    quiet=True
)

outs = runner.run_paths(paths, entry_kwargs={"initial_capital": initial_capital, "plot": False})

equity = np.zeros((n_sims, n_steps + 1))
equity[:, 0] = initial_capital

pos = np.zeros((n_sims, n_steps))  # position held over each step

for i, out in enumerate(outs):
    sr = out["Strategy_Returns"]                  # pandas Series :contentReference[oaicite:1]{index=1}
    rets = np.asarray(sr.fillna(0.0), dtype=float)
    # ensure length matches n_steps (some strategies might return n_steps+1; adjust if needed)
    rets = rets[:n_steps]
    equity[i, 1:] = initial_capital * np.cumprod(1.0 + rets)

    p = np.asarray(out["Position"].fillna(0.0), dtype=float)[:n_steps]
    pos[i] = p


vz.plot_equity_density_heatmap(
    equity,
    bins_y=90,
    title="MT bot: strategy value density across MC sims")

vz.plot_position_prob(pos, title="MT bot: P(in trade) over time")