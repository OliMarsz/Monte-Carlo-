import gbm
import mc_pricer as mpc 
import black_scholes as bs 
import numpy as np
import engine as eng
import strategies as strat

### standard variable numbers for testing 
S0, K, r, sigma, T, n_sims = 100, 100, 0.05, 0.2, 1.0, 200000
n_steps, lookback, threshold, cost = 252, 20, 0.02, 0.001

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


paths = gbm.simulate_paths(S0, r, sigma, T, n_steps, n_sims, seed=1)
res = eng.backtest_positions(paths=paths, position_fn=strat.pos_momentum,
                             cost=cost, lookback=lookback, threshold=threshold)

fr = res["final_return"]
print("mean final return:", fr.mean())
print("prob lose money :", (fr < 0).mean())