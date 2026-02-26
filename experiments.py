import gbm
import mc_pricer as mpc 
import black_scholes as bs 
import numpy as np



def run_experiment(discounted_payoffs_fn, S0,
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

<<<<<<< Updated upstream
S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0

run_experiment(
    discounted_payoffs_fn=mpc.discounted_payoffs_call,
    benchmark_fn=bs.bs_call_price,
    label="European Call (plain)",
    S0=S0, K=K, r=r, sigma=sigma, T=T,
    seed=123
)

run_experiment(
    discounted_payoffs_fn=mpc.DPC_control_variate,
    benchmark_fn=bs.bs_call_price,
    label="European Call (control variate)",
    S0=S0, K=K, r=r, sigma=sigma, T=T,
    seed=123
)
=======
            
>>>>>>> Stashed changes
