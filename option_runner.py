import mc_pricer as mpc


OPTION_PRICERS = {
    "european_call": mpc.discounted_payoffs_call,
    "asian_call": mpc.discounted_payoffs_asian_call,
    "up_and_out_call": mpc.discounted_payoffs_up_and_out_call,
    "european_call_antithetic": mpc.discounted_payoffs_call_antithetic,
    "european_call_control_variate": mpc.DPC_control_variate,
    "asian_call_control_variate": mpc.DPCasian_control_variate,
}


def price_option(option_type, params):
    if option_type not in OPTION_PRICERS:
        raise ValueError(
            f"Unknown option type '{option_type}'. "
            f"Available option types: {list(OPTION_PRICERS.keys())}"
        )

    payoff_fn = OPTION_PRICERS[option_type]
    discounted_payoffs = payoff_fn(**params)

    mean, lo, hi = mpc.mc_price_CI(discounted_payoffs)

    return {
        "option_type": option_type,
        "price": mean,
        "ci_low": lo,
        "ci_high": hi,
        "ci_width": hi - lo,
        "n_samples": len(discounted_payoffs),
    }