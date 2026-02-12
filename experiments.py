import gbm
import mc_pricer as mpc 
import black_scholes as bs 

bs_price = bs.bs_call_price(100, 100, .05, .2, 1)
print(bs_price)
for n in [1000, 5000, 10000, 50000, 100000]:
    mc_vector = mpc.discounted_payoffs_call(100, 100, .05, .2, 1, n)
    mc_CI = mpc.mc_price_CI(mc_vector)
    print(mc_CI)
    if bs_price > mc_CI[1] and bs_price < mc_CI[2]:
        print("In range")
    else:
        print("Out of range")


