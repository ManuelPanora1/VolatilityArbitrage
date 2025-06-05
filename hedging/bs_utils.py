import numpy as np
from scipy.stats import norm

def bs_call_price(S, K, T, sigma, r=0.01):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put_price(S, K, T, sigma, r=0.01):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_straddle_price(S, K, T, sigma, r=0.01):
    return bs_call_price(S, K, T, sigma, r) + bs_put_price(S, K, T, sigma, r)

def bs_delta(S, K, T, sigma, r=0.01):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

def delta_hedge_sim(data, vol_col, K_offset=0, T_days=21):
    portfolio = []
    cash = 0
    shares = 0
    K = None
    T0 = T_days / 252  

    for i in range(len(data) - T_days):
        #Extract data column for given day
        today = data.iloc[i]
        next_day = data.iloc[i+1]
        S = today['Price']
        sigma = today[vol_col]
        if np.isnan(sigma): continue

        # Strike = current price + offset
        if K is None:
            K = S + K_offset

        T = T0 - (i / 252)
        option_price = bs_call_price(S, K, T0, sigma)
        delta = bs_delta(S, K, T0, sigma)

        # First day: buy option, hedge delta
        if i == 0:
            cash = option_price - (delta * S)
            shares = delta
            #portfolio_value = delta * S + cash
        else:
            # Update portfolio value
            dS = next_day['Price']
            option_val = bs_call_price(dS, K, T0 - (i / 252), sigma)
            new_delta = bs_delta(dS, K, T0 - (i / 252), sigma)

            hedge_error = (shares * dS + cash) - option_val
            portfolio.append(hedge_error)

            # Rebalance delta hedge
            shares = new_delta
            cash = (shares * dS + cash) - option_val  # Replicate option value
    return portfolio