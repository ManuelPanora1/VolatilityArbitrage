import numpy as np
import pandas as pd
from hedging.bs_utils import bs_call_price

def generate_vol_arb_signals(data, T_days = 21, K_offset = 0, r = 0.01):
    option_results = []
    T = T_days / 252  # Convert to years
    
    for _, row in data.iterrows():
        S = row['Price']
        K = S + K_offset
        iv = row['ImpliedVol']
        lstm_sigma = row['LSTMVol']

        if np.isnan(iv) or np.isnan(lstm_sigma): 
            continue

        # Price comparison
        market_price = bs_call_price(S, K, T, iv, r)
        model_price = bs_call_price(S, K, T, lstm_sigma, r)
        spread = model_price - market_price

        option_results.append({
            'Date': row.name if 'Date' not in row else row['Date'],  # Handle index vs column
            'MarketPrice': market_price,
            'ModelPrice': model_price,
            'Spread': spread,
            'Signal': 'Buy' if spread > 0 else 'Sell'
        })
    return pd.DataFrame(option_results)

def backtest_vol_arb(data):
    signals = generate_vol_arb_signals(data)

    #KEY ASSUMPTIONS: We are buying or selling right after 1 day, that is the shift, future direction: to add custom holding period.
    signals['PnL'] = np.where(
        signals['Signal'] == 'Buy',  # If buying options today
        signals['MarketPrice'].shift(-1) - signals['MarketPrice'],  # Sell tomorrow - Buy today
        signals['MarketPrice'] - signals['MarketPrice'].shift(-1)   # Sell today - Buy tomorrow 
    )

    signals = apply_transaction_costs(signals)
    signals = calculate_position_size(signals)
    return signals

def apply_transaction_costs(signals, cost_per_contract= 0.05, slippage_bps = 2):
    signals['Cost'] = cost_per_contract * np.abs(signals['Signal'].map({'Buy':1, 'Sell':-1}))
    signals['Slippage'] = signals['MarketPrice'] * (slippage_bps / 10000)
    
    signals['PnL_Net'] = signals['PnL'] - signals['Cost'] - signals['Slippage']
    return signals

def calculate_position_size(signals, capital=100000, risk_per_trade = 0.01) -> pd.DataFrame:
    signals['SpreadZ'] = (signals['Spread'] - signals['Spread'].mean()) / signals['Spread'].std()
    # Transform to 0-1 range 
    signals['Weight'] = 1 / (1 + np.exp(-signals['SpreadZ']))  
    signals['Position'] = (capital * risk_per_trade * signals['Weight'] / signals['MarketPrice'])
    return signals