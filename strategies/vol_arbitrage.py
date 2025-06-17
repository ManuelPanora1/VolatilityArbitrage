import numpy as np
import pandas as pd
from hedging.bs_utils import bs_call_price, bs_straddle_price

def generate_vol_arb_signals(data, T_days = 21, K_offset = 0, r = 0.01, rolling_window=30):
    option_results = []
    T = T_days / 252  # Convert to years

    # Pre-calculate rolling spread thresholds (avoid look-ahead)
    data['Spread'] = data['LSTMVol'] - data['ImpliedVol']
    data['Spread_Threshold'] = data['Spread'].rolling(rolling_window).apply(
        lambda x: np.percentile(x.dropna(), 75)  # 75th percentile of last 30 days
    ).shift(1).ffill()  # Shift to use only past data
    
    for _, row in data.iterrows():
        S = row['Price']
        K = S + K_offset
        iv = row['ImpliedVol']
        lstm_sigma = row['LSTMVol']
        spread_threshold = row['Spread_Threshold']  # Dynamic threshold

        if np.isnan(iv) or np.isnan(lstm_sigma) or np.isnan(spread_threshold):
            continue

        market_straddle = bs_straddle_price(S, K, T, iv, r)
        model_straddle = bs_straddle_price(S, K, T, lstm_sigma, r)
        spread = model_straddle - market_straddle

        if spread > spread_threshold:
            signal = 'LongStraddle'
        else:
            signal = 'Neutral'

        option_results.append({
            'Date': row.name if 'Date' not in row else row['Date'], 
            'MarketPrice': market_straddle,
            'ModelPrice': model_straddle,
            'Spread': spread,
            'Signal': signal,
            'ImpliedVol': iv
        })
    return pd.DataFrame(option_results)

def backtest_vol_arb(data, initial_capital=100000, max_positions=5):
    signals = generate_vol_arb_signals(data)

    # 2. NEW: Filter signals based on VIX regime (add this right after signal generation)
    signals['VIX_MA'] = signals['ImpliedVol'].rolling(20).mean().ffill()  # 10-day moving average
    signals['IV_trend'] = signals['ImpliedVol'].rolling(5).mean().pct_change(3)
    signals = signals[(signals['IV_trend'] > 0) & (signals['ImpliedVol'] > 0.9*signals['VIX_MA'])]
    #signals = signals[signals['ImpliedVol'] > .9 * signals['VIX_MA']]  # Only keep signals when VIX > its 10-day MA

    portfolio = {
        'cash': initial_capital,
        'positions': [],
        'pnl_history': [],
        'active_dates': set(),
        'total_commission': 0,
        'total_slippage': 0
    }
    
    for date, row in signals.iterrows():
        new_positions = []
        for pos in portfolio['positions']:
            current_pnl_pct = (row['MarketPrice'] - pos['entry_price']) / pos['entry_price']
            should_exit = (
                (date - pos['entry_date']) >= pos['max_holding_days'] or  # Time exit
                current_pnl_pct >= 0.25 or                                    # Take profit at +25%
                current_pnl_pct <= -0.10 or                                   # Stop loss at -10%
                #row['Spread'] <= 0.1 * pos['purchase_spread'] 
                row['Spread'] <= 0.25 * pos['purchase_spread']  # Less sensitive
            )
            if should_exit:
                exit_price, commission, slippage = apply_trade_costs(row['MarketPrice'], pos['size'], 'long')
                pnl = calculate_straddle_pnl(pos['entry_price'], exit_price, pos['size'], 'long')
                net_pnl = pnl - commission - slippage
                print(f"""
                    Entry Date: {pos['entry_date']}
                    Exit Date: {date}
                    Holding Days: {(date - pos['entry_date'])}
                    Entry IV: {pos['entry_iv']}
                    Exit IV: {row['ImpliedVol']}
                    Entry Spread: {pos['purchase_spread']}
                    Exit Spread: {row['Spread']}
                    PnL: {net_pnl}
                    Exit Reason: {'Time' if (date - pos['entry_date']) >= pos['max_holding_days'] else 
                                'Profit' if current_pnl_pct >= 0.25 else 
                                'Loss' if current_pnl_pct <= -0.25 else 
                                'Spread Convergence'}
                """)
                

                portfolio['pnl_history'].append({
                    'date': date,
                    'pnl': net_pnl,
                    'holding_days': (date - pos['entry_date']),
                    'signal': pos['signal'],
                    'entry_price': pos['entry_price'],
                    'size': pos['size'],
                    'exit_price': exit_price,
                    'Spread_at_entry': pos['purchase_spread']
                })
                portfolio['cash'] += pos['size'] + net_pnl
                portfolio['total_commission'] += commission
                portfolio['total_slippage'] += slippage
            else:
                new_positions.append(pos)
        
        portfolio['positions'] = new_positions
        
        # Open new positions (with limits)
        if (row['Signal'] == 'LongStraddle' and 
            len(portfolio['positions']) < max_positions and
            date not in portfolio['active_dates'] and
            portfolio['cash'] > 1000):
            position_size = get_position_size(portfolio['cash'], row['MarketPrice'], .01) #Conservative risk percent
            entry_price, commission, slippage = apply_trade_costs(row['MarketPrice'], position_size, 'long')
            portfolio['positions'].append({
                'entry_date': date,
                'entry_price': entry_price,
                'entry_iv': row['ImpliedVol'],
                'size': position_size,
                'type': 'long',
                'max_holding_days': 21 if row['IV_trend'] > 0 else 7,  
                'signal': row['Signal'],
                'purchase_spread': row['Spread']
            })
            portfolio['cash'] -= position_size
            portfolio['total_commission'] += commission
            portfolio['total_slippage'] += slippage
            portfolio['active_dates'].add(date)
    # Convert to DataFrame
    trades = pd.DataFrame(portfolio['pnl_history'])
    trades['PnL_Net'] = trades['pnl']  # Your existing code expects this column
    trades['cumulative_pnl'] = trades['PnL_Net'].cumsum()
    #results = apply_transaction_costs(results)
   # results = calculate_position_size(results)
    
    return trades, portfolio

def calculate_straddle_pnl(entry_price, exit_price, size, trade_type):
    if trade_type == 'long':
        return size * max(exit_price - entry_price, -entry_price)  # Max loss is premium gained
    else:
        return size * min(entry_price - exit_price, entry_price)  # Max gain is premium received
    
def get_position_size(cash, entry_price, max_risk_pct=0.01, margin_req=0.20):
    risk_amount = max_risk_pct * cash
    margin_required = entry_price * margin_req 
    return min(risk_amount / margin_required, risk_amount)

def apply_trade_costs(price, size, trade_type, commission_per_strangle=3.0, bid_ask_spread=0.02):
    execution_price = price * (1 + bid_ask_spread/2 if trade_type == 'long' else 1 - bid_ask_spread/2)
    slippage = abs(execution_price - price) * size
    return execution_price, commission_per_strangle, slippage

def generate_performance_metrics(trades):
    metrics = {
        'total_pnl': trades['PnL_Net'].sum(),
        'win_rate': (trades['PnL_Net'] > 0).mean(),
        'profit_factor': (trades[trades['PnL_Net'] > 0]['PnL_Net'].sum() / 
                        -trades[trades['PnL_Net'] < 0]['PnL_Net'].sum()),
        'max_drawdown': (trades['PnL_Net'].cumsum() - trades['PnL_Net'].cumsum().cummax()).min(),
        'sharpe_ratio': (trades['PnL_Net'].mean() / trades['PnL_Net'].std()) * np.sqrt(252)
    }
    return metrics
    

def adaptive_holding_period(signal_row, base_days):
    spread_ratio = abs(signal_row['Spread']) / signal_row['MarketPrice']
    return min(
        base_days,
        max(5, int(base_days * (1 - spread_ratio * 10))
    ))

#Delta hedging for future implementation
"""
def apply_delta_hedge(position, current_delta):
    hedge_qty = -position['size'] * current_delta
    # Implementation depends on your hedging instrument (futures/underlying)
    return {
        'instrument': 'EQUITY_OR_FUTURE',
        'quantity': hedge_qty,
        'entry_price': current_price
    }
 """