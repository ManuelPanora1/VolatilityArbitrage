import numpy as np
import torch
from data.data_loader import preprocess_data, partition_data
from models.lstm_model import lstm_train, time_series_validation
import matplotlib.pyplot as plt
from strategies.vol_arbitrage import backtest_vol_arb, generate_performance_metrics
import pandas as pd
from analysis.monte_carlo_simulation import monte_carlo

def lstm_arbitrage():
    #Process data
    data = preprocess_data()

    #Train Model
    X_train, y_train, X_test, y_test, scaler = partition_data(data)
    model = lstm_train(X_train, y_train)

    #Save model later

    #Evaluate Model on Predictions
    model.eval()
    with torch.no_grad():
        preds = model(X_test).numpy()
    preds_unscaled = scaler.inverse_transform(preds)
    n = len(y_test)

    #Volatility forecast plot
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[-n:], data['RealizedVol'][-n:], label='True Vol')
    plt.plot(data.index[-n:], preds_unscaled, label='LSTM Predicted Vol')
    plt.legend()
    plt.title("Volatility Forecasting")
    plt.show()
    

    testing = pd.DataFrame({
        'Price': data['Price'][-n:],
        'RealizedVol': data['RealizedVol'][-n:],
        'LSTMVol': preds_unscaled.flatten(),
        'ImpliedVol': np.array((data[['ImpliedVol']].iloc[-n:])).flatten()
    })

    trades, portfolio = backtest_vol_arb(testing)
    metrics = generate_performance_metrics(trades)
       
    plt.plot(trades.index, trades['PnL_Net'].cumsum(), label='Strategy PnL')
    plt.axhline(0, color='black', linestyle='--')
    plt.legend()
    plt.title("Volatility Arbitrage Strategy Performance")
    plt.tight_layout()
    plt.show()
    # Print performance metrics
    print("\nStrategy Metrics:")
    print(f"Total PnL: ${metrics['total_pnl']:,.2f}")
    print(f"Win Rate: {100*metrics['win_rate']:.1f}%")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Max Drawdown: ${metrics['max_drawdown']:,.2f}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Total Costs: ${portfolio['total_commission'] + portfolio['total_slippage']:,.2f}")

    #Run Monte Carlo Simulation to see if the model has any predictive power:
    returns = []
    for _,row in trades.iterrows():
        if np.isnan(row['PnL_Net']) or np.isnan(row['exit_price']): 
            continue
        returns.append(row['PnL_Net']/row['exit_price'])

    p_val = monte_carlo(np.array(returns))

    print()
    print("This is your coveted pval: (Moment of truth)")
    print(p_val)

def transformer_arbitrage():
    return

def main(type):
    if (type == 'LSTM'):
        lstm_arbitrage()
        return
    elif (type == 'Transformer'):
        transformer_arbitrage()
        return
    
def validate_model(type):
    if (type == 'LSTM'):
        data = preprocess_data()
        time_series_validation(data)

if __name__ == "__main__":
    main('LSTM')
    #validate_model('LSTM')


