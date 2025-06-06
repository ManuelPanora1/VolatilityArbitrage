import numpy as np
import torch
from data.data_loader import preprocess_data, partition_data
from sklearn.preprocessing import MinMaxScaler
from models.lstm_model import lstm_train, time_series_validation
import matplotlib.pyplot as plt
from strategies.vol_arbitrage import backtest_vol_arb
import pandas as pd
from analysis.monte_carlo_simulation import monte_carlo

def lstm_arbitrage():
    #Process data
    data = preprocess_data()

    #Train Model
    X_train, y_train, X_test, y_test, scaler = partition_data(data, MinMaxScaler())
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
        #TODO: Found the Bug, the implied volatility is unscaled
        'ImpliedVol': scaler.fit_transform(data[['ImpliedVol']].iloc[-n:]).flatten()
    })

    trades = backtest_vol_arb(testing)
       
    plt.plot(trades.index, trades['PnL_Net'].cumsum(), label='Strategy PnL')
    plt.axhline(0, color='black', linestyle='--')
    plt.legend()
    plt.title("Volatility Arbitrage Strategy Performance")
    plt.tight_layout()
    plt.show()
    # Print performance metrics
    print("\nStrategy Metrics:")
    print(f"Total PnL: ${trades['PnL_Net'].sum():.2f}")
    print(f"Win Rate: {100*(trades['PnL_Net'] > 0).mean():.1f}%")
    print(f"Profit Factor: {trades[trades['PnL_Net'] > 0]['PnL_Net'].sum() / -trades[trades['PnL_Net'] < 0]['PnL_Net'].sum():.2f}")

    #Run Monte Carlo Simulation to see if the model has any predictive power:
    returns = []
    for _,row in trades.iterrows():
        if np.isnan(row['PnL_Net']) or np.isnan(row['MarketPrice']): 
            continue
        returns.append(row['PnL_Net']/row['MarketPrice'])

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
    #main('LSTM')
    validate_model('LSTM')


