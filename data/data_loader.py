import numpy as np
import yfinance as yf
import pandas as pd
from pathlib import Path
import torch
from sklearn.preprocessing import MinMaxScaler


def time_series_split(data, test_size=0.2):
    split_idx = int(len(data) * (1-test_size))
    train = data.iloc[:split_idx]
    test = data.iloc[split_idx:]
    return train, test

def preprocess_data(type="financial", tickers=['SPY', '^VIX'], 
                    start='2010-01-01', 
                    end='2024-01-01',
                    cache_path="data/processed/spy_vix_processed.parquet"):
    # Check for cached data
    if Path(cache_path).exists():
        return pd.read_parquet(cache_path)
    
    if (type == "financial"):
        spy = yf.download(tickers[0], start, end)
        vix = yf.download(tickers[1], start, end)

        #BS requires that our volatility be in decimal form and no percentage
        spy['LogRet'] = np.log(spy['Close'] / spy['Close'].shift(1))
        spy['RealizedVol'] = spy['LogRet'].rolling(window=21).std() * np.sqrt(252)  
        spy['ImpliedVol'] = vix['Close'] / 100
        print(spy['RealizedVol'])
        print(spy['ImpliedVol'])

        data = pd.concat([spy['Close'], spy['RealizedVol'], spy['LogRet'], spy['ImpliedVol']], axis=1)
        data.columns = ['Price', 'RealizedVol', 'LogRet', 'ImpliedVol']
        data.dropna(inplace=True)

    Path(cache_path).parent.mkdir(exist_ok=True)
    data.to_parquet(cache_path)
    
    return data

def partition_data(data, lookback=30):
    train_data, test_data = time_series_split(data, test_size=0.3)  
    scaler = MinMaxScaler().fit(train_data[['RealizedVol']])
    #TODO Found source of potential leak as we are using a scaler fitted to the entire data set
    #Learns and stores the min and max and then transforms data
    #scaled_vol = scaler.fit_transform(data[['RealizedVol']])
    train_scaled = scaler.transform(train_data[['RealizedVol']])
    test_scaled = scaler.transform(test_data[['RealizedVol']])

    X_train = np.array([train_scaled[i-lookback:i] for i in range(lookback, len(train_scaled))])
    y_train = train_scaled[lookback:]
    X_test = np.array([test_scaled[i-lookback:i] for i in range(lookback, len(test_scaled))])
    y_test = test_scaled[lookback:]
    X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
    X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)
    #print(X_train.shape, y_train.shape)

    return X_train, y_train, X_test, y_test, scaler