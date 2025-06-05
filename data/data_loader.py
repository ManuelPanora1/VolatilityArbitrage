import numpy as np
import yfinance as yf
import pandas as pd
from pathlib import Path
import torch

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

        spy['LogRet'] = np.log(spy['Close'] / spy['Close'].shift(1))
        spy['RealizedVol'] = spy['LogRet'].rolling(window=21).std() * np.sqrt(252)  
        spy['ImpliedVol'] = vix['Close']

        data = pd.concat([spy['Close'], spy['RealizedVol'], spy['LogRet'], spy['ImpliedVol']], axis=1)
        data.columns = ['Price', 'RealizedVol', 'LogRet', 'ImpliedVol']
        data.dropna(inplace=True)

    Path(cache_path).parent.mkdir(exist_ok=True)
    data.to_parquet(cache_path)
    
    return data

def partition_data(data, scaler, lookback=30):
    scaled_vol = scaler.fit_transform(data[['RealizedVol']])
    X = np.array([scaled_vol[i-lookback:i] for i in range(lookback, len(scaled_vol))])
    y = np.array([scaled_vol[i] for i in range(lookback, len(scaled_vol))])
    #Training and testing data
    partition = int(len(y.flatten()) * .04)
    X_train, y_train = torch.tensor(X[:-partition], dtype=torch.float32), torch.tensor(y[:-partition], dtype=torch.float32)
    X_test, y_test = torch.tensor(X[-partition:], dtype=torch.float32), torch.tensor(y[-partition:], dtype=torch.float32)
    return X_train, y_train, X_test, y_test, scaler