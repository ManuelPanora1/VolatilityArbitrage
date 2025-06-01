import yfinance as yf
import numpy as np
import pandas as pd

#Returns of S&P 500 over the last 14 years
spy = yf.download('SPY', start='2010-01-01', end='2024-01-01')
#Implied volatility data
vix = yf.download('^VIX', start='2010-01-01', end='2024-01-01')

#Calculated log returns
spy['LogRet'] = np.log(spy['Adj Close'] / spy['Adj Close'].shift(1))
#Actual volatility computed as variance of log returns
spy['RealizedVol'] = spy['LogRet'].rolling(window=21).std() * np.sqrt(252)  # 1-month vol


if __name__ == "__main__":
    # Check type
    print(type(spy))

    # Show first 5 rows
    print(spy.head())

    # List all columns
    print(spy.columns)