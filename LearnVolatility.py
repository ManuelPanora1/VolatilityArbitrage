import yfinance as yf
import numpy as np
import pandas as pd
from arch import arch_model
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


#Returns of S&P 500 over the last 14 years
spy = yf.download('SPY', start='2010-01-01', end='2024-01-01')
#Implied volatility data
vix = yf.download('^VIX', start='2010-01-01', end='2024-01-01')

#Calculated log returns
spy['LogRet'] = np.log(spy['Close'] / spy['Close'].shift(1))
#Actual volatility computed as variance of log returns
spy['RealizedVol'] = spy['LogRet'].rolling(window=21).std() * np.sqrt(252)  # 1-month vol

data = pd.concat([spy['LogRet'], spy['RealizedVol'], vix['Close']], axis=1)
data.columns = ['LogRet', 'RealizedVol', 'VIX']
data.dropna(inplace=True)

returns = data['LogRet'] * 100  
#Initialize Garch(1,1) model
model = arch_model(returns, vol='Garch', p=1, q=1)
#Fits model to data using MLE
res = model.fit()
data['GARCH_Pred'] = res.conditional_volatility / 100 #Convert back to decimal

#Length of timesteps
lookback = 30
#Normalize data to pass into the LSTM
scaler = MinMaxScaler()
scaled_vol = scaler.fit_transform(data[['RealizedVol']])

#Data from the past 30 days are stored in X and will be used to predict y.
X = np.array([scaled_vol[i-lookback:i] for i in range(lookback, len(scaled_vol))])
y = np.array([scaled_vol[i] for i in range(lookback, len(scaled_vol))])

#Training and testing data
X_train, y_train = torch.tensor(X[:-200], dtype=torch.float32), torch.tensor(y[:-200], dtype=torch.float32)
X_test, y_test = torch.tensor(X[-200:], dtype=torch.float32), torch.tensor(y[-200:], dtype=torch.float32)

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        #Input 1 as 1 time dependent feature 32 hidden size initially used, we can change for better performance
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        #Input data into lstm and returns all hidden states at every time step
        x, _ = self.lstm(x)
        x = x[:, -1]
        return self.fc(x)

model = LSTMModel()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#Train
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    output = model(X_train.unsqueeze(-1))
    loss = loss_fn(output, y_train)
    loss.backward()
    optimizer.step()

# Predict
model.eval()
with torch.no_grad():
    preds = model(X_test.unsqueeze(-1)).numpy()

# Reverse scaling
preds_unscaled = scaler.inverse_transform(preds)

if __name__ == "__main__":
    # List all columns
    #print(data['LogRet'])
    #print(returns)
    #print(res)
    #print(data['GARCH_Pred'])
    print(X.shape)
    """
    """