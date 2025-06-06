import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
import numpy as np

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
    
def lstm_train(X, y):
    model = LSTMModel()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

    return model

def time_series_validation(data, lookback=30, n_splits=5):
    scaler = MinMaxScaler()
    scaled_vol = scaler.fit_transform(data[['RealizedVol']])
    
    # Prepare sequences
    X = np.array([scaled_vol[i-lookback:i] for i in range(lookback, len(scaled_vol))])
    y = np.array([scaled_vol[i] for i in range(lookback, len(scaled_vol))])
    
    #For time series cross validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train = torch.tensor(X[train_idx], dtype=torch.float32)
        y_train = torch.tensor(y[train_idx], dtype=torch.float32)
        X_test = torch.tensor(X[test_idx], dtype=torch.float32)
        y_test = torch.tensor(y[test_idx], dtype=torch.float32)
        
        model = lstm_train(X_train, y_train)
        
        model.eval()
        with torch.no_grad():
            train_preds = model(X_train)
            train_loss = nn.MSELoss()(train_preds, y_train).item()
            
            test_preds = model(X_test)
            test_loss = nn.MSELoss()(test_preds, y_test).item()
        results.append({
            'fold': fold + 1,
            'train_loss': train_loss,
            'test_loss': test_loss
        })
        
        print(f"Fold {fold + 1}:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Test Loss: {test_loss:.6f}")
        print("-----------------------")
    return results


def save_model(model, scaler, lookback, save_dir="models/saved_models"):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,          # MinMaxScaler object
        'lookback': lookback,      # Sequence length 
        'input_size': model.lstm.input_size  # Number of features
    }, f"{save_dir}/volatility_lstm.pt")


def load_model(save_path="models/saved_models/volatility_lstm.pt"):
    checkpoint = torch.load(save_path)
    
    # Rebuild model architecture
    model = LSTMModel(
        input_size=checkpoint['input_size'],
        hidden_size=32  
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  
    
    return model, checkpoint['scaler'], checkpoint['lookback']