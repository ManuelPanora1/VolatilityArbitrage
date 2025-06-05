import torch
import torch.nn as nn
from pathlib import Path

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