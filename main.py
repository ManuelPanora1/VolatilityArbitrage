import numpy as np
import torch
from data.data_loader import preprocess_data, partition_data
from sklearn.preprocessing import MinMaxScaler
from models.lstm_model import lstm_train
import matplotlib.pyplot as plt


if __name__ == "__main__":
    data = preprocess_data()
    X_train, y_train, X_test, y_test, scaler = partition_data(data, MinMaxScaler())
    model = lstm_train(X_train, y_train)

    #Save model 
    model.eval()
    with torch.no_grad():
        preds = model(X_test).numpy()

    preds_unscaled = scaler.inverse_transform(preds)

    n = len(y_test)
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[-n:], data['RealizedVol'][-n:], label='True Vol')
    plt.plot(data.index[-n:], preds_unscaled, label='LSTM Predicted Vol')
    plt.legend()
    plt.title("Volatility Forecasting")
    plt.show()

