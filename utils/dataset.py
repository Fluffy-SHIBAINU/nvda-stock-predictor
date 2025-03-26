import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np

def load_stock_data(symbol="NVDA", start="2023-01-01", end="2024-01-01", seq_length=30, predict_day=1):
    df = yf.download(symbol, start=start, end=end)
    prices = df['Close'].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    X, y = [], []
    for i in range(len(scaled) - seq_length - predict_day):
        X.append(scaled[i:i+seq_length])
        y.append(scaled[i+seq_length + predict_day - 1])
    
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    return X, y, scaler
