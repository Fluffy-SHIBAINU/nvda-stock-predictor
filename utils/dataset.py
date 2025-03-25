import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np

def load_nvda_data(seq_length=30):
    df = yf.download("NVDA", period="1y")
    prices = df['Close'].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    X, y = [], []
    for i in range(len(scaled) - seq_length):
        X.append(scaled[i:i+seq_length])
        y.append(scaled[i+seq_length])
    
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    return X, y, scaler
