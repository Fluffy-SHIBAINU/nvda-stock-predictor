import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import matplotlib.pyplot as plt
from utils.dataset import load_nvda_data
from model.lstm_model import StockLSTM
import torch

st.title("📈 NVDA Stock Price Prediction Demo (LSTM)")

st.write("""
This app uses a PyTorch-based LSTM model  
to predict future stock prices of NVIDIA (NVDA)  
based on historical data.
""")

seq_len = st.slider("Sequence Length (days of history)", 10, 60, 30)
epochs = st.slider("Number of Epochs", 5, 100, 20)

if st.button("Run Prediction 🚀"):
    with st.spinner("Loading and training..."):
        X, y, scaler = load_nvda_data(seq_length=seq_len)
        train_size = int(len(X) * 0.8)
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]

        model = StockLSTM()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        for epoch in range(epochs):
            model.train()
            for i in range(0, len(X_train), 32):
                xb = X_train[i:i+32]
                yb = y_train[i:i+32]
                pred = model(xb).squeeze()
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Predict
        model.eval()
        with torch.no_grad():
            preds = model(X_test).squeeze().numpy()
            actual = y_test.numpy()
            preds = scaler.inverse_transform(preds.reshape(-1, 1))
            actual = scaler.inverse_transform(actual.reshape(-1, 1))

    st.success("Prediction completed!")

    # Plot
    fig, ax = plt.subplots()
    ax.plot(actual, label="Actual Price")
    ax.plot(preds, label="Predicted Price")
    ax.set_title("NVDA Stock Price Prediction")
    ax.legend()
    st.pyplot(fig)
