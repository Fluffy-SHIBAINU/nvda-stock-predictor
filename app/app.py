import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import matplotlib.pyplot as plt
from utils.dataset import load_stock_data
from model.lstm_model import StockLSTM
from model.transformer_model import StockTransformer
import torch
import datetime

# -------- UI CONFIG --------
st.set_page_config(page_title="Stock Price Predictor", layout="centered")

# -------- SIDEBAR SETTINGS --------
st.sidebar.title("⚙️ Settings")

ticker = st.sidebar.selectbox("Stock Ticker", ["NVDA", "AAPL", "TSLA", "MSFT", "GOOG"])

today = datetime.date.today()
default_start = today - datetime.timedelta(days=365)

start_date = st.sidebar.date_input("Start Date", default_start)
end_date = st.sidebar.date_input("End Date", today)

seq_len = st.sidebar.slider("Sequence Length", 10, 60, 30)
predict_day = st.sidebar.slider("Days Ahead to Predict", 1, 10, 1)
epochs = st.sidebar.slider("Epochs", 5, 100, 20)

model_type = st.sidebar.selectbox("Model Type", ["LSTM", "Transformer"])
run_btn = st.sidebar.button("🚀 Run Prediction")

# -------- PAGE HEADER --------
st.title(f"📈 {ticker} Stock Price Predictor ({model_type})")
st.markdown("""
Predict future stock prices using an LSTM or Transformer model powered by PyTorch.  
Customize historical data, prediction window, and more from the sidebar.
""")

# -------- MAIN LOGIC --------
if run_btn:
    with st.spinner("Loading data and training the model..."):
        X, y, scaler = load_stock_data(ticker, start_date, end_date, seq_length=seq_len, predict_day=predict_day)
        
        if len(X) == 0:
            st.error("❌ Not enough data in selected date range!")
        else:
            train_size = int(len(X) * 0.8)
            X_train, y_train = X[:train_size], y[:train_size]
            X_test, y_test = X[train_size:], y[train_size:]

            if model_type == "LSTM":
                model = StockLSTM()
            else:
                model = StockTransformer()

            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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

            model.eval()
            with torch.no_grad():
                preds = model(X_test).squeeze().numpy()
                actual = y_test.numpy()
                preds = scaler.inverse_transform(preds.reshape(-1, 1))
                actual = scaler.inverse_transform(actual.reshape(-1, 1))

    st.success("✅ Prediction completed!")

    # -------- PLOT --------
    st.subheader(f"{ticker} Prediction: {predict_day}-Day Ahead using {model_type}")
    fig, ax = plt.subplots()
    ax.plot(actual, label="Actual Price")
    ax.plot(preds, label="Predicted Price")
    ax.legend()
    st.pyplot(fig)
