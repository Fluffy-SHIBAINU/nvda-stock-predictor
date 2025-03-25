# 📈 NVDA Stock Price Predictor (LSTM + PyTorch)

This project builds a simple stock price prediction model using PyTorch and historical stock data for NVIDIA (NVDA).  
The model uses an LSTM (Long Short-Term Memory) neural network to learn from past price trends and forecast future prices.

### 🔗 Live Demo
👉 [Check it out on Streamlit]([(https://nvda-stock-predictor.streamlit.app/)](https://nvda-stock-predictor.streamlit.app/))

<img width="400" alt="image" src="https://github.com/user-attachments/assets/ad24ed02-9c9b-4b6d-8e8b-39b040a31677" />

## 🔧 Features

- 📉 Downloads NVDA stock data via `yfinance`
- 🧠 Trains an LSTM-based model using PyTorch
- 📊 Visualizes prediction vs actual closing prices
- 🌐 Interactive web app with Streamlit

## 🚀 Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run training script
```bash
python3 train.py
```

### 3. Run the web app
```bash
streamlit run app/app.py
```

## 🧪 Model

- **Architecture**: 2-layer LSTM + Fully Connected Layer
- **Input**: Sequence of past closing prices (e.g., 30 days)
- **Output**: Next day’s predicted price
- **Loss Function**: MSELoss
- **Optimizer**: Adam

## 🖼️ Example Output

<img width="400" alt="image" src="https://github.com/user-attachments/assets/8571261a-7530-4ca1-bcea-d59ebd6edb63" />


## 📁 Project Structure

```
nvda-stock-predictor/
├── app/                # Streamlit web app
│   └── app.py
├── model/              # LSTM model definition
│   └── lstm_model.py
├── utils/              # Data loading and preprocessing
│   └── dataset.py
├── train.py            # Training script
├── requirements.txt
└── README.md
```

## 📝 License

MIT License
