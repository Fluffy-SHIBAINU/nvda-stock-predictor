import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model.lstm_model import StockLSTM
from utils.dataset import load_nvda_data
import matplotlib.pyplot as plt

# 하이퍼파라미터
seq_len = 30
batch_size = 32
epochs = 20
lr = 0.001

# 데이터 로딩
X, y, scaler = load_nvda_data(seq_length=seq_len)
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

# 모델 초기화
model = StockLSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 학습 루프
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for xb, yb in train_loader:
        output = model(xb)
        loss = criterion(output.squeeze(), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

# 예측
model.eval()
with torch.no_grad():
    preds = model(X_test).squeeze().numpy()
    actual = y_test.numpy()

    # 역정규화
    preds = scaler.inverse_transform(preds.reshape(-1, 1))
    actual = scaler.inverse_transform(actual.reshape(-1, 1))

# 시각화
plt.plot(actual, label='Actual')
plt.plot(preds, label='Predicted')
plt.legend()
plt.title('NVDA Stock Price Prediction')
plt.show()
