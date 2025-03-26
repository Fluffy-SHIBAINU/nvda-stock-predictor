
import torch
import torch.nn as nn

class StockTransformer(nn.Module):
    def __init__(self, input_size=1, d_model=64, nhead=4, num_layers=2):
        super(StockTransformer, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_projection(x)  # shape: (batch, seq_len, d_model)
        x = self.transformer_encoder(x)
        return self.fc(x[:, -1, :])   # Use last token output
