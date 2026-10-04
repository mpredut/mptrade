"""PyTorch LSTM price prediction research model.

Migrated from legacy Keras/TensorFlow prototype to pure PyTorch (CPU-efficient).
Used for offline research and evaluation of sequence models against Lindy baseline.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


class _LSTMNet(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class PricePrediction:
    """LSTM-based next-step price prediction using sequence windows."""

    def __init__(
        self,
        window_size: int = 10,
        hidden_size: int = 32,
        lr: float = 0.01,
        epochs: int = 30,
    ) -> None:
        self.window_size = int(window_size)
        self.prices: list[float] = []
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = _LSTMNet(input_size=1, hidden_size=hidden_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.trained = False
        self.epochs = epochs

    def process_price(self, price: float) -> None:
        """Append price and trigger periodic model retraining."""
        self.prices.append(float(price))
        if len(self.prices) >= self.window_size * 2:
            self.train_lstm_model()

    def process_prices(self, prices: list[float]) -> None:
        """Batch set prices and trigger retraining if enough points exist."""
        self.prices = [float(p) for p in prices]
        if len(self.prices) >= self.window_size * 2:
            self.train_lstm_model()

    def prepare_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Scale prices and build (window_size -> next_price) supervision tuples."""
        if len(self.prices) <= self.window_size:
            return np.empty((0, self.window_size, 1), dtype=np.float32), np.empty((0, 1), dtype=np.float32)

        prices_array = np.array(self.prices, dtype=np.float32).reshape(-1, 1)
        self.scaler.fit(prices_array)
        scaled = self.scaler.transform(prices_array)

        X, y = [], []
        for i in range(len(scaled) - self.window_size):
            X.append(scaled[i : i + self.window_size])
            y.append(scaled[i + self.window_size])

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def train_lstm_model(self) -> bool:
        """Fit the LSTM model on current historical window."""
        X, y = self.prepare_data()
        if len(X) < 2:
            return False

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        self.model.train()
        for _ in range(self.epochs):
            self.optimizer.zero_grad()
            output = self.model(X_t)
            loss = self.criterion(output, y_t)
            loss.backward()
            self.optimizer.step()

        self.trained = True
        return True

    def predict_next_price(self) -> float | None:
        """Predict the next price step based on the most recent sequence."""
        if not self.trained or len(self.prices) < self.window_size:
            return None

        self.model.eval()
        with torch.no_grad():
            last_seq = np.array(self.prices[-self.window_size :], dtype=np.float32).reshape(-1, 1)
            last_seq_scaled = self.scaler.transform(last_seq).reshape(1, self.window_size, 1)
            X_t = torch.tensor(last_seq_scaled, dtype=torch.float32)
            pred_scaled = self.model(X_t).numpy()
            pred = self.scaler.inverse_transform(pred_scaled)
            return float(pred[0][0])


if __name__ == "__main__":
    print("Testing PricePrediction PyTorch LSTM...")
    predictor = PricePrediction(window_size=10, epochs=25)
    # Feed synthetic rising prices
    base = 100.0
    for i in range(40):
        base += 1.0 + float(np.sin(i / 3.0))
        predictor.process_price(base)

    next_p = predictor.predict_next_price()
    print(f"Last price: {base:.2f}, Predicted next price: {next_p:.2f}")
    assert next_p is not None and next_p > 0
    print("PricePrediction PyTorch model successfully trained and generated prediction.")
