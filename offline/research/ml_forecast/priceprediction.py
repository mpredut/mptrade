"""PyTorch LSTM price prediction research model.

Migrated from legacy Keras/TensorFlow prototype to pure PyTorch (CPU-efficient).
Used for offline research and evaluation of sequence models against Lindy baseline.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - optional heavy dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


if nn is not None:
    class _LSTMNet(nn.Module):
        def __init__(self, input_size: int = 1, hidden_size: int = 32, num_layers: int = 1):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])
else:
    _LSTMNet = None  # type: ignore[assignment,misc]


class PricePrediction:
    """LSTM-based next-step price prediction using sequence windows."""

    def __init__(
        self,
        window_size: int = 10,
        hidden_size: int = 32,
        lr: float = 0.01,
        epochs: int = 30,
    ) -> None:
        def _num(v, default: float) -> float:
            if v is None or isinstance(v, bool):
                return default
            try:
                f = float(v)
                return f if np.isfinite(f) else default
            except (TypeError, ValueError, OverflowError):
                return default

        ws_f = _num(window_size, 10.0)
        hs_f = _num(hidden_size, 32.0)
        lr_f = _num(lr, 0.01)
        ep_f = _num(epochs, 30.0)
        self.window_size = max(2, int(ws_f)) if np.isfinite(ws_f) else 10
        safe_hidden = max(1, int(hs_f)) if np.isfinite(hs_f) else 32
        safe_lr = lr_f if np.isfinite(lr_f) and lr_f > 0 else 0.01
        self.prices: list[float] = []
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self._fallback_weights: np.ndarray | None = None
        if torch is not None and _LSTMNet is not None:
            self.model = _LSTMNet(input_size=1, hidden_size=safe_hidden)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=safe_lr)
            self.criterion = nn.MSELoss()
        else:
            self.model = None
            self.optimizer = None
            self.criterion = None
        self.trained = False
        self.epochs = max(1, int(ep_f)) if np.isfinite(ep_f) else 30

    @staticmethod
    def _valid_price(price: object) -> float | None:
        if isinstance(price, bool) or price is None:
            return None
        try:
            val = float(price)
        except (TypeError, ValueError):
            return None
        return val if np.isfinite(val) and val > 0 else None

    def process_price(self, price: float) -> None:
        """Append price and trigger periodic model retraining."""
        val = self._valid_price(price)
        if val is None:
            return
        self.prices.append(val)
        if len(self.prices) >= self.window_size * 2:
            self.train_lstm_model()

    def process_prices(self, prices: list[float]) -> None:
        """Batch set prices and trigger retraining if enough points exist."""
        if prices is None or isinstance(prices, (str, bytes, bool)):
            self.prices = []
        else:
            try:
                self.prices = [v for p in prices if (v := self._valid_price(p)) is not None]
            except TypeError:
                self.prices = []
        self.trained = False
        self._fallback_weights = None
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

        if torch is None or self.model is None or self.optimizer is None or self.criterion is None:
            X_flat = X.reshape(len(X), -1)
            X_aug = np.hstack([X_flat, np.ones((len(X_flat), 1), dtype=np.float32)])
            w, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
            self._fallback_weights = w.astype(np.float32)
            self.trained = True
            return True

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

        last_seq = np.array(self.prices[-self.window_size :], dtype=np.float32).reshape(-1, 1)
        last_seq_scaled = self.scaler.transform(last_seq).reshape(1, self.window_size, 1)

        if torch is None or self.model is None:
            if self._fallback_weights is None:
                return None
            x_aug = np.append(last_seq_scaled.reshape(1, -1), [[1.0]], axis=1)
            pred_scaled = x_aug @ self._fallback_weights
            pred = self.scaler.inverse_transform(pred_scaled)
            val = float(pred[0][0])
            return val if np.isfinite(val) else None

        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(last_seq_scaled, dtype=torch.float32)
            pred_scaled = self.model(X_t).numpy()
            pred = self.scaler.inverse_transform(pred_scaled)
            val = float(pred[0][0])
            return val if np.isfinite(val) else None


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
