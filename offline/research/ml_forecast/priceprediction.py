"""PyTorch LSTM price prediction research model.

Migrated from legacy Keras/TensorFlow prototype to pure PyTorch (CPU-efficient).
Used for offline research and evaluation of sequence models against Lindy baseline.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.preprocessing import MinMaxScaler

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from botcore import safe_float, safe_int, safe_pos_float  # noqa: E402

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
        max_history: int = 512,
        retrain_every: int = 1,
    ) -> None:
        self.window_size = max(2, safe_int(window_size, default=10))
        self._hidden_size = max(1, safe_int(hidden_size, default=32))
        self._lr = safe_pos_float(lr, default=0.01)
        self._max_history = max(self.window_size * 2 + 2, safe_int(max_history, default=512))
        self._retrain_every = max(1, safe_int(retrain_every, default=1))
        self._steps_since_train = 0
        self.prices: list[float] = []
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self._fallback_weights: np.ndarray | None = None
        self.model = None
        self.optimizer = None
        self.criterion = None
        self._reset_model()
        self.trained = False
        self.epochs = max(1, safe_int(epochs, default=30))

    def _reset_model(self) -> None:
        """Reinitialize model weights and optimizer state to avoid cross-series leakage."""
        if torch is not None and _LSTMNet is not None:
            if hasattr(torch, "manual_seed"):
                torch.manual_seed(0)
            self.model = _LSTMNet(input_size=1, hidden_size=self._hidden_size)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr)
            self.criterion = nn.MSELoss()
        else:
            self.model = None
            self.optimizer = None
            self.criterion = None

    def process_price(self, price: float) -> None:
        """Append price and trigger periodic model retraining."""
        val = safe_pos_float(price, default=None)
        if val is None:
            return
        self.prices.append(val)
        if len(self.prices) > self._max_history:
            self.prices = self.prices[-self._max_history :]
        self._steps_since_train += 1
        if len(self.prices) >= self.window_size * 2 and (
            not self.trained or self._steps_since_train >= self._retrain_every
        ):
            self._steps_since_train = 0
            self.trained = False
            self._fallback_weights = None
            self._reset_model()
            self.train_lstm_model()

    def process_prices(self, prices: list[float]) -> None:
        """Batch set prices and trigger retraining if enough points exist."""
        if prices is None or isinstance(prices, (str, bytes, bool)):
            self.prices = []
        else:
            try:
                parsed = [v for p in prices if (v := safe_pos_float(p, default=None)) is not None]
                self.prices = parsed[-self._max_history :]
            except TypeError:
                self.prices = []
        self._steps_since_train = 0
        self.trained = False
        self._fallback_weights = None
        self._reset_model()
        if len(self.prices) >= self.window_size * 2:
            self.train_lstm_model()

    def prepare_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Scale prices and build (window_size -> next_price) supervision tuples."""
        if len(self.prices) <= self.window_size:
            return np.empty((0, self.window_size, 1), dtype=np.float32), np.empty((0, 1), dtype=np.float32)

        prices_array = np.array(self.prices, dtype=np.float64).reshape(-1, 1)
        if not np.all(np.isfinite(prices_array)):
            return np.empty((0, self.window_size, 1), dtype=np.float32), np.empty((0, 1), dtype=np.float32)
        self.scaler.fit(prices_array[:-1])
        scaled = self.scaler.transform(prices_array)
        if not np.all(np.isfinite(scaled)):
            return np.empty((0, self.window_size, 1), dtype=np.float32), np.empty((0, 1), dtype=np.float32)

        X, y = [], []
        for i in range(len(scaled) - self.window_size):
            X.append(scaled[i : i + self.window_size])
            y.append(scaled[i + self.window_size])

        X_arr = np.array(X, dtype=np.float32)
        y_arr = np.array(y, dtype=np.float32)
        if not (np.all(np.isfinite(X_arr)) and np.all(np.isfinite(y_arr))):
            return np.empty((0, self.window_size, 1), dtype=np.float32), np.empty((0, 1), dtype=np.float32)
        return X_arr, y_arr

    def train_lstm_model(self) -> bool:
        """Fit the LSTM model on current historical window."""
        X, y = self.prepare_data()
        if len(X) < 2:
            return False

        if torch is None or self.model is None or self.optimizer is None or self.criterion is None:
            X_flat = X.reshape(len(X), -1).astype(np.float64)
            X_aug = np.hstack([X_flat, np.ones((len(X_flat), 1), dtype=np.float64)])
            w, _, _, _ = np.linalg.lstsq(X_aug, y.astype(np.float64), rcond=None)
            w_f32 = w.astype(np.float32)
            finite = bool(np.all(np.isfinite(w_f32)))
            self._fallback_weights = w_f32 if finite else None
            self.trained = finite
            return finite

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        loss_finite = True
        self.model.train()
        for _ in range(self.epochs):
            self.optimizer.zero_grad()
            output = self.model(X_t)
            loss = self.criterion(output, y_t)
            if not bool(torch.isfinite(loss).all().item()):
                loss_finite = False
                break
            loss.backward()
            self.optimizer.step()

        params_finite = loss_finite and all(
            bool(torch.isfinite(param).all().item()) for param in self.model.parameters()
        )
        if not params_finite:
            self._reset_model()
            self._fallback_weights = None
        self.trained = params_finite
        return params_finite

    def predict_next_price(self) -> float | None:
        """Predict the next price step based on the most recent sequence."""
        if not self.trained or len(self.prices) < self.window_size:
            return None

        last_seq = np.array(self.prices[-self.window_size :], dtype=np.float64).reshape(-1, 1)
        if not np.all(np.isfinite(last_seq)):
            return None
        last_seq_scaled = self.scaler.transform(last_seq).reshape(1, self.window_size, 1)
        if not np.all(np.isfinite(last_seq_scaled)):
            return None

        if torch is None or self.model is None:
            if self._fallback_weights is None:
                return None
            x_aug = np.append(last_seq_scaled.reshape(1, -1).astype(np.float32), np.array([[1.0]], dtype=np.float32), axis=1)
            pred_scaled = x_aug @ self._fallback_weights
        else:
            self.model.eval()
            with torch.no_grad():
                X_t = torch.tensor(last_seq_scaled, dtype=torch.float32)
                pred_scaled = self.model(X_t).numpy()
        if not np.all(np.isfinite(pred_scaled)):
            return None
        pred = self.scaler.inverse_transform(np.asarray(pred_scaled, dtype=np.float64))
        return safe_pos_float(pred[0][0], default=None)


if __name__ == "__main__":
    print("Testing PricePrediction PyTorch LSTM...")
    predictor = PricePrediction(window_size=10, epochs=25)
    # Feed synthetic rising prices
    base = 100.0
    for i in range(40):
        base += 1.0 + safe_float(np.sin(i / 3.0), 0.0)
        predictor.process_price(base)

    next_p = predictor.predict_next_price()
    print(f"Last price: {base:.2f}, Predicted next price: {next_p:.2f}")
    assert next_p is not None and next_p > 0
    print("PricePrediction PyTorch model successfully trained and generated prediction.")
