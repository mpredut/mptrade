
import os
import sys
import math
import json
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
### my import
from botcore import safe_float, safe_int, safe_pos_int
import symbols as sym
import cacheManager as cm


def _clean_finite_floats(seq) -> List[float]:
    if seq is None or isinstance(seq, bool):
        return []
    try:
        return [
            vf for x in seq
            if (vf := safe_float(x, default=None)) is not None
        ]
    except TypeError:
        return []


def get_cache_for_symbol(symbol: str) -> List[Tuple[int, float]]:
    if not symbol or isinstance(symbol, bool) or not str(symbol).strip():
        return []
    sym_key = str(symbol).strip().upper()
    mgr = getattr(cm, "price_cache_manager", None)
    entry = (mgr.get(sym_key) or mgr.get(symbol)) if isinstance(mgr, dict) else None
    raw = getattr(entry, "cache", None) if entry is not None else None
    if isinstance(raw, dict):
        raw = raw.get(sym_key, raw.get(symbol, []))
    if not raw or not isinstance(raw, (list, tuple)):
        return []
    out: List[Tuple[int, float]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        t_f = safe_float(item[0], default=None)
        p_val = safe_float(item[1], default=None)
        if t_f is None or p_val is None:
            continue
        out.append((int(t_f), p_val))
    return out


def to_series(cache_pairs: List[Tuple[int, float]]):
    """Convert [timestamp_ms, price] pairs to chronologically sorted NumPy arrays."""
    if not cache_pairs:
        return np.array([]), np.array([]), []
    valid = []
    for item in cache_pairs:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        t_f = safe_float(item[0], default=None)
        p_val = safe_float(item[1], default=None)
        if t_f is None or p_val is None:
            continue
        try:
            t_val = int(t_f)
            dt_val = datetime.fromtimestamp(t_val / 1000.0, timezone.utc).replace(tzinfo=None)
        except (TypeError, ValueError, OverflowError, OSError):
            continue
        if 0 <= t_val <= (1 << 63) - 1:
            valid.append((t_val, p_val, dt_val))
    if not valid:
        return np.array([]), np.array([]), []
    # Chronological ordering.
    cache_pairs = sorted(valid, key=lambda x: x[0])
    ts = np.array([int(x[0]) for x in cache_pairs], dtype=np.int64)
    prices = np.array([float(x[1]) for x in cache_pairs], dtype=np.float64)
    dt = [x[2] for x in cache_pairs]
    return ts, prices, dt

def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path)

def min_max_scale(arr: np.ndarray):
    """Scale to [0, 1] without sklearn and return (scaled, minimum, maximum)."""
    if arr is None or isinstance(arr, (bool, str, bytes)) or len(arr) == 0:
        return np.array([], dtype=np.float64), 0.0, 0.0
    try:
        arr_f = np.asarray(arr, dtype=np.float64)
    except (TypeError, ValueError):
        clean = _clean_finite_floats(arr)
        if not clean:
            return np.array([], dtype=np.float64), 0.0, 0.0
        arr_f = np.asarray(clean, dtype=np.float64)
    vmin = float(np.min(arr_f))
    vmax = float(np.max(arr_f))
    if not (math.isfinite(vmin) and math.isfinite(vmax)):
        return np.full_like(arr_f, 0.5, dtype=np.float64), 0.0, 0.0
    if vmax == vmin or not math.isfinite(vmax - vmin):
        # Map a constant series to 0.5 to avoid division by zero.
        return np.full_like(arr_f, 0.5, dtype=np.float64), vmin, vmax
    scaled = (arr_f - vmin) / (vmax - vmin)
    return scaled, vmin, vmax

def min_max_inverse(arr_scaled: np.ndarray, vmin: float, vmax: float):
    if arr_scaled is None or isinstance(arr_scaled, (bool, str, bytes)) or len(arr_scaled) == 0:
        return np.array([], dtype=np.float64)
    arr_f = np.asarray(arr_scaled, dtype=np.float64)
    vmin_f = safe_float(vmin, default=None)
    vmax_f = safe_float(vmax, default=None)
    if vmin_f is None or vmax_f is None:
        return np.zeros_like(arr_f, dtype=np.float64)
    if vmax_f == vmin_f or not math.isfinite(vmax_f - vmin_f):
        return np.full_like(arr_f, vmin_f, dtype=np.float64)
    return arr_f * (vmax_f - vmin_f) + vmin_f


# ==========================
# 3) HISTORICAL PLOT
# ==========================

def plot_history(symbol: str, dt, prices: np.ndarray, outdir="plots"):
    ensure_dir(outdir)
    plt.figure(figsize=(10, 5))
    plt.plot(dt, prices, marker="o", linestyle="-")
    plt.title(f"Price evolution - {symbol}")
    plt.xlabel("Timp")
    plt.ylabel("Price")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    fname = os.path.join(outdir, f"history_{symbol}.png")
    plt.savefig(fname, dpi=150)
    plt.show()   # Display on screen.
    plt.close()
    return fname


# ==========================
# 4) LINEAR REGRESSION (np.polyfit)
# ==========================

def forecast_linear(ts_ms: np.ndarray, prices: np.ndarray, horizon: int = 20):
    """
    Regress on point indices rather than millisecond time for numeric stability.

    For real elapsed time, normalize milliseconds to days or hours and use the
    same formulas.
    """
    if prices is None or isinstance(prices, (bool, str, bytes)) or isinstance(horizon, bool):
        return np.array([], dtype=np.float64)
    hz_f = safe_float(horizon, default=None)
    if hz_f is None or int(hz_f) <= 0:
        return np.array([], dtype=np.float64)
    hz = int(hz_f)
    clean_p = _clean_finite_floats(prices)
    n = len(clean_p)
    if n == 0:
        return np.array([], dtype=np.float64)
    if n == 1:
        return np.full(hz, clean_p[0], dtype=np.float64)
    mean_x = 0.5 * (n - 1)
    x = np.arange(n, dtype=np.float64)
    y = np.asarray(clean_p, dtype=np.float64)
    # fit y ~ a*x + b via closed-form OLS
    dev_x = x - mean_x
    var_x = n * (n * n - 1.0) / 12.0
    a = float(np.dot(dev_x, y) / var_x) if var_x > 0.0 else 0.0
    b = float(np.mean(y)) - a * mean_x
    # Predictions for the following steps.
    x_future = np.arange(n, n + hz, dtype=np.float64)
    y_future = a * x_future + b
    return y_future

def plot_with_linear(symbol: str, dt, prices: np.ndarray, lin_forecast: np.ndarray, outdir="plots"):
    ensure_dir(outdir)
    n = len(prices)
    future_dt = []
    if lin_forecast is not None and len(lin_forecast) > 0 and n >= 2:
        step = dt[-1] - dt[-2]
        if step.total_seconds() <= 0:
            avg_step = (dt[-1] - dt[0]).total_seconds() / max(1, (n - 1))
            step = timedelta(seconds=avg_step if avg_step > 0 else 60.0)
        future_dt = [dt[-1] + (i + 1) * step for i in range(len(lin_forecast))]
    plt.figure(figsize=(10, 5))
    plt.plot(dt, prices, marker="o", linestyle="-", label="Istoric")
    if future_dt:
        plt.plot(future_dt, lin_forecast, marker="o", linestyle="--", label="Regresie (forecast)")
    plt.title(f"Forecast (linear regression) - {symbol}")
    plt.xlabel("Timp")
    plt.ylabel("Price")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(outdir, f"forecast_linear_{symbol}.png")
    plt.savefig(fname, dpi=150)
    plt.show()   # Display on screen.
    plt.close()
    return fname


# ==========================
# 5) LSTM
# ==========================


def make_sequences(series: np.ndarray, window: int):
    win = safe_pos_int(window, 1)
    if series is None or isinstance(series, (bool, str, bytes)) or len(series) <= win:
        return np.empty((0, win, 1), dtype=np.float64), np.empty((0,), dtype=np.float64)
    X, y = [], []
    for i in range(win, len(series)):
        X.append(series[i-win:i])
        y.append(series[i])
    X = np.array(X)
    y = np.array(y)
    # Reshape to (samples, timesteps, features).
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y

def try_import_tf():
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        return True, tf, Sequential, LSTM, Dense
    except Exception as e:
        return False, e, None, None, None

def forecast_lstm(prices: np.ndarray, window: int = 20, epochs: int = 50, batch_size: int = 16,
                  horizon: int = 20, verbose: int = 0):
    """
    Return (predictions, info_text).

    If TensorFlow is unavailable or the series is too short, return (None, reason).
    """
    ok, tf_or_err, Sequential, LSTM, Dense = try_import_tf()
    if not ok:
        return None, f"LSTM indisponibil: {tf_or_err}"

    window = safe_pos_int(window, 20)
    epochs = safe_pos_int(epochs, 50)
    batch_size = safe_pos_int(batch_size, 16)
    horizon = safe_pos_int(horizon, 20)

    if prices is None or isinstance(prices, (bool, str, bytes)):
        return None, f"The series is too short for LSTM (at least ~{window+5} points)."
    prices = np.asarray(_clean_finite_floats(prices), dtype=np.float64)

    if len(prices) < window + 5:
        return None, f"The series is too short for LSTM (at least ~{window+5} points)."

    # Scale to [0, 1].
    scaled, vmin, vmax = min_max_scale(prices)
    X, y = make_sequences(scaled, window)
    # Simple training/validation split.
    split = max(window, int(0.8 * len(X)))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    model = Sequential()
    model.add(LSTM(64, input_shape=(window, 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train,
              validation_data=(X_val, y_val) if len(X_val) else None,
              epochs=epochs, batch_size=batch_size, verbose=verbose)

    # Iterative forecast over the requested horizon.
    last_window = scaled[-window:].reshape(1, window, 1)
    preds_scaled = []
    for _ in range(horizon):
        pred = model.predict(last_window, verbose=0)[0, 0]
        preds_scaled.append(pred)
        # Slide the window.
        last_window = np.concatenate([last_window[:, 1:, :], pred.reshape(1, 1, 1)], axis=1)

    preds = min_max_inverse(np.array(preds_scaled), vmin, vmax)
    return preds, f"LSTM antrenat: window={window}, epochs={epochs}, batch={batch_size}"
    
def plot_with_lstm(symbol: str, dt, prices: np.ndarray, lstm_forecast: Optional[np.ndarray], outdir="plots"):
    ensure_dir(outdir)
    n = len(prices)
    future_dt = []

    if lstm_forecast is not None and len(lstm_forecast) > 0 and n >= 2:
        step = (dt[-1] - dt[-2])
        if step.total_seconds() <= 0:
            avg_step = (dt[-1] - dt[0]).total_seconds() / max(1, (n - 1))
            step = timedelta(seconds=avg_step if avg_step > 0 else 60.0)

        future_dt = [dt[-1] + step*(i+1) for i in range(len(lstm_forecast))]

    plt.figure(figsize=(10, 5))
    plt.plot(dt, prices, marker="o", linestyle="-", label="Istoric")
    if lstm_forecast is not None and len(lstm_forecast) > 0 and future_dt:
        plt.plot(future_dt, lstm_forecast, marker="o", linestyle="--", label="LSTM (forecast)")
    plt.title(f"Forecast (LSTM) - {symbol}")
    plt.xlabel("Timp")
    plt.ylabel("Price")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(outdir, f"forecast_lstm_{symbol}.png")
    plt.savefig(fname, dpi=150)
    plt.show()   # Display on screen.
    plt.close()
    return fname


# ==========================
# 6) PIPELINE FOR ALL SYMBOLS
# ==========================

def run_for_symbols(symbols: List[str],
                    outdir: str = "plots",
                    horizon: int = 20,
                    lstm_window: int = 20,
                    lstm_epochs: int = 50,
                    lstm_batch: int = 16,
                    lstm_verbose: int = 0):
    ensure_dir(outdir)
    results = {}
    print(f" MARIUS OK: ")
    for symbol in symbols:
        print(f" MARIUS SYM:{symbol} ")
        pairs = get_cache_for_symbol(symbol)
        ts, prices, dt = to_series(pairs)

        if len(prices) == 0:
            results[symbol] = {"error": "No data in the cache."}
            continue

        print(f" plot history{symbol} ")
        
        hist_path = plot_history(symbol, dt, prices, outdir=outdir)

        print(f" regresie:{symbol} ")
        # Linear regression.
        lin_pred = forecast_linear(ts, prices, horizon=horizon)
        lin_path = plot_with_linear(symbol, dt, prices, lin_pred, outdir=outdir)

        # LSTM, when available.
        print(f" forecast_lstm :{symbol} ")
        lstm_pred, info = forecast_lstm(prices,
                                        window=lstm_window,
                                        epochs=lstm_epochs,
                                        batch_size=lstm_batch,
                                        horizon=horizon,
                                        verbose=lstm_verbose)
        lstm_path = None
        if lstm_pred is not None:
            lstm_path = plot_with_lstm(symbol, dt, prices, lstm_pred, outdir=outdir)

        results[symbol] = {
            "history_png": hist_path,
            "linear_forecast_png": lin_path,
            "lstm_info": info,
            "lstm_forecast_png": lstm_path,
        }
        print(f"[{symbol}] OK: {results[symbol]}")
    return results


# ==========================
# 7) MAIN
# ==========================

if __name__ == "__main__":
   
    symbols = sym.symbols

    # 2) Run the pipeline.
    res = run_for_symbols(
        symbols,
        outdir="plots",
        horizon=20,       # number of future points to estimate
        lstm_window=10,   # LSTM window; use 20-60 for longer series
        lstm_epochs=50,   # increase for accuracy at the cost of runtime
        lstm_batch=16,
        lstm_verbose=0
    )

    # 3) Optionally save a JSON report of the results.
    with open("forecast_results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("\nThe images were saved in ./plots and the summary in forecast_results.json")
