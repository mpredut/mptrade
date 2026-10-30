#!/usr/bin/env python3
"""Walk-forward machine learning benchmark for future trend and price estimation.

Runs alongside existing analysis without trading or replacing it. Produces forecast.json
and an honest walk-forward report measured on unseen data against the Lindy baseline.

Models:
  * lindy: baseline that the latest 24-hour sign persists
  * logit: logistic regression over scaled technical features
  * boost: HistGradientBoosting over multi-horizon returns, MK Z, Hurst, and volatility
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # offline/research/ml_forecast/ -> repo root
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from intelligence.internal.state.survival import fetch_klines  # noqa: E402
from intelligence.internal.state.persistence import calculate_mann_kendall as mann_kendall, calculate_hurst_exponent as hurst_rs  # noqa: E402
from botcore import safe_float, safe_pos_float, safe_int, safe_pos_int  # noqa: E402
from state_io import atomic_write_json  # noqa: E402

_safe_f = safe_float
_safe_int = safe_int

HORIZON_H = 24
WARMUP = 240            # history hours required for features; Hurst uses 240h

FEATURES = ["r1", "r4", "r8", "r24", "r72", "vol24", "vol72", "mk_z", "hurst", "rsi", "snr24"]


def _feat_row(i: int, logp: np.ndarray, px: np.ndarray) -> list[float]:
    r1 = logp[i] - logp[i - 1]
    r4 = logp[i] - logp[i - 4]
    r8 = logp[i] - logp[i - 8]
    r24 = logp[i] - logp[i - 24]
    r72 = logp[i] - logp[i - 72]
    vol24 = float(np.std(np.diff(logp[i - 24:i + 1])))
    vol72 = float(np.std(np.diff(logp[i - 72:i + 1])))
    _, mk_z_raw, _ = mann_kendall(px[i - 24:i + 1])
    mk_z = safe_float(mk_z_raw, 0.0)
    h_raw = hurst_rs(px[i - WARMUP:i + 1])
    h = safe_float(h_raw, 0.5)
    d = np.diff(px[i - 14:i + 1])
    up, dn = d[d > 0].sum(), -d[d < 0].sum()
    rsi = 100.0 * up / (up + dn) if up + dn > 0 else 50.0
    snr24 = r24 / (vol24 * np.sqrt(24) + 1e-12)
    return [r1, r4, r8, r24, r72, vol24, vol72, mk_z, h, rsi, snr24]


def _to_price_array(px) -> np.ndarray:
    if isinstance(px, np.ndarray) and px.dtype.kind in ("f", "i", "u"):
        return px.astype(float, copy=False).reshape(-1)
    if px is None or isinstance(px, (str, bytes, bool)):
        return np.empty(0, dtype=float)
    try:
        items = list(px)
    except TypeError:
        return np.empty(0, dtype=float)
    cleaned = [safe_float(item, default=np.nan) for item in items]
    return np.asarray(cleaned, dtype=float).reshape(-1)


def build_dataset(px: np.ndarray):
    px_arr = _to_price_array(px)
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(np.isfinite(px_arr) & (px_arr > 0), np.log(px_arr), np.nan)
    X, y_dir, y_mag = [], [], []
    for i in range(WARMUP, len(px_arr) - HORIZON_H):
        row = _feat_row(i, logp, px_arr)
        fut = logp[i + HORIZON_H] - logp[i]
        if not (np.isfinite(fut) and all(np.isfinite(v) for v in row)):
            continue
        X.append(row)
        y_dir.append(1 if fut > 0 else 0)
        y_mag.append(fut)
    if not X:
        return np.empty((0, len(FEATURES)), dtype=float), np.empty(0, dtype=int), np.empty(0, dtype=float)
    return np.array(X, dtype=float), np.array(y_dir, dtype=int), np.array(y_mag, dtype=float)


def walk_forward(X, y_dir, y_mag, train_frac=0.7, refit_every=168):
    """Train on the past, predict the UNSEEN future, and refit weekly."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n = len(X)
    if n == 0:
        return {
            "lindy": 0.0,
            "logit": 0.0,
            "boost": 0.0,
            "n_test": 0,
            "mae_move_pct": 0.0,
            "mae_baseline_pct": 0.0,
        }
    refit_every = safe_pos_int(refit_every, default=168)
    frac = min(max(safe_float(train_frac, default=0.7), 0.05), 0.95)
    start = min(max(1, int(n * frac)), n)
    acc = {"lindy": [], "logit": [], "boost": []}
    mag_err, mag_base = [], []
    i = start
    while i < n:
        j = min(i + refit_every, n)
        sl = slice(i, j)
        y_train = y_dir[:i]
        unique_classes = np.unique(y_train)
        reg = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(X[:i], y_mag[:i])
        acc["lindy"] += list((X[sl, 3] > 0).astype(int) == y_dir[sl])
        if len(unique_classes) >= 2:
            clf = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X[:i], y_train)
            logit = make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=1000)).fit(X[:i], y_train)
            acc["logit"] += list(logit.predict(X[sl]) == y_dir[sl])
            acc["boost"] += list(clf.predict(X[sl]) == y_dir[sl])
        else:
            const_pred = int(unique_classes[0]) if len(unique_classes) == 1 else 0
            acc["logit"] += list(const_pred == y_dir[sl])
            acc["boost"] += list(const_pred == y_dir[sl])
        pm = reg.predict(X[sl])
        mag_err += list(np.abs(pm - y_mag[sl]))
        mag_base += list(np.abs(y_mag[sl]))
        i = j
    return {m: (float(np.mean(v)) if v else 0.0) for m, v in acc.items()} | {
        "n_test": len(acc["boost"]),
        "mae_move_pct": float(np.mean(mag_err) * 100) if mag_err else 0.0,
        "mae_baseline_pct": float(np.mean(mag_base) * 100) if mag_base else 0.0,
    }


def live_forecast(px: np.ndarray, X, y_dir, y_mag, rep: dict, symbol: str) -> dict:
    """Train on ALL history and forecast from the latest candle."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    boost_acc = safe_float(rep.get("boost"), default=None) if isinstance(rep, dict) else None
    logit_acc = safe_float(rep.get("logit"), default=None) if isinstance(rep, dict) else None
    lindy_acc = safe_float(rep.get("lindy"), default=None) if isinstance(rep, dict) else None
    use_boost = boost_acc is not None and logit_acc is not None and boost_acc >= logit_acc
    px_arr = _to_price_array(px)
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(np.isfinite(px_arr) & (px_arr > 0), np.log(px_arr), np.nan)
    raw_row = _feat_row(len(px_arr) - 1, logp, px_arr) if len(px_arr) > WARMUP else [0.0] * len(FEATURES)
    clean_row = [safe_float(v, 0.0) for v in raw_row]
    row = np.array([clean_row], dtype=float)
    unique_classes = np.unique(y_dir) if len(y_dir) > 0 else np.array([], dtype=int)
    if len(X) > 0 and len(unique_classes) >= 2:
        if use_boost:
            clf = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X, y_dir)
        else:
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=1000)).fit(X, y_dir)
        proba_up = float(clf.predict_proba(row)[0][1])
    elif len(unique_classes) == 1:
        proba_up = 1.0 if int(unique_classes[0]) == 1 else 0.0
    else:
        proba_up = 0.5
    if len(X) > 0:
        reg = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(X, y_mag)
        move = float(reg.predict(row)[0])
    else:
        move = 0.0
    if boost_acc is not None and logit_acc is not None:
        best_acc = max(boost_acc, logit_acc)
    else:
        best_acc = boost_acc if boost_acc is not None else logit_acc
    return {
        "trend": "up" if proba_up >= 0.5 else "down",
        "confidence": round(abs(proba_up - 0.5) * 2, 2),
        "ts": time.time(),
        "symbol": symbol,
        "horizon_h": HORIZON_H,
        "proba_up": round(proba_up, 3),
        "expected_move_pct": round(move * 100, 2),
        "model": "boost" if use_boost else "logit",
        "walkforward_accuracy": round(float(best_acc), 3) if best_acc is not None else None,
        "baseline_accuracy": round(float(lindy_acc), 3) if lindy_acc is not None else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Walk-forward machine learning forecast benchmark.")
    ap.add_argument("--symbol", default="TAOUSDC")
    ap.add_argument("--days", type=int, default=400)
    ap.add_argument("--eval", action="store_true", help="the walk-forward report only")
    ap.add_argument("--forecast", action="store_true", help="write forecast.json")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "forecast.json"))
    ap.add_argument("--loop", type=float, default=0, help="minutes between forecasts (0 = once)")
    args = ap.parse_args()

    while True:
        ts, px = fetch_klines(args.symbol, args.days)
        if len(px) < WARMUP + HORIZON_H + 100:
            print(f"! insufficient history ({len(px)} candles)")
            return 1
        X, y_dir, y_mag = build_dataset(px)
        print(f"[{args.symbol}] {len(px)} 1h candles -> {len(X)} samples, horizon {HORIZON_H}h")
        rep = walk_forward(X, y_dir, y_mag)
        print(f"  Direction accuracy on {rep['n_test']} UNSEEN hours:")
        print(f"    lindy (persistence): {rep['lindy']:.3f}   logit: {rep['logit']:.3f}   boost: {rep['boost']:.3f}")
        print(f"  24h amplitude:  MAE model {rep['mae_move_pct']:.2f}%  vs baseline(0) {rep['mae_baseline_pct']:.2f}%")
        if args.forecast:
            out = live_forecast(px, X, y_dir, y_mag, rep, args.symbol)
            atomic_write_json(args.out, out, indent=2)
            print(f"  -> {args.out}: trend={out['trend']} conf={out['confidence']} "
                  f"estimated move {out['expected_move_pct']:+.2f}% / {HORIZON_H}h")
        loop_min = safe_pos_float(args.loop, default=None)
        if loop_min is None:
            return 0
        time.sleep(loop_min * 60)


if __name__ == "__main__":
    sys.exit(main())
