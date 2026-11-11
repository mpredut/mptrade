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
from intelligence.internal.state.persistence import calculate_mann_kendall, calculate_hurst_exponent  # noqa: E402
from botcore import (  # noqa: E402
    normalize_epoch_sec,
    safe_float,
    safe_int,
    safe_nonneg_float,
    safe_pos_float,
    safe_pos_int,
)
from offline.backtests.evaluation import parse_cli_float, parse_cli_int  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from state_io import atomic_write_json  # noqa: E402

HORIZON_H = 24
WARMUP = 240            # history hours required for features; Hurst uses 240h

FEATURES = ["r1", "r4", "r8", "r24", "r72", "vol24", "vol72", "mk_z", "hurst", "rsi", "snr24"]


def _feat_row(i: int, logp: np.ndarray, px: np.ndarray) -> list[float]:
    r1 = logp[i] - logp[i - 1]
    r4 = logp[i] - logp[i - 4]
    r8 = logp[i] - logp[i - 8]
    r24 = logp[i] - logp[i - 24]
    r72 = logp[i] - logp[i - 72]
    vol24 = safe_float(np.std(np.diff(logp[i - 24:i + 1])), default=np.nan)
    vol72 = safe_float(np.std(np.diff(logp[i - 72:i + 1])), default=np.nan)
    _, mk_z_raw, _ = calculate_mann_kendall(px[i - 24:i + 1])
    mk_z = safe_float(mk_z_raw, 0.0)
    h_raw = calculate_hurst_exponent(px[i - WARMUP:i + 1])
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


def build_dataset(px: np.ndarray, ts: np.ndarray | None = None):
    px_arr = _to_price_array(px)
    ts_arr = None
    if ts is not None:
        raw_ts = _to_price_array(ts)
        if len(raw_ts) == len(px_arr):
            norm_ts = [
                normalize_epoch_sec(v, default=np.nan, allow_zero=True)
                for v in raw_ts
            ]
            ts_arr = np.asarray(
                [v if v is not None else np.nan for v in norm_ts],
                dtype=float,
            )
    expected_span_sec = (WARMUP + HORIZON_H) * 3600.0
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(np.isfinite(px_arr) & (px_arr > 0), np.log(px_arr), np.nan)
    X, y_dir, y_mag = [], [], []
    for i in range(WARMUP, len(px_arr) - HORIZON_H):
        if ts_arr is not None:
            t_start = ts_arr[i - WARMUP]
            t_end = ts_arr[i + HORIZON_H]
            if not (np.isfinite(t_start) and np.isfinite(t_end)):
                continue
            if abs((t_end - t_start) - expected_span_sec) > 1800.0:
                continue
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
    min_start = (HORIZON_H + 1) if n > HORIZON_H else 1
    start = min(max(min_start, safe_int(n * frac, default=min_start)), n)
    acc = {"lindy": [], "logit": [], "boost": []}
    mag_err, mag_base = [], []
    i = start
    while i < n:
        j = min(i + refit_every, n)
        sl = slice(i, j)
        tr_end = i - HORIZON_H if i > HORIZON_H else i
        X_tr = X[:tr_end]
        y_train = y_dir[:tr_end]
        y_mag_tr = y_mag[:tr_end]
        unique_classes = np.unique(y_train)
        reg = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(X_tr, y_mag_tr)
        acc["lindy"] += list((X[sl, 3] > 0).astype(int) == y_dir[sl])
        if len(unique_classes) >= 2:
            clf = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X_tr, y_train)
            logit = make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=1000)).fit(X_tr, y_train)
            acc["logit"] += list(logit.predict(X[sl]) == y_dir[sl])
            acc["boost"] += list(clf.predict(X[sl]) == y_dir[sl])
        else:
            const_pred = safe_int(unique_classes[0], default=0) if len(unique_classes) == 1 else 0
            acc["logit"] += list(const_pred == y_dir[sl])
            acc["boost"] += list(const_pred == y_dir[sl])
        pm = reg.predict(X[sl])
        mag_err += list(np.abs(pm - y_mag[sl]))
        mag_base += list(np.abs(y_mag[sl]))
        i = j
    return {m: (safe_float(np.mean(v), 0.0) if v else 0.0) for m, v in acc.items()} | {
        "n_test": len(acc["boost"]),
        "mae_move_pct": safe_float(np.mean(mag_err) * 100, 0.0) if mag_err else 0.0,
        "mae_baseline_pct": safe_float(np.mean(mag_base) * 100, 0.0) if mag_base else 0.0,
    }


def validate_forecast_symbol(raw: object) -> str | None:
    if not isinstance(raw, str) or isinstance(raw, bool):
        return None
    sym = raw.strip().upper()
    if (
        not sym
        or sym.startswith("-")
        or ".." in sym
        or "/" in sym
        or "\\" in sym
        or any(ch.isspace() for ch in sym)
    ):
        return None
    return sym


def live_forecast(
    px: np.ndarray,
    X,
    y_dir,
    y_mag,
    rep: dict,
    symbol: str,
    ts: np.ndarray | None = None,
) -> dict:
    """Train on ALL history and forecast from the latest candle."""
    boost_acc = safe_float(rep.get("boost"), default=None) if isinstance(rep, dict) else None
    logit_acc = safe_float(rep.get("logit"), default=None) if isinstance(rep, dict) else None
    lindy_acc = safe_float(rep.get("lindy"), default=None) if isinstance(rep, dict) else None
    use_boost = boost_acc is not None and (logit_acc is None or boost_acc >= logit_acc)
    px_arr = _to_price_array(px)
    gap_detected = False
    if ts is not None:
        raw_ts = _to_price_array(ts)
        if len(raw_ts) != len(px_arr) or len(px_arr) <= WARMUP:
            gap_detected = True
        else:
            t_start = normalize_epoch_sec(raw_ts[-1 - WARMUP], default=None, allow_zero=True)
            t_end = normalize_epoch_sec(raw_ts[-1], default=None, allow_zero=True)
            if (
                t_start is None
                or t_end is None
                or abs((t_end - t_start) - WARMUP * 3600.0) > 1800.0
            ):
                gap_detected = True
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(np.isfinite(px_arr) & (px_arr > 0), np.log(px_arr), np.nan)
    has_valid_features = False
    if not gap_detected and len(px_arr) > WARMUP:
        raw_row = _feat_row(len(px_arr) - 1, logp, px_arr)
        has_valid_features = all(np.isfinite(v) for v in raw_row)
    else:
        raw_row = [0.0] * len(FEATURES)
    clean_row = [safe_float(v, 0.0) for v in raw_row]
    row = np.array([clean_row], dtype=float)
    unique_classes = np.unique(y_dir) if len(y_dir) > 0 else np.array([], dtype=int)
    if has_valid_features and len(X) > 0 and len(unique_classes) >= 2:
        if use_boost:
            clf = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X, y_dir)
        else:
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=1000)).fit(X, y_dir)
        proba_up = safe_float(clf.predict_proba(row)[0][1], 0.5)
    elif not gap_detected and len(unique_classes) == 1:
        proba_up = 1.0 if safe_int(unique_classes[0], 0) == 1 else 0.0
    else:
        proba_up = 0.5
    if has_valid_features and len(X) > 0:
        reg = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(X, y_mag)
        move = safe_float(reg.predict(row)[0], 0.0)
    else:
        move = 0.0
    selected_acc = boost_acc if use_boost else logit_acc
    if selected_acc is None:
        selected_acc = boost_acc if boost_acc is not None else logit_acc
    out = {
        "trend": "up" if proba_up >= 0.5 else "down",
        "confidence": round(abs(proba_up - 0.5) * 2, 2),
        "ts": time.time(),
        "symbol": symbol,
        "horizon_h": HORIZON_H,
        "proba_up": round(proba_up, 3),
        "expected_move_pct": round(move * 100, 2),
        "model": "boost" if use_boost else "logit",
        "walkforward_accuracy": round(selected_acc, 3) if selected_acc is not None else None,
        "baseline_accuracy": round(lindy_acc, 3) if lindy_acc is not None else None,
    }
    if gap_detected:
        out["gap_detected"] = True
    return out


def create_ml_forecast_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--symbol", default="TAOUSDC")
    ap.add_argument("--days", type=parse_cli_int, default=400)
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = create_ml_forecast_parser("Walk-forward machine learning forecast benchmark.")
    ap.add_argument("--eval", action="store_true", help="the walk-forward report only")
    ap.add_argument("--forecast", action="store_true", help="write forecast.json")
    ap.add_argument(
        "--out",
        default=os.path.join(_ROOT, "offline", "results", "ml_forecast", "forecast.json"),
    )
    ap.add_argument("--loop", type=parse_cli_float, default=0.0, help="minutes between forecasts (0 = once)")
    args = ap.parse_args(argv)
    if (symbol := validate_forecast_symbol(args.symbol)) is None:
        ap.error(f"invalid --symbol: {args.symbol!r}")
    if (days := safe_pos_int(args.days, default=None)) is None:
        ap.error("--days must be positive")
    if (loop_nonneg := safe_nonneg_float(args.loop, default=None)) is None:
        ap.error("--loop must be >= 0")
    loop_min = safe_pos_float(loop_nonneg, default=None)

    while True:
        ts, px = fetch_klines(symbol, days)
        if len(px) < WARMUP + HORIZON_H + 100:
            print(f"! insufficient history ({len(px)} candles)", file=sys.stderr)
            if loop_min is None:
                return 1
            time.sleep(loop_min * 60)
            continue
        X, y_dir, y_mag = build_dataset(px, ts=ts)
        print(f"[{symbol}] {len(px)} 1h candles -> {len(X)} samples, horizon {HORIZON_H}h")
        if len(X) == 0:
            print(f"! no valid contiguous samples for {symbol}; refusing to overwrite {args.out}", file=sys.stderr)
            if loop_min is None:
                return 1
            time.sleep(loop_min * 60)
            continue
        rep = walk_forward(X, y_dir, y_mag)
        print(f"  Direction accuracy on {rep['n_test']} UNSEEN hours:")
        print(f"    lindy (persistence): {rep['lindy']:.3f}   logit: {rep['logit']:.3f}   boost: {rep['boost']:.3f}")
        print(f"  24h amplitude:  MAE model {rep['mae_move_pct']:.2f}%  vs baseline(0) {rep['mae_baseline_pct']:.2f}%")
        if args.forecast:
            out = live_forecast(px, X, y_dir, y_mag, rep, symbol, ts=ts)
            if out.get("gap_detected"):
                print(
                    f"! latest {WARMUP}h window has timestamp gap for {symbol}; refusing to overwrite {args.out}",
                    file=sys.stderr,
                )
                if loop_min is None:
                    return 1
                time.sleep(loop_min * 60)
                continue
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            atomic_write_json(args.out, out, indent=2)
            print(f"  -> {args.out}: trend={out['trend']} conf={out['confidence']} "
                  f"estimated move {out['expected_move_pct']:+.2f}% / {HORIZON_H}h")
        if loop_min is None:
            return 0
        time.sleep(loop_min * 60)


if __name__ == "__main__":
    sys.exit(main())
