#!/usr/bin/env python3
"""Chronos zero-shot foundation model volatility prediction benchmark.

Tests whether Amazon's zero-shot Chronos foundation model predicts future realized
volatility better than simple persistence.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # offline/research/ml_forecast/ -> repo root
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from intelligence.internal.state.survival import fetch_klines  # noqa: E402

MODEL_NAME = "amazon/chronos-t5-tiny"   # smallest 8M-parameter model; CPU-friendly
MAX_CONTEXT = 512                       # maximum history hours supplied per prediction
BATCH = 8                               # test windows batched per call


def realized_vol_series(px: np.ndarray, win: int) -> np.ndarray:
    """Return trailing [i-win, i] log-return standard deviation, with NaN before history."""
    logp = np.log(px)
    r = np.diff(logp)
    rv = np.full(len(px), np.nan)
    for i in range(win, len(px)):
        rv[i] = np.std(r[i - win:i])
    return rv


def _load_pipeline():
    from chronos import ChronosPipeline
    print(f"  loading {MODEL_NAME} (zero-shot, no training)...")
    return ChronosPipeline.from_pretrained(MODEL_NAME, device_map="cpu", torch_dtype=torch.float32)


def walk_forward_vol(rv: np.ndarray, horizon: int, warmup: int, stride: int):
    """For each test hour i sampled by stride, give the model rv[:i+1] capped at
    MAX_CONTEXT and request a +horizon-hour prediction. Compare against persistence
    rv[i] and actual rv[i+horizon]."""
    pipe = _load_pipeline()
    n = len(rv)
    idxs = list(range(warmup, n - horizon, stride))
    preds, bases, actuals = [], [], []
    t0 = time.time()
    for b in range(0, len(idxs), BATCH):
        chunk = idxs[b:b + BATCH]
        contexts = [torch.tensor(rv[max(0, i + 1 - MAX_CONTEXT):i + 1], dtype=torch.float32)
                    for i in chunk]
        forecast = pipe.predict(contexts, prediction_length=horizon)
        for k, i in enumerate(chunk):
            path = forecast[k].numpy()
            point = float(np.median(np.median(path, axis=0)))
            preds.append(point)
            bases.append(float(rv[i]))
            actuals.append(float(rv[i + horizon]))
        done = b + len(chunk)
        print(f"    {done}/{len(idxs)} windows tested ({time.time()-t0:.0f}s)", end="\r")
    print()
    preds, bases, actuals = map(np.array, (preds, bases, actuals))
    mae_model = float(np.mean(np.abs(preds - actuals)))
    mae_base = float(np.mean(np.abs(bases - actuals)))
    dir_actual = (actuals - bases) > 0
    dir_pred = (preds - bases) > 0
    dir_acc = float(np.mean(dir_actual == dir_pred))
    corr = float(np.corrcoef(preds, actuals)[0, 1]) if len(preds) > 2 else float("nan")
    return {
        "n_test": len(preds),
        "mae_model": mae_model,
        "mae_baseline_persistence": mae_base,
        "improvement_pct": round(100 * (1 - mae_model / mae_base), 1) if mae_base else None,
        "direction_accuracy": round(dir_acc, 3),
        "corr_model_vs_actual": round(corr, 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Chronos zero-shot volatility forecast benchmark.")
    ap.add_argument("--symbol", default="TAOUSDC")
    ap.add_argument("--days", type=int, default=400)
    ap.add_argument("--win", type=int, default=24, help="the window (hours) for the realized volatility")
    ap.add_argument("--horizon", type=int, default=24, help="how many hours ahead we predict")
    ap.add_argument("--stride", type=int, default=6, help="the step between test windows (hours)")
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()

    ts, px = fetch_klines(args.symbol, args.days)
    print(f"[{args.symbol}] {len(px)} 1h candles")
    rv = realized_vol_series(px, args.win)
    warmup = args.win + MAX_CONTEXT // 4
    if args.eval:
        rep = walk_forward_vol(rv, args.horizon, warmup, args.stride)
        print(f"  tested on {rep['n_test']} windows (stride={args.stride}h), "
              f"horizon={args.horizon}h, vol_window={args.win}h")
        print(f"    MAE model={rep['mae_model']:.5f}  baseline(persistence)={rep['mae_baseline_persistence']:.5f}"
              f"  -> {rep['improvement_pct']}% {'better' if (rep['improvement_pct'] or 0) > 0 else 'worse'}")
        print(f"    direction accuracy (vol rises/falls): {rep['direction_accuracy']}")
        print(f"    correlation model vs actual: {rep['corr_model_vs_actual']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
