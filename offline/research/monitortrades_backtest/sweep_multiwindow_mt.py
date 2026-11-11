#!/usr/bin/env python3
"""Backtest 1: Multi-Window / Multi-Horizon (`60s`, `90s`, `222s`, `900s`) sweep for monitortrades.py.

Tests whether combining dynamic windows (60s, 90s, 222s, 900s) improves portfolio PnL
over the single 222s window across BTCUSDC, TAOUSDC, ARBUSDC, and HYPEUSD.
"""
from __future__ import annotations

import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

from botcore import safe_pos_float
import pricewindow as pw
from offline.research.monitortrades_backtest.sweep_mt_models import (
    compute_all_signals,
    make_tape,
    run_streaming_mt_model_sweep,
)


def run_multiwindow_sweep(use_365: bool = False):
    label = "DENSE 1-SECOND ARCHIVE (.npz/.jsonl)" if use_365 else "329-DAY SPARSE ARCHIVE"
    print("=" * 108)
    print(f"  BACKTEST 1: MULTI-WINDOW DYNAMIC HORIZONS (60s / 90s / 222s / 900s) IN MONITORTRADES.PY ({label})")
    print("=" * 108)

    def _up_arr(s_win):
        return np.asarray(
            make_tape(s_win["diff"], s_win["eps"], k_eps=0.25, scale_inv=s_win["scale_inv"]),
            dtype=bool,
        )

    models = [
        ("1. Single 60s (1.0m fast)",
         lambda d: _up_arr(d[60]).tolist()),
        ("2. Single 90s (1.5m fast)",
         lambda d: _up_arr(d[90]).tolist()),
        ("3. Single 222s (3.7m PROD Baseline)",
         lambda d: _up_arr(d[222]).tolist()),
        ("4. Single 900s (15.0m medium)",
         lambda d: _up_arr(d[900]).tolist()),
        ("5. Multi OR (222s OR 90s)",
         lambda d: (_up_arr(d[222]) | _up_arr(d[90])).tolist()),
        ("6. Multi AND (222s AND 90s)",
         lambda d: (_up_arr(d[222]) & _up_arr(d[90])).tolist()),
        ("7. Macro-Guarded (222s & !Crash900s)",
         lambda d, sym="": (
             _up_arr(d[222]) & np.where(d[900]["eps"] > 0.0, d[900]["diff"] > (-1.5 * d[900]["scale_inv"] * d[900]["eps"]), True)
         ).tolist()),
        ("8. 2-of-3 Majority (60s, 222s, 900s)",
         lambda d, sym="": ((_up_arr(d[60]).astype(int) + _up_arr(d[222]).astype(int) + _up_arr(d[900]).astype(int)) >= 2).tolist()),
        ("9. Per-Coin PROD (TAO 90s, others 222s)",
         lambda d, sym="": _up_arr(d["prod"]).tolist()),
    ]

    def _build_multiwindow_sigs(sym: str, series):
        prod_w = safe_pos_float(pw.fast_trend_window_for_symbol(sym), default=222.0)
        win_pairs = ((60, 60.0), (90, 90.0), (222, 222.0), (900, 900.0))
        sigs: dict = {
            w: compute_all_signals(series, win_sec=w_sec, symbol=sym, use_symbol_fast_window=False)
            for w, w_sec in win_pairs
        }
        matched = next((sigs[w] for w, w_sec in win_pairs if np.isclose(prod_w, w_sec)), None)
        sigs["prod"] = (
            matched
            if matched is not None
            else compute_all_signals(series, win_sec=prod_w, symbol=sym, use_symbol_fast_window=False)
        )
        return sigs

    run_streaming_mt_model_sweep(
        models,
        _build_multiwindow_sigs,
        use_365=use_365,
        title_col="Multi-Window Policy",
        col_width=36,
    )


if __name__ == "__main__":
    os.chdir(ROOT)
    if "--dense" in sys.argv:
        run_multiwindow_sweep(use_365=True)
    else:
        run_multiwindow_sweep(use_365=False)
