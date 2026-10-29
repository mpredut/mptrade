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
os.chdir(ROOT)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

import pricewindow as pw
import offline.research.monitortrades_backtest.run_replay_backtest as rrb
from offline.research.monitortrades_backtest.sweep_mt_models import (
    compute_all_signals,
    load_archive_series,
)


def run_multiwindow_sweep(use_365: bool = False):
    label = "DENSE 1-SECOND ARCHIVE (.npz)" if use_365 else "329-DAY SPARSE ARCHIVE"
    print("=" * 108)
    print(f"  BACKTEST 1: MULTI-WINDOW DYNAMIC HORIZONS (60s / 90s / 222s / 900s) IN MONITORTRADES.PY ({label})")
    print("=" * 108)

    all_symbols = list(rrb.SYMBOLS._SECTIONS.keys())
    symbols = []
    series_map = {}
    sigs = {}

    for sym in all_symbols:
        series = load_archive_series(sym, use_365=use_365)
        if not series:
            continue
        symbols.append(sym)
        series_map[sym] = series
        sigs[sym] = {
            60:  compute_all_signals(series, win_sec=60.0, symbol=sym),
            90:  compute_all_signals(series, win_sec=90.0, symbol=sym),
            222: compute_all_signals(series, win_sec=222.0, symbol=sym),
            900: compute_all_signals(series, win_sec=900.0, symbol=sym),
        }

    def _up_arr(s_win):
        gc = s_win["diff"]
        eps = s_win["eps"]
        sc = s_win["scale_inv"]
        cond = np.where(eps > 0.0, gc > (0.25 * sc * eps), gc > 0.0)
        cond[:1] = False
        return cond

    def _not_crash_arr(s_win, k_crash: float = 1.5):
        gc = s_win["diff"]
        eps = s_win["eps"]
        sc = s_win["scale_inv"]
        return np.where(eps > 0.0, gc > (-k_crash * sc * eps), True)

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
         lambda d, sym="": (_up_arr(d[222]) & _not_crash_arr(d[900], 1.5)).tolist()),
        ("8. 2-of-3 Majority (60s, 222s, 900s)",
         lambda d, sym="": ((_up_arr(d[60]).astype(int) + _up_arr(d[222]).astype(int) + _up_arr(d[900]).astype(int)) >= 2).tolist()),
        ("9. Per-Coin PROD (TAO 90s, others 222s)",
         lambda d, sym="": _up_arr(d[int(pw.fast_trend_window_for_symbol(sym) or 222)]).tolist()),
    ]

    hdr = f"{'Multi-Window Policy':<36} | " + " | ".join(f"{sym:>14}" for sym in symbols) + f" | {'PORTFOLIO NET':>14}"
    print("\n" + hdr)
    print("-" * len(hdr))

    for m_name, fn in models:
        rrb._TREND_TAPE_CACHE.clear()
        cells = []
        tot_net = 0.0
        for sym in symbols:
            series = series_map[sym]
            key = rrb._series_cache_key(sym, series, rrb.DEFAULT_WINDOW_SECONDS)
            rrb._TREND_TAPE_CACHE[key] = fn(sigs[sym], sym) if fn.__code__.co_argcount >= 2 else fn(sigs[sym])
            cfg = rrb.SYMBOLS[sym]
            res = rrb.run_symbol(sym, cfg["params"], cfg["base"], quiet=True, series=series)
            net = res["net"]
            tot_net += net
            cells.append(f"{net:>+8.2f} ({res['buys']:>2}B/{res['sells']:>2}S)")
        print(f"{m_name:<36} | " + " | ".join(f"{c:>14}" for c in cells) + f" | {tot_net:>+14.2f}")


if __name__ == "__main__":
    if "--dense" in sys.argv:
        run_multiwindow_sweep(use_365=True)
    else:
        run_multiwindow_sweep(use_365=False)
