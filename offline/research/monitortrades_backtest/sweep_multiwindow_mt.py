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
import offline.research.monitortrades_backtest.run_replay_backtest as rrb
from offline.research.monitortrades_backtest.sweep_mt_models import (
    compute_all_signals,
    evaluate_mt_model_symbol_cell,
    load_archive_series,
    make_tape,
    print_mt_sweep_header,
    print_mt_sweep_row,
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

    all_symbols = list(rrb.SYMBOLS._SECTIONS.keys())
    symbols: list[str] = []
    model_cells: list[list[str]] = [[] for _ in models]
    model_totals: list[float] = [0.0 for _ in models]

    saved_tape_cache = dict(rrb._TREND_TAPE_CACHE)
    saved_diag = dict(rrb._LAST_RUN_DIAGNOSTICS)
    saved_htp = dict(rrb._LAST_HARD_TP_TOUCHED)
    try:
        for sym in all_symbols:
            series = load_archive_series(sym, use_365=use_365)
            if not series:
                continue
            symbols.append(sym)
            fast_w = pw.fast_trend_window_for_symbol(sym)
            prod_w = safe_pos_float(fast_w, default=222.0)
            sig_60 = compute_all_signals(series, win_sec=60.0, symbol=sym, use_symbol_fast_window=False)
            sig_90 = compute_all_signals(series, win_sec=90.0, symbol=sym, use_symbol_fast_window=False)
            sig_222 = compute_all_signals(series, win_sec=222.0, symbol=sym, use_symbol_fast_window=False)
            sig_900 = compute_all_signals(series, win_sec=900.0, symbol=sym, use_symbol_fast_window=False)
            if np.isclose(prod_w, 60.0):
                sig_prod = sig_60
            elif np.isclose(prod_w, 90.0):
                sig_prod = sig_90
            elif np.isclose(prod_w, 222.0):
                sig_prod = sig_222
            elif np.isclose(prod_w, 900.0):
                sig_prod = sig_900
            else:
                sig_prod = compute_all_signals(series, win_sec=prod_w, symbol=sym, use_symbol_fast_window=False)
            sym_sigs = {
                60: sig_60,
                90: sig_90,
                222: sig_222,
                900: sig_900,
                "prod": sig_prod,
            }
            for m_idx, (_m_name, fn) in enumerate(models):
                net, _bh, cell = evaluate_mt_model_symbol_cell(sym, series, sym_sigs, fn)
                model_totals[m_idx] += net
                model_cells[m_idx].append(cell)
            rrb._TREND_TAPE_CACHE.clear()
            del sym_sigs, series
    finally:
        rrb._TREND_TAPE_CACHE.clear()
        rrb._TREND_TAPE_CACHE.update(saved_tape_cache)
        rrb._LAST_RUN_DIAGNOSTICS.clear()
        rrb._LAST_RUN_DIAGNOSTICS.update(saved_diag)
        rrb._LAST_HARD_TP_TOUCHED.clear()
        rrb._LAST_HARD_TP_TOUCHED.update(saved_htp)

    print_mt_sweep_header("Multi-Window Policy", symbols, col_width=36)
    for m_idx, (m_name, _fn) in enumerate(models):
        print_mt_sweep_row(m_name, model_cells[m_idx], model_totals[m_idx], col_width=36)


if __name__ == "__main__":
    os.chdir(ROOT)
    if "--dense" in sys.argv:
        run_multiwindow_sweep(use_365=True)
    else:
        run_multiwindow_sweep(use_365=False)
