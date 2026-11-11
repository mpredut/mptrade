#!/usr/bin/env python3
"""Compare MHDSL Structural Break PROPOSED vs BASELINE on historical replay."""
from multiprocessing import Pool
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

import monitortrades as mt
from botcore import _parse_bool_flag, safe_float
from offline.research.monitortrades_backtest.run_replay_backtest import (
    SYMBOLS,
    prewarm_trend_series,
    run_symbol,
)
from offline.research.monitortrades_backtest.sweep_mt_models import load_archive_series

_MIN_DENSE_SPAN_SEC = 30.0 * 86400.0
_WORKER_SERIES: dict[str, list[tuple[float, float]]] = {}


def _series_span_sec(series: list[tuple[float, float]]) -> float:
    if len(series) < 2:
        return 0.0
    return max(0.0, float(series[-1][0]) - float(series[0][0]))


def _run_variant_symbol(task_args):
    label, enabled, sym, params, base = task_args
    orig_break_enabled = getattr(mt, "MT_SL_STRUCTURAL_BREAK_ENABLED", True)
    try:
        mt.MT_SL_STRUCTURAL_BREAK_ENABLED = enabled
        mt._CACHED_EXIT_MGR = None
        mt._CACHED_EXIT_MGR_KEY = None
        res = run_symbol(sym, params, base, quiet=True, series=_WORKER_SERIES.get(sym))
        return label, sym, res
    finally:
        mt.MT_SL_STRUCTURAL_BREAK_ENABLED = orig_break_enabled
        mt._CACHED_EXIT_MGR = None
        mt._CACHED_EXIT_MGR_KEY = None


def main():
    variants = [
        ("PROPOSED (structural break ON)", True),
        ("BASELINE (structural break OFF)", False),
    ]
    results = {label: {} for label, _ in variants}
    use_365 = _parse_bool_flag(os.environ.get("MPTRADE_REPLAY_USE_365"), True)
    series_map = {}
    for sym in SYMBOLS._SECTIONS:
        series = load_archive_series(sym, use_365=use_365)
        if use_365 and _series_span_sec(series) < _MIN_DENSE_SPAN_SEC:
            sparse = load_archive_series(sym, use_365=False)
            if _series_span_sec(sparse) > _series_span_sec(series):
                series = sparse
        series_map[sym] = series
    for sym, series in series_map.items():
        if series:
            prewarm_trend_series(sym, series)
    _WORKER_SERIES.clear()
    _WORKER_SERIES.update(series_map)

    tasks = [
        (label, enabled, sym, SYMBOLS[sym]["params"], SYMBOLS[sym]["base"])
        for label, enabled in variants
        for sym in SYMBOLS._SECTIONS
    ]
    n_workers = min(len(tasks), 4, os.cpu_count() or 1)
    try:
        if n_workers > 1:
            with Pool(processes=n_workers) as pool:
                task_outputs = pool.map(_run_variant_symbol, tasks)
        else:
            task_outputs = [_run_variant_symbol(t) for t in tasks]
    finally:
        _WORKER_SERIES.clear()

    for label, sym, res in task_outputs:
        results[label][sym] = res

    for label, _ in variants:
        print("\n=======================================================")
        print(f"  RUNNING {label}")
        print("=======================================================")
        for sym in SYMBOLS._SECTIONS:
            res = results[label].get(sym)
            if res:
                net_val = safe_float(res.get("net"), 0.0)
                print(f"[{sym}] Net PnL: {net_val:.2f} USD | Sells: {res.get('sells') or 0} | Buys: {res.get('buys') or 0}")
            else:
                print(f"[{sym}] No results or no data.")

    print("\n\n" + "=" * 60)
    print("           BACKTEST COMPARISON SUMMARY")
    print("=" * 60)
    for sym in SYMBOLS._SECTIONS:
        prop = results[variants[0][0]].get(sym) or {}
        base = results[variants[1][0]].get(sym) or {}
        prop_net = safe_float(prop.get("net"), default=None) if isinstance(prop, dict) else None
        base_net = safe_float(base.get("net"), default=None) if isinstance(base, dict) else None
        if prop_net is not None and base_net is not None:
            diff = prop_net - base_net
            print(f"\n{sym}:")
            print(f"  Proposed (Break ON) : {prop_net:.2f} USD ({prop.get('sells', 0)} sells)")
            print(f"  Baseline (Break OFF): {base_net:.2f} USD ({base.get('sells', 0)} sells)")
            print(f"  Difference          : {'+' if diff >= 0 else ''}{diff:.2f} USD ({'OUTPERFORMS' if diff > 0 else 'UNDERPERFORMS' if diff < 0 else 'EQUAL'})")
        else:
            print(f"\n{sym}: Incomplete data.")

if __name__ == "__main__":
    main()
