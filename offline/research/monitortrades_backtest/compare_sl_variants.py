#!/usr/bin/env python3
"""Compare MHDSL Structural Break PROPOSED vs BASELINE on historical replay."""
import gc
import multiprocessing as mp
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

import monitortrades as mt
from botcore import bool_env, safe_float
from offline.research.monitortrades_backtest.run_replay_backtest import (
    SYMBOLS,
    _TREND_TAPE_CACHE,
    _resolve_replay_worker_cap,
    prewarm_trend_series,
    run_symbol,
)
from offline.research.monitortrades_backtest.sweep_mt_models import load_archive_series

_MIN_DENSE_SPAN_SEC = 30.0 * 86400.0
_WORKER_SERIES: dict[str, list[tuple[float, float]]] = {}


def _init_worker(series_map: dict[str, list[tuple[float, float]]]) -> None:
    _WORKER_SERIES.clear()
    _WORKER_SERIES.update(series_map)


def _series_span_sec(series: list[tuple[float, float]]) -> float:
    if len(series) < 2:
        return 0.0
    t0 = safe_float(series[0][0], 0.0)
    t1 = safe_float(series[-1][0], 0.0)
    return max(0.0, t1 - t0)


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
    orig_cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        variants = [
            ("PROPOSED (structural break ON)", True),
            ("BASELINE (structural break OFF)", False),
        ]
        results = {label: {} for label, _ in variants}
        use_365 = bool_env("MPTRADE_REPLAY_USE_365", default=False)
        can_fork = "fork" in mp.get_all_start_methods()

        # Process one symbol at a time and share series + prewarmed trend tape across
        # forked variant workers via copy-on-write (never pickling multi-million-tick
        # series over IPC or holding all symbols in RAM simultaneously).
        for sym in SYMBOLS._SECTIONS:
            series = load_archive_series(sym, use_365=use_365)
            if use_365 and _series_span_sec(series) < _MIN_DENSE_SPAN_SEC:
                sparse = load_archive_series(sym, use_365=False)
                if _series_span_sec(sparse) > _series_span_sec(series):
                    series = sparse
            if not series:
                for label, _ in variants:
                    results[label][sym] = None
                continue

            sym_cfg = SYMBOLS[sym]
            sym_tasks = [
                (label, enabled, sym, sym_cfg["params"], sym_cfg["base"])
                for label, enabled in variants
            ]
            prewarm_trend_series(sym, series)
            _WORKER_SERIES.clear()
            _WORKER_SERIES[sym] = series
            max_w = _resolve_replay_worker_cap(use_365)
            n_workers = min(len(sym_tasks), max_w)
            try:
                if n_workers > 1 and can_fork:
                    ctx = mp.get_context("fork")
                    with ctx.Pool(processes=n_workers) as pool:
                        sym_outputs = pool.map(_run_variant_symbol, sym_tasks)
                elif n_workers > 1:
                    with mp.Pool(
                        processes=n_workers,
                        initializer=_init_worker,
                        initargs=({sym: series},),
                    ) as pool:
                        sym_outputs = pool.map(_run_variant_symbol, sym_tasks)
                else:
                    sym_outputs = [_run_variant_symbol(t) for t in sym_tasks]
            finally:
                _WORKER_SERIES.clear()
                _TREND_TAPE_CACHE.clear()
                del series
                gc.collect()

            for label, out_sym, res in sym_outputs:
                results[label][out_sym] = res

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

        print(f"\n\n{'=' * 60}\n           BACKTEST COMPARISON SUMMARY\n{'=' * 60}")
        compared_count = 0
        for sym in SYMBOLS._SECTIONS:
            prop = results[variants[0][0]].get(sym) or {}
            base = results[variants[1][0]].get(sym) or {}
            prop_net = safe_float(prop.get("net"), default=None) if isinstance(prop, dict) else None
            base_net = safe_float(base.get("net"), default=None) if isinstance(base, dict) else None
            if prop_net is not None and base_net is not None:
                compared_count += 1
                diff = prop_net - base_net
                print(
                    f"\n{sym}:\n"
                    f"  Proposed (Break ON) : {prop_net:.2f} USD ({prop.get('sells', 0)} sells)\n"
                    f"  Baseline (Break OFF): {base_net:.2f} USD ({base.get('sells', 0)} sells)\n"
                    f"  Difference          : {'+' if diff >= 0 else ''}{diff:.2f} USD ({'OUTPERFORMS' if diff > 0 else 'UNDERPERFORMS' if diff < 0 else 'EQUAL'})"
                )
            else:
                print(f"\n{sym}: Incomplete data.")
        if SYMBOLS._SECTIONS and compared_count == 0:
            print("[compare_sl_variants] ERROR: no symbols produced valid baseline/proposed comparisons", file=sys.stderr)
            sys.exit(1)
    finally:
        os.chdir(orig_cwd)

if __name__ == "__main__":
    main()
