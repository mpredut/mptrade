#!/usr/bin/env python3
"""Compare MHDSL Structural Break PROPOSED vs BASELINE on historical replay."""
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


def main():
    results = {}
    variants = [
        ("PROPOSED (structural break ON)", True),
        ("BASELINE (structural break OFF)", False),
    ]
    use_365 = _parse_bool_flag(os.environ.get("MPTRADE_REPLAY_USE_365"), False)
    series_map = {sym: load_archive_series(sym, use_365=use_365) for sym in SYMBOLS._SECTIONS}
    for sym, series in series_map.items():
        if series:
            prewarm_trend_series(sym, series)
    orig_break_enabled = getattr(mt, "MT_SL_STRUCTURAL_BREAK_ENABLED", True)
    try:
        for label, enabled in variants:
            print("\n=======================================================")
            print(f"  RUNNING {label}")
            print("=======================================================")
            mt.MT_SL_STRUCTURAL_BREAK_ENABLED = enabled
            results[label] = {}

            for sym in SYMBOLS._SECTIONS:
                cfg = SYMBOLS[sym]
                print(f"Starting {sym}...")
                res = run_symbol(sym, cfg["params"], cfg["base"], quiet=True, series=series_map.get(sym))
                results[label][sym] = res
                if res:
                    net_val = safe_float(res.get("net"), 0.0)
                    print(f"[{sym}] Net PnL: {net_val:.2f} USD | Sells: {res.get('sells') or 0} | Buys: {res.get('buys') or 0}")
                else:
                    print(f"[{sym}] No results or no data.")
    finally:
        mt.MT_SL_STRUCTURAL_BREAK_ENABLED = orig_break_enabled

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
