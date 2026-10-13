#!/usr/bin/env python3
"""Compare MHDSL Structural Break PROPOSED vs BASELINE on historical replay."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import monitortrades as mt
from offline.research.monitortrades_backtest.run_replay_backtest import SYMBOLS, run_symbol

def main():
    results = {}
    variants = [
        ("PROPOSED (structural break ON)", True),
        ("BASELINE (structural break OFF)", False),
    ]

    for label, enabled in variants:
        print(f"\n=======================================================")
        print(f"  RUNNING {label}")
        print(f"=======================================================")
        mt.MT_SL_STRUCTURAL_BREAK_ENABLED = enabled
        results[label] = {}

        for sym in ["BTCUSDC", "TAOUSDC"]:
            cfg = SYMBOLS[sym]
            print(f"Starting {sym}...")
            res = run_symbol(sym, cfg["params"], cfg["base"], quiet=True)
            results[label][sym] = res
            print(f"[{sym}] Net PnL: {res.get('net'):.2f} USD | Sells: {res.get('sells')} | Buys: {res.get('buys')}")

    print("\n\n" + "=" * 60)
    print("           BACKTEST COMPARISON SUMMARY")
    print("=" * 60)
    for sym in ["BTCUSDC", "TAOUSDC"]:
        prop = results[variants[0][0]][sym]
        base = results[variants[1][0]][sym]
        diff = prop.get("net", 0.0) - base.get("net", 0.0)
        print(f"\n{sym}:")
        print(f"  Proposed (Break ON) : {prop.get('net'):.2f} USD ({prop.get('sells')} sells)")
        print(f"  Baseline (Break OFF): {base.get('net'):.2f} USD ({base.get('sells')} sells)")
        print(f"  Difference          : {'+' if diff >= 0 else ''}{diff:.2f} USD ({'OUTPERFORMS' if diff > 0 else 'UNDERPERFORMS' if diff < 0 else 'EQUAL'})")

if __name__ == "__main__":
    main()
