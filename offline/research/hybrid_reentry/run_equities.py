#!/usr/bin/env python3
"""Run Tests 0 to 4 ablation across Trading 212 US equities."""

import csv
import dataclasses
import json
import os
import sys
import time

import importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
T212_DIR = os.path.join(ROOT, "212trading")
sys.path.insert(0, T212_DIR)
sys.path.insert(0, ROOT)

_COLLIDING = ("strategy", "market_data", "notify", "ipo_notify", "replay")
for name in _COLLIDING:
    sys.modules.pop(name, None)

import strategy as _strat
import replay as rp
from offline.research.hybrid_reentry.harness import load_ohlc_csv

OUT_DIR = os.path.join(ROOT, "offline", "results", "hybrid_reentry")
os.makedirs(OUT_DIR, exist_ok=True)

# Datasets
DATASETS = {
    "NVDA": os.path.join(ROOT, "offline", "research", "t212_dataset", "nvda", "datasets", "NVDA_1d_9e8cbd3c6ff5.csv"),
    "RGNT": os.path.join(ROOT, "offline", "research", "t212_dataset", "rgnt", "datasets", "RGNT_1d_b67a2932ddd6.csv"),
}

BASE_CONFIG = {
    "STRAT_CURRENCY": "USD",
    "STRATEGY_MODE": "avg_tp",
    "STRAT_ENTRY_PCT": "15",
    "STRAT_DCA_PCT": "9.44",
    "STRAT_ENTRY_DISCOUNT_PCT": "0.4",
    "STRAT_DCA_DROP_PCT": "1.3",
    "STRAT_TAKEPROFIT_PCT": "8.0",
    "STRAT_MAX_DCA_BUYS": "auto",
    "STRAT_MAX_BUDGET": "500",
    "STRAT_FX_FEE_PCT": "0.15",
    "STRAT_STOP_LOSS_PCT": "25.0",
    "STRAT_CHECK_MINUTES": "5.0",
    "STRAT_ORDER_TTL_MIN": "10.0",
    "STRAT_REENTRY_DROP_PCT": "2.0",
    "STRAT_REENTRY_TOLERANCE_PCT": "0.05",
    "STRAT_LOSS_ALERT_STEP": "1",
    "STRAT_LADDER_MIN_FREE": "6",
    "STRAT_SL_REBUY_ENABLED": "true",
    "STRAT_SL_REBUY_BOUNCE_PCT": "1.2",
    "STRAT_DCA_TREND_GATE_PCT": "0",
    "STRAT_TRAIL_PCT": "0",
    "STRAT_TRAIL_MIN_PROFIT_PCT": "5",
    "STRAT_REENTRY_HYBRID_ENABLED": "false",
    "STRAT_REENTRY_PEAK_RELATIVE": "false",
    "STRAT_REENTRY_PULLBACK_PCT": "1.5",
    "STRAT_REENTRY_TTL_HOURS": "0.0",
}


def build_equity_variants(base_p: _strat.StratParams):
    R = dataclasses.replace
    return {
        "Test 0": ("Baseline (Control)", R(base_p, reentry_hybrid_enabled=False, reentry_peak_relative=False, reentry_pullback_pct=1.5, reentry_ttl_hours=0.0)),
        "Test 1": ("Peak-Relative", R(base_p, reentry_hybrid_enabled=True, reentry_peak_relative=True, reentry_pullback_pct=1.5, reentry_ttl_hours=0.0)),
        "Test 2": ("Trend Bypass", R(base_p, reentry_hybrid_enabled=True, reentry_peak_relative=False, reentry_pullback_pct=0.0, reentry_ttl_hours=0.0)),
        "Test 3": ("Hybrid", R(base_p, reentry_hybrid_enabled=True, reentry_peak_relative=False, reentry_pullback_pct=1.5, reentry_ttl_hours=0.0)),
        "Test 4": ("Hybrid + TTL", R(base_p, reentry_hybrid_enabled=True, reentry_peak_relative=False, reentry_pullback_pct=1.5, reentry_ttl_hours=336.0)),
    }


def main():
    print("======================================================================")
    print("TRADING 212 EQUITIES HYBRID RE-ENTRY ABLATION SUITE (Tests 0 to 4)")
    print("======================================================================")
    results = []
    
    for symbol, csv_path in DATASETS.items():
        if not os.path.isfile(csv_path):
            print(f"Skipping {symbol}: not found at {csv_path}")
            continue
        ohlc = load_ohlc_csv(csv_path)
        print(f"Evaluating {symbol} ({len(ohlc)} daily bars)...", flush=True)
        
        cfg = dict(BASE_CONFIG)
        cfg["YAHOO_SYMBOL"] = symbol
        base_params = _strat.StratParams.from_env(cfg)
        variants = build_equity_variants(base_params)
        
        warmup_n = min(30, len(ohlc) // 5)
        warmup_ohlc = ohlc[:warmup_n]
        run_ohlc = ohlc[warmup_n:]
        
        for test_id, (variant_name, params) in variants.items():
            res = rp.run_replay(
                ohlc=run_ohlc,
                params=params,
                bar_minutes=1440.0,  # 1 day
                warmup_ohlc=warmup_ohlc,
            )
            ret_pct = (res.get("total", 0.0) / params.max_budget) * 100.0
            max_dd_pct = float(res.get("max_drawdown_pct") or 0.0)
            exposure_pct = float(res.get("exposure_pct") or 0.0)
            results.append({
                "asset": symbol,
                "test_id": test_id,
                "variant": variant_name,
                "total_return_pct": round(ret_pct, 2),
                "net_profit_usd": round(res.get("total", 0.0), 2),
                "max_drawdown_pct": round(max_dd_pct, 2),
                "exposure_pct": round(exposure_pct, 2),
                "cycles": res.get("cycles", 0),
                "wins": res.get("wins", 0),
                "win_rate": round(float(res.get("win_rate") or 0.0) * 100.0, 1),
                "profit_factor": round(float(res.get("profit_factor") or 0.0), 2),
                "fills": res.get("fills", 0),
            })
            print(f"  {test_id} ({variant_name:<18}): Return {ret_pct:+6.2f}% | MaxDD {max_dd_pct:5.2f}% | Exposure {exposure_pct:5.2f}% | Cycles {res.get('cycles', 0)}")

    out_json = os.path.join(OUT_DIR, "equities_ablation_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nEquities results saved to {out_json}")


if __name__ == "__main__":
    main()
