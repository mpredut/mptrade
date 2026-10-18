#!/usr/bin/env python3
"""Run Tests 0 to 4 ablation across Trading 212 US equities."""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
T212_DIR = os.path.join(ROOT, "212trading")
if T212_DIR not in sys.path:
    sys.path.insert(0, T212_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from offline.research.hybrid_reentry.harness import (
    build_ablation_variants,
    format_ablation_result,
    load_ohlc_csv,
)

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


def _import_t212():
    _COLLIDING = ("strategy", "market_data", "notify", "ipo_notify", "replay")
    for name in _COLLIDING:
        sys.modules.pop(name, None)
    kraken_dir = os.path.join(ROOT, "kraken")
    while kraken_dir in sys.path:
        sys.path.remove(kraken_dir)
    while T212_DIR in sys.path:
        sys.path.remove(T212_DIR)
    sys.path.insert(0, T212_DIR)
    import strategy as _strat
    import replay as rp
    return _strat, rp


build_equity_variants = build_ablation_variants


def main():
    print("======================================================================")
    print("TRADING 212 EQUITIES HYBRID RE-ENTRY ABLATION SUITE (Tests 0 to 4)")
    print("======================================================================")
    _strat, rp = _import_t212()
    results = []
    
    for symbol, csv_path in DATASETS.items():
        if not os.path.isfile(csv_path):
            print(f"Skipping {symbol}: not found at {csv_path}")
            continue
        ohlc = load_ohlc_csv(csv_path)
        if not ohlc:
            print(f"Skipping {symbol}: empty OHLC data.")
            continue
        print(f"Evaluating {symbol} ({len(ohlc)} daily bars)...", flush=True)
        
        cfg = dict(BASE_CONFIG)
        cfg["YAHOO_SYMBOL"] = symbol
        base_params = _strat.StratParams.from_env(cfg)
        variants = build_equity_variants(base_params)
        
        warmup_n = min(30, len(ohlc) // 5)
        warmup_ohlc = ohlc[:warmup_n]
        run_ohlc = ohlc[warmup_n:]
        if not run_ohlc:
            print(f"Skipping {symbol}: insufficient bars after warmup.")
            continue
        
        for test_id, (variant_name, params) in variants.items():
            try:
                res = rp.run_replay(
                    ohlc=run_ohlc,
                    params=params,
                    bar_minutes=1440.0,  # 1 day
                    warmup_ohlc=warmup_ohlc,
                )
                budget = float(params.max_budget) if params.max_budget > 0 else 500.0
                row = format_ablation_result(symbol, test_id, variant_name, res, budget)
                results.append(row)
                print(
                    f"  {test_id} ({variant_name:<18}): Return {row['total_return_pct']:+6.2f}% "
                    f"| MaxDD {row['max_drawdown_pct']:5.2f}% | Exposure {row['exposure_pct']:5.2f}% "
                    f"| Cycles {row['cycles']}"
                )
            except Exception as exc:
                print(f"  {test_id} ({variant_name:<18}): ERROR {exc}")
                results.append({
                    "asset": symbol,
                    "test_id": test_id,
                    "variant": variant_name,
                    "error": str(exc),
                })

    out_json = os.path.join(OUT_DIR, "equities_ablation_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nEquities results saved to {out_json}")


if __name__ == "__main__":
    main()
