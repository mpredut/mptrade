#!/usr/bin/env python3
"""Run Tests 0 to 4 ablation across 11 crypto assets and US equities."""

import csv
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "kraken"))

from strategies import spot_dca as strat
from offline.research.hybrid_reentry.harness import evaluate_asset, load_ohlc_csv

DATA_DIR = os.path.join(ROOT, "offline", "results", "kraken_continuous_grid", "data")
OUT_DIR = os.path.join(ROOT, "offline", "results", "hybrid_reentry")
os.makedirs(OUT_DIR, exist_ok=True)

# 11 benchmark crypto assets
ASSETS = ["HYPE", "BTC", "ETH", "SOL", "TAO", "ADA", "AVAX", "BNB", "DOGE", "LINK", "XRP"]

# Base strategy parameters: ov650_t8_brake_gateoff (the approved candidate base)
BASE_PARAMS = strat.StratParams(
    currency="USD",
    entry_amount=100.0,
    entry_discount_pct=0.8,
    dca_amount=50.0,
    dca_drop_pct=1.25,
    check_minutes=2.0,
    takeprofit_pct=5.0,
    max_budget=3900.0,
    max_dca_buys=10,
    enable_takeprofit=True,
    order_ttl_min=10.0,
    stop_loss_pct=25.0,
    adopt_cost=0.0,
    adopt_qty=0.0,
    reentry_drop_pct=2.2,
    reentry_tolerance_pct=0.05,
    reentry_adaptive=False,
    reentry_sl_bounce_pct=1.5,
    tp_tranches=[],
    tp_trend_hold=True,
    tp_regime_gate=False,
    tp_trend_min_pct=3.0,
    tp_trail_pct=4.0,
    tp_trail_profit_floor_pct=0.0,
    trend_overlay=True,
    trend_sma_n=20,
    trend_interval=240,
    regime_min_samples=20,
    trend_topup=650.0,
    trend_trail_pct=8.0,
    trend_exit_break=False,
    tp_trail_adaptive=False,
    tp_trail_k=1.5,
    tp_trail_min=2.0,
    tp_trail_max=8.0,
    tp_trail_vol_interval=240,
    dca_trend_brake=True,
    dca_brake_min_pct=1.5,
    dca_spacing_growth_pct=0.0,
    dca_vol_scale_k=0.0,
    dca_vol_ref=2.0,
    dca_vol_interval=240,
    reentry_hybrid_enabled=False,
    reentry_peak_relative=False,
    reentry_pullback_pct=1.5,
    reentry_ttl_hours=0.0,
)


def main():
    print("======================================================================")
    print("HYBRID RE-ENTRY ABLATION EXPERIMENT SUITE (Tests 0 to 4)")
    print("======================================================================")
    all_results = []
    
    for asset in ASSETS:
        csv_file = os.path.join(DATA_DIR, f"{asset}_240m.csv")
        if not os.path.isfile(csv_file):
            print(f"Skipping {asset}: {csv_file} not found.")
            continue
        ohlc = load_ohlc_csv(csv_file)
        print(f"Running ablation on {asset} ({len(ohlc)} 4h bars)...", flush=True)
        t0 = time.time()
        res = evaluate_asset(
            asset_name=asset,
            ohlc=ohlc,
            base_params=BASE_PARAMS,
            fee_pct=0.26,
            bar_minutes=240.0,
            initial_cash=3900.0,
        )
        elapsed = time.time() - t0
        print(f"  Completed {asset} in {elapsed:.1f}s.")
        all_results.extend(res)

    # Save raw results
    out_json = os.path.join(OUT_DIR, "ablation_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw results saved to {out_json}")

    # Aggregate by test variant
    variants = ["Test 0", "Test 1", "Test 2", "Test 3", "Test 4"]
    variant_names = {
        "Test 0": "Baseline (Control)",
        "Test 1": "Peak-Relative",
        "Test 2": "Trend Bypass",
        "Test 3": "Hybrid",
        "Test 4": "Hybrid + TTL",
    }
    
    summary = {}
    for var in variants:
        runs = [r for r in all_results if r.get("test_id") == var and "error" not in r]
        if not runs:
            continue
        rets = [r["total_return_pct"] for r in runs]
        dds = [r["max_drawdown_pct"] for r in runs]
        exposures = [r["exposure_pct"] for r in runs]
        cycles = [r["cycles"] for r in runs]
        wins = [r["wins"] for r in runs]
        
        rets_sorted = sorted(rets)
        n = len(rets_sorted)
        med_ret = rets_sorted[n // 2] if n % 2 != 0 else (rets_sorted[n // 2 - 1] + rets_sorted[n // 2]) / 2.0
        
        dds_sorted = sorted(dds)
        med_dd = dds_sorted[n // 2] if n % 2 != 0 else (dds_sorted[n // 2 - 1] + dds_sorted[n // 2]) / 2.0
        
        pos_count = sum(1 for ret in rets if ret > 0)
        
        summary[var] = {
            "name": variant_names[var],
            "runs": n,
            "positive_assets": f"{pos_count}/{n}",
            "median_return_pct": round(med_ret, 2),
            "worst_return_pct": round(min(rets), 2),
            "best_return_pct": round(max(rets), 2),
            "median_drawdown_pct": round(med_dd, 2),
            "max_drawdown_pct": round(max(dds), 2),
            "mean_exposure_pct": round(sum(exposures) / n, 2),
            "total_cycles": sum(cycles),
            "total_wins": sum(wins),
        }

    # Print summary table
    print("\n" + "=" * 105)
    print(f"{'Test ID':<8} | {'Variant Name':<20} | {'Pos Assets':<10} | {'Med Ret %':<10} | {'Worst Ret %':<11} | {'Med DD %':<9} | {'Max DD %':<9} | {'Exposure %':<10} | {'Cycles':<6}")
    print("-" * 105)
    for var in variants:
        if var in summary:
            s = summary[var]
            print(f"{var:<8} | {s['name']:<20} | {s['positive_assets']:<10} | {s['median_return_pct']:+9.2f}% | {s['worst_return_pct']:+10.2f}% | {s['median_drawdown_pct']:8.2f}% | {s['max_drawdown_pct']:8.2f}% | {s['mean_exposure_pct']:9.2f}% | {s['total_cycles']:<6}")
    print("=" * 105)

    summary_json = os.path.join(OUT_DIR, "ablation_summary.json")
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
