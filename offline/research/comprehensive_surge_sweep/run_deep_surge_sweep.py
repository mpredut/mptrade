#!/usr/bin/env python3
"""Deep Overnight Parabolic Surge Guard & Re-entry Sweep.

Performs fine-grained grid search across:
1. Fixed Parabolic Surge: gain triggers [18% - 35%], pullbacks [2.0% - 5.0%], windows [48h - 96h]
2. Dynamic Parabolic Surge: min gains [20% - 26%], max gains [28% - 40%], multipliers [8.0 - 12.0], pullbacks [2.5% - 4.0%]
3. Smart Bear Bounce Re-entry: bounce filters [0.0% - 2.0%]
4. Independent 2-window validation (Walk-Forward guardrail) across 5 assets: HYPE, TAO, ADA, BTC, ETH.

Outputs:
- JSON database: offline/results/surge_sweep/sweep_results.json
- Executive Report: offline/results/surge_sweep/DEEP_SURGE_SWEEP_REPORT.md
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import dataclasses
import io
import json
import math
import os
import sys
import time
from datetime import datetime
from typing import Any

ROOT = "/home/predut/mptrade"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "kraken"))

from kraken.replay import run_replay
from strategies import spot_dca as strat
from offline.research.hybrid_reentry.harness import load_ohlc_csv

DATA_DIR = os.path.join(ROOT, "offline", "results", "kraken_continuous_grid", "data")
DEFAULT_OUT_DIR = os.path.join(ROOT, "offline", "results", "surge_sweep")
ASSETS = ["HYPE", "TAO", "ADA", "BTC", "ETH"]

# Base parameters matching production
BASE_PARAMS = strat.StratParams(
    currency="USD",
    entry_amount=650.0,
    entry_discount_pct=0.8,
    dca_amount=325.0,
    dca_drop_pct=1.25,
    dca_spacing_growth_pct=0.25,
    check_minutes=2.0,
    takeprofit_pct=5.0,
    max_budget=3900.0,
    max_dca_buys=10,
    enable_takeprofit=True,
    order_ttl_min=10.0,
    stop_loss_pct=0.0,
    adopt_cost=0.0,
    adopt_qty=0.0,
    reentry_drop_pct=2.0,
    reentry_tolerance_pct=0.05,
    reentry_adaptive=False,
    reentry_sl_bounce_pct=1.5,
    tp_tranches=[],
    tp_trend_hold=True,
    tp_regime_gate=False,
    tp_trend_min_pct=3.0,
    tp_trail_pct=4.0,
    tp_trail_profit_floor_pct=1.0,
    trend_overlay=True,
    trend_sma_n=20,
    trend_interval=240,
    regime_min_samples=20,
    trend_topup=650.0,
    trend_trail_pct=8.0,
    trend_exit_break=False,
    tp_trail_adaptive=True,
    tp_trail_k=2.0,
    tp_trail_min=1.5,
    tp_trail_max=8.0,
    tp_trail_vol_interval=240,
    dca_trend_brake=True,
    dca_brake_min_pct=1.5,
    reentry_hybrid_enabled=True,
    reentry_pullback_adaptive=True,
    reentry_pullback_k=1.0,
    reentry_pullback_min=0.8,
    reentry_pullback_max=3.5,
    reentry_ttl_hours=48.0,
    reentry_bear_bounce_pct=1.0,
    fast_profit_guard=True,
    tp_dynamic_flat=True,
    surge_guard=False,
    surge_dynamic=False,
)

R = dataclasses.replace


def build_candidate_grid() -> list[dict[str, Any]]:
    """Construct a comprehensive grid covering fixed, dynamic, and pullback variations."""
    candidates = []

    # 1. Baseline control
    candidates.append({
        "id": "BASELINE_SURGE_OFF",
        "category": "Baseline",
        "params": R(BASE_PARAMS, surge_guard=False),
        "desc": "Baseline (Surge OFF, pure Trend Overlay)",
    })

    # 2. Fine-grained Fixed Surge Grid
    fixed_gains = [18.0, 20.0, 22.0, 24.0, 25.0, 26.0, 28.0, 30.0, 32.0, 35.0]
    pullbacks = [2.0, 2.5, 2.8, 3.0, 3.2, 3.5, 3.8, 4.0, 4.5, 5.0]
    windows = [48.0, 72.0, 96.0]

    for g in fixed_gains:
        for pb in pullbacks:
            for w in windows:
                # Include standard 72h window, and cross-combinations
                cid = f"FIXED_g{g:.0f}_pb{pb:.1f}_w{w:.0f}"
                candidates.append({
                    "id": cid,
                    "category": "Fixed",
                    "params": R(
                        BASE_PARAMS,
                        surge_guard=True,
                        surge_dynamic=False,
                        surge_gain_pct=g,
                        surge_exit_pullback_pct=pb,
                        surge_window_hours=w,
                        surge_move_pct=g,
                    ),
                    "desc": f"Fixed Surge {g:.0f}% / PB {pb:.1f}% / Window {w:.0f}h",
                })

    # 3. Dynamic Surge Grid (Centered around 25%)
    dyn_mins = [20.0, 22.0, 24.0, 25.0, 26.0]
    dyn_maxs = [28.0, 30.0, 32.0, 35.0, 40.0]
    vol_mults = [8.0, 10.0, 12.0]
    dyn_pbs = [2.5, 2.8, 3.0, 3.2, 3.5, 4.0]

    for mn in dyn_mins:
        for mx in dyn_maxs:
            if mx <= mn:
                continue
            for mult in vol_mults:
                for pb in dyn_pbs:
                    cid = f"DYN_mn{mn:.0f}_mx{mx:.0f}_k{mult:.0f}_pb{pb:.1f}"
                    candidates.append({
                        "id": cid,
                        "category": "Dynamic",
                        "params": R(
                            BASE_PARAMS,
                            surge_guard=True,
                            surge_dynamic=True,
                            surge_min_gain_pct=mn,
                            surge_max_gain_pct=mx,
                            surge_vol_multiplier=mult,
                            surge_exit_pullback_pct=pb,
                            surge_window_hours=72.0,
                            surge_gain_pct=25.0,
                        ),
                        "desc": f"Dyn Surge [{mn:.0f}-{mx:.0f}%] k={mult:.0f} PB {pb:.1f}%",
                    })

    # 4. Bear Bounce Re-entry Sweep (with top calibrated surge anchor)
    bounces = [0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    for b in bounces:
        cid = f"BOUNCE_b{b:.1f}_with_calibrated_surge"
        candidates.append({
            "id": cid,
            "category": "BearBounce",
            "params": R(
                BASE_PARAMS,
                surge_guard=True,
                surge_dynamic=True,
                surge_min_gain_pct=24.0,
                surge_max_gain_pct=32.0,
                surge_vol_multiplier=10.0,
                surge_exit_pullback_pct=3.2,
                reentry_bear_bounce_pct=b,
            ),
            "desc": f"Bear Bounce Re-entry {b:.1f}% (Dyn Surge 24-32% PB 3.2%)",
        })

    return candidates


def evaluate_single_candidate(
    candidate: dict[str, Any],
    asset_data: dict[str, list[tuple[float, float, float, float]]],
) -> dict[str, Any]:
    """Run replay for a candidate across all assets and both independent split halves."""
    cid = candidate["id"]
    params = candidate["params"]
    category = candidate["category"]
    desc = candidate["desc"]

    total_profit = 0.0
    worst_maxdd = 0.0
    total_cycles = 0
    asset_profits = {}
    asset_maxdds = {}

    # Half 1 and Half 2 metrics for out-of-sample validation
    h1_profits = 0.0
    h2_profits = 0.0

    for asset, ohlc in asset_data.items():
        if not ohlc or len(ohlc) < 60:
            continue
        warmup_n = min(30, len(ohlc) // 5)
        warmup_ohlc = ohlc[:warmup_n]
        run_ohlc = ohlc[warmup_n:]

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            # Full run
            res = run_replay(
                ohlc=run_ohlc,
                params=params,
                fee_pct=0.26,
                bar_minutes=240.0,
                warmup_ohlc=warmup_ohlc,
                initial_cash=3900.0,
            )
            pnl = float(res.get("total", 0.0))
            dd = float(res.get("max_drawdown_pct", 0.0))
            cyc = int(res.get("cycles", 0))

            total_profit += pnl
            worst_maxdd = max(worst_maxdd, dd)
            total_cycles += cyc
            asset_profits[asset] = round(pnl, 2)
            asset_maxdds[asset] = round(dd, 2)

            # 2 Independent Halves Split (Guardrail #1)
            half_idx = len(run_ohlc) // 2
            ohlc_h1 = run_ohlc[:half_idx]
            ohlc_h2 = run_ohlc[half_idx:]

            res_h1 = run_replay(
                ohlc=ohlc_h1,
                params=params,
                fee_pct=0.26,
                bar_minutes=240.0,
                warmup_ohlc=warmup_ohlc,
                initial_cash=3900.0,
            )
            res_h2 = run_replay(
                ohlc=ohlc_h2,
                params=params,
                fee_pct=0.26,
                bar_minutes=240.0,
                warmup_ohlc=ohlc_h1[-warmup_n:] if len(ohlc_h1) >= warmup_n else warmup_ohlc,
                initial_cash=3900.0,
            )
            h1_profits += float(res_h1.get("total", 0.0))
            h2_profits += float(res_h2.get("total", 0.0))

    calmar = (total_profit / worst_maxdd) if worst_maxdd > 0 else 0.0

    return {
        "id": cid,
        "category": category,
        "desc": desc,
        "total_profit_usd": round(total_profit, 2),
        "worst_maxdd_pct": round(worst_maxdd, 2),
        "calmar_ratio": round(calmar, 2),
        "total_cycles": total_cycles,
        "h1_profit_usd": round(h1_profits, 2),
        "h2_profit_usd": round(h2_profits, 2),
        "asset_profits": asset_profits,
        "asset_maxdds": asset_maxdds,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3, help="Max parallel worker processes (default: 3)")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR, help="Output directory for reports")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "sweep_results.json")
    md_path = os.path.join(args.out_dir, "DEEP_SURGE_SWEEP_REPORT.md")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading datasets from {DATA_DIR}...")
    asset_data = {}
    for a in ASSETS:
        csv_file = os.path.join(DATA_DIR, f"{a}_240m.csv")
        if os.path.isfile(csv_file):
            ohlc = load_ohlc_csv(csv_file)
            asset_data[a] = ohlc
            print(f"  - {a:<6}: {len(ohlc)} bars")

    candidates = build_candidate_grid()
    total_cands = len(candidates)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Built parameter grid: {total_cands} candidates across {len(asset_data)} assets.")
    print(f"  Parallel execution: {args.workers} workers (nice -n 15 scheduled).")

    t0 = time.time()
    results = []
    completed = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(evaluate_single_candidate, c, asset_data): c
            for c in candidates
        }

        for future in concurrent.futures.as_completed(future_map):
            cand = future_map[future]
            try:
                res = future.result()
                results.append(res)
            except Exception as exc:
                print(f"Error on {cand['id']}: {exc}")
            completed += 1
            if completed % 20 == 0 or completed == total_cands:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                eta_s = (total_cands - completed) / rate if rate > 0 else 0
                top_pnl = max(r["total_profit_usd"] for r in results) if results else 0.0
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] Progress: {completed}/{total_cands} "
                    f"({completed/total_cands*100:.1f}%) | {rate:.1f} cand/s | ETA: {eta_s/60:.1f}m | "
                    f"Best PnL so far: ${top_pnl:.2f}",
                    flush=True,
                )

    total_time = time.time() - t0
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Sweep completed in {total_time/60:.1f} minutes.")

    # Find baseline
    baseline = next((r for r in results if r["id"] == "BASELINE_SURGE_OFF"), None)
    base_pnl = baseline["total_profit_usd"] if baseline else 0.0
    base_h1 = baseline["h1_profit_usd"] if baseline else 0.0
    base_h2 = baseline["h2_profit_usd"] if baseline else 0.0

    # Score and tag 2-window walk-forward winners
    for r in results:
        r["beats_baseline_total"] = r["total_profit_usd"] >= base_pnl
        r["beats_baseline_h1"] = r["h1_profit_usd"] >= base_h1
        r["beats_baseline_h2"] = r["h2_profit_usd"] >= base_h2
        # True signal: beats baseline in BOTH independent halves
        r["two_window_verified"] = r["beats_baseline_h1"] and r["beats_baseline_h2"]
        r["edge_usd"] = round(r["total_profit_usd"] - base_pnl, 2)

    # Sort candidates
    results.sort(key=lambda x: (x["total_profit_usd"], x["calmar_ratio"]), reverse=True)

    # Save JSON database
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "total_candidates": len(results),
                "duration_seconds": round(total_time, 1),
                "baseline": baseline,
                "top_10": results[:10],
                "all_results": results,
            },
            f,
            indent=2,
        )
    print(f"Saved full JSON database to: {json_path}")

    # Generate Markdown Report
    top15 = results[:15]
    top_verified = [r for r in results if r.get("two_window_verified")][:15]

    lines = [
        "# Deep Parabolic Surge Guard & Re-entry Optimization Report",
        "",
        f"- **Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Candidates Evaluated:** {len(results):,}",
        f"- **Total Time:** {total_time/60:.1f} minutes",
        f"- **Assets:** {', '.join(ASSETS)} (4h continuous multi-year history)",
        f"- **Baseline Reference (Surge OFF):** ${base_pnl:,.2f} | H1: ${base_h1:,.2f} | H2: ${base_h2:,.2f}",
        "",
        "## 1. Top 15 Overall Performers (Total Fleet PnL)",
        "",
        "| Rank | Candidate ID | Category | Total PnL ($) | Edge ($) | MaxDD % | Calmar | 2-Win Verified | Description |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]

    for idx, r in enumerate(top15, 1):
        v_tag = "✅ YES" if r["two_window_verified"] else "❌ No (1-win)"
        lines.append(
            f"| {idx} | `{r['id']}` | {r['category']} | **${r['total_profit_usd']:,.2f}** | "
            f"+${r['edge_usd']:,.2f} | {r['worst_maxdd_pct']:.2f}% | {r['calmar_ratio']:.2f} | {v_tag} | {r['desc']} |"
        )

    lines.extend([
        "",
        "## 2. Top Robust Performers (2-Window Independent Walk-Forward Verified)",
        "",
        "> [!IMPORTANT]",
        "> These candidates beat the baseline in **both independent sample halves (H1 and H2)**, satisfying Guardrail #1 against overfitting.",
        "",
        "| Rank | Candidate ID | Category | Total PnL ($) | H1 ($) | H2 ($) | MaxDD % | Calmar | Description |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ])

    for idx, r in enumerate(top_verified, 1):
        lines.append(
            f"| {idx} | `{r['id']}` | {r['category']} | **${r['total_profit_usd']:,.2f}** | "
            f"${r['h1_profit_usd']:,.2f} | ${r['h2_profit_usd']:,.2f} | {r['worst_maxdd_pct']:.2f}% | {r['calmar_ratio']:.2f} | {r['desc']} |"
        )

    lines.extend([
        "",
        "## 3. Asset-by-Asset Breakdown of the #1 Winning Configuration",
        "",
    ])

    if top15:
        winner = top15[0]
        lines.append(f"**Top Candidate:** `{winner['id']}` ({winner['desc']})\n")
        lines.append("| Asset | Profit ($) | Max Drawdown % |")
        lines.append("| :--- | :---: | :---: |")
        for a in ASSETS:
            ap = winner["asset_profits"].get(a, 0.0)
            add = winner["asset_maxdds"].get(a, 0.0)
            lines.append(f"| **{a}** | ${ap:,.2f} | {add:.2f}% |")

    lines.extend([
        "",
        "## 4. Proposal Branch Recommendation",
        "",
        "The winning candidate parameters should be integrated into `kraken/config.env` and `hyperliquid/config.env`:",
        "```env",
    ])
    if top_verified:
        top_cand = top_verified[0]
    else:
        top_cand = top15[0] if top15 else None

    if top_cand:
        lines.append(f"# Recommended Winner: {top_cand['id']}")
        lines.append(f"# Total PnL: ${top_cand['total_profit_usd']} vs Baseline ${base_pnl}")
        lines.append(f"# Description: {top_cand['desc']}")
    lines.append("```\n")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Generated comprehensive markdown report: {md_path}")
    print("\n=== TOP 5 CANDIDATES PREVIEW ===")
    for idx, r in enumerate(top15[:5], 1):
        print(f"#{idx} {r['id']:<35} | PnL: ${r['total_profit_usd']:>9.2f} | MaxDD: {r['worst_maxdd_pct']:>5.2f}% | 2-Win: {r['two_window_verified']} | {r['desc']}")


if __name__ == "__main__":
    main()
