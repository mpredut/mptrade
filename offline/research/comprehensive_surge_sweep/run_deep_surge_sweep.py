#!/usr/bin/env python3
"""Deep Overnight Parabolic Surge Guard & Re-entry Sweep.

Performs fine-grained grid search across:
1. Fixed Parabolic Surge: gain triggers [18% - 35%], pullbacks [2.0% - 5.0%], windows [48h - 96h]
2. Dynamic Parabolic Surge: min gains [18% - 26%], max gains [26% - 40%], multipliers [8.0 - 12.0], pullbacks [2.5% - 4.0%]
3. Smart Bear Bounce Re-entry: bounce filters [0.0% - 2.0%]
4. Two-half sensitivity comparison (not held-out walk-forward validation) across 5 assets: HYPE, TAO, ADA, BTC, ETH.

Outputs:
- JSON database: offline/results/surge_sweep/sweep_results.json
- Executive Report: offline/results/surge_sweep/DEEP_SURGE_SWEEP_REPORT.md
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import dataclasses
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "kraken"))

from botcore import parse_dotenv
from kraken.replay import run_replay
from strategies import spot_dca as strat
from offline.research.hybrid_reentry.harness import load_ohlc_csv

DATA_DIR = os.path.join(ROOT, "offline", "results", "kraken_continuous_grid", "data")
DEFAULT_OUT_DIR = os.path.join(ROOT, "offline", "results", "surge_sweep")
ASSETS = ["HYPE", "TAO", "ADA", "BTC", "ETH"]

DEFAULT_CONFIG = Path(ROOT) / "kraken" / "config.env"


def load_base_params(config_path: str | Path) -> strat.StratParams:
    """Read one explicit profile without inheriting shell variables or secrets."""
    path = Path(config_path).resolve(strict=True)
    return strat.StratParams.from_env(parse_dotenv(str(path)))

R = dataclasses.replace


def build_candidate_grid(base_params: strat.StratParams) -> list[dict[str, Any]]:
    """Construct a comprehensive grid covering fixed, dynamic, and pullback variations."""
    candidates = []

    # 1. Baseline control
    candidates.append({
        "id": "BASELINE_SURGE_OFF",
        "category": "Baseline",
        "params": R(base_params, surge_guard=False, surge_dynamic=False),
        "desc": "Selected profile with Surge OFF",
    })

    candidates.append({
        "id": "CURRENT_CONFIG",
        "category": "Current",
        "params": base_params,
        "desc": "Unmodified selected versioned profile",
    })

    # 2. Fine-grained Fixed Surge Grid
    fixed_gains = [18.0, 20.0, 21.0, 22.0, 24.0, 25.0, 26.0, 28.0, 30.0, 32.0, 35.0]
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
                        base_params,
                        surge_guard=True,
                        surge_dynamic=False,
                        surge_gain_pct=g,
                        surge_exit_pullback_pct=pb,
                        surge_window_hours=w,
                        surge_move_pct=g,
                    ),
                    "desc": f"Fixed Surge {g:.0f}% / PB {pb:.1f}% / Window {w:.0f}h",
                })

    # 3. Dynamic Surge Grid
    dyn_mins = [18.0, 20.0, 22.0, 24.0, 25.0, 26.0]
    dyn_maxs = [26.0, 28.0, 30.0, 32.0, 35.0, 40.0]
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
                            base_params,
                            surge_guard=True,
                            surge_dynamic=True,
                            surge_min_gain_pct=mn,
                            surge_max_gain_pct=mx,
                            surge_vol_multiplier=mult,
                            surge_exit_pullback_pct=pb,
                            surge_window_hours=72.0,
                            surge_gain_pct=mn,
                            surge_move_pct=mn,
                        ),
                        "desc": f"Dyn Surge [{mn:.0f}-{mx:.0f}%] k={mult:.0f} PB {pb:.1f}%",
                    })

    # 4. Bear Bounce Re-entry Sweep (with calibrated fixed 18% surge anchor)
    bounces = [0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    for b in bounces:
        cid = f"BOUNCE_b{b:.1f}_with_calibrated_surge"
        candidates.append({
            "id": cid,
            "category": "BearBounce",
            "params": R(
                base_params,
                surge_guard=True,
                surge_dynamic=False,
                surge_gain_pct=18.0,
                surge_move_pct=18.0,
                surge_exit_pullback_pct=3.5,
                surge_window_hours=72.0,
                reentry_bear_bounce_pct=b,
            ),
            "desc": f"Bear Bounce Re-entry {b:.1f}% (Fixed Surge 18% PB 3.5% w72)",
        })

    return candidates


def evaluate_single_candidate(
    candidate: dict[str, Any],
    asset_data: dict[str, list[tuple[float, float, float, float]]],
    initial_cash: float,
    fee_pct: float,
) -> dict[str, Any]:
    """Run replay for a candidate across all assets and both chronological halves."""
    cid = candidate["id"]
    params = candidate["params"]
    category = candidate["category"]
    desc = candidate["desc"]

    total_profit = 0.0
    worst_maxdd = 0.0
    total_cycles = 0
    asset_profits = {}
    asset_maxdds = {}

    # The same candidate is evaluated on both halves; neither is held out.
    h1_profits = 0.0
    h2_profits = 0.0

    for asset, ohlc in asset_data.items():
        if not ohlc or len(ohlc) < 60:
            continue
        warmup_n = min(30, len(ohlc) // 5)
        warmup_ohlc = ohlc[:warmup_n]
        run_ohlc = ohlc[warmup_n:]

        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
            # Full run
            res = run_replay(
                ohlc=run_ohlc,
                params=params,
                fee_pct=fee_pct,
                bar_minutes=240.0,
                warmup_ohlc=warmup_ohlc,
                initial_cash=initial_cash,
            )
            pnl = float(res.get("total", 0.0))
            dd = float(res.get("max_drawdown_pct", 0.0))
            cyc = int(res.get("cycles", 0))

            total_profit += pnl
            worst_maxdd = max(worst_maxdd, dd)
            total_cycles += cyc
            asset_profits[asset] = round(pnl, 2)
            asset_maxdds[asset] = round(dd, 2)

            # Two chronological halves, each reset to the same cash balance.
            half_idx = len(run_ohlc) // 2
            ohlc_h1 = run_ohlc[:half_idx]
            ohlc_h2 = run_ohlc[half_idx:]

            res_h1 = run_replay(
                ohlc=ohlc_h1,
                params=params,
                fee_pct=fee_pct,
                bar_minutes=240.0,
                warmup_ohlc=warmup_ohlc,
                initial_cash=initial_cash,
            )
            res_h2 = run_replay(
                ohlc=ohlc_h2,
                params=params,
                fee_pct=fee_pct,
                bar_minutes=240.0,
                warmup_ohlc=ohlc_h1[-warmup_n:] if len(ohlc_h1) >= warmup_n else warmup_ohlc,
                initial_cash=initial_cash,
            )
            h1_profits += float(res_h1.get("total", 0.0))
            h2_profits += float(res_h2.get("total", 0.0))

    pnl_per_drawdown_point = (total_profit / worst_maxdd) if worst_maxdd > 0 else 0.0

    return {
        "id": cid,
        "category": category,
        "desc": desc,
        "strategy_params": dataclasses.asdict(params),
        "total_profit_usd": round(total_profit, 2),
        "worst_maxdd_pct": round(worst_maxdd, 2),
        "pnl_per_drawdown_point": round(pnl_per_drawdown_point, 2),
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
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Versioned venue profile; shell variables and .env are ignored")
    parser.add_argument("--data-dir", type=Path, default=Path(DATA_DIR))
    parser.add_argument("--initial-cash", type=float,
                        help="Cash per independent asset replay; default: configured allocation")
    parser.add_argument("--fee-pct", type=float, required=True,
                        help="Explicit per-leg replay fee percentage (not fetched from the venue)")
    args = parser.parse_args()
    base_params = load_base_params(args.config)
    initial_cash = (base_params.effective_max_budget()
                    if args.initial_cash is None else args.initial_cash)
    if not math.isfinite(initial_cash) or initial_cash <= 0:
        parser.error("initial cash must be finite and positive")
    if not math.isfinite(args.fee_pct) or args.fee_pct < 0:
        parser.error("fee percentage must be finite and non-negative")
    if args.workers < 1:
        parser.error("workers must be positive")
    print("Research only: 4h OHLC cannot validate 5-minute guards; "
          "two-half comparison is not held-out walk-forward evidence.")


    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "sweep_results.json")
    md_path = os.path.join(args.out_dir, "DEEP_SURGE_SWEEP_REPORT.md")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading datasets from {args.data_dir}...")
    asset_data = {}
    for a in ASSETS:
        csv_file = os.path.join(args.data_dir, f"{a}_240m.csv")
        if os.path.isfile(csv_file):
            ohlc = load_ohlc_csv(csv_file)
            asset_data[a] = ohlc
            print(f"  - {a:<6}: {len(ohlc)} bars")

    missing = [asset for asset in ASSETS if len(asset_data.get(asset, [])) < 60]
    if missing:
        parser.error(f"Missing or insufficient datasets: {', '.join(missing)}")
    candidates = build_candidate_grid(base_params)
    total_cands = len(candidates)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Built parameter grid: {total_cands} candidates across {len(asset_data)} assets.")
    print(f"  Parallel execution: {args.workers} workers.")

    t0 = time.time()
    results = []
    completed = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(evaluate_single_candidate, c, asset_data, initial_cash, args.fee_pct): c
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
    if baseline is None or len(results) != total_cands:
        raise RuntimeError("Incomplete sweep; refusing to rank results against a missing or partial control")
    base_pnl = baseline["total_profit_usd"]
    base_h1 = baseline["h1_profit_usd"]
    base_h2 = baseline["h2_profit_usd"]

    # Descriptive split comparison, not out-of-sample selection.
    for r in results:
        r["beats_baseline_total"] = r["total_profit_usd"] >= base_pnl
        r["beats_baseline_h1"] = r["h1_profit_usd"] >= base_h1
        r["beats_baseline_h2"] = r["h2_profit_usd"] >= base_h2
        # Both halves participated in comparison; do not interpret this as validation.
        r["beats_baseline_both_halves"] = r["beats_baseline_h1"] and r["beats_baseline_h2"]
        r["edge_usd"] = round(r["total_profit_usd"] - base_pnl, 2)

    # Sort candidates
    results.sort(key=lambda x: (x["total_profit_usd"], x["pnl_per_drawdown_point"]), reverse=True)

    # Save JSON database
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": 2,
                "timestamp": datetime.now().isoformat(),
                "config_path": str(args.config.resolve()),
                "strategy_params": dataclasses.asdict(base_params),
                "initial_cash_per_asset": initial_cash,
                "fee_pct_per_leg": args.fee_pct,
                "bar_minutes": 240,
                "limitations": [
                    "Same candidates compared on both halves; no held-out walk-forward selection.",
                    "Four-hour bars do not validate five-minute guards.",
                    "Assets use independent cash accounts; the sum is not a shared-wallet simulation.",
                    "The replay fee is an explicit assumption, not a verified live venue fee.",
                ],
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
    top_verified = [r for r in results if r.get("beats_baseline_both_halves")][:15]

    lines = [
        "# Deep Parabolic Surge Guard & Re-entry Optimization Report",
        "",
        f"- **Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Candidates Evaluated:** {len(results):,}",
        f"- **Total Time:** {total_time/60:.1f} minutes",
        f"- **Profile:** {args.config.resolve()}",
        f"- **Cash per asset:** {initial_cash} {base_params.currency}; per-leg fee: {args.fee_pct}%",
        "- **Limits:** 4h bars cannot validate 5-minute guards. Assets use independent cash accounts.",
        "- **Selection:** both halves are compared, not held out; no live promotion is justified.",
        f"- **Assets:** {', '.join(ASSETS)} (supplied 4h datasets; durations are not validated)",
        f"- **Baseline Reference (Surge OFF):** ${base_pnl:,.2f} | H1: ${base_h1:,.2f} | H2: ${base_h2:,.2f}",
        "",
        "## 1. Top 15 Overall Performers (Sum of Independent Asset PnLs)",
        "",
        "| Rank | Candidate ID | Category | Total PnL ($) | Edge ($) | MaxDD % | PnL/DD point | Both Halves | Description |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]

    for idx, r in enumerate(top15, 1):
        v_tag = "YES" if r["beats_baseline_both_halves"] else "No"
        lines.append(
            f"| {idx} | `{r['id']}` | {r['category']} | **${r['total_profit_usd']:,.2f}** | "
            f"+${r['edge_usd']:,.2f} | {r['worst_maxdd_pct']:.2f}% | {r['pnl_per_drawdown_point']:.2f} | {v_tag} | {r['desc']} |"
        )

    lines.extend([
        "",
        "## 2. Candidates Matching or Beating Both Halves (Descriptive Comparison)",
        "",
        "> [!IMPORTANT]",
        "> These candidates match or beat the control on both halves. This is an in-sample comparison, not proof against overfitting.",
        "",
        "| Rank | Candidate ID | Category | Total PnL ($) | H1 ($) | H2 ($) | MaxDD % | PnL/DD point | Description |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ])

    for idx, r in enumerate(top_verified, 1):
        lines.append(
            f"| {idx} | `{r['id']}` | {r['category']} | **${r['total_profit_usd']:,.2f}** | "
            f"${r['h1_profit_usd']:,.2f} | ${r['h2_profit_usd']:,.2f} | {r['worst_maxdd_pct']:.2f}% | {r['pnl_per_drawdown_point']:.2f} | {r['desc']} |"
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
        "Research shortlist only. Do not promote to either venue without held-out, fee-calibrated, timeframe-appropriate validation:",
        "```env",
    ])
    if top_verified:
        top_cand = top_verified[0]
    else:
        top_cand = top15[0] if top15 else None

    if top_cand:
        lines.append(f"# Research Candidate: {top_cand['id']}")
        lines.append(f"# Total PnL: ${top_cand['total_profit_usd']} vs Baseline ${base_pnl}")
        lines.append(f"# Description: {top_cand['desc']}")
    lines.append("```\n")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Generated comprehensive markdown report: {md_path}")
    print("\n=== TOP 5 CANDIDATES PREVIEW ===")
    for idx, r in enumerate(top15[:5], 1):
        print(f"#{idx} {r['id']:<35} | PnL: ${r['total_profit_usd']:>9.2f} | MaxDD: {r['worst_maxdd_pct']:>5.2f}% | 2-Win: {r['beats_baseline_both_halves']} | {r['desc']}")


if __name__ == "__main__":
    main()
