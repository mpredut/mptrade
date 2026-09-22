#!/usr/bin/env python3
"""Comprehensive multi-factor backtest and strategy sweep.

Evaluates high-profit / low-drawdown ("profit mare si pierdere mica") candidates across:
1. Walk-Forward 31 out-of-sample folds (628 days, HYPE 240m)
2. Full continuous replay over 628 days (~3,772 bars)
3. Live Kraken forward window (206 forward bars + 720 historical bars)
"""
from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kraken"))
sys.path.insert(0, str(ROOT))

from kraken_common import load_dotenv
import replay as kraken_replay
from strategies.spot_dca import StratParams
from offline.backtests.datasets import load_dataset
from offline.backtests.walk_forward import walk_forward_splits
from offline.backtests.evaluation import evaluate_segment
from offline.backtests.execution import ExecutionModel

# Load live configuration
KRAKEN_DIR = ROOT / "kraken"
load_dotenv(KRAKEN_DIR / ".env")
load_dotenv(KRAKEN_DIR / "config.env")
LIVE_PARAMS = StratParams.from_env()

DATASET_CSV = ROOT / "offline" / "research" / "hype_dataset" / "HYPEUSDC_240m_hlspot.csv"
SHADOW_OHLC_JSON = ROOT / "logs" / "shadow_live" / "HYPEUSD_240m.ohlc.json"
SHADOW_JSONL = ROOT / "logs" / "shadow_live" / "HYPEUSD_240m.jsonl"


def define_candidate_grid() -> dict[str, dict]:
    """Define focused, intelligent candidate variations."""
    return {
        # Baseline
        "00_LIVE_CURRENT": {},

        # --- Take Profit Variations ---
        "01_tp4": {"takeprofit_pct": 4.0},
        "02_tp45": {"takeprofit_pct": 4.5},
        "03_tp6": {"takeprofit_pct": 6.0},

        # --- Adaptive Trailing Variations (Combined with Live tp_regime_gate) ---
        "04_adaptive_trail_k15": {
            "tp_trail_adaptive": True,
            "tp_trail_k": 1.5,
            "tp_trail_min": 1.5,
            "tp_trail_max": 6.0,
            "tp_trail_vol_interval": 240,
        },
        "05_adaptive_trail_k20": {
            "tp_trail_adaptive": True,
            "tp_trail_k": 2.0,
            "tp_trail_min": 1.5,
            "tp_trail_max": 8.0,
            "tp_trail_vol_interval": 240,
        },
        "06_adaptive_trail_k25": {
            "tp_trail_adaptive": True,
            "tp_trail_k": 2.5,
            "tp_trail_min": 2.0,
            "tp_trail_max": 10.0,
            "tp_trail_vol_interval": 240,
        },

        # --- Progressive DCA Spacing (Preserves Cash for Deeper Dips) ---
        "07_dca_spacing_growth_020": {"dca_spacing_growth_pct": 0.20},
        "08_dca_spacing_growth_035": {"dca_spacing_growth_pct": 0.35},
        "09_dca_spacing_growth_050": {"dca_spacing_growth_pct": 0.50},

        # --- DCA Drop Baseline Variations ---
        "10_dca_drop_150": {"dca_drop_pct": 1.50},
        "11_dca_drop_175": {"dca_drop_pct": 1.75},

        # --- DCA Downtrend Brake (Avoids Catching Falling Knives) ---
        "12_dca_trend_brake_10": {
            "dca_trend_brake": True,
            "dca_brake_min_pct": 1.0,
            "trend_interval": 240,
        },
        "13_dca_trend_brake_15": {
            "dca_trend_brake": True,
            "dca_brake_min_pct": 1.5,
            "trend_interval": 240,
        },
        "14_dca_trend_brake_20": {
            "dca_trend_brake": True,
            "dca_brake_min_pct": 2.0,
            "trend_interval": 240,
        },

        # --- Re-entry Pullback After Take Profit ---
        "15_reentry_pullback_15": {"reentry_drop_pct": 1.5},
        "16_reentry_pullback_25": {"reentry_drop_pct": 2.5},
        "17_reentry_pullback_40": {"reentry_drop_pct": 4.0},

        # --- Trend Overlay Variations ---
        "18_overlay_350t6": {
            "trend_overlay": True,
            "trend_topup": 350.0,
            "trend_trail_pct": 6.0,
            "trend_interval": 240,
        },
        "19_overlay_650t8": {
            "trend_overlay": True,
            "trend_topup": 650.0,
            "trend_trail_pct": 8.0,
            "trend_interval": 240,
        },
        "20_overlay_650t8_exitbreak": {
            "trend_overlay": True,
            "trend_topup": 650.0,
            "trend_trail_pct": 8.0,
            "trend_exit_break": True,
            "trend_interval": 240,
        },

        # --- Stop Loss Optimization ---
        "21_sl_15": {"stop_loss_pct": 15.0},
        "22_sl_20": {"stop_loss_pct": 20.0},
        "23_sl_25": {"stop_loss_pct": 25.0},
        "24_sl_off": {"stop_loss_pct": 0.0},

        # --- SMART COMBINATIONS (High Return & Low Drawdown Synergies) ---
        "25_COMBO_smart_flow": {
            # TP 4% + Adaptive Trail + Progressive DCA 0.25
            "takeprofit_pct": 4.0,
            "tp_trail_adaptive": True,
            "tp_trail_k": 2.0,
            "tp_trail_min": 1.5,
            "tp_trail_max": 8.0,
            "tp_trail_vol_interval": 240,
            "dca_spacing_growth_pct": 0.25,
        },
        "26_COMBO_defensive_alpha": {
            # TP 4% + DCA Brake 1.5% + Progressive DCA 0.35 + Reentry 1.5%
            "takeprofit_pct": 4.0,
            "dca_trend_brake": True,
            "dca_brake_min_pct": 1.5,
            "dca_spacing_growth_pct": 0.35,
            "reentry_drop_pct": 1.5,
            "trend_interval": 240,
        },
        "27_COMBO_high_calmar_runner": {
            # TP 4% + Adaptive Trail + DCA Brake 1.5% + Progressive DCA 0.25 + Reentry 1.5%
            "takeprofit_pct": 4.0,
            "tp_trail_adaptive": True,
            "tp_trail_k": 2.0,
            "tp_trail_min": 1.5,
            "tp_trail_max": 8.0,
            "tp_trail_vol_interval": 240,
            "dca_trend_brake": True,
            "dca_brake_min_pct": 1.5,
            "dca_spacing_growth_pct": 0.25,
            "reentry_drop_pct": 1.5,
            "trend_interval": 240,
        },
        "28_COMBO_overlay_safe": {
            # Overlay 350t6 + TP 4% + Progressive DCA 0.25
            "trend_overlay": True,
            "trend_topup": 350.0,
            "trend_trail_pct": 6.0,
            "takeprofit_pct": 4.0,
            "dca_spacing_growth_pct": 0.25,
            "trend_interval": 240,
        },
        "29_COMBO_ultimate_balanced": {
            # TP 4.5% + Adaptive Trail k=1.8 + DCA 1.35% with 0.25 growth + Brake 1.5% + Reentry 1.5%
            "takeprofit_pct": 4.5,
            "tp_trail_adaptive": True,
            "tp_trail_k": 1.8,
            "tp_trail_min": 1.5,
            "tp_trail_max": 7.5,
            "tp_trail_vol_interval": 240,
            "dca_drop_pct": 1.35,
            "dca_spacing_growth_pct": 0.25,
            "dca_trend_brake": True,
            "dca_brake_min_pct": 1.5,
            "reentry_drop_pct": 1.5,
            "trend_interval": 240,
        },
    }


def run_walk_forward_evaluation(candidates: dict[str, dict], records: list[dict]) -> dict[str, dict]:
    """Run 31 out-of-sample walk-forward folds for each candidate."""
    print(">>> Running 31 Walk-Forward Out-Of-Sample Folds (628 days)...")
    folds = walk_forward_splits(
        len(records), train_size=720, validation_size=180, test_size=90, step_size=90
    )
    results = {}
    fee_pct = 0.26
    interval = 240
    warmup_bars = 40

    live_candidate_windows = []

    for name, overrides in candidates.items():
        params = dataclasses.replace(LIVE_PARAMS, **overrides)
        windows = []
        for fold in folds:
            warmup = records[max(0, fold.test.start - warmup_bars):fold.test.start]
            seg = evaluate_segment(
                records[fold.test],
                lambda ohlc, w, _ctx: kraken_replay.run_replay(
                    ohlc, params, fee_pct=fee_pct, bar_minutes=interval,
                    warmup_ohlc=w, execution=ExecutionModel(),
                ),
                warmup_records=warmup,
            )
            windows.append(seg["metrics"])

        returns = [w["return_pct"] for w in windows]
        drawdowns = [w["max_drawdown_pct"] for w in windows]
        cycles = sum(w["cycles"] for w in windows)
        fills = sum(w["fills"] for w in windows)

        mean_ret = sum(returns) / len(returns)
        sorted_rets = sorted(returns)
        median_ret = (sorted_rets[len(sorted_rets)//2] if len(sorted_rets) % 2 == 1 
                      else (sorted_rets[len(sorted_rets)//2 - 1] + sorted_rets[len(sorted_rets)//2]) / 2)
        worst_ret = min(returns)
        worst_dd = max(drawdowns)
        mean_dd = sum(drawdowns) / len(drawdowns)
        positive_count = sum(1 for r in returns if r > 0)

        results[name] = {
            "wf_mean_ret": mean_ret,
            "wf_median_ret": median_ret,
            "wf_worst_ret": worst_ret,
            "wf_worst_dd": worst_dd,
            "wf_mean_dd": mean_dd,
            "wf_pos_count": positive_count,
            "wf_total_folds": len(folds),
            "wf_cycles": cycles,
            "wf_fills": fills,
            "raw_windows": returns,
        }
        if name == "00_LIVE_CURRENT":
            live_candidate_windows = returns

    # Compute pairwise W/T/L vs Live
    for name, data in results.items():
        if name == "00_LIVE_CURRENT":
            data["vs_live_WTL"] = "31/0/0 (Ref)"
            continue
        wins = sum(1 for cand, ref in zip(data["raw_windows"], live_candidate_windows) if cand > ref + 1e-6)
        ties = sum(1 for cand, ref in zip(data["raw_windows"], live_candidate_windows) if abs(cand - ref) <= 1e-6)
        losses = sum(1 for cand, ref in zip(data["raw_windows"], live_candidate_windows) if cand < ref - 1e-6)
        data["vs_live_WTL"] = f"{wins}/{ties}/{losses}"

    return results


def run_continuous_628d_replay(candidates: dict[str, dict], records: list[dict]) -> dict[str, dict]:
    """Run full unbroken continuous replay over the entire 628-day dataset."""
    print(">>> Running Continuous 628-Day Replay (~3,772 bars)...")
    ohlc = [(r["open"], r["high"], r["low"], r["close"]) for r in records]
    results = {}
    fee_pct = 0.26
    interval = 240

    for name, overrides in candidates.items():
        params = dataclasses.replace(LIVE_PARAMS, **overrides)
        budget = float(params.effective_max_budget())
        m = kraken_replay.run_replay(
            ohlc, params, fee_pct=fee_pct, bar_minutes=interval,
            execution=ExecutionModel(),
        )
        net_pct = (m["net"] / budget) * 100.0
        total_pct = (m["total"] / budget) * 100.0
        max_dd = m.get("max_drawdown_pct") or 0.0
        calmar = (net_pct / max_dd) if max_dd > 0 else 0.0
        sharpe = m.get("sharpe_ratio") or 0.0
        results[name] = {
            "cont_net_pct": net_pct,
            "cont_total_pct": total_pct,
            "cont_max_dd": max_dd,
            "cont_calmar": calmar,
            "cont_sharpe": sharpe,
            "cont_cycles": m.get("cycles", 0),
            "cont_wins": m.get("wins", 0),
            "cont_fills": m.get("fills", 0),
        }
    return results


def run_live_forward_shadow(candidates: dict[str, dict]) -> dict[str, dict]:
    """Run replay over the forward live shadow data (from Kraken anchor)."""
    print(">>> Running Live Shadow Evaluation (Forward ~34 days & Full 120-day window)...")
    if not SHADOW_OHLC_JSON.exists():
        print(f"Shadow OHLC not found at {SHADOW_OHLC_JSON}")
        return {}

    with open(SHADOW_OHLC_JSON, encoding="utf-8") as f:
        full_bars = json.load(f)

    # Load anchor ts
    anchor_path = ROOT / "logs" / "shadow_live" / "HYPEUSD_240m.anchor"
    anchor_ts = 0
    if anchor_path.exists():
        with open(anchor_path, encoding="utf-8") as f:
            anchor_ts = int(f.read().strip())

    fwd_bars = [b for b in full_bars if b[0] >= anchor_ts]
    full_ohlc = [(b[1], b[2], b[3], b[4]) for b in full_bars]
    fwd_ohlc = [(b[1], b[2], b[3], b[4]) for b in fwd_bars]

    results = {}
    fee_pct = 0.26
    interval = 240

    for name, overrides in candidates.items():
        params = dataclasses.replace(LIVE_PARAMS, **overrides)
        budget = float(params.effective_max_budget())

        # Forward bars
        m_fwd = kraken_replay.run_replay(
            fwd_ohlc, params, fee_pct=fee_pct, bar_minutes=interval,
            execution=ExecutionModel(),
        )
        fwd_net = (m_fwd["net"] / budget) * 100.0
        fwd_total = (m_fwd["total"] / budget) * 100.0
        fwd_dd = m_fwd.get("max_drawdown_pct") or 0.0

        # Full 720 bars
        m_full = kraken_replay.run_replay(
            full_ohlc, params, fee_pct=fee_pct, bar_minutes=interval,
            execution=ExecutionModel(),
        )
        full_net = (m_full["net"] / budget) * 100.0
        full_total = (m_full["total"] / budget) * 100.0
        full_dd = m_full.get("max_drawdown_pct") or 0.0

        results[name] = {
            "fwd_net": fwd_net,
            "fwd_total": fwd_total,
            "fwd_dd": fwd_dd,
            "fwd_cycles": m_fwd.get("cycles", 0),
            "full_net": full_net,
            "full_total": full_total,
            "full_dd": full_dd,
            "full_cycles": m_full.get("cycles", 0),
        }
    return results


def main():
    records = load_dataset(DATASET_CSV)
    print(f"Loaded {len(records)} bars from {DATASET_CSV}")

    candidates = define_candidate_grid()
    print(f"Defined {len(candidates)} candidate configurations to sweep.")

    wf_results = run_walk_forward_evaluation(candidates, records)
    cont_results = run_continuous_628d_replay(candidates, records)
    shadow_results = run_live_forward_shadow(candidates)

    # Combine into unified master dictionary
    master = []
    for name in candidates:
        row = {"name": name, "overrides": candidates[name]}
        row.update(wf_results.get(name, {}))
        row.update(cont_results.get(name, {}))
        row.update(shadow_results.get(name, {}))
        master.append(row)

    # Save complete JSON
    out_dir = ROOT / "offline" / "results" / "strategy_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "sweep_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2)
    print(f"\nSaved raw JSON to {json_path}")

    # Generate Markdown Summary Report
    md_path = out_dir / "SWEEP_REPORT.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Strategy Sweep: High Profit & Low Drawdown (628 Days + Live Shadow)\n\n")
        f.write("Evaluation across:\n")
        f.write("1. **Walk-Forward (WF)**: 31 Out-of-Sample folds (~628 days, HYPE 240m)\n")
        f.write("2. **Continuous 628d**: Unbroken compounding replay across all 3,772 bars\n")
        f.write("3. **Kraken Shadow**: 206 forward bars (last 34 days) & 720-bar Kraken window (~120 days)\n\n")

        # Table 1: Walk-Forward Out-Of-Sample (The true test of robustness)
        f.write("## 1. Walk-Forward Out-Of-Sample Performance (31 Folds)\n\n")
        f.write("| Rank | Candidate | WF Mean % | WF Median % | WF Worst % | WF Worst DD % | Pos Folds | W/T/L vs Live |\n")
        f.write("|---:|---|---:|---:|---:|---:|---:|---:|\n")
        sorted_wf = sorted(master, key=lambda x: (x.get("wf_mean_ret", 0), -x.get("wf_worst_dd", 100)), reverse=True)
        for rank, r in enumerate(sorted_wf, 1):
            f.write(f"| {rank} | `{r['name']}` | {r.get('wf_mean_ret', 0):+.3f}% | {r.get('wf_median_ret', 0):+.3f}% | "
                    f"{r.get('wf_worst_ret', 0):+.2f}% | {r.get('wf_worst_dd', 0):.2f}% | "
                    f"{r.get('wf_pos_count', 0)}/{r.get('wf_total_folds', 31)} | {r.get('vs_live_WTL', '-')} |\n")

        # Table 2: Continuous 628-Day Replay (Total Compounding & Risk-Adjusted Calmar)
        f.write("\n## 2. Continuous 628-Day Compounding Replay (~3,772 bars)\n\n")
        f.write("| Rank | Candidate | Net % | Total % | MaxDD % | Calmar Ratio | Cycles | Win Rate |\n")
        f.write("|---:|---|---:|---:|---:|---:|---:|---:|\n")
        sorted_cont = sorted(master, key=lambda x: x.get("cont_calmar", 0), reverse=True)
        for rank, r in enumerate(sorted_cont, 1):
            cycles = r.get("cont_cycles", 0)
            wins = r.get("cont_wins", 0)
            win_rate = (wins / cycles * 100.0) if cycles > 0 else 0.0
            f.write(f"| {rank} | `{r['name']}` | {r.get('cont_net_pct', 0):+.2f}% | {r.get('cont_total_pct', 0):+.2f}% | "
                    f"{r.get('cont_max_dd', 0):.2f}% | **{r.get('cont_calmar', 0):.2f}** | {cycles} | {win_rate:.1f}% |\n")

        # Table 3: Kraken Live Forward Shadow Performance
        f.write("\n## 3. Kraken Live Shadow Performance (Forward 34d & Window 120d)\n\n")
        f.write("| Candidate | Forward Net % | Forward Total % | Forward DD % | 120d Net % | 120d Total % | 120d DD % |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in master:
            f.write(f"| `{r['name']}` | {r.get('fwd_net', 0):+.2f}% | {r.get('fwd_total', 0):+.2f}% | "
                    f"{r.get('fwd_dd', 0):.2f}% | {r.get('full_net', 0):+.2f}% | {r.get('full_total', 0):+.2f}% | "
                    f"{r.get('full_dd', 0):.2f}% |\n")

    print(f"Report written to {md_path}")


if __name__ == "__main__":
    main()
