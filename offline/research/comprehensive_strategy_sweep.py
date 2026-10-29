#!/usr/bin/env python3
"""Comprehensive multi-factor backtest and strategy sweep.

Evaluates high-profit / low-drawdown ("profit mare si pierdere mica") candidates across:
1. Walk-Forward 31 out-of-sample folds (628 days, HYPE 240m)
2. Full continuous replay over 628 days (~3,772 bars)
3. Live Kraken forward window (206 forward bars + 720 historical bars)
"""
from __future__ import annotations

import dataclasses
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "kraken") not in sys.path:
    sys.path.append(str(ROOT / "kraken"))

from botcore import (
    safe_float,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
)
from kraken import replay as kraken_replay
from strategies.spot_engine import StratParams
from strategies.spot_replay import align_spot_interval_params
from offline.backtests.datasets import load_dataset
from offline.backtests.walk_forward import walk_forward_splits
from offline.backtests.evaluation import evaluate_segment, load_layered_env_config
from offline.backtests.execution import ExecutionModel

# Load live configuration into an isolated dict without polluting os.environ
KRAKEN_DIR = ROOT / "kraken"
_KRAKEN_CFG = load_layered_env_config(
    KRAKEN_DIR / "config.env",
    KRAKEN_DIR / ".env",
    override_prefixes=("KRAKEN_",),
)
LIVE_PARAMS = StratParams.from_env(_KRAKEN_CFG)

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


def _summarize_replay(m: dict, budget: float) -> dict[str, float | int]:
    """Compute normalized percentage and risk-adjusted replay metrics safely."""
    safe_budget = safe_pos_float(budget, default=1e-9)
    m_dict = m if isinstance(m, dict) else {}
    net_pct = (safe_float(m_dict.get("net"), 0.0) / safe_budget) * 100.0
    total_pct = (safe_float(m_dict.get("total"), 0.0) / safe_budget) * 100.0
    max_dd = safe_float(m_dict.get("max_drawdown_pct"), 0.0)
    calmar = (net_pct / max_dd) if max_dd > 0 else 0.0
    sharpe = safe_float(m_dict.get("sharpe_ratio"), 0.0)
    return {
        "net_pct": net_pct,
        "total_pct": total_pct,
        "max_dd": max_dd,
        "calmar": calmar,
        "sharpe": sharpe,
        "cycles": safe_nonneg_int(m_dict.get("cycles"), default=0),
        "wins": safe_nonneg_int(m_dict.get("wins"), default=0),
        "fills": safe_nonneg_int(m_dict.get("fills"), default=0),
    }


def run_walk_forward_evaluation(candidates: dict[str, dict], records: list[dict]) -> dict[str, dict]:
    """Run 31 out-of-sample walk-forward folds for each candidate."""
    print(">>> Running 31 Walk-Forward Out-Of-Sample Folds (628 days)...")
    folds = walk_forward_splits(
        len(records), train_size=720, validation_size=180, test_size=90, step_size=90
    )
    if not folds:
        return {}
    results = {}
    fee_pct = 0.26
    interval = 240
    warmup_bars = 40

    live_candidate_windows = []

    for name, overrides in candidates.items():
        params = align_spot_interval_params(
            dataclasses.replace(LIVE_PARAMS, **overrides), interval
        )
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

        mean_ret = statistics.fmean(returns) if returns else 0.0
        median_ret = statistics.median(returns) if returns else 0.0
        worst_ret = min(returns) if returns else 0.0
        worst_dd = max(drawdowns) if drawdowns else 0.0
        mean_dd = statistics.fmean(drawdowns) if drawdowns else 0.0
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
            data["vs_live_WTL"] = f"{len(folds)}/0/0 (Ref)"
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
    if len(ohlc) < 2:
        return {}
    results = {}
    fee_pct = 0.26
    interval = 240
    if len(ohlc) > 40:
        kraken_replay.prewarm_trend_series(ohlc, warmup=0, params=LIVE_PARAMS, bar_minutes=interval)

    for name, overrides in candidates.items():
        params = align_spot_interval_params(
            dataclasses.replace(LIVE_PARAMS, **overrides), interval
        )
        m = kraken_replay.run_replay(
            ohlc, params, fee_pct=fee_pct, bar_minutes=interval,
            execution=ExecutionModel(),
        )
        s = _summarize_replay(m, params.effective_max_budget())
        results[name] = {
            "cont_net_pct": s["net_pct"],
            "cont_total_pct": s["total_pct"],
            "cont_max_dd": s["max_dd"],
            "cont_calmar": s["calmar"],
            "cont_sharpe": s["sharpe"],
            "cont_cycles": s["cycles"],
            "cont_wins": s["wins"],
            "cont_fills": s["fills"],
        }
    return results


def run_live_forward_shadow(candidates: dict[str, dict]) -> dict[str, dict]:
    """Run replay over the forward live shadow data (from Kraken anchor)."""
    print(">>> Running Live Shadow Evaluation (Forward ~34 days & Full 120-day window)...")
    if not SHADOW_OHLC_JSON.exists():
        print(f"Shadow OHLC not found at {SHADOW_OHLC_JSON}")
        return {}

    from state_io import read_json_file
    full_bars = read_json_file(SHADOW_OHLC_JSON, default=[])

    if not isinstance(full_bars, list) or len(full_bars) < 2:
        print(f"Shadow OHLC at {SHADOW_OHLC_JSON} has insufficient bars ({len(full_bars) if isinstance(full_bars, list) else 0} < 2)")
        return {}

    # Load anchor ts
    anchor_path = SHADOW_OHLC_JSON.parent / "HYPEUSD_240m.anchor"
    anchor_ts = 0
    if anchor_path.exists():
        try:
            with open(anchor_path, encoding="utf-8") as f:
                anchor_ts = safe_nonneg_int(f.read().strip(), default=0)
        except OSError:
            anchor_ts = 0

    valid_bars = []
    for b in full_bars:
        if not isinstance(b, (list, tuple)) or len(b) < 5:
            continue
        ts_f = safe_nonneg_float(b[0], default=None)
        o_f = safe_pos_float(b[1], default=None)
        h_f = safe_pos_float(b[2], default=None)
        l_f = safe_pos_float(b[3], default=None)
        c_f = safe_pos_float(b[4], default=None)
        if (
            ts_f is not None
            and o_f is not None
            and h_f is not None
            and l_f is not None
            and c_f is not None
            and h_f >= max(o_f, l_f, c_f)
            and l_f <= min(o_f, h_f, c_f)
        ):
            valid_bars.append((ts_f, o_f, h_f, l_f, c_f))
    fwd_bars = [b for b in valid_bars if b[0] >= anchor_ts]
    full_ohlc = [(b[1], b[2], b[3], b[4]) for b in valid_bars]
    fwd_ohlc = [(b[1], b[2], b[3], b[4]) for b in fwd_bars]

    if len(fwd_ohlc) < 2 or len(full_ohlc) < 2:
        print(f"Insufficient forward or full bars for shadow replay (fwd: {len(fwd_ohlc)}, full: {len(full_ohlc)})")
        return {}

    results = {}
    fee_pct = 0.26
    interval = 240

    for name, overrides in candidates.items():
        params = align_spot_interval_params(
            dataclasses.replace(LIVE_PARAMS, **overrides), interval
        )
        budget = params.effective_max_budget()

        # Forward bars
        m_fwd = kraken_replay.run_replay(
            fwd_ohlc, params, fee_pct=fee_pct, bar_minutes=interval,
            execution=ExecutionModel(),
        )
        s_fwd = _summarize_replay(m_fwd, budget)

        # Full 720 bars
        m_full = kraken_replay.run_replay(
            full_ohlc, params, fee_pct=fee_pct, bar_minutes=interval,
            execution=ExecutionModel(),
        )
        s_full = _summarize_replay(m_full, budget)

        results[name] = {
            "fwd_net": s_fwd["net_pct"],
            "fwd_total": s_fwd["total_pct"],
            "fwd_dd": s_fwd["max_dd"],
            "fwd_cycles": s_fwd["cycles"],
            "full_net": s_full["net_pct"],
            "full_total": s_full["total_pct"],
            "full_dd": s_full["max_dd"],
            "full_cycles": s_full["cycles"],
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
    json_path = out_dir / "sweep_results.json"
    from state_io import atomic_write_json, atomic_write_text
    atomic_write_json(json_path, master, indent=2)
    print(f"\nSaved raw JSON to {json_path}")

    # Generate Markdown Summary Report
    md_path = out_dir / "SWEEP_REPORT.md"
    lines = [
        "# Strategy Sweep: High Profit & Low Drawdown (628 Days + Live Shadow)\n\n",
        "Evaluation across:\n",
        "1. **Walk-Forward (WF)**: 31 Out-of-Sample folds (~628 days, HYPE 240m)\n",
        "2. **Continuous 628d**: Unbroken compounding replay across all 3,772 bars\n",
        "3. **Kraken Shadow**: 206 forward bars (last 34 days) & 720-bar Kraken window (~120 days)\n\n",
        "## 1. Walk-Forward Out-Of-Sample Performance (31 Folds)\n\n",
        "| Rank | Candidate | WF Mean % | WF Median % | WF Worst % | WF Worst DD % | Pos Folds | W/T/L vs Live |\n",
        "|---:|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    sorted_wf = sorted(master, key=lambda x: (x.get("wf_mean_ret", 0), -x.get("wf_worst_dd", 100)), reverse=True)
    for rank, r in enumerate(sorted_wf, 1):
        lines.append(
            f"| {rank} | `{r['name']}` | {r.get('wf_mean_ret', 0):+.3f}% | {r.get('wf_median_ret', 0):+.3f}% | "
            f"{r.get('wf_worst_ret', 0):+.2f}% | {r.get('wf_worst_dd', 0):.2f}% | "
            f"{r.get('wf_pos_count', 0)}/{r.get('wf_total_folds', 31)} | {r.get('vs_live_WTL', '-')} |\n"
        )

    lines.append("\n## 2. Continuous 628-Day Compounding Replay (~3,772 bars)\n\n")
    lines.append("| Rank | Candidate | Net % | Total % | MaxDD % | Calmar Ratio | Cycles | Win Rate |\n")
    lines.append("|---:|---|---:|---:|---:|---:|---:|\n")
    sorted_cont = sorted(master, key=lambda x: x.get("cont_calmar", 0), reverse=True)
    for rank, r in enumerate(sorted_cont, 1):
        cycles = r.get("cont_cycles", 0)
        wins = r.get("cont_wins", 0)
        win_rate = (wins / cycles * 100.0) if cycles > 0 else 0.0
        lines.append(
            f"| {rank} | `{r['name']}` | {r.get('cont_net_pct', 0):+.2f}% | {r.get('cont_total_pct', 0):+.2f}% | "
            f"{r.get('cont_max_dd', 0):.2f}% | **{r.get('cont_calmar', 0):.2f}** | {cycles} | {win_rate:.1f}% |\n"
        )

    lines.append("\n## 3. Kraken Live Shadow Performance (Forward 34d & Window 120d)\n\n")
    if not shadow_results:
        lines.append("*Live shadow evaluation skipped: insufficient forward bars recorded in `logs/shadow_live/` (minimum 2 bars required).* \n")
    else:
        lines.append("| Candidate | Forward Net % | Forward Total % | Forward DD % | 120d Net % | 120d Total % | 120d DD % |\n")
        lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in master:
            lines.append(
                f"| `{r['name']}` | {r.get('fwd_net', 0):+.2f}% | {r.get('fwd_total', 0):+.2f}% | "
                f"{r.get('fwd_dd', 0):.2f}% | {r.get('full_net', 0):+.2f}% | {r.get('full_total', 0):+.2f}% | "
                f"{r.get('full_dd', 0):.2f}% |\n"
            )

    atomic_write_text(md_path, "".join(lines))

    print(f"Report written to {md_path}")


if __name__ == "__main__":
    main()
