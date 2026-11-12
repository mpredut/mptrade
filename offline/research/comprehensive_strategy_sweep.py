#!/usr/bin/env python3
"""Comprehensive multi-factor backtest and strategy sweep.

Evaluates high-profit / low-drawdown candidates across:
1. Walk-Forward 31 out-of-sample folds (628 days, HYPE 240m)
2. Full continuous replay over 628 days (~3,772 bars)
3. Live Kraken forward window (206 forward bars + 720 historical bars)
"""
from __future__ import annotations

import dataclasses
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from botcore import (
    load_local_module,
    normalize_epoch_sec,
    safe_float,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
)
from state_io import atomic_write_json, atomic_write_text, read_json_file, read_text_file
from strategies.spot_engine import StratParams
from strategies.replay_metrics import write_markdown_report
from strategies import spot_replay as kraken_replay
from strategies.spot_replay import prepare_spot_interval_params
from offline.backtests.datasets import load_dataset
from offline.backtests.walk_forward import walk_forward_splits
from offline.backtests.evaluation import evaluate_segment
from offline.backtests.execution import ExecutionModel

# Load live configuration into an isolated dict without polluting os.environ
KRAKEN_DIR = ROOT / "kraken"
load_isolated_kraken_cfg = load_local_module(
    str(KRAKEN_DIR), "kraken_common", module_key="kraken_common"
).load_isolated_kraken_cfg
_KRAKEN_CFG = load_isolated_kraken_cfg(KRAKEN_DIR)
LIVE_PARAMS = StratParams.from_env(_KRAKEN_CFG)

DATASET_CSV = ROOT / "offline" / "research" / "hype_dataset" / "HYPEUSDC_240m_hlspot.csv"
SHADOW_OHLC_JSON = ROOT / "logs" / "shadow_live" / "HYPEUSD_240m.ohlc.json"


def define_candidate_grid() -> dict[str, dict]:
    """Define focused, unmasked, non-duplicate candidate variations."""
    return {
        # Baseline (live config: TP=6.0, tp_dynamic_flat=True, tp_mdhtp_enabled=True, brake=1.5%, overlay=650t8, SL=25%)
        "00_LIVE_CURRENT": {},

        # --- Fixed Take Profit Variations (unmasked: disable tp_dynamic_flat, tp_mdhtp_enabled, and tp_trend_hold) ---
        "01_tp4": {"takeprofit_pct": 4.0, "tp_dynamic_flat": False, "tp_mdhtp_enabled": False, "tp_trend_hold": False},
        "02_tp5": {"takeprofit_pct": 5.0, "tp_dynamic_flat": False, "tp_mdhtp_enabled": False, "tp_trend_hold": False},
        "03_tp7": {"takeprofit_pct": 7.0, "tp_dynamic_flat": False, "tp_mdhtp_enabled": False, "tp_trend_hold": False},

        # --- Adaptive Trailing Variations (Combined with Live tp_regime_gate) ---
        "04_adaptive_trail_k15": {
            "tp_mdhtp_enabled": False,
            "tp_trail_adaptive": True,
            "tp_trail_k": 1.5,
            "tp_trail_min": 1.5,
            "tp_trail_max": 6.0,
            "tp_trail_vol_interval": 240,
        },
        "05_adaptive_trail_k20": {
            "tp_mdhtp_enabled": False,
            "tp_trail_adaptive": True,
            "tp_trail_k": 2.0,
            "tp_trail_min": 1.5,
            "tp_trail_max": 8.0,
            "tp_trail_vol_interval": 240,
        },
        "06_adaptive_trail_k25": {
            "tp_mdhtp_enabled": False,
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

        # --- DCA Downtrend Brake (Live already uses 1.5%; test 1.0%, OFF, and 2.0%) ---
        "12_dca_trend_brake_10": {
            "dca_trend_brake": True,
            "dca_brake_min_pct": 1.0,
            "trend_interval": 240,
        },
        "13_dca_trend_brake_off": {
            "dca_trend_brake": False,
        },
        "14_dca_trend_brake_20": {
            "dca_trend_brake": True,
            "dca_brake_min_pct": 2.0,
            "trend_interval": 240,
        },

        # --- Classic Re-entry Drop Variations (unmasked: disable reentry_hybrid_enabled, TTL bypass, and adaptive pullback) ---
        "15_reentry_pullback_15": {"reentry_hybrid_enabled": False, "reentry_ttl_hours": 0.0, "reentry_pullback_adaptive": False, "reentry_drop_pct": 1.5},
        "16_reentry_pullback_30": {"reentry_hybrid_enabled": False, "reentry_ttl_hours": 0.0, "reentry_pullback_adaptive": False, "reentry_drop_pct": 3.0},
        "17_reentry_pullback_40": {"reentry_hybrid_enabled": False, "reentry_ttl_hours": 0.0, "reentry_pullback_adaptive": False, "reentry_drop_pct": 4.0},

        # --- Trend Overlay Variations (Live already uses 650t8; test 350t6, OFF, and 650t8_exitbreak) ---
        "18_overlay_350t6": {
            "trend_overlay": True,
            "trend_topup": 350.0,
            "trend_trail_pct": 6.0,
            "trend_interval": 240,
        },
        "19_overlay_off": {
            "trend_overlay": False,
        },
        "20_overlay_650t8_exitbreak": {
            "trend_overlay": True,
            "trend_topup": 650.0,
            "trend_trail_pct": 8.0,
            "trend_exit_break": True,
            "trend_interval": 240,
        },

        # --- Stop Loss Optimization (Live uses 25.0%; test 12, 15, 20, OFF) ---
        "21_sl_15": {"stop_loss_pct": 15.0},
        "22_sl_20": {"stop_loss_pct": 20.0},
        "23_sl_12": {"stop_loss_pct": 12.0},
        "24_sl_off": {"stop_loss_pct": 0.0},

        # --- SMART COMBINATIONS (High Return & Low Drawdown Synergies) ---
        "25_COMBO_smart_flow": {
            # TP 4% + Adaptive Trail + Progressive DCA 0.25
            "takeprofit_pct": 4.0,
            "tp_dynamic_flat": False,
            "tp_mdhtp_enabled": False,
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
            "tp_dynamic_flat": False,
            "tp_mdhtp_enabled": False,
            "dca_trend_brake": True,
            "dca_brake_min_pct": 1.5,
            "dca_spacing_growth_pct": 0.35,
            "reentry_drop_pct": 1.5,
            "trend_interval": 240,
        },
        "27_COMBO_high_calmar_runner": {
            # TP 4% + Adaptive Trail + DCA Brake 1.5% + Progressive DCA 0.25 + Reentry 1.5%
            "takeprofit_pct": 4.0,
            "tp_dynamic_flat": False,
            "tp_mdhtp_enabled": False,
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
            "tp_dynamic_flat": False,
            "tp_mdhtp_enabled": False,
            "dca_spacing_growth_pct": 0.25,
            "trend_interval": 240,
        },
        "29_COMBO_ultimate_balanced": {
            # TP 4.5% + Adaptive Trail k=1.8 + DCA 1.35% with 0.25 growth + Brake 1.5% + Reentry 1.5%
            "takeprofit_pct": 4.5,
            "tp_dynamic_flat": False,
            "tp_mdhtp_enabled": False,
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
    if "calmar" in m_dict:
        calmar = safe_float(m_dict.get("calmar"), 0.0)
    else:
        calmar = (total_pct / max_dd) if max_dd > 0 else 0.0
    raw_sharpe = m_dict.get("sharpe") if m_dict.get("sharpe") is not None else m_dict.get("sharpe_ratio")
    sharpe = safe_float(raw_sharpe, 0.0)
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
    results, fee_pct, interval, warmup_bars = {}, 0.26, 240, 40

    live_candidate_windows = []

    for name, overrides in candidates.items():
        params = prepare_spot_interval_params(
            dataclasses.replace(LIVE_PARAMS, **overrides), interval
        )
        adapter = kraken_replay.make_spot_replay_adapter(
            params,
            fee_pct=fee_pct,
            bar_minutes=interval,
            execution=ExecutionModel(),
            initial_cash=params.allocated_budget(),
        )
        windows = []
        for fold in folds:
            warmup = records[max(0, fold.test.start - warmup_bars):fold.test.start]
            seg = evaluate_segment(
                records[fold.test],
                adapter,
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
            data["vs_live_WTL"] = f"0/{len(folds)}/0 (Ref)"
            continue
        if not live_candidate_windows:
            data["vs_live_WTL"] = "-"
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
    timestamps = [r["timestamp"] for r in records]
    if len(ohlc) < 2:
        return {}
    results: dict[str, dict] = {}
    fee_pct, interval = 0.26, 240
    if len(ohlc) > 40:
        kraken_replay.prewarm_trend_series(ohlc, warmup=0, params=LIVE_PARAMS, bar_minutes=interval)

    for name, overrides in candidates.items():
        params = prepare_spot_interval_params(dataclasses.replace(LIVE_PARAMS, **overrides), interval)
        budget = params.allocated_budget()
        m = kraken_replay.run_replay(
            ohlc, params, fee_pct=fee_pct, bar_minutes=interval,
            timestamps=timestamps,
            execution=ExecutionModel(),
            initial_cash=budget,
        )
        s = _summarize_replay(m, budget)
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


def run_live_forward_shadow(
    candidates: dict[str, dict],
    records: list[dict] | None = None,
) -> dict[str, dict]:
    """Run replay over the forward live shadow data (from Kraken anchor) and trailing 720-bar window."""
    print(">>> Running Live Shadow Evaluation (Forward from anchor & 720-bar window)...")
    if not SHADOW_OHLC_JSON.exists():
        print(f"Shadow OHLC not found at {SHADOW_OHLC_JSON}")
        return {}

    full_bars = read_json_file(SHADOW_OHLC_JSON, default=[])

    if not isinstance(full_bars, list) or len(full_bars) < 2:
        print(f"Shadow OHLC at {SHADOW_OHLC_JSON} has insufficient bars ({len(full_bars) if isinstance(full_bars, list) else 0} < 2)")
        return {}

    # Load anchor ts
    anchor_path = SHADOW_OHLC_JSON.parent / "HYPEUSD_240m.anchor"
    anchor_raw = (
        normalize_epoch_sec(read_text_file(anchor_path, default="").strip(), default=None)
        if anchor_path.exists()
        else None
    )
    anchor_ts = safe_nonneg_int(round(anchor_raw), default=0) if anchor_raw is not None and anchor_raw >= 0.0 else 0

    def _parse_valid_ohlc_bar(raw_ts, raw_o, raw_h, raw_l, raw_c):
        ts_f = normalize_epoch_sec(raw_ts, default=None)
        if ts_f is None or ts_f < 0.0:
            return None
        o_f = safe_pos_float(raw_o, default=None)
        h_f = safe_pos_float(raw_h, default=None)
        l_f = safe_pos_float(raw_l, default=None)
        c_f = safe_pos_float(raw_c, default=None)
        if (
            None not in (o_f, h_f, l_f, c_f)
            and h_f >= max(o_f, l_f, c_f)
            and l_f <= min(o_f, h_f, c_f)
        ):
            return (ts_f, o_f, h_f, l_f, c_f)
        return None

    valid_bars = [
        bar
        for b in full_bars
        if isinstance(b, (list, tuple))
        and len(b) >= 5
        and (bar := _parse_valid_ohlc_bar(b[0], b[1], b[2], b[3], b[4])) is not None
    ]
    pre_bars = [b for b in valid_bars if b[0] < anchor_ts]
    fwd_bars = [b for b in valid_bars if b[0] >= anchor_ts]

    # shadow_live.py persists only post-anchor bars (>= anchor_ts) in SHADOW_OHLC_JSON;
    # backfill pre-anchor context from historical records when provided so fwd_warmup and
    # the trailing 720-bar (~120d) window are distinct from fwd_bars.
    if not pre_bars and records:
        first_fwd_ts = fwd_bars[0][0] if fwd_bars else safe_nonneg_float(anchor_ts, default=0.0)
        for r in records:
            if isinstance(r, dict) and (
                bar := _parse_valid_ohlc_bar(r.get("timestamp"), r.get("open"), r.get("high"), r.get("low"), r.get("close"))
            ) is not None and bar[0] < first_fwd_ts:
                pre_bars.append(bar)
        pre_bars.sort(key=lambda item: item[0])

    pre_by_ts = {b[0]: b for b in pre_bars}
    fwd_by_ts = {b[0]: b for b in fwd_bars}
    pre_bars = [pre_by_ts[k] for k in sorted(pre_by_ts) if k not in fwd_by_ts]
    fwd_bars = [fwd_by_ts[k] for k in sorted(fwd_by_ts)]

    by_ts = dict(pre_by_ts)
    by_ts.update(fwd_by_ts)
    combined = [by_ts[k] for k in sorted(by_ts)]
    win_bars = combined[-720:] if len(combined) > 720 else combined
    win_warmup_bars = combined[max(0, len(combined) - 760):max(0, len(combined) - len(win_bars))]
    fwd_warmup_bars = pre_bars[-40:]

    full_ohlc = [(b[1], b[2], b[3], b[4]) for b in win_bars]
    full_ts = [b[0] for b in win_bars]
    full_warmup = [(b[1], b[2], b[3], b[4]) for b in win_warmup_bars]
    full_warmup_ts = [b[0] for b in win_warmup_bars]
    fwd_ohlc = [(b[1], b[2], b[3], b[4]) for b in fwd_bars]
    fwd_ts = [b[0] for b in fwd_bars]
    fwd_warmup = [(b[1], b[2], b[3], b[4]) for b in fwd_warmup_bars]
    fwd_warmup_ts = [b[0] for b in fwd_warmup_bars]

    if len(fwd_ohlc) < 2 or len(full_ohlc) < 2:
        print(f"Insufficient forward or full bars for shadow replay (fwd: {len(fwd_ohlc)}, full: {len(full_ohlc)})")
        return {}

    results, (fee_pct, interval) = {}, (0.26, 240)
    if len(fwd_ohlc) + len(fwd_warmup) > 40:
        kraken_replay.prewarm_trend_series(
            fwd_warmup + fwd_ohlc, warmup=len(fwd_warmup), params=LIVE_PARAMS, bar_minutes=interval,
        )
    if len(full_ohlc) + len(full_warmup) > 40:
        kraken_replay.prewarm_trend_series(
            full_warmup + full_ohlc, warmup=len(full_warmup), params=LIVE_PARAMS, bar_minutes=interval,
        )

    for name, overrides in candidates.items():
        params = prepare_spot_interval_params(dataclasses.replace(LIVE_PARAMS, **overrides), interval)
        budget = params.allocated_budget()

        # Forward bars
        m_fwd = kraken_replay.run_replay(
            fwd_ohlc, params, fee_pct=fee_pct, bar_minutes=interval,
            warmup_ohlc=fwd_warmup,
            timestamps=fwd_ts,
            warmup_timestamps=fwd_warmup_ts,
            execution=ExecutionModel(),
            initial_cash=budget,
        )
        s_fwd = _summarize_replay(m_fwd, budget)

        # Trailing 720 bars (~120d)
        m_full = kraken_replay.run_replay(
            full_ohlc, params, fee_pct=fee_pct, bar_minutes=interval,
            warmup_ohlc=full_warmup,
            timestamps=full_ts,
            warmup_timestamps=full_warmup_ts,
            execution=ExecutionModel(),
            initial_cash=budget,
        )
        s_full = _summarize_replay(m_full, budget)

        results[name] = {
            "fwd_net": s_fwd["net_pct"],
            "fwd_total": s_fwd["total_pct"],
            "fwd_dd": s_fwd["max_dd"],
            "fwd_cycles": s_fwd["cycles"],
            "fwd_bars": len(fwd_ohlc),
            "full_net": s_full["net_pct"],
            "full_total": s_full["total_pct"],
            "full_dd": s_full["max_dd"],
            "full_cycles": s_full["cycles"],
            "full_bars": len(full_ohlc),
        }
    return results


def main():
    records = load_dataset(DATASET_CSV)
    print(f"Loaded {len(records)} bars from {DATASET_CSV}")

    candidates = define_candidate_grid()
    print(f"Defined {len(candidates)} candidate configurations to sweep.")

    wf_results = run_walk_forward_evaluation(candidates, records)
    cont_results = run_continuous_628d_replay(candidates, records)
    shadow_results = run_live_forward_shadow(candidates, records=records)

    if not wf_results and not cont_results:
        sys.stderr.write(
            "ERROR: Strategy sweep produced zero walk-forward and continuous results; refusing to write empty reports.\n"
        )
        raise SystemExit(1)

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
    atomic_write_json(json_path, master, indent=2)
    print(f"\nSaved raw JSON to {json_path}")

    sample_shadow = next(iter(shadow_results.values()), {}) if shadow_results else {}
    fwd_bars_cnt = safe_nonneg_int(sample_shadow.get("fwd_bars"), default=0)
    full_bars_cnt = safe_nonneg_int(sample_shadow.get("full_bars"), default=0)
    fwd_days = round(fwd_bars_cnt * 4.0 / 24.0, 1)
    full_days = round(full_bars_cnt * 4.0 / 24.0, 1)

    # Generate Markdown Summary Report
    total_folds = next((safe_nonneg_int(r.get("wf_total_folds"), default=0) for r in master if r.get("wf_total_folds")), 0)
    md_path = out_dir / "SWEEP_REPORT.md"
    lines = [
        "# Strategy Sweep: High Profit & Low Drawdown (628 Days + Live Shadow)\n\n",
        "Evaluation across:\n",
        f"1. **Walk-Forward (WF)**: {total_folds} Out-of-Sample folds (~628 days, HYPE 240m)\n",
        f"2. **Continuous 628d**: Unbroken fixed-allocation replay across all {len(records):,} bars\n",
        f"3. **Kraken Shadow**: {fwd_bars_cnt} forward bars (~{fwd_days}d) & {full_bars_cnt}-bar window (~{full_days}d)\n\n",
        f"## 1. Walk-Forward Out-Of-Sample Performance ({total_folds} Folds)\n\n",
        "| Rank | Candidate | WF Mean % | WF Median % | WF Worst % | WF Worst DD % | Pos Folds | W/T/L vs Live |\n",
        "|---:|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    sorted_wf = sorted(
        master,
        key=lambda x: (safe_float(x.get("wf_mean_ret"), 0.0), -safe_float(x.get("wf_worst_dd"), 100.0)),
        reverse=True,
    )
    for rank, r in enumerate(sorted_wf, 1):
        wf_mean = safe_float(r.get("wf_mean_ret"), 0.0)
        wf_med = safe_float(r.get("wf_median_ret"), 0.0)
        wf_worst = safe_float(r.get("wf_worst_ret"), 0.0)
        wf_wdd = safe_float(r.get("wf_worst_dd"), 0.0)
        wf_pos = safe_nonneg_int(r.get("wf_pos_count"), default=0)
        wf_tf = safe_nonneg_int(r.get("wf_total_folds"), default=total_folds)
        lines.append(
            f"| {rank} | `{r['name']}` | {wf_mean:+.3f}% | {wf_med:+.3f}% | "
            f"{wf_worst:+.2f}% | {wf_wdd:.2f}% | "
            f"{wf_pos}/{wf_tf} | {r.get('vs_live_WTL') or '-'} |\n"
        )

    lines.extend((
        f"\n## 2. Continuous 628-Day Replay ({len(records):,} bars)\n\n",
        "| Rank | Candidate | Net % | Total % | MaxDD % | Calmar Ratio | Cycles | Win Rate |\n",
        "|---:|---|---:|---:|---:|---:|---:|\n",
    ))
    sorted_cont = sorted(
        master,
        key=lambda x: (
            safe_float(x.get("cont_total_pct"), 0.0) > 0.0,
            (
                (
                    math.inf
                    if safe_float(x.get("cont_max_dd"), 0.0) <= 0.0
                    else safe_float(x.get("cont_calmar"), 0.0)
                )
                if safe_float(x.get("cont_total_pct"), 0.0) > 0.0
                else safe_float(x.get("cont_total_pct"), 0.0)
            ),
            safe_float(x.get("cont_total_pct"), 0.0),
            -safe_float(x.get("cont_max_dd"), 100.0),
        ),
        reverse=True,
    )
    for rank, r in enumerate(sorted_cont, 1):
        cycles = safe_nonneg_int(r.get("cont_cycles"), default=0)
        wins = safe_nonneg_int(r.get("cont_wins"), default=0)
        win_rate = (wins / cycles * 100.0) if cycles > 0 else 0.0
        c_net = safe_float(r.get("cont_net_pct"), 0.0)
        c_tot = safe_float(r.get("cont_total_pct"), 0.0)
        c_dd = safe_float(r.get("cont_max_dd"), 0.0)
        c_cal = safe_float(r.get("cont_calmar"), 0.0)
        lines.append(
            f"| {rank} | `{r['name']}` | {c_net:+.2f}% | {c_tot:+.2f}% | "
            f"{c_dd:.2f}% | **{c_cal:.2f}** | {cycles} | {win_rate:.1f}% |\n"
        )

    lines.append(f"\n## 3. Kraken Live Shadow Performance (Forward ~{fwd_days}d & Window ~{full_days}d)\n\n")
    if not shadow_results:
        lines.append("*Live shadow evaluation skipped: insufficient forward bars recorded in `logs/shadow_live/` (minimum 2 bars required).* \n")
    else:
        lines.extend((
            "| Candidate | Forward Net % | Forward Total % | Forward DD % | 120d Net % | 120d Total % | 120d DD % |\n",
            "|---|---:|---:|---:|---:|---:|---:|\n",
        ))
        for r in master:
            f_net = safe_float(r.get("fwd_net"), 0.0)
            f_tot = safe_float(r.get("fwd_total"), 0.0)
            f_dd = safe_float(r.get("fwd_dd"), 0.0)
            w_net = safe_float(r.get("full_net"), 0.0)
            w_tot = safe_float(r.get("full_total"), 0.0)
            w_dd = safe_float(r.get("full_dd"), 0.0)
            lines.append(
                f"| `{r['name']}` | {f_net:+.2f}% | {f_tot:+.2f}% | "
                f"{f_dd:.2f}% | {w_net:+.2f}% | {w_tot:+.2f}% | "
                f"{w_dd:.2f}% |\n"
            )

    write_markdown_report(
        md_path, lines, joiner="", message_prefix="Report written to", write_fn=atomic_write_text
    )


if __name__ == "__main__":
    main()
