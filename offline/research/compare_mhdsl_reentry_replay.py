#!/usr/bin/env python3
"""Compare MHDSL and baseline spot DCA replay on HYPE spot dataset."""

import csv
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pathlib import Path
from strategies import spot_engine as strat
from strategies.spot_replay import run_replay
from offline.backtests.datasets import load_dataset, validate_dataset
from offline.backtests.evaluation import to_ohlc
from botcore import safe_float, safe_int, safe_nonneg_int

_safe_f = safe_float
_safe_int = safe_int


def run_comparison(csv_path=None):
    csv_file = csv_path or os.path.join(ROOT, "offline", "research", "hype_dataset", "HYPEUSDC_240m_hlspot.csv")
    if not csv_file or isinstance(csv_file, (bool, int, float)) or not os.path.exists(str(csv_file)):
        print(f"Dataset not found at {csv_file}")
        return []

    p = Path(csv_file)
    if not p.is_file():
        print(f"Dataset not found at {csv_file}")
        return []

    try:
        records = validate_dataset(load_dataset(p), interval_minutes=240)
        bars = to_ohlc(records)
    except Exception as exc:
        print(f"Error loading dataset {csv_file}: {exc}")
        return []

    if not bars:
        print(f"No valid bars in {csv_file}")
        return []

    print(f"Loaded {len(bars)} historical bars (HYPE spot 4h).")
    sl_levels = [6.0, 8.0, 10.0, 12.0]
    results = []

    for sl in sl_levels:
        common_reentry = dict(
            adopt_cost=0.0, adopt_qty=0.0, reentry_drop_pct=2.0, reentry_tolerance_pct=0.05,
            reentry_adaptive=False, reentry_sl_bounce_pct=1.5, tp_tranches=[],
            reentry_hybrid_enabled=True, reentry_pullback_pct=2.0, reentry_ttl_hours=48.0,
        )
        p_base = strat.StratParams(
            currency="USD", entry_amount=350.0, entry_discount_pct=0.2, dca_amount=100.0,
            dca_drop_pct=2.0, check_minutes=2.0, takeprofit_pct=5.0, max_budget=1050.0,
            max_dca_buys=7, enable_takeprofit=True, order_ttl_min=10.0, stop_loss_pct=sl,
            sl_mhdsl_enabled=False,
            **common_reentry,
        )
        p_prop = strat.StratParams(
            currency="USD", entry_amount=350.0, entry_discount_pct=0.2, dca_amount=100.0,
            dca_drop_pct=2.0, check_minutes=2.0, takeprofit_pct=5.0, max_budget=1050.0,
            max_dca_buys=7, enable_takeprofit=True, order_ttl_min=10.0, stop_loss_pct=sl,
            sl_mhdsl_enabled=True,
            sl_disaster_buffer_ratio=0.30,
            sl_disaster_vol_multiplier=0.85,
            sl_disaster_buffer_min_pct=0.80,
            sl_trend_defer_max_sec=220.0,
            sl_bounce_min_recovery_pct=0.40,
            sl_bounce_pullback_pct=0.30,
            sl_bounce_ratchet_enabled=True,
            sl_structural_break_enabled=True,
            **common_reentry,
        )

        res_base = run_replay(bars, p_base, fee_pct=0.26, bar_minutes=240)
        res_prop = run_replay(bars, p_prop, fee_pct=0.26, bar_minutes=240)
        results.append((sl, res_base, res_prop))

    print("\n" + "=" * 92)
    print(f"{'SL %':<6} | {'CONFIG':<10} | {'NET PnL ($)':<12} | {'CYCLES':<8} | {'WINS':<6} | {'WIN %':<8} | {'MAX DD ($)':<12} | {'FILLS':<8}")
    print("=" * 92)

    for sl, base, prop in results:
        b_cyc = safe_nonneg_int(base.get('cycles'), default=0)
        p_cyc = safe_nonneg_int(prop.get('cycles'), default=0)
        b_wins = safe_nonneg_int(base.get('wins'), default=0)
        p_wins = safe_nonneg_int(prop.get('wins'), default=0)
        b_net = safe_float(base.get('net'), 0.0)
        p_net = safe_float(prop.get('net'), 0.0)
        b_dd = safe_float(base.get('maxdd'), 0.0)
        p_dd = safe_float(prop.get('maxdd'), 0.0)
        b_fills = safe_nonneg_int(base.get('fills'), default=0)
        p_fills = safe_nonneg_int(prop.get('fills'), default=0)
        base_win_pct = (b_wins / b_cyc * 100) if b_cyc > 0 else 0.0
        prop_win_pct = (p_wins / p_cyc * 100) if p_cyc > 0 else 0.0

        print(f"{sl:<6.1f} | {'Baseline':<10} | {b_net:<12.2f} | {b_cyc:<8} | {b_wins:<6} | {base_win_pct:<8.1f} | {b_dd:<12.2f} | {b_fills:<8}")
        print(f"{'':<6} | {'Proposed':<10} | {p_net:<12.2f} | {p_cyc:<8} | {p_wins:<6} | {prop_win_pct:<8.1f} | {p_dd:<12.2f} | {p_fills:<8}")
        diff_net = p_net - b_net
        diff_dd = p_dd - b_dd
        print(f"{'':<6} | {'Delta':<10} | {'+' if diff_net >= 0 else ''}{diff_net:<12.2f} | {p_cyc - b_cyc:<8} | {p_wins - b_wins:<6} | {prop_win_pct - base_win_pct:<+8.1f} | {diff_dd:<+12.2f} | {p_fills - b_fills:<8}")
        print("-" * 92)

    return results


def main():
    has_orig_log = hasattr(strat, "log")
    orig_log = getattr(strat, "log", None)
    strat.log = lambda *a, **k: None
    try:
        run_comparison()
    finally:
        if has_orig_log:
            strat.log = orig_log
        elif hasattr(strat, "log"):
            delattr(strat, "log")


if __name__ == "__main__":
    main()