#!/usr/bin/env python3
"""Compare MHDSL and baseline spot DCA replay on HYPE spot dataset."""

import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from strategies import spot_engine as strat
from strategies.spot_replay import run_replay


def run_comparison(csv_path=None):
    csv_file = csv_path or os.path.join(ROOT, "offline", "research", "hype_dataset", "HYPEUSDC_240m_hlspot.csv")
    if not os.path.exists(csv_file):
        print(f"Dataset not found at {csv_file}")
        return []

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        bars = [(float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])) for r in reader]

    print(f"Loaded {len(bars)} historical bars (HYPE spot 4h).")
    sl_levels = [6.0, 8.0, 10.0, 12.0]
    results = []

    for sl in sl_levels:
        p_base = strat.StratParams(
            currency="USD", entry_amount=350.0, entry_discount_pct=0.2, dca_amount=100.0,
            dca_drop_pct=2.0, check_minutes=2.0, takeprofit_pct=5.0, max_budget=1050.0,
            max_dca_buys=7, enable_takeprofit=True, order_ttl_min=10.0, stop_loss_pct=sl,
            sl_mhdsl_enabled=False,
            adopt_cost=0.0, adopt_qty=0.0, reentry_drop_pct=2.0, reentry_tolerance_pct=0.05,
            reentry_adaptive=False, reentry_sl_bounce_pct=1.5, tp_tranches=[]
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
            adopt_cost=0.0, adopt_qty=0.0, reentry_drop_pct=2.0, reentry_tolerance_pct=0.05,
            reentry_adaptive=False, reentry_sl_bounce_pct=1.5, tp_tranches=[]
        )

        res_base = run_replay(bars, p_base, fee_pct=0.26, bar_minutes=240)
        res_prop = run_replay(bars, p_prop, fee_pct=0.26, bar_minutes=240)
        results.append((sl, res_base, res_prop))

    print("\n" + "=" * 92)
    print(f"{'SL %':<6} | {'CONFIG':<10} | {'NET PnL ($)':<12} | {'CYCLES':<8} | {'WINS':<6} | {'WIN %':<8} | {'MAX DD ($)':<12} | {'FILLS':<8}")
    print("=" * 92)

    for sl, base, prop in results:
        base_win_pct = (base['wins'] / base['cycles'] * 100) if base.get('cycles', 0) > 0 else 0
        prop_win_pct = (prop['wins'] / prop['cycles'] * 100) if prop.get('cycles', 0) > 0 else 0

        print(f"{sl:<6.1f} | {'Baseline':<10} | {base.get('net', 0.0):<12.2f} | {base.get('cycles', 0):<8} | {base.get('wins', 0):<6} | {base_win_pct:<8.1f} | {base.get('maxdd', 0.0):<12.2f} | {base.get('fills', 0):<8}")
        print(f"{'':<6} | {'Proposed':<10} | {prop.get('net', 0.0):<12.2f} | {prop.get('cycles', 0):<8} | {prop.get('wins', 0):<6} | {prop_win_pct:<8.1f} | {prop.get('maxdd', 0.0):<12.2f} | {prop.get('fills', 0):<8}")
        diff_net = prop.get('net', 0.0) - base.get('net', 0.0)
        diff_dd = prop.get('maxdd', 0.0) - base.get('maxdd', 0.0)
        print(f"{'':<6} | {'Delta':<10} | {'+' if diff_net >= 0 else ''}{diff_net:<12.2f} | {prop.get('cycles', 0)-base.get('cycles', 0):<8} | {prop.get('wins', 0)-base.get('wins', 0):<6} | {prop_win_pct-base_win_pct:<+8.1f} | {diff_dd:<+12.2f} | {prop.get('fills', 0)-base.get('fills', 0):<8}")
        print("-" * 92)

    return results


def main():
    orig_log = getattr(strat, "log", None)
    strat.log = lambda *a, **k: None
    try:
        run_comparison()
    finally:
        if orig_log is not None:
            strat.log = orig_log


if __name__ == "__main__":
    main()