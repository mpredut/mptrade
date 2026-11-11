#!/usr/bin/env python3
"""Run Tests 0 to 4 ablation across Trading 212 US equities."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
T212_DIR = os.path.join(ROOT, "212trading")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from botcore import parse_dotenv, safe_pos_float
from offline.backtests.evaluation import load_venue_replay_module
from offline.research.hybrid_reentry.harness import (
    evaluate_asset,
    load_ohlc_with_timestamps_csv,
    print_ablation_suite_banner,
    save_validated_ablation_results,
)
from strategies.spot_replay import align_spot_interval_params, scale_subbar_timeout_params

OUT_DIR = os.path.join(ROOT, "offline", "results", "hybrid_reentry")

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
    "STRAT_REENTRY_PULLBACK_ADAPTIVE": "false",
    "STRAT_REENTRY_PULLBACK_K": "1.0",
    "STRAT_REENTRY_PULLBACK_MIN": "0.8",
    "STRAT_REENTRY_PULLBACK_MAX": "4.0",
    "STRAT_REENTRY_TTL_HOURS": "0.0",
    "STRAT_SL_MHDSL_ENABLED": "true",
    "STRAT_SL_DISASTER_BUFFER_RATIO": "0.30",
    "STRAT_SL_DISASTER_BUFFER_MAX_PCT": "15.0",
    "STRAT_SL_TREND_DEFER_MAX_SEC": "220.0",
    "STRAT_SL_BOUNCE_MIN_RECOVERY_PCT": "0.40",
    "STRAT_SL_BOUNCE_PULLBACK_PCT": "0.30",
    "STRAT_SL_STRUCTURAL_BREAK_ENABLED": "true",
    "STRAT_TP_MDHTP_ENABLED": "true",
    "STRAT_TP_STAGNATION_MAX_SEC": "1080.0",
}


def _import_t212():
    rp = load_venue_replay_module("t212", "t212_replay_equities")
    return rp._strat, rp


def main():
    print_ablation_suite_banner(
        OUT_DIR, "TRADING 212 EQUITIES HYBRID RE-ENTRY ABLATION SUITE (Tests 0 to 4)"
    )
    _strat, rp = _import_t212()
    results = []
    
    for symbol, csv_path in DATASETS.items():
        if not os.path.isfile(csv_path):
            print(f"Skipping {symbol}: not found at {csv_path}")
            continue
        ohlc, timestamps = load_ohlc_with_timestamps_csv(csv_path)
        if not ohlc:
            print(f"Skipping {symbol}: empty OHLC data.")
            continue
        print(f"Evaluating {symbol} ({len(ohlc)} daily bars)...", flush=True)

        profile_path = os.path.join(T212_DIR, f"config.{symbol.lower()}.env")
        if not os.path.isfile(profile_path):
            raise FileNotFoundError(f"Missing T212 profile configuration for {symbol}: {profile_path}")
        cfg = dict(BASE_CONFIG)
        cfg.update(parse_dotenv(profile_path))
        cfg["YAHOO_SYMBOL"] = symbol
        cfg["STRAT_CURRENCY"] = "USD"
        cfg["STRAT_DCA_TREND_GATE_PCT"] = "0"
        cfg["STRAT_MAX_BUDGET"] = "500"
        cfg["STRAT_REENTRY_HYBRID_ENABLED"] = "false"
        cfg["STRAT_REENTRY_PEAK_RELATIVE"] = "false"
        cfg["STRAT_REENTRY_PULLBACK_ADAPTIVE"] = "false"
        cfg["STRAT_REENTRY_PULLBACK_PCT"] = "1.5"
        cfg["STRAT_REENTRY_TTL_HOURS"] = "0.0"
        if safe_pos_float(cfg.get("STRAT_REENTRY_DROP_PCT"), default=None) is None:
            cfg["STRAT_REENTRY_DROP_PCT"] = "2.0"
        base_params = scale_subbar_timeout_params(
            align_spot_interval_params(_strat.StratParams.from_env(cfg), 1440.0), 1440.0
        )
        asset_rows = evaluate_asset(
            asset_name=symbol,
            ohlc=ohlc,
            base_params=base_params,
            bar_minutes=1440.0,
            initial_cash=500.0,
            replay_fn=rp.run_replay,
            timestamps=timestamps or None,
        )
        if not asset_rows:
            print(f"Skipping {symbol}: insufficient bars after warmup.")
            continue
        for row in asset_rows:
            results.append(row)
            if "error" in row:
                print(f"  {row['test_id']} ({row['variant']:<18}): ERROR {row['error']}")
            else:
                print(
                    f"  {row['test_id']} ({row['variant']:<18}): Return {row['total_return_pct']:+6.2f}% "
                    f"| MaxDD {row['max_drawdown_pct']:5.2f}% | Exposure {row['exposure_pct']:5.2f}% "
                    f"| Cycles {row['cycles']}"
                )

    save_validated_ablation_results(
        results,
        os.path.join(OUT_DIR, "equities_ablation_results.json"),
        empty_error_msg="ERROR: No valid equity ablation results produced; refusing to overwrite report JSON.",
        saved_label="Equities results",
    )


if __name__ == "__main__":
    main()
