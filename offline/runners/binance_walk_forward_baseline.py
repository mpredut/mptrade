#!/usr/bin/env python3
"""Walk-forward baseline for the live Binance spot configuration, without optimization.

Loads the production spot configuration, runs the faithful replay engine from
binance_api/replay.py across rolling time windows without data leakage or
shuffling, and validates metrics against the canonical contract.
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from botcore import safe_nonneg_float
from binance_api.market_data import fetch_ohlc_candles
from strategies.spot_replay import StratParams, align_spot_interval_params, make_spot_replay_adapter
from offline.backtests.datasets import (
    load_dataset,
    save_dataset,
    validate_dataset,
)
from offline.backtests.evaluation import (
    add_walk_forward_cli_args,
    build_and_save_interval_baseline_report,
    build_execution_model_from_args,
    extract_cfg_float,
    extract_cfg_int,
    load_layered_env_config,
    parse_dataset_specs,
    parse_intervals,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--config-file", default=str(ROOT / "config.env"))
    parser.add_argument("--symbol", help="defaults to SYMBOL or BINANCE_PAIR (e.g. BTCUSDT)")
    parser.add_argument("--intervals", type=parse_intervals, default=[60, 240, 1440])
    parser.add_argument("--fee", type=float, default=0.10, help="fee per leg (pct), defaults to 0.10%%")
    add_walk_forward_cli_args(parser, include_warmup=True, default_warmup=0)
    parser.add_argument(
        "--dataset", action="append", default=[], metavar="INTERVAL=PATH",
        help="frozen dataset CSV path for an interval (e.g. 60=path/to/candles.csv)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "offline/results/binance_walk_forward",
    )
    args = parser.parse_args(argv)
    if safe_nonneg_float(args.fee, default=None) is None:
        parser.error("--fee cannot be negative")

    cfg = load_layered_env_config(args.config_file, args.env_file, override_prefixes=("BINANCE_",))

    symbol = (args.symbol or cfg.get("SYMBOL") or cfg.get("BINANCE_PAIR") or "BTCUSDT").strip().upper()

    try:
        strat_params = StratParams.from_env(cfg)
    except Exception:
        # If .env does not specify every parameter, fall back to safe default test parameters
        strat_params = StratParams.from_dict({
            "currency": "USDT",
            "entry_amount": extract_cfg_float(cfg, "STRAT_ENTRY", 100.0),
            "entry_discount_pct": extract_cfg_float(cfg, "STRAT_ENTRY_DISCOUNT_PCT", 0.2),
            "dca_amount": extract_cfg_float(cfg, "STRAT_DCA", 50.0),
            "dca_drop_pct": extract_cfg_float(cfg, "STRAT_DCA_DROP_PCT", 2.0),
            "takeprofit_pct": extract_cfg_float(cfg, "STRAT_TAKEPROFIT_PCT", 2.0),
            "max_budget": extract_cfg_float(cfg, "STRAT_MAX_BUDGET", 500.0),
            "max_dca_buys": max(0, extract_cfg_int(cfg, "STRAT_MAX_DCA_BUYS", 5)),
            "stop_loss_pct": extract_cfg_float(cfg, "STRAT_STOP_LOSS_PCT", 0.0),
            "check_minutes": 1.0,
            "enable_takeprofit": True,
            "order_ttl_min": 60.0,
        })

    execution = build_execution_model_from_args(parser, args)
    supplied_datasets = parse_dataset_specs(args.dataset)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = output_dir / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    summary_by_interval = {}

    for interval in args.intervals:
        frozen_path = supplied_datasets.get(interval)
        source_label = "frozen_csv"
        if frozen_path is not None:
            records = load_dataset(frozen_path)
            if len(records) < 20:
                print(f"[!] Insufficient bars for {symbol} {interval}m ({len(records)} < 20); skipping.")
                continue
            records = validate_dataset(records, interval_minutes=interval)
        else:
            source_label = "binance_public_klines"
            raw_records = fetch_ohlc_candles(
                symbol,
                interval=interval,
                lookback_days=180,
                drop_incomplete=True,
            )
            if not raw_records or len(raw_records) < 20:
                print(f"[!] Insufficient bars for {symbol} {interval}m ({len(raw_records or ())} < 20); skipping.")
                continue
            records = validate_dataset(raw_records, interval_minutes=interval)
            frozen_path = dataset_dir / f"{symbol}_{interval}m.csv"
            save_dataset(records, frozen_path)
        adapter = make_spot_replay_adapter(
            params=align_spot_interval_params(strat_params, interval),
            fee_pct=args.fee,
            bar_minutes=float(interval),
            execution=execution,
        )
        summary_by_interval[interval] = build_and_save_interval_baseline_report(
            records,
            adapter,
            args=args,
            interval=interval,
            symbol_or_coin=symbol,
            purpose="Binance live-engine fixed-config walk-forward baseline",
            runtime={"symbol": symbol, "interval_minutes": interval},
            strategy_params=dataclasses.asdict(strat_params),
            fee_pct=args.fee,
            execution=execution,
            source_label=source_label,
            frozen_path=frozen_path,
            output_dir=output_dir,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
