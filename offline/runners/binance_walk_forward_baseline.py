#!/usr/bin/env python3
"""Walk-forward baseline for the live Binance spot configuration, without optimization.

Loads the production spot configuration, runs the faithful replay engine from
binance_api/replay.py across rolling time windows without data leakage or
shuffling, and validates metrics against the canonical contract.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from botcore import parse_dotenv
from binance_api.market_data import fetch_ohlc_candles, parse_interval_minutes
from binance_api.replay import make_binance_replay_adapter, StratParams
from strategies.spot_replay import align_spot_interval_params
from offline.backtests.datasets import (
    dataset_metadata,
    load_dataset as load_frozen_dataset,
    save_dataset as save_frozen_dataset,
    validate_dataset,
)
from offline.backtests.evaluation import (
    add_walk_forward_cli_args,
    assess_decision_evidence,
    automatic_window_sizes,
    build_execution_model_from_args,
    evaluate_segment as _evaluate_segment,
    evaluate_walk_forward as _evaluate_walk_forward,
    iso_utc as _iso,
    parse_dataset_specs as _parse_dataset_specs,
    parse_intervals as _parse_intervals,
    resolve_window_sizes,
)
from offline.backtests.execution import ExecutionModel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--config-file", default=str(ROOT / "config.env"))
    parser.add_argument("--symbol", help="defaults to SYMBOL or BINANCE_PAIR (e.g. BTCUSDT)")
    parser.add_argument("--intervals", type=_parse_intervals, default=[60, 240, 1440])
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

    cfg: dict[str, str] = {}
    for path_str in (args.config_file, args.env_file):
        p = Path(path_str)
        if p.exists():
            cfg.update(dict(parse_dotenv(str(p))))
    for k, v in os.environ.items():
        if k.startswith("BINANCE_") or (k.startswith("STRAT_") and k not in cfg):
            cfg[k] = v

    symbol = (args.symbol or cfg.get("SYMBOL") or cfg.get("BINANCE_PAIR") or "BTCUSDT").strip().upper()

    try:
        strat_params = StratParams.from_env(cfg)
    except Exception:
        # If .env does not specify every parameter, fall back to safe default test parameters
        strat_params = StratParams(
            currency="USDT",
            entry_amount=float(cfg.get("STRAT_ENTRY", "100.0")),
            entry_discount_pct=float(cfg.get("STRAT_ENTRY_DISCOUNT_PCT", "0.2")),
            dca_amount=float(cfg.get("STRAT_DCA", "50.0")),
            dca_drop_pct=float(cfg.get("STRAT_DCA_DROP_PCT", "2.0")),
            takeprofit_pct=float(cfg.get("STRAT_TAKEPROFIT_PCT", "2.0")),
            max_budget=float(cfg.get("STRAT_MAX_BUDGET", "500.0")),
            max_dca_buys=int(float(cfg.get("STRAT_MAX_DCA_BUYS", "5"))),
            stop_loss_pct=float(cfg.get("STRAT_STOP_LOSS_PCT", "0.0")),
            check_minutes=1.0,
            enable_takeprofit=True,
            order_ttl_min=60.0,
        )

    execution = build_execution_model_from_args(parser, args)
    supplied_datasets = _parse_dataset_specs(args.dataset)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = output_dir / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    summary_by_interval = {}

    for interval in args.intervals:
        frozen_path = supplied_datasets.get(interval)
        source_label = "frozen_csv"
        if frozen_path and frozen_path.exists():
            records = load_frozen_dataset(frozen_path)
        else:
            source_label = "binance_public_klines"
            records = fetch_ohlc_candles(
                symbol,
                interval=interval,
                lookback_days=180,
                drop_incomplete=True,
            )
            frozen_path = dataset_dir / f"{symbol}_{interval}m.csv"
            save_frozen_dataset(records, frozen_path)

        if len(records) < 20:
            print(f"[!] Insufficient bars for {symbol} {interval}m ({len(records)} < 20); skipping.")
            continue

        validate_dataset(records, interval_minutes=interval)
        train_size, val_size, test_size, step_size = resolve_window_sizes(args, len(records))

        adapter = make_binance_replay_adapter(
            params=align_spot_interval_params(strat_params, interval),
            fee_pct=args.fee,
            bar_minutes=float(interval),
            execution=execution,
        )

        wf_result = _evaluate_walk_forward(
            records,
            adapter,
            train_size=train_size,
            validation_size=val_size,
            test_size=test_size,
            step_size=step_size,
            warmup_bars=args.warmup,
            validate_contract=True,
        )

        evidence = assess_decision_evidence(records, wf_result["aggregate_test"])
        generated_at = dt.datetime.now(tz=dt.timezone.utc)
        meta = dataset_metadata(records, interval_minutes=interval)

        report = {
            "schema_version": 1,
            "generated_at_utc": generated_at.isoformat(),
            "purpose": "Binance live-engine fixed-config walk-forward baseline",
            "runtime": {
                "symbol": symbol,
                "interval_minutes": interval,
            },
            "strategy_params": dataclasses.asdict(strat_params),
            "fee_pct_per_leg": args.fee,
            "execution_model": dataclasses.asdict(execution),
            "dataset": {
                "source": source_label,
                "frozen_file": str(frozen_path),
                **meta,
            },
            "windows": {
                "train": train_size,
                "validation": val_size,
                "test": test_size,
                "step": step_size,
            },
            "evidence": evidence,
            "aggregate_test": wf_result["aggregate_test"],
            "folds": wf_result["folds"],
        }

        report_path = output_dir / f"{symbol}_{interval}m_baseline.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[{symbol} {interval}m] Verdict: {evidence['verdict']} | "
              f"Worst Return: {wf_result['aggregate_test']['worst_return_pct']:+.2f}% | "
              f"Folds: {wf_result['aggregate_test']['fold_count']} -> {report_path.name}")
        summary_by_interval[interval] = wf_result["aggregate_test"]

    return 0


if __name__ == "__main__":
    sys.exit(main())
