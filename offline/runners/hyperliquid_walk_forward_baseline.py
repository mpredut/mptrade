#!/usr/bin/env python3
"""Walk-forward baseline for the live Hyperliquid perpetual configuration, without optimization.

Loads the production configuration (hyperliquid/config.env), runs the faithful replay
engine from hyperliquid/replay.py across rolling time windows without data leakage or
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
HL_DIR = ROOT / "hyperliquid"
sys.path.insert(0, str(HL_DIR))
sys.path.insert(0, str(ROOT))

from botcore import parse_dotenv
import importlib.util

_md_spec = importlib.util.spec_from_file_location(
    "hl_wf_md", str(HL_DIR / "market_data.py")
)
_hl_md = importlib.util.module_from_spec(_md_spec)
_md_spec.loader.exec_module(_hl_md)
fetch_ohlc_candles = _hl_md.fetch_ohlc_candles
fetch_funding_history = _hl_md.fetch_funding_history

_replay_spec = importlib.util.spec_from_file_location(
    "hl_wf_replay", str(HL_DIR / "replay.py")
)
_hl_replay = importlib.util.module_from_spec(_replay_spec)
_replay_spec.loader.exec_module(_hl_replay)
make_hyperliquid_replay_adapter = _hl_replay.make_hyperliquid_replay_adapter
StratParams = _hl_replay.StratParams
from strategies.spot_engine import StratParams as SpotStratParams
from strategies.spot_replay import align_spot_interval_params

from offline.backtests.datasets import (
    dataset_metadata,
    dataset_sha256,
    load_dataset as load_frozen_dataset,
    save_dataset as save_frozen_dataset,
    validate_dataset,
)
from offline.backtests.evaluation import (
    add_walk_forward_cli_args,
    assess_decision_evidence,
    automatic_window_sizes,
    build_execution_model_from_args,
    evaluate_walk_forward as _evaluate_walk_forward,
    iso_utc as _iso,
    parse_dataset_specs as _parse_dataset_specs,
    parse_intervals as _parse_intervals,
    resolve_window_sizes,
)
from offline.backtests.execution import ExecutionModel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=str(HL_DIR / ".env"))
    parser.add_argument("--config-file", default=str(HL_DIR / "config.env"))
    parser.add_argument("--coin", help="defaults to HL_COIN from live config")
    parser.add_argument("--direction", choices=["long", "short"], help="defaults to HL_DIRECTION")
    parser.add_argument(
        "--engine", choices=["spot", "perp"], default="spot",
        help="replay engine: 'spot' (default, matches live hl_bot.py spot engine) or 'perp' (legacy perpetuals engine)",
    )
    parser.add_argument("--intervals", type=_parse_intervals, default=[60, 240, 1440])
    parser.add_argument("--fee", type=float, default=None, help="fee per leg (pct), defaults to HL_FEE_PCT")
    add_walk_forward_cli_args(parser, include_warmup=True, default_warmup=0)
    parser.add_argument(
        "--dataset", action="append", default=[], metavar="INTERVAL=PATH",
        help="frozen dataset CSV path for an interval (e.g. 60=path/to/candles.csv)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "offline/results/hyperliquid_walk_forward",
    )
    args = parser.parse_args(argv)

    cfg: dict[str, str] = {}
    for path_str in (args.config_file, args.env_file):
        p = Path(path_str)
        if p.exists():
            cfg.update(dict(parse_dotenv(str(p))))
    for k, v in os.environ.items():
        if k.startswith("HL_") or (k.startswith("STRAT_") and k not in cfg):
            cfg[k] = v

    coin = (args.coin or cfg.get("HL_COIN", "HYPE")).strip().upper()
    direction = (args.direction or cfg.get("HL_DIRECTION", "long")).strip().lower()
    fee_pct = args.fee if args.fee is not None else float(cfg.get("HL_FEE_PCT", "0.035"))
    if fee_pct < 0:
        parser.error("--fee cannot be negative")

    if args.engine == "spot":
        strat_params = SpotStratParams.from_env(env=cfg)
    else:
        entry = float(cfg.get("STRAT_ENTRY", "350"))
        dca = float(cfg.get("STRAT_DCA", "100"))
        disc = float(cfg.get("STRAT_ENTRY_DISCOUNT_PCT", "0.2"))
        drop = float(cfg.get("STRAT_DCA_DROP_PCT", "2.0"))
        tp = float(cfg.get("STRAT_TAKEPROFIT_PCT", "5.0"))
        budget = float(cfg.get("STRAT_MAX_BUDGET", "1050"))
        maxdca = int(float(cfg.get("STRAT_MAX_DCA_BUYS", "7")))
        sl = float(cfg.get("STRAT_STOP_LOSS_PCT", "0.0"))

        strat_params = StratParams(
            currency="USDC",
            direction=direction,
            entry_amount=entry,
            entry_discount_pct=disc,
            dca_amount=dca,
            dca_drop_pct=drop,
            takeprofit_pct=tp,
            max_budget=budget,
            max_dca_buys=maxdca,
            stop_loss_pct=sl,
            fee_pct=fee_pct,
        )

    execution = build_execution_model_from_args(parser, args)
    dataset_map = _parse_dataset_specs(args.dataset)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_by_interval = {}
    for interval in args.intervals:
        frozen_file = dataset_map.get(interval)
        if frozen_file:
            frozen_path = frozen_file
            records = validate_dataset(load_frozen_dataset(frozen_file), interval_minutes=interval)
            source_label = f"frozen file: {frozen_file}"
        else:
            raw_candles = fetch_ohlc_candles(coin, interval, drop_incomplete=True)
            if not raw_candles:
                print(f"[!] No candles returned for {coin} {interval}m; skipping.")
                continue
            records = validate_dataset(raw_candles, interval_minutes=interval)
            meta = dataset_metadata(records, interval_minutes=interval)
            frozen_path = output_dir / "datasets" / f"{coin}_{interval}m_{meta['sha256'][:12]}.csv"
            save_frozen_dataset(records, frozen_path)
            source_label = f"Hyperliquid API, frozen to {frozen_path}"

        if len(records) < 20:
            print(f"[!] Insufficient bars for {coin} {interval}m ({len(records)} < 20); skipping.")
            continue

        train_size, val_size, test_size, step_size = resolve_window_sizes(args, len(records))

        if args.engine == "spot":
            adapter = make_hyperliquid_replay_adapter(
                params=align_spot_interval_params(strat_params, interval),
                bar_minutes=float(interval),
                execution=execution,
                fee_pct=fee_pct,
                engine="spot",
            )
        else:
            adapter = make_hyperliquid_replay_adapter(
                params=strat_params,
                direction=direction,
                coin=coin,
                bar_minutes=float(interval),
                execution=execution,
                fee_pct=fee_pct,
                engine="perp",
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
            "purpose": f"Hyperliquid live-{args.engine}-engine fixed-config walk-forward baseline",
            "runtime": {
                "coin": coin,
                "direction": direction,
                "interval_minutes": interval,
                "engine": args.engine,
            },
            "strategy_params": dataclasses.asdict(strat_params),
            "fee_pct_per_leg": fee_pct,
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
                "signal_warmup_bars": args.warmup,
            },
            "evidence": evidence,
            "aggregate_test": wf_result["aggregate_test"],
            "folds": wf_result["folds"],
        }

        report_path = output_dir / f"{coin}_{interval}m_baseline.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[{coin} {interval}m] Verdict: {evidence['verdict']} | "
              f"Worst Return: {wf_result['aggregate_test']['worst_return_pct']:+.2f}% | "
              f"Folds: {wf_result['aggregate_test']['fold_count']} -> {report_path.name}")
        summary_by_interval[interval] = wf_result["aggregate_test"]

    return 0


if __name__ == "__main__":
    sys.exit(main())
