#!/usr/bin/env python3
"""Walk-forward baseline for the live Kraken configuration, without optimisation.

It loads the same configuration as the bot (.env then config.env), runs the faithful
engine from kraken/replay.py over time windows without shuffling, and keeps the
exact dataset used together with its hash. Each segment starts with a clean
state; TEST is the out-of-sample report, TRAIN/VALIDATION are only regime
diagnostics and choose no parameters.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
KRAKEN_DIR = ROOT / "kraken"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(KRAKEN_DIR) not in sys.path:
    sys.path.append(str(KRAKEN_DIR))

from botcore import _parse_bool_flag, safe_nonneg_float  # noqa: E402
from state_io import atomic_write_json  # noqa: E402
from kraken import market_data as kraken_market_data  # noqa: E402
from kraken import replay as kraken_replay  # noqa: E402
from strategies.spot_engine import StratParams  # noqa: E402
from strategies.spot_replay import align_spot_interval_params  # noqa: E402
from offline.backtests.datasets import (  # noqa: E402
    dataset_metadata,
    load_dataset as load_frozen_dataset,
    save_dataset as save_frozen_dataset,
    validate_dataset,
)
from offline.backtests.evaluation import (  # noqa: E402
    add_walk_forward_cli_args,
    build_execution_model_from_args,
    evaluate_segment as _evaluate_segment,
    evaluate_walk_forward as _evaluate_walk_forward,
    load_layered_env_config,
    parse_dataset_specs as _parse_dataset_specs,
    parse_intervals as _parse_intervals,
    resolve_window_sizes,
)
from offline.backtests.execution import ExecutionModel  # noqa: E402


def fetch_closed_candles(pair: str, interval: int) -> list[dict]:
    """Read at most 720 Kraken bars and drop the last, still-forming bar."""
    return kraken_market_data.fetch_ohlc_candles(pair, interval, drop_incomplete=True)


def _segment_result(records: list[dict], params: StratParams, fee_pct: float,
                    interval: int, warmup_records: list[dict] | tuple = (),
                    execution: ExecutionModel | None = None) -> dict:
    adapter = kraken_replay.make_kraken_replay_adapter(
        align_spot_interval_params(params, interval),
        fee_pct=fee_pct,
        bar_minutes=interval,
        execution=execution,
    )
    return _evaluate_segment(
        records,
        adapter,
        warmup_records=warmup_records,
    )


def evaluate_walk_forward(records: list[dict], params: StratParams, *, fee_pct: float,
                          interval: int, train_size: int, validation_size: int,
                          test_size: int, step_size: int, warmup_bars: int = 0,
                          execution: ExecutionModel | None = None) -> dict:
    adapter = kraken_replay.make_kraken_replay_adapter(
        align_spot_interval_params(params, interval),
        fee_pct=fee_pct,
        bar_minutes=interval,
        execution=execution,
    )
    return _evaluate_walk_forward(
        records,
        adapter,
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
        step_size=step_size,
        warmup_bars=warmup_bars,
        validate_contract=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=str(KRAKEN_DIR / ".env"))
    parser.add_argument("--config-file", default=str(KRAKEN_DIR / "config.env"))
    parser.add_argument("--pair", help="defaults to KRAKEN_PAIR from the live configuration")
    parser.add_argument("--intervals", type=_parse_intervals, default=[60, 240, 1440])
    parser.add_argument("--fee", type=float, default=0.26, help="fee per leg, procente")
    add_walk_forward_cli_args(parser, include_warmup=True, default_warmup=0)
    parser.add_argument(
        "--dataset", action="append", default=[], metavar="INTERVAL=CALE",
        help="use a frozen CSV instead of the API; repeatable for every interval",
    )
    parser.add_argument(
        "--output-dir", default=str(ROOT / "offline" / "results" / "kraken_walk_forward")
    )
    args = parser.parse_args(argv)

    if safe_nonneg_float(args.fee, default=None) is None:
        parser.error("--fee cannot be negative")
    execution = build_execution_model_from_args(parser, args)

    # The same order as kraken_bot.py: .env takes priority, config.env fills in the rest.
    cfg = load_layered_env_config(args.config_file, args.env_file, override_prefixes=("KRAKEN_",))
    pair = (args.pair or cfg.get("KRAKEN_PAIR") or "").strip().upper()
    if not pair:
        parser.error("missing pair: --pair or KRAKEN_PAIR")
    params = StratParams.from_env(cfg)
    supplied = _parse_dataset_specs(args.dataset)
    output_dir = Path(args.output_dir).expanduser().resolve()
    dataset_dir = output_dir / "datasets"
    generated_at = dt.datetime.now(tz=dt.timezone.utc)

    report = {
        "schema_version": 1,
        "generated_at_utc": generated_at.isoformat(),
        "purpose": "fixed-live-config walk-forward baseline; no parameter selection",
        "runtime": {
            "pair": pair,
            "strategy_mode": cfg.get("STRATEGY_MODE", "avg_tp"),
            "execute_live": _parse_bool_flag(cfg.get("STRAT_EXECUTE"), False),
        },
        "strategy_params": dataclasses.asdict(params),
        "fee_pct_per_leg": args.fee,
        "execution_model": dataclasses.asdict(execution),
        "walk_forward": {
            "requested_sizes": (
                {"train": args.train, "validation": args.validation,
                 "test": args.test, "step": args.step}
                if args.train is not None else "auto: 45%/10%/15%, step=test"
            ),
            "shuffle": False,
            "segment_state": "reset",
            "signal_warmup_bars": args.warmup,
            "selection": "none; current live config is held fixed",
        },
        "intervals": {},
    }

    for interval in args.intervals:
        source = supplied.get(interval)
        records = load_frozen_dataset(source) if source else fetch_closed_candles(pair, interval)
        if not records:
            raise RuntimeError(f"an empty dataset for {pair} {interval}m")
        records = validate_dataset(records, interval_minutes=interval)
        meta = dataset_metadata(records, interval_minutes=interval)
        digest = meta["sha256"]
        frozen_path = dataset_dir / f"{pair}_{interval}m_{digest[:12]}.csv"
        save_frozen_dataset(records, frozen_path)
        train_size, validation_size, test_size, step_size = resolve_window_sizes(
            args, len(records)
        )
        result = evaluate_walk_forward(
            records, params, fee_pct=args.fee, interval=interval,
            train_size=train_size, validation_size=validation_size,
            test_size=test_size, step_size=step_size, warmup_bars=args.warmup,
            execution=execution,
        )
        report["intervals"][str(interval)] = {
            "dataset": {
                "source": str(source) if source else "Kraken public OHLC",
                "frozen_file": str(frozen_path),
                **meta,
            },
            "window_sizes": {
                "train": train_size, "validation": validation_size,
                "test": test_size, "step": step_size,
            },
            **result,
        }

    stamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    report_path = output_dir / f"baseline_{pair}_{stamp}.json"
    atomic_write_json(report_path, report, indent=2, ensure_ascii=False)

    print(f"Baseline saved: {report_path}")
    for interval, value in report["intervals"].items():
        aggregate = value["aggregate_test"]
        print(
            f"{interval}m | folds={aggregate['fold_count']} "
            f"mean={(aggregate.get('mean_return_pct') or 0.0):+.3f}% "
            f"worst={(aggregate.get('worst_return_pct') or 0.0):+.3f}% "
            f"worstDD={(aggregate.get('worst_max_drawdown_pct') or 0.0):.3f}% "
            f"cycles={aggregate['total_cycles']} fills={aggregate['total_fills']} "
            f"ambiguous={aggregate['total_ambiguous_bars']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
