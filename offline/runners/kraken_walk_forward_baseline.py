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
import os
from pathlib import Path
import sys
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
KRAKEN_DIR = ROOT / "kraken"
sys.path.insert(0, str(KRAKEN_DIR))
sys.path.insert(0, str(ROOT))

from kraken_common import load_dotenv  # noqa: E402
import replay as kraken_replay  # noqa: E402
from strategies.spot_engine import StratParams  # noqa: E402
from offline.backtests.datasets import (  # noqa: E402
    dataset_sha256,
    load_dataset as load_frozen_dataset,
    save_dataset as save_frozen_dataset,
    validate_dataset,
)
from offline.backtests.evaluation import (  # noqa: E402
    automatic_window_sizes,
    evaluate_segment as _evaluate_segment,
    evaluate_walk_forward as _evaluate_walk_forward,
    iso_utc as _iso,
)
from offline.backtests.execution import ExecutionModel  # noqa: E402


def fetch_closed_candles(pair: str, interval: int) -> list[dict]:
    """Read at most 720 Kraken bars and drop the last, still-forming bar."""
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}"
    request = urllib.request.Request(
        url, headers={"User-Agent": "binance-repo/walk-forward-baseline"}
    )
    with urllib.request.urlopen(request, timeout=25) as response:
        payload = json.loads(response.read())
    if payload.get("error"):
        raise RuntimeError(", ".join(payload["error"]))
    result = payload.get("result", {})
    key = next((name for name in result if name != "last"), None)
    rows = result.get(key, []) if key else []
    return [
        {
            "timestamp": int(row[0]),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
        }
        for row in rows[:-1]
    ]


def _segment_result(records: list[dict], params: StratParams, fee_pct: float,
                    interval: int, warmup_records: list[dict] | tuple = (),
                    execution: ExecutionModel | None = None) -> dict:
    original_log = kraken_replay._strat.log
    kraken_replay._strat.log = lambda *_args, **_kwargs: None
    try:
        return _evaluate_segment(
            records,
            lambda ohlc, warmup, _context: kraken_replay.run_replay(
                ohlc, params, fee_pct=fee_pct, bar_minutes=interval,
                warmup_ohlc=warmup, execution=execution,
            ),
            warmup_records=warmup_records,
        )
    finally:
        kraken_replay._strat.log = original_log


def evaluate_walk_forward(records: list[dict], params: StratParams, *, fee_pct: float,
                          interval: int, train_size: int, validation_size: int,
                          test_size: int, step_size: int, warmup_bars: int = 0,
                          execution: ExecutionModel | None = None) -> dict:
    original_log = kraken_replay._strat.log
    kraken_replay._strat.log = lambda *_args, **_kwargs: None
    try:
        return _evaluate_walk_forward(
            records,
            lambda ohlc, warmup, _context: kraken_replay.run_replay(
                ohlc, params, fee_pct=fee_pct, bar_minutes=interval,
                warmup_ohlc=warmup, execution=execution,
            ),
            train_size=train_size,
            validation_size=validation_size,
            test_size=test_size,
            step_size=step_size,
            warmup_bars=warmup_bars,
        )
    finally:
        kraken_replay._strat.log = original_log


def _parse_intervals(value: str) -> list[int]:
    intervals = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not intervals or any(interval <= 0 for interval in intervals):
        raise argparse.ArgumentTypeError("intervals must be positive minutes")
    return intervals


def _parse_dataset_specs(specs: list[str]) -> dict[int, Path]:
    result = {}
    for spec in specs:
        try:
            interval, path = spec.split("=", 1)
            result[int(interval)] = Path(path).expanduser().resolve()
        except (ValueError, OSError) as exc:
            raise ValueError(f"dataset invalid {spec!r}; formatul este INTERVAL=CALE") from exc
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=str(KRAKEN_DIR / ".env"))
    parser.add_argument("--config-file", default=str(KRAKEN_DIR / "config.env"))
    parser.add_argument("--pair", help="defaults to KRAKEN_PAIR from the live configuration")
    parser.add_argument("--intervals", type=_parse_intervals, default=[60, 240, 1440])
    parser.add_argument("--fee", type=float, default=0.26, help="fee per leg, procente")
    parser.add_argument("--spread-bps", type=float, default=0.0)
    parser.add_argument("--market-slippage-bps", type=float, default=0.0)
    parser.add_argument("--partial-fill-ratio", type=float, default=1.0)
    parser.add_argument(
        "--intrabar-policy", choices=("buy_first", "sell_first", "worst_case"),
        default="buy_first",
    )
    parser.add_argument("--train", type=int, help="bars; by default automatically 45%% of the history")
    parser.add_argument("--validation", type=int, help="bare; implicit auto 10%%")
    parser.add_argument("--test", type=int, help="bare; implicit auto 15%%")
    parser.add_argument("--step", type=int, help="bars; by default equal to TEST")
    parser.add_argument(
        "--warmup", type=int, default=0,
        help="preceding bars for the signals; they carry no position or P&L into the segment",
    )
    parser.add_argument(
        "--dataset", action="append", default=[], metavar="INTERVAL=CALE",
        help="use a frozen CSV instead of the API; repeatable for every interval",
    )
    parser.add_argument(
        "--output-dir", default=str(ROOT / "offline" / "results" / "kraken_walk_forward")
    )
    args = parser.parse_args()

    if args.fee < 0:
        parser.error("--fee cannot be negative")
    explicit_windows = (args.train, args.validation, args.test)
    if any(value is not None for value in explicit_windows) and not all(
            value is not None for value in explicit_windows):
        parser.error("--train, --validation and --test are given together or not at all")
    if any(value is not None and value <= 0
           for value in (*explicit_windows, args.step)):
        parser.error("the walk-forward sizes must be positive")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")
    try:
        execution = ExecutionModel(
            spread_bps=args.spread_bps,
            market_slippage_bps=args.market_slippage_bps,
            partial_fill_ratio=args.partial_fill_ratio,
            intrabar_policy=args.intrabar_policy,
        )
    except ValueError as exc:
        parser.error(str(exc))

    # The same order as kraken_bot.py: .env takes priority, config.env fills in the rest.
    load_dotenv(args.env_file)
    load_dotenv(args.config_file)
    pair = (args.pair or os.environ.get("KRAKEN_PAIR") or "").strip().upper()
    if not pair:
        parser.error("missing pair: --pair or KRAKEN_PAIR")
    params = StratParams.from_env()
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
            "strategy_mode": os.environ.get("STRATEGY_MODE", "avg_tp"),
            "execute_live": os.environ.get("STRAT_EXECUTE", "false").lower() == "true",
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
        digest = dataset_sha256(records)
        frozen_path = dataset_dir / f"{pair}_{interval}m_{digest[:12]}.csv"
        save_frozen_dataset(records, frozen_path)
        if args.train is None:
            train_size, validation_size, test_size, step_size = automatic_window_sizes(
                len(records)
            )
        else:
            train_size, validation_size, test_size = (
                args.train, args.validation, args.test,
            )
            step_size = args.step or test_size
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
                "sha256": digest,
                "bars": len(records),
                "start_utc": _iso(records[0]["timestamp"]),
                "end_utc": _iso(records[-1]["timestamp"]),
            },
            "window_sizes": {
                "train": train_size, "validation": validation_size,
                "test": test_size, "step": step_size,
            },
            **result,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    report_path = output_dir / f"baseline_{pair}_{stamp}.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print(f"Baseline saved: {report_path}")
    for interval, value in report["intervals"].items():
        aggregate = value["aggregate_test"]
        print(
            f"{interval}m | folds={aggregate['fold_count']} "
            f"mean={aggregate['mean_return_pct']:+.3f}% "
            f"worst={aggregate['worst_return_pct']:+.3f}% "
            f"worstDD={aggregate['worst_max_drawdown_pct']:.3f}% "
            f"cycles={aggregate['total_cycles']} fills={aggregate['total_fills']} "
            f"ambiguous={aggregate['total_ambiguous_bars']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
