#!/usr/bin/env python3
"""Trading212 walk-forward baseline over the live engine and Yahoo OHLC.

The configuration comes straight from ``212trading/config.<profile>.env``. The runner
optimises no parameters and never contacts the Trading212 order API.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
T212_DIR = ROOT / "212trading"
sys.path.insert(0, str(T212_DIR))
sys.path.insert(0, str(ROOT))

from ipo_common import parse_dotenv  # noqa: E402
import replay as t212_replay  # noqa: E402
from strategy import StratParams  # noqa: E402
from offline.backtests.datasets import (  # noqa: E402
    align_previous_values,
    dataset_metadata,
    drop_incomplete_last_bar,
    load_dataset,
    merge_datasets,
    save_dataset,
    validate_dataset,
)
from offline.backtests.evaluation import (  # noqa: E402
    assess_decision_evidence,
    automatic_window_sizes,
    evaluate_walk_forward,
)
from offline.backtests.execution import ExecutionModel  # noqa: E402


from market_data import fetch_yahoo_candles, parse_interval_minutes


def periods_per_year(interval_minutes: int) -> float:
    # Regular US session: 6.5h and roughly 252 trading days per year.
    if interval_minutes >= 1440:
        return 252.0 * 1440.0 / interval_minutes
    return 252.0 * 390.0 / interval_minutes


def fx_quote(currency: str) -> tuple[str, bool]:
    """Return the Yahoo symbol and whether the value must be inverted to USD per unit."""
    currency = currency.upper()
    if currency == "RON":
        return "USDRON=X", True
    return f"{currency}USD=X", False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="nvda", help="config.<profile>.env")
    parser.add_argument("--range", default="2y", help="range Yahoo: 60d, 1y, 2y etc.")
    parser.add_argument("--interval", default="1d", help="interval Yahoo: 5m, 1h, 1d")
    parser.add_argument("--dataset", type=Path, help="a frozen CSV instead of Yahoo")
    parser.add_argument(
        "--seed-dataset", type=Path,
        help="frozen unit history plus the new Yahoo window (intraday accumulation)",
    )
    parser.add_argument(
        "--fx-to-usd", type=float,
        help="fixed FX override; by default the historical series is downloaded for non-USD",
    )
    parser.add_argument("--fx-dataset", type=Path, help="a frozen FX OHLC CSV")
    parser.add_argument("--fx-symbol", help="override simbol Yahoo FX, ex. EURUSD=X")
    parser.add_argument("--fx-invert", action="store_true", help="invert the FX close")
    parser.add_argument("--spread-bps", type=float, default=0.0)
    parser.add_argument("--market-slippage-bps", type=float, default=0.0)
    parser.add_argument("--partial-fill-ratio", type=float, default=1.0)
    parser.add_argument(
        "--intrabar-policy", choices=("buy_first", "sell_first", "worst_case"),
        default="buy_first",
    )
    parser.add_argument("--train", type=int)
    parser.add_argument("--validation", type=int)
    parser.add_argument("--test", type=int)
    parser.add_argument("--step", type=int)
    parser.add_argument(
        "--warmup", type=int,
        help="preceding signal bars; defaults to 12 when the DCA gate is active",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "offline/results/t212_walk_forward",
    )
    args = parser.parse_args()

    interval_minutes = parse_interval_minutes(args.interval)
    config_path = T212_DIR / f"config.{args.profile}.env"
    if not config_path.exists():
        parser.error(f"profil inexistent: {config_path}")
    config = parse_dotenv(str(config_path))
    params = StratParams.from_env(config)
    symbol = (config.get("YAHOO_SYMBOL") or params.yahoo_sym).strip().upper()
    if not symbol:
        parser.error("the profile does not define YAHOO_SYMBOL")
    if args.fx_to_usd is not None and args.fx_dataset is not None:
        parser.error("--fx-to-usd and --fx-dataset are alternatives")
    if args.dataset is not None and args.seed_dataset is not None:
        parser.error("--dataset and --seed-dataset are alternatives")
    if args.fx_to_usd is not None and args.fx_to_usd <= 0:
        parser.error("--fx-to-usd must be positive")
    try:
        execution = ExecutionModel(
            spread_bps=args.spread_bps,
            market_slippage_bps=args.market_slippage_bps,
            partial_fill_ratio=args.partial_fill_ratio,
            intrabar_policy=args.intrabar_policy,
        )
    except ValueError as exc:
        parser.error(str(exc))

    explicit = (args.train, args.validation, args.test)
    if any(value is not None for value in explicit) and not all(value is not None for value in explicit):
        parser.error("--train, --validation and --test are given together")
    if any(value is not None and value <= 0 for value in (*explicit, args.step)):
        parser.error("the walk-forward sizes must be positive")
    warmup = (12 if params.dca_trend_gate_pct > 0 else 0) if args.warmup is None else args.warmup
    if warmup < 0:
        parser.error("--warmup cannot be negative")

    source = args.dataset.expanduser().resolve() if args.dataset else None
    seed_source = args.seed_dataset.expanduser().resolve() if args.seed_dataset else None
    if source:
        records = load_dataset(source)
        source_label = str(source)
    else:
        fetched = fetch_yahoo_candles(symbol, args.range, args.interval)
        records = merge_datasets(load_dataset(seed_source), fetched) if seed_source else fetched
        source_label = (
            f"Yahoo chart {args.range}/{args.interval} + seed {seed_source}"
            if seed_source else f"Yahoo chart {args.range}/{args.interval}"
        )
    records = validate_dataset(records)
    output_dir = args.output_dir.expanduser().resolve() / args.profile

    fx_records = None
    fx_metadata = None
    fx_fixed = args.fx_to_usd
    fx_source = None
    fx_symbol = None
    fx_invert = False
    if params.currency == "USD":
        if args.fx_dataset or args.fx_symbol or args.fx_invert:
            parser.error("the historical FX options are not needed for a USD profile")
        fx_fixed = 1.0 if fx_fixed is None else fx_fixed
        fx_metadata = {"mode": "identity", "currency": "USD", "usd_per_unit": fx_fixed}
    elif fx_fixed is not None:
        fx_metadata = {
            "mode": "fixed_override", "currency": params.currency,
            "usd_per_unit": fx_fixed,
        }
    else:
        default_symbol, default_invert = fx_quote(params.currency)
        fx_symbol = (args.fx_symbol or default_symbol).upper()
        fx_invert = args.fx_invert or (args.fx_symbol is None and default_invert)
        fx_source = args.fx_dataset.expanduser().resolve() if args.fx_dataset else None
        fx_records = (
            load_dataset(fx_source) if fx_source
            else fetch_yahoo_candles(fx_symbol, args.range, args.interval)
        )
        fx_records = validate_dataset(fx_records)
        fx_data = dataset_metadata(fx_records)
        fx_frozen = (
            output_dir / "datasets"
            / f"{fx_symbol.replace('=', '')}_{args.interval}_{fx_data['sha256'][:12]}.csv"
        )
        save_dataset(fx_records, fx_frozen)
        fx_metadata = {
            "mode": "historical_asof", "currency": params.currency,
            "quote_symbol": fx_symbol, "invert": fx_invert,
            "source": str(fx_source) if fx_source else f"Yahoo chart {args.range}/{args.interval}",
            "frozen_file": str(fx_frozen), **fx_data,
        }
    if args.train is None:
        train, validation, test, step = automatic_window_sizes(len(records))
    else:
        train, validation, test = args.train, args.validation, args.test
        step = args.step or test

    metadata = dataset_metadata(records)
    frozen_path = output_dir / "datasets" / f"{symbol}_{args.interval}_{metadata['sha256'][:12]}.csv"
    save_dataset(records, frozen_path)
    adapter = t212_replay.make_t212_replay_adapter(
        params,
        bar_minutes=interval_minutes,
        fx_to_usd=fx_fixed,
        fx_records=fx_records,
        fx_invert=fx_invert,
        periods_per_year=periods_per_year(interval_minutes),
        execution=execution,
    )
    result = evaluate_walk_forward(
        records,
        adapter,
        train_size=train,
        validation_size=validation,
        test_size=test,
        step_size=step,
        warmup_bars=warmup,
        validate_contract=True,
    )
    evidence = assess_decision_evidence(records, result["aggregate_test"])

    generated_at = dt.datetime.now(tz=dt.timezone.utc)
    report = {
        "schema_version": 1,
        "generated_at_utc": generated_at.isoformat(),
        "purpose": "Trading212 live-engine fixed-config walk-forward baseline",
        "runtime": {
            "profile": args.profile,
            "ticker": config.get("T212_TICKER"),
            "yahoo_symbol": symbol,
            "execute_live": config.get("STRAT_EXECUTE", "false").lower() == "true",
        },
        "strategy_params": dataclasses.asdict(params),
        "fx_model": fx_metadata,
        "dataset": {
            "source": source_label,
            "frozen_file": str(frozen_path),
            **metadata,
        },
        "window_sizes": {
            "train": train, "validation": validation, "test": test, "step": step,
            "signal_warmup_bars": warmup,
        },
        "execution_model": {
            "decision": "live Strategy.step at close",
            "earliest_fill": "next bar",
            "orders": "limit fill against OHLC high/low",
            "fx_fee_pct_per_leg": params.fx_fee_pct,
            "calendar_periods_per_year": periods_per_year(interval_minutes),
            **dataclasses.asdict(execution),
        },
        "evidence_gate": evidence,
        **result,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    report_path = output_dir / f"baseline_{symbol}_{stamp}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    aggregate = result["aggregate_test"]
    print(f"Baseline saved: {report_path}")
    print(
        f"{args.profile}/{symbol} | folds={aggregate['fold_count']} "
        f"mean={aggregate['mean_return_pct']:+.3f}% "
        f"worst={aggregate['worst_return_pct']:+.3f}% "
        f"worstDD={aggregate['worst_max_drawdown_pct']:.3f}% "
        f"cycles={aggregate['total_cycles']} fills={aggregate['total_fills']} "
        f"ambiguous={aggregate['total_ambiguous_bars']} "
        f"evidence={evidence['verdict']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
