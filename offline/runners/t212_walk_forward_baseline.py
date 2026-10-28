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
import math
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
T212_DIR = ROOT / "212trading"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util

from botcore import _parse_bool_flag, parse_dotenv  # noqa: E402
_orig_sys_path = list(sys.path)
try:
    if str(T212_DIR) not in sys.path:
        sys.path.append(str(T212_DIR))
    _replay_spec = importlib.util.spec_from_file_location(
        "t212_wf_replay", str(T212_DIR / "replay.py")
    )
    t212_replay = importlib.util.module_from_spec(_replay_spec)
    sys.modules[_replay_spec.name] = t212_replay
    _replay_spec.loader.exec_module(t212_replay)
    StratParams = t212_replay._strat.StratParams

    _md_spec = importlib.util.spec_from_file_location(
        "t212_wf_market_data", str(T212_DIR / "market_data.py")
    )
    _t212_md = importlib.util.module_from_spec(_md_spec)
    sys.modules[_md_spec.name] = _t212_md
    _md_spec.loader.exec_module(_t212_md)
    fetch_yahoo_candles = _t212_md.fetch_yahoo_candles
    parse_interval_minutes = _t212_md.parse_interval_minutes
finally:
    sys.path[:] = _orig_sys_path

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
    add_walk_forward_cli_args,
    assess_decision_evidence,
    automatic_window_sizes,
    build_execution_model_from_args,
    evaluate_walk_forward,
    resolve_window_sizes,
)
from offline.backtests.execution import ExecutionModel  # noqa: E402


def periods_per_year(interval_minutes: int) -> float:
    if isinstance(interval_minutes, bool):
        raise ValueError("interval_minutes must be positive")
    try:
        im = float(interval_minutes)
    except (TypeError, ValueError) as exc:
        raise ValueError("interval_minutes must be positive") from exc
    if not math.isfinite(im) or im <= 0:
        raise ValueError("interval_minutes must be positive")
    # Regular US session: 6.5h and roughly 252 trading days per year.
    if im >= 1440:
        return 252.0 * 1440.0 / im
    return 252.0 * 390.0 / im


def fx_quote(currency: str) -> tuple[str, bool]:
    """Return the Yahoo symbol and whether the value must be inverted to USD per unit."""
    if currency is None or isinstance(currency, bool):
        raise ValueError("currency must be a non-empty string")
    norm_ccy = str(currency).strip().upper()
    if not norm_ccy:
        raise ValueError("currency must be a non-empty string")
    if norm_ccy == "RON":
        return "USDRON=X", True
    return f"{norm_ccy}USD=X", False


def main(argv: list[str] | None = None) -> int:
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
    add_walk_forward_cli_args(parser, include_warmup=True, default_warmup=None)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "offline/results/t212_walk_forward",
    )
    args = parser.parse_args(argv)

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
    if args.fx_to_usd is not None and (not math.isfinite(args.fx_to_usd) or args.fx_to_usd <= 0):
        parser.error("--fx-to-usd must be positive")
    execution = build_execution_model_from_args(parser, args)
    warmup = (12 if params.dca_trend_gate_pct > 0 else 0) if args.warmup is None else args.warmup

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
    train, validation, test, step = resolve_window_sizes(args, len(records))

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
            "interval": args.interval,
            "interval_minutes": interval_minutes,
            "execute_live": _parse_bool_flag(config.get("STRAT_EXECUTE"), False),
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
        f"mean={(aggregate.get('mean_return_pct') or 0.0):+.3f}% "
        f"worst={(aggregate.get('worst_return_pct') or 0.0):+.3f}% "
        f"worstDD={(aggregate.get('worst_max_drawdown_pct') or 0.0):.3f}% "
        f"cycles={aggregate['total_cycles']} fills={aggregate['total_fills']} "
        f"ambiguous={aggregate['total_ambiguous_bars']} "
        f"evidence={evidence['verdict']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
