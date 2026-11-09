#!/usr/bin/env python3
"""Walk-forward baseline for the live Hyperliquid perpetual configuration, without optimization.

Loads the production configuration (hyperliquid/config.env), runs the faithful replay
engine from hyperliquid/replay.py across rolling time windows without data leakage or
shuffling, and validates metrics against the canonical contract.
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HL_DIR = ROOT / "hyperliquid"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from botcore import required_env, required_float_env, safe_nonneg_float
from strategies.spot_engine import StratParams as SpotStratParams
from strategies.spot_replay import align_spot_interval_params

from offline.backtests.datasets import (
    dataset_metadata,
    load_dataset,
    save_dataset,
    validate_dataset,
)
from offline.backtests.evaluation import (
    add_walk_forward_cli_args,
    build_and_save_interval_baseline_report,
    build_execution_model_from_args,
    load_layered_env_config,
    load_venue_module,
    parse_dataset_specs,
    parse_intervals,
)

_hl_md = load_venue_module("hyperliquid", "market_data.py", "hl_wf_md")
fetch_ohlc_candles = _hl_md.fetch_ohlc_candles

_hl_replay = load_venue_module("hyperliquid", "replay.py", "hl_wf_replay")
make_hyperliquid_replay_adapter = _hl_replay.make_hyperliquid_replay_adapter
StratParams = _hl_replay.StratParams


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
    parser.add_argument("--intervals", type=parse_intervals, default=[60, 240, 1440])
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

    cfg = load_layered_env_config(args.config_file, args.env_file, override_prefixes=("HL_",))

    coin = (args.coin or required_env("HL_COIN", env=cfg)).strip().upper()
    direction = (args.direction or required_env("HL_DIRECTION", env=cfg)).strip().lower()
    if direction not in ("long", "short"):
        parser.error(f"Invalid direction: {direction!r}")
    fee_pct = args.fee if args.fee is not None else required_float_env("HL_FEE_PCT", env=cfg)
    if safe_nonneg_float(fee_pct, default=None) is None:
        parser.error("--fee cannot be negative")

    if args.engine == "spot":
        strat_params = SpotStratParams.from_env(env=cfg)
    else:
        strat_params = StratParams.from_env(env=cfg)
        if args.fee is not None:
            strat_params = dataclasses.replace(strat_params, fee_pct=fee_pct)

    execution = build_execution_model_from_args(parser, args)
    dataset_map = parse_dataset_specs(args.dataset)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_by_interval = {}
    for interval in args.intervals:
        frozen_file = dataset_map.get(interval)
        if frozen_file:
            frozen_path = frozen_file
            records = validate_dataset(load_dataset(frozen_file), interval_minutes=interval)
            source_label = f"frozen file: {frozen_file}"
        else:
            raw_candles = fetch_ohlc_candles(coin, interval, drop_incomplete=True)
            if not raw_candles:
                print(f"[!] No candles returned for {coin} {interval}m; skipping.")
                continue
            records = validate_dataset(raw_candles, interval_minutes=interval)
            meta = dataset_metadata(records, interval_minutes=interval)
            frozen_path = output_dir / "datasets" / f"{coin}_{interval}m_{meta['sha256'][:12]}.csv"
            save_dataset(records, frozen_path)
            source_label = f"Hyperliquid API, frozen to {frozen_path}"

        if len(records) < 20:
            print(f"[!] Insufficient bars for {coin} {interval}m ({len(records)} < 20); skipping.")
            continue

        adapter = make_hyperliquid_replay_adapter(
            params=align_spot_interval_params(strat_params, interval) if args.engine == "spot" else strat_params,
            direction=direction,
            coin=coin,
            bar_minutes=float(interval),
            execution=execution,
            fee_pct=fee_pct,
            engine=args.engine,
        )
        summary_by_interval[interval] = build_and_save_interval_baseline_report(
            records,
            adapter,
            args=args,
            interval=interval,
            symbol_or_coin=coin,
            purpose=f"Hyperliquid live-{args.engine}-engine fixed-config walk-forward baseline",
            runtime={
                "coin": coin,
                "direction": direction,
                "interval_minutes": interval,
                "engine": args.engine,
            },
            strategy_params=dataclasses.asdict(strat_params),
            fee_pct=fee_pct,
            execution=execution,
            source_label=source_label,
            frozen_path=frozen_path,
            output_dir=output_dir,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
