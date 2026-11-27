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

from botcore import load_layered_env_config, safe_pos_float
from cachepaths import split_symbol_assets
from offline.backtests.evaluation import (
    add_walk_forward_cli_args,
    build_and_save_interval_baseline_report,
    ensure_cli_output_dir,
    extract_cfg_float,
    extract_cfg_int,
    load_or_fetch_walk_forward_dataset,
    parse_cli_int,
    parse_dataset_specs,
    parse_optional_lookback_days,
    parse_walk_forward_cli_args,
    validate_bt_cli_token,
)
from providers.market_api import api as mkt
from strategies.spot_engine import StratParams
from strategies.spot_replay import (
    align_spot_interval_params,
    make_spot_replay_adapter,
    scale_subbar_timeout_params,
)


def fetch_ohlc_candles(symbol: str, *, interval: int = 60, **kwargs) -> list[dict]:
    return mkt.fetch_ohlc_candles(symbol, interval=interval, provider_name="binance", **kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--config-file", default=str(ROOT / "config.env"))
    parser.add_argument("--symbol", help="defaults to SYMBOL or BINANCE_PAIR (e.g. BTCUSDT)")
    parser.add_argument(
        "--lookback-days", type=parse_cli_int, default=630,
        help="API fetch lookback in days when --dataset is omitted (defaults to 630)",
    )
    add_walk_forward_cli_args(
        parser,
        include_warmup=True,
        default_warmup=0,
        include_dataset_list=True,
        default_output_dir=ROOT / "offline/results/binance_walk_forward",
        default_intervals=[60, 240, 1440],
        default_fee=0.10,
        fee_help="fee per leg (pct), defaults to 0.10%%",
    )
    args, execution = parse_walk_forward_cli_args(parser, argv)
    lookback_days = parse_optional_lookback_days(
        parser, args.lookback_days, allow_none=False
    ) or 630

    try:
        cfg = load_layered_env_config(
            args.config_file, args.env_file, override_prefixes=("BINANCE_",), fallback_prefixes=(),
        )
    except FileNotFoundError:
        raise
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    try:
        symbol = validate_bt_cli_token(
            args.symbol or cfg.get("SYMBOL") or cfg.get("BINANCE_PAIR") or "BTCUSDT",
            "symbol",
        ).upper()
    except ValueError as exc:
        parser.error(str(exc))

    has_explicit_strat = any(k.startswith("STRAT_") for k in cfg)
    try:
        if has_explicit_strat:
            strat_params = StratParams.from_env(cfg)
        else:
            try:
                strat_params = StratParams.from_env(cfg)
            except (ValueError, KeyError):
                quote_ccy = split_symbol_assets(symbol)[1] or "USDT"
                strat_params = StratParams.from_dict({
                    "currency": quote_ccy,
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
        supplied_datasets = parse_dataset_specs(args.dataset)
        output_dir = ensure_cli_output_dir(args.output_dir)
        dataset_dir = ensure_cli_output_dir(output_dir / "datasets")

        summary_by_interval = {}

        for interval in args.intervals:
            loaded = load_or_fetch_walk_forward_dataset(
                frozen_file=supplied_datasets.get(interval),
                interval=interval,
                symbol_label=symbol,
                fetch_fn=lambda iv=interval: fetch_ohlc_candles(
                    symbol,
                    interval=iv,
                    lookback_days=lookback_days,
                    drop_incomplete=True,
                ),
                frozen_path_fn=lambda meta, iv=interval: dataset_dir / f"{symbol}_{iv}m_{meta['sha256'][:12]}.csv",
                api_source_label_fn=lambda _p: "binance_public_klines",
                frozen_source_label="frozen_csv",
            )
            if loaded is None:
                continue
            records, frozen_path, source_label = loaded
            aligned_params = align_spot_interval_params(strat_params, interval)
            adapter = make_spot_replay_adapter(
                params=scale_subbar_timeout_params(aligned_params, interval),
                fee_pct=args.fee,
                bar_minutes=safe_pos_float(interval, default=60.0),
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
                strategy_params=dataclasses.asdict(aligned_params),
                fee_pct=args.fee,
                execution=execution,
                source_label=source_label,
                frozen_path=frozen_path,
                output_dir=output_dir,
            )
    except FileNotFoundError:
        raise
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))

    if supplied_datasets and not summary_by_interval:
        print(
            f"[!] ERROR: 0 baseline reports generated for {symbol} across supplied --dataset intervals.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
