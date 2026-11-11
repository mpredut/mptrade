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

from botcore import (
    load_local_module,
    required_env,
    required_float_env,
    safe_nonneg_float,
    safe_pos_float,
)
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
    load_venue_replay_module,
    parse_cli_float,
    parse_cli_int,
    parse_dataset_specs,
    parse_intervals,
    parse_optional_lookback_days,
    validate_bt_cli_token,
)
from offline.runners.fetch_hyperliquid_candles import _hl_md, resolve_spot_pair
from providers.hyperliquid_provider import HyperliquidProvider
from providers.market_api import api as mkt


def fetch_ohlc_candles(
    symbol: str,
    interval: int | str = 60,
    *,
    engine: str = "perp",
    spot_pair: str | None = None,
    quote: str = "USDC",
    **kwargs,
) -> list[dict]:
    target = str(symbol or "").strip()
    if engine == "spot" and target and not target.startswith("@"):
        norm_tok = target.upper()
        explicit_pair = str(spot_pair or "").strip()
        if explicit_pair:
            target = explicit_pair
        elif norm_tok in HyperliquidProvider.KNOWN_SPOT_PAIRS and str(quote or "USDC").strip().upper() == "USDC":
            target = HyperliquidProvider.KNOWN_SPOT_PAIRS[norm_tok]
        else:
            spot_meta = _hl_md._post_info({"type": "spotMeta"})
            target = resolve_spot_pair(norm_tok, quote or "USDC", spot_meta)
    return mkt.fetch_ohlc_candles(target, interval, provider_name="hyperliquid", **kwargs)


_hl_common = load_local_module(str(HL_DIR), "common")
load_isolated_hl_cfg = _hl_common.load_isolated_hl_cfg
_hl_replay = load_venue_replay_module("hyperliquid", "hl_wf_replay")
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
    parser.add_argument("--fee", type=parse_cli_float, default=None, help="fee per leg (pct), defaults to HL_FEE_PCT")
    parser.add_argument(
        "--lookback-days", type=parse_cli_int, default=None,
        help="API fetch lookback in days when --dataset is omitted (defaults to full 5000-candle window)",
    )
    add_walk_forward_cli_args(
        parser,
        include_warmup=True,
        default_warmup=0,
        include_dataset_list=True,
        default_output_dir=ROOT / "offline/results/hyperliquid_walk_forward",
    )
    args = parser.parse_args(argv)
    lookback_days = parse_optional_lookback_days(parser, args.lookback_days)

    try:
        cfg = load_isolated_hl_cfg(
            HL_DIR, config_path=args.config_file, env_path=args.env_file,
        )
        coin = validate_bt_cli_token(args.coin or required_env("HL_COIN", env=cfg), "coin").upper()
        direction = (args.direction or required_env("HL_DIRECTION", env=cfg)).strip().lower()
        if direction not in ("long", "short"):
            parser.error(f"Invalid direction: {direction!r}")
        fee_pct = args.fee if args.fee is not None else required_float_env("HL_FEE_PCT", env=cfg)
        fee_val = safe_nonneg_float(fee_pct, default=None)
        if fee_val is None:
            parser.error("--fee cannot be negative")
        if fee_val >= 100.0:
            parser.error("--fee must be < 100")

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
                raw_loaded = load_dataset(frozen_file)
                if len(raw_loaded) < 20:
                    print(f"[!] Insufficient bars for {coin} {interval}m ({len(raw_loaded)} < 20); skipping.")
                    continue
                records = validate_dataset(raw_loaded, interval_minutes=interval)
                source_label = f"frozen file: {frozen_file}"
            else:
                fetch_kwargs = (
                    {"lookback_hours": lookback_days * 24}
                    if lookback_days is not None
                    else {"start_ms": 0}
                )
                raw_candles = fetch_ohlc_candles(
                    coin,
                    interval,
                    drop_incomplete=True,
                    engine=args.engine,
                    spot_pair=cfg.get("HL_SPOT_PAIR"),
                    quote=strat_params.currency if args.engine == "spot" else "USDC",
                    **fetch_kwargs,
                )
                if not raw_candles:
                    print(f"[!] No candles returned for {coin} {interval}m; skipping.")
                    continue
                if len(raw_candles) < 20:
                    print(f"[!] Insufficient bars for {coin} {interval}m ({len(raw_candles)} < 20); skipping.")
                    continue
                records = validate_dataset(raw_candles, interval_minutes=interval)
                meta = dataset_metadata(records, interval_minutes=interval)
                frozen_path = output_dir / "datasets" / f"{coin}_{args.engine}_{interval}m_{meta['sha256'][:12]}.csv"
                save_dataset(records, frozen_path)
                source_label = f"Hyperliquid API, frozen to {frozen_path}"

            effective_params = (
                align_spot_interval_params(strat_params, interval)
                if args.engine == "spot"
                else strat_params
            )
            adapter = make_hyperliquid_replay_adapter(
                params=effective_params,
                direction=direction,
                coin=coin,
                bar_minutes=safe_pos_float(interval, default=60.0),
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
                strategy_params=dataclasses.asdict(effective_params),
                fee_pct=fee_pct,
                execution=execution,
                source_label=source_label,
                frozen_path=frozen_path,
                output_dir=output_dir,
            )
    except FileNotFoundError:
        raise
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))

    if dataset_map and not summary_by_interval:
        print(
            f"[!] ERROR: 0 baseline reports generated for {coin} across supplied --dataset intervals.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
