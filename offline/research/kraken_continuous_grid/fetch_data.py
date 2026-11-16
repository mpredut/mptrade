#!/usr/bin/env python3
"""Build the 4h datasets for the continuous grid: Binance USDT spot since 2018 plus HYPE.

Binance public klines serve as a price proxy for the Kraken pairs (the strategy only sees
OHLC; fees are modelled separately). HYPE comes from the frozen Hyperliquid spot dataset.
Output: $GRID_OUT/data/<ASSET>_240m.csv with timestamp,open,high,low,close.
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from botcore import optional_env, parse_cli_float
from offline.backtests.datasets import (
    DAILY_DATASET_MAX_AGE_SEC,
    has_usable_ohlc_csv,
    save_dataset,
    should_skip_ohlc_csv_refresh,
    validate_dataset,
)
from providers.market_api import api as mkt
from state_io import atomic_snapshot_file

EXP = optional_env("GRID_OUT") or os.path.join(ROOT, "offline", "results", "kraken_continuous_grid")
DATA = os.path.join(EXP, "data")
ASSETS = ("TAO", "ADA", "SOL", "BTC", "ETH", "XRP", "DOGE", "LINK", "AVAX", "BNB")
START_MS = 1514764800000   # 2018-01-01
MIN_GRID_BARS = 43


def _has_usable_csv(path: str, min_bars: int = MIN_GRID_BARS) -> bool:
    return has_usable_ohlc_csv(path, min_bars=min_bars, strict=True)


def fetch(
    symbol: str,
    name: str,
    force: bool = False,
    *,
    missing_only: bool = False,
    max_age_sec: float = DAILY_DATASET_MAX_AGE_SEC,
) -> bool:
    out_path = os.path.join(DATA, f"{name}_240m.csv")
    skip, has_usable = should_skip_ohlc_csv_refresh(
        out_path,
        min_bars=MIN_GRID_BARS,
        strict=True,
        force=force,
        missing_only=missing_only,
        max_age_sec=max_age_sec,
    )
    if skip:
        print(f"[fetch] {name}: using fresh cached {out_path}", flush=True)
        return True
    try:
        candles = mkt.fetch_ohlc_candles(
            symbol, interval=240, provider_name="binance", start_ms=START_MS, drop_incomplete=True,
        )
        if not candles:
            print(f"[fetch] {name}: no candles returned; keeping existing file if present", flush=True)
            return has_usable
        # Allow historical exchange maintenance windows while enforcing strict chronological order & OHLC invariants.
        validated = validate_dataset(candles)
        if len(validated) < MIN_GRID_BARS:
            print(
                f"[fetch] {name}: insufficient bars ({len(validated)} < {MIN_GRID_BARS}); "
                "keeping existing file if present",
                flush=True,
            )
            return has_usable
        save_dataset(validated, out_path)
        print(f"[fetch] {name}: {len(validated)} bars "
              f"{time.strftime('%Y-%m-%d', time.gmtime(validated[0]['timestamp']))} -> "
              f"{time.strftime('%Y-%m-%d', time.gmtime(validated[-1]['timestamp']))}", flush=True)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[fetch] {name}: error {exc}; keeping existing file if present", flush=True)
        return has_usable


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-fetch all symbols regardless of existing CSV age")
    ap.add_argument("--missing-only", action="store_true", help="skip symbols with any usable existing CSV")
    ap.add_argument(
        "--max-age-hours",
        type=parse_cli_float,
        default=DAILY_DATASET_MAX_AGE_SEC / 3600.0,
        help="maximum cached CSV age in hours before refreshing (default: 18h)",
    )
    cli_args = ap.parse_args()
    if cli_args.max_age_hours < 0.0:
        ap.error("--max-age-hours must be >= 0")
    force_refresh = bool(cli_args.force)
    missing_only_flag = bool(cli_args.missing_only and not force_refresh)
    max_age_s = cli_args.max_age_hours * 3600.0
    os.makedirs(DATA, exist_ok=True)
    hype_src = os.path.join(ROOT, "offline", "research", "hype_dataset", "HYPEUSDC_240m_hlspot.csv")
    if not os.path.isfile(hype_src):
        print(f"[fetch] ERROR: missing HYPE dataset at {hype_src}", file=sys.stderr, flush=True)
        sys.exit(1)
    atomic_snapshot_file(hype_src, os.path.join(DATA, "HYPE_240m.csv"))
    ok_count = sum(
        1
        for asset in ASSETS
        if fetch(
            f"{asset}USDT",
            asset,
            force=force_refresh,
            missing_only=missing_only_flag,
            max_age_sec=max_age_s,
        )
    )
    if ok_count == 0:
        print("[fetch] ERROR: failed to obtain datasets for all Binance proxy assets", file=sys.stderr, flush=True)
        sys.exit(1)
