#!/usr/bin/env python3
"""Build the 4h datasets for the continuous grid: Binance USDT spot since 2018 plus HYPE.

Binance public klines serve as a price proxy for the Kraken pairs (the strategy only sees
OHLC; fees are modelled separately). HYPE comes from the frozen Hyperliquid spot dataset.
Output: $GRID_OUT/data/<ASSET>_240m.csv with timestamp,open,high,low,close.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from botcore import optional_env
from offline.backtests.datasets import (
    DAILY_DATASET_MAX_AGE_SEC,
    has_usable_ohlc_csv,
    parse_ohlc_refresh_cli_args,
    refresh_ohlc_csv_dataset,
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
    return refresh_ohlc_csv_dataset(
        os.path.join(DATA, f"{name}_240m.csv"),
        lambda: mkt.fetch_ohlc_candles(
            symbol,
            interval=240,
            provider_name="binance",
            start_ms=START_MS,
            drop_incomplete=True,
        ),
        label=name,
        min_bars=MIN_GRID_BARS,
        strict=True,
        force=force,
        missing_only=missing_only,
        max_age_sec=max_age_sec,
        min_timestamp=START_MS // 1000,
    )


if __name__ == "__main__":
    force_refresh, missing_only_flag, max_age_s = parse_ohlc_refresh_cli_args(__doc__)
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
