#!/usr/bin/env python3
"""Build the 4h datasets for the continuous grid: Binance USDT spot since 2018 plus HYPE.

Binance public klines serve as a price proxy for the Kraken pairs (the strategy only sees
OHLC; fees are modelled separately). HYPE comes from the frozen Hyperliquid spot dataset.
Output: $GRID_OUT/data/<ASSET>_240m.csv with timestamp,open,high,low,close.
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from botcore import optional_env
from offline.backtests.datasets import load_ohlc_tuples_csv, save_dataset, validate_dataset
from providers.market_api import api as mkt
from state_io import atomic_snapshot_file

EXP = optional_env("GRID_OUT") or os.path.join(ROOT, "offline", "results", "kraken_continuous_grid")
DATA = os.path.join(EXP, "data")
ASSETS = ("TAO", "ADA", "SOL", "BTC", "ETH", "XRP", "DOGE", "LINK", "AVAX", "BNB")
START_MS = 1514764800000   # 2018-01-01
MIN_GRID_BARS = 43


def _has_usable_csv(path: str, min_bars: int = MIN_GRID_BARS) -> bool:
    if not os.path.isfile(path):
        return False
    try:
        rows = load_ohlc_tuples_csv(path, strict=True)
    except (OSError, ValueError):
        return False
    return len(rows) >= min_bars


def fetch(symbol: str, name: str, force: bool = False) -> bool:
    out_path = os.path.join(DATA, f"{name}_240m.csv")
    if not force and _has_usable_csv(out_path):
        print(f"[fetch] {name}: using cached {out_path}", flush=True)
        return True
    try:
        candles = mkt.fetch_ohlc_candles(
            symbol, interval=240, provider_name="binance", start_ms=START_MS, drop_incomplete=True,
        )
        if not candles:
            print(f"[fetch] {name}: no candles returned; keeping existing file if present", flush=True)
            return _has_usable_csv(out_path)
        # Allow historical exchange maintenance windows while enforcing strict chronological order & OHLC invariants.
        validated = validate_dataset(candles)
        if len(validated) < MIN_GRID_BARS:
            print(
                f"[fetch] {name}: insufficient bars ({len(validated)} < {MIN_GRID_BARS}); "
                "keeping existing file if present",
                flush=True,
            )
            return _has_usable_csv(out_path)
        save_dataset(validated, out_path)
        print(f"[fetch] {name}: {len(validated)} bars "
              f"{time.strftime('%Y-%m-%d', time.gmtime(validated[0]['timestamp']))} -> "
              f"{time.strftime('%Y-%m-%d', time.gmtime(validated[-1]['timestamp']))}", flush=True)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[fetch] {name}: error {exc}; keeping existing file if present", flush=True)
        return _has_usable_csv(out_path)


if __name__ == "__main__":
    force_refresh = "--force" in sys.argv
    os.makedirs(DATA, exist_ok=True)
    hype_src = os.path.join(ROOT, "offline", "research", "hype_dataset", "HYPEUSDC_240m_hlspot.csv")
    if not os.path.isfile(hype_src):
        print(f"[fetch] ERROR: missing HYPE dataset at {hype_src}", file=sys.stderr, flush=True)
        sys.exit(1)
    atomic_snapshot_file(hype_src, os.path.join(DATA, "HYPE_240m.csv"))
    ok_count = sum(1 for asset in ASSETS if fetch(f"{asset}USDT", asset, force=force_refresh))
    if ok_count == 0:
        print("[fetch] ERROR: failed to obtain datasets for all Binance proxy assets", file=sys.stderr, flush=True)
        sys.exit(1)
