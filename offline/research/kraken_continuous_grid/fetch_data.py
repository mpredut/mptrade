#!/usr/bin/env python3
"""Build the 4h datasets for the continuous grid: Binance USDT spot since 2018 plus HYPE.

Binance public klines serve as a price proxy for the Kraken pairs (the strategy only sees
OHLC; fees are modelled separately). HYPE comes from the frozen Hyperliquid spot dataset.
Output: $GRID_OUT/data/<ASSET>_240m.csv with timestamp,open,high,low,close.
"""
import csv
import json
import os
import shutil
import time
import urllib.request

import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from binance_api.market_data import fetch_ohlc_candles

EXP = os.environ.get("GRID_OUT") or os.path.join(ROOT, "offline", "results", "kraken_continuous_grid")
DATA = os.path.join(EXP, "data")
ASSETS = ("TAO", "ADA", "SOL", "BTC", "ETH", "XRP", "DOGE", "LINK", "AVAX", "BNB")
START_MS = 1514764800000   # 2018-01-01


def fetch(symbol: str, name: str, force: bool = False) -> None:
    out_path = os.path.join(DATA, f"{name}_240m.csv")
    if not force and os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
        print(f"[fetch] {name}: using cached {out_path}", flush=True)
        return
    candles = fetch_ohlc_candles(symbol, interval=240, start_ms=START_MS, drop_incomplete=True)
    if not candles:
        print(f"[fetch] {name}: no candles returned; keeping existing file if present", flush=True)
        return
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp", "open", "high", "low", "close"])
        for c in candles:
            writer.writerow([c["timestamp"], c["open"], c["high"], c["low"], c["close"]])
    print(f"[fetch] {name}: {len(candles)} bars "
          f"{time.strftime('%Y-%m-%d', time.gmtime(candles[0]['timestamp']))} -> "
          f"{time.strftime('%Y-%m-%d', time.gmtime(candles[-1]['timestamp']))}", flush=True)


if __name__ == "__main__":
    force_refresh = "--force" in sys.argv
    os.makedirs(DATA, exist_ok=True)
    shutil.copyfile(
        os.path.join(ROOT, "offline", "research", "hype_dataset", "HYPEUSDC_240m_hlspot.csv"),
        os.path.join(DATA, "HYPE_240m.csv"),
    )
    for asset in ASSETS:
        fetch(f"{asset}USDT", asset, force=force_refresh)
