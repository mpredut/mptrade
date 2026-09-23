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

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EXP = os.environ.get("GRID_OUT") or os.path.join(ROOT, "offline", "results", "kraken_continuous_grid")
DATA = os.path.join(EXP, "data")
ASSETS = ("TAO", "ADA", "SOL", "BTC", "ETH", "XRP", "DOGE", "LINK", "AVAX", "BNB")
START_MS = 1514764800000   # 2018-01-01


def fetch(symbol: str, name: str) -> None:
    rows, start = [], START_MS
    while True:
        url = (f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=4h"
               f"&startTime={start}&limit=1000")
        batch = json.loads(urllib.request.urlopen(url, timeout=20).read())
        if not batch:
            break
        rows += batch
        start = batch[-1][0] + 1
        if len(batch) < 1000:
            break
        time.sleep(0.15)
    closed = [(int(k[0]) // 1000, k[1], k[2], k[3], k[4]) for k in rows[:-1]]   # drop the forming bar
    with open(os.path.join(DATA, f"{name}_240m.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp", "open", "high", "low", "close"])
        writer.writerows(closed)
    print(f"[fetch] {name}: {len(closed)} bars "
          f"{time.strftime('%Y-%m-%d', time.gmtime(closed[0][0]))} -> "
          f"{time.strftime('%Y-%m-%d', time.gmtime(closed[-1][0]))}", flush=True)


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    shutil.copyfile(
        os.path.join(ROOT, "offline", "research", "hype_dataset", "HYPEUSDC_240m_hlspot.csv"),
        os.path.join(DATA, "HYPE_240m.csv"),
    )
    for asset in ASSETS:
        fetch(f"{asset}USDT", asset)
