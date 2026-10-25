#!/usr/bin/env python3
"""
binance_api/market_data.py — Binance public market data and OHLC candle fetching.

Provides canonical OHLC candle fetching from the public Binance REST API
(https://api.binance.com/api/v3/klines) with pagination, cadence validation,
and incomplete bar handling for walk-forward evaluation and offline research.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from typing import Mapping

API_URL = "https://api.binance.com/api/v3/klines"

INTERVALS_MIN_TO_STR: dict[int, str] = {
    1: "1m",
    3: "3m",
    5: "5m",
    15: "15m",
    30: "30m",
    60: "1h",
    120: "2h",
    240: "4h",
    360: "6h",
    480: "8h",
    720: "12h",
    1440: "1d",
    4320: "3d",
    10080: "1w",
}

INTERVALS_STR_TO_MIN: dict[str, int] = {
    v: k for k, v in INTERVALS_MIN_TO_STR.items()
}
# Additional standard aliases
INTERVALS_STR_TO_MIN["1d"] = 1440
INTERVALS_STR_TO_MIN["1w"] = 10080
INTERVALS_STR_TO_MIN["1M"] = 43200


def parse_interval_minutes(interval: int | str) -> int:
    """Convert an interval in minutes or string notation (e.g. '1h', 60) into minutes."""
    if isinstance(interval, str):
        cleaned = interval.strip()
        if cleaned in INTERVALS_STR_TO_MIN:
            return INTERVALS_STR_TO_MIN[cleaned]
        if cleaned.isdigit():
            val = int(cleaned)
            if val <= 0:
                raise ValueError(f"interval must be positive: {interval}")
            return val
        raise ValueError(f"unsupported Binance interval: {interval}")
    val = int(interval)
    if val <= 0:
        raise ValueError(f"interval must be positive: {interval}")
    return val


def interval_to_binance_str(interval: int | str) -> str:
    """Convert an interval in minutes or string notation into Binance interval format."""
    if isinstance(interval, str) and interval in INTERVALS_STR_TO_MIN:
        return interval
    minutes = parse_interval_minutes(interval)
    if minutes in INTERVALS_MIN_TO_STR:
        return INTERVALS_MIN_TO_STR[minutes]
    raise ValueError(f"unsupported Binance interval cadence: {interval} ({minutes}m)")


def fetch_ohlc_candles(
    symbol: str,
    interval: int | str = 60,
    *,
    lookback_days: int | None = None,
    start_ms: int | None = None,
    end_ms: int | None = None,
    limit_per_batch: int = 1000,
    drop_incomplete: bool = True,
    timeout: int = 25,
) -> list[dict]:
    """Fetch completed Binance OHLC candles as canonical records.

    Paginates across the requested time window up to `end_ms` or the current timestamp.
    Drops the still-forming bar if `drop_incomplete` is True.

    Returns:
        list[dict] with keys: 'timestamp' (seconds), 'open', 'high', 'low', 'close'
    """
    sym = symbol.strip().upper()
    interval_str = interval_to_binance_str(interval)
    interval_min = parse_interval_minutes(interval)
    interval_ms = interval_min * 60 * 1000

    now_ms = int(time.time() * 1000)
    final_end = end_ms if end_ms is not None else now_ms
    if start_ms is not None:
        final_start = start_ms
    elif lookback_days is not None:
        final_start = final_end - lookback_days * 86400 * 1000
    else:
        # Default to ~30 days of history
        final_start = final_end - 30 * 86400 * 1000

    if final_start >= final_end:
        raise ValueError(f"start_ms ({final_start}) must be before end_ms ({final_end})")

    records: list[dict] = []
    seen_timestamps: set[int] = set()
    current_start = final_start
    batch_limit = min(1000, max(1, int(limit_per_batch)))

    while current_start < final_end:
        params = {
            "symbol": sym,
            "interval": interval_str,
            "startTime": current_start,
            "endTime": final_end,
            "limit": batch_limit,
        }
        url = f"{API_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "mptrade/binance-ohlc-fetcher",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = json.loads(resp.read().decode("utf-8"))

        if not isinstance(raw, list) or not raw:
            break

        for item in raw:
            # Binance kline record layout:
            # 0: Open time (ms)
            # 1: Open
            # 2: High
            # 3: Low
            # 4: Close
            # 5: Volume
            # 6: Close time (ms)
            # 7: Quote asset volume
            # 8: Number of trades
            # 9: Taker buy base asset volume
            # 10: Taker buy quote asset volume
            # 11: Ignore
            open_time_ms = int(item[0])
            close_time_ms = int(item[6])
            ts_sec = open_time_ms // 1000

            # If drop_incomplete is True and the candle has not closed yet, skip it
            if drop_incomplete and close_time_ms > now_ms:
                continue

            if ts_sec in seen_timestamps:
                continue
            seen_timestamps.add(ts_sec)

            records.append({
                "timestamp": ts_sec,
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
            })

        last_open_time = int(raw[-1][0])
        next_start = last_open_time + interval_ms
        if next_start <= current_start or len(raw) < batch_limit:
            break
        current_start = next_start
        # Respect rate limits for long paginations
        time.sleep(0.05)

    records.sort(key=lambda r: r["timestamp"])
    return records
