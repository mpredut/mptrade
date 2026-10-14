#!/usr/bin/env python3
"""
market_data.py — Hyperliquid coin price and availability.
Price comes from public all_mids; available means the coin is in the perpetual universe.
"""

from __future__ import annotations

import json
import time
import urllib.request

try:
    from hyperliquid.common import log
    from hyperliquid.hl_client import HLClient
except ImportError:
    from common import log
    from hl_client import HLClient

API_URL = "https://api.hyperliquid.xyz/info"
INTERVALS = {
    1: "1m",
    3: "3m",
    5: "5m",
    15: "15m",
    30: "30m",
    60: "1h",
    120: "2h",
    240: "4h",
    480: "8h",
    720: "12h",
    1440: "1d",
    4320: "3d",
    10080: "1w",
}


def _post_info(body: dict, timeout: int = 25) -> object:
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "mptrade/hyperliquid-market-data",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def fetch_ohlc_candles(
    coin: str,
    interval: int | str = 60,
    *,
    lookback_hours: int | None = None,
    start_ms: int | None = None,
    end_ms: int | None = None,
    drop_incomplete: bool = True,
    timeout: int = 25,
) -> list[dict]:
    """Fetch Hyperliquid completed OHLC candles as canonical records.

    Interval can be integer minutes (e.g. 60) or string (e.g. '1h').
    Returns a list of dicts:
        {"timestamp": int, "open": float, "high": float, "low": float, "close": float}
    """
    if isinstance(interval, str):
        rev_intervals = {v: k for k, v in INTERVALS.items()}
        interval_str = interval
        interval_min = rev_intervals.get(interval)
        if interval_min is None:
            raise ValueError(f"unsupported Hyperliquid interval: {interval}")
    else:
        interval_min = int(interval)
        interval_str = INTERVALS.get(interval_min)
        if interval_str is None:
            raise ValueError(f"unsupported Hyperliquid interval: {interval} minutes")

    now_ms = int(time.time() * 1000)
    final_end = end_ms if end_ms is not None else now_ms
    if start_ms is not None:
        final_start = start_ms
    elif lookback_hours is not None:
        final_start = final_end - lookback_hours * 3600 * 1000
    else:
        final_start = final_end - 30 * 24 * 3600 * 1000

    payload = _post_info({
        "type": "candleSnapshot",
        "req": {
            "coin": coin,
            "interval": interval_str,
            "startTime": final_start,
            "endTime": final_end,
        },
    }, timeout=timeout)
    if not isinstance(payload, list):
        raise RuntimeError(f"invalid candleSnapshot response: {payload!r}")

    by_open: dict[int, dict] = {}
    for row in payload:
        open_time = int(row["t"])
        close_time = int(row["T"])
        if drop_incomplete and close_time >= now_ms:
            continue
        by_open[open_time] = {
            "timestamp": open_time // 1000,
            "open": float(row["o"]),
            "high": float(row["h"]),
            "low": float(row["l"]),
            "close": float(row["c"]),
        }
    return [by_open[k] for k in sorted(by_open)]


def fetch_funding_history(coin: str, start_ms: int, timeout: int = 25) -> list[tuple[int, float]]:
    """Return hourly coin funding history sorted as (time_ms, funding_rate) tuples."""
    try:
        data = _post_info({"type": "fundingHistory", "coin": coin, "startTime": start_ms}, timeout=timeout)
        if not isinstance(data, list):
            return []
        return sorted(((int(d["time"]), float(d["fundingRate"])) for d in data), key=lambda x: x[0])
    except Exception as e:  # noqa: BLE001
        log(f"  ! funding history unavailable for {coin}: {e}")
        return []


def get_price(client: HLClient, coin: str) -> float | None:
    return client.mid(coin)


def coin_available(client: HLClient, coin: str) -> bool:
    try:
        return client.coin_listed(coin) and client.mid(coin) is not None
    except Exception as e:  # noqa: BLE001
        log(f"  ! coin_available({coin}) failed: {e}")
        return False
