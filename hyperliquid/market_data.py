#!/usr/bin/env python3
"""
market_data.py — Hyperliquid coin price and availability.
Price comes from public all_mids; available means the coin is in the perpetual universe.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

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


def normalize_closed_candles(
    rows: list[dict], *, now_ms: int, drop_incomplete: bool = True
) -> list[dict]:
    """Sort, deduplicate and drop the candle that is still open."""
    by_open_time: dict[int, dict] = {}
    for row in (rows or []):
        if not isinstance(row, dict):
            continue
        if any(isinstance(row.get(k), bool) for k in ("t", "T", "o", "h", "l", "c")):
            continue
        try:
            t_f = float(row["t"])
            T_f = float(row["T"])
            o_val = float(row["o"])
            h_val = float(row["h"])
            l_val = float(row["l"])
            c_val = float(row["c"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if not (
            math.isfinite(t_f)
            and math.isfinite(T_f)
            and math.isfinite(o_val)
            and math.isfinite(h_val)
            and math.isfinite(l_val)
            and math.isfinite(c_val)
        ):
            continue
        open_time = int(t_f)
        close_time = int(T_f)
        if open_time <= 0 or close_time < open_time:
            continue
        if o_val <= 0 or h_val <= 0 or l_val <= 0 or c_val <= 0:
            continue
        if h_val < l_val or h_val < max(o_val, c_val) or l_val > min(o_val, c_val):
            continue
        if drop_incomplete and close_time >= now_ms:
            continue
        by_open_time[open_time] = {
            "timestamp": open_time // 1000,
            "open": o_val,
            "high": h_val,
            "low": l_val,
            "close": c_val,
        }
    return [by_open_time[key] for key in sorted(by_open_time)]


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
    if isinstance(coin, bool) or not isinstance(coin, str) or not coin.strip():
        raise ValueError(f"invalid coin: {coin!r}")
    coin = coin.strip()
    if isinstance(interval, str):
        rev_intervals = {v: k for k, v in INTERVALS.items()}
        cleaned = interval.strip()
        if cleaned in rev_intervals:
            interval_str = cleaned
            interval_min = rev_intervals[cleaned]
        elif cleaned.isascii() and cleaned.isdigit():
            interval_min = int(cleaned)
            interval_str = INTERVALS.get(interval_min)
            if interval_str is None:
                raise ValueError(f"unsupported Hyperliquid interval: {interval}")
        else:
            raise ValueError(f"unsupported Hyperliquid interval: {interval}")
    else:
        if isinstance(interval, bool):
            raise ValueError(f"unsupported Hyperliquid interval: {interval!r}")
        try:
            interval_f = float(interval)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"unsupported Hyperliquid interval: {interval!r}") from exc
        if not math.isfinite(interval_f) or not interval_f.is_integer():
            raise ValueError(f"unsupported Hyperliquid interval: {interval!r}")
        interval_min = int(interval_f)
        interval_str = INTERVALS.get(interval_min)
        if interval_str is None:
            raise ValueError(f"unsupported Hyperliquid interval: {interval} minutes")

    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(float(timeout)) or float(timeout) <= 0:
        raise ValueError(f"invalid timeout: {timeout!r}")

    now_ms = int(time.time() * 1000)
    if end_ms is not None:
        if isinstance(end_ms, bool) or not isinstance(end_ms, (int, float)) or not math.isfinite(float(end_ms)) or int(end_ms) <= 0:
            raise ValueError(f"invalid end_ms: {end_ms!r}")
        final_end = int(end_ms)
    else:
        final_end = now_ms
    if start_ms is not None:
        if isinstance(start_ms, bool) or not isinstance(start_ms, (int, float)) or not math.isfinite(float(start_ms)) or int(start_ms) < 0:
            raise ValueError(f"invalid start_ms: {start_ms!r}")
        final_start = int(start_ms)
    elif lookback_hours is not None:
        if (
            isinstance(lookback_hours, bool)
            or not isinstance(lookback_hours, (int, float))
            or not math.isfinite(float(lookback_hours))
            or float(lookback_hours) <= 0
        ):
            raise ValueError(f"invalid lookback_hours: {lookback_hours!r}")
        final_start = int(final_end - float(lookback_hours) * 3600 * 1000)
    else:
        final_start = final_end - 30 * 24 * 3600 * 1000
    if final_start >= final_end:
        raise ValueError(f"start_ms ({final_start}) must be before end_ms ({final_end})")

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

    return normalize_closed_candles(payload, now_ms=now_ms, drop_incomplete=drop_incomplete)


def fetch_funding_history(coin: str, start_ms: int, timeout: int = 25) -> list[tuple[int, float]]:
    """Return hourly coin funding history sorted as (time_ms, funding_rate) tuples."""
    if isinstance(coin, bool) or not isinstance(coin, str) or not coin.strip():
        return []
    if isinstance(start_ms, bool) or not isinstance(start_ms, (int, float)) or not math.isfinite(float(start_ms)) or int(start_ms) < 0:
        return []
    try:
        data = _post_info({"type": "fundingHistory", "coin": coin.strip(), "startTime": int(start_ms)}, timeout=timeout)
        if not isinstance(data, list):
            return []
        entries: list[tuple[int, float]] = []
        for d in data:
            if not isinstance(d, dict):
                continue
            if isinstance(d.get("time"), bool) or isinstance(d.get("fundingRate"), bool):
                continue
            try:
                t_f = float(d["time"])
                rate = float(d["fundingRate"])
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(t_f) and int(t_f) > 0 and math.isfinite(rate):
                entries.append((int(t_f), rate))
        return sorted(entries, key=lambda x: x[0])
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
