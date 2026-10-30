#!/usr/bin/env python3
"""
market_data.py — Hyperliquid coin price and availability.
Price comes from public all_mids; available means the coin is in the perpetual universe.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


from botcore import load_local_module  # noqa: E402

_common = load_local_module(_HERE, "common")
_load_hl_local = _common.load_hl_local
log = _common.log
safe_float = _common.safe_float
safe_pos_float = _common.safe_pos_float
safe_nonneg_float = _common.safe_nonneg_float
safe_int = _common.safe_int
exact_pos_int = _common.exact_pos_int
HLClient = _load_hl_local("hl_client").HLClient

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
    if not isinstance(rows, list):
        return []
    if drop_incomplete:
        now_i = safe_int(now_ms, default=None)
        if now_i is None or not isinstance(now_ms, (int, float)):
            return []
        now_ms = now_i
    by_open_time: dict[int, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        t_f = safe_pos_float(row.get("t"), default=None)
        T_f = safe_pos_float(row.get("T"), default=None)
        o_val = safe_pos_float(row.get("o"), default=None)
        h_val = safe_pos_float(row.get("h"), default=None)
        l_val = safe_pos_float(row.get("l"), default=None)
        c_val = safe_pos_float(row.get("c"), default=None)
        if (
            t_f is None
            or T_f is None
            or o_val is None
            or h_val is None
            or l_val is None
            or c_val is None
        ):
            continue
        open_time = int(t_f)
        close_time = int(T_f)
        if open_time <= 0 or close_time < open_time:
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
        cleaned_norm = cleaned.lower() if cleaned.lower() in rev_intervals else cleaned
        if cleaned_norm in rev_intervals:
            interval_str = cleaned_norm
            interval_min = rev_intervals[cleaned_norm]
        elif cleaned.isascii() and cleaned.isdigit():
            interval_min = int(cleaned)
            interval_str = INTERVALS.get(interval_min)
            if interval_str is None:
                raise ValueError(f"unsupported Hyperliquid interval: {interval}")
        else:
            raise ValueError(f"unsupported Hyperliquid interval: {interval}")
    else:
        interval_min = None if isinstance(interval, float) else exact_pos_int(interval, default=None)
        if interval_min is None:
            raise ValueError(f"unsupported Hyperliquid interval: {interval!r}")
        interval_str = INTERVALS.get(interval_min)
        if interval_str is None:
            raise ValueError(f"unsupported Hyperliquid interval: {interval} minutes")

    to_f = safe_pos_float(timeout, default=None)
    if to_f is None or not isinstance(timeout, (int, float)):
        raise ValueError(f"invalid timeout: {timeout!r}")

    now_ms = int(time.time() * 1000)
    if end_ms is not None:
        end_f = safe_pos_float(end_ms, default=None)
        if end_f is None or not isinstance(end_ms, (int, float)):
            raise ValueError(f"invalid end_ms: {end_ms!r}")
        final_end = int(end_f)
    else:
        final_end = now_ms
    if start_ms is not None:
        start_f = safe_nonneg_float(start_ms, default=None)
        if start_f is None or not isinstance(start_ms, (int, float)):
            raise ValueError(f"invalid start_ms: {start_ms!r}")
        final_start = int(start_f)
    elif lookback_hours is not None:
        lb_f = safe_pos_float(lookback_hours, default=None)
        if lb_f is None or not isinstance(lookback_hours, (int, float)):
            raise ValueError(f"invalid lookback_hours: {lookback_hours!r}")
        final_start = int(final_end - lb_f * 3600 * 1000)
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
    }, timeout=to_f)
    if not isinstance(payload, list):
        raise RuntimeError(f"invalid candleSnapshot response: {payload!r}")

    return normalize_closed_candles(payload, now_ms=now_ms, drop_incomplete=drop_incomplete)


def fetch_funding_history(coin: str, start_ms: int, timeout: int = 25) -> list[tuple[int, float]]:
    """Return hourly coin funding history sorted as (time_ms, funding_rate) tuples."""
    if isinstance(coin, bool) or not isinstance(coin, str) or not coin.strip():
        return []
    if not isinstance(start_ms, (int, float)) or not isinstance(timeout, (int, float)):
        return []
    start_f = safe_nonneg_float(start_ms, default=None)
    timeout_f = safe_pos_float(timeout, default=None)
    if start_f is None or int(start_f) < 0 or timeout_f is None:
        return []
    try:
        data = _post_info({"type": "fundingHistory", "coin": coin.strip(), "startTime": int(start_f)}, timeout=timeout_f)
        if not isinstance(data, list):
            return []
        entries: list[tuple[int, float]] = []
        for d in data:
            if not isinstance(d, dict):
                continue
            t_f = safe_pos_float(d.get("time"), default=None)
            rate = safe_float(d.get("fundingRate"), default=None)
            if t_f is not None and int(t_f) > 0 and rate is not None:
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
