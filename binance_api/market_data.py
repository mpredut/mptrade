#!/usr/bin/env python3
"""
binance_api/market_data.py — Binance public market data and OHLC candle fetching.

Provides canonical OHLC candle fetching from the public Binance REST API
(https://api.binance.com/api/v3/klines) with pagination, cadence validation,
and incomplete bar handling for walk-forward evaluation and offline research.
"""

from __future__ import annotations

import json
import math
import time
import urllib.parse
import urllib.request
from typing import Mapping

from botcore import _parse_bool_flag, safe_pos_float

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
    43200: "1M",
}

INTERVALS_STR_TO_MIN: dict[str, int] = {
    v: k for k, v in INTERVALS_MIN_TO_STR.items()
}
# Additional standard aliases
INTERVALS_STR_TO_MIN["1d"] = 1440
INTERVALS_STR_TO_MIN["1w"] = 10080
INTERVALS_STR_TO_MIN["1M"] = 43200

_CASE_INSENSITIVE_INTERVAL_ALIASES: dict[str, str] = {
    "1H": "1h",
    "2H": "2h",
    "4H": "4h",
    "6H": "6h",
    "8H": "8h",
    "12H": "12h",
    "1D": "1d",
    "3D": "3d",
    "1W": "1w",
}


def parse_interval_minutes(interval: int | str) -> int:
    """Convert an interval in minutes or string notation (e.g. '1h', 60) into minutes."""
    if isinstance(interval, bool):
        raise ValueError(f"interval must be positive integer or string, not bool: {interval}")
    if isinstance(interval, str):
        cleaned = interval.strip()
        cleaned = _CASE_INSENSITIVE_INTERVAL_ALIASES.get(cleaned.upper(), cleaned)
        if cleaned in INTERVALS_STR_TO_MIN:
            return INTERVALS_STR_TO_MIN[cleaned]
        if cleaned.isascii() and cleaned.isdigit():
            val = int(cleaned)
            if val <= 0:
                raise ValueError(f"interval must be positive: {interval}")
            return val
        raise ValueError(f"unsupported Binance interval: {interval}")
    if isinstance(interval, float) and (not math.isfinite(interval) or not interval.is_integer()):
        raise ValueError(f"interval must be a finite integer number of minutes: {interval}")
    try:
        val = int(interval)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"interval must be positive integer: {interval}") from exc
    if val <= 0:
        raise ValueError(f"interval must be positive: {interval}")
    return val


def interval_to_binance_str(interval: int | str) -> str:
    """Convert an interval in minutes or string notation into Binance interval format."""
    if isinstance(interval, str):
        cleaned = interval.strip()
        cleaned = _CASE_INSENSITIVE_INTERVAL_ALIASES.get(cleaned.upper(), cleaned)
        if cleaned in INTERVALS_STR_TO_MIN:
            return cleaned
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
    if isinstance(symbol, bool) or not isinstance(symbol, str) or not symbol.strip():
        raise ValueError(f"symbol must be a non-empty string: {symbol!r}")
    sym = symbol.strip().upper()
    interval_str = interval_to_binance_str(interval)
    interval_min = parse_interval_minutes(interval)
    interval_ms = interval_min * 60 * 1000

    now_ms = int(time.time() * 1000)
    if (
        isinstance(end_ms, bool)
        or isinstance(start_ms, bool)
        or isinstance(lookback_days, bool)
        or isinstance(limit_per_batch, bool)
        or isinstance(timeout, bool)
    ):
        raise ValueError("OHLC time window parameters must not be boolean")
    try:
        timeout_f = float(timeout)
        if not math.isfinite(timeout_f) or timeout_f <= 0:
            raise ValueError(f"timeout must be finite and positive: {timeout}")
        if end_ms is not None:
            end_f = float(end_ms)
            if not math.isfinite(end_f):
                raise ValueError(f"end_ms must be finite: {end_ms}")
            final_end = int(end_f)
        else:
            final_end = now_ms
        if final_end <= 0:
            raise ValueError(f"end_ms must be positive: {end_ms}")
        if start_ms is not None:
            start_f = float(start_ms)
            if not math.isfinite(start_f):
                raise ValueError(f"start_ms must be finite: {start_ms}")
            final_start = int(start_f)
            if final_start < 0:
                raise ValueError(f"start_ms must be non-negative: {start_ms}")
        elif lookback_days is not None:
            lb_f = float(lookback_days)
            if not math.isfinite(lb_f):
                raise ValueError(f"lookback_days must be finite: {lookback_days}")
            lb_days = int(lb_f)
            if lb_days <= 0:
                raise ValueError(f"lookback_days must be positive: {lookback_days}")
            final_start = final_end - lb_days * 86400 * 1000
        else:
            # Default to ~30 days of history
            final_start = final_end - 30 * 86400 * 1000
        batch_f = float(limit_per_batch)
        if not math.isfinite(batch_f):
            raise ValueError(f"limit_per_batch must be finite: {limit_per_batch}")
        batch_limit_raw = int(batch_f)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid OHLC window parameters: {exc}") from exc
    if batch_limit_raw <= 0:
        raise ValueError(f"limit_per_batch must be positive: {limit_per_batch}")

    if final_start >= final_end:
        raise ValueError(f"start_ms ({final_start}) must be before end_ms ({final_end})")

    records: list[dict] = []
    seen_timestamps: set[int] = set()
    current_start = final_start
    batch_limit = min(1000, max(1, batch_limit_raw))
    should_drop_incomplete = _parse_bool_flag(drop_incomplete, default=True)

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
        with urllib.request.urlopen(req, timeout=timeout_f) as resp:
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
            if not isinstance(item, (list, tuple)) or len(item) < 7:
                continue
            ot_f = safe_pos_float(item[0], default=None)
            ct_f = safe_pos_float(item[6], default=None)
            o_v = safe_pos_float(item[1], default=None)
            h_v = safe_pos_float(item[2], default=None)
            l_v = safe_pos_float(item[3], default=None)
            c_v = safe_pos_float(item[4], default=None)
            if (
                ot_f is None
                or ct_f is None
                or o_v is None
                or h_v is None
                or l_v is None
                or c_v is None
            ):
                continue
            open_time_ms = int(ot_f)
            close_time_ms = int(ct_f)
            if open_time_ms <= 0 or close_time_ms <= 0:
                continue
            if h_v < l_v or h_v < max(o_v, c_v) or l_v > min(o_v, c_v):
                continue
            ts_sec = open_time_ms // 1000

            # If drop_incomplete is True and the candle has not closed yet, skip it
            if should_drop_incomplete and close_time_ms > now_ms:
                continue

            if ts_sec in seen_timestamps:
                continue
            seen_timestamps.add(ts_sec)

            records.append({
                "timestamp": ts_sec,
                "open": o_v,
                "high": h_v,
                "low": l_v,
                "close": c_v,
            })

        last_open_time = None
        for item in reversed(raw):
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                cand_ot = safe_pos_float(item[0], default=None)
                if cand_ot is not None:
                    last_open_time = int(cand_ot)
                    break
        if last_open_time is None:
            break
        next_start = last_open_time + interval_ms
        if next_start <= current_start or len(raw) < batch_limit:
            break
        current_start = next_start
        # Respect rate limits for long paginations
        time.sleep(0.05)

    records.sort(key=lambda r: r["timestamp"])
    return records
