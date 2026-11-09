#!/usr/bin/env python3
"""
market_data.py - Hyperliquid coin price and availability.
Price comes from public all_mids; available means the coin is in the perpetual universe.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from botcore import (  # noqa: E402
    load_local_module,
    log,
    safe_float,
    safe_int,
    safe_nonneg_float,
    safe_pos_float,
)
from providers.order_ohlc_utils import (  # noqa: E402
    OHLC_INTERVALS_MIN_TO_STR,
    parse_ohlc_interval,
    resolve_ohlc_time_window_ms,
    validate_ohlc_candle_bar,
)

_common = load_local_module(_HERE, "common")
_load_hl_local = _common.load_hl_local
HLClient = _load_hl_local("hl_client").HLClient

API_URL = "https://api.hyperliquid.xyz/info"
INTERVALS = {
    k: v
    for k, v in OHLC_INTERVALS_MIN_TO_STR.items()
    if k not in (360, 43200)
}
_REV_INTERVALS = {v: k for k, v in INTERVALS.items()}


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
        bar = validate_ohlc_candle_bar(
            row.get("t"),
            row.get("o"),
            row.get("h"),
            row.get("l"),
            row.get("c"),
            close_ts=row.get("T"),
            now_ms=now_ms,
            drop_incomplete=drop_incomplete,
            close_inclusive=True,
            ts_in_ms=True,
        )
        if bar is None:
            continue
        t_f = safe_pos_float(row.get("t"), default=None)
        if t_f is not None:
            by_open_time[int(t_f)] = bar
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
    _, interval_str = parse_ohlc_interval(
        interval,
        min_to_str=INTERVALS,
        str_to_min=_REV_INTERVALS,
        reject_float=True,
        venue_label="Hyperliquid",
    )
    now_ms, final_start, final_end, to_f = resolve_ohlc_time_window_ms(
        start_ms=start_ms,
        end_ms=end_ms,
        lookback_hours=lookback_hours,
        timeout=timeout,
        strict_numeric_types=True,
    )

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
