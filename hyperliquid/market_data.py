#!/usr/bin/env python3
"""
market_data.py - Hyperliquid coin price and availability.
Price comes from public all_mids; available means the coin is in the perpetual universe.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from botcore import (  # noqa: E402
    _parse_bool_flag,
    fetch_json_post,
    load_local_module,
    log,
    safe_float,
    safe_int,
    safe_nonneg_float,
    safe_pos_float,
)
from providers.ohlc_utils import (  # noqa: E402
    OHLC_INTERVALS_MIN_TO_STR,
    parse_ohlc_interval,
    resolve_ohlc_time_window_ms,
    validate_ohlc_candle_bar,
)

load_local_module(_HERE, "common")
HLClient = load_local_module(_HERE, "hl_client").HLClient

API_URL = "https://api.hyperliquid.xyz/info"
INTERVALS = {
    k: v
    for k, v in OHLC_INTERVALS_MIN_TO_STR.items()
    if k not in (360, 43200)
}
_REV_INTERVALS = {v: k for k, v in INTERVALS.items()}


def _post_info(body: dict, timeout: int = 25) -> object:
    return fetch_json_post(
        API_URL,
        body,
        timeout=timeout,
        headers={"User-Agent": "mptrade/hyperliquid-market-data"},
    )


def normalize_closed_candles(
    rows: list[dict], *, now_ms: int, drop_incomplete: bool = True
) -> list[dict]:
    """Sort, deduplicate and drop the candle that is still open."""
    if not isinstance(rows, list):
        return []
    should_drop = _parse_bool_flag(drop_incomplete, True)
    if should_drop:
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
            drop_incomplete=should_drop,
            close_inclusive=True,
            ts_in_ms=True,
        )
        if bar is None:
            continue
        t_f = safe_pos_float(row.get("t"), default=None)
        if t_f is not None:
            t_i = safe_int(round(t_f), default=None)
            if t_i is not None and t_i > 0:
                by_open_time[t_i] = bar
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


def fetch_funding_history(
    coin: str,
    start_ms: int,
    timeout: int = 25,
    *,
    end_ms: int | None = None,
) -> list[tuple[int, float]]:
    """Return hourly coin funding history sorted as (time_ms, funding_rate) tuples."""
    if isinstance(coin, bool) or not isinstance(coin, str) or not coin.strip():
        return []
    if (
        isinstance(start_ms, bool)
        or isinstance(timeout, bool)
        or not isinstance(start_ms, (int, float))
        or not isinstance(timeout, (int, float))
    ):
        return []
    start_f = safe_nonneg_float(start_ms, default=None)
    timeout_f = safe_pos_float(timeout, default=None)
    start_i = safe_int(round(start_f), default=None) if start_f is not None else None
    if start_i is None or start_i < 0 or timeout_f is None:
        return []
    end_i: int | None = None
    if end_ms is not None:
        if isinstance(end_ms, bool) or not isinstance(end_ms, (int, float)):
            return []
        end_f = safe_nonneg_float(end_ms, default=None)
        end_i = safe_int(round(end_f), default=None) if end_f is not None else None
        if end_i is None or end_i < start_i:
            return []
    coin_clean = coin.strip()
    by_ts: dict[int, float] = {}
    cur_start = start_i
    try:
        for _ in range(60):
            req_body: dict[str, object] = {
                "type": "fundingHistory",
                "coin": coin_clean,
                "startTime": cur_start,
            }
            if end_i is not None:
                req_body["endTime"] = end_i
            data = _post_info(req_body, timeout=timeout_f)
            if not isinstance(data, list) or not data:
                break
            max_t = -1
            for d in data:
                if not isinstance(d, dict):
                    continue
                t_f = safe_pos_float(d.get("time"), default=None)
                rate = safe_float(d.get("fundingRate"), default=None)
                t_i = safe_int(round(t_f), default=None) if t_f is not None else None
                if t_i is not None and t_i > 0 and rate is not None:
                    if end_i is None or t_i <= end_i:
                        by_ts[t_i] = rate
                    if t_i > max_t:
                        max_t = t_i
            if len(data) < 500 or max_t < cur_start:
                break
            next_start = max_t + 1
            if end_i is not None and next_start > end_i:
                break
            cur_start = next_start
        return [(t_k, by_ts[t_k]) for t_k in sorted(by_ts)]
    except Exception as e:  # noqa: BLE001
        log(f"  ! funding history unavailable for {coin}: {e}")
        return []


def get_price(client: HLClient, coin: str) -> float | None:
    return client.mid(coin)
