#!/usr/bin/env python3
"""
binance_api/market_data.py — Binance public market data and OHLC candle fetching.

Provides canonical OHLC candle fetching from the public Binance REST API
(https://api.binance.com/api/v3/klines) with pagination, cadence validation,
and incomplete bar handling for walk-forward evaluation and offline research.
"""

from __future__ import annotations

import time
import urllib.parse

from botcore import _parse_bool_flag, fetch_public_json, safe_pos_float
from providers.order_ohlc_utils import (
    OHLC_CASE_INSENSITIVE_INTERVAL_ALIASES,
    OHLC_INTERVALS_MIN_TO_STR,
    OHLC_INTERVALS_STR_TO_MIN,
    parse_ohlc_interval,
    parse_ohlc_interval_minutes,
    resolve_ohlc_time_window_ms,
    validate_ohlc_candle_bar,
)

API_URL = "https://api.binance.com/api/v3/klines"

INTERVALS_MIN_TO_STR: dict[int, str] = dict(OHLC_INTERVALS_MIN_TO_STR)
INTERVALS_STR_TO_MIN: dict[str, int] = dict(OHLC_INTERVALS_STR_TO_MIN)
_CASE_INSENSITIVE_INTERVAL_ALIASES: dict[str, str] = dict(OHLC_CASE_INSENSITIVE_INTERVAL_ALIASES)


def parse_interval_minutes(interval: int | str) -> int:
    """Convert an interval in minutes or string notation (e.g. '1h', 60) into minutes."""
    return parse_ohlc_interval_minutes(
        interval,
        str_to_min=INTERVALS_STR_TO_MIN,
        venue_label="Binance",
    )


def interval_to_binance_str(interval: int | str) -> str:
    """Convert an interval in minutes or string notation into Binance interval format."""
    _, interval_str = parse_ohlc_interval(
        interval,
        min_to_str=INTERVALS_MIN_TO_STR,
        str_to_min=INTERVALS_STR_TO_MIN,
        venue_label="Binance",
    )
    return interval_str


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
    interval_min, interval_str = parse_ohlc_interval(
        interval,
        min_to_str=INTERVALS_MIN_TO_STR,
        str_to_min=INTERVALS_STR_TO_MIN,
        venue_label="Binance",
    )
    interval_ms = interval_min * 60 * 1000
    now_ms, final_start, final_end, timeout_f = resolve_ohlc_time_window_ms(
        start_ms=start_ms,
        end_ms=end_ms,
        lookback_days=lookback_days,
        timeout=timeout,
    )
    batch_f = safe_pos_float(limit_per_batch, default=None)
    if batch_f is None or int(batch_f) <= 0:
        raise ValueError(f"limit_per_batch must be positive: {limit_per_batch}")
    batch_limit_raw = int(batch_f)

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
        raw = fetch_public_json(
            url,
            timeout=timeout_f,
            user_agent="mptrade/binance-ohlc-fetcher",
            headers={"Accept": "application/json"},
        )

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
            bar = validate_ohlc_candle_bar(
                item[0],
                item[1],
                item[2],
                item[3],
                item[4],
                close_ts=item[6],
                now_ms=now_ms,
                drop_incomplete=should_drop_incomplete,
                close_inclusive=False,
                ts_in_ms=True,
            )
            if bar is None or bar["timestamp"] in seen_timestamps:
                continue
            seen_timestamps.add(bar["timestamp"])
            records.append(bar)

        last_open_time = next(
            (
                int(cand_ot)
                for item in reversed(raw)
                if isinstance(item, (list, tuple)) and item and (cand_ot := safe_pos_float(item[0], default=None)) is not None
            ),
            None,
        )
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
