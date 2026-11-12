#!/usr/bin/env python3
"""
market_data.py - Kraken pair-price and availability tracking.

Kraken prices come directly from the public Ticker endpoint rather than Yahoo.
Available, analogous to SPCX launched, means the pair exists in AssetPairs and has
a valid price. HYPE qualifies; an unlisted future pair does not.
"""

from __future__ import annotations

import os
import sys
import urllib.parse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from botcore import (
    exact_nonneg_int,
    fetch_public_json,
    log,
    safe_nonneg_float,
    safe_pos_float,
)
from providers.ohlc_utils import parse_ohlc_interval_minutes, validate_ohlc_candle_bar
from kraken.kraken_client import KrakenClient, KrakenError


def fetch_ohlc_candles(
    pair: str,
    interval: int,
    *,
    drop_incomplete: bool = True,
    timeout: int = 25,
) -> list[dict]:
    """Fetch completed Kraken OHLC bars as canonical records.

    Excludes the final still-forming candle when drop_incomplete is True to ensure
    reproducible historical backtests and avoid lookahead bias.
    """
    if isinstance(pair, bool) or not isinstance(pair, str) or not pair.strip():
        raise ValueError(f"invalid pair: {pair!r}")
    pair = pair.strip()
    interval = parse_ohlc_interval_minutes(
        interval,
        str_to_min={},
        reject_float=True,
        venue_label="Kraken",
    )
    to_f = safe_pos_float(timeout, default=None)
    if to_f is None:
        raise ValueError(f"invalid interval={interval!r} or timeout={timeout!r}")
    qs = urllib.parse.urlencode({"pair": pair, "interval": interval})
    url = f"https://api.kraken.com/0/public/OHLC?{qs}"
    data = None
    for attempt in range(2):
        try:
            data = fetch_public_json(url, timeout=to_f, user_agent="mptrade/kraken-ohlc-fetcher")
            break
        except (TimeoutError, OSError):
            if attempt == 1:
                raise
    if not isinstance(data, dict):
        raise RuntimeError(f"invalid Kraken OHLC payload: {data!r}")
    if data.get("error"):
        err_val = data["error"]
        err_msg = ", ".join(str(x) for x in err_val) if isinstance(err_val, (list, tuple)) else str(err_val)
        raise RuntimeError(err_msg)
    res = data.get("result")
    if not isinstance(res, dict):
        return []
    key = next((k for k in res if k != "last"), None)
    if not key:
        return []
    raw = res[key]
    if not isinstance(raw, list):
        return []
    rows = raw[:-1] if drop_incomplete else raw
    records: list[dict] = []
    for x in rows:
        if isinstance(x, (list, tuple)) and len(x) >= 5:
            bar = validate_ohlc_candle_bar(
                x[0], x[1], x[2], x[3], x[4], drop_incomplete=False, ts_in_ms=False,
            )
            if bar is not None:
                records.append(bar)
    return records


def get_price(client: KrakenClient, pair: str) -> float | None:
    """Return the latest pair price, e.g. HYPEUSD, or None when unavailable."""
    try:
        return client.last_price(pair)
    except Exception as e:
        if not (isinstance(e, KrakenError) or e.__class__.__name__ == "KrakenError"):
            raise
        log(f"  ! the price of {pair} is unavailable: {e}")
        return None


TRADABLE_PAIR_STATUSES = ("online", "limit_only", "post_only")


def is_pair_status_tradable(info: dict) -> tuple[bool, str]:
    """Return ``(is_tradable, normalized_status)`` for a Kraken AssetPairs entry."""
    if not isinstance(info, dict):
        return False, "invalid"
    raw_status = info.get("status")
    if raw_status is None:
        return True, "online"
    if not isinstance(raw_status, str):
        return False, str(raw_status)
    status = raw_status.strip().lower() or "online"
    return status in TRADABLE_PAIR_STATUSES, status


def check_pair_listing(client: KrakenClient, pair: str) -> tuple[dict | None, bool]:
    """Return (pair_metadata, confirmed_unlisted).

    confirmed_unlisted is True only when Kraken explicitly reports an unknown pair
    or a non-tradable status, and False on transient network/empty responses.
    """
    try:
        info = client.pair_info(pair)
    except Exception as e:
        if not (isinstance(e, KrakenError) or e.__class__.__name__ == "KrakenError"):
            raise
        is_unknown = "unknown asset pair" in str(e).lower()
        if not is_unknown:
            log(f"  ! transient Kraken error checking {pair}: {e}")
        return None, is_unknown
    if not info:
        return None, False
    # Status 'online' means tradable; Kraken also reports cancel/post/limit/reduce-only.
    tradable, status = is_pair_status_tradable(info)
    if not tradable:
        log(f"  [market] {pair} listed but status={status} (still not tradable)")
        return None, True
    return info, False


def pair_precision(info: dict) -> tuple[int, int, float]:
    """Return (price decimals, volume decimals, minimum order) from pair metadata."""
    if not isinstance(info, dict):
        return 2, 8, 0.0
    p_val = exact_nonneg_int(info.get("pair_decimals"), default=None)
    price_dec = p_val if p_val is not None and p_val <= 18 else 2
    v_val = exact_nonneg_int(info.get("lot_decimals"), default=None)
    vol_dec = v_val if v_val is not None and v_val <= 18 else 8
    ordermin = safe_nonneg_float(info.get("ordermin", 0), default=0.0)
    return price_dec, vol_dec, ordermin
