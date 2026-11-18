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
from cachepaths import CACHE_365_PRICE_PREFIX, cache_path, is_kraken_tokenized_asset
from instrument_registry import resolve_registered_symbol_case
from providers.ohlc_utils import (
    extract_kraken_ohlc_rows,
    parse_ohlc_interval_minutes,
    require_ohlc_symbol,
    validate_ohlc_candle_bar,
)
from state_io import iter_jsonl_dicts
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
    pair = require_ohlc_symbol(pair, label="pair")
    reg_pair = resolve_registered_symbol_case(pair, "kraken")
    if reg_pair != pair and is_kraken_tokenized_asset(reg_pair):
        pair = reg_pair
    interval = parse_ohlc_interval_minutes(
        interval,
        str_to_min={},
        reject_float=True,
        venue_label="Kraken",
    )
    to_f = safe_pos_float(timeout, default=None)
    if to_f is None:
        raise ValueError(f"invalid interval={interval!r} or timeout={timeout!r}")
    q_params: dict[str, object] = {"pair": pair, "interval": interval}
    if is_kraken_tokenized_asset(pair):
        q_params["asset_class"] = "tokenized_asset"
    qs = urllib.parse.urlencode(q_params)
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
    rows = extract_kraken_ohlc_rows(data.get("result"), drop_incomplete=drop_incomplete)
    records: list[dict] = []
    for x in rows:
        if isinstance(x, (list, tuple)) and len(x) >= 5:
            bar = validate_ohlc_candle_bar(
                x[0], x[1], x[2], x[3], x[4], drop_incomplete=False, ts_in_ms=False,
            )
            if bar is not None:
                records.append(bar)
    return records


def load_archive_ohlc_candles(
    pair: str,
    interval: int,
    *,
    drop_incomplete: bool = True,
    archive_path: str | None = None,
) -> list[dict]:
    """Reconstruct completed OHLC bars offline from ``cachedb/prices/cache_365_price_<PAIR>.jsonl``."""
    pair = require_ohlc_symbol(pair, label="pair")
    reg_pair = resolve_registered_symbol_case(pair, "kraken")
    if reg_pair != pair and is_kraken_tokenized_asset(reg_pair):
        pair = reg_pair
    interval_min = parse_ohlc_interval_minutes(
        interval,
        str_to_min={},
        reject_float=True,
        venue_label="Kraken",
    )
    bucket_ms = interval_min * 60 * 1000
    path = archive_path or cache_path(f"{CACHE_365_PRICE_PREFIX}{pair}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"local tick archive not found: {path}")
    buckets: dict[int, list[float]] = {}
    for rec in iter_jsonl_dicts(path):
        item = rec.get("i")
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        ts_ms = exact_nonneg_int(item[0], default=None)
        px = safe_pos_float(item[1], default=None)
        if ts_ms is None or ts_ms <= 0 or px is None:
            continue
        b_key = (ts_ms // bucket_ms) * bucket_ms
        bar = buckets.get(b_key)
        if bar is None:
            buckets[b_key] = [px, px, px, px]
        else:
            if px > bar[1]:
                bar[1] = px
            if px < bar[2]:
                bar[2] = px
            bar[3] = px
    sorted_keys = sorted(buckets)
    if drop_incomplete and len(sorted_keys) > 1:
        sorted_keys = sorted_keys[:-1]
    out: list[dict] = []
    for b_key in sorted_keys:
        o, h, l, c = buckets[b_key]
        vbar = validate_ohlc_candle_bar(b_key // 1000, o, h, l, c, drop_incomplete=False, ts_in_ms=False)
        if vbar is not None:
            out.append(vbar)
    return out


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
