#!/usr/bin/env python3
"""
market_data.py — Kraken pair-price and availability tracking.

Kraken prices come directly from the public Ticker endpoint rather than Yahoo.
Available, analogous to SPCX launched, means the pair exists in AssetPairs and has
a valid price. HYPE qualifies; an unlisted future pair does not.
"""

from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request

try:
    from kraken.kraken_common import log, safe_int, safe_nonneg_float, safe_pos_float
    from kraken.kraken_client import KrakenClient, KrakenError
except ImportError:
    from kraken_common import log, safe_int, safe_nonneg_float, safe_pos_float
    from kraken_client import KrakenClient, KrakenError


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
    if isinstance(interval, bool) or isinstance(timeout, bool):
        raise ValueError("interval and timeout must not be boolean")
    if isinstance(interval, float) or (isinstance(interval, str) and "." in interval.strip()):
        raise ValueError(f"invalid interval={interval!r} or timeout={timeout!r}")
    itv_f = safe_pos_float(interval, default=None)
    to_f = safe_pos_float(timeout, default=None)
    if itv_f is None or to_f is None or not itv_f.is_integer():
        raise ValueError(f"invalid interval={interval!r} or timeout={timeout!r}")
    interval = int(itv_f)
    qs = urllib.parse.urlencode({"pair": pair, "interval": interval})
    url = f"https://api.kraken.com/0/public/OHLC?{qs}"
    req = urllib.request.Request(
        url, headers={"User-Agent": "mptrade/kraken-ohlc-fetcher"}
    )
    with urllib.request.urlopen(req, timeout=to_f) as r:
        data = json.loads(r.read())
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
        if not isinstance(x, (list, tuple)) or len(x) < 5:
            continue
        raw_ts = safe_pos_float(x[0], default=None)
        o_val = safe_pos_float(x[1], default=None)
        h_val = safe_pos_float(x[2], default=None)
        l_val = safe_pos_float(x[3], default=None)
        c_val = safe_pos_float(x[4], default=None)
        if raw_ts is None or o_val is None or h_val is None or l_val is None or c_val is None:
            continue
        ts = int(raw_ts)
        if ts <= 0 or h_val < l_val or h_val < max(o_val, c_val) or l_val > min(o_val, c_val):
            continue
        records.append({
            "timestamp": ts,
            "open": o_val,
            "high": h_val,
            "low": l_val,
            "close": c_val,
        })
    return records


def _is_kraken_error(exc: BaseException) -> bool:
    return isinstance(exc, KrakenError) or exc.__class__.__name__ == "KrakenError"


def get_price(client: KrakenClient, pair: str) -> float | None:
    """Return the latest pair price, e.g. HYPEUSD, or None when unavailable."""
    try:
        return client.last_price(pair)
    except Exception as e:
        if not _is_kraken_error(e):
            raise
        log(f"  ! the price of {pair} is unavailable: {e}")
        return None


def check_pair_listing(client: KrakenClient, pair: str) -> tuple[dict | None, bool]:
    """Return (pair_metadata, confirmed_unlisted).

    confirmed_unlisted is True only when Kraken explicitly reports an unknown pair
    or a non-tradable status, and False on transient network/empty responses.
    """
    try:
        info = client.pair_info(pair)
    except Exception as e:
        if not _is_kraken_error(e):
            raise
        is_unknown = "unknown asset pair" in str(e).lower()
        if not is_unknown:
            log(f"  ! transient Kraken error checking {pair}: {e}")
        return None, is_unknown
    if not info:
        return None, False
    # Status 'online' means tradable; Kraken also reports cancel/post/limit/reduce-only.
    raw_status = info.get("status", "online")
    status = raw_status.strip().lower() if isinstance(raw_status, str) and raw_status.strip() else "online"
    if status not in ("online", "limit_only", "post_only"):
        log(f"  [market] {pair} listed but status={status} (still not tradable)")
        return None, True
    return info, False


def pair_available(client: KrakenClient, pair: str) -> dict | None:
    """Return pair metadata when LISTED and tradable on Kraken, otherwise None.

    Used as a launch detector; the bot waits while the pair is absent.
    """
    info, _ = check_pair_listing(client, pair)
    return info


def pair_precision(info: dict) -> tuple[int, int, float]:
    """Return (price decimals, volume decimals, minimum order) from pair metadata."""
    if not isinstance(info, dict):
        return 2, 8, 0.0
    p_val = safe_nonneg_float(info.get("pair_decimals"), default=None)
    price_dec = int(p_val) if p_val is not None and p_val <= 18 and p_val.is_integer() else 2
    v_val = safe_nonneg_float(info.get("lot_decimals"), default=None)
    vol_dec = int(v_val) if v_val is not None and v_val <= 18 and v_val.is_integer() else 8
    ordermin = safe_nonneg_float(info.get("ordermin", 0), default=0.0)
    return price_dec, vol_dec, ordermin
