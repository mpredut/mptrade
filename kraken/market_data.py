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
    from kraken.kraken_common import log
    from kraken.kraken_client import KrakenClient, KrakenError
except ImportError:
    from kraken_common import log
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
    if isinstance(interval, bool) or isinstance(timeout, bool):
        raise ValueError("interval and timeout must not be boolean")
    try:
        itv_f = float(interval)
        to_f = float(timeout)
        if not (math.isfinite(itv_f) and math.isfinite(to_f)) or int(itv_f) <= 0 or to_f <= 0:
            raise ValueError(f"invalid interval={interval!r} or timeout={timeout!r}")
        interval = int(itv_f)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid interval={interval!r} or timeout={timeout!r}") from exc
    qs = urllib.parse.urlencode({"pair": pair, "interval": interval})
    url = f"https://api.kraken.com/0/public/OHLC?{qs}"
    req = urllib.request.Request(
        url, headers={"User-Agent": "mptrade/kraken-ohlc-fetcher"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    if data.get("error"):
        raise RuntimeError(", ".join(data["error"]))
    res = data.get("result", {})
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
        if any(isinstance(x[i], bool) for i in range(5)):
            continue
        try:
            raw_ts = float(x[0])
            if not math.isfinite(raw_ts):
                continue
            ts = int(raw_ts)
            o_val = float(x[1])
            h_val = float(x[2])
            l_val = float(x[3])
            c_val = float(x[4])
        except (TypeError, ValueError, OverflowError):
            continue
        if not (math.isfinite(o_val) and math.isfinite(h_val) and math.isfinite(l_val) and math.isfinite(c_val)):
            continue
        if ts <= 0 or o_val <= 0 or h_val <= 0 or l_val <= 0 or c_val <= 0:
            continue
        if h_val < l_val or h_val < max(o_val, c_val) or l_val > min(o_val, c_val):
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
    status = info.get("status", "online")
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
    try:
        raw_p = info.get("pair_decimals")
        price_dec = max(0, int(float(raw_p))) if raw_p is not None and not isinstance(raw_p, bool) else 2
    except (TypeError, ValueError, OverflowError):
        price_dec = 2
    try:
        raw_v = info.get("lot_decimals")
        vol_dec = max(0, int(float(raw_v))) if raw_v is not None and not isinstance(raw_v, bool) else 8
    except (TypeError, ValueError, OverflowError):
        vol_dec = 8
    try:
        raw_min = info.get("ordermin", 0)
        ordermin = float(raw_min or 0) if not isinstance(raw_min, bool) else 0.0
        if not math.isfinite(ordermin) or ordermin < 0:
            ordermin = 0.0
    except (TypeError, ValueError, OverflowError):
        ordermin = 0.0
    return price_dec, vol_dec, ordermin
