#!/usr/bin/env python3
"""
kraken_client.py — minimal REST client for Kraken Spot.

Public (no credentials): ticker, asset_pairs, pair_info
Private (credentials required): balance, add_order, cancel_order, query_orders, open_orders

Kraken authentication (different from T212):
    API-Key  : public key in the header
    API-Sign : HMAC-SHA512 over urlpath + SHA256(nonce + postdata),
               keyed by the base64 secret and returned as base64.
See the self-test at the end of the file, validated against Kraken's documentation vector.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import threading
import time
import urllib.parse

from kraken_common import http_get, http_post_form, log

API_URL = "https://api.kraken.com"

# ─── Per-process shared TTL cache for READ calls ──────────────────────────────
# Kraken rate-limits API calls per key. Multiple consumers (monitortrades through the
# provider, kraken_bot, trailing, and xstock_watch), plus the profit guard's two
# TradesHistory queries per placement (window + last_opposite_fill), repeatedly hit
# the same endpoints. Cache reads briefly and SHARE them across every KrakenClient
# instance in the process. WRITE methods (AddOrder/CancelOrder) INVALIDATE account
# state so guards and bots see their own transaction immediately, with no staleness
# window for local actions. This is not cross-process; each process has its own cache,
# and the TTL bounds the resulting lag.
_CACHE = {}                       # (method, params_key) -> (expiry_ts, result)
_CACHE_LOCK = threading.Lock()
_CACHE_MAX = 1024
_READ_TTL = {                     # seconds; unlisted methods are not cached (e.g. QueryOrders)
    # OHLC is capped at 15 minutes; _read_ttl shortens it for short intervals.
    "Ticker": 3.0, "AssetPairs": 3600.0, "OHLC": 900.0,
    "Balance": 15.0, "TradesHistory": 20.0, "ClosedOrders": 20.0, "OpenOrders": 5.0,
}
_WRITE_METHODS = ("AddOrder", "CancelOrder", "CancelAll")
_INVALIDATE_ON_WRITE = ("Balance", "TradesHistory", "ClosedOrders", "OpenOrders")


def _params_key(params: dict) -> tuple:
    return tuple(sorted((str(k), str(v)) for k, v in params.items() if k != "nonce"))


def _read_ttl(method: str, params: dict) -> float | None:
    """Return an interval-aware TTL for cacheable reads."""
    ttl = _READ_TTL.get(method)
    if not ttl or method != "OHLC":
        return ttl
    try:
        interval = max(1.0, float(params.get("interval", 1)))
    except (TypeError, ValueError, OverflowError):
        interval = 1.0
    return min(float(ttl), max(3.0, interval * 30.0))


def _cache_get(method: str, params: dict):
    with _CACHE_LOCK:
        now = time.time()
        key = (method, _params_key(params))
        hit = _CACHE.get(key)
        if hit and hit[0] > now:
            return True, hit[1]
        _CACHE.pop(key, None)
    return False, None


def _cache_put(method: str, params: dict, ttl: float, result) -> None:
    with _CACHE_LOCK:
        now = time.time()
        for key in [key for key, value in _CACHE.items() if value[0] <= now]:
            _CACHE.pop(key, None)
        if len(_CACHE) >= _CACHE_MAX:
            oldest = min(_CACHE, key=lambda key: _CACHE[key][0])
            _CACHE.pop(oldest, None)
        _CACHE[(method, _params_key(params))] = (now + ttl, result)


def _cache_invalidate(methods) -> None:
    with _CACHE_LOCK:
        for k in [k for k in _CACHE if k[0] in methods]:
            _CACHE.pop(k, None)


class KrakenError(Exception):
    pass


class KrakenClient:
    def __init__(self, api_key: str | None = None, api_secret: str | None = None):
        self.api_key = api_key or ""
        self.api_secret = api_secret or ""

    @classmethod
    def public(cls) -> "KrakenClient":
        """Build a client that cannot call authenticated account endpoints."""
        return cls(api_key=None, api_secret=None)

    # ----- signature -----------------------------------------------------------
    @staticmethod
    def _signature(urlpath: str, data: dict, secret: str) -> str:
        postdata = urllib.parse.urlencode(data)
        encoded = (str(data["nonce"]) + postdata).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()
        mac = hmac.new(base64.b64decode(secret), message, hashlib.sha512)
        return base64.b64encode(mac.digest()).decode()

    def _private(self, method: str, data: dict | None = None, fresh: bool = False) -> dict:
        if not self.api_key or not self.api_secret:
            raise KrakenError("Missing Kraken keys (check KRAKEN_API_KEY_BOT/_TRAIL/_CACHE in kraken/.env)")
        data = dict(data or {})
        ttl = _read_ttl(method, data)
        if ttl and not fresh:                       # serve a cacheable read from a fresh cache entry
            ok, val = _cache_get(method, data)
            if ok:
                return val
        urlpath = f"/0/private/{method}"
        # A nanosecond nonce is monotonic at maximum resolution and exceeds prior ms/us nonces on the key.
        max_attempts = 3 if method not in _WRITE_METHODS else 1
        result = None
        for attempt in range(max_attempts):
            data["nonce"] = str(time.time_ns())
            headers = {
                "API-Key": self.api_key,
                "API-Sign": self._signature(urlpath, data, self.api_secret),
            }
            status, body = http_post_form(API_URL + urlpath, data, headers=headers)
            try:
                result = self._parse(status, body)
                break
            except KrakenError as e:
                if "Invalid nonce" in str(e) and attempt < max_attempts - 1:
                    time.sleep(0.15 * (attempt + 1))
                    continue
                raise
        if ttl and result is not None:
            _cache_put(method, data, ttl, result)
        if method in _WRITE_METHODS:                # AddOrder/CancelOrder changed account state
            _cache_invalidate(_INVALIDATE_ON_WRITE)
        return result

    def _public(self, method: str, params: dict | None = None, fresh: bool = False) -> dict:
        params = dict(params or {})
        ttl = _read_ttl(method, params)
        if ttl and not fresh:
            ok, val = _cache_get(method, params)
            if ok:
                return val
        url = f"{API_URL}/0/public/{method}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        status, body = http_get(url)
        result = self._parse(status, body)
        # Never cache an EMPTY result. For public endpoints (AssetPairs/Ticker), {}
        # always means a transient failed fetch, never valid state. Caching it for the
        # one-hour AssetPairs TTL would make pair_info return None and ordermin become 0,
        # disabling monitortrades._place_guarded's 'volume minimum not met' protection
        # for about an hour and causing rejected dust-order churn (HYPE 0.0175 < min 0.1).
        # _parse already raises on reported errors; this catches an empty success response.
        if ttl and result:
            _cache_put(method, params, ttl, result)
        return result

    @staticmethod
    def _parse(status: int, body: bytes) -> dict:
        try:
            payload = json.loads(body or b"{}")
        except ValueError:
            raise KrakenError(f"raspuns invalid (HTTP {status})")
        if payload.get("error"):
            raise KrakenError(", ".join(payload["error"]))
        return payload.get("result", {})

    # ----- PUBLIC --------------------------------------------------------------
    def asset_pairs(self) -> dict:
        return self._public("AssetPairs")

    def pair_info(self, pair: str) -> dict | None:
        """Return pair price/volume precision and minimum order, or None if absent."""
        res = self._public("AssetPairs", {"pair": pair})
        if not res:
            return None
        return next(iter(res.values()))

    def ticker(self, pair: str) -> dict | None:
        res = self._public("Ticker", {"pair": pair})
        return next(iter(res.values())) if res else None

    def ohlc_closes_with_timestamps(
        self, pair: str, interval: int,
    ) -> tuple[list, tuple]:
        """Return completed OHLC closes and every corresponding close timestamp."""
        res = self._public("OHLC", {"pair": pair, "interval": interval})
        key = next((key for key in res if key != "last"), None)
        rows = list(res[key])[:-1] if key else []
        closes = [float(row[4]) for row in rows]
        timestamps = tuple(
            float(row[0]) + int(interval) * 60 for row in rows)
        return closes, timestamps

    def ohlc_closes_with_timestamp(
        self, pair: str, interval: int,
    ) -> tuple[list, float | None]:
        """Return completed closes and the newest close timestamp."""
        closes, timestamps = self.ohlc_closes_with_timestamps(pair, interval)
        return closes, timestamps[-1] if timestamps else None

    def ohlc_closes(self, pair: str, interval: int) -> list:
        """Return completed OHLC closes while preserving the legacy list API."""
        closes, _last_closed_at = self.ohlc_closes_with_timestamp(pair, interval)
        return closes

    def last_price(self, pair: str) -> float | None:
        t = self.ticker(pair)
        try:
            return float(t["c"][0]) if t else None      # 'c' = latest trade [price, volume]
        except (KeyError, IndexError, TypeError, ValueError):
            return None

    # ----- PRIVATE -------------------------------------------------------------
    def balance(self) -> dict:
        """Return asset balances as {asset: quantity}."""
        return self._private("Balance")

    def price_decimals(self, pair: str) -> int:
        """Return Kraken's allowed PRICE decimals for the pair from AssetPairs,
        cached for one hour. Conservatively default to 2 when metadata is missing;
        Kraken accepts fewer decimals but never more (e.g. HYPE/USD = 2)."""
        try:
            info = self.pair_info(pair)
            if info and info.get("pair_decimals") is not None:
                return int(info["pair_decimals"])
        except Exception:  # noqa: BLE001 — conservatively fall back without blocking the order
            pass
        return 2

    def add_order(self, pair: str, side: str, volume: float, price: float | None = None,
                  ordertype: str = "limit", validate: bool = False,
                  cl_ord_id: str | None = None) -> dict:
        """Place an order with side='buy'|'sell'; validate=True validates without placing.
        Round price to the pair's actual pair_decimals precision. This centralizes
        mechanical protection for every Kraken caller (trailing/bot/xstock), analogous
        to Binance place_order_mechanics. Without it, Kraken rejects excess precision
        with 'price can only be specified up to N decimals' and the order fails."""
        data = {
            "pair": pair,
            "type": side,
            "ordertype": ordertype,
            "volume": f"{volume}",
        }
        if ordertype == "limit" and price is not None:
            price = round(float(price), self.price_decimals(pair))
            data["price"] = f"{price}"
        if cl_ord_id is not None:
            if not re.fullmatch(r"[0-9a-fA-F]{32}", str(cl_ord_id)):
                raise ValueError("a Kraken cl_ord_id must be a 128-bit hex UUID")
            data["cl_ord_id"] = str(cl_ord_id).lower()
        if validate:
            data["validate"] = "true"
        return self._private("AddOrder", data)

    def cancel_order(self, txid: str) -> dict:
        return self._private("CancelOrder", {"txid": txid})

    def query_orders(self, txids: str) -> dict:
        """Return order status by txid, including closed orders without T212-style 404s."""
        return self._private("QueryOrders", {"txid": txids})

    def open_orders(self) -> dict:
        return self._private("OpenOrders").get("open", {})

    def closed_orders(self) -> dict:
        """Return recent closed orders for deterministic client-ID recovery."""
        return self._private("ClosedOrders").get("closed", {})


# ---------------------------------------------------------------------------
# Signature self-test using the vector from Kraken documentation; run:
#   python3 kraken_client.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    secret = ("kQH5HW/8p1uGOVjbgWA7FunAmGO8lsSUXNsu3eow76sz84Q18fWxnyRz"
              "BHCd3pd5nE9qa99HAZtuZuj6F1huXg==")
    data = {"nonce": "1616492376594", "ordertype": "limit", "pair": "XBTUSD",
            "price": 37500, "type": "buy", "volume": 1.25}
    expected = ("4/dpxb3iT4tp/ZCVEwSnEsLxx0bqyhLpdfOpc6fn7OR8+UClSV5n9E6aSS"
                "8MPtnRfp32bAb0nmbRn6H8ndwLUQ==")
    got = KrakenClient._signature("/0/private/AddOrder", data, secret)
    ok = got == expected
    log(f"signature self-test: {'OK ✅' if ok else 'FAILED ❌'}")
    if not ok:
        log(f"  asteptat: {expected}")
        log(f"  obtinut : {got}")
