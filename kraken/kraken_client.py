#!/usr/bin/env python3
"""
kraken_client.py - minimal REST client for Kraken Spot.

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
import os
import re

_KRAKEN_CL_ORD_ID_RE = re.compile(r"[0-9a-fA-F]{32}")
import sys
import threading
import time
import urllib.parse

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from botcore import (
    _parse_bool_flag,
    bind_attr_call,
    clamp_price_to_bbo_mid,
    clean_order_id,
    clean_scalar_str,
    exact_nonneg_int,
    exact_pos_int,
    http_get,
    http_post_form,
    log,
    normalize_side,
    safe_nonneg_float,
    safe_pos_float,
)
from instrument_registry import resolve_kraken_tokenized_pair
from providers.ohlc_utils import extract_kraken_ohlc_rows, require_ohlc_symbol
from providers.order_ohlc_utils import get_cached_pair_info as _get_cached_pair_info
from state_io import parse_json_payload
from credentials import kraken_credentials

API_URL = "https://api.kraken.com"

# --- Per-process shared TTL cache for READ calls ------------------------------
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
    "Balance": 15.0, "BalanceEx": 15.0, "TradesHistory": 20.0, "ClosedOrders": 20.0, "OpenOrders": 5.0,
}
_WRITE_METHODS = ("AddOrder", "AmendOrder", "CancelOrder", "CancelAll")
_INVALIDATE_ON_WRITE = ("Balance", "BalanceEx", "TradesHistory", "ClosedOrders", "OpenOrders")


def _read_ttl(method: str, params: dict) -> float | None:
    """Return an interval-aware TTL for cacheable reads."""
    ttl = _READ_TTL.get(method)
    ttl_f = safe_pos_float(ttl, default=None)
    if ttl_f is None or method != "OHLC":
        return ttl
    interval = max(1.0, safe_pos_float(params.get("interval", 1), default=1.0))
    return min(ttl_f, max(3.0, interval * 30.0))


def _cache_get(method: str, params: dict):
    with _CACHE_LOCK:
        now = time.time()
        key = (method, tuple(sorted((str(k), str(v)) for k, v in params.items() if k != "nonce")))
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
        _CACHE[(method, tuple(sorted((str(k), str(v)) for k, v in params.items() if k != "nonce")))] = (now + ttl, result)


def get_cached_pair_info(candidates, cache_dict: dict | None = None) -> dict | None:
    """Return cached AssetPairs metadata for the first matching candidate without network I/O."""
    return _get_cached_pair_info(candidates, cache_dict if isinstance(cache_dict, dict) else _CACHE)


def _resolve_kraken_pair(pair: str, *, upper_default: bool = True) -> tuple[str, bool]:
    """Return ``(effective_pair, is_tokenized)`` resolving uppercase registry keys (e.g. ``SPCXXUSD`` -> ``SPCXxUSD``)."""
    return resolve_kraken_tokenized_pair(pair, upper_default=upper_default)


class KrakenError(Exception):
    pass


class KrakenClient:
    def __init__(self, api_key: str | None = None, api_secret: str | None = None):
        self.api_key = clean_scalar_str(api_key) or ""
        self.api_secret = clean_scalar_str(api_secret) or ""

    @classmethod
    def public(cls) -> "KrakenClient":
        """Build a client that cannot call authenticated account endpoints."""
        return cls(api_key=None, api_secret=None)

    @classmethod
    def from_credentials(cls, role: str = "bot", *, creds_fn=None) -> "KrakenClient":
        """Build an authenticated client for ``role`` using ``kraken_credentials(role)``."""
        fn = creds_fn if callable(creds_fn) else kraken_credentials
        credentials = fn(role)
        return cls(credentials.key, credentials.secret)

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
        fresh_flag = _parse_bool_flag(fresh, default=False)
        if ttl and not fresh_flag:                  # serve a cacheable read from a fresh cache entry
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
            if status and body:
                try:
                    result = self._parse(status, body)
                    break
                except KrakenError as e:
                    err_s = str(e)
                    if not (
                        any(
                            tok in err_s
                            for tok in (
                                "Invalid nonce",
                                "Rate limit",
                                "Temporary lockout",
                                "Service:Unavailable",
                                "Service:Busy",
                                "EService:",
                            )
                        )
                        and (method not in _WRITE_METHODS or "Invalid nonce" in err_s)
                        and attempt < max_attempts - 1
                    ):
                        raise
            elif method in _WRITE_METHODS or attempt >= max_attempts - 1:
                raise KrakenError(f"network error/empty response in {method} (HTTP {status})")
            time.sleep(0.15 * (attempt + 1))
        if ttl and result:
            _cache_put(method, data, ttl, result)
        if method in _WRITE_METHODS:                # AddOrder/CancelOrder changed account state
            with _CACHE_LOCK:
                for k in [k for k in _CACHE if k[0] in _INVALIDATE_ON_WRITE]:
                    _CACHE.pop(k, None)
        return result

    def _public(self, method: str, params: dict | None = None, fresh: bool = False) -> dict:
        params = dict(params or {})
        ttl = _read_ttl(method, params)
        fresh_flag = _parse_bool_flag(fresh, default=False)
        if ttl and not fresh_flag:
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
            payload = parse_json_payload(body or b"{}")
        except ValueError:
            raise KrakenError(f"invalid response (HTTP {status})")
        if not isinstance(payload, dict):
            raise KrakenError(f"invalid response (HTTP {status})")
        if payload.get("error"):
            err_val = payload["error"]
            raise KrakenError(", ".join(str(x) for x in err_val) if isinstance(err_val, (list, tuple)) else str(err_val))
        res = payload.get("result", {})
        return res if isinstance(res, dict) else {}

    # ----- PUBLIC --------------------------------------------------------------
    def asset_pairs(self) -> dict:
        return self._public("AssetPairs")

    def tokenized_asset_pairs(self) -> dict:
        return self._public("AssetPairs", {"aclass_base": "tokenized_asset"})

    def _public_first_dict(self, method: str, pair: str) -> dict | None:
        if not clean_scalar_str(pair):
            return None
        eff_pair, is_tok = _resolve_kraken_pair(pair, upper_default=True)
        params: dict[str, str] = {"pair": eff_pair}
        if is_tok:
            params["aclass_base" if method == "AssetPairs" else "asset_class"] = "tokenized_asset"
        res = self._public(method, params)
        return next((v for v in res.values() if isinstance(v, dict)), None) if res else None

    pair_info = bind_attr_call("_public_first_dict", "AssetPairs")

    def cached_pair_info(self, candidates) -> dict | None:
        """Return in-memory cached pair metadata for ``candidates`` without network I/O."""
        own_cache = getattr(self, "_CACHE", None)
        return get_cached_pair_info(candidates, own_cache if isinstance(own_cache, dict) else None)

    ticker = bind_attr_call("_public_first_dict", "Ticker")

    def trades_page(self, pair: str, since: str = "0") -> tuple[list, str]:
        """Fetch one page of public Trades rows and the next nanosecond cursor."""
        if not clean_scalar_str(pair):
            return [], str(since or "0")
        eff_pair, is_tok = _resolve_kraken_pair(pair, upper_default=True)
        eff_since = str(since or "0").strip() or "0"
        params: dict[str, str] = {"pair": eff_pair, "since": eff_since}
        if is_tok:
            params["asset_class"] = "tokenized_asset"
        res = self._public("Trades", params, fresh=True)
        rows = res.get(params["pair"]) or next(
            (v for k, v in res.items() if k != "last" and isinstance(v, list)), []
        )
        return (rows if isinstance(rows, list) else []), str(res.get("last") or eff_since)

    def ohlc_closes_with_timestamps(
        self, pair: str, interval: int,
    ) -> tuple[list, tuple]:
        """Return completed OHLC closes and every corresponding close timestamp."""
        pair = require_ohlc_symbol(pair, label="pair")
        if (
            isinstance(interval, (bool, float))
            or interval is None
            or (isinstance(interval, str) and "." in interval.strip())
        ):
            raise ValueError("interval must be a positive integer")
        itv_int = exact_pos_int(interval, default=None)
        if itv_int is None:
            raise ValueError("interval must be a positive integer")
        itv_sec = itv_int * 60
        eff_pair, is_tok = _resolve_kraken_pair(pair, upper_default=False)
        ohlc_params: dict[str, object] = {"pair": eff_pair, "interval": itv_int}
        if is_tok:
            ohlc_params["asset_class"] = "tokenized_asset"
        rows = extract_kraken_ohlc_rows(
            self._public("OHLC", ohlc_params),
            drop_incomplete=True,
        )
        closes: list[float] = []
        timestamps: list[float] = []
        volumes: list[float] = []
        highs: list[float | None] = []
        lows: list[float | None] = []
        opens: list[float | None] = []
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) < 5:
                continue
            ts_val = safe_pos_float(row[0], default=None)
            c_val = safe_pos_float(row[4], default=None)
            if ts_val is None or c_val is None:
                continue
            closes.append(c_val)
            timestamps.append(ts_val + itv_sec)
            volumes.append(safe_nonneg_float(row[6], default=0.0) or 0.0 if len(row) > 6 else 0.0)
            opens.append(safe_pos_float(row[1], default=None))
            highs.append(safe_pos_float(row[2], default=None))
            lows.append(safe_pos_float(row[3], default=None))
        if not isinstance(getattr(self, "_last_ohlc_volumes", None), dict):
            self._last_ohlc_volumes = {}
        self._last_ohlc_volumes[(pair, itv_int)] = tuple(volumes)
        if not isinstance(getattr(self, "_last_ohlc_hlo", None), dict):
            self._last_ohlc_hlo = {}
        self._last_ohlc_hlo[(pair, itv_int)] = (
            tuple(highs) if any(v is not None for v in highs) else (),
            tuple(lows) if any(v is not None for v in lows) else (),
            tuple(opens) if any(v is not None for v in opens) else (),
        )
        return closes, tuple(timestamps)

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
        if not isinstance(t, dict) or not t:
            return None
        def _first(key: str):
            val = t.get(key)
            return val[0] if isinstance(val, (list, tuple)) and val else None
        return clamp_price_to_bbo_mid(_first("c"), _first("b"), _first("a"))

    # ----- PRIVATE -------------------------------------------------------------
    def balance(self) -> dict:
        """Return asset balances as {asset: quantity}."""
        return self._private("Balance")

    balance._is_default_kraken_balance = True  # type: ignore[attr-defined]
    balance_ex = bind_attr_call("_private", "BalanceEx")

    def price_decimals(self, pair: str) -> int:
        """Return Kraken's allowed PRICE decimals for the pair from AssetPairs,
        cached for one hour. Conservatively default to 2 when metadata is missing;
        Kraken accepts fewer decimals but never more (e.g. HYPE/USD = 2)."""
        try:
            info = self.pair_info(pair)
            dec_i = exact_nonneg_int(info.get("pair_decimals") if isinstance(info, dict) else None, default=None)
            if dec_i is not None and dec_i <= 18:
                return dec_i
        except Exception:  # noqa: BLE001 - conservatively fall back without blocking the order
            pass
        return 2

    def add_order(self, pair: str, side: str, volume: float, price: float | None = None,
                  ordertype: str = "limit", validate: bool = False,
                  cl_ord_id: str | None = None, *, post_only: bool = True) -> dict:
        """Place an order with side='buy'|'sell'; validate=True validates without placing.
        Round price to the pair's actual pair_decimals precision. This centralizes
        mechanical protection for every Kraken caller (trailing/bot/xstock), analogous
        to Binance place_order_mechanics. Without it, Kraken rejects excess precision
        with 'price can only be specified up to N decimals' and the order fails."""
        if not pair or isinstance(pair, bool) or not str(pair).strip():
            raise ValueError(f"invalid Kraken pair: {pair!r}")
        pair_s, is_tok = _resolve_kraken_pair(str(pair), upper_default=False)
        vol_f = safe_pos_float(volume, default=None)
        if vol_f is None:
            raise ValueError(f"invalid Kraken order volume: {volume!r}")
        side_u = normalize_side(side)
        if not side_u:
            raise ValueError(f"invalid Kraken order side: {side!r}")
        ot_norm = str(ordertype or "").strip().lower() if not isinstance(ordertype, bool) else ""
        if ot_norm not in {"limit", "market"}:
            raise ValueError(f"invalid Kraken ordertype: {ordertype!r}")
        vol_s = f"{vol_f}"
        data = {
            "pair": pair_s,
            "type": side_u.lower(),
            "ordertype": ot_norm,
            "volume": (f"{vol_f:.12f}".rstrip("0").rstrip(".") or "0") if "e" in vol_s.lower() else vol_s,
        }
        if is_tok:
            data["asset_class"] = "tokenized_asset"
        if ot_norm == "limit":
            px_f = safe_pos_float(price, default=None)
            if px_f is None:
                raise ValueError(f"invalid Kraken limit price: {price!r}")
            price = round(px_f, self.price_decimals(pair_s))
            if price <= 0.0:
                raise ValueError(f"invalid Kraken rounded limit price: {price!r}")
            px_s = f"{price}"
            data["price"] = (f"{price:.18f}".rstrip("0").rstrip(".") or "0") if "e" in px_s.lower() else px_s
            if _parse_bool_flag(post_only, default=True):
                data["oflags"] = "post"
        if cl_ord_id is not None:
            if isinstance(cl_ord_id, bool) or not _KRAKEN_CL_ORD_ID_RE.fullmatch(str(cl_ord_id).strip()):
                raise ValueError("a Kraken cl_ord_id must be a 128-bit hex UUID")
            data["cl_ord_id"] = str(cl_ord_id).strip().lower()
        if _parse_bool_flag(validate, default=False):
            data["validate"] = "true"
        return self._private("AddOrder", data)

    @staticmethod
    def _clean_txid(txid: object) -> str:
        if isinstance(txid, float):
            raise ValueError(f"invalid Kraken txid: {txid!r}")
        cleaned = clean_order_id(txid)
        if cleaned is None:
            raise ValueError(f"invalid Kraken txid: {txid!r}")
        return cleaned

    def amend_order(self, txid: str, limit_price: float, pair: str, *, post_only: bool = True) -> dict:
        """Atomically amend a resting limit order's price in place via Kraken's AmendOrder endpoint."""
        if not pair or isinstance(pair, bool) or not str(pair).strip():
            raise ValueError(f"invalid Kraken pair: {pair!r}")
        pair_s = str(pair).strip()
        px_f = safe_pos_float(limit_price, default=None)
        if px_f is None:
            raise ValueError(f"invalid Kraken limit price: {limit_price!r}")
        price = round(px_f, self.price_decimals(pair_s))
        if price <= 0.0:
            raise ValueError(f"invalid Kraken rounded limit price: {price!r}")
        px_s = f"{price}"
        data = {
            "txid": self._clean_txid(txid),
            "limit_price": (f"{price:.18f}".rstrip("0").rstrip(".") or "0") if "e" in px_s.lower() else px_s,
        }
        if _parse_bool_flag(post_only, default=True):
            data["post_only"] = "true"
        return self._private("AmendOrder", data)

    def _private_txid_call(self, method: str, txid: str) -> dict:
        return self._private(method, {"txid": self._clean_txid(txid)})

    cancel_order = bind_attr_call("_private_txid_call", "CancelOrder")
    query_orders = bind_attr_call("_private_txid_call", "QueryOrders")

    def _private_dict_field(self, method: str, field: str) -> dict:
        resp = self._private(method)
        raw = resp.get(field) if isinstance(resp, dict) else None
        return raw if isinstance(raw, dict) else {}

    open_orders = bind_attr_call("_private_dict_field", "OpenOrders", "open")
    closed_orders = bind_attr_call("_private_dict_field", "ClosedOrders", "closed")


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
    log(f"signature self-test: {'OK' if ok else 'FAILED'}")
    if not ok:
        log(f"  expected: {expected}")
        log(f"  got     : {got}")
