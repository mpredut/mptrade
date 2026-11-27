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
import contextlib
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
    FileLock,
    _parse_bool_flag,
    bind_attr_call,
    clamp_price_to_bbo_mid,
    clean_order_id,
    clean_scalar_str,
    exact_nonneg_int,
    exact_pos_int,
    http_get,
    http_post_form,
    http_request,
    is_shared_disk_cache_enabled,
    log,
    normalize_side,
    safe_nonneg_float,
    safe_pos_float,
)
from cachepaths import is_readonly_xstock_symbol, kraken_pair_matches
from instrument_registry import resolve_kraken_tokenized_pair
from providers.ohlc_utils import extract_kraken_ohlc_rows, require_ohlc_symbol
from providers.order_ohlc_utils import get_cached_pair_info as _get_cached_pair_info
from state_io import encode_json_payload, parse_json_payload
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
_WRITE_METHODS = ("AddOrder", "AmendOrder", "CancelOrder", "CancelAll", "Withdraw", "WithdrawCancel")
_INVALIDATE_ON_WRITE = ("Balance", "BalanceEx", "TradesHistory", "ClosedOrders", "OpenOrders")
_NONCE_STATE_LOCK = threading.Lock()
_NONCE_THREAD_LOCKS: dict[str, threading.Lock] = {}
_LAST_NONCE_BY_KEY: dict[str, int] = {}


def _read_ttl(method: str, params: dict) -> float | None:
    """Return an interval-aware TTL for cacheable reads."""
    ttl = _READ_TTL.get(method)
    ttl_f = safe_pos_float(ttl, default=None)
    if ttl_f is None or method != "OHLC":
        return ttl
    interval = max(1.0, safe_pos_float(params.get("interval", 1), default=1.0))
    return min(ttl_f, max(3.0, interval * 30.0))


def _cache_key(method: str, params: dict, client_id: str = "") -> tuple:
    params_key = tuple(sorted((str(k), str(v)) for k, v in params.items() if k != "nonce"))
    # Keep public cache keys as canonical 2-tuples so get_cached_pair_info() continues to work verbatim;
    # scope private method cache entries by client identity so distinct credentials never collide.
    if client_id:
        return (method, client_id, params_key)
    return (method, params_key)


def _cache_get(method: str, params: dict, client_id: str = ""):
    ttl = _read_ttl(method, params)
    with _CACHE_LOCK:
        now = time.time()
        key = _cache_key(method, params, client_id=client_id)
        hit = _CACHE.get(key)
        if hit and hit[0] > now and (ttl is None or (hit[0] - now) <= ttl + 1.0):
            return True, hit[1]
        _CACHE.pop(key, None)
    return False, None


def _cache_put(method: str, params: dict, ttl: float, result, client_id: str = "") -> None:
    with _CACHE_LOCK:
        now = time.time()
        for key in [key for key, value in _CACHE.items() if value[0] <= now]:
            _CACHE.pop(key, None)
        if len(_CACHE) >= _CACHE_MAX:
            oldest = min(_CACHE, key=lambda key: _CACHE[key][0])
            _CACHE.pop(oldest, None)
        _CACHE[_cache_key(method, params, client_id=client_id)] = (now + ttl, result)


def get_cached_pair_info(candidates, cache_dict: dict | None = None) -> dict | None:
    """Return cached AssetPairs metadata for the first matching candidate without network I/O."""
    return _get_cached_pair_info(candidates, cache_dict if isinstance(cache_dict, dict) else _CACHE)


def _resolve_kraken_pair(pair: str, *, upper_default: bool = True) -> tuple[str, bool]:
    """Return ``(effective_pair, is_tokenized)`` resolving case-folded registry keys to canonical mixed-case pairs (e.g. ``SPCXxUSD``)."""
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

    def _private_cache_scope(self) -> str:
        raw = f"{self.api_key}\x00{self.api_secret}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    def _nonce_key_hash(self) -> str:
        return hashlib.sha256(self.api_key.encode("utf-8", errors="ignore")).hexdigest()[:10]

    def _nonce_lock_path(self) -> str:
        override = getattr(self, "_nonce_lock_path_override", None)
        if isinstance(override, str) and override.strip():
            return override.strip()
        return os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f".kraken_nonce_{self._nonce_key_hash()}.lock",
        )

    def _next_nonce_str(self) -> str:
        key_hash = self._nonce_key_hash()
        with _NONCE_STATE_LOCK:
            now_ns = time.time_ns()
            prev = _LAST_NONCE_BY_KEY.get(key_hash, 0)
            nonce = now_ns if now_ns > prev else prev + 1
            _LAST_NONCE_BY_KEY[key_hash] = nonce
            return str(nonce)

    @contextlib.contextmanager
    def _private_nonce_lock(self):
        """Serialize private nonce signing + HTTP dispatch per API key across threads and processes."""
        key_hash = self._nonce_key_hash()
        with _NONCE_STATE_LOCK:
            tlock = _NONCE_THREAD_LOCKS.get(key_hash)
            if tlock is None:
                tlock = threading.Lock()
                _NONCE_THREAD_LOCKS[key_hash] = tlock
        use_file_lock = getattr(self, "_enable_nonce_file_lock", None)
        if use_file_lock is None:
            use_file_lock = is_shared_disk_cache_enabled()
        with tlock:
            if not use_file_lock:
                yield
                return
            lock_cm = FileLock(
                self._nonce_lock_path(),
                timeout=15.0,
                poll_interval=0.02,
                raise_on_timeout=False,
            )
            entered = False
            try:
                lock_cm.__enter__()
                entered = True
            except Exception:
                entered = False
            try:
                yield
            finally:
                if entered:
                    with contextlib.suppress(Exception):
                        lock_cm.__exit__(None, None, None)

    def _private(
        self,
        method: str,
        data: dict | None = None,
        fresh: bool = False,
        *,
        allow_list: bool = False,
    ) -> dict | list:
        if not self.api_key or not self.api_secret:
            raise KrakenError("Missing Kraken keys (check KRAKEN_API_KEY_BOT/_TRAIL/_CACHE/_TRANSFER in kraken/.env)")
        data = dict(data or {})
        ttl = _read_ttl(method, data)
        fresh_flag = _parse_bool_flag(fresh, default=False)
        client_scope = self._private_cache_scope()
        if ttl and not fresh_flag:                  # serve a cacheable read from a fresh cache entry
            ok, val = _cache_get(method, data, client_id=client_scope)
            if ok:
                return val
        urlpath = f"/0/private/{method}"
        # A nanosecond nonce is monotonic at maximum resolution and exceeds prior ms/us nonces on the key.
        is_real_write = (
            method in _WRITE_METHODS
            and str(data.get("validate", "")).strip().lower() != "true"
        )
        max_attempts = 3 if method not in _WRITE_METHODS else 1
        result = None
        write_dispatched = False
        rate_limited = False
        try:
            for attempt in range(max_attempts):
                with self._private_nonce_lock():
                    data["nonce"] = self._next_nonce_str()
                    headers = {
                        "API-Key": self.api_key,
                        "API-Sign": self._signature(urlpath, data, self.api_secret),
                    }
                    if is_real_write:
                        write_dispatched = True
                    status, body = http_post_form(API_URL + urlpath, data, headers=headers)
                if status == 429:
                    rate_limited = True
                if status and body:
                    try:
                        result = self._parse(status, body, allow_list=allow_list)
                        break
                    except KrakenError as e:
                        err_s = str(e)
                        if "rate limit" in err_s.lower():
                            rate_limited = True
                        if not (
                            any(
                                tok in err_s
                                for tok in (
                                    "Invalid nonce",
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
        finally:
            if is_real_write and write_dispatched and not rate_limited:
                # Invalidate account read caches on any dispatched real write (including
                # ambiguous timeouts/5xx) so immediate reconciliation never reads stale state.
                with _CACHE_LOCK:
                    for k in [k for k in _CACHE if k[0] in _INVALIDATE_ON_WRITE]:
                        _CACHE.pop(k, None)
        if ttl and result:
            _cache_put(method, data, ttl, result, client_id=client_scope)
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
    def _parse(status: int, body: bytes, *, allow_list: bool = False) -> dict | list:
        if not body:
            raise KrakenError(f"empty response (HTTP {status})")
        try:
            payload = parse_json_payload(body)
        except ValueError:
            raise KrakenError(f"invalid response (HTTP {status})")
        if not isinstance(payload, dict):
            raise KrakenError(f"invalid response (HTTP {status})")
        if payload.get("error"):
            err_val = payload["error"]
            raise KrakenError(", ".join(str(x) for x in err_val) if isinstance(err_val, (list, tuple)) else str(err_val))
        code = exact_nonneg_int(status, default=0) or 0
        if not (200 <= code < 300):
            raise KrakenError(f"HTTP {status} error response")
        res = payload.get("result", [] if allow_list else {})
        if allow_list:
            return res if isinstance(res, list) else []
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
        if not res:
            return None
        for k, v in res.items():
            if isinstance(v, dict) and kraken_pair_matches(k, eff_pair):
                return v
        if len(res) == 1:
            only_val = next(iter(res.values()))
            return only_val if isinstance(only_val, dict) else None
        return None

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

    # ----- FUNDING / WITHDRAWALS ----------------------------------------------
    @staticmethod
    def _fmt_amount(amount: float, label: str = "amount") -> str:
        amt_f = safe_pos_float(amount, default=None)
        if amt_f is None:
            raise ValueError(f"invalid Kraken {label}: {amount!r}")
        return f"{amt_f:.8f}".rstrip("0").rstrip(".") or "0"

    @staticmethod
    def _norm_wire_asset(asset_s: str) -> str:
        return asset_s if is_readonly_xstock_symbol(asset_s) else asset_s.upper()

    def withdraw_methods(self, asset: str) -> list[dict]:
        """List supported Kraken withdrawal networks/fees for ``asset``."""
        asset_s = clean_scalar_str(asset)
        if not asset_s:
            raise ValueError(f"invalid Kraken asset: {asset!r}")
        res = self._private("WithdrawMethods", {"asset": self._norm_wire_asset(asset_s)}, fresh=True, allow_list=True)
        return [row for row in res if isinstance(row, dict)] if isinstance(res, list) else []

    def withdraw_addresses(self, asset: str | None = None, *, method: str | None = None) -> list[dict]:
        """List pre-whitelisted Kraken withdrawal addresses."""
        params: dict[str, str] = {}
        if asset is not None:
            asset_s = clean_scalar_str(asset)
            if not asset_s:
                raise ValueError(f"invalid Kraken asset: {asset!r}")
            params["asset"] = self._norm_wire_asset(asset_s)
        if method is not None:
            method_s = clean_scalar_str(method)
            if method_s:
                params["method"] = method_s
        res = self._private("WithdrawAddresses", params, fresh=True, allow_list=True)
        return [row for row in res if isinstance(row, dict)] if isinstance(res, list) else []

    def withdraw_info(self, asset: str, key: str, amount: float) -> dict:
        """Query live Kraken withdrawal fee, limit, and net receive amount without withdrawing."""
        asset_s = clean_scalar_str(asset)
        key_s = clean_scalar_str(key)
        if not asset_s or not key_s:
            raise ValueError("Kraken withdraw_info requires non-empty asset and withdrawal key")
        data = {
            "asset": self._norm_wire_asset(asset_s),
            "key": key_s,
            "amount": self._fmt_amount(amount),
        }
        res = self._private("WithdrawInfo", data, fresh=True)
        return res if isinstance(res, dict) else {}

    def withdraw(
        self,
        asset: str,
        key: str,
        amount: float,
        *,
        address: str | None = None,
        max_fee: float | None = None,
    ) -> dict:
        """Submit a real Kraken withdrawal to a pre-whitelisted ``key`` (non-retried write)."""
        asset_s = clean_scalar_str(asset)
        key_s = clean_scalar_str(key)
        if not asset_s or not key_s:
            raise ValueError("Kraken withdraw requires non-empty asset and withdrawal key")
        data: dict[str, str] = {
            "asset": self._norm_wire_asset(asset_s),
            "key": key_s,
            "amount": self._fmt_amount(amount),
        }
        if address is not None:
            addr_s = clean_scalar_str(address)
            if not addr_s:
                raise ValueError(f"invalid Kraken withdrawal address: {address!r}")
            data["address"] = addr_s
        if max_fee is not None:
            data["max_fee"] = self._fmt_amount(max_fee, label="max_fee")
        res = self._private("Withdraw", data, fresh=True)
        return res if isinstance(res, dict) else {}

    def withdraw_status(self, asset: str | None = None, *, method: str | None = None) -> list[dict]:
        """Fetch recent Kraken withdrawal statuses and on-chain txids."""
        params: dict[str, str] = {}
        if asset is not None:
            asset_s = clean_scalar_str(asset)
            if asset_s:
                params["asset"] = self._norm_wire_asset(asset_s)
        if method is not None:
            method_s = clean_scalar_str(method)
            if method_s:
                params["method"] = method_s
        res = self._private("WithdrawStatus", params, fresh=True, allow_list=True)
        return [row for row in res if isinstance(row, dict)] if isinstance(res, list) else []

    # ----- FUNDING v1 API (/funding/v1/...) -----------------------------------
    @staticmethod
    def _funding_signature(uri_path: str, nonce: str, body_bytes: bytes, secret: str) -> str:
        sha = hashlib.sha256(nonce.encode("utf-8") + body_bytes).digest()
        mac = hmac.new(base64.b64decode(secret), uri_path.encode("utf-8") + sha, hashlib.sha512)
        return base64.b64encode(mac.digest()).decode()

    def _funding_v1(
        self,
        http_method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        payload: dict | None = None,
    ) -> dict:
        if not self.api_key or not self.api_secret:
            raise KrakenError("Missing Kraken keys (check KRAKEN_API_KEY_TRANSFER in kraken/.env)")
        verb = http_method.strip().upper()
        is_write = verb != "GET"
        qs = urllib.parse.urlencode(params) if params else ""
        uri_path = f"{path}?{qs}" if qs else path
        body_bytes = (
            encode_json_payload(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            if payload is not None
            else b""
        )
        max_attempts = 1 if is_write else 3
        last_err: Exception | None = None
        write_dispatched = False
        rate_limited = False
        try:
            for attempt in range(max_attempts):
                with self._private_nonce_lock():
                    nonce = self._next_nonce_str()
                    headers = {
                        "API-Key": self.api_key,
                        "API-Nonce": nonce,
                        "API-Sign": self._funding_signature(uri_path, nonce, body_bytes, self.api_secret),
                    }
                    if payload is not None:
                        headers["Content-Type"] = "application/json"
                    if is_write:
                        write_dispatched = True
                    status, resp_body = http_request(
                        verb,
                        API_URL + uri_path,
                        headers=headers,
                        body=body_bytes if payload is not None else None,
                    )
                if status == 429:
                    rate_limited = True
                if not status:
                    last_err = KrakenError(f"network error in {verb} {path} (HTTP 0)")
                    if is_write or attempt >= max_attempts - 1:
                        raise last_err
                    time.sleep(0.15 * (attempt + 1))
                    continue
                try:
                    parsed = parse_json_payload(resp_body or b"{}")
                except ValueError:
                    raise KrakenError(f"invalid Funding v1 response in {verb} {path} (HTTP {status})")
                if 200 <= status < 300:
                    return parsed if isinstance(parsed, dict) else {}
                title = parsed.get("title") if isinstance(parsed, dict) else None
                err_type = parsed.get("type") if isinstance(parsed, dict) else None
                detail = f"{title} ({err_type})" if title or err_type else resp_body.decode("utf-8", errors="replace")[:200]
                last_err = KrakenError(f"Funding v1 {verb} {path} HTTP {status}: {detail}")
                if is_write or status < 500 or attempt >= max_attempts - 1:
                    raise last_err
                time.sleep(0.15 * (attempt + 1))
            raise last_err or KrakenError(f"Funding v1 {verb} {path} failed")
        finally:
            if is_write and write_dispatched and not rate_limited:
                with _CACHE_LOCK:
                    for k in [k for k in _CACHE if k[0] in _INVALIDATE_ON_WRITE]:
                        _CACHE.pop(k, None)

    def funding_addresses(self) -> list[dict]:
        """List saved Funding v1 withdrawal addresses (including modern multi-network address book)."""
        res = self._funding_v1("GET", "/funding/v1/addresses")
        rows = res.get("addresses") if isinstance(res, dict) else None
        return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []

    def funding_fee_quote(self, method_id: str, amount: float, *, fee_included: bool = True) -> dict:
        """Quote live Kraken withdrawal fee, gross/net amounts, and ``withdrawal_fee_token``."""
        mid = clean_scalar_str(method_id)
        if not mid:
            raise ValueError(f"invalid Kraken method_id: {method_id!r}")
        params = {
            "amount": self._fmt_amount(amount),
            "fee_included": "true" if _parse_bool_flag(fee_included, default=True) else "false",
        }
        return self._funding_v1("GET", f"/funding/v1/fees/{mid}", params=params)

    def funding_withdraw(
        self,
        asset: str,
        method_id: str,
        address_id: str,
        amount: float,
        *,
        asset_class: str = "currency",
        fee_included: bool = True,
        withdrawal_fee_token: str | None = None,
        max_fee: float | None = None,
    ) -> dict:
        """Submit a Funding v1 withdrawal (non-retried POST write)."""
        asset_s = clean_scalar_str(asset)
        mid = clean_scalar_str(method_id)
        aid = clean_scalar_str(address_id)
        aclass = clean_scalar_str(asset_class) or "currency"
        if not asset_s or not mid or not aid:
            raise ValueError("Kraken funding_withdraw requires non-empty asset, method_id, and address_id")
        asset_obj = {"class": aclass, "name": self._norm_wire_asset(asset_s)}
        fee_obj: dict[str, object] = {
            "fee_included": bool(_parse_bool_flag(fee_included, default=True)),
        }
        tok = clean_scalar_str(withdrawal_fee_token)
        if tok:
            fee_obj["quoted_fee"] = {"token": tok}
        else:
            cur_fee: dict[str, object] = {}
            if max_fee is not None:
                cur_fee["max_fee"] = {
                    "asset_amount": {
                        "asset": asset_obj,
                        "amount": self._fmt_amount(max_fee, label="max_fee"),
                    }
                }
            fee_obj["current_fee"] = cur_fee
        body: dict[str, object] = {
            "scope": {"method_id": mid},
            "address_id": aid,
            "amount": {
                "asset_amount": {
                    "asset": asset_obj,
                    "amount": self._fmt_amount(amount),
                }
            },
            "fee": fee_obj,
        }
        return self._funding_v1("POST", "/funding/v1/withdrawals", payload=body)

    def funding_withdrawals(self) -> list[dict]:
        """List recent Funding v1 withdrawals with status and on-chain txid."""
        res = self._funding_v1("GET", "/funding/v1/withdrawals")
        rows = res.get("withdrawals") if isinstance(res, dict) else None
        return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


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
