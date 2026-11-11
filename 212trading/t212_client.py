#!/usr/bin/env python3
"""Minimal Trading 212 API client for instruments and order lifecycle.

The documented limit-order schema is:
    POST /equity/orders/limit
    {"ticker": "...", "quantity": <+BUY/-SELL>, "limitPrice": ..., "timeValidity": "DAY"|"GOOD_TILL_CANCEL"}
There is no ``side`` field, and the ticker key is not ``instrumentTicker``.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import os
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from botcore import (
    FileLock,
    _parse_bool_flag,
    clean_order_id,
    http_get,
    http_post_json,
    http_request,
    is_shared_disk_cache_enabled,
    log,
    normalize_side,
    required_env,
    required_float_env,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
)
from cachepaths import CACHE_T212_TRADE_FILENAME, STALE_STAGING_MAX_AGE_SEC, cache_path
from providers.t212_provider import T212Provider
from state_io import (
    atomic_write_json,
    cleanup_stale_staging_siblings,
    encode_json_payload,
    parse_json_payload,
    read_json_dict,
)

LIVE_BASE = "https://live.trading212.com/api/v0"
DEMO_BASE = "https://demo.trading212.com/api/v0"

_BROWSER_UA = "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0"


class T212Client:
    def __init__(self, api_key: str, api_secret: str | None = None, env: str | None = None,
                 *, min_gap_sec: float | None = None,
                 portfolio_ttl_sec: float | None = None,
                 order_min_gap_sec: float | None = None,
                 shared_cache_path: str | None = None,
                 enable_shared_disk_cache: bool | None = None,
                 trade_cache_path: str | None = None):
        if not api_key or isinstance(api_key, bool) or not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("Invalid T212 api_key")
        if api_secret is not None and isinstance(api_secret, bool):
            raise ValueError("Invalid T212 api_secret")
        self.api_key = api_key.strip()
        self.api_secret = api_secret.strip() if isinstance(api_secret, str) and api_secret.strip() else None
        if isinstance(env, bool):
            raise ValueError(f"Invalid T212 environment: {env!r}; expected live or demo")
        self.env = (str(env).strip() if env is not None else required_env("T212_ENV")).lower()
        if self.env not in {"live", "demo"}:
            raise ValueError(f"Invalid T212 environment: {env!r}; expected live or demo")
        self.base = DEMO_BASE if self.env == "demo" else LIVE_BASE
        # A shared client may serve one thread per asset. Serialize and space calls
        # to avoid T212 rate limits; T212_MIN_GAP_SEC sets the minimum gap.
        self._lock = threading.Lock()
        self._order_lock = threading.Lock()
        self._last = 0.0
        self._last_order = 0.0
        self._backoff_until = 0.0
        self._fetch_fail: dict[str, float] = {}
        self._last_attr_fetch: dict[str, float] = {}
        self._min_gap = safe_nonneg_float(
            required_float_env("T212_MIN_GAP_SEC") if min_gap_sec is None else min_gap_sec, default=0.0
        )
        self._order_min_gap = safe_nonneg_float(
            required_float_env("T212_ORDER_MIN_GAP_SEC") if order_min_gap_sec is None else order_min_gap_sec, default=0.0
        )
        # Brief shared account-level caches coalesce redundant per-asset thread
        # reads. A six-second TTL stays above the approximate five-second limits
        # on portfolio and order endpoints.
        self._pf_cache: tuple[float, list] | None = None
        self._ord_cache: tuple[float, list] | None = None
        self._cash_cache: tuple[float, dict] | None = None
        self._hist_cache: tuple[float, list] | None = None
        self._instr_cache: tuple[float, list] | None = None
        self._pf_ttl = safe_nonneg_float(
            required_float_env("T212_PORTFOLIO_TTL_SEC") if portfolio_ttl_sec is None else portfolio_ttl_sec,
            default=6.0,
        )
        # On TTL expiry, one thread fetches while others wait and reuse its result.
        self._fetch_lock = threading.Lock()
        self._lock_tls = threading.local()
        key_hash = hashlib.sha256(self.api_key.encode("utf-8", errors="ignore")).hexdigest()[:10]
        self._shared_cache_path = (
            str(shared_cache_path)
            if shared_cache_path is not None
            else os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                f".t212_shared_cache_{self.env}_{key_hash}.json",
            )
        )
        self._trade_cache_path: str | None = str(trade_cache_path) if trade_cache_path is not None else None
        self._enable_shared_disk_cache: bool | None = (
            bool(enable_shared_disk_cache) if enable_shared_disk_cache is not None else None
        )

    def _shared_cache_enabled(self) -> bool:
        if self._enable_shared_disk_cache is not None:
            return bool(self._enable_shared_disk_cache)
        return is_shared_disk_cache_enabled() and bool(getattr(self, "_shared_cache_path", ""))

    @contextlib.contextmanager
    def _shared_cache_file_lock(self):
        if not self._shared_cache_enabled():
            yield
            return
        tls = getattr(self, "_lock_tls", None)
        if tls is None:
            tls = self._lock_tls = threading.local()
        depth = getattr(tls, "depth", 0)
        if depth > 0:
            tls.depth = depth + 1
            try:
                yield
            finally:
                tls.depth = depth
            return
        lock_path = f"{self._shared_cache_path}.lock"
        tls.depth = 1
        lock_cm = FileLock(lock_path)
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
            tls.depth = 0

    def _instr_shared_cache_path(self) -> str:
        base_path = str(getattr(self, "_shared_cache_path", "") or "").strip()
        if not base_path:
            return ""
        if base_path.endswith(".json"):
            return f"{base_path[:-5]}_instruments.json"
        return f"{base_path}.instruments"

    @staticmethod
    def _cleanup_stale_tmp_files(target_path: str, max_age_sec: float = STALE_STAGING_MAX_AGE_SEC) -> None:
        if target_path:
            abs_p = os.path.abspath(target_path)
            cleanup_stale_staging_siblings(
                os.path.dirname(abs_p), f".{os.path.basename(abs_p)}.", suffix=".tmp", max_age_sec=max_age_sec
            )

    def _read_shared_cache(self, attr: str, max_age_sec: float) -> tuple[bool, list | dict | None, float]:
        if not self._shared_cache_enabled() or max_age_sec <= 0.0:
            return False, None, 0.0
        paths = (
            (self._instr_shared_cache_path(), self._shared_cache_path)
            if attr == "_instr_cache"
            else (self._shared_cache_path,)
        )
        now_ts = time.time()
        for cpath in paths:
            if not cpath:
                continue
            try:
                payload = read_json_dict(cpath, default=None)
                if not isinstance(payload, dict):
                    continue
                entry = payload.get(attr)
                if not isinstance(entry, dict):
                    continue
                ts = safe_pos_float(entry.get("ts"), default=None)
                data = entry.get("data")
                if ts is None or not isinstance(data, (list, dict)):
                    continue
                age = now_ts - ts
                if 0.0 <= age < max_age_sec:
                    return True, data, age
                fail_ts = safe_pos_float(entry.get("fail_ts"), default=None)
                if fail_ts is not None:
                    fail_age = now_ts - fail_ts
                    stale_cap = max(600.0, max_age_sec * 100.0)
                    if 0.0 <= fail_age < max_age_sec and 0.0 <= age < stale_cap:
                        return True, data, min(age, max_age_sec * 0.5)
            except Exception:
                pass
        return False, None, 0.0

    def _record_shared_cache_fail(self, attr: str) -> None:
        if not self._shared_cache_enabled() or attr == "_instr_cache":
            return
        try:
            with self._shared_cache_file_lock():
                payload = read_json_dict(self._shared_cache_path, default=None)
                if not isinstance(payload, dict):
                    return
                entry = payload.get(attr)
                if not isinstance(entry, dict) or not isinstance(entry.get("data"), (list, dict)):
                    return
                entry["fail_ts"] = time.time()
                atomic_write_json(self._shared_cache_path, payload)
        except Exception:
            pass

    def _write_shared_cache(self, attr: str, data: list | dict) -> None:
        if not self._shared_cache_enabled():
            return
        now_ts = time.time()
        try:
            if attr == "_instr_cache":
                instr_path = self._instr_shared_cache_path() or self._shared_cache_path
                self._cleanup_stale_tmp_files(instr_path)
                atomic_write_json(instr_path, {"_instr_cache": {"ts": now_ts, "data": data}})
                return
            with self._shared_cache_file_lock():
                self._cleanup_stale_tmp_files(self._shared_cache_path)
                payload = read_json_dict(self._shared_cache_path, default={})
                if not isinstance(payload, dict):
                    payload = {}
                legacy_instr = payload.pop("_instr_cache", None)
                if isinstance(legacy_instr, dict):
                    instr_path = self._instr_shared_cache_path()
                    if instr_path and not os.path.exists(instr_path):
                        try:
                            atomic_write_json(instr_path, {"_instr_cache": legacy_instr})
                        except Exception:
                            pass
                payload[attr] = {"ts": now_ts, "data": data}
                atomic_write_json(self._shared_cache_path, payload)
            if attr == "_hist_cache" and isinstance(data, list):
                self._sync_trade_cache(data)
        except Exception:
            pass

    def notify_rate_limit(self, backoff_sec: float = 5.0) -> None:
        """Pause all subsequent requests across all threads after HTTP 429."""
        wait_sec = safe_nonneg_float(backoff_sec, default=5.0)
        with self._lock:
            self._backoff_until = max(self._backoff_until, time.monotonic() + wait_sec)

    def _invalidate_order_caches_locked(self) -> None:
        for attr in ("_ord_cache", "_pf_cache", "_cash_cache", "_hist_cache"):
            setattr(self, attr, None)
            self._fetch_fail.pop(attr, None)
        if self._shared_cache_enabled():
            try:
                with self._shared_cache_file_lock():
                    payload = read_json_dict(self._shared_cache_path, default=None)
                    if isinstance(payload, dict) and "_instr_cache" in payload:
                        instr_path = self._instr_shared_cache_path()
                        if instr_path and not os.path.exists(instr_path):
                            atomic_write_json(instr_path, {"_instr_cache": payload["_instr_cache"]})
                    if os.path.exists(self._shared_cache_path):
                        os.remove(self._shared_cache_path)
            except Exception:
                pass

    def _pace(self, endpoint_key: str | None = None) -> None:
        """Enforce at least ``_min_gap`` seconds between calls across all threads,
        per-endpoint ``_order_min_gap`` when ``endpoint_key`` is given, and respect
        rate limit backoff if any recent call returned HTTP 429."""
        with self._lock:
            now = time.monotonic()
            gap_wait = self._min_gap - (now - self._last)
            backoff_wait = self._backoff_until - now
            ep_wait = 0.0
            last_attr_map = getattr(self, "_last_attr_fetch", None)
            min_ep_gap = safe_nonneg_float(getattr(self, "_order_min_gap", 0.0), default=0.0)
            if endpoint_key and isinstance(last_attr_map, dict) and min_ep_gap > 0.0:
                last_ep = last_attr_map.get(endpoint_key)
                if last_ep is not None:
                    ep_wait = min_ep_gap - (now - last_ep)
            wait = max(gap_wait, backoff_wait, ep_wait)
            if wait > 0:
                time.sleep(wait)
            now_after = time.monotonic()
            self._last = now_after
            if endpoint_key and isinstance(last_attr_map, dict):
                last_attr_map[endpoint_key] = now_after

    def _pace_order(self) -> None:
        """Enforce at least ``_order_min_gap`` seconds between mutating order calls
        across all threads, and respect rate limit backoff if any recent call returned HTTP 429."""
        with self._order_lock:
            now = time.monotonic()
            gap_wait = self._order_min_gap - (now - self._last_order)
            with self._lock:
                backoff_wait = self._backoff_until - now
            wait = max(gap_wait, backoff_wait)
            if wait > 0:
                time.sleep(wait)
            self._last_order = time.monotonic()

    # -- auth / headers --------------------------------------------------------
    def _headers(self, endpoint_key: str | None = None) -> dict:
        # Every request builds headers here, providing one throttle point across threads.
        self._pace(endpoint_key)
        auth = (
            f"Basic {base64.b64encode(f'{self.api_key}:{self.api_secret}'.encode()).decode()}"
            if self.api_secret
            else self.api_key
        )
        return {
            "Authorization": auth,
            "User-Agent": _BROWSER_UA,  # Avoid Cloudflare 403 responses.
            "Accept": "application/json",
        }

    # -- Instruments. ----------------------------------------------------------
    def _read_cached_dict_list(self, attr: str, path: str, what: str, *, ttl_mult: float = 1.0) -> list[dict] | None:
        res = self._read_cached(attr, path, what, expect_type=list, ttl_mult=ttl_mult)
        return [item for item in res if isinstance(item, dict)] if isinstance(res, list) else None

    def list_instruments(self) -> list[dict] | None:
        # Instrument metadata changes rarely; cache it across threads and restarts
        # through the shared anti-stampede cache to avoid 1 req / 50s rate limits.
        return self._read_cached_dict_list(
            "_instr_cache", "/equity/metadata/instruments", "instruments metadata", ttl_mult=50.0,
        )

    # -- Orders. ---------------------------------------------------------------
    def _place_order(self, endpoint: str, payload: dict) -> tuple[int, dict]:
        """Send an order without retrying non-idempotent placement endpoints."""
        self._pace_order()
        log(f"  [ORDER] payload: {encode_json_payload(payload)}")
        status, body = http_post_json(
            f"{self.base}/equity/orders/{endpoint}",
            payload=payload,
            headers=self._headers(),
        )
        if status == 429:
            self.notify_rate_limit(max(5.0, self._pf_ttl))
        try:
            parsed = parse_json_payload(body) if body else {}
            data = parsed if isinstance(parsed, dict) else {"raw": parsed}
        except ValueError:
            data = {"raw": body.decode(errors="replace")[:500]}
        # Invalidate unconditionally on non-429 (including ambiguous failure),
        # so reconciliation reads fresh state and does not drop a newly placed order
        # from tracking. On 429, keep the cache intact to avoid an immediate cascading
        # rate limit on subsequent active orders reads.
        if status != 429:
            with self._lock:
                self._invalidate_order_caches_locked()
        return status, data

    @staticmethod
    def _clean_ticker(ticker: str) -> str:
        if not ticker or isinstance(ticker, bool) or not str(ticker).strip():
            raise ValueError(f"invalid T212 ticker: {ticker!r}")
        return str(ticker).strip()

    def place_limit_order(
        self,
        ticker: str,
        quantity: float,
        limit_price: float,
        validity: str = "DAY",
    ) -> tuple[int, dict]:
        """Place a limit order and return HTTP status with response payload."""
        clean_ticker = self._clean_ticker(ticker)
        if isinstance(quantity, bool) or isinstance(limit_price, bool):
            raise ValueError("place_limit_order quantity and limit_price cannot be bool")
        val_u = str(validity or "").strip().upper() if not isinstance(validity, bool) else ""
        if val_u not in {"DAY", "GOOD_TILL_CANCEL"}:
            raise ValueError(f"invalid T212 timeValidity: {validity!r}")
        qty_f = safe_float(quantity, default=None)
        price_f = safe_pos_float(limit_price, default=None)
        if qty_f is None or price_f is None:
            raise ValueError("place_limit_order requires finite quantity and positive limit_price")
        qty_r = round(qty_f, 2)
        price_r = round(price_f, 2)
        if qty_r == 0.0 or price_r <= 0.0:
            raise ValueError("place_limit_order rounded quantity must be non-zero and limit_price positive")
        return self._place_order("limit", {
            "ticker": clean_ticker,
            "quantity": qty_r,   # Positive buys and negative sells.
            "limitPrice": price_r,
            "timeValidity": val_u,
        })

    def place_market_order(
        self,
        ticker: str,
        quantity: float,
        extended_hours: bool = False,
    ) -> tuple[int, dict]:
        """Place a market order, using quantity sign for direction as with limits."""
        clean_ticker = self._clean_ticker(ticker)
        if isinstance(quantity, bool):
            raise ValueError("place_market_order quantity cannot be bool")
        qty_f = safe_float(quantity, default=None)
        if qty_f is None:
            raise ValueError("place_market_order requires finite quantity")
        qty_r = round(qty_f, 2)
        if qty_r == 0.0:
            raise ValueError("place_market_order rounded quantity must be non-zero")
        ext_flag = _parse_bool_flag(extended_hours, default=False)
        return self._place_order("market", {
            "ticker": clean_ticker,
            "quantity": qty_r,
            "extendedHours": ext_flag,
        })

    def _sync_trade_cache(self, items: list) -> None:
        explicit_path = getattr(self, "_trade_cache_path", None)
        if not explicit_path and not is_shared_disk_cache_enabled(getattr(self, "_enable_shared_disk_cache", None)):
            return
        by_symbol = T212Provider.normalize_history_items(items)
        if not by_symbol:
            return
        target_path = str(explicit_path) if explicit_path else str(cache_path(CACHE_T212_TRADE_FILENAME))
        from cacheaccount import CacheTradeManager

        CacheTradeManager.merge_venue_fills_snapshot(target_path, by_symbol)

    def get_historical_order(self, order_id) -> dict | bool | None:
        """Find a terminal order in the first history page.

        The id endpoint focuses on pending orders, which may disappear after fill or
        cancellation. Newly closed orders fit in the first 50-item page. Return None
        when history is unavailable or not yet updated so strict reconciliation retries
        without assuming a fill.
        """
        target_oid = clean_order_id(order_id)
        if target_oid is None:
            return None
        matches: list[tuple[dict, dict | None]] = []
        with self._lock:
            hc = getattr(self, "_hist_cache", None)
            if hc and (time.monotonic() - hc[0]) < self._pf_ttl:
                matches = T212Provider._extract_history_matches(hc[1], target_oid)
        if not matches:
            with self._fetch_lock:
                with self._shared_cache_file_lock():
                    with self._lock:
                        hc = getattr(self, "_hist_cache", None)
                        if hc and (time.monotonic() - hc[0]) < self._pf_ttl:
                            matches = T212Provider._extract_history_matches(hc[1], target_oid)
                    if not matches:
                        s_hit, s_val, s_age = self._read_shared_cache("_hist_cache", self._pf_ttl)
                        if s_hit and isinstance(s_val, list):
                            s_matches = T212Provider._extract_history_matches(s_val, target_oid)
                            if s_matches:
                                with self._lock:
                                    self._hist_cache = (time.monotonic() - s_age, s_val)
                                    self._fetch_fail.pop("_hist_cache", None)
                                matches = s_matches
                    if not matches:
                        items = self._fetch_order_history_locked()
                        if items is None:
                            s_fb_hit, s_fb_val, s_fb_age = self._read_shared_cache("_hist_cache", 600.0)
                            if s_fb_hit and isinstance(s_fb_val, list):
                                self._record_shared_cache_fail("_hist_cache")
                                matches = T212Provider._extract_history_matches(s_fb_val, target_oid)
                            if not matches:
                                with self._lock:
                                    hc = getattr(self, "_hist_cache", None)
                                    if hc and (time.monotonic() - hc[0]) < 600.0:
                                        matches = T212Provider._extract_history_matches(hc[1], target_oid)
                            if not matches:
                                return None
                        else:
                            matches = T212Provider._extract_history_matches(items, target_oid)
        if not matches:
            return False

        # History uses {order, fill}, and one order may have several fills. Return
        # the client's order shape enriched with actual aggregate execution data.
        result = T212Provider._aggregate_history_fills(matches)
        result.pop("_timestamp_ms", None)
        return result

    def _fetch_order_history_locked(self) -> list | None:
        """Fetch latest 50 order history entries under ``_fetch_lock`` and persist caches."""
        status, body = http_get(
            f"{self.base}/equity/history/orders?limit=50",
            headers=self._headers("_hist_cache"),
        )
        if status == 429:
            self.notify_rate_limit(max(5.0, self._pf_ttl))
        if status != 200 or not body:
            self._log_read_fail("order history", status)
            with self._lock:
                self._fetch_fail["_hist_cache"] = time.monotonic()
            return None
        try:
            payload = parse_json_payload(body)
        except ValueError:
            self._log_read_fail("order history (invalid JSON)", status)
            with self._lock:
                self._fetch_fail["_hist_cache"] = time.monotonic()
            return None
        items = payload.get("items", []) if isinstance(payload, dict) else payload
        if isinstance(items, list):
            with self._lock:
                self._hist_cache = (time.monotonic(), items)
                self._fetch_fail.pop("_hist_cache", None)
            self._write_shared_cache("_hist_cache", items)
            return items
        return []

    def refresh_order_history(self, *, max_age_sec: float | None = None) -> list | None:
        """Return recent order history items, refreshing from T212 and syncing trade cache when stale."""
        ttl = max(1.0, safe_nonneg_float(max_age_sec, default=self._pf_ttl))
        with self._lock:
            hc = getattr(self, "_hist_cache", None)
            if hc and (time.monotonic() - hc[0]) < ttl and isinstance(hc[1], list):
                return hc[1]
        with self._fetch_lock:
            with self._shared_cache_file_lock():
                with self._lock:
                    hc = getattr(self, "_hist_cache", None)
                    if hc and (time.monotonic() - hc[0]) < ttl and isinstance(hc[1], list):
                        return hc[1]
                s_hit, s_val, s_age = self._read_shared_cache("_hist_cache", ttl)
                if s_hit and isinstance(s_val, list):
                    with self._lock:
                        self._hist_cache = (time.monotonic() - s_age, s_val)
                        self._fetch_fail.pop("_hist_cache", None)
                    self._sync_trade_cache(s_val)
                    return s_val
                return self._fetch_order_history_locked()

    def latest_filled_order_for_ticker(
        self,
        ticker: str,
        side: str | None = None,
        *,
        refresh: bool = False,
        max_age_sec: float | None = None,
    ) -> dict | None:
        """Return the most recent normalized FILLED order for ``ticker`` (and optional ``side``)."""
        tkr = str(ticker or "").strip().upper()
        if not tkr:
            return None
        items = self.refresh_order_history(max_age_sec=max_age_sec) if refresh else None
        if not isinstance(items, list):
            with self._lock:
                hc = getattr(self, "_hist_cache", None)
                items = hc[1] if (hc and isinstance(hc[1], list)) else None
        if not isinstance(items, list):
            s_hit, s_val, _ = self._read_shared_cache("_hist_cache", 86400.0)
            items = s_val if (s_hit and isinstance(s_val, list)) else []
        by_sym = T212Provider.normalize_history_items(items)
        sym_orders = by_sym.get(tkr)
        if not sym_orders:
            base_prefix = tkr.split("_", 1)[0]
            for k, v in by_sym.items():
                if k.split("_", 1)[0] == base_prefix:
                    sym_orders = v
                    break
        if not sym_orders:
            return None
        target_side = normalize_side(side) if (isinstance(side, str) and side.strip()) else None
        candidates = [
            o for o in sym_orders
            if isinstance(o, dict) and (target_side is None or o.get("side") == target_side)
        ]
        return candidates[-1] if candidates else None

    def get_order_status(self, order_id) -> dict | bool | None:
        target_oid = clean_order_id(order_id)
        if target_oid is None:
            return None
        status, body = http_get(
            f"{self.base}/equity/orders/{target_oid}",
            headers=self._headers("_order_status"),
        )
        if status == 404:
            return self.get_historical_order(target_oid)
        if status == 429:
            self.notify_rate_limit(max(5.0, self._pf_ttl))
        if status != 200:
            self._log_read_fail("order status", status)
            return None
        try:
            parsed = parse_json_payload(body)
        except (TypeError, ValueError):
            self._log_read_fail("order status (invalid JSON)", status)
            return None
        if not isinstance(parsed, dict):
            self._log_read_fail("order status (non-dict JSON)", status)
            return None
        return parsed

    def cancel_order(self, order_id) -> bool:
        """Cancel an order by id and return whether the request was accepted."""
        target_oid = clean_order_id(order_id)
        if target_oid is None:
            return False
        self._pace_order()
        status, _ = http_request("DELETE", f"{self.base}/equity/orders/{target_oid}",
                                 headers=self._headers())
        if status == 429:
            self.notify_rate_limit(max(5.0, self._pf_ttl))
        ok = status in (200, 201, 204)
        if not ok:
            log(f"  ! [T212] cancel order {target_oid} -> HTTP {status}")
        if status != 429:
            with self._lock:
                self._invalidate_order_caches_locked()
        return ok

    def _log_read_fail(self, what: str, status: int) -> None:
        """Log a read-failure reason once per endpoint/status every 60 seconds.

        Debouncing prevents repeated 429 noise while preserving an observable reason
        such as rate limit, authentication, or timeout.
        """
        now = time.monotonic()
        cache = getattr(self, "_read_fail_log", None)
        if cache is None:
            cache = self._read_fail_log = {}
        last = cache.get(what)
        if last and last[0] == status and (now - last[1]) < 60:
            return                                  # Same recent reason; do not log again.
        cache[what] = (status, now)
        if status == 429:
            log(f"  ! T212 {what}: rate limit (429) - temporarily unavailable")
        elif status in (401, 403):
            log(f"  ! T212 {what}: auth failed ({status}) - check the key")
        elif status == 0:
            log(f"  ! T212 {what}: timeout/network error (status 0)")
        else:
            log(f"  ! T212 {what}: HTTP {status}")

    def _peek_cached_locked(self, attr: str, eff_ttl: float | None = None) -> tuple[bool, list | dict | None]:
        c = getattr(self, attr, None)
        now = time.monotonic()
        ttl = self._pf_ttl if eff_ttl is None else eff_ttl
        if c and (now - c[0]) < ttl:
            return True, c[1]
        fail_ts = self._fetch_fail.get(attr, 0.0)
        if (now - fail_ts) < self._pf_ttl:
            stale_limit = max(600.0, ttl * 100.0)
            return True, (c[1] if (c and (now - c[0]) < stale_limit) else None)
        return False, None

    def _read_cached(
        self,
        attr: str,
        path: str,
        what: str,
        *,
        expect_type: type = list,
        ttl_mult: float = 1.0,
    ) -> list | dict | None:
        """GET through a TTL cache with double-checked anti-stampede locking.

        Without it, every asset thread crosses an expired TTL simultaneously and
        triggers near-simultaneous requests against roughly five-second limits.
        """
        eff_ttl = max(self._pf_ttl, self._pf_ttl * safe_pos_float(ttl_mult, default=1.0))
        stale_limit = max(600.0, eff_ttl * 100.0)
        with self._lock:
            hit, val = self._peek_cached_locked(attr, eff_ttl)
            if hit:
                return val
        with self._fetch_lock:
            with self._shared_cache_file_lock():
                with self._lock:   # Another thread may have refilled while this one waited.
                    hit, val = self._peek_cached_locked(attr, eff_ttl)
                    if hit:
                        return val
                s_hit, s_val, s_age = self._read_shared_cache(attr, eff_ttl)
                if s_hit and isinstance(s_val, expect_type):
                    with self._lock:
                        setattr(self, attr, (time.monotonic() - s_age, s_val))
                        self._fetch_fail.pop(attr, None)
                    return s_val
                status, body = http_get(f"{self.base}{path}", headers=self._headers(attr))
                if status != 200 or not body:
                    s_fb_hit, s_fb_val, s_fb_age = self._read_shared_cache(attr, stale_limit)
                    if s_fb_hit and isinstance(s_fb_val, expect_type):
                        self._record_shared_cache_fail(attr)
                        with self._lock:
                            setattr(self, attr, (time.monotonic() - min(s_fb_age, eff_ttl * 0.5), s_fb_val))
                            self._fetch_fail[attr] = time.monotonic()
                        return s_fb_val
                    if status == 429:
                        self.notify_rate_limit(max(5.0, self._pf_ttl))
                    self._log_read_fail(what, status)
                    with self._lock:
                        self._fetch_fail[attr] = time.monotonic()
                        c = getattr(self, attr, None)
                        if c and (time.monotonic() - c[0]) < stale_limit:
                            return c[1]
                    return None
                try:
                    data = parse_json_payload(body)
                except ValueError:
                    self._log_read_fail(f"{what} (invalid JSON)", status)
                    with self._lock:
                        self._fetch_fail[attr] = time.monotonic()
                    return None
                if not isinstance(data, expect_type):
                    return None
                if expect_type is list:
                    data = [item for item in data if isinstance(item, dict)]
                with self._lock:
                    setattr(self, attr, (time.monotonic(), data))
                    self._fetch_fail.pop(attr, None)
                self._write_shared_cache(attr, data)
                return data

    def get_portfolio(self) -> list[dict] | None:
        """Return open account positions, the reconciliation source of truth.

        A brief shared TTL coalesces all asset-thread requests into one call.
        """
        return self._read_cached_dict_list("_pf_cache", "/equity/portfolio", "portfolio")

    def list_active_orders(self) -> list[dict] | None:
        """Return pending orders; executed orders disappear into the portfolio.

        Share the portfolio TTL across threads and invalidate on placement/cancel so
        reconciliation immediately sees new orders instead of incorrectly dropping
        them or marking a take-profit tranche as sold.
        """
        return self._read_cached_dict_list("_ord_cache", "/equity/orders", "active orders")

    def get_cash(self) -> dict | None:
        """Return account cash breakdown (free, invested, blocked, total) in account base currency."""
        res = self._read_cached("_cash_cache", "/equity/account/cash", "account cash", expect_type=dict)
        return res if isinstance(res, dict) else None
