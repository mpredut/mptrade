#!/usr/bin/env python3
"""Minimal Trading 212 API client for instruments and order lifecycle.

The documented limit-order schema is:
    POST /equity/orders/limit
    {"ticker": "...", "quantity": <+BUY/-SELL>, "limitPrice": ..., "timeValidity": "DAY"|"GOOD_TILL_CANCEL"}
There is no ``side`` field, and the ticker key is not ``instrumentTicker``.
"""

from __future__ import annotations

import base64
import json
import threading
import time

try:
    # Package import for the generic provider or installed wheel.
    from .ipo_common import _parse_bool_flag, http_get, http_post_json, http_request, log, required_env, required_float_env, safe_float, safe_nonneg_float, safe_pos_float
except ImportError:
    # Compatibility with direct launch from the 212trading directory.
    import os as _os
    import sys as _sys
    _HERE = _os.path.dirname(_os.path.abspath(__file__))
    _ROOT = _os.path.dirname(_HERE)
    if _ROOT not in _sys.path:
        _sys.path.insert(0, _ROOT)
    from botcore import ensure_local_module_priority
    ensure_local_module_priority(_HERE, "ipo_common")
    from ipo_common import _parse_bool_flag, http_get, http_post_json, http_request, log, required_env, required_float_env, safe_float, safe_nonneg_float, safe_pos_float

from providers.base import _clean_single_order_id, extract_order_id

LIVE_BASE = "https://live.trading212.com/api/v0"
DEMO_BASE = "https://demo.trading212.com/api/v0"

_BROWSER_UA = "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0"


class T212Client:
    def __init__(self, api_key: str, api_secret: str | None = None, env: str | None = None,
                 *, min_gap_sec: float | None = None,
                 portfolio_ttl_sec: float | None = None,
                 order_min_gap_sec: float | None = None):
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
        self._pf_ttl = safe_nonneg_float(
            required_float_env("T212_PORTFOLIO_TTL_SEC") if portfolio_ttl_sec is None else portfolio_ttl_sec,
            default=6.0,
        )
        # On TTL expiry, one thread fetches while others wait and reuse its result.
        self._fetch_lock = threading.Lock()

    def notify_rate_limit(self, backoff_sec: float = 5.0) -> None:
        """Pause all subsequent requests across all threads after HTTP 429."""
        wait_sec = safe_nonneg_float(backoff_sec, default=5.0)
        with self._lock:
            self._backoff_until = max(self._backoff_until, time.monotonic() + wait_sec)

    def _invalidate_order_caches_locked(self) -> None:
        self._ord_cache = None
        self._fetch_fail.pop("_ord_cache", None)
        self._pf_cache = None
        self._fetch_fail.pop("_pf_cache", None)
        self._cash_cache = None
        self._fetch_fail.pop("_cash_cache", None)

    def _pace(self) -> None:
        """Enforce at least ``_min_gap`` seconds between calls across all threads,
        and respect rate limit backoff if any recent call returned HTTP 429."""
        with self._lock:
            now = time.monotonic()
            gap_wait = self._min_gap - (now - self._last)
            backoff_wait = self._backoff_until - now
            wait = max(gap_wait, backoff_wait)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

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
    def _auth(self) -> str:
        if self.api_secret:
            token = base64.b64encode(f"{self.api_key}:{self.api_secret}".encode()).decode()
            return f"Basic {token}"
        return self.api_key

    def _headers(self) -> dict:
        # Every request builds headers here, providing one throttle point across threads.
        self._pace()
        return {
            "Authorization": self._auth(),
            "User-Agent": _BROWSER_UA,  # Avoid Cloudflare 403 responses.
            "Accept": "application/json",
        }

    # -- Instruments. ----------------------------------------------------------
    def list_instruments(self) -> list[dict] | None:
        # Instrument metadata changes rarely; cache it to avoid expensive per-asset
        # startup calls that trigger rate limits.
        c = getattr(self, "_instr_cache", None)
        if c and (time.monotonic() - c[0]) < 300:
            return c[1]
        status, body = http_get(f"{self.base}/equity/metadata/instruments", headers=self._headers())
        if status == 429:
            self.notify_rate_limit(max(5.0, self._pf_ttl))
            log("  ! T212 rate limit (429)")
            return None
        if status in (401, 403):
            log(f"  ! T212 auth failed ({status}) - check the key")
            return None
        if status != 200 or not body:
            return None
        try:
            data = json.loads(body)
            if not isinstance(data, list):
                return None
            clean = [i for i in data if isinstance(i, dict)]
            self._instr_cache = (time.monotonic(), clean)
            return clean
        except ValueError:
            return None

    def search_instruments(self, ticker_substr: str, name_patterns: tuple[str, ...]) -> list[dict] | None:
        """Search by ticker substring or name/shortName pattern."""
        t_q = str(ticker_substr or "").strip().upper() if not isinstance(ticker_substr, bool) else ""
        clean_patterns = tuple(
            str(p).strip().lower()
            for p in (name_patterns or ())
            if not isinstance(p, bool) and str(p or "").strip()
        )
        if not t_q and not clean_patterns:
            return None
        instruments = self.list_instruments()
        if instruments is None:
            return None
        hits = []
        for ins in instruments:
            if not isinstance(ins, dict):
                continue
            ticker = str(ins.get("ticker", ""))
            name   = str(ins.get("name", "")).lower()
            short  = str(ins.get("shortName", "")).lower()
            if (
                (t_q and t_q in ticker.upper())
                or any(p in name  for p in clean_patterns)
                or any(p in short for p in clean_patterns)
            ):
                hits.append(ins)
        return hits or None

    # -- Orders. ---------------------------------------------------------------
    def _place_order(self, endpoint: str, payload: dict) -> tuple[int, dict]:
        """Send an order without retrying non-idempotent placement endpoints."""
        self._pace_order()
        log(f"  [ORDER] payload: {json.dumps(payload)}")
        status, body = http_post_json(
            f"{self.base}/equity/orders/{endpoint}",
            payload=payload,
            headers=self._headers(),
        )
        if status == 429:
            self.notify_rate_limit(max(5.0, self._pf_ttl))
        try:
            parsed = json.loads(body) if body else {}
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

    _clean_order_id = staticmethod(_clean_single_order_id)

    def get_historical_order(self, order_id) -> dict | bool | None:
        """Find a terminal order in the first history page.

        The id endpoint focuses on pending orders, which may disappear after fill or
        cancellation. Newly closed orders fit in the first 50-item page. Return None
        when history is unavailable or not yet updated so strict reconciliation retries
        without assuming a fill.
        """
        target_oid = self._clean_order_id(order_id)
        if target_oid is None:
            return None
        status, body = http_get(
            f"{self.base}/equity/history/orders?limit=50",
            headers=self._headers(),
        )
        if status == 429:
            self.notify_rate_limit(max(5.0, self._pf_ttl))
        if status != 200 or not body:
            self._log_read_fail("order history", status)
            return None
        try:
            payload = json.loads(body)
        except ValueError:
            self._log_read_fail("order history (invalid JSON)", status)
            return None
        items = payload.get("items", []) if isinstance(payload, dict) else payload
        matches = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            order = item.get("order") if isinstance(item.get("order"), dict) else item
            cand_oid = extract_order_id(order)
            if cand_oid is not None and cand_oid == target_oid:
                matches.append((order, item.get("fill")))
        if not matches:
            return False

        # History uses {order, fill}, and one order may have several fills. Return
        # the client's order shape enriched with actual aggregate execution data.
        result = dict(matches[0][0])
        total_qty = total_cost = total_fee = priced_qty = 0.0
        fee_currencies = set()
        for _, fill in matches:
            if not isinstance(fill, dict):
                continue
            qf = safe_float(fill.get("quantity"), default=None)
            pf = safe_float(fill.get("price"), default=None)
            if qf is None or pf is None:
                continue
            qty = abs(qf)
            price = abs(pf)
            total_qty += qty
            if price > 0.0 and qty > 0.0:
                total_cost += qty * price
                priced_qty += qty
            wallet = fill.get("walletImpact")
            taxes = wallet.get("taxes") if isinstance(wallet, dict) else []
            for tax in taxes or []:
                if not isinstance(tax, dict):
                    continue
                tqf = safe_float(tax.get("quantity"), default=None)
                if tqf is None:
                    continue
                total_fee += abs(tqf)
                raw_cur = tax.get("currency")
                currency = raw_cur.strip().upper() if isinstance(raw_cur, str) else ""
                if currency:
                    fee_currencies.add(currency)
        if total_qty > 0:
            result["filledQuantity"] = total_qty
            result["filledValue"] = total_cost
            if priced_qty > 0:
                result["fillPrice"] = total_cost / priced_qty
        result["fee"] = total_fee
        if fee_currencies:
            result["_feeCurrencies"] = sorted(fee_currencies)
        return result

    def get_order_status(self, order_id) -> dict | bool | None:
        target_oid = self._clean_order_id(order_id)
        if target_oid is None:
            return None
        status, body = http_get(f"{self.base}/equity/orders/{target_oid}", headers=self._headers())
        if status == 404:
            return self.get_historical_order(target_oid)
        if status == 429:
            self.notify_rate_limit(max(5.0, self._pf_ttl))
        if status != 200:
            self._log_read_fail("order status", status)
            return None
        try:
            parsed = json.loads(body)
        except (TypeError, ValueError):
            self._log_read_fail("order status (invalid JSON)", status)
            return None
        if not isinstance(parsed, dict):
            self._log_read_fail("order status (non-dict JSON)", status)
            return None
        return parsed

    def cancel_order(self, order_id) -> bool:
        """Cancel an order by id and return whether the request was accepted."""
        target_oid = self._clean_order_id(order_id)
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
            log(f"  ! T212 {what}: rate limit (429) — indisponibil temporar")
        elif status in (401, 403):
            log(f"  ! T212 {what}: auth failed ({status}) — check the key")
        elif status == 0:
            log(f"  ! T212 {what}: timeout/retea (status 0)")
        else:
            log(f"  ! T212 {what}: HTTP {status}")

    def _peek_cached_locked(self, attr: str) -> tuple[bool, list | dict | None]:
        c = getattr(self, attr)
        now = time.monotonic()
        if c and (now - c[0]) < self._pf_ttl:
            return True, c[1]
        fail_ts = self._fetch_fail.get(attr, 0.0)
        if (now - fail_ts) < self._pf_ttl:
            return True, (c[1] if (c and (now - c[0]) < 60.0) else None)
        return False, None

    def _read_cached(self, attr: str, path: str, what: str) -> list | dict | None:
        """GET through a TTL cache with double-checked anti-stampede locking.

        Without it, every asset thread crosses an expired TTL simultaneously and
        triggers near-simultaneous requests against roughly five-second limits.
        """
        with self._lock:
            hit, val = self._peek_cached_locked(attr)
            if hit:
                return val
        with self._fetch_lock:
            with self._lock:   # Another thread may have refilled while this one waited.
                hit, val = self._peek_cached_locked(attr)
                if hit:
                    return val
            status, body = http_get(f"{self.base}{path}", headers=self._headers())
            if status == 429:
                self.notify_rate_limit(max(5.0, self._pf_ttl))
            if status != 200 or not body:
                self._log_read_fail(what, status)
                with self._lock:
                    self._fetch_fail[attr] = time.monotonic()
                    c = getattr(self, attr)
                    if c and (time.monotonic() - c[0]) < 60.0:
                        return c[1]
                return None
            try:
                data = json.loads(body)
            except ValueError:
                self._log_read_fail(f"{what} (JSON invalid)", status)
                with self._lock:
                    self._fetch_fail[attr] = time.monotonic()
                return None
            with self._lock:
                setattr(self, attr, (time.monotonic(), data))
                self._fetch_fail.pop(attr, None)
            return data

    def get_portfolio(self) -> list[dict] | None:
        """Return open account positions, the reconciliation source of truth.

        A brief shared TTL coalesces all asset-thread requests into one call.
        """
        res = self._read_cached("_pf_cache", "/equity/portfolio", "portofoliu")
        return [item for item in res if isinstance(item, dict)] if isinstance(res, list) else None

    def list_active_orders(self) -> list[dict] | None:
        """Return pending orders; executed orders disappear into the portfolio.

        Share the portfolio TTL across threads and invalidate on placement/cancel so
        reconciliation immediately sees new orders instead of incorrectly dropping
        them or marking a take-profit tranche as sold.
        """
        res = self._read_cached("_ord_cache", "/equity/orders", "active orders")
        return [item for item in res if isinstance(item, dict)] if isinstance(res, list) else None

    def get_cash(self) -> dict | None:
        """Return account cash breakdown (free, invested, blocked, total) in account base currency."""
        res = self._read_cached("_cash_cache", "/equity/account/cash", "account cash")
        return res if isinstance(res, dict) else None
