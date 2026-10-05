# kraken_provider.py
"""Kraken spot adapter backed by ``kraken/kraken_client.py``.

Symbols are native Kraken pairs such as ``HYPEUSD``. ``supports_symbol`` deliberately
returns false, so instruments must select this adapter explicitly and cannot collide
with Hyperliquid's HYPE routing. Public prices work without credentials; account and
execution methods require Kraken keys loaded lazily from the environment or Kraken
configuration.

The legacy ``place_order`` path submits validation-only requests unless
``KRAKEN_LIVE_ORDERS=true``. The StrategyExecutor ``submit_order`` path is intentionally
strict and always sends a real request; its caller's dry-run policy is the controlling
gate. Importing this module does not eagerly initialize the Kraken client.
"""
import hashlib
import json
import os
import re
import sys
import math
import time
from typing import Optional, List

from .base import MarketDataProvider, _normalize_order, env_value
from .strategy_executor import (
    OrderReconciliationCapabilities,
    OrderStatus,
    PairPrecision,
    ProviderError,
    SubmissionRefused,
    extract_order_id,
)
from credentials import CredentialProfileMissingError, kraken_credentials
from market_regime import ClosedPriceSeries

_KRAKEN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kraken")
# Shared cross-process fill cache produced by kraken_cachemanager.py.
_KRAKEN_CACHE_FILE = os.path.join(os.path.dirname(_KRAKEN_DIR), "cachedb", "cache_trade_kraken.json")
_CACHE_MAX_STALE_S = 30.0   # Beyond this, assume cachemanager stopped and use TradesHistory.


def _kraken_pair_key(value: str) -> str:
    """Canonicalize common Kraken legacy pair wrappers for strict comparison."""
    key = "".join(ch for ch in str(value or "").upper() if ch.isalnum())
    quote = next((candidate for candidate in (
        "USDC", "USDT", "USD", "EUR", "GBP", "CAD", "JPY", "AUD",
        "BTC", "XBT", "ETH",
    ) if key.endswith(candidate) or key.endswith("Z" + candidate)), None)
    if quote is None:
        return key.replace("XBT", "BTC")
    suffix = "Z" + quote if key.endswith("Z" + quote) else quote
    base = key[:-len(suffix)]
    if base.startswith("X") and len(base) > 3:
        base = base[1:]
    return (base + quote).replace("XBT", "BTC")


def _kraken_pair_matches(expected: str, actual: str) -> bool:
    return bool(actual) and _kraken_pair_key(expected) == _kraken_pair_key(actual)


_KRAKEN_CLIENT_ID_RE = re.compile(r"^[0-9a-fA-F]{32}$")


def _kraken_client_order_id(client_order_id: str) -> str:
    """Map one durable client ID to Kraken's 128-bit hexadecimal UUID form."""
    raw = str(client_order_id or "").strip()
    if not raw or len(raw) > 128:
        raise ProviderError("Kraken client_order_id must contain 1-128 characters")
    try:
        encoded = raw.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ProviderError("Kraken client_order_id must be ASCII") from exc
    if _KRAKEN_CLIENT_ID_RE.fullmatch(raw):
        return raw.lower()
    return hashlib.sha256(
        b"assertguardian:kraken:cloid:v1\0" + encoded).hexdigest()[:32]


def _live() -> bool:
    # os.environ takes precedence, with kraken/.env as the API-key-style fallback.
    v = os.environ.get("KRAKEN_LIVE_ORDERS")
    if v is None:
        v = env_value(_KRAKEN_DIR, "KRAKEN_LIVE_ORDERS")
    return (v or "false").strip().lower() == "true"


def kraken_strategy_dry_run(force_paper: bool, strategy_execute: bool) -> bool:
    """Combine the strategy and provider gates for Kraken orchestration."""
    return bool(force_paper or not strategy_execute or not _live())


class KrakenProvider(MarketDataProvider):
    def __init__(self, client=None):
        # kraken_bot can inject its _BOT client to share connection and nonce.
        # None lazily builds a fleet client using the _SPARE credentials.
        self._cli = client
        self._minqty = {}  # Cache symbol-to-ordermin from pair_info.

    @property
    def name(self) -> str:
        return "Kraken"

    def execution_enabled(self) -> bool:
        """Return whether the generic pipeline may create a real Kraken order."""
        return _live()

    def reconciliation_capabilities(self) -> OrderReconciliationCapabilities:
        return OrderReconciliationCapabilities(
            # ClosedOrders is bounded and therefore cannot prove that an older
            # client ID was never accepted. Keep automatic resubmission disabled;
            # the forwarded cl_ord_id remains useful for manual reconciliation.
            lookup_by_client_order_id=False,
            status_by_order_id=True,
            cancel_by_order_id=True,
            list_open_orders=True,
        )

    def supports_symbol(self, symbol: str) -> bool:
        # Explicit-only: claim no pattern and remain reachable only through Instrument.
        return False

    # -- Lazy client. ----------------------------------------------------------
    def _client(self):
        if self._cli is not None:
            return self._cli
        # Put kraken first on sys.path for lazy client imports, then restore the
        # path so other providers are unaffected. The unique kraken_common name
        # avoids collision with hyperliquid/common.py.
        saved_path = list(sys.path)
        try:
            sys.path.insert(0, _KRAKEN_DIR)
            from kraken_client import KrakenClient  # noqa: import lazy
        finally:
            sys.path[:] = saved_path                  # Restore order for other providers.
        # Read Kraken-specific credentials, with environment variables taking
        # precedence. Preserve the existing whole-profile priority: primary,
        # then SPARE, then BOT. The resolver prevents cross-profile key mixing.
        names = (
            "KRAKEN_API_KEY", "KRAKEN_API_SECRET",
            "KRAKEN_API_KEY_SPARE", "KRAKEN_API_SECRET_SPARE",
            "KRAKEN_API_KEY_BOT", "KRAKEN_API_SECRET_BOT",
        )
        values = {
            name: os.environ.get(name) or env_value(_KRAKEN_DIR, name)
            for name in names
        }
        try:
            credentials = kraken_credentials("fleet", values=values)
            api_key, api_secret = credentials.key, credentials.secret
        except CredentialProfileMissingError:
            api_key = api_secret = None
        cli = KrakenClient(api_key, api_secret)
        # Cache only a credentialed client. When keys are initially absent, retry
        # loading them on the next tick so the process can recover without restart.
        if api_key and api_secret:
            self._cli = cli
        else:
            print("[Kraken] ⚠ STARTED WITHOUT keys (kraken/.env missing or incomplete at the first account call)"
                  " -> the account reads (balance/orders) will fail; NOT caching, retrying on the next tick.")
        return cli

    # -- Public market data without keys. --------------------------------------
    def get_current_price(self, symbol: str) -> Optional[float]:
        try:
            return self._client().last_price(symbol)
        except Exception as e:  # noqa: BLE001
            print(f"[Kraken] price {symbol}: {e}")
            return None

    def min_order_qty(self, symbol: str) -> float:
        """Return public pair minimum volume, caching only a positive result.

        Do not cache zero from a failed lookup, because that would permanently
        disable minimum-volume protection and cause repeated dust rejections.
        """
        cached = self._minqty.get(symbol)
        if cached:  # Only positive values are cached; zero or absent retries.
            return cached
        mn = 0.0
        try:
            info = self._client().pair_info(symbol) or {}
            mn = float(info.get("ordermin", 0) or 0.0)
        except Exception as e:  # noqa: BLE001
            print(f"[Kraken] ordermin {symbol}: {e}")
        if mn > 0:
            self._minqty[symbol] = mn   # Cache only a successful lookup.
        return mn

    def get_price_history(self, symbol: str, lookback_h: float) -> Optional[List]:
        """Return ascending public OHLC closes as timestamp/price mappings."""
        try:
            cli = self._client()
            # Choose a minute interval that fits roughly 720 points or fewer.
            interval = max(1, int(math.ceil((lookback_h * 60.0) / 720.0)))
            res = cli._public("OHLC", {"pair": symbol, "interval": interval})
            rows = next((v for k, v in res.items() if k != "last"), None)
            if not rows:
                return None
            cutoff = time.time() - lookback_h * 3600
            out = []
            for r in rows:                       # [time, o, h, l, c, vwap, vol, cnt]
                t = float(r[0])
                if t < cutoff:
                    continue
                out.append({"timestamp": int(t * 1000), "price": float(r[4])})
            return out or None
        except Exception as e:  # noqa: BLE001
            print(f"[Kraken] history {symbol}: {e}")
            return None

    # -- Credentialed account access. -----------------------------------------
    def free_balance(self, asset: str) -> Optional[float]:
        try:
            bal = self._client().balance() or {}
            for key in (asset, "X" + asset, "Z" + asset, asset + ".F"):
                if key in bal:
                    return float(bal[key] or 0.0)
            return 0.0
        except Exception as e:  # noqa: BLE001
            print(f"[Kraken] balance {asset}: {e}")
            return None

    def get_orders(self, symbol: str, side: Optional[str], since_s: float) -> List[dict]:
        """Return own pair fills filtered by side and age.

        Prefer the fresh shared cross-process cache for one fetch across processes
        and a common guard view; otherwise fall back safely to TradesHistory.
        """
        try:
            rows = self._fills_from_cache(symbol)
            if rows is None:                                  # Missing or stale cache: direct API.
                rows = self._fills_from_api(symbol)
            if rows is None:
                raise ProviderError(f"get_orders({symbol}): fills history unavailable")
            cutoff_ms = (time.time() - since_s) * 1000.0
            want = (side or "").upper()
            out = []
            for r in rows:
                if r["timestamp"] < cutoff_ms:
                    continue
                if want and r["side"] != want:
                    continue
                out.append(_normalize_order(r))
            return out
        except ProviderError:
            raise
        except Exception as e:  # noqa: BLE001
            print(f"[Kraken] get_orders {symbol}: {e}")
            raise ProviderError(f"get_orders({symbol}): {e}") from e

    def open_orders(self, symbol: str) -> List[dict]:
        """Return a strict Binance-shaped snapshot of active Kraken orders."""
        try:
            raw_orders = self._client().open_orders()
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"open_orders({symbol}): {exc}") from exc
        if not isinstance(raw_orders, dict):
            raise ProviderError(
                f"open_orders({symbol}): payload Kraken invalid")

        normalized = []
        for order_id, raw in raw_orders.items():
            if not isinstance(raw, dict):
                raise ProviderError(
                    f"open_orders({symbol}): order {order_id} is invalid")
            description = raw.get("descr") or {}
            if not isinstance(description, dict):
                raise ProviderError(
                    f"open_orders({symbol}): invalid description for order "
                    f"{order_id}")
            pair = str(description.get("pair") or raw.get("pair") or "")
            if not pair:
                raise ProviderError(
                    f"open_orders({symbol}): order {order_id} without a pair")
            if not _kraken_pair_matches(symbol, pair):
                continue
            side = str(description.get("type") or raw.get("type") or "").upper()
            if side not in {"BUY", "SELL"}:
                raise ProviderError(
                    f"open_orders({symbol}): an invalid side for {order_id}")
            try:
                price = float(description.get("price", raw.get("price", 0.0)) or 0.0)
                original_qty = float(raw.get("vol") or 0.0)
                executed_qty = float(raw.get("vol_exec") or 0.0)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ProviderError(
                    f"open_orders({symbol}): invalid values for {order_id}") from exc
            if (not all(math.isfinite(value) for value in (
                    price, original_qty, executed_qty))
                    or price < 0 or original_qty <= 0 or executed_qty < 0
                    or executed_qty > original_qty):
                raise ProviderError(
                    f"open_orders({symbol}): invalid quantities/price for {order_id}")
            normalized.append({
                "orderId": str(order_id),
                "clientOrderId": raw.get("cl_ord_id"),
                "side": side,
                "price": price,
                "origQty": original_qty,
                "executedQty": executed_qty,
                "status": str(raw.get("status") or "open").upper(),
            })
        return normalized

    def _fills_from_cache(self, symbol: str):
        """Read symbol fills from the shared cache, or return None for API fallback."""
        try:
            with open(_KRAKEN_CACHE_FILE) as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        items = data.get("items") or {}
        ft = data.get("fetchtime") or {}
        su = symbol.upper()
        key = next((k for k in items if su in k.upper() or k.upper() in su), None)
        if key is None:
            return None
        if (time.time() * 1000.0 - float(ft.get(key, 0))) > _CACHE_MAX_STALE_S * 1000.0:
            return None                                       # Stale cachemanager data.
        return [{
            "side": "BUY" if t.get("isBuyer") else "SELL",
            "price": t.get("price"), "qty": t.get("qty"),
            "timestamp": int(t.get("time", 0)),
        } for t in items[key]]

    def _fills_from_api(self, symbol: str):
        """Fall back to direct TradesHistory, matching pre-cachemanager behavior."""
        cli = self._client()
        res = cli._private("TradesHistory")
        trades = (res or {}).get("trades", {}) or {}
        su = symbol.upper()
        rows = []
        for tr in trades.values():
            p = str(tr.get("pair", "")).upper()
            if su not in p and p not in su:
                continue
            rows.append({
                "side": "BUY" if str(tr.get("type", "")).lower() == "buy" else "SELL",
                "price": tr.get("price"), "qty": tr.get("vol"),
                "timestamp": int(float(tr.get("time", 0)) * 1000),
            })
        return rows

    # -- Placement remains dry until KRAKEN_LIVE_ORDERS=true. ------------------
    def place_order(self, symbol: str, side: str, price: float, qty: float, **kwargs):
        live = self.execution_enabled()
        s = (side or "").lower()
        s = "buy" if s.startswith("b") else "sell"
        market = bool(kwargs.get("force", False))
        ordertype = "market" if market else "limit"
        submit_price = None if market else price
        client_order_id = kwargs.get("client_order_id")
        try:
            cl_ord_id = (
                _kraken_client_order_id(client_order_id)
                if client_order_id is not None else None)
        except ProviderError as exc:
            print(f"[Kraken] place_order {symbol}: {exc}")
            return None
        if not live:
            print(f"[Kraken][DRY] would place {side} {symbol} qty={qty} @ {price} "
                  f"(real orders are disabled; set KRAKEN_LIVE_ORDERS=true)")
            try:  # Server-side validation without placement.
                return self._client().add_order(
                    symbol, s, qty, submit_price, ordertype=ordertype, validate=True,
                    cl_ord_id=cl_ord_id)
            except Exception as e:  # noqa: BLE001
                print(f"[Kraken][DRY] validate {symbol}: {e}")
                return None
        try:
            print(f"[Kraken][LIVE] {side} {symbol} qty={qty} @ {price}")
            return self._client().add_order(
                symbol, s, qty, submit_price, ordertype=ordertype, validate=False,
                cl_ord_id=cl_ord_id)
        except Exception as e:  # noqa: BLE001
            print(f"[Kraken] place_order {symbol}: {e}")
            return None

    # -- StrategyExecutor contract delegated to kraken_client. -----------------
    # Unlike place_order, this raw method returns an order id and raises typed
    # errors so strategy reconciliation can distinguish a disabled live gate from
    # a venue failure. The provider gate remains authoritative for every caller.
    def submit_order(self, symbol: str, side: str, qty: float,
                     price: Optional[float] = None, *, market: bool = False,
                     kind: Optional[str] = None,
                     client_order_id: Optional[str] = None) -> str:
        s = "buy" if (side or "").lower().startswith("b") else "sell"
        ordertype = "market" if (market or price is None) else "limit"
        try:
            order_kwargs = {"ordertype": ordertype, "validate": False}
            if client_order_id is not None:
                order_kwargs["cl_ord_id"] = _kraken_client_order_id(
                    client_order_id)
            # This is the final authoritative gate for every raw executor path,
            # including SpotDCA and trailing stop. Recheck next to the API call so
            # a switch change after orchestration cannot create a live order.
            if not self.execution_enabled():
                raise SubmissionRefused(
                    "KRAKEN_LIVE_ORDERS is not true; real execution is disabled")
            res = self._client().add_order(
                symbol, s, qty, None if ordertype == "market" else price,
                **order_kwargs) or {}
        except SubmissionRefused:
            raise
        except Exception as e:  # noqa: BLE001 — Normalize venue errors.
            raise ProviderError(f"submit_order {symbol} {s}: {e}") from e
        order_id = extract_order_id(res)
        if order_id is None:
            raise ProviderError(f"submit_order {symbol}: a response without a txid ({res})")
        return order_id

    def preflight_order(self, symbol: str, side: str, qty: float,
                        price: Optional[float] = None, *, market: bool = False,
                        kind: Optional[str] = None) -> None:
        """Reject an impossible SELL before creating a durable submit intent.

        Kraken previously received the same TP repeatedly when persisted strategy
        quantity exceeded the account balance. A read-only balance gate stops that
        deterministic venue rejection without guessing a replacement quantity or
        mutating strategy state. BUY funding remains venue-authoritative because its
        required fee/slippage reserve depends on order type and execution.
        """
        if not str(side or "").lower().startswith("s"):
            return
        try:
            requested = float(qty)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProviderError(
                f"preflight_order {symbol}: an invalid SELL quantity") from exc
        if not math.isfinite(requested) or requested <= 0:
            raise ProviderError(
                f"preflight_order {symbol}: an invalid SELL quantity")

        precision = self.pair_precision(symbol)
        base_asset = precision.base_asset if precision else ""
        if not base_asset:
            raise ProviderError(
                f"preflight_order {symbol}: base asset is unavailable")
        available = self.free_balance(base_asset)
        if available is None or not math.isfinite(float(available)):
            raise ProviderError(
                f"preflight_order {symbol}: {base_asset} balance is unavailable")
        available = max(0.0, float(available))
        decimals = precision.volume_decimals if precision else 8
        tolerance = 0.5 * (10.0 ** -max(0, decimals))
        if requested > available + tolerance:
            raise ProviderError(
                f"preflight_order {symbol}: insufficient funds SELL "
                f"qty={requested} available={available} {base_asset}"
            )

    def order_status(self, symbol: str, order_id: str) -> OrderStatus:
        try:
            res = self._client().query_orders(order_id) or {}
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"order_status {order_id}: {e}") from e
        info = res.get(order_id)
        if info is None:
            raise ProviderError(f"order_status: the order {order_id} was not found")
        return OrderStatus(
            status=str(info.get("status", "")),
            filled_qty=float(info.get("vol_exec") or 0.0),
            cost=float(info.get("cost") or 0.0),
            fee=float(info.get("fee") or 0.0),
            venue_status=str(info.get("status") or ""),
        )

    def order_by_client_id(self, symbol: str, client_order_id: str):
        """Recover an open or recently terminal Kraken order by ``cl_ord_id``."""
        wanted = _kraken_client_order_id(client_order_id)
        try:
            groups = (self._client().open_orders(), self._client().closed_orders())
        except Exception as e:  # noqa: BLE001
            raise ProviderError(
                f"order_by_client_id({symbol},{client_order_id}): {e}") from e
        for orders in groups:
            for order_id, raw in dict(orders or {}).items():
                if str(raw.get("cl_ord_id") or "").lower() != wanted:
                    continue
                descr = raw.get("descr") or {}
                pair = str(descr.get("pair") or raw.get("pair") or "")
                if pair and not _kraken_pair_matches(symbol, pair):
                    continue
                return {"orderId": str(order_id), "status": raw.get("status")}
        return None

    def cancel_order_by_id(self, symbol: str, order_id: str) -> None:
        """Cancel by id, treating an already closed or missing order as success.

        The explicit method name avoids future ambiguity; cancel_order below is the
        StrategyExecutor contract alias.
        """
        try:
            result = self._client().cancel_order(order_id) or {}
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            if "unknown order" in msg or "already" in msg:
                return                       # Idempotent: already closed or canceled.
            raise ProviderError(f"cancel_order {order_id}: {e}") from e
        if "count" in result:
            try:
                if int(result["count"]) < 1:
                    raise ProviderError(
                        f"cancel_order {order_id}: Kraken did not confirm the cancellation"
                    )
            except (TypeError, ValueError) as e:
                raise ProviderError(
                    f"cancel_order {order_id}: invalid response ({result})"
                ) from e

    def cancel_order(self, symbol: str, order_id: str) -> None:  # contract StrategyExecutor
        self.cancel_order_by_id(symbol, order_id)

    def pair_precision(self, symbol: str) -> Optional[PairPrecision]:
        try:
            info = self._client().pair_info(symbol)
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"pair_precision {symbol}: {e}") from e
        if not info:
            return None
        try:
            required = ("pair_decimals", "lot_decimals", "ordermin", "base")
            missing = [key for key in required if key not in info or info[key] in (None, "")]
            if missing:
                raise ProviderError(
                    f"pair_precision {symbol}: missing metadata {', '.join(missing)}"
                )
            return PairPrecision(
                price_decimals=int(info["pair_decimals"]),
                volume_decimals=int(info["lot_decimals"]),
                order_min=float(info["ordermin"]),
                base_asset=str(info["base"]),
            )
        except (TypeError, ValueError) as e:
            raise ProviderError(f"pair_precision {symbol}: info malformat ({e})") from e

    def ohlc_series(self, symbol: str, interval_min: int) -> ClosedPriceSeries:
        try:
            client = self._client()
            series_reader = getattr(
                client, "ohlc_closes_with_timestamps", None)
            if callable(series_reader):
                closes, timestamps = series_reader(symbol, interval_min)
                timestamps = tuple(timestamps or ())
                observed_at = timestamps[-1] if timestamps else None
            else:
                reader = getattr(client, "ohlc_closes_with_timestamp", None)
                if callable(reader):
                    closes, observed_at = reader(symbol, interval_min)
                else:
                    closes = client.ohlc_closes(symbol, interval_min)
                    observed_at = None
                timestamps = ()
            return ClosedPriceSeries(
                tuple(closes or ()),
                int(interval_min),
                observed_at,
                timestamps,
            )
        except Exception as exc:
            raise ProviderError(f"ohlc_series {symbol}: {exc}") from exc

    def ohlc_closes(self, symbol: str, interval_min: int) -> list:
        return list(self.ohlc_series(symbol, interval_min).closes)
