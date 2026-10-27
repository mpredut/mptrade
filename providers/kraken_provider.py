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

from .base import (
    MarketDataProvider,
    _normalize_order,
    bool_env,
    env_value,
    extract_order_side,
    extract_timestamp_ms,
)
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
from state_io import read_recovering_previous

_KRAKEN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kraken")
# Shared cross-process fill cache produced by kraken_cachemanager.py.
_KRAKEN_CACHE_FILE = os.path.join(os.path.dirname(_KRAKEN_DIR), "cachedb", "cache_trade_kraken.json")
_CACHE_MAX_STALE_S = 30.0   # Beyond this, assume cachemanager stopped and use TradesHistory.


def _kraken_pair_key(value: str) -> str:
    """Canonicalize common Kraken legacy pair wrappers for strict comparison."""
    if not isinstance(value, str):
        return ""
    key = "".join(ch for ch in value.upper() if ch.isalnum())
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
    if isinstance(expected, bool) or isinstance(actual, bool):
        return False
    k_exp = _kraken_pair_key(expected)
    k_act = _kraken_pair_key(actual)
    return bool(k_exp and k_act and k_exp == k_act)


def _normalize_kraken_fill(raw: dict) -> dict:
    """Normalize a Kraken fill while preserving ordertxid/orderId for order-level aggregation."""
    norm = _normalize_order(raw)
    if isinstance(raw, dict):
        order_id = raw.get("ordertxid") or raw.get("orderId")
        if order_id is not None and not isinstance(order_id, bool) and str(order_id).strip():
            norm["orderId"] = str(order_id).strip()
    return norm


_KRAKEN_CLIENT_ID_RE = re.compile(r"^[0-9a-fA-F]{32}$")


def _kraken_client_order_id(client_order_id: str) -> str:
    """Map one durable client ID to Kraken's 128-bit hexadecimal UUID form."""
    if isinstance(client_order_id, bool):
        raise ProviderError("Kraken client_order_id must contain 1-128 characters")
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
    return bool_env("KRAKEN_LIVE_ORDERS", _KRAKEN_DIR)


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
            raw_mn = info.get("ordermin", 0)
            if raw_mn is not None and not isinstance(raw_mn, bool):
                cand_mn = float(raw_mn or 0.0)
                if math.isfinite(cand_mn) and cand_mn > 0:
                    mn = cand_mn
        except Exception as e:  # noqa: BLE001
            print(f"[Kraken] ordermin {symbol}: {e}")
        if mn > 0:
            self._minqty[symbol] = mn   # Cache only a successful lookup.
        return mn

    def get_price_history(self, symbol: str, lookback_h: float) -> Optional[List]:
        """Return ascending public OHLC closes as timestamp/price mappings."""
        if isinstance(lookback_h, bool):
            return None
        try:
            lookback_f = float(lookback_h)
            if not math.isfinite(lookback_f) or lookback_f <= 0:
                return None
            cli = self._client()
            # Choose a valid Kraken OHLC minute interval that fits roughly 720 points or fewer.
            raw_interval = max(1, int(math.ceil((lookback_f * 60.0) / 720.0)))
            valid_intervals = (1, 5, 15, 30, 60, 240, 1440, 10080, 21600)
            interval = next((v for v in valid_intervals if v >= raw_interval), valid_intervals[-1])
            res = cli._public("OHLC", {"pair": symbol, "interval": interval})
            rows = next((v for k, v in res.items() if k != "last"), None)
            if not rows:
                return None
            cutoff = time.time() - lookback_f * 3600
            out = []
            for r in rows:                       # [time, o, h, l, c, vwap, vol, cnt]
                if not isinstance(r, (list, tuple)) or len(r) < 5:
                    continue
                if isinstance(r[0], bool) or isinstance(r[4], bool):
                    continue
                try:
                    t = float(r[0])
                    p = float(r[4])
                except (TypeError, ValueError, OverflowError):
                    continue
                if not (math.isfinite(t) and math.isfinite(p)) or p <= 0 or t < cutoff:
                    continue
                out.append({"timestamp": int(t * 1000), "price": p})
            return out or None
        except Exception as e:  # noqa: BLE001
            print(f"[Kraken] history {symbol}: {e}")
            return None

    # -- Credentialed account access. -----------------------------------------
    def free_balance(self, asset: str) -> Optional[float]:
        if not asset or isinstance(asset, bool) or not str(asset).strip():
            return None
        asset_u = str(asset).strip().upper()
        try:
            bal = self._client().balance() or {}
            candidates = [asset_u, "X" + asset_u, "Z" + asset_u, asset_u + ".F"]
            if asset_u in {"BTC", "XBT"}:
                candidates.extend(["BTC", "XBT", "XXBT", "XBT.F", "XXBT.F"])
            for key in candidates:
                if key in bal:
                    raw_val = bal[key]
                    if isinstance(raw_val, bool):
                        return None
                    val = float(raw_val or 0.0)
                    if not math.isfinite(val) or val < 0.0:
                        return None
                    return val
            return 0.0
        except Exception as e:  # noqa: BLE001
            print(f"[Kraken] balance {asset}: {e}")
            return None

    def get_orders(self, symbol: str, side: Optional[str], since_s: float) -> List[dict]:
        """Return own pair orders/fills filtered by side and age, grouped by parent orderId.

        Prefer the fresh shared cross-process cache for one fetch across processes
        and a common guard view; otherwise fall back safely to TradesHistory.
        Multi-fill executions sharing the same ordertxid/orderId are aggregated into
        a single order record (VWAP price, summed qty, max timestamp), symmetrically
        with BinanceProvider.get_orders().
        """
        try:
            if since_s is None or isinstance(since_s, bool):
                return []
            try:
                since_val = float(since_s)
            except (TypeError, ValueError, OverflowError):
                return []
            if not math.isfinite(since_val) or since_val < 0:
                return []
            rows = self._fills_from_cache(symbol)
            if rows is None:                                  # Missing or stale cache: direct API.
                rows = self._fills_from_api(symbol)
            if rows is None:
                raise ProviderError(f"get_orders({symbol}): fills history unavailable")
            cutoff_ms = (time.time() - since_val) * 1000.0
            from .base import aggregate_fills_by_order_id
            return aggregate_fills_by_order_id(rows, cutoff_ms=cutoff_ms, side=side)
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
            raw_px = description.get("price", raw.get("price", 0.0))
            raw_vol = raw.get("vol")
            raw_vol_exec = raw.get("vol_exec")
            if any(isinstance(v, bool) for v in (raw_px, raw_vol, raw_vol_exec)):
                raise ProviderError(
                    f"open_orders({symbol}): invalid values for {order_id}")
            try:
                price = float(raw_px or 0.0)
                original_qty = float(raw_vol or 0.0)
                executed_qty = float(raw_vol_exec or 0.0)
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
        if not symbol or isinstance(symbol, bool) or not str(symbol).strip():
            return None
        data = read_recovering_previous(_KRAKEN_CACHE_FILE, json.loads, default=None)
        if not isinstance(data, dict):
            return None
        items = data.get("items")
        ft = data.get("fetchtime")
        if not isinstance(items, dict) or not isinstance(ft, dict):
            return None
        su = str(symbol).strip().upper()
        key = next((k for k in items if _kraken_pair_matches(symbol, k)), None)
        if key is None:
            key = next(
                (
                    k
                    for k in items
                    if isinstance(k, str) and k.strip() and (su in k.upper() or k.strip().upper() in su)
                ),
                None,
            )
        if key is None or not isinstance(items.get(key), list):
            return None
        raw_ft = ft.get(key)
        if raw_ft is None or isinstance(raw_ft, bool):
            return None
        try:
            ft_val = float(raw_ft)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(ft_val) or ft_val <= 0.0:
            return None
        if ft_val < 1e11:
            ft_val *= 1000.0
        age_ms = time.time() * 1000.0 - ft_val
        if age_ms < -5000.0 or age_ms > _CACHE_MAX_STALE_S * 1000.0:
            return None                                       # Stale or clock-skewed cachemanager data.
        return [
            _normalize_kraken_fill(t)
            for t in items[key]
            if isinstance(t, dict)
        ]

    def _fills_from_api(self, symbol: str):
        """Fall back to direct TradesHistory, matching pre-cachemanager behavior."""
        cli = self._client()
        res = cli._private("TradesHistory")
        if not isinstance(res, dict) or "trades" not in res:
            raise ProviderError(f"TradesHistory({symbol}): trades history payload missing or invalid")
        trades = res.get("trades")
        if not isinstance(trades, dict):
            raise ProviderError(f"TradesHistory({symbol}): trades mapping in payload is invalid")
        su = symbol.upper()
        rows = []
        for tr in trades.values():
            if not isinstance(tr, dict):
                continue
            p = str(tr.get("pair", "")).upper()
            if not (_kraken_pair_matches(symbol, p) or su in p or (p and p in su)):
                continue
            rows.append(_normalize_kraken_fill(tr))
        return rows

    # -- Placement remains dry until KRAKEN_LIVE_ORDERS=true. ------------------
    def place_order(self, symbol: str, side: str, price: float, qty: float, **kwargs):
        live = self.execution_enabled()
        s_norm = str(side or "").strip().upper() if not isinstance(side, bool) else ""
        if s_norm in {"BUY", "B"}:
            s = "buy"
        elif s_norm in {"SELL", "S"}:
            s = "sell"
        else:
            print(f"[Kraken] place_order {symbol}: refused invalid side={side!r}")
            return None
        market = bool(kwargs.get("market", False) or kwargs.get("force", False))
        ordertype = "market" if market else "limit"
        submit_price = None if market else price
        if not market:
            if submit_price is None or isinstance(submit_price, bool):
                print(f"[Kraken] place_order {symbol}: refused invalid price={submit_price!r}")
                return None
            try:
                px_f = float(submit_price)
            except (TypeError, ValueError, OverflowError):
                print(f"[Kraken] place_order {symbol}: refused invalid price={submit_price!r}")
                return None
            if not math.isfinite(px_f) or px_f <= 0:
                print(f"[Kraken] place_order {symbol}: refused non-positive price={submit_price!r}")
                return None
            submit_price = px_f
        client_order_id = kwargs.get("client_order_id")
        try:
            cl_ord_id = (
                _kraken_client_order_id(client_order_id)
                if client_order_id is not None else None)
        except ProviderError as exc:
            print(f"[Kraken] place_order {symbol}: {exc}")
            return None
        qty = self.round_quantity(symbol, qty)
        if qty <= 0:
            print(f"[Kraken] place_order {symbol}: refused non-positive qty={qty}")
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
    @staticmethod
    def _is_definitive_refusal(error: object) -> bool:
        text = str(error or "").lower()
        return any(
            token in text
            for token in (
                "insufficient funds",
                "insufficient margin",
                "order minimum not met",
                "volume minimum not met",
                "cost minimum not met",
                "invalid price",
                "invalid arguments",
                "positions limit exceeded",
                "tick size check failed",
            )
        )

    def submit_order(self, symbol: str, side: str, qty: float,
                     price: Optional[float] = None, *, market: bool = False,
                     kind: Optional[str] = None,
                     client_order_id: Optional[str] = None) -> str:
        s_norm = str(side or "").strip().upper() if not isinstance(side, bool) else ""
        if s_norm in {"BUY", "B"}:
            s = "buy"
        elif s_norm in {"SELL", "S"}:
            s = "sell"
        else:
            raise SubmissionRefused(f"submit_order {symbol}: invalid side={side!r}")
        ordertype = "market" if (market or price is None) else "limit"
        px_f = None
        if ordertype == "limit":
            if isinstance(price, bool):
                raise ProviderError(f"submit_order {symbol} {s}: invalid price={price!r}")
            try:
                px_f = float(price)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ProviderError(f"submit_order {symbol} {s}: invalid price={price!r}") from exc
            if not math.isfinite(px_f) or px_f <= 0:
                raise ProviderError(f"submit_order {symbol} {s}: non-positive price={price!r}")
        qty = self.round_quantity(symbol, qty)
        if qty <= 0:
            raise ProviderError(f"submit_order {symbol} {s}: non-positive qty={qty}")
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
                symbol, s, qty, None if ordertype == "market" else px_f,
                **order_kwargs) or {}
        except SubmissionRefused:
            raise
        except Exception as e:  # noqa: BLE001 — Normalize venue errors.
            if self._is_definitive_refusal(e):
                raise SubmissionRefused(f"submit_order {symbol} {s}: {e}") from e
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
        if not isinstance(side, str) or not side.strip().lower().startswith("s"):
            return
        if qty is None:
            return
        if isinstance(qty, bool):
            raise ProviderError(
                f"preflight_order {symbol}: an invalid SELL quantity")
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
        if available is None or isinstance(available, bool) or not math.isfinite(float(available)):
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
        if (
            isinstance(order_id, bool)
            or not isinstance(order_id, str)
            or not order_id.strip()
            or order_id.strip().lower() in {"nan", "inf", "-inf", "+inf", "none", "null"}
        ):
            raise ProviderError(f"order_status {order_id}: invalid order id")
        try:
            res = self._client().query_orders(order_id) or {}
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"order_status {order_id}: {e}") from e
        if not isinstance(res, dict):
            raise ProviderError(f"order_status {order_id}: invalid Kraken response")
        info = res.get(order_id)
        if not isinstance(info, dict):
            raise ProviderError(f"order_status: the order {order_id} was not found")
        raw_status = str(info.get("status") or "").strip()
        status_map = {
            "pending": "open",
            "open": "open",
            "closed": "closed",
            "canceled": "canceled",
            "cancelled": "canceled",
            "expired": "expired",
        }
        norm_status = status_map.get(raw_status.lower())
        if norm_status is None:
            raise ProviderError(
                f"order_status {order_id}: unknown Kraken status {raw_status!r}"
            )
        raw_vol_exec = info.get("vol_exec")
        raw_cost = info.get("cost")
        raw_fee = info.get("fee")
        raw_vol = info.get("vol")
        if any(isinstance(v, bool) for v in (raw_vol_exec, raw_cost, raw_fee, raw_vol)):
            raise ProviderError(f"order_status {order_id}: invalid fill values")
        try:
            filled_qty = float(raw_vol_exec or 0.0)
            cost = float(raw_cost or 0.0)
            fee = float(raw_fee or 0.0)
            orig_qty = float(raw_vol or 0.0)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProviderError(f"order_status {order_id}: invalid fill values") from exc
        if (not all(math.isfinite(v) for v in (filled_qty, cost, fee, orig_qty))
                or filled_qty < 0.0 or cost < 0.0 or fee < 0.0 or orig_qty < 0.0):
            raise ProviderError(f"order_status {order_id}: invalid fill values")
        return OrderStatus(
            status=norm_status,
            filled_qty=filled_qty,
            cost=cost,
            fee=fee,
            venue_status=raw_status,
            orig_qty=orig_qty,
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
            if not isinstance(orders, dict):
                continue
            for order_id, raw in orders.items():
                if not isinstance(raw, dict):
                    continue
                if str(raw.get("cl_ord_id") or "").lower() != wanted:
                    continue
                descr = raw.get("descr")
                if not isinstance(descr, dict):
                    descr = {}
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
        if (
            isinstance(order_id, bool)
            or not isinstance(order_id, str)
            or not order_id.strip()
            or order_id.strip().lower() in {"nan", "inf", "-inf", "+inf", "none", "null"}
        ):
            raise ProviderError(f"cancel_order {order_id}: invalid order id")
        try:
            result = self._client().cancel_order(order_id) or {}
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            if "unknown order" in msg or "already" in msg:
                return                       # Idempotent: already closed or canceled.
            raise ProviderError(f"cancel_order {order_id}: {e}") from e
        if not isinstance(result, dict) or "count" not in result:
            raise ProviderError(
                f"cancel_order {order_id}: missing count in Kraken response ({result})"
            )
        try:
            if isinstance(result["count"], bool) or int(result["count"]) < 1:
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
            if any(isinstance(info[key], bool) for key in ("pair_decimals", "lot_decimals", "ordermin", "base")):
                raise ValueError("boolean values not allowed in pair_precision")
            p_f = float(info["pair_decimals"])
            v_f = float(info["lot_decimals"])
            if not (math.isfinite(p_f) and math.isfinite(v_f)) or int(p_f) != p_f or int(v_f) != v_f:
                raise ValueError("non-integer decimals in pair_precision")
            p_dec = int(p_f)
            v_dec = int(v_f)
            ord_min = float(info["ordermin"])
            if p_dec < 0 or v_dec < 0 or not math.isfinite(ord_min) or ord_min <= 0.0:
                raise ValueError("non-finite or non-positive values in pair_precision")
            base_str = str(info["base"]).strip()
            if not base_str:
                raise ValueError("empty base asset in pair_precision")
            return PairPrecision(
                price_decimals=p_dec,
                volume_decimals=v_dec,
                order_min=ord_min,
                base_asset=base_str,
            )
        except (TypeError, ValueError, OverflowError) as e:
            raise ProviderError(f"pair_precision {symbol}: info malformat ({e})") from e

    def ohlc_series(self, symbol: str, interval_min: int) -> ClosedPriceSeries:
        if isinstance(interval_min, bool) or interval_min is None:
            raise ProviderError(f"ohlc_series {symbol}: invalid interval_min={interval_min!r}")
        try:
            itv_f = float(interval_min)
            if not math.isfinite(itv_f) or not itv_f.is_integer():
                raise ProviderError(f"ohlc_series {symbol}: invalid interval_min={interval_min!r}")
            interval_i = int(itv_f)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProviderError(f"ohlc_series {symbol}: invalid interval_min={interval_min!r}") from exc
        if interval_i <= 0:
            raise ProviderError(f"ohlc_series {symbol}: invalid interval_min={interval_min!r}")
        try:
            client = self._client()
            series_reader = getattr(
                client, "ohlc_closes_with_timestamps", None)
            if callable(series_reader):
                closes, timestamps = series_reader(symbol, interval_i)
                timestamps = tuple(timestamps or ())
                observed_at = timestamps[-1] if timestamps else None
            else:
                reader = getattr(client, "ohlc_closes_with_timestamp", None)
                if callable(reader):
                    closes, observed_at = reader(symbol, interval_i)
                else:
                    closes = client.ohlc_closes(symbol, interval_i)
                    observed_at = None
                timestamps = ()
            return ClosedPriceSeries(
                tuple(closes or ()),
                interval_i,
                observed_at,
                timestamps,
            )
        except Exception as exc:
            raise ProviderError(f"ohlc_series {symbol}: {exc}") from exc
