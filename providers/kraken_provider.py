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

from botcore import (
    _parse_bool_flag,
    safe_float,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
    safe_pos_int,
)
from .base import (
    MarketDataProvider,
    _clean_single_order_id,
    _derive_128bit_client_order_id,
    _normalize_order,
    bool_env,
    env_value,
    extract_order_side,
    extract_timestamp_ms,
    normalize_side,
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


_KRAKEN_QUOTES = (
    "USDC", "USDT", "USD", "EUR", "GBP", "CAD", "JPY", "AUD",
    "BTC", "XBT", "ETH",
)
_LEGACY_KRAKEN_X_BASES = frozenset({
    "XXBT", "XETH", "XXDG", "XXRP", "XLTC", "XXLM", "XZEC", "XXMR", "XETC", "XREP",
})


def _split_kraken_pair(value: str) -> tuple[str, str]:
    """Split a Kraken pair or bare asset into ``(canonical_base, canonical_quote)``."""
    if not isinstance(value, str):
        return "", ""
    raw = value.strip()
    key = "".join(ch for ch in raw.upper() if ch.isalnum())
    if not key:
        return "", ""
    quote = next(
        (cand for cand in _KRAKEN_QUOTES if key.endswith(cand) and len(key) > len(cand)),
        "",
    )
    if not quote:
        base_raw = raw
    elif "/" in raw:
        base_raw = raw.split("/", 1)[0].strip()
    elif raw.upper().endswith(quote):
        base_raw = raw[: -len(quote)].strip(" -_/")
    else:
        base_raw = key[: -len(quote)]
    from cachepaths import canonical_base_asset
    base = canonical_base_asset(base_raw) or "".join(ch for ch in base_raw.upper() if ch.isalnum())
    if quote == "XBT":
        quote = "BTC"
    return base, quote


def _kraken_pair_key(value: str) -> str:
    """Canonicalize common Kraken legacy pair wrappers for strict comparison."""
    base, quote = _split_kraken_pair(value)
    return base + quote


def _kraken_pair_matches(expected: str, actual: str) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return False
    b_exp, q_exp = _split_kraken_pair(expected)
    b_act, q_act = _split_kraken_pair(actual)
    if not b_exp or not b_act or b_exp != b_act:
        return False
    if q_exp and q_act:
        return q_exp == q_act
    return True


def _kraken_balance_asset_candidates(asset: str) -> List[str]:
    """Return ordered Kraken Balance/BalanceEx lookup keys for an asset or xStock ticker."""
    raw = str(asset or "").strip() if not isinstance(asset, bool) else ""
    if not raw:
        return []
    asset_u = raw.upper()
    candidates: List[str] = [
        raw, asset_u, "X" + asset_u, "Z" + asset_u,
        asset_u + ".F", asset_u + ".S", f"{asset_u}x.T", f"{asset_u}X.T", f"{asset_u}x",
    ]
    if asset_u in {"BTC", "XBT", "XXBT"}:
        candidates.extend(["BTC", "XBT", "XXBT", "XBT.F", "XXBT.F"])
    elif asset_u in {"DOGE", "XDG", "XXDG"}:
        candidates.extend(["DOGE", "XDG", "XXDG", "XDG.F", "XXDG.F"])
    elif len(asset_u) == 4 and (asset_u in _LEGACY_KRAKEN_X_BASES or asset_u.startswith("Z")):
        stripped = asset_u[1:]
        candidates.extend([stripped, f"{stripped}.F", f"{stripped}.S"])
    from cachepaths import canonical_base_asset
    canon = canonical_base_asset(raw)
    if canon and canon != asset_u:
        candidates.extend([canon, f"{canon}.F", f"{canon}.S", f"{canon}x.T", f"{canon}X.T", f"{canon}x"])
    return list(dict.fromkeys(candidates))


def _normalize_kraken_fill(raw: dict) -> dict:
    """Normalize a Kraken fill while preserving ordertxid/orderId for order-level aggregation."""
    norm = _normalize_order(raw)
    if isinstance(raw, dict):
        order_id = _clean_single_order_id(raw.get("ordertxid")) or _clean_single_order_id(raw.get("orderId"))
        if order_id is not None:
            norm["orderId"] = order_id
        else:
            norm.pop("orderId", None)
    return norm


_KRAKEN_CLIENT_ID_RE = re.compile(r"^[0-9a-fA-F]{32}$")


def _kraken_client_order_id(client_order_id: str) -> str:
    """Map one durable client ID to Kraken's 128-bit hexadecimal UUID form."""
    return _derive_128bit_client_order_id(
        client_order_id,
        venue_label="Kraken",
        domain_prefix=b"assertguardian:kraken:cloid:v1\0",
        exact_re=_KRAKEN_CLIENT_ID_RE,
    )


_INVALID_KRAKEN_ID_STRINGS = {"", "0", "0.0", "-0", "+0", "nan", "inf", "-inf", "+inf", "none", "null"}


def _clean_kraken_order_id(order_id: object, caller: str) -> str:
    if order_id is None or isinstance(order_id, (bool, dict, list, tuple, set, frozenset)) or not isinstance(order_id, str):
        raise ProviderError(f"{caller} {order_id}: invalid order id")
    s = order_id.strip()
    if not s or s.startswith("-") or s.lower() in _INVALID_KRAKEN_ID_STRINGS or s in {"+0.0", "-0.0"}:
        raise ProviderError(f"{caller} {order_id}: invalid order id")
    return s


def _live() -> bool:
    # os.environ takes precedence, with kraken/.env as the API-key-style fallback.
    return bool_env("KRAKEN_LIVE_ORDERS", _KRAKEN_DIR)


def kraken_strategy_dry_run(force_paper: bool, strategy_execute: bool) -> bool:
    """Combine the strategy and provider gates for Kraken orchestration."""
    return bool(
        _parse_bool_flag(force_paper, True)
        or not _parse_bool_flag(strategy_execute, False)
        or not _live()
    )


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
        norm_sym = str(symbol or "").strip().upper() if not isinstance(symbol, bool) else ""
        if not norm_sym:
            return 0.0
        cached = self._minqty.get(norm_sym) or self._minqty.get(symbol)
        if cached:  # Only positive values are cached; zero or absent retries.
            return cached
        mn = 0.0
        try:
            info = self._client().pair_info(norm_sym) or {}
            mn = safe_pos_float(info.get("ordermin", 0), default=0.0)
        except Exception as e:  # noqa: BLE001
            print(f"[Kraken] ordermin {symbol}: {e}")
        if mn > 0:
            self._minqty[norm_sym] = mn   # Cache only a successful lookup.
        return mn

    def get_price_history(self, symbol: str, lookback_h: float) -> Optional[List]:
        """Return ascending public OHLC closes as timestamp/price mappings."""
        if not symbol or isinstance(symbol, bool) or not str(symbol).strip():
            return None
        lookback_f = safe_pos_float(lookback_h, default=None)
        if lookback_f is None:
            return None
        try:
            cli = self._client()
            # Choose a valid Kraken OHLC minute interval that fits roughly 720 points or fewer.
            raw_interval = max(1, int(math.ceil((lookback_f * 60.0) / 720.0)))
            valid_intervals = (1, 5, 15, 30, 60, 240, 1440, 10080, 21600)
            interval = next((v for v in valid_intervals if v >= raw_interval), valid_intervals[-1])
            res = cli._public("OHLC", {"pair": str(symbol).strip(), "interval": interval})
            if not isinstance(res, dict):
                return None
            rows = next((v for k, v in res.items() if k != "last"), None)
            if not rows or not isinstance(rows, (list, tuple)):
                return None
            cutoff = time.time() - lookback_f * 3600
            out = []
            for r in rows:                       # [time, o, h, l, c, vwap, vol, cnt]
                if not isinstance(r, (list, tuple)) or len(r) < 5:
                    continue
                t = safe_pos_float(r[0], default=None)
                p = safe_pos_float(r[4], default=None)
                if t is None or p is None or t < cutoff:
                    continue
                out.append({"timestamp": int(t * 1000), "price": p})
            out.sort(key=lambda x: x["timestamp"])
            return out or None
        except Exception as e:  # noqa: BLE001
            print(f"[Kraken] history {symbol}: {e}")
            return None

    # -- Credentialed account access. -----------------------------------------
    def free_balance(self, asset: str) -> Optional[float]:
        if not asset or isinstance(asset, bool) or not str(asset).strip():
            return None
        try:
            bal = self._client().balance() or {}
            candidates = _kraken_balance_asset_candidates(asset)
            for key in candidates:
                if key in bal:
                    return safe_nonneg_float(bal[key], default=None)
            cands_up = {c.upper() for c in candidates}
            for k_raw, raw_val in bal.items():
                if str(k_raw).strip().upper() in cands_up:
                    return safe_nonneg_float(raw_val, default=None)
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
            since_val = safe_nonneg_float(since_s, default=None)
            if since_val is None:
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
            clean_oid = _clean_kraken_order_id(order_id, f"open_orders({symbol})")
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
            side = normalize_side(description.get("type") or raw.get("type"))
            if side not in {"BUY", "SELL"}:
                raise ProviderError(
                    f"open_orders({symbol}): an invalid side for {order_id}")
            price = safe_nonneg_float(description.get("price", raw.get("price", 0.0)), default=None)
            original_qty = safe_pos_float(raw.get("vol"), default=None)
            executed_qty = safe_nonneg_float(raw.get("vol_exec", 0.0), default=None)
            if price is None or original_qty is None or executed_qty is None or executed_qty > original_qty:
                raise ProviderError(
                    f"open_orders({symbol}): invalid quantities/price for {order_id}")
            from .base import _clean_single_order_id
            normalized.append({
                "orderId": clean_oid,
                "clientOrderId": _clean_single_order_id(raw.get("cl_ord_id")),
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
        key = next((k for k in items if _kraken_pair_matches(symbol, k)), None)
        if key is None or not isinstance(items.get(key), list):
            return None
        ft_val = safe_pos_float(ft.get(key), default=None)
        if ft_val is None:
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
        if not symbol or isinstance(symbol, bool) or not str(symbol).strip():
            return []
        cli = self._client()
        res = cli._private("TradesHistory")
        if not isinstance(res, dict) or "trades" not in res:
            raise ProviderError(f"TradesHistory({symbol}): trades history payload missing or invalid")
        trades = res.get("trades")
        if not isinstance(trades, dict):
            raise ProviderError(f"TradesHistory({symbol}): trades mapping in payload is invalid")
        rows = []
        for tr in trades.values():
            if not isinstance(tr, dict):
                continue
            p = str(tr.get("pair", "")).upper()
            if not _kraken_pair_matches(symbol, p):
                continue
            rows.append(_normalize_kraken_fill(tr))
        return rows

    # -- Placement remains dry until KRAKEN_LIVE_ORDERS=true. ------------------
    def place_order(self, symbol: str, side: str, price: float, qty: float,
                    force: bool = False, **kwargs):
        live = self.execution_enabled()
        s_norm = normalize_side(side)
        if s_norm not in {"BUY", "SELL"}:
            print(f"[Kraken] place_order {symbol}: refused invalid side={side!r}")
            return None
        s = s_norm.lower()
        market = (
            _parse_bool_flag(force, False)
            or _parse_bool_flag(kwargs.get("market", False), False)
            or _parse_bool_flag(kwargs.get("force", False), False)
        )
        ordertype = "market" if market else "limit"
        submit_price = None if market else price
        if not market:
            px_f = safe_pos_float(submit_price, default=None)
            if px_f is None:
                print(f"[Kraken] place_order {symbol}: refused invalid/non-positive price={submit_price!r}")
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
        s_norm = normalize_side(side)
        if s_norm not in {"BUY", "SELL"}:
            raise SubmissionRefused(f"submit_order {symbol}: invalid side={side!r}")
        s = s_norm.lower()
        is_market = _parse_bool_flag(market, False)
        ordertype = "market" if (is_market or price is None) else "limit"
        px_f = None
        if ordertype == "limit":
            px_f = safe_pos_float(price, default=None)
            if px_f is None:
                raise ProviderError(f"submit_order {symbol} {s}: invalid/non-positive price={price!r}")
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
        if normalize_side(side) != "SELL":
            return
        if qty is None:
            return
        requested = safe_pos_float(qty, default=None)
        if requested is None:
            raise ProviderError(
                f"preflight_order {symbol}: an invalid SELL quantity")

        precision = self.pair_precision(symbol)
        base_asset = precision.base_asset if precision else ""
        if not base_asset:
            raise ProviderError(
                f"preflight_order {symbol}: base asset is unavailable")
        avail_f = safe_nonneg_float(self.free_balance(base_asset), default=None)
        if avail_f is None:
            raise ProviderError(
                f"preflight_order {symbol}: {base_asset} balance is unavailable")
        available = avail_f
        decimals = precision.volume_decimals if precision else 8
        tolerance = 0.5 * (10.0 ** -max(0, decimals))
        if requested > available + tolerance:
            raise ProviderError(
                f"preflight_order {symbol}: insufficient funds SELL "
                f"qty={requested} available={available} {base_asset}"
            )

    def order_status(self, symbol: str, order_id: str) -> OrderStatus:
        clean_id = _clean_kraken_order_id(order_id, "order_status")
        try:
            res = self._client().query_orders(clean_id) or {}
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"order_status {order_id}: {e}") from e
        if not isinstance(res, dict):
            raise ProviderError(f"order_status {order_id}: invalid Kraken response")
        info = res.get(clean_id) if clean_id in res else res.get(order_id)
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
        filled_qty = safe_nonneg_float(info.get("vol_exec", 0.0), default=None)
        cost = safe_nonneg_float(info.get("cost", 0.0), default=None)
        fee = safe_nonneg_float(info.get("fee", 0.0), default=None)
        orig_qty = safe_nonneg_float(info.get("vol", 0.0), default=None)
        if None in (filled_qty, cost, fee, orig_qty):
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
                clean_oid = _clean_kraken_order_id(order_id, f"order_by_client_id({symbol})")
                return {"orderId": clean_oid, "status": raw.get("status")}
        return None

    def cancel_order_by_id(self, symbol: str, order_id: str) -> None:
        """Cancel by id, treating an already closed or missing order as success.

        The explicit method name avoids future ambiguity; cancel_order below is the
        StrategyExecutor contract alias.
        """
        clean_id = _clean_kraken_order_id(order_id, "cancel_order")
        try:
            result = self._client().cancel_order(clean_id) or {}
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            if "unknown order" in msg or "already" in msg:
                return                       # Idempotent: already closed or canceled.
            raise ProviderError(f"cancel_order {order_id}: {e}") from e
        if not isinstance(result, dict) or "count" not in result:
            raise ProviderError(
                f"cancel_order {order_id}: missing count in Kraken response ({result})"
            )
        cnt_f = safe_pos_float(result.get("count"), default=None)
        if cnt_f is None or not cnt_f.is_integer() or int(cnt_f) < 1:
            raise ProviderError(
                f"cancel_order {order_id}: Kraken did not confirm the cancellation ({result})"
            )

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
            if isinstance(info["base"], bool):
                raise ValueError("boolean values not allowed in pair_precision")
            p_f = safe_nonneg_float(info["pair_decimals"], default=None)
            v_f = safe_nonneg_float(info["lot_decimals"], default=None)
            ord_min = safe_pos_float(info["ordermin"], default=None)
            if p_f is None or v_f is None or ord_min is None or not p_f.is_integer() or not v_f.is_integer():
                raise ValueError("invalid decimals or ordermin in pair_precision")
            base_str = str(info["base"]).strip()
            if not base_str:
                raise ValueError("empty base asset in pair_precision")
            return PairPrecision(
                price_decimals=int(p_f),
                volume_decimals=int(v_f),
                order_min=ord_min,
                base_asset=base_str,
            )
        except (TypeError, ValueError, OverflowError) as e:
            raise ProviderError(f"pair_precision {symbol}: info malformat ({e})") from e

    def ohlc_series(self, symbol: str, interval_min: int) -> ClosedPriceSeries:
        itv_f = safe_pos_float(interval_min, default=None)
        if itv_f is None or not itv_f.is_integer():
            raise ProviderError(f"ohlc_series {symbol}: invalid interval_min={interval_min!r}")
        interval_i = int(itv_f)
        try:
            client = self._client()
            series_reader = getattr(
                client, "ohlc_closes_with_timestamps", None)
            if callable(series_reader):
                closes, timestamps = series_reader(symbol, interval_i)
                timestamps = tuple(timestamps) if timestamps is not None else ()
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
                tuple(closes) if closes is not None else (),
                interval_i,
                observed_at,
                timestamps,
            )
        except Exception as exc:
            raise ProviderError(f"ohlc_series {symbol}: {exc}") from exc
