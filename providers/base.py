"""Provider primitives with no registry or venue-module dependencies."""

import contextlib
import importlib
import inspect
import math
import os
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from botcore import (
    _parse_bool_flag,
    bool_env,
    clamp_price_to_bbo_mid,
    exact_nonneg_int,
    is_shared_disk_cache_enabled,
    resolve_patched_attr,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
    safe_pos_int,
)
from cachepaths import (
    CACHE_CURRENT_PRICE_FILENAME,
    cache_path,
    canonical_symbol_key,
    clean_symbol_key,
    normalize_venue_name,
    split_symbol_assets,
)


_LIVE_FETCH_LOCAL = threading.local()


@contextlib.contextmanager
def live_price_fetch_context():
    """Mark the current thread as performing a live venue price poll for cacheManager."""
    depth = int(getattr(_LIVE_FETCH_LOCAL, "depth", 0) or 0)
    _LIVE_FETCH_LOCAL.depth = depth + 1
    try:
        yield
    finally:
        _LIVE_FETCH_LOCAL.depth = max(0, depth)


def is_in_live_price_fetch() -> bool:
    """Return True when called inside a cacheManager remote price poll."""
    return int(getattr(_LIVE_FETCH_LOCAL, "depth", 0) or 0) > 0


from .order_ohlc_utils import (  # noqa: E402, F401
    _is_fresh_obs_ts,
    _ts_ms_to_s,
    aggregate_fills_by_order_id,
    extract_order_id,
    extract_order_price,
    extract_timestamp_ms,
    extract_timestamp_s,
    normalize_side,
    normalize_timestamp_ms,
)
from .quantity import (  # noqa: E402, F401
    compute_fills_cost_basis,
    decide_quantity,
    venue_fee_cap_quantity,
)
from .strategy_executor import (  # noqa: E402, F401
    OrderReconciliationCapabilities,
    OrderStatus,
    PairPrecision,
    ProviderError,
    SubmissionRefused,
)


class MarketDataProvider(ABC):
    """Shared market-data, account, order, and policy-hook interface."""

    def _fetch_current_price(self, symbol: str) -> Optional[float]:
        """Venue-specific live quote hook invoked on SSOT cache miss."""
        return None

    def _fetch_price_history(self, symbol: str, lookback_h: float) -> Optional[List]:
        """Venue-specific historical price series hook invoked on SSOT cache miss."""
        return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Return fresh current price for ``symbol`` (SSOT cache first when enabled, then venue transport)."""
        self.validate_symbol(symbol)
        if not symbol or isinstance(symbol, bool) or not str(symbol).strip():
            return None
        cached = self._cached_current_price(symbol)
        if cached is not None:
            return cached
        return safe_pos_float(self._fetch_current_price(symbol), default=None)

    def get_price_history(self, symbol: str, lookback_h: float) -> Optional[List]:
        """Return ascending historical prices over ``lookback_h`` hours (SSOT 24h cache first, then venue transport)."""
        self.validate_symbol(symbol)
        if not symbol or isinstance(symbol, bool) or not str(symbol).strip():
            return None
        lookback_f = safe_nonneg_float(lookback_h, default=None)
        if lookback_f is None:
            return None
        cached_hist = self._cached_price_history(symbol, lookback_f)
        if cached_hist is not None:
            return cached_hist
        return self._fetch_price_history(symbol, lookback_f)

    @abstractmethod
    def supports_symbol(self, symbol: str) -> bool:
        ...

    _DEFAULT_FILLS_CACHE_FILENAME: Optional[str] = None
    _FILLS_CACHE_PATH_MODULE_ATTR: Optional[str] = None
    _FILLS_CACHE_DEFAULT_MODULE_ATTR: Optional[str] = None
    _FILLS_CACHE_MAX_STALE_S: float = 60.0
    _CLIENT_CLASS_NAME: Optional[str] = None
    _DEFAULT_QUOTE_ASSET: Optional[str] = None
    _USES_SSOT_MARKET_CACHE: bool = True
    _REQUIRES_CACHE_PERMIT: bool = False
    _DEFERS_SMART_CANCELLATION: bool = False
    _SUPPORTS_BUSINESS_MINIMUM: bool = False
    _VENUE_PROVIDER_SPECS: dict[str, tuple[str, str]] = {
        "binance": ("providers.binance_provider", "BinanceProvider"),
        "hyperliquid": ("providers.hyperliquid_provider", "HyperliquidProvider"),
        "kraken": ("providers.kraken_provider", "KrakenProvider"),
        "t212": ("providers.t212_provider", "T212Provider"),
    }
    _fills_match_fn: Any = None
    _fills_normalize_fn: Any = None
    _ssot_current_price_reader: Any = None
    _policy_weight_limit_fn: Any = None
    _policy_window_cfg_fn: Any = None
    _policy_window_reference_fn: Any = None

    @classmethod
    def bind_ssot_cache_readers(
        cls,
        *,
        current_price_reader: Any = None,
    ) -> None:
        """Bind read-only SSOT memory cache getter for current price."""
        if current_price_reader is not None:
            cls._ssot_current_price_reader = staticmethod(current_price_reader)

    @classmethod
    def bind_order_guard_hooks(
        cls,
        *,
        weight_limit_fn: Any = None,
        window_cfg_fn: Any = None,
        window_reference_fn: Any = None,
    ) -> None:
        """Bind order-guard policy callbacks from the higher Order Gateway layer."""
        if weight_limit_fn is not None:
            cls._policy_weight_limit_fn = staticmethod(weight_limit_fn)
        if window_cfg_fn is not None:
            cls._policy_window_cfg_fn = staticmethod(window_cfg_fn)
        if window_reference_fn is not None:
            cls._policy_window_reference_fn = staticmethod(window_reference_fn)

    @staticmethod
    def _invoke_fee_and_limits_callable(
        fn: Any, *leading_args: Any, symbol: str, price: Optional[float] = None
    ) -> Optional[dict]:
        if not callable(fn):
            return None
        try:
            params = inspect.signature(fn).parameters
            has_price = "price" in params or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
        except (TypeError, ValueError):
            has_price = False
        res = fn(*leading_args, symbol, price=price) if has_price else fn(*leading_args, symbol)
        return res if isinstance(res, dict) else None

    def resolve_cached_fee_and_limits(self, symbol: str, *, price: Optional[float] = None) -> dict:
        """Return non-blocking cached fee tier and precision/limit overrides for ``symbol``."""
        extract_fn = getattr(type(self), "extract_cached_fee_and_limits", None)
        if not callable(extract_fn):
            prov_cls = self.resolve_venue_provider_class(getattr(self, "name", ""))
            extract_fn = getattr(prov_cls, "extract_cached_fee_and_limits", None) if prov_cls is not None else None
        res = self._invoke_fee_and_limits_callable(extract_fn, self, symbol=symbol, price=price)
        return res if res is not None else {}

    @classmethod
    def resolve_venue_provider_class(cls, venue: Any) -> Any:
        """Resolve the concrete ``MarketDataProvider`` subclass for ``venue`` via declarative registry."""
        spec = cls._VENUE_PROVIDER_SPECS.get(normalize_venue_name(venue))
        if spec is None:
            return None
        mod_path, cls_name = spec
        return getattr(importlib.import_module(mod_path), cls_name, None)

    @classmethod
    def resolve_provider_cached_fee_and_limits(
        cls,
        venue: str,
        symbol: str,
        provider: Any,
        *,
        price: Optional[float] = None,
    ) -> dict:
        """Resolve cached fee/limit overrides from ``provider`` via public hook or venue adapter."""
        if provider is None:
            return {}
        inst_hook = getattr(provider, "resolve_cached_fee_and_limits", None)
        if callable(inst_hook) and getattr(type(provider), "resolve_cached_fee_and_limits", None) is not cls.resolve_cached_fee_and_limits:
            try:
                res = cls._invoke_fee_and_limits_callable(inst_hook, symbol=symbol, price=price)
                if res is not None:
                    return res
            except Exception:
                pass
        extract_fn = getattr(type(provider), "extract_cached_fee_and_limits", None)
        if not callable(extract_fn):
            prov_cls = cls.resolve_venue_provider_class(venue or getattr(provider, "name", ""))
            extract_fn = getattr(prov_cls, "extract_cached_fee_and_limits", None) if prov_cls is not None else None
        try:
            res = cls._invoke_fee_and_limits_callable(extract_fn, provider, symbol=symbol, price=price)
            if res is not None:
                return res
        except Exception:
            pass
        return {}

    @classmethod
    def fetch_venue_remote_trades(
        cls,
        venue: Any,
        symbol: str,
        start_time_ms: int = 0,
        *,
        api_client: Any = None,
        **kwargs,
    ) -> List[dict]:
        """Venue-agnostic remote trade fetcher for ``CacheTradeManager`` instances across all venues."""
        if api_client is None:
            return []
        if isinstance(api_client, cls) or "fetch_remote_trades" in getattr(api_client, "__dict__", {}):
            return list(api_client.fetch_remote_trades(symbol, start_time_ms) or [])
        prov_cls = cls.resolve_venue_provider_class(venue)
        if prov_cls is not None and callable(getattr(prov_cls, "fetch_remote_trades_with_client", None)):
            try:
                res = prov_cls.fetch_remote_trades_with_client(symbol, start_time_ms, api_client, **kwargs)
            except TypeError:
                res = prov_cls.fetch_remote_trades_with_client(symbol, start_time_ms, api_client)
            if res is not None:
                return list(res)
        fetch_fn = getattr(api_client, "fetch_remote_trades", None)
        if callable(fetch_fn):
            return list(fetch_fn(symbol, start_time_ms) or [])
        return []

    def validate_symbol(self, symbol: str) -> None:
        """Reject a symbol incompatible with this configured provider instance."""
        return None

    def _ensure_client_lock(self):
        lock = getattr(self, "_client_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._client_lock = lock
        return lock

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    def free_balance(self, asset: str) -> Optional[float]:
        """Return free balance: ``0.0`` is real zero; ``None`` means unavailable."""
        return None

    def _fills_cache_file(self) -> Optional[str]:
        custom = getattr(self, "_ssot_fills_cache_path", None)
        if custom:
            return str(custom)
        mod_name = type(self).__module__
        mod_path = (
            resolve_patched_attr(self._FILLS_CACHE_PATH_MODULE_ATTR, None, modules=(mod_name,))
            if self._FILLS_CACHE_PATH_MODULE_ATTR
            else None
        )
        def_path = (
            resolve_patched_attr(self._FILLS_CACHE_DEFAULT_MODULE_ATTR, None, modules=(mod_name,))
            if self._FILLS_CACHE_DEFAULT_MODULE_ATTR
            else None
        )
        if mod_path and def_path and mod_path != def_path:
            return str(mod_path)
        if self._allow_ssot_price_cache():
            if mod_path:
                return str(mod_path)
            if self._DEFAULT_FILLS_CACHE_FILENAME:
                return cache_path(self._DEFAULT_FILLS_CACHE_FILENAME)
        return None

    def _fills_from_cache(self, symbol: str) -> Optional[List[dict]]:
        """Read symbol fills from the shared SSOT trade cache, or return None for API fallback."""
        cache_file = self._fills_cache_file()
        if not cache_file:
            return None
        stale_s = resolve_patched_attr("_CACHE_MAX_STALE_S", None, modules=(type(self).__module__,))
        return self.read_ssot_fills_cache(
            symbol,
            cache_file,
            max_stale_sec=stale_s if stale_s is not None else self._FILLS_CACHE_MAX_STALE_S,
            match_fn=getattr(type(self), "_fills_match_fn", None),
            normalize_fn=getattr(type(self), "_fills_normalize_fn", None),
        )

    def _invalidate_fills_cache(self, symbol: str) -> None:
        """Expire ``fetchtime`` for ``symbol`` in this provider's configured SSOT trade cache."""
        cache_file = self._fills_cache_file()
        if cache_file:
            self.invalidate_ssot_fills_cache(
                symbol, cache_file, match_fn=getattr(type(self), "_fills_match_fn", None)
            )

    def _invoke_fills_from_api(
        self, symbol: str, *, publish_cache: bool = False
    ) -> Optional[List[dict]]:
        """Invoke ``self._fills_from_api`` without masking internal ``TypeError`` exceptions."""
        api_fn = getattr(self, "_fills_from_api", None)
        if not callable(api_fn):
            return None
        try:
            params = inspect.signature(api_fn).parameters
            accepts_publish = "publish_cache" in params or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
        except (TypeError, ValueError):
            accepts_publish = False
        if accepts_publish:
            return api_fn(symbol, publish_cache=publish_cache)
        rows = api_fn(symbol)
        if publish_cache and rows is not None:
            cache_file = self._fills_cache_file()
            if cache_file:
                self.write_ssot_fills_cache(symbol, cache_file, rows)
        return rows

    def fetch_remote_trades(self, symbol: str, start_time_ms: int = 0) -> List[dict]:
        """Fetch and normalize remote fills for ``symbol`` since ``start_time_ms`` without mutating disk caches."""
        self.validate_symbol(symbol)
        start_ms = 0.0 if start_time_ms is None else safe_nonneg_float(start_time_ms, default=None)
        if start_ms is None:
            return []
        rows = self._invoke_fills_from_api(symbol, publish_cache=False)
        if not isinstance(rows, list):
            return []
        return [
            dict(r)
            for r in rows
            if isinstance(r, dict)
            and (start_ms <= 0 or (extract_timestamp_ms(r, default=0) or 0) >= start_ms)
        ]

    def refresh_ssot_fills_cache(self, symbol: str, fills: Optional[List[dict]] = None) -> List[dict]:
        """Authoritatively fetch (if ``fills`` not provided) and explicitly publish ``symbol`` fills to the SSOT trade cache."""
        self.validate_symbol(symbol)
        if fills is None:
            rows = self._invoke_fills_from_api(symbol, publish_cache=True)
            return list(rows) if isinstance(rows, list) else []
        cache_file = self._fills_cache_file()
        if cache_file:
            self.write_ssot_fills_cache(symbol, cache_file, fills)
        return list(fills)

    def get_orders(
        self,
        symbol: str,
        side: Optional[str],
        since_s: float,
        *,
        authoritative: bool = False,
        publish_cache: bool = False,
    ) -> List[dict]:
        """Return normalized recent orders (fills aggregated by orderId), reading SSOT cache first unless ``authoritative=True``."""
        if not callable(getattr(self, "_fills_from_api", None)):
            return []
        self.validate_symbol(symbol)
        since_val = safe_nonneg_float(since_s, default=None)
        if since_val is None:
            return []
        try:
            cache_fn = getattr(self, "_fills_from_cache", None)
            rows = cache_fn(symbol) if (callable(cache_fn) and not authoritative and not publish_cache) else None
            if rows is None:
                rows = self._invoke_fills_from_api(symbol, publish_cache=publish_cache)
            if rows is None:
                raise ProviderError(f"get_orders({symbol}): fills history unavailable")
            return aggregate_fills_by_order_id(rows, cutoff_ms=(time.time() - since_val) * 1000.0, side=side)
        except ProviderError:
            raise
        except Exception as e:  # noqa: BLE001
            print(f"[{self.name}] get_orders({symbol},{side}) failed: {e}")
            raise ProviderError(f"get_orders({symbol}): {e}") from e

    def get_trades(
        self,
        symbol: str,
        since_s: float,
        *,
        authoritative: bool = False,
        publish_cache: bool = False,
    ) -> List[dict]:
        if not authoritative and not publish_cache:
            try:
                return self.get_orders(
                    symbol,
                    None,
                    since_s,
                    authoritative=False,
                    publish_cache=False,
                )
            except TypeError:
                return self.get_orders(symbol, None, since_s)
        return self.get_orders(
            symbol,
            None,
            since_s,
            authoritative=authoritative,
            publish_cache=publish_cache,
        )

    @staticmethod
    def resolve_venue_instrument_spec(venue: Any, symbol: str) -> Any:
        """Resolve ``InstrumentSpec`` for ``symbol`` on ``venue`` from ``instrument_registry``."""
        if isinstance(symbol, bool) or not isinstance(symbol, str) or not symbol.strip():
            return None
        sym_u = symbol.strip().upper()
        sym_clean = clean_symbol_key(sym_u)
        sym_canon = canonical_symbol_key(sym_u)
        try:
            import instrument_registry as _default_ir

            ir = resolve_patched_attr("instrument_registry", _default_ir, modules=("providers.market_api",)) or _default_ir
            specs = list(ir.select_instruments(normalize_venue_name(getattr(venue, "name", venue) or "")).values())
            for item in specs:
                item_sym = str(getattr(item, "symbol", "") or "").strip().upper()
                if item_sym == sym_u or (sym_clean and clean_symbol_key(item_sym) == sym_clean):
                    return item
            for item in specs:
                item_sym = str(getattr(item, "symbol", "") or "").strip().upper()
                if sym_canon and canonical_symbol_key(item_sym) == sym_canon:
                    return item
            return None
        except Exception:
            return None

    def position_cost_basis(self, symbol: str, held_qty: float, since_s: float) -> Optional[float]:
        """Return inventory-reconciled acquisition cost from fresh SSOT fills, or None when unknown."""
        since_val = safe_pos_float(since_s, default=None)
        if since_val is None:
            raise ValueError("since_s must be finite and positive")
        if isinstance(symbol, bool) or not isinstance(symbol, str) or not symbol.strip():
            return None
        fills_fn = getattr(self, "_fills_from_cache", None)
        rows = fills_fn(symbol) if callable(fills_fn) else None
        if rows is None:
            return None
        sym_u = symbol.strip().upper()
        spec = self.resolve_venue_instrument_spec(getattr(self, "name", ""), sym_u)
        base_asset = spec.base if spec is not None else ""
        quote_asset = spec.quote if spec is not None else ""
        if not base_asset and (parts := split_symbol_assets(sym_u)) is not None:
            base_asset, quote_asset = parts
        return compute_fills_cost_basis(
            rows, held_qty, since_val, base_asset=base_asset, quote_asset=quote_asset
        )

    def open_orders(self, symbol: str) -> List[dict]:
        return []

    def reconciliation_capabilities(self) -> OrderReconciliationCapabilities:
        """Return explicitly supported strict lifecycle operations.

        The default is deliberately conservative. An inherited empty
        ``open_orders`` implementation must never be mistaken for a confirmed
        empty venue snapshot.
        """
        return OrderReconciliationCapabilities()

    def preflight_order(self, symbol: str, side: str, qty: float,
                        price=None, *, market: bool = False,
                        kind: Optional[str] = None) -> Any:
        """Validate venue state and optionally return opaque submit authorization."""
        return None

    def _is_live_order_enabled(self, env_key: str, folder: Optional[str] = None) -> bool:
        """Resolve live-order gate from explicit ``self._live_enabled`` override or ``bool_env``."""
        live_override = getattr(self, "_live_enabled", None)
        if live_override is None:
            return bool_env(env_key, folder)
        return _parse_bool_flag(live_override, False)

    def execution_enabled(self) -> bool:
        """Return whether this provider may create a real venue order now.

        Providers with an explicit dry/live switch override this hook. The shared
        placement pipeline checks it before creating durable retry state so a dry
        validation cannot become a live order after a later configuration change.
        """
        return True

    @staticmethod
    def resolve_strategy_dry_run(
        force_paper: bool, strategy_execute: bool, venue_live_enabled: bool
    ) -> bool:
        """Combine CLI `--paper`, strategy gate (`STRAT_EXECUTE`), and venue live gate (`*_LIVE_ORDERS`)."""
        return bool(
            _parse_bool_flag(force_paper, True)
            or not _parse_bool_flag(strategy_execute, False)
            or not _parse_bool_flag(venue_live_enabled, False)
        )

    def _round_required_qty(self, symbol: str, qty: float, err_prefix: str = "submit_order") -> float:
        """Round ``qty`` using ``round_quantity`` and raise ``ProviderError`` if non-positive."""
        rounded = self.round_quantity(symbol, qty)
        if rounded <= 0:
            raise ProviderError(f"{err_prefix}({symbol}): invalid non-positive quantity qty={rounded}")
        return rounded

    @staticmethod
    def _execute_confirmed_cancel(order_id: Any, cancel_fn: Any) -> None:
        """Execute ``cancel_fn()`` and raise ``ProviderError`` unless cancellation is confirmed."""
        try:
            if not cancel_fn():
                raise ProviderError(
                    f"cancel_order({order_id}): the venue did not confirm the cancellation"
                )
        except ProviderError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"cancel_order({order_id}): {e}") from e

    def prepare_order_state(self) -> Any:
        """Synchronize provider state used by shared pre-submit policy checks."""
        return None

    def validate_order_state(self, expected_state: Any) -> Any:
        """Require that policy checks still describe the prepared provider state."""
        return expected_state

    def place_order(self, symbol: str, side: str, price: float, qty: float, force: bool = False, **kwargs):
        return None

    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: Optional[float] = None,
        *,
        market: bool = False,
        kind: Optional[str] = None,
        client_order_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Submit an order via the strict ``StrategyExecutor`` contract and return its venue order ID."""
        self.validate_symbol(symbol)
        s_norm = normalize_side(side)
        if s_norm not in {"BUY", "SELL"}:
            raise SubmissionRefused(f"submit_order({symbol}): invalid side={side!r}")
        if not self.execution_enabled():
            raise ProviderError(f"submit_order({symbol}): real execution is disabled on {self.name}")
        is_market = _parse_bool_flag(market, False) or price is None
        eff_price = price if price is not None else (self.get_current_price(symbol) or 0.0)
        call_kw = dict(kwargs)
        if is_market:
            call_kw["market"] = True
        if kind is not None:
            call_kw["kind"] = kind
        if client_order_id is not None:
            call_kw["client_order_id"] = client_order_id
        res = self.place_order(symbol, s_norm, eff_price, qty, force=is_market, **call_kw)
        oid = extract_order_id(res)
        if oid is None:
            raise ProviderError(f"submit_order({symbol}): response without orderId ({res})")
        return oid

    def order_status(self, symbol: str, order_id: str) -> OrderStatus:
        raise ProviderError(f"{self.name}: order_status is unsupported")

    def cancel_order(self, symbol: str, order_id: str) -> None:
        return None

    def guards_internally(self) -> bool:
        return False

    def _resolve_provider_attr(self, attr: str, default: Any = None) -> Any:
        inst_dict = getattr(self, "__dict__", {})
        if attr in inst_dict:
            return inst_dict[attr]
        if attr in type(self).__dict__:
            return getattr(self, attr, default)
        prov_cls = self.resolve_venue_provider_class(getattr(self, "name", ""))
        if prov_cls is not None and hasattr(prov_cls, attr):
            return getattr(prov_cls, attr)
        return getattr(self, attr, default)

    def _resolve_capability_flag(self, attr: str) -> bool:
        return bool(self._resolve_provider_attr(attr, False))

    def default_quote_asset(self) -> Optional[str]:
        """Return default quote asset when a venue symbol omits explicit quote suffix."""
        val = self._resolve_provider_attr("_DEFAULT_QUOTE_ASSET", None)
        return str(val).strip().upper() if isinstance(val, str) and val.strip() else None

    def requires_cache_permit(self) -> bool:
        """Return whether this provider enforces account-cache submit permits."""
        return self._resolve_capability_flag("_REQUIRES_CACHE_PERMIT")

    def defers_smart_cancellation(self) -> bool:
        """Return whether opposing order cancellation is deferred until dispatch."""
        return self._resolve_capability_flag("_DEFERS_SMART_CANCELLATION")

    def supports_business_minimum(self) -> bool:
        """Return whether this provider enforces venue business minimum rules."""
        return self._resolve_capability_flag("_SUPPORTS_BUSINESS_MINIMUM")

    def pair_precision(self, symbol: str) -> Optional[PairPrecision]:
        """Return normalized pair precision and limits, or None if unavailable."""
        return None

    def _pair_precision_attr(self, symbol: str, attr: str, default: Any = None) -> Any:
        try:
            pp = self.pair_precision(symbol) if hasattr(self, "pair_precision") else None
            if pp is not None:
                return getattr(pp, attr, default)
        except Exception:
            pass
        return default

    def round_quantity(self, symbol: str, qty: float) -> float:
        """Round or floor quantity to venue volume precision."""
        qty_f = safe_pos_float(qty, default=None)
        if qty_f is None:
            return 0.0
        dec = exact_nonneg_int(self._pair_precision_attr(symbol, "volume_decimals"), default=None)
        if dec is not None:
            factor = 10 ** min(dec, 18)
            return math.floor(qty_f * factor + 1e-12) / factor
        return qty_f

    round_amount = round_quantity

    def round_price(self, symbol: str, price: float) -> float:
        """Round price to venue price precision (tick size / price decimals)."""
        price_f = safe_pos_float(price, default=None)
        if price_f is None:
            return 0.0
        dec = exact_nonneg_int(self._pair_precision_attr(symbol, "price_decimals"), default=None)
        return round(price_f, min(dec, 18)) if dec is not None else price_f

    def price_tick_size(self, symbol: str, reference_price: Optional[float] = None) -> float:
        """Return venue price tick size for ``symbol`` across all providers (0.0 if unknown)."""
        try:
            meta = self.resolve_cached_fee_and_limits(symbol, price=reference_price)
            if isinstance(meta, dict):
                tick = safe_pos_float(meta.get("tick_size"), default=None)
                if tick is not None:
                    return tick
        except Exception:
            pass
        dec = exact_nonneg_int(self._pair_precision_attr(symbol, "price_decimals"), default=None)
        if dec is not None:
            return float(round(10.0 ** (-min(dec, 18)), 18))
        return 0.0

    def min_order_qty(self, symbol: str) -> float:
        """Return venue minimum order quantity; 0.0 means no known minimum."""
        return safe_pos_float(self._pair_precision_attr(symbol, "order_min", 0.0), default=0.0)

    def order_filter_refusal(self, symbol: str, side: str, price: float,
                             qty: float, *, market: bool = False,
                             enforce_business_minimum: bool = True
                             ) -> Optional[str]:
        """Return a provider-specific pre-submit refusal, if one is known."""
        qty_f = safe_pos_float(qty, default=None)
        if qty_f is None:
            return "qty_zero_or_negative"
        try:
            min_qty = self.min_order_qty(symbol)
            if min_qty > 0 and qty_f < min_qty:
                return f"filter_lot_size_min:{qty_f}<{min_qty}"
        except Exception:
            pass
        return None

    def policy_cap_quantity(self, symbol: str, side: str, price: float,
                            qty: float, available_qty: float, **kwargs) -> float:
        fn = self._policy_weight_limit_fn
        if callable(fn):
            return fn(self, symbol, side, price, qty, available_qty=available_qty)
        avail = safe_pos_float(available_qty, default=0.0) or 0.0
        req = float("inf") if qty is None or qty == float("inf") else (safe_pos_float(qty, default=0.0) or 0.0)
        return min(req, avail)

    def fee_cap_quantity(self, symbol: str, side: str, price: float,
                         available_qty: float) -> float:
        return venue_fee_cap_quantity(self.name, side, price, available_qty)

    def quantity_decision(self, symbol: str, side: str, price: float,
                          qty: float, **kwargs):
        return decide_quantity(self, symbol, side, price, qty, **kwargs)

    def adjust_order_price(self, symbol: str, side: str, price: float,
                           cancel_opposite: bool = True) -> float:
        return price

    def cancel_opposite_orders(self, symbol: str, side: str,
                               requested_price: float) -> None:
        """Cancel adverse opposing orders when the venue implements this policy."""
        return None

    def profit_guard_window_ref(
        self, symbol: str, side: str, safeback_sec, regime_context=None, reference_mode=None, module=None,
        *, prefer_safeback: bool = False, default_safeback_sec: float = 0.0,
    ):
        ref_fn = self._policy_window_reference_fn
        if not callable(ref_fn):
            return None
        sb_f = safe_pos_float(safeback_sec, default=None)
        if prefer_safeback and sb_f is not None:
            window_s = sb_f
        else:
            cfg_fn = self._policy_window_cfg_fn
            win_cfg = (
                cfg_fn(
                    self.name,
                    symbol=symbol,
                    order_type=side,
                    regime_context=regime_context,
                    module=module,
                    reference_mode=reference_mode,
                )
                if callable(cfg_fn)
                else None
            )
            win_f = safe_pos_float(win_cfg, default=None)
            window_s = win_f if win_f is not None else (sb_f if sb_f is not None else default_safeback_sec)
        wr_kwargs = {k: v for k, v in (("reference_mode", reference_mode), ("module", module)) if v is not None}
        return ref_fn(self, symbol, side, window_s, **wr_kwargs)

    def last_opposite_fill(self, symbol: str, order_type: str,
                           since_s: float = 90 * 24 * 3600) -> Optional[float]:
        ot_norm = normalize_side(order_type)
        if ot_norm not in ("BUY", "SELL"):
            return None
        opposite = "SELL" if ot_norm == "BUY" else "BUY"
        eff_since_s = safe_nonneg_float(since_s, default=90.0 * 24.0 * 3600.0)
        fills = self.get_orders(symbol, opposite, eff_since_s)
        if fills is None:
            raise ProviderError(f"last_opposite_fill({symbol}): failed to read order history")
        valid_fills = [o for o in fills if isinstance(o, dict) and extract_order_price(o, default=0.0) > 0.0]
        if not valid_fills:
            return None
        latest = max(valid_fills, key=lambda order: extract_timestamp_ms(order, default=0) or 0)
        return extract_order_price(latest) or None

    def ohlc_closes(self, symbol: str, interval_min: int) -> list:
        """Return completed-bar closing prices from closed interval series."""
        if hasattr(self, "ohlc_series"):
            series = self.ohlc_series(symbol, interval_min)
            closes = getattr(series, "closes", None) if series is not None else None
            return list(closes) if closes is not None else []
        return []

    # ── Agnostic SSOT cacheManager coupling (price, 24h history, short trend) ─

    def _ssot_symbol_candidates(self, symbol: str) -> tuple[str, ...]:
        """Return venue-scoped symbol keys to look up in SSOT cacheManager stores."""
        if isinstance(symbol, bool) or not isinstance(symbol, str) or not symbol.strip():
            return ()
        raw_u = symbol.strip().upper()
        cleaned = clean_symbol_key(raw_u)
        parts = split_symbol_assets(raw_u)
        canon = canonical_symbol_key(raw_u) if parts is not None else ""
        return tuple(dict.fromkeys(c for c in (cleaned, raw_u, canon) if c))

    _DEFINITIVE_REFUSAL_TOKENS = (
        "insufficient",
        "minimum value",
        "order minimum",
        "volume minimum",
        "cost minimum",
        "invalid price",
        "invalid size",
        "invalid order",
        "invalid arguments",
        "positions limit exceeded",
        "tick size",
        "could not immediately match",
        "price must be",
        "too far from",
    )

    @classmethod
    def _is_definitive_refusal(cls, error: object) -> bool:
        """Return True when a venue error message indicates a permanent order rejection."""
        text = str(error or "").lower()
        return any(token in text for token in cls._DEFINITIVE_REFUSAL_TOKENS)

    clamp_price_to_bbo_mid = staticmethod(clamp_price_to_bbo_mid)

    def _allow_ssot_price_cache(self) -> bool:
        """Return True when this provider should consult cacheManager SSOT before venue HTTP."""
        if is_in_live_price_fetch():
            return False
        if getattr(self, "_ssot_cache_enabled", None) is False:
            return False
        has_explicit_ssot = bool(
            getattr(self, "_ssot_cache_enabled", None) is True
            or getattr(self, "_force_ssot_cache", False)
            or getattr(self, "_ssot_price_cache_path", None)
            or getattr(self, "_ssot_trend_cache_path", None)
            or getattr(self, "_ssot_24price_cache_dir", None)
            or getattr(self, "_ssot_fills_cache_path", None)
        )
        inst_dict = getattr(self, "__dict__", {})
        inj_client = inst_dict.get("_cli")
        if inj_client is None:
            inj_client = inst_dict.get("_client")
        client_cls_name = self._resolve_provider_attr("_CLIENT_CLASS_NAME", None)
        is_real_client = bool(
            client_cls_name and type(inj_client).__name__ == client_cls_name
        )
        if inj_client is not None and not is_real_client and not has_explicit_ssot:
            return False
        if has_explicit_ssot:
            return True
        return is_shared_disk_cache_enabled()

    def _cached_current_price(self, symbol: str, max_age_sec: float = 12.0) -> Optional[float]:
        """Read fresh current price for ``symbol`` from CacheCurrentPriceManager / cache_current_price.json."""
        if not self._USES_SSOT_MARKET_CACHE or not self._allow_ssot_price_cache():
            return None
        eff_age = 12.0 if max_age_sec is None else safe_pos_float(max_age_sec, default=None)
        if eff_age is None:
            return None
        candidates = self._ssot_symbol_candidates(symbol)
        if not candidates:
            return None
        from cacheprice import CacheCurrentPriceManager

        custom_path = getattr(self, "_ssot_price_cache_path", None)
        price_path = str(custom_path) if custom_path else cache_path(CACHE_CURRENT_PRICE_FILENAME)
        return CacheCurrentPriceManager.read_ssot_current_price_candidates(
            candidates,
            price_path,
            eff_age,
            use_memory_manager=(custom_path is None),
            reader_fn=self._ssot_current_price_reader,
        )

    def _cached_price_history(
        self,
        symbol: str,
        lookback_hours: float,
        max_latest_age_sec: float = 120.0,
    ) -> Optional[list[dict]]:
        """Read 24h tick history from Cache24PriceManager when fresh and covering ``lookback_hours``."""
        if not self._USES_SSOT_MARKET_CACHE or not self._allow_ssot_price_cache():
            return None
        candidates = self._ssot_symbol_candidates(symbol)
        if not candidates:
            return None
        from cacheprice import Cache24PriceManager

        return Cache24PriceManager.read_ssot_price_history(
            candidates,
            lookback_hours,
            max_latest_age_sec=max_latest_age_sec,
            custom_dir=getattr(self, "_ssot_24price_cache_dir", None),
        )

    @classmethod
    def normalize_orderbook_payload(cls, raw_data: Any) -> Optional[dict]:
        """Normalize L2 orderbook payloads into ``{'bids': [(px, qty), ...], 'asks': [(px, qty), ...]}``."""
        if not isinstance(raw_data, dict):
            return None
        data = raw_data
        if "bids" not in data:
            extract_self = getattr(cls, "extract_orderbook_sides", None)
            extracted = extract_self(data) if callable(extract_self) else None
            if not isinstance(extracted, dict):
                for venue_key in cls._VENUE_PROVIDER_SPECS:
                    prov_cls = cls.resolve_venue_provider_class(venue_key)
                    extract_fn = getattr(prov_cls, "extract_orderbook_sides", None) if prov_cls is not None else None
                    if callable(extract_fn):
                        extracted = extract_fn(data)
                        if isinstance(extracted, dict):
                            break
            if not isinstance(extracted, dict):
                return None
            data = extracted

        def _parse_side(rows: Any) -> list[tuple[float, float]]:
            if not isinstance(rows, list):
                return []
            out_side: list[tuple[float, float]] = []
            for item in rows:
                if isinstance(item, dict):
                    raw_p = item.get("px") if item.get("px") is not None else item.get("price")
                    raw_q = next((item.get(k) for k in ("sz", "qty", "volume") if item.get(k) is not None), None)
                    pf = safe_pos_float(raw_p, default=None)
                    qf = safe_pos_float(raw_q, default=None)
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    pf = safe_pos_float(item[0], default=None)
                    qf = safe_pos_float(item[1], default=None)
                else:
                    continue
                if pf is not None and qf is not None:
                    out_side.append((pf, qf))
            return out_side

        bids = _parse_side(data.get("bids"))
        asks = _parse_side(data.get("asks"))
        if not bids or not asks:
            return None
        bids.sort(key=lambda x: x[0], reverse=True)
        asks.sort(key=lambda x: x[0])
        return {"bids": bids, "asks": asks}

    def read_ssot_fills_cache(
        self,
        symbol: str,
        cache_file: str,
        *,
        max_stale_sec: float = 90.0,
        match_fn: Any = None,
        normalize_fn: Any = None,
    ) -> Optional[List[dict]]:
        """Read normalized fills for ``symbol`` from a shared SSOT trade cache file (`items` + `fetchtime`)."""
        if not symbol or isinstance(symbol, bool) or not str(symbol).strip() or not cache_file:
            return None
        from cacheaccount import CacheTradeManager

        return CacheTradeManager.read_venue_fills_snapshot(
            symbol,
            cache_file,
            max_stale_sec=max_stale_sec,
            match_fn=match_fn,
            candidate_keys=None if callable(match_fn) else self._ssot_symbol_candidates(symbol),
            normalize_fn=normalize_fn,
        )

    def write_ssot_fills_cache(self, symbol: str, cache_file: str, fills: List[dict]) -> bool:
        """Persist ``fills`` for ``symbol`` into the shared SSOT trade cache file (`items` + `fetchtime`)."""
        from cacheaccount import CacheTradeManager

        return CacheTradeManager.write_venue_fills_snapshot(symbol, cache_file, fills)

    def invalidate_ssot_fills_cache(
        self, symbol: str, cache_file: str, *, match_fn: Any = None
    ) -> None:
        """Expire ``fetchtime`` for ``symbol`` in the shared SSOT trade cache after order execution."""
        if not symbol or isinstance(symbol, bool) or not str(symbol).strip() or not cache_file:
            return
        from cacheaccount import CacheTradeManager

        CacheTradeManager.invalidate_venue_fills_snapshot(
            symbol,
            cache_file,
            match_fn=match_fn,
            candidate_keys=None if callable(match_fn) else self._ssot_symbol_candidates(symbol),
        )


def mount_connect_retry(session, *, attempts: int = 3, backoff: float = 0.4,
                        read: int = 0, status_forcelist=None,
                        raise_on_status: bool = False) -> bool:
    """Mount an order-safe Retry adapter on ``session`` for https:// and http:// (best-effort)."""
    try:
        if session is None or isinstance(status_forcelist, (bool, str, bytes)):
            return False
        att_i = safe_nonneg_int(attempts, default=None)
        read_i = safe_nonneg_int(read, default=None)
        bo_f = safe_nonneg_float(backoff, default=None)
        if att_i is None or read_i is None or bo_f is None:
            return False
        forcelist = None
        if status_forcelist:
            forcelist = []
            for code in status_forcelist:
                c_i = safe_pos_int(code, default=None)
                if c_i is None:
                    return False
                forcelist.append(c_i)
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        retry = Retry(
            total=att_i,
            connect=att_i,
            read=read_i,
            status=att_i if forcelist else 0,
            redirect=0,
            backoff_factor=bo_f,
            status_forcelist=forcelist if forcelist else None,
            raise_on_status=_parse_bool_flag(raise_on_status, default=False),
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return True
    except Exception:  # noqa: BLE001 -- retries are best-effort; the timeout still guards
        return False

