"""Core MarketDataProvider contract with modular SSOT-bridge, venue-metadata, and WS stream layers."""

import importlib
import inspect
import math
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional

from botcore import (  # noqa: F401
    ClosedPriceSeries,
    _parse_bool_flag,
    _ts_ms_to_s,
    bind_attr_call,
    bool_env,
    clean_upper_token,
    exact_nonneg_int,
    extract_order_id,
    extract_order_price,
    extract_timestamp_ms,
    extract_timestamp_s,
    is_shared_disk_cache_enabled,
    load_env_file_defaults,
    load_local_module,
    normalize_side,
    normalize_timestamp_ms,
    require_pos_float,
    resolve_patched_attr,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
)
from cachepaths import (  # noqa: F401
    cache_path,
    split_symbol_assets,
)
import instrument_registry as _default_ir  # noqa: F401

from .ohlc_utils import extract_yahoo_chart_result_and_quote  # noqa: F401
from .order_ohlc_utils import (  # noqa: F401
    POST_ONLY_CROSS_TOKENS,
    VENUE_PROVIDER_SPECS,
    aggregate_fills_by_order_id,
    compute_post_only_maker_price,
    is_post_only_cross_refusal,
)
from .quantity import (  # noqa: F401
    compute_fills_cost_basis,
    decide_quantity,
    venue_fee_cap_quantity,
)
from .ssot_bridge import (  # noqa: F401
    ProviderSsotBridgeMixin,
    is_in_live_price_fetch,
    live_price_fetch_context,
)
from .strategy_executor import (  # noqa: F401
    OrderReconciliationCapabilities,
    OrderStatus,
    PairPrecision,
    ProviderError,
    SubmissionRefused,
)
from .venue_metadata import ProviderVenueMetadataMixin  # noqa: F401
from .ws_stream import (  # noqa: F401
    AsyncReconnectStreamBase,
    PublicPriceStreamBase,
    WS_CLOSE_TIMEOUT,
    WS_PING_INTERVAL,
    WS_PING_TIMEOUT,
    WS_RECV_TIMEOUT,
    WS_RETRY_INITIAL,
    WS_RETRY_MAX,
    WS_STOP_TIMEOUT,
    _ws_logger,
    mount_connect_retry,
)

_ORIG_LOAD_LOCAL_MODULE = load_local_module
_UNSET_INTERVAL = object()


class MarketDataProvider(ProviderSsotBridgeMixin, ProviderVenueMetadataMixin, ABC):
    """Shared market-data, account, order, and policy-hook interface."""

    def _fetch_current_price(self, symbol: str) -> Optional[float]:
        """Venue-specific live quote hook invoked on SSOT cache miss."""
        return None

    def _fetch_price_history(self, symbol: str, lookback_h: float) -> Optional[List]:
        """Venue-specific historical price series hook invoked on SSOT cache miss."""
        return None

    @staticmethod
    def _log_provider_balance_error(
        tag: str, label: str, asset: Any, exc: Exception, *, fmt: str = "[{tag}] {label} {asset}: {exc}"
    ) -> None:
        if not label and asset is None:
            print(f"[{tag}] {exc}")
        else:
            print(fmt.format(tag=tag, label=label, asset=asset, exc=exc))
        return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Return fresh current price for ``symbol`` (SSOT cache first when enabled, then venue transport)."""
        self.validate_symbol(symbol)
        if not symbol or isinstance(symbol, bool) or not str(symbol).strip():
            return None
        return (
            cached
            if (cached := self._cached_current_price(symbol)) is not None
            else safe_pos_float(self._fetch_current_price(symbol), default=None)
        )

    def get_price_history(self, symbol: str, lookback_h: float) -> Optional[List]:
        """Return ascending historical prices over ``lookback_h`` hours (SSOT 24h cache first, then venue transport)."""
        self.validate_symbol(symbol)
        if (
            not symbol
            or isinstance(symbol, bool)
            or not str(symbol).strip()
            or (lookback_f := safe_nonneg_float(lookback_h, default=None)) is None
        ):
            return None
        return (
            cached_hist
            if (cached_hist := self._cached_price_history(symbol, lookback_f)) is not None
            else self._fetch_price_history(symbol, lookback_f)
        )

    @abstractmethod
    def supports_symbol(self, symbol: str) -> bool:
        ...

    _REQUIRES_CACHE_PERMIT: bool = False
    _DEFERS_SMART_CANCELLATION: bool = False
    _SUPPORTS_BUSINESS_MINIMUM: bool = False
    _SUPPORTS_ORDER_VALIDITY: bool = False
    _DEFERS_PREFLIGHT_QUANTITY: bool = False
    _LIVE_ORDERS_ENV: Optional[str] = None
    _LIVE_ORDERS_DIR_ATTR: Optional[str] = None
    _LIVE_ORDERS_EXEC_READS_DIR: bool = True
    _RECONCILIATION_CAPABILITIES: OrderReconciliationCapabilities = OrderReconciliationCapabilities()
    _OHLC_MARKET_DATA_SPEC: Optional[tuple[Optional[str], str, str]] = None
    _OHLC_DEFAULT_INTERVAL: Any = 60
    _REMOTE_TRADES_CLIENT_ATTR: Optional[str] = None

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

    def validate_open_order_symbol(self, symbol: str) -> None:
        """Validate ``symbol`` before querying open orders on this venue."""
        self.validate_symbol(symbol)

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

    def _require_submitted_order_id(
        self,
        symbol: str,
        order_id: Optional[str],
        missing_msg: str,
        *,
        invalidate_fills_cache: bool = True,
    ) -> str:
        """Require a non-empty ``order_id``, optionally invalidate ``symbol`` fills cache, and return ``order_id``."""
        if order_id is None:
            raise ProviderError(missing_msg)
        if invalidate_fills_cache:
            self._invalidate_fills_cache(symbol)
        return order_id

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

    @classmethod
    def fetch_remote_trades_with_client(
        cls, symbol: str, start_time_ms: int = 0, client=None
    ) -> Optional[List[dict]]:
        """Fetch normalized remote fills for ``symbol`` using an injected ``client`` when compatible."""
        req_attr = getattr(cls, "_REMOTE_TRADES_CLIENT_ATTR", None)
        if not req_attr or client is None or not callable(getattr(client, req_attr, None)):
            return None
        start_ms = safe_nonneg_int(start_time_ms, default=0)
        return cls(client=client).fetch_remote_trades(symbol, start_ms, client=client)

    def _remote_trade_rows_for_symbol(
        self, symbol: str, sym_u: str, start_ms: float, *, client=None
    ) -> Any:
        return self._invoke_fills_from_api(symbol, publish_cache=False)

    def fetch_remote_trades(
        self, symbol: str, start_time_ms: int = 0, *, client=None
    ) -> List[dict]:
        """Fetch and normalize remote fills for ``symbol`` since ``start_time_ms`` without mutating disk caches."""
        if _parse_bool_flag(getattr(self, "_VALIDATE_REMOTE_TRADES_SYMBOL", True), default=True):
            self.validate_symbol(symbol)
        if not symbol or isinstance(symbol, bool) or not str(symbol).strip():
            return []
        start_ms = 0.0 if start_time_ms is None else safe_nonneg_float(start_time_ms, default=None)
        if start_ms is None:
            return []
        sym_u = str(symbol).strip().upper()
        rows = self._remote_trade_rows_for_symbol(symbol, sym_u, start_ms, client=client)
        if not isinstance(rows, list):
            return []
        return [
            dict(r)
            for r in rows
            if isinstance(r, dict)
            and (start_ms <= 0.0 or (extract_timestamp_ms(r, default=0) or 0) >= start_ms)
        ]

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

    def get_trade_orders(
        self,
        order_type: Optional[str],
        symbol: str,
        max_age_seconds: float,
    ) -> List[dict]:
        """Return recent fill rows for ``symbol`` and optional ``order_type`` within ``max_age_seconds``."""
        return self.get_orders(symbol, order_type, max_age_seconds)

    def position_cost_basis(self, symbol: str, held_qty: float, since_s: float) -> Optional[float]:
        """Return inventory-reconciled acquisition cost from fresh SSOT fills, or None when unknown."""
        since_val = require_pos_float(since_s, "since_s")
        if (sym_u := clean_upper_token(symbol)) is None:
            return None
        fills_fn = getattr(self, "_fills_from_cache", None)
        rows = fills_fn(symbol) if callable(fills_fn) else None
        if rows is None:
            return None
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
        """Return explicitly supported strict lifecycle operations."""
        caps = MarketDataProvider._resolve_provider_attr(self, "_RECONCILIATION_CAPABILITIES", None)
        return caps if isinstance(caps, OrderReconciliationCapabilities) else OrderReconciliationCapabilities()

    def preflight_order(self, symbol: str, side: str, qty: float,
                        price=None, *, market: bool = False,
                        kind: Optional[str] = None) -> Any:
        """Validate venue state and optionally return opaque submit authorization."""
        return None

    @classmethod
    def _resolve_live_orders_dir(cls) -> Optional[str]:
        dir_attr = getattr(cls, "_LIVE_ORDERS_DIR_ATTR", None)
        if not dir_attr:
            return None
        val = resolve_patched_attr(dir_attr, None, modules=(cls.__module__,))
        return str(val) if isinstance(val, (str, Path)) and str(val) else None

    @classmethod
    def _venue_live_orders_enabled(cls) -> bool:
        """Return whether the venue-level live-order config gate is enabled."""
        env_key = getattr(cls, "_LIVE_ORDERS_ENV", None)
        return bool_env(env_key, cls._resolve_live_orders_dir()) if env_key else True

    def _before_execution_enabled_check(self) -> None:
        """Optional hook run before reading ``_LIVE_ORDERS_ENV`` when ``_live_enabled`` is unset."""

    def execution_enabled(self) -> bool:
        """Return whether this provider may create a real venue order now."""
        live_override = getattr(self, "_live_enabled", None)
        if live_override is not None:
            return _parse_bool_flag(live_override, False)
        self._before_execution_enabled_check()
        env_key = getattr(type(self), "_LIVE_ORDERS_ENV", None)
        if env_key:
            folder = (
                type(self)._resolve_live_orders_dir()
                if getattr(type(self), "_LIVE_ORDERS_EXEC_READS_DIR", True)
                else None
            )
            live_override = getattr(self, "_live_enabled", None)
            return bool_env(env_key, folder) if live_override is None else _parse_bool_flag(live_override, False)
        return type(self)._venue_live_orders_enabled()

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

    @classmethod
    def strategy_dry_run(cls, force_paper: bool, strategy_execute: bool) -> bool:
        """Combine CLI ``--paper``, strategy gate (``STRAT_EXECUTE``), and this venue's live gate."""
        return cls.resolve_strategy_dry_run(
            force_paper, strategy_execute, cls._venue_live_orders_enabled()
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

    def _resolve_live_order_flag(self, kwargs: dict) -> bool:
        """Pop optional ``live_enabled`` override from ``kwargs`` or fall back to ``self.execution_enabled()``."""
        live_override = kwargs.pop("live_enabled", None) if isinstance(kwargs, dict) else None
        return (
            self.execution_enabled()
            if live_override is None
            else _parse_bool_flag(live_override, False)
        )

    def _validate_submit_order_side(
        self, symbol: str, side: str, *, validate_sym: bool = True
    ) -> str:
        """Validate ``symbol`` (when requested) and return normalized ``'BUY'``/``'SELL'`` or raise ``SubmissionRefused``."""
        if validate_sym:
            self.validate_symbol(symbol)
        s_norm = normalize_side(side)
        if s_norm not in {"BUY", "SELL"}:
            raise SubmissionRefused(f"submit_order({symbol}): invalid side={side!r}")
        return s_norm

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
        s_norm = self._validate_submit_order_side(symbol, side)
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
        return self._require_submitted_order_id(
            symbol,
            extract_order_id(res),
            f"submit_order({symbol}): response without orderId ({res})",
            invalidate_fills_cache=False,
        )

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
        prov_cls = MarketDataProvider.resolve_venue_provider_class(getattr(self, "name", ""))
        if prov_cls is not None and hasattr(prov_cls, attr):
            return getattr(prov_cls, attr)
        return getattr(self, attr, default)

    def default_quote_asset(self) -> Optional[str]:
        """Return default quote asset when a venue symbol omits explicit quote suffix."""
        val = MarketDataProvider._resolve_provider_attr(self, "_DEFAULT_QUOTE_ASSET", None)
        return str(val).strip().upper() if isinstance(val, str) and val.strip() else None

    def _provider_bool_attr(self, attr: str) -> bool:
        return bool(MarketDataProvider._resolve_provider_attr(self, attr, False))

    requires_cache_permit = bind_attr_call("_provider_bool_attr", "_REQUIRES_CACHE_PERMIT")
    defers_smart_cancellation = bind_attr_call("_provider_bool_attr", "_DEFERS_SMART_CANCELLATION")
    supports_business_minimum = bind_attr_call("_provider_bool_attr", "_SUPPORTS_BUSINESS_MINIMUM")
    supports_order_validity = bind_attr_call("_provider_bool_attr", "_SUPPORTS_ORDER_VALIDITY")
    supports_post_only = bind_attr_call("_provider_bool_attr", "_SUPPORTS_POST_ONLY")
    defers_preflight_quantity = bind_attr_call("_provider_bool_attr", "_DEFERS_PREFLIGHT_QUANTITY")

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
        if (qty_f := safe_pos_float(qty, default=None)) is None:
            return 0.0
        dec = exact_nonneg_int(self._pair_precision_attr(symbol, "volume_decimals"), default=None)
        if dec is not None:
            factor = 10 ** min(dec, 18)
            return math.floor(qty_f * factor + 1e-12) / factor
        return qty_f

    round_amount = round_quantity

    def round_price(self, symbol: str, price: float) -> float:
        """Round price to venue price precision (tick size / price decimals)."""
        if (price_f := safe_pos_float(price, default=None)) is None:
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
            return round(10.0 ** (-min(dec, 18)), 18)
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
        req = math.inf if qty is None or (not isinstance(qty, bool) and qty == math.inf) else (safe_pos_float(qty, default=0.0) or 0.0)
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

    _POST_ONLY_CROSS_TOKENS = POST_ONLY_CROSS_TOKENS

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
        *_POST_ONLY_CROSS_TOKENS,
    )

    @classmethod
    def _is_post_only_cross_refusal(cls, error: object) -> bool:
        """Return True when a venue error indicates a post-only order would cross the book."""
        if cls._POST_ONLY_CROSS_TOKENS is POST_ONLY_CROSS_TOKENS:
            return is_post_only_cross_refusal(error)
        text = str(error or "").lower()
        return any(token in text for token in cls._POST_ONLY_CROSS_TOKENS)

    @classmethod
    def _is_definitive_refusal(cls, error: object) -> bool:
        """Return True when a venue error message indicates a permanent order rejection."""
        text = str(error or "").lower()
        return any(token in text for token in cls._DEFINITIVE_REFUSAL_TOKENS)

    def step_post_only_maker_price(
        self,
        symbol: str,
        side: str,
        price: float,
        current_price: Optional[float] = None,
    ) -> float:
        """Step ``price`` onto the passive Maker side of the spread after a post-only cross rejection."""
        px_f = safe_pos_float(price, default=None)
        s_norm = normalize_side(side)
        if px_f is None or s_norm not in ("BUY", "SELL"):
            return safe_pos_float(price, default=0.0) or 0.0
        cur_px = safe_pos_float(current_price, default=None)
        if cur_px is None:
            try:
                cur_px = safe_pos_float(self.get_current_price(symbol), default=None)
            except Exception:
                cur_px = None
        anchor = (min(px_f, cur_px) if s_norm == "BUY" else max(px_f, cur_px)) if cur_px else px_f
        tick = safe_pos_float(self.price_tick_size(symbol, reference_price=anchor), default=0.0) or 0.0
        return compute_post_only_maker_price(
            s_norm,
            px_f,
            current_price=cur_px,
            tick_size=tick,
            round_price_fn=lambda p: self.round_price(symbol, p),
        )

    def fetch_historical_ohlc_points(
        self,
        symbol: str,
        days: int,
        interval: str = "1h",
    ) -> list[tuple[float, float]]:
        """Return historical ``(epoch_sec, close_price)`` OHLC points for ``symbol``."""
        return []

    def fetch_ohlc_candles(
        self,
        symbol: str,
        interval: Any = _UNSET_INTERVAL,
        *,
        drop_incomplete: bool = True,
        **kwargs: Any,
    ) -> List[dict]:
        """Return historical OHLC bar dictionaries ``{'timestamp', 'open', 'high', 'low', 'close'}``."""
        spec = MarketDataProvider._resolve_provider_attr(self, "_OHLC_MARKET_DATA_SPEC", None)
        if not (isinstance(spec, tuple) and len(spec) == 3):
            return []
        eff_interval = (
            MarketDataProvider._resolve_provider_attr(self, "_OHLC_DEFAULT_INTERVAL", 60)
            if interval is _UNSET_INTERVAL
            else interval
        )
        mod_dir, mod_name, fn_name = spec
        if mod_dir:
            loader = resolve_patched_attr(
                "load_local_module",
                _ORIG_LOAD_LOCAL_MODULE,
                local_val=load_local_module,
                defaults=(_ORIG_LOAD_LOCAL_MODULE,),
                modules=(type(self).__module__, __name__),
                require_callable=True,
            ) or load_local_module
            md = loader(mod_dir, mod_name)
        else:
            md = importlib.import_module(mod_name)
        fetch_fn = getattr(md, fn_name)
        return list(
            fetch_fn(
                symbol,
                interval=eff_interval,
                drop_incomplete=drop_incomplete,
                **kwargs,
            )
            or []
        )

    def fetch_intraday_chart(
        self,
        symbol: str,
        *,
        rng: str = "5d",
        interval: str = "5m",
    ) -> tuple[Any, list]:
        """Return ``(meta, bars)`` intraday chart series including extended-hours bars when supported."""
        return None, []
