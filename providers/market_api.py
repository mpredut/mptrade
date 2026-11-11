# market_api.py
"""Multi-venue facade for market data, account reads, and order mechanics.

Symbol routing selects the first provider that claims a symbol and falls back to
the first configured provider (``BinanceProvider``). ``Instrument`` uses the name
registry for explicit venue routing when a symbol alone is ambiguous.

``place_order`` and ``submit_order`` dispatch directly to adapter mechanics and
live-order gates. ``place`` constructs an ``Instrument`` and runs the shared policy
pipeline before those mechanics. Concrete venue SDKs remain lazy inside their
respective ``*_provider.py`` modules so ``cacheManager`` can import this facade
without creating import cycles.
"""
import time
from collections import OrderedDict
from typing import Any, Callable, List, Optional

import botcore
from botcore import (
    _parse_bool_flag,
    lru_dict_put,
    normalize_scalar_order_id,
    require_finite_float,
    require_pos_float,
    safe_nonneg_float,
    safe_nonneg_ts,
    safe_pos_float,
)
from cachepaths import normalize_venue_name
from instrument_registry import is_equity_symbol
import pricewindow as pw
from . import order_id_context as rc
from .base import (
    MarketDataProvider,
    extract_order_price,
    extract_timestamp_ms,
    normalize_side,
)
from .order_ohlc_utils import (
    clean_order_id,
    extract_order_qty,
    extract_order_side,
    extract_orig_qty,
    extract_remaining_qty,
    iter_expired_open_order_ids,
    require_clean_order_id,
    require_int_order_id,
)
from .quantity import resolve_assets
from .strategy_executor import (
    OrderReconciliationCapabilities,
    OrderStatus,
    PairPrecision,
    ProviderError,
    reconciliation_capabilities_of,
)
from market_regime import (
    MarketRegimeContext,
    MarketRegimeDecision,
    MarketRegimeEvaluator,
    MarketRegimeResolution,
    MarketRegimeService,
)

from .binance_provider import (  # noqa: F401
    BinanceProvider,
    _allorders,
    _bapi,
    _get_allorders,
    _get_bapi,
)


def _runtime_market_regime_snapshot(symbol: str, now=None):
    """Read the optional short-trend snapshot at the application boundary."""
    try:
        mgr = MarketDataProvider.get_ssot_short_trend_manager()
        snap_fn = getattr(mgr, "get_fresh_snapshot", None)
        snap = snap_fn(symbol, now=now) if callable(snap_fn) else None
        return pw.resolve_dynamic_trend_snapshot(snap, symbol)
    except Exception:
        return None


class MarketApi:
    """Route each symbol to the first provider that claims it.

    Fall back to the first configured provider, Binance, and memoize routes. The
    get_current_price signature matches bapi, making this a drop-in data facade.
    """

    def __init__(
        self,
        providers: List[MarketDataProvider],
        *,
        regime_service: Optional[MarketRegimeService] = None,
        regime_snapshot_resolver: Optional[Callable] = None,
    ):
        if not providers:
            raise ValueError("MarketApi: the provider list cannot be empty")
        self._providers: List[MarketDataProvider] = list(providers)
        self._route: dict = {}   # Lock-free, idempotent symbol-to-provider memoization.
        # Name registry enables explicit venue routing for Instrument rather than
        # guessing via supports_symbol. It does not alter symbol routing below.
        self._by_name: dict = {p.name.lower(): p for p in self._providers}
        self._regime_service = regime_service or MarketRegimeService()
        self._regime_snapshot_resolver = regime_snapshot_resolver
        self.stop: bool = False

    def _provider_explicit_or_routed(self, symbol: str, provider_name=None):
        if provider_name is None:
            provider = self._provider_for(symbol)
        else:
            provider = self.provider_by_name(provider_name)
            if provider is None:
                raise ValueError(f"Unknown provider: {provider_name!r}")
        validate_fn = getattr(provider, "validate_symbol", None)
        if callable(validate_fn):
            validate_fn(symbol)
        return provider

    def _provider_for(self, symbol: str) -> MarketDataProvider:
        if isinstance(symbol, bool) or not isinstance(symbol, str):
            return self._providers[0]
        sym_clean = symbol.strip()
        provider = self._route.get(symbol) or (self._route.get(sym_clean) if sym_clean else None)
        if provider is not None:
            return provider
        for candidate in self._providers:
            try:
                if candidate.supports_symbol(sym_clean or symbol):
                    self._route[symbol] = candidate
                    if sym_clean:
                        self._route[sym_clean] = candidate
                    return candidate
            except Exception:
                continue
        # Behavior-preserving default: first configured provider.
        default = self._providers[0]
        self._route[symbol] = default
        return default

    def get_current_price(
        self, symbol: str, *, provider_name=None,
    ) -> Optional[float]:
        """Read a quote through explicit venue routing when it is configured."""
        return self._provider_explicit_or_routed(symbol, provider_name).get_current_price(symbol)

    def get_price_history(
        self, symbol: str, lookback_h: float, *, provider_name=None,
    ) -> Optional[List]:
        return self._provider_explicit_or_routed(symbol, provider_name).get_price_history(symbol, lookback_h)

    def ohlc_closes(
        self, symbol: str, interval_min: int, *, provider_name=None,
    ) -> list[float]:
        provider = self._provider_explicit_or_routed(symbol, provider_name)
        return provider.ohlc_closes(symbol, interval_min) if hasattr(provider, "ohlc_closes") else []

    def fetch_ohlc_candles(
        self,
        symbol: str,
        interval: Any = 60,
        *,
        drop_incomplete: bool = True,
        provider_name=None,
        **kwargs: Any,
    ) -> List[dict]:
        """Return historical OHLC bar dictionaries routed through the target venue provider."""
        return self._provider_explicit_or_routed(symbol, provider_name).fetch_ohlc_candles(
            symbol,
            interval,
            drop_incomplete=drop_incomplete,
            **kwargs,
        )

    def _default_provider_for_scope(self, scope: str, symbol: str) -> MarketDataProvider:
        for candidate in self._providers:
            if getattr(candidate, "_ASSET_SCOPE", "crypto") == scope:
                return candidate
        return self._provider_for(symbol)

    def fetch_intraday_chart(
        self,
        symbol: str,
        *,
        rng: str = "5d",
        interval: str = "5m",
        provider_name=None,
    ) -> tuple[Any, list]:
        """Return ``(meta, bars)`` intraday chart data routed through the target venue provider."""
        if provider_name is not None or self.supports_symbol(symbol):
            provider = self._provider_explicit_or_routed(symbol, provider_name)
        else:
            provider = self._default_provider_for_scope("equities", symbol)
            validate_fn = getattr(provider, "validate_symbol", None)
            if callable(validate_fn):
                validate_fn(symbol)
        return provider.fetch_intraday_chart(
            symbol,
            rng=rng,
            interval=interval,
        )

    def fetch_historical_ohlc_points(
        self,
        symbol: str,
        days: int,
        interval: str = "1h",
    ) -> list[tuple[float, float]]:
        """Return historical ``(epoch_sec, close_price)`` OHLC points across configured venue providers."""
        sym_u = str(symbol or "").strip().upper()
        is_eq = is_equity_symbol(sym_u)
        target_scope = "equities" if is_eq else "crypto"
        query_sym = sym_u if is_eq else symbol
        candidates = sorted(
            (
                prov
                for prov in self._providers
                if getattr(prov, "_ASSET_SCOPE", "crypto") == target_scope
                and getattr(prov, "_HISTORICAL_OHLC_PRIORITY", None) is not None
            ),
            key=lambda prov: int(prov._HISTORICAL_OHLC_PRIORITY),
        )
        if not candidates:
            return []
        pts = candidates[0].fetch_historical_ohlc_points(query_sym, days, interval=interval)
        if not is_eq and len(pts) < 50 and len(candidates) > 1 and botcore.is_live_network_backfill_enabled():
            for fallback_prov in candidates[1:]:
                try:
                    fb_pts = fallback_prov.fetch_historical_ohlc_points(query_sym, days, interval=interval)
                    if len(fb_pts) >= 50:
                        pts = fb_pts
                        break
                except Exception:
                    pass
        return pts

    # -- Account routing by symbol/asset, normalized by provider. ---------------
    def free_balance(self, asset: str, *, provider_name=None) -> Optional[float]:
        """Return free balance for asset from its routed or explicitly named provider."""
        if provider_name is None:
            provider = self._provider_for(asset)
        else:
            provider = self.provider_by_name(provider_name)
            if provider is None:
                raise ValueError(f"Unknown provider: {provider_name!r}")
        return provider.free_balance(asset)

    def get_orders(
        self,
        symbol: str,
        side: Optional[str],
        since_s: float,
        *,
        provider_name=None,
        authoritative: bool = False,
        publish_cache: bool = False,
    ) -> List[dict]:
        prov = self._provider_explicit_or_routed(symbol, provider_name)
        if authoritative or publish_cache:
            return prov.get_orders(
                symbol,
                side,
                since_s,
                authoritative=authoritative,
                publish_cache=publish_cache,
            )
        return prov.get_orders(symbol, side, since_s)

    def get_trades(
        self,
        symbol: str,
        since_s: float,
        *,
        provider_name=None,
        authoritative: bool = False,
        publish_cache: bool = False,
    ) -> List[dict]:
        prov = self._provider_explicit_or_routed(symbol, provider_name)
        if authoritative or publish_cache:
            return prov.get_trades(
                symbol,
                since_s,
                authoritative=authoritative,
                publish_cache=publish_cache,
            )
        return prov.get_trades(symbol, since_s)

    def get_trade_orders(
        self,
        order_type: Optional[str],
        symbol: str,
        max_age_seconds: float,
        *,
        provider_name=None,
    ) -> List[dict]:
        """Return recent fill rows from the routed venue provider."""
        return self._provider_explicit_or_routed(symbol, provider_name).get_trade_orders(
            order_type, symbol, max_age_seconds
        )

    def position_cost_basis(self, symbol: str, held_qty: float, since_s: float, *,
                            provider_name=None) -> Optional[float]:
        return self._provider_explicit_or_routed(symbol, provider_name).position_cost_basis(symbol, held_qty, since_s)

    def latest_fill_price(self, symbol: str, side: str, since_s: float, *,
                          provider_name=None, min_notional=None,
                          max_notional=None) -> Optional[float]:
        """Return the newest normalized fill without a Binance-specific API."""
        side = normalize_side(side)
        if not side:
            raise ValueError("side must be BUY or SELL")
        since_s = require_pos_float(since_s, "since_s")

        def _opt_nonneg(val: Any, name: str) -> Optional[float]:
            if val is None:
                return None
            res = safe_nonneg_float(val, default=None)
            if res is None:
                raise ValueError(f"{name} must be finite and non-negative")
            return res

        min_value = _opt_nonneg(min_notional, "min_notional")
        max_value = _opt_nonneg(max_notional, "max_notional")
        if min_value is not None and max_value is not None and min_value > max_value:
            raise ValueError("min_notional cannot exceed max_notional")
        now_ms = time.time() * 1000.0
        cutoff_ms = now_ms - since_s * 1000.0
        candidates = []
        for trade in self.get_trades(
                symbol, since_s, provider_name=provider_name) or []:
            if extract_order_side(trade) != side:
                continue
            timestamp = safe_nonneg_float(extract_timestamp_ms(trade, default=0), default=0.0) or 0.0
            if 0 < timestamp < 10_000_000_000:  # Seconds to milliseconds.
                timestamp *= 1000.0
            price = extract_order_price(trade, default=0.0)
            qty = extract_order_qty(trade, default=0.0)
            if (timestamp < cutoff_ms or timestamp > now_ms + 60_000 or
                    price <= 0 or qty <= 0):
                continue
            notional = safe_pos_float(price * qty, default=None)
            if notional is None:
                continue
            if min_value is not None and notional < min_value:
                continue
            if max_value is not None and notional > max_value:
                continue
            candidates.append((timestamp, price))
        return max(candidates)[1] if candidates else None

    def reconciliation_capabilities(
            self, symbol: str, *, provider_name=None
    ) -> OrderReconciliationCapabilities:
        provider = self._provider_explicit_or_routed(symbol, provider_name)
        return reconciliation_capabilities_of(provider)

    def _require_reconciliation_method(
        self, symbol: str, provider_name, cap_attr: str, method_name: str,
    ):
        provider = self._provider_explicit_or_routed(symbol, provider_name)
        if not getattr(reconciliation_capabilities_of(provider), cap_attr, False):
            raise ProviderError(f"{provider.name}: {method_name} is unsupported")
        method = getattr(provider, method_name, None)
        if not callable(method):
            raise ProviderError(f"{provider.name}: declared {method_name} is missing")
        return provider, method

    def open_orders(self, symbol: str, *, provider_name=None) -> List[dict]:
        _provider, method = self._require_reconciliation_method(
            symbol, provider_name, "list_open_orders", "open_orders"
        )
        return method(symbol)

    def order_by_client_id(self, symbol: str, client_order_id: str, *,
                           provider_name=None):
        """Return a native order found by deterministic client ID, or ``None``.

        This narrow recovery hook lets persistent callers reconcile an ambiguous
        submit without guessing from balances or submitting a duplicate.
        """
        provider, method = self._require_reconciliation_method(
            symbol, provider_name, "lookup_by_client_order_id", "order_by_client_id"
        )
        clean_coid = (
            clean_order_id(client_order_id)
            if isinstance(client_order_id, (str, int)) and not isinstance(client_order_id, bool)
            else None
        )
        if clean_coid is None:
            raise ProviderError(f"{provider.name}: invalid client_order_id")
        return method(symbol, clean_coid)

    @staticmethod
    def _clean_order_id_arg(order_id: Any, provider_name: str) -> str:
        return require_clean_order_id(order_id, provider_name, msg_suffix="invalid order_id")

    def order_status(self, symbol: str, order_id: str, *,
                     provider_name=None) -> OrderStatus:
        """Return venue-neutral status; lookup failures remain fail-closed."""
        provider, method = self._require_reconciliation_method(
            symbol, provider_name, "status_by_order_id", "order_status"
        )
        oid_str = self._clean_order_id_arg(order_id, provider.name)
        try:
            status = method(symbol, oid_str)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                f"{provider.name}: invalid order status for {order_id}: {exc}"
            ) from exc
        if not isinstance(status, OrderStatus):
            raise ProviderError(
                f"{provider.name}: order_status returned {type(status).__name__}")
        return status

    def cancel_order(self, symbol: str, order_id: str, *, provider_name=None) -> None:
        """Cancel through the provider-neutral adapter contract."""
        provider, method = self._require_reconciliation_method(
            symbol, provider_name, "cancel_by_order_id", "cancel_order"
        )
        method(symbol, self._clean_order_id_arg(order_id, provider.name))

    def _validate_open_order_side_and_symbol(
        self, order_type: str, symbol: str, *, provider_name=None
    ) -> str:
        side = normalize_side(order_type)
        if side not in ("BUY", "SELL"):
            raise ValueError(
                f"Invalid order_type '{order_type}'. It must be either 'BUY' or 'SELL'."
            )
        if isinstance(symbol, bool) or not isinstance(symbol, str) or not symbol.strip():
            raise ValueError(f"Invalid symbol '{symbol}'.")
        provider = self._provider_explicit_or_routed(symbol, provider_name)
        validate_open_fn = getattr(provider, "validate_open_order_symbol", None)
        if callable(validate_open_fn):
            validate_open_fn(symbol)
        return side

    def get_open_orders(
        self,
        order_type: str,
        symbol: str,
        *,
        strict: bool = False,
        allowed_owners=None,
        provider_name=None,
    ) -> dict:
        """Return venue-normalized open orders keyed by scalar order ID for ``order_type``."""
        strict_flag = _parse_bool_flag(strict, False)
        side = self._validate_open_order_side_and_symbol(
            order_type, symbol, provider_name=provider_name
        )
        sym_clean = symbol.strip().upper()
        try:
            open_list = self.open_orders(sym_clean, provider_name=provider_name)
            filtered_orders: dict = {}
            accepted_owner_map = None
            for order in (open_list or ()):
                if not isinstance(order, dict):
                    continue
                if extract_order_side(order) != side:
                    continue
                cid = clean_order_id(order.get("clientOrderId")) or ""
                if allowed_owners is not None and not rc.order_matches_owner(cid, allowed_owners):
                    if cid.startswith("OR_"):
                        if accepted_owner_map is None:
                            try:
                                accepted_owner_map = MarketDataProvider.get_accepted_retry_order_owner_map()
                            except Exception:
                                accepted_owner_map = {}
                        inferred = (
                            accepted_owner_map.get(str(order.get("orderId") or ""))
                            or accepted_owner_map.get(cid)
                        )
                        allowed_set = rc._normalize_allowed_owners(allowed_owners)
                        if not inferred or str(inferred).strip().lower() not in allowed_set:
                            continue
                    else:
                        continue
                original_qty = extract_orig_qty(order, default=0.0) or 0.0
                raw_exec_qty = next(
                    (order.get(k) for k in ("executedQty", "z", "vol_exec") if order.get(k) is not None),
                    None,
                )
                executed_qty = safe_nonneg_float(raw_exec_qty, default=0.0)
                remaining_qty = extract_remaining_qty(order, default=max(0.0, original_qty - executed_qty))
                ts_ms = extract_timestamp_ms(order, default=0) or 0
                ts_sec = safe_nonneg_ts(ts_ms, default=0.0)
                raw_time = safe_nonneg_float(order.get("time"), default=None)
                if raw_time is not None:
                    ts_sec = (
                        raw_time / 1000.0
                        if (raw_time < 1_000_000_000.0 or raw_time >= 100_000_000_000.0)
                        else raw_time
                    )
                raw_oid = normalize_scalar_order_id(order.get("orderId"))
                if raw_oid is None:
                    continue
                filtered_orders[raw_oid] = {
                    "price": extract_order_price(order),
                    "quantity": original_qty,
                    "origQty": original_qty,
                    "qty": remaining_qty,
                    "executedQty": executed_qty,
                    "remainingQty": remaining_qty,
                    "timestamp": ts_sec,
                    "time": ts_ms,
                    "clientOrderId": cid,
                }
            return filtered_orders
        except Exception as exc:
            print(f"Error getting open {side} orders: {exc}")
            if strict_flag:
                raise
            return {}

    def _cancel_open_order_and_resolve(
        self, symbol: str, order_id: Any, *, provider_name=None
    ) -> bool:
        try:
            res = self.cancel_order(symbol, order_id, provider_name=provider_name)
            if res is False:
                return False
        except Exception as exc:
            print(f"Error canceling order {order_id}: {exc}")
            return False
        try:
            MarketDataProvider.resolve_accepted_retry_order(symbol, order_id)
        except Exception:
            pass
        return True

    def cancel_open_orders(
        self,
        order_type: str,
        symbol: str,
        allowed_owners=None,
        *,
        provider_name=None,
    ) -> None:
        """Cancel all open ``order_type`` orders for ``symbol`` matching ``allowed_owners``."""
        side = self._validate_open_order_side_and_symbol(
            order_type, symbol, provider_name=provider_name
        )
        try:
            open_map = self.get_open_orders(
                side, symbol, allowed_owners=allowed_owners, provider_name=provider_name
            )
            for order_id in open_map:
                self._cancel_open_order_and_resolve(symbol, order_id, provider_name=provider_name)
        except Exception as exc:
            print(f"Error cancelling orders for {symbol}: {exc}")

    def cancel_expired_orders(
        self,
        order_type: str,
        symbol: str,
        expire_time: float,
        allowed_owners=None,
        *,
        provider_name=None,
    ) -> None:
        """Cancel open ``order_type`` orders for ``symbol`` older than ``expire_time`` seconds."""
        side = self._validate_open_order_side_and_symbol(
            order_type, symbol, provider_name=provider_name
        )
        if (expire_val := safe_pos_float(expire_time, default=None)) is None:
            return
        if not (open_map := self.get_open_orders(side, symbol, allowed_owners=allowed_owners, provider_name=provider_name)):
            return
        current_time = float(time.time())
        for order_id in iter_expired_open_order_ids(open_map, expire_val, current_time_sec=current_time):
            self._cancel_open_order_and_resolve(symbol, order_id, provider_name=provider_name)

    def tracked_order_lifecycle(self, *, provider_name: str, venue=None,
                                missing_confirmations=2,
                                retry_on_lookup_error=False,
                                max_age_seconds=None, audit=None,
                                clock=time.time):
        """Build the persistent lifecycle companion for synchronous ``place``.

        Persistence remains strategy-owned and is supplied to each lifecycle call;
        this facade supplies provider-neutral lookup/status/cancel routing.
        """
        lifecycle_cls = MarketDataProvider.resolve_tracked_order_lifecycle_class()
        return lifecycle_cls(
            self,
            provider_name=provider_name,
            venue=venue,
            missing_confirmations=missing_confirmations,
            retry_on_lookup_error=retry_on_lookup_error,
            max_age_seconds=max_age_seconds,
            audit=audit,
            clock=clock,
        )

    def _regime_service_for(self, strength_threshold=None):
        service = self._regime_service
        if strength_threshold is None:
            return service
        evaluator = MarketRegimeEvaluator(strength_threshold)
        base_eval = getattr(service, "evaluator", None)
        if base_eval is not None and evaluator.strength_threshold == getattr(base_eval, "strength_threshold", None):
            return service
        cache = getattr(self, "_threshold_regime_services", None)
        if not isinstance(cache, OrderedDict):
            cache = OrderedDict()
            self._threshold_regime_services = cache
        key = (
            evaluator.strength_threshold,
            service.cache_ttl_sec,
            service.negative_cache_ttl_sec,
            service.cache_max,
            service.clock,
        )
        cached = cache.get(key)
        if cached is not None:
            cache.move_to_end(key)
            return cached
        created = MarketRegimeService(
            evaluator.strength_threshold,
            cache_ttl_sec=service.cache_ttl_sec,
            negative_cache_ttl_sec=service.negative_cache_ttl_sec,
            cache_max=service.cache_max,
            clock=service.clock,
        )
        lru_dict_put(cache, key, created, 32)
        return created

    def market_regime_context(
        self,
        symbol: str,
        *,
        provider_name=None,
        snapshot=None,
        use_runtime_snapshot=True,
        snapshot_source="snapshot",
        snapshot_max_age_seconds=None,
        ohlc_max_age_seconds=None,
        strength_threshold=None,
        allow_fallback=True,
        now=None,
        trend_duration_seconds=0.0,
    ) -> MarketRegimeContext:
        """Resolve one reusable short-horizon context at the composition root."""
        evaluated_at = time.time() if now is None else require_finite_float(now, "now")
        use_runtime_snapshot = _parse_bool_flag(use_runtime_snapshot, True)
        allow_fallback = _parse_bool_flag(allow_fallback, True)
        if snapshot_max_age_seconds is None:
            snapshot_max_age_seconds = (
                self._regime_service.default_snapshot_max_age_seconds())
        if (
            snapshot is None
            and use_runtime_snapshot
            and self._regime_snapshot_resolver is not None
        ):
            try:
                snapshot = self._regime_snapshot_resolver(
                    symbol, now=evaluated_at)
            except Exception:
                snapshot = None
        resolution = self.market_regime_resolution(
            symbol,
            provider_name=provider_name,
            horizon="short",
            snapshot=snapshot,
            snapshot_source=snapshot_source,
            snapshot_max_age_seconds=snapshot_max_age_seconds,
            ohlc_max_age_seconds=ohlc_max_age_seconds,
            strength_threshold=strength_threshold,
            allow_fallback=allow_fallback,
            now=evaluated_at,
        )
        provider_str = provider_name
        if not provider_str:
            try:
                p_obj = self._provider_explicit_or_routed(symbol, provider_name)
                provider_str = p_obj.name if p_obj is not None else None
            except Exception:
                provider_str = None
        return MarketRegimeContext.from_decision(
            resolution.decision,
            evaluated_at=evaluated_at,
            symbol=symbol,
            provider=provider_str,
            trend_duration_seconds=trend_duration_seconds,
        )

    def market_regime_resolution(
        self,
        symbol: str,
        *,
        provider_name=None,
        horizon="short",
        interval_min=None,
        window_seconds=None,
        snapshot=None,
        snapshot_source="snapshot",
        snapshot_max_age_seconds=None,
        ohlc_max_age_seconds=None,
        strength_threshold=None,
        allow_fallback=True,
        include_alternates=False,
        role=None,
        now=None,
    ) -> MarketRegimeResolution:
        """Return the selected decision and every source examined for its horizon."""
        service = self._regime_service_for(strength_threshold)
        provider = self._provider_explicit_or_routed(symbol, provider_name)
        return service.resolve_with_evidence(
            provider,
            symbol,
            horizon=horizon,
            interval_min=interval_min,
            window_seconds=window_seconds,
            snapshot=snapshot,
            snapshot_source=snapshot_source,
            snapshot_max_age_seconds=snapshot_max_age_seconds,
            ohlc_max_age_seconds=ohlc_max_age_seconds,
            allow_fallback=allow_fallback,
            include_alternates=include_alternates,
            role=role,
            now=now,
        )

    def market_regime(self, symbol: str, *, provider_name=None, horizon="short",
                      interval_min=None, window_seconds=None, snapshot=None,
                      snapshot_source="snapshot", snapshot_max_age_seconds=None,
                      ohlc_max_age_seconds=None, strength_threshold=None,
                      allow_fallback=True) -> MarketRegimeDecision:
        """Return a common regime with explicit horizon and source fallback."""
        return self.market_regime_resolution(
            symbol, provider_name=provider_name, horizon=horizon,
            interval_min=interval_min, window_seconds=window_seconds,
            snapshot=snapshot, snapshot_source=snapshot_source,
            snapshot_max_age_seconds=snapshot_max_age_seconds,
            ohlc_max_age_seconds=ohlc_max_age_seconds,
            strength_threshold=strength_threshold, allow_fallback=allow_fallback,
        ).decision

    def preflight_order(self, symbol: str, side: str, qty: float,
                        price=None, *, market: bool = False,
                        kind: Optional[str] = None, provider_name=None) -> Any:
        """Validate routed venue state before cancellation or submission."""
        provider = self._provider_explicit_or_routed(symbol, provider_name)
        return provider.preflight_order(symbol, side, qty, price, market=market, kind=kind)

    def order_filter_refusal(
        self, symbol: str, side: str, price: float, qty: float, *,
        market: bool = False, enforce_business_minimum: bool = True, provider_name=None,
    ) -> Optional[str]:
        """Return a routed deterministic venue-filter refusal, if any."""
        provider = self._provider_explicit_or_routed(symbol, provider_name)
        return provider.order_filter_refusal(
            symbol, side, price, qty, market=market, enforce_business_minimum=enforce_business_minimum
        )

    def pair_precision(self, symbol: str, provider_name=None) -> Optional[PairPrecision]:
        provider = self._provider_explicit_or_routed(symbol, provider_name)
        return provider.pair_precision(symbol) if hasattr(provider, "pair_precision") else None

    def min_order_qty(self, symbol: str, provider_name=None) -> float:
        provider = self._provider_explicit_or_routed(symbol, provider_name)
        return provider.min_order_qty(symbol) if hasattr(provider, "min_order_qty") else 0.0

    def round_quantity(self, symbol: str, qty: float, provider_name=None) -> float:
        provider = self._provider_explicit_or_routed(symbol, provider_name)
        if hasattr(provider, "round_quantity"):
            return provider.round_quantity(symbol, qty)
        if hasattr(provider, "round_amount"):
            return provider.round_amount(symbol, qty)
        return safe_pos_float(qty, default=0.0)

    round_amount = round_quantity

    def round_price(self, symbol: str, price: float, provider_name=None) -> float:
        provider = self._provider_explicit_or_routed(symbol, provider_name)
        return provider.round_price(symbol, price) if hasattr(provider, "round_price") else safe_pos_float(price, default=0.0)

    def price_tick_size(self, symbol: str, reference_price: Optional[float] = None, provider_name=None) -> float:
        provider = self._provider_explicit_or_routed(symbol, provider_name)
        if hasattr(provider, "price_tick_size"):
            return safe_nonneg_float(provider.price_tick_size(symbol, reference_price=reference_price), default=0.0)
        return 0.0

    def place_order(self, symbol: str, side: str, price: float, qty: float,
                    force: bool = False, **kwargs):
        # Mechanics-only provider dispatch without guards. Real placement must use
        # guarded .place(); this remains for internal and dry-run cases.
        provider_name = kwargs.pop("provider_name", None)
        provider = self._provider_explicit_or_routed(symbol, provider_name)
        if force and "force" not in kwargs:
            kwargs["force"] = force
        return provider.place_order(symbol, side, price, qty, **kwargs)

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
        provider_name=None,
        **kwargs,
    ) -> str:
        """Route strict ``StrategyExecutor`` order submission to the target venue adapter."""
        provider = self._provider_explicit_or_routed(symbol, provider_name)
        return provider.submit_order(
            symbol,
            side,
            qty,
            price=price,
            market=market,
            kind=kind,
            client_order_id=client_order_id,
            **kwargs,
        )

    def place(self, symbol: str, side: str, price: float, qty: Optional[float] = None,
              base: Optional[str] = None, quote: Optional[str] = None, **kwargs):
        """Place through the single guarded proxy using a temporary Instrument.

        Run the complete provider-agnostic policy pipeline with provider hooks.
        This replaces direct legacy smart/safe placement calls. Derive base and
        quote from the symbol when absent.
        """
        inst_cls = MarketDataProvider.resolve_instrument_class()
        explicit_provider_name = kwargs.pop("provider_name", None)
        inst_params = kwargs.pop("params", None)
        provider = self._provider_explicit_or_routed(
            symbol, explicit_provider_name)
        prov_name = provider.name
        if inst_params is None:
            spec = MarketDataProvider.resolve_venue_instrument_spec(prov_name, symbol)
            if spec is not None:
                inst_params = dict(spec.params or {})
                base = base or spec.base or None
                quote = quote or spec.quote or None
            else:
                inst_params = {}
        if base is None or quote is None:
            res_base, res_quote = resolve_assets(symbol, base=base, quote=quote)
            base = base or res_base or None
            default_quote_fn = getattr(provider, "default_quote_asset", None)
            default_quote = (
                default_quote_fn()
                if callable(default_quote_fn)
                else MarketDataProvider.default_quote_asset(provider)
            )
            quote = quote or res_quote or default_quote
        inst = inst_cls(name=symbol, symbol=symbol, provider=prov_name,
                        base=base, quote=quote, params=inst_params or {}, api=self)
        return inst.place(side, price, qty, **kwargs)

    def supports_symbol(self, symbol: str) -> bool:
        if not symbol or isinstance(symbol, bool) or not isinstance(symbol, str):
            return False
        return any(p.supports_symbol(symbol) for p in self._providers)

    def provider_name_for(self, symbol: str) -> str:
        """Return the provider that would serve the symbol for logs and debugging."""
        return self._provider_for(symbol).name

    def provider_by_name(self, name: str) -> Optional[MarketDataProvider]:
        """Return the case-insensitive named provider, or None when absent.

        Instrument uses this for explicit venue routing so identical assets on
        multiple venues are not inferred from the symbol string.
        """
        if hasattr(name, "name") and isinstance(getattr(name, "name", None), str):
            name = name.name
        if isinstance(name, bool) or not isinstance(name, str):
            return None
        key = name.strip().lower()
        found = self._by_name.get(key)
        if found is not None:
            return found
        return self._by_name.get(normalize_venue_name(key))

    @property
    def providers(self) -> List[MarketDataProvider]:
        return list(self._providers)


# Singleton injected when constructors receive api=None. Binance remains first to
# preserve routing for unclaimed symbols. Hyperliquid claims only HYPE and Binance
# explicitly excludes it. Provider construction is cheap and SDKs load lazily;
# import failure falls back cleanly without affecting the fleet.
_configured_providers = []
for _venue_key, (_modname, _clsname) in MarketDataProvider._VENUE_PROVIDER_SPECS.items():
    try:
        _prov_cls = MarketDataProvider.resolve_venue_provider_class(_venue_key)
        if _prov_cls is not None:
            _configured_providers.append(_prov_cls())
    except Exception as _e:  # noqa: BLE001
        print(f"market_api: {_clsname} unavailable ({_e})")

api = MarketApi(
    _configured_providers,
    regime_snapshot_resolver=_runtime_market_regime_snapshot,
)
