"""Binance MarketDataProvider adapter for market data, account reads, and order mechanics."""

import importlib
import math
from typing import Any, List, Optional

import account_cache_health as health
import botcore as cfg
from botcore import (
    _parse_bool_flag,
    bind_module_importer,
    clean_upper_token,
    exact_pos_int,
    require_pos_float,
    resolve_patched_attr,
    safe_nonneg_float,
    safe_pos_float,
)
from cachepaths import (
    cache_symbol_variants,
    canonical_symbol_key,
    normalize_cache_base_asset,
    resolve_binance_lookup_symbol,
    split_symbol_assets,
)
import instrument_registry as sym_reg
from instrument_registry import casefold_get, venue_for_symbol

from .base import (
    MarketDataProvider,
    compute_fills_cost_basis,
    extract_order_id,
    normalize_side,
)
from .order_ohlc_utils import (
    ClosedPriceSeries,
    build_open_order_entry,
    candles_to_historical_points,
    clean_order_id,
    extract_order_fee,
    normalize_order,
    require_int_order_id,
    resolve_market_flag,
)
from .quantity import (
    BinanceFilterError,
    BinanceOrderRules,
    binance_filter_refusal_reason,
    decimal_places,
    fee_cap_quantity,
)
from .strategy_executor import (
    OrderReconciliationCapabilities,
    OrderStatus,
    PairPrecision,
    ProviderError,
    SubmissionRefused,
    candle_interval,
)

# Binance imports reach WebSockets and local keys. Kraken and Hyperliquid must
# remain importable in a clean checkout without Binance secrets or side effects.
# Keep these variables patchable in tests (both here and on providers.market_api).
_bapi = None
_allorders = None

# Binance documents that some zero-fill canceled/expired orders are archived
# after 90 days. This is an upper bound on a NOT_FOUND response's usefulness;
# the retry worker applies the lower operator-configured safety horizon.
_BINANCE_NOT_FOUND_RELIABLE_FOR_SECONDS = 90 * 24 * 60 * 60


def _resolve_or_import_bapi_submodule(attr_name: str, import_path: str):
    patched_mod = resolve_patched_attr(
        attr_name, None, modules=("providers.market_api", "providers.binance_provider")
    )
    if patched_mod is not None:
        return patched_mod
    cur = globals().get(attr_name)
    if cur is None:
        cur = importlib.import_module(import_path)
        globals()[attr_name] = cur
    return cur


def _bind_bapi_getter(fn_name: str, attr_name: str, import_path: str):
    def _default_get():
        return _resolve_or_import_bapi_submodule(attr_name, import_path)

    def _get():
        fn = resolve_patched_attr(fn_name, _default_get, modules=("providers.market_api",))
        if callable(fn) and fn is not _get:
            return fn()
        return _default_get()

    _get.__name__ = _get.__qualname__ = fn_name
    return _default_get, _get


_default_get_bapi, _get_bapi = _bind_bapi_getter("_get_bapi", "_bapi", "binance_api.bapi")
_default_get_allorders, _get_allorders = _bind_bapi_getter("_get_allorders", "_allorders", "binance_api.bapi_allorders")


_get_placeorder = bind_module_importer("binance_api.bapi_placeorder")


def _parse_nonneg_order_field(raw: Any) -> Optional[float]:
    if raw is None or raw == "":
        return 0.0
    return safe_nonneg_float(raw, default=None)


class BinanceProvider(MarketDataProvider):
    """Adapt Binance market data, account reads, and execution mechanics.

    ``get_price_history`` is intentionally unavailable through this adapter; strict
    strategy execution obtains completed OHLC closes through ``ohlc_closes``.
    """

    _USES_SSOT_MARKET_CACHE: bool = False
    _DEFAULT_FEE_SPEC = ("VIP0", 10.0, 10.0, 5.0)
    _USES_SHARED_REGISTRY_SYMBOLS = True
    _HISTORICAL_OHLC_PRIORITY = 1
    _QUOTE_ISOLATED_TELEMETRY_VENUE = False
    _USES_SHARED_TRAILING_MAP = True
    _REQUIRES_CACHE_PERMIT: bool = True
    _DEFERS_SMART_CANCELLATION: bool = True
    _SUPPORTS_BUSINESS_MINIMUM: bool = True
    _INTENT_CLIENT_ID_FORMAT: str = "SD_{token}"
    _OHLC_MARKET_DATA_SPEC = (None, "binance_api.market_data", "fetch_ohlc_candles")
    _RECONCILIATION_CAPABILITIES = OrderReconciliationCapabilities(
        lookup_by_client_order_id=True,
        status_by_order_id=True,
        cancel_by_order_id=True,
        list_open_orders=True,
        not_found_reliable_for_seconds=_BINANCE_NOT_FOUND_RELIABLE_FOR_SECONDS,
    )

    def __init__(self, live_enabled: Optional[bool] = None):
        self._live_enabled = live_enabled

    @property
    def name(self) -> str:
        return "Binance"

    @classmethod
    def _venue_live_orders_enabled(cls) -> bool:
        """Return the authoritative Binance live-order feature gate."""
        return bool(cfg.is_trade_enabled())

    def _fetch_current_price(self, symbol: str) -> Optional[float]:
        return _get_bapi().get_current_price(symbol)

    def supports_symbol(self, symbol: str) -> bool:
        if not symbol or isinstance(symbol, bool) or not isinstance(symbol, str):
            return False
        sym_norm = symbol.strip().upper()
        split = split_symbol_assets(sym_norm)
        if split is None or split[1] != "USDC":
            return False
        return venue_for_symbol(sym_norm) == "binance"

    def validate_open_order_symbol(self, symbol: str) -> None:
        if not self.supports_symbol(symbol):
            sym_reg.validate_symbols(symbol)

    # -- Account data, repackaged through the facade without semantic changes. --
    def free_balance(self, asset: str) -> Optional[float]:
        # Delegate to Binance's direct per-asset balance query. It returns ``0.0``
        # for an absent asset and ``None`` when the API read fails.
        return _get_bapi().get_free_balance(asset)

    @classmethod
    def fetch_remote_trades_with_client(
        cls,
        symbol: str,
        start_time_ms: int = 0,
        client: Any = None,
        *,
        allorders_mod: Any = None,
        strict: bool = True,
    ) -> List[dict]:
        """Paginate raw Binance fills from ``client`` via ``bapi_allorders.paginate_my_trades``."""
        ao = allorders_mod if allorders_mod is not None else _get_allorders()
        raw_client = getattr(client, "client", client) if client is not None else _get_bapi().client
        return list(ao.paginate_my_trades(raw_client, symbol, start_time_ms, limit=1000, strict=strict) or [])

    def get_orders(
        self,
        symbol: str,
        side: Optional[str],
        since_s: float,
        *,
        authoritative: bool = False,
        publish_cache: bool = False,
    ) -> List[dict]:
        # Preserve get_trade_orders side and age filtering, then normalize shape.
        raw = _get_allorders().get_trade_orders(side, symbol, since_s) or []
        return [normalize_order(o) for o in raw]

    def get_trade_orders(
        self,
        order_type: Optional[str],
        symbol: str,
        max_age_seconds: float,
    ) -> List[dict]:
        """Return cached recent fill rows normalized for execution monitoring."""
        return list(
            importlib.import_module("binance_api.bapi_trades").get_trade_orders(
                order_type, symbol, max_age_seconds
            )
            or []
        )

    def fetch_recent_orders(self, symbol: str, limit: int = 1000) -> List[dict]:
        """Return recent raw venue orders for per-owner attribution diagnostics."""
        lim = exact_pos_int(limit, default=1000) or 1000
        return list(_get_bapi().client.get_all_orders(symbol=symbol, limit=lim) or [])

    def position_cost_basis(self, symbol: str, held_qty: float, since_s: float) -> Optional[float]:
        """Use current immutable fills, not order aggregates or a BUY-only average."""
        since_s = require_pos_float(since_s, "since_s")
        if (symbol := clean_upper_token(symbol)) is None:
            return None
        sym_canon = canonical_symbol_key(symbol)
        spec = self.resolve_venue_instrument_spec(self.name, symbol)
        if spec is None:
            return None
        status = health.require_fresh_account_cache()
        self.ensure_ssot_account_cache_readers(status)
        manager = self.get_ssot_cache_manager("BinanceTrade")
        with manager.lock:
            bucket = None
            for cand in [symbol] + cache_symbol_variants(symbol):
                if not cand:
                    continue
                bucket = casefold_get(manager.cache, cand)
                if bucket is not None:
                    break
            raw_rows = list(bucket) if isinstance(bucket, (list, tuple)) else []

        def _valid_binance_fill(row: dict) -> bool:
            row_sym = str(row.get("symbol") or "").strip().upper() if isinstance(row.get("symbol"), str) else ""
            return bool(
                manager._is_valid_item(row)
                and (row_sym == symbol or (sym_canon and canonical_symbol_key(row_sym) == sym_canon))
            )

        return compute_fills_cost_basis(
            raw_rows,
            held_qty,
            since_s,
            base_asset=spec.base,
            quote_asset=spec.quote,
            require_explicit_fee_asset=True,
            validate_row_fn=_valid_binance_fill,
        )

    def open_orders(self, symbol: str) -> List[dict]:
        try:
            raw = _get_bapi().client.get_open_orders(symbol=symbol)
            if raw is None:
                raise ProviderError(f"open_orders({symbol}): received None from Binance client")
            if not isinstance(raw, list):
                raise ProviderError(f"open_orders({symbol}): received invalid payload from Binance client")
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"open_orders({symbol}): {exc}") from exc
        normalized = []
        for order in raw:
            if not isinstance(order, dict):
                raise ProviderError(f"open_orders({symbol}): invalid order entry")
            raw_px, raw_orig, raw_exec = order.get("price"), order.get("origQty"), order.get("executedQty")
            if any(isinstance(v, bool) for v in (raw_px, raw_orig, raw_exec)):
                raise ProviderError(f"open_orders({symbol}): boolean numeric field in order")
            price_f = _parse_nonneg_order_field(raw_px)
            orig_f = _parse_nonneg_order_field(raw_orig)
            exec_f = _parse_nonneg_order_field(raw_exec)
            if None in (price_f, orig_f, exec_f):
                raise ProviderError(f"open_orders({symbol}): invalid or negative numeric field in order")
            try:
                oid_int = require_int_order_id(order.get("orderId"), f"open_orders({symbol})")
            except ProviderError as exc:
                raise ProviderError(f"open_orders({symbol}): missing or invalid orderId in order") from exc
            raw_side, raw_status = order.get("side"), order.get("status")
            if isinstance(raw_side, bool) or isinstance(raw_status, bool):
                raise ProviderError(f"open_orders({symbol}): boolean status/side field in order")
            normalized.append(
                build_open_order_entry(
                    order,
                    order_id=str(oid_int),
                    side=normalize_side(raw_side) or str(raw_side or "").strip().upper(),
                    price=price_f,
                    orig_qty=orig_f,
                    executed_qty=exec_f,
                    client_order_id=clean_order_id(order.get("clientOrderId")),
                    has_explicit_coid=True,
                    status=str(raw_status or "NEW").strip().upper(),
                )
            )
        return normalized

    def order_by_client_id(self, symbol: str, client_order_id: str):
        try:
            return _get_bapi().client.get_order(symbol=symbol, origClientOrderId=client_order_id)
        except Exception as exc:
            # Binance -2013 means missing order; all other errors remain fail closed.
            if getattr(exc, "code", None) == -2013:
                return None
            raise ProviderError(f"order_by_client_id({symbol},{client_order_id}): {exc}") from exc

    def preflight_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price=None,
        *,
        market: bool = False,
        kind: Optional[str] = None,
    ) -> Any:
        _po = _get_placeorder()

        return _po.issue_account_cache_submit_permit(
            symbol,
            side,
            qty,
            price,
            market=_parse_bool_flag(market, False),
            kind=kind,
            api_client=_get_bapi().client,
        )

    def prepare_order_state(self) -> tuple[str, str]:
        """Synchronize Binance account readers before shared policy checks."""
        _po = _get_placeorder()

        status = _po.require_account_cache_for_submit()
        return str(status.order_cache_version), str(status.trade_cache_version)

    def validate_order_state(self, expected_state):
        """Refuse when policy checks raced with a newer account-cache snapshot."""
        current_state = self.prepare_order_state()
        if current_state != tuple(expected_state or ()):
            raise SubmissionRefused("account_cache_snapshot_changed")
        return current_state

    def place_order(
        self,
        symbol: str,
        side: str,
        price: float,
        qty: float,
        force: bool = False,
        cache_permit=None,
        permit_requested_price=None,
        enforce_business_minimum: bool = True,
        **kwargs,
    ):
        _po = _get_placeorder()

        is_market = resolve_market_flag(force=force, **kwargs)
        mechanics_kwargs = {
            "force": is_market,
            "enforce_business_minimum": _parse_bool_flag(enforce_business_minimum, True),
        }
        if "market" in kwargs:
            mechanics_kwargs["market"] = is_market
        for opt_k in ("client_order_id", "_balance_verified", "caller_owns_retry", "known_balance"):
            if kwargs.get(opt_k) is not None:
                mechanics_kwargs[opt_k] = kwargs[opt_k]
        if kwargs.get("_cancel_opposite_requested_price") is not None:
            mechanics_kwargs["cancel_opposite_requested_price"] = kwargs["_cancel_opposite_requested_price"]
        if cache_permit is not None:
            mechanics_kwargs.update(
                {
                    "cache_permit": cache_permit,
                    "permit_requested_price": (
                        price if permit_requested_price is None and not is_market else permit_requested_price
                    ),
                    "kind": kwargs.get("kind") or kwargs.get("motivation"),
                }
            )
        return _po.place_order_mechanics(side, symbol, price, qty, **mechanics_kwargs)

    def adjust_order_price(self, symbol: str, side: str, price: float, cancel_opposite: bool = True) -> float:
        _po = _get_placeorder()

        return _po.adjust_price_and_cancel_opposite(
            side, symbol, price, cancel_opposite=_parse_bool_flag(cancel_opposite, True)
        )

    def cancel_opposite_orders(self, symbol: str, side: str, requested_price: float) -> None:
        _po = _get_placeorder()

        _po.cancel_opposite_orders(side, symbol, requested_price)

    def profit_guard_window_ref(
        self, symbol: str, side: str, safeback_sec, regime_context=None, reference_mode=None, module=None
    ):
        _po = _get_placeorder()

        return super().profit_guard_window_ref(
            symbol,
            side,
            safeback_sec,
            regime_context=regime_context,
            reference_mode=reference_mode,
            module=module,
            prefer_safeback=True,
            default_safeback_sec=_po.PLACE_ORDER_SAFEBACK_SEC,
        )

    def last_opposite_fill(self, symbol: str, order_type: str, since_s: float = 0) -> Optional[float]:
        _po = _get_placeorder()

        ref = self.get_ssot_cache_manager("BinanceTrade").get_last_opposite_fill_price(symbol, order_type)
        if ref is None:
            ref = _po._last_opposite_fill_price_api(symbol, order_type)  # direct API fallback
        return ref

    def policy_cap_quantity(
        self, symbol: str, side: str, price: float, qty: float, available_qty: float, **kwargs
    ) -> float:
        _po = _get_placeorder()

        is_auto = qty is None or (not isinstance(qty, bool) and isinstance(qty, (int, float)) and qty == math.inf)
        return _po.apply_weight_limit(symbol, side, price, None if is_auto else qty, available_qty)

    def fee_cap_quantity(self, symbol: str, side: str, price: float, available_qty: float) -> float:
        _po = _get_placeorder()

        return fee_cap_quantity(available_qty, _po.PLACE_ORDER_FEE_PCT)

    # -- StrategyExecutor contract. --------------------------------------------
    def pair_precision(self, symbol: str):
        try:
            info = _get_bapi().client.get_symbol_info(symbol)
            rules = BinanceOrderRules.from_symbol_info(info)
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"pair_precision({symbol}): {exc}") from exc
        return PairPrecision(
            price_decimals=decimal_places(rules.tick_size),
            volume_decimals=decimal_places(rules.lot_step),
            order_min=safe_nonneg_float(rules.lot_min, default=0.0) or 0.0,
            base_asset=rules.base_asset,
        )

    def order_filter_refusal(
        self,
        symbol: str,
        side: str,
        price: float,
        qty: float,
        *,
        market: bool = False,
        enforce_business_minimum: bool = True,
    ) -> Optional[str]:
        """Validate the candidate with the authoritative Binance filter parser."""
        _po = _get_placeorder()

        is_market = _parse_bool_flag(market, False)
        enforce_min = _parse_bool_flag(enforce_business_minimum, True)
        try:
            info = _get_bapi().client.get_symbol_info(symbol)
            rules = BinanceOrderRules.from_symbol_info(info)
            current_price = price if is_market else _get_bapi().get_current_price(symbol)
            candidate_price = _po.order_candidate_price(side, price, current_price, market=is_market)
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"order rules unavailable for {symbol}: {exc}") from exc
        try:
            rules.normalize(
                quantity=qty,
                price=candidate_price,
                market=is_market,
                reference_price=price if is_market else None,
                business_min_notional=_po.PLACE_ORDER_MIN_NOTIONAL if enforce_min else 0,
            )
        except BinanceFilterError as exc:
            return binance_filter_refusal_reason(exc)
        return None

    @classmethod
    def public_orderbook_depth_urls(cls, symbol: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        canon_sym = canonical_symbol_key(symbol)
        if canon_sym.endswith("EUR"):
            base_asset = normalize_cache_base_asset(canon_sym)
            bin_eur_sym = f"{base_asset}EUR" if base_asset else canon_sym
            return (f"https://api.binance.com/api/v3/depth?symbol={bin_eur_sym}&limit=500",), ()
        lookup_sym = resolve_binance_lookup_symbol(symbol)
        usdc_sym = f"{normalize_cache_base_asset(canon_sym)}USDC" if canon_sym.endswith(("USD", "USDC")) else ""
        spot_urls = (
            (
                f"https://api.binance.com/api/v3/depth?symbol={usdc_sym}&limit=500",
                f"https://api.binance.com/api/v3/depth?symbol={lookup_sym}&limit=500",
            )
            if (usdc_sym and usdc_sym != lookup_sym)
            else (f"https://api.binance.com/api/v3/depth?symbol={lookup_sym}&limit=500",)
        )
        return spot_urls, (f"https://fapi.binance.com/fapi/v1/depth?symbol={lookup_sym}&limit=500",)

    def fetch_historical_ohlc_points(
        self,
        symbol: str,
        days: int,
        interval: str = "1h",
    ) -> list[tuple[float, float]]:
        return candles_to_historical_points(
            self.fetch_ohlc_candles(symbol, interval=interval, lookback_days=days, drop_incomplete=False)
        )

    def ohlc_series(self, symbol: str, interval_min: int) -> ClosedPriceSeries:
        iv = candle_interval(interval_min)
        try:
            klines = _get_bapi().client.get_klines(symbol=symbol, interval=iv, limit=91)
        except Exception as exc:
            raise ProviderError(f"ohlc_series({symbol}): {exc}") from exc
        return ClosedPriceSeries.from_raw_candles(
            klines,
            interval_min,
            lambda item: item[4] if isinstance(item, (list, tuple)) and len(item) >= 5 else None,
            lambda item: item[6] if isinstance(item, (list, tuple)) and len(item) > 6 else None,
            lambda item: item[5] if isinstance(item, (list, tuple)) and len(item) > 5 else None,
        )

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
        cache_permit=None,
    ) -> str:
        order_type = normalize_side(side)
        if not order_type:
            raise SubmissionRefused(f"invalid_side:{side!r}")
        is_market = _parse_bool_flag(market, False)
        permit_requested_price = price
        try:
            client = _get_bapi().client
            rules = BinanceOrderRules.from_symbol_info(client.get_symbol_info(symbol))
            reference_price = _get_bapi().get_current_price(symbol)
            try:
                normalized_qty, normalized_price = rules.normalize(
                    quantity=qty,
                    price=price,
                    market=bool(is_market or price is None),
                    reference_price=reference_price,
                )
            except BinanceFilterError as exc:
                raise SubmissionRefused(binance_filter_refusal_reason(exc)) from exc
            qty = safe_pos_float(normalized_qty, default=0.0) or 0.0
            price = None if normalized_price is None else (safe_pos_float(normalized_price, default=0.0) or 0.0)
            _po = _get_placeorder()

            if is_market or price is None:
                res = _po._submit_binance_order(
                    order_type,
                    symbol,
                    qty,
                    market=True,
                    client_order_id=client_order_id,
                    api_client=client,
                    cache_permit=cache_permit,
                    permit_requested_price=permit_requested_price,
                    kind=kind,
                )
            else:
                res = self.place_order(
                    symbol,
                    order_type,
                    price,
                    qty,
                    force=False,
                    client_order_id=client_order_id,
                    cache_permit=cache_permit,
                    permit_requested_price=permit_requested_price,
                    kind=kind,
                )
        except SubmissionRefused:
            raise
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"submit_order({symbol}): {e}") from e
        return self._require_submitted_order_id(
            symbol,
            extract_order_id(res),
            f"submit_order({symbol}): a response without an orderId ({res})",
            invalidate_fills_cache=False,
        )

    def _order_fee_quote(self, symbol: str, order_id: int) -> float:
        """Return cumulative commission converted into the pair's quote currency.

        Binance omits fees from get_order. Without this query the generic engine
        would systematically overestimate P&L and diverge from live results.
        """
        client = _get_bapi().client
        info = client.get_symbol_info(symbol) or {}
        base = str(info.get("baseAsset") or "")
        quote = str(info.get("quoteAsset") or "")
        trades = client.get_my_trades(symbol=symbol, orderId=order_id) or []
        total = 0.0
        for trade in trades:
            if not isinstance(trade, dict):
                continue
            try:
                if require_int_order_id(trade.get("orderId"), "_order_fee_quote") != order_id:
                    continue
            except ProviderError:
                continue
            fee_info = extract_order_fee(trade, base_asset=base, quote_asset=quote)
            if fee_info is None or fee_info["fee"] <= 0.0:
                continue
            if fee_info["fee_in_quote"] is not None:
                total += fee_info["fee_in_quote"]
            else:
                asset = fee_info["fee_asset"]
                conv_f = safe_pos_float(_get_bapi().get_current_price(f"{asset}{quote}"), default=0.0)
                if conv_f <= 0.0:
                    raise ProviderError(f"order_status({order_id}): cannot convert the fee {asset}->{quote}")
                total += fee_info["fee"] * conv_f
        return total

    def order_status(self, symbol: str, order_id: str):
        oid = require_int_order_id(order_id, f"order_status({order_id})")

        try:
            o = _get_bapi().client.get_order(symbol=symbol, orderId=oid)
        except ProviderError:
            raise
        except Exception as e:  # noqa: BLE001
            err_code = getattr(e, "code", None)
            err_text = str(e).lower()
            if err_code == -2026 or "-2026" in str(e) or ("order was canceled" in err_text and "archived" in err_text):
                return OrderStatus(
                    status="canceled",
                    filled_qty=0.0,
                    cost=0.0,
                    fee=0.0,
                    venue_status="ARCHIVED_CANCELED",
                )
            raise ProviderError(f"order_status({order_id}): {e}") from e

        if not isinstance(o, dict):
            raise ProviderError(f"order_status({order_id}): order payload is invalid")
        executed_qty = _parse_nonneg_order_field(o.get("executedQty"))
        cost_val = _parse_nonneg_order_field(o.get("cummulativeQuoteQty"))
        orig_val = _parse_nonneg_order_field(o.get("origQty"))
        if None in (executed_qty, cost_val, orig_val):
            raise ProviderError(f"order_status({order_id}): invalid fill values")
        fee = self._order_fee_quote(symbol, oid) if executed_qty > 0 else 0.0
        st_map = {
            "FILLED": "closed",
            "CANCELED": "canceled",
            "EXPIRED": "expired",
            "EXPIRED_IN_MATCH": "expired",
            "REJECTED": "canceled",
            "NEW": "open",
            "PARTIALLY_FILLED": "open",
        }
        raw_venue_status = o.get("status")
        if isinstance(raw_venue_status, bool):
            raise ProviderError(f"order_status({order_id}): invalid order status")
        venue_status = str(raw_venue_status or "").strip().upper()
        return OrderStatus(
            status=st_map.get(venue_status, "open"),
            filled_qty=executed_qty,
            cost=cost_val,
            fee=fee,
            venue_status=venue_status,
            orig_qty=orig_val,
        )

    def cancel_order(self, symbol: str, order_id: str) -> None:
        oid = require_int_order_id(order_id, f"cancel_order({order_id})")
        self._execute_confirmed_cancel(order_id, lambda: _get_bapi().cancel_order(symbol, oid))

    resolve_default_api_client = staticmethod(bind_module_importer("binance_api.bapi"))
    resolve_allorders_module = staticmethod(bind_module_importer("binance_api.bapi_allorders"))
    resolve_ws_bridge_module = staticmethod(bind_module_importer("binance_api.bapi_ws_bridge"))

    @classmethod
    def get_default_price_ws_manager(cls):
        """Return or initialize the shared Binance Combined Stream price WebSocket manager."""
        return importlib.import_module("binance_api.bapi_ws").get_ws_manager()

    @classmethod
    def request_stop_price_ws_manager(cls) -> None:
        """Signal the Binance price WebSocket manager to stop without blocking."""
        try:
            bws_mgr = getattr(importlib.import_module("binance_api.bapi_ws"), "bapi_ws_manager", None)
        except Exception:
            bws_mgr = None
        if bws_mgr is not None and hasattr(bws_mgr, "request_stop"):
            try:
                bws_mgr.request_stop()
            except Exception:
                pass

    @classmethod
    def stop_price_ws_manager(cls, timeout: float) -> None:
        """Stop the Binance price WebSocket manager and periodic clock resync thread."""
        try:
            bws_mgr = getattr(importlib.import_module("binance_api.bapi_ws"), "bapi_ws_manager", None)
        except Exception:
            bws_mgr = None
        if bws_mgr is not None and hasattr(bws_mgr, "stop"):
            try:
                bws_mgr.stop(timeout=timeout)
            except Exception:
                pass
        try:
            importlib.import_module("binance_api.bapi_client").stop_periodic_resync()
        except Exception:
            pass

