# market_api.py
"""Multi-venue facade for market data, account reads, and order mechanics.

Symbol routing selects the first provider that claims a symbol and falls back to
the first configured provider, Binance. ``Instrument`` uses the name registry for
explicit venue routing when a symbol alone is ambiguous.

``place_order`` dispatches directly to adapter mechanics and live-order gates.
``place`` constructs an ``Instrument`` and runs the shared policy pipeline before
those mechanics. Account WebSocket streams remain outside this facade.

Imports of Binance modules are lazy because ``cacheManager`` imports this module;
eagerly importing modules that lead back to ``cacheManager`` would create a cycle.
"""
import math
import time
from typing import Any, Callable, List, Optional

from .base import (
    MarketDataProvider,
    _normalize_order,
    extract_order_fee,
    extract_order_price,
    extract_order_qty,
    extract_order_side,
    extract_timestamp_ms,
)
from .binance_filters import (
    BinanceFilterError,
    BinanceOrderRules,
    binance_filter_refusal_reason,
    decimal_places,
)
from .strategy_executor import (
    OrderReconciliationCapabilities,
    OrderStatus,
    PairPrecision,
    ProviderError,
    SubmissionRefused,
    candle_interval,
    extract_order_id,
    reconciliation_capabilities_of,
)
from market_regime import (
    ClosedPriceSeries,
    CompositeMarketRegimeDecision,
    MarketRegimeContext,
    MarketRegimeDecision,
    MarketRegimeBundle,
    MarketRegimeResolution,
    MarketRegimeService,
)


# Binance imports reach WebSockets and local keys. Kraken and Hyperliquid must
# remain importable in a clean checkout without Binance secrets or side effects.
# Keep these variables patchable in tests.
_bapi = None
_allorders = None

# Binance documents that some zero-fill canceled/expired orders are archived
# after 90 days. This is an upper bound on a NOT_FOUND response's usefulness;
# the retry worker applies the lower operator-configured safety horizon.
_BINANCE_NOT_FOUND_RELIABLE_FOR_SECONDS = 90 * 24 * 60 * 60


def _get_bapi():
    global _bapi
    if _bapi is None:
        from binance_api import bapi
        _bapi = bapi
    return _bapi


def _get_allorders():
    global _allorders
    if _allorders is None:
        from binance_api import bapi_allorders
        _allorders = bapi_allorders
    return _allorders


def _runtime_market_regime_snapshot(symbol: str, now=None):
    """Read the optional short-trend snapshot at the application boundary."""
    try:
        import cacheManager as cm
        mgr = cm.get_short_trend_manager()
        snap_fn = getattr(mgr, "get_fresh_snapshot", None)
        return snap_fn(symbol, now=now) if callable(snap_fn) else None
    except Exception:
        return None


_INVALID_API_ID_STRINGS = {"", "0", "0.0", "-0", "+0", "nan", "inf", "-inf", "+inf", "none", "null"}


def _parse_bool_flag(val: Any, default: bool = False) -> bool:
    if isinstance(val, bool):
        return val
    if val is None or isinstance(val, (dict, list, tuple, set)):
        return default
    if isinstance(val, (int, float)):
        if isinstance(val, float) and not math.isfinite(val):
            return default
        if val == 1:
            return True
        if val == 0:
            return False
        return default
    s = str(val).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_binance_order_id(order_id: Any, context: str) -> int:
    if order_id is None or isinstance(order_id, (bool, dict, list, tuple, set)):
        raise ProviderError(f"{context}: invalid order id")
    if isinstance(order_id, float):
        if not math.isfinite(order_id) or not order_id.is_integer() or order_id <= 0:
            raise ProviderError(f"{context}: invalid order id")
        return int(order_id)
    if isinstance(order_id, int):
        if order_id <= 0:
            raise ProviderError(f"{context}: invalid order id")
        return order_id
    s = str(order_id).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    try:
        oid = int(s)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ProviderError(f"{context}: invalid order id") from exc
    if oid <= 0:
        raise ProviderError(f"{context}: invalid order id")
    return oid


def _step_decimals(step: str) -> int:
    """Return significant decimals in a Binance stepSize/tickSize."""
    if step is None or isinstance(step, bool):
        return 0
    try:
        from decimal import Decimal
        d = Decimal(str(step))
        if not d.is_finite() or d <= 0:
            return 0
        return decimal_places(d)
    except Exception:
        s = str(step).rstrip("0")
        return len(s.split(".")[1]) if "." in s and s.split(".", 1)[1] else 0


class BinanceProvider(MarketDataProvider):
    """Adapt Binance market data, account reads, and execution mechanics.

    ``get_price_history`` is intentionally unavailable through this adapter; strict
    strategy execution obtains completed OHLC closes through ``ohlc_closes``.
    """

    @property
    def name(self) -> str:
        return "Binance"

    def execution_enabled(self) -> bool:
        """Return the authoritative Binance live-order feature gate."""
        import config as cfg
        return bool(cfg.is_trade_enabled())

    def reconciliation_capabilities(self) -> OrderReconciliationCapabilities:
        return OrderReconciliationCapabilities(
            lookup_by_client_order_id=True,
            status_by_order_id=True,
            cancel_by_order_id=True,
            list_open_orders=True,
            not_found_reliable_for_seconds=(
                _BINANCE_NOT_FOUND_RELIABLE_FOR_SECONDS),
        )

    def get_current_price(self, symbol: str) -> Optional[float]:
        return _get_bapi().get_current_price(symbol)

    def get_price_history(self, symbol: str, lookback_h: float) -> Optional[List]:
        return None

    def supports_symbol(self, symbol: str) -> bool:
        # The operational Binance account uses USDC pairs exclusively, preserving
        # the facade's Binance default. Hyperliquid owns HYPE spot symbols, so
        # exclude them before Binance's broad USDC claim can capture HYPEUSDC.
        if not symbol or isinstance(symbol, bool) or not isinstance(symbol, str):
            return False
        sym_norm = symbol.strip().upper()
        if sym_norm.startswith("HYPE"):
            return False
        return sym_norm.endswith("USDC")

    # -- Account data, repackaged through the facade without semantic changes. --
    def free_balance(self, asset: str) -> Optional[float]:
        # Delegate to Binance's direct per-asset balance query. It returns ``0.0``
        # for an absent asset and ``None`` when the API read fails.
        return _get_bapi().get_free_balance(asset)

    def get_orders(self, symbol: str, side: Optional[str], since_s: float) -> List[dict]:
        # Preserve get_trade_orders side and age filtering, then normalize shape.
        raw = _get_allorders().get_trade_orders(side, symbol, since_s) or []
        return [_normalize_order(o) for o in raw]

    def position_cost_basis(self, symbol: str, held_qty: float, since_s: float) -> Optional[float]:
        """Use current immutable fills, not order aggregates or a BUY-only average."""
        import binance_cache_health as health
        import cacheManager as cm
        from instrument_registry import select_instruments
        from .quantity import remaining_average_cost

        if isinstance(since_s, bool) or since_s is None:
            raise ValueError("since_s must be finite and positive")
        try:
            since_s = float(since_s)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("since_s must be finite and positive") from exc
        if not math.isfinite(since_s) or since_s <= 0:
            raise ValueError("since_s must be finite and positive")
        if isinstance(symbol, bool) or not isinstance(symbol, str) or not symbol.strip():
            return None
        symbol = symbol.strip().upper()
        spec = next(
            (item for item in select_instruments("binance").values() if item.symbol == symbol),
            None,
        )
        if spec is None:
            return None
        status = health.require_fresh_account_cache()
        cm.ensure_account_cache_readers(status)
        manager = cm.get_cache_manager("Trade", start_sync=False)
        with manager.lock:
            rows = [dict(row) for row in manager.cache.get(symbol, [])]
        now_ms = time.time() * 1000
        cutoff = now_ms - since_s * 1000
        normalized = []
        for row in rows:
            ts_ms = extract_timestamp_ms(row, default=0) or 0
            if (not manager._is_valid_item(row) or row["symbol"] != symbol
                    or ts_ms > now_ms):
                return None
            if ts_ms < cutoff:
                continue
            # Missing commission metadata is unknown history, not an invented zero fee.
            px = extract_order_price(row)
            fee_info = extract_order_fee(
                row,
                base_asset=spec.base,
                quote_asset=spec.quote,
                fill_price=px,
                require_explicit_asset=True,
            )
            if fee_info is None:
                return None
            normalized.append(dict(
                id=row["id"], timestamp=ts_ms, qty=extract_order_qty(row), price=px,
                side=extract_order_side(row),
                base_fee=fee_info["base_fee"],
                quote_fee=fee_info["quote_fee"]))
        def _trade_sort_key(item: dict) -> tuple[int, int, str]:
            raw_id = item.get("id")
            if not isinstance(raw_id, bool):
                try:
                    id_int = int(raw_id)
                    return (int(item["timestamp"]), 0, f"{id_int:020d}")
                except (TypeError, ValueError, OverflowError):
                    pass
            return (int(item["timestamp"]), 1, str(raw_id or ""))

        normalized.sort(key=_trade_sort_key)
        return remaining_average_cost(normalized, held_qty)

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
            raw_px = order.get("price")
            raw_orig = order.get("origQty")
            raw_exec = order.get("executedQty")
            if any(isinstance(v, bool) for v in (raw_px, raw_orig, raw_exec)):
                raise ProviderError(f"open_orders({symbol}): boolean numeric field in order")
            try:
                price_f = float(raw_px or 0.0)
                orig_f = float(raw_orig or 0.0)
                exec_f = float(raw_exec or 0.0)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ProviderError(f"open_orders({symbol}): invalid numeric field in order") from exc
            if not all(math.isfinite(v) and v >= 0.0 for v in (price_f, orig_f, exec_f)):
                raise ProviderError(f"open_orders({symbol}): non-finite or negative field in order")
            raw_oid = order.get("orderId")
            try:
                oid_int = _parse_binance_order_id(raw_oid, f"open_orders({symbol})")
            except ProviderError as exc:
                raise ProviderError(f"open_orders({symbol}): missing or invalid orderId in order") from exc
            raw_side = order.get("side")
            raw_status = order.get("status")
            if isinstance(raw_side, bool) or isinstance(raw_status, bool):
                raise ProviderError(f"open_orders({symbol}): boolean status/side field in order")
            raw_coid = order.get("clientOrderId")
            clean_coid = (
                str(raw_coid).strip()
                if raw_coid is not None
                and not isinstance(raw_coid, (bool, dict, list, tuple, set))
                and str(raw_coid).strip().lower() not in _INVALID_API_ID_STRINGS
                else None
            )
            normalized.append({
                "orderId": str(oid_int),
                "clientOrderId": clean_coid,
                "side": str(raw_side or "").upper(),
                "price": price_f,
                "origQty": orig_f,
                "executedQty": exec_f,
                "status": str(raw_status or "NEW"),
            })
        return normalized

    def order_by_client_id(self, symbol: str, client_order_id: str):
        try:
            return _get_bapi().client.get_order(
                symbol=symbol, origClientOrderId=client_order_id)
        except Exception as exc:
            # Binance -2013 means missing order; all other errors remain fail closed.
            if getattr(exc, "code", None) == -2013:
                return None
            raise ProviderError(
                f"order_by_client_id({symbol},{client_order_id}): {exc}") from exc

    def preflight_order(self, symbol: str, side: str, qty: float,
                        price=None, *, market: bool = False,
                        kind: Optional[str] = None) -> Any:
        from binance_api import bapi_placeorder as _po
        return _po.issue_account_cache_submit_permit(
            symbol, side, qty, price, market=_parse_bool_flag(market, False), kind=kind,
            api_client=_get_bapi().client)

    @staticmethod
    def _account_cache_state(status) -> tuple[str, str]:
        return (
            str(status.order_cache_version),
            str(status.trade_cache_version),
        )

    def prepare_order_state(self):
        """Synchronize Binance account readers before shared policy checks."""
        from binance_api import bapi_placeorder as _po
        status = _po.require_account_cache_for_submit()
        return self._account_cache_state(status)

    def validate_order_state(self, expected_state):
        """Refuse when policy checks raced with a newer account-cache snapshot."""
        from binance_api import bapi_placeorder as _po
        status = _po.require_account_cache_for_submit()
        current_state = self._account_cache_state(status)
        if current_state != tuple(expected_state or ()):
            raise SubmissionRefused("account_cache_snapshot_changed")
        return current_state

    def place_order(self, symbol: str, side: str, price: float, qty: float,
                    force: bool = False, cache_permit=None,
                    permit_requested_price=None,
                    enforce_business_minimum: bool = True, **kwargs):
        # Mechanics only: fee/balance, minimum notional, and dispatch. Instrument.place
        # applies daily, profit, quantity, trend, cooldown, and logging policies via
        # provider-neutral hooks. force and market both specify market order execution.
        from binance_api import bapi_placeorder as _po
        is_market = (
            _parse_bool_flag(force, False)
            or _parse_bool_flag(kwargs.get("market", False), False)
            or _parse_bool_flag(kwargs.get("force", False), False)
        )
        mechanics_kwargs = {
            "force": is_market,
            "enforce_business_minimum": _parse_bool_flag(enforce_business_minimum, True),
        }
        if "market" in kwargs:
            mechanics_kwargs["market"] = is_market
        if kwargs.get("client_order_id") is not None:
            mechanics_kwargs["client_order_id"] = kwargs["client_order_id"]
        if kwargs.get("_balance_verified") is not None:
            mechanics_kwargs["_balance_verified"] = kwargs["_balance_verified"]
        if kwargs.get("caller_owns_retry") is not None:
            mechanics_kwargs["caller_owns_retry"] = kwargs["caller_owns_retry"]
        if kwargs.get("known_balance") is not None:
            mechanics_kwargs["known_balance"] = kwargs["known_balance"]
        cancel_requested_price = kwargs.get(
            "_cancel_opposite_requested_price")
        if cancel_requested_price is not None:
            mechanics_kwargs["cancel_opposite_requested_price"] = (
                cancel_requested_price)
        if cache_permit is not None:
            mechanics_kwargs["cache_permit"] = cache_permit
            mechanics_kwargs["permit_requested_price"] = (
                price if permit_requested_price is None and not is_market
                else permit_requested_price)
            mechanics_kwargs["kind"] = (
                kwargs.get("kind") or kwargs.get("motivation"))
        return _po.place_order_mechanics(
            side, symbol, price, qty, **mechanics_kwargs)

    def adjust_order_price(self, symbol: str, side: str, price: float, cancel_opposite: bool = True) -> float:
        from binance_api import bapi_placeorder as _po
        return _po.adjust_price_and_cancel_opposite(side, symbol, price, cancel_opposite=cancel_opposite)

    def cancel_opposite_orders(self, symbol: str, side: str,
                               requested_price: float) -> None:
        from binance_api import bapi_placeorder as _po
        _po.cancel_opposite_orders(side, symbol, requested_price)

    def profit_guard_window_ref(self, symbol: str, side: str, safeback_sec, regime_context=None, reference_mode=None, module=None):
        # Use the Order-cache safeback window as tier-one reference. When dynamic windowing
        # is active on BUY and safeback_sec was not explicitly passed by caller, use dynamic window.
        import order_guard
        from binance_api import bapi_placeorder as _po
        sb = None
        if safeback_sec is not None and not isinstance(safeback_sec, bool):
            try:
                sb_f = float(safeback_sec)
                if math.isfinite(sb_f) and sb_f > 0:
                    sb = sb_f
            except (TypeError, ValueError, OverflowError):
                sb = None
        if sb is None:
            win_s = order_guard.get_window_cfg(
                "binance",
                symbol=symbol,
                order_type=side,
                regime_context=regime_context,
                module=module,
                reference_mode=reference_mode,
            )
            win_f = None
            if win_s is not None and not isinstance(win_s, bool):
                try:
                    cand = float(win_s)
                    if math.isfinite(cand) and cand > 0:
                        win_f = cand
                except (TypeError, ValueError, OverflowError):
                    win_f = None
            sb = win_f if win_f is not None else _po.PLACE_ORDER_SAFEBACK_SEC
        wr_kwargs = {}
        if reference_mode is not None:
            wr_kwargs["reference_mode"] = reference_mode
        if module is not None:
            wr_kwargs["module"] = module
        return order_guard.window_reference(self, symbol, side, sb, **wr_kwargs)

    def last_opposite_fill(self, symbol: str, order_type: str, since_s: float = 0) -> Optional[float]:
        # Persistent Binance source without a window: fill cache followed by direct
        # get_my_trades. This matches legacy guard tiers two and three. Lazy import
        # avoids a bapi_placeorder/cacheManager/market_api cycle.
        from binance_api import bapi_placeorder as _po
        ref = _po._last_opposite_fill_price(symbol, order_type)        # cache fills (via WS)
        if ref is None:
            ref = _po._last_opposite_fill_price_api(symbol, order_type)  # fallback API direct
        return ref

    def policy_cap_quantity(self, symbol: str, side: str, price: float,
                            qty: float, available_qty: float, **kwargs) -> float:
        from binance_api import bapi_placeorder as _po
        is_auto = False
        if qty is None:
            is_auto = True
        elif not isinstance(qty, bool):
            try:
                is_auto = math.isinf(float(qty)) and float(qty) > 0
            except (TypeError, ValueError, OverflowError):
                is_auto = False
        return _po.apply_weight_limit(
            symbol, side, price,
            None if is_auto else qty,
            available_qty)

    def fee_cap_quantity(self, symbol: str, side: str, price: float,
                         available_qty: float) -> float:
        from binance_api import bapi_placeorder as _po
        from providers.quantity import fee_cap_quantity
        return fee_cap_quantity(available_qty, _po.PLACE_ORDER_FEE_PCT)

    def guards_internally(self) -> bool:
        # False because Binance now uses Instrument.place's shared policy pipeline.
        # Provider hooks preserve its richer mechanics, while mechanics-only
        # place_order prevents duplicated guards.
        return False

    def requires_cache_permit(self) -> bool:
        return True

    def defers_smart_cancellation(self) -> bool:
        return True

    def supports_business_minimum(self) -> bool:
        return True

    # -- StrategyExecutor contract. --------------------------------------------
    # get_current_price and free_balance already satisfy it.
    def pair_precision(self, symbol: str):
        try:
            info = _get_bapi().client.get_symbol_info(symbol)
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"pair_precision({symbol}): {e}") from e
        try:
            rules = BinanceOrderRules.from_symbol_info(info)
        except BinanceFilterError as exc:
            raise ProviderError(f"pair_precision({symbol}): {exc}") from exc
        return PairPrecision(
            price_decimals=decimal_places(rules.tick_size),
            volume_decimals=decimal_places(rules.lot_step),
            order_min=float(rules.lot_min), base_asset=rules.base_asset,
        )

    def order_filter_refusal(self, symbol: str, side: str, price: float,
                             qty: float, *, market: bool = False,
                             enforce_business_minimum: bool = True
                             ) -> Optional[str]:
        """Validate the candidate with the authoritative Binance filter parser."""
        from binance_api import bapi_placeorder as _po
        is_market = _parse_bool_flag(market, False)
        enforce_min = _parse_bool_flag(enforce_business_minimum, True)
        try:
            info = _get_bapi().client.get_symbol_info(symbol)
            rules = BinanceOrderRules.from_symbol_info(info)
            current_price = price if is_market else _get_bapi().get_current_price(symbol)
            candidate_price = _po.order_candidate_price(
                side, price, current_price, market=is_market)
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"order rules unavailable for {symbol}: {exc}") from exc
        try:
            rules.normalize(
                quantity=qty,
                price=candidate_price,
                market=is_market,
                reference_price=price if is_market else None,
                business_min_notional=(
                    _po.PLACE_ORDER_MIN_NOTIONAL
                    if enforce_min else 0),
            )
        except BinanceFilterError as exc:
            return binance_filter_refusal_reason(exc)
        return None

    def ohlc_series(self, symbol: str, interval_min: int) -> ClosedPriceSeries:
        iv = candle_interval(interval_min)
        try:
            klines = _get_bapi().client.get_klines(
                symbol=symbol, interval=iv, limit=91)
        except Exception as exc:
            raise ProviderError(f"ohlc_series({symbol}): {exc}") from exc
        completed = list(klines or ())[:-1]
        valid = []
        for item in completed:
            if not isinstance(item, (list, tuple)) or len(item) < 5 or isinstance(item[4], bool):
                continue
            try:
                c_f = float(item[4])
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(c_f) and c_f > 0:
                valid.append((item, c_f))
        closes = tuple(c_f for _item, c_f in valid)
        ts_list = []
        ts_ok = bool(valid) and all(len(item) > 6 and not isinstance(item[6], bool) for item, _ in valid)
        if ts_ok:
            for item, _ in valid:
                try:
                    t_f = float(item[6]) / 1000.0
                    if not math.isfinite(t_f) or t_f <= 0:
                        ts_ok = False
                        break
                    ts_list.append(t_f)
                except (TypeError, ValueError, OverflowError):
                    ts_ok = False
                    break
        timestamps = tuple(ts_list) if ts_ok else ()
        observed_at = timestamps[-1] if timestamps else None
        return ClosedPriceSeries(
            closes,
            int(interval_min),
            observed_at,
            timestamps,
        )

    def submit_order(self, symbol: str, side: str, qty: float,
                     price: Optional[float] = None, *, market: bool = False,
                     kind: Optional[str] = None,
                     client_order_id: Optional[str] = None,
                     cache_permit=None) -> str:
        s_norm = str(side or "").strip().upper() if not isinstance(side, bool) else ""
        if s_norm in {"BUY", "B"}:
            order_type = "BUY"
        elif s_norm in {"SELL", "S"}:
            order_type = "SELL"
        else:
            raise SubmissionRefused(f"invalid_side:{side!r}")
        is_market = _parse_bool_flag(market, False)
        permit_requested_price = price
        try:
            client = _get_bapi().client
            rules = BinanceOrderRules.from_symbol_info(client.get_symbol_info(symbol))
            reference_price = _get_bapi().get_current_price(symbol)
            try:
                normalized_qty, normalized_price = rules.normalize(
                    quantity=qty, price=price,
                    market=bool(is_market or price is None),
                    reference_price=reference_price,
                )
            except BinanceFilterError as exc:
                raise SubmissionRefused(
                    binance_filter_refusal_reason(exc)) from exc
            qty = float(normalized_qty)
            price = None if normalized_price is None else float(normalized_price)
            from binance_api import bapi_placeorder as _po
            if is_market or price is None:
                res = _po._submit_binance_order(
                    order_type, symbol, qty, market=True,
                    client_order_id=client_order_id, api_client=client,
                    cache_permit=cache_permit,
                    permit_requested_price=permit_requested_price,
                    kind=kind)
            else:
                kwargs = {"force": False}
                if client_order_id is not None:
                    kwargs["client_order_id"] = client_order_id
                if cache_permit is not None:
                    kwargs.update({
                        "cache_permit": cache_permit,
                        "permit_requested_price": permit_requested_price,
                        "kind": kind,
                    })
                res = _po.place_order_mechanics(
                    order_type, symbol, price, qty, **kwargs)
        except SubmissionRefused:
            raise
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"submit_order({symbol}): {e}") from e
        oid = extract_order_id(res)
        if oid is None:
            raise ProviderError(f"submit_order({symbol}): a response without an orderId ({res})")
        return oid

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
            raw_tid = trade.get("orderId", -1)
            if raw_tid is None or isinstance(raw_tid, bool):
                continue
            try:
                if isinstance(raw_tid, int):
                    tid_i = raw_tid
                elif isinstance(raw_tid, str):
                    s_tid = raw_tid.strip()
                    tid_i = int(s_tid) if s_tid.lstrip("+-").isdigit() else int(float(s_tid))
                else:
                    tid_f = float(raw_tid)
                    if not math.isfinite(tid_f) or not tid_f.is_integer():
                        continue
                    tid_i = int(tid_f)
                if tid_i != order_id:
                    continue
            except (TypeError, ValueError, OverflowError):
                continue
            fee_info = extract_order_fee(trade, base_asset=base, quote_asset=quote)
            if fee_info is None or fee_info["fee"] <= 0.0:
                continue
            if fee_info["fee_in_quote"] is not None:
                total += fee_info["fee_in_quote"]
            else:
                asset = fee_info["fee_asset"]
                conversion = _get_bapi().get_current_price(f"{asset}{quote}")
                try:
                    conv_f = float(conversion) if conversion is not None and not isinstance(conversion, bool) else 0.0
                except (TypeError, ValueError, OverflowError):
                    conv_f = 0.0
                if not math.isfinite(conv_f) or conv_f <= 0.0:
                    raise ProviderError(
                        f"order_status({order_id}): cannot convert the fee {asset}->{quote}"
                    )
                total += fee_info["fee"] * conv_f
        return total

    def order_status(self, symbol: str, order_id: str):
        oid = _parse_binance_order_id(order_id, f"order_status({order_id})")

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
        raw_exec = o.get("executedQty")
        raw_cost = o.get("cummulativeQuoteQty")
        raw_orig = o.get("origQty")
        if any(isinstance(v, bool) for v in (raw_exec, raw_cost, raw_orig)):
            raise ProviderError(f"order_status({order_id}): invalid fill values")
        try:
            executed_qty = float(raw_exec or 0.0)
            cost_val = float(raw_cost or 0.0)
            orig_val = float(raw_orig or 0.0)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProviderError(f"order_status({order_id}): invalid fill values") from exc
        if not all(math.isfinite(v) and v >= 0.0 for v in (executed_qty, cost_val, orig_val)):
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
        oid = _parse_binance_order_id(order_id, f"cancel_order({order_id})")
        try:
            canceled = _get_bapi().cancel_order(symbol, oid)
            if not canceled:
                raise ProviderError(
                    f"cancel_order({order_id}): the venue did not confirm the cancellation"
                )
        except ProviderError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"cancel_order({order_id}): {e}") from e


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

    def _provider_explicit_or_routed(self, symbol: str, provider_name=None):
        if provider_name is None:
            provider = self._provider_for(symbol)
        else:
            provider = self.provider_by_name(provider_name)
            if provider is None:
                raise ValueError(f"Unknown provider: {provider_name!r}")
        provider.validate_symbol(symbol)
        return provider

    def _provider_for(self, symbol: str) -> MarketDataProvider:
        if not isinstance(symbol, str):
            return self._providers[0]
        provider = self._route.get(symbol)
        if provider is not None:
            return provider
        for candidate in self._providers:
            try:
                if candidate.supports_symbol(symbol):
                    self._route[symbol] = candidate
                    return candidate
            except Exception:
                continue
        # Behavior-preserving default: first provider (Binance).
        default = self._providers[0]
        self._route[symbol] = default
        return default

    def get_current_price(
        self, symbol: str, *, provider_name=None,
    ) -> Optional[float]:
        """Read a quote through explicit venue routing when it is configured."""
        provider = self._provider_explicit_or_routed(symbol, provider_name)
        return provider.get_current_price(symbol)

    def get_price_history(
        self, symbol: str, lookback_h: float, *, provider_name=None,
    ) -> Optional[List]:
        provider = (
            self._provider_explicit_or_routed(symbol, provider_name)
            if provider_name is not None
            else self._provider_for(symbol)
        )
        return provider.get_price_history(symbol, lookback_h)

    # -- Account routing by symbol/asset, normalized by provider. ---------------
    def free_balance(self, asset: str) -> Optional[float]:
        """Preserve legacy compatibility; new code should use free_balance_for.

        A bare asset is ambiguous across venues. Historical first-provider routing
        remains only to avoid breaking old external integrations.
        """
        return self._provider_for(asset).free_balance(asset)

    def free_balance_for(self, provider_name: str, asset: str) -> Optional[float]:
        """Read free balance explicitly from the requested unambiguous venue."""
        provider = self.provider_by_name(provider_name)
        if provider is None:
            raise ValueError(f"Unknown provider: {provider_name!r}")
        return provider.free_balance(asset)

    def get_orders(
        self, symbol: str, side: Optional[str], since_s: float, *, provider_name=None,
    ) -> List[dict]:
        provider = (
            self._provider_explicit_or_routed(symbol, provider_name)
            if provider_name is not None
            else self._provider_for(symbol)
        )
        return provider.get_orders(symbol, side, since_s)

    def get_trades(self, symbol: str, since_s: float, *,
                   provider_name=None) -> List[dict]:
        provider = self._provider_explicit_or_routed(symbol, provider_name)
        return provider.get_trades(symbol, since_s)

    def position_cost_basis(self, symbol: str, held_qty: float, since_s: float, *,
                            provider_name=None) -> Optional[float]:
        provider = self._provider_explicit_or_routed(symbol, provider_name)
        return provider.position_cost_basis(symbol, held_qty, since_s)

    def latest_fill_price(self, symbol: str, side: str, since_s: float, *,
                          provider_name=None, min_notional=None,
                          max_notional=None) -> Optional[float]:
        """Return the newest normalized fill without a Binance-specific API."""
        if isinstance(side, bool):
            raise ValueError("side must be BUY or SELL")
        side = str(side or "").strip().upper()
        if side == "B":
            side = "BUY"
        elif side == "S":
            side = "SELL"
        if side not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL")
        if isinstance(since_s, bool) or since_s is None:
            raise ValueError("since_s must be finite and positive")
        try:
            since_s = float(since_s)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("since_s must be finite and positive") from exc
        if not math.isfinite(since_s) or since_s <= 0:
            raise ValueError("since_s must be finite and positive")
        if isinstance(min_notional, bool):
            raise ValueError("min_notional must be finite and non-negative")
        if isinstance(max_notional, bool):
            raise ValueError("max_notional must be finite and non-negative")
        try:
            min_value = None if min_notional is None else float(min_notional)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("min_notional must be finite and non-negative") from exc
        try:
            max_value = None if max_notional is None else float(max_notional)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("max_notional must be finite and non-negative") from exc
        if min_value is not None and (not math.isfinite(min_value) or min_value < 0):
            raise ValueError("min_notional must be finite and non-negative")
        if max_value is not None and (not math.isfinite(max_value) or max_value < 0):
            raise ValueError("max_notional must be finite and non-negative")
        if min_value is not None and max_value is not None and min_value > max_value:
            raise ValueError("min_notional cannot exceed max_notional")
        now_ms = time.time() * 1000.0
        cutoff_ms = now_ms - since_s * 1000.0
        candidates = []
        for trade in self.get_trades(
                symbol, since_s, provider_name=provider_name) or []:
            if extract_order_side(trade) != side:
                continue
            try:
                raw_ts = extract_timestamp_ms(trade, default=0) or 0
                timestamp = float(raw_ts)
                if 0 < timestamp < 10_000_000_000:  # Seconds to milliseconds.
                    timestamp *= 1000.0
                price = extract_order_price(trade, default=0.0)
                qty = extract_order_qty(trade, default=0.0)
            except (TypeError, ValueError, OverflowError):
                continue
            if (not all(math.isfinite(value) for value in (timestamp, price, qty)) or
                    timestamp < cutoff_ms or timestamp > now_ms + 60_000 or
                    price <= 0 or qty <= 0):
                continue
            notional = price * qty
            if not math.isfinite(notional):
                continue
            if min_value is not None and notional < min_value:
                continue
            if max_value is not None and notional > max_value:
                continue
            candidates.append((timestamp, price))
        return max(candidates)[1] if candidates else None

    @staticmethod
    def _reconciliation_capabilities(provider) -> OrderReconciliationCapabilities:
        return reconciliation_capabilities_of(provider)

    def reconciliation_capabilities(
            self, symbol: str, *, provider_name=None
    ) -> OrderReconciliationCapabilities:
        provider = self._provider_explicit_or_routed(symbol, provider_name)
        return self._reconciliation_capabilities(provider)

    def open_orders(self, symbol: str, *, provider_name=None) -> List[dict]:
        provider = self._provider_explicit_or_routed(symbol, provider_name)
        capabilities = self._reconciliation_capabilities(provider)
        if not capabilities.list_open_orders:
            raise ProviderError(f"{provider.name}: open_orders is unsupported")
        return provider.open_orders(symbol)

    def order_by_client_id(self, symbol: str, client_order_id: str, *,
                           provider_name=None):
        """Return a native order found by deterministic client ID, or ``None``.

        This narrow recovery hook lets persistent callers reconcile an ambiguous
        submit without guessing from balances or submitting a duplicate.
        """
        provider = self._provider_explicit_or_routed(symbol, provider_name)
        capabilities = self._reconciliation_capabilities(provider)
        if not capabilities.lookup_by_client_order_id:
            raise ProviderError(
                f"{provider.name}: order_by_client_id is unsupported")
        method = getattr(provider, "order_by_client_id", None)
        if not callable(method):
            raise ProviderError(
                f"{provider.name}: declared order_by_client_id is missing")
        if (
            client_order_id is None
            or isinstance(client_order_id, bool)
            or not isinstance(client_order_id, (str, int))
            or (isinstance(client_order_id, int) and client_order_id <= 0)
            or not str(client_order_id).strip()
            or str(client_order_id).strip().lower() in _INVALID_API_ID_STRINGS
        ):
            raise ProviderError(f"{provider.name}: invalid client_order_id")
        return method(symbol, str(client_order_id).strip())

    @staticmethod
    def _clean_order_id_arg(order_id: Any, provider_name: str) -> str:
        if order_id is None or isinstance(order_id, bool):
            raise ProviderError(f"{provider_name}: invalid order_id")
        if isinstance(order_id, float):
            if not math.isfinite(order_id) or not order_id.is_integer() or order_id <= 0:
                raise ProviderError(f"{provider_name}: invalid order_id")
            return str(int(order_id))
        if isinstance(order_id, int):
            if order_id <= 0:
                raise ProviderError(f"{provider_name}: invalid order_id")
            return str(order_id)
        if not isinstance(order_id, str):
            raise ProviderError(f"{provider_name}: invalid order_id")
        text = order_id.strip()
        if text.endswith(".0") and text[:-2].isdigit():
            text = text[:-2]
        if not text or text.lower() in _INVALID_API_ID_STRINGS:
            raise ProviderError(f"{provider_name}: invalid order_id")
        if text.lstrip("+-").isdigit() and int(text) <= 0:
            raise ProviderError(f"{provider_name}: invalid order_id")
        return text

    def order_status(self, symbol: str, order_id: str, *,
                     provider_name=None) -> OrderStatus:
        """Return venue-neutral status; lookup failures remain fail-closed."""
        provider = self._provider_explicit_or_routed(symbol, provider_name)
        capabilities = self._reconciliation_capabilities(provider)
        if not capabilities.status_by_order_id:
            raise ProviderError(f"{provider.name}: order_status is unsupported")
        method = getattr(provider, "order_status", None)
        if not callable(method):
            raise ProviderError(f"{provider.name}: declared order_status is missing")
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
        provider = self._provider_explicit_or_routed(symbol, provider_name)
        capabilities = self._reconciliation_capabilities(provider)
        if not capabilities.cancel_by_order_id:
            raise ProviderError(f"{provider.name}: cancel_order is unsupported")
        method = getattr(provider, "cancel_order", None)
        if not callable(method):
            raise ProviderError(f"{provider.name}: declared cancel_order is missing")
        oid_str = self._clean_order_id_arg(order_id, provider.name)
        method(symbol, oid_str)

    def tracked_order_lifecycle(self, *, provider_name: str, venue=None,
                                missing_confirmations=2,
                                retry_on_lookup_error=False,
                                max_age_seconds=None, audit=None,
                                clock=time.time):
        """Build the persistent lifecycle companion for synchronous ``place``.

        Persistence remains strategy-owned and is supplied to each lifecycle call;
        this facade supplies provider-neutral lookup/status/cancel routing.
        """
        from order_retry import TrackedOrderLifecycle
        return TrackedOrderLifecycle(
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
        return MarketRegimeService(
            strength_threshold,
            cache_ttl_sec=service.cache_ttl_sec,
            negative_cache_ttl_sec=service.negative_cache_ttl_sec,
            cache_max=service.cache_max,
            clock=service.clock,
        )

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
        if isinstance(now, bool):
            raise ValueError("now must be finite")
        try:
            evaluated_at = time.time() if now is None else float(now)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("now must be finite") from exc
        if not math.isfinite(evaluated_at):
            raise ValueError("now must be finite")
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
            symbol,
            provider_name=provider_name,
            horizon=horizon,
            interval_min=interval_min,
            window_seconds=window_seconds,
            snapshot=snapshot,
            snapshot_source=snapshot_source,
            snapshot_max_age_seconds=snapshot_max_age_seconds,
            ohlc_max_age_seconds=ohlc_max_age_seconds,
            strength_threshold=strength_threshold,
            allow_fallback=allow_fallback,
        ).decision

    def market_regime_bundle(
        self,
        symbol: str,
        *,
        benchmarks=(),
        provider_name=None,
        benchmark_provider_name=None,
        use_case="balanced",
        weights=None,
        snapshot=None,
        snapshot_source="snapshot",
        snapshot_max_age_seconds=None,
        ohlc_max_age_seconds=None,
        strength_threshold=None,
        allow_fallback=True,
        include_alternates=False,
        now=None,
    ) -> MarketRegimeBundle:
        """Collect evidence and select only fresh, usable directional sources."""
        service = self._regime_service_for(strength_threshold)
        provider = self._provider_explicit_or_routed(symbol, provider_name)
        if isinstance(now, bool):
            raise ValueError("now must be finite")
        try:
            evaluated_at = time.time() if now is None else float(now)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("now must be finite") from exc
        if not math.isfinite(evaluated_at):
            raise ValueError("now must be finite")
        if snapshot_max_age_seconds is None:
            snapshot_max_age_seconds = (
                service.default_snapshot_max_age_seconds())
        if ohlc_max_age_seconds is None:
            ohlc_max_age_seconds = service.default_ohlc_max_age_seconds()

        if isinstance(benchmarks, str):
            benchmark_candidates = (benchmarks,)
        else:
            benchmark_candidates = tuple(benchmarks or ())
        normalized_benchmarks = []
        seen_benchmarks = set()
        for item in benchmark_candidates:
            if isinstance(item, bool) or not isinstance(item, str):
                raise ValueError("benchmark symbols must be non-empty")
            benchmark = item.strip()
            if not benchmark:
                raise ValueError("benchmark symbols must be non-empty")
            identity = benchmark.upper().replace("/", "")
            if identity not in seen_benchmarks:
                normalized_benchmarks.append(benchmark)
                seen_benchmarks.add(identity)

        def resolve(
            source_provider,
            target,
            horizon,
            role,
            source_snapshot=None,
        ):
            return service.resolve_with_evidence(
                source_provider,
                target,
                horizon=horizon,
                snapshot=source_snapshot,
                snapshot_source=snapshot_source,
                snapshot_max_age_seconds=snapshot_max_age_seconds,
                ohlc_max_age_seconds=ohlc_max_age_seconds,
                allow_fallback=allow_fallback,
                include_alternates=include_alternates,
                role=role,
                now=evaluated_at,
            )

        asset_short = resolve(
            provider, symbol, "short", "asset_short", snapshot)
        asset_long = resolve(provider, symbol, "long", "asset_long")
        benchmark_resolutions = []
        for benchmark in normalized_benchmarks:
            if benchmark_provider_name is not None:
                benchmark_provider = self._provider_explicit_or_routed(
                    benchmark, benchmark_provider_name)
            elif provider_name is not None:
                benchmark_provider = provider
            else:
                benchmark_provider = self._provider_explicit_or_routed(
                    benchmark, None)
            short = resolve(
                benchmark_provider, benchmark, "short", "benchmark_short")
            long = resolve(
                benchmark_provider, benchmark, "long", "benchmark_long")
            benchmark_resolutions.append((benchmark, short, long))

        all_evidence = (
            tuple(asset_short.evidence)
            + tuple(asset_long.evidence)
            + tuple(
                item
                for _benchmark, short, long in benchmark_resolutions
                for item in (*short.evidence, *long.evidence)
            )
        )
        composite = service.compose_evidence(
            all_evidence,
            asset_symbol=symbol,
            use_case=use_case,
            weights=weights,
        )
        return MarketRegimeBundle(
            composite,
            asset_short,
            asset_long,
            tuple(benchmark_resolutions),
        )

    def composite_market_regime(
        self,
        symbol: str,
        *,
        benchmarks=(),
        provider_name=None,
        benchmark_provider_name=None,
        use_case="balanced",
        weights=None,
        strength_threshold=None,
        allow_fallback=True,
    ) -> CompositeMarketRegimeDecision:
        """Blend asset horizons with explicitly configured crypto benchmarks."""
        return self.market_regime_bundle(
            symbol,
            benchmarks=benchmarks,
            provider_name=provider_name,
            benchmark_provider_name=benchmark_provider_name,
            use_case=use_case,
            weights=weights,
            strength_threshold=strength_threshold,
            allow_fallback=allow_fallback,
        ).composite

    def preflight_order(self, symbol: str, side: str, qty: float,
                        price=None, *, market: bool = False,
                        kind: Optional[str] = None, provider_name=None) -> Any:
        """Validate routed venue state before cancellation or submission."""
        provider = self._provider_explicit_or_routed(symbol, provider_name)
        return provider.preflight_order(
            symbol, side, qty, price, market=market, kind=kind)

    def order_filter_refusal(
        self,
        symbol: str,
        side: str,
        price: float,
        qty: float,
        *,
        market: bool = False,
        enforce_business_minimum: bool = True,
        provider_name=None,
    ) -> Optional[str]:
        """Return a routed deterministic venue-filter refusal, if any."""
        provider = self._provider_explicit_or_routed(symbol, provider_name)
        return provider.order_filter_refusal(
            symbol,
            side,
            price,
            qty,
            market=market,
            enforce_business_minimum=enforce_business_minimum,
        )

    def pair_precision(self, symbol: str, provider_name=None) -> Optional[PairPrecision]:
        provider = self._provider_explicit_or_routed(symbol, provider_name)
        if hasattr(provider, "pair_precision"):
            return provider.pair_precision(symbol)
        return None

    def min_order_qty(self, symbol: str, provider_name=None) -> float:
        provider = self._provider_explicit_or_routed(symbol, provider_name)
        if hasattr(provider, "min_order_qty"):
            return provider.min_order_qty(symbol)
        return 0.0

    def round_quantity(self, symbol: str, qty: float, provider_name=None) -> float:
        provider = self._provider_explicit_or_routed(symbol, provider_name)
        if hasattr(provider, "round_quantity"):
            return provider.round_quantity(symbol, qty)
        if hasattr(provider, "round_amount"):
            return provider.round_amount(symbol, qty)
        if qty is None or isinstance(qty, bool):
            return 0.0
        try:
            qf = float(qty)
            return qf if math.isfinite(qf) else 0.0
        except (TypeError, ValueError, OverflowError):
            return 0.0

    def round_amount(self, symbol: str, qty: float, provider_name=None) -> float:
        return self.round_quantity(symbol, qty, provider_name=provider_name)

    def round_price(self, symbol: str, price: float, provider_name=None) -> float:
        provider = self._provider_explicit_or_routed(symbol, provider_name)
        if hasattr(provider, "round_price"):
            return provider.round_price(symbol, price)
        if price is None or isinstance(price, bool):
            return 0.0
        try:
            pf = float(price)
            return pf if math.isfinite(pf) else 0.0
        except (TypeError, ValueError, OverflowError):
            return 0.0

    def place_order(self, symbol: str, side: str, price: float, qty: float, **kwargs):
        # Mechanics-only provider dispatch without guards. Real placement must use
        # guarded .place(); this remains for internal and dry-run cases.
        provider_name = kwargs.pop("provider_name", None)
        return self._provider_explicit_or_routed(symbol, provider_name).place_order(symbol, side, price, qty, **kwargs)

    def place(self, symbol: str, side: str, price: float, qty: Optional[float] = None,
              base: Optional[str] = None, quote: Optional[str] = None, **kwargs):
        """Place through the single guarded proxy using a temporary Instrument.

        Run the complete provider-agnostic policy pipeline with provider hooks.
        This replaces direct legacy smart/safe placement calls. Derive base and
        quote from the symbol when absent. Lazy Instrument import avoids a cycle.
        """
        from instrument import Instrument
        import utils as u
        explicit_provider_name = kwargs.pop("provider_name", None)
        inst_params = kwargs.pop("params", None)
        provider = self._provider_explicit_or_routed(
            symbol, explicit_provider_name)
        prov_name = provider.name
        if inst_params is None:
            try:
                from instrument_registry import select_instruments
                sym_clean = str(symbol or "").strip().upper().replace("/", "")
                for spec in select_instruments(prov_name).values():
                    if str(spec.symbol or "").strip().upper().replace("/", "") == sym_clean:
                        inst_params = dict(spec.params or {})
                        if base is None and spec.base:
                            base = spec.base
                        if quote is None and spec.quote:
                            quote = spec.quote
                        break
            except Exception:
                inst_params = {}
        if base is None:
            try:
                base = u.base_asset(symbol)
            except Exception:
                base = None
        sym_str = str(symbol or "").strip() if not isinstance(symbol, bool) else ""
        if quote is None and base and sym_str.upper().startswith(base.upper()) and sym_str.upper() != base.upper():
            quote = sym_str[len(base):].lstrip("/-_").strip() or None   # For example, BTCUSDC or HYPE/USDC gives BTC/HYPE and USDC.
        if quote is None and str(prov_name).lower() == "hyperliquid":
            quote = "USDC"
        inst = Instrument(name=symbol, symbol=symbol, provider=prov_name,
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
        if not isinstance(name, str):
            return None
        return self._by_name.get(name.strip().lower())

    @property
    def providers(self) -> List[MarketDataProvider]:
        return list(self._providers)


# Singleton injected when constructors receive api=None. Binance remains first to
# preserve routing for unclaimed symbols. Hyperliquid claims only HYPE and Binance
# explicitly excludes it. Provider construction is cheap and SDKs load lazily;
# import failure falls back cleanly without affecting the fleet.
_extra_providers = []
# Isolate every provider import so one missing dependency leaves the others intact.
# Kraken and T212 are explicit-only and reachable through Instrument, so their
# ordering does not affect symbol routing.
for _modname, _clsname in (("hyperliquid_provider", "HyperliquidProvider"),
                           ("kraken_provider", "KrakenProvider"),
                           ("t212_provider", "T212Provider")):
    try:
        _mod = __import__("providers." + _modname, fromlist=[_clsname])
        _extra_providers.append(getattr(_mod, _clsname)())
    except Exception as _e:  # noqa: BLE001
        print(f"market_api: {_clsname} unavailable ({_e})")

api = MarketApi(
    [BinanceProvider()] + _extra_providers,
    regime_snapshot_resolver=_runtime_market_regime_snapshot,
)
