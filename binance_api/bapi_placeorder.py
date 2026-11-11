import os
import time
import threading
from dataclasses import dataclass

# Binance
from binance.exceptions import BinanceAPIException

# MYLIB
import instrument_registry as sym
import botcore as cfg
from botcore import (
    _parse_bool_flag,
    clean_order_id,
    load_dotenv,
    required_float_env,
    required_int_env,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
)
from providers import order_id_context as rc
from providers.base import MarketDataProvider

from . import bapi as api, bapi_allorders as apiorders
from .bapi import _normalize_side
from .bapi_client import client
from providers.order_ohlc_utils import (
    extract_order_id,
    extract_order_price,
    extract_order_side,
    resolve_market_flag,
)
from providers.quantity import (
    BinanceFilterError,
    BinanceOrderRules,
    balance_cap_quantity,
    binance_filter_refusal_reason,
    decimal_places,
    fee_cap_quantity,
)
from providers.strategy_executor import SubmissionRefused
import account_cache_health

# Load versioned, non-secret tuning parameters before reading the environment below.
# ``botcore.load_dotenv`` never overwrites variables already set in the real environment;
# it only fills missing values, matching tradeall_config.env and monitortrades_config.env.
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "bapi_placeorder_config.env"))

# Order policy is mandatory; missing configuration aborts startup.
PLACE_ORDER_FEE_PCT = required_float_env("PLACE_ORDER_FEE_PCT")
PLACE_ORDER_HOURS = required_int_env("PLACE_ORDER_HOURS")
PLACE_ORDER_SAFEBACK_SEC = required_int_env("PLACE_ORDER_SAFEBACK_SEC")
PLACE_ORDER_MIN_NOTIONAL = required_float_env("PLACE_ORDER_MIN_NOTIONAL")


_CACHE_SUBMIT_PERMIT_ISSUER = object()


@dataclass(frozen=True, slots=True)
class _AccountCacheSubmitScope:
    """Immutable financial scope carried by an in-process submit permit."""

    pid: int
    issued_at: float
    expires_at: float
    expires_at_wall: float
    order_cache_version: str
    trade_cache_version: str
    symbol: str
    side: str
    max_qty: float
    requested_price: float | None
    price_tick_size: float
    market: bool
    kind: str


class _AccountCacheSubmitPermit:
    """Opaque, process-local authorization for one exact Binance submission.

    Replacement flows obtain this permit immediately before cancelling an existing
    order. The final submit may then rely on the already-validated cache snapshot
    during a strictly bounded window, avoiding a cancel/second-preflight race.
    """

    __slots__ = ("_scope", "_consumed", "_lock")

    def __init__(self, issuer, *, status, symbol, side, qty, price,
                 price_tick_size, market, kind):
        if issuer is not _CACHE_SUBMIT_PERMIT_ISSUER:
            raise TypeError("account-cache submit permits are issued internally")
        issued_at = time.monotonic()
        issued_at_wall = time.time()
        is_market = _parse_bool_flag(market, False)
        object.__setattr__(self, "_scope", _AccountCacheSubmitScope(
            pid=os.getpid(),
            issued_at=issued_at,
            expires_at=issued_at + account_cache_health.SUBMIT_PERMIT_TTL_SEC,
            expires_at_wall=issued_at_wall + account_cache_health.SUBMIT_PERMIT_TTL_SEC,
            order_cache_version=str(status.order_cache_version),
            trade_cache_version=str(status.trade_cache_version),
            symbol="" if isinstance(symbol, bool) else str(symbol or "").strip().upper(),
            side="" if isinstance(side, bool) else str(_normalize_side(side or "")).strip().upper(),
            max_qty=float(qty),
            requested_price=None if is_market else float(price),
            price_tick_size=float(price_tick_size),
            market=is_market,
            kind=str(kind or ""),
        ))
        object.__setattr__(self, "_consumed", False)
        object.__setattr__(self, "_lock", threading.Lock())

    def __setattr__(self, name, value):
        raise AttributeError("account-cache submit permits are immutable")

    @property
    def pid(self):
        """Expose process identity read-only for diagnostics and tests."""
        return self._scope.pid

    @property
    def expires_at(self):
        """Expose monotonic expiry read-only for diagnostics and tests."""
        return self._scope.expires_at


def apply_weight_limit(symbol, order_type, price, required_qty, available_qty):
    auto_qty = required_qty is None
    order_type = "" if isinstance(order_type, bool) or not isinstance(order_type, str) else _normalize_side(order_type)
    price, available_qty = safe_pos_float(price, default=None), safe_pos_float(available_qty, default=None)
    required_qty = available_qty if auto_qty else safe_pos_float(required_qty, default=None)
    if order_type not in {"BUY", "SELL"} or price is None or available_qty is None or required_qty is None:
        raise SubmissionRefused("invalid_weight_policy_inputs")

    is_sell = order_type == "SELL"

    def _fail_open_or_refuse(reason, exc):
        # An UNAVAILABLE policy is infrastructure downtime, not a legitimate cap of
        # zero. Failing closed is prudent for a BUY (never add exposure we cannot
        # verify), but trapping a SELL for the whole retry TTL is dangerous -- it can
        # leave a position unable to exit for a full day.
        if is_sell:
            print(f"apply_weight_limit -> SELL {symbol}: {reason} ({exc}); failing "
                  f"OPEN to balance cap {available_qty:.8f} so the exit is not trapped.")
            return available_qty
        raise SubmissionRefused(reason) from exc

    try:
        weight = safe_pos_float(MarketDataProvider.resolve_order_guard_trade_weight(
            symbol, order_type, provider_name="binance", raise_on_error=True), default=None)
    except SubmissionRefused:
        raise
    except Exception as exc:
        return _fail_open_or_refuse("weight_policy_unavailable", exc)

    if weight is None or weight > 1:
        raise SubmissionRefused("invalid_weight_policy_weight")

    try:
        stats = apiorders.get_total_traded_stats(symbol)
        traded_value = safe_nonneg_float(stats[order_type.upper()].get("total_value"), default=None)
    except Exception as exc:
        return _fail_open_or_refuse("trade_stats_unavailable", exc)
    if traded_value is None:
        raise SubmissionRefused("invalid_trade_stats")

    adjusted_qty = MarketDataProvider.compute_order_guard_weight_capped_qty(
        weight=weight, price=price, required_qty=required_qty,
        available_qty=available_qty, traded_24h_value=traded_value,
    )
    total_value_reference = traded_value + available_qty * price
    max_trade_value = total_value_reference * weight
    remaining_trade_value = max(0.0, max_trade_value - traded_value)
    remaining_trade_qty = remaining_trade_value / price
    if any(safe_float(v, default=None) is None for v in (
        total_value_reference, max_trade_value, remaining_trade_value, remaining_trade_qty, adjusted_qty
    )):
        raise SubmissionRefused("invalid_weight_policy_result")

    print(f"apply_weight_limit → {order_type} {symbol}, "
          f"Available qty {available_qty:.8f}, Weight {weight}, "
          f"Traded in 24h {traded_value:.2f} USDC, "
          f"Max trade allowed (24h): {max_trade_value:.2f} USDC, "
          f"Remaining: {remaining_trade_value:.2f} USDC, "
          f"Required qty: {required_qty:.8f}, Final qty: {adjusted_qty:.8f}")
    return adjusted_qty


def require_account_cache_for_submit():
    """Fail closed unless the central Binance account caches are current."""
    if not cfg.is_trade_enabled():
        raise SubmissionRefused("trading_disabled")
    try:
        for _attempt in range(2):
            requested = account_cache_health.require_fresh_account_cache()
            MarketDataProvider.ensure_ssot_account_cache_readers(requested)
            confirmed = account_cache_health.require_fresh_account_cache()
            if (requested.order_cache_version, requested.trade_cache_version) == (
                confirmed.order_cache_version, confirmed.trade_cache_version
            ):
                return confirmed
        raise account_cache_health.AccountCacheNotReady(
            "account_cache_changed_during_reader_sync")
    except account_cache_health.AccountCacheNotReady as exc:
        print(f"[BINANCE][GUARD] account cache is not fresh: {exc.reason}")
        raise SubmissionRefused("account_cache_not_fresh") from exc


def _require_submit_permit_headroom(status):
    parsed_ages = [safe_nonneg_float(age, default=None) for age in (status.order_age_sec, status.trade_age_sec)]
    if (
        any(age_f is None for age_f in parsed_ages)
        or max(parsed_ages) + account_cache_health.SUBMIT_PERMIT_TTL_SEC > account_cache_health.MAX_AGE_SEC
    ):
        raise SubmissionRefused("account_cache_not_fresh")
    return status


def issue_account_cache_submit_permit(symbol, side, qty, price=None, *,
                                      market=False, kind=None, api_client=None):
    """Authorize one narrowly scoped submit after a complete cache validation."""
    market = _parse_bool_flag(market, False)
    side_str = "" if isinstance(side, bool) else str(_normalize_side(side or "")).strip().upper()
    symbol_str = "" if isinstance(symbol, bool) else str(symbol or "").strip().upper()
    qty, price = safe_pos_float(qty, default=None), (None if market else safe_pos_float(price, default=None))
    if side_str not in {"BUY", "SELL"} or not symbol_str or qty is None or (not market and price is None):
        raise SubmissionRefused("invalid_order_parameters")
    if not cfg.is_trade_enabled():
        raise SubmissionRefused("trading_disabled")
    _require_submit_permit_headroom(require_account_cache_for_submit())
    price_tick_size = 0.0
    if not market:
        try:
            rules = BinanceOrderRules.from_symbol_info((api_client or client).get_symbol_info(symbol_str))
            price_tick_size = float(rules.tick_size)
        except Exception as exc:
            raise SubmissionRefused("binance_symbol_rules_unavailable") from exc
    status = _require_submit_permit_headroom(require_account_cache_for_submit())
    return _AccountCacheSubmitPermit(
        _CACHE_SUBMIT_PERMIT_ISSUER, status=status, symbol=symbol_str, side=side_str,
        qty=qty, price=price, market=market, price_tick_size=price_tick_size, kind=kind,
    )


def _consume_account_cache_submit_permit(
        permit, *, symbol, side, qty, actual_price, requested_price,
        market, kind):
    """Consume a matching permit once or refuse before any client side effect."""
    market = _parse_bool_flag(market, False)
    if not isinstance(permit, _AccountCacheSubmitPermit):
        raise SubmissionRefused("account_cache_not_fresh")
    with permit._lock:
        if permit._consumed:
            raise SubmissionRefused("account_cache_not_fresh")
        object.__setattr__(permit, "_consumed", True)
        scope = permit._scope
        actual_qty = safe_pos_float(qty, default=None)
        if actual_qty is None:
            raise SubmissionRefused("account_cache_not_fresh")
        if market:
            submitted_price = scoped_price = None
        else:
            submitted_price = safe_pos_float(actual_price, default=None)
            scoped_price = safe_pos_float(requested_price, default=None)
            if submitted_price is None or scoped_price is None:
                raise SubmissionRefused("account_cache_not_fresh")
        price_is_safe = scope.market or (
            scoped_price == scope.requested_price
            and (
                (scope.side == "BUY" and submitted_price <= scoped_price)
                or (scope.side == "SELL" and submitted_price + scope.price_tick_size >= scoped_price)
            )
        )
        matches = (
            scope.pid == os.getpid()
            and time.monotonic() <= scope.expires_at
            and time.time() <= scope.expires_at_wall
            and bool(scope.order_cache_version) and bool(scope.trade_cache_version)
            and symbol is not None and not isinstance(symbol, bool)
            and side is not None and not isinstance(side, bool)
            and scope.symbol == str(symbol).strip().upper()
            and scope.side == str(_normalize_side(side))
            and scope.market is market
            and scope.kind == str(kind or "")
            and actual_qty <= scope.max_qty
            and price_is_safe
        )
        if not matches:
            raise SubmissionRefused("account_cache_not_fresh")


def _submit_binance_order(order_type, symbol, qty, *, price=None, market=False,
                          client_order_id=None, api_client=None,
                          cache_permit=None, permit_requested_price=None,
                          kind=None, cancel_opposite_requested_price=None):
    """Single low-level dispatch after common filter normalization."""
    market = _parse_bool_flag(market, False)
    side = "" if order_type is None or isinstance(order_type, bool) else str(_normalize_side(order_type))
    qty_f = safe_pos_float(qty, default=None)
    price_f = None if market else safe_pos_float(price, default=None)
    if qty_f is None or (not market and price_f is None):
        raise SubmissionRefused("invalid_order_parameters")
    cancel_price = None
    if cancel_opposite_requested_price is not None:
        cancel_price = safe_pos_float(cancel_opposite_requested_price, default=None)
        if cancel_price is None:
            raise SubmissionRefused("invalid_opposing_cancel_request")
        if cache_permit is None:
            raise SubmissionRefused("account_cache_not_fresh")

    target_client = api_client or client
    client_order_id = clean_order_id(client_order_id) or rc.create_client_order_id()
    if market:
        method = target_client.order_market_buy if side == "BUY" else target_client.order_market_sell
        submit_kwargs = {"symbol": symbol, "quantity": qty}
    else:
        method = target_client.order_limit_buy if side == "BUY" else target_client.order_limit_sell
        submit_kwargs = {"symbol": symbol, "quantity": qty, "price": str(price)}
    if cache_permit is None:
        if side not in {"BUY", "SELL"} or not cfg.is_trade_enabled():
            return None
        require_account_cache_for_submit()
    else:
        _consume_account_cache_submit_permit(
            cache_permit, symbol=symbol, side=side, qty=qty,
            actual_price=price, requested_price=permit_requested_price,
            market=market, kind=kind,
        )
    if not cfg.is_trade_enabled():
        raise SubmissionRefused("trading_disabled")
    if cancel_price is not None:
        cancel_opposite_orders(side, symbol, cancel_price)
    try:
        order = method(**submit_kwargs, newClientOrderId=client_order_id)
        if order:
            oid_label = extract_order_id(order, default=None) or (
                order.get("orderId") if isinstance(order, dict) else order
            )
            print(f"{side} order placed successfully: {oid_label} clientId {client_order_id}")
        return order
    except BinanceAPIException as exc:
        print(f"Binance {side} submission failed: {exc}")
        return None


def _last_opposite_fill_price_api(symbol, order_type):
    """Query ``get_my_trades`` when the cache has no opposite fill.

    This covers a new symbol or an unpopulated cache. Raise on API errors so the caller
    fails closed; return ``None`` only when Binance confirms no opposite fill exists.
    """
    side = "" if order_type is None or isinstance(order_type, bool) else str(_normalize_side(order_type))
    if side not in {"BUY", "SELL"}:
        raise ValueError(f"Invalid order_type: {order_type!r}")
    want_side = "BUY" if side == "SELL" else "SELL"
    for tr in reversed(client.get_my_trades(symbol=symbol, limit=200) or ()):
        if not isinstance(tr, dict):
            continue
        if extract_order_side(tr) == want_side:
            px = extract_order_price(tr)
            if px > 0:
                return px
    return None


def cancel_opposite_orders(order_type, symbol, requested_price, allowed_owners=None):
    """Cancel adverse opposing Binance orders without changing the target price."""
    sym_str = "" if isinstance(symbol, bool) else str(symbol or "").strip()
    req_px = None if isinstance(order_type, bool) or not sym_str else safe_pos_float(requested_price, default=None)
    if req_px is None:
        raise SubmissionRefused("invalid_opposing_cancel_request")
    order_type = str(_normalize_side(order_type or "")).strip().upper()
    if order_type not in {"BUY", "SELL"}:
        return
    opp_side = "SELL" if order_type == "BUY" else "BUY"
    try:
        open_kw = {"strict": True}
        if allowed_owners is not None:
            open_kw["allowed_owners"] = allowed_owners
        open_orders = api.get_open_orders(opp_side, sym_str, **open_kw)
    except Exception as exc:
        raise SubmissionRefused("opposing_order_discovery_unavailable") from exc
    for order_id, order_details in open_orders.items():
        order_px = extract_order_price(order_details, default=0.0)
        if order_px > 0 and ((order_px < req_px) if order_type == "BUY" else (order_px > req_px)):
            try:
                canceled = api._cancel_and_resolve(sym_str, order_id)
            except Exception as exc:
                raise SubmissionRefused("opposing_cancel_unconfirmed") from exc
            if not canceled:
                raise SubmissionRefused("opposing_cancel_unconfirmed")


def adjust_price_and_cancel_opposite(order_type, symbol, price, cancel_opposite=True):
    """Apply Binance price mechanics and optionally cancel opposing orders."""
    cancel_opposite = _parse_bool_flag(cancel_opposite, True)
    sym_str = "" if isinstance(symbol, bool) else str(symbol or "").strip()
    price = None if isinstance(order_type, bool) or not sym_str else safe_pos_float(price, default=None)
    if price is None:
        raise SubmissionRefused("invalid_order_parameters")
    order_type = str(_normalize_side(order_type or "")).strip().upper()
    if cancel_opposite:
        cancel_opposite_orders(order_type, sym_str, price)
    cp = safe_pos_float(api.get_current_price(sym_str), default=None)
    if cp is not None and order_type in {"BUY", "SELL"}:
        price = order_candidate_price(order_type, price, cp) * (0.999 if order_type == "BUY" else 1.001)
    try:
        info = api.client.get_symbol_info(sym_str)
        rules = BinanceOrderRules.from_symbol_info(info)
        return round(price, min(decimal_places(rules.tick_size), 18))
    except Exception:
        return round(price, 8)


def order_candidate_price(order_type, requested_price, current_price, *, market=False):
    """Return the exact price used by Binance filter normalization."""
    if _parse_bool_flag(market, False):
        return None
    side = "" if isinstance(order_type, bool) else str(_normalize_side(order_type)).strip().upper()
    if side not in {"BUY", "SELL"}:
        raise ValueError("order_type must be BUY or SELL")
    if isinstance(requested_price, bool):
        raise ValueError("requested_price cannot be bool")
    requested_price = safe_pos_float(requested_price, default=None)
    if requested_price is None:
        raise ValueError("requested_price must be finite and positive")
    cp = safe_pos_float(current_price, default=None)
    if cp is None:
        return requested_price
    return max(requested_price, cp) if side == "SELL" else min(requested_price, cp)


def place_order_mechanics(order_type, symbol, price, qty, force=False,
                          client_order_id=None, cache_permit=None,
                          permit_requested_price=None, kind=None,
                          cancel_opposite_requested_price=None,
                          enforce_business_minimum=True, market=False, **kwargs):
    """Execute Binance-specific submission mechanics."""
    order_type = _normalize_side(order_type)
    sym.validate_params(order_type, symbol, price, qty)
    price, qty = float(price), float(qty)
    enforce_business_minimum = _parse_bool_flag(enforce_business_minimum, True)
    is_market = resolve_market_flag(force=force, market=market, **kwargs)
    try:
        balance_fn = (
            (lambda _asset: kwargs["known_balance"])
            if kwargs.get("known_balance") is not None
            else api.get_free_balance
        )
        available_qty, _balance_asset = balance_cap_quantity(balance_fn, symbol, order_type, price)
        if available_qty is None:
            if (
                _parse_bool_flag(kwargs.get("_balance_verified"), False)
                or _parse_bool_flag(kwargs.get("caller_owns_retry"), False)
            ):
                print(f"Balance check soft-skipped for {order_type} {symbol}; using verified qty={qty:.8f}")
            else:
                print(f"Balance unavailable for {order_type} {symbol}; order skipped.")
                return None
        elif available_qty <= 0:
            print(f"No sufficient quantity available to place the {order_type} order.")
            return None
        else:
            fee_cap = fee_cap_quantity(available_qty, PLACE_ORDER_FEE_PCT)
            if qty > fee_cap:
                print(f"Adjusting {order_type} qty from {qty:.8f} to {fee_cap:.8f} to cover balance and fees")
                qty = fee_cap

        current_price = api.get_current_price(symbol)
        ref_price = safe_pos_float(current_price, default=None) or price
        rules = BinanceOrderRules.from_symbol_info(client.get_symbol_info(symbol))
        candidate_price = order_candidate_price(order_type, price, current_price, market=is_market)
        try:
            normalized_qty, normalized_price = rules.normalize(
                quantity=qty, price=candidate_price, market=is_market,
                reference_price=ref_price,
                business_min_notional=PLACE_ORDER_MIN_NOTIONAL if enforce_business_minimum else 0,
            )
        except BinanceFilterError as exc:
            raise SubmissionRefused(binance_filter_refusal_reason(exc)) from exc
        qty = float(normalized_qty)
        if normalized_price is not None:
            price = float(normalized_price)

        print(f"Trying to place {order_type} {symbol} qty {qty:.8f} at "
              f"{'market price' if is_market else f'price {price}'}")
        if order_type in {"BUY", "SELL"}:
            return _submit_binance_order(
                order_type, symbol, qty, price=None if is_market else price,
                market=is_market, client_order_id=client_order_id,
                cache_permit=cache_permit, permit_requested_price=permit_requested_price,
                kind=kind, cancel_opposite_requested_price=cancel_opposite_requested_price,
            )
        print(f"Invalid order type: {order_type}")
        return None
    except BinanceAPIException as e:
        print(f"Error placing {order_type} order (mechanics): {e}")
        return None
