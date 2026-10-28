import os
import time
import math
import sys
import threading
from dataclasses import dataclass

#### Binance
from binance.exceptions import BinanceAPIException

#### MYLIB
import order_guard
import symbols as sym
import config as cfg
import priceAnalysis as pa  # Kept at module namespace for monkeypatching in tests
from . import order_id_context as rc   # client_order_id and tag context moved into binance_api.

from . import bapi as api
from .bapi_client import client
from providers.base import (
    extract_order_id,
    extract_order_price,
    extract_order_side,
    extract_timestamp_ms,
)
from providers.strategy_executor import SubmissionRefused
import binance_cache_health

# Load versioned, non-secret tuning parameters before reading the environment below.
# ``botcore.load_dotenv`` never overwrites variables already set in the real environment;
# it only fills missing values, matching tradeall_config.env and monitortrades_config.env.
from botcore import _parse_bool_flag, load_dotenv as _load_dotenv, required_float_env, required_int_env
_load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "bapi_placeorder_config.env"))

# Order policy is mandatory; missing configuration aborts startup.
PLACE_ORDER_FEE_PCT = required_float_env("PLACE_ORDER_FEE_PCT")
PLACE_ORDER_HOURS = required_int_env("PLACE_ORDER_HOURS")
PLACE_ORDER_SAFEBACK_SEC = required_int_env("PLACE_ORDER_SAFEBACK_SEC")
PLACE_ORDER_MIN_NOTIONAL = required_float_env("PLACE_ORDER_MIN_NOTIONAL")


class WeightLimitBlock(Exception):
    """Raised when the 24-hour trade limit makes retrying pointless."""
    pass


_CACHE_SUBMIT_PERMIT_TTL_SEC = binance_cache_health.SUBMIT_PERMIT_TTL_SEC
_CACHE_SUBMIT_PERMIT_ISSUER = object()
_INVALID_BAPI_ID_STRINGS = {"", "0", "0.0", "-0", "+0", "nan", "inf", "-inf", "+inf", "none", "null"}


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
            expires_at=issued_at + _CACHE_SUBMIT_PERMIT_TTL_SEC,
            expires_at_wall=issued_at_wall + _CACHE_SUBMIT_PERMIT_TTL_SEC,
            order_cache_version=str(status.order_cache_version),
            trade_cache_version=str(status.trade_cache_version),
            symbol=str(symbol).upper(),
            side=str(side).upper(),
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


def _normalize_side(order_type):
    if isinstance(order_type, str):
        cleaned = order_type.strip().upper()
        if cleaned == "B":
            return "BUY"
        if cleaned == "S":
            return "SELL"
        return cleaned
    return order_type


def _resolve_qty(qty):
    """Normalize the quantity model used by the placement pipeline.

    ``qty=None`` means use the maximum allowed by weight limits and the real-balance
    clamp rather than an arbitrary numeric placeholder. An explicit quantity remains
    unchanged here; downstream safety guards may still cap it without changing the
    caller's stated intent.
    """
    return float("inf") if qty is None else qty


def _fresh_price(symbol):
    """Return the freshest WS price, falling back to ``bapi.get_current_price``."""
    try:
        import cacheManager as cm
        p = cm.get_current_price_manager().get_price_value(symbol)
        if p is not None and not isinstance(p, bool):
            pf = float(p)
            if math.isfinite(pf) and pf > 0:
                return pf
    except Exception:
        pass
    return api.get_current_price(symbol)



def apply_weight_limit(symbol, order_type, price, required_qty, available_qty):
    from . import bapi_allorders as apiorders
    auto_qty = required_qty is None
    if isinstance(order_type, bool) or not isinstance(order_type, str):
        raise SubmissionRefused("invalid_weight_policy_inputs")
    order_type = _normalize_side(order_type)
    if order_type not in {"BUY", "SELL"}:
        raise SubmissionRefused("invalid_weight_policy_inputs")
    if (
        isinstance(price, bool)
        or isinstance(available_qty, bool)
        or isinstance(required_qty, bool)
    ):
        raise SubmissionRefused("invalid_weight_policy_inputs")
    try:
        price = float(price)
        available_qty = float(available_qty)
        required_qty = float(_resolve_qty(required_qty))
    except (TypeError, ValueError, OverflowError) as exc:
        raise SubmissionRefused("invalid_weight_policy_inputs") from exc
    if (not math.isfinite(price) or price <= 0 or
            not math.isfinite(available_qty) or available_qty <= 0 or
            math.isnan(required_qty) or required_qty <= 0 or
            (math.isinf(required_qty) and not auto_qty)):
        raise SubmissionRefused("invalid_weight_policy_inputs")
    # ``qty=None`` is represented by positive infinity until policy and balance
    # caps are known. Resolve only that internal sentinel to the already-validated
    # finite balance cap; explicit invalid values still fail closed above.
    if auto_qty:
        required_qty = available_qty

    is_sell = order_type == "SELL"

    def _fail_open_or_refuse(reason, exc):
        # An UNAVAILABLE policy is infrastructure downtime, not a legitimate cap of
        # zero. Failing closed is prudent for a BUY (never add exposure we cannot
        # verify), but trapping a SELL for the whole retry TTL is dangerous -- it can
        # leave a position unable to exit for a full day. A STOP/trailing exit is
        # already exempt upstream (apply_policy=False); a plain SELL is likewise
        # exposure-reducing, so fail OPEN to the already-validated balance cap. The
        # per-side throttle still applies whenever the policy returns a usable value;
        # only downtime is prevented from blocking an exit.
        if is_sell:
            print(f"apply_weight_limit -> SELL {symbol}: {reason} ({exc}); failing "
                  f"OPEN to balance cap {available_qty:.8f} so the exit is not trapped.")
            return available_qty
        raise SubmissionRefused(reason) from exc

    try:
        weight = order_guard.resolve_trade_weight(
            symbol, order_type, provider_name="binance", raise_on_error=True)
    except SubmissionRefused:
        raise
    except Exception as exc:
        return _fail_open_or_refuse("weight_policy_unavailable", exc)

    if not math.isfinite(weight) or not 0 < weight <= 1:
        raise SubmissionRefused("invalid_weight_policy_weight")

    try:
        stats = apiorders.get_total_traded_stats(symbol)
        side_stats = stats[order_type.upper()]
        traded_value = float(side_stats["total_value"])
    except Exception as exc:
        return _fail_open_or_refuse("trade_stats_unavailable", exc)
    if not math.isfinite(traded_value) or traded_value < 0:
        raise SubmissionRefused("invalid_trade_stats")

    adjusted_qty = order_guard.compute_weight_capped_qty(
        weight=weight,
        price=price,
        required_qty=required_qty,
        available_qty=available_qty,
        traded_24h_value=traded_value,
    )
    total_value_reference = traded_value + available_qty * price
    max_trade_value = total_value_reference * weight
    remaining_trade_value = max(0.0, max_trade_value - traded_value)
    remaining_trade_qty = remaining_trade_value / price
    if not all(math.isfinite(value) for value in (
            total_value_reference, max_trade_value, remaining_trade_value,
            remaining_trade_qty, adjusted_qty)):
        raise SubmissionRefused("invalid_weight_policy_result")

    print(f"apply_weight_limit → {order_type} {symbol}, "
          f"Available qty {available_qty:.8f}, "
          f"Weight {weight}, "
          f"Traded in 24h {traded_value:.2f} USDC, "
          f"Max trade allowed (24h): {max_trade_value:.2f} USDC, "
          f"Remaining: {remaining_trade_value:.2f} USDC, "
          f"Required qty: {required_qty:.8f}, "
          f"Final qty: {adjusted_qty:.8f}")
    return adjusted_qty


def require_account_cache_for_submit():
    """Fail closed unless the central Binance account caches are current."""
    if not cfg.is_trade_enabled():
        raise SubmissionRefused("trading_disabled")
    try:
        import cacheManager as cm
        # A Trade history reload can take long enough for the original marker to
        # become stale or advance. Re-read it after loading and accept only the
        # exact versions now present in memory. One bounded retry tolerates a writer
        # publication that races the first load without creating a wait loop.
        for _attempt in range(2):
            requested = binance_cache_health.require_fresh_account_cache()
            cm.ensure_account_cache_readers(requested)
            confirmed = binance_cache_health.require_fresh_account_cache()
            requested_versions = (
                requested.order_cache_version,
                requested.trade_cache_version,
            )
            confirmed_versions = (
                confirmed.order_cache_version,
                confirmed.trade_cache_version,
            )
            if requested_versions == confirmed_versions:
                return confirmed
        raise binance_cache_health.AccountCacheNotReady(
            "account_cache_changed_during_reader_sync")
    except binance_cache_health.AccountCacheNotReady as exc:
        print(f"[BINANCE][GUARD] account cache is not fresh: {exc.reason}")
        raise SubmissionRefused("account_cache_not_fresh") from exc


def _require_submit_permit_headroom(status):
    cache_ages = (status.order_age_sec, status.trade_age_sec)
    parsed_ages = []
    for age in cache_ages:
        if age is None or isinstance(age, bool):
            raise SubmissionRefused("account_cache_not_fresh")
        try:
            age_f = float(age)
        except (TypeError, ValueError, OverflowError) as exc:
            raise SubmissionRefused("account_cache_not_fresh") from exc
        if not math.isfinite(age_f) or age_f < 0.0:
            raise SubmissionRefused("account_cache_not_fresh")
        parsed_ages.append(age_f)
    if (
        max(parsed_ages) + _CACHE_SUBMIT_PERMIT_TTL_SEC
        > binance_cache_health.MAX_AGE_SEC
    ):
        # Refuse before cancellation unless the validated snapshot remains inside
        # its freshness time bound for the permit's entire configured lifetime.
        raise SubmissionRefused("account_cache_not_fresh")
    return status


def issue_account_cache_submit_permit(symbol, side, qty, price=None, *,
                                      market=False, kind=None, api_client=None):
    """Authorize one narrowly scoped submit after a complete cache validation."""
    market = _parse_bool_flag(market, False)
    if side is None or isinstance(side, bool) or symbol is None or isinstance(symbol, bool):
        raise SubmissionRefused("invalid_order_parameters")
    side = str(side).strip().upper()
    if side == "B":
        side = "BUY"
    elif side == "S":
        side = "SELL"
    symbol = str(symbol).strip().upper()
    if isinstance(qty, bool) or (not market and isinstance(price, bool)):
        raise SubmissionRefused("invalid_order_parameters")
    try:
        qty = float(qty)
        price = None if market else float(price)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SubmissionRefused("invalid_order_parameters") from exc
    if (side not in {"BUY", "SELL"} or not symbol
            or not math.isfinite(qty) or qty <= 0
            or (not market and (not math.isfinite(price) or price <= 0))):
        raise SubmissionRefused("invalid_order_parameters")
    if not cfg.is_trade_enabled():
        raise SubmissionRefused("trading_disabled")
    # Reject near-stale state before metadata I/O, then charge that I/O by
    # validating again immediately before the permit is constructed.
    _require_submit_permit_headroom(require_account_cache_for_submit())
    price_tick_size = 0.0
    if not market:
        try:
            from providers.binance_filters import BinanceOrderRules
            target_client = api_client or client
            rules = BinanceOrderRules.from_symbol_info(
                target_client.get_symbol_info(symbol))
            price_tick_size = float(rules.tick_size)
        except Exception as exc:
            raise SubmissionRefused("binance_symbol_rules_unavailable") from exc
    status = _require_submit_permit_headroom(
        require_account_cache_for_submit())
    return _AccountCacheSubmitPermit(
        _CACHE_SUBMIT_PERMIT_ISSUER,
        status=status,
        symbol=symbol,
        side=side,
        qty=qty,
        price=price,
        market=market,
        price_tick_size=price_tick_size,
        kind=kind,
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
        # Burn malformed or expired permits as well, preventing later reuse.
        object.__setattr__(permit, "_consumed", True)
        scope = permit._scope
        if (
            isinstance(qty, bool)
            or (not market and (isinstance(actual_price, bool) or isinstance(requested_price, bool)))
        ):
            raise SubmissionRefused("account_cache_not_fresh")
        try:
            actual_qty = float(qty)
            if market:
                submitted_price = scoped_price = None
            else:
                submitted_price = float(actual_price)
                scoped_price = float(requested_price)
        except (TypeError, ValueError, OverflowError) as exc:
            raise SubmissionRefused("account_cache_not_fresh") from exc
        price_is_safe = scope.market or (
            math.isfinite(submitted_price)
            and submitted_price > 0
            and math.isfinite(scoped_price)
            and scoped_price == scope.requested_price
            and ((scope.side == "BUY" and submitted_price <= scoped_price)
                or (
                    scope.side == "SELL"
                    and submitted_price + scope.price_tick_size >= scoped_price
                )
            )
        )
        matches = (
            scope.pid == os.getpid()
            and time.monotonic() <= scope.expires_at
            and time.time() <= scope.expires_at_wall
            and bool(scope.order_cache_version)
            and bool(scope.trade_cache_version)
            and symbol is not None and not isinstance(symbol, bool)
            and side is not None and not isinstance(side, bool)
            and scope.symbol == str(symbol).strip().upper()
            and scope.side == str(_normalize_side(side))
            and scope.market is bool(market)
            and scope.kind == str(kind or "")
            and math.isfinite(actual_qty)
            and 0 < actual_qty <= scope.max_qty
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
    if isinstance(qty, bool) or (not market and isinstance(price, bool)):
        raise SubmissionRefused("invalid_order_parameters")
    try:
        qty_f = float(qty)
        price_f = None if market else float(price)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SubmissionRefused("invalid_order_parameters") from exc
    if (
        not math.isfinite(qty_f)
        or qty_f <= 0
        or (not market and (price_f is None or not math.isfinite(price_f) or price_f <= 0))
    ):
        raise SubmissionRefused("invalid_order_parameters")
    cancel_price = None
    if cancel_opposite_requested_price is not None:
        if isinstance(cancel_opposite_requested_price, bool):
            raise SubmissionRefused("invalid_opposing_cancel_request")
        try:
            cancel_price = float(cancel_opposite_requested_price)
        except (TypeError, ValueError, OverflowError) as exc:
            raise SubmissionRefused("invalid_opposing_cancel_request") from exc
        if not math.isfinite(cancel_price) or cancel_price <= 0:
            raise SubmissionRefused("invalid_opposing_cancel_request")
        if cache_permit is None:
            # Smart cancellation is authorized only as part of the exact
            # permit-bound replacement dispatch.
            raise SubmissionRefused("account_cache_not_fresh")

    target_client = api_client or client
    cleaned_cid = (
        extract_order_id(client_order_id, default=None)
        if not isinstance(client_order_id, (dict, list, tuple, set, frozenset))
        else None
    )
    if cleaned_cid is not None and str(cleaned_cid).strip().lower() in _INVALID_BAPI_ID_STRINGS:
        cleaned_cid = None
    client_order_id = cleaned_cid or rc.create_client_order_id()
    if market:
        method = (
            target_client.order_market_buy
            if side == "BUY" else target_client.order_market_sell
        )
        submit_kwargs = {"symbol": symbol, "quantity": qty}
    else:
        method = (
            target_client.order_limit_buy
            if side == "BUY" else target_client.order_limit_sell
        )
        submit_kwargs = {
            "symbol": symbol, "quantity": qty, "price": str(price)}
    if cache_permit is None:
        if side not in {"BUY", "SELL"} or not cfg.is_trade_enabled():
            return None
        require_account_cache_for_submit()
        if not cfg.is_trade_enabled():
            raise SubmissionRefused("trading_disabled")
    else:
        _consume_account_cache_submit_permit(
            cache_permit,
            symbol=symbol,
            side=side,
            qty=qty,
            actual_price=price,
            requested_price=permit_requested_price,
            market=market,
            kind=kind,
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


def place_BUY_order(symbol, price, qty, client_order_id=None):
    return _submit_binance_order(
        "BUY", symbol, qty, price=price, client_order_id=client_order_id)

def place_SELL_order(symbol, price, qty, client_order_id=None):
    return _submit_binance_order(
        "SELL", symbol, qty, price=price, client_order_id=client_order_id)


def place_BUY_order_at_market(symbol, qty, client_order_id=None):
    return _submit_binance_order(
        "BUY", symbol, qty, market=True, client_order_id=client_order_id)


def place_SELL_order_at_market(symbol, qty, client_order_id=None):
    return _submit_binance_order(
        "SELL", symbol, qty, market=True, client_order_id=client_order_id)


def _last_opposite_fill_price(symbol, order_type):
    """Return the latest opposite fill price without a time limit.

    BUY uses the latest SELL fill and SELL uses the latest BUY fill. Return ``None``
    only when the cache is healthy but has no opposite fill. Raise when the manager or
    cache is unavailable so the caller can fail closed. CacheTradeManager supplies real
    WebSocket fills without an API call.
    """
    import cacheManager as cm
    # Retry workers only read the persisted/WS-produced trade cache. Starting another
    # API polling loop here duplicates cacheManager's work and contaminates worker logs.
    return cm.get_cache_manager("Trade", start_sync=False).get_last_opposite_fill_price(symbol, order_type)


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


def if_place_safe_order(order_type, symbol, price, qty, time_back_in_seconds,
                        *, regime_context=None, bypass_guards=None, **kwargs):
    """Legacy preflight safety check for Binance orders.

    Validates daily trade counts, dynamic market regimes,
    and historical price profit references before order placement.
    """
    from . import bapi_allorders as apiorders
    from guard_hierarchy import GUARD_REGISTRY, normalize_bypassed_guards
    from providers.market_api import BinanceProvider

    order_type = _normalize_side(order_type)
    sym.validate_params(order_type, symbol, price, qty)
    provider = BinanceProvider()
    allow_math_reference_sell = _parse_bool_flag(kwargs.pop("allow_math_reference_sell", False), False)
    m = order_guard._load_margins()
    venue_bypass = normalize_bypassed_guards(
        bypass_guards=m.get("binance_bypass_guards", m["default_bypass_guards"])
    )
    bypassed = normalize_bypassed_guards(bypass_guards=bypass_guards) | venue_bypass
    bypass_all = "all" in bypassed or "*" in bypassed
    bypass_profit_guard = "profit_guard" in bypassed or bypass_all
    skip_math_profit_ref = GUARD_REGISTRY["G_PROF"].is_bypassed(bypassed) or bypass_profit_guard
    g_prof_revoked = False
    if skip_math_profit_ref and not bypass_profit_guard and order_type != "BUY" and not allow_math_reference_sell:
        print(
            f"[GUARD] {order_type} {symbol}: bypass math_profit/math_reference ignored; "
            "is allowed only for BUY")
        if bypassed.intersection({"financial", "gf"}):
            bypassed.update({"g_qlimit1", "g_qlimit2"})
        bypassed.difference_update(GUARD_REGISTRY["G_PROF"].aliases)
        bypassed.discard(GUARD_REGISTRY["G_PROF"].canonical)
        bypassed.discard(GUARD_REGISTRY["G_PROF"].code.lower())
        bypassed.difference_update({"profit", "financial", "gf"})
        skip_math_profit_ref = False
        g_prof_revoked = True
    bypass_pg = (
        bypass_all
        or (
            skip_math_profit_ref
            and all(
                GUARD_REGISTRY[gid].is_bypassed(bypassed)
                for gid in ("GT_L", "G_QLIMIT2", "GI_OBOOK", "GI_WHALE", "GI_FUNDING", "GLLM1", "GLLM2", "GLLM3")
            )
        )
    )

    try:
        current_price = api.get_current_price(symbol)
        cp_val = None
        if current_price is not None and not isinstance(current_price, bool):
            try:
                cand_cp = float(current_price)
                if math.isfinite(cand_cp) and cand_cp > 0:
                    cp_val = cand_cp
            except (TypeError, ValueError, OverflowError):
                cp_val = None
        price_f = float(price)
        if cp_val is not None:
            if order_type == "BUY":
                price = round(min(price_f, cp_val), 8)
            else:  # SELL
                price = round(max(price_f, cp_val), 8)
        else:
            price = round(price_f, 8)

        qty = round(float(qty), 8)
        opposite_order_type = "SELL" if order_type == "BUY" else "BUY"

        # Delegate daily trade cap to order_guard.daily_limit_guard
        ok, reason = order_guard.daily_limit_guard(
            provider, symbol, order_type,
            bypass_guards=bypassed)
        if not ok:
            return False, reason

        if not bypass_pg:
            caller_module = order_guard.resolve_caller_module(
                kwargs.get("module"),
                prefix=kwargs.get("prefix"),
                client_order_id=kwargs.get("client_order_id"),
            )
            raw_ref_mode = kwargs.get("reference_mode")
            profit_ref_mode = (
                raw_ref_mode.strip()
                if isinstance(raw_ref_mode, str) and raw_ref_mode.strip()
                else None
            )
            eff_ref_mode = order_guard.get_profit_guard_mode_cfg(
                "binance",
                order_type,
                module=caller_module,
                reference_mode=profit_ref_mode,
            )

            if regime_context is None:
                from providers.market_api import api as market_api
                regime_context = market_api.market_regime_context(
                    symbol, provider_name="binance")

            window_ref = None
            if not skip_math_profit_ref and eff_ref_mode != "off":
                ref_window = order_guard.get_window_cfg(
                    "binance",
                    symbol=symbol,
                    order_type=order_type,
                    regime_context=regime_context,
                    module=caller_module,
                    reference_mode=eff_ref_mode,
                )
                tb_sec = 0.0
                if time_back_in_seconds is not None and not isinstance(time_back_in_seconds, bool):
                    try:
                        tbf = float(time_back_in_seconds)
                        if math.isfinite(tbf) and tbf > 0.0:
                            tb_sec = tbf
                    except (TypeError, ValueError, OverflowError):
                        tb_sec = 0.0
                if not ref_window or not math.isfinite(float(ref_window)) or float(ref_window) <= 0:
                    ref_window = tb_sec if tb_sec > 0.0 else float(PLACE_ORDER_SAFEBACK_SEC)

                fetch_window = max(tb_sec, float(ref_window or 0))
                time_limit = float(time.time() * 1000) - (ref_window * 1000)
                oposite_trades = apiorders.get_trade_orders(opposite_order_type, symbol, max_age_seconds=fetch_window) or []
                recent_opposite_trades = [trade for trade in oposite_trades
                                          if (extract_timestamp_ms(trade, default=0) or 0) >= time_limit
                                          and extract_order_price(trade, default=0.0) > 0]

                if recent_opposite_trades:
                    if len(recent_opposite_trades) > 1:
                        same_trades = apiorders.get_trade_orders(order_type, symbol, max_age_seconds=fetch_window) or []
                        recent_same_trades = [
                            trade for trade in same_trades
                            if (extract_timestamp_ms(trade, default=0) or 0) >= time_limit
                            and extract_order_price(trade, default=0.0) > 0
                        ]
                        window_ref = order_guard.active_cycle_window_ref(
                            recent_opposite_trades,
                            recent_same_trades,
                            order_type,
                            reference_mode=eff_ref_mode,
                        )
                    elif eff_ref_mode in ("average", "last", "hybrid"):
                        window_ref = order_guard.active_cycle_window_ref(
                            recent_opposite_trades,
                            None,
                            order_type,
                            reference_mode=eff_ref_mode,
                        )
                    else:
                        _prices = [extract_order_price(t) for t in recent_opposite_trades]
                        window_ref = order_guard.select_extreme_reference(_prices, order_type, strict=True)

            pg_kwargs = dict(
                window_ref=window_ref,
                regime_context=regime_context,
                qty=qty,
                reference_mode=eff_ref_mode,
            )
            if caller_module is not None:
                pg_kwargs["module"] = caller_module
            if g_prof_revoked or venue_bypass:
                pg_kwargs["bypass_guards"] = bypassed
            elif bypass_guards is not None:
                pg_kwargs["bypass_guards"] = bypass_guards
            elif bypassed:
                pg_kwargs["bypass_guards"] = bypassed

            if not order_guard.profit_guard(
                provider,
                symbol,
                order_type,
                price,
                order_guard.get_margin_cfg("binance"),
                **pg_kwargs,
            ):
                return False, "profit_guard"
        return True, None

    except BinanceAPIException as e:
        print(f"Error checking whether the {order_type} order is safe: {e}")
        return False, "guard_check_api_exception"
    except Exception as e:
        print(f"[GUARD] {order_type} {symbol}: the check failed ({e}) -> "
              f"{'PASSING (bypass)' if bypass_pg else 'BLOCKED (fail-closed)'}")
        return bool(bypass_pg), (None if bypass_pg else "guard_check_failed")




def _guarded_market_place(symbol, order_type, price, qty, **kwargs):
    """Import lazily to avoid the bapi_placeorder/market_api cycle.

    This testable choke point routes legacy API orders through the same
    ``Instrument.place`` pipeline used by rtrade, tradeall, and other providers.
    """
    from providers.market_api import api as market_api
    return market_api.place(symbol, order_type, price, qty, **kwargs)


def place_safe_order(order_type, symbol, price, qty=None,
                     safeback_seconds=PLACE_ORDER_SAFEBACK_SEC, force=False,
                     cancelorders=False, hours=PLACE_ORDER_HOURS,
                     _reason_out=None, **kwargs):
    """Adapt the compatible safe API to the common pipeline without smart repricing."""
    order_type = _normalize_side(order_type)
    # ``None`` is the legacy "maximum permitted" request. Preserve it until the
    # shared QuantityDecision has loaded balance and policy caps; a synthetic
    # infinity here would be indistinguishable from an invalid explicit value.
    sym.validate_params(
        order_type, symbol, price, 1.0 if qty is None else qty)
    call_kwargs = dict(kwargs)
    if isinstance(_reason_out, dict) and "_outcome_context" not in call_kwargs:
        call_kwargs["_outcome_context"] = _reason_out
    order = _guarded_market_place(
        symbol, order_type, price, qty,
        smart=False,
        safeback_seconds=safeback_seconds,
        force=force,
        cancelorders=cancelorders,
        hours=hours,
        **call_kwargs,
    )
    # The common pipeline logs the exact reason. Retain dict compatibility without
    # duplicating guard evaluation in the legacy chain.
    if order is None and _reason_out is not None:
        if not _reason_out.get("reason"):
            _reason_out["reason"] = "common_pipeline_refused"
    return order
    

# The fleet-wide journal lives in order_outcomes_log.py as a single source used by
# Instrument.place for every provider. Re-export the directory for backward compatibility.
import order_outcomes_log as _outcomes_log
ORDER_OUTCOMES_LOG_DIR = _outcomes_log.ORDER_OUTCOMES_LOG_DIR


def _log_order_outcome(symbol, side, price, qty, outcome, refuse_reason, motivation):
    """Write one pipe-delimited fleet-wide record per placement attempt.

    This is observational and cannot affect the caller's return value because the
    logging implementation contains its own exception boundary.
    """
    try:
        caller = os.path.basename(sys._getframe(2).f_code.co_filename)
    except Exception:
        caller = None
    _outcomes_log.log_order_outcome(symbol, side, price, qty, outcome, refuse_reason,
                                    motivation, caller=caller)


# Retain ``pair`` only for compatibility with callers that explicitly pass it. It is
# intentionally ignored legacy code; do not remove it without updating those callers.
def place_order_smart(order_type, symbol, price, qty=None, safeback_seconds=PLACE_ORDER_SAFEBACK_SEC, force=False, cancelorders=True, hours=PLACE_ORDER_HOURS, pair=None, motivation=None, _reason_out=None, **kwargs):
    order_type = _normalize_side(order_type)
    sym.validate_params(
        order_type, symbol, price, 1.0 if qty is None else qty)
    call_kwargs = dict(kwargs)
    if isinstance(_reason_out, dict) and "_outcome_context" not in call_kwargs:
        call_kwargs["_outcome_context"] = _reason_out
    order = _guarded_market_place(
        symbol, order_type, price, qty,
        smart=True,
        safeback_seconds=safeback_seconds,
        force=force,
        cancelorders=cancelorders,
        hours=hours,
        motivation=motivation,
        **call_kwargs,
    )
    if order is None and isinstance(_reason_out, dict):
        if not _reason_out.get("reason"):
            _reason_out["reason"] = "common_pipeline_refused"
    return order


# ============================================================================
# Binance placement mechanics called through Instrument.place and BinanceProvider.
# These functions contain only venue-specific price adjustment, opposite-order cleanup,
# fee/balance and minimum-notional clamps, and limit/market dispatch. Daily limits,
# profit/weight/trend/cooldown guards, and journaling belong to the agnostic layer.
# ============================================================================

def cancel_opposite_orders(order_type, symbol, requested_price, allowed_owners=None):
    """Cancel adverse opposing Binance orders without changing the target price."""
    if isinstance(order_type, bool) or isinstance(requested_price, bool):
        raise SubmissionRefused("invalid_opposing_cancel_request")
    order_type = str(order_type or "").strip().upper()
    if order_type == "B":
        order_type = "BUY"
    elif order_type == "S":
        order_type = "SELL"
    try:
        req_px = float(requested_price)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SubmissionRefused("invalid_opposing_cancel_request") from exc
    if not math.isfinite(req_px) or req_px <= 0:
        raise SubmissionRefused("invalid_opposing_cancel_request")
    if order_type == "BUY":
        try:
            if allowed_owners is not None:
                open_orders = api.get_open_orders("SELL", symbol, strict=True, allowed_owners=allowed_owners)
            else:
                open_orders = api.get_open_orders("SELL", symbol, strict=True)
        except Exception as exc:
            raise SubmissionRefused(
                "opposing_order_discovery_unavailable") from exc
        for order_id, order_details in open_orders.items():
            order_px = extract_order_price(order_details, default=0.0)
            if order_px > 0 and order_px < req_px:
                try:
                    canceled = api.cancel_order(symbol, order_id)
                except Exception as exc:
                    raise SubmissionRefused(
                        "opposing_cancel_unconfirmed") from exc
                if not canceled:
                    raise SubmissionRefused("opposing_cancel_unconfirmed")
                try:
                    import order_retry
                    order_retry.resolve_accepted_order(symbol, order_id)
                except Exception:
                    pass
    elif order_type == "SELL":
        try:
            if allowed_owners is not None:
                open_orders = api.get_open_orders("BUY", symbol, strict=True, allowed_owners=allowed_owners)
            else:
                open_orders = api.get_open_orders("BUY", symbol, strict=True)
        except Exception as exc:
            raise SubmissionRefused(
                "opposing_order_discovery_unavailable") from exc
        for order_id, order_details in open_orders.items():
            order_px = extract_order_price(order_details, default=0.0)
            if order_px > 0 and order_px > req_px:
                try:
                    canceled = api.cancel_order(symbol, order_id)
                except Exception as exc:
                    raise SubmissionRefused(
                        "opposing_cancel_unconfirmed") from exc
                if not canceled:
                    raise SubmissionRefused("opposing_cancel_unconfirmed")
                try:
                    import order_retry
                    order_retry.resolve_accepted_order(symbol, order_id)
                except Exception:
                    pass


def adjust_price_and_cancel_opposite(order_type, symbol, price,
                                     cancel_opposite=True):
    """Apply Binance price mechanics and optionally cancel opposing orders."""
    cancel_opposite = _parse_bool_flag(cancel_opposite, True)
    if isinstance(order_type, bool) or isinstance(price, bool):
        raise SubmissionRefused("invalid_order_parameters")
    order_type = str(order_type or "").strip().upper()
    if order_type == "B":
        order_type = "BUY"
    elif order_type == "S":
        order_type = "SELL"
    try:
        price = float(price)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SubmissionRefused("invalid_order_parameters") from exc
    if not math.isfinite(price) or price <= 0:
        raise SubmissionRefused("invalid_order_parameters")
    if cancel_opposite:
        cancel_opposite_orders(order_type, symbol, price)
    current_price = api.get_current_price(symbol)
    try:
        cp = (
            float(current_price)
            if current_price is not None and not isinstance(current_price, bool)
            else None
        )
    except (TypeError, ValueError, OverflowError):
        cp = None
    if cp is not None and math.isfinite(cp) and cp > 0:
        if order_type == "BUY":
            price = min(price, cp) * 0.999
        elif order_type == "SELL":
            price = max(price, cp) * (1 + 0.001)
    try:
        from providers.market_api import api as mkt
        price = mkt.round_price(symbol, price)
    except Exception:
        price = round(price, 8)
    return price


def order_candidate_price(order_type, requested_price, current_price, *, market=False):
    """Return the exact price used by Binance filter normalization."""
    if _parse_bool_flag(market, False):
        return None
    if isinstance(order_type, bool):
        raise ValueError("order_type must be BUY or SELL")
    side = str(order_type).strip().upper()
    if side == "B":
        side = "BUY"
    elif side == "S":
        side = "SELL"
    if side not in {"BUY", "SELL"}:
        raise ValueError("order_type must be BUY or SELL")
    if isinstance(requested_price, bool):
        raise ValueError("requested_price cannot be bool")
    try:
        req_f = float(requested_price)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("requested_price must be numeric") from exc
    if not math.isfinite(req_f) or req_f <= 0:
        raise ValueError("requested_price must be finite and positive")
    requested_price = req_f
    if current_price is None or isinstance(current_price, bool):
        return requested_price
    try:
        cp = float(current_price)
    except (TypeError, ValueError, OverflowError):
        return requested_price
    if not math.isfinite(cp) or cp <= 0:
        return requested_price
    if side == "SELL":
        return max(requested_price, cp)
    return min(requested_price, cp)


def place_order_mechanics(order_type, symbol, price, qty, force=False,
                          client_order_id=None, cache_permit=None,
                          permit_requested_price=None, kind=None,
                          cancel_opposite_requested_price=None,
                          enforce_business_minimum=True, market=False, **kwargs):
    """Execute Binance-specific submission mechanics.

    Clamp to real balance after fees, enforce configured order filters, round,
    and dispatch a limit or market order. ``qty`` already comes from QuantityDecision.
    Weight, trend, cooldown, and other guards belong to the agnostic layer.
    Instrument.place holds the RAII cooldown around this call. Return an accepted
    order, or raise ``SubmissionRefused`` for a deterministic filter violation.
    """
    order_type = _normalize_side(order_type)
    sym.validate_params(order_type, symbol, price, qty)
    price = float(price)
    qty = float(qty)
    enforce_business_minimum = _parse_bool_flag(enforce_business_minimum, True)
    is_market = (
        _parse_bool_flag(force, False)
        or _parse_bool_flag(market, False)
        or _parse_bool_flag(kwargs.get("market", False), False)
        or _parse_bool_flag(kwargs.get("force", False), False)
    )
    try:
        from providers.quantity import balance_cap_quantity, fee_cap_quantity
        balance_fn = (
            (lambda _asset: kwargs["known_balance"])
            if kwargs.get("known_balance") is not None
            else api.get_free_balance
        )
        available_qty, _balance_asset = balance_cap_quantity(
            balance_fn, symbol, order_type, price)
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
                print(f"Adjusting {order_type} qty from {qty:.8f} to "
                      f"{fee_cap:.8f} to cover balance and fees")
                qty = fee_cap

        current_price = api.get_current_price(symbol)
        cp_val = None
        if current_price is not None and not isinstance(current_price, bool):
            try:
                cand_cp = float(current_price)
                if math.isfinite(cand_cp) and cand_cp > 0:
                    cp_val = cand_cp
            except (TypeError, ValueError, OverflowError):
                cp_val = None
        ref_price = cp_val if cp_val is not None else price
        from providers.binance_filters import (
            BinanceFilterError,
            BinanceOrderRules,
            binance_filter_refusal_reason,
        )
        rules = BinanceOrderRules.from_symbol_info(client.get_symbol_info(symbol))
        candidate_price = order_candidate_price(
            order_type, price, current_price, market=is_market)
        try:
            normalized_qty, normalized_price = rules.normalize(
                quantity=qty,
                price=candidate_price,
                market=is_market,
                reference_price=ref_price,
                business_min_notional=(
                    PLACE_ORDER_MIN_NOTIONAL
                    if enforce_business_minimum else 0),
            )
        except BinanceFilterError as exc:
            raise SubmissionRefused(
                binance_filter_refusal_reason(exc)) from exc
        qty = float(normalized_qty)
        if normalized_price is not None:
            price = float(normalized_price)

        print(f"Trying to place {order_type} {symbol} qty {qty:.8f} at "
              f"{'market price' if is_market else f'price {price}'}")
        if order_type in {"BUY", "SELL"}:
            return _submit_binance_order(
                order_type, symbol, qty, price=None if is_market else price,
                market=is_market, client_order_id=client_order_id,
                cache_permit=cache_permit,
                permit_requested_price=permit_requested_price, kind=kind,
                cancel_opposite_requested_price=(
                    cancel_opposite_requested_price))
        print(f"Invalid order type: {order_type}")
        return None
    except BinanceAPIException as e:
        print(f"Error placing {order_type} order (mechanics): {e}")
        return None
