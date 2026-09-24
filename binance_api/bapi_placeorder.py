import os
import time
import datetime
import math
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta

import signal
import asyncio
#import threading
#from threading import Thread
import json

####Binance
import binance
print(binance.__version__)
from binance.exceptions import BinanceAPIException


####MYLIB
import utils as u
import order_guard
import symbols as sym
import config as cfg
import priceAnalysis as pa
from . import order_id_context as rc   # client_order_id and tag context moved into binance_api.

from . import bapi as api
from .bapi_client import client
from lock import trade_cooldown   # Rapid-fire gate moved into the lock package.
from providers.strategy_executor import SubmissionRefused
import binance_cache_health

# Load versioned, non-secret tuning parameters before reading the environment below.
# ``botcore.load_dotenv`` never overwrites variables already set in the real environment;
# it only fills missing values, matching tradeall_config.env and monitortrades_config.env.
from botcore import load_dotenv as _load_dotenv, required_float_env, required_int_env
_load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "bapi_placeorder_config.env"))

# Order policy is mandatory; missing configuration aborts startup.
PLACE_ORDER_FEE_PCT = required_float_env("PLACE_ORDER_FEE_PCT")
PLACE_ORDER_HOURS = required_int_env("PLACE_ORDER_HOURS")
PLACE_ORDER_SAFEBACK_SEC = required_int_env("PLACE_ORDER_SAFEBACK_SEC")
PLACE_ORDER_MAX_DAILY_TRADES = required_int_env("PLACE_ORDER_MAX_DAILY_TRADES")
PLACE_ORDER_MIN_NOTIONAL = required_float_env("PLACE_ORDER_MIN_NOTIONAL")


class WeightLimitBlock(Exception):
    """Raised when the 24-hour trade limit makes retrying pointless."""
    pass


_CACHE_SUBMIT_PERMIT_TTL_SEC = binance_cache_health.SUBMIT_PERMIT_TTL_SEC
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
            requested_price=None if market else float(price),
            price_tick_size=float(price_tick_size),
            market=bool(market),
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
        if p is not None:
            return p
    except Exception:
        pass
    return api.get_current_price(symbol)



def apply_weight_limit(symbol, order_type, price, required_qty, available_qty):
    from . import bapi_allorders as apiorders
    auto_qty = required_qty is None
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

    is_sell = str(order_type).upper() == "SELL"

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
        weight = float(pa.get_weight_for_cash_permission_at_quant_time(
            symbol, order_type))
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

    total_value_reference = traded_value + available_qty * price
    max_trade_value = total_value_reference * weight
    remaining_trade_value = max(0.0, max_trade_value - traded_value)
    remaining_trade_qty = remaining_trade_value / price
    adjusted_qty = min(required_qty, remaining_trade_qty)
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
    if (any(age is None for age in cache_ages)
            or max(float(age) for age in cache_ages)
            + _CACHE_SUBMIT_PERMIT_TTL_SEC
            > binance_cache_health.MAX_AGE_SEC):
        # Refuse before cancellation unless the validated snapshot remains inside
        # its freshness time bound for the permit's entire configured lifetime.
        raise SubmissionRefused("account_cache_not_fresh")
    return status


def issue_account_cache_submit_permit(symbol, side, qty, price=None, *,
                                      market=False, kind=None, api_client=None):
    """Authorize one narrowly scoped submit after a complete cache validation."""
    side = str(side).upper()
    symbol = str(symbol).upper()
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
    if not isinstance(permit, _AccountCacheSubmitPermit):
        raise SubmissionRefused("account_cache_not_fresh")
    with permit._lock:
        if permit._consumed:
            raise SubmissionRefused("account_cache_not_fresh")
        # Burn malformed or expired permits as well, preventing later reuse.
        object.__setattr__(permit, "_consumed", True)
        scope = permit._scope
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
            and scope.symbol == str(symbol).upper()
            and scope.side == str(side).upper()
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
    side = str(order_type).upper()
    cancel_price = None
    if cancel_opposite_requested_price is not None:
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
    client_order_id = client_order_id or rc.create_client_order_id()
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
            print(f"{side} order placed successfully: {order['orderId']} clientId {client_order_id}")
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


def place_SELL_BUY_order(order_type, symbol, price, qty) :
    return _submit_binance_order(order_type, symbol, qty, price=price)

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
    return cm.get_cache_manager("Trade", start_sync=False).last_opposite_fill_price(symbol, order_type)


def _last_opposite_fill_price_api(symbol, order_type):
    """Query ``get_my_trades`` when the cache has no opposite fill.

    This covers a new symbol or an unpopulated cache. Raise on API errors so the caller
    fails closed; return ``None`` only when Binance confirms no opposite fill exists.
    """
    want_buyer = (order_type.upper() == "SELL")   # The opposite of SELL is BUY (isBuyer=True).
    for tr in reversed(client.get_my_trades(symbol=symbol, limit=200)):
        if tr["isBuyer"] == want_buyer:
            return float(tr["price"])
    return None


def if_place_safe_order(order_type, symbol, price, qty, time_back_in_seconds,
                        bypass_profit_guard=False):
    # Removed dead max_daily_trades and profit_percentage parameters. The only real caller
    # always supplied these exact configured values, now read here as the source of truth.
    # ``bypass_profit_guard=True`` skips profit/history and fail-closed checks, unlike
    # ``force``, which only selects MARKET execution. Daily limits and anti-spam remain.
    # The crash circuit breaker uses both flags; normal trading keeps this bypass disabled.
    #import bapi_trades as apitrades
    from . import bapi_allorders as apiorders
    from providers.market_api import BinanceProvider

    order_type = order_type.upper()
    sym.validate_params(order_type, symbol, price, qty)
    #apitrades.compare_trade_sources(symbol, order_type=order_type, max_age_seconds=time_back_in_seconds, limit=1000)
    provider = BinanceProvider()

    try:

        current_price = api.get_current_price(symbol)

        if order_type == "BUY":
            price = round(min(price, current_price), 0)
        else:  # SELL
            price = round(max(price, current_price), 0)

        qty = round(qty, 4)

        opposite_order_type = "SELL" if order_type == "BUY" else "BUY"
        backdays = math.ceil(time_back_in_seconds / 86400)

        # Delegate daily limits and anti-spam to order_guard.daily_limit_guard, removing
        # a duplicate of the provider-neutral logic. BinanceProvider.get_orders wraps
        # the same apiorders data source. Keep Binance's own limit and lookback settings
        # instead of introducing a second potentially divergent order_guard knob.
        ok, reason = order_guard.daily_limit_guard(
            provider, symbol, order_type,
            max_daily_trades=PLACE_ORDER_MAX_DAILY_TRADES,
            safeback_sec=time_back_in_seconds)
        if not ok:
            return False, reason

        oposite_trades = apiorders.get_trade_orders(opposite_order_type, symbol, max_age_seconds=time_back_in_seconds)  # current data
        print(f"I have {len(oposite_trades)} trades of type {opposite_order_type} for {backdays} days. ")

        ref_window = order_guard.window_for("binance", symbol=symbol, order_type=order_type)
        if not ref_window or ref_window <= 0:
            ref_window = time_back_in_seconds

        time_limit = float(time.time() * 1000) - (ref_window * 1000)  # milliseconds
        # Keep opposite trades in the requested interval. Requiring price > 0 remains a
        # defensive safety net even though canceled orders no longer enter the cache.
        recent_opposite_trades = [trade for trade in oposite_trades
                                  if float(trade['timestamp']) >= float(time_limit)
                                  and float(trade.get('price', 0)) > 0]
        print(f"Considering only those within the last {ref_window:.0f}s, {len(recent_opposite_trades)} of them")
        for trade in recent_opposite_trades:
            readable = datetime.fromtimestamp(trade['timestamp'] / 1000)
            print(f"[CHECK] {readable} - price: {trade['price']} - included: {float(trade['timestamp']) >= time_limit}")
        
        # Provider-neutral profit guard. Reference priority is the window's minimum SELL
        # or maximum BUY, then the provider's latest opposite fill. The bypass skips both;
        # read errors reach the handler below and fail closed.
        if not bypass_profit_guard:
            window_ref = None
            if recent_opposite_trades:                       # Primary time-window reference.
                _prices = [float(t['price']) for t in recent_opposite_trades]
                window_ref = min(_prices) if order_type == "BUY" else max(_prices)
            # Resolve the configured margin lazily only when the guard is used. Reuse the
            # stateless provider already created for the daily-limit guard.
            if not order_guard.profit_guard(provider, symbol, order_type, price,
                                            order_guard.margin_for("binance"), window_ref=window_ref):
                return False, "profit_guard"
        return True, None

    except BinanceAPIException as e:
        print(f"Error checking whether the {order_type} order is safe: {e}")
        return False, "guard_check_api_exception"
    except Exception as e:
        # Data or manager errors fail closed unless the crash circuit breaker explicitly bypasses.
        print(f"[GUARD] {order_type} {symbol}: the check failed ({e}) -> "
              f"{'PASSING (bypass)' if bypass_profit_guard else 'BLOCKED (fail-closed)'}")
        return bool(bypass_profit_guard), (None if bypass_profit_guard else "guard_check_failed")




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
                     bypass_profit_guard=False, _reason_out=None):
    """Adapt the compatible safe API to the common pipeline without smart repricing."""
    order_type = order_type.upper()
    # ``None`` is the legacy "maximum permitted" request. Preserve it until the
    # shared QuantityDecision has loaded balance and policy caps; a synthetic
    # infinity here would be indistinguishable from an invalid explicit value.
    sym.validate_params(
        order_type, symbol, price, 1.0 if qty is None else qty)
    order = _guarded_market_place(
        symbol, order_type, price, qty,
        smart=False,
        safeback_seconds=safeback_seconds,
        force=force,
        cancelorders=cancelorders,
        hours=hours,
        bypass_profit_guard=bypass_profit_guard,
    )
    # The common pipeline logs the exact reason. Retain dict compatibility without
    # duplicating guard evaluation in the legacy chain.
    if order is None and _reason_out is not None:
        _reason_out.setdefault("reason", "common_pipeline_refused")
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
def place_order_smart(order_type, symbol, price, qty=None, safeback_seconds=PLACE_ORDER_SAFEBACK_SEC, force=False, cancelorders=True, hours=PLACE_ORDER_HOURS, pair=None, motivation=None):
    order_type = order_type.upper()
    sym.validate_params(
        order_type, symbol, price, 1.0 if qty is None else qty)
    return _guarded_market_place(
        symbol, order_type, price, qty,
        smart=True,
        safeback_seconds=safeback_seconds,
        force=force,
        cancelorders=cancelorders,
        hours=hours,
        motivation=motivation,
    )


# ============================================================================
# Binance placement mechanics called through Instrument.place and BinanceProvider.
# These functions contain only venue-specific price adjustment, opposite-order cleanup,
# fee/balance and minimum-notional clamps, and limit/market dispatch. Daily limits,
# profit/weight/trend/cooldown guards, and journaling belong to the agnostic layer.
# ============================================================================

def cancel_opposite_orders(order_type, symbol, requested_price):
    """Cancel adverse opposing Binance orders without changing the target price."""
    order_type = order_type.upper()
    if order_type == "BUY":
        try:
            open_orders = api.get_open_orders("SELL", symbol, strict=True)
        except Exception as exc:
            raise SubmissionRefused(
                "opposing_order_discovery_unavailable") from exc
        for order_id, order_details in open_orders.items():
            if order_details["price"] < requested_price:
                try:
                    canceled = api.cancel_order(symbol, order_id)
                except Exception as exc:
                    raise SubmissionRefused(
                        "opposing_cancel_unconfirmed") from exc
                if not canceled:
                    raise SubmissionRefused("opposing_cancel_unconfirmed")
    elif order_type == "SELL":
        try:
            open_orders = api.get_open_orders("BUY", symbol, strict=True)
        except Exception as exc:
            raise SubmissionRefused(
                "opposing_order_discovery_unavailable") from exc
        for order_id, order_details in open_orders.items():
            if order_details["price"] > requested_price:
                try:
                    canceled = api.cancel_order(symbol, order_id)
                except Exception as exc:
                    raise SubmissionRefused(
                        "opposing_cancel_unconfirmed") from exc
                if not canceled:
                    raise SubmissionRefused("opposing_cancel_unconfirmed")


def adjust_price_and_cancel_opposite(order_type, symbol, price,
                                     cancel_opposite=True):
    """Apply Binance price mechanics and optionally cancel opposing orders."""
    order_type = order_type.upper()
    if cancel_opposite:
        cancel_opposite_orders(order_type, symbol, price)
    current_price = api.get_current_price(symbol)
    if order_type == "BUY":
        price = min(price, current_price)
        price = round(price * 0.999, 0)
    elif order_type == "SELL":
        price = max(price, current_price)
        price = round(price * (1 + 0.001), 0)
    return price


def order_candidate_price(order_type, requested_price, current_price, *, market=False):
    """Return the exact price used by Binance filter normalization."""
    if market:
        return None
    side = str(order_type).upper()
    if side == "SELL":
        return max(requested_price, current_price)
    if side == "BUY":
        return min(requested_price, current_price)
    raise ValueError("order_type must be BUY or SELL")


def place_order_mechanics(order_type, symbol, price, qty, force=False,
                          client_order_id=None, cache_permit=None,
                          permit_requested_price=None, kind=None,
                          cancel_opposite_requested_price=None,
                          enforce_business_minimum=True):
    """Execute Binance-specific submission mechanics.

    Clamp to real balance after fees, enforce configured order filters, round,
    and dispatch a limit or market order. ``qty`` already comes from QuantityDecision.
    Weight, trend, cooldown, and other guards belong to the agnostic layer.
    Instrument.place holds the RAII cooldown around this call. Return an accepted
    order, or raise ``SubmissionRefused`` for a deterministic filter violation.
    """
    order_type = order_type.upper()
    sym.validate_params(order_type, symbol, price, qty)
    try:
        from providers.quantity import balance_cap_quantity, fee_cap_quantity
        available_qty, _balance_asset = balance_cap_quantity(
            api.get_free_balance, symbol, order_type, price)
        if available_qty is None:
            print(f"Balance unavailable for {order_type} {symbol}; order skipped.")
            return None
        if available_qty <= 0:
            print(f"No sufficient quantity available to place the {order_type} order.")
            return None

        # Keep the final check next to submission in case balance changed after planning.
        # ``available_qty`` is already expressed as base quantity.
        fee_cap = fee_cap_quantity(available_qty, PLACE_ORDER_FEE_PCT)
        if qty > fee_cap:
            print(f"Adjusting {order_type} qty from {qty:.8f} to "
                  f"{fee_cap:.8f} to cover balance and fees")
            qty = fee_cap

        current_price = api.get_current_price(symbol)
        from providers.binance_filters import (
            BinanceFilterError,
            BinanceOrderRules,
            binance_filter_refusal_reason,
        )
        rules = BinanceOrderRules.from_symbol_info(client.get_symbol_info(symbol))
        candidate_price = order_candidate_price(
            order_type, price, current_price, market=bool(force))
        try:
            normalized_qty, normalized_price = rules.normalize(
                quantity=qty,
                price=candidate_price,
                market=bool(force),
                reference_price=current_price,
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
              f"{'market price' if force else f'price {price}'}")
        if order_type in {"BUY", "SELL"}:
            return _submit_binance_order(
                order_type, symbol, qty, price=None if force else price,
                market=bool(force), client_order_id=client_order_id,
                cache_permit=cache_permit,
                permit_requested_price=permit_requested_price, kind=kind,
                cancel_opposite_requested_price=(
                    cancel_opposite_requested_price))
        print(f"Invalid order type: {order_type}")
        return None
    except BinanceAPIException as e:
        print(f"Error placing {order_type} order (mechanics): {e}")
        return None
