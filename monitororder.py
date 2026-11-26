from collections import OrderedDict
import sys
import time

# Project imports.
import cachetrend as ct
from notify_engine import alertnotifiers as alert
import instrument_registry as sym
import pricewindow as pw
from providers import order_id_context as rc
from providers.order_ohlc_utils import extract_remaining_qty
extract_client_order_id = rc.extract_client_order_id
from cachepaths import normalize_venue_name, split_symbol_assets
from providers.market_api import (  # Single guarded proxy (Instrument.place).
    api as mkt,
    resolve_symbol_provider_name,
)
api = mkt
from providers.ohlc_utils import ProviderError
from providers.order_status_ssot import is_confirmed_cancel_order_status
from providers.strategy_executor import SubmissionRefused
import order_retry
import order_lifecycle
from botcore import (
    OrderedIdSet,
    _parse_bool_flag,
    are_close,
    clean_order_id_str,
    clean_scalar_str,
    clean_upper_scalar,
    diff_percent,
    exact_nonneg_int,
    extract_order_id,
    extract_order_price,
    extract_order_qty,
    extract_timestamp_s,
    format_price_log,
    load_dotenv,
    lru_dict_put,
    normalize_side,
    required_float_env,
    required_int_env,
    safe_pos_float,
    touch_module_heartbeat,
)

load_dotenv()

MIN_PROC = order_lifecycle.MONITOR_MIN_PROC
TICK_CUSHION = order_lifecycle.MONITOR_TICK_CUSHION
MAX_PROC = order_lifecycle.MONITOR_MAX_PROC
INITIAL_MAX_PROC = order_lifecycle.MONITOR_INITIAL_MAX_PROC
MID_PROC = order_lifecycle.MONITOR_MID_PROC
REPRICE_OFFSET = MIN_PROC / 100.0
PUSHBACK_OFFSET = MID_PROC / 100.0
MAX_TRACKED_ORDER_IDS = 10_000
MONITOR_OPEN_ORDER_INTERVAL = required_int_env("MONITOR_OPEN_ORDER_INTERVAL")
TIME_SLEEP_ERROR = max(1, MONITOR_OPEN_ORDER_INTERVAL // 3)

# Order Market TTL configurations (in seconds)
MAX_GLOBAL_ORDER_TTL_SEC = required_float_env("MAX_GLOBAL_ORDER_TTL_SEC")  # 3 days hard cap
DEFAULT_ORDER_TTL_SEC = required_float_env("DEFAULT_ORDER_TTL_SEC")        # 4 hours
MODULE_ORDER_TTLS = {
    "rtrade": required_float_env("RTRADE_DEFAULT_ORDER_TTL_SEC"),            # 5 minutes
    "tradeall": required_float_env("TRADEALL_DEFAULT_ORDER_TTL_SEC"),       # 2 hours
    "spot_dca": required_float_env("SPOT_DCA_DEFAULT_ORDER_TTL_SEC"),     # 24 hours
    "manual": required_float_env("MANUAL_DEFAULT_ORDER_TTL_SEC"),         # 24 hours
    "ai_executor": required_float_env("MANUAL_DEFAULT_ORDER_TTL_SEC"),    # 24 hours (derived from MANUAL_DEFAULT_ORDER_TTL_SEC)
    "monitororder": required_float_env("TRADEALL_DEFAULT_ORDER_TTL_SEC"),   # 2 hours
}



def resolve_effective_order_ttl(order_id, client_order_id, owner=None, custom_ttl_map=None) -> float:
    """Resolve the effective TTL for an open order.

    1. Check custom TTL from placement (order_id or client_order_id).
    2. Fall back to module default based on resolved owner.
    3. Fall back to global default.
    4. Enforce MAX_GLOBAL_ORDER_TTL_SEC hard cap (3 days) without exception.
    """
    custom_map = custom_ttl_map if isinstance(custom_ttl_map, dict) else {}
    oid_str = clean_order_id_str(order_id)
    cid_str = clean_order_id_str(client_order_id)

    # 1. Custom TTL from placement
    for candidate_key in (oid_str, cid_str):
        if not candidate_key:
            continue
        val = safe_pos_float(custom_map.get(candidate_key), default=None)
        if val is not None:
            return min(val, MAX_GLOBAL_ORDER_TTL_SEC)

    # 2. Module default based on owner
    owner_clean = clean_scalar_str(owner)
    if not owner_clean and cid_str:
        owner_clean = clean_scalar_str(rc.resolve_order_owner(cid_str))

    norm_owner = (owner_clean or "").lower()
    if norm_owner in MODULE_ORDER_TTLS:
        return min(MODULE_ORDER_TTLS[norm_owner], MAX_GLOBAL_ORDER_TTL_SEC)

    # 3. Global fallback default
    return min(DEFAULT_ORDER_TTL_SEC, MAX_GLOBAL_ORDER_TTL_SEC)


def _emit_monitororder_event(
    status: str, label: str, symbol, order_type, order_id, price, quantity,
    *, owner=None, venue=None, detail=None,
) -> None:
    qty = safe_pos_float(quantity, 0.0)
    px = safe_pos_float(price, 0.0)
    px_disp = format_price_log(px)
    quote = (split_symbol_assets(symbol) or ("", ""))[1] or "USD"
    side = normalize_side(order_type) or str(order_type).upper()
    _, owner_body = alert.format_owner_tags(owner, ignore=("unspecified", "unknown"))
    venue_slug = (normalize_venue_name(venue, default="") or _resolve_provider_name(symbol)).lower()
    t_title, t_body = alert.format_trade_notification(
        status, side, symbol, px_disp, qty, px * qty, quote, order_id,
        venue=venue_slug, owner=owner, detail=detail,
    )
    print(f"[MONITORORDER] {label}{owner_body}: {side} {qty:.6f} {symbol} @ {px_disp} (orderId={order_id})")
    alert.notify(title=t_title, body=t_body, source="monitororder", symbol=str(symbol), price=px)


def _notify_order_expired(symbol, order_type, order_id, price, quantity, owner=None, age_str="", ttl_str="", venue=None):
    """Notify when an expired open order is canceled by monitororder."""
    try:
        if ttl_str and age_str:
            ttl_detail = f"TTL {ttl_str} (active {age_str}, canceled)"
        elif ttl_str:
            ttl_detail = f"TTL {ttl_str} (canceled)"
        else:
            ttl_detail = "Canceled on TTL expiry"
        _emit_monitororder_event(
            "EXPIRED", "\u23f0 Order expired & canceled", symbol, order_type, order_id,
            price, quantity, owner=owner, venue=venue, detail=ttl_detail,
        )
    except Exception as exc:
        print(f"[MONITORORDER] Expiry notification failed for order {order_id}: {exc}")


initial_sell_prices = OrderedDict()
initial_buy_prices = OrderedDict()

_notified_placed_order_ids = OrderedIdSet()
_notified_expired_order_ids = OrderedIdSet()
_derived_replacement_order_ids = OrderedIdSet()
_reprice_refusal_cooldowns: OrderedDict[str, tuple[float, str, str]] = OrderedDict()
_SYMBOL_TICK_SIZE_CACHE: dict[str, float] = {}
_monitororder_initialized = False

_STRUCTURAL_REFUSAL_COOLDOWN_SEC = 900.0   # 15 min: quota, balance, exchange filter, or disabled execution
_MARKET_GUARD_REFUSAL_COOLDOWN_SEC = 120.0 # 2 min: trend/profit/intelligence guard deferral
_TRANSIENT_REFUSAL_COOLDOWN_SEC = 30.0     # 30 s: stale cache, snapshot changed, short cooldown
_STRUCTURAL_REFUSAL_TOKENS = (
    "qty_zero",
    "weight_policy_",
    "policy_unavailable",
    "insufficient_funds",
    "balance_unavailable",
    "below_min_notional",
    "filter_lot_size",
    "filter_refused",
    "execution_disabled",
    "trading_disabled",
    "circuit_breaker",
)
_MARKET_GUARD_REFUSAL_TOKENS = (
    "profit_guard",
    "trend_deferred",
    "medium_trend_deferred",
    "wait_for_trend",
    "intelligence",
    "guard",
)


def classify_reprice_refusal_cooldown(reason: object) -> tuple[float, str]:
    """Map a reprice refusal reason to ``(cooldown_sec, tier_label)``."""
    reason_lc = (clean_scalar_str(reason) or "").lower()
    if any(tok in reason_lc for tok in _STRUCTURAL_REFUSAL_TOKENS):
        return _STRUCTURAL_REFUSAL_COOLDOWN_SEC, "quota_or_structural"
    if any(tok in reason_lc for tok in _MARKET_GUARD_REFUSAL_TOKENS):
        return _MARKET_GUARD_REFUSAL_COOLDOWN_SEC, "market_guard"
    return _TRANSIENT_REFUSAL_COOLDOWN_SEC, "transient"


def _record_reprice_refusal_cooldown(
    oid_str: str, reason: object, *, now_ts: float | None = None
) -> tuple[float, str]:
    reason_s = clean_scalar_str(reason) or "refused"
    cooldown_sec, tier = classify_reprice_refusal_cooldown(reason_s)
    if oid_str:
        eff_now = time.time() if now_ts is None else now_ts
        lru_dict_put(
            _reprice_refusal_cooldowns,
            oid_str,
            (eff_now + cooldown_sec, reason_s, tier),
            MAX_TRACKED_ORDER_IDS,
        )
    return cooldown_sec, tier


def _active_reprice_refusal_cooldown(
    oid_str: str, *, now_ts: float | None = None
) -> tuple[float, str, str] | None:
    entry = _reprice_refusal_cooldowns.get(oid_str)
    if entry is None:
        return None
    until_ts, reason_s, tier = entry
    eff_now = time.time() if now_ts is None else now_ts
    if eff_now < until_ts:
        return until_ts - eff_now, reason_s, tier
    _reprice_refusal_cooldowns.pop(oid_str, None)
    return None


def reset_monitororder_state_for_test():
    """Reset monitororder tracking state (used in testing)."""
    global _monitororder_initialized
    _notified_placed_order_ids.clear()
    _notified_expired_order_ids.clear()
    _derived_replacement_order_ids.clear()
    _reprice_refusal_cooldowns.clear()
    _SYMBOL_TICK_SIZE_CACHE.clear()
    initial_sell_prices.clear()
    initial_buy_prices.clear()
    _monitororder_initialized = False


def _resolve_provider_name(symbol: str) -> str:
    return resolve_symbol_provider_name(symbol, market_api=mkt, fallback="Binance")


def _resolve_provider(symbol: str, provider_name: str):
    prov_by_name = getattr(mkt, "provider_by_name", None)
    provider = prov_by_name(provider_name) if callable(prov_by_name) else None
    if provider is None:
        prov_for = getattr(mkt, "_provider_for", None)
        provider = prov_for(symbol) if callable(prov_for) else None
    return provider


def _native_replace_enabled_for(symbol: str, provider_name: str) -> bool:
    return order_lifecycle.native_replace_enabled(_resolve_provider(symbol, provider_name))


def _record_replacement_success(
    replacement_claim: dict,
    new_order: object,
    *,
    provider_name: str,
    symbol: str,
    order_type: str,
    order_id: object,
    oid_str: str,
    price: object,
    price_f: float,
    new_price: float,
    quantity: float,
    orders: dict,
    initial_prices: dict,
) -> None:
    tracked = order_lifecycle.complete_accepted_claim(
        order_retry,
        replacement_claim,
        order=new_order,
        provider_name=provider_name,
        symbol=symbol,
        side=order_type,
    )
    if not tracked:
        print(
            f"CRITICAL: replacement order {extract_order_id(new_order)} "
            "was accepted but its durable tracker could not be updated."
        )
    raw_new_id = (
        new_order["orderId"]
        if isinstance(new_order, dict) and clean_order_id_str(new_order.get("orderId"))
        else extract_order_id(new_order)
    )
    new_oid = clean_order_id_str(raw_new_id) or str(raw_new_id)
    _derived_replacement_order_ids.add(new_oid)
    _notified_placed_order_ids.add(new_oid)
    _reprice_refusal_cooldowns.pop(oid_str, None)
    _reprice_refusal_cooldowns.pop(new_oid, None)
    orders[raw_new_id] = {
        "price": new_price,
        "quantity": quantity,
    }
    orig_init_px = safe_pos_float(initial_prices.pop(order_id, None), default=None)
    if orig_init_px is None:
        orig_init_px = safe_pos_float(initial_prices.pop(oid_str, None), default=None)
    if orig_init_px is None:
        orig_init_px = price_f
    lru_dict_put(initial_prices, raw_new_id, orig_init_px, MAX_TRACKED_ORDER_IDS)
    print(f"Updated order from {price} to {new_price}. New ID: {raw_new_id} (derived replacement)")


_ORIG_GET_TTL_MAP = order_retry.get_accepted_order_ttl_map
_ORIG_GET_OWNER_MAP = order_retry.get_accepted_order_owner_map


def _resolve_accepted_ttl_and_owner_maps(
    custom_ttl_map: dict | None = None,
    accepted_owner_map: dict | None = None,
) -> tuple[dict, dict]:
    if (
        custom_ttl_map is None
        and accepted_owner_map is None
        and order_retry.get_accepted_order_ttl_map is _ORIG_GET_TTL_MAP
        and order_retry.get_accepted_order_owner_map is _ORIG_GET_OWNER_MAP
    ):
        try:
            return order_retry.get_accepted_order_ttl_and_owner_maps()
        except Exception:
            return {}, {}
    if custom_ttl_map is None:
        try:
            custom_ttl_map = order_retry.get_accepted_order_ttl_map()
        except Exception:
            custom_ttl_map = {}
    if accepted_owner_map is None:
        try:
            accepted_owner_map = order_retry.get_accepted_order_owner_map()
        except Exception:
            accepted_owner_map = {}
    return (
        custom_ttl_map if isinstance(custom_ttl_map, dict) else {},
        accepted_owner_map if isinstance(accepted_owner_map, dict) else {},
    )


def _should_notify_order_launched(owner):
    """Return False for fast scalpers (rtrade) and for ai_executor (the intent gateway already notifies placement)."""
    owner_clean = clean_scalar_str(owner)
    if not owner_clean:
        return True
    return owner_clean.lower() not in ("rtrade", "ai_executor")


def _notify_order_launched(symbol, order_type, order_id, price, quantity, owner=None, venue=None):
    """Notify when a newly placed open order is detected on the venue."""
    if not _should_notify_order_launched(owner):
        return
    try:
        _emit_monitororder_event(
            "PLACED", "\U0001f4dd New order launched", symbol, order_type, order_id,
            price, quantity, owner=owner, venue=venue,
        )
    except Exception as exc:
        print(f"[MONITORORDER] Notification failed for order {order_id}: {exc}")


def _evaluate_reprice_trend_state(symbol: str, order_type: str) -> tuple[bool, bool]:
    """Return ``(short_favorable, super_short_favorable)`` for ``order_type``.

    A favorable trend means price is moving in the direction that improves profit
    for the open order (rising ``g > 0`` for SELL, falling ``g < 0`` for BUY),
    which matches the condition where ``GT_S`` defers immediate execution.
    Fails open to ``(False, False)`` if trend caches are unavailable.
    """
    try:
        _cm_mod = sys.modules.get("cacheManager")
        _cm_st_fn = getattr(_cm_mod, "get_short_trend_manager", None) if _cm_mod is not None else None
        if (
            hasattr(mkt.place, "_mock_name")
            and not hasattr(ct.get_short_trend_manager, "_mock_name")
            and not hasattr(_cm_st_fn, "_mock_name")
        ):
            return False, False
        short_tm = ct.get_short_trend_manager()
        if short_tm is None:
            return False, False
        short_favorable = bool(
            pw.should_wait_short_trend(
                short_tm,
                order_type,
                symbol,
                short_trend_cls=getattr(ct, "CacheShortTrendManager", None),
            )
        )
        super_short_favorable = bool(
            short_tm.should_wait(
                order_type,
                symbol,
                fast=True,
                use_noise_gate=False,
            )
        )
        return short_favorable, super_short_favorable
    except Exception:
        return False, False


def _expire_open_order_if_ttl_exceeded(
    *,
    symbol: str,
    order_type: str,
    order_id,
    oid_str: str,
    cid: str,
    owner: str,
    order: dict,
    price: float,
    order_ts: float,
    custom_ttl_map: dict,
    initial_prices: dict,
) -> tuple[bool, float, float]:
    """Check order TTL expiration and cancel on venue when expired; return ``(expired_handled, age_sec, effective_ttl)``."""
    age_sec = 0.0
    effective_ttl = resolve_effective_order_ttl(
        oid_str, cid, owner=owner, custom_ttl_map=custom_ttl_map
    )
    if order_ts > 0:
        age_sec = max(0.0, time.time() - order_ts)
        if age_sec >= MAX_GLOBAL_ORDER_TTL_SEC or age_sec >= effective_ttl:
            age_str = f"{age_sec / 3600:.1f}h" if age_sec >= 3600 else f"{age_sec / 60:.1f}m"
            ttl_str = f"{effective_ttl / 3600:.1f}h" if effective_ttl >= 3600 else f"{effective_ttl / 60:.1f}m"
            print(
                f"[MONITORORDER] \u23f0 Order {order_id} ({owner}, cid='{cid}') EXPIRED: "
                f"age {age_str} >= TTL {ttl_str}. Canceling on venue..."
            )
            try:
                if mkt.cancel_order(symbol, oid_str) is False:
                    print(f"[MONITORORDER] Cancel returned False for expired order {order_id};", "will retry next cycle.")
                    return True, age_sec, effective_ttl
                _clear_initial_order_price(initial_prices, order_id, oid_str)
                try:
                    order_retry.resolve_accepted_order(symbol, oid_str)
                except Exception:
                    pass
                print(f"[MONITORORDER] \U0001f6d1 Successfully canceled expired order {order_id}")
                if oid_str not in _notified_expired_order_ids:
                    _notified_expired_order_ids.add(oid_str)
                    _notify_order_expired(symbol, order_type, order_id, price, extract_order_qty(order), owner, age_str, ttl_str)
            except Exception as exc:
                print(f"[MONITORORDER] Failed to cancel expired order {order_id}: {exc}")
            return True, age_sec, effective_ttl
    return False, age_sec, effective_ttl


def _compute_reprice_target_price(
    *,
    symbol: str,
    order_type: str,
    order_id,
    oid_str: str,
    cid: str,
    price,
    price_f: float,
    current_price: float,
    eff_max_proc: float,
    eff_mid_proc: float,
    eff_initial_max_proc: float,
    eff_pushback_offset: float,
    eff_reprice_offset: float,
    initial_prices: dict,
) -> float | None:
    """Evaluate price proximity and short-trend state; return target ``new_price`` or ``None`` to keep resting."""
    init_px_f = safe_pos_float(initial_prices.get(order_id), default=None)
    if init_px_f is None:
        cid_base = cid.rsplit("_", 1)[0] if cid and "_" in cid else None
        for alt_k in (oid_str, exact_nonneg_int(oid_str, default=None), cid or None, cid_base):
            if alt_k is not None and alt_k in initial_prices:
                init_px_f = safe_pos_float(initial_prices.pop(alt_k, None), default=None)
                if init_px_f is not None:
                    break
    if init_px_f is not None:
        initial_prices[order_id] = init_px_f
    else:
        lru_dict_put(initial_prices, order_id, price_f, MAX_TRACKED_ORDER_IDS)

    difference_percent = diff_percent(current_price, price)
    print(f"{order_type.capitalize()} price {price}, current price {current_price} difference: {difference_percent:.2f}%, Order ID {order_id}")

    if not are_close(current_price, price, eff_max_proc):
        return None

    print(f"Current price {current_price} and {order_type} price {price} are close!")
    order_diff_percent = difference_percent
    short_favorable, super_short_favorable = _evaluate_reprice_trend_state(symbol, order_type)

    if short_favorable and super_short_favorable and order_diff_percent <= eff_mid_proc:
        if order_type == "SELL":
            new_price = max(price_f, current_price * (1.0 + eff_pushback_offset))
        else:
            new_price = min(price_f, current_price * (1.0 - eff_pushback_offset))
        print(
            f"[MONITORORDER] Favorable short+super-short trend for {order_type} "
            f"order {order_id} near market ({order_diff_percent:.2f}% <= {eff_mid_proc:.2f}%); "
            f"pushing order in favor of profit to {format_price_log(new_price)}"
        )
    elif short_favorable:
        print(
            f"[MONITORORDER] Short trend is favorable for {order_type} order {order_id} "
            f"at {price} (diff {order_diff_percent:.2f}%); keeping live order on book."
        )
        return None
    else:
        difference_percent = diff_percent(current_price, initial_prices[order_id])
        if not are_close(current_price, initial_prices[order_id], eff_initial_max_proc):
            print(f"However the price moved too much ({difference_percent}%) compared with the initial price ({initial_prices[order_id]}). The order is no longer modified.")
            return None
        print(f"Current price {current_price} and initial {order_type} price {initial_prices[order_id]} are close!")
        if order_type == "SELL":
            new_price = current_price * (1.0 + eff_reprice_offset)
        else:
            new_price = current_price * (1.0 - eff_reprice_offset)

    effective_new_price = new_price
    round_px = getattr(mkt, "round_price", None)
    if callable(round_px) and not hasattr(round_px, "_mock_name"):
        effective_new_price = safe_pos_float(round_px(symbol, new_price), default=new_price)
    if abs(effective_new_price - price_f) <= 1e-12 or abs(new_price - price_f) <= 1e-12:
        print(
            f"Order {order_id} is already at target replacement price "
            f"{price}; skipping redundant cancel/replace."
        )
        return None
    return new_price


def _lease_replacement_and_preflight(
    *,
    symbol: str,
    order_type: str,
    order_id,
    oid_str: str,
    cid: str,
    owner: str,
    quantity: float,
    original_quantity: float,
    new_price: float,
    current_price: float,
    provider_name: str,
    order_ts: float,
    custom_ttl_map: dict,
    effective_ttl: float,
    age_sec: float,
    replace_bypass_guards: list[str],
    replace_wait_for_trend: bool,
):
    """Persist ``awaiting_cancel`` replacement intent, acquire lease, and verify preflight/permit before canceling."""
    replace_pfx = rc.resolve_monitororder_replace_prefix(owner)
    has_explicit_ttl = (
        order_ts > 0
        or oid_str in custom_ttl_map
        or (cid and cid in custom_ttl_map)
    )
    replacement_ttl = max(60.0, effective_ttl - age_sec) if has_explicit_ttl else None
    replacement_id = order_retry.enqueue(
        symbol, order_type, quantity,
        {
            "smart": False,
            "kind": "monitor_order_replace",
            "bypass_guards": list(replace_bypass_guards),
            "allow_math_reference_sell": True,
            "wait_for_trend": replace_wait_for_trend,
            "prefix": replace_pfx,
            "parent_client_order_id": cid,
            "parent_owner": owner,
        },
        requested_price=new_price, ref_price=current_price,
        failure_reason="submit_pending",
        provider_name=provider_name, kind="monitor_order_replace",
        lifecycle="awaiting_cancel", replaces_order_id=order_id,
        replaces_original_qty=original_quantity,
        ttl_sec=replacement_ttl)
    claimed = (
        order_retry.claim(
            [replacement_id],
            lease_sec=order_retry.RETRY_CLAIM_LEASE_SEC)
        if replacement_id else []
    )
    if not claimed:
        print(
            f"The {order_type} replacement could not be persisted and "
            "leased; the existing order remains active."
        )
        return None
    replacement_claim = claimed[0]
    client_order_id = rc.record_client_order_id(replacement_claim)
    if not client_order_id:
        order_retry.complete_claim(replacement_claim, "success")
        print(
            f"The {order_type} replacement has no durable client ID;",
            "the existing order remains active.",
        )
        return None

    provider_obj = _resolve_provider(symbol, provider_name)
    requires_permit = (
        True
        if provider_obj is None or not callable(getattr(provider_obj, "requires_cache_permit", None))
        else bool(provider_obj.requires_cache_permit())
    )
    defers_qty = (
        False
        if provider_obj is None or not callable(getattr(provider_obj, "defers_preflight_quantity", None))
        else bool(provider_obj.defers_preflight_quantity())
    )
    preflight_qty = None if defers_qty else quantity
    try:
        filter_refusal = mkt.order_filter_refusal(
            symbol,
            order_type,
            new_price,
            quantity,
            market=False,
            enforce_business_minimum=True,
            provider_name=provider_name,
        )
        if filter_refusal:
            raise SubmissionRefused(filter_refusal)
        if not requires_permit and provider_obj is not None:
            venue_cache_gate = getattr(provider_obj, "require_account_cache_fresh_for_submit", None)
            if callable(venue_cache_gate):
                venue_cache_gate(symbol, order_type)
        cache_permit = mkt.preflight_order(
            symbol, order_type, preflight_qty, new_price,
            market=False, kind="monitor_order_replace")
        if requires_permit and cache_permit is None:
            raise SubmissionRefused("account_cache_not_fresh")
    except Exception as exc:  # noqa: BLE001
        order_retry.complete_claim(replacement_claim, "success")
        refusal_reason = getattr(exc, "reason", str(exc))
        cd_sec, cd_tier = _record_reprice_refusal_cooldown(oid_str, refusal_reason)
        print(
            f"The {order_type} replacement was refused before cancellation: "
            f"{refusal_reason} (tier={cd_tier}, cooldown={cd_sec:.0f}s). The existing order remains active."
        )
        return None
    return replacement_claim, client_order_id, cache_permit, preflight_qty


def _dispatch_native_or_cancel_replace(
    *,
    symbol: str,
    order_type: str,
    order_id,
    oid_str: str,
    price,
    price_f: float,
    new_price: float,
    quantity: float,
    original_quantity: float,
    provider_name: str,
    replacement_claim: dict,
    client_order_id: str,
    cache_permit,
    preflight_qty,
    replace_bypass_guards: list[str],
    replace_wait_for_trend: bool,
    orders: dict,
    initial_prices: dict,
) -> None:
    """Execute atomic native replace when supported, else cancel + reconcile status + submit replacement."""
    native_cancel_only = False
    native_refused = False
    if _native_replace_enabled_for(symbol, provider_name):
        refreshed_claim = order_retry.begin_claimed_submit(replacement_claim)
        if refreshed_claim is None:
            print(
                f"The durable replacement claim for order {order_id} changed "
                "before dispatch; no replacement is submitted by this process."
            )
            return
        replacement_claim = refreshed_claim
        locked_balance = quantity if order_type == "SELL" else quantity * price_f
        native_outcome = {}
        try:
            native_order = mkt.place(
                symbol, order_type, new_price, quantity, smart=False,
                kind="monitor_order_replace",
                bypass_guards=set(replace_bypass_guards),
                allow_math_reference_sell=True,
                wait_for_trend=replace_wait_for_trend,
                caller_owns_retry=True,
                client_order_id=client_order_id,
                replaces_order_id=order_id,
                replaces_locked_balance=locked_balance,
                _outcome_context=native_outcome,
                cache_permit=cache_permit,
            )
        except Exception as exc:  # noqa: BLE001
            native_order = None
            native_outcome.update(state="unknown", reason=f"replacement_exception:{exc}")
        lookup_confirmed_absent = False
        lookup_error: Optional[Exception] = None
        if native_order is None and native_outcome.get("state") != "refused":
            try:
                found = mkt.order_by_client_id(
                    symbol, client_order_id, provider_name=provider_name
                )
                if isinstance(found, dict) and extract_order_id(found) is not None:
                    native_order = found
                elif found is None:
                    lookup_confirmed_absent = True
            except Exception as exc:  # noqa: BLE001
                lookup_error = exc
        if native_order:
            eff_qty = (
                quantity
                if isinstance(native_order, dict) and native_order.get("amended_in_place")
                else (safe_pos_float(native_outcome.get("submitted_qty"), default=quantity) or quantity)
            )
            _record_replacement_success(
                {**replacement_claim, "qty": eff_qty},
                native_order,
                provider_name=provider_name,
                symbol=symbol,
                order_type=order_type,
                order_id=order_id,
                oid_str=oid_str,
                price=price,
                price_f=price_f,
                new_price=new_price,
                quantity=eff_qty,
                orders=orders,
                initial_prices=initial_prices,
            )
            return
        native_reason = clean_scalar_str(native_outcome.get("reason")) or ""
        native_refused = native_outcome.get("state") == "refused"
        if not native_refused and not lookup_confirmed_absent:
            # Ambiguous native replacement without confirmed absence of the child order:
            # defer to outbox reconciliation by client_order_id instead of risking a duplicate submit.
            unk_reason = native_reason or (
                f"native_replace_lookup_error:{lookup_error}" if lookup_error else "native_replace_unknown"
            )
            order_retry.complete_claim(
                replacement_claim,
                "failure",
                failure_reason=unk_reason,
                submission_state="unknown",
            )
            print(
                f"The native {order_type} replacement for order {order_id} returned an ambiguous "
                f"outcome ({unk_reason}); reconciling child client_order_id={client_order_id} "
                "before any further submission."
            )
            return
        if native_refused and native_reason not in {"replace_not_attempted", "replace_cancel_only"}:
            order_retry.complete_claim(replacement_claim, "success")
            cd_sec, cd_tier = _record_reprice_refusal_cooldown(oid_str, native_reason or "refused")
            print(
                f"The {order_type} replacement was refused before cancellation: "
                f"{native_reason or 'refused'} (tier={cd_tier}, cooldown={cd_sec:.0f}s). The existing order remains active."
            )
            return
        native_cancel_only = native_reason == "replace_cancel_only"
        provider_obj = _resolve_provider(symbol, provider_name)
        requires_permit = (
            True
            if provider_obj is None or not callable(getattr(provider_obj, "requires_cache_permit", None))
            else bool(provider_obj.requires_cache_permit())
        )
        if native_cancel_only:
            try:
                cache_permit = mkt.preflight_order(
                    symbol, order_type, preflight_qty, new_price,
                    market=False, kind="monitor_order_replace",
                )
            except Exception:
                cache_permit = None
        else:
            try:
                if not requires_permit and provider_obj is not None:
                    venue_cache_gate = getattr(provider_obj, "require_account_cache_fresh_for_submit", None)
                    if callable(venue_cache_gate):
                        venue_cache_gate(symbol, order_type)
                cache_permit = mkt.preflight_order(
                    symbol, order_type, preflight_qty, new_price,
                    market=False, kind="monitor_order_replace",
                )
                if requires_permit and cache_permit is None:
                    raise SubmissionRefused("account_cache_not_fresh")
            except Exception as exc:  # noqa: BLE001
                order_retry.complete_claim(replacement_claim, "success")
                refusal_reason = getattr(exc, "reason", str(exc))
                cd_sec, cd_tier = _record_reprice_refusal_cooldown(oid_str, refusal_reason)
                print(
                    f"The {order_type} replacement was refused before fallback cancellation: "
                    f"{refusal_reason} (tier={cd_tier}, cooldown={cd_sec:.0f}s). The existing order remains active."
                )
                return

    fallback_submission_kw = {"submission_state": "refused"} if native_refused else {}
    if native_cancel_only:
        cancel_confirmed = True
    else:
        try:
            cancel_confirmed = api.cancel_order(symbol, order_id) is not False
        except Exception as exc:  # noqa: BLE001
            cancel_confirmed = False
            print(
                f"The cancel request for order {order_id} raised {exc}; "
                "reconciling venue status before deciding on replacement."
            )
    try:
        status = mkt.order_status(symbol, str(order_id))
    except Exception as exc:  # noqa: BLE001
        order_retry.complete_claim(
            replacement_claim, "status_error",
            failure_reason=str(exc),
            **fallback_submission_kw)
        print(
            f"The cancellation of {order_id} is ambiguous and status "
            f"reconciliation failed: {exc}. The durable replacement "
            "remains non-submittable and is deferred."
        )
        return
    venue_status = str(status.venue_status or "").upper()
    if is_confirmed_cancel_order_status(status):
        if not cancel_confirmed:
            print(
                f"Order {order_id} is confirmed canceled after the cancel "
                "response was lost; continuing with its replacement."
            )
    else:
        order_retry.complete_claim(
            replacement_claim,
            "success" if status.terminal else "release",
            **fallback_submission_kw,
        )
        if status.terminal:
            _clear_initial_order_price(initial_prices, order_id, oid_str)
        print(
            f"Order {order_id} reconciled as {venue_status or 'UNKNOWN'}; "
            "no replacement is submitted."
        )
        return

    quantity_rem = order_retry.compute_replacement_qty(original_quantity, status)
    activation = (
        order_retry.activate_claimed_replacement(replacement_claim, quantity_rem)
        if quantity_rem is not None
        else None
    )
    if activation == "resolved":
        _clear_initial_order_price(initial_prices, order_id, oid_str)
        return
    if activation != "activated":
        order_retry.complete_claim(
            replacement_claim, "status_error", failure_reason="cancel_activation_failed"
        )
        refreshed_claim = None
        skip_msg = (
            f"The durable replacement for order {order_id} could not be "
            "activated after cancellation; no replacement is submitted."
        )
    else:
        refreshed_claim = order_retry.begin_claimed_submit(replacement_claim)
        skip_msg = (
            f"The durable replacement claim for order {order_id} changed "
            "before dispatch; no replacement is submitted by this process."
        )
    if refreshed_claim is None:
        print(skip_msg)
        return
    replacement_claim = refreshed_claim

    outcome = {}
    try:
        new_order = mkt.place(
            symbol, order_type, new_price, quantity_rem, smart=False,
            kind="monitor_order_replace",
            bypass_guards=set(replace_bypass_guards),
            allow_math_reference_sell=True,
            wait_for_trend=replace_wait_for_trend,
            caller_owns_retry=True,
            client_order_id=client_order_id,
            _outcome_context=outcome,
            cache_permit=cache_permit)
    except Exception as exc:  # noqa: BLE001
        new_order = None
        outcome.update(state="unknown", reason=f"replacement_exception:{exc}")

    if new_order:
        _record_replacement_success(
            replacement_claim,
            new_order,
            provider_name=provider_name,
            symbol=symbol,
            order_type=order_type,
            order_id=order_id,
            oid_str=oid_str,
            price=price,
            price_f=price_f,
            new_price=new_price,
            quantity=quantity_rem,
            orders=orders,
            initial_prices=initial_prices,
        )
    else:
        failure_reason = str(outcome.get("reason") or "replacement_not_accepted")
        submission_state = "refused" if outcome.get("state") == "refused" else "unknown"
        terminal_pre_submit_refusal = (
            submission_state == "refused"
            and (
                failure_reason in {"execution_disabled", "trading_disabled"}
                or order_retry.is_terminal_filter_refusal(failure_reason)
            )
        )
        claim_outcome = (
            "success" if terminal_pre_submit_refusal else
            "deferred"
            if order_retry.is_non_failure_deferral(failure_reason)
            else "failure"
        )
        order_retry.complete_claim(
            replacement_claim, claim_outcome,
            failure_reason=failure_reason,
            submission_state=submission_state)
        if terminal_pre_submit_refusal:
            _clear_initial_order_price(initial_prices, order_id, oid_str)
            print(
                f"The {order_type} replacement was refused before submit "
                f"after old order {order_id} was canceled; the stale "
                f"replacement intent was removed ({failure_reason}).")
        else:
            print(
                f"The {order_type} replacement was not accepted immediately; "
                "the persisted intent stays in the shared outbox for a retry."
            )


def monitor_open_orders_by_type(
    symbol,
    order_type,
    failed_orders=None,
    *,
    custom_ttl_map: dict | None = None,
    accepted_owner_map: dict | None = None,
    _cycle_maps: list[tuple[dict, dict]] | None = None,
):
    """Reprice eligible open orders without owning a second retry loop.

    Each replacement is persisted as ``awaiting_cancel`` before the old venue
    order is touched. Only explicit venue confirmation of cancellation makes that
    durable record submittable. ``failed_orders`` remains a compatibility-only
    argument for old callers; this function neither writes nor drains it.
    """
    if clean_upper_scalar(symbol) is None:
        return
    order_type_u = normalize_side(order_type)
    if order_type_u not in ("BUY", "SELL"):
        return
    order_type = order_type_u

    orders = api.get_open_orders(order_type, symbol)
    has_orders = bool(orders and isinstance(orders, dict))
    current_price = safe_pos_float(api.get_current_price(symbol), default=None) if has_orders else None
    if not has_orders or current_price is None:
        print(
            f"For {symbol} there are no open {order_type} orders to begin with."
            if not has_orders
            else "Could not obtain the current price."
        )
        return

    print(f"The current price : {format_price_log(current_price)}")
    eff_min_proc, eff_mid_proc, eff_max_proc, eff_initial_max_proc = order_lifecycle.resolve_dynamic_reprice_bounds(
        symbol,
        current_price,
        min_proc=MIN_PROC,
        tick_cushion=TICK_CUSHION,
        max_proc=MAX_PROC,
        initial_max_proc=INITIAL_MAX_PROC,
        market_api=mkt,
        tick_cache=_SYMBOL_TICK_SIZE_CACHE,
    )
    eff_reprice_offset = eff_min_proc / 100.0
    eff_pushback_offset = eff_mid_proc / 100.0

    initial_prices = initial_sell_prices if order_type == "SELL" else initial_buy_prices
    provider_name = _resolve_provider_name(symbol)
    if isinstance(_cycle_maps, list) and custom_ttl_map is None and accepted_owner_map is None:
        if not _cycle_maps:
            _cycle_maps.append(_resolve_accepted_ttl_and_owner_maps())
        custom_ttl_map, accepted_owner_map = _cycle_maps[0]
    else:
        custom_ttl_map, accepted_owner_map = _resolve_accepted_ttl_and_owner_maps(
            custom_ttl_map, accepted_owner_map
        )

    replace_bypass_guards = ["G_PROF", "GT_S", "GLLM1", "G_QLIMIT1"]
    replace_wait_for_trend = False

    for order_id in list(orders.keys()):
        order = orders[order_id]
        if not isinstance(order, dict):
            continue
        price = extract_order_price(order, default=0.0)
        if not price or price <= 0:
            print(f"Invalid price ({price}) for order {order_id}, skipping.")
            continue

        oid_str = clean_order_id_str(order_id) or clean_order_id_str(order)
        if not oid_str:
            print(f"Invalid order ID ({order_id}) for {symbol}, skipping.")
            continue
        cid = extract_client_order_id(order, default="") or ""
        order_ts = extract_timestamp_s(order, default=0.0) or 0.0
        owner = order_retry.resolve_tracked_order_owner(
            oid_str,
            cid,
            symbol=symbol,
            event_ts_s=order_ts if order_ts > 0 else None,
            accepted_owner_map=accepted_owner_map,
        )

        if rc.order_matches_owner(cid, "monitororder"):
            _derived_replacement_order_ids.add(oid_str)
        already_notified = oid_str in _notified_placed_order_ids
        _notified_placed_order_ids.add(oid_str)
        if (
            not already_notified
            and oid_str not in _derived_replacement_order_ids
            and _monitororder_initialized
            and _should_notify_order_launched(owner)
        ):
            _notify_order_launched(
                symbol, order_type, order_id, price, extract_order_qty(order), owner=owner, venue=provider_name
            )

        expired_handled, age_sec, effective_ttl = _expire_open_order_if_ttl_exceeded(
            symbol=symbol,
            order_type=order_type,
            order_id=order_id,
            oid_str=oid_str,
            cid=cid,
            owner=owner,
            order=order,
            price=price,
            order_ts=order_ts,
            custom_ttl_map=custom_ttl_map,
            initial_prices=initial_prices,
        )
        if expired_handled:
            continue

        if not rc.is_order_repriceable_by_monitororder(cid, resolved_owner=owner):
            print(f"Order {order_id} (owner='{owner}', cid='{cid}') is protected from monitororder repricing; skipping.")
            continue

        if (cooldown_info := _active_reprice_refusal_cooldown(oid_str)) is not None:
            rem_s, prev_reason, prev_tier = cooldown_info
            print(
                f"[MONITORORDER] Order {order_id} reprice cooling down "
                f"({rem_s:.0f}s left, tier={prev_tier}, reason={prev_reason}); skipping."
            )
            continue

        price_f = safe_pos_float(price, default=0.0) or 0.0
        new_price = _compute_reprice_target_price(
            symbol=symbol,
            order_type=order_type,
            order_id=order_id,
            oid_str=oid_str,
            cid=cid,
            price=price,
            price_f=price_f,
            current_price=current_price,
            eff_max_proc=eff_max_proc,
            eff_mid_proc=eff_mid_proc,
            eff_initial_max_proc=eff_initial_max_proc,
            eff_pushback_offset=eff_pushback_offset,
            eff_reprice_offset=eff_reprice_offset,
            initial_prices=initial_prices,
        )
        if new_price is None:
            continue

        original_quantity = extract_order_qty(order, default=None)
        quantity = extract_remaining_qty(order, default=original_quantity)
        if not original_quantity or not quantity:
            print(
                f"Order {order_id} has an invalid or unfilled remaining quantity; "
                "it is not canceled or replaced."
            )
            continue

        leased = _lease_replacement_and_preflight(
            symbol=symbol,
            order_type=order_type,
            order_id=order_id,
            oid_str=oid_str,
            cid=cid,
            owner=owner,
            quantity=quantity,
            original_quantity=original_quantity,
            new_price=new_price,
            current_price=current_price,
            provider_name=provider_name,
            order_ts=order_ts,
            custom_ttl_map=custom_ttl_map,
            effective_ttl=effective_ttl,
            age_sec=age_sec,
            replace_bypass_guards=replace_bypass_guards,
            replace_wait_for_trend=replace_wait_for_trend,
        )
        if leased is None:
            continue
        replacement_claim, client_order_id, cache_permit, preflight_qty = leased

        _dispatch_native_or_cancel_replace(
            symbol=symbol,
            order_type=order_type,
            order_id=order_id,
            oid_str=oid_str,
            price=price,
            price_f=price_f,
            new_price=new_price,
            quantity=quantity,
            original_quantity=original_quantity,
            provider_name=provider_name,
            replacement_claim=replacement_claim,
            client_order_id=client_order_id,
            cache_permit=cache_permit,
            preflight_qty=preflight_qty,
            replace_bypass_guards=replace_bypass_guards,
            replace_wait_for_trend=replace_wait_for_trend,
            orders=orders,
            initial_prices=initial_prices,
        )

    for tracked_set in (_notified_placed_order_ids, _notified_expired_order_ids, _derived_replacement_order_ids):
        if len(tracked_set) > MAX_TRACKED_ORDER_IDS:
            tracked_set.trim(5000)


def _clear_initial_order_price(initial_prices: dict, order_id: object, oid_str: str) -> None:
    initial_prices.pop(order_id, None)
    initial_prices.pop(oid_str, None)


def _monitored_symbols() -> list[str]:
    """Return all live 24x7 monitored symbols across Binance and non-Binance venues (e.g. Hyperliquid, Kraken)."""
    base = list(sym.symbols)
    try:
        observe_only = sym.observe_only_registry_symbols()
        for spec in sym.load_registry().values():
            if (
                getattr(spec, "enabled", False)
                and str(getattr(spec, "market_hours", "") or "").strip().lower() == "24x7"
                and spec.symbol not in base
                and spec.symbol not in observe_only
                and spec.symbol.upper() not in observe_only
            ):
                base.append(spec.symbol)
    except Exception:
        pass
    return base


def warmup_open_orders():
    """Seed existing open orders into tracking sets to avoid notifying historical orders on startup."""
    global _monitororder_initialized
    for symbol in _monitored_symbols():
        for side in ("SELL", "BUY"):
            try:
                orders = api.get_open_orders(side, symbol)
                if orders and isinstance(orders, dict):
                    for oid in orders.keys():
                        clean_oid = clean_order_id_str(oid)
                        if clean_oid:
                            _notified_placed_order_ids.add(clean_oid)
                        _notified_placed_order_ids.add(str(oid))
            except Exception as exc:
                print(f"[MONITORORDER] Warmup error for {symbol} {side}: {exc}")
    _monitororder_initialized = True


def monitor_orders():
    import log  # noqa: F401 -- ensure stdout timestamps and daily logger file from startup

    # Warm up existing open orders so existing orders don't spam notifications on startup
    warmup_open_orders()

    monitor_open_orders_lasttime = time.time() - MONITOR_OPEN_ORDER_INTERVAL - TIME_SLEEP_ERROR
    hb_intervals = {
        "open_order": MONITOR_OPEN_ORDER_INTERVAL,
    }
    monitored = _monitored_symbols()
    touch_module_heartbeat("monitororder", interval_sec=30.0, force=True, symbols=monitored, intervals=hb_intervals)

    while not _parse_bool_flag(getattr(api, "stop", False), False):
        try:
            monitored = _monitored_symbols()
            currenttime = time.time()
            if currenttime < monitor_open_orders_lasttime:
                monitor_open_orders_lasttime = currenttime - MONITOR_OPEN_ORDER_INTERVAL - TIME_SLEEP_ERROR
            if currenttime - monitor_open_orders_lasttime > MONITOR_OPEN_ORDER_INTERVAL:
                cycle_maps: list[tuple[dict, dict]] = []
                for symbol in monitored:
                    for side in ("SELL", "BUY"):
                        monitor_open_orders_by_type(symbol, side, _cycle_maps=cycle_maps)
                monitor_open_orders_lasttime = currenttime

            touch_module_heartbeat("monitororder", interval_sec=30.0, symbols=monitored, intervals=hb_intervals)
            time.sleep(MONITOR_OPEN_ORDER_INTERVAL)

        except Exception as e:
            err_prefix = "Provider API error" if isinstance(e, ProviderError) else "Error"
            print(f"{err_prefix}: {e}")
            touch_module_heartbeat("monitororder", interval_sec=30.0, symbols=_monitored_symbols(), intervals=hb_intervals)
            time.sleep(TIME_SLEEP_ERROR)


if __name__ == "__main__":
    monitor_orders()
