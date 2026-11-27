"""Real-time Binance user-data WebSocket to cache bridge and callbacks."""
from __future__ import annotations

import builtins
from collections import defaultdict
import sys
import threading
import time
from typing import Any, Callable

from binance_api import bapi_ws
from binance_api.bapi_ws_normalize import (
    parse_open_order_execution_report,
    parse_order_execution_report,
    parse_outbound_account_position,
    parse_trade_execution_report,
)
from botcore import _parse_bool_flag, normalize_class_name, safe_pos_float, safe_pos_int
from instrument_registry import casefold_get
from providers.ssot_bridge import ProviderSsotBridgeMixin
import log  # noqa: F401 -- keep print behavior aligned with the rest of the runtime

ORDER_CACHE_CLASSES = frozenset({"CacheOrderFilledManager"})
OPEN_ORDER_CACHE_CLASSES = frozenset({"CacheOrderOpenManager"})
TRADE_CACHE_CLASSES = frozenset({"CacheTradeManager"})
ASSET_VALUE_CACHE_CLASSES = frozenset({"CacheAssetValueManager"})
WS_MANAGED_CACHE_CLASSES = (
    ORDER_CACHE_CLASSES | OPEN_ORDER_CACHE_CLASSES | TRADE_CACHE_CLASSES | ASSET_VALUE_CACHE_CLASSES
)

# WS-only mode: when True, polling for Order/Trade/AssetValue is paused while WS is healthy.
WS_ONLY_MODE = False
WS_LOSS_TIMEOUT_SEC = 40
WS_EVENT_LOG_ENABLED = True

_ws_health_lock = threading.Lock()
_ws_available = False
_ws_last_event_ts = 0.0
_ws_is_healthy = False
_ws_needs_rest_resync: set[str] = set()


def mark_ws_available(value: bool) -> bool:
    """Record whether the Binance user-data WebSocket stream is connected."""
    global _ws_available
    with _ws_health_lock:
        _ws_available = _parse_bool_flag(value)
        return _ws_available


def mark_ws_event_received(now: float | None = None) -> float:
    """Record a healthy user-data WebSocket heartbeat or event."""
    global _ws_last_event_ts, _ws_is_healthy
    ts = safe_pos_float(now, default=None) or time.time()
    with _ws_health_lock:
        _ws_last_event_ts = ts
        _ws_is_healthy = True
        return _ws_last_event_ts


def mark_ws_unhealthy() -> None:
    """Mark the user-data WebSocket unhealthy and schedule a mandatory REST resync."""
    builtins.print("[cacheManager][WS] marked unhealthy")
    global _ws_is_healthy
    with _ws_health_lock:
        _ws_is_healthy = False
        _ws_needs_rest_resync.update(WS_MANAGED_CACHE_CLASSES)


def clear_ws_rest_resync_for_manager(cls_name: str) -> None:
    """Clear the pending post-disconnect REST resync obligation for ``cls_name``."""
    norm_name = normalize_class_name(cls_name)
    if not norm_name:
        return
    with _ws_health_lock:
        if norm_name in ORDER_CACHE_CLASSES:
            _ws_needs_rest_resync.difference_update(ORDER_CACHE_CLASSES)
        elif norm_name in OPEN_ORDER_CACHE_CLASSES:
            _ws_needs_rest_resync.difference_update(OPEN_ORDER_CACHE_CLASSES)
        elif norm_name in TRADE_CACHE_CLASSES:
            _ws_needs_rest_resync.difference_update(TRADE_CACHE_CLASSES)
        elif norm_name in ASSET_VALUE_CACHE_CLASSES:
            _ws_needs_rest_resync.difference_update(ASSET_VALUE_CACHE_CLASSES)
        else:
            _ws_needs_rest_resync.discard(norm_name)


def should_poll_for_manager(
    cls_name: str,
    *,
    ws_only_mode: bool | None = None,
    ws_available: bool | None = None,
    ws_is_healthy: bool | None = None,
) -> bool:
    """Return True when ``cls_name`` must run a REST reconciliation poll."""
    norm_name = normalize_class_name(cls_name)
    effective_ws_only = _parse_bool_flag(WS_ONLY_MODE) if ws_only_mode is None else _parse_bool_flag(ws_only_mode)
    if not effective_ws_only:
        return True
    if norm_name not in WS_MANAGED_CACHE_CLASSES:
        return True
    with _ws_health_lock:
        if norm_name in _ws_needs_rest_resync:
            return True
        effective_available = _parse_bool_flag(_ws_available) if ws_available is None else _parse_bool_flag(ws_available)
        effective_healthy = _parse_bool_flag(_ws_is_healthy) if ws_is_healthy is None else _parse_bool_flag(ws_is_healthy)
        return (not effective_available) or (not effective_healthy)


_ws_bridge = None
_ws_bridge_lock = threading.Lock()
_ws_event_stats: defaultdict[str, int] = defaultdict(int)


def _default_cache_getter(name: str, symbols=None):
    if symbols is not None:
        return ProviderSsotBridgeMixin.get_ssot_cache_manager(name, symbols=symbols)
    return ProviderSsotBridgeMixin.get_ssot_cache_manager(name)


def _persist_ws_execution_report(
    event: dict[str, Any],
    *,
    cache_name: str,
    parser: Callable[[dict[str, Any]], tuple[str, dict[str, Any]] | None],
    label: str,
    cache_getter: Callable[..., Any] | None = None,
) -> bool:
    parsed = parser(event)
    if parsed is None:
        return False
    symbol, item = parsed
    getter = cache_getter or _default_cache_getter
    try:
        cache = getter(cache_name)
        if not cache._is_valid_item(item):
            builtins.print(f"[cacheManager][WS] rejected malformed {label} report")
            return False
        cache._persist_items(symbol, [item])
        return True
    except Exception as exc:
        builtins.print(f"[cacheManager][WS] {label.lower()} report persist failed: {exc}")
        return False


def upsert_order_from_execution_report(
    event: dict[str, Any],
    *,
    cache_getter: Callable[..., Any] | None = None,
) -> bool:
    """Normalize and persist an ``executionReport`` fill into the filled-order cache."""
    return _persist_ws_execution_report(
        event,
        cache_name="BinanceOrderFilled",
        parser=parse_order_execution_report,
        label="order",
        cache_getter=cache_getter,
    )


def apply_open_order_from_execution_report(
    event: dict[str, Any],
    *,
    cache_getter: Callable[..., Any] | None = None,
) -> bool:
    """Normalize and apply an ``executionReport`` lifecycle transition into the open-order cache."""
    parsed = parse_open_order_execution_report(event)
    if parsed is None:
        return False
    symbol, action, item = parsed
    getter = cache_getter or _default_cache_getter
    try:
        cache = getter("BinanceOrderOpen")
        apply_fn = getattr(cache, "apply_ws_order_update", None)
        if not callable(apply_fn):
            return False
        return bool(apply_fn(symbol, action, item))
    except Exception as exc:
        builtins.print(f"[cacheManager][WS] open order report update failed: {exc}")
        return False


def apply_balances_from_outbound_account_position(
    event: dict[str, Any],
    *,
    cache_getter: Callable[..., Any] | None = None,
) -> bool:
    """Normalize and apply an ``outboundAccountPosition`` event atomically into ``BinanceBalance``."""
    parsed = parse_outbound_account_position(event)
    if parsed is None:
        return False
    updates, ev_ts = parsed
    getter = cache_getter or _default_cache_getter
    try:
        cache = getter("BinanceBalance")
        apply_fn = getattr(cache, "apply_ws_balance_updates", None)
        if not callable(apply_fn):
            return False
        return bool(apply_fn(updates, event_ts=ev_ts))
    except Exception as exc:
        builtins.print(f"[cacheManager][WS] balance position update failed: {exc}")
        return False


def append_trade_from_execution_report(
    event: dict[str, Any],
    *,
    cache_getter: Callable[..., Any] | None = None,
) -> bool:
    """Normalize and persist an ``executionReport`` fill into the Trade cache."""
    return _persist_ws_execution_report(
        event,
        cache_name="BinanceTrade",
        parser=parse_trade_execution_report,
        label="Trade",
        cache_getter=cache_getter,
    )


def refresh_symbol_in_cache(manager: Any, symbol: str) -> None:
    """Refresh only one manager symbol efficiently for a WebSocket event."""
    try:
        raw_ft = casefold_get(manager.fetchtime_per_symbol, symbol, None)
        start_time = safe_pos_int(raw_ft, default=None) or manager.fallback_time_default
        items = manager.get_remote_items(symbol, start_time)
        if items:
            manager._persist_items(symbol, items)
    except Exception as exc:
        mgr_name = getattr(manager, "cls_name", None) or type(manager).__name__
        print(f"[cacheManager] refresh_symbol_in_cache {mgr_name}/{symbol}: {exc}")


def handle_binance_ws_event(
    event: dict[str, Any],
    *,
    cache_getter: Callable[..., Any] | None = None,
    log_enabled: bool | None = None,
    upsert_order_fn: Callable[[dict[str, Any]], Any] | None = None,
    append_trade_fn: Callable[[dict[str, Any]], Any] | None = None,
    apply_open_order_fn: Callable[[dict[str, Any]], Any] | None = None,
    apply_balance_fn: Callable[[dict[str, Any]], Any] | None = None,
    refresh_balance_fn: Callable[[dict[str, Any]], Any] | None = None,
    persist_fn: Callable[[str], Any] | None = None,
) -> None:
    """Dispatch one Binance user-data WebSocket event into the account caches."""
    if not isinstance(event, dict):
        return
    raw_event_type = event.get("e")
    if not isinstance(raw_event_type, str) or not raw_event_type.strip():
        return
    event_type = raw_event_type.strip()

    _ws_event_stats[event_type] += 1
    should_log = (
        WS_EVENT_LOG_ENABLED
        if log_enabled is None
        else _parse_bool_flag(log_enabled, default=WS_EVENT_LOG_ENABLED)
    )
    if should_log:
        print("cacheManager handler call from binance ....")
    getter = cache_getter or _default_cache_getter

    if event_type == "executionReport":
        raw_sym = event.get("s")
        symbol = raw_sym.strip().upper() if isinstance(raw_sym, str) else ""
        if should_log:
            print(
                "[cacheManager][WS] executionReport "
                f"symbol={symbol} orderId={event.get('i')} "
                f"status={event.get('X')} execType={event.get('x')} side={event.get('S')}"
            )
        # Derive filled orders, open orders, and trade rows directly from the WebSocket payload without REST calls.
        # REST remains a rare fallback for missing symbols, while fallback polling covers
        # gaps created by WebSocket disconnections.
        open_res = False
        if symbol:
            order_res = (upsert_order_fn or (lambda ev: upsert_order_from_execution_report(ev, cache_getter=getter)))(event)
            trade_res = (append_trade_fn or (lambda ev: append_trade_from_execution_report(ev, cache_getter=getter)))(event)
            if apply_open_order_fn is not None:
                open_res = bool(apply_open_order_fn(event))
            elif upsert_order_fn is None and append_trade_fn is None:
                open_res = apply_open_order_from_execution_report(event, cache_getter=getter)
            should_persist = not (order_res is False and trade_res is False and open_res is False)
        else:
            for cache_name in ("BinanceOrderFilled", "BinanceTrade"):
                getter(cache_name).query_remote_and_update_cache()
            try:
                getter("BinanceOrderOpen").query_remote_and_update_cache()
                open_res = True
            except Exception:
                pass
            should_persist = True
        if should_persist:
            if persist_fn is not None:
                persist_fn(event_type)
            else:
                if not symbol or order_res is not False:
                    try:
                        getter("BinanceOrderFilled").save_state_to_file_if_enabled()
                    except Exception as exc:
                        builtins.print(f"[cacheManager][WS] BinanceOrderFilled save failed: {exc}")
                if not symbol or trade_res is not False:
                    try:
                        getter("BinanceTrade").save_state_to_file_if_enabled()
                    except Exception as exc:
                        builtins.print(f"[cacheManager][WS] BinanceTrade save failed: {exc}")
                if open_res:
                    try:
                        getter("BinanceOrderOpen").save_state_to_file_if_enabled()
                    except Exception as exc:
                        builtins.print(f"[cacheManager][WS] BinanceOrderOpen save failed: {exc}")
        return

    if event_type == "outboundAccountPosition":
        if should_log:
            print(f"[cacheManager][WS] {event_type} event received")
        bal_res = (
            apply_balance_fn(event)
            if apply_balance_fn is not None
            else apply_balances_from_outbound_account_position(event, cache_getter=getter)
        )
        if bal_res and persist_fn is not None:
            persist_fn(event_type)
        return

    if event_type == "balanceUpdate":
        if should_log:
            print(f"[cacheManager][WS] {event_type} event received")
        ref_res = False
        if refresh_balance_fn is not None:
            ref_res = bool(refresh_balance_fn(event))
        else:
            try:
                cache = getter("BinanceBalance")
                req_fn = getattr(cache, "request_ws_balance_refresh", None)
                if callable(req_fn):
                    ref_res = bool(req_fn())
            except Exception as exc:
                builtins.print(f"[cacheManager][WS] balance refresh request failed: {exc}")
        if ref_res and persist_fn is not None:
            persist_fn(event_type)
        return


def _bridge_is_usable(bridge: Any) -> bool:
    if bridge is None:
        return False
    raw = getattr(bridge, "is_running", True)
    if callable(raw):
        raw = raw()
    return bool(_parse_bool_flag(raw, False))


def enable_real_ws_event_sync(
    *,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    on_available: Callable[[bool], Any] | None = None,
    on_healthy: Callable[[], Any] | None = None,
    on_unhealthy: Callable[[], Any] | None = None,
    loss_timeout_sec: float | None = None,
):
    """Start the singleton ``BinanceAccountStream`` bridge if not already running."""
    global _ws_bridge
    if sys.modules.get("_cacheManager_initialized"):
        if _bridge_is_usable(_ws_bridge):
            return _ws_bridge
    with _ws_bridge_lock:
        if _bridge_is_usable(_ws_bridge):
            return _ws_bridge

        bridge = bapi_ws.BinanceAccountStream(
            on_event=on_event or handle_binance_ws_event,
            on_available=on_available or mark_ws_available,
            on_healthy=on_healthy or mark_ws_event_received,
            on_unhealthy=on_unhealthy or mark_ws_unhealthy,
            loss_timeout_sec=(
                WS_LOSS_TIMEOUT_SEC
                if loss_timeout_sec is None
                else loss_timeout_sec
            ),
        )
        bridge.start()
        _ws_bridge = bridge
        return _ws_bridge


def initialize_ws_bridge_once(
    *,
    enable_fn: Callable[[], Any] | None = None,
) -> None:
    """Idempotently start the user-data WebSocket bridge once per process."""
    already_init = bool(sys.modules.get("_cacheManager_initialized"))
    if not already_init:
        sys.modules["_cacheManager_initialized"] = True
        try:
            (enable_fn or enable_real_ws_event_sync)()
        except BaseException:
            sys.modules.pop("_cacheManager_initialized", None)
            raise
    print(
        "[cacheManager] Already initialized, skip."
        if already_init
        else "cacheManager: WS user-data bridge started (execution reports)."
    )

