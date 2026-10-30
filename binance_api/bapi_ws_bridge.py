"""Real-time Binance user-data WebSocket to cache bridge and callbacks."""
from __future__ import annotations

import builtins
from collections import defaultdict
import sys
import threading
from typing import Any, Callable

from binance_api import bapi_ws_health as ws_health
from binance_api.bapi_ws_normalize import (
    parse_order_execution_report,
    parse_trade_execution_report,
)
from instrument_registry import casefold_get
import log  # noqa: F401 - keep print behavior aligned with the rest of the runtime

_ws_bridge = None
_ws_bridge_lock = threading.Lock()
_ws_event_stats: defaultdict[str, int] = defaultdict(int)


def _default_cache_getter(name: str, symbols=None, *, start_sync: bool = True):
    import cacheManager as cm

    if symbols is not None:
        return cm.get_cache_manager(name, symbols=symbols, start_sync=start_sync)
    return cm.get_cache_manager(name, start_sync=start_sync)


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
    """Normalize and persist an ``executionReport`` fill into the Order cache."""
    return _persist_ws_execution_report(
        event,
        cache_name="Order",
        parser=parse_order_execution_report,
        label="order",
        cache_getter=cache_getter,
    )


def append_trade_from_execution_report(
    event: dict[str, Any],
    *,
    cache_getter: Callable[..., Any] | None = None,
) -> bool:
    """Normalize and persist an ``executionReport`` fill into the Trade cache."""
    return _persist_ws_execution_report(
        event,
        cache_name="Trade",
        parser=parse_trade_execution_report,
        label="Trade",
        cache_getter=cache_getter,
    )


def refresh_asset_value_from_ws_event(
    *,
    cache_getter: Callable[..., Any] | None = None,
) -> None:
    """Refresh the AssetValue ``TOTAL`` entry after a balance event."""
    getter = cache_getter or _default_cache_getter
    asset_cache = getter("AssetValue", symbols=["TOTAL"])
    items = asset_cache.get_remote_items("TOTAL", None)
    if items:
        asset_cache._persist_items("TOTAL", items)


def persist_ws_updated_caches(
    event_type: str,
    *,
    cache_getter: Callable[..., Any] | None = None,
) -> None:
    """Flush caches affected by ``event_type`` when persistence is enabled."""
    getter = cache_getter or _default_cache_getter
    if event_type == "executionReport":
        getter("Order").save_state_to_file_if_enabled()
        getter("Trade").save_state_to_file_if_enabled()
    elif event_type in ("balanceUpdate", "outboundAccountPosition"):
        getter("AssetValue", symbols=["TOTAL"]).save_state_to_file_if_enabled()


def refresh_symbol_in_cache(manager: Any, symbol: str) -> None:
    """Refresh only one manager symbol efficiently for a WebSocket event."""
    try:
        start_time = casefold_get(
            manager.fetchtime_per_symbol, symbol, manager.fallback_time_default
        )
        items = manager.get_remote_items(symbol, start_time)
        if items:
            manager._persist_items(symbol, items)
    except Exception as exc:
        mgr_name = getattr(manager, "cls_name", None) or type(manager).__name__
        print(f"[cacheManager] _refresh_symbol_in_cache {mgr_name}/{symbol}: {exc}")


def handle_binance_ws_event(
    event: dict[str, Any],
    *,
    cache_getter: Callable[..., Any] | None = None,
    log_enabled: bool | None = None,
    upsert_order_fn: Callable[[dict[str, Any]], Any] | None = None,
    append_trade_fn: Callable[[dict[str, Any]], Any] | None = None,
    persist_fn: Callable[[str], Any] | None = None,
) -> None:
    """Dispatch one Binance user-data WebSocket event into the account caches."""
    if not isinstance(event, dict):
        return
    print("cacheManager handler call from binance ....")
    raw_event_type = event.get("e")
    if not isinstance(raw_event_type, str) or not raw_event_type.strip():
        return
    event_type = raw_event_type.strip()

    _ws_event_stats[event_type] += 1
    should_log = (
        ws_health.WS_EVENT_LOG_ENABLED
        if log_enabled is None
        else ws_health._as_bool(log_enabled, default=ws_health.WS_EVENT_LOG_ENABLED)
    )
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
        # Derive Order and Trade directly from the WebSocket payload without REST calls.
        # REST remains a rare fallback for missing symbols, while fallback polling covers
        # gaps created by WebSocket disconnections.
        if symbol:
            order_res = (upsert_order_fn or (lambda ev: upsert_order_from_execution_report(ev, cache_getter=getter)))(event)
            trade_res = (append_trade_fn or (lambda ev: append_trade_from_execution_report(ev, cache_getter=getter)))(event)
            should_persist = not (order_res is False and trade_res is False)
        else:
            for cache_name in ("Order", "Trade"):
                getter(cache_name).query_remote_and_update_cache()
            should_persist = True
        if should_persist:
            (persist_fn or (lambda et: persist_ws_updated_caches(et, cache_getter=getter)))(event_type)
        return

    if event_type in ("balanceUpdate", "outboundAccountPosition"):
        if should_log:
            print(f"[cacheManager][WS] {event_type} event received")
        return


def _bridge_is_usable(bridge: Any) -> bool:
    return bridge is not None and bool(getattr(bridge, "is_running", True))


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
        from binance_api import bapi_ws

        bridge = bapi_ws.BinanceAccountStream(
            on_event=on_event or handle_binance_ws_event,
            on_available=on_available or ws_health.mark_ws_available,
            on_healthy=on_healthy or ws_health.mark_ws_event_received,
            on_unhealthy=on_unhealthy or ws_health.mark_ws_unhealthy,
            loss_timeout_sec=(
                ws_health.WS_LOSS_TIMEOUT_SEC
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
    if sys.modules.get("_cacheManager_initialized"):
        print("[cacheManager] Already initialized, skip.")
        return
    sys.modules["_cacheManager_initialized"] = True
    try:
        (enable_fn or enable_real_ws_event_sync)()
    except BaseException:
        sys.modules.pop("_cacheManager_initialized", None)
        raise
    print("⚙️ cacheManager: WS user-data bridge started (execution reports).")


def stop_ws_bridge() -> None:
    """Stop the singleton user-data WebSocket bridge if active."""
    global _ws_bridge
    with _ws_bridge_lock:
        bridge = _ws_bridge
        _ws_bridge = None
        sys.modules.pop("_cacheManager_initialized", None)
        cm = sys.modules.get("cacheManager")
        if cm is not None and getattr(cm, "_ws_bridge", None) is bridge:
            cm._ws_bridge = None
        if bridge is not None:
            bridge.stop()
