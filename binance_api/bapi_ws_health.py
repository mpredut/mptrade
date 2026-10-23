"""Binance WebSocket health state and fallback REST polling policies."""
from __future__ import annotations

import builtins
import threading
import time

ORDER_CACHE_CLASSES = frozenset({"CacheOrder", "CacheOrderManager"})
TRADE_CACHE_CLASSES = frozenset({"CacheTrade", "CacheTradeManager"})
ASSET_VALUE_CACHE_CLASSES = frozenset({"CacheAssetValue", "CacheAssetValueManager"})
WS_MANAGED_CACHE_CLASSES = (
    ORDER_CACHE_CLASSES | TRADE_CACHE_CLASSES | ASSET_VALUE_CACHE_CLASSES
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
        _ws_available = bool(value)
        return _ws_available


def mark_ws_event_received(now: float | None = None) -> float:
    """Record a healthy user-data WebSocket heartbeat or event."""
    global _ws_last_event_ts, _ws_is_healthy
    ts = time.time() if now is None else float(now)
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
    with _ws_health_lock:
        if cls_name in ORDER_CACHE_CLASSES:
            _ws_needs_rest_resync.difference_update(ORDER_CACHE_CLASSES)
        elif cls_name in TRADE_CACHE_CLASSES:
            _ws_needs_rest_resync.difference_update(TRADE_CACHE_CLASSES)
        elif cls_name in ASSET_VALUE_CACHE_CLASSES:
            _ws_needs_rest_resync.difference_update(ASSET_VALUE_CACHE_CLASSES)
        else:
            _ws_needs_rest_resync.discard(cls_name)


def should_poll_for_manager(
    cls_name: str,
    *,
    ws_only_mode: bool | None = None,
    ws_available: bool | None = None,
    ws_is_healthy: bool | None = None,
) -> bool:
    """Return True when ``cls_name`` must run a REST reconciliation poll."""
    effective_ws_only = WS_ONLY_MODE if ws_only_mode is None else bool(ws_only_mode)
    if not effective_ws_only:
        return True
    if cls_name not in WS_MANAGED_CACHE_CLASSES:
        return True
    with _ws_health_lock:
        if cls_name in _ws_needs_rest_resync:
            return True
        effective_available = _ws_available if ws_available is None else bool(ws_available)
        effective_healthy = _ws_is_healthy if ws_is_healthy is None else bool(ws_is_healthy)
        return (not effective_available) or (not effective_healthy)


def get_ws_health_snapshot() -> dict:
    """Return an atomic snapshot of the current WS health state."""
    with _ws_health_lock:
        return {
            "ws_only_mode": WS_ONLY_MODE,
            "ws_available": _ws_available,
            "ws_last_event_ts": _ws_last_event_ts,
            "ws_is_healthy": _ws_is_healthy,
            "ws_needs_rest_resync": set(_ws_needs_rest_resync),
        }
