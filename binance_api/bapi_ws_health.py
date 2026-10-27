"""Binance WebSocket health state and fallback REST polling policies."""
from __future__ import annotations

import builtins
import math
import sys
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


def _as_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    fallback = bool(default) if isinstance(default, bool) else False
    if value is None or isinstance(value, (dict, list, tuple, set, frozenset)):
        return fallback
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("1", "true", "yes", "on", "y", "t"):
            return True
        if s in ("0", "false", "no", "off", "n", "f", "", "none", "null", "nan"):
            return False
        return fallback
    if isinstance(value, (int, float)):
        f = float(value)
        return (f != 0.0) if math.isfinite(f) else fallback
    return fallback


def mark_ws_available(value: bool) -> bool:
    """Record whether the Binance user-data WebSocket stream is connected."""
    global _ws_available
    with _ws_health_lock:
        _ws_available = _as_bool(value)
        cm = sys.modules.get("cacheManager")
        if cm is not None:
            cm._ws_available = _ws_available
        return _ws_available


def mark_ws_event_received(now: float | None = None) -> float:
    """Record a healthy user-data WebSocket heartbeat or event."""
    global _ws_last_event_ts, _ws_is_healthy
    ts: float | None = None
    if now is not None and not isinstance(now, bool):
        try:
            cand = float(now)
            if math.isfinite(cand) and cand > 0:
                ts = cand
        except (TypeError, ValueError, OverflowError):
            pass
    if ts is None:
        ts = time.time()
    with _ws_health_lock:
        _ws_last_event_ts = ts
        _ws_is_healthy = True
        cm = sys.modules.get("cacheManager")
        if cm is not None:
            cm._ws_last_event_ts = ts
            cm._ws_is_healthy = True
        return _ws_last_event_ts


def mark_ws_unhealthy() -> None:
    """Mark the user-data WebSocket unhealthy and schedule a mandatory REST resync."""
    builtins.print("[cacheManager][WS] marked unhealthy")
    global _ws_is_healthy
    with _ws_health_lock:
        _ws_is_healthy = False
        _ws_needs_rest_resync.update(WS_MANAGED_CACHE_CLASSES)
        cm = sys.modules.get("cacheManager")
        if cm is not None:
            cm._ws_is_healthy = False


def _normalize_manager_name(cls_name: object) -> str:
    if isinstance(cls_name, type):
        return cls_name.__name__
    if isinstance(cls_name, str):
        return cls_name.strip()
    return ""


def clear_ws_rest_resync_for_manager(cls_name: str) -> None:
    """Clear the pending post-disconnect REST resync obligation for ``cls_name``."""
    norm_name = _normalize_manager_name(cls_name)
    if not norm_name:
        return
    with _ws_health_lock:
        if norm_name in ORDER_CACHE_CLASSES:
            _ws_needs_rest_resync.difference_update(ORDER_CACHE_CLASSES)
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
    norm_name = _normalize_manager_name(cls_name)
    effective_ws_only = _as_bool(WS_ONLY_MODE) if ws_only_mode is None else _as_bool(ws_only_mode)
    if not effective_ws_only:
        return True
    if norm_name not in WS_MANAGED_CACHE_CLASSES:
        return True
    with _ws_health_lock:
        if norm_name in _ws_needs_rest_resync:
            return True
        effective_available = _as_bool(_ws_available) if ws_available is None else _as_bool(ws_available)
        effective_healthy = _as_bool(_ws_is_healthy) if ws_is_healthy is None else _as_bool(ws_is_healthy)
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
