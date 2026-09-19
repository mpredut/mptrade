"""Compact cross-process freshness state for Binance account caches.

The dedicated cacheManager process is the sole writer. Trading processes read the
atomic snapshot before live order actions and fail closed unless both the Order and
Trade caches were reconciled successfully within the configured age. File mtime is
intentionally ignored because it does not prove that Binance confirmed account state.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import sys
import threading
import time
from typing import Callable, Optional

from botcore import load_dotenv, required_float_env
from state_io import atomic_write_json


_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_ROOT, "config.env"))


def _positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


MAX_AGE_SEC = _positive_finite(
    required_float_env("CM_BINANCE_ACCOUNT_CACHE_MAX_AGE_SEC"),
    "CM_BINANCE_ACCOUNT_CACHE_MAX_AGE_SEC")
SUBMIT_PERMIT_TTL_SEC = _positive_finite(
    required_float_env("CM_BINANCE_ACCOUNT_CACHE_PERMIT_TTL_SEC"),
    "CM_BINANCE_ACCOUNT_CACHE_PERMIT_TTL_SEC")
if SUBMIT_PERMIT_TTL_SEC >= MAX_AGE_SEC:
    raise ValueError("account-cache permit TTL must be below cache maximum age")

SCHEMA_VERSION = 2
HEALTH_PATH = os.path.join(
    _ROOT, "cachedb", "binance_account_cache_health.json")
_PROCESS_IDENTITY_REQUIRED = sys.platform.startswith("linux")
_SYNC_FIELDS = {
    "Order": ("order_sync_at_ms", "order_cache_version"),
    "CacheOrderManager": ("order_sync_at_ms", "order_cache_version"),
    "Trade": ("trade_sync_at_ms", "trade_cache_version"),
    "CacheTradeManager": ("trade_sync_at_ms", "trade_cache_version"),
}


class AccountCacheNotReady(RuntimeError):
    """The shared Binance account snapshot cannot authorize a live submit."""

    def __init__(self, reason: str):
        self.reason = str(reason or "account_cache_not_fresh")
        super().__init__(self.reason)


@dataclass(frozen=True)
class CacheHealthStatus:
    """Validated health result returned to every Binance trading process."""

    ready: bool
    reason: str
    order_age_sec: Optional[float] = None
    trade_age_sec: Optional[float] = None
    order_cache_version: str = ""
    trade_cache_version: str = ""


_writer_lock = threading.Lock()
_writer_enabled = False
_writer_path = HEALTH_PATH
_writer_state: dict = {}


def _now_ms() -> int:
    return int(time.time() * 1000)

def _pid_started_at_ms(pid: int) -> Optional[int]:
    """Return the Linux process start time, or None when it is unavailable."""
    try:
        with open(f"/proc/{int(pid)}/stat", "r", encoding="utf-8") as handle:
            stat = handle.read()
        # The command name can contain spaces and parentheses. Fields after the final
        # closing parenthesis start at field 3; process start ticks are field 22.
        start_ticks = int(stat[stat.rfind(")") + 2:].split()[19])
        with open("/proc/stat", "r", encoding="utf-8") as handle:
            boot_seconds = next(
                int(line.split()[1]) for line in handle if line.startswith("btime ")
            )
        ticks_per_second = int(os.sysconf("SC_CLK_TCK"))
        return int((boot_seconds + start_ticks / ticks_per_second) * 1000)
    except (OSError, StopIteration, TypeError, ValueError, IndexError):
        return None


def enable_writer(*, path: str = HEALTH_PATH, pid: Optional[int] = None,
                  now_ms: Optional[int] = None) -> None:
    """Enable the sole cacheManager writer and invalidate the old generation."""
    global _writer_enabled, _writer_path, _writer_state
    timestamp = _now_ms() if now_ms is None else int(now_ms)
    with _writer_lock:
        _writer_path = os.fspath(path)
        writer_pid = os.getpid() if pid is None else int(pid)
        process_started_at_ms = _pid_started_at_ms(writer_pid)
        if _PROCESS_IDENTITY_REQUIRED and process_started_at_ms is None:
            _writer_enabled = False
            _writer_state = {}
            raise AccountCacheNotReady(
                "account_cache_writer_identity_unavailable")
        _writer_state = {
            "schema_version": SCHEMA_VERSION,
            "writer_pid": writer_pid,
            "writer_started_at_ms": process_started_at_ms or timestamp,
            "published_at_ms": timestamp,
            "generation": 0,
            "stopping": False,
            "order_sync_at_ms": 0,
            "trade_sync_at_ms": 0,
            "order_cache_version": "",
            "trade_cache_version": "",
        }
        _writer_enabled = True
        _publish_locked()


def disable_writer(*, now_ms: Optional[int] = None) -> bool:
    """Publish a fail-closed stopping state before disabling the writer."""
    global _writer_enabled, _writer_state
    with _writer_lock:
        published = False
        try:
            if _writer_enabled and _writer_state:
                _writer_state["stopping"] = True
                _writer_state["published_at_ms"] = (
                    _now_ms() if now_ms is None else int(now_ms)
                )
                _writer_state["generation"] += 1
                _publish_locked()
                published = True
        except Exception:
            published = False
        finally:
            _writer_enabled = False
            _writer_state = {}
        return published


def _publish_locked() -> None:
    atomic_write_json(
        _writer_path,
        _writer_state,
        separators=(",", ":"),
        sort_keys=True,
    )


def record_successful_sync(cache_name: str, data_version: str, *,
                           now_ms: Optional[int] = None) -> bool:
    """Publish one complete, persisted Order or Trade reconciliation success."""
    fields = _SYNC_FIELDS.get(str(cache_name))
    version = str(data_version or "")
    if fields is None or not version:
        return False
    sync_timestamp = _now_ms() if now_ms is None else int(now_ms)
    sync_field, version_field = fields
    with _writer_lock:
        if not _writer_enabled:
            return False
        _writer_state[sync_field] = sync_timestamp
        _writer_state[version_field] = version
        # Publication time describes this atomic write, while the conservative
        # synchronization timestamp remains the first request start of the cycle.
        _writer_state["published_at_ms"] = _now_ms()
        _writer_state["generation"] += 1
        _publish_locked()
    return True


def record_persisted_version(cache_name: str, data_version: str, *,
                             now_ms: Optional[int] = None) -> bool:
    """Publish durable content without claiming a new REST reconciliation."""
    fields = _SYNC_FIELDS.get(str(cache_name))
    version = str(data_version or "")
    if fields is None or not version:
        return False
    _sync_field, version_field = fields
    timestamp = _now_ms() if now_ms is None else int(now_ms)
    with _writer_lock:
        if not _writer_enabled:
            return False
        _writer_state[version_field] = version
        _writer_state["published_at_ms"] = timestamp
        _writer_state["generation"] += 1
        _publish_locked()
    return True


def _positive_int(value, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"invalid {field}")
    return value


def _nonnegative_int(value, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"invalid {field}")
    return value


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def inspect_health(*, path: str = HEALTH_PATH,
                   now_ms: Optional[int] = None,
                   max_age_sec: float = MAX_AGE_SEC,
                   pid_is_alive: Callable[[int], bool] = _pid_is_alive,
                   pid_started_at_ms: Callable[
                       [int], Optional[int]] = _pid_started_at_ms,
                   future_tolerance_sec: float = 5.0) -> CacheHealthStatus:
    """Validate the shared snapshot without trusting mtime or defaults."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            state = json.load(handle)
        if not isinstance(state, dict):
            raise ValueError("root is not an object")
        if state.get("schema_version") != SCHEMA_VERSION:
            return CacheHealthStatus(False, "account_cache_schema_mismatch")

        writer_pid = _positive_int(state.get("writer_pid"), "writer_pid")
        writer_started_ms = _positive_int(
            state.get("writer_started_at_ms"), "writer_started_at_ms")
        generation = _nonnegative_int(state.get("generation"), "generation")
        _positive_int(state.get("published_at_ms"), "published_at_ms")
        stopping = state.get("stopping")
        if not isinstance(stopping, bool):
            raise ValueError("invalid stopping")
        order_sync_ms = _nonnegative_int(
            state.get("order_sync_at_ms"), "order_sync_at_ms")
        trade_sync_ms = _nonnegative_int(
            state.get("trade_sync_at_ms"), "trade_sync_at_ms")
        order_version = state.get("order_cache_version")
        trade_version = state.get("trade_cache_version")
        if not isinstance(order_version, str) or not isinstance(trade_version, str):
            raise ValueError("invalid cache version")
        initializing = (
            generation < 2 or not order_sync_ms or not trade_sync_ms
            or not order_version or not trade_version
        )
    except FileNotFoundError:
        return CacheHealthStatus(False, "account_cache_health_missing")
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return CacheHealthStatus(False, "account_cache_health_invalid")

    if not pid_is_alive(writer_pid):
        return CacheHealthStatus(False, "account_cache_writer_not_running")
    observed_start_ms = pid_started_at_ms(writer_pid)
    if _PROCESS_IDENTITY_REQUIRED and observed_start_ms is None:
        return CacheHealthStatus(
            False, "account_cache_writer_identity_unavailable")
    if (observed_start_ms is not None
            and abs(observed_start_ms - writer_started_ms) > 2_000):
        return CacheHealthStatus(False, "account_cache_writer_identity_changed")
    if stopping:
        return CacheHealthStatus(False, "account_cache_writer_stopping")
    if initializing:
        return CacheHealthStatus(False, "account_cache_initializing")

    current_ms = _now_ms() if now_ms is None else int(now_ms)
    future_tolerance_ms = int(float(future_tolerance_sec) * 1000)
    if (order_sync_ms > current_ms + future_tolerance_ms
            or trade_sync_ms > current_ms + future_tolerance_ms):
        return CacheHealthStatus(
            False, "account_cache_timestamp_in_future")

    order_age = max(0.0, (current_ms - order_sync_ms) / 1000.0)
    trade_age = max(0.0, (current_ms - trade_sync_ms) / 1000.0)
    if order_age > max_age_sec:
        return CacheHealthStatus(
            False, "order_cache_stale", order_age, trade_age)
    if trade_age > max_age_sec:
        return CacheHealthStatus(
            False, "trade_cache_stale", order_age, trade_age)
    return CacheHealthStatus(
        True, "", order_age, trade_age, order_version, trade_version)


def require_fresh_account_cache(**kwargs) -> CacheHealthStatus:
    """Return fresh status or raise a definitive pre-submit refusal."""
    status = inspect_health(**kwargs)
    if not status.ready:
        raise AccountCacheNotReady(status.reason)
    return status
