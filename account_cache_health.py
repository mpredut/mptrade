"""Compact cross-process freshness state for Binance account caches.

The dedicated cacheManager process is the sole writer. Trading processes read the
atomic snapshot before live order actions and fail closed unless both the Order and
Trade caches were reconciled successfully within the configured age. File mtime is
intentionally ignored because it does not prove that Binance confirmed account state.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
import sys
import threading
from typing import Callable, Optional

from botcore import (
    INVALID_SCALAR_STRINGS,
    exact_pos_int,
    load_dotenv,
    normalize_class_name,
    require_int,
    require_pos_float,
    required_float_env,
    resolve_now_ms,
    safe_nonneg_float,
    safe_pos_float,
)
from cacheintegrity import AccountCacheNotReady
from cachepaths import CACHE_ACCOUNT_HEALTH_FILENAME, cache_path, resolve_cachedb_filepath
from state_io import atomic_write_json, parse_proc_stat_fields, read_json_dict, read_text_file, read_text_lines


load_dotenv()

MAX_AGE_SEC = require_pos_float(
    required_float_env("CM_BINANCE_ACCOUNT_CACHE_MAX_AGE_SEC"),
    "CM_BINANCE_ACCOUNT_CACHE_MAX_AGE_SEC",
)
SUBMIT_PERMIT_TTL_SEC = require_pos_float(
    required_float_env("CM_BINANCE_ACCOUNT_CACHE_PERMIT_TTL_SEC"),
    "CM_BINANCE_ACCOUNT_CACHE_PERMIT_TTL_SEC",
)
if SUBMIT_PERMIT_TTL_SEC >= MAX_AGE_SEC:
    raise ValueError("account-cache permit TTL must be below cache maximum age")

SCHEMA_VERSION = 2
_DEFAULT_HEALTH_PATH = cache_path(CACHE_ACCOUNT_HEALTH_FILENAME)
HEALTH_PATH = _DEFAULT_HEALTH_PATH
_PROCESS_IDENTITY_REQUIRED = sys.platform.startswith("linux")
_DEAD_PROC_STATES = frozenset({"Z", "X", "x"})
_SYNC_FIELDS = {
    "BinanceOrder": ("order_sync_at_ms", "order_cache_version"),
    "CacheOrderManager": ("order_sync_at_ms", "order_cache_version"),
    "BinanceTrade": ("trade_sync_at_ms", "trade_cache_version"),
    "CacheTradeManager": ("trade_sync_at_ms", "trade_cache_version"),
}


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


def _pid_started_at_ms(pid: int) -> Optional[int]:
    """Return the Linux process start time, or None when it is unavailable."""
    pid_val = exact_pos_int(pid, default=None)
    if pid_val is None:
        return None
    try:
        # Fields after the final ')' start at field 3; process start ticks are field 22 (index 19).
        _, fields = parse_proc_stat_fields(read_text_file(f"/proc/{pid_val}/stat"), min_fields=20)
        if fields[0] in _DEAD_PROC_STATES:
            return None
        start_ticks = exact_pos_int(fields[19], default=None)
        if start_ticks is None:
            return None
        proc_lines = read_text_lines("/proc/stat")
        if not proc_lines:
            return None
        boot_seconds = next(
            btime
            for line in proc_lines
            if line.startswith("btime ")
            and len(parts := line.split()) >= 2
            and (btime := exact_pos_int(parts[1], default=None)) is not None
        )
        clk_tck = exact_pos_int(os.sysconf("SC_CLK_TCK"), default=None)
        if clk_tck is None:
            return None
        return math.trunc((boot_seconds + start_ticks / clk_tck) * 1000)
    except (OSError, StopIteration, TypeError, ValueError, IndexError):
        return None


def enable_writer(*, path: Optional[str] = None, pid: Optional[int] = None, now_ms: Optional[int] = None) -> None:
    """Enable the sole cacheManager writer and invalidate the old generation."""
    global _writer_enabled, _writer_path, _writer_state
    timestamp = resolve_now_ms(now_ms)
    with _writer_lock:
        target_p = HEALTH_PATH if path is None else os.fspath(path)
        _writer_path = cache_path(CACHE_ACCOUNT_HEALTH_FILENAME) if target_p == _DEFAULT_HEALTH_PATH else str(resolve_cachedb_filepath(target_p))
        writer_pid = exact_pos_int(pid, default=None) or os.getpid()
        process_started_at_ms = _pid_started_at_ms(writer_pid)
        if _PROCESS_IDENTITY_REQUIRED and process_started_at_ms is None:
            _writer_enabled, _writer_state = False, {}
            raise AccountCacheNotReady("account_cache_writer_identity_unavailable")
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
        _publish_locked(bump_generation=False)


def disable_writer(*, now_ms: Optional[int] = None) -> bool:
    """Publish a fail-closed stopping state before disabling the writer."""
    global _writer_enabled, _writer_state
    with _writer_lock:
        published = False
        try:
            if _writer_enabled and _writer_state:
                _writer_state["stopping"] = True
                _writer_state["published_at_ms"] = resolve_now_ms(now_ms)
                published = _publish_locked()
        except Exception:
            published = False
        finally:
            _writer_enabled = False
            _writer_state = {}
        return published


def _publish_locked(*, bump_generation: bool = True) -> bool:
    if bump_generation:
        _writer_state["generation"] += 1
    atomic_write_json(_writer_path, _writer_state, separators=(",", ":"), sort_keys=True)
    return True


def _clean_version_str(value: object) -> str:
    if not isinstance(value, str):
        return ""
    s = value.strip()
    return "" if (not s or s.lower() in INVALID_SCALAR_STRINGS) else s


def _record_cache_version(
    cache_name: str,
    data_version: str,
    *,
    now_ms: Optional[int] = None,
    update_sync: bool = False,
) -> bool:
    norm_name = normalize_class_name(cache_name)
    version = _clean_version_str(data_version)
    fields = _SYNC_FIELDS.get(norm_name) if (norm_name and version) else None
    if fields is None:
        return False
    sync_field, version_field = fields
    timestamp = resolve_now_ms(now_ms)
    with _writer_lock:
        if not _writer_enabled:
            return False
        if update_sync:
            _writer_state[sync_field] = timestamp
            # Publication time describes this atomic write, while the conservative
            # synchronization timestamp remains the first request start of the cycle.
            _writer_state["published_at_ms"] = resolve_now_ms()
        else:
            _writer_state["published_at_ms"] = timestamp
        _writer_state[version_field] = version
        return _publish_locked()


def record_successful_sync(cache_name: str, data_version: str, *, now_ms: Optional[int] = None) -> bool:
    """Publish one complete, persisted Order or Trade reconciliation success."""
    return _record_cache_version(cache_name, data_version, now_ms=now_ms, update_sync=True)


def record_persisted_version(cache_name: str, data_version: str, *, now_ms: Optional[int] = None) -> bool:
    """Publish durable content without claiming a new REST reconciliation."""
    return _record_cache_version(cache_name, data_version, now_ms=now_ms, update_sync=False)


def _pid_is_alive(pid: int) -> bool:
    pid_val = exact_pos_int(pid, default=None)
    if pid_val is None:
        return False
    try:
        os.kill(pid_val, 0)
    except PermissionError:
        return True
    except (ProcessLookupError, OverflowError, ValueError, TypeError, OSError):
        return False
    try:
        _, fields = parse_proc_stat_fields(read_text_file(f"/proc/{pid_val}/stat"), min_fields=1)
    except PermissionError:
        return True
    except (FileNotFoundError, ProcessLookupError):
        return not os.path.isdir("/proc")
    except (OSError, ValueError):
        return False
    return fields[0] not in _DEAD_PROC_STATES


def inspect_health(
    *,
    path: Optional[str] = None,
    now_ms: Optional[int] = None,
    max_age_sec: float = MAX_AGE_SEC,
    pid_is_alive: Callable[[int], bool] = _pid_is_alive,
    pid_started_at_ms: Callable[[int], Optional[int]] = _pid_started_at_ms,
    future_tolerance_sec: float = 5.0,
) -> CacheHealthStatus:
    """Validate the shared snapshot without trusting mtime or defaults."""
    target_p = HEALTH_PATH if path is None else os.fspath(path)
    eff_path = cache_path(CACHE_ACCOUNT_HEALTH_FILENAME) if target_p == _DEFAULT_HEALTH_PATH else str(resolve_cachedb_filepath(target_p))
    try:
        state = read_json_dict(eff_path)
        if state.get("schema_version") != SCHEMA_VERSION:
            return CacheHealthStatus(False, "account_cache_schema_mismatch")

        writer_pid = require_int(state.get("writer_pid"), "writer_pid", min_value=1)
        writer_started_ms = require_int(state.get("writer_started_at_ms"), "writer_started_at_ms", min_value=1)
        generation = require_int(state.get("generation"), "generation", min_value=0)
        require_int(state.get("published_at_ms"), "published_at_ms", min_value=1)
        stopping = state.get("stopping")
        if not isinstance(stopping, bool):
            raise ValueError("invalid stopping")
        order_sync_ms = require_int(state.get("order_sync_at_ms"), "order_sync_at_ms", min_value=0)
        trade_sync_ms = require_int(state.get("trade_sync_at_ms"), "trade_sync_at_ms", min_value=0)
        order_version = state.get("order_cache_version")
        trade_version = state.get("trade_cache_version")
        if not isinstance(order_version, str) or not isinstance(trade_version, str):
            raise ValueError("invalid cache version")
        order_version = _clean_version_str(order_version)
        trade_version = _clean_version_str(trade_version)
        initializing = (
            generation < 2 or not order_sync_ms or not trade_sync_ms or not order_version or not trade_version
        )
    except FileNotFoundError:
        return CacheHealthStatus(False, "account_cache_health_missing")
    except (OSError, TypeError, ValueError):
        return CacheHealthStatus(False, "account_cache_health_invalid")

    if not pid_is_alive(writer_pid):
        return CacheHealthStatus(False, "account_cache_writer_not_running")
    observed_start_ms = pid_started_at_ms(writer_pid)
    if _PROCESS_IDENTITY_REQUIRED and observed_start_ms is None:
        return CacheHealthStatus(False, "account_cache_writer_identity_unavailable")
    if observed_start_ms is not None and abs(observed_start_ms - writer_started_ms) > 2_000:
        return CacheHealthStatus(False, "account_cache_writer_identity_changed")
    if stopping:
        return CacheHealthStatus(False, "account_cache_writer_stopping")
    if initializing:
        return CacheHealthStatus(False, "account_cache_initializing")

    current_ms = resolve_now_ms(now_ms)
    max_age_val = safe_pos_float(max_age_sec, default=MAX_AGE_SEC)
    future_tolerance_ms = round(safe_nonneg_float(future_tolerance_sec, default=5.0) * 1000)

    if order_sync_ms > current_ms + future_tolerance_ms or trade_sync_ms > current_ms + future_tolerance_ms:
        return CacheHealthStatus(False, "account_cache_timestamp_in_future")

    order_age = max(0.0, (current_ms - order_sync_ms) / 1000.0)
    trade_age = max(0.0, (current_ms - trade_sync_ms) / 1000.0)
    if order_age > max_age_val:
        return CacheHealthStatus(False, "order_cache_stale", order_age, trade_age)
    if trade_age > max_age_val:
        return CacheHealthStatus(False, "trade_cache_stale", order_age, trade_age)
    return CacheHealthStatus(True, "", order_age, trade_age, order_version, trade_version)


def require_fresh_account_cache(**kwargs) -> CacheHealthStatus:
    """Return fresh status or raise a definitive pre-submit refusal."""
    status = inspect_health(**kwargs)
    if not status.ready:
        raise AccountCacheNotReady(status.reason)
    return status
