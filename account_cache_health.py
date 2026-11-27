"""Compact cross-process freshness state for multi-venue account caches.

The dedicated cacheManager process is the sole writer of ``cache_account_health.json``.
Trading processes read the atomic snapshot before live order actions and fail closed
unless the canonical ``CacheOrderFilledManager`` (``BinanceOrderFilled``) cache was
reconciled successfully within the configured age. Legacy ``BinanceTrade`` sync fields
remain recorded as passive observability metadata, while the nested ``venues`` section
tracks per-venue freshness across ``binance``, ``kraken``, ``hyperliquid``, and ``t212``
for ``filled_orders``, ``open_orders``, ``balances``, and ``trades``. File mtime is
intentionally ignored because it does not prove that a venue confirmed account state.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
import sys
import threading
from typing import Callable, Iterable, Mapping, Optional

from botcore import (
    _parse_bool_flag,
    clean_scalar_str,
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
from cachepaths import (
    CACHE_ACCOUNT_HEALTH_FILENAME,
    cache_path,
    normalize_venue_name,
    resolve_cachedb_filepath,
    venue_capabilities,
)
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
SUPPORTED_HEALTH_VENUES: tuple[str, ...] = ("binance", "kraken", "hyperliquid", "t212")
VENUE_HEALTH_STREAMS: tuple[str, ...] = ("filled_orders", "open_orders", "balances", "trades")
DEFAULT_REQUIRED_VENUE_STREAMS: tuple[str, ...] = ("filled_orders", "open_orders")
_DEFAULT_HEALTH_PATH = cache_path(CACHE_ACCOUNT_HEALTH_FILENAME)
HEALTH_PATH = _DEFAULT_HEALTH_PATH
_PROCESS_IDENTITY_REQUIRED = sys.platform.startswith("linux")
_DEAD_PROC_STATES = frozenset({"Z", "X", "x"})
_SYNC_FIELDS = {
    "BinanceOrderFilled": ("order_sync_at_ms", "order_cache_version"),
    "CacheOrderFilledManager": ("order_sync_at_ms", "order_cache_version"),
    "BinanceTrade": ("trade_sync_at_ms", "trade_cache_version"),
    "CacheTradeManager": ("trade_sync_at_ms", "trade_cache_version"),
}
_BINANCE_STREAM_FOR_CACHE = {
    "BinanceOrderFilled": "filled_orders",
    "CacheOrderFilledManager": "filled_orders",
    "BinanceTrade": "trades",
    "CacheTradeManager": "trades",
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


@dataclass(frozen=True)
class VenueCacheHealthStatus:
    """Validated multi-venue account-cache freshness status for one venue."""

    venue: str
    ready: bool
    reason: str
    filled_orders_age_sec: Optional[float] = None
    open_orders_age_sec: Optional[float] = None
    balances_age_sec: Optional[float] = None
    trades_age_sec: Optional[float] = None
    filled_orders_version: str = ""
    open_orders_version: str = ""
    balances_version: str = ""
    trades_version: str = ""


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


def _resolve_health_path(path: object) -> str:
    raw = HEALTH_PATH if path is None else path
    if not isinstance(raw, (str, os.PathLike)) or (target_p := clean_scalar_str(os.fspath(raw))) is None:
        raise ValueError(f"invalid account cache health path: {path!r}")
    return (
        cache_path(CACHE_ACCOUNT_HEALTH_FILENAME)
        if target_p == _DEFAULT_HEALTH_PATH
        else str(resolve_cachedb_filepath(target_p))
    )


def enable_writer(*, path: Optional[str] = None, pid: Optional[int] = None, now_ms: Optional[int] = None) -> None:
    """Enable the sole cacheManager writer and invalidate the old generation."""
    global _writer_enabled, _writer_path, _writer_state
    timestamp = resolve_now_ms(now_ms)
    with _writer_lock:
        _writer_path = _resolve_health_path(path)
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
            "venues": {},
        }
        _writer_enabled = True
        _publish_locked(bump_generation=False)


def is_writer_enabled() -> bool:
    """Return True when this process is the active sole cacheManager account writer."""
    with _writer_lock:
        return bool(_writer_enabled)


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
    if _parse_bool_flag(bump_generation, True):
        _writer_state["generation"] += 1
    atomic_write_json(
        _writer_path, _writer_state, separators=(",", ":"), sort_keys=True, skip_if_unchanged=True
    )
    return True


def _normalize_health_venue(venue: object) -> Optional[str]:
    v_raw = clean_scalar_str(venue)
    if v_raw is None:
        return None
    v_norm = normalize_venue_name(v_raw, default="")
    return v_norm if v_norm in SUPPORTED_HEALTH_VENUES else None


def _normalize_health_stream(stream: object) -> Optional[str]:
    s_raw = clean_scalar_str(stream)
    if s_raw is None:
        return None
    s_norm = s_raw.lower()
    return s_norm if s_norm in VENUE_HEALTH_STREAMS else None


def _ensure_venue_entry_locked(venue: str) -> dict:
    venues_map = _writer_state.get("venues")
    if not isinstance(venues_map, dict):
        venues_map = {}
        _writer_state["venues"] = venues_map
    v_entry = venues_map.get(venue)
    if not isinstance(v_entry, dict):
        v_entry = {}
        venues_map[venue] = v_entry
    return v_entry


def _advance_sync_timestamp_ms(
    prev_ms: int,
    timestamp_ms: int,
    pub_now_ms: int,
    *,
    rollback_tolerance_ms: int = 5_000,
) -> int:
    """Advance sync/publication timestamp monotonically while healing wall-clock rollbacks."""
    if prev_ms > pub_now_ms + rollback_tolerance_ms and abs(timestamp_ms - pub_now_ms) <= rollback_tolerance_ms:
        return timestamp_ms
    return max(prev_ms, timestamp_ms)


def _record_cache_version(
    cache_name: str,
    data_version: str,
    *,
    now_ms: Optional[int] = None,
    update_sync: bool = False,
) -> bool:
    norm_name = normalize_class_name(cache_name)
    version = clean_scalar_str(data_version) or ""
    fields = _SYNC_FIELDS.get(norm_name) if (norm_name and version) else None
    if fields is None:
        return False
    sync_field, version_field = fields
    binance_stream = _BINANCE_STREAM_FOR_CACHE.get(norm_name)
    timestamp = resolve_now_ms(now_ms)
    do_sync = _parse_bool_flag(update_sync, False)
    with _writer_lock:
        if not _writer_enabled:
            return False
        pub_now_ms = resolve_now_ms()
        prev_pub = exact_pos_int(_writer_state.get("published_at_ms"), default=0) or 0
        if do_sync:
            prev_sync = exact_pos_int(_writer_state.get(sync_field), default=0) or 0
            eff_sync_ms = _advance_sync_timestamp_ms(prev_sync, timestamp, pub_now_ms)
            _writer_state[sync_field] = eff_sync_ms
            # Publication time describes this atomic write, while the conservative
            # synchronization timestamp remains monotonic unless the wall clock rolled back.
            _writer_state["published_at_ms"] = _advance_sync_timestamp_ms(prev_pub, pub_now_ms, pub_now_ms)
        else:
            _writer_state["published_at_ms"] = _advance_sync_timestamp_ms(prev_pub, timestamp, pub_now_ms)
        _writer_state[version_field] = version
        if binance_stream:
            v_entry = _ensure_venue_entry_locked("binance")
            if do_sync:
                v_sync_key = f"{binance_stream}_sync_at_ms"
                prev_v_sync = exact_pos_int(v_entry.get(v_sync_key), default=0) or 0
                v_entry[v_sync_key] = _advance_sync_timestamp_ms(prev_v_sync, timestamp, pub_now_ms)
            v_entry[f"{binance_stream}_version"] = version
        return _publish_locked()


def record_successful_sync(cache_name: str, data_version: str, *, now_ms: Optional[int] = None) -> bool:
    """Publish one complete, persisted Order or Trade reconciliation success."""
    return _record_cache_version(cache_name, data_version, now_ms=now_ms, update_sync=True)


def record_persisted_version(cache_name: str, data_version: str, *, now_ms: Optional[int] = None) -> bool:
    """Publish durable content without claiming a new REST reconciliation."""
    return _record_cache_version(cache_name, data_version, now_ms=now_ms, update_sync=False)


def record_venue_sync(
    venue: str,
    stream: str,
    *,
    now_ms: Optional[int] = None,
    data_version: str = "",
) -> bool:
    """Publish one account-cache synchronization timestamp for any supported venue & stream.

    Returns ``False`` without touching disk outside the active ``cacheManager`` writer or when
    ``venue`` / ``stream`` is unsupported.
    """
    v_norm = _normalize_health_venue(venue)
    s_norm = _normalize_health_stream(stream)
    if v_norm is None or s_norm is None:
        return False
    version = clean_scalar_str(data_version) or ""
    timestamp = resolve_now_ms(now_ms)
    sync_key = f"{s_norm}_sync_at_ms"
    ver_key = f"{s_norm}_version"
    with _writer_lock:
        if not _writer_enabled:
            return False
        pub_now_ms = resolve_now_ms()
        v_entry = _ensure_venue_entry_locked(v_norm)
        prev_ms = exact_pos_int(v_entry.get(sync_key), default=0) or 0
        v_entry[sync_key] = _advance_sync_timestamp_ms(prev_ms, timestamp, pub_now_ms)
        if version:
            v_entry[ver_key] = version
        _writer_state["published_at_ms"] = pub_now_ms
        return _publish_locked()


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


def _validate_health_envelope(
    state: object,
) -> tuple[Optional[dict], Optional[tuple[int, int, int, bool]], str]:
    """Validate the shared writer envelope fields of a decoded health snapshot."""
    if not isinstance(state, dict):
        return None, None, "account_cache_health_invalid"
    try:
        if state.get("schema_version") != SCHEMA_VERSION:
            return None, None, "account_cache_schema_mismatch"

        writer_pid = require_int(state.get("writer_pid"), "writer_pid", min_value=1)
        writer_started_ms = require_int(state.get("writer_started_at_ms"), "writer_started_at_ms", min_value=1)
        generation = require_int(state.get("generation"), "generation", min_value=0)
        require_int(state.get("published_at_ms"), "published_at_ms", min_value=1)
        stopping = state.get("stopping")
        if not isinstance(stopping, bool):
            raise ValueError("invalid stopping")
        return state, (writer_pid, writer_started_ms, generation, stopping), ""
    except (TypeError, ValueError):
        return None, None, "account_cache_health_invalid"


def _read_health_envelope(
    path: Optional[str] = None,
) -> tuple[Optional[dict], Optional[tuple[int, int, int, bool]], str]:
    """Read and validate the shared writer envelope from ``cache_account_health.json``."""
    try:
        eff_path = _resolve_health_path(path)
        state = read_json_dict(eff_path)
    except FileNotFoundError:
        return None, None, "account_cache_health_missing"
    except (OSError, TypeError, ValueError):
        return None, None, "account_cache_health_invalid"
    return _validate_health_envelope(state)


def _check_writer_liveness(
    writer_pid: int,
    writer_started_ms: int,
    stopping: bool,
    *,
    pid_is_alive: Optional[Callable[[int], bool]] = None,
    pid_started_at_ms: Optional[Callable[[int], Optional[int]]] = None,
) -> str:
    """Verify the sole writer PID is still running, identical, and not stopping."""
    alive_fn = _pid_is_alive if pid_is_alive is None else pid_is_alive
    start_fn = _pid_started_at_ms if pid_started_at_ms is None else pid_started_at_ms
    if not alive_fn(writer_pid):
        return "account_cache_writer_not_running"
    observed_start_ms = start_fn(writer_pid)
    if _PROCESS_IDENTITY_REQUIRED and observed_start_ms is None:
        return "account_cache_writer_identity_unavailable"
    if observed_start_ms is not None and abs(observed_start_ms - writer_started_ms) > 2_000:
        return "account_cache_writer_identity_changed"
    if stopping:
        return "account_cache_writer_stopping"
    return ""


def inspect_health(
    *,
    path: Optional[str] = None,
    now_ms: Optional[int] = None,
    max_age_sec: float = MAX_AGE_SEC,
    pid_is_alive: Optional[Callable[[int], bool]] = None,
    pid_started_at_ms: Optional[Callable[[int], Optional[int]]] = None,
    future_tolerance_sec: float = 5.0,
) -> CacheHealthStatus:
    """Validate the shared Binance snapshot without trusting mtime or defaults."""
    state, envelope, err = _read_health_envelope(path)
    if err or state is None or envelope is None:
        return CacheHealthStatus(False, err or "account_cache_health_invalid")
    writer_pid, writer_started_ms, generation, stopping = envelope

    try:
        order_sync_ms = require_int(state.get("order_sync_at_ms"), "order_sync_at_ms", min_value=0)
        raw_trade_sync = state.get("trade_sync_at_ms", 0)
        trade_sync_ms = require_int(raw_trade_sync, "trade_sync_at_ms", min_value=0)
        order_version = state.get("order_cache_version")
        raw_trade_version = state.get("trade_cache_version", "")
        if not isinstance(order_version, str) or not isinstance(raw_trade_version, str):
            raise ValueError("invalid cache version")
        order_version = clean_scalar_str(order_version) or ""
        trade_version = clean_scalar_str(raw_trade_version) or ""
        initializing = (
            generation < 1 or not order_sync_ms or not order_version
        )
    except (OSError, TypeError, ValueError):
        return CacheHealthStatus(False, "account_cache_health_invalid")

    liveness_err = _check_writer_liveness(
        writer_pid,
        writer_started_ms,
        stopping,
        pid_is_alive=pid_is_alive,
        pid_started_at_ms=pid_started_at_ms,
    )
    if liveness_err:
        return CacheHealthStatus(False, liveness_err)
    if initializing:
        return CacheHealthStatus(False, "account_cache_initializing")

    current_ms = resolve_now_ms(now_ms)
    max_age_val = safe_pos_float(max_age_sec, default=MAX_AGE_SEC)
    future_tolerance_ms = round(safe_nonneg_float(future_tolerance_sec, default=5.0) * 1000)

    if order_sync_ms > current_ms + future_tolerance_ms:
        return CacheHealthStatus(False, "account_cache_timestamp_in_future")

    order_age = max(0.0, (current_ms - order_sync_ms) / 1000.0)
    trade_age = (
        max(0.0, (current_ms - trade_sync_ms) / 1000.0)
        if trade_sync_ms > 0
        else None
    )
    if order_age > max_age_val:
        return CacheHealthStatus(False, "order_cache_stale", order_age, trade_age)
    return CacheHealthStatus(True, "", order_age, trade_age, order_version, trade_version)


def _evaluate_venue_health(
    v_norm: str,
    state: dict,
    generation: int,
    *,
    now_ms: Optional[int] = None,
    max_age_sec: float = MAX_AGE_SEC,
    required_streams: Iterable[str] = DEFAULT_REQUIRED_VENUE_STREAMS,
    future_tolerance_sec: float = 5.0,
) -> VenueCacheHealthStatus:
    """Evaluate freshness for one normalized venue from a liveness-validated health snapshot."""
    req_streams: list[str] = []
    for raw_s in required_streams or ():
        s_norm = _normalize_health_stream(raw_s)
        if s_norm is None:
            return VenueCacheHealthStatus(v_norm, False, "account_cache_invalid_required_stream")
        if s_norm not in req_streams:
            req_streams.append(s_norm)
    if not req_streams:
        return VenueCacheHealthStatus(v_norm, False, "account_cache_invalid_required_stream")

    try:
        raw_venues = state.get("venues")
        if raw_venues is not None and not isinstance(raw_venues, dict):
            raise ValueError("invalid venues map")
        v_entry = raw_venues.get(v_norm, {}) if isinstance(raw_venues, dict) else {}
        if not isinstance(v_entry, dict):
            raise ValueError("invalid venue entry")

        sync_ms_by_stream: dict[str, int] = {}
        ver_by_stream: dict[str, str] = {}
        for stream in VENUE_HEALTH_STREAMS:
            raw_sync = v_entry.get(f"{stream}_sync_at_ms", 0)
            sync_ms_by_stream[stream] = require_int(raw_sync, f"{v_norm}.{stream}_sync_at_ms", min_value=0)
            raw_ver = v_entry.get(f"{stream}_version", "")
            if not isinstance(raw_ver, str):
                raise ValueError(f"invalid {v_norm}.{stream}_version")
            ver_by_stream[stream] = clean_scalar_str(raw_ver) or ""

        # Backward compatibility with legacy snapshots written before ``venues`` existed.
        if venue_capabilities(v_norm).legacy_unscoped_health_keys:
            if sync_ms_by_stream["filled_orders"] == 0:
                legacy_order_ms = require_int(state.get("order_sync_at_ms", 0), "order_sync_at_ms", min_value=0)
                if legacy_order_ms > 0:
                    sync_ms_by_stream["filled_orders"] = legacy_order_ms
            if not ver_by_stream["filled_orders"] and isinstance(state.get("order_cache_version"), str):
                ver_by_stream["filled_orders"] = clean_scalar_str(state.get("order_cache_version")) or ""
            if sync_ms_by_stream["trades"] == 0:
                legacy_trade_ms = require_int(state.get("trade_sync_at_ms", 0), "trade_sync_at_ms", min_value=0)
                if legacy_trade_ms > 0:
                    sync_ms_by_stream["trades"] = legacy_trade_ms
            if not ver_by_stream["trades"] and isinstance(state.get("trade_cache_version"), str):
                ver_by_stream["trades"] = clean_scalar_str(state.get("trade_cache_version")) or ""
    except (TypeError, ValueError):
        return VenueCacheHealthStatus(v_norm, False, "account_cache_health_invalid")

    current_ms = resolve_now_ms(now_ms)
    max_age_val = safe_pos_float(max_age_sec, default=MAX_AGE_SEC)
    future_tolerance_ms = round(safe_nonneg_float(future_tolerance_sec, default=5.0) * 1000)

    ages: dict[str, Optional[float]] = {
        stream: (
            max(0.0, (current_ms - sync_ms) / 1000.0)
            if sync_ms > 0
            else None
        )
        for stream, sync_ms in sync_ms_by_stream.items()
    }

    def _status(ready: bool, reason: str) -> VenueCacheHealthStatus:
        return VenueCacheHealthStatus(
            venue=v_norm,
            ready=ready,
            reason=reason,
            filled_orders_age_sec=ages["filled_orders"],
            open_orders_age_sec=ages["open_orders"],
            balances_age_sec=ages["balances"],
            trades_age_sec=ages["trades"],
            filled_orders_version=ver_by_stream["filled_orders"],
            open_orders_version=ver_by_stream["open_orders"],
            balances_version=ver_by_stream["balances"],
            trades_version=ver_by_stream["trades"],
        )

    for stream in req_streams:
        sync_ms = sync_ms_by_stream[stream]
        if sync_ms <= 0:
            reason = (
                "account_cache_initializing"
                if generation < 1 and not any(sync_ms_by_stream.values())
                else f"{v_norm}_{stream}_initializing"
            )
            return _status(False, reason)
        if sync_ms > current_ms + future_tolerance_ms:
            return _status(False, "account_cache_timestamp_in_future")
        age_val = ages[stream]
        if age_val is None or age_val > max_age_val:
            return _status(False, f"{v_norm}_{stream}_stale")
    return _status(True, "")


def inspect_venue_health(
    venue: str,
    *,
    path: Optional[str] = None,
    now_ms: Optional[int] = None,
    max_age_sec: float = MAX_AGE_SEC,
    required_streams: Iterable[str] = DEFAULT_REQUIRED_VENUE_STREAMS,
    pid_is_alive: Optional[Callable[[int], bool]] = None,
    pid_started_at_ms: Optional[Callable[[int], Optional[int]]] = None,
    future_tolerance_sec: float = 5.0,
) -> VenueCacheHealthStatus:
    """Validate account-cache freshness for one venue (``binance``, ``kraken``, ``hyperliquid``, ``t212``)."""
    v_norm = _normalize_health_venue(venue)
    if v_norm is None:
        return VenueCacheHealthStatus(str(venue or ""), False, "account_cache_unknown_venue")

    state, envelope, err = _read_health_envelope(path)
    if err or state is None or envelope is None:
        return VenueCacheHealthStatus(v_norm, False, err or "account_cache_health_invalid")
    writer_pid, writer_started_ms, generation, stopping = envelope

    liveness_err = _check_writer_liveness(
        writer_pid,
        writer_started_ms,
        stopping,
        pid_is_alive=pid_is_alive,
        pid_started_at_ms=pid_started_at_ms,
    )
    if liveness_err:
        return VenueCacheHealthStatus(v_norm, False, liveness_err)

    return _evaluate_venue_health(
        v_norm,
        state,
        generation,
        now_ms=now_ms,
        max_age_sec=max_age_sec,
        required_streams=required_streams,
        future_tolerance_sec=future_tolerance_sec,
    )


def _resolve_requested_venues(venues: Iterable[str]) -> list[str]:
    requested: list[str] = []
    for raw_v in venues or ():
        v_key = _normalize_health_venue(raw_v) or str(raw_v or "")
        if v_key and v_key not in requested:
            requested.append(v_key)
    return requested


def _evaluate_requested_venues(
    requested: list[str],
    state: dict,
    generation: int,
    *,
    now_ms: Optional[int],
    max_age_sec: float,
    required_streams: Iterable[str],
    required_streams_by_venue: Optional[Mapping[str, Iterable[str]]],
    future_tolerance_sec: float,
) -> dict[str, VenueCacheHealthStatus]:
    results: dict[str, VenueCacheHealthStatus] = {}
    for v_key in requested:
        v_norm = _normalize_health_venue(v_key)
        if v_norm is None:
            results[v_key] = VenueCacheHealthStatus(v_key, False, "account_cache_unknown_venue")
        else:
            v_req = (
                required_streams_by_venue.get(v_norm, required_streams)
                if isinstance(required_streams_by_venue, Mapping)
                else required_streams
            )
            results[v_norm] = _evaluate_venue_health(
                v_norm,
                state,
                generation,
                now_ms=now_ms,
                max_age_sec=max_age_sec,
                required_streams=v_req,
                future_tolerance_sec=future_tolerance_sec,
            )
    return results


def inspect_all_venues_health(
    *,
    venues: Iterable[str] = SUPPORTED_HEALTH_VENUES,
    path: Optional[str] = None,
    now_ms: Optional[int] = None,
    max_age_sec: float = MAX_AGE_SEC,
    required_streams: Iterable[str] = DEFAULT_REQUIRED_VENUE_STREAMS,
    required_streams_by_venue: Optional[Mapping[str, Iterable[str]]] = None,
    pid_is_alive: Optional[Callable[[int], bool]] = None,
    pid_started_at_ms: Optional[Callable[[int], Optional[int]]] = None,
    future_tolerance_sec: float = 5.0,
) -> dict[str, VenueCacheHealthStatus]:
    """Validate account-cache freshness across all requested venues in a single snapshot read."""
    requested = _resolve_requested_venues(venues)

    state, envelope, err = _read_health_envelope(path)
    if err or state is None or envelope is None:
        reason = err or "account_cache_health_invalid"
        return {v: VenueCacheHealthStatus(v, False, reason) for v in requested}
    writer_pid, writer_started_ms, generation, stopping = envelope

    liveness_err = _check_writer_liveness(
        writer_pid,
        writer_started_ms,
        stopping,
        pid_is_alive=pid_is_alive,
        pid_started_at_ms=pid_started_at_ms,
    )
    if liveness_err:
        return {v: VenueCacheHealthStatus(v, False, liveness_err) for v in requested}

    return _evaluate_requested_venues(
        requested,
        state,
        generation,
        now_ms=now_ms,
        max_age_sec=max_age_sec,
        required_streams=required_streams,
        required_streams_by_venue=required_streams_by_venue,
        future_tolerance_sec=future_tolerance_sec,
    )


def inspect_all_venues_health_snapshot(
    state: object,
    *,
    venues: Iterable[str] = SUPPORTED_HEALTH_VENUES,
    now_ms: Optional[int] = None,
    max_age_sec: float = MAX_AGE_SEC,
    required_streams: Iterable[str] = DEFAULT_REQUIRED_VENUE_STREAMS,
    required_streams_by_venue: Optional[Mapping[str, Iterable[str]]] = None,
    future_tolerance_sec: float = 5.0,
) -> dict[str, VenueCacheHealthStatus]:
    """Evaluate a decoded remote health snapshot across venues without local PID checks."""
    requested = _resolve_requested_venues(venues)
    valid_state, envelope, err = _validate_health_envelope(state)
    if err or valid_state is None or envelope is None:
        reason = err or "account_cache_health_invalid"
        return {v: VenueCacheHealthStatus(v, False, reason) for v in requested}
    _, _, generation, stopping = envelope
    if stopping:
        return {v: VenueCacheHealthStatus(v, False, "account_cache_writer_stopping") for v in requested}
    return _evaluate_requested_venues(
        requested,
        valid_state,
        generation,
        now_ms=now_ms,
        max_age_sec=max_age_sec,
        required_streams=required_streams,
        required_streams_by_venue=required_streams_by_venue,
        future_tolerance_sec=future_tolerance_sec,
    )


def require_fresh_account_cache(**kwargs) -> CacheHealthStatus:
    """Return fresh status or raise a definitive pre-submit refusal."""
    status = inspect_health(**kwargs)
    if not status.ready:
        raise AccountCacheNotReady(status.reason)
    return status


def require_fresh_venue_account_cache(venue: str, **kwargs) -> VenueCacheHealthStatus:
    """Return fresh multi-venue status or raise a definitive pre-submit refusal."""
    status = inspect_venue_health(venue, **kwargs)
    if not status.ready:
        raise AccountCacheNotReady(status.reason)
    return status
