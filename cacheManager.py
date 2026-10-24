import bisect
import gzip
import hashlib
import json
import glob
import math
import os
import time
import datetime
import asyncio
import threading
import importlib
import builtins
import weakref
import shutil
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Optional
from state_io import (
    atomic_gzip_file,
    atomic_snapshot_file,
    atomic_text_writer,
    atomic_write_json as _atomic_write_json,
    commit_file_and_meta,
    discard_generation_file,
    durable_replace_file,
    generation_lock_path,
    meta_file_path,
    previous_generation_path,
    prune_rotated_archives,
    read_recovering_previous,
    read_tail_lines,
    restore_previous_generation,
    stage_previous_generation,
    truncate_file_bytes,
)
from cachepaths import (
    EXACT_LEGACY_FILENAMES as _EXACT_LEGACY_FILENAMES,
    PREFIX_LEGACY_FILENAMES as _PREFIX_LEGACY_FILENAMES,
    legacy_filenames_for as _legacy_filenames_for,
    migrate_legacy_filename_if_needed as _migrate_legacy_filename_if_needed,
)
import cacheintegrity as _cache_integrity
from binance_api import bapi_ws_bridge as _bapi_ws_bridge
from binance_api import bapi_ws_health as _bapi_ws_health
from lock import FileLock
import binance_cache_health as account_cache_health

#my imports
import log
import utils as u
import symbols as sym
from binance_api import bapi as api
# Safe market-data facade import: market_api imports binance_api.bapi without importing
# cacheManager, so it does not close a cycle. The trend chain's current price passes
# through this singleton while trading remains on bapi.
import providers.market_api as _market_api
from providers.base import (
    extract_order_price,
    extract_order_qty,
    extract_order_side,
    extract_timestamp_ms,
    extract_timestamp_s,
    normalize_timestamp_ms,
)

# Load versioned, non-secret tuning parameters using the same pattern as the other
# component env files. ``botcore.load_dotenv`` never overwrites variables already set
# in the real environment; it only fills missing values.
from botcore import (
    load_dotenv as _load_dotenv,
    required_env,
    required_float_env,
    required_int_env,
    single_instance,
)
_CONFIG_ROOT = os.path.dirname(os.path.abspath(__file__))
_load_dotenv(os.path.join(_CONFIG_ROOT, "config.env"))

# Dynamic-window bounds for ``get_short_trend_for_window``. Values below the minimum
# provide too few samples for a meaningful slope; values above the maximum add unjustified
# cost and exceed the 24-hour history retained by Cache24PriceManager.
CM_DYNAMIC_WINDOW_MIN_SEC = required_float_env("CM_DYNAMIC_WINDOW_MIN_SEC")
CM_DYNAMIC_WINDOW_MAX_SEC = required_float_env("CM_DYNAMIC_WINDOW_MAX_SEC")
# Dynamic windows the writer (cacheManager) computes on every full evaluation and publishes
# in the shared trend file. Only the writer holds the raw Cache24 buffers; a reader process
# such as rtrade gets these windows from the file instead of computing them itself.
CM_PUBLISHED_TREND_WINDOWS_SEC = [
    float(part) for part in required_env("CM_PUBLISHED_TREND_WINDOWS_SEC").split(",")
    if part.strip()
]


def _dynamic_window_key(window_seconds):
    """Key a clamped dynamic window in the published snapshot, e.g. 900.0 -> "900"."""
    return f"{float(window_seconds):g}"
CM_RETENTION_DAYS = required_float_env("CM_RETENTION_DAYS")
CM_RETENTION_CHECK_INTERVAL_SEC = required_float_env("CM_RETENTION_CHECK_INTERVAL_SEC")
CM_APPEND_MAX_FILE_BYTES = required_int_env("CM_APPEND_MAX_FILE_BYTES")
CM_ROTATE_KEEP_FRACTION = required_float_env("CM_ROTATE_KEEP_FRACTION")
CM_ROTATE_ARCHIVE_COUNT = required_int_env("CM_ROTATE_ARCHIVE_COUNT")
CM_RESYNC_INTERVAL_SEC = required_float_env("CM_RESYNC_INTERVAL_SEC")
CM_DEDUP_WINDOW = required_int_env("CM_DEDUP_WINDOW")
CM_LONG_ARCHIVE_MONTHS = required_float_env("CM_LONG_ARCHIVE_MONTHS")
CM_LONG_ARCHIVE_SAMPLE_SEC = required_float_env("CM_LONG_ARCHIVE_SAMPLE_SEC")
CM_LONG_ARCHIVE_FLUSH_SEC = required_float_env("CM_LONG_ARCHIVE_FLUSH_SEC")
CM_LONG_ARCHIVE_MEMORY_ROWS = required_int_env("CM_LONG_ARCHIVE_MEMORY_ROWS")

if not 0 < CM_ROTATE_KEEP_FRACTION <= 1:
    raise ValueError("CM_ROTATE_KEEP_FRACTION must be in (0, 1]")
for _name, _value in (
    ("CM_RETENTION_DAYS", CM_RETENTION_DAYS),
    ("CM_RETENTION_CHECK_INTERVAL_SEC", CM_RETENTION_CHECK_INTERVAL_SEC),
    ("CM_APPEND_MAX_FILE_BYTES", CM_APPEND_MAX_FILE_BYTES),
    ("CM_ROTATE_ARCHIVE_COUNT", CM_ROTATE_ARCHIVE_COUNT),
    ("CM_RESYNC_INTERVAL_SEC", CM_RESYNC_INTERVAL_SEC),
    ("CM_DEDUP_WINDOW", CM_DEDUP_WINDOW),
    ("CM_LONG_ARCHIVE_MONTHS", CM_LONG_ARCHIVE_MONTHS),
    ("CM_LONG_ARCHIVE_SAMPLE_SEC", CM_LONG_ARCHIVE_SAMPLE_SEC),
    ("CM_LONG_ARCHIVE_FLUSH_SEC", CM_LONG_ARCHIVE_FLUSH_SEC),
    ("CM_LONG_ARCHIVE_MEMORY_ROWS", CM_LONG_ARCHIVE_MEMORY_ROWS),
):
    if _value <= 0:
        raise ValueError(f"{_name} must be positive")

#from log import PRINT_CONTEXT


# disable logs by redefine with dummy
#def print(*args, **kwargs):
#   pass
#log.print = lambda *args, **kwargs: None


def atomic_write(path):
    """Compatibility wrapper around the repository-wide atomic writer."""
    return atomic_text_writer(path)


def atomic_write_json(path, obj, indent=None):
    """Atomically write JSON through ``atomic_write`` and propagate failures."""
    _atomic_write_json(path, obj, indent=indent)

#log.disable_print()

# WS-only mode and health state live in ``binance_api.bapi_ws_health``.
WS_ONLY_MODE = _bapi_ws_health.WS_ONLY_MODE
WS_LOSS_TIMEOUT_SEC = _bapi_ws_health.WS_LOSS_TIMEOUT_SEC
WS_EVENT_LOG_ENABLED = _bapi_ws_health.WS_EVENT_LOG_ENABLED

_ws_health_lock = _bapi_ws_health._ws_health_lock
_ws_available = _bapi_ws_health._ws_available
_ws_last_event_ts = _bapi_ws_health._ws_last_event_ts
_ws_is_healthy = _bapi_ws_health._ws_is_healthy
_ws_needs_rest_resync = _bapi_ws_health._ws_needs_rest_resync

_CACHE_META_SCHEMA_VERSION = _cache_integrity.CACHE_META_SCHEMA_VERSION
_TRADE_CACHE_CLASSES = {"CacheTrade", "CacheTradeManager"}
_ORDER_CACHE_CLASSES = {"CacheOrder", "CacheOrderManager"}
_ACCOUNT_CACHE_CLASSES = _ORDER_CACHE_CLASSES | _TRADE_CACHE_CLASSES


def _mark_ws_available(value):
    global _ws_available
    _ws_available = _bapi_ws_health.mark_ws_available(value)


def _mark_ws_event_received():
    global _ws_last_event_ts, _ws_is_healthy
    _ws_last_event_ts = _bapi_ws_health.mark_ws_event_received()
    _ws_is_healthy = _bapi_ws_health._ws_is_healthy


def _mark_ws_unhealthy():
    global _ws_is_healthy
    _bapi_ws_health.mark_ws_unhealthy()
    _ws_is_healthy = _bapi_ws_health._ws_is_healthy


def _should_poll_for_manager(cls_name):
    return _bapi_ws_health.should_poll_for_manager(
        cls_name,
        ws_only_mode=WS_ONLY_MODE,
        ws_available=_ws_available,
        ws_is_healthy=_ws_is_healthy,
    )

class CacheManagerInterface(ABC):
    _live_instances = weakref.WeakSet()
    # ── Periodically enforced retention and rotation policy for append caches ──
    RETENTION_DAYS = CM_RETENTION_DAYS
    MAX_FILE_BYTES = CM_APPEND_MAX_FILE_BYTES
    RETENTION_CHECK_INTERVAL_SEC = CM_RETENTION_CHECK_INTERVAL_SEC
    ROTATE_KEEP_FRACTION = CM_ROTATE_KEEP_FRACTION
    ROTATE_ARCHIVE_COUNT = CM_ROTATE_ARCHIVE_COUNT
    RESYNC_INTERVAL_SEC = CM_RESYNC_INTERVAL_SEC
    DEDUP_WINDOW = CM_DEDUP_WINDOW

    def __init__(self, sync_ts, symbols, filename, append_mode = True, api_client=api,
                 append_persist=False, read_only=False):
        self._live_instances.add(self)
        self.cls_name = self.__class__.__name__

        #self.enable_print = True
        #global PRINT_CONTEXT
        #log.PRINT_CONTEXT = self

        self.sync_ts = sync_ts
        self.symbols = symbols
        self.filename = u.cache_path(filename)   # Store cache data in the cachedb subdirectory.
        self.append_mode = append_mode
        self.api_client = api_client
        # JSONL persistence for append-only Trade and AssetValue caches writes only
        # new lines rather than rewriting the entire file.
        self.append_persist = append_persist
        self.read_only = bool(read_only)
        self._persisted_counts = {}   # Number of items already persisted per symbol.
        self._persisted_data_version = ""
        self._persisted_stream_id = ""
        self._persisted_revision = 0
        self._persisted_committed_bytes = 0
        self._persisted_content_digest = ""
        self._loaded_data_version = ""
        self._loaded_stream_id = ""
        self._loaded_revision = 0
        self._loaded_committed_bytes = 0
        self._loaded_content_digest = ""
        self._loaded_counts = {}
        self._reader_reload_lock = threading.Lock()
        self._legacy_jsonl_needs_rewrite = False
        self._account_cache_dirty = False
        self._last_complete_sync_at_ms = 0

        self.days_back = 30

        self.cache = {}
        self.fetchtime_per_symbol = {}
        self._rest_sync_cursor_per_symbol = {}

        self.thread = None
        self._stop_event = threading.Event()
        self.save_state = False
        # When true, sleep one interval before the first synchronization iteration.
        # Preserve values subclasses set before ``super().__init__`` because the base
        # initializer starts the thread.
        if not hasattr(self, "_first_sleep"):
            self._first_sleep = False
        self.lock = threading.RLock()

        # Shared subscriber pattern forwards prices to other managers and PriceWindow.
        # Initialize before periodic_sync because its thread may notify immediately.
        if not hasattr(self, "_price_subscribers"):
            self._price_subscribers = []

        self.fallback_time_default = int(time.time() * 1000) - self.days_back*24*60*60*1000

        # function calls here after all inint vars
        self.load_state()

    # ── Subscriber pattern for forwarding prices ─────────────────────────────

    def subscribe_price(self, subscriber) -> None:
        """Subscribe an object implementing ``on_price_update(symbol, ts_ms, price)``."""
        with self.lock:
            if subscriber not in self._price_subscribers:
                self._price_subscribers.append(subscriber)

    def unsubscribe_price(self, subscriber) -> None:
        with self.lock:
            if subscriber in self._price_subscribers:
                self._price_subscribers.remove(subscriber)

    def _notify_price_subscribers(self, symbol: str, ts_ms: int, price: float) -> None:
        with self.lock:
            subs = list(self._price_subscribers)
        for sub in subs:
            try:
                sub.on_price_update(symbol, ts_ms, price)
            except Exception as e:
                print(f"[{self.cls_name}] Error notifying subscriber {sub}: {e}")
    

    def get_all_symbols_from_cache(self):
        with self.lock:
            return list(self.cache.keys())       
        
    def rebuild_fetchtimes(self):
        """Allow subclasses to provide custom fetch-time reconstruction.

        Returning None lets ``_rebuild_fetchtimes`` infer timestamps generically
        from ``time``/``timestamp`` dictionaries or ``[timestamp, ...]`` lists.
        """
        return None
    
    
    def _rebuild_fetchtimes(self):
        last_times_per_sym = self.rebuild_fetchtimes()
        if not last_times_per_sym:
            last_times_per_sym = defaultdict(int)
            for symbol, trades in self.cache.items():
                for trade in trades:
                    time_ = self._entry_timestamp_ms(trade)
                    if time_ > last_times_per_sym[symbol]:
                        last_times_per_sym[symbol] = time_
            # Apply a 60-second safety offset.
            for symbol in last_times_per_sym:
                last_times_per_sym[symbol] = max(0, last_times_per_sym[symbol] - 60_000)                
        if not last_times_per_sym:
            # A missing cache starts at the configured bounded history horizon.
            fallback_time_file = self.fallback_time_default
            if os.path.exists(self.filename):
                fallback_time_file = int(os.path.getmtime(self.filename) * 1000) - 60_000
            fallback_time = min(self.fallback_time_default, fallback_time_file)
            return {symbol: fallback_time for symbol in self.symbols}
        return last_times_per_sym
          
        
    def load_state(self):
        _migrate_legacy_filename_if_needed(self.filename)
        print(f"[{self.cls_name}][Info] Load state from {self.filename} ...")
        if self.append_persist:
            self._migrate_legacy_json_if_needed()
            self._load_jsonl()
            if not self.cache and not self.read_only:
                self.query_remote_and_update_cache()
                self.save_state_to_file_if_enabled()
            return
        if self.cls_name in _ORDER_CACHE_CLASSES:
            self._load_order_state()
            return
        backup_path = self._previous_generation_path()
        if os.path.exists(self.filename) or os.path.exists(backup_path):
            try:
                def _read_json(candidate_path):
                    with open(candidate_path, "r") as handle:
                        return json.load(handle)

                data = read_recovering_previous(
                    self.filename, _read_json, backup_path=backup_path
                )
                with self.lock:
                    self.cache = data.get("items", {})
                    if not isinstance(self.cache, dict):
                        # Convert the legacy list format to a dictionary.
                        self.cache = {sym: item for sym, item in zip(self.symbols, self.cache)}
                        print(f"[{self.cls_name}][warning] self.cache is not Dict!!!!")    
                    
                    self.fetchtime_per_symbol = data.get("fetchtime", {})
                    rest_cursor = data.get("rest_sync_cursor")
                    self._rest_sync_cursor_per_symbol = dict(
                        rest_cursor
                        if isinstance(rest_cursor, dict)
                        else self.fetchtime_per_symbol
                    )
                    if not self.cache:
                        print(f"[{self.cls_name}][warning] cache is None")
                    if not self.fetchtime_per_symbol:
                        print(f"[{self.cls_name}][warning] fetchtime_per_symbol is None")
            except Exception as e:
                print(f"[{self.cls_name}][Error] While reading the cache file {self.filename} : {e}")
                if not self.read_only:
                    self.query_remote_and_update_cache()
                    self.save_state_to_file_if_enabled()
        else :
            print(f"[{self.cls_name}][Info] File is missing, may be is it first time run. Creating it ....")
            if not self.read_only:
                self.query_remote_and_update_cache()
                self.save_state_to_file_if_enabled()

    _normalize_timestamp_map = staticmethod(
        _cache_integrity.normalize_timestamp_map
    )

    def _preserve_rest_sync_cursor_before_update_locked(self, symbol):
        """Anchor the REST cursor before a non-REST update advances fetchtime."""
        if symbol in self._rest_sync_cursor_per_symbol:
            return
        if symbol in self.fetchtime_per_symbol:
            prior_cursor = int(self.fetchtime_per_symbol[symbol] or 0)
        else:
            rebuilt = self._rebuild_fetchtimes()
            prior_cursor = int(
                rebuilt.get(symbol, self.fallback_time_default) or 0
            )
        self._rest_sync_cursor_per_symbol[symbol] = max(0, prior_cursor)

    def _read_legacy_order_snapshot_strict(self, legacy_meta, *, path=None):
        with open(path or self.filename, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("Order cache root is not an object")
        if data.get("data_version") is not None:
            raise ValueError("versioned Order cache is missing its versioned manifest")
        items = data.get("items")
        fetchtime = data.get("fetchtime", {})
        if not isinstance(items, dict) or not isinstance(fetchtime, dict):
            raise ValueError("invalid Order cache maps")
        raw_cursor = data.get("rest_sync_cursor")
        if raw_cursor is None and legacy_meta is not None:
            raw_cursor = legacy_meta.get("rest_sync_cursor")
        if raw_cursor is None:
            rest_sync_cursor = dict(fetchtime)
        else:
            rest_sync_cursor = self._normalize_timestamp_map(
                raw_cursor, "Order cache rest_sync_cursor"
            )
        actual_counts = {
            s: len(v) for s, v in items.items() if isinstance(v, list)
        }
        if legacy_meta is not None and "counts" in legacy_meta:
            expected_counts = legacy_meta["counts"]
            if not isinstance(expected_counts, dict) or any(
                not isinstance(s, str) or isinstance(c, bool) or not isinstance(c, int) or c < 0
                for s, c in expected_counts.items()
            ):
                raise ValueError("invalid legacy Order manifest counts")
            if actual_counts != expected_counts:
                raise ValueError("Order cache counts differ from legacy manifest")
        return items, dict(fetchtime), dict(rest_sync_cursor), actual_counts

    def _load_order_state(self):
        """Load Order cache state strictly when durable data or metadata exists."""
        with self._order_generation_lock():
            loaded = self._load_order_state_generation_locked()
        if not loaded and not self.read_only:
            print(f"[{self.cls_name}][Info] File is missing, may be is it first time run. Creating it ....")
            self.query_remote_and_update_cache()
            self.save_state_to_file_if_enabled()

    def _load_order_state_generation_locked(self):
        meta_path = self.filename + ".meta"
        backup_path = self._order_previous_generation_path()
        has_file = os.path.exists(self.filename)
        has_meta = os.path.exists(meta_path)
        has_backup = os.path.exists(backup_path)
        if not has_file and not has_meta and not has_backup:
            return False

        reason = self._account_reader_reason()
        legacy_meta = None
        if has_meta:
            try:
                with open(meta_path, "r", encoding="utf-8") as handle:
                    raw_meta = json.load(handle)
            except (OSError, ValueError, TypeError) as exc:
                raise account_cache_health.AccountCacheNotReady(reason) from exc
            if not isinstance(raw_meta, dict):
                raise account_cache_health.AccountCacheNotReady(reason)
            if raw_meta.get("schema_version") is not None:
                metadata = self._load_versioned_account_meta(optional=False)
                self._ensure_order_version_generation_locked(
                    str(metadata["data_version"])
                )
                return True
            legacy_meta = raw_meta

        if not has_file and not has_backup:
            raise account_cache_health.AccountCacheNotReady(reason)
        try:
            items, fetchtime, rest_sync_cursor, counts = (
                read_recovering_previous(
                    self.filename,
                    lambda candidate_path: self._read_legacy_order_snapshot_strict(
                        legacy_meta, path=candidate_path
                    ),
                    backup_path=backup_path,
                )
            )
            with self.lock:
                self.cache = items
                self.fetchtime_per_symbol = dict(fetchtime)
                self._rest_sync_cursor_per_symbol = dict(rest_sync_cursor)
                self._persisted_counts = dict(counts)
            return True
        except account_cache_health.AccountCacheNotReady:
            raise
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise account_cache_health.AccountCacheNotReady(reason) from exc


    # ── Memory/disk resilience: freshness, overwrite guard, and resync ────────
    def _mem_max_ts(self):
        """Return memory freshness as the latest fetch time in milliseconds."""
        if not self.fetchtime_per_symbol:
            return 0
        try:
            return max(self.fetchtime_per_symbol.values())
        except Exception:
            return 0

    def _persisted_max_ts(self):
        """Read file freshness cheaply from the ``.meta`` sidecar."""
        try:
            with open(self.filename + ".meta") as mf:
                return json.load(mf).get("max_ts", 0)
        except Exception:
            return 0

    def _is_account_cache(self):
        return self.cls_name in _ACCOUNT_CACHE_CLASSES

    def _is_canonical_account_cache(self):
        return (
            self._is_account_cache()
            and set(self.symbols) == set(sym.symbols)
        )

    _snapshot_data_version = staticmethod(
        _cache_integrity.snapshot_data_version
    )
    _trade_data_version = staticmethod(
        _cache_integrity.trade_data_version
    )
    _trade_records_digest = staticmethod(
        _cache_integrity.trade_records_digest
    )

    def _account_meta_snapshot(self, *, data_version, fetchtime, counts,
                               rest_sync_cursor=None,
                               stream_id="", revision=0,
                               committed_bytes=None,
                               content_digest=None):
        return _cache_integrity.build_account_meta_snapshot(
            data_version=data_version,
            fetchtime=fetchtime,
            counts=counts,
            rest_sync_cursor=rest_sync_cursor,
            stream_id=stream_id,
            revision=revision,
            committed_bytes=committed_bytes,
            content_digest=content_digest,
            schema_version=_CACHE_META_SCHEMA_VERSION,
        )

    def _remember_account_commit_locked(self, metadata):
        """Remember and publish one durable account-cache version."""
        version = str(metadata["data_version"])
        self._persisted_data_version = version
        self._persisted_stream_id = str(metadata.get("stream_id") or "")
        self._persisted_revision = int(metadata.get("revision") or 0)
        self._persisted_committed_bytes = int(
            metadata.get("committed_bytes") or 0
        )
        self._persisted_content_digest = str(
            metadata.get("content_digest") or ""
        )
        self._loaded_data_version = version
        self._loaded_stream_id = self._persisted_stream_id
        self._loaded_revision = self._persisted_revision
        self._loaded_committed_bytes = self._persisted_committed_bytes
        self._loaded_content_digest = self._persisted_content_digest
        self._loaded_counts = dict(metadata.get("counts") or {})
        self._account_cache_dirty = False
        if self._is_canonical_account_cache():
            try:
                account_cache_health.record_persisted_version(
                    self.cls_name, version
                )
            except Exception as exc:
                builtins.print(
                    f"[{self.cls_name}][Error] could not publish durable "
                    f"cache version {version}: {exc}"
                )

    def _mark_account_cache_dirty_locked(self):
        if self._is_account_cache():
            self._account_cache_dirty = True

    _trade_semantic_signature = staticmethod(
        _cache_integrity.trade_semantic_signature
    )
    _dedupe_trade_items_by_id = staticmethod(
        _cache_integrity.dedupe_trade_items_by_id
    )

    def _validate_account_rows_locked(self):
        """Reject any financial row that strict readers could not accept."""
        for symbol, items in self.cache.items():
            if not isinstance(symbol, str) or not isinstance(items, list):
                raise ValueError(f"invalid {self.cls_name} cache bucket")
            order_ids = set()
            fill_owners = {}
            for item in items:
                if not self._is_valid_item(item):
                    raise ValueError(f"invalid {self.cls_name} cache row")
                if (
                    self.cls_name in _TRADE_CACHE_CLASSES
                    and str(item["symbol"]).upper() != symbol.upper()
                ):
                    raise ValueError("Trade row symbol differs from its bucket")
                if self.cls_name in _ORDER_CACHE_CLASSES:
                    if (
                        item.get("symbol") is not None
                        and str(item["symbol"]).upper() != symbol.upper()
                    ):
                        raise ValueError("Order row symbol differs from its bucket")
                    order_id = str(int(item["orderId"]))
                    if order_id in order_ids:
                        raise ValueError(
                            f"duplicate Order aggregate ID {order_id}")
                    order_ids.add(order_id)
                    for fill_id in item.get("_fillIds", []):
                        fill_key = str(int(fill_id))
                        previous_order = fill_owners.get(fill_key)
                        if previous_order is not None:
                            raise ValueError(
                                f"immutable fill ID {fill_key} belongs to both "
                                f"orders {previous_order} and {order_id}")
                        fill_owners[fill_key] = order_id
            if (
                self.cls_name in _TRADE_CACHE_CLASSES
                and len(self._dedupe_trade_items_by_id(items)) != len(items)
            ):
                raise ValueError("duplicate immutable Trade ID")

    def _account_reader_reason(self):
        prefix = "trade" if self.cls_name in _TRADE_CACHE_CLASSES else "order"
        return f"{prefix}_cache_reader_not_current"

    def _write_meta(self, snapshot, fetchtime=None):
        """Atomically write an explicit snapshot matching durable cache data."""
        try:
            if fetchtime is not None:
                payload = {
                    "counts": dict(snapshot or {}),
                    "fetchtime": dict(fetchtime or {}),
                }
            else:
                payload = dict(snapshot)
            fetchtime_map = payload.get("fetchtime", {})
            payload["max_ts"] = max(fetchtime_map.values()) if fetchtime_map else 0
            payload["saved_at"] = int(time.time() * 1000)
            atomic_write_json(meta_file_path(self.filename), payload)
            return True
        except Exception as e:
            print(f"[{self.cls_name}][Error] metadata {self.filename}: {e}")
            return False

    def _commit_file_and_meta_locked(
        self,
        write_data_fn,
        metadata_or_fn,
        *,
        backup_path=None,
        had_prior_file=None,
    ):
        """Delegate transactional file + ``.meta`` commit to ``state_io.commit_file_and_meta``."""
        return commit_file_and_meta(
            self.filename,
            write_data_fn,
            lambda: self._write_meta(
                metadata_or_fn() if callable(metadata_or_fn) else metadata_or_fn
            ),
            backup_path=backup_path,
            had_prior_file=had_prior_file,
            on_rollback_error=lambda rollback_error: builtins.print(
                f"[{self.cls_name}][Error] could not restore the prior "
                f"generation for {self.filename}: {rollback_error}"
            ),
        )

    def save_state_to_file_if_enabled(self):
        """Persist enabled writers and return whether the commit succeeded."""
        if self.save_state:
            return self.save_state_to_file()
        return False

    def save_state_to_file(self):
        """Write to disk regardless of ``save_state`` for writers and failover.

        Refuse to overwrite newer data another process has already persisted.
        """
        try:
            if self.append_persist and self.cls_name in _TRADE_CACHE_CLASSES:
                return self._save_account_trade_append()
            if self.cls_name in _ORDER_CACHE_CLASSES:
                return self._save_account_order_snapshot()
            with self.lock:
                if self._persisted_max_ts() > self._mem_max_ts():
                    builtins.print(f"[{self.cls_name}][resync] file newer than memory -> "
                                   f"refusing to overwrite with stale data ({self.filename})")
                    return False
                if self.append_persist:
                    return self._save_jsonl_append()
                fetchtime = dict(self.fetchtime_per_symbol)
                rest_sync_cursor = {
                    str(symbol): max(0, int(cursor or 0))
                    for symbol, cursor in self._rest_sync_cursor_per_symbol.items()
                }
                counts = {symbol: len(items) for symbol, items in self.cache.items()}
                payload = {
                    "items": self.cache,
                    "fetchtime": fetchtime,
                    "rest_sync_cursor": rest_sync_cursor,
                }
                metadata = {
                    "fetchtime": fetchtime,
                    "rest_sync_cursor": rest_sync_cursor,
                    "counts": counts,
                }
                if not self._commit_file_and_meta_locked(
                    lambda: atomic_write_json(
                        self.filename, payload, indent=1
                    ),
                    metadata,
                ):
                    return False
                print(f"[{self.cls_name}][info] Save cache to file {self.filename}")
                return True
        except Exception as e:
            print(f"[{self.cls_name}][Error] While saving the cache file {self.filename} / .tmp : {e}")
            return False

    def _save_account_order_snapshot(self):
        """Commit Order snapshot and its manifest under the generation lock."""
        try:
            with self._order_generation_lock():
                return self._save_account_order_snapshot_generation_locked()
        except Exception as exc:
            print(
                f"[{self.cls_name}][Error] lock Order snapshot "
                f"{self.filename}: {exc}"
            )
            return False

    def _save_account_order_snapshot_generation_locked(self):
        """Write Order data and manifest while retaining the prior certified generation."""
        try:
            with self.lock:
                previous_metadata = self._load_versioned_account_meta(
                    optional=True
                )
                if (
                    previous_metadata is not None
                    and self._persisted_data_version
                    and previous_metadata["data_version"]
                    != self._persisted_data_version
                ):
                    builtins.print(
                        f"[{self.cls_name}][resync] durable Order generation "
                        "differs from this manager; refusing stale overwrite"
                    )
                    return False
                if self._persisted_max_ts() > self._mem_max_ts():
                    builtins.print(
                        f"[{self.cls_name}][resync] file newer than memory -> "
                        f"refusing to overwrite with stale data ({self.filename})"
                    )
                    return False

                self._validate_account_rows_locked()

                backup_path = None
                had_prior_file = False
                if previous_metadata is not None:
                    self._read_order_snapshot_recovering_previous_locked(
                        previous_metadata
                    )
                    backup_path = self._stage_previous_order_generation()
                    had_prior_file = True
                elif os.path.exists(self.filename):
                    backup_path = self._stage_previous_order_generation()
                    had_prior_file = True
                else:
                    self._discard_trade_generation(
                        self._order_previous_generation_path()
                    )

                fetchtime = dict(self.fetchtime_per_symbol)
                rest_sync_cursor = {
                    str(symbol): max(0, int(cursor or 0))
                    for symbol, cursor in self._rest_sync_cursor_per_symbol.items()
                }
                counts = {
                    symbol: len(items)
                    for symbol, items in self.cache.items()
                }
                data_version = self._snapshot_data_version(
                    self.cache, fetchtime
                )
                payload = {
                    "items": self.cache,
                    "fetchtime": fetchtime,
                    "rest_sync_cursor": rest_sync_cursor,
                    "data_version": data_version,
                }
                metadata = self._account_meta_snapshot(
                    data_version=data_version,
                    fetchtime=fetchtime,
                    rest_sync_cursor=rest_sync_cursor,
                    counts=counts,
                )
                if not self._commit_file_and_meta_locked(
                    lambda: atomic_write_json(
                        self.filename, payload, indent=1
                    ),
                    metadata,
                    backup_path=backup_path,
                    had_prior_file=had_prior_file,
                ):
                    self._mark_account_cache_dirty_locked()
                    return False

                self._persisted_counts = dict(counts)
                self._remember_account_commit_locked(metadata)
                print(f"[{self.cls_name}][info] Save cache to file {self.filename}")
                return True
        except Exception as exc:
            with self.lock:
                self._mark_account_cache_dirty_locked()
            print(
                f"[{self.cls_name}][Error] While saving the cache file "
                f"{self.filename} / .tmp : {exc}"
            )
            return False

    def _reload_from_disk(self):
        """Reload the cache when another process has written a newer file."""
        if self.append_persist:
            self._load_jsonl()
            return
        if self.cls_name in _ORDER_CACHE_CLASSES:
            try:
                metadata = self._load_versioned_account_meta(optional=True)
                if metadata is not None:
                    self._ensure_order_version(str(metadata["data_version"]))
                    return
            except Exception as e:
                print(f"[{self.cls_name}][Error] reload {self.filename}: {e}")
                return
        backup_path = self._previous_generation_path()
        if os.path.exists(self.filename) or os.path.exists(backup_path):
            try:
                def _read_json(candidate_path):
                    with open(candidate_path) as handle:
                        return json.load(handle)

                data = read_recovering_previous(
                    self.filename, _read_json, backup_path=backup_path
                )
                with self.lock:
                    items = data.get("items", {})
                    if isinstance(items, dict):
                        if isinstance(self, Cache24PriceManager):
                            items = {s: v for s, v in items.items() if s in self.symbols}
                        self.cache = items
                    self.fetchtime_per_symbol = data.get("fetchtime", {})
                    rest_cursor = data.get("rest_sync_cursor")
                    self._rest_sync_cursor_per_symbol = dict(
                        rest_cursor
                        if isinstance(rest_cursor, dict)
                        else self.fetchtime_per_symbol
                    )
                    self._account_cache_dirty = False
            except Exception as e:
                print(f"[{self.cls_name}][Error] reload {self.filename}: {e}")

    def resync_mem_file(self):
        """Periodically reload newer disk state or persist newer memory state."""
        file_ts = self._persisted_max_ts()
        mem_ts = self._mem_max_ts()
        if file_ts > mem_ts:
            builtins.print(f"[{self.cls_name}][resync] file is newer -> reloading ({self.filename})")
            self._reload_from_disk()
        elif mem_ts > file_ts and self.save_state:
            self.save_state_to_file_if_enabled()

    def sync_account_reader_if_needed(self):
        """Refresh a read-only account cache when durable disk state advances."""
        if not self._is_account_cache() or self.save_state:
            return
        for _attempt in range(2):
            try:
                metadata = self._load_versioned_account_meta(optional=True)
                if metadata is not None:
                    version = str(metadata.get("data_version") or "")
                    with self.lock:
                        loaded_version = self._loaded_data_version
                        dirty = self._account_cache_dirty
                        mem_ts = self._mem_max_ts()
                    file_ts = self._persisted_max_ts()
                    if version and version != loaded_version:
                        if not dirty or file_ts >= mem_ts:
                            self.ensure_persisted_version(version)
                    return
                self.resync_mem_file()
                return
            except account_cache_health.AccountCacheNotReady as exc:
                if _attempt == 0:
                    continue
                print(f"[{self.cls_name}][Warning] reader sync skipped: {exc}")
            except Exception as exc:
                print(f"[{self.cls_name}][Warning] reader sync skipped: {exc}")
                return

    # ── JSONL append persistence for append-only caches ──────────────────────
    def _migrate_legacy_json_if_needed(self):
        """Create a JSONL cache from its legacy full-JSON sibling once.

        Trade migration decides under the generation lock so concurrent startup
        cannot replace a generation another manager has already published.
        """
        if self.cls_name in _TRADE_CACHE_CLASSES:
            with self._trade_generation_lock():
                return self._migrate_legacy_json_generation_locked()
        return self._migrate_legacy_json_generation_locked()

    def _migrate_legacy_json_generation_locked(self):
        """Migrate after any required Trade generation lock has been acquired.

        Migration is deliberately non-destructive: the old JSON file remains available
        for rollback, while all subsequent writes target the new JSONL path.
        """
        if os.path.exists(self.filename) or not self.filename.endswith(".jsonl"):
            return
        legacy = self.filename[:-1]
        if not os.path.exists(legacy):
            return
        try:
            with open(legacy, encoding="utf-8") as handle:
                payload = json.load(handle)
            items = payload.get("items", {})
            if not isinstance(items, dict):
                raise ValueError("legacy cache items must be a dictionary")
            with self.lock:
                self.cache = items
                self.fetchtime_per_symbol = payload.get("fetchtime", {})
                rest_cursor = payload.get("rest_sync_cursor")
                self._rest_sync_cursor_per_symbol = dict(
                    rest_cursor
                    if isinstance(rest_cursor, dict)
                    else self.fetchtime_per_symbol
                )
                self._persisted_counts = {}
            migrated = (
                self._compact_account_trade_jsonl_generation_locked()
                if self.cls_name in _TRADE_CACHE_CLASSES
                else self.compact_jsonl()
            )
            if not migrated:
                raise ValueError(
                    "legacy cache could not be converted into a durable JSONL generation"
                )
            if self.cls_name in _TRADE_CACHE_CLASSES:
                metadata = self._load_versioned_account_meta(optional=False)
                if metadata is None:
                    raise ValueError("versioned Trade metadata was not created")
                self._read_trade_snapshot_strict(metadata)
            builtins.print(
                f"[{self.cls_name}][migration] created {self.filename} from {legacy}; "
                "the legacy file was retained for rollback"
            )
        except (OSError, ValueError, TypeError) as exc:
            raise RuntimeError(f"Cannot migrate legacy cache {legacy}: {exc}") from exc

    def _save_jsonl_append(self):
        """Append only items added since the last flush without rewriting the file."""
        if self.cls_name in _TRADE_CACHE_CLASSES:
            return self._save_account_trade_append()

        previous_counts = {}
        candidate_counts = {}
        initial_size = None
        try:
            with self.lock:
                if not isinstance(self, Cache365PriceManager) and any(
                    self._persisted_counts.get(symbol, 0) > len(items)
                    for symbol, items in self.cache.items()
                ):
                    return self.compact_jsonl()
                previous_counts = dict(self._persisted_counts)
                candidate_counts = dict(previous_counts)
                with open(self.filename, "a+", encoding="utf-8") as f:
                    initial_size = os.fstat(f.fileno()).st_size
                    for symbol, items in self.cache.items():
                        start = self._persisted_counts.get(symbol, 0)
                        if start > len(items):   # The cache was cleared or shortened; resynchronize.
                            start = 0
                        for item in items[start:]:
                            f.write(json.dumps(
                                {"s": symbol, "i": item}, separators=(",", ":"),
                            ) + "\n")
                        candidate_counts[symbol] = len(items)
                    f.flush()
                    os.fsync(f.fileno())
                self._persisted_counts = candidate_counts
                fetchtime = dict(self.fetchtime_per_symbol)
                return self._write_meta({
                    "fetchtime": fetchtime,
                    "rest_sync_cursor": dict(self._rest_sync_cursor_per_symbol),
                    "counts": dict(candidate_counts),
                })
        except Exception as e:
            self._persisted_counts = previous_counts
            if initial_size is not None:
                self._truncate_failed_jsonl_append(initial_size)
            print(f"[{self.cls_name}][Error] append JSONL {self.filename}: {e}")
            return False

    def _save_account_trade_append(self):
        """Commit Trade data and its manifest under the generation lock."""
        try:
            with self._trade_generation_lock():
                return (
                    self._save_account_trade_append_if_current_generation_locked()
                )
        except Exception as exc:
            print(
                f"[{self.cls_name}][Error] lock Trade append "
                f"{self.filename}: {exc}"
            )
            return False

    def _save_account_trade_append_if_current_generation_locked(self):
        """Reject stale memory, then commit while the generation lock is held."""
        with self.lock:
            metadata = self._load_versioned_account_meta(optional=True)
            if metadata is None:
                identity_matches = (
                    not self._persisted_data_version
                    and (
                        self._legacy_jsonl_needs_rewrite
                        or not os.path.exists(self.filename + ".meta")
                    )
                )
            else:
                durable_identity = (
                    metadata["data_version"],
                    metadata["stream_id"],
                    metadata["revision"],
                    metadata["committed_bytes"],
                    metadata["content_digest"],
                )
                manager_identity = (
                    self._persisted_data_version,
                    self._persisted_stream_id,
                    self._persisted_revision,
                    self._persisted_committed_bytes,
                    self._persisted_content_digest,
                )
                identity_matches = durable_identity == manager_identity
            if not identity_matches:
                builtins.print(
                    f"[{self.cls_name}][resync] durable Trade generation "
                    "differs from this manager; refusing stale append"
                )
                return False
            if self._persisted_max_ts() > self._mem_max_ts():
                builtins.print(
                    f"[{self.cls_name}][resync] file newer than memory -> "
                    f"refusing to overwrite with stale data ({self.filename})")
                return False
            return self._save_account_trade_append_generation_locked()

    def _save_account_trade_append_generation_locked(self):
        """Commit new Trade rows and a versioned byte-boundary manifest."""
        with self.lock:
            self._validate_account_rows_locked()
            if self._legacy_jsonl_needs_rewrite:
                return self._compact_account_trade_jsonl_generation_locked()
            if any(
                    self._persisted_counts.get(symbol, 0) > len(items)
                    for symbol, items in self.cache.items()):
                return self._compact_account_trade_jsonl_generation_locked()

            previous_counts = dict(self._persisted_counts)
            previous_version = self._persisted_data_version
            previous_stream = self._persisted_stream_id
            previous_revision = self._persisted_revision
            previous_boundary = self._persisted_committed_bytes
            previous_digest = self._persisted_content_digest
            initial_size = None
            try:
                exists = os.path.exists(self.filename)
                mode = "r+b" if exists else "w+b"
                with open(self.filename, mode) as handle:
                    actual_size = os.fstat(handle.fileno()).st_size
                    if previous_version:
                        if actual_size < previous_boundary:
                            raise OSError(
                                "Trade cache is shorter than its committed boundary"
                            )
                        initial_size = previous_boundary
                        if actual_size != previous_boundary:
                            handle.truncate(previous_boundary)
                    else:
                        initial_size = actual_size
                    handle.seek(initial_size)
                    candidate_counts = dict(previous_counts)
                    wrote_rows = False
                    appended_records = []
                    for symbol, items in self.cache.items():
                        start = previous_counts.get(symbol, 0)
                        for item in items[start:]:
                            if (
                                not self._is_valid_item(item)
                                or str(item["symbol"]).upper()
                                != str(symbol).upper()
                            ):
                                raise ValueError(
                                    "invalid Trade row in append delta"
                                )
                            row = json.dumps(
                                {"s": symbol, "i": item},
                                separators=(",", ":"),
                            ).encode("utf-8") + b"\n"
                            handle.write(row)
                            wrote_rows = True
                            appended_records.append((symbol, item))
                        candidate_counts[symbol] = len(items)
                    handle.flush()
                    os.fsync(handle.fileno())
                    committed_bytes = handle.tell()

                content_digest = self._trade_records_digest(
                    appended_records, previous_digest
                )
                stream_id = previous_stream or uuid.uuid4().hex
                revision = previous_revision
                if wrote_rows or not previous_version:
                    revision += 1
                revision = max(1, revision)
                version = self._trade_data_version(stream_id, revision)
                fetchtime = dict(self.fetchtime_per_symbol)
                rest_sync_cursor = dict(self._rest_sync_cursor_per_symbol)
                metadata = self._account_meta_snapshot(
                    data_version=version,
                    stream_id=stream_id,
                    revision=revision,
                    committed_bytes=committed_bytes,
                    content_digest=content_digest,
                    fetchtime=fetchtime,
                    rest_sync_cursor=rest_sync_cursor,
                    counts=candidate_counts,
                )
                if not self._write_meta(metadata):
                    self._truncate_failed_jsonl_append(initial_size)
                    return False
                self._persisted_counts = candidate_counts
                self._legacy_jsonl_needs_rewrite = False
                self._remember_account_commit_locked(metadata)
                return True
            except Exception as exc:
                self._persisted_counts = previous_counts
                self._persisted_data_version = previous_version
                self._persisted_stream_id = previous_stream
                self._persisted_revision = previous_revision
                self._persisted_committed_bytes = previous_boundary
                self._persisted_content_digest = previous_digest
                if initial_size is not None:
                    self._truncate_failed_jsonl_append(initial_size)
                print(
                    f"[{self.cls_name}][Error] append JSONL "
                    f"{self.filename}: {exc}"
                )
                return False

    def _truncate_failed_jsonl_append(self, initial_size):
        """Best-effort rollback of bytes written by a failed append attempt."""
        try:
            truncate_file_bytes(self.filename, initial_size)
        except Exception as rollback_error:
            builtins.print(
                f"[{self.cls_name}][Error] could not roll back failed JSONL append "
                f"for {self.filename}: {rollback_error}"
            )

    def _load_versioned_account_meta(self, *, optional=False,
                                     required_version=None):
        """Load and validate the small durable account-cache manifest."""
        return _cache_integrity.load_versioned_account_meta_file(
            meta_file_path(self.filename),
            is_trade_cache=self.cls_name in _TRADE_CACHE_CLASSES,
            reader_reason=self._account_reader_reason(),
            optional=optional,
            required_version=required_version,
            schema_version=_CACHE_META_SCHEMA_VERSION,
        )

    def _read_trade_records_strict(self, start, end, *, path=None):
        """Parse only the requested committed JSONL byte range."""
        return _cache_integrity.read_trade_records_range_strict(
            path or self.filename,
            start,
            end,
            is_valid_item_fn=self._is_valid_item,
            reader_reason=self._account_reader_reason(),
        )

    _counts_for_records = staticmethod(_cache_integrity.counts_for_records)
    _nonzero_counts = staticmethod(_cache_integrity.nonzero_counts)

    def _read_trade_snapshot_strict(self, metadata, *, path=None):
        records = self._read_trade_records_strict(
            0, metadata["committed_bytes"], path=path
        )
        if self._trade_records_digest(records) != metadata["content_digest"]:
            raise account_cache_health.AccountCacheNotReady(
                self._account_reader_reason()
            )
        cache = {}
        seen_ids = defaultdict(set)
        for symbol, item in records:
            normalized_symbol = str(symbol).upper()
            trade_id = str(int(item["id"]))
            if trade_id in seen_ids[normalized_symbol]:
                raise account_cache_health.AccountCacheNotReady(
                    self._account_reader_reason()
                )
            seen_ids[normalized_symbol].add(trade_id)
            cache.setdefault(symbol, []).append(item)
        actual = {symbol: len(items) for symbol, items in cache.items()}
        if actual != self._nonzero_counts(metadata["counts"]):
            raise account_cache_health.AccountCacheNotReady(
                self._account_reader_reason()
            )
        return cache, dict(metadata["counts"])

    def _remember_loaded_trade_manifest_locked(self, metadata, counts):
        self._persisted_data_version = metadata["data_version"]
        self._persisted_stream_id = metadata["stream_id"]
        self._persisted_revision = metadata["revision"]
        self._persisted_committed_bytes = metadata["committed_bytes"]
        self._persisted_content_digest = metadata["content_digest"]
        self._loaded_data_version = metadata["data_version"]
        self._loaded_stream_id = metadata["stream_id"]
        self._loaded_revision = metadata["revision"]
        self._loaded_committed_bytes = metadata["committed_bytes"]
        self._loaded_content_digest = metadata["content_digest"]
        self._loaded_counts = dict(counts)
        self._account_cache_dirty = False

    def ensure_persisted_version(self, required_version):
        """Bring this process to the exact durable account-cache version."""
        required_version = str(required_version or "")
        if not self._is_account_cache() or not required_version:
            raise account_cache_health.AccountCacheNotReady(
                self._account_reader_reason()
            )
        with self.lock:
            if (
                self._loaded_data_version == required_version
                and not self._account_cache_dirty
            ):
                return
        with self._reader_reload_lock:
            with self.lock:
                if (
                    self._loaded_data_version == required_version
                    and not self._account_cache_dirty
                ):
                    return
            if self.cls_name in _TRADE_CACHE_CLASSES:
                self._ensure_trade_version(required_version)
            else:
                self._ensure_order_version(required_version)

    def _read_order_snapshot_strict(self, metadata, *, path=None):
        """Read and validate an Order snapshot against its durable manifest."""
        reason = self._account_reader_reason()
        required_version = str(metadata["data_version"])
        try:
            with open(path or self.filename, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                raise ValueError("Order cache root is not an object")
            items = payload.get("items")
            fetchtime = payload.get("fetchtime")
            if not isinstance(items, dict) or not isinstance(fetchtime, dict):
                raise ValueError("invalid Order cache maps")
            if payload.get("data_version") != required_version:
                raise ValueError("Order cache and marker versions differ")
            normalized_fetchtime = self._normalize_timestamp_map(
                fetchtime,
                error_message="invalid Order cache fetchtime",
            )
            raw_rest_cursor = payload.get("rest_sync_cursor")
            if raw_rest_cursor is not None:
                self._normalize_timestamp_map(
                    raw_rest_cursor,
                    error_message="invalid Order cache rest_sync_cursor",
                )
            for symbol, orders in items.items():
                if (not isinstance(symbol, str) or not isinstance(orders, list)
                        or any(
                            not self._is_valid_item(order)
                            for order in orders
                        )):
                    raise ValueError("invalid Order cache items")
            actual_counts = {
                symbol: len(orders) for symbol, orders in items.items()
            }
            if actual_counts != metadata["counts"]:
                raise ValueError("Order cache counts differ from its manifest")
            if normalized_fetchtime != metadata["fetchtime"]:
                raise ValueError(
                    "Order cache fetchtimes differ from its manifest"
                )
            if (
                self._snapshot_data_version(items, normalized_fetchtime)
                != required_version
            ):
                raise ValueError("Order cache digest is invalid")
            return items, normalized_fetchtime, actual_counts
        except account_cache_health.AccountCacheNotReady:
            raise
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise account_cache_health.AccountCacheNotReady(reason) from exc

    def _read_order_snapshot_recovering_previous_locked(self, metadata):
        """Read an Order snapshot against ``metadata``, restoring ``.previous`` if needed."""
        return read_recovering_previous(
            self.filename,
            lambda candidate_path: self._read_order_snapshot_strict(
                metadata, path=candidate_path
            ),
            backup_path=self._order_previous_generation_path(),
        )

    def _read_trade_snapshot_recovering_previous_locked(self, metadata):
        """Read a Trade snapshot against ``metadata``, restoring ``.previous`` if needed."""
        return read_recovering_previous(
            self.filename,
            lambda candidate_path: self._read_trade_snapshot_strict(
                metadata, path=candidate_path
            ),
            backup_path=self._trade_previous_generation_path(),
        )

    def _ensure_order_version(self, required_version):
        with self._order_generation_lock():
            self._ensure_order_version_generation_locked(required_version)

    def _ensure_order_version_generation_locked(self, required_version):
        reason = self._account_reader_reason()
        try:
            metadata = self._load_versioned_account_meta(
                required_version=required_version
            )
            items, fetchtime, actual_counts = (
                self._read_order_snapshot_recovering_previous_locked(metadata)
            )
            after = self._load_versioned_account_meta(
                required_version=required_version
            )
            if (
                after["data_version"] != metadata["data_version"]
                or after["counts"] != metadata["counts"]
                or after["fetchtime"] != metadata["fetchtime"]
                or after["rest_sync_cursor"] != metadata["rest_sync_cursor"]
            ):
                raise ValueError("Order manifest changed during reload")
            with self.lock:
                self.cache = items
                self.fetchtime_per_symbol = dict(fetchtime)
                self._rest_sync_cursor_per_symbol = dict(
                    metadata["rest_sync_cursor"]
                )
                self._persisted_counts = dict(actual_counts)
                self._persisted_data_version = required_version
                self._loaded_data_version = required_version
                self._loaded_counts = dict(actual_counts)
                self._account_cache_dirty = False
        except account_cache_health.AccountCacheNotReady:
            raise
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise account_cache_health.AccountCacheNotReady(reason) from exc

    def _ensure_trade_version(self, required_version):
        with self._trade_generation_lock():
            self._ensure_trade_version_generation_locked(required_version)

    def _ensure_trade_version_generation_locked(self, required_version):
        reason = self._account_reader_reason()
        try:
            metadata = self._load_versioned_account_meta(
                required_version=required_version
            )
            with self.lock:
                can_append = (
                    bool(self._loaded_data_version)
                    and not self._account_cache_dirty
                    and self._loaded_stream_id == metadata["stream_id"]
                    and self._loaded_committed_bytes <= metadata["committed_bytes"]
                )
                old_boundary = self._loaded_committed_bytes
                old_version = self._loaded_data_version
                old_counts = dict(self._loaded_counts)
                old_trade_ids = (
                    {
                        str(symbol).upper(): {
                            str(int(item["id"])) for item in items
                        }
                        for symbol, items in self.cache.items()
                    } if can_append else {}
                )
                old_digest = self._loaded_content_digest

            if can_append:
                try:
                    records = self._read_trade_records_strict(
                        old_boundary, metadata["committed_bytes"]
                    )
                    suffix_ids = defaultdict(set)
                    for symbol, item in records:
                        normalized_symbol = str(symbol).upper()
                        trade_id = str(int(item["id"]))
                        if (
                            trade_id in old_trade_ids.get(normalized_symbol, set())
                            or trade_id in suffix_ids[normalized_symbol]
                        ):
                            raise ValueError("duplicate immutable Trade ID in suffix")
                        suffix_ids[normalized_symbol].add(trade_id)
                    delta_counts = self._counts_for_records(records)
                    candidate_counts = dict(old_counts)
                    for symbol, count in delta_counts.items():
                        candidate_counts[symbol] = (
                            candidate_counts.get(symbol, 0) + count
                        )
                    if candidate_counts != metadata["counts"]:
                        raise ValueError("Trade suffix counts differ from manifest")
                    if (
                        self._trade_records_digest(records, old_digest)
                        != metadata["content_digest"]
                    ):
                        raise ValueError("Trade suffix digest differs from manifest")
                    candidate_cache = None
                    self._discard_trade_generation(
                        self._trade_previous_generation_path()
                    )
                except (
                    account_cache_health.AccountCacheNotReady,
                    OSError,
                    TypeError,
                    ValueError,
                    json.JSONDecodeError,
                ):
                    if not os.path.exists(
                        self._trade_previous_generation_path()
                    ):
                        raise
                    can_append = False
                    candidate_cache, candidate_counts = (
                        self._read_trade_snapshot_recovering_previous_locked(
                            metadata
                        )
                    )
                    records = []
            else:
                candidate_cache, candidate_counts = (
                    self._read_trade_snapshot_recovering_previous_locked(
                        metadata
                    )
                )
                records = []

            after = self._load_versioned_account_meta(
                required_version=required_version
            )
            if after != metadata:
                raise ValueError("Trade manifest changed during reload")

            if can_append:
                with self.lock:
                    append_still_valid = (
                        not self._account_cache_dirty
                        and self._loaded_data_version == old_version
                        and self._loaded_stream_id == metadata["stream_id"]
                        and self._loaded_committed_bytes == old_boundary
                        and self._loaded_content_digest == old_digest
                        and self._loaded_counts == old_counts
                        and {
                            symbol: len(items)
                            for symbol, items in self.cache.items()
                        } == self._nonzero_counts(old_counts)
                    )
                    if append_still_valid:
                        for symbol, item in records:
                            self.cache.setdefault(symbol, []).append(item)
                        self.fetchtime_per_symbol = dict(metadata["fetchtime"])
                        self._rest_sync_cursor_per_symbol = dict(
                            metadata["rest_sync_cursor"]
                        )
                        self._persisted_counts = dict(candidate_counts)
                        self._remember_loaded_trade_manifest_locked(
                            metadata, candidate_counts
                        )
                        return
                candidate_cache, candidate_counts = (
                    self._read_trade_snapshot_recovering_previous_locked(
                        metadata
                    )
                )
                after = self._load_versioned_account_meta(
                    required_version=required_version
                )
                if after != metadata:
                    raise ValueError("Trade manifest changed during full reload")

            with self.lock:
                self.cache = candidate_cache
                self.fetchtime_per_symbol = dict(metadata["fetchtime"])
                self._rest_sync_cursor_per_symbol = dict(
                    metadata["rest_sync_cursor"]
                )
                self._persisted_counts = dict(candidate_counts)
                self._remember_loaded_trade_manifest_locked(
                    metadata, candidate_counts
                )
        except account_cache_health.AccountCacheNotReady:
            raise
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise account_cache_health.AccountCacheNotReady(reason) from exc

    def _previous_generation_path(self):
        return previous_generation_path(self.filename)

    def _generation_lock(self):
        """Serialize cache data/manifest publication and canonical recovery."""
        os.makedirs(os.path.dirname(os.path.abspath(self.filename)), exist_ok=True)
        return FileLock(generation_lock_path(self.filename))

    def _stage_previous_generation(self):
        """Keep the currently certified cache data until the new manifest commits."""
        return stage_previous_generation(self.filename)

    def _restore_previous_generation(self, backup_path=None):
        """Atomically restore data matching the still-current manifest."""
        restore_previous_generation(self.filename, backup_path=backup_path)

    _discard_generation_file = staticmethod(discard_generation_file)

    _order_previous_generation_path = _previous_generation_path
    _order_generation_lock = _generation_lock
    _stage_previous_order_generation = _stage_previous_generation
    _restore_previous_order_generation = _restore_previous_generation

    _trade_previous_generation_path = _previous_generation_path
    _trade_generation_lock = _generation_lock
    _stage_previous_trade_generation = _stage_previous_generation
    _restore_previous_trade_generation = _restore_previous_generation
    _discard_trade_generation = _discard_generation_file

    def _load_jsonl(self):
        """Load JSONL under a lock when canonical recovery is possible."""
        if self.cls_name in _TRADE_CACHE_CLASSES:
            with self._trade_generation_lock():
                return self._load_jsonl_generation_locked()
        return self._load_jsonl_generation_locked()

    def _load_jsonl_generation_locked(self):
        """Load JSONL after any required Trade generation lock is held."""
        if self.cls_name in _TRADE_CACHE_CLASSES:
            metadata = self._load_versioned_account_meta(optional=True)
            if metadata is not None:
                cache, counts = (
                    self._read_trade_snapshot_recovering_previous_locked(
                        metadata
                    )
                )
                with self.lock:
                    self.cache = cache
                    self.fetchtime_per_symbol = dict(metadata["fetchtime"])
                    self._rest_sync_cursor_per_symbol = dict(
                        metadata["rest_sync_cursor"]
                    )
                    self._persisted_counts = counts
                    self._remember_loaded_trade_manifest_locked(metadata, counts)
                return

        legacy_metadata = {}
        metaf = self.filename + ".meta"
        if os.path.exists(metaf):
            try:
                with open(metaf, encoding="utf-8") as handle:
                    loaded_metadata = json.load(handle)
                if isinstance(loaded_metadata, dict):
                    legacy_metadata = loaded_metadata
            except (OSError, ValueError, TypeError):
                legacy_metadata = {}

        with self.lock:
            self.cache = {}
            had_invalid_line = False
            seen_trade_items = defaultdict(dict)
            backup_path = self._previous_generation_path()
            if (
                self.cls_name not in _TRADE_CACHE_CLASSES
                and not os.path.exists(self.filename)
                and os.path.exists(backup_path)
            ):
                try:
                    self._restore_previous_generation(backup_path)
                except OSError:
                    pass
            if os.path.exists(self.filename):
                with open(self.filename, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            if self.cls_name in _TRADE_CACHE_CLASSES:
                                if (
                                    not isinstance(rec, dict)
                                    or not isinstance(rec.get("s"), str)
                                    or not isinstance(rec.get("i"), dict)
                                    or not self._is_valid_item(rec["i"])
                                    or str(rec["i"]["symbol"]).upper()
                                    != str(rec["s"]).upper()
                                ):
                                    raise ValueError("invalid legacy Trade row")
                                trade_id = str(int(rec["i"]["id"]))
                                signature = self._trade_semantic_signature(rec["i"])
                                symbol_items = seen_trade_items[
                                    str(rec["s"]).upper()
                                ]
                                if trade_id in symbol_items:
                                    if symbol_items[trade_id] != signature:
                                        raise ValueError(
                                            "conflicting legacy Trade rows for "
                                            f"immutable ID {trade_id}"
                                        )
                                    had_invalid_line = True
                                    continue
                                symbol_items[trade_id] = signature
                            self.cache.setdefault(rec["s"], []).append(rec["i"])
                        except Exception as exc:
                            if self.cls_name in _TRADE_CACHE_CLASSES:
                                expected_counts = legacy_metadata.get("counts")
                                actual_counts = {
                                    symbol: len(items)
                                    for symbol, items in self.cache.items()
                                }
                                recoverable_tail = (
                                    isinstance(exc, json.JSONDecodeError)
                                    and not any(value.strip() for value in f)
                                    and isinstance(expected_counts, dict)
                                    and all(
                                        isinstance(symbol, str)
                                        and not isinstance(count, bool)
                                        and isinstance(count, int)
                                        and count >= 0
                                        for symbol, count
                                        in expected_counts.items()
                                    )
                                    and self._nonzero_counts(expected_counts)
                                    == actual_counts
                                )
                                if recoverable_tail:
                                    had_invalid_line = True
                                    break
                                raise account_cache_health.AccountCacheNotReady(
                                    self._account_reader_reason()
                                ) from exc
                            had_invalid_line = True
                            continue
            self._legacy_jsonl_needs_rewrite = (
                had_invalid_line
                or self.cls_name in _TRADE_CACHE_CLASSES
            )
            if (
                self.cls_name not in _TRADE_CACHE_CLASSES
                and not had_invalid_line
                and os.path.exists(backup_path)
            ):
                self._discard_generation_file(backup_path)
            self._persisted_counts = {s: len(v) for s, v in self.cache.items()}
            fetchtime = legacy_metadata.get("fetchtime")
            if isinstance(fetchtime, dict):
                self.fetchtime_per_symbol = fetchtime
                rest_cursor = legacy_metadata.get("rest_sync_cursor")
                self._rest_sync_cursor_per_symbol = dict(
                    rest_cursor
                    if isinstance(rest_cursor, dict)
                    else fetchtime
                )

    def compact_jsonl(self):
        """Rewrite JSONL from memory after full in-memory and on-disk deduplication.

        Perform this expensive complete pass periodically rather than on every update.
        """
        if not self.append_persist:
            return False
        if self.cls_name in _TRADE_CACHE_CLASSES:
            return self._compact_account_trade_jsonl()

        original_cache = None
        original_counts = None
        try:
            with self.lock:
                original_cache = {
                    symbol: list(items) for symbol, items in self.cache.items()
                }
                original_counts = dict(self._persisted_counts)
                for symbol, items in list(self.cache.items()):
                    seen = set()
                    deduped = []
                    for item in items:
                        k = json.dumps(item, sort_keys=True)
                        if k not in seen:
                            seen.add(k)
                            deduped.append(item)
                    self.cache[symbol] = deduped   # Keep memory deduplicated too.
                candidate_counts = {s: len(v) for s, v in self.cache.items()}
                fetchtime = dict(self.fetchtime_per_symbol)
                metadata = {
                    "fetchtime": fetchtime,
                    "rest_sync_cursor": dict(self._rest_sync_cursor_per_symbol),
                    "counts": dict(candidate_counts),
                }

                def _write_compacted():
                    with atomic_write(self.filename) as f:
                        for symbol, items in self.cache.items():
                            for item in items:
                                f.write(json.dumps(
                                    {"s": symbol, "i": item},
                                    separators=(",", ":"),
                                ) + "\n")

                if not self._commit_file_and_meta_locked(
                    _write_compacted, metadata
                ):
                    self.cache = original_cache
                    self._persisted_counts = original_counts
                    return False
                self._persisted_counts = candidate_counts
                return True
        except Exception as e:
            if original_cache is not None:
                self.cache = original_cache
            if original_counts is not None:
                self._persisted_counts = original_counts
            print(f"[{self.cls_name}][Error] compact JSONL {self.filename}: {e}")
            return False

    def _compact_account_trade_jsonl(self):
        """Rewrite Trade data and its manifest as one cross-process generation."""
        try:
            with self._trade_generation_lock():
                return self._compact_account_trade_jsonl_generation_locked()
        except Exception as exc:
            self._legacy_jsonl_needs_rewrite = True
            print(
                f"[{self.cls_name}][Error] lock Trade generation "
                f"{self.filename}: {exc}"
            )
            return False

    def _compact_account_trade_jsonl_generation_locked(self):
        """Rewrite Trade data while retaining the prior certified generation."""
        try:
            with self.lock:
                backup_path = None
                had_prior_file = False
                previous_metadata = self._load_versioned_account_meta(
                    optional=True
                )
                if previous_metadata is not None:
                    self._read_trade_snapshot_recovering_previous_locked(
                        previous_metadata
                    )
                    backup_path = self._stage_previous_trade_generation()
                    had_prior_file = True
                elif os.path.exists(self.filename):
                    backup_path = self._stage_previous_trade_generation()
                    had_prior_file = True
                else:
                    self._discard_trade_generation(
                        self._trade_previous_generation_path()
                    )

                candidate_cache = {}
                for symbol, items in self.cache.items():
                    if not isinstance(symbol, str) or not isinstance(items, list):
                        raise ValueError("invalid Trade cache bucket")
                    for item in items:
                        if (
                            not self._is_valid_item(item)
                            or str(item["symbol"]).upper() != symbol.upper()
                        ):
                            raise ValueError("invalid Trade cache row")
                    candidate_cache[symbol] = (
                        self._dedupe_trade_items_by_id(items)
                    )
                self.cache = candidate_cache
                self._validate_account_rows_locked()

                counts = {
                    symbol: len(items)
                    for symbol, items in self.cache.items()
                }
                fetchtime = dict(self.fetchtime_per_symbol)
                rest_sync_cursor = dict(self._rest_sync_cursor_per_symbol)
                records = []
                committed_metadata = {}

                def _write_trade_data():
                    with atomic_write(self.filename) as handle:
                        for symbol, items in self.cache.items():
                            for item in items:
                                records.append((symbol, item))
                                handle.write(json.dumps(
                                    {"s": symbol, "i": item},
                                    separators=(",", ":"),
                                ) + "\n")

                def _build_trade_meta():
                    committed_bytes = os.path.getsize(self.filename)
                    content_digest = self._trade_records_digest(records)
                    stream_id = uuid.uuid4().hex
                    revision = 1
                    version = self._trade_data_version(stream_id, revision)
                    committed_metadata["value"] = self._account_meta_snapshot(
                        data_version=version, stream_id=stream_id,
                        revision=revision, committed_bytes=committed_bytes,
                        content_digest=content_digest,
                        fetchtime=fetchtime,
                        rest_sync_cursor=rest_sync_cursor,
                        counts=counts,
                    )
                    return committed_metadata["value"]

                if not self._commit_file_and_meta_locked(
                    _write_trade_data,
                    _build_trade_meta,
                    backup_path=backup_path,
                    had_prior_file=had_prior_file,
                ):
                    self._legacy_jsonl_needs_rewrite = True
                    return False

                self._persisted_counts = counts
                self._legacy_jsonl_needs_rewrite = False
                self._remember_account_commit_locked(
                    committed_metadata["value"]
                )
                return True
        except Exception as exc:
            self._legacy_jsonl_needs_rewrite = True
            print(
                f"[{self.cls_name}][Error] compact JSONL "
                f"{self.filename}: {exc}"
            )
            return False

    @staticmethod
    def _entry_timestamp_ms(item):
        """Extract an entry timestamp in milliseconds from a dictionary or list."""
        if isinstance(item, dict):
            return extract_timestamp_ms(item, default=0) or 0
        if isinstance(item, (list, tuple)) and item:
            return normalize_timestamp_ms(item[0], default=0) or 0
        return 0

    def maintain_append_persist(self):
        """Perform weekly maintenance for every append-mode cache.

        First prune entries older than ``RETENTION_DAYS`` using the generic timestamp
        extractor. Then rotate oversized JSONL files through ``_rotate_keep_latest``.
        Full-JSON trade/order/asset caches grow slowly enough that time retention, not
        the one-GB size threshold, is their effective bound. Applying this to every
        append-mode cache prevents unbounded growth outside JSONL classes.
        """
        if not self.append_mode:
            return
        # 1) prune time-based
        cutoff_ms = int((time.time() - self.RETENTION_DAYS * 24 * 3600) * 1000)
        changed = False
        with self.lock:
            original_cache = {
                symbol: list(items) for symbol, items in self.cache.items()
            }
            original_dirty = self._account_cache_dirty
            for symbol, items in list(self.cache.items()):
                kept = [it for it in items if self._entry_timestamp_ms(it) >= cutoff_ms]
                if len(kept) != len(items):
                    self.cache[symbol] = kept
                    changed = True
            if changed:
                self._mark_account_cache_dirty_locked()
        if changed:
            builtins.print(f"[{self.cls_name}][maintain] pruning >{self.RETENTION_DAYS}d from {self.filename}")
            if self.save_state:
                persisted = (
                    self.compact_jsonl()
                    if self.append_persist
                    else self.save_state_to_file()   # A full rewrite is normal for these classes.
                )
                if not persisted:
                    with self.lock:
                        self.cache = original_cache
                        self._account_cache_dirty = original_dirty
                    return False
        # Rotate by size only for JSONL; see the docstring.
        if not self.append_persist or not self.save_state:
            return
        try:
            if os.path.exists(self.filename) and os.path.getsize(self.filename) > self.MAX_FILE_BYTES:
                self._rotate_keep_latest()
        except OSError:
            pass

    def _rotate_keep_latest(self):
        """Archive the current file and retain each symbol's latest configured fraction."""
        if self.cls_name in _TRADE_CACHE_CLASSES:
            return self._rotate_account_trade_keep_latest()
        with self.lock:
            archive = f"{self.filename}.{int(time.time())}.archive"
            try:
                atomic_snapshot_file(self.filename, archive)   # Snapshot full history before modifying the active file.
            except OSError as e:
                builtins.print(f"[{self.cls_name}][maintain] archiving failed: {e}")
                return False
            original_cache = {
                symbol: list(items) for symbol, items in self.cache.items()
            }
            original_persisted_counts = dict(self._persisted_counts)
            for symbol, items in self.cache.items():
                keep_n = max(1, int(len(items) * self.ROTATE_KEEP_FRACTION))
                self.cache[symbol] = items[-keep_n:]
            self._persisted_counts = {}
            try:
                compacted = bool(self.compact_jsonl())   # Rewrite the current file with retained entries only.
            except Exception as e:
                builtins.print(f"[{self.cls_name}][maintain] rotation compaction failed: {e}")
                compacted = False
            if not compacted:
                self.cache = original_cache
                self._persisted_counts = original_persisted_counts
                try:
                    self._restore_previous_generation(archive)
                except OSError as restore_err:
                    builtins.print(f"[{self.cls_name}][maintain] failed to restore active file from {archive}: {restore_err}")
                builtins.print(f"[{self.cls_name}][maintain] rotation aborted; restored active file and in-memory state")
                return False
            try:
                archive = atomic_gzip_file(archive, archive + ".gz", remove_source=True)
            except OSError as e:
                builtins.print(f"[{self.cls_name}][maintain] archive compression failed: {e}")
            prune_rotated_archives(self.filename, self.ROTATE_ARCHIVE_COUNT)
        builtins.print(f"[{self.cls_name}][maintain] ROTATION: archived -> {archive}, "
                       f"kept the last {int(self.ROTATE_KEEP_FRACTION*100)}%")
        return True

    def _rotate_account_trade_keep_latest(self):
        """Publish a compact Trade generation without moving canonical data first."""
        archive = None
        original_cache = None
        original_dirty = False
        original_needs_rewrite = False
        published = False
        try:
            with self._trade_generation_lock():
                with self.lock:
                    metadata = self._load_versioned_account_meta()
                    self._read_trade_snapshot_strict(metadata)
                    archive = (
                        f"{self.filename}.{int(time.time())}.archive"
                    )
                    atomic_snapshot_file(self.filename, archive)
                    original_cache = {
                        symbol: list(items)
                        for symbol, items in self.cache.items()
                    }
                    original_dirty = self._account_cache_dirty
                    original_needs_rewrite = self._legacy_jsonl_needs_rewrite
                    for symbol, items in self.cache.items():
                        keep_n = max(
                            1,
                            int(len(items) * self.ROTATE_KEEP_FRACTION),
                        )
                        self.cache[symbol] = items[-keep_n:]
                    self._mark_account_cache_dirty_locked()
                    if not self._compact_account_trade_jsonl_generation_locked():
                        self.cache = original_cache
                        self._account_cache_dirty = original_dirty
                        self._legacy_jsonl_needs_rewrite = (
                            original_needs_rewrite
                        )
                        self._discard_trade_generation(archive)
                        builtins.print(
                            f"[{self.cls_name}][maintain] Trade rotation "
                            "aborted because generation publication failed"
                        )
                        return False
                    published = True

            try:
                archive = atomic_gzip_file(archive, archive + ".gz", remove_source=True)
            except OSError as exc:
                builtins.print(
                    f"[{self.cls_name}][maintain] archive compression "
                    f"failed: {exc}"
                )

            prune_rotated_archives(self.filename, self.ROTATE_ARCHIVE_COUNT)
            builtins.print(
                f"[{self.cls_name}][maintain] ROTATION: archived -> "
                f"{archive}, kept the last "
                f"{int(self.ROTATE_KEEP_FRACTION * 100)}%"
            )
            return True
        except Exception as exc:
            if not published and original_cache is not None:
                with self.lock:
                    self.cache = original_cache
                    self._account_cache_dirty = original_dirty
                    self._legacy_jsonl_needs_rewrite = original_needs_rewrite
            if not published and archive is not None:
                self._discard_trade_generation(archive)
            builtins.print(
                f"[{self.cls_name}][maintain] Trade rotation failed: {exc}"
            )
            return False

    @abstractmethod
    def get_remote_items(self, symbol, startTime):
        """Fetch remote items; subclasses must implement this method."""
        pass 
        
     
    def filter_new_items(self, cache_items, new_items):
        seen = {json.dumps(it, sort_keys=True) for it in cache_items}
        unique_new = []
        for item in new_items:
            key = json.dumps(item, sort_keys=True)
            if key not in seen:
                seen.add(key)
                unique_new.append(item)
        return unique_new
        
    
    def update_cache_per_symbol(self, symbol, new_items):
        if new_items is None:
            return
        current_time = int(time.time() * 1000)
        if self.append_mode:
            if isinstance(new_items, dict):
                new_items = [new_items]
            elif not isinstance(new_items, list):
                new_items = list(new_items) if isinstance(new_items, tuple) else [new_items]

        count_new_items = len(new_items) if isinstance(new_items, (list, tuple, dict)) else 1
        print(f"[{self.cls_name}][Info] {symbol}:  new_items {new_items}")

        with self.lock:  # Protected write.
            self.cache.setdefault(symbol, [])
            if self.append_mode:    # History mode retains every PriceOrders/Price/Trade element.
                # Polling returns overlapping recent data, so deduplicate against only the
                # recent window. This is O(DEDUP_WINDOW) rather than O(entire cache); the
                # periodic compact_jsonl pass performs complete deduplication.
                cache_copy = list(self.cache.get(symbol, []))[-self.DEDUP_WINDOW:]
                new_items = self.filter_new_items(cache_copy, new_items)
                print(f"[{self.cls_name}][Info] {symbol}:  out of {count_new_items} keeping only {len(new_items)}")
                new_items = [item for item in new_items if item is not None]
                if not new_items:
                    return
                self.cache[symbol].extend(new_items)
            else:  # Snapshot mode stores only the latest trend values.
                self.cache[symbol] = new_items if isinstance(new_items, list) else [new_items]             #self.cache[symbol] = new_items[0]

            self._preserve_rest_sync_cursor_before_update_locked(symbol)
            self.fetchtime_per_symbol[symbol] = max(
                int(self.fetchtime_per_symbol.get(symbol, 0) or 0),
                current_time,
            )
            self._mark_account_cache_dirty_locked()

        print(f"[{self.cls_name}][Info] {symbol}: Added {len(new_items)} new items.")

    def _persist_items(self, symbol, new_items):
        """Persist one symbol to the cache by default.

        Subclasses may override this, for example to record a timestamp and notify
        subscribers through ``CacheCurrentPriceManager._push_price``.
        """
        self.update_cache_per_symbol(symbol, new_items)

    def query_remote_and_update_cache(self):
        with self.lock:
            if not self.fetchtime_per_symbol:
                self.fetchtime_per_symbol = self._rebuild_fetchtimes()
            for sym, rebuilt_ft in self.fetchtime_per_symbol.items():
                self._rest_sync_cursor_per_symbol.setdefault(sym, rebuilt_ft)

        account_cycle_high_water = None
        for symbol in list(self.symbols):
            request_high_water = int(time.time() * 1000)
            fallback_ft = self.fetchtime_per_symbol.get(symbol, self.fallback_time_default)
            if self._is_account_cache():
                startTime = min(
                    self._rest_sync_cursor_per_symbol.get(symbol, fallback_ft),
                    fallback_ft,
                )
            else:
                startTime = fallback_ft
            if self._is_account_cache() and account_cycle_high_water is None:
                account_cycle_high_water = request_high_water
            new_items = self.get_remote_items(symbol=symbol, startTime=startTime)
            if new_items is None:
                raise RuntimeError(
                    f"{self.cls_name} returned no synchronization result for {symbol}")
            if not new_items:
                print(f"[{self.cls_name}][Info] {symbol}:  No remote items starting with {u.timestampToTime(startTime)} ")
            else:
                self._persist_items(symbol, new_items)
            if self._is_account_cache():
                # The remote request is not end-time bounded. Resume from its start,
                # so events arriving during the response are deliberately overlapped
                # and eliminated by immutable fill IDs on the next synchronization.
                with self.lock:
                    cursor_ms = max(0, request_high_water - 1)
                    self._rest_sync_cursor_per_symbol[symbol] = cursor_ms
                    self.fetchtime_per_symbol[symbol] = max(
                        int(self.fetchtime_per_symbol.get(symbol, 0) or 0),
                        cursor_ms,
                    )
                    self._mark_account_cache_dirty_locked()
        if self._is_account_cache():
            with self.lock:
                self._last_complete_sync_at_ms = (
                    account_cycle_high_water or 0)
        _bapi_ws_health.clear_ws_rest_resync_for_manager(self.cls_name)
        return True

    def on_items_update(self, symbol, items):
        print(f"[{self.cls_name}][Info] {symbol}: WS Items updated to {items}")
        with self.lock:
            if not self.fetchtime_per_symbol:
                self.fetchtime_per_symbol = self._rebuild_fetchtimes()
                for sym, rebuilt_ft in self.fetchtime_per_symbol.items():
                    self._rest_sync_cursor_per_symbol.setdefault(sym, rebuilt_ft)
        self._persist_items(symbol, items)
        
    def _should_poll(self):
        """Decide whether the synchronization loop polls the API.

        Cache24PriceManager is push-only, while CacheCurrentPriceManager polls only when
        WebSocket is unavailable. The default uses the global ``WS_ONLY_MODE`` gate.
        """
        return _should_poll_for_manager(self.cls_name)

    def _persist_periodic_state(self, sync_complete):
        """Persist one cycle without reversing the Trade or Order generation lock order."""
        if self.cls_name in _TRADE_CACHE_CLASSES and self.save_state:
            with self._trade_generation_lock():
                persisted = (
                    self._save_account_trade_append_if_current_generation_locked()
                )
                with self.lock:
                    publish = (
                        sync_complete
                        and persisted
                        and self._is_canonical_account_cache()
                        and bool(self._persisted_data_version)
                        and self._last_complete_sync_at_ms > 0
                    )
                    version = self._persisted_data_version
                    sync_at_ms = self._last_complete_sync_at_ms
                if publish:
                    account_cache_health.record_successful_sync(
                        self.cls_name, version, now_ms=sync_at_ms
                    )
                return persisted

        persisted = self.save_state_to_file_if_enabled()
        with self.lock:
            if (sync_complete and persisted
                    and self._is_canonical_account_cache()
                    and self._persisted_data_version
                    and self._last_complete_sync_at_ms > 0):
                account_cache_health.record_successful_sync(
                    self.cls_name, self._persisted_data_version,
                    now_ms=self._last_complete_sync_at_ms,
                )
            return persisted

    def periodic_sync(self, sync_ts=None, save_state=True):
        thread = self.thread
        if (
            thread is not None
            and thread.is_alive()
            and not self.read_only
            and sync_ts is None
            and (not save_state or self.save_state)
        ):
            return thread
        with self.lock:
            self.read_only = False
            if sync_ts is not None:
                self.sync_ts = sync_ts

            if self.thread is not None and self.thread.is_alive():
                # A later reader-style factory lookup must never demote the sole writer.
                if save_state:
                    self.save_state = True
                return self.thread  # Return the existing thread; it reads sync_ts dynamically.

            self.save_state = save_state

            self._stop_event.clear()

            def run():
                if self._first_sleep and self._stop_event.wait(self.sync_ts):
                    return   # Run the first iteration after one interval, as CurrentPrice requires.
                last_maint = time.time()
                last_resync = time.time()
                while not self._stop_event.is_set():
                    try:
                        sync_complete = False
                        if self._should_poll():
                            sync_complete = self.query_remote_and_update_cache()
                        self._persist_periodic_state(sync_complete)
                        # Periodically reconcile memory and disk.
                        if (time.time() - last_resync) > self.RESYNC_INTERVAL_SEC:
                            self.resync_mem_file()
                            last_resync = time.time()
                        # Run weekly retention/rotation for every append-mode cache, including
                        # full-JSON managers that previously received no effective retention.
                        if self.append_mode and (time.time() - last_maint) > self.RETENTION_CHECK_INTERVAL_SEC:
                            self.maintain_append_persist()
                            last_maint = time.time()
                    except Exception as _e:   # Transient network/HTTP errors must not kill synchronization.
                        builtins.print(
                            f"[{self.cls_name}] synchronization error (continuing): {_e}")
                    if self._stop_event.wait(self.sync_ts):
                        break

            thread = threading.Thread(target=run, name=self.cls_name, daemon=True)
            thread.daemon = True  # Do not let this thread block process shutdown.
            thread.start()
            self.thread = thread
            return self.thread

    def shutdown(self, timeout=5.0):
        """Deterministically stop the synchronization loop started by ``periodic_sync``."""
        self._stop_event.set()
        thread = self.thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        stopped = thread is None or not thread.is_alive()
        if stopped:
            self.thread = None
        return stopped

    @classmethod
    def shutdown_all_instances(cls, timeout=5.0):
        results = [manager.shutdown(timeout=timeout)
                   for manager in list(cls._live_instances)]
        return all(results) if results else True
    
    def enable_save_state_to_file(self):
        self.save_state = True



# ###### 
# ###### Cache-specific implementations
# ###### 

class CacheTradeManager(CacheManagerInterface):
    def __init__(self, sync_ts, symbols, filename, api_client=api, read_only=False):
        super().__init__(sync_ts, symbols, filename, append_mode=True,
                         api_client=api_client, append_persist=True,
                         read_only=read_only)

    @staticmethod
    def _is_valid_item(trade):
        if not isinstance(trade, dict):
            return False
        required_keys = (
            "symbol", "id", "orderId", "price", "qty", "time", "isBuyer"
        )
        if not all(key in trade for key in required_keys):
            return False
        try:
            price = float(trade["price"])
            quantity = float(trade["qty"])
            timestamp = int(trade["time"])
            trade_id = int(trade["id"])
            order_id = int(trade["orderId"])
        except (TypeError, ValueError, OverflowError):
            return False
        return (
            isinstance(trade["symbol"], str) and bool(trade["symbol"].strip())
            and not isinstance(trade["id"], bool) and trade_id >= 0
            and not isinstance(trade["orderId"], bool) and order_id > 0
            and math.isfinite(price) and price > 0
            and math.isfinite(quantity) and quantity > 0
            and timestamp > 0 and isinstance(trade["isBuyer"], bool)
        )

    def get_remote_items(self, symbol, startTime):
        # Paginate through the injected client so periods with more than 1,000 trades
        # are not truncated.
        from binance_api import bapi_allorders as apiorders
        new_trades = apiorders.paginate_my_trades(
            self.api_client.client, symbol, startTime, limit=1000, strict=True)

        print(f"[{self.cls_name}][info] New trades: {len(new_trades)}")
        for t in new_trades:
            if (
                not self._is_valid_item(t)
                or str(t["symbol"]).upper() != str(symbol).upper()
            ):
                raise RuntimeError(
                    f"{self.cls_name} received a malformed Binance trade row: {t!r}"
                )
        return new_trades

    def update_cache_per_symbol(self, symbol, new_items):
        """Route direct update_cache_per_symbol calls through full-history ID deduplication."""
        if new_items is None:
            return
        if isinstance(new_items, dict):
            new_items = [new_items]
        elif not isinstance(new_items, list):
            new_items = list(new_items) if isinstance(new_items, tuple) else [new_items]
        self._persist_items(symbol, new_items)

    def _persist_items(self, symbol, new_items):
        """Append immutable fills once and reject conflicting duplicate IDs."""
        if not new_items:
            return
        if isinstance(new_items, dict):
            new_items = [new_items]
        elif not isinstance(new_items, list):
            new_items = list(new_items) if isinstance(new_items, tuple) else [new_items]
        with self.lock:
            bucket = self.cache.setdefault(symbol, [])
            persisted_count = self._persisted_counts.get(symbol, 0)
            existing = {}
            for idx, item in enumerate(bucket):
                if (
                    not self._is_valid_item(item)
                    or str(item["symbol"]).upper() != str(symbol).upper()
                ):
                    raise RuntimeError("invalid Trade row already in memory")
                trade_id = str(int(item["id"]))
                signature = self._trade_semantic_signature(item)
                if trade_id in existing:
                    raise RuntimeError("duplicate immutable Trade ID in memory")
                existing[trade_id] = (signature, idx, item)

            unique = []
            enriched_unpersisted = False
            for item in new_items:
                if (
                    not self._is_valid_item(item)
                    or str(item["symbol"]).upper() != str(symbol).upper()
                ):
                    raise RuntimeError("invalid incoming Trade row")
                trade_id = str(int(item["id"]))
                signature = self._trade_semantic_signature(item)
                if trade_id in existing:
                    prev_sig, prev_idx, prev_item = existing[trade_id]
                    if prev_sig != signature:
                        raise RuntimeError(
                            f"conflicting Trade row for immutable ID {trade_id}"
                        )
                    if prev_idx >= persisted_count:
                        for fee_key in ("commission", "commissionAsset"):
                            if fee_key not in prev_item and item.get(fee_key) is not None:
                                prev_item[fee_key] = item[fee_key]
                                enriched_unpersisted = True
                    continue
                item_copy = dict(item)
                existing[trade_id] = (signature, len(bucket) + len(unique), item_copy)
                unique.append(item_copy)
            if not unique:
                if enriched_unpersisted:
                    self._mark_account_cache_dirty_locked()
                return
            bucket.extend(unique)
            if len(bucket) - persisted_count > 1:
                unpersisted_tail = bucket[persisted_count:]
                unpersisted_tail.sort(key=lambda t: (int(t["time"]), int(t["id"])))
                bucket[persisted_count:] = unpersisted_tail
            self._preserve_rest_sync_cursor_before_update_locked(symbol)
            now_ms = int(time.time() * 1000)
            self.fetchtime_per_symbol[symbol] = max(
                int(self.fetchtime_per_symbol.get(symbol, 0) or 0),
                now_ms,
            )
            self._mark_account_cache_dirty_locked()

    def get_last_opposite_fill_price(self, symbol, order_type):
        """Return the latest opposite fill price without a time limit.

        BUY uses the latest SELL and SELL uses the latest BUY. Read the manager's own
        real WebSocket fill cache without an API call or noise from canceled orders.
        REST overlap may append an older fill after a newer WebSocket event, so
        authoritative exchange time and immutable trade ID determine recency.
        """
        self.sync_account_reader_if_needed()
        want_buyer = (order_type.upper() == "SELL")   # The opposite of SELL is BUY.
        latest_key = None
        latest_price = None
        with self.lock:
            for trade in self.cache.get(symbol, []):
                if (
                    not self._is_valid_item(trade)
                    or str(trade["symbol"]).upper() != str(symbol).upper()
                    or trade["isBuyer"] != want_buyer
                ):
                    continue
                key = (int(trade["time"]), int(trade["id"]))
                if latest_key is None or key > latest_key:
                    latest_key = key
                    latest_price = float(trade["price"])
        return latest_price


CacheTrade = CacheTradeManager


class CacheOrderManager(CacheManagerInterface):
    def __init__(self, sync_ts, symbols, filename, api_client=api, read_only=False):
        # Orders are mutable: an execution report updates the same order from partial
        # to filled. Keep an atomic snapshot so an in-place update is persisted.
        super().__init__(sync_ts, symbols, filename, append_mode=True,
                         api_client=api_client, append_persist=False,
                         read_only=read_only)
        
    def _is_valid_item(self, trade):
        if not isinstance(trade, dict):
            return False
        required_keys = ("orderId", "price", "quantity", "timestamp", "side")
        if not all(key in trade for key in required_keys):
            return False
        try:
            order_id = int(trade["orderId"])
            price = float(trade["price"])
            quantity = float(trade["quantity"])
            timestamp = int(trade["timestamp"])
            legacy_cutoff = trade.get("_legacyCoveredThrough")
            if legacy_cutoff is not None:
                legacy_cutoff = int(legacy_cutoff)
        except (TypeError, ValueError, OverflowError):
            return False
        return (
            not isinstance(trade["orderId"], bool) and order_id > 0
            and math.isfinite(price) and price > 0
            and math.isfinite(quantity) and quantity > 0
            and timestamp > 0
            and str(trade["side"]).upper() in ("BUY", "SELL")
            and (
                legacy_cutoff is None
                or (not isinstance(trade["_legacyCoveredThrough"], bool)
                    and legacy_cutoff > 0)
            )
            and (
                "_fillIds" not in trade
                or (
                    isinstance(trade["_fillIds"], list)
                    and len({
                        str(fill_id) for fill_id in trade["_fillIds"]
                    }) == len(trade["_fillIds"])
                    and all(
                        str(fill_id).isdigit()
                        and int(fill_id) >= 0
                        for fill_id in trade["_fillIds"]
                    )
                )
            )
        )

    def get_remote_items(self, symbol, startTime):
        from binance_api import bapi_allorders as apiorders
        fills = apiorders.paginate_my_trades(
            self.api_client.client, symbol, startTime, limit=1000, strict=True
        )
        order_fills = []
        for fill in fills:
            if (
                not CacheTradeManager._is_valid_item(fill)
                or str(fill["symbol"]).upper() != str(symbol).upper()
            ):
                raise RuntimeError(
                    f"{self.cls_name} received a malformed Binance fill row: {fill!r}"
                )
            order_fills.append({
                "orderId": fill["orderId"],
                "price": extract_order_price(fill),
                "quantity": extract_order_qty(fill),
                "timestamp": extract_timestamp_ms(fill, default=0) or 0,
                "side": extract_order_side(fill),
                "_fillId": str(fill["id"]),
            })
        print(f"[{self.cls_name}][info] New order fills: {len(order_fills)}")
        return order_fills

    def update_cache_per_symbol(self, symbol, new_items):
        """Route direct update_cache_per_symbol calls through fill-level order aggregation."""
        if new_items is None:
            return
        if isinstance(new_items, dict):
            new_items = [new_items]
        elif not isinstance(new_items, list):
            new_items = list(new_items) if isinstance(new_items, tuple) else [new_items]
        self._persist_items(symbol, new_items)

    def _persist_items(self, symbol, new_items):
        """Merge each immutable fill exactly once into its order aggregate.

        Each order row tracks the set of exchange trade IDs merged into it under
        ``_fillIds`` so overlapping WebSocket executionReports and REST
        ``get_my_trades`` polls are idempotently deduplicated by ``_fillId``.

        Legacy migration note (``_legacyCoveredThrough``):
        Order aggregates persisted before ``_fillIds`` was introduced do not list
        their constituent fill IDs. When such a row is touched, its initial
        ``timestamp`` is frozen in ``_legacyCoveredThrough`` so any historical
        fill with ``timestamp <= _legacyCoveredThrough`` is recognized as already
        counted. Once all pre-``_fillIds`` open orders have closed and aged past
        ``ACCOUNT_REST_OVERLAP_MS`` (and no manual historical backfill over
        pre-``_fillIds`` rows is needed), ``_legacyCoveredThrough`` can be removed
        and deduplication relies solely on ``_fillIds``.
        """
        if not new_items:
            return
        if isinstance(new_items, dict):
            new_items = [new_items]
        elif not isinstance(new_items, list):
            new_items = list(new_items) if isinstance(new_items, tuple) else [new_items]
        now_ms = int(time.time() * 1000)
        with self.lock:
            bucket = self.cache.setdefault(symbol, [])
            candidate = [dict(item) for item in bucket]
            by_id = {
                str(item.get("orderId")): index
                for index, item in enumerate(candidate)
                if item.get("orderId") is not None
            }
            for item in new_items:
                key = str(item["orderId"])
                incoming = dict(item)
                raw_fill_id = incoming.pop("_fillId", None)
                fill_id = (
                    "" if raw_fill_id is None else str(raw_fill_id).strip()
                )
                existing_index = by_id.get(key)
                if existing_index is None:
                    by_id[key] = len(candidate)
                    incoming["price"] = float(incoming["price"])
                    incoming["quantity"] = float(incoming["quantity"])
                    incoming["timestamp"] = int(incoming["timestamp"])
                    incoming["side"] = str(incoming["side"]).upper()
                    if fill_id:
                        incoming["_fillIds"] = [fill_id]
                    candidate.append(incoming)
                    continue

                existing = candidate[existing_index]
                known_fill_id_list = [
                    str(existing_fill_id)
                    for existing_fill_id in existing.get("_fillIds", [])
                ]
                legacy_cutoff = existing.get("_legacyCoveredThrough")
                if legacy_cutoff is None and "_fillIds" not in existing:
                    # The old aggregate predates immutable fill IDs. Preserve its
                    # last covered timestamp even after newer identified fills arrive.
                    legacy_cutoff = int(existing["timestamp"])
                known_fill_ids = {*known_fill_id_list}
                if fill_id and fill_id in known_fill_ids:
                    continue
                if (
                    legacy_cutoff is not None
                    and int(incoming["timestamp"])
                    <= int(legacy_cutoff)
                ):
                    # Unknown fills at or before the durable legacy cutoff were
                    # already represented by the aggregate and must not be counted.
                    continue
                if (
                    str(existing.get("side", "")).upper()
                    != str(incoming["side"]).upper()
                ):
                    raise RuntimeError(
                        f"{self.cls_name} received conflicting sides for order {key}"
                    )
                previous_quantity = float(existing["quantity"])
                additional_quantity = float(incoming["quantity"])
                total_quantity = previous_quantity + additional_quantity
                aggregate_price = (
                    float(existing["price"]) * previous_quantity
                    + float(incoming["price"]) * additional_quantity
                ) / total_quantity
                merged = dict(existing)
                merged.update(incoming)
                merged.update({
                    "price": round(aggregate_price, 8),
                    "quantity": round(total_quantity, 8),
                    "timestamp": max(
                        int(existing["timestamp"]),
                        int(incoming["timestamp"]),
                    ),
                })
                if fill_id:
                    merged["_fillIds"] = list(
                        dict.fromkeys([*known_fill_id_list, fill_id])
                    )
                if legacy_cutoff is not None:
                    merged["_legacyCoveredThrough"] = int(legacy_cutoff)
                candidate[existing_index] = merged

            if candidate != bucket:
                self.cache[symbol] = candidate
                self._preserve_rest_sync_cursor_before_update_locked(symbol)
                self.fetchtime_per_symbol[symbol] = max(
                    int(self.fetchtime_per_symbol.get(symbol, 0) or 0),
                    now_ms,
                )
                self._mark_account_cache_dirty_locked()


CacheOrder = CacheOrderManager


class CacheSparsePriceManager(CacheManagerInterface):
    """Store sparse three-minute price samples for two years in JSONL.

    Unlike dense WebSocket-pushed Cache24 managers, this periodically asks for the
    current price. ``Sparse`` describes the stable sampling mechanism rather than the
    mutable retention policy. priceAnalysis is its only consumer.
    """

    def __init__(self, sync_ts, symbols, filename, api_client=api, read_only=False):
        # Continuous append-only three-minute history uses JSONL append persistence.
        super().__init__(sync_ts, symbols, filename, append_mode=True,
                         api_client=api_client, append_persist=True,
                         read_only=read_only)

    def rebuild_fetchtimes(self):
        if not self.cache:
            return {}
        last_times = {}
        for symbol in self.symbols:
            entries = self.cache.get(symbol, [])
            if entries:
                last_times[symbol] = max(entry[0] for entry in entries)
        return last_times

    def get_remote_items(self, symbol, startTime):
        # Preserve the observation timestamp supplied by get_price. Using only a cached
        # value with ``time.time`` would record a stale price as freshly observed during
        # a network outage, corrupting the long-term history.
        try:
            entry = get_current_price_manager().get_price(symbol)
        except Exception as e:
            print(f"[{self.cls_name}][Error] get_price {symbol}: {e}")
            return []

        if not entry:
            return []
        ts_ms, price = entry
        if price is None:
            return []
        try:
            ts_val = int(ts_ms)
            price_val = float(price)
        except (TypeError, ValueError, OverflowError):
            return []
        if ts_val <= 0 or not math.isfinite(price_val) or price_val <= 0:
            return []
        with self.lock:
            entries = self.cache.get(symbol)
            if entries and ts_val < int(entries[-1][0]):
                return []

        return [[ts_val, price]]

    def get_all_symbols_from_cache(self):
        return self.symbols


CacheSparsePrice = CacheSparsePriceManager


class Cache24PriceManager(CacheManagerInterface):
    """Collect maximum-granularity prices over the latest ``KEEP_HOURS``.

    This manager neither polls nor subscribes directly to WebSocket. It receives every
    update from CacheCurrentPriceManager and uses ``get_remote_items`` only during
    initialization when the persisted file is missing.
    """
    KEEP_HOURS = 24   # May be configured per instance.

    def __init__(self, sync_ts, symbols, filename, api_client=api, read_only=False):
        super().__init__(sync_ts, symbols, filename, append_mode=True,
                         api_client=api_client, read_only=read_only)
        # Remove unrelated symbols loaded from legacy redundant files that stored every coin.
        with self.lock:
            self.cache = {s: v for s, v in self.cache.items() if s in self.symbols}

    # Inherit price subscription and notification methods from CacheManagerInterface.

    # ── CacheCurrentPriceManager callback ────────────────────────────────────

    def on_price_update(self, symbol: str, ts_ms: int, price: float):
        """Receive every new WebSocket or HTTP price from CacheCurrentPriceManager."""
        # CurrentPrice broadcasts every symbol to every subscriber, while this manager is
        # per-symbol. Store and forward only this instance's symbol.
        if symbol not in self.symbols:
            return
        try:
            if isinstance(ts_ms, bool) or isinstance(price, bool):
                return
            ts_val = int(ts_ms)
            price_val = float(price)
        except (TypeError, ValueError, OverflowError):
            return
        if ts_val <= 0 or not math.isfinite(price_val) or price_val <= 0:
            return
        with self.lock:
            entries = self.cache.get(symbol)
            if entries and ts_val < int(entries[-1][0]):
                return
        if not self.fetchtime_per_symbol:
            self.fetchtime_per_symbol = self._rebuild_fetchtimes()
        self.update_cache_per_symbol(symbol, [[ts_ms, price]])
        self._trim_old_data(symbol)
        self._notify_price_subscribers(symbol, ts_ms, price)

    def _trim_old_data(self, symbol):
        """Remove entries older than ``KEEP_HOURS`` efficiently on every tick.

        Entries are append-only and time-ordered, so bisect finds the first unexpired
        index in O(log n) without copying when nothing expired. This avoids an O(n)
        list comprehension on every tick as long archives grow.
        """
        cutoff_ms = int((time.time() - self.KEEP_HOURS * 3600) * 1000)
        with self.lock:
            entries = self.cache.get(symbol)
            if not entries:
                return
            idx = bisect.bisect_left(entries, cutoff_ms, key=lambda e: e[0])
            if idx > 0:
                self.cache[symbol] = entries[idx:]

    def get_recent_entries(self, symbol: str, last_seconds: float) -> list:
        """Return ``[timestamp_ms, price]`` entries from the latest interval.

        Time-ordered append-only entries let bisect find the cutoff in O(log n); copying
        costs O(k) for returned entries instead of scanning the entire 24-hour buffer.
        This matters for frequent dynamic-window queries.
        """
        cutoff_ms = int((time.time() - last_seconds) * 1000)
        with self.lock:
            entries = self.cache.get(symbol, [])
            if not entries:
                return []
            idx = bisect.bisect_left(entries, cutoff_ms, key=lambda e: e[0])
            return entries[idx:]

    # ── CacheManagerInterface ─────────────────────────────────────────────────

    def rebuild_fetchtimes(self):
        if not self.cache:
            return {}
        last_times = {}
        for symbol in self.symbols:
            entries = self.cache.get(symbol, [])
            if entries:
                last_times[symbol] = max(entry[0] for entry in entries)
        return last_times

    def get_remote_items(self, symbol, startTime):
        """Fetch initialization data only when the persisted file is missing.

        Preserve the real observation timestamp from ``get_price`` so a cached value
        during an outage is not recorded as falsely fresh.
        """
        try:
            entry = get_current_price_manager().get_price(symbol)
            if not entry or entry[1] is None:
                return []
            ts_ms, price = entry
            if isinstance(ts_ms, bool) or isinstance(price, bool):
                return []
            ts_val = int(ts_ms)
            price_val = float(price)
            if ts_val <= 0 or not math.isfinite(price_val) or price_val <= 0:
                return []
            with self.lock:
                entries = self.cache.get(symbol)
                if entries and ts_val < int(entries[-1][0]):
                    return []
            return [[ts_val, price]]
        except Exception as e:
            print(f"[{self.cls_name}][Error] get_price {symbol}: {e}")
            return []

    def get_all_symbols_from_cache(self):
        return self.symbols

    def _should_poll(self):
        # Prices arrive exclusively through CurrentPrice's ``on_price_update`` callback.
        # Base periodic_sync only saves, resynchronizes, and maintains; it does not poll.
        return False


Cache24Price = Cache24PriceManager


class Cache365PriceManager(Cache24PriceManager):
    """Multi-month/multi-year dense archive variant using incremental JSONL persistence.

    This remains separate from the live 24-hour manager and inherits its price-update
    behavior while overriding persistence only. JSONL appends new ticks at constant cost
    instead of rewriting a growing archive every minute.

    The base JSONL writer tracks how many entries it wrote and assumes the list grows only
    at the tail. Trimming from the head would otherwise make retained entries appear new
    and duplicate them, so this class compacts whenever trimming actually removes data.
    """

    LONG_TRIM_INTERVAL_SEC = 24 * 3600

    def __init__(self, sync_ts, symbols, filename, api_client=api, read_only=False):
        # Long archives must not rewrite the entire JSONL whenever a single oldest
        # tick expires.  Initialize the maintenance clock before callbacks can run.
        self._last_long_trim = 0.0
        # Call CacheManagerInterface directly because Cache24PriceManager does not expose
        # append_persist and invokes load_state before this class could change it.
        CacheManagerInterface.__init__(self, sync_ts, symbols, filename,
                                        append_mode=True, api_client=api_client,
                                        append_persist=True, read_only=read_only)
        # Mirror Cache24PriceManager's cleanup of unrelated symbols from legacy files.
        with self.lock:
            self.cache = {s: v for s, v in self.cache.items() if s in self.symbols}

    @staticmethod
    def _tail_lines(path, limit, block_size=1024 * 1024):
        """Read at most the last complete ``limit`` lines without scanning the file."""
        return read_tail_lines(path, limit, chunk_size=block_size)

    def _load_jsonl(self):
        """Load only a recent working set; the JSONL file remains the full archive."""
        with self.lock:
            self.cache = {}
            backup_path = self._previous_generation_path()
            if not os.path.exists(self.filename) and os.path.exists(backup_path):
                try:
                    self._restore_previous_generation(backup_path)
                except OSError:
                    pass
            if os.path.exists(self.filename):
                for raw_line in self._tail_lines(self.filename, CM_LONG_ARCHIVE_MEMORY_ROWS):
                    try:
                        rec = json.loads(raw_line)
                        if rec.get("s") in self.symbols:
                            self.cache.setdefault(rec["s"], []).append(rec["i"])
                    except (json.JSONDecodeError, KeyError, TypeError):
                        continue
            self._persisted_counts = {s: len(v) for s, v in self.cache.items()}
            metaf = self.filename + ".meta"
            if os.path.exists(metaf):
                try:
                    with open(metaf) as handle:
                        meta_payload = json.load(handle)
                        self.fetchtime_per_symbol = meta_payload.get("fetchtime", {})
                        rest_cursor = meta_payload.get("rest_sync_cursor")
                        self._rest_sync_cursor_per_symbol = dict(
                            rest_cursor
                            if isinstance(rest_cursor, dict)
                            else self.fetchtime_per_symbol
                        )
                except (OSError, ValueError, TypeError):
                    pass

    def _trim_old_data(self, symbol):
        # Bound the in-process working set on every tick. Removed rows are already at
        # the head and remain in the append-only disk archive.
        with self.lock:
            entries = self.cache.get(symbol, [])
            overflow = max(0, len(entries) - CM_LONG_ARCHIVE_MEMORY_ROWS)
            if overflow:
                self.cache[symbol] = entries[overflow:]
                self._persisted_counts[symbol] = max(
                    0, self._persisted_counts.get(symbol, 0) - overflow)
        now = time.monotonic()
        if now - self._last_long_trim < self.LONG_TRIM_INTERVAL_SEC:
            return
        self._last_long_trim = now
        with self.lock:
            before = len(self.cache.get(symbol, []))
        super()._trim_old_data(symbol)   # Use the inherited trimming logic unchanged.
        with self.lock:
            after = len(self.cache.get(symbol, []))
            if after < before:
                removed = before - after
                self._persisted_counts[symbol] = max(0, self._persisted_counts.get(symbol, 0) - removed)

    def maintain_append_persist(self):
        """Prune the full disk archive with bounded memory, then reload its recent tail."""
        if not self.save_state:
            return False
        cutoff_ms = int((time.time() - self.RETENTION_DAYS * 86400) * 1000)
        if not os.path.exists(self.filename):
            return False
        changed = False
        backup_path = None
        original_cache = None
        original_persisted_counts = None
        original_fetchtimes = None
        original_rest_cursors = None
        try:
            with self.lock:
                original_cache = {
                    symbol: list(items) for symbol, items in self.cache.items()
                }
                original_persisted_counts = dict(self._persisted_counts)
                original_fetchtimes = dict(self.fetchtime_per_symbol)
                original_rest_cursors = dict(self._rest_sync_cursor_per_symbol)
                backup_path = self.filename + ".maint_backup"
                atomic_snapshot_file(self.filename, backup_path)
                with atomic_write(self.filename) as target, open(backup_path, encoding="utf-8") as source:
                    for line in source:
                        try:
                            rec = json.loads(line)
                            keep = self._entry_timestamp_ms(rec.get("i")) >= cutoff_ms
                        except (json.JSONDecodeError, TypeError, ValueError):
                            keep = False
                        if keep:
                            target.write(line if line.endswith("\n") else line + "\n")
                        else:
                            changed = True
                    for symbol, items in original_cache.items():
                        start = min(max(0, original_persisted_counts.get(symbol, 0)), len(items))
                        for item in items[start:]:
                            if self._entry_timestamp_ms(item) >= cutoff_ms:
                                target.write(
                                    json.dumps({"s": symbol, "i": item}, separators=(",", ":")) + "\n"
                                )
                            changed = True
                self._load_jsonl()
                for sym, ft in original_fetchtimes.items():
                    self.fetchtime_per_symbol[sym] = max(
                        int(self.fetchtime_per_symbol.get(sym, 0) or 0),
                        int(ft or 0),
                    )
                for sym, cur in original_rest_cursors.items():
                    self._rest_sync_cursor_per_symbol[sym] = max(
                        int(self._rest_sync_cursor_per_symbol.get(sym, 0) or 0),
                        int(cur or 0),
                    )
                if changed:
                    if not self._write_meta({
                        "counts": dict(self._persisted_counts),
                        "fetchtime": dict(self.fetchtime_per_symbol),
                        "rest_sync_cursor": dict(self._rest_sync_cursor_per_symbol),
                    }):
                        self.cache = original_cache
                        self._persisted_counts = original_persisted_counts
                        self.fetchtime_per_symbol = original_fetchtimes
                        self._rest_sync_cursor_per_symbol = original_rest_cursors
                        self._restore_previous_generation(backup_path)
                        backup_path = None
                        return False
                if backup_path is not None:
                    self._discard_generation_file(backup_path)
                    backup_path = None
            if changed:
                builtins.print(f"[{self.cls_name}][maintain] pruned expired/corrupt disk records")
            if os.path.getsize(self.filename) > self.MAX_FILE_BYTES:
                return self._rotate_disk_archive()
            return True
        except OSError as exc:
            if original_cache is not None:
                self.cache = original_cache
                self._persisted_counts = original_persisted_counts
                self.fetchtime_per_symbol = original_fetchtimes
                self._rest_sync_cursor_per_symbol = original_rest_cursors
            if backup_path is not None and os.path.exists(backup_path):
                try:
                    self._restore_previous_generation(backup_path)
                except OSError:
                    pass
            builtins.print(f"[{self.cls_name}][Error] streaming maintenance {self.filename}: {exc}")
            return False

    def compact_jsonl(self):
        """Never rebuild an existing long archive from its bounded memory tail."""
        if os.path.exists(self.filename):
            return self.maintain_append_persist()
        return super().compact_jsonl()  # Legacy JSON migration still starts from full memory.

    def _rotate_disk_archive(self):
        """Archive an oversized file while retaining a complete-line tail on disk."""
        with self.lock:
            archive = f"{self.filename}.{int(time.time())}.archive"
            try:
                atomic_snapshot_file(self.filename, archive)
            except OSError as exc:
                builtins.print(f"[{self.cls_name}][maintain] archiving failed: {exc}")
                return False
            original_cache = {
                symbol: list(items) for symbol, items in self.cache.items()
            }
            original_persisted_counts = dict(self._persisted_counts)
            original_fetchtimes = dict(self.fetchtime_per_symbol)
            original_rest_cursors = dict(self._rest_sync_cursor_per_symbol)
            try:
                archive_size = os.path.getsize(archive)
                keep_bytes = max(1, int(archive_size * self.ROTATE_KEEP_FRACTION))
                with open(archive, "rb") as source:
                    source.seek(max(0, archive_size - keep_bytes))
                    if source.tell() > 0:
                        source.readline()
                    tail = source.read()
                    if not tail.strip() and archive_size > 0:
                        source.seek(0)
                        lines = [line for line in source.read().splitlines(keepends=True) if line.strip()]
                        tail = lines[-1] if lines else b""
                with atomic_write(self.filename) as target:
                    target.write(tail.decode("utf-8", errors="ignore"))
                self._load_jsonl()
                for sym, ft in original_fetchtimes.items():
                    self.fetchtime_per_symbol[sym] = max(
                        int(self.fetchtime_per_symbol.get(sym, 0) or 0),
                        int(ft or 0),
                    )
                for sym, cur in original_rest_cursors.items():
                    self._rest_sync_cursor_per_symbol[sym] = max(
                        int(self._rest_sync_cursor_per_symbol.get(sym, 0) or 0),
                        int(cur or 0),
                    )
                meta_ok = self._write_meta({
                    "counts": dict(self._persisted_counts),
                    "fetchtime": dict(self.fetchtime_per_symbol),
                    "rest_sync_cursor": dict(self._rest_sync_cursor_per_symbol),
                })
            except Exception as exc:
                builtins.print(f"[{self.cls_name}][maintain] rotation rewrite failed: {exc}")
                meta_ok = False
            if not meta_ok:
                self.cache = original_cache
                self._persisted_counts = original_persisted_counts
                self.fetchtime_per_symbol = original_fetchtimes
                self._rest_sync_cursor_per_symbol = original_rest_cursors
                try:
                    self._restore_previous_generation(archive)
                except OSError as restore_err:
                    builtins.print(f"[{self.cls_name}][maintain] failed to restore active file from {archive}: {restore_err}")
                builtins.print(f"[{self.cls_name}][maintain] rotation aborted; restored active file and in-memory state")
                return False
            try:
                atomic_gzip_file(archive, archive + ".gz", remove_source=True)
            except OSError as exc:
                builtins.print(f"[{self.cls_name}][maintain] archive compression failed: {exc}")
            prune_rotated_archives(self.filename, self.ROTATE_ARCHIVE_COUNT)
            return True

    # Inherit get_remote_items from Cache24PriceManager, including real observation timestamps.


Cache365Price = Cache365PriceManager


if __name__ == "__main__":
    import sys as _sys
    _sys.modules.setdefault("cacheManager", _sys.modules[__name__])

import cachetrend as _cache_trend

CacheLongTrendManager = getattr(_cache_trend, "CacheLongTrendManager", None)
CacheLongTrend = CacheLongTrendManager


class CacheAssetValueManager(CacheManagerInterface):
    def __init__(self, sync_ts, symbols, filename, api_client=api, read_only=False):
        self._migrated_usdt_on_load = False
        super().__init__(sync_ts, symbols, filename, append_mode=True,
                         api_client=api_client, append_persist=True,
                         read_only=read_only)
        if self._migrated_usdt_on_load:
            self.compact_jsonl()

    def _load_jsonl_generation_locked(self):
        super()._load_jsonl_generation_locked()
        changed = False
        with self.lock:
            for items in self.cache.values():
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    if "total_value_usdc" not in item and "total_value_usdt" in item:
                        item["total_value_usdc"] = item.pop("total_value_usdt")
                        changed = True
        if changed:
            self._migrated_usdt_on_load = True

    def rebuild_fetchtimes(self):
        last_times = {}
        for symbol, items in self.cache.items():
            if not items:
                continue
            max_ts_sec = max(
                (extract_timestamp_s(item, default=0.0) or 0.0 for item in items if isinstance(item, dict)),
                default=0.0,
            )
            if max_ts_sec > 0:
                last_times[symbol] = max(0, int(max_ts_sec * 1000) - 60_000)
        return last_times

    def get_remote_items(self, symbol, startTime):
        try:
            total_usdc = self.api_client.get_total_assets_value_usdc(use_cache=False, raise_errors=True)
        except Exception as e:
            print(f"[{self.cls_name}][Error] Cannot query the total value: {e}")
            return []

        if (
            isinstance(total_usdc, bool)
            or not isinstance(total_usdc, (int, float))
            or not math.isfinite(total_usdc)
            or total_usdc <= 0
        ):
            print(f"[{self.cls_name}][Error] Invalid total value: {total_usdc}")
            return []
            
        now_sec = int(time.time())
        snapshot = {
            "timestamp": now_sec,
            "datetime_local": datetime.now().isoformat(timespec="seconds"),
            "total_value_usdc": round(float(total_usdc), 8),
        }
        return [snapshot]


CacheAssetValue = CacheAssetValueManager


# ######
# ###### CacheCurrentPriceManager: per-symbol current price, WebSocket-first with HTTP fallback
# ######

class CacheCurrentPriceManager(CacheManagerInterface):
    """Maintain the latest timestamped price per symbol.

    This is a drop-in replacement for ``bapi.get_current_price``. WebSocket is primary;
    HTTP polling runs only after WebSocket has been silent beyond ``WS_TIMEOUT_SEC``.
    ``get_price`` forces HTTP when the cached entry exceeds ``STALE_THRESHOLD_MS``.
    Snapshot semantics retain one ``[[timestamp_ms, price]]`` entry per symbol in a
    shared cache_current_price.json file.
    """

    WS_TIMEOUT_SEC     = 15      # Treat WebSocket as unavailable after 15 silent seconds.
    STALE_THRESHOLD_MS = 5_000  # Force HTTP when a price is older than five seconds.
    FREQ_WINDOW_SEC    = 60     # Window used to measure update frequency.

    def __init__(self, sync_ts, symbols, filename, ws_manager=None, api_client=api,
                 market_api=None, provider_names=None, read_only=False):
        self._ws_manager        = ws_manager
        self._ws_default_event_ts = 0.0
        self._ws_last_event_ts_value = 0.0
        self._ws_last_event_ts_per_symbol: dict = {}
        self._price_subscribers = []       # The base initializer preserves existing values.
        self._update_timestamps: dict = defaultdict(deque)  # Also required before super().
        self._first_sleep       = True     # Let WebSocket connect before HTTP fallback.
        # The injectable market-data facade defaults to the global singleton. Set it before
        # the base initializer because load_state may immediately call get_remote_items.
        self.market_api = market_api or _market_api.api
        self._provider_names = {}
        self.bind_providers(provider_names)
        super().__init__(sync_ts, symbols, filename, append_mode=False,
                         api_client=api_client, read_only=read_only)
        if ws_manager is not None:
            ws_manager.subscribe(self)

    # ── WS health ────────────────────────────────────────────────────────────

    @property
    def _ws_last_event_ts(self) -> float:
        return self._ws_last_event_ts_value

    @_ws_last_event_ts.setter
    def _ws_last_event_ts(self, value: float) -> None:
        ts = float(value)
        self._ws_last_event_ts_value = ts
        self._ws_default_event_ts = ts
        self._ws_last_event_ts_per_symbol.clear()

    def bind_provider(self, symbol: str, provider_name: str) -> None:
        """Bind one symbol to its configured venue and reject ambiguity."""
        key = str(symbol or "").strip().upper()
        value = str(provider_name or "").strip()
        if not key or not value:
            raise ValueError("symbol and provider_name must be non-empty")
        previous = self._provider_names.get(key)
        if previous is not None and previous.lower() != value.lower():
            raise ValueError(
                f"conflicting providers for {key}: {previous} and {value}"
            )
        self._provider_names[key] = value

    def bind_providers(self, provider_names) -> None:
        """Merge configured symbol-to-venue bindings without implicit fallback."""
        for symbol, provider_name in dict(provider_names or {}).items():
            self.bind_provider(symbol, provider_name)

    def provider_name_for(self, symbol: str):
        return self._provider_names.get(str(symbol or "").strip().upper())

    def _ws_is_healthy(self, symbol: Optional[str] = None) -> bool:
        now = time.time()
        if symbol is not None:
            last_ts = self._ws_last_event_ts_per_symbol.get(
                symbol, self._ws_default_event_ts
            )
            return (now - last_ts) < self.WS_TIMEOUT_SEC
        if not self.symbols:
            return (now - self._ws_last_event_ts_value) < self.WS_TIMEOUT_SEC
        return all(
            (now - self._ws_last_event_ts_per_symbol.get(s, self._ws_default_event_ts))
            < self.WS_TIMEOUT_SEC
            for s in self.symbols
        )

    # Inherit price subscription and notification methods from CacheManagerInterface.

    # ── WebSocket callback overriding the interface method ───────────────────

    def _record_price_timestamp(self, symbol: str) -> None:
        now = time.time()
        dq = self._update_timestamps[symbol]
        dq.append(now)
        cutoff = now - self.FREQ_WINDOW_SEC
        while dq and dq[0] < cutoff:
            dq.popleft()

    def _push_price(self, symbol: str, price: float, ts_ms: Optional[int] = None) -> None:
        """Insert a price and notify subscribers without changing WebSocket health.

        The polling thread and get_price also use this path; only a real WebSocket event
        in ``on_items_update`` updates ``_ws_last_event_ts``.
        """
        if price is None or isinstance(price, bool):
            return
        try:
            price_val = float(price)
        except (TypeError, ValueError, OverflowError):
            return
        if not math.isfinite(price_val) or price_val <= 0:
            return
        ts_ms = int(ts_ms) if ts_ms is not None else int(time.time() * 1000)
        if not self.fetchtime_per_symbol:
            self.fetchtime_per_symbol = self._rebuild_fetchtimes()
        self._record_price_timestamp(symbol)
        self.update_cache_per_symbol(symbol, [[ts_ms, price]])
        self._notify_price_subscribers(symbol, ts_ms, price)

    @staticmethod
    def _extract_price_and_ts(items):
        """Extract ``(price, ts_ms)`` from canonical ``[[ts_ms, price]]`` or ``[ts_ms, price]``."""
        if not items or not isinstance(items, (list, tuple)):
            return None, None
        first = items[0]
        if isinstance(first, (list, tuple)):
            if len(first) < 2 or first[0] is None or first[1] is None:
                return None, None
            raw_ts, raw_price = first[0], first[1]
        elif len(items) >= 2 and items[0] is not None and items[1] is not None:
            raw_ts, raw_price = items[0], items[1]
        else:
            return None, None
        try:
            if isinstance(raw_ts, bool) or isinstance(raw_price, bool):
                return None, None
            ts_ms = int(raw_ts)
            price = float(raw_price)
        except (TypeError, ValueError, OverflowError):
            return None, None
        if not math.isfinite(price) or price <= 0:
            return None, None
        return price, ts_ms

    def on_items_update(self, symbol: str, items):
        """Handle a real WebSocket event and update WebSocket health only for valid prices."""
        price, ts_ms = self._extract_price_and_ts(items)
        if price is None:
            return
        with self.lock:
            entries = self.cache.get(symbol)
            existing_ts = (
                int(entries[0][0])
                if entries and isinstance(entries[0], (list, tuple)) and entries[0]
                else 0
            )
        if existing_ts > 0 and (existing_ts - ts_ms) > self.STALE_THRESHOLD_MS:
            return
        now = time.time()
        self._ws_last_event_ts_value = now
        self._ws_last_event_ts_per_symbol[symbol] = now
        self._push_price(symbol, price, ts_ms=ts_ms)

    # ── CacheManagerInterface abstract methods ───────────────────────────────

    def query_remote_and_update_cache(self):
        """Poll only symbols whose WebSocket stream is unhealthy."""
        if not self.fetchtime_per_symbol:
            self.fetchtime_per_symbol = self._rebuild_fetchtimes()
            self._rest_sync_cursor_per_symbol = dict(self.fetchtime_per_symbol)
        for symbol in list(self.symbols):
            if self._ws_is_healthy(symbol):
                continue
            startTime = self.fetchtime_per_symbol.get(symbol, self.fallback_time_default)
            new_items = self.get_remote_items(symbol=symbol, startTime=startTime)
            if new_items is None:
                raise RuntimeError(
                    f"{self.cls_name} returned no synchronization result for {symbol}")
            if not new_items:
                print(f"[{self.cls_name}][Info] {symbol}:  No remote items starting with {u.timestampToTime(startTime)} ")
            else:
                self._persist_items(symbol, new_items)
        _bapi_ws_health.clear_ws_rest_resync_for_manager(self.cls_name)
        return True

    def get_remote_items(self, symbol, startTime):
        """Fetch current price through the market-data facade."""
        try:
            provider_name = self.provider_name_for(symbol)
            if provider_name is None:
                price = self.market_api.get_current_price(symbol=symbol)
            else:
                price = self.market_api.get_current_price(
                    symbol=symbol, provider_name=provider_name)
            if price is None:
                return []
            ts_ms = int(time.time() * 1000)
            return [[ts_ms, price]]
        except Exception as e:
            print(f"[{self.cls_name}][Error] HTTP fetch {symbol}: {e}")
            return []

    def _persist_items(self, symbol, new_items):
        """Record sample-rate timing and notify subscribers through ``_push_price``."""
        price, ts_ms = self._extract_price_and_ts(new_items)
        if price is None:
            return
        self._push_price(symbol, price, ts_ms=ts_ms)

    def rebuild_fetchtimes(self):
        last_times = {}
        for symbol in self.symbols:
            entries = self.cache.get(symbol, [])
            if entries:
                last_times[symbol] = entries[0][0]   # Snapshot contains one entry.
        return last_times

    def get_all_symbols_from_cache(self):
        return self.symbols

    # ── Periodic synchronization: poll only when WebSocket is unavailable ────

    def _should_poll(self):
        # Fetch through HTTP only as a fallback. The overridden persistence path
        # propagates results through ``_push_price``.
        return not self._ws_is_healthy()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_sample_rate(self, symbol: str, fallback: float = 1.0) -> float:
        """Return the mean update interval, or ``fallback`` when samples are insufficient."""
        dq = self._update_timestamps.get(symbol)
        if not dq or len(dq) < 2:
            return fallback
        return (dq[-1] - dq[0]) / (len(dq) - 1)

    def get_update_frequency(self, symbol: str) -> float:
        """Return updates per second over the latest frequency window."""
        dq = self._update_timestamps.get(symbol)
        if not dq or len(dq) < 2:
            return 0.0
        return len(dq) / self.FREQ_WINDOW_SEC

    def get_cached_price_observation(self, symbol: str, max_age_sec: float):
        """Return a fresh cached timestamp and price without fetching."""
        max_age = float(max_age_sec)
        if not math.isfinite(max_age) or max_age <= 0:
            raise ValueError("max_age_sec must be finite and positive")
        with self.lock:
            entries = self.cache.get(symbol)
            entry = list(entries[0]) if entries else None
        if not entry or len(entry) < 2:
            return None
        try:
            observed_at = float(entry[0]) / 1000.0
            price = float(entry[1])
        except (TypeError, ValueError, OverflowError):
            return None
        age = time.time() - observed_at
        if (
            not math.isfinite(observed_at)
            or observed_at <= 0
            or not math.isfinite(price)
            or price <= 0
            or age < -5.0
            or age > max_age
        ):
            return None
        return observed_at, price

    def get_price(self, symbol: str):
        """Return ``[timestamp_ms, price]``, forcing HTTP when missing or stale."""
        with self.lock:
            entries = self.cache.get(symbol)
        last_ts = entries[0][0] if entries else 0
        now_ms  = int(time.time() * 1000)
        if not entries or (now_ms - last_ts) > self.STALE_THRESHOLD_MS:
            age = now_ms - last_ts if entries else -1
            print(f"[{self.cls_name}] {symbol} stale ({age}ms) - forced HTTP fetch")
            new = self.get_remote_items(symbol, None)
            if not new:
                # Preserve the old row for diagnostics, but never return it as a
                # usable quote after its bounded-age refresh has failed.
                return None
            self._persist_items(symbol, new)
            self.save_state_to_file_if_enabled()
            with self.lock:
                entries = self.cache.get(symbol)
        return entries[0] if entries else None

    def get_price_value(self, symbol: str) -> float:
        """Return only the float price as a drop-in for ``bapi.get_current_price``."""
        entry = self.get_price(symbol)
        return entry[1] if entry else None

    def attach_ws_manager(self, ws_manager) -> None:
        """Attach BinancePriceStream as the idempotent primary price source."""
        if ws_manager is None:
            return
        self._ws_manager = ws_manager
        ws_manager.subscribe(self)


CacheCurrentPrice = CacheCurrentPriceManager


# ── Singleton ─────────────────────────────────────────────────────────────────

_current_price_instance: Optional[CacheCurrentPriceManager] = None
_current_price_lock = threading.RLock()

def _default_price_tracking():
    """Return ``(symbols, nonbinance_symbols, provider_names)`` from ``instruments.conf``."""
    from instrument_registry import tracked_price_instruments
    return tracked_price_instruments()


def get_current_price_manager(ws_manager=None, symbols=None, sync_ts=None,
                              start_sync=True,
                              provider_names=None,
                              read_only=None) -> CacheCurrentPriceManager:
    """Return or lazily create the CacheCurrentPriceManager singleton.

    ``sync_ts=None`` preserves the current interval and uses the configured default on
    first creation. An explicit value updates the live interval read by the thread.
    Internal calls must pass None so they do not override main-process configuration.
    """
    global _current_price_instance
    inst = _current_price_instance
    if (
        inst is not None
        and sync_ts is None
        and not provider_names
        and ws_manager is None
        and (
            not start_sync
            or (
                (thread := inst.thread) is not None
                and thread.is_alive()
                and not inst.read_only
            )
        )
    ):
        return inst
    with _current_price_lock:
        if _current_price_instance is not None:
            if sync_ts is not None:
                _current_price_instance.sync_ts = sync_ts
            if provider_names:
                _current_price_instance.bind_providers(provider_names)
            if ws_manager is not None:
                _current_price_instance.attach_ws_manager(ws_manager)
            if start_sync:
                _current_price_instance.periodic_sync(save_state=False)
            return _current_price_instance
        if symbols is None:
            _syms, _, _default_providers = _default_price_tracking()
            if provider_names is None and _default_providers:
                provider_names = _default_providers
        else:
            _syms = symbols
        instance = CacheCurrentPriceManager(
            sync_ts     = sync_ts if sync_ts is not None else CURRENT_PRICE_SYNC_INTERVAL_SEC,
            symbols     = _syms,
            filename    = "cache_current_price.json",
            ws_manager  = ws_manager,
            provider_names = provider_names,
            api_client  = api,
            read_only   = (not start_sync) if read_only is None else bool(read_only),
        )
        if start_sync:
            instance.periodic_sync(save_state=False)
        _current_price_instance = instance
    return _current_price_instance


# ######
# ###### CacheShortTrendManager: delegated to cachetrend
# ######

CacheShortTrendManager = getattr(_cache_trend, "CacheShortTrendManager", None)
CacheShortTrend = CacheShortTrendManager


_short_trend_instance = None
_short_trend_lock = threading.RLock()

def get_short_trend_manager(symbols=None, filename="cache_short_trend.json", writer=False):
    """Return the CacheShortTrendManager singleton.

    ``writer=True`` lets the process persist the file; calculators and readers use False.
    """
    global _short_trend_instance
    inst = _short_trend_instance
    if inst is not None and not writer:
        return inst
    with _short_trend_lock:
        if _short_trend_instance is not None:
            if writer:
                _short_trend_instance.become_writer()
            return _short_trend_instance
        _syms = symbols if symbols is not None else _default_price_tracking()[0]
        _short_trend_instance = CacheShortTrendManager(_syms, filename, writer=writer)
    return _short_trend_instance


_cache_trend.get_short_trend_manager = get_short_trend_manager


# ######
# ###### GLOBAL VARIABLE FOR CACHE #######
# ######
     
ORDER_SYNC_INTERVAL_SEC = 0.4 * 60   # 24 seconds
TRADE_SYNC_INTERVAL_SEC = 3 * 60     # 3 minutes
SPARSE_PRICE_SYNC_INTERVAL_SEC = 3 * 60     # 3 minutes
PRICE_24_SYNC_INTERVAL_SEC = 30         # Fallback polling while WebSocket is inactive.
PRICE_365_SYNC_INTERVAL_SEC = 30        # Long-archive 24h/365d price sync interval.
CURRENT_PRICE_SYNC_INTERVAL_SEC = 30    # Same policy for CacheCurrentPriceManager.
SHORT_TREND_EVAL_INTERVAL_SEC = (
    CacheShortTrendManager.FULL_EVAL_INTERVAL_SEC
    if CacheShortTrendManager is not None
    else 3.0
)
LONG_TREND_SYNC_INTERVAL_SEC = 10 * 60   # 10 minutes
ASSET_VALUE_SYNC_INTERVAL_SEC = 10 * 60  # 10 minutes
# TODO: set this to 60 * 60  # 1 hour

_PRICE_TREND_CACHES = frozenset({
    "CurrentPrice",
    "24Price",
    "365Price",
    "SparsePrice",
    "LongTrend",
})


class CacheFactory:
    _lock = threading.RLock()
    _instances = {}
    _sync_started = set()

    _CONFIG = {
        "Trade": {
            "class": CacheTradeManager,
            "filename": "cache_trade.jsonl",
            "sync_ts": lambda: TRADE_SYNC_INTERVAL_SEC,
        },
        "Order": {
            "class": CacheOrderManager,
            "filename": "cache_order.json",
            "sync_ts": lambda: ORDER_SYNC_INTERVAL_SEC,
        },
        "SparsePrice": {
            "class": CacheSparsePriceManager,
            "filename": None,  # Dictionary per symbol.
            "sync_ts": lambda: SPARSE_PRICE_SYNC_INTERVAL_SEC,
        },
        "24Price": {
            "class": Cache24PriceManager,
            "filename": None,  # Dictionary per symbol.
            "sync_ts": lambda: PRICE_24_SYNC_INTERVAL_SEC,
        },
        "365Price": {
            "class": Cache365PriceManager,
            "filename": None,  # Dictionary per symbol.
            "sync_ts": lambda: PRICE_365_SYNC_INTERVAL_SEC,
        },
        "CurrentPrice": {
            "class": CacheCurrentPriceManager,
            "filename": "cache_current_price.json",  # One shared file for all symbols.
            "sync_ts": lambda: CURRENT_PRICE_SYNC_INTERVAL_SEC,
        },
        "LongTrend": {
            "class": CacheLongTrendManager,
            "filename": "cache_long_trend.json",
            "sync_ts": lambda: LONG_TREND_SYNC_INTERVAL_SEC,
        },
        "AssetValue": {
            "class": CacheAssetValueManager,
            "filename": "cache_asset_value.jsonl",
            "sync_ts": lambda: ASSET_VALUE_SYNC_INTERVAL_SEC,
        },
    }

    @classmethod
    def get(cls, name, symbols=None, *, start_sync=True, provider_names=None):
        if name == "ShortTrend":
            return get_short_trend_manager(symbols=symbols)
        inst = cls._instances.get(name)
        if (
            inst is not None
            and symbols is None
            and provider_names is None
            and (not start_sync or name in cls._sync_started)
        ):
            return inst
        with cls._lock:
            if name not in cls._CONFIG:
                raise ValueError(f"Unknown cache type: {name}")
            if provider_names is not None and name != "CurrentPrice":
                raise ValueError("provider_names is supported only for CurrentPrice")

            # This name-keyed singleton fixes symbols on first creation. Warn explicitly
            # when a later call requests a different set that will be ignored.
            if name in cls._instances and symbols is not None:
                inst = cls._instances[name]
                existing = set(inst.keys()) if isinstance(inst, dict) else set(getattr(inst, "symbols", []))
                requested = set(symbols)
                if requested != existing:
                    missing = requested - existing
                    # Use builtins.print so this warning remains visible if module logging
                    # replaces print with a no-op.
                    builtins.print(
                        f"[CacheFactory][WARN] '{name}' already exists with symbols {sorted(existing)}; "
                        f"the request for {sorted(requested)} is IGNORED"
                        + (f" (missing: {sorted(missing)})" if missing else "")
                        + ". Singleton per name — the first instance is used.")
            if name == "CurrentPrice" and name in cls._instances:
                cls._instances[name].bind_providers(provider_names)

            created = name not in cls._instances
            if created:
                config = cls._CONFIG[name]
                manager_class = config["class"]
                sync_ts = config["sync_ts"]()
                extra_kwargs = {
                    key: value for key, value in config.items()
                    if key not in {"class", "filename", "sync_ts"}
                }
                if symbols is None:
                    if name == "AssetValue":
                        symbols = ["TOTAL"]
                    elif name in _PRICE_TREND_CACHES:
                        symbols, _, _default_providers = _default_price_tracking()
                        if name == "CurrentPrice" and provider_names is None and _default_providers:
                            provider_names = _default_providers
                    else:
                        symbols = sym.symbols

                if name == "CurrentPrice":
                    current_price_sync_ts = (
                        None if _current_price_instance is not None else sync_ts
                    )
                    instance = get_current_price_manager(
                        symbols=symbols,
                        sync_ts=current_price_sync_ts,
                        start_sync=False,
                        provider_names=provider_names,
                        read_only=not start_sync,
                    )
                    sync_ts = instance.sync_ts
                elif name in ("SparsePrice", "24Price", "365Price"):
                    # SparsePrice and 365Price are append-JSONL history; 24Price is a bounded full-rewrite cache.
                    if name == "SparsePrice":
                        prefix, ext = ("cache_sparse_price_", "jsonl")
                    elif name == "365Price":
                        prefix, ext = ("cache_365_price_", "jsonl")
                    else:
                        prefix, ext = ("cache_24_price_", "json")
                    instance = {
                        s: manager_class(
                            sync_ts=sync_ts,
                            filename=f"{prefix}{s}.{ext}",
                            symbols=[s],
                            api_client=api,
                            read_only=not start_sync,
                            **extra_kwargs,
                        )
                        for s in symbols
                    }
                else:
                    instance = manager_class(
                        sync_ts=sync_ts,
                        filename=config["filename"],
                        symbols=symbols,
                        api_client=api,
                        read_only=not start_sync,
                        **extra_kwargs,
                    )

                # Publish before starting sync so a retry after a partial
                # periodic_sync failure reuses these managers instead of building
                # duplicates. The lock-free fast path still waits for sync when
                # start_sync=True because it also requires name in _sync_started.
                cls._instances[name] = instance

                managers = (instance.values()
                            if isinstance(instance, dict)
                            else (instance,))
                if start_sync:
                    for manager in managers:
                        manager.periodic_sync(sync_ts, False)
                    cls._sync_started.add(name)

            elif start_sync and name not in cls._sync_started:
                value = cls._instances[name]
                managers = value.values() if isinstance(value, dict) else (value,)
                sync_ts = cls._CONFIG[name]["sync_ts"]()
                for manager in managers:
                    manager.periodic_sync(sync_ts, False)
                cls._sync_started.add(name)

            return cls._instances[name]

    @classmethod
    def shutdown_all(cls, timeout=5.0):
        """Stop every factory-owned synchronization loop and clear the registry."""
        global _current_price_instance, _short_trend_instance
        with cls._lock:
            current_price = cls._instances.get("CurrentPrice")
            short_trend = _short_trend_instance
            managers = []
            seen_ids = set()
            for value in cls._instances.values():
                items = value.values() if isinstance(value, dict) else (value,)
                for manager in items:
                    if id(manager) not in seen_ids:
                        seen_ids.add(id(manager))
                        managers.append(manager)
            if short_trend is not None and id(short_trend) not in seen_ids:
                seen_ids.add(id(short_trend))
                managers.append(short_trend)
            cls._instances = {}
            cls._sync_started = set()
            results = [manager.shutdown(timeout=timeout) for manager in managers
                       if hasattr(manager, "shutdown")]
            if (
                current_price is _current_price_instance
                and getattr(current_price, "thread", None) is None
            ):
                _current_price_instance = None
            if (
                short_trend is _short_trend_instance
                and getattr(short_trend, "_full_eval_thread", None) is None
                and getattr(short_trend, "_flush_thread", None) is None
                and getattr(short_trend, "thread", None) is None
            ):
                _short_trend_instance = None
            return all(results) if results else True

    @classmethod
    def remove(cls, name, timeout=5.0):
        """Remove a singleton cleanly after stopping all threads it owns."""
        global _current_price_instance, _short_trend_instance
        with cls._lock:
            if name == "ShortTrend":
                inst = _short_trend_instance
                _short_trend_instance = None
                if inst is not None and hasattr(inst, "shutdown"):
                    return bool(inst.shutdown(timeout=timeout))
                return True
            value = cls._instances.pop(name, None)
            cls._sync_started.discard(name)
            if value is None:
                return True
            managers = value.values() if isinstance(value, dict) else (value,)
            results = [manager.shutdown(timeout=timeout) for manager in managers
                       if hasattr(manager, "shutdown")]
            stopped = all(results) if results else True
            if name == "CurrentPrice" and value is _current_price_instance and stopped:
                _current_price_instance = None
            return stopped
        
def get_cache_manager(name, symbols=None, *, start_sync=True):
    return CacheFactory.get(name, symbols, start_sync=start_sync)


def get_trade_manager(symbols=None, *, start_sync=True):
    return CacheFactory.get("Trade", symbols, start_sync=start_sync)


def get_order_manager(symbols=None, *, start_sync=True):
    return CacheFactory.get("Order", symbols, start_sync=start_sync)


def get_sparse_price_managers(symbols=None, *, start_sync=True):
    return CacheFactory.get("SparsePrice", symbols, start_sync=start_sync)


def get_24_price_managers(symbols=None, *, start_sync=True):
    return CacheFactory.get("24Price", symbols, start_sync=start_sync)


def get_365_price_managers(symbols=None, *, start_sync=True):
    return CacheFactory.get("365Price", symbols, start_sync=start_sync)


def get_long_trend_manager(symbols=None, *, start_sync=True):
    return CacheFactory.get("LongTrend", symbols, start_sync=start_sync)


def get_asset_value_manager(symbols=None, *, start_sync=True):
    return CacheFactory.get("AssetValue", symbols, start_sync=start_sync)


def ensure_account_cache_readers(status):
    """Synchronize this process with both versions in a fresh health marker."""
    try:
        order_version = status.order_cache_version
        trade_version = status.trade_cache_version
        if not order_version or not trade_version:
            raise ValueError("health marker has no durable cache versions")
        get_cache_manager("Order", start_sync=False).ensure_persisted_version(
            order_version
        )
        get_cache_manager("Trade", start_sync=False).ensure_persisted_version(
            trade_version
        )
    except account_cache_health.AccountCacheNotReady:
        raise
    except (AttributeError, TypeError, ValueError) as exc:
        raise account_cache_health.AccountCacheNotReady(
            "account_cache_reader_not_current"
        ) from exc


# ######
# ###### Real-time Binance user stream -> cache actions (delegated to binance_api.bapi_ws_bridge)
# ######
_ws_bridge = _bapi_ws_bridge._ws_bridge
_ws_bridge_lock = _bapi_ws_bridge._ws_bridge_lock
_ws_event_stats = _bapi_ws_bridge._ws_event_stats


def _upsert_order_from_execution_report(event):
    _bapi_ws_bridge.upsert_order_from_execution_report(
        event, cache_getter=get_cache_manager
    )


def _append_trade_from_execution_report(event):
    _bapi_ws_bridge.append_trade_from_execution_report(
        event, cache_getter=get_cache_manager
    )


def _refresh_asset_value_from_ws_event():
    _bapi_ws_bridge.refresh_asset_value_from_ws_event(
        cache_getter=get_cache_manager
    )


def _persist_ws_updated_caches(event_type):
    _bapi_ws_bridge.persist_ws_updated_caches(
        event_type, cache_getter=get_cache_manager
    )


_refresh_symbol_in_cache = _bapi_ws_bridge.refresh_symbol_in_cache


def _handle_binance_ws_event(event):
    _bapi_ws_bridge.handle_binance_ws_event(
        event,
        cache_getter=get_cache_manager,
        log_enabled=WS_EVENT_LOG_ENABLED,
        upsert_order_fn=_upsert_order_from_execution_report,
        append_trade_fn=_append_trade_from_execution_report,
        persist_fn=_persist_ws_updated_caches,
    )


def enable_real_ws_event_sync():
    global _ws_bridge
    _bapi_ws_bridge._ws_bridge = _ws_bridge
    _ws_bridge = _bapi_ws_bridge.enable_real_ws_event_sync(
        on_event=_handle_binance_ws_event,
        on_available=_mark_ws_available,
        on_healthy=_mark_ws_event_received,
        on_unhealthy=_mark_ws_unhealthy,
        loss_timeout_sec=WS_LOSS_TIMEOUT_SEC,
    )
    return _ws_bridge


def _initialize_once():
    _bapi_ws_bridge.initialize_ws_bridge_once(
        enable_fn=enable_real_ws_event_sync
    )

# The user-data WebSocket bridge is opt-in and does not start during import. Processes
# that need real-time execution reports enable it explicitly:
#     import cacheManager as cm
#     cm.enable_real_ws_event_sync()   # or cm._initialize_once()
# Read-only consumers rely on polling and do not start WebSocket.


_NB_FETCH_DEADLINE_SEC = getattr(_cache_trend, "_NB_FETCH_DEADLINE_SEC", 8)
_fetch_price_with_deadline = getattr(_cache_trend, "_fetch_price_with_deadline", None)
_NonBinanceTrendPoller = getattr(_cache_trend, "_NonBinanceTrendPoller", None)
_start_nonbinance_trend_poller = getattr(_cache_trend, "_start_nonbinance_trend_poller", None)


if __name__ == "__main__":
    single_instance("cacheManager")
    account_cache_health.enable_writer()
    # Construct both authoritative account caches before the user-data stream can
    # deliver events. Their first periodic loop is then started exactly once with
    # persistence enabled, so readiness never depends on a thread-scheduling race.
    for _account_cache_name in ("Order", "Trade"):
        get_cache_manager(_account_cache_name, start_sync=False)
        CacheFactory._sync_started.add(_account_cache_name)
    _initialize_once()   # The dedicated cache process enables WebSocket and persistence.
    threads = []
    _nb_poller = None
    _trend_mgr = None

    _trend_syms, _nb_syms, _nb_provider_names = _default_price_tracking()
    if _nb_syms:
        builtins.print(f"[cacheManager] tracking non-Binance instruments from instruments.conf: {_nb_syms}")

    try:
        from binance_api import bapi_ws
        get_current_price_manager(
            ws_manager=bapi_ws.get_ws_manager(),
            sync_ts=0.8,
            start_sync=False,
        )
    except Exception as _ws_init_err:
        print(f"[cacheManager] Warning pre-attaching price WS: {_ws_init_err}")

    for name, config in CacheFactory._CONFIG.items():
        if name == "365Price":
            continue
        cache = get_cache_manager(name, start_sync=False)
        interval = (
            cache.sync_ts
            if name == "CurrentPrice" and hasattr(cache, "sync_ts")
            else config["sync_ts"]()
        )

        if isinstance(cache, dict):
            # SparsePrice and 24Price are dictionaries keyed by symbol.
            for manager in cache.values():
                threads.append(manager.periodic_sync(interval))
        else:
            threads.append(cache.periodic_sync(interval))
        CacheFactory._sync_started.add(name)

    # Trend chain: market-data WebSocket to CurrentPrice, Cache24, and ShortTrend.
    # Full calculation here keeps the shared snapshot current independently of tradeall.
    try:
        from binance_api import bapi_ws
        _trend_cpm = get_current_price_manager(
            ws_manager=bapi_ws.get_ws_manager(),
            sync_ts=0.8,
        )
        _trend_cache24 = CacheFactory.get("24Price")
        _trend_mgr = get_short_trend_manager(writer=True)   # Sole file writer.
        _trend_mgr.start_computation(_trend_cache24, _trend_cpm, run_full_eval=True)
        print("⚙️ cacheManager: full trend computation started (cache_short_trend.json).")
        if _nb_syms:   # WebSocket covers Binance only; push other prices manually.
            _nb_poller = _start_nonbinance_trend_poller(
                _trend_cpm,
                _nb_syms,
                interval_sec=10,
                provider_names=_nb_provider_names,
            )
    except Exception as e:
        print(f"[cacheManager] Cannot start the trend computation: {e}")

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("Stopped manually.")
    finally:
        account_cache_health.disable_writer()
        print("Cleanup / releasing resources...")
        if _nb_poller is not None:
            _nb_poller.stop()
        if _trend_mgr is not None:
            _trend_mgr.shutdown()
        CacheFactory.shutdown_all()
        if _current_price_instance is not None:
            _current_price_instance.shutdown()
        if _ws_bridge is not None:
            _ws_bridge.stop()
        from binance_api import bapi_client
        bapi_client.stop_periodic_resync()
