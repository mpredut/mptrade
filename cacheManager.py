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
    atomic_snapshot_file,
    atomic_text_writer,
    atomic_write_json as _atomic_write_json,
    durable_replace_file,
)
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

# Load versioned, non-secret tuning parameters using the same pattern as the other
# component env files. ``botcore.load_dotenv`` never overwrites variables already set
# in the real environment; it only fills missing values.
from botcore import (
    load_dotenv as _load_dotenv,
    required_bool_env,
    required_float_env,
    required_int_env,
    single_instance,
)
_CONFIG_ROOT = os.path.dirname(os.path.abspath(__file__))
_load_dotenv(os.path.join(_CONFIG_ROOT, "config.env"))

# Dynamic-window bounds for ``get_instant_trend_for_window``. Values below the minimum
# provide too few samples for a meaningful slope; values above the maximum add unjustified
# cost and exceed the 24-hour history retained by Cache24PriceManager.
CM_DYNAMIC_WINDOW_MIN_SEC = required_float_env("CM_DYNAMIC_WINDOW_MIN_SEC")
CM_DYNAMIC_WINDOW_MAX_SEC = required_float_env("CM_DYNAMIC_WINDOW_MAX_SEC")
LONGTREND_NONBINANCE = required_bool_env("LONGTREND_NONBINANCE")
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

# WS-only mode: when True, polling for Order/Trade/AssetValue is paused while WS is healthy.
WS_ONLY_MODE = False
WS_LOSS_TIMEOUT_SEC = 40 # 600  # 10 minute
WS_EVENT_LOG_ENABLED = True

_ws_health_lock = threading.Lock()
_ws_available = False
_ws_last_event_ts = 0.0
_ws_is_healthy = False

_CACHE_META_SCHEMA_VERSION = 2
_ACCOUNT_CACHE_CLASSES = {"CacheOrderManager", "CacheTradeManager"}


def _mark_ws_available(value):
    global _ws_available
    with _ws_health_lock:
        _ws_available = value


def _mark_ws_event_received():
    global _ws_last_event_ts, _ws_is_healthy
    with _ws_health_lock:
        _ws_last_event_ts = time.time()
        _ws_is_healthy = True


def _mark_ws_unhealthy():
    builtins.print("[cacheManager][WS] marked unhealthy")
    global _ws_is_healthy
    with _ws_health_lock:
        _ws_is_healthy = False


def _should_poll_for_manager(cls_name):
    if not WS_ONLY_MODE:
        return True
    ws_managed_classes = {"CacheOrderManager", "CacheTradeManager", "CacheAssetValueManager"}
    if cls_name not in ws_managed_classes:
        return True
    with _ws_health_lock:
        return (not _ws_available) or (not _ws_is_healthy)

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
                 append_persist=False):
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
        self.fetchtime_time_per_symbol = {}

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
    

    #def get_all_symbols_from_cache(self):
    #    return list(set(t.get("symbol") for t in self.cache if "symbol" in t))
    def get_all_symbols_from_cache(self):
        with self.lock:
            return list(self.cache.keys())       
        
    def rebuild_fetchtime_times(self):
        """Allow subclasses to provide custom fetch-time reconstruction.

        Returning None lets ``__rebuild_fetchtime_times`` infer timestamps generically
        from ``time``/``timestamp`` dictionaries or ``[timestamp, ...]`` lists.
        """
        return None
    
    
    def __rebuild_fetchtime_times(self):
        last_times_per_sym = self.rebuild_fetchtime_times()
        if not last_times_per_sym:
            last_times_per_sym = defaultdict(int)
            for symbol, trades in self.cache.items():
                for trade in trades:
                    # Use ``time`` or ``timestamp`` and fall back to zero.
                    #time_ = trade.get("time") or trade.get("timestamp") or 0
                    if isinstance(trade, dict):
                        time_ = trade.get("time") or trade.get("timestamp") or 0
                    elif isinstance(trade, list) and len(trade) > 0:
                        time_ = trade[0]  # ``[timestamp_ms, price]`` format.
                    else:
                        time_ = 0
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
        print(f"[{self.cls_name}][Info] Load state from {self.filename} ...")
        if self.append_persist:
            self._migrate_legacy_json_if_needed()
            self._load_jsonl()
            if not self.cache:
                self.query_remote_and_update_cache()
                self.save_state_to_file_if_enabled()
            return
        if os.path.exists(self.filename):
            try:
                with open(self.filename, "r") as f:
                    data = json.load(f)
                    with self.lock:
                        self.cache = data.get("items", {})
                        if not isinstance(self.cache, dict):
                            # Convert the legacy list format to a dictionary.
                            self.cache = {sym: item for sym, item in zip(self.symbols, self.cache)}
                            print(f"[{self.cls_name}][warning] self.cache is not Dict!!!!")    
                        
                        self.fetchtime_time_per_symbol = data.get("fetchtime", {})
                        if not self.cache:
                            print(f"[{self.cls_name}][warning] cache is None")
                        if not self.fetchtime_time_per_symbol:
                            print(f"[{self.cls_name}][warning] fetchtime_time_per_symbol is None")    
                    
            except Exception as e:
                print(f"[{self.cls_name}][Error] While reading the cache file {self.filename} : {e}")
                self.query_remote_and_update_cache()
                self.save_state_to_file_if_enabled()
        else :
            print(f"[{self.cls_name}][Info] File is missing, may be is it first time run. Creating it ....")
            self.query_remote_and_update_cache()
            self.save_state_to_file_if_enabled()


    # ── Memory/disk resilience: freshness, overwrite guard, and resync ────────
    def _mem_max_ts(self):
        """Return memory freshness as the latest fetch time in milliseconds."""
        if not self.fetchtime_time_per_symbol:
            return 0
        try:
            return max(self.fetchtime_time_per_symbol.values())
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

    @staticmethod
    def _snapshot_data_version(items, fetchtime):
        """Return a stable version for an exact atomic JSON snapshot."""
        canonical = json.dumps(
            {"items": items, "fetchtime": fetchtime},
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return "snapshot:" + hashlib.blake2s(
            canonical, digest_size=16
        ).hexdigest()

    @staticmethod
    def _trade_data_version(stream_id, revision):
        return f"{stream_id}:{int(revision)}"

    @staticmethod
    def _trade_records_digest(records, initial_digest=""):
        """Extend a deterministic digest chain with canonical Trade rows."""
        if initial_digest:
            if (
                not isinstance(initial_digest, str)
                or not initial_digest.startswith("chain:")
            ):
                raise ValueError("invalid Trade content digest")
            try:
                state = bytes.fromhex(initial_digest.removeprefix("chain:"))
            except ValueError as exc:
                raise ValueError("invalid Trade content digest") from exc
            if len(state) != 16:
                raise ValueError("invalid Trade content digest")
        else:
            state = bytes(16)
        for symbol, item in records:
            row = json.dumps(
                {"s": symbol, "i": item},
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8") + b"\n"
            state = hashlib.blake2s(
                state + row, digest_size=16
            ).digest()
        return "chain:" + state.hex()

    def _account_meta_snapshot(self, *, data_version, fetchtime, counts,
                               stream_id="", revision=0,
                               committed_bytes=None,
                               content_digest=None):
        payload = {
            "schema_version": _CACHE_META_SCHEMA_VERSION,
            "data_version": data_version,
            "fetchtime": dict(fetchtime),
            "counts": dict(counts),
        }
        if committed_bytes is not None:
            payload.update({
                "stream_id": stream_id,
                "revision": int(revision),
                "committed_bytes": int(committed_bytes),
                "content_digest": content_digest,
            })
        return payload

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

    @staticmethod
    def _trade_semantic_signature(item):
        """Normalize the immutable financial fields shared by REST and WebSocket."""
        return (
            str(item["symbol"]).upper(),
            int(item["id"]),
            int(item["orderId"]),
            Decimal(str(item["price"])).normalize(),
            Decimal(str(item["qty"])).normalize(),
            int(item["time"]),
            item["isBuyer"],
        )

    @classmethod
    def _dedupe_trade_items_by_id(cls, items):
        """Collapse equivalent Trade IDs and reject conflicting immutable fills."""
        seen = {}
        deduped = []
        for item in items:
            trade_id = str(int(item["id"]))
            signature = cls._trade_semantic_signature(item)
            if trade_id in seen:
                if seen[trade_id] != signature:
                    raise ValueError(
                        f"conflicting Trade rows for immutable ID {trade_id}"
                    )
                continue
            seen[trade_id] = signature
            deduped.append(item)
        return deduped

    def _validate_account_rows_locked(self):
        """Reject any financial row that strict readers could not accept."""
        for symbol, items in self.cache.items():
            if not isinstance(symbol, str) or not isinstance(items, list):
                raise ValueError(f"invalid {self.cls_name} cache bucket")
            order_ids = set()
            fill_owners = {}
            for item in items:
                if not self._is_valid_trade(item):
                    raise ValueError(f"invalid {self.cls_name} cache row")
                if (
                    self.cls_name == "CacheTradeManager"
                    and str(item["symbol"]).upper() != symbol.upper()
                ):
                    raise ValueError("Trade row symbol differs from its bucket")
                if self.cls_name == "CacheOrderManager":
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
                self.cls_name == "CacheTradeManager"
                and len(self._dedupe_trade_items_by_id(items)) != len(items)
            ):
                raise ValueError("duplicate immutable Trade ID")

    def _account_reader_reason(self):
        prefix = "trade" if self.cls_name == "CacheTradeManager" else "order"
        return f"{prefix}_cache_reader_not_current"

    def _write_meta(self, snapshot):
        """Atomically write an explicit snapshot matching durable cache data."""
        try:
            payload = dict(snapshot)
            fetchtime = payload.get("fetchtime", {})
            payload["max_ts"] = max(fetchtime.values()) if fetchtime else 0
            payload["saved_at"] = int(time.time() * 1000)
            atomic_write_json(self.filename + ".meta", payload)
            return True
        except Exception as e:
            print(f"[{self.cls_name}][Error] metadata {self.filename}: {e}")
            return False

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
            if self.append_persist and self.cls_name == "CacheTradeManager":
                return self._save_account_trade_append()
            with self.lock:
                if self._persisted_max_ts() > self._mem_max_ts():
                    builtins.print(f"[{self.cls_name}][resync] file newer than memory -> "
                                   f"refusing to overwrite with stale data ({self.filename})")
                    return False
                if self.append_persist:
                    return self._save_jsonl_append()
                fetchtime = dict(self.fetchtime_time_per_symbol)
                counts = {symbol: len(items) for symbol, items in self.cache.items()}
                data_version = ""
                payload = {
                    "items": self.cache,
                    "fetchtime": fetchtime,
                }
                metadata = {
                    "fetchtime": fetchtime,
                    "counts": counts,
                }
                if self._is_account_cache():
                    self._validate_account_rows_locked()
                    data_version = self._snapshot_data_version(
                        self.cache, fetchtime
                    )
                    payload["data_version"] = data_version
                    metadata = self._account_meta_snapshot(
                        data_version=data_version,
                        fetchtime=fetchtime,
                        counts=counts,
                    )
                atomic_write_json(self.filename,
                                  payload,
                                  indent=1)
                if not self._write_meta(metadata):
                    return False
                if data_version:
                    self._remember_account_commit_locked(metadata)
                print(f"[{self.cls_name}][info] Save cache to file {self.filename}")
                return True
        except Exception as e:
            print(f"[{self.cls_name}][Error] While saving the cache file {self.filename} / .tmp : {e}")
            return False

    def _reload_from_disk(self):
        """Reload the cache when another process has written a newer file."""
        if self.append_persist:
            self._load_jsonl()
            return
        if os.path.exists(self.filename):
            try:
                with open(self.filename) as f:
                    data = json.load(f)
                with self.lock:
                    items = data.get("items", {})
                    if isinstance(items, dict):
                        self.cache = items
                    self.fetchtime_time_per_symbol = data.get("fetchtime", {})
                    self._mark_account_cache_dirty_locked()
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

    # ── JSONL append persistence for append-only caches ──────────────────────
    def _migrate_legacy_json_if_needed(self):
        """Create a JSONL cache from its legacy full-JSON sibling once.

        Trade migration decides under the generation lock so concurrent startup
        cannot replace a generation another manager has already published.
        """
        if self.cls_name == "CacheTradeManager":
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
                self.fetchtime_time_per_symbol = payload.get("fetchtime", {})
                self._persisted_counts = {}
            migrated = (
                self._compact_account_trade_jsonl_generation_locked()
                if self.cls_name == "CacheTradeManager"
                else self.compact_jsonl()
            )
            if not migrated:
                raise ValueError(
                    "legacy cache could not be converted into a durable JSONL generation"
                )
            if self.cls_name == "CacheTradeManager":
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
        if self.cls_name == "CacheTradeManager":
            return self._save_account_trade_append()

        previous_counts = {}
        candidate_counts = {}
        initial_size = None
        try:
            with self.lock:
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
                fetchtime = dict(self.fetchtime_time_per_symbol)
                self._persisted_counts = candidate_counts
                return self._write_meta({
                    "fetchtime": fetchtime,
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
                                not self._is_valid_trade(item)
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
                fetchtime = dict(self.fetchtime_time_per_symbol)
                metadata = self._account_meta_snapshot(
                    data_version=version,
                    stream_id=stream_id,
                    revision=revision,
                    committed_bytes=committed_bytes,
                    content_digest=content_digest,
                    fetchtime=fetchtime,
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
            with open(self.filename, "r+b") as rollback:
                rollback.truncate(initial_size)
                rollback.flush()
                os.fsync(rollback.fileno())
        except Exception as rollback_error:
            builtins.print(
                f"[{self.cls_name}][Error] could not roll back failed JSONL append "
                f"for {self.filename}: {rollback_error}"
            )

    def _load_versioned_account_meta(self, *, optional=False,
                                     required_version=None):
        """Load and validate the small durable account-cache manifest."""
        try:
            with open(self.filename + ".meta", "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except (OSError, ValueError, TypeError) as exc:
            if optional:
                return None
            raise account_cache_health.AccountCacheNotReady(
                self._account_reader_reason()
            ) from exc

        if not isinstance(metadata, dict):
            if optional:
                return None
            raise account_cache_health.AccountCacheNotReady(
                self._account_reader_reason()
            )
        if metadata.get("schema_version") != _CACHE_META_SCHEMA_VERSION:
            if optional:
                return None
            raise account_cache_health.AccountCacheNotReady(
                self._account_reader_reason()
            )
        if self.cls_name == "CacheTradeManager":
            content_digest = metadata.get("content_digest")
            try:
                if (
                    not isinstance(content_digest, str)
                    or not content_digest.startswith("chain:")
                ):
                    raise ValueError("missing Trade content digest")
                self._trade_records_digest([], content_digest)
            except ValueError as exc:
                if optional:
                    return None
                raise account_cache_health.AccountCacheNotReady(
                    self._account_reader_reason()
                ) from exc
        try:
            version = metadata["data_version"]
            fetchtime = metadata["fetchtime"]
            counts = metadata["counts"]
            if not isinstance(version, str) or not version:
                raise ValueError("missing data version")
            if required_version is not None and version != required_version:
                raise ValueError("cache marker and manifest versions differ")
            if not isinstance(fetchtime, dict) or not isinstance(counts, dict):
                raise ValueError("invalid cache manifest maps")
            normalized_counts = {}
            for symbol, count in counts.items():
                if (not isinstance(symbol, str) or isinstance(count, bool)
                        or not isinstance(count, int) or count < 0):
                    raise ValueError("invalid cache manifest count")
                normalized_counts[symbol] = count
            metadata["counts"] = normalized_counts
            if self.cls_name == "CacheTradeManager":
                stream_id = metadata["stream_id"]
                revision = metadata["revision"]
                committed_bytes = metadata["committed_bytes"]
                content_digest = metadata["content_digest"]
                if not isinstance(stream_id, str) or not stream_id:
                    raise ValueError("missing Trade stream ID")
                if (isinstance(revision, bool) or not isinstance(revision, int)
                        or revision <= 0):
                    raise ValueError("invalid Trade revision")
                if (isinstance(committed_bytes, bool)
                        or not isinstance(committed_bytes, int)
                        or committed_bytes < 0):
                    raise ValueError("invalid Trade committed boundary")
                if version != self._trade_data_version(stream_id, revision):
                    raise ValueError("invalid Trade data version")
            return metadata
        except (KeyError, TypeError, ValueError) as exc:
            raise account_cache_health.AccountCacheNotReady(
                self._account_reader_reason()
            ) from exc

    def _read_trade_records_strict(self, start, end, *, path=None):
        """Parse only the requested committed JSONL byte range."""
        if end < start:
            raise ValueError("Trade committed boundary moved backwards")
        try:
            with open(path or self.filename, "rb") as handle:
                file_size = os.fstat(handle.fileno()).st_size
                if file_size < end:
                    raise ValueError("Trade cache is shorter than its manifest")
                handle.seek(start)
                raw = handle.read(end - start)
            if raw and not raw.endswith(b"\n"):
                raise ValueError("Trade committed data ends with a partial row")
            records = []
            for line in raw.splitlines():
                if not line.strip():
                    continue
                record = json.loads(line.decode("utf-8"))
                if (not isinstance(record, dict)
                        or not isinstance(record.get("s"), str)
                        or not isinstance(record.get("i"), dict)
                        or not self._is_valid_trade(record["i"])
                        or str(record["i"]["symbol"]).upper()
                        != str(record["s"]).upper()
                ):
                    raise ValueError("invalid Trade cache row")
                records.append((record["s"], record["i"]))
            return records
        except (OSError, UnicodeError, TypeError, ValueError,
                json.JSONDecodeError) as exc:
            raise account_cache_health.AccountCacheNotReady(
                self._account_reader_reason()
            ) from exc

    @staticmethod
    def _counts_for_records(records):
        counts = defaultdict(int)
        for symbol, _item in records:
            counts[symbol] += 1
        return dict(counts)

    @staticmethod
    def _nonzero_counts(counts):
        return {symbol: count for symbol, count in counts.items() if count}

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
            if self.cls_name == "CacheTradeManager":
                self._ensure_trade_version(required_version)
            else:
                self._ensure_order_version(required_version)

    def _ensure_order_version(self, required_version):
        reason = self._account_reader_reason()
        try:
            metadata = self._load_versioned_account_meta(
                required_version=required_version
            )
            with open(self.filename, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                raise ValueError("Order cache root is not an object")
            items = payload.get("items")
            fetchtime = payload.get("fetchtime")
            if not isinstance(items, dict) or not isinstance(fetchtime, dict):
                raise ValueError("invalid Order cache maps")
            if payload.get("data_version") != required_version:
                raise ValueError("Order cache and marker versions differ")
            for symbol, orders in items.items():
                if (not isinstance(symbol, str) or not isinstance(orders, list)
                        or any(
                            not self._is_valid_trade(order)
                            for order in orders
                        )):
                    raise ValueError("invalid Order cache items")
            actual_counts = {
                symbol: len(orders) for symbol, orders in items.items()
            }
            if actual_counts != metadata["counts"]:
                raise ValueError("Order cache counts differ from its manifest")
            if self._snapshot_data_version(items, fetchtime) != required_version:
                raise ValueError("Order cache digest is invalid")
            after = self._load_versioned_account_meta(
                required_version=required_version
            )
            if after != metadata:
                raise ValueError("Order manifest changed during reload")
            with self.lock:
                self.cache = items
                self.fetchtime_time_per_symbol = dict(fetchtime)
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
            else:
                candidate_cache, candidate_counts = (
                    self._read_trade_snapshot_strict(metadata)
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
                        self.fetchtime_time_per_symbol = dict(metadata["fetchtime"])
                        self._persisted_counts = dict(candidate_counts)
                        self._remember_loaded_trade_manifest_locked(
                            metadata, candidate_counts
                        )
                        return
                candidate_cache, candidate_counts = (
                    self._read_trade_snapshot_strict(metadata)
                )
                after = self._load_versioned_account_meta(
                    required_version=required_version
                )
                if after != metadata:
                    raise ValueError("Trade manifest changed during full reload")

            with self.lock:
                self.cache = candidate_cache
                self.fetchtime_time_per_symbol = dict(metadata["fetchtime"])
                self._persisted_counts = dict(candidate_counts)
                self._remember_loaded_trade_manifest_locked(
                    metadata, candidate_counts
                )
        except account_cache_health.AccountCacheNotReady:
            raise
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise account_cache_health.AccountCacheNotReady(reason) from exc

    def _trade_previous_generation_path(self):
        return self.filename + ".previous"

    def _trade_generation_lock(self):
        """Serialize Trade data/manifest publication and canonical recovery."""
        os.makedirs(os.path.dirname(os.path.abspath(self.filename)), exist_ok=True)
        return FileLock(self.filename + ".generation.lock")

    def _stage_previous_trade_generation(self):
        """Keep the currently certified Trade data until the new manifest commits."""
        backup_path = self._trade_previous_generation_path()
        atomic_snapshot_file(self.filename, backup_path)
        return backup_path

    def _restore_previous_trade_generation(self, backup_path):
        """Atomically restore data matching the still-current Trade manifest."""
        durable_replace_file(backup_path, self.filename)

    @staticmethod
    def _discard_trade_generation(path):
        try:
            os.remove(path)
        except OSError:
            pass

    def _load_jsonl(self):
        """Load JSONL under a lock when canonical recovery is possible."""
        if self.cls_name == "CacheTradeManager":
            with self._trade_generation_lock():
                return self._load_jsonl_generation_locked()
        return self._load_jsonl_generation_locked()

    def _load_jsonl_generation_locked(self):
        """Load JSONL after any required Trade generation lock is held."""
        if self.cls_name == "CacheTradeManager":
            metadata = self._load_versioned_account_meta(optional=True)
            if metadata is not None:
                backup_path = self._trade_previous_generation_path()
                try:
                    cache, counts = self._read_trade_snapshot_strict(metadata)
                except account_cache_health.AccountCacheNotReady as current_error:
                    if not os.path.exists(backup_path):
                        raise
                    try:
                        cache, counts = self._read_trade_snapshot_strict(
                            metadata, path=backup_path
                        )
                        self._restore_previous_trade_generation(backup_path)
                    except (account_cache_health.AccountCacheNotReady, OSError):
                        raise current_error
                else:
                    self._discard_trade_generation(backup_path)
                with self.lock:
                    self.cache = cache
                    self.fetchtime_time_per_symbol = dict(metadata["fetchtime"])
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
            if os.path.exists(self.filename):
                with open(self.filename, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            if self.cls_name == "CacheTradeManager":
                                if (
                                    not isinstance(rec, dict)
                                    or not isinstance(rec.get("s"), str)
                                    or not isinstance(rec.get("i"), dict)
                                    or not self._is_valid_trade(rec["i"])
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
                            if self.cls_name == "CacheTradeManager":
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
                or self.cls_name == "CacheTradeManager"
            )
            self._persisted_counts = {s: len(v) for s, v in self.cache.items()}
            fetchtime = legacy_metadata.get("fetchtime")
            if isinstance(fetchtime, dict):
                self.fetchtime_time_per_symbol = fetchtime

    def compact_jsonl(self):
        """Rewrite JSONL from memory after full in-memory and on-disk deduplication.

        Perform this expensive complete pass periodically rather than on every update.
        """
        if not self.append_persist:
            return False
        if self.cls_name == "CacheTradeManager":
            return self._compact_account_trade_jsonl()

        try:
            with self.lock:
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
                fetchtime = dict(self.fetchtime_time_per_symbol)
                with atomic_write(self.filename) as f:
                    for symbol, items in self.cache.items():
                        for item in items:
                            f.write(json.dumps(
                                {"s": symbol, "i": item}, separators=(",", ":"),
                            ) + "\n")
                self._persisted_counts = candidate_counts
                return self._write_meta({
                    "fetchtime": fetchtime,
                    "counts": dict(candidate_counts),
                })
        except Exception as e:
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
        backup_path = None
        data_replaced = False
        try:
            with self.lock:
                previous_metadata = self._load_versioned_account_meta(
                    optional=True
                )
                if previous_metadata is not None:
                    self._read_trade_snapshot_strict(previous_metadata)
                    backup_path = self._stage_previous_trade_generation()
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
                            not self._is_valid_trade(item)
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
                fetchtime = dict(self.fetchtime_time_per_symbol)
                records = []
                with atomic_write(self.filename) as handle:
                    for symbol, items in self.cache.items():
                        for item in items:
                            records.append((symbol, item))
                            handle.write(json.dumps(
                                {"s": symbol, "i": item},
                                separators=(",", ":"),
                            ) + "\n")
                data_replaced = True
                committed_bytes = os.path.getsize(self.filename)
                content_digest = self._trade_records_digest(records)
                stream_id = uuid.uuid4().hex
                revision = 1
                version = self._trade_data_version(stream_id, revision)
                metadata = self._account_meta_snapshot(
                    data_version=version, stream_id=stream_id,
                    revision=revision, committed_bytes=committed_bytes,
                    content_digest=content_digest,
                    fetchtime=fetchtime, counts=counts,
                )
                if not self._write_meta(metadata):
                    if backup_path is not None:
                        self._restore_previous_trade_generation(backup_path)
                        backup_path = None
                        data_replaced = False
                    self._legacy_jsonl_needs_rewrite = True
                    return False

                if backup_path is not None:
                    self._discard_trade_generation(backup_path)
                    backup_path = None
                self._persisted_counts = counts
                self._legacy_jsonl_needs_rewrite = False
                self._remember_account_commit_locked(metadata)
                return True
        except Exception as exc:
            if (
                data_replaced
                and backup_path is not None
                and os.path.exists(backup_path)
            ):
                try:
                    self._restore_previous_trade_generation(backup_path)
                except OSError as rollback_error:
                    builtins.print(
                        f"[{self.cls_name}][Error] could not restore the prior "
                        f"Trade generation: {rollback_error}"
                    )
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
            value = item.get("time") or item.get("timestamp") or 0
        elif isinstance(item, (list, tuple)) and item:
            value = item[0]
        else:
            return 0
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            return 0
        # AssetValue uses Unix seconds; exchange fills and price caches use milliseconds.
        return int(value * 1000) if 0 < value < 100_000_000_000 else int(value)

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
            for symbol, items in list(self.cache.items()):
                kept = [it for it in items if self._entry_timestamp_ms(it) >= cutoff_ms]
                if len(kept) != len(items):
                    self.cache[symbol] = kept
                    changed = True
            if changed:
                self._mark_account_cache_dirty_locked()
        if changed:
            builtins.print(f"[{self.cls_name}][maintain] pruning >{self.RETENTION_DAYS}d from {self.filename}")
            if self.append_persist:
                self.compact_jsonl()
            else:
                self.save_state_to_file()   # A full rewrite is normal for these classes.
        # Rotate by size only for JSONL; see the docstring.
        if not self.append_persist:
            return
        try:
            if os.path.exists(self.filename) and os.path.getsize(self.filename) > self.MAX_FILE_BYTES:
                self._rotate_keep_latest()
        except OSError:
            pass

    def _rotate_keep_latest(self):
        """Archive the current file and retain each symbol's latest configured fraction."""
        if self.cls_name == "CacheTradeManager":
            return self._rotate_account_trade_keep_latest()
        with self.lock:
            archive = f"{self.filename}.{int(time.time())}.archive"
            try:
                os.replace(self.filename, archive)   # Move complete history into the archive.
            except OSError as e:
                builtins.print(f"[{self.cls_name}][maintain] archiving failed: {e}")
                return
            for symbol, items in self.cache.items():
                keep_n = max(1, int(len(items) * self.ROTATE_KEEP_FRACTION))
                self.cache[symbol] = items[-keep_n:]
            self._persisted_counts = {}
            self.compact_jsonl()   # Rewrite the current file with retained entries only.
            prefix = f"{self.filename}."
            compressed_archive = archive + ".gz"
            compressed_tmp = compressed_archive + ".tmp"
            try:
                with open(archive, "rb") as source, gzip.open(compressed_tmp, "wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                os.replace(compressed_tmp, compressed_archive)
                os.remove(archive)
                archive = compressed_archive
            except OSError as e:
                try:
                    os.remove(compressed_tmp)
                except OSError:
                    pass
                builtins.print(f"[{self.cls_name}][maintain] archive compression failed: {e}")
            archives = sorted(
                (path for path in glob.glob(prefix + "*.archive*") if not path.endswith(".tmp")),
                key=lambda path: os.path.getmtime(path), reverse=True)
            for old_archive in archives[self.ROTATE_ARCHIVE_COUNT:]:
                try:
                    os.remove(old_archive)
                except OSError:
                    pass
        builtins.print(f"[{self.cls_name}][maintain] ROTATION: archived -> {archive}, "
                       f"kept the last {int(self.ROTATE_KEEP_FRACTION*100)}%")

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

            compressed_archive = archive + ".gz"
            compressed_tmp = compressed_archive + ".tmp"
            try:
                with (
                    open(archive, "rb") as source,
                    gzip.open(compressed_tmp, "wb") as target,
                ):
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                os.replace(compressed_tmp, compressed_archive)
                os.remove(archive)
                archive = compressed_archive
            except OSError as exc:
                try:
                    os.remove(compressed_tmp)
                except OSError:
                    pass
                builtins.print(
                    f"[{self.cls_name}][maintain] archive compression "
                    f"failed: {exc}"
                )

            prefix = f"{self.filename}."
            archives = sorted(
                (
                    path for path in glob.glob(prefix + "*.archive*")
                    if not path.endswith(".tmp")
                ),
                key=lambda path: os.path.getmtime(path),
                reverse=True,
            )
            for old_archive in archives[self.ROTATE_ARCHIVE_COUNT:]:
                try:
                    os.remove(old_archive)
                except OSError:
                    pass
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
        
        current_time = int(time.time() * 1000)
      
        if symbol not in self.cache:
            self.cache[symbol] = [] #self.cache.setdefault(symbol, []).extend(new_items)
            
        count_new_items = len(new_items)
        print(f"[{self.cls_name}][Info] {symbol}:  new_items {new_items}") 
       
        #new_items = self.filter_new_items(self.cache[symbol], new_items)
        # with self.lock:
        #     cache_copy = list(self.cache.get(symbol, []))
        # new_items = self.filter_new_items(cache_copy, new_items)

        with self.lock:  # Protected write.
            if self.append_mode:    # History mode retains every PriceOrders/Price/Trade element.
                #if isinstance(new_items, dict):
                #    new_items = [new_items]
                #elif not isinstance(new_items, list):
                #    new_items = [new_items]                    
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
              
            self.fetchtime_time_per_symbol[symbol] = current_time
            self._mark_account_cache_dirty_locked()

        print(f"[{self.cls_name}][Info] {symbol}: Added {len(new_items)} new items.")

    def _persist_items(self, symbol, new_items):
        """Persist one symbol to the cache by default.

        Subclasses may override this, for example to record a timestamp and notify
        subscribers through ``CacheCurrentPriceManager._push_price``.
        """
        self.update_cache_per_symbol(symbol, new_items)

    def query_remote_and_update_cache(self):
        if not self.fetchtime_time_per_symbol:
            self.fetchtime_time_per_symbol = self.__rebuild_fetchtime_times()

        account_cycle_high_water = None
        for symbol in list(self.symbols):
            request_high_water = int(time.time() * 1000)
            startTime = self.fetchtime_time_per_symbol.get(symbol, self.fallback_time_default)
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
                    self.fetchtime_time_per_symbol[symbol] = max(
                        0, request_high_water - 1
                    )
                    self._mark_account_cache_dirty_locked()
        if self._is_account_cache():
            with self.lock:
                self._last_complete_sync_at_ms = (
                    account_cycle_high_water or 0)
        return True

    def on_items_update(self, symbol, items):
        print(f"[{self.cls_name}][Info] {symbol}: WS Items updated to {items}")
        if not self.fetchtime_time_per_symbol:
            self.fetchtime_time_per_symbol = self.__rebuild_fetchtime_times()
        self.update_cache_per_symbol(symbol, items)
        
    def _should_poll(self):
        """Decide whether the synchronization loop polls the API.

        Cache24PriceManager is push-only, while CacheCurrentPriceManager polls only when
        WebSocket is unavailable. The default uses the global ``WS_ONLY_MODE`` gate.
        """
        return _should_poll_for_manager(self.cls_name)

    def _persist_periodic_state(self, sync_complete):
        """Persist one cycle without reversing the Trade generation lock order."""
        if self.cls_name == "CacheTradeManager" and self.save_state:
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

        with self.lock:
            persisted = self.save_state_to_file_if_enabled()
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

        self.thread = threading.Thread(target=run, name=self.cls_name, daemon=True)
        self.thread.daemon = True  # Do not let this thread block process shutdown.
        self.thread.start()
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
    def __init__(self, sync_ts, symbols, filename, api_client=api):
        super().__init__(sync_ts, symbols, filename, append_mode=True,
                         api_client=api_client, append_persist=True)

    @staticmethod
    def _is_valid_trade(trade):
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
                not self._is_valid_trade(t)
                or str(t["symbol"]).upper() != str(symbol).upper()
            ):
                raise RuntimeError(
                    f"{self.cls_name} received a malformed Binance trade row: {t!r}"
                )
        return new_trades

    def _persist_items(self, symbol, new_items):
        """Append immutable fills once and reject conflicting duplicate IDs."""
        with self.lock:
            bucket = self.cache.setdefault(symbol, [])
            existing = {}
            for item in bucket:
                if (
                    not self._is_valid_trade(item)
                    or str(item["symbol"]).upper() != str(symbol).upper()
                ):
                    raise RuntimeError("invalid Trade row already in memory")
                trade_id = str(int(item["id"]))
                signature = self._trade_semantic_signature(item)
                if trade_id in existing:
                    raise RuntimeError("duplicate immutable Trade ID in memory")
                existing[trade_id] = signature

            unique = []
            for item in new_items:
                if (
                    not self._is_valid_trade(item)
                    or str(item["symbol"]).upper() != str(symbol).upper()
                ):
                    raise RuntimeError("invalid incoming Trade row")
                trade_id = str(int(item["id"]))
                signature = self._trade_semantic_signature(item)
                if trade_id in existing:
                    if existing[trade_id] != signature:
                        raise RuntimeError(
                            f"conflicting Trade row for immutable ID {trade_id}"
                        )
                    continue
                existing[trade_id] = signature
                unique.append(dict(item))
            if not unique:
                return
            bucket.extend(unique)
            self.fetchtime_time_per_symbol[symbol] = int(time.time() * 1000)
            self._mark_account_cache_dirty_locked()

    def last_opposite_fill_price(self, symbol, order_type):
        """Return the latest opposite fill price without a time limit.

        BUY uses the latest SELL and SELL uses the latest BUY. Read the manager's own
        real WebSocket fill cache without an API call or noise from canceled orders.
        REST overlap may append an older fill after a newer WebSocket event, so
        authoritative exchange time and immutable trade ID determine recency.
        """
        want_buyer = (order_type.upper() == "SELL")   # The opposite of SELL is BUY.
        latest_key = None
        latest_price = None
        with self.lock:
            for trade in self.cache.get(symbol, []):
                if (
                    not self._is_valid_trade(trade)
                    or str(trade["symbol"]).upper() != str(symbol).upper()
                    or trade["isBuyer"] != want_buyer
                ):
                    continue
                key = (int(trade["time"]), int(trade["id"]))
                if latest_key is None or key > latest_key:
                    latest_key = key
                    latest_price = float(trade["price"])
        return latest_price


class CacheOrderManager(CacheManagerInterface):
    def __init__(self, sync_ts, symbols, filename, api_client=api):
        # Orders are mutable: an execution report updates the same order from partial
        # to filled. Keep an atomic snapshot so an in-place update is persisted.
        super().__init__(sync_ts, symbols, filename, append_mode=True,
                         api_client=api_client, append_persist=False)
        
    def _is_valid_trade(self, trade):
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
                not CacheTradeManager._is_valid_trade(fill)
                or str(fill["symbol"]).upper() != str(symbol).upper()
            ):
                raise RuntimeError(
                    f"{self.cls_name} received a malformed Binance fill row: {fill!r}"
                )
            order_fills.append({
                "orderId": fill["orderId"],
                "price": float(fill["price"]),
                "quantity": float(fill["qty"]),
                "timestamp": int(fill["time"]),
                "side": "BUY" if fill["isBuyer"] else "SELL",
                "_fillId": str(fill["id"]),
            })
        print(f"[{self.cls_name}][info] New order fills: {len(order_fills)}")
        return order_fills

    def _persist_items(self, symbol, new_items):
        """Merge each immutable fill exactly once into its order aggregate."""
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
                self.fetchtime_time_per_symbol[symbol] = now_ms
                self._mark_account_cache_dirty_locked()


class CacheSparsePriceManager(CacheManagerInterface):
    """Store sparse seven-minute price samples for two years in JSONL.

    Unlike dense WebSocket-pushed Cache24 managers, this periodically asks for the
    current price. ``Sparse`` describes the stable sampling mechanism rather than the
    mutable retention policy. priceAnalysis is its only consumer.
    """

    def __init__(self, sync_ts, symbols, filename, api_client=api):
        # Continuous append-only seven-minute history uses JSONL append persistence.
        super().__init__(sync_ts, symbols, filename, append_mode=True,
                         api_client=api, append_persist=True)

    # def rebuild_fetchtime_times(self):
        # if not self.cache:
            # return {}
        # last_times = {symbol: max(entry[0] for entry in self.cache if entry) for symbol in self.symbols}
        # return last_times

    def rebuild_fetchtime_times(self):
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

        return [[int(ts_ms), price]]

    def get_all_symbols_from_cache(self):
        return self.symbols


class Cache24PriceManager(CacheManagerInterface):
    """Collect maximum-granularity prices over the latest ``KEEP_HOURS``.

    This manager neither polls nor subscribes directly to WebSocket. It receives every
    update from CacheCurrentPriceManager and uses ``get_remote_items`` only during
    initialization when the persisted file is missing.
    """
    KEEP_HOURS = 24   # May be configured per instance.

    def __init__(self, sync_ts, symbols, filename, api_client=api):
        super().__init__(sync_ts, symbols, filename, append_mode=True, api_client=api_client)
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
        if not self.fetchtime_time_per_symbol:
            self.fetchtime_time_per_symbol = self._CacheManagerInterface__rebuild_fetchtime_times()
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

    def rebuild_fetchtime_times(self):
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
            return [[int(entry[0]), entry[1]]]
        except Exception as e:
            print(f"[{self.cls_name}][Error] get_price {symbol}: {e}")
            return []

    def get_all_symbols_from_cache(self):
        return self.symbols

    def _should_poll(self):
        # Prices arrive exclusively through CurrentPrice's ``on_price_update`` callback.
        # Base periodic_sync only saves, resynchronizes, and maintains; it does not poll.
        return False


class Cache24LongPriceManager(Cache24PriceManager):
    """Six-month archive variant using incremental JSONL persistence.

    This remains separate from the live 24-hour manager and inherits its price-update
    behavior while overriding persistence only. JSONL appends new ticks at constant cost
    instead of rewriting a growing archive every minute.

    The base JSONL writer tracks how many entries it wrote and assumes the list grows only
    at the tail. Trimming from the head would otherwise make retained entries appear new
    and duplicate them, so this class compacts whenever trimming actually removes data.
    """

    LONG_TRIM_INTERVAL_SEC = 24 * 3600

    def __init__(self, sync_ts, symbols, filename, api_client=api):
        # Long archives must not rewrite the entire JSONL whenever a single oldest
        # tick expires.  Initialize the maintenance clock before callbacks can run.
        self._last_long_trim = 0.0
        # Call CacheManagerInterface directly because Cache24PriceManager does not expose
        # append_persist and invokes load_state before this class could change it.
        CacheManagerInterface.__init__(self, sync_ts, symbols, filename,
                                        append_mode=True, api_client=api_client,
                                        append_persist=True)
        # Mirror Cache24PriceManager's cleanup of unrelated symbols from legacy files.
        with self.lock:
            self.cache = {s: v for s, v in self.cache.items() if s in self.symbols}

    @staticmethod
    def _tail_lines(path, limit, block_size=1024 * 1024):
        """Read at most the last complete ``limit`` lines without scanning the file."""
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            position = handle.tell()
            chunks = []
            newline_count = 0
            while position > 0 and newline_count <= limit:
                size = min(block_size, position)
                position -= size
                handle.seek(position)
                chunk = handle.read(size)
                chunks.append(chunk)
                newline_count += chunk.count(b"\n")
        return b"".join(reversed(chunks)).splitlines()[-limit:]

    def _load_jsonl(self):
        """Load only a recent working set; the JSONL file remains the full archive."""
        with self.lock:
            self.cache = {}
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
                        self.fetchtime_time_per_symbol = json.load(handle).get("fetchtime", {})
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
                    0, self._persisted_counts.get(symbol, len(entries)) - overflow)
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
            self._persisted_counts[symbol] = max(0, self._persisted_counts.get(symbol, before) - removed)

    def maintain_append_persist(self):
        """Prune the full disk archive with bounded memory, then reload its recent tail."""
        cutoff_ms = int((time.time() - self.RETENTION_DAYS * 86400) * 1000)
        if not os.path.exists(self.filename):
            return
        changed = False
        try:
            with self.lock:
                with atomic_write(self.filename) as target, open(self.filename, encoding="utf-8") as source:
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
                self._load_jsonl()
            if changed:
                builtins.print(f"[{self.cls_name}][maintain] pruned expired/corrupt disk records")
            if os.path.getsize(self.filename) > self.MAX_FILE_BYTES:
                self._rotate_disk_archive()
        except OSError as exc:
            builtins.print(f"[{self.cls_name}][Error] streaming maintenance {self.filename}: {exc}")

    def compact_jsonl(self):
        """Never rebuild an existing long archive from its bounded memory tail."""
        if os.path.exists(self.filename):
            self.maintain_append_persist()
        else:
            super().compact_jsonl()  # Legacy JSON migration still starts from full memory.

    def _rotate_disk_archive(self):
        """Archive an oversized file while retaining a complete-line tail on disk."""
        with self.lock:
            archive = f"{self.filename}.{int(time.time())}.archive"
            os.replace(self.filename, archive)
            keep_bytes = max(1, int(os.path.getsize(archive) * self.ROTATE_KEEP_FRACTION))
            with open(archive, "rb") as source:
                source.seek(max(0, os.path.getsize(archive) - keep_bytes))
                if source.tell() > 0:
                    source.readline()
                tail = source.read()
            with atomic_write(self.filename) as target:
                target.write(tail.decode("utf-8", errors="ignore"))
            self._load_jsonl()
            with open(archive, "rb") as source, gzip.open(archive + ".gz.tmp", "wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            os.replace(archive + ".gz.tmp", archive + ".gz")
            os.remove(archive)
            archives = sorted(
                glob.glob(f"{self.filename}.*.archive.gz"),
                key=os.path.getmtime,
                reverse=True,
            )
            for old_archive in archives[self.ROTATE_ARCHIVE_COUNT:]:
                os.remove(old_archive)

    # Inherit get_remote_items from Cache24PriceManager, including real observation timestamps.


class CachePriceLongTrendManager(CacheManagerInterface):
    def __init__(self, sync_ts, symbols, filename, api_client=api):
        super().__init__(sync_ts, symbols, filename, append_mode=False)

    # Inherit get_all_symbols_from_cache from the base class.

    # def rebuild_fetchtime_times(self):
        # """
        # Infer each symbol's latest record time from self.cache.
        # """
        # last_times = defaultdict(int)
        # for price_trend in self.cache:
            # symbol = price_trend.get("symbol")
            # ts = price_trend.get("timestamp", 0) * 1000
            # if ts > last_times[symbol]:
                # last_times[symbol] = ts

        # Apply a negative 60-second safety offset.
        # for symbol in last_times:
            # last_times[symbol] = max(0, last_times[symbol] - 60_000)

        # return dict(last_times)
        
    def rebuild_fetchtime_times(self):
        last_times = defaultdict(int)
        for symbol, items in self.cache.items():
            for item in items:
                if not isinstance(item, dict):
                    continue
                try:
                    timestamp = float(item.get("timestamp", 0))
                except (TypeError, ValueError, OverflowError):
                    continue
                if not math.isfinite(timestamp) or timestamp <= 0:
                    continue
                ts = int(timestamp * 1000)
                if ts > last_times[symbol]:
                    last_times[symbol] = ts
        for symbol in last_times:
            last_times[symbol] = max(0, last_times[symbol] - 60_000)
        return dict(last_times)

    def get_remote_items(self, symbol, startTime):
        trend_filename = os.path.join(_CONFIG_ROOT, "priceanalysis.json")
        if not os.path.exists(trend_filename):
            print(f"[{self.cls_name}] File {trend_filename} does not exist.")
            return []

        try:
            with open(trend_filename, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            print(f"[{self.cls_name}] Could not read {trend_filename}: {exc}")
            return []

        if not isinstance(data, dict) or symbol not in data:
            return []
        # An explicitly published null is a neutral snapshot, distinct from a
        # missing symbol. Preserve it so stale in-memory trends are cleared.
        return [data[symbol]]
        

class CacheAssetValueManager(CacheManagerInterface):
    def __init__(self, sync_ts, symbols, filename, api_client=api):
        super().__init__(sync_ts, symbols, filename, append_mode=True,
                         api_client=api_client, append_persist=True)
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
            self.save_state_to_file()

    def rebuild_fetchtime_times(self):
        last_times = {}
        for symbol, items in self.cache.items():
            if not items:
                continue
            max_ts_sec = max(int(item.get("timestamp", 0)) for item in items if isinstance(item, dict))
            if max_ts_sec > 0:
                last_times[symbol] = max(0, (max_ts_sec * 1000) - 60_000)
        return last_times

    def get_remote_items(self, symbol, startTime):
        try:
            total_usdc = self.api_client.get_total_assets_value_usdc(use_cache=False)
        except Exception as e:
            print(f"[{self.cls_name}][Error] Cannot query the total value: {e}")
            return []

        if not isinstance(total_usdc, (int, float)) or total_usdc <= 0:
            print(f"[{self.cls_name}][Error] Invalid total value: {total_usdc}")
            return []
            
        now_sec = int(time.time())
        snapshot = {
            "timestamp": now_sec,
            "datetime_local": datetime.now().isoformat(timespec="seconds"),
            "total_value_usdc": round(float(total_usdc), 8),
        }
        return [snapshot]


# ######
# ###### CacheCurrentPriceManager: per-symbol current price, WebSocket-first with HTTP fallback
# ######

class CacheCurrentPriceManager(CacheManagerInterface):
    """Maintain the latest timestamped price per symbol.

    This is a drop-in replacement for ``bapi.get_current_price``. WebSocket is primary;
    HTTP polling runs only after WebSocket has been silent beyond ``WS_TIMEOUT_SEC``.
    ``get_price`` forces HTTP when the cached entry exceeds ``STALE_THRESHOLD_MS``.
    Snapshot semantics retain one ``[[timestamp_ms, price]]`` entry per symbol in a
    shared cache_currentprice.json file.
    """

    WS_TIMEOUT_SEC     = 15      # Treat WebSocket as unavailable after 15 silent seconds.
    STALE_THRESHOLD_MS = 5_000  # Force HTTP when a price is older than five seconds.
    FREQ_WINDOW_SEC    = 60     # Window used to measure update frequency.

    def __init__(self, sync_ts, symbols, filename, ws_manager=None, api_client=api,
                 market_api=None, provider_names=None):
        self._ws_manager        = ws_manager
        self._ws_last_event_ts  = 0.0      # Set before the base initializer.
        self._price_subscribers = []       # The base initializer preserves existing values.
        self._update_timestamps: dict = defaultdict(deque)  # Also required before super().
        self._first_sleep       = True     # Let WebSocket connect before HTTP fallback.
        # The injectable market-data facade defaults to the global singleton. Set it before
        # the base initializer because load_state may immediately call get_remote_items.
        self.market_api = market_api or _market_api.api
        self._provider_names = {}
        self.bind_providers(provider_names)
        super().__init__(sync_ts, symbols, filename, append_mode=False, api_client=api_client)
        if ws_manager is not None:
            ws_manager.subscribe(self)

    # ── WS health ────────────────────────────────────────────────────────────

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

    def _ws_is_healthy(self):
        return (time.time() - self._ws_last_event_ts) < self.WS_TIMEOUT_SEC

    # Inherit price subscription and notification methods from CacheManagerInterface.

    # ── WebSocket callback overriding the interface method ───────────────────

    def _record_price_timestamp(self, symbol: str) -> None:
        now = time.time()
        dq = self._update_timestamps[symbol]
        dq.append(now)
        cutoff = now - self.FREQ_WINDOW_SEC
        while dq and dq[0] < cutoff:
            dq.popleft()

    def _push_price(self, symbol: str, price: float) -> None:
        """Insert a price and notify subscribers without changing WebSocket health.

        The polling thread and get_price also use this path; only a real WebSocket event
        in ``on_items_update`` updates ``_ws_last_event_ts``.
        """
        ts_ms = int(time.time() * 1000)
        if not self.fetchtime_time_per_symbol:
            self.fetchtime_time_per_symbol = self._CacheManagerInterface__rebuild_fetchtime_times()
        self._record_price_timestamp(symbol)
        self.update_cache_per_symbol(symbol, [[ts_ms, price]])
        self._notify_price_subscribers(symbol, ts_ms, price)

    def on_items_update(self, symbol: str, items):
        """Handle a real WebSocket event and update WebSocket health."""
        self._ws_last_event_ts = time.time()   # Real WebSocket events only.
        price = items[0] if items else None
        if price is None:
            return
        self._push_price(symbol, price)

    # ── CacheManagerInterface abstract methods ───────────────────────────────

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
        self._push_price(symbol, new_items[0][1])

    def rebuild_fetchtime_times(self):
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

    def cached_price_observation(self, symbol: str, max_age_sec: float):
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
            self._push_price(symbol, new[0][1])
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


# ── Singleton ─────────────────────────────────────────────────────────────────

_current_price_instance: Optional[CacheCurrentPriceManager] = None
_current_price_lock = threading.Lock()

def get_current_price_manager(ws_manager=None, symbols=None, sync_ts=None,
                              start_sync=True,
                              provider_names=None) -> CacheCurrentPriceManager:
    """Return or lazily create the CacheCurrentPriceManager singleton.

    ``sync_ts=None`` preserves the current interval and uses the configured default on
    first creation. An explicit value updates the live interval read by the thread.
    Internal calls must pass None so they do not override main-process configuration.
    """
    global _current_price_instance
    if _current_price_instance is not None:
        if sync_ts is not None:
            _current_price_instance.sync_ts = sync_ts   # Live update.
        if provider_names:
            _current_price_instance.bind_providers(provider_names)
        if ws_manager is not None:
            _current_price_instance.attach_ws_manager(ws_manager)
        if start_sync:
            _current_price_instance.periodic_sync(save_state=False)
        return _current_price_instance
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
        _syms = symbols if symbols is not None else sym.symbols
        _current_price_instance = CacheCurrentPriceManager(
            sync_ts     = sync_ts if sync_ts is not None else CURRENTPRICE_SYNC_INTERVAL_SEC,
            symbols     = _syms,
            filename    = "cache_currentprice.json",
            ws_manager  = ws_manager,
            provider_names = provider_names,
            api_client  = api,
        )
        if start_sync:
            _current_price_instance.periodic_sync(save_state=False)
    return _current_price_instance


# ######
# ###### CachePriceShortTrendManager: windows, calculated trend, and cross-process cache
# ######

class CachePriceShortTrendManager:
    """Calculate per-symbol instant trends and share snapshots across processes.

    Writers build windows, subscribe to Cache24, and publish fast gradient/epsilon values.
    Readers such as rtrade use only the gate backed by the shared file.
    """
    EPSILON_K         = 1.0     # Informed noise threshold: k * stddev(gradient).
    FAVORABLE_REL_EPS = 1e-5    # Price-relative fallback when epsilon is absent.
    TREND_STALE_SEC   = 15.0    # Do not delay based on older snapshots.
    # Percentage thresholds are scale-invariant and configurable per window or symbol.
    PRICE_CHANGE_THRESHOLD_SMALL = u.calculate_difference_percent(60000, 60000 - 310)
    PRICE_CHANGE_THRESHOLD_BIG   = u.calculate_difference_percent(97000, 95000 - 377)
    FULL_EVAL_INTERVAL_SEC = 3.0   # Cadence for expensive complete metrics.
    FLUSH_INTERVAL_SEC     = 0.5   # Writer-only file flush cadence.
    # Every window is a slice of the same 24-hour Cache24 buffer. The smallest is primary.
    WINDOW_SECONDS = [3.7 * 60, 2.5 * 60 * 60]   # [3.7 min momentum, 2.5 hours trend]
    _live_instances = weakref.WeakSet()

    def __init__(self, symbols, filename="cache_instant_trend.json", writer=False,
                 window_seconds=None, thresholds=None):
        self._live_instances.add(self)
        self.symbols = list(symbols)
        self.filename = u.cache_path(filename)   # Store cache data in the cachedb subdirectory.
        self.writer = writer   # Only the writer process persists the file.
        # Sort N window durations so index zero is primary.
        secs = list(window_seconds) if window_seconds else list(self.WINDOW_SECONDS)
        self.window_seconds = sorted(float(s) for s in secs)
        # Thresholds may be callable, keyed by window, keyed by symbol and window, or None.
        self._threshold_fn = self._build_threshold_fn(thresholds)
        self._mem = {}
        self._lock = threading.RLock()
        self._file_mtime = None
        self._file_cache = None
        # Writer state populated by start_computation: symbol/window to analyzer objects.
        self.windows = {}
        self.analyzers = {}
        self.current_price_mgr = None
        # start_computation supplies raw Cache24 buffers for dynamic windows. Until then,
        # dynamic windows are unavailable and safely return unknown.
        self._cache24_managers = None
        self._computing = False
        self._full_eval_thread = None
        self._flush_thread = None
        self._stop_event = threading.Event()

    # Extreme window durations derived from the sorted list.
    @property
    def window_small_sec(self):
        return self.window_seconds[0]

    @property
    def window_big_sec(self):
        return self.window_seconds[-1]

    def _build_threshold_fn(self, thresholds):
        """Normalize ``thresholds`` to a ``(symbol, seconds) -> percentage`` function."""
        if callable(thresholds):
            return thresholds
        small, big = self.window_seconds[0], self.window_seconds[-1]

        def per_window_default(sec):
            # The smallest window uses SMALL; all others use BIG.
            return self.PRICE_CHANGE_THRESHOLD_SMALL if float(sec) <= small else self.PRICE_CHANGE_THRESHOLD_BIG

        if isinstance(thresholds, dict):
            # Accept either per-symbol/per-window or direct per-window dictionaries.
            per_symbol = all(isinstance(v, dict) for v in thresholds.values()) and len(thresholds) > 0
            if per_symbol:
                tbl = {sym: {float(k): v for k, v in d.items()} for sym, d in thresholds.items()}
                return lambda sym, sec: tbl.get(sym, {}).get(float(sec), per_window_default(sec))
            tbl = {float(k): v for k, v in thresholds.items()}
            return lambda sym, sec: tbl.get(float(sec), per_window_default(sec))

        return lambda sym, sec: per_window_default(sec)

    def threshold_for(self, symbol, seconds):
        """Return the percentage threshold for one symbol window."""
        return self._threshold_fn(symbol, float(seconds))

    # ── Writer: build windows and subscribe to Cache24 ───────────────────────
    def start_computation(self, cache24_managers=None, current_price_mgr=None, run_full_eval=False):
        """Build windows and the fast channel, optionally starting full evaluation."""
        if self._computing:
            if run_full_eval:
                self._start_full_eval_loop()
            return
        self._stop_event.clear()
        if self.writer:
            self.prime_from_file(symbols=self.symbols, overwrite=False)
        import pricewindow as pw
        if cache24_managers is None:
            cache24_managers = get_cache_manager("Price24")
        if current_price_mgr is None:
            current_price_mgr = get_current_price_manager()
        self.current_price_mgr = current_price_mgr
        self._cache24_managers = cache24_managers   # Raw sources for dynamic windows.
        for s in self.symbols:
          try:
            c24 = cache24_managers[s]
            current_price_mgr.subscribe_price(c24)          # CurrentPrice → Cache24
            self.windows[s]   = {}
            self.analyzers[s] = {}
            parts = []
            for sec in self.window_seconds:
                w = pw.PriceWindow.from_cache24(s, sec, c24)
                self.windows[s][sec]   = w
                self.analyzers[s][sec] = pw.WindowAnalyzer(w)
                parts.append(f"{sec:.0f}s: {len(w.prices)} (rate={w.sample_rate_sec:.2f}s)")
            c24.subscribe_price(self)                       # tick signal to the fast channel
            print(f"[InstantTrend][{s}] " + " ".join(parts))
          except Exception as _e:
            builtins.print(
                f"[InstantTrend][{s}] setup failed ({_e}) — skipping; "
                "Binance is unaffected"
            )
        self._computing = True
        self._start_flush_loop()        # Decouple file I/O into a background thread.
        if run_full_eval:
            self._start_full_eval_loop()

    # ── Full metric calculation without trading logic ────────────────────────
    def evaluate_full(self, symbol):
        wins = self.windows.get(symbol)
        ans  = self.analyzers.get(symbol)
        if not wins or not ans:
            return None
        primary = self.window_seconds[0]
        if primary not in wins:
            return None
        if self.current_price_mgr is None:
            return None
        observation = self.current_price_mgr.cached_price_observation(
            symbol, self.TREND_STALE_SEC)
        if observation is None:
            return None
        observed_at, current_price = observation

        # Calculate slopes for every window, keyed by seconds.
        slopes = {}
        primary_pos = None
        for sec in self.window_seconds:
            slope, pos = ans[sec].check_price_change(self._threshold_fn(symbol, sec))
            slopes[sec] = slope
            if sec == primary:
                primary_pos = pos

        # Detailed metrics come only from the smallest primary window.
        pwin, pan = wins[primary], ans[primary]
        gradient, gc, slope_full, gradient_recent = pwin.get_instant_trend()
        # Update memory only; the flush loop writes in the background.
        self._set_mem(
            symbol,
            final_trend=gradient, growth_coefficient=gc,
            slope_full=slope_full, gradient_recent=gradient_recent,
            slope_small=slopes[self.window_seconds[0]],     # Smallest.
            slope_big=slopes[self.window_seconds[-1]],      # Largest.
            slopes={f"{int(s)}": v for s, v in slopes.items()},
            slope_max_min=pan.calculate_slope_max_min(),
            pos=primary_pos, epsilon=pwin.get_noise_epsilon(self.EPSILON_K),
            current_price=current_price,
            ts=observed_at,
        )

    def _start_full_eval_loop(self):
        if self._full_eval_thread is not None and self._full_eval_thread.is_alive():
            return
        def run():
            while not self._stop_event.is_set():
                for s in list(self.symbols):
                    if self._stop_event.is_set():
                        break
                    try:
                        self.evaluate_full(s)
                    except Exception as e:
                        print(f"[CachePriceShortTrendManager] evaluate_full {s}: {e}")
                self._stop_event.wait(self.FULL_EVAL_INTERVAL_SEC)
        self._full_eval_thread = threading.Thread(target=run, name="InstantTrendFullEval", daemon=True)
        self._full_eval_thread.start()

    # ── Fast Cache24 subscriber channel: gradient and epsilon on every tick ───
    def on_price_update(self, symbol, ts_ms, price):
        win = self.get_window(symbol)   # Primary window.
        if win is None:
            return
        try:
            # The fast path computes only a cheap gradient in memory. It writes separate
            # ``_fast`` keys for low-latency gates and never overwrites richer full-evaluation
            # results. Sharing keys previously created a network-timing race between paths.
            g = win.get_recent_gradient()
            eps = win.get_noise_epsilon(self.EPSILON_K)
            self._set_mem(symbol, gradient_recent_fast=g, epsilon=eps,
                          trend_fast=(1 if g > 0 else -1 if g < 0 else 0),
                          current_price=price, ts_fast=float(ts_ms) / 1000.0)
        except Exception as e:
            print(f"[CachePriceShortTrendManager] on_price_update {symbol}: {e}")

    # ── Calculation API ──────────────────────────────────────────────────────
    def get_window(self, symbol, seconds=None):
        """Return the requested window, or the smallest primary window for None."""
        wins = self.windows.get(symbol) or {}
        return wins.get(float(seconds) if seconds is not None else self.window_seconds[0])

    def get_analyzer(self, symbol, seconds=None):
        """Return the requested analyzer, or the smallest primary analyzer for None."""
        ans = self.analyzers.get(symbol) or {}
        return ans.get(float(seconds) if seconds is not None else self.window_seconds[0])

    def get_instant_trend(self, symbol):
        win = self.get_window(symbol)
        return win.get_instant_trend() if win else None

    # ── Dynamic window ────────────────────────────────────────────────────────
    # Calculate only the requested horizon from Cache24's existing raw buffer. Bisect
    # makes cost proportional to samples in that window rather than the full 24 hours.
    def get_instant_trend_for_window(self, symbol, window_seconds, now=None):
        """Calculate trend on demand for an arbitrary window in seconds.

        Use the shared formula directly on a Cache24 slice without constructing a
        PriceWindow. Return metrics or None when Cache24 is unavailable, the symbol is
        not tracked, or fewer than three samples exist. Clamp requests to configured
        bounds and report the adjusted horizon without raising.
        """
        if self._cache24_managers is None:
            return None
        c24 = self._cache24_managers.get(symbol)
        if c24 is None:
            return None

        req = float(window_seconds)
        clamped = min(max(req, CM_DYNAMIC_WINDOW_MIN_SEC), CM_DYNAMIC_WINDOW_MAX_SEC)
        if clamped != req:
            print(f"[get_instant_trend_for_window] {symbol}: window_seconds={req:.0f} "
                  f"outside [{CM_DYNAMIC_WINDOW_MIN_SEC:.0f}, {CM_DYNAMIC_WINDOW_MAX_SEC:.0f}] "
                  f"-> clamped to {clamped:.0f}")

        try:
            entries = c24.get_recent_entries(symbol, last_seconds=clamped)
        except Exception as e:
            print(f"[get_instant_trend_for_window] {symbol}: get_recent_entries failed ({e})")
            return None
        if len(entries) < 3:
            return None   # Too few samples means unknown, not a noisy assumption.

        now = now if now is not None else time.time()
        newest_ts_sec = entries[-1][0] / 1000.0
        if now - newest_ts_sec > self.TREND_STALE_SEC:
            return None   # Treat an old latest tick as stale, matching fresh_snapshot.

        import numpy as np
        import pricewindow as pw
        prices = [e[1] for e in entries]
        sample_rate = pw.PriceWindow._sample_rate_from_entries(entries)
        final_trend, growth_coefficient, slope_full, gradient_recent = pw.instant_trend_from_prices(
            prices, sample_rate)
        epsilon = float(self.EPSILON_K * np.std(np.gradient(np.array(prices))))

        return dict(final_trend=final_trend, growth_coefficient=growth_coefficient,
                    slope_full=slope_full, gradient_recent=gradient_recent,
                    epsilon=epsilon, n_samples=len(entries), window_seconds=clamped,
                    sample_rate_sec=sample_rate, ts=now)

    def is_trend_up_for_window(self, symbol, window_seconds, now=None):
        """Evaluate monitortrades' upward-trend condition over a configurable window.

        Return False as the same neutral fallback when the dynamic window is unavailable.
        """
        dyn = self.get_instant_trend_for_window(symbol, window_seconds, now=now)
        if dyn is None:
            return False
        slope = dyn["gradient_recent"]
        gradient = dyn["final_trend"]
        return slope > 0 or (slope == 0 and gradient > 0)

    # ── Atomic cross-process JSON store ──────────────────────────────────────
    def _write_file(self):
        try:
            atomic_write_json(self.filename, self._mem)
        except Exception as e:
            print(f"[CachePriceShortTrendManager] write failure for {self.filename}: {e}")

    def _read_file(self):
        # Always read this small, infrequently accessed file to guarantee cross-process
        # correctness without relying on coarse mtime resolution.
        try:
            with open(self.filename, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _set_mem(self, symbol, **fields):
        """Merge fields in memory only; the flush loop owns file I/O."""
        with self._lock:
            snap = dict(self._mem.get(symbol) or self._read_file().get(symbol) or {})
            snap.update(fields)
            snap["symbol"] = symbol
            self._mem[symbol] = snap

    def update_snapshot(self, symbol, **fields):
        """Update memory and flush immediately only when this instance is the writer."""
        self._set_mem(symbol, **fields)
        if self.writer:
            with self._lock:
                self._write_file()

    def _start_flush_loop(self):
        """Run a writer-only thread that decouples periodic file I/O from calculation."""
        if not self.writer:
            return
        if self._flush_thread is not None and self._flush_thread.is_alive():
            return
        def run():
            while not self._stop_event.wait(self.FLUSH_INTERVAL_SEC):
                with self._lock:
                    if self._mem:
                        self._write_file()
        self._flush_thread = threading.Thread(target=run, name="InstantTrendFlush", daemon=True)
        self._flush_thread.start()

    def shutdown(self, timeout=5.0):
        """Stop evaluation and flush loops, then unsubscribe Cache24 sources."""
        self._stop_event.set()
        for cache24 in (self._cache24_managers or {}).values():
            unsubscribe = getattr(cache24, "unsubscribe_price", None)
            if unsubscribe is not None:
                unsubscribe(self)
        for thread in (self._full_eval_thread, self._flush_thread):
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=timeout)
        stopped = all(thread is None or not thread.is_alive()
                      for thread in (self._full_eval_thread, self._flush_thread))
        if stopped:
            self._full_eval_thread = None
            self._flush_thread = None
            self._computing = False
        return stopped

    @classmethod
    def shutdown_all_instances(cls, timeout=5.0):
        results = [manager.shutdown(timeout=timeout)
                   for manager in list(cls._live_instances)]
        return all(results) if results else True

    def prime_from_file(self, symbols=None, *, overwrite=True):
        """Prime memory from startup data so a reader can calculate fresh trends locally."""
        data = self._read_file()
        allowed = (
            None
            if symbols is None
            else {str(symbol) for symbol in symbols}
        )
        with self._lock:
            for symbol, snap in data.items():
                if (
                    isinstance(snap, dict)
                    and (allowed is None or symbol in allowed)
                    and (overwrite or symbol not in self._mem)
                ):
                    self._mem[symbol] = dict(snap)
        return len(self._mem)

    def get_snapshot(self, symbol):
        """Return local calculated/primed memory, or the file for a pure reader."""
        with self._lock:
            if symbol in self._mem:
                return dict(self._mem[symbol])
        return self._read_file().get(symbol)

    @staticmethod
    def _snapshot_is_usable(snap, now, max_age_sec):
        """Validate the common full-snapshot price and observation timestamp."""
        if not isinstance(snap, dict):
            return False
        try:
            observed_at = float(snap["ts"])
            price = float(snap["current_price"])
            age = float(now) - observed_at
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        return (
            math.isfinite(observed_at)
            and observed_at > 0
            and math.isfinite(price)
            and price > 0
            and -5.0 <= age <= float(max_age_sec)
        )

    def is_snapshot_fresh(self, symbol=None, max_age_sec=None):
        """Return whether memory or file state is fresh enough to trust the writer."""
        max_age_sec = max_age_sec if max_age_sec is not None else self.TREND_STALE_SEC
        now = time.time()
        if symbol is not None:
            snap = self.get_snapshot(symbol)
            return self._snapshot_is_usable(snap, now, max_age_sec)
        allt = self.get_all_snapshots()
        if not allt:
            return False
        latest = max((s.get("ts", 0) for s in allt.values()), default=0)
        return (now - latest) <= max_age_sec

    def become_writer(self):
        """Promote a calculating reader to writer when the previous writer is stale."""
        self.writer = True
        self._start_flush_loop()

    def get_snapshot_resilient(self, symbol, max_age_sec=None,
                               cache24_managers=None, current_price_mgr=None):
        """Read resiliently with lazy failover.

        Use authoritative memory while calculating, otherwise use a fresh shared file.
        If the file is stale, start local calculation once, become writer, and use memory.
        """
        if self._computing:
            return self.get_snapshot(symbol)
        if self.is_snapshot_fresh(symbol, max_age_sec):
            return self._read_file().get(symbol)
        # A stale file triggers autonomous local calculation and writer failover.
        builtins.print(
            f"[CachePriceShortTrendManager][WARN] stale file -> "
            f"local-computation failover ({symbol})"
        )
        self.prime_from_file()
        self.start_computation(cache24_managers, current_price_mgr)
        self.become_writer()
        return self.get_snapshot(symbol)

    def get_all_snapshots(self):
        with self._lock:
            if self._mem:
                return {s: dict(v) for s, v in self._mem.items()}
        return dict(self._read_file())

    def clear(self):
        with self._lock:
            self._mem.clear()
            self._file_mtime = None
            self._file_cache = None
            try:
                if os.path.exists(self.filename):
                    os.remove(self.filename)
            except Exception:
                pass

    # ── Opportunistic-delay gate with informed epsilon ───────────────────────
    def _epsilon(self, snap):
        eps = snap.get("epsilon")
        if eps is not None and eps > 0:
            return float(eps)
        price = abs(snap.get("current_price") or 0.0)
        return price * self.FAVORABLE_REL_EPS

    def fresh_snapshot(self, symbol, now=None):
        """Return a symbol snapshot only when it is within ``TREND_STALE_SEC``.

        This is the shared public staleness check used by both trend and wait decisions.
        """
        snap = self.get_snapshot(symbol)
        now = now if now is not None else time.time()
        if not self._snapshot_is_usable(snap, now, self.TREND_STALE_SEC):
            return None
        return snap

    def should_wait(self, side, symbol, window_seconds=None, fast=False,
                     use_noise_gate=True, now=None):
        """Decide whether to delay a BUY or SELL intent; True means wait.

        Unknown or stale data returns False and executes immediately, so a failed cache
        cannot block orders. ``window_seconds=None`` uses the precomputed primary window;
        another value calculates a bounded dynamic window where Cache24 is available.
        ``fast`` selects low-latency keys only for the primary window. The optional noise
        gate treats ``|gradient| <= epsilon`` as a reason to wait.
        """
        side = side.upper()

        if window_seconds is not None and float(window_seconds) != self.window_seconds[0]:
            dyn = self.get_instant_trend_for_window(symbol, window_seconds, now=now)
            if dyn is None:
                return False   # Unknown because Cache24 is unavailable, insufficient, or stale.
            g = float(dyn["growth_coefficient"] or 0.0)
            if use_noise_gate and abs(g) <= dyn["epsilon"]:
                return True
            if side == "BUY":
                return g < 0
            if side == "SELL":
                return g > 0
            return False

        snap = self.fresh_snapshot(symbol, now=now)
        if snap is None:
            return False   # Unknown or stale data executes immediately.
        if fast:
            g = snap.get("gradient_recent_fast", snap.get("gradient_recent", 0.0))
        else:
            g = snap.get("growth_coefficient", snap.get("gradient_recent", 0.0))
        g = float(g or 0.0)
        if use_noise_gate:
            eps = self._epsilon(snap)
            if abs(g) <= eps:
                return True
        if side == "BUY":
            return g < 0
        if side == "SELL":
            return g > 0
        return False

    def wait_for_favorable_entry(self, side, symbol, max_wait_sec=10.0,
                                 poll_sec=0.2, sleep_fn=time.sleep, mode="full",
                                 window_seconds=None):
        """Wait while trend favors delay, bounded by ``max_wait_sec``.

        Emit a visual heartbeat roughly once per second and return elapsed seconds.
        Forward ``window_seconds`` to ``should_wait`` for primary or dynamic evaluation.
        """
        deadline = time.time() + max_wait_sec
        waited = 0.0
        next_dot = 1.0
        while time.time() < deadline and self.should_wait(
                side, symbol, window_seconds=window_seconds, fast=(mode == "gradient")):
            sleep_fn(poll_sec)
            waited += poll_sec
            if waited >= next_dot:
                print(".", end="", flush=True)
                next_dot += 1.0
        if waited > 0:
            print()
        return waited


_short_trend_instance = None
_short_trend_lock = threading.Lock()

def get_short_trend_manager(symbols=None, filename="cache_instant_trend.json", writer=False):
    """Return the CachePriceShortTrendManager singleton.

    ``writer=True`` lets the process persist the file; calculators and readers use False.
    """
    global _short_trend_instance
    if _short_trend_instance is not None:
        if writer:
            _short_trend_instance.writer = True   # Idempotent writer promotion.
        return _short_trend_instance
    with _short_trend_lock:
        if _short_trend_instance is not None:
            if writer:
                _short_trend_instance.writer = True
            return _short_trend_instance
        _syms = symbols if symbols is not None else sym.symbols
        _short_trend_instance = CachePriceShortTrendManager(_syms, filename, writer=writer)
    return _short_trend_instance


# ######
# ###### GLOBAL VARIABLE FOR CACHE #######
# ######
     
ORDER_SYNC_INTERVAL_SEC = 0.4 * 60   # 24 seconds
TRADE_SYNC_INTERVAL_SEC = 3 * 60     # 3 minutes
PRICE_SYNC_INTERVAL_SEC = 7 * 60   # 7 minute
PRICE24_SYNC_INTERVAL_SEC = 30         # Fallback polling while WebSocket is inactive.
CURRENTPRICE_SYNC_INTERVAL_SEC = 30   # Same policy for CacheCurrentPriceManager.
PRICETREND_SYNC_INTERVAL_SEC = 10 * 60   # 10 minute
ASSETVALUE_SYNC_INTERVAL_SEC = 10 * 60  # 10 minutes 
# TODO: set this to 60 * 60  # 1 hour

class CacheFactory:
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
        "Price": {
            "class": CacheSparsePriceManager,
            "filename": None,  # Dictionary per symbol.
            "sync_ts": lambda: PRICE_SYNC_INTERVAL_SEC,
        },
        "Price24": {
            "class": Cache24PriceManager,
            "filename": None,  # Dictionary per symbol.
            "sync_ts": lambda: PRICE24_SYNC_INTERVAL_SEC,
        },
        "CurrentPrice": {
            "class": CacheCurrentPriceManager,
            "filename": "cache_currentprice.json",  # One shared file for all symbols.
            "sync_ts": lambda: CURRENTPRICE_SYNC_INTERVAL_SEC,
        },
        "PriceLongTrend": {
            "class": CachePriceLongTrendManager,
            "filename": "cache_price_long_trend.json",
            "sync_ts": lambda: PRICETREND_SYNC_INTERVAL_SEC,
        },
        "AssetValue": {
            "class": CacheAssetValueManager,
            "filename": "cache_asset_value.jsonl",
            "sync_ts": lambda: ASSETVALUE_SYNC_INTERVAL_SEC,
        },
    }

    @classmethod
    def get(cls, name, symbols=None, *, start_sync=True, provider_names=None):
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
                symbols = ["TOTAL"] if name == "AssetValue" else sym.symbols

            if name == "CurrentPrice":
                current_price_sync_ts = (
                    None if _current_price_instance is not None else sync_ts
                )
                cls._instances[name] = get_current_price_manager(
                    symbols=symbols,
                    sync_ts=current_price_sync_ts,
                    start_sync=False,
                    provider_names=provider_names,
                )
                sync_ts = cls._instances[name].sync_ts
            elif name in ("Price", "Price24"):
                # Price is append-JSONL history; Price24 is a bounded full-rewrite cache.
                prefix, ext = ("cache_price_", "jsonl") if name == "Price" else ("cache_24price_", "json")
                cls._instances[name] = {
                    s: manager_class(
                        sync_ts=sync_ts,
                        filename=f"{prefix}{s}.{ext}",
                        symbols=[s],
                        api_client=api,
                        **extra_kwargs,
                    )
                    for s in symbols
                }
            else:
                cls._instances[name] = manager_class(
                    sync_ts=sync_ts,
                    filename=config["filename"],
                    symbols=symbols,
                    api_client=api,
                    **extra_kwargs,
                )

            managers = (cls._instances[name].values()
                        if isinstance(cls._instances[name], dict)
                        else (cls._instances[name],))
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
        global _current_price_instance
        current_price = cls._instances.get("CurrentPrice")
        managers = []
        for value in cls._instances.values():
            managers.extend(value.values() if isinstance(value, dict) else (value,))
        results = [manager.shutdown(timeout=timeout) for manager in managers
                   if hasattr(manager, "shutdown")]
        if (
            current_price is _current_price_instance
            and getattr(current_price, "thread", None) is None
        ):
            _current_price_instance = None
        cls._instances = {}
        cls._sync_started = set()
        return all(results) if results else True

    @classmethod
    def remove(cls, name, timeout=5.0):
        """Remove a singleton cleanly after stopping all threads it owns."""
        global _current_price_instance
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
# ###### Real-time Binance user stream -> cache actions
# ######
_ws_bridge = None
_ws_bridge_lock = threading.Lock()
_ws_event_stats = defaultdict(int)

def _upsert_order_from_execution_report(event):
    symbol = event.get("s")
    if not symbol:
        return

    # Cache only executed orders, matching get_filled_orders. NEW or terminal orders
    # without fills are not transactions and would pollute the profit guard with zero prices.
    if (
        event.get("x") != "TRADE"
        or event.get("X") not in ("FILLED", "PARTIALLY_FILLED")
    ):
        return

    try:
        fill_id = int(event.get("t"))
        if fill_id < 0:
            raise ValueError("missing Binance trade ID")
        last_quantity = float(event.get("l") or 0)
        last_price = float(event.get("L") or 0)
        order_item = {
            "orderId": event.get("i"),
            # Store one immutable fill. A cumulative Z/z aggregate cannot name all
            # constituent fill IDs after reconnect, so REST backfill could count the
            # older fills twice. L/l plus t is exactly deduplicable across both paths.
            "price": last_price,
            "quantity": last_quantity,
            "timestamp": int(
                event.get("T") or event.get("E")
                or int(time.time() * 1000)
            ),
            "side": event.get("S"),
            "status": event.get("X"),
            "symbol": symbol,
            "eventType": event.get("x"),
            "_fillId": str(fill_id),
        }
    except (TypeError, ValueError, OverflowError) as exc:
        builtins.print(
            f"[cacheManager][WS] rejected malformed order report: {exc}"
        )
        return

    order_cache = get_cache_manager("Order")
    if not order_cache._is_valid_trade(order_item):
        builtins.print("[cacheManager][WS] rejected malformed order report")
        return
    order_cache._persist_items(symbol, [order_item])


def _append_trade_from_execution_report(event):
    if event.get("x") != "TRADE":
        return
    symbol = event.get("s")
    if not symbol:
        return

    trade_cache = get_cache_manager("Trade")
    if str(event.get("S") or "").upper() not in ("BUY", "SELL"):
        builtins.print("[cacheManager][WS] rejected Trade report with invalid side")
        return
    raw_trade_id = event.get("t")
    trade_id = str(
        raw_trade_id if raw_trade_id is not None
        else f"{event.get('i')}-{event.get('T')}"
    )
    trade_item = {
        "symbol": symbol,
        "id": trade_id,
        "orderId": event.get("i"),
        "price": event.get("L") or event.get("p"),
        "qty": event.get("l") or event.get("q"),
        "time": int(event.get("T") or event.get("E") or int(time.time() * 1000)),
        "isBuyer": str(event.get("S", "")).upper() == "BUY",
    }

    if not trade_cache._is_valid_trade(trade_item):
        builtins.print("[cacheManager][WS] rejected malformed Trade report")
        return
    trade_cache._persist_items(symbol, [trade_item])


def _refresh_asset_value_from_ws_event():
    asset_cache = get_cache_manager("AssetValue", symbols=["TOTAL"])
    items = asset_cache.get_remote_items("TOTAL", None)   # update_cache_per_symbol expects new_items.
    if items:
        asset_cache.update_cache_per_symbol("TOTAL", items)


def _persist_ws_updated_caches(event_type):
    if event_type == "executionReport":
        get_cache_manager("Order").save_state_to_file_if_enabled()
        get_cache_manager("Trade").save_state_to_file_if_enabled()
    elif event_type in ("balanceUpdate", "outboundAccountPosition"):
        get_cache_manager("AssetValue", symbols=["TOTAL"]).save_state_to_file_if_enabled()


def _refresh_symbol_in_cache(manager, symbol):
    """Refresh only one manager symbol efficiently for a WebSocket event."""
    try:
        start_time = manager.fetchtime_time_per_symbol.get(symbol, manager.fallback_time_default)
        items = manager.get_remote_items(symbol, start_time)
        if items:
            manager.update_cache_per_symbol(symbol, items)
    except Exception as e:
        print(f"[cacheManager] _refresh_symbol_in_cache {manager.cls_name}/{symbol}: {e}")


def _handle_binance_ws_event(event):
    print("cacheManager handler call from binance ....")
    event_type = event.get("e")
    if not event_type:
        return

    _ws_event_stats[event_type] += 1

    if event_type == "executionReport":
        symbol = event.get("s")
        if WS_EVENT_LOG_ENABLED:
            print(
                "[cacheManager][WS] executionReport "
                f"symbol={symbol} orderId={event.get('i')} "
                f"status={event.get('X')} execType={event.get('x')} side={event.get('S')}"
            )
        # Derive Order and Trade directly from the WebSocket payload without REST calls.
        # REST remains a rare fallback for missing symbols, while fallback polling covers
        # gaps created by WebSocket disconnections.
        if symbol:
            _upsert_order_from_execution_report(event)
            _append_trade_from_execution_report(event)
        else:
            for cache_name in ("Order", "Trade"):
                get_cache_manager(cache_name).query_remote_and_update_cache()
        _persist_ws_updated_caches(event_type)
        return

    if event_type in ("balanceUpdate", "outboundAccountPosition"):
        if WS_EVENT_LOG_ENABLED:
            print(
                f"[cacheManager][WS] {event_type} event received"
            )
        #_refresh_asset_value_from_ws_event()
        #_persist_ws_updated_caches(event_type)
        return

def enable_real_ws_event_sync():
    global _ws_bridge
    import sys
    if sys.modules.get("_cacheManager_initialized"):
        # Already started by an earlier import.
        if _ws_bridge is not None:
            return _ws_bridge
    with _ws_bridge_lock:
        if _ws_bridge is not None:
            return _ws_bridge
        from binance_api import bapi_ws
        # bapi_ws owns the stream class; cacheManager wires health callbacks that drive
        # fallback polling through ``_should_poll``.
        _ws_bridge = bapi_ws.BinanceAccountStream(
            on_event=_handle_binance_ws_event,
            on_available=_mark_ws_available,
            on_healthy=_mark_ws_event_received,
            on_unhealthy=_mark_ws_unhealthy,
            loss_timeout_sec=WS_LOSS_TIMEOUT_SEC,
        )
        _ws_bridge.start()
        return _ws_bridge
        
def _initialize_once():
    import sys
    if sys.modules.get("_cacheManager_initialized"):
        print("[cacheManager] Already initialized, skip.")
        return
    sys.modules["_cacheManager_initialized"] = True
    enable_real_ws_event_sync()
    print("⚙️ cacheManager: WS user-data bridge started (execution reports).")

# The user-data WebSocket bridge is opt-in and does not start during import. Processes
# that need real-time execution reports enable it explicitly:
#     import cacheManager as cm
#     cm.enable_real_ws_event_sync()   # or cm._initialize_once()
# Read-only consumers rely on polling and do not start WebSocket.


import concurrent.futures as _futures

# A poll deadline prevents the loop from waiting on one slow provider call. The
# provider's own network timeout still owns the worker lifecycle.
_NB_FETCH_DEADLINE_SEC = 8   # Shorter than the poll interval, so hangs do not extend a cycle.


def _fetch_price_with_deadline(
    market_api,
    symbol,
    pool,
    deadline_sec=_NB_FETCH_DEADLINE_SEC,
    provider_name=None,
):
    """Bound the poller's wait while the provider owns the network-call timeout."""
    kwargs = {"symbol": symbol}
    if provider_name is not None:
        kwargs["provider_name"] = provider_name
    fut = pool.submit(market_api.get_current_price, **kwargs)
    try:
        return fut.result(timeout=deadline_sec)
    except _futures.TimeoutError:
        fut.cancel()
        raise


class _NonBinanceTrendPoller:
    """Own the non-Binance poller's thread and executor lifecycle."""
    def __init__(self, thread, pool, stop_event):
        self.thread = thread
        self.pool = pool
        self.stop_event = stop_event

    def stop(self, timeout=5.0):
        self.stop_event.set()
        if self.thread is not threading.current_thread():
            self.thread.join(timeout=timeout)
        self.pool.shutdown(wait=False, cancel_futures=True)
        return not self.thread.is_alive()


def _start_nonbinance_trend_poller(cpm, symbols, interval_sec=10,
                                   fetch_deadline_sec=_NB_FETCH_DEADLINE_SEC,
                                   provider_names=None):
    """Feed instant trends for non-WebSocket, non-Binance symbols.

    Poll through the facade and push into the CurrentPrice/Cache24/InstantTrend chain.
    Per-symbol failures do not affect Binance. A bounded poll wait and at least two
    workers keep one slow fetch from freezing the poller; provider timeouts remain visible.
    """
    symbols = tuple(str(symbol or "").strip() for symbol in symbols)
    bindings = {
        str(symbol or "").strip().upper(): str(provider_name or "").strip()
        for symbol, provider_name in dict(provider_names or {}).items()
    }
    missing = [
        symbol for symbol in symbols
        if (
            not symbol
            or not bindings.get(symbol.upper())
            or bindings[symbol.upper()].lower() == "binance"
        )
    ]
    if missing:
        raise ValueError(
            "explicit provider bindings are required for non-Binance symbols: "
            + ", ".join(missing)
        )
    cpm.bind_providers(bindings)
    pool = _futures.ThreadPoolExecutor(
        max_workers=max(2, len(symbols)), thread_name_prefix="NBTrendFetch")
    stop_event = threading.Event()

    def run():
        while not stop_event.is_set():
            for s in symbols:
                if stop_event.is_set():
                    break
                try:
                    p = _fetch_price_with_deadline(
                        cpm.market_api,
                        s,
                        pool,
                        fetch_deadline_sec,
                        provider_name=bindings[s.upper()],
                    )
                    if p is not None and float(p) > 0:
                        cpm._push_price(s, float(p))
                except _futures.TimeoutError:
                    builtins.print(f"[NB-trend] {s}: fetch BLOCKED >{fetch_deadline_sec}s "
                                   f"(poll deadline — probably DNS or the network) — skipping, retrying next cycle")
                except Exception as _e:
                    builtins.print(f"[NB-trend] {s}: {_e}")
            stop_event.wait(interval_sec)
    t = threading.Thread(target=run, name="NonBinanceTrendPoller", daemon=True)
    t.start()
    return _NonBinanceTrendPoller(t, pool, stop_event)


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

    # Add configured non-Binance instruments that need cached instant trends. Preserve
    # venue identity because the same symbol cannot be routed safely by spelling alone.
    _nb_syms = []
    _nb_provider_names = {}
    _binance_symbol_keys = {
        str(_symbol).strip().upper() for _symbol in sym.symbols
    }
    from instruments_config import load_for
    for _inst in load_for("mt").values():
        _provider_name = str(_inst.provider_name).strip()
        _symbol = str(_inst.symbol).strip()
        if not _provider_name or not _symbol:
            raise RuntimeError("configured provider and symbol must be non-empty")
        _key = _symbol.upper()
        if _provider_name.lower() == "binance":
            continue
        if _key in _binance_symbol_keys:
            raise RuntimeError(
                f"ambiguous current-price cache symbol {_symbol}: Binance and "
                f"{_provider_name}"
            )
        _previous = _nb_provider_names.get(_key)
        if _previous is not None and _previous.lower() != _provider_name.lower():
            raise RuntimeError(
                f"conflicting providers for {_symbol}: {_previous} and {_provider_name}"
            )
        _nb_provider_names[_key] = _provider_name
        _nb_syms.append(_symbol)
    _nb_syms = list(dict.fromkeys(_nb_syms))
    _trend_syms = list(dict.fromkeys(list(sym.symbols) + _nb_syms))
    # Register non-Binance Price24 managers before the generic loop so trend computation
    # can find a raw buffer for every symbol.
    if _nb_syms:
        # Create the venue-bound source before Price24 can request initialization data.
        CacheFactory.get(
            "CurrentPrice",
            symbols=_trend_syms,
            provider_names=_nb_provider_names,
        )
        CacheFactory.get("Price24", symbols=_trend_syms)
        builtins.print(f"[cacheManager] instant-trend extended with non-Binance: {_nb_syms}")
        # LONGTREND_NONBINANCE optionally starts accumulating sparse history and long-trend
        # data. It is disabled by default until enough lookback data exists.
        if LONGTREND_NONBINANCE:
            CacheFactory.get("Price", symbols=_trend_syms)
            CacheFactory.get("PriceLongTrend", symbols=_trend_syms)
            builtins.print(f"[cacheManager] non-Binance long trend enabled: {_nb_syms}")

    for name, config in CacheFactory._CONFIG.items():
        cache = get_cache_manager(name, start_sync=False)
        interval = config["sync_ts"]()  # Resolve the synchronization interval.

        if isinstance(cache, dict):
            # Price and Price24 are dictionaries keyed by symbol.
            for manager in cache.values():
                threads.append(manager.periodic_sync(interval))
        else:
            threads.append(cache.periodic_sync(interval))
        CacheFactory._sync_started.add(name)

    # Trend chain: market-data WebSocket to CurrentPrice, Cache24, and InstantTrend.
    # Full calculation here keeps the shared snapshot current independently of tradeall.
    try:
        from binance_api import bapi_ws
        _trend_cpm = get_current_price_manager(
            ws_manager=bapi_ws.get_ws_manager(),
            symbols=_trend_syms,
            sync_ts=0.8,
            provider_names=_nb_provider_names,
        )
        _trend_cache24 = CacheFactory.get("Price24")
        _trend_mgr = get_short_trend_manager(symbols=_trend_syms, writer=True)   # Sole file writer.
        _trend_mgr.start_computation(_trend_cache24, _trend_cpm, run_full_eval=True)
        print("⚙️ cacheManager: full trend computation started (cache_instant_trend.json).")
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
