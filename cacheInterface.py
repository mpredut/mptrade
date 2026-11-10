"""Pure abstract base contract (Level 1) for transactional cache managers.

Provides ``CacheManagerInterface(ABC)``, which encapsulates in-memory state,
transactional ``.meta`` + ``.previous`` generation persistence, generic JSON
snapshots, generic JSONL append/compaction/rotation, price pub/sub, polling,
and synchronization thread lifecycle without any domain-specific branches.
"""

from __future__ import annotations

import builtins
from abc import ABC, abstractmethod
from collections import defaultdict
import json
import os
import threading
import time
from typing import Any, Callable, Optional
import weakref

from botcore import (
    load_dotenv,
    required_float_env,
    required_int_env,
    resolve_patched_attr,
    safe_float,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_int,
    strict_ascii_int,
)
import cacheintegrity as _cache_integrity
from cachepaths import (
    clean_symbol_key,
    migrate_legacy_filename_if_needed,
)
from lock import FileLock
from providers.base import (
    _parse_bool_flag,
    extract_timestamp_ms,
    normalize_timestamp_ms,
)
from state_io import (
    atomic_gzip_file,
    atomic_snapshot_file,
    atomic_text_writer,
    atomic_write_json,
    commit_file_and_meta,
    discard_generation_file,
    generation_lock_path,
    meta_file_path,
    previous_generation_path,
    prune_rotated_archives,
    read_json_dict,
    restore_previous_generation,
    truncate_file_bytes,
)
import utils as u

_CONFIG_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv()

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

_CACHE_META_SCHEMA_VERSION = _cache_integrity.CACHE_META_SCHEMA_VERSION

_ORIG_BASE_ATTRS = {
    "time": time,
    "atomic_write_json": atomic_write_json,
    "atomic_snapshot_file": atomic_snapshot_file,
}


def _cm_override(name: str, default=None):
    """Resolve an attribute from ``cacheInterface`` or ``cacheManager`` when patched in tests."""
    orig = _ORIG_BASE_ATTRS.get(name)
    local_val = globals().get(name, default)
    if orig is not None and local_val is not orig:
        return local_val
    return resolve_patched_attr(
        name,
        local_val,
        defaults=(orig,) if orig is not None else (),
        modules=("cacheManager", "cacheaccount", "cacheprice"),
    )


def _now_time() -> float:
    t_mod = _cm_override("time", time)
    return t_mod.time()


def _monotonic_time() -> float:
    t_mod = _cm_override("time", time)
    mono_fn = getattr(t_mod, "monotonic", None)
    return mono_fn() if callable(mono_fn) else time.monotonic()


def _strict_int_field(val, *, min_value: int = 1) -> Optional[int]:
    """Validate and coerce an integer field without accepting booleans or fractional floats."""
    return strict_ascii_int(val, default=None, min_value=min_value)


def _norm_symbol_keys(symbol) -> tuple[str, str]:
    """Return ``(uppercase_trimmed, delimiter_stripped_key)`` for ``symbol``."""
    sym_u = "" if isinstance(symbol, bool) else str(symbol or "").strip().upper()
    return sym_u, clean_symbol_key(sym_u)


def _coerce_symbol_list(symbols) -> list[str]:
    """Normalize a single symbol string or iterable into a clean list of symbol strings."""
    if isinstance(symbols, str):
        return [symbols.strip()] if symbols.strip() else []
    if symbols is None or isinstance(symbols, (bool, bytes)):
        return []
    try:
        return [
            str(s).strip()
            for s in symbols
            if s is not None and not isinstance(s, bool) and str(s).strip()
        ]
    except TypeError:
        return []


def _jsonl_line(symbol, item) -> str:
    """Serialize one ``(symbol, item)`` pair as a compact JSONL line."""
    return json.dumps({"s": symbol, "i": item}, separators=(",", ":")) + "\n"


class CacheManagerInterface(ABC):
    """Independent base contract (Level 1) for transactional cache persistence and polling."""

    _live_instances = weakref.WeakSet()
    _filter_fetchtimes_to_symbols = False
    FILE_PREFIX: Optional[str] = None
    FILE_EXT: Optional[str] = None
    DEFAULT_FILENAME: Optional[str] = None

    @classmethod
    def disk_filename_for_symbol(cls, symbol: str = "") -> str:
        """Return the canonical cache filename for ``symbol`` owned by this concrete cache class."""
        if cls.DEFAULT_FILENAME:
            return cls.DEFAULT_FILENAME
        if cls.FILE_PREFIX and cls.FILE_EXT:
            return f"{cls.FILE_PREFIX}{symbol}.{cls.FILE_EXT}"
        return ""

    # ── Periodically enforced retention and rotation policy for append caches ──
    RETENTION_DAYS = CM_RETENTION_DAYS
    MAX_FILE_BYTES = CM_APPEND_MAX_FILE_BYTES
    RETENTION_CHECK_INTERVAL_SEC = CM_RETENTION_CHECK_INTERVAL_SEC
    ROTATE_KEEP_FRACTION = CM_ROTATE_KEEP_FRACTION
    ROTATE_ARCHIVE_COUNT = CM_ROTATE_ARCHIVE_COUNT
    RESYNC_INTERVAL_SEC = CM_RESYNC_INTERVAL_SEC
    DEDUP_WINDOW = CM_DEDUP_WINDOW

    _default_api_client: Any = None
    _poll_predicate: Optional[Callable[[str], bool]] = None
    _ws_resync_clear_hook: Optional[Callable[[str], None]] = None

    @classmethod
    def bind_runtime_services(
        cls,
        *,
        api_client: Any = None,
        poll_predicate: Optional[Callable[[str], bool]] = None,
        ws_resync_clear_hook: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Bind optional runtime services (API client, poll predicate, WS resync clear hook) from factory/daemon."""
        if api_client is not None:
            cls._default_api_client = api_client
        if callable(poll_predicate):
            cls._poll_predicate = staticmethod(poll_predicate)
        if callable(ws_resync_clear_hook):
            cls._ws_resync_clear_hook = staticmethod(ws_resync_clear_hook)

    def __init__(
        self,
        sync_ts,
        symbols,
        filename,
        append_mode=True,
        api_client=None,
        append_persist=False,
        read_only=False,
    ):
        self._live_instances.add(self)
        self.cls_name = self.__class__.__name__
        self.sync_ts = sync_ts
        self.symbols = _coerce_symbol_list(symbols)
        self.filename = u.cache_path(filename)
        self.append_mode = _parse_bool_flag(append_mode, default=True)
        self.api_client = (
            api_client
            if api_client is not None
            else _cm_override("api", self._default_api_client)
        )
        self.append_persist = _parse_bool_flag(append_persist, default=False)
        self.read_only = _parse_bool_flag(read_only, default=False)
        self._persisted_counts = {}
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
        self._last_disk_sync_max_ts = 0

        self.days_back = 30

        self.cache = {}
        self.fetchtime_per_symbol = {}
        self._rest_sync_cursor_per_symbol = {}

        self.thread = None
        self._stop_event = threading.Event()
        self.save_state = False
        if not hasattr(self, "_first_sleep"):
            self._first_sleep = False
        self.lock = threading.RLock()

        if not hasattr(self, "_price_subscribers"):
            self._price_subscribers = []

        self.fallback_time_default = (
            int(_now_time() * 1000) - self.days_back * 24 * 60 * 60 * 1000
        )

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

    def rebuild_fetchtimes(self):
        """Allow subclasses to provide custom fetch-time reconstruction."""
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
            for symbol in last_times_per_sym:
                last_times_per_sym[symbol] = max(0, last_times_per_sym[symbol] - 60_000)
        if not last_times_per_sym:
            fallback_time_file = self.fallback_time_default
            if os.path.exists(self.filename):
                fallback_time_file = int(os.path.getmtime(self.filename) * 1000) - 60_000
            fallback_time = min(self.fallback_time_default, fallback_time_file)
            return {symbol: fallback_time for symbol in self.symbols}
        return last_times_per_sym

    # ── Polymorphic persistence hooks for specialized subclasses ─────────────

    def _on_stale_memory_detected_locked(self) -> None:
        """Hook invoked under ``self.lock`` when disk state is newer than memory."""

    def _before_snapshot_commit_locked(self, persisted_max_ts: int) -> None:
        """Hook invoked under ``self.lock`` before writing a full JSON snapshot."""

    def _filter_reloaded_items_locked(self, items: dict) -> dict:
        """Hook invoked under ``self.lock`` to filter reloaded snapshot items."""
        return items

    def _should_compact_before_append_locked(self) -> bool:
        """Return whether ``_save_jsonl_append`` should compact because memory shrank."""
        return any(
            self._persisted_counts.get(symbol, 0) > len(items)
            for symbol, items in self.cache.items()
        )

    def load_state(self):
        migrate_legacy_filename_if_needed(self.filename)
        print(f"[{self.cls_name}][Info] Load state from {self.filename} ...")
        if self.append_persist:
            self._migrate_legacy_json_if_needed()
            self._load_jsonl()
            if not self.cache and not self.read_only:
                self.query_remote_and_update_cache()
                self.save_state_to_file_if_enabled()
            return
        self._load_snapshot_state()

    def _load_snapshot_state(self):
        backup_path = self._previous_generation_path()
        if os.path.exists(self.filename) or os.path.exists(backup_path):
            try:
                data = read_json_dict(
                    self.filename, backup_path=backup_path, recover_previous=True
                )
                with self.lock:
                    self.cache = data.get("items", {})
                    if not isinstance(self.cache, dict):
                        if isinstance(self.cache, (list, tuple)):
                            self.cache = {
                                sym_k: item for sym_k, item in zip(self.symbols, self.cache)
                            }
                        else:
                            self.cache = {}
                        print(f"[{self.cls_name}][warning] self.cache is not Dict!!!!")

                    self._hydrate_fetchtimes_from_mapping_locked(data)
                    self._last_disk_sync_max_ts = max(
                        self._persisted_max_ts(), self._mem_max_ts()
                    )
                    if not self.cache:
                        print(f"[{self.cls_name}][warning] cache is None")
                    if not self.fetchtime_per_symbol:
                        print(f"[{self.cls_name}][warning] fetchtime_per_symbol is None")
            except Exception as e:
                print(
                    f"[{self.cls_name}][Error] While reading the cache file {self.filename} : {e}"
                )
                if not self.read_only:
                    self.query_remote_and_update_cache()
                    self.save_state_to_file_if_enabled()
        else:
            print(
                f"[{self.cls_name}][Info] File is missing, may be is it first time run. Creating it ...."
            )
            if not self.read_only:
                self.query_remote_and_update_cache()
                self.save_state_to_file_if_enabled()

    def _preserve_rest_sync_cursor_before_update_locked(self, symbol):
        """Anchor the REST cursor before a non-REST update advances fetchtime."""
        if symbol in self._rest_sync_cursor_per_symbol:
            return
        if symbol in self.fetchtime_per_symbol:
            prior_cursor = int(self.fetchtime_per_symbol[symbol] or 0)
        else:
            rebuilt = self._rebuild_fetchtimes()
            prior_cursor = int(rebuilt.get(symbol, self.fallback_time_default) or 0)
        self._rest_sync_cursor_per_symbol[symbol] = max(0, prior_cursor)

    def _advance_symbol_fetchtime_locked(
        self, symbol, now_ms: Optional[int] = None
    ) -> None:
        """Anchor REST cursor, advance ``fetchtime_per_symbol``, and mark account cache dirty."""
        if now_ms is None:
            now_ms = int(_now_time() * 1000)
        self._preserve_rest_sync_cursor_before_update_locked(symbol)
        self.fetchtime_per_symbol[symbol] = max(
            int(self.fetchtime_per_symbol.get(symbol, 0) or 0),
            int(now_ms),
        )
        self._mark_account_cache_dirty_locked()

    def _ensure_fetchtimes_initialized(self, *, sync_rest_cursor: bool = False) -> None:
        """Lazily populate ``fetchtime_per_symbol`` (and optionally ``_rest_sync_cursor_per_symbol``)."""
        with self.lock:
            if not self.fetchtime_per_symbol:
                self.fetchtime_per_symbol = self._rebuild_fetchtimes()
            if sync_rest_cursor:
                for s_sym, rebuilt_ft in self.fetchtime_per_symbol.items():
                    self._rest_sync_cursor_per_symbol.setdefault(s_sym, rebuilt_ft)

    # ── Memory/disk resilience: freshness, overwrite guard, and resync ────────
    @staticmethod
    def _safe_pos_ts_ms(val) -> int:
        return safe_pos_int(val, default=0)

    def _mem_max_ts(self):
        """Return memory freshness as the latest fetch time in milliseconds."""
        if not self.fetchtime_per_symbol:
            return 0
        try:
            return max(
                (self._safe_pos_ts_ms(v) for v in self.fetchtime_per_symbol.values()),
                default=0,
            )
        except Exception:
            return 0

    def _persisted_max_ts(self):
        """Read file freshness cheaply from the ``.meta`` sidecar."""
        try:
            meta = read_json_dict(meta_file_path(self.filename))
            return self._safe_pos_ts_ms(meta.get("max_ts", 0))
        except Exception:
            return 0

    def _is_account_cache(self):
        return False

    def _is_canonical_account_cache(self):
        return False

    def _mark_account_cache_dirty_locked(self):
        pass

    def _hydrate_fetchtimes_from_mapping_locked(
        self,
        mapping,
        *,
        filter_symbols: bool = False,
        preserve_if_missing: bool = False,
        coerce_nonneg_int: bool = False,
    ) -> None:
        raw_ft = mapping.get("fetchtime") if isinstance(mapping, dict) else None
        if not isinstance(raw_ft, dict):
            if preserve_if_missing:
                return
            raw_ft = {}

        def _clean(raw_map):
            out = {}
            for s, v in raw_map.items():
                if filter_symbols and s not in self.symbols:
                    continue
                if coerce_nonneg_int:
                    v = safe_nonneg_int(v, default=None)
                    if v is None:
                        continue
                out[s] = v
            return out

        ft_map = _clean(raw_ft)
        rest_cursor = (
            mapping.get("rest_sync_cursor") if isinstance(mapping, dict) else None
        )
        cursor_map = (
            _clean(rest_cursor) if isinstance(rest_cursor, dict) else dict(ft_map)
        )
        self.fetchtime_per_symbol = ft_map
        self._rest_sync_cursor_per_symbol = cursor_map

    def _snapshot_memory_state_locked(self):
        return (
            {symbol: list(items) for symbol, items in self.cache.items()},
            dict(self._persisted_counts),
            dict(self.fetchtime_per_symbol),
            dict(self._rest_sync_cursor_per_symbol),
        )

    def _restore_memory_state_locked(self, state) -> None:
        if state is None:
            return
        (
            self.cache,
            self._persisted_counts,
            self.fetchtime_per_symbol,
            self._rest_sync_cursor_per_symbol,
        ) = state

    def _build_basic_meta_locked(
        self, counts=None, *, sanitize_cursor: bool = False
    ) -> dict:
        counts_map = (
            {s: len(v) for s, v in self.cache.items()}
            if counts is None
            else dict(counts)
        )
        cursor_map = (
            self._sanitized_rest_sync_cursor_locked()
            if sanitize_cursor
            else dict(self._rest_sync_cursor_per_symbol)
        )
        return {
            "fetchtime": dict(self.fetchtime_per_symbol),
            "rest_sync_cursor": cursor_map,
            "counts": counts_map,
        }

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
            fetchtime_map = payload.get("fetchtime")
            payload["max_ts"] = (
                max(
                    (self._safe_pos_ts_ms(v) for v in fetchtime_map.values()),
                    default=0,
                )
                if isinstance(fetchtime_map, dict)
                else 0
            )
            payload["saved_at"] = int(_now_time() * 1000)
            write_json_fn = _cm_override("atomic_write_json", atomic_write_json)
            write_json_fn(meta_file_path(self.filename), payload)
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

    def _sanitized_rest_sync_cursor_locked(self) -> dict[str, int]:
        return {
            str(symbol): max(0, int(cursor))
            for symbol, cursor in self._rest_sync_cursor_per_symbol.items()
            if isinstance(cursor, (int, float))
            and safe_float(cursor, default=None) is not None
        }

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
            return self._save_state_to_file_impl()
        except Exception as e:
            print(
                f"[{self.cls_name}][Error] While saving the cache file {self.filename} / .tmp : {e}"
            )
            return False

    def _save_state_to_file_impl(self) -> bool:
        with self.lock:
            persisted_max_ts = self._persisted_max_ts()
            if persisted_max_ts > self._mem_max_ts():
                builtins.print(
                    f"[{self.cls_name}][resync] file newer than memory -> "
                    f"refusing to overwrite with stale data ({self.filename})"
                )
                self._on_stale_memory_detected_locked()
                return False
            if self.append_persist:
                return self._save_jsonl_append()
            self._before_snapshot_commit_locked(persisted_max_ts)
            metadata = self._build_basic_meta_locked(sanitize_cursor=True)
            payload = {
                "items": self.cache,
                "fetchtime": metadata["fetchtime"],
                "rest_sync_cursor": metadata["rest_sync_cursor"],
            }
            write_json_fn = _cm_override("atomic_write_json", atomic_write_json)
            if not self._commit_file_and_meta_locked(
                lambda: write_json_fn(self.filename, payload, indent=1),
                metadata,
            ):
                return False
            self._last_disk_sync_max_ts = max(
                self._persisted_max_ts(), self._mem_max_ts()
            )
            print(f"[{self.cls_name}][info] Save cache to file {self.filename}")
            return True

    def _with_generation_lock_bool(
        self, fn, label: str, *, mark_rewrite_on_error: bool = False
    ) -> bool:
        try:
            with self._generation_lock():
                return fn()
        except Exception as exc:
            if mark_rewrite_on_error:
                self._legacy_jsonl_needs_rewrite = True
            print(f"[{self.cls_name}][Error] lock {label} {self.filename}: {exc}")
            return False

    def _reload_from_disk(self):
        """Reload the cache when another process has written a newer file."""
        if self.append_persist:
            self._load_jsonl()
            return
        self._reload_snapshot_from_disk()

    def _reload_snapshot_from_disk(self):
        backup_path = self._previous_generation_path()
        if os.path.exists(self.filename) or os.path.exists(backup_path):
            try:
                data = read_json_dict(
                    self.filename, backup_path=backup_path, recover_previous=True
                )
                with self.lock:
                    items = data.get("items", {})
                    if isinstance(items, dict):
                        self.cache = self._filter_reloaded_items_locked(items)
                    self._hydrate_fetchtimes_from_mapping_locked(
                        data, filter_symbols=self._filter_fetchtimes_to_symbols
                    )
                    self._last_disk_sync_max_ts = max(
                        self._persisted_max_ts(), self._mem_max_ts()
                    )
                    self._account_cache_dirty = False
            except Exception as e:
                print(f"[{self.cls_name}][Error] reload {self.filename}: {e}")

    def resync_mem_file(self):
        """Periodically reload newer disk state or persist newer memory state."""
        file_ts = self._persisted_max_ts()
        mem_ts = self._mem_max_ts()
        if file_ts > mem_ts:
            builtins.print(
                f"[{self.cls_name}][resync] file is newer -> reloading ({self.filename})"
            )
            self._reload_from_disk()
        elif mem_ts > file_ts and self.save_state:
            self.save_state_to_file_if_enabled()

    # ── JSONL append persistence for append-only caches ──────────────────────
    def _migrate_legacy_json_if_needed(self):
        """Migrate a legacy ``.json`` cache file into ``.jsonl`` format."""
        if os.path.exists(self.filename) or not self.filename.endswith(".jsonl"):
            return
        legacy = self.filename[:-1]
        if not os.path.exists(legacy):
            return
        try:
            payload = read_json_dict(legacy)
            items = payload.get("items", {})
            if not isinstance(items, dict):
                raise ValueError("legacy cache items must be a dictionary")
            with self.lock:
                self.cache = items
                self._hydrate_fetchtimes_from_mapping_locked(payload)
                self._persisted_counts = {}
            if not self.compact_jsonl():
                raise ValueError(
                    "legacy cache could not be converted into a durable JSONL generation"
                )
            builtins.print(
                f"[{self.cls_name}][migration] created {self.filename} from {legacy}; "
                "the legacy file was retained for rollback"
            )
        except (OSError, ValueError, TypeError) as exc:
            raise RuntimeError(f"Cannot migrate legacy cache {legacy}: {exc}") from exc

    def _save_jsonl_append(self):
        """Append only items added since the last flush without rewriting the file."""
        previous_counts = {}
        candidate_counts = {}
        initial_size = None
        try:
            with self.lock:
                if self._should_compact_before_append_locked():
                    return self.compact_jsonl()
                previous_counts = dict(self._persisted_counts)
                candidate_counts = dict(previous_counts)
                with open(self.filename, "a+", encoding="utf-8") as f:
                    initial_size = os.fstat(f.fileno()).st_size
                    for symbol, items in self.cache.items():
                        start = self._persisted_counts.get(symbol, 0)
                        if start > len(items):
                            start = 0
                        for item in items[start:]:
                            f.write(_jsonl_line(symbol, item))
                        candidate_counts[symbol] = len(items)
                    f.flush()
                    os.fsync(f.fileno())
                self._persisted_counts = candidate_counts
                return self._write_meta(self._build_basic_meta_locked(candidate_counts))
        except Exception as e:
            self._persisted_counts = previous_counts
            if initial_size is not None:
                self._truncate_failed_jsonl_append(initial_size)
            print(f"[{self.cls_name}][Error] append JSONL {self.filename}: {e}")
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

    def _previous_generation_path(self):
        return previous_generation_path(self.filename)

    def _generation_lock(self):
        """Serialize cache data/manifest publication and canonical recovery."""
        os.makedirs(os.path.dirname(os.path.abspath(self.filename)), exist_ok=True)
        return FileLock(generation_lock_path(self.filename))

    def _restore_previous_generation(self, backup_path=None):
        """Atomically restore data matching the still-current manifest."""
        restore_previous_generation(self.filename, backup_path=backup_path)

    def _load_jsonl(self):
        """Load generic JSONL state and recover ``.previous`` if interrupted."""
        legacy_metadata = {}
        metaf = self.filename + ".meta"
        if os.path.exists(metaf):
            try:
                legacy_metadata = read_json_dict(metaf)
            except (OSError, ValueError, TypeError):
                legacy_metadata = {}

        with self.lock:
            self.cache = {}
            had_invalid_line = False
            backup_path = self._previous_generation_path()
            if not os.path.exists(self.filename) and os.path.exists(backup_path):
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
                            if (
                                not isinstance(rec, dict)
                                or not isinstance(rec.get("s"), str)
                                or rec.get("i") is None
                            ):
                                raise ValueError("invalid JSONL record")
                            self.cache.setdefault(rec["s"], []).append(rec["i"])
                        except Exception:
                            had_invalid_line = True
                            continue
            self._legacy_jsonl_needs_rewrite = had_invalid_line
            if not had_invalid_line and os.path.exists(backup_path):
                discard_generation_file(backup_path)
            self._persisted_counts = {s: len(v) for s, v in self.cache.items()}
            self._hydrate_fetchtimes_from_mapping_locked(
                legacy_metadata, preserve_if_missing=True
            )

    def compact_jsonl(self):
        """Rewrite JSONL from memory after full in-memory and on-disk deduplication."""
        if not self.append_persist:
            return False
        original_state = None
        try:
            with self.lock:
                original_state = self._snapshot_memory_state_locked()
                for symbol, items in list(self.cache.items()):
                    seen = set()
                    deduped = []
                    for item in items:
                        k = json.dumps(item, sort_keys=True)
                        if k not in seen:
                            seen.add(k)
                            deduped.append(item)
                    self.cache[symbol] = deduped
                metadata = self._build_basic_meta_locked()

                def _write_compacted():
                    with atomic_text_writer(self.filename) as f:
                        for symbol, items in self.cache.items():
                            for item in items:
                                f.write(_jsonl_line(symbol, item))

                if not self._commit_file_and_meta_locked(_write_compacted, metadata):
                    self._restore_memory_state_locked(original_state)
                    return False
                self._persisted_counts = dict(metadata["counts"])
                return True
        except Exception as e:
            self._restore_memory_state_locked(original_state)
            print(f"[{self.cls_name}][Error] compact JSONL {self.filename}: {e}")
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
        """Perform weekly maintenance for every append-mode cache."""
        if not self.append_mode:
            return
        cutoff_ms = int((_now_time() - self.RETENTION_DAYS * 24 * 3600) * 1000)
        changed = False
        with self.lock:
            original_state = self._snapshot_memory_state_locked()
            original_dirty = self._account_cache_dirty
            for symbol, items in list(self.cache.items()):
                kept = [
                    it for it in items if self._entry_timestamp_ms(it) >= cutoff_ms
                ]
                if len(kept) != len(items):
                    self.cache[symbol] = kept
                    changed = True
            if changed:
                self._mark_account_cache_dirty_locked()
        if changed:
            builtins.print(
                f"[{self.cls_name}][maintain] pruning >{self.RETENTION_DAYS}d from {self.filename}"
            )
            if self.save_state:
                persisted = (
                    self.compact_jsonl()
                    if self.append_persist
                    else self.save_state_to_file()
                )
                if not persisted:
                    with self.lock:
                        self._restore_memory_state_locked(original_state)
                        self._account_cache_dirty = original_dirty
                    return False
        if not self.append_persist or not self.save_state:
            return
        try:
            if (
                os.path.exists(self.filename)
                and os.path.getsize(self.filename) > self.MAX_FILE_BYTES
            ):
                self._rotate_keep_latest()
        except OSError:
            pass

    def _compress_and_prune_archive(self, archive: str) -> str:
        """Compress a rotated archive to ``.gz`` and prune excess archives."""
        try:
            archive = atomic_gzip_file(archive, archive + ".gz", remove_source=True)
        except OSError as exc:
            builtins.print(
                f"[{self.cls_name}][maintain] archive compression failed: {exc}"
            )
        prune_rotated_archives(self.filename, self.ROTATE_ARCHIVE_COUNT)
        return archive

    def _trim_cache_to_keep_fraction_locked(self) -> None:
        for symbol, items in self.cache.items():
            keep_n = max(1, int(len(items) * self.ROTATE_KEEP_FRACTION))
            self.cache[symbol] = items[-keep_n:]

    def _finalize_rotation(self, archive: str) -> bool:
        archive = self._compress_and_prune_archive(archive)
        builtins.print(
            f"[{self.cls_name}][maintain] ROTATION: archived -> {archive}, "
            f"kept the last {int(self.ROTATE_KEEP_FRACTION * 100)}%"
        )
        return True

    def _rotate_keep_latest(self):
        """Archive the current file and retain each symbol's latest configured fraction."""
        with self.lock:
            archive = f"{self.filename}.{int(_now_time())}.archive"
            snapshot_fn = _cm_override("atomic_snapshot_file", atomic_snapshot_file)
            try:
                snapshot_fn(self.filename, archive)
            except OSError as e:
                builtins.print(f"[{self.cls_name}][maintain] archiving failed: {e}")
                return False
            original_state = self._snapshot_memory_state_locked()
            self._trim_cache_to_keep_fraction_locked()
            self._persisted_counts = {}
            try:
                compacted = bool(self.compact_jsonl())
            except Exception as e:
                builtins.print(
                    f"[{self.cls_name}][maintain] rotation compaction failed: {e}"
                )
                compacted = False
            if not compacted:
                self._restore_memory_state_locked(original_state)
                try:
                    self._restore_previous_generation(archive)
                except OSError as restore_err:
                    builtins.print(
                        f"[{self.cls_name}][maintain] failed to restore active file from {archive}: {restore_err}"
                    )
                builtins.print(
                    f"[{self.cls_name}][maintain] rotation aborted; restored active file and in-memory state"
                )
                return False
            return self._finalize_rotation(archive)

    @abstractmethod
    def get_remote_items(self, symbol, startTime):
        """Fetch remote items; subclasses must implement this method."""

    def filter_new_items(self, cache_items, new_items):
        seen = {json.dumps(it, sort_keys=True) for it in cache_items}
        unique_new = []
        for item in new_items:
            key = json.dumps(item, sort_keys=True)
            if key not in seen:
                seen.add(key)
                unique_new.append(item)
        return unique_new

    @staticmethod
    def _coerce_item_list(new_items) -> list:
        if isinstance(new_items, dict):
            return [new_items]
        if isinstance(new_items, list):
            return new_items
        return list(new_items) if isinstance(new_items, tuple) else [new_items]

    def update_cache_per_symbol(self, symbol, new_items):
        if new_items is None:
            return
        current_time = int(_now_time() * 1000)
        if self.append_mode:
            new_items = self._coerce_item_list(new_items)

        count_new_items = (
            len(new_items) if isinstance(new_items, (list, tuple, dict)) else 1
        )
        print(f"[{self.cls_name}][Info] {symbol}:  new_items {new_items}")

        with self.lock:
            self.cache.setdefault(symbol, [])
            if self.append_mode:
                cache_copy = list(self.cache.get(symbol, []))[-self.DEDUP_WINDOW :]
                new_items = self.filter_new_items(cache_copy, new_items)
                print(
                    f"[{self.cls_name}][Info] {symbol}:  out of {count_new_items} keeping only {len(new_items)}"
                )
                new_items = [item for item in new_items if item is not None]
                if not new_items:
                    return
                self.cache[symbol].extend(new_items)
            else:
                self.cache[symbol] = (
                    new_items if isinstance(new_items, list) else [new_items]
                )

            self._advance_symbol_fetchtime_locked(symbol, current_time)

        print(f"[{self.cls_name}][Info] {symbol}: Added {len(new_items)} new items.")

    def _persist_items(self, symbol, new_items):
        """Persist one symbol to the cache by default."""
        self.update_cache_per_symbol(symbol, new_items)

    def _resolve_remote_query_start_time(self, symbol: str, fallback_ft):
        return fallback_ft

    def _on_symbol_remote_sync_completed(
        self, symbol: str, request_high_water: int
    ) -> None:
        pass

    def _on_complete_remote_sync_cycle(
        self, account_cycle_high_water: Optional[int]
    ) -> None:
        pass

    def _clear_ws_rest_resync(self) -> None:
        hook = self._ws_resync_clear_hook
        if callable(hook):
            hook(self.cls_name)

    def query_remote_and_update_cache(self):
        self._ensure_fetchtimes_initialized(sync_rest_cursor=True)

        account_cycle_high_water = None
        for symbol in list(self.symbols):
            request_high_water = int(_now_time() * 1000)
            fallback_ft = self.fetchtime_per_symbol.get(
                symbol, self.fallback_time_default
            )
            startTime = self._resolve_remote_query_start_time(symbol, fallback_ft)
            if account_cycle_high_water is None:
                account_cycle_high_water = request_high_water
            new_items = self.get_remote_items(symbol=symbol, startTime=startTime)
            if new_items is None:
                raise RuntimeError(
                    f"{self.cls_name} returned no synchronization result for {symbol}"
                )
            if not new_items:
                print(
                    f"[{self.cls_name}][Info] {symbol}:  No remote items starting with {u.timestampToTime(startTime)} "
                )
            else:
                self._persist_items(symbol, new_items)
            self._on_symbol_remote_sync_completed(symbol, request_high_water)
        self._on_complete_remote_sync_cycle(account_cycle_high_water)
        self._clear_ws_rest_resync()
        return True

    def on_items_update(self, symbol, items):
        print(f"[{self.cls_name}][Info] {symbol}: WS Items updated to {items}")
        self._ensure_fetchtimes_initialized(sync_rest_cursor=True)
        self._persist_items(symbol, items)

    def _should_poll(self):
        """Decide whether the synchronization loop polls the API."""
        poll_fn = _cm_override("_should_poll_for_manager", self._poll_predicate)
        if callable(poll_fn):
            return bool(poll_fn(self.cls_name))
        return True

    def _publish_periodic_sync_health(self, sync_complete, persisted) -> None:
        pass

    def _persist_periodic_state(self, sync_complete):
        """Persist one cycle and publish periodic sync health when applicable."""
        persisted = self.save_state_to_file_if_enabled()
        self._publish_periodic_sync_health(sync_complete, persisted)
        return persisted

    def periodic_sync(self, sync_ts=None, save_state=True):
        save_state = _parse_bool_flag(save_state, default=True)
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
                if save_state:
                    self.save_state = True
                return self.thread

            self.save_state = save_state
            self._stop_event.clear()

            def run():
                if self._first_sleep and self._stop_event.wait(self.sync_ts):
                    return
                maint_stagger = (
                    safe_nonneg_float(
                        getattr(self, "_maint_stagger_sec", 0.0), default=0.0
                    )
                    or 0.0
                )
                last_maint = _now_time() + maint_stagger
                last_resync = _now_time()
                while not self._stop_event.is_set():
                    try:
                        sync_complete = False
                        if self._should_poll():
                            sync_complete = self.query_remote_and_update_cache()
                        else:
                            self._ensure_fetchtimes_initialized(sync_rest_cursor=True)
                        self._persist_periodic_state(sync_complete)
                        if (_now_time() - last_resync) > self.RESYNC_INTERVAL_SEC:
                            self.resync_mem_file()
                            last_resync = _now_time()
                        if (
                            self.append_mode
                            and (_now_time() - last_maint)
                            > self.RETENTION_CHECK_INTERVAL_SEC
                        ):
                            self.maintain_append_persist()
                            last_maint = _now_time()
                    except Exception as _e:
                        builtins.print(
                            f"[{self.cls_name}] synchronization error (continuing): {_e}"
                        )
                    wait_sec = (
                        self._next_sync_wait_sec()
                        if hasattr(self, "_next_sync_wait_sec")
                        else self.sync_ts
                    )
                    if self._stop_event.wait(wait_sec):
                        break

            thread = threading.Thread(target=run, name=self.cls_name, daemon=True)
            thread.daemon = True
            thread.start()
            self.thread = thread
            return self.thread

    def request_stop(self) -> None:
        """Signal the synchronization loop to stop without blocking."""
        self._stop_event.set()

    def is_running(self) -> bool:
        """Return True if a background synchronization thread is active."""
        thread = self.thread
        return thread is not None and thread.is_alive()

    def shutdown(self, timeout=5.0):
        """Deterministically stop the synchronization loop started by ``periodic_sync``."""
        self.request_stop()
        eff_timeout = safe_nonneg_float(timeout, default=5.0)
        if eff_timeout is None:
            eff_timeout = 5.0
        thread = self.thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=eff_timeout)
        stopped = thread is None or not thread.is_alive()
        if stopped:
            self.thread = None
        return stopped

    @classmethod
    def shutdown_all_instances(cls, timeout=5.0):
        instances = list(cls._live_instances)
        for manager in instances:
            manager.request_stop()
        eff_timeout = safe_nonneg_float(timeout, default=5.0)
        if eff_timeout is None:
            eff_timeout = 5.0
        deadline = _monotonic_time() + eff_timeout
        results = [
            manager.shutdown(timeout=max(0.0, deadline - _monotonic_time()))
            for manager in instances
        ]
        return all(results) if results else True
