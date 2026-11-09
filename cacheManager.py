import bisect
import builtins
from collections import defaultdict, deque
from datetime import datetime
import json
import os
import sys
import threading
import time
from typing import Optional
import uuid

from binance_api import bapi as api
from binance_api import bapi_allorders as apiorders
from binance_api import bapi_ws_bridge as _bapi_ws_bridge
import binance_cache_health as account_cache_health
from botcore import (
    is_us_rth_open,
    load_dotenv as _load_dotenv,
    safe_nonneg_float,
    safe_pos_float,
    safe_pos_int,
    single_instance,
    touch_module_heartbeat,
)
from cachebase import (
    AccountCacheManagerBase,
    CM_APPEND_MAX_FILE_BYTES,
    CM_DEDUP_WINDOW,
    CM_LONG_ARCHIVE_FLUSH_SEC,
    CM_LONG_ARCHIVE_MEMORY_ROWS,
    CM_LONG_ARCHIVE_MONTHS,
    CM_LONG_ARCHIVE_SAMPLE_SEC,
    CM_RESYNC_INTERVAL_SEC,
    CM_RETENTION_CHECK_INTERVAL_SEC,
    CM_RETENTION_DAYS,
    CM_ROTATE_ARCHIVE_COUNT,
    CM_ROTATE_KEEP_FRACTION,
    CacheAccountManagerBase,
    CacheManagerInterface,
    CachePriceSeriesMixin,
    WS_EVENT_LOG_ENABLED,
    WS_LOSS_TIMEOUT_SEC,
    WS_ONLY_MODE,
    _ACCOUNT_CACHE_CLASSES,
    _CACHE_META_SCHEMA_VERSION,
    _CONFIG_ROOT,
    _ORDER_CACHE_CLASSES,
    _PriceSeriesCacheMixin,
    _TRADE_CACHE_CLASSES,
    _cm_override,
    _coerce_symbol_list,
    _jsonl_line,
    _monotonic_time,
    _norm_symbol_keys,
    _now_time,
    _strict_int_field,
)
import cacheintegrity as _cache_integrity
from cachepaths import (
    canonical_base_asset,
    canonical_symbol_key,
    clean_symbol_key,
    compute_venue_nav_breakdown,
    migrate_legacy_filename_if_needed as _migrate_legacy_filename_if_needed,
    normalize_venue_name,
)
import cachetrend as _cache_trend
from cachetrend import (
    CM_DYNAMIC_WINDOW_MIN_SEC,
    CM_DYNAMIC_WINDOW_MAX_SEC,
    CM_PUBLISHED_TREND_WINDOWS_SEC,
    CacheLongTrend,
    CacheLongTrendManager,
    CacheShortTrend,
    CacheShortTrendManager,
    HyperliquidPriceStream,
    KrakenPriceStream,
    _NB_FETCH_DEADLINE_SEC,
    _NonBinanceTrendPoller,
    _fetch_price_with_deadline,
    _start_nonbinance_trend_poller,
)
import instrument_registry as sym
from instrument_registry import casefold_get, is_equity_symbol, tracked_price_instruments
from lock import FileLock
import providers.market_api as _market_api
from providers.base import (
    _parse_bool_flag,
    extract_order_price,
    extract_order_qty,
    extract_order_side,
    extract_timestamp_ms,
    extract_timestamp_s,
    live_price_fetch_context,
    normalize_side,
    normalize_timestamp_ms,
)
from state_io import (
    atomic_gzip_file,
    atomic_snapshot_file,
    atomic_text_writer as atomic_write,
    atomic_write_json,
    commit_file_and_meta,
    discard_generation_file,
    generation_lock_path,
    meta_file_path,
    previous_generation_path,
    prune_rotated_archives,
    read_json_dict,
    read_recovering_previous,
    read_tail_lines,
    restore_previous_generation,
    stage_previous_generation,
    truncate_file_bytes,
)
import utils as u

_load_dotenv()

_ws_health_lock = _bapi_ws_bridge._ws_health_lock
_ws_available = _bapi_ws_bridge._ws_available
_ws_last_event_ts = _bapi_ws_bridge._ws_last_event_ts
_ws_is_healthy = _bapi_ws_bridge._ws_is_healthy
_ws_needs_rest_resync = _bapi_ws_bridge._ws_needs_rest_resync


def _mark_ws_available(value):
    global _ws_available
    _ws_available = _bapi_ws_bridge.mark_ws_available(value)


def _mark_ws_event_received():
    global _ws_last_event_ts, _ws_is_healthy
    _ws_last_event_ts = _bapi_ws_bridge.mark_ws_event_received()
    _ws_is_healthy = _bapi_ws_bridge._ws_is_healthy


def _mark_ws_unhealthy():
    global _ws_is_healthy
    _bapi_ws_bridge.mark_ws_unhealthy()
    _ws_is_healthy = _bapi_ws_bridge._ws_is_healthy


def _should_poll_for_manager(cls_name):
    return _bapi_ws_bridge.should_poll_for_manager(
        cls_name,
        ws_only_mode=WS_ONLY_MODE,
        ws_available=_ws_available,
        ws_is_healthy=_ws_is_healthy,
    )


# ── Domain Cache Implementations ──────────────────────────────────────────────


class CacheTradeManager(CacheAccountManagerBase):
    """Account trade fill cache backed by versioned append-only JSONL and schema-v2 manifests."""

    _is_trade_account_cache = True

    def __init__(self, sync_ts, symbols, filename, api_client=api, read_only=False):
        super().__init__(
            sync_ts,
            symbols,
            filename,
            append_mode=True,
            api_client=api_client,
            append_persist=True,
            read_only=read_only,
        )

    @staticmethod
    def _is_valid_item(trade):
        return CacheAccountManagerBase._is_valid_trade_fill_row(trade)

    def _save_state_to_file_impl(self) -> bool:
        if self.append_persist:
            return self._save_account_trade_append()
        return super()._save_state_to_file_impl()

    def _migrate_legacy_json_if_needed(self):
        """Migrate legacy ``.json`` Trade cache under the generation lock."""
        with self._generation_lock():
            return self._migrate_legacy_json_generation_locked()

    def _migrate_legacy_json_generation_locked(self):
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
            if not self._compact_account_trade_jsonl_generation_locked():
                raise ValueError(
                    "legacy cache could not be converted into a durable JSONL generation"
                )
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
        return self._save_account_trade_append()

    def _save_account_trade_append(self):
        """Commit Trade data and its manifest under the generation lock."""
        return self._with_generation_lock_bool(
            self._save_account_trade_append_if_current_generation_locked,
            "Trade append",
        )

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
                    f"refusing to overwrite with stale data ({self.filename})"
                )
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
                for symbol, items in self.cache.items()
            ):
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
                                or str(item["symbol"]).upper() != str(symbol).upper()
                            ):
                                raise ValueError("invalid Trade row in append delta")
                            handle.write(_jsonl_line(symbol, item).encode("utf-8"))
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
                print(f"[{self.cls_name}][Error] append JSONL {self.filename}: {exc}")
                return False

    def _read_trade_records_strict(self, start, end, *, path=None):
        """Parse only the requested committed JSONL byte range."""
        return _cache_integrity.read_trade_records_range_strict(
            path or self.filename,
            start,
            end,
            is_valid_item_fn=self._is_valid_item,
            reader_reason=self._account_reader_reason(),
        )

    def _read_trade_snapshot_strict(self, metadata, *, path=None):
        return _cache_integrity.read_trade_snapshot_strict(
            path or self.filename,
            metadata,
            is_valid_item_fn=self._is_valid_item,
            reader_reason=self._account_reader_reason(),
        )

    def _read_trade_snapshot_recovering_previous_locked(self, metadata):
        """Read a Trade snapshot against ``metadata``, restoring ``.previous`` if needed."""
        return self._read_snapshot_recovering_previous_locked(
            self._read_trade_snapshot_strict, metadata
        )

    def _remember_loaded_trade_manifest_locked(self, metadata, counts):
        if "fetchtime" in metadata:
            self.fetchtime_per_symbol = dict(metadata["fetchtime"])
        if "rest_sync_cursor" in metadata:
            self._rest_sync_cursor_per_symbol = dict(metadata["rest_sync_cursor"])
        self._persisted_counts = dict(counts)
        self._apply_manifest_identity_locked(metadata, counts)

    def _ensure_account_version(self, required_version: str) -> None:
        self._ensure_trade_version(required_version)

    def _ensure_trade_version(self, required_version):
        with self._generation_lock():
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
                        str(symbol).upper(): {str(int(item["id"])) for item in items}
                        for symbol, items in self.cache.items()
                    }
                    if can_append
                    else {}
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
                    self._discard_generation_file(self._previous_generation_path())
                except (
                    account_cache_health.AccountCacheNotReady,
                    OSError,
                    TypeError,
                    ValueError,
                    json.JSONDecodeError,
                ):
                    if not os.path.exists(self._previous_generation_path()):
                        raise
                    can_append = False
                    candidate_cache, candidate_counts = (
                        self._read_trade_snapshot_recovering_previous_locked(metadata)
                    )
                    records = []
            else:
                candidate_cache, candidate_counts = (
                    self._read_trade_snapshot_recovering_previous_locked(metadata)
                )
                records = []

            if (
                self._load_versioned_account_meta(required_version=required_version)
                != metadata
            ):
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
                        and {s: len(v) for s, v in self.cache.items()}
                        == self._nonzero_counts(old_counts)
                    )
                    if append_still_valid:
                        for symbol, item in records:
                            self.cache.setdefault(symbol, []).append(item)
                        self._remember_loaded_trade_manifest_locked(
                            metadata, candidate_counts
                        )
                        return
                candidate_cache, candidate_counts = (
                    self._read_trade_snapshot_recovering_previous_locked(metadata)
                )
                if (
                    self._load_versioned_account_meta(required_version=required_version)
                    != metadata
                ):
                    raise ValueError("Trade manifest changed during full reload")

            with self.lock:
                self.cache = candidate_cache
                self._remember_loaded_trade_manifest_locked(metadata, candidate_counts)
        except account_cache_health.AccountCacheNotReady:
            raise
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise account_cache_health.AccountCacheNotReady(reason) from exc

    def _load_jsonl(self):
        """Load Trade JSONL under the generation lock."""
        with self._generation_lock():
            return self._load_jsonl_generation_locked()

    def _load_jsonl_generation_locked(self):
        """Load Trade JSONL after acquiring the generation lock."""
        metadata = self._load_versioned_account_meta(optional=True)
        if metadata is not None:
            cache, counts = self._read_trade_snapshot_recovering_previous_locked(
                metadata
            )
            with self.lock:
                self.cache = cache
                self._remember_loaded_trade_manifest_locked(metadata, counts)
            return

        legacy_metadata = {}
        metaf = self.filename + ".meta"
        if os.path.exists(metaf):
            try:
                legacy_metadata = read_json_dict(metaf)
            except (OSError, ValueError, TypeError):
                legacy_metadata = {}

        with self.lock:
            self.cache = {}
            seen_trade_items = defaultdict(dict)
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
                                or not isinstance(rec.get("i"), dict)
                                or not self._is_valid_item(rec["i"])
                                or str(rec["i"]["symbol"]).upper()
                                != str(rec["s"]).upper()
                            ):
                                raise ValueError("invalid legacy Trade row")
                            trade_id = str(int(rec["i"]["id"]))
                            signature = self._trade_semantic_signature(rec["i"])
                            symbol_items = seen_trade_items[str(rec["s"]).upper()]
                            if trade_id in symbol_items:
                                if symbol_items[trade_id] != signature:
                                    raise ValueError(
                                        "conflicting legacy Trade rows for "
                                        f"immutable ID {trade_id}"
                                    )
                                continue
                            symbol_items[trade_id] = signature
                            self.cache.setdefault(rec["s"], []).append(rec["i"])
                        except Exception as exc:
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
                                    for symbol, count in expected_counts.items()
                                )
                                and self._nonzero_counts(expected_counts)
                                == actual_counts
                            )
                            if recoverable_tail:
                                break
                            raise account_cache_health.AccountCacheNotReady(
                                self._account_reader_reason()
                            ) from exc
            self._legacy_jsonl_needs_rewrite = True
            self._persisted_counts = {s: len(v) for s, v in self.cache.items()}
            self._hydrate_fetchtimes_from_mapping_locked(
                legacy_metadata, preserve_if_missing=True
            )

    def compact_jsonl(self):
        if not self.append_persist:
            return False
        return self._compact_account_trade_jsonl()

    def _compact_account_trade_jsonl(self):
        """Rewrite Trade data and its manifest as one cross-process generation."""
        return self._with_generation_lock_bool(
            self._compact_account_trade_jsonl_generation_locked,
            "Trade generation",
            mark_rewrite_on_error=True,
        )

    def _compact_account_trade_jsonl_generation_locked(self):
        """Rewrite Trade data while retaining the prior certified generation."""
        try:
            with self.lock:
                previous_metadata = self._load_versioned_account_meta(optional=True)
                backup_path, had_prior_file = self._stage_prior_account_generation_locked(
                    previous_metadata,
                    self._read_trade_snapshot_recovering_previous_locked,
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
                    candidate_cache[symbol] = self._dedupe_trade_items_by_id(items)
                self.cache = candidate_cache
                self._validate_account_rows_locked()

                counts = {
                    symbol: len(items) for symbol, items in self.cache.items()
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
                                handle.write(_jsonl_line(symbol, item))

                def _build_trade_meta():
                    committed_bytes = os.path.getsize(self.filename)
                    content_digest = self._trade_records_digest(records)
                    stream_id = uuid.uuid4().hex
                    revision = 1
                    version = self._trade_data_version(stream_id, revision)
                    committed_metadata["value"] = self._account_meta_snapshot(
                        data_version=version,
                        stream_id=stream_id,
                        revision=revision,
                        committed_bytes=committed_bytes,
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
                self._remember_account_commit_locked(committed_metadata["value"])
                return True
        except Exception as exc:
            self._legacy_jsonl_needs_rewrite = True
            print(f"[{self.cls_name}][Error] compact JSONL {self.filename}: {exc}")
            return False

    def _rotate_keep_latest(self):
        return self._rotate_account_trade_keep_latest()

    def _rotate_account_trade_keep_latest(self):
        """Publish a compact Trade generation without moving canonical data first."""
        archive = None
        original_state = None
        original_dirty = False
        original_needs_rewrite = False
        published = False
        try:
            with self._generation_lock():
                with self.lock:
                    metadata = self._load_versioned_account_meta()
                    self._read_trade_snapshot_strict(metadata)
                    archive = f"{self.filename}.{int(_now_time())}.archive"
                    atomic_snapshot_file(self.filename, archive)
                    original_state = self._snapshot_memory_state_locked()
                    original_dirty = self._account_cache_dirty
                    original_needs_rewrite = self._legacy_jsonl_needs_rewrite
                    self._trim_cache_to_keep_fraction_locked()
                    self._mark_account_cache_dirty_locked()
                    if not self._compact_account_trade_jsonl_generation_locked():
                        self._restore_memory_state_locked(original_state)
                        self._account_cache_dirty = original_dirty
                        self._legacy_jsonl_needs_rewrite = original_needs_rewrite
                        self._discard_generation_file(archive)
                        builtins.print(
                            f"[{self.cls_name}][maintain] Trade rotation "
                            "aborted because generation publication failed"
                        )
                        return False
                    published = True
            return self._finalize_rotation(archive)
        except Exception as exc:
            if not published and original_state is not None:
                with self.lock:
                    self._restore_memory_state_locked(original_state)
                    self._account_cache_dirty = original_dirty
                    self._legacy_jsonl_needs_rewrite = original_needs_rewrite
            if not published and archive is not None:
                self._discard_generation_file(archive)
            builtins.print(f"[{self.cls_name}][maintain] Trade rotation failed: {exc}")
            return False

    def _persist_periodic_state(self, sync_complete):
        """Persist one cycle without reversing the Trade generation lock order."""
        if self.save_state:
            with self._generation_lock():
                persisted = self._save_account_trade_append_if_current_generation_locked()
                self._publish_periodic_sync_health(sync_complete, persisted)
                return persisted
        return super()._persist_periodic_state(sync_complete)

    def get_remote_items(self, symbol, startTime):
        new_trades = self._paginate_validated_trades(symbol, startTime, "trade")
        print(f"[{self.cls_name}][info] New trades: {len(new_trades)}")
        return new_trades

    def update_cache_per_symbol(self, symbol, new_items):
        """Route direct update_cache_per_symbol calls through full-history ID deduplication."""
        if new_items is None:
            return
        self._persist_items(symbol, self._coerce_item_list(new_items))

    def _persist_items(self, symbol, new_items):
        """Append immutable fills once and reject conflicting duplicate IDs."""
        if not new_items:
            return
        new_items = self._coerce_item_list(new_items)
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
                        for fee_key in ("commission", "commissionAsset", "clientOrderId"):
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
            self._advance_symbol_fetchtime_locked(symbol)

    def get_last_opposite_fill_price(self, symbol, order_type):
        """Return the latest opposite fill price without a time limit."""
        self.sync_account_reader_if_needed()
        if not isinstance(order_type, str):
            return None
        side_u = normalize_side(order_type)
        if side_u not in ("BUY", "SELL"):
            return None
        want_buyer = side_u == "SELL"
        sym_clean = clean_symbol_key(symbol)
        latest_key = None
        latest_price = None
        with self.lock:
            bucket = (
                casefold_get(self.cache, symbol)
                or casefold_get(self.cache, sym_clean)
                or []
            )
            for trade in bucket:
                if (
                    not self._is_valid_item(trade)
                    or clean_symbol_key(trade["symbol"]) != sym_clean
                    or trade["isBuyer"] != want_buyer
                ):
                    continue
                key = (int(trade["time"]), int(trade["id"]))
                if latest_key is None or key > latest_key:
                    latest_key = key
                    latest_price = float(trade["price"])
        return latest_price


CacheTrade = CacheTradeManager


class CacheOrderManager(CacheAccountManagerBase):
    """Account order aggregate cache backed by versioned JSON snapshots and schema-v2 manifests."""

    _is_order_account_cache = True

    def __init__(self, sync_ts, symbols, filename, api_client=api, read_only=False):
        super().__init__(
            sync_ts,
            symbols,
            filename,
            append_mode=True,
            api_client=api_client,
            append_persist=False,
            read_only=read_only,
        )

    @staticmethod
    def _normalize_fill_id(raw_fill_id) -> Optional[str]:
        if raw_fill_id is None or isinstance(raw_fill_id, bool):
            return None
        s = str(raw_fill_id).strip()
        if not s or not s.isascii() or not s.isdigit():
            return None
        return str(int(s))

    def _is_valid_item(self, trade):
        if not isinstance(trade, dict):
            return False
        required_keys = ("orderId", "price", "quantity", "timestamp", "side")
        if not all(key in trade for key in required_keys):
            return False
        legacy_cutoff = trade.get("_legacyCoveredThrough")
        return (
            _strict_int_field(trade["orderId"], min_value=1) is not None
            and safe_pos_float(trade["price"], default=None) is not None
            and safe_pos_float(trade["quantity"], default=None) is not None
            and _strict_int_field(trade["timestamp"], min_value=1) is not None
            and str(trade["side"]).upper() in ("BUY", "SELL")
            and (
                legacy_cutoff is None
                or _strict_int_field(legacy_cutoff, min_value=1) is not None
            )
            and (
                "_fillIds" not in trade
                or (
                    isinstance(trade["_fillIds"], list)
                    and len(
                        {
                            norm_fid
                            for fill_id in trade["_fillIds"]
                            if (norm_fid := self._normalize_fill_id(fill_id)) is not None
                        }
                    )
                    == len(trade["_fillIds"])
                )
            )
        )

    def _read_legacy_order_snapshot_strict(self, legacy_meta, *, path=None):
        return _cache_integrity.read_legacy_order_snapshot_strict(
            path or self.filename, legacy_meta
        )

    def _load_snapshot_state(self):
        self._load_order_state()

    def _load_order_state(self):
        """Load Order cache state strictly when durable data or metadata exists."""
        with self._generation_lock():
            loaded = self._load_order_state_generation_locked()
        if not loaded and not self.read_only:
            print(
                f"[{self.cls_name}][Info] File is missing, may be is it first time run. Creating it ...."
            )
            self.query_remote_and_update_cache()
            self.save_state_to_file_if_enabled()

    def _load_order_state_generation_locked(self):
        meta_path = self.filename + ".meta"
        backup_path = self._previous_generation_path()
        has_file = os.path.exists(self.filename)
        has_meta = os.path.exists(meta_path)
        has_backup = os.path.exists(backup_path)
        if not has_file and not has_meta and not has_backup:
            return False

        reason = self._account_reader_reason()
        legacy_meta = None
        if has_meta:
            try:
                raw_meta = read_json_dict(meta_path)
            except (OSError, ValueError, TypeError) as exc:
                raise account_cache_health.AccountCacheNotReady(reason) from exc
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
                self._read_snapshot_recovering_previous_locked(
                    self._read_legacy_order_snapshot_strict, legacy_meta
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

    def _save_state_to_file_impl(self) -> bool:
        return self._save_account_order_snapshot()

    def _save_account_order_snapshot(self):
        """Commit Order snapshot and its manifest under the generation lock."""
        return self._with_generation_lock_bool(
            self._save_account_order_snapshot_generation_locked,
            "Order snapshot",
        )

    def _save_account_order_snapshot_generation_locked(self):
        """Write Order data and manifest while retaining the prior certified generation."""
        try:
            with self.lock:
                previous_metadata = self._load_versioned_account_meta(optional=True)
                if (
                    previous_metadata is not None
                    and self._persisted_data_version
                    and previous_metadata["data_version"] != self._persisted_data_version
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
                backup_path, had_prior_file = self._stage_prior_account_generation_locked(
                    previous_metadata,
                    self._read_order_snapshot_recovering_previous_locked,
                )
                base_meta = self._build_basic_meta_locked(sanitize_cursor=True)
                fetchtime, rest_sync_cursor, counts = (
                    base_meta["fetchtime"],
                    base_meta["rest_sync_cursor"],
                    base_meta["counts"],
                )
                data_version = self._snapshot_data_version(self.cache)
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
                    lambda: atomic_write_json(self.filename, payload, indent=1),
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
                f"[{self.cls_name}][Error] While saving the cache file {self.filename} / .tmp : {exc}"
            )
            return False

    def _reload_snapshot_from_disk(self):
        try:
            metadata = self._load_versioned_account_meta(optional=True)
            if metadata is not None:
                self._ensure_order_version(str(metadata["data_version"]))
                return
            with self._generation_lock():
                self._load_order_state_generation_locked()
        except Exception as e:
            print(f"[{self.cls_name}][Error] reload {self.filename}: {e}")

    def _read_order_snapshot_strict(self, metadata, *, path=None):
        """Read and validate an Order snapshot against its durable manifest."""
        return _cache_integrity.read_order_snapshot_strict(
            path or self.filename,
            metadata,
            is_valid_item_fn=self._is_valid_item,
            reader_reason=self._account_reader_reason(),
        )

    def _read_order_snapshot_recovering_previous_locked(self, metadata):
        """Read an Order snapshot against ``metadata``, restoring ``.previous`` if needed."""
        return self._read_snapshot_recovering_previous_locked(
            self._read_order_snapshot_strict, metadata
        )

    def _ensure_account_version(self, required_version: str) -> None:
        self._ensure_order_version(required_version)

    def _ensure_order_version(self, required_version):
        with self._generation_lock():
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
            if any(
                after[k] != metadata[k]
                for k in ("data_version", "counts", "fetchtime", "rest_sync_cursor")
            ):
                raise ValueError("Order manifest changed during reload")
            with self.lock:
                self.cache = items
                self.fetchtime_per_symbol = dict(fetchtime)
                self._rest_sync_cursor_per_symbol = dict(metadata["rest_sync_cursor"])
                self._persisted_counts = dict(actual_counts)
                self._persisted_data_version = required_version
                self._loaded_data_version = required_version
                self._loaded_counts = dict(actual_counts)
                self._account_cache_dirty = False
        except account_cache_health.AccountCacheNotReady:
            raise
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise account_cache_health.AccountCacheNotReady(reason) from exc

    def get_remote_items(self, symbol, startTime):
        fills = self._paginate_validated_trades(symbol, startTime, "fill")
        order_fills = [
            {
                "orderId": fill["orderId"],
                "price": extract_order_price(fill),
                "quantity": extract_order_qty(fill),
                "timestamp": extract_timestamp_ms(fill, default=0) or 0,
                "side": extract_order_side(fill),
                "_fillId": str(fill["id"]),
            }
            for fill in fills
        ]
        print(f"[{self.cls_name}][info] New order fills: {len(order_fills)}")
        return order_fills

    def update_cache_per_symbol(self, symbol, new_items):
        """Route direct update_cache_per_symbol calls through fill-level order aggregation."""
        if new_items is None:
            return
        self._persist_items(symbol, self._coerce_item_list(new_items))

    def _persist_items(self, symbol, new_items):
        """Merge each immutable fill exactly once into its order aggregate."""
        if not new_items:
            return
        new_items = self._coerce_item_list(new_items)
        now_ms = int(_now_time() * 1000)
        with self.lock:
            bucket = self.cache.setdefault(symbol, [])
            candidate = [dict(item) for item in bucket]
            by_id = {}
            fill_owners = {}
            for index, item in enumerate(candidate):
                if (
                    not isinstance(item, dict)
                    or not self._is_valid_item(item)
                    or (
                        item.get("symbol") is not None
                        and str(item["symbol"]).upper() != str(symbol).upper()
                    )
                ):
                    raise RuntimeError("invalid Order row already in memory")
                oid_key = str(int(item["orderId"]))
                if oid_key in by_id:
                    raise RuntimeError(
                        f"duplicate Order aggregate ID {oid_key} in memory"
                    )
                by_id[oid_key] = index
                for existing_fid in item.get("_fillIds", []):
                    fid_key = self._normalize_fill_id(existing_fid)
                    prev_owner = fill_owners.get(fid_key)
                    if prev_owner is not None and prev_owner != oid_key:
                        raise RuntimeError(
                            f"immutable fill ID {fid_key} belongs to both orders {prev_owner} and {oid_key}"
                        )
                    fill_owners[fid_key] = oid_key
            for item in new_items:
                if not isinstance(item, dict):
                    raise RuntimeError("invalid incoming Order row")
                incoming = dict(item)
                if (
                    incoming.get("symbol") is not None
                    and str(incoming["symbol"]).upper() != str(symbol).upper()
                ):
                    raise RuntimeError("invalid incoming Order row symbol")
                raw_fill_id = incoming.pop("_fillId", None)
                fill_id = ""
                if raw_fill_id is not None:
                    fill_id = self._normalize_fill_id(raw_fill_id)
                    if fill_id is None:
                        raise RuntimeError("invalid incoming Order fill ID")
                if not self._is_valid_item(incoming):
                    raise RuntimeError("invalid incoming Order row")
                key = str(int(incoming["orderId"]))
                if fill_id:
                    prev_owner = fill_owners.get(fill_id)
                    if prev_owner is not None and prev_owner != key:
                        raise RuntimeError(
                            f"immutable fill ID {fill_id} belongs to both orders {prev_owner} and {key}"
                        )
                existing_index = by_id.get(key)
                if existing_index is None:
                    by_id[key] = len(candidate)
                    incoming["orderId"] = int(incoming["orderId"])
                    incoming["price"] = float(incoming["price"])
                    incoming["quantity"] = float(incoming["quantity"])
                    incoming["timestamp"] = int(incoming["timestamp"])
                    incoming["side"] = str(incoming["side"]).upper()
                    if fill_id:
                        incoming["_fillIds"] = [fill_id]
                        fill_owners[fill_id] = key
                    candidate.append(incoming)
                    continue

                existing = candidate[existing_index]
                known_fill_id_list = [
                    self._normalize_fill_id(existing_fill_id)
                    for existing_fill_id in existing.get("_fillIds", [])
                ]
                legacy_cutoff = existing.get("_legacyCoveredThrough")
                if legacy_cutoff is None and "_fillIds" not in existing:
                    legacy_cutoff = int(existing["timestamp"])
                known_fill_ids = {*known_fill_id_list}
                if fill_id and fill_id in known_fill_ids:
                    continue
                if legacy_cutoff is not None and int(incoming["timestamp"]) <= int(
                    legacy_cutoff
                ):
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
                merged.update(
                    {
                        "orderId": int(existing["orderId"]),
                        "price": apiorders._round_fill_val(aggregate_price),
                        "quantity": apiorders._round_fill_val(total_quantity),
                        "timestamp": max(
                            int(existing["timestamp"]),
                            int(incoming["timestamp"]),
                        ),
                        "side": str(existing["side"]).upper(),
                    }
                )
                if fill_id:
                    merged["_fillIds"] = list(
                        dict.fromkeys([*known_fill_id_list, fill_id])
                    )
                    fill_owners[fill_id] = key
                if legacy_cutoff is not None:
                    merged["_legacyCoveredThrough"] = int(legacy_cutoff)
                candidate[existing_index] = merged

            if candidate != bucket:
                self.cache[symbol] = candidate
                self._advance_symbol_fetchtime_locked(symbol, now_ms)


CacheOrder = CacheOrderManager


class CacheSparsePriceManager(CachePriceSeriesMixin, CacheManagerInterface):
    """Store sparse three-minute price samples for two years in JSONL."""

    def __init__(self, sync_ts, symbols, filename, api_client=api, read_only=False):
        super().__init__(
            sync_ts,
            symbols,
            filename,
            append_mode=True,
            api_client=api_client,
            append_persist=True,
            read_only=read_only,
        )


CacheSparsePrice = CacheSparsePriceManager


class Cache24PriceManager(CachePriceSeriesMixin, CacheManagerInterface):
    """Collect maximum-granularity prices over the latest ``KEEP_HOURS``."""

    KEEP_HOURS = 24
    _filter_fetchtimes_to_symbols = True

    def __init__(self, sync_ts, symbols, filename, api_client=api, read_only=False):
        super().__init__(
            sync_ts,
            symbols,
            filename,
            append_mode=True,
            api_client=api_client,
            read_only=read_only,
        )
        self._filter_maps_to_configured_symbols_locked()

    def _filter_maps_to_configured_symbols_locked(self):
        """Remove unrelated symbols loaded from legacy redundant files that stored every coin."""
        with self.lock:
            self.cache = {s: v for s, v in self.cache.items() if s in self.symbols}
            self.fetchtime_per_symbol = {
                s: v for s, v in self.fetchtime_per_symbol.items() if s in self.symbols
            }
            self._rest_sync_cursor_per_symbol = {
                s: v
                for s, v in self._rest_sync_cursor_per_symbol.items()
                if s in self.symbols
            }

    def _on_stale_memory_detected_locked(self) -> None:
        self._reload_from_disk()

    def _before_snapshot_commit_locked(self, persisted_max_ts: int) -> None:
        if persisted_max_ts > self._last_disk_sync_max_ts:
            self._merge_newer_disk_ohlc_series_locked()

    def _filter_reloaded_items_locked(self, items: dict) -> dict:
        return {s: v for s, v in items.items() if s in self.symbols}

    def _merge_newer_disk_ohlc_series_locked(self) -> None:
        """Preserve 3-element OHLCV bars written to disk by enrichment while retaining recent in-memory ticks."""
        backup_path = self._previous_generation_path()
        if not os.path.exists(self.filename) and not os.path.exists(backup_path):
            return
        try:
            data = read_json_dict(
                self.filename, backup_path=backup_path, recover_previous=True
            )
            disk_items = (
                data.get("items")
                if isinstance(data, dict) and isinstance(data.get("items"), dict)
                else {}
            )
            for s in list(self.symbols):
                disk_rows = disk_items.get(s)
                if not isinstance(disk_rows, list) or not any(
                    isinstance(r, (list, tuple))
                    and len(r) >= 3
                    and (safe_pos_float(r[2], default=0.0) or 0.0) > 0.0
                    for r in disk_rows
                ):
                    continue
                by_ts = {
                    self._entry_timestamp_ms(r): r
                    for r in disk_rows
                    if self._entry_timestamp_ms(r) > 0
                }
                ohlc_ts_list = [
                    self._entry_timestamp_ms(r)
                    for r in disk_rows
                    if isinstance(r, (list, tuple))
                    and len(r) >= 3
                    and (safe_pos_float(r[2], default=0.0) or 0.0) > 0.0
                ]
                first_ohlc_ts = min(ohlc_ts_list, default=0)
                last_ohlc_ts = max(ohlc_ts_list, default=0)
                for r in self.cache.get(s) or ():
                    ts = self._entry_timestamp_ms(r)
                    if ts <= 0:
                        continue
                    is_ohlc_r = (
                        isinstance(r, (list, tuple))
                        and len(r) >= 3
                        and (safe_pos_float(r[2], default=0.0) or 0.0) > 0.0
                    )
                    if ts < first_ohlc_ts or ts >= last_ohlc_ts - 15 * 60 * 1000:
                        existing = by_ts.get(ts)
                        if existing is None or (
                            is_ohlc_r
                            and (
                                not isinstance(existing, (list, tuple))
                                or len(existing) < 3
                            )
                        ):
                            by_ts[ts] = r
                self.cache[s] = [by_ts[k] for k in sorted(by_ts)]
                if hasattr(self, "_trim_old_data"):
                    self._trim_old_data(s)
        except Exception as exc:
            print(f"[{self.cls_name}][Error] merge disk OHLC {self.filename}: {exc}")

    def on_price_update(self, symbol: str, ts_ms: int, price: float):
        """Receive every new WebSocket or HTTP price from CacheCurrentPriceManager."""
        if symbol not in self.symbols:
            return
        ts_val = safe_pos_int(ts_ms, default=None)
        price_val = safe_pos_float(price, default=None)
        if ts_val is None or price_val is None:
            return
        with self.lock:
            entries = self.cache.get(symbol)
            if entries and ts_val < self._entry_timestamp_ms(entries[-1]):
                return
        self._ensure_fetchtimes_initialized()
        self.update_cache_per_symbol(symbol, [[ts_val, price_val]])
        self._trim_old_data(symbol)
        self._notify_price_subscribers(symbol, ts_val, price_val)

    def _trim_old_data(self, symbol):
        """Remove entries older than ``KEEP_HOURS`` efficiently on every tick."""
        cutoff_ms = int((_now_time() - self.KEEP_HOURS * 3600) * 1000)
        with self.lock:
            entries = self.cache.get(symbol)
            if not entries or not isinstance(entries, list):
                return
            idx = bisect.bisect_left(
                entries, cutoff_ms, key=self._entry_timestamp_ms
            )
            if is_equity_symbol(symbol):
                idx = min(idx, max(0, len(entries) - 288))
            if idx > 0:
                self.cache[symbol] = entries[idx:]

    def get_recent_entries(self, symbol: str, last_seconds: float) -> list:
        """Return ``[timestamp_ms, price]`` entries from the latest interval."""
        sec_val = safe_pos_float(last_seconds, default=None)
        if sec_val is None:
            return []
        cutoff_ms = int((_now_time() - sec_val) * 1000)
        with self.lock:
            entries = casefold_get(self.cache, symbol, [])
            if not entries or not isinstance(entries, list):
                return []
            idx = bisect.bisect_left(
                entries, cutoff_ms, key=self._entry_timestamp_ms
            )
            return entries[idx:]

    def _should_poll(self):
        return False


Cache24Price = Cache24PriceManager


class Cache365PriceManager(Cache24PriceManager):
    """Multi-month/multi-year dense archive variant using incremental JSONL persistence."""

    KEEP_HOURS = CM_LONG_ARCHIVE_MONTHS * 30 * 24
    RETENTION_DAYS = CM_LONG_ARCHIVE_MONTHS * 30
    LONG_TRIM_INTERVAL_SEC = 24 * 3600

    def __init__(self, sync_ts, symbols, filename, api_client=api, read_only=False):
        self._last_long_trim = 0.0
        CacheManagerInterface.__init__(
            self,
            sync_ts,
            symbols,
            filename,
            append_mode=True,
            api_client=api_client,
            append_persist=True,
            read_only=read_only,
        )
        self._filter_maps_to_configured_symbols_locked()

    def _on_stale_memory_detected_locked(self) -> None:
        pass

    def _before_snapshot_commit_locked(self, persisted_max_ts: int) -> None:
        pass

    def _should_compact_before_append_locked(self) -> bool:
        return False

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
                for raw_line in self._tail_lines(
                    self.filename, CM_LONG_ARCHIVE_MEMORY_ROWS
                ):
                    try:
                        rec = json.loads(raw_line)
                        if (
                            isinstance(rec, dict)
                            and rec.get("s") in self.symbols
                            and rec.get("i") is not None
                        ):
                            self.cache.setdefault(rec["s"], []).append(rec["i"])
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                        continue
            self._persisted_counts = {s: len(v) for s, v in self.cache.items()}
            metaf = self.filename + ".meta"
            if os.path.exists(metaf):
                try:
                    meta_payload = read_json_dict(metaf)
                    if isinstance(meta_payload, dict):
                        self._hydrate_fetchtimes_from_mapping_locked(
                            meta_payload, filter_symbols=True, coerce_nonneg_int=True
                        )
                except (OSError, ValueError, TypeError):
                    pass
            if self.cache:
                for s_sym, rebuilt_ft in self.rebuild_fetchtimes().items():
                    self.fetchtime_per_symbol[s_sym] = max(
                        int(self.fetchtime_per_symbol.get(s_sym, 0) or 0),
                        int(rebuilt_ft or 0),
                    )
                    self._rest_sync_cursor_per_symbol.setdefault(
                        s_sym, self.fetchtime_per_symbol[s_sym]
                    )

    def _trim_old_data(self, symbol):
        with self.lock:
            entries = self.cache.get(symbol, [])
            overflow = max(0, len(entries) - CM_LONG_ARCHIVE_MEMORY_ROWS)
            if overflow:
                self.cache[symbol] = entries[overflow:]
                self._persisted_counts[symbol] = max(
                    0, self._persisted_counts.get(symbol, 0) - overflow
                )
        now = _monotonic_time()
        if now - self._last_long_trim < self.LONG_TRIM_INTERVAL_SEC:
            return
        self._last_long_trim = now
        with self.lock:
            before = len(self.cache.get(symbol, []))
        super()._trim_old_data(symbol)
        with self.lock:
            after = len(self.cache.get(symbol, []))
            if after < before:
                removed = before - after
                self._persisted_counts[symbol] = max(
                    0, self._persisted_counts.get(symbol, 0) - removed
                )

    def _reload_and_merge_timestamps_locked(
        self, original_fetchtimes, original_rest_cursors
    ):
        self._load_jsonl()
        for s_sym, ft in original_fetchtimes.items():
            self.fetchtime_per_symbol[s_sym] = max(
                int(self.fetchtime_per_symbol.get(s_sym, 0) or 0),
                int(ft or 0),
            )
        for s_sym, cur in original_rest_cursors.items():
            self._rest_sync_cursor_per_symbol[s_sym] = max(
                int(self._rest_sync_cursor_per_symbol.get(s_sym, 0) or 0),
                int(cur or 0),
            )

    def _write_unpersisted_memory_tail_locked(
        self, target, original_cache, original_persisted_counts, *, min_ts_ms: int = 0
    ) -> bool:
        wrote = False
        for symbol, items in original_cache.items():
            start = min(max(0, original_persisted_counts.get(symbol, 0)), len(items))
            for item in items[start:]:
                if min_ts_ms <= 0 or self._entry_timestamp_ms(item) >= min_ts_ms:
                    target.write(_jsonl_line(symbol, item))
                wrote = True
        return wrote

    def _write_current_meta_locked(self) -> bool:
        return self._write_meta(self._build_basic_meta_locked(self._persisted_counts))

    def _collect_live_delta_locked(
        self, snap_item_ids: set[int], *, min_ts_ms: int = 0
    ) -> dict[str, list]:
        """Collect in-memory ticks added by WebSocket while disk streaming ran unlocked."""
        delta = {}
        for symbol, items in self.cache.items():
            extra = [
                item
                for item in items
                if id(item) not in snap_item_ids
                and (min_ts_ms <= 0 or self._entry_timestamp_ms(item) >= min_ts_ms)
            ]
            if extra:
                delta[symbol] = extra
        return delta

    def _merge_live_delta_locked(self, delta_by_sym: dict[str, list]) -> None:
        """Re-apply live ticks that arrived while an unlocked disk pass was running."""
        for symbol, extra_items in delta_by_sym.items():
            if not extra_items:
                continue
            bucket = self.cache.setdefault(symbol, [])
            seen = {
                json.dumps(it, sort_keys=True)
                for it in bucket[-self.DEDUP_WINDOW :]
            }
            for item in extra_items:
                key = json.dumps(item, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    bucket.append(item)
            overflow = max(0, len(bucket) - CM_LONG_ARCHIVE_MEMORY_ROWS)
            if overflow:
                self.cache[symbol] = bucket[overflow:]
                self._persisted_counts[symbol] = max(
                    0, self._persisted_counts.get(symbol, 0) - overflow
                )

    def maintain_append_persist(self):
        """Prune the full disk archive without holding ``self.lock`` during disk I/O."""
        if not self.save_state:
            return False
        cutoff_ms = int((_now_time() - self.RETENTION_DAYS * 86400) * 1000)
        if not os.path.exists(self.filename):
            return False
        changed = False
        backup_path = None
        original_state = None
        snap_item_ids: set[int] = set()
        try:
            with self.lock:
                original_state = self._snapshot_memory_state_locked()
                (
                    original_cache,
                    original_persisted_counts,
                    original_fetchtimes,
                    original_rest_cursors,
                ) = original_state
                snap_item_ids = {
                    id(item)
                    for items in original_cache.values()
                    for item in items
                }

            backup_path = self.filename + ".maint_backup"
            atomic_snapshot_file(self.filename, backup_path)
            with (
                atomic_write(self.filename) as target,
                open(backup_path, encoding="utf-8") as source,
            ):
                for line_idx, line in enumerate(source, 1):
                    if line_idx % 4096 == 0:
                        time.sleep(0)
                    try:
                        rec = json.loads(line)
                        keep = (
                            isinstance(rec, dict)
                            and isinstance(rec.get("s"), str)
                            and self._entry_timestamp_ms(rec.get("i")) >= cutoff_ms
                        )
                    except (json.JSONDecodeError, TypeError, ValueError):
                        keep = False
                    if keep:
                        target.write(line if line.endswith("\n") else line + "\n")
                    else:
                        changed = True
                if self._write_unpersisted_memory_tail_locked(
                    target,
                    original_cache,
                    original_persisted_counts,
                    min_ts_ms=cutoff_ms,
                ):
                    changed = True
                with self.lock:
                    delta_by_sym = self._collect_live_delta_locked(
                        snap_item_ids, min_ts_ms=cutoff_ms
                    )
                    for symbol, extra_items in delta_by_sym.items():
                        for item in extra_items:
                            target.write(_jsonl_line(symbol, item))
                            changed = True

            with self.lock:
                post_delta = self._collect_live_delta_locked(
                    snap_item_ids, min_ts_ms=cutoff_ms
                )
                live_fetchtimes = dict(original_fetchtimes)
                for s_sym, ft in self.fetchtime_per_symbol.items():
                    live_fetchtimes[s_sym] = max(
                        int(live_fetchtimes.get(s_sym, 0) or 0), int(ft or 0)
                    )
                live_rest_cursors = dict(original_rest_cursors)
                for s_sym, cur in self._rest_sync_cursor_per_symbol.items():
                    live_rest_cursors[s_sym] = max(
                        int(live_rest_cursors.get(s_sym, 0) or 0), int(cur or 0)
                    )
                self._reload_and_merge_timestamps_locked(
                    live_fetchtimes, live_rest_cursors
                )
                self._merge_live_delta_locked(post_delta)
                if changed:
                    if not self._write_current_meta_locked():
                        self._restore_memory_state_locked(original_state)
                        self._merge_live_delta_locked(post_delta)
                        self._restore_previous_generation(backup_path)
                        backup_path = None
                        return False
                if backup_path is not None:
                    self._discard_generation_file(backup_path)
                    backup_path = None
            if changed:
                builtins.print(
                    f"[{self.cls_name}][maintain] pruned expired/corrupt disk records"
                )
            if os.path.getsize(self.filename) > self.MAX_FILE_BYTES:
                return self._rotate_disk_archive()
            return True
        except OSError as exc:
            with self.lock:
                post_delta = self._collect_live_delta_locked(snap_item_ids)
                self._restore_memory_state_locked(original_state)
                self._merge_live_delta_locked(post_delta)
                if backup_path is not None and os.path.exists(backup_path):
                    try:
                        self._restore_previous_generation(backup_path)
                    except OSError:
                        pass
            builtins.print(
                f"[{self.cls_name}][Error] streaming maintenance {self.filename}: {exc}"
            )
            return False

    def compact_jsonl(self):
        """Never rebuild an existing long archive from its bounded memory tail."""
        if os.path.exists(self.filename):
            return self.maintain_append_persist()
        return super().compact_jsonl()

    def _rotate_disk_archive(self):
        """Archive an oversized file while retaining a complete-line tail without holding ``self.lock`` during disk I/O."""
        archive = f"{self.filename}.{int(_now_time())}.archive"
        with self.lock:
            original_state = self._snapshot_memory_state_locked()
            (
                original_cache,
                original_persisted_counts,
                original_fetchtimes,
                original_rest_cursors,
            ) = original_state
            snap_item_ids = {
                id(item) for items in original_cache.values() for item in items
            }
        try:
            atomic_snapshot_file(self.filename, archive)
        except OSError as exc:
            builtins.print(f"[{self.cls_name}][maintain] archiving failed: {exc}")
            return False
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
                    lines = [
                        line
                        for line in source.read().splitlines(keepends=True)
                        if line.strip()
                    ]
                    tail = lines[-1] if lines else b""
            with atomic_write(self.filename) as target:
                decoded_tail = tail.decode("utf-8", errors="ignore")
                target.write(decoded_tail)
                if decoded_tail and not decoded_tail.endswith("\n"):
                    target.write("\n")
                self._write_unpersisted_memory_tail_locked(
                    target, original_cache, original_persisted_counts
                )
                with self.lock:
                    delta_by_sym = self._collect_live_delta_locked(snap_item_ids)
                    for symbol, extra_items in delta_by_sym.items():
                        for item in extra_items:
                            target.write(_jsonl_line(symbol, item))
            with self.lock:
                post_delta = self._collect_live_delta_locked(snap_item_ids)
                live_fetchtimes = dict(original_fetchtimes)
                for s_sym, ft in self.fetchtime_per_symbol.items():
                    live_fetchtimes[s_sym] = max(
                        int(live_fetchtimes.get(s_sym, 0) or 0), int(ft or 0)
                    )
                live_rest_cursors = dict(original_rest_cursors)
                for s_sym, cur in self._rest_sync_cursor_per_symbol.items():
                    live_rest_cursors[s_sym] = max(
                        int(live_rest_cursors.get(s_sym, 0) or 0), int(cur or 0)
                    )
                self._reload_and_merge_timestamps_locked(
                    live_fetchtimes, live_rest_cursors
                )
                self._merge_live_delta_locked(post_delta)
                meta_ok = self._write_current_meta_locked()
        except Exception as exc:
            builtins.print(
                f"[{self.cls_name}][maintain] rotation rewrite failed: {exc}"
            )
            meta_ok = False
        if not meta_ok:
            with self.lock:
                post_delta = self._collect_live_delta_locked(snap_item_ids)
                self._restore_memory_state_locked(original_state)
                self._merge_live_delta_locked(post_delta)
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
        self._compress_and_prune_archive(archive)
        return True


Cache365Price = Cache365PriceManager


class CacheAssetValueManager(CacheManagerInterface):
    def __init__(self, sync_ts, symbols, filename, api_client=api, read_only=False):
        self._migrated_usdt_on_load = False
        super().__init__(
            sync_ts,
            symbols,
            filename,
            append_mode=True,
            api_client=api_client,
            append_persist=True,
            read_only=read_only,
        )
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
                (
                    extract_timestamp_s(item, default=0.0) or 0.0
                    for item in items
                    if isinstance(item, dict)
                ),
                default=0.0,
            )
            if max_ts_sec > 0:
                last_times[symbol] = max(0, int(max_ts_sec * 1000) - 60_000)
        return last_times

    def _resolve_venue_balances_usd(self, binance_total_val: float) -> dict[str, float]:
        try:
            fdir = os.path.dirname(os.path.abspath(str(self.filename or "")))
            if os.path.basename(fdir) in (
                "orders",
                "prices",
                "state",
                "trends",
                "telemetry",
                "macro",
                "system",
            ):
                cache_root = os.path.dirname(fdir)
            else:
                cache_root = fdir
            vb_data = None
            for cand in (
                os.path.join(cache_root, "state", "venue_balances_state.json"),
                os.path.join(cache_root, "venue_balances_state.json"),
            ):
                if os.path.isfile(cand):
                    vb_data = read_json_dict(cand, default=None)
                    if isinstance(vb_data, dict) and vb_data:
                        break
            if not isinstance(vb_data, dict) or not vb_data:
                return {}
            live_cp_data = None
            for cp_cand in (
                os.path.join(cache_root, "prices", "cache_current_price.json"),
                os.path.join(cache_root, "cache_current_price.json"),
            ):
                if os.path.isfile(cp_cand):
                    live_cp_data = read_json_dict(cp_cand, default=None)
                    if isinstance(live_cp_data, dict) and live_cp_data:
                        break
            nav = compute_venue_nav_breakdown(
                venue_balances_state=vb_data,
                live_prices=live_cp_data,
                exclude_venues=("binance",),
            )
            other_venues = (
                nav.get("venues_usd")
                if isinstance(nav.get("venues_usd"), dict)
                else {}
            )
            if not other_venues:
                return {}
            venues_usd: dict[str, float] = {"binance": round(binance_total_val, 2)}
            for venue in sorted(other_venues):
                if other_venues[venue] > 0:
                    venues_usd[venue] = round(other_venues[venue], 2)
            return venues_usd if len(venues_usd) > 1 else {}
        except Exception:
            return {}

    def get_remote_items(self, symbol, startTime):
        try:
            total_usdc = self.api_client.get_total_assets_value_usdc(
                use_cache=False, raise_errors=True
            )
        except Exception as e:
            print(f"[{self.cls_name}][Error] Cannot query the total value: {e}")
            return []

        total_val = (
            safe_pos_float(total_usdc, default=None)
            if isinstance(total_usdc, (int, float))
            else None
        )
        if total_val is None:
            print(f"[{self.cls_name}][Error] Invalid total value: {total_usdc}")
            return []

        now_sec = int(_now_time())
        snapshot = {
            "timestamp": now_sec,
            "datetime_local": datetime.now().isoformat(timespec="seconds"),
            "total_value_usdc": round(total_val, 8),
        }
        venues_usd = self._resolve_venue_balances_usd(total_val)
        if venues_usd:
            snapshot["venues_usd"] = venues_usd
            snapshot["portfolio_total_usd"] = round(sum(venues_usd.values()), 2)
        return [snapshot]


CacheAssetValue = CacheAssetValueManager


class CacheCurrentPriceManager(CacheManagerInterface):
    """Maintain the latest timestamped price per symbol (WebSocket-first with HTTP fallback)."""

    WS_TIMEOUT_SEC = 15
    STALE_THRESHOLD_MS = 5_000
    FREQ_WINDOW_SEC = 60

    def __init__(
        self,
        sync_ts,
        symbols,
        filename,
        ws_manager=None,
        api_client=api,
        market_api=None,
        provider_names=None,
        read_only=False,
    ):
        self._ws_manager = ws_manager
        self._ws_managers = [ws_manager] if ws_manager is not None else []
        self._ws_default_event_ts = 0.0
        self._ws_last_event_ts_value = 0.0
        self._ws_last_event_ts_per_symbol: dict = {}
        self._ws_stream_event_ts_per_symbol: dict = {}
        self._price_subscribers = []
        self._update_timestamps: dict = defaultdict(deque)
        self._rth_empty_backoff_ts: dict = {}
        self._first_sleep = True
        self.market_api = market_api or _market_api.api
        self._provider_names = {}
        self.bind_providers(provider_names)
        super().__init__(
            sync_ts,
            symbols,
            filename,
            append_mode=False,
            api_client=api_client,
            read_only=read_only,
        )
        self.attach_ws_manager(ws_manager)

    @property
    def _ws_last_event_ts(self) -> float:
        return self._ws_last_event_ts_value

    @_ws_last_event_ts.setter
    def _ws_last_event_ts(self, value: float) -> None:
        ts = safe_nonneg_float(value, default=0.0) or 0.0
        self._ws_last_event_ts_value = ts
        self._ws_default_event_ts = ts
        self._ws_last_event_ts_per_symbol.clear()
        self._ws_stream_event_ts_per_symbol.clear()

    def bind_provider(self, symbol: str, provider_name: str) -> None:
        """Bind one symbol to its configured venue and reject ambiguity."""
        key = str(symbol or "").strip().upper()
        value = str(provider_name or "").strip()
        if not key or not value:
            raise ValueError("symbol and provider_name must be non-empty")
        previous = self._provider_names.get(key)
        if previous is not None and normalize_venue_name(
            previous
        ) != normalize_venue_name(value):
            raise ValueError(
                f"conflicting providers for {key}: {previous} and {value}"
            )
        self._provider_names[key] = value

    def bind_providers(self, provider_names) -> None:
        """Merge configured symbol-to-venue bindings without implicit fallback."""
        for symbol, provider_name in dict(provider_names or {}).items():
            self.bind_provider(symbol, provider_name)

    def provider_name_for(self, symbol: str):
        sym_u, sym_clean = _norm_symbol_keys(symbol)
        ck = canonical_symbol_key(symbol)
        return (
            casefold_get(self._provider_names, sym_u)
            or casefold_get(self._provider_names, sym_clean)
            or (casefold_get(self._provider_names, ck) if ck else None)
        )

    def _resolve_symbol_ws_ts(self, ts_map: dict, symbol: str) -> float:
        sym_u, sym_clean = _norm_symbol_keys(symbol)
        for cand in (symbol, sym_u, sym_clean):
            if cand and (val := ts_map.get(cand)) is not None:
                return val
        return self._ws_default_event_ts

    def _is_ws_ts_map_healthy(
        self, ts_map: dict, symbol: Optional[str] = None
    ) -> bool:
        now = _now_time()
        if symbol is not None:
            last_ts = self._resolve_symbol_ws_ts(ts_map, symbol)
            return -5.0 <= (now - last_ts) < self.WS_TIMEOUT_SEC
        if not self.symbols:
            return -5.0 <= (now - self._ws_last_event_ts_value) < self.WS_TIMEOUT_SEC
        return all(self._is_ws_ts_map_healthy(ts_map, s) for s in self.symbols)

    def _ws_is_healthy(self, symbol: Optional[str] = None) -> bool:
        return self._is_ws_ts_map_healthy(self._ws_last_event_ts_per_symbol, symbol)

    def _real_ws_is_healthy(self, symbol: Optional[str] = None) -> bool:
        return self._is_ws_ts_map_healthy(
            self._ws_stream_event_ts_per_symbol, symbol
        )

    def _record_price_timestamp(self, symbol: str) -> None:
        now = _now_time()
        cutoff = now - self.FREQ_WINDOW_SEC
        with self.lock:
            dq = self._update_timestamps[symbol]
            dq.append(now)
            while dq and dq[0] < cutoff:
                dq.popleft()

    def _push_price(
        self, symbol: str, price: float, ts_ms: Optional[int] = None
    ) -> None:
        if isinstance(ts_ms, bool):
            return
        price_val = safe_pos_float(price, default=None)
        ts_val = (
            safe_pos_int(ts_ms, default=None)
            if ts_ms is not None
            else int(_now_time() * 1000)
        )
        if price_val is None or ts_val is None or ts_val <= 0:
            return
        self._ensure_fetchtimes_initialized()
        self._record_price_timestamp(symbol)
        self.update_cache_per_symbol(symbol, [[ts_val, price_val]])
        self._notify_price_subscribers(symbol, ts_val, price_val)

    @staticmethod
    def _extract_price_and_ts(items):
        if not items or not isinstance(items, (list, tuple)):
            return None, None
        candidates = (
            reversed(items) if isinstance(items[0], (list, tuple)) else (items,)
        )
        for cand in candidates:
            if not isinstance(cand, (list, tuple)) or len(cand) < 2:
                continue
            ts_ms = safe_pos_int(cand[0], default=None)
            price = safe_pos_float(cand[1], default=None)
            if ts_ms is not None and price is not None:
                return price, ts_ms
        return None, None

    def on_items_update(self, symbol: str, items):
        price, ts_ms = self._extract_price_and_ts(items)
        if price is None:
            return
        with self.lock:
            entries = self.cache.get(symbol)
            existing_ts = (
                self._entry_timestamp_ms(entries[0]) if entries else 0
            )
        if existing_ts > 0 and (existing_ts - ts_ms) > self.STALE_THRESHOLD_MS:
            return
        now = _now_time()
        self._ws_last_event_ts_value = now
        self._ws_last_event_ts_per_symbol[symbol] = now
        self._ws_stream_event_ts_per_symbol[symbol] = now
        self._ws_stream_event_ts_per_symbol[str(symbol).strip().upper()] = now
        self._push_price(symbol, price, ts_ms=ts_ms)

    def query_remote_and_update_cache(self):
        if not self.fetchtime_per_symbol:
            self._ensure_fetchtimes_initialized()
            self._rest_sync_cursor_per_symbol = dict(self.fetchtime_per_symbol)
        for symbol in list(self.symbols):
            if self._ws_is_healthy(symbol):
                continue
            is_rth_sym = (
                normalize_venue_name(self.provider_name_for(symbol) or "") == "t212"
                or is_equity_symbol(symbol)
            )
            if is_rth_sym and "PYTEST_CURRENT_TEST" not in os.environ:
                now_s = _now_time()
                if (now_s - self._rth_empty_backoff_ts.get(symbol, 0.0)) < 60.0:
                    continue
                if not is_us_rth_open(now_s):
                    with self.lock:
                        existing_entries = self.cache.get(symbol)
                    if (
                        existing_entries
                        and self._extract_price_and_ts(existing_entries)[0] is not None
                    ):
                        continue
            startTime = self.fetchtime_per_symbol.get(
                symbol, self.fallback_time_default
            )
            new_items = self.get_remote_items(symbol=symbol, startTime=startTime)
            if new_items is None:
                raise RuntimeError(
                    f"{self.cls_name} returned no synchronization result for {symbol}"
                )
            if not new_items:
                if is_rth_sym:
                    self._rth_empty_backoff_ts[symbol] = _now_time()
                print(
                    f"[{self.cls_name}][Info] {symbol}:  No remote items starting with {u.timestampToTime(startTime)} "
                )
            else:
                if is_rth_sym:
                    self._rth_empty_backoff_ts.pop(symbol, None)
                self._persist_items(symbol, new_items)
        _bapi_ws_bridge.clear_ws_rest_resync_for_manager(self.cls_name)
        return True

    def get_remote_items(self, symbol, startTime):
        try:
            provider_name = self.provider_name_for(symbol)
            with live_price_fetch_context():
                if provider_name is None:
                    price = self.market_api.get_current_price(symbol=symbol)
                else:
                    price = self.market_api.get_current_price(
                        symbol=symbol, provider_name=provider_name
                    )
            price_val = safe_pos_float(price, default=None)
            if price_val is None:
                return []
            ts_ms = int(_now_time() * 1000)
            return [[ts_ms, price_val]]
        except Exception as e:
            print(f"[{self.cls_name}][Error] HTTP fetch {symbol}: {e}")
            return []

    def _persist_items(self, symbol, new_items):
        price, ts_ms = self._extract_price_and_ts(new_items)
        if price is None:
            return
        self._push_price(symbol, price, ts_ms=ts_ms)

    rebuild_fetchtimes = CachePriceSeriesMixin.rebuild_fetchtimes
    get_all_symbols_from_cache = CachePriceSeriesMixin.get_all_symbols_from_cache

    def _should_poll(self):
        return not self._ws_is_healthy()

    def _timestamps_deque_locked(self, symbol: str):
        if symbol in self._update_timestamps:
            return self._update_timestamps.get(symbol)
        sym_u, sym_clean = _norm_symbol_keys(symbol)
        return casefold_get(self._update_timestamps, sym_u) or casefold_get(
            self._update_timestamps, sym_clean
        )

    def get_sample_rate(self, symbol: str, fallback: float = 1.0) -> float:
        fb = safe_pos_float(fallback, default=1.0) or 1.0
        with self.lock:
            dq = self._timestamps_deque_locked(symbol)
            samples = list(dq) if dq else []
        if len(samples) < 2:
            return fb
        span = safe_pos_float(samples[-1] - samples[0], default=None)
        if span is None:
            return fb
        return span / (len(samples) - 1)

    def get_update_frequency(self, symbol: str) -> float:
        cutoff = _now_time() - self.FREQ_WINDOW_SEC
        with self.lock:
            dq = self._timestamps_deque_locked(symbol)
            if dq:
                while dq and dq[0] < cutoff:
                    dq.popleft()
                samples = list(dq)
            else:
                samples = []
        if len(samples) < 2:
            return 0.0
        return len(samples) / self.FREQ_WINDOW_SEC

    def get_cached_price_observation(self, symbol: str, max_age_sec: float):
        max_age = safe_pos_float(max_age_sec, default=None)
        if max_age is None:
            raise ValueError("max_age_sec must be finite and positive")
        _, sym_clean = _norm_symbol_keys(symbol)
        with self.lock:
            entries = casefold_get(self.cache, symbol) or casefold_get(
                self.cache, sym_clean
            )
            cached_price, last_ts = self._extract_price_and_ts(entries)
        if cached_price is None or last_ts is None:
            return None
        observed_at = safe_pos_float(float(last_ts) / 1000.0, default=None)
        price = safe_pos_float(cached_price, default=None)
        if observed_at is None or price is None:
            return None
        age = _now_time() - observed_at
        if age < -5.0 or age > max_age:
            return None
        return observed_at, price

    def _resolve_target_symbol_locked(self, symbol: str) -> str:
        sym_u, sym_clean = _norm_symbol_keys(symbol)
        for cand in (symbol, sym_u, sym_clean):
            if cand in self.cache:
                return cand
        for k in self.cache:
            if isinstance(k, str) and k.upper() in (sym_u, sym_clean):
                return k
        configured = getattr(self, "symbols", ())
        if sym_clean in configured:
            return sym_clean
        for s in configured:
            if isinstance(s, str) and s.upper() in (sym_u, sym_clean):
                return s
        return symbol

    def _cached_entry_for_symbol_locked(self, target_sym: str):
        cached_price, last_ts = self._extract_price_and_ts(self.cache.get(target_sym))
        return (
            ([last_ts, cached_price] if cached_price is not None else None),
            last_ts,
        )

    def get_price(self, symbol: str):
        with self.lock:
            target_sym = self._resolve_target_symbol_locked(symbol)
            entry, last_ts = self._cached_entry_for_symbol_locked(target_sym)
        now_ms = int(_now_time() * 1000)
        age_ms = (now_ms - last_ts) if last_ts else None
        stale_threshold_ms = self.STALE_THRESHOLD_MS
        if (
            entry is not None
            and (
                os.environ.get("MPTRADE_FORCE_RTH_GATE") == "1"
                or (
                    "PYTEST_CURRENT_TEST" not in os.environ
                    and "unittest" not in sys.modules
                )
            )
            and hasattr(sym, "is_symbol_market_open")
            and not sym.is_symbol_market_open(target_sym)
        ):
            stale_threshold_ms = max(stale_threshold_ms, 3_600_000)
        if (
            entry is None
            or age_ms is None
            or age_ms > stale_threshold_ms
            or age_ms < -5_000
        ):
            age = age_ms if age_ms is not None else -1
            print(f"[{self.cls_name}] {target_sym} stale ({age}ms) - forced HTTP fetch")
            new = self.get_remote_items(target_sym, None)
            if not new or self._extract_price_and_ts(new)[0] is None:
                return None
            self._persist_items(target_sym, new)
            self.save_state_to_file_if_enabled()
            with self.lock:
                entry, _ = self._cached_entry_for_symbol_locked(target_sym)
        return entry

    def get_price_value(self, symbol: str) -> float:
        entry = self.get_price(symbol)
        return entry[1] if entry else None

    def attach_ws_manager(self, ws_manager) -> None:
        if ws_manager is None:
            return
        if self._ws_manager is None:
            self._ws_manager = ws_manager
        if ws_manager not in self._ws_managers:
            self._ws_managers.append(ws_manager)
        ws_manager.subscribe(self)


CacheCurrentPrice = CacheCurrentPriceManager


# ── Singletons, Factory & Accessors ───────────────────────────────────────────

_current_price_instance: Optional[CacheCurrentPriceManager] = None
_current_price_lock = threading.RLock()


def _default_price_tracking():
    """Return ``(symbols, nonbinance_symbols, provider_names)`` from ``instruments.conf``."""
    return tracked_price_instruments()


def get_current_price_manager(
    ws_manager=None,
    symbols=None,
    sync_ts=None,
    start_sync=True,
    provider_names=None,
    read_only=None,
) -> CacheCurrentPriceManager:
    """Return or lazily create the CacheCurrentPriceManager singleton."""
    global _current_price_instance
    start_sync = _parse_bool_flag(start_sync, default=True)
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
            _syms = (
                _coerce_symbol_list(symbols) if isinstance(symbols, str) else symbols
            )
        instance = CacheCurrentPriceManager(
            sync_ts=(
                sync_ts
                if sync_ts is not None
                else CURRENT_PRICE_SYNC_INTERVAL_SEC
            ),
            symbols=_syms,
            filename="cache_current_price.json",
            ws_manager=ws_manager,
            provider_names=provider_names,
            api_client=api,
            read_only=(
                (not start_sync)
                if read_only is None
                else _parse_bool_flag(read_only, default=False)
            ),
        )
        if start_sync:
            instance.periodic_sync(save_state=False)
        _current_price_instance = instance
    return _current_price_instance


_short_trend_instance = None
_short_trend_lock = threading.RLock()


def get_short_trend_manager(
    symbols=None, filename="cache_short_trend.json", writer=False
):
    """Return the CacheShortTrendManager singleton."""
    global _short_trend_instance
    writer = _parse_bool_flag(writer, default=False)
    inst = _short_trend_instance
    if inst is not None and not writer:
        return inst
    with _short_trend_lock:
        if _short_trend_instance is not None:
            if writer:
                _short_trend_instance.become_writer()
            return _short_trend_instance
        if symbols is None:
            _syms = _default_price_tracking()[0]
        else:
            _syms = (
                _coerce_symbol_list(symbols) if isinstance(symbols, str) else symbols
            )
        _short_trend_instance = CacheShortTrendManager(
            _syms, filename, writer=writer
        )
    return _short_trend_instance


ORDER_SYNC_INTERVAL_SEC = 0.4 * 60
TRADE_SYNC_INTERVAL_SEC = 3 * 60
SPARSE_PRICE_SYNC_INTERVAL_SEC = 3 * 60
PRICE_24_SYNC_INTERVAL_SEC = 30
PRICE_365_SYNC_INTERVAL_SEC = CM_LONG_ARCHIVE_FLUSH_SEC
CURRENT_PRICE_SYNC_INTERVAL_SEC = 30
SHORT_TREND_EVAL_INTERVAL_SEC = (
    CacheShortTrendManager.FULL_EVAL_INTERVAL_SEC
    if CacheShortTrendManager is not None
    else 3.0
)
LONG_TREND_SYNC_INTERVAL_SEC = 10 * 60
ASSET_VALUE_SYNC_INTERVAL_SEC = 10 * 60

_PRICE_TREND_CACHES = frozenset(
    {
        "CurrentPrice",
        "24Price",
        "365Price",
        "SparsePrice",
        "LongTrend",
    }
)


class CacheFactory:
    """Single registry and singleton factory for all domain cache managers."""

    _lock = threading.RLock()
    _instances = {}
    _sync_started = set()
    _CONFIG = {}

    @classmethod
    def register(
        cls,
        name: str,
        manager_class: type,
        *,
        sync_ts,
        filename: Optional[str] = None,
        prefix: Optional[str] = None,
        ext: Optional[str] = None,
        **extra_kwargs,
    ) -> dict:
        """Register or update a domain cache implementation in ``CacheFactory``."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("CacheFactory registration name must be a non-empty string")
        if not callable(sync_ts):
            interval_val = sync_ts
            sync_ts = lambda _v=interval_val: _v
        entry = {
            "class": manager_class,
            "filename": filename,
            "sync_ts": sync_ts,
        }
        if prefix is not None:
            entry["prefix"] = prefix
        if ext is not None:
            entry["ext"] = ext
        entry.update(extra_kwargs)
        with cls._lock:
            cls._CONFIG[name] = entry
        return entry

    @staticmethod
    def _iter_managers(value):
        return value.values() if isinstance(value, dict) else (value,)

    @classmethod
    def get(cls, name, symbols=None, *, start_sync=True, provider_names=None):
        start_sync = _parse_bool_flag(start_sync, default=True)
        if isinstance(symbols, str):
            symbols = _coerce_symbol_list(symbols)
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

            if name in cls._instances and symbols is not None:
                inst = cls._instances[name]
                existing = (
                    set(inst.keys())
                    if isinstance(inst, dict)
                    else set(getattr(inst, "symbols", []))
                )
                requested = set(symbols)
                if requested != existing:
                    missing = requested - existing
                    builtins.print(
                        f"[CacheFactory][WARN] '{name}' already exists with symbols {sorted(existing)}; "
                        f"the request for {sorted(requested)} is IGNORED"
                        + (f" (missing: {sorted(missing)})" if missing else "")
                        + ". Singleton per name — the first instance is used."
                    )
            if name == "CurrentPrice" and name in cls._instances:
                cls._instances[name].bind_providers(provider_names)

            created = name not in cls._instances
            sync_ts = None
            if created:
                config = cls._CONFIG[name]
                manager_class = config["class"]
                sync_ts = config["sync_ts"]()
                extra_kwargs = {
                    key: value
                    for key, value in config.items()
                    if key not in {"class", "filename", "sync_ts", "prefix", "ext"}
                }
                if symbols is None:
                    if name == "AssetValue":
                        symbols = ["TOTAL"]
                    elif name in _PRICE_TREND_CACHES:
                        symbols, _, _default_providers = _default_price_tracking()
                        if (
                            name == "CurrentPrice"
                            and provider_names is None
                            and _default_providers
                        ):
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
                elif "prefix" in config:
                    prefix, ext = config["prefix"], config["ext"]
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

                cls._instances[name] = instance

            if start_sync and name not in cls._sync_started:
                if sync_ts is None:
                    sync_ts = cls._CONFIG[name]["sync_ts"]()
                for manager in cls._iter_managers(cls._instances[name]):
                    manager.periodic_sync(sync_ts, False)
                cls._sync_started.add(name)

            return cls._instances[name]

    @staticmethod
    def _clear_bapi_trade_manager_ref(trade_mgr) -> None:
        if trade_mgr is None:
            return
        _bapi_tr = sys.modules.get("binance_api.bapi_trades")
        if (
            _bapi_tr is not None
            and getattr(_bapi_tr, "cache_trade_manager", None) is trade_mgr
        ):
            _bapi_tr.cache_trade_manager = None

    @classmethod
    def shutdown_all(cls, timeout=5.0):
        """Stop every factory-owned synchronization loop and clear the registry."""
        global _current_price_instance, _short_trend_instance
        with cls._lock:
            current_price = (
                cls._instances.get("CurrentPrice") or _current_price_instance
            )
            trade_mgr = cls._instances.get("Trade")
            short_trend = _short_trend_instance
            managers = []
            seen_ids = set()
            for value in cls._instances.values():
                for manager in cls._iter_managers(value):
                    if id(manager) not in seen_ids:
                        seen_ids.add(id(manager))
                        managers.append(manager)
            if current_price is not None and id(current_price) not in seen_ids:
                seen_ids.add(id(current_price))
                managers.append(current_price)
            if short_trend is not None and id(short_trend) not in seen_ids:
                seen_ids.add(id(short_trend))
                managers.append(short_trend)
            cls._instances = {}
            cls._sync_started = set()
            for manager in managers:
                ev = getattr(manager, "_stop_event", None)
                if ev is not None:
                    ev.set()
            eff_timeout = safe_nonneg_float(timeout, default=5.0)
            if eff_timeout is None:
                eff_timeout = 5.0
            deadline = _monotonic_time() + eff_timeout
            results = [
                manager.shutdown(timeout=max(0.0, deadline - _monotonic_time()))
                for manager in managers
                if hasattr(manager, "shutdown")
            ]
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
            cls._clear_bapi_trade_manager_ref(trade_mgr)
            return all(results) if results else True

    @classmethod
    def remove(cls, name, timeout=5.0):
        """Remove a singleton cleanly after stopping all threads it owns."""
        global _current_price_instance, _short_trend_instance
        with cls._lock:
            if name == "ShortTrend":
                inst = _short_trend_instance
                cls._sync_started.discard("ShortTrend")
                if inst is None:
                    return True
                stopped = (
                    bool(inst.shutdown(timeout=timeout))
                    if hasattr(inst, "shutdown")
                    else True
                )
                if stopped and inst is _short_trend_instance:
                    _short_trend_instance = None
                return stopped
            value = cls._instances.pop(name, None)
            if name == "CurrentPrice" and value is None:
                value = _current_price_instance
            cls._sync_started.discard(name)
            if value is None:
                return True
            results = [
                manager.shutdown(timeout=timeout)
                for manager in cls._iter_managers(value)
                if hasattr(manager, "shutdown")
            ]
            stopped = all(results) if results else True
            if (
                name == "CurrentPrice"
                and value is _current_price_instance
                and stopped
            ):
                _current_price_instance = None
            if name == "Trade" and stopped:
                cls._clear_bapi_trade_manager_ref(value)
            return stopped


# Register all canonical domain cache implementations in one place.
CacheFactory.register(
    "Trade",
    CacheTradeManager,
    filename="cache_trade.jsonl",
    sync_ts=lambda: TRADE_SYNC_INTERVAL_SEC,
)
CacheFactory.register(
    "Order",
    CacheOrderManager,
    filename="cache_order.json",
    sync_ts=lambda: ORDER_SYNC_INTERVAL_SEC,
)
CacheFactory.register(
    "SparsePrice",
    CacheSparsePriceManager,
    filename=None,
    prefix="cache_sparse_price_",
    ext="jsonl",
    sync_ts=lambda: SPARSE_PRICE_SYNC_INTERVAL_SEC,
)
CacheFactory.register(
    "24Price",
    Cache24PriceManager,
    filename=None,
    prefix="cache_24_price_",
    ext="json",
    sync_ts=lambda: PRICE_24_SYNC_INTERVAL_SEC,
)
CacheFactory.register(
    "365Price",
    Cache365PriceManager,
    filename=None,
    prefix="cache_365_price_",
    ext="jsonl",
    sync_ts=lambda: PRICE_365_SYNC_INTERVAL_SEC,
)
CacheFactory.register(
    "CurrentPrice",
    CacheCurrentPriceManager,
    filename="cache_current_price.json",
    sync_ts=lambda: CURRENT_PRICE_SYNC_INTERVAL_SEC,
)
CacheFactory.register(
    "LongTrend",
    CacheLongTrendManager,
    filename="cache_long_trend.json",
    sync_ts=lambda: LONG_TREND_SYNC_INTERVAL_SEC,
)
CacheFactory.register(
    "AssetValue",
    CacheAssetValueManager,
    filename="cache_asset_value.jsonl",
    sync_ts=lambda: ASSET_VALUE_SYNC_INTERVAL_SEC,
)


def get_cache_manager(name, symbols=None, *, start_sync=True):
    return CacheFactory.get(name, symbols, start_sync=start_sync)


def get_trade_manager(symbols=None, *, start_sync=True):
    return CacheFactory.get("Trade", symbols, start_sync=start_sync)


def get_order_manager(symbols=None, *, start_sync=True):
    return CacheFactory.get("Order", symbols, start_sync=start_sync)


def get_sparse_price_managers(symbols=None, *, start_sync=True):
    return CacheFactory.get("SparsePrice", symbols, start_sync=start_sync)


get_sparse_price_manager = get_sparse_price_managers


def get_24_price_managers(symbols=None, *, start_sync=True):
    return CacheFactory.get("24Price", symbols, start_sync=start_sync)


get_24_price_manager = get_24_price_managers


def get_365_price_managers(symbols=None, *, start_sync=True):
    return CacheFactory.get("365Price", symbols, start_sync=start_sync)


get_365_price_manager = get_365_price_managers


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


# ── Real-time Binance user stream -> cache actions ────────────────────────────

_ws_bridge = _bapi_ws_bridge._ws_bridge
_ws_bridge_lock = _bapi_ws_bridge._ws_bridge_lock
_ws_event_stats = _bapi_ws_bridge._ws_event_stats


def _ws_cache_getter(name, symbols=None, *, start_sync=False):
    """Look up caches from WebSocket callbacks without spawning rogue polling loops in consumers."""
    try:
        if symbols is not None:
            return get_cache_manager(name, symbols=symbols, start_sync=start_sync)
        return get_cache_manager(name, start_sync=start_sync)
    except TypeError:
        if symbols is not None:
            return get_cache_manager(name, symbols)
        return get_cache_manager(name)


def _upsert_order_from_execution_report(event):
    return _bapi_ws_bridge.upsert_order_from_execution_report(
        event, cache_getter=_ws_cache_getter
    )


def _append_trade_from_execution_report(event):
    return _bapi_ws_bridge.append_trade_from_execution_report(
        event, cache_getter=_ws_cache_getter
    )


def _refresh_asset_value_from_ws_event():
    _bapi_ws_bridge.refresh_asset_value_from_ws_event(cache_getter=_ws_cache_getter)


def _persist_ws_updated_caches(event_type):
    _bapi_ws_bridge.persist_ws_updated_caches(
        event_type, cache_getter=_ws_cache_getter
    )


_refresh_symbol_in_cache = _bapi_ws_bridge.refresh_symbol_in_cache


def _handle_binance_ws_event(event):
    _bapi_ws_bridge.handle_binance_ws_event(
        event,
        cache_getter=_ws_cache_getter,
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
    _bapi_ws_bridge.initialize_ws_bridge_once(enable_fn=enable_real_ws_event_sync)


if __name__ == "__main__":
    single_instance("cacheManager")
    account_cache_health.enable_writer()
    for _account_cache_name in ("Order", "Trade"):
        _acc_mgr = get_cache_manager(_account_cache_name, start_sync=False)
        CacheFactory._sync_started.add(_account_cache_name)
        _ver = str(
            getattr(_acc_mgr, "_persisted_data_version", "")
            or getattr(_acc_mgr, "_loaded_data_version", "")
            or ""
        )
        _cursors = (
            getattr(_acc_mgr, "_rest_sync_cursor_per_symbol", None)
            or getattr(_acc_mgr, "fetchtime_per_symbol", None)
            or {}
        )
        _sync_ms = (
            min(
                (
                    int(v)
                    for v in _cursors.values()
                    if isinstance(v, (int, float)) and v > 0
                ),
                default=0,
            )
            if isinstance(_cursors, dict)
            else 0
        )
        if (
            _ver
            and _sync_ms > 0
            and (int(time.time() * 1000) - _sync_ms)
            <= int(account_cache_health.MAX_AGE_SEC * 1000)
        ):
            account_cache_health.record_successful_sync(
                _account_cache_name, _ver, now_ms=_sync_ms
            )
    _initialize_once()
    threads = []
    _nb_poller = None
    _trend_mgr = None

    _trend_syms, _nb_syms, _nb_provider_names = _default_price_tracking()
    if _nb_syms:
        builtins.print(
            f"[cacheManager] tracking non-Binance instruments from instruments.conf: {_nb_syms}"
        )

    _trend_cpm = None
    try:
        from binance_api import bapi_ws

        _trend_cpm = get_current_price_manager(
            ws_manager=bapi_ws.get_ws_manager(),
            sync_ts=0.8,
            start_sync=False,
        )
    except Exception as _ws_init_err:
        print(f"[cacheManager] Warning pre-attaching price WS: {_ws_init_err}")

    for name, config in CacheFactory._CONFIG.items():
        if name == "365Price":
            _arch_syms = tracked_price_instruments(role="archive")[0]
            cache = get_cache_manager(
                name, symbols=_arch_syms or None, start_sync=False
            )
        else:
            cache = get_cache_manager(name, start_sync=False)
        interval = (
            cache.sync_ts
            if name == "CurrentPrice" and hasattr(cache, "sync_ts")
            else config["sync_ts"]()
        )

        if isinstance(cache, dict):
            for idx, manager in enumerate(cache.values()):
                if name == "365Price":
                    manager._maint_stagger_sec = float(idx * 300)
                threads.append(manager.periodic_sync(interval))
        else:
            threads.append(cache.periodic_sync(interval))
        CacheFactory._sync_started.add(name)

    try:
        if _trend_cpm is None:
            from binance_api import bapi_ws

            _trend_cpm = get_current_price_manager(
                ws_manager=bapi_ws.get_ws_manager(),
                sync_ts=0.8,
            )
        _trend_cache24 = CacheFactory.get("24Price")
        _trend_cache365 = CacheFactory._instances.get("365Price") or {}
        if isinstance(_trend_cache365, dict):
            for _c365 in _trend_cache365.values():
                _trend_cpm.subscribe_price(_c365)
        _trend_mgr = get_short_trend_manager(writer=True)
        _trend_mgr.start_computation(
            _trend_cache24, _trend_cpm, run_full_eval=True
        )
        print("⚙️ cacheManager: full trend computation started (cache_short_trend.json).")
        if _nb_syms:
            _nb_poller = _start_nonbinance_trend_poller(
                _trend_cpm,
                _nb_syms,
                interval_sec=10,
                provider_names=_nb_provider_names,
            )
    except Exception as e:
        print(f"[cacheManager] Cannot start the trend computation: {e}")

    _stop_requested = threading.Event()

    def _request_stop(_signum=None, _frame=None):
        _stop_requested.set()

    import signal as _signal

    _signal.signal(_signal.SIGTERM, _request_stop)
    _signal.signal(_signal.SIGINT, _request_stop)

    try:
        touch_module_heartbeat(
            "cacheManager",
            interval_sec=30.0,
            force=True,
            symbols=list(_trend_syms or ()),
        )
        while not _stop_requested.is_set() and any(t.is_alive() for t in threads):
            touch_module_heartbeat(
                "cacheManager",
                interval_sec=30.0,
                symbols=list(_trend_syms or ()),
            )
            if _stop_requested.wait(timeout=5.0):
                break
    except KeyboardInterrupt:
        print("Stopped manually.")
    finally:
        account_cache_health.disable_writer()
        print("Cleanup / releasing resources...")
        shutdown_deadline = time.monotonic() + 2.5
        try:
            from binance_api import bapi_ws as _bws

            _bws_mgr = getattr(_bws, "bapi_ws_manager", None)
        except Exception:
            _bws_mgr = None
        for _ws_obj in (_ws_bridge, _bws_mgr):
            if _ws_obj is not None and hasattr(_ws_obj, "_stop_event"):
                try:
                    _ws_obj._stop_event.set()
                except Exception:
                    pass
        if _nb_poller is not None and hasattr(_nb_poller, "stop_event"):
            _nb_poller.stop_event.set()
            for _nb_ws in getattr(_nb_poller, "ws_streams", ()):
                if hasattr(_nb_ws, "_stop_event"):
                    try:
                        _nb_ws._stop_event.set()
                    except Exception:
                        pass
        if _trend_mgr is not None and hasattr(_trend_mgr, "_stop_event"):
            _trend_mgr._stop_event.set()
        _c365_Shutdown_managers = list(
            CacheFactory._iter_managers(
                CacheFactory._instances.get("365Price") or {}
            )
        )
        for _inst in list(CacheFactory._instances.values()):
            for _mgr in CacheFactory._iter_managers(_inst):
                if hasattr(_mgr, "_stop_event"):
                    _mgr._stop_event.set()
        if _current_price_instance is not None and hasattr(
            _current_price_instance, "_stop_event"
        ):
            _current_price_instance._stop_event.set()
        if _nb_poller is not None:
            _nb_poller.stop(
                timeout=min(0.5, max(0.05, shutdown_deadline - time.monotonic()))
            )
        if _trend_mgr is not None:
            _trend_mgr.shutdown(
                timeout=min(0.8, max(0.05, shutdown_deadline - time.monotonic()))
            )
        CacheFactory.shutdown_all(
            timeout=min(1.0, max(0.05, shutdown_deadline - time.monotonic()))
        )
        for _c365_mgr in _c365_Shutdown_managers:
            try:
                _c365_mgr.save_state_to_file_if_enabled()
            except Exception:
                pass
        if _current_price_instance is not None:
            _current_price_instance.shutdown(
                timeout=min(0.3, max(0.05, shutdown_deadline - time.monotonic()))
            )
        for _ws_obj in (_ws_bridge, _bws_mgr):
            if _ws_obj is not None:
                try:
                    _ws_obj.stop(
                        timeout=min(
                            0.3, max(0.05, shutdown_deadline - time.monotonic())
                        )
                    )
                except Exception:
                    pass
        from binance_api import bapi_client

        bapi_client.stop_periodic_resync()
