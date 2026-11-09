"""Price domain cache hierarchy (Level 2 abstract base + Level 3 concrete classes + WS streams).

Provides:
- ``CachePriceManagerBase(CacheManagerInterface, ABC)``:
  Abstract base class for price-domain caches.
- ``CacheSparsePriceManager(CachePriceManagerBase)``:
  Sparse 3-minute price history in append-only JSONL.
- ``Cache24PriceManager(CachePriceManagerBase)``:
  Dense 24-hour rolling price series in per-symbol JSON snapshots.
- ``Cache365PriceManager(Cache24PriceManager)``:
  Long-horizon 365-day dense archive in append-only JSONL with bounded memory tail.
- ``CacheCurrentPriceManager(CachePriceManagerBase)``:
  Real-time per-symbol current price cache (WebSocket-first with HTTP fallback).
- ``HyperliquidPriceStream``, ``KrakenPriceStream``, ``_NonBinanceTrendPoller``,
  ``_start_nonbinance_trend_poller``: Non-Binance venue price streams and fallback poller.
"""

from __future__ import annotations

from abc import ABC
import asyncio
import bisect
import builtins
from collections import defaultdict, deque
import concurrent.futures as _futures
import json
import json as _json
import os
import sys
import threading
import time
from typing import Optional
import websockets as _websockets

from binance_api import bapi as api
from binance_api import bapi_ws_bridge as _bapi_ws_bridge
from binance_api.bapi_ws import (
    PublicPriceStreamBase,
    WS_CLOSE_TIMEOUT,
    WS_PING_INTERVAL,
    WS_PING_TIMEOUT,
    WS_RECV_TIMEOUT,
)
from botcore import (
    is_live_network_backfill_enabled,
    is_rth_gate_enabled,
    is_us_rth_open,
    safe_nonneg_float,
    safe_pos_float,
    safe_pos_int,
)
from cacheInterface import (
    CM_LONG_ARCHIVE_MEMORY_ROWS,
    CM_LONG_ARCHIVE_MONTHS,
    CacheManagerInterface,
    _cm_override,
    _jsonl_line,
    _monotonic_time,
    _norm_symbol_keys,
    _now_time,
)
from cachepaths import (
    cache_path,
    canonical_symbol_key,
    clean_symbol_key,
    normalize_venue_name,
    split_symbol_assets,
)
import instrument_registry as sym
from instrument_registry import (
    casefold_get,
    is_equity_symbol,
    is_symbol_market_open,
)
import providers.market_api as _market_api
from providers.base import (
    _parse_bool_flag,
    live_price_fetch_context,
)
from state_io import (
    atomic_snapshot_file,
    atomic_text_writer as atomic_write,
    discard_generation_file,
    read_json_dict,
    read_tail_lines,
    save_json_state,
)
import utils as u

_NB_FETCH_DEADLINE_SEC = 8


class CachePriceManagerBase(CacheManagerInterface, ABC):
    """Abstract base contract (Level 2) for price-series caches."""

    cls_name = "CachePriceManagerBase"
    FILE_PREFIX: Optional[str] = None
    FILE_EXT: Optional[str] = None
    DEFAULT_FILENAME: Optional[str] = None

    def __init__(self, *args, **kwargs):
        if type(self) is CachePriceManagerBase:
            raise TypeError("Cannot instantiate abstract class CachePriceManagerBase")
        super().__init__(*args, **kwargs)

    @classmethod
    def disk_filename_for_symbol(cls, symbol: str = "") -> str:
        """Return the canonical cache filename for ``symbol`` owned by this concrete price cache class."""
        if cls.DEFAULT_FILENAME:
            return cls.DEFAULT_FILENAME
        if cls.FILE_PREFIX and cls.FILE_EXT:
            return f"{cls.FILE_PREFIX}{symbol}.{cls.FILE_EXT}"
        return ""

    def rebuild_fetchtimes(self):
        if not self.cache:
            return {}
        last_times = {}
        for symbol in self.symbols:
            entries = self.cache.get(symbol, [])
            if isinstance(entries, list) and entries:
                max_ts = max(
                    (self._entry_timestamp_ms(entry) for entry in entries), default=0
                )
                if max_ts > 0:
                    last_times[symbol] = max_ts
        return last_times

    def get_remote_items(self, symbol, startTime):
        if (
            is_equity_symbol(symbol)
            and is_rth_gate_enabled(getattr(self, "_enforce_rth_gate", None), rth_fn=is_us_rth_open)
            and not is_us_rth_open()
        ):
            with self.lock:
                if self.cache.get(symbol):
                    return []
        try:
            get_cpm = _cm_override("get_current_price_manager")
            if get_cpm is None:
                import cacheManager as _cm_mod

                get_cpm = _cm_mod.get_current_price_manager
            entry = get_cpm(start_sync=False).get_price(symbol)
        except Exception as e:
            print(f"[{self.cls_name}][Error] get_price {symbol}: {e}")
            return []

        if not entry or entry[1] is None:
            return []
        ts_ms, price = entry
        ts_val = safe_pos_int(ts_ms, default=None)
        price_val = safe_pos_float(price, default=None)
        if ts_val is None or price_val is None:
            return []
        with self.lock:
            entries = self.cache.get(symbol)
            if entries and ts_val < self._entry_timestamp_ms(entries[-1]):
                return []

        return [[ts_val, price_val]]

    def get_all_symbols_from_cache(self):
        return self.symbols


class CacheSparsePriceManager(CachePriceManagerBase):
    """Store sparse three-minute price samples (Level 3) for two years in JSONL."""

    FILE_PREFIX = "cache_sparse_price_"
    FILE_EXT = "jsonl"

    def __init__(self, sync_ts, symbols, filename, api_client=None, read_only=False):
        if api_client is None:
            api_client = _cm_override("api", api)
        super().__init__(
            sync_ts,
            symbols,
            filename,
            append_mode=True,
            api_client=api_client,
            append_persist=True,
            read_only=read_only,
        )


class Cache24PriceManager(CachePriceManagerBase):
    """Collect maximum-granularity prices (Level 3) over the latest ``KEEP_HOURS``."""

    KEEP_HOURS = 24
    FILE_PREFIX = "cache_24_price_"
    FILE_EXT = "json"
    _filter_fetchtimes_to_symbols = True

    def __init__(self, sync_ts, symbols, filename, api_client=None, read_only=False):
        if api_client is None:
            api_client = _cm_override("api", api)
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


class Cache365PriceManager(Cache24PriceManager):
    """Multi-month/multi-year dense archive variant (Level 3) using incremental JSONL persistence."""

    KEEP_HOURS = CM_LONG_ARCHIVE_MONTHS * 30 * 24
    RETENTION_DAYS = CM_LONG_ARCHIVE_MONTHS * 30
    LONG_TRIM_INTERVAL_SEC = 24 * 3600
    FILE_PREFIX = "cache_365_price_"
    FILE_EXT = "jsonl"

    def __init__(self, sync_ts, symbols, filename, api_client=None, read_only=False):
        if api_client is None:
            api_client = _cm_override("api", api)
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
            snapshot_fn = _cm_override("atomic_snapshot_file", atomic_snapshot_file)
            snapshot_fn(self.filename, backup_path)
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
                    discard_generation_file(backup_path)
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
            snapshot_fn = _cm_override("atomic_snapshot_file", atomic_snapshot_file)
            snapshot_fn(self.filename, archive)
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


class CacheCurrentPriceManager(CachePriceManagerBase):
    """Maintain the latest timestamped price per symbol (Level 3, WebSocket-first with HTTP fallback)."""

    WS_TIMEOUT_SEC = 15
    STALE_THRESHOLD_MS = 5_000
    FREQ_WINDOW_SEC = 60
    DEFAULT_FILENAME = "cache_current_price.json"

    def __init__(
        self,
        sync_ts,
        symbols,
        filename,
        ws_manager=None,
        api_client=None,
        market_api=None,
        provider_names=None,
        read_only=False,
    ):
        if api_client is None:
            api_client = _cm_override("api", api)
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
            if is_rth_sym and is_rth_gate_enabled(
                getattr(self, "_enforce_rth_gate", None), rth_fn=is_us_rth_open
            ):
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
        sym_mod = _cm_override("sym", sym)
        if (
            entry is not None
            and is_rth_gate_enabled(
                getattr(self, "_enforce_rth_gate", None), rth_fn=is_us_rth_open
            )
            and hasattr(sym_mod, "is_symbol_market_open")
            and not sym_mod.is_symbol_market_open(target_sym)
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


# ── Non-Binance Venue WebSocket Price Streams & Fallback Poller ───────────────


class _VenuePriceStreamBase(PublicPriceStreamBase):
    """Shared public WebSocket price stream base for non-Binance venues (Hyperliquid, Kraken)."""

    MIN_SAME_PRICE_EMIT_MS = 1000

    def __init__(self, symbols=None, *, ws_url: str):
        super().__init__(symbols=symbols)
        self.ws_url = ws_url
        self._last_emit_ts_ms: dict[str, int] = {}
        self._last_emit_price: dict[str, float] = {}

    def remove_symbol(self, symbol: str) -> bool:
        norm_s = self._norm_ws_symbol(symbol)
        if not super().remove_symbol(symbol):
            return False
        if norm_s:
            with self._lock:
                self._last_emit_ts_ms.pop(norm_s, None)
                self._last_emit_price.pop(norm_s, None)
        return True

    def _notify_subscribers(self, symbol: str, items) -> None:
        ts_ms = None
        px = None
        if (
            isinstance(items, list)
            and items
            and isinstance(items[-1], (list, tuple))
            and len(items[-1]) >= 2
        ):
            ts_ms = safe_pos_int(items[-1][0], default=None)
            px = safe_pos_float(items[-1][1], default=None)
        with self._lock:
            subscribers = list(self._subscribers)
            if ts_ms is not None and px is not None:
                prev_ts = self._last_emit_ts_ms.get(symbol)
                prev_px = self._last_emit_price.get(symbol)
                if (
                    prev_ts is not None
                    and prev_px == px
                    and 0 <= (ts_ms - prev_ts) < self.MIN_SAME_PRICE_EMIT_MS
                ):
                    now_s = time.time()
                    for sub in subscribers:
                        ws_map = getattr(sub, "_ws_stream_event_ts_per_symbol", None)
                        if isinstance(ws_map, dict):
                            ws_map[symbol] = now_s
                    return
                self._last_emit_ts_ms[symbol] = ts_ms
                self._last_emit_price[symbol] = px
        for sub in subscribers:
            try:
                sub.on_items_update(symbol, items)
            except Exception as exc:
                builtins.print(
                    f"[{self.__class__.__name__}] subscriber error ({symbol}): {exc}"
                )

    async def _connect_and_run(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                has_symbols = bool(self._symbols)
            if has_symbols:
                break
            await self._interruptible_sleep(1.0)
        if not self._symbols or self._stop_event.is_set():
            return
        async with _websockets.connect(
            self.ws_url,
            ping_interval=WS_PING_INTERVAL,
            ping_timeout=WS_PING_TIMEOUT,
            close_timeout=WS_CLOSE_TIMEOUT,
        ) as ws:
            sub_msg = self._build_subscribe_payload()
            if sub_msg:
                await ws.send(sub_msg)
            for extra_msg in self._build_extra_subscribe_payloads():
                if extra_msg:
                    await ws.send(extra_msg)
            while not self._stop_event.is_set():
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=WS_RECV_TIMEOUT)
                    self._process_message(raw)
                except asyncio.TimeoutError:
                    continue
                except _websockets.exceptions.ConnectionClosed:
                    break
                except Exception:
                    continue

    def _build_subscribe_payload(self) -> Optional[str]:
        raise NotImplementedError

    def _build_extra_subscribe_payloads(self) -> tuple[str, ...]:
        return ()

    def _process_message(self, raw: str, *, now_ms: Optional[int] = None) -> None:
        raise NotImplementedError


class HyperliquidPriceStream(_VenuePriceStreamBase):
    """Real-time Hyperliquid WebSocket stream (``allMids`` + optional ``userFills``)."""

    DEFAULT_WS_URL = "wss://api.hyperliquid.xyz/ws"
    _KNOWN_SPOT_PAIRS = {"HYPE": "@107"}

    def __init__(
        self,
        symbols=None,
        *,
        market_api=None,
        ws_url: str = DEFAULT_WS_URL,
        user_address: Optional[str] = None,
        fills_cache_path: Optional[str] = None,
        resolve_env_user_address: Optional[bool] = None,
    ):
        super().__init__(symbols=symbols, ws_url=ws_url)
        self.market_api = market_api
        self._resolved_spot_keys: dict[str, tuple[str, ...]] = {}
        self._fills_cache_path = fills_cache_path
        self._fills_ws_seeded = False
        self._last_fills_touch_ms = 0
        if user_address is None and is_live_network_backfill_enabled(resolve_env_user_address):
            from providers.base import env_value

            hl_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "hyperliquid"
            )
            user_address = os.environ.get("HL_ACCOUNT_ADDRESS") or env_value(
                hl_dir, "HL_ACCOUNT_ADDRESS"
            )
        self._user_address = str(user_address or "").strip() or None

    def remove_symbol(self, symbol: str) -> bool:
        norm_s = self._norm_ws_symbol(symbol)
        removed = super().remove_symbol(symbol)
        if removed and norm_s:
            with self._lock:
                self._resolved_spot_keys.pop(norm_s, None)
        return removed

    def _candidate_mids_keys(self, symbol: str) -> tuple[str, ...]:
        with self._lock:
            cached = self._resolved_spot_keys.get(symbol)
        if cached:
            return cached
        split = split_symbol_assets(symbol)
        base = split[0] if split else symbol.replace("USDC", "").replace("USD", "")
        keys: list[str] = []
        if self.market_api is not None and hasattr(
            self.market_api, "provider_by_name"
        ):
            try:
                prov = self.market_api.provider_by_name("hyperliquid")
                if prov is not None and (
                    not hasattr(prov, "supports_symbol")
                    or prov.supports_symbol(symbol)
                ):
                    pair_fn = getattr(prov, "_pair", None)
                    spot_pair = (
                        pair_fn()
                        if callable(pair_fn)
                        else getattr(prov, "_spot_pair", None)
                    )
                    if isinstance(spot_pair, str) and spot_pair.strip():
                        keys.append(spot_pair.strip())
            except Exception:
                pass
        known_spot = self._KNOWN_SPOT_PAIRS.get(base.upper())
        if known_spot and known_spot not in keys:
            keys.append(known_spot)
        for fallback in (f"{base}/USDC", symbol, base):
            if fallback and fallback not in keys:
                keys.append(fallback)
        resolved = tuple(keys)
        with self._lock:
            self._resolved_spot_keys[symbol] = resolved
        return resolved

    def _build_subscribe_payload(self) -> str:
        return _json.dumps(
            {"method": "subscribe", "subscription": {"type": "allMids"}}
        )

    def _build_extra_subscribe_payloads(self) -> tuple[str, ...]:
        self._fills_ws_seeded = False
        if not self._user_address:
            return ()
        return (
            _json.dumps(
                {
                    "method": "subscribe",
                    "subscription": {
                        "type": "userFills",
                        "user": self._user_address,
                    },
                }
            ),
        )

    def _resolve_fills_cache_file(self) -> str:
        return (
            str(self._fills_cache_path)
            if self._fills_cache_path
            else cache_path("cache_trade_hyperliquid.json")
        )

    def _handle_user_fills(self, data: object, *, ts_ms: int) -> None:
        raw_fills = (
            data.get("fills")
            if isinstance(data, dict)
            else (data if isinstance(data, list) else None)
        )
        if not isinstance(raw_fills, list):
            return
        is_snapshot = (
            bool(data.get("isSnapshot")) if isinstance(data, dict) else False
        )
        cache_file = self._resolve_fills_cache_file()
        loaded = read_json_dict(cache_file, default=None, recover_previous=True)
        cache_obj = (
            loaded
            if isinstance(loaded, dict)
            and isinstance(loaded.get("items"), dict)
            and isinstance(loaded.get("fetchtime"), dict)
            else {"items": {}, "fetchtime": {}}
        )
        with self._lock:
            symbols = self._symbols
        for s_sym in symbols:
            spot_coins = {
                k.upper()
                for k in self._candidate_mids_keys(s_sym)
                if isinstance(k, str) and (k.startswith("@") or "/" in k)
            }
            sym_fills = [
                dict(f)
                for f in raw_fills
                if isinstance(f, dict)
                and str(f.get("coin") or "").strip().upper() in spot_coins
            ]
            if is_snapshot:
                cache_obj["items"][s_sym] = sym_fills
                cache_obj["fetchtime"][s_sym] = ts_ms
            elif sym_fills:
                bucket = list(cache_obj["items"].get(s_sym) or [])
                seen_keys = {
                    (f.get("tid"), f.get("oid"), f.get("time"))
                    for f in bucket
                    if isinstance(f, dict)
                }
                for f in sym_fills:
                    fk = (f.get("tid"), f.get("oid"), f.get("time"))
                    if fk not in seen_keys:
                        bucket.append(f)
                        seen_keys.add(fk)
                cache_obj["items"][s_sym] = bucket
                cache_obj["fetchtime"][s_sym] = ts_ms
            elif s_sym in cache_obj["items"] and s_sym in cache_obj["fetchtime"]:
                cache_obj["fetchtime"][s_sym] = ts_ms
        save_json_state(cache_file, cache_obj)
        if is_snapshot:
            self._fills_ws_seeded = True
        self._last_fills_touch_ms = ts_ms

    def _touch_fills_cache_if_needed(self, ts_ms: int) -> None:
        if (
            not self._fills_ws_seeded
            or (ts_ms - self._last_fills_touch_ms) < 15_000
        ):
            return
        try:
            cache_file = self._resolve_fills_cache_file()
            loaded = read_json_dict(cache_file, default=None, recover_previous=True)
            if (
                isinstance(loaded, dict)
                and isinstance(loaded.get("fetchtime"), dict)
                and isinstance(loaded.get("items"), dict)
            ):
                with self._lock:
                    symbols = self._symbols
                changed = False
                for s_sym in symbols:
                    if s_sym in loaded["fetchtime"] and s_sym in loaded["items"]:
                        loaded["fetchtime"][s_sym] = ts_ms
                        changed = True
                if changed:
                    save_json_state(cache_file, loaded)
                self._last_fills_touch_ms = ts_ms
        except Exception:
            pass

    def _process_message(self, raw: str, *, now_ms: Optional[int] = None) -> None:
        try:
            envelope = _json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(envelope, dict):
                return
            ts_ms = safe_pos_int(now_ms, default=None) or int(time.time() * 1000)
            channel = envelope.get("channel")
            if channel == "userFills":
                self._handle_user_fills(envelope.get("data"), ts_ms=ts_ms)
                return
            if channel != "allMids":
                return
            data = envelope.get("data")
            mids = data.get("mids") if isinstance(data, dict) else None
            if not isinstance(mids, dict):
                return
            with self._lock:
                symbols = self._symbols
            for s_sym in symbols:
                px = None
                for k in self._candidate_mids_keys(s_sym):
                    raw_val = casefold_get(mids, k)
                    if not isinstance(raw_val, bool):
                        px = safe_pos_float(raw_val, default=None)
                    if px is not None:
                        break
                if px is not None:
                    with self._lock:
                        self._prices[s_sym] = px
                    self._notify_subscribers(s_sym, [[ts_ms, px]])
            self._touch_fills_cache_if_needed(ts_ms)
        except Exception:
            return


class KrakenPriceStream(_VenuePriceStreamBase):
    """Real-time Kraken public WebSocket v1 ``ticker`` stream."""

    DEFAULT_WS_URL = "wss://ws.kraken.com"

    def __init__(self, symbols=None, *, ws_url: str = DEFAULT_WS_URL):
        super().__init__(symbols=symbols, ws_url=ws_url)
        self._ws_pairs: list[str] = []
        self._pair_to_symbol: dict[str, str] = {}
        for s_sym in self._symbols:
            self._register_symbol_pair(s_sym)

    def _register_symbol_pair(self, s_sym: str) -> None:
        split = split_symbol_assets(s_sym)
        ws_pair = f"{split[0]}/{split[1]}" if split else s_sym
        if ws_pair not in self._ws_pairs:
            self._ws_pairs.append(ws_pair)
        self._pair_to_symbol[ws_pair.upper()] = s_sym
        self._pair_to_symbol[clean_symbol_key(s_sym)] = s_sym
        for alias_group in (("BTC", "XBT"), ("DOGE", "XDG")):
            if split and split[0] in alias_group:
                for alias in alias_group:
                    self._pair_to_symbol[f"{alias}/{split[1]}"] = s_sym
                    self._pair_to_symbol[f"{alias}{split[1]}"] = s_sym

    def add_symbol(self, symbol: str) -> bool:
        norm_s = self._norm_ws_symbol(symbol)
        if not norm_s:
            return False
        with self._lock:
            if norm_s in self._subscribed:
                return False
            self._subscribed.add(norm_s)
            self._symbols = tuple(dict.fromkeys((*self._symbols, norm_s)))
            self._register_symbol_pair(norm_s)
        return True

    def remove_symbol(self, symbol: str) -> bool:
        norm_s = self._norm_ws_symbol(symbol)
        if not norm_s:
            return False
        with self._lock:
            if norm_s not in self._subscribed:
                return False
            self._subscribed.discard(norm_s)
            self._symbols = tuple(s for s in self._symbols if s != norm_s)
            self._prices.pop(norm_s, None)
            self._last_emit_ts_ms.pop(norm_s, None)
            self._last_emit_price.pop(norm_s, None)
            self._ws_pairs.clear()
            self._pair_to_symbol.clear()
            for s_sym in self._symbols:
                self._register_symbol_pair(s_sym)
        return True

    def _build_subscribe_payload(self) -> str:
        with self._lock:
            pairs = list(self._ws_pairs)
        return _json.dumps(
            {
                "event": "subscribe",
                "pair": pairs,
                "subscription": {"name": "ticker"},
            }
        )

    def _resolve_symbol(self, pair_label: str) -> Optional[str]:
        raw_u = str(pair_label or "").strip().upper()
        if not raw_u:
            return None
        with self._lock:
            return self._pair_to_symbol.get(raw_u) or self._pair_to_symbol.get(
                clean_symbol_key(raw_u)
            )

    def _process_message(self, raw: str, *, now_ms: Optional[int] = None) -> None:
        try:
            msg = _json.loads(raw) if isinstance(raw, str) else raw
            ts_ms = safe_pos_int(now_ms, default=None) or int(time.time() * 1000)
            if isinstance(msg, dict):
                ev = str(msg.get("event") or "").strip().lower()
                if ev == "heartbeat" or (
                    ev == "systemstatus"
                    and str(msg.get("status") or "").strip().lower() == "online"
                ):
                    with self._lock:
                        known_prices = list(self._prices.items())
                    for s_sym, px in known_prices:
                        self._notify_subscribers(s_sym, [[ts_ms, px]])
                return
            if not isinstance(msg, list) or len(msg) < 4:
                return
            if msg[-2] != "ticker" or not isinstance(msg[1], dict):
                return
            s_sym = self._resolve_symbol(str(msg[-1] or ""))
            if not s_sym:
                return
            ticker = msg[1]
            c_raw = ticker.get("c")
            b_raw = ticker.get("b")
            a_raw = ticker.get("a")
            c_val = c_raw[0] if isinstance(c_raw, list) and c_raw else c_raw
            b_val = b_raw[0] if isinstance(b_raw, list) and b_raw else b_raw
            a_val = a_raw[0] if isinstance(a_raw, list) and a_raw else a_raw
            from providers.base import MarketDataProvider

            price = MarketDataProvider.clamp_price_to_bbo_mid(c_val, b_val, a_val)
            if price is None:
                return
            with self._lock:
                self._prices[s_sym] = price
            self._notify_subscribers(s_sym, [[ts_ms, price]])
        except Exception:
            return


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

    def _poll():
        with live_price_fetch_context():
            return market_api.get_current_price(**kwargs)

    fut = pool.submit(_poll)
    try:
        return fut.result(timeout=deadline_sec)
    except _futures.TimeoutError:
        fut.cancel()
        raise


class _NonBinanceTrendPoller:
    """Own the non-Binance WebSocket streams, fallback poller thread, and executor lifecycle."""

    def __init__(self, thread, pool, stop_event, ws_streams=None):
        self.thread = thread
        self.pool = pool
        self.stop_event = stop_event
        self.ws_streams = tuple(ws_streams or ())

    def stop(self, timeout=5.0):
        self.stop_event.set()
        for ws in self.ws_streams:
            try:
                stop_ev = getattr(ws, "_stop_event", None)
                if stop_ev is not None:
                    stop_ev.set()
            except Exception:
                pass
        eff_timeout = safe_nonneg_float(timeout, default=5.0)
        if eff_timeout is None:
            eff_timeout = 5.0
        total_deadline = time.monotonic() + eff_timeout
        ws_budget = min(2.0, eff_timeout)
        ws_deadline = time.monotonic() + ws_budget
        for ws in self.ws_streams:
            try:
                rem = max(0.05, ws_deadline - time.monotonic())
                ws.stop(timeout=rem)
            except Exception:
                pass
        if self.thread is not threading.current_thread():
            self.thread.join(timeout=max(0.0, total_deadline - time.monotonic()))
        self.pool.shutdown(wait=False, cancel_futures=True)
        try:
            for th in list(getattr(self.pool, "_threads", ())):
                _futures.thread._threads_queues.pop(th, None)
        except Exception:
            pass
        return not self.thread.is_alive()


def _start_nonbinance_trend_poller(
    cpm,
    symbols,
    interval_sec=10,
    fetch_deadline_sec=_NB_FETCH_DEADLINE_SEC,
    provider_names=None,
    *,
    start_ws=None,
):
    """Feed short trends for non-Binance symbols using WebSocket-first + REST fallback."""
    symbols = tuple(str(symbol or "").strip() for symbol in symbols)
    bindings = {
        str(symbol or "").strip().upper(): str(provider_name or "").strip()
        for symbol, provider_name in dict(provider_names or {}).items()
    }
    missing = [
        symbol
        for symbol in symbols
        if (
            not symbol
            or not bindings.get(symbol.upper())
            or normalize_venue_name(bindings[symbol.upper()], default="") == "binance"
        )
    ]
    if missing:
        raise ValueError(
            "explicit provider bindings are required for non-Binance symbols: "
            + ", ".join(missing)
        )
    cpm.bind_providers(bindings)

    if start_ws is None:
        auto_ws_env = (
            os.environ.get("BINANCE_AUTO_START_WEBSOCKETS", "1").strip().lower()
        )
        enable_ws = (
            auto_ws_env not in ("0", "false", "no", "off")
            and is_live_network_backfill_enabled()
        )
    else:
        enable_ws = _parse_bool_flag(start_ws, default=False)

    ws_streams = []
    if enable_ws and hasattr(cpm, "on_items_update"):
        for venue_key, ws_name, build_ws in (
            (
                "hyperliquid",
                "HyperliquidPriceWS",
                lambda syms: HyperliquidPriceStream(
                    syms, market_api=getattr(cpm, "market_api", None)
                ),
            ),
            ("kraken", "KrakenPriceWS", KrakenPriceStream),
        ):
            v_syms = [
                s
                for s in symbols
                if normalize_venue_name(bindings[s.upper()], default="") == venue_key
            ]
            if v_syms:
                ws_inst = build_ws(v_syms)
                if hasattr(cpm, "attach_ws_manager"):
                    cpm.attach_ws_manager(ws_inst)
                else:
                    ws_inst.subscribe(cpm)
                ws_inst.start(name=ws_name, daemon=True)
                ws_streams.append(ws_inst)

    pool = _futures.ThreadPoolExecutor(
        max_workers=max(2, len(symbols)), thread_name_prefix="NBTrendFetch"
    )
    stop_event = threading.Event()

    def run():
        rth_seeded: set[str] = set()
        empty_backoff_ts: dict[str, float] = {}
        cpm_cache = getattr(cpm, "cache", None)
        extract_fn = getattr(cpm, "_extract_price_and_ts", None)
        if isinstance(cpm_cache, dict) and callable(extract_fn):
            for s in symbols:
                try:
                    cached_px, _ = extract_fn(cpm_cache.get(s))
                    if safe_pos_float(cached_px, default=None) is not None:
                        rth_seeded.add(s)
                except Exception:
                    pass
        if ws_streams:
            real_ws_fn = getattr(cpm, "_real_ws_is_healthy", None)
            if callable(real_ws_fn):
                for _ in range(20):
                    if stop_event.is_set() or all(real_ws_fn(s) for s in symbols):
                        break
                    stop_event.wait(0.1)
        while not stop_event.is_set():
            for s in symbols:
                if stop_event.is_set():
                    break
                real_ws_fn = getattr(cpm, "_real_ws_is_healthy", None)
                if callable(real_ws_fn) and real_ws_fn(s):
                    continue
                rth_gate_active = is_rth_gate_enabled(
                    getattr(cpm, "_enforce_rth_gate", None), rth_fn=is_us_rth_open
                )
                now_ts = time.time()
                if rth_gate_active and not is_symbol_market_open(s, now_ts):
                    if s in rth_seeded:
                        ws_ts_map = getattr(cpm, "_ws_last_event_ts_per_symbol", None)
                        if isinstance(ws_ts_map, dict):
                            ws_ts_map[s] = now_ts
                        continue
                    last_empty = empty_backoff_ts.get(s, 0.0)
                    if last_empty > 0.0 and (now_ts - last_empty) < 300.0:
                        ws_ts_map = getattr(cpm, "_ws_last_event_ts_per_symbol", None)
                        if isinstance(ws_ts_map, dict):
                            ws_ts_map[s] = now_ts
                        continue
                elif rth_gate_active:
                    last_empty = empty_backoff_ts.get(s, 0.0)
                    if last_empty > 0.0 and (now_ts - last_empty) < 60.0:
                        continue
                try:
                    p = _fetch_price_with_deadline(
                        cpm.market_api,
                        s,
                        pool,
                        fetch_deadline_sec,
                        provider_name=bindings[s.upper()],
                    )
                    pf = safe_pos_float(p, default=None)
                    if pf is not None:
                        rth_seeded.add(s)
                        empty_backoff_ts.pop(s, None)
                        ws_ts_map = getattr(cpm, "_ws_last_event_ts_per_symbol", None)
                        if isinstance(ws_ts_map, dict):
                            ws_ts_map[s] = time.time()
                        cpm._push_price(s, pf)
                    else:
                        empty_backoff_ts[s] = time.time()
                except _futures.TimeoutError:
                    builtins.print(
                        f"[NB-trend] {s}: fetch BLOCKED >{fetch_deadline_sec}s "
                        f"(poll deadline — probably DNS or the network) — skipping, retrying next cycle"
                    )
                except Exception as _e:
                    builtins.print(f"[NB-trend] {s}: {_e}")
            stop_event.wait(interval_sec)

    t = threading.Thread(target=run, name="NonBinanceTrendPoller", daemon=True)
    t.start()
    return _NonBinanceTrendPoller(t, pool, stop_event, ws_streams=ws_streams)
