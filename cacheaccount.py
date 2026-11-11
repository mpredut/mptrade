"""Account domain cache hierarchy (Level 2 abstract base + Level 3 concrete classes).

Provides:
- ``CacheAccountManagerBase(CacheManagerInterface, ABC)``:
  Abstract base class for account-domain caches (``Order``, ``Trade``, ``AssetValue``)
  with schema-v2 manifests, ``.previous`` crash recovery, and reader synchronization.
- ``CacheTradeManager(CacheAccountManagerBase)``:
  Immutable trade fill cache backed by byte-boundary JSONL and Blake2s digests.
- ``CacheOrderManager(CacheAccountManagerBase)``:
  Order aggregate cache backed by versioned JSON snapshots.
- ``CacheAssetValueManager(CacheAccountManagerBase)``:
  Account/portfolio NAV history cache backed by append-only JSONL.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import builtins
from collections import defaultdict
from datetime import datetime
import json
import os
import sys
import threading
from typing import Any, Optional, Sequence
import uuid

import account_cache_health
from botcore import (
    _is_fresh_obs_ts,
    _parse_bool_flag,
    _ts_ms_to_s,
    clean_order_id,
    compute_venue_nav_breakdown,
    extract_order_id,
    extract_order_price,
    extract_order_qty,
    extract_order_side,
    extract_timestamp_ms,
    extract_timestamp_s,
    is_auto_start_websockets_enabled,
    is_live_network_backfill_enabled,
    log,
    normalize_epoch_sec,
    normalize_side,
    parse_symbol_list,
    resolve_now_ms,
    resolve_now_ts,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
    safe_pos_int,
    strict_ascii_int,
)
from cacheInterface import (
    CacheManagerInterface,
    _CACHE_META_SCHEMA_VERSION,
    _cm_override,
    _monotonic_time,
    _now_time,
    _sleep_time,
)
import cacheintegrity as _cache_integrity
from cachepaths import (
    CACHE_ASSET_VALUE_FILENAME,
    CACHE_BINANCE_ORDER_FILENAME,
    CACHE_BINANCE_TRADE_FILENAME,
    CACHE_CURRENT_PRICE_FILENAME,
    CACHE_HYPERLIQUID_TRADE_FILENAME,
    CACHE_KRAKEN_TRADE_FILENAME,
    CACHE_T212_TRADE_FILENAME,
    cache_path,
    cache_symbol_variants,
    cachedb_root_for_file,
    clean_symbol_key,
    is_primary_stream_venue,
    normalize_venue_name,
    resolve_cachedb_filepath,
)
import instrument_registry as sym
from instrument_registry import casefold_get
from providers.base import MarketDataProvider
from state_io import (
    atomic_snapshot_file,
    atomic_text_writer,
    atomic_write_json,
    discard_generation_file,
    meta_file_path,
    parse_json_payload,
    read_json_dict,
    read_json_dict_recovering,
    read_recovering_previous,
    save_json_state,
    stage_previous_generation,
    format_jsonl_bucket_line,
)

_bn_acc_prov_cls = MarketDataProvider.resolve_venue_provider_class("binance")
api = (
    _bn_acc_prov_cls.resolve_default_api_client()
    if _bn_acc_prov_cls is not None
    else None
)
apiorders = (
    _bn_acc_prov_cls.resolve_allorders_module()
    if _bn_acc_prov_cls is not None
    else None
)
_bapi_ws_bridge = (
    _bn_acc_prov_cls.resolve_ws_bridge_module()
    if _bn_acc_prov_cls is not None
    else None
)


class CacheAccountManagerBase(CacheManagerInterface, ABC):
    """Abstract base contract (Level 2) for account-domain caches (``Order``, ``Trade``, ``AssetValue``)."""

    _default_api_client = api
    _is_trade_account_cache = False
    _is_order_account_cache = False
    _persisted_data_version = ""
    _persisted_stream_id = ""
    _persisted_revision = 0
    _persisted_committed_bytes = 0
    _persisted_content_digest = ""
    _loaded_data_version = ""
    _loaded_stream_id = ""
    _loaded_revision = 0
    _loaded_committed_bytes = 0
    _loaded_content_digest = ""
    _loaded_counts: dict = {}
    _account_cache_dirty = False
    _last_complete_sync_at_ms = 0

    def __init__(self, *args, **kwargs):
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
        self._account_cache_dirty = False
        self._last_complete_sync_at_ms = 0
        super().__init__(*args, **kwargs)

    def _is_account_cache(self):
        return bool(self._is_trade_account_cache or self._is_order_account_cache)

    def _should_poll(self):
        poll_fn = _cm_override(
            "_should_poll_for_manager", _bapi_ws_bridge.should_poll_for_manager
        )
        if callable(poll_fn):
            return bool(poll_fn(self.cls_name))
        return True

    def _reload_from_disk(self):
        super()._reload_from_disk()
        with self.lock:
            self._account_cache_dirty = False

    def _snapshot_memory_state_locked(self):
        return (
            *super()._snapshot_memory_state_locked(),
            self._account_cache_dirty,
        )

    def _restore_memory_state_locked(self, state) -> None:
        if state is None:
            return
        if len(state) == 5:
            *base_state, self._account_cache_dirty = state
            super()._restore_memory_state_locked(tuple(base_state))
        else:
            super()._restore_memory_state_locked(state)

    def get_persisted_sync_state(self) -> tuple[str, int]:
        """Return ``(data_version, min_cursor_ms)`` from loaded/persisted account cache state."""
        version = str(
            getattr(self, "_persisted_data_version", "")
            or getattr(self, "_loaded_data_version", "")
            or ""
        )
        cursors = (
            getattr(self, "_rest_sync_cursor_per_symbol", None)
            or getattr(self, "fetchtime_per_symbol", None)
            or {}
        )
        sync_ms = (
            min(
                (
                    vi
                    for v in cursors.values()
                    if (vi := safe_pos_int(v, default=None)) is not None
                ),
                default=0,
            )
            if isinstance(cursors, dict)
            else 0
        )
        return version, sync_ms

    def _is_canonical_account_cache(self):
        sym_mod = _cm_override("sym", sym)
        return self._is_account_cache() and set(self.symbols) == set(sym_mod.symbols)

    def _mark_account_cache_dirty_locked(self):
        if self._is_account_cache():
            self._account_cache_dirty = True

    def update_cache_per_symbol(self, symbol, new_items):
        if not self._is_account_cache():
            return super().update_cache_per_symbol(symbol, new_items)
        if new_items is None:
            return
        self._persist_items(symbol, self._coerce_item_list(new_items))

    def _resolve_remote_query_start_time(self, symbol: str, fallback_ft):
        if self._is_account_cache():
            return min(
                self._rest_sync_cursor_per_symbol.get(symbol, fallback_ft),
                fallback_ft,
            )
        return fallback_ft

    def _on_symbol_remote_sync_completed(
        self, symbol: str, request_high_water: int
    ) -> None:
        if self._is_account_cache():
            with self.lock:
                cursor_ms = max(0, (safe_nonneg_int(request_high_water, default=0) or 0) - 1)
                self._rest_sync_cursor_per_symbol[symbol] = cursor_ms
                self.fetchtime_per_symbol[symbol] = max(
                    safe_nonneg_int(self.fetchtime_per_symbol.get(symbol, 0), default=0) or 0,
                    cursor_ms,
                )
                self._mark_account_cache_dirty_locked()

    def _on_complete_remote_sync_cycle(
        self, account_cycle_high_water: Optional[int]
    ) -> None:
        if self._is_account_cache():
            with self.lock:
                self._last_complete_sync_at_ms = account_cycle_high_water or 0
        _bapi_ws_bridge.clear_ws_rest_resync_for_manager(self.cls_name)

    def _publish_periodic_sync_health(self, sync_complete, persisted) -> None:
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
            ach = _cm_override("account_cache_health", account_cache_health)
            ach.record_successful_sync(
                self.cls_name, version, now_ms=sync_at_ms
            )

    @abstractmethod
    def _is_valid_item(self, item) -> bool:
        """Validate one account row before memory or disk admission."""

    @abstractmethod
    def _ensure_account_version(self, required_version: str) -> None:
        """Synchronize in-memory state with ``required_version``; subclasses override."""
        raise account_cache_health.AccountCacheNotReady(self._account_reader_reason())

    def _account_meta_snapshot(
        self,
        *,
        data_version,
        fetchtime,
        counts,
        rest_sync_cursor=None,
        stream_id="",
        revision=0,
        committed_bytes=None,
        content_digest=None,
    ):
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

    def _apply_manifest_identity_locked(self, metadata, counts):
        version = str(metadata["data_version"])
        self._persisted_data_version = version
        self._persisted_stream_id = str(metadata.get("stream_id") or "")
        self._persisted_revision = safe_nonneg_int(metadata.get("revision"), default=0) or 0
        self._persisted_committed_bytes = safe_nonneg_int(metadata.get("committed_bytes"), default=0) or 0
        self._persisted_content_digest = str(metadata.get("content_digest") or "")
        self._loaded_data_version = version
        self._loaded_stream_id = self._persisted_stream_id
        self._loaded_revision = self._persisted_revision
        self._loaded_committed_bytes = self._persisted_committed_bytes
        self._loaded_content_digest = self._persisted_content_digest
        self._loaded_counts = dict(counts)
        self._account_cache_dirty = False
        return version

    def _remember_account_commit_locked(self, metadata):
        """Remember and publish one durable account-cache version."""
        version = self._apply_manifest_identity_locked(
            metadata, metadata.get("counts") or {}
        )
        if self._is_canonical_account_cache():
            try:
                account_cache_health.record_persisted_version(self.cls_name, version)
            except Exception as exc:
                builtins.print(
                    f"[{self.cls_name}][Error] could not publish durable "
                    f"cache version {version}: {exc}"
                )

    def _validate_account_rows_locked(self):
        """Reject any financial row that strict readers could not accept."""
        _cache_integrity.validate_account_cache_state(
            cls_name=self.cls_name,
            is_trade_cache=bool(self._is_trade_account_cache),
            is_order_cache=bool(self._is_order_account_cache),
            cache=self.cache,
            fetchtime_per_symbol=self.fetchtime_per_symbol,
            rest_sync_cursor_per_symbol=self._rest_sync_cursor_per_symbol,
            is_valid_item_fn=self._is_valid_item,
        )

    def _account_reader_reason(self):
        is_trade = bool(self._is_trade_account_cache)
        prefix = "trade" if is_trade else "order"
        return f"{prefix}_cache_reader_not_current"

    def _stage_prior_account_generation_locked(
        self, previous_metadata, validate_recovering_fn
    ):
        """Validate and stage the prior certified generation before overwriting active data."""
        if previous_metadata is not None:
            validate_recovering_fn(previous_metadata)
            return stage_previous_generation(self.filename), True
        if os.path.exists(self.filename):
            return stage_previous_generation(self.filename), True
        discard_generation_file(self._previous_generation_path())
        return None, False

    def _load_versioned_account_meta(self, *, optional=False, required_version=None):
        """Load and validate the small durable account-cache manifest."""
        is_trade = bool(self._is_trade_account_cache)
        return _cache_integrity.load_versioned_account_meta_file(
            meta_file_path(self.filename),
            is_trade_cache=is_trade,
            reader_reason=self._account_reader_reason(),
            optional=optional,
            required_version=required_version,
            schema_version=_CACHE_META_SCHEMA_VERSION,
        )

    def _read_snapshot_recovering_previous_locked(self, reader_fn, metadata):
        return read_recovering_previous(
            self.filename,
            lambda candidate_path: reader_fn(metadata, path=candidate_path),
            backup_path=self._previous_generation_path(),
        )

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
            self._ensure_account_version(required_version)

    def sync_account_reader_if_needed(self):
        """Refresh a read-only account cache when durable disk state advances."""
        if (
            not self._is_account_cache()
            or self.save_state
            or getattr(self, "data_sync_mode", None) == self.MODE_LIVE_READER
        ):
            return
        for _attempt in range(2):
            try:
                metadata = (
                    self._load_versioned_account_meta(optional=True)
                    if (
                        self.append_persist
                        or not getattr(self, "_is_trade_account_cache", False)
                    )
                    else None
                )
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

    @staticmethod
    def _is_valid_trade_fill_row(trade) -> bool:
        if not isinstance(trade, dict):
            return False
        required_keys = (
            "symbol",
            "id",
            "orderId",
            "price",
            "qty",
            "time",
            "isBuyer",
        )
        if not all(key in trade for key in required_keys):
            return False
        return (
            isinstance(trade["symbol"], str)
            and bool(trade["symbol"].strip())
            and strict_ascii_int(trade["id"], min_value=0) is not None
            and strict_ascii_int(trade["orderId"], min_value=1) is not None
            and safe_pos_float(trade["price"], default=None) is not None
            and safe_pos_float(trade["qty"], default=None) is not None
            and strict_ascii_int(trade["time"], min_value=1) is not None
            and isinstance(trade["isBuyer"], bool)
        )

    def _paginate_validated_trades(self, symbol, startTime, row_label: str = "trade"):
        """Paginate through venue trade fills and validate every returned fill row."""
        rows = MarketDataProvider.fetch_venue_remote_trades(
            getattr(self, "venue", "binance"),
            symbol,
            startTime,
            api_client=self.api_client,
            allorders_mod=apiorders,
        )
        for row in rows:
            if (
                not self._is_valid_trade_fill_row(row)
                or str(row["symbol"]).upper() != str(symbol).upper()
            ):
                raise RuntimeError(
                    f"{self.cls_name} received a malformed Binance {row_label} row: {row!r}"
                )
        return rows


class CacheTradeManager(CacheAccountManagerBase):
    """Unified multi-venue trade fill cache (Level 3) supporting both versioned append-only JSONL (`append_persist=True`) and JSON snapshot dicts (`append_persist=False`)."""

    _is_trade_account_cache = True
    append_persist: bool = True
    DEFAULT_FILENAME = CACHE_BINANCE_TRADE_FILENAME
    VENUE_FILENAMES: dict[str, str] = {
        "binance": CACHE_BINANCE_TRADE_FILENAME,
        "kraken": CACHE_KRAKEN_TRADE_FILENAME,
        "hyperliquid": CACHE_HYPERLIQUID_TRADE_FILENAME,
        "t212": CACHE_T212_TRADE_FILENAME,
    }

    @classmethod
    def _normalize_trade_venue(
        cls,
        venue: Optional[str] = None,
        filename: Optional[str] = None,
        symbols=None,
    ) -> str:
        v_norm = normalize_venue_name(venue)
        if v_norm in cls.VENUE_FILENAMES:
            return v_norm
        base_f = os.path.basename(str(filename or "")).lower()
        for cand_v, cand_f in cls.VENUE_FILENAMES.items():
            if base_f == cand_f.lower() or f"_{cand_v}." in base_f:
                return cand_v
        if not base_f and symbols:
            first_sym = (
                symbols
                if isinstance(symbols, str)
                else (next(iter(symbols), None) if isinstance(symbols, (list, tuple, set)) else None)
            )
            if isinstance(first_sym, str) and first_sym.strip():
                return cls.venue_for_symbol(first_sym)
        return "binance"

    @staticmethod
    def venue_for_symbol(symbol: object, *, default: str = "binance") -> str:
        """Resolve the configured venue ('binance', 'kraken', 'hyperliquid', 't212') for ``symbol`` via ``instrument_registry``."""
        try:
            return sym.venue_for_symbol(symbol, default=default)
        except Exception:
            return default

    @classmethod
    def disk_filename(cls, venue: str = "binance") -> str:
        """Return the canonical on-disk filename for ``venue``."""
        v_key = cls._normalize_trade_venue(venue)
        return str(cls.VENUE_FILENAMES.get(v_key, cls.DEFAULT_FILENAME) or "")

    @classmethod
    def resolve_cache_filepath(cls, cache_dir: Optional[str] = None, *, venue: str = "binance") -> str:
        """Resolve the canonical on-disk path for ``venue``'s trade cache file."""
        fname = cls.disk_filename(venue)
        if cache_dir:
            return str(resolve_cachedb_filepath(os.path.join(str(cache_dir), fname)))
        return str(cache_path(fname))

    @classmethod
    def load_sanitized_trade_snapshot(cls, path: Optional[str] = None, *, venue: str = "kraken") -> dict:
        """Load and sanitize a venue JSON trade snapshot ``{'items': ..., 'fetchtime': ...}`` with ``.previous`` recovery."""
        target = str(path) if path else cls.resolve_cache_filepath(venue=venue)
        loaded = read_json_dict(target, default=None, recover_previous=True)
        if (
            isinstance(loaded, dict)
            and isinstance(loaded.get("items"), dict)
            and isinstance(loaded.get("fetchtime"), dict)
        ):
            clean_items: dict[str, list[dict]] = {}
            for k, v in loaded["items"].items():
                if not isinstance(v, list):
                    continue
                valid_rows: list[dict] = []
                for row in v:
                    if not isinstance(row, dict):
                        continue
                    raw_id = (
                        row.get("id")
                        if row.get("id") is not None
                        else (row.get("tid") if row.get("tid") is not None else row.get("orderId"))
                    )
                    clean_id = clean_order_id(raw_id)
                    if clean_id is None:
                        continue
                    t_f = safe_pos_float(
                        row.get("time") if row.get("time") is not None else row.get("timestamp"),
                        default=None,
                    )
                    if t_f is None:
                        continue
                    row_copy = dict(row)
                    if "id" in row_copy or ("tid" not in row_copy and "orderId" not in row_copy):
                        row_copy["id"] = clean_id
                    row_copy["time"] = int(t_f)
                    if "isBuyer" in row_copy and isinstance(row_copy["isBuyer"], str):
                        if (ib_b := _parse_bool_flag(row_copy["isBuyer"], default=None)) is not None:
                            row_copy["isBuyer"] = ib_b
                        elif row_copy["isBuyer"].strip().lower() == "nan":
                            row_copy["isBuyer"] = False
                    valid_rows.append(row_copy)
                valid_rows.sort(key=lambda t: t["time"])
                clean_items[str(k)] = valid_rows
            clean_fetchtime = {}
            for k, v in loaded["fetchtime"].items():
                v_f = safe_nonneg_float(v, default=None)
                if v_f is not None and (not clean_items or str(k) in clean_items):
                    clean_fetchtime[str(k)] = v_f
            return {
                "items": clean_items,
                "fetchtime": clean_fetchtime,
            }
        return {"items": {}, "fetchtime": {}}

    @staticmethod
    def fill_dedup_key(row: object) -> Optional[tuple]:
        """Return a venue-agnostic deduplication key for a normalized fill or order row."""
        if not isinstance(row, dict):
            return None
        if row.get("id") is not None and not isinstance(row.get("id"), bool):
            return ("id", str(row["id"]))
        if row.get("tid") is not None or row.get("oid") is not None:
            return ("tid_oid", row.get("tid"), row.get("oid"), row.get("time"))
        if (r_oid := extract_order_id(row)):
            return ("oid", str(r_oid))
        return None

    @classmethod
    def matching_snapshot_keys(
        cls,
        mapping: dict,
        symbol: str,
        *,
        match_fn=None,
        candidate_keys=None,
    ) -> list[str]:
        """Return matching symbol keys in a venue trade snapshot ``mapping``."""
        if callable(match_fn):
            return [k for k in mapping if isinstance(k, str) and match_fn(symbol, k)]
        raw_cands = candidate_keys if candidate_keys is not None else (symbol,)
        cands = {str(c).upper() for c in raw_cands if c}
        return [
            k
            for k in mapping
            if isinstance(k, str) and (k.upper() in cands or clean_symbol_key(k) in cands)
        ]

    @staticmethod
    def _load_venue_snapshot_dict(cache_file: str, *, create: bool = False) -> Optional[dict]:
        """Load a venue trade snapshot ``{'items': {...}, 'fetchtime': {...}}`` (or initialize one when ``create=True``)."""
        data = read_json_dict_recovering(str(cache_file), default=None)
        if (
            isinstance(data, dict)
            and isinstance(data.get("items"), dict)
            and isinstance(data.get("fetchtime"), dict)
        ):
            return data
        if not create:
            return None
        base = data if isinstance(data, dict) else {}
        if not isinstance(base.get("items"), dict):
            base["items"] = {}
        if not isinstance(base.get("fetchtime"), dict):
            base["fetchtime"] = {}
        return base

    @classmethod
    def read_venue_fills_snapshot(
        cls,
        symbol: str,
        cache_file: str,
        *,
        max_stale_sec: float = 90.0,
        match_fn=None,
        candidate_keys=None,
        normalize_fn=None,
    ) -> Optional[list[dict]]:
        """Read normalized fills for ``symbol`` from a shared SSOT trade cache file (`items` + `fetchtime`)."""
        if not symbol or isinstance(symbol, bool) or not str(symbol).strip() or not cache_file:
            return None
        data = cls._load_venue_snapshot_dict(cache_file)
        if data is None:
            return None
        items = data["items"]
        ft = data["fetchtime"]
        keys = cls.matching_snapshot_keys(
            items, symbol, match_fn=match_fn, candidate_keys=candidate_keys
        )
        if not keys:
            return None
        now_s = _now_time()
        max_stale_s = 90.0 if max_stale_sec is None else safe_pos_float(max_stale_sec, default=None)
        if max_stale_s is None:
            return None
        sym_clean = clean_symbol_key(symbol) or str(symbol).strip().upper()
        best_rows: Optional[list[dict]] = None
        best_rank: Optional[tuple[float, int]] = None
        for key in keys:
            if not isinstance(items.get(key), list):
                continue
            ft_val = safe_pos_float(casefold_get(ft, key), default=None)
            if ft_val is None or not _is_fresh_obs_ts(_ts_ms_to_s(ft_val), now_s, max_stale_s):
                continue
            rank = (_ts_ms_to_s(ft_val), 1 if key.strip().upper() == sym_clean else 0)
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_rows = (
                    [normalize_fn(t) for t in items[key] if isinstance(t, dict)]
                    if callable(normalize_fn)
                    else [dict(t) for t in items[key] if isinstance(t, dict)]
                )
        return best_rows

    @classmethod
    def merge_venue_fills_into_dict(
        cls,
        cache_obj: dict,
        fills_by_symbol: dict,
        *,
        now_ms: Optional[int] = None,
        is_snapshot: bool = False,
        touch_symbols=None,
    ) -> bool:
        """Venue-agnostic in-memory merger of ``{symbol: [fill_dict, ...]}`` into ``cache_obj`` (`items` + `fetchtime`)."""
        if not isinstance(cache_obj, dict):
            return False
        items = cache_obj.setdefault("items", {})
        ft = cache_obj.setdefault("fetchtime", {})
        eff_now_ms = resolve_now_ms(now_ms)
        changed = False
        for sym_raw, new_rows in (fills_by_symbol or {}).items():
            if not sym_raw or isinstance(sym_raw, bool) or not isinstance(new_rows, list):
                continue
            sym_key = str(sym_raw).strip().upper()
            if not sym_key:
                continue
            if is_snapshot:
                clean_rows = [dict(r) for r in new_rows if isinstance(r, dict)]
                if items.get(sym_key) != clean_rows or ft.get(sym_key) != eff_now_ms:
                    changed = True
                items[sym_key] = clean_rows
                ft[sym_key] = eff_now_ms
                continue
            existing = items.get(sym_key)
            existing_list = existing if isinstance(existing, list) else []
            has_oid_only = any(
                isinstance(r, dict) and r.get("id") is None and r.get("tid") is None and extract_order_id(r)
                for r in new_rows
            )
            if has_oid_only:
                by_oid: dict[str, dict] = {}
                for row in existing_list:
                    if isinstance(row, dict) and (r_oid := extract_order_id(row)):
                        by_oid[str(r_oid)] = row
                for row in new_rows:
                    if not isinstance(row, dict):
                        continue
                    r_oid = extract_order_id(row)
                    if not r_oid:
                        continue
                    r_oid_s = str(r_oid)
                    if by_oid.get(r_oid_s) != row:
                        by_oid[r_oid_s] = dict(row)
                        changed = True
                merged = sorted(
                    by_oid.values(),
                    key=lambda r: (extract_timestamp_ms(r, default=0) or 0, str(extract_order_id(r) or "")),
                )
                items[sym_key] = merged
            else:
                bucket = list(existing_list)
                seen_keys = {
                    dk for r in bucket if (dk := cls.fill_dedup_key(r)) is not None
                }
                added = False
                for row in new_rows:
                    if not isinstance(row, dict):
                        continue
                    dk = cls.fill_dedup_key(row)
                    if dk is not None and dk not in seen_keys:
                        bucket.append(dict(row))
                        seen_keys.add(dk)
                        added = True
                        changed = True
                if added and all(isinstance(r, dict) and "time" in r and "tid" not in r for r in bucket):
                    bucket.sort(key=lambda r: extract_timestamp_ms(r, default=0) or 0)
                items[sym_key] = bucket
            old_ft = safe_pos_float(ft.get(sym_key), default=None)
            if old_ft is None or (eff_now_ms - old_ft) >= 60_000:
                changed = True
            ft[sym_key] = eff_now_ms
        if touch_symbols and cls.touch_venue_ws_fetchtimes_in_dict(
            cache_obj, eff_now_ms, symbols=touch_symbols
        ):
            changed = True
        return changed

    @classmethod
    def merge_venue_fills_snapshot(
        cls,
        cache_file: str,
        fills_by_symbol: dict,
        *,
        now_ms: Optional[int] = None,
        is_snapshot: bool = False,
        touch_symbols=None,
    ) -> bool:
        """Merge ``{symbol: [fill_dict, ...]}`` into a shared SSOT trade cache file (`items` + `fetchtime`)."""
        if not cache_file or not isinstance(fills_by_symbol, dict):
            return False
        if not fills_by_symbol and not touch_symbols and not is_snapshot:
            return False
        target_path = str(cache_file)
        try:
            cache_obj = cls._load_venue_snapshot_dict(target_path, create=True)
            changed = cls.merge_venue_fills_into_dict(
                cache_obj,
                fills_by_symbol,
                now_ms=now_ms,
                is_snapshot=is_snapshot,
                touch_symbols=touch_symbols,
            )
            if changed or is_snapshot or not os.path.exists(target_path):
                save_json_state(target_path, cache_obj)
            return True
        except Exception:
            return False

    @staticmethod
    def touch_venue_ws_fetchtimes_in_dict(
        cache_obj: dict,
        now_ms=None,
        *,
        symbols=None,
        create_missing: bool = False,
    ) -> bool:
        """Refresh ``fetchtime`` in ``cache_obj`` across active WS symbols."""
        if not isinstance(cache_obj, dict):
            return False
        items = cache_obj.setdefault("items", {})
        ft = cache_obj.setdefault("fetchtime", {})
        resolved_ms = resolve_now_ms(now_ms)
        changed = False
        for s_sym in (symbols or ()):
            if not s_sym or s_sym == "*":
                continue
            if create_missing:
                items.setdefault(s_sym, [])
                ft[s_sym] = resolved_ms
                changed = True
            elif s_sym in ft and s_sym in items and ft.get(s_sym) != resolved_ms:
                ft[s_sym] = resolved_ms
                changed = True
        if create_missing:
            for s_sym in list(items.keys()):
                ft[s_sym] = resolved_ms
                changed = True
        return changed

    @classmethod
    def invalidate_venue_fills_snapshot(
        cls,
        symbol: str,
        cache_file: str,
        *,
        match_fn=None,
        candidate_keys=None,
    ) -> None:
        """Expire ``fetchtime`` for ``symbol`` in the shared SSOT trade cache after order execution."""
        if not symbol or isinstance(symbol, bool) or not str(symbol).strip() or not cache_file:
            return
        try:
            data = read_json_dict_recovering(str(cache_file), default=None)
            if not isinstance(data, dict) or not isinstance(data.get("fetchtime"), dict):
                return
            keys = set(
                cls.matching_snapshot_keys(
                    data["fetchtime"],
                    symbol,
                    match_fn=match_fn,
                    candidate_keys=candidate_keys,
                )
            )
            if isinstance(data.get("items"), dict):
                item_cf = {
                    k.casefold()
                    for k in cls.matching_snapshot_keys(
                        data["items"],
                        symbol,
                        match_fn=match_fn,
                        candidate_keys=candidate_keys,
                    )
                }
                keys.update(
                    k
                    for k in data["fetchtime"]
                    if isinstance(k, str) and k.casefold() in item_cf
                )
            for k in keys:
                data["fetchtime"].pop(k, None)
            if keys:
                save_json_state(str(cache_file), data)
        except Exception:
            pass

    def __init__(
        self,
        sync_ts=60,
        symbols=None,
        filename=None,
        api_client=None,
        read_only=False,
        *,
        venue: Optional[str] = None,
        append_persist: Optional[bool] = None,
        stream_format: Optional[str] = None,
        data_sync_mode: Optional[str] = None,
    ):
        self.venue = self._normalize_trade_venue(venue, filename, symbols)
        eff_filename = str(filename) if filename else self.resolve_cache_filepath(venue=self.venue)
        default_disk_name = self.disk_filename(self.venue)
        is_json_snapshot_venue = not default_disk_name.lower().endswith(".jsonl")
        if symbols is None and is_json_snapshot_venue:
            try:
                eff_symbols = list(sym.symbols_for(self.venue))
            except Exception:
                eff_symbols = []
        else:
            eff_symbols = list(symbols) if isinstance(symbols, (list, tuple, set)) else (symbols if symbols is not None else [])
        if append_persist is not None:
            eff_append_persist = _parse_bool_flag(append_persist, default=False)
        elif stream_format is not None:
            fmt = str(stream_format).strip().lower().lstrip(".")
            if fmt not in ("jsonl", "json"):
                raise ValueError(f"Unsupported stream_format={stream_format!r}; expected 'jsonl' or 'json'")
            eff_append_persist = fmt == "jsonl"
        else:
            eff_append_persist = (not is_json_snapshot_venue) or eff_filename.lower().endswith(".jsonl")
        self.stream_format = "jsonl" if eff_append_persist else "json"
        eff_read_only = _parse_bool_flag(read_only, default=False)
        super().__init__(
            sync_ts,
            eff_symbols,
            eff_filename,
            append_mode=True,
            api_client=api_client,
            append_persist=eff_append_persist,
            read_only=eff_read_only,
            data_sync_mode=data_sync_mode,
        )
        if not eff_append_persist and api_client is None:
            self.api_client = None

    @staticmethod
    def _is_valid_item(trade):
        return CacheAccountManagerBase._is_valid_trade_fill_row(trade)

    @staticmethod
    def _is_valid_snapshot_item(item) -> bool:
        return isinstance(item, dict) and (
            extract_order_price(item, default=None) is not None
            or safe_pos_float(item.get("price"), default=None) is not None
        )

    def _persisted_max_ts(self) -> float:
        meta_ts = super()._persisted_max_ts()
        if self.append_persist or not os.path.exists(self.filename):
            return meta_ts
        try:
            data = read_json_dict(self.filename)
            ft = data.get("fetchtime") if isinstance(data, dict) else None
            if isinstance(ft, dict) and ft:
                vals = [
                    fv
                    for v in ft.values()
                    if (fv := safe_nonneg_float(v, default=None)) is not None
                ]
                if vals:
                    return max(meta_ts, max(vals))
        except Exception:
            pass
        return meta_ts

    def _save_state_to_file_impl(self) -> bool:
        if self.append_persist:
            return self._save_jsonl_append()
        return super()._save_state_to_file_impl()

    def _migrate_legacy_json_if_needed(self):
        """Migrate legacy ``.json`` Trade cache under the generation lock."""
        with self._generation_lock():
            return self._migrate_legacy_json_generation_locked()

    def _migrate_legacy_json_generation_locked(self):
        legacy = self._resolve_legacy_json_for_jsonl_locked()
        if not legacy:
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
                        or not os.path.exists(meta_file_path(self.filename))
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
            try:
                (
                    initial_size,
                    committed_bytes,
                    candidate_counts,
                    wrote_rows,
                    appended_records,
                ) = _cache_integrity.append_versioned_trade_jsonl(
                    self.filename,
                    self.cache,
                    previous_counts,
                    previous_version=previous_version,
                    previous_boundary=previous_boundary,
                    is_valid_item_fn=self._is_valid_item,
                    on_rollback_fn=self._truncate_failed_jsonl_append,
                )

                content_digest = _cache_integrity.trade_records_digest(
                    appended_records, previous_digest
                )
                stream_id = previous_stream or uuid.uuid4().hex
                revision = previous_revision
                if wrote_rows or not previous_version:
                    revision += 1
                revision = max(1, revision)
                version = _cache_integrity.trade_data_version(stream_id, revision)
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
        if not self.append_persist:
            self.resync_mem_file()
            return
        with self._generation_lock():
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
                        delta_counts = _cache_integrity.counts_for_records(records)
                        candidate_counts = dict(old_counts)
                        for symbol, count in delta_counts.items():
                            candidate_counts[symbol] = (
                                candidate_counts.get(symbol, 0) + count
                            )
                        if candidate_counts != metadata["counts"]:
                            raise ValueError("Trade suffix counts differ from manifest")
                        if (
                            _cache_integrity.trade_records_digest(records, old_digest)
                            != metadata["content_digest"]
                        ):
                            raise ValueError("Trade suffix digest differs from manifest")
                        candidate_cache = None
                        if not self.read_only:
                            discard_generation_file(self._previous_generation_path())
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
                            == _cache_integrity.nonzero_counts(old_counts)
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
            metaf = meta_file_path(self.filename)
            if os.path.exists(metaf):
                try:
                    legacy_metadata = read_json_dict(metaf)
                except (OSError, ValueError, TypeError):
                    legacy_metadata = {}

            with self.lock:
                self.cache = _cache_integrity.read_legacy_trade_jsonl_strict(
                    self.filename,
                    legacy_metadata,
                    is_valid_item_fn=self._is_valid_item,
                    reader_reason=self._account_reader_reason(),
                )
                self._legacy_jsonl_needs_rewrite = True
                self._persisted_counts = {s: len(v) for s, v in self.cache.items()}
                self._hydrate_fetchtimes_from_mapping_locked(
                    legacy_metadata, preserve_if_missing=True
                )

    def compact_jsonl(self):
        if not self.append_persist:
            return False
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
                    candidate_cache[symbol] = _cache_integrity.dedupe_trade_items_by_id(items)
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
                    with atomic_text_writer(self.filename) as handle:
                        for symbol, items in self.cache.items():
                            for item in items:
                                records.append((symbol, item))
                                handle.write(format_jsonl_bucket_line(symbol, item))

                def _build_trade_meta():
                    committed_bytes = os.path.getsize(self.filename)
                    content_digest = _cache_integrity.trade_records_digest(records)
                    stream_id = uuid.uuid4().hex
                    revision = 1
                    version = _cache_integrity.trade_data_version(stream_id, revision)
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
                    snapshot_fn = _cm_override(
                        "atomic_snapshot_file", atomic_snapshot_file
                    )
                    snapshot_fn(self.filename, archive)
                    original_state = self._snapshot_memory_state_locked()
                    original_dirty = self._account_cache_dirty
                    original_needs_rewrite = self._legacy_jsonl_needs_rewrite
                    self._trim_cache_to_keep_fraction_locked()
                    self._mark_account_cache_dirty_locked()
                    if not self._compact_account_trade_jsonl_generation_locked():
                        self._restore_memory_state_locked(original_state)
                        self._account_cache_dirty = original_dirty
                        self._legacy_jsonl_needs_rewrite = original_needs_rewrite
                        discard_generation_file(archive)
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
                discard_generation_file(archive)
            builtins.print(f"[{self.cls_name}][maintain] Trade rotation failed: {exc}")
            return False

    def _persist_periodic_state(self, sync_complete):
        """Persist one cycle without reversing the Trade generation lock order."""
        if self.save_state and self.append_persist:
            with self._generation_lock():
                persisted = self._save_account_trade_append_if_current_generation_locked()
                self._publish_periodic_sync_health(sync_complete, persisted)
                return persisted
        return super()._persist_periodic_state(sync_complete)

    def get_remote_items(self, symbol, startTime):
        if not self.append_persist:
            return MarketDataProvider.fetch_venue_remote_trades(
                self.venue, symbol, startTime, api_client=self.api_client
            )
        new_trades = self._paginate_validated_trades(symbol, startTime, "trade")
        print(f"[{self.cls_name}][info] New trades: {len(new_trades)}")
        return new_trades

    def _persist_items(self, symbol, new_items):
        """Append immutable fills once and reject conflicting duplicate IDs (or merge JSON snapshot fills when `append_persist=False`)."""
        if not self.append_persist:
            if not new_items:
                return
            new_items = self._coerce_item_list(new_items)
            with self.lock:
                bucket = self.cache.setdefault(symbol, [])
                by_key: dict[tuple, dict] = {}
                for item in bucket:
                    if isinstance(item, dict):
                        k = self.fill_dedup_key(item) or ("idx", len(by_key))
                        by_key[k] = item
                changed = False
                for item in new_items:
                    if not self._is_valid_snapshot_item(item):
                        continue
                    k = self.fill_dedup_key(item) or ("idx", len(by_key))
                    if by_key.get(k) != item:
                        by_key[k] = dict(item)
                        changed = True
                if changed:
                    self.cache[symbol] = sorted(
                        by_key.values(),
                        key=lambda r: (extract_timestamp_ms(r, default=0) or 0, str(r.get("id") or r.get("orderId") or "")),
                    )
                    self._advance_symbol_fetchtime_locked(symbol)
            return
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
                signature = _cache_integrity.trade_semantic_signature(item)
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
                signature = _cache_integrity.trade_semantic_signature(item)
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
        cand_clean_set = (
            {clean_symbol_key(c) for c in cache_symbol_variants(symbol, include_base=True) if c}
            if not self.append_persist
            else set()
        )
        latest_key = None
        latest_price = None
        with self.lock:
            bucket = casefold_get(self.cache, symbol) or casefold_get(self.cache, sym_clean)
            if not bucket and not self.append_persist and isinstance(self.cache, dict):
                items_map = (
                    self.cache.get("items")
                    if isinstance(self.cache.get("items"), dict)
                    else self.cache
                )
                for cand in cache_symbol_variants(symbol, include_base=True):
                    if cand and isinstance(rows := casefold_get(items_map, cand), list) and rows:
                        bucket = rows
                        break
            bucket = bucket or []
            for trade in bucket:
                if self.append_persist:
                    if not self._is_valid_item(trade):
                        continue
                    if (
                        clean_symbol_key(trade["symbol"]) != sym_clean
                        or trade["isBuyer"] != want_buyer
                    ):
                        continue
                    key = (int(trade["time"]), int(trade["id"]))
                    px_val = float(trade["price"])
                else:
                    if not self._is_valid_snapshot_item(trade):
                        continue
                    tr_sym = clean_symbol_key(trade.get("symbol") or symbol)
                    if tr_sym and sym_clean and tr_sym != sym_clean and tr_sym not in cand_clean_set:
                        continue
                    raw_ib = trade.get("isBuyer")
                    is_buyer = (
                        raw_ib
                        if isinstance(raw_ib, bool)
                        else ((extract_order_side(trade) or normalize_side(trade.get("side"))) == "BUY")
                    )
                    if is_buyer != want_buyer:
                        continue
                    ts_ms = extract_timestamp_ms(trade, default=0) or 0
                    raw_id = str(trade.get("id") or trade.get("tid") or trade.get("orderId") or trade.get("oid") or "")
                    px_val = extract_order_price(trade, default=None) or safe_pos_float(trade.get("price"), default=None)
                    if px_val is None:
                        continue
                    key = (int(ts_ms), raw_id)
                if latest_key is None or key > latest_key:
                    latest_key = key
                    latest_price = px_val
        return latest_price


class CacheOrderManager(CacheAccountManagerBase):
    """Account order aggregate cache (Level 3) backed by versioned JSON snapshots and schema-v2 manifests."""

    _is_order_account_cache = True
    DEFAULT_FILENAME = CACHE_BINANCE_ORDER_FILENAME
    VENUE_FILENAMES: dict[str, str] = {"binance": CACHE_BINANCE_ORDER_FILENAME}

    @classmethod
    def disk_filename(cls, venue: str = "binance") -> str:
        """Return the canonical on-disk Order cache filename for ``venue``."""
        v_key = normalize_venue_name(venue, default="binance") or "binance"
        return str(cls.VENUE_FILENAMES.get(v_key, cls.DEFAULT_FILENAME) or "")

    def __init__(
        self,
        sync_ts,
        symbols,
        filename,
        api_client=None,
        read_only=False,
        *,
        data_sync_mode: Optional[str] = None,
    ):
        super().__init__(
            sync_ts,
            symbols,
            filename,
            append_mode=True,
            api_client=api_client,
            append_persist=False,
            read_only=read_only,
            data_sync_mode=data_sync_mode,
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
            strict_ascii_int(trade["orderId"], min_value=1) is not None
            and safe_pos_float(trade["price"], default=None) is not None
            and safe_pos_float(trade["quantity"], default=None) is not None
            and strict_ascii_int(trade["timestamp"], min_value=1) is not None
            and str(trade["side"]).upper() in ("BUY", "SELL")
            and (
                legacy_cutoff is None
                or strict_ascii_int(legacy_cutoff, min_value=1) is not None
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

    def _load_snapshot_state(self):
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
        meta_path = meta_file_path(self.filename)
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
                    lambda meta, path=None: _cache_integrity.read_legacy_order_snapshot_strict(
                        path or self.filename, meta
                    ),
                    legacy_meta,
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
                data_version = _cache_integrity.snapshot_data_version(self.cache)
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
                write_json_fn = _cm_override("atomic_write_json", atomic_write_json)
                if not self._commit_file_and_meta_locked(
                    lambda: write_json_fn(self.filename, payload, indent=1),
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
                self._ensure_account_version(str(metadata["data_version"]))
                return
            with self._generation_lock():
                self._load_order_state_generation_locked()
        except Exception as e:
            print(f"[{self.cls_name}][Error] reload {self.filename}: {e}")

    def _read_order_snapshot_recovering_previous_locked(self, metadata):
        """Read an Order snapshot against ``metadata``, restoring ``.previous`` if needed."""
        def _read_strict(meta, *, path=None):
            return _cache_integrity.read_order_snapshot_strict(
                path or self.filename,
                meta,
                is_valid_item_fn=self._is_valid_item,
                reader_reason=self._account_reader_reason(),
            )
        return self._read_snapshot_recovering_previous_locked(_read_strict, metadata)

    def _ensure_account_version(self, required_version: str) -> None:
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
                legacy_cutoff = safe_nonneg_int(existing.get("_legacyCoveredThrough"), default=None)
                if legacy_cutoff is None and "_fillIds" not in existing:
                    legacy_cutoff = int(existing["timestamp"])
                known_fill_ids = {*known_fill_id_list}
                if fill_id and fill_id in known_fill_ids:
                    continue
                if legacy_cutoff is not None and int(incoming["timestamp"]) <= legacy_cutoff:
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
                        "price": round(aggregate_price, 8) or round(aggregate_price, 16),
                        "quantity": round(total_quantity, 8) or round(total_quantity, 16),
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


class CacheAssetValueManager(CacheAccountManagerBase):
    """Account/portfolio total asset value history cache (Level 3) backed by append-only JSONL."""

    DEFAULT_FILENAME = CACHE_ASSET_VALUE_FILENAME

    @classmethod
    def extract_asset_value_summary(cls, rec: object, *, now_ts: Optional[float] = None) -> dict:
        """Extract normalized portfolio value fields from a ``cache_asset_value.jsonl`` record."""
        if not isinstance(rec, dict) or not isinstance(rec.get("i"), dict):
            return {}
        av_inner = rec["i"]
        out: dict = {"total_value_usdc": av_inner.get("total_value_usdc")}
        if isinstance(av_inner.get("venues_usd"), dict) and av_inner["venues_usd"]:
            out["venues_value_usd"] = av_inner["venues_usd"]
        if av_inner.get("portfolio_total_usd") is not None:
            out["portfolio_total_usd"] = av_inner.get("portfolio_total_usd")
        av_ts_s = normalize_epoch_sec(
            extract_timestamp_s(rec)
            or extract_timestamp_s(av_inner)
            or rec.get("t")
            or av_inner.get("t")
            or av_inner.get("datetime_local"),
            allow_iso=True,
        )
        eff_now_s = resolve_now_ts(now_ts)
        if av_ts_s is not None and eff_now_s is not None:
            out["total_value_usdc_age_min"] = round(max(0.0, eff_now_s - av_ts_s) / 60.0, 1)
        return out

    def __init__(
        self,
        sync_ts,
        symbols,
        filename,
        api_client=None,
        read_only=False,
        *,
        data_sync_mode: Optional[str] = None,
    ):
        self._migrated_usdt_on_load = False
        super().__init__(
            sync_ts,
            symbols,
            filename,
            append_mode=True,
            api_client=api_client,
            append_persist=True,
            read_only=read_only,
            data_sync_mode=data_sync_mode,
        )
        if self._migrated_usdt_on_load and not self.read_only:
            self.compact_jsonl()

    def _is_valid_item(self, item) -> bool:
        return isinstance(item, dict) and safe_pos_float(item.get("total_value_usdc"), default=None) is not None

    def _ensure_account_version(self, required_version: str) -> None:
        self.resync_mem_file()

    def _load_jsonl(self):
        super()._load_jsonl()
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
        venues_usd: dict[str, float] = {}
        try:
            cache_root = cachedb_root_for_file(os.path.abspath(str(self.filename or "")))
            vb_data = None
            for cand in (
                os.path.join(cache_root, "state", "venue_balances_state.json"),
                os.path.join(cache_root, "venue_balances_state.json"),
            ):
                if os.path.isfile(cand):
                    vb_data = read_json_dict(cand, default=None)
                    if isinstance(vb_data, dict) and vb_data:
                        break
            if isinstance(vb_data, dict) and vb_data:
                live_cp_data = None
                cp_default_filename = CACHE_CURRENT_PRICE_FILENAME
                for cp_cand in (
                    os.path.join(cache_root, "prices", cp_default_filename),
                    os.path.join(cache_root, cp_default_filename),
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
                if other_venues:
                    cand_venues: dict[str, float] = {"binance": round(total_val, 2)}
                    for venue in sorted(other_venues):
                        if other_venues[venue] > 0:
                            cand_venues[venue] = round(other_venues[venue], 2)
                    if len(cand_venues) > 1:
                        venues_usd = cand_venues
        except Exception:
            venues_usd = {}
        if venues_usd:
            snapshot["venues_usd"] = venues_usd
            snapshot["portfolio_total_usd"] = round(sum(venues_usd.values()), 2)
        return [snapshot]


CacheAccountManagerBase.venue_order_and_trade_cache_contracts = staticmethod(
    lambda: (
        *((fname, v_key) for v_key, fname in CacheOrderManager.VENUE_FILENAMES.items()),
        *(
            (fname, v_key)
            for v_key, fname in CacheTradeManager.VENUE_FILENAMES.items()
            if not fname.lower().endswith(".jsonl")
        ),
    )
)


# ── Account reader synchronization, user-stream WS bridge & daemon lifecycle ──

_account_cache_getter = None


def bind_account_cache_getter(getter_fn) -> None:
    """Bind the canonical factory cache getter used by account reader and WS helpers."""
    global _account_cache_getter
    _account_cache_getter = getter_fn


def _resolve_account_cache_getter():
    default_getter = _account_cache_getter or _bapi_ws_bridge._default_cache_getter
    return _cm_override("get_cache_manager", default_getter)


def ensure_account_cache_readers(status):
    """Synchronize this process with both versions in a fresh health marker."""
    ach = _cm_override("account_cache_health", account_cache_health)
    getter = _resolve_account_cache_getter()
    try:
        order_version = status.order_cache_version
        trade_version = status.trade_cache_version
        if not order_version or not trade_version:
            raise ValueError("health marker has no durable cache versions")
        getter("BinanceOrder").ensure_persisted_version(order_version)
        getter("BinanceTrade").ensure_persisted_version(trade_version)
    except ach.AccountCacheNotReady:
        raise
    except (AttributeError, TypeError, ValueError) as exc:
        raise ach.AccountCacheNotReady(
            "account_cache_reader_not_current"
        ) from exc


def start_account_cache_daemon(
    cache_getter=None, sync_started_set: Optional[set] = None
) -> None:
    """Enable the account-cache health writer, seed recent sync state, and start the user-stream WS bridge."""
    ach = _cm_override("account_cache_health", account_cache_health)
    getter = cache_getter or _resolve_account_cache_getter()
    ach.enable_writer()
    now_ms = int(_now_time() * 1000)
    max_age_ms = int(ach.MAX_AGE_SEC * 1000)
    for account_cache_name in ("BinanceOrder", "BinanceTrade"):
        acc_mgr = getter(account_cache_name)
        if isinstance(sync_started_set, set):
            sync_started_set.add(account_cache_name)
        ver, sync_ms = (
            acc_mgr.get_persisted_sync_state()
            if hasattr(acc_mgr, "get_persisted_sync_state")
            else ("", 0)
        )
        if ver and sync_ms > 0 and (now_ms - sync_ms) <= max_age_ms:
            ach.record_successful_sync(
                account_cache_name, ver, now_ms=sync_ms
            )
    _bapi_ws_bridge.initialize_ws_bridge_once()
    if is_live_network_backfill_enabled() and is_auto_start_websockets_enabled():
        start_kraken_account_ws_bridge()


def disable_account_cache_daemon_writer() -> None:
    """Mark the account-cache health writer stopping before tearing down daemon threads."""
    ach = _cm_override("account_cache_health", account_cache_health)
    ach.disable_writer()


def request_stop_account_ws_streams() -> None:
    """Signal the account user-data WebSocket bridges (Binance and Kraken) to stop."""
    for bridge in (_bapi_ws_bridge._ws_bridge, _kraken_ws_bridge):
        if bridge is not None and hasattr(bridge, "request_stop"):
            try:
                bridge.request_stop()
            except Exception:
                pass


def stop_account_ws_streams(shutdown_deadline: float) -> None:
    """Join the account user-data WebSocket bridges within ``shutdown_deadline``."""
    for bridge in (_bapi_ws_bridge._ws_bridge, _kraken_ws_bridge):
        if bridge is not None and hasattr(bridge, "stop"):
            try:
                bridge.stop(
                    timeout=min(0.3, max(0.05, shutdown_deadline - _monotonic_time()))
                )
            except Exception:
                pass


def attach_hyperliquid_user_fills_bridge(
    ws_stream,
    *,
    user_address: Optional[str] = None,
    fills_cache_path: Optional[str] = None,
    resolve_env_user_address: Optional[bool] = None,
    **_unused_kwargs,
) -> None:
    """Attach the account-domain ``userFills`` WebSocket bridge onto a Hyperliquid stream instance."""
    hl_prov_cls = MarketDataProvider.resolve_venue_provider_class("hyperliquid")
    ws_stream._fills_cache_path = fills_cache_path
    ws_stream._fills_ws_seeded = False
    ws_stream._last_fills_touch_ms = 0
    ws_stream._user_address = (
        hl_prov_cls.resolve_ws_user_address(
            user_address, resolve_env=resolve_env_user_address
        )
        if hl_prov_cls is not None
        else (str(user_address or "").strip() or None)
    )

    def _resolve_fills_cache_file() -> str:
        return (
            str(ws_stream._fills_cache_path)
            if ws_stream._fills_cache_path
            else cache_path(CACHE_HYPERLIQUID_TRADE_FILENAME)
        )

    def _build_user_fills_subscribe_payloads() -> tuple[str, ...]:
        ws_stream._fills_ws_seeded = False
        if hl_prov_cls is None:
            return ()
        return hl_prov_cls.build_ws_user_fills_subscriptions(ws_stream._user_address)

    def _handle_user_fills(data: object, *, ts_ms: int) -> None:
        if hl_prov_cls is None:
            return
        with ws_stream._lock:
            symbols = ws_stream._symbols
        parsed = hl_prov_cls.extract_ws_user_fills(
            data, symbols, candidate_mids_fn=ws_stream._candidate_mids_keys
        )
        if parsed is None:
            return
        is_snapshot, fills_by_symbol = parsed
        CacheTradeManager.merge_venue_fills_snapshot(
            _resolve_fills_cache_file(),
            fills_by_symbol,
            now_ms=ts_ms,
            is_snapshot=is_snapshot,
            touch_symbols=symbols,
        )
        if is_snapshot:
            ws_stream._fills_ws_seeded = True
        ws_stream._last_fills_touch_ms = ts_ms

    def _touch_fills_cache_if_needed(ts_ms: int) -> None:
        if (
            not ws_stream._fills_ws_seeded
            or (ts_ms - ws_stream._last_fills_touch_ms) < 15_000
        ):
            return
        try:
            with ws_stream._lock:
                symbols = ws_stream._symbols
            cache_file = _resolve_fills_cache_file()
            loaded = CacheTradeManager._load_venue_snapshot_dict(cache_file)
            if loaded is not None and CacheTradeManager.touch_venue_ws_fetchtimes_in_dict(
                loaded, ts_ms, symbols=symbols, create_missing=False
            ):
                save_json_state(cache_file, loaded)
            ws_stream._last_fills_touch_ms = ts_ms
        except Exception:
            pass

    ws_stream._resolve_fills_cache_file = _resolve_fills_cache_file
    ws_stream._handle_user_fills = _handle_user_fills
    ws_stream._touch_fills_cache_if_needed = _touch_fills_cache_if_needed
    if hasattr(ws_stream, "register_channel_handler"):
        ws_stream.register_channel_handler(
            "userFills",
            _handle_user_fills,
            subscribe_builder=_build_user_fills_subscribe_payloads,
            on_tick=_touch_fills_cache_if_needed,
        )


def run_kraken_trade_poll_loop(
    client,
    *,
    cache_file: str,
    pairs: Sequence[str],
    poll_interval: float,
    backoff_init: float,
    backoff_max: float,
    lookback_sec: float,
    heartbeat_fn=None,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Run the REST Ledgers polling loop for Kraken trade cache persistence."""
    kr_prov_cls = MarketDataProvider.resolve_venue_provider_class("kraken")
    if kr_prov_cls is None:
        raise RuntimeError("Kraken provider class is not registered")

    cache = CacheTradeManager.load_sanitized_trade_snapshot(
        cache_file, venue="kraken"
    )
    init_bo = safe_pos_float(backoff_init, default=5.0) or 5.0
    max_bo = safe_pos_float(backoff_max, default=60.0) or 60.0
    poll_s = safe_nonneg_float(poll_interval, default=30.0) or 0.0
    backoff = init_bo
    if callable(heartbeat_fn):
        heartbeat_fn(force=True)
    while not (stop_event is not None and stop_event.is_set()):
        try:
            by_sym = kr_prov_cls.fetch_ledger_fills_by_symbol(
                client, pairs=pairs, lookback_sec=lookback_sec
            )
            CacheTradeManager.merge_venue_fills_into_dict(
                cache, by_sym, is_snapshot=True
            )
            save_json_state(cache_file, cache, indent=1)
            log(
                f"[kraken_cache][poll] {sum(len(v) for v in cache['items'].values())} fills "
                f"for {list(cache['items'].keys())}"
            )
            backoff = init_bo
            if callable(heartbeat_fn):
                heartbeat_fn()
            if stop_event is not None:
                if stop_event.wait(timeout=poll_s):
                    break
            else:
                _sleep_time(poll_s)
        except Exception as exc:
            log(f"[kraken_cache][poll] error (backoff {backoff:.0f}s): {exc}")
            if callable(heartbeat_fn):
                heartbeat_fn()
            if stop_event is not None:
                if stop_event.wait(timeout=max(0.0, backoff)):
                    break
            else:
                _sleep_time(backoff)
            backoff = min(backoff * 2.0, max_bo)


def run_kraken_trade_ws_loop(
    client,
    *,
    cache_file: str,
    pairs: Sequence[str],
    poll_interval: float,
    backoff_init: float,
    backoff_max: float,
    lookback_sec: float,
    ws_url: Optional[str] = None,
    heartbeat_fn=None,
    stop_event: Optional[threading.Event] = None,
    ws_holder: Optional[list] = None,
) -> None:
    """Run the real-time Kraken ``ownTrades`` WebSocket bridge with periodic REST Ledgers reconciliation."""
    kr_prov_cls = MarketDataProvider.resolve_venue_provider_class("kraken")
    if kr_prov_cls is None:
        raise RuntimeError("Kraken provider class is not registered")
    resolved_ws_url = ws_url or kr_prov_cls.PRIVATE_WS_URL

    try:
        import websocket
    except ImportError as exc:
        log(
            f"[kraken_cache][ws] websocket-client unavailable ({exc}); falling back to poll_loop"
        )
        return run_kraken_trade_poll_loop(
            client,
            cache_file=cache_file,
            pairs=pairs,
            poll_interval=poll_interval,
            backoff_init=backoff_init,
            backoff_max=backoff_max,
            lookback_sec=lookback_sec,
            heartbeat_fn=heartbeat_fn,
            stop_event=stop_event,
        )

    cache = CacheTradeManager.load_sanitized_trade_snapshot(
        cache_file, venue="kraken"
    )
    last_flush_ms = [0]
    last_rest_sync_ms = [0]
    rest_sync_interval_ms = int(
        max(30.0, safe_pos_float(poll_interval, default=30.0) or 30.0) * 1000
    )
    if callable(heartbeat_fn):
        heartbeat_fn(force=True)

    def _merge_rest() -> None:
        by_sym = kr_prov_cls.fetch_ledger_fills_by_symbol(
            client, pairs=pairs, lookback_sec=lookback_sec
        )
        CacheTradeManager.merge_venue_fills_into_dict(
            cache, by_sym, is_snapshot=False
        )

    def _touch_fetchtimes(now_ms: Optional[int] = None) -> None:
        CacheTradeManager.touch_venue_ws_fetchtimes_in_dict(
            cache, now_ms=now_ms, symbols=pairs, create_missing=True
        )

    try:
        _merge_rest()
        save_json_state(cache_file, cache, indent=1)
        now_init_ms = resolve_now_ms()
        last_flush_ms[0] = now_init_ms
        last_rest_sync_ms[0] = now_init_ms
    except Exception as exc:
        log(f"[kraken_cache][ws] REST seed failed: {exc}")

    def on_open(ws) -> None:
        try:
            ws.send(kr_prov_cls.build_ws_own_trades_subscription(client))
            log("[kraken_cache][ws] subscribed ownTrades")
        except Exception as exc:
            log(f"[kraken_cache][ws] subscribe failed: {exc}")
            try:
                ws.close()
            except Exception:
                pass

    def on_message(ws, msg) -> None:
        data = parse_json_payload(msg)
        if data is None:
            return
        now_ms = resolve_now_ms()
        fills_by_sym = kr_prov_cls.extract_ws_own_trades(data, pairs=pairs)
        if fills_by_sym is None:
            kind, err_msg = kr_prov_cls.classify_ws_control_event(data)
            if kind == "error":
                log(f"[kraken_cache][ws] ownTrades subscription error: {err_msg}")
                try:
                    ws.close()
                except Exception:
                    pass
                return
            if kind == "keepalive" and (now_ms - last_flush_ms[0]) >= 15_000:
                if (now_ms - last_rest_sync_ms[0]) >= rest_sync_interval_ms:
                    try:
                        _merge_rest()
                        last_rest_sync_ms[0] = now_ms
                        log(
                            f"[kraken_cache][ws] {sum(len(v) for v in cache['items'].values())} fills "
                            f"for {list(cache['items'].keys())}"
                        )
                    except Exception as exc:
                        log(
                            f"[kraken_cache][ws] periodic REST Ledgers sync failed: {exc}"
                        )
                if last_rest_sync_ms[0] > 0:
                    _touch_fetchtimes(now_ms)
                    save_json_state(cache_file, cache, indent=1)
                    last_flush_ms[0] = now_ms
                if callable(heartbeat_fn):
                    heartbeat_fn()
            return
        changed = CacheTradeManager.merge_venue_fills_into_dict(
            cache, fills_by_sym, now_ms=now_ms, is_snapshot=False
        )
        if changed or (now_ms - last_flush_ms[0]) >= 15_000:
            _touch_fetchtimes(now_ms)
            save_json_state(cache_file, cache, indent=1)
            last_flush_ms[0] = now_ms
            if callable(heartbeat_fn):
                heartbeat_fn()

    def on_error(_ws, err) -> None:
        log(f"[kraken_cache][ws] error: {err}")

    while not (stop_event is not None and stop_event.is_set()):
        try:
            ws = websocket.WebSocketApp(
                resolved_ws_url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
            )
            if isinstance(ws_holder, list) and ws_holder:
                ws_holder[0] = ws
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as exc:
            log(f"[kraken_cache][ws] run_forever: {exc}")
        finally:
            if isinstance(ws_holder, list) and ws_holder:
                ws_holder[0] = None
        if stop_event is not None and stop_event.is_set():
            break
        log("[kraken_cache][ws] disconnected; reconnect in 5s")
        if callable(heartbeat_fn):
            heartbeat_fn()
        if stop_event is not None:
            if stop_event.wait(timeout=5.0):
                break
        else:
            _sleep_time(5)
        try:
            _merge_rest()
            save_json_state(cache_file, cache, indent=1)
            now_resync_ms = resolve_now_ms()
            last_flush_ms[0] = now_resync_ms
            last_rest_sync_ms[0] = now_resync_ms
        except Exception as exc:
            log(f"[kraken_cache][ws] REST resync after disconnect failed: {exc}")


class _KrakenAccountTradeBridge:
    """Background daemon bridge managing Kraken ``ownTrades`` WS and ``Ledgers`` REST sync inside cacheManager."""

    def __init__(self) -> None:
        self._stop_event = threading.Event()
        self._ws_holder: list[Any] = [None]
        self._thread: Optional[threading.Thread] = None

    @property
    def is_running(self) -> bool:
        return (
            self._thread is not None
            and self._thread.is_alive()
            and not self._stop_event.is_set()
        )

    def start(self) -> bool:
        from credentials import CredentialConfigurationError

        kr_prov_cls = MarketDataProvider.resolve_venue_provider_class("kraken")
        if kr_prov_cls is None:
            log("[kraken_cache] skipped (Kraken provider not registered)")
            return False
        try:
            cfg = kr_prov_cls.resolve_cache_daemon_config()
            client, profile = kr_prov_cls.build_cache_client()
        except CredentialConfigurationError as exc:
            log(f"[kraken_cache] skipped (no Kraken cache credentials): {exc}")
            return False
        except Exception as exc:
            log(f"[kraken_cache] failed to initialize daemon config: {exc}")
            return False

        self._stop_event.clear()
        mode = cfg["mode"]
        log(
            f"[kraken_cache] start: mode={mode} credentials={profile} "
            f"pairs={cfg['pairs']} poll={cfg['poll_interval']}s -> {cfg['cache_file']}"
        )

        def _runner() -> None:
            if mode in ("ws", "auto"):
                run_kraken_trade_ws_loop(
                    client,
                    cache_file=cfg["cache_file"],
                    pairs=cfg["pairs"],
                    poll_interval=cfg["poll_interval"],
                    backoff_init=cfg["backoff_init"],
                    backoff_max=cfg["backoff_max"],
                    lookback_sec=cfg["lookback_sec"],
                    ws_url=cfg["ws_url"],
                    stop_event=self._stop_event,
                    ws_holder=self._ws_holder,
                )
            else:
                run_kraken_trade_poll_loop(
                    client,
                    cache_file=cfg["cache_file"],
                    pairs=cfg["pairs"],
                    poll_interval=cfg["poll_interval"],
                    backoff_init=cfg["backoff_init"],
                    backoff_max=cfg["backoff_max"],
                    lookback_sec=cfg["lookback_sec"],
                    stop_event=self._stop_event,
                )

        t = threading.Thread(
            target=_runner,
            name="KrakenAccountTradeBridge",
            daemon=True,
        )
        self._thread = t
        t.start()
        return True

    def request_stop(self) -> None:
        self._stop_event.set()
        ws = self._ws_holder[0]
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

    def stop(self, timeout: float = 0.3) -> None:
        self.request_stop()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=max(0.05, safe_pos_float(timeout, default=0.3) or 0.3))


_kraken_ws_bridge: Optional[_KrakenAccountTradeBridge] = None
_kraken_ws_bridge_lock = threading.Lock()


def start_kraken_account_ws_bridge() -> Optional[_KrakenAccountTradeBridge]:
    """Idempotently start the Kraken ``ownTrades``/``Ledgers`` background bridge inside cacheManager."""
    global _kraken_ws_bridge
    with _kraken_ws_bridge_lock:
        if _kraken_ws_bridge is not None and _kraken_ws_bridge.is_running:
            return _kraken_ws_bridge
        bridge = _KrakenAccountTradeBridge()
        if bridge.start():
            _kraken_ws_bridge = bridge
            return bridge
        return None


ORDER_SYNC_INTERVAL_SEC = 0.4 * 60
TRADE_SYNC_INTERVAL_SEC = 3 * 60
ASSET_VALUE_SYNC_INTERVAL_SEC = 10 * 60
_READONLY_ACCOUNT_LOCK = threading.RLock()
_READONLY_ACCOUNT_READERS: dict[str, Any] = {}
_VENUE_TRADE_CACHE_NAMES = {
    "binance": "BinanceTrade",
    "kraken": "KrakenTrade",
    "hyperliquid": "HyperliquidTrade",
    "t212": "T212Trade",
}


def get_asset_value_manager(
    symbols=None,
    filename=CacheAssetValueManager.DEFAULT_FILENAME,
    *,
    data_sync_mode: Optional[str] = CacheManagerInterface.MODE_DISK_READER,
) -> CacheAssetValueManager:
    """Return the ``CacheAssetValueManager`` singleton without importing ``cacheManager`` in read-only consumers."""
    cm_mod = sys.modules.get("cacheManager")
    if cm_mod is not None:
        cm_fn = getattr(cm_mod, "get_asset_value_manager", None)
        if callable(cm_fn) and getattr(cm_fn, "_is_ssot_asset_value_getter", None) is not True:
            if (
                symbols is None
                and filename == CacheAssetValueManager.DEFAULT_FILENAME
                and data_sync_mode == CacheManagerInterface.MODE_DISK_READER
            ):
                return cm_fn()
            return cm_fn(symbols=symbols, filename=filename, data_sync_mode=data_sync_mode)
        cm_gcm = getattr(cm_mod, "get_cache_manager", None)
        if callable(cm_gcm) and getattr(cm_gcm, "_is_ssot_cache_manager_getter", None) is not True:
            if symbols is None and data_sync_mode == CacheManagerInterface.MODE_DISK_READER:
                return cm_gcm("AssetValue")
            return cm_gcm("AssetValue", symbols, data_sync_mode=data_sync_mode)
        factory = getattr(cm_mod, "CacheFactory", None)
        if factory is not None:
            return factory.get(
                "AssetValue",
                symbols=symbols,
                data_sync_mode=data_sync_mode,
                filename=filename,
            )

    resolved_mode, save_state_flag, start_periodic = CacheManagerInterface.resolve_data_sync_mode(data_sync_mode)
    eff_symbols = (
        ["TOTAL"]
        if symbols is None
        else (parse_symbol_list(symbols) if isinstance(symbols, str) else list(symbols))
    )
    with _READONLY_ACCOUNT_LOCK:
        inst = _READONLY_ACCOUNT_READERS.get("AssetValue")
        if (
            inst is not None
            and eff_symbols == list(getattr(inst, "symbols", []))
            and filename == CacheAssetValueManager.DEFAULT_FILENAME
            and not save_state_flag
            and not start_periodic
        ):
            if getattr(inst, "read_only", False) and hasattr(inst, "resync_mem_file"):
                try:
                    inst.resync_mem_file()
                except Exception:
                    pass
            return inst
        inst = CacheAssetValueManager(
            sync_ts=ASSET_VALUE_SYNC_INTERVAL_SEC,
            filename=filename,
            symbols=eff_symbols,
            read_only=(not save_state_flag and not start_periodic),
            data_sync_mode=resolved_mode,
        )
        if filename == CacheAssetValueManager.DEFAULT_FILENAME:
            _READONLY_ACCOUNT_READERS["AssetValue"] = inst
        return inst


get_asset_value_manager._is_ssot_asset_value_getter = True  # type: ignore[attr-defined]


def get_binance_order_manager(
    symbols=None,
    filename=CacheOrderManager.DEFAULT_FILENAME,
    *,
    data_sync_mode: Optional[str] = CacheManagerInterface.MODE_DISK_READER,
) -> CacheOrderManager:
    """Return the ``CacheOrderManager`` singleton without importing ``cacheManager`` in read-only consumers."""
    cm_mod = sys.modules.get("cacheManager")
    if cm_mod is not None:
        cm_fn = getattr(cm_mod, "get_binance_order_manager", None)
        if callable(cm_fn) and getattr(cm_fn, "_is_ssot_binance_order_getter", None) is not True:
            if (
                symbols is None
                and filename == CacheOrderManager.DEFAULT_FILENAME
                and data_sync_mode == CacheManagerInterface.MODE_DISK_READER
            ):
                return cm_fn()
            return cm_fn(symbols=symbols, filename=filename, data_sync_mode=data_sync_mode)
        cm_gcm = getattr(cm_mod, "get_cache_manager", None)
        if callable(cm_gcm) and getattr(cm_gcm, "_is_ssot_cache_manager_getter", None) is not True:
            if symbols is None and data_sync_mode == CacheManagerInterface.MODE_DISK_READER:
                return cm_gcm("BinanceOrder")
            return cm_gcm("BinanceOrder", symbols, data_sync_mode=data_sync_mode)
        factory = getattr(cm_mod, "CacheFactory", None)
        if factory is not None:
            return factory.get(
                "BinanceOrder",
                symbols=symbols,
                data_sync_mode=data_sync_mode,
                filename=filename,
            )

    resolved_mode, save_state_flag, start_periodic = CacheManagerInterface.resolve_data_sync_mode(data_sync_mode)
    eff_symbols = (
        list(sym.symbols)
        if symbols is None
        else (parse_symbol_list(symbols) if isinstance(symbols, str) else list(symbols))
    )
    with _READONLY_ACCOUNT_LOCK:
        inst = _READONLY_ACCOUNT_READERS.get("BinanceOrder")
        if (
            inst is not None
            and symbols is None
            and filename == CacheOrderManager.DEFAULT_FILENAME
            and not save_state_flag
            and not start_periodic
        ):
            return inst
        inst = CacheOrderManager(
            sync_ts=ORDER_SYNC_INTERVAL_SEC,
            filename=filename,
            symbols=eff_symbols,
            read_only=(not save_state_flag and not start_periodic),
            data_sync_mode=resolved_mode,
        )
        if symbols is None and filename == CacheOrderManager.DEFAULT_FILENAME:
            _READONLY_ACCOUNT_READERS["BinanceOrder"] = inst
        return inst


get_binance_order_manager._is_ssot_binance_order_getter = True  # type: ignore[attr-defined]


def get_trade_manager(
    venue: str = "binance",
    symbols=None,
    *,
    data_sync_mode: Optional[str] = CacheManagerInterface.MODE_DISK_READER,
) -> CacheTradeManager:
    """Return the venue ``CacheTradeManager`` singleton without importing ``cacheManager`` in read-only consumers."""
    v_norm = normalize_venue_name(venue) or "binance"
    cache_name = _VENUE_TRADE_CACHE_NAMES.get(v_norm)
    if cache_name is None:
        raise ValueError(f"Unsupported trade cache venue: {venue!r}")
    cm_mod = sys.modules.get("cacheManager")
    if cm_mod is not None:
        cm_gcm = getattr(cm_mod, "get_cache_manager", None)
        if callable(cm_gcm) and getattr(cm_gcm, "_is_ssot_cache_manager_getter", None) is not True:
            if symbols is None and data_sync_mode == CacheManagerInterface.MODE_DISK_READER:
                return cm_gcm(cache_name)
            return cm_gcm(cache_name, symbols, data_sync_mode=data_sync_mode)
        factory = getattr(cm_mod, "CacheFactory", None)
        if factory is not None:
            return factory.get(cache_name, symbols=symbols, data_sync_mode=data_sync_mode)

    is_non_binance = not is_primary_stream_venue(v_norm)
    resolved_mode, save_state_flag, start_periodic = CacheManagerInterface.resolve_data_sync_mode(
        data_sync_mode,
        config_read_only=is_non_binance,
    )
    eff_symbols = (
        (None if is_non_binance else list(sym.symbols))
        if symbols is None
        else (parse_symbol_list(symbols) if isinstance(symbols, str) else list(symbols))
    )
    with _READONLY_ACCOUNT_LOCK:
        inst = _READONLY_ACCOUNT_READERS.get(cache_name)
        if inst is not None and symbols is None and not save_state_flag and not start_periodic:
            return inst
        inst = CacheTradeManager(
            sync_ts=TRADE_SYNC_INTERVAL_SEC,
            filename=CacheTradeManager.disk_filename(v_norm),
            symbols=eff_symbols,
            read_only=(is_non_binance or (not save_state_flag and not start_periodic)),
            data_sync_mode=resolved_mode,
            venue=v_norm,
            stream_format="json" if is_non_binance else "jsonl",
        )
        if symbols is None:
            _READONLY_ACCOUNT_READERS[cache_name] = inst
        return inst


get_trade_manager._is_ssot_trade_getter = True  # type: ignore[attr-defined]



