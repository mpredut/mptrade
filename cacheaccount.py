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
from typing import Optional
import uuid

from binance_api import bapi as api
from binance_api import bapi_allorders as apiorders
from binance_api import bapi_ws_bridge as _bapi_ws_bridge
import binance_cache_health as account_cache_health
from botcore import _parse_bool_flag, normalize_epoch_sec, resolve_now_ts, safe_pos_float
from cacheInterface import (
    CacheManagerInterface,
    _CACHE_META_SCHEMA_VERSION,
    _cm_override,
    _jsonl_line,
    _now_time,
    _strict_int_field,
)
import cacheintegrity as _cache_integrity
from cachepaths import (
    CACHE_ASSET_VALUE_FILENAME,
    CACHE_HYPERLIQUID_TRADE_FILENAME,
    CACHE_KRAKEN_TRADE_FILENAME,
    CACHE_ORDER_FILENAME,
    CACHE_T212_TRADE_FILENAME,
    CACHE_TRADE_FILENAME,
    cache_path,
    cache_symbol_variants,
    cachedb_root_for_file,
    clean_symbol_key,
    compute_venue_nav_breakdown,
    normalize_venue_name,
    resolve_cachedb_filepath,
)
import instrument_registry as sym
from instrument_registry import casefold_get
from providers.base import (
    extract_order_price,
    extract_timestamp_ms,
    extract_timestamp_s,
    normalize_side,
)
from providers.order_ohlc_utils import (
    clean_order_id,
    extract_order_qty,
    extract_order_side,
)
from state_io import (
    atomic_snapshot_file,
    atomic_text_writer,
    atomic_write_json,
    discard_generation_file,
    meta_file_path,
    read_json_dict,
    read_recovering_previous,
    stage_previous_generation,
)


# WS-only mode and health state live in ``binance_api.bapi_ws_bridge``.
WS_ONLY_MODE = _bapi_ws_bridge.WS_ONLY_MODE
WS_LOSS_TIMEOUT_SEC = _bapi_ws_bridge.WS_LOSS_TIMEOUT_SEC
WS_EVENT_LOG_ENABLED = _bapi_ws_bridge.WS_EVENT_LOG_ENABLED

_ws_health_lock = _bapi_ws_bridge._ws_health_lock
_ws_available = _bapi_ws_bridge._ws_available
_ws_last_event_ts = _bapi_ws_bridge._ws_last_event_ts
_ws_is_healthy = _bapi_ws_bridge._ws_is_healthy


_mark_ws_available = _bapi_ws_bridge.mark_ws_available
_mark_ws_event_received = _bapi_ws_bridge.mark_ws_event_received
_mark_ws_unhealthy = _bapi_ws_bridge.mark_ws_unhealthy


def _should_poll_for_manager(cls_name):
    return _bapi_ws_bridge.should_poll_for_manager(
        cls_name,
        ws_only_mode=_cm_override("WS_ONLY_MODE", WS_ONLY_MODE),
        ws_available=_cm_override("_ws_available", _ws_available),
        ws_is_healthy=_cm_override("_ws_is_healthy", _ws_is_healthy),
    )


CacheManagerInterface.bind_runtime_services(
    api_client=api,
    poll_predicate=_should_poll_for_manager,
    ws_resync_clear_hook=_bapi_ws_bridge.clear_ws_rest_resync_for_manager,
)


class CacheAccountManagerBase(CacheManagerInterface, ABC):
    """Abstract base contract (Level 2) for account-domain caches (``Order``, ``Trade``, ``AssetValue``)."""

    _is_trade_account_cache = False
    _is_order_account_cache = False

    def _is_account_cache(self):
        return bool(self._is_trade_account_cache or self._is_order_account_cache)

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
                    int(v)
                    for v in cursors.values()
                    if isinstance(v, (int, float)) and v > 0
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
                cursor_ms = max(0, request_high_water - 1)
                self._rest_sync_cursor_per_symbol[symbol] = cursor_ms
                self.fetchtime_per_symbol[symbol] = max(
                    int(self.fetchtime_per_symbol.get(symbol, 0) or 0),
                    cursor_ms,
                )
                self._mark_account_cache_dirty_locked()

    def _on_complete_remote_sync_cycle(
        self, account_cycle_high_water: Optional[int]
    ) -> None:
        if self._is_account_cache():
            with self.lock:
                self._last_complete_sync_at_ms = account_cycle_high_water or 0

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

    _snapshot_data_version = staticmethod(_cache_integrity.snapshot_data_version)
    _trade_data_version = staticmethod(_cache_integrity.trade_data_version)
    _trade_records_digest = staticmethod(_cache_integrity.trade_records_digest)
    _trade_semantic_signature = staticmethod(_cache_integrity.trade_semantic_signature)
    _dedupe_trade_items_by_id = staticmethod(_cache_integrity.dedupe_trade_items_by_id)
    _counts_for_records = staticmethod(_cache_integrity.counts_for_records)
    _nonzero_counts = staticmethod(_cache_integrity.nonzero_counts)

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
        self._persisted_revision = int(metadata.get("revision") or 0)
        self._persisted_committed_bytes = int(metadata.get("committed_bytes") or 0)
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
            and _strict_int_field(trade["id"], min_value=0) is not None
            and _strict_int_field(trade["orderId"], min_value=1) is not None
            and safe_pos_float(trade["price"], default=None) is not None
            and safe_pos_float(trade["qty"], default=None) is not None
            and _strict_int_field(trade["time"], min_value=1) is not None
            and isinstance(trade["isBuyer"], bool)
        )

    def _paginate_validated_trades(self, symbol, startTime, row_label: str = "trade"):
        """Paginate through ``get_my_trades`` and validate every returned fill row."""
        rows = apiorders.paginate_my_trades(
            self.api_client.client, symbol, startTime, limit=1000, strict=True
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
    DEFAULT_FILENAME = CACHE_TRADE_FILENAME
    VENUE_FILENAMES: dict[str, str] = {
        "binance": CACHE_TRADE_FILENAME,
        "kraken": CACHE_KRAKEN_TRADE_FILENAME,
        "hyperliquid": CACHE_HYPERLIQUID_TRADE_FILENAME,
        "hl": CACHE_HYPERLIQUID_TRADE_FILENAME,
        "t212": CACHE_T212_TRADE_FILENAME,
        "trading212": CACHE_T212_TRADE_FILENAME,
    }

    @classmethod
    def _normalize_trade_venue(
        cls,
        venue: Optional[str] = None,
        filename: Optional[str] = None,
        symbols=None,
    ) -> str:
        v_norm = normalize_venue_name(venue)
        if v_norm in ("binance", "kraken", "hyperliquid", "t212"):
            return v_norm
        base_f = os.path.basename(str(filename or "")).lower()
        for cand_v, cand_f in (
            ("kraken", CACHE_KRAKEN_TRADE_FILENAME),
            ("hyperliquid", CACHE_HYPERLIQUID_TRADE_FILENAME),
            ("t212", CACHE_T212_TRADE_FILENAME),
        ):
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
            from instrument_registry import venue_for_symbol as _reg_venue_for_symbol

            return _reg_venue_for_symbol(symbol, default=default)
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
                    valid_rows.append(row_copy)
                valid_rows.sort(key=lambda t: t["time"])
                clean_items[str(k)] = valid_rows
            clean_fetchtime = {}
            for k, v in loaded["fetchtime"].items():
                if not isinstance(v, bool) and isinstance(v, (int, float)) and v >= 0:
                    clean_fetchtime[str(k)] = float(v)
            return {
                "items": clean_items,
                "fetchtime": clean_fetchtime,
            }
        return {"items": {}, "fetchtime": {}}

    @classmethod
    def extract_venue_symbol_trades(
        cls,
        payload: object,
        symbol: str,
        *,
        cand_keys=None,
    ) -> list:
        """Extract the list of trade rows for ``symbol`` (or ``cand_keys``) from a venue trade snapshot."""
        if not isinstance(payload, dict):
            return []
        items = payload.get("items") if isinstance(payload.get("items"), dict) else payload
        if not isinstance(items, dict):
            return []
        keys = tuple(cand_keys) if cand_keys is not None else cache_symbol_variants(symbol, include_base=True)
        for cand in keys:
            if not cand:
                continue
            rows = casefold_get(items, cand)
            if isinstance(rows, list) and rows:
                return rows
        return []

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
        start_sync: Optional[bool] = None,
    ):
        self.venue = self._normalize_trade_venue(venue, filename, symbols)
        eff_filename = str(filename) if filename else self.resolve_cache_filepath(venue=self.venue)
        if symbols is None and self.venue != "binance":
            try:
                from instrument_registry import symbols_for as _reg_symbols_for

                eff_symbols = list(_reg_symbols_for(self.venue))
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
            eff_append_persist = self.venue == "binance" or eff_filename.lower().endswith(".jsonl")
        self.stream_format = "jsonl" if eff_append_persist else "json"
        eff_read_only = (
            (not _parse_bool_flag(start_sync, default=False))
            if start_sync is not None
            else _parse_bool_flag(read_only, default=False)
        )
        if eff_append_persist and api_client is None:
            api_client = _cm_override("api", api)
        super().__init__(
            sync_ts,
            eff_symbols,
            eff_filename,
            append_mode=True,
            api_client=api_client,
            append_persist=eff_append_persist,
            read_only=eff_read_only,
        )

    @staticmethod
    def _is_valid_item(trade):
        return CacheAccountManagerBase._is_valid_trade_fill_row(trade)

    @staticmethod
    def _is_valid_snapshot_item(item) -> bool:
        return isinstance(item, dict) and (
            extract_order_price(item, default=None) is not None
            or safe_pos_float(item.get("price"), default=None) is not None
        )

    def _save_state_to_file_impl(self) -> bool:
        if self.append_persist:
            return self._save_jsonl_append()
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
        if not self.append_persist:
            self.resync_mem_file()
            return
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
                    with atomic_text_writer(self.filename) as handle:
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
        if self.save_state:
            with self._generation_lock():
                persisted = self._save_account_trade_append_if_current_generation_locked()
                self._publish_periodic_sync_health(sync_complete, persisted)
                return persisted
        return super()._persist_periodic_state(sync_complete)

    def get_remote_items(self, symbol, startTime):
        if not self.append_persist:
            return []
        new_trades = self._paginate_validated_trades(symbol, startTime, "trade")
        print(f"[{self.cls_name}][info] New trades: {len(new_trades)}")
        return new_trades

    def update_cache_per_symbol(self, symbol, new_items):
        """Route direct update_cache_per_symbol calls through full-history ID deduplication."""
        if new_items is None:
            return
        self._persist_items(symbol, self._coerce_item_list(new_items))

    def _persist_snapshot_items(self, symbol, new_items):
        """Merge and deduplicate JSON snapshot trade items (`append_persist=False`) by ``id``/``orderId``."""
        if not new_items:
            return
        new_items = self._coerce_item_list(new_items)
        with self.lock:
            bucket = self.cache.setdefault(symbol, [])
            by_key: dict[str, dict] = {}
            for item in bucket:
                if isinstance(item, dict):
                    k = str(item.get("id") or item.get("orderId") or len(by_key))
                    by_key[k] = item
            changed = False
            for item in new_items:
                if not self._is_valid_snapshot_item(item):
                    continue
                k = str(item.get("id") or item.get("orderId") or len(by_key))
                if by_key.get(k) != item:
                    by_key[k] = dict(item)
                    changed = True
            if changed:
                merged = sorted(
                    by_key.values(),
                    key=lambda r: (extract_timestamp_ms(r, default=0) or 0, str(r.get("id") or r.get("orderId") or "")),
                )
                self.cache[symbol] = merged
                self._advance_symbol_fetchtime_locked(symbol)

    def _persist_items(self, symbol, new_items):
        """Append immutable fills once and reject conflicting duplicate IDs (or merge JSON snapshot fills when `append_persist=False`)."""
        if not self.append_persist:
            self._persist_snapshot_items(symbol, new_items)
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
        if self.append_persist:
            self.sync_account_reader_if_needed()
        elif not self.cache:
            try:
                self.resync_mem_file()
            except Exception:
                pass
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
            bucket = (
                casefold_get(self.cache, symbol)
                or casefold_get(self.cache, sym_clean)
                or (self.extract_venue_symbol_trades(self.cache, symbol) if not self.append_persist else None)
                or []
            )
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
    DEFAULT_FILENAME = CACHE_ORDER_FILENAME

    def __init__(self, sync_ts, symbols, filename, api_client=None, read_only=False):
        if api_client is None:
            api_client = _cm_override("api", api)
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

    def __init__(self, sync_ts, symbols, filename, api_client=None, read_only=False):
        if api_client is None:
            api_client = _cm_override("api", api)
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

    def _resolve_venue_balances_usd(self, binance_total_val: float) -> dict[str, float]:
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
            if not isinstance(vb_data, dict) or not vb_data:
                return {}
            live_cp_data = None
            cp_default_filename = "cache_current_price.json"
            try:
                from cacheprice import CacheCurrentPriceManager as _CCPM

                cp_default_filename = getattr(_CCPM, "DEFAULT_FILENAME", cp_default_filename) or cp_default_filename
            except Exception:
                pass
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


def _venue_trade_cache_contracts() -> tuple[tuple[str, str], ...]:
    return (
        (CacheTradeManager.disk_filename("kraken"), "kraken"),
        (CacheTradeManager.disk_filename("hyperliquid"), "hyperliquid"),
        (CacheTradeManager.disk_filename("t212"), "t212"),
    )


def _venue_order_and_trade_cache_contracts() -> tuple[tuple[str, str], ...]:
    return (
        (CacheOrderManager.DEFAULT_FILENAME, "binance"),
        *_venue_trade_cache_contracts(),
    )


CacheAccountManagerBase.venue_trade_cache_contracts = staticmethod(_venue_trade_cache_contracts)
CacheAccountManagerBase.venue_order_and_trade_cache_contracts = staticmethod(
    _venue_order_and_trade_cache_contracts
)

