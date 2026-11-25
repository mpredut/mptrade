"""Account domain cache hierarchy, multi-venue trade/order bridges, and SSOT reader singletons.

Thematic layout (Sections 1..7):
  1. Level 2 Abstract Base: ``CacheAccountManagerBase(CacheManagerInterface, ABC)``
  2. Trade Fill Cache: ``CacheTradeManager(CacheAccountManagerBase)``
  3. Order Snapshot Caches: ``CacheOrderFilledManager`` & ``CacheOrderOpenManager``
  4. Portfolio NAV Archive: ``CacheAssetValueManager(CacheAccountManagerBase)``
  5. Account Reader Synchronization & Daemon Lifecycle (`ensure_account_cache_readers`, `start_account_cache_daemon`)
  6. Venue Account Bridges (`_AccountTradeBridgeBase`, `_HyperliquidAccountTradeBridge`, `_KrakenAccountTradeBridge`, `_T212AccountTradeBridge`, `_VenueOpenOrdersBridge`)
  7. SSOT Singleton Factory Getters (`get_asset_value_manager`, `get_order_filled_manager`, `get_order_open_manager`, `get_trade_manager`)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import builtins
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
import json
import math
import os
import threading
from typing import Any, Callable, Optional, Sequence
import uuid

import account_cache_health
from credentials import CredentialConfigurationError
from botcore import (
    _is_fresh_obs_ts,
    _parse_bool_flag,
    _ts_ms_to_s,
    clean_order_id,
    clean_upper_scalar,
    extract_order_id,
    extract_order_price,
    extract_order_qty,
    extract_order_side,
    extract_timestamp_ms,
    extract_timestamp_s,
    first_epoch_sec,
    first_pos_float,
    is_auto_start_websockets_enabled,
    is_live_network_backfill_enabled,
    log,
    normalize_epoch_sec,
    normalize_side,
    resolve_now_ms,
    resolve_now_ts,
    resolve_positive_max_age,
    safe_float,
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
    CACHE_BINANCE_BALANCE_FILENAME,
    CACHE_BINANCE_ORDER_FILLED_FILENAME,
    CACHE_BINANCE_ORDER_OPEN_FILENAME,
    CACHE_BINANCE_TRADE_FILENAME,
    CACHE_CURRENT_PRICE_FILENAME,
    CACHE_HYPERLIQUID_BALANCE_FILENAME,
    CACHE_HYPERLIQUID_ORDER_FILLED_FILENAME,
    CACHE_HYPERLIQUID_ORDER_OPEN_FILENAME,
    CACHE_HYPERLIQUID_TRADE_FILENAME,
    CACHE_KRAKEN_BALANCE_FILENAME,
    CACHE_KRAKEN_ORDER_FILLED_FILENAME,
    CACHE_KRAKEN_ORDER_OPEN_FILENAME,
    CACHE_KRAKEN_TRADE_FILENAME,
    CACHE_T212_BALANCE_FILENAME,
    CACHE_T212_ORDER_FILLED_FILENAME,
    CACHE_T212_ORDER_OPEN_FILENAME,
    CACHE_T212_TRADE_FILENAME,
    VENUE_BALANCES_STATE_FILENAME,
    cache_path,
    cache_symbol_variants,
    cachedb_root_for_file,
    clean_scalar_str,
    clean_symbol_key,
    is_equity_venue,
    is_primary_stream_venue,
    legacy_filenames_for,
    normalize_venue_name,
    resolve_cachedb_filepath,
    split_symbol_assets,
)
import instrument_registry as sym
from instrument_registry import canonical_balance_asset_key, casefold_get, is_symbol_market_open
from providers.base import MarketDataProvider
from providers.order_id_context import extract_client_order_id
from providers.order_ledger import (
    OrderLedgerState,
    TERMINAL_LEDGER_STATES,
    classify_fill_evidence_state,
)
from providers.order_ohlc_utils import (
    extract_executed_qty,
    extract_order_fee,
    extract_orig_qty,
)
from providers.order_status_ssot import (
    TERMINAL_AUDIT_VENUE_STATUSES,
    TERMINAL_CANCEL_VENUE_STATUSES,
    TERMINAL_FILL_ROW_STATUSES,
    is_open_working_status,
    merge_monotonic_fill_row_status,
)
from providers.portfolio_valuation import (
    compute_venue_nav_breakdown,
    load_venue_balance_caches,
    load_venue_balances_state_dict,
)
from state_io import (
    atomic_text_writer,
    atomic_write_json,
    discard_generation_file,
    generation_transaction,
    meta_file_path,
    parse_json_payload,
    read_json_dict,
    read_json_dict_cached,
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


# =============================================================================
# 1. LEVEL 2 ABSTRACT BASE: CacheAccountManagerBase
# =============================================================================


class CacheAccountManagerBase(CacheManagerInterface, ABC):
    """Abstract base contract (Level 2) for account-domain caches (``OrderFilled``, ``OrderOpen``, ``Trade``, ``AssetValue``)."""

    _default_api_client = api
    _is_trade_account_cache = False
    _is_order_account_cache = False
    _is_open_order_account_cache = False
    VENUE_FILENAMES: dict[str, str] = {}

    @classmethod
    def disk_filename(cls, venue: str = "binance") -> str:
        """Return the canonical on-disk filename for ``venue``."""
        v_key = normalize_venue_name(venue, default="binance") or "binance"
        return str(cls.VENUE_FILENAMES.get(v_key, cls.DEFAULT_FILENAME) or "")

    @staticmethod
    def venue_for_symbol(symbol: object, *, default: str = "binance") -> str:
        """Resolve the configured venue ('binance', 'kraken', 'hyperliquid', 't212') for ``symbol`` via ``instrument_registry``."""
        try:
            return sym.venue_for_symbol(symbol, default=default)
        except Exception:
            return normalize_venue_name(default, default="binance") or "binance"

    @classmethod
    def _normalize_venue_arg(
        cls,
        venue: Optional[str] = None,
        filename: Optional[str] = None,
        symbols=None,
    ) -> str:
        v_norm = normalize_venue_name(venue)
        if v_norm in cls.VENUE_FILENAMES:
            return v_norm
        clean_fname = clean_scalar_str(os.fspath(filename)) if isinstance(filename, (str, os.PathLike)) else None
        base_f = os.path.basename(clean_fname or "").lower()
        for cand_v, cand_f in cls.VENUE_FILENAMES.items():
            if base_f == cand_f.lower() or f"_{cand_v}." in base_f:
                return cand_v
        if not base_f and symbols:
            first_sym = (
                symbols
                if isinstance(symbols, str)
                else (next(iter(symbols), None) if isinstance(symbols, (list, tuple, set)) else None)
            )
            if clean_scalar_str(first_sym) is not None:
                return cls.venue_for_symbol(first_sym)
        return "binance"
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
        """Return ``(data_version, min_cursor_ms)`` across active configured symbols from loaded/persisted state."""
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
        if not isinstance(cursors, dict) or not cursors:
            return version, 0
        active_syms = [
            s_str
            for s in (getattr(self, "symbols", None) or ())
            if (s_str := clean_scalar_str(s)) is not None
        ]
        if active_syms and (
            self._is_canonical_account_cache()
            or any(casefold_get(cursors, s) is not None for s in active_syms)
        ):
            vals = [safe_pos_int(casefold_get(cursors, s), default=None) for s in active_syms]
            if any(v is None for v in vals):
                return version, 0
            return version, min(v for v in vals if v is not None)
        sync_ms = min(
            (
                vi
                for v in cursors.values()
                if (vi := safe_pos_int(v, default=None)) is not None
            ),
            default=0,
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
        strict_int = bool(getattr(self, "_requires_int_order_id", lambda: True)())
        _cache_integrity.validate_account_cache_state(
            cls_name=self.cls_name,
            is_trade_cache=bool(self._is_trade_account_cache),
            is_order_cache=bool(self._is_order_account_cache),
            cache=self.cache,
            fetchtime_per_symbol=self.fetchtime_per_symbol,
            rest_sync_cursor_per_symbol=self._rest_sync_cursor_per_symbol,
            is_valid_item_fn=self._is_valid_item,
            strict_int_order_id=strict_int,
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
            repair_files=not self.read_only,
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
                else:
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

    def _is_file_newer_than_memory_locked(self) -> bool:
        if self._persisted_max_ts() > self._mem_max_ts():
            builtins.print(
                f"[{self.cls_name}][resync] file newer than memory -> "
                f"refusing to overwrite with stale data ({self.filename})"
            )
            return True
        return False

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

    def _log_account_save_error(self, op_label: str, exc: Exception) -> bool:
        print(f"[{self.cls_name}][Error] {op_label}: {exc}")
        return False


# =============================================================================
# 2. TRADE FILL CACHE: CacheTradeManager
# =============================================================================


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
        clean_fname = clean_scalar_str(os.fspath(filename)) if isinstance(filename, (str, os.PathLike)) else None
        base_f = os.path.basename(clean_fname or "").lower()
        for cand_v, cand_f in cls.VENUE_FILENAMES.items():
            if base_f == cand_f.lower() or f"_{cand_v}." in base_f:
                return cand_v
        if not base_f and symbols:
            first_sym = (
                symbols
                if isinstance(symbols, str)
                else (next(iter(symbols), None) if isinstance(symbols, (list, tuple, set)) else None)
            )
            if clean_scalar_str(first_sym) is not None:
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
    def resolve_cache_filepath(cls, cache_dir: Optional[str] = None, *, venue: str = "binance") -> str:
        """Resolve the canonical on-disk path for ``venue``'s trade cache file."""
        fname = cls.disk_filename(venue)
        clean_dir = clean_scalar_str(os.fspath(cache_dir)) if isinstance(cache_dir, (str, os.PathLike)) else None
        if clean_dir:
            return str(resolve_cachedb_filepath(os.path.join(clean_dir, fname)))
        return str(cache_path(fname))

    @classmethod
    def load_sanitized_trade_snapshot(
        cls,
        path: Optional[str] = None,
        *,
        venue: str = "kraken",
        repair_files: bool = False,
    ) -> dict:
        """Load and sanitize a venue JSON trade snapshot ``{'items': ..., 'fetchtime': ...}`` with ``.previous`` recovery."""
        clean_path = clean_scalar_str(os.fspath(path)) if isinstance(path, (str, os.PathLike)) else None
        target = clean_path if clean_path else cls.resolve_cache_filepath(venue=venue)
        loaded = read_json_dict(
            target, default=None, recover_previous=True, repair_files=repair_files
        )
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
                    row_copy["time"] = math.trunc(t_f)
                    if "isBuyer" in row_copy and isinstance(row_copy["isBuyer"], str):
                        if (ib_b := _parse_bool_flag(row_copy["isBuyer"], default=None)) is not None:
                            row_copy["isBuyer"] = ib_b
                        elif row_copy["isBuyer"].strip().lower() == "nan":
                            row_copy["isBuyer"] = False
                    valid_rows.append(row_copy)
                valid_rows.sort(key=lambda t: t["time"])
                k_canon = cls._canonical_snapshot_symbol(k, venue=venue) or str(k)
                norm_rows = [cls._normalize_fill_row_symbol(rc, k_canon, venue=venue) for rc in valid_rows]
                merged_v, _ = cls._merge_fill_rows_by_dedup_key(clean_items.get(k_canon, []), norm_rows)
                merged_v.sort(key=lambda t: t["time"])
                clean_items[k_canon] = merged_v
            clean_fetchtime = {}
            for k, v in loaded["fetchtime"].items():
                v_f = safe_nonneg_float(v, default=None)
                k_canon_ft = cls._canonical_snapshot_symbol(k, venue=venue) or str(k)
                if v_f is not None and (not clean_items or k_canon_ft in clean_items):
                    clean_fetchtime[k_canon_ft] = max(clean_fetchtime.get(k_canon_ft, 0.0), v_f)
            return {
                "items": clean_items,
                "fetchtime": clean_fetchtime,
            }
        return {"items": {}, "fetchtime": {}}

    @classmethod
    def _canonical_snapshot_symbol(cls, raw_sym: object, *, venue: Optional[str] = None) -> str:
        """Return canonical display key for a venue trade snapshot symbol."""
        if not isinstance(raw_sym, str) or isinstance(raw_sym, bool):
            return ""
        stripped = raw_sym.strip()
        if not stripped:
            return ""
        return sym.canonical_display_symbol(stripped, venue=venue) or stripped.upper()

    @classmethod
    def _normalize_fill_row_symbol(
        cls, row: dict, canon_sym: str, *, venue: Optional[str] = None
    ) -> dict:
        rc = dict(row)
        raw_sym = rc.get("symbol")
        if isinstance(raw_sym, str) and (
            raw_sym.upper() == canon_sym.upper()
            or cls._canonical_snapshot_symbol(raw_sym, venue=venue) == canon_sym
        ):
            rc["symbol"] = canon_sym
        return rc

    @classmethod
    def _merge_fill_rows_by_dedup_key(
        cls, base_rows: list, incoming_rows: list
    ) -> tuple[list[dict], bool]:
        """Merge ``incoming_rows`` into ``base_rows``, deduplicating by ``fill_dedup_key``."""
        bucket: list[dict] = []
        seen_keys: set[tuple] = set()
        changed = False
        for row in base_rows:
            if not isinstance(row, dict):
                continue
            dk = cls.fill_dedup_key(row)
            if dk is not None:
                if dk in seen_keys:
                    changed = True
                    continue
                seen_keys.add(dk)
            bucket.append(dict(row))
        for row in incoming_rows:
            if not isinstance(row, dict):
                continue
            dk = cls.fill_dedup_key(row)
            if dk is not None and dk not in seen_keys:
                bucket.append(dict(row))
                seen_keys.add(dk)
                changed = True
        return bucket, changed

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
        if not isinstance(mapping, dict):
            return []
        if callable(match_fn):
            return [k for k in mapping if isinstance(k, str) and match_fn(symbol, k)]
        if candidate_keys is None:
            raw_cands = (symbol,)
        elif isinstance(candidate_keys, str):
            raw_cands = (candidate_keys,)
        elif isinstance(candidate_keys, (list, tuple, set, frozenset)):
            raw_cands = candidate_keys
        else:
            raw_cands = ()
        cands = {s for c in raw_cands if isinstance(c, str) and (s := c.strip().upper())}
        if not cands:
            return []
        return [
            k
            for k in mapping
            if isinstance(k, str) and (k.strip().upper() in cands or clean_symbol_key(k) in cands)
        ]

    @staticmethod
    def _load_venue_snapshot_dict(
        cache_file: str, *, create: bool = False, repair_files: bool = False
    ) -> Optional[dict]:
        """Load a venue trade snapshot ``{'items': {...}, 'fetchtime': {...}}`` (or initialize one when ``create=True``)."""
        data = read_json_dict_recovering(
            str(cache_file), default=None, repair_files=repair_files
        )
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
        max_stale_s = resolve_positive_max_age(max_stale_sec, 90.0)
        if max_stale_s is None:
            return None
        sym_clean = clean_symbol_key(symbol) or str(symbol).strip().upper()
        grouped_rows: dict[str, list[dict]] = {}
        grouped_counts: dict[str, int] = {}
        grouped_ranks: dict[str, tuple[float, int]] = {}
        for key in keys:
            if not isinstance(items.get(key), list):
                continue
            ft_val = safe_pos_float(casefold_get(ft, key), default=None)
            if ft_val is None or not _is_fresh_obs_ts(_ts_ms_to_s(ft_val), now_s, max_stale_s):
                continue
            k_canon = cls._canonical_snapshot_symbol(key) or key.strip().upper()
            rows_k = (
                [normalize_fn(t) for t in items[key] if isinstance(t, dict)]
                if callable(normalize_fn)
                else [cls._normalize_fill_row_symbol(t, k_canon) for t in items[key] if isinstance(t, dict)]
            )
            if k_canon in grouped_rows:
                merged_rows, _ = cls._merge_fill_rows_by_dedup_key(grouped_rows[k_canon], rows_k)
                grouped_rows[k_canon] = merged_rows
                grouped_counts[k_canon] += 1
            else:
                grouped_rows[k_canon], _ = cls._merge_fill_rows_by_dedup_key(rows_k, [])
                grouped_counts[k_canon] = 1
            rank = (_ts_ms_to_s(ft_val), 1 if key.strip().upper() == sym_clean else 0)
            if k_canon not in grouped_ranks or rank > grouped_ranks[k_canon]:
                grouped_ranks[k_canon] = rank
        if not grouped_ranks:
            return None
        best_canon = max(grouped_ranks, key=lambda kc: grouped_ranks[kc])
        best_rows = grouped_rows[best_canon]
        if grouped_counts[best_canon] > 1 and all(
            isinstance(r, dict) and "time" in r and "tid" not in r for r in best_rows
        ):
            best_rows.sort(key=lambda r: extract_timestamp_ms(r, default=0) or 0)
        return best_rows

    @staticmethod
    def _prepare_venue_cache_dict(
        cache_obj: dict, now_ms=None
    ) -> Optional[tuple[dict, dict, int]]:
        if not isinstance(cache_obj, dict):
            return None
        return (
            cache_obj.setdefault("items", {}),
            cache_obj.setdefault("fetchtime", {}),
            resolve_now_ms(now_ms),
        )

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
        if fills_by_symbol is not None and not isinstance(fills_by_symbol, dict):
            return False
        prep = cls._prepare_venue_cache_dict(cache_obj, now_ms)
        if prep is None:
            return False
        is_snapshot_b = _parse_bool_flag(is_snapshot, False)
        items, ft, eff_now_ms = prep
        changed = False
        for sym_raw, new_rows in (fills_by_symbol or {}).items():
            if not isinstance(sym_raw, str) or not isinstance(new_rows, list):
                continue
            sym_key = cls._canonical_snapshot_symbol(sym_raw)
            if not sym_key:
                continue
            alias_rows: list[dict] = []
            for alt_k in [
                ik
                for ik in list(items)
                if ik != sym_key
                and (str(ik).upper() == sym_key.upper() or cls._canonical_snapshot_symbol(ik) == sym_key)
            ]:
                alt_val = items.pop(alt_k, None)
                if isinstance(alt_val, list):
                    alias_rows.extend(
                        cls._normalize_fill_row_symbol(r, sym_key)
                        for r in alt_val
                        if isinstance(r, dict)
                    )
                changed = True
            for alt_fk in [
                fk
                for fk in list(ft)
                if fk != sym_key
                and (str(fk).upper() == sym_key.upper() or cls._canonical_snapshot_symbol(fk) == sym_key)
            ]:
                alt_ft = ft.pop(alt_fk, None)
                if alt_ft is not None:
                    cur_ft_f = safe_pos_float(ft.get(sym_key), default=None)
                    alt_ft_f = safe_pos_float(alt_ft, default=None)
                    if cur_ft_f is None or (alt_ft_f is not None and alt_ft_f > cur_ft_f):
                        ft[sym_key] = alt_ft
                changed = True
            new_rows = [
                cls._normalize_fill_row_symbol(r, sym_key)
                for r in new_rows
                if isinstance(r, dict)
            ]
            if is_snapshot_b:
                clean_rows = [dict(r) for r in new_rows]
                if items.get(sym_key) != clean_rows or ft.get(sym_key) != eff_now_ms:
                    changed = True
                items[sym_key] = clean_rows
                ft[sym_key] = eff_now_ms
                continue
            existing = items.get(sym_key)
            existing_list = (
                [cls._normalize_fill_row_symbol(r, sym_key) for r in existing if isinstance(r, dict)]
                if isinstance(existing, list)
                else []
            )
            if alias_rows:
                existing_list.extend(alias_rows)
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
                bucket, merged_changed = cls._merge_fill_rows_by_dedup_key(existing_list, new_rows)
                if merged_changed:
                    changed = True
                if (merged_changed or alias_rows) and all(
                    isinstance(r, dict) and "time" in r and "tid" not in r for r in bucket
                ):
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
    def _merge_venue_fills_snapshot_locked(
        cls,
        cache_file: str,
        fills_by_symbol: dict,
        *,
        now_ms: Optional[int] = None,
        is_snapshot: bool = False,
        touch_symbols=None,
        touch_create_missing: bool = False,
        sanitize_venue: Optional[str] = None,
        indent: Optional[int] = None,
        force_save: bool = False,
    ) -> dict:
        """Atomically load, merge, touch, and persist a venue trade snapshot under ``.generation.lock``."""
        target_path = str(cache_file)
        is_snapshot_b = _parse_bool_flag(is_snapshot, False)
        touch_create_b = _parse_bool_flag(touch_create_missing, False)
        force_save_b = _parse_bool_flag(force_save, False)
        with generation_transaction(target_path) as target_path:
            cache_obj = (
                cls.load_sanitized_trade_snapshot(
                    target_path, venue=sanitize_venue, repair_files=True
                )
                if sanitize_venue
                else cls._load_venue_snapshot_dict(
                    target_path, create=True, repair_files=True
                )
            )
            changed = cls.merge_venue_fills_into_dict(
                cache_obj,
                fills_by_symbol,
                now_ms=now_ms,
                is_snapshot=is_snapshot_b,
                touch_symbols=None,
            )
            if touch_symbols and cls.touch_venue_ws_fetchtimes_in_dict(
                cache_obj,
                now_ms,
                symbols=touch_symbols,
                create_missing=touch_create_b,
            ):
                changed = True
            if changed or is_snapshot_b or force_save_b or not os.path.exists(target_path):
                save_json_state(target_path, cache_obj, indent=indent, skip_if_unchanged=True)
            return cache_obj

    @classmethod
    def merge_venue_fills_snapshot(
        cls,
        cache_file: str,
        fills_by_symbol: dict,
        *,
        now_ms: Optional[int] = None,
        is_snapshot: bool = False,
        touch_symbols=None,
        touch_create_missing: bool = False,
        sanitize_venue: Optional[str] = None,
        indent: Optional[int] = None,
    ) -> bool:
        """Merge ``{symbol: [fill_dict, ...]}`` into a shared SSOT trade cache file (`items` + `fetchtime`)."""
        is_snapshot_b = _parse_bool_flag(is_snapshot, False)
        if not cache_file or not isinstance(fills_by_symbol, dict):
            return False
        if not fills_by_symbol and not touch_symbols and not is_snapshot_b:
            return False
        try:
            cls._merge_venue_fills_snapshot_locked(
                str(cache_file),
                fills_by_symbol,
                now_ms=now_ms,
                is_snapshot=is_snapshot_b,
                touch_symbols=touch_symbols,
                touch_create_missing=touch_create_missing,
                sanitize_venue=sanitize_venue,
                indent=indent,
            )
            return True
        except Exception:
            return False

    @classmethod
    def touch_venue_fills_snapshot(
        cls,
        cache_file: str,
        now_ms=None,
        *,
        symbols=None,
        create_missing: bool = False,
        indent: Optional[int] = None,
    ) -> bool:
        """Atomically refresh ``fetchtime`` across active WS symbols under ``.generation.lock``."""
        if not cache_file:
            return False
        target_path = str(cache_file)
        create_missing_b = _parse_bool_flag(create_missing, False)
        try:
            with generation_transaction(target_path) as target_path:
                loaded = cls._load_venue_snapshot_dict(
                    target_path, create=create_missing_b, repair_files=True
                )
                if loaded is None:
                    return False
                touched = cls.touch_venue_ws_fetchtimes_in_dict(
                    loaded, now_ms, symbols=symbols, create_missing=create_missing_b
                )
                if touched or (create_missing_b and not os.path.exists(target_path)):
                    save_json_state(target_path, loaded, indent=indent, skip_if_unchanged=True)
                return touched
        except Exception:
            return False

    @classmethod
    def touch_venue_ws_fetchtimes_in_dict(
        cls,
        cache_obj: dict,
        now_ms=None,
        *,
        symbols=None,
        create_missing: bool = False,
    ) -> bool:
        """Refresh ``fetchtime`` in ``cache_obj`` across active WS symbols."""
        if (prep := cls._prepare_venue_cache_dict(cache_obj, now_ms)) is None:
            return False
        items, ft, resolved_ms = prep
        raw_syms = (
            (symbols,)
            if isinstance(symbols, str)
            else (symbols if isinstance(symbols, (list, tuple, set, frozenset)) else ())
        )
        has_wildcard = any(isinstance(s, str) and s.strip() == "*" for s in raw_syms)
        valid_syms: list[str] = []
        seen_valid: set[str] = set()
        for s in raw_syms:
            if not isinstance(s, str) or not (s_clean := s.strip()) or s_clean == "*":
                continue
            s_key = s_clean if s_clean in items else (cls._canonical_snapshot_symbol(s_clean) or s_clean)
            if s_key not in seen_valid:
                seen_valid.add(s_key)
                valid_syms.append(s_key)
        if _parse_bool_flag(create_missing, False):
            changed = False
            touch_syms = list(items.keys()) if has_wildcard else list(valid_syms)
            for s_sym in valid_syms:
                if s_sym not in items:
                    items[s_sym] = []
                    changed = True
                if has_wildcard and s_sym not in touch_syms:
                    touch_syms.append(s_sym)
            for s_sym in touch_syms:
                if ft.get(s_sym) != resolved_ms:
                    ft[s_sym] = resolved_ms
                    changed = True
            return changed
        cand_syms = list(items.keys()) if has_wildcard else valid_syms
        touch_syms = [
            s_sym for s_sym in cand_syms
            if s_sym in ft and s_sym in items and ft.get(s_sym) != resolved_ms
        ]
        for s_sym in touch_syms:
            ft[s_sym] = resolved_ms
        return bool(touch_syms)

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
        target_path = str(cache_file)
        try:
            with generation_transaction(target_path) as target_path:
                data = read_json_dict_recovering(
                    target_path, default=None, repair_files=True
                )
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
                    save_json_state(target_path, data, skip_if_unchanged=True)
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
        if self.read_only:
            return
        with self._generation_lock():
            return self._migrate_legacy_json_generation_locked()

    def _migrate_legacy_json_generation_locked(self):
        return self._migrate_legacy_json_with_compactor(
            self._compact_account_trade_jsonl_generation_locked,
            post_compact_fn=lambda: self._read_trade_snapshot_strict(
                self._load_versioned_account_meta(optional=False)
                or (_ for _ in ()).throw(ValueError("versioned Trade metadata was not created"))
            ),
            skip_if_read_only=True,
            read_json_fn=read_json_dict,
        )

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
                return self._log_tagged_false(
                    "resync",
                    "durable Trade generation differs from this manager; refusing stale append",
                )
            if self._is_file_newer_than_memory_locked():
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
                metadata = self._account_meta_snapshot(
                    data_version=version,
                    stream_id=stream_id,
                    revision=revision,
                    committed_bytes=committed_bytes,
                    content_digest=content_digest,
                    fetchtime=dict(self.fetchtime_per_symbol),
                    rest_sync_cursor=dict(self._rest_sync_cursor_per_symbol),
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
                return self._log_account_save_error(f"append JSONL {self.filename}", exc)

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
                            str(symbol).upper(): {
                                str(strict_ascii_int(item["id"], default=0))
                                for item in items
                            }
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
                            trade_id = str(strict_ascii_int(item["id"], default=0))
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
                if not can_append:
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

            legacy_metadata = self._read_optional_meta_dict()
            with self.lock:
                self.cache = _cache_integrity.read_legacy_trade_jsonl_strict(
                    self.filename,
                    legacy_metadata,
                    is_valid_item_fn=self._is_valid_item,
                    reader_reason=self._account_reader_reason(),
                )
                self._legacy_jsonl_needs_rewrite = True
                self._sync_persisted_counts_and_legacy_fetchtimes_locked(
                    legacy_metadata
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
            return self._log_account_save_error(f"compact JSONL {self.filename}", exc)

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
                    archive = self._archive_snapshot_path()
                    self._snapshot_file_to(archive)
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
                        return self._log_tagged_false(
                            "maintain",
                            "Trade rotation aborted because generation publication failed",
                        )
                    published = True
            return self._finalize_rotation(archive)
        except Exception as exc:
            if not published and original_state is not None:
                with self.lock:
                    self._restore_memory_state_locked(original_state)
                    self._account_cache_dirty, self._legacy_jsonl_needs_rewrite = (
                        original_dirty,
                        original_needs_rewrite,
                    )
            if not published and archive is not None:
                discard_generation_file(archive)
            return self._log_tagged_false("maintain", f"Trade rotation failed: {exc}")

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
            target_sym = self._canonical_snapshot_symbol(symbol, venue=self.venue) or symbol
            with self.lock:
                alias_keys = [
                    k for k in list(self.cache.keys())
                    if k != target_sym
                    and (
                        str(k).upper() == str(target_sym).upper()
                        or self._canonical_snapshot_symbol(k, venue=self.venue) == target_sym
                    )
                ]
                seed_rows: list = list(self.cache.get(target_sym) or ())
                coalesced_alias = False
                for ak in alias_keys:
                    seed_rows.extend(self.cache.pop(ak, ()) or ())
                    coalesced_alias = True
                for ft_map in (self.fetchtime_per_symbol, self._rest_sync_cursor_per_symbol):
                    for ak in [
                        k for k in list(ft_map.keys())
                        if k != target_sym
                        and (
                            str(k).upper() == str(target_sym).upper()
                            or self._canonical_snapshot_symbol(k, venue=self.venue) == target_sym
                        )
                    ]:
                        ft_val = safe_nonneg_int(ft_map.pop(ak, None), default=None)
                        if ft_val is not None:
                            cur_ft = safe_nonneg_int(ft_map.get(target_sym), default=0) or 0
                            ft_map[target_sym] = max(cur_ft, ft_val)
                by_key: dict[tuple, dict] = {}
                for item in seed_rows:
                    if isinstance(item, dict):
                        norm_item = self._normalize_fill_row_symbol(item, target_sym, venue=self.venue)
                        k = self.fill_dedup_key(norm_item) or ("idx", len(by_key))
                        by_key.setdefault(k, norm_item)
                changed = coalesced_alias
                for item in new_items:
                    if not isinstance(item, dict) or (extract_order_price(item, default=None) is None and safe_pos_float(item.get("price"), default=None) is None):
                        continue
                    norm_item = self._normalize_fill_row_symbol(item, target_sym, venue=self.venue)
                    k = self.fill_dedup_key(norm_item) or ("idx", len(by_key))
                    if by_key.get(k) != norm_item:
                        by_key[k] = norm_item
                        changed = True
                if changed:
                    self.cache[target_sym] = sorted(
                        by_key.values(),
                        key=lambda r: (extract_timestamp_ms(r, default=0) or 0, str(r.get("id") or r.get("orderId") or "")),
                    )
                    self._advance_symbol_fetchtime_locked(target_sym)
            return
        if not new_items:
            return
        new_items = self._coerce_item_list(new_items)

        def _validate_and_sign(item: Any, err_msg: str) -> tuple[str, tuple[Any, ...]]:
            if not self._is_valid_item(item) or str(item["symbol"]).upper() != str(symbol).upper():
                raise RuntimeError(err_msg)
            return str(strict_ascii_int(item["id"], default=0)), _cache_integrity.trade_semantic_signature(item)

        with self.lock:
            bucket = self.cache.setdefault(symbol, [])
            persisted_count = self._persisted_counts.get(symbol, 0)
            existing = {}
            for idx, item in enumerate(bucket):
                trade_id, signature = _validate_and_sign(item, "invalid Trade row already in memory")
                if trade_id in existing:
                    raise RuntimeError("duplicate immutable Trade ID in memory")
                existing[trade_id] = (signature, idx, item)

            unique = []
            enriched_unpersisted = False
            for item in new_items:
                trade_id, signature = _validate_and_sign(item, "invalid incoming Trade row")
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
                unpersisted_tail.sort(
                    key=lambda t: (
                        strict_ascii_int(t["time"], default=0) or 0,
                        strict_ascii_int(t["id"], default=0) or 0,
                    )
                )
                bucket[persisted_count:] = unpersisted_tail
            self._advance_symbol_fetchtime_locked(symbol)

    def get_last_opposite_fill_price(self, symbol, order_type):
        """Return the latest opposite fill price without a time limit."""
        self.sync_account_reader_if_needed()
        if not (side_u := normalize_side(order_type, require_str=True)):
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
                    key = (
                        strict_ascii_int(trade["time"], default=0) or 0,
                        strict_ascii_int(trade["id"], default=0) or 0,
                    )
                    px_val = safe_pos_float(trade["price"], default=0.0) or 0.0
                else:
                    if not isinstance(trade, dict) or (extract_order_price(trade, default=None) is None and safe_pos_float(trade.get("price"), default=None) is None):
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
                    key = (safe_nonneg_int(ts_ms, default=0) or 0, raw_id)
                if latest_key is None or key > latest_key:
                    latest_key = key
                    latest_price = px_val
        return latest_price


# =============================================================================
# 3. ORDER SNAPSHOT CACHES: CacheOrderFilledManager & CacheOrderOpenManager
# =============================================================================


_TERMINAL_NON_OPEN_STATUSES = TERMINAL_AUDIT_VENUE_STATUSES | TERMINAL_CANCEL_VENUE_STATUSES
_OPEN_ORDER_EXEC_KEYS = ("executedQty", "executed_qty", "filledQty", "filled_qty", "filledQuantity", "vol_exec")


def _sibling_order_filled_path(trade_cache_file: str, venue: str) -> str:
    """Resolve the sibling ``cache_order_filled_<venue>.json`` path in the same directory as ``trade_cache_file``."""
    v_norm = normalize_venue_name(venue, default="binance") or "binance"
    fname = CacheOrderFilledManager.disk_filename(v_norm)
    base_dir = os.path.dirname(os.path.abspath(str(trade_cache_file))) if trade_cache_file else ""
    return os.path.join(base_dir, fname) if base_dir else cache_path(fname)


_TERMINAL_FILL_ROW_STATUSES: frozenset[str] = TERMINAL_FILL_ROW_STATUSES


def _enrich_filled_row_metadata(
    existing: dict[str, Any],
    incoming: dict[str, Any],
    *,
    allow_fee_upgrade: bool = True,
) -> bool:
    """Backfill missing metadata (and settled positive fees) on an existing filled-order row without downgrading."""
    changed = False
    if not existing.get("clientOrderId") and incoming.get("clientOrderId"):
        existing["clientOrderId"] = incoming["clientOrderId"]
        changed = True
    in_type = clean_upper_scalar(incoming.get("type"))
    if not existing.get("type") and in_type:
        existing["type"] = in_type
        changed = True
    in_st = clean_upper_scalar(incoming.get("status"))
    ex_st = clean_upper_scalar(existing.get("status"))
    if in_st:
        merged_st = merge_monotonic_fill_row_status(ex_st, in_st, incoming_ts=0, existing_ts=1)
        if merged_st and merged_st != ex_st:
            existing["status"] = merged_st
            changed = True
    in_asset = clean_upper_scalar(incoming.get("feeAsset"))
    if not existing.get("feeAsset") and in_asset:
        existing["feeAsset"] = in_asset
        changed = True
    if _parse_bool_flag(allow_fee_upgrade, True):
        ex_fee = safe_nonneg_float(existing.get("fee"), default=None)
        in_fee = safe_nonneg_float(incoming.get("fee"), default=None)
        if in_fee is not None and (ex_fee is None or (ex_fee == 0.0 and in_fee > 0.0)):
            existing["fee"] = round(in_fee, 10)
            if in_asset:
                existing["feeAsset"] = in_asset
            changed = True
    return changed


def _subsume_provisional_fill_rows(by_oid: dict[str, dict[str, Any]]) -> bool:
    """Evict provisional fill-keyed rows (``orderId == fid``) once a real order row owns that ``fid``.

    Prevents double-counting venues (such as Kraken Ledgers vs ownTrades/TradesHistory) where an
    earlier ledger feed has no parent ``ordertxid`` and falls back to ``orderId = fill_id``.
    """
    real_fill_owner: dict[str, str] = {}
    for oid_key, row in by_oid.items():
        if not isinstance(row, dict):
            continue
        for raw_fid in (row.get("_fillIds") or ()):
            fid_s = clean_order_id(raw_fid)
            if fid_s and fid_s != oid_key and fid_s not in real_fill_owner:
                real_fill_owner[fid_s] = oid_key
    if not real_fill_owner:
        return False
    removed = False
    for oid_key in list(by_oid.keys()):
        real_owner = real_fill_owner.get(oid_key)
        if not real_owner or real_owner == oid_key or real_owner not in by_oid:
            continue
        prov_row = by_oid.get(oid_key)
        if not isinstance(prov_row, dict):
            continue
        prov_fids = [fid_s for raw_f in (prov_row.get("_fillIds") or ()) if (fid_s := clean_order_id(raw_f))]
        if prov_fids and prov_fids != [oid_key]:
            continue
        by_oid.pop(oid_key, None)
        target_row = by_oid[real_owner]
        _enrich_filled_row_metadata(target_row, prov_row, allow_fee_upgrade=False)
        removed = True
    return removed


def _merge_optional_order_fee(existing: dict, incoming: dict, merged: dict) -> None:
    """Accumulate optional ``fee`` and ``feeAsset`` across fills belonging to the same order."""
    ex_fee = safe_nonneg_float(existing.get("fee"), default=None)
    in_fee = safe_nonneg_float(incoming.get("fee"), default=None)
    ex_asset, in_asset = clean_upper_scalar(existing.get("feeAsset")) or "", clean_upper_scalar(incoming.get("feeAsset")) or ""
    if ex_fee is not None and in_fee is not None and (not ex_asset or not in_asset or ex_asset == in_asset):
        merged["fee"] = round(ex_fee + in_fee, 10)
        if eff_asset := (ex_asset or in_asset):
            merged["feeAsset"] = eff_asset
    elif ex_fee is not None:
        merged["fee"] = ex_fee
        if ex_asset:
            merged["feeAsset"] = ex_asset
    elif in_fee is not None:
        merged["fee"] = in_fee
        if in_asset:
            merged["feeAsset"] = in_asset


def _merge_fill_row_status(
    existing_st: Optional[str],
    incoming_st: Optional[str],
    *,
    incoming_ts: int,
    existing_ts: int,
) -> Optional[str]:
    return merge_monotonic_fill_row_status(
        existing_st,
        incoming_st,
        incoming_ts=incoming_ts,
        existing_ts=existing_ts,
    )


def _vwap_merge_filled_row(
    existing: dict[str, Any],
    incoming: dict[str, Any],
    *,
    oid_norm: Any,
    existing_ts: int,
    incoming_ts: int,
    fill_ids: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Compute a VWAP-merged filled-order aggregate row from ``existing`` and ``incoming``."""
    ex_q = safe_pos_float(existing.get("quantity"), default=0.0) or 0.0
    in_q = safe_pos_float(incoming.get("quantity"), default=0.0) or 0.0
    tot_q = ex_q + in_q
    ex_px = safe_pos_float(existing.get("price"), default=0.0) or 0.0
    in_px = safe_pos_float(incoming.get("price"), default=0.0) or 0.0
    vwap = (ex_px * ex_q + in_px * in_q) / tot_q if tot_q > 0.0 else in_px
    merged = dict(existing)
    merged.update(incoming)
    merged.update({
        "orderId": oid_norm,
        "price": round(vwap, 8) or round(vwap, 16),
        "quantity": round(tot_q, 8) or round(tot_q, 16),
        "timestamp": max(existing_ts, incoming_ts),
        "side": str(existing["side"]).upper(),
    })
    if cid_keep := existing.get("clientOrderId") or incoming.get("clientOrderId"):
        merged["clientOrderId"] = cid_keep
    if type_keep := clean_upper_scalar(existing.get("type") or incoming.get("type")):
        merged["type"] = type_keep
    if merged_st := _merge_fill_row_status(
        existing.get("status"), incoming.get("status"), incoming_ts=incoming_ts, existing_ts=existing_ts
    ):
        merged["status"] = merged_st
    _merge_optional_order_fee(existing, incoming, merged)
    if fill_ids:
        merged["_fillIds"] = list(dict.fromkeys(fill_ids))
    return merged


class CacheOrderFilledManager(CacheAccountManagerBase):
    """Cross-venue filled-order aggregate cache (Level 3) backed by JSON snapshots and schema-v2 manifests."""

    _is_order_account_cache = True
    DEFAULT_FILENAME = CACHE_BINANCE_ORDER_FILLED_FILENAME
    VENUE_FILENAMES: dict[str, str] = {
        "binance": CACHE_BINANCE_ORDER_FILLED_FILENAME,
        "kraken": CACHE_KRAKEN_ORDER_FILLED_FILENAME,
        "hyperliquid": CACHE_HYPERLIQUID_ORDER_FILLED_FILENAME,
        "t212": CACHE_T212_ORDER_FILLED_FILENAME,
    }

    def __init__(
        self,
        sync_ts,
        symbols,
        filename,
        api_client=None,
        read_only=False,
        *,
        data_sync_mode: Optional[str] = None,
        venue: Optional[str] = None,
    ):
        self.venue = self._normalize_venue_arg(venue=venue, filename=filename, symbols=symbols)
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

    def _requires_int_order_id(self) -> bool:
        return bool(is_primary_stream_venue(getattr(self, "venue", "binance")))

    def _is_canonical_account_cache(self):
        return self._requires_int_order_id() and super()._is_canonical_account_cache()

    @staticmethod
    def _normalize_fill_id(raw_fill_id, *, strict_int: bool = True) -> Optional[str]:
        if raw_fill_id is None or isinstance(raw_fill_id, bool):
            return None
        s = str(raw_fill_id).strip()
        if not s:
            return None
        if not strict_int:
            return clean_order_id(s)
        if not s.isascii() or not s.isdigit():
            return None
        return s.lstrip("0") or "0"

    @staticmethod
    def _normalize_order_id(raw_order_id, *, strict_int: bool = True) -> Optional[int | str]:
        if raw_order_id is None or isinstance(raw_order_id, bool):
            return None
        if strict_int:
            return strict_ascii_int(raw_order_id, min_value=1)
        if isinstance(raw_order_id, int):
            return str(raw_order_id) if raw_order_id >= 1 else None
        return clean_order_id(raw_order_id)

    def _is_valid_item(self, trade):
        if not isinstance(trade, dict):
            return False
        required_keys = ("orderId", "price", "quantity", "timestamp", "side")
        if not all(key in trade for key in required_keys):
            return False
        strict_int = self._requires_int_order_id()
        legacy_cutoff = trade.get("_legacyCoveredThrough")
        if self._normalize_order_id(trade["orderId"], strict_int=strict_int) is None:
            return False
        if (
            safe_pos_float(trade["price"], default=None) is None
            or safe_pos_float(trade["quantity"], default=None) is None
            or strict_ascii_int(trade["timestamp"], min_value=1) is None
            or str(trade["side"]).upper() not in ("BUY", "SELL")
        ):
            return False
        if legacy_cutoff is not None and strict_ascii_int(legacy_cutoff, min_value=1) is None:
            return False
        for opt_str_key in ("clientOrderId", "status", "type", "feeAsset"):
            if opt_str_key in trade and trade[opt_str_key] is not None:
                if isinstance(trade[opt_str_key], bool) or clean_scalar_str(trade[opt_str_key]) is None:
                    return False
        if "fee" in trade and trade["fee"] is not None:
            if isinstance(trade["fee"], bool) or safe_nonneg_float(trade["fee"], default=None) is None:
                return False
        if "_fillIds" in trade:
            raw_fids = trade["_fillIds"]
            if not isinstance(raw_fids, list):
                return False
            norm_fids = {
                norm_fid
                for fill_id in raw_fids
                if (norm_fid := self._normalize_fill_id(fill_id, strict_int=strict_int)) is not None
            }
            if len(norm_fids) != len(raw_fids):
                return False
        return True

    def _load_snapshot_state(self):
        """Load Order snapshot state strictly when durable data or metadata exists."""
        with self._generation_lock():
            loaded = self._load_order_state_generation_locked()
        if not loaded and not self.read_only:
            self._log_missing_and_seed_remote()

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
                    return self._log_tagged_false(
                        "resync",
                        "durable Order generation differs from this manager; refusing stale overwrite",
                    )
                if self._is_file_newer_than_memory_locked():
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
                    lambda: (
                        write_json_fn(self.filename, payload, indent=1, skip_if_unchanged=True)
                        if write_json_fn is atomic_write_json
                        else write_json_fn(self.filename, payload, indent=1)
                    ),
                    metadata,
                    backup_path=backup_path,
                    had_prior_file=had_prior_file,
                ):
                    self._mark_account_cache_dirty_locked()
                    return False

                self._persisted_counts = dict(counts)
                self._remember_account_commit_locked(metadata)
                return self._log_cache_saved_ok()
        except Exception as exc:
            with self.lock:
                self._mark_account_cache_dirty_locked()
            return self._log_account_save_error(
                f"While saving the cache file {self.filename} / .tmp ", exc
            )

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
                self.cache, self.fetchtime_per_symbol = items, dict(fetchtime)
                self._rest_sync_cursor_per_symbol, self._persisted_counts = (
                    dict(metadata["rest_sync_cursor"]),
                    dict(actual_counts),
                )
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
        order_fills = []
        for fill in fills:
            row: dict[str, Any] = {
                "orderId": fill["orderId"],
                "price": extract_order_price(fill),
                "quantity": extract_order_qty(fill),
                "timestamp": extract_timestamp_ms(fill, default=0) or 0,
                "side": extract_order_side(fill),
                "_fillId": str(fill["id"]),
            }
            if cid := extract_client_order_id(fill, default=""):
                row["clientOrderId"] = cid
            if (fee_val := safe_nonneg_float(fill.get("commission"), default=None)) is not None:
                row["fee"] = fee_val
            if fee_asset := clean_upper_scalar(fill.get("commissionAsset")):
                row["feeAsset"] = fee_asset
            order_fills.append(row)
        print(f"[{self.cls_name}][info] New order fills: {len(order_fills)}")
        return order_fills

    def _persist_items(self, symbol, new_items):
        """Merge each immutable fill exactly once into its filled-order aggregate."""
        if not new_items or not (new_items := self._coerce_item_list(new_items)):
            return
        now_ms = math.trunc(_now_time() * 1000)
        strict_int = self._requires_int_order_id()
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
                oid_val = self._normalize_order_id(item["orderId"], strict_int=strict_int)
                oid_key = str(oid_val)
                if oid_key in by_id:
                    raise RuntimeError(
                        f"duplicate Order aggregate ID {oid_key} in memory"
                    )
                by_id[oid_key] = index
                for existing_fid in item.get("_fillIds", []):
                    fid_key = self._normalize_fill_id(existing_fid, strict_int=strict_int)
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
                    fill_id = self._normalize_fill_id(raw_fill_id, strict_int=strict_int)
                    if fill_id is None:
                        raise RuntimeError("invalid incoming Order fill ID")
                if not self._is_valid_item(incoming):
                    raise RuntimeError("invalid incoming Order row")
                oid_norm = self._normalize_order_id(incoming["orderId"], strict_int=strict_int)
                incoming_ts = strict_ascii_int(incoming["timestamp"], default=0, min_value=1) or 0
                incoming_px = safe_pos_float(incoming["price"], default=0.0) or 0.0
                incoming_qty = safe_pos_float(incoming["quantity"], default=0.0) or 0.0
                key = str(oid_norm)
                if fill_id:
                    prev_owner = fill_owners.get(fill_id)
                    if prev_owner is not None and prev_owner != key:
                        raise RuntimeError(
                            f"immutable fill ID {fill_id} belongs to both orders {prev_owner} and {key}"
                        )
                existing_index = by_id.get(key)
                if existing_index is None:
                    by_id[key] = len(candidate)
                    incoming["orderId"] = oid_norm
                    incoming["price"] = incoming_px
                    incoming["quantity"] = incoming_qty
                    incoming["timestamp"] = incoming_ts
                    incoming["side"] = str(incoming["side"]).upper()
                    if "fee" in incoming and incoming["fee"] is not None:
                        incoming["fee"] = safe_nonneg_float(incoming["fee"], default=0.0) or 0.0
                    if "feeAsset" in incoming and incoming["feeAsset"] is not None:
                        incoming["feeAsset"] = clean_upper_scalar(incoming["feeAsset"])
                    if "status" in incoming and incoming["status"] is not None:
                        incoming["status"] = clean_upper_scalar(incoming["status"])
                    if "type" in incoming and incoming["type"] is not None:
                        incoming["type"] = clean_upper_scalar(incoming["type"])
                    if fill_id:
                        incoming["_fillIds"] = [fill_id]
                        fill_owners[fill_id] = key
                    candidate.append(incoming)
                    continue

                existing = candidate[existing_index]
                existing_ts = strict_ascii_int(existing["timestamp"], default=0, min_value=1) or 0
                known_fill_id_list = [
                    self._normalize_fill_id(existing_fill_id, strict_int=strict_int)
                    for existing_fill_id in existing.get("_fillIds", [])
                ]
                legacy_cutoff = safe_nonneg_int(existing.get("_legacyCoveredThrough"), default=None)
                if legacy_cutoff is None and "_fillIds" not in existing:
                    legacy_cutoff = existing_ts
                known_fill_ids = {*known_fill_id_list}
                if fill_id and fill_id in known_fill_ids:
                    _enrich_filled_row_metadata(
                        existing,
                        incoming,
                        allow_fee_upgrade=(len(known_fill_ids) <= 1),
                    )
                    continue
                if legacy_cutoff is not None and incoming_ts <= legacy_cutoff:
                    continue
                if (
                    str(existing.get("side", "")).upper()
                    != str(incoming["side"]).upper()
                ):
                    raise RuntimeError(
                        f"{self.cls_name} received conflicting sides for order {key}"
                    )
                merged = _vwap_merge_filled_row(
                    existing,
                    {**incoming, "price": incoming_px, "quantity": incoming_qty},
                    oid_norm=self._normalize_order_id(existing["orderId"], strict_int=strict_int),
                    existing_ts=existing_ts,
                    incoming_ts=incoming_ts,
                    fill_ids=[*known_fill_id_list, fill_id] if fill_id else None,
                )
                if fill_id:
                    fill_owners[fill_id] = key
                if legacy_cutoff is not None:
                    merged["_legacyCoveredThrough"] = legacy_cutoff
                candidate[existing_index] = merged

            if candidate != bucket:
                self.cache[symbol] = candidate
                self._advance_symbol_fetchtime_locked(symbol, now_ms)

    @classmethod
    def _extract_compact_fill_record(
        cls,
        raw: Any,
        *,
        strict_int: bool,
    ) -> Optional[dict[str, Any]]:
        """Normalize one raw venue fill into a compact canonical fill dict for exact deduplicated aggregation."""
        if not isinstance(raw, dict):
            return None
        raw_oid = (
            raw.get("orderId")
            or raw.get("order_id")
            or raw.get("ordertxid")
            or raw.get("oid")
            or extract_order_id(raw, default=None)
        )
        oid_norm = cls._normalize_order_id(raw_oid, strict_int=strict_int)
        if oid_norm is None:
            return None
        px = safe_pos_float(extract_order_price(raw, default=None), default=None)
        qty = safe_pos_float(extract_executed_qty(raw, default=None), default=None) or safe_pos_float(
            extract_order_qty(raw, default=None), default=None
        )
        ts_ms = extract_timestamp_ms(raw, default=None)
        side = extract_order_side(raw) or normalize_side(raw.get("side"))
        if px is None or qty is None or ts_ms is None or ts_ms <= 0 or side not in ("BUY", "SELL"):
            return None
        raw_fid = raw.get("id") or raw.get("tid") or raw.get("trade_id") or raw.get("_fillId")
        fid = cls._normalize_fill_id(raw_fid, strict_int=strict_int)
        cid = extract_client_order_id(raw, default="")
        st_u = clean_upper_scalar(raw.get("status"))
        raw_type = clean_upper_scalar(raw.get("type") or raw.get("orderType"))
        otype = raw_type if raw_type not in (None, "BUY", "SELL", "B", "S") else None
        fee_info = extract_order_fee(raw, fill_price=px)
        fee_val = safe_nonneg_float(fee_info["fee"], default=None) if isinstance(fee_info, dict) else None
        fee_asset = clean_upper_scalar(fee_info.get("fee_asset")) if isinstance(fee_info, dict) else None
        rec: dict[str, Any] = {
            "orderId": oid_norm,
            "side": side,
            "price": round(px, 8) or round(px, 16),
            "quantity": round(qty, 8) or round(qty, 16),
            "timestamp": math.trunc(ts_ms),
        }
        if fid:
            rec["_fillId"] = fid
        if cid:
            rec["clientOrderId"] = cid
        if st_u:
            rec["status"] = st_u
        if otype:
            rec["type"] = otype
        if fee_val is not None:
            rec["fee"] = round(fee_val, 10)
        if fee_asset:
            rec["feeAsset"] = fee_asset
        return rec

    @classmethod
    def aggregate_fills_to_order_rows(
        cls,
        fills: Any,
        *,
        venue: str = "",
    ) -> list[dict[str, Any]]:
        """Aggregate raw venue fills into canonical compact per-order rows for ``CacheOrderFilledManager``.

        Preserves executed quantity and only records ``status`` when explicitly provided by the venue
        (never synthesizes ``FILLED`` for raw execution fills that do not carry an order status).
        """
        if not isinstance(fills, (list, tuple)):
            return []
        strict_int = bool(is_primary_stream_venue(venue)) if venue else False
        grouped: dict[str, dict[str, Any]] = {}
        order_sequence: list[str] = []
        for raw in fills:
            rec = cls._extract_compact_fill_record(raw, strict_int=strict_int)
            if rec is None:
                continue
            oid_norm = rec["orderId"]
            oid_key = str(oid_norm)
            px = rec["price"]
            qty = rec["quantity"]
            ts_ms = rec["timestamp"]
            side = rec["side"]
            fid = rec.get("_fillId")
            cid = rec.get("clientOrderId")
            st_u = rec.get("status")
            otype = rec.get("type")
            fee_val = rec.get("fee")
            fee_asset = rec.get("feeAsset")

            if oid_key not in grouped:
                order_sequence.append(oid_key)
                entry: dict[str, Any] = {
                    "orderId": oid_norm,
                    "side": side,
                    "price": px,
                    "quantity": qty,
                    "timestamp": ts_ms,
                }
                if st_u:
                    entry["status"] = st_u
                if cid:
                    entry["clientOrderId"] = cid
                if otype:
                    entry["type"] = otype
                if fee_val is not None:
                    entry["fee"] = fee_val
                if fee_asset:
                    entry["feeAsset"] = fee_asset
                if fid:
                    entry["_fillIds"] = [fid]
                elif isinstance(raw, dict) and isinstance(raw.get("_fillIds"), list):
                    norm_fids = [
                        nf
                        for x in raw["_fillIds"]
                        if (nf := cls._normalize_fill_id(x, strict_int=strict_int)) is not None
                    ]
                    if norm_fids:
                        entry["_fillIds"] = list(dict.fromkeys(norm_fids))
                grouped[oid_key] = entry
                continue

            cur = grouped[oid_key]
            known_fids = cur.get("_fillIds") or []
            if fid and fid in known_fids:
                if cid and not cur.get("clientOrderId"):
                    cur["clientOrderId"] = cid
                if st_u and not cur.get("status"):
                    cur["status"] = st_u
                continue
            prev_q = safe_pos_float(cur["quantity"], default=0.0) or 0.0
            prev_px = safe_pos_float(cur["price"], default=0.0) or 0.0
            tot_q = prev_q + qty
            vwap = (prev_px * prev_q + px * qty) / tot_q if tot_q > 0.0 else px
            cur["price"] = round(vwap, 8) or round(vwap, 16)
            cur["quantity"] = round(tot_q, 8) or round(tot_q, 16)
            prev_ts = cur["timestamp"]
            cur["timestamp"] = max(prev_ts, ts_ms)
            if cid and not cur.get("clientOrderId"):
                cur["clientOrderId"] = cid
            if otype and not cur.get("type"):
                cur["type"] = otype
            if merged_st := _merge_fill_row_status(
                cur.get("status"), st_u, incoming_ts=ts_ms, existing_ts=prev_ts
            ):
                cur["status"] = merged_st
            if fee_val is not None:
                _merge_optional_order_fee(cur, {"fee": fee_val, "feeAsset": fee_asset}, cur)
            if fid:
                cur["_fillIds"] = [*known_fids, fid]
        _subsume_provisional_fill_rows(grouped)
        rows = [grouped[k] for k in order_sequence if k in grouped]
        rows.sort(key=lambda r: (r.get("timestamp") or 0, str(r.get("orderId") or "")))
        return rows

    @classmethod
    def merge_venue_filled_orders_snapshot(
        cls,
        cache_file: str,
        fills_by_symbol: dict,
        *,
        venue: str = "",
        now_ms: Optional[int] = None,
        is_snapshot: bool = False,
        touch_symbols=None,
        touch_create_missing: bool = False,
        indent: Optional[int] = 1,
        force_save: bool = False,
    ) -> dict:
        """Atomically aggregate and merge per-symbol fills into ``cache_order_filled_<venue>.json``."""
        target_path = str(cache_file)
        eff_now_ms = resolve_now_ms(now_ms)
        strict_int = bool(is_primary_stream_venue(venue)) if venue else False
        is_snapshot_b = _parse_bool_flag(is_snapshot, False)
        touch_create_b = _parse_bool_flag(touch_create_missing, False)
        force_save_b = _parse_bool_flag(force_save, False)
        with generation_transaction(target_path) as target_path:
            loaded = read_json_dict_recovering(target_path, repair_files=True)
            if not isinstance(loaded, dict):
                loaded = {}
            items = loaded.get("items") if isinstance(loaded.get("items"), dict) else {}
            ft = loaded.get("fetchtime") if isinstance(loaded.get("fetchtime"), dict) else {}
            cache_obj = {"items": items, "fetchtime": ft}
            changed = False
            for sym_key, raw_rows in (fills_by_symbol or {}).items():
                if clean_scalar_str(sym_key) is None or not isinstance(raw_rows, (list, tuple)):
                    continue
                existing_bucket = items.get(sym_key) if isinstance(items.get(sym_key), list) else []
                by_oid: dict[str, dict[str, Any]] = {}
                for ex in existing_bucket:
                    if isinstance(ex, dict) and (ex_oid := cls._normalize_order_id(ex.get("orderId"), strict_int=strict_int)) is not None:
                        by_oid[str(ex_oid)] = dict(ex)

                # Extract compact individual raw fills (keyed by oid -> fid -> compact fill) so
                # overlapping batches (e.g. [1,2] then [2,3]) deduplicate before aggregating qty/VWAP.
                raw_by_oid: dict[str, dict[str, dict[str, Any]]] = {}
                for r in raw_rows:
                    compact = cls._extract_compact_fill_record(r, strict_int=strict_int)
                    if compact is not None and (fid := compact.get("_fillId")):
                        raw_by_oid.setdefault(str(compact["orderId"]), {})[str(fid)] = compact

                merged_from_fills: set[str] = set()
                for oid_str, new_fills in raw_by_oid.items():
                    if oid_str not in by_oid:
                        continue
                    ex_row = by_oid[oid_str]
                    ex_fids = list(ex_row.get("_fillIds") or ())
                    ex_fills_raw = ex_row.get("_fills") if isinstance(ex_row.get("_fills"), dict) else (
                        ex_row.get("_rawFills") if isinstance(ex_row.get("_rawFills"), dict) else None
                    )
                    can_union_ex = (
                        isinstance(ex_fills_raw, dict)
                        and bool(ex_fills_raw)
                        and (not ex_fids or set(ex_fids).issubset(set(ex_fills_raw.keys())))
                    )
                    can_snapshot_rebuild = (
                        is_snapshot_b and (not ex_fids or set(ex_fids).issubset(set(new_fills.keys())))
                    )
                    if can_union_ex or can_snapshot_rebuild:
                        merged_fills = ({**ex_fills_raw, **new_fills} if can_union_ex else dict(new_fills))
                        re_aggs = cls.aggregate_fills_to_order_rows(list(merged_fills.values()), venue=venue)
                        if re_aggs:
                            merged_order = re_aggs[0]
                            merged_order["_fills"] = merged_fills
                            _enrich_filled_row_metadata(merged_order, ex_row, allow_fee_upgrade=True)
                            by_oid[oid_str] = merged_order
                            merged_from_fills.add(oid_str)

                agg_rows = cls.aggregate_fills_to_order_rows(raw_rows, venue=venue)
                for inc in agg_rows:
                    inc_key = str(inc["orderId"])
                    if inc_key in merged_from_fills:
                        continue
                    if inc_key not in by_oid:
                        new_entry = dict(inc)
                        inc_fids = list(new_entry.get("_fillIds") or ())
                        if inc_key in raw_by_oid and inc_fids and set(inc_fids).issubset(set(raw_by_oid[inc_key].keys())):
                            new_entry["_fills"] = dict(raw_by_oid[inc_key])
                        by_oid[inc_key] = new_entry
                        continue
                    ex_row = by_oid[inc_key]
                    ex_fids = list(ex_row.get("_fillIds") or ())
                    inc_fids = list(inc.get("_fillIds") or ())
                    if not is_snapshot_b and inc_fids and ex_fids and set(inc_fids).issubset(set(ex_fids)):
                        _enrich_filled_row_metadata(
                            ex_row,
                            inc,
                            allow_fee_upgrade=(set(inc_fids) == set(ex_fids)),
                        )
                        continue
                    if (
                        (is_snapshot_b and (not ex_fids or not inc_fids or set(ex_fids).issubset(set(inc_fids))))
                        or (inc_fids and ex_fids and set(ex_fids).issubset(set(inc_fids)))
                    ):
                        merged_row = dict(ex_row)
                        merged_row.update(inc)
                        _enrich_filled_row_metadata(merged_row, ex_row, allow_fee_upgrade=True)
                        if inc_key in raw_by_oid and inc_fids and set(inc_fids).issubset(set(raw_by_oid[inc_key].keys())):
                            merged_row["_fills"] = dict(raw_by_oid[inc_key])
                        by_oid[inc_key] = merged_row
                    else:
                        ex_q = safe_pos_float(ex_row.get("quantity"), default=0.0) or 0.0
                        in_q = safe_pos_float(inc.get("quantity"), default=0.0) or 0.0
                        if inc_fids and ex_fids and not (set(inc_fids) & set(ex_fids)):
                            by_oid[inc_key] = _vwap_merge_filled_row(
                                ex_row,
                                inc,
                                oid_norm=ex_row["orderId"],
                                existing_ts=safe_nonneg_int(ex_row.get("timestamp"), default=0) or 0,
                                incoming_ts=safe_nonneg_int(inc.get("timestamp"), default=0) or 0,
                                fill_ids=[*ex_fids, *inc_fids],
                            )
                        elif in_q > ex_q + 1e-12 or (
                            safe_nonneg_int(inc.get("timestamp"), default=0) or 0
                        ) > (safe_nonneg_int(ex_row.get("timestamp"), default=0) or 0):
                            merged_row = dict(ex_row)
                            merged_row.update(inc)
                            _enrich_filled_row_metadata(merged_row, ex_row, allow_fee_upgrade=True)
                            if ex_fids or inc_fids:
                                merged_row["_fillIds"] = list(dict.fromkeys([*ex_fids, *inc_fids]))
                            by_oid[inc_key] = merged_row
                        else:
                            _enrich_filled_row_metadata(
                                ex_row,
                                inc,
                                allow_fee_upgrade=(abs(in_q - ex_q) <= 1e-12),
                            )
                _subsume_provisional_fill_rows(by_oid)
                new_bucket = sorted(
                    by_oid.values(),
                    key=lambda r: (safe_nonneg_int(r.get("timestamp"), default=0) or 0, str(r.get("orderId") or "")),
                )
                if new_bucket != existing_bucket or sym_key not in items:
                    items[sym_key] = new_bucket
                    changed = True
                old_ft = safe_pos_float(ft.get(sym_key), default=None)
                if old_ft is None or (eff_now_ms - old_ft) >= 60_000:
                    changed = True
                ft[sym_key] = eff_now_ms
            if touch_symbols:
                for ts_sym in touch_symbols:
                    if clean_scalar_str(ts_sym) is None:
                        continue
                    if ts_sym not in items and touch_create_b:
                        items[ts_sym] = []
                        changed = True
                    # When touch_create_missing=False, never resurrect a symbol whose fetchtime
                    # was explicitly invalidated after order execution until a real fill or REST sync arrives.
                    if touch_create_b or (ts_sym in items and ts_sym in ft):
                        old_ft = safe_pos_float(ft.get(ts_sym), default=None)
                        if old_ft is None or (eff_now_ms - old_ft) >= 60_000:
                            changed = True
                        ft[ts_sym] = eff_now_ms
            if changed or is_snapshot_b or force_save_b or not os.path.exists(target_path):
                save_json_state(target_path, cache_obj, indent=indent, skip_if_unchanged=True)
            return cache_obj

    @classmethod
    def touch_venue_filled_orders_snapshot(
        cls,
        cache_file: str,
        now_ms=None,
        *,
        symbols=None,
        create_missing: bool = False,
        indent: Optional[int] = 1,
    ) -> bool:
        """Atomically refresh ``fetchtime`` across active symbols in ``cache_order_filled_<venue>.json``.

        Delegates to ``CacheTradeManager.touch_venue_ws_fetchtimes_in_dict`` so ``create_missing=False``
        never resurrects a symbol whose ``fetchtime`` was invalidated after live order execution.
        """
        if not cache_file:
            return False
        target_path = str(cache_file)
        eff_now_ms = resolve_now_ms(now_ms)
        create_b = _parse_bool_flag(create_missing, False)
        try:
            with generation_transaction(target_path) as resolved_target:
                if not os.path.exists(resolved_target) and not create_b:
                    return False
                loaded = read_json_dict_recovering(resolved_target, repair_files=True)
                if not isinstance(loaded, dict):
                    loaded = {}
                items = loaded.get("items") if isinstance(loaded.get("items"), dict) else {}
                ft = loaded.get("fetchtime") if isinstance(loaded.get("fetchtime"), dict) else {}
                cache_obj = {"items": items, "fetchtime": ft}
                target_syms = symbols if symbols is not None else list(items.keys())
                touched = CacheTradeManager.touch_venue_ws_fetchtimes_in_dict(
                    cache_obj,
                    eff_now_ms,
                    symbols=target_syms,
                    create_missing=create_b,
                )
                if touched or (create_b and not os.path.exists(resolved_target)):
                    save_json_state(resolved_target, cache_obj, indent=indent, skip_if_unchanged=True)
                return touched
        except Exception:
            return False

    @classmethod
    def read_venue_filled_orders_snapshot(
        cls,
        cache_file: str,
        *,
        venue: str = "",
        read_json_fn=None,
        allow_legacy: bool = True,
        allow_trade_fallback: bool = False,
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
        """Passively read one venue's filled-order snapshot without mutating disk."""
        if not cache_file:
            return {}, {}
        resolved = str(resolve_cachedb_filepath(str(cache_file)))
        eff_venue = normalize_venue_name(venue) if venue else None
        if not eff_venue:
            base_name = os.path.basename(resolved)
            eff_venue = next((v_k for v_k, fn in cls.VENUE_FILENAMES.items() if fn == base_name), None)

        candidates: list[str] = []

        def _add_cand(path_str: str) -> None:
            if path_str and path_str not in candidates:
                candidates.append(path_str)

        _add_cand(resolved)
        if _parse_bool_flag(allow_legacy, True):
            for leg_p in legacy_filenames_for(resolved):
                _add_cand(str(resolve_cachedb_filepath(leg_p)))
                _add_cand(leg_p)
        if _parse_bool_flag(allow_trade_fallback, False) and eff_venue:
            try:
                trd_fname = CacheTradeManager.disk_filename(eff_venue)
            except ValueError:
                trd_fname = ""
            if trd_fname and not trd_fname.lower().endswith(".jsonl"):
                for cand_dir in dict.fromkeys((os.path.dirname(resolved), os.path.dirname(str(cache_file)))):
                    trd_p = os.path.join(cand_dir, trd_fname) if cand_dir else trd_fname
                    _add_cand(str(resolve_cachedb_filepath(trd_p)))
                    _add_cand(trd_p)

        first_valid: Optional[tuple[dict[str, list[dict[str, Any]]], dict[str, int]]] = None
        for cand_path in candidates:
            if read_json_fn is not None:
                data = read_json_fn(cand_path)
            else:
                if not os.path.isfile(cand_path):
                    continue
                data = read_json_dict_recovering(cand_path, repair_files=False)
            if not isinstance(data, dict) or not isinstance(data.get("items"), dict):
                continue
            raw_items = data["items"]
            raw_ft = data.get("fetchtime") if isinstance(data.get("fetchtime"), dict) else {}
            items_out: dict[str, list[dict[str, Any]]] = {}
            for sym_k, sym_orders in raw_items.items():
                if clean_scalar_str(sym_k) is None:
                    continue
                if isinstance(sym_orders, list):
                    items_out[sym_k] = [dict(r) for r in sym_orders if isinstance(r, dict)]
                elif isinstance(sym_orders, dict):
                    if extract_order_id(sym_orders, default=None) is not None:
                        items_out[sym_k] = [dict(sym_orders)]
                    else:
                        items_out[sym_k] = [dict(r) for r in sym_orders.values() if isinstance(r, dict)]
            ft_out = {
                str(k): safe_nonneg_int(v, default=0) or 0
                for k, v in raw_ft.items()
                if clean_scalar_str(k) is not None
            }
            if items_out:
                return items_out, ft_out
            if first_valid is None:
                first_valid = (items_out, ft_out)
        return first_valid if first_valid is not None else ({}, {})

    @classmethod
    def read_all_venues_filled_orders(
        cls,
        cachedb_dir: str,
        *,
        read_json_fn=None,
        allow_legacy: bool = True,
        allow_trade_fallback: bool = False,
    ) -> tuple[list[dict[str, Any]], set[str], dict[str, dict[str, int]]]:
        """Passively read all venues' ``cache_order_filled_<venue>.json`` files."""
        flat: list[dict[str, Any]] = []
        covered: set[str] = set()
        fetchtime_by_venue: dict[str, dict[str, int]] = {}
        if not cachedb_dir:
            return flat, covered, fetchtime_by_venue
        for venue_key, fname in cls.VENUE_FILENAMES.items():
            cand_path = str(resolve_cachedb_filepath(os.path.join(cachedb_dir, fname)))
            items, ft = cls.read_venue_filled_orders_snapshot(
                cand_path,
                venue=venue_key,
                read_json_fn=read_json_fn,
                allow_legacy=allow_legacy,
                allow_trade_fallback=allow_trade_fallback,
            )
            if not items and not ft:
                continue
            covered.add(venue_key)
            if ft:
                fetchtime_by_venue[venue_key] = ft
            for sym_k, rows in items.items():
                for row in rows:
                    flat.append({"venue": venue_key, "symbol": str(row.get("symbol") or sym_k), **row})
        return flat, covered, fetchtime_by_venue

    def get_last_opposite_fill_price(self, symbol, order_type):
        """Return the latest opposite filled-order price without a time limit."""
        self.sync_account_reader_if_needed()
        if not (side_u := normalize_side(order_type, require_str=True)):
            return None
        want_side = "BUY" if side_u == "SELL" else "SELL"
        strict_int = self._requires_int_order_id()
        sym_clean = clean_symbol_key(symbol)
        cand_clean_set = {clean_symbol_key(c) for c in cache_symbol_variants(symbol, include_base=True) if c}
        latest_key = None
        latest_price = None
        with self.lock:
            bucket = casefold_get(self.cache, symbol) or casefold_get(self.cache, sym_clean)
            if not bucket and isinstance(self.cache, dict):
                items_map = (
                    self.cache.get("items")
                    if isinstance(self.cache.get("items"), dict)
                    else self.cache
                )
                for cand in cache_symbol_variants(symbol, include_base=True):
                    if cand and isinstance(rows := casefold_get(items_map, cand), list) and rows:
                        bucket = rows
                        break
            for row in bucket or ():
                if not isinstance(row, dict) or not self._is_valid_item(row):
                    continue
                if row.get("symbol") is not None:
                    tr_sym = clean_symbol_key(row.get("symbol"))
                    if tr_sym and sym_clean and tr_sym != sym_clean and tr_sym not in cand_clean_set:
                        continue
                row_side = extract_order_side(row) or normalize_side(row.get("side"))
                if row_side != want_side:
                    continue
                px_val = safe_pos_float(row.get("price"), default=None)
                ts_ms = strict_ascii_int(row.get("timestamp"), default=0) or 0
                if px_val is None or ts_ms <= 0:
                    continue
                oid_norm = self._normalize_order_id(row.get("orderId"), strict_int=strict_int)
                if strict_int:
                    key = (ts_ms, safe_nonneg_int(oid_norm, default=0) or 0)
                else:
                    key = (ts_ms, str(oid_norm or ""))
                if latest_key is None or key > latest_key:
                    latest_key = key
                    latest_price = px_val
        return latest_price

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
        """Read fresh normalized filled-order rows for ``symbol`` from ``cache_order_filled_<venue>.json``."""
        return CacheTradeManager.read_venue_fills_snapshot(
            symbol,
            cache_file,
            max_stale_sec=max_stale_sec,
            match_fn=match_fn,
            candidate_keys=candidate_keys,
            normalize_fn=normalize_fn,
        )

    @classmethod
    def invalidate_venue_fills_snapshot(
        cls,
        symbol: str,
        cache_file: str,
        *,
        match_fn=None,
        candidate_keys=None,
    ) -> None:
        """Expire ``fetchtime`` for ``symbol`` in ``cache_order_filled_<venue>.json`` after order execution."""
        CacheTradeManager.invalidate_venue_fills_snapshot(
            symbol,
            cache_file,
            match_fn=match_fn,
            candidate_keys=candidate_keys,
        )


class CacheOrderOpenManager(CacheAccountManagerBase):
    """Cross-venue live open-order cache (Level 3) using replace-all snapshot semantics."""

    _is_open_order_account_cache = True
    DEFAULT_FILENAME = CACHE_BINANCE_ORDER_OPEN_FILENAME
    VENUE_FILENAMES: dict[str, str] = {
        "binance": CACHE_BINANCE_ORDER_OPEN_FILENAME,
        "kraken": CACHE_KRAKEN_ORDER_OPEN_FILENAME,
        "hyperliquid": CACHE_HYPERLIQUID_ORDER_OPEN_FILENAME,
        "t212": CACHE_T212_ORDER_OPEN_FILENAME,
    }

    def __init__(
        self,
        sync_ts,
        symbols,
        filename,
        api_client=None,
        read_only=False,
        *,
        data_sync_mode: Optional[str] = None,
        venue: Optional[str] = None,
    ):
        self.venue = self._normalize_venue_arg(venue=venue, filename=filename, symbols=symbols)
        self.verified_ts: float = 0.0
        self.symbol_verified_ts: dict[str, float] = {}
        super().__init__(
            sync_ts,
            symbols,
            filename,
            append_mode=False,
            api_client=api_client,
            append_persist=False,
            read_only=read_only,
            data_sync_mode=data_sync_mode,
        )

    @classmethod
    def normalize_open_order_row(
        cls,
        raw: Any,
        *,
        default_status: str = "NEW",
    ) -> Optional[dict[str, Any]]:
        """Normalize any venue open-order dict into the compact canonical open-order cache schema."""
        if not isinstance(raw, dict):
            return None
        oid = clean_order_id(extract_order_id(raw, default=None))
        if not oid:
            return None
        raw_status = clean_upper_scalar(raw.get("status")) or ""
        if raw_status in _TERMINAL_NON_OPEN_STATUSES:
            return None
        side = extract_order_side(raw) or normalize_side(raw.get("side"))
        if side not in ("BUY", "SELL"):
            return None
        orig_q = safe_pos_float(
            extract_orig_qty(raw, default=None), default=None
        ) or safe_pos_float(extract_order_qty(raw, default=None), default=None)
        if orig_q is None:
            return None
        has_exec_key = any(k in raw for k in _OPEN_ORDER_EXEC_KEYS)
        exec_q = (
            safe_nonneg_float(extract_executed_qty(raw, default=0.0), default=0.0) or 0.0
            if has_exec_key
            else 0.0
        )
        if exec_q > orig_q + 1e-9:
            return None
        exec_q = min(exec_q, orig_q)
        price = safe_nonneg_float(
            first_pos_float(raw.get("price"), raw.get("limitPrice"), raw.get("px"), default=0.0),
            default=0.0,
        ) or 0.0
        if exec_q > 0.0 and exec_q < orig_q:
            norm_status = "PARTIALLY_FILLED"
        elif raw_status and is_open_working_status(raw_status, allow_empty=False):
            norm_status = raw_status
        elif raw_status == "PARTIALLY_FILLED":
            norm_status = "PARTIALLY_FILLED"
        else:
            norm_status = clean_upper_scalar(default_status) or "NEW"
        raw_type = clean_upper_scalar(raw.get("type") or raw.get("order_type") or raw.get("orderType"))
        otype = raw_type if raw_type not in (None, "BUY", "SELL", "B", "S") else "LIMIT"
        row: dict[str, Any] = {
            "orderId": oid,
            "side": side,
            "type": otype,
            "price": price,
            "origQty": round(orig_q, 8),
            "executedQty": round(exec_q, 8),
            "status": norm_status,
        }
        if cid := extract_client_order_id(raw, default=""):
            row["clientOrderId"] = cid
        ts_ms = extract_timestamp_ms(raw, default=None)
        if ts_ms is not None and ts_ms > 0:
            row["timestamp"] = math.trunc(ts_ms)
        return row

    def _is_valid_item(self, item) -> bool:
        if not isinstance(item, dict):
            return False
        oid = clean_order_id(item.get("orderId"))
        if not oid:
            return False
        side = str(item.get("side") or "").upper()
        if side not in ("BUY", "SELL"):
            return False
        price = safe_nonneg_float(item.get("price"), default=None)
        orig_q = safe_pos_float(item.get("origQty"), default=None)
        exec_q = safe_nonneg_float(item.get("executedQty", 0.0), default=None)
        status_u = clean_upper_scalar(item.get("status")) or ""
        if (
            price is None
            or orig_q is None
            or exec_q is None
            or exec_q > orig_q + 1e-9
            or not status_u
            or status_u in _TERMINAL_NON_OPEN_STATUSES
        ):
            return False
        if "clientOrderId" in item and item["clientOrderId"] is not None:
            if isinstance(item["clientOrderId"], bool) or clean_order_id(item["clientOrderId"]) is None:
                return False
        if "timestamp" in item and item["timestamp"] is not None:
            if isinstance(item["timestamp"], bool) or strict_ascii_int(item["timestamp"], min_value=1) is None:
                return False
        return True

    def _ensure_account_version(self, required_version: str) -> None:
        self._reload_snapshot_from_disk()

    def _load_snapshot_state(self):
        with self._generation_lock():
            self._load_open_state_generation_locked()

    def _load_open_state_generation_locked(self) -> bool:
        if not os.path.exists(self.filename) and not os.path.exists(self._previous_generation_path()):
            return False
        loaded = read_json_dict_recovering(self.filename, repair_files=not self.read_only)
        if not isinstance(loaded, dict):
            return False
        items = loaded.get("items") if isinstance(loaded.get("items"), dict) else {}
        ft = loaded.get("fetchtime") if isinstance(loaded.get("fetchtime"), dict) else {}
        ver_ts = safe_nonneg_float(loaded.get("verified_ts"), default=0.0) or 0.0
        cleaned_items: dict[str, list[dict[str, Any]]] = {}
        for s_key, rows in items.items():
            if clean_scalar_str(s_key) is None or not isinstance(rows, list):
                continue
            cleaned_items[s_key] = [
                norm for r in rows if (norm := self.normalize_open_order_row(r)) is not None
            ]
        ft_clean = {
            str(k): safe_nonneg_int(v, default=0) or 0
            for k, v in ft.items()
            if clean_scalar_str(k) is not None
        }
        sym_ver = self._parse_symbol_verified_ts(loaded, cleaned_items, ft_clean, ver_ts)
        with self.lock:
            self.cache = cleaned_items
            self.fetchtime_per_symbol = ft_clean
            self.verified_ts = ver_ts
            self.symbol_verified_ts = sym_ver
        return True

    @staticmethod
    def _parse_symbol_verified_ts(
        loaded: dict[str, Any],
        items: dict[str, list[dict[str, Any]]],
        ft_map: dict[str, int],
        ver_ts: float,
    ) -> dict[str, float]:
        """Extract per-symbol complete-snapshot verification timestamps (distinct from omitted symbols)."""
        if isinstance(loaded.get("symbol_verified_ts"), dict):
            out: dict[str, float] = {}
            for k, v in loaded["symbol_verified_ts"].items():
                if clean_scalar_str(k) is not None and (v_f := safe_pos_float(v, default=None)) is not None:
                    out[str(k)] = round(v_f, 3)
            return out
        # Legacy snapshot compatibility (no explicit ``symbol_verified_ts`` field):
        # bare ``fetchtime`` alone is NOT proof of a complete open-order snapshot because single-order WS
        # deltas advance ``fetchtime``. Require an explicit venue-wide ``verified_ts > 0`` and cap any
        # per-symbol timestamp at ``verified_ts`` for symbols explicitly listed in ``items``.
        if ver_ts <= 0.0:
            return {}
        inferred: dict[str, float] = {}
        for sym_k in items.keys():
            ft_ms = ft_map.get(sym_k, 0)
            sym_ts = min(ver_ts, ft_ms / 1000.0) if ft_ms > 0 else ver_ts
            inferred[sym_k] = round(sym_ts, 3)
        return inferred

    def _reload_snapshot_from_disk(self):
        try:
            with self._generation_lock():
                self._load_open_state_generation_locked()
        except Exception as exc:
            print(f"[{self.cls_name}][Error] reload {self.filename}: {exc}")

    def _save_state_to_file_impl(self) -> bool:
        def _save_locked() -> bool:
            with self.lock:
                payload = {
                    "items": {k: [dict(r) for r in v] for k, v in self.cache.items()},
                    "fetchtime": dict(self.fetchtime_per_symbol),
                    "symbol_verified_ts": {
                        k: round(v, 3) for k, v in sorted(self.symbol_verified_ts.items()) if v > 0.0
                    },
                    "verified_ts": round(self.verified_ts, 3) if self.verified_ts > 0.0 else None,
                }
            save_json_state(self.filename, payload, indent=1, skip_if_unchanged=True)
            return True

        return self._with_generation_lock_bool(_save_locked, "OpenOrder snapshot")

    def _fetch_and_apply_remote_items(self, symbol: str, startTime) -> list:
        """Always apply replace-all snapshot semantics, including an empty open-order list."""
        new_items = self.get_remote_items(symbol=symbol, startTime=startTime)
        if new_items is None:
            raise RuntimeError(
                f"{self.cls_name} returned no synchronization result for {symbol}"
            )
        self._persist_items(symbol, new_items)
        return new_items

    def get_remote_items(self, symbol, startTime) -> list[dict[str, Any]]:
        prov_cls = MarketDataProvider.resolve_venue_provider_class(self.venue)
        if prov_cls is None:
            return []
        raw_orders = prov_cls().open_orders(symbol)
        if not isinstance(raw_orders, list):
            raise RuntimeError(f"{self.cls_name}: invalid open_orders payload for {symbol}")
        return [
            norm for row in raw_orders if (norm := self.normalize_open_order_row(row)) is not None
        ]

    def replace_symbol_open_orders(
        self,
        symbol: str,
        orders: Sequence[dict[str, Any]],
        *,
        now_ms: Optional[int] = None,
    ) -> bool:
        """Atomically replace the open-order list for ``symbol`` in memory (deduplicated by ``orderId``).

        Advances only ``symbol_verified_ts[symbol]`` -- never advances the venue-wide ``verified_ts``.
        """
        eff_now_ms = resolve_now_ms(now_ms)
        dedup: dict[str, dict[str, Any]] = {}
        for row in orders or ():
            if (norm := self.normalize_open_order_row(row)) is not None:
                dedup[norm["orderId"]] = norm
        new_bucket = list(dedup.values())
        with self.lock:
            changed = self.cache.get(symbol) != new_bucket
            self.cache[symbol] = new_bucket
            self.fetchtime_per_symbol[symbol] = eff_now_ms
            self.symbol_verified_ts[symbol] = round(eff_now_ms / 1000.0, 3)
        return changed

    def _persist_items(self, symbol, new_items):
        self.replace_symbol_open_orders(symbol, new_items or ())

    def apply_ws_order_update(
        self,
        symbol: str,
        action: str,
        order_item: dict[str, Any],
        *,
        now_ms: Optional[int] = None,
    ) -> bool:
        """Apply one real-time WS open-order transition (``'upsert'`` or ``'remove'``).

        Single-order WS deltas are NOT complete book snapshots, so they must NEVER advance
        either venue-wide ``verified_ts`` or per-symbol ``symbol_verified_ts``.
        """
        if not isinstance(order_item, dict) or clean_scalar_str(symbol) is None:
            return False
        eff_now_ms = resolve_now_ms(now_ms)
        oid = clean_order_id(order_item.get("orderId"))
        if not oid:
            return False
        act = (clean_scalar_str(action) or "").lower()
        with self.lock:
            bucket = list(self.cache.get(symbol) or ())
            if act == "remove":
                filtered = [r for r in bucket if clean_order_id(r.get("orderId")) != oid]
                if len(filtered) == len(bucket):
                    return False
                self.cache[symbol] = filtered
                self.fetchtime_per_symbol[symbol] = eff_now_ms
                return True
            if act == "upsert":
                norm = self.normalize_open_order_row(order_item)
                if norm is None:
                    return False
                updated = False
                for idx, existing in enumerate(bucket):
                    if clean_order_id(existing.get("orderId")) == norm["orderId"]:
                        if existing == norm:
                            return False
                        bucket[idx] = norm
                        updated = True
                        break
                if not updated:
                    bucket.append(norm)
                self.cache[symbol] = bucket
                self.fetchtime_per_symbol[symbol] = eff_now_ms
                return True
        return False

    @classmethod
    def replace_venue_open_orders_snapshot(
        cls,
        cache_file: str,
        orders_by_symbol: dict[str, Sequence[dict[str, Any]]],
        *,
        now_ms: Optional[int] = None,
        verified_ts: Optional[float] = None,
        indent: Optional[int] = 1,
    ) -> dict[str, Any]:
        """Atomically replace per-symbol open orders in ``cache_order_open_<venue>.json`` under ``.generation.lock``.

        Each symbol present in ``orders_by_symbol`` (including a verified empty list ``[]``) receives an
        updated ``symbol_verified_ts[sym]`` entry. Omitted symbols retain their prior state and verification
        timestamp (distinct from verified empty). Venue-wide ``verified_ts`` is updated only when explicitly
        provided by a full-venue snapshot caller.
        """
        target_path = str(cache_file)
        eff_now_ms = resolve_now_ms(now_ms)
        eff_ver_ts = safe_pos_float(verified_ts, default=None)
        sym_snap_ts = round(eff_ver_ts if eff_ver_ts is not None else (eff_now_ms / 1000.0), 3)
        with generation_transaction(target_path) as target_path:
            loaded = read_json_dict_recovering(target_path, repair_files=True)
            if not isinstance(loaded, dict):
                loaded = {}
            items = loaded.get("items") if isinstance(loaded.get("items"), dict) else {}
            ft = loaded.get("fetchtime") if isinstance(loaded.get("fetchtime"), dict) else {}
            prev_ver = safe_nonneg_float(loaded.get("verified_ts"), default=0.0) or 0.0
            ft_prev = {
                str(k): safe_nonneg_int(v, default=0) or 0
                for k, v in ft.items()
                if clean_scalar_str(k) is not None
            }
            sym_ver = cls._parse_symbol_verified_ts(loaded, items, ft_prev, prev_ver)
            for sym_key, raw_list in (orders_by_symbol or {}).items():
                if clean_scalar_str(sym_key) is None or not isinstance(raw_list, (list, tuple)):
                    continue
                dedup: dict[str, dict[str, Any]] = {}
                for row in raw_list:
                    if (norm := cls.normalize_open_order_row(row)) is not None:
                        dedup[norm["orderId"]] = norm
                # Remove case-variant duplicate bucket keys before writing the canonical symbol key
                for stale_k in [k for k in list(items.keys()) if k != sym_key and str(k).upper() == str(sym_key).upper()]:
                    items.pop(stale_k, None)
                    ft.pop(stale_k, None)
                    sym_ver.pop(stale_k, None)
                items[sym_key] = list(dedup.values())
                ft[sym_key] = eff_now_ms
                sym_ver[sym_key] = sym_snap_ts
            out_obj = {
                "items": items,
                "fetchtime": ft,
                "symbol_verified_ts": sym_ver,
                "verified_ts": round(eff_ver_ts, 3) if eff_ver_ts is not None else (round(prev_ver, 3) if prev_ver > 0 else None),
            }
            save_json_state(target_path, out_obj, indent=indent, skip_if_unchanged=True)
            return out_obj

    @classmethod
    def read_venue_open_orders_snapshot_detailed(
        cls,
        cache_file: str,
        *,
        read_json_fn=None,
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int], float, dict[str, float]]:
        """Passively read one venue's open-order snapshot including per-symbol verified complete-snapshot timestamps."""
        resolved = str(resolve_cachedb_filepath(str(cache_file)))
        if read_json_fn is not None:
            data = read_json_fn(resolved)
        else:
            if not os.path.isfile(resolved):
                return {}, {}, 0.0, {}
            data = read_json_dict_recovering(resolved, repair_files=False)
        if not isinstance(data, dict):
            return {}, {}, 0.0, {}
        raw_items = data.get("items") if isinstance(data.get("items"), dict) else {}
        raw_ft = data.get("fetchtime") if isinstance(data.get("fetchtime"), dict) else {}
        items_out: dict[str, list[dict[str, Any]]] = {}
        for sym_k, rows in raw_items.items():
            if clean_scalar_str(sym_k) is None or not isinstance(rows, list):
                continue
            items_out[sym_k] = [dict(r) for r in rows if isinstance(r, dict)]
        ft_out = {
            str(k): safe_nonneg_int(v, default=0) or 0
            for k, v in raw_ft.items()
            if clean_scalar_str(k) is not None
        }
        ver_ts = safe_nonneg_float(data.get("verified_ts"), default=0.0) or 0.0
        sym_ver = cls._parse_symbol_verified_ts(data, items_out, ft_out, ver_ts)
        return items_out, ft_out, ver_ts, sym_ver

    @classmethod
    def read_venue_open_orders_snapshot(
        cls,
        cache_file: str,
        *,
        read_json_fn=None,
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int], float]:
        """Passively read one venue's open-order snapshot without mutating disk."""
        items_out, ft_out, ver_ts, _ = cls.read_venue_open_orders_snapshot_detailed(
            cache_file, read_json_fn=read_json_fn
        )
        return items_out, ft_out, ver_ts

    @classmethod
    def read_all_venues_open_orders(
        cls,
        cachedb_dir: str,
        *,
        read_json_fn=None,
    ) -> tuple[list[dict[str, Any]], set[str], dict[str, float]]:
        """Passively read all venues' ``cache_order_open_<venue>.json`` files."""
        flat: list[dict[str, Any]] = []
        covered: set[str] = set()
        verified_by_venue: dict[str, float] = {}
        if not cachedb_dir:
            return flat, covered, verified_by_venue
        for venue_key, fname in cls.VENUE_FILENAMES.items():
            cand_path = str(resolve_cachedb_filepath(os.path.join(cachedb_dir, fname)))
            items, ft, ver_ts = cls.read_venue_open_orders_snapshot(cand_path, read_json_fn=read_json_fn)
            if not items and not ft and ver_ts <= 0.0:
                continue
            covered.add(venue_key)
            if ver_ts > 0.0:
                verified_by_venue[venue_key] = ver_ts
            for sym_k, rows in items.items():
                for row in rows:
                    flat.append({"venue": venue_key, "symbol": str(row.get("symbol") or sym_k), **row})
        return flat, covered, verified_by_venue


@dataclass(frozen=True, slots=True)
class UnifiedOrderBookEvidence:
    """SSOT read-only open/filled order book snapshot classified in canonical open-first order."""

    resting_open_orders: list[dict[str, Any]]
    open_covered_venues: set[str]
    verified_by_venue: dict[str, float]
    symbol_verified_ts: dict[tuple[str, str], float]
    terminal_order_keys: set[tuple[str, str, str]]
    ambiguous_filled_by_key: dict[tuple[str, str, str], dict[str, Any]]
    open_items_by_venue: dict[str, dict[str, list[dict[str, Any]]]] = field(default_factory=dict)

    @staticmethod
    def proves_symbol_absent(
        symbol_verified_ts: Mapping[Any, float],
        venue: Any,
        symbol_key: str,
        *,
        ref_ts: float,
        grace_sec: float = 0.0,
    ) -> bool:
        """Canonical per-(venue, symbol) absence proof shared across reconciler and active_intents.

        Strictly requires a complete open-order snapshot covering ``(venue, symbol)`` verified at
        least ``grace_sec`` after ``ref_ts`` -- never falls back to venue-wide timestamps when a
        symbol was omitted from a partial snapshot.
        """
        v_norm = str(normalize_venue_name(venue) or venue or "").strip().lower()
        sym_lc = str(symbol_key or "").strip().lower()
        if not v_norm or not sym_lc:
            return False
        raw_ts = symbol_verified_ts.get((v_norm, sym_lc))
        if raw_ts is None:
            raw_ts = symbol_verified_ts.get(f"{v_norm}:{sym_lc}")
        s_ver_ts = safe_pos_float(raw_ts, default=0.0) or 0.0
        return bool(s_ver_ts > 0.0 and (s_ver_ts - ref_ts) >= max(0.0, grace_sec))

    def symbol_verified_ts_for(
        self,
        venue: Any,
        symbol: Any,
        *,
        normalizer: Optional[Callable[[Any], str]] = None,
    ) -> float:
        """Return verified snapshot timestamp for ``(venue, symbol)``, or ``0.0`` if uncovered."""
        v_norm = str(normalize_venue_name(venue) or venue or "").strip().lower()
        sym_lc = (
            str(normalizer(symbol) or "").strip().lower()
            if normalizer is not None
            else str(clean_symbol_key(symbol) or symbol or "").strip().lower()
        )
        if not v_norm or not sym_lc:
            return 0.0
        return safe_pos_float(self.symbol_verified_ts.get((v_norm, sym_lc)), default=0.0) or 0.0

    def proves_absent(
        self,
        venue: Any,
        symbol: Any,
        *,
        ref_ts: float,
        grace_sec: float = 0.0,
        normalizer: Optional[Callable[[Any], str]] = None,
    ) -> bool:
        """Return True iff this snapshot proves ``(venue, symbol)`` was completely verified after ``ref_ts + grace_sec``."""
        sym_lc = (
            str(normalizer(symbol) or "").strip().lower()
            if normalizer is not None
            else str(clean_symbol_key(symbol) or symbol or "").strip().lower()
        )
        return self.proves_symbol_absent(
            self.symbol_verified_ts,
            venue,
            sym_lc,
            ref_ts=ref_ts,
            grace_sec=grace_sec,
        )

    def open_orders_by_venue_sym(self) -> dict[tuple[str, str], list[dict[str, Any]]]:
        """Return ``{(venue, SYMBOL_UPPER): [order_rows]}`` preserving explicit empty covered symbols."""
        by_vs: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for venue_key, sym_map in self.open_items_by_venue.items():
            for sym_k, rows in sym_map.items():
                if not isinstance(rows, list):
                    continue
                by_vs[(venue_key, str(sym_k).strip().upper())] = [
                    {"venue": venue_key, "symbol": str(r.get("symbol") or sym_k), **r}
                    for r in rows
                    if isinstance(r, dict)
                ]
        return by_vs


def load_unified_order_book_evidence(
    cachedb_dir: str,
    *,
    read_json_fn=None,
    symbol_normalizer: Optional[Callable[[Any], str]] = None,
    include_filled: bool = True,
) -> UnifiedOrderBookEvidence:
    """Read open and filled order caches across all venues in open-first order and classify fills via order-ledger SSOT."""
    if not cachedb_dir:
        return UnifiedOrderBookEvidence([], set(), {}, {}, set(), {}, {})

    def _norm_sym(raw_sym: Any) -> str:
        if symbol_normalizer is not None:
            return str(symbol_normalizer(raw_sym) or "").strip().lower()
        return str(clean_symbol_key(raw_sym) or raw_sym or "").strip().lower()

    def _key(v_k: str, s_raw: Any, oid_raw: Any) -> tuple[str, str, str]:
        v_norm = str(normalize_venue_name(v_k) or v_k or "").strip().lower()
        return (v_norm, _norm_sym(s_raw), clean_order_id(oid_raw) or "")

    raw_open_rows: list[tuple[tuple[str, str, str], dict[str, Any]]] = []
    open_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    open_covered_venues: set[str] = set()
    verified_by_venue: dict[str, float] = {}
    symbol_verified_ts: dict[tuple[str, str], float] = {}
    terminal_order_keys: set[tuple[str, str, str]] = set()
    ambiguous_filled_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    filled_open_by_venue: dict[str, list[dict[str, Any]]] = {}
    open_items_by_venue: dict[str, dict[str, list[dict[str, Any]]]] = {}

    # 1. Read open-order snapshots FIRST so fill classification knows resting presence & origQty.
    for venue_key, open_fname in CacheOrderOpenManager.VENUE_FILENAMES.items():
        oo_path = str(resolve_cachedb_filepath(os.path.join(cachedb_dir, open_fname)))
        items, ft, ver_ts, sym_ver = CacheOrderOpenManager.read_venue_open_orders_snapshot_detailed(
            oo_path, read_json_fn=read_json_fn
        )
        if not items and not ft and ver_ts <= 0.0 and not sym_ver:
            continue
        open_covered_venues.add(venue_key)
        open_items_by_venue[venue_key] = items
        if ver_ts > 0.0:
            verified_by_venue[venue_key] = ver_ts
        for sym_k, s_ts in sym_ver.items():
            sym_lc = _norm_sym(sym_k)
            if sym_lc and s_ts > 0.0:
                symbol_verified_ts[(venue_key, sym_lc)] = max(symbol_verified_ts.get((venue_key, sym_lc), 0.0), s_ts)
        for k_sym, rows in items.items():
            for oo in rows:
                if not isinstance(oo, dict):
                    continue
                oid = clean_order_id(extract_order_id(oo, default=None)) or ""
                oo_sym = str(oo.get("symbol") or k_sym)
                key_t = _key(venue_key, oo_sym, oid)
                row_full = {"venue": venue_key, "symbol": oo_sym, **oo}
                if oid:
                    open_by_key[key_t] = row_full
                raw_open_rows.append((key_t, row_full))

    # 2. Classify filled-order cache rows via canonical order-ledger SSOT rules (when requested).
    if include_filled:
        for venue_key, filled_fname in CacheOrderFilledManager.VENUE_FILENAMES.items():
            oc_path = str(resolve_cachedb_filepath(os.path.join(cachedb_dir, filled_fname)))
            items, ft = CacheOrderFilledManager.read_venue_filled_orders_snapshot(
                oc_path, venue=venue_key, read_json_fn=read_json_fn
            )
            if not items and not ft:
                continue
            for k_sym, sym_orders in items.items():
                latest_by_oid: dict[str, dict[str, Any]] = {}
                for oo in sym_orders:
                    oid = clean_order_id(extract_order_id(oo, default=None)) or ""
                    if oid:
                        latest_by_oid[oid] = oo
                for oid, oo in latest_by_oid.items():
                    oo_sym = str(oo.get("symbol") or k_sym)
                    key_t = _key(venue_key, oo_sym, oid)
                    resting_row = open_by_key.get(key_t)
                    req_qty = (
                        safe_pos_float(extract_orig_qty(resting_row, default=None), default=None)
                        if resting_row is not None
                        else first_pos_float(oo.get("origQty"), oo.get("orig_qty"), oo.get("requested_qty"), default=None)
                    )
                    oo_st = oo.get("status")
                    oo_st_up = clean_upper_scalar(oo_st) or ""
                    fill_state = classify_fill_evidence_state(
                        oo, req_qty=req_qty, resting_now=(resting_row is not None)
                    )
                    if fill_state in TERMINAL_LEDGER_STATES:
                        if fill_state == OrderLedgerState.FILLED.value and req_qty is None and resting_row is None:
                            ambiguous_filled_by_key[key_t] = oo
                        else:
                            terminal_order_keys.add(key_t)
                        continue
                    if not (is_open_working_status(oo_st, allow_empty=False) or oo_st_up == "PARTIALLY_FILLED"):
                        continue
                    filled_open_by_venue.setdefault(venue_key, []).append(
                        {"venue": venue_key, "symbol": oo_sym, **oo}
                    )

    resting_out: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for key_t, row_full in raw_open_rows:
        oid = key_t[2]
        if oid and key_t in terminal_order_keys:
            continue
        if not oid or key_t not in seen_keys:
            if oid:
                seen_keys.add(key_t)
            resting_out.append(row_full)

    for venue_key, oc_orders in filled_open_by_venue.items():
        if venue_key in open_covered_venues:
            continue
        for oo in oc_orders:
            oid = clean_order_id(extract_order_id(oo, default=None)) or ""
            key_t = _key(venue_key, oo.get("symbol"), oid)
            if not oid or key_t not in seen_keys:
                if oid:
                    seen_keys.add(key_t)
                resting_out.append(oo)

    return UnifiedOrderBookEvidence(
        resting_open_orders=resting_out,
        open_covered_venues=open_covered_venues,
        verified_by_venue=verified_by_venue,
        symbol_verified_ts=symbol_verified_ts,
        terminal_order_keys=terminal_order_keys,
        ambiguous_filled_by_key=ambiguous_filled_by_key,
        open_items_by_venue=open_items_by_venue,
    )


CacheOrderOpenManager.load_unified_evidence = staticmethod(load_unified_order_book_evidence)


# =============================================================================
# 3B. ACCOUNT BALANCE SNAPSHOT CACHE: CacheBalanceManager
# =============================================================================


class CacheBalanceManager(CacheAccountManagerBase):
    """Cross-venue raw spot account balance cache (Level 3) backed by ``cache_balance_<venue>.json``."""

    _is_balance_account_cache = True
    DEFAULT_FILENAME = CACHE_BINANCE_BALANCE_FILENAME
    VENUE_FILENAMES: dict[str, str] = {
        "binance": CACHE_BINANCE_BALANCE_FILENAME,
        "kraken": CACHE_KRAKEN_BALANCE_FILENAME,
        "hyperliquid": CACHE_HYPERLIQUID_BALANCE_FILENAME,
        "t212": CACHE_T212_BALANCE_FILENAME,
    }

    def __init__(
        self,
        sync_ts,
        symbols,
        filename,
        api_client=None,
        read_only=False,
        *,
        data_sync_mode: Optional[str] = None,
        venue: Optional[str] = None,
    ):
        self.venue = self._normalize_venue_arg(venue=venue, filename=filename, symbols=symbols)
        self.verified_ts: float = 0.0
        self.breakdown: dict[str, dict[str, float]] = {}
        super().__init__(
            sync_ts,
            symbols,
            filename,
            append_mode=False,
            api_client=api_client,
            append_persist=False,
            read_only=read_only,
            data_sync_mode=data_sync_mode,
        )

    @classmethod
    def normalize_balance_entry(
        cls,
        raw: Any,
        *,
        default_ts: Optional[float] = None,
    ) -> Optional[dict[str, Any]]:
        """Normalize a raw balance tuple/mapping into ``{"free", "locked", "ts"}``."""
        free_v: Optional[float] = None
        locked_v: Optional[float] = None
        entry_ts: Optional[float] = None
        if isinstance(raw, (tuple, list)) and len(raw) >= 1:
            free_v = safe_nonneg_float(raw[0], default=None)
            if len(raw) >= 2 and raw[1] is not None:
                locked_v = safe_nonneg_float(raw[1], default=None)
            if len(raw) >= 3 and raw[2] is not None:
                entry_ts = normalize_epoch_sec(raw[2], default=None)
        elif isinstance(raw, Mapping):
            free_v = safe_nonneg_float(raw.get("free"), default=None)
            if raw.get("locked") is not None:
                locked_v = safe_nonneg_float(raw.get("locked"), default=None)
            entry_ts = normalize_epoch_sec(raw.get("ts"), default=None)
        elif isinstance(raw, (int, float)) and not isinstance(raw, bool):
            free_v = safe_nonneg_float(raw, default=None)
        if free_v is None:
            return None
        eff_ts = entry_ts or normalize_epoch_sec(default_ts, default=None)
        row: dict[str, Any] = {
            "free": round(free_v, 8),
            "locked": round(locked_v, 8) if locked_v is not None else None,
        }
        if eff_ts is not None and eff_ts > 0.0:
            row["ts"] = round(eff_ts, 3)
        return row

    def _is_valid_item(self, item) -> bool:
        return self.normalize_balance_entry(item) is not None

    def _ensure_account_version(self, required_version: str) -> None:
        self._load_snapshot_state()

    def _load_snapshot_state(self) -> None:
        with self._generation_lock():
            if not os.path.exists(self.filename) and not os.path.exists(self._previous_generation_path()):
                return
            loaded = read_json_dict_recovering(self.filename, repair_files=not self.read_only)
            if not isinstance(loaded, dict):
                return
            doc_ts = normalize_epoch_sec(loaded.get("verified_ts") or loaded.get("ts"), default=None)
            raw_items = loaded.get("items") if isinstance(loaded.get("items"), dict) else {}
            cleaned: dict[str, list[dict[str, Any]]] = {}
            for a_k, a_val in raw_items.items():
                a_clean = canonical_balance_asset_key(a_k, venue=self.venue)
                if not a_clean:
                    continue
                if (norm := self.normalize_balance_entry(a_val, default_ts=doc_ts)) is not None:
                    cleaned[a_clean] = [norm]
            raw_bd = loaded.get("breakdown") if isinstance(loaded.get("breakdown"), dict) else {}
            clean_bd: dict[str, dict[str, float]] = {}
            for q_k, q_map in raw_bd.items():
                q_u = clean_upper_scalar(q_k)
                if not q_u or not isinstance(q_map, Mapping):
                    continue
                bd_vals = {
                    str(bk): val_f
                    for bk, bv in q_map.items()
                    if clean_scalar_str(bk) is not None and (val_f := safe_nonneg_float(bv, default=None)) is not None
                }
                if bd_vals:
                    clean_bd[q_u] = bd_vals
            with self.lock:
                self.cache = cleaned
                self.verified_ts = safe_nonneg_float(loaded.get("verified_ts"), default=0.0) or 0.0
                self.breakdown = clean_bd

    def get_remote_items(self, symbol, startTime) -> list[dict[str, Any]]:
        return []

    def get_balance(self, asset: str) -> Optional[dict[str, Any]]:
        """Return latest cached ``{"free", "locked", "ts"}`` dict for ``asset``."""
        a_k = canonical_balance_asset_key(asset, venue=self.venue)
        if not a_k:
            return None
        self.resync_mem_file()
        with self.lock:
            rows = self.cache.get(a_k)
            return dict(rows[0]) if rows else None

    def get_all_balances(self) -> dict[str, dict[str, Any]]:
        """Return shallow copy of all cached ``asset -> {"free", "locked", "ts"}`` entries."""
        self.resync_mem_file()
        with self.lock:
            return {a_k: dict(rows[0]) for a_k, rows in self.cache.items() if rows}

    def get_breakdown(self, quote_asset: str = "USD") -> dict[str, float]:
        """Return quote currency breakdown map (e.g. Kraken USD/USDC/USDG) if cached."""
        q_u = clean_upper_scalar(quote_asset) or "USD"
        self.resync_mem_file()
        with self.lock:
            return dict(self.breakdown.get(q_u) or {})

    @classmethod
    def replace_venue_balances_snapshot(
        cls,
        cache_file: str,
        balances_by_asset: Mapping[str, Any],
        *,
        venue: Optional[str] = None,
        now_s: Optional[float] = None,
        verified_ts: Optional[float] = None,
        breakdown: Optional[Mapping[str, Mapping[str, Any]]] = None,
        indent: Optional[int] = 1,
    ) -> dict[str, Any]:
        """Atomically persist one venue's raw account balances into ``cache_balance_<venue>.json`` under ``.generation.lock``.

        When ``verified_ts`` is provided (full-venue snapshot from ``all_free_and_locked_balances``), ``items`` is
        replaced wholesale so sold-out assets cannot linger as stale ghosts. When ``verified_ts`` is ``None``
        (partial per-asset refresh), provided assets merge cleanly into existing ``items``.
        """
        target_path = str(cache_file)
        eff_now_s = resolve_now_ts(now_s)
        eff_ver_ts = safe_pos_float(verified_ts, default=None)
        v_norm = normalize_venue_name(venue) or cls._normalize_venue_arg(venue=venue, filename=target_path)
        with generation_transaction(target_path) as target_path:
            loaded = read_json_dict_recovering(target_path, repair_files=True)
            if not isinstance(loaded, dict):
                loaded = {}
            prev_items = loaded.get("items") if isinstance(loaded.get("items"), dict) else {}
            prev_ver = safe_nonneg_float(loaded.get("verified_ts"), default=0.0) or 0.0
            prev_bd = loaded.get("breakdown") if isinstance(loaded.get("breakdown"), dict) else {}

            items_out: dict[str, dict[str, Any]] = {}
            if eff_ver_ts is None:
                for pk, pv in prev_items.items():
                    if (pk_c := canonical_balance_asset_key(pk, venue=v_norm)) and (norm_p := cls.normalize_balance_entry(pv)) is not None:
                        items_out[pk_c] = norm_p
            for asset_k, raw_entry in (balances_by_asset or {}).items():
                if not (ak_c := canonical_balance_asset_key(asset_k, venue=v_norm)):
                    continue
                if (norm := cls.normalize_balance_entry(raw_entry, default_ts=eff_now_s)) is not None:
                    items_out[ak_c] = norm

            bd_out: dict[str, dict[str, float]] = {}
            bd_src = breakdown if isinstance(breakdown, Mapping) else prev_bd
            for q_k, q_map in bd_src.items():
                q_u = clean_upper_scalar(q_k)
                if not q_u or not isinstance(q_map, Mapping):
                    continue
                clean_q = {
                    str(bk): round(val_f, 4)
                    for bk, bv in q_map.items()
                    if clean_scalar_str(bk) is not None and (val_f := safe_nonneg_float(bv, default=None)) is not None
                }
                if clean_q:
                    bd_out[q_u] = clean_q

            out_obj: dict[str, Any] = {
                "venue": v_norm,
                "ts": round(eff_now_s, 3),
                "verified_ts": round(eff_ver_ts, 3) if eff_ver_ts is not None else (round(prev_ver, 3) if prev_ver > 0 else None),
                "items": items_out,
            }
            if bd_out:
                out_obj["breakdown"] = bd_out
            save_json_state(target_path, out_obj, indent=indent, skip_if_unchanged=True)
            return out_obj

    @classmethod
    def read_venue_balances_snapshot(
        cls,
        cache_file: str,
        *,
        read_json_fn=None,
    ) -> tuple[dict[str, dict[str, Any]], float, dict[str, dict[str, float]]]:
        """Passively read one venue's ``cache_balance_<venue>.json`` snapshot."""
        resolved = str(resolve_cachedb_filepath(str(cache_file)))
        if read_json_fn is not None:
            data = read_json_fn(resolved)
        else:
            if not (os.path.isfile(resolved) or os.path.isfile(f"{resolved}.previous")):
                return {}, 0.0, {}
            data = read_json_dict_recovering(resolved, repair_files=False)
        if not isinstance(data, dict):
            return {}, 0.0, {}
        doc_ts = normalize_epoch_sec(data.get("verified_ts") or data.get("ts"), default=None)
        v_norm = normalize_venue_name(data.get("venue")) or cls._normalize_venue_arg(filename=resolved)
        raw_items = data.get("items") if isinstance(data.get("items"), dict) else {}
        items_out: dict[str, dict[str, Any]] = {}
        for a_k, a_val in raw_items.items():
            if not (ak_c := canonical_balance_asset_key(a_k, venue=v_norm)):
                continue
            if (norm := cls.normalize_balance_entry(a_val, default_ts=doc_ts)) is not None:
                items_out[ak_c] = norm
        ver_ts = safe_nonneg_float(data.get("verified_ts"), default=0.0) or 0.0
        raw_bd = data.get("breakdown") if isinstance(data.get("breakdown"), dict) else {}
        bd_out: dict[str, dict[str, float]] = {}
        for q_k, q_map in raw_bd.items():
            q_u = clean_upper_scalar(q_k)
            if not q_u or not isinstance(q_map, Mapping):
                continue
            clean_q = {
                str(bk): val_f
                for bk, bv in q_map.items()
                if clean_scalar_str(bk) is not None and (val_f := safe_nonneg_float(bv, default=None)) is not None
            }
            if clean_q:
                bd_out[q_u] = clean_q
        return items_out, ver_ts, bd_out


# =============================================================================
# 4. PORTFOLIO NAV ARCHIVE: CacheAssetValueManager
# =============================================================================


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
        av_ts_s = first_epoch_sec(
            extract_timestamp_s(rec),
            extract_timestamp_s(av_inner),
            rec.get("t"),
            av_inner.get("t"),
            av_inner.get("datetime_local"),
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

    _DICT_SEC_FETCHTIMES = True

    def get_remote_items(self, symbol, startTime):
        try:
            total_usdc = self.api_client.get_total_assets_value_usdc(
                use_cache=False, raise_errors=True
            )
        except Exception as e:
            print(f"[{self.cls_name}][Error] Cannot query the total value: {e}")
            return []

        total_val = safe_pos_float(total_usdc, default=None)
        if total_val is None:
            print(f"[{self.cls_name}][Error] Invalid total value: {total_usdc}")
            return []

        now_sec = math.trunc(_now_time())
        snapshot = {
            "timestamp": now_sec,
            "datetime_local": datetime.now().isoformat(timespec="seconds"),
            "total_value_usdc": round(total_val, 8),
        }
        venues_usd: dict[str, float] = {}
        try:
            cache_root = cachedb_root_for_file(os.path.abspath(str(self.filename or "")))
            bal_caches = load_venue_balance_caches(cache_root, use_cache=True)
            non_primary_venues = {v for v in CacheBalanceManager.VENUE_FILENAMES if not is_primary_stream_venue(v)}
            vb_data = (
                {}
                if non_primary_venues.issubset(set(bal_caches))
                else load_venue_balances_state_dict(cache_root, use_cache=True)
            )
            if bal_caches or (isinstance(vb_data, dict) and vb_data):
                live_cp_data = None
                cp_default_filename = CACHE_CURRENT_PRICE_FILENAME
                for cp_cand in (
                    os.path.join(cache_root, "prices", cp_default_filename),
                    os.path.join(cache_root, cp_default_filename),
                ):
                    live_cp_data = read_json_dict_cached(cp_cand)
                    if isinstance(live_cp_data, dict) and live_cp_data:
                        break
                nav = compute_venue_nav_breakdown(
                    venue_balance_caches=bal_caches,
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
    lambda: tuple(
        (fname, v_key) for v_key, fname in CacheOrderFilledManager.VENUE_FILENAMES.items()
    )
)
CacheAccountManagerBase.venue_open_order_cache_contracts = staticmethod(
    lambda: tuple(
        (fname, v_key) for v_key, fname in CacheOrderOpenManager.VENUE_FILENAMES.items()
    )
)


# =============================================================================
# 5. ACCOUNT READER SYNCHRONIZATION & DAEMON LIFECYCLE
# =============================================================================

_account_cache_getter = None


def bind_account_cache_getter(getter_fn) -> None:
    """Bind the canonical factory cache getter used by account reader and WS helpers."""
    global _account_cache_getter
    _account_cache_getter = getter_fn


def ensure_account_cache_readers(status):
    """Synchronize this process with the canonical filled-order version in a fresh health marker."""
    ach = _cm_override("account_cache_health", account_cache_health)
    getter = _cm_override("get_cache_manager", _account_cache_getter or _bapi_ws_bridge._default_cache_getter)
    try:
        order_version = status.order_cache_version
        if not order_version:
            raise ValueError("health marker has no durable filled-order cache version")
        getter("BinanceOrderFilled").ensure_persisted_version(order_version)
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
    getter = cache_getter or _cm_override("get_cache_manager", _account_cache_getter or _bapi_ws_bridge._default_cache_getter)
    ach.enable_writer()
    now_ms = math.trunc(_now_time() * 1000)
    max_age_ms = math.trunc(ach.MAX_AGE_SEC * 1000)
    for account_cache_name in ("BinanceOrderFilled", "BinanceTrade"):
        acc_mgr = getter(account_cache_name)
        if isinstance(sync_started_set, set):
            sync_started_set.add(account_cache_name)
        ver, sync_ms = (
            acc_mgr.get_persisted_sync_state()
            if hasattr(acc_mgr, "get_persisted_sync_state")
            else ("", 0)
        )
        if ver and sync_ms > 0 and -5_000 <= (now_ms - sync_ms) <= max_age_ms:
            ach.record_successful_sync(
                account_cache_name, ver, now_ms=sync_ms
            )
    _bapi_ws_bridge.initialize_ws_bridge_once()
    if is_live_network_backfill_enabled() and is_auto_start_websockets_enabled():
        start_kraken_account_ws_bridge()
        start_t212_account_poll_bridge()
        start_venue_open_orders_bridge()
        start_venue_balances_bridge()


def disable_account_cache_daemon_writer() -> None:
    """Mark the account-cache health writer stopping before tearing down daemon threads."""
    ach = _cm_override("account_cache_health", account_cache_health)
    ach.disable_writer()


def request_stop_account_ws_streams() -> None:
    """Signal the account user-data bridges (Binance, Kraken, T212, open orders, and balances) to stop."""
    for bridge in (
        _bapi_ws_bridge._ws_bridge,
        _kraken_ws_bridge,
        _t212_poll_bridge,
        _venue_open_orders_bridge,
        _venue_balances_bridge,
    ):
        if bridge is not None and hasattr(bridge, "request_stop"):
            try:
                bridge.request_stop()
            except Exception:
                pass


def stop_account_ws_streams(shutdown_deadline: float) -> None:
    """Join the account user-data bridges within ``shutdown_deadline``."""
    for bridge in (
        _bapi_ws_bridge._ws_bridge,
        _kraken_ws_bridge,
        _t212_poll_bridge,
        _venue_open_orders_bridge,
        _venue_balances_bridge,
    ):
        if bridge is not None and hasattr(bridge, "stop"):
            try:
                bridge.stop(
                    timeout=min(0.3, max(0.05, shutdown_deadline - _monotonic_time()))
                )
            except Exception:
                pass


# =============================================================================
# 6. VENUE ACCOUNT TRADE BRIDGES (Base, Hyperliquid, Kraken, Trading 212)
# =============================================================================

def _wait_or_sleep(stop_event: Optional[threading.Event], delay_sec: float) -> bool:
    """Wait on ``stop_event`` when present, else sleep; return True when stop is signaled."""
    eff_delay = max(0.0, delay_sec)
    if stop_event is not None:
        return bool(stop_event.wait(timeout=eff_delay))
    _sleep_time(eff_delay)
    return False


def _run_account_trade_poll_loop(
    *,
    sync_once_fn,
    log_tag: str,
    poll_interval: float,
    default_poll_sec: float,
    backoff_init: float,
    default_backoff_init: float,
    backoff_max: float,
    default_backoff_max: float,
    heartbeat_fn=None,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Shared exponential-backoff REST trade-cache polling loop across venue bridges."""
    init_bo = safe_pos_float(backoff_init, default=default_backoff_init) or default_backoff_init
    max_bo = safe_pos_float(backoff_max, default=default_backoff_max) or default_backoff_max
    poll_s = safe_nonneg_float(poll_interval, default=default_poll_sec) or 0.0
    backoff = init_bo
    if callable(heartbeat_fn):
        heartbeat_fn(force=True)
    while not (stop_event is not None and stop_event.is_set()):
        try:
            cache = sync_once_fn()
            log(
                f"[{log_tag}][poll] {sum(len(v) for v in cache['items'].values())} fills "
                f"for {list(cache['items'].keys())}"
            )
            backoff = init_bo
            if callable(heartbeat_fn):
                heartbeat_fn()
            if _wait_or_sleep(stop_event, poll_s):
                break
        except Exception as exc:
            log(f"[{log_tag}][poll] error (backoff {backoff:.0f}s): {exc}")
            if callable(heartbeat_fn):
                heartbeat_fn()
            if _wait_or_sleep(stop_event, backoff):
                break
            backoff = min(backoff * 2.0, max_bo)


class _AccountTradeBridgeBase(ABC):
    """Abstract base contract (Level 2) for background venue account-trade sync bridges."""

    VENUE_KEY: str = ""
    VENUE_LABEL: str = ""
    LOG_TAG: str = ""
    THREAD_NAME: str = ""

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

    @classmethod
    def _resolve_daemon_config_and_client(cls):
        prov_cls = MarketDataProvider.resolve_venue_provider_class(cls.VENUE_KEY)
        try:
            if prov_cls is None:
                raise RuntimeError(f"{cls.VENUE_LABEL} provider not registered")
            cfg = prov_cls.resolve_cache_daemon_config()
            client, profile = prov_cls.build_cache_client()
            return cfg, client, profile
        except Exception as exc:
            msg = (
                f"skipped ({cls.VENUE_LABEL} provider not registered)"
                if prov_cls is None
                else f"skipped (no {cls.VENUE_LABEL} cache credentials): {exc}"
                if isinstance(exc, CredentialConfigurationError)
                else f"failed to initialize daemon config: {exc}"
            )
            log(f"[{cls.LOG_TAG}] {msg}")
            return None

    def _spawn_bridge_thread(self, target) -> bool:
        self._stop_event.clear()
        t = threading.Thread(
            target=target,
            name=self.THREAD_NAME or self.__class__.__name__,
            daemon=True,
        )
        self._thread = t
        t.start()
        return True

    @abstractmethod
    def start(self) -> bool:
        """Initialize credentials/config and start the background sync thread."""

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


class _HyperliquidAccountTradeBridge:
    """Channel bridge attaching ``userFills`` persistence onto a shared Hyperliquid WebSocket stream."""

    def __init__(
        self,
        ws_stream,
        *,
        user_address: Optional[str] = None,
        fills_cache_path: Optional[str] = None,
        resolve_env_user_address: Optional[bool] = None,
    ) -> None:
        self.ws_stream = ws_stream
        self.hl_prov_cls = MarketDataProvider.resolve_venue_provider_class("hyperliquid")
        ws_stream._fills_cache_path = fills_cache_path
        ws_stream._fills_ws_seeded = False
        ws_stream._last_fills_touch_ms = 0
        ws_stream._user_address = (
            self.hl_prov_cls.resolve_ws_user_address(
                user_address, resolve_env=resolve_env_user_address
            )
            if self.hl_prov_cls is not None
            else (str(user_address or "").strip() or None)
        )

    def resolve_fills_cache_file(self) -> str:
        ws_stream = self.ws_stream
        return (
            str(ws_stream._fills_cache_path)
            if ws_stream._fills_cache_path
            else cache_path(CACHE_HYPERLIQUID_TRADE_FILENAME)
        )

    def handle_user_fills(self, data: object, *, ts_ms: int) -> None:
        if self.hl_prov_cls is None:
            return
        ws_stream = self.ws_stream
        with ws_stream._lock:
            symbols = ws_stream._symbols
        parsed = self.hl_prov_cls.extract_ws_user_fills(
            data, symbols, candidate_mids_fn=ws_stream._candidate_mids_keys
        )
        if parsed is None:
            return
        is_snapshot, fills_by_symbol = parsed
        fills_file = self.resolve_fills_cache_file()
        CacheTradeManager.merge_venue_fills_snapshot(
            fills_file,
            fills_by_symbol,
            now_ms=ts_ms,
            is_snapshot=is_snapshot,
            touch_symbols=symbols,
        )
        try:
            CacheOrderFilledManager.merge_venue_filled_orders_snapshot(
                _sibling_order_filled_path(fills_file, "hyperliquid"),
                fills_by_symbol,
                venue="hyperliquid",
                now_ms=ts_ms,
                is_snapshot=is_snapshot,
                touch_symbols=symbols,
            )
        except Exception:
            pass
        if is_snapshot:
            ws_stream._fills_ws_seeded = True
        ws_stream._last_fills_touch_ms = ts_ms

    def touch_fills_cache_if_needed(self, ts_ms: int) -> None:
        ws_stream = self.ws_stream
        if (
            not ws_stream._fills_ws_seeded
            or (ts_ms - ws_stream._last_fills_touch_ms) < 15_000
        ):
            return
        try:
            with ws_stream._lock:
                symbols = ws_stream._symbols
            fills_file = self.resolve_fills_cache_file()
            CacheTradeManager.touch_venue_fills_snapshot(
                fills_file,
                ts_ms,
                symbols=symbols,
                create_missing=False,
            )
            CacheOrderFilledManager.touch_venue_filled_orders_snapshot(
                _sibling_order_filled_path(fills_file, "hyperliquid"),
                ts_ms,
                symbols=symbols,
                create_missing=False,
            )
            ws_stream._last_fills_touch_ms = ts_ms
        except Exception:
            pass

    def attach(self) -> None:
        ws_stream = self.ws_stream
        ws_stream._resolve_fills_cache_file = self.resolve_fills_cache_file
        ws_stream._handle_user_fills = self.handle_user_fills
        ws_stream._touch_fills_cache_if_needed = self.touch_fills_cache_if_needed
        if hasattr(ws_stream, "register_channel_handler"):
            hl_prov_cls = self.hl_prov_cls
            ws_stream.register_channel_handler(
                "userFills",
                self.handle_user_fills,
                subscribe_builder=lambda: (
                    setattr(ws_stream, "_fills_ws_seeded", False)
                    or (() if hl_prov_cls is None else hl_prov_cls.build_ws_user_fills_subscriptions(ws_stream._user_address))
                ),
                on_tick=self.touch_fills_cache_if_needed,
            )


def attach_hyperliquid_user_fills_bridge(
    ws_stream,
    *,
    user_address: Optional[str] = None,
    fills_cache_path: Optional[str] = None,
    resolve_env_user_address: Optional[bool] = None,
    **_unused_kwargs,
) -> None:
    """Attach the account-domain ``userFills`` WebSocket bridge onto a Hyperliquid stream instance."""
    _HyperliquidAccountTradeBridge(
        ws_stream,
        user_address=user_address,
        fills_cache_path=fills_cache_path,
        resolve_env_user_address=resolve_env_user_address,
    ).attach()


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
    if (kr_prov_cls := MarketDataProvider.resolve_venue_provider_class("kraken")) is None:
        raise RuntimeError("Kraken provider class is not registered")

    def _sync_once() -> dict:
        by_sym = kr_prov_cls.fetch_ledger_fills_by_symbol(
            client, pairs=pairs, lookback_sec=lookback_sec
        )
        eff_ms = resolve_now_ms()
        merged = CacheTradeManager._merge_venue_fills_snapshot_locked(
            cache_file,
            by_sym,
            now_ms=eff_ms,
            is_snapshot=True,
            sanitize_venue="kraken",
            indent=1,
            force_save=True,
        )
        try:
            CacheOrderFilledManager.merge_venue_filled_orders_snapshot(
                _sibling_order_filled_path(cache_file, "kraken"),
                by_sym,
                venue="kraken",
                now_ms=eff_ms,
                is_snapshot=True,
                touch_symbols=pairs,
                touch_create_missing=True,
                indent=1,
                force_save=True,
            )
        except Exception:
            pass
        return merged

    _run_account_trade_poll_loop(
        sync_once_fn=_sync_once,
        log_tag="kraken_cache",
        poll_interval=poll_interval,
        default_poll_sec=30.0,
        backoff_init=backoff_init,
        default_backoff_init=5.0,
        backoff_max=backoff_max,
        default_backoff_max=60.0,
        heartbeat_fn=heartbeat_fn,
        stop_event=stop_event,
    )


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
    if (kr_prov_cls := MarketDataProvider.resolve_venue_provider_class("kraken")) is None:
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

    last_flush_ms = [0]
    last_rest_sync_ms = [0]
    # While ownTrades WS is live, periodic REST Ledgers reconciliation only needs a 5-min safety net (10x less rate-limit cost).
    rest_sync_interval_ms = math.trunc(
        max(300.0, safe_pos_float(poll_interval, default=30.0) or 30.0) * 1000
    )
    if callable(heartbeat_fn):
        heartbeat_fn(force=True)

    backoff_base = max(5.0, safe_pos_float(backoff_init, default=5.0) or 5.0)
    backoff_cap = max(backoff_base, safe_pos_float(backoff_max, default=120.0) or 120.0)
    rate_limit_cap = max(backoff_cap, 960.0)  # Exceed Kraken's 15-min lockout window so recovery is automatic
    backoff_cur = [backoff_base]
    ws_subscribed = [False]
    last_err_text = [""]
    rest_sync_lock = threading.Lock()
    rest_sync_in_flight = [False]

    def _is_rate_limit_err(err_msg: object) -> bool:
        txt = str(err_msg or "").lower()
        return ("rate limit" in txt) or ("temporary lockout" in txt)

    def _bump_backoff(err_msg: object = "") -> None:
        txt = str(err_msg or "").lower()
        if _is_rate_limit_err(txt):
            # Escalate quickly (30 -> 90 -> 270 -> 960s) to clear Kraken's 15-min penalty without extending it.
            backoff_cur[0] = min(rate_limit_cap, max(30.0, backoff_cur[0] * 3.0))
        elif "invalid nonce" in txt:
            backoff_cur[0] = min(backoff_cap, max(15.0, backoff_cur[0] * 2.0))
        else:
            backoff_cur[0] = min(backoff_cap, max(5.0, backoff_cur[0] * 1.5))

    def _merge_rest(err_label: Optional[str] = None, *, now_ms: Optional[int] = None) -> Optional[dict]:
        try:
            by_sym = kr_prov_cls.fetch_ledger_fills_by_symbol(
                client, pairs=pairs, lookback_sec=lookback_sec
            )
            eff_ms = resolve_now_ms(now_ms)
            merged = CacheTradeManager._merge_venue_fills_snapshot_locked(
                cache_file,
                by_sym,
                now_ms=eff_ms,
                is_snapshot=False,
                touch_symbols=pairs,
                touch_create_missing=True,
                sanitize_venue="kraken",
                indent=1,
                force_save=True,
            )
            try:
                CacheOrderFilledManager.merge_venue_filled_orders_snapshot(
                    _sibling_order_filled_path(cache_file, "kraken"),
                    by_sym,
                    venue="kraken",
                    now_ms=eff_ms,
                    is_snapshot=False,
                    touch_symbols=pairs,
                    touch_create_missing=True,
                    indent=1,
                    force_save=True,
                )
            except Exception:
                pass
            last_flush_ms[0] = last_rest_sync_ms[0] = eff_ms
            return merged
        except Exception as exc:
            last_err_text[0] = str(exc)
            if err_label is None:
                raise
            log(f"[kraken_cache][ws] {err_label}: {exc}")
            return None

    # Initial REST seed: if rate-limited, defer WS token call until backoff clears instead of double-hitting Kraken.
    seed_ok = _merge_rest("REST seed failed") is not None
    if not seed_ok and _is_rate_limit_err(last_err_text[0]):
        _bump_backoff(last_err_text[0])
        log(f"[kraken_cache][ws] rate-limited on seed; cooling down {backoff_cur[0]:g}s before WS connect")
        if _wait_or_sleep(stop_event, backoff_cur[0]):
            return

    def on_open(ws) -> None:
        try:
            ws.send(kr_prov_cls.build_ws_own_trades_subscription(client))
            ws_subscribed[0] = True
            backoff_cur[0] = backoff_base
            last_err_text[0] = ""
            log("[kraken_cache][ws] subscribed ownTrades")
        except Exception as exc:
            ws_subscribed[0] = False
            last_err_text[0] = str(exc)
            _bump_backoff(exc)
            log(f"[kraken_cache][ws] subscribe failed: {exc}")
            try:
                ws.keep_running = False
                if getattr(ws, "sock", None) is not None:
                    ws.close()
            except Exception:
                pass

    def on_message(ws, msg) -> None:
        data = parse_json_payload(msg)
        if data is None:
            return
        now_ms = resolve_now_ms()

        fills_by_sym = kr_prov_cls.extract_ws_own_trades(data, pairs=pairs)
        should_hb = False
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
                launched_async = False
                if (now_ms - last_rest_sync_ms[0]) >= rest_sync_interval_ms:
                    with rest_sync_lock:
                        if not rest_sync_in_flight[0]:
                            rest_sync_in_flight[0] = True
                            launched_async = True
                    if launched_async:
                        # Run periodic REST Ledgers reconciliation off the WS reader thread so slow HTTP calls
                        # never starve RFC6455 ping/pong handling and trigger spurious ping/pong timeouts.
                        def _bg_rest_sync(sync_ms: int = now_ms) -> None:
                            try:
                                merged = _merge_rest(now_ms=sync_ms)
                                if merged is not None:
                                    log(
                                        f"[kraken_cache][ws] {sum(len(v) for v in merged['items'].values())} fills "
                                        f"for {list(merged['items'].keys())}"
                                    )
                            except Exception as exc:
                                log(f"[kraken_cache][ws] periodic REST Ledgers sync failed: {exc}")
                            finally:
                                with rest_sync_lock:
                                    rest_sync_in_flight[0] = False

                        threading.Thread(target=_bg_rest_sync, name="kraken-ws-rest-sync", daemon=True).start()
                if not launched_async and last_rest_sync_ms[0] > 0:
                    CacheTradeManager.touch_venue_fills_snapshot(
                        cache_file,
                        now_ms=now_ms,
                        symbols=pairs,
                        create_missing=False,
                        indent=1,
                    )
                    CacheOrderFilledManager.touch_venue_filled_orders_snapshot(
                        _sibling_order_filled_path(cache_file, "kraken"),
                        now_ms=now_ms,
                        symbols=pairs,
                        create_missing=False,
                        indent=1,
                    )
                last_flush_ms[0] = now_ms
                should_hb = True
        else:
            CacheTradeManager._merge_venue_fills_snapshot_locked(
                cache_file,
                fills_by_sym,
                now_ms=now_ms,
                is_snapshot=False,
                touch_symbols=pairs,
                touch_create_missing=False,
                sanitize_venue="kraken",
                indent=1,
            )
            try:
                CacheOrderFilledManager.merge_venue_filled_orders_snapshot(
                    _sibling_order_filled_path(cache_file, "kraken"),
                    fills_by_sym,
                    venue="kraken",
                    now_ms=now_ms,
                    is_snapshot=False,
                    touch_symbols=pairs,
                    touch_create_missing=False,
                    indent=1,
                )
            except Exception:
                pass
            last_flush_ms[0] = now_ms
            should_hb = True
        if should_hb and callable(heartbeat_fn):
            heartbeat_fn()

    def on_error(ws, err) -> None:
        err_s = str(err or "")
        if not getattr(ws, "keep_running", True) and "'NoneType' object has no attribute 'sock'" in err_s:
            return
        log(f"[kraken_cache][ws] error: {err}")

    while not (stop_event is not None and stop_event.is_set()):
        ws_subscribed[0] = False
        try:
            ws = websocket.WebSocketApp(
                resolved_ws_url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
            )
            if isinstance(ws_holder, list) and ws_holder:
                ws_holder[0] = ws
            ws.run_forever(ping_interval=30, ping_timeout=20)
        except Exception as exc:
            log(f"[kraken_cache][ws] run_forever: {exc}")
        finally:
            if isinstance(ws_holder, list) and ws_holder:
                ws_holder[0] = None
        if stop_event is not None and stop_event.is_set():
            break
        had_session = ws_subscribed[0]
        wait_s = backoff_cur[0]
        if had_session:
            backoff_cur[0] = backoff_base
        else:
            _bump_backoff(last_err_text[0])
        log(f"[kraken_cache][ws] disconnected; reconnect in {wait_s:g}s")
        if callable(heartbeat_fn):
            heartbeat_fn()
        if _wait_or_sleep(stop_event, wait_s):
            break
        # Only run REST Ledgers catch-up after an active WS session dropped (never stack Ledgers before GetWebSocketsToken when rate-limited).
        if had_session and (resolve_now_ms() - last_rest_sync_ms[0]) >= rest_sync_interval_ms:
            if _merge_rest("REST resync after disconnect failed") is None:
                _bump_backoff(last_err_text[0] or "rate limit")


class _KrakenAccountTradeBridge(_AccountTradeBridgeBase):
    """Background daemon bridge managing Kraken ``ownTrades`` WS and ``Ledgers`` REST sync inside cacheManager."""

    VENUE_KEY = "kraken"
    VENUE_LABEL = "Kraken"
    LOG_TAG = "kraken_cache"
    THREAD_NAME = "KrakenAccountTradeBridge"

    def start(self) -> bool:
        resolved = self._resolve_daemon_config_and_client()
        if resolved is None:
            return False
        cfg, client, profile = resolved
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

        return self._spawn_bridge_thread(_runner)


def run_t212_trade_poll_loop(
    client,
    *,
    cache_file: str,
    symbols: Sequence[str],
    poll_interval: float,
    backoff_init: float,
    backoff_max: float,
    heartbeat_fn=None,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Run the periodic REST order-history polling loop for Trading 212 trade cache persistence."""
    if (t212_prov_cls := MarketDataProvider.resolve_venue_provider_class("t212")) is None:
        raise RuntimeError("T212 provider class is not registered")

    def _sync_once() -> dict:
        by_sym = t212_prov_cls.fetch_history_fills_by_symbol(
            client, symbols=symbols, publish_shared_cache=True
        )
        eff_ms = resolve_now_ms()
        merged = CacheTradeManager._merge_venue_fills_snapshot_locked(
            cache_file,
            by_sym,
            now_ms=eff_ms,
            is_snapshot=False,
            touch_symbols=symbols,
            touch_create_missing=True,
            sanitize_venue="t212",
            indent=1,
            force_save=True,
        )
        try:
            CacheOrderFilledManager.merge_venue_filled_orders_snapshot(
                _sibling_order_filled_path(cache_file, "t212"),
                by_sym,
                venue="t212",
                now_ms=eff_ms,
                is_snapshot=False,
                touch_symbols=symbols,
                touch_create_missing=True,
                indent=1,
                force_save=True,
            )
        except Exception:
            pass
        return merged

    _run_account_trade_poll_loop(
        sync_once_fn=_sync_once,
        log_tag="t212_cache",
        poll_interval=poll_interval,
        default_poll_sec=60.0,
        backoff_init=backoff_init,
        default_backoff_init=10.0,
        backoff_max=backoff_max,
        default_backoff_max=120.0,
        heartbeat_fn=heartbeat_fn,
        stop_event=stop_event,
    )


class _T212AccountTradeBridge(_AccountTradeBridgeBase):
    """Background daemon bridge managing Trading 212 periodic order-history snapshot sync inside cacheManager."""

    VENUE_KEY = "t212"
    VENUE_LABEL = "T212"
    LOG_TAG = "t212_cache"
    THREAD_NAME = "T212AccountTradeBridge"

    def start(self) -> bool:
        resolved = self._resolve_daemon_config_and_client()
        if resolved is None:
            return False
        cfg, client, profile = resolved
        log(
            f"[t212_cache] start: mode=poll credentials={profile} "
            f"symbols={cfg['symbols']} poll={cfg['poll_interval']}s -> {cfg['cache_file']}"
        )

        def _runner() -> None:
            run_t212_trade_poll_loop(
                client,
                cache_file=cfg["cache_file"],
                symbols=cfg["symbols"],
                poll_interval=cfg["poll_interval"],
                backoff_init=cfg["backoff_init"],
                backoff_max=cfg["backoff_max"],
                stop_event=self._stop_event,
            )

        return self._spawn_bridge_thread(_runner)


class _VenueOpenOrdersBridge(_AccountTradeBridgeBase):
    """Background daemon bridge syncing non-primary venue open orders into ``cache_order_open_<venue>.json``."""

    VENUE_KEY = "multi"
    VENUE_LABEL = "MultiVenueOpenOrders"
    LOG_TAG = "open_orders_cache"
    THREAD_NAME = "VenueOpenOrdersBridge"

    def start(self) -> bool:
        venues = [
            v for v in CacheOrderOpenManager.VENUE_FILENAMES
            if not is_primary_stream_venue(v)
        ]
        if not venues:
            return False

        def _runner() -> None:
            while not self._stop_event.is_set():
                now_ms = resolve_now_ms()
                now_s = now_ms / 1000.0
                for v_key in venues:
                    if self._stop_event.is_set():
                        break
                    try:
                        v_syms = list(sym.symbols_for(v_key))
                        if not v_syms:
                            continue
                        prov_cls = MarketDataProvider.resolve_venue_provider_class(v_key)
                        if prov_cls is None:
                            continue
                        prov = prov_cls()
                        open_by_sym: dict[str, list[dict[str, Any]]] = {}
                        all_ok = True
                        try:
                            batch = prov.open_orders_by_symbol(v_syms)
                            if isinstance(batch, dict):
                                for s_item in v_syms:
                                    raw_open = batch.get(s_item)
                                    if isinstance(raw_open, list):
                                        open_by_sym[s_item] = raw_open
                                    else:
                                        all_ok = False
                            else:
                                all_ok = False
                        except Exception:
                            all_ok = False
                        if open_by_sym:
                            CacheOrderOpenManager.replace_venue_open_orders_snapshot(
                                cache_path(CacheOrderOpenManager.disk_filename(v_key)),
                                open_by_sym,
                                now_ms=now_ms,
                                verified_ts=now_s if all_ok else None,
                            )
                    except Exception as exc:
                        log(f"[{self.LOG_TAG}] {v_key} sync failed: {exc}")
                if _wait_or_sleep(self._stop_event, OPEN_ORDER_SYNC_INTERVAL_SEC):
                    break

        return self._spawn_bridge_thread(_runner)


class _VenueBalancesBridge(_AccountTradeBridgeBase):
    """Background daemon bridge syncing multi-venue raw account balances into ``cache_balance_<venue>.json``."""

    VENUE_KEY = "multi"
    VENUE_LABEL = "MultiVenueBalances"
    LOG_TAG = "balances_cache"
    THREAD_NAME = "VenueBalancesBridge"

    @staticmethod
    def _venue_requested_assets(v_key: str) -> list[str]:
        specs = list(sym.select_instruments(v_key).values())
        assets: list[str] = []
        seen: set[str] = set()
        for spec in specs:
            if is_equity_venue(v_key):
                parts: tuple[Any, ...] = (spec.symbol, "USD")
            else:
                parts = (spec.base, spec.quote) if (spec.base or spec.quote) else (split_symbol_assets(spec.symbol) or ())
            for part in parts:
                p_c = canonical_balance_asset_key(part, venue=v_key)
                if p_c and p_c.upper() not in seen:
                    seen.add(p_c.upper())
                    assets.append(p_c)
        return assets

    def sync_venue_once(self, v_key: str, *, now_s: Optional[float] = None) -> bool:
        """Fetch one venue's atomic ``(free, locked)`` balances in batch and persist ``cache_balance_<venue>.json``."""
        eff_now = resolve_now_ts(now_s)
        prov_cls = MarketDataProvider.resolve_venue_provider_class(v_key)
        if prov_cls is None:
            return False
        req_assets = self._venue_requested_assets(v_key)
        if not req_assets:
            return False
        prov = prov_cls()
        batch = prov.all_free_and_locked_balances(req_assets)
        if not isinstance(batch, dict) or not batch:
            return False
        breakdown: Optional[dict[str, dict[str, float]]] = None
        bd_fn = getattr(prov, "usd_cash_breakdown", None)
        if callable(bd_fn):
            try:
                raw_bd = bd_fn()
                if isinstance(raw_bd, dict) and raw_bd:
                    breakdown = {"USD": raw_bd}
            except Exception:
                pass
        CacheBalanceManager.replace_venue_balances_snapshot(
            cache_path(CacheBalanceManager.disk_filename(v_key)),
            batch,
            venue=v_key,
            now_s=eff_now,
            verified_ts=eff_now,
            breakdown=breakdown,
        )
        return True

    def start(self) -> bool:
        venues = list(CacheBalanceManager.VENUE_FILENAMES.keys())
        if not venues:
            return False

        def _runner() -> None:
            seeded_equity_venues: set[str] = set()
            while not self._stop_event.is_set():
                now_s = resolve_now_ts()
                for v_key in venues:
                    if self._stop_event.is_set():
                        break
                    try:
                        if is_equity_venue(v_key) and v_key in seeded_equity_venues:
                            v_syms = list(sym.symbols_for(v_key))
                            probe_sym = v_syms[0] if v_syms else "NVDA_US_EQ"
                            if not is_symbol_market_open(probe_sym, now_s):
                                continue
                        if self.sync_venue_once(v_key, now_s=now_s) and is_equity_venue(v_key):
                            seeded_equity_venues.add(v_key)
                    except Exception as exc:
                        log(f"[{self.LOG_TAG}] {v_key} sync failed: {exc}")
                if _wait_or_sleep(self._stop_event, BALANCE_SYNC_INTERVAL_SEC):
                    break

        return self._spawn_bridge_thread(_runner)


_kraken_ws_bridge: Optional[_KrakenAccountTradeBridge] = None
_kraken_ws_bridge_lock = threading.Lock()
_t212_poll_bridge: Optional[_T212AccountTradeBridge] = None
_t212_poll_bridge_lock = threading.Lock()
_venue_open_orders_bridge: Optional[_VenueOpenOrdersBridge] = None
_venue_open_orders_bridge_lock = threading.Lock()
_venue_balances_bridge: Optional[_VenueBalancesBridge] = None
_venue_balances_bridge_lock = threading.Lock()


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


def start_t212_account_poll_bridge() -> Optional[_T212AccountTradeBridge]:
    """Idempotently start the Trading 212 periodic order-history bridge inside cacheManager."""
    global _t212_poll_bridge
    with _t212_poll_bridge_lock:
        if _t212_poll_bridge is not None and _t212_poll_bridge.is_running:
            return _t212_poll_bridge
        bridge = _T212AccountTradeBridge()
        if bridge.start():
            _t212_poll_bridge = bridge
            return bridge
        return None


def start_venue_open_orders_bridge() -> Optional[_VenueOpenOrdersBridge]:
    """Idempotently start the cross-venue open-order background bridge inside cacheManager."""
    global _venue_open_orders_bridge
    with _venue_open_orders_bridge_lock:
        if _venue_open_orders_bridge is not None and _venue_open_orders_bridge.is_running:
            return _venue_open_orders_bridge
        bridge = _VenueOpenOrdersBridge()
        if bridge.start():
            _venue_open_orders_bridge = bridge
            return bridge
        return None


def start_venue_balances_bridge() -> Optional[_VenueBalancesBridge]:
    """Idempotently start the multi-venue account balances background bridge inside cacheManager."""
    global _venue_balances_bridge
    with _venue_balances_bridge_lock:
        if _venue_balances_bridge is not None and _venue_balances_bridge.is_running:
            return _venue_balances_bridge
        bridge = _VenueBalancesBridge()
        if bridge.start():
            _venue_balances_bridge = bridge
            return bridge
        return None


# =============================================================================
# 7. SSOT SINGLETON FACTORY GETTERS
# =============================================================================

ORDER_SYNC_INTERVAL_SEC = 0.4 * 60
OPEN_ORDER_SYNC_INTERVAL_SEC = 15.0
BALANCE_SYNC_INTERVAL_SEC = 60.0
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
_VENUE_ORDER_FILLED_CACHE_NAMES = {
    "binance": "BinanceOrderFilled",
    "kraken": "KrakenOrderFilled",
    "hyperliquid": "HyperliquidOrderFilled",
    "t212": "T212OrderFilled",
}
_VENUE_ORDER_OPEN_CACHE_NAMES = {
    "binance": "BinanceOrderOpen",
    "kraken": "KrakenOrderOpen",
    "hyperliquid": "HyperliquidOrderOpen",
    "t212": "T212OrderOpen",
}
_VENUE_BALANCE_CACHE_NAMES = {
    "binance": "BinanceBalance",
    "kraken": "KrakenBalance",
    "hyperliquid": "HyperliquidBalance",
    "t212": "T212Balance",
}


def _get_or_create_readonly_account_reader(
    cache_name: str,
    *,
    can_reuse_fn,
    can_store: bool,
    builder_fn,
    resync_existing: bool = False,
):
    return CacheManagerInterface._get_or_create_readonly_reader(
        _READONLY_ACCOUNT_LOCK,
        _READONLY_ACCOUNT_READERS,
        cache_name,
        can_reuse_fn=can_reuse_fn,
        can_store=can_store,
        builder_fn=builder_fn,
        resync_existing=resync_existing,
    )


def get_asset_value_manager(
    symbols=None,
    filename=CacheAssetValueManager.DEFAULT_FILENAME,
    *,
    data_sync_mode: Optional[str] = CacheManagerInterface.MODE_DISK_READER,
) -> CacheAssetValueManager:
    """Return the ``CacheAssetValueManager`` singleton without importing ``cacheManager`` in read-only consumers."""
    (
        delegated,
        res,
        resolved_mode,
        save_state_flag,
        start_periodic,
        eff_symbols,
    ) = CacheManagerInterface._delegate_or_resolve_sync_and_symbols(
        "AssetValue",
        symbols=symbols,
        default_symbols=["TOTAL"],
        data_sync_mode=data_sync_mode,
        filename=filename,
        default_filename=CacheAssetValueManager.DEFAULT_FILENAME,
        cm_getter_name="get_asset_value_manager",
        getter_tag="_is_ssot_asset_value_getter",
    )
    if delegated:
        return res
    is_default_file = filename == CacheAssetValueManager.DEFAULT_FILENAME
    is_readonly = not save_state_flag and not start_periodic
    return _get_or_create_readonly_account_reader(
        "AssetValue",
        can_reuse_fn=lambda inst: (
            eff_symbols == list(getattr(inst, "symbols", []))
            and is_default_file
            and is_readonly
        ),
        can_store=is_default_file,
        builder_fn=lambda: CacheAssetValueManager(
            sync_ts=ASSET_VALUE_SYNC_INTERVAL_SEC,
            filename=filename,
            symbols=eff_symbols,
            read_only=is_readonly,
            data_sync_mode=resolved_mode,
        ),
        resync_existing=True,
    )


get_asset_value_manager._is_ssot_asset_value_getter = True  # type: ignore[attr-defined]


def _get_venue_order_cache_manager(
    *,
    venue: str,
    symbols: Any,
    filename: Any,
    data_sync_mode: Optional[str],
    cache_cls: type[CacheAccountManagerBase],
    name_map: dict[str, str],
    sync_interval_sec: float,
    cm_getter_name: str,
    getter_tag: str,
    err_label: str,
    resync_existing: bool = False,
) -> Any:
    v_norm = "binance" if (venue is None or (isinstance(venue, str) and not venue.strip())) else normalize_venue_name(venue)
    cache_name = name_map.get(v_norm or "")
    if cache_name is None:
        raise ValueError(f"Unsupported {err_label} cache venue: {venue!r}")
    is_non_primary = not is_primary_stream_venue(v_norm)
    default_fname = cache_cls.disk_filename(v_norm)
    eff_filename = default_fname if filename is None else filename
    delegated, res, resolved_mode, save_state_flag, start_periodic, eff_symbols = (
        CacheManagerInterface._delegate_or_resolve_sync_and_symbols(
            cache_name,
            symbols=symbols,
            default_symbols=None if is_non_primary else (lambda: list(sym.symbols)),
            data_sync_mode=data_sync_mode,
            filename=eff_filename,
            default_filename=default_fname,
            config_read_only=is_non_primary,
            cm_getter_name=cm_getter_name,
            getter_tag=getter_tag,
        )
    )
    if delegated:
        return res
    can_cache = symbols is None and eff_filename == default_fname
    is_readonly = not save_state_flag and not start_periodic
    return _get_or_create_readonly_account_reader(
        cache_name,
        can_reuse_fn=lambda _inst: can_cache and is_readonly,
        can_store=can_cache,
        builder_fn=lambda: cache_cls(
            sync_ts=sync_interval_sec,
            filename=eff_filename,
            symbols=eff_symbols,
            read_only=(is_non_primary or is_readonly),
            data_sync_mode=resolved_mode,
            venue=v_norm,
        ),
        resync_existing=resync_existing,
    )


def get_order_filled_manager(
    venue: str = "binance",
    symbols=None,
    filename=None,
    *,
    data_sync_mode: Optional[str] = CacheManagerInterface.MODE_DISK_READER,
) -> CacheOrderFilledManager:
    """Return the venue ``CacheOrderFilledManager`` singleton without importing ``cacheManager`` in read-only consumers."""
    return _get_venue_order_cache_manager(
        venue=venue,
        symbols=symbols,
        filename=filename,
        data_sync_mode=data_sync_mode,
        cache_cls=CacheOrderFilledManager,
        name_map=_VENUE_ORDER_FILLED_CACHE_NAMES,
        sync_interval_sec=ORDER_SYNC_INTERVAL_SEC,
        cm_getter_name="get_order_filled_manager",
        getter_tag="_is_ssot_order_filled_getter",
        err_label="filled-order",
    )


get_order_filled_manager._is_ssot_order_filled_getter = True  # type: ignore[attr-defined]


def get_order_open_manager(
    venue: str = "binance",
    symbols=None,
    filename=None,
    *,
    data_sync_mode: Optional[str] = CacheManagerInterface.MODE_DISK_READER,
) -> CacheOrderOpenManager:
    """Return the venue ``CacheOrderOpenManager`` singleton without importing ``cacheManager`` in read-only consumers."""
    return _get_venue_order_cache_manager(
        venue=venue,
        symbols=symbols,
        filename=filename,
        data_sync_mode=data_sync_mode,
        cache_cls=CacheOrderOpenManager,
        name_map=_VENUE_ORDER_OPEN_CACHE_NAMES,
        sync_interval_sec=OPEN_ORDER_SYNC_INTERVAL_SEC,
        cm_getter_name="get_order_open_manager",
        getter_tag="_is_ssot_order_open_getter",
        err_label="open-order",
        resync_existing=True,
    )


get_order_open_manager._is_ssot_order_open_getter = True  # type: ignore[attr-defined]


def get_balance_manager(
    venue: str = "binance",
    symbols=None,
    filename=None,
    *,
    data_sync_mode: Optional[str] = CacheManagerInterface.MODE_DISK_READER,
) -> CacheBalanceManager:
    """Return the venue ``CacheBalanceManager`` singleton without importing ``cacheManager`` in read-only consumers."""
    v_norm = (
        "binance"
        if (venue is None or (isinstance(venue, str) and not venue.strip()))
        else normalize_venue_name(venue)
    )
    cache_name = _VENUE_BALANCE_CACHE_NAMES.get(v_norm or "")
    if cache_name is None:
        raise ValueError(f"Unsupported balance cache venue: {venue!r}")
    default_fname = CacheBalanceManager.disk_filename(v_norm)
    eff_filename = default_fname if filename is None else filename
    delegated, res, resolved_mode, save_state_flag, start_periodic, eff_symbols = (
        CacheManagerInterface._delegate_or_resolve_sync_and_symbols(
            cache_name,
            symbols=symbols,
            default_symbols=None,
            data_sync_mode=data_sync_mode,
            filename=eff_filename,
            default_filename=default_fname,
            config_read_only=True,
            cm_getter_name="get_balance_manager",
            getter_tag="_is_ssot_balance_getter",
        )
    )
    if delegated:
        return res
    can_cache = symbols is None and eff_filename == default_fname
    is_readonly = not save_state_flag and not start_periodic
    return _get_or_create_readonly_account_reader(
        cache_name,
        can_reuse_fn=lambda _inst: can_cache and is_readonly,
        can_store=can_cache,
        builder_fn=lambda: CacheBalanceManager(
            sync_ts=BALANCE_SYNC_INTERVAL_SEC,
            filename=eff_filename,
            symbols=eff_symbols,
            read_only=True,
            data_sync_mode=resolved_mode,
            venue=v_norm,
        ),
        resync_existing=True,
    )


get_balance_manager._is_ssot_balance_getter = True  # type: ignore[attr-defined]


def get_trade_manager(
    venue: str = "binance",
    symbols=None,
    *,
    data_sync_mode: Optional[str] = CacheManagerInterface.MODE_DISK_READER,
) -> CacheTradeManager:
    """Return the venue ``CacheTradeManager`` singleton without importing ``cacheManager`` in read-only consumers."""
    v_norm = (
        "binance"
        if (venue is None or (isinstance(venue, str) and not venue.strip()))
        else normalize_venue_name(venue)
    )
    cache_name = _VENUE_TRADE_CACHE_NAMES.get(v_norm or "")
    if cache_name is None:
        raise ValueError(f"Unsupported trade cache venue: {venue!r}")
    is_non_binance = not is_primary_stream_venue(v_norm)
    (
        delegated,
        res,
        resolved_mode,
        save_state_flag,
        start_periodic,
        eff_symbols,
    ) = CacheManagerInterface._delegate_or_resolve_sync_and_symbols(
        cache_name,
        symbols=symbols,
        default_symbols=None if is_non_binance else list(sym.symbols),
        data_sync_mode=data_sync_mode,
        config_read_only=is_non_binance,
    )
    if delegated:
        return res
    is_readonly = not save_state_flag and not start_periodic
    return _get_or_create_readonly_account_reader(
        cache_name,
        can_reuse_fn=lambda _inst: symbols is None and is_readonly,
        can_store=symbols is None,
        builder_fn=lambda: CacheTradeManager(
            sync_ts=TRADE_SYNC_INTERVAL_SEC,
            filename=CacheTradeManager.disk_filename(v_norm),
            symbols=eff_symbols,
            read_only=(is_non_binance or is_readonly),
            data_sync_mode=resolved_mode,
            venue=v_norm,
            stream_format="json" if is_non_binance else "jsonl",
        ),
    )


get_trade_manager._is_ssot_trade_getter = True  # type: ignore[attr-defined]




