"""Price domain cache hierarchy (Level 2 abstract base + Level 3 concrete classes).

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
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import bisect
import builtins
from collections import defaultdict, deque
import concurrent.futures as _futures
import math
import os
import socket
import sys
import threading
import time
from typing import Any, Callable, Optional

from botcore import (
    FileLock,
    PRICE_FALLBACK_HTTP_DEADLINE_SEC,
    PRICE_FRESHNESS_MAX_AGE_MS,
    PRICE_FRESHNESS_MAX_AGE_SEC,
    PRICE_STREAM_POLL_INTERVAL_SEC,
    PRICE_WS_TIMEOUT_SEC,
    SHORT_TREND_FRESHNESS_MAX_AGE_SEC,
    _is_fresh_obs_ts,
    _parse_bool_flag,
    _ts_ms_to_s,
    clean_upper_token,
    extract_current_price_item,
    extract_timestamp_ms,
    extract_timestamp_s,
    is_live_network_backfill_enabled,
    is_rth_gate_enabled,
    is_us_rth_open,
    load_dotenv,
    normalize_epoch_sec,
    normalize_timestamp_ms,
    parse_symbol_list,
    required_float_env,
    required_int_env,
    resolve_now_ms,
    resolve_now_ts,
    resolve_positive_max_age,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
    safe_pos_int,
)
from cacheInterface import (
    CacheManagerInterface,
    _cm_override,
    _monotonic_time,
    _norm_symbol_keys,
    _now_time,
)
from cachepaths import (
    CACHE_24_PRICE_PREFIX,
    CACHE_365_PRICE_PREFIX,
    CACHE_CURRENT_PRICE_FILENAME,
    CACHE_CURRENT_PRICE_TICKS_SOCKET_FILENAME,
    CACHE_SPARSE_PRICE_PREFIX,
    cache_24_price_filename,
    cache_ohlc_24h_filename,
    cache_path,
    cache_symbol_variants,
    cachedb_root_for_file,
    canonical_base_asset,
    canonical_symbol_key,
    clean_symbol_key,
    compatible_quote_suffixes,
    filter_candidates_by_asset_class,
    get_cachedb_dir,
    has_equity_suffix,
    is_equity_venue,
    is_equity_venue_or_suffix,
    normalize_venue_name,
    resolve_cache_dir_path,
    resolve_cachedb_filepath,
    split_symbol_assets,
    telemetry_cache_file_path,
)
import instrument_registry as sym
from instrument_registry import (
    _DEFAULT_USD_QUOTES,
    casefold_get,
    is_equity_symbol,
    is_symbol_market_open,
    tracked_price_instruments,
)
import providers.market_api as _market_api
from providers.base import MarketDataProvider
from providers.ohlc_utils import normalize_yahoo_5d_chart_bars
from providers.price_poller import (  # noqa: F401
    HyperliquidPriceStream,
    KrakenPriceStream,
    NonBinanceTrendPoller as _NonBinanceTrendPoller,
    VenuePriceStreamBase as _VenuePriceStreamBase,
    _NB_FETCH_DEADLINE_SEC,
    fetch_price_with_deadline as _fetch_price_with_deadline,
    start_nonbinance_trend_poller as _start_nonbinance_trend_poller,
)
from providers.ssot_bridge import live_price_fetch_context
from state_io import (
    atomic_snapshot_file,
    atomic_text_writer,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json_key,
    commit_file_and_meta,
    discard_generation_file,
    format_jsonl_bucket_line,
    generation_file_lock,
    generation_lock_path,
    iter_jsonl_retention_lines,
    meta_file_path,
    parse_json_payload,
    previous_generation_path,
    read_json_dict,
    read_json_dict_recovering,
    read_tail_complete_lines_text,
    read_tail_lines,
    save_json_state,
    write_meta_json,
)

load_dotenv()

CM_LONG_ARCHIVE_MONTHS = required_float_env("CM_LONG_ARCHIVE_MONTHS")
CM_LONG_ARCHIVE_SAMPLE_SEC = required_float_env("CM_LONG_ARCHIVE_SAMPLE_SEC")
CM_LONG_ARCHIVE_FLUSH_SEC = required_float_env("CM_LONG_ARCHIVE_FLUSH_SEC")
CM_LONG_ARCHIVE_MEMORY_ROWS = required_int_env("CM_LONG_ARCHIVE_MEMORY_ROWS")
for _name, _value in (
    ("CM_LONG_ARCHIVE_MONTHS", CM_LONG_ARCHIVE_MONTHS),
    ("CM_LONG_ARCHIVE_SAMPLE_SEC", CM_LONG_ARCHIVE_SAMPLE_SEC),
    ("CM_LONG_ARCHIVE_FLUSH_SEC", CM_LONG_ARCHIVE_FLUSH_SEC),
    ("CM_LONG_ARCHIVE_MEMORY_ROWS", CM_LONG_ARCHIVE_MEMORY_ROWS),
):
    if _value <= 0:
        raise ValueError(f"{_name} must be positive")


_bn_price_prov_cls = MarketDataProvider.resolve_venue_provider_class("binance")


class CachePriceManagerBase(CacheManagerInterface, ABC):
    """Abstract base contract (Level 2) for price-series caches."""

    cls_name = "CachePriceManagerBase"
    _default_api_client = (
        _bn_price_prov_cls.resolve_default_api_client()
        if _bn_price_prov_cls is not None
        else None
    )
    _default_current_price_getter: Optional[Callable[..., Any]] = None

    @classmethod
    def bind_current_price_getter(cls, getter: Callable[..., Any]) -> None:
        """Bind the default current-price reader factory from ``cacheManager``."""
        if callable(getter):
            cls._default_current_price_getter = staticmethod(getter)

    def __init__(
        self,
        *args,
        current_price_getter: Optional[Callable[..., Any]] = None,
        current_price_mgr: Any = None,
        **kwargs,
    ):
        if type(self) is CachePriceManagerBase:
            raise TypeError("Cannot instantiate abstract class CachePriceManagerBase")
        self._current_price_getter = current_price_getter
        self._current_price_mgr = current_price_mgr
        self._subscribers_lock = threading.Lock()
        self._deferred_ticks_lock = threading.Lock()
        self._deferred_price_ticks: deque = deque(maxlen=4096)
        self._last_accepted_tick_ts_ms: dict[str, int] = {}
        self._price_subscribers = []
        super().__init__(*args, **kwargs)

    # -- Subscriber pattern for forwarding prices -----------------------------

    def subscribe_price(self, subscriber) -> None:
        """Subscribe an object implementing ``on_price_update(symbol, ts_ms, price)``."""
        with self._subscribers_lock:
            if subscriber not in self._price_subscribers:
                self._price_subscribers.append(subscriber)
        if (
            not getattr(subscriber, "save_state", False)
            and getattr(subscriber, "data_sync_mode", None) == self.MODE_DISK_READER
        ):
            subscriber.data_sync_mode = self.MODE_LIVE_READER

    def unsubscribe_price(self, subscriber) -> None:
        with self._subscribers_lock:
            if subscriber in self._price_subscribers:
                self._price_subscribers.remove(subscriber)

    def _notify_price_subscribers(self, symbol: str, ts_ms: int, price: float) -> None:
        with self._subscribers_lock:
            subs = tuple(self._price_subscribers)
        for sub in subs:
            try:
                sub.on_price_update(symbol, ts_ms, price)
            except Exception as e:
                print(f"[{self.cls_name}] Error notifying subscriber {sub}: {e}")

    @staticmethod
    def extract_jsonl_series_point(
        rec: object,
        allowed_symbols_upper: Optional[set[str]] = None,
    ) -> Optional[list]:
        """Parse one ``{"s": symbol, "i": [ts_ms, price, ...]}`` or ``[ts_ms, price]`` JSONL price archive record."""
        if isinstance(rec, dict):
            rec_s = rec.get("s")
            if (
                isinstance(rec_s, str)
                and rec_s.strip()
                and allowed_symbols_upper
                and rec_s.strip().upper() not in allowed_symbols_upper
            ):
                return None
            item = rec.get("i")
        else:
            item = rec if isinstance(rec, list) else None
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            return [item[0], item[1]]
        return None

    @staticmethod
    def parse_ssot_price_point(
        entry: Any, fallback_ts: Optional[float] = None
    ) -> Optional[tuple[int, float]]:
        """Parse a cached price entry (dict, ``[ts, px]``, or nested list) into ``(ts_ms, price)``."""
        def _finalize_point(ts_ms: Optional[int], px_f: Optional[float]) -> Optional[tuple[int, float]]:
            if (ts_ms is None or ts_ms <= 0) and fallback_ts is not None:
                ts_ms = normalize_timestamp_ms(fallback_ts, default=None)
            return (ts_ms, px_f) if ts_ms is not None and ts_ms > 0 and px_f is not None else None

        if isinstance(entry, dict):
            ts_ms = extract_timestamp_ms(entry, default=None)
            px_f = extract_current_price_item(entry)
            if px_f is None:
                raw_px = entry.get("price") if entry.get("price") is not None else entry.get("px")
                px_f = safe_pos_float(raw_px, default=None)
            return _finalize_point(ts_ms, px_f)
        if isinstance(entry, (list, tuple)):
            if len(entry) >= 2 and not isinstance(entry[0], (list, tuple, dict)):
                return _finalize_point(
                    normalize_timestamp_ms(entry[0], default=None),
                    safe_pos_float(entry[1], default=None),
                )
            for sub in reversed(entry):
                if not isinstance(sub, (list, tuple, dict)):
                    continue
                parsed = CachePriceManagerBase.parse_ssot_price_point(sub, fallback_ts=fallback_ts)
                if parsed is not None:
                    return parsed
            return None
        return _finalize_point(None, extract_current_price_item(entry))

    def rebuild_fetchtimes(self):
        return self._rebuild_symbol_entry_fetchtimes()

    def _validate_monotonic_tick(
        self, symbol: str, ts_ms: object, price: object
    ) -> Optional[tuple[int, float]]:
        ts_val = safe_pos_int(ts_ms, default=None)
        price_val = safe_pos_float(price, default=None)
        if ts_val is None or price_val is None:
            return None
        with self.lock:
            entries = self.cache.get(symbol)
            if entries and ts_val < self._entry_timestamp_ms(entries[-1]):
                return None
        return ts_val, price_val

    def _log_remote_fetch_error(self, label: str, symbol: str, err: Exception) -> list:
        print(f"[{self.cls_name}][Error] {label} {symbol}: {err}")
        return []

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
            cpm = getattr(self, "_current_price_mgr", None)
            if cpm is None:
                explicit_getter = getattr(self, "_current_price_getter", None)
                if callable(explicit_getter):
                    cpm = explicit_getter()
                else:
                    get_cpm = _cm_override("get_current_price_manager", self._default_current_price_getter)
                    if callable(get_cpm):
                        cpm = get_cpm()
            if cpm is None:
                return []
            entry = cpm.get_price(symbol)
        except Exception as e:
            return self._log_remote_fetch_error("get_price", symbol, e)

        if not entry or entry[1] is None:
            return []
        tick = self._validate_monotonic_tick(symbol, entry[0], entry[1])
        return [[tick[0], tick[1]]] if tick is not None else []


class CacheSparsePriceManager(CachePriceManagerBase):
    """Store sparse three-minute price samples (Level 3) for two years in JSONL."""

    FILE_PREFIX = CACHE_SPARSE_PRICE_PREFIX
    FILE_EXT = "jsonl"

    def __init__(
        self,
        sync_ts,
        symbols,
        filename,
        api_client=None,
        read_only=False,
        *,
        current_price_getter=None,
        current_price_mgr=None,
        data_sync_mode: Optional[str] = None,
    ):
        super().__init__(
            sync_ts,
            symbols,
            filename,
            append_mode=True,
            api_client=api_client,
            append_persist=True,
            read_only=read_only,
            current_price_getter=current_price_getter,
            current_price_mgr=current_price_mgr,
            data_sync_mode=data_sync_mode,
        )


class Cache24PriceManager(CachePriceManagerBase):
    """Collect maximum-granularity prices (Level 3) over the latest ``KEEP_HOURS``."""

    KEEP_HOURS = 24
    FILE_PREFIX = CACHE_24_PRICE_PREFIX
    FILE_EXT = "json"
    _filter_fetchtimes_to_symbols = True
    EQUITY_OHLC_RTH_REFRESH_SEC = 240.0
    EQUITY_OHLC_OFF_RTH_REFRESH_SEC = 1800.0

    def __init__(
        self,
        sync_ts,
        symbols,
        filename,
        api_client=None,
        read_only=False,
        *,
        current_price_getter=None,
        current_price_mgr=None,
        data_sync_mode: Optional[str] = None,
    ):
        self._last_eq_ohlc_refresh_ts: dict[str, float] = {}
        super().__init__(
            sync_ts,
            symbols,
            filename,
            append_mode=True,
            api_client=api_client,
            read_only=read_only,
            current_price_getter=current_price_getter,
            current_price_mgr=current_price_mgr,
            data_sync_mode=data_sync_mode,
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
            self._sync_last_accepted_tick_ts_locked()

    def _on_stale_memory_detected_locked(self) -> None:
        self._reload_from_disk()

    def _merge_ohlc_rows_locked(self, symbol: str, base_rows: list) -> list:
        """Merge enriched 3-element OHLCV ``base_rows`` with pre/post-window in-memory ticks for ``symbol``."""
        by_ts = {
            ts_b: (list(r) if isinstance(r, tuple) else r)
            for r in base_rows
            if (ts_b := self._entry_timestamp_ms(r)) > 0
        }
        ohlc_ts_list = [
            self._entry_timestamp_ms(r)
            for r in base_rows
            if isinstance(r, (list, tuple))
            and len(r) >= 3
            and (safe_pos_float(r[2], default=0.0) or 0.0) > 0.0
        ]
        first_ohlc_ts = min(ohlc_ts_list, default=0)
        last_ohlc_ts = max(ohlc_ts_list, default=0)
        for r in self.cache.get(symbol) or ():
            ts_r = self._entry_timestamp_ms(r)
            if ts_r <= 0:
                continue
            is_ohlc_r = (
                isinstance(r, (list, tuple))
                and len(r) >= 3
                and (safe_pos_float(r[2], default=0.0) or 0.0) > 0.0
            )
            if not ohlc_ts_list or ts_r < first_ohlc_ts or ts_r >= last_ohlc_ts - 15 * 60 * 1000:
                existing = by_ts.get(ts_r)
                if existing is None or (
                    is_ohlc_r
                    and (not isinstance(existing, (list, tuple)) or len(existing) < 3)
                ):
                    by_ts[ts_r] = list(r) if isinstance(r, tuple) else r
        merged_pts = [by_ts[k] for k in sorted(by_ts)]
        self.cache[symbol] = merged_pts
        return merged_pts

    def refresh_equity_ohlc_series(
        self,
        *,
        now_ts: Optional[float] = None,
        force: bool = False,
    ) -> bool:
        """Refresh 5d/5m OHLCV bars in ``cacheManager`` for equity symbols and publish ``cache_ohlc_24h_<SYM>.json`` + ``365Price`` seed."""
        if self.read_only or self.append_persist:
            return False
        eq_symbols = [s for s in self.symbols if is_equity_symbol(s)]
        if not eq_symbols:
            return False
        eff_now = resolve_now_ts(now_ts)
        eff_now_ms = math.trunc(eff_now * 1000.0)
        refresh_map = getattr(self, "_last_eq_ohlc_refresh_ts", None)
        if not isinstance(refresh_map, dict):
            refresh_map = {}
            self._last_eq_ohlc_refresh_ts = refresh_map
        base_cachedb = (
            cachedb_root_for_file(self.filename)
            if self.filename
            else ""
        ) or (os.path.dirname(os.path.abspath(self.filename)) if self.filename else "")
        any_refreshed = False
        for sym_eq in eq_symbols:
            with self.lock:
                cur_rows = list(self.cache.get(sym_eq) or ())
            has_pos_vol_ohlc = any(
                isinstance(r, (list, tuple))
                and len(r) >= 3
                and (safe_pos_float(r[2], default=0.0) or 0.0) > 0.0
                for r in cur_rows
            )
            eq_rth_open = bool(is_symbol_market_open(sym_eq, eff_now))
            min_interval = (
                self.EQUITY_OHLC_RTH_REFRESH_SEC
                if (eq_rth_open or len(cur_rows) < 10 or not has_pos_vol_ohlc)
                else self.EQUITY_OHLC_OFF_RTH_REFRESH_SEC
            )
            last_ref = refresh_map.get(sym_eq, -1e18)
            if not force and 0.0 <= (eff_now - last_ref) < min_interval:
                continue
            try:
                chart_res = _market_api.api.fetch_intraday_chart(
                    canonical_base_asset(sym_eq) or sym_eq,
                    rng="5d",
                    interval="5m",
                )
                pts_5d, pts_24h_bars, pts_24h, _, _, _, _ = normalize_yahoo_5d_chart_bars(
                    chart_res,
                    cur_rows,
                    eff_now,
                    eq_rth_open=eq_rth_open,
                )
            except Exception as exc:
                print(f"[{self.cls_name}][Error] equity OHLC refresh {sym_eq}: {exc}")
                continue
            refresh_map[sym_eq] = eff_now
            eff_pts = list(pts_24h or pts_24h_bars)
            if not eff_pts and not pts_5d:
                continue
            if eff_pts:
                with self.lock:
                    merged_pts = self._merge_ohlc_rows_locked(sym_eq, eff_pts)
                    last_pt_ts = (
                        self._entry_timestamp_ms(merged_pts[-1]) if merged_pts else eff_now_ms
                    ) or eff_now_ms
                    prev_ft = safe_nonneg_int(self.fetchtime_per_symbol.get(sym_eq), default=0) or 0
                    new_ft = max(prev_ft + 1, eff_now_ms, last_pt_ts)
                    self.fetchtime_per_symbol[sym_eq] = new_ft
                    self._rest_sync_cursor_per_symbol[sym_eq] = (
                        safe_pos_int(self._rest_sync_cursor_per_symbol.get(sym_eq), default=new_ft) or new_ft
                    )
                self._trim_old_data(sym_eq)
                if base_cachedb:
                    try:
                        ohlc24_target = str(
                            resolve_cachedb_filepath(
                                os.path.join(base_cachedb, cache_ohlc_24h_filename(sym_eq))
                            )
                        )
                        with generation_file_lock(ohlc24_target):
                            save_json_state(
                                ohlc24_target,
                                {"ts": eff_now, "symbol": sym_eq, "points": eff_pts},
                            )
                    except Exception as exc:
                        print(f"[{self.cls_name}][Error] save cache_ohlc_24h {sym_eq}: {exc}")
                any_refreshed = True
            if pts_5d and base_cachedb:
                try:
                    p365_target = str(
                        resolve_cachedb_filepath(
                            os.path.join(base_cachedb, Cache365PriceManager.disk_filename_for_symbol(sym_eq))
                        )
                    )
                    Cache365PriceManager.seed_initial_archive_if_missing(p365_target, sym_eq, pts_5d)
                except Exception as exc:
                    print(f"[{self.cls_name}][Error] seed 365Price {sym_eq}: {exc}")
        return any_refreshed

    def _persist_periodic_state(self, sync_complete):
        if self.save_state and not self.append_persist:
            with self.lock:
                self._drain_deferred_price_ticks_locked()
                persisted_max_ts = self._persisted_max_ts()
                mem_max_ts = self._mem_max_ts()
                if (
                    persisted_max_ts > mem_max_ts
                    and mem_max_ts <= self._last_disk_sync_max_ts
                ):
                    self._reload_from_disk()
                    if not (
                        is_live_network_backfill_enabled()
                        and any(is_equity_symbol(s) for s in self.symbols)
                    ):
                        self._publish_periodic_sync_health(sync_complete, True)
                        return True
            if is_live_network_backfill_enabled():
                self.refresh_equity_ohlc_series()
        return super()._persist_periodic_state(sync_complete)

    def _before_snapshot_commit_locked(self, persisted_max_ts: int) -> None:
        """Preserve 3-element OHLCV bars written to disk by enrichment while retaining recent in-memory ticks."""
        if persisted_max_ts <= self._last_disk_sync_max_ts:
            return
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
            disk_ft = (
                data.get("fetchtime")
                if isinstance(data, dict) and isinstance(data.get("fetchtime"), dict)
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
                self._merge_ohlc_rows_locked(s, disk_rows)
                ft_cand = safe_pos_int(disk_ft.get(s), default=0) or 0
                if ft_cand > safe_nonneg_int(self.fetchtime_per_symbol.get(s), default=0):
                    self.fetchtime_per_symbol[s] = ft_cand
                if hasattr(self, "_trim_old_data"):
                    self._trim_old_data(s)
        except Exception as exc:
            print(f"[{self.cls_name}][Error] merge disk OHLC {self.filename}: {exc}")

    def _filter_reloaded_items_locked(self, items: dict) -> dict:
        filtered = {s: v for s, v in items.items() if s in self.symbols}
        with self._deferred_ticks_lock:
            for sym_k, entries in filtered.items():
                if isinstance(entries, list) and entries:
                    last_ts = self._entry_timestamp_ms(entries[-1])
                    if last_ts > self._last_accepted_tick_ts_ms.get(sym_k, 0):
                        self._last_accepted_tick_ts_ms[sym_k] = last_ts
        return filtered

    def _sync_last_accepted_tick_ts_locked(self) -> None:
        with self._deferred_ticks_lock:
            for sym_k, entries in self.cache.items():
                if isinstance(entries, list) and entries:
                    last_ts = self._entry_timestamp_ms(entries[-1])
                    if last_ts > self._last_accepted_tick_ts_ms.get(sym_k, 0):
                        self._last_accepted_tick_ts_ms[sym_k] = last_ts

    def _drain_deferred_price_ticks_locked(self) -> None:
        """Apply ticks buffered while ``self.lock`` was held by persistence I/O."""
        with self._deferred_ticks_lock:
            if not self._deferred_price_ticks:
                return
            pending = list(self._deferred_price_ticks)
            self._deferred_price_ticks.clear()
        self._ensure_fetchtimes_initialized()
        grouped: dict[str, list[list]] = {}
        for sym_k, ts_val, price_val in pending:
            if sym_k not in self.symbols:
                continue
            entries = self.cache.get(sym_k)
            last_mem_ts = self._entry_timestamp_ms(entries[-1]) if entries else 0
            if sym_k in grouped and grouped[sym_k]:
                last_mem_ts = max(last_mem_ts, grouped[sym_k][-1][0])
            if ts_val < last_mem_ts:
                continue
            grouped.setdefault(sym_k, []).append([ts_val, price_val])
            with self._deferred_ticks_lock:
                if ts_val > self._last_accepted_tick_ts_ms.get(sym_k, 0):
                    self._last_accepted_tick_ts_ms[sym_k] = ts_val
        for sym_k, batch in grouped.items():
            self.update_cache_per_symbol(sym_k, batch)
            self._trim_old_data(sym_k)

    def _before_state_persist_locked(self) -> None:
        self._drain_deferred_price_ticks_locked()

    def _after_state_persisted(self) -> None:
        with self._deferred_ticks_lock:
            if not self._deferred_price_ticks:
                return
        with self.lock:
            self._drain_deferred_price_ticks_locked()

    def on_price_update(self, symbol: str, ts_ms: int, price: float):
        """Receive every new WebSocket or HTTP price from CacheCurrentPriceManager without blocking on disk I/O."""
        if symbol not in self.symbols:
            return
        if self.lock.acquire(blocking=False):
            try:
                self._drain_deferred_price_ticks_locked()
                tick = self._validate_monotonic_tick(symbol, ts_ms, price)
                if tick is None:
                    return
                ts_val, price_val = tick
                with self._deferred_ticks_lock:
                    if ts_val > self._last_accepted_tick_ts_ms.get(symbol, 0):
                        self._last_accepted_tick_ts_ms[symbol] = ts_val
                self._ensure_fetchtimes_initialized()
                self.update_cache_per_symbol(symbol, [[ts_val, price_val]])
                self._trim_old_data(symbol)
            finally:
                self.lock.release()
            self._notify_price_subscribers(symbol, ts_val, price_val)
            return

        ts_val = safe_pos_int(ts_ms, default=None)
        price_val = safe_pos_float(price, default=None)
        if ts_val is None or price_val is None:
            return
        with self._deferred_ticks_lock:
            if ts_val < self._last_accepted_tick_ts_ms.get(symbol, 0):
                return
            self._last_accepted_tick_ts_ms[symbol] = ts_val
            self._deferred_price_ticks.append((symbol, ts_val, price_val))
        self._notify_price_subscribers(symbol, ts_val, price_val)

    def _trim_old_data(self, symbol):
        """Remove entries older than ``KEEP_HOURS`` efficiently on every tick."""
        cutoff_ms = math.trunc((_now_time() - self.KEEP_HOURS * 3600) * 1000)
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
        cutoff_ms = math.trunc((_now_time() - sec_val) * 1000)
        with self.lock:
            self._drain_deferred_price_ticks_locked()
            entries = casefold_get(self.cache, symbol, [])
            if not entries or not isinstance(entries, list):
                return []
            idx = bisect.bisect_left(
                entries, cutoff_ms, key=self._entry_timestamp_ms
            )
            return entries[idx:]

    def _should_poll(self):
        return False

    @classmethod
    def persist_external_ohlc_series(
        cls,
        target_path: str,
        symbol: str,
        points: list,
        now_ts: float,
        *,
        existing_obj: Optional[dict] = None,
    ) -> None:
        """Persist 24h OHLCV series into ``cache_24_price_<SYM>.json`` and ``.meta`` transactionally under ``.generation.lock``."""
        if not target_path or not points:
            return
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
        with FileLock(generation_lock_path(target_path)):
            disk_obj = (
                existing_obj
                if isinstance(existing_obj, dict)
                else (
                    read_json_dict(target_path, recover_previous=True, repair_files=True)
                    if (os.path.exists(target_path) or os.path.exists(previous_generation_path(target_path)))
                    else {}
                )
            )
            now_ms = math.trunc(now_ts * 1000)
            last_pt_ts = (
                safe_pos_int(points[-1][0], default=now_ms) or now_ms
                if isinstance(points[-1], (list, tuple)) and points[-1]
                else now_ms
            )
            meta_path = meta_file_path(target_path)
            prev_meta = read_json_dict(meta_path) if os.path.exists(meta_path) else {}
            prev_ft = (
                prev_meta.get("fetchtime")
                if isinstance(prev_meta.get("fetchtime"), dict)
                else (
                    disk_obj.get("fetchtime")
                    if isinstance(disk_obj, dict) and isinstance(disk_obj.get("fetchtime"), dict)
                    else {}
                )
            )
            prev_rc = (
                prev_meta.get("rest_sync_cursor")
                if isinstance(prev_meta.get("rest_sync_cursor"), dict)
                else (
                    disk_obj.get("rest_sync_cursor")
                    if isinstance(disk_obj, dict) and isinstance(disk_obj.get("rest_sync_cursor"), dict)
                    else {}
                )
            )
            prev_cnts = prev_meta.get("counts") if isinstance(prev_meta.get("counts"), dict) else {}
            eff_ft_ts = max(
                (safe_pos_int(prev_ft.get(symbol), default=0) or 0) + 1,
                (safe_pos_int(prev_meta.get("max_ts"), default=0) or 0) + 1,
                now_ms,
                last_pt_ts,
            )
            ft_map = {**prev_ft, symbol: eff_ft_ts}
            rc_map = {**prev_rc, symbol: safe_pos_int(prev_rc.get(symbol), default=eff_ft_ts) or eff_ft_ts}
            payload = {"items": {symbol: points}, "fetchtime": ft_map, "rest_sync_cursor": rc_map}
            meta_payload = {
                "fetchtime": ft_map,
                "rest_sync_cursor": rc_map,
                "counts": {**prev_cnts, symbol: len(points)},
                "max_ts": max((safe_pos_int(v, default=0) or 0 for v in ft_map.values()), default=eff_ft_ts),
                "saved_at": math.trunc(_now_time() * 1000),
            }

            def _write_meta_ok() -> bool:
                write_meta_json(target_path, meta_payload)
                return True

            commit_file_and_meta(
                target_path,
                lambda: atomic_write_json(target_path, payload, indent=1),
                _write_meta_ok,
            )

    @staticmethod
    def read_ssot_24price_entries(
        c24_path: str,
        cand: str,
        *,
        candidates: Optional[tuple[str, ...] | list[str]] = None,
        allow_single_item_fallback: bool = False,
    ) -> tuple[Optional[dict], Optional[list]]:
        """Read raw 24-hour price entries for ``cand`` from ``c24_path`` (including ``.previous`` recovery)."""
        if not c24_path:
            return None, None
        raw = read_json_dict_recovering(c24_path, default=None)
        if not isinstance(raw, dict):
            return None, None
        items = raw.get("items")
        if isinstance(items, list):
            return (raw, items) if items else (raw, None)
        target_map = items if isinstance(items, dict) else raw
        entries = casefold_get(target_map, cand)
        if not (isinstance(entries, list) and entries) and candidates:
            for sc in candidates:
                cand_rows = casefold_get(target_map, sc)
                if isinstance(cand_rows, list) and cand_rows:
                    entries = cand_rows
                    break
        if (
            not (isinstance(entries, list) and entries)
            and allow_single_item_fallback
            and isinstance(items, dict)
            and len(items) == 1
        ):
            only_val = next(iter(items.values()))
            if isinstance(only_val, list) and only_val:
                entries = only_val
        return (raw, entries) if isinstance(entries, list) and entries else (raw, None)

    @classmethod
    def read_ssot_price_history(
        cls,
        candidates: tuple[str, ...] | list[str],
        lookback_hours: float,
        max_latest_age_sec: float = 120.0,
        *,
        custom_dir: Optional[str] = None,
    ) -> Optional[list[dict]]:
        """Read 24h tick history from ``cache_24_price_<SYM>.json`` when fresh and covering ``lookback_hours``."""
        lh_f = safe_pos_float(lookback_hours, default=None)
        max_latest_age = resolve_positive_max_age(max_latest_age_sec, 120.0)
        if lh_f is None or lh_f > 24.0 or max_latest_age is None or not candidates:
            return None
        now_s = _now_time()
        cutoff_ms = math.trunc((now_s - lh_f * 3600.0) * 1000)
        for cand in candidates:
            c24_fname = cache_24_price_filename(cand)
            c24_path = (
                str(resolve_cachedb_filepath(os.path.join(str(custom_dir), c24_fname)))
                if custom_dir
                else cache_path(c24_fname)
            )
            try:
                _, entries = cls.read_ssot_24price_entries(c24_path, cand)
                if not entries or len(entries) < 2:
                    continue
                valid = [
                    p
                    for e in entries
                    if (p := cls.parse_ssot_price_point(e)) is not None
                ]
                if len(valid) < 2:
                    continue
                valid.sort(key=lambda pt: pt[0])
                if not _is_fresh_obs_ts(valid[-1][0] / 1000.0, now_s, max_latest_age):
                    continue
                out = [
                    {"timestamp": ts, "price": px}
                    for ts, px in valid
                    if ts >= cutoff_ms
                ]
                if len(out) < 2:
                    continue
                if (
                    custom_dir is None
                    and lh_f > 0.25
                    and (out[-1]["timestamp"] - out[0]["timestamp"])
                    < (lh_f * 3600.0 * 1000.0 * 0.75)
                ):
                    continue
                return out
            except Exception:
                continue
        return None


class Cache365PriceManager(Cache24PriceManager):
    """Multi-month/multi-year dense archive variant (Level 3) using incremental JSONL persistence."""

    KEEP_HOURS = CM_LONG_ARCHIVE_MONTHS * 30 * 24
    RETENTION_DAYS = CM_LONG_ARCHIVE_MONTHS * 30
    LONG_TRIM_INTERVAL_SEC = 24 * 3600
    FILE_PREFIX = CACHE_365_PRICE_PREFIX
    FILE_EXT = "jsonl"

    @classmethod
    def seed_initial_archive_if_missing(cls, target_path: str, symbol: str, points: list) -> bool:
        """Bootstrap ``cache_365_price_<SYM>.jsonl`` once when missing on disk without overwriting an existing archive."""
        if not target_path or os.path.exists(target_path) or not isinstance(points, (list, tuple)) or not points:
            return False
        rows = [
            {"s": symbol, "i": [ts_i, px_f]}
            for pt in points
            if isinstance(pt, (list, tuple))
            and len(pt) >= 2
            and (ts_i := normalize_timestamp_ms(pt[0], default=None)) is not None
            and (px_f := safe_pos_float(pt[1], default=None)) is not None
        ]
        if not rows:
            return False
        atomic_write_jsonl(target_path, rows)
        return True

    def __init__(
        self,
        sync_ts,
        symbols,
        filename,
        api_client=None,
        read_only=False,
        *,
        current_price_getter=None,
        current_price_mgr=None,
        data_sync_mode: Optional[str] = None,
    ):
        self._last_long_trim = 0.0
        CachePriceManagerBase.__init__(
            self,
            sync_ts,
            symbols,
            filename,
            append_mode=True,
            api_client=api_client,
            append_persist=True,
            read_only=read_only,
            current_price_getter=current_price_getter,
            current_price_mgr=current_price_mgr,
            data_sync_mode=data_sync_mode,
        )
        self._filter_maps_to_configured_symbols_locked()

    def _on_stale_memory_detected_locked(self) -> None:
        pass

    def _before_snapshot_commit_locked(self, persisted_max_ts: int) -> None:
        pass

    def _should_compact_before_append_locked(self) -> bool:
        return False

    def _load_jsonl(self):
        """Load only a recent working set; the JSONL file remains the full archive."""
        with self.lock:
            self.cache = {}
            self._ensure_file_or_restore_previous_locked()
            if os.path.exists(self.filename):
                for raw_line in read_tail_lines(
                    self.filename, CM_LONG_ARCHIVE_MEMORY_ROWS
                ):
                    rec = parse_json_payload(raw_line, default=None)
                    if (
                        isinstance(rec, dict)
                        and rec.get("s") in self.symbols
                        and rec.get("i") is not None
                    ):
                        self.cache.setdefault(rec["s"], []).append(rec["i"])
            self._persisted_counts = {s: len(v) for s, v in self.cache.items()}
            if meta_payload := self._read_optional_meta_dict():
                self._hydrate_fetchtimes_from_mapping_locked(
                    meta_payload, filter_symbols=True, coerce_nonneg_int=True
                )
            if self.cache:
                for s_sym, rebuilt_ft in self.rebuild_fetchtimes().items():
                    self.fetchtime_per_symbol[s_sym] = max(
                        safe_nonneg_int(self.fetchtime_per_symbol.get(s_sym), default=0),
                        safe_nonneg_int(rebuilt_ft, default=0),
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

    @staticmethod
    def _merge_max_nonneg_int_map(dst: dict, src: dict) -> None:
        for s_sym, val in src.items():
            dst[s_sym] = max(
                safe_nonneg_int(dst.get(s_sym), default=0),
                safe_nonneg_int(val, default=0),
            )

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

    def _merge_live_delta_locked(
        self, delta_by_sym: dict[str, list], *, restore_state=None
    ) -> None:
        """Re-apply live ticks that arrived while an unlocked disk pass was running."""
        if restore_state is not None:
            self._restore_memory_state_locked(restore_state)
        for symbol, extra_items in delta_by_sym.items():
            if not extra_items:
                continue
            bucket = self.cache.setdefault(symbol, [])
            seen = {
                canonical_json_key(it)
                for it in bucket[-self.DEDUP_WINDOW :]
            }
            for item in extra_items:
                key = canonical_json_key(item)
                if key not in seen:
                    seen.add(key)
                    bucket.append(item)
            overflow = max(0, len(bucket) - CM_LONG_ARCHIVE_MEMORY_ROWS)
            if overflow:
                self.cache[symbol] = bucket[overflow:]
                self._persisted_counts[symbol] = max(
                    0, self._persisted_counts.get(symbol, 0) - overflow
                )

    def _snapshot_memory_state_with_item_ids_locked(self):
        original_state = self._snapshot_memory_state_locked()
        snap_item_ids = {
            id(item) for items in original_state[0].values() for item in items
        }
        return original_state, snap_item_ids

    def _reload_and_merge_live_delta_locked(
        self,
        snap_item_ids: set[int],
        original_fetchtimes,
        original_rest_cursors,
        *,
        min_ts_ms=None,
    ):
        post_delta = (
            self._collect_live_delta_locked(snap_item_ids)
            if min_ts_ms is None
            else self._collect_live_delta_locked(snap_item_ids, min_ts_ms=min_ts_ms)
        )
        live_fetchtimes = {
            k: safe_nonneg_int(v, default=0) for k, v in dict(original_fetchtimes).items()
        }
        self._merge_max_nonneg_int_map(live_fetchtimes, self.fetchtime_per_symbol)
        live_rest_cursors = {
            k: safe_nonneg_int(v, default=0) for k, v in dict(original_rest_cursors).items()
        }
        self._merge_max_nonneg_int_map(live_rest_cursors, self._rest_sync_cursor_per_symbol)
        self._load_jsonl()
        self._merge_max_nonneg_int_map(self.fetchtime_per_symbol, live_fetchtimes)
        self._merge_max_nonneg_int_map(self._rest_sync_cursor_per_symbol, live_rest_cursors)
        self._merge_live_delta_locked(post_delta)
        return post_delta

    def _write_unpersisted_and_live_delta(
        self, target, original_state, snap_item_ids: set[int], *, min_ts_ms=None
    ) -> bool:
        wrote = False
        eff_min_ts_ms = 0 if min_ts_ms is None else min_ts_ms
        original_cache, original_persisted_counts = original_state[0], original_state[1]
        for symbol, items in original_cache.items():
            start = min(max(0, original_persisted_counts.get(symbol, 0)), len(items))
            for item in items[start:]:
                if eff_min_ts_ms <= 0 or self._entry_timestamp_ms(item) >= eff_min_ts_ms:
                    target.write(format_jsonl_bucket_line(symbol, item))
                wrote = True
        with self.lock:
            delta_by_sym = (
                self._collect_live_delta_locked(snap_item_ids)
                if min_ts_ms is None
                else self._collect_live_delta_locked(snap_item_ids, min_ts_ms=min_ts_ms)
            )
            for symbol, extra_items in delta_by_sym.items():
                for item in extra_items:
                    target.write(format_jsonl_bucket_line(symbol, item))
                    wrote = True
        return wrote

    def maintain_append_persist(self):
        """Prune the full disk archive without holding ``self.lock`` during disk I/O."""
        if not self.save_state:
            return False
        cutoff_ms = math.trunc((_now_time() - self.RETENTION_DAYS * 86400) * 1000)
        if not os.path.exists(self.filename):
            return False
        changed = False
        backup_path = None
        original_state = None
        snap_item_ids: set[int] = set()
        try:
            with self.lock:
                self._drain_deferred_price_ticks_locked()
                original_state, snap_item_ids = self._snapshot_memory_state_with_item_ids_locked()

            backup_path = self.filename + ".maint_backup"
            self._snapshot_file_to(backup_path)
            with atomic_text_writer(self.filename) as target:
                for line_out, keep in iter_jsonl_retention_lines(
                    backup_path,
                    lambda rec: (
                        isinstance(rec.get("s"), str)
                        and self._entry_timestamp_ms(rec.get("i")) >= cutoff_ms
                    ),
                ):
                    if keep:
                        target.write(line_out)
                    else:
                        changed = True
                if self._write_unpersisted_and_live_delta(
                    target, original_state, snap_item_ids, min_ts_ms=cutoff_ms
                ):
                    changed = True

            with self.lock:
                post_delta = self._reload_and_merge_live_delta_locked(
                    snap_item_ids,
                    original_state[2],
                    original_state[3],
                    min_ts_ms=cutoff_ms,
                )
                if changed:
                    if not self._write_meta(self._build_basic_meta_locked(self._persisted_counts)):
                        self._merge_live_delta_locked(post_delta, restore_state=original_state)
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
                self._merge_live_delta_locked(
                    self._collect_live_delta_locked(snap_item_ids),
                    restore_state=original_state,
                )
                if backup_path is not None and os.path.exists(backup_path):
                    try:
                        self._restore_previous_generation(backup_path)
                    except OSError:
                        pass
            builtins.print(
                f"[{self.cls_name}][Error] streaming maintenance {self.filename}: {exc}"
            )
            return False
        finally:
            self._after_state_persisted()

    def compact_jsonl(self):
        """Never rebuild an existing long archive from its bounded memory tail."""
        if os.path.exists(self.filename):
            return self.maintain_append_persist()
        return super().compact_jsonl()

    def _rotate_disk_archive(self):
        """Archive an oversized file while retaining a complete-line tail without holding ``self.lock`` during disk I/O."""
        archive = self._archive_snapshot_path()
        with self.lock:
            original_state, snap_item_ids = self._snapshot_memory_state_with_item_ids_locked()
        try:
            self._snapshot_file_to(archive)
        except OSError as exc:
            return self._log_tagged_false("maintain", f"archiving failed: {exc}")
        try:
            archive_size = os.path.getsize(archive)
            keep_bytes = max(1, math.trunc(archive_size * self.ROTATE_KEEP_FRACTION))
            decoded_tail = read_tail_complete_lines_text(archive, keep_bytes)
            with atomic_text_writer(self.filename) as target:
                target.write(decoded_tail)
                self._write_unpersisted_and_live_delta(
                    target, original_state, snap_item_ids
                )
            with self.lock:
                self._reload_and_merge_live_delta_locked(
                    snap_item_ids, original_state[2], original_state[3]
                )
                meta_ok = self._write_meta(self._build_basic_meta_locked(self._persisted_counts))
        except Exception as exc:
            builtins.print(
                f"[{self.cls_name}][maintain] rotation rewrite failed: {exc}"
            )
            meta_ok = False
        if not meta_ok:
            with self.lock:
                self._merge_live_delta_locked(
                    self._collect_live_delta_locked(snap_item_ids),
                    restore_state=original_state,
                )
                return self._restore_archive_after_failed_rotation_locked(archive)
        self._compress_and_prune_archive(archive)
        return True


class CacheCurrentPriceManager(CachePriceManagerBase):
    """Maintain the latest timestamped price per symbol (Level 3, WebSocket-first with HTTP fallback)."""

    WS_TIMEOUT_SEC = PRICE_WS_TIMEOUT_SEC
    STALE_THRESHOLD_MS = PRICE_FRESHNESS_MAX_AGE_MS
    FREQ_WINDOW_SEC = 60
    DEFAULT_FILENAME = CACHE_CURRENT_PRICE_FILENAME
    _flush_before_poll = True

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
        *,
        data_sync_mode: Optional[str] = None,
    ):
        self._ws_manager = ws_manager
        self._ws_managers = [ws_manager] if ws_manager is not None else []
        self._ws_default_event_ts = 0.0
        self._ws_last_event_ts_value = 0.0
        self._ws_last_event_ts_per_symbol: dict = {}
        self._ws_stream_event_ts_per_symbol: dict = {}
        self._delegated_poller_symbols: set[str] = set()
        self._update_timestamps: dict = defaultdict(deque)
        self._rth_empty_backoff_ts: dict = {}
        self._last_http_poll_ts_per_symbol: dict[str, float] = {}
        self._fallback_pool_lock = threading.Lock()
        self._fallback_pool: Optional[_futures.ThreadPoolExecutor] = None
        self._first_sleep = True
        self._tick_ipc_lock = threading.Lock()
        self._tick_ipc_stop = threading.Event()
        self._tick_ipc_server = None
        self._tick_ipc_bound_path: Optional[str] = None
        self._tick_ipc_clients: list = []
        self._tick_ipc_thread: Optional[threading.Thread] = None
        self._tick_sub_sock = None
        self._tick_sub_thread: Optional[threading.Thread] = None
        self.market_api = (
            market_api
            if market_api is not None
            else (
                api_client
                if api_client is not None and callable(getattr(api_client, "get_current_price", None))
                else _market_api.api
            )
        )
        self._provider_names = {}
        self.bind_providers(provider_names)
        super().__init__(
            sync_ts,
            symbols,
            filename,
            append_mode=False,
            api_client=api_client,
            read_only=read_only,
            data_sync_mode=data_sync_mode,
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

    def _is_equity_instrument(self, symbol: Optional[str]) -> bool:
        if not isinstance(symbol, str) or not symbol.strip():
            return False
        return bool(
            is_equity_venue(self.provider_name_for(symbol) or "")
            or is_equity_symbol(symbol)
            or has_equity_suffix(symbol)
        )

    def ws_timeout_sec_for(self, symbol: Optional[str]) -> float:
        base_timeout = (
            safe_pos_float(self.WS_TIMEOUT_SEC, default=PRICE_WS_TIMEOUT_SEC)
            or PRICE_WS_TIMEOUT_SEC
        )
        if "WS_TIMEOUT_SEC" not in self.__dict__ and self._is_equity_instrument(symbol):
            return max(
                base_timeout,
                round(SHORT_TREND_FRESHNESS_MAX_AGE_SEC * (2.0 / 3.0), 3),
            )
        return base_timeout

    def stale_threshold_ms_for(self, symbol: Optional[str]) -> int:
        base_ms = (
            safe_pos_int(self.STALE_THRESHOLD_MS, default=PRICE_FRESHNESS_MAX_AGE_MS)
            or PRICE_FRESHNESS_MAX_AGE_MS
        )
        if "STALE_THRESHOLD_MS" not in self.__dict__ and self._is_equity_instrument(symbol):
            return max(base_ms, math.trunc(SHORT_TREND_FRESHNESS_MAX_AGE_SEC * 1000))
        return base_ms

    def _is_ws_ts_map_healthy(
        self, ts_map: dict, symbol: Optional[str] = None
    ) -> bool:
        now = _now_time()
        if symbol is not None:
            sym_u, sym_clean = _norm_symbol_keys(symbol)
            last_ts = next(
                (val for cand in (symbol, sym_u, sym_clean) if cand and (val := ts_map.get(cand)) is not None),
                self._ws_default_event_ts,
            )
            return -5.0 <= (now - last_ts) < self.ws_timeout_sec_for(symbol)
        if not self.symbols:
            return -5.0 <= (now - self._ws_last_event_ts_value) < self.WS_TIMEOUT_SEC
        return all(self._is_ws_ts_map_healthy(ts_map, s) for s in self.symbols)

    def _ws_is_healthy(self, symbol: Optional[str] = None) -> bool:
        return self._is_ws_ts_map_healthy(self._ws_last_event_ts_per_symbol, symbol)

    def is_real_ws_healthy(self, symbol: Optional[str] = None) -> bool:
        return self._is_ws_ts_map_healthy(
            self._ws_stream_event_ts_per_symbol, symbol
        )

    def mark_symbol_ws_event(self, symbol: str, now_ts: Optional[float] = None) -> None:
        ts_val = safe_nonneg_float(now_ts, default=None)
        if ts_val is None:
            ts_val = _now_time()
        self._ws_last_event_ts_per_symbol[symbol] = ts_val

    def push_price(
        self, symbol: str, price: float, ts_ms: Optional[int] = None
    ) -> None:
        if isinstance(ts_ms, bool):
            return
        price_val = safe_pos_float(price, default=None)
        ts_val = (
            safe_pos_int(ts_ms, default=None)
            if ts_ms is not None
            else math.trunc(_now_time() * 1000)
        )
        if price_val is None or ts_val is None or ts_val <= 0:
            return
        self._ensure_fetchtimes_initialized()
        now = _now_time()
        cutoff = now - self.FREQ_WINDOW_SEC
        with self.lock:
            dq = self._update_timestamps[symbol]
            dq.append(now)
            while dq and dq[0] < cutoff:
                dq.popleft()
        self.update_cache_per_symbol(symbol, [[ts_val, price_val]])
        if self._tick_ipc_clients:
            self._broadcast_tick_ipc(symbol, ts_val, price_val)
        self._notify_price_subscribers(symbol, ts_val, price_val)

    @staticmethod
    def extract_price_and_ts(items):
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
        price, ts_ms = self.extract_price_and_ts(items)
        if price is None:
            return
        with self.lock:
            entries = self.cache.get(symbol)
            existing_ts = (
                self._entry_timestamp_ms(entries[0]) if entries else 0
            )
        sym_stale_ms = self.stale_threshold_ms_for(symbol)
        ooo_limit_ms = (
            sym_stale_ms
            if self.is_real_ws_healthy(symbol)
            else max(sym_stale_ms, 5_000)
        )
        if existing_ts > 0 and (existing_ts - ts_ms) > ooo_limit_ms:
            return
        now = _now_time()
        self._ws_last_event_ts_value = now
        self._ws_last_event_ts_per_symbol[symbol] = now
        self._ws_stream_event_ts_per_symbol[symbol] = now
        self._ws_stream_event_ts_per_symbol[str(symbol).strip().upper()] = now
        self.push_price(symbol, price, ts_ms=ts_ms)

    def _get_or_create_fallback_pool(self, worker_hint: int) -> _futures.ThreadPoolExecutor:
        with self._fallback_pool_lock:
            if self._fallback_pool is None:
                self._fallback_pool = _futures.ThreadPoolExecutor(
                    max_workers=max(2, min(6, worker_hint)),
                    thread_name_prefix="CPFallbackFetch",
                )
            return self._fallback_pool

    def _shutdown_fallback_pool(self) -> None:
        with self._fallback_pool_lock:
            pool = self._fallback_pool
            self._fallback_pool = None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
            try:
                for th in list(getattr(pool, "_threads", ())):
                    _futures.thread._threads_queues.pop(th, None)
            except Exception:
                pass

    def query_remote_and_update_cache(self):
        if not self.fetchtime_per_symbol:
            self._ensure_fetchtimes_initialized()
            self._rest_sync_cursor_per_symbol = dict(self.fetchtime_per_symbol)
        delegated = getattr(self, "_delegated_poller_symbols", ())
        is_live_daemon_poll = bool(
            self.save_state
            and self.market_api is _market_api.api
            and is_live_network_backfill_enabled()
            and not self._stop_event.is_set()
        )
        min_poll_gap = max(
            0.45,
            min(safe_pos_float(self.sync_ts, default=0.8) or 0.8, 0.75),
        )
        pending_polls: list[tuple[str, Any, bool]] = []
        for symbol in list(self.symbols):
            if delegated and (
                symbol in delegated or str(symbol).strip().upper() in delegated
            ):
                continue
            if self._ws_is_healthy(symbol):
                continue
            is_rth_sym = self._is_equity_instrument(symbol)
            if is_rth_sym and is_rth_gate_enabled(
                getattr(self, "_enforce_rth_gate", None), rth_fn=is_us_rth_open
            ):
                now_s = _now_time()
                if (last_eb := self._rth_empty_backoff_ts.get(symbol, 0.0)) > 0.0 and 0.0 <= (now_s - last_eb) < 60.0:
                    continue
                if not is_us_rth_open(now_s):
                    with self.lock:
                        existing_entries = self.cache.get(symbol)
                    if (
                        existing_entries
                        and self.extract_price_and_ts(existing_entries)[0] is not None
                    ):
                        continue
            if is_live_daemon_poll and not is_rth_sym:
                now_s = _now_time()
                last_poll = self._last_http_poll_ts_per_symbol.get(symbol, 0.0)
                if last_poll > 0.0 and 0.0 <= (now_s - last_poll) < min_poll_gap:
                    with self.lock:
                        _, cur_ts_ms = self.extract_price_and_ts(self.cache.get(symbol))
                    if cur_ts_ms is not None and (now_s * 1000.0 - cur_ts_ms) < 1_800.0:
                        continue
            startTime = self.fetchtime_per_symbol.get(
                symbol, self.fallback_time_default
            )
            pending_polls.append((symbol, startTime, is_rth_sym))

        if not pending_polls:
            return True

        if not is_live_daemon_poll:
            for symbol, startTime, is_rth_sym in pending_polls:
                new_items = self._fetch_and_apply_remote_items(symbol, startTime)
                if is_rth_sym:
                    if not new_items:
                        self._rth_empty_backoff_ts[symbol] = _now_time()
                    else:
                        self._rth_empty_backoff_ts.pop(symbol, None)
            return True

        pool = self._get_or_create_fallback_pool(len(pending_polls))
        poll_deadline = min(
            safe_pos_float(PRICE_FALLBACK_HTTP_DEADLINE_SEC, default=1.2) or 1.2,
            1.2,
        )
        fut_map = {
            pool.submit(self.get_remote_items, sym_item, st_item): (sym_item, st_item, is_rth_sym)
            for sym_item, st_item, is_rth_sym in pending_polls
        }
        done, not_done = _futures.wait(fut_map.keys(), timeout=poll_deadline)
        for fut in done:
            sym_done, _, is_rth_done = fut_map[fut]
            try:
                new_items = fut.result()
            except Exception as exc:
                self._log_remote_fetch_error("HTTP fetch", sym_done, exc)
                new_items = []
            if new_items:
                self._persist_items(sym_done, new_items)
                self._last_http_poll_ts_per_symbol[sym_done] = _now_time()
                if is_rth_done:
                    self._rth_empty_backoff_ts.pop(sym_done, None)
            elif is_rth_done:
                self._rth_empty_backoff_ts[sym_done] = _now_time()
        for fut in not_done:
            fut.cancel()
        return True

    def get_remote_items(self, symbol, startTime):
        try:
            provider_name = self.provider_name_for(symbol)
            with live_price_fetch_context():
                if provider_name is None:
                    price = self.market_api.get_current_price(symbol=symbol)
                else:
                    try:
                        price = self.market_api.get_current_price(
                            symbol=symbol, provider_name=provider_name
                        )
                    except TypeError:
                        if self.market_api is _market_api.api:
                            raise
                        price = self.market_api.get_current_price(symbol=symbol)
            price_val = safe_pos_float(price, default=None)
            if price_val is None:
                return []
            ts_ms = math.trunc(_now_time() * 1000)
            return [[ts_ms, price_val]]
        except Exception as e:
            return self._log_remote_fetch_error("HTTP fetch", symbol, e)

    def _persist_items(self, symbol, new_items):
        price, ts_ms = self.extract_price_and_ts(new_items)
        if price is None:
            return
        self.push_price(symbol, price, ts_ms=ts_ms)

    def _should_poll(self):
        return not self._ws_is_healthy()

    def _next_sync_wait_sec(self) -> float:
        base_wait = safe_nonneg_float(self.sync_ts, default=0.8)
        if base_wait is None:
            base_wait = 0.8
        if not self.save_state or base_wait > 5.0:
            return base_wait
        wait_sec = min(base_wait, 0.35)
        now = _now_time()
        delegated = getattr(self, "_delegated_poller_symbols", ())
        for sym_item in list(self.symbols):
            if delegated and (
                sym_item in delegated or str(sym_item).strip().upper() in delegated
            ):
                continue
            if self._is_equity_instrument(sym_item):
                continue
            sym_u, sym_clean = _norm_symbol_keys(sym_item)
            last_ts = next(
                (
                    val
                    for cand in (sym_item, sym_u, sym_clean)
                    if cand and (val := self._ws_last_event_ts_per_symbol.get(cand)) is not None
                ),
                self._ws_default_event_ts,
            )
            if last_ts > 0.0:
                rem = self.ws_timeout_sec_for(sym_item) - (now - last_ts)
                if rem > 0.0:
                    wait_sec = min(wait_sec, max(0.05, rem + 0.02))
        return wait_sec

    def _sample_span_and_count(self, symbol: str) -> tuple[Optional[float], int]:
        with self.lock:
            if symbol in self._update_timestamps:
                dq = self._update_timestamps.get(symbol)
            else:
                sym_u, sym_clean = _norm_symbol_keys(symbol)
                dq = casefold_get(self._update_timestamps, sym_u) or casefold_get(self._update_timestamps, sym_clean)
            samples = list(dq) if dq else []
        if len(samples) < 2:
            return None, len(samples)
        return safe_pos_float(samples[-1] - samples[0], default=None), len(samples)

    def get_sample_rate(self, symbol: str, fallback: float = 1.0) -> float:
        fb = safe_pos_float(fallback, default=1.0) or 1.0
        span, n_samples = self._sample_span_and_count(symbol)
        if span is None:
            return fb
        return span / (n_samples - 1)

    def get_sustained_sample_rate(
        self, symbol: str, *, min_span_sec: float = 5.0
    ) -> Optional[float]:
        """Return observed sample rate when at least two ticks span ``>= min_span_sec``, else ``None``."""
        span, n_samples = self._sample_span_and_count(symbol)
        min_span_f = safe_nonneg_float(min_span_sec, default=None)
        if span is None or min_span_f is None or span < min_span_f:
            return None
        return safe_pos_float(self.get_sample_rate(symbol, fallback=None), default=span / (n_samples - 1))

    def get_cached_price_observation(
        self, symbol: str, max_age_sec: float, *, now_s: Optional[float] = None
    ):
        max_age = safe_pos_float(max_age_sec, default=None)
        if max_age is None:
            raise ValueError("max_age_sec must be finite and positive")
        _, sym_clean = _norm_symbol_keys(symbol)
        with self.lock:
            entries = casefold_get(self.cache, symbol) or casefold_get(
                self.cache, sym_clean
            )
            cached_price, last_ts = self.extract_price_and_ts(entries)
        if cached_price is None or last_ts is None:
            return None
        last_ts_f = safe_pos_float(last_ts, default=None)
        observed_at = safe_pos_float(last_ts_f / 1000.0, default=None) if last_ts_f is not None else None
        price = safe_pos_float(cached_price, default=None)
        if observed_at is None or price is None:
            return None
        eff_now = _now_time() if now_s is None else resolve_now_ts(now_s)
        age = eff_now - observed_at
        if age < -5.0 or age > max_age:
            return None
        return observed_at, price

    def _resolve_and_load_cached_entry(self, symbol: str, now_ms: int):
        with self.lock:
            sym_u, sym_clean = _norm_symbol_keys(symbol)
            configured = getattr(self, "symbols", ())
            target_sym = next(
                (
                    cand
                    for cand in (
                        *(c for c in (symbol, sym_u, sym_clean) if c in self.cache),
                        *(k for k in self.cache if isinstance(k, str) and k.upper() in (sym_u, sym_clean)),
                        *((sym_clean,) if sym_clean in configured else ()),
                        *(s for s in configured if isinstance(s, str) and s.upper() in (sym_u, sym_clean)),
                    )
                ),
                symbol,
            )
            cached_price, last_ts = self.extract_price_and_ts(self.cache.get(target_sym))
            entry = [last_ts, cached_price] if cached_price is not None else None
        return target_sym, entry, ((now_ms - last_ts) if last_ts else None)

    def get_price(self, symbol: str):
        now_ms = math.trunc(_now_time() * 1000)
        sym_mod = _cm_override("sym", sym)

        def _eff_stale_ms(sym_key: str, cur_entry) -> int:
            thr_ms = self.stale_threshold_ms_for(sym_key)
            if (
                cur_entry is not None
                and is_rth_gate_enabled(
                    getattr(self, "_enforce_rth_gate", None), rth_fn=is_us_rth_open
                )
                and hasattr(sym_mod, "is_symbol_market_open")
                and not sym_mod.is_symbol_market_open(sym_key)
            ):
                thr_ms = max(thr_ms, 3_600_000)
            return thr_ms

        target_sym, entry, age_ms = self._resolve_and_load_cached_entry(symbol, now_ms)
        stale_threshold_ms = _eff_stale_ms(target_sym, entry)
        if (
            entry is None
            or age_ms is None
            or age_ms > stale_threshold_ms
            or age_ms < -5_000
        ):
            if (
                getattr(self, "data_sync_mode", None) == self.MODE_DISK_READER
                and not self.save_state
            ):
                self.resync_mem_file()
                target_sym, entry, age_ms = self._resolve_and_load_cached_entry(symbol, now_ms)
                stale_threshold_ms = _eff_stale_ms(target_sym, entry)
        if (
            entry is None
            or age_ms is None
            or age_ms > stale_threshold_ms
            or age_ms < -5_000
        ):
            age = age_ms if age_ms is not None else -1
            print(f"[{self.cls_name}] {target_sym} stale ({age}ms) - forced HTTP fetch")
            new = self.get_remote_items(target_sym, None)
            if not new or self.extract_price_and_ts(new)[0] is None:
                return None
            self._persist_items(target_sym, new)
            self.save_state_to_file_if_enabled()
            _, entry, _ = self._resolve_and_load_cached_entry(target_sym, now_ms)
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
        if (
            not self.save_state
            and getattr(self, "data_sync_mode", None) == self.MODE_DISK_READER
        ):
            self.data_sync_mode = self.MODE_LIVE_READER
        ws_manager.subscribe(self)

    @property
    def tick_socket_path(self) -> str:
        return os.path.join(
            os.path.dirname(os.path.abspath(self.filename)),
            CACHE_CURRENT_PRICE_TICKS_SOCKET_FILENAME,
        )

    def _broadcast_tick_ipc(self, symbol: str, ts_ms: int, price: float) -> None:
        payload = f"{symbol}\t{ts_ms}\t{price}\n".encode("utf-8")
        dead = []
        with self._tick_ipc_lock:
            clients = list(self._tick_ipc_clients)
        for conn in clients:
            try:
                if conn.send(payload) < len(payload):
                    dead.append(conn)
            except OSError:
                dead.append(conn)
        if dead:
            with self._tick_ipc_lock:
                for conn in dead:
                    if conn in self._tick_ipc_clients:
                        self._tick_ipc_clients.remove(conn)
                    try:
                        conn.close()
                    except OSError:
                        pass

    def start_tick_fanout_server(self, sock_path: Optional[str] = None) -> Optional[str]:
        """Publish every fresh multi-venue tick over a non-blocking AF_UNIX stream socket."""
        target_path = os.path.abspath(sock_path or self.tick_socket_path) if hasattr(socket, "AF_UNIX") else ""
        if not target_path or len(target_path.encode("utf-8")) > 104:
            if target_path:
                print(f"[{self.cls_name}] tick socket path too long, skipping bind: {target_path}")
            return None
        with self._tick_ipc_lock:
            if self._tick_ipc_thread is not None and self._tick_ipc_thread.is_alive():
                return self._tick_ipc_bound_path
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            if os.path.lexists(target_path):
                probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                try:
                    probe.settimeout(0.05)
                    probe.connect(target_path)
                    return None
                except OSError:
                    try:
                        os.unlink(target_path)
                    except OSError:
                        return None
                finally:
                    try:
                        probe.close()
                    except OSError:
                        pass
            srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                srv.bind(target_path)
                os.chmod(target_path, 0o600)
                srv.listen(8)
                srv.settimeout(0.25)
            except OSError as exc:
                try:
                    srv.close()
                except OSError:
                    pass
                print(f"[{self.cls_name}] cannot bind tick socket {target_path}: {exc}")
                return None
            self._tick_ipc_stop.clear()
            self._tick_ipc_server = srv
            self._tick_ipc_bound_path = target_path

            def _accept_loop() -> None:
                while not self._stop_event.is_set() and not self._tick_ipc_stop.is_set():
                    try:
                        conn, _ = srv.accept()
                    except socket.timeout:
                        continue
                    except OSError:
                        break
                    try:
                        conn.setblocking(False)
                        now_ms = math.trunc(_now_time() * 1000)
                        with self.lock:
                            warm = [
                                (sym_k, ts_v, px_v)
                                for sym_k, entries in self.cache.items()
                                for px_v, ts_v in (self.extract_price_and_ts(entries),)
                                if px_v is not None
                                and ts_v is not None
                                and -5_000 <= (now_ms - ts_v) <= self.stale_threshold_ms_for(sym_k)
                            ]
                        if warm:
                            conn.send("".join(f"{s}\t{t}\t{p}\n" for s, t, p in warm).encode("utf-8"))
                        with self._tick_ipc_lock:
                            self._tick_ipc_clients.append(conn)
                    except OSError:
                        try:
                            conn.close()
                        except OSError:
                            pass

            th = threading.Thread(target=_accept_loop, name="PriceTickFanout", daemon=True)
            th.start()
            self._tick_ipc_thread = th
            return target_path

    def start_local_tick_stream(
        self, sock_path: Optional[str] = None
    ) -> Optional[threading.Thread]:
        """Subscribe in live_reader mode to the local producer AF_UNIX tick stream."""
        if not hasattr(socket, "AF_UNIX"):
            return None
        target_path = os.path.abspath(sock_path or self.tick_socket_path)
        if not self.save_state and getattr(self, "data_sync_mode", None) == self.MODE_DISK_READER:
            self.data_sync_mode = self.MODE_LIVE_READER
        with self._tick_ipc_lock:
            if self._tick_sub_thread is not None and self._tick_sub_thread.is_alive():
                return self._tick_sub_thread
            self._tick_ipc_stop.clear()

            def _read_loop() -> None:
                while not self._stop_event.is_set() and not self._tick_ipc_stop.is_set():
                    sock = None
                    try:
                        if not os.path.exists(target_path):
                            self._tick_ipc_stop.wait(0.2)
                            continue
                        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                        sock.settimeout(0.5)
                        sock.connect(target_path)
                        with self._tick_ipc_lock:
                            self._tick_sub_sock = sock
                        buf = b""
                        sym_lookup_key: Optional[tuple] = None
                        exact_lookup: dict[str, str] = {}
                        clean_lookup: dict[str, str] = {}
                        while not self._stop_event.is_set() and not self._tick_ipc_stop.is_set():
                            try:
                                chunk = sock.recv(4096)
                            except socket.timeout:
                                continue
                            if not chunk:
                                break
                            buf += chunk
                            if len(buf) > 65536:
                                buf = b""
                                break
                            while b"\n" in buf:
                                raw_line, buf = buf.split(b"\n", 1)
                                parts = raw_line.decode("utf-8", errors="ignore").split("\t")
                                if len(parts) != 3:
                                    continue
                                sym_raw, ts_s, px_s = parts
                                sym_u, sym_clean = _norm_symbol_keys(sym_raw)
                                if not sym_u:
                                    continue
                                configured = self.symbols
                                if configured:
                                    cfg_key = tuple(configured)
                                    if cfg_key != sym_lookup_key:
                                        rebuilt_exact: dict[str, str] = {}
                                        rebuilt_clean: dict[str, str] = {}
                                        for s in configured:
                                            if not isinstance(s, str):
                                                continue
                                            s_u = s.strip().upper()
                                            if s_u:
                                                rebuilt_exact.setdefault(s_u, s)
                                                s_clean = clean_symbol_key(s_u)
                                                if s_clean:
                                                    rebuilt_clean.setdefault(s_clean, s)
                                        exact_lookup = rebuilt_exact
                                        clean_lookup = rebuilt_clean
                                        sym_lookup_key = cfg_key
                                    canon_sym = exact_lookup.get(sym_u) or (
                                        clean_lookup.get(sym_clean) if sym_clean else None
                                    )
                                else:
                                    canon_sym = sym_raw.strip()
                                if not canon_sym:
                                    continue
                                ts_i = safe_pos_int(ts_s, default=None)
                                px_f = safe_pos_float(px_s, default=None)
                                if ts_i is not None and px_f is not None:
                                    tick_age_ms = _now_time() * 1000.0 - ts_i
                                    if -5_000.0 <= tick_age_ms <= self.stale_threshold_ms_for(canon_sym):
                                        self.on_items_update(canon_sym, [[ts_i, px_f]])
                    except OSError:
                        pass
                    finally:
                        with self._tick_ipc_lock:
                            if self._tick_sub_sock is sock:
                                self._tick_sub_sock = None
                        if sock is not None:
                            try:
                                sock.close()
                            except OSError:
                                pass
                    if not self._stop_event.is_set() and not self._tick_ipc_stop.is_set():
                        self._tick_ipc_stop.wait(0.2)

            th = threading.Thread(target=_read_loop, name="PriceTickSubscriber", daemon=True)
            th.start()
            self._tick_sub_thread = th
            return th

    def request_stop(self) -> None:
        super().request_stop()
        self._tick_ipc_stop.set()
        with self._tick_ipc_lock:
            srv = self._tick_ipc_server
            sub_sock = self._tick_sub_sock
            clients = list(self._tick_ipc_clients)
            self._tick_ipc_clients.clear()
        for sock_obj in (srv, sub_sock, *clients):
            if sock_obj is not None:
                try:
                    sock_obj.close()
                except OSError:
                    pass

    def shutdown(self, timeout=5.0):
        deadline = _monotonic_time() + safe_nonneg_float(timeout, default=5.0)
        self.request_stop()
        stopped = super().shutdown(timeout=max(0.0, deadline - _monotonic_time()))
        self._shutdown_fallback_pool()
        for attr in ("_tick_ipc_thread", "_tick_sub_thread"):
            th = getattr(self, attr, None)
            if th is not None and th is not threading.current_thread():
                th.join(timeout=max(0.0, deadline - _monotonic_time()))
            if th is None or not th.is_alive():
                setattr(self, attr, None)
            else:
                stopped = False
        with self._tick_ipc_lock:
            bound_path = self._tick_ipc_bound_path
            self._tick_ipc_bound_path = None
            self._tick_ipc_server = None
            self._tick_sub_sock = None
        if bound_path:
            try:
                if os.path.lexists(bound_path):
                    os.unlink(bound_path)
            except OSError:
                pass
        return stopped

    @classmethod
    def read_ssot_current_price_candidates(
        cls,
        candidates: tuple[str, ...] | list[str],
        price_path: str,
        max_age_sec: float,
        *,
        now_s: Optional[float] = None,
        use_memory_manager: bool = True,
        memory_manager: Any = None,
        reader_fn: Any = None,
    ) -> Optional[float]:
        """Resolve a fresh current price for ``candidates`` from in-memory ``CacheCurrentPriceManager`` or ``price_path``."""
        if (max_age := resolve_positive_max_age(max_age_sec, PRICE_FRESHNESS_MAX_AGE_SEC)) is None:
            return None
        norm_cands = (
            (candidates,)
            if isinstance(candidates, str)
            else (
                ()
                if isinstance(candidates, (bool, bytes, bytearray)) or not hasattr(candidates, "__iter__")
                else tuple(c for c in candidates if not isinstance(c, bool) and isinstance(c, str) and c.strip())
            )
        )
        if not norm_cands:
            return None
        eff_now = resolve_now_ts(now_s)
        if _parse_bool_flag(use_memory_manager, True):
            cpm = (
                memory_manager
                if memory_manager is not None
                else cls.resolve_active_instance("CurrentPrice", reader_fn=reader_fn)
            )
            if cpm is None:
                ro_cpm = _READONLY_PRICE_READERS.get("CurrentPrice")
                if getattr(ro_cpm, "data_sync_mode", None) in (
                    cls.MODE_LIVE_READER,
                    cls.MODE_PRODUCER,
                ):
                    cpm = ro_cpm
            cpm_file = getattr(cpm, "filename", None) if cpm is not None else None
            cpm_matches_path = (
                not isinstance(cpm_file, str)
                or not cpm_file
                or os.path.abspath(str(resolve_cachedb_filepath(cpm_file)))
                == os.path.abspath(str(resolve_cachedb_filepath(price_path)))
            )
            if cpm is not None and cpm_matches_path and hasattr(cpm, "get_cached_price_observation"):
                for cand in norm_cands:
                    try:
                        obs = (
                            cpm.get_cached_price_observation(cand, max_age, now_s=now_s)
                            if now_s is not None and isinstance(cpm, cls)
                            else cpm.get_cached_price_observation(cand, max_age)
                        )
                        if obs is not None:
                            obs_ts = normalize_epoch_sec(obs[0], default=None) if now_s is not None else None
                            px = safe_pos_float(obs[1], default=None)
                            if px is not None and (
                                now_s is None or _is_fresh_obs_ts(obs_ts, eff_now, max_age)
                            ):
                                return px
                    except Exception:
                        continue
        if not price_path:
            return None
        try:
            loaded = cls._read_recovering_dict_and_items(
                price_path, read_json_dict_recovering
            )
            if loaded is None:
                return None
            data, items = loaded
            fetchtime_map = data.get("fetchtime") if isinstance(data.get("fetchtime"), dict) else {}
            file_ts = extract_timestamp_s(data, default=None)
            for cand in norm_cands:
                raw_entry = casefold_get(items, cand)
                if raw_entry is None:
                    continue
                fb_ts = casefold_get(fetchtime_map, cand)
                if fb_ts is None:
                    fb_ts = file_ts
                parsed = cls.parse_ssot_price_point(raw_entry, fallback_ts=fb_ts)
                if parsed is None:
                    continue
                ts_ms, px = parsed
                if _is_fresh_obs_ts(_ts_ms_to_s(ts_ms), eff_now, max_age):
                    return px
        except Exception:
            pass
        return None


def resolve_ssot_price_from_cache(
    symbol: str,
    cache_dir: Optional[str] = None,
    max_age_sec: float = 60.0,
    *,
    quote_isolated: bool = True,
    now_ts: Optional[float] = None,
) -> Optional[float]:
    """Resolve a fresh positive spot price for ``symbol`` from SSOT cacheManager stores (0 ms local)."""
    if (sym_u := clean_upper_token(symbol)) is None or not (sym_clean := clean_symbol_key(sym_u)):
        return None
    if (max_age := resolve_positive_max_age(max_age_sec, 60.0)) is None:
        return None
    now_s = resolve_now_ts(now_ts)
    quotes = (
        compatible_quote_suffixes(sym_clean, default_usd=("USDC", "USDT", "USD", "FDUSD"))
        if quote_isolated
        else ("USDC", "USDT", "USD", "EUR")
    )
    cands = list(
        dict.fromkeys(
            c
            for c in (sym_clean, sym_u, *cache_symbol_variants(sym_clean, quotes=quotes))
            if isinstance(c, str) and c and "/" not in c
        )
    )
    is_default_dir = not cache_dir or str(cache_dir).strip() in ("", "cachedb")
    eff_dir = resolve_cache_dir_path(cache_dir) or get_cachedb_dir()
    price_path = str(resolve_cachedb_filepath(os.path.join(eff_dir, CACHE_CURRENT_PRICE_FILENAME)))
    px = CacheCurrentPriceManager.read_ssot_current_price_candidates(
        cands,
        price_path,
        max_age,
        now_s=now_s,
        use_memory_manager=is_default_dir,
    )
    if px is not None:
        return px

    for cand in cands:
        c24_path = str(
            resolve_cachedb_filepath(os.path.join(eff_dir, cache_24_price_filename(cand)))
        )
        try:
            raw, entries = Cache24PriceManager.read_ssot_24price_entries(c24_path, cand)
            if not entries or raw is None:
                continue
            raw_fb_ts = extract_timestamp_s(raw, default=None)
            for entry in reversed(entries):
                parsed = CachePriceManagerBase.parse_ssot_price_point(entry, fallback_ts=raw_fb_ts)
                if parsed is not None:
                    ts_ms, px = parsed
                    if _is_fresh_obs_ts(_ts_ms_to_s(ts_ms), now_s, max_age):
                        return px
                    break
        except Exception:
            continue
    return None


def resolve_cached_symbol_price(
    symbol: str,
    *,
    venue: Optional[str] = None,
    base: Optional[str] = None,
    quote: Optional[str] = None,
    cache_dir: str = "cachedb",
) -> float:
    """Resolve latest price for ``symbol`` from ``cachedb`` snapshots without importing intelligence collectors."""
    if isinstance(symbol, bool) or not isinstance(symbol, str):
        return math.nan
    sym_u = symbol.strip().upper()
    if not sym_u:
        return math.nan
    venue_norm = normalize_venue_name(str(venue or "")) or ""
    inst_is_eq = is_equity_venue_or_suffix(sym_u, venue_norm)
    inst_base = (
        base.strip().upper()
        if isinstance(base, str) and base.strip()
        else ((split_symbol_assets(sym_u) or (sym_u, ""))[0])
    )
    allowed_quotes = compatible_quote_suffixes(sym_u, quote, default_usd=_DEFAULT_USD_QUOTES)
    is_usd_quote = "USD" in allowed_quotes

    cache_file = os.path.join(cache_dir, CacheCurrentPriceManager.DEFAULT_FILENAME)
    try:
        cdata = read_json_dict_recovering(cache_file)
        if isinstance(cdata, dict):
            raw_items = cdata.get("items")
            items = raw_items if isinstance(raw_items, dict) else {}
            candidates = cache_symbol_variants(
                symbol, include_original=True, include_base=is_usd_quote, quotes=allowed_quotes
            )
            if inst_base and is_usd_quote:
                for cand in cache_symbol_variants(
                    str(inst_base), include_original=True, include_base=True, quotes=allowed_quotes
                ):
                    if cand not in candidates:
                        candidates.append(cand)
            candidates = filter_candidates_by_asset_class(candidates, is_equity=inst_is_eq)

            for cand in candidates:
                px_f = extract_current_price_item(casefold_get(items, cand))
                if px_f is not None:
                    return px_f
            if inst_base:
                base_u = canonical_base_asset(str(inst_base))
                for k, v in items.items():
                    k_str = str(k).strip().upper()
                    if has_equity_suffix(k_str) != inst_is_eq:
                        continue
                    if compatible_quote_suffixes(k_str, default_usd=allowed_quotes if is_usd_quote else ("USD",)) != allowed_quotes:
                        continue
                    if not is_usd_quote and not (split_symbol_assets(k_str) or ("", ""))[1]:
                        continue
                    if canonical_base_asset(k_str) == base_u and (px_f := extract_current_price_item(v)) is not None:
                        return px_f
    except Exception:
        pass

    lookup_sym = f"{sym_u}_US_EQ" if inst_is_eq and not has_equity_suffix(sym_u) else sym_u
    series_cands = filter_candidates_by_asset_class(
        cache_symbol_variants(
            lookup_sym, include_original=True, include_base=False, quotes=allowed_quotes
        ),
        is_equity=inst_is_eq,
    )
    cands_upper = {c.strip().upper() for c in series_cands if c.strip()}
    try:
        for cand in series_cands:
            p24_path = os.path.join(cache_dir, Cache24PriceManager.disk_filename_for_symbol(cand))
            _, rows = Cache24PriceManager.read_ssot_24price_entries(
                p24_path, cand, candidates=series_cands, allow_single_item_fallback=True
            )
            if rows:
                for row in reversed(rows):
                    parsed = CachePriceManagerBase.parse_ssot_price_point(row, fallback_ts=1.0)
                    if parsed is not None:
                        return parsed[1]
            for jsonl_fn in (
                CacheSparsePriceManager.disk_filename_for_symbol(cand),
                Cache365PriceManager.disk_filename_for_symbol(cand),
            ):
                jsonl_path = os.path.join(cache_dir, jsonl_fn)
                if not os.path.exists(jsonl_path):
                    continue
                for line in reversed(read_tail_lines(jsonl_path, max_bytes=64 * 1024)):
                    rec = parse_json_payload(line, default=None)
                    if rec is None:
                        continue
                    pt = CachePriceManagerBase.extract_jsonl_series_point(rec, cands_upper)
                    if pt is not None and (px_f := safe_pos_float(pt[1], default=None)) is not None:
                        return px_f
    except Exception:
        pass

    if is_usd_quote:
        try:
            ob_cands = [
                c
                for c in cache_symbol_variants(
                    lookup_sym, include_original=True, include_base=True, quotes=allowed_quotes
                )
                if has_equity_suffix(c) == inst_is_eq
                and compatible_quote_suffixes(c, default_usd=_DEFAULT_USD_QUOTES) == allowed_quotes
            ]
            for cand in ob_cands:
                ob_path = telemetry_cache_file_path(cache_dir, "orderbook_depth", cand)
                if not ob_path or (not os.path.exists(ob_path) and not os.path.exists(previous_generation_path(ob_path))):
                    continue
                ob = read_json_dict_recovering(ob_path)
                if isinstance(ob, dict):
                    raw_mid = ob.get("mid_price") if ob.get("mid_price") is not None else ob.get("ob_mid_price")
                    ob_mid = safe_pos_float(raw_mid, default=None)
                    if ob_mid is None:
                        bb = safe_pos_float(ob.get("best_bid"), default=None)
                        ba = safe_pos_float(ob.get("best_ask"), default=None)
                        if bb is not None and ba is not None:
                            ob_mid = (bb + ba) / 2.0
                    if ob_mid is not None:
                        return ob_mid
        except Exception:
            pass

    return math.nan


def attach_default_price_ws_stream(
    cpm_getter=None,
    *,
    sync_ts: float = PRICE_STREAM_POLL_INTERVAL_SEC,
    data_sync_mode: str = CacheManagerInterface.MODE_DISK_READER,
) -> Optional[CacheCurrentPriceManager]:
    """Attach the default venue price WebSocket manager to ``CacheCurrentPriceManager``."""
    default_getter = CachePriceManagerBase._default_current_price_getter
    getter = cpm_getter or _cm_override("get_current_price_manager", default_getter)
    if not callable(getter) or _bn_price_prov_cls is None:
        return None
    return getter(
        ws_manager=_bn_price_prov_cls.get_default_price_ws_manager(),
        sync_ts=sync_ts,
        data_sync_mode=data_sync_mode,
    )


def request_stop_price_ws_streams() -> None:
    """Signal the default venue price WebSocket manager to stop."""
    if _bn_price_prov_cls is not None:
        _bn_price_prov_cls.request_stop_price_ws_manager()


def stop_price_ws_streams(shutdown_deadline: float) -> None:
    """Stop the default venue price WebSocket manager and periodic clock resync thread."""
    if _bn_price_prov_cls is not None:
        _bn_price_prov_cls.stop_price_ws_manager(
            timeout=min(1.0, max(0.35, shutdown_deadline - _monotonic_time()))
        )


SPARSE_PRICE_SYNC_INTERVAL_SEC = 3 * 60
PRICE_24_SYNC_INTERVAL_SEC = 30
CURRENT_PRICE_SYNC_INTERVAL_SEC = PRICE_STREAM_POLL_INTERVAL_SEC
_READONLY_PRICE_LOCK = threading.RLock()
_READONLY_PRICE_READERS: dict[str, Any] = {}


def get_current_price_manager(
    ws_manager=None,
    symbols=None,
    sync_ts=None,
    provider_names=None,
    read_only=None,
    *,
    data_sync_mode: Optional[str] = CacheManagerInterface.MODE_DISK_READER,
) -> CacheCurrentPriceManager:
    """Return or lazily create the ``CacheCurrentPriceManager`` singleton (disk_reader by default)."""
    cm_mod = CacheManagerInterface._resolve_cache_manager_module()
    if cm_mod is not None:
        cm_fn = getattr(cm_mod, "get_current_price_manager", None)
        if callable(cm_fn) and getattr(cm_fn, "_is_ssot_current_price_getter", None) is not True:
            if (
                ws_manager is None
                and symbols is None
                and sync_ts is None
                and provider_names is None
                and read_only is None
                and data_sync_mode == CacheManagerInterface.MODE_DISK_READER
            ):
                return cm_fn()
            return cm_fn(
                ws_manager=ws_manager,
                symbols=symbols,
                sync_ts=sync_ts,
                provider_names=provider_names,
                read_only=read_only,
                data_sync_mode=data_sync_mode,
            )
        factory = getattr(cm_mod, "CacheFactory", None)
        if factory is not None:
            return factory.get(
                "CurrentPrice",
                symbols=symbols,
                data_sync_mode=data_sync_mode,
                provider_names=provider_names,
                ws_manager=ws_manager,
                sync_ts=sync_ts,
                read_only=read_only,
            )

    resolved_mode, save_state_flag, start_periodic = CacheManagerInterface.resolve_data_sync_mode(data_sync_mode)
    if ws_manager is not None and resolved_mode == CacheManagerInterface.MODE_DISK_READER:
        resolved_mode = CacheManagerInterface.MODE_LIVE_READER
    with _READONLY_PRICE_LOCK:
        inst = _READONLY_PRICE_READERS.get("CurrentPrice")
        if (
            inst is not None
            and getattr(inst, "_is_default_symbol_reader", True)
            and symbols is None
            and not provider_names
            and ws_manager is None
            and sync_ts is None
            and not save_state_flag
            and not start_periodic
        ):
            if resolved_mode == CacheManagerInterface.MODE_LIVE_READER:
                inst.data_sync_mode = resolved_mode
            elif getattr(inst, "data_sync_mode", None) == CacheManagerInterface.MODE_DISK_READER:
                CacheManagerInterface._resync_readonly_reader(inst)
            return inst
        default_symbols, _, default_providers = tracked_price_instruments()
        eff_symbols = (
            default_symbols
            if symbols is None
            else (
                parse_symbol_list(symbols)
                if isinstance(symbols, str)
                else (
                    []
                    if isinstance(symbols, (bool, bytes, bytearray)) or not hasattr(symbols, "__iter__")
                    else [s.strip() for s in symbols if not isinstance(s, bool) and isinstance(s, str) and s.strip()]
                )
            )
        )
        eff_providers = default_providers if (provider_names is None and default_providers) else provider_names
        eff_read_only = (
            _parse_bool_flag(read_only, default=False)
            if read_only is not None
            else (not save_state_flag and not start_periodic)
        )
        inst = CacheCurrentPriceManager(
            sync_ts=CURRENT_PRICE_SYNC_INTERVAL_SEC if sync_ts is None else sync_ts,
            symbols=eff_symbols,
            filename=CacheCurrentPriceManager.DEFAULT_FILENAME,
            ws_manager=ws_manager,
            provider_names=eff_providers,
            read_only=eff_read_only,
            data_sync_mode=resolved_mode,
        )
        is_default_reader = bool(symbols is None and not provider_names)
        inst._is_default_symbol_reader = is_default_reader
        if ws_manager is None and (
            is_default_reader
            or (
                resolved_mode == CacheManagerInterface.MODE_LIVE_READER
                and _READONLY_PRICE_READERS.get("CurrentPrice") is None
            )
        ):
            _READONLY_PRICE_READERS["CurrentPrice"] = inst
        return inst


get_current_price_manager._is_ssot_current_price_getter = True  # type: ignore[attr-defined]


def _get_per_symbol_price_managers(
    cache_name: str,
    manager_cls: type,
    default_sync_ts: float,
    *,
    symbols=None,
    data_sync_mode: Optional[str] = CacheManagerInterface.MODE_DISK_READER,
    getter_tag: str,
    cm_getter_name: str,
):
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
        default_symbols=lambda: tracked_price_instruments()[0],
        data_sync_mode=data_sync_mode,
        cm_getter_name=cm_getter_name,
        getter_tag=getter_tag,
    )
    if delegated:
        return res
    with _READONLY_PRICE_LOCK:
        inst = _READONLY_PRICE_READERS.get(cache_name)
        if isinstance(inst, dict) and not save_state_flag and not start_periodic:
            wanted = {str(s).strip().upper() for s in eff_symbols if s}
            if wanted.issubset({str(k).strip().upper() for k in inst}):
                for sym_k, mgr in inst.items():
                    if str(sym_k).strip().upper() in wanted:
                        CacheManagerInterface._resync_readonly_reader(mgr)
                return inst
        prefix, ext = manager_cls.FILE_PREFIX, manager_cls.FILE_EXT
        eff_read_only = not save_state_flag and not start_periodic
        inst = {
            s: manager_cls(
                sync_ts=default_sync_ts,
                filename=f"{prefix}{s}.{ext}",
                symbols=[s],
                read_only=eff_read_only,
                data_sync_mode=resolved_mode,
            )
            for s in eff_symbols
        }
        _READONLY_PRICE_READERS[cache_name] = inst
        return inst


def get_24_price_manager(
    symbols=None,
    *,
    data_sync_mode: Optional[str] = CacheManagerInterface.MODE_DISK_READER,
):
    """Return the per-symbol ``Cache24PriceManager`` reader dict without importing ``cacheManager``."""
    return _get_per_symbol_price_managers(
        "24Price",
        Cache24PriceManager,
        PRICE_24_SYNC_INTERVAL_SEC,
        symbols=symbols,
        data_sync_mode=data_sync_mode,
        getter_tag="_is_ssot_24_price_getter",
        cm_getter_name="get_24_price_manager",
    )


get_24_price_manager._is_ssot_24_price_getter = True  # type: ignore[attr-defined]


def get_sparse_price_manager(
    symbols=None,
    *,
    data_sync_mode: Optional[str] = CacheManagerInterface.MODE_DISK_READER,
):
    """Return the per-symbol ``CacheSparsePriceManager`` reader dict without importing ``cacheManager``."""
    return _get_per_symbol_price_managers(
        "SparsePrice",
        CacheSparsePriceManager,
        SPARSE_PRICE_SYNC_INTERVAL_SEC,
        symbols=symbols,
        data_sync_mode=data_sync_mode,
        getter_tag="_is_ssot_sparse_price_getter",
        cm_getter_name="get_sparse_price_manager",
    )


get_sparse_price_manager._is_ssot_sparse_price_getter = True  # type: ignore[attr-defined]


