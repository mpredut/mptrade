"""Price domain cache hierarchy, SSOT price resolvers, and singleton reader factories.

Thematic layout (Sections 1..7):
  1. Level 2 Abstract Base: ``CachePriceManagerBase(CacheManagerInterface, ABC)``
  2. Sparse Long-Horizon Archive: ``CacheSparsePriceManager(CachePriceManagerBase)``
  3. Rolling 24-Hour Series: ``Cache24PriceManager(CachePriceManagerBase)``
  4. Dense 365-Day Archive: ``Cache365PriceManager(Cache24PriceManager)``
  5. Real-Time Spot & Stream Fanout: ``CacheCurrentPriceManager(CachePriceManagerBase)``
  6. SSOT Local Price Resolution & WS Stream Lifecycle (`resolve_ssot_price_from_cache`, `resolve_cached_symbol_price`)
  7. SSOT Singleton Factory Getters (`get_current_price_manager`, `get_24_price_manager`, `get_sparse_price_manager`)
"""

from __future__ import annotations

from abc import ABC
import array
import bisect
import builtins
from collections import defaultdict, deque
from collections.abc import MutableSequence, Sequence
import concurrent.futures as _futures
import math
from operator import itemgetter
import os
import socket
import threading
import time
from typing import Any, Callable, Optional

from botcore import (
    NonBlockingRestSlotTracker,
    PRICE_FALLBACK_REST_HARD_TIMEOUT_SEC,
    PRICE_FRESHNESS_MAX_AGE_MS,
    PRICE_FRESHNESS_MAX_AGE_SEC,
    PRICE_STREAM_POLL_INTERVAL_SEC,
    PRICE_WS_TIMEOUT_SEC,
    SHORT_TREND_FRESHNESS_MAX_AGE_SEC,
    _is_fresh_obs_ts,
    _parse_bool_flag,
    _ts_ms_to_s,
    clean_upper_scalar,
    extract_current_price_item,
    extract_timestamp_ms,
    extract_timestamp_s,
    is_auto_start_websockets_enabled,
    is_live_network_backfill_enabled,
    is_rth_gate_enabled,
    is_us_extended_hours_open,
    is_us_rth_open,
    load_dotenv,
    lru_dict_put,
    normalize_epoch_sec,
    normalize_timestamp_ms,
    parse_symbol_list,
    required_float_env,
    required_int_env,
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
    _coerce_symbol_list,
    _monotonic_time,
    _norm_symbol_keys,
    _now_time,
)
from pricewindow import filter_isolated_price_points
from cachepaths import (
    CACHE_24_PRICE_PREFIX,
    CACHE_365_PRICE_PREFIX,
    CACHE_CURRENT_PRICE_FILENAME,
    CACHE_CURRENT_PRICE_TICKS_SOCKET_FILENAME,
    CACHE_SPARSE_PRICE_PREFIX,
    DEFAULT_USD_QUOTES,
    cache_ohlc_24h_filename,
    cache_path,
    cache_symbol_variants,
    cachedb_root_for_file,
    canonical_base_asset,
    canonical_symbol_key,
    clean_scalar_str,
    clean_symbol_key,
    compatible_quote_suffixes,
    ensure_equity_suffix,
    filter_candidates_by_asset_class,
    get_cachedb_dir,
    has_equity_suffix,
    is_equity_venue,
    is_equity_venue_or_suffix,
    is_primary_stream_venue,
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
    VenuePriceStreamBase as _VenuePriceStreamBase,
    fetch_price_with_deadline as _fetch_price_with_deadline,
)
from providers.ssot_bridge import live_price_fetch_context
from state_io import (
    atomic_text_writer,
    atomic_write_json,
    atomic_write_jsonl,
    cachedb_exists,
    canonical_json_key,
    commit_file_and_meta,
    discard_generation_file,
    format_jsonl_bucket_line,
    generation_lock_path,
    generation_transaction,
    is_state_io_dict_reader_unpatched,
    iter_jsonl_retention_lines,
    meta_file_path,
    parse_json_payload,
    previous_generation_path,
    read_json_dict,
    read_json_dict_cached,
    read_json_dict_recovering,
    read_jsonl_bucket_map,
    read_jsonl_bucket_map_from_offset,
    read_jsonl_or_json_dict_cached,
    read_tail_complete_lines_text,
    read_tail_lines,
    read_tail_lines_with_cursor,
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
_ORIG_READ_JSON_DICT = read_json_dict
_ORIG_READ_JSON_DICT_CACHED = read_json_dict_cached
_ORIG_READ_JSON_DICT_RECOVERING = read_json_dict_recovering
_LIST_OR_TUPLE_TYPES = (list, tuple)
_EXACT_NUM_TYPES = (int, float)
_SSOT_PT_TS_KEY = itemgetter(0)
_SSOT_24PRICE_MAX_ENTRIES = 64
_SSOT_24PRICE_DISK_MEMO: dict[str, tuple[tuple, dict[str, Any]]] = {}
_SSOT_24PRICE_PARSED_MEMO: dict[
    tuple[str, str],
    tuple[tuple[Any, Any, Any], tuple[Any, ...], list[tuple[int, float]], int],
] = {}


def _ssot_entry_memo_token(entry: Any) -> Any:
    if type(entry) in _LIST_OR_TUPLE_TYPES and len(entry) >= 2:
        t_v, p_v = entry[0], entry[1]
        if type(t_v) in _EXACT_NUM_TYPES and type(p_v) in _EXACT_NUM_TYPES:
            return (id(entry), t_v, p_v)
    return id(entry)


def _fast_parse_ssot_point(entry: Any) -> Optional[tuple[int, float]]:
    if (
        type(entry) in _LIST_OR_TUPLE_TYPES
        and len(entry) >= 2
        and type(t0 := entry[0]) is int
        and 100_000_000_000 <= t0 <= 9_000_000_000_000_000
    ):
        p1 = entry[1]
        if type(p1) is float and p1 > 0.0 and math.isfinite(p1):
            return (t0, p1)
        if type(p1) is int and 0 < p1 <= 9_000_000_000_000_000:
            return (t0, p1 * 1.0)
    return None


# =============================================================================
# 1. LEVEL 2 ABSTRACT BASE: CachePriceManagerBase
# =============================================================================


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
        if type(rec) is dict or isinstance(rec, dict):
            if allowed_symbols_upper:
                rec_s = rec.get("s")
                if type(rec_s) is str:
                    if rec_s not in allowed_symbols_upper:
                        stripped_s = rec_s.strip()
                        if stripped_s and stripped_s.upper() not in allowed_symbols_upper:
                            return None
                elif isinstance(rec_s, str):
                    stripped_s = rec_s.strip()
                    if stripped_s and stripped_s.upper() not in allowed_symbols_upper:
                        return None
            item = rec.get("i")
        else:
            item = rec if isinstance(rec, list) else None
        if (type(item) is list or isinstance(item, (list, tuple))) and len(item) >= 2:
            return [item[0], item[1]]
        return None

    @staticmethod
    def parse_ssot_price_point(
        entry: Any, fallback_ts: Optional[float] = None
    ) -> Optional[tuple[int, float]]:
        """Parse a cached price entry (dict, ``[ts, px]``, or nested list) into ``(ts_ms, price)``."""
        if fallback_ts is None and (fast_pt := _fast_parse_ssot_point(entry)) is not None:
            return fast_pt

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

    filter_isolated_price_points = staticmethod(filter_isolated_price_points)

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


# =============================================================================
# 2. COMPACT TYPED SERIES & SPARSE LONG-HORIZON ARCHIVE: CacheSparsePriceManager
# =============================================================================


class CompactPriceSeries(MutableSequence):
    """Memory-compact ``[ts_ms, price(, volume, ...)]`` sequence backed by C ``int64`` + ``float64`` arrays.

    Uses ``16 B/point`` (``array('q')`` + ``array('d')``) for standard 2-column ``[ts, px]`` series.
    Lazily allocates an optional ``_vol`` ``array('d')`` buffer (``+8 B/point``) only when a 3+-column
    OHLCV row arrives (e.g. equity 5m bars), and an optional sparse ``_extra`` map only for >3-column
    bars, preserving 100% lossless round-trip semantics with zero memory overhead on crypto ticks.
    """

    __slots__ = ("_ts", "_px", "_vol", "_extra", "_monotonic")

    def __init__(self, items: Any = None) -> None:
        if isinstance(items, CompactPriceSeries):
            self._ts = array.array("q", items._ts)
            self._px = array.array("d", items._px)
            self._vol = array.array("d", items._vol) if items._vol is not None else None
            self._extra = dict(items._extra) if items._extra is not None else None
            self._monotonic = items._monotonic
            return
        self._ts = array.array("q")
        self._px = array.array("d")
        self._vol: Optional[array.array] = None
        self._extra: Optional[dict[int, tuple[Any, ...]]] = None
        self._monotonic = True
        if items is not None:
            self.extend(items)

    @staticmethod
    def _coerce_pair(item: Any) -> Optional[tuple[int, float]]:
        fast = _fast_parse_ssot_point(item)
        if fast is not None:
            return fast
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            raw_ts = item[0]
            if type(raw_ts) is int and 0 < raw_ts <= 9_000_000_000_000_000:
                ts_i: Optional[int] = raw_ts
            else:
                ts_i = normalize_timestamp_ms(raw_ts, default=None)
            px_f = safe_pos_float(item[1], default=None)
            if ts_i is not None and ts_i > 0 and px_f is not None:
                return ts_i, px_f
        return None

    @classmethod
    def _coerce_row(
        cls, item: Any
    ) -> Optional[tuple[int, float, Optional[float], Optional[tuple[Any, ...]]]]:
        if type(item) in _LIST_OR_TUPLE_TYPES and len(item) == 2:
            fast = _fast_parse_ssot_point(item)
            if fast is not None:
                return fast[0], fast[1], None, None
        pair = cls._coerce_pair(item)
        if pair is None:
            return None
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            raw_vol = item[2]
            vol_f = (
                None
                if raw_vol is None or isinstance(raw_vol, bool)
                else safe_nonneg_float(raw_vol, default=None)
            )
            extra = tuple(item[3:]) if len(item) > 3 else None
            return pair[0], pair[1], vol_f, extra
        return pair[0], pair[1], None, None

    def append_pair(self, ts_ms: int, price: float) -> None:
        if self._monotonic and self._ts and ts_ms < self._ts[-1]:
            self._monotonic = False
        self._ts.append(ts_ms)
        self._px.append(price)
        if self._vol is not None:
            self._vol.append(math.nan)

    def append_row(
        self,
        ts_ms: int,
        price: float,
        volume: Optional[float] = None,
        extra: Optional[tuple[Any, ...]] = None,
    ) -> None:
        if volume is None and extra is None:
            self.append_pair(ts_ms, price)
            return
        idx = len(self._ts)
        if self._monotonic and idx and ts_ms < self._ts[-1]:
            self._monotonic = False
        self._ts.append(ts_ms)
        self._px.append(price)
        if self._vol is None:
            self._vol = array.array("d", [math.nan] * idx)
        self._vol.append(math.nan if volume is None else volume * 1.0)
        if extra is not None:
            if self._extra is None:
                self._extra = {}
            self._extra[idx] = extra

    def append(self, value: Any) -> None:
        row = self._coerce_row(value)
        if row is not None:
            self.append_row(row[0], row[1], row[2], row[3])

    def extend(self, values: Any) -> None:
        if isinstance(values, CompactPriceSeries):
            if not values._ts:
                return
            base_len = len(self._ts)
            if self._monotonic:
                self._monotonic = values._monotonic and (
                    not base_len or values._ts[0] >= self._ts[-1]
                )
            self._ts.extend(values._ts)
            self._px.extend(values._px)
            if self._vol is not None or values._vol is not None:
                if self._vol is None:
                    self._vol = array.array("d", [math.nan] * base_len)
                if values._vol is not None:
                    self._vol.extend(values._vol)
                else:
                    self._vol.extend(array.array("d", [math.nan] * len(values._ts)))
            if values._extra:
                if self._extra is None:
                    self._extra = {}
                for rel_idx, ext_val in values._extra.items():
                    self._extra[base_len + rel_idx] = ext_val
            return
        for item in values:
            row = self._coerce_row(item)
            if row is not None:
                self.append_row(row[0], row[1], row[2], row[3])

    def trim_head(self, drop_count: int) -> int:
        if drop_count <= 0 or not self._ts:
            return 0
        n = min(drop_count, len(self._ts))
        del self._ts[:n]
        del self._px[:n]
        if self._vol is not None:
            del self._vol[:n]
            if not self._ts:
                self._vol = None
        if self._extra is not None:
            shifted = {k - n: v for k, v in self._extra.items() if k >= n}
            self._extra = shifted or None
        return n

    def copy(self) -> CompactPriceSeries:
        return CompactPriceSeries(self)

    def clear(self) -> None:
        del self._ts[:]
        del self._px[:]
        self._vol = None
        self._extra = None
        self._monotonic = True

    def prune_older_than(self, cutoff_ms: int) -> int:
        """Drop all points with timestamp strictly below ``cutoff_ms`` in-place; return removed count."""
        if not self._ts:
            return 0
        if self._monotonic:
            return self.trim_head(self.bisect_left_ts(cutoff_ms))
        before = len(self._ts)
        kept = CompactPriceSeries(it for it in self if it[0] >= cutoff_ms)
        removed = before - len(kept)
        if removed > 0:
            self._ts, self._px = kept._ts, kept._px
            self._vol, self._extra = kept._vol, kept._extra
            self._monotonic = kept._monotonic
        return removed

    def last_pair(self) -> Optional[tuple[int, float]]:
        return (self._ts[-1], self._px[-1]) if self._ts else None

    def max_timestamp_ms(self) -> int:
        if not self._ts:
            return 0
        return self._ts[-1] if self._monotonic else max(self._ts)

    def bisect_left_ts(self, ts_ms: int) -> int:
        """Binary-search ``self._ts`` directly in C without allocating Python point lists."""
        return bisect.bisect_left(self._ts, ts_ms)

    def has_positive_volume(self) -> bool:
        """Return True when at least one row carries a positive 3rd-column volume (O(1) False on 2-col series)."""
        if self._vol is None:
            return False
        return any(v > 0.0 and not math.isnan(v) for v in self._vol)

    def _materialize_at(self, idx: int) -> list[Any]:
        n = len(self._ts)
        if idx < 0:
            idx += n
        if self._vol is None:
            return [self._ts[idx], self._px[idx]]
        v = self._vol[idx]
        ext = self._extra.get(idx) if self._extra is not None else None
        if ext is not None:
            return [self._ts[idx], self._px[idx], (None if math.isnan(v) else v), *ext]
        if not math.isnan(v):
            return [self._ts[idx], self._px[idx], v]
        return [self._ts[idx], self._px[idx]]

    def to_numpy(self, *, since_ms: Optional[int] = None) -> tuple[Any, Any]:
        """Return independent ``(ts_ms_int64, px_float64)`` NumPy arrays sliced at ``since_ms``."""
        import numpy as np

        if not self._ts:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        ts_view = np.frombuffer(self._ts, dtype=np.int64)
        px_view = np.frombuffer(self._px, dtype=np.float64)
        try:
            if since_ms is None:
                return ts_view.copy(), px_view.copy()
            if self._monotonic:
                idx = np.searchsorted(ts_view, since_ms, side="left").item()
                return ts_view[idx:].copy(), px_view[idx:].copy()
            mask = ts_view >= since_ms
            return ts_view[mask].copy(), px_view[mask].copy()
        finally:
            del ts_view, px_view

    def __len__(self) -> int:
        return len(self._ts)

    def __bool__(self) -> bool:
        return bool(self._ts)

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            out = CompactPriceSeries()
            out._ts = self._ts[index]
            out._px = self._px[index]
            if self._vol is not None:
                out._vol = self._vol[index]
            if self._extra:
                orig_indices = range(*index.indices(len(self._ts)))
                mapped = {
                    new_i: self._extra[old_i]
                    for new_i, old_i in enumerate(orig_indices)
                    if old_i in self._extra
                }
                out._extra = mapped or None
            if self._monotonic and (index.step is None or index.step > 0):
                out._monotonic = True
            else:
                ts_s = out._ts
                out._monotonic = all(ts_s[i] >= ts_s[i - 1] for i in range(1, len(ts_s)))
            return out
        return self._materialize_at(index)

    def __setitem__(self, index: Any, value: Any) -> None:
        if isinstance(index, slice):
            rows = [r for it in value if (r := self._coerce_row(it)) is not None]
            has_vol_in = any(r[2] is not None or r[3] is not None for r in rows)
            n_before = len(self._ts)
            if self._vol is None and has_vol_in:
                self._vol = array.array("d", [math.nan] * n_before)
            self._ts[index] = array.array("q", (r[0] for r in rows))
            self._px[index] = array.array("d", (r[1] for r in rows))
            if self._vol is not None:
                self._vol[index] = array.array(
                    "d", (math.nan if r[2] is None else r[2] * 1.0 for r in rows)
                )
            if self._extra is not None or any(r[3] is not None for r in rows):
                old_extra_list: list[Optional[tuple[Any, ...]]] = [
                    (self._extra.get(i) if self._extra is not None else None)
                    for i in range(n_before)
                ]
                old_extra_list[index] = [r[3] for r in rows]
                rebuilt = {i: e for i, e in enumerate(old_extra_list) if e is not None}
                self._extra = rebuilt or None
            self._monotonic = False
            return
        row = self._coerce_row(value)
        if row is None:
            raise ValueError(f"invalid compact price point: {value!r}")
        n = len(self._ts)
        eff_idx = index + n if index < 0 else index
        self._ts[eff_idx] = row[0]
        self._px[eff_idx] = row[1]
        if row[2] is not None or row[3] is not None:
            if self._vol is None:
                self._vol = array.array("d", [math.nan] * n)
            self._vol[eff_idx] = math.nan if row[2] is None else row[2] * 1.0
        elif self._vol is not None:
            self._vol[eff_idx] = math.nan
        if row[3] is not None:
            if self._extra is None:
                self._extra = {}
            self._extra[eff_idx] = row[3]
        elif self._extra is not None:
            self._extra.pop(eff_idx, None)
            if not self._extra:
                self._extra = None
        self._monotonic = False

    def __delitem__(self, index: Any) -> None:
        n_before = len(self._ts)
        del self._ts[index]
        del self._px[index]
        if self._vol is not None:
            del self._vol[index]
            if not self._ts:
                self._vol = None
        if self._extra is not None:
            old_extra_list = [self._extra.get(i) for i in range(n_before)]
            del old_extra_list[index]
            rebuilt = {i: e for i, e in enumerate(old_extra_list) if e is not None}
            self._extra = rebuilt or None

    def insert(self, index: int, value: Any) -> None:
        row = self._coerce_row(value)
        if row is None:
            return
        n_before = len(self._ts)
        eff_idx = max(0, min(n_before, index + n_before if index < 0 else index))
        self._ts.insert(eff_idx, row[0])
        self._px.insert(eff_idx, row[1])
        if row[2] is not None or row[3] is not None:
            if self._vol is None:
                self._vol = array.array("d", [math.nan] * n_before)
            self._vol.insert(eff_idx, math.nan if row[2] is None else row[2] * 1.0)
        elif self._vol is not None:
            self._vol.insert(eff_idx, math.nan)
        if self._extra is not None or row[3] is not None:
            shifted = {
                (k + 1 if k >= eff_idx else k): v
                for k, v in (self._extra or {}).items()
            }
            if row[3] is not None:
                shifted[eff_idx] = row[3]
            self._extra = shifted or None
        self._monotonic = False

    def __iter__(self):
        if self._vol is None and self._extra is None:
            return ([t, p] for t, p in zip(self._ts, self._px))
        return (self._materialize_at(i) for i in range(len(self._ts)))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CompactPriceSeries):
            if self._ts != other._ts or self._px != other._px:
                return False
            if self._vol is None and other._vol is None and self._extra is None and other._extra is None:
                return True
            return all(
                self._materialize_at(i) == other._materialize_at(i)
                for i in range(len(self._ts))
            )
        if isinstance(other, list):
            if len(other) != len(self._ts):
                return False
            if self._vol is None and self._extra is None:
                for row, t, p in zip(other, self._ts, self._px):
                    if not isinstance(row, (list, tuple)) or len(row) != 2 or row[0] != t or row[1] != p:
                        return False
                return True
            for i, row in enumerate(other):
                if not isinstance(row, (list, tuple)) or list(row) != self._materialize_at(i):
                    return False
            return True
        return NotImplemented

    def __repr__(self) -> str:
        return f"CompactPriceSeries(len={len(self._ts)})"


class CacheBoundedJsonlArchiveMixin:
    """SSOT bounded-memory compact-buffer mixin for multi-year JSONL archives (Sparse & 365d).

    Enforces by construction:
      1. Compact ``CompactPriceSeries`` storage (16 B/point) in RAM bounded to ``_memory_row_limit()``.
      2. Lossless ``_persisted_counts`` trimming (never drops unpersisted entries before disk flush).
      3. Incremental byte-offset tailing for read-only consumers and bounded cold startup loads.
      4. Streaming disk-level ``maintain_append_persist()`` / ``compact_jsonl()`` so a bounded memory
         tail can NEVER overwrite or truncate the full multi-year JSONL archive on disk.
      5. Direct zero-tuple ``get_series_arrays()`` slicing from the underlying C ``int64``/``float64`` buffers.
    """

    def _memory_row_limit(self) -> int:
        return safe_pos_int(getattr(self, "MEMORY_ROWS", 35_000), default=35_000) or 35_000

    def _new_symbol_bucket(self, symbol: str) -> CompactPriceSeries:
        return CompactPriceSeries()

    def _should_compact_before_append_locked(self) -> bool:
        return False

    def _on_stale_memory_detected_locked(self) -> None:
        pass

    def _before_snapshot_commit_locked(self, persisted_max_ts: int) -> None:
        pass

    def _trim_memory_tail_locked(
        self, symbol: Optional[str] = None, *, min_batch: int = 1
    ) -> None:
        limit = self._memory_row_limit()
        eff_batch = safe_pos_int(min_batch, default=1) or 1
        syms = (symbol,) if symbol is not None else tuple(self.cache.keys())
        for sym_k in syms:
            bucket = self.cache.get(sym_k)
            if not self._is_bucket_seq(bucket):
                continue
            overflow = len(bucket) - limit
            if overflow < eff_batch:
                continue
            persisted = safe_nonneg_int(self._persisted_counts.get(sym_k, 0), default=0) or 0
            trim_n = overflow if self.read_only else min(overflow, persisted)
            if trim_n <= 0:
                continue
            if isinstance(bucket, CompactPriceSeries):
                bucket.trim_head(trim_n)
            else:
                self.cache[sym_k] = bucket[trim_n:]
            self._persisted_counts[sym_k] = max(0, persisted - trim_n)

    def _after_state_persisted(self) -> None:
        drain_fn = getattr(self, "_drain_deferred_price_ticks_locked", None)
        if callable(drain_fn):
            with self._deferred_ticks_lock:
                has_deferred = bool(self._deferred_price_ticks)
            if has_deferred:
                with self.lock:
                    drain_fn()
        with self.lock:
            self._trim_memory_tail_locked()

    def _load_jsonl(self):
        """Load bounded recent working set into compact typed buffers; tail incrementally for readers."""
        limit = self._memory_row_limit()
        allowed_syms = set(self.symbols) if self.symbols else None
        with self.lock:
            backup_path = (
                self._previous_generation_path()
                if self.read_only
                else self._ensure_file_or_restore_previous_locked()
            )
            abs_path = os.path.abspath(self.filename)
            tail_st = getattr(self, "_jsonl_tail_state", None)
            inc_done = False
            if (
                self.read_only
                and os.path.exists(self.filename)
                and isinstance(self.cache, dict)
                and bool(self.cache)
                and tail_st is not None
                and tail_st[0] == abs_path
            ):
                delta_map, had_bad, next_off, dev_ino, new_anchor, inc_used = (
                    read_jsonl_bucket_map_from_offset(
                        self.filename,
                        start_offset=tail_st[2],
                        expected_dev_ino=tail_st[1],
                        expected_anchor=tail_st[3],
                        allow_full_fallback=False,
                    )
                )
                if inc_used:
                    for sym_k, rows_delta in delta_map.items():
                        if allowed_syms is not None and sym_k not in allowed_syms:
                            continue
                        bucket = self.cache.get(sym_k)
                        if bucket is None:
                            bucket = self._new_symbol_bucket(sym_k)
                            self.cache[sym_k] = bucket
                        bucket.extend(rows_delta)
                        if len(bucket) > limit:
                            overflow = len(bucket) - limit
                            if isinstance(bucket, CompactPriceSeries):
                                bucket.trim_head(overflow)
                            else:
                                self.cache[sym_k] = bucket[overflow:]
                    self._jsonl_tail_state = (
                        (abs_path, dev_ino, next_off, new_anchor)
                        if (not had_bad and dev_ino is not None and next_off > 0)
                        else None
                    )
                    inc_done = True

            if not inc_done:
                self.cache = {}
                self._jsonl_tail_state = None
                source_path = (
                    self.filename
                    if os.path.exists(self.filename)
                    else (backup_path if self.read_only and os.path.exists(backup_path) else None)
                )
                if source_path is not None:
                    tail_lines, cursor = read_tail_lines_with_cursor(source_path, limit)
                    for raw_line in tail_lines:
                        rec = parse_json_payload(raw_line, default=None)
                        if not isinstance(rec, dict):
                            continue
                        sym_k = rec.get("s")
                        item = rec.get("i")
                        if not isinstance(sym_k, str) or item is None:
                            continue
                        if allowed_syms is not None and sym_k not in allowed_syms:
                            continue
                        pair = CompactPriceSeries._coerce_pair(item)
                        if pair is None:
                            continue
                        bucket = self.cache.get(sym_k)
                        if bucket is None:
                            bucket = self._new_symbol_bucket(sym_k)
                            self.cache[sym_k] = bucket
                        bucket.append_pair(pair[0], pair[1])
                    if self.read_only and source_path == self.filename and cursor is not None:
                        self._jsonl_tail_state = (abs_path, cursor[0], cursor[1], cursor[2])
            self._persisted_counts = {s: len(v) for s, v in self.cache.items()}
            if meta_payload := self._read_optional_meta_dict():
                self._hydrate_fetchtimes_from_mapping_locked(
                    meta_payload,
                    filter_symbols=bool(allowed_syms),
                    coerce_nonneg_int=True,
                )
            if self.cache:
                for s_sym, rebuilt_ft in self.rebuild_fetchtimes().items():
                    self.fetchtime_per_symbol[s_sym] = max(
                        safe_nonneg_int(self.fetchtime_per_symbol.get(s_sym), default=0) or 0,
                        safe_nonneg_int(rebuilt_ft, default=0) or 0,
                    )
                    self._rest_sync_cursor_per_symbol.setdefault(
                        s_sym, self.fetchtime_per_symbol[s_sym]
                    )

    @staticmethod
    def _merge_max_nonneg_int_map(dst: dict, src: dict) -> None:
        for s_sym, val in src.items():
            dst[s_sym] = max(
                safe_nonneg_int(dst.get(s_sym), default=0) or 0,
                safe_nonneg_int(val, default=0) or 0,
            )

    def _snapshot_memory_watermark_locked(self):
        original_state = self._snapshot_memory_state_locked()
        watermarks: dict[str, tuple[int, int, set[str]]] = {}
        for sym_k, bucket in self.cache.items():
            if isinstance(bucket, CompactPriceSeries):
                max_ts = bucket.max_timestamp_ms()
                tail_keys = {canonical_json_key(it) for it in bucket[-self.DEDUP_WINDOW :]}
                watermarks[sym_k] = (len(bucket), max_ts, tail_keys)
            elif self._is_bucket_seq(bucket):
                max_ts = max((self._entry_timestamp_ms(it) for it in bucket), default=0)
                tail_keys = {canonical_json_key(it) for it in bucket[-self.DEDUP_WINDOW :]}
                watermarks[sym_k] = (len(bucket), max_ts, tail_keys)
        return original_state, watermarks

    def _collect_live_delta_by_watermark_locked(
        self,
        watermarks: dict[str, tuple[int, int, set[str]]],
        *,
        min_ts_ms: int = 0,
    ) -> dict[str, list]:
        """Collect in-memory ticks added while disk streaming ran unlocked (works on CompactPriceSeries & list)."""
        delta: dict[str, list] = {}
        for sym_k, bucket in self.cache.items():
            if not self._is_bucket_seq(bucket):
                continue
            wm = watermarks.get(sym_k)
            if wm is None:
                cands = list(bucket)
            else:
                snap_len, snap_max_ts, snap_tail_keys = wm
                cands = []
                seen_local: set[str] = set()
                # Fast-path: check items appended past snapshot length or newer than snapshot max_ts
                start_idx = snap_len if len(bucket) > snap_len else 0
                if start_idx > 0:
                    for item in bucket[start_idx:]:
                        k = canonical_json_key(item)
                        if k not in snap_tail_keys and k not in seen_local:
                            seen_local.add(k)
                            cands.append(item)
                for item in bucket:
                    if self._entry_timestamp_ms(item) > snap_max_ts:
                        k = canonical_json_key(item)
                        if k not in seen_local:
                            seen_local.add(k)
                            cands.append(item)
            extra = [
                item
                for item in cands
                if min_ts_ms <= 0 or self._entry_timestamp_ms(item) >= min_ts_ms
            ]
            if extra:
                delta[sym_k] = extra
        return delta

    def _merge_live_delta_locked(
        self, delta_by_sym: dict[str, list], *, restore_state=None
    ) -> None:
        if restore_state is not None:
            self._restore_memory_state_locked(restore_state)
        for sym_k, extra_items in delta_by_sym.items():
            if not extra_items:
                continue
            bucket = self.cache.get(sym_k)
            if bucket is None:
                bucket = self._new_symbol_bucket(sym_k)
                self.cache[sym_k] = bucket
            seen = {canonical_json_key(it) for it in bucket[-self.DEDUP_WINDOW :]}
            for item in extra_items:
                key = canonical_json_key(item)
                if key not in seen:
                    seen.add(key)
                    bucket.append(item)
            self._trim_memory_tail_locked(sym_k)

    def _reload_and_merge_live_delta_locked(
        self,
        watermarks: dict[str, tuple[int, int, set[str]]],
        original_fetchtimes,
        original_rest_cursors,
        *,
        min_ts_ms: Optional[int] = None,
    ) -> dict[str, list]:
        post_delta = self._collect_live_delta_by_watermark_locked(
            watermarks, min_ts_ms=0 if min_ts_ms is None else min_ts_ms
        )
        live_fetchtimes = {
            k: safe_nonneg_int(v, default=0) or 0 for k, v in dict(original_fetchtimes).items()
        }
        self._merge_max_nonneg_int_map(live_fetchtimes, self.fetchtime_per_symbol)
        live_rest_cursors = {
            k: safe_nonneg_int(v, default=0) or 0 for k, v in dict(original_rest_cursors).items()
        }
        self._merge_max_nonneg_int_map(live_rest_cursors, self._rest_sync_cursor_per_symbol)
        self._load_jsonl()
        self._merge_max_nonneg_int_map(self.fetchtime_per_symbol, live_fetchtimes)
        self._merge_max_nonneg_int_map(self._rest_sync_cursor_per_symbol, live_rest_cursors)
        self._merge_live_delta_locked(post_delta)
        return post_delta

    def _write_unpersisted_and_live_delta(
        self,
        target,
        original_state,
        watermarks: dict[str, tuple[int, int, set[str]]],
        *,
        min_ts_ms: Optional[int] = None,
    ) -> bool:
        wrote = False
        eff_min_ts_ms = 0 if min_ts_ms is None else min_ts_ms
        original_cache, original_persisted_counts = original_state[0], original_state[1]
        for sym_k, items in original_cache.items():
            start = min(max(0, original_persisted_counts.get(sym_k, 0)), len(items))
            for item in items[start:]:
                if eff_min_ts_ms <= 0 or self._entry_timestamp_ms(item) >= eff_min_ts_ms:
                    target.write(format_jsonl_bucket_line(sym_k, item))
                wrote = True
        with self.lock:
            delta_by_sym = self._collect_live_delta_by_watermark_locked(
                watermarks, min_ts_ms=eff_min_ts_ms
            )
            for sym_k, extra_items in delta_by_sym.items():
                for item in extra_items:
                    target.write(format_jsonl_bucket_line(sym_k, item))
                    wrote = True
        return wrote

    def maintain_append_persist(self):
        """Stream-prune the full multi-year disk archive without holding ``self.lock`` during disk I/O."""
        cutoff_ms = math.trunc((_now_time() - self.RETENTION_DAYS * 86400) * 1000)
        if not self.save_state:
            pruned = False
            with self.lock:
                for sym_k, bucket in list(self.cache.items()):
                    if isinstance(bucket, CompactPriceSeries):
                        if bucket.prune_older_than(cutoff_ms) > 0:
                            pruned = True
                    elif self._is_bucket_seq(bucket):
                        kept = [it for it in bucket if self._entry_timestamp_ms(it) >= cutoff_ms]
                        if len(kept) != len(bucket):
                            self.cache[sym_k] = kept
                            pruned = True
                    if sym_k in self._persisted_counts:
                        self._persisted_counts[sym_k] = min(
                            self._persisted_counts[sym_k], len(self.cache[sym_k])
                        )
            return pruned
        if not os.path.exists(self.filename):
            return False
        changed = False
        backup_path = None
        original_state = None
        watermarks: dict[str, tuple[int, int, set[str]]] = {}
        try:
            with self._generation_lock():
                with self.lock:
                    drain_fn = getattr(self, "_drain_deferred_price_ticks_locked", None)
                    if callable(drain_fn):
                        drain_fn()
                    original_state, watermarks = self._snapshot_memory_watermark_locked()

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
                        target, original_state, watermarks, min_ts_ms=cutoff_ms
                    ):
                        changed = True

                with self.lock:
                    post_delta = self._reload_and_merge_live_delta_locked(
                        watermarks,
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
                    builtins.print(f"[{self.cls_name}][maintain] pruned expired/corrupt disk records")
                if os.path.getsize(self.filename) > self.MAX_FILE_BYTES:
                    return self._rotate_disk_archive()
                return True
        except OSError as exc:
            with self.lock:
                self._merge_live_delta_locked(
                    self._collect_live_delta_by_watermark_locked(watermarks),
                    restore_state=original_state,
                )
                if backup_path is not None and os.path.exists(backup_path):
                    try:
                        self._restore_previous_generation(backup_path)
                    except OSError:
                        pass
            builtins.print(f"[{self.cls_name}][Error] streaming maintenance {self.filename}: {exc}")
            return False
        finally:
            self._after_state_persisted()

    def compact_jsonl(self):
        """Never rebuild an existing multi-year archive from its bounded memory tail."""
        if os.path.exists(self.filename):
            return self.maintain_append_persist()
        return CachePriceManagerBase.compact_jsonl(self)

    def _rotate_disk_archive(self):
        """Archive an oversized file while retaining a complete-line tail without holding ``self.lock`` during disk I/O."""
        archive = self._archive_snapshot_path()
        with self._generation_lock():
            with self.lock:
                original_state, watermarks = self._snapshot_memory_watermark_locked()
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
                    self._write_unpersisted_and_live_delta(target, original_state, watermarks)
                with self.lock:
                    self._reload_and_merge_live_delta_locked(
                        watermarks, original_state[2], original_state[3]
                    )
                    meta_ok = self._write_meta(self._build_basic_meta_locked(self._persisted_counts))
            except Exception as exc:
                builtins.print(f"[{self.cls_name}][maintain] rotation rewrite failed: {exc}")
                meta_ok = False
            if not meta_ok:
                with self.lock:
                    self._merge_live_delta_locked(
                        self._collect_live_delta_by_watermark_locked(watermarks),
                        restore_state=original_state,
                    )
                    return self._restore_archive_after_failed_rotation_locked(archive)
            self._compress_and_prune_archive(archive)
            return True

    def get_series_arrays(
        self,
        symbol: str,
        *,
        since_ms: Optional[int] = None,
        resync: bool = False,
    ) -> tuple[Any, Any]:
        """Return ``(timestamps_ms_int64, prices_float64)`` directly from the compact C buffers."""
        if resync:
            self.resync_mem_file()
        cutoff_ms = safe_nonneg_int(since_ms, default=None) if since_ms is not None else None
        with self.lock:
            raw = self.cache.get(symbol)
            if raw is None:
                raw = casefold_get(self.cache, symbol, None)
            if isinstance(raw, CompactPriceSeries):
                return raw.to_numpy(since_ms=cutoff_ms)
        return CacheManagerInterface.get_series_arrays(self, symbol, since_ms=since_ms, resync=False)


class CacheSparsePriceManager(CacheBoundedJsonlArchiveMixin, CachePriceManagerBase):
    """Store sparse three-minute price samples (Level 3) for two years in JSONL."""

    FILE_PREFIX = CACHE_SPARSE_PRICE_PREFIX
    FILE_EXT = "jsonl"
    SPARSE_MEMORY_ROWS: int = 35_000

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

    def _memory_row_limit(self) -> int:
        return safe_pos_int(self.SPARSE_MEMORY_ROWS, default=35_000) or 35_000

    def update_cache_per_symbol(self, symbol, new_items):
        super().update_cache_per_symbol(symbol, new_items)
        with self.lock:
            self._trim_memory_tail_locked(symbol)


# =============================================================================
# 3. ROLLING 24-HOUR SERIES: Cache24PriceManager
# =============================================================================


class Cache24PriceManager(CachePriceManagerBase):
    """Collect maximum-granularity prices (Level 3) over the latest ``KEEP_HOURS``."""

    KEEP_HOURS = 24
    FILE_PREFIX = CACHE_24_PRICE_PREFIX
    FILE_EXT = "jsonl"
    _filter_fetchtimes_to_symbols = True
    _ENABLE_EQUITY_OHLC_REFRESH = True
    _SNAPSHOT_COMPACT_JSON = True
    _SNAPSHOT_FSYNC = False
    COMPACT_INTERVAL_SEC = 1800.0
    COMPACT_TRIMMED_ROWS_THRESHOLD = 4000
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
        append_persist: Optional[bool] = None,
        current_price_getter=None,
        current_price_mgr=None,
        data_sync_mode: Optional[str] = None,
    ):
        self._last_eq_ohlc_refresh_ts: dict[str, float] = {}
        self._last_24h_compact_mono: float = _monotonic_time()
        self._jsonl_trimmed_disk_rows: int = 0
        self._jsonl_needs_compact: bool = False
        eff_append_persist = (
            _parse_bool_flag(append_persist, default=False)
            if append_persist is not None
            else str(filename or "").strip().lower().endswith(".jsonl")
        )
        super().__init__(
            sync_ts,
            symbols,
            filename,
            append_mode=True,
            api_client=api_client,
            append_persist=eff_append_persist,
            read_only=read_only,
            current_price_getter=current_price_getter,
            current_price_mgr=current_price_mgr,
            data_sync_mode=data_sync_mode,
        )
        self._filter_maps_to_configured_symbols_locked()

    def _new_symbol_bucket(self, symbol: str) -> CompactPriceSeries:
        return CompactPriceSeries()

    def get_series_arrays(
        self,
        symbol: str,
        *,
        since_ms: Optional[int] = None,
        resync: bool = False,
    ) -> tuple[Any, Any]:
        """Return ``(ts_ms_int64, px_float64)`` sliced directly from ``CompactPriceSeries``."""
        if resync:
            self.resync_mem_file()
        cutoff_ms = safe_nonneg_int(since_ms, default=None) if since_ms is not None else None
        with self.lock:
            self._drain_deferred_price_ticks_locked()
            raw = self.cache.get(symbol)
            if raw is None:
                raw = casefold_get(self.cache, symbol, None)
            if isinstance(raw, CompactPriceSeries):
                return raw.to_numpy(since_ms=cutoff_ms)
        return super().get_series_arrays(symbol, since_ms=cutoff_ms, resync=False)

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
                    and (not isinstance(existing, (list, tuple)) or len(existing) < (5 if len(r) >= 5 else 3))
                ):
                    by_ts[ts_r] = list(r) if isinstance(r, tuple) else r
        merged_pts = [by_ts[k] for k in sorted(by_ts)]
        self.cache[symbol] = CompactPriceSeries(merged_pts)
        return merged_pts

    def _resolve_equity_ohlc_refresh_interval_sec(
        self,
        symbol: str,
        now_s: float,
        *,
        row_count: int,
        has_pos_vol_ohlc: bool,
    ) -> tuple[bool, float]:
        """Return ``(eq_rth_open, min_refresh_interval_sec)`` across RTH, extended hours (PRE/POST), and CLOSED."""
        eq_rth_open = bool(is_symbol_market_open(symbol, now_s))
        if eq_rth_open or row_count < 10 or not has_pos_vol_ohlc:
            min_interval = self.EQUITY_OHLC_RTH_REFRESH_SEC
        elif is_us_extended_hours_open(now_s):
            min_interval = round(self.EQUITY_OHLC_RTH_REFRESH_SEC * 2.0, 1)
        else:
            min_interval = self.EQUITY_OHLC_OFF_RTH_REFRESH_SEC
        return eq_rth_open, min_interval

    def _should_compact_before_append_locked(self) -> bool:
        if not self.append_persist:
            return False
        if getattr(self, "_jsonl_needs_compact", False):
            return True
        if super()._should_compact_before_append_locked():
            return True
        trimmed = getattr(self, "_jsonl_trimmed_disk_rows", 0)
        if trimmed <= 0:
            return False
        if trimmed >= self.COMPACT_TRIMMED_ROWS_THRESHOLD:
            return True
        return (_monotonic_time() - getattr(self, "_last_24h_compact_mono", 0.0)) >= self.COMPACT_INTERVAL_SEC

    def compact_jsonl(self):
        if not self.append_persist:
            return False
        with self._generation_lock():
            with self.lock:
                if self._filter_fetchtimes_to_symbols:
                    self._filter_maps_to_configured_symbols_locked()
                for s_trim in list(self.symbols):
                    self._trim_old_data(s_trim, min_batch=1)
                ok = super().compact_jsonl()
                if ok:
                    self._last_24h_compact_mono = _monotonic_time()
                    self._jsonl_trimmed_disk_rows = 0
                    self._jsonl_needs_compact = False
                return ok

    def _legacy_json_path(self) -> Optional[str]:
        if self.append_persist and isinstance(self.filename, str) and self.filename.endswith(".jsonl"):
            return self.filename[:-1]
        return None

    def _persisted_max_ts(self):
        ts = super()._persisted_max_ts()
        if ts > 0 or not self.append_persist or os.path.exists(self.filename):
            return ts
        legacy = self._legacy_json_path()
        if legacy:
            try:
                meta = read_json_dict(meta_file_path(legacy))
                if isinstance(meta, dict):
                    return safe_pos_int(meta.get("max_ts", 0), default=0) or 0
            except Exception:
                pass
        return 0

    def _load_jsonl(self):
        if (
            self.read_only
            and not os.path.exists(self.filename)
            and not os.path.exists(self._previous_generation_path())
        ):
            legacy = self._legacy_json_path()
            legacy_bak = previous_generation_path(legacy) if legacy else ""
            if legacy and (os.path.exists(legacy) or (legacy_bak and os.path.exists(legacy_bak))):
                try:
                    data = read_json_dict(
                        legacy,
                        backup_path=legacy_bak,
                        recover_previous=True,
                        repair_files=False,
                    )
                    if isinstance(data, dict):
                        with self.lock:
                            items = data.get("items", {})
                            if isinstance(items, dict):
                                self.cache = self._filter_reloaded_items_locked(items)
                            self._hydrate_fetchtimes_from_mapping_locked(
                                data, filter_symbols=self._filter_fetchtimes_to_symbols
                            )
                            self._persisted_counts = {s: len(v) for s, v in self.cache.items()}
                            self._last_disk_sync_max_ts = max(
                                self._persisted_max_ts(), self._mem_max_ts()
                            )
                        return
                except Exception:
                    pass
        super()._load_jsonl()
        with self.lock:
            self.cache = self._filter_reloaded_items_locked(self.cache)
            self._persisted_counts = {s: len(v) for s, v in self.cache.items()}
            self._last_disk_sync_max_ts = max(
                self._persisted_max_ts(), self._mem_max_ts()
            )

    @classmethod
    def save_ohlc_24h_snapshot(
        cls,
        cpath: str,
        symbol: str,
        points: Sequence[Any],
        *,
        now_ts: Optional[float] = None,
        preserve_fresh_equity_sec: float = 0.0,
    ) -> bool:
        """Canonical single-owner locked writer for Layer 3 ``prices/cache_ohlc_24h_<SYM>.json``."""
        if not cpath or not symbol:
            return False
        pts_list = [list(p) for p in points] if points else []
        if not pts_list:
            return False
        sym_clean = str(symbol).strip()
        sym_u = sym_clean.upper()
        with generation_transaction(cpath) as target_path:
            now_w = resolve_now_ts(now_ts)
            if preserve_fresh_equity_sec > 0.0 and is_equity_symbol(sym_u):
                cur_disk = read_json_dict_recovering(target_path, default=None, repair_files=True)
                cur_ts = normalize_epoch_sec(
                    cur_disk.get("ts") if isinstance(cur_disk, dict) else None, default=None
                )
                if cur_ts is not None and -5.0 <= (now_w - cur_ts) < preserve_fresh_equity_sec:
                    return False
            save_json_state(target_path, {"ts": now_w, "symbol": sym_clean, "points": pts_list})
            return True

    def refresh_equity_ohlc_series(
        self,
        *,
        now_ts: Optional[float] = None,
        force: bool = False,
    ) -> bool:
        """Refresh 5d/5m OHLCV bars in ``cacheManager`` for equity symbols and publish ``cache_ohlc_24h_<SYM>.json`` + ``365Price`` seed."""
        if self.read_only or not self._ENABLE_EQUITY_OHLC_REFRESH:
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
                raw_bucket = self.cache.get(sym_eq)
                if isinstance(raw_bucket, CompactPriceSeries):
                    row_count = len(raw_bucket)
                    has_pos_vol_ohlc = raw_bucket.has_positive_volume()
                else:
                    cur_rows_tmp = list(raw_bucket or ())
                    row_count = len(cur_rows_tmp)
                    has_pos_vol_ohlc = any(
                        isinstance(r, (list, tuple))
                        and len(r) >= 3
                        and (safe_pos_float(r[2], default=0.0) or 0.0) > 0.0
                        for r in cur_rows_tmp
                    )
            eq_rth_open, min_interval = self._resolve_equity_ohlc_refresh_interval_sec(
                sym_eq, eff_now, row_count=row_count, has_pos_vol_ohlc=has_pos_vol_ohlc
            )
            last_ref = refresh_map.get(sym_eq, -1e18)
            if not force and 0.0 <= (eff_now - last_ref) < min_interval:
                continue
            with self.lock:
                cur_rows = list(self.cache.get(sym_eq) or ())
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
                    if self.append_persist:
                        self._jsonl_needs_compact = True
                self._trim_old_data(sym_eq)
                if base_cachedb:
                    try:
                        ohlc24_raw = os.path.join(base_cachedb, cache_ohlc_24h_filename(sym_eq))
                        self.save_ohlc_24h_snapshot(ohlc24_raw, sym_eq, eff_pts, now_ts=eff_now)
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
        if self.save_state and self._ENABLE_EQUITY_OHLC_REFRESH:
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
        for s_trim in list(self.symbols):
            self._trim_old_data(s_trim)
        if persisted_max_ts <= self._last_disk_sync_max_ts:
            return
        backup_path = self._previous_generation_path()
        if not os.path.exists(self.filename) and not os.path.exists(backup_path):
            return
        try:
            if self.append_persist or str(self.filename or "").lower().endswith(".jsonl"):
                src_path = self.filename if os.path.exists(self.filename) else backup_path
                disk_items, _ = read_jsonl_bucket_map(src_path)
                if not disk_items and src_path != backup_path and os.path.exists(backup_path):
                    disk_items, _ = read_jsonl_bucket_map(backup_path)
                meta_p = meta_file_path(self.filename)
                meta_doc = read_json_dict(meta_p, default={}) if os.path.exists(meta_p) else {}
                disk_ft = (
                    meta_doc.get("fetchtime")
                    if isinstance(meta_doc, dict) and isinstance(meta_doc.get("fetchtime"), dict)
                    else {}
                )
            else:
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
            merged_any = False
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
                merged_any = True
                ft_cand = safe_pos_int(disk_ft.get(s), default=0) or 0
                if ft_cand > safe_nonneg_int(self.fetchtime_per_symbol.get(s), default=0):
                    self.fetchtime_per_symbol[s] = ft_cand
                if hasattr(self, "_trim_old_data"):
                    self._trim_old_data(s)
            if merged_any and self.append_persist:
                self._jsonl_needs_compact = True
        except Exception as exc:
            print(f"[{self.cls_name}][Error] merge disk OHLC {self.filename}: {exc}")

    def _filter_reloaded_items_locked(self, items: dict) -> dict:
        filtered = self._wrap_loaded_buckets_locked(
            {s: v for s, v in items.items() if s in self.symbols}
        )
        with self._deferred_ticks_lock:
            for sym_k, entries in filtered.items():
                if self._is_bucket_seq(entries) and entries:
                    last_ts = (
                        entries._ts[-1]
                        if isinstance(entries, CompactPriceSeries)
                        else self._entry_timestamp_ms(entries[-1])
                    )
                    if last_ts > self._last_accepted_tick_ts_ms.get(sym_k, 0):
                        self._last_accepted_tick_ts_ms[sym_k] = last_ts
        return filtered

    def _sync_last_accepted_tick_ts_locked(self) -> None:
        with self._deferred_ticks_lock:
            for sym_k, entries in self.cache.items():
                if self._is_bucket_seq(entries) and entries:
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
            self._trim_old_data(sym_k, min_batch=self.HOT_TRIM_MIN_BATCH)

    def _before_state_persist_locked(self) -> None:
        self._drain_deferred_price_ticks_locked()

    def _after_state_persisted(self) -> None:
        with self._deferred_ticks_lock:
            if not self._deferred_price_ticks:
                return
        with self.lock:
            self._drain_deferred_price_ticks_locked()

    HOT_TRIM_MIN_BATCH = 64

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
                self._trim_old_data(symbol, min_batch=self.HOT_TRIM_MIN_BATCH)
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

    def _trim_old_data(self, symbol, *, min_batch: int = 1):
        """Remove entries older than ``KEEP_HOURS`` (amortized when ``min_batch > 1``)."""
        cutoff_ms = math.trunc((_now_time() - self.KEEP_HOURS * 3600) * 1000)
        eff_batch = safe_pos_int(min_batch, default=1) or 1
        with self.lock:
            entries = self.cache.get(symbol)
            if not entries or not self._is_bucket_seq(entries):
                return
            probe_idx = eff_batch - 1
            if len(entries) <= probe_idx:
                return
            is_compact = isinstance(entries, CompactPriceSeries)
            probe_ts = (
                entries._ts[probe_idx]
                if is_compact
                else self._entry_timestamp_ms(entries[probe_idx])
            )
            if probe_ts >= cutoff_ms:
                return
            idx = (
                entries.bisect_left_ts(cutoff_ms)
                if is_compact
                else bisect.bisect_left(entries, cutoff_ms, key=self._entry_timestamp_ms)
            )
            if is_equity_symbol(symbol):
                idx = min(idx, max(0, len(entries) - 288))
            if idx >= eff_batch:
                if is_compact:
                    entries.trim_head(idx)
                else:
                    self.cache[symbol] = entries[idx:]
                if self.append_persist:
                    self._persisted_counts[symbol] = max(
                        0, self._persisted_counts.get(symbol, 0) - idx
                    )
                    self._jsonl_trimmed_disk_rows = (
                        getattr(self, "_jsonl_trimmed_disk_rows", 0) + idx
                    )

    def get_recent_entries(self, symbol: str, last_seconds: float) -> list:
        """Return ``[timestamp_ms, price]`` entries from the latest interval."""
        sec_val = safe_pos_float(last_seconds, default=None)
        if sec_val is None:
            return []
        cutoff_ms = math.trunc((_now_time() - sec_val) * 1000)
        with self.lock:
            self._drain_deferred_price_ticks_locked()
            entries = casefold_get(self.cache, symbol, [])
            if not entries or not self._is_bucket_seq(entries):
                return []
            idx = (
                entries.bisect_left_ts(cutoff_ms)
                if isinstance(entries, CompactPriceSeries)
                else bisect.bisect_left(entries, cutoff_ms, key=self._entry_timestamp_ms)
            )
            return list(entries[idx:])

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
        with generation_transaction(target_path) as target_path:
            os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
            is_jsonl = target_path.lower().endswith(".jsonl")
            if isinstance(existing_obj, dict):
                disk_obj = existing_obj
            elif is_jsonl:
                disk_obj = read_jsonl_or_json_dict_cached(target_path, _SSOT_24PRICE_DISK_MEMO) or {}
            elif os.path.exists(target_path) or os.path.exists(previous_generation_path(target_path)):
                disk_obj = read_json_dict(target_path, recover_previous=True, repair_files=True)
            else:
                disk_obj = {}
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
                write_meta_json(target_path, meta_payload, fsync=False)
                return True

            def _write_data_file() -> None:
                if is_jsonl:
                    with atomic_text_writer(target_path, fsync=False) as f_out:
                        for pt in points:
                            f_out.write(format_jsonl_bucket_line(symbol, pt))
                else:
                    atomic_write_json(target_path, payload, separators=(",", ":"), fsync=False)

            commit_file_and_meta(
                target_path,
                _write_data_file,
                _write_meta_ok,
                fsync=False,
            )

    @staticmethod
    def read_ssot_24price_entries(
        c24_path: str,
        cand: str,
        *,
        candidates: Optional[tuple[str, ...] | list[str]] = None,
        allow_single_item_fallback: bool = False,
    ) -> tuple[Optional[dict], Optional[list]]:
        """Read raw 24-hour price entries for ``cand`` from ``c24_path`` (supports ``.jsonl`` and legacy ``.json`` + ``.previous``)."""
        if not c24_path:
            return None, None
        use_cached_reader = (
            read_json_dict_recovering is _ORIG_READ_JSON_DICT_RECOVERING
            and read_json_dict is _ORIG_READ_JSON_DICT
            and read_json_dict_cached is _ORIG_READ_JSON_DICT_CACHED
            and is_state_io_dict_reader_unpatched()
        )
        if not use_cached_reader:
            raw = read_json_dict_recovering(c24_path, default=None)
            if not isinstance(raw, dict) and c24_path.endswith(".jsonl"):
                raw = read_json_dict_recovering(c24_path[:-1], default=None)
        else:
            cand_paths: list[str] = []
            if c24_path.endswith(".jsonl"):
                cand_paths.extend((c24_path, c24_path[:-1]))
            elif c24_path.endswith(".json"):
                cand_paths.extend((c24_path + "l", c24_path))
            else:
                cand_paths.append(c24_path)
            raw = None
            for cp in cand_paths:
                loaded = read_jsonl_or_json_dict_cached(
                    cp, _SSOT_24PRICE_DISK_MEMO, max_entries=_SSOT_24PRICE_MAX_ENTRIES
                )
                if isinstance(loaded, dict):
                    raw = loaded
                    break
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
            c24_fname = cls.disk_filename_for_symbol(cand)
            c24_path = (
                str(resolve_cachedb_filepath(os.path.join(str(custom_dir), c24_fname)))
                if custom_dir
                else cache_path(c24_fname)
            )
            try:
                _, entries = cls.read_ssot_24price_entries(c24_path, cand)
                if not entries or len(entries) < 2:
                    continue
                use_default_parser = (
                    cls.parse_ssot_price_point is CachePriceManagerBase.parse_ssot_price_point
                )
                valid: Optional[list[tuple[int, float]]] = None
                memo_sig: tuple[Any, ...] = ()
                parsed_key = (os.path.abspath(c24_path), str(cand).upper())
                e0, emid, elast = entries[0], entries[len(entries) // 2], entries[-1]
                entries_anchor = (e0, emid, elast)
                if use_default_parser:
                    memo_sig = (
                        len(entries),
                        _ssot_entry_memo_token(e0),
                        _ssot_entry_memo_token(emid),
                        _ssot_entry_memo_token(elast),
                    )
                    cached_parsed = _SSOT_24PRICE_PARSED_MEMO.get(parsed_key)
                    if (
                        cached_parsed is not None
                        and cached_parsed[0][0] is e0
                        and cached_parsed[0][1] is emid
                        and cached_parsed[0][2] is elast
                        and cached_parsed[1] == memo_sig
                        and cutoff_ms >= cached_parsed[3]
                    ):
                        valid = cached_parsed[2]
                if valid is None:
                    valid = []
                    prev_ts = -1
                    is_sorted = True
                    if use_default_parser:
                        for e in entries:
                            if (
                                type(e) in _LIST_OR_TUPLE_TYPES
                                and len(e) >= 2
                                and type(t0_v := e[0]) is int
                                and 100_000_000_000 <= t0_v < cutoff_ms
                            ):
                                continue
                            pt = _fast_parse_ssot_point(e) or CachePriceManagerBase.parse_ssot_price_point(e)
                            if pt is None or pt[0] < cutoff_ms:
                                continue
                            if pt[0] < prev_ts:
                                is_sorted = False
                            prev_ts = pt[0]
                            valid.append(pt)
                    else:
                        for e in entries:
                            pt = cls.parse_ssot_price_point(e)
                            if pt is None or pt[0] < cutoff_ms:
                                continue
                            if pt[0] < prev_ts:
                                is_sorted = False
                            prev_ts = pt[0]
                            valid.append(pt)
                    if not is_sorted:
                        valid.sort(key=_SSOT_PT_TS_KEY)
                    valid = cls.filter_isolated_price_points(valid)
                    if use_default_parser:
                        lru_dict_put(
                            _SSOT_24PRICE_PARSED_MEMO,
                            parsed_key,
                            (entries_anchor, memo_sig, valid, cutoff_ms),
                            _SSOT_24PRICE_MAX_ENTRIES,
                        )
                    start_idx = 0
                else:
                    start_idx = bisect.bisect_left(valid, cutoff_ms, key=_SSOT_PT_TS_KEY)
                if (len(valid) - start_idx) < 2:
                    continue
                if not _is_fresh_obs_ts(valid[-1][0] / 1000.0, now_s, max_latest_age):
                    continue
                if (
                    custom_dir is None
                    and lh_f > 0.25
                    and (valid[-1][0] - valid[start_idx][0])
                    < (lh_f * 3600.0 * 1000.0 * 0.75)
                ):
                    continue
                return [
                    {"timestamp": valid[k][0], "price": valid[k][1]}
                    for k in range(start_idx, len(valid))
                ]
            except Exception:
                continue
        return None


# =============================================================================
# 4. DENSE 365-DAY ARCHIVE: Cache365PriceManager
# =============================================================================


class Cache365PriceManager(CacheBoundedJsonlArchiveMixin, Cache24PriceManager):
    """Multi-month/multi-year dense archive variant (Level 3) using compact incremental JSONL persistence."""

    KEEP_HOURS = CM_LONG_ARCHIVE_MONTHS * 30 * 24
    RETENTION_DAYS = CM_LONG_ARCHIVE_MONTHS * 30
    LONG_TRIM_INTERVAL_SEC = 24 * 3600
    FILE_PREFIX = CACHE_365_PRICE_PREFIX
    FILE_EXT = "jsonl"
    _ENABLE_EQUITY_OHLC_REFRESH = False
    _SNAPSHOT_FSYNC = True

    @classmethod
    def seed_initial_archive_if_missing(cls, target_path: str, symbol: str, points: list) -> bool:
        """Bootstrap ``cache_365_price_<SYM>.jsonl`` once when missing on disk without overwriting an existing archive."""
        if not target_path or not isinstance(points, (list, tuple)) or not points:
            return False
        resolved_target = str(resolve_cachedb_filepath(target_path, migrate=True))
        if cachedb_exists(resolved_target, include_previous=True):
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
        with generation_transaction(resolved_target) as eff_target:
            if cachedb_exists(eff_target, include_previous=True):
                return False
            atomic_write_jsonl(eff_target, rows)
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

    def _memory_row_limit(self) -> int:
        return safe_pos_int(CM_LONG_ARCHIVE_MEMORY_ROWS, default=10_000) or 10_000

    def _trim_old_data(self, symbol, *, min_batch: int = 1):
        req_batch = safe_pos_int(min_batch, default=1) or 1
        eff_batch = (
            1
            if req_batch <= 1
            else min(req_batch, max(1, self._memory_row_limit() // 16))
        )
        with self.lock:
            self._trim_memory_tail_locked(symbol, min_batch=eff_batch)
        now = _monotonic_time()
        if now - self._last_long_trim < self.LONG_TRIM_INTERVAL_SEC:
            return
        self._last_long_trim = now
        Cache24PriceManager._trim_old_data(self, symbol, min_batch=1)


# =============================================================================
# 5. REAL-TIME SPOT & STREAM FANOUT: CacheCurrentPriceManager
# =============================================================================


class CacheCurrentPriceManager(CachePriceManagerBase):
    """Maintain the latest timestamped price per symbol (Level 3, WebSocket-first with HTTP fallback)."""

    WS_TIMEOUT_SEC = PRICE_WS_TIMEOUT_SEC
    STALE_THRESHOLD_MS = PRICE_FRESHNESS_MAX_AGE_MS
    FREQ_WINDOW_SEC = 60
    _SNAPSHOT_COMPACT_JSON = True
    _SNAPSHOT_FSYNC = False
    DEFAULT_FILENAME = CACHE_CURRENT_PRICE_FILENAME
    _flush_before_poll = True
    # Quarantine impossible single-tick price jumps (e.g. cross-symbol leaks or exchange glitches)
    # until corroborated by a second independent tick within OUTLIER_CONFIRM_WINDOW_MS.
    OUTLIER_SHORT_WINDOW_MS = 60_000
    OUTLIER_SHORT_MAX_RATIO = 2.0
    OUTLIER_MEDIUM_WINDOW_MS = 1_800_000
    OUTLIER_MEDIUM_MAX_RATIO = 3.0
    OUTLIER_CONFIRM_WINDOW_MS = 30_000
    OUTLIER_CONFIRM_TOLERANCE = 1.25

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
        self._owned_ws_streams: list[Any] = []
        self._attached_ws_venues: set[str] = set()
        self._ws_startup_grace_until_mono: dict[str, float] = {}
        self._ws_default_event_ts = 0.0
        self._ws_last_event_ts_value = 0.0
        self._ws_last_event_ts_per_symbol: dict = {}
        self._ws_stream_event_ts_per_symbol: dict = {}
        self._update_timestamps: dict = defaultdict(deque)
        self._rth_empty_backoff_ts: dict = {}
        self._empty_backoff_mono: dict[str, float] = {}
        self._empty_http_backoff_ts: dict[str, float] = {}
        self._empty_http_streaks: dict[str, int] = {}
        self._err_backoff_until_mono: dict[str, float] = {}
        self._err_streaks: dict[str, int] = {}
        self._last_http_poll_ts_per_symbol: dict[str, float] = {}
        self._last_equity_poll_mono: dict[str, float] = {}
        self._pending_price_outliers: dict[str, tuple[int, float]] = {}
        self._price_anchor_confirmations: dict[str, int] = {}
        self._fallback_pool_lock = threading.Lock()
        self._fallback_pool: Optional[_futures.ThreadPoolExecutor] = None
        self._rest_slot_tracker = NonBlockingRestSlotTracker()
        self._in_flight_http_futs = self._rest_slot_tracker.slots
        self._sync_wake_event = threading.Event()
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
        resolved_init_path = cache_path(filename) if filename else ""
        file_existed_before_init = bool(
            resolved_init_path
            and (
                os.path.exists(resolved_init_path)
                or os.path.exists(previous_generation_path(resolved_init_path))
                or (filename != resolved_init_path and os.path.exists(filename))
            )
        )
        super().__init__(
            sync_ts,
            symbols,
            filename,
            append_mode=False,
            api_client=api_client,
            read_only=read_only,
            data_sync_mode=data_sync_mode,
        )
        if file_existed_before_init:
            with self.lock:
                for sym_k, rows in self.cache.items():
                    if self.extract_price_and_ts(rows)[0] is not None:
                        self._price_anchor_confirmations[sym_k] = max(
                            self._price_anchor_confirmations.get(sym_k, 0), 2
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

    def _check_price_outlier_locked(
        self, symbol: str, ts_val: int, price_val: float
    ) -> Optional[tuple[float, float, int]]:
        """Quarantine an unconfirmed single-tick price jump vs an established cached quote.

        Returns ``(prev_px, ratio, dt_ms)`` when the tick must be rejected, or ``None``
        when the tick is normal, replaces an unconfirmed 1-tick cold-start seed, or is
        corroborated by a second distinct tick.
        """
        pending_map = getattr(self, "_pending_price_outliers", None)
        if pending_map is None:
            pending_map = {}
            self._pending_price_outliers = pending_map
        conf_map = getattr(self, "_price_anchor_confirmations", None)
        if conf_map is None:
            conf_map = {}
            self._price_anchor_confirmations = conf_map
        prev_px, prev_ts_ms = self.extract_price_and_ts(self.cache.get(symbol))
        if prev_px is None or prev_ts_ms is None or prev_px <= 0.0:
            pending_map.pop(symbol, None)
            conf_map[symbol] = 1
            return None
        dt_ms = ts_val - prev_ts_ms
        ratio = max(price_val / prev_px, prev_px / price_val)
        if -5_000 <= dt_ms <= self.OUTLIER_SHORT_WINDOW_MS:
            max_ratio = self.OUTLIER_SHORT_MAX_RATIO
        elif self.OUTLIER_SHORT_WINDOW_MS < dt_ms <= self.OUTLIER_MEDIUM_WINDOW_MS:
            max_ratio = self.OUTLIER_MEDIUM_MAX_RATIO
        else:
            max_ratio = None
        if max_ratio is None or ratio <= max_ratio:
            pending_map.pop(symbol, None)
            conf_map[symbol] = 1 if max_ratio is None else min(conf_map.get(symbol, 1) + 1, 10)
            return None
        # Do not let a single uncorroborated cold-start seed permanently block the first real tick.
        if conf_map.get(symbol, 0) < 2:
            pending_map.pop(symbol, None)
            conf_map[symbol] = 1
            return None
        pending = pending_map.get(symbol)
        if pending is not None:
            p_ts, p_px = pending
            if (
                0 < (ts_val - p_ts) <= self.OUTLIER_CONFIRM_WINDOW_MS
                and p_px > 0.0
                and max(price_val / p_px, p_px / price_val) <= self.OUTLIER_CONFIRM_TOLERANCE
            ):
                pending_map.pop(symbol, None)
                conf_map[symbol] = 2
                return None
        pending_map[symbol] = (ts_val, price_val)
        return prev_px, ratio, dt_ms

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
        rejected_outlier = None
        with self.lock:
            rejected_outlier = self._check_price_outlier_locked(symbol, ts_val, price_val)
            if rejected_outlier is None:
                self._empty_http_backoff_ts.pop(symbol, None)
                self._empty_http_streaks.pop(symbol, None)
                dq = self._update_timestamps[symbol]
                dq.append(now)
                while dq and dq[0] < cutoff:
                    dq.popleft()
        if rejected_outlier is not None:
            prev_px, ratio, dt_ms = rejected_outlier
            builtins.print(
                f"[{self.cls_name}][OUTLIER_REJECT] {symbol}: rejected unconfirmed tick "
                f"{price_val} vs prev {prev_px} (ratio={ratio:.2f}x, dt={dt_ms}ms)"
            )
            return
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
        now = _now_time()
        if existing_ts > 0 and existing_ts <= (now * 1000.0 + 60_000.0) and (existing_ts - ts_ms) > ooo_limit_ms:
            return
        self._ws_last_event_ts_value = now
        self._ws_last_event_ts_per_symbol[symbol] = now
        self._ws_stream_event_ts_per_symbol[symbol] = now
        self._ws_stream_event_ts_per_symbol[str(symbol).strip().upper()] = now
        self.push_price(symbol, price, ts_ms=ts_ms)

    @staticmethod
    def _clone_snapshot_items(cache_map: dict) -> dict:
        if not isinstance(cache_map, dict):
            return {}
        return {
            sym_k: (
                [list(pt) if isinstance(pt, (list, tuple)) else pt for pt in rows]
                if isinstance(rows, (list, tuple))
                else rows
            )
            for sym_k, rows in cache_map.items()
        }

    def _save_state_to_file_impl(self) -> bool:
        """Persist ``cache_current_price.json`` without holding ``self.lock`` across disk ``fsync``."""
        write_json_fn = _cm_override("atomic_write_json", atomic_write_json)
        commit_state: dict[str, Any] = {}
        with self._generation_lock():
            persisted_max_ts = self._persisted_max_ts()
            with self.lock:
                self._before_state_persist_locked()
                if persisted_max_ts > self._mem_max_ts():
                    builtins.print(
                        f"[{self.cls_name}][resync] file newer than memory -> "
                        f"refusing to overwrite with stale data ({self.filename})"
                    )
                    self._on_stale_memory_detected_locked()
                    return False

            def _write_latest_snapshot():
                with self.lock:
                    self._before_snapshot_commit_locked(persisted_max_ts)
                    items_snapshot = self._clone_snapshot_items(self.cache)
                    meta_snap = self._build_basic_meta_locked(sanitize_cursor=True)
                    snap_max_ts = self._mem_max_ts()
                commit_state["meta"] = meta_snap
                commit_state["snap_max_ts"] = snap_max_ts
                payload = {
                    "items": items_snapshot,
                    "fetchtime": meta_snap["fetchtime"],
                    "rest_sync_cursor": meta_snap["rest_sync_cursor"],
                }
                if write_json_fn is atomic_write_json:
                    return write_json_fn(
                        self.filename,
                        payload,
                        separators=(",", ":"),
                        fsync=self._SNAPSHOT_FSYNC,
                    )
                return write_json_fn(self.filename, payload, indent=1)

            if not self._commit_file_and_meta_locked(
                _write_latest_snapshot,
                lambda: commit_state["meta"],
            ):
                return False
            persisted_after = self._persisted_max_ts()
        committed_ts = max(persisted_after, commit_state.get("snap_max_ts", 0))
        with self.lock:
            self._last_persisted_snapshot_max_ts = committed_ts
            self._last_disk_sync_max_ts = committed_ts
        return self._log_cache_saved_ok()

    def _get_or_create_fallback_pool(self, worker_hint: int) -> _futures.ThreadPoolExecutor:
        with self._fallback_pool_lock:
            if self._fallback_pool is None:
                eff_hint = max(worker_hint, len(getattr(self, "symbols", ()) or ()))
                self._fallback_pool = _futures.ThreadPoolExecutor(
                    max_workers=max(2, min(12, eff_hint)),
                    thread_name_prefix="CPFallbackFetch",
                )
            return self._fallback_pool

    def _shutdown_fallback_pool(self) -> None:
        with self._fallback_pool_lock:
            pool = self._fallback_pool
            self._fallback_pool = None
            tracker = getattr(self, "_rest_slot_tracker", None)
            if tracker is not None:
                tracker.clear()
            else:
                self._in_flight_http_futs.clear()
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
            try:
                for th in list(getattr(pool, "_threads", ())):
                    _futures.thread._threads_queues.pop(th, None)
            except Exception:
                pass

    def _is_live_network_mode(self) -> bool:
        stop_ev = getattr(self, "_stop_event", None)
        return bool(
            getattr(self, "market_api", None) is _market_api.api
            and is_live_network_backfill_enabled()
            and not (stop_ev is not None and stop_ev.is_set())
        )

    def _is_live_daemon_poll_mode(self) -> bool:
        return bool(self.save_state and self._is_live_network_mode())

    def _is_empty_http_backed_off(self, symbol: str, now_s: float) -> bool:
        backoff_map = getattr(self, "_empty_http_backoff_ts", None)
        if not isinstance(backoff_map, dict):
            return False
        last_empty = backoff_map.get(symbol, 0.0)
        if last_empty <= 0.0:
            return False
        streaks = getattr(self, "_empty_http_streaks", None)
        streak = streaks.get(symbol, 1) if isinstance(streaks, dict) else 1
        base_ttl = self.ws_timeout_sec_for(symbol)
        ttl_s = (
            min(300.0, base_ttl * (2.0 ** min(streak - 1, 5)))
            if streak >= 2 and self._cached_entry_ts_ms(symbol) is None
            else base_ttl
        )
        return 0.0 <= (now_s - last_empty) < ttl_s

    def _record_empty_http_result(self, symbol: str, *, ok: bool) -> None:
        backoff_map = getattr(self, "_empty_http_backoff_ts", None)
        streaks = getattr(self, "_empty_http_streaks", None)
        if isinstance(backoff_map, dict):
            if ok:
                backoff_map.pop(symbol, None)
                if isinstance(streaks, dict):
                    streaks.pop(symbol, None)
            else:
                backoff_map[symbol] = _now_time()
                if isinstance(streaks, dict):
                    streaks[symbol] = streaks.get(symbol, 0) + 1

    def _cached_entry_ts_ms(self, symbol: str) -> Optional[int]:
        sym_u, sym_clean = _norm_symbol_keys(symbol)
        with self.lock:
            entries = (
                self.cache.get(symbol)
                or casefold_get(self.cache, sym_u)
                or casefold_get(self.cache, sym_clean)
            )
            _, cur_ts_ms = self.extract_price_and_ts(entries)
        return cur_ts_ms

    def _record_poll_error_backoff(
        self, symbol: str, exc: object, *, now_mono: Optional[float] = None
    ) -> float:
        """Record monotonic exponential error backoff (aggressive on HTTP 429/rate-limit)."""
        streaks = getattr(self, "_err_streaks", None)
        if not isinstance(streaks, dict):
            streaks = {}
            self._err_streaks = streaks
        until_map = getattr(self, "_err_backoff_until_mono", None)
        if not isinstance(until_map, dict):
            until_map = {}
            self._err_backoff_until_mono = until_map
        streak = streaks.get(symbol, 0) + 1
        streaks[symbol] = streak
        eff_interval = safe_pos_float(getattr(self, "sync_ts", None), default=PRICE_STREAM_POLL_INTERVAL_SEC) or PRICE_STREAM_POLL_INTERVAL_SEC
        err_l = str(exc or "").lower()
        if "429" in err_l or "rate limit" in err_l or "too many requests" in err_l:
            delay_s = min(120.0, max(15.0, eff_interval * 4.0) * (2.0 ** min(streak - 1, 3)))
        else:
            delay_s = min(30.0, max(eff_interval * 2.0, 0.5) * (2.0 ** min(streak - 1, 5)))
        ref_mono = _monotonic_time() if now_mono is None else now_mono
        until_map[symbol] = ref_mono + delay_s
        return delay_s

    def _clear_poll_error_backoff(self, symbol: str) -> None:
        if isinstance(streaks := getattr(self, "_err_streaks", None), dict):
            streaks.pop(symbol, None)
        if isinstance(until_map := getattr(self, "_err_backoff_until_mono", None), dict):
            until_map.pop(symbol, None)

    def _was_ws_updated_since(
        self, symbol: str, since_wall_ts: float, *, now_wall_ts: Optional[float] = None
    ) -> bool:
        """Return True when a valid (non-future) WebSocket tick arrived for ``symbol`` at or after ``since_wall_ts``."""
        ws_map = getattr(self, "_ws_stream_event_ts_per_symbol", None)
        if not isinstance(ws_map, dict):
            return False
        eff_now = safe_pos_float(now_wall_ts, default=None)
        if eff_now is None:
            eff_now = _now_time()
        sym_u = str(symbol or "").strip().upper()
        ws_cands = (
            ts_v
            for raw_ts in (ws_map.get(symbol), ws_map.get(sym_u))
            if (ts_v := safe_pos_float(raw_ts, default=None)) is not None
            and ts_v <= eff_now + 60.0
        )
        ws_ts = max(ws_cands, default=0.0)
        return ws_ts > 0.0 and ws_ts >= since_wall_ts

    def _symbol_ws_and_cache_fresh(
        self, symbol: str, *, is_live_daemon_poll: bool, now_ms: Optional[float] = None
    ) -> bool:
        if not self._ws_is_healthy(symbol):
            return False
        if not is_live_daemon_poll or self._is_equity_instrument(symbol):
            return True
        cur_ts_ms = self._cached_entry_ts_ms(symbol)
        if cur_ts_ms is None:
            return False
        eff_now_ms = safe_nonneg_float(now_ms, default=None)
        if eff_now_ms is None:
            eff_now_ms = _now_time() * 1000.0
        return -5_000.0 <= (eff_now_ms - cur_ts_ms) < (self.ws_timeout_sec_for(symbol) * 1000.0)

    def _equity_rth_min_gap_sec(self, symbol: str) -> float:
        sync_s = safe_pos_float(getattr(self, "sync_ts", None), default=PRICE_STREAM_POLL_INTERVAL_SEC) or PRICE_STREAM_POLL_INTERVAL_SEC
        return max(sync_s, round(self.ws_timeout_sec_for(symbol) * 0.6, 3))

    def _should_skip_equity_poll_for_session(
        self,
        symbol: str,
        now_s: float,
        *,
        now_mono: Optional[float] = None,
        is_live_daemon_poll: bool = False,
        mark_idle: bool = False,
    ) -> bool:
        """Return True when an equity remote price poll should be skipped under the active session/RTH gate.

        Allows live RTH polling (throttled in live daemon mode to ``_equity_rth_min_gap_sec``),
        throttled extended-hours (PRE/POST) polling at ``max(60s, 2 * ws_timeout_sec)``,
        and pauses polling during CLOSED overnight/weekend windows when a cached quote already exists.
        """
        eff_mono = _monotonic_time() if now_mono is None else now_mono
        empty_mono_map = getattr(self, "_empty_backoff_mono", None)
        last_empty_mono = empty_mono_map.get(symbol) if isinstance(empty_mono_map, dict) else None
        last_eq_mono_map = getattr(self, "_last_equity_poll_mono", None)
        last_eq_mono = last_eq_mono_map.get(symbol) if isinstance(last_eq_mono_map, dict) else None

        with self.lock:
            existing_entries = self.cache.get(symbol)
        has_cached = bool(
            existing_entries
            and self.extract_price_and_ts(existing_entries)[0] is not None
        )

        if not is_rth_gate_enabled(
            getattr(self, "_enforce_rth_gate", None), rth_fn=is_us_rth_open
        ):
            if is_live_daemon_poll and has_cached and last_eq_mono is not None:
                if (eff_mono - last_eq_mono) < self._equity_rth_min_gap_sec(symbol):
                    if mark_idle:
                        self.mark_symbol_ws_event(symbol, now_s)
                    return True
            return False

        rth_open = is_us_rth_open(now_s)
        ext_open = False if rth_open else is_us_extended_hours_open(now_s)
        if rth_open or ext_open:
            if last_empty_mono is not None and (eff_mono - last_empty_mono) < 60.0:
                return True
            if (last_eb := self._rth_empty_backoff_ts.get(symbol, 0.0)) > 0.0 and 0.0 <= (now_s - last_eb) < 60.0:
                return True
        if rth_open:
            if is_live_daemon_poll and has_cached and last_eq_mono is not None:
                if (eff_mono - last_eq_mono) < self._equity_rth_min_gap_sec(symbol):
                    if mark_idle:
                        self.mark_symbol_ws_event(symbol, now_s)
                    return True
            return False
        if ext_open:
            ext_gap_sec = max(60.0, round(self.ws_timeout_sec_for(symbol) * 2.0, 1))
            if has_cached and last_eq_mono is not None and (eff_mono - last_eq_mono) < ext_gap_sec:
                if mark_idle:
                    self.mark_symbol_ws_event(symbol, now_s)
                return True
            last_poll = self._last_http_poll_ts_per_symbol.get(symbol, 0.0)
            if last_poll > 0.0 and 0.0 <= (now_s - last_poll) < ext_gap_sec:
                if mark_idle:
                    self.mark_symbol_ws_event(symbol, now_s)
                return True
            cur_ts_ms = self._cached_entry_ts_ms(symbol)
            if cur_ts_ms is not None and -5_000.0 <= (now_s * 1000.0 - cur_ts_ms) < (ext_gap_sec * 1000.0):
                if mark_idle:
                    self.mark_symbol_ws_event(symbol, now_s)
                return True
            return False
        if has_cached or (last_empty_mono is not None and (eff_mono - last_empty_mono) < 300.0):
            if mark_idle:
                self.mark_symbol_ws_event(symbol, now_s)
            return True
        return False

    def query_remote_and_update_cache(self):
        wake_ev = getattr(self, "_sync_wake_event", None)
        if wake_ev is not None:
            wake_ev.clear()
        if not self.fetchtime_per_symbol:
            self._ensure_fetchtimes_initialized()
            self._rest_sync_cursor_per_symbol = dict(self.fetchtime_per_symbol)
        is_live_daemon_poll = self._is_live_daemon_poll_mode()
        is_live_net = is_live_daemon_poll or self._is_live_network_mode()
        min_poll_gap = max(
            0.45,
            min(safe_pos_float(self.sync_ts, default=0.8) or 0.8, 0.75),
        )
        tracker = getattr(self, "_rest_slot_tracker", None)
        hard_timeout_sec = (
            safe_pos_float(
                getattr(self, "_rest_hard_timeout_sec", None),
                default=PRICE_FALLBACK_REST_HARD_TIMEOUT_SEC,
            )
            or PRICE_FALLBACK_REST_HARD_TIMEOUT_SEC
        )

        def _apply_remote_items(sym_done: str, new_items: Any, is_rth_done: bool) -> None:
            now_done = _now_time()
            now_done_mono = _monotonic_time()
            if is_live_net:
                self._record_empty_http_result(sym_done, ok=bool(new_items))
            if new_items:
                self._clear_poll_error_backoff(sym_done)
                self._persist_items(sym_done, new_items)
                self._last_http_poll_ts_per_symbol[sym_done] = now_done
                if is_rth_done:
                    self._last_equity_poll_mono[sym_done] = now_done_mono
                    self._rth_empty_backoff_ts.pop(sym_done, None)
                    self._empty_backoff_mono.pop(sym_done, None)
                    self.mark_symbol_ws_event(sym_done, now_done)
            elif is_rth_done:
                self._rth_empty_backoff_ts[sym_done] = now_done
                self._empty_backoff_mono[sym_done] = now_done_mono

        def _harvest_slots(target_syms: Sequence[str], *, cur_mono: float, cur_wall: float) -> set[str]:
            handled: set[str] = set()
            if tracker is None:
                return handled
            for sym_k in target_syms:
                state, val, slot = tracker.inspect_and_harvest(
                    sym_k, now_mono=cur_mono, hard_timeout_sec=hard_timeout_sec
                )
                if state == "IN_FLIGHT":
                    handled.add(sym_k)
                    continue
                if state == "HARD_TIMEOUT":
                    builtins.print(
                        f"[{self.cls_name}] {sym_k}: REST hard-timeout >{hard_timeout_sec:.1f}s -- resetting slot"
                    )
                    continue
                if state == "DONE_ERR":
                    delay_s = self._record_poll_error_backoff(sym_k, val, now_mono=cur_mono)
                    self._log_remote_fetch_error(f"HTTP fetch (backoff {delay_s:.1f}s)", sym_k, val)
                    if is_live_net:
                        self._record_empty_http_result(sym_k, ok=False)
                    handled.add(sym_k)
                    continue
                if state == "DONE_OK" and slot is not None:
                    ws_preempted = self._was_ws_updated_since(
                        sym_k, slot.submit_wall_ts, now_wall_ts=cur_wall
                    ) or (
                        not bool(slot.meta)
                        and self._symbol_ws_and_cache_fresh(
                            sym_k, is_live_daemon_poll=True, now_ms=cur_wall * 1000.0
                        )
                    )
                    if ws_preempted:
                        self._clear_poll_error_backoff(sym_k)
                        if is_live_net:
                            self._record_empty_http_result(sym_k, ok=True)
                        handled.add(sym_k)
                        continue
                    if (cur_mono - slot.submit_mono) <= hard_timeout_sec:
                        _apply_remote_items(sym_k, val, bool(slot.meta))
                        handled.add(sym_k)
            return handled

        now_mono = _monotonic_time()
        now_wall = _now_time()
        skip_submit_syms: set[str] = (
            _harvest_slots(tracker.active_symbols(), cur_mono=now_mono, cur_wall=now_wall)
            if (is_live_daemon_poll and tracker is not None and tracker.has_active())
            else set()
        )

        pending_polls: list[tuple[str, Any, bool]] = []
        now_ms_scan = now_wall * 1000.0
        err_until_map = getattr(self, "_err_backoff_until_mono", {})
        grace_map = getattr(self, "_ws_startup_grace_until_mono", {})
        for symbol in list(self.symbols):
            if symbol in skip_submit_syms:
                continue
            is_rth_sym = self._is_equity_instrument(symbol)
            if not is_rth_sym and self._symbol_ws_and_cache_fresh(
                symbol, is_live_daemon_poll=is_live_daemon_poll, now_ms=now_ms_scan
            ):
                continue
            if is_live_daemon_poll and now_mono < grace_map.get(symbol, 0.0):
                continue
            if now_mono < err_until_map.get(symbol, 0.0):
                continue
            if is_live_net and self._is_empty_http_backed_off(symbol, now_wall):
                continue
            if is_rth_sym:
                if self._should_skip_equity_poll_for_session(
                    symbol,
                    now_wall,
                    now_mono=now_mono,
                    is_live_daemon_poll=is_live_daemon_poll,
                    mark_idle=is_live_daemon_poll,
                ):
                    continue
            elif is_live_daemon_poll:
                last_poll = self._last_http_poll_ts_per_symbol.get(symbol, 0.0)
                if last_poll > 0.0 and 0.0 <= (now_wall - last_poll) < min_poll_gap:
                    cur_ts_ms = self._cached_entry_ts_ms(symbol)
                    if cur_ts_ms is not None and -5_000.0 <= (now_wall * 1000.0 - cur_ts_ms) < 1_800.0:
                        continue
            startTime = self.fetchtime_per_symbol.get(
                symbol, self.fallback_time_default
            )
            pending_polls.append((symbol, startTime, is_rth_sym))

        if not pending_polls:
            return True

        if not is_live_daemon_poll or tracker is None:
            stop_ev = getattr(self, "_stop_event", None)
            for symbol, startTime, is_rth_sym in pending_polls:
                if stop_ev is not None and stop_ev.is_set():
                    break
                new_items = self._fetch_and_apply_remote_items(symbol, startTime)
                now_done = _now_time()
                now_done_mono = _monotonic_time()
                if is_live_net:
                    self._record_empty_http_result(symbol, ok=bool(new_items))
                if is_rth_sym:
                    if not new_items:
                        self._rth_empty_backoff_ts[symbol] = now_done
                        self._empty_backoff_mono[symbol] = now_done_mono
                    else:
                        self._rth_empty_backoff_ts.pop(symbol, None)
                        self._empty_backoff_mono.pop(symbol, None)
                        self._last_equity_poll_mono[symbol] = now_done_mono
                        self._last_http_poll_ts_per_symbol[symbol] = now_done
                        self.mark_symbol_ws_event(symbol, now_done)
            return True

        pool = self._get_or_create_fallback_pool(len(pending_polls))
        submit_mono = _monotonic_time()
        submit_wall = _now_time()
        submitted_futs: dict[_futures.Future, str] = {}
        on_done_cb = (lambda _f: wake_ev.set()) if wake_ev is not None else None
        for sym_item, st_item, is_rth_sym in pending_polls:
            slot = tracker.submit(
                sym_item,
                pool,
                lambda _s=sym_item, _st=st_item: self.get_remote_items(_s, _st),
                now_mono=submit_mono,
                now_wall_ts=submit_wall,
                meta=is_rth_sym,
                on_done=on_done_cb,
            )
            if slot is not None:
                submitted_futs[slot.future] = sym_item
        if submitted_futs:
            micro_wait = min(0.05, max(0.01, PRICE_STREAM_POLL_INTERVAL_SEC * 0.1))
            done_now, _ = _futures.wait(submitted_futs.keys(), timeout=micro_wait)
            if done_now:
                _harvest_slots(
                    [submitted_futs[f] for f in done_now],
                    cur_mono=_monotonic_time(),
                    cur_wall=_now_time(),
                )
        return True

    def get_remote_items(self, symbol, startTime):
        try:
            provider_name = self.provider_name_for(symbol)
            now_s = _now_time()
            prefer_ext = bool(
                self._is_equity_instrument(symbol)
                and (is_us_rth_open(now_s) or is_us_extended_hours_open(now_s))
            )
            with live_price_fetch_context():
                if provider_name is None:
                    if prefer_ext:
                        try:
                            price = self.market_api.get_current_price(symbol=symbol, prefer_public_extended=True)
                        except TypeError:
                            if self.market_api is _market_api.api:
                                raise
                            price = self.market_api.get_current_price(symbol=symbol)
                    else:
                        price = self.market_api.get_current_price(symbol=symbol)
                else:
                    try:
                        price = (
                            self.market_api.get_current_price(
                                symbol=symbol, provider_name=provider_name, prefer_public_extended=True
                            )
                            if prefer_ext
                            else self.market_api.get_current_price(
                                symbol=symbol, provider_name=provider_name
                            )
                        )
                    except TypeError:
                        if prefer_ext:
                            try:
                                price = self.market_api.get_current_price(
                                    symbol=symbol, provider_name=provider_name
                                )
                            except TypeError:
                                if self.market_api is _market_api.api:
                                    raise
                                price = self.market_api.get_current_price(symbol=symbol)
                        else:
                            if self.market_api is _market_api.api:
                                raise
                            price = self.market_api.get_current_price(symbol=symbol)
            price_val = safe_pos_float(price, default=None)
            if price_val is None:
                return []
            ts_ms = math.trunc(_now_time() * 1000)
            return [[ts_ms, price_val]]
        except Exception as e:
            self._record_poll_error_backoff(symbol, e)
            return self._log_remote_fetch_error("HTTP fetch", symbol, e)

    def _persist_items(self, symbol, new_items):
        price, ts_ms = self.extract_price_and_ts(new_items)
        if price is None:
            return
        self.push_price(symbol, price, ts_ms=ts_ms)

    def _should_poll(self):
        tracker = getattr(self, "_rest_slot_tracker", None)
        if tracker is not None and tracker.has_active():
            return True
        if not self._is_live_daemon_poll_mode():
            return not self._ws_is_healthy()
        now_s = _now_time()
        now_ms = now_s * 1000.0
        now_mono = _monotonic_time()
        err_until_map = getattr(self, "_err_backoff_until_mono", {})
        grace_map = getattr(self, "_ws_startup_grace_until_mono", {})
        for sym_item in list(self.symbols):
            if now_mono < grace_map.get(sym_item, 0.0):
                continue
            if now_mono < err_until_map.get(sym_item, 0.0):
                continue
            if self._is_equity_instrument(sym_item):
                if (
                    not self._should_skip_equity_poll_for_session(
                        sym_item, now_s, now_mono=now_mono, is_live_daemon_poll=True
                    )
                    and not self._is_empty_http_backed_off(sym_item, now_s)
                ):
                    return True
            elif not self._symbol_ws_and_cache_fresh(
                sym_item, is_live_daemon_poll=True, now_ms=now_ms
            ):
                return True
        return False

    def _wait_for_next_sync(self, wait_sec: float) -> bool:
        if self._stop_event.is_set():
            return True
        tracker = getattr(self, "_rest_slot_tracker", None)
        if tracker is not None and tracker.has_done():
            return False
        wake_ev = getattr(self, "_sync_wake_event", None)
        if wake_ev is None:
            return bool(self._stop_event.wait(wait_sec))
        wake_ev.wait(timeout=wait_sec)
        return self._stop_event.is_set()

    def _next_sync_wait_sec(self) -> float:
        base_wait = safe_nonneg_float(self.sync_ts, default=0.8)
        if base_wait is None:
            base_wait = 0.8
        if not self.save_state or base_wait > 5.0:
            return base_wait
        tracker = getattr(self, "_rest_slot_tracker", None)
        if tracker is not None and tracker.has_done():
            return min(base_wait, 0.02)
        persisted_snap_ts = safe_nonneg_int(
            getattr(self, "_last_persisted_snapshot_max_ts", 0), default=0
        ) or 0
        if persisted_snap_ts > 0 and (self._mem_max_ts() - persisted_snap_ts) >= 800:
            return min(base_wait, 0.03)
        wait_sec = min(base_wait, 0.35)
        if tracker is not None and tracker.has_active():
            ht_sec = (
                safe_pos_float(
                    getattr(self, "_rest_hard_timeout_sec", None),
                    default=PRICE_FALLBACK_REST_HARD_TIMEOUT_SEC,
                )
                or PRICE_FALLBACK_REST_HARD_TIMEOUT_SEC
            )
            next_ht = tracker.next_hard_timeout_remaining(
                now_mono=_monotonic_time(), hard_timeout_sec=ht_sec
            )
            if next_ht is not None:
                wait_sec = min(wait_sec, max(0.02, next_ht))
        now = _now_time()
        is_live_daemon = self._is_live_daemon_poll_mode()
        for sym_item in list(self.symbols):
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
            eff_last_ts = last_ts
            if is_live_daemon:
                cur_ts_ms = self._cached_entry_ts_ms(sym_item)
                if cur_ts_ms is not None and cur_ts_ms > 0:
                    cur_ts_s = cur_ts_ms / 1000.0
                    if eff_last_ts <= 0.0 or cur_ts_s < eff_last_ts:
                        eff_last_ts = cur_ts_s
            if eff_last_ts > 0.0:
                rem = self.ws_timeout_sec_for(sym_item) - (now - eff_last_ts)
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
            live_net = self._is_live_network_mode()
            if live_net and self._is_empty_http_backed_off(target_sym, now_ms / 1000.0):
                return None
            age = age_ms if age_ms is not None else -1
            streaks = getattr(self, "_empty_http_streaks", None)
            if age_ms is not None or not (isinstance(streaks, dict) and streaks.get(target_sym, 0) >= 1):
                print(f"[{self.cls_name}] {target_sym} stale ({age}ms) - forced HTTP fetch")
            new = self.get_remote_items(target_sym, None)
            if not new or self.extract_price_and_ts(new)[0] is None:
                if live_net:
                    self._record_empty_http_result(target_sym, ok=False)
                return None
            if live_net:
                self._record_empty_http_result(target_sym, ok=True)
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

    def start_venue_ws_streams(
        self, *, start_ws: Optional[bool] = None, startup_grace_sec: float = 2.0
    ) -> list[Any]:
        """Start and attach all registered public price WebSocket streams matching ``self.symbols``."""
        if start_ws is None:
            enable_ws = (
                is_auto_start_websockets_enabled()
                and is_live_network_backfill_enabled()
            )
        else:
            enable_ws = _parse_bool_flag(start_ws, default=False)
        if not enable_ws:
            return []
        attached_venues = getattr(self, "_attached_ws_venues", None)
        if not isinstance(attached_venues, set):
            attached_venues = set()
            self._attached_ws_venues = attached_venues
        owned = getattr(self, "_owned_ws_streams", None)
        if not isinstance(owned, list):
            owned = []
            self._owned_ws_streams = owned
        grace_map = getattr(self, "_ws_startup_grace_until_mono", None)
        if not isinstance(grace_map, dict):
            grace_map = {}
            self._ws_startup_grace_until_mono = grace_map
        eff_grace = safe_nonneg_float(startup_grace_sec, default=2.0) or 0.0
        started: list[Any] = []
        for venue_key, ws_name, build_ws in (
            _VenuePriceStreamBase.iter_registered_venue_streams()
        ):
            v_syms = [
                s
                for s in list(self.symbols)
                if (
                    (raw_prov := self.provider_name_for(s))
                    and normalize_venue_name(raw_prov, default="") == venue_key
                )
                or (not raw_prov and is_primary_stream_venue(venue_key))
            ]
            if not v_syms or venue_key in attached_venues:
                continue
            if is_primary_stream_venue(venue_key) and self._ws_manager is not None:
                attached_venues.add(venue_key)
                continue
            ws_inst = build_ws(v_syms, cpm=self)
            if ws_inst is None:
                continue
            self.attach_ws_manager(ws_inst)
            if not getattr(ws_inst, "is_running", False):
                ws_inst.start(name=ws_name, daemon=True)
            attached_venues.add(venue_key)
            if not is_primary_stream_venue(venue_key) and ws_inst not in owned:
                owned.append(ws_inst)
            if eff_grace > 0.0:
                grace_until = _monotonic_time() + eff_grace
                for s_item in v_syms:
                    if not self.is_real_ws_healthy(s_item):
                        grace_map[s_item] = grace_until
            started.append(ws_inst)
        return started

    def _request_stop_owned_ws_streams(self) -> None:
        for ws in list(getattr(self, "_owned_ws_streams", ()) or ()):
            try:
                req_stop = getattr(ws, "request_stop", None)
                if callable(req_stop):
                    req_stop()
                elif (stop_ev := getattr(ws, "_stop_event", None)) is not None:
                    stop_ev.set()
            except Exception:
                pass

    def _stop_owned_ws_streams(self, timeout: float) -> None:
        owned = getattr(self, "_owned_ws_streams", None)
        if not owned:
            return
        eff_timeout = safe_nonneg_float(timeout, default=2.0)
        if eff_timeout is None:
            eff_timeout = 2.0
        ws_deadline = _monotonic_time() + min(2.0, eff_timeout)
        for ws in list(owned):
            try:
                rem = (
                    min(eff_timeout, max(0.05, ws_deadline - _monotonic_time()))
                    if eff_timeout > 0.0
                    else 0.0
                )
                ws.stop(timeout=rem)
            except Exception:
                pass
        owned.clear()
        if isinstance(attached := getattr(self, "_attached_ws_venues", None), set):
            attached.clear()

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
        if (wake_ev := getattr(self, "_sync_wake_event", None)) is not None:
            wake_ev.set()
        self._request_stop_owned_ws_streams()
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
        eff_timeout = safe_nonneg_float(timeout, default=5.0)
        if eff_timeout is None:
            eff_timeout = 5.0
        deadline = _monotonic_time() + eff_timeout
        self.request_stop()
        self._stop_owned_ws_streams(
            0.0 if eff_timeout == 0.0 else max(0.0, deadline - _monotonic_time())
        )
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
                price_path,
                read_json_dict_recovering,
                orig_reader_fn=_ORIG_READ_JSON_DICT_RECOVERING,
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


# =============================================================================
# 6. SSOT LOCAL PRICE RESOLUTION & WS STREAM LIFECYCLE
# =============================================================================


def _passive_dict_reader():
    return (
        read_json_dict_cached
        if read_json_dict_recovering is _ORIG_READ_JSON_DICT_RECOVERING
        and is_state_io_dict_reader_unpatched()
        else read_json_dict_recovering
    )


def resolve_ssot_price_from_cache(
    symbol: str,
    cache_dir: Optional[str] = None,
    max_age_sec: float = 60.0,
    *,
    quote_isolated: bool = True,
    now_ts: Optional[float] = None,
) -> Optional[float]:
    """Resolve a fresh positive spot price for ``symbol`` from SSOT cacheManager stores (0 ms local)."""
    if (raw_s := clean_scalar_str(symbol)) is None or not (sym_clean := clean_symbol_key(raw_s)):
        return None
    sym_u = raw_s.upper()
    if (max_age := resolve_positive_max_age(max_age_sec, 60.0)) is None:
        return None
    now_s = resolve_now_ts(now_ts)
    quotes = (
        compatible_quote_suffixes(raw_s, default_usd=DEFAULT_USD_QUOTES)
        if quote_isolated
        else ("USDC", "USDT", "USD", "EUR")
    )
    cands = list(
        dict.fromkeys(
            c
            for c in (raw_s, sym_clean, sym_u, *cache_symbol_variants(raw_s, quotes=quotes))
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
            resolve_cachedb_filepath(os.path.join(eff_dir, Cache24PriceManager.disk_filename_for_symbol(cand)))
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
    read_dict = _passive_dict_reader()
    try:
        cdata = read_dict(cache_file)
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

    lookup_sym = ensure_equity_suffix(sym_u, is_equity=inst_is_eq)
    series_cands = filter_candidates_by_asset_class(
        cache_symbol_variants(
            lookup_sym, include_original=True, include_base=False, quotes=allowed_quotes
        ),
        is_equity=inst_is_eq,
    )
    cands_upper = {c.strip().upper() for c in series_cands if c.strip()}
    try:
        for cand in series_cands:
            p24_path = str(
                resolve_cachedb_filepath(os.path.join(cache_dir, Cache24PriceManager.disk_filename_for_symbol(cand)))
            )
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
                jsonl_path = str(resolve_cachedb_filepath(os.path.join(cache_dir, jsonl_fn)))
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
                ob = read_dict(ob_path)
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
    """Signal all attached venue price WebSocket streams to stop."""
    if _bn_price_prov_cls is not None:
        _bn_price_prov_cls.request_stop_price_ws_manager()
    active_cpm = CacheManagerInterface.resolve_active_instance("CurrentPrice")
    if active_cpm is not None and hasattr(active_cpm, "_request_stop_owned_ws_streams"):
        active_cpm._request_stop_owned_ws_streams()


def stop_price_ws_streams(shutdown_deadline: float) -> None:
    """Stop all venue price WebSocket streams and the Binance periodic clock resync thread."""
    active_cpm = CacheManagerInterface.resolve_active_instance("CurrentPrice")
    if active_cpm is not None and hasattr(active_cpm, "_stop_owned_ws_streams"):
        active_cpm._stop_owned_ws_streams(
            min(1.0, max(0.05, shutdown_deadline - _monotonic_time()))
        )
    if _bn_price_prov_cls is not None:
        _bn_price_prov_cls.stop_price_ws_manager(
            timeout=min(1.0, max(0.35, shutdown_deadline - _monotonic_time()))
        )


# =============================================================================
# 7. SSOT SINGLETON FACTORY GETTERS
# =============================================================================

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
            else (parse_symbol_list(symbols) if isinstance(symbols, str) else _coerce_symbol_list(symbols))
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
    eff_read_only = not save_state_flag and not start_periodic
    wanted = {str(s).strip().upper() for s in eff_symbols if s}

    def _can_reuse_dict(inst: Any) -> bool:
        if not (isinstance(inst, dict) and eff_read_only):
            return False
        if not wanted.issubset({str(k).strip().upper() for k in inst}):
            return False
        for sym_k, mgr in inst.items():
            if str(sym_k).strip().upper() in wanted:
                CacheManagerInterface._resync_readonly_reader(mgr)
        return True

    return CacheManagerInterface._get_or_create_readonly_reader(
        _READONLY_PRICE_LOCK,
        _READONLY_PRICE_READERS,
        cache_name,
        can_reuse_fn=_can_reuse_dict,
        can_store=True,
        builder_fn=lambda: {
            s: manager_cls(
                sync_ts=default_sync_ts,
                filename=manager_cls.disk_filename_for_symbol(s),
                symbols=[s],
                read_only=eff_read_only,
                data_sync_mode=resolved_mode,
            )
            for s in eff_symbols
        },
    )


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


