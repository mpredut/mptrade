"""Trend cache managers and non-Binance trend poller extracted from cacheManager.py.

Provides:
- ``CacheLongTrendManager`` (alias ``CacheLongTrend``): ``CacheManagerInterface``
  subclass for long-horizon trend snapshots from ``priceanalysis.json``.
- ``CacheShortTrendManager`` (alias ``CacheShortTrend``): ``CacheManagerInterface``
  subclass for real-time multi-window short trends, O(1) incremental prefix-sum
  dynamic windows, and transactional cross-process snapshot persistence
  (``cache_short_trend.json`` + ``.meta`` + ``.previous`` recovery).
- ``get_short_trend_manager``: thread-safe singleton accessor.
- ``_NonBinanceTrendPoller``, ``_fetch_price_with_deadline``,
  ``_start_nonbinance_trend_poller``: bounded-deadline polling for non-Binance
  instruments feeding the ``CurrentPrice -> Cache24 -> ShortTrend`` chain.
"""

from __future__ import annotations

import bisect
import builtins
import concurrent.futures as _futures
import json
import math
import os
import sys
import threading
import time
import weakref
from collections import defaultdict
from typing import Optional

import utils as u
from cachepaths import (
    migrate_legacy_filename_if_needed as _migrate_legacy_filename_if_needed,
)
from state_io import (
    atomic_write_json,
    discard_generation_file,
    meta_file_path,
    read_recovering_previous,
)
from providers.base import extract_timestamp_s, normalize_timestamp_ms

_cm = sys.modules.get("cacheManager")
if _cm is None or not hasattr(_cm, "CacheManagerInterface"):
    _main_mod = sys.modules.get("__main__")
    if _main_mod is not None and hasattr(_main_mod, "CacheManagerInterface"):
        _cm = _main_mod
        sys.modules.setdefault("cacheManager", _main_mod)
    else:
        import cacheManager as _cm

CacheManagerInterface = _cm.CacheManagerInterface

# Dynamic window bounds for on-demand short-trend queries over Cache24.
CM_DYNAMIC_WINDOW_MIN_SEC = 5.0
CM_DYNAMIC_WINDOW_MAX_SEC = 24.0 * 3600.0
# Windows pre-published by the cacheManager writer into cache_short_trend.json["dynamic"]
# and cachedb/24trend/live_<SYMBOL>.json so reader processes (e.g. rtrade) have O(1)
# access without needing an in-process Cache24 subscription.
CM_PUBLISHED_TREND_WINDOWS_SEC = (
    float(round(3.7 * 60)),   # 222s primary momentum window
    15.0 * 60.0,              # 900s rtrade regime / trend filter window
    2.5 * 3600.0,             # 9000s secondary macro trend window
)

# A poll deadline prevents the loop from waiting on one slow provider call. The
# provider's own network timeout still owns the worker lifecycle.
_NB_FETCH_DEADLINE_SEC = 8


def _cm_attr(name: str, default=None):
    """Resolve an attribute from ``cacheManager`` if loaded, allowing test monkeypatching."""
    mod = sys.modules.get("cacheManager") or _cm
    if mod is not None and hasattr(mod, name):
        return getattr(mod, name)
    return default


def _write_json_atomic(path, data, **kwargs):
    fn = _cm_attr("atomic_write_json", atomic_write_json)
    return fn(path, data, **kwargs)


def _dynamic_window_key(window_seconds: float) -> str:
    """Key a clamped dynamic window in the published snapshot, e.g. 900.0 -> '900'."""
    return f"{float(window_seconds):g}"


class CacheLongTrendManager(CacheManagerInterface):
    """Long-horizon trend cache backed by ``priceanalysis.json``."""

    def __init__(self, sync_ts, symbols, filename, api_client=None, read_only=False):
        if api_client is None:
            api_client = _cm_attr("api", None)
        super().__init__(
            sync_ts,
            symbols,
            filename,
            append_mode=False,
            api_client=api_client,
            read_only=read_only,
        )

    def rebuild_fetchtimes(self):
        last_times = defaultdict(int)
        for symbol, items in self.cache.items():
            for item in items:
                ts_sec = extract_timestamp_s(item, default=0.0) or 0.0
                if ts_sec <= 0:
                    continue
                ts = int(ts_sec * 1000)
                if ts > last_times[symbol]:
                    last_times[symbol] = ts
        for symbol in last_times:
            last_times[symbol] = max(0, last_times[symbol] - 60_000)
        return dict(last_times)

    def get_remote_items(self, symbol, startTime):
        config_root = _cm_attr(
            "_CONFIG_ROOT",
            os.path.dirname(os.path.abspath(__file__)),
        )
        trend_filename = os.path.join(config_root, "priceanalysis.json")
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


CacheLongTrend = CacheLongTrendManager


class CacheShortTrendManager(CacheManagerInterface):
    """Calculate per-symbol short trends and share snapshots across processes.

    Inherits from ``CacheManagerInterface`` so that its on-disk snapshot
    (``cache_short_trend.json``) benefits from the same transactional commit
    and ``.previous`` generation crash recovery as all other cache managers,
    while preserving the top-level ``{symbol: snapshot_dict}`` JSON format
    expected by all direct readers.
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

    def __init__(
        self,
        symbols,
        filename="cache_short_trend.json",
        writer=False,
        window_seconds=None,
        thresholds=None,
        sync_ts=None,
        api_client=None,
        read_only=None,
    ):
        self.writer = bool(writer)   # Only the writer process persists the file.
        # Sort N window durations so index zero is primary.
        secs = list(window_seconds) if window_seconds else list(self.WINDOW_SECONDS)
        self.window_seconds = sorted(float(s) for s in secs)
        # Thresholds may be callable, keyed by window, keyed by symbol and window, or None.
        self._threshold_fn = self._build_threshold_fn(thresholds)
        self._live_windows = {}
        self._hist_windows = {}
        self._trend_buffers = {}
        self._active_dynamic_windows = set()
        self._disk_c24_cache = {}
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

        effective_sync_ts = (
            float(sync_ts) if sync_ts is not None else self.FULL_EVAL_INTERVAL_SEC
        )
        effective_api = api_client if api_client is not None else _cm_attr("api", None)
        effective_read_only = (
            (not self.writer) if read_only is None else bool(read_only)
        )
        super().__init__(
            effective_sync_ts,
            list(symbols),
            filename,
            append_mode=False,
            api_client=effective_api,
            read_only=effective_read_only,
        )
        self._live_instances.add(self)
        self.save_state = self.writer
        self.trend_dir = os.path.join(os.path.dirname(self.filename), "24trend")
        self._lock = self.lock
        self._mem = self.cache

    # ── CacheManagerInterface contract methods ───────────────────────────────
    def load_state(self):
        """Hydrate metadata and heal interrupted commits without pre-priming reader memory."""
        _migrate_legacy_filename_if_needed(self.filename)
        meta_path = meta_file_path(self.filename)
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as handle:
                    raw_meta = json.load(handle)
                if isinstance(raw_meta, dict):
                    ft = raw_meta.get("fetchtime")
                    if isinstance(ft, dict):
                        self.fetchtime_per_symbol = dict(ft)
                    rc = raw_meta.get("rest_sync_cursor")
                    if isinstance(rc, dict):
                        self._rest_sync_cursor_per_symbol = dict(rc)
                    elif self.fetchtime_per_symbol:
                        self._rest_sync_cursor_per_symbol = dict(
                            self.fetchtime_per_symbol
                        )
            except Exception:
                pass
        if os.path.exists(self._previous_generation_path()):
            self._read_file()

    def rebuild_fetchtimes(self):
        last_times = {}
        with self.lock:
            source = self._mem if self._mem else self._read_file()
        for symbol, snap in (source or {}).items():
            if not isinstance(snap, dict):
                continue
            ts_val = snap.get("ts") or snap.get("ts_fast")
            if isinstance(ts_val, (int, float)) and math.isfinite(ts_val) and ts_val > 0:
                last_times[symbol] = int(float(ts_val) * 1000)
        return last_times

    def get_remote_items(self, symbol, startTime):
        """Compute or return the current short-trend snapshot for ``symbol``."""
        self.evaluate_full(symbol)
        snap = self.get_snapshot(symbol)
        return [snap] if snap is not None else []

    # Extreme window durations derived from the sorted list.
    @property
    def window_small_sec(self):
        return self.window_seconds[0]

    @property
    def window_big_sec(self):
        return self.window_seconds[-1]

    def _live_trend_file(self, symbol: str) -> str:
        return os.path.join(self.trend_dir, f"live_{symbol}.json")

    def _hist_trend_file(self, symbol: str) -> str:
        return os.path.join(self.trend_dir, f"hist_{symbol}.json")

    def _get_trend_buffer(self, symbol: str):
        import pricewindow as pw
        buf = self._trend_buffers.get(symbol)
        if buf is None:
            buf = pw.IncrementalTrendBuffer()
            self._trend_buffers[symbol] = buf
        return buf

    def _build_threshold_fn(self, thresholds):
        """Normalize ``thresholds`` to a ``(symbol, seconds) -> percentage`` function."""
        if callable(thresholds):
            return thresholds
        small, big = self.window_seconds[0], self.window_seconds[-1]

        def per_window_default(sec):
            # The smallest window uses SMALL; all others use BIG.
            return (
                self.PRICE_CHANGE_THRESHOLD_SMALL
                if float(sec) <= small
                else self.PRICE_CHANGE_THRESHOLD_BIG
            )

        if isinstance(thresholds, dict):
            # Accept either per-symbol/per-window or direct per-window dictionaries.
            per_symbol = (
                all(isinstance(v, dict) for v in thresholds.values())
                and len(thresholds) > 0
            )
            if per_symbol:
                tbl = {
                    sym: {float(k): v for k, v in d.items()}
                    for sym, d in thresholds.items()
                }
                return lambda sym, sec: tbl.get(sym, {}).get(
                    float(sec), per_window_default(sec)
                )
            tbl = {float(k): v for k, v in thresholds.items()}
            return lambda sym, sec: tbl.get(float(sec), per_window_default(sec))

        return lambda sym, sec: per_window_default(sec)

    def threshold_for(self, symbol, seconds):
        """Return the percentage threshold for one symbol window."""
        return self._threshold_fn(symbol, float(seconds))

    # ── Writer: build windows and subscribe to Cache24 ───────────────────────
    def start_computation(
        self, cache24_managers=None, current_price_mgr=None, run_full_eval=False
    ):
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
            get_cm = _cm_attr("get_cache_manager")
            cache24_managers = get_cm("24Price")
        if current_price_mgr is None:
            get_cpm = _cm_attr("get_current_price_manager")
            current_price_mgr = get_cpm()
        self.current_price_mgr = current_price_mgr
        self._cache24_managers = cache24_managers   # Raw sources for dynamic windows.
        for s in self.symbols:
            try:
                c24 = cache24_managers[s]
                current_price_mgr.subscribe_price(c24)          # CurrentPrice → Cache24
                with getattr(c24, "lock", threading.RLock()):
                    raw_entries = list(getattr(c24, "cache", {}).get(s, []))
                if raw_entries:
                    self._get_trend_buffer(s).sync_entries(raw_entries)
                self.windows[s]   = {}
                self.analyzers[s] = {}
                parts = []
                for sec in self.window_seconds:
                    w = pw.PriceWindow.from_cache24(s, sec, c24)
                    self.windows[s][sec]   = w
                    self.analyzers[s][sec] = pw.WindowAnalyzer(w)
                    parts.append(
                        f"{sec:.0f}s: {len(w.prices)} (rate={w.sample_rate_sec:.2f}s)"
                    )
                c24.subscribe_price(self)                       # tick signal to the fast channel
                print(f"[ShortTrend][{s}] " + " ".join(parts))
            except Exception as _e:
                builtins.print(
                    f"[ShortTrend][{s}] setup failed ({_e}) — skipping; "
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
        observation = self.current_price_mgr.get_cached_price_observation(
            symbol, self.TREND_STALE_SEC
        )
        if observation is None:
            return None
        observed_at, current_price = observation

        # Calculate slopes for every window, keyed by seconds.
        slopes = {}
        primary_pos = None
        for sec in self.window_seconds:
            slope, pos = ans[sec].check_price_change(
                self._threshold_fn(symbol, sec)
            )
            slopes[sec] = slope
            if sec == primary:
                primary_pos = pos

        # Detailed metrics for primary window and published/active windows.
        pwin, pan = wins[primary], ans[primary]
        gradient, gc, slope_full, gradient_recent = pwin.get_short_trend()
        eps_primary = pwin.get_noise_epsilon(self.EPSILON_K)
        published_dyn = self._published_dynamic_windows(symbol, now=observed_at)
        primary_key = _dynamic_window_key(primary)
        if primary_key not in published_dyn:
            published_dyn[primary_key] = dict(
                final_trend=gradient,
                growth_coefficient=gc,
                slope_full=slope_full,
                gradient_recent=gradient_recent,
                epsilon=eps_primary,
                n_samples=len(pwin.prices),
                window_seconds=float(primary),
                sample_rate_sec=float(pwin.sample_rate_sec),
                current_price=current_price,
                ts=observed_at,
            )
        with self._lock:
            live_sym = self._live_windows.setdefault(symbol, {})
            for wk, wval in published_dyn.items():
                entry = dict(wval)
                entry.setdefault("current_price", current_price)
                live_sym[wk] = entry

        w_big = wins.get(self.window_seconds[-1])
        v1h = None
        adapt_re = None
        adapt_dca = None
        if w_big is not None:
            try:
                from intelligence.internal.state.volatility import (
                    adaptive_thresholds,
                    calculate_volatility_1h,
                )
                v1h = calculate_volatility_1h(
                    list(w_big.prices), w_big.sample_rate_sec
                )
                adapt_re, adapt_dca = adaptive_thresholds(v1h)
            except Exception:
                pass

        # Update memory only; the flush loop writes in the background.
        self._set_mem(
            symbol,
            final_trend=gradient,
            growth_coefficient=gc,
            slope_full=slope_full,
            gradient_recent=gradient_recent,
            slope_small=slopes[self.window_seconds[0]],     # Smallest.
            slope_big=slopes[self.window_seconds[-1]],      # Largest.
            slopes={f"{int(s)}": v for s, v in slopes.items()},
            slope_max_min=pan.calculate_slope_max_min(),
            pos=primary_pos,
            epsilon=eps_primary,
            vol_1h_pct=v1h,
            adapt_reentry_pct=adapt_re,
            adapt_dca_pct=adapt_dca,
            current_price=current_price,
            ts=observed_at,
            dynamic=published_dyn,
        )

    def _published_dynamic_windows(self, symbol, now=None):
        """Compute configured, secondary (e.g. 2.5h), and active dynamic windows."""
        published = {}
        published_cfg = _cm_attr(
            "CM_PUBLISHED_TREND_WINDOWS_SEC", CM_PUBLISHED_TREND_WINDOWS_SEC
        )
        target_windows = list(
            dict.fromkeys(
                list(self.window_seconds)
                + list(published_cfg)
                + sorted(self._active_dynamic_windows)
            )
        )
        for window in target_windows:
            dyn = self.get_short_trend_for_window(symbol, window, now=now)
            if dyn is not None:
                published[_dynamic_window_key(dyn["window_seconds"])] = dyn
        return published

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
                        print(f"[{self.cls_name}] evaluate_full {s}: {e}")
                self._stop_event.wait(self.FULL_EVAL_INTERVAL_SEC)

        self._full_eval_thread = threading.Thread(
            target=run, name="ShortTrendFullEval", daemon=True
        )
        self._full_eval_thread.start()

    # ── Fast Cache24 subscriber channel: gradient and epsilon on every tick ───
    def on_price_update(self, symbol, ts_ms, price):
        with self._lock:
            buf = self._get_trend_buffer(symbol)
            if not buf.timestamps:
                buf.append(price, ts_ms)
            elif int(ts_ms) >= buf.timestamps[-1] and not (
                int(ts_ms) == buf.timestamps[-1] and float(price) == buf.prices[-1]
            ):
                buf.append(price, ts_ms)
        win = self.get_window(symbol)   # Primary window.
        if win is None:
            return
        try:
            # The fast path computes only a cheap gradient in memory. It writes separate
            # ``_fast`` keys for low-latency gates and never overwrites richer full-evaluation
            # results. Sharing keys previously created a network-timing race between paths.
            g = win.get_recent_gradient()
            eps = win.get_noise_epsilon(self.EPSILON_K)
            self._set_mem(
                symbol,
                gradient_recent_fast=g,
                epsilon=eps,
                trend_fast=(1 if g > 0 else -1 if g < 0 else 0),
                current_price=price,
                ts_fast=float(ts_ms) / 1000.0,
            )
        except Exception as e:
            print(f"[{self.cls_name}] on_price_update {symbol}: {e}")

    # ── Calculation API ──────────────────────────────────────────────────────
    def get_window(self, symbol, seconds=None):
        """Return the requested window, or the smallest primary window for None."""
        wins = self.windows.get(symbol) or {}
        return wins.get(
            float(seconds) if seconds is not None else self.window_seconds[0]
        )

    def get_analyzer(self, symbol, seconds=None):
        """Return the requested analyzer, or the smallest primary analyzer for None."""
        ans = self.analyzers.get(symbol) or {}
        return ans.get(
            float(seconds) if seconds is not None else self.window_seconds[0]
        )

    def get_short_trend(self, symbol):
        win = self.get_window(symbol)
        return win.get_short_trend() if win else None

    # ── Dynamic window & Historical Subslice (O(1) Prefix Sums + Persistence) ──
    def _eval_entries_with_buffer(self, symbol, entries, a: int, b: int):
        """Evaluate slice [a, b] of entries using the symbol's IncrementalTrendBuffer."""
        import pricewindow as pw
        sub_entries = entries[a : b + 1]
        sample_rate = pw.PriceWindow._sample_rate_from_entries(sub_entries)
        with self._lock:
            buf = self._get_trend_buffer(symbol)
            buf.sync_entries(entries)
            if len(buf.prices) == len(entries):
                ft, gc, sf, gr, eps = buf.eval_slice(
                    a, b, sample_rate_sec=sample_rate, eps_k=self.EPSILON_K
                )
                return ft, gc, sf, gr, eps, sample_rate
        # Fallback if entries were non-monotonic or ad-hoc:
        tmp_buf = pw.IncrementalTrendBuffer()
        tmp_buf.sync_entries(sub_entries)
        ft, gc, sf, gr, eps = tmp_buf.eval_slice(
            0, len(sub_entries) - 1, sample_rate_sec=sample_rate, eps_k=self.EPSILON_K
        )
        return ft, gc, sf, gr, eps, sample_rate

    def _load_disk_cache24_entries(self, symbol):
        """Load raw [ts_ms, price] entries from cachedb/cache_24_price_<symbol>.json if present."""
        base_dir = os.path.dirname(self.filename)
        c24_file = os.path.join(base_dir, f"cache_24_price_{symbol}.json")
        if not os.path.exists(c24_file):
            legacy_file = os.path.join(base_dir, f"cache_24price_{symbol}.json")
            if os.path.exists(legacy_file):
                c24_file = legacy_file
            else:
                return []
        try:
            mtime = os.stat(c24_file).st_mtime_ns
            cached = self._disk_c24_cache.get(symbol)
            if cached and cached[0] == mtime:
                return cached[1]
            with open(c24_file, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            items = (
                (raw.get("items") or {}).get(symbol)
                if isinstance(raw, dict)
                else None
            )
            entries = list(items) if isinstance(items, list) else []
            self._disk_c24_cache[symbol] = (mtime, entries)
            return entries
        except Exception:
            return []

    def get_short_trend_for_window(self, symbol, window_seconds, now=None):
        """Calculate trend on demand for an arbitrary window in seconds.

        Uses O(1) incremental prefix sums over Cache24 and memoizes results in
        per-symbol persistence (cachedb/24trend/live_<SYMBOL>.json) so repeated
        requests for the same tick state are served directly from persistence.
        """
        req = float(window_seconds)
        min_sec = _cm_attr("CM_DYNAMIC_WINDOW_MIN_SEC", CM_DYNAMIC_WINDOW_MIN_SEC)
        max_sec = _cm_attr("CM_DYNAMIC_WINDOW_MAX_SEC", CM_DYNAMIC_WINDOW_MAX_SEC)
        clamped = min(max(req, min_sec), max_sec)
        if clamped != req:
            print(
                f"[get_short_trend_for_window] {symbol}: window_seconds={req:.0f} "
                f"outside [{min_sec:.0f}, {max_sec:.0f}] "
                f"-> clamped to {clamped:.0f}"
            )
        w_key = _dynamic_window_key(clamped)

        if self._cache24_managers is None:
            # Reader process: check published live_<SYMBOL>.json / shared file first,
            # then fall back to on-disk cache_24_price_<SYMBOL>.json if available.
            pub = self._published_window(symbol, clamped, now=now)
            if pub is not None:
                return pub
            disk_entries = self._load_disk_cache24_entries(symbol)
            if not disk_entries:
                return None
            now_val = now if now is not None else time.time()
            cutoff_ms = int((now_val - clamped) * 1000)
            a = bisect.bisect_left(disk_entries, cutoff_ms, key=lambda e: e[0])
            b = len(disk_entries) - 1
            if b - a + 1 < 3:
                return None
            newest_ts_sec = disk_entries[b][0] / 1000.0
            if now_val - newest_ts_sec > self.TREND_STALE_SEC:
                return None
            ft, gc, sf, gr, eps, sample_rate = self._eval_entries_with_buffer(
                symbol, disk_entries, a, b
            )
            res = dict(
                final_trend=ft,
                growth_coefficient=gc,
                slope_full=sf,
                gradient_recent=gr,
                epsilon=eps,
                n_samples=b - a + 1,
                window_seconds=clamped,
                sample_rate_sec=sample_rate,
                current_price=float(disk_entries[b][1]),
                ts=now_val,
            )
            self._persist_live_window(
                symbol, w_key, res, last_tick_ts_ms=disk_entries[b][0]
            )
            return res

        c24 = self._cache24_managers.get(symbol)
        if c24 is None:
            return None

        now_val = now if now is not None else time.time()
        self._active_dynamic_windows.add(clamped)

        # Fast path when c24 is a real Cache24PriceManager instance with direct buffer access:
        if hasattr(c24, "cache") and hasattr(c24, "lock"):
            with c24.lock:
                all_entries = list(c24.cache.get(symbol, []))
            if len(all_entries) < 3:
                return None
            cutoff_ms = int((now_val - clamped) * 1000)
            a = bisect.bisect_left(all_entries, cutoff_ms, key=lambda e: e[0])
            b = len(all_entries) - 1
            n_samples = b - a + 1
            if n_samples < 3:
                return None
            newest_ts_ms = int(all_entries[b][0])
            newest_ts_sec = newest_ts_ms / 1000.0
            if now_val - newest_ts_sec > self.TREND_STALE_SEC:
                return None
            with self._lock:
                cached = (self._live_windows.get(symbol) or {}).get(w_key)
                if (
                    isinstance(cached, dict)
                    and cached.get("_last_tick_ts_ms") == newest_ts_ms
                    and cached.get("n_samples") == n_samples
                ):
                    hit = {
                        k: v for k, v in cached.items() if not k.startswith("_")
                    }
                    hit["ts"] = now_val
                    return hit
            ft, gc, sf, gr, eps, sample_rate = self._eval_entries_with_buffer(
                symbol, all_entries, a, b
            )
            res = dict(
                final_trend=ft,
                growth_coefficient=gc,
                slope_full=sf,
                gradient_recent=gr,
                epsilon=eps,
                n_samples=n_samples,
                window_seconds=clamped,
                sample_rate_sec=sample_rate,
                current_price=float(all_entries[b][1]),
                ts=now_val,
            )
            self._persist_live_window(
                symbol, w_key, res, last_tick_ts_ms=newest_ts_ms
            )
            return res

        # Compatibility path for custom/mock c24 objects implementing only get_recent_entries:
        try:
            entries = c24.get_recent_entries(symbol, last_seconds=clamped)
        except Exception as e:
            print(
                f"[get_short_trend_for_window] {symbol}: get_recent_entries failed ({e})"
            )
            return None
        if len(entries) < 3:
            return None

        newest_ts_sec = entries[-1][0] / 1000.0
        if now_val - newest_ts_sec > self.TREND_STALE_SEC:
            return None

        ft, gc, sf, gr, eps, sample_rate = self._eval_entries_with_buffer(
            symbol, entries, 0, len(entries) - 1
        )
        res = dict(
            final_trend=ft,
            growth_coefficient=gc,
            slope_full=sf,
            gradient_recent=gr,
            epsilon=eps,
            n_samples=len(entries),
            window_seconds=clamped,
            sample_rate_sec=sample_rate,
            current_price=float(entries[-1][1]),
            ts=now_val,
        )
        self._persist_live_window(
            symbol, w_key, res, last_tick_ts_ms=int(entries[-1][0])
        )
        return res

    def _persist_live_window(
        self,
        symbol: str,
        w_key: str,
        data: dict,
        last_tick_ts_ms: Optional[int] = None,
        flush: bool = False,
    ):
        """Store a computed sliding window in memory and optionally flush 24trend/live_<SYMBOL>.json."""
        with self._lock:
            sym_live = self._live_windows.setdefault(symbol, {})
            stored = dict(data)
            if last_tick_ts_ms is not None:
                stored["_last_tick_ts_ms"] = int(last_tick_ts_ms)
            sym_live[w_key] = stored
            if flush and self.writer:
                self._write_live_symbol_file(symbol)

    def _write_live_symbol_file(self, symbol: str):
        """Persist cachedb/24trend/live_<SYMBOL>.json with clean window-keyed entries only."""
        try:
            sym_live = self._live_windows.get(symbol)
            if not sym_live:
                return
            clean = {
                wk: {k: v for k, v in wv.items() if not k.startswith("_")}
                for wk, wv in sym_live.items()
                if isinstance(wv, dict)
            }
            _write_json_atomic(self._live_trend_file(symbol), clean)
        except Exception as e:
            print(f"[{self.cls_name}] live write failure for {symbol}: {e}")

    def _read_live_symbol_file(self, symbol: str) -> dict:
        path = self._live_trend_file(symbol)
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                return data if isinstance(data, dict) else {}
        except Exception:
            pass
        return {}

    def _published_window(self, symbol, window_seconds, now=None):
        """Return a dynamic window published in 24trend/live_<SYMBOL>.json or shared file."""
        req = float(window_seconds)
        min_sec = _cm_attr("CM_DYNAMIC_WINDOW_MIN_SEC", CM_DYNAMIC_WINDOW_MIN_SEC)
        max_sec = _cm_attr("CM_DYNAMIC_WINDOW_MAX_SEC", CM_DYNAMIC_WINDOW_MAX_SEC)
        clamped = min(max(req, min_sec), max_sec)
        w_key = _dynamic_window_key(clamped)
        now = now if now is not None else time.time()

        def _check_fresh(dyn):
            if not isinstance(dyn, dict):
                return None
            try:
                age = float(now) - float(dyn["ts"])
            except (KeyError, TypeError, ValueError, OverflowError):
                return None
            if not -5.0 <= age <= self.TREND_STALE_SEC:
                return None
            return {k: v for k, v in dyn.items() if not k.startswith("_")}

        # 1. Check in-memory _live_windows first:
        with self._lock:
            mem_live = (self._live_windows.get(symbol) or {}).get(w_key)
        if isinstance(mem_live, dict):
            fresh = _check_fresh(mem_live)
            if fresh is not None:
                return fresh

        # 2. Check shared snapshot file dynamic dict (authoritative if present):
        snap = self._read_file().get(symbol)
        if isinstance(snap, dict):
            published = snap.get("dynamic")
            if isinstance(published, dict) and w_key in published:
                fresh = _check_fresh(published.get(w_key))
                if fresh is not None:
                    return fresh
            if (
                clamped == self.window_seconds[0]
                and "gradient_recent" in snap
                and "ts" in snap
            ):
                fresh = _check_fresh(snap)
                if fresh is not None:
                    return fresh

        # 3. Check per-symbol 24trend/live_<SYMBOL>.json:
        file_live = self._read_live_symbol_file(symbol).get(w_key)
        if isinstance(file_live, dict):
            return _check_fresh(file_live)

        return None

    # ── Historical fixed subslice [t1, t2] with immutable persistence ─────────
    def get_trend_for_slice(self, symbol: str, start_ts: float, end_ts: float):
        """Return trend metrics for a fixed historical interval [start_ts, end_ts].

        Serves directly from cachedb/24trend/hist_<SYMBOL>.json on cache hit; otherwise
        computes in O(1) from IncrementalTrendBuffer prefix sums and persists the result.
        Accepts timestamps in either seconds or milliseconds.
        """
        t1_ms = normalize_timestamp_ms(start_ts)
        t2_ms = normalize_timestamp_ms(end_ts)
        if t1_ms is None or t2_ms is None or t2_ms <= t1_ms:
            return None
        slice_key = f"{t1_ms}_{t2_ms}"

        with self._lock:
            sym_hist = self._hist_windows.setdefault(symbol, {})
            if slice_key in sym_hist:
                return dict(sym_hist[slice_key])
            hist_path = self._hist_trend_file(symbol)
            if os.path.exists(hist_path):
                try:
                    with open(hist_path, "r", encoding="utf-8") as fh:
                        loaded = json.load(fh)
                    if isinstance(loaded, dict):
                        sym_hist.update(loaded)
                        if slice_key in sym_hist:
                            return dict(sym_hist[slice_key])
                except Exception:
                    pass

        # Cache miss: resolve raw entries from memory Cache24 or disk Cache24
        entries = []
        if self._cache24_managers is not None:
            c24 = self._cache24_managers.get(symbol)
            if c24 is not None:
                if hasattr(c24, "cache") and hasattr(c24, "lock"):
                    with c24.lock:
                        entries = list(c24.cache.get(symbol, []))
                elif hasattr(c24, "entries"):
                    entries = list(c24.entries)
        if not entries:
            entries = self._load_disk_cache24_entries(symbol)
        if len(entries) < 3:
            return None

        a = bisect.bisect_left(entries, t1_ms, key=lambda e: e[0])
        b = bisect.bisect_right(entries, t2_ms, key=lambda e: e[0]) - 1
        if a < 0 or b >= len(entries) or (b - a + 1) < 3:
            return None

        ft, gc, sf, gr, eps, sample_rate = self._eval_entries_with_buffer(
            symbol, entries, a, b
        )
        res = dict(
            t1_ms=t1_ms,
            t2_ms=t2_ms,
            final_trend=ft,
            growth_coefficient=gc,
            slope_full=sf,
            gradient_recent=gr,
            epsilon=eps,
            n_samples=b - a + 1,
            window_seconds=round((t2_ms - t1_ms) / 1000.0, 3),
            sample_rate_sec=sample_rate,
        )
        with self._lock:
            sym_hist = self._hist_windows.setdefault(symbol, {})
            sym_hist[slice_key] = res
            try:
                _write_json_atomic(self._hist_trend_file(symbol), sym_hist)
            except Exception as e:
                print(f"[{self.cls_name}] hist write failure for {symbol}: {e}")
        return dict(res)

    def is_trend_up_for_slice(
        self, symbol: str, start_ts: float, end_ts: float
    ) -> bool:
        """Evaluate upward-trend condition on a fixed historical interval [start_ts, end_ts]."""
        res = self.get_trend_for_slice(symbol, start_ts, end_ts)
        return self._eval_upward_formula(res)

    @staticmethod
    def _eval_upward_formula(data) -> bool:
        """Single canonical upward-trend formula (SSOT): slope > 0 or (slope == 0 and gradient > 0)."""
        if not isinstance(data, dict):
            return False
        slope = float(
            data.get("gradient_recent", data.get("slope_small", 0.0)) or 0.0
        )
        gradient = float(data.get("final_trend", 0.0) or 0.0)
        return slope > 0 or (slope == 0 and gradient > 0)

    def is_trend_up(self, symbol, window_seconds=None, now=None) -> bool:
        """Evaluate upward-trend condition over a dynamic window or the default primary snapshot."""
        if (
            window_seconds is not None
            and float(window_seconds) != self.window_seconds[0]
        ):
            dyn = self.get_short_trend_for_window(
                symbol, window_seconds, now=now
            )
            return self._eval_upward_formula(dyn)
        snap = self.get_fresh_snapshot(symbol, now=now)
        if snap is not None:
            return self._eval_upward_formula(snap)
        if window_seconds is not None:
            dyn = self.get_short_trend_for_window(
                symbol, window_seconds, now=now
            )
            return self._eval_upward_formula(dyn)
        return False

    def is_trend_up_for_window(self, symbol, window_seconds, now=None):
        """Evaluate upward-trend condition over a configurable window in seconds."""
        dyn = self.get_short_trend_for_window(symbol, window_seconds, now=now)
        return self._eval_upward_formula(dyn)

    # ── Transactional cross-process JSON store (via CacheManagerInterface) ────
    def save_state_to_file(self):
        """Transactionally commit ``cache_short_trend.json`` and its ``.meta`` sidecar."""
        try:
            with self.lock:
                snapshot_mem = {
                    str(sym): dict(val)
                    for sym, val in self._mem.items()
                    if isinstance(val, dict)
                }
                fetchtime = {}
                for sym, val in snapshot_mem.items():
                    ts_val = val.get("ts") or val.get("ts_fast")
                    if (
                        isinstance(ts_val, (int, float))
                        and math.isfinite(ts_val)
                        and ts_val > 0
                    ):
                        fetchtime[sym] = int(float(ts_val) * 1000)
                    elif sym in self.fetchtime_per_symbol:
                        fetchtime[sym] = int(self.fetchtime_per_symbol[sym] or 0)
                self.fetchtime_per_symbol = dict(fetchtime)
                self._rest_sync_cursor_per_symbol = dict(fetchtime)
                counts = {sym: 1 for sym in snapshot_mem}
                metadata = {
                    "counts": counts,
                    "fetchtime": fetchtime,
                    "rest_sync_cursor": dict(fetchtime),
                }
                if not self._commit_file_and_meta_locked(
                    lambda: _write_json_atomic(self.filename, snapshot_mem),
                    metadata,
                ):
                    return False
                for symbol in list(self._live_windows.keys()):
                    self._write_live_symbol_file(symbol)
                return True
        except Exception as e:
            print(f"[{self.cls_name}] write failure for {self.filename}: {e}")
            return False

    def _write_file(self):
        return self.save_state_to_file()

    def _read_file(self):
        # Read through read_recovering_previous so an interrupted writer commit
        # automatically restores the prior valid generation (.previous).
        backup_path = self._previous_generation_path()
        if not os.path.exists(self.filename) and not os.path.exists(backup_path):
            return {}
        try:
            def _load_candidate(candidate_path):
                with open(candidate_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    raise ValueError("ShortTrend cache root must be a dict")
                if isinstance(data.get("items"), dict) and set(data.keys()) <= {
                    "items",
                    "fetchtime",
                    "rest_sync_cursor",
                    "data_version",
                }:
                    return dict(data["items"])
                return data

            return read_recovering_previous(
                self.filename, _load_candidate, backup_path=backup_path
            )
        except Exception:
            return {}

    def _set_mem(self, symbol, **fields):
        """Merge fields in memory only; the flush loop owns file I/O."""
        with self._lock:
            snap = dict(
                self._mem.get(symbol) or self._read_file().get(symbol) or {}
            )
            snap.update(fields)
            snap["symbol"] = symbol
            self._mem[symbol] = snap
            self.cache = self._mem
            ts_val = snap.get("ts") or snap.get("ts_fast")
            if (
                isinstance(ts_val, (int, float))
                and math.isfinite(ts_val)
                and ts_val > 0
            ):
                ts_ms = int(float(ts_val) * 1000)
                self.fetchtime_per_symbol[symbol] = ts_ms
                self._rest_sync_cursor_per_symbol[symbol] = ts_ms

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

        self._flush_thread = threading.Thread(
            target=run, name="ShortTrendFlush", daemon=True
        )
        self._flush_thread.start()

    def shutdown(self, timeout=5.0):
        """Stop evaluation, flush, and base sync loops, then unsubscribe Cache24 sources."""
        base_stopped = super().shutdown(timeout=timeout)
        for cache24 in (self._cache24_managers or {}).values():
            unsubscribe = getattr(cache24, "unsubscribe_price", None)
            if unsubscribe is not None:
                unsubscribe(self)
        for thread in (self._full_eval_thread, self._flush_thread):
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=timeout)
        stopped = base_stopped and all(
            thread is None or not thread.is_alive()
            for thread in (self._full_eval_thread, self._flush_thread)
        )
        if stopped:
            self._full_eval_thread = None
            self._flush_thread = None
            self._computing = False
        return stopped

    @classmethod
    def shutdown_all_instances(cls, timeout=5.0):
        results = [
            manager.shutdown(timeout=timeout)
            for manager in list(cls._live_instances)
        ]
        return all(results) if results else True

    def prime_from_file(self, symbols=None, *, overwrite=True):
        """Prime memory from startup data so a reader can calculate fresh trends locally."""
        data = self._read_file()
        allowed = (
            None if symbols is None else {str(symbol) for symbol in symbols}
        )
        with self._lock:
            for symbol, snap in data.items():
                if (
                    isinstance(snap, dict)
                    and (allowed is None or symbol in allowed)
                    and (overwrite or symbol not in self._mem)
                ):
                    self._mem[symbol] = dict(snap)
            self.cache = self._mem
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
        max_age_sec = (
            max_age_sec if max_age_sec is not None else self.TREND_STALE_SEC
        )
        now = time.time()
        if symbol is not None:
            snap = self.get_snapshot(symbol)
            return self._snapshot_is_usable(snap, now, max_age_sec)
        allt = self.get_all_snapshots()
        if not allt:
            return False
        return any(
            self._snapshot_is_usable(s, now, max_age_sec)
            for s in allt.values()
        )

    def become_writer(self):
        """Promote a calculating reader to writer when the previous writer is stale."""
        self.writer = True
        self.read_only = False
        self.save_state = True
        self._start_flush_loop()

    def get_snapshot_resilient(
        self,
        symbol,
        max_age_sec=None,
        cache24_managers=None,
        current_price_mgr=None,
    ):
        """Read resiliently with lazy failover.

        Use authoritative memory while calculating, otherwise use a fresh shared file.
        If the file is stale, start local calculation once, become writer, and use memory.
        """
        if self._computing:
            return self.get_snapshot(symbol)
        if self.is_snapshot_fresh(symbol, max_age_sec):
            return self.get_snapshot(symbol)
        # A stale file triggers autonomous local calculation and writer failover.
        builtins.print(
            f"[{self.cls_name}][WARN] stale file -> "
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
            self.cache = self._mem
            self.fetchtime_per_symbol.clear()
            self._rest_sync_cursor_per_symbol.clear()
            self._live_windows.clear()
            self._hist_windows.clear()
            self._trend_buffers.clear()
            self._active_dynamic_windows.clear()
            self._disk_c24_cache.clear()
            self._file_mtime = None
            self._file_cache = None
            try:
                for path in (
                    self.filename,
                    meta_file_path(self.filename),
                    self._previous_generation_path(),
                ):
                    discard_generation_file(path)
                if os.path.isdir(self.trend_dir):
                    for fn in os.listdir(self.trend_dir):
                        if fn.startswith(("live_", "hist_")) and fn.endswith(
                            ".json"
                        ):
                            os.remove(os.path.join(self.trend_dir, fn))
            except Exception:
                pass

    # ── Opportunistic-delay gate with informed epsilon ───────────────────────
    def _epsilon(self, snap):
        eps = snap.get("epsilon")
        if eps is not None and eps > 0:
            return float(eps)
        price = abs(snap.get("current_price") or 0.0)
        return price * self.FAVORABLE_REL_EPS

    def get_fresh_snapshot(self, symbol, now=None):
        """Return a symbol snapshot only when it is within ``TREND_STALE_SEC``.

        This is the shared public staleness check used by both trend and wait decisions.
        """
        snap = self.get_snapshot(symbol)
        now = now if now is not None else time.time()
        if not self._snapshot_is_usable(snap, now, self.TREND_STALE_SEC):
            return None
        return snap

    def should_wait(
        self,
        side,
        symbol,
        window_seconds=None,
        fast=False,
        use_noise_gate=True,
        now=None,
    ):
        """Decide whether to delay a BUY or SELL intent; True means wait.

        Unknown or stale data returns False and executes immediately, so a failed cache
        cannot block orders. ``window_seconds=None`` uses the precomputed primary window;
        another value calculates a bounded dynamic window where Cache24 is available.
        ``fast`` selects low-latency keys only for the primary window. The optional noise
        gate treats ``|gradient| <= epsilon`` as a reason to wait.
        """
        side = side.upper()

        if (
            window_seconds is not None
            and float(window_seconds) != self.window_seconds[0]
        ):
            dyn = self.get_short_trend_for_window(
                symbol, window_seconds, now=now
            )
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

        snap = self.get_fresh_snapshot(symbol, now=now)
        if snap is None:
            return False   # Unknown or stale data executes immediately.
        if fast:
            g = snap.get(
                "gradient_recent_fast", snap.get("gradient_recent", 0.0)
            )
        else:
            g = snap.get(
                "growth_coefficient", snap.get("gradient_recent", 0.0)
            )
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

    def wait_for_favorable_entry(
        self,
        side,
        symbol,
        max_wait_sec=10.0,
        poll_sec=0.2,
        sleep_fn=time.sleep,
        mode="full",
        window_seconds=None,
    ):
        """Wait while trend favors delay, bounded by ``max_wait_sec``.

        Emit a visual heartbeat roughly once per second and return elapsed seconds.
        Forward ``window_seconds`` to ``should_wait`` for primary or dynamic evaluation.
        """
        deadline = time.time() + max_wait_sec
        waited = 0.0
        next_dot = 1.0
        while time.time() < deadline and self.should_wait(
            side,
            symbol,
            window_seconds=window_seconds,
            fast=(mode == "gradient"),
        ):
            sleep_fn(poll_sec)
            waited += poll_sec
            if waited >= next_dot:
                print(".", end="", flush=True)
                next_dot += 1.0
        if waited > 0:
            print()
        return waited


CacheShortTrend = CacheShortTrendManager


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


def _start_nonbinance_trend_poller(
    cpm,
    symbols,
    interval_sec=10,
    fetch_deadline_sec=_NB_FETCH_DEADLINE_SEC,
    provider_names=None,
):
    """Feed short trends for non-WebSocket, non-Binance symbols.

    Poll through the facade and push into the CurrentPrice/Cache24/ShortTrend chain.
    Per-symbol failures do not affect Binance. A bounded poll wait and at least two
    workers keep one slow fetch from freezing the poller; provider timeouts remain visible.
    """
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
        max_workers=max(2, len(symbols)), thread_name_prefix="NBTrendFetch"
    )
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
                    builtins.print(
                        f"[NB-trend] {s}: fetch BLOCKED >{fetch_deadline_sec}s "
                        f"(poll deadline — probably DNS or the network) — skipping, retrying next cycle"
                    )
                except Exception as _e:
                    builtins.print(f"[NB-trend] {s}: {_e}")
            stop_event.wait(interval_sec)

    t = threading.Thread(target=run, name="NonBinanceTrendPoller", daemon=True)
    t.start()
    return _NonBinanceTrendPoller(t, pool, stop_event)
