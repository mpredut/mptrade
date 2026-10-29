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
from botcore import safe_float, safe_nonneg_float, safe_pos_float, safe_pos_int
from cachepaths import (
    CACHEDB_SUBDOMAINS,
    migrate_legacy_filename_if_needed as _migrate_legacy_filename_if_needed,
    normalize_venue_name,
    resolve_cachedb_filepath,
)
from state_io import (
    atomic_write_json,
    discard_generation_file,
    meta_file_path,
    previous_generation_path,
    read_json_dict,
    read_recovering_previous,
    restore_previous_generation,
    stage_previous_generation,
)
from providers.base import (
    _parse_bool_flag,
    extract_timestamp_s,
    normalize_side,
    normalize_timestamp_ms,
)

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
    90.0,                     # 90s fast momentum window (TAOUSDC)
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
    backup_path = None
    had_prior = os.path.exists(path)
    if had_prior:
        try:
            backup_path = stage_previous_generation(path)
        except Exception:
            backup_path = None
    try:
        res = fn(path, data, **kwargs)
        if backup_path is not None:
            discard_generation_file(backup_path)
        return res
    except BaseException:
        if backup_path is not None and os.path.exists(backup_path):
            try:
                restore_previous_generation(path, backup_path=backup_path)
            except Exception:
                pass
        elif not had_prior:
            discard_generation_file(path)
        raise


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
            if not isinstance(items, (list, tuple)):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
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
        default_root = os.path.dirname(os.path.abspath(__file__))
        if os.path.abspath(config_root) == os.path.abspath(default_root):
            from cachepaths import cache_path
            trend_filename = cache_path("priceanalysis.json")
            if not os.path.exists(trend_filename) and not os.path.exists(previous_generation_path(trend_filename)):
                legacy_root_pa = os.path.join(config_root, "priceanalysis.json")
                if os.path.exists(legacy_root_pa) or os.path.exists(previous_generation_path(legacy_root_pa)):
                    trend_filename = legacy_root_pa
        else:
            trend_filename = os.path.join(config_root, "priceanalysis.json")
        backup_path = previous_generation_path(trend_filename)
        if not os.path.exists(trend_filename) and not os.path.exists(backup_path):
            print(f"[{self.cls_name}] File {trend_filename} does not exist.")
            return []

        try:
            data = read_recovering_previous(
                trend_filename, read_json_dict, backup_path=backup_path
            )
        except Exception as exc:
            print(f"[{self.cls_name}] Could not read {trend_filename}: {exc}")
            return []

        if not isinstance(data, dict):
            return []
        for cand in _short_trend_symbol_variants(symbol):
            if cand in data:
                # An explicitly published null is a neutral snapshot, distinct from a
                # missing symbol. Preserve it so stale in-memory trends are cleared.
                return [data[cand]]
        return []


CacheLongTrend = CacheLongTrendManager


def _short_trend_symbol_variants(symbol) -> list:
    """Return ordered symbol candidates (exact symbol first, then cross-venue aliases)."""
    from cachepaths import cache_symbol_variants
    return cache_symbol_variants(
        symbol,
        include_original=True,
        include_base=True,
        quotes=("USDC", "USDT", "USD"),
    )


class CacheShortTrendManager(CacheManagerInterface):
    """Calculate per-symbol short trends and share snapshots across processes.

    Inherits from ``CacheManagerInterface`` so that its on-disk snapshot
    (``cache_short_trend.json``) benefits from the same transactional commit
    and ``.previous`` generation crash recovery as all other cache managers,
    while preserving the top-level ``{symbol: snapshot_dict}`` JSON format
    expected by all direct readers.
    """

    EPSILON_K         = 1.0     # Informed noise threshold: k * stddev(gradient).
    UPWARD_EPSILON_K  = 0.25    # Minimum growth_coefficient / epsilon ratio to confirm upward trend above noise.
    FAVORABLE_REL_EPS = 1e-5    # Price-relative fallback when epsilon is absent.
    TREND_STALE_SEC   = 15.0    # Do not delay based on older snapshots.
    # Percentage thresholds are scale-invariant and configurable per window or symbol.
    PRICE_CHANGE_THRESHOLD_SMALL = u.calculate_difference_percent(60000, 60000 - 310)
    PRICE_CHANGE_THRESHOLD_BIG   = u.calculate_difference_percent(97000, 95000 - 377)
    FULL_EVAL_INTERVAL_SEC = 3.0   # Cadence for expensive complete metrics.
    FLUSH_INTERVAL_SEC     = 0.5   # Writer-only file flush cadence.
    # Every window is a slice of the same 24-hour Cache24 buffer. The smallest is primary.
    WINDOW_SECONDS = [round(3.7 * 60, 6), round(2.5 * 60 * 60, 6)]   # [3.7 min momentum, 2.5 hours trend]
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
        self.writer = _parse_bool_flag(writer, default=False)   # Only the writer process persists the file.
        # Sort N window durations so index zero is primary.
        if isinstance(window_seconds, (int, float, str)) and not isinstance(window_seconds, bool):
            secs = [window_seconds]
        else:
            secs = list(window_seconds) if window_seconds else list(self.WINDOW_SECONDS)
        valid_secs = []
        for s in secs:
            sf = safe_pos_float(s, default=None)
            if sf is not None:
                valid_secs.append(round(sf, 6))
        self.window_seconds = sorted(valid_secs) if valid_secs else sorted(round(float(s), 6) for s in self.WINDOW_SECONDS)
        # Thresholds may be callable, keyed by window, keyed by symbol and window, or None.
        self._threshold_fn = self._build_threshold_fn(thresholds)
        self._live_windows = {}
        self._hist_windows = {}
        self._trend_buffers = {}
        self._active_dynamic_windows = set()
        self._disk_c24_cache = {}
        self._mem_dirty = False
        self._dirty_live_symbols = set()
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

        effective_sync_ts = safe_pos_float(sync_ts, default=self.FULL_EVAL_INTERVAL_SEC)
        effective_api = api_client if api_client is not None else _cm_attr("api", None)
        effective_read_only = (
            (not self.writer)
            if read_only is None
            else _parse_bool_flag(read_only, default=False)
        )
        if isinstance(symbols, str):
            norm_symbols = [symbols.strip()] if symbols.strip() else []
        elif symbols is not None:
            norm_symbols = list(symbols)
        else:
            norm_symbols = []
        super().__init__(
            effective_sync_ts,
            norm_symbols,
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
                raw_meta = read_json_dict(meta_path)
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

    @staticmethod
    def _snap_ts_ms(snap: dict) -> Optional[int]:
        if not isinstance(snap, dict):
            return None
        raw_ts = snap.get("ts")
        ts_val = (
            raw_ts
            if not isinstance(raw_ts, bool) and raw_ts
            else snap.get("ts_fast")
        )
        ts_f = safe_pos_float(ts_val, default=None)
        return int(ts_f * 1000) if ts_f is not None else None

    def rebuild_fetchtimes(self):
        last_times = {}
        with self.lock:
            source = self._mem if self._mem else self._read_file()
        for symbol, snap in (source or {}).items():
            ts_ms = self._snap_ts_ms(snap)
            if ts_ms is not None:
                last_times[symbol] = ts_ms
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

    def _trend_symbol_file(self, prefix: str, symbol: str) -> str:
        safe_sym = str(symbol or "").strip().replace("/", "").replace("\\", "")
        return os.path.join(self.trend_dir, f"{prefix}_{safe_sym}.json")

    def _live_trend_file(self, symbol: str) -> str:
        return self._trend_symbol_file("live", symbol)

    def _hist_trend_file(self, symbol: str) -> str:
        return self._trend_symbol_file("hist", symbol)

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
                if round(float(sec), 6) <= small
                else self.PRICE_CHANGE_THRESHOLD_BIG
            )

        if isinstance(thresholds, dict):
            # Accept either per-symbol/per-window or direct per-window dictionaries.
            def _parse_window_map(mapping):
                out = {}
                for k, v in (mapping or {}).items():
                    kf = safe_pos_float(k, default=None)
                    vf = safe_float(v, default=None)
                    if kf is not None and vf is not None:
                        out[round(kf, 6)] = vf
                return out

            per_symbol = (
                all(isinstance(v, dict) for v in thresholds.values())
                and len(thresholds) > 0
            )
            if per_symbol:
                tbl = {
                    sym: _parse_window_map(d)
                    for sym, d in thresholds.items()
                }
                return lambda sym, sec: tbl.get(sym, {}).get(
                    round(float(sec), 6), per_window_default(sec)
                )
            tbl = _parse_window_map(thresholds)
            return lambda sym, sec: tbl.get(round(float(sec), 6), per_window_default(sec))

        return lambda sym, sec: per_window_default(sec)

    def threshold_for(self, symbol, seconds):
        """Return the percentage threshold for one symbol window."""
        sf = safe_pos_float(seconds, default=None)
        sec_val = round(sf, 6) if sf is not None else self.window_seconds[0]
        return self._threshold_fn(symbol, sec_val)

    # ── Writer: build windows and subscribe to Cache24 ───────────────────────
    def start_computation(
        self, cache24_managers=None, current_price_mgr=None, run_full_eval=False
    ):
        """Build windows and the fast channel, optionally starting full evaluation."""
        run_full_eval = _parse_bool_flag(run_full_eval, default=False)
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

        # Keep PriceWindow sampling rate synchronized with sustained live cadence.
        try:
            with getattr(self.current_price_mgr, "lock", threading.RLock()):
                dq = (getattr(self.current_price_mgr, "_update_timestamps", None) or {}).get(symbol)
                samples = list(dq) if dq else []
            if len(samples) >= 2 and (samples[-1] - samples[0]) >= 5.0:
                actual_rate = safe_pos_float(self.current_price_mgr.get_sample_rate(symbol, fallback=None), default=None)
                if actual_rate is not None and actual_rate >= 0.1:
                    for w in wins.values():
                        w.set_sample_rate(actual_rate)
        except Exception:
            pass

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
                prev_tick_ts = (
                    live_sym[wk].get("_last_tick_ts_ms")
                    if isinstance(live_sym.get(wk), dict)
                    else None
                )
                entry = dict(wval)
                entry.setdefault("current_price", current_price)
                if prev_tick_ts is not None:
                    entry["_last_tick_ts_ms"] = prev_tick_ts
                live_sym[wk] = entry
            self._dirty_live_symbols.add(symbol)

        w_big = wins.get(self.window_seconds[-1])
        v1h = None
        if w_big is not None:
            try:
                from intelligence.internal.state.volatility import (
                    calculate_volatility_1h,
                )
                with getattr(w_big, "_lock", threading.RLock()):
                    v1h = calculate_volatility_1h(
                        w_big.prices, w_big.sample_rate_sec
                    )
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
            pos=primary_pos,
            epsilon=eps_primary,
            n_samples=len(pwin.prices),
            window_seconds=float(primary),
            sample_rate_sec=float(pwin.sample_rate_sec),
            vol_1h_pct=v1h,
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
        with self._lock:
            active_dyn = sorted(self._active_dynamic_windows)
        raw_targets = (
            list(self.window_seconds)
            + list(published_cfg or ())
            + active_dyn
        )
        valid_targets = [
            round(wf, 6)
            for w in raw_targets
            if (wf := safe_pos_float(w, default=None)) is not None
        ]
        target_windows = list(dict.fromkeys(valid_targets))
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
        ts_int = safe_pos_int(ts_ms, default=None)
        price_f = safe_pos_float(price, default=None)
        if ts_int is None or price_f is None:
            return
        with self._lock:
            buf = self._get_trend_buffer(symbol)
            if not buf.timestamps:
                buf.append(price_f, ts_int)
            elif ts_int >= buf.timestamps[-1] and not (
                ts_int == buf.timestamps[-1] and price_f == buf.prices[-1]
            ):
                buf.append(price_f, ts_int)
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
                current_price=price_f,
                ts_fast=float(ts_int) / 1000.0,
            )
        except Exception as e:
            print(f"[{self.cls_name}] on_price_update {symbol}: {e}")

    # ── Calculation API ──────────────────────────────────────────────────────
    def _lookup_by_symbol_and_window(self, store: dict, symbol, seconds=None):
        mapping = store.get(symbol)
        if not mapping:
            for cand in _short_trend_symbol_variants(symbol):
                mapping = store.get(cand)
                if mapping:
                    break
        mapping = mapping or {}
        if seconds is None:
            return mapping.get(self.window_seconds[0])
        sf = safe_float(seconds, default=None)
        return mapping.get(round(sf, 6)) if sf is not None else None

    def get_window(self, symbol, seconds=None):
        """Return the requested window, or the smallest primary window for None."""
        return self._lookup_by_symbol_and_window(self.windows, symbol, seconds)

    def get_analyzer(self, symbol, seconds=None):
        """Return the requested analyzer, or the smallest primary analyzer for None."""
        return self._lookup_by_symbol_and_window(self.analyzers, symbol, seconds)

    def get_short_trend(self, symbol):
        win = self.get_window(symbol)
        return win.get_short_trend() if win else None

    # ── Dynamic window & Historical Subslice (O(1) Prefix Sums + Persistence) ──
    def _eval_entries_with_buffer(
        self, symbol, entries, a: int, b: int, window_seconds=None
    ):
        """Evaluate slice [a, b] of entries using the symbol's IncrementalTrendBuffer."""
        import pricewindow as pw
        sub_entries = entries[a : b + 1]
        sample_rate = pw.PriceWindow._sample_rate_from_entries(sub_entries)
        with self._lock:
            buf = self._get_trend_buffer(symbol)
            buf.sync_entries(entries)
            if len(buf.prices) == len(entries):
                ft, gc, sf, gr, eps = buf.eval_slice(
                    a,
                    b,
                    sample_rate_sec=sample_rate,
                    eps_k=self.EPSILON_K,
                    window_seconds=window_seconds,
                    symbol=symbol,
                )
                return ft, gc, sf, gr, eps, sample_rate
        # Fallback if entries were non-monotonic or ad-hoc:
        tmp_buf = pw.IncrementalTrendBuffer()
        tmp_buf.sync_entries(sub_entries)
        ft, gc, sf, gr, eps = tmp_buf.eval_slice(
            0,
            len(sub_entries) - 1,
            sample_rate_sec=sample_rate,
            eps_k=self.EPSILON_K,
            window_seconds=window_seconds,
            symbol=symbol,
        )
        return ft, gc, sf, gr, eps, sample_rate

    def _load_disk_cache24_entries(self, symbol):
        """Load raw [ts_ms, price] entries from cachedb/cache_24_price_<symbol>.json if present."""
        base_dir = os.path.dirname(self.filename)
        if os.path.basename(base_dir) in CACHEDB_SUBDOMAINS:
            base_dir = os.path.dirname(base_dir)
        c24_file = None
        matched_sym = symbol
        for cand in _short_trend_symbol_variants(symbol):
            if not isinstance(cand, str) or not cand:
                continue
            cand_file = str(resolve_cachedb_filepath(os.path.join(base_dir, f"cache_24_price_{cand}.json")))
            _migrate_legacy_filename_if_needed(cand_file)
            if os.path.exists(cand_file) or os.path.exists(
                previous_generation_path(cand_file)
            ):
                c24_file = cand_file
                matched_sym = cand
                break
        if c24_file is None:
            return []
        try:
            backup_path = previous_generation_path(c24_file)
            if os.path.exists(c24_file) and not os.path.exists(backup_path):
                mtime = os.stat(c24_file).st_mtime_ns
                with self._lock:
                    cached = self._disk_c24_cache.get(matched_sym)
                    if cached and cached[0] == mtime:
                        return cached[1]

            def _read_c24(candidate_path):
                raw = read_json_dict(candidate_path)
                items_map = raw.get("items")
                if not isinstance(items_map, dict):
                    raise ValueError("Cache24 items must be a dict")
                items = items_map.get(matched_sym)
                if not isinstance(items, list):
                    for cand in _short_trend_symbol_variants(symbol):
                        if isinstance(items_map.get(cand), list):
                            items = items_map[cand]
                            break
                if not isinstance(items, list):
                    return []
                valid_entries = []
                for entry in items:
                    if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                        continue
                    ts_val = safe_pos_int(entry[0], default=None)
                    price_val = safe_pos_float(entry[1], default=None)
                    if ts_val is not None and price_val is not None:
                        valid_entries.append([ts_val, price_val])
                return valid_entries

            entries = read_recovering_previous(
                c24_file, _read_c24, backup_path=backup_path
            )
            if os.path.exists(c24_file):
                mtime = os.stat(c24_file).st_mtime_ns
                with self._lock:
                    self._disk_c24_cache[matched_sym] = (mtime, entries)
            return entries
        except Exception:
            return []

    def _eval_and_persist_window_slice(
        self,
        target_sym: str,
        entries: list,
        a: int,
        b: int,
        clamped: float,
        w_key: str,
        now_val: float,
        *,
        check_memo: bool = False,
    ) -> Optional[dict]:
        n_samples = b - a + 1
        if n_samples < 3:
            return None
        last_entry = entries[b]
        if not isinstance(last_entry, (list, tuple)) or len(last_entry) < 2:
            return None
        ts_f = safe_pos_float(last_entry[0], default=None)
        latest_price = safe_pos_float(last_entry[1], default=None)
        if ts_f is None or latest_price is None:
            return None
        newest_ts_ms = int(ts_f)
        newest_ts_sec = newest_ts_ms / 1000.0
        age_sec = now_val - newest_ts_sec
        if not (-5.0 <= age_sec <= self.TREND_STALE_SEC):
            return None
        if check_memo:
            with self._lock:
                cached = (self._live_windows.get(target_sym) or {}).get(w_key)
                if (
                    isinstance(cached, dict)
                    and cached.get("_last_tick_ts_ms") == newest_ts_ms
                    and cached.get("n_samples") == n_samples
                ):
                    hit = {k: v for k, v in cached.items() if not k.startswith("_")}
                    hit["ts"] = now_val
                    return hit
        ft, gc, sf, gr, eps, sample_rate = self._eval_entries_with_buffer(
            target_sym, entries, a, b, window_seconds=clamped
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
            current_price=latest_price,
            ts=now_val,
        )
        self._persist_live_window(
            target_sym, w_key, res, last_tick_ts_ms=newest_ts_ms
        )
        return res

    def get_short_trend_for_window(self, symbol, window_seconds, now=None):
        """Calculate trend on demand for an arbitrary window in seconds.

        Uses O(1) incremental prefix sums over Cache24 and memoizes results in
        per-symbol persistence (cachedb/24trend/live_<SYMBOL>.json) so repeated
        requests for the same tick state are served directly from persistence.
        """
        req = safe_pos_float(window_seconds, default=None)
        if req is None:
            return None
        if now is not None:
            now = safe_pos_float(now, default=None)
            if now is None:
                return None
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
            return self._eval_and_persist_window_slice(
                symbol, disk_entries, a, b, clamped, w_key, now_val, check_memo=False
            )

        c24 = None
        c24_sym = symbol
        for cand in _short_trend_symbol_variants(symbol):
            c24 = self._cache24_managers.get(cand)
            if c24 is not None:
                c24_sym = cand
                break
        if c24 is None:
            return None

        now_val = now if now is not None else time.time()
        with self._lock:
            self._active_dynamic_windows.add(clamped)

        # Fast path when c24 is a real Cache24PriceManager instance with direct buffer access:
        if hasattr(c24, "cache") and hasattr(c24, "lock"):
            with c24.lock:
                all_entries = list(c24.cache.get(c24_sym, []))
            if len(all_entries) < 3:
                return None
            cutoff_ms = int((now_val - clamped) * 1000)
            a = bisect.bisect_left(all_entries, cutoff_ms, key=lambda e: e[0])
            b = len(all_entries) - 1
            return self._eval_and_persist_window_slice(
                c24_sym, all_entries, a, b, clamped, w_key, now_val, check_memo=True
            )

        # Compatibility path for custom/mock c24 objects implementing only get_recent_entries:
        try:
            entries = c24.get_recent_entries(c24_sym, last_seconds=clamped)
        except Exception as e:
            print(
                f"[get_short_trend_for_window] {symbol}: get_recent_entries failed ({e})"
            )
            return None
        if len(entries) < 3:
            return None
        return self._eval_and_persist_window_slice(
            symbol, entries, 0, len(entries) - 1, clamped, w_key, now_val, check_memo=False
        )

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
            self._dirty_live_symbols.add(symbol)
            if flush and self.writer:
                self._write_live_symbol_file(symbol)
                self._dirty_live_symbols.discard(symbol)

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

    @staticmethod
    def _read_symbol_trend_dict(path: str, label: str) -> dict:
        backup_path = previous_generation_path(path)
        if not os.path.exists(path) and not os.path.exists(backup_path):
            return {}
        return read_json_dict(path, default={}, backup_path=backup_path, recover_previous=True)

    def _read_live_symbol_file(self, symbol: str) -> dict:
        return self._read_symbol_trend_dict(self._live_trend_file(symbol), "live")

    def _read_hist_symbol_file(self, symbol: str) -> dict:
        return self._read_symbol_trend_dict(self._hist_trend_file(symbol), "hist")

    def _published_window(self, symbol, window_seconds, now=None):
        """Return a dynamic window published in 24trend/live_<SYMBOL>.json or shared file."""
        req = safe_pos_float(window_seconds, default=None)
        if req is None:
            return None
        if now is not None:
            now = safe_pos_float(now, default=None)
            if now is None:
                return None
        min_sec = _cm_attr("CM_DYNAMIC_WINDOW_MIN_SEC", CM_DYNAMIC_WINDOW_MIN_SEC)
        max_sec = _cm_attr("CM_DYNAMIC_WINDOW_MAX_SEC", CM_DYNAMIC_WINDOW_MAX_SEC)
        clamped = min(max(req, min_sec), max_sec)
        w_key = _dynamic_window_key(clamped)
        now = now if now is not None else time.time()

        def _check_fresh(dyn):
            if not isinstance(dyn, dict):
                return None
            ts_val = safe_pos_float(dyn.get("ts"), default=None)
            if ts_val is None:
                return None
            age = float(now) - ts_val
            if not (-5.0 <= age <= self.TREND_STALE_SEC):
                return None
            if "current_price" in dyn and dyn.get("current_price") is not None:
                if safe_pos_float(dyn.get("current_price"), default=None) is None:
                    return None
            return {k: v for k, v in dyn.items() if not k.startswith("_")}

        variants = _short_trend_symbol_variants(symbol)

        # 1. Check in-memory _live_windows and _mem['dynamic'] first:
        with self._lock:
            for cand in variants:
                mem_live = (self._live_windows.get(cand) or {}).get(w_key)
                if isinstance(mem_live, dict):
                    fresh = _check_fresh(mem_live)
                    if fresh is not None:
                        return fresh
            for cand in variants:
                mem_snap = self._mem.get(cand)
                if isinstance(mem_snap, dict):
                    mem_pub = mem_snap.get("dynamic")
                    if isinstance(mem_pub, dict) and w_key in mem_pub:
                        fresh = _check_fresh(mem_pub.get(w_key))
                        if fresh is not None:
                            return fresh

        # 2. Check shared snapshot file dynamic dict (authoritative if present):
        file_data = self._read_file()
        for cand in variants:
            snap = file_data.get(cand)
            if isinstance(snap, dict):
                published = snap.get("dynamic")
                if isinstance(published, dict) and w_key in published:
                    fresh = _check_fresh(published.get(w_key))
                    if fresh is not None:
                        return fresh
                if (
                    math.isclose(clamped, self.window_seconds[0], rel_tol=0.0, abs_tol=1e-6)
                    and "gradient_recent" in snap
                    and "ts" in snap
                ):
                    fresh = _check_fresh(snap)
                    if fresh is not None:
                        return fresh

        # 3. Check per-symbol 24trend/live_<SYMBOL>.json:
        for cand in variants:
            if not isinstance(cand, str) or not cand:
                continue
            file_live = self._read_live_symbol_file(cand).get(w_key)
            if isinstance(file_live, dict):
                fresh = _check_fresh(file_live)
                if fresh is not None:
                    return fresh

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

        variants = _short_trend_symbol_variants(symbol)
        with self._lock:
            for cand in variants:
                sym_hist = self._hist_windows.setdefault(cand, {})
                if isinstance(sym_hist.get(slice_key), dict):
                    return dict(sym_hist[slice_key])
                loaded = self._read_hist_symbol_file(cand)
                if isinstance(loaded, dict):
                    for k, v in loaded.items():
                        if isinstance(v, dict):
                            sym_hist[k] = v
                    if isinstance(sym_hist.get(slice_key), dict):
                        return dict(sym_hist[slice_key])

        # Cache miss: resolve raw entries from memory Cache24 or disk Cache24
        entries = []
        matched_sym = symbol
        if self._cache24_managers is not None:
            for cand in variants:
                c24 = self._cache24_managers.get(cand)
                if c24 is not None:
                    matched_sym = cand
                    if hasattr(c24, "cache") and hasattr(c24, "lock"):
                        with c24.lock:
                            entries = list(c24.cache.get(cand, []))
                    elif hasattr(c24, "entries"):
                        entries = list(c24.entries)
                    if entries:
                        break
        if not entries:
            entries = self._load_disk_cache24_entries(symbol)
        if len(entries) < 3:
            return None

        a = bisect.bisect_left(entries, t1_ms, key=lambda e: e[0])
        b = bisect.bisect_right(entries, t2_ms, key=lambda e: e[0]) - 1
        if a < 0 or b >= len(entries) or (b - a + 1) < 3:
            return None

        slice_win_sec = (t2_ms - t1_ms) / 1000.0
        ft, gc, sf, gr, eps, sample_rate = self._eval_entries_with_buffer(
            matched_sym, entries, a, b, window_seconds=slice_win_sec
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
            hist_path = self._hist_trend_file(symbol)
            loaded = self._read_hist_symbol_file(symbol)
            if isinstance(loaded, dict):
                for k, v in loaded.items():
                    if isinstance(v, dict) and k not in sym_hist:
                        sym_hist[k] = v
            sym_hist[slice_key] = res
            try:
                _write_json_atomic(hist_path, sym_hist)
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
        """Single canonical upward-trend formula (SSOT) aligned with should_wait (GT_S):
        growth_coefficient > UPWARD_EPSILON_K * epsilon (when epsilon > 0).
        """
        if not isinstance(data, dict):
            return False
        for key in (
            "growth_coefficient",
            "gradient_recent",
            "slope_small",
            "final_trend",
            "epsilon",
        ):
            val = data.get(key)
            if val is not None and safe_float(val, default=None) is None:
                return False

        eps = safe_pos_float(data.get("epsilon"), default=0.0)
        scale = 1.0
        n_f = safe_float(data.get("n_samples"), default=None)
        if n_f is not None and (n_f - 1.0) > 10.0:
            scale = math.sqrt(10.0 / (n_f - 1.0))
        noise_floor = eps * CacheShortTrendManager.UPWARD_EPSILON_K * scale

        raw_gc = data.get("growth_coefficient")
        if raw_gc is not None:
            gc = float(raw_gc)
            if gc != 0.0:
                return gc > noise_floor
            raw_ft = data.get("final_trend", 0.0)
            ft = float(0.0 if raw_ft is None else raw_ft)
            return eps == 0.0 and ft > 0

        raw_slope = data.get("gradient_recent")
        if raw_slope is None:
            raw_slope = data.get("slope_small", 0.0)
        raw_gradient = data.get("final_trend", 0.0)
        slope = float(0.0 if raw_slope is None else raw_slope)
        gradient = float(0.0 if raw_gradient is None else raw_gradient)
        if slope != 0.0:
            return slope > noise_floor
        return eps == 0.0 and gradient > 0

    def is_trend_up(self, symbol, window_seconds=None, now=None) -> bool:
        """Evaluate upward-trend condition over a dynamic window or the default primary snapshot."""
        if window_seconds is not None:
            win_sec = safe_pos_float(window_seconds, default=None)
            if win_sec is None:
                return False
            if not math.isclose(win_sec, self.window_seconds[0], rel_tol=0.0, abs_tol=1e-6):
                dyn = self.get_short_trend_for_window(
                    symbol, win_sec, now=now
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
                    ts_ms = self._snap_ts_ms(val)
                    if ts_ms is not None:
                        fetchtime[sym] = ts_ms
                    elif sym in self.fetchtime_per_symbol:
                        raw_ft = safe_nonneg_float(self.fetchtime_per_symbol[sym], default=None)
                        fetchtime[sym] = int(raw_ft) if raw_ft is not None else 0
                self.fetchtime_per_symbol = dict(fetchtime)
                self._rest_sync_cursor_per_symbol = dict(fetchtime)
                counts = {sym: 1 for sym in snapshot_mem}
                metadata = {
                    "counts": counts,
                    "fetchtime": fetchtime,
                    "rest_sync_cursor": dict(fetchtime),
                }
                write_fn = _cm_attr("atomic_write_json", atomic_write_json)
                if not self._commit_file_and_meta_locked(
                    lambda: write_fn(self.filename, snapshot_mem),
                    metadata,
                ):
                    return False
                self._mem_dirty = False
                for symbol in list(self._live_windows.keys()):
                    if symbol in self._dirty_live_symbols or not os.path.exists(
                        self._live_trend_file(symbol)
                    ):
                        self._write_live_symbol_file(symbol)
                        self._dirty_live_symbols.discard(symbol)
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
                data = read_json_dict(candidate_path)
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
            self._mem_dirty = True
            ts_ms = self._snap_ts_ms(snap)
            if ts_ms is not None:
                self.fetchtime_per_symbol[symbol] = ts_ms
                self._rest_sync_cursor_per_symbol[symbol] = ts_ms

    def update_snapshot(self, symbol, **fields):
        """Update memory and flush immediately only when this instance is the writer."""
        with self._lock:
            prev_snap = (
                dict(self._mem[symbol])
                if isinstance(self._mem.get(symbol), dict)
                else None
            )
            self._set_mem(symbol, **fields)
            if self.writer:
                if not self._write_file():
                    if prev_snap is not None:
                        self._mem[symbol] = prev_snap
                    else:
                        self._mem.pop(symbol, None)
                    self._mem_dirty = False
                    return False
        return True

    def _start_flush_loop(self):
        """Run a writer-only thread that decouples periodic file I/O from calculation."""
        if not self.writer:
            return
        if self._flush_thread is not None and self._flush_thread.is_alive():
            return

        def run():
            while not self._stop_event.wait(self.FLUSH_INTERVAL_SEC):
                with self._lock:
                    if self._mem and (self._mem_dirty or self._dirty_live_symbols):
                        self._write_file()

        self._flush_thread = threading.Thread(
            target=run, name="ShortTrendFlush", daemon=True
        )
        self._flush_thread.start()

    def shutdown(self, timeout=5.0):
        """Stop evaluation, flush, and base sync loops, then unsubscribe Cache24 sources."""
        base_stopped = super().shutdown(timeout=timeout)
        for sym_key, cache24 in (self._cache24_managers or {}).items():
            unsubscribe = getattr(cache24, "unsubscribe_price", None)
            if unsubscribe is not None:
                unsubscribe(self)
                for win in (self.windows.get(sym_key) or {}).values():
                    unsubscribe(win)
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
        overwrite = _parse_bool_flag(overwrite, default=True)
        data = self._read_file()
        if symbols is None:
            allowed = None
        elif isinstance(symbols, str):
            allowed = {symbols.strip()} if symbols.strip() else set()
        else:
            allowed = {str(symbol) for symbol in symbols}
        with self._lock:
            for symbol, snap in data.items():
                if (
                    isinstance(snap, dict)
                    and (allowed is None or symbol in allowed)
                    and (overwrite or symbol not in self._mem)
                ):
                    self._mem[symbol] = dict(snap)
                    ts_ms = self._snap_ts_ms(snap)
                    if ts_ms is not None:
                        self.fetchtime_per_symbol[symbol] = ts_ms
                        self._rest_sync_cursor_per_symbol[symbol] = ts_ms
            self.cache = self._mem
            self._account_cache_dirty = False
        return len(self._mem)

    def _reload_from_disk(self):
        """Reload flat symbol snapshots and synchronize fetchtime cursors."""
        self.prime_from_file(overwrite=True)

    def get_snapshot(self, symbol):
        """Return local calculated/primed memory, or the file for a pure reader."""
        variants = _short_trend_symbol_variants(symbol)
        with self._lock:
            for cand in variants:
                val = self._mem.get(cand)
                if isinstance(val, dict):
                    return dict(val)
        file_data = self._read_file()
        for cand in variants:
            val = file_data.get(cand)
            if isinstance(val, dict):
                return dict(val)
        return None

    @staticmethod
    def _snapshot_is_usable(snap, now, max_age_sec):
        """Validate the common full-snapshot price and observation timestamp."""
        if not isinstance(snap, dict):
            return False
        now_f = safe_pos_float(now, default=None)
        max_age_f = safe_pos_float(max_age_sec, default=None)
        observed_at = safe_pos_float(snap.get("ts"), default=None)
        price = safe_pos_float(snap.get("current_price"), default=None)
        if now_f is None or max_age_f is None or observed_at is None or price is None:
            return False
        age = now_f - observed_at
        return -5.0 <= age <= max_age_f

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
                return {
                    s: dict(v) for s, v in self._mem.items() if isinstance(v, dict)
                }
        return {
            s: dict(v) for s, v in self._read_file().items() if isinstance(v, dict)
        }

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
            self._mem_dirty = False
            self._dirty_live_symbols.clear()
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
        eps = safe_pos_float(snap.get("epsilon"), default=None)
        if eps is not None:
            return eps
        price = abs(safe_float(snap.get("current_price"), default=0.0))
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

    def _should_wait_on_snap(
        self,
        snap,
        side: str,
        *,
        fast: bool = False,
        use_noise_gate: bool = True,
        allow_zero_eps: bool = False,
    ) -> bool:
        if not isinstance(snap, dict):
            return False
        if fast:
            raw_g = snap.get("gradient_recent_fast")
            if raw_g is None:
                raw_g = snap.get("gradient_recent")
            if raw_g is None:
                raw_g = snap.get("growth_coefficient", 0.0)
        else:
            raw_g = snap.get("growth_coefficient")
            if raw_g is None:
                raw_g = snap.get("gradient_recent", 0.0)
        g = safe_float(raw_g if raw_g is not None else 0.0, default=None)
        if g is None:
            return False
        if use_noise_gate:
            raw_eps = snap.get("epsilon") if allow_zero_eps else None
            eps_zero = safe_nonneg_float(raw_eps, default=None) if allow_zero_eps else None
            eps = eps_zero if eps_zero is not None else self._epsilon(snap)
            if abs(g) <= eps:
                return True
        if side == "BUY":
            return g < 0
        if side == "SELL":
            return g > 0
        return False

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
        side = normalize_side(side)
        if side not in ("BUY", "SELL"):
            return False
        fast = _parse_bool_flag(fast, default=False)
        use_noise_gate = _parse_bool_flag(use_noise_gate, default=True)

        if window_seconds is not None:
            win_sec = safe_pos_float(window_seconds, default=None)
            if win_sec is None:
                return False
            if not math.isclose(win_sec, self.window_seconds[0], rel_tol=0.0, abs_tol=1e-6):
                dyn = self.get_short_trend_for_window(
                    symbol, win_sec, now=now
                )
                return self._should_wait_on_snap(
                    dyn,
                    side,
                    fast=False,
                    use_noise_gate=use_noise_gate,
                    allow_zero_eps=True,
                )

        snap = self.get_fresh_snapshot(symbol, now=now)
        if snap is None and window_seconds is not None:
            snap = self.get_short_trend_for_window(
                symbol, window_seconds, now=now
            )
        return self._should_wait_on_snap(
            snap,
            side,
            fast=fast,
            use_noise_gate=use_noise_gate,
            allow_zero_eps=False,
        )

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
        max_wait_f = safe_pos_float(max_wait_sec, default=None)
        poll_f = safe_pos_float(poll_sec, default=None)
        if max_wait_f is None or poll_f is None:
            return 0.0
        deadline = time.time() + max_wait_f
        waited = 0.0
        next_dot = 1.0
        while time.time() < deadline and self.should_wait(
            side,
            symbol,
            window_seconds=window_seconds,
            fast=(str(mode or "").strip().lower() == "gradient"),
        ):
            sleep_fn(poll_f)
            waited += poll_f
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
            or normalize_venue_name(bindings[symbol.upper()], default="") == "binance"
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
                    pf = safe_pos_float(p, default=None)
                    if pf is not None:
                        ws_ts_map = getattr(cpm, "_ws_last_event_ts_per_symbol", None)
                        if isinstance(ws_ts_map, dict):
                            ws_ts_map[s] = time.time()
                        cpm._push_price(s, pf)
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


get_short_trend_manager = _cm_attr("get_short_trend_manager")

if _cm is not None:
    _cm.CacheLongTrendManager = CacheLongTrendManager
    _cm.CacheLongTrend = CacheLongTrend
    _cm.CacheShortTrendManager = CacheShortTrendManager
    _cm.CacheShortTrend = CacheShortTrend
    _cm.SHORT_TREND_EVAL_INTERVAL_SEC = CacheShortTrendManager.FULL_EVAL_INTERVAL_SEC
    _cm._NB_FETCH_DEADLINE_SEC = _NB_FETCH_DEADLINE_SEC
    _cm._fetch_price_with_deadline = _fetch_price_with_deadline
    _cm._NonBinanceTrendPoller = _NonBinanceTrendPoller
    _cm._start_nonbinance_trend_poller = _start_nonbinance_trend_poller
    if hasattr(_cm, "CacheFactory") and "LongTrend" in _cm.CacheFactory._CONFIG:
        _cm.CacheFactory._CONFIG["LongTrend"]["class"] = CacheLongTrendManager
