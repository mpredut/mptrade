"""Trend survival models, duration estimation, and empirical distribution metrics.

Estimates trend longevity distributions (Weibull / Kaplan-Meier) and calibrated Gaussian horizon T:
- fetch_klines: Paginated candlestick historical fetch.
- block_slopes, episodes: Historical trend duration episode segmentation.
- hybrid_T: Combines empirical P90 / median durations with prior T.
- estimate_T: Caches and returns per-symbol estimated duration metrics.
- get_trend_survival_metrics: Fast lookup/estimation for guards and composite state.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import math
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from botcore import (
    _parse_bool_flag,
    gaussian_weights_from_idx,
    is_live_network_backfill_enabled,
    lru_dict_put,
    normalize_epoch_sec,
    normalize_trend_direction,
    obj_attr,
    resolve_now_ts,
    resolve_patched_attr,
    safe_float,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
    safe_pos_int,
)
from cachepaths import (
    CACHE_LONG_TREND_FILENAME,
    CACHE_T_TREND_FILENAME,
    CACHE_WATCHER_ALL_PRICE_FILENAME,
    cache_24_price_filename,
    cache_path,
    cache_symbol_variants,
    canonical_symbol_key,
    get_cachedb_dir,
    has_equity_suffix,
    normalize_cache_base_asset,
    resolve_cachedb_filepath,
)
from instrument_registry import is_equity_symbol
from intelligence.macro_sentiment_ssot import lookup_symbol_cache_items, scan_latest_long_trend_value
from providers.base import normalize_side
from state_io import read_json_dict_cached, save_json_state

_DEFAULT_T_CACHE_FILE = cache_path(CACHE_T_TREND_FILENAME)
T_CACHE_FILE = _DEFAULT_T_CACHE_FILE
_ON_DEMAND_T_MISS_TS: dict[str, float] = {}
_ON_DEMAND_T_MISS_TTL_SEC = 600.0


def _resolve_facade_attr(name: str, current: Any, default: Any) -> Any:
    return resolve_patched_attr(
        name,
        default,
        local_val=current,
        modules=("intelligence.internal.state.survival", "forecast.trend_survival"),
    )


def _effective_t_cache_file() -> str:
    cand = _resolve_facade_attr("T_CACHE_FILE", T_CACHE_FILE, _DEFAULT_T_CACHE_FILE)
    if isinstance(cand, str) and cand.strip() and cand != _DEFAULT_T_CACHE_FILE:
        return cand
    try:
        return cache_path(CACHE_T_TREND_FILENAME)
    except Exception:
        return T_CACHE_FILE


def fetch_klines(symbol: str, days: int, interval: str = "1h") -> tuple[np.ndarray, np.ndarray]:
    """Return historical ``(timestamps_sec, closes)`` OHLC arrays via the canonical ``MarketApi`` provider facade."""
    from providers.market_api import api as _market_api

    pts = _market_api.fetch_historical_ohlc_points(symbol, days, interval=interval)
    if not pts:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    return (
        np.asarray([t for t, _ in pts], dtype=float),
        np.asarray([p for _, p in pts], dtype=float),
    )


def _coerce_xy_pairs(xs: Any, ys: Any, parser_x: Any, parser_y: Any) -> tuple[list[float], list[float]]:
    if xs is None or ys is None or isinstance(xs, (bool, str, bytes)) or isinstance(ys, (bool, str, bytes)):
        return [], []
    try:
        raw_x, raw_y = list(xs), list(ys)
    except TypeError:
        return [], []
    n_common = min(len(raw_x), len(raw_y))
    clean_x, clean_y = [], []
    for xv, yv in zip(raw_x[:n_common], raw_y[:n_common]):
        xf, yf = parser_x(xv), parser_y(yv)
        if xf is not None and yf is not None:
            clean_x.append(xf)
            clean_y.append(yf)
    return clean_x, clean_y


def block_slopes(ts, px, window_h, step_h):
    """Return slope signs for each [t-window, t] window stepped by step_h."""
    out_t, out_s = [], []
    clean_ts, clean_px = _coerce_xy_pairs(
        ts,
        px,
        lambda v: normalize_epoch_sec(v, default=None),
        lambda v: safe_pos_float(v, default=None),
    )
    if len(clean_ts) < 5:
        return np.array(out_t), np.array(out_s)
    ts = np.asarray(clean_ts, dtype=float)
    px = np.asarray(clean_px, dtype=float)
    if np.any(np.diff(ts) < 0):
        order = np.argsort(ts, kind="mergesort")
        ts, px = ts[order], px[order]
    w_h, s_h = safe_pos_float(window_h, default=0.0), safe_pos_float(step_h, default=0.0)
    if w_h <= 0.0 or s_h <= 0.0:
        return np.array(out_t), np.array(out_s)
    w, s = w_h * 3600.0, s_h * 3600.0
    t, t_end = float(ts[0]) + w, float(ts[-1])
    while t <= t_end:
        lo, hi = int(np.searchsorted(ts, t - w, "left")), int(np.searchsorted(ts, t, "right"))
        if hi - lo >= 5:
            dx = ts[lo:hi] - ts[lo]
            dev_x = dx - float(np.mean(dx))
            var_x = float(np.dot(dev_x, dev_x))
            if var_x > 0.0:
                sl = float(np.dot(dev_x, px[lo:hi]) / var_x)
                out_t.append(t)
                out_s.append(1.0 if sl > 0 else (-1.0 if sl < 0 else 0.0))
        t += s
    return np.array(out_t), np.array(out_s)


def episodes(bt, bs, window_h, noise_tolerance=2, min_confirm=3):
    """Return trend episodes in hours using the detector's semantics."""
    eps = []
    w_h = safe_pos_float(window_h, default=0.0)
    if w_h <= 0.0:
        return eps
    nt, mc = safe_nonneg_int(noise_tolerance, default=2), safe_pos_int(min_confirm, default=3)
    clean_bt, clean_bs = _coerce_xy_pairs(
        bt,
        bs,
        lambda v: (tf / 1000.0 if tf > 1e11 else tf) if (tf := safe_float(v, default=None)) is not None else None,
        lambda v: (1.0 if sf > 0 else (-1.0 if sf < 0 else 0.0)) if (sf := safe_float(v, default=None)) is not None else None,
    )
    n = len(clean_bt)
    if n == 0:
        return eps
    bt, bs = np.asarray(clean_bt, dtype=float), np.asarray(clean_bs, dtype=float)
    if n > 1 and np.any(np.diff(bt) < 0):
        order = np.argsort(bt, kind="mergesort")
        bt, bs = bt[order], bs[order]
    i = 0
    while i < n:
        sign = bs[i]
        if sign == 0.0:
            i += 1
            continue
        start, last_confirm = bt[i] - w_h * 3600.0, bt[i]
        confirms, noise = 1, 0
        for j in range(i + 1, n):
            if bs[j] == sign:
                confirms += 1
                noise = 0
                last_confirm = bt[j]
            elif noise < nt:
                noise += 1
            else:
                break
        if confirms >= mc:
            eps.append({"dir": "up" if sign > 0 else "down", "dur_h": (last_confirm - start) / 3600.0})
        i = max(int(np.searchsorted(bt, last_confirm, "right")), i + 1)
    return eps


def survival_report(durs_h: list[float], label: str, t_grid_days, horizon_h=24.0):
    """Calculate and format empirical survival table."""
    if (
        durs_h is None
        or t_grid_days is None
        or isinstance(durs_h, (bool, str, bytes))
        or isinstance(t_grid_days, (bool, str, bytes))
    ):
        return None
    try:
        clean_d = [
            f for x in durs_h
            if (f := safe_nonneg_float(x, default=None)) is not None
        ]
    except TypeError:
        return None
    d = np.array(clean_d, dtype=float)
    if len(d) < 15:
        return None
    hz = safe_pos_float(horizon_h, default=24.0)
    cont = {}
    try:
        grid_iter = list(t_grid_days)
    except TypeError:
        return None
    for t_days in grid_iter:
        t_days_f = safe_nonneg_float(t_days, default=None)
        if t_days_f is None:
            continue
        t_h = t_days_f * 24.0
        alive = d > t_h
        if alive.sum() < 10:
            break
        p = float((d > t_h + hz).sum() / alive.sum())
        cont[t_days] = p
    return cont


def verdict(cont: dict, mid: float) -> str:
    """Compare young (t < mid) and old (t >= mid) trend continuation."""
    if not isinstance(cont, dict):
        return "insufficient_data"
    mid_f = safe_pos_float(mid, default=0.0)
    if mid_f <= 0.0:
        return "insufficient_data"
    young: list[float] = []
    old: list[float] = []
    for t, p in cont.items():
        tf = safe_float(t, default=None)
        pf = safe_float(p, default=None)
        if tf is None or pf is None:
            continue
        if tf < mid_f:
            young.append(pf)
        else:
            old.append(pf)
    if not young or not old:
        return "insufficient_data"
    ym, om = float(np.mean(young)), float(np.mean(old))
    if om >= ym - 0.05:
        return "VALIDATED: old trend continues as long as middle (Lindy)"
    return "INVALIDATED: old trends terminate faster than young trends"


def hybrid_T(dur_hours, prior_T: float = 14.0, k: float = 30.0, t_min: int = 4, t_max: int = 30) -> dict:
    """Return hybrid T by combining empirical data with prior."""
    prior_f = safe_pos_float(prior_T, default=14.0)
    k_f = safe_pos_float(k, default=30.0)
    t_min_f = safe_pos_float(t_min, default=4.0)
    t_max_f = safe_pos_float(t_max, default=30.0)
    if t_min_f > t_max_f:
        t_min_f, t_max_f = t_max_f, t_min_f
    empty_res = {
        "T": int(round(prior_f)),
        "n": 0,
        "w": 0.0,
        "median_d": None,
        "p90_d": None,
        "T_emp": None,
    }
    if dur_hours is None or isinstance(dur_hours, (str, bytes, bool)):
        return empty_res
    if isinstance(dur_hours, np.ndarray):
        if dur_hours.dtype == np.bool_:
            d = np.empty(0, dtype=float)
        elif dur_hours.dtype.kind in ("f", "i", "u"):
            arr = dur_hours.astype(float, copy=False).ravel() / 24.0
            d = arr[np.isfinite(arr) & (arr > 0)]
        else:
            vals = [vf / 24.0 for v in dur_hours.ravel() if (vf := safe_pos_float(v, default=None)) is not None]
            d = np.asarray(vals, dtype=float)
    else:
        vals = []
        try:
            for v in dur_hours:
                vf = safe_pos_float(v, default=None)
                if vf is not None:
                    vals.append(vf / 24.0)
        except TypeError:
            pass
        d = np.asarray(vals, dtype=float)
    n = len(d)
    if n == 0:
        return empty_res
    med, p90 = float(np.median(d)), float(np.percentile(d, 90))
    t_emp = max(p90, 2.0 * med)
    w = n / (n + k_f)
    T = min(max(w * t_emp + (1 - w) * prior_f, t_min_f), t_max_f)
    return {
        "T": int(round(T)),
        "n": n,
        "w": round(w, 2),
        "median_d": round(med, 1),
        "p90_d": round(p90, 1),
        "T_emp": round(t_emp, 1),
    }


_T_CACHE_FILE_MEMO: dict = {}


def _lookup_t_cache_entry(data: Dict[str, Any], symbol: str) -> Optional[dict]:
    ent = lookup_symbol_cache_items(data, symbol, require_dict=True)
    return ent if isinstance(ent, dict) else None


def get_cached_t_entry(symbol: str) -> Optional[dict]:
    """Read and look up the cached T entry for ``symbol`` from disk in 0 ms."""
    try:
        data = read_json_dict_cached(_effective_t_cache_file(), _T_CACHE_FILE_MEMO, max_entries=16)
        if isinstance(data, dict):
            ent = _lookup_t_cache_entry(data, symbol)
            if isinstance(ent, dict):
                return ent
    except Exception:
        pass
    return None


def estimate_T(
    symbol: str,
    days: int = 540,
    window_h: int = 24,
    step_h: int = 8,
    prior_T: float = 14.0,
    ttl_days: float = 7.0,
) -> dict:
    """Estimate asset-specific trend lifetime T empirically from historical candlesticks."""
    prior_f = safe_pos_float(prior_T, default=14.0)
    ttl_f = safe_pos_float(ttl_days, default=7.0)
    fallback_res = {
        "T": int(round(prior_f)),
        "n": 0,
        "w": 0.0,
        "source_symbol": None,
        "median_d": None,
        "p90_d": None,
        "T_emp": None,
        "ts": 0,
    }
    if not isinstance(symbol, str) or not symbol.strip():
        return fallback_res
    t_cache_path = _effective_t_cache_file()
    loaded_cache = read_json_dict_cached(t_cache_path, _T_CACHE_FILE_MEMO, max_entries=16)
    cache = dict(loaded_cache) if isinstance(loaded_cache, dict) else {}

    ent = _lookup_t_cache_entry(cache, symbol)
    if ent:
        ts_val = normalize_epoch_sec(ent.get("ts"), default=None)
        if ts_val is not None:
            now_curr = time.time()
            if -60.0 <= (now_curr - ts_val) < ttl_f * 86400.0:
                return ent

    sym_u = symbol.upper().strip()
    is_eq = is_equity_symbol(sym_u)
    clean_sym_u = sym_u if is_eq else canonical_symbol_key(sym_u)
    base = normalize_cache_base_asset(sym_u)
    candidates = [clean_sym_u] if clean_sym_u else ([sym_u] if sym_u else [symbol])
    if not is_eq and base and clean_sym_u != "USDCUSDT":
        for suffix in ("USDC", "USDT"):
            cand = f"{base}{suffix}"
            if cand not in candidates:
                candidates.append(cand)
    ts = px = None
    used = None
    _fk = _resolve_facade_attr("fetch_klines", fetch_klines, _ORIG_FETCH_KLINES)
    for sym in candidates:
        try:
            ts, px = _fk(sym, days)
            if len(ts) >= 100:
                used = sym
                break
        except Exception:
            continue

    if used is None:
        return ent if ent else fallback_res

    bt, bs = block_slopes(ts, px, window_h, step_h)
    eps = episodes(bt, bs, window_h)
    durs = [e["dur_h"] for e in eps]
    res = hybrid_T(durs, prior_T=prior_f)

    d = np.asarray(durs, dtype=float)
    p_cont = {}
    for t_days in range(1, 15):
        alive = d > t_days * 24.0
        if alive.sum() < 10:
            break
        p_cont[str(t_days)] = round(float((d > (t_days + 1) * 24.0).sum() / alive.sum()), 3)

    res["p_cont"] = p_cont
    res["source_symbol"] = used
    res["ts"] = time.time()
    cache_key = clean_sym_u if clean_sym_u else str(symbol or "")
    if (safe_pos_float(res.get("n"), default=0.0) or 0.0) > 0 or not (
        isinstance(ent, dict) and (safe_pos_float(ent.get("n"), default=0.0) or 0.0) > 0
    ):
        cache[cache_key] = res
        try:
            save_json_state(t_cache_path, cache, indent=2)
            _T_CACHE_FILE_MEMO.clear()
        except OSError:
            pass
    return res


def get_trend_survival_metrics(
    symbol: str,
    trend_duration_seconds: float = 0.0,
    *,
    allow_network: bool = False,
) -> Dict[str, Any]:
    """Retrieve or compute empirical trend survival metrics for a symbol."""
    dur_sec = safe_pos_float(trend_duration_seconds, default=0.0) or 0.0
    duration_days = max(0.0, dur_sec / 86400.0)
    median_d, p90_d, T_val = 3.0, 7.0, 8.0
    cached_t: Optional[float] = None

    def _apply_entry(ent_dict: dict, *, require_n: bool = False) -> None:
        nonlocal median_d, p90_d, T_val, cached_t
        median_d = safe_pos_float(ent_dict.get("median_d"), default=median_d)
        p90_d = safe_pos_float(ent_dict.get("p90_d"), default=p90_d)
        t_cand = safe_pos_float(ent_dict.get("T"), default=0.0)
        has_valid_n = (
            ("n" not in ent_dict and not require_n)
            or (safe_pos_float(ent_dict.get("n"), default=0.0) or 0.0) > 0
        )
        if t_cand > 0.0 and has_valid_n:
            T_val = cached_t = t_cand

    cached_entry = get_cached_t_entry(symbol)
    if cached_entry:
        _apply_entry(cached_entry)
    elif _parse_bool_flag(allow_network, False):
        sym_key = str(symbol or "").strip().upper()
        now_s = time.time()
        last_miss = _ON_DEMAND_T_MISS_TS.get(sym_key, 0.0)
        if not sym_key or (now_s - last_miss) >= _ON_DEMAND_T_MISS_TTL_SEC or not is_live_network_backfill_enabled():
            try:
                _est_fn = _resolve_facade_attr("estimate_T", estimate_T, _ORIG_ESTIMATE_T)
                est = _est_fn(symbol)
                if est:
                    _ON_DEMAND_T_MISS_TS.pop(sym_key, None)
                    _apply_entry(est, require_n=True)
                elif sym_key:
                    lru_dict_put(_ON_DEMAND_T_MISS_TS, sym_key, now_s, 128)
            except Exception:
                if sym_key:
                    lru_dict_put(_ON_DEMAND_T_MISS_TS, sym_key, now_s, 128)

    ratio = duration_days / p90_d if p90_d > 0 else 0.0
    is_exhausted = duration_days > p90_d if duration_days > 0 else False
    gauss_h = round(cached_t * 1.4, 4) if cached_t and cached_t > 0 else None

    return {
        "symbol": symbol,
        "duration_days": round(duration_days, 2),
        "median_days": round(median_d, 2),
        "p90_days": round(p90_d, 2),
        "T": round(T_val, 1),
        "gaussian_horizon_days": gauss_h,
        "duration_ratio_to_p90": round(ratio, 2),
        "is_exhausted": is_exhausted,
    }


def estimate_remaining_trend_hours(
    duration_hours: float,
    median_d: Optional[float] = None,
    p90_d: Optional[float] = None,
    p_cont: Optional[Dict[str, Any]] = None,
    symbol: Optional[str] = None,
    *,
    allow_network: bool = False,
) -> float:
    """Estimate conditional remaining trend duration (hours) from Weibull + empirical survival distribution."""
    dur_h = safe_nonneg_float(duration_hours, default=0.0) or 0.0
    if dur_h <= 0.0:
        return 0.0
    med_f = safe_pos_float(median_d, default=None)
    p90_f = safe_pos_float(p90_d, default=None)
    if (med_f is None or p90_f is None or p_cont is None) and isinstance(symbol, str) and symbol.strip():
        cached_ent = get_cached_t_entry(symbol)
        _est_fn = _resolve_facade_attr("estimate_T", estimate_T, _ORIG_ESTIMATE_T)
        is_est_patched = (
            _est_fn is not _ORIG_ESTIMATE_T
            or _resolve_facade_attr("fetch_klines", fetch_klines, _ORIG_FETCH_KLINES) is not _ORIG_FETCH_KLINES
        )
        if (
            cached_ent is None
            and (_parse_bool_flag(allow_network, False) or is_est_patched)
            and (is_live_network_backfill_enabled() or is_est_patched)
        ):
            sym_key = symbol.strip().upper()
            now_s = time.time()
            if (now_s - _ON_DEMAND_T_MISS_TS.get(sym_key, 0.0)) >= _ON_DEMAND_T_MISS_TTL_SEC or is_est_patched:
                try:
                    cached_ent = _est_fn(symbol)
                    if cached_ent is None:
                        lru_dict_put(_ON_DEMAND_T_MISS_TS, sym_key, now_s, 128)
                    else:
                        _ON_DEMAND_T_MISS_TS.pop(sym_key, None)
                except Exception:
                    lru_dict_put(_ON_DEMAND_T_MISS_TS, sym_key, now_s, 128)
                    cached_ent = None
        if isinstance(cached_ent, dict):
            if med_f is None:
                med_f = safe_pos_float(cached_ent.get("median_d"), default=None)
            if p90_f is None:
                p90_f = safe_pos_float(cached_ent.get("p90_d"), default=None)
            if p_cont is None and isinstance(cached_ent.get("p_cont"), dict):
                p_cont = cached_ent.get("p_cont")
    if med_f is None or p90_f is None or p90_f <= med_f:
        return round(dur_h * 0.5, 1)
    k = max(0.8, min(3.5, math.log(math.log(10.0) / math.log(2.0)) / math.log(p90_f / med_f)))
    lam = med_f / (math.log(2.0) ** (1.0 / k))
    t_d = dur_h / 24.0
    try:
        base_pow = (t_d / lam) ** k
    except OverflowError:
        return 1.0
    step = max(0.05, p90_f / 150.0)
    rem_d = 0.0
    for i in range(1, 600):
        u = (i - 0.5) * step
        try:
            s_cond = min(1.0, math.exp(min(0.0, base_pow - ((t_d + u) / lam) ** k)))
        except OverflowError:
            break
        rem_d += s_cond * step
        if s_cond < 1e-4:
            break
    weibull_rem_h = rem_d * 24.0
    if isinstance(p_cont, dict) and p_cont:
        day_idx = max(1, int(math.floor(t_d)))
        valid_days = [int(x) for x in p_cont if str(x).isdigit()]
        if valid_days:
            target_d = min(day_idx, max(valid_days))
            p_vals = [
                pf
                for k_s, v in p_cont.items()
                if str(k_s).isdigit()
                and int(k_s) >= target_d
                and (pf := safe_pos_float(v, default=0.0) or 0.0) > 0.05
                and pf < 0.98
            ]
            if p_vals:
                p_local = sum(p_vals) / len(p_vals)
                if t_d > p90_f:
                    p_local *= max(0.35, p90_f / t_d)
                emp_rem_h = -24.0 / math.log(max(0.05, min(0.95, p_local)))
                weibull_rem_h = 0.65 * weibull_rem_h + 0.35 * emp_rem_h
    return round(max(1.0, weibull_rem_h), 1)


def _dbg_weight(*args: Any, **kwargs: Any) -> None:
    pa_mod = sys.modules.get("priceAnalysis")
    dbg_flag = getattr(pa_mod, "_DEBUG", None) if pa_mod is not None else None
    if dbg_flag is None:
        dbg_flag = _parse_bool_flag(os.environ.get("PRICEANALYSIS_DEBUG"), False)
    if dbg_flag:
        print(*args, **kwargs)


def format_duration(seconds: Any) -> str:
    """Convert seconds to the human-readable ``Xd Yh Zm`` format."""
    sec_f = safe_pos_float(seconds, default=None)
    if sec_f is None:
        return "0m"
    days = int(sec_f // 86400)
    hours = int((sec_f % 86400) // 3600)
    minutes = int((sec_f % 3600) // 60)

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")

    return " ".join(parts) if parts else "0m"


def format_timestamp(ts: Any) -> str:
    """Convert a timestamp to a human-readable representation."""
    ts_f = safe_nonneg_float(ts, default=0.0)
    if ts_f >= 1e11:
        ts_f /= 1000.0
    try:
        return datetime.fromtimestamp(ts_f).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, OSError, OverflowError):
        return datetime.fromtimestamp(0).strftime("%Y-%m-%d %H:%M:%S")


last_timestamp: dict = {}
last_w: dict = {}
last_calc_time: dict = {}
last_duration: dict = {}


# Zone 1: 0 to T is Gaussian, confident in the middle and uncertain at endpoints.
# Zone 2: T to T*(1+percentage) is an over-age but persistent trend; high weight (0.86).
# Zone 3: above T*(1+percentage) is a very old trend; use a conservative weight (0.22).
# Aligned BUY+UP or SELL+DOWN scales the Gaussian peak to peak_weight.
# Countertrend SELL+UP or BUY+DOWN uses the inverse global Gaussian.
def get_trade_weight(
    T: Any,
    trend_len: Any,
    trend: Any,
    order_type: Any,
    exceed_percent: float = 0.4,
    max_against_trend: float = 0.15,
    peak_weight: float = 0.95,
    min_weight: float = 0.02,
    lindy_plateau: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Gaussian survival-adjusted allocation weight curve for a trend episode."""
    if not isinstance(order_type, str) or not isinstance(trend, str):
        return np.array([]), np.array([])
    T_f = safe_pos_float(T, default=None)
    trend_len_f = safe_nonneg_float(trend_len, default=None)
    if T_f is None or trend_len_f is None:
        return np.array([]), np.array([])
    T = T_f
    trend_len = trend_len_f
    exceed_percent = safe_nonneg_float(exceed_percent, 0.4)
    max_against_trend = safe_nonneg_float(max_against_trend, 0.15)
    peak_weight = safe_nonneg_float(peak_weight, 0.95)
    min_weight = safe_nonneg_float(min_weight, 0.02)
    side_u = normalize_side(order_type)
    if side_u not in ("BUY", "SELL"):
        return np.array([]), np.array([])
    trend_norm = normalize_trend_direction(trend, default=None)
    if trend_norm is None:
        return np.array([]), np.array([])
    aligned = (side_u == "BUY" and trend_norm == "up") or (side_u == "SELL" and trend_norm == "down")

    T_extended = T * (1 + exceed_percent)

    if T < trend_len <= T_extended:
        w_val = 0.86 if aligned else max_against_trend
        _dbg_weight(
            f"[DEBUG] Zone 2: trend_len={trend_len:.2f} exceeds T={T} "
            f"but is below T_extended={T_extended}. Aligned={aligned}, return {w_val}  "
        )
        return np.array([0.0]), np.array([w_val])

    if trend_len > T_extended:
        w_val = 0.22 if aligned else max_against_trend
        _dbg_weight(
            f"[DEBUG] Zone 3: trend_len={trend_len:.2f} is above "
            f"T_extended={T_extended}. Aligned={aligned}, return {w_val} "
        )
        return np.array([0.0]), np.array([w_val])

    t_steps = max(1, int(round(T)))
    idx = max(0, min(int(trend_len), t_steps - 1))
    t_full, w_full = gaussian_weights_from_idx(T=t_steps, idx=0)
    if len(w_full) == 0:
        _dbg_weight("[DEBUG] Zone 1: gaussian_weights_from_idx returned empty. return [0.05]")
        return np.array([0.0]), np.array([0.05])
    w01_full = w_full / w_full.max()
    if _parse_bool_flag(lindy_plateau, True):
        peak_i = int(np.argmax(w01_full))
        w01_full = w01_full.copy()
        w01_full[peak_i:] = 1.0
    t_seq, w01 = t_full[idx:], w01_full[idx:]
    _dbg_weight(
        f"[DEBUG] Zone 1: trend_len={trend_len:.2f}, slice from idx={idx} "
        f"up to T={T}. Aligned={aligned}, gauss01[0]={w01[0]:.4f}"
    )

    if aligned:
        w_seq = w01 * peak_weight
    else:
        _dbg_weight(
            f"[DEBUG] Order type {order_type} is not aligned with the trend {trend}, "
            f"globally inverted, max_against_trend={max_against_trend}"
        )
        w_seq = min_weight + (1.0 - w01) * (max_against_trend - min_weight)

    return t_seq, w_seq


def get_weight_for_cash_permission_at_quant_time(
    symbol: Any,
    order_type: Any,
    T_quanta: Any = None,
    quant_seconds: Any = 3600 * 24,
    draw: bool = False,
    *,
    allow_network: bool = False,
) -> Optional[float]:
    """Resolve Gaussian cash-permission allocation weight from cached long-term trend and empirical T.

    Read-only consumers (`allow_network=False`, the default on the order guard path) read
    `cache_long_trend.json` and `cache_T_trend.json` without triggering live OHLC network fetches
    or importing heavy plotting daemons.
    """
    import cachetrend as ct

    if not isinstance(symbol, str) or not symbol.strip():
        return None
    if not isinstance(order_type, str):
        return None
    side_u = normalize_side(order_type)
    if side_u not in ("BUY", "SELL"):
        return None
    if isinstance(T_quanta, bool) or isinstance(quant_seconds, bool):
        return None

    lt_mgr = ct.get_long_trend_manager()
    if hasattr(lt_mgr, "resync_mem_file"):
        lt_mgr.resync_mem_file()
    all_trend_data = lt_mgr.cache
    items = lookup_symbol_cache_items(all_trend_data, symbol)
    if not items:
        print(f"Symbol {symbol} is not present in the trends that were read.")
        return None

    trend = scan_latest_long_trend_value(
        items, lambda e: e if e.get("duration_seconds") is not None else None
    )
    if trend is None:
        print(f" No trend in cache for symbol {symbol}.")
        return None

    est = None
    if T_quanta is None:
        try:
            _est_fn = resolve_patched_attr(
                "estimate_T",
                _ORIG_ESTIMATE_T,
                local_val=estimate_T,
                modules=("priceAnalysis", "intelligence.internal.state.survival", "forecast.trend_survival"),
                require_callable=True,
            )
            _fk_fn = resolve_patched_attr(
                "fetch_klines",
                _ORIG_FETCH_KLINES,
                local_val=fetch_klines,
                modules=("priceAnalysis", "intelligence.internal.state.survival", "forecast.trend_survival"),
                require_callable=True,
            )
            is_est_patched = (_est_fn is not _ORIG_ESTIMATE_T) or (_fk_fn is not _ORIG_FETCH_KLINES)
            if _parse_bool_flag(allow_network, False) or is_est_patched:
                est = _est_fn(symbol)
            else:
                est = get_cached_t_entry(symbol) or {"T": 14}
            T_quanta = est["T"]
        except Exception as e:
            T_quanta = 14
            print(f"[{symbol}] T estimation failed ({e}) — using the prior T=14")

    dur_sec = safe_pos_float(trend.get("duration_seconds"), default=None)
    t_quanta_f = safe_pos_float(T_quanta, default=None)
    quant_sec_f = safe_pos_float(quant_seconds, default=None)
    if dur_sec is None or t_quanta_f is None or quant_sec_f is None:
        print(f"[{symbol}] duration_seconds/T_quanta invalid, return None")
        return None

    direction = normalize_trend_direction(trend.get("direction"), default=None)
    if direction is None:
        return None

    duration_days = dur_sec / 86400.0
    start_ts_val = safe_float(trend.get("start_timestamp"), default=0.0)
    timestamp = safe_pos_float(trend.get("timestamp"), default=None)
    memo_key = (
        (symbol, side_u, t_quanta_f, direction)
        if quant_sec_f == 86400.0
        else (symbol, side_u, t_quanta_f, direction, quant_sec_f)
    )
    curr_now = time.time()
    cache_age = curr_now - last_calc_time.get(memo_key, 0.0)
    dur_drift = abs(dur_sec - last_duration.get(memo_key, 0.0))

    if (
        timestamp is not None
        and timestamp == last_timestamp.get(memo_key)
        and 0.0 <= cache_age < 300.0
        and dur_drift < 300.0
    ):
        cached_w = last_w.get(memo_key)
        if cached_w is not None and len(cached_w) > 0 and not np.isnan(cached_w[0]):
            print(f"not new timestamp (age={cache_age:.1f}s), use weight from mem cache.")
            return float(cached_w[0])

    if isinstance(est, dict):
        print(
            f"[{symbol}] T AUTO (empirical hybrid): {T_quanta} days  "
            f"(n={est.get('n')} episodes, w_empirical={est.get('w')}, "
            f"median={est.get('median_d')}d, P90={est.get('p90_d')}d)"
        )
    print(f"Trend read from the cache manager for symbol {symbol}: {trend}")
    print(f"   Start trend:     {format_timestamp(start_ts_val)}")
    print(f"   Duration:        {format_duration(dur_sec)} ({duration_days:.1f} days)")

    trend_len_quanta = dur_sec / quant_sec_f
    if safe_pos_float(trend_len_quanta, default=None) is None:
        print(f"[{symbol}] duration_seconds invalid, return None")
        return None

    _tw_fn = resolve_patched_attr(
        "get_trade_weight",
        _ORIG_GET_TRADE_WEIGHT,
        local_val=get_trade_weight,
        modules=("priceAnalysis", "intelligence.internal.state.survival", "forecast.trend_survival"),
        require_callable=True,
    )
    t, w = _tw_fn(
        T=t_quanta_f,
        trend_len=trend_len_quanta,
        trend=direction,
        order_type=side_u,
    )

    if len(w) == 0:
        print(f"[{symbol}] w is empty, returning None")
        return None

    current_weight = float(w[0])
    if np.isnan(current_weight) or current_weight <= 0:
        print(f"[{symbol}] Invalid weight: {current_weight}, returning None")
        return None

    print(f"[{symbol}] first 5 weights: {w[:5]}")
    print(f"Sum of all {len(w)} weights = {w.sum():.4f}")

    if _parse_bool_flag(draw, False):
        import matplotlib.pyplot as plt

        plt.plot(t, w, label=symbol)
        plt.legend()
        plt.show()

    lru_dict_put(last_w, memo_key, w, 256)
    lru_dict_put(last_timestamp, memo_key, timestamp, 256)
    lru_dict_put(last_calc_time, memo_key, curr_now, 256)
    lru_dict_put(last_duration, memo_key, dur_sec, 256)
    return current_weight


_JSON_RECOVERING_MEMO: Dict[str, Tuple[tuple, dict]] = {}
_ORIG_GET_CACHEDB_DIR = get_cachedb_dir


def _resolve_guard_cachedb_dir() -> str:
    cdir_fn = resolve_patched_attr(
        "get_cachedb_dir",
        _ORIG_GET_CACHEDB_DIR,
        local_val=get_cachedb_dir,
        modules=("order_guard", "intelligence.internal.state.survival", "cachepaths"),
        require_callable=True,
    )
    return str(cdir_fn())


def _scan_cached_long_trend(symbol: str, parser: Callable[[Dict[str, Any]], Any]) -> Any:
    if not symbol or not isinstance(symbol, str):
        return None
    try:
        p = str(resolve_cachedb_filepath(os.path.join(_resolve_guard_cachedb_dir(), CACHE_LONG_TREND_FILENAME)))
        data = read_json_dict_cached(p, _JSON_RECOVERING_MEMO, max_entries=32)
        entries = (
            lookup_symbol_cache_items(data, symbol, require_list=True, unwrap_items=True)
            if isinstance(data, dict)
            else None
        )
        return scan_latest_long_trend_value(entries, parser)
    except Exception:
        return None


def _resolve_trend_duration(symbol: str) -> float:
    """Resolve trend duration in seconds from memory/cache files once."""
    return (
        _scan_cached_long_trend(
            symbol, lambda e: safe_pos_float(e.get("duration_seconds"), default=None)
        )
        or 0.0
    )


def resolve_effective_trend_duration(
    symbol: str,
    explicit_duration_sec: float = 0.0,
    regime_context: Any = None,
    *,
    fallback_resolver: Optional[Callable[[str], float]] = None,
) -> float:
    """Resolve trend duration from explicit override -> regime_context -> live cache."""
    dur_sec = safe_pos_float(explicit_duration_sec, 0.0)
    if dur_sec <= 0.0 and regime_context is not None:
        dur_sec = safe_pos_float(obj_attr(regime_context, "trend_duration_seconds", 0.0), 0.0)
    if dur_sec <= 0.0 and symbol:
        resolver = fallback_resolver or resolve_patched_attr(
            "_resolve_trend_duration",
            _ORIG_RESOLVE_TREND_DURATION,
            local_val=_resolve_trend_duration,
            modules=("order_guard", "intelligence.internal.state.survival"),
            require_callable=True,
        )
        dur_sec = safe_pos_float(resolver(symbol), 0.0)
    return dur_sec


def _resolve_trend_direction(symbol: str, regime_context: Any = None) -> Optional[str]:
    """Resolve long-term trend direction ('up'/'down'/'bull'/'bear') from regime_context or cache."""
    if regime_context is not None:
        rt = obj_attr(regime_context, "resolved_trend") or obj_attr(regime_context, "regime")
        norm_rt = normalize_trend_direction(rt, default=None, preserve_bull_bear=True)
        if norm_rt is not None:
            return norm_rt
    return _scan_cached_long_trend(
        symbol,
        lambda e: normalize_trend_direction(
            e.get("direction"), default=None, preserve_bull_bear=True
        ),
    )


def _read_cached_price_history(
    symbol: str,
    window_seconds: float = 7200.0,
    now: Optional[float] = None,
) -> Optional[List[Tuple[float, float]]]:
    """Read rolling price history from local cache for parabolic surge evaluation.

    Prefers the authoritative 24h per-symbol cache (``cachedb/cache_24_price_{symbol}.json``)
    which captures every high-resolution tick, and falls back to ``cache_watcher_all_price.json``
    with exact normalized base-symbol matching for cross-venue assets (e.g. HYPE on Kraken).
    Never uses substring matching so e.g. ETHFI does not accidentally match ETH.
    """
    if not symbol or not isinstance(symbol, str):
        return None
    su = symbol.strip().upper()
    if not su:
        return None
    win_sec = safe_pos_float(window_seconds, 7200.0)
    now_ts = resolve_now_ts(now)
    cachedb_dir = _resolve_guard_cachedb_dir()
    cutoff_ts = now_ts - win_sec
    history: List[Tuple[float, float]] = []
    cand_symbols = cache_symbol_variants(symbol, include_base=True)

    def _parse_tick(item: Any) -> Optional[Tuple[float, float]]:
        if isinstance(item, dict):
            ts_raw = item.get("ts") if item.get("ts") is not None else item.get("timestamp")
            px_raw = item.get("price") if item.get("price") is not None else item.get("close")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            ts_raw, px_raw = item[0], item[1]
        else:
            return None
        ts_sec = normalize_epoch_sec(ts_raw)
        px = safe_pos_float(px_raw, default=None)
        return (ts_sec, px) if (ts_sec is not None and px is not None) else None

    def _extract_valid_ticks(raw_items: Any) -> None:
        if isinstance(raw_items, list) and len(raw_items) > 512:
            first_tick = next((t for x in raw_items[:16] if (t := _parse_tick(x)) is not None), None)
            last_tick = next((t for x in reversed(raw_items[-16:]) if (t := _parse_tick(x)) is not None), None)
            iter_seq = raw_items if (first_tick and last_tick and first_tick[0] > last_tick[0]) else reversed(raw_items)
            tail_buf: List[Tuple[float, float]] = []
            older_streak = 0
            for item in iter_seq:
                parsed = _parse_tick(item)
                if parsed is None:
                    continue
                ts_sec, px = parsed
                if ts_sec < cutoff_ts:
                    older_streak += 1
                    if older_streak >= 64:
                        break
                    continue
                older_streak = 0
                if ts_sec <= (now_ts + 5.0):
                    tail_buf.append((ts_sec, px))
            tail_buf.reverse()
            if len(tail_buf) > 1 and any(tail_buf[i][0] > tail_buf[i + 1][0] for i in range(len(tail_buf) - 1)):
                tail_buf.sort(key=lambda pair: pair[0])
            history.extend(tail_buf)
            return

        for item in raw_items or []:
            parsed = _parse_tick(item)
            if parsed is not None:
                ts_sec, px = parsed
                if cutoff_ts <= ts_sec <= (now_ts + 5.0):
                    history.append((ts_sec, px))
        if len(history) > 1 and any(history[i][0] > history[i + 1][0] for i in range(len(history) - 1)):
            history.sort(key=lambda pair: pair[0])

    p24_paths = list(dict.fromkeys(
        str(resolve_cachedb_filepath(os.path.join(cachedb_dir, cache_24_price_filename(cand))))
        for cand in cand_symbols
        if "/" not in cand and ("_" not in cand or has_equity_suffix(cand))
    ))
    watcher_path = str(resolve_cachedb_filepath(os.path.join(cachedb_dir, CACHE_WATCHER_ALL_PRICE_FILENAME)))
    for p_path in (*p24_paths, watcher_path):
        data = read_json_dict_cached(p_path, _JSON_RECOVERING_MEMO, max_entries=32)
        if not isinstance(data, (dict, list)):
            continue
        try:
            raw_items = (
                data
                if isinstance(data, list)
                else lookup_symbol_cache_items(data, su, cand_keys=cand_symbols, require_list=True, unwrap_items=True)
            )
            _extract_valid_ticks(raw_items)
            if history:
                break
        except Exception:
            pass

    return history if history else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Empirical trend survival analysis.")
    ap.add_argument("--symbol", default="BTCUSDC")
    ap.add_argument("--days", type=int, default=700)
    ap.add_argument("--window", type=int, default=24, help="hours per window")
    ap.add_argument("--step", type=int, default=8)
    ap.add_argument("--T", type=float, default=14.0, help="prior Gaussian horizon (days)")
    ap.add_argument("--estimate", action="store_true", help="only estimate T and exit")
    args = ap.parse_args()

    if args.estimate:
        _est_fn = _resolve_facade_attr("estimate_T", estimate_T, _ORIG_ESTIMATE_T)
        est = _est_fn(args.symbol, days=args.days, window_h=args.window, step_h=args.step, prior_T=args.T)
        print(f"[{args.symbol}] Estimated T = {est['T']} days (P90 {est.get('p90_d')}d, median {est.get('median_d')}d)")
        return 0

    _fk = _resolve_facade_attr("fetch_klines", fetch_klines, _ORIG_FETCH_KLINES)
    ts, px = _fk(args.symbol, args.days)
    if len(ts) < 100:
        print(f"Not enough data for {args.symbol}")
        return 1
    bt, bs = block_slopes(ts, px, args.window, args.step)
    eps = episodes(bt, bs, args.window)
    t_grid = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 17, 21]

    all_d = [e["dur_h"] for e in eps]
    cont = survival_report(all_d, "ALL trends", t_grid)
    if cont:
        print("    " + verdict(cont, args.T / 2))
    return 0


_ORIG_FETCH_KLINES = fetch_klines
_ORIG_ESTIMATE_T = estimate_T
_ORIG_GET_TRADE_WEIGHT = get_trade_weight
_ORIG_RESOLVE_TREND_DURATION = _resolve_trend_duration
_ORIG_RESOLVE_TREND_DIRECTION = _resolve_trend_direction
_ORIG_READ_CACHED_PRICE_HISTORY = _read_cached_price_history


if __name__ == "__main__":
    sys.exit(main())
