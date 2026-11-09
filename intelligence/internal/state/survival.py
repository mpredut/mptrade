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
import math
import os
import sys
import time
from typing import Any, Dict, Optional

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from binance_api.market_data import fetch_ohlc_candles
from botcore import (
    _parse_bool_flag,
    is_live_network_backfill_enabled,
    lru_dict_put,
    normalize_epoch_sec,
    resolve_patched_attr,
    safe_float,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
    safe_pos_int,
)
from cachepaths import cache_path, canonical_symbol_key, normalize_cache_base_asset
from instrument_registry import is_equity_symbol
from intelligence.macro_sentiment_ssot import lookup_symbol_cache_items
from state_io import read_json_dict_cached, save_json_state

_DEFAULT_T_CACHE_FILE = os.path.join(_ROOT, "cachedb", "cache_T_trend.json")
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
        return cache_path("cache_T_trend.json")
    except Exception:
        return T_CACHE_FILE


def fetch_klines(symbol: str, days: int, interval: str = "1h") -> tuple[np.ndarray, np.ndarray]:
    """Return paginated Binance history with 1,000 candles per request (falling back to Kraken 1h OHLC or Yahoo Finance for equities)."""
    sym_u = str(symbol or "").strip().upper()
    if is_equity_symbol(sym_u):
        try:
            import importlib

            market_data = importlib.import_module("212trading.market_data")
            yahoo_sym = market_data.t212_to_yahoo(sym_u)
            iv_norm = str(interval or "1h").strip()
            if iv_norm not in ("5m", "15m", "30m", "1h", "1d"):
                iv_norm = "1h"
            max_days = 59 if iv_norm in ("5m", "15m", "30m") else 720
            days_i = min(max(1, safe_pos_int(days, default=30) or 30), max_days)
            bars = market_data.fetch_yahoo_candles(
                yahoo_sym, f"{days_i}d", iv_norm, drop_incomplete=False
            ) or []
            y_ts: list[float] = []
            y_px: list[float] = []
            for b in bars:
                if not isinstance(b, dict):
                    continue
                tf = normalize_epoch_sec(b.get("timestamp"), default=None)
                pf = safe_pos_float(b.get("close"), default=None)
                if tf is not None and pf is not None:
                    y_ts.append(tf)
                    y_px.append(pf)
            return np.asarray(y_ts, dtype=float), np.asarray(y_px, dtype=float)
        except Exception:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
    candles = fetch_ohlc_candles(symbol, interval=interval, lookback_days=days, drop_incomplete=False) or []
    ts_list: list[float] = []
    px_list: list[float] = []
    for c in candles:
        if not isinstance(c, dict):
            continue
        tf = normalize_epoch_sec(c.get("timestamp"), default=None)
        pf = safe_pos_float(c.get("close"), default=None)
        if tf is not None and pf is not None:
            ts_list.append(tf)
            px_list.append(pf)
    if len(ts_list) < 50 and is_live_network_backfill_enabled():
        try:
            from providers.kraken_provider import KrakenProvider

            base_u = normalize_cache_base_asset(symbol)
            krk_sym = f"{base_u}USD" if base_u else str(symbol or "").strip().upper()
            s60 = KrakenProvider().ohlc_series(krk_sym, 60)
            k_ts = [float(t) for t in getattr(s60, "timestamps", ()) if (safe_pos_float(t, default=None) or 0.0) > 0]
            k_px = [float(p) for p in getattr(s60, "closes", ()) if (safe_pos_float(p, default=None) or 0.0) > 0]
            if len(k_ts) >= 50 and len(k_ts) == len(k_px):
                ts_list, px_list = k_ts, k_px
        except Exception:
            pass
    return np.asarray(ts_list, dtype=float), np.asarray(px_list, dtype=float)


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


if __name__ == "__main__":
    sys.exit(main())
