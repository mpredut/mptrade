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
import json
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
    safe_float,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
    safe_pos_int,
)
from state_io import read_recovering_previous, save_json_state

_DEFAULT_T_CACHE_FILE = os.path.join(_ROOT, "cachedb", "cache_T_trend.json")
T_CACHE_FILE = _DEFAULT_T_CACHE_FILE


def _effective_t_cache_file() -> str:
    if isinstance(T_CACHE_FILE, str) and T_CACHE_FILE.strip() and T_CACHE_FILE != _DEFAULT_T_CACHE_FILE:
        return T_CACHE_FILE
    facade = sys.modules.get("forecast.trend_survival")
    facade_file = getattr(facade, "T_CACHE_FILE", None) if facade is not None else None
    if isinstance(facade_file, str) and facade_file.strip() and facade_file != _DEFAULT_T_CACHE_FILE:
        return facade_file
    try:
        import cachepaths
        return cachepaths.cache_path("cache_T_trend.json")
    except Exception:
        return T_CACHE_FILE


def fetch_klines(symbol: str, days: int, interval: str = "1h") -> tuple[np.ndarray, np.ndarray]:
    """Return paginated Binance history with 1,000 candles per request."""
    candles = fetch_ohlc_candles(symbol, interval=interval, lookback_days=days, drop_incomplete=False) or []
    ts_list: list[float] = []
    px_list: list[float] = []
    for c in candles:
        if not isinstance(c, dict):
            continue
        tf = safe_pos_float(c.get("timestamp"), default=None)
        pf = safe_pos_float(c.get("close"), default=None)
        if tf is not None and pf is not None:
            ts_list.append(tf)
            px_list.append(pf)
    return np.asarray(ts_list, dtype=float), np.asarray(px_list, dtype=float)


def block_slopes(ts, px, window_h, step_h):
    """Return slope signs for each [t-window, t] window stepped by step_h."""
    out_t, out_s = [], []
    if (
        ts is None
        or px is None
        or isinstance(ts, (bool, str, bytes))
        or isinstance(px, (bool, str, bytes))
    ):
        return np.array(out_t), np.array(out_s)
    try:
        raw_ts = list(ts)
        raw_px = list(px)
    except TypeError:
        return np.array(out_t), np.array(out_s)
    n_common = min(len(raw_ts), len(raw_px))
    if n_common < 5:
        return np.array(out_t), np.array(out_s)
    clean_ts = []
    clean_px = []
    for t_val, p_val in zip(raw_ts[:n_common], raw_px[:n_common]):
        tf = safe_pos_float(t_val, default=None)
        pf = safe_pos_float(p_val, default=None)
        if tf is not None and pf is not None:
            clean_ts.append(tf)
            clean_px.append(pf)
    if len(clean_ts) < 5:
        return np.array(out_t), np.array(out_s)
    ts = np.asarray(clean_ts, dtype=float)
    px = np.asarray(clean_px, dtype=float)
    if np.any(np.diff(ts) < 0):
        order = np.argsort(ts, kind="mergesort")
        ts = ts[order]
        px = px[order]
    w_h = safe_pos_float(window_h, default=0.0)
    s_h = safe_pos_float(step_h, default=0.0)
    if w_h <= 0.0 or s_h <= 0.0:
        return np.array(out_t), np.array(out_s)
    w, s = w_h * 3600.0, s_h * 3600.0
    t = float(ts[0]) + w
    t_end = float(ts[-1])
    while t <= t_end:
        lo = int(np.searchsorted(ts, t - w, "left"))
        hi = int(np.searchsorted(ts, t, "right"))
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
    if (
        bt is None
        or bs is None
        or isinstance(bt, (bool, str, bytes))
        or isinstance(bs, (bool, str, bytes))
    ):
        return eps
    w_h = safe_pos_float(window_h, default=0.0)
    if w_h <= 0.0:
        return eps
    nt = safe_nonneg_int(noise_tolerance, default=2)
    mc = safe_pos_int(min_confirm, default=3)
    try:
        raw_bt = list(bt)
        raw_bs = list(bs)
    except TypeError:
        return eps
    n_common = min(len(raw_bt), len(raw_bs))
    clean_bt = []
    clean_bs = []
    for t_val, s_val in zip(raw_bt[:n_common], raw_bs[:n_common]):
        tf = safe_float(t_val, default=None)
        sf = safe_float(s_val, default=None)
        if tf is not None and sf is not None:
            clean_bt.append(tf)
            clean_bs.append(1.0 if sf > 0 else (-1.0 if sf < 0 else 0.0))
    n = len(clean_bt)
    if n == 0:
        return eps
    bt = np.asarray(clean_bt, dtype=float)
    bs = np.asarray(clean_bs, dtype=float)
    i = 0
    while i < n:
        sign = bs[i]
        if sign == 0.0:
            i += 1
            continue
        start = bt[i] - w_h * 3600.0
        last_confirm = bt[i]
        confirms, noise = 1, 0
        j = i + 1
        while j < n:
            if bs[j] == sign:
                confirms += 1
                noise = 0
                last_confirm = bt[j]
            elif noise < nt:
                noise += 1
            else:
                break
            j += 1
        if confirms >= mc:
            eps.append(
                {
                    "dir": "up" if sign > 0 else "down",
                    "dur_h": (last_confirm - start) / 3600.0,
                }
            )
        nxt = int(np.searchsorted(bt, last_confirm, "right"))
        i = max(nxt, i + 1)
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


def _extract_base_symbol(symbol: str) -> str:
    from cachepaths import normalize_cache_base_asset
    return normalize_cache_base_asset(symbol)


_T_CACHE_FILE_MEMO: dict = {}


def _read_t_cache_dict(t_cache_path: str) -> Optional[dict]:
    if not t_cache_path or not isinstance(t_cache_path, str):
        return None
    stat_sig = None
    try:
        st = os.stat(t_cache_path)
        stat_sig = (st.st_mtime_ns, st.st_size, id(read_recovering_previous))
    except OSError:
        stat_sig = None
    if stat_sig is not None:
        cached = _T_CACHE_FILE_MEMO.get(t_cache_path)
        if cached is not None and cached[0] == stat_sig:
            return cached[1]
    loaded = read_recovering_previous(t_cache_path, json.loads, default=None)
    if isinstance(loaded, dict):
        if stat_sig is not None:
            if len(_T_CACHE_FILE_MEMO) > 16:
                _T_CACHE_FILE_MEMO.clear()
            _T_CACHE_FILE_MEMO[t_cache_path] = (stat_sig, loaded)
        return loaded
    return None


def _lookup_t_cache_entry(data: Dict[str, Any], symbol: str) -> Optional[dict]:
    if not isinstance(data, dict) or not isinstance(symbol, str) or not symbol.strip():
        return None
    from cachepaths import cache_symbol_variants
    for cand in (symbol, *cache_symbol_variants(symbol)):
        ent = data.get(cand)
        if isinstance(ent, dict):
            return ent
    target_base = _extract_base_symbol(symbol)
    if target_base:
        for k, v in data.items():
            if isinstance(v, dict) and _extract_base_symbol(str(k)) == target_base:
                return v
    return None


def get_cached_t_entry(symbol: str) -> Optional[dict]:
    """Read and look up the cached T entry for ``symbol`` from disk in 0 ms."""
    try:
        data = _read_t_cache_dict(_effective_t_cache_file())
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
    loaded_cache = _read_t_cache_dict(t_cache_path)
    cache = dict(loaded_cache) if isinstance(loaded_cache, dict) else {}

    ent = _lookup_t_cache_entry(cache, symbol)
    if ent:
        ts_val = safe_pos_float(ent.get("ts"), default=None)
        if ts_val is not None:
            now_curr = time.time()
            if -60.0 <= (now_curr - ts_val) < ttl_f * 86400.0:
                return ent

    sym_u = symbol.upper().strip()
    clean_sym_u = sym_u.replace("/", "").replace("-", "").replace("_", "")
    base = _extract_base_symbol(sym_u)
    candidates = [clean_sym_u] if clean_sym_u else ([sym_u] if sym_u else [symbol])
    if base and clean_sym_u != "USDCUSDT":
        for suffix in ("USDC", "USDT"):
            cand = f"{base}{suffix}"
            if cand not in candidates:
                candidates.append(cand)
    ts = px = None
    used = None
    for sym in candidates:
        try:
            ts, px = fetch_klines(sym, days)
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
    cache[cache_key] = res
    try:
        save_json_state(t_cache_path, cache, indent=2)
        _T_CACHE_FILE_MEMO.pop(t_cache_path, None)
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
    median_d = 3.0
    p90_d = 7.0
    T_val = 8.0
    cached_t: Optional[float] = None

    cached_entry = get_cached_t_entry(symbol)
    allow_net = _parse_bool_flag(allow_network, False)
    if cached_entry:
        median_d = safe_pos_float(cached_entry.get("median_d"), default=median_d)
        p90_d = safe_pos_float(cached_entry.get("p90_d"), default=p90_d)
        t_cand = safe_pos_float(cached_entry.get("T"), default=0.0)
        if t_cand > 0.0:
            T_val = t_cand
            cached_t = T_val
    elif allow_net:
        try:
            est = estimate_T(symbol)
            if est:
                median_d = safe_pos_float(est.get("median_d"), default=median_d)
                p90_d = safe_pos_float(est.get("p90_d"), default=p90_d)
                t_cand = safe_pos_float(est.get("T"), default=0.0)
                if t_cand > 0.0 and safe_pos_float(est.get("n"), default=0.0) > 0:
                    T_val = t_cand
                    cached_t = T_val
        except Exception:
            pass

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
        est = estimate_T(args.symbol, days=args.days, window_h=args.window, step_h=args.step, prior_T=args.T)
        print(f"[{args.symbol}] Estimated T = {est['T']} days (P90 {est.get('p90_d')}d, median {est.get('median_d')}d)")
        return 0

    ts, px = fetch_klines(args.symbol, args.days)
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


if __name__ == "__main__":
    sys.exit(main())
