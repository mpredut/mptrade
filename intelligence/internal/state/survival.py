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
import urllib.request
from typing import Any, Dict, Optional

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from state_io import atomic_write_json

T_CACHE_FILE = os.path.join(_ROOT, "cachedb", "cache_T_trend.json")


def fetch_klines(symbol: str, days: int, interval: str = "1h") -> tuple[np.ndarray, np.ndarray]:
    """Return paginated Binance history with 1,000 candles per request."""
    end = int(time.time() * 1000)
    start = end - days * 86400 * 1000
    ts, px = [], []
    cur = start
    while cur < end:
        url = (
            f"https://api.binance.com/api/v3/klines?symbol={symbol}"
            f"&interval={interval}&startTime={cur}&limit=1000"
        )
        rows = json.loads(urllib.request.urlopen(url, timeout=25).read())
        if not rows:
            break
        for r in rows:
            ts.append(r[0] / 1000.0)
            px.append(float(r[4]))
        cur = rows[-1][0] + 1
        if len(rows) < 1000:
            break
    return np.array(ts), np.array(px)


def block_slopes(ts, px, window_h, step_h):
    """Return slope signs for each [t-window, t] window stepped by step_h."""
    out_t, out_s = [], []
    w, s = window_h * 3600.0, step_h * 3600.0
    t = ts[0] + w
    while t <= ts[-1]:
        lo = int(np.searchsorted(ts, t - w, "left"))
        hi = int(np.searchsorted(ts, t, "right"))
        if hi - lo >= 5:
            sl, _ = np.polyfit(ts[lo:hi] - ts[lo], px[lo:hi], 1)
            out_t.append(t)
            out_s.append(1.0 if sl > 0 else -1.0)
        t += s
    return np.array(out_t), np.array(out_s)


def episodes(bt, bs, window_h, noise_tolerance=2, min_confirm=3):
    """Return trend episodes in hours using the detector's semantics."""
    eps = []
    i = 0
    n = len(bs)
    while i < n:
        sign = bs[i]
        start = bt[i] - window_h * 3600.0
        last_confirm = bt[i]
        confirms, noise = 1, 0
        j = i + 1
        while j < n:
            if bs[j] == sign:
                confirms += 1
                noise = 0
                last_confirm = bt[j]
            elif noise < noise_tolerance:
                noise += 1
            else:
                break
            j += 1
        if confirms >= min_confirm:
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
    d = np.array(durs_h)
    if len(d) < 15:
        return None
    cont = {}
    for t_days in t_grid_days:
        t_h = t_days * 24.0
        alive = d > t_h
        if alive.sum() < 10:
            break
        p = float((d > t_h + horizon_h).sum() / alive.sum())
        cont[t_days] = p
    return cont


def verdict(cont: dict, mid: float) -> str:
    """Compare young (t < mid) and old (t >= mid) trend continuation."""
    young = [p for t, p in cont.items() if t < mid]
    old = [p for t, p in cont.items() if t >= mid]
    if not young or not old:
        return "insufficient_data"
    ym, om = float(np.mean(young)), float(np.mean(old))
    if om >= ym - 0.05:
        return "VALIDATED: old trend continues as long as middle (Lindy)"
    return "INVALIDATED: old trends terminate faster than young trends"


def hybrid_T(dur_hours, prior_T: float = 14.0, k: float = 30.0, t_min: int = 4, t_max: int = 30) -> dict:
    """Return hybrid T by combining empirical data with prior."""
    n = len(dur_hours)
    if n == 0:
        return {
            "T": int(round(prior_T)),
            "n": 0,
            "w": 0.0,
            "median_d": None,
            "p90_d": None,
            "T_emp": None,
        }
    d = np.asarray(dur_hours, dtype=float) / 24.0
    med, p90 = float(np.median(d)), float(np.percentile(d, 90))
    t_emp = max(p90, 2.0 * med)
    w = n / (n + k)
    T = min(max(w * t_emp + (1 - w) * prior_T, t_min), t_max)
    return {
        "T": int(round(T)),
        "n": n,
        "w": round(w, 2),
        "median_d": round(med, 1),
        "p90_d": round(p90, 1),
        "T_emp": round(t_emp, 1),
    }


def estimate_T(
    symbol: str,
    days: int = 540,
    window_h: int = 24,
    step_h: int = 8,
    prior_T: float = 14.0,
    ttl_days: float = 7.0,
) -> dict:
    """Estimate asset-specific trend lifetime T empirically from historical candlesticks."""
    cache = {}
    try:
        with open(T_CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
    except (OSError, ValueError):
        pass

    ent = cache.get(symbol)
    if ent and time.time() - ent.get("ts", 0) < ttl_days * 86400:
        return ent

    candidates = [symbol]
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
        if ent:
            return ent
        return {
            "T": int(round(prior_T)),
            "n": 0,
            "w": 0.0,
            "source_symbol": None,
            "median_d": None,
            "p90_d": None,
            "T_emp": None,
            "ts": 0,
        }

    bt, bs = block_slopes(ts, px, window_h, step_h)
    eps = episodes(bt, bs, window_h)
    durs = [e["dur_h"] for e in eps]
    res = hybrid_T(durs, prior_T=prior_T)

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
    cache[symbol] = res
    try:
        atomic_write_json(T_CACHE_FILE, cache, indent=2)
    except OSError:
        pass
    return res


def get_trend_survival_metrics(symbol: str, trend_duration_seconds: float = 0.0) -> Dict[str, Any]:
    """Retrieve or compute empirical trend survival metrics for a symbol."""
    duration_days = max(0.0, float(trend_duration_seconds) / 86400.0)
    median_d = 3.0
    p90_d = 7.0
    T_val = 8.0

    cached_entry: Optional[dict] = None
    if os.path.exists(T_CACHE_FILE):
        try:
            with open(T_CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                cached_entry = data.get(symbol)
        except Exception:
            cached_entry = None

    if cached_entry:
        median_d = float(cached_entry.get("median_d") or median_d)
        p90_d = float(cached_entry.get("p90_d") or p90_d)
        T_val = float(cached_entry.get("T") or T_val)
    else:
        try:
            est = estimate_T(symbol)
            if est:
                median_d = float(est.get("median_d") or median_d)
                p90_d = float(est.get("p90_d") or p90_d)
                T_val = float(est.get("T") or T_val)
        except Exception:
            pass

    ratio = duration_days / p90_d if p90_d > 0 else 0.0
    is_exhausted = duration_days > p90_d if duration_days > 0 else False

    return {
        "symbol": symbol,
        "duration_days": round(duration_days, 2),
        "median_days": round(median_d, 2),
        "p90_days": round(p90_d, 2),
        "T": round(T_val, 1),
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
