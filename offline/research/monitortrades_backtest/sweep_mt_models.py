#!/usr/bin/env python3
"""Fast vectorized multi-model portfolio replay backtest for monitortrades.py.

Evaluates Legacy, uniform Step-WOLS (p=2, p=3), Adaptive Noise-Gating (k=0.0..0.25
and scale-invariant z-score gating), and the Differentiated Per-Coin Step-WOLS
architecture (BTC p=3.0, Altcoins p=2.0 from instruments.conf) across all
monitortrades instruments (BTCUSDC, TAOUSDC, ARBUSDC, HYPEUSD) on:
  1. Full 329-day sparse archive (cache_sparse_price_*.jsonl)
  2. Dense 1-second archive (cache_365_price_*.jsonl / *.npz)
"""
from __future__ import annotations

import math
import os
import sys
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

import pricewindow as pw
from cachepaths import cache_path
from providers.replay_provider import load_price_series
import offline.research.monitortrades_backtest.run_replay_backtest as rrb


def compute_window_bounds(ts: np.ndarray, win_sec: float = rrb.DEFAULT_WINDOW_SECONDS):
    """Replicate exact cur_len, sample_rate, and [a, b] bounds of prewarm_trend_series."""
    N = len(ts)
    a_arr = np.zeros(N, dtype=np.int64)
    sr_arr = np.full(N, pw.DEFAULT_SAMPLE_RATE_SEC, dtype=np.float64)
    win_size = 200
    sample_rate = pw.DEFAULT_SAMPLE_RATE_SEC
    cur_len = 0
    prev_ts = None
    for i in range(N):
        t = float(ts[i])
        if prev_ts is not None and t > prev_ts:
            rate = t - prev_ts
            if rate > 0.0 and math.isfinite(rate):
                sample_rate = rate
                new_size = min(100000, max(10, int(win_sec / rate)))
                if new_size != win_size:
                    win_size = new_size
                    if cur_len > new_size:
                        cur_len = new_size
        prev_ts = t
        if cur_len >= win_size:
            cur_len = win_size
        else:
            cur_len += 1
        a_arr[i] = i - cur_len + 1
        sr_arr[i] = sample_rate
    return a_arr, sr_arr


def compute_all_signals(
    series: list[tuple[float, float]],
    win_sec: float = rrb.DEFAULT_WINDOW_SECONDS,
    symbol: str | None = None,
):
    """Vectorized O(1) prefix-sum evaluation of short-trend estimators over series."""
    N = len(series)
    ts = np.fromiter((t for t, _ in series), dtype=np.float64, count=N)
    px = np.fromiter((p for _, p in series), dtype=np.float64, count=N)

    a, sr = compute_window_bounds(ts, win_sec=win_sec)
    b = np.arange(N, dtype=np.int64)
    n = b - a + 1
    m = b - a
    valid = n >= 2

    y_ref = px[0]
    dy_ref = px - y_ref
    k_idx = np.arange(N, dtype=np.float64)

    P0 = np.cumsum(dy_ref)
    P1 = np.cumsum(k_idx * dy_ref)
    P2 = np.cumsum(dy_ref * dy_ref)

    def _range_sum(P, lo, hi):
        prev = np.where(lo > 0, P[np.maximum(0, lo - 1)], 0.0)
        return P[hi] - prev

    s0 = _range_sum(P0, a, b)
    s1 = _range_sum(P1, a, b)
    s2 = _range_sum(P2, a, b)

    n_f = n.astype(np.float64)
    m_f = np.maximum(1.0, m.astype(np.float64))
    ss_y = s2 - (s0 * s0) / n_f
    cov_xy = s1 - (a.astype(np.float64) + 0.5 * (n_f - 1.0)) * s0
    denom_ols = np.maximum(1.0, n_f * (n_f * n_f - 1.0))
    slope_full = np.where(valid & (ss_y > 1e-18), (12.0 * cov_xy) / denom_ols, 0.0)

    recent_sec = pw.effective_recent_gradient_seconds(None)
    r_arr = np.maximum(2, (recent_sec / np.maximum(1e-9, sr)).astype(np.int64))
    b_m1 = np.maximum(0, b - 1)
    a_p1 = np.minimum(N - 1, a + 1)
    sum_g_all = (px[b] - px[a]) + 0.5 * ((px[b] - px[b_m1]) + (px[a_p1] - px[a]))

    b_mr = np.maximum(0, b - r_arr)
    b_mr1 = np.maximum(0, b - r_arr + 1)
    gr_sub = ((px[b] - px[b_m1]) + (px[b] - px[b_mr1]) + (px[b] - px[b_mr])) / (2.0 * r_arr.astype(np.float64))
    gr_all = sum_g_all / n_f
    gradient_recent = np.where(n > r_arr, gr_sub, gr_all)
    gradient_recent = np.where(np.abs(gradient_recent) <= 1e-12, 0.0, gradient_recent)

    gc_legacy = np.where(valid, 0.5 * (slope_full + gradient_recent), 0.0)
    tol_raw = 1e-10 * np.maximum(1.0, np.abs(px[b]))
    gc_legacy = np.where(np.abs(gc_legacy) <= tol_raw, 0.0, gc_legacy)

    Q2 = np.cumsum(k_idx * k_idx * dy_ref)
    a0 = _range_sum(P0, a, b_m1)
    a1 = _range_sum(P1, a, b_m1)
    a2 = _range_sum(Q2, a, b_m1)
    c_f = a.astype(np.float64)

    # p=1:
    total_p1 = m_f * dy_ref[b] - a0
    s1_w = 0.5 * m_f * (m_f + 1.0)
    gc_sw1 = np.where(valid, total_p1 / s1_w, 0.0)
    gc_sw1 = np.where(np.abs(gc_sw1) <= tol_raw, 0.0, gc_sw1)

    # p=2:
    sum_2r1 = 2.0 * a1 - (2.0 * c_f - 1.0) * a0
    total_p2 = (m_f * m_f) * dy_ref[b] - sum_2r1
    s2_w = (m_f * (m_f + 1.0) * (2.0 * m_f + 1.0)) / 6.0
    gc_sw2 = np.where(valid, total_p2 / s2_w, 0.0)
    gc_sw2 = np.where(np.abs(gc_sw2) <= tol_raw, 0.0, gc_sw2)

    # p=3:
    sum_r2 = a2 - 2.0 * c_f * a1 + c_f * c_f * a0
    sum_r1 = a1 - c_f * a0
    total_p3 = (m_f ** 3) * dy_ref[b] - (3.0 * sum_r2 + 3.0 * sum_r1 + a0)
    s3_w = (0.5 * m_f * (m_f + 1.0)) ** 2
    gc_sw3 = np.where(valid, total_p3 / s3_w, 0.0)
    gc_sw3 = np.where(np.abs(gc_sw3) <= tol_raw, 0.0, gc_sw3)

    sym_p = pw.step_wols_power_for_symbol(symbol)
    gc_diff = gc_sw3 if sym_p >= 2.5 else gc_sw2

    # Epsilon (population std of np.gradient(px[a:b+1]))
    pg2 = np.zeros(N, dtype=np.float64)
    if N >= 3:
        g_int = 0.5 * (px[2:] - px[:-2])
        cs_g2 = np.cumsum(g_int * g_int)
        pg2[1:-1] = cs_g2
        pg2[-1] = cs_g2[-1]
    g0 = px[a_p1] - px[a]
    gn = px[b] - px[b_m1]
    sum_g2 = g0 * g0 + (pg2[b_m1] - pg2[a]) + gn * gn
    mean_g = sum_g_all / n_f
    var_g = np.maximum(0.0, sum_g2 / n_f - mean_g * mean_g)
    eps = np.where(n >= 3, np.sqrt(var_g), 0.0)

    scale_inv = np.where(m_f > 10.0, np.sqrt(10.0 / m_f), 1.0)

    return {
        "n": n,
        "m": m,
        "eps": eps,
        "scale_inv": scale_inv,
        "legacy": gc_legacy,
        "sw1": gc_sw1,
        "sw2": gc_sw2,
        "sw3": gc_sw3,
        "diff": gc_diff,
    }


def make_tape(
    gc: np.ndarray,
    eps: np.ndarray,
    k_eps: float = 0.0,
    scale_inv: np.ndarray | None = None,
) -> tuple[bool, ...]:
    cond = gc > 0.0
    if k_eps > 0.0:
        eff_k = k_eps * scale_inv if scale_inv is not None else k_eps
        cond = np.where(eps > 0.0, gc > (eff_k * eps), cond)
    cond[:1] = False
    return cond.tolist()


def evaluate_dataset(use_365: bool = False):
    label = "DENSE 1-SECOND ARCHIVE (cache_365_price_*.jsonl)" if use_365 else "FULL 329-DAY SPARSE ARCHIVE (cache_sparse_price_*.jsonl)"
    print(f"\n{'=' * 105}")
    print(f"  PORTFOLIO REPLAY BACKTEST: {label}")
    print(f"{'=' * 105}")

    all_symbols = list(rrb.SYMBOLS._SECTIONS.keys())
    symbols = []
    series_map = {}
    sig_map = {}

    for sym in all_symbols:
        t0 = time.time()
        npz_path = cache_path(f"{sym}.npz")
        if use_365 and os.path.exists(npz_path):
            z = np.load(npz_path)
            series = list(zip(z["ts"].tolist(), z["px"].tolist()))
        else:
            fname = f"cache_365_price_{sym}.jsonl" if use_365 else f"cache_sparse_price_{sym}.jsonl"
            path = cache_path(fname)
            if not os.path.exists(path):
                path = cache_path(f"cache_sparse_price_{sym}.jsonl")
            series = load_price_series(path, sym)
        if not series:
            continue
        symbols.append(sym)
        series_map[sym] = series
        sig_map[sym] = compute_all_signals(series, symbol=sym)
        dt = time.time() - t0
        print(f"  [loaded & vectorized] {sym:<8} (p={pw.step_wols_power_for_symbol(sym):.1f}): "
              f"{len(series):>9,} ticks in {dt:.2f}s (median m={int(np.median(sig_map[sym]['m']))} steps)")

    models = [
        ("1. LEGACY (sf+gr)/2 (no NG)",       lambda s: make_tape(s["legacy"], s["eps"], k_eps=0.0)),
        ("2. LEGACY + NG k=0.25",             lambda s: make_tape(s["legacy"], s["eps"], k_eps=0.25)),
        ("3. SW p2 + NG k=0.25",              lambda s: make_tape(s["sw2"], s["eps"], k_eps=0.25)),
        ("4. SW p3 + NG k=0.25",              lambda s: make_tape(s["sw3"], s["eps"], k_eps=0.25)),
        ("5. SW p2 + ScaleInv NG k=0.25",     lambda s: make_tape(s["sw2"], s["eps"], k_eps=0.25, scale_inv=s["scale_inv"])),
        ("6. SW p3 + ScaleInv NG k=0.25",     lambda s: make_tape(s["sw3"], s["eps"], k_eps=0.25, scale_inv=s["scale_inv"])),
        ("7. DIFFERENTIATED (BTC p3/ALT p2)", lambda s: make_tape(s["diff"], s["eps"], k_eps=0.25, scale_inv=s["scale_inv"])),
    ]

    hdr = f"{'Model Architecture':<34} | " + " | ".join(f"{sym:>14}" for sym in symbols) + f" | {'PORTFOLIO NET':>14}"
    print("\n" + hdr)
    print("-" * len(hdr))

    bh_Wait = {}
    for m_name, tape_fn in models:
        rrb._TREND_TAPE_CACHE.clear()
        cells = []
        tot_net = 0.0
        for sym in symbols:
            series = series_map[sym]
            key = rrb._series_cache_key(sym, series, rrb.DEFAULT_WINDOW_SECONDS)
            rrb._TREND_TAPE_CACHE[key] = tape_fn(sig_map[sym])
            cfg = rrb.SYMBOLS[sym]
            res = rrb.run_symbol(sym, cfg["params"], cfg["base"], quiet=True, series=series)
            net = res["net"]
            bh_Wait[sym] = res["buy_hold"]
            tot_net += net
            cells.append(f"{net:>+8.2f} ({res['buys']:>2}B/{res['sells']:>2}S)")
        print(f"{m_name:<34} | " + " | ".join(f"{c:>14}" for c in cells) + f" | {tot_net:>+14.2f}")

    bh_tot = sum(bh_Wait.values())
    bh_cells = [f"{bh_Wait[sym]:>+14.2f}" for sym in symbols]
    print("-" * len(hdr))
    print(f"{'Buy & Hold Benchmark':<34} | " + " | ".join(bh_cells) + f" | {bh_tot:>+14.2f}")


if __name__ == "__main__":
    evaluate_dataset(use_365=False)
    if "--dense" in sys.argv:
        evaluate_dataset(use_365=True)
