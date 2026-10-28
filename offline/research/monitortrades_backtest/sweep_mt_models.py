#!/usr/bin/env python3
"""Fast vectorized multi-model portfolio replay backtest for monitortrades.py.

Evaluates Legacy, Step-WOLS (p=2, p=3, med1, med5), Adaptive Noise-Gating (k=0.0..0.25
and scale-invariant z-score gating), Kaufman Efficiency Ratio (ER), and Tiered Hybrid
models across all monitortrades instruments (BTCUSDC, TAOUSDC, ARBUSDC, HYPEUSD) on:
  1. Full 329-day sparse archive (cache_sparse_price_*.jsonl)
  2. Dense 1-second archive (cache_365_price_*.jsonl / *.npz)
"""
from __future__ import annotations

import math
import os
import sys
import time
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

import pricewindow as pw
from cachepaths import cache_path
from providers.replay_provider import load_price_series
import offline.research.monitortrades_backtest.run_replay_backtest as rrb


def _causal_median_5(px: np.ndarray) -> np.ndarray:
    n = len(px)
    out = np.empty(n, dtype=np.float64)
    for i in range(min(4, n)):
        out[i] = float(np.median(px[: i + 1]))
    if n > 4:
        out[4:] = np.median(sliding_window_view(px, 5), axis=-1)
    return out


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


def compute_all_signals(series: list[tuple[float, float]], win_sec: float = rrb.DEFAULT_WINDOW_SECONDS):
    """Vectorized O(1) prefix-sum evaluation of all short-trend estimators over series."""
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

    # Recent gradient
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

    # Step-WOLS med1 (raw prices): p=1, p=2, p=3 via exact Abel summation
    Q0 = P0
    Q1 = P1
    Q2 = np.cumsum(k_idx * k_idx * dy_ref)
    a0 = _range_sum(Q0, a, b_m1)
    a1 = _range_sum(Q1, a, b_m1)
    a2 = _range_sum(Q2, a, b_m1)
    c_f = a.astype(np.float64)

    # p=1: sum_{k=1..m} k (f_{a+k} - f_{a+k-1}) = m f_b - sum_{r=0..m-1} f_{a+r}
    total_p1 = m_f * dy_ref[b] - a0
    s1_w = 0.5 * m_f * (m_f + 1.0)
    gc_sw1_med1 = np.where(valid, total_p1 / s1_w, 0.0)
    gc_sw1_med1 = np.where(np.abs(gc_sw1_med1) <= tol_raw, 0.0, gc_sw1_med1)

    # p=2:
    sum_2r1 = 2.0 * a1 - (2.0 * c_f - 1.0) * a0
    total_p2 = (m_f * m_f) * dy_ref[b] - sum_2r1
    s2_w = (m_f * (m_f + 1.0) * (2.0 * m_f + 1.0)) / 6.0
    gc_sw2_med1 = np.where(valid, total_p2 / s2_w, 0.0)
    gc_sw2_med1 = np.where(np.abs(gc_sw2_med1) <= tol_raw, 0.0, gc_sw2_med1)

    # p=3:
    sum_r2 = a2 - 2.0 * c_f * a1 + c_f * c_f * a0
    sum_r1 = a1 - c_f * a0
    total_p3 = (m_f ** 3) * dy_ref[b] - (3.0 * sum_r2 + 3.0 * sum_r1 + a0)
    s3_w = (0.5 * m_f * (m_f + 1.0)) ** 2
    gc_sw3_med1 = np.where(valid, total_p3 / s3_w, 0.0)
    gc_sw3_med1 = np.where(np.abs(gc_sw3_med1) <= tol_raw, 0.0, gc_sw3_med1)

    # Step-WOLS p=3 med5 (with slice-local causal median head adjustment)
    med5 = _causal_median_5(px)
    z5 = med5 - y_ref
    Q0_5 = np.cumsum(z5)
    Q1_5 = np.cumsum(k_idx * z5)
    Q2_5 = np.cumsum(k_idx * k_idx * z5)
    a0_5 = _range_sum(Q0_5, a, b_m1)
    a1_5 = _range_sum(Q1_5, a, b_m1)
    a2_5 = _range_sum(Q2_5, a, b_m1)
    sum_r2_5 = a2_5 - 2.0 * c_f * a1_5 + c_f * c_f * a0_5
    sum_r1_5 = a1_5 - c_f * a0_5
    total_p3_5 = (m_f ** 3) * z5[b] - (3.0 * sum_r2_5 + 3.0 * sum_r1_5 + a0_5)
    # Local head adjustment for r = 0, 1, 2, 3 when kernel=5 (sr <= 1.25s) or kernel=3
    # For general sr, kernel is determined by pw.median_filter_kernel(sr, 5.0):
    # r=0: local_head[0] = px[a]
    # r=1: local_head[1] = 0.5 * (px[a] + px[a+1])
    # r=2: local_head[2] = median(px[a], px[a+1], px[a+2])
    # r=3: local_head[3] = median(px[a..a+3])
    a_1 = np.minimum(N - 1, a + 1)
    a_2 = np.minimum(N - 1, a + 2)
    a_3 = np.minimum(N - 1, a + 3)
    lh0 = px[a]
    lh1 = 0.5 * (px[a] + px[a_1])
    lh2 = np.median(np.stack([px[a], px[a_1], px[a_2]], axis=0), axis=0)
    lh3 = np.median(np.stack([px[a], px[a_1], px[a_2], px[a_3]], axis=0), axis=0)
    for r_idx, lh, a_r in ((0, lh0, a), (1, lh1, a_1), (2, lh2, a_2), (3, lh3, a_3)):
        active_r = m >= r_idx
        coeff = 3.0 * (r_idx ** 2) + 3.0 * r_idx + 1.0
        total_p3_5 = np.where(active_r, total_p3_5 - coeff * (lh - med5[a_r]), total_p3_5)
    tol_med5 = 1e-10 * np.maximum(1.0, np.abs(med5[b]))
    gc_sw3_med5 = np.where(valid, total_p3_5 / s3_w, 0.0)
    gc_sw3_med5 = np.where(np.abs(gc_sw3_med5) <= tol_med5, 0.0, gc_sw3_med5)
    # Note: when sample_rate > 2.5s (as in sparse data), median_filter_kernel(sr, 5.0) == 1!
    kern_arr = np.where(sr >= 2.5, 1, 5)
    gc_sw3_med5_adaptive = np.where(kern_arr == 1, gc_sw3_med1, gc_sw3_med5)

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

    # Kaufman Efficiency Ratio (ER) on [a, b]
    abs_dy = np.abs(np.diff(px, prepend=px[0]))
    P_abs = np.cumsum(abs_dy)
    path_len = P_abs[b] - P_abs[a]
    net_chg = np.abs(px[b] - px[a])
    er = np.where(valid, net_chg / np.maximum(path_len, 1e-12), 0.0)

    # Exact L2 weight norm ||w||_2 for scale-invariant z-score gating:
    # For Step-WOLS power p on m steps, ||w||_2 = sqrt(sum(k^(2p))) / sum(k^p) ~= sqrt(2p+1) / ((p+1)*sqrt(m))
    w_norm_p2 = np.sqrt(5.0) / (3.0 * np.sqrt(m_f))
    w_norm_p3 = np.sqrt(7.0) / (4.0 * np.sqrt(m_f))

    return {
        "n": n,
        "m": m,
        "eps": eps,
        "er": er,
        "w_norm_p2": w_norm_p2,
        "w_norm_p3": w_norm_p3,
        "legacy": gc_legacy,
        "sw1_med1": gc_sw1_med1,
        "sw2_med1": gc_sw2_med1,
        "sw3_med1": gc_sw3_med1,
        "sw3_med5": gc_sw3_med5,
        "sw3_med5_adaptive": gc_sw3_med5_adaptive,
    }


def make_tape(gc: np.ndarray, eps: np.ndarray, k_eps: float = 0.0,
              er: np.ndarray | None = None, er_min: float = 0.0,
              w_norm: np.ndarray | None = None, z_min: float = 0.0) -> tuple[bool, ...]:
    cond = gc > 0.0
    if k_eps > 0.0:
        cond = np.where(eps > 0.0, gc > (k_eps * eps), cond)
    if z_min > 0.0 and w_norm is not None:
        se = eps * w_norm
        cond = np.where(se > 0.0, gc > (z_min * se), cond)
    if er_min > 0.0 and er is not None:
        cond = cond & (er >= er_min)
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
        sig_map[sym] = compute_all_signals(series)
        dt = time.time() - t0
        print(f"  [loaded & vectorized] {sym:<8}: {len(series):>9,} ticks in {dt:.2f}s (median m={int(np.median(sig_map[sym]['m']))} steps)")

    models = [
        ("1. LEGACY (sf+gr)/2 (no NG)",       lambda s: make_tape(s["legacy"], s["eps"], k_eps=0.0)),
        ("2. LEGACY + NG k=0.25",             lambda s: make_tape(s["legacy"], s["eps"], k_eps=0.25)),
        ("3. SW p3 med5 (always 5-sample)",   lambda s: make_tape(s["sw3_med5"], s["eps"], k_eps=0.0)),
        ("4. SW p3 med1 (no NG)",             lambda s: make_tape(s["sw3_med1"], s["eps"], k_eps=0.0)),
        ("5. SW p2 med1 + NG k=0.25 (curr)",  lambda s: make_tape(s["sw2_med1"], s["eps"], k_eps=0.25)),
        ("6. SW p3 med1 + NG k=0.25",         lambda s: make_tape(s["sw3_med1"], s["eps"], k_eps=0.25)),
        ("7. SW p3 med1 + NG k=0.05",         lambda s: make_tape(s["sw3_med1"], s["eps"], k_eps=0.05)),
        ("8. SW p3 med1 + NG k=0.04",         lambda s: make_tape(s["sw3_med1"], s["eps"], k_eps=0.04)),
        ("9. SW p3 med1 + ScaleInv z=1.0",    lambda s: make_tape(s["sw3_med1"], s["eps"], w_norm=s["w_norm_p3"], z_min=1.0)),
        ("10. SW p2 med1 + ScaleInv z=1.0",   lambda s: make_tape(s["sw2_med1"], s["eps"], w_norm=s["w_norm_p2"], z_min=1.0)),
        ("11. HYBRID p3 ER0.05 + NG k=0.05",  lambda s: make_tape(s["sw3_med1"], s["eps"], k_eps=0.05, er=s["er"], er_min=0.05)),
        ("12. HYBRID p3 ER0.05 + z=1.0",      lambda s: make_tape(s["sw3_med1"], s["eps"], w_norm=s["w_norm_p3"], z_min=1.0, er=s["er"], er_min=0.05)),
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
            tape = tape_fn(sig_map[sym])
            key = rrb._series_cache_key(sym, series, rrb.DEFAULT_WINDOW_SECONDS)
            rrb._TREND_TAPE_CACHE[key] = tape
            info = rrb.SYMBOLS[sym]
            res = rrb.run_symbol(sym, info["params"], info["base"], quiet=True, series=series)
            net = res["net"] if res else 0.0
            buys = res["buys"] if res else 0
            sells = res["sells"] if res else 0
            if sym not in bh_Wait and res:
                bh_Wait[sym] = res["buy_hold"]
            tot_net += net
            cells.append(f"{net:>+8.2f} ({buys}/{sells})")
        print(f"{m_name:<34} | " + " | ".join(f"{c:>14}" for c in cells) + f" | {tot_net:>+14.2f}")
        sys.stdout.flush()

    bh_tot = sum(bh_Wait.values())
    bh_cells = [f"{bh_Wait.get(s, 0.0):>+8.2f} (B&H)" for s in symbols]
    print("-" * len(hdr))
    print(f"{'Buy & Hold Benchmark ($1000/sym)':<34} | " + " | ".join(f"{c:>14}" for c in bh_cells) + f" | {bh_tot:>+14.2f}")


if __name__ == "__main__":
    evaluate_dataset(use_365=False)
    if os.environ.get("SWEEP_365") == "1":
        evaluate_dataset(use_365=True)
