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

import os
import sys
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

from botcore import safe_float, safe_int, safe_pos_float
import pricewindow as pw
from cachepaths import cache_path
from providers.replay_provider import load_price_series
import offline.research.monitortrades_backtest.run_replay_backtest as rrb


def compute_window_bounds(ts: np.ndarray, win_sec: float = rrb.DEFAULT_WINDOW_SECONDS):
    """Replicate exact cur_len, sample_rate, and [a, b] bounds of prewarm_trend_series."""
    w_sec = safe_pos_float(win_sec, rrb.DEFAULT_WINDOW_SECONDS)
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
            rate = safe_pos_float(t - prev_ts, default=None)
            if rate is not None:
                sample_rate = rate
                new_size = min(100000, max(10, int(w_sec / rate)))
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


def compute_window_signals_from_bounds(
    px: np.ndarray,
    a: np.ndarray,
    sr: np.ndarray,
    *,
    symbol: str | None = None,
    recent_win_sec: float | None = None,
    min_valid_n: int = 2,
) -> dict[str, np.ndarray]:
    """Vectorized chunked O(1) prefix-sum evaluation of short-trend estimators given window bounds [a, b]."""
    N = len(px)
    if N == 0:
        empty_i = np.zeros(0, dtype=np.int64)
        empty_f = np.zeros(0, dtype=np.float64)
        return {
            "n": empty_i,
            "m": empty_i,
            "eps": empty_f,
            "scale_inv": empty_f,
            "gr": empty_f,
            "sf": empty_f,
            "legacy_raw": empty_f,
            "legacy": empty_f,
            "sw1": empty_f,
            "sw2": empty_f,
            "sw3": empty_f,
            "diff": empty_f,
            "gc": empty_f,
        }

    b = np.arange(N, dtype=np.int64)
    n = b - a + 1
    m = b - a
    valid = n >= min_valid_n
    n_f = np.maximum(1.0, n.astype(np.float64))
    m_f = np.maximum(1.0, m.astype(np.float64))

    recent_sec = pw.effective_recent_gradient_seconds(recent_win_sec)
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

    g_int = np.zeros(N, dtype=np.float64)
    if N >= 3:
        g_int[1:-1] = 0.5 * (px[2:] - px[:-2])

    slope_full = np.zeros(N, dtype=np.float64)
    gc_sw1 = np.zeros(N, dtype=np.float64)
    gc_sw2 = np.zeros(N, dtype=np.float64)
    gc_sw3 = np.zeros(N, dtype=np.float64)
    eps = np.zeros(N, dtype=np.float64)

    # Chunked prefix sums prevent float64 precision loss when k^2 * dy exceeds 2^53 on long series
    CHUNK = 4096
    for hi_start in range(0, N, CHUNK):
        hi_end = min(N, hi_start + CHUNK)
        lo_start = int(np.min(a[hi_start:hi_end]))
        px_sub = px[lo_start:hi_end]
        g_sub = g_int[lo_start:hi_end]
        L = len(px_sub)

        y_ref = px_sub[0]
        dy = px_sub - y_ref
        k_loc = np.arange(L, dtype=np.float64)

        P0 = np.cumsum(dy)
        P1 = np.cumsum(k_loc * dy)
        P2 = np.cumsum(dy * dy)
        Q2 = np.cumsum(k_loc * k_loc * dy)
        PG2 = np.cumsum(g_sub * g_sub)

        def _rsum(P, lo, hi):
            prev = np.where(lo > 0, P[np.maximum(0, lo - 1)], 0.0)
            return P[hi] - prev

        a_loc = a[hi_start:hi_end] - lo_start
        b_loc = b[hi_start:hi_end] - lo_start
        bm1_loc = np.maximum(0, b_loc - 1)
        ap1_loc = np.minimum(L - 1, a_loc + 1)
        m_sub = m_f[hi_start:hi_end]
        n_sub = n_f[hi_start:hi_end]
        val_sub = valid[hi_start:hi_end]

        s0 = _rsum(P0, a_loc, b_loc)
        s1 = _rsum(P1, a_loc, b_loc)
        s2 = _rsum(P2, a_loc, b_loc)
        ss_y = s2 - (s0 * s0) / n_sub
        cov_xy = s1 - (a_loc.astype(np.float64) + 0.5 * (n_sub - 1.0)) * s0
        denom_ols = np.maximum(1.0, n_sub * (n_sub * n_sub - 1.0))
        slope_full[hi_start:hi_end] = np.where(val_sub & (ss_y > 1e-18), (12.0 * cov_xy) / denom_ols, 0.0)

        a0 = _rsum(P0, a_loc, bm1_loc)
        a1 = _rsum(P1, a_loc, bm1_loc)
        a2 = _rsum(Q2, a_loc, bm1_loc)
        c_f = a_loc.astype(np.float64)
        dy_b = dy[b_loc]

        # p=1:
        total_p1 = m_sub * dy_b - a0
        s1_w = 0.5 * m_sub * (m_sub + 1.0)
        gc_sw1[hi_start:hi_end] = np.where(val_sub, total_p1 / s1_w, 0.0)

        # p=2:
        sum_2r1 = 2.0 * a1 - (2.0 * c_f - 1.0) * a0
        total_p2 = (m_sub * m_sub) * dy_b - sum_2r1
        s2_w = (m_sub * (m_sub + 1.0) * (2.0 * m_sub + 1.0)) / 6.0
        gc_sw2[hi_start:hi_end] = np.where(val_sub, total_p2 / s2_w, 0.0)

        # p=3:
        sum_r2 = a2 - 2.0 * c_f * a1 + c_f * c_f * a0
        sum_r1 = a1 - c_f * a0
        total_p3 = (m_sub ** 3) * dy_b - (3.0 * sum_r2 + 3.0 * sum_r1 + a0)
        s3_w = (0.5 * m_sub * (m_sub + 1.0)) ** 2
        gc_sw3[hi_start:hi_end] = np.where(val_sub, total_p3 / s3_w, 0.0)

        # Epsilon (population std of np.gradient(px[a:b+1])):
        g0 = px_sub[ap1_loc] - px_sub[a_loc]
        gn = px_sub[b_loc] - px_sub[bm1_loc]
        sum_g2 = g0 * g0 + (PG2[bm1_loc] - PG2[a_loc]) + gn * gn
        mean_g = sum_g_all[hi_start:hi_end] / n_sub
        var_g = np.maximum(0.0, sum_g2 / n_sub - mean_g * mean_g)
        eps[hi_start:hi_end] = np.where(val_sub & (n_sub >= 3), np.sqrt(var_g), 0.0)

    tol_raw = 1e-10 * np.maximum(1.0, np.abs(px[b]))
    legacy_raw = 0.5 * (slope_full + gradient_recent)
    gc_legacy = np.where(valid, legacy_raw, 0.0)
    gc_legacy = np.where(np.abs(gc_legacy) <= tol_raw, 0.0, gc_legacy)
    gc_sw1 = np.where(np.abs(gc_sw1) <= tol_raw, 0.0, gc_sw1)
    gc_sw2 = np.where(np.abs(gc_sw2) <= tol_raw, 0.0, gc_sw2)
    gc_sw3 = np.where(np.abs(gc_sw3) <= tol_raw, 0.0, gc_sw3)

    sym_p = pw.step_wols_power_for_symbol(symbol)
    gc_diff = gc_sw3 if sym_p >= 2.5 else gc_sw2

    scale_inv = np.where(m_f > 10.0, np.sqrt(10.0 / m_f), 1.0)

    return {
        "n": n,
        "m": m,
        "eps": eps,
        "scale_inv": scale_inv,
        "gr": gradient_recent,
        "sf": slope_full,
        "legacy_raw": legacy_raw,
        "legacy": gc_legacy,
        "sw1": gc_sw1,
        "sw2": gc_sw2,
        "sw3": gc_sw3,
        "diff": gc_diff,
        "gc": gc_diff,
    }


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
    return compute_window_signals_from_bounds(
        px, a, sr, symbol=symbol, recent_win_sec=None, min_valid_n=2
    )


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


def load_archive_series(sym: str, use_365: bool = False) -> list[tuple[float, float]]:
    """Load dense 1s (.npz / .jsonl) or sparse 329d price series for `sym`."""
    npz_path = cache_path(f"cache_365_price_{sym}.npz")
    if use_365 and os.path.exists(npz_path):
        with np.load(npz_path) as z:
            return list(zip(z["ts"].tolist(), z["px"].tolist()))
    fname = f"cache_365_price_{sym}.jsonl" if use_365 else f"cache_sparse_price_{sym}.jsonl"
    path = cache_path(fname)
    if not os.path.exists(path):
        path = cache_path(f"cache_sparse_price_{sym}.jsonl")
    if not os.path.exists(path):
        return []
    return load_price_series(path, sym)


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
        series = load_archive_series(sym, use_365=use_365)
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
            res = rrb.run_symbol(sym, cfg["params"], cfg["base"], quiet=True, series=series) or {}
            net = safe_float(res.get("net"), 0.0)
            bh_Wait[sym] = safe_float(res.get("buy_hold"), 0.0)
            tot_net += net
            cells.append(f"{net:>+8.2f} ({safe_int(res.get('buys'), 0):>2}B/{safe_int(res.get('sells'), 0):>2}S)")
        print(f"{m_name:<34} | " + " | ".join(f"{c:>14}" for c in cells) + f" | {tot_net:>+14.2f}")

    bh_tot = sum(bh_Wait.values())
    bh_cells = [f"{bh_Wait[sym]:>+14.2f}" for sym in symbols]
    print("-" * len(hdr))
    print(f"{'Buy & Hold Benchmark':<34} | " + " | ".join(bh_cells) + f" | {bh_tot:>+14.2f}")


if __name__ == "__main__":
    evaluate_dataset(use_365=False)
    if "--dense" in sys.argv:
        evaluate_dataset(use_365=True)
