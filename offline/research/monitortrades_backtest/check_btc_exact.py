#!/usr/bin/env python3
import os
import sys
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
from offline.research.monitortrades_backtest.sweep_mt_models import compute_window_bounds, make_tape


def compute_exact_signals_chunked(series, win_sec=222.0):
    N = len(series)
    ts = np.fromiter((t for t, _ in series), dtype=np.float64, count=N)
    px = np.fromiter((p for _, p in series), dtype=np.float64, count=N)
    a, sr = compute_window_bounds(ts, win_sec=win_sec)
    b = np.arange(N, dtype=np.int64)
    n = b - a + 1
    m = b - a
    valid = n >= 2
    m_f = np.maximum(1.0, m.astype(np.float64))
    n_f = n.astype(np.float64)
    b_m1 = np.maximum(0, b - 1)
    a_p1 = np.minimum(N - 1, a + 1)

    sum_g_all = (px[b] - px[a]) + 0.5 * ((px[b] - px[b_m1]) + (px[a_p1] - px[a]))
    g_int = np.zeros(N, dtype=np.float64)
    if N >= 3:
        g_int[1:-1] = 0.5 * (px[2:] - px[:-2])

    gc_sw1 = np.zeros(N, dtype=np.float64)
    gc_sw2 = np.zeros(N, dtype=np.float64)
    gc_sw3 = np.zeros(N, dtype=np.float64)
    eps = np.zeros(N, dtype=np.float64)

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

        a0 = _rsum(P0, a_loc, bm1_loc)
        a1 = _rsum(P1, a_loc, bm1_loc)
        a2 = _rsum(Q2, a_loc, bm1_loc)
        c_f = a_loc.astype(np.float64)
        dy_b = dy[b_loc]

        tot1 = m_sub * dy_b - a0
        s1_w = 0.5 * m_sub * (m_sub + 1.0)
        gc_sw1[hi_start:hi_end] = np.where(val_sub, tot1 / s1_w, 0.0)

        sum_2r1 = 2.0 * a1 - (2.0 * c_f - 1.0) * a0
        tot2 = (m_sub * m_sub) * dy_b - sum_2r1
        s2_w = (m_sub * (m_sub + 1.0) * (2.0 * m_sub + 1.0)) / 6.0
        gc_sw2[hi_start:hi_end] = np.where(val_sub, tot2 / s2_w, 0.0)

        sum_r2 = a2 - 2.0 * c_f * a1 + c_f * c_f * a0
        sum_r1 = a1 - c_f * a0
        tot3 = (m_sub ** 3) * dy_b - (3.0 * sum_r2 + 3.0 * sum_r1 + a0)
        s3_w = (0.5 * m_sub * (m_sub + 1.0)) ** 2
        gc_sw3[hi_start:hi_end] = np.where(val_sub, tot3 / s3_w, 0.0)

        g0 = px_sub[ap1_loc] - px_sub[a_loc]
        gn = px_sub[b_loc] - px_sub[bm1_loc]
        int_g2 = PG2[bm1_loc] - PG2[a_loc]
        sum_g2 = g0 * g0 + int_g2 + gn * gn
        mean_g = sum_g_all[hi_start:hi_end] / n_sub
        var_g = np.maximum(0.0, sum_g2 / n_sub - mean_g * mean_g)
        eps[hi_start:hi_end] = np.where(n_sub >= 3, np.sqrt(var_g), 0.0)

    tol_raw = 1e-10 * np.maximum(1.0, np.abs(px[b]))
    gc_sw1 = np.where(np.abs(gc_sw1) <= tol_raw, 0.0, gc_sw1)
    gc_sw2 = np.where(np.abs(gc_sw2) <= tol_raw, 0.0, gc_sw2)
    gc_sw3 = np.where(np.abs(gc_sw3) <= tol_raw, 0.0, gc_sw3)
    scale_inv = np.where(m_f > 10.0, np.sqrt(10.0 / m_f), 1.0)
    return {"sw1": gc_sw1, "sw2": gc_sw2, "sw3": gc_sw3, "eps": eps, "scale_inv": scale_inv, "m": m}


if __name__ == "__main__":
    series = load_price_series(cache_path("cache_sparse_price_BTCUSDC.jsonl"), "BTCUSDC")
    cfg = rrb.SYMBOLS["BTCUSDC"]
    s_ex = compute_exact_signals_chunked(series, win_sec=222.0)
    for k_val in [0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5, 2.0]:
        tape = make_tape(s_ex["sw3"], s_ex["eps"], k_eps=k_val, scale_inv=s_ex["scale_inv"])
        rrb._TREND_TAPE_CACHE.clear()
        key = rrb._series_cache_key("BTCUSDC", series, rrb.DEFAULT_WINDOW_SECONDS)
        rrb._TREND_TAPE_CACHE[key] = tape
        res = rrb.run_symbol("BTCUSDC", cfg["params"], cfg["base"], quiet=True, series=series)
        print(f"EXACT BTC win=222s p=3.0 k={k_val:<4.2f} -> net={res['net']:>+8.2f} ({res['buys']}B/{res['sells']}S) up%={100*sum(tape)/len(tape):.1f}%")
