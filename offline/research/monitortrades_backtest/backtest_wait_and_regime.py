#!/usr/bin/env python3
"""Backtest 2 & 3:
- Backtest 2: Opportunistic Execution Delay Gate (`should_wait` / `GT_S`) across
  1-second dense and 329-day archives (BTCUSDC, TAOUSDC, ARBUSDC, HYPEUSD).
- Backtest 3: 900s Dynamic Window Regime Filter (`rtrade.py` / `MarketRegimeEvaluator`)
  comparing unscaled vs scale-invariant `strength` on 15-minute windows.
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
from offline.research.monitortrades_backtest.sweep_mt_models import compute_all_signals


def compute_exact_time_window_signals(
    ts: np.ndarray,
    px: np.ndarray,
    win_sec: float,
    symbol: str,
):
    """Exact O(N) sliding two-pointer + O(1) prefix-sum evaluation for dynamic window `win_sec`."""
    N = len(ts)
    # Two-pointer to find a[b] = first index with ts[a] >= ts[b] - win_sec
    a = np.searchsorted(ts, ts - win_sec, side="left")
    b = np.arange(N, dtype=np.int64)
    n = b - a + 1
    m = b - a
    valid = n >= 3

    y_ref = px[0]
    dy_ref = px - y_ref
    k_idx = np.arange(N, dtype=np.float64)

    P0 = np.cumsum(dy_ref)
    P1 = np.cumsum(k_idx * dy_ref)
    P2 = np.cumsum(dy_ref * dy_ref)
    Q2 = np.cumsum(k_idx * k_idx * dy_ref)

    def _range_sum(P, lo, hi):
        prev = np.where(lo > 0, P[np.maximum(0, lo - 1)], 0.0)
        return P[hi] - prev

    n_f = np.maximum(1.0, n.astype(np.float64))
    m_f = np.maximum(1.0, m.astype(np.float64))
    b_m1 = np.maximum(0, b - 1)
    a_p1 = np.minimum(N - 1, a + 1)

    s0 = _range_sum(P0, a, b)
    s1 = _range_sum(P1, a, b)
    s2 = _range_sum(P2, a, b)
    ss_y = s2 - (s0 * s0) / n_f
    cov_xy = s1 - (a.astype(np.float64) + 0.5 * (n_f - 1.0)) * s0
    denom_ols = np.maximum(1.0, n_f * (n_f * n_f - 1.0))
    slope_full = np.where(valid & (ss_y > 1e-18), (12.0 * cov_xy) / denom_ols, 0.0)

    # Estimate sample rate per slice as elapsed / m
    elapsed = np.maximum(1.0, ts[b] - ts[a])
    sr = np.maximum(0.5, elapsed / m_f)
    recent_sec = pw.effective_recent_gradient_seconds(win_sec)
    r_arr = np.maximum(2, (recent_sec / sr).astype(np.int64))

    sum_g_all = (px[b] - px[a]) + 0.5 * ((px[b] - px[b_m1]) + (px[a_p1] - px[a]))
    b_mr = np.maximum(0, b - r_arr)
    b_mr1 = np.maximum(0, b - r_arr + 1)
    gr_sub = ((px[b] - px[b_m1]) + (px[b] - px[b_mr1]) + (px[b] - px[b_mr])) / (2.0 * r_arr.astype(np.float64))
    gr_all = sum_g_all / n_f
    gradient_recent = np.where(n > r_arr, gr_sub, gr_all)
    gradient_recent = np.where(np.abs(gradient_recent) <= 1e-12, 0.0, gradient_recent)

    a0 = _range_sum(P0, a, b_m1)
    a1 = _range_sum(P1, a, b_m1)
    a2 = _range_sum(Q2, a, b_m1)
    c_f = a.astype(np.float64)
    tol_raw = 1e-10 * np.maximum(1.0, np.abs(px[b]))

    sym_p = pw.step_wols_power_for_symbol(symbol)
    if sym_p >= 2.5:
        sum_r2 = a2 - 2.0 * c_f * a1 + c_f * c_f * a0
        sum_r1 = a1 - c_f * a0
        total_p3 = (m_f ** 3) * dy_ref[b] - (3.0 * sum_r2 + 3.0 * sum_r1 + a0)
        s3_w = (0.5 * m_f * (m_f + 1.0)) ** 2
        gc = np.where(valid, total_p3 / s3_w, 0.0)
    else:
        sum_2r1 = 2.0 * a1 - (2.0 * c_f - 1.0) * a0
        total_p2 = (m_f * m_f) * dy_ref[b] - sum_2r1
        s2_w = (m_f * (m_f + 1.0) * (2.0 * m_f + 1.0)) / 6.0
        gc = np.where(valid, total_p2 / s2_w, 0.0)
    gc = np.where(np.abs(gc) <= tol_raw, 0.0, gc)

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
    eps = np.where(valid, np.sqrt(var_g), 0.0)
    scale_inv = np.where(m_f > 10.0, np.sqrt(10.0 / m_f), 1.0)

    return {
        "n": n,
        "m": m,
        "gc": gc,
        "gr": gradient_recent,
        "sf": slope_full,
        "legacy": 0.5 * (slope_full + gradient_recent),
        "eps": eps,
        "scale_inv": scale_inv,
    }


def simulate_should_wait(
    ts: np.ndarray,
    px: np.ndarray,
    wait_buy_mask: np.ndarray,
    wait_sell_mask: np.ndarray,
    max_wait_sec: float = 30.0,
    stride: int = 300,
):
    """Evaluate average execution improvement (in bps) over sampled intent arrival points."""
    N = len(ts)
    sample_idx = np.arange(1000, N - 1000, stride, dtype=np.int64)
    buy_bps = np.zeros(len(sample_idx), dtype=np.float64)
    sell_bps = np.zeros(len(sample_idx), dtype=np.float64)
    buy_waited_s = np.zeros(len(sample_idx), dtype=np.float64)
    sell_waited_s = np.zeros(len(sample_idx), dtype=np.float64)

    for k, idx in enumerate(sample_idx):
        t0 = ts[idx]
        p0 = px[idx]
        deadline = t0 + max_wait_sec

        # BUY: wait while wait_buy_mask[j] is True and ts[j] < deadline
        j = idx
        while j + 1 < N and ts[j] < deadline and wait_buy_mask[j]:
            j += 1
        buy_bps[k] = ((p0 - px[j]) / p0) * 10000.0
        buy_waited_s[k] = ts[j] - t0

        # SELL: wait while wait_sell_mask[j] is True and ts[j] < deadline
        j = idx
        while j + 1 < N and ts[j] < deadline and wait_sell_mask[j]:
            j += 1
        sell_bps[k] = ((px[j] - p0) / p0) * 10000.0
        sell_waited_s[k] = ts[j] - t0

    return {
        "buy_bps": float(np.mean(buy_bps)),
        "sell_bps": float(np.mean(sell_bps)),
        "roundtrip_bps": float(np.mean(buy_bps) + np.mean(sell_bps)),
        "buy_wait_pct": float(np.mean(wait_buy_mask[sample_idx]) * 100.0),
        "sell_wait_pct": float(np.mean(wait_sell_mask[sample_idx]) * 100.0),
        "avg_wait_s": float(0.5 * (np.mean(buy_waited_s) + np.mean(sell_waited_s))),
    }


def evaluate_regime_900s(
    ts: np.ndarray,
    px: np.ndarray,
    sig900: dict,
    horizon_sec: float = 900.0,
    stride: int = 300,
):
    """Evaluate 900s regime filter for rtrade: sideways % vs forward 15m adverse excursion & spread fill."""
    N = len(ts)
    sample_idx = np.arange(2000, N - 2000, stride, dtype=np.int64)
    # Forward 15m endpoint for each sample_idx
    fwd_end = np.searchsorted(ts, ts[sample_idx] + horizon_sec, side="left")
    fwd_end = np.minimum(N - 1, fwd_end)

    # Calculate forward 15m absolute move (%) and whether a symmetric +/-0.32% spread fills both sides
    fwd_abs_move_pct = np.zeros(len(sample_idx), dtype=np.float64)
    fwd_max_adverse_pct = np.zeros(len(sample_idx), dtype=np.float64)
    both_sides_filled = np.zeros(len(sample_idx), dtype=bool)
    half_spread = 0.0032  # 0.32% each side (RTRADE_DEFAULT_ADJUSTMENT_PCT = 0.64%)

    for k, (i0, i1) in enumerate(zip(sample_idx, fwd_end)):
        if i1 <= i0:
            continue
        p0 = px[i0]
        window_px = px[i0 : i1 + 1]
        p_min = float(np.min(window_px))
        p_max = float(np.max(window_px))
        p_end = float(px[i1])
        fwd_abs_move_pct[k] = abs(p_end - p0) / p0 * 100.0
        fwd_max_adverse_pct[k] = max(abs(p_max - p0), abs(p0 - p_min)) / p0 * 100.0
        both_sides_filled[k] = (p_min <= p0 * (1.0 - half_spread)) and (p_max >= p0 * (1.0 + half_spread))

    gc = sig900["gc"][sample_idx]
    eps = sig900["eps"][sample_idx]
    scale_inv = sig900["scale_inv"][sample_idx]
    raw_strength = np.divide(np.abs(gc), eps, out=np.zeros_like(gc), where=eps > 0)
    scaled_denom = eps * scale_inv
    scaled_strength = np.divide(np.abs(gc), scaled_denom, out=np.zeros_like(gc), where=scaled_denom > 0)

    results = []
    configs = [
        ("1. Unscaled |gc|/eps <= 2.0 (Current)", raw_strength <= 2.0),
        ("2. Unscaled |gc|/eps <= 0.5",           raw_strength <= 0.5),
        ("3. Scaled z-strength <= 3.0",           scaled_strength <= 3.0),
        ("4. Scaled z-strength <= 2.0 (K=2.0)",   scaled_strength <= 2.0),
        ("5. Scaled z-strength <= 1.5 (K=1.5)",   scaled_strength <= 1.5),
        ("6. Scaled z-strength <= 1.0 (K=1.0)",   scaled_strength <= 1.0),
    ]
    for name, is_sideways in configs:
        sw_pct = float(np.mean(is_sideways) * 100.0)
        if np.any(is_sideways) and np.any(~is_sideways):
            move_sw = float(np.mean(fwd_abs_move_pct[is_sideways]))
            move_tr = float(np.mean(fwd_abs_move_pct[~is_sideways]))
            adv_sw = float(np.mean(fwd_max_adverse_pct[is_sideways]))
            adv_tr = float(np.mean(fwd_max_adverse_pct[~is_sideways]))
            fill_sw = float(np.mean(both_sides_filled[is_sideways]) * 100.0)
        else:
            move_sw = float(np.mean(fwd_abs_move_pct))
            move_tr = 0.0
            adv_sw = float(np.mean(fwd_max_adverse_pct))
            adv_tr = 0.0
            fill_sw = float(np.mean(both_sides_filled) * 100.0)
        results.append((name, sw_pct, move_sw, move_tr, adv_sw, adv_tr, fill_sw))
    return results, raw_strength, scaled_strength


def main():
    symbols = ["BTCUSDC", "TAOUSDC", "ARBUSDC", "HYPEUSD"]
    print("=" * 110)
    print("  BACKTEST 2: OPPORTUNISTIC EXECUTION DELAY GATE (should_wait / GT_S) ON 222s PRIMARY WINDOW")
    print("=" * 110)

    for sym in symbols:
        npz_path = cache_path(f"{sym}.npz")
        if os.path.exists(npz_path):
            z = np.load(npz_path)
            ts = z["ts"].astype(np.float64)
            px = z["px"].astype(np.float64)
            src_label = "1s-dense"
            stride = max(60, len(ts) // 25000)
        else:
            path = cache_path(f"cache_sparse_price_{sym}.jsonl")
            series = load_price_series(path, sym)
            ts = np.fromiter((t for t, _ in series), dtype=np.float64, count=len(series))
            px = np.fromiter((p for _, p in series), dtype=np.float64, count=len(series))
            src_label = "329d-sparse"
            stride = max(5, len(ts) // 15000)

        sig222 = compute_exact_time_window_signals(ts, px, 222.0, sym)
        gc = sig222["gc"]
        gr = sig222["gr"]
        leg = sig222["legacy"]
        eps = sig222["eps"]
        sc = sig222["scale_inv"]
        noise_floor = 0.25 * sc * eps

        print(f"\n--- {sym} ({src_label}, N={len(ts):,}, p={pw.step_wols_power_for_symbol(sym):.1f}, median m={int(np.median(sig222['m']))}) ---")
        print(f"{'Gate Policy (max_wait=30s)':<46} | {'BUY Wait%':>9} | {'SELL Wait%':>10} | {'AvgWait':>7} | {'BUY bps':>8} | {'SELL bps':>8} | {'RT Edge bps':>11}")
        print("-" * 110)

        policies = [
            ("1. use_noise_gate=True (|gc|<=eps waits BUG)",
             (np.abs(gc) <= eps) | (gc < 0),
             (np.abs(gc) <= eps) | (gc > 0)),
            ("2. Legacy (sf+gr)/2, use_noise_gate=False",
             leg < 0,
             leg > 0),
            ("3. Fast (gradient_recent), use_noise_gate=False",
             gr < 0,
             gr > 0),
            ("4. Step-WOLS, use_noise_gate=False (Current)",
             gc < 0,
             gc > 0),
            ("5. Step-WOLS Deadband (|gc|>0.25*sc*eps)",
             gc < -noise_floor,
             gc > noise_floor),
        ]
        for pname, wb, ws in policies:
            r = simulate_should_wait(ts, px, wb, ws, max_wait_sec=30.0, stride=stride)
            print(f"{pname:<46} | {r['buy_wait_pct']:>8.1f}% | {r['sell_wait_pct']:>9.1f}% | {r['avg_wait_s']:>6.1f}s | "
                  f"{r['buy_bps']:>+8.2f} | {r['sell_bps']:>+8.2f} | {r['roundtrip_bps']:>+11.2f}")

    print("\n\n" + "=" * 110)
    print("  BACKTEST 3: 900s DYNAMIC WINDOW REGIME FILTER (rtrade.py / MarketRegimeEvaluator)")
    print("=" * 110)

    for sym in ["TAOUSDC", "BTCUSDC", "ARBUSDC", "HYPEUSD"]:
        npz_path = cache_path(f"{sym}.npz")
        if os.path.exists(npz_path):
            z = np.load(npz_path)
            ts = z["ts"].astype(np.float64)
            px = z["px"].astype(np.float64)
            src_label = "1s-dense"
            stride = max(60, len(ts) // 20000)
        else:
            path = cache_path(f"cache_sparse_price_{sym}.jsonl")
            series = load_price_series(path, sym)
            ts = np.fromiter((t for t, _ in series), dtype=np.float64, count=len(series))
            px = np.fromiter((p for _, p in series), dtype=np.float64, count=len(series))
            src_label = "329d-sparse"
            stride = max(5, len(ts) // 12000)

        sig900 = compute_exact_time_window_signals(ts, px, 900.0, sym)
        results, raw_s, scaled_s = evaluate_regime_900s(ts, px, sig900, horizon_sec=900.0, stride=stride)
        q50_raw, q90_raw, q99_raw = np.percentile(raw_s, [50, 90, 99])
        q50_sc, q90_sc, q99_sc = np.percentile(scaled_s, [50, 90, 99])
        print(f"\n--- {sym} 900s Window ({src_label}, median m={int(np.median(sig900['m']))}) ---")
        print(f"    Unscaled |gc|/eps percentiles:         p50={q50_raw:.4f}, p90={q90_raw:.4f}, p99={q99_raw:.4f}")
        print(f"    Scaled   |gc|/(eps*sc) percentiles:    p50={q50_sc:.4f}, p90={q90_sc:.4f}, p99={q99_sc:.4f}")
        print(f"{'Regime Filter Policy':<40} | {'Sideways%':>9} | {'15m Move(SW)':>12} | {'15m Move(TR)':>12} | {'MaxAdv(SW)':>10} | {'MaxAdv(TR)':>10} | {'2-SidedFill%':>12}")
        print("-" * 116)
        for name, sw_pct, m_sw, m_tr, a_sw, a_tr, f_sw in results:
            print(f"{name:<40} | {sw_pct:>8.1f}% | {m_sw:>11.3f}% | {m_tr:>11.3f}% | {a_sw:>9.3f}% | {a_tr:>9.3f}% | {f_sw:>11.1f}%")


if __name__ == "__main__":
    main()
