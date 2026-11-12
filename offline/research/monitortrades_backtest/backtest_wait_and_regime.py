#!/usr/bin/env python3
"""Backtest 2 & 3:
- Backtest 2: Opportunistic Execution Delay Gate (`should_wait` / `GT_S`) across
  1-second dense and 329-day archives (BTCUSDC, TAOUSDC, ARBUSDC, HYPEUSD).
- Backtest 3: 900s Dynamic Window Regime Filter (`rtrade.py` / `MarketRegimeEvaluator`)
  comparing unscaled vs scale-invariant `strength` on 15-minute windows.
"""
from __future__ import annotations

import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

from botcore import safe_float, safe_int, safe_nonneg_float, safe_pos_int
import pricewindow as pw
from cachepaths import cache_path
from offline.research.monitortrades_backtest.sweep_mt_models import (
    compute_window_signals_from_bounds,
    load_archive_series,
    load_fresh_365_npz_arrays,
)


def compute_exact_time_window_signals(
    ts: np.ndarray,
    px: np.ndarray,
    win_sec: float,
    symbol: str,
):
    """Exact O(N) sliding two-pointer + chunked O(1) prefix-sum evaluation for dynamic window `win_sec`."""
    N = len(ts)
    w_sec = safe_nonneg_float(win_sec, default=0.0)
    # Two-pointer to find a[b] = first index with ts[a] >= ts[b] - win_sec
    a = np.searchsorted(ts, ts - w_sec, side="left")
    if N > 0:
        a = np.minimum(a, N - 1)
    b = np.arange(N, dtype=np.int64)
    m_f = np.maximum(1.0, (b - a).astype(np.float64))
    elapsed = np.maximum(1.0, ts[b] - ts[a]) if N > 0 else np.zeros(0, dtype=np.float64)
    sr = np.maximum(0.5, elapsed / m_f)

    res = compute_window_signals_from_bounds(
        px, a, sr, symbol=symbol, recent_win_sec=w_sec, min_valid_n=3
    )
    return res


def _build_strided_sample_idx(ts: np.ndarray, stride: int, trim: int) -> tuple[int, int, np.ndarray]:
    n = len(ts)
    stride_i = safe_pos_int(stride, default=1)
    sample_idx = (
        np.arange(trim, n - trim, stride_i, dtype=np.int64)
        if n > 2 * trim
        else np.arange(0, n, stride_i, dtype=np.int64)
    )
    return n, stride_i, sample_idx


def simulate_should_wait(
    ts: np.ndarray,
    px: np.ndarray,
    wait_buy_mask: np.ndarray,
    wait_sell_mask: np.ndarray,
    max_wait_sec: float = 30.0,
    stride: int = 300,
):
    """Evaluate average execution improvement (in bps) over sampled intent arrival points."""
    N, stride, sample_idx = _build_strided_sample_idx(ts, stride, 1000)
    wait_cap = safe_nonneg_float(max_wait_sec, default=30.0)
    if len(sample_idx) == 0:
        return {
            "buy_bps": 0.0,
            "sell_bps": 0.0,
            "roundtrip_bps": 0.0,
            "buy_wait_pct": 0.0,
            "sell_wait_pct": 0.0,
            "avg_wait_s": 0.0,
        }
    buy_bps = np.zeros(len(sample_idx), dtype=np.float64)
    sell_bps = np.zeros(len(sample_idx), dtype=np.float64)
    buy_waited_s = np.zeros(len(sample_idx), dtype=np.float64)
    sell_waited_s = np.zeros(len(sample_idx), dtype=np.float64)
    valid_mask = np.zeros(len(sample_idx), dtype=bool)
    t_last = safe_float(ts[-1], 0.0) if N > 0 else 0.0

    for k, idx in enumerate(sample_idx):
        t0 = safe_float(ts[idx], default=None)
        p0 = safe_float(px[idx], default=None)
        if t0 is None or p0 is None or p0 <= 0.0:
            continue
        valid_mask[k] = True
        deadline = t0 + wait_cap
        max_avail_wait = min(wait_cap, max(0.0, t_last - t0))

        # BUY: wait while wait_buy_mask[j] is True and next sample stays within deadline
        j = idx
        while j + 1 < N and ts[j + 1] <= deadline and wait_buy_mask[j]:
            j += 1
        cand_buy = safe_float(px[j], default=None)
        pj_buy = cand_buy if (cand_buy is not None and cand_buy > 0.0) else p0
        buy_bps[k] = ((p0 - pj_buy) / p0) * 10000.0
        buy_waited_s[k] = max_avail_wait if bool(wait_buy_mask[j]) else max(0.0, safe_float(ts[j] - t0, 0.0))

        # SELL: wait while wait_sell_mask[j] is True and next sample stays within deadline
        j = idx
        while j + 1 < N and ts[j + 1] <= deadline and wait_sell_mask[j]:
            j += 1
        cand_sell = safe_float(px[j], default=None)
        pj_sell = cand_sell if (cand_sell is not None and cand_sell > 0.0) else p0
        sell_bps[k] = ((pj_sell - p0) / p0) * 10000.0
        sell_waited_s[k] = max_avail_wait if bool(wait_sell_mask[j]) else max(0.0, safe_float(ts[j] - t0, 0.0))

    if not np.any(valid_mask):
        return {
            "buy_bps": 0.0,
            "sell_bps": 0.0,
            "roundtrip_bps": 0.0,
            "buy_wait_pct": 0.0,
            "sell_wait_pct": 0.0,
            "avg_wait_s": 0.0,
        }
    v_idx = sample_idx[valid_mask]
    return {
        "buy_bps": safe_float(np.mean(buy_bps[valid_mask]), 0.0),
        "sell_bps": safe_float(np.mean(sell_bps[valid_mask]), 0.0),
        "roundtrip_bps": safe_float(np.mean(buy_bps[valid_mask]) + np.mean(sell_bps[valid_mask]), 0.0),
        "buy_wait_pct": safe_float(np.mean(wait_buy_mask[v_idx]) * 100.0, 0.0),
        "sell_wait_pct": safe_float(np.mean(wait_sell_mask[v_idx]) * 100.0, 0.0),
        "avg_wait_s": safe_float(0.5 * (np.mean(buy_waited_s[valid_mask]) + np.mean(sell_waited_s[valid_mask])), 0.0),
    }


def evaluate_regime_900s(
    ts: np.ndarray,
    px: np.ndarray,
    sig900: dict,
    horizon_sec: float = 900.0,
    stride: int = 300,
):
    """Evaluate 900s regime filter for rtrade: sideways % vs forward 15m adverse excursion & spread fill."""
    N, stride, sample_idx = _build_strided_sample_idx(ts, stride, 2000)
    h_sec = safe_nonneg_float(horizon_sec, default=900.0)
    z = np.zeros(1, dtype=np.float64)
    if len(sample_idx) == 0 or N < 2:
        return [], z, z
    # Forward 15m endpoint strictly within [t0, t0 + h_sec] without looking across archive gaps
    fwd_end = np.searchsorted(ts, ts[sample_idx] + h_sec, side="right") - 1
    if (ts[-1] - ts[0]) >= h_sec:
        has_full_horizon = ((ts[sample_idx] + h_sec) <= ts[-1]) & (fwd_end > sample_idx)
    else:
        has_full_horizon = fwd_end > sample_idx
    if not np.any(has_full_horizon):
        return [], z, z
    sample_idx = sample_idx[has_full_horizon]
    fwd_end = fwd_end[has_full_horizon]

    # Calculate forward 15m absolute move (%) and whether a symmetric +/-0.32% spread fills both sides
    fwd_abs_move_pct = np.zeros(len(sample_idx), dtype=np.float64)
    fwd_max_adverse_pct = np.zeros(len(sample_idx), dtype=np.float64)
    both_sides_filled = np.zeros(len(sample_idx), dtype=bool)
    valid_mask = np.zeros(len(sample_idx), dtype=bool)
    half_spread = 0.0032  # 0.32% each side (RTRADE_DEFAULT_ADJUSTMENT_PCT = 0.64%)

    for k, (i0, i1) in enumerate(zip(sample_idx, fwd_end)):
        if i1 <= i0:
            continue
        p0 = safe_float(px[i0], default=None)
        if p0 is None or p0 <= 0.0:
            continue
        window_px = px[i0 : i1 + 1]
        finite_px = window_px[np.isfinite(window_px) & (window_px > 0.0)]
        if len(finite_px) == 0:
            continue
        valid_mask[k] = True
        p_min = safe_float(np.min(finite_px), p0)
        p_max = safe_float(np.max(finite_px), p0)
        p_end = safe_float(finite_px[-1], p0)
        fwd_abs_move_pct[k] = abs(p_end - p0) / p0 * 100.0
        fwd_max_adverse_pct[k] = max(abs(p_max - p0), abs(p0 - p_min)) / p0 * 100.0
        both_sides_filled[k] = (p_min <= p0 * (1.0 - half_spread)) and (p_max >= p0 * (1.0 + half_spread))

    if not np.any(valid_mask):
        return [], z, z
    sample_idx = sample_idx[valid_mask]
    fwd_abs_move_pct = fwd_abs_move_pct[valid_mask]
    fwd_max_adverse_pct = fwd_max_adverse_pct[valid_mask]
    both_sides_filled = both_sides_filled[valid_mask]

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
        sw_pct = safe_float(np.mean(is_sideways) * 100.0, 0.0)
        has_sw = bool(np.any(is_sideways))
        has_tr = bool(np.any(~is_sideways))
        move_sw = safe_float(np.mean(fwd_abs_move_pct[is_sideways]), 0.0) if has_sw else 0.0
        move_tr = safe_float(np.mean(fwd_abs_move_pct[~is_sideways]), 0.0) if has_tr else 0.0
        adv_sw = safe_float(np.mean(fwd_max_adverse_pct[is_sideways]), 0.0) if has_sw else 0.0
        adv_tr = safe_float(np.mean(fwd_max_adverse_pct[~is_sideways]), 0.0) if has_tr else 0.0
        fill_sw = safe_float(np.mean(both_sides_filled[is_sideways]) * 100.0, 0.0) if has_sw else 0.0
        results.append((name, sw_pct, move_sw, move_tr, adv_sw, adv_tr, fill_sw))
    return results, raw_strength, scaled_strength


def main():
    os.chdir(ROOT)
    symbols = ["BTCUSDC", "TAOUSDC", "ARBUSDC", "HYPEUSD"]
    print(
        f"{'=' * 110}\n  BACKTEST 2: OPPORTUNISTIC EXECUTION DELAY GATE (should_wait / GT_S) ON 222s PRIMARY WINDOW\n{'=' * 110}"
    )

    def _iter_loaded_symbol_signals(sym_list, window_sec: float, dense_div: int, sparse_div: int):
        for s in sym_list:
            arrs = load_fresh_365_npz_arrays(s)
            if arrs is not None:
                ts_arr, px_arr, src_lbl = arrs[0], arrs[1], "1s-dense"
            else:
                series_365 = (
                    load_archive_series(s, use_365=True)
                    if os.path.exists(cache_path(f"cache_365_price_{s}.jsonl"))
                    else None
                )
                series = series_365 or load_archive_series(s, use_365=False)
                ts_arr = np.fromiter((t for t, _ in series), dtype=np.float64, count=len(series))
                px_arr = np.fromiter((p for _, p in series), dtype=np.float64, count=len(series))
                src_lbl = "1s-dense" if series_365 else "329d-sparse"
            if len(ts_arr) == 0:
                continue
            step_s = max(60, len(ts_arr) // dense_div) if src_lbl == "1s-dense" else max(5, len(ts_arr) // sparse_div)
            yield s, ts_arr, px_arr, src_lbl, step_s, compute_exact_time_window_signals(ts_arr, px_arr, window_sec, s)

    for sym, ts, px, src_label, stride, sig222 in _iter_loaded_symbol_signals(symbols, 222.0, 25000, 15000):
        gc = sig222["gc"]
        gr = sig222["gr"]
        leg = sig222["legacy"]
        eps = sig222["eps"]
        sc = sig222["scale_inv"]
        noise_floor = 0.25 * sc * eps

        print(
            f"\n--- {sym} ({src_label}, N={len(ts):,}, p={pw.step_wols_power_for_symbol(sym):.1f}, median m={safe_int(np.median(sig222['m']), default=0)}) ---\n"
            f"{'Gate Policy (max_wait=30s)':<46} | {'BUY Wait%':>9} | {'SELL Wait%':>10} | {'AvgWait':>7} | {'BUY bps':>8} | {'SELL bps':>8} | {'RT Edge bps':>11}\n"
            + ("-" * 110)
        )

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

    print(
        f"\n\n{'=' * 110}\n  BACKTEST 3: 900s DYNAMIC WINDOW REGIME FILTER (rtrade.py / MarketRegimeEvaluator)\n{'=' * 110}"
    )

    for sym, ts, px, src_label, stride, sig900 in _iter_loaded_symbol_signals(
        ["TAOUSDC", "BTCUSDC", "ARBUSDC", "HYPEUSD"], 900.0, 20000, 12000
    ):
        results, raw_s, scaled_s = evaluate_regime_900s(ts, px, sig900, horizon_sec=900.0, stride=stride)
        q50_raw, q90_raw, q99_raw = np.percentile(raw_s, [50, 90, 99]) if len(raw_s) else (0.0, 0.0, 0.0)
        q50_sc, q90_sc, q99_sc = np.percentile(scaled_s, [50, 90, 99]) if len(scaled_s) else (0.0, 0.0, 0.0)
        print(
            f"\n--- {sym} 900s Window ({src_label}, median m={safe_int(np.median(sig900['m']), default=0)}) ---\n"
            f"    Unscaled |gc|/eps percentiles:         p50={q50_raw:.4f}, p90={q90_raw:.4f}, p99={q99_raw:.4f}\n"
            f"    Scaled   |gc|/(eps*sc) percentiles:    p50={q50_sc:.4f}, p90={q90_sc:.4f}, p99={q99_sc:.4f}\n"
            f"{'Regime Filter Policy':<40} | {'Sideways%':>9} | {'15m Move(SW)':>12} | {'15m Move(TR)':>12} | {'MaxAdv(SW)':>10} | {'MaxAdv(TR)':>10} | {'2-SidedFill%':>12}\n"
            + ("-" * 116)
        )
        for name, sw_pct, m_sw, m_tr, a_sw, a_tr, f_sw in results:
            print(f"{name:<40} | {sw_pct:>8.1f}% | {m_sw:>11.3f}% | {m_tr:>11.3f}% | {a_sw:>9.3f}% | {a_tr:>9.3f}% | {f_sw:>11.1f}%")


if __name__ == "__main__":
    main()
