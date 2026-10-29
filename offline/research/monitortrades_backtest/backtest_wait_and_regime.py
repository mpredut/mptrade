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
os.chdir(ROOT)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")

import pricewindow as pw
from cachepaths import cache_path
from offline.research.monitortrades_backtest.sweep_mt_models import (
    compute_window_signals_from_bounds,
    load_archive_series,
)


def _load_ts_px(sym: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Load (ts, px, src_label) for `sym`, preferring dense 1s .npz when available."""
    npz_path = cache_path(f"cache_365_price_{sym}.npz")
    if os.path.exists(npz_path):
        with np.load(npz_path) as z:
            return z["ts"].astype(np.float64), z["px"].astype(np.float64), "1s-dense"
    series = load_archive_series(sym, use_365=False)
    ts = np.fromiter((t for t, _ in series), dtype=np.float64, count=len(series))
    px = np.fromiter((p for _, p in series), dtype=np.float64, count=len(series))
    return ts, px, "329d-sparse"


def compute_exact_time_window_signals(
    ts: np.ndarray,
    px: np.ndarray,
    win_sec: float,
    symbol: str,
):
    """Exact O(N) sliding two-pointer + chunked O(1) prefix-sum evaluation for dynamic window `win_sec`."""
    N = len(ts)
    w_sec = max(0.0, float(win_sec))
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
    res["legacy"] = res["legacy_raw"]
    return res


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
    stride = max(1, int(stride))
    if N > 2000:
        sample_idx = np.arange(1000, N - 1000, stride, dtype=np.int64)
    else:
        sample_idx = np.arange(0, N, stride, dtype=np.int64)
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

    for k, idx in enumerate(sample_idx):
        t0 = ts[idx]
        p0 = px[idx]
        if p0 <= 0:
            continue
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
    stride = max(1, int(stride))
    if N > 4000:
        sample_idx = np.arange(2000, N - 2000, stride, dtype=np.int64)
    else:
        sample_idx = np.arange(0, N, stride, dtype=np.int64)
    if len(sample_idx) == 0:
        z = np.zeros(1, dtype=np.float64)
        return [], z, z
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
        if p0 <= 0:
            continue
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
        ts, px, src_label = _load_ts_px(sym)
        if len(ts) == 0:
            continue
        stride = max(60, len(ts) // 25000) if src_label == "1s-dense" else max(5, len(ts) // 15000)

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
        ts, px, src_label = _load_ts_px(sym)
        if len(ts) == 0:
            continue
        stride = max(60, len(ts) // 20000) if src_label == "1s-dense" else max(5, len(ts) // 12000)

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
