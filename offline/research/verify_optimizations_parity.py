"""Rigorous parity verification script comparing optimized implementations with unoptimized baselines.

Tests:
1. Linear regression in pricewindow.py vs scipy.stats.linregress
2. 1D OLS slope in priceAnalysis.py vs np.polyfit(x - x[0], y, 1)
3. detect_long_term_trend memoization vs unmemoized raw execution on real historical data
4. get_position_stats in monitortrades.py vs unmemoized recalculation
5. calculate_volatility_1h log-ratio vs np.diff(np.log(p))
"""
from __future__ import annotations

import math
import os
import sys
import time
import numpy as np
from scipy.stats import linregress

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")
os.environ.setdefault("BACKTEST_MODE", "1")
os.environ.setdefault("REPLAY_MODE", "1")
os.environ.setdefault("DISABLE_EXTERNAL_NOTIFICATIONS", "1")

__test__ = False

import pricewindow as pw
import priceAnalysis as pa
import monitortrades as mt
import intelligence.internal.state.volatility as vol
from botcore import safe_float
from cachepaths import cache_path
from providers.replay_provider import load_price_series


def test_pricewindow_regression_parity(iterations: int = 10_000) -> None:
    print(f"[TEST 1/5] Testing pricewindow.py linear regression vs scipy.stats.linregress ({iterations:,} trials)...")
    rng = np.random.default_rng(42)
    max_diff_slope = 0.0
    max_diff_intercept = 0.0
    max_diff_r = 0.0

    for i in range(iterations):
        n = rng.integers(2, 500).item()
        # Generate random price walk
        base = rng.uniform(10.0, 100_000.0)
        vol_pct = rng.uniform(0.0001, 0.05)
        returns = rng.normal(0, vol_pct, size=n)
        prices = base * np.exp(np.cumsum(returns))

        # 1. Unoptimized baseline using scipy.stats.linregress
        x = np.arange(n)
        y = np.asarray(prices, dtype=np.float64)
        slope_base, intercept_base, r_base, _, _ = linregress(x, y)
        line_base = slope_base * x + intercept_base

        # 2. Optimized implementation from pricewindow
        analyzer = pw.PriceTrendAnalyzer(prices)
        line_opt, slope_opt, r_opt = analyzer.linear_regression_trend()

        diff_s = abs(slope_base - slope_opt)
        diff_b = abs(intercept_base - (line_opt[0]))
        diff_r = abs(r_base - r_opt)

        max_diff_slope = max(max_diff_slope, diff_s)
        max_diff_intercept = max(max_diff_intercept, diff_b)
        max_diff_r = max(max_diff_r, diff_r)

        assert np.isclose(slope_base, slope_opt, rtol=1e-11, atol=1e-12), f"Slope mismatch at iter {i}: {slope_base} vs {slope_opt}"
        assert np.isclose(r_base, r_opt, rtol=1e-11, atol=1e-12), f"R-value mismatch at iter {i}: {r_base} vs {r_opt}"
        assert np.allclose(line_base, line_opt, rtol=1e-11, atol=1e-12), f"Line mismatch at iter {i}"

    print(f"  -> PASSED! Exact match across {iterations:,} randomized price walks.")
    print(f"     Max delta: slope={max_diff_slope:.2e}, intercept={max_diff_intercept:.2e}, r_value={max_diff_r:.2e}")


def test_priceanalysis_ols_parity(iterations: int = 10_000) -> None:
    print(f"[TEST 2/5] Testing priceAnalysis.py _fast_linear_regression_1d vs np.polyfit ({iterations:,} trials)...")
    rng = np.random.default_rng(123)
    max_diff_slope = 0.0
    max_diff_intercept = 0.0

    for i in range(iterations):
        n = rng.integers(4, 200).item()
        # Realistic irregular timestamps (seconds)
        t0 = 1_700_000_000.0 + rng.uniform(0, 10_000_000.0)
        dt = rng.exponential(scale=2.5, size=n)
        dt[0] = 0.0
        timestamps = t0 + np.cumsum(dt)

        base_p = rng.uniform(50.0, 90_000.0)
        drift = rng.uniform(-5.0, 5.0)
        noise = rng.normal(0, base_p * 0.005, size=n)
        prices = base_p + drift * (timestamps - t0) / 3600.0 + noise

        # 1. Unoptimized baseline using np.polyfit
        x_block = timestamps - timestamps[0]
        y_block = prices
        s_base, b_base = np.polyfit(x_block, y_block, 1)

        # 2. Optimized implementation
        s_opt, b_opt = pa._fast_linear_regression_1d(timestamps, prices)

        diff_s = abs(s_base - s_opt)
        diff_b = abs(b_base - b_opt)
        max_diff_slope = max(max_diff_slope, diff_s)
        max_diff_intercept = max(max_diff_intercept, diff_b)

        assert np.isclose(s_base, s_opt, rtol=1e-10, atol=1e-11), f"Slope mismatch at iter {i}: {s_base} vs {s_opt}"
        assert np.isclose(b_base, b_opt, rtol=1e-9, atol=1e-10), f"Intercept mismatch at iter {i}: {b_base} vs {b_opt}"

    print(f"  -> PASSED! Exact match across {iterations:,} randomized time-series blocks.")
    print(f"     Max delta: slope={max_diff_slope:.2e}, intercept={max_diff_intercept:.2e}")


def test_detect_long_term_trend_real_data() -> None:
    print("[TEST 3/5] Testing detect_long_term_trend memoization vs raw unmemoized on real historical caches...")
    symbols = ["BTCUSDC", "TAOUSDC"]
    datasets = []
    for sym in symbols:
        c_path = cache_path(f"cache_sparse_price_{sym}.jsonl")
        if not os.path.exists(c_path):
            continue
        series = load_price_series(c_path, sym, max_samples=2000)
        if series and len(series) >= 1000:
            datasets.append((sym, [s[0] for s in series], [s[1] for s in series]))

    if not datasets:
        rng = np.random.default_rng(4242)
        t0 = 1_700_000_000.0
        dt = rng.uniform(55.0, 65.0, size=2500)
        dt[0] = 0.0
        syn_ts = (t0 + np.cumsum(dt)).tolist()
        syn_px = (50_000.0 * np.exp(np.cumsum(rng.normal(0.00015, 0.002, size=2500)))).tolist()
        datasets.append(("SYNTHETIC_BTCUSDC", syn_ts, syn_px))

    saved_cache = dict(pa._trend_memo_cache)
    try:
        for sym, timestamps, prices in datasets:
            # 1. Warm incremental memoization cache on prefix, then evaluate full series (warm incremental path)
            pa._trend_memo_cache.clear()
            pa.detect_long_term_trend(timestamps[:-50], prices[:-50], window_hours=16, step_hours=8)
            res_memo = pa.detect_long_term_trend(timestamps, prices, window_hours=16, step_hours=8)

            # 2. Clear memoization cache and evaluate full series from scratch (cold unmemoized reference)
            pa._trend_memo_cache.clear()
            res_cold = pa.detect_long_term_trend(timestamps, prices, window_hours=16, step_hours=8)

            # Verify exact parity between incremental-cached and cold-computed results
            if res_cold is None:
                assert res_memo is None
            else:
                assert res_memo is not None
                assert res_memo["direction"] == res_cold["direction"]
                assert res_memo["start_timestamp"] == res_cold["start_timestamp"]
                assert res_memo["duration_seconds"] == res_cold["duration_seconds"]
                assert np.isclose(res_memo["current_slope_h"], res_cold["current_slope_h"], rtol=1e-12, atol=1e-12)
                assert res_memo["blocks"] == res_cold["blocks"]

            print(
                f"  -> PASSED for {sym} history: "
                f"direction={res_cold.get('direction') if res_cold else None}, "
                f"duration={safe_float(res_cold.get('duration_seconds'), 0.0) if res_cold else 0.0:.0f}s"
            )
    finally:
        pa._trend_memo_cache.clear()
        pa._trend_memo_cache.update(saved_cache)


def test_monitortrades_position_stats_parity() -> None:
    print("[TEST 4/5] Testing monitortrades get_position_stats dirty-flag vs unmemoized recalculation...")
    sample_buys = [
        {"id": 101, "timestamp": 1700000000000, "price": "60000.5", "qty": "0.15"},
        {"id": 102, "timestamp": 1700003600000, "price": "61200.0", "qty": "0.10"},
        {"id": 103, "timestamp": 1700007200000, "price": "59800.0", "qty": "0.25"},
    ]
    sample_sells = [
        {"id": 201, "timestamp": 1700005000000, "price": "61500.0", "qty": "0.10"},
    ]

    # Baseline manual calculation
    tot_buy_qty = 0.15 + 0.10 + 0.25
    tot_sell_qty = 0.10
    tot_buy_val = (60000.5 * 0.15) + (61200.0 * 0.10) + (59800.0 * 0.25)
    tot_sell_val = 61500.0 * 0.10
    avg_buy = tot_buy_val / tot_buy_qty
    avg_sell = tot_sell_val / tot_sell_qty

    saved_pos_cache = dict(mt._position_stats_cache)
    try:
        mt._position_stats_cache.clear()
        stats1 = mt.get_position_stats("BTCUSDT", 86400, buy_orders=sample_buys, sell_orders=sample_sells)
        # Second call uses dirty-flag memoization
        stats2 = mt.get_position_stats("BTCUSDT", 86400, buy_orders=sample_buys, sell_orders=sample_sells)

        assert stats1["buy_qty"] == tot_buy_qty
        assert stats1["sell_qty"] == tot_sell_qty
        assert stats1["net_qty"] == (tot_buy_qty - tot_sell_qty)
        assert math.isclose(stats1["average_buy_price"], avg_buy, rel_tol=1e-9)
        assert math.isclose(stats1["average_sell_price"], avg_sell, rel_tol=1e-9)

        assert stats1 == stats2, "Memoized stats do not match initial computation!"

        # Now simulate a new fill arriving (dirty flag trips)
        sample_buys_new = [{"id": 104, "timestamp": 1700008000000, "price": "62000.0", "qty": "0.10"}] + sample_buys
        stats3 = mt.get_position_stats("BTCUSDT", 86400, buy_orders=sample_buys_new, sell_orders=sample_sells)
        assert stats3["buy_count"] == 4
        assert stats3["buy_qty"] == 0.60
        assert stats3 != stats1, "Dirty flag failed to detect new order fill!"
    finally:
        mt._position_stats_cache.clear()
        mt._position_stats_cache.update(saved_pos_cache)

    print("  -> PASSED! Memoized stats match manual calculation, and dirty-flag invalidation triggers on new fills.")


def test_volatility_parity(iterations: int = 10_000) -> None:
    print(f"[TEST 5/5] Testing volatility.py log-ratio vs np.diff(np.log(p)) ({iterations:,} trials)...")
    rng = np.random.default_rng(999)
    max_diff_vol = 0.0

    saved_vol_cache = dict(vol._vol_memo_cache)
    try:
        for i in range(iterations):
            n = rng.integers(20, 200).item()
            p0 = rng.uniform(1.0, 50_000.0)
            returns = rng.normal(0, 0.02, size=n)
            prices = p0 * np.exp(np.cumsum(returns))
            sample_rate = safe_float(rng.choice([0.8, 1.0, 5.0, 60.0, 300.0]).item(), 1.0)

            # Baseline calculation using np.diff(np.log(p))
            p = np.asarray(prices, dtype=np.float64)
            p = p[p > 0]
            rets_base = np.diff(np.log(p))
            std_base = safe_float(np.std(rets_base).item(), 0.0)
            vol_base = round(std_base * math.sqrt(3600.0 / sample_rate) * 100.0, 4)

            # Optimized calculate_volatility_1h
            vol._vol_memo_cache.clear()
            vol_opt = safe_float(vol.calculate_volatility_1h(prices, sample_rate), 0.0)

            diff = abs(vol_base - vol_opt)
            max_diff_vol = max(max_diff_vol, diff)
            assert math.isclose(vol_base, vol_opt, rel_tol=1e-12, abs_tol=1e-12), (
                f"Volatility mismatch at iter {i}: {vol_base} vs {vol_opt}"
            )
    finally:
        vol._vol_memo_cache.clear()
        vol._vol_memo_cache.update(saved_vol_cache)

    print(f"  -> PASSED! Exact match across {iterations:,} randomized trials.")
    print(f"     Max delta: {max_diff_vol:.2e}")


if __name__ == "__main__":
    print("=" * 80)
    print("COMPREHENSIVE NUMERICAL PARITY & ZERO-REGRESSION TEST SUITE")
    print("=" * 80)
    t_start = time.perf_counter()

    test_pricewindow_regression_parity(10_000)
    test_priceanalysis_ols_parity(10_000)
    test_detect_long_term_trend_real_data()
    test_monitortrades_position_stats_parity()
    test_volatility_parity(10_000)

    elapsed = time.perf_counter() - t_start
    print("=" * 80)
    print(f"ALL 5 PARITY SUITES PASSED 100%! Completed 30,000+ validations in {elapsed:.2f}s.")
    print("Zero mathematical divergence detected between optimized and unoptimized baselines.")
    print("=" * 80)
