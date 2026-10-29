#!/usr/bin/env python3
"""Ablation experiment harness for Hybrid Re-Entry architecture (Tests 0 to 4).

Evaluates the five isolated re-entry variants:
  - Test 0: Baseline (Control) - Static post-sale drop anchor, no trend bypass.
  - Test 1: Peak-Relative - Pullback measured from post-sale peak across all regimes.
  - Test 2: Trend Bypass - Immediate unconstrained re-entry in confirmed bull regime.
  - Test 3: Hybrid - Confirmed bull allows peak pullback; chop/bear retains drop anchor.
  - Test 4: Hybrid + TTL - Hybrid logic plus 14-day decay of stale sale barrier.

Runs across historical OHLC datasets with finite cash account tracking (3900 USD budget).
"""

from __future__ import annotations

import csv
import dataclasses
import os
import sys
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
KRAKEN_DIR = os.path.join(ROOT, "kraken")
if KRAKEN_DIR not in sys.path:
    sys.path.append(KRAKEN_DIR)

from strategies.spot_replay import run_replay
from strategies import spot_engine as strat
from botcore import safe_float, safe_int, safe_nonneg_int, safe_pos_float

_safe_f = safe_float
_safe_int = safe_int


def load_ohlc_csv(csv_path: str) -> list[tuple[float, float, float, float]]:
    """Load OHLC rows as (open, high, low, close) tuples."""
    rows: list[tuple[float, float, float, float]] = []
    if not csv_path or isinstance(csv_path, (bool, int, float)) or not os.path.isfile(str(csv_path)):
        return rows
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        first = next(reader, None)
        if not first:
            return rows
        col_map = {col.strip().lower(): idx for idx, col in enumerate(first)}
        has_header = any(k in col_map for k in ("open", "high", "low", "close", "timestamp", "time", "date"))
        o_idx = col_map.get("open", 1) if has_header else 1
        h_idx = col_map.get("high", 2) if has_header else 2
        l_idx = col_map.get("low", 3) if has_header else 3
        c_idx = col_map.get("close", 4) if has_header else 4
        min_len = max(o_idx, h_idx, l_idx, c_idx) + 1
        iterable = reader if has_header else [first, *reader]
        for line in iterable:
            if not line or len(line) < min_len:
                continue
            try:
                o = float(line[o_idx])
                h = float(line[h_idx])
                l = float(line[l_idx])
                c = float(line[c_idx])
            except (ValueError, IndexError, OverflowError):
                continue
            if (
                all(safe_pos_float(x) is not None for x in (o, h, l, c))
                and h >= max(o, l, c)
                and l <= min(o, h, c)
            ):
                rows.append((o, h, l, c))
    return rows


def build_ablation_variants(base_params: strat.StratParams) -> dict[str, tuple[str, strat.StratParams]]:
    """Build the parameter sets for Tests 0 through 4."""
    R = dataclasses.replace
    return {
        "Test 0": (
            "Baseline (Control)",
            R(
                base_params,
                reentry_hybrid_enabled=False,
                reentry_peak_relative=False,
                reentry_pullback_pct=1.5,
                reentry_ttl_hours=0.0,
            ),
        ),
        "Test 1": (
            "Peak-Relative",
            R(
                base_params,
                reentry_hybrid_enabled=True,
                reentry_peak_relative=True,
                reentry_pullback_pct=1.5,
                reentry_ttl_hours=0.0,
            ),
        ),
        "Test 2": (
            "Trend Bypass",
            R(
                base_params,
                reentry_hybrid_enabled=True,
                reentry_peak_relative=False,
                reentry_pullback_pct=0.0,
                reentry_ttl_hours=0.0,
            ),
        ),
        "Test 3": (
            "Hybrid",
            R(
                base_params,
                reentry_hybrid_enabled=True,
                reentry_peak_relative=False,
                reentry_pullback_pct=1.5,
                reentry_ttl_hours=0.0,
            ),
        ),
        "Test 4": (
            "Hybrid + TTL",
            R(
                base_params,
                reentry_hybrid_enabled=True,
                reentry_peak_relative=False,
                reentry_pullback_pct=1.5,
                reentry_ttl_hours=336.0,  # 14 days
            ),
        ),
    }


def format_ablation_result(
    asset_name: str,
    test_id: str,
    variant_name: str,
    res: dict[str, Any],
    capital: float,
) -> dict[str, Any]:
    """Format a replay result dict into the canonical ablation summary row."""
    res_map = res if isinstance(res, dict) else {}
    total_usd = safe_float(res_map.get("total"), 0.0)
    cap_f = safe_pos_float(capital, 0.0)
    ret_pct = (total_usd / cap_f * 100.0) if cap_f > 0 else 0.0
    max_dd_pct = safe_float(res_map.get("max_drawdown_pct"), 0.0)
    exposure_pct = safe_float(res_map.get("exposure_pct"), 0.0)
    cycles = safe_nonneg_int(res_map.get("cycles"), default=0)
    wins = safe_nonneg_int(res_map.get("wins"), default=0)
    raw_wr = res_map.get("win_rate_pct")
    default_wr = (wins / cycles * 100.0) if cycles > 0 else 0.0
    win_rate = safe_float(raw_wr, default_wr) if raw_wr is not None else default_wr
    return {
        "asset": asset_name,
        "test_id": test_id,
        "variant": variant_name,
        "total_return_pct": round(ret_pct, 2),
        "net_profit_usd": round(total_usd, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "exposure_pct": round(exposure_pct, 2),
        "cycles": cycles,
        "wins": wins,
        "win_rate": round(win_rate, 1),
        "profit_factor": round(safe_float(res_map.get("profit_factor"), 0.0), 2),
        "fills": safe_nonneg_int(res_map.get("fills"), default=0),
    }


def evaluate_asset(
    asset_name: str,
    ohlc: list[tuple[float, float, float, float]],
    base_params: strat.StratParams,
    fee_pct: float = 0.26,
    bar_minutes: float = 240.0,
    initial_cash: float = 3900.0,
) -> list[dict[str, Any]]:
    """Run all ablation tests on an asset and return structured metrics."""
    if not ohlc:
        return []
    variants = build_ablation_variants(base_params)
    results = []
    # Warmup first 30 bars if available to seed regime
    warmup_n = min(30, len(ohlc) // 5)
    warmup_ohlc = ohlc[:warmup_n]
    run_ohlc = ohlc[warmup_n:]
    if not run_ohlc:
        return []

    for test_id, (variant_name, params) in variants.items():
        try:
            res = run_replay(
                ohlc=run_ohlc,
                params=params,
                fee_pct=fee_pct,
                bar_minutes=bar_minutes,
                warmup_ohlc=warmup_ohlc,
                initial_cash=initial_cash,
            )
            results.append(format_ablation_result(asset_name, test_id, variant_name, res, initial_cash))
        except Exception as exc:
            results.append({
                "asset": asset_name,
                "test_id": test_id,
                "variant": variant_name,
                "error": str(exc),
            })
    return results
