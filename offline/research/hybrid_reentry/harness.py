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
import json
import math
import os
import sys
import time
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
KRAKEN_DIR = os.path.join(ROOT, "kraken")
if KRAKEN_DIR not in sys.path:
    sys.path.insert(0, KRAKEN_DIR)

from kraken.replay import run_replay
from strategies import spot_dca as strat


def load_ohlc_csv(csv_path: str) -> list[tuple[float, float, float, float]]:
    """Load OHLC rows as (open, high, low, close) tuples."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        # Identify columns
        col_map = {col.lower(): idx for idx, col in enumerate(header)} if header else {}
        o_idx = col_map.get("open", 1)
        h_idx = col_map.get("high", 2)
        l_idx = col_map.get("low", 3)
        c_idx = col_map.get("close", 4)
        for line in reader:
            if not line or len(line) < 5:
                continue
            try:
                o = float(line[o_idx])
                h = float(line[h_idx])
                l = float(line[l_idx])
                c = float(line[c_idx])
                rows.append((o, h, l, c))
            except (ValueError, IndexError):
                continue
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


def evaluate_asset(
    asset_name: str,
    ohlc: list[tuple[float, float, float, float]],
    base_params: strat.StratParams,
    fee_pct: float = 0.26,
    bar_minutes: float = 240.0,
    initial_cash: float = 3900.0,
) -> list[dict[str, Any]]:
    """Run all ablation tests on an asset and return structured metrics."""
    variants = build_ablation_variants(base_params)
    results = []
    # Warmup first 30 bars if available to seed regime
    warmup_n = min(30, len(ohlc) // 5)
    warmup_ohlc = ohlc[:warmup_n]
    run_ohlc = ohlc[warmup_n:]

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
            ret_pct = (res.get("total", 0.0) / initial_cash) * 100.0
            max_dd_pct = float(res.get("max_drawdown_pct") or 0.0)
            exposure_pct = float(res.get("exposure_pct") or 0.0)
            results.append({
                "asset": asset_name,
                "test_id": test_id,
                "variant": variant_name,
                "total_return_pct": round(ret_pct, 2),
                "net_profit_usd": round(res.get("total", 0.0), 2),
                "max_drawdown_pct": round(max_dd_pct, 2),
                "exposure_pct": round(exposure_pct, 2),
                "cycles": res.get("cycles", 0),
                "wins": res.get("wins", 0),
                "win_rate": round(float(res.get("win_rate") or 0.0) * 100.0, 1),
                "profit_factor": round(float(res.get("profit_factor") or 0.0), 2),
                "fills": res.get("fills", 0),
            })
        except Exception as exc:
            results.append({
                "asset": asset_name,
                "test_id": test_id,
                "variant": variant_name,
                "error": str(exc),
            })
    return results
