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
import inspect
import os
import sys
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from strategies.spot_replay import align_spot_interval_params, prewarm_trend_series, run_replay
from strategies import spot_engine as strat
from botcore import normalize_epoch_sec, safe_float, safe_nonneg_int, safe_pos_float
from state_io import read_text_lines


def load_ohlc_with_timestamps_csv(
    csv_path: str,
) -> tuple[list[tuple[float, float, float, float]], list[int]]:
    """Load OHLC rows as (open, high, low, close) tuples alongside integer timestamps when present."""
    rows: list[tuple[float, float, float, float]] = []
    timestamps: list[int] = []
    reader = csv.reader(read_text_lines(csv_path))
    first = next(reader, None)
    if not first:
        return rows, timestamps
    col_map = {col.strip().lower(): idx for idx, col in enumerate(first)}
    has_header = any(k in col_map for k in ("open", "high", "low", "close", "timestamp", "time", "date"))
    if has_header:
        ts_idx = next((col_map[k] for k in ("timestamp", "time", "date") if k in col_map), None)
        o_idx = col_map.get("open", 1 if ts_idx is not None else 0)
        h_idx = col_map.get("high", o_idx + 1)
        l_idx = col_map.get("low", o_idx + 2)
        c_idx = col_map.get("close", o_idx + 3)
    elif len(first) >= 5:
        ts_idx, o_idx, h_idx, l_idx, c_idx = 0, 1, 2, 3, 4
    else:
        ts_idx, o_idx, h_idx, l_idx, c_idx = None, 0, 1, 2, 3
    min_len = max(o_idx, h_idx, l_idx, c_idx) + 1
    iterable = reader if has_header else [first, *reader]
    for line in iterable:
        if not line or len(line) < min_len:
            continue
        o = safe_pos_float(line[o_idx], default=None)
        h = safe_pos_float(line[h_idx], default=None)
        l = safe_pos_float(line[l_idx], default=None)
        c = safe_pos_float(line[c_idx], default=None)
        if (
            None not in (o, h, l, c)
            and h >= max(o, l, c)
            and l <= min(o, h, c)
        ):
            rows.append((o, h, l, c))
            if ts_idx is not None and ts_idx < len(line):
                ts_sec = normalize_epoch_sec(line[ts_idx], default=None)
                if ts_sec is not None and ts_sec >= 0.0:
                    ts_int = safe_nonneg_int(round(ts_sec), default=None)
                    if ts_int is not None:
                        timestamps.append(ts_int)
    if timestamps and len(timestamps) == len(rows):
        by_ts: dict[int, tuple[float, float, float, float]] = {}
        for ts_val, row in zip(timestamps, rows):
            by_ts[ts_val] = row
        ordered_ts = sorted(by_ts)
        timestamps = ordered_ts
        rows = [by_ts[t] for t in ordered_ts]
    else:
        timestamps = []
    return rows, timestamps


def load_ohlc_csv(csv_path: str) -> list[tuple[float, float, float, float]]:
    """Load OHLC rows as (open, high, low, close) tuples."""
    rows, _ = load_ohlc_with_timestamps_csv(csv_path)
    return rows


def build_ablation_variants(base_params: strat.StratParams) -> dict[str, tuple[str, strat.StratParams]]:
    """Build the parameter sets for Tests 0 through 4."""
    R = dataclasses.replace
    field_names = {f.name for f in dataclasses.fields(base_params)} if dataclasses.is_dataclass(base_params) else set()
    no_pb_adapt = {"reentry_pullback_adaptive": False} if "reentry_pullback_adaptive" in field_names else {}
    no_re_adapt = {"reentry_adaptive": False} if "reentry_adaptive" in field_names else {}
    return {
        "Test 0": (
            "Baseline (Control)",
            R(
                base_params,
                reentry_hybrid_enabled=False,
                reentry_peak_relative=False,
                reentry_pullback_pct=1.5,
                reentry_ttl_hours=0.0,
                **no_pb_adapt,
                **no_re_adapt,
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
                **no_pb_adapt,
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
                **no_pb_adapt,
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
                **no_pb_adapt,
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
                **no_pb_adapt,
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
    base_params: Any,
    fee_pct: float = 0.26,
    bar_minutes: float = 240.0,
    initial_cash: float | None = 3900.0,
    replay_fn: Any = None,
    timestamps: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Run all ablation tests on an asset and return structured metrics."""
    if not ohlc:
        return []
    aligned_base = align_spot_interval_params(base_params, bar_minutes)
    variants = build_ablation_variants(aligned_base)
    results = []
    # Warmup first 30 bars if available to seed regime
    warmup_n = min(30, len(ohlc) // 5)
    warmup_ohlc = ohlc[:warmup_n]
    run_ohlc = ohlc[warmup_n:]
    has_full_ts = bool(timestamps and len(timestamps) == len(ohlc))
    warmup_timestamps = timestamps[:warmup_n] if has_full_ts else None
    run_timestamps = timestamps[warmup_n:] if has_full_ts else None
    if not run_ohlc:
        return []

    runner = replay_fn or run_replay
    try:
        sig = inspect.signature(runner)
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        accepts_timestamps = "timestamps" in sig.parameters or has_var_kw
        accepts_warmup_ts = "warmup_timestamps" in sig.parameters or has_var_kw
    except (TypeError, ValueError):
        accepts_timestamps = replay_fn is not None
        accepts_warmup_ts = False
    if replay_fn is None or hasattr(aligned_base, "surge_guard"):
        prewarm_trend_series(ohlc, warmup=warmup_n, params=aligned_base, bar_minutes=bar_minutes)
    for test_id, (variant_name, params) in variants.items():
        try:
            replay_kw: dict[str, Any] = {
                "ohlc": run_ohlc,
                "params": params,
                "bar_minutes": bar_minutes,
                "warmup_ohlc": warmup_ohlc,
            }
            if replay_fn is None:
                replay_kw["fee_pct"] = fee_pct
            if initial_cash is not None:
                replay_kw["initial_cash"] = initial_cash
            if run_timestamps is not None and accepts_timestamps:
                replay_kw["timestamps"] = run_timestamps
            if warmup_timestamps is not None and accepts_warmup_ts:
                replay_kw["warmup_timestamps"] = warmup_timestamps
            res = runner(**replay_kw)
            res_cap = safe_pos_float(res.get("initial_capital"), default=None) if isinstance(res, dict) else None
            alloc_fn = getattr(params, "allocated_budget", None)
            alloc_cap = safe_pos_float(alloc_fn(), default=None) if callable(alloc_fn) else None
            capital = (
                initial_cash
                if initial_cash is not None
                else (
                    res_cap
                    if res_cap is not None
                    else (
                        alloc_cap
                        if alloc_cap is not None
                        else safe_pos_float(getattr(params, "max_budget", None), default=0.0)
                    )
                )
            )
            results.append(format_ablation_result(asset_name, test_id, variant_name, res, capital))
        except Exception as exc:
            results.append({
                "asset": asset_name,
                "test_id": test_id,
                "variant": variant_name,
                "error": str(exc),
            })
    return results
