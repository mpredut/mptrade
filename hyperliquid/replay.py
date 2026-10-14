#!/usr/bin/env python3
"""
replay.py — Canonical replay engine for Hyperliquid perpetual strategies.

Executes the production Strategy engine bar-by-bar over historical OHLC candles,
modeling perpetual funding, limit fills against candle extremes (high/low),
stop-loss execution, and equity drawdown without network or side-effects.
Standardizes metrics reporting through calculate_performance_metrics.
"""

from __future__ import annotations

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_HERE, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from strategy import Strategy, StratParams, HL_FEE_PCT
from offline.backtests.metrics import calculate_performance_metrics


def funding_at(fund: list[tuple[int, float]] | None, t_ms: int) -> float:
    """Return the funding rate active at t_ms, using the most recent rate at or before t."""
    if not fund:
        return 0.0
    lo, hi = 0, len(fund)
    while lo < hi:
        mid = (lo + hi) // 2
        if fund[mid][0] <= t_ms:
            lo = mid + 1
        else:
            hi = mid
    return fund[lo - 1][1] if lo > 0 else (fund[0][1] if fund else 0.0)


def run_replay(
    ohlc: list[tuple[float, float, float, float]],
    ts: list[int] | None = None,
    fund: list[tuple[int, float]] | None = None,
    sig: list[str] | None = None,
    params: StratParams | dict | None = None,
    direction: str = "long",
    coin: str = "HYPE",
    bar_minutes: float | None = 60.0,
    periods_per_year: float | None = None,
) -> dict:
    """Replay historical candles through the live Strategy engine."""
    if not ohlc:
        return {
            "net": 0.0,
            "realized": 0.0,
            "fees": 0.0,
            "funding": 0.0,
            "cycles": 0,
            "wins": 0,
            "maxdd": 0.0,
            "open_qty": 0.0,
            "final_upnl": 0.0,
            "total": 0.0,
        }

    if isinstance(params, StratParams):
        strat_params = params
    elif isinstance(params, dict):
        strat_params = StratParams.from_dict(params, direction=direction)
    else:
        strat_params = StratParams(direction=direction)

    strategy = Strategy(
        client=None,
        coin=coin,
        params=strat_params,
        dry_run=True,
        replay_mode=True,
    )

    initial_capital = float(strat_params.max_budget)
    equity_curve = [initial_capital]
    exposure: list[bool] = []
    trade_pnls: list[float] = []
    cycle_net_start = 0.0
    turnover_notional = 0.0
    fills = 0
    funding = 0.0
    c = ohlc[0][3]

    for i in range(len(ohlc)):
        o, h, l, c = ohlc[i]
        t_ms = ts[i] if ts and i < len(ts) else 0

        # 1. Funding on held perpetuals: shorts receive funding; longs pay it.
        qty = strategy.s["qty"]
        if qty > 1e-12:
            rate = funding_at(fund, t_ms)
            f = rate * qty * c
            funding += (f if strategy.sign < 0 else -f)

        # 2. Check active orders against the bar range
        # Check open order first (entry or DCA)
        open_order = next((x for x in list(strategy.s["orders"]) if x["role"] == "open"), None)
        if open_order:
            px = open_order["px"]
            sz = open_order["sz"]
            hit = (l <= px) if strategy.sign > 0 else (h >= px)
            if hit:
                strategy._remove(open_order)
                strategy._apply_open(sz, px, open_order.get("amount", 0.0), open_order.get("kind"))
                turnover_notional += sz * px
                fills += 1

        # Check close order (TP) if position is held
        close_order = next((x for x in list(strategy.s["orders"]) if x["role"] == "close"), None)
        if close_order and strategy.s["qty"] > 1e-12:
            px = close_order["px"]
            sz = close_order["sz"]
            hit = (h >= px) if strategy.sign > 0 else (l <= px)
            if hit:
                strategy._remove(close_order)
                avg = strategy._avg() or px
                cycle_before = strategy.s.get("cycles_completed", 0)
                strategy._apply_close(avg, px, sz)
                turnover_notional += sz * px
                fills += 1
                if strategy.s.get("cycles_completed", 0) > cycle_before:
                    trade_pnls.append(strategy.s["realized_net"] - cycle_net_start)
                    cycle_net_start = strategy.s["realized_net"]

        # 3. Strategy decisions at bar close
        signal_val = sig[i] if sig and i < len(sig) else "__any__"
        cycle_before_step = strategy.s.get("cycles_completed", 0)
        strategy.step(c, signal=signal_val)
        if strategy.s.get("cycles_completed", 0) > cycle_before_step:
            trade_pnls.append(strategy.s["realized_net"] - cycle_net_start)
            cycle_net_start = strategy.s["realized_net"]
            fills += 1

        # 4. Equity curve and exposure tracking
        qty = strategy.s["qty"]
        cost = strategy.s["cost"]
        realized = strategy.s["realized_net"]
        upnl = (strategy.sign * (c - cost / qty) * qty) if qty > 1e-12 else 0.0
        fee_pct = getattr(strategy.p, "fee_pct", HL_FEE_PCT)
        open_fee = (fee_pct / 100.0) * cost if qty > 1e-12 else 0.0
        current_equity = initial_capital + realized + funding + upnl - open_fee
        equity_curve.append(current_equity)
        exposure.append(qty > 1e-12)

    qty = strategy.s["qty"]
    cost = strategy.s["cost"]
    realized = strategy.s["realized_net"]
    final_upnl = (strategy.sign * (c - cost / qty) * qty) if qty > 1e-12 else 0.0
    cycles = strategy.s.get("cycles_completed", 0)
    wins = strategy.s.get("wins", 0)
    total_fees = strategy.s["fees_total"]
    if qty > 1e-12:
        fee_pct = getattr(strategy.p, "fee_pct", HL_FEE_PCT)
        total_fees += (fee_pct / 100.0) * cost

    if periods_per_year is None and bar_minutes:
        periods_per_year = 365.0 * 24 * 60 / bar_minutes

    performance = calculate_performance_metrics(
        equity_curve,
        initial_capital=initial_capital,
        periods_per_year=periods_per_year,
        exposure=exposure,
        trade_pnls=trade_pnls,
        turnover_notional=turnover_notional,
    )

    rounded = lambda v: round(float(v), 10) if isinstance(v, (int, float)) and math.isfinite(v) else v
    result = {
        "realized": rounded(realized),
        "net": rounded(realized + funding),
        "fees": rounded(total_fees),
        "funding": rounded(funding),
        "cycles": cycles,
        "wins": wins,
        "maxdd": rounded(performance["max_drawdown_abs"]),
        "open_qty": rounded(qty),
        "final_upnl": rounded(final_upnl),
        "total": rounded(realized + funding + final_upnl),
        "fills": fills,
        "ambiguous_bars": 0,
        "initial_capital": rounded(initial_capital),
        "account_currency": strat_params.currency,
    }
    result.update({
        key: rounded(value)
        for key, value in performance.items()
    })
    return result


def make_hyperliquid_replay_adapter(
    params: StratParams | dict,
    direction: str = "long",
    coin: str = "HYPE",
    fund: list[tuple[int, float]] | None = None,
    bar_minutes: float | None = 60.0,
):
    """Return a generic ReplayFunction compatible with evaluation.py walk-forward benchmarks."""
    def _adapter(ohlc, _warmup_ohlc, context=None):
        ts = list(context.timestamps) if context and getattr(context, "timestamps", None) else None
        return run_replay(
            ohlc,
            ts=ts,
            fund=fund,
            params=params,
            direction=direction,
            coin=coin,
            bar_minutes=bar_minutes,
        )
    return _adapter
