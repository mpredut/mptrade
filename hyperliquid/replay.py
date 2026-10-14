#!/usr/bin/env python3
"""
replay.py — Canonical replay engine for Hyperliquid perpetual strategies.

Executes the production Strategy engine bar-by-bar over historical OHLC candles,
modeling perpetual funding, limit fills against candle extremes (high/low),
stop-loss execution, and equity drawdown without network or side-effects.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from strategy import Strategy, StratParams, HL_FEE_PCT


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

    funding = 0.0
    peak = 0.0
    maxdd = 0.0
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

        # Check close order (TP) if position is held
        close_order = next((x for x in list(strategy.s["orders"]) if x["role"] == "close"), None)
        if close_order and strategy.s["qty"] > 1e-12:
            px = close_order["px"]
            sz = close_order["sz"]
            hit = (h >= px) if strategy.sign > 0 else (l <= px)
            if hit:
                strategy._remove(close_order)
                avg = strategy._avg() or px
                strategy._apply_close(avg, px, sz)

        # 3. Strategy decisions at bar close
        signal_val = sig[i] if sig and i < len(sig) else "__any__"
        strategy.step(c, signal=signal_val)

        # 4. Equity curve and drawdown tracking
        qty = strategy.s["qty"]
        cost = strategy.s["cost"]
        realized = strategy.s["realized_net"]
        upnl = (strategy.sign * (c - cost / qty) * qty) if qty > 1e-12 else 0.0
        eq = realized + funding + upnl
        peak = max(peak, eq)
        maxdd = max(maxdd, peak - eq)

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

    return {
        "net": realized + funding,
        "realized": realized,
        "fees": total_fees,
        "funding": funding,
        "cycles": cycles,
        "wins": wins,
        "maxdd": maxdd,
        "open_qty": qty,
        "final_upnl": final_upnl,
        "total": realized + funding + final_upnl,
    }
