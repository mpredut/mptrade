#!/usr/bin/env python3
"""
replay.py — Canonical replay engine for Hyperliquid perpetual strategies.

Executes the production Strategy engine bar-by-bar over historical OHLC candles,
modeling perpetual funding, limit fills against candle extremes (high/low),
stop-loss execution, and equity drawdown without network or side-effects.
Standardizes metrics reporting through calculate_performance_metrics and execution
modeling through ExecutionModel.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, _HERE):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
_strat_mod = sys.modules.get("strategy")
if _strat_mod is not None and os.path.dirname(os.path.abspath(getattr(_strat_mod, "__file__", ""))) != _HERE:
    sys.modules.pop("strategy", None)

from botcore import safe_float, safe_nonneg_float, safe_pos_float
from strategy import Strategy, StratParams, HL_FEE_PCT
from offline.backtests.execution import (
    ExecutionModel,
    choose_intrabar_scenario,
    split_order_fill,
)
from offline.backtests.metrics import calculate_performance_metrics


def funding_at(fund: list[tuple[int, float]] | None, t_ms: int) -> float:
    """Return the funding rate active at t_ms, using the most recent rate at or before t.

    Automatically normalises Unix-second timestamps (< 1e10) to milliseconds so
    both direct replay callers (ms) and walk-forward evaluation contexts (s) match.
    """
    if not fund:
        return 0.0
    t_val = safe_float(t_ms, default=None)
    if t_val is None:
        return 0.0
    clean_fund: list[tuple[float, float]] = []
    for item in fund:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        ts_f = safe_float(item[0], default=None)
        rate_f = safe_float(item[1], default=None)
        if ts_f is not None and rate_f is not None:
            if 0 < ts_f < 10_000_000_000:
                ts_f *= 1000.0
            clean_fund.append((ts_f, rate_f))
    if not clean_fund:
        return 0.0
    if any(clean_fund[i][0] > clean_fund[i + 1][0] for i in range(len(clean_fund) - 1)):
        clean_fund.sort(key=lambda item: item[0])
    if 0 < t_val < 10_000_000_000 and clean_fund[0][0] >= 10_000_000_000:
        t_val = t_val * 1000
    lo, hi = 0, len(clean_fund)
    while lo < hi:
        mid = (lo + hi) // 2
        if clean_fund[mid][0] <= t_val:
            lo = mid + 1
        else:
            hi = mid
    return clean_fund[lo - 1][1] if lo > 0 else clean_fund[0][1]


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
    execution: ExecutionModel | None = None,
    fee_pct: float | None = None,
) -> dict:
    """Replay historical candles through the live Strategy engine."""
    if not ohlc:
        init_cap = 500.0
        if isinstance(params, StratParams):
            init_cap = float(params.max_budget)
        elif isinstance(params, dict):
            raw_b = params.get("budget") if params.get("budget") is not None else params.get("max_budget", 500.0)
            init_cap = safe_pos_float(raw_b, default=500.0)
        raw_ccy = (
            getattr(params, "currency", "USDC")
            if isinstance(params, StratParams)
            else params.get("currency", "USDC")
            if isinstance(params, dict)
            else "USDC"
        )
        ccy = (
            str(raw_ccy).strip().upper()
            if raw_ccy is not None and not isinstance(raw_ccy, bool) and str(raw_ccy).strip()
            else "USDC"
        )
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
            "fills": 0,
            "ambiguous_bars": 0,
            "initial_capital": init_cap,
            "account_currency": ccy,
            "return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "net_pnl": 0.0,
        }

    model = execution or ExecutionModel()
    return choose_intrabar_scenario(
        model,
        lambda scenario: _run_once(
            ohlc,
            ts=ts,
            fund=fund,
            sig=sig,
            params=params,
            direction=direction,
            coin=coin,
            bar_minutes=bar_minutes,
            periods_per_year=periods_per_year,
            execution=scenario,
            fee_pct=fee_pct,
        ),
    )


def _run_once(
    ohlc: list[tuple[float, float, float, float]],
    ts: list[int] | None = None,
    fund: list[tuple[int, float]] | None = None,
    sig: list[str] | None = None,
    params: StratParams | dict | None = None,
    direction: str = "long",
    coin: str = "HYPE",
    bar_minutes: float | None = 60.0,
    periods_per_year: float | None = None,
    execution: ExecutionModel | None = None,
    fee_pct: float | None = None,
) -> dict:
    import dataclasses
    exec_model = execution or ExecutionModel()
    if isinstance(params, StratParams):
        strat_params = dataclasses.replace(params)
    elif isinstance(params, dict):
        strat_params = StratParams.from_dict(params, direction=direction)
    else:
        strat_params = StratParams(direction=direction)

    if fee_pct is not None:
        fee_f = safe_nonneg_float(fee_pct, default=None)
        if fee_f is None or not isinstance(fee_pct, (int, float)):
            raise ValueError(f"invalid fee_pct: {fee_pct!r}")
        strat_params.fee_pct = fee_f

    strategy = Strategy(
        client=None,
        coin=coin,
        params=strat_params,
        dry_run=True,
        replay_mode=True,
    )

    for idx, row in enumerate(ohlc):
        if not isinstance(row, (list, tuple)) or len(row) < 4:
            raise ValueError(f"invalid OHLC bar at index {idx}")
        if any(isinstance(v, bool) for v in row[:4]):
            raise ValueError(f"OHLC bar at index {idx} contains bool")
        try:
            o_f, h_f, l_f, c_f = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"OHLC bar at index {idx} has non-numeric prices") from exc
        if (
            not all(math.isfinite(v) and v > 0 for v in (o_f, h_f, l_f, c_f))
            or h_f < l_f
            or h_f < max(o_f, c_f)
            or l_f > min(o_f, c_f)
        ):
            raise ValueError(f"OHLC bar at index {idx} has non-finite or invalid prices")

    initial_capital = float(strat_params.max_budget)
    equity_curve = [initial_capital]
    exposure: list[bool] = []
    trade_pnls: list[float] = []
    cycle_net_start = 0.0
    turnover_notional = 0.0
    fills = 0
    funding = 0.0
    ambiguous_bars = 0
    c = float(ohlc[0][3])

    open_side = "buy" if strategy.sign > 0 else "sell"
    close_side = "sell" if strategy.sign > 0 else "buy"
    order_placed_bar: dict[str, int] = {}
    valid_bar_minutes = safe_pos_float(bar_minutes, default=None)
    eff_bar_minutes = valid_bar_minutes if valid_bar_minutes is not None else 60.0

    for i in range(len(ohlc)):
        o, h, l, c = float(ohlc[i][0]), float(ohlc[i][1]), float(ohlc[i][2]), float(ohlc[i][3])
        t_ms = ts[i] if ts and i < len(ts) else 0

        # 1. Funding on held perpetuals: shorts receive funding; longs pay it.
        qty = strategy.s["qty"]
        if qty > 1e-12:
            rate = funding_at(fund, t_ms)
            f = rate * qty * c * (eff_bar_minutes / 60.0)
            funding += (f if strategy.sign < 0 else -f)

        # 2. Check active limit orders against the bar range
        open_order = next((x for x in list(strategy.s["orders"]) if x["role"] == "open"), None)
        open_eligible = False
        if open_order:
            open_eligible = exec_model.limit_touched(
                open_side, high=h, low=l, limit=open_order["px"]
            )

        close_order = next((x for x in list(strategy.s["orders"]) if x["role"] == "close"), None)
        close_eligible = False
        if close_order and strategy.s["qty"] > 1e-12:
            close_eligible = exec_model.limit_touched(
                close_side, high=h, low=l, limit=close_order["px"]
            )

        if open_eligible and close_eligible:
            ambiguous_bars += 1

        for side in exec_model.side_order():
            if side == open_side and open_eligible and open_order in strategy.s["orders"]:
                px = open_order["px"]
                fill_order, sz, complete = split_order_fill(
                    open_order,
                    quantity_key="sz",
                    amount_key="amount",
                    ratio=exec_model.partial_fill_ratio,
                    force_full=False,
                )
                if complete:
                    strategy._remove(open_order)
                    order_placed_bar.pop(open_order.get("oid", ""), None)
                strategy._apply_open(sz, px, fill_order.get("amount", 0.0), fill_order.get("kind"))
                turnover_notional += sz * px
                fills += 1

            if side == close_side and close_eligible and close_order in strategy.s["orders"] and strategy.s["qty"] > 1e-12:
                px = close_order["px"]
                fill_order, sz, complete = split_order_fill(
                    close_order,
                    quantity_key="sz",
                    amount_key=None,
                    ratio=exec_model.partial_fill_ratio,
                    force_full=False,
                )
                sz = min(sz, strategy.s["qty"])
                if sz <= 1e-12:
                    strategy._remove(close_order)
                    continue
                if complete:
                    strategy._remove(close_order)
                avg = strategy._avg() or px
                cycle_before = strategy.s.get("cycles_completed", 0)
                strategy._apply_close(avg, px, sz)
                turnover_notional += sz * px
                fills += 1
                if strategy.s.get("cycles_completed", 0) > cycle_before:
                    trade_pnls.append(strategy.s["realized_net"] - cycle_net_start)
                    cycle_net_start = strategy.s["realized_net"]

        # 3. Intrabar Stop-Loss Check (if held after limit fills)
        if strategy.s["qty"] > 1e-12 and strategy.p.stop_loss_pct > 0:
            worst_px = l if strategy.sign > 0 else h
            avg = strategy._avg()
            if avg:
                loss_pct = -strategy.sign * (worst_px - avg) / avg * 100.0
                if loss_pct >= strategy.p.stop_loss_pct - 1e-9:
                    sl_level = avg * (1.0 - strategy.sign * strategy.p.stop_loss_pct / 100.0)
                    raw_exec_px = min(o, sl_level) if strategy.sign > 0 else max(o, sl_level)
                    exec_px = exec_model.market_price(close_side, raw_exec_px)
                    held_qty = strategy.s["qty"]
                    cycle_before = strategy.s.get("cycles_completed", 0)
                    if strategy._check_stop_loss(exec_px):
                        turnover_notional += held_qty * exec_px
                        fills += 1
                        if strategy.s.get("cycles_completed", 0) > cycle_before:
                            trade_pnls.append(strategy.s["realized_net"] - cycle_net_start)
                            cycle_net_start = strategy.s["realized_net"]

        # 4. TTL / Runaway price cancellation
        open_order = next((x for x in list(strategy.s["orders"]) if x["role"] == "open"), None)
        if open_order:
            oid = open_order.get("oid", "")
            placed_b = order_placed_bar.setdefault(oid, i)
            elapsed_min = (i - placed_b) * eff_bar_minutes
            if strategy._open_is_stale(open_order, c, elapsed_min=elapsed_min):
                strategy._remove(open_order)
                order_placed_bar.pop(oid, None)

        # 5. Strategy decisions at bar close
        signal_val = sig[i] if sig and i < len(sig) else "__any__"
        cycle_before_step = strategy.s.get("cycles_completed", 0)
        strategy.step(c, signal=signal_val)
        if strategy.s.get("cycles_completed", 0) > cycle_before_step:
            trade_pnls.append(strategy.s["realized_net"] - cycle_net_start)
            cycle_net_start = strategy.s["realized_net"]
            fills += 1

        for o_ord in strategy.s["orders"]:
            oid = o_ord.get("oid", "")
            if o_ord["role"] == "open" and oid not in order_placed_bar:
                order_placed_bar[oid] = i

        # 6. Equity curve and exposure tracking
        qty = strategy.s["qty"]
        cost = strategy.s["cost"]
        realized = strategy.s["realized_net"]
        upnl = (strategy.sign * (c - cost / qty) * qty) if qty > 1e-12 else 0.0
        fee_pct_eff = getattr(strategy.p, "fee_pct", HL_FEE_PCT)
        open_fee = (fee_pct_eff / 100.0) * cost if qty > 1e-12 else 0.0
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
    final_open_fee = 0.0
    if qty > 1e-12:
        fee_pct_eff = getattr(strategy.p, "fee_pct", HL_FEE_PCT)
        final_open_fee = (fee_pct_eff / 100.0) * cost
        total_fees += final_open_fee

    if periods_per_year is None and valid_bar_minutes is not None:
        periods_per_year = 365.0 * 24 * 60 / valid_bar_minutes

    performance = calculate_performance_metrics(
        equity_curve,
        initial_capital=initial_capital,
        periods_per_year=periods_per_year,
        exposure=exposure,
        trade_pnls=trade_pnls,
        turnover_notional=turnover_notional,
    )

    rounded = lambda v: round(float(v), 10) if not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v) else v
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
        "total": rounded(realized + funding + final_upnl - final_open_fee),
        "fills": fills,
        "ambiguous_bars": ambiguous_bars,
        "initial_capital": rounded(initial_capital),
        "account_currency": strat_params.currency,
    }
    result.update({
        key: rounded(value)
        for key, value in performance.items()
    })
    return result


def make_hyperliquid_spot_replay_adapter(params: Any, fee_pct: float = 0.035, **kwargs):
    """Return a spot ReplayFunction using strategies.spot_replay (SSOT) compatible with evaluation.py."""
    from strategies.spot_replay import make_spot_replay_adapter

    return make_spot_replay_adapter(params, fee_pct=fee_pct, **kwargs)


def make_hyperliquid_replay_adapter(
    params: StratParams | dict,
    direction: str = "long",
    coin: str = "HYPE",
    fund: list[tuple[int, float]] | None = None,
    bar_minutes: float | None = 60.0,
    execution: ExecutionModel | None = None,
    fee_pct: float | None = None,
    engine: str = "perp",
):
    """Return a generic ReplayFunction compatible with evaluation.py walk-forward benchmarks.

    Supports both 'perp' (Strategy engine) and 'spot' (strategies.spot_replay SSOT).
    """
    engine_norm = str(engine or "perp").strip().lower() if not isinstance(engine, bool) else ""
    if engine_norm not in ("perp", "spot"):
        raise ValueError(f"Unsupported Hyperliquid replay engine: {engine!r}")
    if engine_norm == "spot":
        return make_hyperliquid_spot_replay_adapter(
            params=params,
            fee_pct=fee_pct if fee_pct is not None else 0.035,
            bar_minutes=bar_minutes,
            execution=execution,
        )

    def _adapter(ohlc, _warmup_ohlc=None, context=None):
        ts = list(context.timestamps) if context and getattr(context, "timestamps", None) else None
        return run_replay(
            ohlc,
            ts=ts,
            fund=fund,
            params=params,
            direction=direction,
            coin=coin,
            bar_minutes=bar_minutes,
            execution=execution,
            fee_pct=fee_pct,
        )
    return _adapter
