"""Replay the live Kraken strategy over historical OHLC.

The strategy decides orders. The harness separately models fills, costs, and intrabar
ambiguity without network, notifications, or persistent state.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

from offline.backtests.execution import (
    ExecutionModel,
    FeeModel,
    choose_intrabar_scenario,
    split_order_fill,
)
from offline.backtests.metrics import calculate_performance_metrics
from strategies import spot_dca as _strat
from providers.strategy_executor import PairPrecision


def _silent(*_args, **_kwargs):
    return None


def _validate_replay(ohlc, params, bar_minutes: float | None) -> None:
    if not ohlc:
        raise ValueError("ohlc cannot be empty")
    if (params.trend_overlay or params.dca_trend_brake
            or (params.tp_trend_hold and params.tp_regime_gate)) and (
            bar_minutes is None or float(bar_minutes) != float(params.trend_interval)):
        raise ValueError(
            "regime-aware policies require bar_minutes to equal trend_interval "
            f"({params.trend_interval} minutes); resampling is not implemented"
        )
    if params.tp_trail_adaptive and (
            bar_minutes is None
            or float(bar_minutes) != float(params.tp_trail_vol_interval)):
        raise ValueError(
            "tp_trail_adaptive requires bar_minutes to equal "
            f"tp_trail_vol_interval ({params.tp_trail_vol_interval} minute)"
        )
    if params.dca_vol_scale_k and (
            bar_minutes is None
            or float(bar_minutes) != float(params.dca_vol_interval)):
        raise ValueError(
            "dca_vol_scale_k requires bar_minutes to equal "
            f"dca_vol_interval ({params.dca_vol_interval} minute)"
        )
    if params.reentry_adaptive and bar_minutes is None:
        raise ValueError("reentry_adaptive requires bar_minutes for the time-based volatility")


def run_replay(
    ohlc,
    params,
    fee_pct: float = 0.26,
    bar_minutes: float | None = None,
    warmup_ohlc=(),
    execution: ExecutionModel | None = None,
    fee_model: FeeModel | None = None,
    include_decision_trace: bool = False,
    initial_cash: float | None = None,
) -> dict:
    """Replay decisions, optionally enforcing a finite cash account and fee reserves.

    Omitted cash retains legacy funding assumptions for baseline compatibility.
    Explicit cash rejects unaffordable requests, reserves open BUYs, and settles
    fill notional/fees; it does not inject capital after losses or cycle resets.
    """
    _validate_replay(ohlc, params, bar_minutes)
    if initial_cash is not None and (isinstance(initial_cash, bool)
            or not math.isfinite(initial_cash) or initial_cash <= 0):
        raise ValueError("initial_cash must be finite and positive")
    model = execution or ExecutionModel()
    fees = fee_model or FeeModel(fee_pct, fee_pct)
    return choose_intrabar_scenario(
        model,
        lambda scenario: _run_once(
            ohlc,
            params,
            fee_model=fees,
            bar_minutes=bar_minutes,
            warmup_ohlc=warmup_ohlc,
            execution=scenario,
            include_decision_trace=include_decision_trace,
            initial_cash=initial_cash,
        ),
    )


def _run_once(
    ohlc,
    params,
    *,
    fee_model: FeeModel,
    bar_minutes: float | None,
    warmup_ohlc,
    execution: ExecutionModel,
    include_decision_trace: bool,
    initial_cash: float | None,
) -> dict:
    client = MagicMock()
    # Replay declares its synthetic execution precision explicitly; live engines
    # must obtain this metadata from the venue.
    client.pair_precision.return_value = PairPrecision(5, 8, 0.0, "REPLAY")
    # Legacy replay explicitly assumes funding up to the configured cycle cap.
    # A mock or unavailable live balance must never imply unlimited buying power.
    client.free_balance.return_value = float(params.effective_max_budget())
    original_notify = _strat.notify
    _strat.notify = _silent
    try:
        strategy = _strat.Strategy(
            client,
            "REPLAY",
            params,
            dry_run=True,
            initial_state=_strat._new_state(),
            replay_mode=True,
        )
        warmup_step = bar_minutes * 60 if bar_minutes else 1.0
        for index, (_open, _high, _low, close) in enumerate(
                warmup_ohlc, start=-len(warmup_ohlc)):
            strategy._shadow_prices.append((index * warmup_step, float(close)))
        strategy._save = _silent
        decision_trace = []
        current_bar = -1
        original_place = strategy._place
        cash = float(initial_cash) if initial_cash is not None else None
        min_cash = cash
        cash_refusals = 0

        def free_quote(_asset):
            reserved = sum(o["vol"] * o["price"] *
                           (1 + fee_model.rate_pct(market=bool(o.get("market"))) / 100)
                           for o in strategy.s["orders"] if o["side"] == "buy")
            return max(0.0, cash - reserved)

        if cash is not None:
            client.free_balance.side_effect = free_quote

        def traced_place(side, vol, price, kind, amount=0.0, market=False):
            nonlocal cash_refusals
            if cash is not None and side == "buy":
                required = round(vol, strategy.vol_dec) * round(price, strategy.price_dec)
                required *= 1 + fee_model.rate_pct(market=market) / 100
                if required > free_quote(params.currency) + 1e-9:
                    cash_refusals += 1
                    return False
            accepted = original_place(
                side, vol, price, kind, amount=amount, market=market,
            )
            if accepted and include_decision_trace:
                decision_trace.append({
                    "bar": current_bar,
                    "side": side,
                    "kind": kind,
                    "market": bool(market),
                    "price": round(float(price), 10),
                    "qty": round(float(vol), 10),
                })
            return accepted

        if include_decision_trace or cash is not None:
            strategy._place = traced_place
        cycle0 = strategy.s.get("cycle", 1)
        wins = fill_count = ambiguous_bars = 0
        turnover_notional = 0.0
        trade_pnls = []
        cycle_net_start = strategy.s["realized_net"]
        initial_capital = float(params.effective_max_budget() if cash is None else cash)
        equity_curve = [initial_capital]
        exposure = []

        for bar_index, (open_, high, low, close) in enumerate(ohlc):
            current_bar = bar_index
            def eligible(order: dict) -> bool:
                if order.get("market"):
                    return True
                return execution.limit_touched(
                    order["side"], high=high, low=low, limit=order["price"],
                )

            eligible_sides = {
                order["side"].lower()
                for order in strategy.s["orders"]
                if eligible(order)
            }
            if {"buy", "sell"}.issubset(eligible_sides):
                ambiguous_bars += 1

            for side in execution.side_order():
                for order in list(strategy.s["orders"]):
                    if order not in strategy.s["orders"] or order["side"] != side:
                        continue
                    if not eligible(order):
                        continue
                    market = bool(order.get("market"))
                    fill_order, volume, complete = split_order_fill(
                        order,
                        quantity_key="vol",
                        amount_key="amount" if side == "buy" else None,
                        ratio=execution.partial_fill_ratio,
                        force_full=market,
                    )
                    if complete:
                        strategy._remove(order)
                    price = (
                        execution.market_price(side, open_)
                        if market else float(order["price"])
                    )
                    if side == "sell":
                        volume = min(volume, strategy.s["qty"])
                        if volume <= 1e-12:
                            strategy._remove(order)
                            continue
                        fill_order["vol"] = volume
                        gross_before = strategy.s["realized_gross"]
                    fee = fee_model.rate_pct(market=market) / 100.0 * volume * price
                    if cash is not None:
                        cash += (-1 if side == "buy" else 1) * volume * price - fee
                        min_cash = min(min_cash, cash)
                        if cash < -1e-8:
                            raise ValueError("cash account overdrawn by simulated execution")
                    strategy._apply_fill(
                        fill_order, volume, price, fee=fee, final=complete,
                    )
                    fill_count += 1
                    turnover_notional += volume * price
                    if side == "sell":
                        if strategy.s["realized_gross"] > gross_before:
                            wins += 1
                        if strategy.s.get("cycle", 1) > cycle0 + len(trade_pnls):
                            trade_pnls.append(
                                strategy.s["realized_net"] - cycle_net_start
                            )
                            cycle_net_start = strategy.s["realized_net"]

            # Live reconcile() cancels an unfilled limit BUY once it is older than
            # order_ttl_min and the price has risen more than 0.3% above it; step() then
            # re-places it at the current level. Replay never calls reconcile(), so without
            # this an ENTRY left below a rally would wait forever for a pullback.
            if bar_minutes is None or bar_minutes >= params.order_ttl_min:
                for order in list(strategy.s["orders"]):
                    if (order["side"] == "buy" and not order.get("market")
                            and close > order["price"] * 1.003):
                        strategy._remove(order)
            replay_time = bar_index * bar_minutes * 60 if bar_minutes else bar_index
            strategy.step(close, timestamp=replay_time)
            qty = strategy.s["qty"]
            unrealized = (
                (close - strategy.s["cost"] / qty) * qty if qty > 1e-12 else 0.0
            )
            equity_curve.append(initial_capital + strategy.s["realized_net"] + unrealized)
            exposure.append(qty > 1e-12)
    finally:
        _strat.notify = original_notify

    qty = strategy.s["qty"]
    final_upnl = (
        (ohlc[-1][3] - strategy.s["cost"] / qty) * qty if qty > 1e-12 else 0.0
    )
    periods_per_year = 365.0 * 24 * 60 / bar_minutes if bar_minutes else None
    performance = calculate_performance_metrics(
        equity_curve,
        initial_capital=initial_capital,
        periods_per_year=periods_per_year,
        exposure=exposure,
        trade_pnls=trade_pnls,
        turnover_notional=turnover_notional,
    )
    rounded = lambda value: round(value, 10)
    result = {
        "realized": rounded(strategy.s["realized_gross"]),
        "net": rounded(strategy.s["realized_net"]),
        "fees": rounded(strategy.s["fees_total"]),
        "total": rounded(strategy.s["realized_net"] + final_upnl),
        "final_upnl": rounded(final_upnl),
        "cycles": strategy.s.get("cycle", 1) - cycle0,
        "wins": wins,
        "maxdd": rounded(performance["max_drawdown_abs"]),
        "open_qty": rounded(qty),
        "fills": fill_count,
        "ambiguous_bars": ambiguous_bars,
    }
    result.update({
        key: (round(value, 10) if isinstance(value, float) else value)
        for key, value in performance.items()
    })
    if include_decision_trace:
        result["decision_trace"] = decision_trace
    if cash is not None:
        result["funding"] = {"initial_cash": initial_cash, "final_cash": rounded(cash),
                             "minimum_cash": rounded(min_cash), "refused_buys": cash_refusals}
    return result
