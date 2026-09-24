"""Faithful OHLC replay for the live Trading 212 engine.

``strategy.Strategy.step`` produces decisions. This file models only fills, time,
FX, and metrics. Orders decided at close can execute no earlier than the next bar.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
import math
import os
import sys
from unittest.mock import MagicMock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import strategy as _strat
from offline.backtests.execution import (
    ExecutionModel,
    choose_intrabar_scenario,
    split_order_fill,
)
from offline.backtests.metrics import calculate_performance_metrics


class _SimClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def _ols_slope_pct(closes: list[float]) -> float | None:
    values = closes[-12:]
    if len(values) < 6:
        return None
    n = len(values)
    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n
    denominator = sum((index - mean_x) ** 2 for index in range(n))
    if denominator == 0 or mean_y == 0:
        return None
    slope = sum(
        (index - mean_x) * (value - mean_y)
        for index, value in enumerate(values)
    ) / denominator
    return slope / mean_y * 100.0


def _fx_rates(value: float | Sequence[float], count: int) -> list[float]:
    if isinstance(value, (int, float)):
        rates = [float(value)] * count
    else:
        rates = [float(rate) for rate in value]
        if len(rates) != count:
            raise ValueError(f"the FX series has {len(rates)} values for {count} bars")
    if any(rate <= 0 for rate in rates):
        raise ValueError("all fx_to_usd values must be positive")
    return rates


def run_replay(
    ohlc,
    params: _strat.StratParams,
    *,
    bar_minutes: float | None = None,
    fx_to_usd: float | Sequence[float] = 1.0,
    periods_per_year: float | None = None,
    warmup_ohlc=(),
    timestamps: Sequence[int] = (),
    execution: ExecutionModel | None = None,
) -> dict:
    """Run the live T212 strategy over ``(open, high, low, close)`` bars.

    ``fx_to_usd`` may be scalar or one historical value per bar. The live DCA gate
    reads five-minute Yahoo bars and rejects other cadences.
    """
    if not ohlc:
        raise ValueError("ohlc cannot be empty")
    if params.dca_trend_gate_pct > 0 and bar_minutes != 5:
        raise ValueError(
            "dca_trend_gate_pct requires 5-minute bars, the same cadence as the live Yahoo signal"
        )
    if timestamps and len(timestamps) != len(ohlc):
        raise ValueError("timestamps must have the same length as ohlc")
    rates = _fx_rates(fx_to_usd, len(ohlc))
    model = execution or ExecutionModel()
    return choose_intrabar_scenario(
        model,
        lambda scenario: _run_once(
            ohlc,
            params,
            bar_minutes=bar_minutes,
            fx_rates=rates,
            periods_per_year=periods_per_year,
            warmup_ohlc=warmup_ohlc,
            timestamps=timestamps,
            execution=scenario,
        ),
    )


def _run_once(
    ohlc,
    params: _strat.StratParams,
    *,
    bar_minutes: float | None,
    fx_rates: list[float],
    periods_per_year: float | None,
    warmup_ohlc,
    timestamps: Sequence[int],
    execution: ExecutionModel,
) -> dict:
    client = MagicMock()
    clock = _SimClock()
    trend_closes: deque[float] = deque(maxlen=12)
    trend_closes.extend(float(row[3]) for row in warmup_ohlc)
    original_log = _strat.log
    original_notify = _strat.notify
    _strat.log = lambda *_args, **_kwargs: None
    _strat.notify = lambda *_args, **_kwargs: None
    try:
        engine = _strat.Strategy(
            client,
            "REPLAY_US_EQ",
            params,
            dry_run=True,
            initial_state=_strat._new_state(),
            fx_to_usd=fx_rates[0],
            clock=clock,
            trend_slope_provider=lambda _symbol: _ols_slope_pct(list(trend_closes)),
        )
        engine._save = lambda: None
        # The curve uses account currency. Asset P&L is produced in USD and converted
        # at the available rate when realized or marked to market.
        initial_capital = float(params.max_budget)
        equity_curve = [initial_capital]
        exposure = []
        trade_pnls = []
        turnover_account = 0.0
        realized_net_account = 0.0
        fills = wins = ambiguous_bars = 0
        cycle0 = engine.s.get("cycle", 1)

        for bar_index, (open_, high, low, close) in enumerate(ohlc):
            rate = fx_rates[bar_index]
            clock.value = (
                float(timestamps[bar_index]) if timestamps
                else bar_index * bar_minutes * 60 if bar_minutes
                else float(bar_index)
            )

            def eligible(order: dict) -> bool:
                if order.get("market"):
                    return True
                return execution.limit_touched(
                    order["side"], high=high, low=low, limit=order["limit"],
                )

            eligible_sides = {
                order["side"].lower()
                for order in engine.s["orders"]
                if eligible(order)
            }
            if {"buy", "sell"}.issubset(eligible_sides):
                ambiguous_bars += 1

            for side in execution.side_order():
                for order in list(engine.s["orders"]):
                    if order not in engine.s["orders"] or order["side"].lower() != side:
                        continue
                    if not eligible(order):
                        continue
                    market = bool(order.get("market"))
                    fill_order, quantity, complete = split_order_fill(
                        order,
                        quantity_key="qty",
                        amount_key="amount" if side == "buy" else None,
                        ratio=execution.partial_fill_ratio,
                        force_full=market,
                    )
                    if complete:
                        engine._remove_order(order)
                    price = (
                        execution.market_price(side, open_)
                        if market else float(order["limit"])
                    )
                    if side == "sell":
                        quantity = min(quantity, engine.s["qty"])
                        if quantity <= 1e-12:
                            engine._remove_order(order)
                            continue
                        fill_order["qty"] = quantity
                        avg = engine._avg_cost() or price
                        _gross, _fee, net = _strat._sell_pnl(
                            avg, price, quantity, params.fx_fee_pct,
                        )
                        if complete and fill_order.get("level") is not None:
                            engine.s.setdefault("tp_sold_levels", []).append(
                                fill_order["level"]
                            )
                        realized_net_account += net / rate
                        wins += int(net > 0)
                        trade_pnls.append(net / rate)
                    engine._apply_fill(fill_order, quantity, price)
                    fills += 1
                    turnover_account += quantity * price / rate

            # Live _reconcile_real() cancels a stale resting BUY and step() places it
            # again at the current price; replay never reconciles, so apply the same rule.
            for order in list(engine.s["orders"]):
                if engine._buy_is_stale(order, float(close)):
                    engine._remove_order(order)
            trend_closes.append(float(close))
            engine.fx_to_usd = rate
            engine.step(float(close))

            qty = engine.s["qty"]
            avg = engine._avg_cost()
            unrealized_usd = (float(close) - avg) * qty if avg and qty > 1e-9 else 0.0
            open_buy_fee_usd = params.fx_fee_pct / 100.0 * engine.s["cost_usd"]
            equity_curve.append(
                initial_capital
                + realized_net_account
                + (unrealized_usd - open_buy_fee_usd) / rate
            )
            exposure.append(qty > 1e-9)
    finally:
        _strat.log = original_log
        _strat.notify = original_notify

    if periods_per_year is None and bar_minutes:
        periods_per_year = 365.0 * 24 * 60 / bar_minutes
    performance = calculate_performance_metrics(
        equity_curve,
        initial_capital=initial_capital,
        periods_per_year=periods_per_year,
        exposure=exposure,
        trade_pnls=trade_pnls,
        turnover_notional=turnover_account,
    )
    qty = engine.s["qty"]
    avg = engine._avg_cost()
    final_upnl_usd = (float(ohlc[-1][3]) - avg) * qty if avg and qty > 1e-9 else 0.0
    open_buy_fee_usd = params.fx_fee_pct / 100.0 * engine.s["cost_usd"]
    rounded = lambda value: round(float(value), 10)
    result = {
        "realized": rounded(engine.s["realized_pnl_usd"]),
        "net": rounded(engine.s["realized_net_usd"]),
        "fees": rounded(engine.s["fees_usd"] + open_buy_fee_usd),
        "total": rounded(engine.s["realized_net_usd"] + final_upnl_usd - open_buy_fee_usd),
        "final_upnl": rounded(final_upnl_usd),
        "open_buy_fee": rounded(open_buy_fee_usd),
        "account_currency": params.currency,
        "cycles": engine.s.get("cycle", 1) - cycle0,
        "wins": wins,
        "maxdd": rounded(performance["max_drawdown_abs"]),
        "open_qty": rounded(qty),
        "fills": fills,
        "ambiguous_bars": ambiguous_bars,
    }
    result.update({
        key: (rounded(value) if isinstance(value, float) and math.isfinite(value) else value)
        for key, value in performance.items()
    })
    return result
