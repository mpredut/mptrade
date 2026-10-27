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

_T212_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_T212_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if _T212_DIR not in sys.path:
    sys.path.append(_T212_DIR)

import importlib.util
_md_spec = importlib.util.spec_from_file_location("t212_market_data", os.path.join(_T212_DIR, "market_data.py"))
_md = importlib.util.module_from_spec(_md_spec)
sys.modules[_md_spec.name] = _md
_prev_md = sys.modules.get("market_data")
sys.modules["market_data"] = _md
_md_spec.loader.exec_module(_md)
try:
    _strat_spec = importlib.util.spec_from_file_location("t212_strategy_replay", os.path.join(_T212_DIR, "strategy.py"))
    _strat = importlib.util.module_from_spec(_strat_spec)
    sys.modules[_strat_spec.name] = _strat
    _strat_spec.loader.exec_module(_strat)
finally:
    if _prev_md is not None:
        sys.modules["market_data"] = _prev_md
    else:
        sys.modules.pop("market_data", None)
from offline.backtests.execution import (
    ExecutionModel,
    choose_intrabar_scenario,
    split_order_fill,
)
from offline.backtests.metrics import calculate_performance_metrics
from providers.replay_clock import SimClock as _SimClock


def _ols_slope_pct(closes: list[float]) -> float | None:
    clean_closes: list[float] = []
    for c in closes or []:
        if isinstance(c, bool):
            continue
        try:
            cf = float(c)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(cf) and cf > 0:
            clean_closes.append(cf)
    values = clean_closes[-12:]
    if len(values) < 6:
        return None
    n = len(values)
    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n
    denominator = sum((index - mean_x) ** 2 for index in range(n))
    if denominator == 0 or mean_y == 0 or not math.isfinite(mean_y):
        return None
    slope = sum(
        (index - mean_x) * (value - mean_y)
        for index, value in enumerate(values)
    ) / denominator
    res = slope / mean_y * 100.0
    return res if math.isfinite(res) else None


def _fx_rates(value: float | Sequence[float], count: int) -> list[float]:
    if isinstance(value, bool) or isinstance(count, bool) or count < 0:
        raise ValueError("fx_to_usd and count must be valid non-bool values")
    if isinstance(value, (int, float, str)):
        try:
            scalar = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid scalar fx_to_usd: {value!r}") from exc
        rates = [scalar] * count
    else:
        rates = []
        try:
            iterator = iter(value)
        except TypeError as exc:
            raise ValueError(f"invalid fx_to_usd sequence: {value!r}") from exc
        for rate in iterator:
            if isinstance(rate, bool):
                raise ValueError("fx_to_usd values cannot be bool")
            try:
                rates.append(float(rate))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"invalid fx_to_usd element: {rate!r}") from exc
        if len(rates) != count:
            raise ValueError(f"the FX series has {len(rates)} values for {count} bars")
    if any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError("all fx_to_usd values must be finite and positive")
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
    if not isinstance(ohlc, (list, tuple)) or not ohlc:
        raise ValueError("ohlc cannot be empty")
    if bar_minutes is not None:
        try:
            bm_f = float(bar_minutes)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("bar_minutes must be a finite positive number") from exc
        if isinstance(bar_minutes, bool) or not math.isfinite(bm_f) or bm_f <= 0:
            raise ValueError("bar_minutes must be a finite positive number")
        bar_minutes = bm_f
    if periods_per_year is not None:
        try:
            ppy_f = float(periods_per_year)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("periods_per_year must be a finite positive number") from exc
        if isinstance(periods_per_year, bool) or not math.isfinite(ppy_f) or ppy_f <= 0:
            raise ValueError("periods_per_year must be a finite positive number")
        periods_per_year = ppy_f
    for idx, row in enumerate(ohlc):
        if not isinstance(row, (list, tuple)) or len(row) != 4:
            raise ValueError(f"invalid OHLC bar at index {idx}")
        if any(isinstance(v, bool) for v in row):
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
    if params.dca_trend_gate_pct > 0 and bar_minutes != 5:
        raise ValueError(
            "dca_trend_gate_pct requires 5-minute bars, the same cadence as the live Yahoo signal"
        )
    if timestamps:
        if len(timestamps) != len(ohlc):
            raise ValueError("timestamps must have the same length as ohlc")
        for idx, ts_val in enumerate(timestamps):
            if isinstance(ts_val, bool):
                raise ValueError(f"timestamp at index {idx} cannot be bool")
            try:
                ts_f = float(ts_val)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"timestamp at index {idx} is not numeric") from exc
            if not math.isfinite(ts_f):
                raise ValueError(f"timestamp at index {idx} must be finite")
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


class _ReplayClient:
    __slots__ = ()


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
    client = _ReplayClient()
    clock = _SimClock(0.0)
    trend_closes: deque[float] = deque(maxlen=12)
    for row in warmup_ohlc or ():
        if not isinstance(row, (list, tuple)) or len(row) < 4 or isinstance(row[3], bool):
            continue
        try:
            cf = float(row[3])
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(cf) and cf > 0:
            trend_closes.append(cf)
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
        side_order = execution.side_order()

        for bar_index, row in enumerate(ohlc):
            open_, high, low, close = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            rate = fx_rates[bar_index]
            clock.value = (
                float(timestamps[bar_index]) if timestamps
                else bar_index * bar_minutes * 60 if bar_minutes
                else float(bar_index)
            )

            orders = engine.s["orders"]
            if orders:
                eligible_orders = [
                    o for o in orders
                    if o.get("market") or execution.limit_touched(
                        o["side"], high=high, low=low, limit=o["limit"],
                    )
                ]
                if eligible_orders:
                    if len(eligible_orders) > 1 and {"buy", "sell"}.issubset(
                        {o["side"].lower() for o in eligible_orders}
                    ):
                        ambiguous_bars += 1

                    for side in side_order:
                        for order in eligible_orders:
                            if order not in engine.s["orders"] or order["side"].lower() != side:
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
                if engine.s["orders"]:
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
    rounded = lambda value: round(float(value), 10) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else value
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
        "initial_capital": rounded(initial_capital),
    }
    result.update({
        key: (rounded(value) if isinstance(value, float) and math.isfinite(value) else value)
        for key, value in performance.items()
    })
    return result


def make_t212_replay_adapter(
    params: _strat.StratParams,
    bar_minutes: float | None = None,
    fx_to_usd: float | Sequence[float] | None = None,
    fx_records: Sequence[dict] | None = None,
    fx_invert: bool = False,
    periods_per_year: float | None = None,
    execution: ExecutionModel | None = None,
):
    """Return a generic ReplayFunction compatible with evaluation.py walk-forward benchmarks."""
    def _adapter(ohlc, warmup_ohlc, context=None):
        ts = list(context.timestamps) if context and getattr(context, "timestamps", None) else ()
        rate = fx_to_usd if fx_to_usd is not None else 1.0
        if fx_records is not None and ts:
            from offline.backtests.datasets import align_previous_values
            rate = align_previous_values(
                ts,
                fx_records,
                transform=(lambda value: 1.0 / value) if fx_invert else None,
            )
        return run_replay(
            ohlc,
            params,
            bar_minutes=bar_minutes,
            fx_to_usd=rate,
            periods_per_year=periods_per_year,
            warmup_ohlc=warmup_ohlc,
            timestamps=ts,
            execution=execution,
        )
    return _adapter

