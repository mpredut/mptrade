"""Faithful OHLC replay for the live Trading 212 engine.

``strategy.Strategy.step`` produces decisions. This file models only fills, time,
FX, and metrics. Orders decided at close can execute no earlier than the next bar.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
import os
import sys

_T212_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_T212_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from botcore import load_local_module, safe_float, safe_pos_float
_md = load_local_module(_T212_DIR, "market_data", module_key="t212_market_data")
_prev_md = sys.modules.get("market_data")
sys.modules["market_data"] = _md
try:
    _strat = load_local_module(_T212_DIR, "strategy", module_key="t212_strategy_replay")
finally:
    if _prev_md is not None:
        sys.modules["market_data"] = _prev_md
    else:
        sys.modules.pop("market_data", None)
from offline.backtests.datasets import align_previous_values
from offline.backtests.execution import (
    ExecutionModel,
    choose_intrabar_scenario,
    select_eligible_intrabar_orders,
    split_order_fill,
    validate_ohlc_bars,
)
from offline.backtests.metrics import round_metric
from providers.replay_provider import SimClock
from strategies.spot_replay import build_replay_summary


def _ols_slope_pct(closes: list[float]) -> float | None:
    return _md.compute_ols_slope_pct(closes, bars=12)


def _fx_rates(value: float | Sequence[float], count: int) -> list[float]:
    if isinstance(value, bool) or isinstance(count, bool) or count < 0:
        raise ValueError("fx_to_usd and count must be valid non-bool values")
    if isinstance(value, (int, float, str)):
        scalar = safe_pos_float(value, default=None)
        if scalar is None:
            raise ValueError(f"invalid scalar fx_to_usd: {value!r}")
        return [scalar] * count
    try:
        iterator = iter(value)
    except TypeError as exc:
        raise ValueError(f"invalid fx_to_usd sequence: {value!r}") from exc
    rates = []
    for rate in iterator:
        rate_f = safe_pos_float(rate, default=None)
        if rate_f is None:
            raise ValueError(f"invalid fx_to_usd element: {rate!r}")
        rates.append(rate_f)
    if len(rates) != count:
        raise ValueError(f"the FX series has {len(rates)} values for {count} bars")
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
    """Run the live T212 strategy over ``(open, high, low, close)`` bars."""
    if not isinstance(ohlc, (list, tuple)) or not ohlc:
        raise ValueError("ohlc cannot be empty")
    if bar_minutes is not None:
        bar_minutes = safe_pos_float(bar_minutes, default=None)
        if bar_minutes is None:
            raise ValueError("bar_minutes must be a finite positive number")
    if periods_per_year is not None:
        periods_per_year = safe_pos_float(periods_per_year, default=None)
        if periods_per_year is None:
            raise ValueError("periods_per_year must be a finite positive number")
    validate_ohlc_bars(ohlc, exact_len_4=True, check_envelope=True)
    if params.dca_trend_gate_pct > 0 and bar_minutes != 5:
        raise ValueError("dca_trend_gate_pct requires 5-minute bars, the same cadence as the live Yahoo signal")
    if timestamps:
        if len(timestamps) != len(ohlc):
            raise ValueError("timestamps must have the same length as ohlc")
        for idx, ts_val in enumerate(timestamps):
            if safe_float(ts_val, default=None) is None:
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
    clock = SimClock(0.0)
    trend_closes: deque[float] = deque(maxlen=12)
    for row in warmup_ohlc or ():
        if isinstance(row, (list, tuple)) and len(row) >= 4:
            cf = safe_pos_float(row[3], default=None)
            if cf is not None:
                trend_closes.append(cf)
    original_log, original_notify = _strat.log, _strat.notify
    _strat.log = _strat.notify = lambda *_args, **_kwargs: None
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
        initial_capital = float(params.max_budget)
        equity_curve = [initial_capital]
        exposure = []
        trade_pnls = []
        turnover_account = realized_net_account = 0.0
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
                eligible_orders, is_ambiguous = select_eligible_intrabar_orders(
                    orders, execution, high=high, low=low, limit_key="limit",
                )
                if eligible_orders:
                    ambiguous_bars += int(is_ambiguous)
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
                                engine._remove(order)
                            price = execution.market_price(side, open_) if market else float(order["limit"])
                            if side == "sell":
                                quantity = min(quantity, engine.s["qty"])
                                if quantity <= 1e-12:
                                    engine._remove(order)
                                    continue
                                fill_order["qty"] = quantity
                                avg = engine._avg() or price
                                _gross, _fee, net = _strat._sell_pnl(avg, price, quantity, params.fx_fee_pct)
                                if complete and fill_order.get("level") is not None:
                                    engine._record_tp_sold_level(fill_order["level"])
                                realized_net_account += net / rate
                                wins += int(net > 0)
                                trade_pnls.append(net / rate)
                            engine._apply_fill(fill_order, quantity, price)
                            fills += 1
                            turnover_account += quantity * price / rate

                if engine.s["orders"]:
                    for order in list(engine.s["orders"]):
                        if engine._buy_is_stale(order, float(close)):
                            engine._remove(order)
            trend_closes.append(float(close))
            engine.fx_to_usd = rate
            engine.step(float(close))

            qty = engine.s["qty"]
            avg = engine._avg()
            unrealized_usd = (float(close) - avg) * qty if avg and qty > 1e-9 else 0.0
            open_buy_fee_usd = params.fx_fee_pct / 100.0 * engine.s["cost_usd"]
            equity_curve.append(initial_capital + realized_net_account + (unrealized_usd - open_buy_fee_usd) / rate)
            exposure.append(qty > 1e-9)
    finally:
        _strat.log, _strat.notify = original_log, original_notify

    qty = engine.s["qty"]
    avg = engine._avg()
    final_upnl_usd = (float(ohlc[-1][3]) - avg) * qty if avg and qty > 1e-9 else 0.0
    open_buy_fee_usd = params.fx_fee_pct / 100.0 * engine.s["cost_usd"]
    return build_replay_summary(
        realized=engine.s["realized_pnl_usd"],
        net=engine.s["realized_net_usd"],
        fees=engine.s["fees_usd"] + open_buy_fee_usd,
        total=engine.s["realized_net_usd"] + final_upnl_usd - open_buy_fee_usd,
        final_upnl=final_upnl_usd,
        cycles=engine.s.get("cycle", 1) - cycle0,
        wins=wins,
        open_qty=qty,
        fills=fills,
        ambiguous_bars=ambiguous_bars,
        initial_capital=initial_capital,
        account_currency=params.currency,
        equity_curve=equity_curve,
        exposure=exposure,
        trade_pnls=trade_pnls,
        turnover_notional=turnover_account,
        bar_minutes=bar_minutes,
        periods_per_year=periods_per_year,
        extra_fields={"open_buy_fee": round_metric(open_buy_fee_usd)},
    )


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
            rate = align_previous_values(
                ts, fx_records, transform=(lambda value: 1.0 / value) if fx_invert else None,
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
