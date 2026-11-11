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

from botcore import (
    load_local_module,
    require_pos_float,
    safe_nonneg_float,
    safe_pos_float,
)
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
from providers.order_ohlc_utils import align_previous_values
from providers.replay_provider import SimClock
from strategies.replay_execution import (
    ExecutionModel,
    choose_intrabar_scenario,
    select_eligible_intrabar_orders,
    split_order_fill,
    validate_ohlc_bars,
)
from strategies.replay_metrics import round_metric
from strategies.spot_replay import build_replay_summary, scale_subbar_timeout_params


def equity_periods_per_year(interval_minutes: float) -> float:
    """Return US equity annualization periods for a bar interval in minutes."""
    im = safe_pos_float(interval_minutes, default=None)
    if im is None:
        raise ValueError("interval_minutes must be positive")
    if im >= 1440:
        return 252.0 * 1440.0 / im
    return 252.0 * 390.0 / im


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
    warmup_timestamps: Sequence[int] = (),
    execution: ExecutionModel | None = None,
    initial_cash: float | None = None,
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
    if initial_cash is not None:
        initial_cash = safe_pos_float(initial_cash, default=None)
        if initial_cash is None:
            raise ValueError("initial_cash must be a finite positive number")
    validate_ohlc_bars(ohlc, exact_len_4=True, check_envelope=True)
    if warmup_ohlc:
        validate_ohlc_bars(warmup_ohlc, exact_len_4=True, check_envelope=False)
    if params.dca_trend_gate_pct > 0 and bar_minutes != 5:
        raise ValueError("dca_trend_gate_pct requires 5-minute bars, the same cadence as the live Yahoo signal")
    if timestamps:
        if len(timestamps) != len(ohlc):
            raise ValueError("timestamps must have the same length as ohlc")
        prev_ts: float | None = None
        for idx, ts_val in enumerate(timestamps):
            ts_f = safe_nonneg_float(ts_val, default=None)
            if ts_f is None:
                raise ValueError(f"timestamp at index {idx} must be a non-negative finite number")
            if prev_ts is not None and ts_f <= prev_ts:
                raise ValueError(f"timestamps must be strictly increasing at index {idx}")
            prev_ts = ts_f
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
            warmup_timestamps=warmup_timestamps,
            execution=scenario,
            initial_cash=initial_cash,
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
    warmup_timestamps: Sequence[int] = (),
    execution: ExecutionModel,
    initial_cash: float | None = None,
) -> dict:
    client = _ReplayClient()
    clock = SimClock(0.0)
    trend_closes: deque[float] = deque(maxlen=12)
    warmup_rows = list(warmup_ohlc or ())
    warmup_ts_list = list(warmup_timestamps or ())
    use_warmup_ts = len(warmup_ts_list) == len(warmup_rows) and bool(warmup_rows)
    warmup_step = bar_minutes * 60.0 if bar_minutes else 1.0
    base_ts = safe_nonneg_float(timestamps[0], 0.0) if timestamps else 0.0
    warmup_shadow: list[tuple[float, float]] = []
    for pos_idx, row in enumerate(warmup_rows):
        w_idx = pos_idx - len(warmup_rows)
        if isinstance(row, (list, tuple)) and len(row) >= 4:
            cf = safe_pos_float(row[3], default=None)
            if cf is not None:
                trend_closes.append(cf)
                w_ts = (
                    safe_nonneg_float(warmup_ts_list[pos_idx], base_ts + w_idx * warmup_step)
                    if use_warmup_ts
                    else (base_ts + w_idx * warmup_step)
                )
                warmup_shadow.append((w_ts, cf))
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
            trend_slope_provider=lambda _symbol: _md.compute_ols_slope_pct(list(trend_closes), bars=12),
        )
        engine._shadow_prices.extend(warmup_shadow)
        engine._save = lambda: None
        cash = require_pos_float(initial_cash, "initial_cash") if initial_cash is not None else None
        min_cash = cash
        cash_refusals = 0
        original_place_buy = engine._place_buy

        def _free_account_cash(cur_rate: float) -> float:
            reserved = sum(
                (o["qty"] * o["limit"] / cur_rate) * (1.0 + params.fx_fee_pct / 100.0)
                for o in engine.s["orders"]
                if str(o.get("side") or "").lower() == "buy"
            )
            return max(0.0, (cash or 0.0) - reserved)

        def _guarded_place_buy(amount: float, limit: float, kind: str = "ENTRY") -> bool:
            nonlocal cash_refusals
            lim_f = safe_pos_float(limit, default=None)
            amt_f = safe_pos_float(amount, default=None)
            fx_f = safe_pos_float(engine.fx_to_usd, default=None)
            if lim_f is not None and amt_f is not None and round(lim_f, 2) > 0 and fx_f is not None:
                qty_est = round((amt_f * fx_f) / lim_f, 2)
                if qty_est > 0:
                    required = (qty_est * round(lim_f, 2) / fx_f) * (1.0 + params.fx_fee_pct / 100.0)
                    if required > _free_account_cash(fx_f) + 1e-9:
                        cash_refusals += 1
                        return False
            return original_place_buy(amount, limit, kind)

        if cash is not None:
            engine._place_buy = _guarded_place_buy

        initial_capital = require_pos_float(params.max_budget if cash is None else cash, "initial_capital")
        equity_curve = [initial_capital]
        exposure = []
        trade_pnls = []
        turnover_account = realized_gross_account = realized_net_account = fees_account = 0.0
        open_cost_account = open_buy_fee_account = 0.0
        fills = wins = ambiguous_bars = 0
        cycle0 = engine.s.get("cycle", 1)
        prev_cycle = cycle0
        cycle_net_start = 0.0
        side_order = execution.side_order()

        for bar_index, row in enumerate(ohlc):
            open_ = require_pos_float(row[0], "open")
            high = require_pos_float(row[1], "high")
            low = require_pos_float(row[2], "low")
            close = require_pos_float(row[3], "close")
            rate = fx_rates[bar_index]
            engine.fx_to_usd = rate
            clock.value = (
                safe_nonneg_float(timestamps[bar_index], 0.0) if timestamps
                else bar_index * bar_minutes * 60.0 if bar_minutes
                else bar_index * 1.0
            )

            orders = engine.s["orders"]
            if orders:
                eligible_orders, is_ambiguous = select_eligible_intrabar_orders(
                    orders, execution, high=high, low=low, limit_key="limit",
                )
                if eligible_orders:
                    ambiguous_bars += 1 if is_ambiguous else 0
                    for side in side_order:
                        for order in eligible_orders:
                            if order not in engine.s["orders"] or order["side"].lower() != side:
                                continue
                            market = bool(order.get("market"))
                            cand_price = (
                                execution.market_price(side, open_)
                                if market
                                else require_pos_float(order["limit"], "order_limit")
                            )
                            if side == "buy" and cash is not None:
                                rem_qty = require_pos_float(order["qty"], "order_qty")
                                orig_qty = safe_pos_float(order.get("_replay_original_qty"), default=None) or rem_qty
                                cand_qty = rem_qty if market else min(rem_qty, orig_qty * execution.partial_fill_ratio)
                                fee_mult = 1.0 + params.fx_fee_pct / 100.0
                                cand_cost = (cand_qty * cand_price / rate) * fee_mult
                                if cand_cost > cash + 1e-9:
                                    unit_cost = (cand_price / rate) * fee_mult
                                    max_qty = round((cash / unit_cost) - 0.004999, 2) if unit_cost > 0 else 0.0
                                    if max_qty < 0.01:
                                        cash_refusals += 1
                                        engine._remove(order)
                                        continue
                                    order["qty"] = min(rem_qty, max_qty)
                                    order["_replay_original_qty"] = order["qty"]
                            fill_order, quantity, complete = split_order_fill(
                                order,
                                quantity_key="qty",
                                amount_key="amount" if side == "buy" else None,
                                ratio=execution.partial_fill_ratio,
                                force_full=market,
                            )
                            if complete:
                                engine._remove(order)
                            price = cand_price
                            if side == "buy":
                                buy_notional_account = quantity * price / rate
                                buy_fee_account = (params.fx_fee_pct / 100.0) * buy_notional_account
                                fill_order["amount"] = buy_notional_account
                                open_cost_account += buy_notional_account
                                open_buy_fee_account += buy_fee_account
                                if cash is not None:
                                    cash = max(0.0, cash - (buy_notional_account + buy_fee_account))
                                    min_cash = min(min_cash, cash)
                            else:
                                held_before = engine.s["qty"]
                                quantity = min(quantity, held_before)
                                if quantity <= 1e-12:
                                    engine._remove(order)
                                    continue
                                fill_order["qty"] = quantity
                                frac = min(1.0, quantity / held_before) if held_before > 1e-12 else 1.0
                                released_cost_account = open_cost_account * frac
                                released_buy_fee_account = open_buy_fee_account * frac
                                open_cost_account = max(0.0, open_cost_account - released_cost_account)
                                open_buy_fee_account = max(0.0, open_buy_fee_account - released_buy_fee_account)
                                sell_notional_account = quantity * price / rate
                                sell_fee_account = (params.fx_fee_pct / 100.0) * sell_notional_account
                                if cash is not None:
                                    cash += sell_notional_account - sell_fee_account
                                    min_cash = min(min_cash, cash)
                                gross_acc = sell_notional_account - released_cost_account
                                fee_acc = released_buy_fee_account + sell_fee_account
                                net_acc = gross_acc - fee_acc
                                if complete and fill_order.get("level") is not None:
                                    engine._record_tp_sold_level(fill_order["level"])
                                realized_gross_account += gross_acc
                                fees_account += fee_acc
                                realized_net_account += net_acc
                            dust_before = max(0.0, engine.s["qty"] - quantity) if side == "sell" else 0.0
                            engine._apply_fill(fill_order, quantity, price)
                            if side == "sell":
                                if engine.s["qty"] <= 1e-9 and (open_cost_account > 0.0 or open_buy_fee_account > 0.0):
                                    dust_val_acc = dust_before * price / rate
                                    dust_sell_fee = (params.fx_fee_pct / 100.0) * dust_val_acc
                                    if cash is not None and dust_val_acc > 0.0:
                                        cash += dust_val_acc - dust_sell_fee
                                        min_cash = min(min_cash, cash)
                                    dust_gross = dust_val_acc - open_cost_account
                                    dust_fee_total = open_buy_fee_account + dust_sell_fee
                                    realized_gross_account += dust_gross
                                    fees_account += dust_fee_total
                                    realized_net_account += dust_gross - dust_fee_total
                                    turnover_account += dust_val_acc
                                    open_cost_account = 0.0
                                    open_buy_fee_account = 0.0
                                cur_cycle = engine.s.get("cycle", 1)
                                if cur_cycle > prev_cycle:
                                    cycle_pnl = realized_net_account - cycle_net_start
                                    trade_pnls.append(cycle_pnl)
                                    if cycle_pnl > 0.0:
                                        wins += 1
                                    cycle_net_start = realized_net_account
                                    prev_cycle = cur_cycle
                            fills += 1
                            turnover_account += quantity * price / rate

                if engine.s["orders"]:
                    for order in list(engine.s["orders"]):
                        if engine._buy_is_stale(order, close):
                            engine._remove(order)
            trend_closes.append(close)
            engine.step(close)

            qty = engine.s["qty"]
            unrealized_account = (qty * close / rate - open_cost_account) if qty > 1e-9 else 0.0
            cur_open_fee_account = open_buy_fee_account if qty > 1e-9 else 0.0
            equity_curve.append(initial_capital + realized_net_account + unrealized_account - cur_open_fee_account)
            exposure.append(qty > 1e-9)

        final_rate = fx_rates[-1]
        last_close = require_pos_float(ohlc[-1][3], "last_close")
        settled_end_exit = False
        for order in list(engine.s["orders"]):
            if order.get("market") and str(order.get("side") or "").lower() == "sell" and engine.s["qty"] > 1e-9:
                held_before = engine.s["qty"]
                quantity = min(require_pos_float(order["qty"], "order_qty"), held_before)
                if quantity <= 1e-12:
                    engine._remove(order)
                    continue
                price = execution.market_price("sell", last_close)
                fill_order = dict(order, qty=quantity)
                frac = min(1.0, quantity / held_before) if held_before > 1e-12 else 1.0
                released_cost_account = open_cost_account * frac
                released_buy_fee_account = open_buy_fee_account * frac
                open_cost_account = max(0.0, open_cost_account - released_cost_account)
                open_buy_fee_account = max(0.0, open_buy_fee_account - released_buy_fee_account)
                sell_notional_account = quantity * price / final_rate
                sell_fee_account = (params.fx_fee_pct / 100.0) * sell_notional_account
                if cash is not None:
                    cash += sell_notional_account - sell_fee_account
                    min_cash = min(min_cash, cash)
                gross_acc = sell_notional_account - released_cost_account
                fee_acc = released_buy_fee_account + sell_fee_account
                if fill_order.get("level") is not None:
                    engine._record_tp_sold_level(fill_order["level"])
                realized_gross_account += gross_acc
                fees_account += fee_acc
                realized_net_account += gross_acc - fee_acc
                dust_before = max(0.0, held_before - quantity)
                engine._remove(order)
                engine._apply_fill(fill_order, quantity, price)
                if engine.s["qty"] <= 1e-9 and (open_cost_account > 0.0 or open_buy_fee_account > 0.0):
                    dust_val_acc = dust_before * price / final_rate
                    dust_sell_fee = (params.fx_fee_pct / 100.0) * dust_val_acc
                    if cash is not None and dust_val_acc > 0.0:
                        cash += dust_val_acc - dust_sell_fee
                        min_cash = min(min_cash, cash)
                    dust_gross = dust_val_acc - open_cost_account
                    dust_fee_total = open_buy_fee_account + dust_sell_fee
                    realized_gross_account += dust_gross
                    fees_account += dust_fee_total
                    realized_net_account += dust_gross - dust_fee_total
                    turnover_account += dust_val_acc
                    open_cost_account = 0.0
                    open_buy_fee_account = 0.0
                cur_cycle = engine.s.get("cycle", 1)
                if cur_cycle > prev_cycle:
                    cycle_pnl = realized_net_account - cycle_net_start
                    trade_pnls.append(cycle_pnl)
                    if cycle_pnl > 0.0:
                        wins += 1
                    cycle_net_start = realized_net_account
                    prev_cycle = cur_cycle
                fills += 1
                turnover_account += sell_notional_account
                settled_end_exit = True
        if settled_end_exit:
            rem_qty = engine.s["qty"]
            rem_upnl = (rem_qty * last_close / final_rate - open_cost_account) if rem_qty > 1e-9 else 0.0
            rem_open_fee = open_buy_fee_account if rem_qty > 1e-9 else 0.0
            if equity_curve:
                equity_curve[-1] = initial_capital + realized_net_account + rem_upnl - rem_open_fee
            if exposure:
                exposure[-1] = rem_qty > 1e-9
    finally:
        _strat.log, _strat.notify = original_log, original_notify

    qty = engine.s["qty"]
    final_upnl_account = (qty * last_close / final_rate - open_cost_account) if qty > 1e-9 else 0.0
    final_open_buy_fee_account = open_buy_fee_account if qty > 1e-9 else 0.0
    eff_periods_per_year = (
        periods_per_year
        if periods_per_year is not None
        else (equity_periods_per_year(bar_minutes) if bar_minutes is not None else None)
    )
    extra_fields: dict[str, object] = {"open_buy_fee": round_metric(final_open_buy_fee_account)}
    if cash is not None:
        extra_fields["min_cash"] = round_metric(min_cash)
        extra_fields["cash_refusals"] = cash_refusals
    return build_replay_summary(
        realized=realized_gross_account,
        net=realized_net_account,
        fees=fees_account + final_open_buy_fee_account,
        total=realized_net_account + final_upnl_account - final_open_buy_fee_account,
        final_upnl=final_upnl_account,
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
        periods_per_year=eff_periods_per_year,
        extra_fields=extra_fields,
    )


def make_t212_replay_adapter(
    params: _strat.StratParams,
    bar_minutes: float | None = None,
    fx_to_usd: float | Sequence[float] | None = None,
    fx_records: Sequence[dict] | None = None,
    fx_invert: bool = False,
    periods_per_year: float | None = None,
    execution: ExecutionModel | None = None,
    initial_cash: float | None = None,
):
    """Return a generic ReplayFunction compatible with evaluation.py walk-forward benchmarks."""
    bm = safe_pos_float(bar_minutes, default=None)
    eff_params = scale_subbar_timeout_params(params, bm) if bm is not None else params

    def _adapter(ohlc, warmup_ohlc, context=None):
        ts = list(context.timestamps) if context and getattr(context, "timestamps", None) else ()
        w_ts = list(context.warmup_timestamps) if context and getattr(context, "warmup_timestamps", None) else ()
        rate = fx_to_usd if fx_to_usd is not None else 1.0
        if fx_records is not None and ts:
            rate = align_previous_values(
                ts, fx_records, transform=(lambda value: 1.0 / value) if fx_invert else None,
            )
        return run_replay(
            ohlc,
            eff_params,
            bar_minutes=bar_minutes,
            fx_to_usd=rate,
            periods_per_year=periods_per_year,
            warmup_ohlc=warmup_ohlc,
            timestamps=ts,
            warmup_timestamps=w_ts,
            execution=execution,
            initial_cash=initial_cash,
        )
    return _adapter
