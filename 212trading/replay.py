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
from providers.ohlc_utils import align_previous_values, validate_ohlc_bars
from providers.replay_provider import SimClock
from strategies.replay_execution import (
    ExecutionModel,
    build_warmup_shadow_prices,
    choose_intrabar_scenario,
    iter_eligible_orders_in_side_order,
    normalize_replay_timestamps,
    require_optional_pos_float,
    resolve_order_fill_price,
    select_eligible_intrabar_orders,
    split_order_fill,
)
from strategies.replay_metrics import round_metric
from strategies.spot_replay import (
    ReplaySessionBase,
    extract_adapter_context_timestamps,
    iter_end_of_replay_market_sell_orders,
    scale_subbar_timeout_params,
    silence_strategy_log_notify,
)


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
    bar_minutes = require_optional_pos_float(bar_minutes, "bar_minutes must be a finite positive number")
    periods_per_year = require_optional_pos_float(periods_per_year, "periods_per_year must be a finite positive number")
    initial_cash = require_optional_pos_float(initial_cash, "initial_cash must be a finite positive number")
    validate_ohlc_bars(ohlc, exact_len_4=True, check_envelope=True)
    if warmup_ohlc:
        validate_ohlc_bars(warmup_ohlc, exact_len_4=True, check_envelope=False)
    if params.dca_trend_gate_pct > 0 and bar_minutes != 5:
        raise ValueError("dca_trend_gate_pct requires 5-minute bars, the same cadence as the live Yahoo signal")
    ts_tuple = (
        normalize_replay_timestamps(
            timestamps, len(ohlc), strictly_increasing=True, empty_as_none=True
        )
        or ()
    )
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
            timestamps=ts_tuple,
            warmup_timestamps=warmup_timestamps,
            execution=scenario,
            initial_cash=initial_cash,
        ),
    )


class _ReplayClient:
    __slots__ = ()


class _T212ReplaySession(ReplaySessionBase):
    """Concrete replay session (Level 3) executing the live Trading 212 strategy."""

    def __init__(
        self,
        ohlc,
        params: _strat.StratParams,
        *,
        bar_minutes: float | None,
        fx_rates: list[float],
        periods_per_year: float | None,
        warmup_ohlc,
        timestamps: Sequence[float],
        warmup_timestamps: Sequence[int] = (),
        execution: ExecutionModel,
        initial_cash: float | None = None,
    ) -> None:
        self.params = params
        self.fx_rates = fx_rates
        self.warmup_ohlc = warmup_ohlc
        self.timestamps = timestamps
        self.warmup_timestamps = warmup_timestamps
        self.client = _ReplayClient()
        self.clock = SimClock(0.0)
        self.cash = require_pos_float(initial_cash, "initial_cash") if initial_cash is not None else None
        self.min_cash = self.cash
        self.cash_refusals = 0
        self.realized_gross_account = 0.0
        self.realized_net_account = 0.0
        self.fees_account = 0.0
        self.open_cost_account = 0.0
        self.open_buy_fee_account = 0.0
        self.side_order = execution.side_order()
        self.trend_closes: deque[float] = deque(maxlen=12)
        self.engine = None
        self._original_place_buy = None
        self.cycle0 = 1
        self.prev_cycle = 1
        self.last_close = 0.0
        self.final_rate = fx_rates[-1]
        eff_periods = (
            periods_per_year
            if periods_per_year is not None
            else (equity_periods_per_year(bar_minutes) if bar_minutes is not None else None)
        )
        super().__init__(
            ohlc,
            initial_capital=require_pos_float(params.max_budget if self.cash is None else self.cash, "initial_capital"),
            account_currency=params.currency,
            execution=execution,
            bar_minutes=bar_minutes,
            periods_per_year=eff_periods,
        )

    def _execution_context(self):
        return silence_strategy_log_notify(_strat)

    def _setup_session(self) -> None:
        warmup_shadow = build_warmup_shadow_prices(
            self.warmup_ohlc,
            bar_minutes=self.bar_minutes,
            base_ts=self.timestamps[0] if self.timestamps else 0.0,
            warmup_timestamps=self.warmup_timestamps,
        )
        self.trend_closes = deque((px for _, px in warmup_shadow), maxlen=12)
        engine = _strat.Strategy(
            self.client,
            "REPLAY_US_EQ",
            self.params,
            dry_run=True,
            initial_state=_strat._new_state(),
            fx_to_usd=self.fx_rates[0],
            clock=self.clock,
            trend_slope_provider=lambda _symbol: _md.compute_ols_slope_pct(list(self.trend_closes), bars=12),
        )
        engine._shadow_prices.extend(warmup_shadow)
        engine._save = lambda: None
        self.engine = engine
        self._original_place_buy = engine._place_buy
        if self.cash is not None:
            engine._place_buy = self._guarded_place_buy
        self.cycle0 = engine.s.get("cycle", 1)
        self.prev_cycle = self.cycle0

    def _guarded_place_buy(self, amount: float, limit: float, kind: str = "ENTRY") -> bool:
        lim_f = safe_pos_float(limit, default=None)
        amt_f = safe_pos_float(amount, default=None)
        fx_f = safe_pos_float(self.engine.fx_to_usd, default=None)
        if lim_f is not None and amt_f is not None and round(lim_f, 2) > 0 and fx_f is not None:
            qty_est = round((amt_f * fx_f) / lim_f, 2)
            if qty_est > 0:
                required = (qty_est * round(lim_f, 2) / fx_f) * (1.0 + self.params.fx_fee_pct / 100.0)
                reserved = sum(
                    (o["qty"] * o["limit"] / fx_f) * (1.0 + self.params.fx_fee_pct / 100.0)
                    for o in self.engine.s["orders"]
                    if str(o.get("side") or "").lower() == "buy"
                )
                if required > max(0.0, (self.cash or 0.0) - reserved) + 1e-9:
                    self.cash_refusals += 1
                    return False
        return self._original_place_buy(amount, limit, kind)

    def _current_upnl_and_open_fee(self, px: float, fx: float) -> tuple[float, float, float]:
        q = self.engine.s["qty"]
        if q <= 1e-9:
            return q, 0.0, 0.0
        return q, q * px / fx - self.open_cost_account, self.open_buy_fee_account

    def _execute_sell_fill(
        self,
        order: dict,
        fill_order: dict,
        quantity: float,
        price: float,
        rate: float,
        *,
        held_before: float,
        mark_tp_level: bool,
        remove_before_apply: bool,
    ) -> None:
        fill_order["qty"] = quantity
        frac = min(1.0, quantity / held_before) if held_before > 1e-12 else 1.0
        released_cost_account = self.open_cost_account * frac
        released_buy_fee_account = self.open_buy_fee_account * frac
        self.open_cost_account = max(0.0, self.open_cost_account - released_cost_account)
        self.open_buy_fee_account = max(0.0, self.open_buy_fee_account - released_buy_fee_account)
        sell_notional_account = quantity * price / rate
        sell_fee_account = (self.params.fx_fee_pct / 100.0) * sell_notional_account
        if self.cash is not None:
            self.cash += sell_notional_account - sell_fee_account
            self.min_cash = min(self.min_cash, self.cash)
        gross_acc = sell_notional_account - released_cost_account
        fee_acc = released_buy_fee_account + sell_fee_account
        if mark_tp_level and fill_order.get("level") is not None:
            self.engine._record_tp_sold_level(fill_order["level"])
        self.realized_gross_account += gross_acc
        self.fees_account += fee_acc
        self.realized_net_account += gross_acc - fee_acc
        dust_before = max(0.0, held_before - quantity)
        if remove_before_apply:
            self.engine._remove(order)
        self.engine._apply_fill(fill_order, quantity, price)
        if self.engine.s["qty"] <= 1e-9 and (self.open_cost_account > 0.0 or self.open_buy_fee_account > 0.0):
            dust_val_acc = dust_before * price / rate
            dust_sell_fee = (self.params.fx_fee_pct / 100.0) * dust_val_acc
            if self.cash is not None and dust_val_acc > 0.0:
                self.cash += dust_val_acc - dust_sell_fee
                self.min_cash = min(self.min_cash, self.cash)
            dust_gross = dust_val_acc - self.open_cost_account
            dust_fee_total = self.open_buy_fee_account + dust_sell_fee
            self.realized_gross_account += dust_gross
            self.fees_account += dust_fee_total
            self.realized_net_account += dust_gross - dust_fee_total
            self.turnover_notional += dust_val_acc
            self.open_cost_account = 0.0
            self.open_buy_fee_account = 0.0
        cur_cycle = self.engine.s.get("cycle", 1)
        if cur_cycle > self.prev_cycle:
            self._record_closed_cycle(self.realized_net_account)
            self.prev_cycle = cur_cycle
        self.fills += 1
        self.turnover_notional += sell_notional_account

    def _cap_buy_order_to_cash(self, order: dict, cand_price: float, rate: float, *, market: bool) -> bool:
        if self.cash is None:
            return True
        rem_qty = require_pos_float(order["qty"], "order_qty")
        orig_qty = safe_pos_float(order.get("_replay_original_qty"), default=None) or rem_qty
        cand_qty = rem_qty if market else min(rem_qty, orig_qty * self.execution.partial_fill_ratio)
        fee_mult = 1.0 + self.params.fx_fee_pct / 100.0
        cand_cost = (cand_qty * cand_price / rate) * fee_mult
        if cand_cost <= self.cash + 1e-9:
            return True
        unit_cost = (cand_price / rate) * fee_mult
        max_qty = round((self.cash / unit_cost) - 0.004999, 2) if unit_cost > 0 else 0.0
        if max_qty < 0.01:
            self.engine._remove(order)
            self.cash_refusals += 1
            return False
        order["qty"] = min(rem_qty, max_qty)
        order["_replay_original_qty"] = order["qty"]
        return True

    def _settle_intrabar_orders(self, open_: float, high: float, low: float, close: float, rate: float) -> None:
        orders = self.engine.s["orders"]
        if not orders:
            return
        eligible_orders, is_ambiguous = select_eligible_intrabar_orders(
            orders, self.execution, high=high, low=low, limit_key="limit",
        )
        if eligible_orders:
            self.ambiguous_bars += 1 if is_ambiguous else 0
            for side, order in iter_eligible_orders_in_side_order(
                eligible_orders, self.engine.s["orders"], self.side_order
            ):
                market, cand_price = resolve_order_fill_price(
                    order, side, open_, self.execution, limit_key="limit"
                )
                if side == "buy" and not self._cap_buy_order_to_cash(order, cand_price, rate, market=market):
                    continue
                fill_order, quantity, complete = split_order_fill(
                    order,
                    quantity_key="qty",
                    amount_key="amount" if side == "buy" else None,
                    ratio=self.execution.partial_fill_ratio,
                    force_full=market,
                )
                if complete:
                    self.engine._remove(order)
                price = cand_price
                if side == "buy":
                    buy_notional_account = quantity * price / rate
                    buy_fee_account = (self.params.fx_fee_pct / 100.0) * buy_notional_account
                    fill_order["amount"] = buy_notional_account
                    self.open_cost_account += buy_notional_account
                    self.open_buy_fee_account += buy_fee_account
                    if self.cash is not None:
                        self.cash = max(0.0, self.cash - (buy_notional_account + buy_fee_account))
                        self.min_cash = min(self.min_cash, self.cash)
                    self.engine._apply_fill(fill_order, quantity, price)
                    self.fills += 1
                    self.turnover_notional += buy_notional_account
                else:
                    held_before = self.engine.s["qty"]
                    quantity = min(quantity, held_before)
                    if quantity <= 1e-12:
                        self.engine._remove(order)
                    else:
                        self._execute_sell_fill(
                            order,
                            fill_order,
                            quantity,
                            price,
                            rate,
                            held_before=held_before,
                            mark_tp_level=complete,
                            remove_before_apply=False,
                        )
        if self.engine.s["orders"]:
            for order in list(self.engine.s["orders"]):
                if self.engine._buy_is_stale(order, close):
                    self.engine._remove(order)

    def _step_bar(self, bar_index: int, open_: float, high: float, low: float, close: float) -> None:
        rate = self.fx_rates[bar_index]
        self.engine.fx_to_usd = rate
        self.clock.value = (
            safe_nonneg_float(self.timestamps[bar_index], 0.0) if self.timestamps
            else bar_index * self.bar_minutes * 60.0 if self.bar_minutes
            else bar_index * 1.0
        )
        self._settle_intrabar_orders(open_, high, low, close, rate)
        self.trend_closes.append(close)
        self.engine.step(close)
        qty, unrealized_account, cur_open_fee_account = self._current_upnl_and_open_fee(close, rate)
        self._record_bar_equity(
            self.initial_capital + self.realized_net_account + unrealized_account - cur_open_fee_account,
            qty > 1e-9,
        )

    def _finalize_end_of_replay(self) -> None:
        final_rate = self.fx_rates[-1]
        self.final_rate = final_rate
        last_close, end_exit_orders = iter_end_of_replay_market_sell_orders(
            self.ohlc, self.engine.s["orders"], self.engine.s["qty"], qty_eps=1e-9
        )
        self.last_close = last_close
        for order in end_exit_orders:
            held_before = self.engine.s["qty"]
            if held_before <= 1e-9:
                break
            quantity = min(require_pos_float(order["qty"], "order_qty"), held_before)
            if quantity <= 1e-12:
                self.engine._remove(order)
                continue
            price = self.execution.market_price("sell", last_close)
            self._execute_sell_fill(
                order,
                dict(order, qty=quantity),
                quantity,
                price,
                final_rate,
                held_before=held_before,
                mark_tp_level=True,
                remove_before_apply=True,
            )
            rem_qty, rem_upnl, rem_open_fee = self._current_upnl_and_open_fee(last_close, final_rate)
            self._overwrite_tail_equity(
                self.initial_capital + self.realized_net_account + rem_upnl - rem_open_fee,
                rem_qty > 1e-9,
            )

    def _build_summary(self) -> dict:
        qty, final_upnl_account, final_open_buy_fee_account = self._current_upnl_and_open_fee(
            self.last_close, self.final_rate
        )
        extra_fields: dict[str, object] = {"open_buy_fee": round_metric(final_open_buy_fee_account)}
        if self.cash is not None:
            extra_fields["min_cash"] = round_metric(self.min_cash)
            extra_fields["cash_refusals"] = self.cash_refusals
        return self._build_summary_from_state(
            realized=self.realized_gross_account,
            net=self.realized_net_account,
            fees=self.fees_account + final_open_buy_fee_account,
            total=self.realized_net_account + final_upnl_account - final_open_buy_fee_account,
            final_upnl=final_upnl_account,
            cycles=self.engine.s.get("cycle", 1) - self.cycle0,
            open_qty=qty,
            extra_fields=extra_fields,
        )


def _run_once(
    ohlc,
    params: _strat.StratParams,
    *,
    bar_minutes: float | None,
    fx_rates: list[float],
    periods_per_year: float | None,
    warmup_ohlc,
    timestamps: Sequence[float],
    warmup_timestamps: Sequence[int] = (),
    execution: ExecutionModel,
    initial_cash: float | None = None,
) -> dict:
    return _T212ReplaySession(
        ohlc,
        params,
        bar_minutes=bar_minutes,
        fx_rates=fx_rates,
        periods_per_year=periods_per_year,
        warmup_ohlc=warmup_ohlc,
        timestamps=timestamps,
        warmup_timestamps=warmup_timestamps,
        execution=execution,
        initial_cash=initial_cash,
    ).run()


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
        ts, w_ts = extract_adapter_context_timestamps(context, empty=())
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
