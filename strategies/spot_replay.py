#!/usr/bin/env python3
"""strategies/spot_replay.py -- Canonical replay engine for spot trading strategies.

Replays spot_engine.py over historical OHLC bars. The strategy decides orders,
while the harness separately models order fills, fee schedules, slippage, and
intrabar ambiguity resolution without network, notifications, or persistent state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import contextlib
import copy
import dataclasses

from botcore import (
    exact_nonneg_int,
    is_market_order,
    normalize_epoch_sec,
    normalize_side,
    require_pos_float,
    safe_float,
    safe_int,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
    safe_pos_int,
    _parse_bool_flag,
)
from providers.ohlc_utils import validate_ohlc_bars
from strategies.replay_execution import (
    ExecutionModel,
    FeeModel,
    build_warmup_shadow_prices,
    choose_intrabar_scenario,
    clamp_sell_fill_to_held,
    iter_eligible_orders_in_side_order,
    normalize_replay_timestamps,
    require_optional_pos_float,
    resolve_order_fill_price,
    select_eligible_intrabar_orders,
    split_order_fill,
    unpack_ohlc_bar,
)
from strategies.replay_metrics import calculate_performance_metrics, round_metric
from providers.strategy_executor import PairPrecision
from strategies import spot_engine as _strat
from strategies.spot_engine import StratParams

_REPLAY_PRECISION = PairPrecision(5, 8, 0.0, "REPLAY")
_LAST_REPLAY_DIAG: dict = {}


def _silent(*_args, **_kwargs):
    return None


@contextlib.contextmanager
def silence_strategy_log_notify(strat_mod):
    """Temporarily silence ``strat_mod.log`` and ``strat_mod.notify`` during replay execution."""
    original_log, original_notify = strat_mod.log, strat_mod.notify
    strat_mod.log = strat_mod.notify = _silent
    try:
        yield
    finally:
        strat_mod.log, strat_mod.notify = original_log, original_notify


def extract_adapter_context_timestamps(context, *, empty=()):
    """Extract ``(timestamps, warmup_timestamps)`` lists from a walk-forward benchmark context."""
    ts = list(context.timestamps) if context and getattr(context, "timestamps", None) else empty
    w_ts = list(context.warmup_timestamps) if context and getattr(context, "warmup_timestamps", None) else empty
    return ts, w_ts


def record_closed_cycle_pnl(
    realized_net: float,
    cycle_net_start: float,
    trade_pnls: list[float],
    wins: int,
) -> tuple[float, int]:
    """Append the closed cycle PnL to ``trade_pnls`` and return ``(new_cycle_net_start, new_wins)``."""
    cycle_pnl = realized_net - cycle_net_start
    trade_pnls.append(cycle_pnl)
    if cycle_pnl > 0.0:
        wins += 1
    return realized_net, wins


def overwrite_tail_equity_and_exposure(
    equity_curve: list[float],
    exposure: list[bool],
    eq_val: float,
    exp_val: bool,
) -> None:
    """Update the final bar's equity and exposure in-place after end-of-replay market settlements."""
    if equity_curve:
        equity_curve[-1] = eq_val
    if exposure:
        exposure[-1] = exp_val


def iter_end_of_replay_market_sell_orders(
    ohlc: list, orders: list[dict], open_qty: float, *, qty_eps: float = 1e-12
) -> tuple[float, list[dict]]:
    """Return ``(last_close, eligible_market_sell_orders)`` for end-of-replay settlement."""
    if not isinstance(ohlc, (list, tuple)) or not ohlc:
        raise ValueError("ohlc cannot be empty")
    _, _, _, last_close = unpack_ohlc_bar(ohlc[-1])
    qty_f = safe_pos_float(open_qty, default=0.0) or 0.0
    eps_f = safe_nonneg_float(qty_eps, default=1e-12) or 0.0
    if qty_f <= eps_f:
        return last_close, []
    return last_close, [
        o for o in list(orders or ())
        if isinstance(o, dict) and is_market_order(o) and normalize_side(o.get("side")) == "SELL"
    ]


def build_replay_summary(
    *,
    realized: float,
    net: float,
    fees: float,
    total: float,
    final_upnl: float,
    cycles: int,
    wins: int,
    open_qty: float,
    fills: int,
    ambiguous_bars: int,
    initial_capital: float,
    account_currency: str,
    equity_curve: list[float],
    exposure: list[bool],
    trade_pnls: list[float],
    turnover_notional: float,
    bar_minutes: float | None = None,
    periods_per_year: float | None = None,
    extra_fields: dict | None = None,
) -> dict:
    """Calculate standardized performance metrics and build the rounded replay summary dict."""
    bm_f = safe_pos_float(bar_minutes, default=None)
    if periods_per_year is None and bm_f is not None:
        periods_per_year = 365.0 * 24 * 60 / bm_f
    performance = calculate_performance_metrics(
        equity_curve,
        initial_capital=initial_capital,
        periods_per_year=periods_per_year,
        exposure=exposure,
        trade_pnls=trade_pnls,
        turnover_notional=turnover_notional,
    )
    rounded = round_metric
    result = {
        "realized": rounded(realized),
        "net": rounded(net),
        "fees": rounded(fees),
        "total": rounded(total),
        "final_upnl": rounded(final_upnl),
        "cycles": exact_nonneg_int(cycles, "cycles"),
        "wins": exact_nonneg_int(wins, "wins"),
        "maxdd": rounded(performance["max_drawdown_abs"]),
        "open_qty": rounded(open_qty),
        "fills": exact_nonneg_int(fills, "fills"),
        "ambiguous_bars": exact_nonneg_int(ambiguous_bars, "ambiguous_bars"),
        "initial_capital": rounded(initial_capital),
        "account_currency": account_currency,
    }
    if extra_fields:
        result.update(extra_fields)
    result.update({key: rounded(value) for key, value in performance.items()})
    return result


class _ReplayClient:
    """Lightweight replay venue client without mock call-recording overhead."""

    __slots__ = ("_default_balance", "_free_balance_fn", "buy_fee_pct")

    def __init__(self, default_balance: float) -> None:
        self._default_balance = default_balance
        self._free_balance_fn = None
        self.buy_fee_pct = 0.0

    def pair_precision(self, _pair: str) -> PairPrecision:
        return _REPLAY_PRECISION

    def free_balance(self, asset: str) -> float:
        return self._free_balance_fn(asset) if self._free_balance_fn is not None else self._default_balance


def params_require_trend_interval(params) -> bool:
    """Return True if ``params`` enable any regime/trend policy requiring ``trend_interval`` bars."""
    surge_window_enabled = (
        _parse_bool_flag(getattr(params, "surge_guard", False), False)
        and safe_pos_float(getattr(params, "surge_window_hours", 0), default=0.0) > 0.0
        and safe_pos_float(getattr(params, "surge_move_pct", 0), default=0.0) > 0.0
    )
    return bool(
        _parse_bool_flag(getattr(params, "trend_overlay", False), False)
        or _parse_bool_flag(getattr(params, "dca_trend_brake", False), False)
        or _parse_bool_flag(getattr(params, "reentry_hybrid_enabled", False), False)
        or surge_window_enabled
        or _parse_bool_flag(getattr(params, "tp_dynamic_flat", False), False)
    )


def _params_require_tp_trail_vol_interval(params) -> bool:
    return bool(
        _parse_bool_flag(getattr(params, "tp_trail_adaptive", False), False)
        or (
            _parse_bool_flag(getattr(params, "surge_guard", False), False)
            and _parse_bool_flag(getattr(params, "surge_dynamic", False), False)
        )
    )


def required_replay_intervals(params) -> set[int]:
    """Return the set of bar intervals (in minutes) required by active policies on ``params``."""
    required = {safe_int(params.trend_interval, 0)} if params_require_trend_interval(params) else set()
    if _params_require_tp_trail_vol_interval(params):
        required.add(safe_int(params.tp_trail_vol_interval, 0))
    if safe_float(getattr(params, "dca_vol_scale_k", 0.0), default=0.0) != 0.0:
        required.add(safe_int(params.dca_vol_interval, 0))
    required.discard(0)
    return required


def _validate_replay(ohlc, params, bar_minutes: float | None) -> None:
    validate_ohlc_bars(ohlc, check_envelope=False)
    bm_val = None
    if bar_minutes is not None:
        bm_val = safe_pos_float(bar_minutes, default=None)
        if bm_val is None:
            raise ValueError("bar_minutes must be finite and positive")
    if params_require_trend_interval(params) and (
        bm_val is None or bm_val != safe_float(params.trend_interval, default=-1.0)
    ):
        raise ValueError(
            f"regime-aware policies require bar_minutes to equal trend_interval ({params.trend_interval} minutes); resampling is not implemented"
        )
    if _params_require_tp_trail_vol_interval(params) and (
        bm_val is None or bm_val != safe_float(params.tp_trail_vol_interval, default=-1.0)
    ):
        raise ValueError(
            f"adaptive trailing or surge requires bar_minutes to equal tp_trail_vol_interval ({params.tp_trail_vol_interval} minute)"
        )
    if safe_float(getattr(params, "dca_vol_scale_k", 0.0), default=0.0) != 0.0 and (
        bm_val is None or bm_val != safe_float(params.dca_vol_interval, default=-1.0)
    ):
        raise ValueError(
            f"dca_vol_scale_k requires bar_minutes to equal dca_vol_interval ({params.dca_vol_interval} minute)"
        )
    if _parse_bool_flag(getattr(params, "reentry_adaptive", False), False) and bm_val is None:
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
    dca_drop_series=None,
    reentry_series=None,
    intrabar_high: bool = False,
    timestamps=None,
    warmup_timestamps=None,
) -> dict:
    """Replay decisions, optionally enforcing a finite cash account and fee reserves."""
    _validate_replay(ohlc, params, bar_minutes)
    if bar_minutes is not None:
        bar_minutes = safe_pos_float(bar_minutes, default=None)
    include_decision_trace = _strat._parse_bool_flag(include_decision_trace, False)
    intrabar_high = _strat._parse_bool_flag(intrabar_high, False)
    initial_cash = require_optional_pos_float(initial_cash, "initial_cash must be finite and positive")
    ts_tuple = normalize_replay_timestamps(timestamps, len(ohlc), normalize_ms=True)
    warmup_ts_tuple: tuple[float | None, ...] | None = (
        tuple(normalize_epoch_sec(wt, default=None, allow_zero=True) for wt in warmup_timestamps)
        if warmup_timestamps is not None
        else None
    )
    model = execution or ExecutionModel()
    fees = fee_model or FeeModel(fee_pct, fee_pct)
    warmup_rows = tuple(warmup_ohlc) if warmup_ohlc else ()
    scenario_diags: dict[str, dict] = {}

    chosen = choose_intrabar_scenario(
        model,
        lambda scenario: (
            out := _run_once(
                ohlc,
                params,
                fee_model=fees,
                bar_minutes=bar_minutes,
                warmup_ohlc=warmup_rows,
                execution=scenario,
                include_decision_trace=include_decision_trace,
                initial_cash=initial_cash,
                dca_drop_series=dca_drop_series,
                reentry_series=reentry_series,
                intrabar_high=intrabar_high,
                timestamps=ts_tuple,
                warmup_timestamps=warmup_ts_tuple,
            ),
            scenario_diags.__setitem__(scenario.intrabar_policy, dict(_LAST_REPLAY_DIAG)),
        )[0],
    )
    selected_policy = chosen.get("intrabar_policy_selected")
    if selected_policy in scenario_diags:
        global _LAST_REPLAY_DIAG
        _LAST_REPLAY_DIAG = scenario_diags[selected_policy]
    return chosen


class ReplaySessionBase(ABC):
    """Abstract base (Level 2) for bar-by-bar OHLC replay sessions sharing accounting and summary logic."""

    def __init__(
        self,
        ohlc,
        *,
        initial_capital: float,
        account_currency: str,
        execution: ExecutionModel,
        bar_minutes: float | None = None,
        periods_per_year: float | None = None,
        initial_cycle_net_start: float = 0.0,
    ) -> None:
        self.ohlc = ohlc
        self.initial_capital = initial_capital
        self.account_currency = account_currency
        self.execution = execution
        self.bar_minutes = bar_minutes
        self.periods_per_year = periods_per_year
        self.equity_curve: list[float] = [initial_capital]
        self.exposure: list[bool] = []
        self.trade_pnls: list[float] = []
        self.fills = 0
        self.wins = 0
        self.ambiguous_bars = 0
        self.turnover_notional = 0.0
        self.cycle_net_start = initial_cycle_net_start

    def _execution_context(self):
        return contextlib.nullcontext()

    def _setup_session(self) -> None:
        return None

    @abstractmethod
    def _step_bar(self, bar_index: int, open_: float, high: float, low: float, close: float) -> None:
        """Advance the replay session through a single OHLC bar."""

    def _finalize_end_of_replay(self) -> None:
        return None

    @abstractmethod
    def _build_summary(self) -> dict:
        """Build the standardized replay summary dict after the bar loop completes."""

    def _record_bar_equity(self, equity: float, exposed: bool) -> None:
        self.equity_curve.append(equity)
        self.exposure.append(exposed)

    def _overwrite_tail_equity(self, equity: float, exposed: bool) -> None:
        overwrite_tail_equity_and_exposure(self.equity_curve, self.exposure, equity, exposed)

    def _record_closed_cycle(self, realized_net: float) -> None:
        self.cycle_net_start, self.wins = record_closed_cycle_pnl(
            realized_net, self.cycle_net_start, self.trade_pnls, self.wins
        )

    def _build_summary_from_state(
        self,
        *,
        realized: float,
        net: float,
        fees: float,
        total: float,
        final_upnl: float,
        cycles: int,
        open_qty: float,
        extra_fields: dict | None = None,
    ) -> dict:
        return build_replay_summary(
            realized=realized,
            net=net,
            fees=fees,
            total=total,
            final_upnl=final_upnl,
            cycles=cycles,
            wins=self.wins,
            open_qty=open_qty,
            fills=self.fills,
            ambiguous_bars=self.ambiguous_bars,
            initial_capital=self.initial_capital,
            account_currency=self.account_currency,
            equity_curve=self.equity_curve,
            exposure=self.exposure,
            trade_pnls=self.trade_pnls,
            turnover_notional=self.turnover_notional,
            bar_minutes=self.bar_minutes,
            periods_per_year=self.periods_per_year,
            extra_fields=extra_fields,
        )

    def run(self) -> dict:
        """Execute the full replay session lifecycle and return the summary dict."""
        with self._execution_context():
            self._setup_session()
            for bar_index, row in enumerate(self.ohlc):
                open_, high, low, close = unpack_ohlc_bar(row)
                self._step_bar(bar_index, open_, high, low, close)
            self._finalize_end_of_replay()
        return self._build_summary()


class _SpotReplaySession(ReplaySessionBase):
    """Concrete replay session (Level 3) executing ``strategies.spot_engine.Strategy``."""

    def __init__(
        self,
        ohlc,
        params,
        *,
        fee_model: FeeModel,
        bar_minutes: float | None,
        warmup_ohlc,
        execution: ExecutionModel,
        include_decision_trace: bool,
        initial_cash: float | None,
        dca_drop_series=None,
        reentry_series=None,
        intrabar_high: bool = False,
        timestamps: tuple[float, ...] | None = None,
        warmup_timestamps: tuple[float | None, ...] | None = None,
    ) -> None:
        if dca_drop_series is not None or reentry_series is not None:
            params = dataclasses.replace(params) if dataclasses.is_dataclass(params) else copy.copy(params)
        self.params = params
        self.fee_model = fee_model
        self.warmup_ohlc = warmup_ohlc
        self.include_decision_trace = include_decision_trace
        self.initial_cash = initial_cash
        self.dca_drop_series = dca_drop_series
        self.reentry_series = reentry_series
        self.intrabar_high = intrabar_high
        self.timestamps = timestamps
        self.warmup_timestamps = warmup_timestamps
        self.allocated_budget = require_pos_float(params.allocated_budget(), "allocated_budget")
        self.cash = require_pos_float(initial_cash, "initial_cash") if initial_cash is not None else None
        self.min_cash = self.cash
        self.cash_refusals = 0
        self.decision_trace: list[dict] = []
        self.current_bar = -1
        self.default_dca_drop = params.dca_drop_pct
        self.default_reentry_drop = params.reentry_drop_pct
        self.check_ttl = bar_minutes is None or bar_minutes >= params.order_ttl_min
        self.bar_step_sec = bar_minutes * 60 if bar_minutes else 1
        self.side_order = execution.side_order()
        self.client = _ReplayClient(self.allocated_budget)
        self.strategy = None
        self._original_place = None
        self.cycle0 = 1
        self.last_close = 0.0
        super().__init__(
            ohlc,
            initial_capital=self.allocated_budget if self.cash is None else self.cash,
            account_currency=params.currency,
            execution=execution,
            bar_minutes=bar_minutes,
        )

    def _execution_context(self):
        return silence_strategy_log_notify(_strat)

    def _setup_session(self) -> None:
        strategy = _strat.Strategy(
            self.client, "REPLAY", self.params, dry_run=True, initial_state=_strat._new_state(), replay_mode=True,
        )
        base_ts = self.timestamps[0] if self.timestamps else 0.0
        for eff_wt, close_f in build_warmup_shadow_prices(
            self.warmup_ohlc,
            bar_minutes=self.bar_minutes,
            base_ts=base_ts,
            warmup_timestamps=self.warmup_timestamps,
            normalize_ms=True,
        ):
            strategy._shadow_prices.append((eff_wt, close_f))
            if strategy._surge_replay_closes is not None:
                strategy._surge_replay_closes.append(close_f)
        strategy.set_simulated_time(base_ts)
        strategy._save = _silent
        self.strategy = strategy
        self._original_place = strategy._place
        self.client._free_balance_fn = self._free_balance_for_asset
        if self.cash is not None:
            self.client.buy_fee_pct = self.fee_model.rate_pct(market=False)
        if self.include_decision_trace or self.cash is not None:
            strategy._place = self._traced_place
        self.cycle0 = strategy.s.get("cycle", 1)
        self.cycle_net_start = strategy.s["realized_net"]

    def _free_balance_for_asset(self, asset) -> float:
        asset_norm = str(asset or "").strip().upper()
        if asset_norm == _REPLAY_PRECISION.base_asset:
            reserved_sell = sum(
                safe_nonneg_float(o.get("vol"), default=0.0)
                for o in self.strategy.s["orders"]
                if normalize_side(o.get("side")) == "SELL"
            )
            return max(0.0, safe_nonneg_float(self.strategy.s.get("qty"), default=0.0) - reserved_sell)
        if self.cash is None:
            return self.client._default_balance
        reserved = sum(
            o["vol"] * o["price"] * (1 + self.fee_model.rate_pct(market=is_market_order(o)) / 100)
            for o in self.strategy.s["orders"]
            if normalize_side(o.get("side")) == "BUY"
        )
        return max(0.0, self.cash - reserved)

    def _traced_place(self, side, vol, price, kind, amount=0.0, market=False, **place_kw):
        market = _parse_bool_flag(market, False)
        side_norm = normalize_side(side)
        vol_f = safe_pos_float(vol, default=None)
        px_f = safe_pos_float(price, default=None)
        if self.cash is not None and side_norm == "BUY":
            insufficient = (
                vol_f is not None
                and px_f is not None
                and (
                    round(vol_f, self.strategy.vol_dec)
                    * round(px_f, self.strategy.price_dec)
                    * (1 + self.fee_model.rate_pct(market=market) / 100)
                    > self._free_balance_for_asset(self.params.currency) + 1e-9
                )
            )
            if vol_f is None or px_f is None or insufficient:
                self.cash_refusals += 1 if insufficient else 0
                return False
        accepted = self._original_place(side, vol, price, kind, amount=amount, market=market, **place_kw)
        if accepted and self.include_decision_trace and px_f is not None and vol_f is not None:
            self.decision_trace.append({
                "bar": self.current_bar,
                "side": side,
                "kind": kind,
                "market": market,
                "price": round(px_f, 10),
                "qty": round(vol_f, 10),
            })
        return accepted

    def _settle_order_fill(
        self,
        order_to_remove: dict | None,
        fill_order: dict,
        side: str,
        volume: float,
        price: float,
        *,
        market: bool,
        final: bool,
    ) -> None:
        if order_to_remove is not None:
            self.strategy._remove(order_to_remove)
        fee = self.fee_model.rate_pct(market=market) / 100.0 * volume * price
        if self.cash is not None:
            self.cash += (-1 if side == "buy" else 1) * volume * price - fee
            if self.cash < -1e-8:
                raise ValueError("cash account overdrawn by simulated execution")
            if self.cash < 0.0:
                self.cash = 0.0
            self.min_cash = min(self.min_cash, self.cash)
        self.strategy._apply_fill(fill_order, volume, price, fee=fee, final=final)
        self.fills += 1
        self.turnover_notional += volume * price
        if side == "sell" and self.strategy.s.get("cycle", 1) > self.cycle0 + len(self.trade_pnls):
            self._record_closed_cycle(self.strategy.s["realized_net"])

    def _cap_buy_fill_to_cash(
        self,
        order: dict,
        fill_order: dict,
        volume: float,
        price: float,
        *,
        market: bool,
        complete: bool,
    ) -> tuple[float, bool] | None:
        if self.cash is None:
            return volume, complete
        fee_mult = 1.0 + self.fee_model.rate_pct(market=market) / 100.0
        needed = volume * price * fee_mult
        if needed <= self.cash + 1e-8:
            return volume, complete
        max_vol = max(0.0, self.cash / (price * fee_mult)) if price * fee_mult > 0 else 0.0
        if max_vol <= 1e-12:
            self.strategy._remove(order)
            self.cash_refusals += 1
            return None
        orig_vol = volume
        volume = min(volume, max_vol)
        fill_order["vol"] = volume
        orig_amt = safe_nonneg_float(fill_order.get("amount"), default=None)
        fill_order["amount"] = (
            orig_amt * (volume / orig_vol)
            if orig_amt is not None and orig_vol > 0.0
            else volume * price
        )
        order["vol"] = 0.0
        if "amount" in order:
            order["amount"] = 0.0
        return volume, True

    def _settle_intrabar_orders(self, open_: float, high: float, low: float, close: float) -> None:
        orders = self.strategy.s["orders"]
        if not orders:
            return
        eligible_orders, is_ambiguous = select_eligible_intrabar_orders(
            orders, self.execution, high=high, low=low, limit_key="price",
        )
        if eligible_orders:
            self.ambiguous_bars += 1 if is_ambiguous else 0
            for side, order in iter_eligible_orders_in_side_order(
                eligible_orders, self.strategy.s["orders"], self.side_order
            ):
                market, price = resolve_order_fill_price(
                    order, side, open_, self.execution, limit_key="price"
                )
                fill_order, volume, complete = split_order_fill(
                    order,
                    quantity_key="vol",
                    amount_key="amount" if side == "buy" else None,
                    ratio=self.execution.partial_fill_ratio,
                    force_full=market,
                )
                if side == "sell":
                    volume, complete = clamp_sell_fill_to_held(volume, self.strategy.s["qty"], complete)
                    if volume <= 0.0:
                        self.strategy._remove(order)
                        continue
                    fill_order["vol"] = volume
                elif side == "buy":
                    capped = self._cap_buy_fill_to_cash(
                        order, fill_order, volume, price, market=market, complete=complete
                    )
                    if capped is None:
                        continue
                    volume, complete = capped
                self._settle_order_fill(
                    order if complete else None,
                    fill_order,
                    side,
                    volume,
                    price,
                    market=market,
                    final=complete,
                )
        if self.check_ttl and self.strategy.s["orders"]:
            for order in list(self.strategy.s["orders"]):
                if order["side"] == "buy" and not is_market_order(order) and close > order["price"] * 1.003:
                    self.strategy._remove(order)

    def _step_bar(self, bar_index: int, open_: float, high: float, low: float, close: float) -> None:
        self.current_bar = bar_index
        replay_time = self.timestamps[bar_index] if self.timestamps else bar_index * self.bar_step_sec
        self.strategy.set_simulated_time(replay_time)
        if self.dca_drop_series is not None:
            self.strategy.p.dca_drop_pct = (
                safe_pos_float(self.dca_drop_series[bar_index], default=self.default_dca_drop)
                if bar_index < len(self.dca_drop_series)
                else self.default_dca_drop
            )
        if self.reentry_series is not None:
            self.strategy.p.reentry_drop_pct = (
                safe_nonneg_float(self.reentry_series[bar_index], default=self.default_reentry_drop)
                if bar_index < len(self.reentry_series)
                else self.default_reentry_drop
            )
        self._settle_intrabar_orders(open_, high, low, close)
        self.strategy.step(close, timestamp=replay_time, high=high if self.intrabar_high else None)
        q = self.strategy.s["qty"]
        upnl = (close - self.strategy.s["cost"] / q) * q if q > 1e-12 else 0.0
        self._record_bar_equity(self.initial_capital + self.strategy.s["realized_net"] + upnl, q > 1e-12)

    def _finalize_end_of_replay(self) -> None:
        last_close, end_exit_orders = iter_end_of_replay_market_sell_orders(
            self.ohlc, self.strategy.s["orders"], self.strategy.s["qty"]
        )
        self.last_close = last_close
        settled_end_exit = False
        for order in end_exit_orders:
            if self.strategy.s["qty"] <= 1e-12:
                break
            volume = min(require_pos_float(order["vol"], "order_vol"), self.strategy.s["qty"])
            self._settle_order_fill(
                order,
                order,
                "sell",
                volume,
                self.execution.market_price("sell", last_close),
                market=True,
                final=True,
            )
            settled_end_exit = True
        if settled_end_exit:
            q = self.strategy.s["qty"]
            upnl = (last_close - self.strategy.s["cost"] / q) * q if q > 1e-12 else 0.0
            self._overwrite_tail_equity(self.initial_capital + self.strategy.s["realized_net"] + upnl, q > 1e-12)

    def _build_summary(self) -> dict:
        qty = self.strategy.s["qty"]
        final_upnl = (self.last_close - self.strategy.s["cost"] / qty) * qty if qty > 1e-12 else 0.0
        result = self._build_summary_from_state(
            realized=self.strategy.s["realized_gross"],
            net=self.strategy.s["realized_net"],
            fees=self.strategy.s["fees_total"],
            total=self.strategy.s["realized_net"] + final_upnl,
            final_upnl=final_upnl,
            cycles=self.strategy.s.get("cycle", 1) - self.cycle0,
            open_qty=qty,
        )
        if self.include_decision_trace:
            result["decision_trace"] = self.decision_trace
        if self.cash is not None:
            result["funding"] = {
                "initial_cash": self.initial_cash,
                "final_cash": round_metric(self.cash),
                "minimum_cash": round_metric(self.min_cash),
                "refused_buys": self.cash_refusals,
            }
        global _LAST_REPLAY_DIAG
        _LAST_REPLAY_DIAG = {
            "max_active_surge_vol": self.strategy._max_active_surge_vol,
            "surge_win_check_ts": tuple(self.strategy._surge_win_check_ts),
        }
        return result


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
    dca_drop_series=None,
    reentry_series=None,
    intrabar_high: bool = False,
    timestamps: tuple[float, ...] | None = None,
    warmup_timestamps: tuple[float | None, ...] | None = None,
) -> dict:
    return _SpotReplaySession(
        ohlc,
        params,
        fee_model=fee_model,
        bar_minutes=bar_minutes,
        warmup_ohlc=warmup_ohlc,
        execution=execution,
        include_decision_trace=include_decision_trace,
        initial_cash=initial_cash,
        dca_drop_series=dca_drop_series,
        reentry_series=reentry_series,
        intrabar_high=intrabar_high,
        timestamps=timestamps,
        warmup_timestamps=warmup_timestamps,
    ).run()


def make_spot_replay_adapter(
    params: StratParams,
    fee_pct: float = 0.26,
    bar_minutes: float | None = 60.0,
    execution: ExecutionModel | None = None,
    fee_model: FeeModel | None = None,
    include_decision_trace: bool = False,
    initial_cash: float | None = None,
    dca_drop_series=None,
    reentry_series=None,
    intrabar_high: bool = False,
):
    """Return a generic spot ReplayFunction compatible with evaluation.py walk-forward benchmarks."""
    def _adapter(ohlc, warmup_ohlc=(), context=None):
        ts, w_ts = extract_adapter_context_timestamps(context, empty=None)
        return run_replay(
            ohlc,
            params,
            fee_pct=fee_pct,
            bar_minutes=bar_minutes,
            warmup_ohlc=warmup_ohlc or (),
            timestamps=ts,
            warmup_timestamps=w_ts,
            execution=execution,
            fee_model=fee_model,
            include_decision_trace=include_decision_trace,
            initial_cash=initial_cash,
            dca_drop_series=dca_drop_series,
            reentry_series=reentry_series,
            intrabar_high=intrabar_high,
        )

    return _adapter


def _clone_params_with_updates(params, updates: dict[str, object]):
    if not updates:
        return params
    if dataclasses.is_dataclass(params):
        field_names = {f.name for f in dataclasses.fields(params) if f.init}
        valid_updates = {k: v for k, v in updates.items() if k in field_names}
        return dataclasses.replace(params, **valid_updates) if valid_updates else params
    cloned = copy.copy(params)
    for k, v in updates.items():
        setattr(cloned, k, v)
    return cloned


def align_spot_interval_params(params, bar_minutes: float | int | None):
    """Return a copy of params with trend/volatility intervals and regime sample cap aligned to bar_minutes."""
    bm_f = safe_pos_float(bar_minutes, default=None)
    if bm_f is None:
        return params
    itv = safe_int(bm_f, default=0)
    if itv <= 0:
        return params
    cap = _strat.MarketRegimeService.horizon_sample_capacity("long", itv)
    cur_min_samples = safe_int(getattr(params, "regime_min_samples", 20), default=20)
    clamped_min_samples = max(3, min(cur_min_samples, cap))
    if (
        getattr(params, "trend_interval", itv) == itv
        and getattr(params, "tp_trail_vol_interval", itv) == itv
        and getattr(params, "dca_vol_interval", itv) == itv
        and getattr(params, "regime_min_samples", 20) == clamped_min_samples
        and not isinstance(getattr(params, "regime_min_samples", 20), bool)
    ):
        return params

    return _clone_params_with_updates(
        params,
        {
            "trend_interval": itv,
            "tp_trail_vol_interval": itv,
            "dca_vol_interval": itv,
            "regime_min_samples": clamped_min_samples,
        },
    )


def scale_subbar_timeout_params(params, bar_minutes: float | int | None):
    """Scale sub-bar stagnation and trend-defer timeouts proportionally when bar_minutes > check_minutes."""
    bm_f = safe_pos_float(bar_minutes, default=None)
    check_min = safe_pos_float(getattr(params, "check_minutes", None), default=None)
    if bm_f is None or check_min is None or bm_f <= check_min:
        return params
    bar_scale = bm_f / check_min
    updates: dict[str, float] = {"check_minutes": bm_f}
    for attr in ("tp_stagnation_max_sec", "sl_trend_defer_max_sec"):
        cur_val = safe_pos_float(getattr(params, attr, None), default=None)
        if cur_val is not None:
            updates[attr] = round(cur_val * bar_scale, 1)
    return _clone_params_with_updates(params, updates)


def prepare_spot_interval_params(params, bar_minutes: float | int | None):
    """Align interval-dependent parameters and scale sub-bar timeouts for ``bar_minutes``."""
    return scale_subbar_timeout_params(align_spot_interval_params(params, bar_minutes), bar_minutes)


@contextlib.contextmanager
def post_exit_peak_step_scope(strat_cls, flags: dict, peak, step_fn, orig_step_fn):
    """Temporarily toggle ``flags['peak']`` and patch ``strat_cls.step`` for continuous-grid replay."""
    flags_d = flags if isinstance(flags, dict) else {}
    prev_peak = flags_d.get("peak", False)
    prev_step = getattr(strat_cls, "step", orig_step_fn)
    use_peak = _parse_bool_flag(peak, default=False)
    flags_d["peak"] = use_peak
    strat_cls.step = step_fn if use_peak else orig_step_fn
    try:
        yield
    finally:
        flags_d["peak"] = prev_peak
        strat_cls.step = prev_step


def prewarm_trend_series(
    ohlc,
    *,
    warmup: int = 40,
    params=None,
    bar_minutes: float = 240.0,
) -> None:
    """Prewarm process-level replay regime and volatility caches for an OHLC series."""
    if not ohlc or warmup is None or isinstance(warmup, bool):
        return
    bm_f = safe_pos_float(bar_minutes, default=None)
    if bm_f is None:
        return
    w_i = safe_int(warmup, default=None)
    warmup_i = max(0, w_i) if w_i is not None else 40
    if len(ohlc) <= warmup_i:
        return
    try:
        raw_params = _strat.StratParams.from_env() if params is None else params
    except (KeyError, ValueError, OSError):
        raw_params = None
    if raw_params is not None and hasattr(raw_params, "surge_guard"):
        global _LAST_REPLAY_DIAG
        saved_diag = dict(_LAST_REPLAY_DIAG)
        try:
            run_replay(
                ohlc[warmup_i:],
                align_spot_interval_params(raw_params, bm_f),
                fee_pct=0.26,
                bar_minutes=bm_f,
                warmup_ohlc=ohlc[:warmup_i],
            )
        finally:
            _LAST_REPLAY_DIAG = saved_diag


def ratchet_post_exit_peak_sell_price(
    state: dict,
    price,
    *,
    qty_eps: float = 1e-9,
    require_tp_kind: bool = False,
) -> None:
    """Ratchet ``state["last_sell_price"]`` up to the highest close since a post-TP flat state."""
    eps_f = safe_nonneg_float(qty_eps, default=1e-9)
    if not isinstance(state, dict) or safe_float(state.get("qty"), 0.0) > eps_f:
        return
    raw_sl_rb = state.get("sl_rebuy")
    if bool(raw_sl_rb) if isinstance(raw_sl_rb, dict) else bool(_parse_bool_flag(raw_sl_rb, False)):
        return
    if _parse_bool_flag(require_tp_kind, False) and state.get("last_exit_kind") != "TP":
        return
    p_f = safe_pos_float(price, 0.0)
    lsp_f = safe_pos_float(state.get("last_sell_price"), 0.0)
    if p_f > 0.0 and lsp_f > 0.0:
        state["last_sell_price"] = max(lsp_f, p_f)


def summarize_grid_replay_metrics(
    metrics: dict,
    *,
    budget: float,
    p0: float,
    p1: float,
    fee_pct: float = 0.0,
    bh_digits: int = 3,
    include_fills: bool = False,
) -> dict:
    """Summarize continuous-grid replay metrics and round-trip buy-and-hold return."""
    m_dict = metrics if isinstance(metrics, dict) else {}
    tot = safe_float(m_dict.get("total"), 0.0)
    budget_f = safe_pos_float(budget, 0.0)
    p0_f = safe_pos_float(p0, 0.0)
    p1_f = safe_nonneg_float(p1, default=None)
    digits = exact_nonneg_int(bh_digits, default=3)
    if digits is None:
        digits = 3
    fee_r = min(0.9999, max(0.0, safe_float(fee_pct, 0.0) / 100.0))
    bh_pct = (
        (((p1_f * (1.0 - fee_r)) / (p0_f * (1.0 + fee_r))) - 1.0) * 100.0
        if p0_f > 0.0 and p1_f is not None and (1.0 + fee_r) > 0.0
        else 0.0
    )
    out = {
        "ret": round(tot / budget_f * 100.0, 3) if budget_f > 0.0 else 0.0,
        "dd": round(safe_float(m_dict.get("max_drawdown_pct"), 0.0), 3),
        "cyc": safe_nonneg_int(m_dict.get("cycles"), default=0),
        "expo": round(safe_float(m_dict.get("exposure_pct"), 0.0), 1),
        "bh": round(bh_pct, digits),
    }
    if _parse_bool_flag(include_fills, False):
        out["fills"] = safe_nonneg_int(m_dict.get("fills"), default=0)
    return out


def split_warmup_ohlc(
    ohlc: list, max_warmup: int = 30, divisor: int = 5
) -> tuple[int, list, list]:
    """Partition ``ohlc`` into ``(warmup_n, warmup_ohlc, run_ohlc)`` using ``min(max_warmup, len(ohlc) // divisor)``."""
    bars = list(ohlc) if ohlc is not None and not isinstance(ohlc, (str, bytes, bytearray, dict)) else []
    div_i = safe_pos_int(divisor, default=5) or 5
    max_w = safe_nonneg_int(max_warmup, default=30)
    warmup_n = min(max_w, len(bars) // div_i)
    return warmup_n, bars[:warmup_n], bars[warmup_n:]
