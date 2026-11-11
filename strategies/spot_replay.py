#!/usr/bin/env python3
"""strategies/spot_replay.py -- Canonical replay engine for spot trading strategies.

Replays spot_engine.py over historical OHLC bars. The strategy decides orders,
while the harness separately models order fills, fee schedules, slippage, and
intrabar ambiguity resolution without network, notifications, or persistent state.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses

from botcore import (
    _parse_bool_flag,
    exact_nonneg_int,
    normalize_epoch_sec,
    normalize_side,
    require_pos_float,
    safe_float,
    safe_int,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
)
from strategies.replay_execution import (
    ExecutionModel,
    FeeModel,
    choose_intrabar_scenario,
    select_eligible_intrabar_orders,
    split_order_fill,
    validate_ohlc_bars,
)
from strategies.replay_metrics import calculate_performance_metrics, round_metric
from providers.strategy_executor import PairPrecision
from strategies import spot_engine as _strat
from strategies.spot_engine import StratParams

_REPLAY_PRECISION = PairPrecision(5, 8, 0.0, "REPLAY")
_LAST_REPLAY_DIAG: dict = {}


def _silent(*_args, **_kwargs):
    return None


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
        getattr(params, "surge_guard", False)
        and getattr(params, "surge_window_hours", 0) > 0
        and getattr(params, "surge_move_pct", 0) > 0
    )
    return bool(
        getattr(params, "trend_overlay", False)
        or getattr(params, "dca_trend_brake", False)
        or getattr(params, "reentry_hybrid_enabled", False)
        or surge_window_enabled
        or getattr(params, "tp_dynamic_flat", False)
    )


def required_replay_intervals(params) -> set[int]:
    """Return the set of bar intervals (in minutes) required by active policies on ``params``."""
    required = {safe_int(params.trend_interval, 0)} if params_require_trend_interval(params) else set()
    if getattr(params, "tp_trail_adaptive", False) or (
        getattr(params, "surge_guard", False) and getattr(params, "surge_dynamic", False)
    ):
        required.add(safe_int(params.tp_trail_vol_interval, 0))
    if getattr(params, "dca_vol_scale_k", 0):
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
    if (params.tp_trail_adaptive or (params.surge_guard and params.surge_dynamic)) and (
        bm_val is None or bm_val != safe_float(params.tp_trail_vol_interval, default=-1.0)
    ):
        raise ValueError(
            f"adaptive trailing or surge requires bar_minutes to equal tp_trail_vol_interval ({params.tp_trail_vol_interval} minute)"
        )
    if params.dca_vol_scale_k and (bm_val is None or bm_val != safe_float(params.dca_vol_interval, default=-1.0)):
        raise ValueError(
            f"dca_vol_scale_k requires bar_minutes to equal dca_vol_interval ({params.dca_vol_interval} minute)"
        )
    if params.reentry_adaptive and bm_val is None:
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
    if initial_cash is not None:
        initial_cash = safe_pos_float(initial_cash, default=None)
        if initial_cash is None:
            raise ValueError("initial_cash must be finite and positive")
    ts_tuple: tuple[float, ...] | None = None
    if timestamps is not None:
        ts_raw = list(timestamps)
        if len(ts_raw) != len(ohlc):
            raise ValueError(f"timestamps length ({len(ts_raw)}) must match ohlc length ({len(ohlc)})")
        parsed_ts: list[float] = []
        for t_item in ts_raw:
            tf = normalize_epoch_sec(t_item, default=None, allow_zero=True)
            if tf is None:
                raise ValueError(f"invalid replay timestamp: {t_item!r}")
            parsed_ts.append(tf)
        ts_tuple = tuple(parsed_ts)
    warmup_ts_tuple: tuple[float | None, ...] | None = None
    if warmup_timestamps is not None:
        warmup_ts_tuple = tuple(
            normalize_epoch_sec(wt, default=None, allow_zero=True) for wt in warmup_timestamps
        )
    model = execution or ExecutionModel()
    fees = fee_model or FeeModel(fee_pct, fee_pct)
    warmup_rows = tuple(warmup_ohlc) if warmup_ohlc else ()
    scenario_diags: dict[str, dict] = {}

    def _eval_scenario(scenario: ExecutionModel) -> dict:
        out = _run_once(
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
        )
        scenario_diags[scenario.intrabar_policy] = dict(_LAST_REPLAY_DIAG)
        return out

    chosen = choose_intrabar_scenario(model, _eval_scenario)
    selected_policy = chosen.get("intrabar_policy_selected")
    if selected_policy in scenario_diags:
        global _LAST_REPLAY_DIAG
        _LAST_REPLAY_DIAG = scenario_diags[selected_policy]
    return chosen


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
    if dca_drop_series is not None or reentry_series is not None:
        params = dataclasses.replace(params) if dataclasses.is_dataclass(params) else copy.copy(params)
    allocated_budget = require_pos_float(params.allocated_budget(), "allocated_budget")
    client = _ReplayClient(allocated_budget)
    original_notify, original_log = _strat.notify, _strat.log
    _strat.notify = _strat.log = _silent
    try:
        strategy = _strat.Strategy(
            client, "REPLAY", params, dry_run=True, initial_state=_strat._new_state(), replay_mode=True,
        )
        warmup_step = bar_minutes * 60.0 if bar_minutes else 1.0
        base_ts = timestamps[0] if timestamps else 0.0
        use_warm_ts = bool(warmup_timestamps is not None and len(warmup_timestamps) == len(warmup_ohlc))
        valid_warmup: list[tuple[float | None, float]] = []
        for raw_idx, w_row in enumerate(warmup_ohlc):
            if not isinstance(w_row, (list, tuple)) or len(w_row) < 4:
                continue
            close_f = safe_pos_float(w_row[3], default=None)
            if close_f is not None:
                wt_val = warmup_timestamps[raw_idx] if use_warm_ts else None
                valid_warmup.append((wt_val, close_f))
        n_warm = len(valid_warmup)
        for idx_w, (wt_val, close_f) in enumerate(valid_warmup):
            fallback_wt = base_ts + (idx_w - n_warm) * warmup_step
            eff_wt = wt_val if wt_val is not None else fallback_wt
            strategy._shadow_prices.append((eff_wt, close_f))
            if strategy._surge_replay_closes is not None:
                strategy._surge_replay_closes.append(close_f)
        strategy.set_simulated_time(base_ts)
        strategy._save = _silent
        decision_trace = []
        current_bar = -1
        original_place = strategy._place
        cash = require_pos_float(initial_cash, "initial_cash") if initial_cash is not None else None
        min_cash = cash
        cash_refusals = 0

        def free_balance_for_asset(asset):
            asset_norm = str(asset or "").strip().upper()
            if asset_norm == _REPLAY_PRECISION.base_asset:
                reserved_sell = sum(
                    safe_nonneg_float(o.get("vol"), default=0.0)
                    for o in strategy.s["orders"]
                    if normalize_side(o.get("side")) == "SELL"
                )
                return max(0.0, safe_nonneg_float(strategy.s.get("qty"), default=0.0) - reserved_sell)
            if cash is None:
                return client._default_balance
            reserved = sum(
                o["vol"] * o["price"] * (1 + fee_model.rate_pct(market=bool(o.get("market"))) / 100)
                for o in strategy.s["orders"]
                if normalize_side(o.get("side")) == "BUY"
            )
            return max(0.0, cash - reserved)

        client._free_balance_fn = free_balance_for_asset
        if cash is not None:
            client.buy_fee_pct = fee_model.rate_pct(market=False)

        def traced_place(side, vol, price, kind, amount=0.0, market=False, **place_kw):
            nonlocal cash_refusals
            market = _parse_bool_flag(market, False)
            side_norm = normalize_side(side)
            vol_f = safe_pos_float(vol, default=None)
            px_f = safe_pos_float(price, default=None)
            if cash is not None and side_norm == "BUY":
                if vol_f is None or px_f is None:
                    return False
                required = round(vol_f, strategy.vol_dec) * round(px_f, strategy.price_dec)
                required *= 1 + fee_model.rate_pct(market=market) / 100
                if required > free_balance_for_asset(params.currency) + 1e-9:
                    cash_refusals += 1
                    return False
            accepted = original_place(side, vol, price, kind, amount=amount, market=market, **place_kw)
            if accepted and include_decision_trace and px_f is not None and vol_f is not None:
                decision_trace.append({
                    "bar": current_bar,
                    "side": side,
                    "kind": kind,
                    "market": market,
                    "price": round(px_f, 10),
                    "qty": round(vol_f, 10),
                })
            return accepted

        if include_decision_trace or cash is not None:
            strategy._place = traced_place
        cycle0 = strategy.s.get("cycle", 1)
        wins = fill_count = ambiguous_bars = 0
        turnover_notional = 0.0
        trade_pnls = []
        cycle_net_start = strategy.s["realized_net"]
        initial_capital = allocated_budget if cash is None else cash
        equity_curve = [initial_capital]
        exposure = []

        default_dca_drop, default_reentry_drop = params.dca_drop_pct, params.reentry_drop_pct
        check_ttl = bar_minutes is None or bar_minutes >= params.order_ttl_min
        bar_step_sec = bar_minutes * 60 if bar_minutes else 1
        side_order = execution.side_order()

        def _record_sell_outcome() -> None:
            nonlocal wins, cycle_net_start
            if strategy.s.get("cycle", 1) > cycle0 + len(trade_pnls):
                cycle_pnl = strategy.s["realized_net"] - cycle_net_start
                trade_pnls.append(cycle_pnl)
                if cycle_pnl > 0.0:
                    wins += 1
                cycle_net_start = strategy.s["realized_net"]

        for bar_index, row in enumerate(ohlc):
            open_ = require_pos_float(row[0], "open")
            high = require_pos_float(row[1], "high")
            low = require_pos_float(row[2], "low")
            close = require_pos_float(row[3], "close")
            current_bar = bar_index
            replay_time = timestamps[bar_index] if timestamps else bar_index * bar_step_sec
            strategy.set_simulated_time(replay_time)
            if dca_drop_series is not None:
                strategy.p.dca_drop_pct = (
                    safe_pos_float(dca_drop_series[bar_index], default=default_dca_drop)
                    if bar_index < len(dca_drop_series)
                    else default_dca_drop
                )
            if reentry_series is not None:
                strategy.p.reentry_drop_pct = (
                    safe_nonneg_float(reentry_series[bar_index], default=default_reentry_drop)
                    if bar_index < len(reentry_series)
                    else default_reentry_drop
                )

            orders = strategy.s["orders"]
            if orders:
                eligible_orders, is_ambiguous = select_eligible_intrabar_orders(
                    orders, execution, high=high, low=low, limit_key="price",
                )
                if eligible_orders:
                    ambiguous_bars += 1 if is_ambiguous else 0
                    for side in side_order:
                        for order in eligible_orders:
                            if order not in strategy.s["orders"] or order["side"] != side:
                                continue
                            market = bool(order.get("market"))
                            fill_order, volume, complete = split_order_fill(
                                order,
                                quantity_key="vol",
                                amount_key="amount" if side == "buy" else None,
                                ratio=execution.partial_fill_ratio,
                                force_full=market,
                            )
                            price = (
                                execution.market_price(side, open_)
                                if market
                                else require_pos_float(order["price"], "order_price")
                            )
                            if side == "sell":
                                held_qty = strategy.s["qty"]
                                if volume >= held_qty - 1e-12:
                                    volume = held_qty
                                    complete = True
                                if volume <= 1e-12:
                                    strategy._remove(order)
                                    continue
                                fill_order["vol"] = volume
                            elif cash is not None and market:
                                fee_mult = 1.0 + fee_model.rate_pct(market=True) / 100.0
                                needed = volume * price * fee_mult
                                if needed > cash + 1e-8:
                                    max_vol = max(0.0, cash / (price * fee_mult)) if price * fee_mult > 0 else 0.0
                                    if max_vol <= 1e-12:
                                        cash_refusals += 1
                                        strategy._remove(order)
                                        continue
                                    volume = min(volume, max_vol)
                                    fill_order["vol"] = volume
                                    complete = True
                            if complete:
                                strategy._remove(order)
                            fee = fee_model.rate_pct(market=market) / 100.0 * volume * price
                            if cash is not None:
                                cash += (-1 if side == "buy" else 1) * volume * price - fee
                                min_cash = min(min_cash, cash)
                                if cash < -1e-8:
                                    raise ValueError("cash account overdrawn by simulated execution")
                            strategy._apply_fill(fill_order, volume, price, fee=fee, final=complete)
                            fill_count += 1
                            turnover_notional += volume * price
                            if side == "sell":
                                _record_sell_outcome()

                if check_ttl and strategy.s["orders"]:
                    for order in list(strategy.s["orders"]):
                        if order["side"] == "buy" and not order.get("market") and close > order["price"] * 1.003:
                            strategy._remove(order)
            strategy.step(close, timestamp=replay_time, high=high if intrabar_high else None)
            qty = strategy.s["qty"]
            unrealized = (close - strategy.s["cost"] / qty) * qty if qty > 1e-12 else 0.0
            equity_curve.append(initial_capital + strategy.s["realized_net"] + unrealized)
            exposure.append(qty > 1e-12)

        last_close = require_pos_float(ohlc[-1][3], "last_close")
        settled_end_exit = False
        for order in list(strategy.s["orders"]):
            if order.get("market") and order["side"] == "sell" and strategy.s["qty"] > 1e-12:
                volume = min(require_pos_float(order["vol"], "order_vol"), strategy.s["qty"])
                price = execution.market_price("sell", last_close)
                fee = fee_model.rate_pct(market=True) / 100.0 * volume * price
                if cash is not None:
                    cash += volume * price - fee
                    min_cash = min(min_cash, cash)
                strategy._remove(order)
                strategy._apply_fill(order, volume, price, fee=fee, final=True)
                fill_count += 1
                turnover_notional += volume * price
                _record_sell_outcome()
                settled_end_exit = True
        if settled_end_exit:
            rem_qty = strategy.s["qty"]
            rem_upnl = (last_close - strategy.s["cost"] / rem_qty) * rem_qty if rem_qty > 1e-12 else 0.0
            if equity_curve:
                equity_curve[-1] = initial_capital + strategy.s["realized_net"] + rem_upnl
            if exposure:
                exposure[-1] = rem_qty > 1e-12
    finally:
        _strat.notify, _strat.log = original_notify, original_log

    qty = strategy.s["qty"]
    final_upnl = (last_close - strategy.s["cost"] / qty) * qty if qty > 1e-12 else 0.0
    result = build_replay_summary(
        realized=strategy.s["realized_gross"],
        net=strategy.s["realized_net"],
        fees=strategy.s["fees_total"],
        total=strategy.s["realized_net"] + final_upnl,
        final_upnl=final_upnl,
        cycles=strategy.s.get("cycle", 1) - cycle0,
        wins=wins,
        open_qty=qty,
        fills=fill_count,
        ambiguous_bars=ambiguous_bars,
        initial_capital=initial_capital,
        account_currency=params.currency,
        equity_curve=equity_curve,
        exposure=exposure,
        trade_pnls=trade_pnls,
        turnover_notional=turnover_notional,
        bar_minutes=bar_minutes,
    )
    if include_decision_trace:
        result["decision_trace"] = decision_trace
    if cash is not None:
        result["funding"] = {
            "initial_cash": initial_cash,
            "final_cash": round_metric(cash),
            "minimum_cash": round_metric(min_cash),
            "refused_buys": cash_refusals,
        }
    global _LAST_REPLAY_DIAG
    _LAST_REPLAY_DIAG = {
        "max_active_surge_vol": strategy._max_active_surge_vol,
        "surge_win_check_ts": tuple(strategy._surge_win_check_ts),
    }
    return result


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
        ts = list(context.timestamps) if context and getattr(context, "timestamps", None) else None
        w_ts = (
            list(context.warmup_timestamps)
            if context and getattr(context, "warmup_timestamps", None)
            else None
        )
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
    prev_peak = flags.get("peak", False)
    prev_step = getattr(strat_cls, "step", orig_step_fn)
    use_peak = bool(peak)
    flags["peak"] = use_peak
    strat_cls.step = step_fn if use_peak else orig_step_fn
    try:
        yield
    finally:
        flags["peak"] = prev_peak
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
    if not isinstance(state, dict) or safe_float(state.get("qty"), 0.0) > qty_eps:
        return
    if not state.get("last_sell_price") or state.get("sl_rebuy"):
        return
    if require_tp_kind and state.get("last_exit_kind") != "TP":
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
