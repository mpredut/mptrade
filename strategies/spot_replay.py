#!/usr/bin/env python3
"""strategies/spot_replay.py — Canonical replay engine for spot trading strategies.

Replays spot_engine.py over historical OHLC bars. The strategy decides orders,
while the harness separately models order fills, fee schedules, slippage, and
intrabar ambiguity resolution without network, notifications, or persistent state.
"""

from __future__ import annotations

from botcore import safe_float, safe_int, safe_nonneg_float, safe_pos_float
from offline.backtests.execution import (
    ExecutionModel,
    FeeModel,
    choose_intrabar_scenario,
    split_order_fill,
    validate_ohlc_bars,
)
from offline.backtests.metrics import calculate_performance_metrics
from providers.strategy_executor import PairPrecision
from strategies import spot_engine as _strat

_REPLAY_PRECISION = PairPrecision(5, 8, 0.0, "REPLAY")


def _silent(*_args, **_kwargs):
    return None


class _ReplayClient:
    """Lightweight replay venue client without unittest.mock call-recording overhead."""

    __slots__ = ("_default_balance", "_free_balance_fn")

    def __init__(self, default_balance: float) -> None:
        self._default_balance = default_balance
        self._free_balance_fn = None

    def pair_precision(self, _pair: str) -> PairPrecision:
        return _REPLAY_PRECISION

    def free_balance(self, asset: str) -> float:
        if self._free_balance_fn is not None:
            return self._free_balance_fn(asset)
        return self._default_balance


def _validate_replay(ohlc, params, bar_minutes: float | None) -> None:
    validate_ohlc_bars(ohlc, check_envelope=False)
    if bar_minutes is not None and safe_pos_float(bar_minutes, default=None) is None:
        raise ValueError("bar_minutes must be finite and positive")
    surge_window_enabled = (
        params.surge_guard and params.surge_window_hours > 0 and params.surge_move_pct > 0
    )
    if (params.trend_overlay or params.dca_trend_brake
            or params.reentry_hybrid_enabled or surge_window_enabled
            or getattr(params, "tp_dynamic_flat", False)) and (
            bar_minutes is None or float(bar_minutes) != float(params.trend_interval)):
        raise ValueError(
            "regime-aware policies require bar_minutes to equal trend_interval "
            f"({params.trend_interval} minutes); resampling is not implemented"
        )
    if (params.tp_trail_adaptive or (params.surge_guard and params.surge_dynamic)) and (
            bar_minutes is None
            or float(bar_minutes) != float(params.tp_trail_vol_interval)):
        raise ValueError(
            "adaptive trailing or surge requires bar_minutes to equal "
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
    dca_drop_series=None,
    reentry_series=None,
    intrabar_high: bool = False,
) -> dict:
    """Replay decisions, optionally enforcing a finite cash account and fee reserves.

    Omitted cash retains legacy funding assumptions for baseline compatibility.
    Explicit cash rejects unaffordable requests, reserves open BUYs, and settles
    fill notional/fees; it does not inject capital after losses or cycle resets.
    """
    _validate_replay(ohlc, params, bar_minutes)
    if bar_minutes is not None:
        bar_minutes = safe_pos_float(bar_minutes, default=None)
    include_decision_trace = _strat._parse_bool_flag(include_decision_trace, False)
    intrabar_high = _strat._parse_bool_flag(intrabar_high, False)
    if initial_cash is not None:
        ic_f = safe_pos_float(initial_cash, default=None)
        if ic_f is None:
            raise ValueError("initial_cash must be finite and positive")
        initial_cash = ic_f
    model = execution or ExecutionModel()
    fees = fee_model or FeeModel(fee_pct, fee_pct)
    return choose_intrabar_scenario(
        model,
        lambda scenario: _run_once(
            ohlc,
            params,
            fee_model=fees,
            bar_minutes=bar_minutes,
            warmup_ohlc=warmup_ohlc or (),
            execution=scenario,
            include_decision_trace=include_decision_trace,
            initial_cash=initial_cash,
            dca_drop_series=dca_drop_series,
            reentry_series=reentry_series,
            intrabar_high=intrabar_high,
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
    dca_drop_series=None,
    reentry_series=None,
    intrabar_high: bool = False,
) -> dict:
    # Replay declares its synthetic execution precision explicitly; live engines
    # must obtain this metadata from the venue.
    # Legacy replay explicitly assumes funding up to the configured cycle cap.
    # A mock or unavailable live balance must never imply unlimited buying power.
    if dca_drop_series is not None or reentry_series is not None:
        import copy
        import dataclasses
        params = dataclasses.replace(params) if dataclasses.is_dataclass(params) else copy.copy(params)
    client = _ReplayClient(float(params.effective_max_budget()))
    original_notify = _strat.notify
    original_log = _strat.log
    _strat.notify = _silent
    _strat.log = _silent
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
        for index, w_row in enumerate(
                warmup_ohlc, start=-len(warmup_ohlc)):
            if not isinstance(w_row, (list, tuple)) or len(w_row) < 4:
                continue
            close_f = safe_pos_float(w_row[3], default=None)
            if close_f is None:
                continue
            strategy._shadow_prices.append((index * warmup_step, close_f))
            if strategy._surge_replay_closes is not None:
                strategy._surge_replay_closes.append(close_f)
        strategy.set_simulated_time(0.0)
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
            client._free_balance_fn = free_quote

        def traced_place(side, vol, price, kind, amount=0.0, market=False):
            nonlocal cash_refusals
            market = _strat._parse_bool_flag(market, False)
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

        default_dca_drop = params.dca_drop_pct
        default_reentry_drop = params.reentry_drop_pct
        check_ttl = bar_minutes is None or bar_minutes >= params.order_ttl_min
        bar_step_sec = bar_minutes * 60 if bar_minutes else 1
        side_order = execution.side_order()

        def _record_sell_outcome(gross_before: float) -> None:
            nonlocal wins, cycle_net_start
            if strategy.s["realized_gross"] > gross_before:
                wins += 1
            if strategy.s.get("cycle", 1) > cycle0 + len(trade_pnls):
                trade_pnls.append(strategy.s["realized_net"] - cycle_net_start)
                cycle_net_start = strategy.s["realized_net"]

        for bar_index, row in enumerate(ohlc):
            open_, high, low, close = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            current_bar = bar_index
            replay_time = bar_index * bar_step_sec
            strategy.set_simulated_time(replay_time)
            if dca_drop_series is not None and bar_index < len(dca_drop_series):
                strategy.p.dca_drop_pct = safe_pos_float(dca_drop_series[bar_index], default=default_dca_drop)
            if reentry_series is not None and bar_index < len(reentry_series):
                strategy.p.reentry_drop_pct = safe_nonneg_float(reentry_series[bar_index], default=default_reentry_drop)

            orders = strategy.s["orders"]
            if orders:
                eligible_orders = [
                    o for o in orders
                    if o.get("market") or execution.limit_touched(
                        o["side"], high=high, low=low, limit=o["price"],
                    )
                ]
                if eligible_orders:
                    if len(eligible_orders) > 1 and {"buy", "sell"}.issubset(
                        {o["side"].lower() for o in eligible_orders}
                    ):
                        ambiguous_bars += 1

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
                                _record_sell_outcome(gross_before)

                # Live reconcile() cancels an unfilled limit BUY once it is older than
                # order_ttl_min and the price has risen more than 0.3% above it; step() then
                # re-places it at the current level. Replay never calls reconcile(), so without
                # this an ENTRY left below a rally would wait forever for a pullback.
                if check_ttl and strategy.s["orders"]:
                    for order in list(strategy.s["orders"]):
                        if (order["side"] == "buy" and not order.get("market")
                                and close > order["price"] * 1.003):
                            strategy._remove(order)
            strategy.step(close, timestamp=replay_time, high=high if intrabar_high else None)
            qty = strategy.s["qty"]
            unrealized = (
                (close - strategy.s["cost"] / qty) * qty if qty > 1e-12 else 0.0
            )
            equity_curve.append(initial_capital + strategy.s["realized_net"] + unrealized)
            exposure.append(qty > 1e-12)

        for order in list(strategy.s["orders"]):
            if order.get("market") and order["side"] == "sell" and strategy.s["qty"] > 1e-12:
                volume = min(float(order["vol"]), strategy.s["qty"])
                price = execution.market_price("sell", float(ohlc[-1][3]))
                gross_before = strategy.s["realized_gross"]
                fee = fee_model.rate_pct(market=True) / 100.0 * volume * price
                if cash is not None:
                    cash += volume * price - fee
                    min_cash = min(min_cash, cash)
                strategy._remove(order)
                strategy._apply_fill(
                    order, volume, price, fee=fee, final=True,
                )
                fill_count += 1
                turnover_notional += volume * price
                _record_sell_outcome(gross_before)
                if equity_curve:
                    equity_curve[-1] = initial_capital + strategy.s["realized_net"]
    finally:
        _strat.notify = original_notify
        _strat.log = original_log

    qty = strategy.s["qty"]
    final_upnl = (
        (float(ohlc[-1][3]) - strategy.s["cost"] / qty) * qty if qty > 1e-12 else 0.0
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
    def rounded(value):
        vf = safe_float(value, default=None) if isinstance(value, (int, float)) else None
        return round(vf, 10) if vf is not None else value
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
        "initial_capital": rounded(initial_capital),
        "account_currency": params.currency,
    }
    result.update({key: rounded(value) for key, value in performance.items()})
    if include_decision_trace:
        result["decision_trace"] = decision_trace
    if cash is not None:
        result["funding"] = {"initial_cash": initial_cash, "final_cash": rounded(cash),
                             "minimum_cash": rounded(min_cash), "refused_buys": cash_refusals}
    global _LAST_REPLAY_DIAG
    _LAST_REPLAY_DIAG = {
        "max_active_surge_vol": strategy._max_active_surge_vol,
        "surge_win_check_ts": tuple(strategy._surge_win_check_ts),
    }
    return result


_LAST_REPLAY_DIAG: dict = {}


def make_spot_replay_adapter(
    params,
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
    def _adapter(ohlc, warmup_ohlc=(), _context=None):
        return run_replay(
            ohlc,
            params,
            fee_pct=fee_pct,
            bar_minutes=bar_minutes,
            warmup_ohlc=warmup_ohlc or (),
            execution=execution,
            fee_model=fee_model,
            include_decision_trace=include_decision_trace,
            initial_cash=initial_cash,
            dca_drop_series=dca_drop_series,
            reentry_series=reentry_series,
            intrabar_high=intrabar_high,
        )

    return _adapter


def align_spot_interval_params(params, bar_minutes: float | int | None):
    """Return a copy of params with trend/volatility intervals and regime sample cap aligned to bar_minutes."""
    bm_f = safe_pos_float(bar_minutes, default=None)
    if bm_f is None or int(bm_f) <= 0:
        return params
    itv = int(bm_f)
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
    import copy
    import dataclasses

    if dataclasses.is_dataclass(params):
        field_names = {f.name for f in dataclasses.fields(params) if f.init}
        updates = {
            k: v
            for k, v in (
                ("trend_interval", itv),
                ("tp_trail_vol_interval", itv),
                ("dca_vol_interval", itv),
                ("regime_min_samples", clamped_min_samples),
            )
            if k in field_names
        }
        return dataclasses.replace(params, **updates) if updates else params
    cloned = copy.copy(params)
    cloned.trend_interval = itv
    cloned.tp_trail_vol_interval = itv
    cloned.dca_vol_interval = itv
    cloned.regime_min_samples = clamped_min_samples
    return cloned


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
    w_f = safe_float(warmup, default=None)
    warmup_i = max(0, int(w_f)) if w_f is not None else 40
    if len(ohlc) <= warmup_i:
        return
    try:
        raw_params = _strat.StratParams.from_env() if params is None else params
    except Exception:
        raw_params = None
    if raw_params is None:
        return
    p = align_spot_interval_params(raw_params, bm_f)
    run_replay(
        ohlc[warmup_i:],
        p,
        fee_pct=0.26,
        bar_minutes=bm_f,
        warmup_ohlc=ohlc[:warmup_i],
    )
