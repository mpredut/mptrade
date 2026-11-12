#!/usr/bin/env python3
"""
replay.py -- Canonical replay engine for Hyperliquid perpetual strategies.

Executes the production Strategy engine bar-by-bar over historical OHLC candles,
modeling perpetual funding, limit fills against candle extremes (high/low),
stop-loss execution, and equity drawdown without network or side-effects.
Standardizes metrics reporting through calculate_performance_metrics and execution
modeling through ExecutionModel.
"""

from __future__ import annotations

import bisect
import dataclasses
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from botcore import (  # noqa: E402
    _parse_bool_flag,
    ensure_local_module_priority,
    require_nonneg_float,
    require_pos_float,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
)
ensure_local_module_priority(_HERE, "strategy")
from strategies.spot_engine import StratParams as SpotStratParams  # noqa: E402
from strategies.spot_replay import (  # noqa: E402
    align_spot_interval_params,
    build_replay_summary,
    make_spot_replay_adapter,
    scale_subbar_timeout_params,
)
from strategy import Strategy, StratParams, HL_FEE_PCT  # noqa: E402
from strategies.replay_execution import (  # noqa: E402
    ExecutionModel,
    choose_intrabar_scenario,
    split_order_fill,
    validate_ohlc_bars,
)
from strategies.replay_metrics import round_metric  # noqa: E402


def _to_funding_ms(t_raw: int | float | None) -> float | None:
    t_val = safe_float(t_raw, default=None)
    if t_val is None:
        return None
    if 0 < t_val < 10_000_000_000:
        t_val *= 1000.0
    return t_val


def _lookup_sorted_funding_ms(fund_ts: list[float], fund_rates: list[float], t_ms_norm: float) -> float:
    if not fund_ts:
        return 0.0
    idx = bisect.bisect_right(fund_ts, t_ms_norm) - 1
    return fund_rates[idx] if idx >= 0 else 0.0


def _normalize_funding_series(fund: list[tuple[int, float]] | None) -> tuple[list[float], list[float]]:
    if not fund:
        return [], []
    clean_fund: list[tuple[float, float]] = []
    for item in fund:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        ts_f = _to_funding_ms(item[0])
        rate_f = safe_float(item[1], default=None)
        if ts_f is not None and rate_f is not None:
            clean_fund.append((ts_f, rate_f))
    if not clean_fund:
        return [], []
    if any(clean_fund[i][0] > clean_fund[i + 1][0] for i in range(len(clean_fund) - 1)):
        clean_fund.sort(key=lambda item: item[0])
    return [item[0] for item in clean_fund], [item[1] for item in clean_fund]


def _funding_rate_from_sorted(fund_ts: list[float], fund_rates: list[float], t_ms: int | float) -> float:
    if not fund_ts:
        return 0.0
    t_val = _to_funding_ms(t_ms)
    if t_val is None:
        return 0.0
    return _lookup_sorted_funding_ms(fund_ts, fund_rates, t_val)


def funding_at(fund: list[tuple[int, float]] | None, t_ms: int) -> float:
    """Return the funding rate active at t_ms, using the most recent rate at or before t.

    Automatically normalises Unix-second timestamps (< 1e10) to milliseconds so
    both direct replay callers (ms) and walk-forward evaluation contexts (s) match.
    """
    fund_ts, fund_rates = _normalize_funding_series(fund)
    return _funding_rate_from_sorted(fund_ts, fund_rates, t_ms)


def _resolve_hl_strat_params(
    params: StratParams | dict | None,
    direction: str | None,
    fee_pct: float | None,
) -> StratParams:
    if isinstance(params, StratParams):
        strat_params = dataclasses.replace(params)
        if direction is not None:
            if isinstance(direction, bool) or not str(direction).strip():
                raise ValueError(f"Invalid Strategy direction: {direction!r}")
            strat_params.direction = str(direction).strip().lower()
    elif isinstance(params, dict):
        strat_params = StratParams.from_dict(params, direction=direction)
    elif params is None:
        eff_dir = direction if direction is not None else "long"
        if isinstance(eff_dir, bool) or not str(eff_dir).strip():
            raise ValueError(f"Invalid Strategy direction: {eff_dir!r}")
        strat_params = StratParams(direction=str(eff_dir).strip().lower())
    else:
        raise ValueError(f"Unsupported params type: {type(params).__name__}")

    if fee_pct is not None:
        fee_f = safe_nonneg_float(fee_pct, default=None)
        if fee_f is None or isinstance(fee_pct, bool) or fee_f >= 100.0:
            raise ValueError(f"invalid fee_pct: {fee_pct!r}")
        strat_params.fee_pct = fee_f
    return strat_params


def run_replay(
    ohlc: list[tuple[float, float, float, float]],
    ts: list[int] | None = None,
    fund: list[tuple[int, float]] | None = None,
    sig: list[str] | None = None,
    params: StratParams | dict | None = None,
    direction: str | None = None,
    coin: str = "HYPE",
    bar_minutes: float | None = 60.0,
    periods_per_year: float | None = None,
    execution: ExecutionModel | None = None,
    fee_pct: float | None = None,
) -> dict:
    """Replay historical candles through the live Strategy engine."""
    if bar_minutes is not None and safe_pos_float(bar_minutes, default=None) is None:
        raise ValueError(f"bar_minutes must be finite and positive, got {bar_minutes!r}")
    if ts is not None and len(ts) != len(ohlc):
        raise ValueError(f"ts length ({len(ts)}) must match ohlc length ({len(ohlc)})")
    if fund and ts is None and ohlc:
        raise ValueError("ts is required when fund is provided")
    if sig is not None and len(sig) != len(ohlc):
        raise ValueError(f"sig length ({len(sig)}) must match ohlc length ({len(ohlc)})")

    strat_params = _resolve_hl_strat_params(params, direction, fee_pct)
    Strategy(
        client=None,
        coin=coin,
        params=dataclasses.replace(strat_params),
        dry_run=True,
        replay_mode=True,
    )
    init_cap = require_pos_float(strat_params.max_budget, "max_budget")
    valid_bar_minutes = safe_pos_float(bar_minutes, default=None)

    if not ohlc:
        return build_replay_summary(
            realized=0.0,
            net=0.0,
            fees=0.0,
            total=0.0,
            final_upnl=0.0,
            cycles=0,
            wins=0,
            open_qty=0.0,
            fills=0,
            ambiguous_bars=0,
            initial_capital=init_cap,
            account_currency=strat_params.currency,
            equity_curve=[init_cap],
            exposure=[],
            trade_pnls=[],
            turnover_notional=0.0,
            bar_minutes=valid_bar_minutes,
            periods_per_year=periods_per_year,
            extra_fields={"funding": 0.0},
        )

    fund_ts, fund_rates = _normalize_funding_series(fund)
    model = execution or ExecutionModel()
    return choose_intrabar_scenario(
        model,
        lambda scenario: _run_once(
            ohlc,
            ts=ts,
            fund_ts=fund_ts,
            fund_rates=fund_rates,
            sig=sig,
            params=strat_params,
            direction=None,
            coin=coin,
            bar_minutes=bar_minutes,
            periods_per_year=periods_per_year,
            execution=scenario,
            fee_pct=None,
        ),
    )


def _run_once(
    ohlc: list[tuple[float, float, float, float]],
    ts: list[int] | None = None,
    fund_ts: list[float] | None = None,
    fund_rates: list[float] | None = None,
    sig: list[str] | None = None,
    params: StratParams | dict | None = None,
    direction: str | None = None,
    coin: str = "HYPE",
    bar_minutes: float | None = 60.0,
    periods_per_year: float | None = None,
    execution: ExecutionModel | None = None,
    fee_pct: float | None = None,
) -> dict:
    exec_model = execution or ExecutionModel()
    strat_params = _resolve_hl_strat_params(params, direction, fee_pct)
    initial_capital = require_pos_float(strat_params.max_budget, "max_budget")

    strategy = Strategy(
        client=None,
        coin=coin,
        params=strat_params,
        dry_run=True,
        replay_mode=True,
    )

    validate_ohlc_bars(ohlc, check_envelope=True)

    equity_curve = [initial_capital]
    exposure: list[bool] = []
    trade_pnls: list[float] = []
    cycle_net_start = 0.0
    cycle_funding_start = 0.0
    net_wins = 0
    turnover_notional = 0.0
    fills = 0
    funding = 0.0
    ambiguous_bars = 0
    liquidated = False
    c = require_pos_float(ohlc[0][3], "close")

    open_side = "buy" if strategy.sign > 0 else "sell"
    close_side = "sell" if strategy.sign > 0 else "buy"
    order_placed_bar: dict[str, int] = {}
    valid_bar_minutes = safe_pos_float(bar_minutes, default=None)
    eff_bar_minutes = valid_bar_minutes if valid_bar_minutes is not None else 60.0
    lot_step = 10.0 ** (-strategy.sz_dec)
    dust_eps = max(1e-12, 0.5 * lot_step)
    f_ts = fund_ts or []
    f_rates = fund_rates or []

    def _bar_funding_rate_sum(t_ms_raw: int | float | None) -> float:
        if not f_ts:
            return 0.0
        t_val = _to_funding_ms(t_ms_raw)
        if t_val is None:
            return 0.0
        if eff_bar_minutes <= 60.0 or t_val < 0:
            return _lookup_sorted_funding_ms(f_ts, f_rates, t_val) * (eff_bar_minutes / 60.0)
        rem_hours = eff_bar_minutes / 60.0
        acc = 0.0
        hour_idx = 0
        while rem_hours > 1e-9:
            weight = min(1.0, rem_hours)
            acc += _lookup_sorted_funding_ms(f_ts, f_rates, t_val + hour_idx * 3_600_000.0) * weight
            rem_hours -= weight
            hour_idx += 1
        return acc

    def _record_closed_cycle() -> None:
        nonlocal cycle_net_start, cycle_funding_start, net_wins
        cycle_pnl = (strategy.s["realized_net"] - cycle_net_start) + (funding - cycle_funding_start)
        trade_pnls.append(cycle_pnl)
        if cycle_pnl > 0.0:
            net_wins += 1
        cycle_net_start = strategy.s["realized_net"]
        cycle_funding_start = funding

    def _quantize_partial_fill(order_dict: dict, *, is_open: bool) -> tuple[dict, float, bool]:
        orig_sz = require_pos_float(order_dict.get("sz"), "sz")
        orig_amt = safe_nonneg_float(order_dict.get("amount", 0.0), default=0.0) if is_open else 0.0
        fill_order, sz, complete = split_order_fill(
            order_dict,
            quantity_key="sz",
            amount_key="amount" if is_open else None,
            ratio=exec_model.partial_fill_ratio,
            force_full=False,
        )
        if not complete:
            q_sz = round(sz, strategy.sz_dec)
            rem_sz = max(0.0, round(orig_sz - q_sz, strategy.sz_dec))
            if q_sz <= 0.0 or q_sz >= orig_sz - dust_eps or rem_sz <= dust_eps:
                q_sz = orig_sz
                complete = True
                order_dict["sz"] = 0.0
                if is_open:
                    order_dict["amount"] = 0.0
                    fill_order["amount"] = orig_amt
                fill_order["sz"] = orig_sz
            else:
                order_dict["sz"] = rem_sz
                fill_order["sz"] = q_sz
                if is_open and orig_sz > 0:
                    fill_amt = orig_amt * (q_sz / orig_sz)
                    fill_order["amount"] = fill_amt
                    order_dict["amount"] = max(0.0, orig_amt - fill_amt)
            sz = q_sz
        return fill_order, sz, complete

    def _bankruptcy_price_if_breached(adverse_px: float) -> float | None:
        held_q = strategy.s["qty"]
        if held_q <= dust_eps:
            return None
        avg_now = strategy._avg()
        if not avg_now or avg_now <= 0.0:
            return None
        fee_rate = getattr(strategy.p, "fee_pct", HL_FEE_PCT) / 100.0
        avail = initial_capital + strategy.s["realized_net"] + funding
        if avail <= 0.0:
            return max(1e-12, adverse_px)
        denom = strategy.sign - fee_rate
        if abs(denom) <= 1e-12:
            return None
        liq_px = ((strategy.sign + fee_rate) * avg_now - (avail / held_q)) / denom
        if strategy.sign > 0:
            if liq_px > 0.0 and adverse_px <= liq_px + 1e-12:
                return max(1e-12, liq_px)
        else:
            if liq_px > 0.0 and adverse_px >= liq_px - 1e-12:
                return liq_px
        return None

    def _record_fill_and_closed_cycle(filled_sz: float, filled_px: float, cycle_before: int) -> None:
        nonlocal fills, turnover_notional
        turnover_notional += filled_sz * filled_px
        fills += 1
        if strategy.s.get("cycles_completed", 0) > cycle_before:
            _record_closed_cycle()

    def _execute_liquidation(liq_px: float, *, append_bar: bool = True) -> None:
        nonlocal liquidated
        held_q = strategy.s["qty"]
        avg_now = strategy._avg() or liq_px
        strategy.s["orders"].clear()
        order_placed_bar.clear()
        cycle_before = strategy.s.get("cycles_completed", 0)
        strategy._apply_close(avg_now, liq_px, held_q)
        # Clamp residual float rounding so ruined account equity is absorbed at 0.0.
        strategy.s["realized_net"] = -(initial_capital + funding)
        _record_fill_and_closed_cycle(held_q, liq_px, cycle_before)
        liquidated = True
        if append_bar:
            equity_curve.append(0.0)
            exposure.append(False)

    for i in range(len(ohlc)):
        o = require_pos_float(ohlc[i][0], "open")
        h = require_pos_float(ohlc[i][1], "high")
        l = require_pos_float(ohlc[i][2], "low")
        c = require_pos_float(ohlc[i][3], "close")
        t_ms = ts[i] if ts is not None and i < len(ts) else None

        if liquidated:
            equity_curve.append(0.0)
            exposure.append(False)
            continue

        held_at_bar_open = strategy.s["qty"] > dust_eps

        # 1. Funding on held perpetuals at bar open notional: shorts receive funding; longs pay it.
        qty = strategy.s["qty"]
        if qty > dust_eps and f_ts:
            f = _bar_funding_rate_sum(t_ms) * qty * o
            funding += (f if strategy.sign < 0 else -f)
            liq_open = _bankruptcy_price_if_breached(o)
            if liq_open is not None:
                _execute_liquidation(liq_open)
                continue

        # 2. Check active limit orders against the bar range
        open_order = next((x for x in list(strategy.s["orders"]) if x["role"] == "open"), None)
        open_eligible = False
        if open_order:
            open_eligible = exec_model.limit_touched(
                open_side, high=h, low=l, limit=open_order["px"]
            )
            # If a held position breaches stop-loss or bankruptcy before reaching a deeper DCA limit, abort DCA.
            if open_eligible and strategy.s["qty"] > dust_eps:
                liq_open_bar = _bankruptcy_price_if_breached(o)
                liq_pre = _bankruptcy_price_if_breached(l if strategy.sign > 0 else h)
                if liq_open_bar is not None or (
                    liq_pre is not None and strategy.sign * (liq_pre - open_order["px"]) >= 0
                ):
                    open_eligible = False
                elif strategy.p.stop_loss_pct > 0:
                    avg_now = strategy._avg()
                    if avg_now:
                        sl_level_pre = avg_now * (1.0 - strategy.sign * strategy.p.stop_loss_pct / 100.0)
                        open_loss_pct = -strategy.sign * (o - avg_now) / avg_now * 100.0
                        worst_px_pre = l if strategy.sign > 0 else h
                        loss_pre = -strategy.sign * (worst_px_pre - avg_now) / avg_now * 100.0
                        if open_loss_pct >= strategy.p.stop_loss_pct - 1e-9 or (
                            loss_pre >= strategy.p.stop_loss_pct - 1e-9
                            and strategy.sign * (sl_level_pre - open_order["px"]) >= 0
                        ):
                            open_eligible = False

        close_order = next((x for x in list(strategy.s["orders"]) if x["role"] == "close"), None)
        close_eligible = False
        if close_order and strategy.s["qty"] > dust_eps:
            close_eligible = exec_model.limit_touched(
                close_side, high=h, low=l, limit=close_order["px"]
            )

        if open_eligible and close_eligible:
            ambiguous_bars += 1

        for side in exec_model.side_order():
            if side == open_side and open_eligible and open_order in strategy.s["orders"]:
                px = min(open_order["px"], o) if strategy.sign > 0 else max(open_order["px"], o)
                fill_order, sz, complete = _quantize_partial_fill(open_order, is_open=True)
                open_kind = fill_order.get("kind")
                if open_kind == "ENTRY":
                    count_dca = False
                else:
                    count_dca = not _parse_bool_flag(open_order.get("dca_counted"), False)
                    open_order["dca_counted"] = True
                if complete:
                    strategy._remove(open_order)
                    order_placed_bar.pop(open_order.get("oid", ""), None)
                strategy._apply_open(
                    sz, px, fill_order.get("amount", 0.0), open_kind, count_dca=count_dca,
                )
                turnover_notional += sz * px
                fills += 1

            if side == close_side and close_eligible and close_order in strategy.s["orders"] and strategy.s["qty"] > dust_eps:
                px = max(close_order["px"], o) if strategy.sign > 0 else min(close_order["px"], o)
                fill_order, sz, complete = _quantize_partial_fill(close_order, is_open=False)
                held_now = strategy.s["qty"]
                if sz >= held_now - dust_eps:
                    sz = held_now
                    complete = True
                if sz <= dust_eps:
                    strategy._remove(close_order)
                    continue
                if complete:
                    strategy._remove(close_order)
                avg = strategy._avg() or px
                cycle_before = strategy.s.get("cycles_completed", 0)
                strategy._apply_close(avg, px, sz)
                _record_fill_and_closed_cycle(sz, px, cycle_before)

        # 3. Intrabar Stop-Loss & Margin Wipeout Check (if held after limit fills)
        if strategy.s["qty"] > dust_eps:
            worst_px = l if strategy.sign > 0 else h
            if strategy.p.stop_loss_pct > 0:
                avg = strategy._avg()
                if avg:
                    loss_pct = -strategy.sign * (worst_px - avg) / avg * 100.0
                    if loss_pct >= strategy.p.stop_loss_pct - 1e-9:
                        sl_level = avg * (1.0 - strategy.sign * strategy.p.stop_loss_pct / 100.0)
                        if held_at_bar_open:
                            raw_exec_px = min(o, sl_level) if strategy.sign > 0 else max(o, sl_level)
                        else:
                            raw_exec_px = sl_level
                        exec_px = exec_model.market_price(close_side, raw_exec_px)
                        liq_sl = _bankruptcy_price_if_breached(exec_px)
                        if liq_sl is not None:
                            _execute_liquidation(liq_sl)
                            continue
                        held_qty = strategy.s["qty"]
                        cycle_before = strategy.s.get("cycles_completed", 0)
                        if strategy._check_stop_loss(exec_px):
                            _record_fill_and_closed_cycle(held_qty, exec_px, cycle_before)

            if strategy.s["qty"] > dust_eps:
                liq_intrabar = _bankruptcy_price_if_breached(worst_px)
                if liq_intrabar is not None:
                    _execute_liquidation(liq_intrabar)
                    continue

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
        signal_val = sig[i] if sig and i < len(sig) else (None if strategy.p.signal_gate else "__any__")
        cycle_before_step = strategy.s.get("cycles_completed", 0)
        held_before_step = strategy.s["qty"]
        strategy.step(c, signal=signal_val)
        if strategy.s.get("cycles_completed", 0) > cycle_before_step:
            if held_before_step > dust_eps:
                turnover_notional += held_before_step * c
            _record_closed_cycle()
            fills += 1

        for o_ord in strategy.s["orders"]:
            oid = o_ord.get("oid", "")
            if o_ord["role"] == "open" and oid not in order_placed_bar:
                order_placed_bar[oid] = i

        # 6. Equity curve and exposure tracking
        qty = strategy.s["qty"]
        cost = strategy.s["cost"]
        realized = strategy.s["realized_net"]
        upnl = (strategy.sign * (c - cost / qty) * qty) if qty > dust_eps else 0.0
        fee_pct_eff = getattr(strategy.p, "fee_pct", HL_FEE_PCT)
        open_fee = (fee_pct_eff / 100.0) * cost if qty > dust_eps else 0.0
        current_equity = initial_capital + realized + funding + upnl - open_fee
        if current_equity <= 0.0:
            if qty > dust_eps:
                _execute_liquidation(c, append_bar=False)
            else:
                strategy.s["orders"].clear()
                strategy.s["realized_net"] = -(initial_capital + funding)
                liquidated = True
            equity_curve.append(0.0)
            exposure.append(False)
            continue
        equity_curve.append(current_equity)
        exposure.append(qty > dust_eps)

    qty = strategy.s["qty"]
    cost = strategy.s["cost"]
    realized_gross = strategy.s["realized_gross"]
    realized_net = strategy.s["realized_net"]
    final_upnl = (strategy.sign * (c - cost / qty) * qty) if qty > dust_eps else 0.0
    cycles = strategy.s.get("cycles_completed", 0)
    wins = net_wins
    total_fees = strategy.s["fees_total"]
    final_open_fee = 0.0
    if qty > dust_eps:
        fee_pct_eff = getattr(strategy.p, "fee_pct", HL_FEE_PCT)
        final_open_fee = (fee_pct_eff / 100.0) * cost
        total_fees += final_open_fee

    return build_replay_summary(
        realized=realized_gross,
        net=realized_net + funding,
        fees=total_fees,
        total=realized_net + funding + final_upnl - final_open_fee,
        final_upnl=final_upnl,
        cycles=cycles,
        wins=wins,
        open_qty=qty,
        fills=fills,
        ambiguous_bars=ambiguous_bars,
        initial_capital=initial_capital,
        account_currency=strat_params.currency,
        equity_curve=equity_curve,
        exposure=exposure,
        trade_pnls=trade_pnls,
        turnover_notional=turnover_notional,
        bar_minutes=valid_bar_minutes,
        periods_per_year=periods_per_year,
        extra_fields={"funding": round_metric(funding)},
    )


def _to_spot_params(params: SpotStratParams | StratParams | dict, direction: str | None) -> SpotStratParams:
    if direction is not None:
        if isinstance(direction, bool) or str(direction).strip().lower() != "long":
            raise ValueError(f"Hyperliquid spot engine supports only 'long' direction, got {direction!r}")
    if isinstance(params, SpotStratParams):
        return params
    if isinstance(params, StratParams):
        if str(params.direction).strip().lower() != "long":
            raise ValueError(f"Hyperliquid spot engine supports only 'long' direction, got {params.direction!r}")
        return SpotStratParams(
            currency=params.currency,
            entry_amount=params.entry_amount,
            entry_discount_pct=params.entry_discount_pct,
            dca_amount=params.dca_amount,
            dca_drop_pct=params.dca_drop_pct,
            check_minutes=params.check_minutes,
            takeprofit_pct=params.takeprofit_pct,
            max_budget=params.max_budget,
            max_dca_buys=params.max_dca_buys,
            enable_takeprofit=params.enable_takeprofit,
            order_ttl_min=params.order_ttl_min,
            stop_loss_pct=params.stop_loss_pct,
            reentry_tolerance_pct=params.reentry_tolerance_pct,
            sl_mhdsl_enabled=False,
        )
    if isinstance(params, dict):
        if "direction" in params and params["direction"] is not None:
            if isinstance(params["direction"], bool) or str(params["direction"]).strip().lower() != "long":
                raise ValueError(f"Hyperliquid spot engine supports only 'long' direction, got {params['direction']!r}")
        hl_defaults = StratParams.from_dict(params, direction="long")
        spot_defaults = {
            "currency": hl_defaults.currency,
            "entry_amount": hl_defaults.entry_amount,
            "entry_discount_pct": hl_defaults.entry_discount_pct,
            "dca_amount": hl_defaults.dca_amount,
            "dca_drop_pct": hl_defaults.dca_drop_pct,
            "check_minutes": hl_defaults.check_minutes,
            "takeprofit_pct": hl_defaults.takeprofit_pct,
            "max_budget": hl_defaults.max_budget,
            "max_dca_buys": hl_defaults.max_dca_buys,
            "enable_takeprofit": hl_defaults.enable_takeprofit,
            "order_ttl_min": hl_defaults.order_ttl_min,
            "stop_loss_pct": hl_defaults.stop_loss_pct,
            "reentry_tolerance_pct": hl_defaults.reentry_tolerance_pct,
            "sl_mhdsl_enabled": False,
        }
        return SpotStratParams.from_dict(params, defaults=spot_defaults)
    raise ValueError(f"Unsupported spot params type: {type(params).__name__}")


def make_hyperliquid_replay_adapter(
    params: StratParams | SpotStratParams | dict,
    direction: str | None = None,
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
        spot_params = _to_spot_params(params, direction)
        if bar_minutes is not None:
            bm = require_pos_float(bar_minutes, "bar_minutes")
            spot_params = scale_subbar_timeout_params(
                align_spot_interval_params(spot_params, bm), bm
            )
        if fee_pct is not None:
            eff_fee = require_nonneg_float(fee_pct, "fee_pct")
        elif isinstance(params, StratParams):
            eff_fee = params.fee_pct
        elif isinstance(params, dict) and (params.get("fee") is not None or params.get("fee_pct") is not None):
            raw_f = params.get("fee") if params.get("fee") is not None else params.get("fee_pct")
            eff_fee = require_nonneg_float(raw_f, "fee")
        else:
            eff_fee = HL_FEE_PCT
        if eff_fee >= 100.0:
            raise ValueError(f"fee_pct must be < 100, got {eff_fee!r}")
        return make_spot_replay_adapter(
            params=spot_params,
            fee_pct=eff_fee,
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
