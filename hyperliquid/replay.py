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
    ReplaySessionBase,
    build_replay_summary,
    extract_adapter_context_timestamps,
    make_spot_replay_adapter,
    prepare_spot_interval_params,
)
from strategy import Strategy, StratParams, HL_FEE_PCT  # noqa: E402
from providers.ohlc_utils import validate_ohlc_bars
from strategies.replay_execution import (
    choose_intrabar_scenario,
    clamp_sell_fill_to_held,
    ExecutionModel,
    split_order_fill,
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


def funding_at(fund: list[tuple[int, float]] | None, t_ms: int) -> float:
    """Return the funding rate active at t_ms, using the most recent rate at or before t.

    Automatically normalises Unix-second timestamps (< 1e10) to milliseconds so
    both direct replay callers (ms) and walk-forward evaluation contexts (s) match.
    """
    fund_ts, fund_rates = _normalize_funding_series(fund)
    if not fund_ts:
        return 0.0
    t_val = _to_funding_ms(t_ms)
    return 0.0 if t_val is None else _lookup_sorted_funding_ms(fund_ts, fund_rates, t_val)


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


class _HyperliquidReplaySession(ReplaySessionBase):
    """Concrete replay session (Level 3) executing the live Hyperliquid perpetual strategy."""

    def __init__(
        self,
        ohlc: list[tuple[float, float, float, float]],
        *,
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
    ) -> None:
        exec_model = execution or ExecutionModel()
        strat_params = _resolve_hl_strat_params(params, direction, fee_pct)
        initial_capital = require_pos_float(strat_params.max_budget, "max_budget")
        self.strategy = Strategy(
            client=None,
            coin=coin,
            params=strat_params,
            dry_run=True,
            replay_mode=True,
        )
        validate_ohlc_bars(ohlc, check_envelope=True)
        self.ts = ts
        self.sig = sig
        self.f_ts = fund_ts or []
        self.f_rates = fund_rates or []
        self.funding = 0.0
        self.liquidated = False
        self.last_close = require_pos_float(ohlc[0][3], "close")
        self.open_side = "buy" if self.strategy.sign > 0 else "sell"
        self.close_side = "sell" if self.strategy.sign > 0 else "buy"
        self.order_placed_bar: dict[str, int] = {}
        valid_bar_minutes = safe_pos_float(bar_minutes, default=None)
        self.eff_bar_minutes = valid_bar_minutes if valid_bar_minutes is not None else 60.0
        lot_step = 10.0 ** (-self.strategy.sz_dec)
        self.dust_eps = max(1e-12, 0.5 * lot_step)
        super().__init__(
            ohlc,
            initial_capital=initial_capital,
            account_currency=strat_params.currency,
            execution=exec_model,
            bar_minutes=valid_bar_minutes,
            periods_per_year=periods_per_year,
        )

    def _quantize_partial_fill(self, order_dict: dict, *, is_open: bool) -> tuple[dict, float, bool]:
        orig_sz = require_pos_float(order_dict.get("sz"), "sz")
        orig_amt = safe_nonneg_float(order_dict.get("amount", 0.0), default=0.0) if is_open else 0.0
        fill_order, sz, complete = split_order_fill(
            order_dict,
            quantity_key="sz",
            amount_key="amount" if is_open else None,
            ratio=self.execution.partial_fill_ratio,
            force_full=False,
        )
        if not complete:
            q_sz = round(sz, self.strategy.sz_dec)
            rem_sz = max(0.0, round(orig_sz - q_sz, self.strategy.sz_dec))
            if q_sz <= 0.0 or q_sz >= orig_sz - self.dust_eps or rem_sz <= self.dust_eps:
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

    def _bankruptcy_price_if_breached(self, adverse_px: float) -> float | None:
        held_q = self.strategy.s["qty"]
        if held_q <= self.dust_eps:
            return None
        avg_now = self.strategy._avg()
        if not avg_now or avg_now <= 0.0:
            return None
        fee_rate = getattr(self.strategy.p, "fee_pct", HL_FEE_PCT) / 100.0
        avail = self.initial_capital + self.strategy.s["realized_net"] + self.funding
        if avail <= 0.0:
            return max(1e-12, adverse_px)
        denom = self.strategy.sign - fee_rate
        if abs(denom) <= 1e-12:
            return None
        liq_px = ((self.strategy.sign + fee_rate) * avg_now - (avail / held_q)) / denom
        if self.strategy.sign > 0:
            if liq_px > 0.0 and adverse_px <= liq_px + 1e-12:
                return max(1e-12, liq_px)
        elif liq_px > 0.0 and adverse_px >= liq_px - 1e-12:
            return liq_px
        return None

    def _record_fill_and_closed_cycle(self, filled_sz: float, filled_px: float, cycle_before: int) -> None:
        self.turnover_notional += filled_sz * filled_px
        self.fills += 1
        if self.strategy.s.get("cycles_completed", 0) > cycle_before:
            self._record_closed_cycle(self.strategy.s["realized_net"] + self.funding)

    def _execute_liquidation(self, liq_px: float) -> None:
        held_q = self.strategy.s["qty"]
        avg_now = self.strategy._avg() or liq_px
        self.strategy.s["orders"].clear()
        self.order_placed_bar.clear()
        cycle_before = self.strategy.s.get("cycles_completed", 0)
        self.strategy._apply_close(avg_now, liq_px, held_q)
        # Clamp residual float rounding so ruined account equity is absorbed at 0.0.
        self.strategy.s["realized_net"] = -(self.initial_capital + self.funding)
        self._record_fill_and_closed_cycle(held_q, liq_px, cycle_before)
        self.liquidated = True
        self._record_bar_equity(0.0, False)

    def _compute_bar_funding_rate(self, t_ms: int | None) -> float:
        t_val = _to_funding_ms(t_ms)
        if t_val is None:
            return 0.0
        if self.eff_bar_minutes <= 60.0 or t_val < 0:
            return _lookup_sorted_funding_ms(self.f_ts, self.f_rates, t_val) * (self.eff_bar_minutes / 60.0)
        rem_hours = self.eff_bar_minutes / 60.0
        bar_f_rate = 0.0
        hour_idx = 0
        while rem_hours > 1e-9:
            weight = min(1.0, rem_hours)
            bar_f_rate += _lookup_sorted_funding_ms(self.f_ts, self.f_rates, t_val + hour_idx * 3_600_000.0) * weight
            rem_hours -= weight
            hour_idx += 1
        return bar_f_rate

    def _evaluate_open_order_eligible(self, open_order: dict | None, o: float, h: float, l: float) -> bool:
        if not open_order or not self.execution.limit_touched(self.open_side, high=h, low=l, limit=open_order["px"]):
            return False
        if self.strategy.s["qty"] <= self.dust_eps:
            return True
        liq_open_bar = self._bankruptcy_price_if_breached(o)
        liq_pre = self._bankruptcy_price_if_breached(l if self.strategy.sign > 0 else h)
        if liq_open_bar is not None or (
            liq_pre is not None and self.strategy.sign * (liq_pre - open_order["px"]) >= 0
        ):
            return False
        if self.strategy.p.stop_loss_pct > 0 and (avg_now := self.strategy._avg()):
            sl_level_pre = avg_now * (1.0 - self.strategy.sign * self.strategy.p.stop_loss_pct / 100.0)
            open_loss_pct = -self.strategy.sign * (o - avg_now) / avg_now * 100.0
            worst_px_pre = l if self.strategy.sign > 0 else h
            loss_pre = -self.strategy.sign * (worst_px_pre - avg_now) / avg_now * 100.0
            if open_loss_pct >= self.strategy.p.stop_loss_pct - 1e-9 or (
                loss_pre >= self.strategy.p.stop_loss_pct - 1e-9
                and self.strategy.sign * (sl_level_pre - open_order["px"]) >= 0
            ):
                return False
        return True

    def _settle_intrabar_limit_fills(
        self,
        open_order: dict | None,
        close_order: dict | None,
        *,
        open_eligible: bool,
        close_eligible: bool,
        o: float,
    ) -> None:
        for side in self.execution.side_order():
            if side == self.open_side and open_eligible and open_order in self.strategy.s["orders"]:
                px = min(open_order["px"], o) if self.strategy.sign > 0 else max(open_order["px"], o)
                fill_order, sz, complete = self._quantize_partial_fill(open_order, is_open=True)
                open_kind = fill_order.get("kind")
                if open_kind == "ENTRY":
                    count_dca = False
                else:
                    count_dca = not _parse_bool_flag(open_order.get("dca_counted"), False)
                    open_order["dca_counted"] = True
                if complete:
                    self.strategy._remove(open_order)
                    self.order_placed_bar.pop(open_order.get("oid", ""), None)
                self.strategy._apply_open(
                    sz, px, fill_order.get("amount", 0.0), open_kind, count_dca=count_dca,
                )
                self.turnover_notional += sz * px
                self.fills += 1

            if (
                side == self.close_side
                and close_eligible
                and close_order in self.strategy.s["orders"]
                and self.strategy.s["qty"] > self.dust_eps
            ):
                px = max(close_order["px"], o) if self.strategy.sign > 0 else min(close_order["px"], o)
                _, sz, complete = self._quantize_partial_fill(close_order, is_open=False)
                sz, complete = clamp_sell_fill_to_held(sz, self.strategy.s["qty"], complete, eps=self.dust_eps)
                if sz <= 0.0:
                    self.strategy._remove(close_order)
                    continue
                if complete:
                    self.strategy._remove(close_order)
                avg = self.strategy._avg() or px
                cycle_before = self.strategy.s.get("cycles_completed", 0)
                self.strategy._apply_close(avg, px, sz)
                self._record_fill_and_closed_cycle(sz, px, cycle_before)

    def _check_intrabar_stop_and_liquidation(self, o: float, h: float, l: float, *, held_at_bar_open: bool) -> bool:
        """Return True when the account liquidated inside the bar."""
        if self.strategy.s["qty"] <= self.dust_eps:
            return False
        worst_px = l if self.strategy.sign > 0 else h
        if self.strategy.p.stop_loss_pct > 0 and (avg := self.strategy._avg()):
            loss_pct = -self.strategy.sign * (worst_px - avg) / avg * 100.0
            if loss_pct >= self.strategy.p.stop_loss_pct - 1e-9:
                sl_level = avg * (1.0 - self.strategy.sign * self.strategy.p.stop_loss_pct / 100.0)
                raw_exec_px = (min(o, sl_level) if self.strategy.sign > 0 else max(o, sl_level)) if held_at_bar_open else sl_level
                exec_px = self.execution.market_price(self.close_side, raw_exec_px)
                liq_sl = self._bankruptcy_price_if_breached(exec_px)
                if liq_sl is not None:
                    self._execute_liquidation(liq_sl)
                    return True
                held_qty = self.strategy.s["qty"]
                cycle_before = self.strategy.s.get("cycles_completed", 0)
                if self.strategy._check_stop_loss(exec_px):
                    self._record_fill_and_closed_cycle(held_qty, exec_px, cycle_before)
        if self.strategy.s["qty"] > self.dust_eps:
            liq_intrabar = self._bankruptcy_price_if_breached(worst_px)
            if liq_intrabar is not None:
                self._execute_liquidation(liq_intrabar)
                return True
        return False

    def _step_bar(self, bar_index: int, o: float, h: float, l: float, c: float) -> None:
        self.last_close = c
        t_ms = self.ts[bar_index] if self.ts is not None and bar_index < len(self.ts) else None
        if self.liquidated:
            self._record_bar_equity(0.0, False)
            return
        held_at_bar_open = self.strategy.s["qty"] > self.dust_eps

        # 1. Funding on held perpetuals at bar open notional: shorts receive funding; longs pay it.
        qty = self.strategy.s["qty"]
        if qty > self.dust_eps and self.f_ts:
            f = self._compute_bar_funding_rate(t_ms) * qty * o
            self.funding += (f if self.strategy.sign < 0 else -f)
            liq_open = self._bankruptcy_price_if_breached(o)
            if liq_open is not None:
                self._execute_liquidation(liq_open)
                return

        # 2. Check active limit orders against the bar range
        open_order = next((x for x in list(self.strategy.s["orders"]) if x["role"] == "open"), None)
        open_eligible = self._evaluate_open_order_eligible(open_order, o, h, l)
        close_order = next((x for x in list(self.strategy.s["orders"]) if x["role"] == "close"), None)
        close_eligible = bool(
            close_order
            and self.strategy.s["qty"] > self.dust_eps
            and self.execution.limit_touched(self.close_side, high=h, low=l, limit=close_order["px"])
        )
        if open_eligible and close_eligible:
            self.ambiguous_bars += 1
        self._settle_intrabar_limit_fills(
            open_order, close_order, open_eligible=open_eligible, close_eligible=close_eligible, o=o
        )

        # 3. Intrabar Stop-Loss & Margin Wipeout Check (if held after limit fills)
        if self._check_intrabar_stop_and_liquidation(o, h, l, held_at_bar_open=held_at_bar_open):
            return

        # 4. TTL / Runaway price cancellation
        open_order = next((x for x in list(self.strategy.s["orders"]) if x["role"] == "open"), None)
        if open_order:
            oid = open_order.get("oid", "")
            placed_b = self.order_placed_bar.setdefault(oid, bar_index)
            elapsed_min = (bar_index - placed_b) * self.eff_bar_minutes
            if self.strategy._open_is_stale(open_order, c, elapsed_min=elapsed_min):
                self.strategy._remove(open_order)
                self.order_placed_bar.pop(oid, None)

        # 5. Strategy decisions at bar close
        signal_val = self.sig[bar_index] if self.sig and bar_index < len(self.sig) else (
            None if self.strategy.p.signal_gate else "__any__"
        )
        cycle_before_step = self.strategy.s.get("cycles_completed", 0)
        held_before_step = self.strategy.s["qty"]
        self.strategy.step(c, signal=signal_val)
        if self.strategy.s.get("cycles_completed", 0) > cycle_before_step:
            if held_before_step > self.dust_eps:
                self.turnover_notional += held_before_step * c
            self._record_closed_cycle(self.strategy.s["realized_net"] + self.funding)
            self.fills += 1

        for o_ord in self.strategy.s["orders"]:
            oid = o_ord.get("oid", "")
            if o_ord["role"] == "open" and oid not in self.order_placed_bar:
                self.order_placed_bar[oid] = bar_index

        # 6. Equity curve and exposure tracking
        qty = self.strategy.s["qty"]
        cost = self.strategy.s["cost"]
        realized = self.strategy.s["realized_net"]
        upnl = (self.strategy.sign * (c - cost / qty) * qty) if qty > self.dust_eps else 0.0
        fee_pct_eff = getattr(self.strategy.p, "fee_pct", HL_FEE_PCT)
        open_fee = (fee_pct_eff / 100.0) * cost if qty > self.dust_eps else 0.0
        current_equity = self.initial_capital + realized + self.funding + upnl - open_fee
        if current_equity <= 0.0:
            if qty > self.dust_eps:
                self._execute_liquidation(c)
                return
            self.strategy.s["orders"].clear()
            self.strategy.s["realized_net"] = -(self.initial_capital + self.funding)
            self.liquidated = True
            current_equity = 0.0
        self._record_bar_equity(current_equity, qty > self.dust_eps)

    def _build_summary(self) -> dict:
        qty = self.strategy.s["qty"]
        cost = self.strategy.s["cost"]
        realized_gross = self.strategy.s["realized_gross"]
        realized_net = self.strategy.s["realized_net"]
        final_upnl = (self.strategy.sign * (self.last_close - cost / qty) * qty) if qty > self.dust_eps else 0.0
        total_fees = self.strategy.s["fees_total"]
        final_open_fee = 0.0
        if qty > self.dust_eps:
            fee_pct_eff = getattr(self.strategy.p, "fee_pct", HL_FEE_PCT)
            final_open_fee = (fee_pct_eff / 100.0) * cost
            total_fees += final_open_fee
        return self._build_summary_from_state(
            realized=realized_gross,
            net=realized_net + self.funding,
            fees=total_fees,
            total=realized_net + self.funding + final_upnl - final_open_fee,
            final_upnl=final_upnl,
            cycles=self.strategy.s.get("cycles_completed", 0),
            open_qty=qty,
            extra_fields={"funding": round_metric(self.funding)},
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
    return _HyperliquidReplaySession(
        ohlc,
        ts=ts,
        fund_ts=fund_ts,
        fund_rates=fund_rates,
        sig=sig,
        params=params,
        direction=direction,
        coin=coin,
        bar_minutes=bar_minutes,
        periods_per_year=periods_per_year,
        execution=execution,
        fee_pct=fee_pct,
    ).run()


_HL_TO_SPOT_PARAM_FIELDS = (
    "currency",
    "entry_amount",
    "entry_discount_pct",
    "dca_amount",
    "dca_drop_pct",
    "check_minutes",
    "takeprofit_pct",
    "max_budget",
    "max_dca_buys",
    "enable_takeprofit",
    "order_ttl_min",
    "stop_loss_pct",
    "reentry_tolerance_pct",
)


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
        def _require_spot_long(d: object) -> None:
            if d is not None and (isinstance(d, bool) or str(d).strip().lower() != "long"):
                raise ValueError(f"Hyperliquid spot engine supports only 'long' direction, got {d!r}")

        _require_spot_long(direction)
        if isinstance(params, SpotStratParams):
            spot_params = params
        elif isinstance(params, StratParams):
            _require_spot_long(params.direction)
            spot_params = SpotStratParams(
                **{k: getattr(params, k) for k in _HL_TO_SPOT_PARAM_FIELDS},
                sl_mhdsl_enabled=False,
            )
        elif isinstance(params, dict):
            _require_spot_long(params.get("direction"))
            hl_defaults = StratParams.from_dict(params, direction="long")
            spot_params = SpotStratParams.from_dict(
                params,
                defaults={
                    **{k: getattr(hl_defaults, k) for k in _HL_TO_SPOT_PARAM_FIELDS},
                    "sl_mhdsl_enabled": False,
                },
            )
        else:
            raise ValueError(f"Unsupported spot params type: {type(params).__name__}")
        if bar_minutes is not None:
            bm = require_pos_float(bar_minutes, "bar_minutes")
            spot_params = prepare_spot_interval_params(spot_params, bm)
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
        ts, _ = extract_adapter_context_timestamps(context, empty=None)
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
