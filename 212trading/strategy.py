#!/usr/bin/env python3
"""Instrument- and currency-generic DCA plus take-profit engine.

Average-TP mode buys a discounted entry, adds DCA after configured drops, and closes
the position at its average-cost profit target. DCA-only mode accumulates without sale.
Dry-run is paper-only, per-cycle budget and buy-count caps bound exposure, and per-ticker
state survives restarts. T212 FX conversion fees apply on both BUY and SELL legs.
"""

from __future__ import annotations

import math
import os
import re
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_HERE)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
from botcore import (  # noqa: E402
    _parse_bool_flag,
    are_close,
    clean_order_id,
    ensure_local_module_priority,
    first_pos_float,
    is_market_order,
    log as _log,
    normalize_epoch_sec,
    require_pos_float,
    required_bool_env,
    required_env,
    required_float_env,
    safe_float,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
)
ensure_local_module_priority(_HERE, "ipo_common", "market_data", "t212_client")

from typing import Any
from cachepaths import canonical_base_asset
from intelligence.internal.state.volatility import vol_pct_to_ratio
from ipo_common import notify
from market_data import get_eur_usd, get_usd_ron, get_price_usd, trend_slope_pct
from t212_client import T212Client
from providers.base import (
    extract_order_id,
    extract_order_price,
)
from providers.order_ohlc_utils import (
    extract_order_side,
    extract_orig_qty,
)
from providers.execution_audit import AuditedStrategyExecutor, ExecutionAudit, new_intent_id
from providers.strategy_executor import OrderNotFound, ProviderError
from providers.t212_provider import T212Provider, _order_ticker, _t212_tickers_match
from strategies.base_spot_mixin import (
    BaseSpotMixin,
    SpotParamsMixin,
    adapt_ladder_rungs,
    generate_tp_ladder,
    load_common_core_strat_env,
    load_common_mhdpa_env,
    parse_max_dca_buys_env,
    parse_tp_ladder_env,
    sanitize_common_spot_state,
)
from strategies.state_store import (
    state_path_for as _shared_state_path_for,
)
from strategies.position_exit_manager import (
    PositionExitManager, ExitAction,
)
from order_lifecycle import restore_cancelled_orders

FX_FEE_PCT = 0.15  # T212 currency-conversion fee per direction.


def log(msg: str) -> None:
    """Log with the asset's thread name: t212_bot runs one thread per asset into one log."""
    name = threading.current_thread().name
    _log(msg if name == "MainThread" else f"[{name}] {msg}")


def state_path_for(ticker: str) -> str:
    return _shared_state_path_for(ticker, _HERE, venue="t212")


@dataclass
class StratParams(SpotParamsMixin):
    currency: str
    entry_amount: float
    entry_discount_pct: float
    dca_amount: float
    dca_drop_pct: float
    check_minutes: float
    takeprofit_pct: float
    max_budget: float
    max_dca_buys: int
    validity: str
    enable_takeprofit: bool
    order_ttl_min: float
    stop_loss_pct: float
    yahoo_sym: str = ""
    reentry_drop_pct: float = 0.0
    reentry_tolerance_pct: float = 0.0
    tp_ladder: list = field(default_factory=list)
    fx_fee_pct: float = FX_FEE_PCT
    loss_alert_step: float = 1.0
    ladder_min_free: float = 6.0
    sl_rebuy_enabled: bool = False
    sl_rebuy_bounce_pct: float = 1.2
    dca_trend_gate_pct: float = 0.0
    trail_pct: float = 0.0
    trail_min_profit_pct: float = 5.0
    # --- Hybrid Re-Entry ---
    reentry_hybrid_enabled: bool = False
    reentry_peak_relative: bool = False
    reentry_pullback_pct: float = 1.5
    reentry_pullback_adaptive: bool = False
    reentry_pullback_k: float = 1.0
    reentry_pullback_min: float = 0.8
    reentry_pullback_max: float = 4.0
    reentry_ttl_hours: float = 0.0
    # --- Multi-Horizon Dynamic Stop-Loss (MHDSL) ---
    sl_mhdsl_enabled: bool = False
    sl_disaster_buffer_ratio: float = 0.30
    sl_disaster_buffer_max_pct: float = 15.0
    sl_trend_defer_max_sec: float = 220.0
    sl_bounce_min_recovery_pct: float = 0.40
    sl_bounce_pullback_pct: float = 0.30
    sl_structural_break_enabled: bool = True
    # --- Multi-Horizon Dynamic Take-Profit (MDHTP) ---
    tp_mdhtp_enabled: bool = False
    tp_stagnation_max_sec: float = 1080.0

    _DEFAULT_DICT_FIELDS = {
        "currency": "USD", "entry_amount": 50.0, "entry_discount_pct": 0.2,
        "dca_amount": 30.0, "dca_drop_pct": 2.0, "check_minutes": 5.0,
        "takeprofit_pct": 5.0, "max_budget": 500.0, "max_dca_buys": 10,
        "validity": "DAY", "enable_takeprofit": True, "order_ttl_min": 10.0,
        "stop_loss_pct": 10.0,
    }

    @classmethod
    def from_env(cls, env: dict | None = None) -> "StratParams":
        """Read the environment or an explicit mapping for collision-free multi-asset use."""
        b, f = required_bool_env, required_float_env
        ladder = parse_tp_ladder_env(env, generate_tp_ladder)
        core_cfg = load_common_core_strat_env(env, include_entry_dca=False)
        budget = core_cfg["max_budget"]
        entry_amt = budget * f("STRAT_ENTRY_PCT", env) / 100.0
        dca_amt = budget * f("STRAT_DCA_PCT", env) / 100.0
        max_dca, entry_amt = parse_max_dca_buys_env(env, budget, entry_amt, dca_amt)
        return cls(
            **core_cfg,
            entry_amount=entry_amt,
            dca_amount=dca_amt,
            max_dca_buys=max_dca,
            validity="GOOD_TILL_CANCEL",
            yahoo_sym=required_env("YAHOO_SYMBOL", env),
            reentry_drop_pct=f("STRAT_REENTRY_DROP_PCT", env),
            tp_ladder=ladder,
            fx_fee_pct=f("STRAT_FX_FEE_PCT", env),
            loss_alert_step=f("STRAT_LOSS_ALERT_STEP", env),
            ladder_min_free=f("STRAT_LADDER_MIN_FREE", env),
            sl_rebuy_enabled=b("STRAT_SL_REBUY_ENABLED", env),
            sl_rebuy_bounce_pct=f("STRAT_SL_REBUY_BOUNCE_PCT", env),
            dca_trend_gate_pct=f("STRAT_DCA_TREND_GATE_PCT", env),
            trail_pct=f("STRAT_TRAIL_PCT", env),
            trail_min_profit_pct=f("STRAT_TRAIL_MIN_PROFIT_PCT", env),
            **load_common_mhdpa_env(env, include_spot_sl=False),
        )


def _new_state() -> dict:
    return {
        "cycle": 1, "qty": 0.0, "cost_usd": 0.0, "spent_cash": 0.0, "dca_buys": 0,
        "entry_price": None, "last_buy_price": None,
        "last_sell_price": None, "last_sell_ts": None,
        "post_sell_peak": None, "post_sell_trough": None, "last_exit_kind": None,
        "sl_bounce_state": {}, "sl_episode_state": {}, "tp_ride_state": {}, "reentry_state": {},
        "realized_pnl_usd": 0.0, "realized_net_usd": 0.0, "fees_usd": 0.0,
        "loss_band": 0, "tp_sold_levels": [], "orders": [], "pending_submit": None,
        "pos_peak": 0.0, "tr_alerted": False, "tr_armed": False,
        "tp_pending_exit": None, "sl_pending_exit": None,
    }


def _sanitize_t212_state(st: dict) -> dict:
    """Ensure persisted or injected state has valid types for all structural and numeric fields in-place."""
    if not isinstance(st, dict):
        return _new_state()
    st = sanitize_common_spot_state(
        st,
        _new_state,
        nonneg_float_keys=("qty", "cost_usd", "spent_cash", "fees_usd", "pos_peak"),
        signed_float_keys=("realized_pnl_usd", "realized_net_usd"),
        nonneg_int_keys=("dca_buys", "loss_band"),
        pos_float_keys=(
            "entry_price", "last_buy_price", "last_sell_price",
            "last_sell_ts", "post_sell_peak", "post_sell_trough",
        ),
        bool_keys=("tr_alerted", "tr_armed", "sl_alerted", "sl_pending"),
        dict_keys=("sl_bounce_state", "sl_episode_state", "tp_ride_state", "reentry_state"),
        nullable_dict_keys=("pending_submit", "tp_pending_exit", "sl_pending_exit"),
        pos_float_list_keys=("tp_sold_levels",),
        epoch_keys=("buy_backoff_until", "locked_zero_until", "tr_fail_until", "sl_fail_until"),
        epoch_dict_keys=("tp_fail_until",),
        sanitize_orders=False,
    )
    if isinstance(st.get("pending_submit"), dict):
        ps = st["pending_submit"]
        ps_side = str(ps.get("side") or "").strip().upper()
        ps_market = bool(_parse_bool_flag(ps.get("market"), False))
        ps_qty = safe_pos_float(ps.get("qty"), None)
        ps_limit = (
            safe_nonneg_float(ps.get("limit"), 0.0 if ps.get("limit") is None else None)
            if ps_market
            else safe_pos_float(ps.get("limit"), None)
        )
        ps_intent = clean_order_id(ps.get("intent_id"))
        if ps_side not in ("BUY", "SELL") or ps_qty is None or ps_limit is None or ps_intent is None:
            st["pending_submit"] = None
        else:
            ps_copy = dict(ps)
            ps_copy.update(
                side=ps_side, qty=ps_qty, limit=ps_limit, intent_id=ps_intent,
                market=ps_market, level=safe_pos_float(ps.get("level"), None),
            )
            if "ts" in ps_copy and ps_copy.get("ts") is not None:
                ps_copy["ts"] = normalize_epoch_sec(ps_copy.get("ts"), default=0.0, allow_zero=True)
            if "portfolio_fill_observed" in ps_copy:
                ps_copy["portfolio_fill_observed"] = _parse_bool_flag(ps_copy.get("portfolio_fill_observed"), False)
            st["pending_submit"] = ps_copy
    raw_orders = st.get("orders")
    if isinstance(raw_orders, list):
        clean_orders = []
        for o in raw_orders:
            if not isinstance(o, dict):
                continue
            clean_oid = clean_order_id(o.get("id"))
            o_side = str(o.get("side") or "").strip().upper()
            if clean_oid is None or o_side not in ("BUY", "SELL"):
                continue
            o_qty = safe_pos_float(o.get("qty"), None)
            o_market = _parse_bool_flag(o.get("market"), False)
            o_limit = safe_nonneg_float(o.get("limit"), -1.0) if o_market else safe_pos_float(o.get("limit"), None)
            if o_qty is None or o_limit is None or o_limit < 0.0:
                continue
            o_copy = dict(o)
            o_copy.update(
                id=clean_oid, side=o_side, qty=o_qty, limit=o_limit,
                market=o_market, level=safe_pos_float(o.get("level"), None),
            )
            if "amount" in o_copy and o_copy.get("amount") is not None:
                o_copy["amount"] = safe_pos_float(o_copy.get("amount"), None)
            if "ts" in o_copy and o_copy.get("ts") is not None:
                o_copy["ts"] = normalize_epoch_sec(o_copy.get("ts"), default=0.0, allow_zero=True)
            if "intent_id" in o_copy:
                o_copy["intent_id"] = clean_order_id(o.get("intent_id"))
            for bk in ("cancel_requested", "dca_counted"):
                if bk in o_copy:
                    o_copy[bk] = _parse_bool_flag(o_copy.get(bk), False)
            clean_orders.append(o_copy)
        st["orders"] = clean_orders
    else:
        st["orders"] = []
    return st


class Strategy(BaseSpotMixin):
    venue_label = "T212"
    _cost_state_key = "cost_usd"
    _spent_state_key = "spent_cash"
    _pending_submit_state_key = "pending_submit"
    _qty_eps = 1e-9
    _vol_default_source = "none"
    _tp_floor_param_attr = "trail_min_profit_pct"
    _supports_fast_profit_guard = False
    _track_stop_sl_low = False
    _include_shadow_in_snapshot = False
    _use_shadow_tick_time_default = False
    _exit_config_ratio_fn = staticmethod(vol_pct_to_ratio)
    _raw_reentry_shadow_vol_pct = BaseSpotMixin._resolve_shadow_vol_1h_pct
    _sanitize_state_dict = staticmethod(_sanitize_t212_state)

    def __init__(self, client: T212Client, ticker: str, params: StratParams,
                 dry_run: bool = True, desktop: bool = False,
                 initial_state: dict | None = None,
                 fx_to_usd: float | None = None,
                 clock: Callable[[], float] | None = None,
                 trend_slope_provider: Callable[[str], float | None] | None = None,
                 execution_audit: ExecutionAudit | None = None,
                 replay_mode: bool = False,
                 observe_only: bool = False):
        self.client = client
        self.ticker = ticker
        self.yahoo_sym = params.yahoo_sym or canonical_base_asset(ticker)
        self._clock = clock or time.time
        self._trend_slope_provider = trend_slope_provider or trend_slope_pct
        self.execution_audit = execution_audit or ExecutionAudit()
        dry_run_b = _parse_bool_flag(dry_run, True)
        replay_b = _parse_bool_flag(replay_mode, False) or (clock is not None)
        self.executor = AuditedStrategyExecutor(
            T212Provider(client=client, live_enabled=not dry_run_b, order_validity=params.validity),
            audit=self.execution_audit, venue="T212",
        )
        self.fx_to_usd = (
            self._fx_to_usd(params.currency) if fx_to_usd is None else (safe_pos_float(fx_to_usd) or 1.0)
        )
        self._init_common_spot_runtime_state(
            params,
            state_file=state_path_for(ticker),
            initial_state=initial_state,
            dry_run=dry_run_b,
            desktop=desktop,
            replay_mode=replay_b,
            observe_only=observe_only,
            require_dict_initial_state=False,
        )

    # -- Currency & Helpers ----------------------------------------------------
    def _fx_to_usd(self, currency: str) -> float:
        """How many USD in one unit of the given currency -- generic for any Yahoo currency."""
        ccy = str(currency or "USD").strip().upper()
        if ccy == "USD":
            return 1.0
        if ccy == "EUR":
            return safe_pos_float(get_eur_usd(), default=1.08)
        if ccy == "RON":
            return 1.0 / safe_pos_float(get_usd_ron(), default=4.65)
        rate = safe_pos_float(get_price_usd(f"{ccy}USD=X"), None)
        if rate:
            return rate
        log(f"  ! {ccy}/USD rate unavailable - treating amounts as USD (1:1). Check STRAT_CURRENCY!")
        return 1.0

    def _new_pending_submit(self, *, side: str, qty: float, limit: float,
                            amount: float | None, kind: str,
                            level: float | None, market: bool) -> dict:
        return {
            "intent_id": new_intent_id("T212", self.ticker, kind),
            "side": str(side).upper(),
            "qty": require_pos_float(qty, "qty"),
            "limit": round(require_pos_float(limit, "limit"), 2),
            "amount": amount,
            "kind": kind,
            "level": level,
            "market": _parse_bool_flag(market, False),
            "ts": self._now(),
            "before_qty": safe_nonneg_float(self.s.get("qty"), default=0.0) or 0.0,
            "lookup_misses": 0,
        }

    def _persist_pending_submit(self, pending: dict | None) -> None:
        self._persist_pending_state_slot("pending_submit", pending)

    def _adopt_pending_submit(self, order_id: str) -> dict:
        pending = dict(self.s.get("pending_submit") or {})
        if not pending:
            raise RuntimeError("there is no pending T212 intent to adopt")
        order = {
            "id": str(order_id),
            "side": pending["side"],
            "qty": safe_float(pending.get("qty"), default=0.0) or 0.0,
            "limit": safe_float(pending.get("limit"), default=0.0) or 0.0,
            "kind": pending.get("kind"),
            "level": pending.get("level"),
            "intent_id": pending.get("intent_id"),
            "market": is_market_order(pending),
            "ts": safe_float(pending.get("ts"), default=None) or self._now(),
        }
        if pending.get("amount") is not None:
            order["amount"] = pending["amount"]
        self.s["orders"].append(order)
        self._persist_pending_submit(None)
        return order

    def _submit_pending_order(self, pending: dict):
        """Persist once, submit through the common typed contract, and adopt acceptance."""
        market = is_market_order(pending)
        qty = safe_pos_float(pending.get("qty"), default=None)
        limit = safe_pos_float(pending.get("limit"), default=None)
        if qty is None or (not market and limit is None):
            raise ValueError("invalid pending T212 order qty or limit")
        self._persist_pending_submit(pending)
        outcome = self.executor.submit_outcome_with_intent(
            pending["intent_id"], self.ticker, str(pending["side"]).lower(),
            qty, None if market else limit,
            market=market, kind=pending.get("kind"),
            reference_price=limit if market else None,
        )
        pending["submission_outcome"] = outcome.state
        if outcome.state == "accepted":
            self._adopt_pending_submit(outcome.order_id)
        elif outcome.state == "refused":
            pending["submit_error"] = outcome.reason
            self._persist_pending_submit(None)
        else:
            pending["submit_error"] = outcome.reason
            self._persist_pending_submit(pending)
        return outcome

    @staticmethod
    def _active_order_identity(order: dict) -> tuple[str, str, float | None, float | None]:
        ticker = _order_ticker(order).upper()
        side = extract_order_side(order) or ""
        raw_qty = order.get("quantity", order.get("orderedQuantity", order.get("qty")))
        qty: float | None = safe_pos_float(raw_qty, default=None) or extract_orig_qty(order, default=0.0) or None
        raw_limit = order.get("limitPrice", order.get("limit"))
        limit: float | None = safe_pos_float(raw_limit, default=None) or extract_order_price(order, default=0.0) or None
        return ticker, side, qty, limit

    def _active_matches_pending(self, order: dict, pending: dict) -> bool:
        if not isinstance(order, dict) or not isinstance(pending, dict):
            return False
        ticker, side, qty, limit = self._active_order_identity(order)
        raw_ps = pending.get("side")
        pending_side = str(raw_ps or "").strip().upper() if not isinstance(raw_ps, bool) else ""
        if not pending_side or not _t212_tickers_match(ticker, self.ticker) or side != pending_side:
            return False
        expected_qty = safe_float(pending.get("qty"), 0.0)
        if qty is None or abs(qty - expected_qty) > max(0.011, expected_qty * 1e-6):
            return False
        if is_market_order(pending):
            return True
        expected_limit = safe_float(pending.get("limit"), 0.0)
        return limit is not None and abs(limit - expected_limit) <= 0.011

    def _recover_pending_submit(self, real_qty: float,
                                active_orders: list[dict] | None) -> str:
        """Recover one response-lost T212 submit without blocking the loop."""
        pending = self.s.get("pending_submit")
        if not isinstance(pending, dict) or not pending:
            return "none"
        if active_orders is None:
            return "waiting"
        matches = [
            order for order in active_orders
            if isinstance(order, dict) and self._active_matches_pending(order, pending)
        ]
        matched_oid = extract_order_id(matches[0]) if len(matches) == 1 else None
        if len(matches) == 1 and matched_oid is not None:
            order = self._adopt_pending_submit(matched_oid)
            log(f"  [STRAT] T212 submit recovered from the active orders: {order['id']}")
            return "adopted"
        if len(matches) > 1:
            log("  ! [STRAT] ambiguous T212 submit: several active orders match — keeping it pending")
            return "waiting"

        before = safe_float(pending.get("before_qty"), 0.0)
        side = str(pending.get("side") or "").upper()
        delta = safe_float(real_qty, before) - before
        if (side == "BUY" and delta > 1e-6) or (side == "SELL" and delta < -1e-6):
            pending["portfolio_fill_observed"] = True
            self._persist_pending_submit(pending)
            return "filled"

        misses = safe_nonneg_int(pending.get("lookup_misses"), default=0) + 1
        pending["lookup_misses"] = misses
        if misses < 2:
            self._persist_pending_submit(pending)
            return "waiting"
        log(
            f"  [STRAT] T212 submit absent in 2 snapshots and no portfolio delta "
            f"({side} {self.ticker}) — releasing it for re-evaluation/retry"
        )
        self._persist_pending_submit(None)
        return "retryable"

    def _sync_real_position_cost(
        self, real_qty: float, avg_price: float, *, sync_spent_only_when_held: bool = False
    ) -> None:
        self.s["qty"] = real_qty
        self.s["cost_usd"] = real_qty * avg_price
        if not sync_spent_only_when_held or real_qty > 1e-9:
            fx_rate = safe_pos_float(self.fx_to_usd) or 1.0
            self.s["spent_cash"] = round(real_qty * avg_price / fx_rate, 2)

    def _close_cycle(
        self, exit_price: float, *, keep_pending_submit: bool = False, exit_kind: str | None = None
    ) -> None:
        self._cancel_all_orders()
        saved_orders = list(self.s.get("orders") or [])
        saved_pending_submit = self.s.get("pending_submit") if keep_pending_submit else None
        was_sl = bool(
            self._state_flag("sl_pending")
            or self._state_flag("sl_pending_exit")
            or str(exit_kind or "").strip().upper() in {"SL", "TR", "STOP"}
        )
        nxt = self._reset_cycle_state(
            ("realized_pnl_usd", "realized_net_usd", "fees_usd"),
            orders=saved_orders,
            **({"pending_submit": saved_pending_submit} if keep_pending_submit else {}),
        )
        exit_px = first_pos_float(exit_price) or safe_nonneg_float(exit_price, 0.0) or 0.0
        self._seed_post_exit_anchors(
            exit_px, "STOP" if was_sl else "TP", self._now(), include_sl_low=False
        )
        if was_sl and self.p.sl_rebuy_enabled:
            self.s["sl_rebuy"] = {"low": exit_px, "sell_price": exit_px}
            log(f"  🟢 [STRAT] pullback re-buy ARMED after the stop-loss "
                f"(expecting +{self.p.sl_rebuy_bounce_pct}% from the low)")
        log(f"  [STRAT] === cycle closed, restarting (cycle {nxt}) ===")

    # -- Fill Application ------------------------------------------------------
    def _record_realized_sell_pnl(self, avg_px: float, sell_px: float, qty: float) -> tuple[float, float, float]:
        gross = (sell_px - avg_px) * qty
        fee = (self.p.fx_fee_pct / 100.0) * (avg_px * qty + sell_px * qty)
        net = gross - fee
        for state_k, delta in (("realized_pnl_usd", gross), ("realized_net_usd", net), ("fees_usd", fee)):
            self.s[state_k] += delta
        return gross, fee, net

    def _can_close_dust_cycle(self) -> bool:
        return bool(
            self._all_tp_ladder_levels_sold()
            or self._state_flag("sl_pending")
            or self._state_flag("sl_pending_exit")
            or not self.p.tp_ladder
        )

    def _apply_fill(self, order: dict, qty: float, price: float) -> None:
        tag = "[PAPER] " if self.dry_run else ""
        if order["side"] == "BUY":
            self._apply_buy_fill_accounting(order, qty, price)
            avg = self._avg()
            log(f"  [STRAT] {tag}BUY FILLED {qty} @ {price:.2f} USD ({order.get('kind')})  "
                f"qty_total={self.s['qty']:.2f} avg={avg:.2f}")
            notify(
                title=f"{tag}{self.yahoo_sym} BUY {qty}@{price:.2f}",
                body=(f"{order.get('kind')} | q{self.s['qty']:.2f} a{avg:.2f} | "
                      f"dep{self.s['spent_cash']:.0f}{self.ccy} DCA{self.s['dca_buys']}/{self.p.max_dca_buys}"),
                source="T212", price=price, desktop=self.desktop,
            )
            self._cancel_non_market_sells()
        else:
            qty = min(qty, self.s["qty"])
            avg = self._avg() or price
            gross, fee, net = self._record_realized_sell_pnl(avg, price, qty)
            self._release_sell_cost_basis(qty, avg)
            log(f"  [STRAT] {tag}SELL FILLED {qty} @ {price:.2f} USD  "
                f"gross={gross:+.2f}  fee={fee:.2f}  net={net:+.2f} USD")
            notify(
                title=f"{tag}{self.yahoo_sym} SELL {qty}@{price:.2f} N{net:+.2f}$",
                body=(f"a{avg:.2f} · gr{gross:+.2f} fee{fee:.2f} N{net:+.2f}$ | "
                      f"Ntot{self.s['realized_net_usd']:+.2f}$ | cycle{self.s['cycle']} closed"),
                source="T212", price=price, desktop=self.desktop,
            )
            is_flat = self.s["qty"] <= 1e-9
            is_dust_exit = (
                self._is_dust(self.s["qty"], price)
                and self._can_close_dust_cycle()
            )
            if is_flat or is_dust_exit:
                self._close_cycle(price, exit_kind=order.get("kind"))

    # -- Placement & Cancellation ----------------------------------------------
    def _place_buy(self, amount: float, limit: float, kind: str) -> bool:
        lim_f = safe_pos_float(limit, None)
        amt_f = safe_pos_float(amount, None)
        if lim_f is None or amt_f is None or round(lim_f, 2) <= 0:
            log("  ! [STRAT] invalid BUY amount/limit — skipping")
            return False
        limit, amount = lim_f, amt_f
        fx_f = safe_pos_float(self.fx_to_usd, None)
        usd = safe_pos_float(amount * fx_f, default=None) if fx_f is not None else None
        qty = round(usd / limit, 2) if usd is not None else 0.0
        if qty <= 0:
            log("  ! [STRAT] qty 0 — skipping")
            return False
        if self.dry_run:
            log(f"  [STRAT] [PAPER] placing BUY {kind} {qty} @ {limit:.2f} USD (~{amount:.0f} {self.ccy})")
            self.s["orders"].append({
                "id": self._next_paper_order_id(), "side": "BUY", "qty": qty,
                "limit": round(limit, 2), "amount": amount, "kind": kind, "ts": self._now(),
            })
            return True
        if self.s.get("pending_submit"):
            log(f"  [STRAT] BUY {kind} deferred — a T212 submit is already pending")
            return False
        pending = self._new_pending_submit(
            side="BUY", qty=qty, limit=limit, amount=amount, kind=kind, level=None, market=False
        )
        outcome = self._submit_pending_order(pending)
        if outcome.state != "accepted":
            definitive = outcome.state == "refused"
            if definitive:
                log(f"  ! [STRAT] BUY {kind} explicitly refused: {outcome.reason}")
            else:
                log(f"  ! [STRAT] BUY {kind} outcome unknown: {outcome.reason}; keeping pending")
            if "insufficient" in outcome.reason.lower():
                self.s["buy_backoff_until"] = self._now() + 1800
                log("  [STRAT] insufficient funds — pausing buys for 30 minutes")
            return not definitive
        log(f"  [STRAT] BUY {kind} placed id={outcome.order_id} {qty} @ {limit:.2f}")
        return True

    def _place_sell(self, qty: float, limit: float, level: float | None = None,
                    kind: str = "TP", *, market: bool = False) -> bool:
        market = bool(_parse_bool_flag(market, False))
        qty_f = safe_pos_float(qty, None)
        lim_f = safe_pos_float(limit, None)
        if qty_f is None or lim_f is None or round(lim_f, 2) <= 0:
            log("  ! [STRAT] invalid SELL qty/limit — sending no order")
            return False
        limit, qty = lim_f, round(qty_f, 2)
        if qty <= 0:
            log("  ! [STRAT] SELL qty 0 after rounding — keeping the dust, sending no order")
            return False
        level = safe_pos_float(level, None)
        tag = f"+{level:g}%" if level is not None else ""
        if self.dry_run:
            order_type = "MKT" if market else f"@ {limit:.2f}"
            log(f"  [STRAT] [PAPER] placing SELL {kind}{tag} {qty:.2f} {order_type} USD")
            self.s["orders"].append({
                "id": self._next_paper_order_id(), "side": "SELL", "qty": qty,
                "limit": round(limit, 2), "kind": kind, "level": level, "market": market, "ts": self._now(),
            })
            return True
        if self.s.get("pending_submit"):
            log(f"  [STRAT] SELL {kind}{tag} deferred — a T212 submit is already pending")
            return False
        pending = self._new_pending_submit(
            side="SELL", qty=qty, limit=limit, amount=None, kind=kind, level=level, market=market
        )
        outcome = self._submit_pending_order(pending)
        if outcome.state != "accepted":
            message = outcome.reason
            definitive = outcome.state == "refused"
            if definitive:
                log(f"  ! [STRAT] SELL {kind}{tag} explicitly refused: {message}")
            else:
                log(f"  ! [STRAT] SELL {kind}{tag} outcome unknown: {message}; keeping pending")
            if "selling-equity-not-owned" not in message:
                return not definitive
            _m = re.search(r'owned["\']?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)', message)
            if _m is None:
                log("  ! [STRAT] selling-not-owned without the owned quantity — NOT resetting the position")
                return False
            _owned = safe_nonneg_float(_m.group(1), default=None)
            if _owned is None:
                return False
            if _owned <= 1e-6:
                log("  ! [STRAT] T212 confirms owned=0 — resetting the state (position closed/sold)")
                exit_px = safe_pos_float(self.s.get("last_sell_price")) or limit
                self._close_cycle(exit_px, exit_kind=kind)
                self.s["locked_zero_until"] = self._now() + 300
                self._save()
            else:
                log(f"  ! [STRAT] selling-not-owned but owned={_owned} (free<order) — NOT resetting the position")
            return False
        order_type = "MARKET" if market else f"@ {limit:.2f}"
        log(f"  [STRAT] SELL {kind}{tag} placed id={outcome.order_id} {qty:.2f} {order_type}")
        return True

    def _cancel_order(self, o: dict) -> bool:
        return self._cancel_tracked_order(
            o,
            "id",
            lambda order: self.executor.cancel_order_with_intent(
                order.get("intent_id") or f"legacy-t212-{self.ticker}-{order['id']}",
                self.ticker,
                str(order["id"]),
            ),
            log_prefix="cancelled order",
        )

    def _local_vol_1h_sources(self):
        return ((self._resolve_shadow_vol_1h_pct, "shadow"),)

    def _reentry_pullback_labels(self, _sp_len: int) -> tuple[str, str]:
        return "fixed (fallback)", "vol"

    def _reentry_sl_bounce_ratio(self) -> float:
        p = self.p
        return (
            vol_pct_to_ratio(p.sl_rebuy_bounce_pct)
            if p.sl_rebuy_enabled
            else (0.015 if p.sl_mhdsl_enabled else 0.0)
        )

    def _exit_config_overrides(self) -> dict[str, Any]:
        p = self.p
        return dict(
            reentry_sl_bounce_pct=self._reentry_sl_bounce_ratio(),
            reentry_bear_gate_enabled=False,
            reentry_bear_bounce_pct=0.0,
            tp_trend_ride_enabled=p.tp_mdhtp_enabled,
            tp_profit_floor_min_pct=vol_pct_to_ratio(p.trail_min_profit_pct) if p.trail_min_profit_pct > 0 else 0.0,
            tp_pullback_pct=vol_pct_to_ratio(p.trail_pct) if p.trail_pct > 0 else 0.015,
        )

    def _snapshot_provider_name(self) -> str:
        return "t212"

    def _snapshot_trend_up(self, _regime: Any) -> bool:
        slope_fn = getattr(self, "_trend_slope_provider", None)
        if slope_fn:
            try:
                if safe_nonneg_float(slope_fn(self.yahoo_sym), default=None) is not None:
                    return True
            except Exception:
                pass
        return False

    def _pre_reentry_gate_hook(self, price: float) -> tuple[bool, str] | None:
        last_sell_ts = normalize_epoch_sec(self.s.get("last_sell_ts"), None)
        if self.p.reentry_ttl_hours > 0 and last_sell_ts:
            if (self._now() - last_sell_ts) >= self.p.reentry_ttl_hours * 3600.0:
                self.s.pop("sl_rebuy", None)
        rb = self.s.get("sl_rebuy")
        if isinstance(rb, dict) and rb:
            prev_low = safe_pos_float(rb.get("low"), None)
            rb["low"] = min(prev_low, price) if prev_low is not None else price
            target = rb["low"] * (1 + self.p.sl_rebuy_bounce_pct / 100.0)
            if price >= target or are_close(price, target, self.p.reentry_tolerance_pct):
                if self.s["spent_cash"] + self.p.entry_amount > self.p.max_budget:
                    log(f"  [STRAT] re-buy SL cancelled — budget cap {self.p.max_budget:.0f} {self.ccy} reached")
                elif self._place_buy(self.p.entry_amount, price * (1 - self.p.entry_discount_pct / 100), kind="ENTRY"):
                    self.s.pop("sl_rebuy", None)
                    self.s["last_exit_kind"] = "TP"
                    if self.exit_manager is not None:
                        self.exit_manager.reentry_state.pop(self.ticker, None)
                    log(f"  🟢 [STRAT] RE-BUY after the SL: a +{self.p.sl_rebuy_bounce_pct}% pullback from the low {rb['low']:.2f} — re-entering ENTRY")
                    notify(
                        title=f"🟢 {self.yahoo_sym} RE-BUY after the stop-loss",
                        body=(f"pullback +{self.p.sl_rebuy_bounce_pct}% from low {rb['low']:.2f} — "
                              f"re-entering with {self.p.entry_amount:.0f}{self.ccy}"),
                        source="T212", price=price, desktop=self.desktop,
                    )
            return False, "handling_sl_rebuy"
        return None

    def _resolve_reentry_context(
        self,
        mgr: PositionExitManager,
        _norm_exit_kind: str,
        is_stop_exit: bool,
        _regime: Any,
    ) -> tuple[str | None, bool, bool]:
        last_kind = "STOP" if is_stop_exit else "TP"
        mgr.config.reentry_sl_bounce_pct = self._reentry_sl_bounce_ratio()
        mgr.config.reentry_bear_gate_enabled = False
        mgr.config.reentry_bear_bounce_pct = 0.0
        trend_up, trend_down = bool(self.p.reentry_peak_relative), False
        slope_fn = getattr(self, "_trend_slope_provider", None)
        if not trend_up and slope_fn:
            try:
                slope_f = safe_float(slope_fn(self.yahoo_sym), default=None)
                if slope_f is not None:
                    if slope_f >= 0.0:
                        trend_up = True
                    elif slope_f < (-self.p.dca_trend_gate_pct if self.p.dca_trend_gate_pct > 0 else -0.1):
                        trend_down = True
            except Exception:
                pass
        return last_kind, trend_up, trend_down

    def _resolve_reentry_custom_drop(self, _is_bull_reentry: bool, _last_kind: str | None) -> float | None:
        return None

    def _place_entry_order(self, entry_amt: float, entry_px: float) -> bool:
        return self._place_buy(entry_amt, entry_px, kind="ENTRY")

    def _cancel_all_orders(self) -> bool:
        success = True
        for order in list(self.s["orders"]):
            if not self._cancel_order(order):
                success = False
        return success

    def _cancel_non_market_sells(self) -> None:
        for order in [
            x for x in list(self.s["orders"]) if str(x.get("side") or "").upper() == "SELL" and not is_market_order(x)
        ]:
            self._cancel_order(order)

    def _restore_cancelled_orders(self, cancelled_orders: list[dict], reason: str = "") -> None:
        """Best-effort rollback: restore previously resting orders after a replacement or exit fails."""
        def _restore_one(p: dict) -> bool:
            if p["side"] == "SELL":
                return bool(self._place_sell(p["qty"], p["limit"], level=p.get("level"), kind=p.get("kind", "TP")))
            if p["side"] == "BUY":
                raw_ord = p.get("raw") if isinstance(p.get("raw"), dict) else p
                usd_notional = p["qty"] * p["limit"]
                fallback_amt = (usd_notional / self.fx_to_usd) if self.fx_to_usd > 0 else usd_notional
                amount = safe_pos_float(raw_ord.get("amount"), default=fallback_amt)
                return bool(self._place_buy(amount, p["limit"], kind=p.get("kind", "ENTRY")))
            return False

        restore_cancelled_orders(cancelled_orders, restore_fn=_restore_one, log_fn=log, reason=reason)

    def _is_dust(self, qty: float, ref_price: float | None = None) -> bool:
        """Return True if qty is negligible dust (below minimum broker holding / trading value)."""
        if qty <= 1e-9:
            return True
        if self.p.ladder_min_free <= 0:
            return False
        p = ref_price or self._avg() or 0.0
        if p <= 0:
            return False
        min_dust = max(5.0, self.p.ladder_min_free * 1.5)
        max_allowed = (self.p.entry_amount * 0.5) if self.p.entry_amount > 0 else min_dust
        return (qty * p) <= min(min_dust, max_allowed)

    def _sellable_qty(self, held: float, ref_price: float | None = None) -> float:
        """Return the maximum safely sellable quantity avoiding T212 min-opened-position."""
        p = ref_price or self._avg() or 0.0
        if self._is_dust(held, p):
            return 0.0
        q_round = round(held, 2)
        dust = held - q_round
        if abs(dust) > 1e-6 and self.p.ladder_min_free > 0 and p > 0:
            if abs(dust) * p < self.p.ladder_min_free:
                buf = self.p.ladder_min_free / p
                if held > buf:
                    return round(held - buf, 2)
        return q_round

    def _manage_tp_ladder(self, held: float, avg: float) -> None:
        """Maintain one SELL per unsold ladder level at its target over average."""
        for o in [x for x in self.s["orders"] if x["side"] == "SELL" and x.get("level") is None]:
            if not self._cancel_order(o) or o in self.s["orders"]:
                return
        sold = set(self.s.get("tp_sold_levels", []))
        remaining = [(lvl, frac) for (lvl, frac) in self.p.tp_ladder if lvl not in sold]
        if not remaining or held <= 1e-9:
            return
        open_sells = {o.get("level"): o for o in self.s["orders"] if o["side"] == "SELL" and o.get("level") is not None}
        locked_partial: dict[float, tuple[float, float]] = {}
        reserved_partial_qty = 0.0
        for lvl, _ in remaining:
            o = open_sells.get(lvl)
            lim_px = round(avg * (1 + lvl / 100.0), 2)
            if (
                o is not None
                and _parse_bool_flag(o.get("_replay_had_fill"), False)
                and lim_px > 0
                and abs(o["limit"] - lim_px) / lim_px <= 0.001
                and 0 < o["qty"] <= held - reserved_partial_qty + 1e-6
            ):
                locked_partial[lvl] = (o["qty"], o["limit"])
                reserved_partial_qty += o["qty"]
        unstarted = [(lvl, frac) for (lvl, frac) in remaining if lvl not in locked_partial]
        free_held = max(0.0, held - reserved_partial_qty)
        desired = dict(locked_partial)
        if unstarted and free_held > 1e-9:
            unstarted = adapt_ladder_rungs(unstarted, free_held, min_tranche_qty=0.01)
            total = sum(f for _, f in unstarted)
            if total > 0:
                buf = (self.p.ladder_min_free / avg) if (self.p.ladder_min_free > 0 and avg > 0) else 0.0
                unstarted_sorted = sorted(unstarted, key=lambda x: x[0])
                acc = 0.0
                for i, (lvl, frac) in enumerate(unstarted_sorted):
                    last = (i == len(unstarted_sorted) - 1)
                    q = round(free_held - acc - buf, 2) if last else round(free_held * frac / total, 2)
                    acc += q
                    lim_px = round(avg * (1 + lvl / 100.0), 2)
                    if q > 0 and lim_px > 0:
                        desired[lvl] = (q, lim_px)
        cancelled_by_level = {}
        for lvl, o in list(open_sells.items()):
            d = desired.get(lvl)
            if d is None or d[1] <= 0 or abs(o["limit"] - d[1]) / d[1] > 0.001 or abs(o["qty"] - d[0]) > 1e-6:
                cancelled_by_level[lvl] = dict(o)
                if self._cancel_order(o) and o not in self.s["orders"]:
                    open_sells.pop(lvl, None)
        now = self._now()
        if not isinstance(self.s.get("tp_fail_until"), dict):
            self.s["tp_fail_until"] = {}
        fails = self.s["tp_fail_until"]
        for lvl, (q, lim) in desired.items():
            if lvl in open_sells or now < safe_nonneg_float(fails.get(str(lvl)), 0.0):
                continue
            if self.s.get("pending_submit"):
                break
            if not self._place_sell(q, lim, level=lvl):
                if not self.s.get("pending_submit"):
                    fails[str(lvl)] = now + 1800
                    old_o = cancelled_by_level.get(lvl)
                    if old_o:
                        self._restore_cancelled_orders([old_o], reason=f"TP ladder +{lvl:g}% replace failed")
                break

    def _buy_is_stale(self, order: dict, price: float) -> bool:
        """Return whether a resting limit BUY should be cancelled and placed again."""
        if str(order.get("side") or "").strip().upper() != "BUY" or is_market_order(order):
            return False
        order_ts = safe_nonneg_float(order.get("ts"), 0.0)
        order_lim = safe_pos_float(order.get("limit"), None)
        if order_lim is None or (self._now() - order_ts) / 60 <= self.p.order_ttl_min:
            return False
        return price * (1 - self.p.entry_discount_pct / 100) > order_lim * 1.003

    # -- Reconciliation --------------------------------------------------------
    def reconcile(self, price: float) -> bool:
        px = safe_pos_float(price, None)
        if px is None:
            return False
        if self.dry_run:
            self._reconcile_paper(px)
            return True
        return self._reconcile_real(px)

    def _reconcile_paper(self, price: float) -> None:
        for side in ("BUY", "SELL"):
            for o in [x for x in self.s["orders"] if x["side"] == side]:
                if o not in self.s["orders"]:
                    continue
                if side == "BUY":
                    self._remove(o)
                    self._apply_fill(o, o["qty"], o["limit"])
                elif is_market_order(o) or price >= o["limit"]:
                    self._remove(o)
                    if o.get("level") is not None:
                        self._record_tp_sold_level(o["level"])
                    self._apply_fill(o, o["qty"], price if is_market_order(o) else o["limit"])

    def _portfolio_position(self) -> tuple[float, float] | None:
        pf = self.client.get_portfolio()
        if pf is None:
            return None
        for p in pf:
            if isinstance(p, dict) and _t212_tickers_match(p.get("ticker", ""), self.ticker):
                q_f = safe_nonneg_float(p.get("quantity"), default=0.0 if p.get("quantity") is None else None)
                avg_f = safe_nonneg_float(p.get("averagePrice"), default=0.0 if p.get("averagePrice") is None else None)
                return None if (q_f is None or avg_f is None) else (q_f, avg_f)
        return 0.0, 0.0

    def _tracked_order_status(self, order: dict, cache: dict[str, object]):
        """Return strict order status, cached for one reconciliation pass."""
        key = str(order.get("id"))
        if key in cache:
            return cache[key]
        try:
            status = self.executor.order_status_with_intent(
                order.get("intent_id") or f"legacy-t212-{self.ticker}-{key}",
                self.ticker, key,
            )
        except OrderNotFound:
            log(f"  [STRAT] T212 order {key} definitively not found on venue")
            cache[key] = "not_found"
            return "not_found"
        except ProviderError as exc:
            log(f"  ! [STRAT] T212 status {key} unavailable: {exc} — keeping the order")
            cache[key] = None
            return None
        cache[key] = status
        return status

    def _exact_execution_price(self, side: str, position_delta: float,
                               cache: dict[str, object]) -> float | None:
        """Return the weighted price from tracked orders' cumulative deltas."""
        rows = []
        total_qty = total_cost = 0.0
        for order in [o for o in self.s["orders"] if o.get("side") == side]:
            status = self._tracked_order_status(order, cache)
            if status is None or isinstance(status, str):
                continue
            cq = safe_nonneg_float(status.filled_qty, default=None)
            cc = safe_nonneg_float(status.cost, default=None)
            cf = safe_float(status.fee, default=0.0)
            aq = safe_nonneg_float(order.get("applied_fill_qty"), default=0.0)
            ac = safe_nonneg_float(order.get("applied_fill_cost"), default=0.0)
            af = safe_float(order.get("applied_fill_fee"), default=0.0)
            if cq is None or cc is None or cf is None or aq is None or ac is None or af is None:
                continue
            if cq + 1e-9 < aq or cc + 1e-9 < ac:
                log(f"  ! [STRAT] T212 fill {order.get('id')} regressed cumulatively — ignoring")
                continue
            delta_qty = max(0.0, cq - aq)
            delta_cost = max(0.0, cc - ac)
            if delta_qty > 1e-9:
                rows.append((order, cq, cc, cf))
                total_qty += delta_qty
                total_cost += delta_cost

        tolerance = max(0.011, abs(position_delta) * 1e-6)
        if not rows or abs(total_qty - position_delta) > tolerance or total_cost <= 0:
            if rows:
                log(f"  ! [STRAT] T212 {side} fills ({total_qty:.4f}) != portfolio delta "
                    f"({position_delta:.4f}) - using portfolio fallback")
            hist_fn = getattr(self.client, "latest_filled_order_for_ticker", None)
            if callable(hist_fn) and not hasattr(hist_fn, "_mock_name"):
                latest = hist_fn(self.ticker, side=side, refresh=True)
                if isinstance(latest, dict):
                    fill_oid = str(latest.get("orderId") or latest.get("id") or "").strip()
                    fill_ts_s = (
                        normalize_epoch_sec(latest.get("timestamp"), None)
                        or normalize_epoch_sec(latest.get("time"), None)
                    )
                    last_sell_ts = safe_pos_float(self.s.get("last_sell_ts"), None)
                    stale_hist = bool(
                        (fill_oid and fill_oid == str(self.s.get("last_hist_fill_id") or ""))
                        or (side == "SELL" and fill_ts_s is not None and last_sell_ts is not None and fill_ts_s <= last_sell_ts + 0.5)
                    )
                    if not stale_hist and fill_ts_s is not None:
                        latest_any = hist_fn(self.ticker, side=None, refresh=False)
                        if isinstance(latest_any, dict):
                            any_side = str(latest_any.get("side") or "").strip().upper()
                            any_ts_s = (
                                normalize_epoch_sec(latest_any.get("timestamp"), None)
                                or normalize_epoch_sec(latest_any.get("time"), None)
                            )
                            if any_side and any_side != side.upper() and any_ts_s is not None and any_ts_s >= fill_ts_s:
                                stale_hist = True
                    fill_qty = first_pos_float(latest.get("executedQty"), latest.get("qty"))
                    fill_px = safe_pos_float(latest.get("price"), None)
                    if (
                        not stale_hist
                        and fill_qty is not None
                        and fill_px is not None
                        and abs(fill_qty - abs(position_delta)) <= max(0.011, abs(position_delta) * 0.05)
                    ):
                        if fill_oid:
                            self.s["last_hist_fill_id"] = fill_oid
                        return fill_px
            return None

        for order, cq, cc, cf in rows:
            order["applied_fill_qty"] = cq
            order["applied_fill_cost"] = cc
            order["applied_fill_fee"] = cf
        return total_cost / total_qty

    def _reconcile_real(self, price: float) -> bool:
        real = self._portfolio_position()
        if real is None:
            now = self._now()
            if now - getattr(self, "_pf_unavail_logged", 0) > 600:
                log("  [STRAT] portfolio unavailable - skipping reconciliation (suppressing repeats for 10 min)")
                self._pf_unavail_logged = now
            return False
        real_qty, real_avg = real
        active_orders = self.client.list_active_orders()
        pending_outcome = self._recover_pending_submit(real_qty, active_orders)
        pending_for_fill = dict(self.s.get("pending_submit") or {}) if pending_outcome == "filled" else None
        if active_orders is None:
            active = {str(o["id"]) for o in self.s["orders"]}
        else:
            active = {
                oid
                for o in active_orders
                if isinstance(o, dict)
                and _t212_tickers_match(self._active_order_identity(o)[0], self.ticker)
                and (oid := extract_order_id(o)) is not None
            }

        prev_qty = self.s["qty"]
        prev_avg = self._avg() or real_avg
        status_cache: dict[str, object] = {}

        if real_qty > prev_qty + 1e-6 and self._now() < safe_nonneg_float(self.s.get("locked_zero_until"), 0.0):
            log("  [STRAT] adoption ignored — stale portfolio (recent not-owned, lock active)")
            return False
        if real_qty > prev_qty + 1e-6:
            fq = real_qty - prev_qty
            exact_price = self._exact_execution_price("BUY", fq, status_cache)
            source_order = next((o for o in self.s["orders"] if o.get("side") == "BUY"), None)
            if source_order is None and pending_for_fill and pending_for_fill.get("side") == "BUY":
                source_order = pending_for_fill
            raw_marginal = (real_avg * real_qty - prev_avg * prev_qty) / fq if (fq > 0 and real_avg > 0) else 0.0
            fp = (
                exact_price
                or (safe_pos_float(raw_marginal, None) if raw_marginal > 0 else None)
                or safe_pos_float((source_order or {}).get("limit"), None)
                or safe_pos_float(real_avg, None)
                or price
            )
            eff_real_avg = real_avg if real_avg > 0 else (prev_avg if (prev_avg and prev_avg > 0) else fp)
            is_adoption = source_order is None and prev_qty < 1e-9
            if is_adoption and self._is_dust(real_qty, eff_real_avg):
                now = self._now()
                if now - getattr(self, "_dust_logged", 0) > 600:
                    log(
                        f"  [STRAT] portfolio dust ({real_qty:.4f} shares, ~{real_qty*eff_real_avg:.2f} USD) "
                        "ignored — staying flat for new cycle"
                    )
                    self._dust_logged = now
                return True
            if source_order is not None and source_order.get("kind") == "DCA" and not source_order.get("dca_counted"):
                source_order["dca_counted"] = True
                self.s["dca_buys"] += 1
            elif source_order is None and prev_qty > 1e-9:
                self.s["dca_buys"] += 1
            self.s["last_buy_price"] = fp
            if self.s["entry_price"] is None:
                self.s["entry_price"] = fp
            self._sync_real_position_cost(real_qty, eff_real_avg)
            kind_label = (
                "ADOPTED" if is_adoption
                else str((source_order or {}).get("kind") or ("DCA" if prev_qty > 1e-9 else "ENTRY"))
            )
            log(f"  [STRAT] BUY EXECUTED {fq:.4f} @ {fp:.2f} USD ({kind_label})  qty={real_qty:.4f} avg={eff_real_avg:.2f}")
            notify(
                title=f"{self.yahoo_sym} {'ADOPTED' if is_adoption else 'BUY'} {fq:.4f}@a{eff_real_avg:.2f}",
                body=(f"{kind_label} | q{real_qty:.4f} a{eff_real_avg:.2f} p~{fp:.2f} | "
                      f"dep{self.s['spent_cash']:.0f}{self.ccy} DCA{self.s['dca_buys']}/{self.p.max_dca_buys}"),
                source="T212", price=fp, desktop=self.desktop,
            )
            if source_order is pending_for_fill:
                self._persist_pending_submit(None)
            self._unconfirmed_zero_polls = 0
            self._cancel_non_market_sells()

        elif real_qty < prev_qty - 1e-6:
            sold = prev_qty - real_qty
            exact_price = self._exact_execution_price("SELL", sold, status_cache)
            has_tracked_sell = (
                any(o.get("side") == "SELL" for o in self.s["orders"])
                or bool(pending_for_fill and pending_for_fill.get("side") == "SELL")
            )
            hist_fn = getattr(self.client, "latest_filled_order_for_ticker", None)
            if (
                real_qty <= 1e-9
                and exact_price is None
                and not has_tracked_sell
                and callable(hist_fn)
                and not hasattr(hist_fn, "_mock_name")
                and getattr(self, "_unconfirmed_zero_polls", 0) < 1
            ):
                self._unconfirmed_zero_polls = getattr(self, "_unconfirmed_zero_polls", 0) + 1
                log(f"  [STRAT] {self.yahoo_sym} portfolio qty transiently 0 with no tracked SELL or fill history — awaiting confirmation")
                return False
            self._unconfirmed_zero_polls = 0
            sell_price = exact_price or price
            gross, fee, net = self._record_realized_sell_pnl(prev_avg, sell_price, sold)
            self._sync_real_position_cost(real_qty, real_avg)
            self.s["last_sell_price"] = sell_price
            self.s["last_sell_ts"] = self._now()
            approx = "" if exact_price is not None else "~"
            log(f"  [STRAT] SELL EXECUTED {sold:.4f} @ {approx}{sell_price:.2f} USD  "
                f"gross={gross:+.2f}  fee={fee:.2f}  net={net:+.2f} USD")
            notify(
                title=f"{self.yahoo_sym} SELL {sold:.4f}@{approx}{sell_price:.2f} N{net:+.2f}$",
                body=f"a{prev_avg:.2f} · gr{gross:+.2f} fee{fee:.2f} N{net:+.2f}$ | Ntot{self.s['realized_net_usd']:+.2f}$",
                source="T212", price=sell_price, desktop=self.desktop,
            )
            if pending_for_fill and pending_for_fill.get("side") == "SELL":
                self._persist_pending_submit(None)

        else:
            self._unconfirmed_zero_polls = 0
            self._sync_real_position_cost(real_qty, real_avg, sync_spent_only_when_held=True)

        for o in list(self.s["orders"]):
            if str(o["id"]).startswith("PAPER"):
                continue
            if str(o["id"]) not in active:
                status = self._tracked_order_status(o, status_cache)
                if status is not None and not isinstance(status, str) and status.status in {"closed", "canceled", "expired"}:
                    if (
                        o["side"] == "SELL" and o.get("level") is not None
                        and safe_nonneg_float(status.filled_qty, default=0.0) > 1e-9
                    ):
                        self._record_tp_sold_level(o["level"])
                    self._remove(o)
                elif status == "not_found":
                    misses = safe_nonneg_int(o.get("lookup_misses"), default=0) + 1
                    o["lookup_misses"] = misses
                    if misses >= (2 if _parse_bool_flag(o.get("cancel_requested"), False) else 3):
                        log(f"  [STRAT] order {o.get('side', '?')} {o['id']} confirmed absent ({misses} misses) — removing from tracking")
                        if (
                            o.get("side") == "SELL" and o.get("level") is not None
                            and safe_nonneg_float(o.get("applied_fill_qty"), default=0.0) > 1e-9
                        ):
                            self._record_tp_sold_level(o["level"])
                        self._remove(o)
                continue
            o.pop("lookup_misses", None)
            if self._buy_is_stale(o, price):
                self._log_and_cancel_stale_buy(o, "id", "BUY", "—")

        is_flat = real_qty <= 1e-9
        is_dust = self._is_dust(real_qty, real_avg or price)
        cycle_closed = (is_flat and prev_qty > 1e-9) or (
            prev_qty > 1e-9 and is_dust and not self._has_open("SELL")
            and self._can_close_dust_cycle()
        )
        if cycle_closed:
            exit_kind = (pending_for_fill or {}).get("kind") if isinstance(pending_for_fill, dict) else None
            self._close_cycle(
                first_pos_float(self.s.get("last_sell_price"), price) or price,
                keep_pending_submit=True,
                exit_kind=exit_kind,
            )
        return True

    # -- Protective Exits & Decision Step --------------------------------------
    def _execute_protective_exit(
        self, kind: str, price: float, label: str, fail_key: str, sell_qty: float | None = None
    ) -> bool:
        """Cancel existing non-protective orders and submit a protective market sell."""
        existing = next((o for o in self.s["orders"] if o.get("kind") == kind), None)
        if existing is None or (not is_market_order(existing) and price < existing["limit"]):
            cancelled_orders = [dict(o) for o in self.s["orders"] if o.get("kind") != kind]
            if not self._cancel_all_orders():
                log(f"  ! [STRAT] {label} deferred: at least one order could not be cancelled")
                return False
            pending_orders = [o for o in self.s["orders"] if o.get("kind") != kind]
            if pending_orders:
                log(f"  [STRAT] {label} deferred: waiting for {len(pending_orders)} existing order(s) to reconcile cancellation")
                return False
            if self.s.get("pending_submit"):
                log(f"  [STRAT] {label} deferred: waiting for pending submit to reconcile")
                return False
            target_qty = sell_qty if sell_qty is not None else self._sellable_qty(self.s["qty"], price)
            if not self._place_sell(target_qty, round(price, 2), kind=kind, market=True):
                if not self.s.get("pending_submit"):
                    self.s[fail_key] = self._now() + 1800
                    log(f"  ! [STRAT] {label} sell failed or refused — backoff 30 min (retaining TP ladder)")
                    self._restore_cancelled_orders(cancelled_orders, reason=f"{label.lower()} sell failed")
                return False
            self.s["sl_pending"] = True
            if kind == "SL":
                self.s["tp_pending_exit"] = None
        return True

    def _check_trailing(self, price: float) -> bool:
        """Sell the entire position after a ``trail_pct`` drop from its peak."""
        if self.p.trail_pct <= 0 or self.s["qty"] <= 1e-9:
            self.s["pos_peak"] = 0.0
            self.s["tr_alerted"] = False
            self.s["tr_armed"] = False
            return False
        if any(o.get("kind") in {"TR", "SL"} for o in self.s["orders"]):
            return True
        if self.p.tp_mdhtp_enabled and self.exit_manager and self.ticker in self.exit_manager.tp_ride_state:
            return False

        avg = self._avg()
        peak = safe_pos_float(self.s.get("pos_peak")) or 0.0
        if not peak or peak <= 0:
            self.s["pos_peak"] = peak = max(avg or price, price)
        if price > peak:
            self.s["pos_peak"] = peak = price
            self.s.pop("tr_fail_until", None)
            self.s["tr_alerted"] = False

        if not self._state_flag("tr_armed"):
            minp = self.p.trail_min_profit_pct
            if minp > 0 and (not avg or price < avg * (1 + minp / 100.0)):
                return False
            self.s["tr_armed"] = True
            self.s["pos_peak"] = peak = price
            log(
                f"  [STRAT] trailing ARMED: price {price:.2f} >= avg{(avg or 0):.2f}+{minp}% — "
                f"protecting from now on (-{self.p.trail_pct}% from the peak)"
            )

        drop_pct = (peak - price) / peak * 100 if peak > 0 else 0.0
        if drop_pct < self.p.trail_pct or self._now() < safe_nonneg_float(self.s.get("tr_fail_until"), 0.0):
            return False
        if self._execute_protective_exit("TR", price, "TRAILING", "tr_fail_until"):
            if not self._state_flag("tr_alerted"):
                self.s["tr_alerted"] = True
                log(f"  📉 [STRAT] TRAILING: -{drop_pct:.2f}% from peak {peak:.2f} >= {self.p.trail_pct}% — SELLING ALL")
                notify(
                    title=f"📉 TRAILING {self.yahoo_sym} -{drop_pct:.1f}%",
                    body=f"from peak {peak:.2f} ≥{self.p.trail_pct}% — selling all",
                    source="T212", price=price, desktop=self.desktop,
                )
        return True

    def _alert_stop_loss_once(self, price: float, loss_pct: float, exit_reason: str = "") -> None:
        if self._state_flag("sl_alerted"):
            return
        self.s["sl_alerted"] = True
        reason_sfx = f" ({exit_reason})" if exit_reason else ""
        label_pfx = "MHDSL " if exit_reason else ""
        log(
            f"  🛑 [STRAT] {label_pfx}STOP-LOSS: loss {loss_pct:.2f}% >= {self.p.stop_loss_pct}%{reason_sfx} "
            "— SELLING EVERYTHING (cutting the loss)"
        )
        notify(
            title=f"🛑 SL {self.yahoo_sym} -{loss_pct:.1f}%",
            body=f"loss {loss_pct:.1f}% >=threshold{self.p.stop_loss_pct}%{reason_sfx} — selling everything",
            source="T212", price=price, desktop=self.desktop,
        )

    def _reset_stop_loss_state(self) -> bool:
        self.s["sl_alerted"] = False
        self.s["sl_pending"] = False
        self.s.pop("sl_fail_until", None)
        return False

    def _check_stop_loss(self, price: float) -> bool:
        """Close the entire position when unrealized loss exceeds the threshold."""
        if self.p.stop_loss_pct <= 0:
            return False
        if any(o.get("kind") in {"TR", "SL"} for o in self.s["orders"]):
            return True
        sl_ctx = self._resolve_stop_loss_avg_and_loss_pct(price)
        if not sl_ctx:
            return False
        avg, loss_pct = sl_ctx

        if self.exit_manager is not None and self.p.sl_mhdsl_enabled:
            if not self._in_mhdsl_territory(loss_pct):
                return self._reset_stop_loss_state()

            snapshot = self._create_position_snapshot(price, avg, self.s.get("qty", 0.0))
            sell_qty = snapshot.sellable_qty
            raw_pending_sl = self.s.get("sl_pending_exit")
            pending_sl = raw_pending_sl if isinstance(raw_pending_sl, dict) else None
            if pending_sl:
                should_exit_sl, exit_reason, decision = True, str(pending_sl.get("reason") or "mhdsl_pending_exit"), None
            else:
                decision = self.exit_manager.evaluate_stop_loss(snapshot, custom_threshold=self.p.stop_loss_pct / 100.0)
                should_exit_sl = bool(decision.action == ExitAction.SELL and decision.should_exit)
                exit_reason = decision.reason if should_exit_sl else ""

            if should_exit_sl:
                if self._now() < safe_nonneg_float(self.s.get("sl_fail_until"), 0.0):
                    return False
                if pending_sl is None:
                    self.s["sl_pending_exit"] = {"reason": exit_reason, "time": self._now()}
                if self._execute_protective_exit("SL", price, "STOP", "sl_fail_until", sell_qty=sell_qty):
                    self.s["sl_pending_exit"] = None
                    self._alert_stop_loss_once(price, loss_pct, exit_reason)
                elif self.s.get("sl_fail_until"):
                    self.s["sl_pending_exit"] = None
                return True
            if decision is not None and decision.action == ExitAction.HOLD:
                self._handle_mhdsl_hold_decision(decision)
            return False

        if loss_pct < self.p.stop_loss_pct:
            return self._reset_stop_loss_state()
        if self._now() < safe_nonneg_float(self.s.get("sl_fail_until"), 0.0):
            return False
        if self._execute_protective_exit("SL", price, "STOP", "sl_fail_until"):
            self._alert_stop_loss_once(price, loss_pct)
        return True

    # -- Step Pipeline Sub-Stages ----------------------------------------------
    def _step_sync_market(self, price: float) -> tuple[float, float, float, bool] | None:
        """Stage 1: Validate market price, normalize held quantity, check loss alerts, and observe_only."""
        px = safe_pos_float(price, None)
        if px is None:
            return None
        price = px
        self._shadow_prices.append((self._now(), price))
        held = safe_nonneg_float(self.s.get("qty"), 0.0)
        self.s["qty"] = held
        step = self.p.loss_alert_step
        avg = self._avg()
        if step > 0 and avg:
            loss_pct = (avg - price) / avg * 100
            band = math.floor(loss_pct / step) if loss_pct > 0 else 0
            if band > self.s.get("loss_band", 0):
                log(f"  📉 [STRAT] {self.yahoo_sym} loss -{loss_pct:.1f}% (threshold {band*step:.0f}%)")
                notify(
                    title=f"📉 {self.yahoo_sym} -{loss_pct:.1f}%",
                    body=f"loss -{loss_pct:.1f}% (over {band*step:.0f}%) | q{self.s['qty']:.2f} a{avg:.2f} p{price:.2f}",
                    source="T212", price=price, desktop=self.desktop,
                )
                self.s["loss_band"] = band
        if self.observe_only:
            if held > 1e-9:
                avg = self._avg()
                peak = safe_pos_float(self.s.get("pos_peak")) or 0.0
                if peak <= 0:
                    self.s["pos_peak"] = max(avg or price, price)
                elif price > peak:
                    self.s["pos_peak"] = price
            if not self.dry_run and self.s.get("orders"):
                self._cancel_all_orders()
            return None
        disc = 1 - self.p.entry_discount_pct / 100
        in_backoff = self._now() < safe_nonneg_float(self.s.get("buy_backoff_until"), 0.0)
        return price, held, disc, in_backoff

    def _step_evaluate_exits(self, price: float) -> bool:
        """Stage 2: Evaluate trailing crash breaker and stop-loss exits."""
        return self._check_trailing(price) or self._check_stop_loss(price)

    def _step_evaluate_reentry(self, price: float, disc: float, *, in_backoff: bool) -> bool:
        """Stage 3: Evaluate flat-position re-entry gate and place initial ENTRY buy order."""
        self._clear_state_slots("tp_pending_exit", "sl_pending_exit")
        if in_backoff or self._has_open("BUY"):
            return False
        return self._execute_flat_reentry(
            price, price * disc, self.p.entry_amount, self.p.max_budget, cap_label="budget cap"
        )

    def _evaluate_mdhtp_tp(self, price: float, avg: float, held: float) -> bool:
        """Evaluate Multi-Horizon Dynamic Take-Profit (MDHTP); return True if tick TP handling is complete."""
        tick_time = self._now()
        raw_pending_exit = self.s.get("tp_pending_exit")
        pending_exit = raw_pending_exit if isinstance(raw_pending_exit, dict) else None
        decision = None

        if pending_exit is None:
            tp_thresh = self.p.takeprofit_pct / 100.0
            in_tp_zone = (
                not self.replay_mode
                or self.ticker in self.exit_manager.tp_ride_state
                or self.exit_manager.config.hard_tp_enabled
                or (price > avg and (price - avg) / avg >= tp_thresh * 0.98)
            )
            if not in_tp_zone and price > 0 and avg > 0:
                self.s["trail_peak"] = None
                if self.s["orders"]:
                    self._cancel_non_market_sells()
            else:
                self._sync_mdhtp_config(
                    self.p.trail_pct if self.p.trail_pct > 0 else 1.5, trend_ride_enabled=True
                )
                snapshot = self._create_position_snapshot(
                    price, avg, held, include_vol=not self.replay_mode, include_trend=in_tp_zone
                )
                decision = self.exit_manager.evaluate_take_profit(snapshot, custom_threshold=tp_thresh)
                if decision.action in (ExitAction.SELL, ExitAction.PARTIAL_SELL) and decision.should_exit:
                    pending_exit = {
                        "action": decision.action.value,
                        "sell_fraction": (
                            decision.sell_fraction
                            if decision.action == ExitAction.PARTIAL_SELL and decision.sell_fraction
                            else 1.0
                        ),
                        "reason": decision.reason,
                        "time": tick_time,
                    }
                    self.s["tp_pending_exit"] = pending_exit
                    log(f"  [STRAT] MDHTP EXIT TRIGGERED ({decision.reason}) — initiating cancellation & exit sequence")

        if pending_exit:
            def _is_market_exit(o: dict) -> bool:
                return bool(
                    is_market_order(o) and str(o.get("side", "")).upper() == "SELL" and o.get("kind") in ("TP", "SL")
                )

            existing_exit = next((o for o in self.s.get("orders", []) if _is_market_exit(o)), None)
            if existing_exit:
                for o in [o for o in list(self.s["orders"]) if not _is_market_exit(o)]:
                    self._cancel_order(o)
                if self.s["qty"] <= 1e-6:
                    log("  [STRAT] MDHTP exit: position closed — clearing exit intent")
                    self.s["tp_pending_exit"] = None
                else:
                    log(
                        f"  [STRAT] MDHTP exit: existing market exit order {existing_exit.get('id')} "
                        "in progress — awaiting fill reconciliation"
                    )
                return True

            if not self._cancel_all_orders():
                log("  ! [STRAT] MDHTP exit deferred: at least one order could not be cancelled")
                return True
            pending_orders = [o for o in self.s["orders"] if not _is_market_exit(o)]
            if pending_orders:
                log(f"  [STRAT] MDHTP exit deferred: waiting for {len(pending_orders)} existing order(s) to reconcile cancellation")
                return True
            if self.s.get("pending_submit"):
                log("  [STRAT] MDHTP exit deferred: waiting for pending submit to reconcile")
                return True

            fresh_held = self.s["qty"]
            raw_sf = safe_pos_float(pending_exit.get("sell_fraction"), None)
            sell_fraction = min(1.0, raw_sf) if raw_sf is not None else 1.0
            target_qty = fresh_held * sell_fraction if sell_fraction < 1.0 else fresh_held
            sell_qty = self._sellable_qty(target_qty, price)
            if sell_qty <= 0 or self._is_dust(sell_qty, price):
                log("  [STRAT] MDHTP exit: no sellable quantity remaining — clearing exit intent")
                self.s["tp_pending_exit"] = None
                return True

            exit_px = round(price * 0.999, 2)
            if self._place_sell(sell_qty, exit_px, kind="TP", market=True):
                if pending_exit.get("action") == ExitAction.PARTIAL_SELL.value and self.exit_manager is not None:
                    self.exit_manager.record_hard_tp(self.ticker, tick_time)
                log(f"  [STRAT] MDHTP EXIT ({pending_exit.get('reason')}) -> reference {exit_px} ({sell_qty} shares)")
                self.s["tp_pending_exit"] = None
            else:
                log(f"  ! [STRAT] MDHTP EXIT order refused/failed ({pending_exit.get('reason')}) — intent retained for retry")
            return True

        if decision is not None and decision.reason == "not_in_tp_territory":
            self.s["trail_peak"] = None
            self._cancel_non_market_sells()
        elif decision is not None and decision.action == ExitAction.HOLD:
            if self.ticker in self.exit_manager.tp_ride_state:
                ride_state = self.exit_manager.tp_ride_state.get(self.ticker, {})
                peak_px = safe_pos_float(ride_state.get("peak_price")) or price
                self.s["trail_peak"] = peak_px
                self._cancel_non_market_sells()
                log(f"  [STRAT] above the TP, RIDING IT ({decision.reason}) (peak {peak_px:.2f})")
                return True
            self.s["trail_peak"] = None
            self._cancel_non_market_sells()
        return False

    def _dispatch_take_profit(self, price: float, avg: float | None, held: float) -> bool:
        """Manage take-profit (MDHTP, ladder, or single TP). Return True to skip DCA."""
        if not (self.p.enable_takeprofit and avg and not self._state_flag("sl_pending")):
            return False
        if self.p.tp_mdhtp_enabled and self.exit_manager is not None:
            return self._evaluate_mdhtp_tp(price, avg, held)
        if self.p.tp_ladder:
            self._manage_tp_ladder(held, avg)
            return False
        target = avg * (1 + self.p.takeprofit_pct / 100)
        if target > 0:
            sell_qty = self._sellable_qty(held, target)
            sell = self._find_open("SELL")
            target_lim = round(target, 2)
            if sell is None:
                self._place_sell(sell_qty, target)
            elif (
                target_lim <= 0
                or abs(sell["limit"] - target_lim) / target_lim > 0.001
                or (
                    not (_parse_bool_flag(sell.get("_replay_had_fill"), False) and 0 < sell["qty"] <= held + 1e-6)
                    and abs(sell["qty"] - sell_qty) > 1e-6
                )
            ):
                old_sell = dict(sell)
                if self._cancel_open("SELL") and sell not in self.s["orders"]:
                    if not self._place_sell(sell_qty, target) and not self.s.get("pending_submit"):
                        self._restore_cancelled_orders([old_sell], reason="single TP replace failed")
        return False

    def _dispatch_dca_buy(self, price: float, disc: float, *, in_backoff: bool) -> None:
        """Evaluate DCA dip conditions, trend gate, and place BUY order."""
        dca_threshold = self.s["last_buy_price"] * (1 - self.p.dca_drop_pct / 100) if self.s["last_buy_price"] else None
        in_sl_state = bool(
            self._state_flag("sl_pending")
            or (self._now() < safe_nonneg_float(self.s.get("sl_fail_until"), 0.0))
            or (self.ticker in self.exit_manager.sl_bounce_state if self.exit_manager else False)
        )
        if not (
            not in_backoff
            and not in_sl_state
            and self.s["dca_buys"] < self.p.max_dca_buys
            and dca_threshold
            and (price <= dca_threshold or are_close(price, dca_threshold, self.p.reentry_tolerance_pct))
            and self.s["spent_cash"] + self.p.dca_amount <= self.p.max_budget
            and not self._has_open("BUY")
        ):
            return
        if self.p.dca_trend_gate_pct > 0:
            if self._now() < getattr(self, "_dca_gate_until", 0):
                return
            slope_f = safe_float(self._trend_slope_provider(self.yahoo_sym), default=None)
            if slope_f is not None and slope_f < -self.p.dca_trend_gate_pct:
                self._dca_gate_until = self._now() + 300
                log(
                    f"  [STRAT] DCA BLOCKED by trend: slope {slope_f:+.3f}%/bar "
                    f"< -{self.p.dca_trend_gate_pct}% (downtrend) — re-checking in 5 min"
                )
                return
        log(f"  [STRAT] dip: {price:.2f} <= {self.s['last_buy_price']:.2f}×(1-{self.p.dca_drop_pct}%) — DCA")
        self._place_buy(self.p.dca_amount, price * disc, kind="DCA")

    def _step_dispatch_orders(
        self, price: float, held: float, disc: float, *, in_backoff: bool
    ) -> None:
        """Stage 4: Evaluate take-profit (MDHTP, ladder, or single TP) and DCA order dispatch."""
        if self._dispatch_take_profit(price, self._avg(), held):
            return
        self._dispatch_dca_buy(price, disc, in_backoff=in_backoff)

    def step(self, price: float) -> None:
        sync_ctx = self._step_sync_market(price)
        if sync_ctx is None:
            return
        price, held, disc, in_backoff = sync_ctx

        is_flat = held <= 1e-9 or (self._is_dust(held, price) and not self._has_open("SELL"))
        if is_flat:
            self._step_evaluate_reentry(price, disc, in_backoff=in_backoff)
            return

        if self._step_evaluate_exits(price):
            return

        self._step_dispatch_orders(price, held, disc, in_backoff=in_backoff)

    # -- Main Loop -------------------------------------------------------------
    def run(self, stop: threading.Event | None = None) -> None:
        mode, exec_banner = self._resolve_run_mode_and_banner(
            real_banner="⚠ REAL — REAL MONEY",
            ai_banner="[OBSERVE ONLY — AI MANAGED]",
        )
        log("  === STRATEGY STARTED ===")
        log(f"      instrument : {self.ticker}  (price through {self.yahoo_sym})")
        log(f"      mode       : {mode}   {exec_banner}")
        log(f"      entry      : {self.p.entry_amount:.0f} {self.ccy} @ market-{self.p.entry_discount_pct}%")
        log(f"      DCA        : {self.p.dca_amount:.0f} {self.ccy} at every -{self.p.dca_drop_pct}% (max {self.p.max_dca_buys})")
        if self.p.enable_takeprofit:
            log(f"      take-profit: +{self.p.takeprofit_pct}% above the average price")
        else:
            log("      take-profit: disabled (dca_only)")
        log(f"      BUDGET CAP : {self.p.max_budget:.0f} {self.ccy} / cycle")
        log(f"      check      : every {self.p.check_minutes:g} min   |  1 {self.ccy} = {self.fx_to_usd:.4f} USD")
        log(f"      ! break-even threshold ~{self.p.fx_fee_pct*2:.2f}% (FX) plus spread; TP={self.p.takeprofit_pct}%")

        def _log_status(price: float) -> None:
            avg = self._avg()
            net = self.s.get("realized_net_usd", 0.0)
            fees = self.s.get("fees_usd", 0.0)
            if avg:
                log(
                    f"  [STRAT] price={price:.2f}  qty={self.s['qty']:.2f}  avg={avg:.2f}  "
                    f"deployed={self.s['spent_cash']:.0f}{self.ccy}  "
                    f"NET={net:+.2f}USD (gross {self.s['realized_pnl_usd']:+.2f}, fee {fees:.2f})  "
                    f"ord={len(self.s['orders'])}"
                )
            else:
                log(
                    f"  [STRAT] price={price:.2f}  qty=0  "
                    f"NET={net:+.2f}USD (gross {self.s['realized_pnl_usd']:+.2f}, fee {fees:.2f})  "
                    f"(waiting for an entry)"
                )

        self._run_strategy_loop(
            price_fn=lambda: get_price_usd(self.yahoo_sym),
            status_log_fn=_log_status,
            stop=stop,
            reload_external=True,
        )
