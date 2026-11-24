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
    ensure_local_module_priority,
    first_pos_float,
    is_market_order,
    log as _log,
    normalize_epoch_sec,
    required_bool_env,
    required_env,
    required_float_env,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
)
ensure_local_module_priority(_HERE, "ipo_common", "market_data", "t212_broker_lifecycle", "t212_client")

from typing import Any  # noqa: E402
from cachepaths import canonical_base_asset  # noqa: E402
from pricewindow import vol_pct_to_ratio  # noqa: E402
from ipo_common import notify  # noqa: E402
from market_data import get_eur_usd, get_usd_ron, get_price_usd, trend_slope_pct  # noqa: E402
from t212_broker_lifecycle import (  # noqa: E402,F401
    T212BrokerLifecycleMixin,
    _new_state,
    _sanitize_t212_state,
)
from t212_client import T212Client  # noqa: E402
from providers.execution_audit import AuditedStrategyExecutor, ExecutionAudit  # noqa: E402
from providers.t212_provider import T212Provider  # noqa: E402
from strategies.base_spot_mixin import (  # noqa: E402
    BaseSpotMixin,
    SpotParamsMixin,
    adapt_ladder_rungs,
    generate_tp_ladder,
    load_common_core_strat_env,
    load_common_mhdpa_env,
    parse_max_dca_buys_env,
    parse_tp_ladder_env,
)
from strategies.state_store import (  # noqa: E402
    state_path_for as _shared_state_path_for,
)
from strategies.position_exit_manager import (  # noqa: E402
    PositionExitManager, ExitAction,
)


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
    fx_fee_pct: float = 0.0
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


class Strategy(T212BrokerLifecycleMixin):
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

    def _close_cycle(
        self, exit_price: float, *, keep_pending_submit: bool = False, exit_kind: str | None = None
    ) -> None:
        self._cancel_all_orders()
        saved_orders = list(self.s.get("orders") or [])
        saved_pending_submit = self.s.get("pending_submit") if keep_pending_submit else None
        norm_exit_kind = str(exit_kind or "").strip().upper()
        was_sl = bool(
            self._state_flag("sl_pending")
            or self._state_flag("sl_pending_exit")
            or norm_exit_kind in {"SL", "TR", "STOP"}
        )
        nxt = self._reset_cycle_state(
            ("realized_pnl_usd", "realized_net_usd", "fees_usd"),
            orders=saved_orders,
            **({"pending_submit": saved_pending_submit} if keep_pending_submit else {}),
        )
        exit_px = first_pos_float(exit_price) or safe_nonneg_float(exit_price, 0.0) or 0.0
        seed_kind = "STOP" if was_sl else ("REDUCE" if norm_exit_kind == "REDUCE" else "TP")
        self._seed_post_exit_anchors(
            exit_px, seed_kind, self._now(), include_sl_low=False
        )
        if was_sl and self.p.sl_rebuy_enabled:
            self.s["sl_rebuy"] = {"low": exit_px, "sell_price": exit_px}
            log(f"  \U0001f7e2 [STRAT] pullback re-buy ARMED after the stop-loss "
                f"(expecting +{self.p.sl_rebuy_bounce_pct}% from the low)")
        log(f"  [STRAT] === cycle closed, restarting (cycle {nxt}) ===")

    def _local_vol_1h_sources(self):
        return ((self._resolve_shadow_vol_1h_pct, "shadow"),)

    def _reentry_pullback_labels(self, _sp_len: int) -> tuple[str, str]:
        return "fixed (fallback)", "vol"

    def _exit_config_overrides(self) -> dict[str, Any]:
        p = self.p
        return dict(
            reentry_sl_bounce_pct=(
                vol_pct_to_ratio(p.sl_rebuy_bounce_pct)
                if p.sl_rebuy_enabled
                else (0.015 if p.sl_mhdsl_enabled else 0.0)
            ),
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
                    log(f"  [STRAT] re-buy SL cancelled \u2014 budget cap {self.p.max_budget:.0f} {self.ccy} reached")
                elif self._place_buy(self.p.entry_amount, price * (1 - self.p.entry_discount_pct / 100), kind="ENTRY"):
                    self.s.pop("sl_rebuy", None)
                    self.s["last_exit_kind"] = "TP"
                    if self.exit_manager is not None:
                        self.exit_manager.reentry_state.pop(self.ticker, None)
                    log(f"  \U0001f7e2 [STRAT] RE-BUY after the SL: a +{self.p.sl_rebuy_bounce_pct}% pullback from the low {rb['low']:.2f} \u2014 re-entering ENTRY")
                    notify(
                        title=f"\U0001f7e2 {self.yahoo_sym} RE-BUY after the stop-loss",
                        body=(f"pullback +{self.p.sl_rebuy_bounce_pct}% from low {rb['low']:.2f} \u2014 "
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
        mgr.config.reentry_sl_bounce_pct = (
            vol_pct_to_ratio(self.p.sl_rebuy_bounce_pct)
            if self.p.sl_rebuy_enabled
            else (0.015 if self.p.sl_mhdsl_enabled else 0.0)
        )
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

    # -- Protective Exits & Decision Step --------------------------------------
    def _ratchet_pos_peak(self, price: float, avg: float | None) -> tuple[float, bool]:
        """Initialize or ratchet ``pos_peak`` and return ``(peak, hit_new_high)``."""
        peak = safe_pos_float(self.s.get("pos_peak")) or 0.0
        if peak <= 0:
            self.s["pos_peak"] = peak = max(avg or price, price)
            return peak, False
        if price > peak:
            self.s["pos_peak"] = peak = price
            return peak, True
        return peak, False

    def _execute_protective_exit(
        self, kind: str, price: float, label: str, fail_key: str, sell_qty: float | None = None
    ) -> bool:
        """Cancel existing non-protective orders and submit a protective market sell."""
        existing = next((o for o in self.s["orders"] if o.get("kind") == kind), None)
        if existing is None or (not is_market_order(existing) and price < existing["limit"]):
            cancelled_orders = [dict(o) for o in self.s["orders"] if o.get("kind") != kind]
            if not self._cancel_and_await_exit_clearance(label, lambda o: o.get("kind") == kind):
                return False
            target_qty = sell_qty if sell_qty is not None else self._sellable_qty(self.s["qty"], price)
            if not self._place_sell(target_qty, round(price, 2), kind=kind, market=True):
                if not self.s.get("pending_submit"):
                    self.s[fail_key] = self._now() + 1800
                    log(f"  ! [STRAT] {label} sell failed or refused \u2014 backoff 30 min (retaining TP ladder)")
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
        peak, hit_new_high = self._ratchet_pos_peak(price, avg)
        if hit_new_high:
            self.s.pop("tr_fail_until", None)
            self.s["tr_alerted"] = False

        if not self._state_flag("tr_armed"):
            minp = self.p.trail_min_profit_pct
            if minp > 0 and (not avg or price < avg * (1 + minp / 100.0)):
                return False
            self.s["tr_armed"] = True
            self.s["pos_peak"] = peak = price
            log(
                f"  [STRAT] trailing ARMED: price {price:.2f} >= avg{(avg or 0):.2f}+{minp}% \u2014 "
                f"protecting from now on (-{self.p.trail_pct}% from the peak)"
            )

        drop_pct = (peak - price) / peak * 100 if peak > 0 else 0.0
        if drop_pct < self.p.trail_pct or self._now() < safe_nonneg_float(self.s.get("tr_fail_until"), 0.0):
            return False
        if self._execute_protective_exit("TR", price, "TRAILING", "tr_fail_until"):
            if not self._state_flag("tr_alerted"):
                self.s["tr_alerted"] = True
                log(f"  \U0001f4c9 [STRAT] TRAILING: -{drop_pct:.2f}% from peak {peak:.2f} >= {self.p.trail_pct}% \u2014 SELLING ALL")
                notify(
                    title=f"\U0001f4c9 TRAILING {self.yahoo_sym} -{drop_pct:.1f}%",
                    body=f"from peak {peak:.2f} \u2265{self.p.trail_pct}% \u2014 selling all",
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
            f"  \U0001f6d1 [STRAT] {label_pfx}STOP-LOSS: loss {loss_pct:.2f}% >= {self.p.stop_loss_pct}%{reason_sfx} "
            "\u2014 SELLING EVERYTHING (cutting the loss)"
        )
        notify(
            title=f"\U0001f6d1 SL {self.yahoo_sym} -{loss_pct:.1f}%",
            body=f"loss {loss_pct:.1f}% >=threshold{self.p.stop_loss_pct}%{reason_sfx} \u2014 selling everything",
            source="T212", price=price, desktop=self.desktop,
        )

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

        use_mhdsl = self.exit_manager is not None and self.p.sl_mhdsl_enabled
        if not (self._in_mhdsl_territory(loss_pct) if use_mhdsl else loss_pct >= self.p.stop_loss_pct):
            self.s["sl_alerted"] = self.s["sl_pending"] = False
            self.s.pop("sl_fail_until", None)
            return False

        if use_mhdsl:
            raw_pending_sl = self.s.get("sl_pending_exit")
            pending_sl = raw_pending_sl if isinstance(raw_pending_sl, dict) else None
            if pending_sl:
                sell_qty = self._sellable_qty(self.s.get("qty", 0.0), price)
                should_exit_sl, exit_reason, decision = True, str(pending_sl.get("reason") or "mhdsl_pending_exit"), None
            else:
                snapshot, decision = self._evaluate_mhdsl_stop_decision(price, avg)
                sell_qty = snapshot.sellable_qty
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
                log(f"  \U0001f4c9 [STRAT] {self.yahoo_sym} loss -{loss_pct:.1f}% (threshold {band*step:.0f}%)")
                notify(
                    title=f"\U0001f4c9 {self.yahoo_sym} -{loss_pct:.1f}%",
                    body=f"loss -{loss_pct:.1f}% (over {band*step:.0f}%) | q{self.s['qty']:.2f} a{avg:.2f} p{price:.2f}",
                    source="T212", price=price, desktop=self.desktop,
                )
                self.s["loss_band"] = band
        if self.observe_only:
            if held > 1e-9:
                self._ratchet_pos_peak(price, avg)
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
                    log(f"  [STRAT] MDHTP EXIT TRIGGERED ({decision.reason}) \u2014 initiating cancellation & exit sequence")

        if pending_exit:
            def _is_market_exit(o: dict) -> bool:
                return bool(
                    is_market_order(o) and str(o.get("side", "")).upper() == "SELL" and o.get("kind") in ("TP", "SL")
                )

            def _clear_exit_intent(msg: str) -> bool:
                log(msg)
                self.s["tp_pending_exit"] = None
                return True

            existing_exit = next((o for o in self.s.get("orders", []) if _is_market_exit(o)), None)
            if existing_exit:
                for o in [o for o in list(self.s["orders"]) if not _is_market_exit(o)]:
                    self._cancel_order(o)
                if self.s["qty"] <= 1e-6:
                    return _clear_exit_intent("  [STRAT] MDHTP exit: position closed \u2014 clearing exit intent")
                log(
                    f"  [STRAT] MDHTP exit: existing market exit order {existing_exit.get('id')} "
                    "in progress \u2014 awaiting fill reconciliation"
                )
                return True

            if not self._cancel_and_await_exit_clearance("MDHTP exit", _is_market_exit):
                return True

            fresh_held = self.s["qty"]
            raw_sf = safe_pos_float(pending_exit.get("sell_fraction"), None)
            sell_fraction = min(1.0, raw_sf) if raw_sf is not None else 1.0
            target_qty = fresh_held * sell_fraction if sell_fraction < 1.0 else fresh_held
            sell_qty = self._sellable_qty(target_qty, price)
            if sell_qty <= 0 or self._is_dust(sell_qty, price):
                return _clear_exit_intent("  [STRAT] MDHTP exit: no sellable quantity remaining \u2014 clearing exit intent")

            exit_px = round(price * 0.999, 2)
            if self._place_sell(sell_qty, exit_px, kind="TP", market=True):
                if pending_exit.get("action") == ExitAction.PARTIAL_SELL.value and self.exit_manager is not None:
                    self.exit_manager.record_hard_tp(self.ticker, tick_time)
                return _clear_exit_intent(
                    f"  [STRAT] MDHTP EXIT ({pending_exit.get('reason')}) -> reference {exit_px} ({sell_qty} shares)"
                )
            log(f"  ! [STRAT] MDHTP EXIT order refused/failed ({pending_exit.get('reason')}) \u2014 intent retained for retry")
            return True

        if decision is not None and (decision.reason == "not_in_tp_territory" or decision.action == ExitAction.HOLD):
            if (
                decision.reason != "not_in_tp_territory"
                and decision.action == ExitAction.HOLD
                and self.ticker in self.exit_manager.tp_ride_state
            ):
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
            if self._now() < safe_nonneg_float(getattr(self, "_dca_gate_until", 0.0), default=0.0):
                return
            slope_f = safe_float(self._trend_slope_provider(self.yahoo_sym), default=None)
            if slope_f is not None and slope_f < -self.p.dca_trend_gate_pct:
                self._dca_gate_until = self._now() + 300
                log(
                    f"  [STRAT] DCA BLOCKED by trend: slope {slope_f:+.3f}%/bar "
                    f"< -{self.p.dca_trend_gate_pct}% (downtrend) \u2014 re-checking in 5 min"
                )
                return
        log(f"  [STRAT] dip: {price:.2f} <= {self.s['last_buy_price']:.2f}\u00d7(1-{self.p.dca_drop_pct}%) \u2014 DCA")
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
            real_banner="\u26a0 REAL \u2014 REAL MONEY",
            ai_banner="[OBSERVE ONLY \u2014 AI MANAGED]",
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
