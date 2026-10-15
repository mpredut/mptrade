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
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from ipo_common import (
    log as _log, now_str, are_close, float_env, required_env, required_float_env,
    required_bool_env,
)
from ipo_notify import notify
from market_data import get_eur_usd, get_usd_ron, get_price_usd, t212_to_yahoo, trend_slope_pct
from t212_client import T212Client
from providers.execution_audit import AuditedStrategyExecutor, ExecutionAudit, new_intent_id
from providers.strategy_executor import OrderNotFound, ProviderError
from providers.t212_provider import T212Provider
from strategies.state_store import JsonStateStore
from strategies import spot_rules as sr
from strategies.position_exit_manager import (
    PositionExitManager,
    PositionExitConfig,
    PositionSnapshot,
    ExitAction,
    ReentrySnapshot,
    ReentryDecision,
)
from order_rollback import restore_cancelled_orders
from intelligence.internal.state.volatility import resolve_cached_volatility_1h, vol_pct_to_ratio

FX_FEE_PCT = 0.15  # T212 currency-conversion fee per direction.


def log(msg: str) -> None:
    """Log with the asset's thread name: t212_bot runs one thread per asset into one log."""
    name = threading.current_thread().name
    _log(msg if name == "MainThread" else f"[{name}] {msg}")
_HERE = os.path.dirname(os.path.abspath(__file__))


def state_path_for(ticker: str) -> str:
    safe = "".join(c for c in ticker if c.isalnum() or c in "._-")
    return os.path.join(_HERE, f".state_{safe}.json")


def generate_tp_ladder(
    min_pct: float,
    max_pct: float,
    steps: int = 10,
    curve: str = "linear",
) -> list[tuple[float, float]]:
    """Generate a scale-out take-profit ladder from min_pct to max_pct across `steps` rungs.

    Supported curves:
    - 'linear' / 'uniform': equal fractions across all rungs.
    - 'bell' / 'gaussian' / 'pyramid': small start, bulk in the middle, small at the end.
    - 'front_heavy': larger fractions on lower rungs for faster de-risking.
    - 'back_heavy': larger fractions on higher rungs for maximum upside capture.
    - 'geometric': geometrically increasing percentage steps.
    """
    if steps < 1:
        return []
    if steps == 1:
        return [(round(float(min_pct), 1), 1.0)]

    min_f = float(min_pct)
    max_f = float(max_pct)
    curve_lower = str(curve or "linear").lower()

    if curve_lower == "geometric":
        r_start = 1.0 + min_f / 100.0
        r_end = 1.0 + max_f / 100.0
        ratio = (r_end / r_start) ** (1.0 / (steps - 1)) if steps > 1 else 1.0
        levels = [round(((r_start * (ratio ** i)) - 1.0) * 100.0, 1) for i in range(steps)]
    else:
        step_size = (max_f - min_f) / (steps - 1)
        levels = [round(min_f + i * step_size, 1) for i in range(steps)]

    if curve_lower in ("bell", "gaussian", "normal", "pyramid"):
        center = (steps - 1) / 2.0
        sigma = max(0.5, (steps - 1) / 2.8) if steps > 1 else 1.0
        weights = [math.exp(-0.5 * ((i - center) / sigma) ** 2) for i in range(steps)]
    elif curve_lower == "front_heavy":
        weights = [float(steps - i) for i in range(steps)]
    elif curve_lower == "back_heavy":
        weights = [float(i + 1) for i in range(steps)]
    else:
        weights = [1.0 for _ in range(steps)]

    total_w = sum(weights)
    return [(lvl, round(w / total_w, 4)) for lvl, w in zip(levels, weights)]


def adapt_ladder_rungs(
    remaining: list[tuple[float, float]],
    held: float,
    min_tranche_qty: float = 0.01,
) -> list[tuple[float, float]]:
    """Dynamically clamp remaining rungs so each order has viable trading quantity.

    If held quantity is too small to distribute across all rungs (e.g. 0.05 shares for 10 rungs),
    subsamples rungs evenly across the remaining min-to-max span so every order is valid.
    """
    if not remaining or held <= 1e-9:
        return []
    max_viable_rungs = max(1, int(held / max(1e-6, min_tranche_qty)))
    if len(remaining) <= max_viable_rungs:
        return list(remaining)
    if max_viable_rungs == 1:
        return [remaining[-1]]
    indices = [
        int(round(i * (len(remaining) - 1) / (max_viable_rungs - 1)))
        for i in range(max_viable_rungs)
    ]
    selected_indices = sorted(set(indices))
    return [remaining[idx] for idx in selected_indices]


@dataclass
class StratParams:
    currency: str            # Currency used by the monetary settings below.
    entry_amount: float      # Initial purchase size in that currency.
    entry_discount_pct: float
    dca_amount: float        # Size of one dip purchase in configured currency.
    dca_drop_pct: float
    check_minutes: float
    takeprofit_pct: float
    max_budget: float        # Per-cycle total cap in configured currency.
    max_dca_buys: int
    validity: str
    enable_takeprofit: bool
    order_ttl_min: float
    stop_loss_pct: float     # Sell all at this loss percentage; zero disables it.
    yahoo_sym: str = ""             # Configured Yahoo symbol or one derived from ticker.
    reentry_drop_pct: float = 0.0   # After TP, reenter only below the sale price by X percent.
    reentry_tolerance_pct: float = 0.0  # Near-threshold tolerance for reentry and DCA.
    tp_ladder: list = field(default_factory=list)   # Scale-out level/fraction pairs.
    fx_fee_pct: float = FX_FEE_PCT   # T212 FX fee per direction.
    loss_alert_step: float = 1.0     # Alert for each further unrealized-loss band.
    ladder_min_free: float = 6.0     # Minimum dollar amount left unreserved by TP ladder.
    sl_rebuy_enabled: bool = False   # Rebuy on a bounce from the post-stop low.
    sl_rebuy_bounce_pct: float = 1.2 # Bounce confirming the fall has stopped.
    dca_trend_gate_pct: float = 0.0  # Skip DCA below this confirmed downtrend slope; zero disables.
    trail_pct: float = 0.0           # Sell all after this drop from position peak; zero disables.
    trail_min_profit_pct: float = 5.0  # Keep trailing inactive until price gains this much over average.
    # --- HYBRID RE-ENTRY (EXPERIMENTAL, default OFF) -----------------------------
    reentry_hybrid_enabled: bool = False  # Decouple macro trend permission from micro pullback trigger.
    reentry_peak_relative: bool = False   # Force peak-relative pullback without trend filter (Test 1).
    reentry_pullback_pct: float = 1.5     # Pullback from post-sale peak required in confirmed bull trend.
    reentry_pullback_adaptive: bool = False  # Scale pullback by volatility: clamp(K * vol, min, max)
    reentry_pullback_k: float = 1.0       # Multiplier for volatility-adaptive pullback
    reentry_pullback_min: float = 0.8     # Minimum pullback clamp (%)
    reentry_pullback_max: float = 4.0     # Maximum pullback clamp (%)
    reentry_ttl_hours: float = 0.0        # 0 = off. After this many hours, stale sale barrier expires.
    # --- MULTI-HORIZON DYNAMIC STOP-LOSS (MHDSL) & RE-ENTRY MIRROR ---
    sl_mhdsl_enabled: bool = False
    sl_disaster_buffer_ratio: float = 0.30
    sl_trend_defer_max_sec: float = 220.0
    sl_bounce_min_recovery_pct: float = 0.40
    sl_bounce_pullback_pct: float = 0.30
    sl_structural_break_enabled: bool = True
    # --- MULTI-HORIZON DYNAMIC TAKE-PROFIT (MDHTP) ---
    tp_mdhtp_enabled: bool = False
    tp_stagnation_max_sec: float = 1080.0

    @classmethod
    def from_env(cls, env: dict | None = None) -> "StratParams":
        """Read the environment or an explicit mapping for collision-free multi-asset use."""
        e = os.environ if env is None else env
        mode = required_env("STRATEGY_MODE", e).lower()
        if mode not in {"avg_tp", "dca_only"}:
            raise ValueError(f"Invalid STRATEGY_MODE: {mode!r}")
        # Parse scale-out ladder percentages and fractions (explicit list or dynamic range).
        _ladder = []
        raw_ladder = (e.get("STRAT_TP_LADDER") or "").strip()
        ladder_curve = (e.get("STRAT_TP_LADDER_CURVE") or "linear").strip().lower()
        if raw_ladder:
            clean_raw = raw_ladder.removeprefix("range:")
            if clean_raw.count(":") == 2 and "," not in clean_raw:
                try:
                    p_min, p_max, p_steps = clean_raw.split(":")
                    _ladder = generate_tp_ladder(float(p_min), float(p_max), int(p_steps), curve=ladder_curve)
                except ValueError:
                    _ladder = []
            else:
                for _part in raw_ladder.split(","):
                    _part = _part.strip()
                    if ":" in _part:
                        _lvl, _frac = _part.split(":", 1)
                        try:
                            _ladder.append((float(_lvl), float(_frac) / 100.0))
                        except ValueError:
                            pass
        elif e.get("STRAT_TP_LADDER_MIN") and e.get("STRAT_TP_LADDER_MAX"):
            try:
                p_min = float(e["STRAT_TP_LADDER_MIN"])
                p_max = float(e["STRAT_TP_LADDER_MAX"])
                p_steps = int(e.get("STRAT_TP_LADDER_STEPS", 10))
                _ladder = generate_tp_ladder(p_min, p_max, p_steps, curve=ladder_curve)
            except (ValueError, TypeError):
                _ladder = []
        _budget = required_float_env("STRAT_MAX_BUDGET", e)
        # Percentage sizing is the single supported policy; no legacy hidden amounts.
        _entry_pct = required_float_env("STRAT_ENTRY_PCT", e)
        _entry = _budget * _entry_pct / 100.0
        _dca_pct = required_float_env("STRAT_DCA_PCT", e)
        _dca = _budget * _dca_pct / 100.0
        # Auto DCA count derives from budget; otherwise use an explicit value or ten.
        _mdb = (e.get("STRAT_MAX_DCA_BUYS") or "").strip().lower()
        if _mdb in ("auto", "buget", "budget"):
            _max_dca = max(0, int((_budget - _entry) // _dca)) if _dca > 0 else 0
            _entry = _budget - _max_dca * _dca   # Entry absorbs remainder to cover the full budget.
        elif _mdb:
            _max_dca = int(float(_mdb))
        else:
            raise ValueError("Missing or empty mandatory setting: STRAT_MAX_DCA_BUYS")
        return cls(
            currency           = required_env("STRAT_CURRENCY", e).upper(),
            entry_amount       = _entry,
            entry_discount_pct = required_float_env("STRAT_ENTRY_DISCOUNT_PCT", e),
            dca_amount         = _dca,
            dca_drop_pct       = required_float_env("STRAT_DCA_DROP_PCT", e),
            check_minutes      = required_float_env("STRAT_CHECK_MINUTES", e),
            takeprofit_pct     = required_float_env("STRAT_TAKEPROFIT_PCT", e),
            max_budget         = _budget,
            max_dca_buys       = _max_dca,
            validity           = "GOOD_TILL_CANCEL",
            enable_takeprofit  = (mode != "dca_only"),
            order_ttl_min      = required_float_env("STRAT_ORDER_TTL_MIN", e),
            stop_loss_pct      = required_float_env("STRAT_STOP_LOSS_PCT", e),
            yahoo_sym          = required_env("YAHOO_SYMBOL", e),
            reentry_drop_pct   = required_float_env("STRAT_REENTRY_DROP_PCT", e),
            reentry_tolerance_pct = required_float_env("STRAT_REENTRY_TOLERANCE_PCT", e),
            tp_ladder          = _ladder,
            fx_fee_pct         = required_float_env("STRAT_FX_FEE_PCT", e),
            loss_alert_step    = required_float_env("STRAT_LOSS_ALERT_STEP", e),
            ladder_min_free    = required_float_env("STRAT_LADDER_MIN_FREE", e),
            sl_rebuy_enabled   = required_bool_env("STRAT_SL_REBUY_ENABLED", e),
            sl_rebuy_bounce_pct= required_float_env("STRAT_SL_REBUY_BOUNCE_PCT", e),
            dca_trend_gate_pct = required_float_env("STRAT_DCA_TREND_GATE_PCT", e),
            trail_pct          = required_float_env("STRAT_TRAIL_PCT", e),
            trail_min_profit_pct = required_float_env("STRAT_TRAIL_MIN_PROFIT_PCT", e),
            reentry_hybrid_enabled = (
                str(e.get("STRAT_REENTRY_HYBRID_ENABLED", "")).lower() in ("true", "1")
            ),
            reentry_peak_relative = (
                str(e.get("STRAT_REENTRY_PEAK_RELATIVE", "")).lower() in ("true", "1")
            ),
            reentry_pullback_pct = (
                float_env("STRAT_REENTRY_PULLBACK_PCT", e)
                if float_env("STRAT_REENTRY_PULLBACK_PCT", e) is not None else 1.5
            ),
            reentry_pullback_adaptive = (
                str(e.get("STRAT_REENTRY_PULLBACK_ADAPTIVE", "")).lower() in ("true", "1")
            ),
            reentry_pullback_k = (
                float_env("STRAT_REENTRY_PULLBACK_K", e)
                if float_env("STRAT_REENTRY_PULLBACK_K", e) is not None else 1.0
            ),
            reentry_pullback_min = (
                float_env("STRAT_REENTRY_PULLBACK_MIN", e)
                if float_env("STRAT_REENTRY_PULLBACK_MIN", e) is not None else 0.8
            ),
            reentry_pullback_max = (
                float_env("STRAT_REENTRY_PULLBACK_MAX", e)
                if float_env("STRAT_REENTRY_PULLBACK_MAX", e) is not None else 4.0
            ),
            reentry_ttl_hours = (
                float_env("STRAT_REENTRY_TTL_HOURS", e)
                if float_env("STRAT_REENTRY_TTL_HOURS", e) is not None else 0.0
            ),
            sl_mhdsl_enabled = (
                str(e.get("STRAT_SL_MHDSL_ENABLED", "")).lower() in ("true", "1")
            ),
            sl_disaster_buffer_ratio = (
                float_env("STRAT_SL_DISASTER_BUFFER_RATIO", e)
                if float_env("STRAT_SL_DISASTER_BUFFER_RATIO", e) is not None else 0.30
            ),
            sl_trend_defer_max_sec = (
                float_env("STRAT_SL_TREND_DEFER_MAX_SEC", e)
                if float_env("STRAT_SL_TREND_DEFER_MAX_SEC", e) is not None else 220.0
            ),
            sl_bounce_min_recovery_pct = (
                float_env("STRAT_SL_BOUNCE_MIN_RECOVERY_PCT", e)
                if float_env("STRAT_SL_BOUNCE_MIN_RECOVERY_PCT", e) is not None else 0.40
            ),
            sl_bounce_pullback_pct = (
                float_env("STRAT_SL_BOUNCE_PULLBACK_PCT", e)
                if float_env("STRAT_SL_BOUNCE_PULLBACK_PCT", e) is not None else 0.30
            ),
            sl_structural_break_enabled = (
                str(e.get("STRAT_SL_STRUCTURAL_BREAK_ENABLED", "true")).lower() in ("true", "1")
            ),
            tp_mdhtp_enabled = (
                str(e.get("STRAT_TP_MDHTP_ENABLED", "")).lower() in ("true", "1")
            ),
            tp_stagnation_max_sec = (
                float_env("STRAT_TP_STAGNATION_MAX_SEC", e)
                if float_env("STRAT_TP_STAGNATION_MAX_SEC", e) is not None else 1080.0
            ),
        )


def _new_state() -> dict:
    return {
        "cycle": 1,
        "qty": 0.0,
        "cost_usd": 0.0,        # USD cost basis of held quantity.
        "spent_cash": 0.0,      # Current-cycle deployed cash for cap enforcement.
        "dca_buys": 0,
        "entry_price": None,
        "last_buy_price": None,
        "last_sell_price": None,
        "last_sell_ts": None,
        "post_sell_peak": None,
        "post_sell_trough": None,
        "last_exit_kind": None,
        "sl_bounce_state": {},
        "tp_ride_state": {},
        "reentry_state": {},
        "realized_pnl_usd": 0.0,   # Cumulative gross profit.
        "realized_net_usd": 0.0,   # Cumulative net profit after FX fees.
        "fees_usd": 0.0,           # Total paid FX fees.
        "loss_band": 0,         # Deepest alerted loss band for anti-spam.
        "tp_sold_levels": [],   # Ladder levels sold in the current cycle.
        "orders": [],           # {id, side, qty, limit, amount, kind, ts, level}
        # Submit intent persisted before the non-idempotent T212 POST.  T212 has no
        # client-order-id, so recovery correlates active orders plus portfolio delta.
        "pending_submit": None,
        "pos_peak": 0.0,        # Highest price while holding, for trailing crash breaker.
        "tr_alerted": False,    # Current-episode trailing notification anti-spam.
        "tr_armed": False,      # Arm only after the configured gain over average.
        "tp_pending_exit": None, # Persistent MDHTP exit intent awaiting execution.
    }


def _sell_pnl(avg: float, price: float, qty: float, fee_pct: float = FX_FEE_PCT) -> tuple[float, float, float]:
    """Return gross, round-trip FX fee, and net for selling quantity at price."""
    gross = (price - avg) * qty
    fee = (fee_pct / 100.0) * (avg * qty + price * qty)
    return gross, fee, gross - fee


class Strategy:
    def __init__(self, client: T212Client, ticker: str, params: StratParams,
                 dry_run: bool = True, desktop: bool = False,
                 initial_state: dict | None = None,
                 fx_to_usd: float | None = None,
                 clock: Callable[[], float] | None = None,
                 trend_slope_provider: Callable[[str], float | None] | None = None,
                 execution_audit: ExecutionAudit | None = None):
        self.client = client
        self.ticker = ticker
        self.yahoo_sym = params.yahoo_sym or t212_to_yahoo(ticker)
        self.p = params
        self.dry_run = dry_run
        self.desktop = desktop
        self.ccy = params.currency
        self._clock = clock or time.time
        self._trend_slope_provider = trend_slope_provider or trend_slope_pct
        self.execution_audit = execution_audit or ExecutionAudit()
        # Financial strategy remains separate while submission/status/cancellation use
        # the same strict audited contract as spot_dca.
        self.executor = AuditedStrategyExecutor(
            T212Provider(
                client=client, live_enabled=not dry_run,
                order_validity=params.validity,
            ),
            audit=self.execution_audit, venue="T212",
        )
        self.fx_to_usd = self._fx_to_usd(params.currency) if fx_to_usd is None else fx_to_usd
        self.state_file = state_path_for(ticker)
        self._state_write_failed = False
        self.s = initial_state if initial_state is not None else self._load()
        self.s.setdefault("pending_submit", None)
        self.s.setdefault("post_sell_trough", None)
        self.s.setdefault("last_exit_kind", None)
        self.s.setdefault("sl_bounce_state", {})
        self.s.setdefault("tp_ride_state", {})
        self.s.setdefault("reentry_state", {})
        self.s.setdefault("tp_pending_exit", None)
        self.exit_manager: PositionExitManager | None = None
        if self.p.sl_mhdsl_enabled or self.p.tp_mdhtp_enabled:
            exit_cfg = PositionExitConfig(
                lost_threshold=self.p.stop_loss_pct / 100.0,
                sl_disaster_buffer_ratio=self.p.sl_disaster_buffer_ratio,
                sl_trend_defer_max_sec=self.p.sl_trend_defer_max_sec,
                sl_bounce_min_recovery_pct=self.p.sl_bounce_min_recovery_pct / 100.0,
                sl_bounce_pullback_pct=self.p.sl_bounce_pullback_pct / 100.0,
                sl_structural_break_enabled=self.p.sl_structural_break_enabled,
                smart_sl_enabled=False,
                reentry_enabled=True,
                reentry_sl_bounce_pct=self.p.sl_rebuy_bounce_pct / 100.0 if self.p.sl_rebuy_enabled else 0.015,
                reentry_drop_pct=self.p.reentry_drop_pct / 100.0,
                reentry_pullback_pct=self.p.reentry_pullback_pct / 100.0,
                reentry_ttl_hours=self.p.reentry_ttl_hours,
                reentry_tolerance_pct=self.p.reentry_tolerance_pct,
                gain_threshold=self.p.takeprofit_pct / 100.0,
                tp_trend_ride_enabled=self.p.tp_mdhtp_enabled,
                tp_profit_floor_min_pct=(self.p.trail_min_profit_pct / 100.0) if self.p.trail_min_profit_pct > 0 else 0.0,
                tp_pullback_pct=(self.p.trail_pct / 100.0) if self.p.trail_pct > 0 else 0.015,
                tp_stagnation_max_sec=self.p.tp_stagnation_max_sec,
            )
            self.exit_manager = PositionExitManager(config=exit_cfg)
            self.exit_manager.sl_bounce_state = self.s["sl_bounce_state"]
            self.exit_manager.tp_ride_state = self.s["tp_ride_state"]
            self.exit_manager.reentry_state = self.s["reentry_state"]
        self._paper_seq = 0

    def _now(self) -> float:
        return float(self._clock())

    # -- valuta ----------------------------------------------------------------
    def _fx_to_usd(self, currency: str) -> float:
        """How many USD in one unit of the given currency — generic for any Yahoo currency."""
        if currency == "USD":
            return 1.0
        if currency == "EUR":
            return get_eur_usd()          # USD per EUR
        if currency == "RON":
            return 1.0 / get_usd_ron()    # USD per RON = 1 / (RON per USD)
        rate = get_price_usd(f"{currency}USD=X")   # generic: GBP, CHF, PLN...
        if rate:
            return rate
        log(f"  ! curs {currency}/USD indisponibil — tratez sumele ca USD (1:1). "
            f"Check STRAT_CURRENCY!")
        return 1.0

    # -- persistenta -----------------------------------------------------------
    def _store(self) -> JsonStateStore:
        return JsonStateStore(
            self.state_file, _new_state, label="T212",
            logger=log, fail_closed=not self.dry_run,
        )

    def _load(self) -> dict:
        return self._store().load()

    def _save(self) -> None:
        self._state_write_failed = True
        if self._store().save(self.s):
            self._state_write_failed = False

    # -- helperi ---------------------------------------------------------------
    def _avg_cost(self) -> float | None:
        return self.s["cost_usd"] / self.s["qty"] if self.s["qty"] > 1e-9 else None

    def _qty_for_amount(self, amount: float, price: float) -> float:
        usd = amount * self.fx_to_usd
        return round(usd / price, 2) if price > 0 else 0.0

    def _has_open(self, side: str) -> bool:
        side_u = str(side).upper()
        pending = self.s.get("pending_submit") or {}
        return (
            str(pending.get("side") or "").upper() == side_u
            or any(str(o["side"]).upper() == side_u for o in self.s["orders"])
        )

    def _find_open(self, side: str) -> dict | None:
        for o in self.s["orders"]:
            if o["side"] == side:
                return o
        return None

    @staticmethod
    def _definitive_submit_rejection(exc: Exception) -> bool:
        """Return true only for an explicit non-retryable T212 HTTP response."""
        message = str(exc)
        match = re.search(r"T212 HTTP\s+(\d{3})", message, flags=re.IGNORECASE)
        if match is None:
            return False
        status = int(match.group(1))
        if 400 <= status < 500 and status not in {408, 425, 429}:
            return True
        # A structured response that explicitly says the request was rejected is
        # terminal even when the gateway reports 5xx.  A bare 5xx/timeout remains
        # ambiguous because the non-idempotent POST may have reached the venue.
        return "rejected" in message.lower()

    def _new_pending_submit(self, *, side: str, qty: float, limit: float,
                            amount: float | None, kind: str,
                            level: float | None, market: bool) -> dict:
        intent_id = new_intent_id("T212", self.ticker, kind)
        return {
            "intent_id": intent_id,
            "side": str(side).upper(),
            "qty": float(qty),
            "limit": round(float(limit), 2),
            "amount": amount,
            "kind": kind,
            "level": level,
            "market": bool(market),
            "ts": self._now(),
            "before_qty": float(self.s.get("qty") or 0.0),
            "lookup_misses": 0,
        }

    def _persist_pending_submit(self, pending: dict | None) -> None:
        self.s["pending_submit"] = None if pending is None else dict(pending)
        self._save()

    def _adopt_pending_submit(self, order_id: str) -> dict:
        pending = dict(self.s.get("pending_submit") or {})
        if not pending:
            raise RuntimeError("there is no pending T212 intent to adopt")
        order = {
            "id": str(order_id),
            "side": pending["side"],
            "qty": float(pending["qty"]),
            "limit": float(pending["limit"]),
            "kind": pending.get("kind"),
            "level": pending.get("level"),
            "intent_id": pending.get("intent_id"),
            "market": bool(pending.get("market")),
            "ts": float(pending.get("ts") or self._now()),
        }
        if pending.get("amount") is not None:
            order["amount"] = pending["amount"]
        self.s["orders"].append(order)
        self.s["pending_submit"] = None
        self._save()
        return order

    def _submit_pending_order(self, pending: dict):
        """Persist once, submit through the common typed contract, and adopt acceptance."""
        self._persist_pending_submit(pending)
        market = bool(pending.get("market"))
        limit = float(pending["limit"])
        outcome = self.executor.submit_outcome_with_intent(
            pending["intent_id"], self.ticker, pending["side"].lower(),
            float(pending["qty"]), None if market else limit,
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
        instrument = order.get("instrument")
        nested_ticker = instrument.get("ticker") if isinstance(instrument, dict) else None
        ticker = str(order.get("ticker") or nested_ticker or "").upper()
        side = str(order.get("side") or "").upper()
        raw_qty = order.get("quantity", order.get("qty"))
        try:
            signed_qty = float(raw_qty)
            qty = abs(signed_qty)
            if not side:
                side = "BUY" if signed_qty > 0 else "SELL" if signed_qty < 0 else ""
        except (TypeError, ValueError, OverflowError):
            qty = None
        raw_limit = order.get("limitPrice", order.get("limit"))
        try:
            limit = float(raw_limit) if raw_limit is not None else None
        except (TypeError, ValueError, OverflowError):
            limit = None
        return ticker, side, qty, limit

    def _active_matches_pending(self, order: dict, pending: dict) -> bool:
        ticker, side, qty, limit = self._active_order_identity(order)
        if ticker != self.ticker.upper() or side != str(pending.get("side") or "").upper():
            return False
        expected_qty = float(pending.get("qty") or 0.0)
        if qty is None or abs(qty - expected_qty) > max(0.011, expected_qty * 1e-6):
            return False
        if pending.get("market"):
            return True
        expected_limit = float(pending.get("limit") or 0.0)
        return limit is not None and abs(limit - expected_limit) <= 0.011

    def _recover_pending_submit(self, real_qty: float,
                                active_orders: list[dict] | None) -> str:
        """Recover one response-lost T212 submit without blocking the loop.

        Outcomes are ``none``, ``adopted``, ``filled``, ``waiting`` or
        ``retryable``.  A portfolio delta is independent evidence of execution;
        otherwise two fresh active-order snapshots with no unique match release the
        intent for strategy revalidation and a later at-least-once submit.
        """
        pending = self.s.get("pending_submit")
        if not isinstance(pending, dict) or not pending:
            return "none"
        if active_orders is None:
            return "waiting"
        matches = [
            order for order in active_orders
            if isinstance(order, dict) and self._active_matches_pending(order, pending)
        ]
        if len(matches) == 1 and matches[0].get("id") is not None:
            order = self._adopt_pending_submit(str(matches[0]["id"]))
            log(f"  [STRAT] T212 submit recovered from the active orders: {order['id']}")
            return "adopted"
        if len(matches) > 1:
            log("  ! [STRAT] ambiguous T212 submit: several active orders match — keeping it pending")
            return "waiting"

        before = float(pending.get("before_qty") or 0.0)
        side = str(pending.get("side") or "").upper()
        delta = float(real_qty) - before
        if (side == "BUY" and delta > 1e-6) or (side == "SELL" and delta < -1e-6):
            pending["portfolio_fill_observed"] = True
            self._persist_pending_submit(pending)
            return "filled"

        misses = int(pending.get("lookup_misses") or 0) + 1
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

    # -- aplicare fill ---------------------------------------------------------
    def _apply_fill(self, order: dict, qty: float, price: float) -> None:
        tag = "[PAPER] " if self.dry_run else ""
        if order["side"] == "BUY":
            self.s["qty"] += qty
            self.s["cost_usd"] += qty * price
            self.s["last_buy_price"] = price
            if self.s["entry_price"] is None:
                self.s["entry_price"] = price
            self.s["spent_cash"] += order.get("amount", 0.0)
            if order.get("kind") == "DCA":
                self.s["dca_buys"] += 1
            avg = self._avg_cost()
            log(f"  [STRAT] {tag}BUY FILLED {qty} @ {price:.2f} USD ({order.get('kind')})  "
                f"qty_total={self.s['qty']:.2f} avg={avg:.2f}")
            notify(title=f"{tag}{self.yahoo_sym} BUY {qty}@{price:.2f}",
                   body=(f"{order.get('kind')} | q{self.s['qty']:.2f} a{avg:.2f} | "
                         f"desf{self.s['spent_cash']:.0f}{self.ccy} DCA{self.s['dca_buys']}/{self.p.max_dca_buys}"),
                   source="T212", price=price, desktop=self.desktop)
            self._cancel_open("SELL")     # avg s-a schimbat -> reasezam TP
        else:  # SELL
            qty = min(qty, self.s["qty"])
            avg = self._avg_cost() or price
            gross, fee, net = _sell_pnl(avg, price, qty, self.p.fx_fee_pct)
            self.s["realized_pnl_usd"] += gross
            self.s["realized_net_usd"] += net
            self.s["fees_usd"] += fee
            self.s["qty"] -= qty
            # Cost basis belongs to remaining quantity; reduce it after partial scale-out
            # to avoid inflating the residual position's average.
            self.s["cost_usd"] = max(0.0, self.s["cost_usd"] - avg * qty)
            log(f"  [STRAT] {tag}SELL FILLED {qty} @ {price:.2f} USD  "
                f"brut={gross:+.2f}  fee={fee:.2f}  net={net:+.2f} USD")
            notify(title=f"{tag}{self.yahoo_sym} SELL {qty}@{price:.2f} N{net:+.2f}$",
                   body=(f"a{avg:.2f} · br{gross:+.2f} fee{fee:.2f} N{net:+.2f}$ | "
                         f"Ntot{self.s['realized_net_usd']:+.2f}$ | ciclu{self.s['cycle']} inchis"),
                   source="T212", price=price, desktop=self.desktop)
            ladder_levels = {lvl for (lvl, _) in self.p.tp_ladder} if self.p.tp_ladder else set()
            all_tp_sold = bool(ladder_levels and ladder_levels.issubset(set(self.s.get("tp_sold_levels", []))))
            is_flat = self.s["qty"] <= 1e-9
            is_dust_exit = (
                self._is_dust(self.s["qty"], price)
                and (all_tp_sold or self.s.get("sl_pending") or not self.p.tp_ladder)
            )
            if is_flat or is_dust_exit:
                was_sl = bool(self.s.get("sl_pending"))
                pnl, net_tot, fees = (self.s["realized_pnl_usd"],
                                      self.s["realized_net_usd"], self.s["fees_usd"])
                nxt = self.s.get("cycle", 1) + 1
                self.s = _new_state()
                self.s["realized_pnl_usd"] = pnl
                self.s["realized_net_usd"] = net_tot
                self.s["fees_usd"] = fees
                self.s["cycle"] = nxt
                self.s["last_sell_price"] = price   # Reentry rule: do not buy back higher.
                self.s["post_sell_peak"] = price
                self.s["post_sell_trough"] = price
                self.s["last_exit_kind"] = "STOP" if was_sl else "TP"
                self.s["last_sell_ts"] = self._now()
                if self.exit_manager is not None:
                    self.exit_manager.clear_state(self.ticker)
                    self.exit_manager.sl_bounce_state = self.s.setdefault("sl_bounce_state", {})
                    self.exit_manager.tp_ride_state = self.s.setdefault("tp_ride_state", {})
                    self.exit_manager.reentry_state = self.s.setdefault("reentry_state", {})
                if was_sl and self.p.sl_rebuy_enabled:
                    self.s["sl_rebuy"] = {"low": price, "sell_price": price}
                    log(f"  🟢 [STRAT] pullback re-buy ARMED after the stop-loss "
                        f"(expecting +{self.p.sl_rebuy_bounce_pct}% from the low)")
                log(f"  [STRAT] === ciclu inchis, reincep (ciclu {nxt}) ===")

    # -- plasare / anulare -----------------------------------------------------
    def _place_buy(self, amount: float, limit: float, kind: str) -> None:
        qty = self._qty_for_amount(amount, limit)
        if qty <= 0:
            log("  ! [STRAT] qty 0 — sar")
            return
        if self.dry_run:
            self._paper_seq += 1
            log(f"  [STRAT] [PAPER] plasez BUY {kind} {qty} @ {limit:.2f} USD (~{amount:.0f} {self.ccy})")
            self.s["orders"].append({"id": f"PAPER-{self._paper_seq}", "side": "BUY", "qty": qty,
                                     "limit": round(limit, 2), "amount": amount, "kind": kind,
                                     "ts": self._now()})
            return
        if self.s.get("pending_submit"):
            log(f"  [STRAT] BUY {kind} deferred — a T212 submit is already pending")
            return
        pending = self._new_pending_submit(
            side="BUY", qty=qty, limit=limit, amount=amount, kind=kind,
            level=None, market=False,
        )
        outcome = self._submit_pending_order(pending)
        if outcome.state != "accepted":
            if outcome.state == "refused":
                log(f"  ! [STRAT] BUY {kind} explicitly refused: {outcome.reason}")
            else:
                log(f"  ! [STRAT] BUY {kind} outcome unknown: {outcome.reason}; keeping pending")
            if "insufficient" in outcome.reason.lower():
                self.s["buy_backoff_until"] = self._now() + 1800
                log("  [STRAT] insufficient funds — pausing buys for 30 minutes")
            return
        log(f"  [STRAT] BUY {kind} placed id={outcome.order_id} {qty} @ {limit:.2f}")

    def _place_sell(self, qty: float, limit: float, level: float | None = None,
                    kind: str = "TP", *, market: bool = False) -> bool:
        qty = round(float(qty), 2)
        if qty <= 0:
            log("  ! [STRAT] SELL qty 0 after rounding — keeping the dust, sending no order")
            return False
        tag = f"+{level:g}%" if level is not None else ""
        if self.dry_run:
            self._paper_seq += 1
            order_type = "MKT" if market else f"@ {limit:.2f}"
            log(f"  [STRAT] [PAPER] plasez SELL {kind}{tag} {qty:.2f} {order_type} USD")
            self.s["orders"].append({"id": f"PAPER-{self._paper_seq}", "side": "SELL",
                                     "qty": qty, "limit": round(limit, 2),
                                     "kind": kind, "level": level, "market": market,
                                     "ts": self._now()})
            return True
        if self.s.get("pending_submit"):
            log(f"  [STRAT] SELL {kind}{tag} deferred — a T212 submit is already pending")
            return False
        pending = self._new_pending_submit(
            side="SELL", qty=qty, limit=limit, amount=None, kind=kind,
            level=level, market=market,
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
            # Reset only when the venue confirms owned is effectively zero. A positive
            # owned amount with lower free balance must not erase a real position.
            _m = re.search(r'owned["\']?\s*[:=]\s*([0-9.]+)', message)
            if _m is None:
                log("  ! [STRAT] selling-not-owned without the owned quantity — NOT resetting the position")
                return False
            _owned = float(_m.group(1))
            if _owned <= 1e-6:
                log("  ! [STRAT] T212 confirms owned=0 — resetting the state (position closed/sold)")
                self.s["qty"] = 0.0
                self.s["cost_usd"] = 0.0
                self.s["spent_cash"] = 0.0
                self.s["orders"] = []
                self.s["locked_zero_until"] = self._now() + 300  # ignora adoptie stala 5 min
                self._save()
            else:
                log(f"  ! [STRAT] selling-not-owned but owned={_owned} (free<order) — NOT resetting the position")
            return False
        order_type = "MARKET" if market else f"@ {limit:.2f}"
        log(f"  [STRAT] SELL {kind}{tag} placed id={outcome.order_id} {qty:.2f} {order_type}")
        return True

    def _cancel_open(self, side: str) -> bool:
        o = self._find_open(side)
        if not o:
            return True
        return self._cancel_specific(o)

    def _cancel_specific(self, o: dict) -> bool:
        if self.dry_run or str(o["id"]).startswith("PAPER"):
            self._remove_order(o)
            log(f"  [STRAT] cancelled order {o.get('side', '?')} {o['id']}")
            return True
        if o.get("cancel_requested"):
            return True
        try:
            self.executor.cancel_order_with_intent(
                o.get("intent_id") or f"legacy-t212-{self.ticker}-{o['id']}",
                self.ticker, str(o["id"]),
            )
        except ProviderError as exc:
            log(f"  ! [STRAT] cancel failed for {o['id']}: {exc} — the order stays tracked")
            return False
        o["cancel_requested"] = True
        o["cancel_ts"] = self._now()
        self._save()
        log(f"  [STRAT] cancel requested {o.get('side', '?')} {o['id']} — waiting for a terminal status")
        return True

    def _cancel_all_orders(self) -> bool:
        success = True
        for order in list(self.s["orders"]):
            if not self._cancel_specific(order):
                success = False
        return success

    def _restore_cancelled_orders(self, cancelled_orders: list[dict], reason: str = "") -> None:
        """Best-effort rollback: restore previously resting orders after a replacement or exit fails."""
        def _restore_one(p: dict) -> bool:
            if p["side"] == "SELL":
                return self._place_sell(p["qty"], p["limit"], level=p.get("level"), kind=p.get("kind", "TP"))
            elif p["side"] == "BUY":
                amount = p.get("amount") or (p["qty"] * p["limit"])
                self._place_buy(amount, p["limit"], kind=p.get("kind", "ENTRY"))
                return True
            return False

        restore_cancelled_orders(
            cancelled_orders,
            restore_fn=_restore_one,
            log_fn=log,
            reason=reason,
        )

    def _is_dust(self, qty: float, ref_price: float | None = None) -> bool:
        """Return True if qty is negligible dust (below minimum broker holding / trading value)."""
        if qty <= 1e-9:
            return True
        if self.p.ladder_min_free <= 0:
            return False
        p = ref_price or self._avg_cost() or 0.0
        if p <= 0:
            return False
        # If ladder_min_free is configured (>0), residual dust value is around ladder_min_free.
        # Allow up to 1.5x ladder_min_free (or at least $5 broker minimum position value).
        # Also ensure dust threshold never exceeds half of the configured entry amount.
        min_dust = max(5.0, self.p.ladder_min_free * 1.5)
        max_allowed = (self.p.entry_amount * 0.5) if self.p.entry_amount > 0 else min_dust
        threshold = min(min_dust, max_allowed)
        return (qty * p) <= threshold

    def _sellable_qty(self, held: float, ref_price: float | None = None) -> float:
        """Return the maximum safely sellable quantity avoiding T212 min-opened-position.

        When held has fractional dust beyond 2 decimal places, selling round(held, 2)
        would leave sub-penny/sub-$5 dust that T212 rejects with min-opened-position.
        In that case, retain ladder_min_free buffer so the remaining position is >= min.
        For clean quantities without dust, sell the exact rounded quantity.
        """
        p = ref_price or self._avg_cost() or 0.0
        if self._is_dust(held, p):
            return 0.0
        q_round = round(held, 2)
        dust = held - q_round
        if abs(dust) > 1e-6 and self.p.ladder_min_free > 0 and p > 0:
            dust_val = abs(dust) * p
            if dust_val < self.p.ladder_min_free:
                buf = self.p.ladder_min_free / p
                if held > buf:
                    return round(held - buf, 2)
        return q_round

    def _manage_tp_ladder(self, held: float, avg: float) -> None:
        """Maintain one SELL per unsold ladder level at its target over average.

        Size remaining levels proportionally from current holdings and replace them when
        average cost changes after DCA. Never recreate levels already sold.
        """
        # Cancel any legacy non-level SELL before creating a ladder.
        for o in [x for x in self.s["orders"] if x["side"] == "SELL" and x.get("level") is None]:
            if not self._cancel_specific(o):
                return  # Do not overlap a ladder with a potentially active SELL.
            if o in self.s["orders"]:
                return  # Live: the request is accepted, but we wait for the terminal status.
        sold = set(self.s.get("tp_sold_levels", []))
        remaining = [(lvl, frac) for (lvl, frac) in self.p.tp_ladder if lvl not in sold]
        if not remaining or held <= 1e-9:
            return

        # Adapt remaining ladder rungs dynamically to current held volume
        remaining = adapt_ladder_rungs(remaining, held, min_tranche_qty=0.01)

        total = sum(f for _, f in remaining)
        if total <= 0:
            return
        # Highest tranche receives the remainder minus an unreserved buffer, preventing
        # rounding oversell while satisfying T212's minimum-open-position rule.
        desired = {}   # nivel -> (qty, limit)
        buf = (self.p.ladder_min_free / avg) if (self.p.ladder_min_free > 0 and avg > 0) else 0.0
        remaining_sorted = sorted(remaining, key=lambda x: x[0])
        acc = 0.0
        for i, (lvl, frac) in enumerate(remaining_sorted):
            last = (i == len(remaining_sorted) - 1)
            q = round(held - acc - buf, 2) if last else round(held * frac / total, 2)
            acc += q
            if q > 0:
                desired[lvl] = (q, round(avg * (1 + lvl / 100.0), 2))
        open_sells = {o.get("level"): o for o in self.s["orders"]
                      if o["side"] == "SELL" and o.get("level") is not None}
        cancelled_by_level = {}
        # Cancel unwanted orders or orders whose price/quantity changed after DCA.
        for lvl, o in list(open_sells.items()):
            d = desired.get(lvl)
            if d is None or abs(o["limit"] - d[1]) / d[1] > 0.001 or abs(o["qty"] - d[0]) > 1e-6:
                cancelled_by_level[lvl] = dict(o)
                if self._cancel_specific(o) and o not in self.s["orders"]:
                    open_sells.pop(lvl, None)
        # Place missing levels. Back off for 30 minutes after persistent tranche failure
        # instead of retrying on every tick. Defer and break if a submit is already in flight.
        now = self._now()
        fails = self.s.setdefault("tp_fail_until", {})
        for lvl, (q, lim) in desired.items():
            if lvl in open_sells:
                continue
            if now < fails.get(str(lvl), 0):
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
        """Return whether a resting limit BUY should be cancelled and placed again.

        It must have waited ``order_ttl_min`` and a fresh order at the current price
        (every BUY is placed at price minus ``entry_discount_pct``) must sit more than
        0.3% above it. Comparing the raw price instead re-placed an identical order on
        every TTL whenever the discount exceeded 0.3% (RGNT: 5%, NVDA: 0.4%).
        """
        if str(order.get("side")).upper() != "BUY" or order.get("market"):
            return False
        if (self._now() - order.get("ts", 0)) / 60 <= self.p.order_ttl_min:
            return False
        fresh_limit = price * (1 - self.p.entry_discount_pct / 100)
        return fresh_limit > float(order["limit"]) * 1.003

    # -- Reconciliation --------------------------------------------------------
    def _remove_order(self, o: dict) -> None:
        if o in self.s["orders"]:
            self.s["orders"].remove(o)

    def reconcile(self, price: float) -> None:
        if self.dry_run:
            self._reconcile_paper(price)
        else:
            self._reconcile_real(price)

    # -- Paper reconciliation --------------------------------------------------
    def _reconcile_paper(self, price: float) -> None:
        # Assume BUY fills at limit; SELL fills when price reaches its limit.
        for side in ("BUY", "SELL"):
            for o in [x for x in self.s["orders"] if x["side"] == side]:
                if o not in self.s["orders"]:
                    continue
                if side == "BUY":
                    self._remove_order(o)
                    self._apply_fill(o, o["qty"], o["limit"])
                elif o.get("market") or price >= o["limit"]:
                    self._remove_order(o)
                    if o.get("level") is not None:
                        self.s.setdefault("tp_sold_levels", []).append(o["level"])
                    self._apply_fill(o, o["qty"], price if o.get("market") else o["limit"])

    # -- Live reconciliation: T212 portfolio is the source of truth -------------
    # Filled orders disappear from the order endpoint while their position appears in portfolio.
    def _portfolio_position(self) -> tuple[float, float] | None:
        pf = self.client.get_portfolio()
        if pf is None:
            return None
        for p in pf:
            if str(p.get("ticker", "")).upper() == self.ticker.upper():
                return float(p.get("quantity") or 0.0), float(p.get("averagePrice") or 0.0)
        return 0.0, 0.0   # No holding.

    def _active_order_ids(self, orders: list[dict] | None = None) -> set | None:
        if orders is None:
            orders = self.client.list_active_orders()
        if orders is None:
            return None
        return {str(o.get("id")) for o in orders
                if str(o.get("ticker", "")).upper() == self.ticker.upper()}

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
        """Return the weighted price from tracked orders' cumulative deltas.

        The portfolio remains the source of truth for quantity. Order prices are
        used only when the sum of the deltas matches the position change.
        """
        rows = []
        total_qty = total_cost = 0.0
        for order in [o for o in self.s["orders"] if o.get("side") == side]:
            status = self._tracked_order_status(order, cache)
            if status is None or isinstance(status, str):
                continue
            try:
                cumulative_qty = float(status.filled_qty or 0.0)
                cumulative_cost = float(status.cost or 0.0)
                cumulative_fee = float(status.fee or 0.0)
                applied_qty = float(order.get("applied_fill_qty") or 0.0)
                applied_cost = float(order.get("applied_fill_cost") or 0.0)
                applied_fee = float(order.get("applied_fill_fee") or 0.0)
            except (TypeError, ValueError):
                continue
            if cumulative_qty + 1e-9 < applied_qty or cumulative_cost + 1e-9 < applied_cost:
                log(f"  ! [STRAT] fill T212 {order.get('id')} a regresat cumulativ — ignor")
                continue
            delta_qty = max(0.0, cumulative_qty - applied_qty)
            delta_cost = max(0.0, cumulative_cost - applied_cost)
            delta_fee = cumulative_fee - applied_fee
            if delta_qty > 1e-9:
                rows.append((order, cumulative_qty, cumulative_cost, cumulative_fee,
                             delta_qty, delta_cost, delta_fee, status.status))
                total_qty += delta_qty
                total_cost += delta_cost

        tolerance = max(0.011, abs(position_delta) * 1e-6)
        if not rows or abs(total_qty - position_delta) > tolerance or total_cost <= 0:
            if rows:
                log(f"  ! [STRAT] fill-uri T212 {side} ({total_qty:.4f}) != delta portfolio "
                    f"({position_delta:.4f}) — folosesc fallback-ul de portofoliu")
            return None

        for (order, cumulative_qty, cumulative_cost, cumulative_fee,
             _delta_qty, _delta_cost, _delta_fee, _venue_status) in rows:
            order["applied_fill_qty"] = cumulative_qty
            order["applied_fill_cost"] = cumulative_cost
            order["applied_fill_fee"] = cumulative_fee
        return total_cost / total_qty

    def _reconcile_real(self, price: float) -> None:
        real = self._portfolio_position()
        if real is None:
            # Debounce: log initially and every ten minutes, not on every tick.
            now = self._now()
            last = getattr(self, "_pf_unavail_logged", 0)
            if now - last > 600:
                log("  [STRAT] portofoliu indisponibil — sar reconcilierea (suprima repetatele 10 min)")
                self._pf_unavail_logged = now
            return
        real_qty, real_avg = real
        active_orders = self.client.list_active_orders()
        pending_outcome = self._recover_pending_submit(real_qty, active_orders)
        pending_for_fill = (
            dict(self.s.get("pending_submit") or {})
            if pending_outcome == "filled" else None
        )
        active = self._active_order_ids(active_orders)
        if active is None:
            active = {str(o["id"]) for o in self.s["orders"]}  # Cannot list, so retain tracked orders.

        prev_qty = self.s["qty"]
        prev_avg = self._avg_cost() or real_avg
        status_cache: dict[str, object] = {}

        # --- Executed BUY: the position increased (or adopt an existing position). ---
        if real_qty > prev_qty + 1e-6 and self._now() < self.s.get("locked_zero_until", 0):
            log("  [STRAT] adoptie ignorata — portfolio stale (not-owned recent, lock activ)")
            return
        if real_qty > prev_qty + 1e-6:
            fq = real_qty - prev_qty
            exact_price = self._exact_execution_price("BUY", fq, status_cache)
            fp = exact_price or (
                (real_avg * real_qty - prev_avg * prev_qty) / fq if fq > 0 else real_avg
            )
            source_order = next(
                (o for o in self.s["orders"] if o.get("side") == "BUY"), None,
            )
            if source_order is None and pending_for_fill and pending_for_fill.get("side") == "BUY":
                source_order = pending_for_fill
            is_adoption = source_order is None and prev_qty < 1e-9
            if is_adoption and self._is_dust(real_qty, real_avg):
                now = self._now()
                if now - getattr(self, "_dust_logged", 0) > 600:
                    log(f"  [STRAT] portfolio dust ({real_qty:.4f} shares, ~{real_qty*real_avg:.2f} USD) ignored — staying flat for new cycle")
                    self._dust_logged = now
                return
            is_dca = bool(
                source_order is not None
                and source_order.get("kind") == "DCA"
                and not source_order.get("dca_counted")
            )
            if is_dca:
                # Count a partially filled DCA order once, not once per poll.
                source_order["dca_counted"] = True
            self.s["last_buy_price"] = fp
            if self.s["entry_price"] is None:
                self.s["entry_price"] = fp
            if is_dca:
                self.s["dca_buys"] += 1
            self.s["qty"] = real_qty
            self.s["cost_usd"] = real_qty * real_avg
            self.s["spent_cash"] = round(real_qty * real_avg / self.fx_to_usd, 2)
            kind_label = (
                "ADOPTAT" if is_adoption
                else str((source_order or {}).get("kind") or ("DCA" if prev_qty > 1e-9 else "ENTRY"))
            )
            log(f"  [STRAT] BUY EXECUTED {fq:.4f} @ {fp:.2f} USD "
                f"({kind_label})  qty={real_qty:.4f} avg={real_avg:.2f}")
            notify(title=f"{self.yahoo_sym} {'ADOPTAT' if is_adoption else 'BUY'} {fq:.4f}@a{real_avg:.2f}",
                   body=(f"{kind_label} | q{real_qty:.4f} a{real_avg:.2f} p~{fp:.2f} | "
                         f"desf{self.s['spent_cash']:.0f}{self.ccy} DCA{self.s['dca_buys']}/{self.p.max_dca_buys}"),
                   source="T212", price=fp, desktop=self.desktop)
            if source_order is pending_for_fill:
                self._persist_pending_submit(None)
            self._cancel_open("SELL")   # The average changed; rebuild TP on the next step.

        # --- Executed SELL: the position decreased. ---
        elif real_qty < prev_qty - 1e-6:
            sold = prev_qty - real_qty
            exact_price = self._exact_execution_price("SELL", sold, status_cache)
            sell_price = exact_price or price
            gross, fee, net = _sell_pnl(prev_avg, sell_price, sold, self.p.fx_fee_pct)
            self.s["realized_pnl_usd"] += gross
            self.s["realized_net_usd"] += net
            self.s["fees_usd"] += fee
            self.s["qty"] = real_qty
            self.s["cost_usd"] = real_qty * real_avg
            self.s["spent_cash"] = round(real_qty * real_avg / self.fx_to_usd, 2)
            self.s["last_sell_price"] = sell_price
            approx = "" if exact_price is not None else "~"
            log(f"  [STRAT] SELL EXECUTED {sold:.4f} @ {approx}{sell_price:.2f} USD  "
                f"brut={gross:+.2f}  fee={fee:.2f}  net={net:+.2f} USD")
            notify(title=f"{self.yahoo_sym} SELL {sold:.4f}@{approx}{sell_price:.2f} N{net:+.2f}$",
                   body=f"a{prev_avg:.2f} · br{gross:+.2f} fee{fee:.2f} N{net:+.2f}$ | Ntot{self.s['realized_net_usd']:+.2f}$",
                   source="T212", price=sell_price, desktop=self.desktop)
            if pending_for_fill and pending_for_fill.get("side") == "SELL":
                self._persist_pending_submit(None)

        else:
            # Unchanged position: synchronize state with the actual position.
            self.s["qty"] = real_qty
            self.s["cost_usd"] = real_qty * real_avg
            if real_qty > 1e-9:
                self.s["spent_cash"] = round(real_qty * real_avg / self.fx_to_usd, 2)

        # --- Remove inactive orders and enforce the TTL for stale BUY orders. ---
        for o in list(self.s["orders"]):
            if str(o["id"]).startswith("PAPER"):
                continue
            if str(o["id"]) not in active:
                status = self._tracked_order_status(o, status_cache)
                if status is not None and not isinstance(status, str) and status.status in {"closed", "canceled", "expired"}:
                    if (o["side"] == "SELL" and o.get("level") is not None
                            and float(status.filled_qty or 0.0) > 1e-9):
                        self.s.setdefault("tp_sold_levels", []).append(o["level"])
                    self._remove_order(o)
                elif status == "not_found":
                    misses = int(o.get("lookup_misses") or 0) + 1
                    o["lookup_misses"] = misses
                    max_misses = 2 if o.get("cancel_requested") else 3
                    if misses >= max_misses:
                        log(f"  [STRAT] order {o.get('side', '?')} {o['id']} confirmed absent ({misses} misses) — removing from tracking")
                        if (o.get("side") == "SELL" and o.get("level") is not None
                                and float(o.get("applied_fill_qty") or 0.0) > 1e-9):
                            self.s.setdefault("tp_sold_levels", []).append(o["level"])
                        self._remove_order(o)
                elif status is None:
                    # Transient status lookup failure (e.g. rate limit, network);
                    # retain order tracking without incrementing misses.
                    continue
                continue
            else:
                o.pop("lookup_misses", None)
                if self._buy_is_stale(o, price):
                    log(f"  [STRAT] BUY {o['id']} unfilled, the price rose — cancelling and re-placing")
                    self._cancel_specific(o)

        # --- Closed cycle (the full position or all exit tranches were sold): start a new cycle. ---
        is_flat = real_qty <= 1e-9
        ladder_levels = {lvl for (lvl, _) in self.p.tp_ladder} if self.p.tp_ladder else set()
        all_tp_sold = bool(ladder_levels and ladder_levels.issubset(set(self.s.get("tp_sold_levels", []))))
        has_open_sells = any(o.get("side") == "SELL" for o in self.s["orders"])
        is_dust = self._is_dust(real_qty, real_avg or price)
        cycle_closed = (is_flat and prev_qty > 1e-9) or (
            prev_qty > 1e-9 and is_dust and not has_open_sells and (all_tp_sold or self.s.get("sl_pending") or not self.p.tp_ladder)
        )
        if cycle_closed:
            pnl, net_tot, fees = (self.s["realized_pnl_usd"],
                                  self.s["realized_net_usd"], self.s["fees_usd"])
            was_sl = bool(self.s.get("sl_pending"))   # Was this a catastrophic stop-loss exit?
            exit_price = self.s.get("last_sell_price") or price
            nxt = self.s.get("cycle", 1) + 1
            self.s = _new_state()
            self.s["realized_pnl_usd"] = pnl
            self.s["realized_net_usd"] = net_tot
            self.s["fees_usd"] = fees
            self.s["cycle"] = nxt
            self.s["last_sell_price"] = exit_price
            self.s["post_sell_peak"] = exit_price
            self.s["post_sell_trough"] = exit_price
            self.s["last_exit_kind"] = "STOP" if was_sl else "TP"
            self.s["last_sell_ts"] = self._now()
            if self.exit_manager is not None:
                self.exit_manager.clear_state(self.ticker)
                self.exit_manager.sl_bounce_state = self.s.setdefault("sl_bounce_state", {})
                self.exit_manager.tp_ride_state = self.s.setdefault("tp_ride_state", {})
                self.exit_manager.reentry_state = self.s.setdefault("reentry_state", {})
            if was_sl and self.p.sl_rebuy_enabled:   # Rebuy on a bounce, not below the sale, to catch recovery.
                self.s["sl_rebuy"] = {"low": exit_price, "sell_price": exit_price}
                log(f"  🟢 [STRAT] pullback re-buy ARMED after the stop-loss (expecting +{self.p.sl_rebuy_bounce_pct}% from the low)")
            log(f"  [STRAT] === ciclu inchis, reincep (ciclu {nxt}) ===")

    # -- decision step ---------------------------------------------------------
    def _check_trailing(self, price: float) -> bool:
        """Sell the entire position after a ``trail_pct`` drop from its peak.

        This crash breaker complements the fixed average-cost stop loss by exiting
        a sustained decline earlier. It reuses ``sl_rebuy`` to reenter on a bounce.
        Return ``True`` when triggered so processing of the current tick stops.
        """
        if self.p.trail_pct <= 0 or self.s["qty"] <= 1e-9:
            self.s["pos_peak"] = 0.0            # Flat or disabled trailing: reset the peak.
            self.s["tr_alerted"] = False
            self.s["tr_armed"] = False          # A new cycle starts with a new warm-up.
            return False
        if any(o.get("kind") in {"TR", "SL"} for o in self.s["orders"]):
            return True
        if self.p.tp_mdhtp_enabled and self.exit_manager and self.ticker in self.exit_manager.tp_ride_state:
            return False

        avg = self._avg_cost()
        peak = self.s.get("pos_peak", 0.0) or 0.0
        if not peak or peak <= 0:
            self.s["pos_peak"] = peak = max(avg or price, price)
        if price > peak:
            self.s["pos_peak"] = peak = price   # Track the high while holding the position.
            self.s.pop("tr_fail_until", None)
            self.s["tr_alerted"] = False

        drop_pct = (peak - price) / peak * 100 if peak > 0 else 0.0
        is_crash = peak > 0 and drop_pct >= self.p.trail_pct

        # WARM-UP (matching Binance/Kraken trailing_core): keep trailing inactive on normal dips
        # until price rises trail_min_profit_pct above average.
        # Extreme crash protection is decoupled: a crash exceeding trail_pct triggers exit immediately.
        if not self.s.get("tr_armed"):
            minp = self.p.trail_min_profit_pct
            if minp > 0 and (not avg or price < avg * (1 + minp / 100.0)):
                if not is_crash:
                    return False                # Warming up: do not sell on normal dip.
            self.s["tr_armed"] = True
            log(f"  [STRAT] trailing ARMED: price {price:.2f} >= avg{(avg or 0):.2f}+{minp}% — protecting from now on (-{self.p.trail_pct}% from the peak)")

        if not is_crash:
            return False
        if self._now() < self.s.get("tr_fail_until", 0):
            return False
        # The peak drawdown reached the threshold: sell all once, replacing only a stale limit.
        tr = next((o for o in self.s["orders"] if o.get("kind") == "TR"), None)
        if tr is None or (not tr.get("market") and price < tr["limit"]):
            cancelled_orders = [dict(o) for o in self.s["orders"] if o.get("kind") != "TR"]
            if not self._cancel_all_orders():
                log("  ! [STRAT] TRAILING deferred: at least one order could not be cancelled")
                return True
            pending_orders = [o for o in self.s["orders"] if o.get("kind") != "TR"]
            if pending_orders:
                log(f"  [STRAT] TRAILING deferred: waiting for {len(pending_orders)} existing order(s) to reconcile cancellation")
                return True
            if self.s.get("pending_submit"):
                log("  [STRAT] TRAILING deferred: waiting for pending submit to reconcile")
                return True
            sell_qty = self._sellable_qty(self.s["qty"], price)
            placed = self._place_sell(
                sell_qty, round(price, 2), kind="TR", market=True,
            )
            if not placed:
                if not self.s.get("pending_submit"):
                    self.s["tr_fail_until"] = self._now() + 1800
                    log("  ! [STRAT] TRAILING sell failed or refused — backoff 30 min (retaining TP ladder)")
                    self._restore_cancelled_orders(cancelled_orders, reason="trailing sell failed")
                return True
            self.s["sl_pending"] = True         # Arm bounce reentry as for a catastrophic stop loss.
        if not self.s.get("tr_alerted"):
            self.s["tr_alerted"] = True
            log(f"  📉 [STRAT] TRAILING: -{drop_pct:.2f}% de la peak {peak:.2f} >= {self.p.trail_pct}% — VAND TOT")
            notify(title=f"📉 TRAILING {self.yahoo_sym} -{drop_pct:.1f}%",
                   body=f"de la peak{peak:.2f} ≥{self.p.trail_pct}% — vand tot",
                   source="T212", price=price, desktop=self.desktop)
        return True

    def _vol_1h_ratio(self) -> tuple[float, str]:
        """Return 1h volatility as a decimal ratio [e.g. 0.025 for 2.5%] and its source."""
        if hasattr(self, "_shadow_vol_1h"):
            try:
                vol = self._shadow_vol_1h()
                if vol is not None and math.isfinite(float(vol)) and float(vol) > 0:
                    return vol_pct_to_ratio(vol), "shadow"
            except Exception:
                pass
        cached = resolve_cached_volatility_1h(self.ticker)
        if cached is not None:
            return cached, "cache"
        return 0.0, "none"

    def _check_stop_loss(self, price: float) -> bool:
        """Close the entire position when unrealized loss exceeds the threshold."""
        if self.p.stop_loss_pct <= 0:
            return False
        if any(o.get("kind") in {"TR", "SL"} for o in self.s["orders"]):
            return True
        avg = self._avg_cost()
        if not avg:
            return False
        loss_pct = (avg - price) / avg * 100   # A long loses when price is below average cost.

        # Multi-Horizon Dynamic Stop Loss (MHDSL) via centralized PositionExitManager
        if self.exit_manager is not None and self.p.sl_mhdsl_enabled:
            in_sl = (
                loss_pct >= self.p.stop_loss_pct
                or are_close(loss_pct, self.p.stop_loss_pct, self.p.reentry_tolerance_pct)
            )
            in_bounce_watch = bool(self.ticker in self.exit_manager.sl_bounce_state)
            if not in_sl and not in_bounce_watch:
                self.s["sl_alerted"] = False
                self.s["sl_pending"] = False
                self.s.pop("sl_fail_until", None)
                return False

            trend_up = False
            if self._trend_slope_provider:
                try:
                    slope = self._trend_slope_provider(self.yahoo_sym)
                    if slope is not None and slope >= 0.0:
                        trend_up = True
                except Exception:
                    pass

            vol_1h, vol_source = self._vol_1h_ratio()

            tick_time = self._now()
            held = self.s.get("qty", 0.0)
            sell_qty = self._sellable_qty(held, price)

            snapshot = PositionSnapshot(
                symbol=self.ticker,
                current_price=price,
                avg_buy_price=avg,
                net_qty=held,
                sellable_qty=sell_qty,
                current_time_s=tick_time,
                trend_up=trend_up,
                vol_1h=vol_1h,
                vol_source=vol_source,
                provider="t212",
            )

            decision = self.exit_manager.evaluate_stop_loss(
                snapshot, custom_threshold=self.p.stop_loss_pct / 100.0,
            )

            if decision.action == ExitAction.SELL and decision.should_exit:
                if self._now() < self.s.get("sl_fail_until", 0):
                    return False
                sl = next((o for o in self.s["orders"] if o.get("kind") == "SL"), None)
                if sl is None or (not sl.get("market") and price < sl["limit"]):
                    cancelled_orders = [dict(o) for o in self.s["orders"] if o.get("kind") != "SL"]
                    if not self._cancel_all_orders():
                        log("  ! [STRAT] STOP deferred: at least one order could not be cancelled")
                        return True
                    pending_orders = [o for o in self.s["orders"] if o.get("kind") != "SL"]
                    if pending_orders:
                        log(f"  [STRAT] STOP deferred: waiting for {len(pending_orders)} existing order(s) to reconcile cancellation")
                        return True
                    if self.s.get("pending_submit"):
                        log("  [STRAT] STOP deferred: waiting for pending submit to reconcile")
                        return True
                    placed = self._place_sell(
                        sell_qty, round(price, 2), kind="SL", market=True,
                    )
                    if not placed:
                        if not self.s.get("pending_submit"):
                            self.s["sl_fail_until"] = self._now() + 1800
                            log("  ! [STRAT] STOP sell failed or refused — backoff 30 min (retaining TP ladder)")
                            self._restore_cancelled_orders(cancelled_orders, reason="stop-loss sell failed")
                        return True
                    self.s["sl_pending"] = True
                    self.s["tp_pending_exit"] = None
                if not self.s.get("sl_alerted"):
                    self.s["sl_alerted"] = True
                    log(f"  🛑 [STRAT] MHDSL STOP-LOSS: loss {loss_pct:.2f}% >= {self.p.stop_loss_pct}% ({decision.reason}) — SELLING EVERYTHING (cutting the loss)")
                    notify(title=f"🛑 SL {self.yahoo_sym} -{loss_pct:.1f}%",
                           body=f"loss {loss_pct:.1f}% >=threshold{self.p.stop_loss_pct}% ({decision.reason}) — selling everything",
                           source="T212", price=price, desktop=self.desktop)
                return True
            elif decision.action == ExitAction.HOLD:
                if decision.metadata.get("priority") in (1, 2) or (
                    self.exit_manager and self.ticker in self.exit_manager.sl_bounce_state
                ):
                    log(f"  🛡️ [STRAT] MHDSL BOUNCE DEFERRAL: {decision.reason}")
                return False
            return False

        if loss_pct < self.p.stop_loss_pct:
            self.s["sl_alerted"] = False        # Recovery above threshold permits a new alert.
            self.s["sl_pending"] = False        # End this stop-loss episode after recovery.
            self.s.pop("sl_fail_until", None)
            return False
        if self._now() < self.s.get("sl_fail_until", 0):
            return False
        # Anti-spam: place the stop once and replace it only if price moved below its limit.
        sl = next((o for o in self.s["orders"] if o.get("kind") == "SL"), None)
        if sl is None or (not sl.get("market") and price < sl["limit"]):
            cancelled_orders = [dict(o) for o in self.s["orders"] if o.get("kind") != "SL"]
            if not self._cancel_all_orders():
                log("  ! [STRAT] STOP deferred: at least one order could not be cancelled")
                return True
            pending_orders = [o for o in self.s["orders"] if o.get("kind") != "SL"]
            if pending_orders:
                log(f"  [STRAT] STOP deferred: waiting for {len(pending_orders)} existing order(s) to reconcile cancellation")
                return True
            if self.s.get("pending_submit"):
                log("  [STRAT] STOP deferred: waiting for pending submit to reconcile")
                return True
            sell_qty = self._sellable_qty(self.s["qty"], price)
            placed = self._place_sell(
                sell_qty, round(price, 2), kind="SL", market=True,
            )
            if not placed:
                if not self.s.get("pending_submit"):
                    self.s["sl_fail_until"] = self._now() + 1800
                    log("  ! [STRAT] STOP sell failed or refused — backoff 30 min (retaining TP ladder)")
                    self._restore_cancelled_orders(cancelled_orders, reason="stop-loss sell failed")
                return True
            self.s["sl_pending"] = True         # Mark this episode so cycle closure arms bounce reentry.
        if not self.s.get("sl_alerted"):           # Notify only once per episode.
            self.s["sl_alerted"] = True
            log(f"  🛑 [STRAT] STOP-LOSS: loss {loss_pct:.2f}% >= {self.p.stop_loss_pct}% — SELLING EVERYTHING (cutting the loss)")
            notify(title=f"🛑 SL {self.yahoo_sym} -{loss_pct:.1f}%",
                   body=f"loss {loss_pct:.1f}% >=threshold{self.p.stop_loss_pct}% — selling everything",
                   source="T212", price=price, desktop=self.desktop)
        return True

    def _check_loss_alert(self, price: float) -> None:
        """Send an informational alert for each new unrealized-loss band.

        This does not sell. One notification per ``loss_alert_step`` band avoids
        spam, and recovery is silent.
        """
        step = self.p.loss_alert_step
        avg = self._avg_cost()
        if step <= 0 or not avg:
            return
        loss_pct = (avg - price) / avg * 100
        band = int(loss_pct // step) if loss_pct > 0 else 0
        # High-water mark: notify only when loss reaches a new maximum band.
        # Do not lower it on recovery, so revisiting an alerted level does not notify again.
        # Cycle closure resets it through _new_state (loss_band=0).
        if band > self.s.get("loss_band", 0):
            log(f"  📉 [STRAT] {self.yahoo_sym} loss -{loss_pct:.1f}% (threshold {band*step:.0f}%)")
            notify(title=f"📉 {self.yahoo_sym} -{loss_pct:.1f}%",
                   body=f"loss -{loss_pct:.1f}% (over {band*step:.0f}%) | q{self.s['qty']:.2f} a{avg:.2f} p{price:.2f}",
                   source="T212", price=price, desktop=self.desktop)
            self.s["loss_band"] = band

    def _handle_sl_rebuy(self, price: float) -> None:
        """Rebuy after a catastrophic stop loss once price bounces from its low.

        Track the post-sale low and reenter when price recovers by
        ``sl_rebuy_bounce_pct``. This mirrors Binance/Kraken trailing reentry and
        targets recovery rather than a falling price.
        """
        rb = self.s.get("sl_rebuy")
        if not rb:
            return
        rb["low"] = min(rb.get("low", price), price)          # Track the post-stop low.
        if price < rb["low"] * (1 + self.p.sl_rebuy_bounce_pct / 100.0):
            return                                            # Wait until the bounce is confirmed.
        self.s.pop("sl_rebuy", None)                          # Consume the one-tranche reentry arm.
        if self.s["spent_cash"] + self.p.entry_amount > self.p.max_budget:
            log(f"  [STRAT] re-buy SL cancelled — budget cap {self.p.max_budget:.0f} {self.ccy} reached")
            return
        disc = 1 - self.p.entry_discount_pct / 100
        log(f"  🟢 [STRAT] RE-BUY after the SL: a +{self.p.sl_rebuy_bounce_pct}% pullback from the low {rb['low']:.2f} — re-entering ENTRY")
        notify(title=f"🟢 {self.yahoo_sym} RE-BUY after the stop-loss",
               body=(f"recul +{self.p.sl_rebuy_bounce_pct}% de la min {rb['low']:.2f} — "
                     f"re-entering with {self.p.entry_amount:.0f}{self.ccy}"),
               source="T212", price=price, desktop=self.desktop)
        self._place_buy(self.p.entry_amount, price * disc, kind="ENTRY")

    def _effective_reentry_pullback_pct(self) -> tuple[float, str]:
        """Return the dynamic volatility-scaled or fixed pullback threshold from peak.

        When STRAT_REENTRY_PULLBACK_ADAPTIVE is enabled, scale pullback by volatility:
        pullback = clamp(K_PULLBACK * vol, min_pullback, max_pullback).
        """
        if not self.p.reentry_pullback_adaptive:
            return self.p.reentry_pullback_pct, "fixed"
        vol = None
        if hasattr(self, "_shadow_vol_1h"):
            try:
                vol = self._shadow_vol_1h()
            except Exception:
                vol = None
        if vol is None:
            return self.p.reentry_pullback_pct, "fixed (fallback)"
        val = max(self.p.reentry_pullback_min, min(self.p.reentry_pullback_max, self.p.reentry_pullback_k * vol))
        return val, f"adaptive (vol {vol:.2f}% x k={self.p.reentry_pullback_k} -> {val:.2f}%)"

    def _evaluate_reentry_gate(self, price: float) -> tuple[bool, str]:
        """Evaluate re-entry gate conditions symmetrically with PositionExitManager or legacy fallback."""
        lsp = self.s.get("last_sell_price")
        if not lsp:
            return True, "no_last_sell"

        cur_peak = self.s.get("post_sell_peak") or lsp
        self.s["post_sell_peak"] = max(cur_peak, price)
        cur_trough = self.s.get("post_sell_trough") or lsp
        self.s["post_sell_trough"] = min(cur_trough, price)

        if self.exit_manager is not None and (self.p.sl_mhdsl_enabled or self.p.tp_mdhtp_enabled):
            self.exit_manager.config.reentry_drop_pct = self.p.reentry_drop_pct / 100.0
            pullback_pct, pb_src = self._effective_reentry_pullback_pct()
            self.exit_manager.config.reentry_pullback_pct = pullback_pct / 100.0
            self.exit_manager.config.reentry_sl_bounce_pct = (
                self.p.sl_rebuy_bounce_pct / 100.0 if self.p.sl_rebuy_enabled else 0.015
            )
            self.exit_manager.config.reentry_ttl_hours = self.p.reentry_ttl_hours
            self.exit_manager.config.reentry_tolerance_pct = self.p.reentry_tolerance_pct
            self.exit_manager.config.reentry_hybrid_enabled = self.p.reentry_hybrid_enabled

            trend_up = False
            trend_down = False
            if self._trend_slope_provider:
                try:
                    slope = self._trend_slope_provider(self.yahoo_sym)
                    if slope is not None:
                        if slope >= 0.0:
                            trend_up = True
                        elif slope < (-self.p.dca_trend_gate_pct if self.p.dca_trend_gate_pct > 0 else -0.1):
                            trend_down = True
                except Exception:
                    pass

            vol_1h, vol_source = self._vol_1h_ratio()

            last_exit_kind = "STOP" if self.s.get("sl_rebuy") or self.s.get("last_exit_kind") == "STOP" else "TP"

            reentry_snap = ReentrySnapshot(
                symbol=self.ticker,
                current_price=price,
                last_sell_price=lsp,
                last_exit_kind=last_exit_kind,
                post_sell_peak=self.s.get("post_sell_peak"),
                post_sell_trough=self.s.get("post_sell_trough"),
                current_time_s=self._now(),
                last_sell_time_s=self.s.get("last_sell_ts"),
                trend_up=trend_up,
                trend_down=trend_down,
                vol_1h=vol_1h,
                vol_source=vol_source,
            )
            reentry_dec = self.exit_manager.evaluate_reentry(reentry_snap)
            return reentry_dec.should_enter, reentry_dec.reason

        # Rebuy on a bounce after a catastrophic stop loss, as on Binance/Kraken.
        # While armed, it takes priority over reentry below the previous sale.
        if self.s.get("sl_rebuy"):
            self._handle_sl_rebuy(price)
            return False, "handling_sl_rebuy"
        if self.p.reentry_hybrid_enabled:
            is_bull = True if self.p.reentry_peak_relative else False
            if not is_bull and self._trend_slope_provider:
                try:
                    slope = self._trend_slope_provider(self.yahoo_sym)
                    if slope is not None and slope >= 0.0:
                        is_bull = True
                except Exception:
                    pass
            pullback_pct, pb_src = self._effective_reentry_pullback_pct()
            elapsed = (self._now() - self.s["last_sell_ts"]) if self.s.get("last_sell_ts") else None
            ttl_s = (self.p.reentry_ttl_hours * 3600.0) if self.p.reentry_ttl_hours > 0 else None
            blocked, reason = sr.reentry_hybrid_blocked(
                price=price,
                last_sell=lsp,
                post_sell_peak=self.s.get("post_sell_peak"),
                is_bull=is_bull,
                drop_pct=self.p.reentry_drop_pct,
                pullback_pct=pullback_pct,
                tol_pct=self.p.reentry_tolerance_pct,
                elapsed_sec=elapsed,
                ttl_sec=ttl_s,
            )
            if blocked:
                return False, f"hybrid: {reason} [{pb_src}]"
            return True, f"hybrid_unblocked ({reason}) [{pb_src}]"
        else:
            rdp = self.p.reentry_drop_pct
            if rdp > 0:
                prag = lsp * (1 - rdp / 100)
                if price > prag and not are_close(price, prag, self.p.reentry_tolerance_pct):
                    return False, f"price {price:.2f} > threshold {prag:.2f} (sold at {lsp:.2f}, waiting for -{rdp}%)"
            return True, "drop_target_met"

    def step(self, price: float) -> None:
        held = self.s["qty"]
        self._check_loss_alert(price)   # Alert on deeper loss while holding; never sells.
        disc = 1 - self.p.entry_discount_pct / 100

        in_backoff = self._now() < self.s.get("buy_backoff_until", 0)

        is_flat = held <= 1e-9 or (self._is_dust(held, price) and not self._has_open("SELL"))
        if is_flat:
            self.s["tp_pending_exit"] = None
            if in_backoff:   # After insufficient funds, avoid buys while the account is empty.
                return
            if self._has_open("BUY"):
                return
            # Kraken-style reentry rule: do not buy back above the previous sale price.
            allowed, reason = self._evaluate_reentry_gate(price)
            if not allowed:
                log(f"  [STRAT] re-entry blocked: {reason}")
                return
            lsp = self.s.get("last_sell_price")
            if lsp:
                log(f"  [STRAT] re-entry approved: {reason} — placing ENTRY")
            if self.s["spent_cash"] + self.p.entry_amount > self.p.max_budget:
                log(f"  [STRAT] budget cap {self.p.max_budget:.0f} {self.ccy} reached — not entering")
                return
            self._place_buy(self.p.entry_amount, price * disc, kind="ENTRY")
            return

        # Check the tighter trailing crash breaker before the catastrophic stop loss.
        if self._check_trailing(price):
            return

        # Stop loss cuts the loss before DCA or take-profit handling.
        if self._check_stop_loss(price):
            return

        avg = self._avg_cost()

        if self.p.enable_takeprofit and avg and not self.s.get("sl_pending"):
            if self.p.tp_mdhtp_enabled and self.exit_manager is not None:
                tick_time = self._now()
                pending_exit = self.s.get("tp_pending_exit")

                if pending_exit is None:
                    trend_up = False
                    if self._trend_slope_provider:
                        try:
                            slope = self._trend_slope_provider(self.yahoo_sym)
                            if slope is not None and slope >= 0.0:
                                trend_up = True
                        except Exception:
                            pass

                    vol_1h, vol_source = self._vol_1h_ratio()

                    self.exit_manager.config.tp_trend_ride_enabled = True
                    self.exit_manager.config.tp_profit_floor_min_pct = (
                        (self.p.trail_min_profit_pct or 0.0) / 100.0
                    )
                    self.exit_manager.config.tp_pullback_pct = (
                        (self.p.trail_pct / 100.0) if self.p.trail_pct > 0 else 0.015
                    )
                    self.exit_manager.config.tp_stagnation_max_sec = self.p.tp_stagnation_max_sec

                    sell_qty_eval = self._sellable_qty(held, price)
                    snapshot = PositionSnapshot(
                        symbol=self.ticker,
                        current_price=price,
                        avg_buy_price=avg,
                        net_qty=held,
                        sellable_qty=sell_qty_eval,
                        current_time_s=tick_time,
                        trend_up=trend_up,
                        vol_1h=vol_1h,
                        vol_source=vol_source,
                        provider="t212",
                    )

                    decision = self.exit_manager.evaluate_take_profit(
                        snapshot, custom_threshold=self.p.takeprofit_pct / 100.0
                    )

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
                    elif decision.action == ExitAction.HOLD:
                        if self.ticker in self.exit_manager.tp_ride_state:
                            ride_state = self.exit_manager.tp_ride_state.get(self.ticker, {})
                            peak_px = ride_state.get("peak_price", price)
                            self.s["trail_peak"] = peak_px
                            self._cancel_open("SELL")
                            log(f"  [STRAT] above the TP, RIDING IT ({decision.reason}) (peak {peak_px:.2f})")
                            return
                        else:
                            self.s["trail_peak"] = None
                            self._cancel_open("SELL")

                if pending_exit:
                    if not self._cancel_all_orders():
                        log("  ! [STRAT] MDHTP exit deferred: at least one order could not be cancelled")
                        return
                    pending_orders = [o for o in self.s["orders"] if not (o.get("market") and o.get("side") == "sell" and o.get("kind") == "TP")]
                    if pending_orders:
                        log(f"  [STRAT] MDHTP exit deferred: waiting for {len(pending_orders)} existing order(s) to reconcile cancellation")
                        return
                    if self.s.get("pending_submit"):
                        log("  [STRAT] MDHTP exit deferred: waiting for pending submit to reconcile")
                        return

                    fresh_held = self.s["qty"]
                    sell_fraction = float(pending_exit.get("sell_fraction", 1.0) or 1.0)
                    target_qty = fresh_held * sell_fraction if sell_fraction < 1.0 else fresh_held
                    sell_qty = self._sellable_qty(target_qty, price)
                    if sell_qty <= 0 or self._is_dust(sell_qty, price):
                        log("  [STRAT] MDHTP exit: no sellable quantity remaining — clearing exit intent")
                        self.s["tp_pending_exit"] = None
                        return

                    exit_px = round(price * 0.999, 2)
                    if self._place_sell(sell_qty, exit_px, kind="TP", market=True):
                        log(f"  [STRAT] MDHTP EXIT ({pending_exit.get('reason')}) -> reference {exit_px} ({sell_qty} shares)")
                        self.s["tp_pending_exit"] = None
                        return
                    else:
                        log(f"  ! [STRAT] MDHTP EXIT order refused/failed ({pending_exit.get('reason')}) — intent retained for retry")
                        return
                elif decision.action == ExitAction.HOLD:
                    if self.ticker in self.exit_manager.tp_ride_state:
                        ride_state = self.exit_manager.tp_ride_state.get(self.ticker, {})
                        peak_px = ride_state.get("peak_price", price)
                        self.s["trail_peak"] = peak_px
                        self._cancel_open("SELL")
                        log(f"  [STRAT] above the TP, RIDING IT ({decision.reason}) (peak {peak_px:.2f})")
                        return
                    else:
                        self.s["trail_peak"] = None
                        self._cancel_open("SELL")
                elif decision.reason == "not_in_tp_territory":
                    self.s["trail_peak"] = None
                    self._cancel_open("SELL")
            elif self.p.tp_ladder:
                self._manage_tp_ladder(held, avg)        # Scale out in ladder steps.
            else:
                target = avg * (1 + self.p.takeprofit_pct / 100)
                sell_qty = self._sellable_qty(held, target)
                sell = self._find_open("SELL")
                if sell is None:
                    self._place_sell(sell_qty, target)
                elif abs(sell["limit"] - target) / target > 0.001 or abs(sell["qty"] - sell_qty) > 1e-6:
                    old_sell = dict(sell)
                    if self._cancel_open("SELL") and sell not in self.s["orders"]:
                        if not self._place_sell(sell_qty, target):
                            if not self.s.get("pending_submit"):
                                self._restore_cancelled_orders([old_sell], reason="single TP replace failed")

        prag_dca = (self.s["last_buy_price"] * (1 - self.p.dca_drop_pct / 100)
                    if self.s["last_buy_price"] else None)
        in_sl_state = bool(
            self.s.get("sl_pending")
            or (self._now() < self.s.get("sl_fail_until", 0))
            or (self.ticker in self.exit_manager.sl_bounce_state if self.exit_manager else False)
        )
        if (not in_backoff
                and not in_sl_state
                and self.s["dca_buys"] < self.p.max_dca_buys
                and prag_dca
                # Treat a price close to the threshold as reached, using reentry tolerance.
                and (price <= prag_dca or are_close(price, prag_dca, self.p.reentry_tolerance_pct))
                and self.s["spent_cash"] + self.p.dca_amount <= self.p.max_budget
                and not self._has_open("BUY")):
            # Binance/Kraken-style trend gate: skip DCA during a confirmed downtrend
            # so capital is not added to a falling asset. Zero disables the gate.
            # During a sustained dip, the DCA condition is true on every tick (~18s),
            # so a five-minute cache prevents repeated Yahoo calls and log spam. The
            # slope already uses five-minute bars.
            if self.p.dca_trend_gate_pct > 0:
                if self._now() < getattr(self, "_dca_gate_until", 0):
                    return   # Recently trend-blocked; retry after the cache window.
                slope = self._trend_slope_provider(self.yahoo_sym)
                if slope is not None and slope < -self.p.dca_trend_gate_pct:
                    self._dca_gate_until = self._now() + 300
                    log(f"  [STRAT] DCA BLOCKED de trend: slope {slope:+.3f}%/bar "
                        f"< -{self.p.dca_trend_gate_pct}% (downtrend) — re-verific in 5 min")
                    return
            log(f"  [STRAT] dip: {price:.2f} <= {self.s['last_buy_price']:.2f}"
                f"×(1-{self.p.dca_drop_pct}%) — DCA")
            self._place_buy(self.p.dca_amount, price * disc, kind="DCA")

    # -- main loop -------------------------------------------------------------
    def run(self, stop: threading.Event | None = None) -> None:
        mode = "avg_tp" if self.p.enable_takeprofit else "dca_only"
        log("  === STRATEGIE PORNITA ===")
        log(f"      instrument : {self.ticker}  (price through {self.yahoo_sym})")
        log(f"      mod        : {mode}   {'[PAPER]' if self.dry_run else '⚠ REAL — BANI ADEVARATI'}")
        log(f"      intrare    : {self.p.entry_amount:.0f} {self.ccy} @ market-{self.p.entry_discount_pct}%")
        log(f"      DCA        : {self.p.dca_amount:.0f} {self.ccy} at every -{self.p.dca_drop_pct}% "
            f"(max {self.p.max_dca_buys})")
        if self.p.enable_takeprofit:
            log(f"      take-profit: +{self.p.takeprofit_pct}% above the average price")
        else:
            log("      take-profit: dezactivat (dca_only)")
        log(f"      PLAFON     : {self.p.max_budget:.0f} {self.ccy} / ciclu")
        log(f"      check      : la {self.p.check_minutes:.0f} min   |  1 {self.ccy} = {self.fx_to_usd:.4f} USD")
        log(f"      ! break-even threshold ~{self.p.fx_fee_pct*2:.2f}% (FX) plus spread; TP={self.p.takeprofit_pct}%")

        def _delay(delay_sec: float) -> None:
            if stop is not None:
                stop.wait(delay_sec)
            else:
                time.sleep(delay_sec)

        try:
            while not (stop is not None and stop.is_set()):
                price = get_price_usd(self.yahoo_sym)
                if price is None:
                    log("  [STRAT] price unavailable — reincerc")
                    _delay(self.p.check_minutes * 60)
                    continue
                try:
                    # After a disk error, make no new decisions until the current
                    # snapshot can be persisted again.
                    if self._state_write_failed:
                        self._save()
                    self.reconcile(price)
                    self.step(price)
                    self._save()
                except Exception as e:  # noqa: BLE001 — Resilience: retry network/API failures.
                    log(f"  ! [STRAT] eroare ({e.__class__.__name__}: {e}) — reincerc")
                    _delay(self.p.check_minutes * 60)
                    continue

                avg = self._avg_cost()
                net = self.s.get("realized_net_usd", 0.0)
                fees = self.s.get("fees_usd", 0.0)
                if avg:
                    log(f"  [STRAT] price={price:.2f}  qty={self.s['qty']:.2f}  avg={avg:.2f}  "
                        f"desf={self.s['spent_cash']:.0f}{self.ccy}  "
                        f"NET={net:+.2f}USD (brut {self.s['realized_pnl_usd']:+.2f}, fee {fees:.2f})  "
                        f"ord={len(self.s['orders'])}")
                else:
                    log(f"  [STRAT] price={price:.2f}  qty=0  "
                        f"NET={net:+.2f}USD (brut {self.s['realized_pnl_usd']:+.2f}, fee {fees:.2f})  "
                        f"(waiting for an entry)")
                _delay(self.p.check_minutes * 60)
        except KeyboardInterrupt:
            log("  [STRAT] stopped manually.")
            self._save()
