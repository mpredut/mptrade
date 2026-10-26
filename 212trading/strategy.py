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
if sys.path[:1] != [_HERE]:
    if _HERE in sys.path:
        sys.path.remove(_HERE)
    sys.path.insert(0, _HERE)
for _mod_name in ("ipo_common", "ipo_notify", "market_data", "t212_client"):
    _mod_file = getattr(sys.modules.get(_mod_name), "__file__", None)
    if _mod_file and os.path.dirname(os.path.abspath(_mod_file)) != _HERE:
        sys.modules.pop(_mod_name, None)

from ipo_common import (
    log as _log, are_close, required_env, required_float_env,
    required_int_env, required_bool_env,
)
from ipo_notify import notify
from market_data import get_eur_usd, get_usd_ron, get_price_usd, t212_to_yahoo, trend_slope_pct
from t212_client import T212Client
from providers.base import (
    extract_order_id,
    extract_order_price,
    extract_order_side,
    extract_orig_qty,
)
from providers.execution_audit import AuditedStrategyExecutor, ExecutionAudit, new_intent_id
from providers.strategy_executor import OrderNotFound, ProviderError
from providers.t212_provider import T212Provider
from strategies.state_store import JsonStateStore
from strategies.position_exit_manager import (
    PositionExitManager,
    PositionExitConfig,
    PositionSnapshot,
    ExitAction,
    ReentrySnapshot,
)
from order_rollback import restore_cancelled_orders
from intelligence.internal.state.volatility import (
    resolve_cached_volatility_1h,
    resolve_cached_volatility_1h_pct,
    vol_pct_to_ratio,
)

FX_FEE_PCT = 0.15  # T212 currency-conversion fee per direction.


def log(msg: str) -> None:
    """Log with the asset's thread name: t212_bot runs one thread per asset into one log."""
    name = threading.current_thread().name
    _log(msg if name == "MainThread" else f"[{name}] {msg}")
_HERE = os.path.dirname(os.path.abspath(__file__))


def state_path_for(ticker: str) -> str:
    if isinstance(ticker, bool) or not isinstance(ticker, str):
        safe = "unknown"
    else:
        safe = "".join(c for c in ticker if c.isalnum() or c in "._-").strip("._-") or "unknown"
    return os.path.join(_HERE, f".state_{safe}.json")


def _clean_pos_float(val: object) -> float | None:
    if isinstance(val, bool) or val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError, OverflowError):
        return None
    return f if (math.isfinite(f) and f > 0.0) else None


def _clean_nonneg_float(val: object, default: float = 0.0) -> float:
    if isinstance(val, bool) or val is None:
        return default
    try:
        f = float(val)
    except (TypeError, ValueError, OverflowError):
        return default
    return f if (math.isfinite(f) and f >= 0.0) else default


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
    if isinstance(min_pct, bool) or isinstance(max_pct, bool) or isinstance(steps, bool):
        raise ValueError("generate_tp_ladder parameters cannot be bool")
    try:
        min_f = float(min_pct)
        max_f = float(max_pct)
        steps_f = float(steps)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid generate_tp_ladder parameters: {exc}") from exc
    if not (math.isfinite(min_f) and math.isfinite(max_f) and math.isfinite(steps_f) and steps_f.is_integer()):
        raise ValueError("generate_tp_ladder requires finite min_pct, max_pct, and integer steps")
    steps_i = int(steps_f)
    if steps_i < 1:
        return []
    if steps_i > 1 and max_f < min_f:
        raise ValueError("generate_tp_ladder requires max_pct >= min_pct")
    if steps_i == 1:
        return [(round(min_f, 1), 1.0)]
    steps = steps_i
    curve_lower = str(curve or "linear").strip().lower()
    if curve_lower not in (
        "linear", "uniform", "bell", "gaussian", "normal", "pyramid",
        "front_heavy", "back_heavy", "geometric",
    ):
        raise ValueError(f"unsupported generate_tp_ladder curve: {curve!r}")

    if curve_lower == "geometric":
        r_start = 1.0 + min_f / 100.0
        r_end = 1.0 + max_f / 100.0
        if r_start <= 0.0 or r_end <= 0.0:
            raise ValueError("geometric generate_tp_ladder requires min_pct > -100 and max_pct > -100")
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
    held_f = _clean_pos_float(held)
    if not remaining or held_f is None or held_f <= 1e-9:
        return []
    min_q = _clean_pos_float(min_tranche_qty) or 0.01
    max_viable_rungs = max(1, int(held_f / max(1e-6, min_q)))
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
    sl_disaster_buffer_max_pct: float = 15.0  # Max ceiling cap for disaster buffer (15% default)
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
        mode = required_env("STRATEGY_MODE", e).strip().lower()
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
                    steps_f = float(p_steps)
                    if not math.isfinite(steps_f) or not steps_f.is_integer():
                        raise ValueError("non-integer or non-finite steps")
                    _ladder = generate_tp_ladder(float(p_min), float(p_max), int(steps_f), curve=ladder_curve)
                except (ValueError, OverflowError) as exc:
                    raise ValueError(f"Invalid STRAT_TP_LADDER: {raw_ladder!r}") from exc
            else:
                for _part in raw_ladder.split(","):
                    _part = _part.strip()
                    if not _part:
                        continue
                    if ":" not in _part:
                        raise ValueError(f"Invalid STRAT_TP_LADDER: {raw_ladder!r}")
                    _lvl, _frac = _part.split(":", 1)
                    try:
                        lvl_f = float(_lvl)
                        frac_f = float(_frac) / 100.0
                        if not (math.isfinite(lvl_f) and math.isfinite(frac_f)) or lvl_f <= 0.0 or frac_f <= 0.0:
                            raise ValueError("non-finite or non-positive ladder rung")
                        _ladder.append((lvl_f, frac_f))
                    except ValueError as exc:
                        raise ValueError(f"Invalid STRAT_TP_LADDER: {raw_ladder!r}") from exc
                if sum(f for _, f in _ladder) > 1.0 + 1e-6:
                    raise ValueError(f"Invalid STRAT_TP_LADDER (fractions exceed 100%): {raw_ladder!r}")
        elif e.get("STRAT_TP_LADDER_MIN") and e.get("STRAT_TP_LADDER_MAX"):
            p_min = required_float_env("STRAT_TP_LADDER_MIN", e)
            p_max = required_float_env("STRAT_TP_LADDER_MAX", e)
            p_steps = required_int_env("STRAT_TP_LADDER_STEPS", e)
            _ladder = generate_tp_ladder(p_min, p_max, p_steps, curve=ladder_curve)
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
            try:
                _mdb_f = float(_mdb)
                if not math.isfinite(_mdb_f) or _mdb_f < 0 or not _mdb_f.is_integer():
                    raise ValueError("invalid STRAT_MAX_DCA_BUYS")
                _max_dca = int(_mdb_f)
            except (ValueError, OverflowError) as exc:
                raise ValueError(f"Invalid STRAT_MAX_DCA_BUYS: {_mdb!r}") from exc
        else:
            raise ValueError("Missing or empty mandatory setting: STRAT_MAX_DCA_BUYS")
        return cls(
            currency           = required_env("STRAT_CURRENCY", e).strip().upper(),
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
            reentry_hybrid_enabled = required_bool_env("STRAT_REENTRY_HYBRID_ENABLED", e),
            reentry_peak_relative = required_bool_env("STRAT_REENTRY_PEAK_RELATIVE", e),
            reentry_pullback_pct = required_float_env("STRAT_REENTRY_PULLBACK_PCT", e),
            reentry_pullback_adaptive = required_bool_env("STRAT_REENTRY_PULLBACK_ADAPTIVE", e),
            reentry_pullback_k = required_float_env("STRAT_REENTRY_PULLBACK_K", e),
            reentry_pullback_min = required_float_env("STRAT_REENTRY_PULLBACK_MIN", e),
            reentry_pullback_max = required_float_env("STRAT_REENTRY_PULLBACK_MAX", e),
            reentry_ttl_hours = required_float_env("STRAT_REENTRY_TTL_HOURS", e),
            sl_mhdsl_enabled = required_bool_env("STRAT_SL_MHDSL_ENABLED", e),
            sl_disaster_buffer_ratio = required_float_env("STRAT_SL_DISASTER_BUFFER_RATIO", e),
            sl_disaster_buffer_max_pct = required_float_env("STRAT_SL_DISASTER_BUFFER_MAX_PCT", e),
            sl_trend_defer_max_sec = required_float_env("STRAT_SL_TREND_DEFER_MAX_SEC", e),
            sl_bounce_min_recovery_pct = required_float_env("STRAT_SL_BOUNCE_MIN_RECOVERY_PCT", e),
            sl_bounce_pullback_pct = required_float_env("STRAT_SL_BOUNCE_PULLBACK_PCT", e),
            sl_structural_break_enabled = required_bool_env("STRAT_SL_STRUCTURAL_BREAK_ENABLED", e),
            tp_mdhtp_enabled = required_bool_env("STRAT_TP_MDHTP_ENABLED", e),
            tp_stagnation_max_sec = required_float_env("STRAT_TP_STAGNATION_MAX_SEC", e),
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
        "sl_episode_state": {},
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
        "sl_pending_exit": None, # Persistent MHDSL exit intent awaiting order cancellation/execution.
    }


def _sanitize_t212_state(st: dict) -> dict:
    """Ensure persisted or injected state has valid types for all structural and numeric fields in-place."""
    defaults = _new_state()
    if not isinstance(st, dict):
        return defaults
    for k, v in defaults.items():
        st.setdefault(k, v)
    for k in ("qty", "cost_usd", "spent_cash", "fees_usd", "pos_peak"):
        st[k] = _clean_nonneg_float(st.get(k), 0.0)
    for k in ("realized_pnl_usd", "realized_net_usd"):
        raw_v = st.get(k)
        if isinstance(raw_v, bool) or raw_v is None:
            st[k] = 0.0
        else:
            try:
                vf = float(raw_v)
                st[k] = vf if math.isfinite(vf) else 0.0
            except (TypeError, ValueError, OverflowError):
                st[k] = 0.0
    for k, min_i in (("cycle", 1), ("dca_buys", 0), ("loss_band", 0)):
        raw_i = st.get(k)
        if isinstance(raw_i, bool) or raw_i is None:
            st[k] = min_i
        else:
            try:
                if_val = float(raw_i)
                st[k] = max(min_i, int(if_val)) if math.isfinite(if_val) else min_i
            except (TypeError, ValueError, OverflowError):
                st[k] = min_i
    for k in ("entry_price", "last_buy_price", "last_sell_price", "last_sell_ts", "post_sell_peak", "post_sell_trough"):
        st[k] = _clean_pos_float(st.get(k))
    for k in ("sl_bounce_state", "sl_episode_state", "tp_ride_state", "reentry_state"):
        if not isinstance(st.get(k), dict):
            st[k] = {}
    for k in ("pending_submit", "tp_pending_exit", "sl_pending_exit"):
        if not isinstance(st.get(k), dict):
            st[k] = None
    raw_orders = st.get("orders")
    if isinstance(raw_orders, list):
        st["orders"] = [o for o in raw_orders if isinstance(o, dict)]
    else:
        st["orders"] = []
    raw_sold = st.get("tp_sold_levels")
    if isinstance(raw_sold, list):
        clean_sold = []
        for item in raw_sold:
            lvl_f = _clean_pos_float(item)
            if lvl_f is not None:
                clean_sold.append(lvl_f)
        st["tp_sold_levels"] = clean_sold
    else:
        st["tp_sold_levels"] = []
    for k in ("buy_backoff_until", "locked_zero_until", "tr_fail_until", "sl_fail_until"):
        if k in st:
            cleaned = _clean_nonneg_float(st.get(k), -1.0)
            if cleaned >= 0.0:
                st[k] = cleaned
            else:
                st.pop(k, None)
    if "tp_fail_until" in st:
        raw_fails = st.get("tp_fail_until")
        if isinstance(raw_fails, dict):
            st["tp_fail_until"] = {
                str(fk): _clean_nonneg_float(fv, 0.0)
                for fk, fv in raw_fails.items()
                if _clean_nonneg_float(fv, -1.0) > 0.0
            }
        else:
            st.pop("tp_fail_until", None)
    return st


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
                 execution_audit: ExecutionAudit | None = None,
                 replay_mode: bool = False,
                 observe_only: bool = False):
        self.client = client
        self.ticker = ticker
        self.yahoo_sym = params.yahoo_sym or t212_to_yahoo(ticker)
        self.p = params
        self.dry_run = dry_run
        self.observe_only = bool(observe_only)
        self.desktop = desktop
        self.ccy = params.currency
        self.replay_mode = replay_mode or (clock is not None)
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
        self.fx_to_usd = (
            self._fx_to_usd(params.currency)
            if fx_to_usd is None
            else (_clean_pos_float(fx_to_usd) or 1.0)
        )
        self.state_file = state_path_for(ticker)
        self._state_write_failed = False
        self.s = _sanitize_t212_state(initial_state if initial_state is not None else self._load())
        self.exit_manager: PositionExitManager | None = (
            self._init_exit_manager() if (self.p.sl_mhdsl_enabled or self.p.tp_mdhtp_enabled) else None
        )
        self._paper_seq = 0

    def _init_exit_manager(self) -> PositionExitManager:
        exit_cfg = PositionExitConfig(
            lost_threshold=vol_pct_to_ratio(self.p.stop_loss_pct),
            sl_disaster_buffer_ratio=self.p.sl_disaster_buffer_ratio,
            sl_disaster_buffer_max_pct=vol_pct_to_ratio(self.p.sl_disaster_buffer_max_pct),
            sl_trend_defer_max_sec=self.p.sl_trend_defer_max_sec,
            sl_bounce_min_recovery_pct=vol_pct_to_ratio(self.p.sl_bounce_min_recovery_pct),
            sl_bounce_pullback_pct=vol_pct_to_ratio(self.p.sl_bounce_pullback_pct),
            sl_structural_break_enabled=self.p.sl_structural_break_enabled,
            smart_sl_enabled=False,
            reentry_enabled=True,
            reentry_sl_bounce_pct=(
                vol_pct_to_ratio(self.p.sl_rebuy_bounce_pct)
                if self.p.sl_rebuy_enabled
                else (0.015 if self.p.sl_mhdsl_enabled else 0.0)
            ),
            reentry_drop_pct=vol_pct_to_ratio(self.p.reentry_drop_pct),
            reentry_pullback_pct=vol_pct_to_ratio(self.p.reentry_pullback_pct),
            reentry_ttl_hours=self.p.reentry_ttl_hours,
            reentry_tolerance_pct=self.p.reentry_tolerance_pct,
            reentry_bear_gate_enabled=False,
            reentry_bear_bounce_pct=0.0,
            gain_threshold=vol_pct_to_ratio(self.p.takeprofit_pct),
            tp_trend_ride_enabled=self.p.tp_mdhtp_enabled,
            tp_profit_floor_min_pct=vol_pct_to_ratio(self.p.trail_min_profit_pct) if self.p.trail_min_profit_pct > 0 else 0.0,
            tp_pullback_pct=vol_pct_to_ratio(self.p.trail_pct) if self.p.trail_pct > 0 else 0.015,
            tp_stagnation_max_sec=self.p.tp_stagnation_max_sec,
            allow_external_cache=not self.replay_mode,
        )
        mgr = PositionExitManager(
            config=exit_cfg,
            vol_resolver=self._resolve_volatility_tuple,
        )
        mgr.bind_state(self.s)
        return mgr

    def _resolve_volatility_tuple(self, symbol: str) -> tuple[float, str]:
        """Callable adapter for PositionExitManager vol_resolver."""
        return self._vol_1h_ratio()

    def _now(self) -> float:
        return float(self._clock())

    # -- valuta ----------------------------------------------------------------
    def _fx_to_usd(self, currency: str) -> float:
        """How many USD in one unit of the given currency — generic for any Yahoo currency."""
        ccy = str(currency or "USD").strip().upper()
        if ccy == "USD":
            return 1.0
        if ccy == "EUR":
            return _clean_pos_float(get_eur_usd()) or 1.08          # USD per EUR
        if ccy == "RON":
            ron_rate = _clean_pos_float(get_usd_ron()) or 4.65
            return 1.0 / ron_rate                                   # USD per RON = 1 / (RON per USD)
        rate = _clean_pos_float(get_price_usd(f"{ccy}USD=X"))       # generic: GBP, CHF, PLN...
        if rate:
            return rate
        log(f"  ! curs {ccy}/USD indisponibil — tratez sumele ca USD (1:1). "
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
        qty = _clean_pos_float(self.s.get("qty"))
        cost = _clean_pos_float(self.s.get("cost_usd"))
        if qty is None or qty <= 1e-9 or cost is None:
            return None
        avg = cost / qty
        return avg if (math.isfinite(avg) and avg > 0.0) else None

    def _qty_for_amount(self, amount: float, price: float) -> float:
        amt_f = _clean_pos_float(amount)
        px_f = _clean_pos_float(price)
        fx_f = _clean_pos_float(self.fx_to_usd)
        if amt_f is None or px_f is None or fx_f is None:
            return 0.0
        usd = amt_f * fx_f
        return round(usd / px_f, 2) if (math.isfinite(usd) and px_f > 0) else 0.0

    def _has_open(self, side: str) -> bool:
        side_u = str(side or "").strip().upper()
        pending = self.s.get("pending_submit") or {}
        return (
            str(pending.get("side") or "").strip().upper() == side_u
            or any(
                isinstance(o, dict) and str(o.get("side") or "").strip().upper() == side_u
                for o in self.s["orders"]
            )
        )

    def _find_open(self, side: str) -> dict | None:
        side_u = str(side or "").strip().upper()
        for o in self.s["orders"]:
            if isinstance(o, dict) and str(o.get("side") or "").strip().upper() == side_u:
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
        side = extract_order_side(order) or ""
        raw_qty = order.get("quantity", order.get("orderedQuantity", order.get("qty")))
        qty: float | None = None
        if raw_qty is not None and not isinstance(raw_qty, bool):
            try:
                q_val = abs(float(raw_qty))
                if math.isfinite(q_val) and q_val > 0.0:
                    qty = q_val
            except (TypeError, ValueError, OverflowError):
                pass
        if qty is None:
            qty = extract_orig_qty(order, default=0.0) or None
        raw_limit = order.get("limitPrice", order.get("limit"))
        limit: float | None = None
        if raw_limit is not None and not isinstance(raw_limit, bool):
            try:
                l_val = float(raw_limit)
                if math.isfinite(l_val) and l_val > 0.0:
                    limit = l_val
            except (TypeError, ValueError, OverflowError):
                pass
        if limit is None:
            limit = extract_order_price(order, default=0.0) or None
        return ticker, side, qty, limit

    def _active_matches_pending(self, order: dict, pending: dict) -> bool:
        ticker, side, qty, limit = self._active_order_identity(order)
        if ticker != self.ticker.upper() or side != str(pending.get("side") or "").upper():
            return False
        raw_expected_qty = pending.get("qty")
        try:
            expected_qty = float(raw_expected_qty) if raw_expected_qty is not None and not isinstance(raw_expected_qty, bool) else 0.0
            if not math.isfinite(expected_qty):
                expected_qty = 0.0
        except (TypeError, ValueError, OverflowError):
            expected_qty = 0.0
        if qty is None or abs(qty - expected_qty) > max(0.011, expected_qty * 1e-6):
            return False
        if pending.get("market"):
            return True
        raw_expected_limit = pending.get("limit")
        try:
            expected_limit = float(raw_expected_limit) if raw_expected_limit is not None and not isinstance(raw_expected_limit, bool) else 0.0
            if not math.isfinite(expected_limit):
                expected_limit = 0.0
        except (TypeError, ValueError, OverflowError):
            expected_limit = 0.0
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
        if len(matches) == 1 and extract_order_id(matches[0]) is not None:
            order = self._adopt_pending_submit(extract_order_id(matches[0]))
            log(f"  [STRAT] T212 submit recovered from the active orders: {order['id']}")
            return "adopted"
        if len(matches) > 1:
            log("  ! [STRAT] ambiguous T212 submit: several active orders match — keeping it pending")
            return "waiting"

        raw_before = pending.get("before_qty")
        try:
            before = float(raw_before) if raw_before is not None and not isinstance(raw_before, bool) else 0.0
            if not math.isfinite(before):
                before = 0.0
        except (TypeError, ValueError, OverflowError):
            before = 0.0
        side = str(pending.get("side") or "").upper()
        try:
            real_qty_f = float(real_qty) if real_qty is not None and not isinstance(real_qty, bool) else before
            if not math.isfinite(real_qty_f):
                real_qty_f = before
        except (TypeError, ValueError, OverflowError):
            real_qty_f = before
        delta = real_qty_f - before
        if (side == "BUY" and delta > 1e-6) or (side == "SELL" and delta < -1e-6):
            pending["portfolio_fill_observed"] = True
            self._persist_pending_submit(pending)
            return "filled"

        raw_misses = pending.get("lookup_misses")
        try:
            misses_base = int(float(raw_misses)) if raw_misses is not None and not isinstance(raw_misses, bool) and math.isfinite(float(raw_misses)) else 0
        except (TypeError, ValueError, OverflowError):
            misses_base = 0
        misses = max(0, misses_base) + 1
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
                    self.exit_manager.bind_state(self.s)
                if was_sl and self.p.sl_rebuy_enabled:
                    self.s["sl_rebuy"] = {"low": price, "sell_price": price}
                    log(f"  🟢 [STRAT] pullback re-buy ARMED after the stop-loss "
                        f"(expecting +{self.p.sl_rebuy_bounce_pct}% from the low)")
                log(f"  [STRAT] === ciclu inchis, reincep (ciclu {nxt}) ===")

    # -- plasare / anulare -----------------------------------------------------
    def _place_buy(self, amount: float, limit: float, kind: str) -> bool:
        lim_f = _clean_pos_float(limit)
        amt_f = _clean_pos_float(amount)
        if lim_f is None or amt_f is None or round(lim_f, 2) <= 0:
            log("  ! [STRAT] invalid BUY amount/limit — sar")
            return False
        limit = lim_f
        amount = amt_f
        qty = self._qty_for_amount(amount, limit)
        if qty <= 0:
            log("  ! [STRAT] qty 0 — sar")
            return False
        if self.dry_run:
            self._paper_seq += 1
            log(f"  [STRAT] [PAPER] plasez BUY {kind} {qty} @ {limit:.2f} USD (~{amount:.0f} {self.ccy})")
            self.s["orders"].append({"id": f"PAPER-{self._paper_seq}", "side": "BUY", "qty": qty,
                                     "limit": round(limit, 2), "amount": amount, "kind": kind,
                                     "ts": self._now()})
            return True
        if self.s.get("pending_submit"):
            log(f"  [STRAT] BUY {kind} deferred — a T212 submit is already pending")
            return False
        pending = self._new_pending_submit(
            side="BUY", qty=qty, limit=limit, amount=amount, kind=kind,
            level=None, market=False,
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
        qty_f = _clean_pos_float(qty)
        lim_f = _clean_pos_float(limit)
        if qty_f is None or lim_f is None or round(lim_f, 2) <= 0:
            log("  ! [STRAT] invalid SELL qty/limit — sending no order")
            return False
        limit = lim_f
        qty = round(qty_f, 2)
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
            _m = re.search(r'owned["\']?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)', message)
            if _m is None:
                log("  ! [STRAT] selling-not-owned without the owned quantity — NOT resetting the position")
                return False
            try:
                _owned = float(_m.group(1))
            except (TypeError, ValueError, OverflowError):
                return False
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
                return bool(self._place_sell(p["qty"], p["limit"], level=p.get("level"), kind=p.get("kind", "TP")))
            elif p["side"] == "BUY":
                raw_ord = p.get("raw") if isinstance(p.get("raw"), dict) else p
                raw_amount = raw_ord.get("amount")
                usd_notional = p["qty"] * p["limit"]
                fallback_amt = (usd_notional / self.fx_to_usd) if self.fx_to_usd > 0 else usd_notional
                amount = fallback_amt
                if raw_amount is not None and not isinstance(raw_amount, bool):
                    try:
                        cand_amt = float(raw_amount)
                        if math.isfinite(cand_amt) and cand_amt > 0.0:
                            amount = cand_amt
                    except (TypeError, ValueError, OverflowError):
                        pass
                return bool(self._place_buy(amount, p["limit"], kind=p.get("kind", "ENTRY")))
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
            lim_px = round(avg * (1 + lvl / 100.0), 2)
            if q > 0 and lim_px > 0:
                desired[lvl] = (q, lim_px)
        open_sells = {o.get("level"): o for o in self.s["orders"]
                      if o["side"] == "SELL" and o.get("level") is not None}
        cancelled_by_level = {}
        # Cancel unwanted orders or orders whose price/quantity changed after DCA.
        for lvl, o in list(open_sells.items()):
            d = desired.get(lvl)
            if d is None or d[1] <= 0 or abs(o["limit"] - d[1]) / d[1] > 0.001 or abs(o["qty"] - d[0]) > 1e-6:
                cancelled_by_level[lvl] = dict(o)
                if self._cancel_specific(o) and o not in self.s["orders"]:
                    open_sells.pop(lvl, None)
        # Place missing levels. Back off for 30 minutes after persistent tranche failure
        # instead of retrying on every tick. Defer and break if a submit is already in flight.
        now = self._now()
        if not isinstance(self.s.get("tp_fail_until"), dict):
            self.s["tp_fail_until"] = {}
        fails = self.s["tp_fail_until"]
        for lvl, (q, lim) in desired.items():
            if lvl in open_sells:
                continue
            if now < _clean_nonneg_float(fails.get(str(lvl)), 0.0):
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
        if str(order.get("side") or "").strip().upper() != "BUY" or order.get("market"):
            return False
        order_ts = _clean_nonneg_float(order.get("ts"), 0.0)
        order_lim = _clean_pos_float(order.get("limit"))
        if order_lim is None:
            return False
        if (self._now() - order_ts) / 60 <= self.p.order_ttl_min:
            return False
        fresh_limit = price * (1 - self.p.entry_discount_pct / 100)
        return fresh_limit > order_lim * 1.003

    # -- Reconciliation --------------------------------------------------------
    def _remove_order(self, o: dict) -> None:
        if o in self.s["orders"]:
            self.s["orders"].remove(o)

    def reconcile(self, price: float) -> bool:
        px = _clean_pos_float(price)
        if px is None:
            return False
        if self.dry_run:
            self._reconcile_paper(px)
            return True
        return self._reconcile_real(px)

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
            if not isinstance(p, dict):
                continue
            if str(p.get("ticker", "")).upper() == self.ticker.upper():
                raw_q = p.get("quantity")
                raw_avg = p.get("averagePrice")
                if isinstance(raw_q, bool) or isinstance(raw_avg, bool):
                    return None
                try:
                    q_f = float(0.0 if raw_q is None else raw_q)
                    avg_f = float(0.0 if raw_avg is None else raw_avg)
                except (TypeError, ValueError, OverflowError):
                    return None
                if not (math.isfinite(q_f) and q_f >= 0.0 and math.isfinite(avg_f) and avg_f >= 0.0):
                    return None
                return q_f, avg_f
        return 0.0, 0.0   # No holding.

    def _active_order_ids(self, orders: list[dict] | None = None) -> set | None:
        if orders is None:
            orders = self.client.list_active_orders()
        if orders is None:
            return None
        return {
            str(o.get("id")) for o in orders
            if isinstance(o, dict) and o.get("id") is not None and self._active_order_identity(o)[0] == self.ticker.upper()
        }

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
            raw_vals = (
                status.filled_qty,
                status.cost,
                status.fee,
                order.get("applied_fill_qty"),
                order.get("applied_fill_cost"),
                order.get("applied_fill_fee"),
            )
            if any(isinstance(v, bool) for v in raw_vals):
                continue
            try:
                cumulative_qty = float(status.filled_qty or 0.0)
                cumulative_cost = float(status.cost or 0.0)
                cumulative_fee = float(status.fee or 0.0)
                applied_qty = float(order.get("applied_fill_qty") or 0.0)
                applied_cost = float(order.get("applied_fill_cost") or 0.0)
                applied_fee = float(order.get("applied_fill_fee") or 0.0)
            except (TypeError, ValueError, OverflowError):
                continue
            if not all(math.isfinite(v) for v in (cumulative_qty, cumulative_cost, cumulative_fee, applied_qty, applied_cost, applied_fee)):
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

    def _reconcile_real(self, price: float) -> bool:
        real = self._portfolio_position()
        if real is None:
            # Debounce: log initially and every ten minutes, not on every tick.
            now = self._now()
            last = getattr(self, "_pf_unavail_logged", 0)
            if now - last > 600:
                log("  [STRAT] portofoliu indisponibil — sar reconcilierea (suprima repetatele 10 min)")
                self._pf_unavail_logged = now
            return False
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
        if real_qty > prev_qty + 1e-6 and self._now() < _clean_nonneg_float(self.s.get("locked_zero_until"), 0.0):
            log("  [STRAT] adoptie ignorata — portfolio stale (not-owned recent, lock activ)")
            return False
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
                return True
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
                    raw_m = o.get("lookup_misses")
                    try:
                        m_base = 0 if (raw_m is None or isinstance(raw_m, bool)) else max(0, int(float(raw_m)))
                    except (TypeError, ValueError, OverflowError):
                        m_base = 0
                    misses = m_base + 1
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
        has_open_sells = self._has_open("SELL")
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
            saved_pending_submit = self.s.get("pending_submit")
            self.s = _new_state()
            self.s["pending_submit"] = saved_pending_submit
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
                self.exit_manager.bind_state(self.s)
            if was_sl and self.p.sl_rebuy_enabled:   # Rebuy on a bounce, not below the sale, to catch recovery.
                self.s["sl_rebuy"] = {"low": exit_price, "sell_price": exit_price}
                log(f"  🟢 [STRAT] pullback re-buy ARMED after the stop-loss (expecting +{self.p.sl_rebuy_bounce_pct}% from the low)")
            log(f"  [STRAT] === ciclu inchis, reincep (ciclu {nxt}) ===")
        return True

    # -- decision step ---------------------------------------------------------
    def _execute_protective_exit(
        self,
        kind: str,
        price: float,
        label: str,
        fail_key: str,
        sell_qty: float | None = None,
    ) -> bool:
        """Cancel existing non-protective orders and submit a protective market sell.

        Returns True if the protective exit order was successfully placed or is
        already resting active. Returns False if submission was deferred or failed
        (caller should not alert yet).
        """
        existing = next((o for o in self.s["orders"] if o.get("kind") == kind), None)
        if existing is None or (not existing.get("market") and price < existing["limit"]):
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
            placed = self._place_sell(
                target_qty, round(price, 2), kind=kind, market=True,
            )
            if not placed:
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
        peak = _clean_pos_float(self.s.get("pos_peak")) or 0.0
        if not peak or peak <= 0:
            self.s["pos_peak"] = peak = max(avg or price, price)
        if price > peak:
            self.s["pos_peak"] = peak = price   # Track the high while holding the position.
            self.s.pop("tr_fail_until", None)
            self.s["tr_alerted"] = False

        # WARM-UP: keep trailing inactive until price rises trail_min_profit_pct above average.
        # This protects gains without selling on ordinary dips during accumulation;
        # DCA manages dips, and STRAT_STOP_LOSS_PCT / MHDSL handles catastrophic loss.
        # When trail_min_profit_pct == 0, trailing arms immediately on entry.
        if not self.s.get("tr_armed"):
            minp = self.p.trail_min_profit_pct
            if minp > 0 and (not avg or price < avg * (1 + minp / 100.0)):
                return False                    # Warming up: do not sell; let DCA and MHDSL operate.
            self.s["tr_armed"] = True
            self.s["pos_peak"] = peak = price   # Reset peak anchor to activation price
            log(f"  [STRAT] trailing ARMED: price {price:.2f} >= avg{(avg or 0):.2f}+{minp}% — protecting from now on (-{self.p.trail_pct}% from the peak)")

        drop_pct = (peak - price) / peak * 100 if peak > 0 else 0.0
        if drop_pct < self.p.trail_pct:
            return False
        if self._now() < _clean_nonneg_float(self.s.get("tr_fail_until"), 0.0):
            return False
        # The peak drawdown reached the threshold: sell all once, replacing only a stale limit.
        if self._execute_protective_exit("TR", price, "TRAILING", "tr_fail_until"):
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
        if not self.replay_mode:
            cached = resolve_cached_volatility_1h(self.ticker)
            if cached is not None:
                return cached, "cache"
        return 0.0, "none"

    def _create_position_snapshot(
        self,
        price: float,
        avg: float,
        held: float,
        include_vol: bool = True,
        include_trend: bool = True,
    ) -> PositionSnapshot:
        """Create a standardized PositionSnapshot using T212 state and time source."""
        trend_up = False
        if include_trend and self._trend_slope_provider:
            try:
                slope = self._trend_slope_provider(self.yahoo_sym)
                if slope is not None and not isinstance(slope, bool) and math.isfinite(float(slope)) and float(slope) >= 0.0:
                    trend_up = True
            except Exception:
                pass

        if include_vol:
            vol_1h, vol_source = self._vol_1h_ratio()
        else:
            vol_1h, vol_source = 0.0, "default"
        sell_qty = self._sellable_qty(held, price)
        return PositionSnapshot(
            symbol=self.ticker,
            current_price=price,
            avg_buy_price=avg,
            net_qty=held,
            sellable_qty=sell_qty,
            current_time_s=self._now(),
            trend_up=trend_up,
            vol_1h=vol_1h,
            vol_source=vol_source,
            provider="t212",
        )

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
            in_sl = loss_pct > 0.0 and (
                loss_pct >= self.p.stop_loss_pct
                or are_close(loss_pct, self.p.stop_loss_pct, self.p.reentry_tolerance_pct)
            )
            in_bounce_watch = bool(self.ticker in self.exit_manager.sl_bounce_state)
            if not in_sl:
                self.s["sl_pending_exit"] = None
            if not in_sl and not in_bounce_watch:
                if self.exit_manager.sl_episode_state:
                    self.exit_manager.reset_sl_episode(self.ticker)  # Leaving SL territory ends the episode floor lock
                self.s["sl_alerted"] = False
                self.s["sl_pending"] = False
                self.s.pop("sl_fail_until", None)
                return False

            held = self.s.get("qty", 0.0)
            snapshot = self._create_position_snapshot(price, avg, held)
            sell_qty = snapshot.sellable_qty

            raw_pending_sl = self.s.get("sl_pending_exit") if in_sl else None
            pending_sl = raw_pending_sl if isinstance(raw_pending_sl, dict) else None
            if pending_sl:
                should_exit_sl = True
                exit_reason = str(pending_sl.get("reason") or "mhdsl_pending_exit")
                decision = None
            else:
                decision = self.exit_manager.evaluate_stop_loss(
                    snapshot, custom_threshold=self.p.stop_loss_pct / 100.0,
                )
                should_exit_sl = bool(decision.action == ExitAction.SELL and decision.should_exit)
                exit_reason = decision.reason if should_exit_sl else ""

            if should_exit_sl:
                if self._now() < _clean_nonneg_float(self.s.get("sl_fail_until"), 0.0):
                    return False
                if pending_sl is None:
                    self.s["sl_pending_exit"] = {"reason": exit_reason, "time": self._now()}
                if self._execute_protective_exit("SL", price, "STOP", "sl_fail_until", sell_qty=sell_qty):
                    self.s["sl_pending_exit"] = None
                    if not self.s.get("sl_alerted"):
                        self.s["sl_alerted"] = True
                        log(f"  🛑 [STRAT] MHDSL STOP-LOSS: loss {loss_pct:.2f}% >= {self.p.stop_loss_pct}% ({exit_reason}) — SELLING EVERYTHING (cutting the loss)")
                        notify(title=f"🛑 SL {self.yahoo_sym} -{loss_pct:.1f}%",
                               body=f"loss {loss_pct:.1f}% >=threshold{self.p.stop_loss_pct}% ({exit_reason}) — selling everything",
                               source="T212", price=price, desktop=self.desktop)
                elif self.s.get("sl_fail_until"):
                    self.s["sl_pending_exit"] = None
                return True
            elif decision is not None and decision.action == ExitAction.HOLD:
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
        if self._now() < _clean_nonneg_float(self.s.get("sl_fail_until"), 0.0):
            return False
        # Anti-spam: place the stop once and replace it only if price moved below its limit.
        if self._execute_protective_exit("SL", price, "STOP", "sl_fail_until"):
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
        if not isinstance(rb, dict) or not rb:
            return
        prev_low = _clean_pos_float(rb.get("low"))
        rb["low"] = min(prev_low, price) if prev_low is not None else price  # Track the post-stop low.
        target = rb["low"] * (1 + self.p.sl_rebuy_bounce_pct / 100.0)
        if price < target and not are_close(price, target, self.p.reentry_tolerance_pct):
            return                                            # Wait until the bounce is confirmed.
        if self.s["spent_cash"] + self.p.entry_amount > self.p.max_budget:
            log(f"  [STRAT] re-buy SL cancelled — budget cap {self.p.max_budget:.0f} {self.ccy} reached")
            return
        disc = 1 - self.p.entry_discount_pct / 100
        if not self._place_buy(self.p.entry_amount, price * disc, kind="ENTRY"):
            return
        self.s.pop("sl_rebuy", None)                          # Consume the one-tranche reentry arm.
        self.s["last_exit_kind"] = "TP"
        if self.exit_manager is not None:
            self.exit_manager.reentry_state.pop(self.ticker, None)
        log(f"  🟢 [STRAT] RE-BUY after the SL: a +{self.p.sl_rebuy_bounce_pct}% pullback from the low {rb['low']:.2f} — re-entering ENTRY")
        notify(title=f"🟢 {self.yahoo_sym} RE-BUY after the stop-loss",
               body=(f"recul +{self.p.sl_rebuy_bounce_pct}% de la min {rb['low']:.2f} — "
                     f"re-entering with {self.p.entry_amount:.0f}{self.ccy}"),
               source="T212", price=price, desktop=self.desktop)

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
                v = self._shadow_vol_1h()
                if v is not None and math.isfinite(float(v)) and float(v) > 0:
                    vol = float(v)
            except Exception:
                vol = None
        if vol is None and not self.replay_mode:
            vol = resolve_cached_volatility_1h_pct(self.ticker)
        if vol is None:
            return self.p.reentry_pullback_pct, "fixed (fallback)"
        val = max(self.p.reentry_pullback_min, min(self.p.reentry_pullback_max, self.p.reentry_pullback_k * vol))
        return val, f"adaptive (vol {vol:.2f}% x k={self.p.reentry_pullback_k} -> {val:.2f}%)"

    def _evaluate_reentry_gate(self, price: float) -> tuple[bool, str]:
        """Evaluate re-entry gate conditions symmetrically with PositionExitManager or legacy fallback."""
        lsp = _clean_pos_float(self.s.get("last_sell_price"))
        if not lsp:
            return True, "no_last_sell"

        cur_peak = _clean_pos_float(self.s.get("post_sell_peak")) or lsp
        self.s["post_sell_peak"] = max(cur_peak, price)
        cur_trough = _clean_pos_float(self.s.get("post_sell_trough")) or lsp
        self.s["post_sell_trough"] = min(cur_trough, price)

        last_sell_ts = _clean_pos_float(self.s.get("last_sell_ts"))
        if self.p.reentry_ttl_hours > 0 and last_sell_ts:
            if (self._now() - last_sell_ts) >= self.p.reentry_ttl_hours * 3600.0:
                self.s.pop("sl_rebuy", None)

        # Rebuy on a bounce after a catastrophic stop loss, as on Binance/Kraken.
        # While armed, it takes priority over reentry below the previous sale.
        if self.s.get("sl_rebuy"):
            self._handle_sl_rebuy(price)
            return False, "handling_sl_rebuy"

        if self.exit_manager is None:
            self.exit_manager = self._init_exit_manager()

        mgr = self.exit_manager
        mgr.config.reentry_drop_pct = vol_pct_to_ratio(self.p.reentry_drop_pct)
        mgr.config.reentry_sl_bounce_pct = (
            vol_pct_to_ratio(self.p.sl_rebuy_bounce_pct)
            if self.p.sl_rebuy_enabled
            else (0.015 if self.p.sl_mhdsl_enabled else 0.0)
        )
        mgr.config.reentry_ttl_hours = self.p.reentry_ttl_hours
        mgr.config.reentry_tolerance_pct = self.p.reentry_tolerance_pct
        mgr.config.reentry_hybrid_enabled = self.p.reentry_hybrid_enabled
        mgr.config.reentry_peak_relative = self.p.reentry_peak_relative
        mgr.config.reentry_bear_gate_enabled = False
        mgr.config.reentry_bear_bounce_pct = 0.0

        trend_up = bool(self.p.reentry_peak_relative)
        trend_down = False
        if not trend_up and self._trend_slope_provider:
            try:
                slope = self._trend_slope_provider(self.yahoo_sym)
                if slope is not None and not isinstance(slope, bool) and math.isfinite(float(slope)):
                    slope_f = float(slope)
                    if slope_f >= 0.0:
                        trend_up = True
                    elif slope_f < (-self.p.dca_trend_gate_pct if self.p.dca_trend_gate_pct > 0 else -0.1):
                        trend_down = True
            except Exception:
                pass

        last_exit_kind = "STOP" if self.s.get("last_exit_kind") == "STOP" else "TP"
        if (last_exit_kind != "STOP" and self.p.reentry_hybrid_enabled and (trend_up or self.p.reentry_peak_relative)) or not self.replay_mode:
            pullback_pct, pb_src = self._effective_reentry_pullback_pct()
        else:
            pullback_pct = self.p.reentry_pullback_pct
        mgr.config.reentry_pullback_pct = vol_pct_to_ratio(pullback_pct)
        if self.replay_mode:
            vol_1h, vol_source = 0.0, "default"
        else:
            vol_1h, vol_source = self._vol_1h_ratio()

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
        reentry_dec = mgr.evaluate_reentry(reentry_snap)
        return reentry_dec.should_enter, reentry_dec.reason

    def step(self, price: float) -> None:
        px = _clean_pos_float(price)
        if px is None:
            return
        price = px
        held = _clean_nonneg_float(self.s.get("qty"), 0.0)
        self.s["qty"] = held
        self._check_loss_alert(price)   # Alert on deeper loss while holding; never sells.
        if self.observe_only:
            if not self.dry_run and self.s.get("orders"):
                self._cancel_all_orders()
            return
        disc = 1 - self.p.entry_discount_pct / 100

        in_backoff = self._now() < _clean_nonneg_float(self.s.get("buy_backoff_until"), 0.0)

        is_flat = held <= 1e-9 or (self._is_dust(held, price) and not self._has_open("SELL"))
        if is_flat:
            self.s["tp_pending_exit"] = None
            self.s["sl_pending_exit"] = None
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
                if self.exit_manager is not None:
                    self.exit_manager.restore_reentry_state(self.ticker)
                log(f"  [STRAT] budget cap {self.p.max_budget:.0f} {self.ccy} reached — not entering")
                return
            if not self._place_buy(self.p.entry_amount, price * disc, kind="ENTRY"):
                if self.exit_manager is not None:
                    self.exit_manager.restore_reentry_state(self.ticker)
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
                            self._cancel_open("SELL")
                    else:
                        self.exit_manager.config.tp_trend_ride_enabled = True
                        self.exit_manager.config.tp_profit_floor_min_pct = (
                            (self.p.trail_min_profit_pct or 0.0) / 100.0
                        )
                        self.exit_manager.config.tp_pullback_pct = (
                            (self.p.trail_pct / 100.0) if self.p.trail_pct > 0 else 0.015
                        )
                        self.exit_manager.config.tp_stagnation_max_sec = self.p.tp_stagnation_max_sec

                        snapshot = self._create_position_snapshot(
                            price, avg, held,
                            include_vol=not self.replay_mode,
                            include_trend=in_tp_zone,
                        )

                        decision = self.exit_manager.evaluate_take_profit(
                            snapshot, custom_threshold=tp_thresh
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

                if pending_exit:
                    existing_exit = next(
                        (o for o in self.s.get("orders", [])
                         if o.get("market") and str(o.get("side", "")).upper() == "SELL" and o.get("kind") in ("TP", "SL")),
                        None
                    )
                    if existing_exit:
                        non_exit_orders = [o for o in list(self.s["orders"])
                                           if not (o.get("market") and str(o.get("side", "")).upper() == "SELL" and o.get("kind") in ("TP", "SL"))]
                        for o in non_exit_orders:
                            self._cancel_specific(o)

                        if self.s["qty"] <= 1e-6:
                            log("  [STRAT] MDHTP exit: position closed — clearing exit intent")
                            self.s["tp_pending_exit"] = None
                        else:
                            log(f"  [STRAT] MDHTP exit: existing market exit order {existing_exit.get('id')} in progress — awaiting fill reconciliation")
                        return

                    if not self._cancel_all_orders():
                        log("  ! [STRAT] MDHTP exit deferred: at least one order could not be cancelled")
                        return
                    pending_orders = [o for o in self.s["orders"]
                                      if not (o.get("market") and str(o.get("side", "")).upper() == "SELL" and o.get("kind") in ("TP", "SL"))]
                    if pending_orders:
                        log(f"  [STRAT] MDHTP exit deferred: waiting for {len(pending_orders)} existing order(s) to reconcile cancellation")
                        return
                    if self.s.get("pending_submit"):
                        log("  [STRAT] MDHTP exit deferred: waiting for pending submit to reconcile")
                        return

                    fresh_held = self.s["qty"]
                    raw_sf = _clean_pos_float(pending_exit.get("sell_fraction"))
                    sell_fraction = min(1.0, raw_sf) if raw_sf is not None else 1.0
                    target_qty = fresh_held * sell_fraction if sell_fraction < 1.0 else fresh_held
                    sell_qty = self._sellable_qty(target_qty, price)
                    if sell_qty <= 0 or self._is_dust(sell_qty, price):
                        log("  [STRAT] MDHTP exit: no sellable quantity remaining — clearing exit intent")
                        self.s["tp_pending_exit"] = None
                        return

                    exit_px = round(price * 0.999, 2)
                    if self._place_sell(sell_qty, exit_px, kind="TP", market=True):
                        if pending_exit.get("action") == ExitAction.PARTIAL_SELL.value and self.exit_manager is not None:
                            self.exit_manager.record_hard_tp(self.ticker, tick_time)
                        log(f"  [STRAT] MDHTP EXIT ({pending_exit.get('reason')}) -> reference {exit_px} ({sell_qty} shares)")
                        self.s["tp_pending_exit"] = None
                        return
                    else:
                        log(f"  ! [STRAT] MDHTP EXIT order refused/failed ({pending_exit.get('reason')}) — intent retained for retry")
                        return
                elif decision is not None and decision.reason == "not_in_tp_territory":
                    self.s["trail_peak"] = None
                    self._cancel_open("SELL")
                elif decision is not None and decision.action == ExitAction.HOLD:
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
            elif self.p.tp_ladder:
                self._manage_tp_ladder(held, avg)        # Scale out in ladder steps.
            else:
                target = avg * (1 + self.p.takeprofit_pct / 100)
                if target > 0:
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
            or (self._now() < _clean_nonneg_float(self.s.get("sl_fail_until"), 0.0))
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
                if (
                    slope is not None
                    and not isinstance(slope, bool)
                    and math.isfinite(float(slope))
                    and float(slope) < -self.p.dca_trend_gate_pct
                ):
                    self._dca_gate_until = self._now() + 300
                    log(f"  [STRAT] DCA BLOCKED de trend: slope {float(slope):+.3f}%/bar "
                        f"< -{self.p.dca_trend_gate_pct}% (downtrend) — re-verific in 5 min")
                    return
            log(f"  [STRAT] dip: {price:.2f} <= {self.s['last_buy_price']:.2f}"
                f"×(1-{self.p.dca_drop_pct}%) — DCA")
            self._place_buy(self.p.dca_amount, price * disc, kind="DCA")

    # -- main loop -------------------------------------------------------------
    def run(self, stop: threading.Event | None = None) -> None:
        mode = "avg_tp" if self.p.enable_takeprofit else "dca_only"
        exec_banner = "[OBSERVE ONLY — AI MANAGED]" if self.observe_only else ("[PAPER]" if self.dry_run else "⚠ REAL — BANI ADEVARATI")
        log("  === STRATEGIE PORNITA ===")
        log(f"      instrument : {self.ticker}  (price through {self.yahoo_sym})")
        log(f"      mod        : {mode}   {exec_banner}")
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
            eff_sec = max(1.0, float(delay_sec)) if (_clean_pos_float(delay_sec) is not None) else 60.0
            if stop is not None:
                stop.wait(eff_sec)
            else:
                time.sleep(eff_sec)

        try:
            while not (stop is not None and stop.is_set()):
                price = get_price_usd(self.yahoo_sym)
                if price is None:
                    log("  [STRAT] price unavailable — retrying")
                    _delay(self.p.check_minutes * 60)
                    continue
                try:
                    # After a disk error, make no new decisions until the current
                    # snapshot can be persisted again.
                    if self._state_write_failed:
                        self._save()
                        if self._state_write_failed:
                            _delay(self.p.check_minutes * 60)
                            continue
                    if self.reconcile(price) is False:
                        _delay(self.p.check_minutes * 60)
                        continue
                    self.step(price)
                    self._save()
                except Exception as e:  # noqa: BLE001 — Resilience: retry network/API failures.
                    log(f"  ! [STRAT] error ({e.__class__.__name__}: {e}) — retrying")
                    _delay(self.p.check_minutes * 60)
                    continue

                avg = self._avg_cost()
                net = self.s.get("realized_net_usd", 0.0)
                fees = self.s.get("fees_usd", 0.0)
                if avg:
                    log(f"  [STRAT] price={price:.2f}  qty={self.s['qty']:.2f}  avg={avg:.2f}  "
                        f"deployed={self.s['spent_cash']:.0f}{self.ccy}  "
                        f"NET={net:+.2f}USD (gross {self.s['realized_pnl_usd']:+.2f}, fee {fees:.2f})  "
                        f"ord={len(self.s['orders'])}")
                else:
                    log(f"  [STRAT] price={price:.2f}  qty=0  "
                        f"NET={net:+.2f}USD (gross {self.s['realized_pnl_usd']:+.2f}, fee {fees:.2f})  "
                        f"(waiting for an entry)")
                _delay(self.p.check_minutes * 60)
        except KeyboardInterrupt:
            log("  [STRAT] stopped manually.")
            self._save()
