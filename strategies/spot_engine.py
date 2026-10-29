#!/usr/bin/env python3
"""Venue-independent spot trading engine with multi-regime DCA, take-profit, trailing exits, and surge guards.

The engine receives all data and executes every order through ``StrategyExecutor``.
Venue launchers inject the provider, state directory, and presentation details while
live and replay share one implementation of the financial rules.
"""

from __future__ import annotations

import math
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

from alertnotifiers import bind_notify
from botcore import (
    _parse_bool_flag, are_close, log, defined_env, required_env,
    required_float_env, required_int_env, required_bool_env,
    safe_float, safe_int, safe_nonneg_float, safe_nonneg_int, safe_pos_float, safe_pos_int,
)
from market_regime import MarketRegimeDecision, MarketRegimeService
from providers.base import _clean_single_order_id as _clean_order_id, normalize_side
from providers.execution_audit import intent_client_order_id
from providers.strategy_executor import ProviderError
from order_retry import (
    StrategyExecutorLifecycleApi,
    TrackedOrderLifecycle,
)

from intelligence.internal.state.volatility import (
    adaptive_thresholds,
    resolve_cached_volatility_1h,
    resolve_cached_volatility_1h_pct,
    vol_pct_to_ratio,
)

from . import spot_rules as sr
from .state_store import JsonStateStore
from .position_exit_manager import (
    ExitAction,
    PositionExitConfig,
    PositionExitManager,
    PositionSnapshot,
    ReentrySnapshot,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LEGACY_STATE_DIR = os.path.join(_ROOT, "kraken")
_DEFAULT_FEE_NOTE = "fee Kraken ~0.26% taker / ~0.16% maker per leg"
_PLACEMENT_BACKOFF_CAP_SEC = 3600.0
_REGIME_BAR_SETTLE_MAX_SEC = 900.0


notify = bind_notify(("SYMBOL_LABEL", "KRAKEN_PAIR"), "CRYPTO")


_clean_pos_float = lambda val: safe_pos_float(val, default=None)
_clean_nonneg_float = lambda val: safe_nonneg_float(val, default=None)


def state_path_for(pair: str, state_dir: str | None = None) -> str:
    """Return the state path, retaining the Kraken fallback for live compatibility."""
    if isinstance(pair, bool) or not isinstance(pair, str):
        safe = "unknown"
    else:
        safe = "".join(c for c in pair if c.isalnum() or c in "._-").strip("._-") or "unknown"
    if isinstance(state_dir, os.PathLike):
        directory = os.fspath(state_dir) or _LEGACY_STATE_DIR
    elif isinstance(state_dir, str) and state_dir.strip():
        directory = state_dir.strip()
    else:
        directory = _LEGACY_STATE_DIR
    return os.path.join(directory, f".state_{safe}.json")


@dataclass
class StratParams:
    currency: str          # Quote currency used only for presentation.
    entry_amount: float    # Entry size in quote currency.
    entry_discount_pct: float
    dca_amount: float
    dca_drop_pct: float
    check_minutes: float
    takeprofit_pct: float
    max_budget: float
    max_dca_buys: int
    enable_takeprofit: bool
    order_ttl_min: float
    stop_loss_pct: float = 0.0     # Sell everything at this loss percentage; zero disables it (spot DCA default).
    adopt_cost: float = 0.0        # Positive value adopts an existing position at this average cost.
    adopt_qty: float = 0.0         # Adopted quantity; zero reads the base-asset balance automatically.
    reentry_drop_pct: float = 2.0  # After TP, reenter only this far below the sale price; zero is immediate.
    reentry_tolerance_pct: float = 0.05  # Treat prices within tolerance of the threshold as reached.
    reentry_adaptive: bool = False # Deprecated legacy flag (dominated by hybrid re-entry).
    shadow_k_reentry: float = 2.0  # Volatility multiplier for adaptive re-entry and shadow logging.
    reentry_sl_bounce_pct: float = 1.5  # After stop-loss, reenter on a bounce from the post-sale low.
    tp_tranches: list = None       # Gradual ``(percentage, share)`` sales; empty means one full TP.
    # --- TP trend-aware (EXPERIMENTAL, default OFF) -------------------------------
    tp_trend_hold: bool = False       # Use trailing instead of a fixed TP after the target is reached.
    tp_regime_gate: bool = False      # Deprecated legacy binary gate: superseded by continuous MHDPA rules.
    tp_trend_min_pct: float = 0.5     # Minimum fitted bullish move required by the TP policy.
    tp_trail_pct: float = 2.0         # Above TP, exit after this pullback from the peak.
    tp_trail_profit_floor_pct: float = 0.0  # 0 preserves live compatibility; >0 allows MARKET trailing only
                                      # when the order reference is >= average*(1+floor%);
                                      # the MARKET hard stop retains priority.
    # --- TREND OVERLAY (combine strategies by regime; EXPERIMENTAL, default OFF) ---
    trend_overlay: bool = False       # Use top-up and trailing in confirmed uptrends; classic in ranges.
    trend_sma_n: int = 30             # SMA lookback for the optional overlay exit break.
    trend_interval: int = 240         # Common regime cadence in live and replay; 240 means 4h.
    regime_min_samples: int = 20      # Shared warm-up floor before any regime policy can act.
    trend_topup: float = 2000.0       # amount bought on trend entry; larger values capture more trend
    trend_trail_pct: float = 5.0      # Trend-position exit pullback from peak.
    trend_exit_break: bool = False    # False: trailing only; True: trailing or price below SMA.
    # --- VOLATILITY ADAPTIVE (overlay redesign: modulation, not amplification; default OFF) ---
    tp_trail_adaptive: bool = False   # A: TP trailing becomes k×vol_1h: wide in volatile trends
                                      # to ride longer, tight in chop. Warm-up safely falls back
                                      # to fixed tp_trail_pct, like adaptive reentry.
    tp_trail_k: float = 2.0           # vol_1h multiplier for adaptive trailing
    tp_trail_min: float = 1.5         # lower clamp (%) avoids exits on negligible noise
    tp_trail_max: float = 8.0         # upper clamp (%) prevents giving all profit back
    tp_trail_vol_interval: int = 240  # Minutes per volatility bar; fixed in live and replay.
    dca_trend_brake: bool = False     # Skip DCA in confirmed downtrends to reduce risk.
    dca_brake_min_pct: float = 1.5    # Minimum fitted bearish move required by the DCA policy.
    dca_spacing_growth_pct: float = 0.0  # Increase the threshold after every filled DCA;
                                         # Zero preserves byte-identical live behavior.
    # --- #2: VOLATILITY-SCALED DCA SIZING (default OFF) ---
    dca_vol_scale_k: float = 0.0      # 0=OFF. eff_dca = dca × (vol_ref/vol_1h)^k, clamp [0.3,3].
                                      # k>0 means smaller DCA in high volatility (defensive);
                                      # k<0 means larger DCA (aggressive harvesting). Safe warm-up fallback.
    dca_vol_ref: float = 2.0          # reference vol_1h percentage for scaling
    dca_vol_interval: int = 240       # Identical OHLC cadence in live and replay.
    # --- PERCENTAGE SIZING (optional; all four zero values preserve fixed sizing) ---
    total_budget: float = 0.0   # venue's TOTAL budget, e.g. the whole Kraken account
    alloc_pct: float = 0.0      # percentage allocated to THIS coin; venue allocations total 100
    entry_pct: float = 0.0      # Entry is this percentage of the asset allocation.
    dca_pct: float = 0.0        # DCA is this percentage of the asset allocation.
    # --- HYBRID RE-ENTRY (EXPERIMENTAL, default OFF) -----------------------------
    reentry_hybrid_enabled: bool = False  # Live profiles must explicitly enable or disable hybrid re-entry.
    reentry_peak_relative: bool = False   # Force peak-relative pullback without trend filter (Test 1).
    reentry_pullback_pct: float = 1.5     # Pullback from post-sale peak required in confirmed bull trend.
    reentry_pullback_adaptive: bool = False  # Scale pullback by volatility: clamp(K_PULLBACK * vol_1h, min, max)
    reentry_pullback_k: float = 1.0       # Multiplier for volatility-adaptive pullback
    reentry_pullback_min: float = 0.8     # Minimum pullback clamp (%)
    reentry_pullback_max: float = 4.0     # Maximum pullback clamp (%)
    reentry_ttl_hours: float = 0.0        # 0 = off. After this many hours, stale sale barrier expires.
    reentry_bear_bounce_pct: float = 0.0  # In chop/bear, require this % bounce from local post-sell trough.
    # --- MULTI-HORIZON DYNAMIC PROFIT ARCHITECTURE (MHDPA, default OFF) ---
    tp_dynamic_flat: bool = False
    tp_min_pct: float = 3.0
    tp_max_pct: float = 7.0
    trend_trail_dynamic: bool = False
    trend_trail_base_pct: float = 8.0
    trend_trail_min_pct: float = 3.0
    trend_trail_ratchet_k: float = 0.5
    trend_trail_gain_threshold: float = 6.0
    fast_profit_guard: bool = False
    fast_profit_mult: float = 2.0
    fast_profit_window_min: float = 5.0
    fast_profit_drop_pct: float = 1.0
    surge_guard: bool = False
    surge_gain_pct: float = 25.0
    surge_window_hours: float = 72.0
    surge_move_pct: float = 25.0
    surge_exit_pullback_pct: float = 3.2
    surge_dynamic: bool = False
    surge_min_gain_pct: float = 24.0
    surge_max_gain_pct: float = 32.0
    surge_vol_multiplier: float = 10.0
    slow_grind_guard: bool = False
    slow_grind_days: float = 7.0
    slow_grind_min_gain_pct: float = 15.0
    slow_grind_flash_window_min: float = 15.0
    slow_grind_flash_drop_pct: float = 1.5
    slow_grind_hourly_drop_pct: float = 2.5
    # --- MULTI-HORIZON DYNAMIC STOP-LOSS (MHDSL, default ON) ---
    sl_mhdsl_enabled: bool = True
    sl_disaster_buffer_ratio: float = 0.30
    sl_disaster_vol_multiplier: float = 0.85
    sl_disaster_buffer_min_pct: float = 0.80
    sl_disaster_buffer_max_pct: float = 15.0
    sl_trend_defer_max_sec: float = 220.0
    sl_bounce_min_recovery_pct: float = 0.40
    sl_bounce_pullback_pct: float = 0.30
    sl_bounce_ratchet_enabled: bool = True
    sl_bounce_ratchet_k: float = 0.50
    sl_bounce_min_pullback_pct: float = 0.15
    sl_structural_break_enabled: bool = True
    # --- MULTI-HORIZON DYNAMIC TAKE-PROFIT (MDHTP) ---
    tp_mdhtp_enabled: bool = False
    tp_stagnation_max_sec: float = 1080.0

    def __post_init__(self):
        def finite(name: str, value) -> float:
            parsed = safe_float(value, default=None)
            if parsed is None:
                raise ValueError(f"{name} must be finite")
            return parsed

        def integer(name: str, value) -> int:
            val_f = finite(name, value)
            if not val_f.is_integer():
                raise ValueError(f"{name} must be an integer")
            return int(val_f)

        self.trend_topup = finite("STRAT_TREND_TOPUP", self.trend_topup)
        self.entry_discount_pct = finite("STRAT_ENTRY_DISCOUNT_PCT", self.entry_discount_pct)
        self.dca_drop_pct = finite("STRAT_DCA_DROP_PCT", self.dca_drop_pct)
        self.takeprofit_pct = finite("STRAT_TAKEPROFIT_PCT", self.takeprofit_pct)
        self.stop_loss_pct = finite("STRAT_STOP_LOSS_PCT", self.stop_loss_pct)
        _bool_defaults = {
            "enable_takeprofit": True,
            "reentry_adaptive": False,
            "tp_trend_hold": False,
            "tp_regime_gate": False,
            "trend_overlay": False,
            "trend_exit_break": False,
            "tp_trail_adaptive": False,
            "dca_trend_brake": False,
            "reentry_hybrid_enabled": False,
            "reentry_peak_relative": False,
            "reentry_pullback_adaptive": False,
            "tp_dynamic_flat": False,
            "trend_trail_dynamic": False,
            "fast_profit_guard": False,
            "surge_guard": False,
            "surge_dynamic": False,
            "slow_grind_guard": False,
            "sl_mhdsl_enabled": True,
            "sl_bounce_ratchet_enabled": True,
            "sl_structural_break_enabled": True,
            "tp_mdhtp_enabled": False,
        }
        for b_attr, b_def in _bool_defaults.items():
            setattr(self, b_attr, _parse_bool_flag(getattr(self, b_attr, b_def), b_def))
        if isinstance(self.tp_tranches, str):
            self.tp_tranches = _parse_tranches(self.tp_tranches)
        elif self.tp_tranches is None:
            self.tp_tranches = []

        if self.trend_topup <= 0:
            raise ValueError("STRAT_TREND_TOPUP must be > 0")
        if not 0 <= self.entry_discount_pct < 100:
            raise ValueError("STRAT_ENTRY_DISCOUNT_PCT must be in [0, 100)")
        if self.dca_drop_pct <= 0:
            raise ValueError("STRAT_DCA_DROP_PCT must be > 0")
        if self.takeprofit_pct < 0 or (self.enable_takeprofit and self.takeprofit_pct == 0):
            raise ValueError("STRAT_TAKEPROFIT_PCT must be > 0 when take-profit is enabled")
        if self.stop_loss_pct < 0:
            raise ValueError("STRAT_STOP_LOSS_PCT must be >= 0")
        self.sl_disaster_buffer_ratio = finite("STRAT_SL_DISASTER_BUFFER_RATIO", self.sl_disaster_buffer_ratio)
        self.sl_disaster_vol_multiplier = finite("STRAT_SL_DISASTER_VOL_MULTIPLIER", self.sl_disaster_vol_multiplier)
        self.sl_disaster_buffer_min_pct = finite("STRAT_SL_DISASTER_BUFFER_MIN_PCT", self.sl_disaster_buffer_min_pct)
        self.sl_disaster_buffer_max_pct = finite("STRAT_SL_DISASTER_BUFFER_MAX_PCT", self.sl_disaster_buffer_max_pct)
        self.sl_trend_defer_max_sec = finite("STRAT_SL_TREND_DEFER_MAX_SEC", self.sl_trend_defer_max_sec)
        self.sl_bounce_min_recovery_pct = finite("STRAT_SL_BOUNCE_MIN_RECOVERY_PCT", self.sl_bounce_min_recovery_pct)
        self.sl_bounce_pullback_pct = finite("STRAT_SL_BOUNCE_PULLBACK_PCT", self.sl_bounce_pullback_pct)
        self.sl_bounce_ratchet_k = finite("STRAT_SL_BOUNCE_RATCHET_K", self.sl_bounce_ratchet_k)
        self.sl_bounce_min_pullback_pct = finite("STRAT_SL_BOUNCE_MIN_PULLBACK_PCT", self.sl_bounce_min_pullback_pct)
        if self.sl_disaster_buffer_ratio < 0:
            raise ValueError("STRAT_SL_DISASTER_BUFFER_RATIO must be >= 0")
        if self.sl_disaster_vol_multiplier < 0:
            raise ValueError("STRAT_SL_DISASTER_VOL_MULTIPLIER must be >= 0")
        if self.sl_disaster_buffer_min_pct < 0:
            raise ValueError("STRAT_SL_DISASTER_BUFFER_MIN_PCT must be >= 0")
        if self.sl_disaster_buffer_max_pct <= 0:
            raise ValueError("STRAT_SL_DISASTER_BUFFER_MAX_PCT must be > 0")
        if self.sl_trend_defer_max_sec < 0:
            raise ValueError("STRAT_SL_TREND_DEFER_MAX_SEC must be >= 0")
        if self.sl_bounce_min_recovery_pct < 0:
            raise ValueError("STRAT_SL_BOUNCE_MIN_RECOVERY_PCT must be >= 0")
        if self.sl_bounce_pullback_pct < 0:
            raise ValueError("STRAT_SL_BOUNCE_PULLBACK_PCT must be >= 0")
        if self.sl_bounce_ratchet_k < 0:
            raise ValueError("STRAT_SL_BOUNCE_RATCHET_K must be >= 0")
        if self.sl_bounce_min_pullback_pct < 0:
            raise ValueError("STRAT_SL_BOUNCE_MIN_PULLBACK_PCT must be >= 0")
        self.trend_interval = integer("STRAT_TREND_INTERVAL", self.trend_interval)
        if self.trend_interval <= 0:
            raise ValueError("STRAT_TREND_INTERVAL must be > 0")
        self.regime_min_samples = integer("STRAT_REGIME_MIN_SAMPLES", self.regime_min_samples)
        regime_capacity = MarketRegimeService.horizon_sample_capacity(
            "long", self.trend_interval,
        )
        if not 3 <= self.regime_min_samples <= regime_capacity:
            raise ValueError(f"STRAT_REGIME_MIN_SAMPLES must be in [3, {regime_capacity}]")

        percentage_names = (
            "STRAT_TOTAL_BUDGET", "STRAT_ALLOC_PCT",
            "STRAT_ENTRY_PCT", "STRAT_DCA_PCT",
        )
        values = tuple(
            finite(name, value)
            for name, value in zip(percentage_names, (
                self.total_budget, self.alloc_pct, self.entry_pct, self.dca_pct,
            ))
        )
        self.total_budget, self.alloc_pct, self.entry_pct, self.dca_pct = values
        if not any(value != 0 for value in values):
            return
        if self.total_budget <= 0:
            raise ValueError("STRAT_TOTAL_BUDGET must be > 0 when percentage sizing is configured")
        for name, value in zip(percentage_names[1:], values[1:]):
            if not 0 < value <= 100:
                raise ValueError(f"{name} must be in (0, 100] when percentage sizing is configured")

    def pct_sizing_on(self) -> bool:
        return self.total_budget > 0

    def allocated_budget(self) -> float:
        return self.total_budget * self.alloc_pct / 100.0 if self.pct_sizing_on() else self.max_budget

    def effective_entry_amount(self) -> float:
        return self.allocated_budget() * self.entry_pct / 100.0 if self.pct_sizing_on() else self.entry_amount

    def effective_dca_amount_base(self) -> float:
        return self.allocated_budget() * self.dca_pct / 100.0 if self.pct_sizing_on() else self.dca_amount

    def effective_max_budget(self) -> float:
        return self.allocated_budget()


    @classmethod
    def from_dict(cls, values: dict, *, defaults: dict | None = None, warn_dropped: bool = False) -> "StratParams":
        """Build StratParams from a dict (e.g. a stored baseline report), ignoring removed fields."""
        from dataclasses import fields as _dc_fields
        base_defaults = dict(defaults) if isinstance(defaults, dict) else {}
        base_values = dict(values) if isinstance(values, dict) else {}
        merged = {**base_defaults, **base_values}
        valid = {f.name for f in _dc_fields(cls)}
        if _parse_bool_flag(warn_dropped, False):
            dropped = sorted(set(merged) - valid)
            if dropped:
                print(f"[params] ignoring parameters removed from StratParams: {dropped}")
        return cls(**{k: v for k, v in merged.items() if k in valid})

    @classmethod
    def from_env(cls, env: dict | None = None) -> "StratParams":
        """Load an explicit profile; missing financial policy is a startup error."""
        mode = required_env("STRATEGY_MODE", env).lower()
        if mode not in {"avg_tp", "dca_only"}:
            raise ValueError(f"Invalid STRATEGY_MODE: {mode!r}")
        return cls(
            currency           = required_env("STRAT_CURRENCY", env).upper(),
            entry_amount       = required_float_env("STRAT_ENTRY", env),
            entry_discount_pct = required_float_env("STRAT_ENTRY_DISCOUNT_PCT", env),
            dca_amount         = required_float_env("STRAT_DCA", env),
            dca_drop_pct       = required_float_env("STRAT_DCA_DROP_PCT", env),
            check_minutes      = required_float_env("STRAT_CHECK_MINUTES", env),
            takeprofit_pct     = required_float_env("STRAT_TAKEPROFIT_PCT", env),
            max_budget         = required_float_env("STRAT_MAX_BUDGET", env),
            max_dca_buys       = required_int_env("STRAT_MAX_DCA_BUYS", env),
            total_budget       = required_float_env("STRAT_TOTAL_BUDGET", env),
            alloc_pct          = required_float_env("STRAT_ALLOC_PCT", env),
            entry_pct          = required_float_env("STRAT_ENTRY_PCT", env),
            dca_pct            = required_float_env("STRAT_DCA_PCT", env),
            enable_takeprofit  = (mode != "dca_only"),
            order_ttl_min      = required_float_env("STRAT_ORDER_TTL_MIN", env),
            stop_loss_pct      = required_float_env("STRAT_STOP_LOSS_PCT", env),
            adopt_cost         = required_float_env("STRAT_ADOPT_COST", env),
            adopt_qty          = required_float_env("STRAT_ADOPT_QTY", env),
            reentry_drop_pct   = required_float_env("STRAT_REENTRY_DROP_PCT", env),
            reentry_tolerance_pct = required_float_env("STRAT_REENTRY_TOLERANCE_PCT", env),
            reentry_adaptive   = required_bool_env("STRAT_REENTRY_ADAPTIVE", env),
            shadow_k_reentry   = required_float_env("SHADOW_K_REENTRY", env),
            reentry_sl_bounce_pct = required_float_env("STRAT_REENTRY_SL_BOUNCE_PCT", env),
            tp_tranches        = _parse_tranches(defined_env("STRAT_TP_TRANCHES", env)),
            tp_trend_hold      = required_bool_env("STRAT_TP_TREND_HOLD", env),
            tp_regime_gate     = False,  # Retired legacy binary gate: superseded by continuous MHDPA rules.
            tp_trend_min_pct   = required_float_env("STRAT_TP_TREND_MIN_PCT", env),
            tp_trail_pct       = required_float_env("STRAT_TP_TRAIL_PCT", env),
            tp_trail_profit_floor_pct = max(
                0.0, required_float_env("STRAT_TP_TRAIL_PROFIT_FLOOR_PCT", env),
            ),
            trend_overlay      = required_bool_env("STRAT_TREND_OVERLAY", env),
            trend_sma_n        = required_int_env("STRAT_TREND_SMA_N", env),
            trend_interval     = required_int_env("STRAT_TREND_INTERVAL", env),
            regime_min_samples = required_int_env("STRAT_REGIME_MIN_SAMPLES", env),
            trend_topup        = required_float_env("STRAT_TREND_TOPUP", env),
            trend_trail_pct    = required_float_env("STRAT_TREND_TRAIL_PCT", env),
            trend_exit_break   = required_bool_env("STRAT_TREND_EXIT_BREAK", env),
            tp_trail_adaptive  = required_bool_env("STRAT_TP_TRAIL_ADAPTIVE", env),
            tp_trail_k         = required_float_env("STRAT_TP_TRAIL_K", env),
            tp_trail_min       = required_float_env("STRAT_TP_TRAIL_MIN", env),
            tp_trail_max       = required_float_env("STRAT_TP_TRAIL_MAX", env),
            tp_trail_vol_interval = required_int_env("STRAT_TP_TRAIL_VOL_INTERVAL", env),
            dca_trend_brake    = required_bool_env("STRAT_DCA_TREND_BRAKE", env),
            dca_brake_min_pct  = required_float_env("STRAT_DCA_BRAKE_MIN_PCT", env),
            dca_spacing_growth_pct = max(
                0.0, required_float_env("STRAT_DCA_SPACING_GROWTH_PCT", env),
            ),
            dca_vol_scale_k    = required_float_env("STRAT_DCA_VOL_SCALE_K", env),
            dca_vol_ref        = required_float_env("STRAT_DCA_VOL_REF", env),
            dca_vol_interval   = required_int_env("STRAT_DCA_VOL_INTERVAL", env),
            reentry_hybrid_enabled = required_bool_env("STRAT_REENTRY_HYBRID_ENABLED", env),
            reentry_peak_relative = required_bool_env("STRAT_REENTRY_PEAK_RELATIVE", env),
            reentry_pullback_pct = required_float_env("STRAT_REENTRY_PULLBACK_PCT", env),
            reentry_pullback_adaptive = required_bool_env("STRAT_REENTRY_PULLBACK_ADAPTIVE", env),
            reentry_pullback_k = required_float_env("STRAT_REENTRY_PULLBACK_K", env),
            reentry_pullback_min = required_float_env("STRAT_REENTRY_PULLBACK_MIN", env),
            reentry_pullback_max = required_float_env("STRAT_REENTRY_PULLBACK_MAX", env),
            reentry_ttl_hours = required_float_env("STRAT_REENTRY_TTL_HOURS", env),
            reentry_bear_bounce_pct = required_float_env("STRAT_REENTRY_BEAR_BOUNCE_PCT", env),
            tp_dynamic_flat = required_bool_env("STRAT_TP_DYNAMIC_FLAT", env),
            tp_min_pct = required_float_env("STRAT_TP_MIN_PCT", env),
            tp_max_pct = required_float_env("STRAT_TP_MAX_PCT", env),
            trend_trail_dynamic = required_bool_env("STRAT_TREND_TRAIL_DYNAMIC", env),
            trend_trail_base_pct = required_float_env("STRAT_TREND_TRAIL_BASE_PCT", env),
            trend_trail_min_pct = required_float_env("STRAT_TREND_TRAIL_MIN_PCT", env),
            trend_trail_ratchet_k = required_float_env("STRAT_TREND_TRAIL_RATCHET_K", env),
            trend_trail_gain_threshold = required_float_env("STRAT_TREND_TRAIL_GAIN_THRESH", env),
            fast_profit_guard = required_bool_env("STRAT_FAST_PROFIT_GUARD", env),
            fast_profit_mult = required_float_env("STRAT_FAST_PROFIT_MULT", env),
            fast_profit_window_min = required_float_env("STRAT_FAST_PROFIT_WINDOW_MIN", env),
            fast_profit_drop_pct = required_float_env("STRAT_FAST_PROFIT_DROP_PCT", env),
            surge_guard = required_bool_env("STRAT_SURGE_GUARD", env),
            surge_gain_pct = required_float_env("STRAT_SURGE_GAIN_PCT", env),
            surge_window_hours = required_float_env("STRAT_SURGE_WINDOW_HOURS", env),
            surge_move_pct = required_float_env("STRAT_SURGE_MOVE_PCT", env),
            surge_exit_pullback_pct = required_float_env("STRAT_SURGE_EXIT_PULLBACK_PCT", env),
            surge_dynamic = required_bool_env("STRAT_SURGE_DYNAMIC", env),
            surge_min_gain_pct = required_float_env("STRAT_SURGE_MIN_GAIN_PCT", env),
            surge_max_gain_pct = required_float_env("STRAT_SURGE_MAX_GAIN_PCT", env),
            surge_vol_multiplier = required_float_env("STRAT_SURGE_VOL_MULTIPLIER", env),
            slow_grind_guard = required_bool_env("STRAT_SLOW_GRIND_GUARD", env),
            slow_grind_days = required_float_env("STRAT_SLOW_GRIND_DAYS", env),
            slow_grind_min_gain_pct = required_float_env("STRAT_SLOW_GRIND_MIN_GAIN_PCT", env),
            slow_grind_flash_window_min = required_float_env("STRAT_SLOW_GRIND_FLASH_WINDOW_MIN", env),
            slow_grind_flash_drop_pct = required_float_env("STRAT_SLOW_GRIND_FLASH_DROP_PCT", env),
            slow_grind_hourly_drop_pct = required_float_env("STRAT_SLOW_GRIND_HOURLY_DROP_PCT", env),
            sl_mhdsl_enabled   = required_bool_env("STRAT_SL_MHDSL_ENABLED", env),
            sl_disaster_buffer_ratio = required_float_env("STRAT_SL_DISASTER_BUFFER_RATIO", env),
            sl_disaster_vol_multiplier = required_float_env("STRAT_SL_DISASTER_VOL_MULTIPLIER", env),
            sl_disaster_buffer_min_pct = required_float_env("STRAT_SL_DISASTER_BUFFER_MIN_PCT", env),
            sl_disaster_buffer_max_pct = required_float_env("STRAT_SL_DISASTER_BUFFER_MAX_PCT", env),
            sl_trend_defer_max_sec = required_float_env("STRAT_SL_TREND_DEFER_MAX_SEC", env),
            sl_bounce_min_recovery_pct = required_float_env("STRAT_SL_BOUNCE_MIN_RECOVERY_PCT", env),
            sl_bounce_pullback_pct = required_float_env("STRAT_SL_BOUNCE_PULLBACK_PCT", env),
            sl_bounce_ratchet_enabled = required_bool_env("STRAT_SL_BOUNCE_RATCHET_ENABLED", env),
            sl_bounce_ratchet_k = required_float_env("STRAT_SL_BOUNCE_RATCHET_K", env),
            sl_bounce_min_pullback_pct = required_float_env("STRAT_SL_BOUNCE_MIN_PULLBACK_PCT", env),
            sl_structural_break_enabled = required_bool_env("STRAT_SL_STRUCTURAL_BREAK_ENABLED", env),
            tp_mdhtp_enabled   = required_bool_env("STRAT_TP_MDHTP_ENABLED", env),
            tp_stagnation_max_sec = required_float_env("STRAT_TP_STAGNATION_MAX_SEC", env),
        )


def _parse_tranches(spec: str) -> list:
    """Parse ``3:50,6:50`` into percentage/share pairs whose shares total 100."""
    if isinstance(spec, bool):
        raise ValueError(f"Invalid STRAT_TP_TRANCHES: {spec!r}")
    if spec is None or not str(spec).strip():
        return []
    spec = str(spec)
    out = []
    try:
        for part in spec.split(","):
            p_str, s_str = part.split(":")
            pct = safe_nonneg_float(p_str, default=None)
            share = safe_pos_float(s_str, default=None)
            if pct is None or share is None:
                raise ValueError("Invalid tranche values")
            out.append((pct, share))
        if abs(sum(share for _, share in out) - 100) >= 1e-6:
            raise ValueError("Tranche shares must total 100")
    except ValueError as exc:
        raise ValueError(f"Invalid STRAT_TP_TRANCHES: {spec!r}") from exc
    return out


def _new_state() -> dict:
    return {
        "cycle": 1,
        "qty": 0.0,
        "cost": 0.0,            # Cost basis in quote currency.
        "spent": 0.0,           # Deployed in the current cycle, subject to its cap.
        "dca_buys": 0,
        "entry_price": None,
        "last_buy_price": None,
        "cycle_fees": 0.0,      # Actual fees accumulated in the current cycle.
        "realized_gross": 0.0,
        "realized_net": 0.0,
        "fees_total": 0.0,
        "last_sell_price": None,  # Latest sale price for reentry rules.
        "last_exit_kind": None,   # TP/STOP/etc. for stop-aware reentry.
        "sl_low": None,           # Post-stop low used for bounce reentry.
        "post_sell_peak": None,   # Peak tracked after sale for hybrid/peak pullback reentry.
        "post_sell_trough": None, # Trough tracked after sale for smart bear bounce reentry.
        "last_sell_ts": None,     # Timestamp of last exit for TTL barrier decay.
        "trail_peak": None,       # Peak tracked after price exceeds TP.
        "trail_stop": None,       # Trailing floor ratchets upward even as volatility changes.
        "trend_mode": False,      # overlay: currently in a hold+trailing trend position
        "trend_peak": None,       # peak tracked in trend mode
        "trend_confirm_count": 0, # consecutive uptrend bars confirming the signal
        "orders": [],           # {txid, side, vol, price, amount, kind, ts}
        "pending_intent": None, # durable pre-submit boundary before venue acceptance
        "pending_exit": None,   # Triggered exit retained through cancellation and restart.
        # Per side/kind operational cooldown after a deterministic funds refusal.
        # This survives restarts but never blocks urgent MARKET protection.
        "placement_backoffs": {},
        # Set only after a definitive insufficient-balance rejection. The next
        # decision must compare strategy holdings with the venue ledger.
        "ledger_reconcile_required": False,
        "surge_active": False,
        "surge_peak": None,
        "slow_grind_active": False,
        "slow_grind_peak": None,
        "entry_ts": None,
        "sl_bounce_state": {},
        "sl_episode_state": {},
        "reentry_state": {},
        "tp_ride_state": {},
    }


class Strategy:
    def __init__(self, client: "StrategyExecutor", pair: str, params: StratParams,
                 dry_run: bool = True, desktop: bool = False,
                 initial_state: dict | None = None,
                 replay_mode: bool = False, state_dir: str | None = None,
                 notifier: Callable[..., None] | None = None,
                 notification_source: str = "kraken", venue_label: str = "Kraken",
                 fee_note: str = _DEFAULT_FEE_NOTE,
                 regime_service: MarketRegimeService | None = None,
                 observe_only: bool = False):
        self.client = client
        self.pair = pair
        self.p = params
        self.ccy = params.currency
        self.dry_run = _parse_bool_flag(dry_run, True)
        self.observe_only = _parse_bool_flag(observe_only, False)
        # ``dry_run`` includes paper-live, not only backtests. Replay must be identified
        # separately so the trend signal uses injected bars.
        self.replay_mode = _parse_bool_flag(replay_mode, False)
        self.desktop = _parse_bool_flag(desktop, False)
        self._notifier = notifier
        self.notification_source = notification_source
        self.venue_label = venue_label
        self.fee_note = fee_note
        self._regime_service = regime_service or MarketRegimeService()
        self._regime_bar_context = None
        # Keep the legacy one-argument call so existing replays can replace
        # state_path_for and guarantee zero I/O.
        self.state_file = (
            state_path_for(pair) if state_dir is None else state_path_for(pair, state_dir)
        )
        self._state_write_failed = False
        raw_state = initial_state if isinstance(initial_state, dict) else self._load()
        self.s = self._sanitize_spot_state(raw_state)
        self.exit_manager: PositionExitManager | None = (
            self._init_exit_manager() if (self.p.sl_mhdsl_enabled or self.p.tp_mdhtp_enabled) else None
        )
        self._paper_seq = 0
        # Observational adaptive-volatility shadow history stays in memory for sigma.
        # It is rebuilt after restart and never enters persistent state.
        self._shadow_prices = deque(maxlen=90)
        self._surge_replay_closes = (
            deque(maxlen=self._surge_window_bars() + 1)
            if self.replay_mode and params.surge_guard and params.surge_window_hours > 0
            and params.surge_move_pct > 0 else None
        )
        self._max_active_surge_vol = 0.0
        self._surge_win_check_ts: list[float] = []
        # Pair precision normalized by the provider.
        try:
            precision = client.pair_precision(pair)
        except ProviderError as exc:
            raise RuntimeError(
                f"Cannot start {pair}: venue precision is unavailable"
            ) from exc
        if precision is None:
            raise RuntimeError(f"Cannot start {pair}: venue returned no pair metadata")
        self.price_dec = precision.price_decimals
        self.vol_dec = precision.volume_decimals
        self.ordermin = precision.order_min
        self._simulated_ts: float | None = None
        self._order_lifecycle = TrackedOrderLifecycle(
            StrategyExecutorLifecycleApi(client),
            provider_name=self.venue_label,
            venue=self.venue_label,
            missing_confirmations=2,
            retry_on_lookup_error=False,
        )

    def _now(self) -> float:
        """Return the current timestamp honoring simulated time in replay mode."""
        if self.replay_mode:
            if self._simulated_ts is not None:
                return self._simulated_ts
            if self._shadow_prices:
                return self._shadow_prices[-1][0]
            return 0.0
        return time.time()

    def set_simulated_time(self, ts: float | None) -> None:
        """Explicitly set simulated timestamp for replay and backtesting drivers."""
        self._simulated_ts = _clean_nonneg_float(ts)

    def _emit(self, **event) -> None:
        """Send an event through the venue sink or legacy compatibility wrapper."""
        # Replay shares live strategy logic but must not create external effects.
        # ``dry_run`` alone is insufficient because paper-live still needs real alerts.
        if self.replay_mode:
            return
        (self._notifier or notify)(**event)

    # -- Persistence -----------------------------------------------------------
    def _store(self) -> JsonStateStore:
        return JsonStateStore(
            self.state_file, _new_state, label=self.venue_label,
            logger=log, fail_closed=not self.dry_run,
        )

    def _load(self) -> dict:
        return self._sanitize_spot_state(self._store().load())

    @staticmethod
    def _sanitize_spot_state(state: dict | None) -> dict:
        if not isinstance(state, dict):
            state = _new_state()
        for k, v in _new_state().items():
            state.setdefault(k, v)
        state["cycle"] = safe_pos_int(state.get("cycle"), default=1) or 1
        for nkey in ("qty", "cost", "spent", "cycle_fees", "fees_total"):
            state[nkey] = safe_nonneg_float(state.get(nkey), default=0.0) or 0.0
        for fkey in ("realized_gross", "realized_net"):
            state[fkey] = safe_float(state.get(fkey), default=0.0) or 0.0
        for ikey in ("dca_buys", "trend_confirm_count"):
            i = safe_int(state.get(ikey), default=0)
            state[ikey] = i if (i is not None and i >= 0) else 0
        for pkey in (
            "entry_price",
            "last_buy_price",
            "last_sell_price",
            "sl_low",
            "post_sell_peak",
            "post_sell_trough",
            "last_sell_ts",
            "trail_peak",
            "trail_stop",
            "trend_peak",
            "surge_peak",
            "slow_grind_peak",
            "entry_ts",
        ):
            if pkey in state and state.get(pkey) is not None:
                state[pkey] = _clean_pos_float(state.get(pkey))
        for bkey in (
            "trend_mode",
            "ledger_reconcile_required",
            "surge_active",
            "slow_grind_active",
            "adopted",
        ):
            if bkey in state:
                state[bkey] = _parse_bool_flag(state.get(bkey), False)
        if isinstance(state.get("orders"), list):
            clean_orders = []
            for o in state["orders"]:
                if not isinstance(o, dict):
                    continue
                if "txid" in o and o.get("txid") is not None:
                    o["txid"] = _clean_order_id(o.get("txid"))
                if "side" in o and not isinstance(o.get("side"), bool):
                    s_norm = normalize_side(o.get("side"))
                    if s_norm in ("BUY", "SELL"):
                        o["side"] = s_norm.lower()
                if "intent_id" in o:
                    o["intent_id"] = _clean_order_id(o.get("intent_id"))
                if "vol" in o:
                    o["vol"] = safe_pos_float(o.get("vol"), 0.0)
                if "price" in o:
                    o["price"] = safe_pos_float(o.get("price"), 0.0)
                if "amount" in o:
                    o["amount"] = safe_nonneg_float(o.get("amount"), 0.0)
                if "ts" in o:
                    o["ts"] = safe_nonneg_float(o.get("ts"), 0.0)
                if "market" in o:
                    o["market"] = _parse_bool_flag(o.get("market"), False)
                if "cancel_requested" in o:
                    o["cancel_requested"] = _parse_bool_flag(o.get("cancel_requested"), False)
                clean_orders.append(o)
            state["orders"] = clean_orders
        else:
            state["orders"] = []
        for dkey in (
            "placement_backoffs",
            "sl_bounce_state",
            "sl_episode_state",
            "reentry_state",
            "tp_ride_state",
        ):
            if not isinstance(state.get(dkey), dict):
                state[dkey] = {}
        pi = state.get("pending_intent")
        if isinstance(pi, dict):
            if "order_id" in pi and pi.get("order_id") is not None:
                pi["order_id"] = _clean_order_id(pi.get("order_id"))
            if "intent_id" in pi and pi.get("intent_id") is not None:
                pi["intent_id"] = _clean_order_id(pi.get("intent_id"))
            if "market" in pi:
                pi["market"] = _parse_bool_flag(pi.get("market"), False)
            meta = pi.get("metadata")
            if isinstance(meta, dict) and "market" in meta:
                meta["market"] = _parse_bool_flag(meta.get("market"), False)
        pe = state.get("pending_exit")
        if isinstance(pe, dict) and "soft_floor" in pe:
            pe["soft_floor"] = _parse_bool_flag(pe.get("soft_floor"), False)
        return state

    def _save(self) -> None:
        self._state_write_failed = True
        if self._store().save(self.s):
            self._state_write_failed = False

    def _persist_pending_intent(self, pending: dict | None) -> None:
        self.s["pending_intent"] = None if pending is None else dict(pending)
        self._save()
        if self._state_write_failed:
            raise ProviderError("the intent state could not be persisted")

    def _intent_identity(self, side: str, kind: str, vol: float, price: float) -> str:
        """Stable semantic ID for retries of one rounded strategy instruction."""
        return (
            f"spot-dca:{self.venue_label}:{self.pair}:cycle-{self.s.get('cycle', 1)}:"
            f"{kind}:{side}:{vol:.{self.vol_dec}f}:{price:.{self.price_dec}f}"
        )

    def _order_from_pending(self, pending: dict) -> dict:
        req_qty = safe_pos_float(pending.get("requested_qty"), 0.0)
        sub_qty = safe_pos_float(pending.get("submitted_qty"), None)
        eff_vol = sub_qty if sub_qty is not None else req_qty
        meta = pending.get("metadata")
        meta_amount = meta.get("amount") if isinstance(meta, dict) else None
        amount = safe_nonneg_float(pending.get("amount"), None) or safe_nonneg_float(meta_amount, 0.0)
        if 0.0 < eff_vol < req_qty and amount > 0.0:
            amount = amount * (eff_vol / req_qty)
        price = safe_pos_float(pending.get("requested_price"), 0.0)
        ts = safe_pos_float(pending.get("created_at"), self._now())
        market_val = pending.get("market")
        if market_val is None and isinstance(meta, dict):
            market_val = meta.get("market")
        txid = _clean_order_id(pending.get("order_id")) or ""
        return {
            "txid": txid,
            "side": str(pending["side"]).lower(),
            "vol": eff_vol,
            "price": price,
            "amount": amount,
            "kind": pending.get("kind"),
            "market": _parse_bool_flag(market_val, False),
            "intent_id": _clean_order_id(pending.get("intent_id")),
            "ts": ts,
        }

    def _adopt_pending_order(self, pending: dict) -> None:
        order_id = _clean_order_id(pending.get("order_id"))
        if not order_id:
            return
        if not any(_clean_order_id(order.get("txid")) == order_id for order in self.s["orders"]):
            self.s["orders"].append(self._order_from_pending(pending))
        self.s["pending_intent"] = None
        self._save()

    def _reconcile_pending_submit(self) -> None:
        pending = self.s.get("pending_intent")
        if not pending or self.dry_run:
            return
        submit_error = pending.get("submit_error") or pending.get("refusal_reason")
        if not _clean_order_id(pending.get("order_id")) and self._insufficient_funds_error(submit_error):
            self.s["ledger_reconcile_required"] = True
            if not _parse_bool_flag(pending.get("market"), False):
                self._record_placement_backoff(
                    str(pending.get("side") or "buy"),
                    str(pending.get("kind") or "ENTRY"),
                    submit_error,
                )
            self._persist_pending_intent(None)
            log(
                f"  ! [STRAT] released stale insufficient-funds pending intent "
                f"({pending.get('side')} {pending.get('kind')})"
            )
            return
        try:
            result = self._order_lifecycle.reconcile(
                pending, persist=self._persist_pending_intent)
        except (ProviderError, RuntimeError, ValueError) as exc:
            log(f"  ! [STRAT] unreconciled pending intent ({exc}) — not resending")
            return
        if result.order_known:
            self._adopt_pending_order(result.intent)
            log(
                f"  [STRAT] recovered intent {result.intent['side']} "
                f"txid={result.intent['order_id']}"
            )
        elif result.outcome in {"absent", "retryable"}:
            log("  [STRAT] intent absence confirmed — the strategy may re-evaluate it")

    # -- Helpers ---------------------------------------------------------------
    def _avg(self) -> float | None:
        qty = _clean_pos_float(self.s.get("qty"))
        cost = _clean_pos_float(self.s.get("cost"))
        return cost / qty if (qty is not None and cost is not None and qty > 1e-12) else None

    def _qty_for(self, amount: float, price: float) -> float:
        amt_f = safe_pos_float(amount, default=None)
        px_f = safe_pos_float(price, default=None)
        if amt_f is None or px_f is None:
            return 0.0
        return round(amt_f / px_f, self.vol_dec)

    def _dust_safe_qty(self, qty: float) -> float:
        """Leave one volume tick to tolerate a venue's rounded balance.

        Selling the full internally tracked quantity can exceed a rounded ledger and
        cause permanent insufficient-funds retries. Preserve historical high-precision
        behavior without magnifying dust on low-precision venues.
        """
        qty_f = safe_pos_float(qty, default=None)
        if qty_f is None:
            return 0.0
        dust_decimals = self.vol_dec if self.vol_dec <= 2 else self.vol_dec - 1
        step = 10.0 ** -max(dust_decimals, 1)
        # Epsilon prevents floating-point division from leaving two ticks after floor.
        ticks = math.floor((qty_f + step * 1e-9) / step) - 1
        return round(max(ticks, 0) * step, self.vol_dec)

    def _has_open(self, side: str) -> bool:
        target = str(side or "").strip().lower() if not isinstance(side, bool) else ""
        if not target:
            return False
        orders = self.s.get("orders")
        if not isinstance(orders, list):
            return False
        return any(
            isinstance(o, dict) and str(o.get("side") or "").strip().lower() == target
            for o in orders
        )

    def _find_open(self, side: str) -> dict | None:
        target = str(side or "").strip().lower() if not isinstance(side, bool) else ""
        if not target:
            return None
        orders = self.s.get("orders")
        if not isinstance(orders, list):
            return None
        return next(
            (o for o in orders if isinstance(o, dict) and str(o.get("side") or "").strip().lower() == target),
            None,
        )

    def _remove(self, o: dict) -> None:
        if o in self.s["orders"]:
            self.s["orders"].remove(o)

    @staticmethod
    def _insufficient_funds_error(error) -> bool:
        text = str(error or "").lower()
        return any(marker in text for marker in (
            "insufficient funds", "insufficient balance",
            "insufficient spot balance",
        ))

    @staticmethod
    def _placement_backoff_key(side: str, kind: str) -> str:
        return f"{str(side).lower()}:{str(kind or 'UNKNOWN').upper()}"

    def _placement_backoff_remaining(self, side: str, kind: str) -> float:
        key = self._placement_backoff_key(side, kind)
        records = self.s.get("placement_backoffs")
        if not isinstance(records, dict):
            return 0.0
        record = records.get(key)
        if not isinstance(record, dict):
            return 0.0
        until_f = safe_float(record.get("until"), default=None)
        now_f = safe_float(self._now(), default=None)
        if until_f is None or now_f is None:
            return 0.0
        rem = max(0.0, until_f - now_f)
        return min(_PLACEMENT_BACKOFF_CAP_SEC, rem)

    def _record_placement_backoff(self, side: str, kind: str, error) -> None:
        key = self._placement_backoff_key(side, kind)
        records = self.s.get("placement_backoffs")
        if not isinstance(records, dict):
            records = {}
            self.s["placement_backoffs"] = records
        previous = records.get(key)
        if not isinstance(previous, dict):
            previous = {}
        attempts = safe_nonneg_int(previous.get("attempts"), default=0) + 1
        base = max(60.0, float(self.p.check_minutes) * 60.0)
        delay = min(_PLACEMENT_BACKOFF_CAP_SEC, base * (2 ** min(attempts - 1, 10)))
        records[key] = {
            "attempts": attempts,
            "until": self._now() + delay,
            "reason": str(error),
        }
        self._save()
        log(
            f"  ! [STRAT] {side} {kind} insufficient funds — "
            f"operational cooldown {delay:.0f}s (refusal {attempts})"
        )

    def _clear_placement_backoff(self, side: str, kind: str) -> None:
        records = self.s.get("placement_backoffs")
        if not isinstance(records, dict):
            self.s["placement_backoffs"] = {}
            return
        records.pop(self._placement_backoff_key(side, kind), None)

    # -- Placement -------------------------------------------------------------
    def _place(self, side: str, vol: float, price: float, kind: str, amount: float = 0.0,
               market: bool = False) -> bool:
        market = _parse_bool_flag(market, False)
        s_norm = normalize_side(side)
        if s_norm in ("BUY", "SELL"):
            side = s_norm.lower()
        # A market trailing/stop exit executes immediately instead of risking a missed
        # sharp drop with a limit order. Backtests fill it at the next bar's open.
        vol_f = safe_pos_float(vol, default=None)
        price_f = safe_pos_float(price, default=None)
        if vol_f is None or price_f is None:
            log(f"  ! [STRAT] invalid order parameters vol={vol!r} price={price!r} — skipping")
            return False
        vol = round(vol_f, self.vol_dec)
        price = round(price_f, self.price_dec)
        if price <= 0:
            log(f"  ! [STRAT] price {price} <= 0 — skipping")
            return False
        if vol <= 0 or (self.ordermin and vol < self.ordermin):
            log(f"  ! [STRAT] volume {vol} < minimum order {self.ordermin} — skipping")
            return False
        # Refuse a new non-STOP limit SELL below known average cost after rounding.
        # This gross-price check is not a net-profit or fill guarantee: fees are not
        # added, equality is allowed, and unknown average cost does not block here.
        # MARKET and STOP intents bypass this guard, but still pass the remaining
        # lifecycle, cancellation, balance, and venue checks. The optional soft
        # trailing floor is a separate strategy policy applied before this method.
        if not market and str(side).lower() == "sell" and str(kind).upper() != "STOP":
            _avg = self._avg()
            if _avg is not None and price < _avg:
                log(f"  ! [STRAT] [GUARD] {kind} sell rejected: price {price} < avg "
                    f"{_avg:.6f} (profit-floor; STOP/MARKET excepted)")
                return False
        # A previous deterministic funds rejection cannot become executable merely
        # by hammering the venue every tick. MARKET exits deliberately bypass this:
        # missing a protective exit is a larger risk than a repeated refusal.
        if not market:
            remaining = self._placement_backoff_remaining(side, kind)
            if remaining > 0:
                log(
                    f"  ! [STRAT] {side} {kind} in cooldown after insufficient funds "
                    f"({remaining:.0f}s remaining)"
                )
                return False
        if self.dry_run:
            self._paper_seq += 1
            log(f"  [STRAT] [PAPER] {side.upper()} {kind}{' MKT' if market else ''} {vol} @ {price} {self.ccy}")
            self.s["orders"].append({"txid": f"PAPER-{self._paper_seq}", "side": side,
                                     "vol": vol, "price": price, "amount": amount,
                                     "kind": kind, "market": market, "ts": self._now()})
            return True
        if self.s.get("pending_intent"):
            log("  ! [STRAT] an unreconciled pre-submit intent exists — not duplicating")
            return False
        try:
            # Venues that might accept an underfunded BUY and cancel its remainder can
            # reject the intent during preflight before audit/submission.
            preflight = getattr(type(self.client), "preflight_order", None)
            if callable(preflight):
                preflight(
                    self.client, self.pair, side, vol, None if market else price,
                    market=market, kind=kind,
                )
            intent_id = self._intent_identity(side, kind, vol, price)
            client_order_id = intent_client_order_id(self.venue_label, intent_id)
            if not client_order_id:
                raise ProviderError(
                    f"{self.venue_label}: client_order_id unavailable for the lifecycle")
            intent = self._order_lifecycle.new_intent(
                intent_id=intent_id,
                client_order_id=client_order_id,
                symbol=self.pair,
                side=side,
                requested_qty=vol,
                requested_price=price,
                kind=kind,
                metadata={"amount": amount, "market": market},
            )
            submit_outcome = getattr(type(self.client), "submit_outcome_with_intent", None)
            submit_with_intent = getattr(type(self.client), "submit_order_with_intent", None)
            def submit():
                if callable(submit_outcome):
                    return submit_outcome(
                        self.client, intent_id, self.pair, side, vol,
                        None if market else price,
                        market=market, kind=kind,
                        reference_price=price if market else None,
                        client_order_id=client_order_id,
                    )
                if callable(submit_with_intent):
                    return submit_with_intent(
                        self.client, intent_id, self.pair, side, vol,
                        None if market else price,
                        market=market, kind=kind,
                        reference_price=price if market else None,
                        client_order_id=client_order_id,
                    )
                return self.client.submit_order(
                    self.pair, side, vol, None if market else price,
                    market=market, kind=kind,
                    client_order_id=client_order_id,
                )

            tracked = self._order_lifecycle.submit(
                intent, persist=self._persist_pending_intent, submit=submit)
            if not tracked.order_known:
                submit_error = tracked.intent.get("submit_error")
                if tracked.outcome == "refused":
                    refusal = tracked.intent.get("refusal_reason") or "venue refusal"
                    self._persist_pending_intent(None)
                    if self._insufficient_funds_error(refusal):
                        self.s["ledger_reconcile_required"] = True
                        self._record_placement_backoff(side, kind, refusal)
                    log(
                        f"  ! [STRAT] {side} {kind} rejected definitively; "
                        "the intent was released for safe reevaluation"
                    )
                    return False
                if self._insufficient_funds_error(submit_error):
                    self.s["ledger_reconcile_required"] = True
                    if not market:
                        self._record_placement_backoff(
                            side, kind, submit_error)
                    self._persist_pending_intent(None)
                    log(
                        f"  ! [STRAT] {side} {kind} rejected due to insufficient funds; "
                        "the intent was released for safe reevaluation"
                    )
                    return False
                log(
                    f"  ! [STRAT] {side} {kind} ambiguous submit; "
                    "the intent stays persisted for reconciliation"
                )
                return False
            txid = str(tracked.intent["order_id"])
            log(f"  [STRAT] {side.upper()} {kind} placed txid={txid} {vol} @ {price}")
            self._clear_placement_backoff(side, kind)
            self._adopt_pending_order(tracked.intent)
            return True
        except (ProviderError, RuntimeError, ValueError) as e:
            if self._insufficient_funds_error(e):
                self.s["ledger_reconcile_required"] = True
                if not market:
                    self._record_placement_backoff(side, kind, e)
            log(f"  ! [STRAT] {side} {kind} failed: {e}")
            return False

    def _cancel_order(self, o: dict) -> bool:
        """Request cancellation without forgetting the order before terminal confirmation.

        An accepted request does not prove that no concurrent fill occurred. Retain the
        order until the provider reports terminal state so reconciliation applies that fill.
        """
        if self.dry_run or str(o["txid"]).startswith("PAPER"):
            self._remove(o)
            log(f"  [STRAT] cancelled {o['side']} {o['txid']}")
            return True
        if o.get("cancel_requested"):
            return True
        try:
            cancel_with_intent = getattr(type(self.client), "cancel_order_with_intent", None)
            if callable(cancel_with_intent):
                cancel_with_intent(
                    self.client, o.get("intent_id") or f"legacy-{self.pair}-{o['txid']}",
                    self.pair, o["txid"],
                )
            else:
                self.client.cancel_order(self.pair, o["txid"])
        except ProviderError as e:
            log(f"  ! [STRAT] cancel failed for {o['txid']}: {e} — the order stays tracked")
            return False
        o["cancel_requested"] = True
        o["cancel_ts"] = self._now()
        # Persist cancellation intent across restart and track through terminal status.
        self._save()
        log(f"  [STRAT] cancel requested {o['side']} {o['txid']} — waiting for a terminal status")
        return True

    def _cancel_open(self, side: str) -> bool:
        o = self._find_open(side)
        if not o:
            return True
        return self._cancel_order(o)

    def _cancel_orders(self, side: str | None = None, *, exclude_market: bool = False) -> bool:
        """Request cancellation of selected orders; return False if any request fails."""
        orders = self.s["orders"]
        if not orders:
            return True
        selected = [
            o for o in list(orders)
            if (side is None or o["side"] == side)
            and not (exclude_market and o.get("market"))
        ]
        ok = True
        for o in selected:
            ok = self._cancel_order(o) and ok
        return ok

    def _has_pending_market_exit(self) -> bool:
        for o in self.s["orders"]:
            if o["side"] == "sell" and o.get("market"):
                return True
        return False

    def _request_market_exit(self, price: float, kind: str, *, soft_floor: bool = False,
                             min_profit_pct: float | None = None) -> bool:
        """Persist exit ownership before cancellation, then sell reconciled holdings.

        A cancel acknowledgement is not terminal venue truth. Do not submit a
        competing SELL or leave a DCA capable of filling after exit. The next
        reconcile/step resumes this request, even after restart or a rebound.
        """
        pending = self.s.get("pending_exit")
        soft_floor = _parse_bool_flag(soft_floor, False)
        if not isinstance(pending, dict) or (kind == "STOP" and pending.get("kind") != "STOP"):
            self.s["pending_exit"] = {"kind": kind, "soft_floor": soft_floor}
            if min_profit_pct is not None and kind != "STOP":
                # Freeze the signal's floor across retries, config changes, and restart.
                self.s["pending_exit"]["min_profit_pct"] = min_profit_pct
        return self._continue_market_exit(price)

    def _continue_market_exit(self, price: float) -> bool:
        pending = self.s.get("pending_exit")
        if not isinstance(pending, dict):
            self.s["pending_exit"] = None
            return False
        if self.s.get("pending_intent") or self._has_pending_market_exit():
            return False
        # This boundary also covers resumption after an earlier failed save.
        self._save()
        if self._state_write_failed:
            return False
        if not self._cancel_orders() or self.s["orders"]:
            log("  [STRAT] exit pending: waiting for terminal cancellation and fill reconciliation")
            return False
        if self.s["qty"] <= 1e-12:
            if "fill_price" in pending:
                self._finalize_cycle_if_flat(pending, pending["fill_price"])
            return False
        kind = str(pending.get("kind") or "TP")
        factor = 0.995 if kind == "STOP" else 0.999
        reference = round(price * factor, self.price_dec)
        if "min_profit_pct" in pending and kind != "STOP":
            # Recompute against reconciled cost basis, including late BUY fills.
            min_profit_pct = safe_nonneg_float(pending["min_profit_pct"], default=None)
            avg = safe_pos_float(self._avg(), default=None)
            if min_profit_pct is None or avg is None:
                log("  [STRAT] profit exit pending: invalid stored floor or cost basis")
                return False
            floor = safe_float(sr.tp_price(avg, min_profit_pct), default=None)
            ref_f = safe_pos_float(reference, default=None)
            if floor is None or ref_f is None or ref_f < floor:
                log("  [STRAT] profit exit pending: current reference does not satisfy the stored floor")
                return False
        if _parse_bool_flag(pending.get("soft_floor"), False) and kind != "STOP":
            floor = self._trail_profit_floor_price(self._avg() or 0.0)
            if floor is not None and reference < floor:
                log("  [STRAT] soft exit pending: current reference is below the configured floor")
                return False
        placed = self._place(
            "sell", self._dust_safe_qty(self.s["qty"]),
            reference, kind=kind, market=True)
        if placed and kind == "STOP":
            self._emit(
                title=f"🛑 SL {self.pair}",
                body="Protective MARKET exit accepted; waiting for confirmed execution.",
                source=self.notification_source, price=price, desktop=self.desktop)
        return placed

    # -- Reconciliation --------------------------------------------------------
    def reconcile(self, price: float) -> None:
        self._reconcile_pending_submit()
        if not self.s.get("pending_exit"):
            # Preserve ownership of an accepted exit restored from pre-marker
            # state (or recovered by client ID) before applying a racing BUY.
            exit_order = next((o for o in self.s["orders"]
                               if o["side"] == "sell" and o.get("market")), None)
            if exit_order:
                self.s["pending_exit"] = {
                    "kind": exit_order.get("kind") or "TP",
                    "soft_floor": (self.s.get("trail_peak") is not None
                                   and not self.s.get("trend_mode")),
                }
                self._save()
        for side in ("buy", "sell"):
            for o in [x for x in self.s["orders"] if x["side"] == side]:
                if o not in self.s["orders"]:
                    continue
                if self.dry_run:
                    if o.get("market"):
                        # In paper-live, a MARKET order fills at the currently observed
                        # price, including a drop below the stop/trailing reference.
                        fill_price = price
                    elif ((side == "buy" and price <= o["price"])
                          or (side == "sell" and price >= o["price"])):
                        fill_price = o["price"]
                    else:
                        age = (self._now() - (_clean_nonneg_float(o.get("ts")) or 0.0)) / 60
                        if (side == "buy" and not o.get("cancel_requested")
                                and age > self.p.order_ttl_min and price > o["price"] * 1.003):
                            log(f"  [STRAT] buy {o['txid']} unfilled, the price rose — cancelling and re-placing")
                            self._cancel_order(o)
                        continue
                    if o in self.s["orders"]:
                        self._remove(o)
                        self._apply_fill(o, o["vol"], fill_price, fee=0.0)
                    continue
                # Providers report cumulative quantity, cost, and fee even while an order
                # remains open. Apply only the delta since the saved reconciliation marker.
                try:
                    status_with_intent = getattr(type(self.client), "order_status_with_intent", None)
                    if callable(status_with_intent):
                        status = status_with_intent(
                            self.client, o.get("intent_id") or f"legacy-{self.pair}-{o['txid']}",
                            self.pair, o["txid"],
                        )
                    else:
                        status = self.client.order_status(self.pair, o["txid"])
                except ProviderError as e:
                    log(f"  ! [STRAT] status {o['txid']} failed: {e} — keeping the order")
                    continue
                st = status.status
                terminal = st in ("closed", "canceled", "expired")
                total_vol = safe_nonneg_float(status.filled_qty or 0.0, default=None)
                total_fee = safe_float(status.fee or 0.0, default=None)
                ord_px = safe_pos_float(o.get("price"), default=None)
                if total_vol is None or total_fee is None or ord_px is None:
                    log(f"  ! [STRAT] status {o['txid']} has an invalid execution — keeping the order")
                    continue
                total_cost = safe_nonneg_float(status.cost or total_vol * ord_px, default=None)
                if total_cost is None:
                    log(f"  ! [STRAT] status {o['txid']} has an invalid execution — keeping the order")
                    continue
                reported_price = safe_pos_float(total_cost / total_vol if total_vol > 0 else ord_px, default=None)
                if reported_price is None:
                    log(f"  ! [STRAT] status {o['txid']} has an invalid execution — keeping the order")
                    continue

                applied_vol = safe_nonneg_float(o.get("applied_vol"), 0.0)
                applied_cost = safe_nonneg_float(o.get("applied_cost"), 0.0)
                applied_fee = safe_float(o.get("applied_fee"), 0.0)
                eps = max(1e-12, float(o["vol"]) * 1e-12)
                if total_vol + eps < applied_vol or total_cost + eps < applied_cost:
                    log(f"  ! [STRAT] status {o['txid']} regressed cumulatively "
                        f"(vol {total_vol}<{applied_vol}) — not reapplying")
                    continue

                delta_vol = max(0.0, total_vol - applied_vol)
                delta_cost = max(0.0, total_cost - applied_cost)
                # Fees are signed because some venues report negative maker rebates.
                delta_fee = total_fee - applied_fee
                # Persist markers before accounting because a final SELL may replace state.
                o["applied_vol"] = total_vol
                o["applied_cost"] = total_cost
                o["applied_fee"] = total_fee

                if delta_vol > eps:
                    fill_order = dict(o)
                    if o["side"] == "buy":
                        # ``amount`` is the order's nominal cap; partial fills consume it proportionally.
                        fill_order["amount"] = (
                            float(o.get("amount") or 0.0) * delta_vol / float(o["vol"])
                            if float(o["vol"]) > 0 else delta_cost
                        )
                        if applied_vol > eps and fill_order.get("kind") in {"DCA", "TREND_ENTRY"}:
                            fill_order["kind"] = f"{fill_order['kind']}_PARTIAL"
                    fill_price = delta_cost / delta_vol if delta_cost > 0 else reported_price
                    if terminal:
                        self._remove(o)
                    self._apply_fill(
                        fill_order, delta_vol, fill_price, fee=delta_fee, final=False,
                    )
                elif abs(delta_fee) > eps:
                    # Charge a late final fee without simulating a zero-volume fill.
                    self.s["cycle_fees"] += delta_fee
                    self.s["fees_total"] += delta_fee
                    self.s["realized_net"] -= delta_fee

                if terminal:
                    if o in self.s["orders"]:
                        self._remove(o)
                    if o["side"] == "sell":
                        self._finalize_cycle_if_flat(o, reported_price)
                    log(f"  [STRAT] {o['txid']} {st} (executed {total_vol}/{o['vol']})")
                else:
                    age = (self._now() - (_clean_nonneg_float(o.get("ts")) or 0.0)) / 60
                    if (side == "buy" and not o.get("cancel_requested")
                            and age > self.p.order_ttl_min and price > o["price"] * 1.003):
                        log(f"  [STRAT] buy {o['txid']} unfilled, the price rose — cancelling and re-placing")
                        self._cancel_order(o)

    def _apply_fill(self, o: dict, vol: float, price: float, fee: float,
                    *, final: bool = True) -> None:
        tag = "[PAPER] " if self.dry_run else ""
        self.s["cycle_fees"] += fee
        self.s["fees_total"] += fee
        # Charge every fee once even while the position remains open. This prevents net
        # mark-to-market P&L from overstating positions with a filled BUY.
        self.s["realized_net"] -= fee
        if o["side"] == "buy":
            self.s["qty"] += vol
            self.s["cost"] += vol * price
            self.s["last_buy_price"] = price
            if self.s["entry_price"] is None:
                self.s["entry_price"] = price
            if self.s.get("entry_ts") is None:
                self.s["entry_ts"] = self._shadow_prices[-1][0] if self._shadow_prices else self._now()
            raw_amt = o.get("amount")
            amt_val = _clean_nonneg_float(raw_amt)
            self.s["spent"] += amt_val if (amt_val is not None and raw_amt is not None) else (vol * price)
            if o.get("kind") == "DCA":
                self.s["dca_buys"] += 1
            if o.get("kind") == "TREND_ENTRY":
                # A submitted order is not a trend position. Enable the mode only after
                # the venue or replay confirms the fill.
                self.s["trend_mode"] = True
                self.s["trend_peak"] = price
            avg = self._avg()
            log(f"  [STRAT] {tag}BUY FILLED {vol} @ {price} {self.ccy} ({o.get('kind')})  "
                f"qty={self.s['qty']:.8f} avg={avg:.{self.price_dec}f} fee={fee}")
            self._emit(title=f"{tag}{self.pair} BUY {vol:.2f}@{price:.2f}",
                       body=(f"{o.get('kind')} | q{self.s['qty']:.2f} a{avg:.2f} | "
                             f"desf{self.s['spent']:.0f}{self.ccy}"),
                       source=self.notification_source, price=price, desktop=self.desktop)
            # Reprice LIMIT take-profits after a BUY, but never revoke an already
            # submitted protective MARKET exit when a late BUY fill is reconciled.
            self._cancel_orders("sell", exclude_market=True)
        else:  # sell
            avg = self._avg() or price
            gross = (price - avg) * vol
            net = gross - fee
            self.s["realized_gross"] += gross
            self.s["realized_net"] += gross
            self.s["qty"] -= vol
            # A partial SELL releases cost basis proportionally. Otherwise the full cost
            # remains assigned to the reduced quantity and distorts the average.
            self.s["cost"] = max(0.0, self.s["cost"] - avg * vol)
            log(f"  [STRAT] {tag}SELL FILLED {vol} @ {price} {self.ccy}  "
                f"brut={gross:+.4f} fee_ciclu={self.s['cycle_fees']:.4f} net={net:+.4f}")
            self._emit(title=f"{tag}{self.pair} SELL {vol:.2f}@{price:.2f} N{net:+.2f}{self.ccy}",
                       body=(f"a{avg:.2f} · br{gross:+.2f} fee{self.s['cycle_fees']:.2f} N{net:+.2f} | "
                             f"Ntot{self.s['realized_net']:+.2f}{self.ccy}"),
                       source=self.notification_source, price=price, desktop=self.desktop)
            if final:
                self._finalize_cycle_if_flat(o, price)

    def _finalize_cycle_if_flat(self, o: dict, price: float) -> None:
        """Close a cycle once, only after terminal exit status."""
        # Dust left by _dust_safe_qty must not keep a cycle open. Apply the threshold
        # only at terminal state because an open partial fill may validly retain dust.
        dust = 2 * 10.0 ** -(max(self.vol_dec - 1, 1))
        if abs(self.s["qty"]) < dust:
            self.s["qty"] = 0.0
            self.s["cost"] = 0.0
        if self.s["qty"] > 1e-12:
            return
        if self.s["orders"] or self.s.get("pending_intent"):
            # A terminal SELL can race an earlier DCA/TP. Closing the cycle here
            # would erase that order and lose a later fill. Freeze new entries,
            # resolve the remaining orders, then either exit their net remainder
            # or finish this cycle once. Keep the actual sale reference for reentry.
            pending = self.s.get("pending_exit") or {"kind": o.get("kind") or "TP"}
            pending["fill_price"] = price
            self.s["pending_exit"] = pending
            self._save()
            return
        keep = (self.s["realized_gross"], self.s["realized_net"],
                self.s["fees_total"], self.s.get("cycle", 1) + 1)
        self.s = _new_state()
        (self.s["realized_gross"], self.s["realized_net"],
         self.s["fees_total"], self.s["cycle"]) = keep
        self.s["last_sell_price"] = price   # reentry rule must not buy back higher
        self.s["last_exit_kind"] = o.get("kind")   # TP/STOP enables stop-aware reentry
        self.s["sl_low"] = price            # Initial low for post-stop bounce reentry.
        self.s["post_sell_peak"] = price    # Initial peak for post-sale pullback tracking.
        self.s["post_sell_trough"] = price  # Initial trough for post-sale bear bounce tracking.
        self.s["last_sell_ts"] = self._shadow_prices[-1][0] if self._shadow_prices else self._now()
        if self.exit_manager is not None:
            self.exit_manager.clear_state(self.pair)
            self.exit_manager.bind_state(self.s)
        log(f"  [STRAT] === cycle closed; restarting (cycle {self.s['cycle']}) ===")

    # -- Decision logic --------------------------------------------------------
    def _is_trend_direction(
        self,
        regime: MarketRegimeDecision | None,
        expected_regime: str,
        client_tokens: tuple[str, ...],
    ) -> bool:
        if regime is not None and self._regime_matches(regime, expected_regime, min_samples=self.p.regime_min_samples):
            return True
        try:
            if hasattr(self.client, "trend"):
                t = self.client.trend(self.pair)
                if t and str(t).lower() in client_tokens:
                    return True
        except Exception:
            pass
        if regime is None:
            try:
                reg, _ = self._regime_context()
                if self._regime_matches(reg, expected_regime, min_samples=self.p.regime_min_samples):
                    return True
            except Exception:
                pass
        return False

    def _is_trend_up(self, regime: MarketRegimeDecision | None = None) -> bool:
        """Return True if an uptrend is identified via regime, trend_mode, or venue client."""
        if self.s.get("trend_mode"):
            return True
        return self._is_trend_direction(regime, "bull", ("up", "bull", "uptrend"))

    def _is_trend_down(self, regime: MarketRegimeDecision | None = None) -> bool:
        """Return True if a downtrend is identified via regime or venue client."""
        return self._is_trend_direction(regime, "bear", ("down", "bear", "downtrend"))

    def _vol_1h_ratio(self) -> tuple[float, str]:
        """Return 1h volatility as a decimal ratio [e.g. 0.025 for 2.5%] and its source."""
        try:
            vol = safe_pos_float(self._trail_vol_1h(), default=None)
            if vol is not None:
                return vol_pct_to_ratio(vol), "trail_ohlc"
        except Exception:
            pass
        try:
            vol = safe_pos_float(self._shadow_vol_1h(), default=None)
            if vol is not None:
                return vol_pct_to_ratio(vol), "shadow_ticks"
        except Exception:
            pass
        if not self.replay_mode:
            cached = resolve_cached_volatility_1h(self.pair)
            if cached is not None:
                return cached, "cache"
        return 0.0, "default"

    def _resolve_volatility_tuple(self, symbol: str) -> tuple[float, str]:
        """Callable adapter for PositionExitManager vol_resolver."""
        return self._vol_1h_ratio()

    def _init_exit_manager(self) -> PositionExitManager:
        exit_cfg = PositionExitConfig(
            lost_threshold=self.p.stop_loss_pct / 100.0,
            sl_disaster_buffer_ratio=self.p.sl_disaster_buffer_ratio,
            sl_disaster_vol_multiplier=self.p.sl_disaster_vol_multiplier,
            sl_disaster_buffer_min_pct=self.p.sl_disaster_buffer_min_pct / 100.0,
            sl_disaster_buffer_max_pct=self.p.sl_disaster_buffer_max_pct / 100.0,
            sl_trend_defer_max_sec=self.p.sl_trend_defer_max_sec,
            sl_bounce_min_recovery_pct=self.p.sl_bounce_min_recovery_pct / 100.0,
            sl_bounce_pullback_pct=self.p.sl_bounce_pullback_pct / 100.0,
            sl_bounce_ratchet_enabled=self.p.sl_bounce_ratchet_enabled,
            sl_bounce_ratchet_k=self.p.sl_bounce_ratchet_k,
            sl_bounce_min_pullback_pct=self.p.sl_bounce_min_pullback_pct / 100.0,
            sl_structural_break_enabled=self.p.sl_structural_break_enabled,
            smart_sl_enabled=False,
            gain_threshold=self.p.takeprofit_pct / 100.0,
            tp_trend_ride_enabled=self.p.tp_trend_hold,
            tp_profit_floor_min_pct=(self.p.tp_trail_profit_floor_pct or 0.0) / 100.0,
            tp_pullback_pct=self.p.tp_trail_pct / 100.0,
            tp_stagnation_max_sec=self.p.tp_stagnation_max_sec,
            fast_profit_guard_enabled=self.p.fast_profit_guard,
            fast_profit_mult=self.p.fast_profit_mult,
            fast_profit_window_sec=self.p.fast_profit_window_min * 60.0,
            fast_profit_drop_pct=self.p.fast_profit_drop_pct,
            reentry_enabled=True,
            reentry_sl_bounce_pct=self.p.reentry_sl_bounce_pct / 100.0,
            reentry_drop_pct=self.p.reentry_drop_pct / 100.0,
            reentry_pullback_pct=self.p.reentry_pullback_pct / 100.0,
            reentry_bear_bounce_pct=self.p.reentry_bear_bounce_pct / 100.0,
            reentry_ttl_hours=self.p.reentry_ttl_hours,
            reentry_tolerance_pct=self.p.reentry_tolerance_pct,
            allow_external_cache=not self.replay_mode,
        )
        mgr = PositionExitManager(
            config=exit_cfg,
            vol_resolver=self._resolve_volatility_tuple,
        )
        mgr.bind_state(self.s)
        return mgr

    def _create_position_snapshot(
        self,
        price: float,
        avg: float,
        held: float,
        regime: MarketRegimeDecision | None = None,
        include_vol: bool = True,
        include_shadow: bool = True,
    ) -> PositionSnapshot:
        """Construct a standardized PositionSnapshot using the strategy's current context."""
        tick_time = self._shadow_prices[-1][0] if self._shadow_prices else self._now()
        if include_vol:
            vol_1h, vol_source = self._vol_1h_ratio()
        else:
            vol_1h, vol_source = 0.0, "default"
        trend_up = self._is_trend_up(regime)
        return PositionSnapshot(
            symbol=self.pair,
            current_price=price,
            avg_buy_price=avg,
            net_qty=held,
            sellable_qty=held,
            current_time_s=tick_time,
            trend_up=trend_up,
            vol_1h=vol_1h,
            vol_source=vol_source,
            provider=self.venue_label,
            shadow_prices=list(self._shadow_prices) if include_shadow else [],
        )

    def _check_stop_loss(self, price: float) -> bool:
        """Close everything when unrealized loss crosses the anti-runaway threshold."""
        if (self.s.get("pending_exit") or {}).get("kind") == "STOP":
            if self._has_pending_market_exit():
                return True
            self._continue_market_exit(price)
            return True
        if self.p.stop_loss_pct <= 0:
            return False
        avg = self._avg()
        if not avg:
            return False
        loss_pct = (avg - price) / avg * 100   # A long position loses below average cost.

        if not self.p.sl_mhdsl_enabled:
            if sr.hit_stop(avg, price, self.p.stop_loss_pct):
                # Reconcile an already submitted MARKET exit instead of canceling or duplicating it.
                if self._has_pending_market_exit():
                    return True
                log(f"  🛑 [STRAT] STOP-LOSS: loss {loss_pct:.2f}% >= {self.p.stop_loss_pct}% — SELLING EVERYTHING (cutting the loss)")
                placed = self._request_market_exit(price, "STOP")
                if not placed:
                    log("  [STRAT] STOP triggered; exit remains pending reconciliation or submission")
                return True
            return False
        in_sl = loss_pct > 0.0 and (
            loss_pct >= self.p.stop_loss_pct
            or are_close(loss_pct, self.p.stop_loss_pct, self.p.reentry_tolerance_pct)
        )
        mgr = self.exit_manager
        in_bounce_watch = bool(mgr is not None and self.pair in mgr.sl_bounce_state)
        if not in_sl and not in_bounce_watch:
            if mgr is not None and mgr.sl_episode_state:
                mgr.reset_sl_episode(self.pair)  # Leaving SL territory ends the episode floor lock
            return False

        if self.exit_manager is None:
            self.exit_manager = self._init_exit_manager()

        snapshot = self._create_position_snapshot(
            price, avg, self.s.get("qty", 0.0), include_shadow=not self.replay_mode
        )
        decision = self.exit_manager.evaluate_stop_loss(
            snapshot, custom_threshold=self.p.stop_loss_pct / 100.0,
        )

        if decision.action == ExitAction.SELL and decision.should_exit:
            if self._has_pending_market_exit():
                return True
            log(
                f"  🛑 [STRAT] MHDSL STOP-LOSS: loss {loss_pct:.2f}% >= {self.p.stop_loss_pct}% "
                f"({decision.reason}) — SELLING EVERYTHING"
            )
            placed = self._request_market_exit(price, "STOP")
            if not placed:
                log("  [STRAT] STOP triggered; exit remains pending reconciliation or submission")
            return True
        elif decision.action == ExitAction.HOLD:
            if decision.metadata.get("priority") in (1, 2) or self.pair in self.exit_manager.sl_bounce_state:
                log(f"  🛡️ [STRAT] MHDSL BOUNCE DEFERRAL: {decision.reason}")
            return False

        return False

    def _trail_profit_floor_price(self, avg: float) -> float | None:
        """Return the minimum soft-exit price, rounded up to venue precision.

        The threshold is intentionally gross and simple. Configuration must include
        enough fee buffer; central and stress benchmarks then measure net profit using
        scenario fees and fills. ``0`` exactly preserves the existing MARKET trailing.
        """
        pct = float(self.p.tp_trail_profit_floor_pct or 0.0)
        if pct <= 0 or avg <= 0:
            return None
        raw = sr.tp_price(avg, pct)
        scale = 10 ** self.price_dec
        return math.ceil(raw * scale - 1e-12) / scale

    def _maybe_adopt(self) -> None:
        """Adopt an existing account position instead of buying an entry.

        Run once on fresh state only so an active cycle cannot be corrupted.
        """
        if self.p.adopt_cost <= 0 or self.s.get("adopted"):
            return
        if (self.s["qty"] > 1e-12 or self.s["orders"]
                or self.s["cycle"] != 1 or self.s["spent"] > 0):
            log("  [STRAT] adopt: the state is not fresh — NOT adopting (cycle in progress)")
            return
        qty = _clean_nonneg_float(self.p.adopt_qty) or 0.0
        if qty <= 0:  # Read quantity from the pair's base-asset balance.
            try:
                precision = self.client.pair_precision(self.pair)
                base = precision.base_asset if precision else ""
                raw_bal = self.client.free_balance(base)
            except ProviderError as e:
                log(f"  ! [STRAT] adopt: cannot read the balance ({e})")
                return
            if raw_bal is None:
                qty = 0.0
            else:
                clean_bal = _clean_nonneg_float(raw_bal)
                if clean_bal is None:
                    log("  ! [STRAT] adopt: invalid balance on the base asset")
                    return
                qty = clean_bal
        if qty <= 1e-12:
            log("  [STRAT] adopt: balance 0 on the base asset — waiting for the allocation")
            return
        if self.p.adopt_qty <= 0:
            qty = self._dust_safe_qty(qty)
            if qty <= 0:
                return
        self.s["qty"] = qty
        self.s["cost"] = qty * self.p.adopt_cost
        self.s["entry_price"] = self.p.adopt_cost
        self.s["last_buy_price"] = self.p.adopt_cost
        self.s["adopted"] = True
        self._save()
        log(f"  📥 [STRAT] POZITIE ADOPTATA: {qty} @ {self.p.adopt_cost} {self.ccy} — gestionez TP/DCA/SL")
        self._emit(title=f"📥 {self.pair} ADOPTAT {qty:.2f}@{self.p.adopt_cost}",
                   body=f"TP+{self.p.takeprofit_pct}% DCA-{self.p.dca_drop_pct}% SL{self.p.stop_loss_pct}%",
                   source=f"{self.notification_source}-bot", price=self.p.adopt_cost,
                   desktop=self.desktop)

    # -- Percentage sizing; all-zero settings retain fixed sizing --------------
    def _pct_sizing_on(self) -> bool:
        return self.p.pct_sizing_on()

    def _alloc_budget(self) -> float:
        """Return this asset's allocation from the total budget."""
        return self.p.allocated_budget()

    def _effective_entry_amount(self) -> float:
        return self.p.effective_entry_amount()

    def _effective_max_budget(self) -> float:
        return self.p.effective_max_budget()

    def _base_dca_amount(self) -> float:
        return self.p.effective_dca_amount_base()

    # -- Adaptive-volatility shadow observation; does not decide ----------------
    def _effective_dca_amount(self) -> float:
        """Scale base DCA size by volatility when enabled, falling back during warm-up."""
        base = self._base_dca_amount()
        scale_k = float(self.p.dca_vol_scale_k)
        vol_ref = float(self.p.dca_vol_ref)
        k_val = safe_float(scale_k, default=0.0)
        ref_val = safe_pos_float(vol_ref, default=None)
        if not k_val or ref_val is None:
            return base
        try:
            vol = safe_pos_float(self._dca_vol_1h(), default=None)
            if vol is None:
                return base
            scale = safe_pos_float((ref_val / vol) ** k_val, default=None)
            if scale is None:
                return base
        except (ArithmeticError, TypeError, ValueError):
            return base
        return base * max(0.3, min(3.0, scale))

    def _dca_vol_1h(self) -> float | None:
        """Return DCA volatility from OHLC at the same cadence in live and replay."""
        if self.replay_mode:
            closes = [price for _, price in self._shadow_prices]
        else:
            closes = self.client.ohlc_closes(self.pair, self.p.dca_vol_interval)
        return self._hourly_vol_from_closes(closes, self.p.dca_vol_interval)

    _SHADOW_VOL_CACHE: dict = {}
    _TRAIL_VOL_CACHE: dict = {}
    _REPLAY_REGIME_CACHE: dict = {}
    _HOURLY_VOL_CACHE: dict = {}
    _REPLAY_CACHE_MAX = 262144

    def _shadow_vol_1h(self) -> float | None:
        """Return one-hour volatility from local tick history, or None during warm-up."""
        sp = self._shadow_prices
        n_sp = len(sp)
        if n_sp < 20:
            return None
        if self.replay_mode:
            cache_key = (
                n_sp,
                round(sp[-1][0] - sp[0][0], 3),
                sp[0][1],
                sp[1][1],
                sp[n_sp >> 1][1],
                sp[-2][1],
                sp[-1][1],
            )
            if getattr(self, "_shadow_vol_cache_key", None) == cache_key:
                return self._shadow_vol_cache_val
            if cache_key in Strategy._SHADOW_VOL_CACHE:
                val = Strategy._SHADOW_VOL_CACHE[cache_key]
                self._shadow_vol_cache_key = cache_key
                self._shadow_vol_cache_val = val
                return val
        mean_dt = (sp[-1][0] - sp[0][0]) / (n_sp - 1)
        if mean_dt <= 0:
            return None
        pts = list(sp)
        rets = [math.log(pts[i][1] / pts[i - 1][1]) for i in range(1, n_sp)
                if pts[i - 1][1] > 0 and pts[i][1] > 0]
        n_r = len(rets)
        if n_r < 19:
            return None
        mean_r = sum(rets) / n_r
        var_r = sum((r - mean_r) ** 2 for r in rets) / (n_r - 1)
        std = math.sqrt(var_r)
        val = std * math.sqrt(3600.0 / mean_dt) * 100.0
        if self.replay_mode:
            self._shadow_vol_cache_key = cache_key
            self._shadow_vol_cache_val = val
            if len(Strategy._SHADOW_VOL_CACHE) >= Strategy._REPLAY_CACHE_MAX:
                Strategy._SHADOW_VOL_CACHE.clear()
            Strategy._SHADOW_VOL_CACHE[cache_key] = val
        return val

    def _shadow_reentry_line(self, price: float, lsp: float, prag_fix: float) -> None:
        try:
            k_re = self.p.shadow_k_reentry
            vol = self._shadow_vol_1h()
            if vol is None and not self.replay_mode:
                vol = resolve_cached_volatility_1h_pct(self.pair)
            if vol is None:
                log(f"  [SHADOW] adaptive threshold: warm-up ({len(self._shadow_prices)}/20 points)")
                return
            adapt_pct, _ = adaptive_thresholds(vol, k_reentry=k_re)
            if adapt_pct is None:
                adapt_pct = k_re * vol
            prag_adapt = lsp * (1 - adapt_pct / 100)
            verdict = "WOULD HAVE ENTERED" if price <= prag_adapt else "would not have entered either"
            log(f"  [SHADOW] fixed threshold {prag_fix:.2f} vs adaptive {prag_adapt:.2f} "
                f"(vol_1h {vol:.2f}% x k={k_re}) → {verdict}")
        except Exception as e:  # noqa: BLE001 — observational
            log(f"  [SHADOW] calculation error ({e}) — ignored")

    def _effective_reentry_drop_pct(self) -> tuple[float, str]:
        """Return the effective adaptive or fixed reentry threshold.

        Use ``K_REENTRY * one-hour volatility`` when enabled and available. Otherwise
        fail safely to the fixed percentage so missing signals never stop trading.
        """
        if not self.p.reentry_adaptive:
            return self.p.reentry_drop_pct, "fix"
        k_re = self.p.shadow_k_reentry
        try:
            vol = self._shadow_vol_1h()
            if vol is None and not self.replay_mode:
                vol = resolve_cached_volatility_1h_pct(self.pair)
        except Exception as e:  # noqa: BLE001 — observational failure must not stop trading.
            log(f"  [REENTRY-ADAPTIVE] calculation error ({e}) — fallback to fixed")
            return self.p.reentry_drop_pct, "fix (fallback, error)"
        if vol is None:
            return self.p.reentry_drop_pct, f"fix (fallback, warm-up {len(self._shadow_prices)}/20)"
        adapt_drop, _ = adaptive_thresholds(vol, k_reentry=k_re)
        eff_drop = adapt_drop if adapt_drop is not None else (k_re * vol)
        return eff_drop, f"adaptiv (vol_1h {vol:.2f}% x k={k_re})"

    def _effective_reentry_pullback_pct(self) -> tuple[float, str]:
        """Return the dynamic volatility-scaled or fixed pullback threshold from peak.

        When STRAT_REENTRY_PULLBACK_ADAPTIVE is enabled, scale pullback by volatility:
        pullback = clamp(K_PULLBACK * vol_1h, min_pullback, max_pullback).
        """
        if not self.p.reentry_pullback_adaptive:
            return self.p.reentry_pullback_pct, "fixed"
        try:
            vol = self._shadow_vol_1h()
            if vol is None and not self.replay_mode:
                vol = resolve_cached_volatility_1h_pct(self.pair)
        except Exception as e:
            log(f"  [REENTRY-PULLBACK-ADAPTIVE] calculation error ({e}) — fallback to fixed")
            vol = None
        if vol is None:
            return self.p.reentry_pullback_pct, f"fixed (fallback, warm-up {len(self._shadow_prices)}/20)"
        val = max(self.p.reentry_pullback_min, min(self.p.reentry_pullback_max, self.p.reentry_pullback_k * vol))
        return val, f"adaptive (vol_1h {vol:.2f}% x k={self.p.reentry_pullback_k} -> {val:.2f}%)"

    def _effective_trail_pct(self) -> float:
        """Return the clamped volatility-adaptive or fixed trailing pullback.

        Warm-up and errors fall back to the fixed value. Volatile trends receive more
        room while choppy markets receive tighter trailing without buying higher.
        """
        if not self.p.tp_trail_adaptive:
            return self.p.tp_trail_pct
        try:
            vol = self._trail_vol_1h()
        except Exception as e:  # noqa: BLE001 — observational failure must not stop trading.
            log(f"  [STRAT] trailing adaptiv: OHLC indisponibil ({e}) — fallback pe fix")
            vol = None
        if vol is None:
            return self.p.tp_trail_pct
        return max(self.p.tp_trail_min, min(self.p.tp_trail_max, self.p.tp_trail_k * vol))

    def _effective_surge_gain_pct(self) -> float:
        """Return the dynamic volatility-scaled or fixed parabolic surge trigger."""
        if not self.p.surge_dynamic:
            return self.p.surge_gain_pct
        try:
            vol = self._trail_vol_1h()
        except Exception:
            vol = None
        return sr.dynamic_surge_gain_pct(
            vol,
            min_gain_pct=self.p.surge_min_gain_pct,
            max_gain_pct=self.p.surge_max_gain_pct,
            vol_multiplier=self.p.surge_vol_multiplier,
            fallback_gain_pct=self.p.surge_gain_pct,
        )

    def _surge_window_bars(self) -> int:
        interval_hours = max(1.0, float(self.p.trend_interval)) / 60.0
        return max(1, round(self.p.surge_window_hours / interval_hours))

    def _surge_window_move_pct(self, price: float, closes: list | None = None) -> float | None:
        """Calculate the configured-window move, or None when unavailable.

        A missing or disabled window must not use the classifier's different
        horizon as a substitute. Position-gain arming remains independent.
        """
        if not self.p.surge_window_hours or self.p.surge_window_hours <= 0:
            return None
        if closes is None:
            closes = (
                self._surge_replay_closes
                if self._surge_replay_closes is not None else self._trend_closes()
            )
        if not closes:
            return None
        n_bars = self._surge_window_bars()
        offset = 1 if self.replay_mode else 0
        idx = -n_bars - offset
        if abs(idx) > len(closes):
            return None
        past_price = closes[idx]
        if past_price <= 0:
            return None
        return (price - past_price) / past_price * 100.0

    def _trail_vol_1h(self) -> float | None:
        """Return one-hour-normalized volatility at one OHLC cadence in live and replay.

        Different live-tick and replay-bar cadences produced different signals even after
        square-root-of-time scaling. Live mode reads closed
        provider bars; replay is separately validated against the same interval.
        """
        if self.replay_mode:
            sp = self._shadow_prices
            n_sp = len(sp)
            if n_sp < 20:
                return None
            cache_key = (
                n_sp,
                sp[0][1],
                sp[1][1],
                sp[n_sp >> 1][1],
                sp[-2][1],
                sp[-1][1],
                self.p.tp_trail_vol_interval,
            )
            if getattr(self, "_trail_vol_cache_key", None) == cache_key:
                return self._trail_vol_cache_val
            if cache_key in Strategy._TRAIL_VOL_CACHE:
                val = Strategy._TRAIL_VOL_CACHE[cache_key]
                self._trail_vol_cache_key = cache_key
                self._trail_vol_cache_val = val
                return val
            closes = [price for _, price in sp]
            val = self._hourly_vol_from_closes(
                closes, self.p.tp_trail_vol_interval,
            )
            self._trail_vol_cache_key = cache_key
            self._trail_vol_cache_val = val
            if len(Strategy._TRAIL_VOL_CACHE) >= Strategy._REPLAY_CACHE_MAX:
                Strategy._TRAIL_VOL_CACHE.clear()
            Strategy._TRAIL_VOL_CACHE[cache_key] = val
            return val
        closes = self.client.ohlc_closes(self.pair, self.p.tp_trail_vol_interval)
        return self._hourly_vol_from_closes(
            closes, self.p.tp_trail_vol_interval,
        )

    @staticmethod
    def _hourly_vol_from_closes(
        closes: list[float], interval_minutes: float,
    ) -> float | None:
        """Return log-return deviation normalized to one hour for a fixed cadence."""
        itv_f = safe_pos_float(interval_minutes, default=None)
        if not closes or itv_f is None:
            return None
        clean_closes = [
            cf for c in closes[-90:]
            if (cf := safe_pos_float(c, default=None)) is not None
        ]
        closes_slice = tuple(clean_closes)
        if len(closes_slice) < 20:
            return None
        cache_key = (closes_slice, itv_f)
        if cache_key in Strategy._HOURLY_VOL_CACHE:
            return Strategy._HOURLY_VOL_CACHE[cache_key]
        returns = [
            math.log(current / previous)
            for previous, current in zip(closes_slice, closes_slice[1:])
            if previous > 0 and current > 0
        ]
        n_r = len(returns)
        if n_r < 19:
            val = None
        else:
            mean_r = sum(returns) / n_r
            var_r = 0.0
            for r in returns:
                diff = r - mean_r
                var_r += diff * diff
            var_r /= (n_r - 1)
            std = math.sqrt(var_r)
            val = std * math.sqrt(60.0 / itv_f) * 100.0
        if len(Strategy._HOURLY_VOL_CACHE) >= Strategy._REPLAY_CACHE_MAX:
            Strategy._HOURLY_VOL_CACHE.clear()
        Strategy._HOURLY_VOL_CACHE[cache_key] = val
        return val

    # -- Common market regime and trend overlay --------------------------------
    def _trend_closes(self) -> list:
        """Return long-trend closes at the same cadence in live and replay.

        Replay uses injected bar closes; live and paper-live use provider OHLC at
        ``trend_interval``. The optional exit SMA therefore keeps the same cadence.
        """
        if self.replay_mode:
            return [p for _, p in self._shadow_prices]
        try:
            return self.client.ohlc_closes(self.pair, self.p.trend_interval)
        except Exception as e:  # noqa: BLE001 - unavailable data yields an unknown regime.
            log(f"  [STRAT] regime OHLC fetch failed ({e}) - regime unavailable")
            return []

    def _regime_context(self) -> tuple[MarketRegimeDecision, list]:
        """Classify and reuse one completed-bar snapshot across regime policies."""
        if not self.replay_mode:
            now = self._now()
            interval_sec = max(60, self.p.trend_interval * 60)
            slot = int(now // interval_sec)
            settle_sec = min(_REGIME_BAR_SETTLE_MAX_SEC, interval_sec / 4)
            refresh_epoch = slot * interval_sec + settle_sec
            cached = self._regime_bar_context
            if cached and cached[0] == slot:
                _, stable, decision, closes = cached
                if stable or now < refresh_epoch:
                    return decision, closes
        elif (
            len(self._shadow_prices) >= 2
            and type(self._regime_service) is MarketRegimeService
            and "evaluate_closes_for_horizon" not in self._regime_service.__dict__
            and "_trend_closes" not in self.__dict__
        ):
            sp = self._shadow_prices
            n_sp = len(sp)
            replay_key = (
                n_sp,
                sp[0][1],
                sp[1][1],
                sp[n_sp >> 1][1],
                sp[-2][1],
                sp[-1][1],
                self.p.trend_interval,
                self._regime_service.evaluator.strength_threshold,
            )
            if getattr(self, "_replay_regime_key", None) == replay_key:
                return self._replay_regime_val
            cached_ctx = Strategy._REPLAY_REGIME_CACHE.get(replay_key)
            if cached_ctx is not None:
                self._replay_regime_key = replay_key
                self._replay_regime_val = cached_ctx
                return cached_ctx
            closes = self._trend_closes()
            decision = self._regime_service.evaluate_closes_for_horizon(
                closes,
                horizon="long",
                interval_min=self.p.trend_interval,
            )
            res = (decision, closes)
            self._replay_regime_key = replay_key
            self._replay_regime_val = res
            if len(Strategy._REPLAY_REGIME_CACHE) >= Strategy._REPLAY_CACHE_MAX:
                Strategy._REPLAY_REGIME_CACHE.clear()
            Strategy._REPLAY_REGIME_CACHE[replay_key] = res
            return res
        closes = self._trend_closes()
        decision = self._regime_service.evaluate_closes_for_horizon(
            closes,
            horizon="long",
            interval_min=self.p.trend_interval,
        )
        if not self.replay_mode and decision.fresh:
            # Refresh a boundary read once after upstream candle caches settle;
            # the stable result is then reused until the next completed-bar slot.
            self._regime_bar_context = (slot, now >= refresh_epoch, decision, closes)
        return decision, closes

    @staticmethod
    def _regime_matches(decision: MarketRegimeDecision | None, expected: str, *,
                        min_move_pct: float = 0.0, min_samples: int = 3) -> bool:
        """Apply policy thresholds without changing the shared classification."""
        if (decision is None or not decision.fresh or decision.regime != expected
                or (decision.n_samples or 0) < min_samples):
            return False
        move = decision.fitted_move_pct
        return min_move_pct <= 0 or (move is not None and abs(move) >= min_move_pct)

    def _evaluate_reentry_gate(
        self, price: float, regime: MarketRegimeDecision | None, tick_time: float
    ) -> tuple[bool, str]:
        """Evaluate re-entry gate conditions symmetrically across trend overlay and DCA entries."""
        lsp = safe_pos_float(self.s.get("last_sell_price"), None) or safe_pos_float(self.s.get("last_sell"), None)
        if not lsp:
            return True, "no_last_sell"

        cur_peak = safe_pos_float(self.s.get("post_sell_peak"), lsp)
        self.s["post_sell_peak"] = max(cur_peak, price)
        cur_trough = safe_pos_float(self.s.get("post_sell_trough"), lsp)
        self.s["post_sell_trough"] = min(cur_trough, price)
        last_kind = self.s.get("last_exit_kind")
        if last_kind == "STOP":
            cur_sl_low = safe_pos_float(self.s.get("sl_low"), price)
            self.s["sl_low"] = min(cur_sl_low, price)

        # Symmetrical Re-entry Evaluation via PositionExitManager
        if self.exit_manager is None:
            self.exit_manager = self._init_exit_manager()
        mgr = self.exit_manager
        mgr.config.reentry_drop_pct = vol_pct_to_ratio(self.p.reentry_drop_pct)
        mgr.config.reentry_bear_bounce_pct = vol_pct_to_ratio(self.p.reentry_bear_bounce_pct)
        mgr.config.reentry_sl_bounce_pct = vol_pct_to_ratio(self.p.reentry_sl_bounce_pct)
        mgr.config.reentry_ttl_hours = self.p.reentry_ttl_hours
        mgr.config.reentry_tolerance_pct = self.p.reentry_tolerance_pct
        mgr.config.reentry_hybrid_enabled = self.p.reentry_hybrid_enabled
        mgr.config.reentry_peak_relative = self.p.reentry_peak_relative
        needs_reentry_regime = bool(
            self.p.trend_overlay
            or self.p.reentry_hybrid_enabled
            or (last_kind == "STOP" and self.p.sl_mhdsl_enabled)
        )
        if regime is None and needs_reentry_regime:
            try:
                regime, _ = self._regime_context()
            except Exception:
                regime = None
        trend_up = (
            self._is_trend_up(regime)
            if (needs_reentry_regime or self.p.reentry_bear_bounce_pct > 0)
            else False
        )
        trend_down = self._is_trend_down(regime) if needs_reentry_regime else False
        is_bull_reentry = bool(
            last_kind != "STOP"
            and self.p.reentry_hybrid_enabled
            and (trend_up or self.p.reentry_peak_relative)
        )
        if is_bull_reentry or not self.replay_mode:
            pullback_pct, _ = self._effective_reentry_pullback_pct()
        else:
            pullback_pct = self.p.reentry_pullback_pct
        mgr.config.reentry_pullback_pct = vol_pct_to_ratio(pullback_pct)
        if (not is_bull_reentry and last_kind != "STOP") or not self.replay_mode:
            drop_pct, _ = self._effective_reentry_drop_pct()
        else:
            drop_pct = self.p.reentry_drop_pct
        if self.replay_mode:
            vol_1h, vol_source = 0.0, "default"
        else:
            vol_1h, vol_source = self._vol_1h_ratio()
        reentry_snap = ReentrySnapshot(
            symbol=self.pair,
            current_price=price,
            last_sell_price=lsp,
            last_exit_kind=last_kind,
            post_sell_peak=self.s.get("post_sell_peak"),
            post_sell_trough=self.s.get("post_sell_trough"),
            current_time_s=tick_time,
            last_sell_time_s=self.s.get("last_sell_ts"),
            trend_up=trend_up,
            trend_down=trend_down,
            vol_1h=vol_1h,
            vol_source=vol_source,
        )
        reentry_dec = mgr.evaluate_reentry(reentry_snap, custom_drop_pct=vol_pct_to_ratio(drop_pct))
        return reentry_dec.should_enter, reentry_dec.reason

    def _restore_reentry_state(self) -> None:
        if self.exit_manager is not None:
            self.exit_manager.restore_reentry_state(self.pair)

    def _check_advanced_profit_guards(
        self, price: float, avg: float, held: float,
        closes: list | None = None, tick_time: float | None = None,
    ) -> bool:
        if not avg or held <= 1e-12:
            return False

        current_gain = (price - avg) / avg * 100.0

        # 1. Micro-gradient 2X Fast Profit Guard (Horizon 3)
        if (
            self.p.fast_profit_guard
            and self.p.takeprofit_pct > 0
            and self.p.fast_profit_mult > 0
            and current_gain >= self.p.takeprofit_pct * self.p.fast_profit_mult
        ):
            triggered, gain, drop = sr.check_fast_profit_reversal(
                current_price=price,
                avg_cost=avg,
                base_tp_pct=self.p.takeprofit_pct,
                mult=self.p.fast_profit_mult,
                shadow_prices=self._shadow_prices,
                window_sec=self.p.fast_profit_window_min * 60.0,
                drop_pct=self.p.fast_profit_drop_pct,
                current_time=tick_time,
            )
            if triggered:
                exit_px = round(price * 0.999, self.price_dec)
                if self._request_market_exit(price, "TP"):
                    log(f"  [STRAT] FAST PROFIT GUARD EXIT (+{gain:.2f}% gain, -{drop:.2f}% drop in {self.p.fast_profit_window_min:.0f}m) -> reference {exit_px}")
                    return True

        # 2. Parabolic Surge Guard (2-3 Day Exhaustion, Horizon 4)
        if self.p.surge_guard:
            surge_peak = max(safe_pos_float(self.s.get("surge_peak"), price), price)
            self.s["surge_peak"] = surge_peak
            peak_gain = (surge_peak - avg) / avg * 100.0
            is_armed = bool(self.s.get("surge_active"))
            min_g = (
                min(self.p.surge_min_gain_pct, self.p.surge_max_gain_pct, self.p.surge_gain_pct)
                if self.p.surge_dynamic else self.p.surge_gain_pct
            )
            in_min_surge_zone = (
                is_armed
                or (min_g > 0 and peak_gain >= min_g)
                or (self.p.surge_move_pct > 0 and current_gain >= min_g * 0.8)
            )
            if self.replay_mode and in_min_surge_zone:
                v = self._trail_vol_1h()
                if v is not None and v > self._max_active_surge_vol:
                    self._max_active_surge_vol = v
            if (
                self.p.surge_dynamic
                and not in_min_surge_zone
                and min_g > 0
                and type(self)._effective_surge_gain_pct is Strategy._effective_surge_gain_pct
                and "_effective_surge_gain_pct" not in self.__dict__
                and type(self)._trail_vol_1h is Strategy._trail_vol_1h
                and "_trail_vol_1h" not in self.__dict__
            ):
                eff_surge_gain = min_g
            else:
                eff_surge_gain = (
                    self._effective_surge_gain_pct()
                    if self.p.surge_dynamic else self.p.surge_gain_pct
                )
            if (
                is_armed
                or (eff_surge_gain > 0 and peak_gain >= eff_surge_gain)
                or (self.p.surge_move_pct > 0 and current_gain >= eff_surge_gain * 0.8)
            ):
                window_move = self._surge_window_move_pct(
                    price, None if self.replay_mode else closes)
                pos_surged = eff_surge_gain > 0 and (current_gain >= eff_surge_gain or peak_gain >= eff_surge_gain)
                if self.replay_mode and not pos_surged and self.p.surge_move_pct > 0 and current_gain >= eff_surge_gain * 0.8:
                    self._surge_win_check_ts.append(tick_time if tick_time is not None else self._now())
                if not is_armed:
                    win_surged = (
                        window_move is not None
                        and self.p.surge_move_pct > 0
                        and window_move >= self.p.surge_move_pct
                        and current_gain >= eff_surge_gain * 0.8
                    )
                    if pos_surged or win_surged:
                        is_armed = True
                        self.s["surge_active"] = True
                        log(f"  [STRAT] PARABOLIC SURGE ARMED (+{peak_gain:.2f}% peak gain, window move {window_move})")
                elif current_gain < self.p.takeprofit_pct:
                    self.s["surge_active"] = False
                    is_armed = False

                triggered, reason = sr.check_parabolic_surge_exhaustion(
                    current_price=price,
                    avg_cost=avg,
                    surge_peak=surge_peak,
                    surge_gain_pct=eff_surge_gain,
                    exit_pullback_pct=self.p.surge_exit_pullback_pct,
                    window_move_pct=window_move,
                    surge_move_pct=self.p.surge_move_pct,
                    surge_active=is_armed,
                )
                if triggered:
                    exit_px = round(price * 0.999, self.price_dec)
                    if self._request_market_exit(
                        price, "TP", min_profit_pct=sr.parabolic_surge_profit_floor_pct(
                            eff_surge_gain, self.p.surge_exit_pullback_pct),
                    ):
                        log(f"  [STRAT] SURGE EXHAUSTION EXIT ({reason}) peak {surge_peak:.{self.price_dec}f} -> reference {exit_px}")
                        return True

        # 3. Slow-Grind Guard (1-3 Week Drift, Horizon 5)
        if self.p.slow_grind_guard:
            slow_peak = max(safe_pos_float(self.s.get("slow_grind_peak"), price), price)
            self.s["slow_grind_peak"] = slow_peak
            entry_ts = self.s.get("entry_ts")
            if entry_ts is not None and current_gain >= self.p.slow_grind_min_gain_pct:
                n = self.p.trend_sma_n
                sma = sum(closes[-n:]) / n if closes and len(closes) >= n else None
                triggered, reason = sr.check_slow_grind_exhaustion(
                    current_price=price,
                    avg_cost=avg,
                    entry_ts=entry_ts,
                    current_ts=tick_time if tick_time is not None else self._now(),
                    min_days=self.p.slow_grind_days,
                    min_gain_pct=self.p.slow_grind_min_gain_pct,
                    recent_peak=slow_peak,
                    shadow_prices=self._shadow_prices,
                    flash_window_sec=self.p.slow_grind_flash_window_min * 60.0,
                    flash_drop_pct=self.p.slow_grind_flash_drop_pct,
                    structural_drop_pct=self.p.slow_grind_hourly_drop_pct,
                    sma_value=sma,
                )
                if triggered:
                    exit_px = round(price * 0.999, self.price_dec)
                    if self._request_market_exit(price, "TP"):
                        log(f"  [STRAT] SLOW GRIND EXIT ({reason}) peak {slow_peak:.{self.price_dec}f} -> reference {exit_px}")
                        return True

        return False

    def _overlay_step(self, price: float, regime: MarketRegimeDecision,
                      closes: list, tick_time: float | None = None) -> bool:
        """Apply the regime overlay and return whether it handled the tick.

        Confirmed uptrend uses top-up, hold, and trailing or SMA-break exit. A range
        returns False so classic DCA/TP logic runs.
        """
        if self.s.get("trend_mode"):
            avg = self._avg()

            if self._check_advanced_profit_guards(price, avg, self.s["qty"], closes, tick_time):
                return True

            # 4. Dynamic Profit Ratchet Trailing (Horizon 2)
            peak = max(safe_pos_float(self.s.get("trend_peak"), price), price)
            self.s["trend_peak"] = peak
            if self.p.trend_trail_dynamic and avg:
                peak_gain = (peak - avg) / avg * 100.0
                eff_trail = sr.dynamic_trend_trail_pct(
                    peak_gain_pct=peak_gain,
                    base_trail_pct=self.p.trend_trail_base_pct,
                    min_trail_pct=self.p.trend_trail_min_pct,
                    ratchet_k=self.p.trend_trail_ratchet_k,
                    gain_threshold_pct=self.p.trend_trail_gain_threshold,
                    surge_guard_active=self.p.surge_guard,
                )
            else:
                eff_trail = self.p.trend_trail_pct
            trail_stop = peak * (1 - eff_trail / 100)
            broke = False
            if self.p.trend_exit_break:
                n = self.p.trend_sma_n
                sma = sum(closes[-n:]) / n if len(closes) >= n else None
                broke = sma is not None and price < sma
            if (price <= trail_stop or broke) and self.s["qty"] > 1e-12:
                exit_px = round(price * 0.999, self.price_dec)
                if self._request_market_exit(price, "TP"):
                    log(f"  [STRAT] TREND EXIT ({'break' if broke else 'trailing'} "
                        f"{eff_trail:.2f}%) peak {peak:.{self.price_dec}f} -> reference {exit_px}")
            else:
                self._cancel_orders("sell", exclude_market=True)  # Ride the trend; do not sell.
            return True
        up = self._regime_matches(
            regime,
            "bull",
            min_samples=self.p.regime_min_samples,
        )
        pending_trend_entry = None
        for o in self.s["orders"]:
            if o["side"] == "buy" and o.get("kind") == "TREND_ENTRY":
                pending_trend_entry = o
                break
        if pending_trend_entry:
            if up:
                return True                         # Wait for the top-up fill.
            self._cancel_open("buy")                # Signal disappeared before fill.
            log("  [STRAT] TREND ENTER cancelled: the signal vanished before the fill")
            return False                            # Return to the range strategy.
        if up and self.s["spent"] + self.p.trend_topup <= self._effective_max_budget():
            # Brand new entry while flat must obey re-entry restrictions!
            if self.s["qty"] <= 1e-12:
                allowed, reason = self._evaluate_reentry_gate(price, regime, tick_time)
                if not allowed:
                    self._reentry_blocked_tick = (tick_time, price, reason)
                    log(f"  [STRAT] trend overlay entry blocked by re-entry: {reason}")
                    return False
            self._cancel_orders("buy")              # Cancel pending range orders.
            self._cancel_orders("sell")
            if self.s["orders"]:                    # Live mode waits for terminal confirmations.
                if self.s["qty"] <= 1e-12:
                    self._restore_reentry_state()
                return True
            if not self._place("buy", self._qty_for(self.p.trend_topup, price), price,
                               kind="TREND_ENTRY", amount=self.p.trend_topup):
                if self.s["qty"] <= 1e-12:
                    self._restore_reentry_state()
                return True
            log(f"  [STRAT] TREND ENTER pending: top-up {self.p.trend_topup} {self.ccy} @ {price} "
                f"(common {regime.horizon} regime: {regime.regime})")
            return True
        return False

    def step(
        self,
        price: float,
        timestamp: float | None = None,
        high: float | None = None,
        low: float | None = None,
    ) -> None:
        clean_price = _clean_pos_float(price)
        if clean_price is None:
            log("  ! [STRAT] invalid price — deferring step")
            return
        price = clean_price
        if high is not None:
            high = _clean_pos_float(high)
        if self.s.get("pending_intent"):
            log("  [STRAT] pre-submit intent in reconciliation — deferring new decisions")
            return
        clean_ts = _clean_nonneg_float(timestamp)
        had_persisted_reconcile = bool(self.s.get("ledger_reconcile_required"))
        if self.observe_only:
            tick_time = clean_ts if clean_ts is not None else self._now()
            self._shadow_prices.append((tick_time, price))
            if not self.dry_run and not self.s.get("orders"):
                self.s["ledger_reconcile_required"] = True
            else:
                return
        held = self.s["qty"]
        if (not self.dry_run and self.s.get("ledger_reconcile_required")
                and (held > 1e-12 or self.observe_only) and not self.s.get("orders")):
            try:
                precision = self.client.pair_precision(self.pair)
                base = precision.base_asset if precision else ""
                ledger_val = self.client.free_balance(base)
                if ledger_val is None:
                    log("  ! [STRAT] ledger reconciliation unavailable (balance query returned None); decisions blocked")
                    return
                ledger_qty = _clean_nonneg_float(ledger_val)
                if ledger_qty is None:
                    log("  ! [STRAT] ledger reconciliation unavailable (invalid balance); decisions blocked")
                    return
            except (ProviderError, TypeError, ValueError, OverflowError) as exc:
                log(f"  ! [STRAT] ledger reconciliation unavailable ({exc}); decisions blocked")
                return
            tolerance = max(2 * 10.0 ** -max(self.vol_dec, 1), 1e-12)
            if ledger_qty <= tolerance:
                if held > 1e-12:
                    log(
                        f"  [STRAT] LEDGER RECONCILE: position closed on venue ({base}=0); resetting cycle"
                    )
                    self.s["qty"] = 0.0
                    self.s["cost"] = 0.0
                    self.s["spent"] = 0.0
                    self.s["dca_buys"] = 0
                    self.s["entry_price"] = None
                    self.s["last_buy_price"] = None
                    self.s["last_sell_price"] = price
                    self.s["last_exit_kind"] = "TP"
                    self.s["last_sell_ts"] = clean_ts if clean_ts is not None else self._now()
                    self.s["entry_ts"] = None
                    self.s["trail_peak"] = None
                    self.s["trail_stop"] = None
                    self.s["trend_mode"] = False
                    self.s["trend_peak"] = None
                    self.s["trend_confirm_count"] = 0
                    self.s["pending_exit"] = None
                    self.s["surge_active"] = False
                    self.s["surge_peak"] = None
                    self.s["slow_grind_active"] = False
                    self.s["slow_grind_peak"] = None
                    self.s["sl_bounce_state"] = {}
                    self.s["sl_episode_state"] = {}
                    self.s["tp_ride_state"] = {}
                    self.s["reentry_state"] = {}
                    if self.exit_manager is not None:
                        self.exit_manager.clear_state(self.pair)
                        self.exit_manager.bind_state(self.s)
                    self.s["ledger_reconcile_required"] = False
                    self._save()
                else:
                    self.s["ledger_reconcile_required"] = False
                    if had_persisted_reconcile:
                        self._save()
                return
            if ledger_qty + tolerance < held:
                log(
                    f"  ! [STRAT] LEDGER RECONCILE: state qty={held:.{self.vol_dec}f} -> "
                    f"available {base}={ledger_qty:.{self.vol_dec}f}; synchronizing state"
                )
                ratio = ledger_qty / held if held > 0 else 1.0
                self.s["qty"] = ledger_qty
                self.s["cost"] = float(self.s.get("cost", 0.0)) * ratio
                self.s["spent"] = float(self.s.get("spent", 0.0)) * ratio
                self.s["ledger_reconcile_required"] = False
                self._save()
                return
            if self.observe_only and ledger_qty > held + tolerance and (ledger_qty - held) * price >= 5.0:
                log(
                    f"  ! [STRAT] LEDGER RECONCILE: external BUY on venue ({base}: "
                    f"{held:.{self.vol_dec}f} -> {ledger_qty:.{self.vol_dec}f}); synchronizing state"
                )
                added_cost = (ledger_qty - held) * price
                self.s["qty"] = ledger_qty
                self.s["cost"] = float(self.s.get("cost", 0.0)) + added_cost
                self.s["spent"] = float(self.s.get("spent", 0.0)) + added_cost
                if held <= tolerance or self.s.get("entry_price") is None:
                    self.s["entry_price"] = price
                    self.s["entry_ts"] = clean_ts if clean_ts is not None else self._now()
                else:
                    self.s["dca_buys"] = int(self.s.get("dca_buys") or 0) + 1
                self.s["last_buy_price"] = price
                self.s["ledger_reconcile_required"] = False
                self._save()
                return
            self.s["ledger_reconcile_required"] = False
            if had_persisted_reconcile:
                self._save()
        if self.observe_only:
            return
        entry_px = sr.entry_price(price, self.p.entry_discount_pct)   # = price*(1 - disc%)
        # Live uses wall time while replay injects bar time. Without this distinction, a
        # backtest processing hundreds of bars per second produces meaningless hourly
        # volatility and changes the adaptive threshold.
        if clean_ts is not None:
            self.set_simulated_time(clean_ts)
            tick_time = clean_ts
        else:
            tick_time = self._now()
        self._shadow_prices.append((tick_time, price))
        if self._surge_replay_closes is not None:
            self._surge_replay_closes.append(price)

        # While adoption is pending, do not buy a new entry; the allocation is in transit.
        if self.p.adopt_cost > 0 and not self.s.get("adopted") and held <= 1e-12:
            self._maybe_adopt()
            if not self.s.get("adopted"):
                return
            held = self.s["qty"]

        # STOP-LOSS is a safety invariant and takes priority over all regime logic,
        # including overlay hold and trailing behavior.
        if held > 1e-12 and self._check_stop_loss(price):
            return
        # Reconcile a MARKET exit from an earlier tick before any new TP/DCA decision,
        # even if price has recovered meanwhile.
        if self._has_pending_market_exit():
            return
        if self.s.get("pending_exit"):
            self._continue_market_exit(price)
            return

        regime = None
        regime_closes = []
        needs_regime = self.p.trend_overlay or self.p.reentry_hybrid_enabled or (
            held > 1e-12 and (
                self.p.dca_trend_brake
                or self.p.tp_dynamic_flat
                or self.p.surge_guard
                or self.p.slow_grind_guard
            )
        )
        if needs_regime and not (
            self.replay_mode
            and "_regime_context" not in self.__dict__
            and "_trend_closes" not in self.__dict__
            and type(self._regime_service) is MarketRegimeService
            and "evaluate_closes_for_horizon" not in self._regime_service.__dict__
            and (
                (
                    self.p.trend_overlay
                    and bool(self.s.get("trend_mode"))
                    and not self.p.trend_exit_break
                    and not self.p.slow_grind_guard
                )
                or (
                    not self.p.trend_overlay
                    and held <= 1e-12
                    and self._has_open("buy")
                )
            )
        ):
            regime, regime_closes = self._regime_context()

        # Combine range DCA/TP with trend hold/trailing behavior.
        self._reentry_blocked_tick = None
        if (self.p.trend_overlay
                and self._overlay_step(price, regime, regime_closes, tick_time=tick_time)):
            return

        if held <= 1e-12:
            if self._has_open("buy"):
                return
            blocked_tick = getattr(self, "_reentry_blocked_tick", None)
            if blocked_tick is not None and blocked_tick[:2] == (tick_time, price):
                allowed, reason = False, blocked_tick[2]
            else:
                allowed, reason = self._evaluate_reentry_gate(price, regime, tick_time)
            if not allowed:
                log(f"  [STRAT] re-entry blocked: {reason}")
                return
            lsp = self.s.get("last_sell_price") or self.s.get("last_sell")
            if lsp:
                log(f"  [STRAT] re-entry approved: {reason} — placing ENTRY")
            entry_amt = self._effective_entry_amount()
            budget = self._effective_max_budget()
            if self.s["spent"] + entry_amt > budget:
                self._restore_reentry_state()
                log(f"  [STRAT] cap {budget:.0f} {self.ccy} reached — not entering")
                return
            if not self._place("buy", self._qty_for(entry_amt, entry_px),
                               entry_px, kind="ENTRY", amount=entry_amt):
                self._restore_reentry_state()
            return

        avg = self._avg()
        if held > 1e-12 and self.s.get("entry_ts") is None:
            self.s["entry_ts"] = tick_time

        eff_tp = self.p.takeprofit_pct
        if self.p.tp_dynamic_flat and regime and regime.fresh and regime.regime == "sideways":
            thresh = 2.0
            if hasattr(self, "_regime_service") and hasattr(self._regime_service, "evaluator"):
                thresh = getattr(self._regime_service.evaluator, "strength_threshold", 2.0)
            eff_tp = sr.dynamic_flat_tp_pct(
                strength=regime.strength,
                strength_threshold=thresh,
                tp_min_pct=self.p.tp_min_pct,
                tp_max_pct=self.p.tp_max_pct,
                base_tp_pct=self.p.takeprofit_pct,
            )

        trail_armed = _clean_pos_float(self.s.get("trail_peak")) is not None
        trend_hold_active = self.p.tp_trend_hold and not bool(self.p.tp_tranches)
        tp_ref_price = high if high is not None else price
        if (self.p.enable_takeprofit and avg and trend_hold_active
                and (trail_armed or tp_ref_price >= sr.tp_price(avg, eff_tp))):
            # MDHTP (Multi-Horizon Dynamic Take-Profit) via PositionExitManager
            if self.p.tp_mdhtp_enabled:
                if self.exit_manager is None:
                    self.exit_manager = self._init_exit_manager()
                if (
                    not self.replay_mode
                    or self.pair in self.exit_manager.tp_ride_state
                    or (
                        self.p.tp_trend_hold
                        and self._is_trend_up(regime)
                        and high is not None
                        and high > price
                    )
                ):
                    eff_trail = self._effective_trail_pct()
                else:
                    eff_trail = self.p.tp_trail_pct
                self.exit_manager.config.tp_pullback_pct = eff_trail / 100.0
                self.exit_manager.config.tp_trend_ride_enabled = self.p.tp_trend_hold
                self.exit_manager.config.tp_profit_floor_min_pct = (self.p.tp_trail_profit_floor_pct or 0.0) / 100.0
                self.exit_manager.config.tp_stagnation_max_sec = self.p.tp_stagnation_max_sec
                self.exit_manager.config.fast_profit_guard_enabled = self.p.fast_profit_guard
                self.exit_manager.config.fast_profit_mult = self.p.fast_profit_mult
                self.exit_manager.config.fast_profit_window_sec = self.p.fast_profit_window_min * 60.0
                self.exit_manager.config.fast_profit_drop_pct = self.p.fast_profit_drop_pct

                inc_vol = not self.replay_mode
                inc_shd = bool(self.p.fast_profit_guard) or not self.replay_mode
                snapshot = self._create_position_snapshot(
                    price, avg, held, regime=regime, include_vol=inc_vol, include_shadow=inc_shd
                )
                prev_ride_state = dict(self.exit_manager.tp_ride_state.get(self.pair, {}))
                if high is not None and high > price:
                    high_snap = self._create_position_snapshot(
                        high, avg, held, regime=regime, include_vol=inc_vol, include_shadow=inc_shd
                    )
                    existing_ride = self.exit_manager.tp_ride_state.get(self.pair)
                    if isinstance(existing_ride, dict) and existing_ride.get("buy_price") is not None:
                        ride_bp = _clean_pos_float(existing_ride.get("buy_price"))
                        if ride_bp is None or not math.isclose(ride_bp, avg, rel_tol=1e-9, abs_tol=1e-12):
                            self.exit_manager.tp_ride_state.pop(self.pair, None)
                            existing_ride = None
                    if isinstance(existing_ride, dict):
                        cur_peak = _clean_pos_float(existing_ride.get("peak_price")) or 0.0
                        if high > cur_peak:
                            existing_ride["peak_price"] = high
                            existing_ride["peak_gain"] = high_snap.gain_ratio
                            existing_ride["last_peak_time"] = high_snap.current_time_s
                    elif snapshot.trend_up and self.exit_manager.config.tp_trend_ride_enabled:
                        self.exit_manager.evaluate_take_profit(
                            high_snap, custom_threshold=eff_tp / 100.0
                        )
                    prev_ride_state = dict(self.exit_manager.tp_ride_state.get(self.pair, {}))
                decision = self.exit_manager.evaluate_take_profit(
                    snapshot, custom_threshold=eff_tp / 100.0
                )
                if decision.action == ExitAction.SELL and decision.should_exit:
                    exit_px = round(price * 0.999, self.price_dec)
                    use_soft = str(decision.reason or "").startswith("mdhtp_peak_pullback")
                    profit_floor = self._trail_profit_floor_price(avg) if use_soft else None
                    if use_soft and profit_floor is not None and exit_px < profit_floor:
                        if prev_ride_state:
                            self.exit_manager.tp_ride_state[self.pair] = prev_ride_state
                        log(f"  [STRAT] soft trailing blocked: reference {exit_px:.{self.price_dec}f} below the "
                            f"profitable floor {profit_floor:.{self.price_dec}f}; hard stop stays MARKET")
                        return
                    if self._request_market_exit(price, "TP", soft_floor=use_soft):
                        log(f"  [STRAT] MDHTP EXIT ({decision.reason}) -> reference {exit_px}")
                    return
                elif decision.reason == "not_in_tp_territory":
                    self.s["trail_peak"] = None
                    self.s["trail_stop"] = None
                    self._cancel_orders("sell", exclude_market=True)
                    return
                elif decision.action == ExitAction.HOLD:
                    ride_state = self.exit_manager.tp_ride_state.get(self.pair)
                    peak_px = (safe_pos_float(ride_state.get("peak_price"), None) if isinstance(ride_state, dict) else None) or price
                    self.s["trail_peak"] = peak_px
                    if self._check_advanced_profit_guards(price, avg, held, regime_closes, tick_time):
                        return
                    self._cancel_orders("sell", exclude_market=True)
                    log(f"  [STRAT] above the TP, RIDING IT ({decision.reason}) (peak {peak_px:.{self.price_dec}f})")
                    return

            # Arm trailing at the first TP crossing and keep it armed through exit, even
            # if price later falls below TP, so the target pullback does not reset the peak.
            peak = max(safe_pos_float(self.s.get("trail_peak"), tp_ref_price), tp_ref_price)
            self.s["trail_peak"] = peak

            if self._check_advanced_profit_guards(price, avg, held, regime_closes, tick_time):
                return

            eff_trail = self._effective_trail_pct()   # Adaptive when enabled, otherwise fixed.
            candidate_stop = peak * (1 - eff_trail / 100)
            # A trailing stop ratchets one way: volatility may widen distance for future
            # peaks but cannot surrender profit already protected by the current stop.
            previous_stop = safe_pos_float(self.s.get("trail_stop"), default=None)
            trail_stop = max(candidate_stop, previous_stop or candidate_stop)
            self.s["trail_stop"] = trail_stop
            # Use the same conservative reference as the MARKET order. Apply the floor
            # to that reference rather than the raw observed price so the 0.1% buffer
            # cannot turn an exit exactly at the floor into an implicit loss.
            exit_px = round(price * 0.999, self.price_dec)
            profit_floor = self._trail_profit_floor_price(avg)
            if price <= trail_stop and profit_floor is not None and exit_px < profit_floor:
                log(f"  [STRAT] soft trailing blocked: reference {exit_px:.{self.price_dec}f} below the "
                    f"profitable floor {profit_floor:.{self.price_dec}f}; hard stop stays MARKET")
                # Re-evaluate on the next tick. Leave no persistent order and do not open
                # a contradictory DCA on the tick that triggered trailing.
                return
            if price <= trail_stop:
                if self._request_market_exit(price, "TP", soft_floor=True):
                    log(f"  [STRAT] trailing: pullback {eff_trail:.2f}% from peak "
                        f"{peak:.{self.price_dec}f} -> exit reference {exit_px}")
                # Do not open a contradictory DCA on the exit tick.
                return
            else:
                self._cancel_orders("sell", exclude_market=True)
                log(f"  [STRAT] above the TP, RIDING IT (peak {peak:.{self.price_dec}f}, "
                    f"trail-stop {trail_stop:.{self.price_dec}f})")
        elif self.p.enable_takeprofit and avg and trend_hold_active:
            # Below TP with ride enabled, do not place a fixed TP. Wait to cross TP and
            # arm trailing; stop-loss remains the safety exit.
            self.s["trail_peak"] = None
            self.s["trail_stop"] = None
            self._cancel_orders("sell", exclude_market=True)
        elif self.p.enable_takeprofit and avg:
            self.s["trail_peak"] = None   # below TP/classic mode: reset peak
            self.s["trail_stop"] = None
            # Optional tranche TP (STRAT_TP_TRANCHES="3:50,6:50") sells gradually.
            # No configured tranches means the classic one-order full TP default.
            tranches = self.p.tp_tranches or [(eff_tp, 100.0)]
            desired, rem = [], held
            for i, (pct, share) in enumerate(tranches):
                # The final tranche sells the remainder, risking "Insufficient funds".
                # Venue balance may be one decimal smaller than internally tracked holdings;
                # leave the same dust buffer used during adoption.
                q = self._dust_safe_qty(rem) if i == len(tranches) - 1 \
                    else min(rem, round(held * share / 100, self.vol_dec))
                rem = round(rem - q, self.vol_dec)
                if q > 0:
                    desired.append((round(sr.tp_price(avg, pct), self.price_dec), q))
            if self.ordermin and any(q < self.ordermin for _, q in desired):
                safe_held = self._dust_safe_qty(held)
                desired = (
                    [(round(sr.tp_price(avg, tranches[0][0]), self.price_dec), safe_held)]
                    if safe_held >= self.ordermin
                    else []
                )
            sells = [o for o in self.s["orders"] if o["side"] == "sell"]
            ok = (len(sells) == len(desired) and
                  all(not o.get("cancel_requested") for o in sells) and
                  all(abs(o["price"] - p) / p <= 0.001 and abs(o["vol"] - q) <= 1e-9
                      for o, (p, q) in zip(sorted(sells, key=lambda x: x["price"]),
                                           sorted(desired))))
            if not ok:
                self._cancel_orders("sell")
                if not any(o["side"] == "sell" for o in self.s["orders"]):
                    for p_, q_ in desired:
                        self._place("sell", q_, p_, kind="TP")

        effective_dca_drop = sr.progressive_dca_drop_pct(
            self.p.dca_drop_pct,
            self.p.dca_spacing_growth_pct,
            self.s["dca_buys"],
        )
        effective_dca_amount = self._effective_dca_amount()
        in_sl_bounce = (
            self.pair in self.exit_manager.sl_bounce_state
            if self.exit_manager is not None
            else False
        )
        if (self.s["dca_buys"] < self.p.max_dca_buys
                and not in_sl_bounce
                and self.s["last_buy_price"]
                # DCA threshold + "close to the threshold" = reached (rule shared with the backtest)
                and sr.dca_price_hit(
                    price, self.s["last_buy_price"], effective_dca_drop,
                    self.p.reentry_tolerance_pct,
                )
                and self.s["spent"] < self._effective_max_budget()
                and not (
                    self.p.dca_trend_brake
                    and self._regime_matches(
                        regime, "bear", min_move_pct=self.p.dca_brake_min_pct,
                        min_samples=self.p.regime_min_samples,
                    )
                )
                and not self._has_open("buy")):
            # Resize the request, not an already accepted order. Unknown funding
            # cannot justify a new BUY. Venue preflight still owns fee/minimum rules.
            try:
                free_quote = self.client.free_balance(self.ccy)
            except Exception as exc:
                log(f"  [STRAT] DCA deferred: quote balance unavailable ({exc})")
                return
            clean_quote = _clean_nonneg_float(free_quote)
            if clean_quote is None:
                log("  [STRAT] DCA deferred: quote balance is not a finite non-negative number")
                return
            available = min(clean_quote, self._effective_max_budget() - self.s["spent"])
            spend = min(effective_dca_amount, available)
            qty = self._qty_for(spend, entry_px)
            if spend < effective_dca_amount or qty * round(entry_px, self.price_dec) > available:
                # Round down when funds or the remaining cycle budget are binding.
                entry_px = round(entry_px, self.price_dec)
                qty = (math.floor(spend / entry_px * 10 ** self.vol_dec) / 10 ** self.vol_dec
                       if entry_px > 0 else 0.0)
            if qty <= 0 or qty < self.ordermin:
                log(f"  [STRAT] dip {price}: DCA skipped — available {spend:.2f} "
                    f"{self.ccy} too small for a fill")
            else:
                partial = " (resized to available funds/budget)" if spend < effective_dca_amount else ""
                log(
                    f"  [STRAT] dip {price} <= {self.s['last_buy_price']}"
                    f"×(1-{effective_dca_drop}%) "
                    f"(tol {self.p.reentry_tolerance_pct}%) — "
                    f"DCA {spend:.0f}{partial}"
                )
                self._place("buy", qty, entry_px, kind="DCA", amount=spend)

    # -- main loop -------------------------------------------------------------
    def run(self) -> None:
        mode = "avg_tp" if self.p.enable_takeprofit else "dca_only"
        exec_banner = "[OBSERVE ONLY — AI MANAGED]" if self.observe_only else ("[PAPER]" if self.dry_run else "⚠ REAL — BANI")
        log(f"  === STRATEGIE {self.venue_label.upper()} PORNITA ===")
        log(f"      pereche    : {self.pair}   {exec_banner}")
        log(f"      mod        : {mode}")
        log(f"      intrare    : {self._effective_entry_amount():.0f} {self.ccy} @ market-{self.p.entry_discount_pct}%")
        log(f"      DCA        : {self._base_dca_amount():.0f} {self.ccy} la -{self.p.dca_drop_pct}% (max {self.p.max_dca_buys})")
        if self._pct_sizing_on():
            log(f"      SIZING %   : felie {self.p.alloc_pct}% × total {self.p.total_budget:.0f} "
                f"= {self._alloc_budget():.0f} {self.ccy} (entry {self.p.entry_pct}%, DCA {self.p.dca_pct}%)")
        if self.p.dca_spacing_growth_pct > 0:
            log(
                "      DCA growth : +"
                f"{self.p.dca_spacing_growth_pct}pp after each executed DCA"
            )
        if self.p.dca_vol_scale_k:
            log(
                f"      DCA vol    : k={self.p.dca_vol_scale_k}, "
                f"reference={self.p.dca_vol_ref}%, "
                f"OHLC={self.p.dca_vol_interval}m"
            )
        log(f"      take-profit: +{self.p.takeprofit_pct}%" if self.p.enable_takeprofit else "      take-profit: off")
        log(f"      PLAFON     : {self._effective_max_budget():.0f} {self.ccy} / ciclu")
        if self.fee_note:
            log(f"      ! {self.fee_note}; TP={self.p.takeprofit_pct}%")
        self._maybe_adopt()
        check_min = safe_pos_float(self.p.check_minutes, default=None)
        sleep_sec = max(1.0, check_min * 60.0) if check_min is not None else 60.0
        try:
            while True:
                price = safe_pos_float(self.client.get_current_price(self.pair), default=None)
                if price is None:
                    log("  [STRAT] price unavailable — retrying")
                    time.sleep(sleep_sec)
                    continue
                try:
                    # After a disk error, make no new decisions until current memory state
                    # can be persisted again.
                    if self._state_write_failed:
                        self._save()
                        if self._state_write_failed:
                            time.sleep(sleep_sec)
                            continue
                    self.reconcile(price)
                    self.step(price)
                    self._save()
                except Exception as e:  # noqa: BLE001 — RESILIENCE: network/API down -> retry
                    log(f"  ! [STRAT] error ({e.__class__.__name__}: {e}) — retrying")
                    time.sleep(sleep_sec)
                    continue
                avg = self._avg()
                pos = f"qty={self.s['qty']:.8f} avg={avg:.{self.price_dec}f}" if avg else "qty=0 (waiting for an entry)"
                log(f"  [STRAT] price={price}  {pos}  "
                    f"NET={self.s['realized_net']:+.2f} (gross {self.s['realized_gross']:+.2f}, "
                    f"fee {self.s['fees_total']:.2f}) {self.ccy}  ord={len(self.s['orders'])}")
                time.sleep(sleep_sec)
        except KeyboardInterrupt:
            log("  [STRAT] stopped manually.")
            self._save()
