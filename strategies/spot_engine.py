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
from dataclasses import dataclass, fields as _dc_fields
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
from providers.strategy_executor import ProviderError, StrategyExecutor
from order_retry import StrategyExecutorLifecycleApi, TrackedOrderLifecycle

from intelligence.internal.state.volatility import (
    adaptive_thresholds, resolve_cached_volatility_1h,
    resolve_cached_volatility_1h_pct, vol_pct_to_ratio,
)

from . import spot_rules as sr
from .base_spot_mixin import BaseSpotMixin, load_common_mhdpa_env, sanitize_common_spot_state
from .state_store import (
    JsonStateStore, build_dataclass_from_dict, build_strategy_exit_manager,
    clean_nonneg_float as _clean_nonneg_float, clean_pos_float as _clean_pos_float,
    compute_adaptive_reentry_pullback_pct, find_open_order,
    state_path_for as _shared_state_path_for, update_post_sell_anchors,
)
from .position_exit_manager import (
    ExitAction, PositionExitConfig, PositionExitManager, PositionSnapshot, ReentrySnapshot,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LEGACY_STATE_DIR = os.path.join(_ROOT, "kraken")
_DEFAULT_FEE_NOTE = "fee Kraken ~0.26% taker / ~0.16% maker per leg"
_PLACEMENT_BACKOFF_CAP_SEC = 3600.0
_REGIME_BAR_SETTLE_MAX_SEC = 900.0

notify = bind_notify(("SYMBOL_LABEL", "KRAKEN_PAIR"), "CRYPTO")


def state_path_for(pair: str, state_dir: str | None = None) -> str:
    """Return the state path, retaining the Kraken fallback for live compatibility."""
    return _shared_state_path_for(pair, state_dir, default_dir=_LEGACY_STATE_DIR)


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
    stop_loss_pct: float = 0.0
    adopt_cost: float = 0.0
    adopt_qty: float = 0.0
    reentry_drop_pct: float = 2.0
    reentry_tolerance_pct: float = 0.05
    reentry_adaptive: bool = False
    shadow_k_reentry: float = 2.0
    reentry_sl_bounce_pct: float = 1.5
    tp_tranches: list = None
    # --- TP trend-aware ---
    tp_trend_hold: bool = False
    tp_regime_gate: bool = False
    tp_trend_min_pct: float = 0.5
    tp_trail_pct: float = 2.0
    tp_trail_profit_floor_pct: float = 0.0
    # --- Trend Overlay ---
    trend_overlay: bool = False
    trend_sma_n: int = 30
    trend_interval: int = 240
    regime_min_samples: int = 20
    trend_topup: float = 2000.0
    trend_trail_pct: float = 5.0
    trend_exit_break: bool = False
    # --- Volatility Adaptive ---
    tp_trail_adaptive: bool = False
    tp_trail_k: float = 2.0
    tp_trail_min: float = 1.5
    tp_trail_max: float = 8.0
    tp_trail_vol_interval: int = 240
    dca_trend_brake: bool = False
    dca_brake_min_pct: float = 1.5
    dca_spacing_growth_pct: float = 0.0
    # --- Volatility-Scaled DCA Sizing ---
    dca_vol_scale_k: float = 0.0
    dca_vol_ref: float = 2.0
    dca_vol_interval: int = 240
    # --- Percentage Sizing ---
    total_budget: float = 0.0
    alloc_pct: float = 0.0
    entry_pct: float = 0.0
    dca_pct: float = 0.0
    # --- Hybrid Re-Entry ---
    reentry_hybrid_enabled: bool = False
    reentry_peak_relative: bool = False
    reentry_pullback_pct: float = 1.5
    reentry_pullback_adaptive: bool = False
    reentry_pullback_k: float = 1.0
    reentry_pullback_min: float = 0.8
    reentry_pullback_max: float = 4.0
    reentry_ttl_hours: float = 0.0
    reentry_bear_bounce_pct: float = 0.0
    # --- Multi-Horizon Dynamic Profit Architecture (MHDPA) ---
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
    # --- Multi-Horizon Dynamic Stop-Loss (MHDSL) ---
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
    # --- Multi-Horizon Dynamic Take-Profit (MDHTP) ---
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
            "enable_takeprofit": True, "reentry_adaptive": False, "tp_trend_hold": False,
            "tp_regime_gate": False, "trend_overlay": False, "trend_exit_break": False,
            "tp_trail_adaptive": False, "dca_trend_brake": False, "reentry_hybrid_enabled": False,
            "reentry_peak_relative": False, "reentry_pullback_adaptive": False, "tp_dynamic_flat": False,
            "trend_trail_dynamic": False, "fast_profit_guard": False, "surge_guard": False,
            "surge_dynamic": False, "slow_grind_guard": False, "sl_mhdsl_enabled": True,
            "sl_bounce_ratchet_enabled": True, "sl_structural_break_enabled": True, "tp_mdhtp_enabled": False,
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
        for attr, env_key, strictly_pos in (
            ("sl_disaster_buffer_ratio", "STRAT_SL_DISASTER_BUFFER_RATIO", False),
            ("sl_disaster_vol_multiplier", "STRAT_SL_DISASTER_VOL_MULTIPLIER", False),
            ("sl_disaster_buffer_min_pct", "STRAT_SL_DISASTER_BUFFER_MIN_PCT", False),
            ("sl_disaster_buffer_max_pct", "STRAT_SL_DISASTER_BUFFER_MAX_PCT", True),
            ("sl_trend_defer_max_sec", "STRAT_SL_TREND_DEFER_MAX_SEC", False),
            ("sl_bounce_min_recovery_pct", "STRAT_SL_BOUNCE_MIN_RECOVERY_PCT", False),
            ("sl_bounce_pullback_pct", "STRAT_SL_BOUNCE_PULLBACK_PCT", False),
            ("sl_bounce_ratchet_k", "STRAT_SL_BOUNCE_RATCHET_K", False),
            ("sl_bounce_min_pullback_pct", "STRAT_SL_BOUNCE_MIN_PULLBACK_PCT", False),
        ):
            val = finite(env_key, getattr(self, attr))
            setattr(self, attr, val)
            if (val <= 0) if strictly_pos else (val < 0):
                raise ValueError(f"{env_key} must be {'> 0' if strictly_pos else '>= 0'}")
        self.trend_interval = integer("STRAT_TREND_INTERVAL", self.trend_interval)
        if self.trend_interval <= 0:
            raise ValueError("STRAT_TREND_INTERVAL must be > 0")
        self.regime_min_samples = integer("STRAT_REGIME_MIN_SAMPLES", self.regime_min_samples)
        regime_capacity = MarketRegimeService.horizon_sample_capacity("long", self.trend_interval)
        if not 3 <= self.regime_min_samples <= regime_capacity:
            raise ValueError(f"STRAT_REGIME_MIN_SAMPLES must be in [3, {regime_capacity}]")

        percentage_names = ("STRAT_TOTAL_BUDGET", "STRAT_ALLOC_PCT", "STRAT_ENTRY_PCT", "STRAT_DCA_PCT")
        values = tuple(
            finite(name, value)
            for name, value in zip(percentage_names, (self.total_budget, self.alloc_pct, self.entry_pct, self.dca_pct))
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

    _DEFAULT_DICT_FIELDS = {
        "currency": "USDC", "entry_amount": 50.0, "entry_discount_pct": 0.2,
        "dca_amount": 30.0, "dca_drop_pct": 2.0, "check_minutes": 60.0,
        "takeprofit_pct": 5.0, "max_budget": 500.0, "max_dca_buys": 10,
        "enable_takeprofit": True, "order_ttl_min": 120.0,
    }

    @classmethod
    def from_dict(cls, values: dict, *, defaults: dict | None = None, warn_dropped: bool = False) -> "StratParams":
        """Build StratParams from a dict (e.g. a stored baseline report), ignoring removed fields."""
        return build_dataclass_from_dict(cls, values, defaults=defaults, warn_dropped=warn_dropped)

    @classmethod
    def from_env(cls, env: dict | None = None) -> "StratParams":
        """Load an explicit profile; missing financial policy is a startup error."""
        mode = required_env("STRATEGY_MODE", env).lower()
        if mode not in {"avg_tp", "dca_only"}:
            raise ValueError(f"Invalid STRATEGY_MODE: {mode!r}")
        return cls(
            currency=required_env("STRAT_CURRENCY", env).upper(),
            entry_amount=required_float_env("STRAT_ENTRY", env),
            entry_discount_pct=required_float_env("STRAT_ENTRY_DISCOUNT_PCT", env),
            dca_amount=required_float_env("STRAT_DCA", env),
            dca_drop_pct=required_float_env("STRAT_DCA_DROP_PCT", env),
            check_minutes=required_float_env("STRAT_CHECK_MINUTES", env),
            takeprofit_pct=required_float_env("STRAT_TAKEPROFIT_PCT", env),
            max_budget=required_float_env("STRAT_MAX_BUDGET", env),
            max_dca_buys=required_int_env("STRAT_MAX_DCA_BUYS", env),
            total_budget=required_float_env("STRAT_TOTAL_BUDGET", env),
            alloc_pct=required_float_env("STRAT_ALLOC_PCT", env),
            entry_pct=required_float_env("STRAT_ENTRY_PCT", env),
            dca_pct=required_float_env("STRAT_DCA_PCT", env),
            enable_takeprofit=(mode != "dca_only"),
            order_ttl_min=required_float_env("STRAT_ORDER_TTL_MIN", env),
            stop_loss_pct=required_float_env("STRAT_STOP_LOSS_PCT", env),
            adopt_cost=required_float_env("STRAT_ADOPT_COST", env),
            adopt_qty=required_float_env("STRAT_ADOPT_QTY", env),
            reentry_drop_pct=required_float_env("STRAT_REENTRY_DROP_PCT", env),
            reentry_tolerance_pct=required_float_env("STRAT_REENTRY_TOLERANCE_PCT", env),
            reentry_adaptive=required_bool_env("STRAT_REENTRY_ADAPTIVE", env),
            shadow_k_reentry=required_float_env("SHADOW_K_REENTRY", env),
            reentry_sl_bounce_pct=required_float_env("STRAT_REENTRY_SL_BOUNCE_PCT", env),
            tp_tranches=_parse_tranches(defined_env("STRAT_TP_TRANCHES", env)),
            tp_trend_hold=required_bool_env("STRAT_TP_TREND_HOLD", env),
            tp_regime_gate=False,
            tp_trend_min_pct=required_float_env("STRAT_TP_TREND_MIN_PCT", env),
            tp_trail_pct=required_float_env("STRAT_TP_TRAIL_PCT", env),
            tp_trail_profit_floor_pct=max(0.0, required_float_env("STRAT_TP_TRAIL_PROFIT_FLOOR_PCT", env)),
            trend_overlay=required_bool_env("STRAT_TREND_OVERLAY", env),
            trend_sma_n=required_int_env("STRAT_TREND_SMA_N", env),
            trend_interval=required_int_env("STRAT_TREND_INTERVAL", env),
            regime_min_samples=required_int_env("STRAT_REGIME_MIN_SAMPLES", env),
            trend_topup=required_float_env("STRAT_TREND_TOPUP", env),
            trend_trail_pct=required_float_env("STRAT_TREND_TRAIL_PCT", env),
            trend_exit_break=required_bool_env("STRAT_TREND_EXIT_BREAK", env),
            tp_trail_adaptive=required_bool_env("STRAT_TP_TRAIL_ADAPTIVE", env),
            tp_trail_k=required_float_env("STRAT_TP_TRAIL_K", env),
            tp_trail_min=required_float_env("STRAT_TP_TRAIL_MIN", env),
            tp_trail_max=required_float_env("STRAT_TP_TRAIL_MAX", env),
            tp_trail_vol_interval=required_int_env("STRAT_TP_TRAIL_VOL_INTERVAL", env),
            dca_trend_brake=required_bool_env("STRAT_DCA_TREND_BRAKE", env),
            dca_brake_min_pct=required_float_env("STRAT_DCA_BRAKE_MIN_PCT", env),
            dca_spacing_growth_pct=max(0.0, required_float_env("STRAT_DCA_SPACING_GROWTH_PCT", env)),
            dca_vol_scale_k=required_float_env("STRAT_DCA_VOL_SCALE_K", env),
            dca_vol_ref=required_float_env("STRAT_DCA_VOL_REF", env),
            dca_vol_interval=required_int_env("STRAT_DCA_VOL_INTERVAL", env),
            **load_common_mhdpa_env(env, include_spot_sl=True),
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
        "cycle": 1, "qty": 0.0, "cost": 0.0, "spent": 0.0, "dca_buys": 0,
        "entry_price": None, "last_buy_price": None,
        "cycle_fees": 0.0, "realized_gross": 0.0, "realized_net": 0.0, "fees_total": 0.0,
        "last_sell_price": None, "last_exit_kind": None, "sl_low": None,
        "post_sell_peak": None, "post_sell_trough": None, "last_sell_ts": None,
        "trail_peak": None, "trail_stop": None,
        "trend_mode": False, "trend_peak": None, "trend_confirm_count": 0,
        "orders": [], "pending_intent": None, "pending_exit": None,
        "placement_backoffs": {}, "ledger_reconcile_required": False,
        "surge_active": False, "surge_peak": None,
        "slow_grind_active": False, "slow_grind_peak": None, "entry_ts": None,
        "sl_bounce_state": {}, "sl_episode_state": {}, "reentry_state": {}, "tp_ride_state": {},
    }


class Strategy(BaseSpotMixin):
    _exit_manager_mode = "spot"
    _cost_state_key = "cost"
    _spent_state_key = "spent"
    _qty_eps = 1e-12
    _vol_default_source = "default"

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
        self.replay_mode = _parse_bool_flag(replay_mode, False)
        self.desktop = _parse_bool_flag(desktop, False)
        self._notifier = notifier
        self.notification_source = notification_source
        self.venue_label = venue_label
        self.fee_note = fee_note
        self._regime_service = regime_service or MarketRegimeService()
        self._regime_bar_context = None
        self.state_file = state_path_for(pair) if state_dir is None else state_path_for(pair, state_dir)
        self._state_write_failed = False
        self._state_file_mtime: float = 0.0
        raw_state = initial_state if isinstance(initial_state, dict) else self._load()
        self.s = self._sanitize_spot_state(raw_state)
        self.exit_manager: PositionExitManager | None = (
            self._init_exit_manager() if (self.p.sl_mhdsl_enabled or self.p.tp_mdhtp_enabled) else None
        )
        self._paper_seq = 0
        self._shadow_prices = deque(maxlen=90)
        self._surge_replay_closes = (
            deque(maxlen=self._surge_window_bars() + 1)
            if self.replay_mode and params.surge_guard and params.surge_window_hours > 0 and params.surge_move_pct > 0
            else None
        )
        self._max_active_surge_vol = 0.0
        self._surge_win_check_ts: list[float] = []
        try:
            precision = client.pair_precision(pair)
        except ProviderError as exc:
            raise RuntimeError(f"Cannot start {pair}: venue precision is unavailable") from exc
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

    def _emit(self, **event) -> None:
        """Send an event through the venue sink or legacy compatibility wrapper."""
        if not self.replay_mode:
            (self._notifier or notify)(**event)

    # -- Persistence -----------------------------------------------------------
    @staticmethod
    def _sanitize_spot_state(state: dict | None) -> dict:
        state = sanitize_common_spot_state(
            state,
            _new_state,
            nonneg_float_keys=("qty", "cost", "spent", "cycle_fees", "fees_total"),
            signed_float_keys=("realized_gross", "realized_net"),
            nonneg_int_keys=("dca_buys", "trend_confirm_count"),
            pos_float_keys=(
                "entry_price", "last_buy_price", "last_sell_price", "sl_low",
                "post_sell_peak", "post_sell_trough", "last_sell_ts", "trail_peak",
                "trail_stop", "trend_peak", "surge_peak", "slow_grind_peak", "entry_ts",
            ),
            bool_keys=("trend_mode", "ledger_reconcile_required", "surge_active", "slow_grind_active", "adopted"),
            dict_keys=("placement_backoffs", "sl_bounce_state", "sl_episode_state", "reentry_state", "tp_ride_state"),
            order_id_key="txid",
            lowercase_order_side=True,
            order_pos_float_keys=("vol", "price"),
            order_nonneg_float_keys=("amount", "ts"),
            order_bool_keys=("market", "cancel_requested"),
        )
        pe = state.get("pending_exit")
        if isinstance(pe, dict) and "soft_floor" in pe:
            pe["soft_floor"] = _parse_bool_flag(pe.get("soft_floor"), False)
        return state

    _sanitize_state_dict = _sanitize_spot_state

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
        market_val = pending.get("market")
        if market_val is None and isinstance(meta, dict):
            market_val = meta.get("market")
        return {
            "txid": _clean_order_id(pending.get("order_id")) or "",
            "side": str(pending["side"]).lower(),
            "vol": eff_vol,
            "price": safe_pos_float(pending.get("requested_price"), 0.0),
            "amount": amount,
            "kind": pending.get("kind"),
            "market": _parse_bool_flag(market_val, False),
            "intent_id": _clean_order_id(pending.get("intent_id")),
            "ts": safe_pos_float(pending.get("created_at"), self._now()),
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
                    str(pending.get("side") or "buy"), str(pending.get("kind") or "ENTRY"), submit_error
                )
            self._persist_pending_intent(None)
            log(
                f"  ! [STRAT] released stale insufficient-funds pending intent "
                f"({pending.get('side')} {pending.get('kind')})"
            )
            return
        try:
            result = self._order_lifecycle.reconcile(pending, persist=self._persist_pending_intent)
        except (ProviderError, RuntimeError, ValueError) as exc:
            log(f"  ! [STRAT] unreconciled pending intent ({exc}) — not resending")
            return
        if result.order_known:
            self._adopt_pending_order(result.intent)
            log(f"  [STRAT] recovered intent {result.intent['side']} txid={result.intent['order_id']}")
        elif result.outcome in {"absent", "retryable"}:
            log("  [STRAT] intent absence confirmed — the strategy may re-evaluate it")

    # -- Helpers ---------------------------------------------------------------
    def _qty_for(self, amount: float, price: float) -> float:
        amt_f = safe_pos_float(amount, default=None)
        px_f = safe_pos_float(price, default=None)
        return round(amt_f / px_f, self.vol_dec) if (amt_f is not None and px_f is not None) else 0.0

    def _dust_safe_qty(self, qty: float) -> float:
        """Leave one volume tick to tolerate a venue's rounded balance."""
        qty_f = safe_pos_float(qty, default=None)
        if qty_f is None:
            return 0.0
        dust_decimals = self.vol_dec if self.vol_dec <= 2 else self.vol_dec - 1
        step = 10.0 ** -max(dust_decimals, 1)
        ticks = math.floor((qty_f + step * 1e-9) / step) - 1
        return round(max(ticks, 0) * step, self.vol_dec)

    @staticmethod
    def _insufficient_funds_error(error) -> bool:
        text = str(error or "").lower()
        return any(m in text for m in ("insufficient funds", "insufficient balance", "insufficient spot balance"))

    @staticmethod
    def _placement_backoff_key(side: str, kind: str) -> str:
        return f"{str(side).lower()}:{str(kind or 'UNKNOWN').upper()}"

    def _placement_backoff_remaining(self, side: str, kind: str) -> float:
        records = self.s.get("placement_backoffs")
        record = records.get(self._placement_backoff_key(side, kind)) if isinstance(records, dict) else None
        if not isinstance(record, dict):
            return 0.0
        until_f = safe_float(record.get("until"), default=None)
        now_f = safe_float(self._now(), default=None)
        if until_f is None or now_f is None:
            return 0.0
        return min(_PLACEMENT_BACKOFF_CAP_SEC, max(0.0, until_f - now_f))

    def _record_placement_backoff(self, side: str, kind: str, error) -> None:
        key = self._placement_backoff_key(side, kind)
        records = self.s.get("placement_backoffs")
        if not isinstance(records, dict):
            records = {}
            self.s["placement_backoffs"] = records
        previous = records.get(key) if isinstance(records.get(key), dict) else {}
        attempts = safe_nonneg_int(previous.get("attempts"), default=0) + 1
        base = max(60.0, float(self.p.check_minutes) * 60.0)
        delay = min(_PLACEMENT_BACKOFF_CAP_SEC, base * (2 ** min(attempts - 1, 10)))
        records[key] = {"attempts": attempts, "until": self._now() + delay, "reason": str(error)}
        self._save()
        log(f"  ! [STRAT] {side} {kind} insufficient funds — operational cooldown {delay:.0f}s (refusal {attempts})")

    def _clear_placement_backoff(self, side: str, kind: str) -> None:
        records = self.s.get("placement_backoffs")
        if not isinstance(records, dict):
            self.s["placement_backoffs"] = {}
            return
        records.pop(self._placement_backoff_key(side, kind), None)

    # -- Placement & Cancellation ----------------------------------------------
    def _place(self, side: str, vol: float, price: float, kind: str, amount: float = 0.0,
               market: bool = False) -> bool:
        market = _parse_bool_flag(market, False)
        s_norm = normalize_side(side)
        if s_norm in ("BUY", "SELL"):
            side = s_norm.lower()
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
        if not market and str(side).lower() == "sell" and str(kind).upper() != "STOP":
            _avg = self._avg()
            if _avg is not None and price < _avg:
                log(f"  ! [STRAT] [GUARD] {kind} sell rejected: price {price} < avg "
                    f"{_avg:.6f} (profit-floor; STOP/MARKET excepted)")
                return False
        if not market:
            remaining = self._placement_backoff_remaining(side, kind)
            if remaining > 0:
                log(f"  ! [STRAT] {side} {kind} in cooldown after insufficient funds ({remaining:.0f}s remaining)")
                return False
        if self.dry_run:
            self._paper_seq += 1
            log(f"  [STRAT] [PAPER] {side.upper()} {kind}{' MKT' if market else ''} {vol} @ {price} {self.ccy}")
            self.s["orders"].append({
                "txid": f"PAPER-{self._paper_seq}", "side": side, "vol": vol, "price": price,
                "amount": amount, "kind": kind, "market": market, "ts": self._now(),
            })
            return True
        if self.s.get("pending_intent"):
            log("  ! [STRAT] an unreconciled pre-submit intent exists — not duplicating")
            return False
        try:
            preflight = getattr(type(self.client), "preflight_order", None)
            if callable(preflight):
                preflight(self.client, self.pair, side, vol, None if market else price, market=market, kind=kind)
            intent_id = self._intent_identity(side, kind, vol, price)
            client_order_id = intent_client_order_id(self.venue_label, intent_id)
            if not client_order_id:
                raise ProviderError(f"{self.venue_label}: client_order_id unavailable for the lifecycle")
            intent = self._order_lifecycle.new_intent(
                intent_id=intent_id, client_order_id=client_order_id, symbol=self.pair,
                side=side, requested_qty=vol, requested_price=price, kind=kind,
                metadata={"amount": amount, "market": market},
            )
            submit_outcome = getattr(type(self.client), "submit_outcome_with_intent", None)
            submit_with_intent = getattr(type(self.client), "submit_order_with_intent", None)

            def submit():
                for fn in (submit_outcome, submit_with_intent):
                    if callable(fn):
                        return fn(
                            self.client, intent_id, self.pair, side, vol,
                            None if market else price, market=market, kind=kind,
                            reference_price=price if market else None, client_order_id=client_order_id,
                        )
                return self.client.submit_order(
                    self.pair, side, vol, None if market else price,
                    market=market, kind=kind, client_order_id=client_order_id,
                )

            tracked = self._order_lifecycle.submit(intent, persist=self._persist_pending_intent, submit=submit)
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
                        self._record_placement_backoff(side, kind, submit_error)
                    self._persist_pending_intent(None)
                    log(
                        f"  ! [STRAT] {side} {kind} rejected due to insufficient funds; "
                        "the intent was released for safe reevaluation"
                    )
                    return False
                log(f"  ! [STRAT] {side} {kind} ambiguous submit; the intent stays persisted for reconciliation")
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
        """Request cancellation without forgetting the order before terminal confirmation."""
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
                    self.client, o.get("intent_id") or f"legacy-{self.pair}-{o['txid']}", self.pair, o["txid"]
                )
            else:
                self.client.cancel_order(self.pair, o["txid"])
        except ProviderError as e:
            log(f"  ! [STRAT] cancel failed for {o['txid']}: {e} — the order stays tracked")
            return False
        return self._mark_order_cancel_requested(o, "txid")

    def _cancel_orders(self, side: str | None = None, *, exclude_market: bool = False) -> bool:
        """Request cancellation of selected orders; return False if any request fails."""
        orders = self.s["orders"]
        if not orders:
            return True
        selected = [
            o for o in list(orders)
            if (side is None or o["side"] == side) and not (exclude_market and o.get("market"))
        ]
        ok = True
        for o in selected:
            ok = self._cancel_order(o) and ok
        return ok

    def _cancel_if_stale_buy(self, o: dict, price: float) -> None:
        age = (self._now() - (_clean_nonneg_float(o.get("ts")) or 0.0)) / 60
        if (
            o.get("side") == "buy"
            and not o.get("cancel_requested")
            and age > self.p.order_ttl_min
            and price > o["price"] * 1.003
        ):
            log(f"  [STRAT] buy {o['txid']} unfilled, the price rose — cancelling and re-placing")
            self._cancel_order(o)

    def _has_pending_market_exit(self) -> bool:
        return any(o["side"] == "sell" and o.get("market") for o in self.s["orders"])

    def _request_market_exit(self, price: float, kind: str, *, soft_floor: bool = False,
                             min_profit_pct: float | None = None) -> bool:
        """Persist exit ownership before cancellation, then sell reconciled holdings."""
        pending = self.s.get("pending_exit")
        soft_floor = _parse_bool_flag(soft_floor, False)
        if not isinstance(pending, dict) or (kind == "STOP" and pending.get("kind") != "STOP"):
            self.s["pending_exit"] = {"kind": kind, "soft_floor": soft_floor}
            if min_profit_pct is not None and kind != "STOP":
                self.s["pending_exit"]["min_profit_pct"] = min_profit_pct
        return self._continue_market_exit(price)

    def _continue_market_exit(self, price: float) -> bool:
        pending = self.s.get("pending_exit")
        if not isinstance(pending, dict):
            self.s["pending_exit"] = None
            return False
        if self.s.get("pending_intent") or self._has_pending_market_exit():
            return False
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
        placed = self._place("sell", self._dust_safe_qty(self.s["qty"]), reference, kind=kind, market=True)
        if placed and kind == "STOP":
            self._emit(
                title=f"🛑 SL {self.pair}",
                body="Protective MARKET exit accepted; waiting for confirmed execution.",
                source=self.notification_source, price=price, desktop=self.desktop,
            )
        return placed

    # -- Reconciliation --------------------------------------------------------
    def reconcile(self, price: float) -> None:
        self._reconcile_pending_submit()
        if not self.s.get("pending_exit"):
            exit_order = next((o for o in self.s["orders"] if o["side"] == "sell" and o.get("market")), None)
            if exit_order:
                self.s["pending_exit"] = {
                    "kind": exit_order.get("kind") or "TP",
                    "soft_floor": (self.s.get("trail_peak") is not None and not self.s.get("trend_mode")),
                }
                self._save()
        for side in ("buy", "sell"):
            for o in [x for x in self.s["orders"] if x["side"] == side]:
                if o not in self.s["orders"]:
                    continue
                if self.dry_run:
                    if o.get("market"):
                        fill_price = price
                    elif (side == "buy" and price <= o["price"]) or (side == "sell" and price >= o["price"]):
                        fill_price = o["price"]
                    else:
                        self._cancel_if_stale_buy(o, price)
                        continue
                    if o in self.s["orders"]:
                        self._remove(o)
                        self._apply_fill(o, o["vol"], fill_price, fee=0.0)
                    continue
                try:
                    status_with_intent = getattr(type(self.client), "order_status_with_intent", None)
                    if callable(status_with_intent):
                        status = status_with_intent(
                            self.client, o.get("intent_id") or f"legacy-{self.pair}-{o['txid']}", self.pair, o["txid"]
                        )
                    else:
                        status = self.client.order_status(self.pair, o["txid"])
                except ProviderError as e:
                    log(f"  ! [STRAT] status {o['txid']} failed: {e} — keeping the order")
                    continue
                st = status.status
                terminal = st in ("closed", "canceled", "expired")
                total_vol = 0.0 if status.filled_qty is None else safe_nonneg_float(status.filled_qty, default=None)
                total_fee = 0.0 if status.fee is None else safe_float(status.fee, default=None)
                ord_px = safe_pos_float(o.get("price"), default=None)
                if total_vol is None or total_fee is None or ord_px is None:
                    log(f"  ! [STRAT] status {o['txid']} has an invalid execution — keeping the order")
                    continue
                if status.cost is None:
                    total_cost = total_vol * ord_px
                else:
                    raw_cost = safe_nonneg_float(status.cost, default=None)
                    total_cost = (total_vol * ord_px) if raw_cost == 0.0 and total_vol > 0 else raw_cost
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
                ord_vol = safe_pos_float(o.get("vol"), default=0.0) or 0.0
                eps = max(1e-12, ord_vol * 1e-12)
                if total_vol + eps < applied_vol or total_cost + eps < applied_cost:
                    log(f"  ! [STRAT] status {o['txid']} regressed cumulatively "
                        f"(vol {total_vol}<{applied_vol}) — not reapplying")
                    continue

                delta_vol = max(0.0, total_vol - applied_vol)
                delta_cost = max(0.0, total_cost - applied_cost)
                delta_fee = total_fee - applied_fee
                o["applied_vol"] = total_vol
                o["applied_cost"] = total_cost
                o["applied_fee"] = total_fee

                if delta_vol > eps:
                    fill_order = dict(o)
                    if o["side"] == "buy":
                        ord_amt = safe_nonneg_float(o.get("amount"), default=0.0) or 0.0
                        fill_order["amount"] = ord_amt * delta_vol / ord_vol if ord_vol > 0 else delta_cost
                        if applied_vol > eps and fill_order.get("kind") in {"DCA", "TREND_ENTRY"}:
                            fill_order["kind"] = f"{fill_order['kind']}_PARTIAL"
                    fill_price = delta_cost / delta_vol if delta_cost > 0 else reported_price
                    if terminal:
                        self._remove(o)
                    self._apply_fill(fill_order, delta_vol, fill_price, fee=delta_fee, final=False)
                elif abs(delta_fee) > eps:
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
                    self._cancel_if_stale_buy(o, price)

    def _apply_fill(self, o: dict, vol: float, price: float, fee: float,
                    *, final: bool = True) -> None:
        tag = "[PAPER] " if self.dry_run else ""
        self.s["cycle_fees"] += fee
        self.s["fees_total"] += fee
        self.s["realized_net"] -= fee
        if o["side"] == "buy":
            self._apply_buy_fill_accounting(o, vol, price, fallback_to_notional=True)
            if self.s.get("entry_ts") is None:
                self.s["entry_ts"] = self._shadow_prices[-1][0] if self._shadow_prices else self._now()
            if o.get("kind") == "TREND_ENTRY":
                self.s["trend_mode"] = True
                self.s["trend_peak"] = price
            avg = self._avg()
            log(f"  [STRAT] {tag}BUY FILLED {vol} @ {price} {self.ccy} ({o.get('kind')})  "
                f"qty={self.s['qty']:.8f} avg={avg:.{self.price_dec}f} fee={fee}")
            self._emit(
                title=f"{tag}{self.pair} BUY {vol:.2f}@{price:.2f}",
                body=f"{o.get('kind')} | q{self.s['qty']:.2f} a{avg:.2f} | desf{self.s['spent']:.0f}{self.ccy}",
                source=self.notification_source, price=price, desktop=self.desktop,
            )
            self._cancel_orders("sell", exclude_market=True)
        else:
            avg = self._avg() or price
            gross = (price - avg) * vol
            net = gross - fee
            self.s["realized_gross"] += gross
            self.s["realized_net"] += gross
            self._release_sell_cost_basis(vol, avg)
            log(f"  [STRAT] {tag}SELL FILLED {vol} @ {price} {self.ccy}  "
                f"brut={gross:+.4f} fee_ciclu={self.s['cycle_fees']:.4f} net={net:+.4f}")
            self._emit(
                title=f"{tag}{self.pair} SELL {vol:.2f}@{price:.2f} N{net:+.2f}{self.ccy}",
                body=(f"a{avg:.2f} · br{gross:+.2f} fee{self.s['cycle_fees']:.2f} N{net:+.2f} | "
                      f"Ntot{self.s['realized_net']:+.2f}{self.ccy}"),
                source=self.notification_source, price=price, desktop=self.desktop,
            )
            if final:
                self._finalize_cycle_if_flat(o, price)

    def _finalize_cycle_if_flat(self, o: dict, price: float) -> None:
        """Close a cycle once, only after terminal exit status."""
        dust = 2 * 10.0 ** -(max(self.vol_dec - 1, 1))
        if abs(self.s["qty"]) < dust:
            self.s["qty"] = 0.0
            self.s["cost"] = 0.0
        if self.s["qty"] > 1e-12:
            return
        if self.s["orders"] or self.s.get("pending_intent"):
            pending = self.s.get("pending_exit") or {"kind": o.get("kind") or "TP"}
            pending["fill_price"] = price
            self.s["pending_exit"] = pending
            self._save()
            return
        keep = (
            self.s["realized_gross"], self.s["realized_net"],
            self.s["fees_total"], self.s.get("cycle", 1) + 1,
        )
        self.s = _new_state()
        self.s["realized_gross"], self.s["realized_net"], self.s["fees_total"], self.s["cycle"] = keep
        exit_ts = self._shadow_prices[-1][0] if self._shadow_prices else self._now()
        self._seed_post_exit_anchors(price, o.get("kind"), exit_ts)
        log(f"  [STRAT] === cycle closed; restarting (cycle {self.s['cycle']}) ===")

    # -- Decision logic --------------------------------------------------------
    def _is_trend_direction(
        self, regime: MarketRegimeDecision | None, expected_regime: str, client_tokens: tuple[str, ...]
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
        return bool(self.s.get("trend_mode")) or self._is_trend_direction(regime, "bull", ("up", "bull", "uptrend"))

    def _is_trend_down(self, regime: MarketRegimeDecision | None = None) -> bool:
        """Return True if a downtrend is identified via regime or venue client."""
        return self._is_trend_direction(regime, "bear", ("down", "bear", "downtrend"))

    def _check_stop_loss(self, price: float) -> bool:
        """Close everything when unrealized loss crosses the anti-runaway threshold."""
        if (self.s.get("pending_exit") or {}).get("kind") == "STOP":
            if not self._has_pending_market_exit():
                self._continue_market_exit(price)
            return True
        if self.p.stop_loss_pct <= 0:
            return False
        avg = self._avg()
        if not avg:
            return False
        loss_pct = (avg - price) / avg * 100

        if not self.p.sl_mhdsl_enabled:
            if sr.hit_stop(avg, price, self.p.stop_loss_pct):
                if self._has_pending_market_exit():
                    return True
                log(f"  🛑 [STRAT] STOP-LOSS: loss {loss_pct:.2f}% >= {self.p.stop_loss_pct}% — SELLING EVERYTHING (cutting the loss)")
                if not self._request_market_exit(price, "STOP"):
                    log("  [STRAT] STOP triggered; exit remains pending reconciliation or submission")
                return True
            return False
        if not self._in_mhdsl_territory(loss_pct):
            return False

        if self.exit_manager is None:
            self.exit_manager = self._init_exit_manager()
        snapshot = self._create_position_snapshot(
            price, avg, self.s.get("qty", 0.0), include_shadow=not self.replay_mode
        )
        decision = self.exit_manager.evaluate_stop_loss(snapshot, custom_threshold=self.p.stop_loss_pct / 100.0)
        if decision.action == ExitAction.SELL and decision.should_exit:
            if self._has_pending_market_exit():
                return True
            log(
                f"  🛑 [STRAT] MHDSL STOP-LOSS: loss {loss_pct:.2f}% >= {self.p.stop_loss_pct}% "
                f"({decision.reason}) — SELLING EVERYTHING"
            )
            if not self._request_market_exit(price, "STOP"):
                log("  [STRAT] STOP triggered; exit remains pending reconciliation or submission")
            return True
        if decision.action == ExitAction.HOLD:
            self._handle_mhdsl_hold_decision(decision)
        return False

    def _trail_profit_floor_price(self, avg: float) -> float | None:
        """Return the minimum soft-exit price, rounded up to venue precision."""
        pct = float(self.p.tp_trail_profit_floor_pct or 0.0)
        if pct <= 0 or avg <= 0:
            return None
        scale = 10 ** self.price_dec
        return math.ceil(sr.tp_price(avg, pct) * scale - 1e-12) / scale

    def _maybe_adopt(self) -> None:
        """Adopt an existing account position instead of buying an entry."""
        if self.p.adopt_cost <= 0 or self.s.get("adopted"):
            return
        if self.s["qty"] > 1e-12 or self.s["orders"] or self.s["cycle"] != 1 or self.s["spent"] > 0:
            log("  [STRAT] adopt: the state is not fresh — NOT adopting (cycle in progress)")
            return
        qty = _clean_nonneg_float(self.p.adopt_qty) or 0.0
        if qty <= 0:
            try:
                precision = self.client.pair_precision(self.pair)
                raw_bal = self.client.free_balance(precision.base_asset if precision else "")
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
        self._emit(
            title=f"📥 {self.pair} ADOPTAT {qty:.2f}@{self.p.adopt_cost}",
            body=f"TP+{self.p.takeprofit_pct}% DCA-{self.p.dca_drop_pct}% SL{self.p.stop_loss_pct}%",
            source=f"{self.notification_source}-bot", price=self.p.adopt_cost, desktop=self.desktop,
        )

    # -- Percentage sizing; all-zero settings retain fixed sizing --------------
    def _pct_sizing_on(self) -> bool:
        return self.p.pct_sizing_on()

    def _alloc_budget(self) -> float:
        return self.p.allocated_budget()

    def _effective_entry_amount(self) -> float:
        return self.p.effective_entry_amount()

    def _effective_max_budget(self) -> float:
        return self.p.effective_max_budget()

    def _base_dca_amount(self) -> float:
        return self.p.effective_dca_amount_base()

    def _effective_dca_amount(self) -> float:
        """Scale base DCA size by volatility when enabled, falling back during warm-up."""
        base = self._base_dca_amount()
        k_val = safe_float(float(self.p.dca_vol_scale_k), default=0.0)
        ref_val = safe_pos_float(float(self.p.dca_vol_ref), default=None)
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

    _REPLAY_REGIME_CACHE: dict = {}

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
        """Calculate the configured-window move, or None when unavailable."""
        if not self.p.surge_window_hours or self.p.surge_window_hours <= 0:
            return None
        if closes is None:
            closes = self._surge_replay_closes if self._surge_replay_closes is not None else self._trend_closes()
        if not closes:
            return None
        idx = -self._surge_window_bars() - (1 if self.replay_mode else 0)
        if abs(idx) > len(closes) or (past_price := closes[idx]) <= 0:
            return None
        return (price - past_price) / past_price * 100.0

    # -- Common market regime and trend overlay --------------------------------
    def _trend_closes(self) -> list:
        """Return long-trend closes at the same cadence in live and replay."""
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
            refresh_epoch = slot * interval_sec + min(_REGIME_BAR_SETTLE_MAX_SEC, interval_sec / 4)
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
                n_sp, sp[0][1], sp[1][1], sp[n_sp >> 1][1], sp[-2][1], sp[-1][1],
                self.p.trend_interval, self._regime_service.evaluator.strength_threshold,
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
                closes, horizon="long", interval_min=self.p.trend_interval
            )
            res = (decision, closes)
            self._replay_regime_key = replay_key
            self._replay_regime_val = res
            return self._cache_store(Strategy._REPLAY_REGIME_CACHE, replay_key, res)
        closes = self._trend_closes()
        decision = self._regime_service.evaluate_closes_for_horizon(
            closes, horizon="long", interval_min=self.p.trend_interval
        )
        if not self.replay_mode and decision.fresh:
            self._regime_bar_context = (slot, now >= refresh_epoch, decision, closes)
        return decision, closes

    @staticmethod
    def _regime_matches(decision: MarketRegimeDecision | None, expected: str, *,
                        min_move_pct: float = 0.0, min_samples: int = 3) -> bool:
        """Apply policy thresholds without changing the shared classification."""
        if (
            decision is None or not decision.fresh or decision.regime != expected
            or (decision.n_samples or 0) < min_samples
        ):
            return False
        move = decision.fitted_move_pct
        return min_move_pct <= 0 or (move is not None and abs(move) >= min_move_pct)

    def _check_advanced_profit_guards(
        self, price: float, avg: float, held: float,
        closes: list | None = None, tick_time: float | None = None,
    ) -> bool:
        if not avg or held <= 1e-12:
            return False
        current_gain = (price - avg) / avg * 100.0

        # 1. Micro-gradient 2X Fast Profit Guard (Horizon 3)
        if (
            self.p.fast_profit_guard and self.p.takeprofit_pct > 0 and self.p.fast_profit_mult > 0
            and current_gain >= self.p.takeprofit_pct * self.p.fast_profit_mult
        ):
            triggered, gain, drop = sr.check_fast_profit_reversal(
                current_price=price, avg_cost=avg, base_tp_pct=self.p.takeprofit_pct,
                mult=self.p.fast_profit_mult, shadow_prices=self._shadow_prices,
                window_sec=self.p.fast_profit_window_min * 60.0, drop_pct=self.p.fast_profit_drop_pct,
                current_time=tick_time,
            )
            if triggered:
                exit_px = round(price * 0.999, self.price_dec)
                if self._request_market_exit(price, "TP"):
                    log(
                        f"  [STRAT] FAST PROFIT GUARD EXIT (+{gain:.2f}% gain, -{drop:.2f}% drop in "
                        f"{self.p.fast_profit_window_min:.0f}m) -> reference {exit_px}"
                    )
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
                self.p.surge_dynamic and not in_min_surge_zone and min_g > 0
                and type(self)._effective_surge_gain_pct is Strategy._effective_surge_gain_pct
                and "_effective_surge_gain_pct" not in self.__dict__
                and type(self)._trail_vol_1h is Strategy._trail_vol_1h
                and "_trail_vol_1h" not in self.__dict__
            ):
                eff_surge_gain = min_g
            else:
                eff_surge_gain = self._effective_surge_gain_pct() if self.p.surge_dynamic else self.p.surge_gain_pct
            if (
                is_armed
                or (eff_surge_gain > 0 and peak_gain >= eff_surge_gain)
                or (self.p.surge_move_pct > 0 and current_gain >= eff_surge_gain * 0.8)
            ):
                window_move = self._surge_window_move_pct(price, None if self.replay_mode else closes)
                pos_surged = eff_surge_gain > 0 and (current_gain >= eff_surge_gain or peak_gain >= eff_surge_gain)
                if self.replay_mode and not pos_surged and self.p.surge_move_pct > 0 and current_gain >= eff_surge_gain * 0.8:
                    self._surge_win_check_ts.append(tick_time if tick_time is not None else self._now())
                if not is_armed:
                    win_surged = (
                        window_move is not None and self.p.surge_move_pct > 0
                        and window_move >= self.p.surge_move_pct and current_gain >= eff_surge_gain * 0.8
                    )
                    if pos_surged or win_surged:
                        is_armed = True
                        self.s["surge_active"] = True
                        log(f"  [STRAT] PARABOLIC SURGE ARMED (+{peak_gain:.2f}% peak gain, window move {window_move})")
                elif current_gain < self.p.takeprofit_pct:
                    self.s["surge_active"] = False
                    is_armed = False

                triggered, reason = sr.check_parabolic_surge_exhaustion(
                    current_price=price, avg_cost=avg, surge_peak=surge_peak,
                    surge_gain_pct=eff_surge_gain, exit_pullback_pct=self.p.surge_exit_pullback_pct,
                    window_move_pct=window_move, surge_move_pct=self.p.surge_move_pct, surge_active=is_armed,
                )
                if triggered:
                    exit_px = round(price * 0.999, self.price_dec)
                    floor_pct = sr.parabolic_surge_profit_floor_pct(eff_surge_gain, self.p.surge_exit_pullback_pct)
                    if self._request_market_exit(price, "TP", min_profit_pct=floor_pct):
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
                    current_price=price, avg_cost=avg, entry_ts=entry_ts,
                    current_ts=tick_time if tick_time is not None else self._now(),
                    min_days=self.p.slow_grind_days, min_gain_pct=self.p.slow_grind_min_gain_pct,
                    recent_peak=slow_peak, shadow_prices=self._shadow_prices,
                    flash_window_sec=self.p.slow_grind_flash_window_min * 60.0,
                    flash_drop_pct=self.p.slow_grind_flash_drop_pct,
                    structural_drop_pct=self.p.slow_grind_hourly_drop_pct, sma_value=sma,
                )
                if triggered:
                    exit_px = round(price * 0.999, self.price_dec)
                    if self._request_market_exit(price, "TP"):
                        log(f"  [STRAT] SLOW GRIND EXIT ({reason}) peak {slow_peak:.{self.price_dec}f} -> reference {exit_px}")
                        return True

        return False

    def _overlay_step(self, price: float, regime: MarketRegimeDecision,
                      closes: list, tick_time: float | None = None) -> bool:
        """Apply the regime overlay and return whether it handled the tick."""
        if self.s.get("trend_mode"):
            avg = self._avg()
            if self._check_advanced_profit_guards(price, avg, self.s["qty"], closes, tick_time):
                return True
            peak = max(safe_pos_float(self.s.get("trend_peak"), price), price)
            self.s["trend_peak"] = peak
            if self.p.trend_trail_dynamic and avg:
                eff_trail = sr.dynamic_trend_trail_pct(
                    peak_gain_pct=(peak - avg) / avg * 100.0,
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
                self._cancel_orders("sell", exclude_market=True)
            return True
        up = self._regime_matches(regime, "bull", min_samples=self.p.regime_min_samples)
        pending_trend_entry = next(
            (o for o in self.s["orders"] if o["side"] == "buy" and o.get("kind") == "TREND_ENTRY"), None
        )
        if pending_trend_entry:
            if up:
                return True
            self._cancel_open("buy")
            log("  [STRAT] TREND ENTER cancelled: the signal vanished before the fill")
            return False
        if up and self.s["spent"] + self.p.trend_topup <= self._effective_max_budget():
            if self.s["qty"] <= 1e-12:
                allowed, reason = self._evaluate_reentry_gate(price, regime, tick_time)
                if not allowed:
                    self._reentry_blocked_tick = (tick_time, price, reason)
                    log(f"  [STRAT] trend overlay entry blocked by re-entry: {reason}")
                    return False
            self._cancel_orders("buy")
            self._cancel_orders("sell")
            if self.s["orders"]:
                if self.s["qty"] <= 1e-12:
                    self._restore_reentry_state()
                return True
            if not self._place(
                "buy", self._qty_for(self.p.trend_topup, price), price,
                kind="TREND_ENTRY", amount=self.p.trend_topup,
            ):
                if self.s["qty"] <= 1e-12:
                    self._restore_reentry_state()
                return True
            log(f"  [STRAT] TREND ENTER pending: top-up {self.p.trend_topup} {self.ccy} @ {price} "
                f"(common {regime.horizon} regime: {regime.regime})")
            return True
        return False

    def _reconcile_venue_ledger(
        self, price: float, clean_ts: float | None, had_persisted_reconcile: bool
    ) -> bool:
        """Synchronize strategy holdings with the venue balance when flagged; return True to stop step."""
        held = self.s["qty"]
        try:
            precision = self.client.pair_precision(self.pair)
            base = precision.base_asset if precision else ""
            ledger_val = self.client.free_balance(base)
            if ledger_val is None:
                log("  ! [STRAT] ledger reconciliation unavailable (balance query returned None); decisions blocked")
                return True
            ledger_qty = _clean_nonneg_float(ledger_val)
            if ledger_qty is None:
                log("  ! [STRAT] ledger reconciliation unavailable (invalid balance); decisions blocked")
                return True
        except (ProviderError, TypeError, ValueError, OverflowError) as exc:
            log(f"  ! [STRAT] ledger reconciliation unavailable ({exc}); decisions blocked")
            return True
        tolerance = max(2 * 10.0 ** -max(self.vol_dec, 1), 1e-12)
        if self.observe_only and ledger_qty + tolerance < held:
            try:
                raw_cli = getattr(self.client, "_inner", self.client)
                if hasattr(raw_cli, "open_orders"):
                    open_rows = raw_cli.open_orders(self.pair)
                    if isinstance(open_rows, list):
                        open_sell_qty = sum(
                            _clean_pos_float(r.get("remaining_qty") or r.get("origQty") or r.get("qty")) or 0.0
                            for r in open_rows
                            if isinstance(r, dict) and str(r.get("side") or "").strip().upper() == "SELL"
                        )
                        if open_sell_qty > 0.0:
                            ledger_qty = min(held, ledger_qty + open_sell_qty)
            except Exception:
                pass
        if ledger_qty <= tolerance:
            if held > 1e-12:
                log(f"  [STRAT] LEDGER RECONCILE: position closed on venue ({base}=0); resetting cycle")
                defaults = _new_state()
                for k in (
                    "qty", "cost", "spent", "cycle_fees", "dca_buys", "entry_price",
                    "last_buy_price", "entry_ts", "trail_peak", "trail_stop", "trend_mode",
                    "trend_peak", "trend_confirm_count", "pending_exit", "surge_active",
                    "surge_peak", "slow_grind_active", "slow_grind_peak", "sl_bounce_state",
                    "sl_episode_state", "tp_ride_state", "reentry_state",
                ):
                    self.s[k] = defaults[k]
                self._seed_post_exit_anchors(price, "TP", clean_ts if clean_ts is not None else self._now())
                self.s["ledger_reconcile_required"] = False
                self._save()
            else:
                self.s["ledger_reconcile_required"] = False
                if had_persisted_reconcile:
                    self._save()
            return True
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
            return True
        if self.observe_only and ledger_qty > held + tolerance and (ledger_qty - held) * price >= 5.0:
            log(
                f"  ! [STRAT] LEDGER RECONCILE: external BUY on venue ({base}: "
                f"{held:.{self.vol_dec}f} -> {ledger_qty:.{self.vol_dec}f}); synchronizing state"
            )
            fill_px = price
            fill_ts = clean_ts if clean_ts is not None else self._now()
            try:
                raw_cli = getattr(self.client, "_inner", self.client)
                if hasattr(raw_cli, "get_orders"):
                    buy_rows = raw_cli.get_orders(self.pair, "BUY", 3600.0)
                    if isinstance(buy_rows, list) and buy_rows and isinstance(buy_rows[-1], dict):
                        cand_px = _clean_pos_float(buy_rows[-1].get("price"))
                        if cand_px is not None:
                            fill_px = cand_px
                        cand_ts_ms = _clean_pos_float(buy_rows[-1].get("timestamp"))
                        if cand_ts_ms is not None:
                            fill_ts = cand_ts_ms / 1000.0 if cand_ts_ms > 1e11 else cand_ts_ms
            except Exception:
                pass
            added_cost = (ledger_qty - held) * fill_px
            self.s["qty"] = ledger_qty
            self.s["cost"] = float(self.s.get("cost", 0.0)) + added_cost
            self.s["spent"] = float(self.s.get("spent", 0.0)) + added_cost
            if held <= tolerance or self.s.get("entry_price") is None:
                self.s["entry_price"] = fill_px
                self.s["entry_ts"] = fill_ts
            else:
                self.s["dca_buys"] = int(self.s.get("dca_buys") or 0) + 1
            self.s["last_buy_price"] = fill_px
            self.s["ledger_reconcile_required"] = False
            self._save()
            return True
        self.s["ledger_reconcile_required"] = False
        if had_persisted_reconcile:
            self._save()
        return False

    # -- Step Pipeline Sub-Stages ----------------------------------------------
    def _step_sync_market(
        self, price: float, timestamp: float | None = None, high: float | None = None
    ) -> tuple[float, float, float, float, float | None] | None:
        """Stage 1: Validate market price, sync venue ledger, record shadow ticks, and check adoption."""
        clean_price = _clean_pos_float(price)
        if clean_price is None:
            log("  ! [STRAT] invalid price — deferring step")
            return None
        price = clean_price
        if high is not None:
            high = _clean_pos_float(high)
        if self.s.get("pending_intent"):
            log("  [STRAT] pre-submit intent in reconciliation — deferring new decisions")
            return None
        clean_ts = _clean_nonneg_float(timestamp)
        had_persisted_reconcile = bool(self.s.get("ledger_reconcile_required"))
        if self.observe_only:
            tick_time = clean_ts if clean_ts is not None else self._now()
            self._shadow_prices.append((tick_time, price))
            if not self.dry_run and not self.s.get("orders"):
                self.s["ledger_reconcile_required"] = True
            else:
                return None
        held = self.s["qty"]
        if (
            not self.dry_run and self.s.get("ledger_reconcile_required")
            and (held > 1e-12 or self.observe_only) and not self.s.get("orders")
        ):
            if self._reconcile_venue_ledger(price, clean_ts, had_persisted_reconcile):
                return None
        if self.observe_only:
            return None
        entry_px = sr.entry_price(price, self.p.entry_discount_pct)
        if clean_ts is not None:
            self.set_simulated_time(clean_ts)
            tick_time = clean_ts
        else:
            tick_time = self._now()
        self._shadow_prices.append((tick_time, price))
        if self._surge_replay_closes is not None:
            self._surge_replay_closes.append(price)

        if self.p.adopt_cost > 0 and not self.s.get("adopted") and held <= 1e-12:
            self._maybe_adopt()
            if not self.s.get("adopted"):
                return None
            held = self.s["qty"]
        return price, entry_px, held, tick_time, high

    def _step_evaluate_exits(
        self, price: float, held: float, *, tick_time: float
    ) -> tuple[bool, MarketRegimeDecision | None, list]:
        """Stage 2: Evaluate hard stop-loss, pending market exits, regime context, and trend overlay."""
        if held > 1e-12 and self._check_stop_loss(price):
            return True, None, []
        if self._has_pending_market_exit():
            return True, None, []
        if self.s.get("pending_exit"):
            self._continue_market_exit(price)
            return True, None, []

        regime = None
        regime_closes = []
        needs_regime = self.p.trend_overlay or self.p.reentry_hybrid_enabled or (
            held > 1e-12 and (
                self.p.dca_trend_brake or self.p.tp_dynamic_flat
                or self.p.surge_guard or self.p.slow_grind_guard
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
                    self.p.trend_overlay and bool(self.s.get("trend_mode"))
                    and not self.p.trend_exit_break and not self.p.slow_grind_guard
                )
                or (not self.p.trend_overlay and held <= 1e-12 and self._has_open("buy"))
            )
        ):
            regime, regime_closes = self._regime_context()

        self._reentry_blocked_tick = None
        if self.p.trend_overlay and self._overlay_step(price, regime, regime_closes, tick_time=tick_time):
            return True, regime, regime_closes
        return False, regime, regime_closes

    def _step_evaluate_reentry(
        self, price: float, entry_px: float, regime: MarketRegimeDecision | None, tick_time: float
    ) -> bool:
        """Stage 3: Evaluate flat-position re-entry gate and place initial ENTRY buy order."""
        if self._has_open("buy"):
            return False
        return self._execute_flat_reentry(
            price, entry_px, self._effective_entry_amount(), self._effective_max_budget(),
            regime=regime, tick_time=tick_time,
        )

    def _evaluate_mdhtp_tp(
        self,
        price: float,
        avg: float,
        held: float,
        eff_tp: float,
        *,
        high: float | None = None,
        regime: MarketRegimeDecision | None = None,
        regime_closes: list | None = None,
        tick_time: float | None = None,
    ) -> bool:
        """Evaluate Multi-Horizon Dynamic Take-Profit (MDHTP); return True if tick TP handling is complete."""
        if self.exit_manager is None:
            self.exit_manager = self._init_exit_manager()
        if (
            not self.replay_mode
            or self.pair in self.exit_manager.tp_ride_state
            or (self.p.tp_trend_hold and self._is_trend_up(regime) and high is not None and high > price)
        ):
            eff_trail = self._effective_trail_pct()
        else:
            eff_trail = self.p.tp_trail_pct
        self._sync_mdhtp_config(eff_trail)

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
                self.exit_manager.evaluate_take_profit(high_snap, custom_threshold=eff_tp / 100.0)
            prev_ride_state = dict(self.exit_manager.tp_ride_state.get(self.pair, {}))

        decision = self.exit_manager.evaluate_take_profit(snapshot, custom_threshold=eff_tp / 100.0)
        if decision.action == ExitAction.SELL and decision.should_exit:
            exit_px = round(price * 0.999, self.price_dec)
            use_soft = str(decision.reason or "").startswith("mdhtp_peak_pullback")
            profit_floor = self._trail_profit_floor_price(avg) if use_soft else None
            if use_soft and profit_floor is not None and exit_px < profit_floor:
                if prev_ride_state:
                    self.exit_manager.tp_ride_state[self.pair] = prev_ride_state
                log(
                    f"  [STRAT] soft trailing blocked: reference {exit_px:.{self.price_dec}f} below the "
                    f"profitable floor {profit_floor:.{self.price_dec}f}; hard stop stays MARKET"
                )
                return True
            if self._request_market_exit(price, "TP", soft_floor=use_soft):
                log(f"  [STRAT] MDHTP EXIT ({decision.reason}) -> reference {exit_px}")
            return True
        if decision.reason == "not_in_tp_territory":
            self.s["trail_peak"] = None
            self.s["trail_stop"] = None
            self._cancel_orders("sell", exclude_market=True)
            return True
        if decision.action == ExitAction.HOLD:
            ride_state = self.exit_manager.tp_ride_state.get(self.pair)
            peak_px = (
                safe_pos_float(ride_state.get("peak_price"), None) if isinstance(ride_state, dict) else None
            ) or price
            self.s["trail_peak"] = peak_px
            if self._check_advanced_profit_guards(price, avg, held, regime_closes, tick_time):
                return True
            self._cancel_orders("sell", exclude_market=True)
            log(f"  [STRAT] above the TP, RIDING IT ({decision.reason}) (peak {peak_px:.{self.price_dec}f})")
            return True
        return False

    def _dispatch_take_profit(
        self,
        price: float,
        avg: float | None,
        held: float,
        *,
        high: float | None = None,
        regime: MarketRegimeDecision | None = None,
        regime_closes: list | None = None,
        tick_time: float | None = None,
    ) -> bool:
        """Manage take-profit (MDHTP, trailing stop, or classic/tranche limit sells). Return True to skip DCA."""
        if not (self.p.enable_takeprofit and avg):
            return False
        eff_tp = self.p.takeprofit_pct
        if self.p.tp_dynamic_flat and regime and regime.fresh and regime.regime == "sideways":
            thresh = getattr(getattr(self, "_regime_service", None), "evaluator", None)
            eff_tp = sr.dynamic_flat_tp_pct(
                strength=regime.strength,
                strength_threshold=getattr(thresh, "strength_threshold", 2.0),
                tp_min_pct=self.p.tp_min_pct,
                tp_max_pct=self.p.tp_max_pct,
                base_tp_pct=self.p.takeprofit_pct,
            )

        trail_armed = _clean_pos_float(self.s.get("trail_peak")) is not None
        trend_hold_active = self.p.tp_trend_hold and not bool(self.p.tp_tranches)
        tp_ref_price = high if high is not None else price
        if trend_hold_active and (trail_armed or tp_ref_price >= sr.tp_price(avg, eff_tp)):
            if self.p.tp_mdhtp_enabled and self._evaluate_mdhtp_tp(
                price, avg, held, eff_tp, high=high, regime=regime,
                regime_closes=regime_closes, tick_time=tick_time,
            ):
                return True
            peak = max(safe_pos_float(self.s.get("trail_peak"), tp_ref_price), tp_ref_price)
            self.s["trail_peak"] = peak
            if self._check_advanced_profit_guards(price, avg, held, regime_closes, tick_time):
                return True
            eff_trail = self._effective_trail_pct()
            candidate_stop = peak * (1 - eff_trail / 100)
            previous_stop = safe_pos_float(self.s.get("trail_stop"), default=None)
            trail_stop = max(candidate_stop, previous_stop or candidate_stop)
            self.s["trail_stop"] = trail_stop
            exit_px = round(price * 0.999, self.price_dec)
            profit_floor = self._trail_profit_floor_price(avg)
            if price <= trail_stop and profit_floor is not None and exit_px < profit_floor:
                log(
                    f"  [STRAT] soft trailing blocked: reference {exit_px:.{self.price_dec}f} below the "
                    f"profitable floor {profit_floor:.{self.price_dec}f}; hard stop stays MARKET"
                )
                return True
            if price <= trail_stop:
                if self._request_market_exit(price, "TP", soft_floor=True):
                    log(
                        f"  [STRAT] trailing: pullback {eff_trail:.2f}% from peak "
                        f"{peak:.{self.price_dec}f} -> exit reference {exit_px}"
                    )
                return True
            self._cancel_orders("sell", exclude_market=True)
            log(
                f"  [STRAT] above the TP, RIDING IT (peak {peak:.{self.price_dec}f}, "
                f"trail-stop {trail_stop:.{self.price_dec}f})"
            )
            return False

        self.s["trail_peak"] = None
        self.s["trail_stop"] = None
        if trend_hold_active:
            self._cancel_orders("sell", exclude_market=True)
            return False

        tranches = self.p.tp_tranches or [(eff_tp, 100.0)]
        desired, rem = [], held
        for i, (pct, share) in enumerate(tranches):
            q = (
                self._dust_safe_qty(rem)
                if i == len(tranches) - 1
                else min(rem, round(held * share / 100, self.vol_dec))
            )
            rem = round(rem - q, self.vol_dec)
            if q > 0:
                desired.append((round(sr.tp_price(avg, pct), self.price_dec), q))
        if self.ordermin and any(q < self.ordermin for _, q in desired):
            safe_held = self._dust_safe_qty(held)
            desired = (
                [(round(sr.tp_price(avg, tranches[0][0]), self.price_dec), safe_held)]
                if safe_held >= self.ordermin else []
            )
        sells = [o for o in self.s["orders"] if o["side"] == "sell"]
        ok = (
            len(sells) == len(desired)
            and all(not o.get("cancel_requested") for o in sells)
            and all(
                abs(o["price"] - p) / p <= 0.001 and abs(o["vol"] - q) <= 1e-9
                for o, (p, q) in zip(sorted(sells, key=lambda x: x["price"]), sorted(desired))
            )
        )
        if not ok:
            self._cancel_orders("sell")
            if not any(o["side"] == "sell" for o in self.s["orders"]):
                for p_, q_ in desired:
                    self._place("sell", q_, p_, kind="TP")
        return False

    def _dispatch_dca_buy(
        self, price: float, entry_px: float, regime: MarketRegimeDecision | None
    ) -> None:
        """Evaluate progressive/volatility-scaled DCA dip conditions and place BUY order."""
        effective_dca_drop = sr.progressive_dca_drop_pct(
            self.p.dca_drop_pct, self.p.dca_spacing_growth_pct, self.s["dca_buys"]
        )
        effective_dca_amount = self._effective_dca_amount()
        in_sl_bounce = bool(self.exit_manager is not None and self.pair in self.exit_manager.sl_bounce_state)
        if not (
            self.s["dca_buys"] < self.p.max_dca_buys
            and not in_sl_bounce
            and self.s["last_buy_price"]
            and sr.dca_price_hit(
                price, self.s["last_buy_price"], effective_dca_drop, self.p.reentry_tolerance_pct
            )
            and self.s["spent"] < self._effective_max_budget()
            and not (
                self.p.dca_trend_brake
                and self._regime_matches(
                    regime, "bear", min_move_pct=self.p.dca_brake_min_pct, min_samples=self.p.regime_min_samples
                )
            )
            and not self._has_open("buy")
        ):
            return
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
            entry_px = round(entry_px, self.price_dec)
            qty = (
                math.floor(spend / entry_px * 10 ** self.vol_dec) / 10 ** self.vol_dec
                if entry_px > 0 else 0.0
            )
        if qty <= 0 or qty < self.ordermin:
            log(f"  [STRAT] dip {price}: DCA skipped — available {spend:.2f} {self.ccy} too small for a fill")
            return
        partial = " (resized to available funds/budget)" if spend < effective_dca_amount else ""
        log(
            f"  [STRAT] dip {price} <= {self.s['last_buy_price']}"
            f"×(1-{effective_dca_drop}%) (tol {self.p.reentry_tolerance_pct}%) — DCA {spend:.0f}{partial}"
        )
        self._place("buy", qty, entry_px, kind="DCA", amount=spend)

    def _step_dispatch_orders(
        self,
        price: float,
        entry_px: float,
        held: float,
        *,
        high: float | None = None,
        regime: MarketRegimeDecision | None = None,
        regime_closes: list | None = None,
        tick_time: float | None = None,
    ) -> None:
        """Stage 4: Evaluate take-profit (MDHTP, trailing, or classic/tranche) and DCA order dispatch."""
        avg = self._avg()
        if held > 1e-12 and self.s.get("entry_ts") is None:
            self.s["entry_ts"] = tick_time
        if self._dispatch_take_profit(
            price, avg, held, high=high, regime=regime, regime_closes=regime_closes, tick_time=tick_time
        ):
            return
        self._dispatch_dca_buy(price, entry_px, regime)

    def step(
        self,
        price: float,
        timestamp: float | None = None,
        high: float | None = None,
        low: float | None = None,
    ) -> None:
        sync_ctx = self._step_sync_market(price, timestamp=timestamp, high=high)
        if sync_ctx is None:
            return
        price, entry_px, held, tick_time, high = sync_ctx

        exited, regime, regime_closes = self._step_evaluate_exits(price, held, tick_time=tick_time)
        if exited:
            return

        if held <= 1e-12:
            self._step_evaluate_reentry(price, entry_px, regime, tick_time)
            return

        self._step_dispatch_orders(
            price, entry_px, held, high=high, regime=regime,
            regime_closes=regime_closes, tick_time=tick_time,
        )

    # -- Main Loop -------------------------------------------------------------
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
            log(f"      DCA growth : +{self.p.dca_spacing_growth_pct}pp after each executed DCA")
        if self.p.dca_vol_scale_k:
            log(
                f"      DCA vol    : k={self.p.dca_vol_scale_k}, "
                f"reference={self.p.dca_vol_ref}%, OHLC={self.p.dca_vol_interval}m"
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
                    if not self._recover_state_write_or_skip():
                        time.sleep(sleep_sec)
                        continue
                    self._reload_if_externally_modified()
                    self.reconcile(price)
                    self.step(price)
                    self._save()
                except Exception as e:  # noqa: BLE001 — RESILIENCE: network/API down -> retry
                    log(f"  ! [STRAT] error ({e.__class__.__name__}: {e}) — retrying")
                    time.sleep(sleep_sec)
                    continue
                avg = self._avg()
                pos = f"qty={self.s['qty']:.8f} avg={avg:.{self.price_dec}f}" if avg else "qty=0 (waiting for an entry)"
                log(
                    f"  [STRAT] price={price}  {pos}  "
                    f"NET={self.s['realized_net']:+.2f} (gross {self.s['realized_gross']:+.2f}, "
                    f"fee {self.s['fees_total']:.2f}) {self.ccy}  ord={len(self.s['orders'])}"
                )
                time.sleep(sleep_sec)
        except KeyboardInterrupt:
            log("  [STRAT] stopped manually.")
            self._save()
