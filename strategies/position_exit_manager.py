"""Centralized Position Exit Manager (MHDSL Stop-Loss & MDHTP Take-Profit Mirror).

Provides a provider-agnostic, mathematical mirror state machine for position exits:
  - Take-Profit (MDHTP): Trend-riding deferral, peak tracking, dynamic micro-trailing
    pullback, guaranteed profit floor, stagnation timeout, and trend reversal.
  - Stop-Loss (MHDSL): Priority 1 Hard Disaster Floor (parallel sigma_1h volatility scaling),
    Priority 2 Bounce Protection (trend deferral, micro-trailing bounce rollover, dynamic
    ratchet, and dual-variant structural support breakdown sensor), Priority 3 Smart Stop-Loss.

Mathematical symmetry:
  - Zone Trigger: Gain >= gain_threshold (TP) vs Loss >= lost_threshold (SL)
  - Circuit Breaker: Guaranteed profit floor (TP) vs Hard disaster floor (SL)
  - Trend Deferral: Ride trend up (TP) vs Watch bounce on trend up (SL)
  - Micro-Trailing: Peak pullback (TP) vs Bounce rollover (SL)
  - Dynamic Ratchet: Tightens trail as profit grows (TP) vs tightens allowed pullback
    as bounce recovery expands (SL)
  - Timeout: Stagnation timeout (TP) vs Bounce timeout (SL)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from strategies.spot_rules import are_close, diff_percent
except ImportError:
    def diff_percent(v1: float, v2: float) -> float:
        if v1 == 0 and v2 == 0:
            return 0.0
        return abs(v1 - v2) / ((abs(v1) + abs(v2)) / 2) * 100

    def are_close(v1: float, v2: float, tol_pct: float) -> bool:
        return diff_percent(v1, v2) <= tol_pct

try:
    from intelligence.internal.state.volatility import resolve_cached_volatility_1h, vol_ratio_to_pct
except ImportError:
    from intelligence import resolve_cached_volatility_1h, vol_ratio_to_pct


class ExitAction(str, Enum):
    """Action recommended by PositionExitManager."""
    HOLD = "HOLD"                  # Do not exit; position is healthy or deferred
    SELL = "SELL"                  # Execute complete position exit (market or limit)
    PARTIAL_SELL = "PARTIAL_SELL"  # Execute partial position exit (e.g. spike Hard-TP)


@dataclass
class ExitDecision:
    """Decision returned after evaluating a position snapshot."""
    action: ExitAction
    should_exit: bool
    reason: str
    forced: bool = False
    sell_fraction: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class PriceGeometrySnapshot:
    """Base mixin providing unified, fuzzy-tolerant price geometry primitives."""

    @property
    def is_valid_price(self) -> bool:
        """Verify whether current market price is valid and strictly positive."""
        return self.current_price > 0

    def bounce_ratio(self, trough_price: float) -> float:
        """Calculate recovery bounce ratio above trough: (current - trough) / trough."""
        if trough_price <= 0:
            return 0.0
        diff = self.current_price - trough_price
        return diff / trough_price if diff > 0.0 else 0.0

    def pullback_ratio(self, peak_price: float) -> float:
        """Normalized pullback ratio from peak: (peak - current) / peak."""
        if peak_price <= 0:
            return 0.0
        diff = peak_price - self.current_price
        return diff / peak_price if diff > 0.0 else 0.0

    def discount_drop_ratio(self, reference_price: float) -> float:
        """Calculate price discount ratio below reference sale: (reference - current) / reference."""
        if reference_price <= 0:
            return 0.0
        return (reference_price - self.current_price) / reference_price

    def has_bounced(self, trough_price: float, threshold: float, tol_pct: float = 0.0) -> bool:
        """Determine if price has bounced above trough honoring fuzzy tolerance."""
        if trough_price <= 0 or threshold <= 0:
            return False
        if self.bounce_ratio(trough_price) >= threshold:
            return True
        return tol_pct > 0 and are_close(self.current_price, trough_price * (1.0 + threshold), tol_pct)

    def has_pulled_back(self, peak_price: float, threshold: float, tol_pct: float = 0.0) -> bool:
        """Determine if price has pulled back from peak honoring fuzzy tolerance."""
        if peak_price <= 0 or threshold <= 0:
            return False
        if self.pullback_ratio(peak_price) >= threshold:
            return True
        return tol_pct > 0 and are_close(self.current_price, peak_price * (1.0 - threshold), tol_pct)

    def has_dropped(self, reference_price: float, threshold: float, tol_pct: float = 0.0) -> bool:
        """Determine if price has dropped below reference sale honoring fuzzy tolerance."""
        if reference_price <= 0:
            return False
        target = reference_price * (1.0 - threshold)
        px = self.current_price
        if px <= target:
            return True
        return tol_pct > 0 and are_close(px, target, tol_pct)


@dataclass
class PositionSnapshot(PriceGeometrySnapshot):
    """Provider-agnostic snapshot of an open position and current market state."""
    symbol: str
    current_price: float
    avg_buy_price: float
    net_qty: float = 0.0
    sellable_qty: float = 0.0
    current_time_s: float = 0.0
    trend_up: bool = False
    vol_1h: float = 0.0  # Hourly realized volatility as decimal ratio (e.g. 0.02 for 2.0%)
    vol_source: str = "default"
    provider: Optional[str] = None
    shadow_prices: List[Tuple[float, float]] = field(default_factory=list)
    sma_value: Optional[float] = None
    window_move_pct: Optional[float] = None

    @property
    def price_change_ratio(self) -> float:
        """Normalized price change relative to cost: (current - buy) / buy."""
        if self.avg_buy_price <= 0:
            return 0.0
        return (self.current_price - self.avg_buy_price) / self.avg_buy_price

    @property
    def gain_ratio(self) -> float:
        """Positive price gain ratio, or 0.0 if position is in loss."""
        bp = self.avg_buy_price
        if bp <= 0:
            return 0.0
        diff = self.current_price - bp
        return diff / bp if diff > 0.0 else 0.0

    @property
    def loss_ratio(self) -> float:
        """Positive price loss ratio, or 0.0 if position is in profit."""
        bp = self.avg_buy_price
        if bp <= 0:
            return 0.0
        diff = bp - self.current_price
        return diff / bp if diff > 0.0 else 0.0

    @property
    def gain_pct(self) -> float:
        """Percentage gain: e.g. 7.5 for +7.5%."""
        return self.gain_ratio * 100.0

    @property
    def loss_pct(self) -> float:
        """Percentage loss: e.g. 3.2 for -3.2%."""
        return self.loss_ratio * 100.0

    @property
    def is_valid_price(self) -> bool:
        """Verify whether price geometry is valid and strictly positive."""
        return self.avg_buy_price > 0 and self.current_price > 0

    def in_sl_territory(self, threshold: float, tol_pct: float = 0.0) -> bool:
        """Determine if position is in stop-loss territory honoring fuzzy tolerance."""
        loss = self.loss_ratio
        if loss >= threshold:
            return True
        if (loss > 0.0 or tol_pct >= 200.0) and tol_pct > 0 and are_close(loss, threshold, tol_pct):
            return True
        return False

    def in_tp_territory(self, threshold: float, tol_pct: float = 0.0) -> bool:
        """Determine if position is in take-profit territory honoring fuzzy tolerance."""
        gain = self.gain_ratio
        if gain >= threshold:
            return True
        if (gain > 0.0 or tol_pct >= 200.0) and tol_pct > 0 and are_close(gain, threshold, tol_pct):
            return True
        return False


class ReentryAction(str, Enum):
    """Action recommended by PositionExitManager for market re-entry."""
    WAIT = "WAIT"    # Do not re-enter yet (blocked by drop, bounce, or trend gate)
    ENTER = "ENTER"  # Conditions met to execute re-entry buy


@dataclass
class ReentryDecision:
    """Decision returned after evaluating a re-entry snapshot."""
    action: ReentryAction
    should_enter: bool
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReentrySnapshot(PriceGeometrySnapshot):
    """Provider-agnostic snapshot of last exit state and market context for re-entry."""
    symbol: str
    current_price: float
    last_sell_price: Optional[float] = None
    last_exit_kind: Optional[str] = None  # "STOP", "TP", "SELL"
    post_sell_peak: Optional[float] = None
    post_sell_trough: Optional[float] = None
    current_time_s: float = 0.0
    last_sell_time_s: Optional[float] = None
    trend_up: bool = False
    trend_down: bool = False
    vol_1h: float = 0.0  # Hourly realized volatility as decimal ratio (e.g. 0.02 for 2.0%)
    vol_source: str = "default"

    @property
    def sale_drop_ratio(self) -> float:
        """Calculate price discount ratio below last_sell_price."""
        return self.discount_drop_ratio(self.last_sell_price) if self.last_sell_price else 0.0

    @property
    def sale_drop_pct(self) -> float:
        """Calculate price discount percentage below last_sell_price."""
        return self.sale_drop_ratio * 100.0


@dataclass
class PositionExitConfig:
    """Configuration governing Stop-Loss (MHDSL) and Take-Profit (MDHTP) mechanics."""

    # --- Stop-Loss (MHDSL) Parameters ---
    lost_threshold: float = 0.05
    sl_disaster_buffer_ratio: float = 0.30
    sl_disaster_vol_multiplier: float = 0.85
    sl_disaster_buffer_min_pct: float = 0.008
    sl_disaster_buffer_max_pct: float = 0.15  # Upper bound clamp on disaster buffer (15% decimal)
    sl_trend_defer_max_sec: float = 220.0
    sl_bounce_min_recovery_pct: float = 0.004
    sl_bounce_pullback_pct: float = 0.003
    sl_bounce_ratchet_enabled: bool = True
    sl_bounce_ratchet_k: float = 0.50
    sl_bounce_min_pullback_pct: float = 0.0015
    sl_structural_break_enabled: bool = True
    smart_sl_enabled: bool = True

    # --- Take-Profit (MDHTP) Parameters ---
    gain_threshold: float = 0.07
    tp_trend_ride_enabled: bool = True
    tp_profit_floor_ratio: float = 0.50
    tp_profit_floor_min_pct: float = 0.008
    tp_pullback_pct: float = 0.006
    tp_stagnation_max_sec: float = 1080.0
    tp_ratchet_enabled: bool = True
    tp_ratchet_base_trail: float = 0.006
    tp_ratchet_min_trail: float = 0.0025
    tp_ratchet_k: float = 0.10

    # --- Hard TP (Spike Exit) ---
    hard_tp_enabled: bool = False
    hard_tp_pct: float = 0.20
    hard_tp_frac: float = 0.50
    hard_tp_cooldown_sec: float = 3600.0

    # --- Tolerance & Fuzzy Matching ---
    are_close_tolerance_pct: float = 1.0

    # --- Advanced Guards (spot_rules hooks) ---
    fast_profit_guard_enabled: bool = False
    fast_profit_mult: float = 2.0
    fast_profit_window_sec: float = 60.0
    fast_profit_drop_pct: float = 0.40
    surge_guard_enabled: bool = False
    surge_gain_pct: float = 18.0
    surge_exit_pullback_pct: float = 3.5
    surge_move_pct: float = 25.0
    slow_grind_guard_enabled: bool = False
    slow_grind_min_days: float = 7.0
    slow_grind_min_gain_pct: float = 15.0
    slow_grind_flash_window_sec: float = 900.0
    slow_grind_flash_drop_pct: float = 1.5
    slow_grind_structural_drop_pct: float = 2.5

    # --- Re-Entry Mirror (MHDRE) Parameters ---
    reentry_enabled: bool = True
    reentry_sl_bounce_pct: float = 0.015       # 1.5% bounce from trough after STOP
    reentry_drop_pct: float = 0.020            # 2.0% drop below last sell after TP
    reentry_pullback_pct: float = 0.015        # 1.5% pullback from peak in bull trend
    reentry_bear_bounce_pct: float = 0.010     # 1.0% bounce from trough in bear trend
    reentry_ttl_hours: float = 0.0             # 0 = disabled (hours)
    reentry_tolerance_pct: float = 0.05        # 0.05% tolerance
    reentry_ratchet_enabled: bool = True       # Ratchet prevents re-entry on rollover
    reentry_bear_gate_enabled: bool = True     # Block/require stronger bounce in bear trend
    reentry_hybrid_enabled: bool = False       # Enable hybrid pullback in confirmed bull trends
    reentry_peak_relative: bool = False        # Enable peak-relative pullback across all regimes
    reentry_post_stop_cooldown_sec: float = 0.0  # Minimum stabilization cooldown (seconds) after STOP before re-entry
    allow_external_cache: bool = True          # False in replay/backtests to enforce strict simulation determinism

def _clear_smart_sl(symbol: str, provider: Optional[str] = None) -> None:
    try:
        from intelligence.external.guards.smart_stop_loss_guard import clear_smart_stop_loss
        clear_smart_stop_loss(symbol, provider=provider)
    except Exception:
        pass


PERSISTED_EXIT_STATE_KEYS: Tuple[str, ...] = (
    "sl_bounce_state",
    "sl_episode_state",
    "tp_ride_state",
    "reentry_state",
)


class PositionExitManager:
    """Fleet-wide centralized manager for Stop-Loss and Take-Profit position exits."""

    def __init__(
        self,
        config: Optional[PositionExitConfig] = None,
        vol_resolver: Optional[Callable[[str], Tuple[float, str]]] = None,
        smart_sl_evaluator: Optional[Callable[..., Tuple[bool, str, Dict[str, Any]]]] = None,
    ):
        self.config = config or PositionExitConfig()
        self.vol_resolver = vol_resolver
        self.smart_sl_evaluator = smart_sl_evaluator

        # In-memory lifecycle states keyed by symbol
        self.sl_bounce_state: Dict[str, Dict[str, Any]] = {}
        # SSOT for the whole Stop-Loss episode (shared by every deferral layer)
        self.sl_episode_state: Dict[str, Dict[str, Any]] = {}
        self.tp_ride_state: Dict[str, Dict[str, Any]] = {}
        self.reentry_state: Dict[str, Dict[str, Any]] = {}
        self._last_popped_reentry_state: Dict[str, Dict[str, Any]] = {}
        self.hard_tp_last: Dict[str, float] = {}

    def _pop_reentry_state(self, symbol: str) -> None:
        popped = self.reentry_state.pop(symbol, None)
        if popped is not None:
            self._last_popped_reentry_state[symbol] = dict(popped)

    def restore_reentry_state(self, symbol: str) -> None:
        """Restore the most recently popped re-entry state if an approved entry was not placed."""
        popped = self._last_popped_reentry_state.pop(symbol, None)
        if popped is not None and symbol not in self.reentry_state:
            self.reentry_state[symbol] = dict(popped)

    def bind_state(self, store: Dict[str, Any]) -> None:
        """Bind durable lifecycle dicts to an external persisted state store."""
        for key in PERSISTED_EXIT_STATE_KEYS:
            setattr(self, key, store.setdefault(key, {}))

    def clear_state(self, symbol: Optional[str] = None) -> None:
        """Clear tracking state for a single symbol or all symbols."""
        if symbol:
            self.sl_bounce_state.pop(symbol, None)
            self.sl_episode_state.pop(symbol, None)
            self.tp_ride_state.pop(symbol, None)
            self.reentry_state.pop(symbol, None)
            self._last_popped_reentry_state.pop(symbol, None)
            self.hard_tp_last.pop(symbol, None)
        else:
            self.sl_bounce_state.clear()
            self.sl_episode_state.clear()
            self.tp_ride_state.clear()
            self.reentry_state.clear()
            self._last_popped_reentry_state.clear()
            self.hard_tp_last.clear()

    def get_state(self, symbol: str) -> Dict[str, Any]:
        """Return the current active states for a symbol."""
        return {
            "sl_bounce_state": self.sl_bounce_state.get(symbol),
            "sl_episode_state": self.sl_episode_state.get(symbol),
            "tp_ride_state": self.tp_ride_state.get(symbol),
            "reentry_state": self.reentry_state.get(symbol),
            "hard_tp_last": self.hard_tp_last.get(symbol),
        }

    def reset_sl_episode(self, symbol: str) -> None:
        """End the current Stop-Loss episode (position left SL territory, closed or was replaced)."""
        self.sl_episode_state.pop(symbol, None)

    def _lock_episode_disaster_floor(
        self,
        symbol: str,
        candidate_floor: float,
        buy_price: float,
        now: float,
    ) -> Tuple[float, bool]:
        """Monotonic ratchet: the disaster floor armed for an SL episode can only tighten.

        Applies to the whole episode regardless of which module defers the exit
        (trend bounce protection or Smart-SL), so a regime switch or a volatility
        spike can never loosen an already armed emergency limit. A change of the
        average entry price (new position / DCA) starts a fresh episode.

        Returns:
            (effective_floor, locked) where locked=True when the armed floor overrode the candidate.
        """
        episode = self.sl_episode_state.get(symbol)
        same_buy = False
        if episode is not None and episode.get("buy_price") is not None:
            try:
                same_buy = math.isclose(float(episode["buy_price"]), float(buy_price), rel_tol=1e-9, abs_tol=1e-12)
            except (TypeError, ValueError):
                same_buy = False
        if episode is None or not same_buy:
            self.sl_episode_state[symbol] = {
                "started_at": now,
                "buy_price": buy_price,
                "disaster_floor": candidate_floor,
            }
            return candidate_floor, False
        armed = episode.get("disaster_floor")
        if armed is not None and armed < candidate_floor:
            return float(armed), True
        episode["disaster_floor"] = candidate_floor
        return candidate_floor, False

    def record_hard_tp(self, symbol: str, timestamp: Optional[float] = None) -> None:
        """Record successful execution of a Hard-TP order to arm cooldown."""
        self.hard_tp_last[symbol] = timestamp if timestamp is not None else time.time()

    # -------------------------------------------------------------------------
    # Stop-Loss: Volatility-Scaled Disaster Buffer (Priority 1)
    # -------------------------------------------------------------------------
    def compute_disaster_buffer(
        self,
        symbol: str,
        lost_threshold: float,
        vol_1h: float = 0.0,
        vol_source: str = "default",
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute volatility-scaled disaster buffer running Variant A and Variant B in parallel.

        Variant A (Ratio Floor): buffer_a = lost_threshold * sl_disaster_buffer_ratio
        Variant B (Dynamic Volatility): buffer_b = vol_1h * sl_disaster_vol_multiplier
        Effective: max(buffer_a, buffer_b, sl_disaster_buffer_min_pct)
        """
        buf_a = lost_threshold * self.config.sl_disaster_buffer_ratio

        eff_vol = vol_1h if vol_1h is not None else 0.0
        eff_src = vol_source
        if eff_vol <= 0:
            if self.vol_resolver is not None:
                eff_vol, eff_src = self.vol_resolver(symbol)
                if eff_vol is None:
                    eff_vol = 0.0
            elif getattr(self.config, "allow_external_cache", True):
                cached = resolve_cached_volatility_1h(symbol)
                if cached is not None and cached > 0:
                    eff_vol = cached
                    eff_src = "cache"

        buf_b = eff_vol * self.config.sl_disaster_vol_multiplier if eff_vol > 0 else 0.0
        min_floor = self.config.sl_disaster_buffer_min_pct
        buf_raw = max(buf_a, buf_b, min_floor)
        max_cap = self.config.sl_disaster_buffer_max_pct
        capped = False
        if max_cap > 0 and buf_raw > max_cap:
            buf_effective = max_cap
            capped = True
        else:
            buf_effective = buf_raw

        source = (
            "capped_max"
            if capped
            else ("variant_b_vol1h" if (buf_b > buf_a and buf_b >= min_floor) else "variant_a_proportional")
        )

        metadata = {
            "variant_a_buffer": buf_a,
            "variant_b_buffer": buf_b,
            "variant_a_pct": round(vol_ratio_to_pct(buf_a), 3),
            "variant_b_pct": round(vol_ratio_to_pct(buf_b), 3) if eff_vol > 0 else None,
            "vol_1h_pct": round(vol_ratio_to_pct(eff_vol), 3) if eff_vol > 0 else None,
            "floor_min": min_floor,
            "buffer_max": max_cap,
            "capped": capped,
            "effective_buffer": buf_effective,
            "chosen_buffer_pct": round(vol_ratio_to_pct(buf_effective), 3),
            "source": source,
            "vol_1h": eff_vol,
            "vol_source": eff_src,
            "divergence_pct": vol_ratio_to_pct(buf_b - buf_a),
        }
        return buf_effective, metadata

    # -------------------------------------------------------------------------
    # Stop-Loss: Evaluation Engine (MHDSL)
    # -------------------------------------------------------------------------
    def evaluate_stop_loss(
        self,
        snapshot: PositionSnapshot,
        custom_threshold: Optional[float] = None,
    ) -> ExitDecision:
        decision = self._evaluate_stop_loss_core(snapshot, custom_threshold=custom_threshold)
        if decision.action == ExitAction.SELL or decision.should_exit:
            _clear_smart_sl(snapshot.symbol, snapshot.provider)
        return decision

    def _evaluate_stop_loss_core(
        self,
        snapshot: PositionSnapshot,
        custom_threshold: Optional[float] = None,
    ) -> ExitDecision:
        """Evaluate Stop-Loss with strict hierarchical priorities.

        Priority 1: Hard Disaster Floor Circuit Breaker
        Priority 2: Bounce Protection (Trend Deferral, Dynamic Ratchet, Structural Breakdown)
        Priority 3: Smart Stop-Loss Controller / Direct Exit
        """
        symbol = snapshot.symbol
        current_price = snapshot.current_price
        buy_price = snapshot.avg_buy_price
        now = snapshot.current_time_s if snapshot.current_time_s is not None else time.time()
        lost_threshold = custom_threshold if custom_threshold is not None else self.config.lost_threshold

        if not snapshot.is_valid_price:
            return ExitDecision(ExitAction.HOLD, False, "invalid_prices")

        price_decrease = snapshot.loss_ratio

        # Stop-loss boundary check
        existing_bounce = self.sl_bounce_state.get(symbol)
        if existing_bounce is not None and existing_bounce.get("buy_price") is not None:
            try:
                if not math.isclose(float(existing_bounce["buy_price"]), buy_price, rel_tol=1e-9, abs_tol=1e-12):
                    self.sl_bounce_state.pop(symbol, None)
            except (TypeError, ValueError):
                self.sl_bounce_state.pop(symbol, None)

        in_sl_territory = snapshot.in_sl_territory(lost_threshold, self.config.are_close_tolerance_pct)

        if not in_sl_territory:
            # Full recovery above stop-loss territory ends the SL episode
            self.sl_bounce_state.pop(symbol, None)
            self.reset_sl_episode(symbol)
            return ExitDecision(ExitAction.HOLD, False, "not_in_sl_territory")

        # Compute dynamic disaster floor
        disaster_buffer, disaster_meta = self.compute_disaster_buffer(
            symbol=symbol,
            lost_threshold=lost_threshold,
            vol_1h=snapshot.vol_1h,
            vol_source=snapshot.vol_source,
        )
        # Preserve already activated protection for the whole SL episode: once armed,
        # the disaster floor cannot loosen, whichever layer (bounce / Smart-SL) defers.
        disaster_floor, floor_locked = self._lock_episode_disaster_floor(
            symbol, lost_threshold + disaster_buffer, buy_price, now
        )
        if floor_locked:
            disaster_buffer = max(0.0, disaster_floor - lost_threshold)
            disaster_meta = {**disaster_meta, "source": "episode_locked", "effective_buffer": disaster_buffer}

        episode_state = self.sl_episode_state.get(symbol) or {}
        episode_started_at = float(episode_state.get("started_at", now))
        sl_episode_expired = (now - episode_started_at) >= self.config.sl_trend_defer_max_sec

        # ---------------------------------------------------------------------
        # PRIORITY 1: Hard Disaster Floor Circuit Breaker
        # ---------------------------------------------------------------------
        if price_decrease >= disaster_floor:
            self.sl_bounce_state.pop(symbol, None)
            reason = (
                f"hard_disaster_floor_breached (loss {price_decrease*100:.2f}% >= "
                f"floor {disaster_floor*100:.2f}%, buffer={disaster_buffer*100:.2f}% [{disaster_meta['source']}])"
            )
            return ExitDecision(
                action=ExitAction.SELL,
                should_exit=True,
                reason=reason,
                forced=True,
                metadata={
                    "priority": 1,
                    "disaster_floor": disaster_floor,
                    "disaster_meta": disaster_meta,
                    "sl_timer_expired": sl_episode_expired,
                },
            )

        # ---------------------------------------------------------------------
        # PRIORITY 2: Bounce Protection (Trend Up)
        # ---------------------------------------------------------------------
        if snapshot.trend_up:
            bounce_state = self.sl_bounce_state.get(symbol)
            if bounce_state is None:
                elapsed_ep = now - episode_started_at
                if elapsed_ep >= self.config.sl_trend_defer_max_sec:
                    reason = f"mhdsl_bounce_timeout ({elapsed_ep:.0f}s >= {self.config.sl_trend_defer_max_sec:.0f}s without full recovery)"
                    return ExitDecision(
                        action=ExitAction.SELL,
                        should_exit=True,
                        reason=reason,
                        forced=True,
                        metadata={"priority": 2, "sub_reason": "timeout", "elapsed": elapsed_ep, "sl_timer_expired": True},
                    )
                self.sl_bounce_state[symbol] = {
                    "started_at": episode_started_at,
                    "entry_loss": price_decrease,
                    "lowest_price": current_price,
                    "peak_bounce_price": current_price,
                    "min_loss": price_decrease,
                    "buy_price": buy_price,
                }
                reason = (
                    f"mhdsl_bounce_watch_initiated (dip={price_decrease*100:.2f}%, "
                    f"disaster_floor={disaster_floor*100:.2f}%, max_defer={self.config.sl_trend_defer_max_sec:.0f}s)"
                )
                return ExitDecision(
                    action=ExitAction.HOLD,
                    should_exit=False,
                    reason=reason,
                    forced=False,
                    metadata={"priority": 2, "state": "initiated", "disaster_floor": disaster_floor},
                )
            else:
                elapsed = now - bounce_state["started_at"]
                if elapsed >= self.config.sl_trend_defer_max_sec or sl_episode_expired:
                    eff_elapsed = max(elapsed, now - episode_started_at)
                    reason = f"mhdsl_bounce_timeout ({eff_elapsed:.0f}s >= {self.config.sl_trend_defer_max_sec:.0f}s without full recovery)"
                    self.sl_bounce_state.pop(symbol, None)
                    return ExitDecision(
                        action=ExitAction.SELL,
                        should_exit=True,
                        reason=reason,
                        forced=True,
                        metadata={"priority": 2, "sub_reason": "timeout", "elapsed": eff_elapsed, "sl_timer_expired": True},
                    )

                lowest_px = bounce_state.get("lowest_price") or current_price
                peak_px = bounce_state.get("peak_bounce_price") or current_price
                bounce_recovery = (peak_px - lowest_px) / lowest_px if lowest_px > 0 else 0.0
                bounce_attempted = bounce_recovery >= self.config.sl_bounce_min_recovery_pct

                sl_forced_reason: Optional[str] = None

                # Dual-variant support for structural support break
                if current_price < lowest_px and bounce_attempted and self.config.sl_structural_break_enabled:
                    sl_forced_reason = (
                        f"mhdsl_structural_break (failed bounce from {peak_px:.4f} "
                        f"[+{bounce_recovery*100:.2f}%] broke below support {lowest_px:.4f} to {current_price:.4f})"
                    )
                    self.sl_bounce_state.pop(symbol, None)
                elif current_price <= lowest_px:
                    # Baseline variant: reset trough and peak
                    bounce_state["lowest_price"] = current_price
                    bounce_state["peak_bounce_price"] = current_price
                    lowest_px = current_price
                    peak_px = current_price
                    bounce_recovery = 0.0
                elif current_price > peak_px:
                    bounce_state["peak_bounce_price"] = current_price
                    peak_px = current_price
                    bounce_recovery = (peak_px - lowest_px) / lowest_px if lowest_px > 0 else 0.0

                if sl_forced_reason:
                    return ExitDecision(
                        action=ExitAction.SELL,
                        should_exit=True,
                        reason=sl_forced_reason,
                        forced=True,
                        metadata={
                            "priority": 2,
                            "sub_reason": "structural_break",
                            "lowest_px": lowest_px,
                            "sl_timer_expired": False,
                        },
                    )

                if price_decrease < bounce_state["min_loss"]:
                    bounce_state["min_loss"] = price_decrease

                pullback_from_peak = snapshot.pullback_ratio(peak_px)

                eff_pullback_pct = self.config.sl_bounce_pullback_pct
                if self.config.sl_bounce_ratchet_enabled and bounce_recovery > self.config.sl_bounce_min_recovery_pct:
                    excess_rec = bounce_recovery - self.config.sl_bounce_min_recovery_pct
                    tightened = self.config.sl_bounce_pullback_pct - (excess_rec * self.config.sl_bounce_ratchet_k)
                    eff_pullback_pct = max(self.config.sl_bounce_min_pullback_pct, tightened)

                # Dynamic micro-trailing bounce rollover
                if bounce_attempted and pullback_from_peak >= eff_pullback_pct:
                    reason = (
                        f"mhdsl_bounce_rollover (peaked at {peak_px:.4f} [+{bounce_recovery*100:.2f}%], "
                        f"pulled back {pullback_from_peak*100:.2f}% >= {eff_pullback_pct*100:.2f}%)"
                    )
                    self.sl_bounce_state.pop(symbol, None)
                    return ExitDecision(
                        action=ExitAction.SELL,
                        should_exit=True,
                        reason=reason,
                        forced=True,
                        metadata={
                            "priority": 2,
                            "sub_reason": "rollover",
                            "pullback": pullback_from_peak,
                            "eff_pullback": eff_pullback_pct,
                            "sl_timer_expired": False,
                        },
                    )
                else:
                    reason = (
                        f"mhdsl_bounce_deferred ({elapsed:.0f}s/{self.config.sl_trend_defer_max_sec:.0f}s, "
                        f"loss={price_decrease*100:.2f}%, peak_px={peak_px:.4f}, floor={disaster_floor*100:.2f}%)"
                    )
                    return ExitDecision(
                        action=ExitAction.HOLD,
                        should_exit=False,
                        reason=reason,
                        forced=False,
                        metadata={
                            "priority": 2, "state": "riding_bounce", "elapsed": elapsed,
                            "peak_px": peak_px, "disaster_floor": disaster_floor,
                        },
                    )

        # Trend is NOT UP: clear any bounce deferral (the SL episode floor lock persists)
        self.sl_bounce_state.pop(symbol, None)

        # ---------------------------------------------------------------------
        # PRIORITY 3: Smart Stop-Loss Controller / Trend Check
        # ---------------------------------------------------------------------
        if self.config.smart_sl_enabled and not sl_episode_expired and not episode_state.get("smart_sl_done"):
            eval_fn = self.smart_sl_evaluator
            if eval_fn is None:
                try:
                    from intelligence.external.guards.smart_stop_loss_guard import evaluate_smart_stop_loss
                    eval_fn = evaluate_smart_stop_loss
                except ImportError:
                    eval_fn = None

            if eval_fn is not None:
                try:
                    should_exec, sl_reason, sl_meta = eval_fn(
                        symbol=symbol,
                        price_decrease=price_decrease,
                        lost_threshold=lost_threshold,
                        now=now,
                        provider=snapshot.provider,
                        current_price=current_price,
                        buy_price=buy_price,
                        disaster_buffer_pct=disaster_buffer * 100.0,  # episode-locked effective buffer
                    )
                    if not should_exec:
                        return ExitDecision(
                            action=ExitAction.HOLD,
                            should_exit=False,
                            reason=f"smart_stop_loss_deferred: {sl_reason}",
                            forced=False,
                            metadata={"priority": 3, "smart_sl_meta": sl_meta, "disaster_floor": disaster_floor},
                        )
                    else:
                        if "grace_period_expired" in str(sl_reason) or "active_crash_cascade" in str(sl_reason):
                            episode_state["smart_sl_done"] = True
                        return ExitDecision(
                            action=ExitAction.SELL,
                            should_exit=True,
                            reason=sl_reason,
                            forced=False,
                            metadata={"priority": 3, "smart_sl_meta": sl_meta, "disaster_floor": disaster_floor},
                        )
                except Exception:
                    pass

        # Baseline SL exit when trend is not up and smart controller is inactive
        reason = f"stop_loss_threshold_breached (loss {price_decrease*100:.2f}% >= {lost_threshold*100:.2f}%)"
        return ExitDecision(
            action=ExitAction.SELL,
            should_exit=True,
            reason=reason,
            forced=False,
            metadata={"priority": 3, "loss": price_decrease, "sl_timer_expired": sl_episode_expired},
        )

    def evaluate_take_profit(
        self,
        snapshot: PositionSnapshot,
        custom_threshold: Optional[float] = None,
    ) -> ExitDecision:
        decision = self._evaluate_take_profit_core(snapshot, custom_threshold=custom_threshold)
        if decision.action == ExitAction.SELL or decision.should_exit:
            _clear_smart_sl(snapshot.symbol, snapshot.provider)
        return decision

    def compute_tp_anomaly_pullback(
        self,
        symbol: str,
        gain_threshold: float,
        peak_gain: float,
        vol_1h: Optional[float] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute dynamic peak pullback threshold tightened on out-of-pattern spikes.

        Variant A (Proportional to TP target): anomaly_a = gain_threshold * 1.5
        Variant B (Realized 1h volatility anomaly): anomaly_b = gain_threshold + vol_1h * 2.5
        Anomaly threshold = max(anomaly_a, anomaly_b).
        When peak_gain exceeds the anomaly threshold, allowed pullback tightens from
        tp_pullback_pct down toward tp_ratchet_min_trail without selling against the trend.
        """
        eff_vol = vol_1h if vol_1h is not None else 0.0
        if eff_vol <= 0:
            if self.vol_resolver is not None:
                eff_vol, _ = self.vol_resolver(symbol)
                if eff_vol is None:
                    eff_vol = 0.0
            elif getattr(self.config, "allow_external_cache", True):
                cached = resolve_cached_volatility_1h(symbol)
                if cached is not None and cached > 0:
                    eff_vol = cached

        anomaly_a = gain_threshold * 1.5
        anomaly_b = (gain_threshold + eff_vol * 2.5) if eff_vol > 0 else 0.0
        anomaly_threshold = max(anomaly_a, anomaly_b)

        base_pullback = self.config.tp_pullback_pct
        min_pullback = min(base_pullback, self.config.tp_ratchet_min_trail)
        if peak_gain <= anomaly_threshold or self.config.tp_ratchet_k <= 0:
            eff_pullback = base_pullback
        else:
            excess_gain = peak_gain - anomaly_threshold
            eff_pullback = max(min_pullback, base_pullback - excess_gain * self.config.tp_ratchet_k)

        return eff_pullback, {
            "anomaly_threshold": anomaly_threshold,
            "vol_1h": eff_vol,
            "base_pullback": base_pullback,
            "eff_pullback": eff_pullback,
        }

    def _evaluate_take_profit_core(
        self,
        snapshot: PositionSnapshot,
        custom_threshold: Optional[float] = None,
    ) -> ExitDecision:
        """Evaluate Take-Profit with trend-riding mirror symmetry.

        Priority 0: Hard Take-Profit spike exit
        Priority 1: Trend Riding Initiation
        Priority 2: Guaranteed Profit Floor
        Priority 3: Dynamic Micro-Trailing Peak Pullback
        Priority 4: Stagnation Timeout (18 minutes)
        Priority 5: Trend Reversal Exit
        """
        symbol = snapshot.symbol
        current_price = snapshot.current_price
        buy_price = snapshot.avg_buy_price
        now = snapshot.current_time_s if snapshot.current_time_s is not None else time.time()
        gain_threshold = custom_threshold if custom_threshold is not None else self.config.gain_threshold

        if not snapshot.is_valid_price:
            return ExitDecision(ExitAction.HOLD, False, "invalid_prices")

        price_increase = snapshot.gain_ratio

        # ---------------------------------------------------------------------
        # PRIORITY 0: Hard TP Spike Exit
        # ---------------------------------------------------------------------
        if (
            self.config.hard_tp_enabled
            and 0 < self.config.hard_tp_frac <= 1.0
            and self.config.hard_tp_pct > 0
            and snapshot.in_tp_territory(self.config.hard_tp_pct, self.config.are_close_tolerance_pct)
        ):
            last_hard = self.hard_tp_last.get(symbol, 0.0)
            if last_hard <= 0 or (now - last_hard) >= self.config.hard_tp_cooldown_sec:
                reason = (
                    f"hard_tp_spike (+{snapshot.gain_pct:.1f}% >= {self.config.hard_tp_pct*100:.0f}%, "
                    f"selling {self.config.hard_tp_frac*100:.0f}%)"
                )
                return ExitDecision(
                    action=ExitAction.PARTIAL_SELL,
                    should_exit=True,
                    reason=reason,
                    forced=True,
                    sell_fraction=self.config.hard_tp_frac,
                    metadata={"priority": 0, "gain": snapshot.gain_ratio},
                )

        # ---------------------------------------------------------------------
        # Take-Profit Boundary Check
        # ---------------------------------------------------------------------
        profit_floor = max(gain_threshold * self.config.tp_profit_floor_ratio, self.config.tp_profit_floor_min_pct)

        existing_ride = self.tp_ride_state.get(symbol)
        if existing_ride is not None and existing_ride.get("buy_price") is not None:
            try:
                if not math.isclose(float(existing_ride["buy_price"]), buy_price, rel_tol=1e-9, abs_tol=1e-12):
                    self.tp_ride_state.pop(symbol, None)
            except (TypeError, ValueError):
                self.tp_ride_state.pop(symbol, None)

        # Gap down below cost while riding profit is an emergency breach of the profit floor
        if snapshot.gain_ratio <= 0:
            if symbol in self.tp_ride_state:
                ride_state = self.tp_ride_state.pop(symbol)
                peak_px = ride_state.get("peak_price", current_price)
                reason = (
                    f"mdhtp_profit_floor_breached (price {current_price:.4f} dropped below cost "
                    f"under floor {profit_floor*100:.2f}%, peak={peak_px:.4f})"
                )
                return ExitDecision(
                    action=ExitAction.SELL,
                    should_exit=True,
                    reason=reason,
                    forced=True,
                    metadata={"priority": 2, "profit_floor": profit_floor, "peak_px": peak_px, "gap_down_below_cost": True},
                )
            return ExitDecision(ExitAction.HOLD, False, "not_in_tp_territory")
        in_tp_territory = (
            snapshot.in_tp_territory(gain_threshold, self.config.are_close_tolerance_pct)
            or (symbol in self.tp_ride_state)
        )

        if not in_tp_territory:
            self.tp_ride_state.pop(symbol, None)
            return ExitDecision(ExitAction.HOLD, False, "not_in_tp_territory")

        ride_state = self.tp_ride_state.get(symbol)
        if ride_state is None:
            if snapshot.trend_up and self.config.tp_trend_ride_enabled:
                self.tp_ride_state[symbol] = {
                    "started_at": now,
                    "entry_gain": price_increase,
                    "peak_price": current_price,
                    "peak_gain": price_increase,
                    "last_peak_time": now,
                    "buy_price": buy_price,
                }
                reason = (
                    f"mdhtp_trend_ride_initiated (gain={price_increase*100:.2f}%, "
                    f"floor={profit_floor*100:.2f}%, max_stagnation={self.config.tp_stagnation_max_sec:.0f}s)"
                )
                return ExitDecision(
                    action=ExitAction.HOLD,
                    should_exit=False,
                    reason=reason,
                    forced=False,
                    metadata={"priority": 1, "state": "initiated", "profit_floor": profit_floor},
                )
            else:
                reason = "gain_threshold_exceeded_and_trend_not_up"
                return ExitDecision(
                    action=ExitAction.SELL,
                    should_exit=True,
                    reason=reason,
                    forced=False,
                    metadata={"priority": 1, "gain": price_increase},
                )
        else:
            # Update peak trajectory
            if current_price > ride_state["peak_price"]:
                ride_state["peak_price"] = current_price
                ride_state["peak_gain"] = price_increase
                ride_state["last_peak_time"] = now

            peak_px = ride_state["peak_price"]
            pullback_from_peak = snapshot.pullback_ratio(peak_px)
            stagnation_elapsed = now - ride_state["last_peak_time"]

            # Optional Advanced Guard: Fast profit reversal
            if self.config.fast_profit_guard_enabled:
                try:
                    from strategies.spot_rules import check_fast_profit_reversal
                    fp_triggered, fp_gain, fp_drop = check_fast_profit_reversal(
                        current_price=current_price,
                        avg_cost=buy_price,
                        base_tp_pct=gain_threshold * 100.0,
                        mult=self.config.fast_profit_mult,
                        shadow_prices=snapshot.shadow_prices,
                        window_sec=self.config.fast_profit_window_sec,
                        drop_pct=self.config.fast_profit_drop_pct,
                        current_time=now,
                    )
                    if fp_triggered:
                        self.tp_ride_state.pop(symbol, None)
                        return ExitDecision(
                            action=ExitAction.SELL,
                            should_exit=True,
                            reason=f"fast_profit_reversal (gain={fp_gain:.2f}%, micro_drop={fp_drop:.2f}%)",
                            forced=True,
                            metadata={"priority": 1.5, "fp_gain": fp_gain, "fp_drop": fp_drop},
                        )
                except Exception:
                    pass

            # Priority 2: Guaranteed profit floor breached
            if price_increase < profit_floor:
                self.tp_ride_state.pop(symbol, None)
                reason = (
                    f"mdhtp_profit_floor_breached (gain {price_increase*100:.2f}% < "
                    f"floor {profit_floor*100:.2f}%, peak={peak_px:.4f})"
                )
                return ExitDecision(
                    action=ExitAction.SELL,
                    should_exit=True,
                    reason=reason,
                    forced=True,
                    metadata={"priority": 2, "profit_floor": profit_floor, "peak_px": peak_px},
                )

            # Pullback threshold with dynamic anomaly-scaled ratchet (relative to gain_threshold & vol_1h)
            eff_pullback_pct = self.config.tp_pullback_pct
            if self.config.tp_ratchet_enabled:
                eff_pullback_pct, _ = self.compute_tp_anomaly_pullback(
                    symbol=symbol,
                    gain_threshold=gain_threshold,
                    peak_gain=ride_state["peak_gain"],
                    vol_1h=snapshot.vol_1h,
                )

            # Priority 3: Dynamic micro-trailing peak pullback
            if pullback_from_peak >= eff_pullback_pct:
                self.tp_ride_state.pop(symbol, None)
                reason = (
                    f"mdhtp_peak_pullback (peaked at {peak_px:.4f} [+ {ride_state['peak_gain']*100:.2f}%], "
                    f"pulled back {pullback_from_peak*100:.2f}% >= {eff_pullback_pct*100:.2f}%)"
                )
                return ExitDecision(
                    action=ExitAction.SELL,
                    should_exit=True,
                    reason=reason,
                    forced=True,
                    metadata={"priority": 3, "pullback": pullback_from_peak, "eff_pullback": eff_pullback_pct},
                )

            # Priority 4: Stagnation timeout (18 minutes)
            if stagnation_elapsed >= self.config.tp_stagnation_max_sec:
                self.tp_ride_state.pop(symbol, None)
                reason = (
                    f"mdhtp_stagnation_timeout (no new peak for {stagnation_elapsed:.0f}s >= "
                    f"{self.config.tp_stagnation_max_sec:.0f}s, current_gain={price_increase*100:.2f}%)"
                )
                return ExitDecision(
                    action=ExitAction.SELL,
                    should_exit=True,
                    reason=reason,
                    forced=True,
                    metadata={"priority": 4, "stagnation": stagnation_elapsed},
                )

            # Priority 5: Trend reversal (trend turned down)
            if not snapshot.trend_up:
                self.tp_ride_state.pop(symbol, None)
                reason = (
                    f"mdhtp_trend_reversal (trend turned down after peak {peak_px:.4f} "
                    f"[+ {ride_state['peak_gain']*100:.2f}%], current={current_price:.4f})"
                )
                return ExitDecision(
                    action=ExitAction.SELL,
                    should_exit=True,
                    reason=reason,
                    forced=True,
                    metadata={"priority": 5, "peak_px": peak_px},
                )

            # Position is healthy and actively riding trend
            reason = (
                f"mdhtp_trend_riding (peak={peak_px:.4f} [+ {ride_state['peak_gain']*100:.2f}%], "
                f"pullback={pullback_from_peak*100:.2f}%, stagnation={stagnation_elapsed:.0f}s)"
            )
            return ExitDecision(
                action=ExitAction.HOLD,
                should_exit=False,
                reason=reason,
                forced=False,
                metadata={"priority": 5, "state": "riding", "peak_px": peak_px, "pullback": pullback_from_peak},
            )

    # -------------------------------------------------------------------------
    # Re-Entry Mirror Evaluator (MHDRE)
    # -------------------------------------------------------------------------
    def evaluate_reentry(
        self,
        snapshot: ReentrySnapshot,
        custom_drop_pct: Optional[float] = None,
    ) -> ReentryDecision:
        """Evaluate market re-entry following position closure (Mirror of Exit).

        Symmetry:
          - Post-STOP: Mirror of SL Bounce Watch. Tracks trough (low), requires confirmed
            recovery bounce above trough, prevents re-entry if bounce rolls over (ratchet),
            and requires stronger support in confirmed bear downtrends.
          - Post-TP / Sale: Mirror of TP Trailing. In bull trends, tracks peak and awaits
            pullback (avoids buying top). In chop/range, enforces discount drop below sale.
          - Safety: Decays restrictions once TTL expires.
        """
        symbol = snapshot.symbol
        current_price = snapshot.current_price
        last_sell = snapshot.last_sell_price
        last_kind = snapshot.last_exit_kind or "SELL"
        now = snapshot.current_time_s if snapshot.current_time_s is not None else time.time()

        if current_price <= 0:
            return ReentryDecision(ReentryAction.WAIT, False, "invalid_current_price")

        # Case 0: No prior sale reference to constrain re-entry
        if not last_sell or last_sell <= 0:
            self.reentry_state.pop(symbol, None)
            return ReentryDecision(ReentryAction.ENTER, True, "no_last_sale_barrier")

        # Case 1: TTL barrier expiration
        if self.config.reentry_ttl_hours > 0 and snapshot.last_sell_time_s:
            elapsed_sec = now - snapshot.last_sell_time_s
            ttl_sec = self.config.reentry_ttl_hours * 3600.0
            if elapsed_sec >= ttl_sec:
                self._pop_reentry_state(symbol)
                return ReentryDecision(
                    ReentryAction.ENTER,
                    True,
                    f"reentry_ttl_expired (elapsed={elapsed_sec/3600.0:.2f}h >= {self.config.reentry_ttl_hours:.2f}h)",
                    metadata={"elapsed_hours": elapsed_sec / 3600.0},
                )

        existing_state = self.reentry_state.get(symbol)
        if existing_state and snapshot.last_sell_time_s:
            st_sell_time = existing_state.get("sell_time")
            st_started = existing_state.get("started_at")
            if (
                (st_sell_time is not None and float(st_sell_time) != float(snapshot.last_sell_time_s))
                or (st_sell_time is None and st_started is not None and float(st_started) < float(snapshot.last_sell_time_s))
            ):
                self.reentry_state.pop(symbol, None)

        state = self.reentry_state.setdefault(symbol, {
            "trough": snapshot.post_sell_trough or min(last_sell, current_price),
            "peak": snapshot.post_sell_peak or max(last_sell, current_price),
            "peak_bounce": current_price,
            "started_at": now,
            "sell_time": snapshot.last_sell_time_s,
        })
        if snapshot.last_sell_time_s and state.get("sell_time") is None:
            state["sell_time"] = snapshot.last_sell_time_s

        if current_price < state["trough"]:
            state["trough"] = current_price
            state["peak_bounce"] = current_price
        else:
            state["trough"] = min(state["trough"], current_price)
            state["peak_bounce"] = max(state.get("peak_bounce", current_price), current_price)
        state["peak"] = max(state["peak"], current_price)
        trough = state["trough"]
        peak = state["peak"]

        # Case 2: Post-STOP Re-entry (Bounce Trough Mirror)
        if last_kind == "STOP" and self.config.reentry_sl_bounce_pct > 0:
            if (
                self.config.reentry_post_stop_cooldown_sec > 0
                and snapshot.last_sell_time_s
                and now > 0
                and (now - snapshot.last_sell_time_s) < self.config.reentry_post_stop_cooldown_sec
            ):
                elapsed_s = now - snapshot.last_sell_time_s
                return ReentryDecision(
                    ReentryAction.WAIT,
                    False,
                    f"post_stop_stabilization_cooldown (elapsed={elapsed_s:.0f}s < {self.config.reentry_post_stop_cooldown_sec:.0f}s)",
                    metadata={"trough": trough, "elapsed_sec": elapsed_s},
                )

            bounce_ratio = snapshot.bounce_ratio(trough)
            bounce_required = self.config.reentry_sl_bounce_pct

            # Bear Gate: If trend is actively down, demand extra recovery confirmation
            if snapshot.trend_down and self.config.reentry_bear_gate_enabled:
                if self.config.reentry_bear_bounce_pct > 0:
                    bounce_required = max(bounce_required, self.config.reentry_bear_bounce_pct)
                else:
                    return ReentryDecision(
                        ReentryAction.WAIT,
                        False,
                        f"post_stop_reentry_blocked_by_bear_trend (price={current_price:.4f}, trough={trough:.4f})",
                        metadata={"trough": trough, "bounce_ratio": bounce_ratio, "trend_down": True},
                    )

            if not snapshot.has_bounced(trough, bounce_required, self.config.reentry_tolerance_pct):
                state["peak_bounce"] = max(state.get("peak_bounce", current_price), current_price)
                target_px = trough * (1.0 + bounce_required)
                return ReentryDecision(
                    ReentryAction.WAIT,
                    False,
                    f"post_stop_bounce_waiting (price={current_price:.4f} < {target_px:.4f}, bounce={bounce_ratio*100:.2f}% < {bounce_required*100:.2f}%)",
                    metadata={"trough": trough, "bounce_ratio": bounce_ratio, "target_price": target_px},
                )

            # Ratchet Rollover Check: Track peak since bounce
            state["peak_bounce"] = max(state.get("peak_bounce", current_price), current_price)
            peak_bounce = state["peak_bounce"]
            if self.config.reentry_ratchet_enabled and peak_bounce > current_price:
                pullback_from_bounce = (peak_bounce - current_price) / peak_bounce
                allowed_rollover = bounce_required * 0.50
                if pullback_from_bounce > allowed_rollover:
                    return ReentryDecision(
                        ReentryAction.WAIT,
                        False,
                        f"post_stop_bounce_rollover (peak_bounce={peak_bounce:.4f}, drop={pullback_from_bounce*100:.2f}% > {allowed_rollover*100:.2f}%)",
                        metadata={"peak_bounce": peak_bounce, "pullback": pullback_from_bounce},
                    )

            self._pop_reentry_state(symbol)
            return ReentryDecision(
                ReentryAction.ENTER,
                True,
                f"post_stop_bounce_confirmed (price={current_price:.4f} >= trough={trough:.4f} + {bounce_required*100:.2f}%)",
                metadata={"trough": trough, "bounce_ratio": bounce_ratio},
            )

        # Case 3: Post-TP / Normal Sale Re-entry (Pullback in Bull vs Discount Drop in Range)
        is_bull_reentry = (
            self.config.reentry_hybrid_enabled
            and (snapshot.trend_up or self.config.reentry_peak_relative)
        )
        if is_bull_reentry:
            pullback_required = self.config.reentry_pullback_pct
            pullback_ratio = snapshot.pullback_ratio(peak)
            if not snapshot.has_pulled_back(peak, pullback_required, self.config.reentry_tolerance_pct):
                target_px = peak * (1.0 - pullback_required)
                return ReentryDecision(
                    ReentryAction.WAIT,
                    False,
                    f"bull_reentry_pullback_waiting (price={current_price:.4f} > {target_px:.4f}, pullback={pullback_ratio*100:.2f}% < {pullback_required*100:.2f}%)",
                    metadata={"peak": peak, "pullback_ratio": pullback_ratio, "target_price": target_px},
                )
            self._pop_reentry_state(symbol)
            return ReentryDecision(
                ReentryAction.ENTER,
                True,
                f"bull_reentry_pullback_confirmed (price={current_price:.4f} <= peak={peak:.4f} - {pullback_required*100:.2f}%)",
                metadata={"peak": peak, "pullback_ratio": pullback_ratio},
            )
        else:
            drop_required = custom_drop_pct if custom_drop_pct is not None else self.config.reentry_drop_pct
            if drop_required > 0:
                if not snapshot.has_dropped(last_sell, drop_required, self.config.reentry_tolerance_pct):
                    discount_target = last_sell * (1.0 - drop_required)
                    return ReentryDecision(
                        ReentryAction.WAIT,
                        False,
                        f"range_reentry_drop_waiting (price={current_price:.4f} > {discount_target:.4f}, drop required={drop_required*100:.2f}%)",
                        metadata={"last_sell": last_sell, "discount_target": discount_target},
                    )

            if not snapshot.trend_up:
                if snapshot.trend_down and self.config.reentry_bear_gate_enabled and self.config.reentry_bear_bounce_pct <= 0:
                    return ReentryDecision(
                        ReentryAction.WAIT,
                        False,
                        f"reentry_blocked_by_bear_trend (price={current_price:.4f})",
                        metadata={"trend_down": True},
                    )

                if self.config.reentry_bear_bounce_pct > 0:
                    bear_bounce_ratio = snapshot.bounce_ratio(trough)
                    if not snapshot.has_bounced(trough, self.config.reentry_bear_bounce_pct, self.config.reentry_tolerance_pct):
                        target_bear = trough * (1.0 + self.config.reentry_bear_bounce_pct)
                        return ReentryDecision(
                            ReentryAction.WAIT,
                            False,
                            f"bear_reentry_trough_bounce_waiting (price={current_price:.4f} < {target_bear:.4f}, trough={trough:.4f})",
                            metadata={"trough": trough, "bear_bounce_ratio": bear_bounce_ratio},
                        )


            self._pop_reentry_state(symbol)
            return ReentryDecision(
                ReentryAction.ENTER,
                True,
                f"range_reentry_confirmed (price={current_price:.4f} <= last_sell={last_sell:.4f} - {drop_required*100:.2f}%)",
                metadata={"last_sell": last_sell, "trough": trough},
            )

    # -------------------------------------------------------------------------
    # Unified Evaluator
    # -------------------------------------------------------------------------
    def evaluate(
        self,
        snapshot: PositionSnapshot,
        sl_threshold: Optional[float] = None,
        tp_threshold: Optional[float] = None,
    ) -> ExitDecision:
        """Unified entrypoint evaluating position exits in strict mirror sequence."""
        if not snapshot.is_valid_price:
            return ExitDecision(ExitAction.HOLD, False, "invalid_prices")

        # Check Hard TP (Spike Exit) first
        if self.config.hard_tp_enabled and snapshot.in_tp_territory(self.config.hard_tp_pct, self.config.are_close_tolerance_pct):
            tp_spike = self.evaluate_take_profit(snapshot, custom_threshold=tp_threshold)
            if tp_spike.action == ExitAction.PARTIAL_SELL:
                return tp_spike

        # Check Stop-Loss (unless an active MDHTP profit ride is in progress, in which
        # case evaluate_take_profit must fire mdhtp_profit_floor_breached immediately)
        eff_sl = sl_threshold if sl_threshold is not None else self.config.lost_threshold
        in_sl = (snapshot.symbol not in self.tp_ride_state) and (
            snapshot.in_sl_territory(eff_sl, self.config.are_close_tolerance_pct)
            or (snapshot.symbol in self.sl_bounce_state)
        )
        if in_sl:
            sl_decision = self.evaluate_stop_loss(snapshot, custom_threshold=sl_threshold)
            if sl_decision.should_exit:
                self.tp_ride_state.pop(snapshot.symbol, None)
                return sl_decision
            elif sl_decision.action == ExitAction.HOLD and (
                "mhdsl_bounce" in sl_decision.reason or "smart_stop_loss" in sl_decision.reason
            ):
                self.tp_ride_state.pop(snapshot.symbol, None)
                return sl_decision
        else:
            self.sl_bounce_state.pop(snapshot.symbol, None)
            self.reset_sl_episode(snapshot.symbol)

        # Check Take-Profit
        eff_tp = tp_threshold if tp_threshold is not None else self.config.gain_threshold
        in_tp = (
            snapshot.in_tp_territory(eff_tp, self.config.are_close_tolerance_pct)
            or (snapshot.symbol in self.tp_ride_state)
        )
        if in_tp:
            tp_decision = self.evaluate_take_profit(snapshot, custom_threshold=tp_threshold)
            if tp_decision.should_exit:
                self.sl_bounce_state.pop(snapshot.symbol, None)
                self.reset_sl_episode(snapshot.symbol)
                return tp_decision
            elif tp_decision.action == ExitAction.HOLD and "mdhtp" in tp_decision.reason:
                self.sl_bounce_state.pop(snapshot.symbol, None)
                self.reset_sl_episode(snapshot.symbol)
                return tp_decision

        # Neutral corridor: clean up any remaining stale state
        self.sl_bounce_state.pop(snapshot.symbol, None)
        self.reset_sl_episode(snapshot.symbol)
        self.tp_ride_state.pop(snapshot.symbol, None)
        return ExitDecision(ExitAction.HOLD, False, "neutral_corridor")
