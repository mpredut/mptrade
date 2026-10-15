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


@dataclass
class PositionSnapshot:
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
class ReentrySnapshot:
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
    tp_ratchet_enabled: bool = False
    tp_ratchet_base_trail: float = 0.008
    tp_ratchet_min_trail: float = 0.003
    tp_ratchet_k: float = 0.50

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
        self.tp_ride_state: Dict[str, Dict[str, Any]] = {}
        self.reentry_state: Dict[str, Dict[str, Any]] = {}
        self.hard_tp_last: Dict[str, float] = {}

    def clear_state(self, symbol: Optional[str] = None) -> None:
        """Clear tracking state for a single symbol or all symbols."""
        if symbol:
            self.sl_bounce_state.pop(symbol, None)
            self.tp_ride_state.pop(symbol, None)
            self.reentry_state.pop(symbol, None)
            self.hard_tp_last.pop(symbol, None)
        else:
            self.sl_bounce_state.clear()
            self.tp_ride_state.clear()
            self.reentry_state.clear()
            self.hard_tp_last.clear()

    def get_state(self, symbol: str) -> Dict[str, Any]:
        """Return the current active states for a symbol."""
        return {
            "sl_bounce_state": self.sl_bounce_state.get(symbol),
            "tp_ride_state": self.tp_ride_state.get(symbol),
            "reentry_state": self.reentry_state.get(symbol),
            "hard_tp_last": self.hard_tp_last.get(symbol),
        }

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
            else:
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
        """Evaluate Stop-Loss with strict hierarchical priorities.

        Priority 1: Hard Disaster Floor Circuit Breaker
        Priority 2: Bounce Protection (Trend Deferral, Dynamic Ratchet, Structural Breakdown)
        Priority 3: Smart Stop-Loss Controller / Direct Exit
        """
        symbol = snapshot.symbol
        current_price = snapshot.current_price
        buy_price = snapshot.avg_buy_price
        now = snapshot.current_time_s or time.time()
        lost_threshold = custom_threshold if custom_threshold is not None else self.config.lost_threshold

        if buy_price <= 0 or current_price <= 0:
            return ExitDecision(ExitAction.HOLD, False, "invalid_prices")

        price_decrease = (buy_price - current_price) / buy_price

        # Stop-loss boundary check
        in_sl_territory = (
            price_decrease >= lost_threshold
            or are_close(price_decrease, lost_threshold, self.config.are_close_tolerance_pct)
        )

        if not in_sl_territory:
            # Full recovery above stop-loss territory
            self.sl_bounce_state.pop(symbol, None)
            return ExitDecision(ExitAction.HOLD, False, "not_in_sl_territory")

        # Compute dynamic disaster floor
        disaster_buffer, disaster_meta = self.compute_disaster_buffer(
            symbol=symbol,
            lost_threshold=lost_threshold,
            vol_1h=snapshot.vol_1h,
            vol_source=snapshot.vol_source,
        )
        disaster_floor = lost_threshold + disaster_buffer

        # Preserve already activated protection: once armed, disaster floor cannot loosen (retreat)
        bounce_state = self.sl_bounce_state.get(symbol)
        if bounce_state is not None:
            prev_floor = bounce_state.get("disaster_floor")
            if prev_floor is not None and prev_floor < disaster_floor:
                disaster_floor = prev_floor
            else:
                bounce_state["disaster_floor"] = disaster_floor

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
                metadata={"priority": 1, "disaster_floor": disaster_floor, "disaster_meta": disaster_meta},
            )

        # ---------------------------------------------------------------------
        # PRIORITY 2: Bounce Protection (Trend Up)
        # ---------------------------------------------------------------------
        if snapshot.trend_up:
            bounce_state = self.sl_bounce_state.get(symbol)
            if bounce_state is None:
                self.sl_bounce_state[symbol] = {
                    "started_at": now,
                    "entry_loss": price_decrease,
                    "lowest_price": current_price,
                    "peak_bounce_price": current_price,
                    "min_loss": price_decrease,
                    "disaster_floor": disaster_floor,
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
                        metadata={"priority": 2, "sub_reason": "structural_break", "lowest_px": lowest_px},
                    )

                if price_decrease < bounce_state["min_loss"]:
                    bounce_state["min_loss"] = price_decrease

                elapsed = now - bounce_state["started_at"]
                pullback_from_peak = (peak_px - current_price) / peak_px if peak_px > 0 else 0.0

                eff_pullback_pct = self.config.sl_bounce_pullback_pct
                if self.config.sl_bounce_ratchet_enabled and bounce_recovery > self.config.sl_bounce_min_recovery_pct:
                    excess_rec = bounce_recovery - self.config.sl_bounce_min_recovery_pct
                    tightened = self.config.sl_bounce_pullback_pct - (excess_rec * self.config.sl_bounce_ratchet_k)
                    eff_pullback_pct = max(self.config.sl_bounce_min_pullback_pct, tightened)

                # Timeout expiration (~220s)
                if elapsed >= self.config.sl_trend_defer_max_sec:
                    reason = f"mhdsl_bounce_timeout ({elapsed:.0f}s >= {self.config.sl_trend_defer_max_sec:.0f}s without full recovery)"
                    self.sl_bounce_state.pop(symbol, None)
                    return ExitDecision(
                        action=ExitAction.SELL,
                        should_exit=True,
                        reason=reason,
                        forced=True,
                        metadata={"priority": 2, "sub_reason": "timeout", "elapsed": elapsed},
                    )
                # Dynamic micro-trailing bounce rollover
                elif bounce_attempted and pullback_from_peak >= eff_pullback_pct:
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
                        metadata={"priority": 2, "sub_reason": "rollover", "pullback": pullback_from_peak, "eff_pullback": eff_pullback_pct},
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
                        metadata={"priority": 2, "state": "riding_bounce", "elapsed": elapsed, "peak_px": peak_px},
                    )

        # Trend is NOT UP: clear any bounce deferral
        self.sl_bounce_state.pop(symbol, None)

        # ---------------------------------------------------------------------
        # PRIORITY 3: Smart Stop-Loss Controller / Trend Check
        # ---------------------------------------------------------------------
        if self.config.smart_sl_enabled:
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
                        disaster_buffer_pct=disaster_buffer * 100.0,
                    )
                    if not should_exec:
                        return ExitDecision(
                            action=ExitAction.HOLD,
                            should_exit=False,
                            reason=f"smart_stop_loss_deferred: {sl_reason}",
                            forced=False,
                            metadata={"priority": 3, "smart_sl_meta": sl_meta},
                        )
                    else:
                        return ExitDecision(
                            action=ExitAction.SELL,
                            should_exit=True,
                            reason=sl_reason,
                            forced=False,
                            metadata={"priority": 3, "smart_sl_meta": sl_meta},
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
            metadata={"priority": 3, "loss": price_decrease},
        )

    # -------------------------------------------------------------------------
    # Take-Profit: Evaluation Engine (MDHTP)
    # -------------------------------------------------------------------------
    def evaluate_take_profit(
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
        now = snapshot.current_time_s or time.time()
        gain_threshold = custom_threshold if custom_threshold is not None else self.config.gain_threshold

        if buy_price <= 0 or current_price <= 0:
            return ExitDecision(ExitAction.HOLD, False, "invalid_prices")

        price_increase = (current_price - buy_price) / buy_price

        # ---------------------------------------------------------------------
        # PRIORITY 0: Hard TP Spike Exit
        # ---------------------------------------------------------------------
        hard_hit = price_increase >= self.config.hard_tp_pct or are_close(
            price_increase, self.config.hard_tp_pct, self.config.are_close_tolerance_pct
        )
        hard_valid = 0 < self.config.hard_tp_frac <= 1.0 and self.config.hard_tp_pct > 0
        if self.config.hard_tp_enabled and hard_hit and hard_valid:
            last_hard = self.hard_tp_last.get(symbol, 0.0)
            if now - last_hard >= self.config.hard_tp_cooldown_sec:
                reason = (
                    f"hard_tp_spike (+{price_increase*100:.1f}% >= {self.config.hard_tp_pct*100:.0f}%, "
                    f"selling {self.config.hard_tp_frac*100:.0f}%)"
                )
                return ExitDecision(
                    action=ExitAction.PARTIAL_SELL,
                    should_exit=True,
                    reason=reason,
                    forced=True,
                    sell_fraction=self.config.hard_tp_frac,
                    metadata={"priority": 0, "gain": price_increase},
                )

        # ---------------------------------------------------------------------
        # Take-Profit Boundary Check
        # ---------------------------------------------------------------------
        profit_floor = max(gain_threshold * self.config.tp_profit_floor_ratio, self.config.tp_profit_floor_min_pct)

        # Gap down below cost while riding profit is an emergency breach of the profit floor
        if price_increase <= 0:
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
            price_increase > gain_threshold
            or are_close(price_increase, gain_threshold, self.config.are_close_tolerance_pct)
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
            pullback_from_peak = (peak_px - current_price) / peak_px if peak_px > 0 else 0.0
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

            # Pullback threshold (with optional ratchet from spot_rules)
            eff_pullback_pct = self.config.tp_pullback_pct
            if self.config.tp_ratchet_enabled:
                try:
                    from strategies.spot_rules import dynamic_trend_trail_pct
                    eff_pullback_pct = dynamic_trend_trail_pct(
                        peak_gain_pct=ride_state["peak_gain"] * 100.0,
                        base_trail_pct=self.config.tp_ratchet_base_trail * 100.0,
                        min_trail_pct=self.config.tp_ratchet_min_trail * 100.0,
                        ratchet_k=self.config.tp_ratchet_k,
                    ) / 100.0
                except Exception:
                    pass

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
        now = snapshot.current_time_s or time.time()

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
                self.reentry_state.pop(symbol, None)
                return ReentryDecision(
                    ReentryAction.ENTER,
                    True,
                    f"reentry_ttl_expired (elapsed={elapsed_sec/3600.0:.2f}h >= {self.config.reentry_ttl_hours:.2f}h)",
                    metadata={"elapsed_hours": elapsed_sec / 3600.0},
                )

        state = self.reentry_state.setdefault(symbol, {
            "trough": snapshot.post_sell_trough or min(last_sell, current_price),
            "peak": snapshot.post_sell_peak or max(last_sell, current_price),
            "peak_bounce": current_price,
            "started_at": now,
        })

        if current_price < state["trough"]:
            state["trough"] = current_price
            state["peak_bounce"] = current_price
        else:
            state["trough"] = min(state["trough"], current_price)
        state["peak"] = max(state["peak"], current_price)
        trough = state["trough"]
        peak = state["peak"]

        # Case 2: Post-STOP Re-entry (Bounce Trough Mirror)
        if last_kind == "STOP" and self.config.reentry_sl_bounce_pct > 0:
            bounce_ratio = (current_price - trough) / trough if trough > 0 else 0.0
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

            has_bounced = (
                bounce_ratio >= bounce_required
                or are_close(current_price, trough * (1.0 + bounce_required), self.config.reentry_tolerance_pct)
            )

            if not has_bounced:
                state["peak_bounce"] = current_price
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

            self.reentry_state.pop(symbol, None)
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
            pullback_ratio = (peak - current_price) / peak if peak > 0 else 0.0
            has_pulled_back = (
                pullback_ratio >= pullback_required
                or are_close(current_price, peak * (1.0 - pullback_required), self.config.reentry_tolerance_pct)
            )
            if not has_pulled_back:
                target_px = peak * (1.0 - pullback_required)
                return ReentryDecision(
                    ReentryAction.WAIT,
                    False,
                    f"bull_reentry_pullback_waiting (price={current_price:.4f} > {target_px:.4f}, pullback={pullback_ratio*100:.2f}% < {pullback_required*100:.2f}%)",
                    metadata={"peak": peak, "pullback_ratio": pullback_ratio, "target_price": target_px},
                )
            self.reentry_state.pop(symbol, None)
            return ReentryDecision(
                ReentryAction.ENTER,
                True,
                f"bull_reentry_pullback_confirmed (price={current_price:.4f} <= peak={peak:.4f} - {pullback_required*100:.2f}%)",
                metadata={"peak": peak, "pullback_ratio": pullback_ratio},
            )
        else:
            drop_required = custom_drop_pct if custom_drop_pct is not None else self.config.reentry_drop_pct
            if drop_required > 0:
                discount_target = last_sell * (1.0 - drop_required)
                has_dropped = (
                    current_price <= discount_target
                    or are_close(current_price, discount_target, self.config.reentry_tolerance_pct)
                )
                if not has_dropped:
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
                    bear_bounce_ratio = (current_price - trough) / trough if trough > 0 else 0.0
                    if bear_bounce_ratio < self.config.reentry_bear_bounce_pct:
                        target_bear = trough * (1.0 + self.config.reentry_bear_bounce_pct)
                        return ReentryDecision(
                            ReentryAction.WAIT,
                            False,
                            f"bear_reentry_trough_bounce_waiting (price={current_price:.4f} < {target_bear:.4f}, trough={trough:.4f})",
                            metadata={"trough": trough, "bear_bounce_ratio": bear_bounce_ratio},
                        )


            self.reentry_state.pop(symbol, None)
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
        if snapshot.avg_buy_price <= 0 or snapshot.current_price <= 0:
            return ExitDecision(ExitAction.HOLD, False, "invalid_prices")

        price_increase = (snapshot.current_price - snapshot.avg_buy_price) / snapshot.avg_buy_price
        price_decrease = (snapshot.avg_buy_price - snapshot.current_price) / snapshot.avg_buy_price

        # Check Hard TP (Spike Exit) first
        if self.config.hard_tp_enabled and price_increase >= self.config.hard_tp_pct:
            tp_spike = self.evaluate_take_profit(snapshot, custom_threshold=tp_threshold)
            if tp_spike.action == ExitAction.PARTIAL_SELL:
                return tp_spike

        # Check Stop-Loss
        eff_sl = sl_threshold if sl_threshold is not None else self.config.lost_threshold
        in_sl = (
            price_decrease >= eff_sl
            or are_close(price_decrease, eff_sl, self.config.are_close_tolerance_pct)
        )
        if in_sl:
            sl_decision = self.evaluate_stop_loss(snapshot, custom_threshold=sl_threshold)
            if sl_decision.should_exit:
                self.tp_ride_state.pop(snapshot.symbol, None)
                return sl_decision
            elif sl_decision.action == ExitAction.HOLD and "mhdsl_bounce" in sl_decision.reason:
                self.tp_ride_state.pop(snapshot.symbol, None)
                return sl_decision
        else:
            self.sl_bounce_state.pop(snapshot.symbol, None)

        # Check Take-Profit
        eff_tp = tp_threshold if tp_threshold is not None else self.config.gain_threshold
        in_tp = (
            price_increase >= eff_tp
            or are_close(price_increase, eff_tp, self.config.are_close_tolerance_pct)
            or (snapshot.symbol in self.tp_ride_state)
        )
        if in_tp:
            tp_decision = self.evaluate_take_profit(snapshot, custom_threshold=tp_threshold)
            if tp_decision.should_exit:
                self.sl_bounce_state.pop(snapshot.symbol, None)
                return tp_decision
            elif tp_decision.action == ExitAction.HOLD and "mdhtp" in tp_decision.reason:
                self.sl_bounce_state.pop(snapshot.symbol, None)
                return tp_decision

        # Neutral corridor: clean up any remaining stale state
        self.sl_bounce_state.pop(snapshot.symbol, None)
        self.tp_ride_state.pop(snapshot.symbol, None)
        return ExitDecision(ExitAction.HOLD, False, "neutral_corridor")
