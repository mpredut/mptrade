"""Whale positioning and Open Interest divergence guard.

Guards against buying false breakout rallies fueled only by short-covering (falling OI)
rather than genuine whale capital accumulation (rising OI).
"""

from __future__ import annotations

import math
from typing import Any, Optional

from intelligence.internal.guards.guard_decision import GuardDecision, BrakeAction
from intelligence.external.collectors.whale_positioning import (
    WhalePositioningCollector,
    WhalePositioningSnapshot,
)


def _safe_float(val: Any, default: float) -> float:
    if val is None or isinstance(val, bool):
        return default
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


class WhaleDivergenceGuard:
    """Blocks or defers orders when whale positioning or OI flow contradicts the move."""

    def __init__(
        self,
        collector: Optional[WhalePositioningCollector] = None,
        block_short_covering_fakeout: bool = True,
        block_aggressive_shorting: bool = True,
        block_liquidation_cascade: bool = True,
        severe_cascade_oi_pct: float = -5.0,
        min_taker_ratio: float = 0.75,
        taker_downscale_threshold: float = 0.90,
        taker_downscaled_scale: float = 0.50,
    ) -> None:
        self.collector = collector
        self.block_short_covering_fakeout = bool(block_short_covering_fakeout)
        self.block_aggressive_shorting = bool(block_aggressive_shorting)
        self.block_liquidation_cascade = bool(block_liquidation_cascade)
        self.severe_cascade_oi_pct = _safe_float(severe_cascade_oi_pct, -5.0)
        min_tr = _safe_float(min_taker_ratio, 0.75)
        self.min_taker_ratio = min_tr if min_tr > 0.0 else 0.75
        down_tr = _safe_float(taker_downscale_threshold, 0.90)
        self.taker_downscale_threshold = down_tr if down_tr > 0.0 else 0.90
        scale_f = _safe_float(taker_downscaled_scale, 0.50)
        self.taker_downscaled_scale = scale_f if 0.0 < scale_f <= 1.0 else 0.50

    def check(
        self,
        symbol: str,
        side: str,
        snapshot: Optional[WhalePositioningSnapshot] = None,
        now: Optional[float] = None,
    ) -> GuardDecision:
        """Evaluate if whale positioning or OI divergence conflicts with the order."""
        side_norm = str(side or "").strip().upper()
        if side_norm not in ("BUY", "SELL"):
            return GuardDecision.allow(
                guard_name="WhaleDivergenceGuard",
                reason="unsupported_side_allowed",
            )
        if snapshot is None and self.collector is not None:
            try:
                snapshot = self.collector.fetch(symbol, now=now) if now is not None else self.collector.fetch(symbol)
            except TypeError:
                snapshot = self.collector.fetch(symbol)

        if snapshot is None:
            return GuardDecision.allow(
                guard_name="WhaleDivergenceGuard",
                reason="no_whale_snapshot_data",
            )

        regime = str(getattr(snapshot, "divergence_regime", None) or "neutral")
        oi_1h = _safe_float(getattr(snapshot, "open_interest_1h_change_pct", None), 0.0)
        taker_ratio = _safe_float(getattr(snapshot, "taker_buy_sell_ratio", None), 1.0)
        top_long_pct = _safe_float(getattr(snapshot, "top_traders_long_pct", None), 0.5)

        # 1. Buying into a short-covering fakeout (price up, but OI falling: no new whale money and passive taker flow)
        if side_norm == "BUY" and self.block_short_covering_fakeout and regime == "short_covering":
            return GuardDecision.defer(
                guard_name="WhaleDivergenceGuard",
                reason=f"short_covering_fakeout (OI_1h={oi_1h:+.1f}%, no whale accumulation)",
                open_interest_change=oi_1h,
                divergence_regime=regime,
            )

        # 2. Buying into aggressive whale short expansion (price down, but OI expanding: heavy short momentum)
        if side_norm == "BUY" and self.block_aggressive_shorting and regime in ("aggressive_shorting", "short_building"):
            return GuardDecision.defer(
                guard_name="WhaleDivergenceGuard",
                reason=f"aggressive_whale_shorting (OI_1h={oi_1h:+.1f}% expanding down)",
                open_interest_change=oi_1h,
                divergence_regime=regime,
            )

        # 3. Buying into active severe liquidation cascade (falling price, OI crashing <= severe_cascade_oi_pct, weak taker buy)
        if (
            side_norm == "BUY"
            and self.block_liquidation_cascade
            and regime == "long_liquidation"
            and oi_1h <= self.severe_cascade_oi_pct
            and taker_ratio < 0.85
        ):
            return GuardDecision.defer(
                guard_name="WhaleDivergenceGuard",
                reason=f"liquidation_cascade_active (OI_1h={oi_1h:+.1f}%, taker_ratio={taker_ratio:.2f})",
                open_interest_change=oi_1h,
                divergence_regime=regime,
                taker_ratio=taker_ratio,
            )

        # 4. Severe taker selling panic (< min_taker_ratio, heavy market dumping)
        if side_norm == "BUY" and taker_ratio < self.min_taker_ratio:
            return GuardDecision.defer(
                guard_name="WhaleDivergenceGuard",
                reason=f"severe_taker_selling_dominated (taker_ratio={taker_ratio:.2f} < {self.min_taker_ratio:.2f})",
                taker_ratio=taker_ratio,
                divergence_regime=regime,
            )

        # 4b. Moderate taker selling pressure (< taker_downscale_threshold: downscale buy order)
        if (
            side_norm == "BUY"
            and self.taker_downscale_threshold > self.min_taker_ratio
            and taker_ratio < self.taker_downscale_threshold
        ):
            return GuardDecision.downscale(
                guard_name="WhaleDivergenceGuard",
                scale=self.taker_downscaled_scale,
                reason=f"moderate_taker_selling_pressure (taker_ratio={taker_ratio:.2f} < {self.taker_downscale_threshold:.2f})",
                taker_ratio=taker_ratio,
                divergence_regime=regime,
            )

        # 5. Selling into massive whale accumulation, active short squeeze, or long-liquidation capitulation
        if side_norm == "SELL":
            if (
                self.block_liquidation_cascade
                and regime == "long_liquidation"
                and oi_1h <= self.severe_cascade_oi_pct
            ):
                return GuardDecision.defer(
                    guard_name="WhaleDivergenceGuard",
                    reason=f"selling_into_long_liquidation_capitulation (OI_1h={oi_1h:+.1f}%, taker_ratio={taker_ratio:.2f})",
                    open_interest_change=oi_1h,
                    divergence_regime=regime,
                    taker_ratio=taker_ratio,
                )
            if (
                (regime == "accumulation" and top_long_pct >= 0.70)
                or (regime in ("short_squeeze", "short_squeeze_primed") and top_long_pct >= 0.50)
            ):
                return GuardDecision.defer(
                    guard_name="WhaleDivergenceGuard",
                    reason=f"whale_squeeze_or_accumulation_active (regime={regime}, whales {top_long_pct*100:.1f}% long, taker_ratio={taker_ratio:.2f})",
                    open_interest_change=oi_1h,
                    divergence_regime=regime,
                    top_long_pct=top_long_pct,
                )
            max_taker_ratio = (1.0 / self.min_taker_ratio) if self.min_taker_ratio > 0 else float("inf")
            if taker_ratio > max_taker_ratio:
                return GuardDecision.defer(
                    guard_name="WhaleDivergenceGuard",
                    reason=f"severe_taker_buying_dominated (taker_ratio={taker_ratio:.2f} > {max_taker_ratio:.2f})",
                    taker_ratio=taker_ratio,
                    divergence_regime=regime,
                )
            max_downscale_ratio = (
                (1.0 / self.taker_downscale_threshold)
                if self.taker_downscale_threshold > self.min_taker_ratio > 0
                else float("inf")
            )
            if taker_ratio > max_downscale_ratio:
                return GuardDecision.downscale(
                    guard_name="WhaleDivergenceGuard",
                    scale=self.taker_downscaled_scale,
                    reason=f"moderate_taker_buying_pressure (taker_ratio={taker_ratio:.2f} > {max_downscale_ratio:.2f})",
                    taker_ratio=taker_ratio,
                    divergence_regime=regime,
                )

        return GuardDecision.allow(
            guard_name="WhaleDivergenceGuard",
            reason=f"whale_flow_consistent ({regime})",
            divergence_regime=regime,
            top_long_pct=top_long_pct,
        )
