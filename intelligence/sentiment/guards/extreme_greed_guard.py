"""Extreme greed anti-FOMO guard protecting against top-buying in euphoric conditions."""
from __future__ import annotations

import logging
from typing import Optional

from intelligence.internal.guards.guard_decision import GuardDecision
from intelligence.sentiment.collectors.fear_greed_collector import FearGreedSnapshot

logger = logging.getLogger("intelligence.sentiment.extreme_greed_guard")


class ExtremeGreedGuard:
    """Brakes or downscales BUY orders when market-wide sentiment reaches extreme euphoria.

    - Hard Veto: When Fear & Greed >= hard_veto_threshold (default 90).
    - Downscale: When Fear & Greed >= downscale_threshold (default 80) reduces order size.
    - SELL orders are always unconstrained (encouraged profit-taking).
    """

    def __init__(
        self,
        downscale_threshold: int = 80,
        hard_veto_threshold: int = 90,
        downscale_factor: float = 0.35,
    ) -> None:
        self.downscale_threshold = downscale_threshold
        self.hard_veto_threshold = hard_veto_threshold
        self.downscale_factor = downscale_factor

    def check(
        self,
        symbol: str,
        side: str,
        snapshot: Optional[FearGreedSnapshot],
    ) -> GuardDecision:
        """Evaluate order against extreme greed conditions."""
        if snapshot is None:
            return GuardDecision.allow("ExtremeGreedGuard", "no_sentiment_data")

        # Greed guard only restricts BUY orders; SELL orders in greed are approved
        if side.upper() != "BUY":
            return GuardDecision.allow("ExtremeGreedGuard", "sell_permitted_in_greed")

        val = snapshot.value

        if val >= self.hard_veto_threshold:
            return GuardDecision.veto(
                "ExtremeGreedGuard",
                f"veto_extreme_market_euphoria (F&G={val} >= {self.hard_veto_threshold})",
                fear_greed_value=val,
                classification=snapshot.classification,
            )

        if val >= self.downscale_threshold:
            return GuardDecision.downscale(
                "ExtremeGreedGuard",
                scale=self.downscale_factor,
                reason=f"downscale_market_greed (F&G={val} >= {self.downscale_threshold})",
                fear_greed_value=val,
                classification=snapshot.classification,
            )

        return GuardDecision.allow("ExtremeGreedGuard", f"greed_within_tolerance (F&G={val})")
