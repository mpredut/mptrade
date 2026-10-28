"""Extreme greed anti-FOMO guard protecting against top-buying in euphoric conditions."""
from __future__ import annotations

import logging
import math
from typing import Optional

from intelligence.internal.guards.guard_decision import GuardDecision
from intelligence.macro_sentiment_ssot import clean_scale, normalize_order_side, obj_attr, safe_float
from intelligence.sentiment.collectors.fear_greed_collector import FearGreedSnapshot

logger = logging.getLogger("intelligence.sentiment.extreme_greed_guard")


def _safe_thresh(val, default: int) -> int:
    f = safe_float(val, -1.0)
    return int(round(f)) if 0.0 <= f <= 100.0 else default


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
        self.downscale_threshold = _safe_thresh(downscale_threshold, 80)
        self.hard_veto_threshold = _safe_thresh(hard_veto_threshold, 90)
        if self.downscale_threshold > self.hard_veto_threshold:
            self.downscale_threshold = self.hard_veto_threshold
        self.downscale_factor = clean_scale(downscale_factor, 0.35, min_positive=True)

    def check(
        self,
        symbol: str,
        side: str,
        snapshot: Optional[FearGreedSnapshot],
    ) -> GuardDecision:
        """Evaluate order against extreme greed conditions."""
        if snapshot is None:
            return GuardDecision.allow("ExtremeGreedGuard", "no_sentiment_data")

        if normalize_order_side(side) != "BUY":
            return GuardDecision.allow("ExtremeGreedGuard", "sell_permitted_in_greed")

        val_f = safe_float(obj_attr(snapshot, "value", None), float("nan"))
        if not math.isfinite(val_f):
            return GuardDecision.allow("ExtremeGreedGuard", "no_sentiment_data")
        val = int(round(val_f))
        cls_label = str(obj_attr(snapshot, "classification", None) or "Unknown")

        if val >= self.hard_veto_threshold:
            return GuardDecision.veto(
                "ExtremeGreedGuard",
                f"veto_extreme_market_euphoria (F&G={val} >= {self.hard_veto_threshold})",
                fear_greed_value=val,
                classification=cls_label,
            )

        if val >= self.downscale_threshold:
            return GuardDecision.downscale(
                "ExtremeGreedGuard",
                scale=self.downscale_factor,
                reason=f"downscale_market_greed (F&G={val} >= {self.downscale_threshold})",
                fear_greed_value=val,
                classification=cls_label,
            )

        return GuardDecision.allow("ExtremeGreedGuard", f"greed_within_tolerance (F&G={val})")
