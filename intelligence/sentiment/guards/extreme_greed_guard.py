"""Extreme greed anti-FOMO guard protecting against top-buying in euphoric conditions."""
from __future__ import annotations

import logging
import math
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
        def _safe_thresh(val, default: int) -> int:
            if val is None or isinstance(val, bool):
                return default
            try:
                f = float(val)
            except (TypeError, ValueError):
                return default
            if not math.isfinite(f) or not (0.0 <= f <= 100.0):
                return default
            return int(round(f))

        self.downscale_threshold = _safe_thresh(downscale_threshold, 80)
        self.hard_veto_threshold = _safe_thresh(hard_veto_threshold, 90)
        if self.downscale_threshold > self.hard_veto_threshold:
            self.downscale_threshold = self.hard_veto_threshold
        if downscale_factor is None or isinstance(downscale_factor, bool):
            self.downscale_factor = 0.35
        else:
            try:
                df = float(downscale_factor)
                self.downscale_factor = df if (math.isfinite(df) and 0.0 < df <= 1.0) else 0.35
            except (TypeError, ValueError):
                self.downscale_factor = 0.35

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
        if not isinstance(side, str) or side.strip().upper() not in ("BUY", "B", "BID", "LONG"):
            return GuardDecision.allow("ExtremeGreedGuard", "sell_permitted_in_greed")

        raw_val = snapshot.get("value") if isinstance(snapshot, dict) else getattr(snapshot, "value", None)
        if raw_val is None or isinstance(raw_val, bool):
            return GuardDecision.allow("ExtremeGreedGuard", "no_sentiment_data")
        try:
            val_f = float(raw_val)
            if not math.isfinite(val_f):
                return GuardDecision.allow("ExtremeGreedGuard", "no_sentiment_data")
            val = int(round(val_f))
        except (TypeError, ValueError):
            return GuardDecision.allow("ExtremeGreedGuard", "no_sentiment_data")
        raw_cls = snapshot.get("classification") if isinstance(snapshot, dict) else getattr(snapshot, "classification", None)
        cls_label = str(raw_cls or "Unknown")

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
