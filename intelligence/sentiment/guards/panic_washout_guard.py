"""Panic washout guard protecting against catching falling knives during macro panics."""
from __future__ import annotations

import logging
import math
from typing import Any, Optional

from intelligence.internal.guards.guard_decision import GuardDecision
from intelligence.macro_sentiment_ssot import clean_scale, normalize_order_side, obj_attr, safe_float
from intelligence.sentiment.collectors.fear_greed_collector import FearGreedSnapshot
from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthSnapshot

logger = logging.getLogger("intelligence.sentiment.panic_washout_guard")


def _opt_float(val: Any) -> Optional[float]:
    f = safe_float(val, float("nan"))
    return f if math.isfinite(f) else None


class PanicWashoutGuard:
    """Guards against uncontained macro panics and falling knives.

    During market-wide liquidation storms or accelerating panic drops:
    - Defers or downscales standard BUY orders until selling decelerates.
    - Permits SELL / stop-loss execution without interference.
    """

    def __init__(
        self,
        severe_panic_fng_threshold: int = 12,
        panic_advance_threshold: float = 0.10,
        downscale_factor: float = 0.50,
    ) -> None:
        fng_f = _opt_float(severe_panic_fng_threshold)
        self.severe_panic_fng_threshold = int(round(fng_f)) if (fng_f is not None and 0.0 <= fng_f <= 100.0) else 12
        self.panic_advance_threshold = clean_scale(panic_advance_threshold, 0.10)
        self.downscale_factor = clean_scale(downscale_factor, 0.50, min_positive=True)

    def check(
        self,
        symbol: str,
        side: str,
        *,
        breadth_snapshot: Optional[MarketBreadthSnapshot] = None,
        fear_greed_snapshot: Optional[FearGreedSnapshot] = None,
        asset_24h_change_pct: Optional[float] = None,
    ) -> GuardDecision:
        """Evaluate order against macro panic and market-wide capitulation."""
        if normalize_order_side(side) != "BUY":
            return GuardDecision.allow("PanicWashoutGuard", "sell_permitted_in_panic")

        # 1. Check severe accelerating Fear & Greed free-fall
        if fear_greed_snapshot is not None:
            val_f = _opt_float(obj_attr(fear_greed_snapshot, "value", None))
            raw_history = obj_attr(fear_greed_snapshot, "historical_values", None) or ()
            if val_f is not None and val_f <= self.severe_panic_fng_threshold:
                val = int(round(val_f))
                if isinstance(raw_history, (list, tuple)) and len(raw_history) >= 2:
                    h0 = _opt_float(raw_history[0])
                    h1 = _opt_float(raw_history[1])
                    if h0 is not None and h1 is not None and (h1 - h0) > 8:
                        drop_1d = int(round(h1 - h0))
                        return GuardDecision.defer(
                            "PanicWashoutGuard",
                            f"defer_accelerating_panic_capitulation (F&G={val}, 1d drop={drop_1d})",
                            fear_greed_value=val,
                        )

        # 2. Check Market Breadth Washout
        if breadth_snapshot is not None:
            adv_ratio = _opt_float(obj_attr(breadth_snapshot, "advance_ratio", None))
            med_chg = _opt_float(obj_attr(breadth_snapshot, "median_change_pct", None))
            if (
                adv_ratio is not None
                and med_chg is not None
                and adv_ratio <= self.panic_advance_threshold
                and med_chg <= -5.0
            ):
                chg_24h = _opt_float(asset_24h_change_pct)
                if chg_24h is not None and chg_24h <= -7.0:
                    return GuardDecision.defer(
                        "PanicWashoutGuard",
                        f"defer_market_panic_washout (adv_ratio={adv_ratio:.2f}, asset_change={chg_24h:.1f}%)",
                        advance_ratio=adv_ratio,
                    )
                return GuardDecision.downscale(
                    "PanicWashoutGuard",
                    scale=self.downscale_factor,
                    reason=f"downscale_market_panic_washout (adv_ratio={adv_ratio:.2f})",
                    advance_ratio=adv_ratio,
                )

        return GuardDecision.allow("PanicWashoutGuard", "macro_conditions_stable")
