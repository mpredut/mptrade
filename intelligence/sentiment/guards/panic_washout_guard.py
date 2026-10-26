"""Panic washout guard protecting against catching falling knives during macro panics."""
from __future__ import annotations

import logging
import math
from typing import Any, Optional

from intelligence.internal.guards.guard_decision import GuardDecision
from intelligence.sentiment.collectors.fear_greed_collector import FearGreedSnapshot
from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthSnapshot

logger = logging.getLogger("intelligence.sentiment.panic_washout_guard")


def _safe_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    if val is None or isinstance(val, bool):
        return default
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


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
        self.severe_panic_fng_threshold = severe_panic_fng_threshold
        self.panic_advance_threshold = panic_advance_threshold
        self.downscale_factor = downscale_factor

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
        # Only guard BUY orders; stop-losses or exits are never impeded
        if str(side or "").upper() != "BUY":
            return GuardDecision.allow("PanicWashoutGuard", "sell_permitted_in_panic")

        # 1. Check severe accelerating Fear & Greed free-fall
        if fear_greed_snapshot is not None:
            val_f = _safe_float(getattr(fear_greed_snapshot, "value", None), None)
            raw_history = getattr(fear_greed_snapshot, "historical_values", None) or ()
            if val_f is not None and val_f <= self.severe_panic_fng_threshold:
                val = int(round(val_f))
                if isinstance(raw_history, (list, tuple)) and len(raw_history) >= 2:
                    h0 = _safe_float(raw_history[0], None)
                    h1 = _safe_float(raw_history[1], None)
                    if h0 is not None and h1 is not None and (h1 - h0) > 8:
                        drop_1d = int(round(h1 - h0))
                        return GuardDecision.defer(
                            "PanicWashoutGuard",
                            f"defer_accelerating_panic_capitulation (F&G={val}, 1d drop={drop_1d})",
                            fear_greed_value=val,
                        )

        # 2. Check Market Breadth Washout
        if breadth_snapshot is not None:
            adv_ratio = _safe_float(getattr(breadth_snapshot, "advance_ratio", None), None)
            med_chg = _safe_float(getattr(breadth_snapshot, "median_change_pct", None), None)
            if (
                adv_ratio is not None
                and med_chg is not None
                and adv_ratio <= self.panic_advance_threshold
                and med_chg <= -5.0
            ):
                chg_24h = _safe_float(asset_24h_change_pct, None)
                # If asset is also plunging heavily (> 7% drop), defer entry
                if chg_24h is not None and chg_24h <= -7.0:
                    return GuardDecision.defer(
                        "PanicWashoutGuard",
                        f"defer_market_panic_washout (adv_ratio={adv_ratio:.2f}, asset_change={chg_24h:.1f}%)",
                        advance_ratio=adv_ratio,
                    )
                # Otherwise, allow with reduced risk allocation
                return GuardDecision.downscale(
                    "PanicWashoutGuard",
                    scale=self.downscale_factor,
                    reason=f"downscale_market_panic_washout (adv_ratio={adv_ratio:.2f})",
                    advance_ratio=adv_ratio,
                )

        return GuardDecision.allow("PanicWashoutGuard", "macro_conditions_stable")
