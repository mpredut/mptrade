"""Market breadth capitulation bounce and blowoff exhaustion trigger."""
from __future__ import annotations

import logging
import math
import time
from typing import Optional, Tuple

from intelligence.internal.triggers.trigger_event import TriggerAction, TriggerEvent, TriggerSide
from intelligence.sentiment.collectors.market_breadth_collector import MarketBreadthSnapshot

logger = logging.getLogger("intelligence.sentiment.market_breadth_trigger")


def _safe_float(val, default: float) -> float:
    if val is None or isinstance(val, bool):
        return default
    try:
        f = float(val)
    except (TypeError, ValueError, OverflowError):
        return default
    return f if math.isfinite(f) else default


class MarketBreadthTrigger:
    """Emits directional triggers based on cross-asset market participation extremes.

    - ENTRY BUY: Triggered on capitulation washouts (advance_ratio <= 0.15, median_change <= -4.0%) when relative strength appears.
    - EXIT SELL: Triggered on euphoric blowoffs (advance_ratio >= 0.88, median_change >= 6.0%) to trim into overheating rallies.
    """

    def __init__(
        self,
        washout_advance_threshold: float = 0.15,
        washout_median_threshold: float = -4.0,
        blowoff_advance_threshold: float = 0.88,
        blowoff_median_threshold: float = 6.0,
    ) -> None:
        wat = _safe_float(washout_advance_threshold, 0.15)
        self.washout_advance_threshold = wat if (0.0 <= wat <= 1.0) else 0.15
        self.washout_median_threshold = _safe_float(washout_median_threshold, -4.0)
        bat = _safe_float(blowoff_advance_threshold, 0.88)
        self.blowoff_advance_threshold = bat if (0.0 <= bat <= 1.0) else 0.88
        self.blowoff_median_threshold = _safe_float(blowoff_median_threshold, 6.0)
        if self.washout_advance_threshold > self.blowoff_advance_threshold:
            self.washout_advance_threshold, self.blowoff_advance_threshold = (
                self.blowoff_advance_threshold,
                self.washout_advance_threshold,
            )
        if self.washout_median_threshold > self.blowoff_median_threshold:
            self.washout_median_threshold, self.blowoff_median_threshold = (
                self.blowoff_median_threshold,
                self.washout_median_threshold,
            )

    def evaluate(
        self,
        symbol: str,
        price: float,
        snapshot: Optional[MarketBreadthSnapshot],
        *,
        asset_24h_change_pct: Optional[float] = None,
        ts: Optional[float] = None,
    ) -> Tuple[Optional[str], Optional[TriggerEvent]]:
        """Evaluate market breadth snapshot and generate actionable triggers."""
        if snapshot is None or price is None or isinstance(price, bool):
            return None, None
        def _snap_attr(key: str, default=None):
            return snapshot.get(key, default) if isinstance(snapshot, dict) else getattr(snapshot, key, default)

        raw_adv = _snap_attr("advance_ratio", None)
        raw_med = _snap_attr("median_change_pct", None)
        raw_regime = _snap_attr("regime", None)
        regime = str(raw_regime).strip() if (isinstance(raw_regime, str) and raw_regime.strip()) else "NEUTRAL"
        if raw_adv is None or raw_med is None or isinstance(raw_adv, bool) or isinstance(raw_med, bool):
            return None, None
        try:
            price_f = float(price)
            adv_r = float(raw_adv)
            med_c = float(raw_med)
        except (TypeError, ValueError):
            return None, None
        if not (math.isfinite(price_f) and price_f > 0.0 and math.isfinite(adv_r) and math.isfinite(med_c)):
            return None, None
        price = price_f

        now = time.time()
        if ts is not None and not isinstance(ts, bool):
            try:
                tf = float(ts)
                if math.isfinite(tf) and tf > 0.0:
                    now = tf
            except (TypeError, ValueError):
                pass

        # 1. Capitulation Washout Bounce Entry
        if (
            adv_r <= self.washout_advance_threshold
            and med_c <= self.washout_median_threshold
        ):
            # If asset is outperforming the bleeding median (relative strength leader)
            relative_strength = False
            if asset_24h_change_pct is not None and not isinstance(asset_24h_change_pct, bool):
                try:
                    a24 = float(asset_24h_change_pct)
                    if math.isfinite(a24) and a24 > med_c:
                        relative_strength = True
                except (TypeError, ValueError):
                    pass

            strength = round(min(0.95, 0.70 + (1.0 - adv_r) * 0.25), 2)
            event = TriggerEvent(
                action=TriggerAction.ENTRY,
                side=TriggerSide.BUY,
                source="market_breadth_washout",
                symbol=symbol,
                price=price,
                strength=strength,
                ts=now,
                reason=f"market_washout_capitulation_bounce (adv_ratio={adv_r:.2f}, median={med_c:.1f}%)",
                metadata={
                    "advance_ratio": adv_r,
                    "median_change_pct": med_c,
                    "relative_strength": relative_strength,
                    "regime": regime,
                },
            )
            return "BREADTH_WASHOUT_BUY", event

        # 2. Overheated Blowoff Distribution Exit
        if (
            adv_r >= self.blowoff_advance_threshold
            and med_c >= self.blowoff_median_threshold
        ):
            strength = round(min(0.95, 0.70 + adv_r * 0.25), 2)
            event = TriggerEvent(
                action=TriggerAction.EXIT,
                side=TriggerSide.SELL,
                source="market_breadth_blowoff",
                symbol=symbol,
                price=price,
                strength=strength,
                ts=now,
                reason=f"market_blowoff_overheated_exhaustion (adv_ratio={adv_r:.2f}, median={med_c:.1f}%)",
                metadata={
                    "advance_ratio": adv_r,
                    "median_change_pct": med_c,
                    "regime": regime,
                },
            )
            return "BREADTH_BLOWOFF_SELL", event

        return None, None
