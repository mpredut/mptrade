"""Alternative.me Crypto Fear & Greed Index collector with caching and trend metrics."""
from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any, Dict, Optional, Tuple

from intelligence.macro_sentiment_ssot import (
    CachedHttpJsonCollector,
    fetch_public_json,
    get_fear_greed_endpoint_url,
    is_cache_fresh,
    resolve_now_ts,
    safe_float,
    safe_nonneg_ts,
)
from providers.base import extract_timestamp_s

logger = logging.getLogger("intelligence.sentiment.fear_greed")


@dataclass(frozen=True)
class FearGreedSnapshot:
    """Snapshot of current market sentiment derived from the Fear & Greed Index."""

    value: int                           # 0 to 100
    classification: str                  # "Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"
    timestamp: float                     # Unix epoch timestamp of calculation
    historical_values: Tuple[int, ...]   # Past daily values (up to 14 days, newest to oldest)
    trend_7d_change: int                 # Net change over last 7 available days (positive = increasing greed)
    is_extreme_fear: bool                # True if value <= 25
    is_extreme_greed: bool               # True if value >= 75


class FearGreedCollector(CachedHttpJsonCollector):
    """Fetches and caches the Alternative.me Crypto Fear & Greed Index."""

    _COLLECTOR_LABEL = "Fear & Greed Index"
    _LOGGER = logger

    def __init__(
        self,
        cache_ttl_sec: float = 900.0,
        endpoint_url: Optional[str] = None,
    ) -> None:
        super().__init__(cache_ttl_sec, 900.0, endpoint_url, get_fear_greed_endpoint_url)

    @staticmethod
    def parse_payload(payload: Dict[str, Any]) -> Optional[FearGreedSnapshot]:
        """Parse raw JSON response payload into a FearGreedSnapshot."""
        data = payload.get("data") if isinstance(payload, dict) else None
        if not data or not isinstance(data, list) or not isinstance(data[0], dict):
            return None

        current = data[0]
        val_f = safe_float(current.get("value"), default=None)
        if val_f is None:
            return None
        val = max(0, min(100, int(round(val_f))))
        raw_cls = current.get("value_classification")
        classification = raw_cls.strip() if isinstance(raw_cls, str) and raw_cls.strip() else "Neutral"
        ts = extract_timestamp_s(current) or time.time()
        history = [
            max(0, min(100, int(round(v_f))))
            for item in data
            if isinstance(item, dict) and (v_f := safe_float(item.get("value"), default=None)) is not None
        ]
        trend_7d = (history[0] - history[6]) if len(history) >= 7 else ((history[0] - history[-1]) if len(history) > 1 else 0)

        return FearGreedSnapshot(
            value=val,
            classification=classification,
            timestamp=ts,
            historical_values=tuple(history),
            trend_7d_change=trend_7d,
            is_extreme_fear=val <= 25,
            is_extreme_greed=val >= 75,
        )

    def _parse_fetched_payload(self, data: Any, now: float) -> Optional[FearGreedSnapshot]:
        return self.parse_payload(data)
