"""Alternative.me Crypto Fear & Greed Index collector with caching and trend metrics."""
from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any, Dict, Optional, Tuple

from intelligence.macro_sentiment_ssot import (
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


class FearGreedCollector:
    """Fetches and caches the Alternative.me Crypto Fear & Greed Index."""

    def __init__(
        self,
        cache_ttl_sec: float = 900.0,
        endpoint_url: Optional[str] = None,
    ) -> None:
        self.cache_ttl_sec = safe_nonneg_ts(cache_ttl_sec, 900.0)
        self.endpoint_url = (endpoint_url or "").strip() or get_fear_greed_endpoint_url()
        self._cached_snapshot: Optional[FearGreedSnapshot] = None
        self._last_fetch_ts: float = 0.0

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

    def fetch(self, force_refresh: bool = False, now_ts: Optional[float] = None) -> Optional[FearGreedSnapshot]:
        """Fetch the latest Fear & Greed Index snapshot, respecting cache TTL."""
        now = resolve_now_ts(now_ts)
        if self._cached_snapshot and is_cache_fresh(self._last_fetch_ts, now, self.cache_ttl_sec, force_refresh=force_refresh):
            return self._cached_snapshot
        try:
            snapshot = self.parse_payload(fetch_public_json(self.endpoint_url, timeout=5, user_agent="mptrade-sentiment/1.0"))
            if snapshot:
                self._cached_snapshot, self._last_fetch_ts = snapshot, now
                return snapshot
        except Exception as e:
            logger.warning("Failed to fetch Fear & Greed Index: %s. Using cached fallback if available.", e)
        return self._cached_snapshot
