"""Alternative.me Crypto Fear & Greed Index collector with caching and trend metrics."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from intelligence.macro.daemon_ssot import CachedHttpJsonCollector
from intelligence.macro_sentiment_ssot import (
    FearGreedSnapshot,
    fetch_public_json,
    get_fear_greed_endpoint_url,
    resolve_now_ts,
    safe_float,
)
from intelligence.external.collectors.base_collector import (
    TelemetryCacheCollectorInterface,
    TelemetryCollectorFactory,
)
from providers.base import extract_timestamp_s

logger = logging.getLogger("intelligence.sentiment.fear_greed")


class FearGreedCollector(CachedHttpJsonCollector, TelemetryCacheCollectorInterface):
    """Fetches and caches the Alternative.me Crypto Fear & Greed Index."""

    cls_name = "FearGreedCollector"
    COLLECTOR_ID = "sentiment_fear_greed"
    _COLLECTOR_LABEL = "Fear & Greed Index"
    _LOGGER = logger

    def __init__(
        self,
        cache_ttl_sec: float = 900.0,
        endpoint_url: Optional[str] = None,
        *,
        start_sync: Optional[bool] = None,
    ) -> None:
        super().__init__(cache_ttl_sec, 900.0, endpoint_url, get_fear_greed_endpoint_url, start_sync=start_sync)

    @staticmethod
    def parse_payload(payload: Dict[str, Any], *, now_ts: Optional[float] = None) -> Optional[FearGreedSnapshot]:
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
        ts = extract_timestamp_s(current) or resolve_now_ts(now_ts)
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
        return self.parse_payload(data, now_ts=now)


TelemetryCollectorFactory.register(FearGreedCollector.COLLECTOR_ID, FearGreedCollector)
