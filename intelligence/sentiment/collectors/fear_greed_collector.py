"""Alternative.me Crypto Fear & Greed Index collector with caching and trend metrics."""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
import time
from typing import Any, Dict, Optional, Tuple
import urllib.request

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
    """Fetches and caches the Alternative.me Crypto Fear & Greed Index.

    The index aggregates volatility, market momentum/volume, social media, surveys,
    dominance, and trends. Updates approximately once per day.
    """

    def __init__(
        self,
        cache_ttl_sec: float = 900.0,
        endpoint_url: str = "https://api.alternative.me/fng/?limit=14",
    ) -> None:
        ttl = 900.0
        if cache_ttl_sec is not None and not isinstance(cache_ttl_sec, bool):
            try:
                tf = float(cache_ttl_sec)
                if math.isfinite(tf) and tf >= 0.0:
                    ttl = tf
            except (TypeError, ValueError):
                pass
        self.cache_ttl_sec = ttl
        self.endpoint_url = endpoint_url
        self._cached_snapshot: Optional[FearGreedSnapshot] = None
        self._last_fetch_ts: float = 0.0

    @staticmethod
    def parse_payload(payload: Dict[str, Any]) -> Optional[FearGreedSnapshot]:
        """Parse raw JSON response payload into a FearGreedSnapshot."""
        if not isinstance(payload, dict):
            return None
        data = payload.get("data")
        if not data or not isinstance(data, list):
            return None

        current = data[0]
        if not isinstance(current, dict):
            return None
        raw_val = current.get("value")
        raw_ts = current.get("timestamp")
        if raw_val is None or isinstance(raw_val, bool):
            return None
        try:
            val_f = float(raw_val)
            if not math.isfinite(val_f):
                return None
            val = max(0, min(100, int(round(val_f))))
        except (ValueError, TypeError):
            return None
        raw_cls = current.get("value_classification")
        classification = raw_cls.strip() if isinstance(raw_cls, str) and raw_cls.strip() else "Neutral"
        ts = time.time()
        if raw_ts is not None and not isinstance(raw_ts, bool):
            try:
                ts_f = float(raw_ts)
                if math.isfinite(ts_f) and ts_f > 0:
                    ts = ts_f
            except (ValueError, TypeError):
                pass

        history: list[int] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            item_val = item.get("value")
            if item_val is None or isinstance(item_val, bool):
                continue
            try:
                v_f = float(item_val)
                if math.isfinite(v_f):
                    history.append(max(0, min(100, int(round(v_f)))))
            except (ValueError, TypeError):
                continue

        # Trend over 7 days (history[0] is newest, history[6] or last is ~7 days ago)
        trend_7d = 0
        if len(history) >= 7:
            trend_7d = history[0] - history[6]
        elif len(history) > 1:
            trend_7d = history[0] - history[-1]

        return FearGreedSnapshot(
            value=val,
            classification=classification,
            timestamp=ts,
            historical_values=tuple(history),
            trend_7d_change=trend_7d,
            is_extreme_fear=val <= 25,
            is_extreme_greed=val >= 75,
        )

    def fetch(self, force_refresh: bool = False) -> Optional[FearGreedSnapshot]:
        """Fetch the latest Fear & Greed Index snapshot, respecting cache TTL."""
        now = time.time()
        if not force_refresh and self._cached_snapshot and 0.0 <= (now - self._last_fetch_ts) < self.cache_ttl_sec:
            return self._cached_snapshot

        try:
            req = urllib.request.Request(
                self.endpoint_url,
                headers={"User-Agent": "mptrade-sentiment/1.0"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                raw_bytes = resp.read()
                data = json.loads(raw_bytes.decode("utf-8"))

            snapshot = self.parse_payload(data)
            if snapshot:
                self._cached_snapshot = snapshot
                self._last_fetch_ts = now
                return snapshot
        except Exception as e:
            logger.warning("Failed to fetch Fear & Greed Index: %s. Using cached fallback if available.", e)

        return self._cached_snapshot
