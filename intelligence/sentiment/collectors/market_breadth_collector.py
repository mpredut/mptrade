"""Market breadth and cross-asset sentiment dispersion collector."""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import statistics
from typing import Any, Dict, List, Optional, Tuple
import urllib.request

from intelligence.macro_sentiment_ssot import (
    fetch_public_json,
    get_breadth_excluded_bases,
    get_breadth_quote_suffixes,
    get_market_breadth_endpoint_url,
    is_cache_fresh,
    resolve_now_ts,
    safe_float,
    safe_nonneg_ts,
)

logger = logging.getLogger("intelligence.sentiment.market_breadth")


@dataclass(frozen=True)
class MarketBreadthSnapshot:
    """Snapshot of cross-market breadth, advance/decline participation, and dispersion."""

    advance_ratio: float                          # Fraction of liquid coins with positive 24h change (0.0 - 1.0)
    advancing_count: int                          # Count of positive 24h pairs
    declining_count: int                          # Count of negative 24h pairs
    total_count: int                              # Total liquid pairs analyzed
    median_change_pct: float                      # Median 24h price change percentage
    mean_change_pct: float                        # Mean 24h price change percentage
    dispersion_std: float                         # Standard deviation of 24h returns (dispersion)
    regime: str                                   # "PANIC_WASHOUT", "EUPHORIC_BLOWOFF", "BULLISH_BREADTH", "BEARISH_BREADTH", "NEUTRAL"
    top_gainers: Tuple[Tuple[str, float], ...]    # (symbol, change_pct) top 3 gainers
    top_losers: Tuple[Tuple[str, float], ...]     # (symbol, change_pct) top 3 losers
    ts: float


class MarketBreadthCollector:
    """Computes real-time market-wide breadth and sentiment dispersion from liquid pairs."""

    def __init__(
        self,
        cache_ttl_sec: float = 60.0,
        min_volume_usd: float = 5_000_000.0,
        endpoint_url: Optional[str] = None,
    ) -> None:
        self.cache_ttl_sec = safe_nonneg_ts(cache_ttl_sec, 60.0)
        self.min_volume_usd = safe_nonneg_ts(min_volume_usd, 5_000_000.0)
        self.endpoint_url = (
            endpoint_url.strip()
            if isinstance(endpoint_url, str) and endpoint_url.strip()
            else get_market_breadth_endpoint_url()
        )
        self._cached_snapshot: Optional[MarketBreadthSnapshot] = None
        self._last_fetch_ts: float = 0.0

    @staticmethod
    def parse_tickers(
        tickers: List[Dict[str, Any]],
        min_volume_usd: float = 5_000_000.0,
        now_ts: Optional[float] = None,
    ) -> Optional[MarketBreadthSnapshot]:
        """Parse raw 24hr ticker data and calculate market breadth statistics."""
        if not tickers or not isinstance(tickers, (list, tuple)):
            return None

        now = resolve_now_ts(now_ts)
        min_vol = safe_nonneg_ts(min_volume_usd, 5_000_000.0)
        quote_suffixes = get_breadth_quote_suffixes()
        excluded_suffixes = tuple(f"{b}{q}" for b in get_breadth_excluded_bases() for q in quote_suffixes)
        filtered: List[Tuple[str, float]] = []

        for item in tickers:
            if not isinstance(item, dict):
                continue
            raw_sym = item.get("symbol")
            if not isinstance(raw_sym, str):
                continue
            symbol = raw_sym.strip().upper()
            if not symbol.endswith(quote_suffixes) or any(sp in symbol for sp in excluded_suffixes):
                continue

            quote_vol = safe_float(item.get("quoteVolume"), default=None)
            change_pct = safe_float(item.get("priceChangePercent"), default=None)
            if quote_vol is not None and change_pct is not None and quote_vol >= min_vol:
                filtered.append((symbol, change_pct))

        if len(filtered) < 5:
            return None

        filtered.sort(key=lambda x: x[1])
        changes = [x[1] for x in filtered]
        advancing = sum(1 for c in changes if c > 0)
        declining = sum(1 for c in changes if c < 0)
        total = len(changes)
        advance_ratio = advancing / total if total > 0 else 0.5

        median_change = float(statistics.median(changes))
        mean_change = float(statistics.fmean(changes))
        dispersion_std = float(statistics.pstdev(changes)) if len(changes) > 1 else 0.0

        if advance_ratio <= 0.20 and median_change <= -4.0:
            regime = "PANIC_WASHOUT"
        elif advance_ratio >= 0.85 and median_change >= 5.0:
            regime = "EUPHORIC_BLOWOFF"
        elif advance_ratio >= 0.60:
            regime = "BULLISH_BREADTH"
        elif advance_ratio <= 0.40:
            regime = "BEARISH_BREADTH"
        else:
            regime = "NEUTRAL"

        return MarketBreadthSnapshot(
            advance_ratio=round(advance_ratio, 4),
            advancing_count=advancing,
            declining_count=declining,
            total_count=total,
            median_change_pct=round(median_change, 3),
            mean_change_pct=round(mean_change, 3),
            dispersion_std=round(dispersion_std, 3),
            regime=regime,
            top_gainers=tuple(filtered[-3:][::-1]),
            top_losers=tuple(filtered[:3]),
            ts=now,
        )

    def fetch(self, force_refresh: bool = False, now_ts: Optional[float] = None) -> Optional[MarketBreadthSnapshot]:
        """Fetch latest 24hr tickers from public Binance API and return market breadth snapshot."""
        now = resolve_now_ts(now_ts)
        if self._cached_snapshot and is_cache_fresh(
            self._last_fetch_ts, now, self.cache_ttl_sec, force_refresh=force_refresh
        ):
            return self._cached_snapshot

        try:
            data = fetch_public_json(self.endpoint_url, timeout=5, user_agent="mptrade-sentiment/1.0")
            if isinstance(data, list):
                snapshot = self.parse_tickers(data, min_volume_usd=self.min_volume_usd, now_ts=now)
                if snapshot:
                    self._cached_snapshot = snapshot
                    self._last_fetch_ts = now
                    return snapshot
        except Exception as e:
            logger.warning("Failed to fetch market breadth: %s. Using cached fallback if available.", e)

        return self._cached_snapshot
