"""Market breadth and cross-asset sentiment dispersion collector."""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
import statistics
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.request

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
        endpoint_url: str = "https://api.binance.com/api/v3/ticker/24hr",
    ) -> None:
        ttl = 60.0
        if cache_ttl_sec is not None and not isinstance(cache_ttl_sec, bool):
            try:
                tf = float(cache_ttl_sec)
                if math.isfinite(tf) and tf >= 0.0:
                    ttl = tf
            except (TypeError, ValueError):
                pass
        mv = 5_000_000.0
        if min_volume_usd is not None and not isinstance(min_volume_usd, bool):
            try:
                mf = float(min_volume_usd)
                if math.isfinite(mf) and mf >= 0.0:
                    mv = mf
            except (TypeError, ValueError):
                pass
        self.cache_ttl_sec = ttl
        self.min_volume_usd = mv
        self.endpoint_url = endpoint_url
        self._cached_snapshot: Optional[MarketBreadthSnapshot] = None
        self._last_fetch_ts: float = 0.0

    @staticmethod
    def parse_tickers(
        tickers: List[Dict[str, Any]],
        min_volume_usd: float = 5_000_000.0,
        now_ts: Optional[float] = None,
    ) -> Optional[MarketBreadthSnapshot]:
        """Parse raw 24hr ticker data and calculate market breadth statistics."""
        if not tickers or not isinstance(tickers, list):
            return None

        now = time.time()
        if now_ts is not None and not isinstance(now_ts, bool):
            try:
                now_f = float(now_ts)
                if math.isfinite(now_f) and now_f > 0.0:
                    now = now_f
            except (ValueError, TypeError):
                pass
        min_vol = 5_000_000.0
        if min_volume_usd is not None and not isinstance(min_volume_usd, bool):
            try:
                mv_f = float(min_volume_usd)
                if math.isfinite(mv_f) and mv_f >= 0.0:
                    min_vol = mv_f
            except (ValueError, TypeError):
                pass
        filtered: List[Tuple[str, float]] = []

        for item in tickers:
            if not isinstance(item, dict):
                continue
            raw_sym = item.get("symbol")
            if not isinstance(raw_sym, str):
                continue
            symbol = raw_sym.strip().upper()
            if not symbol.endswith("USDT"):
                continue
            # Exclude leveraged tokens or stablecoin pairs
            if any(sym_part in symbol for sym_part in ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT", "USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "EURUSDT")):
                continue

            raw_qv = item.get("quoteVolume", 0.0)
            raw_cp = item.get("priceChangePercent", 0.0)
            if isinstance(raw_qv, bool) or isinstance(raw_cp, bool) or raw_qv is None or raw_cp is None:
                continue
            try:
                quote_vol = float(raw_qv)
                change_pct = float(raw_cp)
            except (ValueError, TypeError):
                continue
            if not (math.isfinite(quote_vol) and math.isfinite(change_pct)):
                continue

            if quote_vol >= min_vol:
                filtered.append((symbol, change_pct))

        if len(filtered) < 5:
            return None

        # Sort by performance
        filtered.sort(key=lambda x: x[1])
        changes = [x[1] for x in filtered]

        advancing = sum(1 for c in changes if c > 0)
        declining = sum(1 for c in changes if c < 0)
        total = len(changes)
        advance_ratio = advancing / total if total > 0 else 0.5

        median_change = float(statistics.median(changes))
        mean_change = float(statistics.mean(changes))
        dispersion_std = float(statistics.pstdev(changes)) if len(changes) > 1 else 0.0

        # Classify market regime
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

        top_losers = tuple(filtered[:3])
        top_gainers = tuple(filtered[-3:][::-1])

        return MarketBreadthSnapshot(
            advance_ratio=round(advance_ratio, 4),
            advancing_count=advancing,
            declining_count=declining,
            total_count=total,
            median_change_pct=round(median_change, 3),
            mean_change_pct=round(mean_change, 3),
            dispersion_std=round(dispersion_std, 3),
            regime=regime,
            top_gainers=top_gainers,
            top_losers=top_losers,
            ts=now,
        )

    def fetch(self, force_refresh: bool = False) -> Optional[MarketBreadthSnapshot]:
        """Fetch latest 24hr tickers from public Binance API and return market breadth snapshot."""
        now = time.time()
        if not force_refresh and self._cached_snapshot and 0.0 <= (now - self._last_fetch_ts) < self.cache_ttl_sec:
            return self._cached_snapshot

        try:
            req = urllib.request.Request(
                self.endpoint_url,
                headers={"User-Agent": "mptrade-sentiment/1.0"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            if isinstance(data, list):
                snapshot = self.parse_tickers(data, min_volume_usd=self.min_volume_usd, now_ts=now)
                if snapshot:
                    self._cached_snapshot = snapshot
                    self._last_fetch_ts = now
                    return snapshot
        except Exception as e:
            logger.warning("Failed to fetch market breadth: %s. Using cached fallback if available.", e)

        return self._cached_snapshot
