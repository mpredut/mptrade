"""Order book depth imbalance and whale limit walls collector.

Analyzes the top 1-2% of the bid/ask order book:
- Depth Imbalance Ratio: Bid Liquidity / (Bid Liquidity + Ask Liquidity)
- Whale Walls: Detects unusually large single limit orders placed on the book.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("intelligence.external.orderbook_depth")


@dataclass(frozen=True)
class OrderbookSnapshot:
    """Normalized order book liquidity snapshot."""

    symbol: str
    mid_price: float
    bid_depth_usd: float
    ask_depth_usd: float
    imbalance_ratio: float           # 0.0 (all asks) to 1.0 (all bids). 0.5 = balanced
    largest_bid_wall_usd: float
    largest_bid_wall_price: float
    largest_ask_wall_usd: float
    largest_ask_wall_price: float
    ts: float


class OrderbookDepthCollector:
    """Collects and analyzes Binance Spot order book depth."""

    def __init__(
        self,
        cache_ttl_sec: float = 5.0,
        depth_band_pct: float = 1.5,     # Analyze depth within 1.5% of mid-price
        whale_wall_usd: float = 500_000.0,
    ) -> None:
        self.cache_ttl_sec = cache_ttl_sec
        self.depth_band_pct = depth_band_pct
        self.whale_wall_usd = whale_wall_usd
        self._cache: Dict[str, tuple[float, OrderbookSnapshot]] = {}

    def fetch(self, symbol: str) -> Optional[OrderbookSnapshot]:
        """Fetch spot depth snapshot from Binance public API."""
        symbol = symbol.upper()
        now = time.time()
        cached = self._cache.get(symbol)
        if cached and (now - cached[0]) < self.cache_ttl_sec:
            return cached[1]

        lookup_sym = symbol.replace("USDC", "USDT") if "USDC" in symbol and symbol != "USDCUSDT" else symbol
        try:
            url = f"https://api.binance.com/api/v3/depth?symbol={lookup_sym}&limit=50"
            req = urllib.request.Request(url, headers={"User-Agent": "mptrade-orderbook/1.0"})
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            bids: List[Tuple[float, float]] = [(float(p), float(q)) for p, q in data.get("bids", [])]
            asks: List[Tuple[float, float]] = [(float(p), float(q)) for p, q in data.get("asks", [])]

            if not bids or not asks:
                return None

            best_bid = bids[0][0]
            best_ask = asks[0][0]
            mid = (best_bid + best_ask) / 2.0

            bid_lower_bound = mid * (1.0 - self.depth_band_pct / 100.0)
            ask_upper_bound = mid * (1.0 + self.depth_band_pct / 100.0)

            bid_depth_usd = 0.0
            max_bid_usd = 0.0
            max_bid_px = 0.0
            for p, q in bids:
                if p < bid_lower_bound:
                    break
                val = p * q
                bid_depth_usd += val
                if val > max_bid_usd:
                    max_bid_usd = val
                    max_bid_px = p

            ask_depth_usd = 0.0
            max_ask_usd = 0.0
            max_ask_px = 0.0
            for p, q in asks:
                if p > ask_upper_bound:
                    break
                val = p * q
                ask_depth_usd += val
                if val > max_ask_usd:
                    max_ask_usd = val
                    max_ask_px = p

            total_depth = bid_depth_usd + ask_depth_usd
            imbalance = bid_depth_usd / total_depth if total_depth > 0 else 0.5

            snapshot = OrderbookSnapshot(
                symbol=symbol,
                mid_price=mid,
                bid_depth_usd=bid_depth_usd,
                ask_depth_usd=ask_depth_usd,
                imbalance_ratio=round(imbalance, 3),
                largest_bid_wall_usd=round(max_bid_usd, 2),
                largest_bid_wall_price=max_bid_px,
                largest_ask_wall_usd=round(max_ask_usd, 2),
                largest_ask_wall_price=max_ask_px,
                ts=now,
            )
            self._cache[symbol] = (now, snapshot)
            return snapshot
        except Exception as e:
            logger.warning("Could not fetch orderbook depth for %s: %s", symbol, e)
            if cached:
                return cached[1]
            return None

    def record_snapshot(self, snapshot: OrderbookSnapshot) -> None:
        """Inject snapshot directly (useful for tests and synthetic feeds)."""
        self._cache[snapshot.symbol.upper()] = (snapshot.ts, snapshot)
