"""Order book depth imbalance and whale limit walls collector.

Analyzes the top 1-2% of the bid/ask order book:
- Depth Imbalance Ratio: Bid Liquidity / (Bid Liquidity + Ask Liquidity)
- Whale Walls: Detects unusually large single limit orders placed on the book.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OrderbookSnapshot:
        return cls(
            symbol=str(data.get("symbol", "")),
            mid_price=float(data.get("mid_price", 0.0)),
            bid_depth_usd=float(data.get("bid_depth_usd", 0.0)),
            ask_depth_usd=float(data.get("ask_depth_usd", 0.0)),
            imbalance_ratio=float(data.get("imbalance_ratio", 0.5)),
            largest_bid_wall_usd=float(data.get("largest_bid_wall_usd", 0.0)),
            largest_bid_wall_price=float(data.get("largest_bid_wall_price", 0.0)),
            largest_ask_wall_usd=float(data.get("largest_ask_wall_usd", 0.0)),
            largest_ask_wall_price=float(data.get("largest_ask_wall_price", 0.0)),
            ts=float(data.get("ts", 0.0)),
        )


class OrderbookDepthCollector:
    """Collects and analyzes Binance Spot order book depth."""

    def __init__(
        self,
        cache_ttl_sec: float = 5.0,
        depth_band_pct: float = 1.5,     # Analyze depth within 1.5% of mid-price
        whale_wall_usd: float = 500_000.0,
        cache_dir: Optional[str] = "cachedb",
    ) -> None:
        self.cache_ttl_sec = cache_ttl_sec
        self.depth_band_pct = depth_band_pct
        self.whale_wall_usd = whale_wall_usd
        self.cache_dir = cache_dir
        self._cache: Dict[str, tuple[float, OrderbookSnapshot]] = {}

    def _disk_path(self, symbol: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, f"orderbook_depth_{symbol.upper()}.json")

    def _load_from_disk(self, symbol: str, now: float) -> Optional[OrderbookSnapshot]:
        p = self._disk_path(symbol)
        if not p or not os.path.exists(p):
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            snapshot = OrderbookSnapshot.from_dict(data)
            if (now - snapshot.ts) < self.cache_ttl_sec:
                self._cache[symbol.upper()] = (snapshot.ts, snapshot)
                return snapshot
        except Exception as e:
            logger.debug("Failed loading orderbook snapshot from %s: %s", p, e)
        return None

    def _save_to_disk(self, snapshot: OrderbookSnapshot) -> None:
        p = self._disk_path(snapshot.symbol)
        if not p:
            return
        try:
            from state_io import atomic_write_json
            atomic_write_json(p, snapshot.to_dict(), indent=2)
        except Exception as e:
            logger.debug("Failed writing orderbook snapshot to %s: %s", p, e)

    def fetch(self, symbol: str, allow_network: bool = True) -> Optional[OrderbookSnapshot]:
        """Fetch spot depth snapshot from Binance public API."""
        symbol = symbol.upper()
        now = time.time()
        cached = self._cache.get(symbol)
        if cached and (now - cached[0]) < self.cache_ttl_sec:
            return cached[1]

        disk_snap = self._load_from_disk(symbol, now)
        if disk_snap is not None:
            return disk_snap

        if not allow_network:
            return None

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
            self._save_to_disk(snapshot)
            return snapshot
        except Exception as e:
            logger.warning("Could not fetch orderbook depth for %s: %s", symbol, e)
            if cached:
                return cached[1]
            p = self._disk_path(symbol)
            if p and os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    return OrderbookSnapshot.from_dict(data)
                except Exception:
                    pass
            return None

    def record_snapshot(self, snapshot: OrderbookSnapshot, save_to_disk: bool = False) -> None:
        """Inject snapshot directly (useful for tests and synthetic feeds)."""
        self._cache[snapshot.symbol.upper()] = (snapshot.ts, snapshot)
        if save_to_disk:
            self._save_to_disk(snapshot)
