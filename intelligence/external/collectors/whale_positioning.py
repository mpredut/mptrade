"""Binance top-trader whale positioning and Open Interest flow collector.

Fetches public, unauthenticated metrics from Binance Futures:
- topLongShortPositionRatio: Long/short ratio of the top 20% largest balance accounts (whales).
- takerlongshortRatio: Taker buy volume vs taker sell volume (whale aggression).
- openInterestHist: Historical Open Interest flow (USD & coin amounts).
- Evaluates OI-Price divergence regimes (accumulation vs short-covering vs distribution).
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger("intelligence.external.whale_positioning")


@dataclass(frozen=True)
class WhalePositioningSnapshot:
    """Consolidated snapshot of whale positioning and OI flow."""

    symbol: str
    top_traders_long_ratio: float      # e.g. 1.83
    top_traders_long_pct: float        # e.g. 0.647 (64.7% longs)
    taker_buy_sell_ratio: float        # e.g. 1.43 (>1.0 = aggressive taker buying)
    taker_buy_vol_usd: float
    taker_sell_vol_usd: float
    open_interest_usd: float
    open_interest_1h_change_pct: float # e.g. +2.5%
    divergence_regime: str             # "accumulation", "short_covering", "aggressive_shorting", "long_liquidation", "neutral"
    ts: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WhalePositioningSnapshot:
        return cls(
            symbol=str(data.get("symbol", "")),
            top_traders_long_ratio=float(data.get("top_traders_long_ratio", 1.0)),
            top_traders_long_pct=float(data.get("top_traders_long_pct", 0.5)),
            taker_buy_sell_ratio=float(data.get("taker_buy_sell_ratio", 1.0)),
            taker_buy_vol_usd=float(data.get("taker_buy_vol_usd", 0.0)),
            taker_sell_vol_usd=float(data.get("taker_sell_vol_usd", 0.0)),
            open_interest_usd=float(data.get("open_interest_usd", 0.0)),
            open_interest_1h_change_pct=float(data.get("open_interest_1h_change_pct", 0.0)),
            divergence_regime=str(data.get("divergence_regime", "neutral")),
            ts=float(data.get("ts", 0.0)),
        )


class WhalePositioningCollector:
    """Collects and caches Binance Futures top trader and open interest metrics."""

    def __init__(
        self,
        cache_ttl_sec: float = 120.0,
        cache_dir: Optional[str] = "cachedb",
        disk_ttl_sec: float = 300.0,
    ) -> None:
        self.cache_ttl_sec = cache_ttl_sec
        self.cache_dir = cache_dir
        self.disk_ttl_sec = disk_ttl_sec
        self._cache: Dict[str, tuple[float, WhalePositioningSnapshot]] = {}

    def _disk_path(self, symbol: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, f"whale_snapshot_{symbol.upper()}.json")

    def _load_from_disk(self, symbol: str, now: float) -> Optional[WhalePositioningSnapshot]:
        p = self._disk_path(symbol)
        if not p or not os.path.exists(p):
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            snapshot = WhalePositioningSnapshot.from_dict(data)
            max_age = max(self.cache_ttl_sec, self.disk_ttl_sec)
            if (now - snapshot.ts) < max_age:
                self._cache[symbol.upper()] = (snapshot.ts, snapshot)
                return snapshot
        except Exception as e:
            logger.debug("Failed loading whale snapshot from %s: %s", p, e)
        return None

    def _save_to_disk(self, snapshot: WhalePositioningSnapshot) -> None:
        p = self._disk_path(snapshot.symbol)
        if not p:
            return
        try:
            from state_io import atomic_write_json
            atomic_write_json(p, snapshot.to_dict(), indent=2)
        except Exception as e:
            logger.debug("Failed writing whale snapshot to %s: %s", p, e)

    def fetch(self, symbol: str, price_1h_change_pct: float = 0.0, allow_network: bool = True) -> Optional[WhalePositioningSnapshot]:
        """Fetch latest whale positioning metrics from public Binance Futures endpoints."""
        symbol = symbol.upper()
        now = time.time()
        cached = self._cache.get(symbol)
        if cached and (now - cached[0]) < self.cache_ttl_sec:
            return cached[1]

        # Check disk cache before network
        disk_snap = self._load_from_disk(symbol, now)
        if disk_snap is not None:
            return disk_snap

        if not allow_network:
            return None

        lookup_sym = symbol.replace("USDC", "USDT")
        try:
            # 1. Top Trader Long/Short Position Ratio
            url_top = f"https://fapi.binance.com/futures/data/topLongShortPositionRatio?symbol={lookup_sym}&period=1h&limit=2"
            req_top = urllib.request.Request(url_top, headers={"User-Agent": "mptrade-whale/1.0"})
            with urllib.request.urlopen(req_top, timeout=5) as resp:
                data_top = json.loads(resp.read().decode("utf-8"))

            top_ratio = float(data_top[-1].get("longShortRatio") or 1.0)
            top_long_pct = float(data_top[-1].get("longAccount") or 0.5)

            # 2. Taker Buy/Sell Volume Ratio
            url_taker = f"https://fapi.binance.com/futures/data/takerlongshortRatio?symbol={lookup_sym}&period=1h&limit=2"
            req_taker = urllib.request.Request(url_taker, headers={"User-Agent": "mptrade-whale/1.0"})
            with urllib.request.urlopen(req_taker, timeout=5) as resp:
                data_taker = json.loads(resp.read().decode("utf-8"))

            taker_ratio = float(data_taker[-1].get("buySellRatio") or 1.0)
            taker_buy = float(data_taker[-1].get("buyVol") or 0.0)
            taker_sell = float(data_taker[-1].get("sellVol") or 0.0)

            # 3. Open Interest History (compute 1h change)
            url_oi = f"https://fapi.binance.com/futures/data/openInterestHist?symbol={lookup_sym}&period=1h&limit=3"
            req_oi = urllib.request.Request(url_oi, headers={"User-Agent": "mptrade-whale/1.0"})
            with urllib.request.urlopen(req_oi, timeout=5) as resp:
                data_oi = json.loads(resp.read().decode("utf-8"))

            latest_oi_usd = float(data_oi[-1].get("sumOpenInterestValue") or 0.0)
            prev_oi_usd = float(data_oi[-2].get("sumOpenInterestValue") or latest_oi_usd) if len(data_oi) >= 2 else latest_oi_usd
            oi_change_pct = ((latest_oi_usd - prev_oi_usd) / prev_oi_usd * 100.0) if prev_oi_usd > 0 else 0.0

            # 4. Classify OI-Price divergence regime
            # Price rising + OI rising = True accumulation
            # Price rising + OI falling = Short covering (exhaustion)
            # Price falling + OI rising = Aggressive shorting
            # Price falling + OI falling = Long liquidation
            if price_1h_change_pct > 0.5:
                regime = "accumulation" if oi_change_pct > 0.5 else ("short_covering" if oi_change_pct < -0.5 else "neutral")
            elif price_1h_change_pct < -0.5:
                regime = "aggressive_shorting" if oi_change_pct > 0.5 else ("long_liquidation" if oi_change_pct < -0.5 else "neutral")
            else:
                regime = "neutral"

            snapshot = WhalePositioningSnapshot(
                symbol=symbol,
                top_traders_long_ratio=top_ratio,
                top_traders_long_pct=top_long_pct,
                taker_buy_sell_ratio=taker_ratio,
                taker_buy_vol_usd=taker_buy,
                taker_sell_vol_usd=taker_sell,
                open_interest_usd=latest_oi_usd,
                open_interest_1h_change_pct=round(oi_change_pct, 2),
                divergence_regime=regime,
                ts=now,
            )
            self._cache[symbol] = (now, snapshot)
            self._save_to_disk(snapshot)
            return snapshot
        except Exception as e:
            logger.warning("Could not fetch whale positioning for %s: %s", symbol, e)
            if cached:
                return cached[1]
            # Try stale disk snapshot as last resort
            p = self._disk_path(symbol)
            if p and os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    return WhalePositioningSnapshot.from_dict(data)
                except Exception:
                    pass
            return None

    def record_snapshot(self, snapshot: WhalePositioningSnapshot, save_to_disk: bool = False) -> None:
        """Inject snapshot directly (useful for testing and deterministic replays)."""
        self._cache[snapshot.symbol.upper()] = (snapshot.ts, snapshot)
        if save_to_disk:
            self._save_to_disk(snapshot)
