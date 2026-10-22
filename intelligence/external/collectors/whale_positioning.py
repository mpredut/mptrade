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

from intelligence.external.collectors.orderbook_depth import (
    _resolve_binance_lookup_symbol,
    _symbol_variants,
)

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
    divergence_regime: str             # "accumulation", "short_covering", "aggressive_shorting", "long_liquidation", "neutral", "short_squeeze", "short_squeeze_primed", "long_unwinding"
    ts: float
    funding_rate: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WhalePositioningSnapshot:
        fr = data.get("funding_rate")
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
            funding_rate=float(fr) if fr is not None else None,
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

    def _effective_cache_dir(self) -> Optional[str]:
        if not self.cache_dir:
            return None
        if self.cache_dir == "cachedb":
            try:
                from cachepaths import get_cachedb_dir
                return get_cachedb_dir()
            except Exception:
                return os.environ.get("MPTRADE_CACHEDB_DIR", "cachedb")
        return self.cache_dir

    def _disk_path(self, symbol: str) -> Optional[str]:
        cache_dir = self._effective_cache_dir()
        if not cache_dir:
            return None
        safe_sym = symbol.upper().strip().replace("/", "").replace("-", "")
        return os.path.join(cache_dir, f"whale_snapshot_{safe_sym}.json")

    def _load_from_disk(self, symbol: str, now: float, *, ignore_ttl: bool = False) -> Optional[WhalePositioningSnapshot]:
        max_age = max(self.cache_ttl_sec, self.disk_ttl_sec)
        for cand in _symbol_variants(symbol):
            p = self._disk_path(cand)
            if not p or not os.path.exists(p):
                continue
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                snapshot = WhalePositioningSnapshot.from_dict(data)
                if (now - snapshot.ts) < max_age:
                    self._cache[symbol.upper()] = (snapshot.ts, snapshot)
                    return snapshot
                if ignore_ttl:
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

    def _resolve_price_1h_change_pct(self, symbol: str) -> float:
        """Resolve 1-hour price change percentage from cached price history (0 ms local)."""
        cache_dir = self._effective_cache_dir()
        if not cache_dir:
            return 0.0
        try:
            sym_u = symbol.upper()
            variants = _symbol_variants(sym_u)

            for sym_key in variants:
                p = os.path.join(cache_dir, f"cache_24_price_{sym_key}.json")
                if not os.path.exists(p):
                    continue
                with open(p, "r", encoding="utf-8") as f:
                    cdata = json.load(f)
                items_dict = cdata.get("items")
                pts = None
                if isinstance(items_dict, dict):
                    pts = items_dict.get(sym_key) or items_dict.get(sym_u)
                    if not pts and items_dict:
                        pts = next(iter(items_dict.values()))
                elif isinstance(items_dict, list):
                    pts = items_dict
                if pts and len(pts) >= 2:
                    now_ms = pts[-1][0]
                    target_ms = now_ms - 3600.0 * 1000.0
                    latest_p = float(pts[-1][1])
                    ref_p = float(pts[0][1])
                    for pt in reversed(pts):
                        if pt[0] <= target_ms:
                            ref_p = float(pt[1])
                            break
                    if ref_p > 0:
                        return (latest_p - ref_p) / ref_p * 100.0
        except Exception as e:
            logger.debug("Failed calculating 1h price change from cache for %s: %s", symbol, e)
        return 0.0

    def _resolve_cached_funding_rate(self, symbol: str, now: Optional[float] = None) -> Optional[float]:
        """Lookup cached funding rate from derivatives telemetry if available and fresh."""
        cache_dir = self._effective_cache_dir()
        if not cache_dir:
            return None
        try:
            now_ts = now if now is not None else time.time()
            max_age = max(self.cache_ttl_sec, self.disk_ttl_sec)
            variants = _symbol_variants(symbol)
            for s in variants:
                p = os.path.join(cache_dir, f"derivatives_telemetry_{s}.json")
                if os.path.exists(p):
                    with open(p, "r", encoding="utf-8") as f:
                        d = json.load(f)
                    ts_val = float(d.get("ts") or 0.0)
                    if ts_val > 0 and (now_ts - ts_val) >= max_age:
                        continue
                    fr = d.get("funding_rate")
                    if fr is not None:
                        return float(fr)
        except Exception as e:
            logger.debug("Failed reading funding rate from cache for %s: %s", symbol, e)
        return None

    @staticmethod
    def classify_regime(
        price_1h_change_pct: float,
        oi_change_pct: float,
        taker_ratio: float,
        funding_rate: Optional[float] = None,
        prev_regime: Optional[str] = None,
        top_long_pct: Optional[float] = None,
    ) -> str:
        """Classify derivatives OI-Price divergence regime with short squeeze & liquidation distinction.

        Correlates:
        1. Price 1h change (trend direction)
        2. Open Interest 1h change (capital expansion vs contraction)
        3. Taker Buy/Sell ratio (aggressive market execution dominance)
        4. Funding Rate (crowded short vs long pressure)
        """
        # 1. Price rising (> +0.2%)
        if price_1h_change_pct > 0.2:
            if oi_change_pct > 0.3:
                # True accumulation vs Trapped shorts / Short Squeeze Primed
                is_trapped_shorts = (
                    taker_ratio >= 1.15
                    and (
                        (funding_rate is not None and funding_rate < -0.00005)
                        or prev_regime in ("aggressive_shorting", "short_squeeze_primed")
                        or (top_long_pct is not None and top_long_pct < 0.45)
                    )
                )
                return "short_squeeze_primed" if is_trapped_shorts else "accumulation"
            elif oi_change_pct < -0.3:
                # Distinguish active short squeeze (forced buying cascade) from passive short covering (fakeout)
                if taker_ratio >= 1.25 or (oi_change_pct <= -2.5 and taker_ratio >= 1.10):
                    return "short_squeeze"
                return "short_covering"
            else:
                # Flat OI with rising price: check for trapped shorts pressure
                if taker_ratio >= 1.20 and (
                    prev_regime in ("aggressive_shorting", "short_squeeze_primed")
                    or (funding_rate is not None and funding_rate < 0.0)
                ):
                    return "short_squeeze_primed"
                return "neutral"

        # 2. Price falling (< -0.2%)
        elif price_1h_change_pct < -0.2:
            if oi_change_pct > 0.3:
                # Aggressive short expansion vs mild short building
                return "aggressive_shorting" if (taker_ratio < 0.85 or oi_change_pct > 1.0) else "short_building"
            elif oi_change_pct < -0.3:
                # Distinguish forced liquidation cascade from orderly voluntary position unwinding
                is_severe_oi_drop = oi_change_pct <= -3.0
                is_aggressive_sell = taker_ratio < 0.85
                is_violent_dump = price_1h_change_pct < -1.5 and taker_ratio < 0.80
                if (is_severe_oi_drop and is_aggressive_sell) or is_violent_dump or (taker_ratio < 0.80 and oi_change_pct <= -1.5):
                    return "long_liquidation"
                return "long_unwinding"
            else:
                if taker_ratio < 0.80:
                    return "aggressive_shorting"
                return "neutral"

        # 3. Flat price (abs <= 0.2%): consolidation / range absorption
        else:
            if oi_change_pct > 0.8 and taker_ratio < 0.85:
                return "aggressive_shorting"
            elif oi_change_pct > 0.8 and taker_ratio > 1.15:
                if funding_rate is not None and funding_rate < -0.00005:
                    return "short_squeeze_primed"
                return "accumulation"
            return "neutral"

    def fetch(self, symbol: str, price_1h_change_pct: float = 0.0, allow_network: bool = True) -> Optional[WhalePositioningSnapshot]:
        """Fetch latest whale positioning metrics from public Binance Futures endpoints."""
        symbol = symbol.upper()
        now = time.time()
        cached = self._cache.get(symbol)
        if cached:
            if isinstance(cached, tuple) and len(cached) >= 2:
                if (now - cached[0]) < self.cache_ttl_sec:
                    return cached[1]
            elif isinstance(cached, WhalePositioningSnapshot):
                if (now - cached.ts) < self.cache_ttl_sec:
                    return cached

        # Check disk cache before network
        disk_snap = self._load_from_disk(symbol, now)
        if disk_snap is not None:
            return disk_snap

        if not allow_network:
            return None

        if abs(price_1h_change_pct) < 1e-6:
            price_1h_change_pct = self._resolve_price_1h_change_pct(symbol)

        lookup_sym = _resolve_binance_lookup_symbol(symbol)
        try:
            # 1. Top Trader Long/Short Position Ratio
            url_top = f"https://fapi.binance.com/futures/data/topLongShortPositionRatio?symbol={lookup_sym}&period=1h&limit=2"
            req_top = urllib.request.Request(url_top, headers={"User-Agent": "mptrade-whale/1.0"})
            with urllib.request.urlopen(req_top, timeout=5) as resp:
                data_top = json.loads(resp.read().decode("utf-8"))
            if not isinstance(data_top, list) or not data_top:
                raise ValueError(f"Empty topLongShortPositionRatio response for {lookup_sym}")

            top_ratio = float(data_top[-1].get("longShortRatio") or 1.0)
            top_long_pct = float(data_top[-1].get("longAccount") or 0.5)

            # 2. Taker Buy/Sell Volume Ratio
            url_taker = f"https://fapi.binance.com/futures/data/takerlongshortRatio?symbol={lookup_sym}&period=1h&limit=2"
            req_taker = urllib.request.Request(url_taker, headers={"User-Agent": "mptrade-whale/1.0"})
            with urllib.request.urlopen(req_taker, timeout=5) as resp:
                data_taker = json.loads(resp.read().decode("utf-8"))
            if not isinstance(data_taker, list) or not data_taker:
                raise ValueError(f"Empty takerlongshortRatio response for {lookup_sym}")

            taker_ratio = float(data_taker[-1].get("buySellRatio") or 1.0)
            taker_buy = float(data_taker[-1].get("buyVol") or 0.0)
            taker_sell = float(data_taker[-1].get("sellVol") or 0.0)

            # 3. Open Interest History (compute 1h change)
            url_oi = f"https://fapi.binance.com/futures/data/openInterestHist?symbol={lookup_sym}&period=1h&limit=3"
            req_oi = urllib.request.Request(url_oi, headers={"User-Agent": "mptrade-whale/1.0"})
            with urllib.request.urlopen(req_oi, timeout=5) as resp:
                data_oi = json.loads(resp.read().decode("utf-8"))
            if not isinstance(data_oi, list) or not data_oi:
                raise ValueError(f"Empty openInterestHist response for {lookup_sym}")

            latest_oi_usd = float(data_oi[-1].get("sumOpenInterestValue") or 0.0)
            prev_oi_usd = float(data_oi[-2].get("sumOpenInterestValue") or latest_oi_usd) if len(data_oi) >= 2 else latest_oi_usd
            oi_change_pct = ((latest_oi_usd - prev_oi_usd) / prev_oi_usd * 100.0) if prev_oi_usd > 0 else 0.0

            # 4. Classify OI-Price divergence regime
            if cached:
                prev_snap = cached[1] if (isinstance(cached, tuple) and len(cached) >= 2) else cached
            else:
                prev_snap = self._load_from_disk(symbol, now, ignore_ttl=True)
            prev_reg = prev_snap.divergence_regime if prev_snap else None
            funding_rate = self._resolve_cached_funding_rate(symbol, now=now)

            regime = self.classify_regime(
                price_1h_change_pct=price_1h_change_pct,
                oi_change_pct=oi_change_pct,
                taker_ratio=taker_ratio,
                funding_rate=funding_rate,
                prev_regime=prev_reg,
                top_long_pct=top_long_pct,
            )

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
                funding_rate=funding_rate,
            )
            self._cache[symbol] = (now, snapshot)
            self._save_to_disk(snapshot)
            return snapshot
        except Exception as e:
            logger.warning("Could not fetch whale positioning for %s: %s", symbol, e)
            if cached:
                if isinstance(cached, tuple) and len(cached) >= 2:
                    return cached[1]
                if isinstance(cached, WhalePositioningSnapshot):
                    return cached
            # Try stale disk snapshot as last resort
            for cand in _symbol_variants(symbol):
                p = self._disk_path(cand)
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
