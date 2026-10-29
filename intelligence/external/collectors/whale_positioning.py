"""Binance top-trader whale positioning and Open Interest flow collector.

Fetches public, unauthenticated metrics from Binance Futures:
- topLongShortPositionRatio: Long/short ratio of the top 20% largest balance accounts (whales).
- takerlongshortRatio: Taker buy volume vs taker sell volume (whale aggression).
- openInterestHist: Historical Open Interest flow (USD & coin amounts).
- Evaluates OI-Price divergence regimes (accumulation vs short-covering vs distribution).
"""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from botcore import safe_float, safe_pos_float
from cachepaths import telemetry_cache_file_path
from intelligence.external.collectors.orderbook_depth import (
    _DiskBackedSymbolCollector,
    _clean_dict_float,
    _effective_now_ts,
    _resolve_binance_lookup_symbol,
    _symbol_variants,
    is_cache_fresh,
)
from state_io import read_json_dict

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
        if not isinstance(data, dict):
            data = {}
        fr_val = safe_float(data.get("funding_rate"), default=None)
        raw_sym = data.get("symbol")
        sym_str = raw_sym.strip() if isinstance(raw_sym, str) else ""
        raw_reg = data.get("divergence_regime")
        reg_str = raw_reg.strip() if isinstance(raw_reg, str) and raw_reg.strip() else "neutral"
        return cls(
            symbol=sym_str,
            top_traders_long_ratio=max(0.0, _clean_dict_float(data, "top_traders_long_ratio", 1.0)),
            top_traders_long_pct=min(1.0, max(0.0, _clean_dict_float(data, "top_traders_long_pct", 0.5))),
            taker_buy_sell_ratio=max(0.0, _clean_dict_float(data, "taker_buy_sell_ratio", 1.0)),
            taker_buy_vol_usd=max(0.0, _clean_dict_float(data, "taker_buy_vol_usd", 0.0)),
            taker_sell_vol_usd=max(0.0, _clean_dict_float(data, "taker_sell_vol_usd", 0.0)),
            open_interest_usd=max(0.0, _clean_dict_float(data, "open_interest_usd", 0.0)),
            open_interest_1h_change_pct=_clean_dict_float(data, "open_interest_1h_change_pct", 0.0),
            divergence_regime=reg_str,
            ts=max(0.0, _clean_dict_float(data, "ts", 0.0)),
            funding_rate=fr_val,
        )


class WhalePositioningCollector(_DiskBackedSymbolCollector):
    """Collects and caches Binance Futures top trader and open interest metrics."""

    _DISK_FILE_PREFIX = "whale_snapshot"
    _SNAPSHOT_CLS = WhalePositioningSnapshot

    def __init__(
        self,
        cache_ttl_sec: float = 120.0,
        cache_dir: Optional[str] = "cachedb",
        disk_ttl_sec: float = 300.0,
    ) -> None:
        super().__init__(cache_ttl_sec, 120.0, cache_dir, disk_ttl_sec, 300.0)

    def _resolve_price_1h_change_pct(self, symbol: str) -> float:
        """Resolve 1-hour price change percentage from cached price history (0 ms local)."""
        cache_dir = self._effective_cache_dir()
        if not cache_dir or not isinstance(symbol, str) or not symbol.strip():
            return 0.0
        try:
            from intelligence.sentiment.guards.high_stake_guard import load_symbol_price_series
            series_24h, _ = load_symbol_price_series(cache_dir, symbol.upper(), include_7d=False)
            if len(series_24h) >= 2:
                now_ms = float(series_24h[-1][0])
                target_ms = now_ms - 3_600_000.0
                latest_p = float(series_24h[-1][1])
                ref_p = next(
                    (float(pt[1]) for pt in reversed(series_24h) if float(pt[0]) <= target_ms),
                    float(series_24h[0][1]),
                )
                if ref_p > 0:
                    return (latest_p - ref_p) / ref_p * 100.0
        except Exception as e:
            logger.debug("Failed calculating 1h price change from cache for %s: %s", symbol, e)
        return 0.0

    def _resolve_cached_funding_rate(self, symbol: str, now: Optional[float] = None) -> Optional[float]:
        """Lookup cached funding rate from derivatives telemetry if available and fresh."""
        if not self._effective_cache_dir() or not isinstance(symbol, str) or not symbol.strip():
            return None
        try:
            now_ts = _effective_now_ts(now)
            max_age = max(self.cache_ttl_sec, self.disk_ttl_sec)
            for s in _symbol_variants(symbol):
                p = telemetry_cache_file_path(self.cache_dir, "derivatives_telemetry", s)
                if not p:
                    continue
                try:
                    d = read_json_dict(p, default=None, recover_previous=True)
                    if not isinstance(d, dict):
                        continue
                    if not is_cache_fresh(d.get("ts"), now_ts, max_age):
                        continue
                    fr_val = safe_float(d.get("funding_rate"), default=None)
                    if fr_val is not None:
                        return fr_val
                except Exception:
                    continue
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
        price_1h_change_pct = safe_float(price_1h_change_pct, default=0.0)
        oi_change_pct = safe_float(oi_change_pct, default=0.0)
        taker_ratio = safe_pos_float(taker_ratio, default=1.0) or 1.0
        funding_rate = safe_float(funding_rate, default=None)
        top_long_pct = safe_float(top_long_pct, default=None)
        prev_reg_norm = prev_regime.strip().lower() if isinstance(prev_regime, str) else ""
        # 1. Price rising (> +0.2%)
        if price_1h_change_pct > 0.2:
            if oi_change_pct > 0.3:
                # True accumulation vs Trapped shorts / Short Squeeze Primed
                is_trapped_shorts = (
                    taker_ratio >= 1.15
                    and (
                        (funding_rate is not None and funding_rate < -0.00005)
                        or prev_reg_norm in ("aggressive_shorting", "short_squeeze_primed")
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
                    prev_reg_norm in ("aggressive_shorting", "short_squeeze_primed")
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

    def fetch(
        self,
        symbol: str,
        price_1h_change_pct: float = 0.0,
        allow_network: bool = True,
        now: Optional[float] = None,
    ) -> Optional[WhalePositioningSnapshot]:
        """Fetch latest whale positioning metrics from public Binance Futures endpoints."""
        symbol, now, cached, ready, should_fetch = self._prepare_fetch(symbol, allow_network, now)
        if not should_fetch or symbol is None:
            return ready

        price_1h_change_pct = safe_float(price_1h_change_pct, default=0.0)
        if abs(price_1h_change_pct) < 1e-6:
            price_1h_change_pct = self._resolve_price_1h_change_pct(symbol)

        lookup_sym = _resolve_binance_lookup_symbol(symbol)
        try:
            # 1. Top Trader Long/Short Position Ratio
            url_top = f"https://fapi.binance.com/futures/data/topLongShortPositionRatio?symbol={lookup_sym}&period=1h&limit=2"
            latest_top, _ = self._fetch_latest_json_row(url_top, "topLongShortPositionRatio", lookup_sym)
            top_ratio = _clean_dict_float(latest_top, "longShortRatio", 1.0)
            top_long_pct = _clean_dict_float(latest_top, "longAccount", 0.5)

            # 2. Taker Buy/Sell Volume Ratio
            url_taker = f"https://fapi.binance.com/futures/data/takerlongshortRatio?symbol={lookup_sym}&period=1h&limit=2"
            latest_taker, _ = self._fetch_latest_json_row(url_taker, "takerlongshortRatio", lookup_sym)
            taker_ratio = _clean_dict_float(latest_taker, "buySellRatio", 1.0)
            taker_buy = max(0.0, _clean_dict_float(latest_taker, "buyVol", 0.0))
            taker_sell = max(0.0, _clean_dict_float(latest_taker, "sellVol", 0.0))

            # 3. Open Interest History (compute 1h change)
            url_oi = f"https://fapi.binance.com/futures/data/openInterestHist?symbol={lookup_sym}&period=1h&limit=3"
            latest_oi, data_oi = self._fetch_latest_json_row(url_oi, "openInterestHist", lookup_sym)
            latest_oi_usd = max(0.0, _clean_dict_float(latest_oi, "sumOpenInterestValue", 0.0))
            prev_oi_usd = (
                max(0.0, _clean_dict_float(data_oi[-2], "sumOpenInterestValue", latest_oi_usd))
                if (len(data_oi) >= 2 and isinstance(data_oi[-2], dict))
                else latest_oi_usd
            )
            oi_change_pct = ((latest_oi_usd - prev_oi_usd) / prev_oi_usd * 100.0) if prev_oi_usd > 0 else 0.0

            # 4. Classify OI-Price divergence regime
            if cached:
                prev_snap = cached[1] if (isinstance(cached, tuple) and len(cached) >= 2) else cached
            else:
                prev_snap = self._load_from_disk(symbol, now, ignore_ttl=True)
            prev_reg = (
                (prev_snap.get("divergence_regime") if isinstance(prev_snap, dict) else getattr(prev_snap, "divergence_regime", None))
                if prev_snap else None
            )
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
            return self._commit_snapshot(symbol, now, snapshot)
        except Exception as e:
            logger.warning("Could not fetch whale positioning for %s: %s", symbol, e)
            return self._fallback_stale(symbol, cached)

    def record_snapshot(self, snapshot: WhalePositioningSnapshot, save_to_disk: bool = False) -> None:
        """Inject snapshot directly (useful for testing and deterministic replays)."""
        self._record_item(snapshot, save_to_disk=save_to_disk)
