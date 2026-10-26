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
from state_io import read_recovering_previous

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
        import math

        def _f(key: str, default: float) -> float:
            raw = data.get(key)
            if raw is None or isinstance(raw, bool):
                return default
            try:
                val = float(raw)
                return val if math.isfinite(val) else default
            except (TypeError, ValueError):
                return default

        fr = data.get("funding_rate")
        fr_val: Optional[float] = None
        if fr is not None and not isinstance(fr, bool):
            try:
                cand = float(fr)
                if math.isfinite(cand):
                    fr_val = cand
            except (TypeError, ValueError):
                fr_val = None
        return cls(
            symbol=str(data.get("symbol") or ""),
            top_traders_long_ratio=_f("top_traders_long_ratio", 1.0),
            top_traders_long_pct=_f("top_traders_long_pct", 0.5),
            taker_buy_sell_ratio=_f("taker_buy_sell_ratio", 1.0),
            taker_buy_vol_usd=_f("taker_buy_vol_usd", 0.0),
            taker_sell_vol_usd=_f("taker_sell_vol_usd", 0.0),
            open_interest_usd=_f("open_interest_usd", 0.0),
            open_interest_1h_change_pct=_f("open_interest_1h_change_pct", 0.0),
            divergence_regime=str(data.get("divergence_regime") or "neutral"),
            ts=_f("ts", 0.0),
            funding_rate=fr_val,
        )


class WhalePositioningCollector:
    """Collects and caches Binance Futures top trader and open interest metrics."""

    def __init__(
        self,
        cache_ttl_sec: float = 120.0,
        cache_dir: Optional[str] = "cachedb",
        disk_ttl_sec: float = 300.0,
    ) -> None:
        import math

        def _clean_ttl(val: Any, default: float) -> float:
            if val is None or isinstance(val, bool):
                return default
            try:
                vf = float(val)
                return vf if math.isfinite(vf) and vf >= 0.0 else default
            except (TypeError, ValueError, OverflowError):
                return default

        self.cache_ttl_sec = _clean_ttl(cache_ttl_sec, 120.0)
        self.cache_dir = cache_dir
        self.disk_ttl_sec = _clean_ttl(disk_ttl_sec, 300.0)
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
        safe_sym = symbol.upper().strip().replace("/", "").replace("-", "").replace("_", "")
        return os.path.join(cache_dir, f"whale_snapshot_{safe_sym}.json")

    def _load_from_disk(self, symbol: str, now: float, *, ignore_ttl: bool = False) -> Optional[WhalePositioningSnapshot]:
        from state_io import read_recovering_previous
        max_age = max(self.cache_ttl_sec, self.disk_ttl_sec)
        for cand in _symbol_variants(symbol):
            p = self._disk_path(cand)
            if not p:
                continue
            try:
                data = read_recovering_previous(p, json.loads, default=None)
                if not isinstance(data, dict):
                    continue
                snapshot = WhalePositioningSnapshot.from_dict(data)
                if snapshot.ts > 0.0 and 0.0 <= (now - snapshot.ts) < max_age:
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
            from state_io import save_json_state
            save_json_state(p, snapshot.to_dict(), indent=2)
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
                if "/" in sym_key:
                    continue
                p = os.path.join(cache_dir, f"cache_24_price_{sym_key}.json")
                try:
                    cdata = read_recovering_previous(p, json.loads, default=None)
                    if not isinstance(cdata, dict):
                        continue
                    items_dict = cdata.get("items")
                    pts = None
                    if isinstance(items_dict, dict):
                        for k_cand in variants:
                            pts = items_dict.get(k_cand)
                            if pts:
                                break
                        if not pts and items_dict:
                            pts = next(iter(items_dict.values()))
                    elif isinstance(items_dict, list):
                        pts = items_dict
                    if pts and len(pts) >= 2:
                        import math
                        valid_pts = []
                        for pt in pts:
                            try:
                                if (
                                    not isinstance(pt, (list, tuple))
                                    or len(pt) < 2
                                    or isinstance(pt[0], bool)
                                    or isinstance(pt[1], bool)
                                ):
                                    continue
                                raw_ts = float(pt[0])
                                px = float(pt[1])
                                if not (math.isfinite(raw_ts) and raw_ts > 0 and math.isfinite(px) and px > 0):
                                    continue
                                ts_s = raw_ts / 1000.0 if raw_ts > 1e11 else raw_ts
                                valid_pts.append((ts_s, px))
                            except Exception:
                                continue
                        if len(valid_pts) >= 2:
                            valid_pts.sort(key=lambda x: x[0])
                            now_sec = valid_pts[-1][0]
                            target_sec = now_sec - 3600.0
                            latest_p = valid_pts[-1][1]
                            ref_p = valid_pts[0][1]
                            for pt_sec, pt_px in reversed(valid_pts):
                                if pt_sec <= target_sec:
                                    ref_p = pt_px
                                    break
                            if ref_p > 0:
                                return (latest_p - ref_p) / ref_p * 100.0
                except Exception:
                    continue
        except Exception as e:
            logger.debug("Failed calculating 1h price change from cache for %s: %s", symbol, e)
        return 0.0

    def _resolve_cached_funding_rate(self, symbol: str, now: Optional[float] = None) -> Optional[float]:
        """Lookup cached funding rate from derivatives telemetry if available and fresh."""
        import math
        cache_dir = self._effective_cache_dir()
        if not cache_dir:
            return None
        try:
            now_ts = (
                float(now)
                if (now is not None and not isinstance(now, bool) and math.isfinite(float(now)) and float(now) > 0.0)
                else time.time()
            )
            max_age = max(self.cache_ttl_sec, self.disk_ttl_sec)
            variants = _symbol_variants(symbol)
            for s in variants:
                clean = s.upper().replace("/", "").replace("-", "").replace("_", "")
                p = os.path.join(cache_dir, f"derivatives_telemetry_{clean}.json")
                try:
                    d = read_recovering_previous(p, json.loads, default=None)
                    if not isinstance(d, dict):
                        continue
                    raw_ts = d.get("ts")
                    if raw_ts is None or isinstance(raw_ts, bool):
                        continue
                    ts_val = float(raw_ts)
                    if not math.isfinite(ts_val) or ts_val <= 0 or not (0.0 <= (now_ts - ts_val) < max_age):
                        continue
                    fr = d.get("funding_rate")
                    if fr is not None and not isinstance(fr, bool):
                        fr_val = float(fr)
                        if math.isfinite(fr_val):
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
        import math
        try:
            p_chg = float(price_1h_change_pct) if not isinstance(price_1h_change_pct, bool) else 0.0
            price_1h_change_pct = p_chg if math.isfinite(p_chg) else 0.0
        except (TypeError, ValueError, OverflowError):
            price_1h_change_pct = 0.0
        try:
            o_chg = float(oi_change_pct) if not isinstance(oi_change_pct, bool) else 0.0
            oi_change_pct = o_chg if math.isfinite(o_chg) else 0.0
        except (TypeError, ValueError, OverflowError):
            oi_change_pct = 0.0
        try:
            t_rat = float(taker_ratio) if not isinstance(taker_ratio, bool) else 1.0
            taker_ratio = t_rat if math.isfinite(t_rat) and t_rat > 0 else 1.0
        except (TypeError, ValueError, OverflowError):
            taker_ratio = 1.0
        if funding_rate is not None:
            if isinstance(funding_rate, bool):
                funding_rate = None
            else:
                try:
                    fr_f = float(funding_rate)
                    funding_rate = fr_f if math.isfinite(fr_f) else None
                except (TypeError, ValueError, OverflowError):
                    funding_rate = None
        if top_long_pct is not None:
            if isinstance(top_long_pct, bool):
                top_long_pct = None
            else:
                try:
                    tl_f = float(top_long_pct)
                    top_long_pct = tl_f if math.isfinite(tl_f) else None
                except (TypeError, ValueError, OverflowError):
                    top_long_pct = None
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

    def fetch(
        self,
        symbol: str,
        price_1h_change_pct: float = 0.0,
        allow_network: bool = True,
        now: Optional[float] = None,
    ) -> Optional[WhalePositioningSnapshot]:
        """Fetch latest whale positioning metrics from public Binance Futures endpoints."""
        import math
        symbol = str(symbol or "").strip().upper()
        if not symbol:
            return None
        try:
            now_f = float(now) if (now is not None and not isinstance(now, bool)) else time.time()
            now = now_f if (math.isfinite(now_f) and now_f > 0.0) else time.time()
        except (TypeError, ValueError, OverflowError):
            now = time.time()
        cached = None
        for cand in _symbol_variants(symbol):
            cached = self._cache.get(cand)
            if cached is not None:
                break
        if cached:
            if isinstance(cached, tuple) and len(cached) >= 2:
                if 0.0 <= (now - cached[0]) < self.cache_ttl_sec:
                    return cached[1]
            elif isinstance(cached, WhalePositioningSnapshot):
                if 0.0 <= (now - cached.ts) < self.cache_ttl_sec:
                    return cached

        # Check disk cache before network
        disk_snap = self._load_from_disk(symbol, now)
        if disk_snap is not None:
            return disk_snap

        if not allow_network:
            return None

        p1h_f = 0.0
        if price_1h_change_pct is not None and not isinstance(price_1h_change_pct, bool):
            try:
                cand_p1h = float(price_1h_change_pct)
                if math.isfinite(cand_p1h):
                    p1h_f = cand_p1h
            except (TypeError, ValueError, OverflowError):
                p1h_f = 0.0
        price_1h_change_pct = p1h_f
        if abs(price_1h_change_pct) < 1e-6:
            price_1h_change_pct = self._resolve_price_1h_change_pct(symbol)

        def _safe_float(raw: Any, default: float) -> float:
            if raw is None or isinstance(raw, bool):
                return default
            try:
                val = float(raw)
                return val if math.isfinite(val) else default
            except (TypeError, ValueError, OverflowError):
                return default

        lookup_sym = _resolve_binance_lookup_symbol(symbol)
        try:
            # 1. Top Trader Long/Short Position Ratio
            url_top = f"https://fapi.binance.com/futures/data/topLongShortPositionRatio?symbol={lookup_sym}&period=1h&limit=2"
            req_top = urllib.request.Request(url_top, headers={"User-Agent": "mptrade-whale/1.0"})
            with urllib.request.urlopen(req_top, timeout=5) as resp:
                data_top = json.loads(resp.read().decode("utf-8"))
            if not isinstance(data_top, list) or not data_top or not isinstance(data_top[-1], dict):
                raise ValueError(f"Empty topLongShortPositionRatio response for {lookup_sym}")

            top_ratio = _safe_float(data_top[-1].get("longShortRatio"), 1.0)
            top_long_pct = _safe_float(data_top[-1].get("longAccount"), 0.5)

            # 2. Taker Buy/Sell Volume Ratio
            url_taker = f"https://fapi.binance.com/futures/data/takerlongshortRatio?symbol={lookup_sym}&period=1h&limit=2"
            req_taker = urllib.request.Request(url_taker, headers={"User-Agent": "mptrade-whale/1.0"})
            with urllib.request.urlopen(req_taker, timeout=5) as resp:
                data_taker = json.loads(resp.read().decode("utf-8"))
            if not isinstance(data_taker, list) or not data_taker or not isinstance(data_taker[-1], dict):
                raise ValueError(f"Empty takerlongshortRatio response for {lookup_sym}")

            taker_ratio = _safe_float(data_taker[-1].get("buySellRatio"), 1.0)
            taker_buy = max(0.0, _safe_float(data_taker[-1].get("buyVol"), 0.0))
            taker_sell = max(0.0, _safe_float(data_taker[-1].get("sellVol"), 0.0))

            # 3. Open Interest History (compute 1h change)
            url_oi = f"https://fapi.binance.com/futures/data/openInterestHist?symbol={lookup_sym}&period=1h&limit=3"
            req_oi = urllib.request.Request(url_oi, headers={"User-Agent": "mptrade-whale/1.0"})
            with urllib.request.urlopen(req_oi, timeout=5) as resp:
                data_oi = json.loads(resp.read().decode("utf-8"))
            if not isinstance(data_oi, list) or not data_oi or not isinstance(data_oi[-1], dict):
                raise ValueError(f"Empty openInterestHist response for {lookup_sym}")

            latest_oi_usd = max(0.0, _safe_float(data_oi[-1].get("sumOpenInterestValue"), 0.0))
            prev_oi_usd = (
                max(0.0, _safe_float(data_oi[-2].get("sumOpenInterestValue"), latest_oi_usd))
                if (len(data_oi) >= 2 and isinstance(data_oi[-2], dict))
                else latest_oi_usd
            )
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
                if p:
                    try:
                        data = read_recovering_previous(p, json.loads, default=None)
                        if isinstance(data, dict):
                            return WhalePositioningSnapshot.from_dict(data)
                    except Exception:
                        pass
            return None

    def record_snapshot(self, snapshot: WhalePositioningSnapshot, save_to_disk: bool = False) -> None:
        """Inject snapshot directly (useful for testing and deterministic replays)."""
        self._cache[snapshot.symbol.upper()] = (snapshot.ts, snapshot)
        if save_to_disk:
            self._save_to_disk(snapshot)
