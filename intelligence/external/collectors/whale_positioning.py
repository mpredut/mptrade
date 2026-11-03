"""Binance top-trader whale positioning and Open Interest flow collector.

Fetches public, unauthenticated metrics from Binance Futures:
- topLongShortPositionRatio: Long/short ratio of the top 20% largest balance accounts (whales).
- takerlongshortRatio: Taker buy volume vs taker sell volume (whale aggression).
- openInterestHist: Historical Open Interest flow (USD & coin amounts).
- Evaluates OI-Price divergence regimes (accumulation vs short-covering vs distribution).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from botcore import obj_attr, safe_float, safe_pos_float
from cachepaths import canonical_symbol_key, telemetry_cache_file_path
from intelligence.external.collectors.base_collector import (
    _DiskBackedSymbolCollector,
    _clean_dict_float,
    _clean_snapshot_symbol,
    _effective_now_ts,
    _extract_base_asset,
    _has_hyperliquid_instrument,
    _resolve_binance_lookup_symbol,
    _symbol_variants,
    is_cache_fresh,
)
from intelligence.macro_sentiment_ssot import fetch_public_json
from state_io import read_json_dict, save_json_state

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
    taker_ratio_5m: Optional[float] = None
    taker_ratio_1h: Optional[float] = None
    taker_ratio_4h: Optional[float] = None
    cvd_5m_usd: Optional[float] = None
    cvd_1h_usd: Optional[float] = None
    cvd_4h_usd: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], symbol: str = "") -> WhalePositioningSnapshot:
        if not isinstance(data, dict):
            data = {}
        raw_reg = data.get("divergence_regime")
        reg_str = raw_reg.strip() if isinstance(raw_reg, str) and raw_reg.strip() else "neutral"
        return cls(
            symbol=_clean_snapshot_symbol(data, symbol),
            top_traders_long_ratio=_clean_dict_float(data, "top_traders_long_ratio", 1.0, "whale_long_ratio", nonneg=True),
            top_traders_long_pct=min(1.0, _clean_dict_float(data, "top_traders_long_pct", 0.5, "whale_long_pct", nonneg=True)),
            taker_buy_sell_ratio=_clean_dict_float(data, "taker_buy_sell_ratio", 1.0, nonneg=True),
            taker_buy_vol_usd=_clean_dict_float(data, "taker_buy_vol_usd", 0.0, nonneg=True),
            taker_sell_vol_usd=_clean_dict_float(data, "taker_sell_vol_usd", 0.0, nonneg=True),
            open_interest_usd=_clean_dict_float(data, "open_interest_usd", 0.0, "oi_usd", nonneg=True),
            open_interest_1h_change_pct=_clean_dict_float(data, "open_interest_1h_change_pct", 0.0, "oi_1h_chg_pct"),
            divergence_regime=reg_str,
            ts=_clean_dict_float(data, "ts", 0.0, nonneg=True),
            funding_rate=safe_float(data.get("funding_rate"), default=None),
            taker_ratio_5m=safe_pos_float(data.get("taker_ratio_5m"), default=None),
            taker_ratio_1h=safe_pos_float(data.get("taker_ratio_1h"), default=None),
            taker_ratio_4h=safe_pos_float(data.get("taker_ratio_4h"), default=None),
            cvd_5m_usd=safe_float(data.get("cvd_5m_usd"), default=None),
            cvd_1h_usd=safe_float(data.get("cvd_1h_usd"), default=None),
            cvd_4h_usd=safe_float(data.get("cvd_4h_usd"), default=None),
        )


class WhalePositioningCollector(_DiskBackedSymbolCollector):
    """Collects and caches Binance Futures top trader and open interest metrics."""

    _DISK_FILE_PREFIX = "whale_snapshot"
    _SNAPSHOT_CLS = WhalePositioningSnapshot

    def __init__(
        self,
        cache_ttl_sec: float = 60.0,
        cache_dir: Optional[str] = "cachedb",
        disk_ttl_sec: float = 180.0,
    ) -> None:
        super().__init__(cache_ttl_sec, 120.0, cache_dir, disk_ttl_sec, 300.0)

    def _resolve_price_1h_change_pct(self, symbol: str) -> float:
        """Resolve 1-hour price change percentage from cached price history (0 ms local)."""
        cache_dir = self._effective_cache_dir()
        if not cache_dir or not isinstance(symbol, str) or not symbol.strip():
            return 0.0
        try:
            from intelligence.sentiment.collectors.microstructure_collector import load_symbol_price_series
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
                if not (p := telemetry_cache_file_path(self.cache_dir, "derivatives_telemetry", s)):
                    continue
                try:
                    d = read_json_dict(p, default=None, recover_previous=True)
                    if isinstance(d, dict) and is_cache_fresh(d.get("ts"), now_ts, max_age):
                        if (fr_val := safe_float(d.get("funding_rate"), default=None)) is not None:
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
            # 1. Top Trader Long/Short Position Ratio (5m granularity for fresh intra-cycle updates)
            url_top = f"https://fapi.binance.com/futures/data/topLongShortPositionRatio?symbol={lookup_sym}&period=5m&limit=2"
            latest_top, _ = self._fetch_latest_json_row(url_top, "topLongShortPositionRatio", lookup_sym)
            top_ratio = _clean_dict_float(latest_top, "longShortRatio", 1.0)
            top_long_pct = _clean_dict_float(latest_top, "longAccount", 0.5)

            # 2. Taker Buy/Sell Volume Ratio (48x5m = 4h multi-horizon 5m/15m/1h/4h rolling windows for live CVD / taker flow)
            url_taker = f"https://fapi.binance.com/futures/data/takerlongshortRatio?symbol={lookup_sym}&period=5m&limit=48"
            latest_taker, data_taker = self._fetch_latest_json_row(url_taker, "takerlongshortRatio", lookup_sym)
            all_taker_rows = [r for r in (data_taker if isinstance(data_taker, list) else []) if isinstance(r, dict)] or [latest_taker]
            taker_rows = all_taker_rows[-3:]
            taker_buy = sum(max(0.0, _clean_dict_float(r, "buyVol", 0.0)) for r in taker_rows)
            taker_sell = sum(max(0.0, _clean_dict_float(r, "sellVol", 0.0)) for r in taker_rows)
            taker_ratio = (
                round(taker_buy / taker_sell, 4)
                if taker_buy > 0.0 and taker_sell > 0.0
                else _clean_dict_float(latest_taker, "buySellRatio", 1.0)
            )
            def _horizon_taker_stats(rows_slice: list[Dict[str, Any]], px_mult: float) -> tuple[Optional[float], Optional[float]]:
                if not rows_slice:
                    return None, None
                b_sum = sum(max(0.0, _clean_dict_float(r, "buyVol", 0.0)) for r in rows_slice)
                s_sum = sum(max(0.0, _clean_dict_float(r, "sellVol", 0.0)) for r in rows_slice)
                r_val = round(b_sum / s_sum, 4) if b_sum > 0.0 and s_sum > 0.0 else None
                cvd_val = round((b_sum - s_sum) * (px_mult if px_mult > 0.0 else 1.0), 0) if (b_sum + s_sum) > 0.0 else None
                return r_val, cvd_val

            # 3. Open Interest History (13x5m = 1h rolling window for live 1h OI change and USD volume conversion)
            url_oi = f"https://fapi.binance.com/futures/data/openInterestHist?symbol={lookup_sym}&period=5m&limit=13"
            latest_oi, data_oi = self._fetch_latest_json_row(url_oi, "openInterestHist", lookup_sym)
            latest_oi_usd = max(0.0, _clean_dict_float(latest_oi, "sumOpenInterestValue", 0.0))
            latest_oi_coins = max(0.0, _clean_dict_float(latest_oi, "sumOpenInterest", 0.0))
            mark_px = 0.0
            if latest_oi_usd > 0.0 and latest_oi_coins > 0.0:
                mark_px = latest_oi_usd / latest_oi_coins
            else:
                mark_px = self._resolve_ssot_price(symbol, now=now) or 0.0
            if mark_px > 0.0:
                taker_buy = round(taker_buy * mark_px, 2)
                taker_sell = round(taker_sell * mark_px, 2)
            tr_5m, cvd_5m = _horizon_taker_stats(all_taker_rows[-1:], mark_px)
            tr_1h, cvd_1h = _horizon_taker_stats(all_taker_rows[-12:], mark_px) if len(all_taker_rows) >= 4 else (None, None)
            tr_4h, cvd_4h = _horizon_taker_stats(all_taker_rows[-48:], mark_px) if len(all_taker_rows) >= 13 else (None, None)
            oi_rows = [r for r in (data_oi if isinstance(data_oi, list) else []) if isinstance(r, dict)]
            prev_oi_dict = oi_rows[0] if len(oi_rows) >= 2 else None
            prev_oi_usd = (
                max(0.0, _clean_dict_float(prev_oi_dict, "sumOpenInterestValue", latest_oi_usd))
                if prev_oi_dict is not None
                else latest_oi_usd
            )
            if abs(price_1h_change_pct) < 1e-6 and mark_px > 0.0 and prev_oi_dict is not None:
                prev_oi_coins = max(0.0, _clean_dict_float(prev_oi_dict, "sumOpenInterest", 0.0))
                if prev_oi_usd > 0.0 and prev_oi_coins > 0.0:
                    prev_mark_px = prev_oi_usd / prev_oi_coins
                    if prev_mark_px > 0.0:
                        price_1h_change_pct = (mark_px - prev_mark_px) / prev_mark_px * 100.0
            oi_change_pct = ((latest_oi_usd - prev_oi_usd) / prev_oi_usd * 100.0) if prev_oi_usd > 0 else 0.0

            # 4. Classify OI-Price divergence regime
            if cached:
                prev_snap = cached[1] if (isinstance(cached, tuple) and len(cached) >= 2) else cached
            else:
                prev_snap = self._load_from_disk(symbol, now, ignore_ttl=True)
            prev_reg = obj_attr(prev_snap, "divergence_regime", None) if prev_snap else None
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
                taker_ratio_5m=tr_5m,
                taker_ratio_1h=tr_1h,
                taker_ratio_4h=tr_4h,
                cvd_5m_usd=cvd_5m,
                cvd_1h_usd=cvd_1h,
                cvd_4h_usd=cvd_4h,
            )
            return self._commit_snapshot(symbol, now, snapshot)
        except Exception as e:
            try:
                if (hl_snap := self._fetch_hyperliquid_whale_fallback(symbol, price_1h_change_pct, now, cached)) is not None:
                    return hl_snap
            except Exception:
                pass
            logger.warning("Could not fetch whale positioning for %s: %s", symbol, e)
            return self._fallback_stale(symbol, cached)

    def _fetch_hyperliquid_whale_fallback(
        self,
        symbol: str,
        price_1h_change_pct: float,
        now: float,
        cached: Any,
    ) -> Optional[WhalePositioningSnapshot]:
        """Fallback to Hyperliquid L1 recentTrades and metaAndAssetCtxs for HYPE/HL symbols."""
        if not _has_hyperliquid_instrument(symbol):
            return None
        base_coin = _extract_base_asset(symbol)
        if not base_coin:
            return None
        trades_raw = self._fetch_json_dict(
            f"https://api.hyperliquid.xyz/info?type=recentTrades&coin={base_coin}",
            "recentTrades",
            base_coin,
            timeout=4.0,
        )
        trades = trades_raw.get("result_list") if isinstance(trades_raw, dict) else None
        if not isinstance(trades, list) or not trades:
            return None
        taker_buy = 0.0
        taker_sell = 0.0
        for tr in trades:
            if not isinstance(tr, dict):
                continue
            px = safe_pos_float(tr.get("px"), default=0.0) or 0.0
            sz = safe_pos_float(tr.get("sz"), default=0.0) or 0.0
            side = str(tr.get("side") or "").strip().upper()
            if px > 0.0 and sz > 0.0:
                if side == "B":
                    taker_buy += px * sz
                elif side == "A":
                    taker_sell += px * sz
        taker_buy = round(taker_buy, 2)
        taker_sell = round(taker_sell, 2)
        taker_ratio = round(taker_buy / taker_sell, 4) if taker_buy > 0.0 and taker_sell > 0.0 else 1.0
        oi_usd = 0.0
        funding_rate = self._resolve_cached_funding_rate(symbol, now=now)
        try:
            ctx = self._fetch_hyperliquid_asset_ctx(symbol, timeout=4.0)
            if isinstance(ctx, dict):
                oi_c = max(0.0, _clean_dict_float(ctx, "openInterest", 0.0))
                m_px = max(0.0, _clean_dict_float(ctx, "markPx", 0.0, "oraclePx"))
                oi_usd = round(oi_c * m_px, 2)
                if funding_rate is None:
                    funding_rate = _clean_dict_float(ctx, "funding", 0.0)
        except Exception:
            pass
        prev_snap = (cached[1] if (isinstance(cached, tuple) and len(cached) >= 2) else cached) if cached else self._load_from_disk(symbol, now, ignore_ttl=True)
        prev_oi = safe_pos_float(obj_attr(prev_snap, "open_interest_usd", None), default=0.0) or 0.0
        oi_chg_pct = round(((oi_usd - prev_oi) / prev_oi) * 100.0, 2) if (oi_usd > 0.0 and prev_oi > 0.0) else 0.0
        prev_reg = obj_attr(prev_snap, "divergence_regime", None) if prev_snap else None
        regime = self.classify_regime(
            price_1h_change_pct=price_1h_change_pct,
            oi_change_pct=oi_chg_pct,
            taker_ratio=taker_ratio,
            funding_rate=funding_rate,
            prev_regime=prev_reg,
            top_long_pct=0.5,
        )
        snapshot = WhalePositioningSnapshot(
            symbol=symbol,
            top_traders_long_ratio=1.0,
            top_traders_long_pct=0.5,
            taker_buy_sell_ratio=taker_ratio,
            taker_buy_vol_usd=taker_buy,
            taker_sell_vol_usd=taker_sell,
            open_interest_usd=oi_usd,
            open_interest_1h_change_pct=oi_chg_pct,
            divergence_regime=regime,
            ts=now,
            funding_rate=funding_rate,
        )
        return self._commit_snapshot(symbol, now, snapshot)

    def record_snapshot(self, snapshot: WhalePositioningSnapshot, save_to_disk: bool = False) -> None:
        """Inject snapshot directly (useful for testing and deterministic replays)."""
        self._record_item(snapshot, save_to_disk=save_to_disk)
