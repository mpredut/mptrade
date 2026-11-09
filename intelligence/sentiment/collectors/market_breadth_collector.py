"""Market breadth and cross-asset sentiment dispersion collector."""
from __future__ import annotations

from dataclasses import dataclass
import logging
import statistics
from typing import Any, Dict, List, Optional, Tuple

from cachepaths import canonical_base_asset
from intelligence.macro_sentiment_ssot import (
    CachedHttpJsonCollector,
    fetch_public_json,
    get_breadth_excluded_bases,
    get_breadth_min_volume_usd,
    get_breadth_quote_suffixes,
    get_breadth_top_mover_min_volume_usd,
    get_market_breadth_endpoint_url,
    is_cache_fresh,
    resolve_now_ts,
    safe_float,
    safe_nonneg_float,
    safe_nonneg_ts,
)

from intelligence.sentiment.collectors.microstructure_collector import (
    TelemetryCacheCollectorInterface,
    TelemetryCollectorFactory,
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
    btc_price: Optional[float] = None
    btc_chg_24h_pct: Optional[float] = None
    eth_price: Optional[float] = None
    eth_chg_24h_pct: Optional[float] = None
    eth_btc_ratio: Optional[float] = None
    eth_btc_chg_24h_pct: Optional[float] = None
    btc_vol_share_pct: Optional[float] = None
    eth_vol_share_pct: Optional[float] = None
    alt_vol_share_pct: Optional[float] = None
    rotation_regime: Optional[str] = None


class MarketBreadthCollector(CachedHttpJsonCollector, TelemetryCacheCollectorInterface):
    """Computes real-time market-wide breadth and sentiment dispersion from liquid pairs."""

    cls_name = "MarketBreadthCollector"
    COLLECTOR_ID = "sentiment_market_breadth"
    _COLLECTOR_LABEL = "market breadth"
    _LOGGER = logger

    def __init__(
        self,
        cache_ttl_sec: float = 60.0,
        min_volume_usd: float = 5_000_000.0,
        endpoint_url: Optional[str] = None,
        *,
        start_sync: Optional[bool] = None,
    ) -> None:
        super().__init__(cache_ttl_sec, 60.0, endpoint_url, get_market_breadth_endpoint_url, start_sync=start_sync)
        self.min_volume_usd = safe_nonneg_float(min_volume_usd, 5_000_000.0)

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
        min_vol = safe_nonneg_float(min_volume_usd, get_breadth_min_volume_usd())
        top_mover_min_vol = max(min_vol, get_breadth_top_mover_min_volume_usd())
        quote_suffixes = get_breadth_quote_suffixes()
        excluded_bases = set(get_breadth_excluded_bases())
        all_upper_symbols: set[str] = (
            {
                t["symbol"].strip().upper()
                for t in tickers
                if isinstance(t, dict) and isinstance(t.get("symbol"), str)
            }
            if len(tickers) >= 100
            else set()
        )
        rows: List[Tuple[str, float, float]] = []
        btc_px: Optional[float] = None
        btc_chg: Optional[float] = None
        eth_px: Optional[float] = None
        eth_chg: Optional[float] = None
        eth_btc_px: Optional[float] = None
        eth_btc_chg: Optional[float] = None
        btc_vol_usd: float = 0.0
        eth_vol_usd: float = 0.0

        for item in tickers:
            if not isinstance(item, dict) or not isinstance(item.get("symbol"), str):
                continue
            symbol = item["symbol"].strip().upper()
            last_px = safe_float(item.get("lastPrice"), default=None)
            change_pct = safe_float(item.get("priceChangePercent"), default=None)
            quote_vol = safe_float(item.get("quoteVolume"), default=None)
            if symbol == "ETHBTC":
                if last_px is not None and last_px > 0.0:
                    eth_btc_px = round(last_px, 6)
                if change_pct is not None:
                    eth_btc_chg = round(change_pct, 2)
                continue
            if not (symbol.isascii() and symbol.isalnum() and symbol.endswith(quote_suffixes)):
                continue
            matching_quote = next((q for q in quote_suffixes if symbol.endswith(q)), "")
            base_candidate = symbol[:-len(matching_quote)] if matching_quote else symbol
            base_sym = canonical_base_asset(base_candidate) or canonical_base_asset(symbol) or base_candidate
            is_unpaired_tokenized_stock_b = (
                bool(all_upper_symbols)
                and len(base_candidate) >= 3
                and base_candidate.endswith("B")
                and not any(
                    f"{base_candidate}{q}" in all_upper_symbols
                    for q in ("USDC", "BTC", "ETH", "BNB", "FDUSD", "TRY", "EUR")
                )
            )
            if (
                base_candidate in excluded_bases
                or base_sym in excluded_bases
                or is_unpaired_tokenized_stock_b
                or (len(base_sym) > 4 and base_sym.endswith(("UP", "DOWN", "BULL", "BEAR")))
                or (
                    base_candidate.endswith("USD")
                    and last_px is not None
                    and 0.98 <= last_px <= 1.02
                    and change_pct is not None
                    and abs(change_pct) <= 0.50
                )
            ):
                continue
            if base_sym == "BTC":
                if symbol in ("BTCUSDC", "BTCUSDT", "BTCUSD"):
                    if last_px is not None and last_px > 0.0 and (btc_px is None or symbol == "BTCUSDC"):
                        btc_px = round(last_px, 2)
                    if change_pct is not None and (btc_chg is None or symbol == "BTCUSDC"):
                        btc_chg = round(change_pct, 2)
                if quote_vol is not None and quote_vol > 0.0:
                    btc_vol_usd += quote_vol
            elif base_sym == "ETH":
                if symbol in ("ETHUSDC", "ETHUSDT", "ETHUSD"):
                    if last_px is not None and last_px > 0.0 and (eth_px is None or symbol == "ETHUSDC"):
                        eth_px = round(last_px, 2)
                    if change_pct is not None and (eth_chg is None or symbol == "ETHUSDC"):
                        eth_chg = round(change_pct, 2)
                if quote_vol is not None and quote_vol > 0.0:
                    eth_vol_usd += quote_vol
            if quote_vol is not None and change_pct is not None and quote_vol >= min_vol:
                rows.append((symbol, change_pct, quote_vol))

        if len(rows) < 5:
            return None

        rows.sort(key=lambda x: x[1])
        filtered = [(s, c) for s, c, _ in rows]
        top_pool = [(s, c) for s, c, qv in rows if qv >= top_mover_min_vol]
        mover_pool = top_pool if len(top_pool) >= 3 else filtered

        def _dedup_top_movers(pairs: List[Tuple[str, float]], limit: int = 3) -> Tuple[Tuple[str, float], ...]:
            out: List[Tuple[str, float]] = []
            seen_bases: set[str] = set()
            for sym_m, chg_m in pairs:
                base_m = canonical_base_asset(sym_m) or sym_m
                if base_m in seen_bases:
                    continue
                seen_bases.add(base_m)
                out.append((sym_m, chg_m))
                if len(out) >= limit:
                    break
            return tuple(out)

        changes = [c for _, c, _ in rows]
        advancing = sum(1 for c in changes if c > 0)
        declining = sum(1 for c in changes if c < 0)
        total = len(changes)
        advance_ratio = advancing / total if total > 0 else 0.5

        median_change = float(statistics.median(changes))
        mean_change = float(statistics.fmean(changes))
        dispersion_std = float(statistics.pstdev(changes)) if total > 1 else 0.0

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

        if eth_btc_px is None and eth_px and btc_px and btc_px > 0.0:
            eth_btc_px = round(eth_px / btc_px, 6)
        if eth_btc_chg is None and eth_chg is not None and btc_chg is not None and (100.0 + btc_chg) > 0.0:
            eth_btc_chg = round(((100.0 + eth_chg) / (100.0 + btc_chg) - 1.0) * 100.0, 2)
        total_vol_usd = sum(qv for _, _, qv in rows if qv > 0.0)
        eff_total_vol = max(total_vol_usd, btc_vol_usd + eth_vol_usd)
        btc_share = round((btc_vol_usd / eff_total_vol) * 100.0, 2) if eff_total_vol > 0.0 and btc_vol_usd > 0.0 else None
        eth_share = round((eth_vol_usd / eff_total_vol) * 100.0, 2) if eff_total_vol > 0.0 and eth_vol_usd > 0.0 else None
        alt_share = (
            round(max(0.0, eff_total_vol - btc_vol_usd - eth_vol_usd) / eff_total_vol * 100.0, 2)
            if eff_total_vol > 0.0 and (btc_vol_usd > 0.0 or eth_vol_usd > 0.0)
            else None
        )
        rotation_regime: Optional[str] = None
        if alt_share is not None or eth_btc_chg is not None:
            ref_btc_c = btc_chg if btc_chg is not None else 0.0
            if alt_share is not None and alt_share >= 45.0 and median_change > ref_btc_c + 0.5:
                rotation_regime = "ALT_INFLOW_ROTATION"
            elif btc_share is not None and btc_share >= 50.0 and median_change < ref_btc_c - 0.5:
                rotation_regime = "BTC_DOMINANCE_FLIGHT"
            elif eth_btc_chg is not None and eth_btc_chg >= 1.5:
                rotation_regime = "ETH_OUTPERFORMING"
            elif eth_btc_chg is not None and eth_btc_chg <= -1.5:
                rotation_regime = "BTC_OUTPERFORMING_ETH"
            else:
                rotation_regime = "BALANCED_ROTATION"

        gainer_candidates = (
            [(s, c) for s, c in mover_pool[::-1] if c > 0]
            + [(s, c) for s, c in filtered[::-1] if c > 0]
            + mover_pool[::-1]
            + filtered[::-1]
        )
        loser_candidates = (
            [(s, c) for s, c in mover_pool if c < 0]
            + [(s, c) for s, c in filtered if c < 0]
            + mover_pool
            + filtered
        )

        return MarketBreadthSnapshot(
            advance_ratio=round(advance_ratio, 4),
            advancing_count=advancing,
            declining_count=declining,
            total_count=total,
            median_change_pct=round(median_change, 3),
            mean_change_pct=round(mean_change, 3),
            dispersion_std=round(dispersion_std, 3),
            regime=regime,
            top_gainers=tuple(sorted(_dedup_top_movers(gainer_candidates, 3), key=lambda x: -x[1])),
            top_losers=tuple(sorted(_dedup_top_movers(loser_candidates, 3), key=lambda x: x[1])),
            ts=now,
            btc_price=btc_px,
            btc_chg_24h_pct=btc_chg,
            eth_price=eth_px,
            eth_chg_24h_pct=eth_chg,
            eth_btc_ratio=eth_btc_px,
            eth_btc_chg_24h_pct=eth_btc_chg,
            btc_vol_share_pct=btc_share,
            eth_vol_share_pct=eth_share,
            alt_vol_share_pct=alt_share,
            rotation_regime=rotation_regime,
        )

    def _parse_fetched_payload(self, data: Any, now: float) -> Optional[MarketBreadthSnapshot]:
        if isinstance(data, list):
            return self.parse_tickers(data, min_volume_usd=self.min_volume_usd, now_ts=now)
        return None


TelemetryCollectorFactory.register(MarketBreadthCollector.COLLECTOR_ID, MarketBreadthCollector)
