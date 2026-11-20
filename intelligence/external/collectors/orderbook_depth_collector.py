"""Order book depth imbalance and whale limit walls collector.

Analyzes the top 1-2% of the bid/ask order book:
- Depth Imbalance Ratio: Bid Liquidity / (Bid Liquidity + Ask Liquidity)
- Whale Walls: Detects unusually large single limit orders placed on the book.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from botcore import clean_scale, fetch_json_post, lru_dict_put, safe_float, safe_nonneg_float, safe_pos_float
from intelligence.macro_sentiment_ssot import (
    fetch_public_json,
    resolve_derived_telemetry_intervals,
)
from cachepaths import (
    canonical_symbol_key,
    resolve_binance_lookup_symbol,
)
from instrument_registry import resolve_symbol_identity, venue_for_symbol
from providers.base import MarketDataProvider

logger = logging.getLogger("intelligence.external.orderbook_depth")

from intelligence.external.collectors.base_collector import (
    TelemetryCollectorFactory,
    _DiskBackedSymbolCollector,
    _clean_dict_float,
    _clean_snapshot_symbol,
    _clean_snapshot_ts,
    _is_multi_venue_base,
    round_spread_bps,
)


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
    best_bid: float = 0.0
    best_ask: float = 0.0
    spread_bps: float = 0.0
    depth_1pct_bid_usd: float = 0.0
    depth_1pct_ask_usd: float = 0.0
    book_span_pct: float = 0.0
    spot_cvd_15m_usd: Optional[float] = None
    spot_cvd_source: Optional[str] = None
    spot_vol_15m_usd: Optional[float] = None
    agg_spot_cvd_15m_usd: Optional[float] = None
    agg_spot_vol_15m_usd: Optional[float] = None
    agg_spot_cvd_source: Optional[str] = None

    to_dict = asdict

    @classmethod
    def from_dict(cls, data: Dict[str, Any], symbol: str = "") -> OrderbookSnapshot:
        if not isinstance(data, dict):
            data = {}
        mid_price = _clean_dict_float(data, "mid_price", 0.0, "ob_mid_price", nonneg=True)
        best_bid = _clean_dict_float(data, "best_bid", 0.0, nonneg=True)
        best_ask = _clean_dict_float(data, "best_ask", 0.0, nonneg=True)
        if mid_price <= 0.0 and best_bid > 0.0 and best_ask >= best_bid:
            mid_price = round((best_bid + best_ask) / 2.0, 6)
        spread_bps = round_spread_bps(data.get("spread_bps"), best_bid=best_bid, best_ask=best_ask, mid_price=mid_price) or 0.0
        bid_depth = _clean_dict_float(data, "bid_depth_usd", 0.0, nonneg=True)
        ask_depth = _clean_dict_float(data, "ask_depth_usd", 0.0, nonneg=True)
        d1_bid = _clean_dict_float(data, "depth_1pct_bid_usd", 0.0, nonneg=True) or (round(bid_depth * (1.0 / 1.5), 2) if bid_depth > 0.0 else 0.0)
        d1_ask = _clean_dict_float(data, "depth_1pct_ask_usd", 0.0, nonneg=True) or (round(ask_depth * (1.0 / 1.5), 2) if ask_depth > 0.0 else 0.0)
        raw_imb_val = data.get("imbalance_ratio") if data.get("imbalance_ratio") is not None else data.get("ob_imbalance")
        imb_clean = clean_scale(raw_imb_val, default=0.5, allow_percent=True, clamp=True)
        return cls(
            symbol=_clean_snapshot_symbol(data, symbol),
            mid_price=mid_price,
            bid_depth_usd=bid_depth,
            ask_depth_usd=ask_depth,
            imbalance_ratio=imb_clean,
            largest_bid_wall_usd=_clean_dict_float(data, "largest_bid_wall_usd", 0.0, "bid_wall_usd", nonneg=True),
            largest_bid_wall_price=_clean_dict_float(data, "largest_bid_wall_price", 0.0, "bid_wall_px", nonneg=True),
            largest_ask_wall_usd=_clean_dict_float(data, "largest_ask_wall_usd", 0.0, "ask_wall_usd", nonneg=True),
            largest_ask_wall_price=_clean_dict_float(data, "largest_ask_wall_price", 0.0, "ask_wall_px", nonneg=True),
            ts=_clean_snapshot_ts(data),
            best_bid=best_bid,
            best_ask=best_ask,
            spread_bps=spread_bps,
            depth_1pct_bid_usd=d1_bid,
            depth_1pct_ask_usd=d1_ask,
            book_span_pct=_clean_dict_float(data, "book_span_pct", 0.0, nonneg=True),
            spot_cvd_15m_usd=safe_float(data.get("spot_cvd_15m_usd"), default=None),
            spot_cvd_source=str(data["spot_cvd_source"]).strip() if data.get("spot_cvd_source") else None,
            spot_vol_15m_usd=safe_nonneg_float(data.get("spot_vol_15m_usd"), default=None),
            agg_spot_cvd_15m_usd=safe_float(data.get("agg_spot_cvd_15m_usd"), default=None),
            agg_spot_vol_15m_usd=safe_nonneg_float(data.get("agg_spot_vol_15m_usd"), default=None),
            agg_spot_cvd_source=str(data["agg_spot_cvd_source"]).strip() if data.get("agg_spot_cvd_source") else None,
        )


class OrderbookDepthCollector(_DiskBackedSymbolCollector):
    """Collects and analyzes Binance/Kraken Spot order book depth."""

    cls_name = "OrderbookDepthCollector"
    COLLECTOR_ID = "external_orderbook_depth"
    _DISK_FILE_PREFIX = "orderbook_depth"
    _SNAPSHOT_CLS = OrderbookSnapshot
    _QUOTE_ISOLATED = True
    _DEFAULT_CACHE_TTL_SEC = 5.0
    _DEFAULT_DISK_TTL_SEC = 60.0
    _FALLBACK_CACHE_TTL_SEC = 5.0
    _FALLBACK_DISK_TTL_SEC = 60.0

    def __init__(
        self,
        cache_ttl_sec: float = 5.0,
        depth_band_pct: float = 1.5,     # Analyze depth within 1.5% of mid-price
        whale_wall_usd: float = 500_000.0,
        cache_dir: Optional[str] = "cachedb",
        disk_ttl_sec: float = 60.0,
        *,
        start_sync: Optional[bool] = None,
    ) -> None:
        super().__init__(cache_ttl_sec, cache_dir, disk_ttl_sec, start_sync=start_sync)
        self.depth_band_pct = safe_pos_float(depth_band_pct, default=1.5) or 1.5
        self.whale_wall_usd = safe_pos_float(whale_wall_usd, default=500_000.0) or 500_000.0
        self._preferred_url: Dict[str, str] = {}
        self._unlisted_xstock_until: Dict[str, float] = {}
        self._spot_cvd_memo: Dict[Tuple[str, str], Tuple[float, float, Optional[float]]] = {}
        self._agg_spot_cvd_memo: Dict[Tuple[str, str], Tuple[float, Optional[float], Optional[float], str]] = {}

    def _fetch_spot_cvd_payload(self, url: str, post_body: Optional[dict]) -> Any:
        if post_body is not None:
            return fetch_json_post(
                url,
                post_body,
                timeout=3.5,
                headers={"User-Agent": "mptrade-orderbook/1.0"},
            )
        return fetch_public_json(url, timeout=3.5, user_agent="mptrade-orderbook/1.0")

    def _resolve_real_spot_cvd_15m(
        self, symbol: str, *, venue: str, now: float
    ) -> Optional[float]:
        """Fetch and memoize real venue-local 15m spot taker CVD & total volume (USD) at ``whale_interval_sec`` cadence."""
        if not venue:
            return None
        if "_fetch_json_dict" in self.__dict__ and "_fetch_spot_cvd_payload" not in self.__dict__:
            return None
        memo_key = (venue, symbol)
        ttl_sec = safe_pos_float(resolve_derived_telemetry_intervals().get("whale_interval_sec"), 60.0) or 60.0
        cached_cvd = self._spot_cvd_memo.get(memo_key)
        if cached_cvd is not None and 0.0 <= (now - cached_cvd[0]) < ttl_sec:
            return cached_cvd[1]
        req_spec, extractor = MarketDataProvider.resolve_spot_flow_15m_spec(symbol, venue=venue, now_ts=now)
        if req_spec is None or not callable(extractor):
            return None
        url, post_body = req_spec
        try:
            raw_payload = self._fetch_spot_cvd_payload(url, post_body)
            flow_res = extractor(raw_payload, since_ts=max(0.0, now - 900.0))
            if isinstance(flow_res, (tuple, list)) and len(flow_res) >= 2:
                cvd_f = safe_float(flow_res[0], default=None)
                vol_f = safe_nonneg_float(flow_res[1], default=None)
            else:
                cvd_f = safe_float(flow_res, default=None)
                vol_f = None
            if cvd_f is not None:
                lru_dict_put(self._spot_cvd_memo, memo_key, (now, cvd_f, vol_f), 128)
                self._resolve_agg_spot_cvd_15m(
                    symbol,
                    venue=venue,
                    now=now,
                    local_url=url,
                    local_cvd=cvd_f,
                    local_vol=vol_f,
                    ttl_sec=ttl_sec,
                )
                return cvd_f
        except Exception as exc:
            logger.debug("Could not fetch real 15m spot CVD for %s on %s: %s", symbol, venue, exc)
        return cached_cvd[1] if cached_cvd is not None and 0.0 <= (now - cached_cvd[0]) < ttl_sec * 3.0 else None

    def _resolve_agg_spot_cvd_15m(
        self,
        symbol: str,
        *,
        venue: str,
        now: float,
        local_url: str,
        local_cvd: float,
        local_vol: Optional[float],
        ttl_sec: float,
    ) -> Optional[Tuple[float, Optional[float], str]]:
        """Combine venue-local USDC spot flow with deep primary-quote (USDT) spot flow when available."""
        memo_key = (venue, symbol)
        cached_agg = self._agg_spot_cvd_memo.get(memo_key)
        if cached_agg is not None:
            if cached_agg[1] is None and 0.0 <= (now - cached_agg[0]) < ttl_sec * 5.0:
                return None
            if cached_agg[1] is not None and 0.0 <= (now - cached_agg[0]) < ttl_sec:
                return cached_agg[1], cached_agg[2], cached_agg[3]
        agg_spec, agg_ext = MarketDataProvider.resolve_agg_spot_flow_15m_spec(symbol, venue=venue, now_ts=now)
        if agg_spec is None or not callable(agg_ext):
            return None
        agg_url, agg_body = agg_spec
        if agg_url == local_url:
            return None
        try:
            raw_agg = self._fetch_spot_cvd_payload(agg_url, agg_body)
            agg_res = agg_ext(raw_agg, since_ts=max(0.0, now - 900.0))
            if isinstance(agg_res, (tuple, list)) and len(agg_res) >= 2:
                sib_cvd = safe_float(agg_res[0], default=None)
                sib_vol = safe_nonneg_float(agg_res[1], default=None)
            else:
                sib_cvd = safe_float(agg_res, default=None)
                sib_vol = None
            if sib_cvd is not None:
                tot_cvd = round(local_cvd + sib_cvd, 0)
                tot_vol = (
                    round((local_vol or 0.0) + sib_vol, 0)
                    if (local_vol is not None or sib_vol is not None)
                    else None
                )
                src_lbl = "usdc+usdt" if venue == "binance" else f"{venue}+bn_usdt"
                lru_dict_put(self._agg_spot_cvd_memo, memo_key, (now, tot_cvd, tot_vol, src_lbl), 128)
                return tot_cvd, tot_vol, src_lbl
            lru_dict_put(self._agg_spot_cvd_memo, memo_key, (now, None, None, ""), 128)
        except Exception as exc:
            logger.debug("Could not fetch aggregated 15m spot CVD for %s on %s: %s", symbol, venue, exc)
            if cached_agg is None or cached_agg[1] is None:
                lru_dict_put(self._agg_spot_cvd_memo, memo_key, (now, None, None, ""), 128)
            elif (now - cached_agg[0]) >= ttl_sec * 3.0:
                self._agg_spot_cvd_memo.pop(memo_key, None)
        return (
            (cached_agg[1], cached_agg[2], cached_agg[3])
            if cached_agg is not None and cached_agg[1] is not None and 0.0 <= (now - cached_agg[0]) < ttl_sec * 3.0
            else None
        )

    def is_unlisted_xstock_backoff_active(self, symbol: str, now: float) -> bool:
        """Return True when ``symbol`` is a pre-IPO/empty Kraken tokenized pair within its quiet backoff window."""
        ident = resolve_symbol_identity(symbol)
        return bool(ident.is_tokenized_equity and now < self._unlisted_xstock_until.get(ident.wire_symbol, 0.0))

    def fetch(
        self,
        symbol: str,
        allow_network: Optional[bool] = None,
        now: Optional[float] = None,
    ) -> Optional[OrderbookSnapshot]:
        """Fetch spot depth snapshot from Binance or Kraken public API."""
        symbol, now, cached, ready, should_fetch = self._prepare_fetch_or_equity(
            symbol,
            allow_network,
            now,
            lambda s, n, _c: self._fetch_equity_orderbook(s, n),
            "equity orderbook depth",
            logger,
        )
        if not should_fetch or symbol is None:
            return ready

        ident = resolve_symbol_identity(symbol)
        tok_sym, is_tok_sym = ident.wire_symbol, ident.is_tokenized_equity
        if is_tok_sym and now < self._unlisted_xstock_until.get(tok_sym, 0.0):
            return self._fallback_stale(symbol, cached)

        lookup_sym = resolve_binance_lookup_symbol(symbol)
        canon_sym = canonical_symbol_key(symbol)
        query_sym = tok_sym if is_tok_sym else canon_sym
        try:
            data = None
            last_err: Optional[Exception] = None
            # BY DESIGN: L2 orderbook walls/spread/depth are strictly venue-specific; never substitute
            # another venue's book for a registered instrument (cross-venue discovery is only for unregistered tickers).
            # Preserve mixed-case tokenized xStock wire symbols (e.g. SPCXxUSD, ANTxUSD) because
            # canonical_symbol_key strips the lowercase 'x' (SPCXUSD/ANTUSD), which loses Kraken tokenized routing.
            reg_venue = venue_for_symbol(symbol, default="") or venue_for_symbol(query_sym, default="")
            base_urls, hl_book_url = MarketDataProvider.resolve_orderbook_depth_urls(
                query_sym,
                venue=reg_venue,
                multi_venue=bool(reg_venue) or _is_multi_venue_base(query_sym),
            )
            pref_url = self._preferred_url.get(symbol)
            pref_matches_base = bool(
                pref_url
                and (
                    pref_url in base_urls
                    or (
                        "&limit=1000" in pref_url
                        and pref_url.replace("&limit=1000", "&limit=500") in base_urls
                    )
                )
            )
            urls = (
                (
                    pref_url,
                    *(
                        u
                        for u in base_urls
                        if u != pref_url and u != pref_url.replace("&limit=1000", "&limit=500")
                    ),
                )
                if pref_matches_base and pref_url
                else base_urls
            )
            for url in urls:
                try:
                    raw_data = self._fetch_json_dict(
                        url, "depth", lookup_sym, timeout=4, user_agent="mptrade-orderbook/1.0"
                    )
                    if is_tok_sym and isinstance(raw_data, dict) and raw_data.get("error"):
                        err_txt = " ".join(str(x) for x in (raw_data.get("error") or ()))
                        if "Invalid arguments" in err_txt or "Unknown asset pair" in err_txt:
                            # BY DESIGN: Pre-IPO watchlisted xStocks (e.g. ANTxUSD) are not yet listed on Kraken;
                            # back off quietly for 5m instead of spamming WARNING logs every telemetry tick.
                            self._unlisted_xstock_until[tok_sym] = now + 300.0
                            return self._fallback_stale(symbol, cached)
                    normalized_book = MarketDataProvider.normalize_orderbook_payload(raw_data)
                    if normalized_book is None:
                        cached_ts = (
                            safe_nonneg_float(
                                cached[0] if isinstance(cached, tuple) and cached else getattr(cached, "ts", 0.0),
                                0.0,
                            )
                            if cached is not None
                            else 0.0
                        )
                        if (
                            is_tok_sym
                            and (cached is None or (cached_ts > 0.0 and (now - cached_ts) > 600.0))
                            and isinstance(raw_data, dict)
                            and not raw_data.get("error")
                            and isinstance(raw_data.get("result"), dict)
                        ):
                            # BY DESIGN: Pre-IPO Kraken tokenized pairs (e.g. ANTxUSD) return an empty result book
                            # ({'asks': [], 'bids': []}) before trading opens; back off 5m without WARNING spam.
                            self._unlisted_xstock_until[tok_sym] = now + 300.0
                            return self._fallback_stale(symbol, cached)
                        raise ValueError(f"Empty or malformed depth payload for {symbol}")
                    data = normalized_book
                    self._preferred_url[symbol] = url
                    break
                except Exception as url_exc:
                    last_err = url_exc
            if data is None:
                raise last_err if last_err is not None else ValueError(f"No depth data for {symbol}")

            bids = data["bids"]
            asks = data["asks"]
            if not bids or not asks:
                return None

            best_bid, best_ask = round(bids[0][0], 8), round(asks[0][0], 8)
            if best_bid <= 0 or best_ask <= 0 or best_bid > best_ask:
                return None
            mid = safe_pos_float(round((best_bid + best_ask) / 2.0, 8), default=None)
            if mid is None:
                return None

            # Auto-widen Binance Spot limit=500 -> limit=1000 when tight-tick books (e.g. BTCUSDC) saturate
            # 500 levels before reaching +/-1.05%, keeping thin pairs at lighter weight while uncovering +/-1% depth.
            cur_pref_url = self._preferred_url.get(symbol) or ""
            if (
                "api.binance.com/api/v3/depth" in cur_pref_url
                and "&limit=500" in cur_pref_url
                and (
                    (len(bids) >= 500 and bids[-1][0] > mid * 0.9895)
                    or (len(asks) >= 500 and asks[-1][0] < mid * 1.0105)
                )
            ):
                wide_url = cur_pref_url.replace("&limit=500", "&limit=1000")
                try:
                    wide_raw = self._fetch_json_dict(
                        wide_url, "depth_wide", lookup_sym, timeout=4, user_agent="mptrade-orderbook/1.0"
                    )
                    wide_book = MarketDataProvider.normalize_orderbook_payload(wide_raw)
                    if isinstance(wide_book, dict) and wide_book.get("bids") and wide_book.get("asks"):
                        w_bid, w_ask = round(wide_book["bids"][0][0], 8), round(wide_book["asks"][0][0], 8)
                        w_mid = safe_pos_float(round((w_bid + w_ask) / 2.0, 8), default=None)
                        if w_bid > 0 and w_ask >= w_bid and w_mid is not None:
                            bids, asks = wide_book["bids"], wide_book["asks"]
                            best_bid, best_ask, mid = w_bid, w_ask, w_mid
                            self._preferred_url[symbol] = wide_url
                except Exception:
                    pass

            # Hyperliquid's default l2Book returns 20 fine tick levels (~0.02% around mid).
            # Extend deeper levels within the +/-1.5% band via aggregated nSigFigs=4 (and nSigFigs=3 if needed) while preserving top-of-book spread.
            if (
                hl_book_url
                and self._preferred_url.get(symbol) == hl_book_url
                and (bids[-1][0] > mid * 0.992 or asks[-1][0] < mid * 1.008)
            ):
                for sig_figs in (4, 3):
                    if bids[-1][0] <= mid * 0.99 and asks[-1][0] >= mid * 1.01:
                        break
                    try:
                        deep_raw = self._fetch_json_dict(
                            f"{hl_book_url}&nSigFigs={sig_figs}", "depth_agg", lookup_sym, timeout=4, user_agent="mptrade-orderbook/1.0"
                        )
                        deep_book = MarketDataProvider.normalize_orderbook_payload(deep_raw)
                        if isinstance(deep_book, dict):
                            if bids[-1][0] > mid * 0.99:
                                min_b_px = bids[-1][0]
                                b_step = 10.0 ** (math.floor(math.log10(min_b_px)) - sig_figs + 1) if min_b_px > 0 else 0.0
                                b_cutoff = (math.floor(min_b_px / b_step + 1e-9) * b_step - b_step * 1e-6) if b_step > 0 else min_b_px
                                bids = list(bids) + [lv for lv in (deep_book.get("bids") or ()) if lv[0] < b_cutoff]
                            if asks[-1][0] < mid * 1.01:
                                max_a_px = asks[-1][0]
                                a_step = 10.0 ** (math.floor(math.log10(max_a_px)) - sig_figs + 1) if max_a_px > 0 else 0.0
                                a_cutoff = (math.ceil(max_a_px / a_step - 1e-9) * a_step + a_step * 1e-6) if a_step > 0 else max_a_px
                                asks = list(asks) + [lv for lv in (deep_book.get("asks") or ()) if lv[0] > a_cutoff]
                    except Exception:
                        pass

            def _accumulate_side(
                levels: List[Tuple[float, float]], bound: float, bound_1pct: float, is_bid: bool
            ) -> Tuple[float, float, float, float]:
                depth_usd = depth_1pct = max_usd = max_px = wall_nt_usd = wall_nt_px = 0.0
                limit_px = min(bound, bound_1pct) if is_bid else max(bound, bound_1pct)
                min_wall_dist = mid * 0.001  # Prefer structural walls at least 10 bps (0.10%) away from mid over micro-tick touch
                for p, q in levels:
                    if (p < limit_px) if is_bid else (p > limit_px):
                        break
                    val = p * q
                    if (p >= bound) if is_bid else (p <= bound):
                        depth_usd += val
                        if val > max_usd:
                            max_usd, max_px = val, p
                        if abs(p - mid) >= min_wall_dist and val > wall_nt_usd:
                            wall_nt_usd, wall_nt_px = val, p
                    if (p >= bound_1pct) if is_bid else (p <= bound_1pct):
                        depth_1pct += val
                if (
                    wall_nt_usd > 0.0
                    and (wall_nt_usd >= self.whale_wall_usd or max_usd < self.whale_wall_usd)
                    and (wall_nt_usd >= 5000.0 or max_usd < 5000.0)
                ):
                    return depth_usd, depth_1pct, wall_nt_usd, wall_nt_px
                return depth_usd, depth_1pct, max_usd, max_px

            bid_depth_usd, depth_1pct_bid_usd, max_bid_usd, max_bid_px = _accumulate_side(
                bids, mid * (1.0 - self.depth_band_pct / 100.0), mid * 0.99, True
            )
            ask_depth_usd, depth_1pct_ask_usd, max_ask_usd, max_ask_px = _accumulate_side(
                asks, mid * (1.0 + self.depth_band_pct / 100.0), mid * 1.01, False
            )

            total_depth = bid_depth_usd + ask_depth_usd
            imbalance = bid_depth_usd / total_depth if total_depth > 0 else 0.5
            spread_bps = round_spread_bps(best_bid=best_bid, best_ask=best_ask, mid_price=mid) or 0.0
            book_span_pct = 0.0
            is_hl_book = bool(hl_book_url and self._preferred_url.get(symbol) == hl_book_url)
            min_cap_levels = 20 if is_hl_book else 100
            if mid > 0:
                capped_spans = [
                    max(0.0, dist_pct)
                    for side_lvls, dist_pct in (
                        (bids, ((mid - bids[-1][0]) / mid) * 100.0 if bids else 0.0),
                        (asks, ((asks[-1][0] - mid) / mid) * 100.0 if asks else 0.0),
                    )
                    if len(side_lvls) >= min_cap_levels
                ]
                if capped_spans:
                    book_span_pct = round(min(*capped_spans, self.depth_band_pct), 3)

            eff_spot_venue = reg_venue or MarketDataProvider.resolve_venue_for_orderbook_url(
                self._preferred_url.get(symbol)
            )
            real_spot_cvd = self._resolve_real_spot_cvd_15m(query_sym, venue=eff_spot_venue, now=now)
            real_spot_vol: Optional[float] = None
            agg_spot_cvd: Optional[float] = None
            agg_spot_vol: Optional[float] = None
            agg_spot_src: Optional[str] = None
            if real_spot_cvd is not None and eff_spot_venue:
                memo_flow = self._spot_cvd_memo.get((eff_spot_venue, query_sym))
                if isinstance(memo_flow, tuple) and len(memo_flow) >= 3:
                    real_spot_vol = safe_nonneg_float(memo_flow[2], default=None)
                memo_agg = self._agg_spot_cvd_memo.get((eff_spot_venue, query_sym))
                if isinstance(memo_agg, tuple) and len(memo_agg) >= 4:
                    agg_spot_cvd = safe_float(memo_agg[1], default=None)
                    agg_spot_vol = safe_nonneg_float(memo_agg[2], default=None)
                    agg_spot_src = memo_agg[3] or None
            snapshot = OrderbookSnapshot(
                symbol=symbol,
                mid_price=round(mid, 8),
                bid_depth_usd=bid_depth_usd,
                ask_depth_usd=ask_depth_usd,
                imbalance_ratio=round(imbalance, 3),
                largest_bid_wall_usd=round(max_bid_usd, 2),
                largest_bid_wall_price=round(max_bid_px, 8),
                largest_ask_wall_usd=round(max_ask_usd, 2),
                largest_ask_wall_price=round(max_ask_px, 8),
                ts=now,
                best_bid=round(best_bid, 8),
                best_ask=round(best_ask, 8),
                spread_bps=spread_bps,
                depth_1pct_bid_usd=round(depth_1pct_bid_usd, 2),
                depth_1pct_ask_usd=round(depth_1pct_ask_usd, 2),
                book_span_pct=book_span_pct,
                spot_cvd_15m_usd=real_spot_cvd,
                spot_cvd_source="real_spot" if real_spot_cvd is not None else None,
                spot_vol_15m_usd=real_spot_vol,
                agg_spot_cvd_15m_usd=agg_spot_cvd,
                agg_spot_vol_15m_usd=agg_spot_vol,
                agg_spot_cvd_source=agg_spot_src,
            )
            return self._commit_snapshot(symbol, now, snapshot)
        except Exception as e:
            return self._warn_and_fallback_stale(logger, "orderbook depth", symbol, e, cached)

    def _fetch_equity_orderbook(self, symbol: str, now: float) -> Optional[OrderbookSnapshot]:
        """Build an equity OrderbookSnapshot from Nasdaq L1 quote and Yahoo 5m volume-by-price liquidity."""
        resolved = self._resolve_equity_context_bars_and_price(
            symbol, now, include_options=False
        )
        if resolved is None:
            return None
        ctx, bars, mid = resolved
        best_bid = safe_pos_float(ctx.get("best_bid"), default=0.0) or 0.0
        best_ask = safe_pos_float(ctx.get("best_ask"), default=0.0) or 0.0
        quote_mid = (best_bid + best_ask) / 2.0 if (best_bid > 0.0 and best_ask >= best_bid) else 0.0
        stub_quote = (
            best_bid <= 0.0
            or best_ask <= 0.0
            or best_bid > best_ask
            or (best_ask - best_bid) / mid > 0.02
            or abs(quote_mid - mid) / mid > 0.01
        )
        if stub_quote:
            recent_ranges = [
                (b[2] - b[3]) / b[4]
                for b in bars[-78:]
                if b[4] > 0.0 and b[2] > b[3]
            ]
            recent_ranges.sort()
            med_rng = recent_ranges[len(recent_ranges) // 2] if recent_ranges else 0.001
            half_sp_pct = min(max(med_rng * 0.10, 0.0001), 0.0025)
            half_sp = max(0.005, round(mid * half_sp_pct, 4))
            best_bid = round(max(0.01, mid - half_sp), 4)
            best_ask = round(mid + half_sp, 4)
        mid = round((best_bid + best_ask) / 2.0, 4)

        bid_sz_usd = 0.0 if stub_quote else (safe_pos_float(ctx.get("bid_size_sh"), default=0.0) or 0.0) * best_bid
        ask_sz_usd = 0.0 if stub_quote else (safe_pos_float(ctx.get("ask_size_sh"), default=0.0) or 0.0) * best_ask
        band_pct = self.depth_band_pct / 100.0
        lo_band, hi_band = mid * (1.0 - band_pct), mid * (1.0 + band_pct)
        lo_1pct, hi_1pct = mid * 0.99, mid * 1.01

        bid_buckets: Dict[float, float] = {}
        ask_buckets: Dict[float, float] = {}
        tick_step = max(0.001 if mid < 5.0 else 0.01, round(mid * 0.0015, 3 if mid < 5.0 else 2))
        b_cap = min(mid * 0.9985, mid - tick_step)
        a_floor = max(mid * 1.0015, mid + tick_step)
        window_bars = bars[-78:]
        n_wb = len(window_bars)
        for idx, (_t, _o, h_px, l_px, c_px, vol) in enumerate(window_bars):
            if vol <= 0.0 or c_px <= 0.0 or h_px < lo_band or l_px > hi_band:
                continue
            clv = (c_px - l_px) / (h_px - l_px) if h_px > l_px else 0.5
            overlap_frac = (
                max(0.05, min(1.0, (min(h_px, hi_band) - max(l_px, lo_band)) / (h_px - l_px)))
                if h_px > l_px
                else 1.0
            )
            recency_decay = 0.92 ** (n_wb - 1 - idx)
            # Scale in-band 5m bar dollar volume to passive lit-book depth (~0.25% per level, capped against MOO/MOC cross spikes)
            bar_passive_usd = min(vol * c_px * 0.0025, 1_500_000.0) * overlap_frac * recency_decay
            if l_px < mid:
                eff_l = l_px if l_px >= lo_band else 0.5 * (lo_band + min(c_px, b_cap))
                b_px = round(min(max(eff_l, lo_band), b_cap) / tick_step) * tick_step
                if b_px >= mid:
                    b_px = max(0.01, round(mid - tick_step, 4))
                b_val = bar_passive_usd * max(0.15, clv)
                bid_buckets[round(b_px, 4)] = bid_buckets.get(round(b_px, 4), 0.0) + b_val
            if h_px > mid:
                eff_h = h_px if h_px <= hi_band else 0.5 * (hi_band + max(c_px, a_floor))
                a_px = round(max(min(eff_h, hi_band), a_floor) / tick_step) * tick_step
                if a_px <= mid:
                    a_px = round(mid + tick_step, 4)
                a_val = bar_passive_usd * max(0.15, 1.0 - clv)
                ask_buckets[round(a_px, 4)] = ask_buckets.get(round(a_px, 4), 0.0) + a_val

        if bid_sz_usd > 0.0:
            bid_buckets[best_bid] = bid_buckets.get(best_bid, 0.0) + bid_sz_usd
        if ask_sz_usd > 0.0:
            ask_buckets[best_ask] = ask_buckets.get(best_ask, 0.0) + ask_sz_usd
        fallback_depth_usd = min(
            250_000.0,
            max(1000.0, (safe_pos_float(ctx.get("daily_vol_sh"), default=1000.0) or 1000.0) * mid * 0.0005),
        )
        if not bid_buckets:
            bid_buckets[round(mid * 0.995, 4)] = fallback_depth_usd
        if not ask_buckets:
            ask_buckets[round(mid * 1.005, 4)] = fallback_depth_usd

        bid_depth_usd = sum(bid_buckets.values())
        ask_depth_usd = sum(ask_buckets.values())
        d1_bid = max(
            sum(v for p, v in bid_buckets.items() if p >= lo_1pct),
            round(bid_depth_usd * (1.0 / 1.5), 2),
        )
        d1_ask = max(
            sum(v for p, v in ask_buckets.items() if p <= hi_1pct),
            round(ask_depth_usd * (1.0 / 1.5), 2),
        )
        max_bid_px, max_bid_usd = max(bid_buckets.items(), key=lambda kv: kv[1])
        max_ask_px, max_ask_usd = max(ask_buckets.items(), key=lambda kv: kv[1])
        total_depth = bid_depth_usd + ask_depth_usd
        imbalance = bid_depth_usd / total_depth if total_depth > 0.0 else 0.5
        spread_bps = round_spread_bps(best_bid=best_bid, best_ask=best_ask, mid_price=mid) or 0.0

        snapshot = OrderbookSnapshot(
            symbol=symbol,
            mid_price=mid,
            bid_depth_usd=round(bid_depth_usd, 2),
            ask_depth_usd=round(ask_depth_usd, 2),
            imbalance_ratio=round(imbalance, 3),
            largest_bid_wall_usd=round(max_bid_usd, 2),
            largest_bid_wall_price=round(max_bid_px, 4),
            largest_ask_wall_usd=round(max_ask_usd, 2),
            largest_ask_wall_price=round(max_ask_px, 4),
            ts=now,
            best_bid=best_bid,
            best_ask=best_ask,
            spread_bps=spread_bps,
            depth_1pct_bid_usd=round(d1_bid, 2),
            depth_1pct_ask_usd=round(d1_ask, 2),
        )
        return self._commit_snapshot(symbol, now, snapshot)


TelemetryCollectorFactory.register(OrderbookDepthCollector.COLLECTOR_ID, OrderbookDepthCollector)
