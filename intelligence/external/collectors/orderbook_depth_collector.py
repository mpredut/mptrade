"""Order book depth imbalance and whale limit walls collector.

Analyzes the top 1-2% of the bid/ask order book:
- Depth Imbalance Ratio: Bid Liquidity / (Bid Liquidity + Ask Liquidity)
- Whale Walls: Detects unusually large single limit orders placed on the book.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from botcore import _parse_bool_flag, safe_float, safe_nonneg_float, safe_pos_float
from intelligence.macro_sentiment_ssot import (
    clean_scale,
    fetch_public_json,
    is_cache_fresh,
    resolve_now_ts as _effective_now_ts,
)
from cachepaths import (
    cache_symbol_variants,
    canonical_symbol_key,
    compatible_quote_suffixes,
    normalize_cache_base_asset,
    resolve_binance_lookup_symbol,
    resolve_cache_dir_path,
    resolve_kraken_lookup_symbol,
    telemetry_cache_file_path,
)
from providers.base import MarketDataProvider, resolve_ssot_price_from_cache
from state_io import read_json_dict, read_json_dict_recovering, save_json_state

logger = logging.getLogger("intelligence.external.orderbook_depth")

from intelligence.external.collectors.base_collector import (
    _DiskBackedSymbolCollector,
    _REGISTRY_SPECS_CACHE,
    _cached_registry_specs,
    _clean_dict_float,
    _clean_snapshot_symbol,
    _clean_snapshot_ts,
    _extract_base_asset,
    _fetch_hyperliquid_asset_ctx,
    _has_hyperliquid_instrument,
    _is_multi_venue_base,
    _resolve_binance_lookup_symbol,
    _resolve_configured_venue,
    _symbol_variants,
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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

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
        )


class OrderbookDepthCollector(_DiskBackedSymbolCollector):
    """Collects and analyzes Binance/Kraken Spot order book depth."""

    _DISK_FILE_PREFIX = "orderbook_depth"
    _SNAPSHOT_CLS = OrderbookSnapshot
    _QUOTE_ISOLATED = True

    def __init__(
        self,
        cache_ttl_sec: float = 5.0,
        depth_band_pct: float = 1.5,     # Analyze depth within 1.5% of mid-price
        whale_wall_usd: float = 500_000.0,
        cache_dir: Optional[str] = "cachedb",
        disk_ttl_sec: float = 60.0,
    ) -> None:
        super().__init__(cache_ttl_sec, 5.0, cache_dir, disk_ttl_sec, 60.0)
        self.depth_band_pct = safe_pos_float(depth_band_pct, default=1.5) or 1.5
        self.whale_wall_usd = safe_pos_float(whale_wall_usd, default=500_000.0) or 500_000.0
        self._preferred_url: Dict[str, str] = {}

    def fetch(
        self,
        symbol: str,
        allow_network: bool = True,
        now: Optional[float] = None,
    ) -> Optional[OrderbookSnapshot]:
        """Fetch spot depth snapshot from Binance or Kraken public API."""
        symbol, now, cached, ready, should_fetch = self._prepare_fetch(symbol, allow_network, now)
        if not should_fetch or symbol is None:
            return ready

        lookup_sym = _resolve_binance_lookup_symbol(symbol)
        canon_sym = canonical_symbol_key(symbol)
        try:
            data = None
            last_err: Optional[Exception] = None
            kraken_pair = resolve_kraken_lookup_symbol(symbol)
            hl_book_url = ""
            if canon_sym.endswith("EUR"):
                base_asset = _extract_base_asset(canon_sym)
                bin_eur_sym = f"{base_asset}EUR" if base_asset else canon_sym
                krk_eur_pair = (
                    "XBTEUR"
                    if base_asset == "BTC"
                    else ("XDGEUR" if base_asset == "DOGE" else bin_eur_sym)
                )
                base_urls: Tuple[str, ...] = (
                    f"https://api.binance.com/api/v3/depth?symbol={bin_eur_sym}&limit=500",
                    f"https://api.kraken.com/0/public/Depth?pair={krk_eur_pair}&count=200",
                )
            else:
                usdc_sym = f"{_extract_base_asset(canon_sym)}USDC" if canon_sym.endswith(("USD", "USDC")) else ""
                spot_urls = (
                    (
                        f"https://api.binance.com/api/v3/depth?symbol={usdc_sym}&limit=500",
                        f"https://api.binance.com/api/v3/depth?symbol={lookup_sym}&limit=500",
                    )
                    if (usdc_sym and usdc_sym != lookup_sym)
                    else (f"https://api.binance.com/api/v3/depth?symbol={lookup_sym}&limit=500",)
                )
                hl_coin = _extract_base_asset(canon_sym)
                hl_book_url = f"https://api.hyperliquid.xyz/info?type=l2Book&coin={hl_coin}" if hl_coin else ""
                venue = _resolve_configured_venue(canon_sym)
                multi_venue = _is_multi_venue_base(canon_sym)
                if venue == "kraken":
                    base_urls = (
                        (f"https://api.kraken.com/0/public/Depth?pair={kraken_pair}&count=200",)
                        if multi_venue
                        else (
                            f"https://api.kraken.com/0/public/Depth?pair={kraken_pair}&count=200",
                            *spot_urls,
                            f"https://fapi.binance.com/fapi/v1/depth?symbol={lookup_sym}&limit=500",
                            *((hl_book_url,) if hl_book_url else ()),
                        )
                    )
                elif venue == "hyperliquid" and hl_book_url:
                    base_urls = (
                        (hl_book_url,)
                        if multi_venue
                        else (
                            hl_book_url,
                            *spot_urls,
                            f"https://api.kraken.com/0/public/Depth?pair={kraken_pair}&count=200",
                            f"https://fapi.binance.com/fapi/v1/depth?symbol={lookup_sym}&limit=500",
                        )
                    )
                else:
                    base_urls = (
                        (*spot_urls, f"https://fapi.binance.com/fapi/v1/depth?symbol={lookup_sym}&limit=500")
                        if multi_venue
                        else (
                            *spot_urls,
                            f"https://api.kraken.com/0/public/Depth?pair={kraken_pair}&count=200",
                            f"https://fapi.binance.com/fapi/v1/depth?symbol={lookup_sym}&limit=500",
                            *((hl_book_url,) if hl_book_url else ()),
                        )
                    )
            pref_url = self._preferred_url.get(symbol)
            urls = (
                (pref_url, *(u for u in base_urls if u != pref_url))
                if (pref_url and pref_url in base_urls)
                else base_urls
            )
            for url in urls:
                try:
                    raw_data = self._fetch_json_dict(
                        url, "depth", lookup_sym, timeout=4, user_agent="mptrade-orderbook/1.0"
                    )
                    normalized_book = MarketDataProvider.normalize_orderbook_payload(raw_data)
                    if normalized_book is None:
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

            best_bid, best_ask = bids[0][0], asks[0][0]
            if best_bid <= 0 or best_ask <= 0 or best_bid > best_ask:
                return None
            mid = safe_pos_float((best_bid + best_ask) / 2.0, default=None)
            if mid is None:
                return None

            # Hyperliquid's default l2Book returns 20 fine tick levels (~0.02% around mid).
            # Extend deeper levels within the ±1.5% band via aggregated nSigFigs=4 (and nSigFigs=3 if needed) while preserving top-of-book spread.
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
                best_bid=best_bid,
                best_ask=best_ask,
                spread_bps=spread_bps,
                depth_1pct_bid_usd=round(depth_1pct_bid_usd, 2),
                depth_1pct_ask_usd=round(depth_1pct_ask_usd, 2),
            )
            return self._commit_snapshot(symbol, now, snapshot)
        except Exception as e:
            logger.warning("Could not fetch orderbook depth for %s: %s", symbol, e)
            return self._fallback_stale(symbol, cached)

    def record_snapshot(self, snapshot: OrderbookSnapshot, save_to_disk: bool = False) -> None:
        """Inject snapshot directly (useful for tests and synthetic feeds)."""
        self._record_item(snapshot, save_to_disk=save_to_disk)
