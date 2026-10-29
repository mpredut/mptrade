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
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from botcore import _parse_bool_flag, safe_float, safe_nonneg_float, safe_pos_float
from intelligence.macro_sentiment_ssot import (
    fetch_public_json,
    is_cache_fresh,
    resolve_now_ts as _effective_now_ts,
)
from cachepaths import (
    cache_symbol_variants,
    normalize_cache_base_asset,
    resolve_binance_lookup_symbol,
    resolve_cache_dir_path,
    resolve_kraken_lookup_symbol,
    telemetry_cache_file_path,
)
from state_io import read_json_dict, save_json_state

logger = logging.getLogger("intelligence.external.orderbook_depth")


def _extract_base_asset(symbol: str) -> str:
    return normalize_cache_base_asset(symbol)


def _resolve_binance_lookup_symbol(symbol: str) -> str:
    return resolve_binance_lookup_symbol(symbol)


def _symbol_variants(symbol: str) -> List[str]:
    return cache_symbol_variants(symbol)


def _clean_dict_float(data: Dict[str, Any], key: str, default: float) -> float:
    return safe_float(data.get(key) if isinstance(data, dict) else None, default)


def round_spread_bps(
    spread_bps: Any = None,
    *,
    best_bid: Any = None,
    best_ask: Any = None,
    mid_price: Any = None,
) -> Optional[float]:
    """Return bid-ask spread in bps rounded to 4 decimals when < 0.1 bps else 2 decimals."""
    sp = safe_pos_float(spread_bps, default=None)
    if sp is None:
        b_bid = safe_pos_float(best_bid, default=None)
        b_ask = safe_pos_float(best_ask, default=None)
        if b_bid is not None and b_ask is not None and b_ask >= b_bid:
            eff_mid = safe_pos_float(mid_price, default=None) or ((b_bid + b_ask) / 2.0)
            if eff_mid > 0.0:
                sp = ((b_ask - b_bid) / eff_mid) * 10000.0
    if sp is None:
        return None
    return round(sp, 4 if sp < 0.1 else 2)


class _DiskBackedSymbolCollector:
    """Shared in-memory and disk snapshot caching base for external symbol telemetry collectors."""

    _DISK_FILE_PREFIX: str = ""
    _SNAPSHOT_CLS: Any = None

    def __init__(
        self,
        cache_ttl_sec: float,
        default_cache_ttl: float,
        cache_dir: Optional[str],
        disk_ttl_sec: float,
        default_disk_ttl: float,
    ) -> None:
        self.cache_ttl_sec = safe_nonneg_float(cache_ttl_sec, default=default_cache_ttl)
        self.cache_dir = cache_dir
        self.disk_ttl_sec = safe_nonneg_float(disk_ttl_sec, default=default_disk_ttl)
        self._cache: Dict[str, tuple[float, Any]] = {}

    def _effective_cache_dir(self) -> Optional[str]:
        return resolve_cache_dir_path(self.cache_dir)

    def _disk_path(self, symbol: str) -> Optional[str]:
        return telemetry_cache_file_path(self.cache_dir, self._DISK_FILE_PREFIX, symbol)

    _as_bool_flag = staticmethod(_parse_bool_flag)

    def _lookup_cached(self, symbol: str, now: float) -> Tuple[Any, Any]:
        cached = None
        for cand in _symbol_variants(symbol):
            cached = self._cache.get(cand)
            if cached is not None:
                break
        if cached:
            c_ts = cached[0] if isinstance(cached, tuple) and len(cached) >= 2 else getattr(cached, "ts", None)
            if is_cache_fresh(c_ts, now, self.cache_ttl_sec):
                return cached, (cached[1] if isinstance(cached, tuple) else cached)
        return cached, None

    def _load_from_disk(self, symbol: str, now: float, *, ignore_ttl: bool = False) -> Optional[Any]:
        ignore_ttl_b = self._as_bool_flag(ignore_ttl, default=False)
        max_age = max(self.cache_ttl_sec, self.disk_ttl_sec)
        for cand in _symbol_variants(symbol):
            p = self._disk_path(cand)
            if not p:
                continue
            try:
                data = read_json_dict(p, default=None, recover_previous=True)
                if not isinstance(data, dict):
                    continue
                snapshot = self._SNAPSHOT_CLS.from_dict(data)
                if is_cache_fresh(snapshot.ts, now, max_age):
                    self._cache[symbol.strip().upper()] = (snapshot.ts, snapshot)
                    return snapshot
                if ignore_ttl_b:
                    return snapshot
            except Exception as e:
                logger.debug("Failed loading %s snapshot from %s: %s", self._DISK_FILE_PREFIX, p, e)
        return None

    def _save_to_disk(self, snapshot: Any) -> None:
        p = self._disk_path(snapshot.symbol)
        if not p:
            return
        try:
            save_json_state(p, snapshot.to_dict(), indent=2)
        except Exception as e:
            logger.debug("Failed writing %s snapshot to %s: %s", self._DISK_FILE_PREFIX, p, e)

    def _fallback_stale(self, symbol: str, cached: Any) -> Optional[Any]:
        if cached:
            if isinstance(cached, tuple) and len(cached) >= 2:
                return cached[1]
            if self._SNAPSHOT_CLS is not None and isinstance(cached, self._SNAPSHOT_CLS):
                return cached
        return self._load_from_disk(symbol, 0.0, ignore_ttl=True)

    def _record_item(self, item: Any, save_to_disk: bool = False) -> None:
        if isinstance(item, dict):
            item = self._SNAPSHOT_CLS.from_dict(item)
        if item is None or not isinstance(getattr(item, "symbol", None), str) or not item.symbol.strip():
            return
        ts_val = safe_pos_float(getattr(item, "ts", None), default=None) or time.time()
        self._cache[item.symbol.strip().upper()] = (ts_val, item)
        if self._as_bool_flag(save_to_disk, default=False):
            self._save_to_disk(item)

    def _prepare_fetch(
        self,
        symbol: Any,
        allow_network: Any,
        now: Optional[float],
    ) -> Tuple[Optional[str], float, Any, Optional[Any], bool]:
        if not isinstance(symbol, str):
            return None, 0.0, None, None, False
        sym_clean = symbol.strip().upper()
        if not sym_clean:
            return None, 0.0, None, None, False
        now_f = _effective_now_ts(now)
        cached, fresh = self._lookup_cached(sym_clean, now_f)
        if fresh is not None:
            return sym_clean, now_f, cached, fresh, False
        disk_snap = self._load_from_disk(sym_clean, now_f)
        if disk_snap is not None:
            return sym_clean, now_f, cached, disk_snap, False
        allow_net_bool = True if allow_network is None else self._as_bool_flag(allow_network, default=False)
        return sym_clean, now_f, cached, None, allow_net_bool

    def _commit_snapshot(self, symbol: str, now: float, snapshot: Any) -> Any:
        self._cache[symbol] = (now, snapshot)
        self._save_to_disk(snapshot)
        return snapshot

    @staticmethod
    def _fetch_json_dict(
        url: str,
        label: str,
        lookup_sym: str,
        *,
        timeout: float = 5.0,
        user_agent: str = "mptrade-intelligence/1.0",
    ) -> Dict[str, Any]:
        raw_data = fetch_public_json(url, timeout=timeout, user_agent=user_agent)
        if not isinstance(raw_data, dict):
            raise ValueError(f"Unexpected {label} payload for {lookup_sym}: {type(raw_data).__name__}")
        return raw_data

    @staticmethod
    def _fetch_latest_json_row(
        url: str,
        label: str,
        lookup_sym: str,
        *,
        timeout: float = 5.0,
        user_agent: str = "mptrade-whale/1.0",
    ) -> Tuple[Dict[str, Any], List[Any]]:
        rows = fetch_public_json(url, timeout=timeout, user_agent=user_agent)
        if not isinstance(rows, list) or not rows or not isinstance(rows[-1], dict):
            raise ValueError(f"Empty {label} response for {lookup_sym}")
        return rows[-1], rows


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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OrderbookSnapshot:
        if not isinstance(data, dict):
            data = {}
        raw_sym = data.get("symbol")
        sym_str = raw_sym.strip() if isinstance(raw_sym, str) else ""
        mid_price = max(0.0, _clean_dict_float(data, "mid_price", 0.0))
        best_bid = max(0.0, _clean_dict_float(data, "best_bid", 0.0))
        best_ask = max(0.0, _clean_dict_float(data, "best_ask", 0.0))
        spread_bps = round_spread_bps(
            data.get("spread_bps"), best_bid=best_bid, best_ask=best_ask, mid_price=mid_price
        ) or 0.0
        return cls(
            symbol=sym_str,
            mid_price=mid_price,
            bid_depth_usd=max(0.0, _clean_dict_float(data, "bid_depth_usd", 0.0)),
            ask_depth_usd=max(0.0, _clean_dict_float(data, "ask_depth_usd", 0.0)),
            imbalance_ratio=min(1.0, max(0.0, _clean_dict_float(data, "imbalance_ratio", 0.5))),
            largest_bid_wall_usd=max(0.0, _clean_dict_float(data, "largest_bid_wall_usd", 0.0)),
            largest_bid_wall_price=max(0.0, _clean_dict_float(data, "largest_bid_wall_price", 0.0)),
            largest_ask_wall_usd=max(0.0, _clean_dict_float(data, "largest_ask_wall_usd", 0.0)),
            largest_ask_wall_price=max(0.0, _clean_dict_float(data, "largest_ask_wall_price", 0.0)),
            ts=max(0.0, _clean_dict_float(data, "ts", 0.0)),
            best_bid=best_bid,
            best_ask=best_ask,
            spread_bps=spread_bps,
        )


class OrderbookDepthCollector(_DiskBackedSymbolCollector):
    """Collects and analyzes Binance/Kraken Spot order book depth."""

    _DISK_FILE_PREFIX = "orderbook_depth"
    _SNAPSHOT_CLS = OrderbookSnapshot

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
        try:
            data = None
            last_err: Optional[Exception] = None
            kraken_pair = resolve_kraken_lookup_symbol(symbol)
            urls = (
                f"https://api.binance.com/api/v3/depth?symbol={lookup_sym}&limit=500",
                f"https://api.kraken.com/0/public/Depth?pair={kraken_pair}&count=200",
                f"https://fapi.binance.com/fapi/v1/depth?symbol={lookup_sym}&limit=500",
            )
            for url in urls:
                try:
                    raw_data = self._fetch_json_dict(
                        url, "depth", lookup_sym, timeout=4, user_agent="mptrade-orderbook/1.0"
                    )
                    if "bids" not in raw_data and isinstance(raw_data.get("result"), dict):
                        krk_book = next(
                            (v for v in raw_data["result"].values() if isinstance(v, dict) and "bids" in v),
                            None,
                        )
                        if isinstance(krk_book, dict):
                            raw_data = krk_book
                        else:
                            raise ValueError(f"Empty Kraken depth result for {kraken_pair}")
                    data = raw_data
                    break
                except Exception as url_exc:
                    last_err = url_exc
            if data is None:
                raise last_err if last_err is not None else ValueError(f"No depth data for {symbol}")

            def _parse_side(rows: Any) -> List[Tuple[float, float]]:
                if not isinstance(rows, list):
                    return []
                return [
                    (pf, qf)
                    for item in rows
                    if isinstance(item, (list, tuple))
                    and len(item) >= 2
                    and (pf := safe_pos_float(item[0], default=None)) is not None
                    and (qf := safe_pos_float(item[1], default=None)) is not None
                ]

            bids = _parse_side(data.get("bids"))
            asks = _parse_side(data.get("asks"))
            if not bids or not asks:
                return None

            bids.sort(key=lambda x: x[0], reverse=True)
            asks.sort(key=lambda x: x[0])

            best_bid, best_ask = bids[0][0], asks[0][0]
            if best_bid <= 0 or best_ask <= 0 or best_bid > best_ask:
                return None
            mid = safe_pos_float((best_bid + best_ask) / 2.0, default=None)
            if mid is None:
                return None

            def _accumulate_side(
                levels: List[Tuple[float, float]], bound: float, is_bid: bool
            ) -> Tuple[float, float, float]:
                depth_usd = max_usd = max_px = 0.0
                for p, q in levels:
                    if (p < bound) if is_bid else (p > bound):
                        break
                    val = p * q
                    depth_usd += val
                    if val > max_usd:
                        max_usd, max_px = val, p
                return depth_usd, max_usd, max_px

            bid_depth_usd, max_bid_usd, max_bid_px = _accumulate_side(
                bids, mid * (1.0 - self.depth_band_pct / 100.0), True
            )
            ask_depth_usd, max_ask_usd, max_ask_px = _accumulate_side(
                asks, mid * (1.0 + self.depth_band_pct / 100.0), False
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
            )
            return self._commit_snapshot(symbol, now, snapshot)
        except Exception as e:
            logger.warning("Could not fetch orderbook depth for %s: %s", symbol, e)
            return self._fallback_stale(symbol, cached)

    def record_snapshot(self, snapshot: OrderbookSnapshot, save_to_disk: bool = False) -> None:
        """Inject snapshot directly (useful for tests and synthetic feeds)."""
        self._record_item(snapshot, save_to_disk=save_to_disk)
