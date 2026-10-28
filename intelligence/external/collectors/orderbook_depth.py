"""Order book depth imbalance and whale limit walls collector.

Analyzes the top 1-2% of the bid/ask order book:
- Depth Imbalance Ratio: Bid Liquidity / (Bid Liquidity + Ask Liquidity)
- Whale Walls: Detects unusually large single limit orders placed on the book.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from botcore import _parse_bool_flag
from cachepaths import (
    cache_symbol_variants,
    normalize_cache_base_asset,
    resolve_binance_lookup_symbol,
    resolve_cache_dir_path,
    resolve_kraken_lookup_symbol,
    telemetry_cache_file_path,
)
from state_io import read_recovering_previous, save_json_state

logger = logging.getLogger("intelligence.external.orderbook_depth")


def _extract_base_asset(symbol: str) -> str:
    return normalize_cache_base_asset(symbol)


def _resolve_binance_lookup_symbol(symbol: str) -> str:
    return resolve_binance_lookup_symbol(symbol)


def _symbol_variants(symbol: str) -> List[str]:
    return cache_symbol_variants(symbol)


def _clean_dict_float(data: Dict[str, Any], key: str, default: float) -> float:
    raw = data.get(key) if isinstance(data, dict) else None
    if raw is None or isinstance(raw, bool):
        return default
    try:
        val = float(raw)
        return val if math.isfinite(val) else default
    except (TypeError, ValueError, OverflowError):
        return default


def _clean_nonneg_float(val: Any, default: float) -> float:
    if val is None or isinstance(val, bool):
        return default
    try:
        vf = float(val)
        return vf if math.isfinite(vf) and vf >= 0.0 else default
    except (TypeError, ValueError, OverflowError):
        return default


def _clean_pos_float(val: Any, default: float) -> float:
    if val is None or isinstance(val, bool):
        return default
    try:
        vf = float(val)
        return vf if math.isfinite(vf) and vf > 0.0 else default
    except (TypeError, ValueError, OverflowError):
        return default


def _effective_now_ts(now: Optional[float]) -> float:
    if now is not None and not isinstance(now, bool):
        try:
            now_f = float(now)
            if math.isfinite(now_f) and now_f > 0.0:
                return now_f
        except (TypeError, ValueError, OverflowError):
            pass
    return time.time()


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
        self.cache_ttl_sec = _clean_nonneg_float(cache_ttl_sec, default_cache_ttl)
        self.cache_dir = cache_dir
        self.disk_ttl_sec = _clean_nonneg_float(disk_ttl_sec, default_disk_ttl)
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
            if isinstance(cached, tuple) and len(cached) >= 2:
                c_ts = cached[0]
                if isinstance(c_ts, (int, float)) and not isinstance(c_ts, bool) and math.isfinite(float(c_ts)):
                    if 0.0 <= (now - float(c_ts)) < self.cache_ttl_sec:
                        return cached, cached[1]
            elif self._SNAPSHOT_CLS is not None and isinstance(cached, self._SNAPSHOT_CLS):
                c_ts = getattr(cached, "ts", None)
                if isinstance(c_ts, (int, float)) and not isinstance(c_ts, bool) and math.isfinite(float(c_ts)):
                    if 0.0 <= (now - float(c_ts)) < self.cache_ttl_sec:
                        return cached, cached
        return cached, None

    def _load_from_disk(self, symbol: str, now: float, *, ignore_ttl: bool = False) -> Optional[Any]:
        ignore_ttl_b = self._as_bool_flag(ignore_ttl, default=False)
        max_age = max(self.cache_ttl_sec, self.disk_ttl_sec)
        for cand in _symbol_variants(symbol):
            p = self._disk_path(cand)
            if not p:
                continue
            try:
                data = read_recovering_previous(p, json.loads, default=None)
                if not isinstance(data, dict):
                    continue
                snapshot = self._SNAPSHOT_CLS.from_dict(data)
                if snapshot.ts > 0.0 and 0.0 <= (now - snapshot.ts) < max_age:
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
        for cand in _symbol_variants(symbol):
            p = self._disk_path(cand)
            if p:
                try:
                    data = read_recovering_previous(p, json.loads, default=None)
                    if isinstance(data, dict):
                        return self._SNAPSHOT_CLS.from_dict(data)
                except Exception:
                    pass
        return None

    def _record_item(self, item: Any, save_to_disk: bool = False) -> None:
        if isinstance(item, dict):
            item = self._SNAPSHOT_CLS.from_dict(item)
        if item is None or not isinstance(getattr(item, "symbol", None), str) or not item.symbol.strip():
            return
        raw_ts = getattr(item, "ts", None)
        ts_val = (
            float(raw_ts)
            if (
                isinstance(raw_ts, (int, float))
                and not isinstance(raw_ts, bool)
                and math.isfinite(float(raw_ts))
                and float(raw_ts) > 0.0
            )
            else time.time()
        )
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
        return sym_clean, now_f, cached, None, self._as_bool_flag(allow_network, default=True)

    def _commit_snapshot(self, symbol: str, now: float, snapshot: Any) -> Any:
        self._cache[symbol] = (now, snapshot)
        self._save_to_disk(snapshot)
        return snapshot


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
        if not isinstance(data, dict):
            data = {}
        raw_sym = data.get("symbol")
        sym_str = raw_sym.strip() if isinstance(raw_sym, str) else ""
        return cls(
            symbol=sym_str,
            mid_price=max(0.0, _clean_dict_float(data, "mid_price", 0.0)),
            bid_depth_usd=max(0.0, _clean_dict_float(data, "bid_depth_usd", 0.0)),
            ask_depth_usd=max(0.0, _clean_dict_float(data, "ask_depth_usd", 0.0)),
            imbalance_ratio=min(1.0, max(0.0, _clean_dict_float(data, "imbalance_ratio", 0.5))),
            largest_bid_wall_usd=max(0.0, _clean_dict_float(data, "largest_bid_wall_usd", 0.0)),
            largest_bid_wall_price=max(0.0, _clean_dict_float(data, "largest_bid_wall_price", 0.0)),
            largest_ask_wall_usd=max(0.0, _clean_dict_float(data, "largest_ask_wall_usd", 0.0)),
            largest_ask_wall_price=max(0.0, _clean_dict_float(data, "largest_ask_wall_price", 0.0)),
            ts=max(0.0, _clean_dict_float(data, "ts", 0.0)),
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
        self.depth_band_pct = _clean_pos_float(depth_band_pct, 1.5)
        self.whale_wall_usd = _clean_pos_float(whale_wall_usd, 500_000.0)

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
                f"https://api.binance.com/api/v3/depth?symbol={lookup_sym}&limit=50",
                f"https://api.kraken.com/0/public/Depth?pair={kraken_pair}&count=50",
                f"https://fapi.binance.com/fapi/v1/depth?symbol={lookup_sym}&limit=50",
            )
            for url in urls:
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "mptrade-orderbook/1.0"})
                    with urllib.request.urlopen(req, timeout=4) as resp:
                        raw_data = json.loads(resp.read().decode("utf-8"))
                    if not isinstance(raw_data, dict):
                        raise ValueError(f"Unexpected depth payload for {lookup_sym}: {type(raw_data).__name__}")
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
                out: List[Tuple[float, float]] = []
                if not isinstance(rows, list):
                    return out
                for item in rows:
                    if not isinstance(item, (list, tuple)) or len(item) < 2:
                        continue
                    p, q = item[0], item[1]
                    if p is None or q is None or isinstance(p, bool) or isinstance(q, bool):
                        continue
                    try:
                        pf, qf = float(p), float(q)
                        if math.isfinite(pf) and pf > 0.0 and math.isfinite(qf) and qf > 0.0:
                            out.append((pf, qf))
                    except (TypeError, ValueError, OverflowError):
                        continue
                return out

            bids = _parse_side(data.get("bids"))
            asks = _parse_side(data.get("asks"))

            if not bids or not asks:
                return None

            bids.sort(key=lambda x: x[0], reverse=True)
            asks.sort(key=lambda x: x[0])

            best_bid = bids[0][0]
            best_ask = asks[0][0]
            if best_bid <= 0 or best_ask <= 0 or best_bid > best_ask:
                return None
            mid = (best_bid + best_ask) / 2.0
            if not math.isfinite(mid) or mid <= 0:
                return None

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
            return self._commit_snapshot(symbol, now, snapshot)
        except Exception as e:
            logger.warning("Could not fetch orderbook depth for %s: %s", symbol, e)
            return self._fallback_stale(symbol, cached)

    def record_snapshot(self, snapshot: OrderbookSnapshot, save_to_disk: bool = False) -> None:
        """Inject snapshot directly (useful for tests and synthetic feeds)."""
        self._record_item(snapshot, save_to_disk=save_to_disk)
