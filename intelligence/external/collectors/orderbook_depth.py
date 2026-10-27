"""Order book depth imbalance and whale limit walls collector.

Analyzes the top 1-2% of the bid/ask order book:
- Depth Imbalance Ratio: Bid Liquidity / (Bid Liquidity + Ask Liquidity)
- Whale Walls: Detects unusually large single limit orders placed on the book.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from state_io import read_recovering_previous

logger = logging.getLogger("intelligence.external.orderbook_depth")


def _extract_base_asset(symbol: str) -> str:
    if not isinstance(symbol, str):
        return ""
    s = symbol.upper().strip()
    if s == "USDCUSDT":
        return s
    if "/" in s:
        return s.split("/", 1)[0].strip()
    if "-" in s:
        return s.split("-", 1)[0].strip()
    if "_" in s:
        return s.split("_", 1)[0].strip()
    for q in ("USDC", "USDT", "FDUSD", "USD", "EUR", "RON", "GBP"):
        if s.endswith(q) and len(s) > len(q):
            return s[: -len(q)]
    return s


def _resolve_binance_lookup_symbol(symbol: str) -> str:
    if not isinstance(symbol, str):
        return ""
    s = symbol.upper().strip()
    if s == "USDCUSDT":
        return s
    base = _extract_base_asset(s)
    if base and base != s:
        return f"{base}USDT"
    return s.replace("USDC", "USDT") if "USDC" in s else s


def _symbol_variants(symbol: str) -> List[str]:
    if not isinstance(symbol, str):
        return []
    s = symbol.upper().strip()
    safe_s = s.replace("/", "").replace("-", "").replace("_", "")
    variants: List[str] = []
    for cand in (safe_s, s):
        if cand and cand not in variants:
            variants.append(cand)
    if safe_s != "USDCUSDT":
        base = _extract_base_asset(s)
        if base:
            for suffix in ("USDC", "USDT", "USD", "EUR"):
                cand = f"{base}{suffix}"
                if cand not in variants:
                    variants.append(cand)
    return variants


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
        import math
        if not isinstance(data, dict):
            data = {}

        def _f(key: str, default: float) -> float:
            raw = data.get(key)
            if raw is None or isinstance(raw, bool):
                return default
            try:
                val = float(raw)
                return val if math.isfinite(val) else default
            except (TypeError, ValueError):
                return default

        raw_sym = data.get("symbol")
        sym_str = raw_sym.strip() if isinstance(raw_sym, str) else ""
        return cls(
            symbol=sym_str,
            mid_price=max(0.0, _f("mid_price", 0.0)),
            bid_depth_usd=max(0.0, _f("bid_depth_usd", 0.0)),
            ask_depth_usd=max(0.0, _f("ask_depth_usd", 0.0)),
            imbalance_ratio=min(1.0, max(0.0, _f("imbalance_ratio", 0.5))),
            largest_bid_wall_usd=max(0.0, _f("largest_bid_wall_usd", 0.0)),
            largest_bid_wall_price=max(0.0, _f("largest_bid_wall_price", 0.0)),
            largest_ask_wall_usd=max(0.0, _f("largest_ask_wall_usd", 0.0)),
            largest_ask_wall_price=max(0.0, _f("largest_ask_wall_price", 0.0)),
            ts=max(0.0, _f("ts", 0.0)),
        )


class OrderbookDepthCollector:
    """Collects and analyzes Binance Spot order book depth."""

    def __init__(
        self,
        cache_ttl_sec: float = 5.0,
        depth_band_pct: float = 1.5,     # Analyze depth within 1.5% of mid-price
        whale_wall_usd: float = 500_000.0,
        cache_dir: Optional[str] = "cachedb",
        disk_ttl_sec: float = 60.0,
    ) -> None:
        import math

        def _clean_nonneg(val: Any, default: float) -> float:
            if val is None or isinstance(val, bool):
                return default
            try:
                vf = float(val)
                return vf if math.isfinite(vf) and vf >= 0.0 else default
            except (TypeError, ValueError, OverflowError):
                return default

        def _clean_pos(val: Any, default: float) -> float:
            if val is None or isinstance(val, bool):
                return default
            try:
                vf = float(val)
                return vf if math.isfinite(vf) and vf > 0.0 else default
            except (TypeError, ValueError, OverflowError):
                return default

        self.cache_ttl_sec = _clean_nonneg(cache_ttl_sec, 5.0)
        self.depth_band_pct = _clean_pos(depth_band_pct, 1.5)
        self.whale_wall_usd = _clean_pos(whale_wall_usd, 500_000.0)
        self.cache_dir = cache_dir
        self.disk_ttl_sec = _clean_nonneg(disk_ttl_sec, 60.0)
        self._cache: Dict[str, tuple[float, OrderbookSnapshot]] = {}

    def _disk_path(self, symbol: str) -> Optional[str]:
        if not self.cache_dir or not isinstance(symbol, str) or not symbol.strip():
            return None
        cache_dir = self.cache_dir
        if cache_dir == "cachedb":
            try:
                from cachepaths import get_cachedb_dir
                cache_dir = get_cachedb_dir()
            except Exception:
                cache_dir = os.environ.get("MPTRADE_CACHEDB_DIR", "cachedb")
        safe_sym = symbol.upper().strip().replace("/", "").replace("-", "").replace("_", "")
        return os.path.join(cache_dir, f"orderbook_depth_{safe_sym}.json")

    @staticmethod
    def _as_bool_flag(val: Any, default: bool = False) -> bool:
        import math
        if val is None:
            return default
        if isinstance(val, str):
            s = val.strip().lower()
            if s in ("1", "1.0", "true", "yes", "on"):
                return True
            if s in ("0", "0.0", "false", "no", "off", "disabled", "nan", ""):
                return False
            return default
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return math.isfinite(float(val)) and float(val) != 0.0
        return bool(val)

    def _load_from_disk(self, symbol: str, now: float) -> Optional[OrderbookSnapshot]:
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
                snapshot = OrderbookSnapshot.from_dict(data)
                if snapshot.ts > 0.0 and 0.0 <= (now - snapshot.ts) < max_age:
                    self._cache[symbol.strip().upper()] = (snapshot.ts, snapshot)
                    return snapshot
            except Exception as e:
                logger.debug("Failed loading orderbook snapshot from %s: %s", p, e)
        return None

    def _save_to_disk(self, snapshot: OrderbookSnapshot) -> None:
        p = self._disk_path(snapshot.symbol)
        if not p:
            return
        try:
            from state_io import save_json_state
            save_json_state(p, snapshot.to_dict(), indent=2)
        except Exception as e:
            logger.debug("Failed writing orderbook snapshot to %s: %s", p, e)

    def fetch(
        self,
        symbol: str,
        allow_network: bool = True,
        now: Optional[float] = None,
    ) -> Optional[OrderbookSnapshot]:
        """Fetch spot depth snapshot from Binance public API."""
        import math
        if not isinstance(symbol, str):
            return None
        symbol = symbol.strip().upper()
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
            elif isinstance(cached, OrderbookSnapshot):
                if 0.0 <= (now - cached.ts) < self.cache_ttl_sec:
                    return cached

        disk_snap = self._load_from_disk(symbol, now)
        if disk_snap is not None:
            return disk_snap

        if not self._as_bool_flag(allow_network, default=True):
            return None

        lookup_sym = _resolve_binance_lookup_symbol(symbol)
        try:
            url = f"https://api.binance.com/api/v3/depth?symbol={lookup_sym}&limit=50"
            req = urllib.request.Request(url, headers={"User-Agent": "mptrade-orderbook/1.0"})
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError(f"Unexpected depth payload for {lookup_sym}: {type(data).__name__}")

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
            self._cache[symbol] = (now, snapshot)
            self._save_to_disk(snapshot)
            return snapshot
        except Exception as e:
            logger.warning("Could not fetch orderbook depth for %s: %s", symbol, e)
            if cached:
                if isinstance(cached, tuple) and len(cached) >= 2:
                    return cached[1]
                if isinstance(cached, OrderbookSnapshot):
                    return cached
            for cand in _symbol_variants(symbol):
                p = self._disk_path(cand)
                if p:
                    try:
                        data = read_recovering_previous(p, json.loads, default=None)
                        if isinstance(data, dict):
                            return OrderbookSnapshot.from_dict(data)
                    except Exception:
                        pass
            return None

    def record_snapshot(self, snapshot: OrderbookSnapshot, save_to_disk: bool = False) -> None:
        """Inject snapshot directly (useful for tests and synthetic feeds)."""
        if snapshot is None or not isinstance(getattr(snapshot, "symbol", None), str) or not snapshot.symbol.strip():
            return
        self._cache[snapshot.symbol.strip().upper()] = (snapshot.ts, snapshot)
        if self._as_bool_flag(save_to_disk, default=False):
            self._save_to_disk(snapshot)
