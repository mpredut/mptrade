"""Order book depth imbalance and whale limit walls collector.

Analyzes the top 1-2% of the bid/ask order book:
- Depth Imbalance Ratio: Bid Liquidity / (Bid Liquidity + Ask Liquidity)
- Whale Walls: Detects unusually large single limit orders placed on the book.
"""

from __future__ import annotations

import logging
import time
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

_extract_base_asset = normalize_cache_base_asset
_resolve_binance_lookup_symbol = resolve_binance_lookup_symbol
_symbol_variants = cache_symbol_variants


_REGISTRY_SPECS_CACHE: Tuple[float, Tuple[Any, ...]] = (0.0, ())


def _cached_registry_specs() -> Tuple[Any, ...]:
    """Return InstrumentSpec items from ``instrument_registry.load_registry()``, cached by file mtime."""
    global _REGISTRY_SPECS_CACHE
    import instrument_registry as _ir

    load_fn = getattr(_ir, "load_registry", None)
    if not callable(load_fn):
        return ()
    def_path = getattr(_ir, "DEFAULT_PATH", None)
    mtime = 0.0
    if getattr(load_fn, "__module__", "") == "instrument_registry" and def_path is not None:
        try:
            mtime = os.path.getmtime(str(def_path))
        except OSError:
            mtime = -1.0
        if mtime > 0.0 and _REGISTRY_SPECS_CACHE[0] == mtime and _REGISTRY_SPECS_CACHE[1]:
            return _REGISTRY_SPECS_CACHE[1]
    reg = load_fn()
    if isinstance(reg, dict):
        specs = tuple(reg.values())
    elif hasattr(reg, "all") and callable(reg.all):
        specs = tuple(reg.all(enabled_only=False))
    else:
        specs = ()
    if mtime > 0.0:
        _REGISTRY_SPECS_CACHE = (mtime, specs)
    return specs


def _resolve_configured_venue(symbol: str) -> str:
    """Return the SSOT configured provider ('hyperliquid', 'kraken', or 'binance') for symbol."""
    canon_sym = canonical_symbol_key(symbol)
    if not canon_sym:
        return "binance"
    try:
        specs = _cached_registry_specs()
        for enabled_only in (True, False):
            for inst in specs:
                if enabled_only and not getattr(inst, "enabled", True):
                    continue
                if str(getattr(inst, "symbol", "")).upper() == canon_sym and getattr(inst, "provider", None):
                    return str(inst.provider).strip().lower()
    except Exception:
        pass
    if canon_sym in ("HYPEUSD", "TAOUSD", "ADAUSD"):
        return "kraken"
    if canon_sym == "HYPEUSDC":
        return "hyperliquid"
    return "binance"


def _is_multi_venue_base(symbol: str) -> bool:
    """Return True when the symbol's base asset is tracked across multiple venues/quotes."""
    base = _extract_base_asset(symbol)
    if not base:
        return False
    try:
        specs = _cached_registry_specs()
        venues = {
            str(inst.provider).strip().lower()
            for inst in specs
            if str(getattr(inst, "base", "")).upper() == base and getattr(inst, "provider", None)
        }
        if base in ("HYPE", "TAO", "ADA"):
            venues.add("kraken")
        return len(venues) > 1
    except Exception:
        return base in ("HYPE", "TAO", "ADA")


def _has_hyperliquid_instrument(symbol: str) -> bool:
    """Return True when the symbol or its base asset is configured on Hyperliquid in SSOT."""
    if _resolve_configured_venue(symbol) == "hyperliquid":
        return True
    base = _extract_base_asset(symbol)
    if not base:
        return False
    try:
        specs = _cached_registry_specs()
        return any(
            str(getattr(inst, "base", "")).upper() == base
            and str(getattr(inst, "provider", "")).strip().lower() == "hyperliquid"
            for inst in specs
        )
    except Exception:
        return base == "HYPE"


def _clean_dict_float(
    data: Dict[str, Any],
    key: str,
    default: float = 0.0,
    alt_key: Optional[str] = None,
    *,
    nonneg: bool = False,
) -> float:
    if not isinstance(data, dict):
        return max(0.0, default) if nonneg else default
    fb = _clean_dict_float(data, alt_key, default) if alt_key else default
    val = safe_float(data.get(key), fb)
    return max(0.0, val) if nonneg else val


def _clean_snapshot_symbol(data: Dict[str, Any], fallback: str = "") -> str:
    raw = (data.get("symbol") if isinstance(data, dict) else None) or fallback
    return raw.strip() if isinstance(raw, str) else ""


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
        b_bid, b_ask = safe_pos_float(best_bid, default=None), safe_pos_float(best_ask, default=None)
        if b_bid is not None and b_ask is not None and b_ask >= b_bid:
            eff_mid = safe_pos_float(mid_price, default=None) or ((b_bid + b_ask) / 2.0)
            if eff_mid > 0.0:
                sp = ((b_ask - b_bid) / eff_mid) * 10000.0
    return round(sp, 4 if sp < 0.1 else 2) if sp is not None else None


class _DiskBackedSymbolCollector:
    """Shared in-memory and disk snapshot caching base for external symbol telemetry collectors."""

    _DISK_FILE_PREFIX: str = ""
    _SNAPSHOT_CLS: Any = None
    _QUOTE_ISOLATED: bool = False

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

    def _candidate_variants(self, symbol: str) -> List[Any]:
        if self._QUOTE_ISOLATED:
            quotes = compatible_quote_suffixes(symbol, default_usd=("USDC", "USDT", "USD", "FDUSD"))
            return _symbol_variants(symbol, quotes=quotes)
        return _symbol_variants(symbol)

    def _lookup_cached(self, symbol: str, now: float) -> Tuple[Any, Any]:
        cached = next((self._cache[c] for c in self._candidate_variants(symbol) if c in self._cache), None)
        if cached:
            c_ts = cached[0] if isinstance(cached, tuple) and len(cached) >= 2 else getattr(cached, "ts", None)
            if is_cache_fresh(c_ts, now, self.cache_ttl_sec):
                return cached, (cached[1] if isinstance(cached, tuple) else cached)
        return cached, None

    def _load_from_disk(self, symbol: str, now: float, *, ignore_ttl: bool = False) -> Optional[Any]:
        ignore_ttl_b = self._as_bool_flag(ignore_ttl, default=False)
        max_age = max(self.cache_ttl_sec, self.disk_ttl_sec)
        for cand in self._candidate_variants(symbol):
            p = self._disk_path(cand)
            if not p:
                continue
            try:
                data = read_json_dict_recovering(p, default=None)
                if not isinstance(data, dict):
                    continue
                try:
                    snapshot = self._SNAPSHOT_CLS.from_dict(data, cand)
                except TypeError:
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
        if p := self._disk_path(snapshot.symbol):
            try:
                save_json_state(p, snapshot.to_dict(), indent=2)
            except Exception as e:
                logger.debug("Failed writing %s snapshot to %s: %s", self._DISK_FILE_PREFIX, p, e)

    def _fallback_stale(self, symbol: str, cached: Any) -> Optional[Any]:
        if isinstance(cached, tuple) and len(cached) >= 2:
            return cached[1]
        if cached and self._SNAPSHOT_CLS is not None and isinstance(cached, self._SNAPSHOT_CLS):
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

    def _sync_existing_sibling_disk_file(self, sym_clean: str, snapshot: Any) -> None:
        if not (p_exact := self._disk_path(sym_clean)):
            return
        try:
            import os as _os

            if not _os.path.isfile(p_exact):
                return
            cur_d = read_json_dict_recovering(p_exact, default=None)
            cur_ts = safe_pos_float(cur_d.get("ts") if isinstance(cur_d, dict) else None, default=0.0) or 0.0
            snap_ts = safe_pos_float(getattr(snapshot, "ts", None), default=0.0) or 0.0
            if snap_ts > cur_ts + 5.0 and hasattr(snapshot, "to_dict"):
                d_copy = dict(snapshot.to_dict())
                d_copy["symbol"] = sym_clean
                save_json_state(p_exact, d_copy, indent=2)
        except Exception:
            pass

    def _prepare_fetch(
        self,
        symbol: Any,
        allow_network: Any,
        now: Optional[float],
    ) -> Tuple[Optional[str], float, Any, Optional[Any], bool]:
        if not isinstance(symbol, str) or not (sym_clean := symbol.strip().upper()):
            return None, 0.0, None, None, False
        now_f = _effective_now_ts(now)
        allow_net_bool = True if allow_network is None else self._as_bool_flag(allow_network, default=False)
        venue = _resolve_configured_venue(sym_clean)
        require_exact_venue_sym = bool(
            allow_net_bool
            and (
                venue == "hyperliquid"
                or (
                    self._QUOTE_ISOLATED
                    and (venue != "binance" or _is_multi_venue_base(sym_clean))
                )
            )
        )
        cached, fresh = self._lookup_cached(sym_clean, now_f)
        if fresh is not None and (not require_exact_venue_sym or str(getattr(fresh, "symbol", "") or "").strip().upper() == sym_clean):
            if allow_net_bool and not self._QUOTE_ISOLATED and str(getattr(fresh, "symbol", "") or "").strip().upper() != sym_clean:
                self._sync_existing_sibling_disk_file(sym_clean, fresh)
            return sym_clean, now_f, cached, fresh, False
        disk_snap = self._load_from_disk(sym_clean, now_f)
        if disk_snap is not None and (not require_exact_venue_sym or str(getattr(disk_snap, "symbol", "") or "").strip().upper() == sym_clean):
            if allow_net_bool and not self._QUOTE_ISOLATED and str(getattr(disk_snap, "symbol", "") or "").strip().upper() != sym_clean:
                self._sync_existing_sibling_disk_file(sym_clean, disk_snap)
            return sym_clean, now_f, cached, disk_snap, False
        return sym_clean, now_f, cached, None, allow_net_bool

    def _commit_snapshot(self, symbol: str, now: float, snapshot: Any) -> Any:
        self._cache[symbol] = (now, snapshot)
        self._save_to_disk(snapshot)
        return snapshot

    def _resolve_ssot_price(
        self,
        symbol: str,
        *,
        max_age_sec: float = 60.0,
        now: Optional[float] = None,
    ) -> Optional[float]:
        """Resolve spot price for ``symbol`` from cacheManager SSOT stores (`cache_current_price.json` / `cache_24_price`)."""
        return resolve_ssot_price_from_cache(
            symbol,
            cache_dir=self.cache_dir,
            max_age_sec=max_age_sec,
            quote_isolated=bool(self._QUOTE_ISOLATED),
            now_ts=now,
        )

    @staticmethod
    def _post_hyperliquid_info(
        payload: Dict[str, Any],
        *,
        timeout: float = 4.0,
        user_agent: str = "mptrade-intelligence/1.0",
    ) -> Any:
        import json as _json
        import urllib.request as _urlreq

        body = _json.dumps(payload).encode("utf-8")
        req = _urlreq.Request(
            "https://api.hyperliquid.xyz/info",
            data=body,
            headers={"Content-Type": "application/json", "User-Agent": user_agent},
            method="POST",
        )
        with _urlreq.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return _json.loads(raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw)

    @staticmethod
    def _fetch_json_dict(
        url: str,
        label: str,
        lookup_sym: str,
        *,
        timeout: float = 5.0,
        user_agent: str = "mptrade-intelligence/1.0",
    ) -> Dict[str, Any]:
        if isinstance(url, str) and url.startswith("https://api.hyperliquid.xyz/info?"):
            import urllib.parse as _urlparse

            q_params = dict(_urlparse.parse_qsl(url.split("?", 1)[1]))
            raw_data = _DiskBackedSymbolCollector._post_hyperliquid_info(
                q_params, timeout=timeout, user_agent=user_agent
            )
            if isinstance(raw_data, list):
                return {"result_list": raw_data}
        else:
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
        imb_raw = _clean_dict_float(data, "imbalance_ratio", 0.5, "ob_imbalance")
        return cls(
            symbol=_clean_snapshot_symbol(data, symbol),
            mid_price=mid_price,
            bid_depth_usd=bid_depth,
            ask_depth_usd=ask_depth,
            imbalance_ratio=min(1.0, max(0.0, imb_raw)),
            largest_bid_wall_usd=_clean_dict_float(data, "largest_bid_wall_usd", 0.0, "bid_wall_usd", nonneg=True),
            largest_bid_wall_price=_clean_dict_float(data, "largest_bid_wall_price", 0.0, "bid_wall_px", nonneg=True),
            largest_ask_wall_usd=_clean_dict_float(data, "largest_ask_wall_usd", 0.0, "ask_wall_usd", nonneg=True),
            largest_ask_wall_price=_clean_dict_float(data, "largest_ask_wall_price", 0.0, "ask_wall_px", nonneg=True),
            ts=_clean_dict_float(data, "ts", 0.0, nonneg=True),
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
                if venue == "kraken":
                    base_urls = (
                        f"https://api.kraken.com/0/public/Depth?pair={kraken_pair}&count=200",
                        *spot_urls,
                        f"https://fapi.binance.com/fapi/v1/depth?symbol={lookup_sym}&limit=500",
                        *((hl_book_url,) if hl_book_url else ()),
                    )
                elif venue == "hyperliquid" and hl_book_url:
                    base_urls = (
                        hl_book_url,
                        *spot_urls,
                        f"https://api.kraken.com/0/public/Depth?pair={kraken_pair}&count=200",
                        f"https://fapi.binance.com/fapi/v1/depth?symbol={lookup_sym}&limit=500",
                    )
                else:
                    base_urls = (
                        *spot_urls,
                        f"https://api.kraken.com/0/public/Depth?pair={kraken_pair}&count=200",
                        f"https://fapi.binance.com/fapi/v1/depth?symbol={lookup_sym}&limit=500",
                        *((hl_book_url,) if hl_book_url else ()),
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

            def _accumulate_side(
                levels: List[Tuple[float, float]], bound: float, bound_1pct: float, is_bid: bool
            ) -> Tuple[float, float, float, float]:
                depth_usd = depth_1pct = max_usd = max_px = 0.0
                limit_px = min(bound, bound_1pct) if is_bid else max(bound, bound_1pct)
                for p, q in levels:
                    if (p < limit_px) if is_bid else (p > limit_px):
                        break
                    val = p * q
                    if (p >= bound) if is_bid else (p <= bound):
                        depth_usd += val
                        if val > max_usd:
                            max_usd, max_px = val, p
                    if (p >= bound_1pct) if is_bid else (p <= bound_1pct):
                        depth_1pct += val
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
