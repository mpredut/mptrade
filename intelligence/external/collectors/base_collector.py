"""Shared base class and SSOT helpers for external microstructure collectors.

Unifies:
- In-memory + disk snapshot caching (`_DiskBackedSymbolCollector`)
- SSOT venue/symbol resolution (`_resolve_configured_venue`, `_is_multi_venue_base`, `_has_hyperliquid_instrument`)
- Hyperliquid L1 `metaAndAssetCtxs` lookup (`_fetch_hyperliquid_asset_ctx`)
- Numeric and spread normalization (`_clean_dict_float`, `_clean_snapshot_symbol`, `round_spread_bps`)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from botcore import (
    _parse_bool_flag,
    fetch_json_post,
    is_rth_gate_enabled,
    normalize_epoch_sec,
    resolve_patched_attr,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
)
from cachepaths import (
    cache_symbol_variants,
    canonical_symbol_key,
    compatible_quote_suffixes,
    normalize_cache_base_asset,
    resolve_cache_dir_path,
    telemetry_cache_file_path,
)
from instrument_registry import is_equity_symbol, is_symbol_market_open, venue_for_symbol
from intelligence.macro_sentiment_ssot import (
    fetch_public_json,
    get_cboe_delayed_options_url,
    get_cboe_index_tickers,
    get_nasdaq_equity_quote_url,
    get_yahoo_equity_chart_5d_url,
    is_cache_fresh,
    resolve_derived_telemetry_intervals,
    resolve_now_ts,
)
from providers.base import resolve_ssot_price_from_cache
from state_io import read_json_dict, read_json_dict_recovering, save_json_state

logger = logging.getLogger("intelligence.external.base_collector")

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
    """Return the SSOT configured provider ('t212', 'hyperliquid', 'kraken', or 'binance') for symbol."""
    return venue_for_symbol(symbol, default="binance")


def _is_multi_venue_base(symbol: str) -> bool:
    """Return True when the symbol's base asset is tracked across multiple venues/quotes."""
    base = normalize_cache_base_asset(symbol)
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
    base = normalize_cache_base_asset(symbol)
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


def _clean_snapshot_ts(data: Dict[str, Any], default: float = 0.0) -> float:
    fb = normalize_epoch_sec(default, default=0.0, allow_zero=True, allow_iso=True)
    eff_default = 0.0 if fb is None else fb
    if not isinstance(data, dict):
        return eff_default
    for k in ("ts", "timestamp"):
        ts = normalize_epoch_sec(data.get(k), default=None, allow_zero=True, allow_iso=True)
        if ts is not None:
            return ts
    return eff_default


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


_ORIG_MODULE_FNS: Dict[str, Callable[..., Any]] = {
    "fetch_public_json": fetch_public_json,
    "read_json_dict": read_json_dict,
    "read_json_dict_recovering": read_json_dict_recovering,
    "save_json_state": save_json_state,
}


def _resolve_module_fn(cls_or_self: Any, attr_name: str, default_fn: Callable[..., Any]) -> Callable[..., Any]:
    """Resolve a helper function from the concrete collector's module or `orderbook_depth_collector` if monkeypatched."""
    orig_fn = _ORIG_MODULE_FNS.get(attr_name, default_fn)
    cls = cls_or_self if isinstance(cls_or_self, type) else type(cls_or_self)
    return resolve_patched_attr(
        attr_name,
        default_fn,
        defaults=(orig_fn,),
        modules=(
            getattr(cls, "__module__", ""),
            "intelligence.external.collectors.orderbook_depth_collector",
            "intelligence.external.collectors.orderbook_depth",
            "intelligence.external.collectors.base_collector",
            "intelligence.macro_sentiment_ssot",
        ),
        require_callable=True,
    )


class _DiskBackedSymbolCollector:
    """Shared in-memory and disk snapshot caching base for external symbol telemetry collectors."""

    _DISK_FILE_PREFIX: str = ""
    _SNAPSHOT_CLS: Any = None
    _QUOTE_ISOLATED: bool = False
    start_sync: bool = False

    def __init__(
        self,
        cache_ttl_sec: float,
        default_cache_ttl: float,
        cache_dir: Optional[str],
        disk_ttl_sec: float,
        default_disk_ttl: float,
        *,
        start_sync: Optional[bool] = None,
    ) -> None:
        self.cache_ttl_sec = safe_nonneg_float(cache_ttl_sec, default=default_cache_ttl)
        self.cache_dir = cache_dir
        self.disk_ttl_sec = safe_nonneg_float(disk_ttl_sec, default=default_disk_ttl)
        self._explicit_start_sync: Optional[bool] = (
            _parse_bool_flag(start_sync, default=False) if start_sync is not None else None
        )
        self.start_sync = bool(self._explicit_start_sync) if self._explicit_start_sync is not None else False
        self._cache: Dict[str, tuple[float, Any]] = {}

    def is_read_only_consumer(self, allow_network: Optional[bool] = None) -> bool:
        """Return True when this collector operates in read-only consumer mode (``start_sync=False``)."""
        if allow_network is not None:
            return not _parse_bool_flag(allow_network, default=False)
        return not bool(self.start_sync)

    def read_snapshot(self, symbol: str, *, now: Optional[float] = None) -> Optional[Any]:
        """Read-only consumer accessor: return fresh snapshot from memory/disk cache in 0 ms without network or disk writes."""
        fetch_fn = getattr(self, "fetch", None)
        if not callable(fetch_fn):
            return None
        return fetch_fn(symbol, allow_network=False, now=now)

    def refresh_snapshot(self, symbol: str, *, now: Optional[float] = None) -> Optional[Any]:
        """Producer accessor: fetch live snapshot from external venue and persist to disk cache."""
        fetch_fn = getattr(self, "fetch", None)
        if not callable(fetch_fn):
            return None
        return fetch_fn(symbol, allow_network=True, now=now)

    @classmethod
    def disk_prefix(cls) -> str:
        """Return the canonical filename prefix (e.g. ``'orderbook_depth_'``) for this collector."""
        pfx = str(getattr(cls, "_DISK_FILE_PREFIX", "") or "").strip()
        return f"{pfx}_" if pfx and not pfx.endswith("_") else pfx

    def collect(
        self,
        symbol: str,
        *,
        allow_network: Optional[bool] = None,
        now: Optional[float] = None,
    ) -> Optional[Any]:
        """Polymorphic collector entrypoint delegating to ``fetch(symbol, allow_network=..., now=...)``."""
        fetch_fn = getattr(self, "fetch", None)
        if not callable(fetch_fn):
            return None
        return fetch_fn(symbol, allow_network=allow_network, now=now)

    def _effective_cache_dir(self) -> Optional[str]:
        return resolve_cache_dir_path(self.cache_dir)

    def _disk_path(self, symbol: str) -> Optional[str]:
        return telemetry_cache_file_path(self.cache_dir, self._DISK_FILE_PREFIX, symbol)

    def _candidate_variants(self, symbol: str) -> List[Any]:
        if self._QUOTE_ISOLATED:
            quotes = compatible_quote_suffixes(symbol, default_usd=("USDC", "USDT", "USD", "FDUSD"))
            return cache_symbol_variants(symbol, quotes=quotes)
        return cache_symbol_variants(symbol)

    def _lookup_cached(self, symbol: str, now: float, *, exact_only: bool = False) -> Tuple[Any, Any]:
        sym_clean = symbol.strip().upper()
        cands = [sym_clean] if exact_only else self._candidate_variants(symbol)
        first_cached = None
        for c in cands:
            if c not in self._cache:
                continue
            entry = self._cache[c]
            snap_obj = entry[1] if isinstance(entry, tuple) and len(entry) >= 2 else entry
            if exact_only and str(getattr(snap_obj, "symbol", "") or "").strip().upper() != sym_clean:
                continue
            if first_cached is None:
                first_cached = entry
            c_ts = entry[0] if isinstance(entry, tuple) and len(entry) >= 2 else getattr(entry, "ts", None)
            if is_cache_fresh(c_ts, now, self.cache_ttl_sec):
                return entry, snap_obj
        return first_cached, None

    def _read_disk_json(self, path: str) -> Optional[Dict[str, Any]]:
        read_fn = _resolve_module_fn(self, "read_json_dict", read_json_dict)
        if read_fn is not _ORIG_MODULE_FNS["read_json_dict"]:
            return read_fn(path, default=None, recover_previous=True)
        rec_fn = _resolve_module_fn(self, "read_json_dict_recovering", read_json_dict_recovering)
        return rec_fn(path, default=None)

    def _load_from_disk(self, symbol: str, now: float, *, ignore_ttl: bool = False, exact_only: bool = False) -> Optional[Any]:
        ignore_ttl_b = _parse_bool_flag(ignore_ttl, default=False)
        max_age = max(self.cache_ttl_sec, self.disk_ttl_sec)
        sym_clean = symbol.strip().upper()
        cands = [sym_clean] if exact_only else self._candidate_variants(symbol)
        for cand in cands:
            p = self._disk_path(cand)
            if not p:
                continue
            try:
                data = self._read_disk_json(p)
                if not isinstance(data, dict):
                    continue
                try:
                    snapshot = self._SNAPSHOT_CLS.from_dict(data, cand)
                except TypeError:
                    snapshot = self._SNAPSHOT_CLS.from_dict(data)
                snap_sym = str(getattr(snapshot, "symbol", "") or cand).strip().upper()
                if exact_only and snap_sym != sym_clean:
                    continue
                if is_cache_fresh(snapshot.ts, now, max_age):
                    if (
                        not ignore_ttl_b
                        and is_equity_symbol(sym_clean)
                        and self._DISK_FILE_PREFIX == "orderbook_depth"
                        and (safe_pos_float(getattr(snapshot, "spread_bps", None), default=0.0) or 0.0) > 200.0
                    ):
                        continue
                    if snap_sym:
                        self._cache[snap_sym] = (snapshot.ts, snapshot)
                    if not exact_only or snap_sym == sym_clean:
                        self._cache[sym_clean] = (snapshot.ts, snapshot)
                    return snapshot
                if ignore_ttl_b:
                    return snapshot
            except Exception as e:
                logger.debug("Failed loading %s snapshot from %s: %s", self._DISK_FILE_PREFIX, p, e)
        return None

    def _save_to_disk(self, snapshot: Any) -> None:
        if p := self._disk_path(snapshot.symbol):
            try:
                save_fn = _resolve_module_fn(self, "save_json_state", save_json_state)
                save_fn(p, snapshot.to_dict(), indent=2)
            except Exception as e:
                logger.debug("Failed writing %s snapshot to %s: %s", self._DISK_FILE_PREFIX, p, e)

    def _is_exact_venue_symbol(self, sym_clean: str) -> bool:
        venue = _resolve_configured_venue(sym_clean)
        return bool(
            venue in ("hyperliquid", "t212")
            or is_equity_symbol(sym_clean)
            or (
                self._QUOTE_ISOLATED
                and (venue != "binance" or _is_multi_venue_base(sym_clean))
            )
        )

    def _fallback_stale(self, symbol: str, cached: Any) -> Optional[Any]:
        if isinstance(cached, tuple) and len(cached) >= 2:
            return cached[1]
        if cached and self._SNAPSHOT_CLS is not None and isinstance(cached, self._SNAPSHOT_CLS):
            return cached
        sym_clean = symbol.strip().upper() if isinstance(symbol, str) else ""
        return self._load_from_disk(symbol, 0.0, ignore_ttl=True, exact_only=self._is_exact_venue_symbol(sym_clean))

    def _record_item(self, item: Any, save_to_disk: bool = False) -> None:
        if isinstance(item, dict):
            item = self._SNAPSHOT_CLS.from_dict(item)
        if item is None or not isinstance(getattr(item, "symbol", None), str) or not item.symbol.strip():
            return
        ts_val = safe_pos_float(getattr(item, "ts", None), default=None) or time.time()
        self._cache[item.symbol.strip().upper()] = (ts_val, item)
        if _parse_bool_flag(save_to_disk, default=False):
            self._save_to_disk(item)

    def _sync_existing_sibling_disk_file(self, sym_clean: str, snapshot: Any) -> None:
        if not (p_exact := self._disk_path(sym_clean)):
            return
        try:
            import os as _os

            if not _os.path.isfile(p_exact):
                return
            cur_d = self._read_disk_json(p_exact)
            cur_ts = safe_pos_float(cur_d.get("ts") if isinstance(cur_d, dict) else None, default=0.0) or 0.0
            snap_ts = safe_pos_float(getattr(snapshot, "ts", None), default=0.0) or 0.0
            if snap_ts > cur_ts + 5.0 and hasattr(snapshot, "to_dict"):
                d_copy = dict(snapshot.to_dict())
                d_copy["symbol"] = sym_clean
                save_fn = _resolve_module_fn(self, "save_json_state", save_json_state)
                save_fn(p_exact, d_copy, indent=2)
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
        now_f = resolve_now_ts(now)
        if allow_network is None:
            allow_net_bool = (
                True
                if getattr(self, "_explicit_start_sync", None) is None
                else bool(getattr(self, "start_sync", False))
            )
        else:
            allow_net_bool = _parse_bool_flag(allow_network, default=False)
        if self._is_exact_venue_symbol(sym_clean):
            cached_ex, fresh_ex = self._lookup_cached(sym_clean, now_f, exact_only=True)
            if fresh_ex is not None:
                return sym_clean, now_f, cached_ex, fresh_ex, False
            disk_ex = self._load_from_disk(sym_clean, now_f, exact_only=True)
            if disk_ex is not None:
                return sym_clean, now_f, cached_ex, disk_ex, False
            if (
                is_equity_symbol(sym_clean)
                and is_rth_gate_enabled(getattr(self, "_enforce_rth_gate", None))
            ):
                try:
                    from instrument_registry import is_symbol_market_open

                    if not is_symbol_market_open(sym_clean, now_f):
                        stale_disk = self._load_from_disk(sym_clean, now_f, ignore_ttl=True, exact_only=True)
                        stale_ts = safe_pos_float(getattr(stale_disk, "ts", None), default=0.0) or 0.0
                        stale_sp = safe_pos_float(getattr(stale_disk, "spread_bps", None), default=0.0) or 0.0
                        share_exec_sec = resolve_derived_telemetry_intervals().get("share_executor_interval_sec", 1800.0)
                        closed_rth_max_age_sec = 192.0 * share_exec_sec  # 192 * 30m = 96h (covers 3-day holiday weekend + overnight)
                        stale_was_open_rth = (
                            allow_net_bool
                            and "MPTRADE_FORCE_RTH_GATE" not in os.environ
                            and stale_ts > 0.0
                            and bool(is_symbol_market_open(sym_clean, stale_ts))
                        )
                        stale_whale_legacy = self._DISK_FILE_PREFIX == "whale_snapshot" and (
                            (safe_pos_float(getattr(stale_disk, "taker_ratio_5m", None), default=1.0) or 1.0) < 0.176
                            or (safe_pos_float(getattr(stale_disk, "taker_ratio_5m", None), default=1.0) or 1.0) > 5.67
                            or (safe_pos_float(getattr(stale_disk, "top_traders_long_pct", None), default=0.0) or 0.0) <= 0.0
                            or abs(safe_float(getattr(stale_disk, "funding_rate", None), default=0.0) or 0.0) > 0.0
                        )
                        if (
                            stale_disk is not None
                            and 0.0 <= (now_f - stale_ts) < closed_rth_max_age_sec
                            and not stale_was_open_rth
                            and not (allow_net_bool and stale_whale_legacy)
                            and (self._DISK_FILE_PREFIX != "orderbook_depth" or stale_sp <= 200.0)
                        ):
                            self._cache[sym_clean] = (now_f, stale_disk)
                            return sym_clean, now_f, cached_ex, stale_disk, False
                except Exception:
                    pass
            if allow_net_bool:
                return sym_clean, now_f, cached_ex, None, True
        cached, fresh = self._lookup_cached(sym_clean, now_f, exact_only=False)
        if fresh is not None:
            if allow_net_bool and not self._QUOTE_ISOLATED and str(getattr(fresh, "symbol", "") or "").strip().upper() != sym_clean:
                self._sync_existing_sibling_disk_file(sym_clean, fresh)
            return sym_clean, now_f, cached, fresh, False
        disk_snap = self._load_from_disk(sym_clean, now_f, exact_only=False)
        if disk_snap is not None:
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
        return fetch_json_post(
            "https://api.hyperliquid.xyz/info",
            payload,
            timeout=timeout,
            headers={"User-Agent": user_agent},
        )

    @classmethod
    def _fetch_json_dict(
        cls,
        url: str,
        label: str,
        lookup_sym: str,
        *,
        timeout: float = 5.0,
        user_agent: str = "mptrade-intelligence/1.0",
    ) -> Dict[str, Any]:
        fetch_fn = _resolve_module_fn(cls, "fetch_public_json", fetch_public_json)
        is_hl_info = isinstance(url, str) and url.startswith("https://api.hyperliquid.xyz/info?")
        if fetch_fn is _ORIG_MODULE_FNS["fetch_public_json"] and is_hl_info:
            import urllib.parse as _urlparse

            q_params: Dict[str, Any] = dict(_urlparse.parse_qsl(url.split("?", 1)[1]))
            for int_k in ("nSigFigs", "mantissa"):
                if int_k in q_params:
                    try:
                        q_params[int_k] = int(q_params[int_k])
                    except (TypeError, ValueError):
                        pass
            raw_data = cls._post_hyperliquid_info(
                q_params, timeout=timeout, user_agent=user_agent
            )
        else:
            raw_data = fetch_fn(url, timeout=timeout, user_agent=user_agent)
        if is_hl_info and isinstance(raw_data, list):
            return {"result_list": raw_data}
        if not isinstance(raw_data, dict):
            raise ValueError(f"Unexpected {label} payload for {lookup_sym}: {type(raw_data).__name__}")
        return raw_data

    @classmethod
    def _fetch_latest_json_row(
        cls,
        url: str,
        label: str,
        lookup_sym: str,
        *,
        timeout: float = 5.0,
        user_agent: str = "mptrade-whale/1.0",
    ) -> Tuple[Dict[str, Any], List[Any]]:
        fetch_fn = _resolve_module_fn(cls, "fetch_public_json", fetch_public_json)
        rows = fetch_fn(url, timeout=timeout, user_agent=user_agent)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Empty {label} response for {lookup_sym}")
        dict_rows = [r for r in rows if isinstance(r, dict)]
        if not dict_rows:
            raise ValueError(f"Empty {label} response for {lookup_sym}")
        if len(dict_rows) >= 2:
            ts_vals = [safe_float(r.get("timestamp") if r.get("timestamp") is not None else r.get("ts"), default=None) for r in dict_rows]
            if all(t is not None for t in ts_vals) and any(ts_vals[i] > ts_vals[i + 1] for i in range(len(ts_vals) - 1)):
                dict_rows = [r for _, r in sorted(zip(ts_vals, dict_rows), key=lambda pair: pair[0])]
                rows = dict_rows
        return dict_rows[-1], rows

    def _fetch_hyperliquid_asset_ctx(
        self,
        symbol: str,
        *,
        timeout: float = 4.0,
    ) -> Optional[Dict[str, Any]]:
        """Fetch the Hyperliquid `metaAndAssetCtxs` context dict for `symbol`'s base coin."""
        base_coin = normalize_cache_base_asset(symbol)
        if not base_coin:
            return None
        raw = self._fetch_json_dict(
            "https://api.hyperliquid.xyz/info?type=metaAndAssetCtxs",
            "metaAndAssetCtxs",
            base_coin,
            timeout=timeout,
        )
        res_list = raw.get("result_list") if isinstance(raw, dict) else None
        if not isinstance(res_list, list) or len(res_list) < 2:
            return None
        meta, ctxs = res_list[0], res_list[1]
        universe = (meta.get("universe") if isinstance(meta, dict) else None) or []
        if not isinstance(universe, list) or not isinstance(ctxs, list):
            return None
        base_cf = base_coin.casefold()
        for idx, asset_info in enumerate(universe):
            if (
                isinstance(asset_info, dict)
                and str(asset_info.get("name") or "").casefold() == base_cf
                and idx < len(ctxs)
                and isinstance(ctxs[idx], dict)
            ):
                return ctxs[idx]
        return None

    _EQUITY_CTX_MEMO: Dict[str, Tuple[float, Optional[Dict[str, Any]]]] = {}
    _EQUITY_OPTIONS_MEMO: Dict[str, Tuple[float, Tuple[float, float, float, float]]] = {}
    _EQUITY_NO_OPTIONS_TS: Dict[str, float] = {}

    def _populate_equity_options_context(
        self,
        ticker: str,
        ctx_out: Dict[str, Any],
        now: float,
        *,
        timeout: float = 2.5,
        opt_memo_ttl: float = 60.0,
        no_options_ttl: float = 1800.0,
    ) -> None:
        import re as _re

        cached_opt = self._EQUITY_OPTIONS_MEMO.get(ticker)
        if cached_opt is not None and 0.0 <= (now - cached_opt[0]) < opt_memo_ttl:
            c_oi, p_oi, c_vol, p_vol = cached_opt[1]
            ctx_out.update(call_oi=c_oi, put_oi=p_oi, call_vol=c_vol, put_vol=p_vol, _options_loaded=True)
            return

        call_oi = 0.0
        put_oi = 0.0
        call_vol = 0.0
        put_vol = 0.0
        no_opt_ts = self._EQUITY_NO_OPTIONS_TS.get(ticker, 0.0)
        if not (no_opt_ts > 0.0 and 0.0 <= (now - no_opt_ts) < no_options_ttl):
            opt_rx = _re.compile(r"\d{6}([CP])\d{8}$")
            cboe_paths = (
                (f"_{ticker}.json", f"{ticker}.json")
                if ticker in get_cboe_index_tickers()
                else (f"{ticker}.json",)
            )
            found_opts = False
            for cboe_path in cboe_paths:
                try:
                    cb = self._fetch_json_dict(
                        get_cboe_delayed_options_url(cboe_path),
                        "cboe_options",
                        ticker,
                        timeout=timeout,
                        user_agent="Mozilla/5.0",
                    )
                    opts = ((cb.get("data") or {}).get("options") if isinstance(cb, dict) else None) or []
                    if isinstance(opts, list) and opts:
                        found_opts = True
                        for opt in opts:
                            if not isinstance(opt, dict):
                                continue
                            opt_sym = str(opt.get("option") or "")
                            m = opt_rx.search(opt_sym)
                            if not m:
                                continue
                            oi_v = safe_nonneg_float(opt.get("open_interest"), default=0.0) or 0.0
                            vol_v = safe_nonneg_float(opt.get("volume"), default=0.0) or 0.0
                            if m.group(1) == "C":
                                call_oi += oi_v
                                call_vol += vol_v
                            else:
                                put_oi += oi_v
                                put_vol += vol_v
                        break
                except Exception:
                    continue
            if not found_opts:
                if len(self._EQUITY_NO_OPTIONS_TS) > 64:
                    self._EQUITY_NO_OPTIONS_TS.clear()
                self._EQUITY_NO_OPTIONS_TS[ticker] = now
            else:
                self._EQUITY_NO_OPTIONS_TS.pop(ticker, None)
                if len(self._EQUITY_OPTIONS_MEMO) > 32:
                    self._EQUITY_OPTIONS_MEMO.clear()
                self._EQUITY_OPTIONS_MEMO[ticker] = (now, (call_oi, put_oi, call_vol, put_vol))

        ctx_out.update(call_oi=call_oi, put_oi=put_oi, call_vol=call_vol, put_vol=put_vol, _options_loaded=True)

    def _fetch_equity_market_context(
        self,
        symbol: str,
        now: float,
        *,
        timeout: float = 2.5,
        include_options: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Fetch and memoize zero-cost equity quote, intraday 5m OHLCV, and Cboe options OI for ``symbol``."""
        ticker = normalize_cache_base_asset(symbol)
        if not ticker:
            return None
        derived_intervals = resolve_derived_telemetry_intervals()
        pos_memo_ttl = derived_intervals.get("derivatives_interval_sec", 30.0)
        neg_memo_ttl = derived_intervals.get("whale_interval_sec", 60.0)
        no_options_ttl = derived_intervals.get("share_executor_interval_sec", 1800.0)
        cached_ctx = self._EQUITY_CTX_MEMO.get(ticker)
        if cached_ctx is not None:
            memo_ttl = neg_memo_ttl if cached_ctx[1] is None else pos_memo_ttl
            if 0.0 <= (now - cached_ctx[0]) < memo_ttl:
                if cached_ctx[1] is not None and include_options and not cached_ctx[1].get("_options_loaded"):
                    self._populate_equity_options_context(
                        ticker,
                        cached_ctx[1],
                        now,
                        timeout=timeout,
                        opt_memo_ttl=neg_memo_ttl,
                        no_options_ttl=no_options_ttl,
                    )
                return cached_ctx[1]

        def _parse_num(val: Any) -> Optional[float]:
            if val is None or isinstance(val, bool):
                return None
            if isinstance(val, (int, float)):
                return safe_pos_float(val, default=None)
            s = str(val).strip().replace("$", "").replace(",", "")
            if not s or s.upper() in ("N/A", "NA", "NONE", "-"):
                return None
            if "x" in s.lower():
                parts = [p.strip() for p in s.lower().split("x") if p.strip()]
                if len(parts) == 2:
                    a, b = safe_pos_float(parts[0], default=None), safe_pos_float(parts[1], default=None)
                    if a is not None and b is not None:
                        return a * b
            return safe_pos_float(s, default=None)

        last_px = self._resolve_ssot_price(symbol, max_age_sec=86400.0, now=now) or 0.0
        best_bid = 0.0
        best_ask = 0.0
        bid_size_sh = 0.0
        ask_size_sh = 0.0
        daily_vol_sh = 0.0

        # 1. Nasdaq Public Quote API (L1 top-of-book & daily volume)
        try:
            nq = self._fetch_json_dict(
                get_nasdaq_equity_quote_url(ticker),
                "nasdaq_quote",
                ticker,
                timeout=timeout,
                user_agent="Mozilla/5.0",
            )
            pdata = ((nq.get("data") or {}).get("primaryData") if isinstance(nq, dict) else None) or {}
            if isinstance(pdata, dict):
                if (lp := _parse_num(pdata.get("lastSalePrice"))) is not None:
                    last_px = lp
                if (bp := _parse_num(pdata.get("bidPrice"))) is not None:
                    best_bid = bp
                if (ap := _parse_num(pdata.get("askPrice"))) is not None:
                    best_ask = ap
                if (bs := _parse_num(pdata.get("bidSize"))) is not None:
                    bid_size_sh = bs
                if (as_ := _parse_num(pdata.get("askSize"))) is not None:
                    ask_size_sh = as_
                if (dv := _parse_num(pdata.get("volume"))) is not None:
                    daily_vol_sh = dv
        except Exception:
            pass

        # 2. Yahoo 5d/5m Intraday OHLCV (volume profile & CLV buy/sell flow)
        bars: List[Tuple[float, float, float, float, float, float]] = []
        try:
            yh = self._fetch_json_dict(
                get_yahoo_equity_chart_5d_url(ticker),
                "yahoo_chart_5m",
                ticker,
                timeout=timeout,
                user_agent="Mozilla/5.0",
            )
            res0 = (((yh.get("chart") or {}).get("result") or [None])[0] if isinstance(yh, dict) else None) or {}
            meta = res0.get("meta") if isinstance(res0, dict) else {}
            if last_px <= 0.0 and isinstance(meta, dict):
                last_px = safe_pos_float(meta.get("regularMarketPrice"), default=0.0) or 0.0
            ts_seq = res0.get("timestamp") or []
            q0 = (((res0.get("indicators") or {}).get("quote") or [None])[0] if isinstance(res0, dict) else None) or {}
            opens = q0.get("open") or []
            highs = q0.get("high") or []
            lows = q0.get("low") or []
            closes = q0.get("close") or []
            vols = q0.get("volume") or []
            for idx, raw_t in enumerate(ts_seq):
                tf = normalize_epoch_sec(raw_t, default=None)
                cf = safe_pos_float(closes[idx] if idx < len(closes) else None, default=None)
                if tf is None or cf is None:
                    continue
                of = safe_pos_float(opens[idx] if idx < len(opens) else None, default=cf) or cf
                hf = safe_pos_float(highs[idx] if idx < len(highs) else None, default=cf) or cf
                lf = safe_pos_float(lows[idx] if idx < len(lows) else None, default=cf) or cf
                vf = safe_nonneg_float(vols[idx] if idx < len(vols) else None, default=0.0) or 0.0
                bars.append((tf, of, max(hf, cf, of), min(lf, cf, of), cf, vf))
            if bars and last_px <= 0.0:
                last_px = bars[-1][4]
        except Exception:
            pass

        if last_px <= 0.0 and not bars:
            if len(self._EQUITY_CTX_MEMO) > 32:
                self._EQUITY_CTX_MEMO.clear()
            self._EQUITY_CTX_MEMO[ticker] = (now, None)
            return None

        ctx_out: Dict[str, Any] = {
            "ticker": ticker,
            "last_price": last_px,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "bid_size_sh": bid_size_sh,
            "ask_size_sh": ask_size_sh,
            "daily_vol_sh": daily_vol_sh,
            "bars": bars,
            "call_oi": 0.0,
            "put_oi": 0.0,
            "call_vol": 0.0,
            "put_vol": 0.0,
            "_options_loaded": False,
        }
        if len(self._EQUITY_CTX_MEMO) > 32:
            self._EQUITY_CTX_MEMO.clear()
        self._EQUITY_CTX_MEMO[ticker] = (now, ctx_out)
        if include_options:
            self._populate_equity_options_context(
                ticker,
                ctx_out,
                now,
                timeout=timeout,
                opt_memo_ttl=neg_memo_ttl,
                no_options_ttl=no_options_ttl,
            )
        return ctx_out
