"""providers/portfolio_state_store.py -- Layer 3 portfolio persistence & composition facade.

Owns disk path resolution and read/write delegation across:
- Layer 3 per-venue ``cache_balance_<venue>.json`` (authoritative raw wallet balances)
- Layer 2.5 ``PortfolioPositionsStateStore`` (Layer 5 ``portfolio_positions_state.json``)
- Layer 2.5 ``VenueBalancesStateStore`` (legacy ``venue_balances_state.json`` fallback)

Pure in-memory accounting, deduplication, and multi-venue NAV math live exclusively in
``providers/portfolio_valuation.py`` (zero disk I/O).
"""
from __future__ import annotations

import os
from typing import Any, Callable, Mapping, Optional

from cachepaths import (
    CACHE_BINANCE_BALANCE_FILENAME,
    CACHE_HYPERLIQUID_BALANCE_FILENAME,
    CACHE_KRAKEN_BALANCE_FILENAME,
    CACHE_KRAKEN_ORDER_FILLED_FILENAME,
    CACHE_KRAKEN_TRADE_FILENAME,
    CACHE_T212_BALANCE_FILENAME,
    clean_scalar_str,
    normalize_venue_name,
    resolve_cachedb_filepath,
    split_symbol_assets,
)
from state_io import (
    read_json_dict,
    read_json_dict_cached,
)
from state_stores.portfolio_positions_state import PortfolioPositionsStateStore
from state_stores.venue_balances_state import VenueBalancesStateStore

from .portfolio_valuation import compose_unified_portfolio_state
from .quantity import summarize_fills_position_and_cost

VENUE_BALANCE_CACHE_FILENAMES: dict[str, str] = {
    "binance": CACHE_BINANCE_BALANCE_FILENAME,
    "kraken": CACHE_KRAKEN_BALANCE_FILENAME,
    "hyperliquid": CACHE_HYPERLIQUID_BALANCE_FILENAME,
    "t212": CACHE_T212_BALANCE_FILENAME,
}


def resolve_venue_balance_cache_path(cachedb_dir: str, venue: str) -> Optional[str]:
    """Resolve canonical disk path for ``cache_balance_<venue>.json`` inside ``cachedb_dir``."""
    v_norm = normalize_venue_name(venue)
    fname = VENUE_BALANCE_CACHE_FILENAMES.get(v_norm or "")
    if not cachedb_dir or not fname:
        return None
    return str(resolve_cachedb_filepath(os.path.join(cachedb_dir, fname)))


def load_venue_balance_caches(
    cachedb_dir: Optional[str],
    *,
    use_cache: bool = False,
    read_json_fn: Optional[Callable[[str], Any]] = None,
) -> dict[str, dict[str, Any]]:
    """SSOT owner reader for Layer 3 per-venue ``cache_balance_<venue>.json`` caches."""
    if not cachedb_dir:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for v_key, fname in VENUE_BALANCE_CACHE_FILENAMES.items():
        candidates = list(
            dict.fromkeys((
                str(resolve_cachedb_filepath(os.path.join(cachedb_dir, fname))),
                os.path.join(cachedb_dir, "account", fname),
                os.path.join(cachedb_dir, "orders", fname),
                os.path.join(cachedb_dir, fname),
            ))
        )
        for cand in candidates:
            has_disk_file = os.path.exists(cand) or os.path.exists(f"{cand}.previous")
            if callable(read_json_fn):
                raw = read_json_fn(cand)
            elif use_cache:
                if not has_disk_file:
                    continue
                raw = read_json_dict_cached(cand)
            else:
                if not has_disk_file:
                    continue
                raw = read_json_dict(cand, default=None, recover_previous=True)
            if isinstance(raw, dict) and isinstance(raw.get("items"), dict) and raw.get("items"):
                out[v_key] = raw
                break
    return out


def resolve_portfolio_positions_state_path(cachedb_dir: str) -> str:
    """Resolve the canonical disk path for Layer 5 ``portfolio_positions_state.json`` via its 1:1 store."""
    return PortfolioPositionsStateStore.resolve_path(cachedb_dir)


def load_portfolio_positions_state_dict(
    cachedb_dir: Optional[str],
    *,
    use_cache: bool = False,
    read_json_fn: Optional[Callable[[str], Any]] = None,
    require_exists: bool = False,
) -> dict[str, Any]:
    """SSOT owner reader for Layer 5 ``portfolio_positions_state.json`` (delegated to 1:1 store)."""
    return PortfolioPositionsStateStore.load_dict(
        cachedb_dir,
        use_cache=use_cache,
        read_json_fn=read_json_fn,
        require_exists=require_exists,
    )


def upsert_portfolio_positions_snapshot(
    pos_path: str,
    updated_positions: Mapping[str, Any],
    *,
    fallback_state: Optional[Mapping[str, Any]] = None,
    read_disk: bool = True,
    save_fn: Optional[Callable[..., Any]] = None,
    indent: int = 2,
) -> bool:
    """Single canonical locked producer writer for Layer 5 ``portfolio_positions_state.json`` (delegated to 1:1 store)."""
    return PortfolioPositionsStateStore.upsert_locked(
        pos_path,
        updated_positions,
        fallback_state=fallback_state,
        read_disk=read_disk,
        save_fn=save_fn,
        indent=indent,
    )


def resolve_venue_balances_state_path(cachedb_dir: str) -> str:
    """Resolve the canonical disk path for legacy ``venue_balances_state.json`` via its 1:1 store."""
    return VenueBalancesStateStore.resolve_path(cachedb_dir)


def _load_legacy_venue_balances_state_dict(
    cachedb_dir: Optional[str],
    *,
    use_cache: bool = False,
    read_json_fn: Optional[Callable[[str], Any]] = None,
    require_exists: bool = False,
) -> dict[str, Any]:
    return VenueBalancesStateStore.load_dict(
        cachedb_dir,
        use_cache=use_cache,
        read_json_fn=read_json_fn,
        require_exists=require_exists,
    )


def load_venue_balances_state_dict(
    cachedb_dir: Optional[str],
    *,
    use_cache: bool = False,
    read_json_fn: Optional[Callable[[str], Any]] = None,
    require_exists: bool = False,
) -> dict[str, Any]:
    """SSOT owner reader composing Layer 3 ``cache_balance_<venue>.json`` + Layer 5 ``portfolio_positions_state.json`` with legacy fallback."""
    if not cachedb_dir:
        return {}
    bal_caches = load_venue_balance_caches(cachedb_dir, use_cache=use_cache, read_json_fn=read_json_fn)
    pos_state = load_portfolio_positions_state_dict(
        cachedb_dir, use_cache=use_cache, read_json_fn=read_json_fn, require_exists=require_exists
    )
    legacy_vb = _load_legacy_venue_balances_state_dict(
        cachedb_dir, use_cache=use_cache, read_json_fn=read_json_fn, require_exists=require_exists
    )
    if bal_caches or pos_state:
        return compose_unified_portfolio_state(bal_caches, pos_state, legacy_vb)
    return legacy_vb


def upsert_venue_balances_snapshot(
    bcache_path: str,
    updated_bals: Mapping[str, Any],
    *,
    fallback_state: Optional[Mapping[str, Any]] = None,
    read_disk: bool = True,
    save_fn: Optional[Callable[..., Any]] = None,
    indent: int = 2,
) -> bool:
    """Single canonical locked producer writer for ``venue_balances_state.json`` (delegated to 1:1 store)."""
    return VenueBalancesStateStore.upsert_locked(
        bcache_path,
        updated_bals,
        fallback_state=fallback_state,
        read_disk=read_disk,
        save_fn=save_fn,
        indent=indent,
    )


def resolve_kraken_trade_cache_position(
    symbol: str,
    cachedb_dir: str,
    *,
    held_qty: Optional[float] = None,
    read_json_fn: Any = None,
) -> Optional[dict[str, Any]]:
    """Resolve live Kraken position/cost-basis summary from ``cache_order_filled_kraken.json`` (with Stage-1 fallback to ``cache_trade_kraken.json``)."""
    sym_s = clean_scalar_str(symbol) or ""
    if not sym_s or not cachedb_dir:
        return None
    sym_cf = sym_s.casefold()
    for fname in (CACHE_KRAKEN_ORDER_FILLED_FILENAME, CACHE_KRAKEN_TRADE_FILENAME):
        cache_file = resolve_cachedb_filepath(os.path.join(cachedb_dir, fname))
        if not os.path.isfile(cache_file):
            continue
        payload = read_json_fn(cache_file) if callable(read_json_fn) else read_json_dict(cache_file, default=None)
        if not isinstance(payload, dict):
            continue
        items = payload.get("items")
        if not isinstance(items, dict) or not items:
            continue
        rows = items.get(sym_s)
        matched_sym = sym_s
        if not isinstance(rows, list):
            for k, v in items.items():
                if isinstance(v, list) and str(k or "").strip().casefold() == sym_cf:
                    rows = v
                    matched_sym = str(k).strip()
                    break
        if not isinstance(rows, list) or not rows:
            continue
        base_asset, quote_asset = split_symbol_assets(matched_sym)
        return summarize_fills_position_and_cost(
            rows,
            base_asset=base_asset or "",
            quote_asset=quote_asset or "",
            held_qty=held_qty,
        )
    return None
