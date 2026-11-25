"""providers/portfolio_valuation.py -- Pure in-memory multi-venue NAV, balance observation, and accounting SSOT.

Strictly zero disk I/O (no ``os``, ``pathlib``, ``state_io``, or ``state_stores`` imports).
All file resolution and persistence live in ``providers/portfolio_state_store.py`` and ``state_stores/``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from cachepaths import (
    cache_symbol_variants,
    canonical_base_asset,
    clean_scalar_str,
    compatible_quote_suffixes,
    ensure_quote_suffix,
    has_equity_suffix,
    is_equity_venue_or_suffix,
    normalize_venue_name,
    safe_nonneg_float,
    safe_pos_float,
)
from core.math_coercion import first_pos_float
from core.time_market import (
    extract_current_price_item,
    extract_current_price_timestamp_sec,
    normalize_epoch_sec,
)
from instrument_registry import canonical_balance_asset_key, canonical_display_symbol


def extract_position_held_qty(entry: Any) -> Optional[float]:
    """Return total held base quantity (``free_base + locked_base_qty``) from a position dict, or ``None`` if absent."""
    if not isinstance(entry, Mapping):
        return None
    fb = safe_nonneg_float(
        entry.get("free_base") if entry.get("free_base") is not None else entry.get("free"),
        default=None,
    )
    lb = first_pos_float(entry.get("locked_base_qty"), entry.get("locked"), default=0.0) or 0.0
    if fb is None:
        return lb if lb > 0.0 else None
    return fb + lb


def reprice_position_balance_fields(entry: dict[str, Any], cur_px: float) -> None:
    """Update ``value_usd`` and ``locked_base_usd`` consistently when a position's mark price changes."""
    px_f = safe_pos_float(cur_px, default=0.0) or 0.0
    if px_f <= 0.0:
        return
    fb = safe_nonneg_float(
        entry.get("free_base") if entry.get("free_base") is not None else entry.get("free"),
        default=None,
    )
    if fb is not None:
        entry["value_usd"] = round(fb * px_f, 2)
    lb = first_pos_float(entry.get("locked_base_qty"), entry.get("locked"), default=0.0) or 0.0
    if lb > 0.0:
        entry["locked_base_usd"] = round(lb * px_f, 2)


@dataclass(frozen=True, slots=True)
class BalanceObservation:
    """Atomic balance observation binding raw free quantity to its own timestamp, reflected locked amount, and priority."""

    free: float
    ts: Optional[float] = None
    priority: int = 0
    reflected_locked: Optional[float] = None


def merge_balance_observation(
    cur: Optional[BalanceObservation],
    free: Optional[float],
    ts: Optional[float],
    priority: int,
    reflected_locked: Optional[float] = None,
) -> Optional[BalanceObservation]:
    """Combine atomic ``BalanceObservation`` records without mixing value, timestamp, or reflected locks."""
    if free is None:
        return cur
    cand = BalanceObservation(free=free, ts=ts, priority=priority, reflected_locked=reflected_locked)
    if cur is None:
        return cand
    if cur.ts is not None and ts is not None:
        if ts > cur.ts + 1e-6:
            return cand
        if cur.ts > ts + 1e-6:
            return cur
        winner, other = (cand, cur) if priority >= cur.priority else (cur, cand)
        if winner.reflected_locked is None and other.reflected_locked is not None:
            return BalanceObservation(
                free=winner.free, ts=winner.ts, priority=winner.priority, reflected_locked=other.reflected_locked
            )
        return winner
    if ts is not None:
        return cand
    if cur.ts is not None:
        return cur
    winner, other = (cand, cur) if priority >= cur.priority else (cur, cand)
    if winner.reflected_locked is None and other.reflected_locked is not None:
        return BalanceObservation(
            free=winner.free, ts=winner.ts, priority=winner.priority, reflected_locked=other.reflected_locked
        )
    return winner


def build_cached_balance_observations(
    cached_bals: Any,
    *,
    resolve_assets_fn: Callable[[str, Any, Any], tuple[Any, Any]],
    canonical_symbol_fn: Optional[Callable[..., str]] = None,
    balance_caches: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> dict[tuple[str, str], BalanceObservation]:
    """Build atomic ``(venue, asset) -> BalanceObservation`` records from Layer 3 balance caches and state snapshots.

    Prefers canonical Layer 3 ``cache_balance_<venue>.json`` observations when present and fresher, and preserves
    ``reflected_locked`` grouped strictly by matching timestamp bucket so stale position snapshots cannot
    contaminate newer observations.
    """
    obs_by_va: dict[tuple[str, str], BalanceObservation] = {}
    if isinstance(balance_caches, Mapping) and balance_caches:
        cache_obs, _ = extract_balance_observations_from_caches(balance_caches)
        obs_by_va.update(cache_obs)
    if not isinstance(cached_bals, Mapping):
        return obs_by_va

    # 1. Scalar "<venue>:<ASSET>" entries (lowest precedence; untimestamped unless corroborated)
    for k, v in cached_bals.items():
        if not isinstance(k, str) or ":" not in k or k.startswith(("pos:", "cash:", "breakdown:", "open_orders:", "open_orders_verified_ts:")):
            continue
        raw_v, raw_a = k.split(":", 1)
        if ":" in raw_a:
            continue
        v_n = normalize_venue_name(raw_v) or raw_v.strip().lower()
        a_u = canonical_balance_asset_key(raw_a, venue=v_n)
        if v_n and a_u and (val_f := safe_nonneg_float(v, default=None)) is not None:
            va_k = (v_n, a_u)
            if (merged := merge_balance_observation(obs_by_va.get(va_k), val_f, None, 0)) is not None:
                obs_by_va[va_k] = merged

    # 2. Structured "pos:<venue>:<symbol>" snapshots: accumulate locks per (venue, asset, ts_bucket) first
    pos_rows: list[tuple[str, str, str, Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]] = []
    pos_locked_base_by_vat: dict[tuple[str, str, Optional[float]], float] = {}
    pos_locked_quote_by_vat: dict[tuple[str, str, Optional[float]], float] = {}
    for k, v in cached_bals.items():
        if not (isinstance(k, str) and k.startswith("pos:") and isinstance(v, Mapping)):
            continue
        v_n = normalize_venue_name(v.get("venue"))
        if not v_n:
            continue
        p_sym_raw = str(v.get("symbol") or "").strip()
        p_sym = (
            (canonical_symbol_fn(p_sym_raw, venue=v_n) or p_sym_raw.upper())
            if callable(canonical_symbol_fn)
            else (canonical_display_symbol(p_sym_raw, venue=v_n) or p_sym_raw.upper())
        )
        res_b, res_q = resolve_assets_fn(p_sym, v.get("base"), v.get("quote"))
        b_u = canonical_balance_asset_key(res_b, venue=v_n)
        q_u = str(res_q or "USD").strip().upper()
        pos_ts = safe_pos_float(v.get("ts"), default=None)
        base_ts = safe_pos_float(v.get("base_bal_ts"), default=None) or pos_ts
        quote_ts = safe_pos_float(v.get("quote_bal_ts"), default=None) or pos_ts
        b_bucket = round(base_ts, 3) if base_ts is not None else None
        q_bucket = round(quote_ts, 3) if quote_ts is not None else None
        fb = safe_nonneg_float(v.get("free_base") if v.get("free_base") is not None else v.get("free"), default=None)
        fq = safe_nonneg_float(v.get("free_quote"), default=None)
        lb = safe_nonneg_float(v.get("locked_base_qty"), default=None)
        lq = safe_nonneg_float(v.get("locked_quote_usd"), default=None)
        if b_u and lb is not None:
            pos_locked_base_by_vat[(v_n, b_u, b_bucket)] = pos_locked_base_by_vat.get((v_n, b_u, b_bucket), 0.0) + lb
        if q_u and lq is not None:
            pos_locked_quote_by_vat[(v_n, q_u, q_bucket)] = pos_locked_quote_by_vat.get((v_n, q_u, q_bucket), 0.0) + lq
        pos_rows.append((v_n, b_u, q_u, fb, base_ts, b_bucket, fq, quote_ts, q_bucket))

    for v_n, b_u, q_u, fb, base_ts, b_bucket, fq, quote_ts, q_bucket in pos_rows:
        if b_u and fb is not None:
            va_b = (v_n, b_u)
            if (
                merged_b := merge_balance_observation(
                    obs_by_va.get(va_b), fb, base_ts, 1, reflected_locked=pos_locked_base_by_vat.get((v_n, b_u, b_bucket))
                )
            ) is not None:
                obs_by_va[va_b] = merged_b
        if q_u and fq is not None:
            va_q = (v_n, q_u)
            if (
                merged_q := merge_balance_observation(
                    obs_by_va.get(va_q), fq, quote_ts, 1, reflected_locked=pos_locked_quote_by_vat.get((v_n, q_u, q_bucket))
                )
            ) is not None:
                obs_by_va[va_q] = merged_q

    # 3. Structured "cash:<venue>:<asset>" snapshots (highest precedence on equal/absent timestamps)
    for k, v in cached_bals.items():
        if not (isinstance(k, str) and k.startswith("cash:") and isinstance(v, Mapping)):
            continue
        v_n = normalize_venue_name(v.get("venue"))
        a_u = canonical_balance_asset_key(v.get("asset"), venue=v_n)
        if not v_n or not a_u:
            continue
        cf = safe_nonneg_float(v.get("free"), default=None)
        if cf is not None:
            c_ts = safe_pos_float(v.get("ts"), default=None)
            c_lq = safe_nonneg_float(v.get("locked_quote_usd"), default=None)
            va_c = (v_n, a_u)
            if (
                merged_c := merge_balance_observation(obs_by_va.get(va_c), cf, c_ts, 2, reflected_locked=c_lq)
            ) is not None:
                obs_by_va[va_c] = merged_c

    return obs_by_va


_NAV_QUOTE_CASH_ASSETS: frozenset[str] = frozenset(
    {"USD", "USDC", "USDT", "FDUSD", "TUSD", "BUSD", "USDG"}
)


def extract_balance_observations_from_caches(
    balance_caches: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[tuple[str, str], BalanceObservation], dict[tuple[str, str], dict[str, float]]]:
    """Extract atomic ``(venue, asset) -> BalanceObservation`` records and quote breakdowns from balance caches."""
    obs_by_va: dict[tuple[str, str], BalanceObservation] = {}
    breakdowns: dict[tuple[str, str], dict[str, float]] = {}
    if not isinstance(balance_caches, Mapping):
        return obs_by_va, breakdowns
    for raw_v, doc in balance_caches.items():
        if not isinstance(doc, Mapping):
            continue
        v_n = normalize_venue_name(doc.get("venue") or raw_v)
        if not v_n:
            continue
        doc_ts = normalize_epoch_sec(doc.get("verified_ts") or doc.get("ts"), default=None)
        raw_items = doc.get("items")
        if isinstance(raw_items, Mapping):
            for raw_a, entry in raw_items.items():
                a_clean = clean_scalar_str(raw_a)
                if not a_clean or not isinstance(entry, Mapping):
                    continue
                a_u = canonical_balance_asset_key(a_clean, venue=v_n)
                if not a_u:
                    continue
                free_v = safe_nonneg_float(entry.get("free"), default=None)
                if free_v is None:
                    continue
                locked_v = (
                    safe_nonneg_float(entry.get("locked"), default=None)
                    if entry.get("locked") is not None
                    else None
                )
                entry_ts = normalize_epoch_sec(entry.get("ts"), default=None) or doc_ts
                va_k = (v_n, a_u)
                if (
                    merged := merge_balance_observation(
                        obs_by_va.get(va_k), free_v, entry_ts, 3, reflected_locked=locked_v
                    )
                ) is not None:
                    obs_by_va[va_k] = merged
        raw_bd = doc.get("breakdown")
        if isinstance(raw_bd, Mapping):
            for q_raw, q_map in raw_bd.items():
                q_u = (clean_scalar_str(q_raw) or "").upper()
                if not q_u or not isinstance(q_map, Mapping):
                    continue
                clean_bd = {
                    str(bk): val_f
                    for bk, bv in q_map.items()
                    if clean_scalar_str(bk) is not None and (val_f := safe_nonneg_float(bv, default=None)) is not None
                }
                if clean_bd:
                    breakdowns[(v_n, q_u)] = clean_bd
    return obs_by_va, breakdowns


def compose_unified_portfolio_state(
    balance_caches: Optional[Mapping[str, Mapping[str, Any]]] = None,
    positions_state: Optional[Mapping[str, Any]] = None,
    legacy_state: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Compose a unified portfolio view from Layer 3 ``cache_balance_<venue>.json`` + Layer 5 ``portfolio_positions_state.json``."""
    bal_map = balance_caches if isinstance(balance_caches, Mapping) else {}
    pos_map = positions_state if isinstance(positions_state, Mapping) else {}
    leg_map = legacy_state if isinstance(legacy_state, Mapping) else {}
    if not bal_map and not pos_map:
        return dict(leg_map)

    out: dict[str, Any] = {}
    cache_venues: set[str] = set()
    cache_obs: dict[tuple[str, str], BalanceObservation] = {}
    cache_bds: dict[tuple[str, str], dict[str, float]] = {}
    if bal_map:
        cache_obs, cache_bds = extract_balance_observations_from_caches(bal_map)
        for raw_v, doc in bal_map.items():
            if isinstance(doc, Mapping) and isinstance(doc.get("items"), Mapping) and doc.get("items"):
                if (v_n := normalize_venue_name(doc.get("venue") or raw_v)):
                    cache_venues.add(v_n)

    # 1. Preserve legacy entries only for venues/positions not superseded by the split files
    pos_upper = {str(k).upper() for k in pos_map if isinstance(k, str) and k.startswith("pos:")}
    for k, v in leg_map.items():
        if not isinstance(k, str):
            continue
        if k.startswith(("open_orders:", "open_orders_verified_ts:")):
            continue
        if k.startswith("pos:"):
            if k.upper() not in pos_upper:
                out[k] = dict(v) if isinstance(v, Mapping) else v
            continue
        if k.startswith(("cash:", "breakdown:")):
            parts = k.split(":", 2)
            v_n = normalize_venue_name(parts[1] if len(parts) > 1 else "")
            if v_n and v_n in cache_venues:
                continue
            out[k] = dict(v) if isinstance(v, Mapping) else v
            continue
        if ":" in k:
            v_n = normalize_venue_name(k.split(":", 1)[0])
            if v_n and v_n in cache_venues:
                continue
        out[k] = dict(v) if isinstance(v, Mapping) else v

    # 2. Layer 5 derived position metadata (portfolio_positions_state.json)
    for k, v in pos_map.items():
        if isinstance(k, str) and k.startswith("pos:") and isinstance(v, Mapping):
            out[k] = dict(v)

    # 3. Layer 3 authoritative account balances (cache_balance_<venue>.json)
    for (v_n, a_k), obs in cache_obs.items():
        out[f"{v_n}:{a_k}"] = obs.free
        if a_k in _NAV_QUOTE_CASH_ASSETS:
            c_entry: dict[str, Any] = {"venue": v_n, "asset": a_k, "free": round(obs.free, 2)}
            if obs.ts is not None:
                c_entry["ts"] = round(obs.ts, 3)
            if obs.reflected_locked is not None and obs.reflected_locked > 1e-6:
                c_entry["locked_quote_usd"] = round(obs.reflected_locked, 2)
            out[f"cash:{v_n}:{a_k}"] = c_entry
    for (v_n, q_u), bd in cache_bds.items():
        out[f"breakdown:{v_n}:{q_u}"] = dict(bd)

    # 4. Overlay fresher Layer 3 balance observations onto derived pos:* entries without losing strategy metadata
    if cache_obs:
        for k, entry in list(out.items()):
            if not (isinstance(k, str) and k.startswith("pos:") and isinstance(entry, dict)):
                continue
            parts = k.split(":", 2)
            v_n = normalize_venue_name(entry.get("venue") or (parts[1] if len(parts) > 1 else ""))
            if not v_n or v_n not in cache_venues:
                continue
            base_k = canonical_balance_asset_key(entry.get("base"), venue=v_n)
            quote_u = (clean_scalar_str(entry.get("quote")) or "").upper()
            pos_ts = normalize_epoch_sec(entry.get("ts"), default=0.0) or 0.0
            if base_k and (b_obs := cache_obs.get((v_n, base_k))) is not None:
                cur_bts = normalize_epoch_sec(entry.get("base_bal_ts"), default=0.0) or pos_ts
                if b_obs.ts is None or b_obs.ts >= cur_bts - 1e-6:
                    entry["free_base"] = round(b_obs.free, 6)
                    if b_obs.ts is not None:
                        entry["base_bal_ts"] = round(b_obs.ts, 3)
                    if b_obs.reflected_locked is not None:
                        if b_obs.reflected_locked > 1e-8:
                            entry["locked_base_qty"] = round(b_obs.reflected_locked, 6)
                        else:
                            entry.pop("locked_base_qty", None)
                            entry.pop("locked_base_usd", None)
                    if (px_v := safe_pos_float(entry.get("price"), default=0.0) or 0.0) > 0.0:
                        reprice_position_balance_fields(entry, px_v)
            if quote_u and (q_obs := cache_obs.get((v_n, quote_u))) is not None:
                cur_qts = normalize_epoch_sec(entry.get("quote_bal_ts"), default=0.0) or pos_ts
                if q_obs.ts is None or q_obs.ts >= cur_qts - 1e-6:
                    entry["free_quote"] = round(q_obs.free, 2)
                    if q_obs.ts is not None:
                        entry["quote_bal_ts"] = round(q_obs.ts, 3)
    return out


@dataclass(frozen=True, slots=True)
class BalanceSnapshotRecord:
    """Immutable SSOT balance record separating raw venue free balance from synthetic post-balance order locks.

    Enforces the conservation invariant ``runtime_free + total_locked == free_raw + venue_reflected_locked``
    (whenever ``free_raw >= synthetic_deduct``) so runtime prompt views and persisted Unified NAV cannot drift.
    """

    free_raw: float
    total_locked: float = 0.0
    pre_balance_locked: float = 0.0
    reflected_locked: Optional[float] = None

    @property
    def synthetic_deduct(self) -> float:
        pre = max(0.0, self.pre_balance_locked)
        if pre <= 0.0:
            return 0.0
        eff = pre
        if self.reflected_locked is not None:
            eff = min(pre, max(0.0, max(0.0, self.total_locked) - max(0.0, self.reflected_locked)))
        return eff if eff > 1e-8 else 0.0

    @property
    def runtime_free(self) -> float:
        return max(0.0, max(0.0, self.free_raw) - self.synthetic_deduct)

    @property
    def venue_reflected_locked(self) -> float:
        return max(0.0, max(0.0, self.total_locked) - self.synthetic_deduct)

    @property
    def deduct_scale(self) -> float:
        pre = max(0.0, self.pre_balance_locked)
        return (self.synthetic_deduct / pre) if pre > 1e-12 else 0.0

    def persisted_symbol_locked(self, symbol_total_locked: float, symbol_pre_balance_locked: float) -> float:
        return max(0.0, max(0.0, symbol_total_locked) - max(0.0, symbol_pre_balance_locked) * self.deduct_scale)

    def to_persisted_cash_fields(
        self,
        *,
        decimals: int = 2,
        quote_locked: Optional[float] = None,
        pre_quote_locked: float = 0.0,
    ) -> dict[str, float]:
        pers_lq = (
            self.persisted_symbol_locked(quote_locked, pre_quote_locked)
            if quote_locked is not None
            else self.venue_reflected_locked
        )
        return {"locked_quote_usd": round(pers_lq, decimals)} if pers_lq > 1e-6 else {}

    def to_persisted_position_fields(
        self,
        *,
        symbol_locked_quote: float,
        symbol_pre_quote: float,
        quote_record: Optional["BalanceSnapshotRecord"],
        symbol_locked_base: float,
        symbol_pre_base: float,
        price: float = 0.0,
        trailing_peak: Optional[float] = None,
        dd_from_peak_pct: Optional[float] = None,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        pers_lq = (
            quote_record.persisted_symbol_locked(symbol_locked_quote, symbol_pre_quote)
            if quote_record is not None
            else max(0.0, symbol_locked_quote)
        )
        pers_lb = self.persisted_symbol_locked(symbol_locked_base, symbol_pre_base)
        if pers_lq > 1e-6:
            out["locked_quote_usd"] = round(pers_lq, 2)
        if pers_lb > 1e-12:
            out["locked_base_qty"] = round(pers_lb, 8)
            if price > 0.0:
                out["locked_base_usd"] = round(pers_lb * price, 2)
        if trailing_peak is not None:
            out["trailing_peak"] = trailing_peak
            if dd_from_peak_pct is not None:
                out["dd_from_peak_pct"] = dd_from_peak_pct
        return out


def compute_venue_nav_breakdown(
    *,
    positions: object = None,
    quote_cash: object = None,
    venue_balance_caches: object = None,
    venue_balances_state: object = None,
    live_prices: object = None,
    venue_overrides: object = None,
    exclude_venues: tuple[str, ...] = (),
    scope: str = "all",
) -> dict[str, object]:
    """Canonical SSOT multi-venue NAV calculator and position/cash deduplicator."""

    excl_set = {normalize_venue_name(v) or str(v).strip().lower() for v in (exclude_venues or ()) if v}
    scope_norm = str(scope or "all").strip().lower()
    live_src = (
        live_prices.get("items")
        if isinstance(live_prices, dict) and isinstance(live_prices.get("items"), dict)
        else (live_prices if isinstance(live_prices, dict) else {})
    )
    live_ft = (
        live_prices.get("fetchtime")
        if isinstance(live_prices, dict) and isinstance(live_prices.get("fetchtime"), dict)
        else {}
    )
    live_px_map: dict[str, float] = {}
    live_px_ts_map: dict[str, Optional[float]] = {}
    for raw_k, raw_v in live_src.items():
        if not (isinstance(raw_k, str) and raw_k.strip()):
            continue
        if (px := extract_current_price_item(raw_v)) is None:
            continue
        canon_k = canonical_display_symbol(raw_k) or raw_k.strip().upper()
        ts_v, ts_ok = extract_current_price_timestamp_sec(raw_v, live_ft.get(raw_k))
        live_px_map[canon_k] = px
        live_px_ts_map[canon_k] = ts_v if (ts_ok and ts_v > 0.0) else None

    disk_pos_rows: list[dict] = []
    disk_cash_items: list[tuple[str, object]] = []
    live_pos_rows: list[dict] = []
    live_cash_items: list[tuple[str, object]] = []

    fallback_px_by_va: dict[tuple[str, str], float] = {}
    if isinstance(venue_balances_state, dict):
        for key, entry in venue_balances_state.items():
            if not (isinstance(key, str) and key.startswith("pos:") and isinstance(entry, dict)):
                continue
            parts = key.split(":", 2)
            v_f = normalize_venue_name(entry.get("venue") or (parts[1] if len(parts) > 1 else ""))
            s_raw = str(entry.get("symbol") or (parts[2] if len(parts) > 2 else "")).strip()
            s_f = canonical_display_symbol(s_raw, venue=v_f) or s_raw.upper()
            b_f = canonical_balance_asset_key(entry.get("base") or s_f, venue=v_f) or s_f
            b_canon = canonical_base_asset(entry.get("base") or s_f) or ""
            px_f = safe_pos_float(entry.get("price"), default=0.0) or 0.0
            if v_f and px_f > 0.0:
                if s_f:
                    fallback_px_by_va[(v_f, s_f)] = px_f
                if b_f:
                    fallback_px_by_va[(v_f, b_f)] = px_f
                if b_canon:
                    fallback_px_by_va[(v_f, b_canon)] = px_f

    balance_cache_venues: set[str] = set()
    if isinstance(venue_balance_caches, Mapping) and venue_balance_caches:
        cache_obs, _ = extract_balance_observations_from_caches(venue_balance_caches)
        for raw_v, doc in venue_balance_caches.items():
            if isinstance(doc, Mapping) and isinstance(doc.get("items"), Mapping) and doc.get("items"):
                if (v_n := normalize_venue_name(doc.get("venue") or raw_v)):
                    balance_cache_venues.add(v_n)
        for (v_n, a_k), obs in cache_obs.items():
            free_v = obs.free
            locked_v = obs.reflected_locked or 0.0
            if free_v <= 0.0 and locked_v <= 0.0:
                continue
            if a_k in _NAV_QUOTE_CASH_ASSETS:
                disk_cash_items.append(
                    (f"{v_n}:{a_k}", {"venue": v_n, "asset": a_k, "free": free_v, "locked_quote_usd": locked_v})
                )
            else:
                is_eq_a = is_equity_venue_or_suffix(a_k, v_n)
                sym_a = a_k if is_eq_a else ensure_quote_suffix(a_k, "USD")
                fb_px = (
                    fallback_px_by_va.get((v_n, a_k))
                    or fallback_px_by_va.get((v_n, sym_a))
                    or fallback_px_by_va.get((v_n, canonical_base_asset(a_k) or a_k))
                    or 0.0
                )
                row: dict[str, Any] = {
                    "venue": v_n,
                    "symbol": sym_a,
                    "base": a_k,
                    "free_base": free_v,
                    "locked_base_qty": locked_v,
                }
                if fb_px > 0.0:
                    row["price"] = fb_px
                disk_pos_rows.append(row)

    if isinstance(venue_balances_state, dict):
        for key, entry in venue_balances_state.items():
            if not isinstance(key, str) or not isinstance(entry, dict):
                continue
            parts = key.split(":", 2)
            v_cand = normalize_venue_name(entry.get("venue") or (parts[1] if len(parts) > 1 else ""))
            if v_cand and v_cand in balance_cache_venues:
                continue
            if key.startswith("cash:"):
                a_cand = entry.get("asset") or (parts[2] if len(parts) > 2 else "USD")
                disk_cash_items.append((f"{v_cand}:{a_cand}", {**entry, "venue": v_cand, "asset": a_cand}))
            elif key.startswith("pos:"):
                s_cand = entry.get("symbol") or (parts[2] if len(parts) > 2 else key)
                disk_pos_rows.append({**entry, "venue": v_cand, "symbol": s_cand})

    if isinstance(positions, (list, tuple)):
        live_pos_rows.extend(p for p in positions if isinstance(p, dict))
    if isinstance(quote_cash, dict):
        live_cash_items.extend((str(ck or ""), cv) for ck, cv in quote_cash.items())

    def _matches_scope(v_name: str, sym_u: str = "") -> bool:
        if v_name in excl_set:
            return False
        if scope_norm not in ("equities", "crypto"):
            return True
        is_eq = is_equity_venue_or_suffix(sym_u, v_name)
        return is_eq if scope_norm == "equities" else (not is_eq)

    # Keep authoritative runtime views (positions/quote_cash) separate from raw disk snapshots
    # (venue_balances_state) so raw pre-deduction free balances are never combined with runtime locked balances.
    grouped_pos: dict[tuple[str, str], list[dict]] = {}
    live_pos_keys: set[tuple[str, str]] = set()
    for p in live_pos_rows:
        v_name = normalize_venue_name(p.get("venue")) or "unknown"
        sym_raw = canonical_display_symbol(p.get("symbol"), venue=v_name) or str(p.get("symbol") or "").strip().upper()
        if not _matches_scope(v_name, sym_raw):
            continue
        is_eq_p = is_equity_venue_or_suffix(sym_raw, v_name)
        base_src = canonical_balance_asset_key(p.get("base"), venue=v_name) or sym_raw
        base_p = canonical_base_asset(base_src) or sym_raw or f"IDX_{len(grouped_pos)}"
        g_key = (v_name, f"EQ:{base_p}" if is_eq_p else base_p)
        live_pos_keys.add(g_key)
        grouped_pos.setdefault(g_key, []).append(p)
    for p in disk_pos_rows:
        v_name = normalize_venue_name(p.get("venue")) or "unknown"
        sym_raw = canonical_display_symbol(p.get("symbol"), venue=v_name) or str(p.get("symbol") or "").strip().upper()
        if not _matches_scope(v_name, sym_raw):
            continue
        is_eq_p = is_equity_venue_or_suffix(sym_raw, v_name)
        base_src = canonical_balance_asset_key(p.get("base"), venue=v_name) or sym_raw
        base_p = canonical_base_asset(base_src) or sym_raw or f"IDX_{len(grouped_pos)}"
        g_key = (v_name, f"EQ:{base_p}" if is_eq_p else base_p)
        if g_key in live_pos_keys:
            continue
        grouped_pos.setdefault(g_key, []).append(p)

    venue_pos_usd: dict[str, float] = {}
    grouped_positions_out: list[tuple[str, dict, float]] = []
    for (v_name, _), grp_list in grouped_pos.items():
        free_val_u = 0.0
        locked_by_sym: dict[str, float] = {}
        for idx_g, gp in enumerate(grp_list):
            sym_raw_g = str(gp.get("symbol") or "").strip()
            sym_u = canonical_display_symbol(sym_raw_g, venue=v_name) or sym_raw_g.upper()
            live_px: float | None = None
            if live_px_map and sym_u:
                if has_equity_suffix(sym_u):
                    live_px = live_px_map.get(sym_u)
                else:
                    cand_keys: list[str] = [sym_u]
                    for cand in cache_symbol_variants(sym_u, quotes=compatible_quote_suffixes(sym_u)):
                        if isinstance(cand, str) and not has_equity_suffix(cand):
                            cand_k = canonical_display_symbol(cand, venue=v_name) or cand.upper()
                            if cand_k not in cand_keys:
                                cand_keys.append(cand_k)
                    matched_cands = [
                        (ck, live_px_map[ck], live_px_ts_map.get(ck))
                        for ck in cand_keys
                        if ck in live_px_map
                    ]
                    if len(matched_cands) == 1:
                        live_px = matched_cands[0][1]
                    elif matched_cands:
                        ts_cands = [c for c in matched_cands if c[2] is not None]
                        if ts_cands:
                            max_cand_ts = max(c[2] for c in ts_cands if c[2] is not None)
                            pool = [c for c in ts_cands if c[2] is not None and (max_cand_ts - c[2]) <= 600.0]
                        else:
                            pool = matched_cands
                        live_px = pool[0][1]
            if live_px is not None and (
                fb_live := safe_nonneg_float(gp.get("free_base"), None)
            ) is not None:
                gp_free_u = round(fb_live * live_px, 2)
            elif (val_u := safe_nonneg_float(gp.get("value_usd"), None)) is not None:
                gp_free_u = val_u
            else:
                gp_free_u = round(
                    (safe_nonneg_float(gp.get("free_base"), 0.0) or 0.0)
                    * (safe_nonneg_float(gp.get("price"), 0.0) or 0.0),
                    2,
                )
            free_val_u = max(free_val_u, gp_free_u)
            s_g = sym_u or f"SYM_{idx_g}"
            if live_px is not None and (
                lb_live := safe_pos_float(gp.get("locked_base_qty"), None)
            ) is not None:
                locked_usd = round(lb_live * live_px, 2)
            else:
                locked_usd = safe_nonneg_float(gp.get("locked_base_usd"), 0.0) or 0.0
            locked_by_sym[s_g] = max(locked_by_sym.get(s_g, 0.0), locked_usd)
        val_u = free_val_u + sum(locked_by_sym.values())
        venue_pos_usd[v_name] = venue_pos_usd.get(v_name, 0.0) + val_u
        grouped_positions_out.append((v_name, grp_list[0], val_u))

    def _parse_cash_entry(c_key: str, c: object) -> tuple[str, str, float, float] | None:
        k_parts = str(c_key or "").split(":", 1)
        v_fallback = normalize_venue_name(k_parts[0])
        if isinstance(c, dict):
            v_name = normalize_venue_name(c.get("venue")) or v_fallback or "unknown"
            c_asset_u = str(
                c.get("asset") or (k_parts[1] if len(k_parts) > 1 and k_parts[1].strip() else c_key or "USD")
            ).strip().upper() or "USD"
            f_u = first_pos_float(c.get("free"), c.get("free_quote_usd"), c.get("free_usd"), default=0.0) or 0.0
            l_u = safe_pos_float(c.get("locked_quote_usd"), default=0.0) or 0.0
        else:
            f_u = safe_pos_float(c, default=0.0) or 0.0
            if f_u <= 0.0:
                return None
            v_name = v_fallback or "unknown"
            c_asset_u = k_parts[1].strip().upper() if len(k_parts) > 1 and k_parts[1].strip() else "USD"
            l_u = 0.0
        if not _matches_scope(v_name):
            return None
        return v_name, c_asset_u, f_u, l_u

    dedup_cash: dict[tuple[str, str], tuple[float, float]] = {}
    live_cash_keys: set[tuple[str, str]] = set()
    for c_key, c in live_cash_items:
        parsed_c = _parse_cash_entry(c_key, c)
        if parsed_c is None:
            continue
        v_name, c_asset_u, f_u, l_u = parsed_c
        live_cash_keys.add((v_name, c_asset_u))
        prev_f, prev_l = dedup_cash.get((v_name, c_asset_u), (0.0, 0.0))
        dedup_cash[(v_name, c_asset_u)] = (max(prev_f, f_u), max(prev_l, l_u))
    for c_key, c in disk_cash_items:
        parsed_c = _parse_cash_entry(c_key, c)
        if parsed_c is None:
            continue
        v_name, c_asset_u, f_u, l_u = parsed_c
        if (v_name, c_asset_u) in live_cash_keys:
            continue
        prev_f, prev_l = dedup_cash.get((v_name, c_asset_u), (0.0, 0.0))
        dedup_cash[(v_name, c_asset_u)] = (max(prev_f, f_u), max(prev_l, l_u))

    venue_free_cash_usd: dict[str, float] = {}
    venue_locked_quote_usd: dict[str, float] = {}
    venue_cash_usd: dict[str, float] = {}
    for (v_name, _), (f_u, l_u) in dedup_cash.items():
        venue_free_cash_usd[v_name] = venue_free_cash_usd.get(v_name, 0.0) + f_u
        venue_locked_quote_usd[v_name] = venue_locked_quote_usd.get(v_name, 0.0) + l_u
        venue_cash_usd[v_name] = venue_cash_usd.get(v_name, 0.0) + f_u + l_u

    venues_usd: dict[str, float] = {}
    venue_breakdown: dict[str, dict[str, float]] = {}
    for v_k in sorted(set(venue_pos_usd) | set(venue_cash_usd)):
        p_u = round(venue_pos_usd.get(v_k, 0.0), 2)
        fc_u = round(venue_free_cash_usd.get(v_k, 0.0), 2)
        lq_u = round(venue_locked_quote_usd.get(v_k, 0.0), 2)
        c_u = round(venue_cash_usd.get(v_k, 0.0), 2)
        v_tot = round(p_u + c_u, 2)
        if v_tot > 0.0:
            venues_usd[v_k] = v_tot
            venue_breakdown[v_k] = {
                "pos_usd": p_u,
                "free_cash_usd": fc_u,
                "locked_quote_usd": lq_u,
                "cash_usd": c_u,
                "total_usd": v_tot,
            }

    if isinstance(venue_overrides, dict):
        for ov_k, ov_v in venue_overrides.items():
            v_norm = normalize_venue_name(ov_k) or str(ov_k or "").strip().lower()
            if not v_norm or not _matches_scope(v_norm):
                continue
            ov_f = safe_pos_float(ov_v, default=0.0) or 0.0
            if ov_f > venues_usd.get(v_norm, 0.0):
                venues_usd[v_norm] = round(ov_f, 2)
                if v_norm in venue_breakdown:
                    venue_breakdown[v_norm]["total_usd"] = round(ov_f, 2)

    venues_usd = {k: venues_usd[k] for k in sorted(venues_usd)}
    invested_usd = round(sum(venue_pos_usd.values()), 2)
    free_cash_usd = round(sum(venue_free_cash_usd.values()), 2)
    locked_quote_usd = round(sum(venue_locked_quote_usd.values()), 2)
    total_quote_reserve_usd = round(free_cash_usd + locked_quote_usd, 2)
    portfolio_total_usd = round(max(sum(venues_usd.values()), invested_usd + total_quote_reserve_usd), 2)

    return {
        "venues_usd": venues_usd,
        "venue_breakdown": venue_breakdown,
        "grouped_positions": grouped_positions_out,
        "venue_pos_usd": venue_pos_usd,
        "venue_free_cash_usd": venue_free_cash_usd,
        "venue_locked_quote_usd": venue_locked_quote_usd,
        "venue_cash_usd": venue_cash_usd,
        "invested_usd": invested_usd,
        "free_cash_usd": free_cash_usd,
        "locked_quote_usd": locked_quote_usd,
        "total_quote_reserve_usd": total_quote_reserve_usd,
        "portfolio_total_usd": portfolio_total_usd,
    }
