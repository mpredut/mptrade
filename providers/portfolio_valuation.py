"""providers/portfolio_valuation.py -- Pure multi-venue NAV, balance snapshot SSOT, and deduplicator."""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Callable, Mapping, Optional

from cachepaths import (
    VENUE_BALANCES_STATE_FILENAME,
    cache_symbol_variants,
    canonical_base_asset,
    compatible_quote_suffixes,
    has_equity_suffix,
    is_equity_venue_or_suffix,
    normalize_venue_name,
    resolve_cachedb_filepath,
    safe_nonneg_float,
    safe_pos_float,
)
from core.math_coercion import first_pos_float
from core.time_market import extract_current_price_item
from state_io import read_json_dict, read_json_dict_cached


INTERNAL_BALANCE_SCRATCH_KEYS: frozenset[str] = frozenset(
    {
        "_balance_ts",
        "_base_balance_ts",
        "_quote_balance_ts",
        "_prev_locked_quote_usd",
        "_prev_locked_base_qty",
    }
)


def strip_internal_balance_scratch_keys(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of ``entry`` with all internal balance scratch keys removed."""
    return {k: v for k, v in entry.items() if k not in INTERNAL_BALANCE_SCRATCH_KEYS}


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
) -> dict[tuple[str, str], BalanceObservation]:
    """Build atomic ``(venue, asset) -> BalanceObservation`` records from ``venue_balances_state.json``.

    Preserves ``reflected_locked`` grouped strictly by matching timestamp bucket so stale position snapshots
    cannot contaminate newer observations, while retaining quote locks even when ``cash:*`` is absent or older.
    """
    obs_by_va: dict[tuple[str, str], BalanceObservation] = {}
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
        a_u = raw_a.strip().upper()
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
            else p_sym_raw.upper()
        )
        res_b, res_q = resolve_assets_fn(p_sym, v.get("base"), v.get("quote"))
        b_u = str(res_b or "").strip().upper()
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
        a_u = str(v.get("asset") or "").strip().upper()
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


def resolve_venue_balances_state_path(cachedb_dir: str) -> str:
    """Resolve the canonical disk path for ``venue_balances_state.json`` inside ``cachedb_dir``."""
    return str(resolve_cachedb_filepath(os.path.join(cachedb_dir, VENUE_BALANCES_STATE_FILENAME)))


def load_venue_balances_state_dict(
    cachedb_dir: Optional[str],
    *,
    use_cache: bool = False,
    read_json_fn: Optional[Callable[[str], Any]] = None,
    require_exists: bool = False,
) -> dict[str, Any]:
    """SSOT owner reader for ``venue_balances_state.json`` across ``state/`` and root cache layouts."""
    if not cachedb_dir:
        return {}
    candidates = list(
        dict.fromkeys((
            resolve_venue_balances_state_path(cachedb_dir),
            os.path.join(cachedb_dir, "state", VENUE_BALANCES_STATE_FILENAME),
            os.path.join(cachedb_dir, VENUE_BALANCES_STATE_FILENAME),
        ))
    )
    for cand in candidates:
        if require_exists and not (os.path.exists(cand) or os.path.exists(f"{cand}.previous")):
            continue
        if callable(read_json_fn):
            raw = read_json_fn(cand)
        elif use_cache:
            raw = read_json_dict_cached(cand)
        else:
            raw = read_json_dict(cand, default={})
        if isinstance(raw, dict) and raw:
            return raw
    return {}


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

    def to_persisted_cash_fields(self, *, decimals: int = 2) -> dict[str, float]:
        pers_lq = self.venue_reflected_locked
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
    live_px_map = {
        raw_k.strip().upper(): px
        for raw_k, raw_v in live_src.items()
        if isinstance(raw_k, str) and raw_k.strip() and (px := extract_current_price_item(raw_v)) is not None
    }

    disk_pos_rows: list[dict] = []
    disk_cash_items: list[tuple[str, object]] = []
    live_pos_rows: list[dict] = []
    live_cash_items: list[tuple[str, object]] = []

    if isinstance(venue_balances_state, dict):
        for key, entry in venue_balances_state.items():
            if not isinstance(key, str) or not isinstance(entry, dict):
                continue
            parts = key.split(":", 2)
            v_cand = entry.get("venue") or (parts[1] if len(parts) > 1 else "")
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
        sym_raw = str(p.get("symbol") or "").strip().upper()
        if not _matches_scope(v_name, sym_raw):
            continue
        is_eq_p = is_equity_venue_or_suffix(sym_raw, v_name)
        base_p = canonical_base_asset(p.get("base") or sym_raw) or sym_raw or f"IDX_{len(grouped_pos)}"
        g_key = (v_name, f"EQ:{base_p}" if is_eq_p else base_p)
        live_pos_keys.add(g_key)
        grouped_pos.setdefault(g_key, []).append(p)
    for p in disk_pos_rows:
        v_name = normalize_venue_name(p.get("venue")) or "unknown"
        sym_raw = str(p.get("symbol") or "").strip().upper()
        if not _matches_scope(v_name, sym_raw):
            continue
        is_eq_p = is_equity_venue_or_suffix(sym_raw, v_name)
        base_p = canonical_base_asset(p.get("base") or sym_raw) or sym_raw or f"IDX_{len(grouped_pos)}"
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
            sym_u = str(gp.get("symbol") or "").strip().upper()
            live_px: float | None = None
            if live_px_map and sym_u:
                if sym_u in live_px_map:
                    live_px = live_px_map[sym_u]
                elif not has_equity_suffix(sym_u):
                    for cand in cache_symbol_variants(sym_u, quotes=compatible_quote_suffixes(sym_u)):
                        if isinstance(cand, str) and not has_equity_suffix(cand) and cand in live_px_map:
                            live_px = live_px_map[cand]
                            break
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
