"""providers/portfolio_valuation.py -- Pure multi-venue NAV and position/cash deduplicator."""
from __future__ import annotations

from cachepaths import (
    cache_symbol_variants,
    canonical_base_asset,
    compatible_quote_suffixes,
    has_equity_suffix,
    is_equity_venue_or_suffix,
    normalize_venue_name,
    safe_nonneg_float,
    safe_pos_float,
)
from core.math_coercion import first_pos_float
from core.time_market import extract_current_price_item


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
