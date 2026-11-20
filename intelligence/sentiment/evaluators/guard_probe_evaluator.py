"""Pure read-only advisory probe of deterministic order guards (0 ms, zero LLM, zero ntfy).

Allows the autonomous AI executor to inspect what deterministic bot guards (G_PROF, G_QLIMIT2,
GI_WHALE, GI_OBOOK, GI_FUNDING, GLLM3) would decide for a candidate order without blocking
execution or emitting side-effect alerts, so the AI can consciously heed or override them.
"""
from __future__ import annotations

import os
from typing import Any, Mapping, Optional

from botcore import (
    _parse_bool_flag,
    clean_scalar_str,
    clean_scale,
    normalize_side,
    profit_diff_pct,
    resolve_now_ts,
    safe_float,
    safe_pos_float,
)
from cachepaths import (
    cache_path,
    cache_symbol_variants,
    is_equity_venue_or_suffix,
    matches_asset_scope,
    normalize_venue_name,
)
from guard_hierarchy import _load_margins, validate_guard_mode_cfg
from instrument_registry import casefold_get, resolve_symbol_identity
from intelligence.external.collectors.base_collector import (
    get_derivatives_collector,
    get_orderbook_collector,
    get_whale_collector,
)
from intelligence.external.guards.funding_crowding_guard import FundingCrowdingGuard
from intelligence.external.guards.orderbook_wall_guard import OrderbookWallGuard
from intelligence.external.guards.whale_divergence_guard import WhaleDivergenceGuard
from intelligence.internal.guards.exhaustion_guard import WeibullExhaustionGuard
from intelligence.internal.guards.guard_decision import (
    BrakeAction,
    fetch_guard_snapshot,
    unpack_guard_decision,
)
from intelligence.internal.guards.parabolic_guard import ParabolicSurgeGuard
from intelligence.internal.state.survival import (
    _read_cached_price_history,
    _resolve_trend_direction,
    _resolve_trend_duration,
    resolve_cached_gaussian_horizon,
)
from intelligence.macro.geopolitical_guard import GeopoliticalShockGuard
from state_io import read_json_dict


def _safe_load_margins(margins: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    if isinstance(margins, Mapping) and margins:
        return dict(margins)
    try:
        loaded = _load_margins()
        if isinstance(loaded, Mapping):
            return dict(loaded)
    except Exception:
        pass
    return {}


def _mode_enabled(m: Mapping[str, Any], key: str) -> bool:
    if key not in m:
        return True
    try:
        return validate_guard_mode_cfg(m[key], key) != "off"
    except Exception:
        return str(m.get(key) or "").strip().lower() != "off"


def _compact_reason(raw_reason: str, max_len: int = 64) -> str:
    txt = " ".join(str(raw_reason or "").split()).strip()
    if len(txt) <= max_len:
        return txt
    return txt[: max_len - 3].rstrip() + "..."


def _record_decision_flag(
    flags: list[dict[str, Any]],
    guard_code: str,
    dec: Any,
    default_reason: str,
) -> float:
    allowed, brake, scale, reason = unpack_guard_decision(dec, default_reason=default_reason)
    if allowed and brake not in (BrakeAction.DEFER_WAIT, BrakeAction.DOWNSCALE_QTY) and scale >= 1.0:
        return 1.0
    action = "VETO" if (not allowed or scale <= 0.0) else "SCALE"
    flags.append({
        "guard": guard_code,
        "action": action,
        "scale": 0.0 if action == "VETO" else round(clean_scale(scale, default=1.0), 3),
        "reason": _compact_reason(reason),
    })
    return 0.0 if action == "VETO" else clean_scale(scale, default=1.0)


def _lookup_symbol_entry(mapping: Any, venue: str, symbol: str) -> Optional[dict[str, Any]]:
    if not isinstance(mapping, Mapping) or not mapping:
        return None
    ven = normalize_venue_name(clean_scalar_str(venue) or "")
    sym = clean_scalar_str(symbol) or ""
    if not sym:
        return None
    cands: list[str] = []
    if ven:
        cands.append(f"{ven}:{sym}")
    cands.append(sym)
    for var in cache_symbol_variants(sym, include_base=False):
        if ven:
            cands.append(f"{ven}:{var}")
        cands.append(var)
    for key in cands:
        hit = casefold_get(mapping, key)
        if isinstance(hit, dict):
            return hit
    return None


def _read_local_geo_assessment() -> Optional[dict[str, Any]]:
    try:
        p = cache_path("geopolitical_eval.json")
        if p and os.path.exists(p):
            data = read_json_dict(p, default=None, recover_previous=True)
            if isinstance(data, dict) and data:
                return data
    except Exception:
        pass
    return None


def probe_deterministic_guards(
    symbol: str,
    side: str,
    price: float,
    *,
    venue: str = "",
    margins: Optional[Mapping[str, Any]] = None,
    microstructure_entry: Optional[Mapping[str, Any]] = None,
    price_info: Optional[Mapping[str, Any]] = None,
    geopolitical_assessment: Optional[Any] = None,
    opposite_fill_px: Optional[float] = None,
    price_history: Optional[list[tuple[float, float]]] = None,
    now_ts: Optional[float] = None,
    allow_disk_cache: bool = True,
) -> dict[str, Any]:
    """Dry-run local deterministic guards (0 ms, no LLM, no ntfy) and return all advisory flags.

    Evaluates G_PROF, G_QLIMIT2 (Parabolic + Weibull), GI_WHALE, GI_OBOOK, GI_FUNDING, and GLLM3.
    """
    side_u = normalize_side(side)
    sym_s = clean_scalar_str(symbol) or ""
    ven_s = normalize_venue_name(clean_scalar_str(venue) or "")
    px_f = safe_pos_float(price, default=0.0) or 0.0
    eff_now = resolve_now_ts(now_ts)

    base_out: dict[str, Any] = {
        "symbol": sym_s,
        "venue": ven_s,
        "side": side_u,
        "price": px_f,
        "overall": "OK",
        "suggested_scale": 1.0,
        "flags": [],
    }
    if side_u not in ("BUY", "SELL") or not sym_s or px_f <= 0.0:
        return base_out

    m = _safe_load_margins(margins)
    flags: list[dict[str, Any]] = []
    min_scale = 1.0
    is_tok_or_eq = bool(
        resolve_symbol_identity(sym_s).is_tokenized_equity or is_equity_venue_or_suffix(sym_s, ven_s)
    )

    # 1. G_PROF: Deterministic profit margin vs last opposite fill
    ref_px = safe_pos_float(opposite_fill_px, default=None)
    if ref_px is not None:
        prof_pct = safe_pos_float(m.get(ven_s), default=None) or safe_pos_float(m.get("default"), default=1.15) or 1.15
        diff_pct = profit_diff_pct(side_u, px_f, ref_px)
        if diff_pct < prof_pct:
            flags.append({
                "guard": "G_PROF",
                "action": "VETO",
                "scale": 0.0,
                "reason": f"margin={diff_pct:+.2f}%<{prof_pct:.2f}%(ref={ref_px:g})",
            })
            min_scale = 0.0

    # 2. G_QLIMIT2: Parabolic surge & Weibull exhaustion
    if _mode_enabled(m, "internal_guard_mode"):
        hist = price_history
        if hist is None and allow_disk_cache:
            try:
                hist = _read_cached_price_history(sym_s, window_seconds=7200.0, now=eff_now)
            except Exception:
                hist = None
        if hist:
            try:
                p_guard = ParabolicSurgeGuard(
                    surge_threshold_pct=safe_pos_float(m.get("parabolic_surge_pct"), default=15.0) or 15.0,
                    pullback_required_pct=safe_float(m.get("parabolic_pullback_pct"), default=2.0),
                )
                p_dec = p_guard.check(sym_s, side_u, px_f, price_history=hist, now=eff_now)
                min_scale = min(min_scale, _record_decision_flag(flags, "G_QLIMIT2:parabolic", p_dec, "parabolic_surge"))
            except Exception:
                pass

        dur_sec = 0.0
        trend_dir: Optional[str] = None
        p90_d: Optional[float] = None
        med_d: Optional[float] = None
        gauss_h: Optional[float] = None
        if isinstance(price_info, Mapping):
            lt_h = safe_pos_float(price_info.get("long_trend_hours"), default=None)
            if lt_h is not None:
                dur_sec = lt_h * 3600.0
            trend_dir = clean_scalar_str(price_info.get("long_trend_dir"))
            p90_d = safe_pos_float(price_info.get("weibull_p90_d"), default=None)
            med_d = safe_pos_float(price_info.get("weibull_median_d"), default=None)
        elif allow_disk_cache:
            try:
                dur_sec = safe_pos_float(_resolve_trend_duration(sym_s), default=0.0) or 0.0
                trend_dir = _resolve_trend_direction(sym_s)
                p90_d, med_d, gauss_h = resolve_cached_gaussian_horizon(sym_s)
            except Exception:
                pass
        if dur_sec > 0.0:
            try:
                e_guard = WeibullExhaustionGuard(
                    default_p90_days=safe_pos_float(m.get("weibull_p90_days"), default=7.0) or 7.0,
                    policy=str(m.get("weibull_exhaustion_policy") or "downscale"),
                    exhausted_scale=clean_scale(m.get("weibull_exhausted_scale"), default=0.25, min_positive=True),
                )
                e_dec = e_guard.check(
                    sym_s,
                    side_u,
                    trend_duration_seconds=dur_sec,
                    p90_days=p90_d,
                    median_days=med_d,
                    trend_direction=trend_dir,
                    gaussian_horizon_days=gauss_h,
                )
                min_scale = min(min_scale, _record_decision_flag(flags, "G_QLIMIT2:weibull", e_dec, "weibull_exhausted"))
            except Exception:
                pass

    # 3. External Microstructure Guards (GI_WHALE, GI_OBOOK, GI_FUNDING)
    ext_enabled = _mode_enabled(m, "external_guard_mode")
    micro = microstructure_entry if isinstance(microstructure_entry, Mapping) else None

    if ext_enabled and not is_tok_or_eq and _mode_enabled(m, "whale_guard_mode"):
        whale_snap: Any = None
        if micro is not None and (
            micro.get("taker_buy_sell_ratio") is not None or micro.get("whale_long_pct") is not None
        ):
            whale_snap = micro
        elif micro is None and allow_disk_cache:
            try:
                whale_snap = fetch_guard_snapshot(get_whale_collector(), sym_s, now=eff_now, allow_network=False)
            except Exception:
                whale_snap = None
        if whale_snap is not None:
            try:
                w_guard = WhaleDivergenceGuard(
                    severe_cascade_oi_pct=safe_float(m.get("severe_cascade_oi_pct"), default=-5.0),
                    min_taker_ratio=safe_pos_float(m.get("min_taker_ratio"), default=0.75) or 0.75,
                    taker_downscale_threshold=safe_pos_float(m.get("taker_downscale_threshold"), default=0.90) or 0.90,
                    taker_downscaled_scale=clean_scale(m.get("taker_downscaled_scale"), default=0.50, min_positive=True),
                )
                w_dec = w_guard.check(sym_s, side_u, snapshot=whale_snap, now=eff_now)
                min_scale = min(min_scale, _record_decision_flag(flags, "GI_WHALE", w_dec, "whale_divergence"))
            except Exception:
                pass

    if ext_enabled and _mode_enabled(m, "orderbook_wall_guard_mode"):
        ob_snap: Any = None
        if micro is not None:
            if not _parse_bool_flag(micro.get("ob_stale_frozen"), False) and (
                micro.get("ob_imbalance") is not None or micro.get("ask_wall_usd") is not None or micro.get("bid_wall_usd") is not None
            ):
                ob_snap = micro
        elif allow_disk_cache:
            try:
                ob_snap = fetch_guard_snapshot(get_orderbook_collector(), sym_s, now=eff_now, allow_network=False)
            except Exception:
                ob_snap = None
        if ob_snap is not None:
            try:
                ob_guard = OrderbookWallGuard(
                    min_buy_imbalance=clean_scale(m.get("min_buy_imbalance"), default=0.25, min_positive=True),
                    max_sell_imbalance=clean_scale(m.get("max_sell_imbalance"), default=0.75, min_positive=True),
                    whale_wall_usd_limit=safe_pos_float(m.get("whale_wall_usd_limit"), default=1_000_000.0) or 1_000_000.0,
                )
                ob_dec = ob_guard.check(sym_s, side_u, snapshot=ob_snap, now=eff_now)
                min_scale = min(min_scale, _record_decision_flag(flags, "GI_OBOOK", ob_dec, "orderbook_wall"))
            except Exception:
                pass

    if ext_enabled and not is_tok_or_eq and _mode_enabled(m, "funding_guard_mode"):
        deriv_snap: Any = None
        if micro is not None and micro.get("funding_rate") is not None:
            deriv_snap = micro
        elif micro is None and allow_disk_cache:
            try:
                deriv_snap = fetch_guard_snapshot(get_derivatives_collector(), sym_s, now=eff_now, allow_network=False)
            except Exception:
                deriv_snap = None
        if deriv_snap is not None:
            try:
                f_guard = FundingCrowdingGuard(
                    max_long_funding_rate=safe_pos_float(m.get("funding_max_long_rate"), default=0.0005) or 0.0005,
                    min_short_funding_rate=safe_float(m.get("funding_min_short_rate"), default=-0.0005),
                    crowding_policy=str(m.get("funding_crowding_policy") or "downscale"),
                    crowded_scale=clean_scale(m.get("funding_crowded_scale"), default=0.50, min_positive=True),
                )
                f_dec = f_guard.check(sym_s, side_u, telemetry=deriv_snap, now=eff_now)
                min_scale = min(min_scale, _record_decision_flag(flags, "GI_FUNDING", f_dec, "funding_crowding"))
            except Exception:
                pass

    # 4. GLLM3: Cached Geopolitical Shock Shield (0 ms local cache)
    if _mode_enabled(m, "geopolitical_guard_mode"):
        geo_assess = geopolitical_assessment
        if geo_assess is None and allow_disk_cache:
            geo_assess = _read_local_geo_assessment()
        if geo_assess is not None:
            try:
                g_dec = GeopoliticalShockGuard().check(sym_s, side_u, geo_assess)
                min_scale = min(min_scale, _record_decision_flag(flags, "GLLM3", g_dec, "geopolitical_shock"))
            except Exception:
                pass

    overall = "OK"
    if any(f["action"] == "VETO" for f in flags):
        overall = "VETO"
    elif flags:
        overall = "SCALE"

    base_out["overall"] = overall
    base_out["suggested_scale"] = round(min_scale, 3)
    base_out["flags"] = flags
    return base_out


def summarize_symbol_guard_advisories(
    tradable_positions: Any,
    prices_and_trends: Any,
    microstructure: Any,
    recent_orders: Any,
    geopolitical_threat: Any,
    *,
    now_ts: Optional[float] = None,
    margins: Optional[Mapping[str, Any]] = None,
    scope: str = "all",
) -> list[dict[str, Any]]:
    """Build compact non-blocking deterministic guard probe summaries ONLY for flagged symbols in scope."""
    if not isinstance(tradable_positions, list) or not tradable_positions:
        return []
    m = _safe_load_margins(margins)
    eff_now = resolve_now_ts(now_ts)
    scope_norm = (clean_scalar_str(scope) or "all").lower()
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for pos in tradable_positions:
        if not isinstance(pos, Mapping):
            continue
        sym = clean_scalar_str(pos.get("symbol")) or ""
        ven = normalize_venue_name(clean_scalar_str(pos.get("venue")) or "")
        px = safe_pos_float(pos.get("price"), default=0.0) or 0.0
        if not sym or px <= 0.0:
            continue
        if scope_norm != "all" and not matches_asset_scope(sym, ven, scope_norm):
            continue
        key = (ven, sym)
        if key in seen:
            continue
        seen.add(key)

        pinfo = _lookup_symbol_entry(prices_and_trends, ven, sym)
        minfo = _lookup_symbol_entry(microstructure, ven, sym)
        ro = _lookup_symbol_entry(recent_orders, ven, sym)

        side_results: dict[str, list[dict[str, Any]]] = {}
        for side_u in ("BUY", "SELL"):
            opp_key = "last_sell" if side_u == "BUY" else "last_buy"
            opp_entry = ro.get(opp_key) if isinstance(ro, Mapping) else None
            opp_px = safe_pos_float(
                opp_entry.get("price") if isinstance(opp_entry, Mapping) else pos.get(f"{opp_key}_px"),
                default=None,
            )
            probe = probe_deterministic_guards(
                sym,
                side_u,
                px,
                venue=ven,
                margins=m,
                microstructure_entry=minfo,
                price_info=pinfo,
                geopolitical_assessment=geopolitical_threat,
                opposite_fill_px=opp_px,
                now_ts=eff_now,
                allow_disk_cache=False,
            )
            if probe["flags"]:
                side_results[side_u] = probe["flags"]

        if side_results:
            out.append({
                "venue": ven,
                "symbol": sym,
                "price": px,
                "BUY": side_results.get("BUY", []),
                "SELL": side_results.get("SELL", []),
            })
    return out


def format_guard_advisory_line(entry: Mapping[str, Any]) -> str:
    """Format a single flagged instrument's non-blocking guard probe for Section 3C."""
    ven = clean_scalar_str(entry.get("venue")) or ""
    sym = clean_scalar_str(entry.get("symbol")) or ""
    if not sym:
        return ""

    def _fmt_side(flags_list: Any) -> str:
        if not isinstance(flags_list, list) or not flags_list:
            return "OK"
        tokens: list[str] = []
        for f in flags_list:
            if not isinstance(f, Mapping):
                continue
            act = str(f.get("action") or "WARN").upper()
            g = str(f.get("guard") or "GUARD")
            sc = safe_pos_float(f.get("scale"), default=None)
            rsn = _compact_reason(str(f.get("reason") or ""), max_len=64)
            if act == "SCALE" and sc is not None and sc < 1.0:
                tokens.append(f"SCALE({sc:g}x):{g}({rsn})")
            else:
                tokens.append(f"{act}:{g}({rsn})")
        return ", ".join(tokens) if tokens else "OK"

    buy_s = _fmt_side(entry.get("BUY"))
    sell_s = _fmt_side(entry.get("SELL"))
    ven_tag = f"[{ven}] " if ven else ""
    return f"    - {ven_tag}{sym}: BUY={buy_s} | SELL={sell_s}"


__all__ = [
    "format_guard_advisory_line",
    "probe_deterministic_guards",
    "summarize_symbol_guard_advisories",
]
