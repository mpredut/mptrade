"""Deterministic mathematical and structural invariant verifier for AI executor telemetry & prompts.

Single Source of Truth (SSOT) for cross-section prompt accuracy checks:
- Portfolio NAV conservation (`crypto_nav == invested + free_cash + locked_quote == sum(venues)`)
- Non-tradable (`disabled` / `observe_only`) registry ghost exclusion in live & fallback views
- Quote breakdown sum consistency against runtime `free_quote`
- Cross-venue arbitrage spread symmetry (`arb(A, B) == -arb(B, A)`)
- Price range position math (`15m` / `24h` / `7d`), swing extreme bounds, and orderbook wall ordering
- Scorecard LIFO temporal stability across clock ticks with zero new fills
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Any, Callable, Collection, Mapping, Optional, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from botcore import (
    clean_scalar_str,
    clamp_float,
    load_key_value_conf,
    resolve_now_ts,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
    topic_spec,
)
from cachepaths import has_equity_suffix, matches_asset_scope, normalize_venue_name
from instrument_registry import (
    casefold_get,
    disabled_registry_symbols,
    observe_only_registry_symbols,
)
from intelligence.sentiment.evaluators.price_series_evaluator import (
    stale_closed_equity_metric_keys,
)
from providers.ai_survival_guard import (
    _collect_canceled_order_ids,
    _extract_intent_exec_price,
    _extract_intent_exec_ts,
    _is_filled_ai_intent,
    build_normalized_filled_ai_intent_row,
    match_filled_intent_rows_lifo_in_window,
)
from providers.intent_state import extract_intent_side, parse_intent_gateway_config
from providers.venue_metadata import ProviderVenueMetadataMixin
from state_io import read_text_file

_NAV_LINE_RE = re.compile(
    r"(?:crypto|equities|total)_nav=\$([0-9,]+\.\d+),\s*invested=\$([0-9,]+\.\d+),\s*"
    r"free_cash=\$([0-9,]+\.\d+)[^,]*(?:,\s*locked_quote=\$([0-9,]+\.\d+)[^,]*)?,\s*"
    r"venues\(([^)]+)\)"
)
_VENUE_VAL_RE = re.compile(r"([a-z0-9_]+)=\$([0-9,]+\.\d+)")
_SEC3C_INST_RE = re.compile(r"^\s*-\s*\[([a-z0-9_]+)\]\s+([A-Za-z0-9_]+):(.*)$", re.MULTILINE)
_FREE_QUOTE_BD_RE = re.compile(r"free_([A-Z]+)=\$([0-9,]+\.?\d*)\s*\(([^)]+)\)")
_BD_ITEM_RE = re.compile(r"([A-Z0-9]+)=\$([0-9,]+\.?\d*)")
_SEC3_SYM_RE = re.compile(
    r"^\s*[\*-]\s*([A-Za-z0-9_]+)(?:\s*(?:\[[^\]]*\]|\([^)]*\)))*:\s*Price=\$?([0-9,]+(?:\.[0-9]+)?)(.*)$",
    re.MULTILINE,
)
_SEC3B_SYM_RE = re.compile(r"^\s*[\*-]\s*([A-Za-z0-9_]+)(?:\s*(?:\[[^\]]*\]|\([^)]*\)))*:(.*)$", re.MULTILINE)
_UNIFIED_SUBLINE_RE = re.compile(r"^\s{2,}-\s*(Trend|Book|Flow):\s*(.*)$")
_RANGE_RE = re.compile(
    r"(24h|7d)_range=\[([0-9.]+)(?:\.\.|,\s*)([0-9.]+)\]\s*(?:\(pos=|pos_(?:24h|7d)=)([0-9.]+)%"
)
_ARB_VS_RE = re.compile(r"arb_vs_([A-Za-z0-9_]+)=([+-]?[0-9.]+)bps")
_VOL_PROFILE_RE = re.compile(
    r"vol_profile\(POC=([0-9.]+)\[([+-]?[0-9.]+)%\].*?VAL=([0-9.]+)\[([+-]?[0-9.]+)%\],\s*VAH=([0-9.]+)\[([+-]?[0-9.]+)%\](?:,\s*va_pos=([0-9.-]+)%)?"
)
_SESSION_VWAP_RE = re.compile(
    r"session_vwap=\$([0-9,]+(?:\.[0-9]+)?)\s*\(([+-]?[0-9.]+)%(?:,\s*z=([+-]?[0-9.]+))?"
)
_OB_IMB_RE = re.compile(
    r"ob_imbalance=([0-9.]+)\s*\(bid_depth=\$([0-9.,]+[KMBkmb]?),\s*ask_depth=\$([0-9.,]+[KMBkmb]?)\)"
)
_VENUE_SPEC_ROW_RE = re.compile(
    r"^\s*\|\s*([a-z0-9_]+)\s*\|\s*([A-Za-z0-9_]+)\s*\|[^|]*\|[^|]*\|[^|]*\|\s*([0-9.eE+-]+)\s*\|\s*([0-9.eE+-]+)\s*\|\s*([0-9.eE+-]+)\s*\|$",
    re.MULTILINE,
)
_OPEN_ORDER_ROW_RE = re.compile(
    r"^\s*\|\s*([a-z0-9_]+)\s*\|\s*([A-Za-z0-9_]+)\s*\|\s*(BUY|SELL)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([^|]+?)\s*\|"
    r"[^|]*\|[^|]*\|[^|]*\|\s*([^|]+?)\s*\|[^|]*\|(?:\s*([^|]+?)\s*\|)?\s*([^|]+?)\s*\|$",
    re.MULTILINE,
)
_DIST_PCT_CELL_RE = re.compile(r"^([+-]?[0-9.]+)%$")
_FILLED_REM_RE = re.compile(r"^([0-9.]+)\(rem=([0-9.]+)\)$")
_PLAN_RR_RE = re.compile(r"([0-9.]+)(?:\([^)]*\))?/([0-9.]+)(?:\([^)]*\))?\s+RR=([0-9.]+)x")
_FRONT_RUN_RR_RE = re.compile(r"front_run=([0-9.]+)\(RR=([0-9.]+)x\)")
_ACTIVE_PLAN_LINE_RE = re.compile(
    r"^\s*-\s*\[([a-z0-9_]+)\]\s+([A-Za-z0-9_]+)\s+(BUY|SELL)@([0-9.]+)\s*\(([^)]+)\):\s*(.+)$",
    re.MULTILINE,
)
_ACTIVE_PLAN_SPOT_RE = re.compile(r"\bspot=([0-9.]+)")
_ACTIVE_PLAN_TGT_RE = re.compile(r"\btarget=([0-9.]+)(?:\s*\(([+-]?[0-9.]+)%\s*(vs_spot|vs_limit)\))?")
_ACTIVE_PLAN_INV_RE = re.compile(r"\binvalidation=([0-9.]+)(?:\s*\(([+-]?[0-9.]+)%\s*(vs_spot|vs_limit)\))?")
_ACTIVE_PLAN_RR_RE = re.compile(r"\bRR=([0-9.]+)x\b")
_VENUE_CASH_BD_RE = re.compile(r"\b([a-z0-9_]+)\s+([A-Z]+)=\$([0-9,]+\.?\d*)\s*\(([^)]+)\)")
_COMPACT_USD_MULT: dict[str, float] = {"K": 1_000.0, "M": 1_000_000.0, "B": 1_000_000_000.0}
_OB_MID_RE = re.compile(r"ob_mid=([0-9.]+)")
_BID_WALL_RE = re.compile(r"(?:bid_wall|est_bid_zone)=([0-9.]+)\s*\(")
_ASK_WALL_RE = re.compile(r"(?:ask_wall|est_ask_zone)=([0-9.]+)\s*\(")
_LIMITS_RE = re.compile(r"limits\(([^)]+)\)")
_LIMIT_NUM_RE = re.compile(r"\b(min_qty|step|tick)=([0-9.eE+-]+)")
_LIMIT_CAP_RE = re.compile(r"\b(max_buy|max_sell)=\$([0-9,.]+)")
_PX_FIELD_RE = re.compile(r"\bpx=([0-9.]+)")
_FREE_BASE_VAL_RE = re.compile(r"\bfree_[A-Za-z0-9_]+=([0-9.]+)\s*\(\$([0-9,.]+)\s*USD\)")
_AVG_COST_PNL_RE = re.compile(r"\bavg_cost=([0-9.]+)\s*\(pnl_vs_cost=([+-]?[0-9.]+)%\)")
_PEAK_DD_RE = re.compile(r"\bpeak_px=([0-9.]+)\s*\(dd_from_peak=([+-]?[0-9.]+)%\)")
_CYCLE_ANCHOR_LINE_RE = re.compile(
    r"^\s*-\s*\[([a-z0-9_]+)\]\s+([A-Za-z0-9_]+):\s*qty=[0-9.]+,.*?\bavg_cost=([0-9.]+)\b",
    re.MULTILINE,
)
_RANGE_EXT_RE = re.compile(
    r"(24h|7d)_range=\[([0-9.]+),\s*([0-9.]+)\]\s*pos_(?:24h|7d)=([0-9.]+)%\s*"
    r"\(dd_high=([+-]?[0-9.]+)%,\s*bounce_low=([+-]?[0-9.]+)%,\s*span=([0-9.]+)%\)"
)
_RANGE_15M_RE = re.compile(r"15m_range=\[([0-9.]+),\s*([0-9.]+)\]")
_BOTTOM_6H_RE = re.compile(
    r"bottom_6h\(prior_low=([0-9.]+),\s*1h_low=([0-9.]+)(?:,\s*(higher_low_pct|reclaim_pct)=([+-]?[0-9.]+)%)?\)"
)
_HORIZON_CHG_RE = re.compile(r"(?:^|[\s|,(])(15m|1h|6h|24h|7d)=([+-][0-9.]+)%")
_VS_BTC_BLOCK_RE = re.compile(r"vs_btc\(([^)]+)\)")
_OB_SPREAD_BBO_RE = re.compile(r"spread=([0-9.]+)bps\s*\(bid=([0-9.]+)/ask=([0-9.]+)\)")
_WALL_DIST_RE = re.compile(
    r"(bid_wall|est_bid_zone|ask_wall|est_ask_zone)=([0-9.]+)\s*\([^)]*dist=([+-]?[0-9.]+)%\)"
)
_XSTOCK_REF_RE = re.compile(
    r"(?:SPCXxUSD=\$([0-9.]+).*?)?SPCX=\$([0-9.]+).*?xstock_vs_underlying=([+-]?[0-9.]+)%"
)
_NUMERIC_HYGIENE_SPECS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("NONE_LITERAL", re.compile(r"(?:=None\b|\|\s*None\s*\||\(None\)|\[None\])")),
    ("NAN_OR_INF", re.compile(r"(?:[=:(,\[\s\$])(?:nan|NaN|[+-]?inf|Inf)\b")),
    ("DOUBLE_SIGN", re.compile(r"(?:[+-]{2}\d|\$\s*[+-]{2}|\$-[+])")),
    ("NEG_ZERO", re.compile(r"(?:-0\.0+%|-\$0\.00\b)")),
    ("SCI_NOTATION", re.compile(r"(?:[=:(,\[\s|])\d+(?:\.\d+)?[eE][+-]\d+\b")),
    ("DOUBLE_UNIT", re.compile(r"(?:%%|\$\$)")),
)


@dataclass(frozen=True)
class InvariantViolation:
    code: str
    subject: str
    detail: str

    def format_line(self) -> str:
        return f"[{self.code}] {self.subject}: {self.detail}"


def _parse_money(raw: str) -> Optional[float]:
    s = str(raw or "").strip().replace(",", "")
    if not s:
        return None
    mult = _COMPACT_USD_MULT.get(s[-1].upper(), 1.0)
    num_s = s[:-1] if mult > 1.0 else s
    val = safe_float(num_s, default=None)
    return (val * mult) if val is not None else None


def _check_range_pos(
    violations: list[InvariantViolation],
    code: str,
    sym: str,
    label: str,
    px: float,
    lo: Optional[float],
    hi: Optional[float],
    pos: Optional[float],
    tol_pct: float,
) -> None:
    if lo is None or hi is None or pos is None or hi <= lo or px <= 0.0:
        return
    expected = clamp_float(((px - lo) / (hi - lo)) * 100.0, 0.0, 100.0)
    if abs(pos - expected) > tol_pct:
        violations.append(
            InvariantViolation(
                code=code,
                subject=f"{sym}:{label}",
                detail=f"reported pos={pos:.2f}% vs expected {expected:.2f}% (px={px:g}, range=[{lo:g}..{hi:g}])",
            )
        )


def audit_snapshot_invariants(
    snapshot: Any,
    *,
    non_tradable_symbols: Collection[str] = (),
    nav_tol_usd: float = 0.05,
    arb_tol_bps: float = 0.3,
    range_tol_pct: float = 1.0,
    breakdown_tol_frac: float = 0.05,
    gateway_min_usd: float = 0.0,
) -> list[InvariantViolation]:
    """Audit mathematical and structural invariants on a ``MacroTelemetrySnapshot`` or serialized dict."""
    violations: list[InvariantViolation] = []
    non_trad_u = {str(s).strip().upper() for s in non_tradable_symbols if str(s).strip()}

    if isinstance(snapshot, Mapping):
        pt = snapshot.get("prices_and_trends") or {}
        micro = snapshot.get("microstructure") or {}
        ps = snapshot.get("portfolio_summary") or {}
    else:
        pt = getattr(snapshot, "prices_and_trends", None) or {}
        micro = getattr(snapshot, "microstructure", None) or {}
        ps = getattr(snapshot, "portfolio_summary", None) or {}

    if not isinstance(pt, Mapping):
        pt = {}
    if not isinstance(micro, Mapping):
        micro = {}
    if not isinstance(ps, Mapping):
        ps = {}

    # 1. Portfolio NAV conservation
    pr = ps.get("portfolio_risk_engine") if isinstance(ps.get("portfolio_risk_engine"), Mapping) else ps.get("portfolio_risk")
    if isinstance(pr, Mapping):
        nav = safe_nonneg_float(
            pr.get("unified_nav_usd")
            or pr.get("crypto_nav_usd")
            or pr.get("equities_nav_usd")
            or pr.get("total_nav_usd"),
            default=None,
        )
        inv = safe_nonneg_float(
            pr.get("invested_exposure_usd") if pr.get("invested_exposure_usd") is not None else pr.get("invested_usd"),
            default=None,
        )
        free_c = safe_nonneg_float(
            pr.get("global_free_cash_usd") if pr.get("global_free_cash_usd") is not None else pr.get("free_cash_usd"),
            default=None,
        )
        locked_q = safe_nonneg_float(pr.get("locked_quote_usd"), default=0.0)
        if nav is not None and inv is not None and free_c is not None:
            comp_sum = inv + free_c + (locked_q or 0.0)
            if abs(nav - comp_sum) > nav_tol_usd:
                violations.append(
                    InvariantViolation(
                        "NAV_CONSERVATION",
                        "portfolio_risk",
                        f"nav={nav:.2f} != invested({inv:.2f}) + free_cash({free_c:.2f}) + locked({locked_q:.2f}) = {comp_sum:.2f}",
                    )
                )
        v_map = (
            pr.get("venue_nav_breakdown_usd")
            if isinstance(pr.get("venue_nav_breakdown_usd"), Mapping)
            else pr.get("venue_nav_usd")
        )
        if nav is not None and isinstance(v_map, Mapping) and v_map:
            v_sum = sum(safe_nonneg_float(v, default=0.0) or 0.0 for v in v_map.values())
            if abs(nav - v_sum) > nav_tol_usd:
                violations.append(
                    InvariantViolation(
                        "NAV_CONSERVATION",
                        "portfolio_risk.venues",
                        f"nav={nav:.2f} != sum(venues)={v_sum:.2f}",
                    )
                )

    # 2 & 3. Non-tradable ghost exclusion & free_quote_breakdown reconciliation
    tpos = ps.get("tradable_positions")
    if isinstance(tpos, Sequence) and not isinstance(tpos, (str, bytes)):
        seen_vs: set[tuple[str, str]] = set()
        for pos in tpos:
            if not isinstance(pos, Mapping):
                continue
            ven = normalize_venue_name(pos.get("venue"))
            sym = str(pos.get("symbol") or "").strip()
            sym_u = sym.upper()
            if sym_u and sym_u in non_trad_u:
                violations.append(
                    InvariantViolation(
                        "NO_NON_TRADABLE_GHOSTS",
                        f"{ven}:{sym}",
                        "disabled/observe_only registry symbol leaked into tradable_positions",
                    )
                )
            if sym_u and (ven, sym_u) in seen_vs:
                violations.append(
                    InvariantViolation(
                        "DUPLICATE_TRADABLE_POSITION",
                        f"{ven}:{sym}",
                        "duplicate (venue, symbol) entry in tradable_positions",
                    )
                )
            seen_vs.add((ven, sym_u))
            fq = safe_nonneg_float(pos.get("free_quote"), default=None)
            bd = pos.get("free_quote_breakdown")
            if fq is not None and isinstance(bd, Mapping) and bd:
                bd_sum = sum(safe_nonneg_float(v, default=0.0) or 0.0 for v in bd.values())
                if bd_sum > 0.0 and abs(bd_sum - fq) / max(bd_sum, fq, 1.0) > breakdown_tol_frac:
                    violations.append(
                        InvariantViolation(
                            "FREE_QUOTE_BREAKDOWN_CONSISTENCY",
                            f"{ven}:{sym}",
                            f"free_quote={fq:.2f} diverges from sum(breakdown)={bd_sum:.2f}",
                        )
                    )

    # 4. Cross-venue arbitrage symmetry
    for sym, minfo in micro.items():
        if not isinstance(minfo, Mapping):
            continue
        peer = clean_scalar_str(minfo.get("cross_venue_peer"))
        arb_a = safe_float(minfo.get("cross_venue_arb_bps"), default=None)
        if not peer or arb_a is None:
            continue
        peer_m = casefold_get(micro, peer)
        if not isinstance(peer_m, Mapping):
            continue
        peer_back = clean_scalar_str(peer_m.get("cross_venue_peer"))
        arb_b = safe_float(peer_m.get("cross_venue_arb_bps"), default=None)
        if peer_back and peer_back.upper() == str(sym).upper() and arb_b is not None:
            if str(sym).upper() < str(peer).upper() and abs(arb_a + arb_b) > arb_tol_bps:
                violations.append(
                    InvariantViolation(
                        "CROSS_VENUE_ARB_SYMMETRY",
                        f"{sym}<->{peer}",
                        f"arb({sym})={arb_a:+.1f}bps vs arb({peer})={arb_b:+.1f}bps (sum={arb_a + arb_b:+.1f}bps)",
                    )
                )

    # 5, 6 & 7. Price range math, swing wave bounds, and orderbook wall ordering
    for sym, pinfo in pt.items():
        if not isinstance(pinfo, Mapping):
            continue
        px = safe_pos_float(pinfo.get("price"), default=None)
        if px is None:
            continue
        _check_range_pos(
            violations,
            "RANGE_POS_MATH",
            str(sym),
            "24h",
            px,
            safe_pos_float(pinfo.get("low_24h"), default=None),
            safe_pos_float(pinfo.get("high_24h"), default=None),
            safe_float(pinfo.get("pos_24h_pct"), default=None),
            range_tol_pct,
        )
        _check_range_pos(
            violations,
            "RANGE_POS_MATH",
            str(sym),
            "7d",
            px,
            safe_pos_float(pinfo.get("low_7d"), default=None),
            safe_pos_float(pinfo.get("high_7d"), default=None),
            safe_float(pinfo.get("pos_7d_pct"), default=None),
            range_tol_pct,
        )
        _check_range_pos(
            violations,
            "RANGE_POS_MATH",
            str(sym),
            "15m",
            px,
            safe_pos_float(pinfo.get("low_15m"), default=None),
            safe_pos_float(pinfo.get("high_15m"), default=None),
            safe_float(pinfo.get("pos_15m_pct"), default=None),
            range_tol_pct,
        )
        maj_hi = safe_pos_float(
            pinfo.get("major_swing_high_px") if pinfo.get("major_swing_high_px") is not None else pinfo.get("major_high"),
            default=None,
        )
        maj_lo = safe_pos_float(
            pinfo.get("major_swing_low_px") if pinfo.get("major_swing_low_px") is not None else pinfo.get("major_low"),
            default=None,
        )
        if maj_hi is not None and px > maj_hi * 1.001:
            violations.append(
                InvariantViolation(
                    "SWING_WAVE_EXTREMES",
                    str(sym),
                    f"live price={px:g} exceeds major_high={maj_hi:g}",
                )
            )
        if maj_lo is not None and px < maj_lo * 0.999:
            violations.append(
                InvariantViolation(
                    "SWING_WAVE_EXTREMES",
                    str(sym),
                    f"live price={px:g} below major_low={maj_lo:g}",
                )
            )
        if has_equity_suffix(str(sym)):
            if leaked_eq := [k_s for k_s in stale_closed_equity_metric_keys(pinfo.get("quote_age_min")) if pinfo.get(k_s) is not None]:
                violations.append(
                    InvariantViolation(
                        "CLOSED_EQUITY_STALE_METRICS",
                        str(sym),
                        f"quote_age_min={pinfo.get('quote_age_min')} leaked stale keys: {', '.join(leaked_eq)}",
                    )
                )

    cmc_bw = (ps.get("cross_market_context") or {}) if isinstance(ps.get("cross_market_context"), Mapping) else {}
    eq_bw_map = cmc_bw.get("equity_bellwethers") if isinstance(cmc_bw.get("equity_bellwethers"), Mapping) else {}
    for bw_sym, bw_info in eq_bw_map.items():
        if isinstance(bw_info, Mapping) and has_equity_suffix(str(bw_sym)):
            if leaked_bw := [k_s for k_s in stale_closed_equity_metric_keys(bw_info.get("quote_age_min")) if bw_info.get(k_s) is not None]:
                violations.append(
                    InvariantViolation(
                        "CLOSED_EQUITY_STALE_METRICS",
                        f"bellwether:{bw_sym}",
                        f"quote_age_min={bw_info.get('quote_age_min')} leaked stale keys: {', '.join(leaked_bw)}",
                    )
                )

    gw_min_snap = safe_nonneg_float(gateway_min_usd, default=0.0) or 0.0
    if gw_min_snap > 0.0:
        for pos in ps.get("tradable_positions") or ():
            if not isinstance(pos, Mapping):
                continue
            p_ven = normalize_venue_name(pos.get("venue")) or str(pos.get("venue") or "").strip().lower()
            p_sym = str(pos.get("symbol") or "").strip()
            for cap_k in ("effective_max_buy_usd", "effective_max_sell_usd"):
                cap_v = safe_nonneg_float(pos.get(cap_k), default=None)
                if cap_v is not None and 0.0 < cap_v < gw_min_snap - 1e-6:
                    violations.append(
                        InvariantViolation(
                            "SNAPSHOT_SUB_GATEWAY_MIN_CAP",
                            f"{p_ven}:{p_sym}:{cap_k}",
                            f"{cap_k}=${cap_v:.2f} is below gateway_min_usd=${gw_min_snap:.2f}",
                        )
                    )

    for sym, minfo in micro.items():
        if not isinstance(minfo, Mapping):
            continue
        if has_equity_suffix(str(sym)):
            for px_k in ("best_bid", "best_ask", "ob_mid_price", "bid_wall_px", "ask_wall_px"):
                if (pv := safe_pos_float(minfo.get(px_k), default=None)) is not None:
                    frac = f"{pv:.8f}".rstrip("0").partition(".")[2]
                    if len(frac) > 4:
                        violations.append(
                            InvariantViolation(
                                "EQUITY_OB_PRECISION",
                                f"{sym}:{px_k}",
                                f"equity orderbook price {px_k}={pv} exceeds 4 decimal places",
                            )
                        )
        ob_mid = safe_pos_float(minfo.get("ob_mid_price"), default=None)
        if ob_mid is None:
            continue
        bid_w = safe_pos_float(minfo.get("bid_wall_px"), default=None)
        ask_w = safe_pos_float(minfo.get("ask_wall_px"), default=None)
        if bid_w is not None and bid_w > ob_mid * 1.001:
            violations.append(
                InvariantViolation(
                    "ORDERBOOK_WALL_ORDERING",
                    str(sym),
                    f"bid_wall_px={bid_w:g} > ob_mid={ob_mid:g}",
                )
            )
        if ask_w is not None and ask_w < ob_mid * 0.999:
            violations.append(
                InvariantViolation(
                    "ORDERBOOK_WALL_ORDERING",
                    str(sym),
                    f"ask_wall_px={ask_w:g} < ob_mid={ob_mid:g}",
                )
            )

    _audit_snapshot_positions_and_cycles(violations, ps, pt)
    _audit_snapshot_price_geometry(violations, pt)
    _audit_snapshot_micro_spread_and_walls(violations, micro)
    return violations


def _audit_snapshot_positions_and_cycles(
    violations: list[InvariantViolation],
    ps: Mapping[str, Any],
    pt: Mapping[str, Any],
) -> None:
    """Validate position valuation/PnL/peak math, cycle avg_cost parity, open-order dist%, and xStock premium."""
    pos_avg_by_vs: dict[tuple[str, str], float] = {}
    tpos = ps.get("tradable_positions")
    if isinstance(tpos, Sequence) and not isinstance(tpos, (str, bytes)):
        for pos in tpos:
            if not isinstance(pos, Mapping):
                continue
            ven = normalize_venue_name(pos.get("venue")) or str(pos.get("venue") or "").strip().lower()
            sym = str(pos.get("symbol") or "").strip()
            px = safe_pos_float(pos.get("price"), default=None)
            if not sym or px is None:
                continue
            fb = safe_nonneg_float(pos.get("free_base"), default=None)
            val = safe_nonneg_float(pos.get("value_usd"), default=None)
            if fb is not None and val is not None and abs(fb * px - val) > max(0.05, val * 0.005):
                violations.append(
                    InvariantViolation(
                        "POSITION_VALUE_AND_ANCHOR_MATH",
                        f"{ven}:{sym}:value_usd",
                        f"free_base({fb:g})*px({px:g})={fb * px:.2f} != value_usd={val:.2f}",
                    )
                )
            avg_b = safe_pos_float(pos.get("active_avg_buy_px"), default=None)
            if avg_b is not None:
                pos_avg_by_vs[(ven, sym.upper())] = avg_b
                pnl_b = safe_float(
                    pos.get("pnl_vs_avg_pct") if pos.get("pnl_vs_avg_pct") is not None else pos.get("pnl_vs_avg_buy_pct"),
                    default=None,
                )
                if pnl_b is not None:
                    exp_p = ((px - avg_b) / avg_b) * 100.0
                    if abs(exp_p - pnl_b) > 0.15:
                        violations.append(
                            InvariantViolation(
                                "POSITION_VALUE_AND_ANCHOR_MATH",
                                f"{ven}:{sym}:pnl_vs_avg",
                                f"reported pnl={pnl_b:+.2f}% != expected {exp_p:+.2f}% (px={px:g}, avg={avg_b:g})",
                            )
                        )
            pk = safe_pos_float(
                pos.get("trailing_peak") if pos.get("trailing_peak") is not None else pos.get("pos_peak"),
                default=None,
            )
            if pk is not None:
                if px > pk * 1.001:
                    violations.append(
                        InvariantViolation(
                            "POSITION_VALUE_AND_ANCHOR_MATH",
                            f"{ven}:{sym}:peak",
                            f"live price={px:g} exceeds peak={pk:g}",
                        )
                    )
                dd_pk = safe_float(pos.get("dd_from_peak_pct"), default=None)
                if dd_pk is not None:
                    exp_dd = ((px - pk) / pk) * 100.0
                    if abs(exp_dd - dd_pk) > 0.15:
                        violations.append(
                            InvariantViolation(
                                "POSITION_VALUE_AND_ANCHOR_MATH",
                                f"{ven}:{sym}:dd_from_peak",
                                f"reported dd={dd_pk:+.2f}% != expected {exp_dd:+.2f}% (px={px:g}, peak={pk:g})",
                            )
                        )

    for cyc_k, ven_name in (("kraken_cycles", "kraken"), ("hyperliquid_cycles", "hyperliquid"), ("t212_cycles", "t212")):
        cmap = ps.get(cyc_k)
        if not isinstance(cmap, Mapping):
            continue
        for csym, cdata in cmap.items():
            if not isinstance(cdata, Mapping):
                continue
            c_avg = safe_pos_float(cdata.get("avg_cost_px"), default=None)
            p_avg = pos_avg_by_vs.get((ven_name, str(csym).strip().upper()))
            if c_avg is not None and p_avg is not None and abs(c_avg - p_avg) > max(1e-4, p_avg * 1e-4):
                violations.append(
                    InvariantViolation(
                        "POSITION_VALUE_AND_ANCHOR_MATH",
                        f"{ven_name}:{csym}:cycle_avg_cost",
                        f"cycle avg_cost_px={c_avg:g} != position active_avg_buy_px={p_avg:g}",
                    )
                )

    raw_oo = ps.get("open_limit_orders")
    if not isinstance(raw_oo, Sequence) or isinstance(raw_oo, (str, bytes)):
        raw_oo = ps.get("open_orders") if isinstance(ps.get("open_orders"), Sequence) else ()
    for oo in raw_oo:
        if not isinstance(oo, Mapping):
            continue
        sym_o = str(oo.get("symbol") or "").strip()
        px_o = safe_pos_float(oo.get("price"), default=None)
        dist_o = safe_float(oo.get("dist_from_spot_pct"), default=None)
        p_ref = casefold_get(pt, sym_o) if sym_o else None
        spot_p = safe_pos_float(p_ref.get("price"), default=None) if isinstance(p_ref, Mapping) else None
        if px_o is not None and spot_p is not None and dist_o is not None:
            exp_d = ((px_o - spot_p) / spot_p) * 100.0
            if abs(exp_d - dist_o) > 0.20:
                violations.append(
                    InvariantViolation(
                        "OPEN_ORDER_DIST_MATH",
                        f"{normalize_venue_name(oo.get('venue'))}:{sym_o}",
                        f"reported dist={dist_o:+.2f}% != expected {exp_d:+.2f}% (limit={px_o:g}, spot={spot_p:g})",
                    )
                )

    xs = ps.get("kraken_xstocks")
    if isinstance(xs, Mapping) and (rep_prem := safe_float(xs.get("xstock_vs_underlying_pct"), default=None)) is not None:
        u_px = safe_pos_float(xs.get("underlying_px"), default=None)
        pair_s = str(xs.get("listed_pair") or "SPCXxUSD").strip()
        x_ref = casefold_get(pt, pair_s)
        x_px = safe_pos_float(x_ref.get("price"), default=None) if isinstance(x_ref, Mapping) else safe_pos_float(xs.get("xstock_px"), default=None)
        if x_px is not None and u_px is not None:
            exp_x = ((x_px - u_px) / u_px) * 100.0
            if abs(exp_x - rep_prem) > 0.15:
                violations.append(
                    InvariantViolation(
                        "XSTOCK_PREMIUM_MATH",
                        pair_s,
                        f"reported xstock_vs_underlying={rep_prem:+.2f}% != expected {exp_x:+.2f}% (x={x_px:g}, u={u_px:g})",
                    )
                )


def _audit_snapshot_price_geometry(
    violations: list[InvariantViolation],
    pt: Mapping[str, Any],
) -> None:
    """Validate vs_btc deltas, range bounds/dd_high/bounce_low/span, swing bounds, bottom_6h, and VWAP z sign."""
    btc_ref = casefold_get(pt, "BTCUSDC")
    btc_map = btc_ref if isinstance(btc_ref, Mapping) else {}
    for sym, pinfo in pt.items():
        if not isinstance(pinfo, Mapping):
            continue
        sym_s = str(sym)
        px = safe_pos_float(pinfo.get("price"), default=None)
        if px is None:
            continue
        if sym_s.upper() != "BTCUSDC" and btc_map:
            for hor in ("15m", "1h", "6h", "24h", "7d"):
                vb = safe_float(pinfo.get(f"vs_btc_{hor}_pct"), default=None)
                sc = safe_float(pinfo.get(f"chg_{hor}_pct"), default=None)
                bc = safe_float(btc_map.get(f"chg_{hor}_pct"), default=None)
                if vb is not None and sc is not None and bc is not None and abs(vb - (sc - bc)) > 0.15:
                    violations.append(
                        InvariantViolation(
                            "VS_BTC_MATH",
                            f"{sym_s}:{hor}",
                            f"reported vs_btc={vb:+.2f}% != sym({sc:+.2f}%)-btc({bc:+.2f}%)={sc - bc:+.2f}%",
                        )
                    )
        for tag in ("15m", "24h", "7d"):
            lo = safe_pos_float(pinfo.get(f"low_{tag}"), default=None)
            hi = safe_pos_float(pinfo.get(f"high_{tag}"), default=None)
            if lo is None or hi is None:
                continue
            if lo > hi * 1.0001 or px < lo * 0.999 or px > hi * 1.001:
                violations.append(
                    InvariantViolation(
                        "RANGE_BOUNDS_AND_EXTREMES_MATH",
                        f"{sym_s}:{tag}",
                        f"px={px:g} inconsistent with {tag} range [{lo:g}, {hi:g}]",
                    )
                )
            dd_h = safe_float(pinfo.get(f"dd_{tag}_high_pct"), default=None)
            if dd_h is not None and abs(dd_h - ((px - hi) / hi) * 100.0) > 0.15:
                violations.append(
                    InvariantViolation(
                        "RANGE_BOUNDS_AND_EXTREMES_MATH",
                        f"{sym_s}:{tag}:dd_high",
                        f"reported dd_high={dd_h:+.2f}% != expected {((px - hi) / hi) * 100.0:+.2f}%",
                    )
                )
            bn_l = safe_float(pinfo.get(f"bounce_{tag}_low_pct"), default=None)
            if bn_l is not None and abs(bn_l - ((px - lo) / lo) * 100.0) > 0.15:
                violations.append(
                    InvariantViolation(
                        "RANGE_BOUNDS_AND_EXTREMES_MATH",
                        f"{sym_s}:{tag}:bounce_low",
                        f"reported bounce_low={bn_l:+.2f}% != expected {((px - lo) / lo) * 100.0:+.2f}%",
                    )
                )
            sp = safe_float(pinfo.get(f"span_{tag}_pct"), default=None)
            if sp is not None and abs(sp - ((hi - lo) / lo) * 100.0) > 0.25:
                violations.append(
                    InvariantViolation(
                        "RANGE_BOUNDS_AND_EXTREMES_MATH",
                        f"{sym_s}:{tag}:span",
                        f"reported span={sp:.2f}% != expected {((hi - lo) / lo) * 100.0:.2f}%",
                    )
                )
        sw_l = safe_pos_float(pinfo.get("swing_low_px"), default=None)
        sw_h = safe_pos_float(pinfo.get("swing_high_px"), default=None)
        if sw_l is not None and sw_h is not None and (sw_l > sw_h * 1.0001 or px < sw_l * 0.999 or px > sw_h * 1.001):
            violations.append(
                InvariantViolation(
                    "RANGE_BOUNDS_AND_EXTREMES_MATH",
                    f"{sym_s}:swing_wave",
                    f"px={px:g} outside swing_wave=[{sw_l:g}, {sw_h:g}]",
                )
            )
        if b6 := clean_scalar_str(pinfo.get("bottom_structure_6h")):
            p6l = safe_pos_float(pinfo.get("prior_6h_low_px"), default=None)
            r1l = safe_pos_float(pinfo.get("recent_1h_low_px"), default=None)
            l15 = safe_pos_float(pinfo.get("low_15m"), default=None)
            l24 = safe_pos_float(pinfo.get("low_24h"), default=None)
            if r1l is not None and l15 is not None and r1l > l15 * 1.0005:
                violations.append(
                    InvariantViolation(
                        "BOTTOM_STRUCTURE_MATH",
                        sym_s,
                        f"recent_1h_low_px={r1l:g} > low_15m={l15:g}",
                    )
                )
            if p6l is not None and r1l is not None and l24 is not None and min(p6l, r1l) < l24 * 0.999:
                violations.append(
                    InvariantViolation(
                        "BOTTOM_STRUCTURE_MATH",
                        sym_s,
                        f"min(prior_6h_low={p6l:g}, 1h_low={r1l:g}) < low_24h={l24:g}",
                    )
                )
            if b6 == "HIGHER_LOW_DOUBLE_BOTTOM" and p6l is not None and r1l is not None and r1l < p6l * 0.9995:
                violations.append(
                    InvariantViolation(
                        "BOTTOM_STRUCTURE_MATH",
                        sym_s,
                        f"HIGHER_LOW_DOUBLE_BOTTOM but 1h_low={r1l:g} < prior_6h_low={p6l:g}",
                    )
                )
        dv = safe_float(pinfo.get("dist_vs_session_vwap_pct"), default=None)
        vz = safe_float(pinfo.get("session_vwap_z"), default=None)
        if dv is not None and vz is not None and dv * vz < -0.02:
            violations.append(
                InvariantViolation(
                    "SESSION_VWAP_SIGN_MATH",
                    sym_s,
                    f"dist_vs_session_vwap_pct={dv:+.2f}% has opposite sign to session_vwap_z={vz:+.2f}",
                )
            )


def _audit_snapshot_micro_spread_and_walls(
    violations: list[InvariantViolation],
    micro: Mapping[str, Any],
) -> None:
    """Validate non-crossed BBO, spread_bps formula, and wall distance percentages vs ob_mid."""
    for sym, minfo in micro.items():
        if not isinstance(minfo, Mapping):
            continue
        sym_s = str(sym)
        bb = safe_pos_float(minfo.get("best_bid"), default=None)
        ba = safe_pos_float(minfo.get("best_ask"), default=None)
        if bb is not None and ba is not None:
            if bb > ba * 1.00001:
                violations.append(
                    InvariantViolation(
                        "ORDERBOOK_SPREAD_AND_WALL_DIST_MATH",
                        f"{sym_s}:bbo",
                        f"crossed book best_bid={bb:g} > best_ask={ba:g}",
                    )
                )
            sp_bps = safe_float(minfo.get("spread_bps"), default=None)
            if sp_bps is not None:
                exp_sp = ((ba - bb) / ((ba + bb) / 2.0)) * 10000.0
                if abs(exp_sp - sp_bps) > max(1.0, abs(exp_sp) * 0.05):
                    violations.append(
                        InvariantViolation(
                            "ORDERBOOK_SPREAD_AND_WALL_DIST_MATH",
                            f"{sym_s}:spread_bps",
                            f"reported spread={sp_bps:.2f}bps != expected {exp_sp:.2f}bps (bid={bb:g}, ask={ba:g})",
                        )
                    )
        mid = safe_pos_float(minfo.get("ob_mid_price"), default=None)
        if mid is not None:
            for wk in ("bid_wall", "ask_wall"):
                wpx = safe_pos_float(minfo.get(f"{wk}_px"), default=None)
                wdist = safe_float(minfo.get(f"{wk}_dist_pct"), default=None)
                if wpx is not None and wdist is not None:
                    exp_wd = ((wpx - mid) / mid) * 100.0
                    if abs(exp_wd - wdist) > 0.25:
                        violations.append(
                            InvariantViolation(
                                "ORDERBOOK_SPREAD_AND_WALL_DIST_MATH",
                                f"{sym_s}:{wk}_dist",
                                f"reported {wk}_dist={wdist:+.2f}% != expected {exp_wd:+.2f}% (wall={wpx:g}, mid={mid:g})",
                            )
                        )


def audit_scorecard_temporal_invariants(
    intents: Sequence[Mapping[str, Any]],
    *,
    t_early: float,
    t_late: float,
    scope: str = "crypto",
    lookback_hours: float = 48.0,
    fee_resolver: Optional[Callable[[str, str, str], Optional[float]]] = None,
) -> list[InvariantViolation]:
    """Verify LIFO closed pairs inside ``[t_late - lookback, t_early]`` stay immutable when no new fills occur."""
    if t_late <= t_early:
        return []
    items = [it for it in intents if isinstance(it, Mapping)]
    canceled = _collect_canceled_order_ids(items)

    def _rows_up_to(ts_limit: float) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for it in items:
            if not _is_filled_ai_intent(it, canceled):
                continue
            sym = clean_scalar_str(it.get("symbol")) or ""
            ven = normalize_venue_name(clean_scalar_str(it.get("venue")) or "")
            if scope and scope.strip().lower() != "all" and not matches_asset_scope(sym, ven, scope):
                continue
            side = extract_intent_side(it)
            if side not in ("BUY", "SELL"):
                continue
            ts_s = _extract_intent_exec_ts(it)
            px = _extract_intent_exec_price(it)
            if ts_s <= 0.0 or ts_s > ts_limit + 60.0 or px <= 0.0:
                continue
            rows.append(
                build_normalized_filled_ai_intent_row(
                    it,
                    venue=ven,
                    symbol=sym,
                    side=side,
                    ts_s=ts_s,
                    px=px,
                    fee_resolver=fee_resolver,
                )
            )
        return rows

    rows_early = _rows_up_to(t_early)
    rows_late = _rows_up_to(t_late)
    if len(rows_late) != len(rows_early):
        # New fills occurred between t_early and t_late; temporal invariance only applies with zero new fills.
        return []

    lb_sec = max(3600.0, (safe_pos_float(lookback_hours, default=48.0) or 48.0) * 3600.0)
    cut_early = t_early - lb_sec
    cut_late = t_late - lb_sec
    c_early, _, _ = match_filled_intent_rows_lifo_in_window(rows_early, cutoff_ts=cut_early)
    c_late, _, _ = match_filled_intent_rows_lifo_in_window(rows_late, cutoff_ts=cut_late)

    def _sig(pair: tuple[str, dict[str, Any], dict[str, Any]]) -> tuple[str, str, str, float, float, float]:
        kind, er, xr = pair
        return (
            kind,
            str(er.get("venue") or ""),
            str(er.get("symbol") or ""),
            round(safe_float(er.get("px"), default=0.0) or 0.0, 6),
            round(safe_float(xr.get("px"), default=0.0) or 0.0, 6),
            round(safe_float(xr.get("ts"), default=0.0) or 0.0, 3),
        )

    early_overlapping = sorted(_sig(p) for p in c_early if (safe_float(p[2].get("ts"), default=0.0) or 0.0) >= cut_late)
    late_sigs = sorted(_sig(p) for p in c_late)
    if early_overlapping != late_sigs:
        return [
            InvariantViolation(
                "SCORECARD_TEMPORAL_STABILITY",
                f"{scope}:{lookback_hours:g}h",
                f"closed pairs mutated without new fills: early={early_overlapping} vs late={late_sigs}",
            )
        ]
    return []


def audit_rendered_prompt_text(
    prompt_text: str,
    *,
    non_tradable_symbols: Collection[str] = (),
    nav_tol_usd: float = 0.05,
    arb_tol_bps: float = 0.3,
    range_tol_pct: float = 1.2,
    breakdown_tol_frac: float = 0.05,
    gateway_min_usd: float = 0.0,
) -> list[InvariantViolation]:
    """Audit mathematical and structural invariants directly on a rendered LLM prompt string."""
    violations: list[InvariantViolation] = []
    non_trad_u = {str(s).strip().upper() for s in non_tradable_symbols if str(s).strip()}
    gw_min_f = safe_nonneg_float(gateway_min_usd, default=0.0) or 0.0

    # 1. Section 3C NAV conservation
    if (m_nav := _NAV_LINE_RE.search(prompt_text)) is not None:
        nav = _parse_money(m_nav.group(1))
        inv = _parse_money(m_nav.group(2))
        free_c = _parse_money(m_nav.group(3))
        locked_q = _parse_money(m_nav.group(4)) if m_nav.group(4) is not None else 0.0
        if nav is not None and inv is not None and free_c is not None and locked_q is not None:
            comp = inv + free_c + locked_q
            if abs(nav - comp) > nav_tol_usd:
                violations.append(
                    InvariantViolation(
                        "PROMPT_NAV_CONSERVATION",
                        "Section3C.PortfolioRisk",
                        f"nav={nav:.2f} != invested+free+locked={comp:.2f}",
                    )
                )
        v_sum = sum(_parse_money(v_s) or 0.0 for _, v_s in _VENUE_VAL_RE.findall(m_nav.group(5)))
        if nav is not None and v_sum > 0.0 and abs(nav - v_sum) > nav_tol_usd:
            violations.append(
                InvariantViolation(
                    "PROMPT_NAV_CONSERVATION",
                    "Section3C.Venues",
                    f"nav={nav:.2f} != sum(venues)={v_sum:.2f}",
                )
            )

    # 2 & 3. Section 3C live tradable instruments: ghost check + free_quote_breakdown sum + precision/caps + position math
    px_by_vs: dict[tuple[str, str], float] = {}
    avg_by_vs: dict[tuple[str, str], float] = {}
    for m_inst in _SEC3C_INST_RE.finditer(prompt_text):
        ven, sym, rest = m_inst.group(1), m_inst.group(2), m_inst.group(3)
        if "free_" not in rest or "limits(" not in rest:
            continue
        m_px_row = _PX_FIELD_RE.search(rest)
        px_r = safe_pos_float(m_px_row.group(1), default=None) if m_px_row else None
        if px_r is not None:
            px_by_vs[(ven.lower(), sym.upper())] = px_r
            if (m_fbv := _FREE_BASE_VAL_RE.search(rest)) is not None:
                fb_q = safe_nonneg_float(m_fbv.group(1), default=None)
                val_u = _parse_money(m_fbv.group(2))
                if fb_q is not None and val_u is not None and abs(fb_q * px_r - val_u) > max(0.05, val_u * 0.005):
                    violations.append(
                        InvariantViolation(
                            "PROMPT_POSITION_MATH",
                            f"{ven}:{sym}:value_usd",
                            f"free_base({fb_q:g})*px({px_r:g})={fb_q * px_r:.2f} != reported ${val_u:.2f}",
                        )
                    )
            if (m_avg := _AVG_COST_PNL_RE.search(rest)) is not None:
                avg_v = safe_pos_float(m_avg.group(1), default=None)
                rep_pnl = safe_float(m_avg.group(2), default=None)
                if avg_v is not None:
                    avg_by_vs[(ven.lower(), sym.upper())] = avg_v
                    if rep_pnl is not None:
                        exp_pnl = ((px_r - avg_v) / avg_v) * 100.0
                        if abs(exp_pnl - rep_pnl) > 0.15:
                            violations.append(
                                InvariantViolation(
                                    "PROMPT_POSITION_MATH",
                                    f"{ven}:{sym}:pnl_vs_cost",
                                    f"reported pnl_vs_cost={rep_pnl:+.2f}% != expected {exp_pnl:+.2f}% (px={px_r:g}, avg={avg_v:g})",
                                )
                            )
            if (m_pk := _PEAK_DD_RE.search(rest)) is not None:
                pk_v = safe_pos_float(m_pk.group(1), default=None)
                rep_dd = safe_float(m_pk.group(2), default=None)
                if pk_v is not None:
                    if px_r > pk_v * 1.001:
                        violations.append(
                            InvariantViolation(
                                "PROMPT_POSITION_MATH",
                                f"{ven}:{sym}:peak_px",
                                f"px={px_r:g} exceeds peak_px={pk_v:g}",
                            )
                        )
                    if rep_dd is not None:
                        exp_dd = ((px_r - pk_v) / pk_v) * 100.0
                        if abs(exp_dd - rep_dd) > 0.15:
                            violations.append(
                                InvariantViolation(
                                    "PROMPT_POSITION_MATH",
                                    f"{ven}:{sym}:dd_from_peak",
                                    f"reported dd_from_peak={rep_dd:+.2f}% != expected {exp_dd:+.2f}% (px={px_r:g}, peak={pk_v:g})",
                                )
                            )
        if sym.upper() in non_trad_u:
            violations.append(
                InvariantViolation(
                    "PROMPT_NO_NON_TRADABLE_GHOSTS",
                    f"{ven}:{sym}",
                    "disabled/observe_only symbol rendered in Live Tradable Spot Instruments",
                )
            )
        for m_bd in _FREE_QUOTE_BD_RE.finditer(rest):
            q_name = m_bd.group(1)
            fq = _parse_money(m_bd.group(2))
            bd_vals = [_parse_money(v_s) for _, v_s in _BD_ITEM_RE.findall(m_bd.group(3))]
            bd_nums = [v for v in bd_vals if v is not None]
            if fq is not None and bd_nums:
                bd_sum = sum(bd_nums)
                if bd_sum > 0.0 and abs(bd_sum - fq) / max(bd_sum, fq, 1.0) > breakdown_tol_frac:
                    violations.append(
                        InvariantViolation(
                            "PROMPT_FREE_QUOTE_BREAKDOWN",
                            f"{ven}:{sym}:{q_name}",
                            f"free_{q_name}={fq:.2f} diverges from breakdown sum={bd_sum:.2f}",
                        )
                    )
        if (m_lim := _LIMITS_RE.search(rest)) is not None:
            lim_s = m_lim.group(1)
            lim_nums = {
                k: safe_float(v, default=None)
                for k, v in _LIMIT_NUM_RE.findall(lim_s)
            }
            for k_lim, v_lim in lim_nums.items():
                if v_lim is not None and v_lim <= 0.0:
                    violations.append(
                        InvariantViolation(
                            "PROMPT_VENUE_PRECISION",
                            f"{ven}:{sym}:{k_lim}",
                            f"non-positive {k_lim}={v_lim}",
                        )
                    )
            if ProviderVenueMetadataMixin.prefers_static_precision(ven):
                px_c = px_by_vs.get((ven.lower(), sym.upper()), 0.0)
                _, _, _, _, exp_step, exp_tick, _ = (
                    ProviderVenueMetadataMixin.resolve_venue_default_fee_and_precision(ven, px_c)
                )
                for k_lim, exp_v in (("min_qty", exp_step), ("step", exp_step), ("tick", exp_tick)):
                    got_v = lim_nums.get(k_lim)
                    if got_v is not None and abs(got_v - exp_v) > 1e-9:
                        violations.append(
                            InvariantViolation(
                                "PROMPT_VENUE_PRECISION",
                                f"{ven}:{sym}:{k_lim}",
                                f"reported {k_lim}={got_v:g} != static venue precision {exp_v:g}",
                            )
                        )
            if gw_min_f > 0.0:
                for cap_k, cap_s in _LIMIT_CAP_RE.findall(lim_s):
                    cap_v = _parse_money(cap_s)
                    if cap_v is not None and 0.0 < cap_v < gw_min_f - 1e-6:
                        violations.append(
                            InvariantViolation(
                                "PROMPT_SUB_GATEWAY_MIN_CAP",
                                f"{ven}:{sym}:{cap_k}",
                                f"{cap_k}=${cap_v:.2f} is below gateway_min_usd=${gw_min_f:.2f}",
                            )
                        )

    for m_vspec in _VENUE_SPEC_ROW_RE.finditer(prompt_text):
        ven, sym = m_vspec.group(1), m_vspec.group(2)
        lim_nums = {
            "min_qty": safe_float(m_vspec.group(3), default=None),
            "step": safe_float(m_vspec.group(4), default=None),
            "tick": safe_float(m_vspec.group(5), default=None),
        }
        for k_lim, v_lim in lim_nums.items():
            if v_lim is not None and v_lim <= 0.0:
                violations.append(
                    InvariantViolation(
                        "PROMPT_VENUE_PRECISION",
                        f"{ven}:{sym}:{k_lim}",
                        f"non-positive {k_lim}={v_lim}",
                    )
                )
        if ProviderVenueMetadataMixin.prefers_static_precision(ven):
            px_c = px_by_vs.get((ven.lower(), sym.upper()), 0.0)
            _, _, _, _, exp_step, exp_tick, _ = (
                ProviderVenueMetadataMixin.resolve_venue_default_fee_and_precision(ven, px_c)
            )
            for k_lim, exp_v in (("min_qty", exp_step), ("step", exp_step), ("tick", exp_tick)):
                got_v = lim_nums.get(k_lim)
                if got_v is not None and abs(got_v - exp_v) > 1e-9:
                    violations.append(
                        InvariantViolation(
                            "PROMPT_VENUE_PRECISION",
                            f"{ven}:{sym}:{k_lim}",
                            f"reported {k_lim}={got_v:g} != static venue precision {exp_v:g}",
                        )
                    )

    for m_cyc in _CYCLE_ANCHOR_LINE_RE.finditer(prompt_text):
        c_ven, c_sym = m_cyc.group(1).lower(), m_cyc.group(2).upper()
        c_avg = safe_pos_float(m_cyc.group(3), default=None)
        p_avg = avg_by_vs.get((c_ven, c_sym))
        if c_avg is not None and p_avg is not None and abs(c_avg - p_avg) > max(1e-4, p_avg * 1e-4):
            violations.append(
                InvariantViolation(
                    "PROMPT_CYCLE_ANCHOR_PARITY",
                    f"{c_ven}:{c_sym}",
                    f"cycle avg_cost={c_avg:g} != position avg_cost={p_avg:g}",
                )
            )

    open_order_dist_rows: list[tuple[str, str, float, float]] = []
    for m_oo in _OPEN_ORDER_ROW_RE.finditer(prompt_text):
        ven_o, sym_o, side_o = m_oo.group(1), m_oo.group(2), m_oo.group(3)
        px_o = safe_pos_float(m_oo.group(4), default=None)
        qty_o = safe_pos_float(m_oo.group(5), default=None)
        filled_cell = (m_oo.group(6) or "").strip()
        dist_cell = (m_oo.group(7) or "").strip()
        plan_cell = (m_oo.group(8) or "").strip()
        note_cell = (m_oo.group(9) or "").strip()
        if px_o is not None and (m_dc := _DIST_PCT_CELL_RE.match(dist_cell)) is not None:
            if (dist_v := safe_float(m_dc.group(1), default=None)) is not None:
                open_order_dist_rows.append((ven_o, sym_o, px_o, dist_v))
        if qty_o is not None and (m_rem := _FILLED_REM_RE.match(filled_cell)) is not None:
            f_v = safe_nonneg_float(m_rem.group(1), default=None)
            r_v = safe_nonneg_float(m_rem.group(2), default=None)
            if f_v is not None and r_v is not None and abs((f_v + r_v) - qty_o) > max(1e-6, qty_o * 1e-4):
                violations.append(
                    InvariantViolation(
                        "PROMPT_OPEN_ORDER_REM_MATH",
                        f"{ven_o}:{sym_o}",
                        f"filled({f_v:g})+rem({r_v:g}) != qty({qty_o:g})",
                    )
                )
        if px_o is not None and (m_prr := _PLAN_RR_RE.search(plan_cell)) is not None:
            t_v = safe_pos_float(m_prr.group(1), default=None)
            i_v = safe_pos_float(m_prr.group(2), default=None)
            rep_rr = safe_nonneg_float(m_prr.group(3), default=None)
            if t_v is not None and i_v is not None and rep_rr is not None:
                rew = (t_v - px_o) if side_o == "BUY" else (px_o - t_v)
                rsk = (px_o - i_v) if side_o == "BUY" else (i_v - px_o)
                if rsk > 0.0:
                    exp_rr = max(0.0, rew / rsk)
                    if abs(rep_rr - exp_rr) > 0.02:
                        violations.append(
                            InvariantViolation(
                                "PROMPT_OPEN_ORDER_RR_MATH",
                                f"{ven_o}:{sym_o}",
                                f"plan RR={rep_rr:.2f}x != expected {exp_rr:.2f}x (px={px_o}, tgt={t_v}, inv={i_v})",
                            )
                        )
                if (m_fr := _FRONT_RUN_RR_RE.search(note_cell)) is not None:
                    fr_px = safe_pos_float(m_fr.group(1), default=None)
                    rep_fr_rr = safe_nonneg_float(m_fr.group(2), default=None)
                    if fr_px is not None and rep_fr_rr is not None:
                        rew_fr = (t_v - fr_px) if side_o == "BUY" else (fr_px - t_v)
                        rsk_fr = (fr_px - i_v) if side_o == "BUY" else (i_v - fr_px)
                        if rsk_fr > 0.0:
                            exp_fr_rr = max(0.0, rew_fr / rsk_fr)
                            if abs(rep_fr_rr - exp_fr_rr) > 0.02:
                                violations.append(
                                    InvariantViolation(
                                        "PROMPT_OPEN_ORDER_RR_MATH",
                                        f"{ven_o}:{sym_o}:front_run",
                                        f"front_run RR={rep_fr_rr:.2f}x != expected {exp_fr_rr:.2f}x (fr={fr_px}, tgt={t_v}, inv={i_v})",
                                    )
                                )

    for m_ap in _ACTIVE_PLAN_LINE_RE.finditer(prompt_text):
        ven_p, sym_p, side_p = m_ap.group(1), m_ap.group(2), m_ap.group(3)
        entry_p = safe_pos_float(m_ap.group(4), default=None)
        meta_p, body_p = m_ap.group(5) or "", m_ap.group(6) or ""
        spot_m = _ACTIVE_PLAN_SPOT_RE.search(meta_p)
        spot_p = safe_pos_float(spot_m.group(1), default=None) if spot_m else None
        is_resting_p = "RESTING" in meta_p.upper()
        anc_p = entry_p if is_resting_p else spot_p
        m_tgt = _ACTIVE_PLAN_TGT_RE.search(body_p)
        m_inv = _ACTIVE_PLAN_INV_RE.search(body_p)
        tgt_p = safe_pos_float(m_tgt.group(1), default=None) if m_tgt else None
        inv_p = safe_pos_float(m_inv.group(1), default=None) if m_inv else None
        if anc_p is not None and anc_p > 0.0:
            for lbl_k, m_lvl, lvl_px in (("target", m_tgt, tgt_p), ("invalidation", m_inv, inv_p)):
                if m_lvl is not None and m_lvl.group(2) is not None and lvl_px is not None:
                    rep_pct = safe_float(m_lvl.group(2), default=None)
                    exp_pct = ((lvl_px - anc_p) / anc_p) * 100.0
                    if rep_pct is not None and abs(rep_pct - exp_pct) > 0.15:
                        violations.append(
                            InvariantViolation(
                                "PROMPT_ACTIVE_PLAN_RR_MATH",
                                f"{ven_p}:{sym_p}:{lbl_k}",
                                f"reported {lbl_k} dist={rep_pct:+.2f}% != expected {exp_pct:+.2f}% (px={lvl_px}, anc={anc_p})",
                            )
                        )
            if tgt_p is not None and inv_p is not None and (m_rrp := _ACTIVE_PLAN_RR_RE.search(body_p)) is not None:
                rep_rrp = safe_nonneg_float(m_rrp.group(1), default=None)
                rew_p = (tgt_p - anc_p) if side_p == "BUY" else (anc_p - tgt_p)
                rsk_p = (anc_p - inv_p) if side_p == "BUY" else (inv_p - anc_p)
                if rep_rrp is not None and rsk_p > 0.0:
                    exp_rrp = max(0.0, rew_p / rsk_p)
                    if abs(rep_rrp - exp_rrp) > 0.02:
                        violations.append(
                            InvariantViolation(
                                "PROMPT_ACTIVE_PLAN_RR_MATH",
                                f"{ven_p}:{sym_p}",
                                f"active plan RR={rep_rrp:.2f}x != expected {exp_rrp:.2f}x (anc={anc_p}, tgt={tgt_p}, inv={inv_p})",
                            )
                        )

    for ln_cash in prompt_text.splitlines():
        if "Available Free Quote Cash per Venue:" not in ln_cash:
            continue
        for m_cbd in _VENUE_CASH_BD_RE.finditer(ln_cash):
            ven_c, q_name = m_cbd.group(1), m_cbd.group(2)
            fq = _parse_money(m_cbd.group(3))
            bd_vals = [_parse_money(v_s) for _, v_s in _BD_ITEM_RE.findall(m_cbd.group(4))]
            bd_nums = [v for v in bd_vals if v is not None]
            if fq is not None and bd_nums:
                bd_sum = sum(bd_nums)
                if bd_sum > 0.0 and abs(bd_sum - fq) / max(bd_sum, fq, 1.0) > breakdown_tol_frac:
                    violations.append(
                        InvariantViolation(
                            "PROMPT_FREE_QUOTE_BREAKDOWN",
                            f"{ven_c}:{q_name}",
                            f"{q_name}={fq:.2f} diverges from breakdown sum={bd_sum:.2f}",
                        )
                    )

    # 4 & 5. Collect both single-line and unified multi-line symbol cards (Trend / Book / Flow)
    sec3_items: list[tuple[str, float, str]] = []
    sec3b_items: list[tuple[str, str]] = []
    prompt_lines = prompt_text.splitlines()
    cur_card_sym: Optional[str] = None
    cur_card_px: Optional[float] = None
    cur_trend_parts: list[str] = []
    cur_micro_parts: list[str] = []

    def _flush_card() -> None:
        if cur_card_sym is None:
            return
        if cur_card_px is not None and cur_card_px > 0.0 and cur_trend_parts:
            sec3_items.append((cur_card_sym, cur_card_px, " | ".join(cur_trend_parts)))
        if cur_micro_parts:
            sec3b_items.append((cur_card_sym, ", ".join(cur_micro_parts)))

    for raw_ln in prompt_lines:
        if (m_sub := _UNIFIED_SUBLINE_RE.match(raw_ln)) is not None and cur_card_sym is not None:
            sub_kind, sub_body = m_sub.group(1), m_sub.group(2)
            if sub_kind == "Trend":
                cur_trend_parts.append(sub_body)
            else:
                cur_micro_parts.append(sub_body)
            continue
        _flush_card()
        cur_card_sym = None
        cur_card_px = None
        cur_trend_parts = []
        cur_micro_parts = []
        if (m_s3 := _SEC3_SYM_RE.match(raw_ln)) is not None:
            sym = m_s3.group(1)
            px = _parse_money(m_s3.group(2))
            rest = m_s3.group(3)
            cur_card_sym = sym
            cur_card_px = px
            if rest.strip():
                cur_trend_parts.append(rest)
            continue
        if (m_3b := _SEC3B_SYM_RE.match(raw_ln)) is not None:
            sym, rest = m_3b.group(1), m_3b.group(2)
            cur_card_sym = sym
            if rest.strip():
                cur_micro_parts.append(rest)
    _flush_card()

    # 4. Section 3 range pos math, vol_profile order/deltas, and session_vwap delta
    for sym, px, rest in sec3_items:
        for m_rng in _RANGE_RE.finditer(rest):
            lbl = m_rng.group(1)
            lo = safe_pos_float(m_rng.group(2), default=None)
            hi = safe_pos_float(m_rng.group(3), default=None)
            pos = safe_float(m_rng.group(4), default=None)
            _check_range_pos(violations, "PROMPT_RANGE_POS_MATH", sym, lbl, px, lo, hi, pos, range_tol_pct)
        if (m_vp := _VOL_PROFILE_RE.search(rest)) is not None:
            poc = safe_pos_float(m_vp.group(1), default=None)
            poc_d = safe_float(m_vp.group(2), default=None)
            val = safe_pos_float(m_vp.group(3), default=None)
            val_d = safe_float(m_vp.group(4), default=None)
            vah = safe_pos_float(m_vp.group(5), default=None)
            vah_d = safe_float(m_vp.group(6), default=None)
            if poc is not None and val is not None and vah is not None:
                if not (val <= poc <= vah):
                    violations.append(
                        InvariantViolation(
                            "PROMPT_VOL_PROFILE_ORDER",
                            sym,
                            f"vol_profile out of order: VAL={val:g}, POC={poc:g}, VAH={vah:g}",
                        )
                    )
                for vp_lbl, lv, rep_d in (("POC", poc, poc_d), ("VAL", val, val_d), ("VAH", vah, vah_d)):
                    if rep_d is not None:
                        exp_d = ((px - lv) / lv) * 100.0
                        if abs(exp_d - rep_d) > 0.15:
                            violations.append(
                                InvariantViolation(
                                    "PROMPT_VOL_PROFILE_DELTA",
                                    f"{sym}:{vp_lbl}",
                                    f"reported {rep_d:+.2f}% vs expected {exp_d:+.2f}% (px={px:g}, {vp_lbl}={lv:g})",
                                )
                            )
        if (m_vw := _SESSION_VWAP_RE.search(rest)) is not None:
            vwap_px = _parse_money(m_vw.group(1))
            rep_vd = safe_float(m_vw.group(2), default=None)
            rep_vz = safe_float(m_vw.group(3), default=None) if m_vw.group(3) is not None else None
            if vwap_px is not None and vwap_px > 0.0 and rep_vd is not None:
                exp_vd = ((px - vwap_px) / vwap_px) * 100.0
                if abs(exp_vd - rep_vd) > 0.15:
                    violations.append(
                        InvariantViolation(
                            "PROMPT_SESSION_VWAP_DELTA",
                            sym,
                            f"reported {rep_vd:+.2f}% vs expected {exp_vd:+.2f}% (px={px:g}, vwap={vwap_px:g})",
                        )
                    )
            if rep_vd is not None and rep_vz is not None and rep_vd * rep_vz < -0.02:
                violations.append(
                    InvariantViolation(
                        "PROMPT_SESSION_VWAP_DELTA",
                        f"{sym}:z_sign",
                        f"session_vwap delta={rep_vd:+.2f}% has opposite sign to z={rep_vz:+.2f}",
                    )
                )

    # 5. Section 4 / unified-card cross-venue arb symmetry, OB imbalance math, and wall ordering
    arb_pairs: dict[tuple[str, str], float] = {}
    for sym, rest in sec3b_items:
        if (m_arb := _ARB_VS_RE.search(rest)) is not None:
            peer = m_arb.group(1)
            val = safe_float(m_arb.group(2), default=None)
            if val is not None:
                arb_pairs[(sym.upper(), peer.upper())] = val
        if (m_imb := _OB_IMB_RE.search(rest)) is not None:
            rep_imb = safe_float(m_imb.group(1), default=None)
            bd_v = _parse_money(m_imb.group(2))
            ad_v = _parse_money(m_imb.group(3))
            if rep_imb is not None and bd_v is not None and ad_v is not None and (bd_v + ad_v) > 0.0:
                exp_imb = bd_v / (bd_v + ad_v)
                if abs(exp_imb - rep_imb) > 0.015:
                    violations.append(
                        InvariantViolation(
                            "PROMPT_OB_IMBALANCE_MATH",
                            sym,
                            f"reported ob_imbalance={rep_imb:.3f} vs expected {exp_imb:.3f} (bid=${bd_v:.1f}, ask=${ad_v:.1f})",
                        )
                    )
        if (m_mid := _OB_MID_RE.search(rest)) is not None:
            ob_mid = safe_pos_float(m_mid.group(1), default=None)
            if ob_mid is not None:
                if (m_bw := _BID_WALL_RE.search(rest)) is not None:
                    bw_px = safe_pos_float(m_bw.group(1), default=None)
                    if bw_px is not None and bw_px > ob_mid * 1.001:
                        violations.append(
                            InvariantViolation(
                                "PROMPT_OB_WALL_ORDERING",
                                sym,
                                f"bid_wall={bw_px:g} > ob_mid={ob_mid:g}",
                            )
                        )
                if (m_aw := _ASK_WALL_RE.search(rest)) is not None:
                    aw_px = safe_pos_float(m_aw.group(1), default=None)
                    if aw_px is not None and aw_px < ob_mid * 0.999:
                        violations.append(
                            InvariantViolation(
                                "PROMPT_OB_WALL_ORDERING",
                                sym,
                                f"ask_wall={aw_px:g} < ob_mid={ob_mid:g}",
                            )
                        )
    for (s_a, s_b), v_a in arb_pairs.items():
        if s_a < s_b and (s_b, s_a) in arb_pairs:
            v_b = arb_pairs[(s_b, s_a)]
            if abs(v_a + v_b) > arb_tol_bps:
                violations.append(
                    InvariantViolation(
                        "PROMPT_CROSS_VENUE_ARB_SYMMETRY",
                        f"{s_a}<->{s_b}",
                        f"arb_vs_{s_b}={v_a:+.1f}bps vs arb_vs_{s_a}={v_b:+.1f}bps",
                    )
                )

    _audit_prompt_cross_section_math(violations, prompt_text, sec3_items, sec3b_items, px_by_vs, open_order_dist_rows)
    _audit_prompt_numeric_hygiene(violations, prompt_text)
    return violations


def _audit_prompt_cross_section_math(
    violations: list[InvariantViolation],
    prompt_text: str,
    sec3_items: Sequence[tuple[str, float, str]],
    sec3b_items: Sequence[tuple[str, str]],
    px_by_vs: Mapping[tuple[str, str], float],
    open_order_dist_rows: Sequence[tuple[str, str, float, float]],
) -> None:
    """Validate rendered range extremes, bottom_6h, vs_btc, OB spread/wall dist%, open-order Dist%, and xStock premium."""
    sec3_px_by_sym: dict[str, float] = {}
    hor_chg_by_sym: dict[str, dict[str, float]] = {}
    for sym, px, rest in sec3_items:
        sym_u = sym.upper()
        sec3_px_by_sym[sym_u] = px
        h_map: dict[str, float] = {}
        for m_hc in _HORIZON_CHG_RE.finditer(rest.split("vs_btc(", 1)[0]):
            if (hv := safe_float(m_hc.group(2), default=None)) is not None:
                h_map.setdefault(m_hc.group(1), hv)
        hor_chg_by_sym[sym_u] = h_map

        l24_found: Optional[float] = None
        for m_re in _RANGE_EXT_RE.finditer(rest):
            lbl = m_re.group(1)
            lo = safe_pos_float(m_re.group(2), default=None)
            hi = safe_pos_float(m_re.group(3), default=None)
            dd_h = safe_float(m_re.group(5), default=None)
            bn_l = safe_float(m_re.group(6), default=None)
            sp = safe_float(m_re.group(7), default=None)
            if lo is None or hi is None:
                continue
            if lbl == "24h":
                l24_found = lo
            if lo > hi * 1.0001 or px < lo * 0.999 or px > hi * 1.001:
                violations.append(
                    InvariantViolation(
                        "PROMPT_RANGE_EXTREMES_MATH",
                        f"{sym}:{lbl}",
                        f"px={px:g} inconsistent with {lbl}_range=[{lo:g}, {hi:g}]",
                    )
                )
            if dd_h is not None and abs(dd_h - ((px - hi) / hi) * 100.0) > 0.15:
                violations.append(
                    InvariantViolation(
                        "PROMPT_RANGE_EXTREMES_MATH",
                        f"{sym}:{lbl}:dd_high",
                        f"reported dd_high={dd_h:+.2f}% != expected {((px - hi) / hi) * 100.0:+.2f}%",
                    )
                )
            if bn_l is not None and abs(bn_l - ((px - lo) / lo) * 100.0) > 0.15:
                violations.append(
                    InvariantViolation(
                        "PROMPT_RANGE_EXTREMES_MATH",
                        f"{sym}:{lbl}:bounce_low",
                        f"reported bounce_low={bn_l:+.2f}% != expected {((px - lo) / lo) * 100.0:+.2f}%",
                    )
                )
            if sp is not None and abs(sp - ((hi - lo) / lo) * 100.0) > 0.25:
                violations.append(
                    InvariantViolation(
                        "PROMPT_RANGE_EXTREMES_MATH",
                        f"{sym}:{lbl}:span",
                        f"reported span={sp:.2f}% != expected {((hi - lo) / lo) * 100.0:.2f}%",
                    )
                )

        if (m_b6 := _BOTTOM_6H_RE.search(rest)) is not None:
            p6h = safe_pos_float(m_b6.group(1), default=None)
            r1h = safe_pos_float(m_b6.group(2), default=None)
            b_lbl, b_pct = m_b6.group(3), safe_float(m_b6.group(4), default=None) if m_b6.group(4) else None
            m_r15 = _RANGE_15M_RE.search(rest)
            l15 = safe_pos_float(m_r15.group(1), default=None) if m_r15 else None
            if r1h is not None and l15 is not None and r1h > l15 * 1.0005:
                violations.append(
                    InvariantViolation(
                        "PROMPT_BOTTOM_STRUCTURE_MATH",
                        sym,
                        f"bottom_6h 1h_low={r1h:g} > 15m_range low={l15:g}",
                    )
                )
            if p6h is not None and r1h is not None and l24_found is not None and min(p6h, r1h) < l24_found * 0.999:
                violations.append(
                    InvariantViolation(
                        "PROMPT_BOTTOM_STRUCTURE_MATH",
                        sym,
                        f"bottom_6h min({p6h:g}, {r1h:g}) < low_24h={l24_found:g}",
                    )
                )
            if p6h is not None and r1h is not None and b_lbl and b_pct is not None:
                num_px = r1h if b_lbl == "higher_low_pct" else px
                exp_bp = ((num_px - p6h) / p6h) * 100.0
                if abs(exp_bp - b_pct) > 0.15:
                    violations.append(
                        InvariantViolation(
                            "PROMPT_BOTTOM_STRUCTURE_MATH",
                            f"{sym}:{b_lbl}",
                            f"reported {b_lbl}={b_pct:+.2f}% != expected {exp_bp:+.2f}%",
                        )
                    )

    btc_h = hor_chg_by_sym.get("BTCUSDC", {})
    if btc_h:
        for sym, _, rest in sec3_items:
            sym_u = sym.upper()
            if sym_u == "BTCUSDC" or (m_vb := _VS_BTC_BLOCK_RE.search(rest)) is None:
                continue
            sym_h = hor_chg_by_sym.get(sym_u, {})
            for m_vh in _HORIZON_CHG_RE.finditer(m_vb.group(1)):
                hor = m_vh.group(1)
                rep_vb = safe_float(m_vh.group(2), default=None)
                if rep_vb is not None and hor in sym_h and hor in btc_h:
                    exp_vb = sym_h[hor] - btc_h[hor]
                    if abs(rep_vb - exp_vb) > 0.15:
                        violations.append(
                            InvariantViolation(
                                "PROMPT_VS_BTC_MATH",
                                f"{sym}:{hor}",
                                f"reported vs_btc {hor}={rep_vb:+.2f}% != sym({sym_h[hor]:+.2f}%)-btc({btc_h[hor]:+.2f}%)={exp_vb:+.2f}%",
                            )
                        )

    for sym, rest in sec3b_items:
        if (m_sp := _OB_SPREAD_BBO_RE.search(rest)) is not None:
            rep_sp = safe_float(m_sp.group(1), default=None)
            bb = safe_pos_float(m_sp.group(2), default=None)
            ba = safe_pos_float(m_sp.group(3), default=None)
            if bb is not None and ba is not None:
                if bb > ba * 1.00001:
                    violations.append(
                        InvariantViolation(
                            "PROMPT_OB_SPREAD_AND_WALL_DIST",
                            f"{sym}:bbo",
                            f"crossed book bid={bb:g} > ask={ba:g}",
                        )
                    )
                if rep_sp is not None:
                    exp_sp = ((ba - bb) / ((ba + bb) / 2.0)) * 10000.0
                    if abs(exp_sp - rep_sp) > max(1.0, abs(exp_sp) * 0.05):
                        violations.append(
                            InvariantViolation(
                                "PROMPT_OB_SPREAD_AND_WALL_DIST",
                                f"{sym}:spread",
                                f"reported spread={rep_sp:.2f}bps != expected {exp_sp:.2f}bps (bid={bb:g}, ask={ba:g})",
                            )
                        )
        if (m_mid := _OB_MID_RE.search(rest)) is not None and (ob_mid := safe_pos_float(m_mid.group(1), default=None)) is not None:
            for m_wd in _WALL_DIST_RE.finditer(rest):
                w_lbl = m_wd.group(1)
                w_px = safe_pos_float(m_wd.group(2), default=None)
                rep_wd = safe_float(m_wd.group(3), default=None)
                if w_px is not None and rep_wd is not None:
                    exp_wd = ((w_px - ob_mid) / ob_mid) * 100.0
                    if abs(exp_wd - rep_wd) > 0.25:
                        violations.append(
                            InvariantViolation(
                                "PROMPT_OB_SPREAD_AND_WALL_DIST",
                                f"{sym}:{w_lbl}_dist",
                                f"reported {w_lbl} dist={rep_wd:+.2f}% != expected {exp_wd:+.2f}% (wall={w_px:g}, mid={ob_mid:g})",
                            )
                        )

    for ven_o, sym_o, px_o, rep_dist in open_order_dist_rows:
        spot_p = sec3_px_by_sym.get(sym_o.upper()) or px_by_vs.get((ven_o.lower(), sym_o.upper()))
        if spot_p is not None and spot_p > 0.0:
            exp_d = ((px_o - spot_p) / spot_p) * 100.0
            if abs(exp_d - rep_dist) > 0.20:
                violations.append(
                    InvariantViolation(
                        "PROMPT_OPEN_ORDER_DIST_MATH",
                        f"{ven_o}:{sym_o}",
                        f"reported Dist={rep_dist:+.2f}% != expected {exp_d:+.2f}% (limit={px_o:g}, spot={spot_p:g})",
                    )
                )

    for m_xs in _XSTOCK_REF_RE.finditer(prompt_text):
        x_px = (
            safe_pos_float(m_xs.group(1), default=None)
            if m_xs.group(1) is not None
            else (sec3_px_by_sym.get("SPCXXUSD") or px_by_vs.get(("kraken", "SPCXXUSD")))
        )
        u_px = safe_pos_float(m_xs.group(2), default=None)
        rep_prem = safe_float(m_xs.group(3), default=None)
        if x_px is not None and u_px is not None and rep_prem is not None:
            exp_prem = ((x_px - u_px) / u_px) * 100.0
            if abs(exp_prem - rep_prem) > 0.15:
                violations.append(
                    InvariantViolation(
                        "PROMPT_XSTOCK_PREMIUM_MATH",
                        "SPCXxUSD",
                        f"reported xstock_vs_underlying={rep_prem:+.2f}% != expected {exp_prem:+.2f}% (x={x_px:g}, u={u_px:g})",
                    )
                )


def _audit_prompt_numeric_hygiene(
    violations: list[InvariantViolation],
    prompt_text: str,
) -> None:
    """Detect leaked Python None, NaN/Inf, double signs, negative zeros, scientific notation, or double units."""
    body = prompt_text.split("LIVE TELEMETRY SNAPSHOT:", 1)[1] if "LIVE TELEMETRY SNAPSHOT:" in prompt_text else prompt_text
    in_news = False
    for raw_ln in body.splitlines():
        s_ln = raw_ln.strip()
        if s_ln.startswith("- Live Breaking News Feed"):
            in_news = True
            continue
        if in_news:
            if s_ln.startswith("* "):
                continue
            in_news = False
        if s_ln.startswith("```") or s_ln.startswith('"'):
            continue
        for kind, pat in _NUMERIC_HYGIENE_SPECS:
            if (m_h := pat.search(raw_ln)) is not None:
                violations.append(
                    InvariantViolation(
                        "PROMPT_NUMERIC_HYGIENE",
                        kind,
                        f"malformed numeric token {m_h.group(0).strip()!r} in: {s_ln[:120]}",
                    )
                )
                break


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit AI executor telemetry snapshot or rendered prompt invariants.")
    parser.add_argument("--workspace-dir", default=".", help="Workspace directory containing instruments.conf")
    parser.add_argument("--cache-dir", default="cachedb", help="Cache directory")
    parser.add_argument("--prompt-file", default="", help="Path to a rendered prompt text file to audit")
    parser.add_argument("--live-snapshot", action="store_true", help="Collect an offline snapshot from cachedb and audit it")
    parser.add_argument("--check-rendered-prompt", action="store_true", help="Also render full and compact prompts from --live-snapshot and audit them")
    parser.add_argument("--check-scorecard-stability", action="store_true", help="Verify scorecard LIFO stability on durable intents")
    parser.add_argument("--scope", default="crypto", help="Asset scope for live snapshot / scorecard audit")
    args = parser.parse_args(argv)

    non_trad = disabled_registry_symbols(args.workspace_dir) | observe_only_registry_symbols(args.workspace_dir)
    try:
        conf = load_key_value_conf(args.workspace_dir, str(_ROOT))
        _, gw_min_usd, gw_max_usd, gw_min_conf, _ = parse_intent_gateway_config(conf)
    except Exception:
        gw_min_usd, gw_max_usd, gw_min_conf = 10.0, 2000.0, 0.85
    violations: list[InvariantViolation] = []

    if args.prompt_file:
        text = read_text_file(args.prompt_file)
        violations.extend(
            audit_rendered_prompt_text(text, non_tradable_symbols=non_trad, gateway_min_usd=gw_min_usd)
        )

    if args.live_snapshot:
        from intelligence.sentiment.collectors.macro_telemetry_collector import MarketAndPortfolioSnapshotOrchestrator

        snap = MarketAndPortfolioSnapshotOrchestrator().collect(
            args.cache_dir,
            "*",
            workspace_dir=args.workspace_dir,
            scope=args.scope,
            allow_network=False,
            persist=False,
        )
        violations.extend(
            audit_snapshot_invariants(
                snap, non_tradable_symbols=non_trad, gateway_min_usd=gw_min_usd
            )
        )
        if args.check_rendered_prompt:
            from orchestratorOS.supervisor.prompt_formatters import build_supervisor_prompt

            scope_l = str(args.scope or "crypto").strip().lower()
            tkey = "auto_sharetrade_executor" if scope_l in ("equities", "equity") else "auto_cryptotrade_executor"
            spec = topic_spec(tkey)
            for compact_flag in (False, True):
                rendered = build_supervisor_prompt(
                    snap,
                    logs=None,
                    spec=spec,
                    min_usd=gw_min_usd,
                    max_usd=gw_max_usd,
                    min_conf=gw_min_conf,
                    compact_rules=compact_flag,
                )
                violations.extend(
                    audit_rendered_prompt_text(
                        rendered, non_tradable_symbols=non_trad, gateway_min_usd=gw_min_usd
                    )
                )

    if args.check_scorecard_stability:
        from providers.intent_state_store import load_durable_intents_from_cachedb

        intents = load_durable_intents_from_cachedb(args.cache_dir)
        now_s = resolve_now_ts(None)
        violations.extend(
            audit_scorecard_temporal_invariants(
                intents,
                t_early=now_s - 3600.0,
                t_late=now_s,
                scope=args.scope,
            )
        )

    if violations:
        for v in violations:
            print(f"FAIL {v.format_line()}", file=sys.stderr)
        print(f"Audit failed with {len(violations)} invariant violation(s).", file=sys.stderr)
        return 1

    print("OK: 0 prompt/telemetry invariant violations detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
