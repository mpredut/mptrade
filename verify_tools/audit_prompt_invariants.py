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
import re
import sys
from typing import Any, Callable, Collection, Mapping, Optional, Sequence

from botcore import (
    clean_scalar_str,
    clamp_float,
    resolve_now_ts,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
)
from cachepaths import matches_asset_scope, normalize_venue_name
from instrument_registry import (
    casefold_get,
    disabled_registry_symbols,
    observe_only_registry_symbols,
)
from providers.ai_survival_guard import (
    _collect_canceled_order_ids,
    _extract_intent_exec_price,
    _extract_intent_exec_ts,
    _is_filled_ai_intent,
    build_normalized_filled_ai_intent_row,
    match_filled_intent_rows_lifo_in_window,
)
from providers.intent_state import extract_intent_side
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
_SEC3B_SYM_RE = re.compile(r"^\s*[\*-]\s*([A-Za-z0-9_]+)(?:\s*\[[^\]]*\])?:(.*)$", re.MULTILINE)
_RANGE_RE = re.compile(
    r"(24h|7d)_range=\[([0-9.]+)(?:\.\.|,\s*)([0-9.]+)\]\s*(?:\(pos=|pos_(?:24h|7d)=)([0-9.]+)%"
)
_ARB_VS_RE = re.compile(r"arb_vs_([A-Za-z0-9_]+)=([+-]?[0-9.]+)bps")


@dataclass(frozen=True)
class InvariantViolation:
    code: str
    subject: str
    detail: str

    def format_line(self) -> str:
        return f"[{self.code}] {self.subject}: {self.detail}"


def _parse_money(raw: str) -> Optional[float]:
    return safe_float(str(raw or "").replace(",", ""), default=None)


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

    for sym, minfo in micro.items():
        if not isinstance(minfo, Mapping):
            continue
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

    return violations


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
) -> list[InvariantViolation]:
    """Audit mathematical and structural invariants directly on a rendered LLM prompt string."""
    violations: list[InvariantViolation] = []
    non_trad_u = {str(s).strip().upper() for s in non_tradable_symbols if str(s).strip()}

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

    # 2 & 3. Section 3C live tradable instruments: ghost check + free_quote_breakdown sum
    for m_inst in _SEC3C_INST_RE.finditer(prompt_text):
        ven, sym, rest = m_inst.group(1), m_inst.group(2), m_inst.group(3)
        if "free_" not in rest or "limits(" not in rest:
            continue
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

    # 4. Section 3 range pos math
    for m_s3 in _SEC3_SYM_RE.finditer(prompt_text):
        sym = m_s3.group(1)
        px = _parse_money(m_s3.group(2))
        rest = m_s3.group(3)
        if px is None or px <= 0.0:
            continue
        for m_rng in _RANGE_RE.finditer(rest):
            lbl = m_rng.group(1)
            lo = safe_pos_float(m_rng.group(2), default=None)
            hi = safe_pos_float(m_rng.group(3), default=None)
            pos = safe_float(m_rng.group(4), default=None)
            _check_range_pos(violations, "PROMPT_RANGE_POS_MATH", sym, lbl, px, lo, hi, pos, range_tol_pct)

    # 5. Section 3B cross-venue arb symmetry
    arb_pairs: dict[tuple[str, str], float] = {}
    for m_3b in _SEC3B_SYM_RE.finditer(prompt_text):
        sym, rest = m_3b.group(1), m_3b.group(2)
        if (m_arb := _ARB_VS_RE.search(rest)) is not None:
            peer = m_arb.group(1)
            val = safe_float(m_arb.group(2), default=None)
            if val is not None:
                arb_pairs[(sym.upper(), peer.upper())] = val
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

    return violations


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit AI executor telemetry snapshot or rendered prompt invariants.")
    parser.add_argument("--workspace-dir", default=".", help="Workspace directory containing instruments.conf")
    parser.add_argument("--cache-dir", default="cachedb", help="Cache directory")
    parser.add_argument("--prompt-file", default="", help="Path to a rendered prompt text file to audit")
    parser.add_argument("--live-snapshot", action="store_true", help="Collect an offline snapshot from cachedb and audit it")
    parser.add_argument("--check-scorecard-stability", action="store_true", help="Verify scorecard LIFO stability on durable intents")
    parser.add_argument("--scope", default="crypto", help="Asset scope for live snapshot / scorecard audit")
    args = parser.parse_args(argv)

    non_trad = disabled_registry_symbols(args.workspace_dir) | observe_only_registry_symbols(args.workspace_dir)
    violations: list[InvariantViolation] = []

    if args.prompt_file:
        text = read_text_file(args.prompt_file)
        violations.extend(audit_rendered_prompt_text(text, non_tradable_symbols=non_trad))

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
        violations.extend(audit_snapshot_invariants(snap, non_tradable_symbols=non_trad))

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
