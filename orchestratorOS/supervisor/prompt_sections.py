"""Pure formatting helpers for Autonomous AI Reconciler Gemini prompts.

Separates prompt presentation (Sections 1, 2, 3A, 3B, 3C) and symbol price
resolution from telemetry harvesting and order execution in
`autonomous_ai_reconciler.py`.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from botcore import safe_float as _parse_finite_float, safe_pos_float
from cachepaths import canonical_base_asset, compatible_quote_suffixes
from intelligence.macro_sentiment_ssot import resolve_macro_economic_calendar
from intelligence.sentiment.collectors.macro_telemetry_collector import (
    format_bottom_structure_6h_parts,
    format_macro_benchmarks_summary_line,
)
from intelligence.sentiment.collectors.microstructure_collector import (
    _build_vp_position_dict,
    _compute_horizon_range_dict,
    compute_liquidation_clusters,
    enrich_microstructure_cvd,
    round_opt_px as _round_opt_px,
)
from intelligence.sentiment.collectors.portfolio_collector import (
    resolve_symbol_price as _resolve_symbol_price,
    resolve_venue_fee_and_limits,
)
from orchestratorOS.agy.topics import topic_spec
from orchestratorOS.lib.technical_repair_ssot import (
    ALL_DIAGNOSED_CATEGORIES,
    ALL_REPAIR_ACTIONS,
    AUDITED_CONFIG_FILES,
    DIAGNOSED_SEVERITIES,
)
from providers.base import format_actor_tag, normalize_timestamp_ms


def _age_sfx(d: Dict[str, Any], parens: bool = False) -> str:
    if d.get("age_min") is None:
        return ""
    return f" (age={d['age_min']}m)" if parens else f", age={d['age_min']}m"


def _format_quote_breakdown(bd: Any, primary_asset: Any) -> str:
    if isinstance(bd, dict) and (len(bd) > 1 or any(k != primary_asset for k in bd)):
        return f" ({', '.join(f'{bk}=${_parse_finite_float(bv, 0.0):,.2f}' for bk, bv in bd.items())})"
    return ""


def _format_venue_cycle_line(venue: str, sym: str, cyc: Dict[str, Any]) -> str:
    """Format a normalized venue cycle anchor entry for Section 3C of the AI prompt."""
    extra_parts: List[str] = []
    if cyc.get("pos_peak") is not None and not _same_anchor_px(cyc.get("pos_peak"), cyc.get("avg_cost_px")):
        pk_sfx = f" (dd_from_peak={cyc.get('dd_from_peak_pct')}%)" if cyc.get("dd_from_peak_pct") is not None else ""
        extra_parts.append(f"pos_peak={cyc.get('pos_peak')}{pk_sfx}")
    if cyc.get("realized_pnl_usd") is not None:
        extra_parts.append(f"realized_pnl=${cyc.get('realized_pnl_usd')}")
    for lbl, k in (("tp_sold", "tp_sold_levels"), ("resting_orders", "resting_orders")):
        if cyc.get(k):
            extra_parts.append(f"{lbl}={cyc.get(k)}")
    extra_sfx = f", {', '.join(extra_parts)}" if extra_parts else ""
    exit_kind = cyc.get("last_exit_kind")
    exit_kind_sfx = f" ({exit_kind})" if exit_kind else ""
    dd_val = cyc.get("pnl_vs_avg_pct") if cyc.get("pnl_vs_avg_pct") is not None else cyc.get("dd_pct")
    dd_f = _parse_finite_float(dd_val, None)
    dd_field = (
        f"pnl_vs_avg={dd_f:+.2f}%"
        if dd_f is not None and dd_f > 0.0
        else f"dd={f'{dd_val}%' if dd_val is not None else 'N/A'}"
    )
    return (
        f"    - [{venue}] {sym}: qty={cyc.get('qty')}, cost=${cyc.get('cost_usd')}, "
        f"avg_cost={cyc.get('avg_cost_px')}, entry_px={cyc.get('entry_price')}, "
        f"last_buy={cyc.get('last_buy_price')}, last_sell={cyc.get('last_sell_price')}"
        f"{exit_kind_sfx}, dca_buys={cyc.get('dca_buys')}, {dd_field}{extra_sfx}"
    )


def _format_horizon_range(pinfo: Dict[str, Any], px_f: float, horizon: str) -> str:
    """Format `[low, high] pos_<horizon>=X% (dd_high=..., bounce_low=..., span=...)` for 24h or 7d."""
    lo, hi = pinfo.get(f"low_{horizon}"), pinfo.get(f"high_{horizon}")
    lo_f, hi_f = _parse_finite_float(lo, 0.0), _parse_finite_float(hi, 0.0)
    pos_sfx = ""
    if px_f > 0 and lo_f > 0 and hi_f > lo_f:
        fb_rng = _compute_horizon_range_dict(px_f, lo_f, hi_f, horizon)
        pos_v, dd_v, bnc_v, spn_v = (
            pinfo[k] if pinfo.get(k) is not None else fb_rng.get(k)
            for k in (f"pos_{horizon}_pct", f"dd_{horizon}_high_pct", f"bounce_{horizon}_low_pct", f"span_{horizon}_pct")
        )
        pos_sfx = (
            f" pos_{horizon}={pos_v}% (dd_high={_parse_finite_float(dd_v, 0.0):+.2f}%, "
            f"bounce_low={_parse_finite_float(bnc_v, 0.0):+.2f}%, span={spn_v}%)"
        )
    return f"{horizon}_range=[{lo}, {hi}]{pos_sfx}"


def _extract_macro_benchmarks_line(mb_raw: Any) -> str:
    """Extract or synthesize the 1-line SPY / QQQ / ES=F / NQ=F / DXY / US10Y benchmark summary."""
    if not isinstance(mb_raw, dict) or not mb_raw:
        return ""
    summary_s = str(mb_raw.get("summary_line") or "").strip() or format_macro_benchmarks_summary_line(mb_raw)
    return f"{summary_s}{_age_sfx(mb_raw, parens=True)}" if summary_s else ""


_AI_THREAT_LEVEL_MAP = {
    "CRITICAL_SHOCK": "HIGH_VOLATILITY_SHOCK",
    "CRITICAL": "HIGH_VOLATILITY",
}
_AI_BRAKE_MAP = {
    "HARD_VETO_NEW_BUYS": "SELECTIVE_DIP_SCALE_IN_ONLY",
    "BLOCK_BUYS": "SELECTIVE_DIP_SCALE_IN_ONLY",
    "HALT_BUYS": "SELECTIVE_DIP_SCALE_IN_ONLY",
    "VETO_BUYS": "SELECTIVE_DIP_SCALE_IN_ONLY",
}


def _fmt_liq_usd(val: Any) -> str:
    vf = _parse_finite_float(val, 0.0)
    if vf >= 1_000_000.0:
        return f"${vf / 1_000_000.0:.2f}M"
    if vf >= 1_000.0:
        return f"${vf / 1_000.0:.1f}K"
    return f"${vf:.0f}"


def _fmt_px(px_f: float, raw_px: Any = None) -> str:
    if px_f >= 1.0:
        if px_f < 1000.0 and abs(px_f - round(px_f, 2)) >= 5e-4:
            return f"${px_f:,.4f}".rstrip("0").rstrip(".")
        return f"${px_f:,.2f}"
    if px_f > 0:
        return f"${px_f:.6g}"
    if raw_px is not None and not isinstance(raw_px, bool):
        return f"${raw_px}"
    return "N/A"


def _fmt_slope_val(val: Any, decimals: int = 5, *, signed: bool = True) -> str:
    vf = _parse_finite_float(val, 0.0)
    if abs(vf) < 5e-8:
        vf = 0.0
    elif 0.0 < abs(vf) < 1e-4:
        return f"{vf:+.2e}" if signed else f"{vf:.2e}"
    return f"{vf:+.{decimals}f}" if signed else f"{vf:.{decimals}f}"


def _format_macro_and_news_section(sensors: Any) -> List[str]:
    """Format Section 1 (Macro, Sentiment, Geopolitical & Live Breaking News) for the AI prompt."""
    macro = getattr(sensors, "macro_and_sentiment", None) or {}
    fg = getattr(sensors, "fear_greed", None) or macro.get("fear_greed") or {}
    mb = getattr(sensors, "market_breadth", None) or macro.get("market_breadth") or {}
    sa = getattr(sensors, "sentiment_advisor", None) or macro.get("sentiment_advisor") or {}
    gt = getattr(sensors, "geopolitical_threat", None) or macro.get("geopolitical") or {}
    ps = getattr(sensors, "portfolio_summary", None) or {}
    mb_bench = (
        getattr(sensors, "macro_benchmarks", None)
        or macro.get("macro_benchmarks")
        or (ps.get("macro_benchmarks") if isinstance(ps, dict) else None)
        or {}
    )

    fg_hist = fg.get("historical_values")[:7] if isinstance(fg.get("historical_values"), list) else None
    fg_line = (
        f"{fg.get('value', 'N/A')} ({fg.get('classification') or fg.get('sentiment') or 'N/A'}, "
        f"7d_chg={fg.get('trend_7d_change', 'N/A')}"
        + (f", 7d_hist={fg_hist}" if fg_hist else "")
        + f"{_age_sfx(fg)})"
        if fg else "None"
    )
    mb_gainers = mb.get("top_gainers")[:3] if isinstance(mb.get("top_gainers"), list) else None
    mb_losers = mb.get("top_losers")[:3] if isinstance(mb.get("top_losers"), list) else None
    mb_rot_parts: List[str] = []
    if mb:
        if mb.get("eth_btc_ratio") is not None:
            eb_c = f" ({_parse_finite_float(mb.get('eth_btc_chg_24h_pct'), 0.0):+.2f}% 24h)" if mb.get("eth_btc_chg_24h_pct") is not None else ""
            mb_rot_parts.append(f"ETH/BTC={mb.get('eth_btc_ratio')}{eb_c}")
        if mb.get("eth_price") is not None:
            eth_c = f" ({_parse_finite_float(mb.get('eth_chg_24h_pct'), 0.0):+.2f}% 24h)" if mb.get("eth_chg_24h_pct") is not None else ""
            mb_rot_parts.append(f"ETH=${mb.get('eth_price')}{eth_c}")
        if mb.get("btc_vol_share_pct") is not None:
            mb_rot_parts.append(f"BTCDOM={mb.get('btc_vol_share_pct')}%")
        if mb.get("alt_vol_share_pct") is not None:
            mb_rot_parts.append(f"TOTAL3_share={mb.get('alt_vol_share_pct')}%")
        if mb.get("rotation_regime"):
            mb_rot_parts.append(f"rotation={mb.get('rotation_regime')}")
    mb_line = (
        f"advance_ratio={mb.get('advance_ratio', 'N/A')}, regime={mb.get('regime', 'N/A')}, "
        f"median_chg={mb.get('median_change_pct', 'N/A')}%, "
        f"mean_chg={mb.get('mean_change_pct', 'N/A')}%, disp_std={mb.get('dispersion_std', 'N/A')}, "
        f"advancing={mb.get('advancing', 'N/A')}/{mb.get('total_symbols', 'N/A')}"
        + (f", {', '.join(mb_rot_parts)}" if mb_rot_parts else "")
        + (f", top_gainers={mb_gainers}" if mb_gainers else "")
        + (f", top_losers={mb_losers}" if mb_losers else "")
        + (f", age={mb.get('age_min')}m" if mb.get("age_min") is not None else "")
        if mb else "None"
    )
    sa_line = (
        f"bias={sa.get('market_bias', 'N/A')}, risk={sa.get('risk_level', 'N/A')}, "
        f"action={sa.get('recommended_action', 'N/A')}, conf={sa.get('confidence', 'N/A')}"
        f"{_age_sfx(sa)}"
        + (f" | live_flow_override={sa.get('live_flow_divergence')}" if sa.get("live_flow_divergence") else "")
        + f" | summary={sa.get('summary', '')} | key_risks={sa.get('key_risks', [])}"
        if sa else "None"
    )
    raw_gt_level = str(gt.get("threat_level", "N/A"))
    raw_gt_brake = str(gt.get("recommended_brake", "N/A"))
    ai_gt_level = _AI_THREAT_LEVEL_MAP.get(raw_gt_level.strip().upper(), raw_gt_level)
    ai_gt_brake = _AI_BRAKE_MAP.get(raw_gt_brake.strip().upper(), raw_gt_brake)
    gt_line = (
        f"level={ai_gt_level}, risk_score={gt.get('risk_score', 'N/A')}, "
        f"brake={ai_gt_brake}{_age_sfx(gt)} | summary={gt.get('summary', '')}"
        if gt else "None"
    )
    lines = [
        "### 1. ACTIVE MACRO, SENTIMENT, GEOPOLITICAL & LIVE BREAKING NEWS TELEMETRY:",
        f"- Fear & Greed: {fg_line}",
        f"- Market Breadth: {mb_line}",
        f"- Macro Sentiment Advisor: {sa_line}",
        f"- Geopolitical Threat Shield: {gt_line}",
    ]
    breaking_news = getattr(sensors, "breaking_news", None) or macro.get("recent_news") or []
    mc = (
        getattr(sensors, "macro_calendar", None)
        or macro.get("macro_calendar")
        or resolve_macro_economic_calendar(breaking_news=breaking_news)
    )
    if isinstance(mc, dict) and mc.get("next_high_impact_event"):
        mc_parts = [
            f'session="{mc.get("session_label")}"',
            f'next_high_impact_event="{mc.get("next_high_impact_event")}"',
        ]
        if mc.get("active_release_window") and mc.get("next_scheduled_event"):
            mc_parts.append(f'next_scheduled="{mc.get("next_scheduled_event")}"')
        if mc.get("fomc_today") is not None:
            mc_parts.append(f'fomc_today={mc.get("fomc_today")}')
        if mc.get("next_fomc_date"):
            mc_parts.append(f'next_fomc_date="{mc.get("next_fomc_date")}"')
        if mc.get("upcoming_windows"):
            mc_parts.append(f'upcoming={mc.get("upcoming_windows")}')
        if mc.get("detected_macro_headlines"):
            mc_parts.append(f'live_macro_headlines={mc.get("detected_macro_headlines")}')
        if mc.get("live_macro_prints"):
            mc_parts.append(f'live_macro_prints={mc.get("live_macro_prints")}')
        lines.append(f"- Macro Economic Calendar & Session Window: {' | '.join(mc_parts)}")
    mb_bench_line = _extract_macro_benchmarks_line(mb_bench)
    if mb_bench_line:
        lines.append(
            f"- US Equities, Futures & Macro Rates Benchmarks (SPY / QQQ / ES=F / NQ=F / DXY / US10Y 24h Delta): {mb_bench_line}"
        )
    if breaking_news:
        lines.append("- Live Breaking News Feed (cachedb/news_feed_collect.json, 1m refresh):")
        for hn in breaking_news:
            sev_tag = "[HIGH-SEV] " if hn.get("is_high_severity") else ""
            kw_sfx = f" kw={hn.get('keywords')}" if hn.get("keywords") else ""
            lines.append(f"  * {sev_tag}{hn.get('title')}{kw_sfx}{_age_sfx(hn, parens=True)}")
    return lines


def _format_sre_and_fleet_section(
    sensors: Any,
    logs: Any,
    lookback_hours: float,
) -> List[str]:
    """Format SRE functional telemetry, log anomalies, fleet heartbeats, and OS/VPN resources."""
    lines = [
        "### 1. FUNCTIONAL SYSTEM TELEMETRY, HEARTBEATS & WATCHDOG STATES:",
        f"- Active Heartbeats: {list(sensors.active_heartbeats.keys())}",
        f"- Watchdog & Runtime State Files: {sensors.state_files}",
        f"\n### 2. LOG ANOMALIES & GUARD VETOES / PERSISTENT UNRESOLVED ERRORS (PAST {lookback_hours:g}H):",
        f"- Scanned Logs: {logs.total_scanned_files}",
        f"- Error Count: {logs.error_count}",
        f"- Guard Interventions: {logs.warning_count}",
    ]
    if getattr(logs, "persistent_errors", None):
        lines.append("- Persistent / Unresolved Errors in Logs:")
        for pe in logs.persistent_errors[:15]:
            tail_lbl = "UNRESOLVED_AT_TAIL" if pe.get("unresolved_at_tail") else "RECURRING"
            lines.append(
                f"  * [{tail_lbl} x{pe.get('count', 1)} in {pe.get('file', 'log')}] {pe.get('sample', '')}"
            )
    lines.append("- Sample Errors & Interventions:")
    for evt in logs.critical_events[:15]:
        lines.append(f"  * {evt}")
    for g_evt in logs.guard_interventions[:10]:
        lines.append(f"  * [GUARD] {g_evt}")

    if sensors.fleet_processes:
        running_p = [p for p in sensors.fleet_processes if p.get("is_running")]
        stopped_p = [p["name"] for p in sensors.fleet_processes if not p.get("is_running")]
        stale_p = [
            f"{p['name']} (age={p.get('age_seconds')}s > {p.get('max_stale_seconds')}s)"
            for p in sensors.fleet_processes
            if p.get("is_running") and not p.get("heartbeat_fresh", True)
        ]
        lines.append(
            f"\n### 2B. FLEET PROCESSES & BOT HEARTBEATS (procs.conf): {len(running_p)}/{len(sensors.fleet_processes)} running"
            + (f" | STOPPED: {stopped_p}" if stopped_p else "")
            + (f" | STALE HEARTBEAT: {stale_p}" if stale_p else " | All heartbeats fresh")
        )

    if sensors.system_and_vpn:
        sv = sensors.system_and_vpn
        cfg_audit = sv.get("config_audit") if isinstance(sv.get("config_audit"), dict) else {}
        cfg_issues = cfg_audit.get("issues") or []
        cfg_summary = (
            f"ISSUES DETECTED: {cfg_issues}"
            if cfg_issues
            else f"OK (verified {cfg_audit.get('checked_files', [])})"
        )
        lines.append(
            "\n### 2C. LINUX SYSTEM RESOURCES & VPN TUNNEL HEALTH:\n"
            f"- CPU Load Avg (1m/5m/15m): {sv.get('load_avg', 'N/A')}\n"
            f"- Memory: used={sv.get('mem_used_pct', 'N/A')}%, avail={sv.get('mem_avail_mb', 'N/A')} MB\n"
            f"- Disk: used={sv.get('disk_used_pct', 'N/A')}%, free={sv.get('disk_free_gb', 'N/A')} GB\n"
            f"- VPN Tunnel ({sv.get('vpn_interface', 'wgpia0')}): {'UP' if sv.get('vpn_tunnel_up') else 'DOWN / NOT PRESENT'}\n"
            f"- Internet & DNS Connectivity: healthy={sv.get('internet_healthy', True)}, "
            f"default_route={sv.get('default_route_present', 'N/A')}, dns_servers={sv.get('dns_servers', 'N/A')}, "
            f"vpn_outage_active={sv.get('vpn_outage_active', False)}\n"
            f"- Configuration Files Audit: {cfg_summary}"
            + (
                f"\n- Sustained High-Resource Processes (.resource_watchdog_state.json): {sv['resource_watchdog_offenders']}"
                if sv.get("resource_watchdog_offenders")
                else ""
            )
        )
    return lines


def _format_market_prices_section(prices_and_trends: Dict[str, Any]) -> List[str]:
    """Format Section 3A (Live Market Prices, Multi-Horizon Trends & 24h/7d Evolution)."""
    lines = ["\n### 3. LIVE MARKET PRICES, MULTI-HORIZON TRENDS & 24H EVOLUTION (ALL COINS):"]
    btc_candidates = [
        (k, entry) for k, entry in prices_and_trends.items()
        if isinstance(entry, dict) and canonical_base_asset(k) == "BTC"
    ]
    btc_entry = next(
        (e for k, e in btc_candidates if "USD" in compatible_quote_suffixes(k)),
        btc_candidates[0][1] if btc_candidates else None,
    )
    grouped_3a: Dict[Tuple[str, Tuple[str, ...]], List[Tuple[str, Dict[str, Any]]]] = {}
    for s_k, s_v in sorted(prices_and_trends.items()):
        if isinstance(s_v, dict):
            grouped_3a.setdefault(
                (canonical_base_asset(s_k), compatible_quote_suffixes(s_k)), []
            ).append((s_k, s_v))

    for _, sym_group in sorted(grouped_3a.items(), key=lambda kv: kv[1][0][0]):
        sym_group.sort(
            key=lambda item: (
                0 if item[0].endswith(("USDC", "_US_EQ", "_EQ")) else 1,
                -len(item[1]),
                item[0],
            )
        )
        sym, raw_pinfo = sym_group[0]
        pinfo = dict(raw_pinfo)
        px = pinfo.get("price")
        px_f = _parse_finite_float(px, 0.0)
        px_str = _fmt_px(px_f, px)
        sib_lbl = ""
        if len(sym_group) > 1:
            sib_parts: List[str] = []
            for sib_sym, sib_info in sym_group[1:]:
                s_px = _parse_finite_float(sib_info.get("price"), 0.0)
                if px_f <= 0 or s_px <= 0 or abs(px_f - s_px) / max(px_f, s_px) <= 0.10:
                    for sk, sv in sib_info.items():
                        if pinfo.get(sk) is None and sv is not None:
                            pinfo[sk] = sv
                if s_px > 0:
                    diff_s = ""
                    if px_f > 0 and abs(s_px - px_f) > 1e-9:
                        diff_pct = (s_px - px_f) / px_f * 100.0
                        diff_s = (
                            f" [{diff_pct * 100.0:+.1f}bps]"
                            if abs(diff_pct) < 0.005
                            else f" [{diff_pct:+.2f}%]"
                        )
                    sib_parts.append(f"{sib_sym}={_fmt_px(s_px)}{diff_s}")
                else:
                    sib_parts.append(sib_sym)
            if sib_parts:
                sib_lbl = f" (sibling venue: {', '.join(sib_parts)})"

        extras: List[str] = []
        if "chg_15m_pct" in pinfo:
            micro_horizons = "".join(
                f"{m_lbl}={_parse_finite_float(pinfo.get(m_k), 0.0):+.2f}% "
                for m_lbl, m_k in (("1m", "chg_1m_pct"), ("3m", "chg_3m_pct"), ("5m", "chg_5m_pct"))
                if m_k in pinfo and pinfo.get(m_k) is not None
            )
            opt_horizons = "".join(
                f"{h_lbl}={_parse_finite_float(pinfo.get(h_k), 0.0):+.2f}% "
                for h_lbl, h_k in (("6h", "chg_6h_pct"), ("12h", "chg_12h_pct"), ("24h", "chg_24h_pct"))
                if h_k in pinfo and pinfo.get(h_k) is not None
            )
            extras.append(
                f"{micro_horizons}"
                f"15m={_parse_finite_float(pinfo.get('chg_15m_pct'), 0.0):+.2f}% "
                f"30m={_parse_finite_float(pinfo.get('chg_30m_pct'), 0.0):+.2f}% "
                f"1h={_parse_finite_float(pinfo.get('chg_1h_pct'), 0.0):+.2f}% "
                f"3h={_parse_finite_float(pinfo.get('chg_3h_pct'), 0.0):+.2f}% "
                f"{opt_horizons}"
                f"{_format_horizon_range(pinfo, px_f, '24h')}"
            )
            if isinstance(pinfo.get("recent_tape_15m"), list) and pinfo["recent_tape_15m"]:
                tape_str = f"tape_15m=[{', '.join(str(x) for x in pinfo['recent_tape_15m'])}]"
                if pinfo.get("low_15m") is not None and pinfo.get("high_15m") is not None:
                    pos15_sfx = f" pos_15m={pinfo['pos_15m_pct']}%" if pinfo.get("pos_15m_pct") is not None else ""
                    tape_str += f" 15m_range=[{pinfo['low_15m']}, {pinfo['high_15m']}]{pos15_sfx}"
                extras.append(tape_str)
        if all(pinfo.get(k) is not None for k in ("poc_px", "val_px", "vah_px")):
            fb_vp = _build_vp_position_dict(
                px_f,
                _parse_finite_float(pinfo.get("poc_px"), 0.0),
                _parse_finite_float(pinfo.get("val_px"), 0.0),
                _parse_finite_float(pinfo.get("vah_px"), 0.0),
            )
            poc_d, val_d, vah_d = (
                _parse_finite_float(pinfo[k] if pinfo.get(k) is not None else fb_vp.get(k), 0.0)
                for k in ("poc_dist_pct", "val_dist_pct", "vah_dist_pct")
            )
            va_pos = pinfo["va_pos_pct"] if pinfo.get("va_pos_pct") is not None else fb_vp.get("va_pos_pct")
            va_reg = pinfo.get("va_regime") or fb_vp.get("va_regime") or "INSIDE_VA"
            sec_poc_val = pinfo.get("sec_poc_px") if pinfo.get("sec_poc_px") is not None else pinfo.get("hvn2_px")
            sec_poc_sfx = f", sec_POC={sec_poc_val}" if sec_poc_val is not None else ""
            lvn_sfx = f", LVN={pinfo.get('lvn_px')}" if pinfo.get("lvn_px") is not None else ""
            shape_sfx = f", shape={pinfo.get('profile_shape')}" if pinfo.get("profile_shape") == "BIMODAL" else ""
            vp_src_sfx = f", vp_source={pinfo['vp_source']}" if pinfo.get("vp_source") else ""
            extras.append(
                f"vol_profile(POC={pinfo.get('poc_px')}[{poc_d:+.2f}%]{sec_poc_sfx}{lvn_sfx}, "
                f"VAL={pinfo.get('val_px')}[{val_d:+.2f}%], "
                f"VAH={pinfo.get('vah_px')}[{vah_d:+.2f}%], "
                f"va_pos={va_pos}%, regime={va_reg}{shape_sfx}{vp_src_sfx})"
            )

        is_non_btc_crypto = not sym.endswith(("_EQ", "_US_EQ")) and canonical_base_asset(sym) != "BTC"
        if is_non_btc_crypto:
            vs_btc_parts: List[str] = []
            if btc_entry:
                for h_lbl, h_k in (
                    ("5m", "chg_5m_pct"), ("15m", "chg_15m_pct"), ("1h", "chg_1h_pct"), ("3h", "chg_3h_pct"),
                    ("6h", "chg_6h_pct"), ("12h", "chg_12h_pct"), ("24h", "chg_24h_pct"), ("7d", "chg_7d_pct"),
                ):
                    if h_k in pinfo and h_k in btc_entry:
                        a_v = _parse_finite_float(pinfo.get(h_k), None)
                        b_v = _parse_finite_float(btc_entry.get(h_k), None)
                        if a_v is not None and b_v is not None:
                            vs_btc_parts.append(f"{h_lbl}={a_v - b_v:+.2f}%")
            if vs_btc_parts:
                extras.append(f"vs_btc({', '.join(vs_btc_parts)})")
            cross_parts = (
                ([f"pos={pinfo['alt_btc_pos_24h_pct']}%"] if pinfo.get("alt_btc_pos_24h_pct") is not None else [])
                + [
                    f"{c_lbl}={_parse_finite_float(pinfo.get(c_k), 0.0):+.2f}%"
                    for c_lbl, c_k in (("dd_high", "alt_btc_dd_24h_high_pct"), ("bounce_low", "alt_btc_bounce_24h_low_pct"))
                    if pinfo.get(c_k) is not None
                ]
                + [
                    f"{sc_lbl}={_parse_finite_float(pinfo.get(sc_k), 0.0):.2f}"
                    for sc_lbl, sc_k in (("beta", "beta_btc_24h"), ("corr", "corr_btc_24h"))
                    if pinfo.get(sc_k) is not None
                ]
            )
            if cross_parts:
                extras.append(f"alt_btc_24h({', '.join(cross_parts)})")

        if pinfo.get("bottom_structure_6h"):
            b_struct, b_parts = format_bottom_structure_6h_parts(
                pinfo,
                btc_entry,
                prior_label="prior_low",
                px_prefix="",
                check_smt=bool(is_non_btc_crypto and btc_entry),
            )
            if b_struct:
                extras.append(f"bottom_6h={b_struct}({', '.join(b_parts)})")

        if "chg_7d_pct" in pinfo:
            extras.append(
                f"7d={_parse_finite_float(pinfo.get('chg_7d_pct'), 0.0):+.2f}% "
                f"{_format_horizon_range(pinfo, px_f, '7d')}"
            )
        if pinfo.get("swing_low_px") is not None and pinfo.get("swing_high_px") is not None:
            sw_ral = _parse_finite_float(pinfo.get("rally_from_swing_low_pct"), 0.0)
            sw_pb = _parse_finite_float(pinfo.get("pullback_from_swing_high_pct"), 0.0)
            maj_sfx = (
                f", major_low={pinfo.get('major_swing_low_px')}"
                f"(rally={_parse_finite_float(pinfo.get('rally_from_major_low_pct'), 0.0):+.2f}%)"
                if pinfo.get("rally_from_major_low_pct") is not None and pinfo.get("major_swing_low_px") is not None
                else ""
            )
            hist_sfx = ""
            if isinstance(pinfo.get("recent_swings_7d"), list) and pinfo["recent_swings_7d"]:
                amp_s = (
                    f", avg_amp={pinfo.get('avg_swing_amp_7d_pct')}%"
                    if pinfo.get("avg_swing_amp_7d_pct") is not None
                    else ""
                )
                hist_sfx = f" swings_7d=[{', '.join(str(x) for x in pinfo['recent_swings_7d'])}]{amp_s}"
            extras.append(
                f"swing_wave(low={pinfo.get('swing_low_px')}, high={pinfo.get('swing_high_px')}, "
                f"rally_low={sw_ral:+.2f}%, pullback_high={sw_pb:+.2f}%{maj_sfx}){hist_sfx}"
            )
        if "slope_15m" in pinfo:
            s90_v = _parse_finite_float(pinfo.get("slope_90s"), None)
            s90_pfx = f"90s={_fmt_slope_val(s90_v, 5)}, " if s90_v is not None else ""
            gr_v = _parse_finite_float(pinfo.get("gradient_recent"), None)
            grad_rec_sfx = f", grad_rec={_fmt_slope_val(gr_v, 4)}" if gr_v is not None and abs(gr_v) > 1e-9 else ""
            gf_v = _parse_finite_float(pinfo.get("gradient_recent_fast"), None)
            eps_v = _parse_finite_float(pinfo.get("epsilon"), None)
            fast_sfx = (
                f", grad_fast={_fmt_slope_val(gf_v, 4)}/eps={_fmt_slope_val(eps_v, 4, signed=False)}"
                if gf_v is not None and eps_v is not None
                else ""
            )
            g_parts = [
                f"{g_lbl}={_fmt_slope_val(gv, 5)}"
                for g_lbl, g_k in (("90s", "growth_90s"), ("3.7m", "growth_3_7m"), ("15m", "growth_15m"), ("2.5h", "growth_2_5h"))
                if (gv := _parse_finite_float(pinfo.get(g_k), None)) is not None
            ]
            g_sfx = f" g_coeff({', '.join(g_parts)})" if g_parts else ""
            extras.append(
                f"slopes({s90_pfx}3.7m={_fmt_slope_val(pinfo.get('slope_3_7m'), 5)}, "
                f"15m={_fmt_slope_val(pinfo.get('slope_15m'), 5)}, "
                f"2.5h={_fmt_slope_val(pinfo.get('slope_2_5h'), 5)}{grad_rec_sfx}{fast_sfx}){g_sfx}"
            )
        if pinfo.get("vol_1h_pct") is not None:
            vol_part = f"vol_1h={pinfo.get('vol_1h_pct')}%"
            if pinfo.get("adapt_reentry_pct") is not None:
                vol_part += f"(adapt_reentry={pinfo.get('adapt_reentry_pct')}%)"
            if pinfo.get("adapt_dca_pct") is not None:
                vol_part += f" adapt_dca={pinfo.get('adapt_dca_pct')}%"
            extras.append(vol_part)
        if (kv := _parse_finite_float(pinfo.get("kalman_vel"), None)) is not None:
            extras.append(f"kalman_vel={_fmt_slope_val(kv, 5)}")
        if pinfo.get("long_trend_dir"):
            lt_str = f"long_trend={pinfo.get('long_trend_dir')}({pinfo.get('long_trend_hours')}h"
            if pinfo.get("long_trend_est_rem_hours") is not None:
                lt_str += f", est_rem={pinfo.get('long_trend_est_rem_hours')}h"
            if pinfo.get("weibull_median_d") is not None:
                lt_str += f", med_d={pinfo.get('weibull_median_d')}d/p90={pinfo.get('weibull_p90_d')}d"
            lt_str += ")"
            extras.append(lt_str)
        elif pinfo.get("weibull_median_d") is not None or pinfo.get("long_trend_neutral"):
            if pinfo.get("weibull_median_d") is not None:
                extras.append(f"long_trend=NEUTRAL(med_d={pinfo.get('weibull_median_d')}d/p90={pinfo.get('weibull_p90_d')}d)")
            else:
                extras.append("long_trend=NEUTRAL")
        extra_str = f" | {' | '.join(extras)}" if extras else ""
        lines.append(f"  * {sym}{sib_lbl}: Price={px_str}{extra_str}")
    return lines


def _format_microstructure_section(
    microstructure: Dict[str, Any],
    prices_and_trends: Dict[str, Any],
) -> List[str]:
    """Format Section 3B (Orderbook Walls, Derivatives & Whale Flow)."""
    lines = ["\n### 3B. PILLAR 2 MICROSTRUCTURE, ORDERBOOK WALLS, DERIVATIVES & WHALE FLOW:"]
    seen_3b_groups: set[Tuple[str, Tuple[str, ...]]] = set()

    def _group_key_3b(s: str, m: Dict[str, Any]) -> Tuple[str, Tuple[str, ...]]:
        b_c = canonical_base_asset(s)
        q_s = compatible_quote_suffixes(s)
        has_px_anchors = any(
            m.get(k) is not None
            for k in ("ob_mid_price", "bid_wall_px", "ask_wall_px", "long_liq_px", "short_liq_px")
        )
        if "USD" in q_s and has_px_anchors and m.get("bid_depth_usd") is not None:
            has_distinct_usd_sib = any(
                other_s != s
                and isinstance(other_m, dict)
                and canonical_base_asset(other_s) == b_c
                and "USD" in compatible_quote_suffixes(other_s)
                and other_m.get("bid_depth_usd") is not None
                and abs(_parse_finite_float(other_m.get("bid_depth_usd"), 0.0) - _parse_finite_float(m.get("bid_depth_usd"), 0.0)) > 1.0
                for other_s, other_m in microstructure.items()
            )
            if has_distinct_usd_sib:
                return (b_c, (s,))
        if "USD" in q_s or not has_px_anchors:
            return (b_c, ("USDC", "USDT", "USD"))
        return (b_c, q_s)

    sorted_3b = sorted(
        microstructure.items(),
        key=lambda kv: (
            0 if kv[0].endswith("USDC") else (1 if "USD" in compatible_quote_suffixes(kv[0]) else 2),
            kv[0],
        ),
    )
    for sym, minfo in sorted_3b:
        if not isinstance(minfo, dict):
            continue
        b_canon_3b = canonical_base_asset(sym)
        grp_3b = _group_key_3b(sym, minfo)
        if grp_3b in seen_3b_groups:
            continue
        seen_3b_groups.add(grp_3b)
        sib_3b = [
            other_s
            for other_s, other_m in microstructure.items()
            if other_s != sym and isinstance(other_m, dict) and _group_key_3b(other_s, other_m) == grp_3b
        ]
        minfo = dict(minfo)
        sym_quotes_3b = compatible_quote_suffixes(sym)
        _px_keys_3b = {
            "ob_mid_price", "best_bid", "best_ask", "bid_wall_px", "ask_wall_px",
            "long_liq_px", "short_liq_px", "long_liq_deep_px", "short_liq_deep_px",
        }
        for other_s in sib_3b:
            other_m = microstructure.get(other_s)
            if not isinstance(other_m, dict):
                continue
            same_q_3b = compatible_quote_suffixes(other_s) == sym_quotes_3b
            for mk, mv in other_m.items():
                if minfo.get(mk) is None and mv is not None and (same_q_3b or mk not in _px_keys_3b):
                    minfo[mk] = mv
        sym_3b_lbl = f"{sym} ({', '.join(sorted(sib_3b))})" if sib_3b else sym
        cur_px = _resolve_symbol_price(sym, prices_and_trends)
        bw_px = minfo.get("bid_wall_px")
        aw_px = minfo.get("ask_wall_px")
        bw_px_f = _parse_finite_float(bw_px, 0.0)
        aw_px_f = _parse_finite_float(aw_px, 0.0)
        b_bid = _parse_finite_float(minfo.get("best_bid"), 0.0)
        b_ask = _parse_finite_float(minfo.get("best_ask"), 0.0)
        ob_mid = _parse_finite_float(minfo.get("ob_mid_price"), 0.0) or (
            (b_bid + b_ask) / 2.0 if b_bid > 0 and b_ask > 0 else 0.0
        )
        if ob_mid > 0 and (
            cur_px <= 0
            or (bw_px_f > 0 and cur_px < bw_px_f)
            or (aw_px_f > 0 and cur_px > aw_px_f)
        ):
            cur_px = ob_mid
        ob_ref_px = ob_mid if ob_mid > 0 else cur_px
        bw_dist = f", dist={(bw_px_f - ob_ref_px) / ob_ref_px * 100.0:+.2f}%" if ob_ref_px > 0 and bw_px_f > 0 else ""
        aw_dist = f", dist={(aw_px_f - ob_ref_px) / ob_ref_px * 100.0:+.2f}%" if ob_ref_px > 0 and aw_px_f > 0 else ""
        spread_sfx = ""
        if minfo.get("spread_bps") is not None:
            ba_lbl = f" (bid={b_bid}/ask={b_ask})" if b_bid > 0 and b_ask > 0 else ""
            ob_mid_lbl = f", ob_mid={round(ob_mid, 6)}" if ob_mid > 0 else ""
            spread_sfx = f", spread={minfo.get('spread_bps')}bps{ba_lbl}{ob_mid_lbl}"

        raw_p = (
            prices_and_trends.get(sym)
            if isinstance(prices_and_trends.get(sym), dict)
            else next(
                (
                    v
                    for k, v in prices_and_trends.items()
                    if isinstance(v, dict)
                    and canonical_base_asset(k) == b_canon_3b
                    and compatible_quote_suffixes(k) == compatible_quote_suffixes(sym)
                ),
                {},
            )
        )
        p_ctx = {**raw_p, "price": cur_px} if cur_px > 0.0 else (raw_p or None)
        if (
            minfo.get("depth_1pct_bid_usd") is None
            or minfo.get("est_slip_2k_bps") is None
            or minfo.get("cvd_15m_usd") is None
            or minfo.get("cvd_regime") is None
            or (minfo.get("cvd_15m_base") is None and cur_px > 0.0)
            or (isinstance(p_ctx, dict) and p_ctx.get("chg_15m_pct") is not None)
        ):
            enrich_microstructure_cvd(minfo, p_ctx)
        if any(
            minfo.get(k) is None
            for k in ("long_liq_px", "short_liq_px", "long_liq_deep_px", "short_liq_deep_px", "liq_magnet_bias")
        ):
            liq_lazy = compute_liquidation_clusters(p_ctx, minfo)
            for lk, lv in liq_lazy.items():
                if minfo.get(lk) is None:
                    minfo[lk] = lv
        d1_b = _parse_finite_float(minfo.get("depth_1pct_bid_usd"), 0.0)
        d1_a = _parse_finite_float(minfo.get("depth_1pct_ask_usd"), 0.0)
        depth_1pct_sfx = (
            f", depth_1pct=(bid={_fmt_liq_usd(d1_b)}/ask={_fmt_liq_usd(d1_a)}, slip_2k={minfo.get('est_slip_2k_bps')}bps)"
            if (d1_b > 0.0 or d1_a > 0.0)
            else ""
        )

        cvd_sfx = ""
        if (cvd_usd_raw := minfo.get("cvd_15m_usd")) is not None:
            cvd_u = _parse_finite_float(cvd_usd_raw, 0.0)
            cvd_reg = minfo.get("cvd_regime") or "balanced_flow"
            usd_s = f"{'+' if cvd_u >= 0.0 else '-'}{_fmt_liq_usd(abs(cvd_u))}"
            cvd_b = minfo.get("cvd_15m_base")
            cvd_sfx = (
                f", cvd_15m={_parse_finite_float(cvd_b, 0.0):+g} {b_canon_3b} ({usd_s}, {cvd_reg})"
                if cvd_b is not None
                else f", cvd_15m={usd_s} ({cvd_reg})"
            )
            mh_parts: List[str] = []
            for h_lbl, tr_k, cvd_k in (
                ("5m", "taker_ratio_5m", "cvd_5m_usd"),
                ("1h", "taker_ratio_1h", "cvd_1h_usd"),
                ("4h", "taker_ratio_4h", "cvd_4h_usd"),
            ):
                tr_v = _parse_finite_float(minfo.get(tr_k), None)
                c_v = _parse_finite_float(minfo.get(cvd_k), None)
                if tr_v is not None and c_v is not None:
                    c_s = f"{'+' if c_v >= 0.0 else '-'}{_fmt_liq_usd(abs(c_v))}"
                    mh_parts.append(f"{h_lbl}=tr:{tr_v:.2f}/cvd:{c_s}")
            if mh_parts:
                cvd_sfx += f", flow_horizons({', '.join(mh_parts)})"
            if (sp_cvd := _parse_finite_float(minfo.get("spot_cvd_15m_usd"), None)) is not None and minfo.get("spot_perp_flow_regime"):
                sp_s = f"{'+' if sp_cvd >= 0.0 else '-'}{_fmt_liq_usd(abs(sp_cvd))}"
                cvd_sfx += f", spot_cvd_15m={sp_s} ({minfo['spot_perp_flow_regime']})"

        basis_sfx = ""
        basis_parts: List[str] = []
        if (b_bps := _parse_finite_float(minfo.get("perp_spot_basis_bps"), None)) is not None:
            basis_parts.append(f"perp_basis={b_bps:+.1f}bps")
        if (f_ann := _parse_finite_float(minfo.get("funding_annualized_pct"), None)) is not None:
            basis_parts.append(f"funding_ann={f_ann:+.1f}%")
        if (arb_bps := _parse_finite_float(minfo.get("cross_venue_arb_bps"), None)) is not None and minfo.get("cross_venue_peer"):
            basis_parts.append(f"arb_vs_{minfo['cross_venue_peer']}={arb_bps:+.1f}bps")
            if (net_b := _parse_finite_float(minfo.get("net_buy_edge_vs_peer_bps"), None)) is not None and minfo.get("best_buy_venue"):
                basis_parts.append(f"net_buy_edge={net_b:+.1f}bps(best_buy={minfo['best_buy_venue']})")
        if basis_parts:
            basis_sfx = f", {', '.join(basis_parts)}"

        liq_sfx = ""
        if minfo.get("long_liq_px") is not None and minfo.get("short_liq_px") is not None:
            liq_parts: List[str] = []
            for l_lbl, px_k, usd_k, dist_k in (
                ("long_liq", "long_liq_px", "long_liq_usd", "long_liq_dist_pct"),
                ("short_liq", "short_liq_px", "short_liq_usd", "short_liq_dist_pct"),
                ("deep_long", "long_liq_deep_px", "long_liq_deep_usd", "long_liq_deep_dist_pct"),
                ("deep_short", "short_liq_deep_px", "short_liq_deep_usd", "short_liq_deep_dist_pct"),
            ):
                if minfo.get(px_k) is None:
                    continue
                l_px = _parse_finite_float(minfo.get(px_k), 0.0)
                l_d = _parse_finite_float(
                    minfo.get(dist_k),
                    ((l_px - cur_px) / cur_px) * 100.0 if cur_px > 0 and l_px > 0 else 0.0,
                )
                liq_parts.append(
                    f"{l_lbl}=${minfo.get(px_k)}[{_fmt_liq_usd(minfo.get(usd_k))}, {l_d:+.2f}%]"
                )
            liq_parts.append(f"magnet={minfo.get('liq_magnet_bias') or 'BALANCED_LIQ_MAGNET'}")
            liq_sfx = f", liq_clusters({', '.join(liq_parts)})"
        cd_sfx = ""
        if isinstance(cd_map := minfo.get("cycle_delta"), dict):
            cd_parts: List[str] = []
            if cd_map.get("dt_min") is not None:
                cd_parts.append(f"dt={cd_map['dt_min']}m")
            if (d_ob := _parse_finite_float(cd_map.get("ob_imbalance_delta"), None)) is not None and abs(d_ob) >= 0.0005:
                cd_parts.append(f"d_ob={d_ob:+.3f}")
            if (d_cvd := _parse_finite_float(cd_map.get("cvd_15m_delta_usd"), None)) is not None and abs(d_cvd) >= 1.0:
                cd_parts.append(f"d_cvd={'+' if d_cvd >= 0 else '-'}{_fmt_liq_usd(abs(d_cvd))}")
            if (d_wl := _parse_finite_float(cd_map.get("whale_long_delta_pct"), None)) is not None and abs(d_wl) >= 0.005:
                cd_parts.append(f"d_whale_long={d_wl:+.2f}%")
            if (d_oi := _parse_finite_float(cd_map.get("oi_delta_pct"), None)) is not None and abs(d_oi) >= 0.005:
                cd_parts.append(f"d_oi={d_oi:+.2f}%")
            if len(cd_parts) > 1:
                cd_sfx = f", cycle_delta({', '.join(cd_parts)})"
        lines.append(
            f"  * {sym_3b_lbl}: ob_imbalance={minfo.get('ob_imbalance')} "
            f"(bid_depth=${minfo.get('bid_depth_usd')}, ask_depth=${minfo.get('ask_depth_usd')}){depth_1pct_sfx}, "
            f"bid_wall={bw_px} (${minfo.get('bid_wall_usd')}{bw_dist}), "
            f"ask_wall={aw_px} (${minfo.get('ask_wall_usd')}{aw_dist}), "
            f"whale_long_pct={minfo.get('whale_long_pct')}, div_regime={minfo.get('divergence_regime')}, "
            f"taker_ratio={minfo.get('taker_buy_sell_ratio')} "
            f"(buy_v=${minfo.get('taker_buy_vol_usd')}/sell_v=${minfo.get('taker_sell_vol_usd')}){cvd_sfx}, "
            f"funding={minfo.get('funding_rate')} (pred={minfo.get('pred_funding_rate')}), "
            f"oi_usd={minfo.get('oi_usd')}, oi_1h_chg={minfo.get('oi_1h_chg_pct')}%{basis_sfx}{spread_sfx}{liq_sfx}{cd_sfx}"
        )
    return lines


def _same_anchor_px(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return False
    af, bf = _parse_finite_float(a, None), _parse_finite_float(b, None)
    return abs(af - bf) <= max(1e-6, max(abs(af), abs(bf)) * 2e-4) if (af is not None and bf is not None) else (a == b)


def _format_tradable_position_line(pos: Dict[str, Any], max_usd: float = 2000.0) -> str:
    """Format a single tradable position line for Section 3C."""
    t_extra = ""
    px_f = _parse_finite_float(pos.get("price"), 0.0)
    for px_k, lbl, pct_lbl, pct_k, dedup_keys in (
        ("active_avg_buy_px", "avg_buy_px", "pnl_vs_avg", "pnl_vs_avg_pct", ()),
        ("tactical_avg_buy_px", "tactical_avg_px", "pnl_vs_tactical", "pnl_vs_tactical_avg_pct", ("active_avg_buy_px", "last_buy_px")),
        ("swing_subcycle_avg_px", "swing_avg_px", "pnl_vs_swing", "pnl_vs_swing_avg_pct", ("active_avg_buy_px", "tactical_avg_buy_px", "last_buy_px")),
        ("cycle_avg_cost_px", "cycle_avg_cost", "pnl_vs_cycle_cost", "pnl_vs_cycle_cost_pct", ("active_avg_buy_px",)),
        ("last_buy_px", "last_buy_px", "pnl_vs_last_buy", "pnl_vs_last_buy_pct", ()),
        ("lowest_buy_px", "lowest_buy_px", "pnl_vs_lowest_buy", "pnl_vs_lowest_buy_pct", ("last_buy_px",)),
        ("last_sell_px", "last_sell_px", "chg_vs_last_sell", "chg_vs_last_sell_pct", ()),
        ("trailing_peak", "trail_peak", "dd_from_peak", "dd_from_peak_pct", ("active_avg_buy_px",)),
    ):
        if pos.get(px_k) is not None and not any(_same_anchor_px(pos.get(px_k), pos.get(dk)) for dk in dedup_keys):
            pct_val = pos.get(pct_k)
            if pct_val is None and px_f > 0.0:
                anchor_f = _parse_finite_float(pos.get(px_k), 0.0)
                if anchor_f > 0.0:
                    if px_k == "trailing_peak" and anchor_f < px_f:
                        anchor_f = px_f
                    pct_val = round(((px_f - anchor_f) / anchor_f) * 100.0, 2)
            pct_sfx = f" ({pct_lbl}={pct_val}%)" if pct_val is not None else ""
            t_extra += f", {lbl}={pos.get(px_k)}{pct_sfx}"
    if (lifo_p := _parse_finite_float(pos.get("lifo_realized_pnl_usd"), None)) is not None and abs(lifo_p) >= 0.01:
        t_extra += f", lifo_realized_pnl=${lifo_p:+.2f}"
    if pos.get("mt_gain_pct") or pos.get("mt_lost_pct"):
        t_extra += f", bot_ref_gain={pos.get('mt_gain_pct')}%/lost={pos.get('mt_lost_pct')}%"
    venue = pos.get("venue")
    fb_fees = resolve_venue_fee_and_limits(
        str(venue or "t212"),
        str(pos.get("symbol") or ""),
        price=_parse_finite_float(pos.get("price"), 0.0),
        free_quote=pos.get("free_quote"),
        free_base=pos.get("free_base"),
        locked_quote_usd=pos.get("locked_quote_usd"),
        locked_base_qty=pos.get("locked_base_qty"),
        spread_bps=pos.get("spread_bps"),
        gateway_max_usd=max_usd,
    )
    eff_pos = {**fb_fees, **{k: v for k, v in pos.items() if v is not None}}
    for cap_k in ("effective_max_buy_usd", "effective_max_sell_usd", "replace_max_buy_usd", "replace_max_sell_usd"):
        cap_v = _parse_finite_float(eff_pos.get(cap_k), None)
        if cap_v is not None and max_usd > 0 and cap_v > max_usd:
            eff_pos[cap_k] = round(float(max_usd), 2)
    taker_bps = _parse_finite_float(eff_pos.get("taker_fee_bps"), 15.0)
    maker_bps = _parse_finite_float(eff_pos.get("maker_fee_bps"), 15.0)
    fee_tier = eff_pos.get("fee_tier") or "STANDARD"
    sp_part = f", spread={eff_pos.get('spread_bps')}bps" if eff_pos.get("spread_bps") is not None else ""
    t_extra += (
        f", venue_fee_per_leg={taker_bps:g}bps "
        f"(tier={fee_tier}, maker={maker_bps:g}bps/taker={taker_bps:g}bps{sp_part})"
    )
    lim_parts: List[str] = []
    if eff_pos.get("exchange_min_notional_usd") is not None:
        lim_parts.append(f"min_notional=${_parse_finite_float(eff_pos.get('exchange_min_notional_usd'), 0.0):g}")
    for lim_pfx, lim_k in (
        ("min_qty=", "min_qty"),
        ("step=", "step_size"),
        ("tick=", "tick_size"),
        ("max_buy=$", "effective_max_buy_usd"),
        ("locked_quote_usd=$", "locked_quote_usd"),
        ("replace_max_buy=$", "replace_max_buy_usd"),
        ("max_sell=$", "effective_max_sell_usd"),
        ("locked_base_qty=", "locked_base_qty"),
        ("replace_max_sell=$", "replace_max_sell_usd"),
    ):
        if eff_pos.get(lim_k) is not None:
            lim_parts.append(f"{lim_pfx}{eff_pos[lim_k]}")
    if lim_parts:
        t_extra += f", limits({', '.join(lim_parts)})"
    fb_val, val_usd = pos.get("free_base"), pos.get("value_usd")
    fb_str = (
        f"{fb_val} (${val_usd} USD)"
        if fb_val is not None and val_usd is not None
        else (f"{fb_val}" if fb_val is not None else "N/A")
    )
    if (lb_q := _parse_finite_float(eff_pos.get("locked_base_qty"), None)) is not None and lb_q > 0.0:
        lb_u = _parse_finite_float(eff_pos.get("locked_base_usd"), None)
        lb_u_s = f" (${lb_u:.2f} USD)" if lb_u is not None and lb_u > 0.0 else ""
        fb_str += f", locked_base_qty={lb_q:g}{lb_u_s}"
    fq_val = pos.get("free_quote")
    base_lbl = pos.get("base") or canonical_base_asset(str(pos.get("symbol") or "")) or "BASE"
    quote_lbl = pos.get("quote") or "USD"
    fq_str = f"${fq_val}{_format_quote_breakdown(pos.get('free_quote_breakdown'), quote_lbl)}" if fq_val is not None else "N/A"
    if (lq_u := _parse_finite_float(eff_pos.get("locked_quote_usd"), None)) is not None and lq_u > 0.0:
        fq_str += f", locked_quote_usd=${lq_u:.2f}"
    return (
        f"    - [{venue}] {pos.get('symbol')}: free_{base_lbl}={fb_str}, "
        f"free_{quote_lbl}={fq_str}, "
        f"px={pos.get('price')}{t_extra}"
    )


def _format_recent_order_line(sym: str, oinfo: Dict[str, Any], now_ms: int) -> str:
    """Format a single recent order summary line for Section 3C."""
    def _side_summary(key: str) -> str:
        entry = oinfo.get(key) if isinstance(oinfo.get(key), dict) else {}
        px = _round_opt_px(entry.get("price"), 6)
        if px is None:
            return "none"
        ts = normalize_timestamp_ms(entry.get("ts_ms"), default=0) or 0
        age_h = max(0.0, round((now_ms - ts) / 3_600_000.0, 1)) if ts > 0 else None
        tag = entry.get("actor_tag") or format_actor_tag(entry.get("actor"), entry.get("module"), entry.get("rule"))
        qty_v = entry.get("qty")
        qty_pfx = f"{qty_v} @ " if qty_v is not None else ""
        meta_parts = ([f"age={age_h}h"] if age_h is not None else []) + ([f"by={tag}"] if tag else [])
        meta_sfx = f" ({', '.join(meta_parts)})" if meta_parts else ""
        return f"{qty_pfx}{px}{meta_sfx}"

    last_b_px = _round_opt_px((oinfo.get("last_buy") or {}).get("price") if isinstance(oinfo.get("last_buy"), dict) else None, 6)
    avg_b_px = _round_opt_px(oinfo.get("active_avg_buy_px"), 6)
    avg_b_str = f" | Active Cycle Avg BUY={avg_b_px}" if avg_b_px else ""
    tac_b_px = _round_opt_px(oinfo.get("tactical_avg_buy_px"), 6)
    tac_b_str = (
        f" | Tactical Sub-Cycle Avg BUY={tac_b_px}"
        if tac_b_px and not _same_anchor_px(tac_b_px, avg_b_px) and not _same_anchor_px(tac_b_px, last_b_px)
        else ""
    )
    sw_b_px = _round_opt_px(oinfo.get("swing_subcycle_avg_px"), 6)
    sw_b_str = (
        f" | Swing Sub-Cycle Avg BUY={sw_b_px}"
        if sw_b_px
        and not _same_anchor_px(sw_b_px, avg_b_px)
        and not _same_anchor_px(sw_b_px, tac_b_px)
        and not _same_anchor_px(sw_b_px, last_b_px)
        else ""
    )
    lifo_v = _parse_finite_float(oinfo.get("lifo_realized_pnl_usd"), None)
    lifo_str = f" | Sub-Cycle LIFO Realized PnL=${lifo_v:+.2f}" if lifo_v is not None and abs(lifo_v) >= 0.01 else ""
    markout_parts: List[str] = []
    if (bm_v := _parse_finite_float(oinfo.get("buy_markout_bps"), None)) is not None:
        markout_parts.append(f"buy={bm_v:+.0f}bps")
    if (sm_v := _parse_finite_float(oinfo.get("sell_markout_bps"), None)) is not None:
        markout_parts.append(f"sell={sm_v:+.0f}bps")
    markout_str = f" | Post-Trade Markout({', '.join(markout_parts)})" if markout_parts else ""
    r_buys = [f"{b.get('qty')}@{_round_opt_px(b.get('price'), 6)}" for b in (oinfo.get("recent_buys") or []) if isinstance(b, dict)]
    tranches_str = f" | Recent BUY Tranches=[{', '.join(r_buys)}]" if len(r_buys) > 1 else ""
    r_sells = [f"{s.get('qty')}@{_round_opt_px(s.get('price'), 6)}" for s in (oinfo.get("recent_sells") or []) if isinstance(s, dict) and s.get("qty") is not None]
    sell_tranches_str = f" | Recent SELL Tranches=[{', '.join(r_sells)}]" if len(r_sells) > 1 else ""
    last_act_tag = oinfo.get("last_action_actor_tag") or format_actor_tag(
        oinfo.get("last_action_actor"),
        oinfo.get("last_action_module"),
        oinfo.get("last_action_rule"),
    )
    last_act_sfx = f" (by={last_act_tag})" if last_act_tag else ""
    return (
        f"  * {sym} Last Action={oinfo.get('last_action_side')}{last_act_sfx}: "
        f"Last BUY={_side_summary('last_buy')} | "
        f"Last SELL={_side_summary('last_sell')}"
        f"{avg_b_str}{tac_b_str}{sw_b_str}{lifo_str}{markout_str}{tranches_str}{sell_tranches_str}"
    )


def _format_portfolio_section(
    ps: Dict[str, Any],
    asset_scope: str,
    *,
    max_usd: float = 2000.0,
    include_macro_benchmarks: bool = True,
) -> List[str]:
    """Format Section 3C (Spot Wallet Balances, Tradable Positions, Cycle Cost Anchors & Last Fills)."""
    lines = ["\n### 3C. SPOT WALLET BALANCES, TRADABLE POSITIONS, CYCLE COST ANCHORS & LAST FILLS:"]
    if isinstance(pre := ps.get("portfolio_risk_engine"), dict) and pre.get("unified_nav_usd"):
        pre_parts = [
            f"unified_nav=${_parse_finite_float(pre.get('unified_nav_usd'), 0.0):,.2f}",
            f"invested=${_parse_finite_float(pre.get('invested_exposure_usd'), 0.0):,.2f}",
            f"free_cash=${_parse_finite_float(pre.get('global_free_cash_usd'), 0.0):,.2f} ({pre.get('global_free_cash_pct')}%)",
        ]
        if (lq_pre := _parse_finite_float(pre.get("locked_quote_usd"), 0.0)) > 0.0:
            pre_parts.append(f"locked_quote=${lq_pre:,.2f} ({pre.get('locked_quote_pct')}%)")
        pre_parts.extend([
            f"portfolio_btc_beta={pre.get('portfolio_btc_beta')}",
            f"avg_corr_24h={pre.get('avg_corr_vs_btc_24h')}",
            f"intraday_var_95=${_parse_finite_float(pre.get('intraday_var_95_usd'), 0.0):,.2f} ({pre.get('intraday_var_95_pct')}%)",
            f"regime={pre.get('concentration_regime')}",
        ])
        if pre.get("intraday_cvar_975_usd") is not None:
            pre_parts.append(
                f"cvar_975=${_parse_finite_float(pre.get('intraday_cvar_975_usd'), 0.0):,.2f} ({pre.get('intraday_cvar_975_pct')}%)"
            )
        if pre.get("vol_target_tranche_usd") is not None:
            pre_parts.append(f"vol_target_tranche=${_parse_finite_float(pre.get('vol_target_tranche_usd'), 0.0):,.0f}")
        if pre.get("subcycle_lifo_realized_pnl_usd") is not None:
            pre_parts.append(f"subcycle_lifo_pnl=${_parse_finite_float(pre.get('subcycle_lifo_realized_pnl_usd'), 0.0):+.2f}")
        if pre.get("est_recent_fee_drag_usd") is not None:
            pre_parts.append(f"est_fee_drag=${_parse_finite_float(pre.get('est_recent_fee_drag_usd'), 0.0):.2f}")
        pt_parts: List[str] = []
        if (em_v := _parse_finite_float(pre.get("avg_entry_markout_bps"), None)) is not None:
            pt_parts.append(f"entry={em_v:+.0f}bps")
        if (xm_v := _parse_finite_float(pre.get("avg_exit_markout_bps"), None)) is not None:
            pt_parts.append(f"exit={xm_v:+.0f}bps")
        if pre.get("post_trade_win_rate_pct") is not None:
            pt_parts.append(f"win_rate={pre.get('post_trade_win_rate_pct')}%")
        if pt_parts:
            pre_parts.append(f"post_trade_edge({', '.join(pt_parts)})")
        if isinstance(cv_map := pre.get("cross_venue_routing"), dict) and cv_map:
            cv_s = ", ".join(
                f"{b_k}:buy={v_d.get('best_buy')}[+{v_d.get('net_buy_edge_bps')}bps]/sell={v_d.get('best_sell')}[+{v_d.get('net_sell_edge_bps')}bps]"
                for b_k, v_d in sorted(cv_map.items())
                if isinstance(v_d, dict)
            )
            if cv_s:
                pre_parts.append(f"smart_routing({cv_s})")
        lines.append(f"  * Enterprise Global Portfolio Risk Engine: {', '.join(pre_parts)}")
    if ps.get("total_value_usdc") is not None:
        tv_extras: List[str] = []
        if ps.get("total_value_usdc_age_min") is not None:
            tv_extras.append(f"snapshot_age={ps['total_value_usdc_age_min']}m")
        if (live_bn := _parse_finite_float(ps.get("live_binance_equity_usdc"), None)) is not None and live_bn > 0.0:
            tv_extras.append(f"live_reconciled=${live_bn:,.2f} USDC")
        tv_sfx = f" ({', '.join(tv_extras)})" if tv_extras else ""
        lines.append(f"  * Total Binance Portfolio Value: ${_parse_finite_float(ps.get('total_value_usdc'), 0.0):,.2f} USDC{tv_sfx}")
    if ps.get("venue_quote_cash"):
        cash_parts: List[str] = []
        for v in ps["venue_quote_cash"].values():
            if not isinstance(v, dict):
                continue
            part = (
                f"{v.get('venue')} {v.get('asset')}=${_parse_finite_float(v.get('free'), 0.0):,.2f}"
                f"{_format_quote_breakdown(v.get('breakdown'), v.get('asset'))}"
            )
            if (lq_v := _parse_finite_float(v.get("locked_quote_usd"), 0.0)) > 0.0:
                part += f" [locked_quote_usd=${lq_v:,.2f}]"
            cash_parts.append(part)
        if cash_parts:
            lines.append(f"  * Available Free Quote Cash per Venue: {', '.join(cash_parts)}")
    if ps.get("tradable_positions"):
        lines.append("  * Live Tradable Spot Instruments (Eligible for Autonomous BUY / SELL):")
        for pos in ps["tradable_positions"]:
            lines.append(_format_tradable_position_line(pos, max_usd=max_usd))
    open_lim = (
        ps.get("open_limit_orders")
        if isinstance(ps.get("open_limit_orders"), list)
        else (ps.get("open_orders") if isinstance(ps.get("open_orders"), list) else [])
    )
    if open_lim:
        _dup_oo_keys = {"orderId", "origQty", "executedQty", "clientOrderId", "client_order_id", "created_ts", "age_min"}
        clean_open_lim = [
            {
                k: v
                for k, v in o.items()
                if k not in _dup_oo_keys
                and not (
                    k in ("orig_qty", "remaining_qty")
                    and v == o.get("qty")
                    and (_parse_finite_float(o.get("filled_qty"), 0.0) or 0.0) == 0.0
                )
            }
            if isinstance(o, dict)
            else o
            for o in open_lim
        ]
        lines.append(
            f"  * Active Open Orders on Exchange (Locking Liquidity) — open_limit_orders: {clean_open_lim}"
        )
    else:
        lines.append(
            "  * Active Open Limit Orders on Exchange — open_limit_orders: [] (none resting)"
        )
    for cyc_key, venue_lbl, venue_id in (
        ("kraken_cycles", "Kraken", "kraken"),
        ("hyperliquid_cycles", "Hyperliquid", "hyperliquid"),
        ("t212_cycles", "Trading212", "t212"),
    ):
        if ps.get(cyc_key):
            lines.append(f"  * {venue_lbl} Active Cycle Cost & Drawdown Anchors:")
            for c_sym, c_data in sorted(ps[cyc_key].items()):
                lines.append(_format_venue_cycle_line(venue_id, c_sym, c_data))
    if ps.get("kraken_xstocks"):
        xs_raw = ps["kraken_xstocks"]
        if isinstance(xs_raw, dict):
            xs: Any = {k: v for k, v in xs_raw.items() if v is not None}
            if isinstance(xs.get("holdings"), dict):
                xs["holdings"] = {hk: hv for hk, hv in xs["holdings"].items() if _parse_finite_float(hv, 0.0) >= 1e-4}
            if isinstance(xs.get("t212_sibling_cycle"), dict):
                xs["t212_sibling_cycle"] = {
                    ck: cv
                    for ck, cv in xs["t212_sibling_cycle"].items()
                    if cv is not None and not (ck == "pnl_vs_avg_pct" and xs["t212_sibling_cycle"].get("dd_pct") is not None)
                }
        else:
            xs = xs_raw
        lines.append(
            f"  * Kraken xStocks Pre-IPO Holdings (READ-ONLY on Kraken — NOT API-tradable; cross-venue SpaceX exposure for Trading212 {xs.get('t212_sibling') if isinstance(xs, dict) else None}): {xs}"
        )
    if ps.get("cross_market_context"):
        cmc = ps["cross_market_context"]
        if isinstance(cmc, dict):
            if cmc.get("crypto_bellwethers") and asset_scope == "equities":
                lines.append(
                    f"  * Cross-Market Live Crypto Telemetry (Raw 24/7 Prices, Multi-Horizon Deltas & Microstructure): {cmc['crypto_bellwethers']}"
                )
            if cmc.get("equity_bellwethers") and asset_scope != "equities":
                lines.append(
                    f"  * Cross-Market US Equities Telemetry (Trading212 Raw Prices, Multi-Horizon Deltas & Active Cycles): {cmc['equity_bellwethers']}"
                )
    mb_bench_3c = (
        _extract_macro_benchmarks_line(
            ps.get("macro_benchmarks")
            or (
                ps["cross_market_context"].get("macro_benchmarks")
                if isinstance(ps.get("cross_market_context"), dict)
                else None
            )
        )
        if include_macro_benchmarks
        else ""
    )
    if mb_bench_3c:
        lines.append(
            f"  * Macro & US Equities Live Correlation Index (SPY / QQQ / ES=F / NQ=F / DXY / US10Y 24h Delta): {mb_bench_3c}"
        )
    if ps.get("hyperliquid_readonly"):
        lines.append(f"  * Hyperliquid Spot/Perp Equity Summary: {ps['hyperliquid_readonly']}")
    now_ms = int(time.time() * 1000)
    for sym, oinfo in sorted((ps.get("recent_orders") or {}).items()):
        if isinstance(oinfo, dict):
            lines.append(_format_recent_order_line(sym, oinfo, now_ms))
    shown_open_oids = {
        str(oo.get("order_id") or oo.get("orderId"))
        for oo in open_lim
        if isinstance(oo, dict) and (oo.get("order_id") or oo.get("orderId"))
    }
    if ps.get("last_cycle_execution_status"):
        raw_lces = ps["last_cycle_execution_status"]
        if isinstance(raw_lces, list):
            cleaned_list = [{k: v for k, v in item.items() if v is not None} if isinstance(item, dict) else item for item in raw_lces]
            non_dup_lces = [
                item for item in cleaned_list
                if not (
                    isinstance(item, dict)
                    and item.get("status") == "OPEN_RESTING"
                    and _parse_finite_float(item.get("age_min"), 0.0) > 30.0
                    and str(item.get("order_id") or "") in shown_open_oids
                )
            ]
            clean_lces: Any = non_dup_lces if non_dup_lces else cleaned_list[-1:]
        elif isinstance(raw_lces, dict):
            clean_lces = {k: v for k, v in raw_lces.items() if v is not None}
        else:
            clean_lces = raw_lces
        lines.append(
            f"  * Last Cycle Execution Status — last_cycle_execution_status: {clean_lces}"
        )
    for st_key, st_label in (
        ("active_retry_queue", "Active Queued Retry Orders (order_retry_queue.jsonl)"),
        ("monitortrades_active_state", "Active MonitorTrades Ride/Stop State (mt_state.json)"),
        ("assetguardian_state", "Active AssetGuardian State (assetguardian_state.json)"),
        ("smart_stop_loss_state", "Active Smart Stop-Loss State (smart_stop_loss_state.json)"),
        ("recent_ai_intents", "Recent Autonomous Intents"),
    ):
        if ps.get(st_key):
            st_val = ps[st_key]
            if st_key == "recent_ai_intents" and isinstance(st_val, list):
                fmt_intents = []
                last_norm_intent = None
                for it in st_val:
                    if not isinstance(it, dict):
                        fmt_intents.append(it)
                        last_norm_intent = it
                        continue
                    v_st = str(it.get("venue_status") or "").upper()
                    oid_s = str(it.get("order_id") or "")
                    it_copy = dict(it)
                    if it_copy.get("status") == "EXECUTED" and v_st in ("OPEN", "NEW", "RESTING"):
                        it_copy["status"] = "RESTING_IN_BOOK"
                    last_norm_intent = it_copy
                    if (
                        v_st in ("OPEN", "NEW", "RESTING")
                        and oid_s in shown_open_oids
                        and _parse_finite_float(it.get("age_h"), 0.0) > 0.5
                    ):
                        continue
                    fmt_intents.append(it_copy)
                st_val = fmt_intents if fmt_intents else ([last_norm_intent] if last_norm_intent is not None else st_val[-1:])
            lines.append(f"  * {st_label}: {st_val}")
    return lines


