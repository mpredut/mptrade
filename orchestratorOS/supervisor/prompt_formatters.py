"""Pure formatting helpers for Autonomous AI Reconciler Gemini prompts.

Separates prompt presentation (Sections 1, 2, 3A, 3B, 3C) and symbol price
resolution from telemetry harvesting and order execution in
`autonomous_ai_reconciler.py`.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from botcore import safe_float as _parse_finite_float
from cachepaths import canonical_base_asset, normalize_venue_name
from intelligence.sentiment.guards.high_stake_guard import (
    _VENUE_DEFAULT_FEE_SPECS,
    _build_vp_position_dict,
    _compute_horizon_range_dict,
    compute_liquidation_clusters,
    enrich_microstructure_cvd,
    format_macro_benchmarks_summary_line,
    resolve_symbol_price as _resolve_symbol_price,
    round_opt_px as _round_opt_px,
)
from orchestratorOS.agy.topics import topic_spec
from orchestratorOS.lib.technical_repair_ssot import (
    ALL_DIAGNOSED_CATEGORIES,
    ALL_REPAIR_ACTIONS,
    AUDITED_CONFIG_FILES,
    DIAGNOSED_SEVERITIES,
)
from providers.base import format_actor_tag, normalize_timestamp_ms


def _format_venue_cycle_line(venue: str, sym: str, cyc: Dict[str, Any]) -> str:
    """Format a normalized venue cycle anchor entry for Section 3C of the AI prompt."""
    extra_parts: List[str] = []
    if cyc.get("pos_peak") is not None:
        extra_parts.append(f"pos_peak={cyc.get('pos_peak')}")
    if cyc.get("realized_pnl_usd") is not None:
        extra_parts.append(f"realized_pnl=${cyc.get('realized_pnl_usd')}")
    if cyc.get("tp_sold_levels"):
        extra_parts.append(f"tp_sold={cyc.get('tp_sold_levels')}")
    if cyc.get("resting_orders"):
        extra_parts.append(f"resting_orders={cyc.get('resting_orders')}")
    extra_sfx = f", {', '.join(extra_parts)}" if extra_parts else ""
    exit_kind = cyc.get("last_exit_kind")
    exit_kind_sfx = f" ({exit_kind})" if exit_kind else ""
    dd_val = cyc.get("dd_pct")
    dd_str = f"{dd_val}%" if dd_val is not None else "N/A"
    return (
        f"    - [{venue}] {sym}: qty={cyc.get('qty')}, cost=${cyc.get('cost_usd')}, "
        f"avg_cost={cyc.get('avg_cost_px')}, entry_px={cyc.get('entry_price')}, "
        f"last_buy={cyc.get('last_buy_price')}, last_sell={cyc.get('last_sell_price')}"
        f"{exit_kind_sfx}, dca_buys={cyc.get('dca_buys')}, dd={dd_str}{extra_sfx}"
    )


def _format_horizon_range(pinfo: Dict[str, Any], px_f: float, horizon: str) -> str:
    """Format `[low, high] pos_<horizon>=X% (dd_high=..., bounce_low=..., span=...)` for 24h or 7d."""
    lo, hi = pinfo.get(f"low_{horizon}"), pinfo.get(f"high_{horizon}")
    lo_f = _parse_finite_float(lo, 0.0)
    hi_f = _parse_finite_float(hi, 0.0)
    pos_sfx = ""
    if px_f > 0 and lo_f > 0 and hi_f > lo_f:
        fb_rng = _compute_horizon_range_dict(px_f, lo_f, hi_f, horizon)
        pos_v = pinfo.get(f"pos_{horizon}_pct") if pinfo.get(f"pos_{horizon}_pct") is not None else fb_rng.get(f"pos_{horizon}_pct")
        dd_v = pinfo.get(f"dd_{horizon}_high_pct") if pinfo.get(f"dd_{horizon}_high_pct") is not None else fb_rng.get(f"dd_{horizon}_high_pct")
        bnc_v = pinfo.get(f"bounce_{horizon}_low_pct") if pinfo.get(f"bounce_{horizon}_low_pct") is not None else fb_rng.get(f"bounce_{horizon}_low_pct")
        spn_v = pinfo.get(f"span_{horizon}_pct") if pinfo.get(f"span_{horizon}_pct") is not None else fb_rng.get(f"span_{horizon}_pct")
        pos_sfx = (
            f" pos_{horizon}={pos_v}% (dd_high={_parse_finite_float(dd_v, 0.0):+.2f}%, "
            f"bounce_low={_parse_finite_float(bnc_v, 0.0):+.2f}%, span={spn_v}%)"
        )
    return f"{horizon}_range=[{lo}, {hi}]{pos_sfx}"


def _extract_macro_benchmarks_line(mb_raw: Any) -> str:
    """Extract or synthesize the 1-line SPY / QQQ / ES=F / NQ=F / DXY / US10Y benchmark summary."""
    if not isinstance(mb_raw, dict) or not mb_raw:
        return ""
    summary_s = str(mb_raw.get("summary_line") or "").strip()
    if not summary_s:
        summary_s = format_macro_benchmarks_summary_line(mb_raw)
    if not summary_s:
        return ""
    age_sfx = f" (age={mb_raw.get('age_min')}m)" if mb_raw.get("age_min") is not None else ""
    return f"{summary_s}{age_sfx}"


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
        + (f", age={fg.get('age_min')}m" if fg.get("age_min") is not None else "")
        + ")"
        if fg else "None"
    )
    mb_gainers = mb.get("top_gainers")[:3] if isinstance(mb.get("top_gainers"), list) else None
    mb_losers = mb.get("top_losers")[:3] if isinstance(mb.get("top_losers"), list) else None
    mb_line = (
        f"advance_ratio={mb.get('advance_ratio', 'N/A')}, regime={mb.get('regime', 'N/A')}, "
        f"median_chg={mb.get('median_change_pct', 'N/A')}%, "
        f"mean_chg={mb.get('mean_change_pct', 'N/A')}%, disp_std={mb.get('dispersion_std', 'N/A')}, "
        f"advancing={mb.get('advancing', 'N/A')}/{mb.get('total_symbols', 'N/A')}"
        + (f", top_gainers={mb_gainers}" if mb_gainers else "")
        + (f", top_losers={mb_losers}" if mb_losers else "")
        if mb else "None"
    )
    sa_line = (
        f"bias={sa.get('market_bias', 'N/A')}, risk={sa.get('risk_level', 'N/A')}, "
        f"action={sa.get('recommended_action', 'N/A')}, conf={sa.get('confidence', 'N/A')}"
        + (f", age={sa.get('age_min')}m" if sa.get("age_min") is not None else "")
        + f" | summary={sa.get('summary', '')} | key_risks={sa.get('key_risks', [])}"
        if sa else "None"
    )
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
    raw_gt_level = str(gt.get("threat_level", "N/A"))
    raw_gt_brake = str(gt.get("recommended_brake", "N/A"))
    ai_gt_level = _AI_THREAT_LEVEL_MAP.get(raw_gt_level.strip().upper(), raw_gt_level)
    ai_gt_brake = _AI_BRAKE_MAP.get(raw_gt_brake.strip().upper(), raw_gt_brake)
    gt_line = (
        f"level={ai_gt_level}, risk_score={gt.get('risk_score', 'N/A')}, "
        f"brake={ai_gt_brake}"
        + (f", age={gt.get('age_min')}m" if gt.get("age_min") is not None else "")
        + f" | summary={gt.get('summary', '')}"
        if gt else "None"
    )
    lines = [
        "### 1. ACTIVE MACRO, SENTIMENT, GEOPOLITICAL & LIVE BREAKING NEWS TELEMETRY:",
        f"- Fear & Greed: {fg_line}",
        f"- Market Breadth: {mb_line}",
        f"- Macro Sentiment Advisor: {sa_line}",
        f"- Geopolitical Threat Shield: {gt_line}",
    ]
    mb_bench_line = _extract_macro_benchmarks_line(mb_bench)
    if mb_bench_line:
        lines.append(
            f"- US Equities, Futures & Macro Rates Benchmarks (SPY / QQQ / ES=F / NQ=F / DXY / US10Y 24h Delta): {mb_bench_line}"
        )
    breaking_news = getattr(sensors, "breaking_news", None) or macro.get("recent_news") or []
    if breaking_news:
        lines.append("- Live Breaking News Feed (cachedb/news_feed_collect.json, 1m refresh):")
        for hn in breaking_news:
            sev_tag = "[HIGH-SEV] " if hn.get("is_high_severity") else ""
            kw_sfx = f" kw={hn.get('keywords')}" if hn.get("keywords") else ""
            age_sfx = f" (age={hn.get('age_min')}m)" if hn.get("age_min") is not None else ""
            lines.append(f"  * {sev_tag}{hn.get('title')}{kw_sfx}{age_sfx}")
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
    btc_entry = next(
        (
            entry for k, entry in prices_and_trends.items()
            if isinstance(entry, dict) and canonical_base_asset(k) == "BTC"
        ),
        None,
    )
    grouped_3a: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for s_k, s_v in sorted(prices_and_trends.items()):
        if isinstance(s_v, dict):
            grouped_3a.setdefault(canonical_base_asset(s_k), []).append((s_k, s_v))

    for _, sym_group in sorted(grouped_3a.items(), key=lambda kv: kv[1][0][0]):
        sym_group.sort(
            key=lambda item: (
                0 if item[0].endswith(("USDC", "_US_EQ", "_EQ")) else 1,
                -len(item[1]),
                item[0],
            )
        )
        sym, pinfo = sym_group[0]
        px = pinfo.get("price")
        px_f = _parse_finite_float(px, 0.0)
        px_str = (
            f"${px_f:,.2f}"
            if px_f >= 1.0
            else (f"${px_f:.6g}" if px_f > 0 else (f"${px}" if px is not None and not isinstance(px, bool) else "N/A"))
        )
        sib_lbl = ""
        if len(sym_group) > 1:
            sib_parts: List[str] = []
            for sib_sym, sib_info in sym_group[1:]:
                s_px = _parse_finite_float(sib_info.get("price"), 0.0)
                if s_px > 0:
                    s_px_s = f"${s_px:,.2f}" if s_px >= 1.0 else f"${s_px:.6g}"
                    if px_f > 0 and abs(s_px - px_f) > 1e-9:
                        sib_parts.append(f"{sib_sym}={s_px_s} [{(s_px - px_f) / px_f * 100.0:+.2f}%]")
                    else:
                        sib_parts.append(f"{sib_sym}={s_px_s}")
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
        if (
            pinfo.get("poc_px") is not None
            and pinfo.get("val_px") is not None
            and pinfo.get("vah_px") is not None
        ):
            fb_vp = _build_vp_position_dict(
                px_f,
                _parse_finite_float(pinfo.get("poc_px"), 0.0),
                _parse_finite_float(pinfo.get("val_px"), 0.0),
                _parse_finite_float(pinfo.get("vah_px"), 0.0),
            )
            poc_d = _parse_finite_float(
                pinfo.get("poc_dist_pct") if pinfo.get("poc_dist_pct") is not None else fb_vp.get("poc_dist_pct"), 0.0
            )
            val_d = _parse_finite_float(
                pinfo.get("val_dist_pct") if pinfo.get("val_dist_pct") is not None else fb_vp.get("val_dist_pct"), 0.0
            )
            vah_d = _parse_finite_float(
                pinfo.get("vah_dist_pct") if pinfo.get("vah_dist_pct") is not None else fb_vp.get("vah_dist_pct"), 0.0
            )
            va_pos = pinfo.get("va_pos_pct") if pinfo.get("va_pos_pct") is not None else fb_vp.get("va_pos_pct")
            va_reg = pinfo.get("va_regime") or fb_vp.get("va_regime") or "INSIDE_VA"
            extras.append(
                f"vol_profile(POC={pinfo.get('poc_px')}[{poc_d:+.2f}%], "
                f"VAL={pinfo.get('val_px')}[{val_d:+.2f}%], "
                f"VAH={pinfo.get('vah_px')}[{vah_d:+.2f}%], "
                f"va_pos={va_pos}%, regime={va_reg})"
            )

        if not sym.endswith(("_EQ", "_US_EQ")) and canonical_base_asset(sym) != "BTC":
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
            cross_parts: List[str] = []
            if pinfo.get("alt_btc_pos_24h_pct") is not None:
                cross_parts.append(f"pos={pinfo.get('alt_btc_pos_24h_pct')}%")
            if pinfo.get("alt_btc_dd_24h_high_pct") is not None:
                cross_parts.append(f"dd_high={_parse_finite_float(pinfo.get('alt_btc_dd_24h_high_pct'), 0.0):+.2f}%")
            if pinfo.get("alt_btc_bounce_24h_low_pct") is not None:
                cross_parts.append(f"bounce_low={_parse_finite_float(pinfo.get('alt_btc_bounce_24h_low_pct'), 0.0):+.2f}%")
            if cross_parts:
                extras.append(f"alt_btc_24h({', '.join(cross_parts)})")

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
            s90_pfx = f"90s={s90_v:+.5f}, " if s90_v is not None else ""
            gr_v = _parse_finite_float(pinfo.get("gradient_recent"), None)
            grad_rec_sfx = f", grad_rec={gr_v:+.4f}" if gr_v is not None and abs(gr_v) > 1e-9 else ""
            gf_v = _parse_finite_float(pinfo.get("gradient_recent_fast"), None)
            eps_v = _parse_finite_float(pinfo.get("epsilon"), None)
            fast_sfx = f", grad_fast={gf_v:+.4f}/eps={eps_v:.4f}" if gf_v is not None and eps_v is not None else ""
            g_parts = [
                f"{g_lbl}={gv:+.5f}"
                for g_lbl, g_k in (("90s", "growth_90s"), ("3.7m", "growth_3_7m"), ("15m", "growth_15m"), ("2.5h", "growth_2_5h"))
                if (gv := _parse_finite_float(pinfo.get(g_k), None)) is not None
            ]
            g_sfx = f" g_coeff({', '.join(g_parts)})" if g_parts else ""
            extras.append(
                f"slopes({s90_pfx}3.7m={_parse_finite_float(pinfo.get('slope_3_7m'), 0.0):+.5f}, "
                f"15m={_parse_finite_float(pinfo.get('slope_15m'), 0.0):+.5f}, "
                f"2.5h={_parse_finite_float(pinfo.get('slope_2_5h'), 0.0):+.5f}{grad_rec_sfx}{fast_sfx}){g_sfx}"
            )
        if pinfo.get("vol_1h_pct") is not None:
            vol_part = f"vol_1h={pinfo.get('vol_1h_pct')}%"
            if pinfo.get("adapt_reentry_pct") is not None:
                vol_part += f"(adapt_reentry={pinfo.get('adapt_reentry_pct')}%)"
            if pinfo.get("adapt_dca_pct") is not None:
                vol_part += f" adapt_dca={pinfo.get('adapt_dca_pct')}%"
            extras.append(vol_part)
        if (kv := _parse_finite_float(pinfo.get("kalman_vel"), None)) is not None:
            extras.append(f"kalman_vel={kv:+.5f}")
        if pinfo.get("long_trend_dir"):
            lt_str = f"long_trend={pinfo.get('long_trend_dir')}({pinfo.get('long_trend_hours')}h"
            if pinfo.get("long_trend_est_rem_hours") is not None:
                lt_str += f", est_rem={pinfo.get('long_trend_est_rem_hours')}h"
            if pinfo.get("weibull_median_d") is not None:
                lt_str += f", med_d={pinfo.get('weibull_median_d')}d/p90={pinfo.get('weibull_p90_d')}d"
            lt_str += ")"
            extras.append(lt_str)
        extra_str = f" | {' | '.join(extras)}" if extras else ""
        lines.append(f"  * {sym}{sib_lbl}: Price={px_str}{extra_str}")
    return lines


def _format_microstructure_section(
    microstructure: Dict[str, Any],
    prices_and_trends: Dict[str, Any],
) -> List[str]:
    """Format Section 3B (Orderbook Walls, Derivatives & Whale Flow)."""
    lines = ["\n### 3B. PILLAR 2 MICROSTRUCTURE, ORDERBOOK WALLS, DERIVATIVES & WHALE FLOW:"]
    seen_3b_bases: set[str] = set()
    sorted_3b = sorted(
        microstructure.items(),
        key=lambda kv: (0 if kv[0].endswith("USDC") else 1, kv[0]),
    )
    for sym, minfo in sorted_3b:
        if not isinstance(minfo, dict):
            continue
        b_canon_3b = canonical_base_asset(sym)
        if b_canon_3b in seen_3b_bases:
            continue
        seen_3b_bases.add(b_canon_3b)
        sib_3b = [
            other_s
            for other_s in microstructure
            if other_s != sym and canonical_base_asset(other_s) == b_canon_3b
        ]
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
        bw_dist = f", dist={(bw_px_f - cur_px) / cur_px * 100.0:+.2f}%" if cur_px > 0 and bw_px_f > 0 else ""
        aw_dist = f", dist={(aw_px_f - cur_px) / cur_px * 100.0:+.2f}%" if cur_px > 0 and aw_px_f > 0 else ""
        spread_sfx = ""
        if minfo.get("spread_bps") is not None:
            ba_lbl = f" (bid={b_bid}/ask={b_ask})" if b_bid > 0 and b_ask > 0 else ""
            spread_sfx = f", spread={minfo.get('spread_bps')}bps{ba_lbl}"

        def _fmt_liq_usd(val: Any) -> str:
            vf = _parse_finite_float(val, 0.0)
            if vf >= 1_000_000.0:
                return f"${vf / 1_000_000.0:.2f}M"
            if vf >= 1_000.0:
                return f"${vf / 1_000.0:.1f}K"
            return f"${vf:.0f}"

        raw_p = (
            prices_and_trends.get(sym)
            if isinstance(prices_and_trends.get(sym), dict)
            else next(
                (
                    v
                    for k, v in prices_and_trends.items()
                    if isinstance(v, dict) and canonical_base_asset(k) == b_canon_3b
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
        ):
            enrich_microstructure_cvd(minfo, p_ctx)
        if minfo.get("long_liq_px") is None or minfo.get("short_liq_px") is None:
            liq_lazy = compute_liquidation_clusters(p_ctx, minfo)
            if liq_lazy:
                minfo.update(liq_lazy)
        d1_b = _parse_finite_float(minfo.get("depth_1pct_bid_usd"), 0.0)
        d1_a = _parse_finite_float(minfo.get("depth_1pct_ask_usd"), 0.0)
        depth_1pct_sfx = (
            f", depth_1pct=(bid={_fmt_liq_usd(d1_b)}/ask={_fmt_liq_usd(d1_a)}, slip_2k={minfo.get('est_slip_2k_bps')}bps)"
            if (d1_b > 0.0 or d1_a > 0.0)
            else ""
        )

        cvd_sfx = ""
        cvd_usd_raw = minfo.get("cvd_15m_usd")
        if cvd_usd_raw is not None:
            cvd_u = _parse_finite_float(cvd_usd_raw, 0.0)
            cvd_b = minfo.get("cvd_15m_base")
            cvd_reg = minfo.get("cvd_regime") or "balanced_flow"
            sign_s = "+" if cvd_u >= 0.0 else "-"
            usd_s = f"{sign_s}{_fmt_liq_usd(abs(cvd_u))}"
            if cvd_b is not None:
                cvd_b_f = _parse_finite_float(cvd_b, 0.0)
                cvd_sfx = f", cvd_15m={cvd_b_f:+g} {b_canon_3b} ({usd_s}, {cvd_reg})"
            else:
                cvd_sfx = f", cvd_15m={usd_s} ({cvd_reg})"

        liq_sfx = ""
        if minfo.get("long_liq_px") is not None and minfo.get("short_liq_px") is not None:
            ll_d = _parse_finite_float(minfo.get("long_liq_dist_pct"), 0.0)
            sl_d = _parse_finite_float(minfo.get("short_liq_dist_pct"), 0.0)
            lld_d = _parse_finite_float(minfo.get("long_liq_deep_dist_pct"), 0.0)
            sld_d = _parse_finite_float(minfo.get("short_liq_deep_dist_pct"), 0.0)
            liq_sfx = (
                f", liq_clusters("
                f"long_liq=${minfo.get('long_liq_px')}[{_fmt_liq_usd(minfo.get('long_liq_usd'))}, {ll_d:+.2f}%], "
                f"short_liq=${minfo.get('short_liq_px')}[{_fmt_liq_usd(minfo.get('short_liq_usd'))}, {sl_d:+.2f}%], "
                f"deep_long=${minfo.get('long_liq_deep_px')}[{_fmt_liq_usd(minfo.get('long_liq_deep_usd'))}, {lld_d:+.2f}%], "
                f"deep_short=${minfo.get('short_liq_deep_px')}[{_fmt_liq_usd(minfo.get('short_liq_deep_usd'))}, {sld_d:+.2f}%], "
                f"magnet={minfo.get('liq_magnet_bias')})"
            )
        lines.append(
            f"  * {sym_3b_lbl}: ob_imbalance={minfo.get('ob_imbalance')} "
            f"(bid_depth=${minfo.get('bid_depth_usd')}, ask_depth=${minfo.get('ask_depth_usd')}){depth_1pct_sfx}, "
            f"bid_wall={bw_px} (${minfo.get('bid_wall_usd')}{bw_dist}), "
            f"ask_wall={aw_px} (${minfo.get('ask_wall_usd')}{aw_dist}), "
            f"whale_long_pct={minfo.get('whale_long_pct')}, div_regime={minfo.get('divergence_regime')}, "
            f"taker_ratio={minfo.get('taker_buy_sell_ratio')} "
            f"(buy_v=${minfo.get('taker_buy_vol_usd')}/sell_v=${minfo.get('taker_sell_vol_usd')}){cvd_sfx}, "
            f"funding={minfo.get('funding_rate')} (pred={minfo.get('pred_funding_rate')}), "
            f"oi_usd={minfo.get('oi_usd')}, oi_1h_chg={minfo.get('oi_1h_chg_pct')}%{spread_sfx}{liq_sfx}"
        )
    return lines


def _format_tradable_position_line(pos: Dict[str, Any]) -> str:
    """Format a single tradable position line for Section 3C."""
    t_extra = ""
    if pos.get("active_avg_buy_px") is not None:
        t_extra += f", avg_buy_px={pos.get('active_avg_buy_px')} (pnl_vs_avg={pos.get('pnl_vs_avg_pct')}%)"
    if (
        pos.get("cycle_avg_cost_px") is not None
        and pos.get("cycle_avg_cost_px") != pos.get("active_avg_buy_px")
    ):
        t_extra += f", cycle_avg_cost={pos.get('cycle_avg_cost_px')} (pnl_vs_cycle_cost={pos.get('pnl_vs_cycle_cost_pct')}%)"
    if pos.get("last_buy_px") is not None:
        t_extra += f", last_buy_px={pos.get('last_buy_px')} (pnl_vs_last_buy={pos.get('pnl_vs_last_buy_pct')}%)"
    if (
        pos.get("lowest_buy_px") is not None
        and pos.get("lowest_buy_px") != pos.get("last_buy_px")
    ):
        t_extra += f", lowest_buy_px={pos.get('lowest_buy_px')} (pnl_vs_lowest_buy={pos.get('pnl_vs_lowest_buy_pct')}%)"
    if pos.get("last_sell_px") is not None:
        t_extra += f", last_sell_px={pos.get('last_sell_px')} (chg_vs_last_sell={pos.get('chg_vs_last_sell_pct')}%)"
    if pos.get("trailing_peak") is not None:
        t_extra += f", trail_peak={pos.get('trailing_peak')} (dd_from_peak={pos.get('dd_from_peak_pct')}%)"
    if pos.get("mt_gain_pct") or pos.get("mt_lost_pct"):
        t_extra += f", bot_ref_gain={pos.get('mt_gain_pct')}%/lost={pos.get('mt_lost_pct')}%"
    venue = pos.get("venue")
    def_tier, def_maker, def_taker, _ = _VENUE_DEFAULT_FEE_SPECS.get(
        normalize_venue_name(venue), ("INVEST_RTH", 15.0, 15.0, 1.0)
    )
    taker_bps = _parse_finite_float(pos.get("taker_fee_bps"), def_taker)
    maker_bps = _parse_finite_float(pos.get("maker_fee_bps"), def_maker)
    fee_tier = pos.get("fee_tier") or def_tier
    sp_part = f", spread={pos.get('spread_bps')}bps" if pos.get("spread_bps") is not None else ""
    t_extra += (
        f", venue_fee_per_leg={taker_bps:g}bps "
        f"(tier={fee_tier}, maker={maker_bps:g}bps/taker={taker_bps:g}bps{sp_part})"
    )
    lim_parts: List[str] = []
    if pos.get("exchange_min_notional_usd") is not None:
        lim_parts.append(f"min_notional=${_parse_finite_float(pos.get('exchange_min_notional_usd'), 0.0):g}")
    if pos.get("min_qty") is not None:
        lim_parts.append(f"min_qty={pos.get('min_qty')}")
    if pos.get("step_size") is not None:
        lim_parts.append(f"step={pos.get('step_size')}")
    if pos.get("tick_size") is not None:
        lim_parts.append(f"tick={pos.get('tick_size')}")
    if pos.get("effective_max_buy_usd") is not None:
        lim_parts.append(f"max_buy=${pos.get('effective_max_buy_usd')}")
    if pos.get("effective_max_sell_usd") is not None:
        lim_parts.append(f"max_sell=${pos.get('effective_max_sell_usd')}")
    if lim_parts:
        t_extra += f", limits({', '.join(lim_parts)})"
    fb_val = pos.get("free_base")
    val_usd = pos.get("value_usd")
    fb_str = (
        f"{fb_val} (${val_usd} USD)"
        if fb_val is not None and val_usd is not None
        else (f"{fb_val}" if fb_val is not None else "N/A")
    )
    fq_val = pos.get("free_quote")
    fq_str = f"${fq_val}" if fq_val is not None else "N/A"
    fq_bd = pos.get("free_quote_breakdown")
    if fq_val is not None and isinstance(fq_bd, dict) and (len(fq_bd) > 1 or any(k != pos.get("quote") for k in fq_bd)):
        bd_s = ", ".join(f"{bk}=${_parse_finite_float(bv, 0.0):,.2f}" for bk, bv in fq_bd.items())
        fq_str += f" ({bd_s})"
    return (
        f"    - [{venue}] {pos['symbol']}: free_{pos['base']}={fb_str}, "
        f"free_{pos['quote']}={fq_str}, "
        f"px={pos.get('price')}{t_extra}"
    )


def _format_recent_order_line(sym: str, oinfo: Dict[str, Any], now_ms: int) -> str:
    """Format a single recent order summary line for Section 3C."""
    lb = oinfo.get("last_buy") if isinstance(oinfo.get("last_buy"), dict) else {}
    ls = oinfo.get("last_sell") if isinstance(oinfo.get("last_sell"), dict) else {}
    lb_ts = normalize_timestamp_ms(lb.get("ts_ms"), default=0) or 0
    ls_ts = normalize_timestamp_ms(ls.get("ts_ms"), default=0) or 0
    lb_age_h = round((now_ms - lb_ts) / 3_600_000.0, 1) if lb_ts > 0 else None
    ls_age_h = round((now_ms - ls_ts) / 3_600_000.0, 1) if ls_ts > 0 else None
    avg_b_px = _round_opt_px(oinfo.get("active_avg_buy_px"), 6)
    avg_b_str = f" | Active Cycle Avg BUY={avg_b_px}" if avg_b_px else ""
    r_buys = oinfo.get("recent_buys") or []
    tranches_str = ""
    if len(r_buys) > 1:
        t_parts = [f"{b.get('qty')}@{_round_opt_px(b.get('price'), 6)}" for b in r_buys if isinstance(b, dict)]
        tranches_str = f" | Recent BUY Tranches=[{', '.join(t_parts)}]"

    lb_px = _round_opt_px(lb.get("price"), 6)
    ls_px = _round_opt_px(ls.get("price"), 6)
    lb_tag = lb.get("actor_tag") or format_actor_tag(lb.get("actor"), lb.get("module"), lb.get("rule"))
    ls_tag = ls.get("actor_tag") or format_actor_tag(ls.get("actor"), ls.get("module"), ls.get("rule"))
    lb_by = f", by={lb_tag}" if lb_tag else ""
    ls_by = f", by={ls_tag}" if ls_tag else ""
    lb_str = f"{lb.get('qty')} @ {lb_px} (age={lb_age_h}h{lb_by})" if lb_px is not None else "none"
    ls_str = f"{ls.get('qty')} @ {ls_px} (age={ls_age_h}h{ls_by})" if ls_px is not None else "none"
    last_act_tag = oinfo.get("last_action_actor_tag") or format_actor_tag(
        oinfo.get("last_action_actor"),
        oinfo.get("last_action_module"),
        oinfo.get("last_action_rule"),
    )
    last_act_sfx = f" (by={last_act_tag})" if last_act_tag else ""
    return (
        f"  * {sym} Last Action={oinfo.get('last_action_side')}{last_act_sfx}: "
        f"Last BUY={lb_str} | "
        f"Last SELL={ls_str}"
        f"{avg_b_str}{tranches_str}"
    )


def _format_portfolio_section(ps: Dict[str, Any], asset_scope: str) -> List[str]:
    """Format Section 3C (Spot Wallet Balances, Tradable Positions, Cycle Cost Anchors & Last Fills)."""
    lines = ["\n### 3C. SPOT WALLET BALANCES, TRADABLE POSITIONS, CYCLE COST ANCHORS & LAST FILLS:"]
    if ps.get("total_value_usdc") is not None:
        tot_usdc = _parse_finite_float(ps.get("total_value_usdc"), 0.0)
        lines.append(f"  * Total Binance Portfolio Value: ${tot_usdc:,.2f} USDC")
    if ps.get("venue_quote_cash"):
        cash_parts: List[str] = []
        for v in ps["venue_quote_cash"].values():
            if not isinstance(v, dict):
                continue
            part = f"{v.get('venue')} {v.get('asset')}=${_parse_finite_float(v.get('free'), 0.0):,.2f}"
            bd = v.get("breakdown")
            if isinstance(bd, dict) and (len(bd) > 1 or any(k != v.get("asset") for k in bd)):
                bd_str = ", ".join(f"{bk}=${_parse_finite_float(bv, 0.0):,.2f}" for bk, bv in bd.items())
                part += f" ({bd_str})"
            cash_parts.append(part)
        if cash_parts:
            lines.append(f"  * Available Free Quote Cash per Venue: {', '.join(cash_parts)}")
    if ps.get("tradable_positions"):
        lines.append("  * Live Tradable Spot Instruments (Eligible for Autonomous BUY / SELL):")
        for pos in ps["tradable_positions"]:
            lines.append(_format_tradable_position_line(pos))
    open_lim = (
        ps.get("open_limit_orders")
        if isinstance(ps.get("open_limit_orders"), list)
        else (ps.get("open_orders") if isinstance(ps.get("open_orders"), list) else [])
    )
    if open_lim:
        lines.append(
            f"  * Active Open Orders on Exchange (Locking Liquidity) — open_limit_orders: {open_lim}"
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
        xs = ps["kraken_xstocks"]
        lines.append(
            f"  * Kraken xStocks Pre-IPO Holdings (READ-ONLY on Kraken — NOT API-tradable; cross-venue SpaceX exposure for Trading212 {xs.get('t212_sibling')}): {xs}"
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
    mb_bench_3c = _extract_macro_benchmarks_line(
        ps.get("macro_benchmarks")
        or (
            ps["cross_market_context"].get("macro_benchmarks")
            if isinstance(ps.get("cross_market_context"), dict)
            else None
        )
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
    if ps.get("last_cycle_execution_status"):
        lines.append(
            f"  * Last Cycle Execution Status — last_cycle_execution_status: {ps['last_cycle_execution_status']}"
        )
    for st_key, st_label in (
        ("active_retry_queue", "Active Queued Retry Orders (order_retry_queue.jsonl)"),
        ("monitortrades_active_state", "Active MonitorTrades Ride/Stop State (mt_state.json)"),
        ("assetguardian_state", "Active AssetGuardian State (assetguardian_state.json)"),
        ("smart_stop_loss_state", "Active Smart Stop-Loss State (smart_stop_loss_state.json)"),
        ("recent_ai_intents", "Recent Autonomous Intents"),
    ):
        if ps.get(st_key):
            lines.append(f"  * {st_label}: {ps[st_key]}")
    return lines


def build_supervisor_prompt(
    sensors: Any,
    logs: Any,
    spec: Any = None,
    lookback_hours: float = 1.0,
    focus_symbols: Optional[List[str]] = None,
    *,
    min_usd: float = 10.0,
    max_usd: float = 2000.0,
    min_conf: float = 0.85,
    topic_key: Optional[str] = None,
    min_notional_usd: Optional[float] = None,
    max_notional_usd: Optional[float] = None,
    min_confidence: Optional[float] = None,
    **_extra: Any,
) -> str:
    """Build the topic-specific prompt for Gemini (auto_technical_repair or auto_*trade_executor)."""
    if spec is None:
        spec = topic_spec(topic_key or "auto_cryptotrade_executor")
    if min_notional_usd is not None:
        min_usd = float(min_notional_usd)
    if max_notional_usd is not None:
        max_usd = float(max_notional_usd)
    if min_confidence is not None:
        min_conf = float(min_confidence)

    is_pure_sre = bool(spec.can_self_heal and not spec.can_trade)
    context_items: List[str] = []
    if not is_pure_sre:
        context_items.extend(_format_macro_and_news_section(sensors))
    if spec.can_self_heal:
        context_items.extend(_format_sre_and_fleet_section(sensors, logs, lookback_hours))
    prices_and_trends = getattr(sensors, "prices_and_trends", None) or {}
    microstructure = getattr(sensors, "microstructure", None) or {}
    portfolio_summary = getattr(sensors, "portfolio_summary", None) or {}
    if not is_pure_sre and prices_and_trends:
        context_items.extend(_format_market_prices_section(prices_and_trends))
    if not is_pure_sre and microstructure:
        context_items.extend(_format_microstructure_section(microstructure, prices_and_trends))
    if not is_pure_sre and portfolio_summary:
        context_items.extend(
            _format_portfolio_section(portfolio_summary, getattr(spec, "asset_scope", "all"))
        )
    if not is_pure_sre and focus_symbols:
        raw_focus = [focus_symbols] if isinstance(focus_symbols, str) else focus_symbols
        clean_syms = [s.strip().upper() for s in raw_focus if isinstance(s, str) and s.strip()]
        if clean_syms:
            context_items.append(
                f"\n### 4. TACTICAL PRIORITY SYMBOLS: {', '.join(clean_syms)}\n"
                "- Pay heightened attention to these symbols alongside all other live tradable coins above."
            )

    context_str = "\n".join(context_items)
    tasks: List[str] = []
    schema: List[str] = []
    if spec.can_self_heal:
        sev_union = " | ".join(f'"{s}"' for s in DIAGNOSED_SEVERITIES)
        cat_union = " | ".join(f'"{c}"' for c in ALL_DIAGNOSED_CATEGORIES)
        act_union = " | ".join(f'"{a}"' for a in ALL_REPAIR_ACTIONS)
        cfg_list_str = ", ".join(AUDITED_CONFIG_FILES)
        tasks.extend([
            f"1. Audit functional system anomalies: fleet processes and modules (procs.conf), bot heartbeats, Linux OS resources (CPU load, RAM, disk), internet & VPN tunnel (wgpia0), configuration files ({cfg_list_str}), and persistent unresolved errors in logs: explain the technical root cause of any bot exceptions, fail-closed blocks, stale heartbeats, OS/network/VPN issues, or configuration errors.",
            "2. Determine whether hung/errored bots or modules require restart (RESTART_BOT / RESTART_MODULE), whether the entire fleet or VPN/DNS requires restart (RESTART_FLEET / RESTART_VPN / PIN_DNS), or whether a critical unrecoverable Linux OS / kernel / system-wide hang requires a full machine reboot (REBOOT_SYSTEM).",
            "3. If a reproducible code bug or configuration error exists in the workspace, formulate an exact search_block and replace_block in `code_remediation` to fix the code or configuration file.",
        ])
        schema.extend([
            '  "summary": "1-2 sentence executive technical assessment of functional system, Linux OS, internet/VPN, configurations, and fleet health"',
            f'  "diagnosed_issues": [{{"severity": {sev_union}, '
            f'"category": {cat_union}, '
            '"description": "Short explanation", "root_cause": "Detailed technical root cause"}]',
            f'  "venue_actions": [{{"action": {act_union}, '
            '"venue": "fleet" | "system" | "binance" | "kraken" | "hyperliquid" | "t212", "symbol": "bot_or_module_name or SYSTEM", '
            '"order_id": "optional order id", "reason": "Why this technical repair or restart/reboot action is needed"}]',
            '  "code_remediation": {"has_fix": false, "target_file": "", "rationale": "", '
            '"search_block": "", "replace_block": ""}',
        ])
    else:
        schema.append(
            '  "summary": "1-2 sentence executive assessment of multi-coin market posture and trading decisions"'
        )
    if spec.can_trade:
        is_equities = getattr(spec, "asset_scope", "all") == "equities"
        target_desc = (
            "ALL live tradable US equity shares on 't212' (SPCX_US_EQ, NVDA_US_EQ, RGNT_US_EQ during US Regular Trading Hours, cross-referencing Kraken read-only xStocks SPCXx.T exposure)"
            if is_equities
            else "ALL live tradable crypto spot coins (BTCUSDC, TAOUSDC, ARBUSDC on 'binance'; HYPEUSDC intensively on 'hyperliquid'; HYPEUSD on 'kraken', while TAOUSD/ADAUSD on Kraken serve as cross-venue reference telemetry)"
        )
        venue_schema = '"t212"' if is_equities else '"binance" | "kraken" | "hyperliquid"'
        ex_sym = '"NVDA_US_EQ"' if is_equities else '"BTCUSDC"'
        tasks.append(
            f"{len(tasks) + 1}. Formulate symmetric autonomous trading decisions across {target_desc} by synthesizing the "
            "entire MPTrade system telemetry as a whole:\n"
            "   - FULL ACCESS TO EVERYTHING IN `cachedb/` AS A DECISION INFORMATION SOURCE: The telemetry above is a compact "
            "synthesized dashboard (zero raw file dumps); whenever you want deeper detail, you have direct, unrestricted read access "
            "to the entire `cachedb/` (`cachedb/prices/`, `cachedb/trends/`, `cachedb/trends/24trend/`, `cachedb/telemetry/`, "
            "`cachedb/macro/`, `cachedb/orders/`, `cachedb/state/`), `logger/execution_audit/`, `logs/`, and `providers.market_api` "
            "to inspect any file or query any venue on your own.\n"
            "   - BUSINESS OBJECTIVE & FULL AUTONOMOUS JUDGMENT (ZERO PROFIT OR STRATEGY GUARDS): Your objective "
            "is to maximize net portfolio value and risk-adjusted returns (net of venue fees) across all platforms while "
            "protecting capital. ALL deterministic order_guard gates (G_PROF profit reference, GO1 daily cap/cooldown, "
            "GT_* trend deferrals, G_QLIMIT* weight caps, GI_* microstructure, and GLLM* secondary vetoes) are BYPASSED "
            "symmetrically on both BUY and SELL as binding constraints. Deterministic bots (`tradeall.py`, `monitortrades.py`, `assetguardian.py`, "
            "`rtrade.py`, `kraken/bot.py`, `hyperliquid/bot.py`, `212trading/bot.py`) and rigid code filters/configs "
            "(`order_guard.py`, `order_guard.conf`, `instruments.conf`, `intelligence/`) continue working in parallel in the "
            "system and can be consulted by you on demand as advisory indicators and reference guidance—you have full quantitative authority "
            "to validate, adapt, or override them according to live market structure, with ZERO obligation to consult or follow them.\n"
            "   - SUPERVISORY AUTHORITY TO AUDIT & REPAIR / REVERSE UNWANTED BOT OR FLEET TRADES (`by=BOT` vs `by=AI`): "
            "Section 3C explicitly tags every recent fill (`Last Action`, `Last BUY`, `Last SELL`) and every resting order in "
            "`open_limit_orders` with `by=BOT` / `\"actor\": \"BOT\"` (executed by deterministic fleet bots) vs `by=AI` / `\"actor\": \"AI\"` "
            "(executed by you). Actively audit any interim `by=BOT` transaction or resting order executed between your cycles: if a bot or "
            "fleet process made a trade that you disagree with based on your holistic multi-horizon/microstructure/macro view, you have full "
            "authority and duty to immediately repair or reverse it:\n"
            "     1) If a bot SOLD prematurely or when it should NOT have (e.g. panic stop-loss on a liquidity wick, premature micro-take-profit, "
            "or selling at support when absorption/trend structure warrants holding), immediately submit a corrective `BUY` (`MARKET` or `LIMIT`) "
            "to buy back and restore the position.\n"
            "     2) If a bot BOUGHT prematurely or when it should NOT have (e.g. catching a falling knife without bid absorption, buying into "
            "resistance/breakdown, or over-allocating when cash should be preserved), immediately submit a corrective `SELL` (`MARKET` or `LIMIT`) "
            "to liquidate or reduce the unwanted position (all profit/cooldown guards `G_PROF`/`GO1` are bypassed so you can exit immediately "
            "even at a scratch or small loss to repair a bad bot entry).\n"
            "     3) If a bot placed a resting order in `open_limit_orders` (`\"actor\": \"BOT\"`) that you disagree with or that locks wallet "
            "liquidity needed for your repair trade, set `\"replace_open_orders\": true` (or `\"cancel_bot_open_orders\": true`) to cancel "
            "the bot's resting order and unlock the balance.\n"
            "   - UNBIASED QUANTITATIVE FREEDOM & DYNAMIC POSITION SIZING: No rigid trading rules, fixed percentage thresholds, "
            "or directional biases are imposed on you—judge 100% freely on your own WHEN, WHAT, HOW MUCH, on WHICH platform "
            "(`binance`, `kraken`, `hyperliquid`, `t212`), and WHY to `BUY`, `SELL`, or `HOLD`. Do NOT restrict yourself to tiny "
            "fixed slices when conviction or profitability is clear: when a position is clearly in profit (e.g. above "
            "`active_avg_buy_px` / `last_buy_px` or after a strong upward swing), you are completely free to sell a large "
            f"quantity or the entire profitable position (`50%–100%` of `value_usd`, up to {max_usd:g} USD) at once.\n"
            "   - CONTRARIAN DIP ACCUMULATION (SCALE-IN): Elevated macro/geopolitical risk scores, extreme fear, or compressed "
            "breadth (`advance_ratio < 0.20`) signal high-volatility regimes where capitulation dips and liquidation cascades "
            "create the highest-EV mean-reversion entries. Whenever microstructure confirms strong absorption at support "
            "(e.g. a large `bid_wall` / positive `ob_imbalance`, `taker_ratio > 1.3`, seller exhaustion at 24h/7d/`VAL` support, "
            "and ample dry powder), execute contrarian dip accumulation (scale-in) in controlled small/medium tranches ($200–$500 USD).\n"
            "   - DUAL EXECUTION MODE — IMMEDIATE 'MARKET' ORDERS OR PASSIVE 'LIMIT' ORDERS IN THE ORDERBOOK (`open_limit_orders` & `last_cycle_execution_status`): "
            "Because intra-cycle wicks between 10-minute evaluations can touch key support or resistance levels and bounce before the next cycle, "
            "you may submit either immediate market orders (`\"order_type\": \"MARKET\"`, `\"limit_price\": null`) OR passive resting "
            "limit orders (`\"order_type\": \"LIMIT\"`, `\"limit_price\": <float>`, `\"replace_open_orders\": true`) in the exchange "
            "orderbook (e.g. resting a passive `BUY LIMIT` at a major `bid_wall` / support level below spot with $200–$500 tranches, "
            "or a passive `SELL LIMIT` at an `ask_wall` / take-profit target above spot). Section 3C explicitly reports your active resting orders "
            "(`open_limit_orders`, including `filled_qty`, `remaining_qty`, `status`) and mathematical fill confirmation from the previous cycle "
            "(`last_cycle_execution_status` with `status`, `fill_px`, `ref_px`, `filled_qty`, `slippage_bps`), plus the 1-line US Equities/Futures/Macro "
            "correlation index (`SPY / QQQ / ES=F / NQ=F / DXY / US10Y` 24h/1h deltas). Setting `\"replace_open_orders\": true` "
            "automatically cancels any prior unfilled AI limit order on the same symbol and side before placing the updated order; "
            "returning `'HOLD'` leaves existing resting open orders untouched in the book.\n"
            "   - VOLUME PROFILE (`vol_profile` IN SECTION 3A), LIQUIDATION CLUSTERS (`liq_clusters` IN SECTION 3B) & VENUE FEE/LIMITS "
            "(`venue_fee_per_leg`, `maker`/`taker`, `spread`, `limits` IN SECTION 3C): Synthesize Session Volume Profile "
            "(`POC`, 70% Value Area `[VAL, VAH]`, `va_pos`, `regime` in `ABOVE_VAH` | `INSIDE_VA_UPPER` | `INSIDE_VA_LOWER` | `BELOW_VAL`) "
            "to distinguish genuine breakouts/breakdowns from value-area retests or mean-reversion rotations toward `POC`/`VAL`/`VAH`; "
            "use symmetric `LONG` (downside `long_liq`/`deep_long`) and `SHORT` (upside `short_liq`/`deep_short`) liquidation heatmap "
            "clusters and `magnet` (`SHORT_SQUEEZE_MAGNET_UP` | `LONG_SWEEP_MAGNET_DOWN` | `BALANCED_LIQ_MAGNET`) to identify where "
            "leveraged liquidity sweeps sit on both sides of spot; and weigh each venue's active `tier`, `maker` vs `taker` fee bps, "
            "live bid-ask `spread` bps, and `limits(min_notional, min_qty, step, tick, max_buy, max_sell)` across `binance` (10/10 bps), "
            "`kraken` (16/26 bps), `hyperliquid` (4/7 bps), and `t212` (15/15 bps) when choosing between aggressive `MARKET` and passive "
            "`LIMIT` execution and sizing `suggested_notional_usd` / `limit_price` cleanly within `[min_notional, max_buy/max_sell]`.\n"
            f"   - Emit one entry per actionable instrument in `market_trading_decisions` with `confidence >= {min_conf:g}` "
            f"and `suggested_notional_usd` sized between {min_usd:g} and {max_usd:g} USD (the gateway automatically "
            "clamps notional strictly to available free wallet balance with zero artificial cooldowns or dedup locks). "
            "If no instrument warrants action right now, return a single 'HOLD' entry."
        )
        schema.append(
            f'  "market_trading_decisions": [{{"decision": "BUY" | "SELL" | "HOLD", "symbol": {ex_sym}, '
            f'"venue": {venue_schema}, "order_type": "MARKET" | "LIMIT", "limit_price": null, "replace_open_orders": false, '
            f'"confidence": 0.90, "suggested_notional_usd": 1000.0, "urgency": "NORMAL" | "HIGH", "thesis": "..."}}]'
        )

    if spec.can_self_heal:
        role_intro = (
            "You are the Lead SRE and Autonomous Technical Repair Supervisor for MPTrade.\n"
            "Your STRICT mandate is solely to detect functional system anomalies (bots, fleet modules, heartbeats), "
            "Linux OS resource issues, internet & VPN (wgpia0) connectivity failures, configuration errors, and "
            "persistent unresolved errors in logs, and to remediate them by patching code or configurations and "
            "restarting/rebooting bots, modules, the trading fleet, VPN/DNS, or the entire Linux system.\n"
            "Do NOT analyze or report on market prices, trends/slopes, macro sentiment, portfolio balances, or active trading positions.\n\n"
            f"{context_str}\n\n"
            "Perform an executive technical diagnostic and self-healing audit:\n" + "\n".join(tasks) + "\n\n"
        )
        tool_rule = (
            "- If you diagnose a reproducible code bug or configuration error and need to verify exact lines before proposing `code_remediation`, "
            "you MAY read the target file in the workspace so `search_block` matches verbatim.\n"
        )
        report_format_rule = (
            "- Format your response in TWO parts so this thread stays cleanly readable in the Web UI:\n"
            "  1) A concise, well-structured Markdown report in **Romanian** strictly covering technical & SRE health "
            "(bullet points for: Starea Flotei, Boților, Sistemului Linux & VPN (wgpia0) & Loguri, "
            "Conectivitate Internet & Audit Configurații, Erori Persistente Nerezolvate, și "
            "Acțiuni de Remediere Cod/Config & Reboot Boți/Module/Flotă/Sistem). "
            "NU analiza și NU menționa prețuri de piață, pante, portofoliu, solduri sau poziții active!\n"
        )
    else:
        role_intro = (
            "You are the Quantitative Autonomous Portfolio Manager and Trade Executor for MPTrade.\n"
            "Analyze the complete market telemetry, macro/geopolitical sensors, microstructure, spot wallet balances, and cycle states:\n\n"
            f"{context_str}\n\n"
            "Evaluate the market context and formulate autonomous BUY/SELL trading decisions:\n" + "\n".join(tasks) + "\n\n"
        )
        tool_rule = (
            "- You have full, unrestricted access to everything in `cachedb/` (`cachedb/prices/`, `cachedb/trends/`, `cachedb/trends/24trend/`, "
            "`cachedb/telemetry/`, `cachedb/macro/`, `cachedb/orders/`, `cachedb/state/`) as a direct source of information for your trading decisions, "
            "as well as full freedom to query live market APIs (`providers.market_api`, `binance_api`, `kraken`, `hyperliquid`, `212trading`) "
            "and read trade/execution logs (`logs/`, `logger/execution_audit/`) whenever you want deeper verification or to answer user questions.\n"
            "- Deterministic bots (`tradeall.py`, `monitortrades.py`, `assetguardian.py`, `rtrade.py`, venue bots) and rigid code filters/guards "
            "(`order_guard.py`, `order_guard.conf`, `instruments.conf`, `intelligence/`) run in parallel and may be inspected on demand if you wish, "
            "but consulting or following them is never required; if any bot executed an interim BUY or SELL (`by=BOT`) that you disagree with, "
            "you are fully empowered to immediately repair/reverse it with a counter-trade (`BUY` back what a bot wrongly sold, or `SELL` what a bot wrongly bought).\n"
            "- Keep this thread focused on market analysis, portfolio management, and autonomous BUY/SELL execution across venues.\n"
        )
        report_format_rule = (
            "- Format your response in TWO parts so this thread stays cleanly readable in the Web UI:\n"
            "  1) A concise, well-structured Markdown report in **Romanian** (bullet points for: "
            "Piața Live & Pante (Toate Monedele), Portofoliu, Solduri & Poziții Active, și Decizia AI de Tranzacționare / Răspuns la intervențiile din chat).\n"
        )

    return (
        role_intro
        + "INTERACTIVE THREAD RULES:\n"
        "- If the user wrote any tactical instructions, price targets, overrides, or questions earlier in this "
        "conversation thread, honor them with highest priority and address them directly.\n"
        + tool_rule
        + report_format_rule
        + "  2) At the very end, a single fenced ```json ... ``` block matching this exact schema:\n"
        "```json\n{\n" + ",\n".join(schema) + "\n}\n```"
    )

