"""Pure formatting helpers for Autonomous AI Reconciler Gemini prompts.

Separates prompt presentation (Sections 1, 2, 3A, 3B, 3C) and symbol price
resolution from telemetry harvesting and order execution in
`autonomous_ai_reconciler.py`.

Intentional prompt compression & cycle-boundary invariants (BY DESIGN -- DO NOT REVERT):
- Section numbering (`### 1.` -> `### 3.`): Sections `2/2B/2C` (SRE/logs/VPN) are emitted only for `can_self_heal` topics and omitted on trading topics.
- `bottom_6h(...)` (Section 3): Conditional pattern detector emitted only when an active 6h bottom structure forms (`pos_6h <= 70%` with meaningful drawdown).
- `open_lifo_lots` (Section 3C): Capped to the latest 4 lots (`<=14d`) for top-of-stack LIFO harvesting and omitted when `len(open_lots) <= 1` (already covered by `avg_cost`/`last_buy_px`).
- `last_cycle_execution_status` (Section 3C): Emitted when recent (`<=60m`) executions/fills/rejections or newly resting (`<=30m`) orders exist; omitted when idle or when all `OPEN_RESTING`/`PARTIALLY_FILLED` orders are `>30m` old and already displayed in `open_limit_orders`.
- `equity_bellwethers` on `scope="crypto"` (Section 3C): Micro-cap `RGNT_US_EQ` is omitted when major US tech/AI bellwethers (`NVDA_US_EQ`, `SPCX_US_EQ`) are present, and `t212_cycle` is omitted on `scope="crypto"` because `SPCX_US_EQ`'s cycle is already shown in `kraken_xstocks.t212_sibling_cycle` and `auto_cryptotrade_executor` cannot trade T212 shares (full cycles remain on `scope="equities"` and `scope="all"`).
- `last_exit_kind` & `realized_pnl` across cycle boundaries (Section 3C): A prior cycle's exit label (e.g. `SL`) is never reclassified against a newer cycle's re-entry cost (`last_sell_ts < entry_ts`), and multi-cycle cumulative PnL (`cum_cycles`) is preserved alongside active `subcycle` PnL.
- `above_vah` / `below_val` (Section 3) & `d_tr` (Section 3B): Value-Area breakout hold duration is emitted conditionally only when price is outside `[VAL, VAH]`, and `d_tr` (`taker_ratio_delta`) is emitted only when `|d_tr| >= 0.05` so quiet cycles incur zero prompt token bloat.
"""

import math
import re
from typing import Any, Dict, List, Optional

from botcore import (
    MAX_FUTURE_TIMESTAMP_SKEW_MS,
    _parse_bool_flag,
    first_epoch_sec,
    fmt_signed,
    is_us_rth_open,
    normalize_epoch_sec,
    normalize_timestamp_ms,
    parse_symbol_list,
    resolve_now_ts,
    safe_float,
    safe_int,
    safe_pos_float,
)
from cachepaths import canonical_base_asset, clean_symbol_key
from intelligence.macro.calendar_ssot import resolve_macro_economic_calendar
from intelligence.macro_sentiment_ssot import (
    TELEMETRY_SOURCE_STALE_INTERVAL_MULT,
    format_age_sec_compact,
    format_interval_sec_label,
    resolve_derived_telemetry_intervals,
)
from intelligence.sentiment.evaluators.long_horizon_evaluator import (
    format_long_horizon_section_lines,
)
from intelligence.sentiment.evaluators.macro_telemetry_evaluator import (
    compute_futures_premarket_lead,
    format_bottom_structure_6h_parts,
    format_compact_usd,
    format_equity_macro_regime_line,
    format_etf_flows_summary_line,
    format_macro_benchmarks_summary_line,
    resolve_futures_since_cash_close_pct,
)
from intelligence.sentiment.evaluators.microstructure_evaluator import (
    format_agg_spot_cvd_part,
    format_book_window_cap_tag,
    format_flow_horizon_parts,
    format_realized_liq_sources_tag,
    format_spot_cvd_source_tag,
    format_spot_volume_rvol_parts,
)
from intelligence.sentiment.evaluators.portfolio_evaluator import (
    compute_open_lots_vwap,
    format_recent_fill_subcycle_parts,
    is_last_buy_lot_harvested,
    same_anchor_px as _same_anchor_px,
)
from intelligence.sentiment.evaluators.price_series_evaluator import (
    build_vp_position_dict,
    compute_horizon_range_dict,
    format_atr_parts,
    format_equity_quote_meta_suffix,
    format_kinematic_parts,
    format_session_extremes_parts,
    format_session_vwap_summary,
    format_slope_val as _fmt_slope_val,
    format_va_hold_suffix,
    round_opt_px,
)
from intelligence.sentiment.evaluators.prompt_view_evaluator import (
    evaluate_market_prices_prompt_view,
    evaluate_microstructure_prompt_view,
    evaluate_portfolio_prompt_view,
    evaluate_tradable_position_prompt_view,
    resolve_scope_presentation,
)
from instrument_registry import (
    is_equity_or_tokenized_symbol,
    is_equity_symbol,
    observe_only_registry_symbols,
    resolve_symbol_identity,
)
from providers.ai_survival_guard import (
    format_ai_calibration_feedback_line,
    format_ai_decision_scorecard_line,
    format_signed_usd,
)
from providers.intent_telemetry import (
    compute_order_plan_rr,
    match_open_order_resting_plan,
)
from orchestratorOS.supervisor.crypto_executor_delta_gate import _extract_open_order_filled_qty
from providers.order_id_context import format_actor_tag


def _resolve_sensor_age_min(d: Any, now_ts: Optional[float] = None) -> Optional[float]:
    """Resolve sensor age in minutes anchored to ``now_ts`` (matching Section 0) when raw epoch ``ts`` is available."""
    if not isinstance(d, dict):
        return None
    ref_now = safe_pos_float(now_ts, default=None)
    if ref_now is not None:
        ts_v = first_epoch_sec(d.get("ts"), d.get("timestamp"), default=None)
        if ts_v is not None and ts_v > 0.0:
            if ts_v > ref_now + 60.0:
                return None
            if ref_now >= ts_v - 5.0:
                return round(max(0.0, (ref_now - ts_v) / 60.0), 1)
    return safe_float(d.get("age_min"), default=None)


def _age_sfx(d: Dict[str, Any], parens: bool = False, now_ts: Optional[float] = None) -> str:
    eff_age = _resolve_sensor_age_min(d, now_ts=now_ts)
    if eff_age is None:
        return ""
    pub = str(d.get("published_utc") or "").strip()
    inner = f"{pub}, age={eff_age}m" if pub else f"age={eff_age}m"
    return f" ({inner})" if parens else f", {inner}"


def _format_lifo_rem_avg_tag(
    lots: Any,
    held_qty: float,
    ref_avg_px: Optional[float],
    spot_px: Optional[float],
    *,
    label: str = "lifo_avg",
) -> str:
    vwap, _, _ = compute_open_lots_vwap(
        lots,
        held_qty=held_qty,
        clip_to_held=True,
        min_lot_notional=1.0,
        ref_avg_px=ref_avg_px,
        min_rel_diff=0.0025,
        min_coverage_ratio=0.95,
    )
    if vwap is None:
        return ""
    sfx = (
        f" (pnl_vs_lifo={fmt_signed(((spot_px - vwap) / vwap) * 100.0)}%)"
        if spot_px is not None and spot_px > 0.0 and vwap > 0.0
        else ""
    )
    return f"{label}={vwap}{sfx}"


def _format_venue_cycle_line(venue: str, sym: str, cyc: Dict[str, Any]) -> str:
    """Format a normalized venue cycle anchor entry for Section 3C of the AI prompt."""
    extra_parts: List[str] = []
    dir_norm = str(cyc.get("direction") or "").strip().upper()
    dir_sign = -1.0 if dir_norm == "SHORT" else 1.0
    if dir_norm == "SHORT":
        extra_parts.append("direction=SHORT")
    lb_px_f = safe_float(cyc.get("last_buy_price"), None)
    avg_c_f = safe_float(cyc.get("avg_cost_px"), None)
    spot_px_f = safe_float(cyc.get("_spot_px"), None)
    if cyc.get("pos_peak") is not None and not _same_anchor_px(cyc.get("pos_peak"), cyc.get("avg_cost_px")):
        pk_dd = safe_float(cyc.get("dd_from_peak_pct"), None)
        pk_dd_val = (pk_dd + 0.0) if pk_dd is not None else cyc.get("dd_from_peak_pct")
        pk_sfx = f" (dd_from_peak={pk_dd_val}%)" if pk_dd_val is not None else ""
        extra_parts.append(f"peak_px={cyc.get('pos_peak')}{pk_sfx}")
    if (
        isinstance(open_lots_c_init := cyc.get("_open_lots"), list)
        and open_lots_c_init
        and avg_c_f is not None
        and avg_c_f > 0.0
    ):
        if lifo_c_tag := _format_lifo_rem_avg_tag(
            open_lots_c_init,
            safe_float(cyc.get("qty"), 0.0) or 0.0,
            avg_c_f,
            spot_px_f,
            label="lifo_avg",
        ):
            extra_parts.append(lifo_c_tag)
    if cyc.get("realized_pnl_usd") is not None:
        rp_v = cyc.get("realized_pnl_usd")
        rp_num = safe_float(rp_v, None)
        rp_f = rp_num if rp_num is not None else 0.0
        rp_s = format_signed_usd(rp_num, decimals=2) if rp_num is not None else f"${rp_v}"
        sub_rp = safe_float(cyc.get("subcycle_realized_pnl_usd"), None)
        cum_rp = safe_float(cyc.get("cum_realized_pnl_usd"), None)
        if sub_rp is not None and abs(sub_rp) < 0.01 and abs(rp_f) >= 0.01:
            extra_parts.append(f"realized_pnl={rp_s} (subcycle=$0.00)")
        elif cum_rp is not None and abs(rp_f) >= 0.01 and abs(cum_rp - rp_f) >= 0.01:
            extra_parts.append(f"realized_pnl={rp_s} (cum_cycles={format_signed_usd(cum_rp, decimals=2)})")
        else:
            extra_parts.append(f"realized_pnl={rp_s}")
    for lbl, k in (("tp_sold", "tp_sold_levels"), ("resting_orders", "resting_orders")):
        if cyc.get(k):
            extra_parts.append(f"{lbl}={cyc.get(k)}")
    extra_sfx = f", {', '.join(extra_parts)}" if extra_parts else ""
    exit_kind = cyc.get("last_exit_kind")
    exit_kind_sfx = f" ({exit_kind})" if exit_kind else ""
    dd_val = cyc.get("pnl_vs_avg_pct") if cyc.get("pnl_vs_avg_pct") is not None else cyc.get("dd_pct")
    dd_f = safe_float(dd_val, None)
    dd_field = (
        f"pnl_vs_cost={fmt_signed(dd_f)}%"
        if dd_f is not None
        else "pnl_vs_cost=N/A"
    )
    lb_pnl_sfx = ""
    if (
        lb_px_f is not None
        and lb_px_f > 0.0
        and avg_c_f is not None
        and avg_c_f > 0.0
        and (dd_f is not None or (spot_px_f is not None and spot_px_f > 0.0))
        and not _same_anchor_px(lb_px_f, avg_c_f)
    ):
        if spot_px_f is not None and spot_px_f > 0.0:
            est_spot = spot_px_f
        else:
            est_spot = avg_c_f * (1.0 + dir_sign * dd_f / 100.0)
            if avg_c_f >= 100.0 and abs(est_spot - round(est_spot, 2)) <= avg_c_f * 0.00005:
                est_spot = round(est_spot, 2)
        lb_pnl = dir_sign * ((est_spot - lb_px_f) / lb_px_f) * 100.0
        if is_last_buy_lot_harvested(lb_px_f, cyc.get("_open_lots")):
            lb_pnl_sfx = f" (chg_vs_last_buy={fmt_signed(lb_pnl)}%, lot_harvested)"
        else:
            lb_pnl_sfx = f" (pnl_vs_last_buy={fmt_signed(lb_pnl)}%)"
    entry_px_part = f"entry_px={cyc.get('entry_price')}, " if cyc.get("entry_price") is not None else ""
    return (
        f"    - [{venue}] {sym}: qty={_fmt_plain_float(cyc.get('qty'))}, cost=${cyc.get('cost_usd')}, "
        f"avg_cost={cyc.get('avg_cost_px')}, {entry_px_part}"
        f"last_buy={cyc.get('last_buy_price')}{lb_pnl_sfx}, last_sell={cyc.get('last_sell_price')}"
        f"{exit_kind_sfx}, dca_buys={cyc.get('dca_buys')}, {dd_field}{extra_sfx}"
    )


def _format_horizon_range(pinfo: Dict[str, Any], px_f: float, horizon: str) -> str:
    """Format `[low, high] pos_<horizon>=X% (dd_high=..., bounce_low=..., span=...)` for 24h or 7d."""
    lo, hi = pinfo.get(f"low_{horizon}"), pinfo.get(f"high_{horizon}")
    lo_f, hi_f = safe_float(lo, 0.0), safe_float(hi, 0.0)
    pos_sfx = ""
    if px_f > 0 and lo_f > 0 and hi_f > lo_f:
        fb_rng = compute_horizon_range_dict(px_f, lo_f, hi_f, horizon)
        pos_v, dd_v, bnc_v, spn_v = (
            pinfo[k] if pinfo.get(k) is not None else fb_rng.get(k)
            for k in (f"pos_{horizon}_pct", f"dd_{horizon}_high_pct", f"bounce_{horizon}_low_pct", f"span_{horizon}_pct")
        )
        pos_sfx = (
            f" pos_{horizon}={pos_v}% (dd_high={fmt_signed(dd_v)}%, "
            f"bounce_low={fmt_signed(bnc_v)}%, span={spn_v}%)"
        )
    return f"{horizon}_range=[{lo}, {hi}]{pos_sfx}"


def _extract_macro_benchmarks_line(
    mb_raw: Any,
    *,
    include_crypto: bool = True,
    now_ts: Optional[float] = None,
) -> str:
    """Extract or synthesize the 1-line SPY / QQQ / ES=F / NQ=F / DXY / US10Y benchmark summary."""
    if not isinstance(mb_raw, dict) or not mb_raw:
        return ""
    if not include_crypto:
        summary_s = format_macro_benchmarks_summary_line(mb_raw, include_crypto=False)
    else:
        summary_s = str(mb_raw.get("summary_line") or "").strip() or format_macro_benchmarks_summary_line(mb_raw)
    return f"{summary_s}{_age_sfx(mb_raw, parens=True, now_ts=now_ts)}" if summary_s else ""


def _fmt_liq_usd(val: Any, *, signed: bool = False) -> str:
    return format_compact_usd(val, signed=signed)


def _fmt_signed_usd(val: Any) -> str:
    return format_compact_usd(val, signed=True)


def _fmt_plain_float(val: Any, decimals: int = 8, *, compact: bool = False) -> str:
    """Format finite numbers without scientific notation (e.g. 6e-05 -> 0.00006, 1.5e+06 -> 1500000)."""
    if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val):
        abs_v = abs(val)
        if (0.0 < abs_v < 1e-4) or abs_v >= 1e6:
            return f"{val:.{decimals}f}".rstrip("0").rstrip(".") or "0"
        if compact:
            return f"{val:g}"
    return str(val)


def _fmt_px(px_f: float, raw_px: Any = None) -> str:
    if px_f >= 1.0:
        if px_f < 1000.0 and abs(px_f - round(px_f, 2)) >= 1e-5:
            return f"${px_f:,.4f}".rstrip("0").rstrip(".")
        return f"${px_f:,.2f}"
    if px_f > 0:
        return f"${_fmt_plain_float(round(px_f, 8), 8)}" if px_f < 1e-4 else f"${px_f:.6g}"
    if raw_px is not None and not isinstance(raw_px, bool):
        return f"${raw_px}"
    return "N/A"


_SA_FLOW_POSTURE_RE = re.compile(
    r"(-old\s+)(?:(?:ACCUMULATE|HOLD|DEFENSIVE|TRIM_PROFITS|BULLISH|BEARISH|NEUTRAL|CAUTION)(?:/(?:ACCUMULATE|HOLD|DEFENSIVE|TRIM_PROFITS|BULLISH|BEARISH|NEUTRAL|CAUTION))?)\b",
    re.IGNORECASE,
)
_SA_FLOW_AGE_RE = re.compile(r"(vs\s+)\d+(?:\.\d+)?(m-old\b)")
_SA_SUMMARY_DIRECTIVE_RE = re.compile(
    r"(?:,\s*(?:so\s+|therefore\s+|thus\s+)?|\s*)\b(?:pausing|shifting|maintaining|staying|switching|recommending)\b[^.]*\b(?:HOLD|ACCUMULATE|DEFENSIVE|TRIM_PROFITS)\b[^.]*\.?",
    re.IGNORECASE,
)
_GT_SUMMARY_DIRECTIVE_RE = re.compile(
    r",?\s*(?:so|therefore|thus)\s+(?:maintain|apply|use|keep|enforce)\s+[^.]*(?:downscale|brake|veto|halt|exposure)[^.]*\.?",
    re.IGNORECASE,
)


def _format_live_macro_prints(raw_prints: Any) -> str:
    """Format live_macro_prints dicts cleanly without raw Python dict reprs."""
    items = raw_prints if isinstance(raw_prints, list) else ([raw_prints] if raw_prints is not None else [])
    formatted: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            formatted.append(str(item))
            continue
        ev = str(item.get("event") or "").strip()
        act = item.get("actual")
        cons = item.get("consensus")
        if ev and act is not None and cons is not None:
            meta: List[str] = []
            if item.get("prior") is not None:
                meta.append(f"prior={item['prior']}")
            if item.get("surprise") is not None:
                meta.append(f"surprise={item['surprise']}")
            if (z_v := safe_float(item.get("surprise_zscore"), default=None)) is not None:
                meta.append(f"z={fmt_signed(z_v)}")
            if st := str(item.get("stance") or "").strip():
                meta.append(st)
            meta_s = f" ({', '.join(meta)})" if meta else ""
            formatted.append(f"{ev}: act={act} vs cons={cons}{meta_s}")
        else:
            formatted.append(_format_compact_kv_value(item))
    return f"[{'; '.join(formatted)}]"


def _format_macro_calendar_line(mc: Any) -> Optional[str]:
    """Format the Section 1 Macro Economic Calendar & Session Window bullet line."""
    if not (isinstance(mc, dict) and mc.get("next_high_impact_event")):
        return None
    if "next_event_is_high_impact" in mc:
        is_confirmed_macro = bool(_parse_bool_flag(mc.get("next_event_is_high_impact"), False))
    elif "macro_release_confirmed" in mc or "fomc_today" in mc:
        is_confirmed_macro = bool(
            _parse_bool_flag(mc.get("macro_release_confirmed"), False)
            or _parse_bool_flag(mc.get("fomc_today"), False)
        )
    else:
        is_confirmed_macro = True
    ev_lbl = "next_high_impact_event" if is_confirmed_macro else "next_routine_session_window"
    mc_parts = [
        f'session="{mc.get("session_label")}"',
        f'{ev_lbl}="{mc.get("next_high_impact_event")}"',
    ]
    if mc.get("us_market_holiday"):
        mc_parts.append(f'us_market_holiday="{mc.get("us_market_holiday")} (US markets closed)"')
    elif mc.get("us_market_early_close"):
        mc_parts.append(f'us_market_early_close="{mc.get("us_market_early_close")} (NYSE 13:00 ET early close)"')
    elif mc.get("upcoming_us_market_holiday"):
        mc_parts.append(f'next_us_market_holiday="{mc.get("upcoming_us_market_holiday")} tomorrow (US markets closed)"')
    if _parse_bool_flag(mc.get("active_release_window"), False) and mc.get("next_scheduled_event"):
        mc_parts.append(f'next_scheduled="{mc.get("next_scheduled_event")}"')
    if mc.get("fomc_today") is not None:
        mc_parts.append(f'fomc_today={bool(_parse_bool_flag(mc.get("fomc_today"), False))}')
    if mc.get("next_fomc_date"):
        mc_parts.append(f'next_fomc_date="{mc.get("next_fomc_date")}"')
    if mc.get("upcoming_windows"):
        up_wins = mc.get("upcoming_windows")
        next_ev_s = str(mc.get("next_high_impact_event") or "").strip()
        if isinstance(up_wins, list) and up_wins and str(up_wins[0] or "").strip() == next_ev_s:
            up_wins = up_wins[1:]
        if up_wins:
            has_macro_in_upcoming = isinstance(up_wins, list) and any(
                isinstance(w, str) and ("[HIGH]" in w or "[MEDIUM]" in w)
                for w in up_wins
            )
            up_lbl = "upcoming" if (is_confirmed_macro or has_macro_in_upcoming) else "routine_session_transitions"
            mc_parts.append(f'{up_lbl}={up_wins}')
    if _parse_bool_flag(mc.get("pre_event_risk_window"), False) and mc.get("pre_event_risk_reason"):
        mc_parts.append(f'PRE_EVENT_RISK="{mc.get("pre_event_risk_reason")}"')
    if isinstance(conf_today := mc.get("confirmed_events_today"), list) and conf_today:
        ev_sums = [
            str(ev.get("summary")).strip()
            for ev in conf_today[:5]
            if isinstance(ev, dict) and str(ev.get("summary") or "").strip()
        ]
        if ev_sums:
            mc_parts.append(f"confirmed_macro_today={ev_sums}")
    if isinstance(er_today := mc.get("earnings_today"), list) and er_today:
        er_sums = [
            str(er.get("summary")).strip()
            for er in er_today[:5]
            if isinstance(er, dict) and str(er.get("summary") or "").strip()
        ]
        if er_sums:
            mc_parts.append(f"earnings_today={er_sums}")
    if isinstance(up_er := mc.get("upcoming_earnings"), list) and up_er:
        up_er_sums = [
            f"{er.get('date_iso')} {er.get('summary')}".strip()
            for er in up_er[:4]
            if isinstance(er, dict) and str(er.get("summary") or "").strip()
        ]
        if up_er_sums:
            mc_parts.append(f"upcoming_earnings={up_er_sums}")
    if mc.get("detected_macro_headlines"):
        mc_parts.append(f'live_macro_headlines={mc.get("detected_macro_headlines")}')
    if mc.get("live_macro_prints"):
        mc_parts.append(f'live_macro_prints={_format_live_macro_prints(mc.get("live_macro_prints"))}')
    return f"- Macro Economic Calendar & Session Window: {' | '.join(mc_parts)}"


def _resolve_macro_sensor(
    sensors: Any,
    macro: Dict[str, Any],
    attr: str,
    *,
    fallback_key: Optional[str] = None,
    ps: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve a macro sensor dictionary from ``sensors.<attr>``, ``macro_and_sentiment``, or ``portfolio_summary``."""
    val = (
        getattr(sensors, attr, None)
        or macro.get(fallback_key or attr)
        or (ps.get(attr) if isinstance(ps, dict) else None)
    )
    return val if isinstance(val, dict) else {}


def _format_macro_and_news_section(
    sensors: Any,
    asset_scope: Any = "all",
    *,
    now_ts: Optional[float] = None,
    delta_plan: Optional[Any] = None,
    include_cross_market: bool = False,
) -> List[str]:
    """Format Section 2 (Macro, Regime, Cross-Market & Live Breaking News) for the AI prompt."""
    scope_spec = resolve_scope_presentation(asset_scope)
    scope_s = scope_spec.scope_key
    is_equities_scope = scope_spec.is_equities
    ref_now = (
        safe_pos_float(now_ts, default=None)
        or safe_pos_float(getattr(sensors, "evaluated_at_ts", None), default=None)
    )
    macro = getattr(sensors, "macro_and_sentiment", None) or {}
    ps = getattr(sensors, "portfolio_summary", None) or {}
    fg = _resolve_macro_sensor(sensors, macro, "fear_greed")
    mb = _resolve_macro_sensor(sensors, macro, "market_breadth")
    sa = _resolve_macro_sensor(sensors, macro, "sentiment_advisor")
    gt = _resolve_macro_sensor(sensors, macro, "geopolitical_threat", fallback_key="geopolitical")
    mb_bench = _resolve_macro_sensor(sensors, macro, "macro_benchmarks", ps=ps)

    fg_hist = fg.get("historical_values")[:7] if isinstance(fg.get("historical_values"), list) else None
    fg_line = (
        f"{fg.get('value', 'N/A')} ({fg.get('classification') or fg.get('sentiment') or 'N/A'}, "
        f"7d_chg={fg.get('trend_7d_change', 'N/A')}"
        + (f", 7d_hist={fg_hist}" if fg_hist else "")
        + f"{_age_sfx(fg, now_ts=ref_now)})"
        if fg else "None"
    )
    mb_gainers = mb.get("top_gainers")[:3] if isinstance(mb.get("top_gainers"), list) else None
    mb_losers = mb.get("top_losers")[:3] if isinstance(mb.get("top_losers"), list) else None
    mb_rot_parts: List[str] = []
    if mb:
        if mb.get("eth_btc_ratio") is not None:
            eb_c = f" ({fmt_signed(mb.get('eth_btc_chg_24h_pct'))}% 24h)" if mb.get("eth_btc_chg_24h_pct") is not None else ""
            mb_rot_parts.append(f"ETH/BTC={mb.get('eth_btc_ratio')}{eb_c}")
        if mb.get("eth_price") is not None:
            eth_c = f" ({fmt_signed(mb.get('eth_chg_24h_pct'))}% 24h)" if mb.get("eth_chg_24h_pct") is not None else ""
            mb_rot_parts.append(f"ETH=${mb.get('eth_price')}{eth_c}")
        if mb.get("btc_vol_share_pct") is not None:
            mb_rot_parts.append(f"BTC_vol_share={mb.get('btc_vol_share_pct')}%")
        if mb.get("alt_vol_share_pct") is not None:
            mb_rot_parts.append(f"Alt_vol_share={mb.get('alt_vol_share_pct')}%")
    mb_trim_sfx = ""
    if mb and (mb_tm := safe_float(mb.get("trimmed_mean_change_pct"), None)) is not None:
        mb_mn = safe_float(mb.get("mean_change_pct"), None)
        if mb_mn is None or abs(mb_mn - mb_tm) >= 1.0:
            mb_trim_sfx = f" (trimmed={mb_tm}%)"
    mb_age_m = _resolve_sensor_age_min(mb, now_ts=ref_now)
    mb_line = (
        f"advance_ratio={mb.get('advance_ratio', 'N/A')}, "
        f"median_chg={mb.get('median_change_pct', 'N/A')}%, "
        f"mean_chg={mb.get('mean_change_pct', 'N/A')}%{mb_trim_sfx}, disp_std={mb.get('dispersion_std', 'N/A')}, "
        f"advancing={mb.get('advancing', 'N/A')}/{mb.get('total_symbols', 'N/A')}"
        + (f", {', '.join(mb_rot_parts)}" if mb_rot_parts else "")
        + (f", top_gainers={mb_gainers}" if mb_gainers else "")
        + (f", top_losers={mb_losers}" if mb_losers else "")
        + (f", age={mb_age_m}m" if mb_age_m is not None else "")
        if mb else "None"
    )
    sa_age_m = _resolve_sensor_age_min(sa, now_ts=ref_now)
    sa_flow_raw = str(sa.get("live_flow_divergence") or "").strip() if sa else ""
    sa_flow_delta = (
        sa_flow_raw[sa_flow_raw.find("("):]
        if "(" in sa_flow_raw
        else (f"({sa_flow_raw})" if sa_flow_raw else "")
    )
    sa_summary_raw = str(sa.get("summary") or "").strip() if sa else ""
    sa_fut_lead_v: Optional[float] = None
    if isinstance(mb_bench, dict) and mb_bench:
        sa_sum_low = sa_summary_raw.lower()
        sa_mentions_fut = any(tok in sa_sum_low for tok in ("fut ", "futures", "es/nq", "es=f", "nq=f"))
        fl_raw = safe_float(compute_futures_premarket_lead(mb_bench).get("futures_lead_pct"), None)
        if fl_raw is None and sa_mentions_fut:
            mb_it = mb_bench.get("items") if isinstance(mb_bench.get("items"), dict) else mb_bench
            for _fsym in ("NQ=F", "ES=F"):
                if (_scc := resolve_futures_since_cash_close_pct(mb_it, _fsym)) is not None:
                    fl_raw = _scc
                    break
        if fl_raw is not None and (abs(fl_raw) >= 0.25 or sa_mentions_fut):
            sa_fut_lead_v = fl_raw
    if sa_flow_delta:
        sa_flow_delta = _SA_FLOW_POSTURE_RE.sub(r"\1summary", sa_flow_delta)
        if sa_age_m is not None:
            sa_flow_delta = _SA_FLOW_AGE_RE.sub(rf"\g<1>{sa_age_m}\g<2>", sa_flow_delta)
        if (
            "CB_Prem=" not in sa_flow_delta
            and " vs " in sa_flow_delta
            and isinstance(mb_bench, dict)
            and (cb_v := safe_float(mb_bench.get("coinbase_premium_bps"), None)) is not None
        ):
            sa_flow_delta = sa_flow_delta.replace(" vs ", f", CB_Prem={fmt_signed(cb_v, 1)}bps vs ", 1)
        if (
            sa_fut_lead_v is not None
            and "Fut_Lead=" not in sa_flow_delta
            and " vs " in sa_flow_delta
        ):
            sa_flow_delta = sa_flow_delta.replace(" vs ", f", Fut_Lead={fmt_signed(sa_fut_lead_v, 2)}% vs ", 1)
    if not sa_flow_delta and sa and (sa_age_m or 0.0) >= 5.0:
        micro_map = getattr(sensors, "microstructure", None) or {}
        btc_m = (
            micro_map.get("BTCUSDC")
            or micro_map.get("BTCUSDT")
            or micro_map.get("BTCUSD")
            if isinstance(micro_map, dict)
            else None
        )
        if isinstance(btc_m, dict):
            d_parts: List[str] = []
            tr_v = safe_float(btc_m.get("taker_buy_sell_ratio"), None)
            cvd_v = safe_float(btc_m.get("cvd_15m_usd"), None)
            tr5_v = safe_float(btc_m.get("taker_ratio_5m"), None)
            cvd5_v = safe_float(btc_m.get("cvd_5m_usd"), None)
            tr1h_v = safe_float(btc_m.get("taker_ratio_1h"), None)
            cvd1h_v = safe_float(btc_m.get("cvd_1h_usd"), None)
            if tr_v is not None:
                d_parts.append(f"taker_ratio={tr_v:.2f}")
            elif tr1h_v is not None:
                d_parts.append(f"1h_taker={tr1h_v:.2f}")
            if cvd_v is not None:
                d_parts.append(f"cvd_15m={fmt_signed(cvd_v / 1e6)}M")
            elif cvd1h_v is not None:
                d_parts.append(f"cvd_1h={fmt_signed(cvd1h_v / 1e6)}M")
            if tr5_v is not None:
                d_parts.append(f"5m_tr={tr5_v:.2f}")
            if cvd5_v is not None:
                d_parts.append(f"cvd_5m={fmt_signed(cvd5_v / 1e6)}M")
            if tr_v is not None and tr1h_v is not None:
                d_parts.append(f"1h_tr={tr1h_v:.2f}")
            if cvd_v is not None and cvd1h_v is not None:
                d_parts.append(f"cvd_1h={fmt_signed(cvd1h_v / 1e6)}M")
            if isinstance(mb_bench, dict) and (cb_v := safe_float(mb_bench.get("coinbase_premium_bps"), None)) is not None:
                d_parts.append(f"CB_Prem={fmt_signed(cb_v, 1)}bps")
            if sa_fut_lead_v is not None:
                d_parts.append(f"Fut_Lead={fmt_signed(sa_fut_lead_v, 2)}%")
            if d_parts:
                sa_flow_delta = f"({', '.join(d_parts)} vs {sa_age_m}m-old summary)"
    sa_poc_live_part = ""
    if sa and (sa_age_m or 0.0) >= 5.0 and any(tok in sa_summary_raw.lower() for tok in ("poc", "vah", "val", "value area")):
        pt_map_sa = getattr(sensors, "prices_and_trends", None) or {}
        btc_pt = (
            pt_map_sa.get("BTCUSDC")
            or pt_map_sa.get("BTCUSDT")
            or pt_map_sa.get("BTCUSD")
            if isinstance(pt_map_sa, dict)
            else None
        )
        if isinstance(btc_pt, dict):
            btc_px_live = safe_pos_float(btc_pt.get("price"), default=None)
            btc_poc_live = safe_pos_float(btc_pt.get("poc_px"), default=None)
            if btc_px_live is not None and btc_poc_live is not None:
                sa_poc_live_part = f"BTC={_fmt_px(btc_px_live)}(POC={_fmt_px(btc_poc_live)})"
            elif btc_px_live is not None:
                sa_poc_live_part = f"BTC={_fmt_px(btc_px_live)}"
    if sa_poc_live_part:
        if sa_flow_delta and "BTC=" not in sa_flow_delta and " vs " in sa_flow_delta:
            sa_flow_delta = sa_flow_delta.replace(" vs ", f", {sa_poc_live_part} vs ", 1)
        elif not sa_flow_delta:
            sa_flow_delta = f"({sa_poc_live_part} vs {sa_age_m}m-old summary)"
    sa_summary = (
        _SA_SUMMARY_DIRECTIVE_RE.sub(
            lambda m: "." if m.group(0).lstrip().startswith(",") else "",
            sa_summary_raw,
        )
        .replace("..", ".")
        .strip()
    )
    if not sa_summary or sa_summary == ".":
        sa_summary = sa_summary_raw
    if len(sa_summary) > 360:
        cut_s = sa_summary[:357]
        if (dot_i := cut_s.rfind(". ")) >= 160:
            sa_summary = cut_s[: dot_i + 1].strip()
        else:
            sa_summary = cut_s.rstrip(" ,;:") + "..."
    sa_sum_key = (
        f"historical_summary(as_of -{sa_age_m}m; use Sec 3 live px/POC)"
        if (sa and sa_poc_live_part)
        else (
            f"historical_summary(as_of -{sa_age_m}m)"
            if (sa and sa_age_m is not None and sa_age_m >= 15.0)
            else "summary"
        )
    )
    show_sa_sum = delta_plan is None or delta_plan.should_show("sentiment_advisor_summary")
    sa_tag = delta_plan.tag_for("sentiment_advisor_summary") if (delta_plan is not None and sa) else ""
    sa_sum_part = (
        f" | {sa_sum_key}={sa_summary}"
        if show_sa_sum
        else " | summary=[unchanged in thread]"
    )
    sa_line = (
        f"conf={sa.get('confidence', 'N/A')}"
        f"{_age_sfx(sa, now_ts=ref_now)}"
        + (f" | live_btc_flow_delta={sa_flow_delta}" if sa_flow_delta else "")
        + sa_sum_part
        if sa else "None"
    )
    gt_summary_raw = str(gt.get("summary") or "").strip() if gt else ""
    gt_summary = (
        _GT_SUMMARY_DIRECTIVE_RE.sub(".", gt_summary_raw)
        .replace("..", ".")
        .strip()
    )
    if not gt_summary or gt_summary == ".":
        gt_summary = gt_summary_raw
    gt_age_m = _resolve_sensor_age_min(gt, now_ts=ref_now)
    gt_sum_low = gt_summary.lower()
    gt_live_macro_delta = ""
    if (
        isinstance(gt, dict)
        and gt_age_m is not None
        and gt_age_m >= 15.0
        and isinstance(mb_bench, dict)
        and mb_bench
        and not gt.get("live_deescalation_note")
    ):
        gt_d_parts: List[str] = []
        gt_fl = safe_float(compute_futures_premarket_lead(mb_bench).get("futures_lead_pct"), None)
        if gt_fl is None and any(tok in gt_sum_low for tok in ("es=f", "nq=f", "futures")):
            mb_it_gt = mb_bench.get("items") if isinstance(mb_bench.get("items"), dict) else mb_bench
            for _fsym in ("NQ=F", "ES=F"):
                if (_scc := resolve_futures_since_cash_close_pct(mb_it_gt, _fsym)) is not None:
                    gt_fl = _scc
                    break
        if gt_fl is not None and (
            (
                gt_fl <= -0.15
                and any(
                    tok in gt_sum_low
                    for tok in ("futures advancing", "futures rising", "futures higher", "futures climbing", "futures rally", "es=f +", "nq=f +")
                )
            )
            or (
                gt_fl >= 0.15
                and any(
                    tok in gt_sum_low
                    for tok in ("futures falling", "futures dropping", "futures sliding", "futures lower", "futures selloff", "es=f -", "nq=f -")
                )
            )
        ):
            gt_d_parts.append(f"Fut_Lead={fmt_signed(gt_fl, 2)}%")
        if (
            (btc_sk := safe_float(mb_bench.get("btc_skew_25d_pct"), None)) is not None
            and abs(btc_sk) >= 4.0
            and any(tok in gt_sum_low for tok in ("calm", "steady", "subdued"))
            and (gt_d_parts or any(tok in gt_sum_low for tok in ("volatility", "dvol", "options", "skew")))
        ):
            gt_d_parts.append(f"Skew_25d={fmt_signed(btc_sk, 2)}%")
        if gt_d_parts:
            gt_live_macro_delta = f"({', '.join(gt_d_parts)} vs {gt_age_m}m-old summary)"
    gt_stale_ctx_note = (
        "historical_eval_gt_90m_verify_live_section2"
        if (
            isinstance(gt, dict)
            and gt_age_m is not None
            and gt_age_m >= 90.0
            and not gt.get("live_deescalation_note")
            and any(tok in gt_sum_low for tok in ("90-minute window", "spy", "qqq", "vix", "es=f", "nq=f", "futures", "dvol", "dxy", "past 15 minutes", "1h)"))
        )
        else ""
    )
    show_gt_sum = delta_plan is None or delta_plan.should_show("geopolitical_summary")
    gt_tag = delta_plan.tag_for("geopolitical_summary") if (delta_plan is not None and gt) else ""
    gt_sum_part = f" | summary={gt_summary}" if show_gt_sum else " | summary=[unchanged in thread]"
    gt_line = (
        f"risk_score={gt.get('risk_score', 'N/A')}{_age_sfx(gt, now_ts=ref_now)}"
        + (f" | live_deescalation={gt.get('live_deescalation_note')}" if gt.get("live_deescalation_note") else "")
        + (f" | live_unassessed={gt.get('live_unassessed_high_sev_note')}" if gt.get("live_unassessed_high_sev_note") else "")
        + (f" | live_macro_delta={gt_live_macro_delta}" if gt_live_macro_delta else "")
        + (f" | shield_note={gt_stale_ctx_note}" if gt_stale_ctx_note else "")
        + gt_sum_part
        if gt else "None"
    )
    if is_equities_scope:
        eq_reg_line = format_equity_macro_regime_line(mb_bench) if isinstance(mb_bench, dict) and mb_bench else "NEUTRAL"
        lines = [
            "\n### 2. MACRO, REGIME & CROSS-MARKET:",
        ]
    else:
        lines = [
            "\n### 2. MACRO, REGIME & CROSS-MARKET:",
        ]
    op_note = _resolve_macro_sensor(sensors, macro, "operator_regime_note")
    if isinstance(op_note, dict) and str(op_note.get("text") or "").strip() and (
        delta_plan is None or delta_plan.should_show("operator_regime_note")
    ):
        op_tag = delta_plan.tag_for("operator_regime_note") if delta_plan is not None else ""
        op_upd = op_note.get("updated_date") or "-"
        op_exp = safe_float(op_note.get("expires_in_days"), None)
        exp_s = f", expires_in={op_exp:.1f}d" if op_exp is not None else ""
        lines.append(
            f"- Operator Strategic Note (updated={op_upd}{exp_s}){op_tag}: {str(op_note['text']).strip()}"
        )
    if is_equities_scope:
        lines.extend([
            f"- US Equity & Macro Regime: {eq_reg_line}{_age_sfx(mb_bench, parens=True, now_ts=ref_now) if isinstance(mb_bench, dict) else ''}",
            f"- Geopolitical Threat Shield{gt_tag}: {gt_line}",
        ])
    else:
        if delta_plan is None or delta_plan.should_show("fear_greed"):
            fg_tag = delta_plan.tag_for("fear_greed") if (delta_plan is not None and fg) else ""
            lines.append(f"- Fear & Greed{fg_tag}: {fg_line}")
        lines.extend([
            f"- Market Breadth: {mb_line}",
            f"- Macro Sentiment Advisor{sa_tag}: {sa_line}",
            f"- Geopolitical Threat Shield{gt_tag}: {gt_line}",
        ])
    breaking_news = getattr(sensors, "breaking_news", None) or macro.get("recent_news") or []
    mc = _resolve_macro_sensor(sensors, macro, "macro_calendar") or resolve_macro_economic_calendar(
        now_ts=ref_now,
        breaking_news=breaking_news,
    )
    if (delta_plan is None or delta_plan.should_show("macro_calendar")) and (
        mc_line := _format_macro_calendar_line(mc)
    ):
        mc_tag = delta_plan.tag_for("macro_calendar") if delta_plan is not None else ""
        if mc_tag and ": " in mc_line:
            mc_line = mc_line.replace(": ", f"{mc_tag}: ", 1)
        lines.append(mc_line)
    mb_bench_line = _extract_macro_benchmarks_line(mb_bench, include_crypto=True, now_ts=ref_now)
    eq_already_rendered_benchmarks = bool(
        is_equities_scope
        and isinstance(mb_bench, dict)
        and mb_bench
        and eq_reg_line not in ("RISK_ON", "RISK_OFF", "NEUTRAL")
    )
    if mb_bench_line and not eq_already_rendered_benchmarks:
        lines.append(
            f"- US Equities, Futures & Macro Rates Benchmarks (SPY / QQQ / ES=F / NQ=F / DXY / US10Y 24h Delta): {mb_bench_line}"
        )
    if include_cross_market and isinstance(ps.get("cross_market_context"), dict) and ps["cross_market_context"]:
        cm_view = evaluate_portfolio_prompt_view(
            {"cross_market_context": ps["cross_market_context"]},
            asset_scope=asset_scope,
        )
        if cm_view["crypto_bellwethers"]:
            lines.extend(_format_cross_market_crypto_table(cm_view["crypto_bellwethers"]))
        if (delta_plan is None or delta_plan.should_show("equity_bellwethers")) and cm_view["equity_bellwethers"]:
            eq_tag = delta_plan.tag_for("equity_bellwethers") if delta_plan is not None else ""
            lines.extend(_format_cross_market_equities_table(cm_view["equity_bellwethers"], scope_s=scope_s, tag=eq_tag))
    if not is_equities_scope and (delta_plan is None or delta_plan.should_show("etf_flows")):
        etf_flows = _resolve_macro_sensor(sensors, macro, "etf_flows", ps=ps)
        if etf_flows:
            etf_s = format_etf_flows_summary_line(etf_flows) or str(etf_flows.get("summary_line") or "").strip()
            if etf_s:
                etf_tag = delta_plan.tag_for("etf_flows") if delta_plan is not None else ""
                lines.append(
                    f"- US Spot Crypto ETF Net Flows (Farside Daily){etf_tag}: {etf_s}{_age_sfx(etf_flows, parens=True, now_ts=ref_now)}"
                )
    if delta_plan is None or delta_plan.should_show("long_horizon"):
        lh_snap = _resolve_macro_sensor(sensors, macro, "long_horizon")
        if lh_snap:
            lh_tag = delta_plan.tag_for("long_horizon") if delta_plan is not None else ""
            lh_extra_px: Dict[str, float] = {}
            if isinstance(mb_bench, dict):
                bench_items = mb_bench.get("items") if isinstance(mb_bench.get("items"), dict) else {}
                for bk, bv in bench_items.items():
                    if isinstance(bv, dict) and (bp := safe_pos_float(bv.get("price"), default=None)) is not None:
                        lh_extra_px[str(bk).strip().upper()] = bp
            if isinstance(ps.get("cross_market_context"), dict):
                cb_map = ps["cross_market_context"].get("crypto_bellwethers")
                if isinstance(cb_map, dict):
                    for ck, cv in cb_map.items():
                        if isinstance(cv, dict) and (cp := safe_pos_float(cv.get("price"), default=None)) is not None:
                            lh_extra_px[str(ck).strip().upper()] = cp
            lines.extend(
                format_long_horizon_section_lines(
                    lh_snap,
                    prices_and_trends=getattr(sensors, "prices_and_trends", None),
                    extra_live_prices=lh_extra_px or None,
                    asset_scope=scope_s,
                    tag=lh_tag,
                )
            )
    if breaking_news and (delta_plan is None or delta_plan.should_show("breaking_news")):
        if delta_plan is not None and delta_plan.new_news_titles is not None:
            shown_news = [
                hn for hn in breaking_news
                if isinstance(hn, dict) and str(hn.get("title") or "").strip() in delta_plan.new_news_titles
            ]
            if shown_news:
                om_n = safe_int(getattr(delta_plan, "omitted_news_count", 0), 0)
                prior_sfx = f"; {om_n} prior unchanged in thread" if om_n > 0 else ""
                lines.append(f"- Live Breaking News Feed [{len(shown_news)} NEW{prior_sfx}]:")
                for hn in shown_news:
                    lines.append(f"  * {hn.get('title')}{_age_sfx(hn, parens=True, now_ts=ref_now)}")
        else:
            lines.append("- Live Breaking News Feed:")
            for hn in breaking_news:
                lines.append(f"  * {hn.get('title')}{_age_sfx(hn, parens=True, now_ts=ref_now)}")
    return lines


def _format_sre_and_fleet_section(
    sensors: Any,
    logs: Any,
    lookback_hours: float,
) -> List[str]:
    """Format SRE functional telemetry, log anomalies, fleet heartbeats, and OS/VPN resources."""
    hb_keys = list(sensors.active_heartbeats.keys())
    stopped_hb_note = ""
    if hb_keys and getattr(sensors, "fleet_processes", None):
        stopped_names = {
            f"{p.get('name')}.heartbeat"
            for p in sensors.fleet_processes
            if isinstance(p, dict) and not _parse_bool_flag(p.get("is_running"), False) and p.get("name")
        }
        stopped_hbs = [k for k in hb_keys if k in stopped_names]
        if stopped_hbs:
            hb_keys = [k for k in hb_keys if k not in stopped_names]
            stopped_hb_note = f" (disk heartbeats for stopped processes: {stopped_hbs})"
    state_files_s = _format_compact_kv_value(sensors.state_files) if sensors.state_files else "none"
    lines = [
        "### 1. FUNCTIONAL SYSTEM TELEMETRY, HEARTBEATS & WATCHDOG STATES:",
        f"- Active Heartbeats: {hb_keys}{stopped_hb_note}",
        f"- Watchdog & Runtime State Files: {state_files_s}",
        f"\n### 2. LOG ANOMALIES & GUARD VETOES / PERSISTENT UNRESOLVED ERRORS (PAST {lookback_hours:g}H):",
        f"- Scanned Logs: {logs.total_scanned_files}",
        f"- Error Count: {logs.error_count}",
        f"- Guard Interventions: {logs.warning_count}",
    ]
    if getattr(logs, "persistent_errors", None):
        lines.append("- Persistent / Unresolved Errors in Logs:")
        for pe in logs.persistent_errors[:15]:
            tail_lbl = "UNRESOLVED_AT_TAIL" if _parse_bool_flag(pe.get("unresolved_at_tail"), False) else "RECURRING"
            lines.append(
                f"  * [{tail_lbl} x{pe.get('count', 1)} in {pe.get('file', 'log')}] {pe.get('sample', '')}"
            )
    lines.append("- Sample Errors & Interventions:")
    for evt in logs.critical_events[:15]:
        lines.append(f"  * {evt}")
    for g_evt in logs.guard_interventions[:10]:
        lines.append(f"  * [GUARD] {g_evt}")

    if sensors.fleet_processes:
        running_p = [p for p in sensors.fleet_processes if _parse_bool_flag(p.get("is_running"), False)]
        stopped_p = [p["name"] for p in sensors.fleet_processes if not _parse_bool_flag(p.get("is_running"), False)]
        stale_p = [
            f"{p['name']} (age={p.get('age_seconds')}s > {p.get('max_stale_seconds')}s)"
            for p in sensors.fleet_processes
            if _parse_bool_flag(p.get("is_running"), False) and not _parse_bool_flag(p.get("heartbeat_fresh"), True)
        ]
        hb_summary = (
            f" | STALE HEARTBEAT: {stale_p}"
            if stale_p
            else (
                " | No running heartbeats (all fleet processes stopped)"
                if not running_p
                else (" | All running heartbeats fresh" if stopped_p else " | All heartbeats fresh")
            )
        )
        lines.append(
            f"\n### 2B. FLEET PROCESSES & BOT HEARTBEATS (procs.conf): {len(running_p)}/{len(sensors.fleet_processes)} running"
            + (f" | STOPPED: {stopped_p}" if stopped_p else "")
            + hb_summary
        )

    if sensors.system_and_vpn:
        sv = sensors.system_and_vpn
        cfg_audit = sv.get("config_audit") if isinstance(sv.get("config_audit"), dict) else {}
        cfg_issues = cfg_audit.get("issues") or []
        cfg_summary = (
            f"ISSUES DETECTED: {_format_compact_kv_value(cfg_issues)}"
            if cfg_issues
            else f"OK (verified {cfg_audit.get('checked_files', [])})"
        )
        lines.append(
            "\n### 2C. LINUX SYSTEM RESOURCES & VPN TUNNEL HEALTH:\n"
            f"- CPU Load Avg (1m/5m/15m): {sv.get('load_avg', 'N/A')}\n"
            f"- Memory: used={sv.get('mem_used_pct', 'N/A')}%, avail={sv.get('mem_avail_mb', 'N/A')} MB\n"
            f"- Disk: used={sv.get('disk_used_pct', 'N/A')}%, free={sv.get('disk_free_gb', 'N/A')} GB\n"
            f"- VPN Tunnel ({sv.get('vpn_interface', 'wgpia0')}): {'UP' if _parse_bool_flag(sv.get('vpn_tunnel_up'), False) else 'DOWN / NOT PRESENT'}\n"
            f"- Internet & DNS Connectivity: healthy={sv.get('internet_healthy', True)}, "
            f"default_route={sv.get('default_route_present', 'N/A')}, dns_servers={sv.get('dns_servers', 'N/A')}, "
            f"vpn_outage_active={sv.get('vpn_outage_active', False)}\n"
            f"- Configuration Files Audit: {cfg_summary}"
            + (
                f"\n- Clock Sync (NTP): service={(sv['clock_sync'].get('os_clock') or {}).get('service', 'unknown')}, "
                f"synced={(sv['clock_sync'].get('os_clock') or {}).get('synchronized', False)}, "
                f"offset_ms={(sv['clock_sync'].get('os_clock') or {}).get('offset_ms', 'N/A')}, "
                f"healthy={sv['clock_sync'].get('healthy', False)}"
                + (
                    f", reasons={sv['clock_sync'].get('reasons')}"
                    if sv["clock_sync"].get("reasons")
                    else ""
                )
                if isinstance(sv.get("clock_sync"), dict)
                else ""
            )
            + (
                f"\n- Sustained High-Resource Processes (os_resources_watchdog_state.json): "
                f"{_format_compact_kv_value(sv['resource_watchdog_offenders'])}"
                if sv.get("resource_watchdog_offenders")
                else ""
            )
        )
        cat = sv.get("llm_catalog_audit") if isinstance(sv.get("llm_catalog_audit"), dict) else None
        if cat:
            bin_ver = cat.get("binary_versions") if isinstance(cat.get("binary_versions"), dict) else {}
            bin_str = ", ".join(f"{k}={v}" for k, v in bin_ver.items()) or "N/A"
            missing_m = cat.get("missing_configured_models") or []
            new_fams = cat.get("new_families") or []
            if not _parse_bool_flag(cat.get("agy_models_ok"), False):
                cat_status = f"PROBE_FAILED ({cat.get('error', 'unknown')})"
            elif missing_m:
                cat_status = f"MISSING_CONFIGURED_MODELS: {missing_m}"
            else:
                cat_status = f"OK ({len(cat.get('configured_models') or [])} configured in {cat.get('available_count', 0)} agy models)"
            new_str = f" | NEW_MODEL_FAMILIES: {new_fams}" if new_fams else ""
            rec_upd = cat.get("recommended_config_updates") if isinstance(cat.get("recommended_config_updates"), dict) else {}
            upd_str = f" | AUTO_CONFIG_UPDATES: {_format_compact_kv_value(rec_upd)}" if rec_upd else ""
            lines.append(f"- LLM Model Catalog & Standalone Binaries: [{bin_str}] | configured_models={cat_status}{new_str}{upd_str}")
    return lines


def _extract_active_portfolio_symbols(
    portfolio_summary: Any,
    *,
    focus_symbols: Optional[List[str]] = None,
) -> set[str]:
    """Collect normalized symbol keys for active positions, cycles, open orders, and tactical focus symbols."""
    active: set[str] = set()

    def _add_sym(raw_sym: Any) -> None:
        s_u = str(raw_sym or "").strip().upper()
        s_k = clean_symbol_key(raw_sym) or ""
        if s_u:
            active.add(s_u)
        if s_k:
            active.add(s_k)

    ps = portfolio_summary if isinstance(portfolio_summary, dict) else {}
    for pos in (ps.get("tradable_positions") or ()):
        if isinstance(pos, dict) and (
            (safe_pos_float(pos.get("value_usd"), default=0.0) or 0.0) >= 2.0
            or (safe_pos_float(pos.get("free_base"), default=0.0) or 0.0) > 0.0
            or (safe_pos_float(pos.get("locked_base_qty"), default=0.0) or 0.0) > 0.0
        ):
            _add_sym(pos.get("disp_symbol") or pos.get("symbol"))
    for oo in (ps.get("open_limit_orders") or ()):
        if isinstance(oo, dict):
            _add_sym(oo.get("symbol") or oo.get("pair") or oo.get("ticker"))
    for cyc_key in ("binance_cycles", "kraken_cycles", "hyperliquid_cycles", "t212_cycles"):
        cyc_map = ps.get(cyc_key)
        if isinstance(cyc_map, dict):
            for c_sym, c_val in cyc_map.items():
                if isinstance(c_val, dict) and (
                    (safe_pos_float(c_val.get("qty"), default=0.0) or 0.0) > 0.0
                    or bool(c_val.get("resting_orders"))
                ):
                    _add_sym(c_sym)
    if focus_symbols:
        for f_sym in parse_symbol_list(focus_symbols):
            _add_sym(f_sym)
    return active


def _is_compact_observe_only_symbol(
    sym: str,
    obs_syms: set[str],
    *,
    compact_observe_only: bool,
    active_symbols: Optional[set[str]],
) -> tuple[bool, bool]:
    """Return ``(is_observe_only, should_render_compact_bellwether)`` for ``sym``."""
    sym_u = str(sym or "").strip().upper()
    sym_k = clean_symbol_key(sym) or ""
    is_obs = bool(obs_syms and (sym_u in obs_syms or (sym_k and sym_k in obs_syms)))
    if not (compact_observe_only and is_obs):
        return is_obs, False
    if active_symbols and (sym_u in active_symbols or (sym_k and sym_k in active_symbols)):
        return is_obs, False
    return is_obs, True


def _build_market_price_symbol_record(
    sym: str,
    pinfo: Dict[str, Any],
    sib_specs: List[Any],
    *,
    btc_entry: Optional[Dict[str, Any]],
    obs_syms: set[str],
    compact_observe_only: bool,
    active_symbols: Optional[set[str]],
) -> Dict[str, Any]:
    """Build reusable price/trend prompt parts for both single-line and unified-card market sections."""
    px = pinfo.get("price")
    px_f = safe_float(px, 0.0)
    px_str = _fmt_px(px_f, px)
    is_obs_sym, is_compact_obs = _is_compact_observe_only_symbol(
        sym,
        obs_syms,
        compact_observe_only=compact_observe_only,
        active_symbols=active_symbols,
    )
    obs_lbl = " [OBSERVE-ONLY]" if is_obs_sym else ""
    sib_lbl = ""
    sib_syms: List[str] = []
    if sib_specs:
        sib_parts: List[str] = []
        for sib_sym, s_px, diff_pct in sib_specs:
            sib_syms.append(str(sib_sym))
            if s_px > 0:
                diff_s = (
                    ""
                    if diff_pct is None
                    else (f" [{fmt_signed(diff_pct * 100.0, 1)}bps]" if abs(diff_pct) < 0.005 else f" [{fmt_signed(diff_pct)}%]")
                )
                sib_parts.append(f"{sib_sym}={_fmt_px(s_px)}{diff_s}")
            else:
                sib_parts.append(sib_sym)
        if sib_parts:
            sib_lbl = f" (sibling venue: {', '.join(sib_parts)})"

    eq_qmeta_sfx = format_equity_quote_meta_suffix(pinfo) if is_equity_symbol(sym) else ""
    if is_compact_obs:
        c_extras: List[str] = []
        c_hmap: Dict[str, str] = {}
        for h_lbl, h_k in (("15m", "chg_15m_pct"), ("1h", "chg_1h_pct"), ("6h", "chg_6h_pct"), ("24h", "chg_24h_pct")):
            if h_k in pinfo and pinfo.get(h_k) is not None:
                c_hmap[h_lbl] = f"{fmt_signed(pinfo.get(h_k))}%"
        c_horizons = " ".join(f"{k}={v}" for k, v in c_hmap.items())
        pos_24h = pinfo.get("pos_24h_pct")
        if pos_24h is None and px_f > 0 and pinfo.get("low_24h") and pinfo.get("high_24h"):
            pos_24h = compute_horizon_range_dict(
                px_f, safe_float(pinfo.get("low_24h"), 0.0), safe_float(pinfo.get("high_24h"), 0.0), "24h"
            ).get("pos_24h_pct")
        if c_horizons or pos_24h is not None:
            c_extras.append(" ".join(p for p in (c_horizons, f"pos_24h={pos_24h}%" if pos_24h is not None else "") if p))
        ctx_parts: List[str] = []
        span_7d_c = safe_pos_float(pinfo.get("span_7d_hours"), default=None)
        is_part_7d_c = bool(pinfo.get("is_partial_7d")) or (span_7d_c is not None and span_7d_c < 120.0)
        if "chg_7d_pct" in pinfo and pinfo.get("chg_7d_pct") is not None and not is_part_7d_c:
            c_7d = f"7d={fmt_signed(pinfo.get('chg_7d_pct'))}%"
            c_extras.append(c_7d)
            ctx_parts.append(c_7d)
        if not is_equity_or_tokenized_symbol(sym) and canonical_base_asset(sym) != "BTC" and btc_entry:
            vb_c = [
                f"{h_lbl}={fmt_signed(a_v - b_v)}%"
                for h_lbl, h_k in (("1h", "chg_1h_pct"), ("24h", "chg_24h_pct"))
                if (a_v := safe_float(pinfo.get(h_k), None)) is not None
                and (b_v := safe_float(btc_entry.get(h_k), None)) is not None
            ]
            if vb_c:
                vb_s = f"vs_btc({', '.join(vb_c)})"
                c_extras.append(vb_s)
                ctx_parts.append(vb_s)
        if pinfo.get("long_trend_dir"):
            lt_c = f"long_trend={str(pinfo.get('long_trend_dir')).upper()}({pinfo.get('long_trend_hours')}h)"
            c_extras.append(lt_c)
            ctx_parts.append(lt_c)
        if eq_qmeta_sfx:
            ctx_parts.append(eq_qmeta_sfx.strip())
        c_extra_s = f" | {' | '.join(c_extras)}" if c_extras else ""
        return {
            "sym": sym,
            "sib_lbl": sib_lbl,
            "sib_syms": sib_syms,
            "px_str": px_str,
            "is_obs_sym": is_obs_sym,
            "is_compact_obs": True,
            "obs_lbl": obs_lbl,
            "eq_qmeta_sfx": eq_qmeta_sfx,
            "full_extras": [],
            "compact_cols": {
                "px_str": px_str,
                "15m": c_hmap.get("15m", "-"),
                "1h": c_hmap.get("1h", "-"),
                "6h": c_hmap.get("6h", "-"),
                "24h": c_hmap.get("24h", "-"),
                "pos_24h": f"{pos_24h}%" if pos_24h is not None else "-",
                "context": ", ".join(ctx_parts) if ctx_parts else "-",
            },
            "legacy_line": f"  * {sym}{sib_lbl}: Price={px_str}{obs_lbl}{eq_qmeta_sfx}{c_extra_s}",
        }

    extras: List[str] = []
    _m24_h_keys = (
        "chg_1m_pct", "chg_3m_pct", "chg_5m_pct", "chg_15m_pct", "chg_30m_pct",
        "chg_1h_pct", "chg_3h_pct", "chg_6h_pct", "chg_12h_pct", "chg_24h_pct",
    )
    if "chg_15m_pct" in pinfo or any(pinfo.get(k) is not None for k in _m24_h_keys):
        all_horizons = "".join(
            f"{h_lbl}={fmt_signed(pinfo.get(h_k))}% "
            for h_lbl, h_k in (
                ("1m", "chg_1m_pct"), ("3m", "chg_3m_pct"), ("5m", "chg_5m_pct"),
                ("15m", "chg_15m_pct"), ("30m", "chg_30m_pct"), ("1h", "chg_1h_pct"),
                ("3h", "chg_3h_pct"), ("6h", "chg_6h_pct"), ("12h", "chg_12h_pct"), ("24h", "chg_24h_pct"),
            )
            if h_k in pinfo and pinfo.get(h_k) is not None
        )
        extras.append(f"{all_horizons}{_format_horizon_range(pinfo, px_f, '24h')}")
        if isinstance(pinfo.get("recent_tape_15m"), list) and pinfo["recent_tape_15m"]:
            tape_str = f"tape_15m=[{', '.join(str(x) for x in pinfo['recent_tape_15m'])}]"
            if pinfo.get("low_15m") is not None and pinfo.get("high_15m") is not None:
                pos15_sfx = f" pos_15m={pinfo['pos_15m_pct']}%" if pinfo.get("pos_15m_pct") is not None else ""
                tape_str += f" 15m_range=[{pinfo['low_15m']}, {pinfo['high_15m']}]{pos15_sfx}"
            extras.append(tape_str)
    if all(pinfo.get(k) is not None for k in ("poc_px", "val_px", "vah_px")):
        fb_vp = build_vp_position_dict(
            px_f,
            safe_float(pinfo.get("poc_px"), 0.0),
            safe_float(pinfo.get("val_px"), 0.0),
            safe_float(pinfo.get("vah_px"), 0.0),
        )
        poc_d, val_d, vah_d = (
            safe_float(pinfo[k] if pinfo.get(k) is not None else fb_vp.get(k), 0.0)
            for k in ("poc_dist_pct", "val_dist_pct", "vah_dist_pct")
        )
        va_pos_raw = pinfo["va_pos_pct"] if pinfo.get("va_pos_pct") is not None else fb_vp.get("va_pos_pct")
        va_pos = round(max(-100.0, min(200.0, safe_float(va_pos_raw, 50.0))), 1) if va_pos_raw is not None else None
        sec_poc_val = pinfo.get("sec_poc_px") if pinfo.get("sec_poc_px") is not None else pinfo.get("hvn2_px")
        sec_poc_sfx = f", sec_POC={sec_poc_val}" if sec_poc_val is not None else ""
        lvn_sfx = f", LVN={pinfo.get('lvn_px')}" if pinfo.get("lvn_px") is not None else ""
        va_hold_sfx = format_va_hold_suffix(pinfo, va_regime=fb_vp.get("va_regime") or pinfo.get("va_regime"))
        vp_src_sfx = f", vp_source={pinfo['vp_source']}" if pinfo.get("vp_source") else ""
        extras.append(
            f"vol_profile(POC={pinfo.get('poc_px')}[{fmt_signed(poc_d)}%]{sec_poc_sfx}{lvn_sfx}, "
            f"VAL={pinfo.get('val_px')}[{fmt_signed(val_d)}%], "
            f"VAH={pinfo.get('vah_px')}[{fmt_signed(vah_d)}%], "
            f"va_pos={va_pos}%{va_hold_sfx}{vp_src_sfx})"
        )
    if sv_summary := format_session_vwap_summary(pinfo, cur_px=px_f, fmt_px_fn=_fmt_px):
        extras.append(f"session_vwap={sv_summary}")
    if sess_parts := format_session_extremes_parts(pinfo):
        extras.append(f"sessions({', '.join(sess_parts)})")

    span_7d_h = safe_pos_float(pinfo.get("span_7d_hours"), default=None)
    is_partial_7d = bool(pinfo.get("is_partial_7d")) or (span_7d_h is not None and span_7d_h < 120.0)
    is_sub24h_dup_7d = is_partial_7d or (
        "chg_24h_pct" not in pinfo
        and pinfo.get("low_24h") is not None
        and (pinfo.get("low_7d"), pinfo.get("high_7d")) == (pinfo.get("low_24h"), pinfo.get("high_24h"))
    )
    is_non_btc_crypto = not is_equity_or_tokenized_symbol(sym) and canonical_base_asset(sym) != "BTC"
    if is_non_btc_crypto:
        vs_btc_parts: List[str] = []
        if btc_entry:
            for h_lbl, h_k in (
                ("15m", "chg_15m_pct"), ("1h", "chg_1h_pct"),
                ("6h", "chg_6h_pct"), ("24h", "chg_24h_pct"), ("7d", "chg_7d_pct"),
            ):
                if h_lbl == "7d" and is_sub24h_dup_7d:
                    continue
                if h_k in pinfo and h_k in btc_entry:
                    a_v = safe_float(pinfo.get(h_k), None)
                    b_v = safe_float(btc_entry.get(h_k), None)
                    if a_v is not None and b_v is not None:
                        vs_btc_parts.append(f"{h_lbl}={fmt_signed(a_v - b_v)}%")
        if vs_btc_parts:
            extras.append(f"vs_btc({', '.join(vs_btc_parts)})")
        cross_parts = (
            ([f"pos={pinfo['alt_btc_pos_24h_pct']}%"] if pinfo.get("alt_btc_pos_24h_pct") is not None else [])
            + [
                f"{c_lbl}={fmt_signed(pinfo.get(c_k))}%"
                for c_lbl, c_k in (("dd_high", "alt_btc_dd_24h_high_pct"), ("bounce_low", "alt_btc_bounce_24h_low_pct"))
                if pinfo.get(c_k) is not None
            ]
            + [
                f"{sc_lbl}={safe_float(pinfo.get(sc_k), 0.0):.2f}"
                for sc_lbl, sc_k in (("beta", "beta_btc_24h"), ("corr", "corr_btc_24h"))
                if pinfo.get(sc_k) is not None
            ]
        )
        if cross_parts:
            extras.append(f"alt_btc_24h({', '.join(cross_parts)})")

    if pinfo.get("bottom_structure_6h"):
        _, b_parts = format_bottom_structure_6h_parts(
            pinfo,
            prior_label="prior_low",
            px_prefix="",
        )
        if b_parts:
            extras.append(f"bottom_6h({', '.join(b_parts)})")

    if "chg_7d_pct" in pinfo and not is_sub24h_dup_7d:
        extras.append(
            f"7d={fmt_signed(pinfo.get('chg_7d_pct'))}% "
            f"{_format_horizon_range(pinfo, px_f, '7d')}"
        )
    is_dup_swing = is_partial_7d or (
        is_sub24h_dup_7d
        and not pinfo.get("recent_swings_7d")
        and (pinfo.get("swing_low_px"), pinfo.get("swing_high_px")) == (pinfo.get("low_24h"), pinfo.get("high_24h"))
    )
    if pinfo.get("swing_low_px") is not None and pinfo.get("swing_high_px") is not None and not is_dup_swing:
        sw_ral = safe_float(pinfo.get("rally_from_swing_low_pct"), 0.0)
        sw_pb = safe_float(pinfo.get("pullback_from_swing_high_pct"), 0.0)
        maj_parts: List[str] = []
        if pinfo.get("rally_from_major_low_pct") is not None and pinfo.get("major_swing_low_px") is not None:
            maj_parts.append(
                f"major_low={pinfo.get('major_swing_low_px')}"
                f"(rally={fmt_signed(pinfo.get('rally_from_major_low_pct'))}%)"
            )
        if pinfo.get("pullback_from_major_high_pct") is not None and pinfo.get("major_swing_high_px") is not None:
            maj_parts.append(
                f"major_high={pinfo.get('major_swing_high_px')}"
                f"(pb={fmt_signed(pinfo.get('pullback_from_major_high_pct'))}%)"
            )
        maj_sfx = f", {', '.join(maj_parts)}" if maj_parts else ""
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
            f"rally_low={fmt_signed(sw_ral)}%, pullback_high={fmt_signed(sw_pb)}%{maj_sfx}){hist_sfx}"
        )
    if pinfo.get("vol_1h_pct") is not None and safe_pos_float(pinfo.get("atr_1h_pct"), default=None) is None:
        extras.append(f"vol_1h={pinfo.get('vol_1h_pct')}%")
    if atr_parts := format_atr_parts(pinfo):
        extras.append(f"atr({', '.join(atr_parts)})")
    if (kv := safe_float(pinfo.get("kalman_vel"), None)) is not None and abs(kv) >= 1e-9:
        kv_s = fmt_signed(kv, 5) if abs(kv) >= 5e-6 else fmt_signed(kv, 8, strip_zeros=True)
        extras.append(f"kalman_vel={kv_s}")
    if kin_parts := format_kinematic_parts(pinfo, compact=True):
        extras.append(f"kinematics({', '.join(kin_parts)})")
    if pinfo.get("long_trend_dir"):
        lt_str = f"long_trend={str(pinfo.get('long_trend_dir')).upper()}({pinfo.get('long_trend_hours')}h"
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
    return {
        "sym": sym,
        "sib_lbl": sib_lbl,
        "sib_syms": sib_syms,
        "px_str": px_str,
        "is_obs_sym": is_obs_sym,
        "is_compact_obs": False,
        "obs_lbl": obs_lbl,
        "eq_qmeta_sfx": eq_qmeta_sfx,
        "full_extras": extras,
        "compact_cols": {},
        "legacy_line": f"  * {sym}{sib_lbl}: Price={px_str}{obs_lbl}{eq_qmeta_sfx}{extra_str}",
    }


def _format_market_prices_section(
    prices_and_trends: Dict[str, Any],
    asset_scope: Any = "all",
    *,
    compact_observe_only: bool = False,
    active_symbols: Optional[set[str]] = None,
) -> List[str]:
    """Format standalone Section 3 (Live Market Prices, Multi-Horizon Trends & 24h/7d Evolution)."""
    view = evaluate_market_prices_prompt_view(prices_and_trends)
    btc_entry = view["btc_entry"]
    obs_syms = observe_only_registry_symbols()
    scope_spec = resolve_scope_presentation(asset_scope)
    lines = [f"\n### 3. MARKET PRICES & TRENDS ({scope_spec.market_prices_hdr}):"]
    for sym, pinfo, sib_specs in view["rows"]:
        rec = _build_market_price_symbol_record(
            sym,
            pinfo,
            sib_specs,
            btc_entry=btc_entry,
            obs_syms=obs_syms,
            compact_observe_only=compact_observe_only,
            active_symbols=active_symbols,
        )
        lines.append(rec["legacy_line"])
    return lines


def _format_3b_source_ages_suffix(
    minfo: Dict[str, Any],
    *,
    now_ts: Optional[float],
    intervals: Dict[str, float],
    ob_only: bool = False,
    include_deriv: bool = True,
    is_equity: bool = False,
    no_flow: bool = False,
) -> str:
    explicit_now = safe_pos_float(now_ts, default=None)
    ref_now = resolve_now_ts(explicit_now)
    ob_itv = (
        max(intervals["whale_interval_sec"], intervals["orderbook_interval_sec"] * 6.0)
        if is_equity
        else intervals["orderbook_interval_sec"]
    )
    if ob_only or (is_equity and no_flow):
        specs: tuple[tuple[str, str, float], ...] = (("ob", "_ob_raw_ts", ob_itv),)
    else:
        flow_lbl = "rth_flow" if is_equity else "whale"
        specs = (
            ("ob", "_ob_raw_ts", ob_itv),
            *((("deriv", "_deriv_ts", intervals["derivatives_interval_sec"]),) if include_deriv else ()),
            (flow_lbl, "_whale_ts", intervals["whale_interval_sec"]),
        )
    eq_market_closed = bool(is_equity and not is_us_rth_open(ref_now))
    stale_parts: List[str] = []
    for lbl, ts_k, itv_s in specs:
        raw_t = minfo.get(ts_k)
        if raw_t is None and lbl == "ob":
            raw_t = minfo.get("_ob_ts")
        ts_v = normalize_epoch_sec(raw_t, default=0.0) or 0.0
        if ts_v <= 0.0:
            continue
        age_s = ref_now - ts_v
        if explicit_now is None and not (0.0 <= age_s <= 7.0 * 86400.0):
            continue
        age_clamped = max(0.0, age_s)
        if age_clamped > itv_s * TELEMETRY_SOURCE_STALE_INTERVAL_MULT:
            itv_lbl = format_interval_sec_label(itv_s)
            tag = "[market_closed]" if eq_market_closed else "[stale]"
            stale_parts.append(f"{lbl}={format_age_sec_compact(age_clamped)}/{itv_lbl}{tag}")
    if not stale_parts:
        return ""
    return f", source_ages({', '.join(stale_parts)})"


def _build_microstructure_symbol_record(
    row: Dict[str, Any],
    rendered_agg_spot: set[tuple[str, str]],
    *,
    now_ts: Optional[float],
    derived_3b_itvs: Dict[str, float],
    obs_syms_3b: set[str],
    compact_observe_only: bool,
    active_symbols: Optional[set[str]],
) -> Dict[str, Any]:
    """Build reusable book/liq and flow/deriv prompt parts for both single-line and unified-card views."""
    sym = row["sym"]
    sib_3b = row["sib_3b"]
    b_canon_3b = row["b_canon"]
    minfo = row["minfo"]
    cur_px = row["cur_px"]
    ob_mid = row["ob_mid"]
    ob_ref_px = row["ob_ref_px"]
    b_bid = row["b_bid"]
    b_ask = row["b_ask"]
    caps = row["caps"]
    sym_is_tok_xstock_3b = row["sym_is_tok"]
    shared_perp_sym = row["shared_perp_sym"]
    ob_age_3b = row["ob_age_s"]
    ob_is_frozen = row["ob_is_frozen"]
    has_perp_cvd = row["has_perp_cvd"]
    no_flow = row["no_flow"]
    show_spot_cvd = row["show_spot_cvd"]
    show_whale_long = row["show_whale_long"]
    show_whale_long_delta = row["show_whale_long_delta"]
    deriv_mode = row["deriv_mode"]
    tok_spot_only = row["tok_spot_only"]

    sym_3b_lbl = f"{sym} ({', '.join(sib_3b)})" if sib_3b else sym
    _, is_compact_obs_3b = _is_compact_observe_only_symbol(
        sym,
        obs_syms_3b,
        compact_observe_only=compact_observe_only,
        active_symbols=active_symbols,
    )
    if is_compact_obs_3b:
        c3b_parts: List[str] = []
        if ob_is_frozen:
            c3b_parts.append(f"orderbook=STALE_FROZEN({ob_age_3b:.0f}s)")
        elif tok_spot_only or no_flow or shared_perp_sym is not None:
            if minfo.get("ob_imbalance") is not None:
                c3b_parts.append(f"{caps.book_source_prefix}ob_imbalance={minfo.get('ob_imbalance')}")
        if shared_perp_sym is not None:
            c3b_parts.append(f"perp_whale_flow=shared_with_{shared_perp_sym}")
        elif not tok_spot_only and not no_flow:
            if minfo.get("taker_buy_sell_ratio") is not None:
                c3b_parts.append(f"taker_ratio={minfo.get('taker_buy_sell_ratio')}")
            if has_perp_cvd:
                c3b_parts.append(f"cvd_15m={_fmt_signed_usd(safe_float(minfo.get('cvd_15m_usd'), 0.0))}")
            if deriv_mode == "cboe_options":
                if minfo.get("oi_1h_chg_pct") is not None:
                    c3b_parts.append(f"oi_1h_chg={minfo.get('oi_1h_chg_pct')}%")
            elif deriv_mode != "none":
                if (f_ann_c := safe_float(minfo.get("funding_annualized_pct"), None)) is not None:
                    c3b_parts.append(f"funding_ann={fmt_signed(f_ann_c, 1)}%")
                if minfo.get("oi_1h_chg_pct") is not None:
                    c3b_parts.append(f"oi_1h_chg={minfo.get('oi_1h_chg_pct')}%")
                if caps.has_crypto_perpetuals and (b_bps_c := safe_float(minfo.get("perp_spot_basis_bps"), None)) is not None:
                    c3b_parts.append(f"perp_basis={fmt_signed(b_bps_c, 1)}bps")
        if not ob_is_frozen and (sp_c := safe_float(minfo.get("spot_cvd_15m_usd"), None)) is not None and minfo.get("spot_cvd_source") == "real_spot":
            c3b_parts.append(f"spot_cvd_15m={_fmt_signed_usd(sp_c)}[real_spot]")
        if not ob_is_frozen and (rv1h := safe_pos_float(minfo.get("rvol_1h"), default=None)) is not None:
            c3b_parts.append(f"rvol_1h={rv1h:.2f}x")
        obs_str = ", ".join(c3b_parts)
        return {
            "sym": sym,
            "sym_lbl": sym_3b_lbl,
            "is_compact_obs": True,
            "shared_perp_sym": shared_perp_sym,
            "book_str": "",
            "flow_str": "",
            "obs_flow_str": obs_str,
            "legacy_line": f"  * {sym_3b_lbl} [OBSERVE-ONLY]: {obs_str}" if obs_str else "",
        }

    bw_px = minfo.get("bid_wall_px")
    aw_px = minfo.get("ask_wall_px")
    bw_px_f = safe_float(bw_px, 0.0)
    aw_px_f = safe_float(aw_px, 0.0)
    bw_dist = f", dist={fmt_signed((bw_px_f - ob_ref_px) / ob_ref_px * 100.0)}%" if ob_ref_px > 0 and bw_px_f > 0 else ""
    aw_dist = f", dist={fmt_signed((aw_px_f - ob_ref_px) / ob_ref_px * 100.0)}%" if ob_ref_px > 0 and aw_px_f > 0 else ""
    spread_sfx = ""
    if not ob_is_frozen and minfo.get("spread_bps") is not None:
        ba_lbl = f" (bid={b_bid}/ask={b_ask})" if b_bid > 0 and b_ask > 0 else ""
        ob_mid_lbl = f", ob_mid={round(ob_mid, 6)}" if ob_mid > 0 else ""
        spread_sfx = f", spread={minfo.get('spread_bps')}bps{ba_lbl}{ob_mid_lbl}"

    d1_b = safe_float(minfo.get("depth_1pct_bid_usd"), 0.0)
    d1_a = safe_float(minfo.get("depth_1pct_ask_usd"), 0.0)
    b_dep_v = safe_float(minfo.get("bid_depth_usd"), 0.0)
    a_dep_v = safe_float(minfo.get("ask_depth_usd"), 0.0)
    b_span_pct = safe_pos_float(minfo.get("book_span_pct"), default=0.0) or 0.0
    bw_cap_tag = format_book_window_cap_tag(b_span_pct)
    if bw_cap_tag and b_span_pct < 0.95 and abs(d1_b - b_dep_v) < 1.0 and abs(d1_a - a_dep_v) < 1.0:
        depth_1pct_sfx = f", {bw_cap_tag}, slip_2k={minfo.get('est_slip_2k_bps')}bps"
    elif bw_cap_tag and (d1_b > 0.0 or d1_a > 0.0):
        depth_1pct_sfx = (
            f", {bw_cap_tag}, "
            f"depth_1pct=(bid={_fmt_liq_usd(d1_b)}/ask={_fmt_liq_usd(d1_a)}, slip_2k={minfo.get('est_slip_2k_bps')}bps)"
        )
    else:
        depth_1pct_sfx = (
            f", depth_1pct=(bid={_fmt_liq_usd(d1_b)}/ask={_fmt_liq_usd(d1_a)}, slip_2k={minfo.get('est_slip_2k_bps')}bps)"
            if (d1_b > 0.0 or d1_a > 0.0)
            else ""
        )

    cvd_sfx = ""
    if has_perp_cvd:
        cvd_u = safe_float(minfo.get("cvd_15m_usd"), 0.0)
        usd_s = _fmt_signed_usd(cvd_u)
        cvd_b = minfo.get("cvd_15m_base")
        cvd_b_s = fmt_signed(cvd_b, 3, strip_zeros=True) if cvd_b is not None else ""
        cvd_sfx = (
            f", cvd_15m={cvd_b_s} {b_canon_3b} ({usd_s})"
            if cvd_b is not None
            else f", cvd_15m={usd_s}"
        )
        if mh_parts := format_flow_horizon_parts(minfo, _fmt_signed_usd):
            cvd_sfx += f", flow_horizons({', '.join(mh_parts)})"
    if show_spot_cvd:
        sp_cvd = safe_float(minfo.get("spot_cvd_15m_usd"), 0.0)
        cvd_sfx += f", spot_cvd_15m={_fmt_signed_usd(sp_cvd)}{format_spot_cvd_source_tag(minfo)}"
    if (
        shared_perp_sym is None
        and not ob_is_frozen
        and (agg_sp_part := format_agg_spot_cvd_part(minfo, _fmt_signed_usd, _fmt_liq_usd))
        and (b_canon_3b, agg_sp_part) not in rendered_agg_spot
    ):
        rendered_agg_spot.add((b_canon_3b, agg_sp_part))
        cvd_sfx += f", {agg_sp_part}"
    if sp_vol_parts := format_spot_volume_rvol_parts(minfo, _fmt_liq_usd):
        cvd_sfx += f", {', '.join(sp_vol_parts)}"

    basis_sfx = ""
    basis_parts: List[str] = []
    if caps.has_crypto_perpetuals and (b_bps := safe_float(minfo.get("perp_spot_basis_bps"), None)) is not None:
        b_d15 = safe_float(minfo.get("basis_delta_15m_bps"), None)
        b_d15_s = f"(d15m={fmt_signed(b_d15, 1)}bps)" if (b_d15 is not None and abs(b_d15) >= 0.1) else ""
        basis_parts.append(f"perp_basis={fmt_signed(b_bps, 1)}bps{b_d15_s}")
    if caps.has_crypto_perpetuals and (f_ann := safe_float(minfo.get("funding_annualized_pct"), None)) is not None:
        f_d1h = safe_float(minfo.get("funding_delta_1h_bps"), None)
        f_d1h_s = f"(d1h={fmt_signed(f_d1h)}bps)" if (f_d1h is not None and abs(f_d1h) >= 0.05) else ""
        basis_parts.append(f"funding_ann={fmt_signed(f_ann, 1)}%{f_d1h_s}")
    if (arb_bps := safe_float(minfo.get("cross_venue_arb_bps"), None)) is not None and minfo.get("cross_venue_peer"):
        basis_parts.append(f"arb_vs_{minfo['cross_venue_peer']}={fmt_signed(arb_bps, 1)}bps")
        if (net_b := safe_float(minfo.get("net_buy_edge_vs_peer_bps"), None)) is not None and minfo.get("best_buy_venue"):
            fee_basis = str(minfo.get("arb_fee_basis") or "maker").strip().lower()
            basis_parts.append(f"net_buy_edge={fmt_signed(net_b, 1)}bps[{fee_basis}](best_buy={minfo['best_buy_venue']})")
    if basis_parts:
        basis_sfx = f", {', '.join(basis_parts)}"

    liq_sfx = ""
    if caps.has_liquidation_clusters and minfo.get("long_liq_px") is not None and minfo.get("short_liq_px") is not None:
        liq_ref_px = ob_ref_px if (ob_ref_px > 0 and not ob_is_frozen) else cur_px
        liq_parts: List[str] = []
        for l_lbl, px_k, usd_k, dist_k in (
            ("long_liq", "long_liq_px", "long_liq_usd", "long_liq_dist_pct"),
            ("short_liq", "short_liq_px", "short_liq_usd", "short_liq_dist_pct"),
            ("deep_long", "long_liq_deep_px", "long_liq_deep_usd", "long_liq_deep_dist_pct"),
            ("deep_short", "short_liq_deep_px", "short_liq_deep_usd", "short_liq_deep_dist_pct"),
        ):
            if minfo.get(px_k) is None:
                continue
            l_px = safe_float(minfo.get(px_k), 0.0)
            if liq_ref_px > 0 and l_px > 0:
                l_d = ((l_px - liq_ref_px) / liq_ref_px) * 100.0
            else:
                l_d = safe_float(minfo.get(dist_k), 0.0)
            liq_parts.append(
                f"{l_lbl}=${minfo.get(px_k)}[{_fmt_liq_usd(minfo.get(usd_k))}, {fmt_signed(l_d)}%]"
            )
        liq_parts.append(f"src={minfo.get('liq_source') or 'model_est'}")
        liq_sfx = f", liq_clusters({', '.join(liq_parts)})"
    if caps.has_liquidation_clusters and (
        (rl_l := safe_float(minfo.get("realized_liq_15m_long_usd"), None)) is not None
        and (rl_s := safe_float(minfo.get("realized_liq_15m_short_usd"), None)) is not None
        and (rl_l > 0.0 or rl_s > 0.0 or (safe_float(minfo.get("realized_liq_15m_events"), 0.0) or 0.0) > 0.0)
    ):
        rl_n = safe_int(minfo.get("realized_liq_15m_events"), 0) or 0
        rl_n_sfx = f", n={rl_n}" if rl_n > 0 else ""
        rl_src_tag = format_realized_liq_sources_tag(minfo, multi_only=True)
        rl_src_sfx = f", src={rl_src_tag}" if rl_src_tag else ""
        liq_sfx += f", realized_liq_15m(long={_fmt_liq_usd(rl_l)}, short={_fmt_liq_usd(rl_s)}{rl_n_sfx}{rl_src_sfx})"

    cd_sfx = ""
    if isinstance(cd_map := minfo.get("cycle_delta"), dict):
        cd_parts: List[str] = []
        if cd_map.get("dt_min") is not None:
            cd_parts.append(f"dt={cd_map['dt_min']}m")
        if (d_ob := safe_float(cd_map.get("ob_imbalance_delta"), None)) is not None and abs(d_ob) >= 0.0005:
            cd_parts.append(f"d_ob={fmt_signed(d_ob, 3)}")
        if shared_perp_sym is None:
            if (d_cvd := safe_float(cd_map.get("cvd_15m_delta_usd"), None)) is not None and abs(d_cvd) >= 1.0:
                cd_parts.append(f"d_cvd={_fmt_signed_usd(d_cvd)}")
            if (
                not no_flow
                and (d_tr := safe_float(cd_map.get("taker_ratio_delta"), None)) is not None
                and abs(d_tr) >= 0.05
            ):
                cd_parts.append(f"d_tr={fmt_signed(d_tr)}")
            if (
                show_whale_long_delta
                and (d_wl := safe_float(cd_map.get("whale_long_delta_pct"), None)) is not None
                and abs(d_wl) >= 0.005
            ):
                cd_parts.append(f"d_whale_long={fmt_signed(d_wl * 100.0, 1)}pp")
            if (d_oi := safe_float(cd_map.get("oi_delta_pct"), None)) is not None and abs(d_oi) >= 0.005:
                cd_parts.append(f"d_oi={fmt_signed(d_oi)}%")
        if len(cd_parts) > 1:
            cd_sfx = f", cycle_delta({', '.join(cd_parts)})"

    l1_oi_v = safe_float(minfo.get("native_l1_oi_usd"), None)
    oi_v = safe_float(minfo.get("oi_usd"), None)
    has_distinct_l1_oi = bool(l1_oi_v is not None and oi_v is not None and abs(l1_oi_v - oi_v) > 1.0)
    oi_disp = _fmt_liq_usd(oi_v) if oi_v is not None else str(minfo.get("oi_usd"))
    oi_lbl = f"{oi_disp} (l1_oi={_fmt_liq_usd(l1_oi_v)})" if has_distinct_l1_oi else oi_disp
    book_core = (
        f"orderbook=STALE_FROZEN({ob_age_3b:.0f}s; walls/imbalance/spot_cvd_15m suppressed)"
        if ob_is_frozen
        else (
            f"{caps.book_source_prefix}ob_imbalance={minfo.get('ob_imbalance')} "
            f"(bid_depth={_fmt_liq_usd(b_dep_v)}, ask_depth={_fmt_liq_usd(a_dep_v)}){depth_1pct_sfx}, "
            f"{caps.bid_zone_label}={bw_px} ({_fmt_liq_usd(minfo.get('bid_wall_usd'))}{bw_dist}), "
            f"{caps.ask_zone_label}={aw_px} ({_fmt_liq_usd(minfo.get('ask_wall_usd'))}{aw_dist})"
        )
    )

    if shared_perp_sym is not None:
        arb_only_parts = [p for p in basis_parts if p.startswith(("arb_vs_", "net_buy_edge="))]
        arb_sfx = f", {', '.join(arb_only_parts)}" if arb_only_parts else ""
        sib_sp_parts: List[str] = []
        if not ob_is_frozen and (sib_sp_cvd := safe_float(minfo.get("spot_cvd_15m_usd"), None)) is not None and minfo.get("spot_cvd_source") == "real_spot":
            sib_sp_parts.append(f"spot_cvd_15m={_fmt_signed_usd(sib_sp_cvd)}[real_spot]")
        if (
            not ob_is_frozen
            and (sib_agg_part := format_agg_spot_cvd_part(minfo, _fmt_signed_usd, _fmt_liq_usd))
            and (b_canon_3b, sib_agg_part) not in rendered_agg_spot
        ):
            rendered_agg_spot.add((b_canon_3b, sib_agg_part))
            sib_sp_parts.append(sib_agg_part)
        if not ob_is_frozen:
            sib_sp_parts.extend(format_spot_volume_rvol_parts(minfo, _fmt_liq_usd))
        sib_sp_sfx = f", {', '.join(sib_sp_parts)}" if sib_sp_parts else ""
        ob_age_sfx = _format_3b_source_ages_suffix(
            minfo, now_ts=now_ts, intervals=derived_3b_itvs, ob_only=True
        )
        book_str = f"{book_core}{spread_sfx}{ob_age_sfx}" if not ob_is_frozen else f"{book_core}{ob_age_sfx}"
        flow_bits = [p for p in (*sib_sp_parts, *arb_only_parts) if p]
        if cd_sfx:
            flow_bits.append(cd_sfx.lstrip(", "))
        flow_bits.append(f"perp_whale_flow=shared_with_{shared_perp_sym}")
        legacy_line = (
            f"  * {sym_3b_lbl}: {book_core}{arb_sfx}{cd_sfx}{ob_age_sfx}, perp_whale_flow=shared_with_{shared_perp_sym}"
            if ob_is_frozen
            else f"  * {sym_3b_lbl}: {book_core}{sib_sp_sfx}{arb_sfx}{spread_sfx}{cd_sfx}{ob_age_sfx}, perp_whale_flow=shared_with_{shared_perp_sym}"
        )
        return {
            "sym": sym,
            "sym_lbl": sym_3b_lbl,
            "is_compact_obs": False,
            "shared_perp_sym": shared_perp_sym,
            "book_str": book_str,
            "flow_str": ", ".join(flow_bits),
            "obs_flow_str": "",
            "legacy_line": legacy_line,
        }

    oi_1h_v = safe_float(minfo.get("oi_1h_chg_pct"), None)
    oi_1h_lbl = (oi_1h_v + 0.0) if oi_1h_v is not None else minfo.get("oi_1h_chg_pct")
    oi_15m_v = safe_float(minfo.get("oi_15m_chg_pct"), None)
    oi_15m_sfx = f", oi_15m_chg={fmt_signed(oi_15m_v)}%" if oi_15m_v is not None else ""
    if deriv_mode == "none":
        deriv_oi_sfx = ""
    elif deriv_mode == "cboe_options":
        deriv_oi_sfx = f", oi_usd={oi_lbl} (cboe_options_oi), oi_1h_chg={oi_1h_lbl}%"
    else:
        fr_v = safe_float(minfo.get("funding_rate"), default=None)
        pfr_v = safe_float(minfo.get("pred_funding_rate"), default=None)
        has_ann_fund = safe_float(minfo.get("funding_annualized_pct"), default=None) is not None
        fund_part = ""
        if has_ann_fund:
            if fr_v is not None and pfr_v is not None and abs(pfr_v - fr_v) > 1e-7:
                fund_part = f", pred_funding_8h={_fmt_plain_float(pfr_v)}"
        elif fr_v is not None:
            pred_sfx = (
                f" (pred={_fmt_plain_float(pfr_v)})"
                if pfr_v is not None and abs(pfr_v - fr_v) > 1e-9
                else ""
            )
            fund_part = f", funding={_fmt_plain_float(fr_v)}{pred_sfx}"
        flow_src_sfx = ", perp_flow_src=bn_perp" if has_distinct_l1_oi else ""
        deriv_oi_sfx = f"{fund_part}, oi_usd={oi_lbl}, oi_1h_chg={oi_1h_lbl}%{oi_15m_sfx}{flow_src_sfx}"
    src_age_sfx = _format_3b_source_ages_suffix(
        minfo,
        now_ts=now_ts,
        intervals=derived_3b_itvs,
        ob_only=sym_is_tok_xstock_3b,
        include_deriv=bool(deriv_oi_sfx),
        is_equity=bool(row.get("sym_is_eq")),
        no_flow=bool(no_flow),
    )
    wl_raw = safe_float(minfo.get("whale_long_pct"), default=None) if show_whale_long else None
    wl_prefix = (
        f"whale_long_pct={(wl_raw * 100.0 if wl_raw <= 1.0 else wl_raw):.1f}%, "
        if wl_raw is not None
        else (f"whale_long_pct={minfo.get('whale_long_pct')}, " if show_whale_long else "")
    )
    if tok_spot_only:
        tok_sp_parts: List[str] = []
        if not cvd_sfx and not ob_is_frozen and (tok_sp := safe_float(minfo.get("spot_cvd_15m_usd"), None)) is not None and minfo.get("spot_cvd_source") == "real_spot":
            tok_sp_parts.append(f"spot_cvd_15m={_fmt_signed_usd(tok_sp)}[real_spot]")
        if not cvd_sfx and not ob_is_frozen:
            tok_sp_parts.extend(format_spot_volume_rvol_parts(minfo, _fmt_liq_usd))
        tok_sp_sfx = f", {', '.join(tok_sp_parts)}" if tok_sp_parts else ""
        flow_3b_core = f"perp_derivatives=N/A[spot_tokenized_equity]{cvd_sfx}{tok_sp_sfx}"
    elif no_flow:
        flow_3b_core = f"{wl_prefix}intraday_rth_flow=N/A[no_fresh_vol_bars]{cvd_sfx}"
    else:
        bv_s = _fmt_liq_usd(minfo.get("taker_buy_vol_usd"))
        sv_s = _fmt_liq_usd(minfo.get("taker_sell_vol_usd"))
        flow_3b_core = (
            f"{wl_prefix}"
            f"taker_ratio={minfo.get('taker_buy_sell_ratio')} "
            f"(buy_v={bv_s}/sell_v={sv_s}){cvd_sfx}"
        )
    book_str = (
        f"{book_core}{liq_sfx}"
        if ob_is_frozen
        else f"{book_core}{spread_sfx}{liq_sfx}"
    )
    flow_str = f"{flow_3b_core}{deriv_oi_sfx}{basis_sfx}{cd_sfx}{src_age_sfx}"
    legacy_line = (
        f"  * {sym_3b_lbl}: {book_core}, {flow_3b_core}{deriv_oi_sfx}{basis_sfx}{liq_sfx}{cd_sfx}{src_age_sfx}"
        if ob_is_frozen
        else f"  * {sym_3b_lbl}: {book_core}, {flow_3b_core}{deriv_oi_sfx}{basis_sfx}{spread_sfx}{liq_sfx}{cd_sfx}{src_age_sfx}"
    )
    return {
        "sym": sym,
        "sym_lbl": sym_3b_lbl,
        "is_compact_obs": False,
        "shared_perp_sym": None,
        "book_str": book_str,
        "flow_str": flow_str,
        "obs_flow_str": "",
        "legacy_line": legacy_line,
    }


def _format_microstructure_section(
    microstructure: Dict[str, Any],
    prices_and_trends: Dict[str, Any],
    *,
    now_ts: Optional[float] = None,
    asset_scope: Any = "all",
    compact_observe_only: bool = False,
    active_symbols: Optional[set[str]] = None,
) -> List[str]:
    """Format standalone Section 4 (Orderbook Walls, Derivatives & Whale Flow)."""
    scope_spec = resolve_scope_presentation(asset_scope)
    lines = [scope_spec.microstructure_hdr]
    derived_3b_itvs = resolve_derived_telemetry_intervals()
    obs_syms_3b = observe_only_registry_symbols() if compact_observe_only else set()
    rendered_agg_spot: set[tuple[str, str]] = set()
    for row in evaluate_microstructure_prompt_view(microstructure, prices_and_trends, now_ts=now_ts):
        rec = _build_microstructure_symbol_record(
            row,
            rendered_agg_spot,
            now_ts=now_ts,
            derived_3b_itvs=derived_3b_itvs,
            obs_syms_3b=obs_syms_3b,
            compact_observe_only=compact_observe_only,
            active_symbols=active_symbols,
        )
        if rec["legacy_line"]:
            lines.append(rec["legacy_line"])
    return lines


def _format_unified_markets_and_bellwethers_sections(
    prices_and_trends: Dict[str, Any],
    microstructure: Dict[str, Any],
    *,
    now_ts: Optional[float] = None,
    asset_scope: Any = "all",
    compact_observe_only: bool = True,
    active_symbols: Optional[set[str]] = None,
    held_symbols: Optional[set[str]] = None,
    open_order_symbols: Optional[set[str]] = None,
    priority_symbols: Optional[set[str]] = None,
) -> List[str]:
    """Format Section 3 (Per-Symbol Unified Price/Book/Flow Cards) and Section 4 (Observe-Only Bellwether Table)."""
    scope_spec = resolve_scope_presentation(asset_scope)
    obs_syms = observe_only_registry_symbols()
    obs_syms_3b = obs_syms if compact_observe_only else set()
    prio_u: set[str] = set()
    for s in (priority_symbols or ()):
        if s:
            prio_u.add(str(s).strip().upper())
            if sk_p := clean_symbol_key(s):
                prio_u.add(sk_p)
    held_u: set[str] = set()
    for s in (held_symbols or ()):
        if s:
            held_u.add(str(s).strip().upper())
            if sk_h := clean_symbol_key(s):
                held_u.add(sk_h)
    open_u: set[str] = set()
    for s in (open_order_symbols or ()):
        if s:
            open_u.add(str(s).strip().upper())
            if sk_o := clean_symbol_key(s):
                open_u.add(sk_o)

    p_records: Dict[str, Dict[str, Any]] = {}
    if prices_and_trends:
        p_view = evaluate_market_prices_prompt_view(prices_and_trends)
        btc_entry = p_view["btc_entry"]
        for sym, pinfo, sib_specs in p_view["rows"]:
            p_records[sym] = _build_market_price_symbol_record(
                sym,
                pinfo,
                sib_specs,
                btc_entry=btc_entry,
                obs_syms=obs_syms,
                compact_observe_only=compact_observe_only,
                active_symbols=active_symbols,
            )

    m_records: Dict[str, Dict[str, Any]] = {}
    if microstructure:
        derived_itvs = resolve_derived_telemetry_intervals()
        rendered_agg_spot: set[tuple[str, str]] = set()
        for row in evaluate_microstructure_prompt_view(microstructure, prices_and_trends, now_ts=now_ts):
            mrec = _build_microstructure_symbol_record(
                row,
                rendered_agg_spot,
                now_ts=now_ts,
                derived_3b_itvs=derived_itvs,
                obs_syms_3b=obs_syms_3b,
                compact_observe_only=compact_observe_only,
                active_symbols=active_symbols,
            )
            m_records[mrec["sym"]] = mrec

    tradable_syms: List[str] = []
    obs_only_syms: List[str] = []
    for sym, prec in p_records.items():
        if prec["is_compact_obs"]:
            obs_only_syms.append(sym)
        else:
            tradable_syms.append(sym)

    def _sym_matches_group(s: str, grp: set[str]) -> bool:
        if not grp:
            return False
        su_m = s.strip().upper()
        sk_m = clean_symbol_key(s) or su_m
        if su_m in grp or sk_m in grp:
            return True
        prec_m = p_records.get(s)
        if prec_m:
            for sib in (prec_m.get("sib_syms") or ()):
                sib_u = str(sib).strip().upper()
                sib_k = clean_symbol_key(sib) or sib_u
                if sib_u in grp or sib_k in grp:
                    return True
        return False

    def _tradable_sort_key(s: str) -> tuple[int, str]:
        su = s.strip().upper()
        if _sym_matches_group(s, prio_u):
            return (0, su)
        if not scope_spec.is_equities and not is_equity_or_tokenized_symbol(s) and canonical_base_asset(s) == "BTC":
            return (1, su)
        if _sym_matches_group(s, held_u):
            return (2, su)
        if _sym_matches_group(s, open_u):
            return (3, su)
        return (4, su)

    tradable_syms.sort(key=_tradable_sort_key)

    lines: List[str] = []
    used_micro: set[str] = set()
    if tradable_syms or any(not mr["is_compact_obs"] for mr in m_records.values()):
        s3_hdr = (
            "\n### 3. TRADABLE US EQUITIES (UNIFIED PRICE, VOLUME ZONES & FLOW):"
            if scope_spec.is_equities
            else "\n### 3. TRADABLE MARKETS (UNIFIED PRICE, BOOK & FLOW \u2014 ALL COINS):"
        )
        lines.append(s3_hdr)
        for sym in tradable_syms:
            prec = p_records[sym]
            lines.append(f"  * {sym}{prec['sib_lbl']}: Price={prec['px_str']}{prec['obs_lbl']}{prec['eq_qmeta_sfx']}")
            if prec["full_extras"]:
                lines.append(f"    - Trend: {' | '.join(prec['full_extras'])}")
            mrec = m_records.get(sym)
            if mrec is not None:
                used_micro.add(sym)
                if mrec["book_str"]:
                    lines.append(f"    - Book: {mrec['book_str']}")
                if mrec["flow_str"]:
                    lines.append(f"    - Flow: {mrec['flow_str']}")
            for m_sym, cand_m in sorted(m_records.items()):
                if m_sym in used_micro or cand_m["is_compact_obs"]:
                    continue
                if cand_m["shared_perp_sym"] == sym or m_sym in prec["sib_syms"]:
                    used_micro.add(m_sym)
                    lines.append(f"  * {cand_m['sym_lbl']} (sibling of {sym}):")
                    if cand_m["book_str"]:
                        lines.append(f"    - Book: {cand_m['book_str']}")
                    if cand_m["flow_str"]:
                        lines.append(f"    - Flow: {cand_m['flow_str']}")
        for m_sym, mrec in sorted(m_records.items()):
            if m_sym in used_micro or mrec["is_compact_obs"]:
                continue
            used_micro.add(m_sym)
            lines.append(f"  * {mrec['sym_lbl']}:")
            if mrec["book_str"]:
                lines.append(f"    - Book: {mrec['book_str']}")
            if mrec["flow_str"]:
                lines.append(f"    - Flow: {mrec['flow_str']}")

    for m_sym, mrec in sorted(m_records.items()):
        if m_sym not in used_micro and mrec["is_compact_obs"] and m_sym not in obs_only_syms:
            obs_only_syms.append(m_sym)

    if obs_only_syms:
        def _obs_sort_key(s: str) -> tuple[int, str]:
            b = canonical_base_asset(s)
            if b == "ETH":
                return (0, s)
            if b == "SOL":
                return (1, s)
            return (2, s)

        obs_only_syms.sort(key=_obs_sort_key)
        lines.extend([
            "\n### 4. MARKET BELLWETHERS (OBSERVE-ONLY):",
            "  | Symbol | Price | 15m | 1h | 6h | 24h | Pos24h | Context | Flow & Deriv |",
            "  |---|---|---|---|---|---|---|---|---|",
        ])
        for sym in obs_only_syms:
            prec = p_records.get(sym)
            cc = prec["compact_cols"] if prec else {}
            mrec = m_records.get(sym)
            flow_col = (mrec["obs_flow_str"] if mrec and mrec["obs_flow_str"] else "-")
            lines.append(
                f"  | {sym} | {cc.get('px_str', '-')} | {cc.get('15m', '-')} | {cc.get('1h', '-')} | "
                f"{cc.get('6h', '-')} | {cc.get('24h', '-')} | {cc.get('pos_24h', '-')} | "
                f"{cc.get('context', '-')} | {flow_col} |"
            )
    return lines


def _format_tradable_position_line(
    pos: Dict[str, Any],
    max_usd: float = 2000.0,
    *,
    min_usd: float = 0.0,
    include_static_specs: bool = True,
    include_free_quote: bool = True,
) -> str:
    """Format a single tradable position line for Section 3C from its evaluated view model."""
    pv = (
        pos
        if isinstance(pos, dict) and pos.get("_is_tradable_pos_view") is True
        else evaluate_tradable_position_prompt_view(pos, max_usd=max_usd, min_usd=min_usd)
    )
    t_extra = ""
    lot_base_lbl = pv["lot_base_lbl"]
    for anc in pv["anchors"]:
        pct_val = anc["pct_val"]
        if anc["lot_harvested"]:
            pct_sfx = f" (chg_vs_last_buy={pct_val}%, lot_harvested)"
        else:
            pct_sfx = f" ({anc['pct_lbl']}={pct_val}%)" if pct_val is not None else ""
        t_extra += f", {anc['lbl']}={anc['px']}{pct_sfx}"
    if shown_lots := pv["shown_lifo_lots"]:
        lot_parts = [
            f"{_fmt_plain_float(l_q, compact=True)}@{_fmt_plain_float(l_px)}"
            + (f"({fmt_signed(l_pnl)}%)" if l_pnl is not None else "")
            for l_q, l_px, l_pnl in shown_lots
        ]
        t_extra += f", open_lifo_lots=[{', '.join(lot_parts)}]"
        if (prof := pv["profitable_lifo"]) is not None:
            prof_usd, prof_q, prof_avg_pct = prof
            t_extra += (
                f", profitable_lifo_usd=${prof_usd:.2f}"
                f"({_fmt_plain_float(prof_q, compact=True)}_{lot_base_lbl}@{fmt_signed(prof_avg_pct)}%)"
            )
        if (older_q := pv["older_core_qty"]) is not None:
            t_extra += f", older_core_qty={_fmt_plain_float(older_q)}_{lot_base_lbl}"
    if (lifo_vwap := pv["lifo_rem_vwap"]) is not None:
        vwap, vwap_pnl = lifo_vwap
        vwap_sfx = f" (pnl_vs_lifo={fmt_signed(vwap_pnl)}%)" if vwap_pnl is not None else ""
        t_extra += f", lifo_avg={vwap}{vwap_sfx}"
    if (lifo_p := pv["lifo_realized_pnl_usd"]) is not None:
        t_extra += f", lifo_realized_pnl={format_signed_usd(lifo_p, decimals=2)}"
    lim_parts: List[str] = []
    if include_static_specs:
        taker_bps = pv["taker_bps"]
        maker_bps = pv["maker_bps"]
        leg_fee_bps = max(maker_bps, taker_bps)
        sp_part = f", spread={pv['spread_bps']}bps" if pv["spread_bps"] is not None else ""
        t_extra += (
            f", venue_fee_per_leg={leg_fee_bps:g}bps "
            f"(tier={pv['fee_tier']}, maker={maker_bps:g}bps/taker={taker_bps:g}bps{sp_part})"
        )
        if pv["min_notional_usd"] is not None:
            lim_parts.append(f"min_notional=${pv['min_notional_usd']:g}")
    for lim_pfx, lim_val in pv["limit_items"]:
        if not include_static_specs and lim_pfx in ("min_qty=", "step=", "tick="):
            continue
        disp_lim = _fmt_plain_float(lim_val) if lim_pfx in ("min_qty=", "step=", "tick=") else lim_val
        lim_parts.append(f"{lim_pfx}{disp_lim}")
    if lim_parts:
        t_extra += f", limits({', '.join(lim_parts)})"
    fb_val, val_usd = pv["free_base"], pv["value_usd"]
    fb_disp = _fmt_plain_float(fb_val) if fb_val is not None else None
    fb_str = (
        f"{fb_disp} (${val_usd} USD)"
        if fb_disp is not None and val_usd is not None
        else (fb_disp if fb_disp is not None else "N/A")
    )
    if (lb_q := pv["locked_base_qty"]) is not None:
        lb_u = pv["locked_base_usd"]
        lb_u_s = f" (${lb_u:.2f} USD)" if lb_u is not None else ""
        fb_str += f", locked_base_qty={_fmt_plain_float(lb_q)}{lb_u_s}"
    fq_part = ""
    if include_free_quote:
        fq_val = pv["free_quote"]
        fq_bd_items = pv["free_quote_breakdown_items"]
        fq_bd_s = (
            f" ({', '.join(f'{bk}=${bv:,.2f}' for bk, bv in fq_bd_items)})"
            if fq_bd_items
            else ""
        )
        fq_str = f"${fq_val}{fq_bd_s}" if fq_val is not None else "N/A"
        fq_part = f"free_{pv['quote_lbl']}={fq_str}, "
    if (lq_u := pv["locked_quote_usd"]) is not None:
        fq_part += f"locked_quote_usd=${lq_u:.2f}, "
    return (
        f"    - [{pv['venue']}] {pv['disp_symbol']}: free_{pv['disp_base_lbl']}={fb_str}, "
        f"{fq_part}"
        f"px={pv['px_disp']}{t_extra}"
    )


def _format_venue_specs_table(
    tradable_positions: List[Dict[str, Any]], *, tag: str = ""
) -> List[str]:
    """Format static venue fees and order precision specs as a compact Markdown table (G2 delta-managed)."""
    if not tradable_positions:
        return []
    out = [
        f"  * Venue Specs & Fees (static maker/taker bps & precision){tag}:",
        "    | Venue | Symbol | Tier | Fee/Leg (M/T bps) | MinNotional | MinQty | Step | Tick |",
        "    |---|---|---|---|---|---|---|---|",
    ]
    for pv in tradable_positions:
        if not isinstance(pv, dict):
            continue
        maker_bps = pv.get("maker_bps", 0.0)
        taker_bps = pv.get("taker_bps", 0.0)
        leg_bps = max(maker_bps, taker_bps)
        lim_map = {pfx: val for pfx, val in (pv.get("limit_items") or ())}
        min_not = f"${pv['min_notional_usd']:g}" if pv.get("min_notional_usd") is not None else "-"
        min_q = _fmt_plain_float(lim_map["min_qty="]) if "min_qty=" in lim_map else "-"
        step_v = _fmt_plain_float(lim_map["step="]) if "step=" in lim_map else "-"
        tick_v = _fmt_plain_float(lim_map["tick="]) if "tick=" in lim_map else "-"
        out.append(
            f"    | {pv.get('venue', '-')} | {pv.get('disp_symbol', '-')} | {pv.get('fee_tier', 'STANDARD')} | "
            f"{leg_bps:g} ({maker_bps:g}/{taker_bps:g}) | {min_not} | {min_q} | {step_v} | {tick_v} |"
        )
    return out


def _format_open_limit_orders_table(
    clean_open_orders: List[Any],
    *,
    active_plans: Optional[List[Dict[str, Any]]] = None,
    return_linked: bool = False,
) -> Any:
    """Format active resting exchange orders as a compact Markdown table, optionally linking matching RESTING plans."""
    dict_orders = [o for o in clean_open_orders if isinstance(o, dict)]
    if not dict_orders:
        res_lines = [
            f"  * Active Open Orders on Exchange (Locking Liquidity) \u2014 open_limit_orders: {clean_open_orders}"
        ]
        return (res_lines, set()) if return_linked else res_lines

    plan_by_order_idx: Dict[int, str] = {}
    linked_plan_by_order_idx: Dict[int, Dict[str, Any]] = {}
    linked_plan_indices: set[int] = set()
    if active_plans:
        for o_idx, o in enumerate(dict_orders):
            o_side = str(o.get("side") or "").strip().upper()
            o_px = safe_pos_float(o.get("price"), default=None)
            best_p_idx, matched_pl = match_open_order_resting_plan(
                o, active_plans, used_plan_indices=linked_plan_indices
            )
            if best_p_idx is not None and isinstance(matched_pl, dict):
                linked_plan_indices.add(best_p_idx)
                pl = matched_pl
                linked_plan_by_order_idx[o_idx] = pl
                t_px_v = safe_pos_float(pl.get("target_px"), default=None)
                i_px_v = safe_pos_float(pl.get("invalidation_px"), default=None)
                t_dist = (
                    round(((t_px_v - o_px) / o_px) * 100.0, 2)
                    if (o_px is not None and t_px_v is not None)
                    else pl.get("target_dist_pct")
                )
                i_dist = (
                    round(((i_px_v - o_px) / o_px) * 100.0, 2)
                    if (o_px is not None and i_px_v is not None)
                    else pl.get("invalidation_dist_pct")
                )
                t_s = (
                    f"{_fmt_plain_float(pl['target_px'])}({fmt_signed(t_dist)}%)"
                    if pl.get("target_px") is not None and t_dist is not None
                    else (_fmt_plain_float(pl.get("target_px")) if pl.get("target_px") is not None else "-")
                )
                i_s = (
                    f"{_fmt_plain_float(pl['invalidation_px'])}({fmt_signed(i_dist)}%)"
                    if pl.get("invalidation_px") is not None and i_dist is not None
                    else (_fmt_plain_float(pl.get("invalidation_px")) if pl.get("invalidation_px") is not None else "-")
                )
                rr_cur = compute_order_plan_rr(o_px, o_side, pl.get("target_px"), pl.get("invalidation_px"))
                rr_sfx = f" RR@limit={rr_cur:.2f}x" if rr_cur is not None else ""
                st_lbl = str(pl.get("status") or "ACTIVE")
                plan_by_order_idx[o_idx] = f"{t_s}/{i_s}{rr_sfx} [{st_lbl}]"

    has_linked = bool(plan_by_order_idx)
    hdr_lbl = (
        "  * Active Open Orders & Linked Trade Plans (Locking Liquidity) \u2014 open_limit_orders:"
        if has_linked
        else "  * Active Open Orders on Exchange (Locking Liquidity) \u2014 open_limit_orders:"
    )
    cols_hdr = (
        "    | Venue | Symbol | Side | Price | Qty | Filled | ID | Actor | Age | Dist | Fill1h | Plan (Tgt/Inv) | Note |"
        if has_linked
        else "    | Venue | Symbol | Side | Price | Qty | Filled | ID | Actor | Age | Dist | Fill1h | Note |"
    )
    cols_sep = (
        "    |---|---|---|---|---|---|---|---|---|---|---|---|---|"
        if has_linked
        else "    |---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    out = [hdr_lbl, cols_hdr, cols_sep]
    for o_idx, o in enumerate(dict_orders):
        ven = str(o.get("venue") or "-")
        sym = str(o.get("symbol") or "-")
        side = str(o.get("side") or "-")
        px = _fmt_plain_float(o.get("price")) if o.get("price") is not None else "-"
        raw_qty = (
            o.get("qty")
            if o.get("qty") is not None
            else (o.get("origQty") if o.get("origQty") is not None else o.get("origSz"))
        )
        qty_v = safe_pos_float(raw_qty, default=None)
        qty = _fmt_plain_float(raw_qty) if raw_qty is not None else "-"
        filled_v = _extract_open_order_filled_qty(o)
        if filled_v > 0.0:
            if qty_v is not None and qty_v > filled_v:
                rem_v = round(max(0.0, qty_v - filled_v), 8)
                rem_s = _fmt_plain_float(rem_v)
                if rem_s.endswith(".0"):
                    rem_s = rem_s[:-2]
                filled = f"{_fmt_plain_float(filled_v)}(rem={rem_s})"
            else:
                filled = _fmt_plain_float(filled_v)
        else:
            filled = "0"
        oid = str(o.get("order_id") or o.get("orderId") or "-")
        actor = str(o.get("actor") or "-")
        age = str(o.get("age") or "-")
        dist_v = o.get("dist_from_spot_pct")
        dist = f"{fmt_signed(dist_v)}%" if dist_v is not None else "-"
        fp_v = o.get("fill_prob_1h_pct")
        fp = f"{fp_v}%" if fp_v is not None else "-"
        notes: List[str] = []
        if st := str(o.get("status") or "").strip().upper():
            if st not in ("NEW", "OPEN"):
                notes.append(st)
        if o.get("ob_flag"):
            ob_age = safe_float(o.get("ob_age_s"), None)
            notes.append(f"{o['ob_flag']} {format_age_sec_compact(ob_age)}" if ob_age is not None else str(o["ob_flag"]))
        if o.get("recommended_front_run_px") is not None:
            fr_px = o["recommended_front_run_px"]
            linked_pl = linked_plan_by_order_idx.get(o_idx)
            rr_fr = (
                compute_order_plan_rr(fr_px, side, linked_pl.get("target_px"), linked_pl.get("invalidation_px"))
                if linked_pl is not None
                else None
            )
            notes.append(f"front_run={fr_px}(RR={rr_fr:.2f}x)" if rr_fr is not None else f"front_run={fr_px}")
        elif (wq := str(o.get("wall_queue_pos") or "").strip().upper()) and not wq.startswith("FRONT_OF_"):
            w_pfx = "ask" if str(side).strip().upper() == "SELL" else "bid"
            w_px = safe_pos_float(o.get(f"{w_pfx}_wall_ahead_px"), default=None)
            w_usd = safe_pos_float(o.get(f"{w_pfx}_wall_ahead_usd"), default=None)
            if w_px is not None and w_usd is not None:
                notes.append(f"{wq}(wall={_fmt_plain_float(w_px)}/{format_compact_usd(w_usd)})")
            elif w_px is not None:
                notes.append(f"{wq}(wall={_fmt_plain_float(w_px)})")
            else:
                notes.append(wq)
        note_s = "; ".join(notes) if notes else "-"
        plan_col = f" {plan_by_order_idx.get(o_idx, '-')} |" if has_linked else ""
        out.append(
            f"    | {ven} | {sym} | {side} | {px} | {qty} | {filled} | {oid} | {actor} | {age} | {dist} | {fp} |{plan_col} {note_s} |"
        )
    return (out, linked_plan_indices) if return_linked else out


def _format_exec_intent_item(item: Any) -> str:
    """Format a single execution status or recent AI intent dict cleanly without raw Python dict quotes."""
    if not isinstance(item, dict):
        return str(item)
    ven = item.get("venue") or "-"
    sym = item.get("symbol") or "-"
    side = item.get("side") or item.get("decision") or "-"
    otype = item.get("order_type") or ""
    raw_st = str(item.get("status") or "").strip()
    raw_st_u = raw_st.upper()
    ven_st = str(item.get("venue_status") or "").strip()
    st = (
        "RESTING_IN_BOOK"
        if raw_st_u == "RESTING_IN_BOOK"
        else ((ven_st or raw_st or "-") if raw_st_u in ("", "EXECUTED") else raw_st)
    )
    hdr = f"[{ven}] {sym} {side}" + (f" {otype}" if otype else "") + f" {st}"
    details: List[str] = []
    if item.get("fill_px") is not None:
        details.append(f"fill_px={item['fill_px']}")
    if item.get("limit_px") is not None:
        details.append(f"limit_px={item['limit_px']}")
    elif item.get("limit_price") is not None and item.get("limit_price") != item.get("fill_px"):
        details.append(f"limit_px={item['limit_price']}")
    if item.get("ref_px") is not None:
        details.append(f"ref_px={item['ref_px']}")
    if item.get("filled_qty") is not None and safe_float(item.get("filled_qty"), 0.0) > 0.0:
        details.append(f"qty={_fmt_plain_float(item['filled_qty'])}")
    if item.get("notional_usd") is not None:
        details.append(f"notional=${safe_float(item['notional_usd'], 0.0):,.2f}")
    if item.get("target_px") is not None:
        details.append(f"target={item['target_px']}")
    if item.get("invalidation_px") is not None:
        details.append(f"inval={item['invalidation_px']}")
    if item.get("slippage_bps") is not None:
        details.append(f"slippage_bps={item['slippage_bps']}")
    if item.get("arrival_shortfall_bps") is not None:
        details.append(f"arr_bps={item['arrival_shortfall_bps']}")
    if item.get("order_id") or item.get("orderId"):
        details.append(f"id={item.get('order_id') or item.get('orderId')}")
    if item.get("age_min") is not None:
        details.append(f"age={item['age_min']}m")
    elif item.get("age_h") is not None:
        details.append(f"age={item['age_h']}h")
    if item.get("reason"):
        details.append(f"reason={item['reason']}")
    return f"{hdr} ({', '.join(details)})" if details else hdr


def _format_retry_queue_item(item: Any) -> str:
    """Format a single active retry-queue record cleanly without raw Python dict quotes."""
    if not isinstance(item, dict):
        return str(item)
    ven = item.get("venue") or "-"
    sym = item.get("symbol") or "-"
    side = item.get("side") or "-"
    hdr = f"[{ven}] {sym} {side}"
    details: List[str] = []
    if item.get("qty") is not None:
        details.append(f"qty={_fmt_plain_float(item['qty'])}")
    if item.get("requested_price") is not None:
        details.append(f"px={_fmt_plain_float(item['requested_price'])}")
    if item.get("kind"):
        details.append(f"kind={item['kind']}")
    if item.get("attempts") is not None:
        details.append(f"attempts={item['attempts']}")
    if item.get("last_failure_reason"):
        details.append(f"reason={item['last_failure_reason']}")
    if item.get("age_min") is not None:
        details.append(f"age={item['age_min']}m")
    return f"{hdr} ({', '.join(details)})" if details else hdr


def _format_compact_kv_value(val: Any) -> str:
    """Format nested runtime state dicts/lists into clean key=val notation without raw Python dict quotes."""
    if isinstance(val, dict):
        parts: List[str] = []
        for k, v in val.items():
            if v is None:
                continue
            if isinstance(v, dict):
                inner = _format_compact_kv_value(v)
                parts.append(f"{k}({inner})" if inner else f"{k}()")
            elif isinstance(v, list):
                parts.append(f"{k}=[{', '.join(_format_compact_kv_value(x) for x in v if x is not None)}]")
            else:
                parts.append(f"{k}={_format_compact_kv_value(v)}")
        return ", ".join(parts)
    if isinstance(val, list):
        return f"[{', '.join(_format_compact_kv_value(x) for x in val if x is not None)}]"
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float)):
        return _fmt_plain_float(val)
    return str(val)


def _format_kraken_xstocks_line(xs_view: Dict[str, Any], *, scope_s: str, tag: str = "") -> str:
    """Format Kraken tokenized xStocks cross-venue context concisely without raw dict repr."""
    xs = xs_view.get("payload")
    xs_pair_lbl = xs_view.get("pair_lbl") or "xStocks"
    xs_ipo_lbl = xs_view.get("ipo_lbl") or ""
    xs_sib = xs_view.get("t212_sibling") or ""
    if not isinstance(xs, dict):
        if scope_s in ("equities", "equity"):
            return (
                f"  * Cross-Venue Kraken Tokenized Sibling Reference (READ-ONLY in Equities Scope; "
                f"24/7 {xs_pair_lbl} quote/inventory context for T212 {xs_sib}){tag}: {xs}"
            )
        return (
            f"  * Kraken Tokenized Equities / xStocks (API-TRADABLE 24/7 on 'kraken' at 0bps maker/taker fee [SELL-ONLY EU API]; "
            f"{xs_pair_lbl}{xs_ipo_lbl} & T212 {xs_sib} context){tag}: {xs}"
        )
    parts: List[str] = []
    if scope_s in ("equities", "equity"):
        if xs.get("live_price") is not None:
            alloc_disp = round_opt_px(xs.get("alloc_price"), 6)
            pnl_s = f" ({fmt_signed(xs.get('pnl_vs_alloc_pct'))}% vs alloc {alloc_disp})" if xs.get("pnl_vs_alloc_pct") is not None else ""
            parts.append(f"{xs_pair_lbl}=${xs['live_price']}{pnl_s}")
        if isinstance(xs.get("holdings"), dict) and xs["holdings"]:
            h_s = ", ".join(f"{k}={_fmt_plain_float(v)}" for k, v in xs["holdings"].items())
            parts.append(f"holdings=({h_s})")
    if xs.get("underlying_live_price") is not None:
        u_lbl = xs.get("underlying") or "underlying"
        sess_s = f", {xs['underlying_session']}" if xs.get("underlying_session") else ""
        vs_s = f", xstock_vs_underlying={fmt_signed(xs.get('xstock_vs_underlying_pct'))}%" if xs.get("xstock_vs_underlying_pct") is not None else ""
        parts.append(f"{u_lbl}=${xs['underlying_live_price']}{sess_s}{vs_s}")
    if isinstance(sib_c := xs.get("t212_sibling_cycle"), dict) and sib_c:
        sc_parts: List[str] = []
        if sib_c.get("qty") is not None:
            sc_parts.append(f"qty={_fmt_plain_float(sib_c['qty'])}")
        if sib_c.get("avg_cost_px") is not None:
            sc_parts.append(f"avg_cost={sib_c['avg_cost_px']}")
        if sib_c.get("dd_pct") is not None:
            sc_parts.append(f"pnl_vs_cost={fmt_signed(sib_c['dd_pct'])}%")
        if sib_c.get("pos_peak") is not None:
            dd_pk = f"[{sib_c['dd_from_peak_pct']}%]" if sib_c.get("dd_from_peak_pct") is not None else ""
            sc_parts.append(f"peak_px={sib_c['pos_peak']}{dd_pk}")
        if sib_c.get("last_sell_price") is not None:
            ek = f"({sib_c['last_exit_kind']})" if sib_c.get("last_exit_kind") else ""
            sc_parts.append(f"last_sell={sib_c['last_sell_price']}{ek}")
        if sib_c.get("realized_pnl_usd") is not None:
            rp_s = format_signed_usd(sib_c["realized_pnl_usd"], decimals=2)
            cum_s = (
                f" cum={format_signed_usd(sib_c['cum_realized_pnl_usd'], decimals=2)}"
                if sib_c.get("cum_realized_pnl_usd") is not None
                else ""
            )
            sc_parts.append(f"realized={rp_s}{cum_s}")
        parts.append(f"t212_sibling({xs_sib}: {', '.join(sc_parts)})")
    if isinstance(other_sibs := xs.get("other_t212_sibling_cycles"), dict) and other_sibs:
        parts.append(f"other_t212_siblings({_format_compact_kv_value(other_sibs)})")
    if xs.get("watched_ipo_alerts"):
        parts.append(f"ipo_alerts={_format_compact_kv_value(xs['watched_ipo_alerts'])}")
    body = " | ".join(parts) if parts else _format_compact_kv_value(xs)
    if scope_s in ("equities", "equity"):
        return (
            f"  * Cross-Venue Kraken Tokenized Sibling Reference (READ-ONLY in Equities Scope; "
            f"24/7 {xs_pair_lbl} quote/inventory context for T212 {xs_sib}){tag}: {body}"
        )
    return (
        f"  * Kraken Tokenized Equities / xStocks (API-TRADABLE 24/7 on 'kraken' at 0bps maker/taker fee [SELL-ONLY EU API]; "
        f"{xs_pair_lbl}{xs_ipo_lbl} & T212 {xs_sib} context){tag}: {body}"
    )


def _format_cross_market_equities_table(
    eq_bw: Any, *, scope_s: str, tag: str = ""
) -> List[str]:
    """Format Cross-Market US Equities Telemetry as a compact Markdown table."""
    eq_hdr_cyc_sfx = "" if scope_s == "crypto" else " & Active Cycles"
    hdr = f"  * Cross-Market US Equities Telemetry (Live/Extended & T212 Quotes, Multi-Horizon Deltas{eq_hdr_cyc_sfx}){tag}:"
    if not isinstance(eq_bw, dict) or not eq_bw:
        return [f"{hdr} {eq_bw}"]
    has_cycle = any(isinstance(v, dict) and v.get("t212_cycle") for v in eq_bw.values())
    cols = "    | Symbol | Px | 24h | 7d | Pos24h | ExtVsRTH | Close | Source (Age) |" + (" Cycle |" if has_cycle else "")
    sep = "    |---|---|---|---|---|---|---|---|" + ("---|" if has_cycle else "")
    out = [hdr, cols, sep]
    for sym, ev in sorted(eq_bw.items()):
        if not isinstance(ev, dict):
            out.append(f"    | {sym} | {ev} | - | - | - | - | - | - |" + (" - |" if has_cycle else ""))
            continue
        px_v = safe_pos_float(ev.get("price"), default=None)
        px_s = _fmt_px(px_v, ev.get("price")) if px_v is not None else "-"
        c24 = f"{fmt_signed(ev.get('chg_24h_pct'))}%" if ev.get("chg_24h_pct") is not None else "-"
        c7d = f"{fmt_signed(ev.get('chg_7d_pct'))}%" if ev.get("chg_7d_pct") is not None else "-"
        p24 = f"{ev['pos_24h_pct']}%" if ev.get("pos_24h_pct") is not None else "-"
        ext = f"{fmt_signed(ev.get('ext_vs_rth_pct'))}%" if ev.get("ext_vs_rth_pct") is not None else "-"
        cl_v = safe_pos_float(ev.get("t212_close_px"), default=None)
        cl_s = _fmt_px(cl_v, ev.get("t212_close_px")) if cl_v is not None else "-"
        src = str(ev.get("quote_source") or "-")
        age_m = safe_float(ev.get("quote_age_min"), None)
        src_s = f"{src} ({age_m:.0f}m)" if age_m is not None else src
        cyc_raw = ev.get("t212_cycle")
        cyc_val = _format_compact_kv_value(cyc_raw) if isinstance(cyc_raw, (dict, list)) else (cyc_raw if cyc_raw is not None else "-")
        cyc_s = f" {cyc_val} |" if has_cycle else ""
        out.append(f"    | {sym} | {px_s} | {c24} | {c7d} | {p24} | {ext} | {cl_s} | {src_s} |{cyc_s}")
    return out


def _format_cross_market_crypto_table(cr_bw: Any) -> List[str]:
    """Format Cross-Market Live Crypto Telemetry (equities scope) as a compact Markdown table."""
    hdr = "  * Cross-Market Live Crypto Telemetry (Raw 24/7 Prices, Multi-Horizon Deltas & Microstructure):"
    if not isinstance(cr_bw, dict) or not cr_bw:
        return [f"{hdr} {cr_bw}"]
    out = [
        hdr,
        "    | Coin | Px | 15m | 1h | 24h | 7d | Pos24h | FundAnn | OI1h | CVD15m |",
        "    |---|---|---|---|---|---|---|---|---|---|",
    ]
    for coin, cv in sorted(cr_bw.items()):
        if not isinstance(cv, dict):
            out.append(f"    | {coin} | {cv} | - | - | - | - | - | - | - | - |")
            continue
        px_v = safe_pos_float(cv.get("price"), default=None)
        px_s = _fmt_px(px_v, cv.get("price")) if px_v is not None else "-"
        c15 = f"{fmt_signed(cv.get('chg_15m_pct'))}%" if cv.get("chg_15m_pct") is not None else "-"
        c1h = f"{fmt_signed(cv.get('chg_1h_pct'))}%" if cv.get("chg_1h_pct") is not None else "-"
        c24 = f"{fmt_signed(cv.get('chg_24h_pct'))}%" if cv.get("chg_24h_pct") is not None else "-"
        c7d = f"{fmt_signed(cv.get('chg_7d_pct'))}%" if cv.get("chg_7d_pct") is not None else "-"
        p24 = f"{cv['pos_24h_pct']}%" if cv.get("pos_24h_pct") is not None else "-"
        fa = f"{fmt_signed(cv.get('funding_annualized_pct'), 1)}%" if cv.get("funding_annualized_pct") is not None else "-"
        oi1 = f"{fmt_signed(cv.get('oi_1h_chg_pct'))}%" if cv.get("oi_1h_chg_pct") is not None else "-"
        cvd = _fmt_signed_usd(cv.get("cvd_15m_usd")) if cv.get("cvd_15m_usd") is not None else "-"
        out.append(f"    | {coin} | {px_s} | {c15} | {c1h} | {c24} | {c7d} | {p24} | {fa} | {oi1} | {cvd} |")
    return out


def _format_hyperliquid_summary_line(hl_ro: Any, *, tag: str = "") -> str:
    """Format Hyperliquid spot/perp equity summary concisely without raw Python dict repr."""
    if not isinstance(hl_ro, dict):
        return f"  * Hyperliquid Spot/Perp Equity Summary{tag}: {hl_ro}"
    parts: List[str] = []
    if hl_ro.get("engine"):
        parts.append(f"engine={hl_ro['engine']}")
    if hl_ro.get("paper") is not None:
        parts.append(f"paper={hl_ro['paper']}")
    if hl_ro.get("total_equity_usd") is not None:
        parts.append(f"equity=${safe_float(hl_ro.get('total_equity_usd'), 0.0):,.2f}")
    fc_hl = hl_ro.get("free_cash_usd") if hl_ro.get("free_cash_usd") is not None else hl_ro.get("cash_usd")
    if fc_hl is not None:
        parts.append(f"free_cash=${safe_float(fc_hl, 0.0):,.2f}")
    if (lq_hl := safe_float(hl_ro.get("locked_quote_usd"), 0.0) or 0.0) > 0.0:
        parts.append(f"locked_quote=${lq_hl:,.2f}")
    if (q_hl := safe_float(hl_ro.get("qty"), 0.0) or 0.0) > 0.0:
        val_s = f" (${safe_float(hl_ro.get('value_usd'), 0.0):,.2f})" if hl_ro.get("value_usd") is not None else ""
        parts.append(f"qty={_fmt_plain_float(q_hl)}{val_s}")
    body = ", ".join(parts) if parts else _format_compact_kv_value(hl_ro)
    return f"  * Hyperliquid Spot/Perp Equity Summary{tag}: {body}"


def _format_recent_order_line(sym: str, oinfo: Dict[str, Any], now_ms: int) -> str:
    """Format a single recent order summary line for Section 3C."""
    ro_ident = resolve_symbol_identity(sym)
    if ro_ident.is_tokenized_equity and ro_ident.display_symbol:
        sym = ro_ident.display_symbol
    open_lots_ro = oinfo.get("open_lots") if isinstance(oinfo.get("open_lots"), list) else []

    def _entry_ts_ms(entry_obj: Dict[str, Any]) -> int:
        raw_ms = normalize_timestamp_ms(entry_obj.get("ts_ms"), default=None) or None
        if raw_ms is None:
            ts_s = first_epoch_sec(entry_obj.get("ts_ms"), entry_obj.get("ts"), default=None, allow_iso=True)
            raw_ms = (ts_s * 1000.0) if ts_s is not None else None
        return round(raw_ms) if (raw_ms is not None and 0 < raw_ms <= now_ms + MAX_FUTURE_TIMESTAMP_SKEW_MS) else 0

    def _side_summary(key: str) -> str:
        entry = oinfo.get(key) if isinstance(oinfo.get(key), dict) else {}
        px = round_opt_px(entry.get("price"), 6)
        if px is None:
            return "none"
        ts = _entry_ts_ms(entry)
        age_h = max(0.0, round((now_ms - ts) / 3_600_000.0, 1)) if ts > 0 else None
        tag = entry.get("actor_tag") or format_actor_tag(entry.get("actor"), entry.get("module"), entry.get("rule"))
        qty_v = entry.get("qty")
        qty_pfx = f"{_fmt_plain_float(qty_v)} @ " if qty_v is not None else ""
        meta_parts = ([f"age={age_h}h"] if age_h is not None else []) + ([f"by={tag}"] if tag else [])
        other_entry = oinfo.get("last_sell") if isinstance(oinfo.get("last_sell"), dict) else {}
        other_ts = _entry_ts_ms(other_entry)
        if key == "last_buy" and is_last_buy_lot_harvested(
            px,
            open_lots_ro,
            last_buy_ts_ms=ts * 1.0,
            last_sell_ts_ms=other_ts * 1.0,
        ):
            meta_parts.append("lot_harvested")
        meta_sfx = f" ({', '.join(meta_parts)})" if meta_parts else ""
        return f"{qty_pfx}{_fmt_plain_float(px)}{meta_sfx}"

    last_b_px = round_opt_px((oinfo.get("last_buy") or {}).get("price") if isinstance(oinfo.get("last_buy"), dict) else None, 6)
    subcycle_sfx = "".join(
        f" | {part}"
        for part in format_recent_fill_subcycle_parts(
            oinfo,
            last_buy_px=last_b_px,
            open_lots=open_lots_ro,
            round_anchors=True,
            include_open_lots=False,
            include_markout=True,
        )
    )
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
        f"{subcycle_sfx}"
    )


def _format_portfolio_section(
    ps: Dict[str, Any],
    asset_scope: Any = "all",
    *,
    min_usd: float = 0.0,
    max_usd: float = 2000.0,
    include_macro_benchmarks: bool = True,
    include_cross_market: bool = True,
    now_ts: Optional[float] = None,
    delta_plan: Optional[Any] = None,
) -> List[str]:
    """Format Section 3C (Spot Wallet Balances, Tradable Positions, Cycle Cost Anchors & Last Fills)."""
    scope_spec = resolve_scope_presentation(asset_scope)
    scope_s = scope_spec.scope_key
    lines = ["\n### 1. PORTFOLIO, OPEN ORDERS & TRADE PLANS:"]
    if isinstance(pre := ps.get("portfolio_risk_engine"), dict) and pre.get("unified_nav_usd"):
        pre_hdr = scope_spec.risk_engine_hdr
        nav_key_lbl = scope_spec.risk_engine_nav_lbl
        fc_pct = pre.get("global_free_cash_pct")
        pre_parts = [
            f"{nav_key_lbl}=${safe_float(pre.get('unified_nav_usd'), 0.0):,.2f}",
            f"invested=${safe_float(pre.get('invested_exposure_usd'), 0.0):,.2f}",
            f"free_cash=${safe_float(pre.get('global_free_cash_usd'), 0.0):,.2f}"
            + (f" ({fc_pct}%)" if fc_pct is not None else ""),
        ]
        if (lq_pre := safe_float(pre.get("locked_quote_usd"), 0.0)) > 0.0:
            lq_pct = pre.get("locked_quote_pct")
            pre_parts.append(
                f"locked_quote=${lq_pre:,.2f}" + (f" ({lq_pct}%)" if lq_pct is not None else "")
            )
        if isinstance(vn_bd := pre.get("venue_nav_breakdown_usd"), dict) and vn_bd:
            vn_s = ", ".join(
                f"{vk}=${safe_float(vv, 0.0):,.2f}"
                for vk, vv in sorted(vn_bd.items())
                if safe_float(vv, 0.0) > 0.0
            )
            if vn_s:
                pre_parts.append(f"venues({vn_s})")
        if scope_spec.include_crypto_beta:
            if pre.get("portfolio_btc_beta") is not None:
                pre_parts.append(f"portfolio_btc_beta={pre.get('portfolio_btc_beta')}")
            if pre.get("avg_corr_vs_btc_24h") is not None:
                pre_parts.append(f"avg_corr_24h={pre.get('avg_corr_vs_btc_24h')}")
        if pre.get("intraday_var_95_usd") is not None:
            var_pct = pre.get("intraday_var_95_pct")
            pre_parts.append(
                f"intraday_var_95=${safe_float(pre.get('intraday_var_95_usd'), 0.0):,.2f}"
                + (f" ({var_pct}%)" if var_pct is not None else "")
            )
        if pre.get("intraday_cvar_975_usd") is not None:
            cvar_pct = pre.get("intraday_cvar_975_pct")
            pre_parts.append(
                f"cvar_975=${safe_float(pre.get('intraday_cvar_975_usd'), 0.0):,.2f}"
                + (f" ({cvar_pct}%)" if cvar_pct is not None else "")
            )
        if pre.get("daily_nav_drawdown_pct") is not None:
            pre_parts.append(f"nav_dd_day={fmt_signed(pre.get('daily_nav_drawdown_pct'))}%")
        if pre.get("vol_target_tranche_usd") is not None:
            pre_parts.append(f"vol_target_tranche=${safe_float(pre.get('vol_target_tranche_usd'), 0.0):,.0f}")
        if pre.get("subcycle_lifo_realized_pnl_usd") is not None:
            pre_parts.append(
                f"subcycle_lifo_pnl={format_signed_usd(pre.get('subcycle_lifo_realized_pnl_usd'), decimals=2)}"
            )
        if pre.get("est_recent_fee_drag_usd") is not None:
            pre_parts.append(f"est_fee_drag=${safe_float(pre.get('est_recent_fee_drag_usd'), 0.0):.2f}")
        pt_parts: List[str] = []
        if (em_v := safe_float(pre.get("avg_entry_markout_bps"), None)) is not None:
            pt_parts.append(f"entry={fmt_signed(em_v, 0)}bps")
        if (xm_v := safe_float(pre.get("avg_exit_markout_bps"), None)) is not None:
            pt_parts.append(f"exit={fmt_signed(xm_v, 0)}bps")
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
        lines.append(f"  * {pre_hdr}: {', '.join(pre_parts)}")
    if ps.get("total_value_usdc") is not None:
        tv_extras: List[str] = []
        if ps.get("total_value_usdc_age_min") is not None:
            tv_extras.append(f"snapshot_age={ps['total_value_usdc_age_min']}m")
        if (live_bn := safe_float(ps.get("live_binance_equity_usdc"), None)) is not None and live_bn > 0.0:
            bnb_res = safe_pos_float(ps.get("binance_bnb_fee_reserve_usdc"), default=None)
            bnb_sfx = f" [excl. ~${bnb_res:,.2f} BNB fee reserve]" if bnb_res is not None else ""
            tv_extras.append(f"live_reconciled=${live_bn:,.2f} USDC{bnb_sfx}")
        tv_sfx = f" ({', '.join(tv_extras)})" if tv_extras else ""
        lines.append(f"  * Total Binance Portfolio Value: ${safe_float(ps.get('total_value_usdc'), 0.0):,.2f} USDC{tv_sfx}")
    has_venue_cash_line = False
    if ps.get("venue_quote_cash"):
        cash_parts: List[str] = []
        for v in ps["venue_quote_cash"].values():
            if not isinstance(v, dict):
                continue
            v_bd = v.get("breakdown")
            v_bd_s = (
                f" ({', '.join(f'{bk}=${safe_float(bv, 0.0):,.2f}' for bk, bv in v_bd.items())})"
                if isinstance(v_bd, dict) and (len(v_bd) > 1 or any(k != v.get("asset") for k in v_bd))
                else ""
            )
            part = f"{v.get('venue')} {v.get('asset')}=${safe_float(v.get('free'), 0.0):,.2f}{v_bd_s}"
            if (lq_v := safe_float(v.get("locked_quote_usd"), 0.0)) > 0.0:
                part += f" [locked_quote_usd=${lq_v:,.2f}]"
            cash_parts.append(part)
        if cash_parts:
            has_venue_cash_line = True
            lines.append(f"  * Available Free Quote Cash per Venue: {', '.join(cash_parts)}")
    view = evaluate_portfolio_prompt_view(ps, asset_scope=asset_scope, max_usd=max_usd, min_usd=min_usd)
    if view["tradable_positions"]:
        sorted_positions = sorted(
            enumerate(view["tradable_positions"]),
            key=lambda it: (
                0
                if (
                    (safe_float(it[1].get("value_usd"), 0.0) or 0.0) >= 2.0
                    or (safe_float(it[1].get("locked_base_qty"), 0.0) or 0.0) > 0.0
                )
                else (1 if (safe_float(it[1].get("locked_quote_usd"), 0.0) or 0.0) > 0.0 else 2),
                -(
                    (safe_float(it[1].get("value_usd"), 0.0) or 0.0)
                    + (safe_float(it[1].get("locked_base_usd"), 0.0) or 0.0)
                ),
                it[0],
            ),
        )
        ordered_positions = [pv for _, pv in sorted_positions]
        lines.append(
            f"  * Live Tradable Spot Instruments (Eligible for Autonomous BUY / SELL) "
            f"[per-order max_buy <= ${max_usd:g}; simultaneous BUYs share venue free cash above]:"
        )
        for pos_view in ordered_positions:
            lines.append(
                _format_tradable_position_line(
                    pos_view,
                    max_usd=max_usd,
                    min_usd=min_usd,
                    include_static_specs=False,
                    include_free_quote=not has_venue_cash_line,
                )
            )
        if delta_plan is None or delta_plan.should_show("venue_specs"):
            vs_tag = delta_plan.tag_for("venue_specs") if delta_plan is not None else ""
            lines.extend(_format_venue_specs_table(ordered_positions, tag=vs_tag))
    active_plans = view.get("active_ai_trade_plans") or []
    linked_plan_indices: set[int] = set()
    if view["has_open_orders"]:
        oo_lines, linked_plan_indices = _format_open_limit_orders_table(
            view["clean_open_limit_orders"],
            active_plans=active_plans,
            return_linked=True,
        )
        lines.extend(oo_lines)
    else:
        lines.append(
            "  * Active Open Limit Orders on Exchange \u2014 open_limit_orders: [] (none resting)"
        )
    unlinked_plans = [pl for idx, pl in enumerate(active_plans) if idx not in linked_plan_indices]
    if unlinked_plans:
        lines.append("  * Active AI Trade Plans (Persistent Exit/Invalidation Memory):")
        for pl in unlinked_plans:
            p_parts: List[str] = []
            ord_st = str(pl.get("order_state") or "FILLED")
            is_lim_anc = pl.get("dist_anchor") == "limit" or ord_st.strip().upper() == "RESTING"
            anc_lbl = "vs_limit" if is_lim_anc else "vs_spot"
            if pl.get("target_px") is not None:
                td = pl.get("target_dist_pct")
                td_s = f" ({fmt_signed(td)}% {anc_lbl})" if td is not None else ""
                p_parts.append(f"target={pl['target_px']}{td_s}")
            if pl.get("invalidation_px") is not None:
                id_v = pl.get("invalidation_dist_pct")
                id_s = f" ({fmt_signed(id_v)}% {anc_lbl})" if id_v is not None else ""
                p_parts.append(f"invalidation={pl['invalidation_px']}{id_s}")
            pl_st = str(pl.get("status") or "ACTIVE").strip().upper()
            if pl_st not in ("INVALIDATED", "TARGET_HIT", "CLOSED_EARLY"):
                anc_px = pl.get("entry_px") if is_lim_anc else pl.get("spot_px")
                rr_lbl = "RR@limit" if is_lim_anc else "RR@spot"
                if (rr_u := compute_order_plan_rr(anc_px, pl.get("side"), pl.get("target_px"), pl.get("invalidation_px"))) is not None:
                    p_parts.append(f"{rr_lbl}={rr_u:.2f}x")
            entry_s = f"@{pl['entry_px']}" if pl.get("entry_px") is not None else ""
            meta_p = [ord_st]
            if not is_lim_anc and pl.get("spot_px") is not None:
                meta_p.append(f"spot={pl['spot_px']}")
            if pl.get("age_h") is not None:
                meta_p.append(f"age={pl['age_h']}h")
            lines.append(
                f"    - [{pl.get('venue')}] {pl.get('symbol')} {pl.get('side')}{entry_s} "
                f"({', '.join(meta_p)}): {', '.join(p_parts)} [{pl.get('status', 'ACTIVE')}]"
            )
    if (delta_plan is None or delta_plan.should_show("scorecard")) and (
        sc_line := format_ai_decision_scorecard_line(view.get("ai_decision_scorecard"))
    ):
        sc_tag = delta_plan.tag_for("scorecard") if delta_plan is not None else ""
        if sc_tag and ": " in sc_line:
            sc_line = sc_line.replace(": ", f"{sc_tag}: ", 1)
        lines.append(f"  * {sc_line}")
    if (delta_plan is None or delta_plan.should_show("calibration")) and (
        cal_line := format_ai_calibration_feedback_line(view.get("ai_calibration_feedback"))
    ):
        cal_tag = delta_plan.tag_for("calibration") if delta_plan is not None else ""
        if cal_tag and ": " in cal_line:
            cal_line = cal_line.replace(": ", f"{cal_tag}: ", 1)
        lines.append(f"  * {cal_line}")
    if (delta_plan is None or delta_plan.should_show("venue_cycles")) and view["venue_cycles"]:
        vc_tag = delta_plan.tag_for("venue_cycles") if delta_plan is not None else ""
        for venue_lbl, venue_id, cyc_items in view["venue_cycles"]:
            lines.append(f"  * {venue_lbl} Active Cycle Cost & Drawdown Anchors{vc_tag}:")
            for c_sym, c_data in cyc_items:
                lines.append(_format_venue_cycle_line(venue_id, c_sym, c_data))
    if (delta_plan is None or delta_plan.should_show("kraken_xstocks")) and (xs_view := view["kraken_xstocks"]):
        xs_tag = delta_plan.tag_for("kraken_xstocks") if delta_plan is not None else ""
        lines.append(_format_kraken_xstocks_line(xs_view, scope_s=scope_s, tag=xs_tag))
    if include_cross_market and view["crypto_bellwethers"]:
        lines.extend(_format_cross_market_crypto_table(view["crypto_bellwethers"]))
    if include_cross_market and (delta_plan is None or delta_plan.should_show("equity_bellwethers")) and view["equity_bellwethers"]:
        eq_tag = delta_plan.tag_for("equity_bellwethers") if delta_plan is not None else ""
        lines.extend(_format_cross_market_equities_table(view["equity_bellwethers"], scope_s=scope_s, tag=eq_tag))
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
    if (delta_plan is None or delta_plan.should_show("hyperliquid_readonly")) and ps.get("hyperliquid_readonly"):
        hl_tag = delta_plan.tag_for("hyperliquid_readonly") if delta_plan is not None else ""
        lines.append(_format_hyperliquid_summary_line(ps["hyperliquid_readonly"], tag=hl_tag))
    now_ms = math.trunc(resolve_now_ts(now_ts) * 1000)
    obs_only_syms = observe_only_registry_symbols()
    active_pos_syms = {
        k
        for p in (view.get("tradable_positions") or ())
        if isinstance(p, dict)
        for p_sym in (p.get("disp_symbol") or p.get("symbol"),)
        for k in (
            str(p_sym or "").strip().upper(),
            clean_symbol_key(p_sym) or "",
        )
        if k
    }
    for _, _, cyc_items in (view.get("venue_cycles") or ()):
        for c_sym, _ in cyc_items:
            c_u = str(c_sym or "").strip().upper()
            c_k = clean_symbol_key(c_sym) or ""
            if c_u:
                active_pos_syms.add(c_u)
            if c_k:
                active_pos_syms.add(c_k)
    for sym, oinfo in sorted((ps.get("recent_orders") or {}).items()):
        if isinstance(oinfo, dict):
            s_u = str(sym or "").strip().upper()
            s_k = clean_symbol_key(sym) or ""
            if (
                obs_only_syms
                and (s_u in obs_only_syms or (s_k and s_k in obs_only_syms))
                and s_u not in active_pos_syms
                and (not s_k or s_k not in active_pos_syms)
            ):
                continue
            if delta_plan is not None and not delta_plan.should_show_recent_order(str(sym)):
                continue
            ro_line = _format_recent_order_line(sym, oinfo, now_ms)
            if delta_plan is not None and (ro_tag := delta_plan.recent_order_tag(str(sym))) and ": " in ro_line:
                ro_line = ro_line.replace(": ", f"{ro_tag}: ", 1)
            lines.append(ro_line)
    if clean_lces := view["clean_last_cycle_execution_status"]:
        lces_s = (
            "; ".join(_format_exec_intent_item(x) for x in clean_lces)
            if isinstance(clean_lces, list)
            else _format_exec_intent_item(clean_lces)
        )
        lines.append(
            f"  * Last Cycle Execution Status \u2014 last_cycle_execution_status: {lces_s}"
        )
    for st_key, st_label, st_val in view["runtime_states"]:
        if st_key == "recent_ai_intents" and isinstance(st_val, list):
            lines.append(f"  * {st_label}:")
            for it in st_val:
                lines.append(f"    - {_format_exec_intent_item(it)}")
        elif st_key == "active_retry_queue" and isinstance(st_val, list):
            lines.append(f"  * {st_label}:")
            for it in st_val:
                lines.append(f"    - {_format_retry_queue_item(it)}")
        elif isinstance(st_val, dict):
            lines.append(f"  * {st_label}:")
            for sk, sv in st_val.items():
                lines.append(f"    - {sk}: {_format_compact_kv_value(sv)}")
        elif isinstance(st_val, list):
            lines.append(f"  * {st_label}:")
            for it in st_val:
                lines.append(f"    - {_format_compact_kv_value(it)}")
        else:
            lines.append(f"  * {st_label}: {st_val}")
    return lines


