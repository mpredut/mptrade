"""Pure formatting helpers for Autonomous AI Reconciler Gemini prompts.

Separates prompt presentation (Sections 1, 2, 3A, 3B, 3C) and symbol price
resolution from telemetry harvesting and order execution in
`autonomous_ai_reconciler.py`.

Intentional prompt compression & cycle-boundary invariants (BY DESIGN -- DO NOT REVERT):
- Section numbering (`### 1.` -> `### 3.`): Sections `2/2B/2C` (SRE/logs/VPN) are emitted only for `can_self_heal` topics and omitted on trading topics.
- `bottom_6h(...)` (Section 3): Conditional pattern detector emitted only when an active 6h bottom structure forms (`pos_6h <= 70%` with meaningful drawdown).
- `open_lifo_lots` (Section 3C): Capped to the latest 4 lots (`<=14d`) for top-of-stack LIFO harvesting and omitted when `len(open_lots) <= 1` (already covered by `avg_buy_px`/`last_buy_px`).
- `last_cycle_execution_status` (Section 3C): Emitted when recent (`<=60m`) executions/fills/rejections or newly resting (`<=30m`) orders exist; omitted when idle or when all `OPEN_RESTING`/`PARTIALLY_FILLED` orders are `>30m` old and already displayed in `open_limit_orders`.
- `equity_bellwethers` on `scope="crypto"` (Section 3C): Micro-cap `RGNT_US_EQ` is omitted when major US tech/AI bellwethers (`NVDA_US_EQ`, `SPCX_US_EQ`) are present, and `t212_cycle` is omitted on `scope="crypto"` because `SPCX_US_EQ`'s cycle is already shown in `kraken_xstocks.t212_sibling_cycle` and `auto_cryptotrade_executor` cannot trade T212 shares (full cycles remain on `scope="equities"` and `scope="all"`).
- `last_exit_kind` & `realized_pnl` across cycle boundaries (Section 3C): A prior cycle's exit label (e.g. `SL`) is never reclassified against a newer cycle's re-entry cost (`last_sell_ts < entry_ts`), and multi-cycle cumulative PnL (`cum_cycles`) is preserved alongside active `subcycle` PnL.
"""

import math
import re
import time
from typing import Any, Dict, List, Optional

from botcore import _parse_bool_flag, normalize_epoch_sec, normalize_timestamp_ms, safe_float, safe_int, safe_pos_float
from cachepaths import (
    canonical_base_asset,
    has_equity_suffix,
)
from intelligence.macro.calendar_ssot import resolve_macro_economic_calendar
from intelligence.macro_sentiment_ssot import (
    TELEMETRY_SOURCE_STALE_INTERVAL_MULT,
    format_equity_macro_regime_line,
    format_interval_sec_label,
    format_macro_benchmarks_summary_line,
    resolve_derived_telemetry_intervals,
)
from intelligence.sentiment.evaluators.macro_telemetry_evaluator import (
    format_bottom_structure_6h_parts,
)
from intelligence.sentiment.evaluators.microstructure_evaluator import (
    format_book_window_cap_tag,
    format_flow_horizon_parts,
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
    format_kinematic_parts,
    format_multi_window_slopes_core,
    format_slope_val as _fmt_slope_val,
    round_opt_px,
)
from intelligence.sentiment.evaluators.prompt_view_evaluator import (
    evaluate_market_prices_prompt_view,
    evaluate_microstructure_prompt_view,
    evaluate_portfolio_prompt_view,
    evaluate_tradable_position_prompt_view,
)
from instrument_registry import is_equity_symbol, resolve_symbol_identity
from providers.order_id_context import format_actor_tag


def _age_sfx(d: Dict[str, Any], parens: bool = False) -> str:
    if d.get("age_min") is None:
        return ""
    pub = str(d.get("published_utc") or "").strip()
    inner = f"{pub}, age={d['age_min']}m" if pub else f"age={d['age_min']}m"
    return f" ({inner})" if parens else f", {inner}"


def _format_lifo_rem_avg_tag(
    lots: Any,
    held_qty: float,
    ref_avg_px: Optional[float],
    spot_px: Optional[float],
    *,
    label: str = "lifo_rem_avg_px",
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
        f" (pnl_vs_lifo={round(((spot_px - vwap) / vwap) * 100.0, 2) + 0.0:+.2f}%)"
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
        extra_parts.append(f"pos_peak={cyc.get('pos_peak')}{pk_sfx}")
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
            label="lifo_rem_avg",
        ):
            extra_parts.append(lifo_c_tag)
    if cyc.get("realized_pnl_usd") is not None:
        rp_v = cyc.get("realized_pnl_usd")
        rp_num = safe_float(rp_v, None)
        rp_f = rp_num if rp_num is not None else 0.0
        rp_s = f"{rp_num:.2f}" if rp_num is not None else str(rp_v)
        sub_rp = safe_float(cyc.get("subcycle_realized_pnl_usd"), None)
        cum_rp = safe_float(cyc.get("cum_realized_pnl_usd"), None)
        if sub_rp is not None and abs(sub_rp) < 0.01 and abs(rp_f) >= 0.01:
            extra_parts.append(f"realized_pnl=${rp_s} (subcycle=$0.00)")
        elif cum_rp is not None and abs(rp_f) >= 0.01 and abs(cum_rp - rp_f) >= 0.01:
            extra_parts.append(f"realized_pnl=${rp_s} (cum_cycles=${cum_rp:.2f})")
        else:
            extra_parts.append(f"realized_pnl=${rp_s}")
    for lbl, k in (("tp_sold", "tp_sold_levels"), ("resting_orders", "resting_orders")):
        if cyc.get(k):
            extra_parts.append(f"{lbl}={cyc.get(k)}")
    extra_sfx = f", {', '.join(extra_parts)}" if extra_parts else ""
    exit_kind = cyc.get("last_exit_kind")
    exit_kind_sfx = f" ({exit_kind})" if exit_kind else ""
    dd_val = cyc.get("pnl_vs_avg_pct") if cyc.get("pnl_vs_avg_pct") is not None else cyc.get("dd_pct")
    dd_f = safe_float(dd_val, None)
    dd_field = (
        f"pnl_vs_avg={dd_f:+.2f}%"
        if dd_f is not None and dd_f > 0.0
        else f"dd={f'{dd_val}%' if dd_val is not None else 'N/A'}"
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
            lb_pnl_sfx = f" (chg_vs_last_buy={lb_pnl:+.2f}%, lot_harvested)"
        else:
            lb_pnl_sfx = f" (pnl_vs_last_buy={lb_pnl:+.2f}%)"
    entry_px_part = f"entry_px={cyc.get('entry_price')}, " if cyc.get("entry_price") is not None else ""
    return (
        f"    - [{venue}] {sym}: qty={cyc.get('qty')}, cost=${cyc.get('cost_usd')}, "
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
            f" pos_{horizon}={pos_v}% (dd_high={safe_float(dd_v, 0.0):+.2f}%, "
            f"bounce_low={safe_float(bnc_v, 0.0):+.2f}%, span={spn_v}%)"
        )
    return f"{horizon}_range=[{lo}, {hi}]{pos_sfx}"


def _extract_macro_benchmarks_line(mb_raw: Any, *, include_crypto: bool = True) -> str:
    """Extract or synthesize the 1-line SPY / QQQ / ES=F / NQ=F / DXY / US10Y benchmark summary."""
    if not isinstance(mb_raw, dict) or not mb_raw:
        return ""
    if not include_crypto:
        summary_s = format_macro_benchmarks_summary_line(mb_raw, include_crypto=False)
    else:
        summary_s = str(mb_raw.get("summary_line") or "").strip() or format_macro_benchmarks_summary_line(mb_raw)
    return f"{summary_s}{_age_sfx(mb_raw, parens=True)}" if summary_s else ""


def _fmt_liq_usd(val: Any) -> str:
    vf = safe_float(val, 0.0)
    for threshold, suffix, decimals in (
        (1_000_000_000.0, "B", 2),
        (1_000_000.0, "M", 2),
        (1_000.0, "K", 1),
    ):
        if vf >= threshold:
            return f"${vf / threshold:.{decimals}f}{suffix}"
    return f"${vf:.0f}"


def _fmt_px(px_f: float, raw_px: Any = None) -> str:
    if px_f >= 1.0:
        if px_f < 1000.0 and abs(px_f - round(px_f, 2)) >= 1e-5:
            return f"${px_f:,.4f}".rstrip("0").rstrip(".")
        return f"${px_f:,.2f}"
    if px_f > 0:
        return f"${px_f:.6g}"
    if raw_px is not None and not isinstance(raw_px, bool):
        return f"${raw_px}"
    return "N/A"


def _format_macro_and_news_section(sensors: Any, asset_scope: Any = "all") -> List[str]:
    """Format Section 1 (Macro, Sentiment, Geopolitical & Live Breaking News) for the AI prompt."""
    is_equities_scope = str(asset_scope or "all").strip().lower() == "equities"
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
            eb_c = f" ({safe_float(mb.get('eth_btc_chg_24h_pct'), 0.0):+.2f}% 24h)" if mb.get("eth_btc_chg_24h_pct") is not None else ""
            mb_rot_parts.append(f"ETH/BTC={mb.get('eth_btc_ratio')}{eb_c}")
        if mb.get("eth_price") is not None:
            eth_c = f" ({safe_float(mb.get('eth_chg_24h_pct'), 0.0):+.2f}% 24h)" if mb.get("eth_chg_24h_pct") is not None else ""
            mb_rot_parts.append(f"ETH=${mb.get('eth_price')}{eth_c}")
        if mb.get("btc_vol_share_pct") is not None:
            mb_rot_parts.append(f"BTCDOM={mb.get('btc_vol_share_pct')}%")
        if mb.get("alt_vol_share_pct") is not None:
            mb_rot_parts.append(f"TOTAL3_share={mb.get('alt_vol_share_pct')}%")
    mb_line = (
        f"advance_ratio={mb.get('advance_ratio', 'N/A')}, "
        f"median_chg={mb.get('median_change_pct', 'N/A')}%, "
        f"mean_chg={mb.get('mean_change_pct', 'N/A')}%, disp_std={mb.get('dispersion_std', 'N/A')}, "
        f"advancing={mb.get('advancing', 'N/A')}/{mb.get('total_symbols', 'N/A')}"
        + (f", {', '.join(mb_rot_parts)}" if mb_rot_parts else "")
        + (f", top_gainers={mb_gainers}" if mb_gainers else "")
        + (f", top_losers={mb_losers}" if mb_losers else "")
        + (f", age={mb.get('age_min')}m" if mb.get("age_min") is not None else "")
        if mb else "None"
    )
    sa_flow_raw = str(sa.get("live_flow_divergence") or "").strip() if sa else ""
    sa_flow_delta = (
        sa_flow_raw[sa_flow_raw.find("("):]
        if "(" in sa_flow_raw
        else (f"({sa_flow_raw})" if sa_flow_raw else "")
    )
    if sa_flow_delta:
        sa_flow_delta = re.sub(
            r"(-old\s+)(?:(?:ACCUMULATE|HOLD|DEFENSIVE|TRIM_PROFITS|BULLISH|BEARISH|NEUTRAL|CAUTION)(?:/(?:ACCUMULATE|HOLD|DEFENSIVE|TRIM_PROFITS|BULLISH|BEARISH|NEUTRAL|CAUTION))?)\b",
            r"\1summary",
            sa_flow_delta,
            flags=re.IGNORECASE,
        )
        if (
            "CB_Prem=" not in sa_flow_delta
            and " vs " in sa_flow_delta
            and isinstance(mb_bench, dict)
            and (cb_v := safe_float(mb_bench.get("coinbase_premium_bps"), None)) is not None
        ):
            sa_flow_delta = sa_flow_delta.replace(" vs ", f", CB_Prem={cb_v:+.1f}bps vs ", 1)
    if not sa_flow_delta and sa and (safe_float(sa.get("age_min"), 0.0) or 0.0) >= 5.0:
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
                d_parts.append(f"cvd_15m={cvd_v / 1e6:+.2f}M")
            elif cvd1h_v is not None:
                d_parts.append(f"cvd_1h={cvd1h_v / 1e6:+.2f}M")
            if tr5_v is not None:
                d_parts.append(f"5m_tr={tr5_v:.2f}")
            if cvd5_v is not None:
                d_parts.append(f"cvd_5m={cvd5_v / 1e6:+.2f}M")
            if tr_v is not None and tr1h_v is not None:
                d_parts.append(f"1h_tr={tr1h_v:.2f}")
            if cvd_v is not None and cvd1h_v is not None:
                d_parts.append(f"cvd_1h={cvd1h_v / 1e6:+.2f}M")
            if isinstance(mb_bench, dict) and (cb_v := safe_float(mb_bench.get("coinbase_premium_bps"), None)) is not None:
                d_parts.append(f"CB_Prem={cb_v:+.1f}bps")
            if d_parts:
                sa_flow_delta = f"({', '.join(d_parts)} vs {sa.get('age_min')}m-old summary)"
    sa_summary_raw = str(sa.get("summary") or "").strip() if sa else ""
    sa_summary = (
        re.sub(
            r"(?:,\s*(?:so\s+|therefore\s+|thus\s+)?|\s*)\b(?:pausing|shifting|maintaining|staying|switching|recommending)\b[^.]*\b(?:HOLD|ACCUMULATE|DEFENSIVE|TRIM_PROFITS)\b[^.]*\.?",
            lambda m: "." if m.group(0).lstrip().startswith(",") else "",
            sa_summary_raw,
            flags=re.IGNORECASE,
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
    sa_line = (
        f"conf={sa.get('confidence', 'N/A')}"
        f"{_age_sfx(sa)}"
        + (f" | live_btc_flow_delta={sa_flow_delta}" if sa_flow_delta else "")
        + f" | summary={sa_summary}"
        if sa else "None"
    )
    gt_summary_raw = str(gt.get("summary") or "").strip() if gt else ""
    gt_summary = (
        re.sub(
            r",?\s*(?:so|therefore|thus)\s+(?:maintain|apply|use|keep|enforce)\s+[^.]*(?:downscale|brake|veto|halt|exposure)[^.]*\.?",
            ".",
            gt_summary_raw,
            flags=re.IGNORECASE,
        )
        .replace("..", ".")
        .strip()
    )
    if not gt_summary or gt_summary == ".":
        gt_summary = gt_summary_raw
    gt_line = (
        f"risk_score={gt.get('risk_score', 'N/A')}{_age_sfx(gt)}"
        + (f" | live_deescalation={gt.get('live_deescalation_note')}" if gt.get("live_deescalation_note") else "")
        + f" | summary={gt_summary}"
        if gt else "None"
    )
    if is_equities_scope:
        eq_reg_line = format_equity_macro_regime_line(mb_bench) if isinstance(mb_bench, dict) and mb_bench else "NEUTRAL"
        lines = [
            "### 1. ACTIVE MACRO, SENTIMENT, GEOPOLITICAL & LIVE BREAKING NEWS TELEMETRY:",
            f"- US Equity & Macro Regime: {eq_reg_line}{_age_sfx(mb_bench, parens=True) if isinstance(mb_bench, dict) else ''}",
            f"- Geopolitical Threat Shield: {gt_line}",
        ]
    else:
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
        if _parse_bool_flag(mc.get("active_release_window"), False) and mc.get("next_scheduled_event"):
            mc_parts.append(f'next_scheduled="{mc.get("next_scheduled_event")}"')
        if mc.get("fomc_today") is not None:
            mc_parts.append(f'fomc_today={bool(_parse_bool_flag(mc.get("fomc_today"), False))}')
        if mc.get("next_fomc_date"):
            mc_parts.append(f'next_fomc_date="{mc.get("next_fomc_date")}"')
        if mc.get("upcoming_windows"):
            up_lbl = "upcoming" if is_confirmed_macro else "routine_session_transitions"
            mc_parts.append(f'{up_lbl}={mc.get("upcoming_windows")}')
        if mc.get("detected_macro_headlines"):
            mc_parts.append(f'live_macro_headlines={mc.get("detected_macro_headlines")}')
        if mc.get("live_macro_prints"):
            mc_parts.append(f'live_macro_prints={mc.get("live_macro_prints")}')
        lines.append(f"- Macro Economic Calendar & Session Window: {' | '.join(mc_parts)}")
    mb_bench_line = _extract_macro_benchmarks_line(mb_bench, include_crypto=True)
    if mb_bench_line and not (
        is_equities_scope
        and f"{eq_reg_line}{_age_sfx(mb_bench, parens=True) if isinstance(mb_bench, dict) else ''}" == mb_bench_line
    ):
        lines.append(
            f"- US Equities, Futures & Macro Rates Benchmarks (SPY / QQQ / ES=F / NQ=F / DXY / US10Y 24h Delta): {mb_bench_line}"
        )
    if breaking_news:
        lines.append("- Live Breaking News Feed (cachedb/news_feed_collect.json, 1m refresh):")
        for hn in breaking_news:
            lines.append(f"  * {hn.get('title')}{_age_sfx(hn, parens=True)}")
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
    lines = [
        "### 1. FUNCTIONAL SYSTEM TELEMETRY, HEARTBEATS & WATCHDOG STATES:",
        f"- Active Heartbeats: {hb_keys}{stopped_hb_note}",
        f"- Watchdog & Runtime State Files: {sensors.state_files}",
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
            f"ISSUES DETECTED: {cfg_issues}"
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
                f"\n- Sustained High-Resource Processes (.resource_watchdog_state.json): {sv['resource_watchdog_offenders']}"
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
            upd_str = f" | AUTO_CONFIG_UPDATES: {rec_upd}" if rec_upd else ""
            lines.append(f"- LLM Model Catalog & Standalone Binaries: [{bin_str}] | configured_models={cat_status}{new_str}{upd_str}")
    return lines


def _format_market_prices_section(prices_and_trends: Dict[str, Any]) -> List[str]:
    """Format Section 3A (Live Market Prices, Multi-Horizon Trends & 24h/7d Evolution)."""
    view = evaluate_market_prices_prompt_view(prices_and_trends)
    btc_entry = view["btc_entry"]
    lines = ["\n### 3. LIVE MARKET PRICES, MULTI-HORIZON TRENDS & 24H EVOLUTION (ALL COINS):"]
    for sym, pinfo, sib_specs in view["rows"]:
        px = pinfo.get("price")
        px_f = safe_float(px, 0.0)
        px_str = _fmt_px(px_f, px)
        sib_lbl = ""
        if sib_specs:
            sib_parts: List[str] = []
            for sib_sym, s_px, diff_pct in sib_specs:
                if s_px > 0:
                    diff_s = (
                        ""
                        if diff_pct is None
                        else (f" [{diff_pct * 100.0:+.1f}bps]" if abs(diff_pct) < 0.005 else f" [{diff_pct:+.2f}%]")
                    )
                    sib_parts.append(f"{sib_sym}={_fmt_px(s_px)}{diff_s}")
                else:
                    sib_parts.append(sib_sym)
            if sib_parts:
                sib_lbl = f" (sibling venue: {', '.join(sib_parts)})"

        extras: List[str] = []
        _m24_h_keys = (
            "chg_1m_pct", "chg_3m_pct", "chg_5m_pct", "chg_15m_pct", "chg_30m_pct",
            "chg_1h_pct", "chg_3h_pct", "chg_6h_pct", "chg_12h_pct", "chg_24h_pct",
        )
        if "chg_15m_pct" in pinfo or any(pinfo.get(k) is not None for k in _m24_h_keys):
            all_horizons = "".join(
                f"{h_lbl}={safe_float(pinfo.get(h_k), 0.0):+.2f}% "
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
            vp_src_sfx = f", vp_source={pinfo['vp_source']}" if pinfo.get("vp_source") else ""
            extras.append(
                f"vol_profile(POC={pinfo.get('poc_px')}[{poc_d:+.2f}%]{sec_poc_sfx}{lvn_sfx}, "
                f"VAL={pinfo.get('val_px')}[{val_d:+.2f}%], "
                f"VAH={pinfo.get('vah_px')}[{vah_d:+.2f}%], "
                f"va_pos={va_pos}%{vp_src_sfx})"
            )

        is_sub24h_dup_7d = (
            "chg_24h_pct" not in pinfo
            and pinfo.get("low_24h") is not None
            and (pinfo.get("low_7d"), pinfo.get("high_7d")) == (pinfo.get("low_24h"), pinfo.get("high_24h"))
        )
        is_non_btc_crypto = not has_equity_suffix(sym) and canonical_base_asset(sym) != "BTC"
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
                            vs_btc_parts.append(f"{h_lbl}={a_v - b_v:+.2f}%")
            if vs_btc_parts:
                extras.append(f"vs_btc({', '.join(vs_btc_parts)})")
            cross_parts = (
                ([f"pos={pinfo['alt_btc_pos_24h_pct']}%"] if pinfo.get("alt_btc_pos_24h_pct") is not None else [])
                + [
                    f"{c_lbl}={safe_float(pinfo.get(c_k), 0.0):+.2f}%"
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
                f"7d={safe_float(pinfo.get('chg_7d_pct'), 0.0):+.2f}% "
                f"{_format_horizon_range(pinfo, px_f, '7d')}"
            )
        is_dup_swing = (
            is_sub24h_dup_7d
            and not pinfo.get("recent_swings_7d")
            and (pinfo.get("swing_low_px"), pinfo.get("swing_high_px")) == (pinfo.get("low_24h"), pinfo.get("high_24h"))
        )
        if pinfo.get("swing_low_px") is not None and pinfo.get("swing_high_px") is not None and not is_dup_swing:
            sw_ral = safe_float(pinfo.get("rally_from_swing_low_pct"), 0.0)
            sw_pb = safe_float(pinfo.get("pullback_from_swing_high_pct"), 0.0)
            maj_sfx = (
                f", major_low={pinfo.get('major_swing_low_px')}"
                f"(rally={safe_float(pinfo.get('rally_from_major_low_pct'), 0.0):+.2f}%)"
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
        if any(k in pinfo for k in ("slope_90s", "slope_3_7m", "slope_15m", "slope_2_5h")):
            gr_v = safe_float(pinfo.get("gradient_recent"), None)
            grad_rec_sfx = f", grad_rec={_fmt_slope_val(gr_v, 4)}" if gr_v is not None and abs(gr_v) > 1e-9 else ""
            gf_v = safe_float(pinfo.get("gradient_recent_fast"), None)
            eps_v = safe_float(pinfo.get("epsilon"), None)
            fast_sfx = (
                f", grad_fast={_fmt_slope_val(gf_v, 4)}/eps={_fmt_slope_val(eps_v, 4, signed=False)}"
                if gf_v is not None and eps_v is not None
                else ""
            )
            g_parts = [
                f"{g_lbl}={_fmt_slope_val(gv, 5)}"
                for g_lbl, g_k in (("90s", "growth_90s"), ("3.7m", "growth_3_7m"), ("15m", "growth_15m"), ("2.5h", "growth_2_5h"))
                if (gv := safe_float(pinfo.get(g_k), None)) is not None
            ]
            if s_core := format_multi_window_slopes_core(pinfo, fallback_full=False):
                extras.append(f"slopes({s_core}{grad_rec_sfx}{fast_sfx})")
            elif g_parts:
                extras.append(f"g_coeff({', '.join(g_parts)})")
        if pinfo.get("vol_1h_pct") is not None:
            extras.append(f"vol_1h={pinfo.get('vol_1h_pct')}%")
        if (kv := safe_float(pinfo.get("kalman_vel"), None)) is not None and abs(kv) >= 1e-9:
            extras.append(f"kalman_vel={_fmt_slope_val(kv, 5)}")
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
        eq_qmeta_sfx = ""
        if is_equity_symbol(sym) and pinfo.get("quote_source"):
            qm_parts = [f"src={pinfo['quote_source']}"]
            if pinfo.get("quote_age_min") is not None:
                qm_parts.append(f"age={pinfo['quote_age_min']}m")
            if pinfo.get("t212_close_px") is not None:
                qm_parts.append(f"t212_close=${pinfo['t212_close_px']}")
            eq_qmeta_sfx = f" [{', '.join(qm_parts)}]"
        extra_str = f" | {' | '.join(extras)}" if extras else ""
        lines.append(f"  * {sym}{sib_lbl}: Price={px_str}{eq_qmeta_sfx}{extra_str}")
    return lines


def _format_3b_source_ages_suffix(
    minfo: Dict[str, Any],
    *,
    now_ts: Optional[float],
    intervals: Dict[str, float],
    ob_only: bool = False,
    include_deriv: bool = True,
) -> str:
    explicit_now = safe_pos_float(now_ts, default=None)
    ref_now = explicit_now if explicit_now is not None else time.time()
    specs = (
        (("ob", "_ob_raw_ts", intervals["orderbook_interval_sec"]),)
        if ob_only
        else (
            ("ob", "_ob_raw_ts", intervals["orderbook_interval_sec"]),
            *((("deriv", "_deriv_ts", intervals["derivatives_interval_sec"]),) if include_deriv else ()),
            ("whale", "_whale_ts", intervals["whale_interval_sec"]),
        )
    )
    parts: List[str] = []
    any_stale = False
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
        is_stale = age_clamped > itv_s * TELEMETRY_SOURCE_STALE_INTERVAL_MULT
        if is_stale:
            any_stale = True
        itv_lbl = format_interval_sec_label(itv_s)
        tag = "[stale]" if is_stale else ""
        parts.append(f"{lbl}={age_clamped:.1f}s/{itv_lbl}{tag}")
    if not parts or (explicit_now is None and not any_stale):
        return ""
    return f", source_ages({', '.join(parts)})"


def _format_microstructure_section(
    microstructure: Dict[str, Any],
    prices_and_trends: Dict[str, Any],
    *,
    now_ts: Optional[float] = None,
) -> List[str]:
    """Format Section 3B (Orderbook Walls, Derivatives & Whale Flow)."""
    lines = ["\n### 3B. PILLAR 2 MICROSTRUCTURE, ORDERBOOK WALLS, DERIVATIVES & WHALE FLOW:"]
    derived_3b_itvs = resolve_derived_telemetry_intervals()
    for row in evaluate_microstructure_prompt_view(microstructure, prices_and_trends):
        sym = row["sym"]
        sib_3b = row["sib_3b"]
        b_canon_3b = row["b_canon"]
        minfo = row["minfo"]
        cur_px = row["cur_px"]
        ob_mid = row["ob_mid"]
        ob_ref_px = row["ob_ref_px"]
        b_bid = row["b_bid"]
        b_ask = row["b_ask"]
        sym_is_eq_3b = row["sym_is_eq"]
        sym_is_tok_xstock_3b = row["sym_is_tok"]
        sym_no_crypto_perp_3b = row["sym_no_perp"]
        shared_perp_sym = row["shared_perp_sym"]

        sym_3b_lbl = f"{sym} ({', '.join(sib_3b)})" if sib_3b else sym
        ref_now_3b = safe_pos_float(now_ts, default=None) or time.time()
        raw_ob_t_3b = minfo.get("_ob_raw_ts") if minfo.get("_ob_raw_ts") is not None else minfo.get("_ob_ts")
        ob_t_3b = normalize_epoch_sec(raw_ob_t_3b, default=0.0) or 0.0
        ob_age_3b = (ref_now_3b - ob_t_3b) if ob_t_3b > 0.0 else None
        ob_is_frozen = bool(
            not sym_is_eq_3b
            and ob_age_3b is not None
            and ob_age_3b > 600.0
            and (now_ts is not None or ob_age_3b <= 7.0 * 86400.0)
        )
        bw_px = minfo.get("bid_wall_px")
        aw_px = minfo.get("ask_wall_px")
        bw_px_f = safe_float(bw_px, 0.0)
        aw_px_f = safe_float(aw_px, 0.0)
        bw_dist = f", dist={(bw_px_f - ob_ref_px) / ob_ref_px * 100.0:+.2f}%" if ob_ref_px > 0 and bw_px_f > 0 else ""
        aw_dist = f", dist={(aw_px_f - ob_ref_px) / ob_ref_px * 100.0:+.2f}%" if ob_ref_px > 0 and aw_px_f > 0 else ""
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
        if (cvd_usd_raw := minfo.get("cvd_15m_usd")) is not None:
            cvd_u = safe_float(cvd_usd_raw, 0.0)
            usd_s = f"{'+' if cvd_u >= 0.0 else '-'}{_fmt_liq_usd(abs(cvd_u))}"
            cvd_b = minfo.get("cvd_15m_base")
            cvd_b_s = f"{safe_float(cvd_b, 0.0):+.3f}".rstrip("0").rstrip(".") if cvd_b is not None else ""
            if cvd_b_s in ("+", "-", ""):
                cvd_b_s = "+0"
            cvd_sfx = (
                f", cvd_15m={cvd_b_s} {b_canon_3b} ({usd_s})"
                if cvd_b is not None
                else f", cvd_15m={usd_s}"
            )
            if mh_parts := format_flow_horizon_parts(
                minfo, lambda v: f"{'+' if v >= 0.0 else '-'}{_fmt_liq_usd(abs(v))}"
            ):
                cvd_sfx += f", flow_horizons({', '.join(mh_parts)})"
            if not ob_is_frozen and (sp_cvd := safe_float(minfo.get("spot_cvd_15m_usd"), None)) is not None:
                sp_s = f"{'+' if sp_cvd >= 0.0 else '-'}{_fmt_liq_usd(abs(sp_cvd))}"
                cvd_sfx += f", spot_cvd_15m={sp_s}{format_spot_cvd_source_tag(minfo)}"
            if sp_vol_parts := format_spot_volume_rvol_parts(minfo, _fmt_liq_usd):
                cvd_sfx += f", {', '.join(sp_vol_parts)}"

        basis_sfx = ""
        basis_parts: List[str] = []
        eq_no_oi_3b = sym_no_crypto_perp_3b and (safe_float(minfo.get("oi_usd"), 0.0) or 0.0) <= 0.0
        if not sym_no_crypto_perp_3b and (b_bps := safe_float(minfo.get("perp_spot_basis_bps"), None)) is not None:
            basis_parts.append(f"perp_basis={b_bps:+.1f}bps")
        if not sym_no_crypto_perp_3b and (f_ann := safe_float(minfo.get("funding_annualized_pct"), None)) is not None:
            basis_parts.append(f"funding_ann={f_ann:+.1f}%")
        if (arb_bps := safe_float(minfo.get("cross_venue_arb_bps"), None)) is not None and minfo.get("cross_venue_peer"):
            basis_parts.append(f"arb_vs_{minfo['cross_venue_peer']}={arb_bps:+.1f}bps")
            if (net_b := safe_float(minfo.get("net_buy_edge_vs_peer_bps"), None)) is not None and minfo.get("best_buy_venue"):
                basis_parts.append(f"net_buy_edge={net_b:+.1f}bps(best_buy={minfo['best_buy_venue']})")
        if basis_parts:
            basis_sfx = f", {', '.join(basis_parts)}"

        liq_sfx = ""
        if not sym_no_crypto_perp_3b and minfo.get("long_liq_px") is not None and minfo.get("short_liq_px") is not None:
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
                l_d = safe_float(
                    minfo.get(dist_k),
                    ((l_px - cur_px) / cur_px) * 100.0 if cur_px > 0 and l_px > 0 else 0.0,
                )
                liq_parts.append(
                    f"{l_lbl}=${minfo.get(px_k)}[{_fmt_liq_usd(minfo.get(usd_k))}, {l_d:+.2f}%]"
                )
            liq_parts.append(f"src={minfo.get('liq_source') or 'model_est'}")
            liq_sfx = f", liq_clusters({', '.join(liq_parts)})"
        if not sym_no_crypto_perp_3b and (
            (rl_l := safe_float(minfo.get("realized_liq_15m_long_usd"), None)) is not None
            and (rl_s := safe_float(minfo.get("realized_liq_15m_short_usd"), None)) is not None
            and (rl_l > 0.0 or rl_s > 0.0 or (safe_float(minfo.get("realized_liq_15m_events"), 0.0) or 0.0) > 0.0)
        ):
            rl_n = safe_int(minfo.get("realized_liq_15m_events"), 0) or 0
            rl_n_sfx = f", n={rl_n}" if rl_n > 0 else ""
            liq_sfx += f", realized_liq_15m(long={_fmt_liq_usd(rl_l)}, short={_fmt_liq_usd(rl_s)}{rl_n_sfx})"
        cd_sfx = ""
        if isinstance(cd_map := minfo.get("cycle_delta"), dict):
            cd_parts: List[str] = []
            if cd_map.get("dt_min") is not None:
                cd_parts.append(f"dt={cd_map['dt_min']}m")
            if (d_ob := safe_float(cd_map.get("ob_imbalance_delta"), None)) is not None and abs(d_ob) >= 0.0005:
                cd_parts.append(f"d_ob={d_ob:+.3f}")
            if (d_cvd := safe_float(cd_map.get("cvd_15m_delta_usd"), None)) is not None and abs(d_cvd) >= 1.0:
                cd_parts.append(f"d_cvd={'+' if d_cvd >= 0 else '-'}{_fmt_liq_usd(abs(d_cvd))}")
            if (d_wl := safe_float(cd_map.get("whale_long_delta_pct"), None)) is not None and abs(d_wl) >= 0.005:
                cd_parts.append(f"d_whale_long={d_wl:+.2f}%")
            if (d_oi := safe_float(cd_map.get("oi_delta_pct"), None)) is not None and abs(d_oi) >= 0.005:
                cd_parts.append(f"d_oi={d_oi:+.2f}%")
            if len(cd_parts) > 1:
                cd_sfx = f", cycle_delta({', '.join(cd_parts)})"
        l1_oi_v = safe_float(minfo.get("native_l1_oi_usd"), None)
        oi_v = safe_float(minfo.get("oi_usd"), None)
        oi_lbl = (
            f"{minfo.get('oi_usd')} (l1_oi={_fmt_liq_usd(l1_oi_v)})"
            if (l1_oi_v is not None and oi_v is not None and abs(l1_oi_v - oi_v) > 1.0)
            else f"{minfo.get('oi_usd')}"
        )
        if shared_perp_sym is not None:
            arb_only_parts = [p for p in basis_parts if p.startswith(("arb_vs_", "net_buy_edge="))]
            arb_sfx = f", {', '.join(arb_only_parts)}" if arb_only_parts else ""
            sib_sp_parts: List[str] = []
            if not ob_is_frozen and (sib_sp_cvd := safe_float(minfo.get("spot_cvd_15m_usd"), None)) is not None and minfo.get("spot_cvd_source") == "real_spot":
                sib_sp_s = f"{'+' if sib_sp_cvd >= 0.0 else '-'}{_fmt_liq_usd(abs(sib_sp_cvd))}"
                sib_sp_parts.append(f"spot_cvd_15m={sib_sp_s}[real_spot]")
            if not ob_is_frozen:
                sib_sp_parts.extend(format_spot_volume_rvol_parts(minfo, _fmt_liq_usd))
            sib_sp_sfx = f", {', '.join(sib_sp_parts)}" if sib_sp_parts else ""
            ob_age_sfx = _format_3b_source_ages_suffix(
                minfo, now_ts=now_ts, intervals=derived_3b_itvs, ob_only=True
            )
            if ob_is_frozen:
                lines.append(
                    f"  * {sym_3b_lbl}: orderbook=STALE_FROZEN({ob_age_3b:.0f}s; walls/imbalance/spot_cvd_15m suppressed)"
                    f"{arb_sfx}{cd_sfx}{ob_age_sfx}, perp_whale_flow=shared_with_{shared_perp_sym}"
                )
            else:
                lines.append(
                    f"  * {sym_3b_lbl}: ob_imbalance={minfo.get('ob_imbalance')} "
                    f"(bid_depth=${minfo.get('bid_depth_usd')}, ask_depth=${minfo.get('ask_depth_usd')}){depth_1pct_sfx}, "
                    f"bid_wall={bw_px} (${minfo.get('bid_wall_usd')}{bw_dist}), "
                    f"ask_wall={aw_px} (${minfo.get('ask_wall_usd')}{aw_dist})"
                    f"{sib_sp_sfx}{arb_sfx}{spread_sfx}{cd_sfx}{ob_age_sfx}, perp_whale_flow=shared_with_{shared_perp_sym}"
                )
            continue
        eq_zero_fund_3b = sym_is_eq_3b
        deriv_oi_sfx = (
            ""
            if (sym_is_tok_xstock_3b or (eq_no_oi_3b and eq_zero_fund_3b))
            else (
                f", oi_usd={oi_lbl} (cboe_options_oi), oi_1h_chg={minfo.get('oi_1h_chg_pct')}%"
                if eq_zero_fund_3b
                else (
                    f", funding={minfo.get('funding_rate')} (pred={minfo.get('pred_funding_rate')}), "
                    f"oi_usd={oi_lbl}, oi_1h_chg={minfo.get('oi_1h_chg_pct')}%"
                )
            )
        )
        src_age_sfx = _format_3b_source_ages_suffix(
            minfo,
            now_ts=now_ts,
            intervals=derived_3b_itvs,
            ob_only=sym_is_tok_xstock_3b,
            include_deriv=bool(deriv_oi_sfx),
        )
        if sym_is_tok_xstock_3b and minfo.get("whale_long_pct") is None and minfo.get("taker_buy_sell_ratio") is None:
            tok_sp_parts: List[str] = []
            if not cvd_sfx and not ob_is_frozen and (tok_sp := safe_float(minfo.get("spot_cvd_15m_usd"), None)) is not None and minfo.get("spot_cvd_source") == "real_spot":
                tok_sp_s = f"{'+' if tok_sp >= 0.0 else '-'}{_fmt_liq_usd(abs(tok_sp))}"
                tok_sp_parts.append(f"spot_cvd_15m={tok_sp_s}[real_spot]")
            if not cvd_sfx and not ob_is_frozen:
                tok_sp_parts.extend(format_spot_volume_rvol_parts(minfo, _fmt_liq_usd))
            tok_sp_sfx = f", {', '.join(tok_sp_parts)}" if tok_sp_parts else ""
            flow_3b_core = f"perp_derivatives=N/A[spot_tokenized_equity]{cvd_sfx}{tok_sp_sfx}"
        else:
            flow_3b_core = (
                f"whale_long_pct={minfo.get('whale_long_pct')}, "
                f"taker_ratio={minfo.get('taker_buy_sell_ratio')} "
                f"(buy_v=${minfo.get('taker_buy_vol_usd')}/sell_v=${minfo.get('taker_sell_vol_usd')}){cvd_sfx}"
            )
        if ob_is_frozen:
            lines.append(
                f"  * {sym_3b_lbl}: orderbook=STALE_FROZEN({ob_age_3b:.0f}s; walls/imbalance/spot_cvd_15m suppressed), "
                f"{flow_3b_core}"
                f"{deriv_oi_sfx}{basis_sfx}{liq_sfx}{cd_sfx}{src_age_sfx}"
            )
        else:
            lines.append(
                f"  * {sym_3b_lbl}: ob_imbalance={minfo.get('ob_imbalance')} "
                f"(bid_depth=${minfo.get('bid_depth_usd')}, ask_depth=${minfo.get('ask_depth_usd')}){depth_1pct_sfx}, "
                f"bid_wall={bw_px} (${minfo.get('bid_wall_usd')}{bw_dist}), "
                f"ask_wall={aw_px} (${minfo.get('ask_wall_usd')}{aw_dist}), "
                f"{flow_3b_core}"
                f"{deriv_oi_sfx}{basis_sfx}{spread_sfx}{liq_sfx}{cd_sfx}{src_age_sfx}"
            )
    return lines


def _format_tradable_position_line(pos: Dict[str, Any], max_usd: float = 2000.0) -> str:
    """Format a single tradable position line for Section 3C from its evaluated view model."""
    pv = (
        pos
        if isinstance(pos, dict) and pos.get("_is_tradable_pos_view") is True
        else evaluate_tradable_position_prompt_view(pos, max_usd=max_usd)
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
            f"{l_q:g}@{l_px}" + (f"({l_pnl:+.2f}%)" if l_pnl is not None else "")
            for l_q, l_px, l_pnl in shown_lots
        ]
        t_extra += f", open_lifo_lots=[{', '.join(lot_parts)}]"
        if (prof := pv["profitable_lifo"]) is not None:
            prof_usd, prof_q, prof_avg_pct = prof
            t_extra += f", profitable_lifo_usd=${prof_usd:.2f}({prof_q:g}_{lot_base_lbl}@{prof_avg_pct:+.2f}%)"
        if (older_q := pv["older_core_qty"]) is not None:
            t_extra += f", older_core_qty={older_q:g}_{lot_base_lbl}"
    if (lifo_vwap := pv["lifo_rem_vwap"]) is not None:
        vwap, vwap_pnl = lifo_vwap
        vwap_sfx = f" (pnl_vs_lifo={vwap_pnl:+.2f}%)" if vwap_pnl is not None else ""
        t_extra += f", lifo_rem_avg_px={vwap}{vwap_sfx}"
    if (lifo_p := pv["lifo_realized_pnl_usd"]) is not None:
        t_extra += f", lifo_realized_pnl=${lifo_p:+.2f}"
    taker_bps = pv["taker_bps"]
    maker_bps = pv["maker_bps"]
    sp_part = f", spread={pv['spread_bps']}bps" if pv["spread_bps"] is not None else ""
    t_extra += (
        f", venue_fee_per_leg={taker_bps:g}bps "
        f"(tier={pv['fee_tier']}, maker={maker_bps:g}bps/taker={taker_bps:g}bps{sp_part})"
    )
    lim_parts: List[str] = []
    if pv["min_notional_usd"] is not None:
        lim_parts.append(f"min_notional=${pv['min_notional_usd']:g}")
    for lim_pfx, lim_val in pv["limit_items"]:
        lim_parts.append(f"{lim_pfx}{lim_val}")
    if lim_parts:
        t_extra += f", limits({', '.join(lim_parts)})"
    fb_val, val_usd = pv["free_base"], pv["value_usd"]
    fb_str = (
        f"{fb_val} (${val_usd} USD)"
        if fb_val is not None and val_usd is not None
        else (f"{fb_val}" if fb_val is not None else "N/A")
    )
    if (lb_q := pv["locked_base_qty"]) is not None:
        lb_u = pv["locked_base_usd"]
        lb_u_s = f" (${lb_u:.2f} USD)" if lb_u is not None else ""
        fb_str += f", locked_base_qty={lb_q:g}{lb_u_s}"
    fq_val = pv["free_quote"]
    fq_bd_items = pv["free_quote_breakdown_items"]
    fq_bd_s = (
        f" ({', '.join(f'{bk}=${bv:,.2f}' for bk, bv in fq_bd_items)})"
        if fq_bd_items
        else ""
    )
    fq_str = f"${fq_val}{fq_bd_s}" if fq_val is not None else "N/A"
    if (lq_u := pv["locked_quote_usd"]) is not None:
        fq_str += f", locked_quote_usd=${lq_u:.2f}"
    return (
        f"    - [{pv['venue']}] {pv['disp_symbol']}: free_{pv['disp_base_lbl']}={fb_str}, "
        f"free_{pv['quote_lbl']}={fq_str}, "
        f"px={pv['px_disp']}{t_extra}"
    )


def _format_recent_order_line(sym: str, oinfo: Dict[str, Any], now_ms: int) -> str:
    """Format a single recent order summary line for Section 3C."""
    ro_ident = resolve_symbol_identity(sym)
    if ro_ident.is_tokenized_equity and ro_ident.display_symbol:
        sym = ro_ident.display_symbol
    open_lots_ro = oinfo.get("open_lots") if isinstance(oinfo.get("open_lots"), list) else []

    def _side_summary(key: str) -> str:
        entry = oinfo.get(key) if isinstance(oinfo.get(key), dict) else {}
        px = round_opt_px(entry.get("price"), 6)
        if px is None:
            return "none"
        ts = normalize_timestamp_ms(entry.get("ts_ms"), default=0) or 0
        age_h = max(0.0, round((now_ms - ts) / 3_600_000.0, 1)) if ts > 0 else None
        tag = entry.get("actor_tag") or format_actor_tag(entry.get("actor"), entry.get("module"), entry.get("rule"))
        qty_v = entry.get("qty")
        qty_pfx = f"{qty_v} @ " if qty_v is not None else ""
        meta_parts = ([f"age={age_h}h"] if age_h is not None else []) + ([f"by={tag}"] if tag else [])
        other_entry = oinfo.get("last_sell") if isinstance(oinfo.get("last_sell"), dict) else {}
        other_ts = normalize_timestamp_ms(other_entry.get("ts_ms"), default=0) or 0
        if key == "last_buy" and is_last_buy_lot_harvested(
            px,
            open_lots_ro,
            last_buy_ts_ms=ts * 1.0,
            last_sell_ts_ms=other_ts * 1.0,
        ):
            meta_parts.append("lot_harvested")
        meta_sfx = f" ({', '.join(meta_parts)})" if meta_parts else ""
        return f"{qty_pfx}{px}{meta_sfx}"

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
    max_usd: float = 2000.0,
    include_macro_benchmarks: bool = True,
) -> List[str]:
    """Format Section 3C (Spot Wallet Balances, Tradable Positions, Cycle Cost Anchors & Last Fills)."""
    scope_s = str(asset_scope or "all").strip().lower() if isinstance(asset_scope, str) else "all"
    lines = ["\n### 3C. SPOT WALLET BALANCES, TRADABLE POSITIONS, CYCLE COST ANCHORS & LAST FILLS:"]
    if isinstance(pre := ps.get("portfolio_risk_engine"), dict) and pre.get("unified_nav_usd"):
        if scope_s == "crypto":
            pre_hdr = "Portfolio Risk Engine (Crypto Scope: binance + kraken + hyperliquid; excl. T212 equities)"
            nav_key_lbl = "crypto_nav"
        elif scope_s in ("equities", "equity"):
            pre_hdr = "Portfolio Risk Engine (Equities Scope: t212; excl. crypto venues)"
            nav_key_lbl = "equities_nav"
        else:
            pre_hdr = "Enterprise Global Portfolio Risk Engine (All Venues)"
            nav_key_lbl = "unified_nav"
        pre_parts = [
            f"{nav_key_lbl}=${safe_float(pre.get('unified_nav_usd'), 0.0):,.2f}",
            f"invested=${safe_float(pre.get('invested_exposure_usd'), 0.0):,.2f}",
            f"free_cash=${safe_float(pre.get('global_free_cash_usd'), 0.0):,.2f} ({pre.get('global_free_cash_pct')}%)",
        ]
        if (lq_pre := safe_float(pre.get("locked_quote_usd"), 0.0)) > 0.0:
            pre_parts.append(f"locked_quote=${lq_pre:,.2f} ({pre.get('locked_quote_pct')}%)")
        if isinstance(vn_bd := pre.get("venue_nav_breakdown_usd"), dict) and vn_bd:
            vn_s = ", ".join(
                f"{vk}=${safe_float(vv, 0.0):,.2f}"
                for vk, vv in sorted(vn_bd.items())
                if safe_float(vv, 0.0) > 0.0
            )
            if vn_s:
                pre_parts.append(f"venues({vn_s})")
        if scope_s not in ("equities", "equity"):
            pre_parts.extend([
                f"portfolio_btc_beta={pre.get('portfolio_btc_beta')}",
                f"avg_corr_24h={pre.get('avg_corr_vs_btc_24h')}",
            ])
        pre_parts.append(
            f"intraday_var_95=${safe_float(pre.get('intraday_var_95_usd'), 0.0):,.2f} ({pre.get('intraday_var_95_pct')}%)"
        )
        if pre.get("intraday_cvar_975_usd") is not None:
            pre_parts.append(
                f"cvar_975=${safe_float(pre.get('intraday_cvar_975_usd'), 0.0):,.2f} ({pre.get('intraday_cvar_975_pct')}%)"
            )
        for k_pre, lbl_pre, fmt_pre in (
            ("vol_target_tranche_usd", "vol_target_tranche=$", ",.0f"),
            ("subcycle_lifo_realized_pnl_usd", "subcycle_lifo_pnl=$", "+.2f"),
            ("est_recent_fee_drag_usd", "est_fee_drag=$", ".2f"),
        ):
            if pre.get(k_pre) is not None:
                pre_parts.append(f"{lbl_pre}{format(safe_float(pre.get(k_pre), 0.0), fmt_pre)}")
        pt_parts: List[str] = []
        if (em_v := safe_float(pre.get("avg_entry_markout_bps"), None)) is not None:
            pt_parts.append(f"entry={em_v:+.0f}bps")
        if (xm_v := safe_float(pre.get("avg_exit_markout_bps"), None)) is not None:
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
            lines.append(f"  * Available Free Quote Cash per Venue: {', '.join(cash_parts)}")
    view = evaluate_portfolio_prompt_view(ps, asset_scope=asset_scope, max_usd=max_usd)
    if view["tradable_positions"]:
        lines.append("  * Live Tradable Spot Instruments (Eligible for Autonomous BUY / SELL):")
        for pos_view in view["tradable_positions"]:
            lines.append(_format_tradable_position_line(pos_view, max_usd=max_usd))
    if view["has_open_orders"]:
        lines.append(
            f"  * Active Open Orders on Exchange (Locking Liquidity) \u2014 open_limit_orders: {view['clean_open_limit_orders']}"
        )
    else:
        lines.append(
            "  * Active Open Limit Orders on Exchange \u2014 open_limit_orders: [] (none resting)"
        )
    for venue_lbl, venue_id, cyc_items in view["venue_cycles"]:
        lines.append(f"  * {venue_lbl} Active Cycle Cost & Drawdown Anchors:")
        for c_sym, c_data in cyc_items:
            lines.append(_format_venue_cycle_line(venue_id, c_sym, c_data))
    if xs_view := view["kraken_xstocks"]:
        xs_payload = xs_view["payload"]
        xs_pair_lbl = xs_view["pair_lbl"]
        xs_ipo_lbl = xs_view["ipo_lbl"]
        xs_sib = xs_view["t212_sibling"]
        if scope_s in ("equities", "equity"):
            lines.append(
                f"  * Cross-Venue Kraken Tokenized Sibling Reference (READ-ONLY in Equities Scope; "
                f"24/7 {xs_pair_lbl} quote/inventory context for T212 {xs_sib}): {xs_payload}"
            )
        else:
            lines.append(
                f"  * Kraken Tokenized Equities / xStocks (API-TRADABLE 24/7 on 'kraken' at 0bps maker/taker fee; "
                f"{xs_pair_lbl} Active Swing Directive{xs_ipo_lbl} & T212 {xs_sib} context): {xs_payload}"
            )
    if view["crypto_bellwethers"]:
        lines.append(
            f"  * Cross-Market Live Crypto Telemetry (Raw 24/7 Prices, Multi-Horizon Deltas & Microstructure): {view['crypto_bellwethers']}"
        )
    if view["equity_bellwethers"]:
        eq_hdr_cyc_sfx = "" if scope_s == "crypto" else " & Active Cycles"
        lines.append(
            f"  * Cross-Market US Equities Telemetry (Live/Extended & T212 Quotes, Multi-Horizon Deltas{eq_hdr_cyc_sfx}): {view['equity_bellwethers']}"
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
    now_ms = math.trunc(time.time() * 1000)
    for sym, oinfo in sorted((ps.get("recent_orders") or {}).items()):
        if isinstance(oinfo, dict):
            lines.append(_format_recent_order_line(sym, oinfo, now_ms))
    if view["clean_last_cycle_execution_status"]:
        lines.append(
            f"  * Last Cycle Execution Status \u2014 last_cycle_execution_status: {view['clean_last_cycle_execution_status']}"
        )
    for _, st_label, st_val in view["runtime_states"]:
        lines.append(f"  * {st_label}: {st_val}")
    return lines


