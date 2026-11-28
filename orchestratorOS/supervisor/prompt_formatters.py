"""Pure formatting helpers for Autonomous AI Reconciler Gemini prompts.

Separates prompt presentation (Sections 1, 2, 3A, 3B, 3C) and symbol price
resolution from telemetry harvesting and order execution in
`autonomous_ai_reconciler.py`.
Section-level and line-level formatters live in `orchestratorOS.supervisor.prompt_sections`.
"""

from datetime import datetime
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from botcore import (
    first_epoch_sec,
    is_us_rth_open,
    load_key_value_conf,
    normalize_epoch_sec,
    parse_symbol_list,
    safe_float,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
    topic_spec,
)
from cachepaths import is_kraken_tokenized_asset
from instrument_registry import discover_tradable_specs
from intelligence.macro_sentiment_ssot import (
    CRYPTO_ONLY_MACRO_SENSOR_KEYS,
    format_age_sec_compact,
    format_telemetry_domain_age_tag,
    resolve_telemetry_domain_stale_thresholds_sec,
)
from intelligence.sentiment.evaluators.prompt_view_evaluator import resolve_scope_presentation
from orchestratorOS.lib.technical_repair_ssot import (
    ALL_DIAGNOSED_CATEGORIES,
    ALL_REPAIR_ACTIONS,
    AUDITED_CONFIG_FILES,
    DIAGNOSED_SEVERITIES,
)
from orchestratorOS.supervisor.crypto_executor_delta_gate import _extract_open_order_filled_qty
from orchestratorOS.supervisor.prompt_sections import (
    _extract_active_portfolio_symbols,
    _fmt_plain_float,
    _format_macro_and_news_section,
    _format_market_prices_section,
    _format_microstructure_section,
    _format_operator_regime_note_line,
    _format_portfolio_section,
    _format_sre_and_fleet_section,
    _format_unified_markets_and_bellwethers_sections,
)
from providers.ai_circuit_breaker import (
    DEFAULT_MAX_HOURLY_REPLACES_PER_SYMBOL,
    DEFAULT_MIN_TARGET_RR,
    DEFAULT_REPLACE_MIN_PRICE_DELTA_PCT,
    DEFAULT_TARGET_EV_FEE_MULT,
    DEFAULT_XSTOCK_SELL_COOLDOWN_HOURS,
    DEFAULT_XSTOCK_SELL_CRASH_MIN_CONFIDENCE,
    DEFAULT_XSTOCK_SELL_CRASH_MIN_TARGET_DROP_PCT,
    DEFAULT_XSTOCK_SELL_MIN_CONFIDENCE,
    DEFAULT_XSTOCK_SELL_MIN_PROFIT_PCT,
    DEFAULT_XSTOCK_SELL_MIN_TARGET_DROP_PCT,
    parse_intent_circuit_breaker_config,
)
from providers.intent_telemetry import evaluate_open_order_front_run_rr

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LIVE_DOMAIN_AGE_KEYS = frozenset({
    "prices",
    "prices_newest",
    "prices_oldest",
    "orderbook",
    "orderbook_newest",
    "orderbook_oldest",
    "derivatives_oldest",
    "whale_oldest",
    "short_trends",
    "venue_balances",
})
_TRIGGER_SYM_TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_]{2,18}\b")

# SSOT breakout & maker-LIMIT execution policy clauses shared by cold (FULL_KEYFRAME) and warm (WARM_DELTA) prompts.
_CRYPTO_BREAKOUT_CONFIRM_CONDITIONS = (
    "`vel_90s>0`, `vel_5m>0`, `acc_90s>=0`, `rvol>=1.2x`, supportive `cvd`/`taker_ratio`, clear runway below ask walls"
)
_CRYPTO_BREAKOUT_EXEC_CLAUSE = (
    'enter near-BBO post-only maker `LIMIT` (explicit near-BBO `limit_price`, `urgency:"NORMAL"`, '
    "positive-EV `target_px`, tight `invalidation_px`) even at high `pos_15m` (80\u2013100%); skip if wall-capped/distributing"
)
_CRYPTO_POST_ONLY_LIMIT_EXEC_SPEC = (
    'post-only `"order_type": "LIMIT"` (never MARKET; gateway re-checks live price & steps 1 tick onto maker book)'
)
_CRYPTO_POST_ONLY_LIMIT_COMPACT_SPEC = "post-only spot maker-LIMIT (never MARKET; 1-tick maker step)"


def _format_crypto_entry_playbook_clauses(
    half_max_usd: float,
    full_max_usd: float,
    *,
    compact: bool,
) -> str:
    """Render crypto deep-dip and confirmed-breakout entry rules from one SSOT definition for cold and warm prompts."""
    if compact:
        return (
            f"- Entry playbook: (1) Deep dips (deceleration+absorption): size decisively at ${half_max_usd:g}\u2013${full_max_usd:g} when cash is ample; "
            f"(2) Confirmed breakouts ({_CRYPTO_BREAKOUT_CONFIRM_CONDITIONS}): {_CRYPTO_BREAKOUT_EXEC_CLAUSE}.\n"
        )
    return (
        f"1) Deep dips (deceleration+absorption): size decisively at ${half_max_usd:g}\u2013${full_max_usd:g} when cash is ample (no timid $500 tickets). "
        f"2) Confirmed breakouts ({_CRYPTO_BREAKOUT_CONFIRM_CONDITIONS}): {_CRYPTO_BREAKOUT_EXEC_CLAUSE}. "
        "3) De-risk on negative velocity + unabsorbed selling.\n"
    )


def _build_section0_trigger_context(
    wake_reason: Optional[str],
    focus_symbols: Optional[List[str]],
    portfolio_summary: Dict[str, Any],
    known_symbols: set[str],
    *,
    now_ts: Optional[float] = None,
    min_target_rr: float = DEFAULT_MIN_TARGET_RR,
) -> Tuple[Optional[str], set[str]]:
    """Build Section 0 Trigger line and extract symbols that should be promoted to top of Section 3."""
    parts: List[str] = []
    trigger_syms: set[str] = set()
    known_upper = {str(s).strip().upper(): str(s).strip() for s in known_symbols if s}

    clean_wake = str(wake_reason or "").strip()
    wake_lc = clean_wake.lower()
    is_prev_actionable_wake = wake_lc == "prev_eval_actionable" or wake_lc.startswith("prev_eval_actionable")
    if clean_wake and not is_prev_actionable_wake:
        parts.append(clean_wake)
        for tok in _TRIGGER_SYM_TOKEN_RE.findall(clean_wake):
            tu = tok.upper()
            if tu in known_upper:
                trigger_syms.add(known_upper[tu])

    if is_prev_actionable_wake and isinstance(portfolio_summary, dict):
        raw_lces_pa = portfolio_summary.get("last_cycle_execution_status")
        lces_pa_items = (
            raw_lces_pa
            if isinstance(raw_lces_pa, list)
            else ([raw_lces_pa] if isinstance(raw_lces_pa, dict) else [])
        )
        pa_summaries: List[str] = []
        seen_pa: set[Tuple[str, str, str, str]] = set()
        for item in lces_pa_items:
            if not isinstance(item, dict):
                continue
            age_m = safe_float(item.get("age_min"), None)
            if age_m is not None and age_m > 30.0:
                continue
            pa_sym = str(item.get("symbol") or "").strip()
            if not pa_sym:
                continue
            pa_side = str(item.get("side") or item.get("decision") or "").strip().upper()
            raw_st_pa = str(item.get("status") or "").strip().upper()
            ven_st_pa = str(item.get("venue_status") or "").strip().upper()
            pa_st = "RESTING_IN_BOOK" if raw_st_pa == "RESTING_IN_BOOK" else (ven_st_pa or raw_st_pa)
            if pa_side in ("HOLD", "SKIPPED", "NOOP") or pa_st in ("HOLD", "SKIPPED", "NOOP"):
                continue
            pa_ven = str(item.get("venue") or "").strip()
            ven_lbl = pa_ven.capitalize() if pa_ven else ""
            pa_otype = str(item.get("order_type") or "").strip().upper()
            pa_px = item.get("fill_px")
            if pa_px is None:
                pa_px = item.get("limit_px") if item.get("limit_px") is not None else item.get("limit_price")
            dedup_pa = (ven_lbl.lower(), pa_sym.upper(), pa_side, pa_st)
            if dedup_pa in seen_pa:
                continue
            seen_pa.add(dedup_pa)
            ven_pfx = f"{ven_lbl}:" if ven_lbl else ""
            otype_s = f" {pa_otype}" if pa_otype else ""
            px_s = f"@{_fmt_plain_float(pa_px)}" if pa_px is not None else ""
            st_s = f" {pa_st}" if pa_st else ""
            pa_summaries.append(f"{ven_pfx}{pa_sym} {pa_side}{otype_s}{px_s}{st_s}".strip())
            trigger_syms.add(pa_sym)
            if len(pa_summaries) >= 3:
                break
        if pa_summaries:
            parts.append("post_action")
            parts.extend(pa_summaries)
        else:
            parts.append(clean_wake)
    elif is_prev_actionable_wake:
        parts.append(clean_wake)

    clean_focus = parse_symbol_list(focus_symbols) if focus_symbols else []
    if clean_focus:
        trigger_syms.update(clean_focus)
        if not clean_wake or "focus_symbols(" not in wake_lc:
            parts.append(f"focus={','.join(clean_focus)}")

    raw_open = (
        portfolio_summary.get("open_limit_orders")
        if isinstance(portfolio_summary, dict) and isinstance(portfolio_summary.get("open_limit_orders"), list)
        else (portfolio_summary.get("open_orders") if isinstance(portfolio_summary, dict) else None)
    )
    raw_plans = (
        portfolio_summary.get("active_ai_trade_plans")
        if isinstance(portfolio_summary, dict) and isinstance(portfolio_summary.get("active_ai_trade_plans"), list)
        else []
    )
    eff_min_rr = max(0.0, safe_float(min_target_rr, default=DEFAULT_MIN_TARGET_RR) or 0.0)
    used_wt_plans: set[int] = set()
    if isinstance(raw_open, list):
        for oo in raw_open:
            if not isinstance(oo, dict):
                continue
            sym = str(oo.get("symbol") or "").strip()
            if not sym:
                continue
            raw_qty_v = (
                oo.get("qty")
                if oo.get("qty") is not None
                else (oo.get("origQty") if oo.get("origQty") is not None else oo.get("origSz"))
            )
            qty_v = safe_pos_float(raw_qty_v, default=None)
            filled_v = _extract_open_order_filled_qty(oo)
            st_u = str(oo.get("status") or "").strip().upper()
            if filled_v > 0.0 or st_u == "PARTIALLY_FILLED":
                rem_s = ""
                if qty_v is not None and qty_v > filled_v:
                    rem_str = _fmt_plain_float(round(max(0.0, qty_v - filled_v), 8))
                    if rem_str.endswith(".0"):
                        rem_str = rem_str[:-2]
                    rem_s = f"(rem={rem_str})"
                qty_s = f"/{_fmt_plain_float(qty_v)}" if qty_v is not None else ""
                parts.append(f"{sym} partial_fill {_fmt_plain_float(filled_v)}{qty_s}{rem_s}")
                trigger_syms.add(sym)
            if "open_order_wall_trapped" in wake_lc and (fr_v := safe_pos_float(oo.get("recommended_front_run_px"), default=None)) is not None:
                fr_ok, rr_fr = evaluate_open_order_front_run_rr(
                    oo, raw_plans, used_plan_indices=used_wt_plans, min_target_rr=eff_min_rr
                )
                if fr_ok:
                    rr_sfx = f",RR={rr_fr:.2f}x" if rr_fr is not None else ""
                    fr_lbl = _fmt_plain_float(fr_v) if (0.0 < abs(fr_v) < 1e-4 or abs(fr_v) >= 1e6) else f"{fr_v:g}"
                    parts.append(f"{sym} wall_trapped(front_run={fr_lbl}{rr_sfx})")
                    trigger_syms.add(sym)
            if "open_order_stale_or_far" in wake_lc:
                dist_v = safe_float(oo.get("dist_from_spot_pct"), None)
                fp_v = safe_float(oo.get("fill_prob_1h_pct"), None)
                if (fp_v is not None and fp_v <= 5.0) or (dist_v is not None and abs(dist_v) > 4.0):
                    parts.append(f"{sym} stale_or_far")
                    trigger_syms.add(sym)

    want_fill_ctx = "new_fill" in wake_lc
    want_fresh_fill_with_focus = not want_fill_ctx and "focus_symbols(" in wake_lc
    if (want_fill_ctx or want_fresh_fill_with_focus) and isinstance(portfolio_summary, dict):
        fill_max_age = 20.0 if want_fill_ctx else 5.0
        seen_fills: set[Tuple[str, str]] = set()
        raw_lces = portfolio_summary.get("last_cycle_execution_status")
        lces_items = raw_lces if isinstance(raw_lces, list) else ([raw_lces] if isinstance(raw_lces, dict) else [])
        for item in lces_items:
            if not isinstance(item, dict):
                continue
            st_l = str(item.get("status") or item.get("venue_status") or "").strip().upper()
            age_m = safe_float(item.get("age_min"), None)
            if st_l in ("FILLED", "PARTIALLY_FILLED") and (
                (age_m is None and want_fill_ctx) or (age_m is not None and age_m <= fill_max_age)
            ):
                if f_sym := str(item.get("symbol") or "").strip():
                    f_side = str(item.get("side") or item.get("decision") or "").strip().upper()
                    f_px = item.get("fill_px") or item.get("limit_px") or item.get("limit_price")
                    px_s = f"@{_fmt_plain_float(f_px)}" if f_px is not None else ""
                    parts.append(f"{f_sym} {f_side} {st_l}{px_s}".strip())
                    trigger_syms.add(f_sym)
                    seen_fills.add((f_sym.upper(), f_side))
        recent_orders = portfolio_summary.get("recent_orders") if want_fill_ctx else None
        if isinstance(recent_orders, dict):
            eff_now_nf = safe_pos_float(now_ts, default=None) or time.time()
            recent_hits: List[Tuple[float, str, str, Any, str]] = []
            for ro_sym, ro_info in recent_orders.items():
                if not isinstance(ro_info, dict) or not str(ro_sym or "").strip():
                    continue
                ro_sym_s = str(ro_sym).strip()
                for leg_k, leg_side in (("last_buy", "BUY"), ("last_sell", "SELL")):
                    leg = ro_info.get(leg_k)
                    if not isinstance(leg, dict) or (ro_sym_s.upper(), leg_side) in seen_fills:
                        continue
                    leg_ts = first_epoch_sec(
                        leg.get("ts_ms"),
                        leg.get("ts"),
                        default=0.0,
                        allow_iso=True,
                    ) or 0.0
                    if leg_ts > 0.0 and -60.0 <= (eff_now_nf - leg_ts) <= 1200.0:
                        actor_u = str(leg.get("actor") or "").strip().upper()
                        recent_hits.append((leg_ts, ro_sym_s, leg_side, leg.get("price"), actor_u))
            for _, ro_sym_s, leg_side, leg_px, actor_u in sorted(recent_hits, key=lambda x: x[0], reverse=True):
                px_s = f"@{_fmt_plain_float(leg_px)}" if leg_px is not None else ""
                act_s = f"({actor_u})" if actor_u in ("BOT", "MANUAL") else ""
                parts.append(f"{ro_sym_s} {leg_side} FILLED{px_s}{act_s}")
                trigger_syms.add(ro_sym_s)
                seen_fills.add((ro_sym_s.upper(), leg_side))

    if ("invalidat" in wake_lc or "target_hit" in wake_lc) and isinstance(portfolio_summary, dict):
        raw_plans = portfolio_summary.get("active_ai_trade_plans")
        if isinstance(raw_plans, list):
            for pl in raw_plans:
                if not isinstance(pl, dict):
                    continue
                pl_st = str(pl.get("status") or "").strip().upper()
                if (pl_st == "INVALIDATED" and "invalidat" in wake_lc) or (pl_st == "TARGET_HIT" and "target_hit" in wake_lc):
                    if p_sym := str(pl.get("symbol") or "").strip():
                        trigger_syms.add(p_sym)

    if not parts:
        return None, trigger_syms
    return f"- Trigger: {' | '.join(parts)}", trigger_syms


def _resolve_gateway_survival_prompt_thresholds(
    workspace_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Read AI survival circuit-breaker and tokenized-equity sell thresholds from order_guard.conf SSOT."""
    ws = workspace_dir or _ROOT_DIR
    try:
        conf = load_key_value_conf(ws, _ROOT_DIR)
        cb_cfg = parse_intent_circuit_breaker_config(conf)
    except Exception:
        cb_cfg = {}
    return {
        "xstock_sell_min_profit_pct": safe_nonneg_float(
            cb_cfg.get("xstock_sell_min_profit_pct"), default=DEFAULT_XSTOCK_SELL_MIN_PROFIT_PCT
        ),
        "xstock_sell_min_target_drop_pct": safe_nonneg_float(
            cb_cfg.get("xstock_sell_min_target_drop_pct"), default=DEFAULT_XSTOCK_SELL_MIN_TARGET_DROP_PCT
        ),
        "xstock_sell_min_confidence": safe_nonneg_float(
            cb_cfg.get("xstock_sell_min_confidence"), default=DEFAULT_XSTOCK_SELL_MIN_CONFIDENCE
        ),
        "xstock_sell_cooldown_hours": safe_nonneg_float(
            cb_cfg.get("xstock_sell_cooldown_hours"), default=DEFAULT_XSTOCK_SELL_COOLDOWN_HOURS
        ),
        "xstock_sell_crash_min_target_drop_pct": safe_nonneg_float(
            cb_cfg.get("xstock_sell_crash_min_target_drop_pct"),
            default=DEFAULT_XSTOCK_SELL_CRASH_MIN_TARGET_DROP_PCT,
        ),
        "xstock_sell_crash_min_confidence": safe_nonneg_float(
            cb_cfg.get("xstock_sell_crash_min_confidence"), default=DEFAULT_XSTOCK_SELL_CRASH_MIN_CONFIDENCE
        ),
        "target_ev_fee_mult": safe_nonneg_float(
            cb_cfg.get("target_ev_fee_mult"), default=DEFAULT_TARGET_EV_FEE_MULT
        ),
        "min_target_rr": safe_nonneg_float(
            cb_cfg.get("min_target_rr"), default=DEFAULT_MIN_TARGET_RR
        ),
        "max_hourly_replaces_per_symbol": safe_nonneg_int(
            cb_cfg.get("max_hourly_replaces_per_symbol"),
            default=DEFAULT_MAX_HOURLY_REPLACES_PER_SYMBOL,
        ),
        "replace_min_price_delta_pct": safe_nonneg_float(
            cb_cfg.get("replace_min_price_delta_pct"), default=DEFAULT_REPLACE_MIN_PRICE_DELTA_PCT
        ),
    }


def _resolve_xstock_sell_prompt_thresholds(
    workspace_dir: Optional[str] = None,
) -> Tuple[float, float, float, float, float]:
    """Read SELL-only tokenized equity profit/drop/confidence/crash thresholds from order_guard.conf SSOT."""
    th = _resolve_gateway_survival_prompt_thresholds(workspace_dir)
    return (
        th["xstock_sell_min_profit_pct"],
        th["xstock_sell_min_target_drop_pct"],
        th["xstock_sell_min_confidence"],
        th["xstock_sell_crash_min_target_drop_pct"],
        th["xstock_sell_crash_min_confidence"],
    )


def _build_dynamic_target_desc(
    is_equities: bool, *, workspace_dir: Optional[str] = None
) -> Tuple[str, str]:
    """Format active tradable instruments per venue dynamically from instruments.conf SSOT."""
    ws = workspace_dir or _ROOT_DIR
    scope = "equities" if is_equities else "crypto"
    try:
        specs = discover_tradable_specs(ws, asset_scope=scope, include_closed_rth=True)
    except Exception:
        specs = {}
    by_venue: Dict[str, List[str]] = {}
    xstock_syms: List[str] = []
    for item in specs.values():
        prov = str(getattr(item, "provider", "") or "").strip().lower()
        sym = str(getattr(item, "symbol", "") or "").strip()
        if not (prov and sym):
            continue
        is_xs = is_kraken_tokenized_asset(sym)
        if is_xs and sym not in xstock_syms:
            xstock_syms.append(sym)
        lbl = f"0bps SELL-only xStock {sym}" if (not is_equities and is_xs) else sym
        bucket = by_venue.setdefault(prov, [])
        if lbl not in bucket:
            bucket.append(lbl)

    if is_equities:
        eq_syms = [s for syms in by_venue.values() for s in syms]
        eq_list = ", ".join(eq_syms) if eq_syms else "enabled T212 US equities"
        return (
            f"ALL live tradable US equity shares on 't212' ({eq_list} during US Regular Trading Hours, "
            "cross-referencing Kraken 24/7 xStock exposure)",
            "xStocks",
        )
    venue_parts = [f"{', '.join(syms)} on '{prov}'" for prov, syms in by_venue.items() if syms]
    target_desc = (
        f"ALL live tradable crypto & 24/7 tokenized xStock pairs ({'; '.join(venue_parts)})"
        if venue_parts
        else "ALL live tradable crypto & 24/7 tokenized xStock pairs configured in instruments.conf"
    )
    xs_lbl = (
        ", ".join(f"`{s}`" for s in xstock_syms)
        if xstock_syms
        else "tokenized xStocks"
    )
    return target_desc, xs_lbl


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
    compact_rules: bool = False,
    delta_plan: Optional[Any] = None,
    wake_reason: Optional[str] = None,
    iteration_num: Optional[int] = None,
    prev_eval_elapsed_sec: Optional[float] = None,
    **_extra: Any,
) -> str:
    """Build the topic-specific prompt for Gemini (auto_technical_repair or auto_*trade_executor)."""
    if spec is None:
        spec = topic_spec(topic_key or _extra.get("topic") or "auto_cryptotrade_executor")
    lookback_hours = safe_pos_float(lookback_hours, default=1.0)
    min_usd = safe_pos_float(min_notional_usd if min_notional_usd is not None else min_usd, default=10.0)
    max_usd = safe_pos_float(max_notional_usd if max_notional_usd is not None else max_usd, default=2000.0)
    min_conf = safe_pos_float(min_confidence if min_confidence is not None else min_conf, default=0.85)

    is_pure_sre = bool(spec.can_self_heal and not spec.can_trade)
    eval_now_ts = safe_pos_float(getattr(sensors, "evaluated_at_ts", None), default=None)
    clock_dt = (
        datetime.fromtimestamp(eval_now_ts, tz=ZoneInfo("Europe/Bucharest"))
        if eval_now_ts is not None
        else datetime.now(ZoneInfo("Europe/Bucharest"))
    )
    now_ts_str = clock_dt.strftime("%Y-%m-%d %H:%M:%S %Z (%a)").strip()
    iter_n = safe_nonneg_int(iteration_num, default=0) if iteration_num is not None else 0
    elapsed_f = safe_nonneg_float(prev_eval_elapsed_sec, default=None) if prev_eval_elapsed_sec is not None else None
    if elapsed_f is not None and 0.0 <= elapsed_f < 7 * 86400.0:
        if elapsed_f >= 48 * 3600.0:
            elapsed_lbl = f"{elapsed_f / 86400.0:.1f}d"
        elif elapsed_f >= 3600.0:
            elapsed_lbl = f"{elapsed_f / 3600.0:.1f}h"
        else:
            elapsed_lbl = format_age_sec_compact(elapsed_f)
        dt_sfx = f" (+{elapsed_lbl} vs prev)"
    else:
        dt_sfx = ""
    iter_hdr_sfx = (
        f" | Iter #{iter_n}{dt_sfx}"
        if iter_n > 0
        else (f" |{dt_sfx}" if dt_sfx else "")
    )
    context_items: List[str] = [
        f"### 0. CLOCK & SNAPSHOT: {now_ts_str}{iter_hdr_sfx}\n"
    ]
    prices_and_trends = getattr(sensors, "prices_and_trends", None) or {}
    microstructure = getattr(sensors, "microstructure", None) or {}
    portfolio_summary = getattr(sensors, "portfolio_summary", None) or {}
    trigger_syms: set[str] = set()
    gw_th = _resolve_gateway_survival_prompt_thresholds()
    op_note_rendered_in_s0 = False
    if not is_pure_sre:
        known_syms = set(prices_and_trends.keys()) | set(microstructure.keys())
        trigger_line, trigger_syms = _build_section0_trigger_context(
            wake_reason,
            focus_symbols,
            portfolio_summary,
            known_syms,
            now_ts=eval_now_ts,
            min_target_rr=gw_th["min_target_rr"],
        )
        if trigger_line:
            context_items.append(trigger_line)
        wake_is_op_note_upd = bool(wake_reason and "operator_note_updated" in str(wake_reason).lower())
        delta_op_note_new = bool(
            delta_plan is not None and delta_plan.action_for("operator_regime_note") == "SHOW_NEW"
        )
        if wake_is_op_note_upd or delta_op_note_new:
            op_note_s0_line = _format_operator_regime_note_line(
                sensors,
                delta_plan=delta_plan,
                force_show=wake_is_op_note_upd,
            )
            if op_note_s0_line:
                context_items.append(op_note_s0_line)
                op_note_rendered_in_s0 = True
    domain_ages = getattr(sensors, "source_ages_sec", None)
    spec_scope_norm = str(getattr(spec, "asset_scope", "all") or "all").strip().lower()
    skip_domain_keys = (
        CRYPTO_ONLY_MACRO_SENSOR_KEYS | {"derivatives_oldest", "whale_oldest"}
        if spec_scope_norm in ("equities", "equity")
        else frozenset()
    )
    if not is_pure_sre and isinstance(domain_ages, dict) and domain_ages:
        eff_now_s0 = eval_now_ts if eval_now_ts is not None else time.time()
        stale_thrs = resolve_telemetry_domain_stale_thresholds_sec(spec_scope_norm, now_ts=eff_now_s0)
        us_rth_open = is_us_rth_open(eff_now_s0)
        live_age_tokens: List[str] = []
        macro_age_tokens: List[str] = []
        for k, v in domain_ages.items():
            if k in skip_domain_keys:
                continue
            v_f = safe_pos_float(v, default=None)
            if v_f is None and v != 0.0:
                continue
            stale_tag = format_telemetry_domain_age_tag(
                k,
                v_f or 0.0,
                asset_scope=spec_scope_norm,
                us_rth_open=us_rth_open,
                thresholds=stale_thrs,
            )
            tok = f"{k}={format_age_sec_compact(v_f or 0.0)}{stale_tag}"
            if k in _LIVE_DOMAIN_AGE_KEYS:
                live_age_tokens.append(tok)
            else:
                macro_age_tokens.append(tok)
        if live_age_tokens and macro_age_tokens:
            context_items.append("- Snapshot Domain Ages:")
            context_items.append(f"  * Live: {', '.join(live_age_tokens)}")
            context_items.append(f"  * Macro: {', '.join(macro_age_tokens)}")
        elif live_age_tokens or macro_age_tokens:
            context_items.append(f"- Snapshot Domain Ages: {', '.join(live_age_tokens or macro_age_tokens)}")
    if not is_pure_sre and delta_plan is not None:
        context_items.extend(delta_plan.format_section0_lines())
    if spec.can_self_heal:
        context_items.extend(_format_sre_and_fleet_section(sensors, logs, lookback_hours))
    scope_spec = resolve_scope_presentation(spec_scope_norm)
    is_equities = scope_spec.is_equities
    active_obs_syms = (
        _extract_active_portfolio_symbols(portfolio_summary, focus_symbols=focus_symbols)
        if not is_pure_sre
        else set()
    )
    if not is_pure_sre and portfolio_summary:
        context_items.extend(
            _format_portfolio_section(
                portfolio_summary,
                getattr(spec, "asset_scope", "all"),
                min_usd=min_usd,
                max_usd=max_usd,
                include_macro_benchmarks=False,
                include_cross_market=False,
                now_ts=eval_now_ts,
                delta_plan=delta_plan,
            )
        )
    if not is_pure_sre:
        context_items.extend(
            _format_macro_and_news_section(
                sensors,
                getattr(spec, "asset_scope", "all"),
                now_ts=eval_now_ts,
                delta_plan=delta_plan,
                include_cross_market=True,
                suppress_operator_note=op_note_rendered_in_s0,
            )
        )
    if not is_pure_sre and (prices_and_trends or microstructure):
        held_syms = {
            str(p.get("symbol") or "").strip().upper()
            for p in (portfolio_summary.get("tradable_positions") or ())
            if isinstance(p, dict)
            and p.get("symbol")
            and (
                (safe_nonneg_float(p.get("value_usd"), default=0.0) or 0.0) >= 2.0
                or (safe_nonneg_float(p.get("locked_base_qty"), default=0.0) or 0.0) > 0.0
            )
        }
        open_ord_syms = {
            str(o.get("symbol") or "").strip().upper()
            for o in (portfolio_summary.get("open_limit_orders") or ())
            if isinstance(o, dict) and o.get("symbol")
        }
        context_items.extend(
            _format_unified_markets_and_bellwethers_sections(
                prices_and_trends,
                microstructure,
                now_ts=eval_now_ts,
                asset_scope=spec_scope_norm,
                compact_observe_only=True,
                active_symbols=active_obs_syms,
                held_symbols=held_syms,
                open_order_symbols=open_ord_syms,
                priority_symbols=trigger_syms,
            )
        )
    if not is_pure_sre and focus_symbols:
        clean_syms = parse_symbol_list(focus_symbols)
        if clean_syms:
            context_items.append(
                f"\n### 5. TACTICAL PRIORITY SYMBOLS: {', '.join(clean_syms)}\n"
                f"- Pay heightened attention to these symbols alongside all other live tradable {scope_spec.priority_universe_lbl} above."
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
            f"1. Audit functional system anomalies: fleet processes and modules (procs.conf), bot heartbeats, Linux OS resources (CPU load, RAM, disk), internet & VPN tunnel (wgpia0), configuration files ({cfg_list_str}), LLM model catalog availability (config.env vs agy models), and persistent unresolved errors in logs: explain the technical root cause of any bot exceptions, fail-closed blocks, stale heartbeats, OS/network/VPN issues, or configuration errors.",
            "2. Determine whether hung/errored bots or modules require restart (RESTART_BOT / RESTART_MODULE), whether the entire fleet or VPN/DNS/clock requires repair (RESTART_FLEET / RESTART_VPN / PIN_DNS / SYNC_CLOCK), or whether a critical unrecoverable Linux OS / kernel / system-wide hang requires a full machine reboot (REBOOT_SYSTEM).",
            "3. If a reproducible code bug or configuration error exists in the workspace, formulate an exact search_block and replace_block in `code_remediation` to fix the code or configuration file. Note: when AUTO_CONFIG_UPDATES or MISSING_CONFIGURED_MODELS are reported for config.env, auto_technical_repair updates config.env automatically via verified code_remediation so the next fleet hot-reload adopts the new models.",
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
            f'  "summary": "1-2 sentence executive assessment of {scope_spec.summary_scope_lbl} market posture and trading decisions"'
        )
    xs_min_profit_pct = gw_th["xstock_sell_min_profit_pct"]
    xs_min_drop_pct = gw_th["xstock_sell_min_target_drop_pct"]
    xs_min_conf = gw_th["xstock_sell_min_confidence"]
    xs_cd_h = gw_th["xstock_sell_cooldown_hours"]
    xs_crash_drop_pct = gw_th["xstock_sell_crash_min_target_drop_pct"]
    xs_crash_conf = gw_th["xstock_sell_crash_min_confidence"]
    ev_fee_mult = gw_th["target_ev_fee_mult"]
    min_tgt_rr = gw_th["min_target_rr"]
    max_hr_rep = gw_th["max_hourly_replaces_per_symbol"]
    rep_min_delta_pct = gw_th["replace_min_price_delta_pct"]
    xs_target_lbl = "tokenized xStocks"
    if spec.can_trade:
        target_desc, xs_target_lbl = _build_dynamic_target_desc(is_equities)
        xs_conf_sfx = f" `conf>={xs_min_conf:g}`" if xs_min_conf > 0 else ""
        xs_crash_sfx = (
            f", OR imminent crash `urgency=HIGH` `conf>={xs_crash_conf:g}` `target_px` >={xs_crash_drop_pct:g}% lower"
            if xs_crash_drop_pct > 0
            else ""
        )
        xs_cd_sfx = f"; {xs_cd_h:g}h cooldown" if xs_cd_h > 0 else ""
        xs_exception_sfx = (
            ""
            if is_equities
            else (
                f" (EXCEPTION: SELL-only Kraken {xs_target_lbl} = long-term `HOLD`, heavy manual rebuy fee; "
                f"sell rarely on >+{xs_min_profit_pct:g}% profit{xs_conf_sfx} + `target_px` >={xs_min_drop_pct:g}% lower{xs_crash_sfx}{xs_cd_sfx}; emailed)"
            )
        )
        venue_schema = scope_spec.venue_schema
        venue_list_desc = scope_spec.venue_list_desc
        ex_sym = scope_spec.example_symbol
        half_max_usd = round(max_usd * 0.5)
        full_max_usd = round(max_usd)
        dip_flow_rule = (
            "   - CONTRARIAN DIP ACCUMULATION (SCALE-IN) & KINEMATICS: Weigh `kinematics(vel_90s, acc_90s, vel_5m, acc_5m, kalman_z)`, `kalman_vel`, "
            "`bottom_6h`, RTH CLV flow (`cvd_5m`/`cvd_15m`, `taker_ratio`), `spot_vol_15m`/`1h` & `rvol_15m`/`1h`, "
            "and `ob_proxy[yahoo_5m_vol_profile]` (`est_bid_zone`/`est_ask_zone` are synthetic 5m OHLCV proxies, NOT real L2 walls). "
            "Scale in on downside deceleration + absorption; de-risk when negative velocity/acceleration aligns with unabsorbed selling.\n"
            if is_equities
            else (
                "   - CONTRARIAN DIP ACCUMULATION (SCALE-IN), BREAKOUT MOMENTUM & KINEMATICS: Weigh `kinematics(vel_90s/5m, acc_90s/5m, kalman_z)`, `kalman_vel`, "
                "`bottom_6h`, `vs_btc`, `cvd_5m/15m`, `spot_cvd_15m` (`[real_spot]` vs `[est:ob+15m]`, never double-count with `ob_imbalance` `[0..1, 0.5=neutral]`), `rvol`, `taker_ratio`, L2 walls. "
                + _format_crypto_entry_playbook_clauses(half_max_usd, full_max_usd, compact=False)
            )
        )
        limit_exec_rule = (
            "   - SPOT 'LIMIT' EXECUTION (PASSIVE OR NEAR-SPOT): Submit `\"order_type\": \"LIMIT\"` (never MARKET) to control fill price and spread: "
            "passive `limit_price` near `vol_profile`/`session_vwap` for swings; explicit near-spot `limit_price` (`urgency:\"NORMAL\"`) for momentum; `limit_price: null` (`urgency:\"HIGH\"`) only for urgent defensive exits/reversals. "
            "Audit `open_limit_orders` (`Dist`, `Fill1h`), "
            "`last_cycle_execution_status` (`<=60m` fill/reject or `<=30m` resting), and calibration `arr_bps`/`slip_bps` (negative = favorable price improvement), `unfilled_off_bps`, & `fill_by_offset` fill-rate bands: "
            "re-anchor stale/far (`Fill1h<=5%` or `|Dist|>4%`) orders via `replace_open_orders: true` -- always re-evaluate R:R from the new `limit_price` (`\"CANCEL\"`+`cancel_order_ids` or `HOLD` instead of chasing into poor R:R); `'HOLD'` keeps resting orders.\n"
            if is_equities
            else (
                f"   - SPOT MAKER 'LIMIT' EXECUTION IN THE ORDERBOOK: Submit {_CRYPTO_POST_ONLY_LIMIT_EXEC_SPEC} to capture spread and cut fees: "
                "passive `limit_price` at bid/ask walls for swings; explicit BBO `limit_price` (`urgency:\"NORMAL\"`) for breakouts; `limit_price: null` (`urgency:\"HIGH\"`) only for urgent defensive exits/reversals. "
                "Audit `open_limit_orders` (`Dist`, `Fill1h`, `Note`; watch `[ob_stale]` when `ob_age > 30s`), `last_cycle_execution_status`, and `arr_bps`/`slip_bps`/`unfilled_off_bps`/`fill_by_offset`: "
                "re-anchor stale/far (`Fill1h<=5%` or `|Dist|>4%`) or wall-trapped (`front_run=` / `Note` wall flags) orders via `replace_open_orders: true` -- always re-evaluate R:R from the new `limit_price` (`\"CANCEL\"`+`cancel_order_ids` or `HOLD` instead of chasing into poor R:R); `'HOLD'` keeps resting orders.\n"
            )
        )
        structure_macro_rule = (
            "   - STRUCTURE & CROSS-ASSET MACRO: Combine `vol_profile` (`POC`/`VAL`/`VAH`/`LVN`, `REAL_VOL` vs `TPO`), "
            "`session_vwap`, venue fees/spread/`limits`, and Section 2 cross-asset telemetry "
            "(`SPY/QQQ/ES=F/NQ=F/VIX/DXY/US10Y`, `Futures_Lead`, `IBIT`, `CB_Premium`, `BTC_DVOL/ETH_DVOL`, `BTC_Options` & `ETH_Options`; note `delayed=` on Yahoo futures/VIX; judge risk-on/off vs decoupling from the data; `Skew_25d`=Put_IV-Call_IV, >0 fear, <0 call demand).\n"
            if is_equities
            else (
                "   - STRUCTURE, LIQUIDATIONS & CROSS-ASSET MACRO: Combine `vol_profile` (`POC`/`VAL`/`VAH`/`LVN`, `REAL_VOL` vs `TPO`), "
                "`liq_clusters` (`src=model_est`; prioritize `realized_liq_15m`), `book_window` cap tags, venue fees/spread/`limits`, and Section 2 cross-asset telemetry "
                "(`SPY/QQQ/ES=F/NQ=F/VIX/DXY/US10Y`, `Futures_Lead`, `IBIT`, `CB_Premium`, `BTC_DVOL/ETH_DVOL`, `BTC_Options` & `ETH_Options`; judge risk-on/off vs decoupling from the data; `Skew_25d`=Put_IV-Call_IV, >0 fear, <0 call demand).\n"
            )
        )
        ev_rr_parts: List[str] = []
        if ev_fee_mult > 0:
            ev_rr_parts.append(f"positive-EV `target_px` >= max(0.10%, {ev_fee_mult:g}x venue RT fee)")
        if min_tgt_rr > 0:
            ev_rr_parts.append(f"min R:R >= {min_tgt_rr:g}x when both `target_px` & `invalidation_px` set")
        ev_rr_clause = (
            f"{' and '.join(ev_rr_parts)} on all `BUY` & `urgency=NORMAL` `SELL`, "
            if ev_rr_parts
            else ""
        )
        tasks.append(
            f"{len(tasks) + 1}. Formulate symmetric autonomous trading decisions across {target_desc}:\n"
            "   - FULL READ ACCESS TO `cachedb/` & `statedb/` (`mptrade` workspace): Unrestricted read access to `cachedb/` (`cachedb/account/`), "
            "`statedb/orders/` (Order Ledger SSoT), `logs/`, `providers.market_api`, and on-demand multi-year chart zoom via `verify_tools/chart_view.py` beyond this dashboard.\n"
            "   - ZERO PROFIT OR STRATEGY GUARDS (AI SURVIVAL BREAKERS ONLY): Maximize risk-adjusted net NAV (after fees). Deterministic order_guard gates "
            "(G_PROF, GO1, GT_*, G_QLIMIT*, GI_*, GLLM*) are BYPASSED on BUY & SELL. Deterministic bots/configs run in parallel as optional advisory references only--ZERO obligation to follow them. "
            "Only AI survival breakers "
            "(flip cooldown unless `invalidation_px`/`target_px` or net edge triggers, 6h whipsaw cap, "
            + ev_rr_clause
            + (
                f"hourly replace churn cap ({max_hr_rep}/h if move < {rep_min_delta_pct:g}%), "
                if max_hr_rep > 0
                else ""
            )
            + "daily `by=AI` BUY cap, 48h `AI Decision Scorecard` BUY shadow cooldown) apply.\n"
            "   - SUPERVISORY AUTHORITY TO AUDIT & REPAIR / REVERSE UNWANTED BOT OR FLEET TRADES (`by=BOT`/`by=MANUAL` vs `by=AI`): "
            "Audit interim `by=BOT`/`by=MANUAL` fills/orders independently. Cancel unwanted AI/bot orders pre-fill via `\"decision\":\"CANCEL\"` + `cancel_order_ids:[...]` "
            "(or with `BUY`/`SELL` via `cancel_bot_open_orders:true` / `cancel_order_ids`); do NOT sell minutes after a bot fill unless structure genuinely breaks down "
            "(avoid burning round-trip fees on noise). When repair is warranted, submit "
            + (
                "a `LIMIT` (`\"order_type\": \"LIMIT\"`) at live spot:\n"
                "     1) If a bot SOLD prematurely or when it should NOT have, submit corrective `BUY` `LIMIT` at live spot.\n"
                "     2) If a bot BOUGHT prematurely or when it should NOT have (breakdown), submit corrective `SELL` `LIMIT` at live spot.\n"
                if is_equities
                else (
                    "spot `LIMIT` (`\"order_type\": \"LIMIT\"`) "
                    "right at live BBO (`best_bid` for BUY / `best_ask` for SELL):\n"
                    "     1) If a bot SOLD prematurely or when it should NOT have, submit corrective `BUY` `LIMIT` at `best_bid` (maker fee).\n"
                    "     2) If a bot BOUGHT prematurely or when it should NOT have (breakdown), submit corrective `SELL` `LIMIT` at `best_ask`.\n"
                )
            )
            + "     3) `cancel_order_ids:[...]` cancels only listed IDs "
            "(standalone `CANCEL` targets AI/bot orders; `BUY`/`SELL` without it cancels same-side only; failed cancel aborts replace).\n"
            f"   - UNBIASED QUANTITATIVE FREEDOM & DYNAMIC POSITION SIZING: Decide freely WHEN/WHAT/HOW MUCH/WHICH venue ({venue_list_desc}). "
            f"Cost basis (`avg_cost`, `lifo_avg`, `open_lifo_lots`) is informational, NEVER a constraint on 2-way {'equities' if is_equities else 'pairs'}: harvest profitable LIFO tranches or up to `50%\u2013100%` of `value_usd` "
            f"(up to {max_usd:g} USD), OR trim/exit via near-market `LIMIT` even at breakeven or below cost basis on breakdown{xs_exception_sfx}.\n"
            + dip_flow_rule
            + limit_exec_rule
            + structure_macro_rule
            + f"   - Emit one entry per actionable instrument in `market_trading_decisions` (`confidence >= {min_conf:g}`, `suggested_notional_usd` between {min_usd:g} and {max_usd:g} USD; "
            "gateway clamps to free balance, zero artificial cooldowns or dedup locks beyond AI breakers), or a single `'HOLD'` entry. "
            "Set numeric `invalidation_px`, `target_px`, and optional `horizon_h` on BUY/SELL; active plans persist in Section 1."
        )
        schema.append(
            f'  "market_trading_decisions": [{{"decision": "BUY" | "SELL" | "CANCEL" | "HOLD", "symbol": {ex_sym}, '
            f'"venue": {venue_schema}, "order_type": "LIMIT", "limit_price": null, '
            f'"invalidation_px": null, "target_px": null, "horizon_h": null, '
            f'"replace_open_orders": false, "cancel_bot_open_orders": false, "cancel_order_ids": [], '
            f'"confidence": 0.90, "suggested_notional_usd": 1000.0, "urgency": "NORMAL" | "HIGH", "thesis": "..."}}]'
        )

    unit_by_unit = scope_spec.unit_by_unit
    unit_tables = scope_spec.unit_tables
    ro_all_assets = scope_spec.ro_all_assets
    if compact_rules and not spec.can_self_heal:
        compact_conf_sfx = f" `conf>={xs_min_conf:g}`" if xs_min_conf > 0 else ""
        compact_crash_sfx = (
            f", OR imminent crash (`urgency=HIGH`, `conf>={xs_crash_conf:g}`, `target_px` >={xs_crash_drop_pct:g}% lower)"
            if xs_crash_drop_pct > 0
            else ""
        )
        compact_cd_sfx = f"; {xs_cd_h:g}h cooldown" if xs_cd_h > 0 else ""
        compact_xs_sfx = (
            ""
            if is_equities
            else (
                f"; remember SELL-only Kraken {xs_target_lbl} = long-term HOLD with heavy manual rebuy fee: "
                f"sell rarely on >+{xs_min_profit_pct:g}% profit{compact_conf_sfx} + `target_px` >={xs_min_drop_pct:g}% lower (1-2d+ drop){compact_crash_sfx}{compact_cd_sfx}"
            )
        )
        compact_exec_desc = (
            "spot LIMIT execution, active open-order management"
            if is_equities
            else f"{_CRYPTO_POST_ONLY_LIMIT_COMPACT_SPEC}, active orderbook management"
        )
        surv_bits: List[str] = []
        if ev_fee_mult > 0:
            surv_bits.append(f"EV>={ev_fee_mult:g}xRT")
        if min_tgt_rr > 0:
            surv_bits.append(f"RR>={min_tgt_rr:g}x")
        surv_bits.append("flip-cd")
        if max_hr_rep > 0:
            surv_bits.append(f"{max_hr_rep}/h-replace")
        surv_bits.extend(("daily-buy-cap", "scorecard"))
        compact_surv = ", ".join(surv_bits)
        compact_half_usd = round(max_usd * 0.5)
        compact_full_usd = round(max_usd)
        compact_entry_bullet = (
            ""
            if is_equities
            else _format_crypto_entry_playbook_clauses(compact_half_usd, compact_full_usd, compact=True)
        )
        compact_prefix = (
            "CONTINUATION TURN (ACTIVE THREAD POLICY APPLIES):\n"
            f"- Apply thread rules ({compact_exec_desc}, `by=BOT` vs `by=AI` supervision, `mptrade` read access to `cachedb/`, `statedb/`, `logs/` & `verify_tools/chart_view.py`; "
            f"zero deterministic order_guard gates; AI survival breakers active: {compact_surv}; `conf>={min_conf:g}`, notional {min_usd:g}-{max_usd:g} USD{compact_xs_sfx}). "
            "All AI orders submit as `LIMIT`; re-evaluate R:R from new `limit_price` on replace (use `CANCEL`+`cancel_order_ids` or `HOLD` instead of chasing into poor R:R).\n"
            + compact_entry_bullet
            + "- Honor user chat instructions/overrides strictly for their stated instrument, action, and timeframe.\n"
            "- Format your response in TWO parts:\n"
            "  1) Concise Markdown in **Romanian** (`### \u23f1\ufe0f Raport Execu\u021bie [YYYY-MM-DD HH:MM:SS EET]`): "
            f"on routine pure `HOLD` cycles (no BUY/SELL/cancel, no Section 5 priority symbols, no user chat), output ONLY "
            f"2-3 ultra-compact bullets (NAV + free cash, 1-line market pulse, HOLD rationale; omit {unit_by_unit} tables); "
            f"otherwise provide the concise {unit_by_unit} Romanian breakdown.\n"
            "  2) At the very end, a single fenced ```json ... ``` block matching this exact schema:\n"
            "```json\n{\n" + ",\n".join(schema) + "\n}\n```\n\n"
            "LIVE TELEMETRY SNAPSHOT:\n"
        )
        return compact_prefix + context_str

    role_intro, tool_rule, report_format_rule = (
        (
            "You are the Lead SRE and Autonomous Technical Repair Supervisor for MPTrade.\n"
            "Your STRICT mandate is solely to detect functional system anomalies (bots, fleet modules, heartbeats), "
            "Linux OS resource issues, internet & VPN (wgpia0) connectivity failures, configuration errors, and "
            "persistent unresolved errors in logs, and to remediate them by patching code or configurations and "
            "restarting/rebooting bots, modules, the trading fleet, VPN/DNS, or the entire Linux system.\n"
            "Do NOT analyze or report on market prices, trends/slopes, macro sentiment, portfolio balances, or active trading positions.\n\n"
            "Perform an executive technical diagnostic and self-healing audit based on the live telemetry snapshot below:\n"
            + "\n".join(tasks)
            + "\n\n",
            "- If you diagnose a reproducible code bug or configuration error and need to verify exact lines before proposing `code_remediation`, "
            "you MAY read the target file in the workspace so `search_block` matches verbatim. "
            "Do NOT edit or modify workspace files directly with tools; propose all fixes exclusively in `code_remediation`.\n",
            "- Format your response in TWO parts so this thread stays cleanly readable in the Web UI:\n"
            "  1) A concise, well-structured Markdown report in **Romanian** strictly covering technical & SRE health "
            "with the cycle timestamp in the main header (e.g. `### \U0001f6e0\ufe0f Raport Audit Sistem [YYYY-MM-DD HH:MM:SS EET]`) "
            "(bullet points for: Starea Flotei, Bo\u021bilor, Sistemului Linux & VPN (wgpia0) & Loguri, "
            "Conectivitate Internet & Audit Configura\u021bii, Erori Persistente Nerezolvate, \u0219i "
            "Ac\u021biuni de Remediere Cod/Config & Reboot Bo\u021bi/Module/Flot\u0103/Sistem). "
            "NU analiza \u0219i NU men\u021biona pre\u021buri de pia\u021b\u0103, pante, portofoliu, solduri sau pozi\u021bii active!\n",
        )
        if spec.can_self_heal
        else (
            "You are the Quantitative Autonomous Portfolio Manager and Trade Executor for MPTrade.\n"
            "Evaluate the telemetry snapshot below and formulate autonomous BUY/SELL trading decisions:\n"
            + "\n".join(tasks)
            + "\n\n",
            "- Keep this thread focused on market analysis, portfolio management, and autonomous BUY/SELL execution across venues.\n",
            "- Format your response in TWO parts:\n"
            "  1) A concise Markdown report in **Romanian** with cycle timestamp header (`### \u23f1\ufe0f Raport Execu\u021bie [YYYY-MM-DD HH:MM:SS EET]`):\n"
            "     - On routine pure `HOLD` cycles (no BUY/SELL/cancel action, no Section 5 priority symbols, no user chat message), "
            f"keep the report ULTRA-COMPACT (2-3 short bullet lines max: NAV + free cash, 1-line market pulse, HOLD rationale; omit {unit_tables}).\n"
            "     - On BUY/SELL/cancel, Section 5 priority symbols, or user chat replies, include concise bullet points for: "
            f"Pia\u021ba Live & Pante ({ro_all_assets}), Portofoliu, Solduri & Pozi\u021bii Active, \u0219i Decizia AI de Tranzac\u021bionare / R\u0103spuns la interven\u021biile din chat.\n",
        )
    )

    static_prefix = (
        role_intro
        + "INTERACTIVE THREAD RULES:\n"
        "- Honor any earlier user instructions, targets, overrides, or questions in this thread with top priority.\n"
        + tool_rule
        + report_format_rule
        + "  2) At the very end, a single fenced ```json ... ``` block matching this exact schema:\n"
        "```json\n{\n" + ",\n".join(schema) + "\n}\n```\n\n"
        "LIVE TELEMETRY SNAPSHOT:\n"
    )
    return static_prefix + context_str

