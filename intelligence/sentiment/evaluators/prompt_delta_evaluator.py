"""Pure deterministic Warm-Turn Delta & Keyframe evaluator for AI Supervisor prompts.

Computes content-only fingerprints for slow/medium/event-only telemetry blocks and decides per turn
whether each block should render in full (``FULL_KEYFRAME``), as newly changed (``SHOW_NEW``),
as a periodic cadence refresh (``SHOW_REFRESH``), or be omitted with an explicit transparency marker
(``OMIT``) while still active in the warm uncompacted conversation thread.

Strictly zero disk or network I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import math
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

from botcore import (
    _parse_bool_flag,
    clean_scalar_str,
    normalize_epoch_sec,
    resolve_now_ts,
    safe_float,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
)
from cachepaths import clean_symbol_key
from instrument_registry import observe_only_registry_symbols
from intelligence.macro.calendar_ssot import resolve_macro_economic_calendar
from intelligence.macro_sentiment_ssot import format_age_sec_compact
from intelligence.sentiment.evaluators.macro_telemetry_evaluator import (
    format_etf_flows_summary_line,
)
from intelligence.sentiment.evaluators.price_series_evaluator import TelemetryEvaluatorFactory
from intelligence.sentiment.evaluators.prompt_view_evaluator import (
    PromptViewEvaluatorBase,
    evaluate_portfolio_prompt_view,
)
from providers.ai_survival_guard import (
    format_ai_calibration_feedback_line,
    format_ai_decision_scorecard_line,
)

_BUCHAREST_TZ = ZoneInfo("Europe/Bucharest")
_XSTOCK_NEWS_RE = re.compile(
    r"\b(?:spacex|starlink|spcx|xstock|xstocks|anthropic|antx)\b",
    re.IGNORECASE,
)
_MACRO_AGO_RE = re.compile(r"\b\d+m\s+ago\b", re.IGNORECASE)
_MACRO_IN_RE = re.compile(r"\bin\s+(?:\d+d\s*)?(?:\d+h\s*)?\d+m\b|\bin\s+\d+h\b", re.IGNORECASE)
_XSTOCK_VOLATILE_KEYS: frozenset[str] = frozenset({
    "price", "px", "spot_px", "_spot_px", "live_px", "live_price", "underlying_live_price", "value_usd",
    "chg_1m_pct", "chg_5m_pct", "chg_15m_pct", "chg_1h_pct", "chg_24h_pct", "chg_7d_pct",
    "pnl_pct", "pnl_vs_avg_pct", "pnl_vs_alloc_pct", "pnl_vs_cost_pct", "pnl_vs_lifo_pct",
    "dd_pct", "dd_from_peak_pct", "dist_pct", "xstock_vs_underlying_pct",
    "age_min", "age_s", "ts", "timestamp",
})

PROMPT_DELTA_GROUP_CADENCES: Dict[str, int] = {
    "G1": 3,
    "G2": 5,
    "G3": 0,
}


@dataclass(frozen=True)
class PromptDeltaBlockSpec:
    """Declarative specification for one warm-turn delta-managed telemetry block."""

    key: str
    label: str
    group: str
    refresh_turns: int
    category: str


PROMPT_DELTA_BLOCK_SPECS: Tuple[PromptDeltaBlockSpec, ...] = (
    # G1: Medium-cadence context (refresh every 3 unchanged warm turns, or immediately on change)
    PromptDeltaBlockSpec("macro_calendar", "Calendar", "G1", PROMPT_DELTA_GROUP_CADENCES["G1"], "macro"),
    PromptDeltaBlockSpec("equity_bellwethers", "US_Equities", "G1", PROMPT_DELTA_GROUP_CADENCES["G1"], "equities"),
    PromptDeltaBlockSpec("scorecard", "Scorecard", "G1", PROMPT_DELTA_GROUP_CADENCES["G1"], "execution"),
    PromptDeltaBlockSpec("calibration", "Calibration", "G1", PROMPT_DELTA_GROUP_CADENCES["G1"], "execution"),
    PromptDeltaBlockSpec("venue_cycles", "Cycle_Anchors", "G1", PROMPT_DELTA_GROUP_CADENCES["G1"], "execution"),
    PromptDeltaBlockSpec("recent_orders", "Recent_Fills", "G1", PROMPT_DELTA_GROUP_CADENCES["G1"], "execution"),
    # G2: Slow / near-static context (refresh every 5 unchanged warm turns, or immediately on change)
    PromptDeltaBlockSpec("venue_specs", "Venue_Specs", "G2", PROMPT_DELTA_GROUP_CADENCES["G2"], "execution"),
    PromptDeltaBlockSpec("fear_greed", "Fear&Greed", "G2", PROMPT_DELTA_GROUP_CADENCES["G2"], "macro"),
    PromptDeltaBlockSpec("etf_flows", "ETF_Flows", "G2", PROMPT_DELTA_GROUP_CADENCES["G2"], "macro"),
    PromptDeltaBlockSpec("long_horizon", "Long_Horizon", "G2", PROMPT_DELTA_GROUP_CADENCES["G2"], "macro"),
    PromptDeltaBlockSpec("kraken_xstocks", "xStocks", "G2", PROMPT_DELTA_GROUP_CADENCES["G2"], "xstocks"),
    PromptDeltaBlockSpec("hyperliquid_readonly", "HL_Summary", "G2", PROMPT_DELTA_GROUP_CADENCES["G2"], "execution"),
    # G3: Event-only anti-bias blocks (never repeated unchanged on warm turns; refreshed on FULL_KEYFRAME)
    PromptDeltaBlockSpec("operator_regime_note", "Operator_Note", "G3", PROMPT_DELTA_GROUP_CADENCES["G3"], "macro"),
    PromptDeltaBlockSpec("sentiment_advisor_summary", "Advisor_Summary", "G3", PROMPT_DELTA_GROUP_CADENCES["G3"], "macro"),
    PromptDeltaBlockSpec("geopolitical_summary", "Geo_Summary", "G3", PROMPT_DELTA_GROUP_CADENCES["G3"], "macro"),
    PromptDeltaBlockSpec("breaking_news", "News", "G3", PROMPT_DELTA_GROUP_CADENCES["G3"], "news"),
)
PROMPT_DELTA_SPECS_BY_KEY: Dict[str, PromptDeltaBlockSpec] = {s.key: s for s in PROMPT_DELTA_BLOCK_SPECS}


@dataclass
class PromptDeltaPlan:
    """Evaluated per-turn rendering plan and next persisted fingerprint state."""

    mode: str = "FULL_KEYFRAME"
    reason: str = "cold_turn"
    block_actions: Dict[str, str] = field(default_factory=dict)
    recent_order_actions: Dict[str, str] = field(default_factory=dict)
    new_news_titles: Optional[set[str]] = None
    omitted_news_count: int = 0
    omitted_summary_tokens: List[str] = field(default_factory=list)
    next_state: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_warm_delta(self) -> bool:
        return self.mode == "WARM_DELTA"

    def action_for(self, key: str) -> str:
        return self.block_actions.get(key, "SHOW_FULL")

    def should_show(self, key: str) -> bool:
        return self.action_for(key) != "OMIT"

    def tag_for(self, key: str) -> str:
        act = self.action_for(key)
        if act == "SHOW_NEW":
            return " [NEW]"
        if act == "SHOW_REFRESH":
            return " [REFRESH]"
        return ""

    def should_show_recent_order(self, sym: str) -> bool:
        return self.recent_order_actions.get(sym, "SHOW_FULL") != "OMIT"

    def recent_order_tag(self, sym: str) -> str:
        act = self.recent_order_actions.get(sym, "SHOW_FULL")
        if act == "SHOW_NEW":
            return " [NEW]"
        if act == "SHOW_REFRESH":
            return " [REFRESH]"
        return ""

    def format_section0_lines(self) -> List[str]:
        """Format transparent snapshot-mode and omitted-block lines for Section 0."""
        if self.mode == "FULL_KEYFRAME":
            return [f"- Snapshot Mode: FULL_KEYFRAME (reason={self.reason})"]
        lines = [
            f"- Snapshot Mode: WARM_DELTA (cadence: G1={PROMPT_DELTA_GROUP_CADENCES['G1']}t, "
            f"G2={PROMPT_DELTA_GROUP_CADENCES['G2']}t, G3=event_only; "
            "tags: [NEW]=changed, [REFRESH]=cadence_keyframe, [src_age=]=feed_not_refreshed_else_verified)"
        ]
        if self.omitted_summary_tokens:
            lines.append(
                f"- Unchanged in thread (active from prior turn): {', '.join(self.omitted_summary_tokens)}"
            )
        return lines


def _canonical_fingerprint_str(obj: Any) -> str:
    """Serialize Python primitives and containers deterministically without disk/state I/O modules."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return repr(obj)
    if isinstance(obj, Mapping):
        items = ",".join(
            f"{repr(str(k))}:{_canonical_fingerprint_str(v)}"
            for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))
        )
        return "{" + items + "}"
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_canonical_fingerprint_str(v) for v in obj) + "]"
    if isinstance(obj, (set, frozenset)):
        return "{" + ",".join(sorted(_canonical_fingerprint_str(v) for v in obj)) + "}"
    return repr(str(obj))


def _hash_payload(payload: Any) -> str:
    raw = _canonical_fingerprint_str(payload)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _strip_volatile_keys(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {
            str(k): _strip_volatile_keys(v)
            for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))
            if str(k) not in _XSTOCK_VOLATILE_KEYS
        }
    if isinstance(obj, list):
        return [_strip_volatile_keys(v) for v in obj]
    return obj


def _normalize_macro_event_fingerprint(val: Any) -> Any:
    """Strip minute-ticking countdown/elapsed suffixes while preserving event identity and active-window state."""
    if not isinstance(val, str):
        return val
    return _MACRO_IN_RE.sub("<in>", _MACRO_AGO_RE.sub("<ago>", val))


def _resolve_macro_map(sensors: Any, attr: str, *, fallback_key: Optional[str] = None) -> Dict[str, Any]:
    macro = getattr(sensors, "macro_and_sentiment", None) or {}
    ps = getattr(sensors, "portfolio_summary", None) or {}
    val = (
        getattr(sensors, attr, None)
        or (macro.get(fallback_key or attr) if isinstance(macro, Mapping) else None)
        or (ps.get(attr) if isinstance(ps, Mapping) else None)
    )
    return dict(val) if isinstance(val, Mapping) else {}


def _resolve_block_age_sec(
    domain_ages: Mapping[str, Any],
    domain_key: str,
    payload: Mapping[str, Any],
    now_ts: float,
) -> Optional[float]:
    """Return upstream source age in seconds for a delta-managed block when deterministically known."""
    if (d_age := safe_nonneg_float(domain_ages.get(domain_key), default=None)) is not None:
        return d_age
    raw_ts = payload.get("ts") if payload.get("ts") is not None else payload.get("timestamp")
    ts_s = normalize_epoch_sec(raw_ts, default=0.0, allow_iso=True) or 0.0
    if ts_s > 0.0 and now_ts >= ts_s:
        return now_ts - ts_s
    return None


def _extract_block_fingerprints(
    sensors: Any,
    *,
    asset_scope: str,
    now_ts: float,
    min_usd: float,
    max_usd: float,
) -> Tuple[Dict[str, str], Dict[str, str], List[str], bool, Dict[str, float]]:
    """Compute content-only fingerprints and source ages for every delta-managed block on ``sensors``.

    Returns ``(block_hashes, recent_order_hashes, news_titles, macro_pre_event_active, block_src_ages_sec)``.
    """
    is_equities = asset_scope in ("equities", "equity")
    macro = getattr(sensors, "macro_and_sentiment", None) or {}
    ps = getattr(sensors, "portfolio_summary", None) or {}
    raw_dom_ages = getattr(sensors, "source_ages_sec", None)
    dom_ages: Mapping[str, Any] = raw_dom_ages if isinstance(raw_dom_ages, Mapping) else {}
    breaking_news = getattr(sensors, "breaking_news", None) or (macro.get("recent_news") if isinstance(macro, Mapping) else None) or []

    block_hashes: Dict[str, str] = {}
    block_src_ages: Dict[str, float] = {}
    if not is_equities:
        fg = _resolve_macro_map(sensors, "fear_greed")
        if fg:
            fg_hist = fg.get("historical_values")[:7] if isinstance(fg.get("historical_values"), list) else None
            block_hashes["fear_greed"] = _hash_payload({
                "value": fg.get("value"),
                "cls": fg.get("classification") or fg.get("sentiment"),
                "chg": fg.get("trend_7d_change"),
                "hist": fg_hist,
            })
            if (age_v := _resolve_block_age_sec(dom_ages, "fear_greed", fg, now_ts)) is not None:
                block_src_ages["fear_greed"] = age_v
        etf = _resolve_macro_map(sensors, "etf_flows")
        if etf:
            etf_s = format_etf_flows_summary_line(etf) or str(etf.get("summary_line") or "").strip()
            if etf_s:
                block_hashes["etf_flows"] = _hash_payload(etf_s)
                if (age_v := _resolve_block_age_sec(dom_ages, "etf_flows", etf, now_ts)) is not None:
                    block_src_ages["etf_flows"] = age_v
        sa = _resolve_macro_map(sensors, "sentiment_advisor")
        if sa and str(sa.get("summary") or "").strip():
            block_hashes["sentiment_advisor_summary"] = _hash_payload({
                "ts": sa.get("ts") if sa.get("ts") is not None else sa.get("timestamp"),
                "conf": sa.get("confidence"),
                "summary": str(sa.get("summary") or "").strip(),
            })
            if (age_v := _resolve_block_age_sec(dom_ages, "sentiment_advisor", sa, now_ts)) is not None:
                block_src_ages["sentiment_advisor_summary"] = age_v

    gt = _resolve_macro_map(sensors, "geopolitical_threat", fallback_key="geopolitical")
    if gt and str(gt.get("summary") or "").strip():
        block_hashes["geopolitical_summary"] = _hash_payload({
            "ts": gt.get("ts") if gt.get("ts") is not None else gt.get("timestamp"),
            "risk": gt.get("risk_score"),
            "summary": str(gt.get("summary") or "").strip(),
            "deesc": gt.get("live_deescalation_note"),
            "unass": gt.get("live_unassessed_high_sev_note"),
        })
        if (age_v := _resolve_block_age_sec(dom_ages, "geopolitical_threat", gt, now_ts)) is not None:
            block_src_ages["geopolitical_summary"] = age_v

    lh = _resolve_macro_map(sensors, "long_horizon")
    if lh and isinstance(lh.get("items"), Mapping) and lh["items"]:
        block_hashes["long_horizon"] = _hash_payload({
            "ts": lh.get("ts"),
            "months": lh.get("months"),
            "syms": sorted(str(k) for k in lh["items"].keys()),
        })
        if (age_v := _resolve_block_age_sec(dom_ages, "long_horizon", lh, now_ts)) is not None:
            block_src_ages["long_horizon"] = age_v

    op_note = _resolve_macro_map(sensors, "operator_regime_note")
    if op_note and str(op_note.get("text") or "").strip():
        block_hashes["operator_regime_note"] = _hash_payload({
            "text": str(op_note.get("text") or "").strip(),
            "exp": op_note.get("expires_ts"),
        })

    mc = _resolve_macro_map(sensors, "macro_calendar") or resolve_macro_economic_calendar(
        now_ts=now_ts,
        breaking_news=breaking_news,
    )
    macro_pre_event = bool(isinstance(mc, Mapping) and _parse_bool_flag(mc.get("pre_event_risk_window"), False))
    if isinstance(mc, Mapping) and mc.get("next_high_impact_event"):
        block_hashes["macro_calendar"] = _hash_payload({
            "session": mc.get("session_label"),
            "next_ev": _normalize_macro_event_fingerprint(mc.get("next_high_impact_event")),
            "holiday": mc.get("us_market_holiday"),
            "early": mc.get("us_market_early_close"),
            "pre_risk": macro_pre_event,
            "pre_rsn": _normalize_macro_event_fingerprint(mc.get("pre_event_risk_reason")),
            "confirmed": mc.get("confirmed_events_today"),
            "earnings": mc.get("earnings_today"),
            "prints": mc.get("live_macro_prints"),
            "headlines": mc.get("detected_macro_headlines"),
        })

    news_titles: List[str] = []
    if isinstance(breaking_news, Sequence) and not isinstance(breaking_news, (str, bytes)):
        for hn in breaking_news:
            if isinstance(hn, Mapping) and (t_s := clean_scalar_str(hn.get("title"))):
                news_titles.append(t_s)
    if news_titles:
        block_hashes["breaking_news"] = _hash_payload(sorted(set(news_titles)))

    recent_order_hashes: Dict[str, str] = {}
    if isinstance(ps, Mapping) and ps:
        pview = evaluate_portfolio_prompt_view(dict(ps), asset_scope=asset_scope, max_usd=max_usd, min_usd=min_usd)
        if t_pos := pview.get("tradable_positions"):
            spec_rows = [
                (
                    pv.get("venue"),
                    pv.get("disp_symbol"),
                    pv.get("fee_tier"),
                    pv.get("maker_bps"),
                    pv.get("taker_bps"),
                    pv.get("min_notional_usd"),
                    tuple(
                        (pfx, val)
                        for pfx, val in (pv.get("limit_items") or ())
                        if pfx in ("min_qty=", "step=", "tick=")
                    ),
                )
                for pv in t_pos
                if isinstance(pv, Mapping)
            ]
            if spec_rows:
                block_hashes["venue_specs"] = _hash_payload(
                    sorted(spec_rows, key=lambda r: (str(r[0] or ""), str(r[1] or "")))
                )
        if sc_line := format_ai_decision_scorecard_line(pview.get("ai_decision_scorecard")):
            block_hashes["scorecard"] = _hash_payload(sc_line)
        if cal_line := format_ai_calibration_feedback_line(pview.get("ai_calibration_feedback")):
            block_hashes["calibration"] = _hash_payload(cal_line)
        if vc_rows := pview.get("venue_cycles"):
            vc_fp = [
                (
                    v_id,
                    [
                        (c_sym, _strip_volatile_keys(c_data))
                        for c_sym, c_data in cyc_items
                        if isinstance(c_data, Mapping)
                    ],
                )
                for _, v_id, cyc_items in vc_rows
            ]
            if vc_fp:
                block_hashes["venue_cycles"] = _hash_payload(vc_fp)
        if xs_view := pview.get("kraken_xstocks"):
            block_hashes["kraken_xstocks"] = _hash_payload(_strip_volatile_keys(xs_view))
        if eq_bw := pview.get("equity_bellwethers"):
            if isinstance(eq_bw, Mapping):
                quant_eq: Dict[str, Any] = {}
                max_eq_age_m: Optional[float] = None
                for sk, sv in sorted(eq_bw.items()):
                    if isinstance(sv, Mapping):
                        px_v = safe_pos_float(sv.get("price"), default=None)
                        px_bkt = round(math.log(px_v) / 0.003) if (px_v is not None and px_v > 0.0) else None
                        quant_eq[str(sk)] = {
                            "px_bkt": px_bkt,
                            "src": sv.get("quote_source"),
                            "st": sv.get("market_state"),
                        }
                        if (q_age_m := safe_pos_float(sv.get("quote_age_min"), default=None)) is not None:
                            if max_eq_age_m is None or q_age_m > max_eq_age_m:
                                max_eq_age_m = q_age_m
                    else:
                        quant_eq[str(sk)] = sv
                block_hashes["equity_bellwethers"] = _hash_payload(quant_eq)
                if max_eq_age_m is not None:
                    block_src_ages["equity_bellwethers"] = max_eq_age_m * 60.0
            else:
                block_hashes["equity_bellwethers"] = _hash_payload(eq_bw)
        if isinstance(hl_ro := ps.get("hyperliquid_readonly"), Mapping) and hl_ro:
            hl_cash_raw = (
                hl_ro.get("free_cash_usd")
                if hl_ro.get("free_cash_usd") is not None
                else hl_ro.get("cash_usd")
            )
            block_hashes["hyperliquid_readonly"] = _hash_payload({
                "engine": hl_ro.get("engine"),
                "paper": hl_ro.get("paper"),
                "qty": round(safe_float(hl_ro.get("qty"), 0.0) or 0.0, 6),
                "cash": round(safe_float(hl_cash_raw, 0.0) or 0.0, 1),
                "locked": round(safe_float(hl_ro.get("locked_quote_usd"), 0.0) or 0.0, 1),
            })
        raw_ro = ps.get("recent_orders")
        if isinstance(raw_ro, Mapping):
            obs_only_syms = observe_only_registry_symbols()
            active_pos_syms = {
                k
                for p in (pview.get("tradable_positions") or ())
                if isinstance(p, Mapping)
                for p_sym in (p.get("disp_symbol") or p.get("symbol"),)
                for k in (
                    str(p_sym or "").strip().upper(),
                    clean_symbol_key(p_sym) or "",
                )
                if k
            }
            for _, _, cyc_items in (pview.get("venue_cycles") or ()):
                for c_sym, _ in cyc_items:
                    c_u = str(c_sym or "").strip().upper()
                    c_k = clean_symbol_key(c_sym) or ""
                    if c_u:
                        active_pos_syms.add(c_u)
                    if c_k:
                        active_pos_syms.add(c_k)
            for r_sym, oinfo in sorted(raw_ro.items()):
                if not isinstance(oinfo, Mapping):
                    continue
                s_u = str(r_sym or "").strip().upper()
                s_k = clean_symbol_key(r_sym) or ""
                if (
                    obs_only_syms
                    and (s_u in obs_only_syms or (s_k and s_k in obs_only_syms))
                    and s_u not in active_pos_syms
                    and (not s_k or s_k not in active_pos_syms)
                ):
                    continue
                lb = oinfo.get("last_buy") if isinstance(oinfo.get("last_buy"), Mapping) else {}
                ls = oinfo.get("last_sell") if isinstance(oinfo.get("last_sell"), Mapping) else {}
                recent_order_hashes[str(r_sym)] = _hash_payload({
                    "side": oinfo.get("last_action_side"),
                    "lb_px": lb.get("price"),
                    "lb_ts": lb.get("ts_ms"),
                    "lb_qty": lb.get("qty"),
                    "ls_px": ls.get("price"),
                    "ls_ts": ls.get("ts_ms"),
                    "ls_qty": ls.get("qty"),
                    "lots_n": len(oinfo.get("open_lots")) if isinstance(oinfo.get("open_lots"), list) else 0,
                    "rpnl": oinfo.get("lifo_realized_pnl_usd"),
                })
        if recent_order_hashes:
            block_hashes["recent_orders"] = _hash_payload(recent_order_hashes)

    return block_hashes, recent_order_hashes, news_titles, macro_pre_event, block_src_ages


class PromptDeltaKeyframeEvaluator(PromptViewEvaluatorBase):
    """Concrete evaluator (Level 3) planning warm-turn delta elision and keyframe refreshes."""

    cls_name = "PromptDeltaKeyframeEvaluator"

    def evaluate(
        self,
        sensors: Any,
        *,
        asset_scope: str = "all",
        now_ts: Optional[float] = None,
        min_usd: float = 10.0,
        max_usd: float = 2000.0,
        is_warm_resume: bool = False,
        prev_state: Optional[Mapping[str, Any]] = None,
        conversation_id: str = "",
        compacted_at_step: int = 0,
        escalate_high_effort: bool = False,
        urgent_escalation: bool = False,
        extreme_escalation: bool = False,
        ui_turn_pending: bool = False,
        focus_symbols: Optional[Sequence[str]] = None,
        prev_eval_was_actionable: bool = False,
        cadence_overrides: Optional[Mapping[str, int]] = None,
        **kwargs: Any,
    ) -> PromptDeltaPlan:
        eff_now = safe_pos_float(now_ts, default=None) or safe_pos_float(getattr(sensors, "evaluated_at_ts", None), default=None) or resolve_now_ts(None)
        date_iso = datetime.fromtimestamp(eff_now, tz=_BUCHAREST_TZ).strftime("%Y-%m-%d")
        scope_norm = str(asset_scope or "all").strip().lower()
        cid_clean = clean_scalar_str(conversation_id) or ""
        comp_step = safe_nonneg_int(compacted_at_step, default=0)

        block_hashes, ro_hashes, news_titles, macro_pre_event, block_src_ages = _extract_block_fingerprints(
            sensors,
            asset_scope=scope_norm,
            now_ts=eff_now,
            min_usd=min_usd,
            max_usd=max_usd,
        )

        prev_map = prev_state if isinstance(prev_state, Mapping) else {}
        prev_cid = clean_scalar_str(prev_map.get("conversation_id")) or ""
        prev_comp = safe_nonneg_int(prev_map.get("compacted_at_step"), default=-1)
        prev_date = clean_scalar_str(prev_map.get("date_iso")) or ""
        prev_blocks = prev_map.get("blocks") if isinstance(prev_map.get("blocks"), Mapping) else {}
        prev_ro = prev_map.get("recent_orders") if isinstance(prev_map.get("recent_orders"), Mapping) else {}
        prev_news_raw = prev_map.get("news_titles") if isinstance(prev_map.get("news_titles"), list) else []
        prev_news_set = {str(t) for t in prev_news_raw if isinstance(t, str) and t.strip()}

        # Determine whether a global FULL_KEYFRAME is mandatory
        full_reason = ""
        if not is_warm_resume:
            if prev_blocks and cid_clean and prev_cid and cid_clean == prev_cid and prev_comp >= 0 and comp_step > prev_comp:
                full_reason = "thread_compacted"
            else:
                full_reason = "cold_turn"
        elif not prev_blocks:
            full_reason = "no_prior_delta_state"
        elif cid_clean and prev_cid and cid_clean != prev_cid:
            full_reason = "thread_rotated"
        elif prev_comp >= 0 and comp_step != prev_comp:
            full_reason = "thread_compacted"
        elif prev_date and prev_date != date_iso:
            full_reason = "day_rollover"
        elif extreme_escalation:
            full_reason = "extreme_escalation"
        elif urgent_escalation:
            full_reason = "urgent_escalation"
        elif escalate_high_effort:
            full_reason = "high_effort_escalation"
        elif ui_turn_pending:
            full_reason = "ui_turn"
        elif focus_symbols:
            full_reason = "focus_symbols"
        elif prev_eval_was_actionable:
            full_reason = "post_action"

        def _make_full_state() -> Dict[str, Any]:
            return {
                "schema_version": 1,
                "conversation_id": cid_clean,
                "compacted_at_step": comp_step,
                "date_iso": date_iso,
                "ts": round(eff_now, 3),
                "blocks": {k: {"hash": h, "turns_omitted": 0} for k, h in block_hashes.items()},
                "recent_orders": {s: {"hash": h, "turns_omitted": 0} for s, h in ro_hashes.items()},
                "news_titles": list(dict.fromkeys(news_titles)),
            }

        if full_reason:
            return PromptDeltaPlan(
                mode="FULL_KEYFRAME",
                reason=full_reason,
                block_actions={k: "SHOW_FULL" for k in block_hashes},
                recent_order_actions={s: "SHOW_FULL" for s in ro_hashes},
                new_news_titles=None,
                omitted_news_count=0,
                omitted_summary_tokens=[],
                next_state=_make_full_state(),
            )

        # Warm delta evaluation with per-group cadence counters & category-level invalidations
        eff_cadences = dict(PROMPT_DELTA_GROUP_CADENCES)
        if isinstance(cadence_overrides, Mapping):
            for gk, gv in cadence_overrides.items():
                eff_cadences[str(gk)] = safe_nonneg_int(gv, default=eff_cadences.get(str(gk), 0))

        # 1. Detect newly changed blocks and category-level couplings
        changed_blocks: set[str] = set()
        for b_key, b_hash in block_hashes.items():
            prev_entry = prev_blocks.get(b_key) if isinstance(prev_blocks.get(b_key), Mapping) else {}
            if prev_entry.get("hash") != b_hash:
                changed_blocks.add(b_key)

        changed_ro_syms: set[str] = set()
        for r_sym, r_hash in ro_hashes.items():
            prev_r = prev_ro.get(r_sym) if isinstance(prev_ro.get(r_sym), Mapping) else {}
            if prev_r.get("hash") != r_hash:
                changed_ro_syms.add(r_sym)

        new_titles = [t for t in news_titles if t not in prev_news_set]
        new_titles_set = set(new_titles)
        omitted_news_n = len(news_titles) - len(new_titles)

        forced_categories: set[str] = set()
        if macro_pre_event or "macro_calendar" in changed_blocks:
            forced_categories.add("macro")
        if changed_ro_syms:
            forced_categories.add("execution")
        if any(_XSTOCK_NEWS_RE.search(t) for t in new_titles):
            forced_categories.add("xstocks")

        gt_map = _resolve_macro_map(sensors, "geopolitical_threat", fallback_key="geopolitical")
        force_geo_on_news = bool(
            new_titles
            and (
                gt_map.get("live_unassessed_high_sev_note")
                or (safe_float(gt_map.get("risk_score"), 0.0) or 0.0) >= 0.6
            )
        )

        block_actions: Dict[str, str] = {}
        next_blocks_state: Dict[str, Dict[str, Any]] = {}
        omitted_tokens: List[str] = []

        for spec in PROMPT_DELTA_BLOCK_SPECS:
            b_key = spec.key
            if b_key not in block_hashes:
                if b_key == "operator_regime_note" and b_key in prev_blocks:
                    omitted_tokens.append(f"{spec.label}(EXPIRED_CLEARED)")
                continue
            b_hash = block_hashes[b_key]
            k_turns = eff_cadences.get(spec.group, spec.refresh_turns)
            prev_entry = prev_blocks.get(b_key) if isinstance(prev_blocks.get(b_key), Mapping) else {}
            prev_omitted = safe_nonneg_int(prev_entry.get("turns_omitted"), default=0)

            if b_key == "breaking_news":
                if new_titles:
                    block_actions[b_key] = "SHOW_NEW"
                    next_blocks_state[b_key] = {"hash": b_hash, "turns_omitted": 0}
                else:
                    block_actions[b_key] = "OMIT"
                    next_blocks_state[b_key] = {"hash": b_hash, "turns_omitted": prev_omitted + 1}
                    if omitted_news_n > 0:
                        omitted_tokens.append(f"{spec.label}({omitted_news_n} prior)")
                continue

            if b_key == "recent_orders":
                # Handled per-symbol below; record aggregate status
                continue

            is_changed = b_key in changed_blocks or (spec.category in forced_categories) or (
                b_key == "geopolitical_summary" and force_geo_on_news
            )
            if is_changed:
                block_actions[b_key] = "SHOW_NEW"
                next_blocks_state[b_key] = {"hash": b_hash, "turns_omitted": 0}
            elif k_turns > 0 and (prev_omitted + 1) >= k_turns:
                block_actions[b_key] = "SHOW_REFRESH"
                next_blocks_state[b_key] = {"hash": b_hash, "turns_omitted": 0}
            else:
                new_omitted = prev_omitted + 1
                block_actions[b_key] = "OMIT"
                next_blocks_state[b_key] = {"hash": b_hash, "turns_omitted": new_omitted}
                cnt_sfx = f"({new_omitted}/{k_turns})" if k_turns > 0 else "(unchanged)"
                src_age_s = block_src_ages.get(b_key)
                age_sfx = f"[src_age={format_age_sec_compact(src_age_s)}]" if (src_age_s is not None and src_age_s >= 120.0) else ""
                omitted_tokens.append(f"{spec.label}{cnt_sfx}{age_sfx}")

        # Per-symbol recent_orders delta evaluation (G1 cadence)
        ro_spec = PROMPT_DELTA_SPECS_BY_KEY["recent_orders"]
        ro_k = eff_cadences.get(ro_spec.group, ro_spec.refresh_turns)
        ro_actions: Dict[str, str] = {}
        next_ro_state: Dict[str, Dict[str, Any]] = {}
        omitted_ro_max = 0
        omitted_ro_cnt = 0
        for r_sym, r_hash in ro_hashes.items():
            prev_r = prev_ro.get(r_sym) if isinstance(prev_ro.get(r_sym), Mapping) else {}
            prev_r_om = safe_nonneg_int(prev_r.get("turns_omitted"), default=0)
            if r_sym in changed_ro_syms:
                ro_actions[r_sym] = "SHOW_NEW"
                next_ro_state[r_sym] = {"hash": r_hash, "turns_omitted": 0}
            elif ro_k > 0 and (prev_r_om + 1) >= ro_k:
                ro_actions[r_sym] = "SHOW_REFRESH"
                next_ro_state[r_sym] = {"hash": r_hash, "turns_omitted": 0}
            else:
                new_r_om = prev_r_om + 1
                ro_actions[r_sym] = "OMIT"
                next_ro_state[r_sym] = {"hash": r_hash, "turns_omitted": new_r_om}
                omitted_ro_cnt += 1
                if new_r_om > omitted_ro_max:
                    omitted_ro_max = new_r_om
        if ro_hashes:
            any_ro_shown = any(a != "OMIT" for a in ro_actions.values())
            block_actions["recent_orders"] = "SHOW_NEW" if changed_ro_syms else ("SHOW_REFRESH" if any_ro_shown else "OMIT")
            next_blocks_state["recent_orders"] = {
                "hash": block_hashes["recent_orders"],
                "turns_omitted": 0 if any_ro_shown else omitted_ro_max,
            }
            if omitted_ro_cnt > 0:
                cnt_sfx = f"({omitted_ro_max}/{ro_k})" if ro_k > 0 else "(unchanged)"
                omitted_tokens.append(f"{ro_spec.label}[{omitted_ro_cnt}]{cnt_sfx}")

        # Keep union of seen headlines on this thread (bounded to latest 40) so headlines never oscillate
        merged_news = list(dict.fromkeys([*news_titles, *prev_news_raw]))[:40]
        next_state = {
            "schema_version": 1,
            "conversation_id": cid_clean,
            "compacted_at_step": comp_step,
            "date_iso": date_iso,
            "ts": round(eff_now, 3),
            "blocks": next_blocks_state,
            "recent_orders": next_ro_state,
            "news_titles": merged_news,
        }
        return PromptDeltaPlan(
            mode="WARM_DELTA",
            reason="warm_delta",
            block_actions=block_actions,
            recent_order_actions=ro_actions,
            new_news_titles=new_titles_set,
            omitted_news_count=omitted_news_n,
            omitted_summary_tokens=omitted_tokens,
            next_state=next_state,
        )


TelemetryEvaluatorFactory.register("prompt_delta_keyframe", PromptDeltaKeyframeEvaluator)
_PROMPT_DELTA_KEYFRAME_EVALUATOR = TelemetryEvaluatorFactory.create("prompt_delta_keyframe")


def evaluate_prompt_delta_plan(
    sensors: Any,
    *,
    asset_scope: str = "all",
    now_ts: Optional[float] = None,
    min_usd: float = 10.0,
    max_usd: float = 2000.0,
    is_warm_resume: bool = False,
    prev_state: Optional[Mapping[str, Any]] = None,
    conversation_id: str = "",
    compacted_at_step: int = 0,
    escalate_high_effort: bool = False,
    urgent_escalation: bool = False,
    extreme_escalation: bool = False,
    ui_turn_pending: bool = False,
    focus_symbols: Optional[Sequence[str]] = None,
    prev_eval_was_actionable: bool = False,
    cadence_overrides: Optional[Mapping[str, int]] = None,
) -> PromptDeltaPlan:
    """Evaluate the deterministic warm-turn delta & keyframe plan for ``sensors``."""
    return _PROMPT_DELTA_KEYFRAME_EVALUATOR.evaluate(
        sensors,
        asset_scope=asset_scope,
        now_ts=now_ts,
        min_usd=min_usd,
        max_usd=max_usd,
        is_warm_resume=is_warm_resume,
        prev_state=prev_state,
        conversation_id=conversation_id,
        compacted_at_step=compacted_at_step,
        escalate_high_effort=escalate_high_effort,
        urgent_escalation=urgent_escalation,
        extreme_escalation=extreme_escalation,
        ui_turn_pending=ui_turn_pending,
        focus_symbols=focus_symbols,
        prev_eval_was_actionable=prev_eval_was_actionable,
        cadence_overrides=cadence_overrides,
    )


__all__ = [
    "PROMPT_DELTA_BLOCK_SPECS",
    "PROMPT_DELTA_GROUP_CADENCES",
    "PROMPT_DELTA_SPECS_BY_KEY",
    "PromptDeltaBlockSpec",
    "PromptDeltaKeyframeEvaluator",
    "PromptDeltaPlan",
    "evaluate_prompt_delta_plan",
]
