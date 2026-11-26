"""Pure SSOT Delta-Gate and dynamic reasoning effort evaluator for ``auto_cryptotrade_executor``."""
from __future__ import annotations

import time
from typing import Any, Collection, Dict, List, Mapping, Optional, Sequence, Tuple

from botcore import (
    _parse_bool_flag,
    are_close,
    extract_order_id,
    first_epoch_sec,
    normalize_epoch_sec,
    parse_symbol_list,
    safe_float,
    safe_pos_float,
    value_diff_to_percent,
)
from cachepaths import clean_symbol_key, normalize_venue_name
from instrument_registry import (
    canonical_display_symbol,
    disabled_registry_symbols,
    observe_only_registry_symbols,
)
from intelligence.macro.lexicon_ssot import has_acute_shock_keywords
from intelligence.macro_sentiment_ssot import normalize_threat_level
from intelligence.sentiment.llm_quota_router import is_degraded_trading_model
from providers.ai_circuit_breaker import DEFAULT_MIN_TARGET_RR
from providers.intent_state import eval_doc_was_actionable
from providers.intent_telemetry import (
    evaluate_open_order_front_run_rr,
    evaluate_trade_plan_status,
)

_ATR_15M_DELTA_GATE_MULT = 0.75


def _is_geopolitical_shock(gt: Any) -> bool:
    if not isinstance(gt, Mapping):
        return False
    geo_level = normalize_threat_level(gt.get("threat_level"))
    geo_score = safe_float(gt.get("risk_score"), default=0.0)
    return bool(geo_level == "CRITICAL_SHOCK" or (geo_score is not None and geo_score >= 0.70))


def _extract_macro_verdict_sig(sa: Any) -> str:
    """Return canonical verdict signature for macro sentiment so identical-verdict refreshes do not wake crypto LLM."""
    if not isinstance(sa, Mapping):
        return ""
    parts = [
        str(sa.get(k) or "").strip().upper()
        for k in ("market_bias", "risk_level", "recommended_action", "cross_asset_regime")
    ]
    if not any(parts):
        return ""
    return "|".join(parts)


def _extract_geo_verdict_sig(gt: Any) -> str:
    """Return canonical verdict signature for geopolitical threat so routine identical NORMAL/NONE polls do not wake LLM."""
    if not isinstance(gt, Mapping):
        return ""
    raw_lvl = str(gt.get("threat_level") or "").strip()
    raw_brk = str(gt.get("recommended_brake") or "").strip().upper()
    if not raw_lvl and not raw_brk:
        return ""
    norm_lvl = normalize_threat_level(raw_lvl) if raw_lvl else ""
    return f"{norm_lvl}|{raw_brk}"


def _extract_delta_gate_geo_ts(gt: Any, src_ts_map: Optional[Mapping[str, Any]] = None) -> float:
    s_map = src_ts_map if isinstance(src_ts_map, Mapping) else {}
    return (
        first_epoch_sec(
            gt.get("ts") if isinstance(gt, Mapping) else None,
            gt.get("evaluated_at_ts") if isinstance(gt, Mapping) else None,
            gt.get("updated_ts") if isinstance(gt, Mapping) else None,
            s_map.get("geopolitical"),
            s_map.get("geopolitical_threat"),
            default=0.0,
            allow_iso=True,
        )
        or 0.0
    )


def _has_unacknowledged_geo_shock(
    gt: Any,
    prev_eval: Optional[Mapping[str, Any]],
    *,
    src_ts_map: Optional[Mapping[str, Any]] = None,
    geo_ts: Optional[float] = None,
) -> bool:
    if not _is_geopolitical_shock(gt):
        return False
    if not isinstance(prev_eval, Mapping) or not prev_eval:
        return True
    eff_geo_ts = _extract_delta_gate_geo_ts(gt, src_ts_map) if geo_ts is None else geo_ts
    prev_meta = prev_eval.get("delta_gate_meta") if isinstance(prev_eval.get("delta_gate_meta"), Mapping) else {}
    if eff_geo_ts <= 0.0:
        cur_verdict = _extract_geo_verdict_sig(gt)
        prev_verdict = str(prev_meta.get("geo_verdict") or "").strip().upper()
        return bool(not prev_verdict or cur_verdict != prev_verdict)
    prev_ts = normalize_epoch_sec(prev_eval.get("ts"), default=0.0, allow_iso=True) or 0.0
    prev_geo_ts = normalize_epoch_sec(prev_meta.get("geo_ts"), default=0.0, allow_iso=True) or 0.0
    if prev_geo_ts > 0.0:
        return eff_geo_ts > prev_geo_ts + 0.5
    return prev_ts <= 0.0 or eff_geo_ts > prev_ts + 0.5


def _has_unacknowledged_acute_geo_headline(
    gt: Any,
    prev_eval: Optional[Mapping[str, Any]],
) -> bool:
    """Return True when an unassessed breaking headline carries acute shock keywords and differs from prev_eval."""
    note = (
        str(gt.get("live_unassessed_high_sev_note") or "").strip()
        if isinstance(gt, Mapping)
        else ""
    )
    if not (note and has_acute_shock_keywords(note)):
        return False
    prev_meta = (
        prev_eval.get("delta_gate_meta")
        if isinstance(prev_eval, Mapping) and isinstance(prev_eval.get("delta_gate_meta"), Mapping)
        else {}
    )
    return note != str(prev_meta.get("geo_unassessed_note") or "").strip()


def _has_new_fill_since_prev(
    ps_map: Mapping[str, Any],
    prev_eval: Optional[Mapping[str, Any]],
    *,
    latest_fill_ts: Optional[float] = None,
) -> bool:
    eff_latest = _extract_delta_gate_latest_fill_ts(ps_map) if latest_fill_ts is None else latest_fill_ts
    if eff_latest <= 0.0:
        return False
    prev_map = prev_eval if isinstance(prev_eval, Mapping) else {}
    prev_ts = normalize_epoch_sec(prev_map.get("ts"), default=0.0, allow_iso=True) or 0.0
    prev_meta = prev_map.get("delta_gate_meta") if isinstance(prev_map.get("delta_gate_meta"), Mapping) else {}
    prev_fill_ts = normalize_epoch_sec(prev_meta.get("latest_fill_ts"), default=0.0, allow_iso=True) or 0.0
    return bool(
        (prev_fill_ts > 0.0 and eff_latest > prev_fill_ts + 0.5)
        or (prev_ts > 0.0 and eff_latest > prev_ts + 0.5)
    )


def _has_unacknowledged_trade_plan_invalidation(
    sensors: Any,
    prev_eval: Optional[Mapping[str, Any]],
) -> bool:
    ps = getattr(sensors, "portfolio_summary", None)
    if not isinstance(ps, Mapping) or prev_eval is None:
        return False
    prev_meta = (
        prev_eval.get("delta_gate_meta")
        if isinstance(prev_eval.get("delta_gate_meta"), Mapping)
        else {}
    )
    prev_plan_raw = prev_meta.get("plan_states")
    prev_plans: Dict[str, str] = (
        {str(k): str(v).strip().upper() for k, v in prev_plan_raw.items() if str(k).strip()}
        if isinstance(prev_plan_raw, Mapping)
        else {}
    )
    eval_prices = _collect_delta_gate_prices(getattr(sensors, "prices_and_trends", None), ps)
    unact = _resolve_unactionable_delta_gate_symbols(ps)
    _, plan_trig = _evaluate_delta_gate_trade_plans(ps, eval_prices, prev_plans, unactionable_syms=unact)
    return str(plan_trig or "").strip().lower().startswith("trade_plan_invalidated")


def is_urgent_delta_gate_wakeup(
    gate_reason: str = "",
    *,
    focus_symbols: Optional[Sequence[str]] = None,
    sensors: Any = None,
    prev_eval: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Return True when a wakeup reflects an urgent catalyst (impulse, fill, trade-plan break, or fresh geo shock)."""
    if focus_symbols and parse_symbol_list(focus_symbols):
        return True
    reason_lc = str(gate_reason or "").strip().lower()
    if reason_lc.startswith(("focus_symbols(", "trade_plan_", "new_fill", "geo_unassessed_acute")):
        return True
    if sensors is not None:
        gt = getattr(sensors, "geopolitical_threat", None)
        src_ts = getattr(sensors, "source_timestamps", None)
        if _has_unacknowledged_geo_shock(
            gt, prev_eval, src_ts_map=src_ts if isinstance(src_ts, Mapping) else None
        ) or _has_unacknowledged_acute_geo_headline(gt, prev_eval):
            return True
        ps = getattr(sensors, "portfolio_summary", None)
        if isinstance(ps, Mapping) and prev_eval is not None and _has_new_fill_since_prev(ps, prev_eval):
            return True
        if _has_unacknowledged_trade_plan_invalidation(sensors, prev_eval):
            return True
    return False


def is_extreme_stakes_delta_gate_wakeup(
    gate_reason: str = "",
    *,
    sensors: Any = None,
    prev_eval: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Return True ONLY for ultra-rare extreme-complexity catalysts warranting ``-xhigh`` reasoning:
    active trade-plan invalidation (thesis break vs stop-hunt), new real fill (next-cycle planning),
    or unacknowledged critical geopolitical shock."""
    reason_lc = str(gate_reason or "").strip().lower()
    if reason_lc.startswith(("trade_plan_invalidated", "new_fill")):
        return True
    if sensors is not None:
        gt = getattr(sensors, "geopolitical_threat", None)
        src_ts = getattr(sensors, "source_timestamps", None)
        if _has_unacknowledged_geo_shock(gt, prev_eval, src_ts_map=src_ts if isinstance(src_ts, Mapping) else None):
            return True
        ps = getattr(sensors, "portfolio_summary", None)
        if isinstance(ps, Mapping) and prev_eval is not None and _has_new_fill_since_prev(ps, prev_eval):
            return True
        if _has_unacknowledged_trade_plan_invalidation(sensors, prev_eval):
            return True
    return False


def _extract_delta_gate_latest_fill_ts(portfolio_summary: Mapping[str, Any]) -> float:
    """Extract the newest fill/execution epoch-second timestamp across recent_orders and last_cycle_execution_status."""
    latest = 0.0
    recent_orders = portfolio_summary.get("recent_orders")
    if isinstance(recent_orders, Mapping):
        for oinfo in recent_orders.values():
            if not isinstance(oinfo, Mapping):
                continue
            for cand in (
                oinfo.get("ts_ms"),
                oinfo.get("ts"),
                (oinfo.get("last_buy") or {}).get("ts_ms") if isinstance(oinfo.get("last_buy"), Mapping) else None,
                (oinfo.get("last_buy") or {}).get("ts") if isinstance(oinfo.get("last_buy"), Mapping) else None,
                (oinfo.get("last_sell") or {}).get("ts_ms") if isinstance(oinfo.get("last_sell"), Mapping) else None,
                (oinfo.get("last_sell") or {}).get("ts") if isinstance(oinfo.get("last_sell"), Mapping) else None,
            ):
                ts_v = normalize_epoch_sec(cand, default=0.0, allow_iso=True) or 0.0
                if ts_v > latest:
                    latest = ts_v
    raw_lces = portfolio_summary.get("last_cycle_execution_status")
    lces_items = raw_lces if isinstance(raw_lces, list) else ([raw_lces] if isinstance(raw_lces, Mapping) else [])
    for item in lces_items:
        if not isinstance(item, Mapping):
            continue
        st = str(item.get("status") or "").strip().upper()
        if st not in ("FILLED", "PARTIALLY_FILLED", "REJECTED", "FAILED"):
            continue
        ts_v = first_epoch_sec(
            item.get("filled_ts"),
            item.get("updated_ts"),
            item.get("ts_ms"),
            item.get("ts"),
            default=0.0,
            allow_iso=True,
        ) or 0.0
        if ts_v > latest:
            latest = ts_v
    return latest


_prev_eval_was_actionable = eval_doc_was_actionable


def _collect_delta_gate_prices(
    prices_and_trends: Any,
    ps_map: Mapping[str, Any],
) -> Dict[str, float]:
    """Collect canonical symbol -> rounded spot price baseline across market prices and tradable positions."""
    eval_prices: Dict[str, float] = {}
    if isinstance(prices_and_trends, Mapping):
        for sym, pinfo in prices_and_trends.items():
            if not isinstance(pinfo, Mapping):
                continue
            sym_raw = str(sym or "").strip()
            sym_k = canonical_display_symbol(sym_raw) or sym_raw.upper()
            px = safe_pos_float(pinfo.get("price"), default=None)
            if sym_k and px is not None:
                eval_prices[sym_k] = round(px, 8)
    raw_positions = ps_map.get("tradable_positions")
    if isinstance(raw_positions, list):
        for pos in raw_positions:
            if not isinstance(pos, Mapping):
                continue
            sym_raw = str(pos.get("symbol") or "").strip()
            sym_k = canonical_display_symbol(sym_raw, venue=pos.get("venue")) or sym_raw.upper()
            px = safe_pos_float(pos.get("price"), default=None)
            if sym_k and px is not None and sym_k not in eval_prices:
                eval_prices[sym_k] = round(px, 8)
    return eval_prices


def _collect_open_order_attention(
    ps_map: Mapping[str, Any],
    prev_order_flags: Optional[Mapping[str, str]] = None,
    *,
    min_target_rr: float = DEFAULT_MIN_TARGET_RR,
) -> Tuple[List[str], Dict[str, str], str]:
    """Return sorted unique open order IDs, active per-order attention flags, and first transition wakeup reason."""
    raw_open = ps_map.get("open_limit_orders")
    if not isinstance(raw_open, list):
        raw_open = ps_map.get("open_orders") if isinstance(ps_map.get("open_orders"), list) else []
    raw_plans = ps_map.get("active_ai_trade_plans")
    active_plans = raw_plans if isinstance(raw_plans, list) else []
    eff_min_rr = max(0.0, safe_float(min_target_rr, default=DEFAULT_MIN_TARGET_RR) or 0.0)
    prev_flags = prev_order_flags if isinstance(prev_order_flags, Mapping) else {}
    open_order_ids: List[str] = []
    order_attention_flags: Dict[str, str] = {}
    attention_reason = ""
    used_plan_indices: set[int] = set()
    for oo in raw_open:
        if not isinstance(oo, Mapping):
            continue
        oid = (
            extract_order_id(oo.get("order_id"), default="")
            or extract_order_id(oo.get("orderId"), default="")
            or extract_order_id(oo.get("client_order_id"), default="")
            or extract_order_id(oo.get("clientOrderId"), default="")
            or f"{oo.get('venue', '')}:{oo.get('symbol', '')}:{oo.get('side', '')}:{oo.get('price', '')}"
        )
        if oid:
            open_order_ids.append(oid)
        st = str(oo.get("status") or "").strip().upper()
        filled_q = safe_float(oo.get("filled_qty"), default=0.0) or 0.0
        dist_pct = safe_float(oo.get("dist_from_spot_pct"), default=None)
        fill_prob = safe_float(oo.get("fill_prob_1h_pct"), default=None)
        fr_px = safe_pos_float(oo.get("recommended_front_run_px"), default=None)
        cur_flag = ""
        cur_reason = ""
        if st == "PARTIALLY_FILLED" or filled_q > 0.0:
            cur_flag = f"open_order_partial_fill:{round(filled_q, 8)}"
            cur_reason = "open_order_partial_fill"
        elif (fill_prob is not None and fill_prob <= 5.0) or (dist_pct is not None and abs(dist_pct) > 4.0):
            cur_flag = "open_order_stale_or_far"
            cur_reason = "open_order_stale_or_far"
        elif fr_px is not None:
            fr_ok, _ = evaluate_open_order_front_run_rr(
                oo, active_plans, used_plan_indices=used_plan_indices, min_target_rr=eff_min_rr
            )
            if fr_ok:
                cur_flag = "open_order_wall_trapped"
                cur_reason = "open_order_wall_trapped"
        elif oo.get("ob_flag") == "[ob_stale]" and prev_flags.get(oid) == "open_order_wall_trapped":
            cur_flag = "open_order_wall_trapped"
        if oid and cur_flag:
            order_attention_flags[oid] = cur_flag
            if not attention_reason and prev_flags.get(oid) != cur_flag:
                attention_reason = cur_reason
    return sorted(set(open_order_ids)), order_attention_flags, attention_reason


def _evaluate_delta_gate_trade_plans(
    ps_map: Mapping[str, Any],
    eval_prices: Mapping[str, float],
    prev_plan_states: Mapping[str, str],
    *,
    unactionable_syms: Optional[Collection[str]] = None,
) -> Tuple[Dict[str, str], str]:
    """Evaluate active AI trade plans and detect first-transition INVALIDATED / TARGET_HIT wake triggers."""
    plan_states: Dict[str, str] = {}
    trigger_reason = ""
    raw_plans = ps_map.get("active_ai_trade_plans")
    if not isinstance(raw_plans, list):
        return plan_states, trigger_reason
    ignored_syms = set(unactionable_syms) if unactionable_syms else set()
    for pl in raw_plans:
        if not isinstance(pl, Mapping):
            continue
        pl_ven = normalize_venue_name(pl.get("venue"), default="")
        pl_sym_raw = str(pl.get("symbol") or "").strip()
        pl_sym = canonical_display_symbol(pl_sym_raw, venue=pl_ven) or pl_sym_raw.upper()
        pl_side = str(pl.get("side") or "").strip().upper()
        if not pl_sym or pl_side not in ("BUY", "SELL"):
            continue
        spot_px = eval_prices.get(pl_sym) or safe_pos_float(pl.get("spot_px"), default=None)
        inv_px = safe_pos_float(pl.get("invalidation_px"), default=None)
        tgt_px = safe_pos_float(pl.get("target_px"), default=None)
        pl_is_resting = (str(pl.get("order_state") or "").strip().upper() == "RESTING")
        stored_status = str(pl.get("status") or "").strip().upper()
        if not pl_is_resting and stored_status in ("INVALIDATED", "TARGET_HIT"):
            pl_status = stored_status
        elif spot_px is not None and (inv_px is not None or tgt_px is not None):
            pl_status = evaluate_trade_plan_status(pl_side, spot_px, inv_px, tgt_px, is_resting=pl_is_resting)
        else:
            pl_status = stored_status or "ACTIVE"
        pl_key = f"{pl_ven}:{pl_sym}:{pl_side}"
        _sev = {"ACTIVE": 0, "TARGET_HIT": 1, "INVALIDATED": 2}
        if pl_key not in plan_states or _sev.get(pl_status, 0) > _sev.get(plan_states[pl_key], 0):
            plan_states[pl_key] = pl_status
        if (
            not trigger_reason
            and not _is_ignored_delta_gate_symbol(pl_sym, ignored_syms)
            and pl_status in ("INVALIDATED", "TARGET_HIT")
            and prev_plan_states.get(pl_key) != pl_status
        ):
            trigger_reason = f"trade_plan_{pl_status.lower()}({pl_sym})"
    return plan_states, trigger_reason


def _collect_delta_gate_atr_15m(prices_and_trends: Any) -> Dict[str, float]:
    """Collect canonical symbol -> 15m ATR (%) map (or ``vol_1h_pct * 0.5`` proxy) when volatility telemetry is present."""
    atr_map: Dict[str, float] = {}
    if isinstance(prices_and_trends, Mapping):
        for sym, pinfo in prices_and_trends.items():
            if not isinstance(pinfo, Mapping):
                continue
            sym_raw = str(sym or "").strip()
            sym_k = canonical_display_symbol(sym_raw) or sym_raw.upper()
            atr_15m = safe_pos_float(pinfo.get("atr_15m_pct"), default=None)
            if atr_15m is None:
                vol_1h = safe_pos_float(pinfo.get("vol_1h_pct"), default=None)
                if vol_1h is not None:
                    atr_15m = vol_1h * 0.5
            if sym_k and atr_15m is not None:
                atr_map[sym_k] = atr_15m
    return atr_map


def _is_ignored_delta_gate_symbol(sym_u: str, ignored_syms: Collection[str]) -> bool:
    if not ignored_syms or not sym_u:
        return False
    if sym_u in ignored_syms or sym_u.upper() in ignored_syms:
        return True
    sym_k = clean_symbol_key(sym_u) or ""
    return bool(sym_k and sym_k in ignored_syms)


def _resolve_unactionable_delta_gate_symbols(ps_map: Mapping[str, Any]) -> set[str]:
    """Return canonical symbols that are observe-only/disabled or whose tracked positions all have zero trade capacity."""
    non_tradable: set[str] = set()
    for raw_s in observe_only_registry_symbols() | disabled_registry_symbols():
        if raw_s:
            non_tradable.add(raw_s)
            if canon := canonical_display_symbol(raw_s):
                non_tradable.add(canon)
    raw_positions = ps_map.get("tradable_positions")
    if not isinstance(raw_positions, list) or not raw_positions:
        return non_tradable
    raw_open = ps_map.get("open_limit_orders")
    if not isinstance(raw_open, list):
        raw_open = ps_map.get("open_orders") if isinstance(ps_map.get("open_orders"), list) else []
    open_syms: set[str] = set()
    for oo in raw_open:
        if not isinstance(oo, Mapping):
            continue
        oo_raw = str(oo.get("symbol") or "").strip()
        oo_sym = canonical_display_symbol(oo_raw, venue=oo.get("venue")) or oo_raw.upper()
        if oo_sym:
            open_syms.add(oo_sym)
    seen_syms: set[str] = set()
    actionable_syms: set[str] = set()
    for pos in raw_positions:
        if not isinstance(pos, Mapping):
            continue
        sym_raw = str(pos.get("symbol") or "").strip()
        sym_k = canonical_display_symbol(sym_raw, venue=pos.get("venue")) or sym_raw.upper()
        if not sym_k:
            continue
        seen_syms.add(sym_k)
        if "effective_max_buy_usd" not in pos and "effective_max_sell_usd" not in pos:
            actionable_syms.add(sym_k)
            continue
        mb = safe_pos_float(pos.get("effective_max_buy_usd"), default=0.0) or 0.0
        ms = safe_pos_float(pos.get("effective_max_sell_usd"), default=0.0) or 0.0
        rmb = safe_pos_float(pos.get("replace_max_buy_usd"), default=0.0) or 0.0
        rms = safe_pos_float(pos.get("replace_max_sell_usd"), default=0.0) or 0.0
        if mb > 0.0 or ms > 0.0 or rmb > 0.0 or rms > 0.0 or sym_k in open_syms:
            actionable_syms.add(sym_k)
    act_keys = {k for s in actionable_syms for k in (s, s.upper(), clean_symbol_key(s) or "") if k}
    return (seen_syms - actionable_syms) | (non_tradable - act_keys)


def _compute_max_spot_move(
    eval_prices: Mapping[str, float],
    prev_prices_raw: Any,
    *,
    min_move_pct: float = 0.35,
    high_move_pct: float = 1.0,
    atr_15m_by_sym: Optional[Mapping[str, float]] = None,
    unactionable_syms: Optional[Collection[str]] = None,
) -> Tuple[float, str, float, str, float, float, int, Dict[str, float], bool]:
    """Compute raw max spot move, strongest ATR-scaled wakeup breach, and high-effort breach via ``are_close``."""
    prev_prices: Dict[str, float] = {}
    if isinstance(prev_prices_raw, Mapping):
        for k, v in prev_prices_raw.items():
            k_raw = str(k or "").strip()
            sym_k = canonical_display_symbol(k_raw) or k_raw.upper()
            px = safe_pos_float(v, default=None)
            if sym_k and px is not None:
                prev_prices[sym_k] = px
    atr_map = atr_15m_by_sym if isinstance(atr_15m_by_sym, Mapping) else {}
    ignored_syms = set(unactionable_syms) if unactionable_syms else set()
    max_move_pct = 0.0
    max_move_sym = ""
    max_move_thresh_pct = min_move_pct
    breach_sym = ""
    breach_move_pct = 0.0
    breach_thresh_pct = min_move_pct
    best_breach_ratio = 0.0
    high_effort_breach = False
    common_count = 0
    for sym_u, cur_px in eval_prices.items():
        if sym_u not in prev_prices:
            continue
        common_count += 1
        if _is_ignored_delta_gate_symbol(sym_u, ignored_syms):
            continue
        base_px = prev_prices[sym_u]
        move_pct = abs(value_diff_to_percent(base_px, cur_px))
        atr_15m = safe_pos_float(atr_map.get(sym_u), default=None)
        sym_thresh = (
            min(high_move_pct, max(min_move_pct, atr_15m * _ATR_15M_DELTA_GATE_MULT))
            if atr_15m is not None
            else min_move_pct
        )
        if move_pct > max_move_pct:
            max_move_pct = move_pct
            max_move_sym = sym_u
            max_move_thresh_pct = sym_thresh
        if high_move_pct > 0.0 and not are_close(cur_px, base_px, tolerance_percent=high_move_pct):
            high_effort_breach = True
        if sym_thresh > 0.0 and not are_close(cur_px, base_px, tolerance_percent=sym_thresh):
            ratio = move_pct / sym_thresh
            if ratio > best_breach_ratio:
                best_breach_ratio = ratio
                breach_sym = sym_u
                breach_move_pct = move_pct
                breach_thresh_pct = sym_thresh
    return (
        max_move_pct,
        max_move_sym,
        max_move_thresh_pct,
        breach_sym,
        breach_move_pct,
        breach_thresh_pct,
        common_count,
        prev_prices,
        high_effort_breach,
    )


def evaluate_crypto_executor_delta_gate(
    sensors: Any,
    prev_eval: Optional[Mapping[str, Any]],
    *,
    now_ts: float,
    focus_symbols: Optional[Sequence[str]] = None,
    min_price_move_pct: float = 0.35,
    max_idle_sec: float = 1080.0,
    high_effort_price_move_pct: float = 1.0,
    ui_turn_pending: bool = False,
    min_target_rr: float = DEFAULT_MIN_TARGET_RR,
) -> Tuple[bool, str, bool, Dict[str, Any]]:
    """Pure SSOT Delta-Gate and dynamic reasoning effort evaluator for ``auto_cryptotrade_executor``.

    Returns ``(should_skip_llm, reason, escalate_high_effort, snapshot_meta)``.
    Skips redundant LLM calls only on quiet flat repeat cycles where:
    - no ``--focus-symbols`` impulse wakeup and no pending Web UI user turn exist;
    - previous real eval was a healthy non-degraded pure ``HOLD`` with a valid price baseline;
    - elapsed time since last real eval is ``< max_idle_sec``;
    - cumulative max spot move across all tracked pairs since last real eval is ``< min_price_move_pct``;
    - no new fills, no macro/geopolitical verdict changes, and no open-order changes or
      stale / wall-trapped / partially-filled open limit orders require attention.
    """
    eff_min_move_pct = safe_pos_float(min_price_move_pct, default=0.35) or 0.35
    eff_max_idle_sec = safe_pos_float(max_idle_sec, default=1080.0) or 1080.0
    eff_high_move_pct = safe_pos_float(high_effort_price_move_pct, default=1.0) or 1.0
    eff_now = safe_pos_float(now_ts, default=0.0) or time.time()

    ps = getattr(sensors, "portfolio_summary", None)
    ps_map = ps if isinstance(ps, Mapping) else {}
    prices_and_trends = getattr(sensors, "prices_and_trends", None)
    eval_prices = _collect_delta_gate_prices(prices_and_trends, ps_map)
    atr_15m_by_sym = _collect_delta_gate_atr_15m(prices_and_trends)

    sa = getattr(sensors, "sentiment_advisor", None)
    gt = getattr(sensors, "geopolitical_threat", None)
    src_ts = getattr(sensors, "source_timestamps", None)
    src_ts_map = src_ts if isinstance(src_ts, Mapping) else {}
    macro_ts = first_epoch_sec(
        sa.get("ts") if isinstance(sa, Mapping) else None,
        sa.get("evaluated_at_ts") if isinstance(sa, Mapping) else None,
        sa.get("updated_ts") if isinstance(sa, Mapping) else None,
        src_ts_map.get("sentiment_advisor"),
        default=0.0,
        allow_iso=True,
    ) or 0.0
    geo_ts = _extract_delta_gate_geo_ts(gt, src_ts_map)
    geo_shock = _has_unacknowledged_geo_shock(gt, prev_eval, src_ts_map=src_ts_map, geo_ts=geo_ts)

    latest_fill_ts = _extract_delta_gate_latest_fill_ts(ps_map)

    prev_map = prev_eval if isinstance(prev_eval, Mapping) else {}
    prev_ts = normalize_epoch_sec(prev_map.get("ts"), default=0.0, allow_iso=True) or 0.0
    prev_meta_raw = prev_map.get("delta_gate_meta")
    prev_meta = prev_meta_raw if isinstance(prev_meta_raw, Mapping) else {}
    prev_order_flags_raw = prev_meta.get("order_attention_flags")
    prev_order_flags: Dict[str, str] = (
        {str(k).strip(): str(v).strip() for k, v in prev_order_flags_raw.items() if str(k).strip() and str(v).strip()}
        if isinstance(prev_order_flags_raw, Mapping)
        else {}
    )
    open_order_ids, order_attention_flags, open_order_attention_reason = _collect_open_order_attention(
        ps_map, prev_order_flags, min_target_rr=min_target_rr
    )
    prev_plan_states_raw = prev_meta.get("plan_states")
    prev_plan_states: Dict[str, str] = (
        {str(k): str(v).strip().upper() for k, v in prev_plan_states_raw.items() if str(k).strip()}
        if isinstance(prev_plan_states_raw, Mapping)
        else {}
    )
    unactionable_syms = _resolve_unactionable_delta_gate_symbols(ps_map)
    plan_states, plan_trigger_reason = _evaluate_delta_gate_trade_plans(
        ps_map, eval_prices, prev_plan_states, unactionable_syms=unactionable_syms
    )

    macro_verdict = _extract_macro_verdict_sig(sa)
    geo_verdict = _extract_geo_verdict_sig(gt)
    geo_unassessed_note = (
        str(gt.get("live_unassessed_high_sev_note") or "").strip()
        if isinstance(gt, Mapping)
        else ""
    )
    geo_unassessed_acute_new = _has_unacknowledged_acute_geo_headline(gt, prev_map)
    snapshot_meta: Dict[str, Any] = {
        "eval_prices": eval_prices,
        "macro_ts": round(macro_ts, 3),
        "macro_verdict": macro_verdict,
        "geo_ts": round(geo_ts, 3),
        "geo_verdict": geo_verdict,
        "latest_fill_ts": round(latest_fill_ts, 3),
        "open_order_ids": open_order_ids,
        "order_attention_flags": order_attention_flags,
        "plan_states": plan_states,
    }
    if geo_unassessed_note:
        snapshot_meta["geo_unassessed_note"] = geo_unassessed_note
    op_note_obj = getattr(sensors, "operator_regime_note", None)
    op_note_sig = (
        f"{op_note_obj.get('mtime_ts', '')}|{str(op_note_obj.get('text') or '').strip()}"
        if isinstance(op_note_obj, Mapping) and str(op_note_obj.get("text") or "").strip()
        else ""
    )
    if op_note_sig:
        snapshot_meta["operator_note_sig"] = op_note_sig

    (
        max_move_pct,
        _max_move_sym,
        max_move_thresh_pct,
        breach_sym,
        breach_move_pct,
        breach_thresh_pct,
        common_count,
        prev_prices,
        high_effort_breach,
    ) = _compute_max_spot_move(
        eval_prices,
        prev_meta.get("eval_prices"),
        min_move_pct=eff_min_move_pct,
        high_move_pct=eff_high_move_pct,
        atr_15m_by_sym=atr_15m_by_sym,
        unactionable_syms=unactionable_syms,
    )
    has_new_fill = _has_new_fill_since_prev(ps_map, prev_map, latest_fill_ts=latest_fill_ts)

    clean_focus = parse_symbol_list(focus_symbols) if focus_symbols else []
    escalate_high_effort = bool(
        clean_focus
        or high_effort_breach
        or has_new_fill
        or geo_shock
        or geo_unassessed_acute_new
        or plan_trigger_reason
    )

    if clean_focus:
        return False, f"focus_symbols({','.join(clean_focus)})", escalate_high_effort, snapshot_meta
    if _parse_bool_flag(ui_turn_pending, False):
        return False, "ui_turn_pending", escalate_high_effort, snapshot_meta
    prev_op_note_sig = str(prev_meta.get("operator_note_sig") or "")
    if op_note_sig != prev_op_note_sig:
        return False, ("operator_note_updated" if op_note_sig else "operator_note_cleared"), escalate_high_effort, snapshot_meta
    if not prev_map or prev_ts <= 0.0 or prev_ts > eff_now + 5.0:
        return False, "no_prev_eval", escalate_high_effort, snapshot_meta
    if is_degraded_trading_model(prev_map.get("model_used")):
        return False, "prev_eval_degraded_model", escalate_high_effort, snapshot_meta
    if _prev_eval_was_actionable(prev_map):
        return False, "prev_eval_actionable", escalate_high_effort, snapshot_meta
    if not prev_prices or not eval_prices or common_count == 0:
        return False, "no_price_baseline", escalate_high_effort, snapshot_meta

    elapsed_sec = max(0.0, eff_now - prev_ts)
    if plan_trigger_reason:
        return False, plan_trigger_reason, True, snapshot_meta
    if elapsed_sec >= eff_max_idle_sec:
        return False, f"max_idle_elapsed({elapsed_sec:.0f}s>={eff_max_idle_sec:.0f}s)", escalate_high_effort, snapshot_meta
    if breach_sym:
        return False, f"price_move({breach_sym}:{breach_move_pct:.2f}%>={breach_thresh_pct:.2f}%)", escalate_high_effort, snapshot_meta
    if has_new_fill:
        return False, "new_fill", escalate_high_effort, snapshot_meta

    prev_macro_ts = normalize_epoch_sec(prev_meta.get("macro_ts"), default=0.0, allow_iso=True) or 0.0
    prev_macro_verdict = str(prev_meta.get("macro_verdict") or "").strip().upper()
    if macro_ts > 0.0 and (
        (prev_macro_ts > 0.0 and macro_ts > prev_macro_ts + 0.5)
        or (prev_macro_ts <= 0.0 and macro_ts > prev_ts + 0.5)
    ):
        if not macro_verdict or not prev_macro_verdict or macro_verdict != prev_macro_verdict:
            return False, "macro_update", escalate_high_effort, snapshot_meta

    prev_geo_ts = normalize_epoch_sec(prev_meta.get("geo_ts"), default=0.0, allow_iso=True) or 0.0
    prev_geo_verdict = str(prev_meta.get("geo_verdict") or "").strip().upper()
    if geo_shock or (
        geo_ts > 0.0
        and (
            (prev_geo_ts > 0.0 and geo_ts > prev_geo_ts + 0.5)
            or (prev_geo_ts <= 0.0 and geo_ts > prev_ts + 0.5)
        )
        and (not geo_verdict or not prev_geo_verdict or geo_verdict != prev_geo_verdict)
    ):
        return False, "geo_update", escalate_high_effort, snapshot_meta
    if geo_unassessed_acute_new:
        return False, "geo_unassessed_acute_headline", escalate_high_effort, snapshot_meta

    prev_oids = sorted({str(x).strip() for x in (prev_meta.get("open_order_ids") or []) if str(x).strip()})
    if open_order_ids != prev_oids:
        return False, "open_orders_changed", escalate_high_effort, snapshot_meta
    if open_order_attention_reason:
        return False, open_order_attention_reason, escalate_high_effort, snapshot_meta

    return (
        True,
        f"quiet_flat(max_move={max_move_pct:.2f}%<{max_move_thresh_pct:.2f}%, idle={elapsed_sec:.0f}s<{eff_max_idle_sec:.0f}s)",
        False,
        snapshot_meta,
    )


evaluate_trade_executor_delta_gate = evaluate_crypto_executor_delta_gate
