"""Pure SSOT Delta-Gate and dynamic reasoning effort evaluator for ``auto_cryptotrade_executor``."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from botcore import (
    _parse_bool_flag,
    extract_order_id,
    first_epoch_sec,
    normalize_epoch_sec,
    parse_symbol_list,
    safe_float,
    safe_pos_float,
)
from cachepaths import normalize_venue_name
from instrument_registry import canonical_display_symbol
from intelligence.macro_sentiment_ssot import normalize_threat_level
from intelligence.sentiment.llm_quota_router import is_degraded_trading_model
from providers.intent_state import decision_has_cancel_directive
from providers.intent_telemetry import evaluate_trade_plan_status

_ATR_15M_DELTA_GATE_MULT = 0.75


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


def _prev_eval_was_actionable(prev_eval: Mapping[str, Any]) -> bool:
    """Return True when the prior eval emitted any BUY/SELL/cancel directive or non-HOLD intent."""
    decs = prev_eval.get("market_trading_decisions")
    dec_list = decs if isinstance(decs, list) else []
    if not dec_list and isinstance(prev_eval.get("market_trading_decision"), Mapping):
        dec_list = [prev_eval["market_trading_decision"]]
    for d in dec_list:
        if not isinstance(d, Mapping):
            continue
        if str(d.get("decision") or "HOLD").strip().upper() in ("BUY", "SELL") or decision_has_cancel_directive(dict(d)):
            return True
    intents = prev_eval.get("intents_emitted")
    if isinstance(intents, list):
        for it in intents:
            if not isinstance(it, Mapping):
                continue
            side = str(it.get("side") or it.get("decision") or "HOLD").strip().upper()
            if side in ("BUY", "SELL"):
                return True
    return False


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
) -> Tuple[List[str], Dict[str, str], str]:
    """Return sorted unique open order IDs, active per-order attention flags, and first transition wakeup reason."""
    raw_open = ps_map.get("open_limit_orders")
    if not isinstance(raw_open, list):
        raw_open = ps_map.get("open_orders") if isinstance(ps_map.get("open_orders"), list) else []
    prev_flags = prev_order_flags if isinstance(prev_order_flags, Mapping) else {}
    open_order_ids: List[str] = []
    order_attention_flags: Dict[str, str] = {}
    attention_reason = ""
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
        cur_flag = ""
        cur_reason = ""
        if st == "PARTIALLY_FILLED" or filled_q > 0.0:
            cur_flag = f"open_order_partial_fill:{round(filled_q, 8)}"
            cur_reason = "open_order_partial_fill"
        elif dist_pct is not None and abs(dist_pct) <= 0.50:
            cur_flag = "open_order_near_fill"
            cur_reason = "open_order_near_fill"
        elif (fill_prob is not None and fill_prob <= 5.0) or (dist_pct is not None and abs(dist_pct) > 4.0):
            cur_flag = "open_order_stale_or_far"
            cur_reason = "open_order_stale_or_far"
        elif oo.get("recommended_front_run_px") or oo.get("ask_wall_ahead_px") or oo.get("bid_wall_ahead_px"):
            cur_flag = "open_order_wall_trapped"
            cur_reason = "open_order_wall_trapped"
        if oid and cur_flag:
            order_attention_flags[oid] = cur_flag
            if not attention_reason and prev_flags.get(oid) != cur_flag:
                attention_reason = cur_reason
    return sorted(set(open_order_ids)), order_attention_flags, attention_reason


def _evaluate_delta_gate_trade_plans(
    ps_map: Mapping[str, Any],
    eval_prices: Mapping[str, float],
    prev_plan_states: Mapping[str, str],
) -> Tuple[Dict[str, str], str]:
    """Evaluate active AI trade plans and detect first-transition INVALIDATED / TARGET_HIT wake triggers."""
    plan_states: Dict[str, str] = {}
    trigger_reason = ""
    raw_plans = ps_map.get("active_ai_trade_plans")
    if not isinstance(raw_plans, list):
        return plan_states, trigger_reason
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
        if spot_px is not None and (inv_px is not None or tgt_px is not None):
            pl_status = evaluate_trade_plan_status(pl_side, spot_px, inv_px, tgt_px)
        else:
            pl_status = str(pl.get("status") or "ACTIVE").strip().upper() or "ACTIVE"
        pl_key = f"{pl_ven}:{pl_sym}:{pl_side}"
        _sev = {"ACTIVE": 0, "TARGET_HIT": 1, "INVALIDATED": 2}
        if pl_key not in plan_states or _sev.get(pl_status, 0) > _sev.get(plan_states[pl_key], 0):
            plan_states[pl_key] = pl_status
        if (
            not trigger_reason
            and pl_status in ("INVALIDATED", "TARGET_HIT")
            and prev_plan_states.get(pl_key) != pl_status
        ):
            trigger_reason = f"trade_plan_{pl_status.lower()}({pl_sym})"
    return plan_states, trigger_reason


def _collect_delta_gate_atr_15m(prices_and_trends: Any) -> Dict[str, float]:
    """Collect canonical symbol -> 15m ATR (%) map when Faza 1 ATR telemetry is present."""
    atr_map: Dict[str, float] = {}
    if isinstance(prices_and_trends, Mapping):
        for sym, pinfo in prices_and_trends.items():
            if not isinstance(pinfo, Mapping):
                continue
            sym_raw = str(sym or "").strip()
            sym_k = canonical_display_symbol(sym_raw) or sym_raw.upper()
            atr_15m = safe_pos_float(pinfo.get("atr_15m_pct"), default=None)
            if sym_k and atr_15m is not None:
                atr_map[sym_k] = atr_15m
    return atr_map


def _compute_max_spot_move(
    eval_prices: Mapping[str, float],
    prev_prices_raw: Any,
    *,
    min_move_pct: float = 0.35,
    high_move_pct: float = 1.0,
    atr_15m_by_sym: Optional[Mapping[str, float]] = None,
) -> Tuple[float, str, float, str, float, float, int, Dict[str, float]]:
    """Compute raw max spot move and strongest per-symbol ATR-scaled threshold breach since the prior real eval."""
    prev_prices: Dict[str, float] = {}
    if isinstance(prev_prices_raw, Mapping):
        for k, v in prev_prices_raw.items():
            k_raw = str(k or "").strip()
            sym_k = canonical_display_symbol(k_raw) or k_raw.upper()
            px = safe_pos_float(v, default=None)
            if sym_k and px is not None:
                prev_prices[sym_k] = px
    atr_map = atr_15m_by_sym if isinstance(atr_15m_by_sym, Mapping) else {}
    max_move_pct = 0.0
    max_move_sym = ""
    max_move_thresh_pct = min_move_pct
    breach_sym = ""
    breach_move_pct = 0.0
    breach_thresh_pct = min_move_pct
    best_breach_ratio = 0.0
    common_count = 0
    for sym_u, cur_px in eval_prices.items():
        if sym_u not in prev_prices:
            continue
        common_count += 1
        base_px = prev_prices[sym_u]
        move_pct = abs((cur_px - base_px) / base_px) * 100.0
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
        if sym_thresh > 0.0 and move_pct >= sym_thresh:
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
) -> Tuple[bool, str, bool, Dict[str, Any]]:
    """Pure SSOT Delta-Gate and dynamic reasoning effort evaluator for ``auto_cryptotrade_executor``.

    Returns ``(should_skip_llm, reason, escalate_high_effort, snapshot_meta)``.
    Skips redundant LLM calls only on quiet flat repeat cycles where:
    - no ``--focus-symbols`` impulse wakeup and no pending Web UI user turn exist;
    - previous real eval was a healthy non-degraded pure ``HOLD`` with a valid price baseline;
    - elapsed time since last real eval is ``< max_idle_sec``;
    - cumulative max spot move across all tracked pairs since last real eval is ``< min_price_move_pct``;
    - no new fills, no macro/geopolitical assessment updates, and no open-order changes or
      near-fill / stale / wall-trapped / partially-filled open limit orders require attention.
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
    geo_ts = first_epoch_sec(
        gt.get("ts") if isinstance(gt, Mapping) else None,
        gt.get("evaluated_at_ts") if isinstance(gt, Mapping) else None,
        gt.get("updated_ts") if isinstance(gt, Mapping) else None,
        src_ts_map.get("geopolitical"),
        src_ts_map.get("geopolitical_threat"),
        default=0.0,
        allow_iso=True,
    ) or 0.0
    geo_level = normalize_threat_level(gt.get("threat_level")) if isinstance(gt, Mapping) else ""
    geo_score = safe_float(gt.get("risk_score"), default=0.0) if isinstance(gt, Mapping) else 0.0
    geo_shock = bool(geo_level == "CRITICAL_SHOCK" or (geo_score is not None and geo_score >= 0.70))

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
        ps_map, prev_order_flags
    )
    prev_plan_states_raw = prev_meta.get("plan_states")
    prev_plan_states: Dict[str, str] = (
        {str(k): str(v).strip().upper() for k, v in prev_plan_states_raw.items() if str(k).strip()}
        if isinstance(prev_plan_states_raw, Mapping)
        else {}
    )
    plan_states, plan_trigger_reason = _evaluate_delta_gate_trade_plans(ps_map, eval_prices, prev_plan_states)

    snapshot_meta: Dict[str, Any] = {
        "eval_prices": eval_prices,
        "macro_ts": round(macro_ts, 3),
        "geo_ts": round(geo_ts, 3),
        "latest_fill_ts": round(latest_fill_ts, 3),
        "open_order_ids": open_order_ids,
        "order_attention_flags": order_attention_flags,
        "plan_states": plan_states,
    }

    (
        max_move_pct,
        _max_move_sym,
        max_move_thresh_pct,
        breach_sym,
        breach_move_pct,
        breach_thresh_pct,
        common_count,
        prev_prices,
    ) = _compute_max_spot_move(
        eval_prices,
        prev_meta.get("eval_prices"),
        min_move_pct=eff_min_move_pct,
        high_move_pct=eff_high_move_pct,
        atr_15m_by_sym=atr_15m_by_sym,
    )
    prev_fill_ts = normalize_epoch_sec(prev_meta.get("latest_fill_ts"), default=0.0, allow_iso=True) or 0.0
    has_new_fill = bool(
        latest_fill_ts > 0.0
        and (
            (prev_fill_ts > 0.0 and latest_fill_ts > prev_fill_ts + 0.5)
            or (prev_ts > 0.0 and latest_fill_ts > prev_ts + 0.5)
        )
    )

    clean_focus = parse_symbol_list(focus_symbols) if focus_symbols else []
    escalate_high_effort = bool(
        clean_focus
        or max_move_pct >= eff_high_move_pct
        or has_new_fill
        or geo_shock
        or plan_trigger_reason
    )

    if clean_focus:
        return False, f"focus_symbols({','.join(clean_focus)})", escalate_high_effort, snapshot_meta
    if _parse_bool_flag(ui_turn_pending, False):
        return False, "ui_turn_pending", escalate_high_effort, snapshot_meta
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
    if macro_ts > 0.0 and (
        (prev_macro_ts > 0.0 and macro_ts > prev_macro_ts + 0.5)
        or (prev_macro_ts <= 0.0 and macro_ts > prev_ts + 0.5)
    ):
        return False, "macro_update", escalate_high_effort, snapshot_meta

    prev_geo_ts = normalize_epoch_sec(prev_meta.get("geo_ts"), default=0.0, allow_iso=True) or 0.0
    if geo_ts > 0.0 and (
        (prev_geo_ts > 0.0 and geo_ts > prev_geo_ts + 0.5)
        or (prev_geo_ts <= 0.0 and geo_ts > prev_ts + 0.5)
    ):
        return False, "geo_update", escalate_high_effort, snapshot_meta

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
