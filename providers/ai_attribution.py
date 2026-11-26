"""Pure in-memory AI vs. BOT attribution, model-tier audit, override audit, and Delta-Gate telemetry.

Strictly zero disk I/O (no ``os``, ``glob``, ``pathlib``, ``state_io``, or ``state_stores`` imports).
Workspace loading lives in ``providers/ai_attribution_store.py``.
"""
from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any, Callable, Mapping, Optional, Sequence

from botcore import (
    BUCHAREST,
    clean_order_id,
    clean_scalar_str,
    clean_upper_scalar,
    fmt_signed,
    normalize_side,
    resolve_now_ts,
    safe_float,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
)
from cachepaths import matches_asset_scope, normalize_venue_name
from .ai_survival_guard import (
    _LIMIT_PASSIVE_OFFSET_BUCKETS,
    _collect_canceled_order_ids,
    _extract_intent_exec_price,
    _extract_intent_exec_ts,
    _is_filled_ai_intent,
    _is_terminal_or_stale_unfilled_intent,
    _resolve_live_price_for_symbol,
    build_normalized_filled_ai_intent_row,
    compute_signed_price_edge_pct,
    evaluate_filled_trade_plans,
    extract_fill_execution_slippage,
    extract_limit_passive_offset_bps,
    format_signed_usd,
    match_filled_intent_rows_lifo_in_window,
    resolve_matched_notional_usd,
    resolve_venue_round_trip_fee_bps,
    summarize_execution_quality_bucket,
    summarize_execution_quality_by_venue_and_order_type,
    summarize_limit_passive_offset_fill_rates,
)
from .execution_audit import _extract_audit_fill_metrics
from .intent_state import (
    _extract_intent_id,
    _intent_venue_symbol_side_key,
    collect_known_ai_order_ids,
    extract_intent_order_type,
    extract_intent_side,
    extract_intent_timestamp_s,
    parse_pos_epoch_s,
)
from .intent_telemetry import (
    evaluate_trade_plan_path_status,
    extract_frozen_trade_plan_status,
    resolve_symbol_price_bars,
)
from .nav_drawdown import (
    evaluate_nav_drawdown_metrics,
    format_nav_drawdown_summary_line,
)
from .order_id_context import resolve_order_actor, resolve_order_owner

_GUARD_CODE_RE = re.compile(r"^([A-Za-z0-9_]+)")
_CB_TAG_RE = re.compile(r"^([a-z0-9_]+)")
_LOG_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})")
_ESCALATE_MODEL_RE = re.compile(r"Escalating reasoning effort to\s+([^\s]+)\s+for high-signal cycle")
_QUOTA_SUPPRESS_RE = re.compile(r"Suppressing high-effort escalation to\s+([^\s]+)")
_QUOTA_SOFT_SHIFT_RE = re.compile(r"Quota soft-conserve routing routine cycle to\s+([^\s]+)")
_DELTA_GATE_WAKE_RE = re.compile(r"\[DELTA GATE\]\s+Wake LLM\s+\(([A-Za-z0-9_]+)")
_AUDIT_SKIP_EVENTS = frozenset({"submit_requested", "submit_refused", "cancel_requested", "cancel_accepted", "cancel_rejected", "status_error"})


def _extract_guard_code(raw_entry: Any) -> str:
    txt = clean_scalar_str(raw_entry) or ""
    if not txt:
        return ""
    m = _GUARD_CODE_RE.match(txt)
    return m.group(1).upper() if m else ""


def _extract_cb_tag(raw_reason: Any) -> str:
    txt = (clean_scalar_str(raw_reason) or "").lower()
    if not txt:
        return ""
    m = _CB_TAG_RE.match(txt)
    return m.group(1) if m else txt[:48]


_CONFIDENCE_CAL_BUCKETS: tuple[tuple[str, Optional[float], Optional[float]], ...] = (
    ("lt_0.85", None, 0.85),
    ("0.85_0.89", 0.85, 0.90),
    ("0.90_0.94", 0.90, 0.95),
    ("ge_0.95", 0.95, None),
)


def _summarize_plan_confidence_calibration(plan_row_outcomes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize empirical TARGET_HIT vs INVALIDATED rates by confidence bucket and declared horizon_h."""
    by_bucket: dict[str, dict[str, Any]] = {}
    hor_declared = 0
    hor_hit = 0
    hor_inv = 0
    hor_exp = 0
    for row in plan_row_outcomes:
        if not isinstance(row, Mapping):
            continue
        st = clean_upper_scalar(row.get("status")) or ""
        conf = safe_float(row.get("confidence"), default=None)
        if conf is not None and st in ("TARGET_HIT", "INVALIDATED"):
            for b_key, lo, hi in _CONFIDENCE_CAL_BUCKETS:
                if (lo is None or conf >= lo) and (hi is None or conf < hi):
                    bkt = by_bucket.setdefault(b_key, {"target_hit": 0, "invalidated": 0})
                    if st == "TARGET_HIT":
                        bkt["target_hit"] += 1
                    else:
                        bkt["invalidated"] += 1
                    break
        if safe_pos_float(row.get("horizon_h"), default=None) is not None:
            hor_declared += 1
            h_st = clean_upper_scalar(row.get("within_horizon_status")) or ""
            if h_st == "TARGET_HIT":
                hor_hit += 1
            elif h_st == "INVALIDATED":
                hor_inv += 1
            elif h_st == "EXPIRED":
                hor_exp += 1
    out_buckets: dict[str, dict[str, Any]] = {}
    for b_key, _, _ in _CONFIDENCE_CAL_BUCKETS:
        if (bkt := by_bucket.get(b_key)) is None:
            continue
        tot = bkt["target_hit"] + bkt["invalidated"]
        out_buckets[b_key] = {
            **bkt,
            "resolved": tot,
            "hit_rate_pct": round((bkt["target_hit"] / tot) * 100.0, 1) if tot > 0 else None,
        }
    hor_term = hor_hit + hor_inv + hor_exp
    return {
        "by_confidence_bucket": out_buckets,
        "horizon_declared": hor_declared,
        "within_horizon_target_hit": hor_hit,
        "within_horizon_invalidated": hor_inv,
        "within_horizon_expired": hor_exp,
        "within_horizon_hit_rate_pct": round((hor_hit / hor_term) * 100.0, 1) if hor_term > 0 else None,
    }


def _compute_profit_factor(outcomes: list[dict[str, Any]]) -> Optional[float]:
    gross_win = sum(o["net_usd"] for o in outcomes if o["net_usd"] > 0.0)
    gross_loss = abs(sum(o["net_usd"] for o in outcomes if o["net_usd"] < 0.0))
    if gross_loss <= 1e-9:
        return None
    return round(gross_win / gross_loss, 3)


def _summarize_outcomes_bucket(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    closed = [o for o in outcomes if not str(o.get("kind") or "").endswith("_MTM")]
    open_mtm = [o for o in outcomes if str(o.get("kind") or "").endswith("_MTM")]
    wins = sum(1 for o in outcomes if o["net_edge_pct"] > 0.0)
    losses = sum(1 for o in outcomes if o["net_edge_pct"] < 0.0)
    flat = len(outcomes) - wins - losses
    n = len(outcomes)
    realized_gross = sum(o["gross_usd"] for o in closed)
    realized_fees = sum(o["fee_usd"] for o in closed)
    realized_net = sum(o["net_usd"] for o in closed)
    open_net = sum(o["net_usd"] for o in open_mtm)
    open_fees = sum(o["fee_usd"] for o in open_mtm)
    avg_net_edge = (sum(o["net_edge_pct"] for o in outcomes) / n) if n > 0 else 0.0
    win_rate = ((wins / n) * 100.0) if n > 0 else 0.0
    return {
        "scored": n,
        "closed_trades": len(closed),
        "open_mtm": len(open_mtm),
        "wins": wins,
        "losses": losses,
        "flat": flat,
        "win_rate_pct": round(win_rate, 1),
        "avg_net_edge_pct": round(avg_net_edge, 3),
        "realized_gross_usd": round(realized_gross, 2),
        "realized_net_usd": round(realized_net, 2),
        "open_mtm_net_usd": round(open_net, 2),
        "total_net_usd": round(realized_net + open_net, 2),
        "est_fees_usd": round(realized_fees + open_fees, 2),
        "profit_factor": _compute_profit_factor(outcomes),
    }


def _classify_intent_lifecycle(it: dict[str, Any], canceled_oids: set[str]) -> str:
    st = clean_upper_scalar(it.get("status")) or "UNKNOWN"
    if st == "EXECUTED":
        return "EXECUTED_FILLED" if _is_filled_ai_intent(it, canceled_oids) else "EXECUTED_UNFILLED"
    if st in (
        "OBSERVED_SHADOW",
        "REJECTED_AI_CIRCUIT_BREAKER",
        "SKIPPED_DUPLICATE_OPEN_ORDER",
        "BLOCKED_BY_GUARD",
    ):
        return st
    return "FAILED_OR_OTHER"


def format_execution_quality_parts(eq: Mapping[str, Any]) -> list[str]:
    """Format an execution quality bucket into compact LIMIT/MARKET summary segments."""
    if not isinstance(eq, Mapping):
        return []
    lim_n = safe_nonneg_int(eq.get("limit_filled"), 0)
    mkt_n = safe_nonneg_int(eq.get("market_filled"), 0)
    lim_unf = safe_nonneg_int(eq.get("limit_unfilled"), 0)
    if (lim_n + mkt_n + lim_unf) <= 0:
        return []
    lim_s = safe_float(eq.get("avg_limit_slippage_bps"), default=None)
    arr_s = safe_float(eq.get("avg_arrival_shortfall_bps"), default=None)
    mkt_s = safe_float(eq.get("avg_market_slippage_bps"), default=None)
    fill_rate = safe_float(eq.get("limit_fill_rate_pct"), default=None)
    parts = [
        f"LIMIT(n={lim_n}"
        + (f", fill_rate={fill_rate:.1f}%" if (lim_unf > 0 and fill_rate is not None) else "")
        + (f", slip={fmt_signed(lim_s, 1)}bps" if lim_s is not None else "")
        + (f", arrival_sf={fmt_signed(arr_s, 1)}bps" if arr_s is not None else "")
        + ")"
    ]
    if mkt_n > 0:
        parts.append(
            f"MARKET(n={mkt_n}"
            + (f", slip={fmt_signed(mkt_s, 1)}bps" if mkt_s is not None else "")
            + ")"
        )
    return parts


def _build_closed_round_trip_outcome(
    kind: str,
    entry_row: Mapping[str, Any],
    exit_row: Mapping[str, Any],
    *,
    plan_outcome: str,
    cutoff_ts: float = 0.0,
) -> dict[str, Any]:
    gross_pct = compute_signed_price_edge_pct(entry_row["side"], entry_row["px"], exit_row["px"])
    rt_bps = entry_row["leg_fee_bps"] + exit_row["leg_fee_bps"]
    net_pct = gross_pct - (rt_bps / 100.0)
    entry_usd = safe_pos_float(entry_row.get("usd"), default=0.0) or 0.0
    exit_usd = safe_pos_float(exit_row.get("usd"), default=0.0) or 0.0
    matched_usd = entry_usd if entry_usd > 0.0 else resolve_matched_notional_usd(entry_usd, exit_usd)
    gross_usd = matched_usd * (gross_pct / 100.0)
    fee_usd = matched_usd * (rt_bps / 10000.0)
    entry_ts = safe_float(entry_row.get("ts"), default=0.0) or 0.0
    return {
        "ts": exit_row["ts"],
        "entry_ts": entry_ts,
        "entry_in_window": bool(cutoff_ts <= 0.0 or entry_ts >= cutoff_ts),
        "kind": kind,
        "venue": exit_row["venue"],
        "symbol": exit_row["symbol"],
        "model_used": entry_row["model_used"],
        "matched_usd": matched_usd,
        "gross_edge_pct": gross_pct,
        "net_edge_pct": net_pct,
        "gross_usd": gross_usd,
        "fee_usd": fee_usd,
        "net_usd": gross_usd - fee_usd,
        "has_plan": entry_row["has_plan"],
        "plan_outcome": plan_outcome,
        "override_codes": list(entry_row["override_codes"]),
    }


def _merge_audit_order_entry(dst: dict[str, Any], src: dict[str, Any]) -> None:
    """Monotonically merge two audit order entries without downgrading fill state, qty, cost, fee, or price."""
    if not dst:
        dst.update(src)
        return
    for str_k in ("venue", "symbol", "order_id", "intent_id", "client_order_id", "side"):
        if not dst.get(str_k) and src.get(str_k):
            dst[str_k] = src[str_k]
    if src.get("order_type") == "MARKET":
        dst["order_type"] = "MARKET"
    if src.get("ts", 0.0) > dst.get("ts", 0.0):
        dst["ts"] = src["ts"]
    dst_was_fill = bool(dst.get("is_fill"))
    if src.get("is_fill"):
        dst["is_fill"] = True
    for num_k in ("filled_qty", "cost_usd", "fee_usd"):
        if src.get(num_k, 0.0) > dst.get(num_k, 0.0):
            dst[num_k] = src[num_k]
    if src.get("price", 0.0) > 0.0 and (dst.get("price", 0.0) <= 0.0 or (src.get("is_fill") and not dst_was_fill)):
        dst["price"] = src["price"]


def _evaluate_execution_audit_actors(
    audit_records: list[dict[str, Any]],
    *,
    known_ai_order_ids: set[str],
    cutoff_ts: float,
    eff_now: float,
    scope: str,
    fee_resolver: Optional[Callable[[str, str, str], Optional[float]]] = None,
) -> dict[str, Any]:
    """Deduplicate multi-venue execution_audit JSONL events into filled orders and compare AI vs BOT vs MANUAL."""
    orders: dict[tuple[str, str, str], dict[str, Any]] = {}
    intent_to_key: dict[str, tuple[str, str, str]] = {}

    for rec in audit_records:
        if not isinstance(rec, dict):
            continue
        ev = (clean_scalar_str(rec.get("event")) or "").lower()
        if ev in _AUDIT_SKIP_EVENTS:
            continue
        ts_s = parse_pos_epoch_s(rec.get("ts")) or 0.0
        if ts_s <= 0.0 or ts_s > eff_now + 60.0:
            continue
        ven = normalize_venue_name(clean_scalar_str(rec.get("venue")) or "")
        sym = clean_scalar_str(rec.get("symbol")) or ""
        if not sym:
            continue
        if scope and scope != "all" and not matches_asset_scope(sym, ven, scope):
            continue
        oid = clean_order_id(rec.get("order_id")) or ""
        iid = _extract_intent_id(rec) or ""
        sym_lc = sym.lower()
        key: Optional[tuple[str, str, str]] = None
        if oid:
            key = (ven, sym_lc, oid)
            if iid:
                prev_key = intent_to_key.get(iid)
                if prev_key is not None and prev_key != key and prev_key in orders:
                    merged = orders.pop(prev_key)
                    _merge_audit_order_entry(orders.setdefault(key, {}), merged)
                intent_to_key[iid] = key
        elif iid:
            key = intent_to_key.get(iid) or (ven, sym_lc, f"intent:{iid}")
            intent_to_key[iid] = key
        if key is None:
            continue

        entry = orders.setdefault(
            key,
            {
                "venue": ven,
                "symbol": sym,
                "order_id": oid,
                "intent_id": iid,
                "client_order_id": "",
                "side": "",
                "order_type": "LIMIT",
                "ts": ts_s,
                "filled_qty": 0.0,
                "cost_usd": 0.0,
                "price": 0.0,
                "fee_usd": 0.0,
                "is_fill": False,
            },
        )
        if oid and not entry["order_id"]:
            entry["order_id"] = oid
        if cid := clean_order_id(rec.get("client_order_id")):
            entry["client_order_id"] = cid
        if sd := normalize_side(rec.get("side")):
            entry["side"] = sd
        if rec.get("market") is True:
            entry["order_type"] = "MARKET"
        if ts_s > entry["ts"]:
            entry["ts"] = ts_s

        _, _, _, f_q, _, cost_f, f_px, is_fill = _extract_audit_fill_metrics(rec, fallback_ref_price=False)
        if f_px is not None and f_px > 0.0 and entry["price"] <= 0.0:
            entry["price"] = f_px
        if is_fill:
            entry["is_fill"] = True
            prev_fq = entry["filled_qty"]
            if f_q is not None and f_q > prev_fq:
                entry["filled_qty"] = f_q
            if cost_f is not None and cost_f > entry["cost_usd"]:
                entry["cost_usd"] = cost_f
            if f_px is not None and f_px > 0.0 and (
                entry["price"] <= 0.0 or (f_q is not None and f_q >= prev_fq - 1e-12)
            ):
                entry["price"] = f_px
            if (fee_f := safe_nonneg_float(rec.get("fee"), default=0.0)) > entry["fee_usd"]:
                entry["fee_usd"] = fee_f

    filled_entries: list[dict[str, Any]] = []
    for entry in orders.values():
        if not entry["is_fill"] or entry["side"] not in ("BUY", "SELL"):
            continue
        px = entry["price"]
        qty = entry["filled_qty"]
        cost = entry["cost_usd"]
        if cost <= 0.0 and qty > 0.0 and px > 0.0:
            cost = qty * px
        if px <= 0.0 and cost > 0.0 and qty > 0.0:
            px = cost / qty
        if px <= 0.0 or cost <= 0.0:
            continue
        oid = entry["order_id"]
        cid = entry["client_order_id"]
        owner = resolve_order_owner(cid) if cid else ""
        is_ai = (
            owner == "ai_executor"
            or resolve_order_actor(
                {"orderId": oid, "clientOrderId": cid},
                known_ai_order_ids,
                venue=entry["venue"],
                symbol=entry["symbol"],
            )
            == "AI"
        )
        if is_ai:
            actor = "AI"
            owner = "ai_executor"
        elif owner == "manual":
            actor = "MANUAL"
        elif owner:
            actor = "BOT"
        else:
            actor = "OTHER"
            owner = "unattributed"
        live_rt = fee_resolver(entry["venue"], entry["symbol"], entry["order_type"]) if callable(fee_resolver) else None
        leg_fee_bps = resolve_venue_round_trip_fee_bps(
            entry["venue"],
            entry["symbol"],
            order_type=entry["order_type"],
            round_trip_fee_bps=live_rt,
        ) / 2.0
        est_fee_usd = entry["fee_usd"] if entry["fee_usd"] > 0.0 else cost * (leg_fee_bps / 10000.0)
        filled_entries.append({
            **entry,
            "price": px,
            "px": px,
            "qty": qty if qty > 0.0 else (cost / px),
            "usd": cost,
            "cost_usd": cost,
            "actor": actor,
            "owner": owner,
            "leg_fee_bps": leg_fee_bps,
            "est_fee_usd": est_fee_usd,
        })

    filled_entries.sort(key=lambda r: r["ts"])

    def _aggregate(group_field: str) -> dict[str, dict[str, Any]]:
        buckets: dict[str, dict[str, Any]] = {}
        lifo_inputs: list[dict[str, Any]] = []

        def _ensure_bucket(g_name: str) -> dict[str, Any]:
            return buckets.setdefault(
                g_name,
                {
                    "filled_orders": 0,
                    "buys": 0,
                    "sells": 0,
                    "buy_notional_usd": 0.0,
                    "sell_notional_usd": 0.0,
                    "est_fees_usd": 0.0,
                    "matched_round_trips": 0,
                    "realized_net_usd": 0.0,
                },
            )

        for row in filled_entries:
            g_name = str(row[group_field])
            if row["ts"] >= cutoff_ts:
                b = _ensure_bucket(g_name)
                b["filled_orders"] += 1
                b["est_fees_usd"] += row["est_fee_usd"]
                if row["side"] == "BUY":
                    b["buys"] += 1
                    b["buy_notional_usd"] += row["cost_usd"]
                else:
                    b["sells"] += 1
                    b["sell_notional_usd"] += row["cost_usd"]

            v_k, base_k, q_k, _ = _intent_venue_symbol_side_key(row["venue"], row["symbol"], row["side"])
            lifo_inputs.append({**row, "inst_key": (g_name, v_k, base_k, q_k), "_group": g_name})

        # Reuse the canonical full-history LIFO matcher + post-match window filter so pre-window entries
        # never vanish off the stack and corrupt in-window realized PnL.
        closed_pairs, _, _ = match_filled_intent_rows_lifo_in_window(lifo_inputs, cutoff_ts=cutoff_ts)
        for _kind, entry_row, exit_row in closed_pairs:
            b = _ensure_bucket(entry_row["_group"])
            gross_pct = compute_signed_price_edge_pct(entry_row["side"], entry_row["px"], exit_row["px"])
            rt_bps = entry_row["leg_fee_bps"] + exit_row["leg_fee_bps"]
            entry_usd = safe_pos_float(entry_row.get("usd"), default=0.0) or 0.0
            exit_usd = safe_pos_float(exit_row.get("usd"), default=0.0) or 0.0
            matched_usd = entry_usd if entry_usd > 0.0 else resolve_matched_notional_usd(entry_usd, exit_usd)
            b["matched_round_trips"] += 1
            b["realized_net_usd"] += matched_usd * ((gross_pct - rt_bps / 100.0) / 100.0)

        out: dict[str, dict[str, Any]] = {}
        for g_name, b in sorted(buckets.items()):
            out[g_name] = {
                "filled_orders": b["filled_orders"],
                "buys": b["buys"],
                "sells": b["sells"],
                "buy_notional_usd": round(b["buy_notional_usd"], 2),
                "sell_notional_usd": round(b["sell_notional_usd"], 2),
                "est_fees_usd": round(b["est_fees_usd"], 2),
                "matched_round_trips": b["matched_round_trips"],
                "realized_net_usd": round(b["realized_net_usd"], 2),
            }
        return out

    window_filled_count = sum(1 for row in filled_entries if row["ts"] >= cutoff_ts)
    return {
        "total_filled_orders": window_filled_count,
        "by_actor": _aggregate("actor"),
        "by_owner": _aggregate("owner"),
    }


def summarize_delta_gate_logs(
    log_lines: list[str],
    *,
    cutoff_ts: float = 0.0,
    eval_state: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Summarize Delta-Gate LLM call savings, quota routing shifts, and high-effort escalations."""
    skips = 0
    llm_cycles = 0
    escalations = 0
    quota_suppressions = 0
    quota_soft_shifts = 0
    escalated_models: dict[str, int] = {}
    soft_shift_models: dict[str, int] = {}
    wake_reasons: dict[str, int] = {}

    for line in log_lines:
        if not isinstance(line, str) or not line:
            continue
        if cutoff_ts > 0.0:
            m_ts = _LOG_TS_RE.search(line)
            if m_ts:
                try:
                    dt = datetime.strptime(m_ts.group(1).replace("T", " "), "%Y-%m-%d %H:%M:%S").replace(
                        tzinfo=BUCHAREST
                    )
                    if dt.timestamp() < cutoff_ts:
                        continue
                except ValueError:
                    pass
        if "[DELTA GATE] Delta-Gate HOLD carry-forward" in line:
            skips += 1
        elif m_wake := _DELTA_GATE_WAKE_RE.search(line):
            w_tag = m_wake.group(1).strip().lower()
            if w_tag:
                wake_reasons[w_tag] = wake_reasons.get(w_tag, 0) + 1
        elif "Autonomous AI Supervisor cycle complete:" in line:
            llm_cycles += 1
        elif m_esc := _ESCALATE_MODEL_RE.search(line):
            escalations += 1
            mdl = m_esc.group(1).strip()
            if mdl:
                escalated_models[mdl] = escalated_models.get(mdl, 0) + 1
        elif _QUOTA_SUPPRESS_RE.search(line):
            quota_suppressions += 1
        elif m_shift := _QUOTA_SOFT_SHIFT_RE.search(line):
            quota_soft_shifts += 1
            mdl = m_shift.group(1).strip()
            if mdl:
                soft_shift_models[mdl] = soft_shift_models.get(mdl, 0) + 1

    total_cycles = skips + llm_cycles
    saved_pct = round((skips / total_cycles) * 100.0, 1) if total_cycles > 0 else 0.0
    ev_map = eval_state if isinstance(eval_state, Mapping) else {}
    return {
        "delta_gate_skips": skips,
        "llm_cycles_completed": llm_cycles,
        "total_cycles_observed": total_cycles,
        "calls_saved_pct": saved_pct,
        "high_effort_escalations": escalations,
        "quota_escalation_suppressions": quota_suppressions,
        "quota_soft_conserve_shifts": quota_soft_shifts,
        "escalated_models": escalated_models,
        "soft_shift_models": soft_shift_models,
        "wake_reasons": wake_reasons,
        "last_eval_model": clean_scalar_str(ev_map.get("model_used")) or "",
        "last_eval_ts": round(parse_pos_epoch_s(ev_map.get("ts")) or 0.0, 3),
    }


def evaluate_ai_attribution(
    intents: list[dict[str, Any]],
    *,
    now_ts: Optional[float] = None,
    lookback_days: float = 7.0,
    scope: str = "all",
    live_prices: Optional[Mapping[str, Any]] = None,
    price_bars_by_symbol: Optional[Mapping[str, Sequence[Any]]] = None,
    audit_records: Optional[list[dict[str, Any]]] = None,
    reconciler_log_lines: Optional[list[str]] = None,
    crypto_eval_state: Optional[Mapping[str, Any]] = None,
    nav_history: Optional[list[dict[str, Any]]] = None,
    current_nav_point: Optional[Mapping[str, Any]] = None,
    fee_resolver: Optional[Callable[[str, str, str], Optional[float]]] = None,
) -> dict[str, Any]:
    """Compute standalone multi-venue AI attribution, model-tier audit, override audit, and AI vs BOT comparison."""
    eff_now = resolve_now_ts(now_ts)
    lb_days = safe_pos_float(lookback_days, default=7.0) or 7.0
    eff_scope = (clean_scalar_str(scope) or "all").lower()
    cutoff = eff_now - (lb_days * 86400.0)
    prices_map = live_prices if isinstance(live_prices, Mapping) else {}
    items = intents if isinstance(intents, list) else []

    canceled_oids = _collect_canceled_order_ids(items)
    known_ai_oids = collect_known_ai_order_ids(items)

    status_counts: dict[str, int] = {}
    cb_reason_counts: dict[str, int] = {}
    model_intent_counts: dict[str, dict[str, int]] = {}
    override_intent_counts: dict[str, int] = {}
    unfilled_counts: dict[tuple[str, str], int] = {}
    unfilled_limit_offsets_by_venue: dict[str, list[float]] = {}
    all_unfilled_lim_offs: list[float] = []
    all_filled_rows: list[dict[str, Any]] = []
    filled_rows: list[dict[str, Any]] = []

    for it in items:
        if not isinstance(it, dict):
            continue
        sym = clean_scalar_str(it.get("symbol")) or ""
        ven = normalize_venue_name(clean_scalar_str(it.get("venue")) or "")
        if not sym:
            continue
        if eff_scope != "all" and not matches_asset_scope(sym, ven, eff_scope):
            continue
        _, base_ts = extract_intent_timestamp_s(it)
        exec_ts = _extract_intent_exec_ts(it)
        ref_ts = exec_ts if exec_ts > 0.0 else base_ts
        if ref_ts <= 0.0 or ref_ts > eff_now + 60.0:
            continue

        in_window = ref_ts >= cutoff
        lifecycle = _classify_intent_lifecycle(it, canceled_oids)
        mdl = clean_scalar_str(it.get("model_used")) or "unknown"

        if in_window:
            status_counts[lifecycle] = status_counts.get(lifecycle, 0) + 1
            m_cnt = model_intent_counts.setdefault(mdl, {"intents": 0, "filled": 0})
            m_cnt["intents"] += 1

            if cb_rsn := (
                it.get("circuit_breaker_reason")
                or it.get("scorecard_downgraded_to_shadow")
                or it.get("daily_nav_dd_downgraded_to_shadow")
                or it.get("daily_nav_dd_shadow_warning")
                or (it.get("history")[-1].get("guard_reason") if isinstance(it.get("history"), list) and it.get("history") and isinstance(it.get("history")[-1], dict) and it.get("history")[-1].get("guard_allowed") is False else "")
            ):
                if tag := _extract_cb_tag(cb_rsn):
                    cb_reason_counts[tag] = cb_reason_counts.get(tag, 0) + 1

            if lifecycle == "EXECUTED_UNFILLED":
                if _is_terminal_or_stale_unfilled_intent(it, canceled_oids, now_ts=eff_now, ref_ts=ref_ts):
                    unf_ven = ven or "unknown"
                    unf_ot = "LIMIT" if extract_intent_order_type(it) == "LIMIT" else "MARKET"
                    k_unf = (unf_ven, unf_ot)
                    unfilled_counts[k_unf] = unfilled_counts.get(k_unf, 0) + 1
                    if unf_ot == "LIMIT":
                        unf_off = extract_limit_passive_offset_bps(it, order_type="LIMIT")
                        if unf_off is not None:
                            unfilled_limit_offsets_by_venue.setdefault(unf_ven, []).append(unf_off)
                            all_unfilled_lim_offs.append(unf_off)
                continue

        if lifecycle != "EXECUTED_FILLED":
            continue
        side = extract_intent_side(it)
        if side not in ("BUY", "SELL"):
            continue
        px = _extract_intent_exec_price(it)
        if px <= 0.0:
            continue

        raw_ov = it.get("advisory_guards_overridden")
        ov_codes = sorted({
            code
            for raw_item in (raw_ov if isinstance(raw_ov, list) else [])
            if (code := _extract_guard_code(raw_item))
        })
        norm_row = build_normalized_filled_ai_intent_row(
            it,
            venue=ven,
            symbol=sym,
            side=side,
            ts_s=ref_ts,
            px=px,
            fee_resolver=fee_resolver,
            include_slippage=in_window,
            include_attribution=True,
            model_used=mdl,
            override_codes=ov_codes,
        )
        all_filled_rows.append(norm_row)
        if in_window:
            model_intent_counts[mdl]["filled"] += 1
            for code in ov_codes:
                override_intent_counts[code] = override_intent_counts.get(code, 0) + 1
            filled_rows.append(norm_row)

    closed_pairs, open_buys, open_sells = match_filled_intent_rows_lifo_in_window(
        all_filled_rows, cutoff_ts=cutoff
    )
    outcomes: list[dict[str, Any]] = []
    plan_row_outcomes: list[dict[str, Any]] = []
    plan_status_counts, _, planned_filled = evaluate_filled_trade_plans(
        filled_rows,
        prices_map,
        price_bars_by_symbol=price_bars_by_symbol,
        row_outcomes_out=plan_row_outcomes,
        now_ts=eff_now,
    )

    def _record_closed_plan_outcome(entry_row: dict[str, Any], exit_row: dict[str, Any]) -> str:
        if not entry_row["has_plan"]:
            return ""
        if frozen_st := extract_frozen_trade_plan_status(entry_row):
            return frozen_st
        sym_bars = resolve_symbol_price_bars(
            price_bars_by_symbol, entry_row.get("venue"), entry_row.get("symbol")
        )
        st = evaluate_trade_plan_path_status(
            entry_row["side"],
            exit_row["px"],
            entry_row["invalidation_px"],
            entry_row["target_px"],
            start_ts=safe_float(entry_row.get("ts"), default=0.0) or 0.0,
            end_ts=safe_float(exit_row.get("ts"), default=0.0) or None,
            price_bars=sym_bars,
        )
        return st if st in ("TARGET_HIT", "INVALIDATED") else "CLOSED_EARLY"

    for kind, entry_row, exit_row in closed_pairs:
        plan_st = _record_closed_plan_outcome(entry_row, exit_row)
        outcomes.append(
            _build_closed_round_trip_outcome(kind, entry_row, exit_row, plan_outcome=plan_st, cutoff_ts=cutoff)
        )

    def _evaluate_open_mtm_row(entry_row: dict[str, Any], side: str) -> Optional[dict[str, Any]]:
        spot_px = _resolve_live_price_for_symbol(prices_map, entry_row["venue"], entry_row["symbol"])
        if spot_px is None or spot_px <= 0.0:
            return None
        plan_st = ""
        if entry_row["has_plan"]:
            if frozen_st := extract_frozen_trade_plan_status(entry_row):
                plan_st = frozen_st
            else:
                sym_bars = resolve_symbol_price_bars(
                    price_bars_by_symbol, entry_row.get("venue"), entry_row.get("symbol")
                )
                plan_st = evaluate_trade_plan_path_status(
                    side,
                    spot_px,
                    entry_row["invalidation_px"],
                    entry_row["target_px"],
                    start_ts=safe_float(entry_row.get("ts"), default=0.0) or 0.0,
                    price_bars=sym_bars,
                )
        gross_pct = compute_signed_price_edge_pct(side, entry_row["px"], spot_px)
        rt_bps = entry_row["rt_fee_bps"]
        net_pct = gross_pct - (rt_bps / 100.0)
        gross_usd = entry_row["usd"] * (gross_pct / 100.0)
        fee_usd = entry_row["usd"] * (rt_bps / 10000.0)
        return {
            "ts": entry_row["ts"],
            "entry_ts": entry_row["ts"],
            "entry_in_window": True,
            "kind": f"{side}_MTM",
            "venue": entry_row["venue"],
            "symbol": entry_row["symbol"],
            "model_used": entry_row["model_used"],
            "matched_usd": entry_row["usd"],
            "gross_edge_pct": gross_pct,
            "net_edge_pct": net_pct,
            "gross_usd": gross_usd,
            "fee_usd": fee_usd,
            "net_usd": gross_usd - fee_usd,
            "has_plan": entry_row["has_plan"],
            "plan_outcome": plan_st,
            "override_codes": list(entry_row["override_codes"]),
        }

    for buy_row in open_buys:
        if (mtm_out := _evaluate_open_mtm_row(buy_row, "BUY")) is not None:
            outcomes.append(mtm_out)

    planned_sell_mtm_outcomes: list[dict[str, Any]] = []
    for sell_row in open_sells:
        if not sell_row["has_plan"]:
            continue
        if (sell_mtm := _evaluate_open_mtm_row(sell_row, "SELL")) is not None:
            planned_sell_mtm_outcomes.append(sell_mtm)

    in_window_outcomes = [o for o in outcomes if o.get("entry_in_window", True)]
    legacy_pre_window_outcomes = [o for o in outcomes if not o.get("entry_in_window", True)]

    # Groupings: by venue, by venue:symbol, by model_used, by overridden guard, by trade plan coverage
    by_venue_rows: dict[str, list[dict[str, Any]]] = {}
    by_symbol_rows: dict[str, list[dict[str, Any]]] = {}
    by_model_rows: dict[str, list[dict[str, Any]]] = {}
    by_override_rows: dict[str, list[dict[str, Any]]] = {}
    for out in outcomes:
        by_venue_rows.setdefault(out["venue"] or "unknown", []).append(out)
        by_symbol_rows.setdefault(f"{out['venue']}:{out['symbol']}", []).append(out)
        mdl_bucket = (
            "legacy_pre_window"
            if (not out.get("entry_in_window", True) and out["model_used"] == "unknown")
            else out["model_used"]
        )
        by_model_rows.setdefault(mdl_bucket, []).append(out)
        for code in out["override_codes"]:
            by_override_rows.setdefault(code, []).append(out)

    by_venue = {v: _summarize_outcomes_bucket(rows) for v, rows in sorted(by_venue_rows.items())}
    by_symbol = {k: _summarize_outcomes_bucket(rows) for k, rows in sorted(by_symbol_rows.items())}

    by_model: dict[str, dict[str, Any]] = {}
    all_models = sorted(set(model_intent_counts.keys()) | set(by_model_rows.keys()))
    for mdl in all_models:
        cnts = model_intent_counts.get(mdl, {"intents": 0, "filled": 0})
        by_model[mdl] = {
            "intents": cnts["intents"],
            "filled": cnts["filled"],
            **_summarize_outcomes_bucket(by_model_rows.get(mdl, [])),
        }

    override_audit: dict[str, dict[str, Any]] = {}
    all_ov_codes = sorted(set(override_intent_counts.keys()) | set(by_override_rows.keys()))
    for code in all_ov_codes:
        ov_outcomes = by_override_rows.get(code, [])
        ai_right = sum(1 for o in ov_outcomes if o["net_edge_pct"] > 0.0)
        guard_right = sum(1 for o in ov_outcomes if o["net_edge_pct"] < 0.0)
        n_ov = len(ov_outcomes)
        override_audit[code] = {
            "overrides_filled": override_intent_counts.get(code, 0),
            "scored": n_ov,
            "ai_right": ai_right,
            "guard_right": guard_right,
            "ai_win_rate_pct": round((ai_right / n_ov) * 100.0, 1) if n_ov > 0 else 0.0,
            "avg_net_edge_pct": round(sum(o["net_edge_pct"] for o in ov_outcomes) / n_ov, 3) if n_ov > 0 else 0.0,
            "net_usd": round(sum(o["net_usd"] for o in ov_outcomes), 2),
        }

    unplanned_filled = len(filled_rows) - planned_filled
    planned_outcomes = [o for o in outcomes if o["has_plan"]] + planned_sell_mtm_outcomes
    unplanned_outcomes = [o for o in outcomes if not o["has_plan"] and o.get("entry_in_window", True)]
    legacy_unplanned_outcomes = [o for o in outcomes if not o["has_plan"] and not o.get("entry_in_window", True)]
    trade_plan_calibration = {
        "planned_filled": planned_filled,
        "unplanned_filled": unplanned_filled,
        "plan_coverage_pct": round((planned_filled / len(filled_rows)) * 100.0, 1) if filled_rows else 0.0,
        "status_counts": plan_status_counts,
        "planned_performance": _summarize_outcomes_bucket(planned_outcomes),
        "unplanned_performance": _summarize_outcomes_bucket(unplanned_outcomes),
        "legacy_pre_window_performance": _summarize_outcomes_bucket(legacy_unplanned_outcomes),
        "confidence_calibration": _summarize_plan_confidence_calibration(plan_row_outcomes),
    }

    actor_comparison = _evaluate_execution_audit_actors(
        audit_records if isinstance(audit_records, list) else [],
        known_ai_order_ids=known_ai_oids,
        cutoff_ts=cutoff,
        eff_now=eff_now,
        scope=eff_scope,
        fee_resolver=fee_resolver,
    )
    eq_global = summarize_execution_quality_bucket(filled_rows)
    eq_by_venue, eq_by_vot = summarize_execution_quality_by_venue_and_order_type(
        filled_rows,
        unfilled_counts=unfilled_counts,
        unfilled_limit_offsets_by_venue=unfilled_limit_offsets_by_venue,
    )
    all_filled_lim_offs = [
        off_v for r in filled_rows
        if (clean_upper_scalar(r.get("order_type")) or "") == "LIMIT"
        and (off_v := safe_float(r.get("limit_passive_offset_bps"), default=None)) is not None
    ]
    lim_offset_fill_rates = summarize_limit_passive_offset_fill_rates(all_filled_lim_offs, all_unfilled_lim_offs)
    lim_unf_total = sum(v for (_, ot_k), v in unfilled_counts.items() if ot_k == "LIMIT")
    lim_sub_total = eq_global["limit_filled"] + lim_unf_total
    execution_quality = {
        **eq_global,
        "limit_unfilled": lim_unf_total,
        "limit_fill_rate_pct": round((eq_global["limit_filled"] / lim_sub_total) * 100.0, 1) if lim_sub_total > 0 else None,
        "by_venue": eq_by_venue,
        "by_venue_order_type": eq_by_vot,
        "limit_offset_fill_rates": lim_offset_fill_rates,
    }
    delta_gate = summarize_delta_gate_logs(
        reconciler_log_lines if isinstance(reconciler_log_lines, list) else [],
        cutoff_ts=cutoff,
        eval_state=crypto_eval_state,
    )
    nav_dd = evaluate_nav_drawdown_metrics(
        nav_history if isinstance(nav_history, list) else [],
        now_ts=eff_now,
        current_point=current_nav_point,
    )

    return {
        "generated_at_ts": round(eff_now, 3),
        "lookback_days": lb_days,
        "scope": eff_scope,
        "ai_intents_total": sum(status_counts.values()),
        "ai_filled_total": len(filled_rows),
        "ai_status_counts": status_counts,
        "ai_circuit_breaker_triggers": cb_reason_counts,
        "ai_summary": _summarize_outcomes_bucket(outcomes),
        "ai_summary_in_window": _summarize_outcomes_bucket(in_window_outcomes),
        "ai_summary_legacy_pre_window": _summarize_outcomes_bucket(legacy_pre_window_outcomes),
        "ai_by_venue": by_venue,
        "ai_by_symbol": by_symbol,
        "ai_by_model": by_model,
        "advisory_guard_override_audit": override_audit,
        "trade_plan_calibration": trade_plan_calibration,
        "execution_quality": execution_quality,
        "execution_audit_ai_vs_bot": actor_comparison,
        "delta_gate_efficiency": delta_gate,
        "unified_nav_drawdown": nav_dd,
    }


def _fmt_usd(val: Any) -> str:
    return format_signed_usd(val, decimals=2)


def _fmt_pf(pf: Any) -> str:
    v = safe_pos_float(pf, default=None)
    return f"{v:.2f}" if v is not None else "n/a"


def format_ai_attribution_report(report: Mapping[str, Any]) -> str:
    """Format an attribution payload into a compact, readable ASCII summary report."""
    if not isinstance(report, Mapping):
        return ""
    lb_days = safe_pos_float(report.get("lookback_days"), default=7.0) or 7.0
    scope = clean_scalar_str(report.get("scope")) or "all"
    ai_sum = report.get("ai_summary") if isinstance(report.get("ai_summary"), Mapping) else {}
    lines: list[str] = [
        f"=== MPTrade AI Attribution & Audit Report ({lb_days:g}d | scope={scope}) ===",
        (
            f"AI Intents: total={safe_nonneg_int(report.get('ai_intents_total'), 0)}, "
            f"filled={safe_nonneg_int(report.get('ai_filled_total'), 0)}, "
            f"scored={safe_nonneg_int(ai_sum.get('scored'), 0)} "
            f"({safe_nonneg_int(ai_sum.get('closed_trades'), 0)} closed / {safe_nonneg_int(ai_sum.get('open_mtm'), 0)} open MTM)"
        ),
        (
            f"AI Net P&L: realized={_fmt_usd(ai_sum.get('realized_net_usd'))} "
            f"(gross={_fmt_usd(ai_sum.get('realized_gross_usd'))}), "
            f"open_mtm={_fmt_usd(ai_sum.get('open_mtm_net_usd'))}, "
            f"total={_fmt_usd(ai_sum.get('total_net_usd'))}, "
            f"est_fees=${safe_nonneg_float(ai_sum.get('est_fees_usd'), 0.0):.2f}, "
            f"WR={safe_nonneg_float(ai_sum.get('win_rate_pct'), 0.0):.1f}%, "
            f"avg_edge={fmt_signed(safe_float(ai_sum.get('avg_net_edge_pct'), 0.0) or 0.0, 2)}%, "
            f"PF={_fmt_pf(ai_sum.get('profit_factor'))}"
        ),
    ]

    in_win_sum = report.get("ai_summary_in_window") if isinstance(report.get("ai_summary_in_window"), Mapping) else {}
    legacy_sum = (
        report.get("ai_summary_legacy_pre_window")
        if isinstance(report.get("ai_summary_legacy_pre_window"), Mapping)
        else {}
    )
    if safe_nonneg_int(legacy_sum.get("scored"), 0) > 0:
        lines.append(
            f"AI Cohort Split: in_window={safe_nonneg_int(in_win_sum.get('scored'), 0)} "
            f"(net={_fmt_usd(in_win_sum.get('total_net_usd'))}, "
            f"WR={safe_nonneg_float(in_win_sum.get('win_rate_pct'), 0.0):.1f}%, "
            f"PF={_fmt_pf(in_win_sum.get('profit_factor'))}) | "
            f"legacy_pre_window_exits={safe_nonneg_int(legacy_sum.get('scored'), 0)} "
            f"(realized={_fmt_usd(legacy_sum.get('realized_net_usd'))}, "
            f"WR={safe_nonneg_float(legacy_sum.get('win_rate_pct'), 0.0):.1f}%)"
        )

    if st_counts := (report.get("ai_status_counts") if isinstance(report.get("ai_status_counts"), Mapping) else {}):
        st_str = ", ".join(f"{k}={v}" for k, v in sorted(st_counts.items()))
        lines.append(f"Lifecycle: {st_str}")

    if cb_tr := (report.get("ai_circuit_breaker_triggers") if isinstance(report.get("ai_circuit_breaker_triggers"), Mapping) else {}):
        cb_str = ", ".join(f"{k}={v}" for k, v in sorted(cb_tr.items()))
        lines.append(f"AI Guard Triggers: {cb_str}")

    by_model = report.get("ai_by_model") if isinstance(report.get("ai_by_model"), Mapping) else {}
    if by_model:
        lines.append("")
        lines.append("-- By model_used --")
        lines.append(f"  {'Model':<28} {'Int':>4} {'Fil':>4} {'Scr':>4} {'WR%':>6} {'Avg%':>7} {'NetUSD':>10} {'PF':>6}")
        for mdl, m in sorted(by_model.items()):
            lines.append(
                f"  {mdl:<28} {safe_nonneg_int(m.get('intents'), 0):>4} "
                f"{safe_nonneg_int(m.get('filled'), 0):>4} {safe_nonneg_int(m.get('scored'), 0):>4} "
                f"{safe_nonneg_float(m.get('win_rate_pct'), 0.0):>5.1f}% "
                f"{fmt_signed(safe_float(m.get('avg_net_edge_pct'), 0.0) or 0.0, 2):>6}% "
                f"{_fmt_usd(m.get('total_net_usd')):>10} {_fmt_pf(m.get('profit_factor')):>6}"
            )

    by_sym = report.get("ai_by_symbol") if isinstance(report.get("ai_by_symbol"), Mapping) else {}
    if by_sym:
        lines.append("")
        lines.append("-- AI By Venue:Symbol --")
        lines.append(f"  {'Venue:Symbol':<24} {'Cls':>4} {'MTM':>4} {'WR%':>6} {'Avg%':>7} {'RealUSD':>10} {'TotUSD':>10} {'Fees':>7}")
        for vs, m in sorted(by_sym.items()):
            lines.append(
                f"  {vs:<24} {safe_nonneg_int(m.get('closed_trades'), 0):>4} "
                f"{safe_nonneg_int(m.get('open_mtm'), 0):>4} "
                f"{safe_nonneg_float(m.get('win_rate_pct'), 0.0):>5.1f}% "
                f"{fmt_signed(safe_float(m.get('avg_net_edge_pct'), 0.0) or 0.0, 2):>6}% "
                f"{_fmt_usd(m.get('realized_net_usd')):>10} {_fmt_usd(m.get('total_net_usd')):>10} "
                f"${safe_nonneg_float(m.get('est_fees_usd'), 0.0):>6.2f}"
            )

    ov_audit = report.get("advisory_guard_override_audit") if isinstance(report.get("advisory_guard_override_audit"), Mapping) else {}
    if ov_audit:
        lines.append("")
        lines.append("-- Advisory Guard Override Audit (AI_RIGHT vs GUARD_RIGHT) --")
        lines.append(f"  {'Guard':<14} {'Filled':>6} {'Scored':>6} {'AI_OK':>6} {'G_OK':>6} {'AI_WR%':>7} {'Avg%':>7} {'NetUSD':>10}")
        for g_code, g in sorted(ov_audit.items()):
            lines.append(
                f"  {g_code:<14} {safe_nonneg_int(g.get('overrides_filled'), 0):>6} "
                f"{safe_nonneg_int(g.get('scored'), 0):>6} {safe_nonneg_int(g.get('ai_right'), 0):>6} "
                f"{safe_nonneg_int(g.get('guard_right'), 0):>6} {safe_nonneg_float(g.get('ai_win_rate_pct'), 0.0):>6.1f}% "
                f"{fmt_signed(safe_float(g.get('avg_net_edge_pct'), 0.0) or 0.0, 2):>6}% {_fmt_usd(g.get('net_usd')):>10}"
            )

    plans = report.get("trade_plan_calibration") if isinstance(report.get("trade_plan_calibration"), Mapping) else {}
    if plans:
        psc = plans.get("status_counts") if isinstance(plans.get("status_counts"), Mapping) else {}
        p_perf = plans.get("planned_performance") if isinstance(plans.get("planned_performance"), Mapping) else {}
        u_perf = plans.get("unplanned_performance") if isinstance(plans.get("unplanned_performance"), Mapping) else {}
        lines.append("")
        lines.append(
            f"-- 4A Trade Plan Calibration: coverage={safe_nonneg_float(plans.get('plan_coverage_pct'), 0.0):.1f}% "
            f"({safe_nonneg_int(plans.get('planned_filled'), 0)} planned / {safe_nonneg_int(plans.get('unplanned_filled'), 0)} unplanned) --"
        )
        lines.append(
            f"  Outcomes: TARGET_HIT={safe_nonneg_int(psc.get('TARGET_HIT'), 0)}, "
            f"INVALIDATED={safe_nonneg_int(psc.get('INVALIDATED'), 0)}, "
            f"CLOSED_EARLY={safe_nonneg_int(psc.get('CLOSED_EARLY'), 0)}, "
            f"ACTIVE={safe_nonneg_int(psc.get('ACTIVE'), 0)}"
        )
        leg_perf = (
            plans.get("legacy_pre_window_performance")
            if isinstance(plans.get("legacy_pre_window_performance"), Mapping)
            else {}
        )
        leg_sfx = (
            f" | Legacy pre-window avg={fmt_signed(safe_float(leg_perf.get('avg_net_edge_pct'), 0.0) or 0.0, 2)}% "
            f"({_fmt_usd(leg_perf.get('total_net_usd'))}, n={safe_nonneg_int(leg_perf.get('scored'), 0)})"
            if safe_nonneg_int(leg_perf.get("scored"), 0) > 0
            else ""
        )
        lines.append(
            f"  Planned avg={fmt_signed(safe_float(p_perf.get('avg_net_edge_pct'), 0.0) or 0.0, 2)}% ({_fmt_usd(p_perf.get('total_net_usd'))}) | "
            f"Unplanned avg={fmt_signed(safe_float(u_perf.get('avg_net_edge_pct'), 0.0) or 0.0, 2)}% ({_fmt_usd(u_perf.get('total_net_usd'))})"
            f"{leg_sfx}"
        )
        conf_cal = plans.get("confidence_calibration") if isinstance(plans.get("confidence_calibration"), Mapping) else {}
        cal_parts: list[str] = []
        by_cb = conf_cal.get("by_confidence_bucket") if isinstance(conf_cal.get("by_confidence_bucket"), Mapping) else {}
        for b_k, _, _ in _CONFIDENCE_CAL_BUCKETS:
            if isinstance(b_row := by_cb.get(b_k), Mapping) and safe_nonneg_int(b_row.get("resolved"), 0) > 0:
                hr = safe_float(b_row.get("hit_rate_pct"), default=None)
                cal_parts.append(
                    f"{b_k}:{hr:.0f}%[{safe_nonneg_int(b_row.get('target_hit'), 0)}/{safe_nonneg_int(b_row.get('resolved'), 0)}]"
                    if hr is not None
                    else f"{b_k}:n/a"
                )
        hor_decl = safe_nonneg_int(conf_cal.get("horizon_declared"), 0)
        if hor_decl > 0:
            h_hr = safe_float(conf_cal.get("within_horizon_hit_rate_pct"), default=None)
            h_hit = safe_nonneg_int(conf_cal.get("within_horizon_target_hit"), 0)
            h_inv = safe_nonneg_int(conf_cal.get("within_horizon_invalidated"), 0)
            h_exp = safe_nonneg_int(conf_cal.get("within_horizon_expired"), 0)
            h_rate_s = f"{h_hr:.0f}%" if h_hr is not None else "active"
            cal_parts.append(f"horizon(n={hor_decl}, in_hor={h_rate_s}[{h_hit}T/{h_inv}I/{h_exp}X])")
        if cal_parts:
            lines.append(f"  Confidence/Horizon Calibration: {', '.join(cal_parts)}")

    exec_cmp = report.get("execution_audit_ai_vs_bot") if isinstance(report.get("execution_audit_ai_vs_bot"), Mapping) else {}
    by_actor = exec_cmp.get("by_actor") if isinstance(exec_cmp.get("by_actor"), Mapping) else {}
    if by_actor:
        lines.append("")
        lines.append("-- Multi-Venue Execution Audit (AI vs BOT vs MANUAL) --")
        lines.append(f"  {'Actor':<12} {'Fills':>5} {'B/S':>7} {'BuyUSD':>10} {'SellUSD':>10} {'RT':>4} {'RealNet':>10} {'Fees':>7}")
        for actor, a in sorted(by_actor.items()):
            bs = f"{safe_nonneg_int(a.get('buys'), 0)}/{safe_nonneg_int(a.get('sells'), 0)}"
            lines.append(
                f"  {actor:<12} {safe_nonneg_int(a.get('filled_orders'), 0):>5} {bs:>7} "
                f"${safe_nonneg_float(a.get('buy_notional_usd'), 0.0):>9.0f} "
                f"${safe_nonneg_float(a.get('sell_notional_usd'), 0.0):>9.0f} "
                f"{safe_nonneg_int(a.get('matched_round_trips'), 0):>4} "
                f"{_fmt_usd(a.get('realized_net_usd')):>10} "
                f"${safe_nonneg_float(a.get('est_fees_usd'), 0.0):>6.2f}"
            )

    dg = report.get("delta_gate_efficiency") if isinstance(report.get("delta_gate_efficiency"), Mapping) else {}
    if dg and (safe_nonneg_int(dg.get("total_cycles_observed"), 0) > 0 or dg.get("last_eval_model")):
        q_sup = safe_nonneg_int(dg.get("quota_escalation_suppressions"), 0)
        q_shf = safe_nonneg_int(dg.get("quota_soft_conserve_shifts"), 0)
        q_sfx = ""
        if q_sup > 0:
            q_sfx += f", quota_suppressed={q_sup}"
        if q_shf > 0:
            q_sfx += f", soft_shifts={q_shf}"
        lines.append("")
        lines.append(
            f"-- Delta-Gate Efficiency: skips={safe_nonneg_int(dg.get('delta_gate_skips'), 0)}, "
            f"llm_cycles={safe_nonneg_int(dg.get('llm_cycles_completed'), 0)}, "
            f"saved={safe_nonneg_float(dg.get('calls_saved_pct'), 0.0):.1f}%, "
            f"high_escalations={safe_nonneg_int(dg.get('high_effort_escalations'), 0)}"
            f"{q_sfx}, "
            f"last_model={dg.get('last_eval_model') or 'n/a'} --"
        )
        if wake_rs := (dg.get("wake_reasons") if isinstance(dg.get("wake_reasons"), Mapping) else {}):
            top_wakes = ", ".join(
                f"{k}={v}" for k, v in sorted(wake_rs.items(), key=lambda kv: (-safe_nonneg_int(kv[1], 0), str(kv[0])))[:6]
            )
            if top_wakes:
                lines.append(f"  Wake reasons: {top_wakes}")

    eq = report.get("execution_quality") if isinstance(report.get("execution_quality"), Mapping) else {}
    if eq_parts := format_execution_quality_parts(eq):
        lines.append(f"Execution Quality: {', '.join(eq_parts)}")
    lim_off = eq.get("limit_offset_fill_rates") if isinstance(eq.get("limit_offset_fill_rates"), Mapping) else {}
    if lim_off:
        off_segs: list[str] = []
        for b_key, default_lbl, _, _ in _LIMIT_PASSIVE_OFFSET_BUCKETS:
            bkt = lim_off.get(b_key)
            if not isinstance(bkt, Mapping):
                continue
            b_t = safe_nonneg_int(bkt.get("total"), 0)
            if b_t <= 0:
                continue
            b_f = safe_nonneg_int(bkt.get("filled"), 0)
            b_fr = safe_float(bkt.get("fill_rate_pct"), default=None)
            lbl = clean_scalar_str(bkt.get("label")) or default_lbl
            off_segs.append(f"{lbl}: {b_fr:.1f}% ({b_f}/{b_t})" if b_fr is not None else f"{lbl}: n/a ({b_f}/{b_t})")
        if off_segs:
            lines.append(f"  LIMIT Offset Fill Rate: {', '.join(off_segs)}")
    by_vot = eq.get("by_venue_order_type") if isinstance(eq.get("by_venue_order_type"), Mapping) else {}
    if by_vot:
        lines.append("  -- Execution TCA by Venue:OrderType --")
        lines.append(f"  {'Venue:Type':<20} {'Fil':>4} {'Unf':>4} {'FR%':>6} {'SlipBps':>8} {'ArrBps':>8} {'UnfOff':>8} {'FeeBps':>7} {'Notional':>10}")
        for vot_k, v_row in sorted(by_vot.items()):
            if not isinstance(v_row, Mapping):
                continue
            fr_v = safe_float(v_row.get("fill_rate_pct"), default=None)
            sl_v = safe_float(v_row.get("avg_slippage_bps"), default=None)
            ar_v = safe_float(v_row.get("avg_arrival_shortfall_bps"), default=None)
            uo_v = safe_float(v_row.get("med_unfilled_offset_bps"), default=None)
            fe_v = safe_float(v_row.get("avg_leg_fee_bps"), default=None)
            fr_s = f"{fr_v:>5.1f}%" if fr_v is not None else f"{'n/a':>6}"
            sl_s = f"{fmt_signed(sl_v, 1):>8}" if sl_v is not None else f"{'n/a':>8}"
            ar_s = f"{fmt_signed(ar_v, 1):>8}" if ar_v is not None else f"{'n/a':>8}"
            uo_s = f"{fmt_signed(uo_v, 1):>8}" if uo_v is not None else f"{'n/a':>8}"
            fe_s = f"{fe_v:>7.1f}" if fe_v is not None else f"{'n/a':>7}"
            lines.append(
                f"  {vot_k:<20} {safe_nonneg_int(v_row.get('filled'), 0):>4} "
                f"{safe_nonneg_int(v_row.get('unfilled'), 0):>4} {fr_s} {sl_s} {ar_s} {uo_s} {fe_s} "
                f"${safe_nonneg_float(v_row.get('notional_usd'), 0.0):>9.0f}"
            )

    nav_dd = report.get("unified_nav_drawdown") if isinstance(report.get("unified_nav_drawdown"), Mapping) else {}
    if nav_line := format_nav_drawdown_summary_line(nav_dd):
        lines.append("")
        lines.append("-- Unified NAV & Drawdown Curve --")
        lines.append(f"  {nav_line}")

    return "\n".join(lines)


__all__ = [
    "evaluate_ai_attribution",
    "extract_fill_execution_slippage",
    "format_ai_attribution_report",
    "format_execution_quality_parts",
    "summarize_delta_gate_logs",
    "summarize_execution_quality_bucket",
    "summarize_execution_quality_by_venue_and_order_type",
]
