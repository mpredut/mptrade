"""Pure deterministic AI survival guards: anti-ping-pong flip protection and recent decision scorecard.

Single Source of Truth (SSOT) for AI-specific failure modes (whipsaw ping-pong reversals and
recent-decision loss streaks). Operates strictly on durable intent history, live spot prices,
and order_guard.conf thresholds in 0 ms with zero network calls or extra disk state.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

from botcore import (
    _parse_bool_flag,
    clean_order_id,
    clean_scalar_str,
    clean_upper_scalar,
    extract_order_price,
    first_pos_float,
    normalize_side,
    opposite_side,
    resolve_now_ts,
    safe_float,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
)
from cachepaths import (
    cache_symbol_variants,
    matches_asset_scope,
    normalize_venue_name,
)
from instrument_registry import casefold_get
from .intent_state import (
    _extract_intent_id,
    _intent_executed_notional_usd,
    _intent_venue_symbol_side_key,
    extract_intent_order_type,
    extract_intent_primary_order_id,
    extract_intent_side,
    extract_intent_timestamp_s,
    extract_intent_trade_plan_levels,
    parse_pos_epoch_s,
    symbols_match_intent_scope_and_quote,
)
from .intent_telemetry import evaluate_trade_plan_status
from .order_ohlc_utils import extract_executed_qty
from .order_status_ssot import NON_FILL_ORDER_STATUSES

_FILLED_VENUE_STATUSES: frozenset[str] = frozenset({"FILLED", "CLOSED", "PARTIALLY_FILLED"})
_WIN_EDGE_PCT = 0.20
_LOSS_EDGE_PCT = -0.35
_OPEN_MTM_LOSS_EDGE_PCT = -1.50
_EMERGENCY_SELL_EDGE_MULT = 2.5
_MAX_BOT_FLIP_COOLDOWN_SEC = 1800.0
_WHIPSAW_WINDOW_SEC = 21600.0


def _collect_canceled_order_ids(items: list[dict[str, Any]], exclude_iid: str = "") -> set[str]:
    canceled: set[str] = set()
    for it in items:
        if not isinstance(it, dict) or (exclude_iid and _extract_intent_id(it) == exclude_iid):
            continue
        for c_id in (it.get("canceled_prior_open_orders") or ()):
            if clean_c := clean_order_id(c_id):
                canceled.add(clean_c)
    return canceled


def _has_explicit_fill_marker(it: dict[str, Any], ep: dict[str, Any], ep_st: str, ep_msg: str) -> bool:
    if ep_st in _FILLED_VENUE_STATUSES or ep_msg.startswith("filled"):
        return True
    for key in ("filled_at", "filled_ts", "avg_fill_price", "avgPrice", "fill_price"):
        if ep.get(key) is not None:
            return True
    return it.get("filled_ts") is not None or it.get("fill_px") is not None


def _is_filled_ai_intent(it: dict[str, Any], canceled_oids: set[str]) -> bool:
    """Return True when an executed AI intent actually filled (excludes canceled/expired/unfilled resting orders)."""
    if (clean_upper_scalar(it.get("status")) or "") != "EXECUTED":
        return False
    ep = it.get("execution_payload") if isinstance(it.get("execution_payload"), dict) else {}
    filled_q = extract_executed_qty(ep, default=0.0) if ep else 0.0
    if filled_q <= 0.0:
        filled_q = safe_nonneg_float(it.get("filled_qty"), default=0.0)
    if filled_q > 0.0:
        return True
    ep_st = (clean_upper_scalar(ep.get("status")) if ep else "") or ""
    ep_msg = (clean_scalar_str(ep.get("msg")) or "").lower() if ep else ""
    if ep_st in NON_FILL_ORDER_STATUSES or ep_msg == "resting":
        return False
    prim_oid = extract_intent_primary_order_id(it, default="", include_top_level=True)
    if prim_oid and str(prim_oid) in canceled_oids:
        return False
    if extract_intent_order_type(it) == "LIMIT":
        return _has_explicit_fill_marker(it, ep, ep_st, ep_msg)
    if _parse_bool_flag(ep.get("reconciled_from_audit"), False) and ep_st not in _FILLED_VENUE_STATUSES:
        return False
    return True


def _extract_intent_exec_price(it: dict[str, Any]) -> float:
    ep = it.get("execution_payload") if isinstance(it.get("execution_payload"), dict) else {}
    hist = it.get("history") if isinstance(it.get("history"), list) else []
    last_h = hist[-1] if hist and isinstance(hist[-1], dict) else {}
    ep_px = extract_order_price(ep, default=None) if ep else None
    px = first_pos_float(
        ep_px,
        ep.get("avg_fill_price"),
        ep.get("avgPrice"),
        ep.get("fill_price"),
        it.get("fill_px"),
        last_h.get("price"),
        it.get("limit_price"),
        it.get("requested_price"),
        it.get("reference_price"),
    )
    return px if px is not None else 0.0


def _extract_intent_exec_ts(it: dict[str, Any]) -> float:
    _, base_ts = extract_intent_timestamp_s(it)
    fill_ts = base_ts
    for key in ("filled_ts", "filled_at"):
        if (ts_v := parse_pos_epoch_s(it.get(key))) is not None and ts_v >= base_ts:
            fill_ts = max(fill_ts, ts_v)
            break
    ep = it.get("execution_payload")
    if isinstance(ep, dict):
        for key in ("filled_at", "filled_ts", "terminal_ts", "updated_ts", "updateTime", "workingTime", "transactTime"):
            if (ts_v := parse_pos_epoch_s(ep.get(key))) is not None and ts_v >= base_ts:
                fill_ts = max(fill_ts, ts_v)
                break
    if fill_ts <= base_ts:
        hist = it.get("history") if isinstance(it.get("history"), list) else []
        last_h = hist[-1] if hist and isinstance(hist[-1], dict) else {}
        if str(last_h.get("action") or "") in ("RECONCILED_EXECUTED", "RECONCILED_PARTIAL"):
            if (lh_ts := parse_pos_epoch_s(last_h.get("ts"))) is not None and lh_ts > fill_ts:
                fill_ts = lh_ts
    if fill_ts <= base_ts:
        for key in ("executed_at", "executed_ts"):
            if (ts_v := parse_pos_epoch_s(it.get(key))) is not None and ts_v > fill_ts:
                fill_ts = ts_v
                break
    return fill_ts


def _extract_intent_notional_usd(it: dict[str, Any]) -> float:
    return _intent_executed_notional_usd(it)


def _resolve_live_price_for_symbol(
    live_prices: Mapping[str, Any],
    venue: str,
    symbol: str,
) -> Optional[float]:
    if not isinstance(live_prices, Mapping) or not live_prices:
        return None
    ven = normalize_venue_name(clean_scalar_str(venue) or "")
    sym = clean_scalar_str(symbol) or ""
    if not sym:
        return None
    cands: list[str] = []
    if ven:
        cands.append(f"{ven}:{sym}")
    cands.append(sym)
    for var in cache_symbol_variants(sym, include_base=True):
        if ven:
            cands.append(f"{ven}:{var}")
        cands.append(var)
    for key in cands:
        raw = casefold_get(live_prices, key)
        if isinstance(raw, Mapping):
            px = safe_pos_float(raw.get("price"), default=None)
        else:
            px = safe_pos_float(raw, default=None)
        if px is not None:
            return px
    return None


def evaluate_ai_ping_pong_guard(
    intent: dict[str, Any],
    existing_intents: list[dict[str, Any]],
    *,
    decision: str,
    symbol: str,
    venue: str,
    price: float,
    now_ts: float,
    flip_cooldown_min: float = 30.0,
    flip_min_edge_pct: float = 0.60,
    max_flips_6h: int = 2,
    spot_price: Optional[float] = None,
    recent_bot_fill: Optional[Mapping[str, Any]] = None,
) -> tuple[bool, str, dict[str, Any]]:
    """Deterministic anti-ping-pong guard protecting against rapid unprofitable opposite-side flips.

    Returns ``(allowed, reason, metrics)``.
    - Allows opposite-side flips immediately when the AI's pre-committed ``invalidation_px`` or
      ``target_px`` plan level is triggered.
    - Allows opposite-side flips when favorable edge >= ``flip_min_edge_pct``, or on emergency
      defensive ``SELL`` drops <= ``-2.5 * flip_min_edge_pct`` (or ``urgency == "HIGH"`` <= ``-flip_min_edge_pct``).
    - Blocks narrow-band/adverse rapid flips within ``flip_cooldown_min`` (including fresh ``by=BOT`` fills
      within 30m) and caps 6h opposite-side reversals per ``(venue, symbol)`` at ``max_flips_6h``.
    """
    side = normalize_side(decision) or extract_intent_side(intent)
    if side not in ("BUY", "SELL"):
        return True, "", {}

    opp_side = opposite_side(side)
    cur_iid = _extract_intent_id(intent)
    eff_now = resolve_now_ts(now_ts)
    eff_px = safe_pos_float(price, default=0.0) or 0.0
    eff_spot = safe_pos_float(spot_price, default=eff_px) or eff_px
    cd_min = safe_nonneg_float(flip_cooldown_min, default=30.0)
    min_edge = safe_nonneg_float(flip_min_edge_pct, default=0.60)
    cap_flips = safe_nonneg_int(max_flips_6h, default=2)
    emergency_drop_pct = min_edge * _EMERGENCY_SELL_EDGE_MULT
    urgency = clean_upper_scalar(intent.get("urgency")) or "NORMAL"

    if cd_min <= 0.0 and cap_flips <= 0:
        return True, "", {}

    items = existing_intents if isinstance(existing_intents, list) else []
    canceled_oids = _collect_canceled_order_ids(items, exclude_iid=cur_iid)
    whipsaw_cutoff = eff_now - _WHIPSAW_WINDOW_SEC

    sym_fills: list[tuple[float, str, float, dict[str, Any]]] = []
    latest_opp_ai: Optional[tuple[float, float, dict[str, Any]]] = None
    latest_same_ai_ts = 0.0
    for it in items:
        if not isinstance(it, dict) or (cur_iid and _extract_intent_id(it) == cur_iid):
            continue
        if not _is_filled_ai_intent(it, canceled_oids):
            continue
        if not symbols_match_intent_scope_and_quote(
            it.get("symbol"), symbol, venue_a=it.get("venue"), venue_b=venue
        ):
            continue
        it_side = extract_intent_side(it)
        if it_side not in ("BUY", "SELL"):
            continue
        ts_s = _extract_intent_exec_ts(it)
        if ts_s <= 0.0 or ts_s > eff_now + 60.0:
            continue
        fill_px = _extract_intent_exec_price(it)
        if ts_s >= whipsaw_cutoff:
            sym_fills.append((ts_s, it_side, fill_px, it))
        if it_side == opp_side and (latest_opp_ai is None or ts_s >= latest_opp_ai[0]):
            latest_opp_ai = (ts_s, fill_px, it)
        elif it_side == side and ts_s > latest_same_ai_ts:
            latest_same_ai_ts = ts_s

    # Resolve any fresh by=BOT opposite-side fill within bot cooldown (<= 30m)
    cd_sec = cd_min * 60.0
    bot_cd_sec = min(cd_sec, _MAX_BOT_FLIP_COOLDOWN_SEC) if cd_sec > 0.0 else 0.0
    bot_opp_ts = 0.0
    bot_opp_px = 0.0
    if bot_cd_sec > 0.0 and isinstance(recent_bot_fill, Mapping):
        b_side = normalize_side(recent_bot_fill.get("side"))
        b_ts = parse_pos_epoch_s(recent_bot_fill.get("ts")) or 0.0
        b_px = safe_pos_float(recent_bot_fill.get("price"), default=0.0) or 0.0
        if (
            (not b_side or b_side == opp_side)
            and b_ts > 0.0
            and b_ts <= eff_now + 60.0
            and (eff_now - b_ts) <= bot_cd_sec
        ):
            bot_opp_ts, bot_opp_px = b_ts, b_px

    # 1. Pre-committed AI trade plan check (only for an active, unclosed opposite AI fill not superseded by a newer BOT fill)
    plan_status = "ACTIVE"
    if (
        latest_opp_ai is not None
        and latest_opp_ai[0] > latest_same_ai_ts
        and latest_opp_ai[0] >= bot_opp_ts
        and eff_spot > 0.0
    ):
        _, opp_px_ai, opp_intent_ai = latest_opp_ai
        inv_px, tgt_px = extract_intent_trade_plan_levels(
            opp_intent_ai, side=opp_side, ref_price=opp_px_ai if opp_px_ai > 0.0 else None
        )
        if inv_px is not None or tgt_px is not None:
            plan_status = evaluate_trade_plan_status(opp_side, eff_spot, inv_px, tgt_px)
            if plan_status in ("INVALIDATED", "TARGET_HIT"):
                return True, "", {"plan_triggered": plan_status, "flip_bypassed_by_plan": True}

    # Count 6h opposite-side reversals on this symbol (strictly by=AI fills)
    sym_fills.sort(key=lambda r: r[0])
    flips_6h = 0
    prev_side_6h = ""
    for _, f_side, _, _ in sym_fills:
        if prev_side_6h and f_side != prev_side_6h:
            flips_6h += 1
        prev_side_6h = f_side

    metrics: dict[str, Any] = {
        "flips_6h": flips_6h,
        "last_6h_side": prev_side_6h or None,
    }

    # Resolve the most recent opposite-side fill within cooldown (AI fill or fresh BOT fill <= 30m)
    ref_opp_ts = 0.0
    ref_opp_px = 0.0
    ref_opp_actor = ""
    if cd_sec > 0.0 and latest_opp_ai is not None and (eff_now - latest_opp_ai[0]) <= cd_sec:
        ref_opp_ts, ref_opp_px, ref_opp_actor = latest_opp_ai[0], latest_opp_ai[1], "AI"
    if bot_opp_ts > ref_opp_ts:
        ref_opp_ts, ref_opp_px, ref_opp_actor = bot_opp_ts, bot_opp_px, "BOT"

    fav_pct: Optional[float] = None
    if ref_opp_px > 0.0 and eff_px > 0.0:
        fav_pct = (
            ((eff_px - ref_opp_px) / ref_opp_px) * 100.0
            if side == "SELL"
            else ((ref_opp_px - eff_px) / ref_opp_px) * 100.0
        )
        metrics["flip_fav_pct"] = round(fav_pct, 3)

    is_emergency_sell = bool(
        side == "SELL"
        and fav_pct is not None
        and (
            fav_pct <= -emergency_drop_pct
            or (urgency == "HIGH" and fav_pct <= -min_edge)
        )
    )

    # 2. Opposite-side flip cooldown check
    if ref_opp_ts > 0.0:
        elapsed_min = max(0.0, (eff_now - ref_opp_ts) / 60.0)
        metrics["flip_elapsed_min"] = round(elapsed_min, 1)
        metrics["flip_ref_actor"] = ref_opp_actor
        has_valid_edge = fav_pct is not None and fav_pct >= min_edge
        if not has_valid_edge and not is_emergency_sell:
            fav_lbl = f"{fav_pct:+.2f}%" if fav_pct is not None else "n/a"
            return (
                False,
                (
                    f"ai_ping_pong_flip_cooldown({symbol}@{venue}:{opp_side}->{side} "
                    f"by={ref_opp_actor}, dt={elapsed_min:.1f}m<{cd_min:g}m, "
                    f"fav={fav_lbl}<+{min_edge:.2f}%)"
                ),
                metrics,
            )

    # 3. 6-hour whipsaw reversal cap per symbol (always allow emergency crash SELL vs latest AI BUY fill)
    if cap_flips > 0 and prev_side_6h and side != prev_side_6h and flips_6h >= cap_flips:
        ai_sell_fav_pct = fav_pct
        if side == "SELL" and ai_sell_fav_pct is None and latest_opp_ai is not None and latest_opp_ai[1] > 0.0 and eff_px > 0.0:
            ai_sell_fav_pct = ((eff_px - latest_opp_ai[1]) / latest_opp_ai[1]) * 100.0
        if not (side == "SELL" and ai_sell_fav_pct is not None and ai_sell_fav_pct <= -emergency_drop_pct):
            return (
                False,
                f"ai_ping_pong_whipsaw_cap({symbol}@{venue}:{flips_6h}/{cap_flips} flips in 6h)",
                metrics,
            )

    return True, "", metrics


def evaluate_ai_decision_scorecard(
    intents: list[dict[str, Any]],
    *,
    now_ts: float,
    live_prices: Optional[Mapping[str, Any]] = None,
    window_trades: int = 5,
    max_loss_streak: int = 3,
    cooldown_hours: float = 4.0,
    lookback_hours: float = 48.0,
    min_mature_minutes: float = 60.0,
    scope: str = "all",
) -> dict[str, Any]:
    """Deterministically score recent filled ``by=AI`` decisions and compute BUY shadow cooldown state.

    - Closed ``BUY -> SELL`` round-trips score realized price edge (``LOSS`` < -0.35%).
    - Closed unmatched ``SELL -> BUY`` reloads score reload timing edge (``LOSS`` < -0.35%).
    - Open ``BUY`` entries score ``LOSS`` immediately when ``invalidation_px`` is breached, or
      once matured (>= ``min_mature_minutes``, default 60m) if mark-to-market < -1.50% (so
      normal intraday DCA dip noise above invalidation never triggers false BUY cooldowns).
    - Standalone unmatched ``SELL`` exits (e.g. taking profit on older inventory) are never penalized.
    """
    eff_now = resolve_now_ts(now_ts)
    win_n = safe_nonneg_int(window_trades, default=5)
    streak_cap = safe_nonneg_int(max_loss_streak, default=3)
    cd_hours = safe_nonneg_float(cooldown_hours, default=4.0)
    lb_sec = max(3600.0, (safe_pos_float(lookback_hours, default=48.0) or 48.0) * 3600.0)
    mature_sec = max(0.0, safe_nonneg_float(min_mature_minutes, default=60.0)) * 60.0
    prices_map = live_prices if isinstance(live_prices, Mapping) else {}
    items = intents if isinstance(intents, list) else []

    empty_res: dict[str, Any] = {
        "count": 0,
        "window_trades": win_n,
        "wins": 0,
        "scratches": 0,
        "losses": 0,
        "net_usd": 0.0,
        "avg_edge_pct": 0.0,
        "loss_streak": 0,
        "weak_window": False,
        "latest_loss_ts": 0.0,
        "buy_shadow_active": False,
        "cooldown_remaining_h": 0.0,
        "status": "HEALTHY",
        "reason": "",
    }
    if win_n <= 0 or not items:
        return empty_res

    canceled_oids = _collect_canceled_order_ids(items)
    cutoff = eff_now - lb_sec

    filled_rows: list[dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict) or not _is_filled_ai_intent(it, canceled_oids):
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
        if ts_s <= 0.0 or ts_s < cutoff or ts_s > eff_now + 60.0 or px <= 0.0:
            continue
        inv_px, tgt_px = extract_intent_trade_plan_levels(it, side=side, ref_price=px)
        v_k, b_k, q_k, _ = _intent_venue_symbol_side_key(ven, sym, side)
        filled_rows.append({
            "ts": ts_s,
            "venue": ven,
            "symbol": sym,
            "inst_key": (v_k, b_k, q_k),
            "side": side,
            "px": px,
            "usd": _extract_intent_notional_usd(it),
            "invalidation_px": inv_px,
            "target_px": tgt_px,
        })

    if not filled_rows:
        return empty_res

    filled_rows.sort(key=lambda r: r["ts"])

    def _classify(
        edge_pct: float,
        *,
        invalidated: bool = False,
        loss_threshold: float = _LOSS_EDGE_PCT,
    ) -> str:
        if invalidated or edge_pct < loss_threshold:
            return "LOSS"
        if edge_pct > _WIN_EDGE_PCT:
            return "WIN"
        return "SCRATCH"

    pending_buys: dict[tuple[str, str, tuple[str, ...]], list[dict[str, Any]]] = {}
    pending_sells: dict[tuple[str, str, tuple[str, ...]], list[dict[str, Any]]] = {}
    outcomes: list[dict[str, Any]] = []

    for row in filled_rows:
        ikey = row["inst_key"]
        if row["side"] == "BUY":
            s_queue = pending_sells.get(ikey)
            if s_queue:
                prev_sell = s_queue.pop()
                edge_pct = ((prev_sell["px"] - row["px"]) / prev_sell["px"]) * 100.0
                matched_usd = min(prev_sell["usd"], row["usd"]) if (prev_sell["usd"] > 0.0 and row["usd"] > 0.0) else max(prev_sell["usd"], row["usd"])
                outcomes.append({
                    "ts": row["ts"],
                    "kind": "SELL->BUY",
                    "symbol": row["symbol"],
                    "edge_pct": edge_pct,
                    "usd_impact": matched_usd * (edge_pct / 100.0),
                    "grade": _classify(edge_pct),
                })
            else:
                pending_buys.setdefault(ikey, []).append(row)
        else:
            b_queue = pending_buys.get(ikey)
            if b_queue:
                prev_buy = b_queue.pop()
                edge_pct = ((row["px"] - prev_buy["px"]) / prev_buy["px"]) * 100.0
                matched_usd = min(prev_buy["usd"], row["usd"]) if (prev_buy["usd"] > 0.0 and row["usd"] > 0.0) else max(prev_buy["usd"], row["usd"])
                outcomes.append({
                    "ts": row["ts"],
                    "kind": "BUY->SELL",
                    "symbol": row["symbol"],
                    "edge_pct": edge_pct,
                    "usd_impact": matched_usd * (edge_pct / 100.0),
                    "grade": _classify(edge_pct),
                })
            else:
                pending_sells.setdefault(ikey, []).append(row)

    for b_queue in pending_buys.values():
        for buy_row in b_queue:
            spot_px = _resolve_live_price_for_symbol(prices_map, buy_row["venue"], buy_row["symbol"])
            if spot_px is None or spot_px <= 0.0:
                continue
            plan_st = evaluate_trade_plan_status(
                "BUY", spot_px, buy_row["invalidation_px"], buy_row["target_px"]
            )
            is_invalidated = plan_st == "INVALIDATED"
            if not is_invalidated and (eff_now - buy_row["ts"]) < mature_sec:
                continue
            edge_pct = ((spot_px - buy_row["px"]) / buy_row["px"]) * 100.0
            outcomes.append({
                "ts": buy_row["ts"],
                "kind": "BUY_MTM",
                "symbol": buy_row["symbol"],
                "edge_pct": edge_pct,
                "usd_impact": buy_row["usd"] * (edge_pct / 100.0),
                "grade": _classify(
                    edge_pct,
                    invalidated=is_invalidated,
                    loss_threshold=_OPEN_MTM_LOSS_EDGE_PCT,
                ),
            })

    if not outcomes:
        return empty_res

    outcomes.sort(key=lambda o: o["ts"])
    recent = outcomes[-win_n:]
    wins = sum(1 for o in recent if o["grade"] == "WIN")
    scratches = sum(1 for o in recent if o["grade"] == "SCRATCH")
    losses = sum(1 for o in recent if o["grade"] == "LOSS")
    net_usd = sum(o["usd_impact"] for o in recent)
    avg_edge = sum(o["edge_pct"] for o in recent) / len(recent)

    loss_streak = 0
    for o in reversed(recent):
        if o["grade"] == "LOSS":
            loss_streak += 1
        else:
            break

    weak_threshold = max(win_n - 1, streak_cap) if streak_cap > 0 else max(win_n - 1, 1)
    weak_window = bool(len(recent) >= win_n and losses >= weak_threshold and net_usd < 0.0)
    tripped_streak = bool(streak_cap > 0 and loss_streak >= streak_cap)
    latest_loss_ts = max((o["ts"] for o in recent if o["grade"] == "LOSS"), default=0.0)

    buy_shadow_active = False
    cooldown_remaining_h = 0.0
    reason = ""
    if (tripped_streak or weak_window) and cd_hours > 0.0 and latest_loss_ts > 0.0:
        cd_until = latest_loss_ts + cd_hours * 3600.0
        if cd_until > eff_now:
            buy_shadow_active = True
            cooldown_remaining_h = round((cd_until - eff_now) / 3600.0, 2)
            trig = f"streak={loss_streak}L" if tripped_streak else f"weak_window={losses}/{len(recent)}L"
            net_lbl = f"+${abs(net_usd):.0f}" if net_usd >= 0.0 else f"-${abs(net_usd):.0f}"
            reason = (
                f"ai_scorecard_buy_shadow_cooldown({trig}, net={net_lbl}, "
                f"remaining={cooldown_remaining_h:.1f}h)"
            )

    if buy_shadow_active:
        status = "BUY_SHADOW_COOLDOWN"
    elif streak_cap > 1 and loss_streak >= streak_cap - 1:
        status = "CAUTION"
    else:
        status = "HEALTHY"

    return {
        "count": len(recent),
        "window_trades": win_n,
        "wins": wins,
        "scratches": scratches,
        "losses": losses,
        "net_usd": round(net_usd, 2),
        "avg_edge_pct": round(avg_edge, 2),
        "loss_streak": loss_streak,
        "weak_window": weak_window,
        "latest_loss_ts": round(latest_loss_ts, 3),
        "buy_shadow_active": buy_shadow_active,
        "cooldown_remaining_h": cooldown_remaining_h,
        "status": status,
        "reason": reason,
    }


def format_ai_decision_scorecard_line(scorecard: Any) -> str:
    """Format a compact single-line summary of the AI Decision Scorecard for Section 3C."""
    if not isinstance(scorecard, Mapping):
        return ""
    cnt = safe_nonneg_int(scorecard.get("count"), default=0)
    if cnt <= 0:
        return ""
    w = safe_nonneg_int(scorecard.get("wins"), default=0)
    s = safe_nonneg_int(scorecard.get("scratches"), default=0)
    lo = safe_nonneg_int(scorecard.get("losses"), default=0)
    net_u = safe_float(scorecard.get("net_usd"), default=0.0) or 0.0
    avg_e = safe_float(scorecard.get("avg_edge_pct"), default=0.0) or 0.0
    streak = safe_nonneg_int(scorecard.get("loss_streak"), default=0)
    status = str(scorecard.get("status") or "HEALTHY").strip().upper()
    rem_h = safe_nonneg_float(scorecard.get("cooldown_remaining_h"), default=0.0)
    streak_sfx = f", streak={streak}L" if streak >= 2 else ""
    if status == "BUY_SHADOW_COOLDOWN" and rem_h > 0.0:
        tag = f"[BUY_SHADOW_COOLDOWN: {rem_h:.1f}h left]"
    else:
        tag = f"[{status}]"
    net_lbl = f"+${abs(net_u):.0f}" if net_u >= 0.0 else f"-${abs(net_u):.0f}"
    return (
        f"AI Decision Scorecard (48h, n={cnt}): "
        f"{w}W/{s}S/{lo}L, avg={avg_e:+.2f}%, net={net_lbl}{streak_sfx} {tag}"
    )


__all__ = [
    "evaluate_ai_decision_scorecard",
    "evaluate_ai_ping_pong_guard",
    "format_ai_decision_scorecard_line",
]
