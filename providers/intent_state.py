"""Pure in-memory AI intent schema normalization, validation, and ID correlation SSOT.

Strictly zero disk I/O (no ``os``, ``pathlib``, ``state_io``, or ``state_stores`` imports).
Durable JSON persistence wrappers live in ``providers/intent_state_store.py`` and
``state_stores/autonomous_trade_intents_state.py``.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Callable, Mapping, Optional

from botcore import (
    _parse_bool_flag,
    _parse_epoch_or_iso_s,
    _ts_ms_to_s,
    clean_json_value,
    clean_order_id,
    clean_scalar_str,
    clean_scale,
    clean_upper_scalar,
    conf_or_override,
    extract_order_id,
    normalize_side,
    parse_notify_enabled_flag,
    safe_float,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
)
from cachepaths import (
    canonical_base_asset,
    compatible_quote_suffixes,
    has_equity_suffix,
    is_equity_venue,
    is_equity_venue_or_suffix,
    normalize_venue_name,
    same_canonical_base_asset,
    symbol_quote_asset,
)
from instrument_registry import canonical_display_symbol, resolve_symbol_identity
from .order_id_context import (
    ScopedOrderIdSet,
    all_venue_native_client_order_ids,
    derive_ai_intent_client_order_id,
    extract_client_order_id,
    resolve_order_actor,
)
from .order_ohlc_utils import extract_executed_qty, extract_orig_qty
from .order_status_ssot import FILL_COMPLETE_QTY_RATIO, TERMINAL_FILL_ROW_STATUSES

_CURRENCY_STRIP_RE = re.compile(r"[\$,\s]|(?:USD[CT]?)\b", re.IGNORECASE)
_HORIZON_STR_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([mhd]?)\s*$", re.IGNORECASE)
_EXPLICIT_INTENT_OID_KEYS: tuple[str, ...] = ("orderId", "order_id", "ordertxid", "txid", "oid")
_INTENT_TIMESTAMP_KEYS: tuple[str, ...] = ("timestamp", "executed_at", "executed_ts", "created_at", "ts")
_NOTIONAL_KEYS: tuple[str, ...] = (
    "suggested_notional_usd",
    "notional_usd",
    "amount_usd",
    "size_usd",
    "usd_notional",
    "notional",
    "value_usd",
)


def _clean_intent_symbol(raw_sym: Any) -> str:
    if raw_sym is None or isinstance(raw_sym, (bool, dict, list, tuple, set, frozenset)):
        return ""
    s = clean_scalar_str(str(raw_sym))
    if s is None:
        return ""
    return resolve_symbol_identity(s).storage_key or s.upper()


def _extract_intent_id(record: Any, default: str = "") -> str:
    """Return a clean non-boolean intent_id (falling back to ``id``) from ``record``."""
    if not isinstance(record, dict):
        return default
    for key in ("intent_id", "id"):
        clean = clean_order_id(record.get(key))
        if clean:
            return clean
    return default


def append_intent_history(intent: dict[str, Any], entry: dict[str, Any]) -> None:
    """Append an entry to ``intent['history']``, coercing null or non-list history fields."""
    hist = intent.get("history")
    if not isinstance(hist, list):
        hist = []
        intent["history"] = hist
    hist.append(entry)


def _record_intent_state(
    intent: dict[str, Any],
    status: str,
    now_ts: float,
    *,
    action: Optional[str] = None,
    **fields: Any,
) -> None:
    """Set ``intent['status'] = status`` and append a timestamped history entry."""
    intent["status"] = status
    append_intent_history(intent, {"ts": now_ts, "action": action or status, **fields})


def ai_intent_client_order_id(intent_id: Any) -> Optional[str]:
    """Durable ``AI_`` client ID of an autonomous intent (<= 36 chars, Binance newClientOrderId compatible)."""
    return derive_ai_intent_client_order_id(intent_id)


def intent_venue_client_order_ids(intent: Any) -> set[str]:
    """Return all lowercase venue-native client IDs an autonomous intent's order may carry on any venue."""
    if not isinstance(intent, dict):
        return set()
    return all_venue_native_client_order_ids(ai_intent_client_order_id(_extract_intent_id(intent)))


def _intent_order_ids(item: dict[str, Any]) -> set[str]:
    ids: set[str] = set(intent_venue_client_order_ids(item))
    oid_rec = extract_order_id(item.get("execution_payload"), default="")
    if oid_rec:
        ids.add(str(oid_rec))
    for h_entry in (item.get("history") if isinstance(item.get("history"), list) else ()):
        if isinstance(h_entry, dict):
            h_oid = extract_order_id(h_entry.get("order_id"), default="") or extract_order_id(h_entry.get("payload"), default="")
            if h_oid:
                ids.add(str(h_oid))
    for r_id in (item.get("restored_ai_order_ids") if isinstance(item.get("restored_ai_order_ids"), list) else ()):
        if clean_r := clean_order_id(r_id):
            ids.add(clean_r)
            if clean_r.upper().startswith("AI_"):
                ids.update(all_venue_native_client_order_ids(clean_r))
    return ids


def collect_known_ai_order_ids(
    intents: Any,
    initial_ids: Optional[set[str]] = None,
    *,
    venue: Any = None,
    symbol: Any = None,
    account: Any = None,
) -> ScopedOrderIdSet:
    """Collect venue- and symbol-scoped AI order IDs and client IDs across intents."""
    ai_oids = ScopedOrderIdSet()
    want_v = normalize_venue_name(venue) if venue else ""
    want_s = (canonical_display_symbol(symbol, venue=want_v or None) or "").lower() if symbol else ""
    want_a = (clean_scalar_str(account) or "").lower() if account else ""

    if isinstance(initial_ids, ScopedOrderIdSet):
        for k in initial_ids.scoped_keys:
            if want_v and k.venue and k.venue != want_v:
                continue
            if want_s and k.symbol and k.symbol != want_s:
                continue
            if want_a and k.account and k.account != want_a:
                continue
            ai_oids.add_scoped(k.order_id, venue=k.venue, symbol=k.symbol, account=k.account)
        for u_id in initial_ids.unscoped_ids:
            ai_oids.add_unscoped(u_id)
    elif isinstance(initial_ids, str):
        if clean_s := clean_order_id(initial_ids):
            if want_v or want_s or want_a:
                ai_oids.add_scoped(clean_s, venue=want_v, symbol=want_s, account=want_a)
            else:
                ai_oids.add_unscoped(clean_s)
    elif initial_ids is not None and not isinstance(initial_ids, (bool, bytes, dict)) and hasattr(initial_ids, "__iter__"):
        for item_id in initial_ids:
            if clean_s := clean_order_id(item_id):
                if want_v or want_s or want_a:
                    ai_oids.add_scoped(clean_s, venue=want_v, symbol=want_s, account=want_a)
                else:
                    ai_oids.add_unscoped(clean_s)
    if isinstance(intents, dict):
        intents = intents.get("intents")
    if not isinstance(intents, list):
        return ai_oids
    for item in intents:
        if not isinstance(item, dict):
            continue
        item_v = normalize_venue_name(item.get("venue") or item.get("provider_name") or item.get("provider")) or ""
        item_s = (canonical_display_symbol(item.get("symbol"), venue=item_v or None) or "").lower()
        item_a = (clean_scalar_str(item.get("account_ref") or item.get("account")) or "").lower()
        if want_v and item_v and item_v != want_v:
            continue
        if want_s and item_s and item_s != want_s:
            continue
        if want_a and item_a and item_a != want_a:
            continue
        for oid_tok in _intent_order_ids(item):
            ai_oids.add_scoped(oid_tok, venue=item_v, symbol=item_s, account=item_a)
    return ai_oids


def prune_reconciled_terminal_open_orders(
    open_orders: Any,
    intents: Any,
) -> list[dict[str, Any]]:
    """Drop stale cached open orders already confirmed terminal (FILLED/CANCELED/EXPIRED) in AI intent state."""
    if not isinstance(open_orders, list) or not open_orders:
        return list(open_orders) if isinstance(open_orders, list) else []
    if isinstance(intents, dict):
        intents = intents.get("intents")
    if not isinstance(intents, list) or not intents:
        return list(open_orders)

    terminal_scoped: set[tuple[str, str]] = set()
    terminal_unscoped: set[str] = set()
    for item in intents:
        if not isinstance(item, dict):
            continue
        ep = item.get("execution_payload")
        if not isinstance(ep, dict):
            continue
        ep_st = clean_upper_scalar(ep.get("status")) or ""
        if ep_st not in TERMINAL_FILL_ROW_STATUSES:
            continue
        is_reconciled = _parse_bool_flag(ep.get("reconciled_terminal"), False)
        if not is_reconciled:
            if ep_st not in ("FILLED", "CLOSED") or clean_upper_scalar(item.get("status")) != "EXECUTED":
                continue
            exec_q = safe_nonneg_float(extract_executed_qty(ep, default=0.0), default=0.0) or 0.0
            orig_q = safe_pos_float(extract_orig_qty(ep, default=None), default=None)
            if exec_q <= 0.0 or (orig_q is not None and exec_q < orig_q * FILL_COMPLETE_QTY_RATIO):
                continue
        item_v = normalize_venue_name(item.get("venue") or item.get("provider_name") or item.get("provider")) or ""
        for oid_tok in _intent_order_ids(item):
            if item_v:
                terminal_scoped.add((item_v, oid_tok))
            else:
                terminal_unscoped.add(oid_tok)

    if not terminal_scoped and not terminal_unscoped:
        return list(open_orders)

    out: list[dict[str, Any]] = []
    for order in open_orders:
        if not isinstance(order, dict):
            out.append(order)
            continue
        ord_v = normalize_venue_name(order.get("venue") or order.get("provider_name") or order.get("provider")) or ""
        cand_ids: list[str] = []
        if oid := clean_order_id(extract_order_id(order, default=None)):
            cand_ids.append(oid)
        if cid := clean_order_id(extract_client_order_id(order, default=None)):
            cand_ids.append(cid)
            if cid.upper().startswith("AI_"):
                cand_ids.extend(all_venue_native_client_order_ids(cid))
        is_terminal = any(
            (ord_v and (ord_v, tok) in terminal_scoped)
            or (not ord_v and any(t_id == tok for _, t_id in terminal_scoped))
            or tok in terminal_unscoped
            for tok in cand_ids
        )
        if not is_terminal:
            out.append(order)
    return out


def extract_intent_side(data: dict[str, Any]) -> str:
    """Extract normalized BUY/SELL/HOLD side from an intent dictionary."""
    if not isinstance(data, dict):
        return ""
    for k in ("decision", "action", "side"):
        if data.get(k) is not None:
            norm = normalize_side(data.get(k))
            if norm:
                return norm
    return ""


def extract_intent_notional(data: dict[str, Any], default: Any = None) -> Any:
    """Extract and parse positive suggested notional USD across aliases and currency strings."""
    if not isinstance(data, dict):
        return default
    for nk in _NOTIONAL_KEYS:
        if data.get(nk) is not None:
            raw_nk = data.get(nk)
            val = safe_pos_float(_CURRENCY_STRIP_RE.sub("", raw_nk).strip() if isinstance(raw_nk, str) else raw_nk, None)
            if val is not None:
                return val
    return default


def _resolve_clamped_intent_notional(
    data: dict[str, Any],
    min_notional_usd: float,
    max_notional_usd: float,
) -> Optional[float]:
    """Resolve and clamp an explicit positive suggested notional USD, or return None if missing/non-positive."""
    if not isinstance(data, dict):
        return None
    for nk in _NOTIONAL_KEYS:
        raw = data.get(nk)
        if raw is None or isinstance(raw, bool):
            continue
        cleaned = _CURRENCY_STRIP_RE.sub("", raw).strip() if isinstance(raw, str) else raw
        if cleaned == "":
            continue
        num = safe_float(cleaned, None)
        if num is None or num <= 0.0:
            return None
        min_usd = safe_nonneg_float(min_notional_usd, default=0.0)
        max_usd = safe_nonneg_float(max_notional_usd, default=min_usd)
        return max(min(min_usd, max_usd), min(num, max_usd))
    return None


def extract_intent_limit_price(data: dict[str, Any], default: Any = None) -> Any:
    """Extract and parse positive passive limit price across aliases and currency strings."""
    if not isinstance(data, dict):
        return default
    keys: tuple[str, ...] = ("limit_price", "target_price", "order_price", "limit_px")
    if (clean_upper_scalar(data.get("order_type")) or clean_upper_scalar(data.get("type")) or "") == "LIMIT":
        keys = (*keys, "price")
    for pk in keys:
        if data.get(pk) is not None:
            raw_pk = data.get(pk)
            val = safe_pos_float(_CURRENCY_STRIP_RE.sub("", raw_pk).strip() if isinstance(raw_pk, str) else raw_pk, None)
            if val is not None:
                return val
    return default


def extract_intent_order_type(data: dict[str, Any]) -> str:
    """Resolve normalized order execution type ('MARKET' or 'LIMIT') for an intent."""
    if not isinstance(data, dict):
        return "MARKET"
    norm_type = clean_upper_scalar(data.get("order_type")) or clean_upper_scalar(data.get("type")) or ""
    limit_px = extract_intent_limit_price(data, default=None)
    if norm_type == "MARKET":
        return "MARKET"
    if norm_type == "LIMIT" or limit_px is not None:
        return "LIMIT"
    return "MARKET"


_INVALIDATION_PX_KEYS: tuple[str, ...] = ("invalidation_px", "invalidation_price", "stop_loss_px")
_TARGET_PX_KEYS: tuple[str, ...] = ("target_px", "take_profit_px", "tp_px")
_HORIZON_MAX_HOURS: float = 720.0


def extract_intent_horizon_hours(data: Any) -> Optional[float]:
    """Extract optional thesis horizon in positive hours (``<= 720h``) from ``horizon_h`` / ``horizon``."""
    if not isinstance(data, Mapping):
        return None
    for key in ("horizon_h", "horizon"):
        raw = data.get(key)
        if raw is None or isinstance(raw, bool):
            continue
        hours: Optional[float] = None
        if isinstance(raw, (int, float)):
            hours = safe_pos_float(raw, default=None)
        elif isinstance(raw, str):
            m = _HORIZON_STR_RE.match(raw)
            if m:
                val = safe_pos_float(m.group(1), default=None)
                unit = m.group(2).lower()
                if val is not None:
                    hours = val / 60.0 if unit == "m" else (val * 24.0 if unit == "d" else val)
        if hours is not None and 0.0 < hours <= _HORIZON_MAX_HOURS:
            return round(hours, 2)
    return None


def _parse_optional_plan_px(data: dict[str, Any], keys: tuple[str, ...]) -> Optional[float]:
    for pk in keys:
        raw_v = data.get(pk)
        if raw_v is None or isinstance(raw_v, bool):
            continue
        cleaned = _CURRENCY_STRIP_RE.sub("", raw_v).strip() if isinstance(raw_v, str) else raw_v
        val = safe_pos_float(cleaned, None)
        if val is not None:
            return round(val, 8)
    return None


def extract_intent_trade_plan_levels(
    data: dict[str, Any],
    *,
    side: str = "",
    ref_price: Optional[float] = None,
) -> tuple[Optional[float], Optional[float]]:
    """Extract and geometrically validate ``(invalidation_px, target_px)`` for a BUY/SELL trade plan.

    Invalid, inverted, or wildly out-of-range levels (< 0.25x or > 4.0x anchor) are dropped to
    ``None`` cleanly without blocking order execution.
    """
    if not isinstance(data, dict):
        return None, None
    eff_side = normalize_side(side) or extract_intent_side(data)
    if eff_side not in ("BUY", "SELL"):
        return None, None
    inv_px = _parse_optional_plan_px(data, _INVALIDATION_PX_KEYS)
    tgt_px = _parse_optional_plan_px(data, _TARGET_PX_KEYS)
    anchor_px = safe_pos_float(ref_price, default=None) or extract_intent_limit_price(data, default=None)
    if anchor_px is not None and anchor_px > 0.0:
        lo_bound = anchor_px * 0.25
        hi_bound = anchor_px * 4.0
        if eff_side == "BUY":
            if inv_px is not None and not (lo_bound <= inv_px < anchor_px):
                inv_px = None
            if tgt_px is not None and not (anchor_px < tgt_px <= hi_bound):
                tgt_px = None
        else:
            if tgt_px is not None and not (lo_bound <= tgt_px < anchor_px):
                tgt_px = None
            if inv_px is not None and not (anchor_px < inv_px <= hi_bound):
                inv_px = None
    elif inv_px is not None and tgt_px is not None:
        if eff_side == "BUY" and not (inv_px < tgt_px):
            inv_px, tgt_px = None, None
        elif eff_side == "SELL" and not (tgt_px < inv_px):
            inv_px, tgt_px = None, None
    return inv_px, tgt_px


def extract_intent_cancel_order_ids(data: dict[str, Any]) -> list[str]:
    """Extract deduplicated explicit venue order IDs targeted for cancellation before placement."""
    if not isinstance(data, dict):
        return []
    found: list[str] = []
    for key in ("cancel_order_ids", "target_order_ids", "cancel_order_id", "replace_order_id"):
        raw = data.get(key)
        if raw is None or isinstance(raw, bool):
            continue
        items: list[Any]
        if isinstance(raw, (list, tuple, set, frozenset)):
            items = list(raw)
        elif isinstance(raw, str) and "," in raw:
            items = [part.strip() for part in raw.split(",")]
        else:
            items = [raw]
        for item in items:
            if item is None or isinstance(item, bool):
                continue
            oid = extract_order_id(item, default="")
            if oid:
                found.append(str(oid))
    return list(dict.fromkeys(found))


def collect_canceled_order_ids(items: Sequence[Any], exclude_iid: str = "") -> set[str]:
    """Collect deduplicated canceled order IDs across both replace-cancel and standalone-cancel intent records."""
    canceled: set[str] = set()
    for it in items:
        if not isinstance(it, Mapping) or (exclude_iid and _extract_intent_id(it) == exclude_iid):
            continue
        for field_k in ("canceled_prior_open_orders", "canceled_order_ids"):
            raw_ids = it.get(field_k)
            if isinstance(raw_ids, (list, tuple, set, frozenset)):
                for c_id in raw_ids:
                    if clean_c := clean_order_id(c_id):
                        canceled.add(clean_c)
    return canceled


def decision_has_cancel_directive(d: Any) -> bool:
    """Return True when a trading decision dict specifies an executable bot/order cancellation."""
    if not isinstance(d, dict):
        return False
    if extract_intent_cancel_order_ids(d):
        return True
    return extract_intent_side(d) in ("BUY", "SELL") and _parse_bool_flag(
        d.get("cancel_bot_open_orders"), False
    )


def eval_doc_was_actionable(eval_doc: Any) -> bool:
    """Return True when a supervisor eval artifact emitted any BUY/SELL/cancel directive or non-HOLD intent."""
    if not isinstance(eval_doc, Mapping):
        return False
    decs = eval_doc.get("market_trading_decisions")
    dec_list = decs if isinstance(decs, list) else []
    if not dec_list and isinstance(eval_doc.get("market_trading_decision"), Mapping):
        dec_list = [eval_doc["market_trading_decision"]]
    for d in dec_list:
        if not isinstance(d, Mapping):
            continue
        if str(d.get("decision") or "HOLD").strip().upper() in ("BUY", "SELL") or decision_has_cancel_directive(dict(d)):
            return True
    intents = eval_doc.get("intents_emitted")
    if isinstance(intents, list):
        for it in intents:
            if not isinstance(it, Mapping):
                continue
            if str(it.get("side") or it.get("decision") or "HOLD").strip().upper() in ("BUY", "SELL"):
                return True
    return False


def resolve_hold_cancel_target(
    decision: Any,
    *,
    min_confidence: float,
    resolve_instrument_fn: Callable[[str, str], Any],
) -> tuple[Any, list[str], bool]:
    """Resolve (instrument, target_order_ids, only_bot_orders) for a CANCEL or HOLD decision with targeted cancellation."""
    if not isinstance(decision, dict):
        return None, [], True
    raw_dec = clean_upper_scalar(decision.get("decision") or decision.get("action") or decision.get("side")) or "HOLD"
    if extract_intent_side(decision) in ("BUY", "SELL") or raw_dec not in ("HOLD", "CANCEL"):
        return None, [], True
    target_oids = extract_intent_cancel_order_ids(decision)
    if not target_oids or clean_scale(decision.get("confidence"), 0.0, allow_percent=True) < clean_scale(min_confidence, 1.0, allow_percent=True):
        return None, [], True
    sym = _clean_intent_symbol(decision.get("symbol", ""))
    ven = normalize_venue_name(clean_scalar_str(decision.get("venue")) or "")
    if not (sym and ven):
        return None, [], True
    inst = resolve_instrument_fn(sym, ven)
    if inst is None or is_non_fiat_btc_quote_instrument(inst):
        return None, [], True
    only_bots = False if raw_dec == "CANCEL" else (not _parse_bool_flag(decision.get("replace_open_orders"), default=False))
    return inst, target_oids, only_bots


def _intent_venue_symbol_side_key(
    venue: Any, symbol: Any, side: Any
) -> tuple[str, str, tuple[str, ...], str]:
    s_raw = _clean_intent_symbol(symbol)
    v_norm = normalize_venue_name(venue)
    base = canonical_base_asset(s_raw) or s_raw
    if base and is_equity_venue_or_suffix(s_raw, v_norm):
        base = f"EQ:{base}"
    side_s = "" if side is None or isinstance(side, bool) else (normalize_side(side) or str(side).strip().upper())
    return (
        v_norm,
        base,
        compatible_quote_suffixes(s_raw),
        side_s,
    )


def normalize_trading_decisions(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract and deduplicate single or multi-coin trading decisions from Gemini's JSON plan."""
    if not isinstance(plan, dict):
        return []
    raw_multi = None
    for k in ("market_trading_decisions", "trading_decisions", "decisions"):
        if isinstance(plan.get(k), list):
            raw_multi = plan.get(k)
            break
    raw_single = plan.get("market_trading_decision")
    candidates: list[dict[str, Any]] = []
    for item in (
        *(raw_multi if isinstance(raw_multi, list) else ()),
        *([raw_single] if isinstance(raw_single, dict) and raw_single else ()),
    ):
        if isinstance(item, dict):
            d = dict(item)
            raw_val = d.get("decision") or d.get("action") or d.get("side")
            if isinstance(raw_val, bool) or raw_val is None:
                raw_side = "HOLD"
            else:
                raw_side = normalize_side(raw_val) or str(raw_val).strip().upper() or "HOLD"
            d["decision"] = raw_side
            raw_sym = d.get("symbol")
            clean_sym = _clean_intent_symbol(raw_sym)
            if clean_sym:
                d["symbol"] = clean_sym
            elif raw_sym is not None:
                d.pop("symbol", None)
            v_norm = normalize_venue_name(d.get("venue"))
            if v_norm:
                d["venue"] = v_norm
            if "confidence" in d and d.get("confidence") is not None:
                d["confidence"] = clean_scale(d.get("confidence"), default=0.0, allow_percent=True)
            extracted_usd = extract_intent_notional(d, default=None)
            if extracted_usd is not None:
                d["suggested_notional_usd"] = extracted_usd
            limit_px = extract_intent_limit_price(d, default=None)
            order_type = extract_intent_order_type(d)
            d["order_type"] = order_type
            if limit_px is not None and order_type == "LIMIT":
                d["limit_price"] = limit_px
            inv_px, tgt_px = extract_intent_trade_plan_levels(d, side=raw_side, ref_price=limit_px)
            if inv_px is not None:
                d["invalidation_px"] = inv_px
            elif "invalidation_px" in d:
                d.pop("invalidation_px", None)
            if tgt_px is not None:
                d["target_px"] = tgt_px
            elif "target_px" in d:
                d.pop("target_px", None)
            if (hor_h := extract_intent_horizon_hours(d)) is not None:
                d["horizon_h"] = hor_h
            elif "horizon_h" in d:
                d.pop("horizon_h", None)
            if "replace_open_orders" in d:
                d["replace_open_orders"] = _parse_bool_flag(d.get("replace_open_orders"), False)
            if "cancel_bot_open_orders" in d:
                d["cancel_bot_open_orders"] = _parse_bool_flag(d.get("cancel_bot_open_orders"), False)
            cancel_oids = extract_intent_cancel_order_ids(d)
            if cancel_oids:
                d["cancel_order_ids"] = cancel_oids
            elif "cancel_order_ids" in d:
                d["cancel_order_ids"] = []
            candidates.append(d)

    def _sort_key(d: dict[str, Any]) -> tuple[int, int, float]:
        side = str(d.get("decision") or "HOLD").strip().upper()
        is_action = 1 if side in ("BUY", "SELL") else 0
        has_cancel = 1 if decision_has_cancel_directive(d) else 0
        conf = clean_scale(d.get("confidence"), default=0.0, allow_percent=True)
        return (-is_action, -has_cancel, -conf)

    candidates.sort(key=_sort_key)

    decisions: list[dict[str, Any]] = []
    seen_keys: set[tuple[Any, ...]] = set()
    covered_cancel_oids: dict[tuple[Any, ...], set[str]] = {}
    for item in candidates:
        venue_norm, sym_canon, quotes, side_norm = _intent_venue_symbol_side_key(
            item.get("venue"), item.get("symbol"), item.get("decision") or "HOLD"
        )
        key = (venue_norm, sym_canon, quotes, "") if sym_canon else (venue_norm, "", (), side_norm)
        cancel_oids = extract_intent_cancel_order_ids(item)
        if side_norm not in ("BUY", "SELL") and cancel_oids and sym_canon:
            seen_oids = covered_cancel_oids.setdefault(key, set())
            new_oids = [oid for oid in cancel_oids if oid not in seen_oids]
            if new_oids:
                seen_oids.update(new_oids)
                seen_keys.add(key)
                decisions.append({**item, "cancel_order_ids": new_oids} if len(new_oids) != len(cancel_oids) else item)
            continue
        if key not in seen_keys:
            seen_keys.add(key)
            if cancel_oids and sym_canon:
                covered_cancel_oids.setdefault(key, set()).update(cancel_oids)
            decisions.append(item)
    return decisions


def generate_unique_intent_id(
    now_ts: float,
    side: str,
    symbol: str,
    existing_intents: list[dict[str, Any]],
) -> str:
    """Generate a unique intent_id for the durable intent queue."""
    base_id = f"intent_{safe_nonneg_int(now_ts, default=0)}_{side.lower()}_{symbol.lower()}"
    existing_ids = {_extract_intent_id(i) for i in existing_intents if isinstance(i, dict)} - {""}
    intent_id = base_id
    suffix = len(existing_intents) + 1
    while intent_id in existing_ids:
        intent_id = f"{base_id}_{suffix}"
        suffix += 1
    return intent_id


def normalize_intent_fields(
    intent: dict[str, Any],
    now_ts: float,
) -> tuple[str, str, str, str]:
    """Extract and normalize (intent_id, decision, symbol, venue) from an intent dict."""
    intent_id = _extract_intent_id(intent, default=f"intent_{safe_nonneg_int(now_ts, default=0)}")
    decision = extract_intent_side(intent)
    symbol = _clean_intent_symbol(intent.get("symbol", "") if isinstance(intent, dict) else "")
    raw_venue = intent.get("venue", "binance") if isinstance(intent, dict) else "binance"
    if raw_venue is None or (isinstance(raw_venue, str) and not raw_venue.strip()):
        venue = "binance"
    elif (clean_v := clean_scalar_str(raw_venue)) is None:
        venue = "__invalid__"
    else:
        venue = normalize_venue_name(clean_v) or clean_v.lower()
    return intent_id, decision, symbol, venue


def is_non_fiat_btc_quote_instrument(inst: Any) -> bool:
    """Return True when ``inst`` is a crypto instrument quoted in BTC rather than fiat/stablecoin."""
    if not inst:
        return False
    raw_prov = getattr(inst, "provider_name", None) or getattr(inst, "provider", "")
    inst_prov = getattr(raw_prov, "name", raw_prov)
    if is_equity_venue(inst_prov):
        return False
    inst_sym_u = _clean_intent_symbol(getattr(inst, "symbol", ""))
    inst_quote_u = _clean_intent_symbol(getattr(inst, "quote", ""))
    return symbol_quote_asset(inst_sym_u) == "BTC" or inst_quote_u == "BTC"


def validate_and_build_intent_dict(
    decision: dict[str, Any],
    *,
    source: str,
    min_confidence: float,
    min_notional_usd: float,
    max_notional_usd: float,
    existing_intents: list[dict[str, Any]],
    now_ts: float,
    resolve_instrument_fn: Optional[Callable[[str, str], Any]] = None,
    max_mode: Optional[str] = None,
    log: Optional[Any] = None,
) -> Optional[dict[str, Any]]:
    """Validate an LLM trading decision and build a canonical TradeIntentRecord dict, or return None if non-actionable."""
    if not isinstance(decision, dict):
        return None
    side = extract_intent_side(decision)
    if side not in ("BUY", "SELL"):
        raw_dec = (
            str(decision.get("decision") or decision.get("action") or decision.get("side") or "HOLD").strip().upper()
            or "HOLD"
        )
        if log:
            log.info("[INTENT GATEWAY] %s decision is %s (no intent).", source, raw_dec)
        return None
    symbol = _clean_intent_symbol(decision.get("symbol", ""))
    venue = normalize_venue_name(clean_scalar_str(decision.get("venue")) or "")
    if not symbol or not venue:
        if log:
            log.warning("[INTENT GATEWAY] %s %s decision missing symbol/venue (no intent).", source, side)
        return None
    min_conf = clean_scale(min_confidence, 1.0, allow_percent=True)
    confidence = clean_scale(decision.get("confidence"), 0.0, allow_percent=True)
    if confidence < min_conf:
        if log:
            log.info(
                "[INTENT GATEWAY] %s %s %s confidence %.2f < %.2f (no intent).",
                source, side, symbol, confidence, min_conf,
            )
        return None
    notional = _resolve_clamped_intent_notional(decision, min_notional_usd, max_notional_usd)
    if notional is None:
        if log:
            log.warning(
                "[INTENT GATEWAY] %s %s %s missing or non-positive notional USD (no intent).",
                source, side, symbol,
            )
        return None
    if callable(resolve_instrument_fn):
        inst = resolve_instrument_fn(symbol, venue)
        inst_sym_u = _clean_intent_symbol(getattr(inst, "symbol", ""))
        if not inst or is_non_fiat_btc_quote_instrument(inst):
            if log:
                log.warning(
                    "[INTENT GATEWAY] %s %s %s on %s unknown, ambiguous, or non-fiat instrument (no intent).",
                    source, side, symbol, venue,
                )
            return None
        if inst_sym_u:
            symbol = inst_sym_u
    intent_id = generate_unique_intent_id(now_ts, side, symbol, existing_intents)
    urgency = clean_upper_scalar(decision.get("urgency")) or "NORMAL"
    thesis = clean_scalar_str(decision.get("thesis")) or ""
    order_type = extract_intent_order_type(decision)
    limit_px = extract_intent_limit_price(decision, default=None) if order_type == "LIMIT" else None
    inv_px, tgt_px = extract_intent_trade_plan_levels(decision, side=side, ref_price=limit_px)
    replace_open = _parse_bool_flag(decision.get("replace_open_orders"), default=False)
    cancel_bot_open = _parse_bool_flag(decision.get("cancel_bot_open_orders"), default=False)
    cancel_oids = extract_intent_cancel_order_ids(decision)
    rec = {
        "intent_id": intent_id,
        "timestamp": now_ts,
        "decision": side,
        "symbol": symbol,
        "venue": venue,
        "confidence": confidence,
        "suggested_notional_usd": notional,
        "order_type": order_type,
        "limit_price": limit_px,
        "replace_open_orders": replace_open,
        "urgency": urgency,
        "thesis": thesis,
        "status": "PENDING",
        "execution_payload": None,
        "created_by": clean_scalar_str(source) or "",
        "history": [],
    }
    if inv_px is not None:
        rec["invalidation_px"] = inv_px
    if tgt_px is not None:
        rec["target_px"] = tgt_px
    if (hor_h := extract_intent_horizon_hours(decision)) is not None:
        rec["horizon_h"] = hor_h
    if (mm := clean_scalar_str(max_mode)) is not None:
        rec["max_mode"] = mm
    if (mu := clean_scalar_str(decision.get("model_used"))) is not None:
        rec["model_used"] = mu
    if cancel_bot_open:
        rec["cancel_bot_open_orders"] = True
    if cancel_oids:
        rec["cancel_order_ids"] = cancel_oids
    return rec


def parse_intent_gateway_config(
    conf: dict[str, str],
    *,
    mode: Optional[str] = None,
    max_notional_usd: Optional[float] = None,
    min_confidence: Optional[float] = None,
    notify_enabled: Optional[bool] = None,
) -> tuple[str, float, float, float, bool]:
    """Validate and parse (mode, min_notional_usd, max_notional_usd, min_confidence, notify_enabled) from config."""
    raw_mode = conf_or_override(conf, "autonomous_trading_mode", mode)
    norm_mode = "" if isinstance(raw_mode, bool) else str(raw_mode).strip().lower()
    if norm_mode not in ("off", "shadow", "enforce"):
        raise ValueError(
            f"Invalid autonomous_trading_mode {raw_mode!r}; must be one of ['enforce', 'off', 'shadow']"
        )

    raw_min = conf_or_override(conf, "autonomous_trading_min_notional_usd")
    min_usd = safe_pos_float(raw_min, default=None)
    if min_usd is None:
        raise ValueError(f"Invalid autonomous_trading_min_notional_usd {raw_min!r}")

    raw_max = conf_or_override(conf, "autonomous_trading_max_notional_usd", max_notional_usd)
    max_usd = safe_pos_float(raw_max, default=None)
    if max_usd is None or max_usd < min_usd:
        raise ValueError(
            f"Invalid autonomous_trading_max_notional_usd {raw_max!r}; "
            f"must be >= autonomous_trading_min_notional_usd ({min_usd})"
        )

    raw_conf = conf_or_override(conf, "autonomous_trading_min_confidence", min_confidence)
    conf_val = safe_nonneg_float(raw_conf, default=None)
    if conf_val is None or conf_val > 1.0:
        raise ValueError(
            f"Invalid autonomous_trading_min_confidence {raw_conf!r}; must be in [0.0, 1.0]"
        )

    notify_flag = parse_notify_enabled_flag(notify_enabled, conf, "autonomous_trading_notify")
    return norm_mode, min_usd, max_usd, conf_val, notify_flag


def _intent_executed_notional_usd(item: dict[str, Any]) -> float:
    if (val := safe_pos_float(item.get("filled_notional_usd"), default=None)) is not None:
        return val
    ep = item.get("execution_payload")
    if isinstance(ep, dict):
        for key in ("cummulativeQuoteQty", "cumulativeQuoteQty", "cost", "filledValue", "quote_notional", "notional_usd"):
            if (val := safe_pos_float(ep.get(key), default=None)) is not None:
                return val
    if (val := safe_pos_float(item.get("clamped_notional_usd"), default=None)) is not None:
        return val
    hist = item.get("history")
    if isinstance(hist, list) and hist and isinstance(hist[-1], dict):
        if (val := safe_pos_float(hist[-1].get("notional_usd"), default=None)) is not None:
            return val
    return safe_pos_float(extract_intent_notional(item, default=None), default=0.0) or 0.0


def parse_pos_epoch_s(raw: Any) -> Optional[float]:
    """Parse an epoch-seconds, epoch-ms, or ISO-8601 timestamp into positive UTC seconds."""
    parsed = _parse_epoch_or_iso_s(raw)
    return _ts_ms_to_s(parsed) if (parsed is not None and parsed > 0.0) else None


def extract_intent_timestamp_s(
    intent: Any,
    *,
    keys: tuple[str, ...] = _INTENT_TIMESTAMP_KEYS,
) -> tuple[Any, float]:
    """Extract ``(raw_ts, epoch_sec)`` from an intent record across canonical timestamp fields."""
    if not isinstance(intent, dict):
        return None, 0.0
    ts_raw: Any = None
    for tk in keys:
        cand = intent.get(tk)
        if cand is None:
            continue
        if ts_raw is None:
            ts_raw = cand
        if (parsed_s := parse_pos_epoch_s(cand)) is not None:
            return cand, parsed_s
    return ts_raw, 0.0


def extract_intent_primary_order_id(
    intent: Any,
    default: Any = None,
    *,
    include_top_level: bool = False,
) -> Any:
    """Extract the primary venue order ID from ``execution_payload``, explicit top-level keys, or ``history[-1]``."""
    if not isinstance(intent, dict):
        return default
    exec_pay = intent.get("execution_payload") if isinstance(intent.get("execution_payload"), dict) else {}
    oid = extract_order_id(exec_pay, default=None)
    if oid is not None:
        return oid
    if include_top_level:
        for k in _EXPLICIT_INTENT_OID_KEYS:
            if (cand := extract_order_id(intent.get(k), default=None)) is not None:
                return cand
    if isinstance(hist := intent.get("history"), list) and hist and isinstance(hist[-1], dict):
        last_h = hist[-1]
        oid = extract_order_id(last_h.get("order_id") or last_h.get("payload"), default=None)
        if oid is not None:
            return oid
    return default


def _symbols_match_intent_scope_and_quote_impl(
    sym_a: Any,
    sym_b: Any,
    venue_a: Any,
    venue_b: Any,
    allow_missing_symbol: bool,
) -> bool:
    sa = _clean_intent_symbol(sym_a)
    sb = _clean_intent_symbol(sym_b)
    va = normalize_venue_name(clean_scalar_str(venue_a) or "") if venue_a else ""
    vb = normalize_venue_name(clean_scalar_str(venue_b) or "") if venue_b else ""
    if va and vb and va != vb:
        return False
    if (va or sa) and (vb or sb):
        eq_a = is_equity_venue_or_suffix(sa, va) if (va or vb) else has_equity_suffix(sa)
        eq_b = is_equity_venue_or_suffix(sb, vb) if (va or vb) else has_equity_suffix(sb)
        if eq_a != eq_b:
            return False
    if not sa or not sb:
        return bool(allow_missing_symbol)
    return sa == sb or (
        same_canonical_base_asset(sa, sb)
        and compatible_quote_suffixes(sa) == compatible_quote_suffixes(sb)
    )


@lru_cache(maxsize=2048)
def _symbols_match_intent_scope_and_quote_cached(
    sym_a: str,
    sym_b: str,
    venue_a: str,
    venue_b: str,
    allow_missing_symbol: bool,
) -> bool:
    return _symbols_match_intent_scope_and_quote_impl(
        sym_a, sym_b, venue_a, venue_b, allow_missing_symbol
    )


def symbols_match_intent_scope_and_quote(
    sym_a: Any,
    sym_b: Any,
    *,
    venue_a: Any = None,
    venue_b: Any = None,
    allow_missing_symbol: bool = False,
) -> bool:
    """Return True when two symbols/venues share the same asset class and compatible quote currency."""
    if (
        (sym_a is None or type(sym_a) is str)
        and (sym_b is None or type(sym_b) is str)
        and (venue_a is None or type(venue_a) is str)
        and (venue_b is None or type(venue_b) is str)
    ):
        return _symbols_match_intent_scope_and_quote_cached(
            sym_a or "", sym_b or "", venue_a or "", venue_b or "", bool(allow_missing_symbol)
        )
    return _symbols_match_intent_scope_and_quote_impl(
        sym_a, sym_b, venue_a, venue_b, bool(allow_missing_symbol)
    )


def order_matches_intent_client_id(
    order: Any,
    intent: dict[str, Any],
    *,
    expected_cids: Optional[set[str]] = None,
    allow_untagged_ai_actor: bool = False,
) -> bool:
    """Return True when ``order`` carries a venue-native client ID belonging to ``intent``."""
    if not isinstance(order, dict):
        return False
    cids = expected_cids if expected_cids is not None else intent_venue_client_order_ids(intent)
    oo_cid = (extract_client_order_id(order, default="") or "").strip().lower()
    if oo_cid:
        return bool(cids and oo_cid in cids)
    return bool(allow_untagged_ai_actor and resolve_order_actor(order) == "AI")


__all__ = [
    "_clean_intent_symbol",
    "_extract_intent_id",
    "_intent_executed_notional_usd",
    "_intent_order_ids",
    "_intent_venue_symbol_side_key",
    "_record_intent_state",
    "_resolve_clamped_intent_notional",
    "ai_intent_client_order_id",
    "append_intent_history",
    "collect_canceled_order_ids",
    "collect_known_ai_order_ids",
    "decision_has_cancel_directive",
    "eval_doc_was_actionable",
    "extract_intent_cancel_order_ids",
    "extract_intent_horizon_hours",
    "extract_intent_limit_price",
    "extract_intent_notional",
    "extract_intent_order_type",
    "extract_intent_primary_order_id",
    "extract_intent_side",
    "extract_intent_timestamp_s",
    "extract_intent_trade_plan_levels",
    "generate_unique_intent_id",
    "intent_venue_client_order_ids",
    "is_non_fiat_btc_quote_instrument",
    "normalize_intent_fields",
    "normalize_trading_decisions",
    "order_matches_intent_client_id",
    "parse_intent_gateway_config",
    "parse_pos_epoch_s",
    "prune_reconciled_terminal_open_orders",
    "resolve_hold_cancel_target",
    "symbols_match_intent_scope_and_quote",
    "validate_and_build_intent_dict",
]
