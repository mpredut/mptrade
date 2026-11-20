"""Durable AI intent state persistence, schema normalization, and ID correlation."""
from __future__ import annotations

import os
import re
import threading
from contextlib import contextmanager
from typing import Any, Callable, Optional

from botcore import (
    FileLock,
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
    resolve_cachedb_filepath,
    same_canonical_base_asset,
    symbol_quote_asset,
)
from instrument_registry import resolve_symbol_identity
from state_io import generation_lock_path, load_json_state, save_json_state
from .order_id_context import (
    all_venue_native_client_order_ids,
    derive_ai_intent_client_order_id,
    extract_client_order_id,
    resolve_order_actor,
)
from .order_ohlc_utils import extract_executed_qty
from .order_status_ssot import CANCEL_OR_EXPIRE_VENUE_STATUSES

_CURRENCY_STRIP_RE = re.compile(r"[\$,\s]|(?:USD[CT]?)\b", re.IGNORECASE)
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
_INTENTS_LOCK_LOCAL = threading.local()


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
) -> set[str]:
    """Collect venue order IDs (``execution_payload`` / ``history``) and venue-native AI client IDs across intents."""
    ai_oids: set[str] = set()
    if isinstance(initial_ids, str):
        if clean_s := clean_order_id(initial_ids):
            ai_oids.add(clean_s)
    elif initial_ids is not None and not isinstance(initial_ids, (bool, bytes, dict)) and hasattr(initial_ids, "__iter__"):
        for item_id in initial_ids:
            if clean_s := clean_order_id(item_id):
                ai_oids.add(clean_s)
    if isinstance(intents, dict):
        intents = intents.get("intents")
    if not isinstance(intents, list):
        return ai_oids
    for item in intents:
        if isinstance(item, dict):
            ai_oids.update(_intent_order_ids(item))
    return ai_oids


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


def decision_has_cancel_directive(d: Any) -> bool:
    """Return True when a trading decision dict specifies bot/order cancellation."""
    return isinstance(d, dict) and (
        _parse_bool_flag(d.get("cancel_bot_open_orders"), False)
        or bool(extract_intent_cancel_order_ids(d))
    )


def resolve_hold_cancel_target(
    decision: Any,
    *,
    min_confidence: float,
    resolve_instrument_fn: Callable[[str, str], Any],
) -> tuple[Any, list[str], bool]:
    """Resolve (instrument, target_order_ids, only_bot_orders) for a HOLD decision with targeted cancellation."""
    if not isinstance(decision, dict):
        return None, [], True
    raw_dec = clean_upper_scalar(decision.get("decision") or decision.get("action") or decision.get("side")) or "HOLD"
    if extract_intent_side(decision) in ("BUY", "SELL") or raw_dec != "HOLD":
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
    return inst, target_oids, not _parse_bool_flag(decision.get("replace_open_orders"), default=False)


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
    for item in candidates:
        venue_norm, sym_canon, quotes, side_norm = _intent_venue_symbol_side_key(
            item.get("venue"), item.get("symbol"), item.get("decision") or "HOLD"
        )
        key = (venue_norm, sym_canon, quotes, "") if sym_canon else (venue_norm, "", (), side_norm)
        if key not in seen_keys:
            seen_keys.add(key)
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


def parse_intent_circuit_breaker_config(conf: dict[str, str]) -> dict[str, Any]:
    """Parse optional asymmetric ``by=AI`` circuit-breaker parameters from ``order_guard.conf``."""
    cfg = conf if isinstance(conf, dict) else {}
    return {
        "max_daily_buys_per_symbol": safe_nonneg_int(
            cfg.get("autonomous_trading_max_daily_buys_per_symbol"), default=6
        ),
        "max_hourly_replaces_per_symbol": safe_nonneg_int(
            cfg.get("autonomous_trading_max_hourly_replaces_per_symbol"), default=3
        ),
        "replace_min_price_delta_pct": safe_nonneg_float(
            cfg.get("autonomous_trading_replace_min_price_delta_pct"), default=0.50
        ),
        "max_daily_buy_notional_usd": safe_nonneg_float(
            cfg.get("autonomous_trading_max_daily_buy_notional_usd"), default=6000.0
        ),
    }


def _intent_executed_notional_usd(item: dict[str, Any]) -> float:
    for key in ("filled_notional_usd", "clamped_notional_usd"):
        if (val := safe_pos_float(item.get(key), default=None)) is not None:
            return val
    ep = item.get("execution_payload")
    if isinstance(ep, dict):
        for key in ("cummulativeQuoteQty", "cumulativeQuoteQty", "quote_notional", "cost", "notional_usd"):
            if (val := safe_pos_float(ep.get(key), default=None)) is not None:
                return val
    return safe_pos_float(extract_intent_notional(item, default=None), default=0.0) or 0.0


def evaluate_ai_intent_circuit_breaker(
    intent: dict[str, Any],
    existing_intents: list[dict[str, Any]],
    *,
    decision: str,
    symbol: str,
    venue: str,
    price: float,
    effective_usd: float,
    min_notional_usd: float,
    now_ts: float,
    max_daily_buys_per_symbol: int = 6,
    max_hourly_replaces_per_symbol: int = 3,
    replace_min_price_delta_pct: float = 0.50,
    max_daily_buy_notional_usd: float = 6000.0,
) -> tuple[bool, float, str, dict[str, Any]]:
    """Asymmetric safety circuit-breaker strictly on ``by=AI`` intents.

    Returns ``(allowed, adjusted_usd, reason, metrics)``.
    - Never counts deterministic ``by=BOT`` trades.
    - Protective ``SELL`` exits bypass all 24h ``BUY`` count/notional caps.
    - ``urgency == "HIGH"``, ``cancel_bot_open_orders``, and explicit ``cancel_order_ids``
      bypass the 1h micro-replace churn cap; price moves >= ``replace_min_price_delta_pct``
      also bypass the 1h micro-replace churn cap.
    - Smoothly clamps ``BUY`` notional to remaining 24h budget when headroom >= ``min_notional_usd``.
    """
    side = normalize_side(decision) or extract_intent_side(intent)
    cur_iid = _extract_intent_id(intent)
    eff_now = safe_pos_float(now_ts, default=0.0) or 0.0
    eff_px = safe_pos_float(price, default=0.0) or 0.0
    eff_usd = safe_pos_float(effective_usd, default=0.0) or 0.0
    eff_min_usd = safe_pos_float(min_notional_usd, default=10.0) or 10.0
    cap_sym_buys = safe_nonneg_int(max_daily_buys_per_symbol, default=6)
    cap_hr_replaces = safe_nonneg_int(max_hourly_replaces_per_symbol, default=3)
    min_rep_delta_pct = safe_nonneg_float(replace_min_price_delta_pct, default=0.50)
    cap_daily_notional = safe_nonneg_float(max_daily_buy_notional_usd, default=6000.0)

    items = existing_intents if isinstance(existing_intents, list) else []
    all_canceled_oids: set[str] = set()
    for it in items:
        if not isinstance(it, dict) or (cur_iid and _extract_intent_id(it) == cur_iid):
            continue
        for c_id in (it.get("canceled_prior_open_orders") or ()):
            if clean_c := clean_order_id(c_id):
                all_canceled_oids.add(clean_c)

    def _is_counted_executed(it: dict[str, Any]) -> bool:
        if (clean_upper_scalar(it.get("status")) or "") != "EXECUTED":
            return False
        ep = it.get("execution_payload")
        filled_q = extract_executed_qty(ep, default=0.0) if isinstance(ep, dict) else 0.0
        if filled_q <= 0.0:
            filled_q = safe_nonneg_float(it.get("filled_qty"), default=0.0)
        ep_st = (clean_upper_scalar(ep.get("status")) if isinstance(ep, dict) else "") or ""
        if ep_st in CANCEL_OR_EXPIRE_VENUE_STATUSES and filled_q <= 0.0:
            return False
        prim_oid = extract_intent_primary_order_id(it, default="", include_top_level=True)
        if prim_oid and str(prim_oid) in all_canceled_oids and filled_q <= 0.0:
            return False
        return True

    day_cutoff = eff_now - 86400.0
    hour_cutoff = eff_now - 3600.0
    daily_sym_buys = 0
    daily_buy_notional = 0.0
    hourly_replaces = 0
    latest_same_side_px = 0.0
    latest_same_side_ts = 0.0

    for it in items:
        if not isinstance(it, dict) or (cur_iid and _extract_intent_id(it) == cur_iid):
            continue
        _, ts_s = extract_intent_timestamp_s(it)
        if ts_s <= 0.0 or ts_s < day_cutoff or ts_s > eff_now + 60.0:
            continue
        it_side = extract_intent_side(it)
        it_sym = it.get("symbol")
        it_ven = it.get("venue")
        same_inst = symbols_match_intent_scope_and_quote(it_sym, symbol, venue_a=it_ven, venue_b=venue)
        is_exec = (clean_upper_scalar(it.get("status")) or "") == "EXECUTED"

        if same_inst and it_side == side and is_exec:
            if ts_s >= latest_same_side_ts:
                cand_px = (
                    safe_pos_float(it.get("limit_price"), default=None)
                    or safe_pos_float(it.get("requested_price"), default=None)
                    or safe_pos_float(it.get("reference_price"), default=None)
                )
                if cand_px is not None:
                    latest_same_side_px = cand_px
                    latest_same_side_ts = ts_s
            if (
                ts_s >= hour_cutoff
                and _parse_bool_flag(it.get("replace_open_orders"), False)
                and bool(it.get("canceled_prior_open_orders"))
            ):
                hourly_replaces += 1

        if it_side == "BUY" and _is_counted_executed(it):
            daily_buy_notional += _intent_executed_notional_usd(it)
            if same_inst:
                daily_sym_buys += 1

    metrics: dict[str, Any] = {
        "daily_sym_buys": daily_sym_buys,
        "daily_buy_notional_usd": round(daily_buy_notional, 2),
        "hourly_replaces": hourly_replaces,
    }

    urgency = (clean_upper_scalar(intent.get("urgency")) or "NORMAL")
    replace_open = _parse_bool_flag(intent.get("replace_open_orders"), default=False)
    cancel_bot_open = _parse_bool_flag(intent.get("cancel_bot_open_orders"), default=False)
    target_oids = extract_intent_cancel_order_ids(intent)

    if (
        replace_open
        and cap_hr_replaces > 0
        and urgency != "HIGH"
        and not cancel_bot_open
        and not target_oids
        and hourly_replaces >= cap_hr_replaces
    ):
        delta_pct: Optional[float] = None
        if latest_same_side_px > 0.0 and eff_px > 0.0:
            delta_pct = abs((eff_px - latest_same_side_px) / latest_same_side_px) * 100.0
            metrics["replace_price_delta_pct"] = round(delta_pct, 4)
        if delta_pct is None or delta_pct < min_rep_delta_pct:
            delta_lbl = f"{delta_pct:.2f}%" if delta_pct is not None else "n/a"
            return (
                False,
                0.0,
                f"ai_hourly_replace_churn_cap({symbol}@{venue}:{hourly_replaces}/{cap_hr_replaces}, delta={delta_lbl}<{min_rep_delta_pct:.2f}%)",
                metrics,
            )

    if side != "BUY":
        return True, eff_usd, "", metrics

    if cap_sym_buys > 0 and daily_sym_buys >= cap_sym_buys:
        return (
            False,
            0.0,
            f"ai_daily_buy_count_cap({symbol}@{venue}:{daily_sym_buys}/{cap_sym_buys})",
            metrics,
        )

    if cap_daily_notional > 0.0 and (daily_buy_notional + eff_usd) > cap_daily_notional + 1e-6:
        remaining = max(0.0, cap_daily_notional - daily_buy_notional)
        if remaining >= eff_min_usd:
            clamped_usd = round(remaining, 2)
            return (
                True,
                clamped_usd,
                f"ai_daily_buy_notional_clamped(${daily_buy_notional:.0f}+${clamped_usd:.0f}/${cap_daily_notional:.0f})",
                metrics,
            )
        return (
            False,
            0.0,
            f"ai_daily_buy_notional_cap(${daily_buy_notional:.0f}/${cap_daily_notional:.0f})",
            metrics,
        )

    return True, eff_usd, "", metrics


@contextmanager
def durable_intents_lock(intents_file: str):
    """Reentrant (per thread) cross-process lock serializing read-modify-write of the durable intents file."""
    raw_p = clean_scalar_str(os.fspath(intents_file)) if isinstance(intents_file, (str, os.PathLike)) else None
    if raw_p is None:
        raise ValueError(f"Invalid durable intents_file: {intents_file!r}")
    key = generation_lock_path(str(resolve_cachedb_filepath(raw_p)))
    depths = getattr(_INTENTS_LOCK_LOCAL, "depths", None)
    if depths is None:
        depths = _INTENTS_LOCK_LOCAL.depths = {}
    if depths.get(key):
        depths[key] += 1
        try:
            yield
        finally:
            depths[key] -= 1
        return
    with FileLock(key):
        depths[key] = 1
        try:
            yield
        finally:
            depths.pop(key, None)


def load_durable_intents_state(
    intents_file: str,
    reconcile_fn: Optional[Callable[[list[dict[str, Any]]], Any]] = None,
) -> list[dict[str, Any]]:
    """Load all trade intents from the durable JSON queue with `.previous` crash recovery."""
    with durable_intents_lock(intents_file):
        data = load_json_state(
            intents_file,
            default_factory=list,
            fail_closed=False,
            label="autonomous_trade_intents",
            root_type=list,
        )
        items = [item for item in data if isinstance(item, dict)]
        if callable(reconcile_fn):
            reconcile_fn(items)
        return items


def save_durable_intents_state(
    intents_file: str,
    intents: list[dict[str, Any]],
    log: Optional[Any] = None,
) -> bool:
    """Transactionally persist all intents to disk using `.previous` staging and atomic replace."""
    try:
        cleaned = clean_json_value(intents if isinstance(intents, list) else [])
        with durable_intents_lock(intents_file):
            save_json_state(intents_file, cleaned, indent=2)
        return True
    except Exception as e:
        if log:
            log.error("Failed saving intents to %s: %s", intents_file, e)
        return False


def upsert_durable_intent(
    intents_file: str,
    intent: dict[str, Any],
    *,
    reconcile_fn: Optional[Callable[[list[dict[str, Any]]], Any]] = None,
    log: Optional[Any] = None,
) -> bool:
    """Atomically merge one intent (by ``intent_id``) into the freshly reloaded durable intents file."""
    if not isinstance(intent, dict):
        return False
    iid = _extract_intent_id(intent)
    with durable_intents_lock(intents_file):
        intents = load_durable_intents_state(intents_file, reconcile_fn=reconcile_fn)
        idx = next(
            (k for k in range(len(intents) - 1, -1, -1) if iid and _extract_intent_id(intents[k]) == iid),
            None,
        )
        if idx is None:
            intents.append(intent)
        else:
            intents[idx] = intent
        return save_durable_intents_state(intents_file, intents, log=log)


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


def symbols_match_intent_scope_and_quote(
    sym_a: Any,
    sym_b: Any,
    *,
    venue_a: Any = None,
    venue_b: Any = None,
    allow_missing_symbol: bool = False,
) -> bool:
    """Return True when two symbols/venues share the same asset class and compatible quote currency."""
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
    "_intent_order_ids",
    "_intent_venue_symbol_side_key",
    "_record_intent_state",
    "_resolve_clamped_intent_notional",
    "ai_intent_client_order_id",
    "append_intent_history",
    "collect_known_ai_order_ids",
    "decision_has_cancel_directive",
    "durable_intents_lock",
    "evaluate_ai_intent_circuit_breaker",
    "extract_intent_cancel_order_ids",
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
    "load_durable_intents_state",
    "normalize_intent_fields",
    "normalize_trading_decisions",
    "order_matches_intent_client_id",
    "parse_intent_circuit_breaker_config",
    "parse_intent_gateway_config",
    "parse_pos_epoch_s",
    "resolve_hold_cancel_target",
    "save_durable_intents_state",
    "symbols_match_intent_scope_and_quote",
    "upsert_durable_intent",
    "validate_and_build_intent_dict",
]
