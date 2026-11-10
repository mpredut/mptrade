"""Durable AI intent state persistence, schema normalization, and ID correlation."""
from __future__ import annotations

import hashlib
import os
import re
import threading
from contextlib import contextmanager
from typing import Any, Callable, Optional

from botcore import (
    _parse_bool_flag,
    clean_json_value,
    clean_scale,
    conf_or_override,
    normalize_side,
    parse_notify_enabled_flag,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
)
from cachepaths import (
    canonical_base_asset,
    compatible_quote_suffixes,
    is_equity_venue_or_suffix,
    normalize_venue_name,
    resolve_cachedb_filepath,
)
from state_io import generation_lock_path, load_json_state, save_json_state
from .order_ohlc_utils import all_venue_native_client_order_ids, extract_order_id

_CURRENCY_STRIP_RE = re.compile(r"[\$,\s]|(?:USD[CT]?)\b", re.IGNORECASE)
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


def _extract_intent_id(record: Any, default: str = "") -> str:
    """Return a clean non-boolean intent_id (falling back to ``id``) from ``record``."""
    if not isinstance(record, dict):
        return default
    for key in ("intent_id", "id"):
        raw = record.get(key)
        if raw is not None and not isinstance(raw, bool):
            clean = str(raw).strip()
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
    raw = "" if isinstance(intent_id, (bool, list, tuple, dict, set, frozenset)) else str(intent_id or "").strip()
    if not raw:
        return None
    cid_body = (
        f"{raw[:18]}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:6]}_{raw[-7:]}"
        if len(raw) > 33
        else raw
    )
    return f"AI_{cid_body}"


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
    return ids


def collect_known_ai_order_ids(
    intents: Any,
    initial_ids: Optional[set[str]] = None,
) -> set[str]:
    """Collect venue order IDs (``execution_payload`` / ``history``) and venue-native AI client IDs across intents."""
    ai_oids: set[str] = set(initial_ids or ())
    if isinstance(intents, dict):
        intents = intents.get("intents")
    if not isinstance(intents, list):
        return ai_oids
    for item in intents:
        if isinstance(item, dict):
            ai_oids.update(_intent_order_ids(item))
    return ai_oids


def _parse_currency_pos_float(raw: Any, default: Any = None) -> Any:
    if isinstance(raw, str):
        raw = _CURRENCY_STRIP_RE.sub("", raw).strip()
    return safe_pos_float(raw, default)


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
            val = _parse_currency_pos_float(data.get(nk), None)
            if val is not None:
                return val
    return default


def _resolve_clamped_intent_notional(
    data: dict[str, Any],
    min_notional_usd: float,
    max_notional_usd: float,
) -> float:
    """Resolve and clamp suggested notional USD; explicit non-positive numeric values clamp to min_notional_usd."""
    min_usd = safe_nonneg_float(min_notional_usd, default=0.0) or 0.0
    max_usd = safe_nonneg_float(max_notional_usd, default=min_usd) or min_usd
    pos = extract_intent_notional(data, None)
    if pos is not None:
        return max(min_usd, min(pos, max_usd))
    if isinstance(data, dict):
        for nk in _NOTIONAL_KEYS:
            raw = data.get(nk)
            if raw is None or isinstance(raw, bool):
                continue
            if isinstance(raw, str):
                raw = _CURRENCY_STRIP_RE.sub("", raw).strip()
            num = safe_float(raw, None)
            if num is not None and num <= 0.0:
                return min_usd
    return max(min_usd, max_usd)


def extract_intent_limit_price(data: dict[str, Any], default: Any = None) -> Any:
    """Extract and parse positive passive limit price across aliases and currency strings."""
    if not isinstance(data, dict):
        return default
    keys: tuple[str, ...] = ("limit_price", "target_price", "order_price", "limit_px")
    if str(data.get("order_type") or data.get("type") or "").strip().upper() == "LIMIT":
        keys = (*keys, "price")
    for pk in keys:
        if data.get(pk) is not None:
            val = _parse_currency_pos_float(data.get(pk), None)
            if val is not None:
                return val
    return default


def extract_intent_order_type(data: dict[str, Any]) -> str:
    """Resolve normalized order execution type ('MARKET' or 'LIMIT') for an intent."""
    if not isinstance(data, dict):
        return "MARKET"
    raw_type = data.get("order_type") or data.get("type")
    norm_type = "" if isinstance(raw_type, bool) or raw_type is None else str(raw_type).strip().upper()
    limit_px = extract_intent_limit_price(data, default=None)
    if norm_type == "MARKET":
        return "MARKET"
    if norm_type == "LIMIT" or limit_px is not None:
        return "LIMIT"
    return "MARKET"


def parse_intent_confidence(raw_conf: Any, default: float = 0.0) -> float:
    """Parse confidence into [0.0, 1.0], normalizing percentage strings or 1..100 values."""
    return clean_scale(raw_conf, default, allow_percent=True)


def _intent_venue_symbol_side_key(
    venue: Any, symbol: Any, side: Any
) -> tuple[str, str, tuple[str, ...], str]:
    s_raw = str(symbol or "").strip().upper()
    v_norm = normalize_venue_name(venue)
    base = canonical_base_asset(s_raw) or s_raw
    if base and is_equity_venue_or_suffix(s_raw, v_norm):
        base = f"EQ:{base}"
    return (
        v_norm,
        base,
        compatible_quote_suffixes(s_raw),
        str(side or "").strip().upper(),
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
            if isinstance(raw_sym, str) and raw_sym.strip():
                d["symbol"] = raw_sym.strip().upper()
            elif isinstance(raw_sym, bool):
                d.pop("symbol", None)
            v_norm = normalize_venue_name(d.get("venue"))
            if v_norm:
                d["venue"] = v_norm
            if "confidence" in d and d.get("confidence") is not None:
                d["confidence"] = parse_intent_confidence(d.get("confidence"), default=0.0)
            extracted_usd = extract_intent_notional(d, default=None)
            if extracted_usd is not None:
                d["suggested_notional_usd"] = extracted_usd
            limit_px = extract_intent_limit_price(d, default=None)
            order_type = extract_intent_order_type(d)
            d["order_type"] = order_type
            if limit_px is not None and order_type == "LIMIT":
                d["limit_price"] = limit_px
            if "replace_open_orders" in d:
                d["replace_open_orders"] = _parse_bool_flag(d.get("replace_open_orders"), False)
            if "cancel_bot_open_orders" in d:
                d["cancel_bot_open_orders"] = _parse_bool_flag(d.get("cancel_bot_open_orders"), False)
            candidates.append(d)

    def _sort_key(d: dict[str, Any]) -> tuple[int, float]:
        side = str(d.get("decision") or "HOLD").strip().upper()
        is_action = 1 if side in ("BUY", "SELL") else 0
        conf = parse_intent_confidence(d.get("confidence"), default=0.0)
        return (-is_action, -conf)

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
    base_id = f"intent_{int(now_ts)}_{side.lower()}_{symbol.lower()}"
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
    intent_id = _extract_intent_id(intent, default=f"intent_{int(now_ts)}")
    decision = extract_intent_side(intent)
    raw_sym = intent.get("symbol", "")
    symbol = "" if isinstance(raw_sym, bool) or raw_sym is None else str(raw_sym).strip().upper()
    raw_venue = intent.get("venue", "binance")
    if isinstance(raw_venue, bool):
        venue = "__invalid__"
    elif raw_venue is None or not str(raw_venue).strip():
        venue = "binance"
    else:
        venue = normalize_venue_name(str(raw_venue)) or str(raw_venue).strip().lower()
    return intent_id, decision, symbol, venue


def validate_and_build_intent_dict(
    decision: dict[str, Any],
    *,
    source: str,
    min_confidence: float,
    min_notional_usd: float,
    max_notional_usd: float,
    existing_intents: list[dict[str, Any]],
    now_ts: float,
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
    raw_sym = decision.get("symbol", "")
    symbol = "" if isinstance(raw_sym, bool) or raw_sym is None else str(raw_sym).strip().upper()
    raw_venue = decision.get("venue", "")
    venue = "" if isinstance(raw_venue, bool) or raw_venue is None else normalize_venue_name(str(raw_venue))
    if not symbol or not venue:
        if log:
            log.warning("[INTENT GATEWAY] %s %s decision missing symbol/venue (no intent).", source, side)
        return None
    confidence = parse_intent_confidence(decision.get("confidence"), 0.0)
    if confidence < min_confidence:
        if log:
            log.info(
                "[INTENT GATEWAY] %s %s %s confidence %.2f < %.2f (no intent).",
                source, side, symbol, confidence, min_confidence,
            )
        return None
    notional = _resolve_clamped_intent_notional(decision, min_notional_usd, max_notional_usd)
    intent_id = generate_unique_intent_id(now_ts, side, symbol, existing_intents)
    raw_urg = decision.get("urgency", "NORMAL")
    urgency = (
        "NORMAL"
        if isinstance(raw_urg, bool) or raw_urg is None or not str(raw_urg).strip()
        else str(raw_urg).strip().upper()
    )
    raw_thesis = decision.get("thesis", "")
    thesis = "" if isinstance(raw_thesis, bool) or raw_thesis is None else str(raw_thesis).strip()
    order_type = extract_intent_order_type(decision)
    limit_px = extract_intent_limit_price(decision, default=None) if order_type == "LIMIT" else None
    replace_open = _parse_bool_flag(decision.get("replace_open_orders"), default=False)
    cancel_bot_open = _parse_bool_flag(decision.get("cancel_bot_open_orders"), default=False)
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
        "created_by": str(source or "").strip(),
        "history": [],
    }
    if cancel_bot_open:
        rec["cancel_bot_open_orders"] = True
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


@contextmanager
def durable_intents_lock(intents_file: str):
    """Reentrant (per thread) cross-process lock serializing read-modify-write of the durable intents file."""
    from lock.cooldown import FileLock

    key = generation_lock_path(str(resolve_cachedb_filepath(os.fspath(intents_file))))
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


__all__ = [
    "_extract_intent_id",
    "_intent_order_ids",
    "_intent_venue_symbol_side_key",
    "_record_intent_state",
    "_resolve_clamped_intent_notional",
    "ai_intent_client_order_id",
    "append_intent_history",
    "collect_known_ai_order_ids",
    "durable_intents_lock",
    "extract_intent_limit_price",
    "extract_intent_notional",
    "extract_intent_order_type",
    "extract_intent_side",
    "generate_unique_intent_id",
    "intent_venue_client_order_ids",
    "load_durable_intents_state",
    "normalize_intent_fields",
    "normalize_trading_decisions",
    "parse_intent_confidence",
    "parse_intent_gateway_config",
    "save_durable_intents_state",
    "upsert_durable_intent",
    "validate_and_build_intent_dict",
]
