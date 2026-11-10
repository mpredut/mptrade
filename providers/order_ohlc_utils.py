"""Stateless order, fill, client-order-ID, and OHLC normalization utilities."""

import hashlib
import re
import time
from typing import Any, List, Optional

from botcore import (
    _parse_bool_flag,
    clean_order_id,
    exact_pos_int,
    normalize_epoch_sec,
    normalize_side,
    obj_attr,
    resolve_now_ts,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
)
from cachepaths import canonical_base_asset


class ProviderError(Exception):
    """Venue-neutral error raised by strict strategy-execution adapters."""


_TIMESTAMP_KEYS = (
    "timestamp", "time", "updateTime", "transactTime", "opentm", "closetm", "workingTime", "T", "E", "ts", "ts_ms",
)
_ORDER_ID_KEYS = ("orderId", "order_id", "ordertxid", "txid", "oid", "i", "id")
_PRICE_KEYS = ("price", "px", "limitPx", "fillPrice", "limit", "limitPrice", "averagePrice", "avgPrice", "rate", "L", "p")
_QTY_KEYS = (
    "qty", "quantity", "executedQty", "executed_qty", "filledQuantity", "filledQty", "filled_qty", "vol_exec",
    "sz", "vol", "origQty", "origSz", "orig_qty", "requested_qty", "orderedQuantity", "amount_shares", "l", "z", "q",
)
_ORIG_QTY_KEYS = (
    "origQty", "orig_qty", "origSz", "vol", "orderedQuantity", "original_qty", "requested_qty",
    "order_qty", "qty", "amount", "quantity", "sz", "amount_shares", "q",
)
_EXEC_QTY_KEYS = ("executedQty", "executed_qty", "filledQuantity", "filledQty", "vol_exec", "filled_qty", "z")
_UNFILLED_ORDER_STATUSES = frozenset({
    "NEW", "OPEN", "RESTING", "WORKING", "PENDING",
    "CANCELED", "CANCELLED", "EXPIRED", "REJECTED", "UNFILLED",
})
_FEE_KEYS = ("commission", "fee", "filled_fee", "n")
_FEE_ASSET_KEYS = ("commissionAsset", "feeToken", "fee_asset", "feeAsset", "N")


def _first_mapped(mapping: Any, keys: tuple[str, ...], parser: Any, default: Any = None) -> Any:
    """Return the first non-None parsed value from ``mapping`` across ``keys``."""
    if not isinstance(mapping, dict):
        return default
    for key in keys:
        if key in mapping and mapping[key] is not None:
            val = parser(mapping[key])
            if val is not None:
                return val
    return default


def _pos_abs_float(value: Any) -> Optional[float]:
    val = abs(safe_float(value, default=0.0))
    return val if val > 0.0 else None


def _nonneg_abs_float(value: Any) -> Optional[float]:
    val = safe_float(value, default=None)
    return abs(val) if val is not None else None


def _parse_epoch_or_iso_s(value: Any) -> Optional[float]:
    numeric = safe_nonneg_float(value, default=None)
    return numeric if numeric is not None else normalize_epoch_sec(value, default=None, allow_zero=True, allow_iso=True)


def normalize_timestamp_ms(value: Any, default: Optional[int] = None) -> Optional[int]:
    """Normalize a raw timestamp (in seconds, milliseconds, or ISO-8601 string) to integer milliseconds."""
    numeric = safe_nonneg_float(value, default=None)
    if numeric is None:
        numeric = _parse_epoch_or_iso_s(value)
    if numeric is None:
        return default
    if numeric == 0.0:
        return 0
    if 1_000_000_000.0 <= numeric < 100_000_000_000.0 or (
        0.0 < numeric < 1_000_000_000.0 and isinstance(value, float) and not numeric.is_integer()
    ):
        return int(round(numeric * 1000.0))
    return int(round(numeric))


def _ts_ms_to_s(ts_ms: float | int) -> float:
    return normalize_epoch_sec(ts_ms, default=0.0, allow_zero=True) or 0.0


def _is_fresh_obs_ts(obs_ts_s: Optional[float], now_s: float, max_age_s: float) -> bool:
    return obs_ts_s is not None and -5.0 <= (now_s - obs_ts_s) <= max_age_s


def extract_timestamp_ms(order: Any, default: Optional[int] = None) -> Optional[int]:
    """Extract and normalize an order/fill timestamp in milliseconds from any venue shape."""
    return _first_mapped(order, _TIMESTAMP_KEYS, lambda v: normalize_timestamp_ms(v, default=None), default)


def extract_timestamp_s(order: Any, default: Optional[float] = None) -> Optional[float]:
    """Extract and normalize an order/fill timestamp in seconds from any venue shape."""
    return _first_mapped(order, _TIMESTAMP_KEYS, lambda v: normalize_epoch_sec(v, allow_zero=True, allow_iso=True), default)


def extract_order_id(order: Any, default: Optional[str] = None) -> Optional[str]:
    """Extract and normalize a venue order ID from any response shape."""
    if order is None or isinstance(order, bool):
        return default
    if isinstance(order, (list, tuple)):
        for item in order:
            if (cleaned := extract_order_id(item)) is not None:
                return cleaned
        return default
    if hasattr(order, "order_id") and hasattr(order, "state"):
        cleaned = clean_order_id(getattr(order, "order_id", None))
        return cleaned if cleaned is not None else default
    if isinstance(order, dict):
        return _first_mapped(order, _ORDER_ID_KEYS, extract_order_id, default)
    cleaned = clean_order_id(order)
    return cleaned if cleaned is not None else default


def resolve_market_flag(force: Any = False, market: Any = False, **kwargs) -> bool:
    """Return True when any market/force flag in args or kwargs is truthy."""
    return any(_parse_bool_flag(v, False) for v in (force, market, kwargs.get("market"), kwargs.get("force")))


def extract_order_side(order: Any) -> str:
    """Extract and normalize order side (``'BUY'`` or ``'SELL'``) from any venue shape."""
    if not isinstance(order, dict):
        return ""
    s_norm = _first_mapped(order, ("side", "S", "dir"), lambda v: normalize_side(v) or None)
    if s_norm:
        return s_norm
    descr = order.get("descr")
    for raw_type in (order.get("type"), descr.get("type") if isinstance(descr, dict) else None):
        if (t_norm := normalize_side(raw_type)) in ("BUY", "SELL"):
            return t_norm
    raw_buyer = order.get("isBuyer")
    if raw_buyer is not None:
        parsed_b = _parse_bool_flag(raw_buyer, None)
        if parsed_b is not None:
            return "BUY" if parsed_b else "SELL"
        if isinstance(raw_buyer, str) and (b_norm := raw_buyer.strip().lower()) in ("b", "buy", "s", "sell"):
            return "BUY" if b_norm.startswith("b") else "SELL"
    for sq_key in ("signed_qty", "orderedQuantity", "quantity"):
        sq = safe_float(order.get(sq_key), default=0.0)
        if sq < 0.0:
            return "SELL"
        if sq > 0.0 and (sq_key in ("signed_qty", "orderedQuantity") or "ticker" in order or "limitPrice" in order):
            return "BUY"
    return ""


_COST_EXEC_KEY_PAIRS = (
    ("cummulativeQuoteQty", "executedQty"),
    ("cummulativeQuoteQty", "qty"),
    ("cummulativeQuoteQty", "quantity"),
    ("quoteQty", "qty"),
    ("quoteQty", "quantity"),
    ("cost", "vol_exec"),
    ("cost", "vol"),
    ("filledValue", "filledQuantity"),
    ("Y", "l"),
    ("Z", "z"),
)


def extract_order_price(order: Any, default: float = 0.0) -> float:
    """Extract order/fill price from any venue shape."""
    if not isinstance(order, dict):
        return default
    val = _first_mapped(order, _PRICE_KEYS, lambda v: safe_pos_float(v, default=None))
    if val is not None:
        return val
    descr = order.get("descr")
    if isinstance(descr, dict) and (val := safe_pos_float(descr.get("price"), default=None)) is not None:
        return val
    for cost_key, exec_key in _COST_EXEC_KEY_PAIRS:
        if cost_key in order and exec_key in order:
            cost_val = _pos_abs_float(order.get(cost_key))
            exec_val = _pos_abs_float(order.get(exec_key))
            if cost_val and exec_val and (res := safe_pos_float(cost_val / exec_val, default=None)) is not None:
                return res
    fills = order.get("fills")
    if isinstance(fills, list) and fills:
        total_quote, total_qty = 0.0, 0.0
        for f in fills:
            if isinstance(f, dict):
                f_px = safe_pos_float(f.get("price", f.get("px", 0.0)), default=None)
                f_qty = _pos_abs_float(f.get("qty", f.get("quantity", f.get("sz", 0.0))))
                if f_px is not None and f_qty is not None:
                    total_quote += f_px * f_qty
                    total_qty += f_qty
        if total_qty > 0 and (res := safe_pos_float(total_quote / total_qty, default=None)) is not None:
            return res
    msg_str = order.get("msg")
    if isinstance(msg_str, str) and "@" in msg_str and msg_str.strip().lower().startswith("filled "):
        if (msg_px := safe_pos_float(msg_str.rsplit("@", 1)[1].strip(), default=None)) is not None:
            return msg_px
    return default


def extract_order_qty(order: Any, default: float = 0.0) -> float:
    """Extract order/fill quantity from any venue shape."""
    return _first_mapped(order, _QTY_KEYS, _pos_abs_float, default)


def extract_executed_qty(order: Any, default: float = 0.0) -> float:
    """Extract executed/filled quantity without falling back to origQty on unfilled/resting orders."""
    if not isinstance(order, dict):
        return default
    val = _first_mapped(order, _EXEC_QTY_KEYS, _nonneg_abs_float)
    if val is not None:
        return val
    filled_sub = order.get("filled")
    val = _first_mapped(filled_sub, ("totalSz", "sz", "qty", *_EXEC_QTY_KEYS), _nonneg_abs_float)
    if val is not None:
        return val
    if "origSz" in order and "sz" in order:
        orig_sz = _nonneg_abs_float(order.get("origSz"))
        rem_sz = _nonneg_abs_float(order.get("sz"))
        if orig_sz is not None and rem_sz is not None:
            return round(max(0.0, orig_sz - rem_sz), 8)
    msg_str = str(order.get("msg") or "").strip()
    if msg_str.lower() == "resting":
        return 0.0
    if msg_str.lower().startswith("filled "):
        if (msg_qty := safe_pos_float(msg_str[7:].split("@", 1)[0].strip(), default=None)) is not None:
            return msg_qty
    raw_st = str(order.get("status") or "").strip().upper()
    if raw_st in _UNFILLED_ORDER_STATUSES or (isinstance(order.get("resting"), dict) and not filled_sub):
        return 0.0
    return extract_order_qty(order, default=default)


def extract_orig_qty(order: Any, default: Optional[float] = None) -> Optional[float]:
    """Extract the venue-accepted original order quantity from any response shape."""
    if hasattr(order, "native") and hasattr(order, "state"):
        order = getattr(order, "native", None)
    if order is None or isinstance(order, bool):
        return default
    for key in (*_ORIG_QTY_KEYS, *_EXEC_QTY_KEYS):
        if (val := _pos_abs_float(obj_attr(order, key))) is not None:
            return val
    return default


def extract_remaining_qty(order: Any, default: float = 0.0) -> float:
    """Extract remaining unfilled quantity for an open/cancelled order across any venue shape."""
    if not isinstance(order, dict):
        return default
    val = _first_mapped(order, ("remainingQty", "remaining_qty", "leavesQty", "leaves_qty"), lambda v: safe_nonneg_float(v, default=None))
    if val is not None:
        return val
    if "origSz" in order and "sz" in order and not any(k in order for k in _EXEC_QTY_KEYS):
        if (sz_f := _nonneg_abs_float(order.get("sz"))) is not None:
            return sz_f
    orig = extract_orig_qty(order, default=0.0) or 0.0
    if orig > 0:
        exec_q = _first_mapped(order, _EXEC_QTY_KEYS, _pos_abs_float, 0.0)
        return max(0.0, orig - exec_q) if exec_q > 0 else orig
    return default


_QUOTE_FEE_ALIASES = {"ZUSD": "USD", "ZEUR": "EUR", "ZUSDT": "USDT"}


def extract_order_fee(
    order: Any,
    base_asset: str = "",
    quote_asset: str = "",
    fill_price: Optional[float] = None,
    require_explicit_asset: bool = False,
) -> Optional[dict]:
    """Extract and normalize order/fill fee metadata across any venue shape."""
    if not isinstance(order, dict):
        return None
    fee = _first_mapped(order, _FEE_KEYS, lambda v: safe_nonneg_float(v, default=None))
    if fee is None:
        return None

    has_asset_key = any(k in order for k in _FEE_ASSET_KEYS)
    fee_asset = _first_mapped(
        order, _FEE_ASSET_KEYS, lambda v: v.strip().upper() if isinstance(v, str) and v.strip() else None, ""
    )

    req_explicit = _parse_bool_flag(require_explicit_asset, False)
    if req_explicit and (not has_asset_key or (fee > 0.0 and not fee_asset)):
        return None

    base_u = base_asset.strip().upper() if isinstance(base_asset, str) else ""
    quote_u = quote_asset.strip().upper() if isinstance(quote_asset, str) else ""
    if not fee_asset and not req_explicit and quote_u:
        fee_asset = quote_u

    px = safe_pos_float(fill_price, default=0.0) or extract_order_price(order, default=0.0)
    base_fee, quote_fee = 0.0, 0.0
    fee_in_quote: Optional[float] = 0.0 if fee == 0.0 else None

    if fee > 0.0:
        fee_base_canon = canonical_base_asset(fee_asset) if fee_asset else ""
        base_canon = canonical_base_asset(base_u) if base_u else ""
        fee_quote_canon = _QUOTE_FEE_ALIASES.get(fee_asset, fee_asset)
        quote_canon = _QUOTE_FEE_ALIASES.get(quote_u, quote_u)
        if base_u and (fee_asset == base_u or (fee_base_canon and fee_base_canon == base_canon)):
            base_fee = fee
            if px > 0.0:
                fee_in_quote = fee * px
        elif (quote_u and (fee_asset == quote_u or (fee_quote_canon and fee_quote_canon == quote_canon))) or (
            not fee_asset and not req_explicit
        ):
            quote_fee = fee
            fee_in_quote = fee

    return {
        "fee": fee,
        "fee_asset": fee_asset,
        "base_fee": base_fee,
        "quote_fee": quote_fee,
        "fee_in_quote": fee_in_quote,
    }


def normalize_order(order: dict) -> dict:
    """Normalize a native order/fill to canonical ``side``, ``price``, ``qty``, and ``timestamp``."""
    return {
        "side": extract_order_side(order),
        "price": extract_order_price(order),
        "qty": extract_executed_qty(order),
        "timestamp": extract_timestamp_ms(order),
    }


_BOT_CID_PREFIX_MODULES: tuple[tuple[str, str], ...] = (
    ("MO_MN", "manual"), ("MO_TA", "tradeall"), ("MO_MT", "monitortrades"), ("MO_RT", "rtrade"), ("MO_SD", "trailing"),
    ("TA_OR", "tradeall"), ("MT_OR", "monitortrades"), ("AG_OR", "assetguardian"), ("RT_OR", "rtrade"),
    ("SD_OR", "trailing"), ("MO_OR", "monitororder"), ("MN_OR", "manual"),
    ("AGS", "assetguardian"), ("AGB", "assetguardian"), ("TRL", "trailing"),
    ("TA", "tradeall"), ("TO", "tradeall"), ("MT", "monitortrades"), ("AG", "assetguardian"),
    ("RT", "rtrade"), ("MO", "monitororder"), ("TS", "trailing"), ("SD", "trailing"), ("MN", "manual"),
)
MANUAL_CID_PREFIXES: tuple[str, ...] = ("and_", "ios_", "web_", "x-", "electron_")
_GENERIC_CID_TOKENS = frozenset({"OR", "MAINTHREAD", "WORKER", "BOT", "ORDER", "RETRY", "OUTBOX"})
_CLIENT_ORDER_ID_KEYS = ("clientOrderId", "client_order_id", "cloid", "cl_ord_id", "userref")


def extract_client_order_id(order: Any, default: Optional[str] = None) -> Optional[str]:
    """Extract and normalize client order ID or user reference from any venue shape."""
    return _first_mapped(order, _CLIENT_ORDER_ID_KEYS, clean_order_id, default)


def resolve_order_actor(order: Any, known_ai_order_ids: Optional[set[str]] = None) -> str:
    """Classify an order or fill record as placed by ``'AI'`` (autonomous executor) or ``'BOT'`` (deterministic fleet)."""
    if not isinstance(order, dict):
        return "BOT"
    explicit = str(order.get("actor") or "").strip().upper()
    if explicit in ("AI", "BOT"):
        return explicit
    oid = extract_order_id(order, default="")
    if oid and known_ai_order_ids and str(oid) in known_ai_order_ids:
        return "AI"
    raw_cid = (extract_client_order_id(order, default="") or "").upper()
    if raw_cid.startswith("AI_") or (raw_cid and known_ai_order_ids and raw_cid.lower() in known_ai_order_ids):
        return "AI"
    return "BOT"


def _first_nonempty_str(mapping: dict, keys: tuple[str, ...], *, lower: bool = False) -> Optional[str]:
    val = _first_mapped(mapping, keys, lambda v: str(v).strip() or None)
    return val.lower() if (val and lower) else val


def resolve_order_origin(
    order: Any,
    known_ai_order_ids: Optional[set[str]] = None,
    audit_origin_map: Optional[dict[str, tuple[Optional[str], Optional[str]]]] = None,
) -> tuple[str, Optional[str], Optional[str]]:
    """Resolve ``(actor, module, rule)`` for an order or fill record."""
    actor = resolve_order_actor(order, known_ai_order_ids)
    if actor == "AI" or not isinstance(order, dict):
        return actor, None, None

    mod = _first_nonempty_str(order, ("module", "origin_module", "bot", "caller"), lower=True)
    rule = _first_nonempty_str(order, ("rule", "decision_rule", "trigger_rule", "kind", "role", "motivation"))
    oid = extract_order_id(order, default="")
    raw_cid = extract_client_order_id(order, default="") or ""

    if isinstance(audit_origin_map, dict):
        for lookup_key in (str(oid) if oid else "", raw_cid):
            if lookup_key and lookup_key in audit_origin_map:
                a_mod, a_rule = audit_origin_map[lookup_key]
                mod = mod or a_mod
                rule = rule or a_rule

    if raw_cid:
        cid_up = raw_cid.upper()
        for pfx, bot_name in _BOT_CID_PREFIX_MODULES:
            if cid_up.startswith(f"{pfx}_") or cid_up == pfx or (pfx in ("AGS", "AGB") and cid_up.startswith(pfx)):
                mod = mod or bot_name
                if not rule:
                    if pfx in ("AGS", "AGB"):
                        rule = "SL_PROTECT" if pfx == "AGS" else "BUY_TIER"
                    else:
                        rem = raw_cid[len(pfx):].lstrip("_-")
                        tokens = [t for t in re.split(r"[_-]+", rem) if t and t.upper() not in _GENERIC_CID_TOKENS]
                        if tokens:
                            cand = (
                                "_".join(tokens[:-1])
                                if len(tokens) > 1 and (tokens[-1].isdigit() or len(tokens[-1]) >= 8)
                                else tokens[0]
                            )
                            if not (len(cand) >= 8 and all(c in "0123456789abcdefABCDEF" for c in cand)) and not cand.isdigit():
                                rule = cand.upper()
                break
        if mod is None and raw_cid.lower().startswith(MANUAL_CID_PREFIXES):
            mod = "manual"
    return "BOT", mod, rule


def format_actor_tag(actor: Optional[str], module: Optional[str] = None, rule: Optional[str] = None) -> str:
    """Format compact actor attribution string (e.g. ``'AI'`` or ``'BOT(module=assetguardian, rule=SL_PANIC)'``)."""
    act_u = str(actor or "").strip().upper()
    if act_u != "BOT":
        return act_u or "BOT"
    parts = [f"{k}={v}" for k, v in (("module", module), ("rule", rule)) if v]
    return f"BOT({', '.join(parts)})" if parts else "BOT"


def normalize_open_order(
    order: Any,
    *,
    venue: str = "",
    symbol: str = "",
    known_ai_order_ids: Optional[set[str]] = None,
    audit_origin_map: Optional[dict[str, tuple[Optional[str], Optional[str]]]] = None,
    now_ts: Optional[float] = None,
    order_ts_map: Optional[dict[str, int]] = None,
) -> Optional[dict]:
    """Normalize a venue open-order dict into a uniform telemetry record."""
    if not isinstance(order, dict):
        return None
    oid = extract_order_id(order)
    px = extract_order_price(order)
    orig_q = extract_orig_qty(order)
    cid = extract_client_order_id(order)
    orig_f = safe_nonneg_float(orig_q, default=0.0) or 0.0
    exec_f = (
        extract_executed_qty(order, default=0.0)
        if (any(k in order for k in _EXEC_QTY_KEYS) or ("origSz" in order and "sz" in order))
        else 0.0
    )
    rem_f = extract_remaining_qty(order, default=max(0.0, orig_f - exec_f) if orig_f > 0 else 0.0)
    qty_f = rem_f if rem_f > 0 else orig_f
    raw_status = str(order.get("status") or "").strip().upper()
    norm_status = (
        "PARTIALLY_FILLED"
        if (exec_f > 0 and rem_f > 0)
        else ("NEW" if raw_status in ("OPEN", "PENDING", "WORKING", "RESTING", "") else raw_status)
    )
    descr = order.get("descr") if isinstance(order.get("descr"), dict) else {}
    raw_otype = (
        order.get("order_type")
        or order.get("orderType")
        or descr.get("ordertype")
        or (order.get("type") if str(order.get("type") or "").strip().upper() not in ("BUY", "SELL", "B", "S", "") else "")
        or "LIMIT"
    )
    actor, mod, rule = resolve_order_origin(order, known_ai_order_ids=known_ai_order_ids, audit_origin_map=audit_origin_map)
    rec = {
        "venue": venue,
        "symbol": symbol or str(order.get("symbol") or "").strip().upper(),
        "side": extract_order_side(order) or str(order.get("side") or order.get("type") or "").upper(),
        "order_type": str(raw_otype).strip().upper() or "LIMIT",
        "order_id": str(oid) if oid is not None else None,
        "orderId": oid,
        "actor": actor,
        "price": px,
        "qty": round(qty_f, 8) if qty_f > 0 else orig_q,
        "orig_qty": round(orig_f, 8) if orig_f > 0 else orig_q,
        "filled_qty": round(exec_f, 8),
        "remaining_qty": round(rem_f, 8),
        "status": norm_status,
        "origQty": orig_q,
        "executedQty": round(exec_f, 8) if exec_f > 0 else order.get("executedQty", 0),
    }
    if mod:
        rec["module"] = mod
    if rule:
        rec["rule"] = rule
    if cid:
        rec["clientOrderId"] = cid
        rec["client_order_id"] = cid
    ts_ms = extract_timestamp_ms(order, default=None)
    if ts_ms is None and cid:
        if m_ts := re.search(r"(?:^|[_-])(1[789]\d{8}(?:\d{3})?)(?:$|[_-])", str(cid)):
            ts_ms = normalize_timestamp_ms(m_ts.group(1), default=None)
    if ts_ms is None and isinstance(order_ts_map, dict):
        for tk in (str(oid) if oid is not None else "", str(cid or ""), str(cid or "").lower()):
            if tk and (mapped_ts := normalize_timestamp_ms(order_ts_map.get(tk), default=None)) is not None:
                ts_ms = mapped_ts
                break
    if ts_ms is not None and ts_ms > 0:
        eff_now_s = resolve_now_ts(now_ts)
        age_m = round(max(0.0, (eff_now_s * 1000.0 - float(ts_ms)) / 60000.0), 1)
        rec["created_ts"] = round(float(ts_ms) / 1000.0, 1)
        rec["age_min"] = age_m
        rec["age"] = f"{age_m:.1f}m" if age_m < 60.0 else f"{age_m / 60.0:.1f}h"
    return rec


_PARENT_ORDER_ID_KEYS = ("orderId", "order_id", "ordertxid", "oid")


def aggregate_fills_by_order_id(
    rows: list, *, cutoff_ms: Optional[float] = None, side: Optional[str] = None
) -> List[dict]:
    """Normalize and aggregate fills sharing the same parent order ID and side."""
    if rows is None or isinstance(rows, (str, bytes, bool)) or not hasattr(rows, "__iter__"):
        return []
    want = None
    if side is not None:
        if isinstance(side, bool) or not (want := normalize_side(side)):
            return []
    out: List[dict] = []
    grouped: dict[tuple[str, str], dict] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        parent_oid = _first_mapped(r, _PARENT_ORDER_ID_KEYS, clean_order_id)
        norm = normalize_order(r)
        ts_ms = norm["timestamp"]
        if cutoff_ms is not None:
            cutoff_f = safe_float(cutoff_ms, default=None)
            ts_f = safe_float(ts_ms, default=None)
            if cutoff_f is None or ts_f is None or ts_f < cutoff_f:
                continue
        if want and norm["side"] != want:
            continue
        if parent_oid:
            key = (parent_oid, norm["side"])
            px = safe_nonneg_float(norm.get("price"), default=0.0) or 0.0
            qty = safe_nonneg_float(norm.get("qty"), default=0.0) or 0.0
            priced_qty = qty if (px > 0.0 and qty > 0.0) else 0.0
            ts_val = ts_ms if ts_ms is not None else 0
            if key not in grouped:
                grouped[key] = {
                    "orderId": parent_oid,
                    "side": norm["side"],
                    "cost": px * priced_qty,
                    "cost_qty": priced_qty,
                    "qty": qty,
                    "fallback_price": px,
                    "timestamp": ts_val,
                }
                out.append(grouped[key])
            else:
                g = grouped[key]
                g["cost"] += px * priced_qty
                g["cost_qty"] += priced_qty
                g["qty"] += qty
                if px > 0:
                    g["fallback_price"] = px
                if ts_val > (g["timestamp"] or 0):
                    g["timestamp"] = ts_val
        else:
            out.append(norm)
    return [
        {
            "orderId": item["orderId"],
            "side": item["side"],
            "price": (item["cost"] / cost_qty) if (cost_qty := item.get("cost_qty", item["qty"])) > 0 else item["fallback_price"],
            "qty": item["qty"],
            "timestamp": item["timestamp"],
        }
        if "cost" in item
        else item
        for item in out
    ]


def build_open_order_entry(
    raw: Any,
    *,
    order_id: Any,
    side: str,
    price: Optional[float],
    orig_qty: float,
    executed_qty: float,
    client_order_id: Any = None,
    has_explicit_coid: bool = False,
    status: Optional[str] = None,
    default_status: str = "NEW",
    include_time: bool = True,
    extra: Optional[dict] = None,
) -> dict:
    """Build a canonical Binance-shaped open order dictionary."""
    coid = client_order_id if has_explicit_coid else extract_client_order_id(raw)
    st_raw = status if status is not None else (raw.get("status") if isinstance(raw, dict) else None)
    entry = {
        "orderId": order_id,
        "clientOrderId": coid,
        "side": side,
        "price": price,
        "origQty": orig_qty,
        "executedQty": executed_qty,
        "status": str(st_raw or default_status).strip().upper(),
    }
    if include_time and isinstance(raw, dict):
        if (ts_ms := extract_timestamp_ms(raw, default=None)) is not None:
            entry["time"] = ts_ms
    if extra:
        entry.update(extra)
    return entry


def format_strategy_execution_mode(
    observe_only: bool,
    dry_run: bool,
    *,
    paper_label: str = "PAPER (no money)",
    real_label: str = "⚠ REAL — REAL MONEY",
) -> str:
    """Format the standard operator banner string for bot execution mode."""
    if _parse_bool_flag(observe_only, False):
        return "OBSERVE ONLY (AI managed)"
    return paper_label if _parse_bool_flag(dry_run, True) else real_label


def _derive_128bit_client_order_id(
    client_order_id: Any,
    *,
    venue_label: str,
    domain_prefix: bytes,
    exact_re: re.Pattern[str],
    hex_prefix: str = "",
) -> str:
    """Map one durable client ID to a venue's 128-bit hex clientOrderId form."""
    if isinstance(client_order_id, (bool, list, tuple, dict, set, frozenset)):
        raise ProviderError(f"{venue_label} client_order_id must contain 1-128 characters")
    raw = clean_order_id(client_order_id) or ""
    if not raw or len(raw) > 128:
        raise ProviderError(f"{venue_label} client_order_id must contain 1-128 characters")
    try:
        encoded = raw.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ProviderError(f"{venue_label} client_order_id must be ASCII") from exc
    if exact_re.fullmatch(raw):
        return raw.lower()
    return hex_prefix + hashlib.sha256(domain_prefix + encoded).hexdigest()[:32]


def require_clean_order_id(
    order_id: Any, err_prefix: str, *, str_only: bool = False, msg_suffix: str = "invalid order id"
) -> str:
    """Validate and return a clean string order ID, or raise ``ProviderError``."""
    clean = clean_order_id(order_id) if (not str_only or isinstance(order_id, str)) else None
    if not clean:
        raise ProviderError(f"{err_prefix}: {msg_suffix}")
    return clean


def require_int_order_id(order_id: Any, err_prefix: str) -> int:
    """Validate and return a positive integer order ID, or raise ``ProviderError``."""
    oid = exact_pos_int(clean_order_id(order_id), default=None)
    if oid is None:
        raise ProviderError(f"{err_prefix}: invalid order id")
    return oid


def build_price_history_points(
    rows: Any,
    extract_ts: Any,
    extract_px: Any,
    *,
    cutoff_s: Optional[float] = None,
    ts_in_seconds: bool = False,
) -> List[dict]:
    """Extract, filter, and sort ascending ``{'timestamp': ms, 'price': float}`` rows."""
    if not isinstance(rows, (list, tuple)):
        return []
    out: List[dict] = []
    for row in rows:
        t_f = safe_pos_float(extract_ts(row), default=None)
        p_f = safe_pos_float(extract_px(row), default=None)
        if t_f is None or p_f is None or (cutoff_s is not None and t_f < cutoff_s):
            continue
        out.append({"timestamp": int(round(t_f * 1000.0)) if ts_in_seconds else int(round(t_f)), "price": p_f})
    out.sort(key=lambda x: x["timestamp"])
    return out


OHLC_INTERVALS_MIN_TO_STR: dict[int, str] = {
    1: "1m", 3: "3m", 5: "5m", 15: "15m", 30: "30m",
    60: "1h", 120: "2h", 240: "4h", 360: "6h", 480: "8h", 720: "12h",
    1440: "1d", 4320: "3d", 10080: "1w", 43200: "1M",
}
OHLC_INTERVALS_STR_TO_MIN: dict[str, int] = {v: k for k, v in OHLC_INTERVALS_MIN_TO_STR.items()}
OHLC_CASE_INSENSITIVE_INTERVAL_ALIASES: dict[str, str] = {
    "1H": "1h", "2H": "2h", "4H": "4h", "6H": "6h", "8H": "8h", "12H": "12h",
    "1D": "1d", "3D": "3d", "1W": "1w",
}
_OHLC_UNIT_MULTIPLIERS: dict[str, int] = {"m": 1, "h": 60, "d": 1440}


def parse_ohlc_interval_minutes(
    interval: Any,
    *,
    str_to_min: Optional[dict[str, int]] = None,
    reject_float: bool = False,
    allow_unit_multiplier: bool = False,
    require_unit_suffix: bool = False,
    venue_label: str = "OHLC",
) -> int:
    """Parse an OHLC interval (integer minutes or string e.g. '1h', '60') into positive integer minutes."""
    if require_unit_suffix and not isinstance(interval, str):
        raise ValueError(f"invalid {venue_label} interval: {interval!r}; examples: 5m, 1h, 1d")
    if isinstance(interval, bool):
        raise ValueError(f"interval must be positive integer or string, not bool: {interval}")
    lookup = str_to_min if str_to_min is not None else OHLC_INTERVALS_STR_TO_MIN
    if isinstance(interval, str):
        cleaned = interval.strip()
        if lookup:
            alias = OHLC_CASE_INSENSITIVE_INTERVAL_ALIASES.get(cleaned.upper(), cleaned)
            if alias not in lookup and alias.lower() in lookup and alias.upper() != "1M":
                alias = alias.lower()
            if alias in lookup:
                return lookup[alias]
        if allow_unit_multiplier and len(cleaned) >= 2:
            num_part, unit = cleaned[:-1], cleaned[-1].lower()
            if num_part.isascii() and num_part.isdigit() and unit in _OHLC_UNIT_MULTIPLIERS:
                minutes = int(num_part) * _OHLC_UNIT_MULTIPLIERS[unit]
                if minutes <= 0:
                    raise ValueError(
                        "the interval must be positive" if require_unit_suffix else f"interval must be positive: {interval}"
                    )
                return minutes
        if require_unit_suffix:
            raise ValueError(f"invalid {venue_label} interval: {interval!r}; examples: 5m, 1h, 1d")
        digits = cleaned[1:] if cleaned.startswith("+") else cleaned
        if digits.isascii() and digits.isdigit():
            val = int(digits)
            if val <= 0:
                raise ValueError(f"interval must be positive: {interval}")
            return val
        raise ValueError(f"unsupported {venue_label} interval: {interval}")
    if reject_float and isinstance(interval, float):
        raise ValueError(f"invalid interval={interval!r}: unsupported {venue_label} interval")
    val_i = exact_pos_int(interval, default=None)
    if val_i is None:
        raise ValueError(f"invalid interval={interval!r}: unsupported {venue_label} interval")
    return val_i


def parse_ohlc_interval(
    interval: Any,
    *,
    min_to_str: Optional[dict[int, str]] = None,
    str_to_min: Optional[dict[str, int]] = None,
    reject_float: bool = False,
    venue_label: str = "OHLC",
) -> tuple[int, str]:
    """Resolve ``interval`` into ``(interval_minutes, interval_str)`` using ``min_to_str``."""
    m2s = min_to_str if min_to_str is not None else OHLC_INTERVALS_MIN_TO_STR
    s2m = str_to_min if str_to_min is not None else {v: k for k, v in m2s.items()}
    minutes = parse_ohlc_interval_minutes(
        interval, str_to_min=s2m, reject_float=reject_float, venue_label=venue_label
    )
    interval_str = m2s.get(minutes)
    if interval_str is None:
        raise ValueError(f"unsupported {venue_label} interval cadence: {interval} ({minutes}m)")
    return minutes, interval_str


def resolve_ohlc_time_window_ms(
    *,
    start_ms: Any = None,
    end_ms: Any = None,
    lookback_days: Any = None,
    lookback_hours: Any = None,
    timeout: Any = 25,
    default_lookback_days: int = 30,
    strict_numeric_types: bool = False,
) -> tuple[int, int, int, float]:
    """Validate and resolve ``(now_ms, final_start, final_end, timeout_f)`` for OHLC candle fetches."""
    def _valid_num(val: Any, name: str, *, pos: bool = True, scale_ms: float = 1.0) -> float:
        f = (safe_pos_float if pos else safe_nonneg_float)(val, default=None)
        if f is None or (pos and int(f * scale_ms) <= 0 and scale_ms >= 1.0 and name != "timeout") or (
            strict_numeric_types and not isinstance(val, (int, float))
        ):
            kind = "positive" if pos else "non-negative"
            raise ValueError(f"invalid OHLC window parameters: {name} must be finite and {kind}: {val!r}")
        return f

    now_ms = int(time.time() * 1000)
    timeout_f = _valid_num(timeout, "timeout")
    final_end = int(_valid_num(end_ms, "end_ms")) if end_ms is not None else now_ms
    if final_end <= 0:
        raise ValueError(f"end_ms must be positive: {end_ms!r}")
    if start_ms is not None:
        final_start = int(_valid_num(start_ms, "start_ms", pos=False))
    elif lookback_days is not None:
        final_start = int(final_end - _valid_num(lookback_days, "lookback_days", scale_ms=86400 * 1000) * 86400 * 1000)
    elif lookback_hours is not None:
        final_start = int(final_end - _valid_num(lookback_hours, "lookback_hours", scale_ms=3600 * 1000) * 3600 * 1000)
    else:
        final_start = final_end - default_lookback_days * 86400 * 1000
    if final_start >= final_end:
        raise ValueError(f"start_ms ({final_start}) must be before end_ms ({final_end})")
    return now_ms, final_start, final_end, timeout_f


_MISSING = object()


def validate_ohlc_candle_bar(
    open_ts: Any,
    open_px: Any,
    high_px: Any,
    low_px: Any,
    close_px: Any,
    *,
    close_ts: Any = _MISSING,
    now_ms: Optional[int] = None,
    drop_incomplete: bool = True,
    close_inclusive: bool = False,
    ts_in_ms: bool = True,
) -> Optional[dict]:
    """Validate and normalize a single OHLC candle into ``{'timestamp': int, 'open': float, 'high': float, 'low': float, 'close': float}``."""
    ot_f = safe_pos_float(open_ts, default=None)
    o_v = safe_pos_float(open_px, default=None)
    h_v = safe_pos_float(high_px, default=None)
    l_v = safe_pos_float(low_px, default=None)
    c_v = safe_pos_float(close_px, default=None)
    if ot_f is None or o_v is None or h_v is None or l_v is None or c_v is None:
        return None
    open_i = int(ot_f)
    if open_i <= 0:
        return None
    if close_ts is not _MISSING:
        ct_f = safe_pos_float(close_ts, default=None)
        if ct_f is None:
            return None
        close_i = int(ct_f)
        if close_i < open_i:
            return None
        if drop_incomplete and now_ms is not None and ((close_i >= now_ms) if close_inclusive else (close_i > now_ms)):
            return None
    if h_v < l_v or h_v < max(o_v, c_v) or l_v > min(o_v, c_v):
        return None
    ts_sec = open_i // 1000 if ts_in_ms else open_i
    if ts_sec <= 0:
        return None
    return {"timestamp": ts_sec, "open": o_v, "high": h_v, "low": l_v, "close": c_v}
