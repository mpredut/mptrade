"""Provider primitives with no registry or venue-module dependencies."""

import hashlib
import math
import os
import re
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from botcore import (
    _parse_bool_flag,
    exact_nonneg_int,
    exact_pos_int,
    obj_attr,
    parse_dotenv,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
)
from cachepaths import normalize_venue_name


def env_value(folder: str, key: str) -> Optional[str]:
    """Read one key from ``.env``/``config.env``/``runtime.env`` without mutating ``os.environ``."""
    if not isinstance(folder, str) or not folder.strip() or not isinstance(key, str) or not key.strip():
        return None
    for fname in (".env", "config.env", "runtime.env"):
        path = os.path.join(folder, fname)
        if not os.path.exists(path):
            continue
        try:
            parsed = parse_dotenv(path)
            if key in parsed:
                val = str(parsed[key]).strip()
                return val or None
        except Exception:
            continue
    return None


def env_values_map(folder: str, names: tuple[str, ...] | List[str]) -> dict[str, Optional[str]]:
    """Resolve each key from ``os.environ`` first, falling back to ``env_value(folder, name)``."""
    return {name: os.environ.get(name) or env_value(folder, name) for name in names}


def bool_env(key: str, folder: Optional[str] = None, default: bool = False) -> bool:
    """Resolve a boolean flag from os.environ or optional provider .env/config.env."""
    def_flag = _parse_bool_flag(default, False)
    if not isinstance(key, str) or not key.strip():
        return def_flag
    v = os.environ.get(key)
    if v is None and folder is not None:
        v = env_value(folder, key)
    if v is None:
        return def_flag
    return _parse_bool_flag(v, def_flag)


_TIMESTAMP_KEYS = ("timestamp", "time", "updateTime", "transactTime", "opentm", "closetm", "workingTime", "T", "E", "ts")
_ORDER_ID_KEYS = ("orderId", "order_id", "ordertxid", "txid", "oid", "i", "id")
_PRICE_KEYS = ("price", "px", "limitPx", "fillPrice", "limit", "limitPrice", "averagePrice", "avgPrice", "rate", "L", "p")
_QTY_KEYS = (
    "qty", "quantity", "executedQty", "filledQuantity", "filledQty", "vol_exec",
    "sz", "vol", "origQty", "origSz", "orig_qty", "orderedQuantity", "amount_shares", "l", "z", "q",
)
_ORIG_QTY_KEYS = (
    "origQty", "orig_qty", "origSz", "vol", "orderedQuantity", "original_qty",
    "order_qty", "qty", "amount", "quantity", "sz", "amount_shares", "q",
)
_EXEC_QTY_KEYS = ("executedQty", "filledQuantity", "filledQty", "vol_exec", "filled_qty", "z")
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


def normalize_timestamp_ms(value: Any, default: Optional[int] = None) -> Optional[int]:
    """Normalize a raw timestamp (in seconds or milliseconds) to integer milliseconds.

    Epoch values in ``[1_000_000_000, 100_000_000_000)`` (all Unix seconds between
    2001 and 2286) and non-integer float seconds are converted to milliseconds.
    Millisecond values (``>= 100_000_000_000``) and small synthetic test integers
    (``< 1_000_000_000``) are preserved as-is.
    """
    numeric = safe_nonneg_float(value, default=None)
    if numeric is None:
        return default
    if numeric == 0.0:
        return 0
    if 1_000_000_000.0 <= numeric < 100_000_000_000.0 or (
        0.0 < numeric < 1_000_000_000.0 and isinstance(value, float) and not numeric.is_integer()
    ):
        return int(round(numeric * 1000.0))
    return int(round(numeric))


def extract_timestamp_ms(order: Any, default: Optional[int] = None) -> Optional[int]:
    """Extract and normalize an order/fill timestamp in milliseconds from any venue shape."""
    return _first_mapped(order, _TIMESTAMP_KEYS, lambda v: normalize_timestamp_ms(v, default=None), default)


def extract_timestamp_s(order: Any, default: Optional[float] = None) -> Optional[float]:
    """Extract and normalize an order/fill timestamp in seconds from any venue shape."""
    numeric = _first_mapped(order, _TIMESTAMP_KEYS, lambda v: safe_nonneg_float(v, default=None))
    if numeric is None:
        return default
    return numeric / 1000.0 if numeric >= 100_000_000_000.0 else numeric


def clean_order_id(val: Any) -> Optional[str]:
    if val is None or isinstance(val, (bool, dict, list, tuple, set, frozenset)):
        return None
    if isinstance(val, (int, float)):
        val = exact_pos_int(val, default=None)
        if val is None:
            return None
    if not isinstance(val, (str, int)):
        return None
    text = str(val).strip()
    if (
        not text
        or text.startswith("-")
        or text.lower() in {
            "0", "0.0", "-0", "+0", "+0.0", "-0.0",
            "nan", "inf", "-inf", "+inf", "infinity", "-infinity", "+infinity",
            "none", "null",
        }
    ):
        return None
    if text.endswith(".0") and text[:-2].lstrip("+").isdigit():
        text = text[:-2]
    if text in {"0", "+0"}:
        return None
    return text[1:] if text.startswith("+") and text[1:].isdigit() else text


_clean_single_order_id = clean_order_id


def extract_order_id(order: Any, default: Optional[str] = None) -> Optional[str]:
    """Extract and normalize a venue order ID from any response shape.

    Handles ``SubmissionOutcome`` objects, venue dicts (``orderId``, ``order_id``,
    ``id``, ``txid``, ``oid``, ``i``), Kraken ``txid`` lists/tuples, and scalar IDs.
    """
    if order is None or isinstance(order, bool):
        return default
    if isinstance(order, (list, tuple)):
        for item in order:
            cleaned = extract_order_id(item)
            if cleaned is not None:
                return cleaned
        return default
    if hasattr(order, "order_id") and hasattr(order, "state"):
        cleaned = clean_order_id(getattr(order, "order_id", None))
        return cleaned if cleaned is not None else default
    if isinstance(order, dict):
        return _first_mapped(order, _ORDER_ID_KEYS, extract_order_id, default)
    cleaned = clean_order_id(order)
    return cleaned if cleaned is not None else default


def normalize_side(side: Any) -> str:
    """Normalize a scalar order side token to ``'BUY'`` or ``'SELL'``."""
    if side is None or isinstance(side, bool):
        return ""
    s = str(side).strip().upper()
    if s in ("B", "BUY", "BID", "LONG", "OPEN LONG", "CLOSE SHORT"):
        return "BUY"
    if s in ("A", "S", "SELL", "SOLD", "ASK", "SHORT", "CLOSE LONG", "OPEN SHORT"):
        return "SELL"
    return ""


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
        t_norm = normalize_side(raw_type)
        if t_norm in ("BUY", "SELL"):
            return t_norm
    raw_buyer = order.get("isBuyer")
    if raw_buyer is not None:
        parsed_b = _parse_bool_flag(raw_buyer, None)
        if parsed_b is not None:
            return "BUY" if parsed_b else "SELL"
        if isinstance(raw_buyer, str):
            b_norm = raw_buyer.strip().lower()
            if b_norm in ("b", "buy", "s", "sell"):
                return "BUY" if b_norm.startswith("b") else "SELL"
    for sq_key in ("signed_qty", "orderedQuantity", "quantity"):
        sq = safe_float(order.get(sq_key), default=0.0)
        if sq < 0.0:
            return "SELL"
        if sq > 0.0 and (sq_key in ("signed_qty", "orderedQuantity") or "ticker" in order or "limitPrice" in order):
            return "BUY"
    return ""


def extract_order_price(order: Any, default: float = 0.0) -> float:
    """Extract order/fill price from any venue shape."""
    if not isinstance(order, dict):
        return default
    val = _first_mapped(order, _PRICE_KEYS, lambda v: safe_pos_float(v, default=None))
    if val is not None:
        return val
    descr = order.get("descr")
    if isinstance(descr, dict):
        val = safe_pos_float(descr.get("price"), default=None)
        if val is not None:
            return val
    for cost_key, exec_key in (
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
    ):
        if cost_key in order and exec_key in order:
            cost_val = _pos_abs_float(order.get(cost_key))
            exec_val = _pos_abs_float(order.get(exec_key))
            if cost_val and exec_val:
                res = safe_pos_float(cost_val / exec_val, default=None)
                if res is not None:
                    return res
    fills = order.get("fills")
    if isinstance(fills, list) and fills:
        total_quote = 0.0
        total_qty = 0.0
        for f in fills:
            if isinstance(f, dict):
                f_px = safe_pos_float(f.get("price", f.get("px", 0.0)), default=None)
                f_qty = _pos_abs_float(f.get("qty", f.get("quantity", f.get("sz", 0.0))))
                if f_px is not None and f_qty is not None:
                    total_quote += f_px * f_qty
                    total_qty += f_qty
        if total_qty > 0:
            res = safe_pos_float(total_quote / total_qty, default=None)
            if res is not None:
                return res
    msg_str = order.get("msg")
    if isinstance(msg_str, str) and "@" in msg_str and msg_str.strip().lower().startswith("filled "):
        msg_px = safe_pos_float(msg_str.rsplit("@", 1)[1].strip(), default=None)
        if msg_px is not None:
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
        msg_qty = safe_pos_float(msg_str[7:].split("@", 1)[0].strip(), default=None)
        if msg_qty is not None:
            return msg_qty
    raw_st = str(order.get("status") or "").strip().upper()
    if raw_st in _UNFILLED_ORDER_STATUSES or (isinstance(order.get("resting"), dict) and not filled_sub):
        return 0.0
    return extract_order_qty(order, default=default)


def extract_orig_qty(order: Any, default: Optional[float] = None) -> Optional[float]:
    """Extract the venue-accepted original order quantity from any response shape.

    Prioritizes original-quantity fields (``origQty``, ``origSz``, ``vol``,
    ``orderedQuantity``, etc.) over executed-quantity fields so recovering a
    ``PARTIALLY_FILLED`` order never overwrites the total order quantity with
    the partial fill quantity.
    """
    if hasattr(order, "native") and hasattr(order, "state"):
        order = getattr(order, "native", None)
    if order is None or isinstance(order, bool):
        return default
    for key in (*_ORIG_QTY_KEYS, *_EXEC_QTY_KEYS):
        val = _pos_abs_float(obj_attr(order, key))
        if val is not None:
            return val
    return default


def extract_remaining_qty(order: Any, default: float = 0.0) -> float:
    """Extract remaining unfilled quantity for an open/cancelled order across any venue shape."""
    if not isinstance(order, dict):
        return default
    rem_f = safe_nonneg_float(order.get("remainingQty"), default=None)
    if rem_f is not None:
        return rem_f
    if "origSz" in order and "sz" in order and not any(k in order for k in _EXEC_QTY_KEYS):
        sz_f = _nonneg_abs_float(order.get("sz"))
        if sz_f is not None:
            return sz_f
    orig = extract_orig_qty(order, default=0.0) or 0.0
    if orig > 0:
        exec_q = _first_mapped(order, _EXEC_QTY_KEYS, _pos_abs_float, 0.0)
        return max(0.0, orig - exec_q) if exec_q > 0 else orig
    return default


def extract_order_fee(
    order: Any,
    base_asset: str = "",
    quote_asset: str = "",
    fill_price: Optional[float] = None,
    require_explicit_asset: bool = False,
) -> Optional[dict]:
    """Extract and normalize order/fill fee metadata across any venue shape.

    Returns ``None`` when no fee field is present or when fee metadata is invalid,
    otherwise returns ``{"fee": float, "fee_asset": str, "base_fee": float,
    "quote_fee": float, "fee_in_quote": Optional[float]}``.
    """
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
    base_fee = 0.0
    quote_fee = 0.0
    fee_in_quote: Optional[float] = 0.0 if fee == 0.0 else None

    if fee > 0.0:
        if base_u and fee_asset == base_u:
            base_fee = fee
            if px > 0.0:
                fee_in_quote = fee * px
        elif (quote_u and fee_asset == quote_u) or (not fee_asset and not req_explicit):
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
    ("MO_TA", "tradeall"),
    ("MO_MT", "monitortrades"),
    ("MO_RT", "rtrade"),
    ("MO_SD", "trailing"),
    ("TA_OR", "tradeall"),
    ("MT_OR", "monitortrades"),
    ("AG_OR", "assetguardian"),
    ("RT_OR", "rtrade"),
    ("SD_OR", "trailing"),
    ("MO_OR", "monitororder"),
    ("AGS", "assetguardian"),
    ("AGB", "assetguardian"),
    ("TRL", "trailing"),
    ("TA", "tradeall"),
    ("TO", "tradeall"),
    ("MT", "monitortrades"),
    ("AG", "assetguardian"),
    ("RT", "rtrade"),
    ("MO", "monitororder"),
    ("TS", "trailing"),
    ("SD", "trailing"),
)
_GENERIC_CID_TOKENS = frozenset({"OR", "MAINTHREAD", "WORKER", "BOT", "ORDER", "RETRY", "OUTBOX"})
_CLIENT_ORDER_ID_KEYS = ("clientOrderId", "client_order_id", "cloid", "cl_ord_id", "userref")


def extract_client_order_id(order: Any, default: Optional[str] = None) -> Optional[str]:
    """Extract and normalize client order ID or user reference from any venue shape."""
    return _first_mapped(order, _CLIENT_ORDER_ID_KEYS, clean_order_id, default)


def _extract_order_cid(order: dict) -> str:
    return extract_client_order_id(order, default="") or ""


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
    raw_cid = _extract_order_cid(order).upper()
    if raw_cid.startswith("AI_"):
        return "AI"
    if raw_cid.startswith("SD_") and len(raw_cid) >= 20 and all(c in "0123456789ABCDEF" for c in raw_cid[3:]):
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

    mod = _first_nonempty_str(order, ("module", "bot", "caller"), lower=True)
    rule = _first_nonempty_str(order, ("rule", "trigger_rule", "kind", "role", "motivation"))
    oid = extract_order_id(order, default="")
    raw_cid = _extract_order_cid(order)

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
                        rem = raw_cid[len(pfx) :].lstrip("_-")
                        tokens = [t for t in re.split(r"[_-]+", rem) if t and t.upper() not in _GENERIC_CID_TOKENS]
                        if tokens:
                            cand = "_".join(tokens[:-1]) if len(tokens) > 1 and (tokens[-1].isdigit() or len(tokens[-1]) >= 8) else tokens[0]
                            if not (len(cand) >= 8 and all(c in "0123456789abcdefABCDEF" for c in cand)) and not cand.isdigit():
                                rule = cand.upper()
                break
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
        or (
            order.get("type")
            if str(order.get("type") or "").strip().upper() not in ("BUY", "SELL", "B", "S", "")
            else ""
        )
        or "LIMIT"
    )
    actor, mod, rule = resolve_order_origin(
        order, known_ai_order_ids=known_ai_order_ids, audit_origin_map=audit_origin_map
    )
    rec = {
        "venue": venue,
        "symbol": symbol,
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
    return rec


_PARENT_ORDER_ID_KEYS = ("orderId", "order_id", "ordertxid", "oid")


def aggregate_fills_by_order_id(
    rows: list,
    *,
    cutoff_ms: Optional[float] = None,
    side: Optional[str] = None,
) -> List[dict]:
    """Normalize and aggregate fills sharing the same parent order ID and side.

    Multi-fill executions sharing the same parent order ID (``orderId``, ``order_id``,
    ``ordertxid``, ``oid``) and side are combined into a single order record with
    volume-weighted average ``price``, summed ``qty``, and ``max(timestamp)``,
    matching ``BinanceProvider.get_orders()``.
    """
    if rows is None or isinstance(rows, (str, bytes, bool)) or not hasattr(rows, "__iter__"):
        return []
    want = None
    if side is not None:
        if isinstance(side, bool):
            return []
        want = normalize_side(side)
        if not want:
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


# Compatibility alias retained for existing internal and external imports.
_normalize_order = normalize_order

from .quantity import decide_quantity, venue_fee_cap_quantity  # noqa: E402
from .strategy_executor import (  # noqa: E402
    OrderReconciliationCapabilities,
    PairPrecision,
    ProviderError,
)


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
    order_id: Any,
    err_prefix: str,
    *,
    str_only: bool = False,
    msg_suffix: str = "invalid order id",
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
        out.append({"timestamp": int(t_f * 1000) if ts_in_seconds else int(t_f), "price": p_f})
    out.sort(key=lambda x: x["timestamp"])
    return out


def read_client_ohlc_series(
    client_or_factory: Any,
    symbol: str,
    interval_min: int,
    *,
    err_prefix: Optional[str] = None,
):
    """Read a ``ClosedPriceSeries`` from a client or executor supporting ``ohlc_series`` or ``ohlc_closes*``."""
    from market_regime import ClosedPriceSeries

    pfx = err_prefix or f"ohlc_series({symbol})"
    direct_series = getattr(client_or_factory, "ohlc_series", None)
    if callable(direct_series):
        return direct_series(symbol, interval_min)
    interval_i = exact_pos_int(interval_min, default=None)
    if interval_i is None:
        raise ProviderError(f"{pfx}: invalid interval_min={interval_min!r}")
    try:
        client = (
            client_or_factory()
            if callable(client_or_factory) and not hasattr(client_or_factory, "ohlc_closes")
            else client_or_factory
        )
        series_reader = getattr(client, "ohlc_closes_with_timestamps", None)
        if callable(series_reader):
            closes, timestamps = series_reader(symbol, interval_i)
            timestamps = tuple(timestamps) if timestamps is not None else ()
            observed_at = timestamps[-1] if timestamps else None
        else:
            reader = getattr(client, "ohlc_closes_with_timestamp", None)
            if callable(reader):
                closes, observed_at = reader(symbol, interval_i)
            else:
                closes = client.ohlc_closes(symbol, interval_i)
                observed_at = None
            timestamps = ()
        return ClosedPriceSeries(
            tuple(closes) if closes is not None else (),
            interval_i,
            observed_at,
            timestamps,
        )
    except ProviderError:
        raise
    except Exception as exc:
        raise ProviderError(f"{pfx}: {exc}") from exc


class MarketDataProvider(ABC):
    """Shared market-data, account, order, and policy-hook interface."""

    @abstractmethod
    def get_current_price(self, symbol: str) -> Optional[float]:
        ...

    def get_price_history(self, symbol: str, lookback_h: float) -> Optional[List]:
        return None

    @abstractmethod
    def supports_symbol(self, symbol: str) -> bool:
        ...

    def validate_symbol(self, symbol: str) -> None:
        """Reject a symbol incompatible with this configured provider instance."""
        return None

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    def free_balance(self, asset: str) -> Optional[float]:
        """Return free balance: ``0.0`` is real zero; ``None`` means unavailable."""
        return None

    def get_orders(self, symbol: str, side: Optional[str], since_s: float) -> List[dict]:
        return []

    def get_trades(self, symbol: str, since_s: float) -> List[dict]:
        return self.get_orders(symbol, None, since_s)

    def position_cost_basis(self, symbol: str, held_qty: float, since_s: float) -> Optional[float]:
        """Return inventory-reconciled acquisition cost, or unknown when unsupported."""
        return None

    def open_orders(self, symbol: str) -> List[dict]:
        return []

    def reconciliation_capabilities(self) -> OrderReconciliationCapabilities:
        """Return explicitly supported strict lifecycle operations.

        The default is deliberately conservative. An inherited empty
        ``open_orders`` implementation must never be mistaken for a confirmed
        empty venue snapshot.
        """
        return OrderReconciliationCapabilities()

    def preflight_order(self, symbol: str, side: str, qty: float,
                        price=None, *, market: bool = False,
                        kind: Optional[str] = None) -> Any:
        """Validate venue state and optionally return opaque submit authorization."""
        return None

    def _is_live_order_enabled(self, env_key: str, folder: Optional[str] = None) -> bool:
        """Resolve live-order gate from explicit ``self._live_enabled`` override or ``bool_env``."""
        live_override = getattr(self, "_live_enabled", None)
        if live_override is None:
            return bool_env(env_key, folder)
        return _parse_bool_flag(live_override, False)

    def _preflight_sell_base_balance(self, symbol: str, side: str, qty: float) -> None:
        """Reject an impossible SELL before creating a durable submit intent."""
        if normalize_side(side) != "SELL" or qty is None:
            return
        requested = safe_pos_float(qty, default=None)
        if requested is None:
            raise ProviderError(f"preflight_order {symbol}: an invalid SELL quantity")
        precision = self.pair_precision(symbol)
        base_asset = precision.base_asset if precision else ""
        if not base_asset:
            raise ProviderError(f"preflight_order {symbol}: base asset is unavailable")
        avail_f = safe_nonneg_float(self.free_balance(base_asset), default=None)
        if avail_f is None:
            raise ProviderError(f"preflight_order {symbol}: {base_asset} balance is unavailable")
        decimals = precision.volume_decimals if precision else 8
        tolerance = 0.5 * (10.0 ** -max(0, decimals))
        if requested > avail_f + tolerance:
            raise ProviderError(
                f"preflight_order {symbol}: insufficient funds SELL "
                f"qty={requested} available={avail_f} {base_asset}"
            )

    def _preflight_buy_quote_balance(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: Optional[float] = None,
        *,
        quote_asset: str = "USDC",
        market: bool = False,
        kind: Optional[str] = None,
    ) -> None:
        """Reject a BUY that free quote currency cannot fund in full."""
        if normalize_side(side) != "BUY" or qty is None:
            return
        qty_f = safe_pos_float(qty, default=None)
        if qty_f is None:
            raise ProviderError(f"preflight_order({symbol}): invalid BUY quantity")
        if price is None:
            mid_f = safe_pos_float(self.get_current_price(symbol), default=None)
            if mid_f is None:
                raise ProviderError(f"preflight_order({symbol}): price unavailable")
            price_f = mid_f * (1.05 if _parse_bool_flag(market, False) else 1.0)
        else:
            price_f = safe_pos_float(price, default=None)
            if price_f is None:
                raise ProviderError(f"preflight_order({symbol}): invalid BUY price")
        required = safe_pos_float(qty_f * price_f, default=None)
        if required is None:
            raise ProviderError(f"preflight_order({symbol}): invalid BUY notional")
        avail_f = safe_nonneg_float(self.free_balance(quote_asset), default=None)
        if avail_f is None:
            raise ProviderError(f"preflight_order({symbol}): the {quote_asset} balance cannot be confirmed")
        tolerance = max(1e-9, required * 1e-12)
        if avail_f + tolerance < required:
            raise ProviderError(
                f"preflight_order({symbol}) {kind or 'BUY'}: insufficient {quote_asset} balance "
                f"({avail_f:.8f} < {required:.8f}) — the order was not sent"
            )

    def execution_enabled(self) -> bool:
        """Return whether this provider may create a real venue order now.

        Providers with an explicit dry/live switch override this hook. The shared
        placement pipeline checks it before creating durable retry state so a dry
        validation cannot become a live order after a later configuration change.
        """
        return True

    def prepare_order_state(self) -> Any:
        """Synchronize provider state used by shared pre-submit policy checks."""
        return None

    def validate_order_state(self, expected_state: Any) -> Any:
        """Require that policy checks still describe the prepared provider state."""
        return expected_state

    def place_order(self, symbol: str, side: str, price: float, qty: float, force: bool = False, **kwargs):
        return None

    def cancel_order(self, symbol: str, order_id: str) -> None:
        return None

    def guards_internally(self) -> bool:
        return False

    def _is_binance(self) -> bool:
        return normalize_venue_name(getattr(self, "name", "")) == "binance"

    def requires_cache_permit(self) -> bool:
        """Return whether this provider enforces account-cache submit permits."""
        return self._is_binance()

    def defers_smart_cancellation(self) -> bool:
        """Return whether opposing order cancellation is deferred until dispatch."""
        return self._is_binance()

    def supports_business_minimum(self) -> bool:
        """Return whether this provider enforces venue business minimum rules."""
        return self._is_binance()

    def pair_precision(self, symbol: str) -> Optional[PairPrecision]:
        """Return normalized pair precision and limits, or None if unavailable."""
        return None

    def _pair_precision_attr(self, symbol: str, attr: str, default: Any = None) -> Any:
        try:
            pp = self.pair_precision(symbol) if hasattr(self, "pair_precision") else None
            if pp is not None:
                return getattr(pp, attr, default)
        except Exception:
            pass
        return default

    def round_quantity(self, symbol: str, qty: float) -> float:
        """Round or floor quantity to venue volume precision."""
        qty_f = safe_pos_float(qty, default=None)
        if qty_f is None:
            return 0.0
        dec = exact_nonneg_int(self._pair_precision_attr(symbol, "volume_decimals"), default=None)
        if dec is not None:
            factor = 10 ** min(dec, 18)
            return math.floor(qty_f * factor + 1e-12) / factor
        return qty_f

    round_amount = round_quantity

    def round_price(self, symbol: str, price: float) -> float:
        """Round price to venue price precision (tick size / price decimals)."""
        price_f = safe_pos_float(price, default=None)
        if price_f is None:
            return 0.0
        dec = exact_nonneg_int(self._pair_precision_attr(symbol, "price_decimals"), default=None)
        return round(price_f, min(dec, 18)) if dec is not None else price_f

    def min_order_qty(self, symbol: str) -> float:
        """Return venue minimum order quantity; 0.0 means no known minimum."""
        return safe_pos_float(self._pair_precision_attr(symbol, "order_min", 0.0), default=0.0)

    def order_filter_refusal(self, symbol: str, side: str, price: float,
                             qty: float, *, market: bool = False,
                             enforce_business_minimum: bool = True
                             ) -> Optional[str]:
        """Return a provider-specific pre-submit refusal, if one is known."""
        qty_f = safe_pos_float(qty, default=None)
        if qty_f is None:
            return "qty_zero_or_negative"
        try:
            min_qty = self.min_order_qty(symbol)
            if min_qty > 0 and qty_f < min_qty:
                return f"filter_lot_size_min:{qty_f}<{min_qty}"
        except Exception:
            pass
        return None

    def policy_cap_quantity(self, symbol: str, side: str, price: float,
                            qty: float, available_qty: float, **kwargs) -> float:
        import order_guard
        return order_guard.weight_limit(self, symbol, side, price, qty, available_qty=available_qty)

    def fee_cap_quantity(self, symbol: str, side: str, price: float,
                         available_qty: float) -> float:
        return venue_fee_cap_quantity(self.name, side, price, available_qty)

    def quantity_decision(self, symbol: str, side: str, price: float,
                          qty: float, **kwargs):
        return decide_quantity(self, symbol, side, price, qty, **kwargs)

    def adjust_order_price(self, symbol: str, side: str, price: float,
                           cancel_opposite: bool = True) -> float:
        return price

    def cancel_opposite_orders(self, symbol: str, side: str,
                               requested_price: float) -> None:
        """Cancel adverse opposing orders when the venue implements this policy."""
        return None

    def profit_guard_window_ref(
        self, symbol: str, side: str, safeback_sec, regime_context=None, reference_mode=None, module=None,
        *, prefer_safeback: bool = False, default_safeback_sec: float = 0.0,
    ):
        import order_guard
        sb_f = safe_pos_float(safeback_sec, default=None)
        if prefer_safeback and sb_f is not None:
            window_s = sb_f
        else:
            win_cfg = order_guard.get_window_cfg(
                self.name,
                symbol=symbol,
                order_type=side,
                regime_context=regime_context,
                module=module,
                reference_mode=reference_mode,
            )
            win_f = safe_pos_float(win_cfg, default=None)
            window_s = win_f if win_f is not None else (sb_f if sb_f is not None else default_safeback_sec)
        wr_kwargs = {k: v for k, v in (("reference_mode", reference_mode), ("module", module)) if v is not None}
        return order_guard.window_reference(self, symbol, side, window_s, **wr_kwargs)

    def last_opposite_fill(self, symbol: str, order_type: str,
                           since_s: float = 90 * 24 * 3600) -> Optional[float]:
        ot_norm = normalize_side(order_type)
        if ot_norm not in ("BUY", "SELL"):
            return None
        opposite = "SELL" if ot_norm == "BUY" else "BUY"
        eff_since_s = safe_nonneg_float(since_s, default=90.0 * 24.0 * 3600.0)
        fills = self.get_orders(symbol, opposite, eff_since_s)
        if fills is None:
            raise ProviderError(f"last_opposite_fill({symbol}): failed to read order history")
        valid_fills = [o for o in fills if isinstance(o, dict) and extract_order_price(o, default=0.0) > 0.0]
        if not valid_fills:
            return None
        latest = max(valid_fills, key=lambda order: extract_timestamp_ms(order, default=0) or 0)
        return extract_order_price(latest) or None

    def ohlc_closes(self, symbol: str, interval_min: int) -> list:
        """Return completed-bar closing prices from closed interval series."""
        if hasattr(self, "ohlc_series"):
            series = self.ohlc_series(symbol, interval_min)
            closes = getattr(series, "closes", None) if series is not None else None
            return list(closes) if closes is not None else []
        return []
