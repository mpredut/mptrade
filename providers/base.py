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
    if 1_000_000_000.0 <= numeric < 100_000_000_000.0:
        return int(round(numeric * 1000.0))
    if 0.0 < numeric < 1_000_000_000.0 and isinstance(value, float) and not numeric.is_integer():
        return int(round(numeric * 1000.0))
    return int(round(numeric))


def extract_timestamp_ms(order: Any, default: Optional[int] = None) -> Optional[int]:
    """Extract and normalize an order/fill timestamp in milliseconds from any venue shape."""
    if not isinstance(order, dict):
        return default
    for key in _TIMESTAMP_KEYS:
        if key in order and order[key] is not None:
            normalized = normalize_timestamp_ms(order[key], default=None)
            if normalized is not None:
                return normalized
    return default


def extract_timestamp_s(order: Any, default: Optional[float] = None) -> Optional[float]:
    """Extract and normalize an order/fill timestamp in seconds from any venue shape."""
    if not isinstance(order, dict):
        return default
    for key in _TIMESTAMP_KEYS:
        if key in order and order[key] is not None:
            numeric = safe_nonneg_float(order[key], default=None)
            if numeric is not None:
                if numeric >= 100_000_000_000.0:
                    return numeric / 1000.0
                return numeric
    return default


def clean_order_id(val: Any) -> Optional[str]:
    if val is None or isinstance(val, (bool, dict, list, tuple, set, frozenset)):
        return None
    if isinstance(val, float):
        if not math.isfinite(val) or not val.is_integer() or val <= 0:
            return None
        val = int(val)
    if isinstance(val, int):
        if val <= 0:
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
    if text.startswith("+") and text[1:].isdigit():
        text = text[1:]
    return text


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
        cleaned = _clean_single_order_id(getattr(order, "order_id", None))
        return cleaned if cleaned is not None else default
    if isinstance(order, dict):
        for key in _ORDER_ID_KEYS:
            if key in order:
                cleaned = extract_order_id(order[key])
                if cleaned is not None:
                    return cleaned
        return default
    cleaned = _clean_single_order_id(order)
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
    return (
        _parse_bool_flag(force, False)
        or _parse_bool_flag(market, False)
        or _parse_bool_flag(kwargs.get("market", False), False)
        or _parse_bool_flag(kwargs.get("force", False), False)
    )


def extract_order_side(order: Any) -> str:
    """Extract and normalize order side (``'BUY'`` or ``'SELL'``) from any venue shape."""
    if not isinstance(order, dict):
        return ""
    for k in ("side", "S", "dir"):
        s_norm = normalize_side(order.get(k))
        if s_norm:
            return s_norm
    for raw_type in (
        order.get("type"),
        order["descr"].get("type") if isinstance(order.get("descr"), dict) else None,
    ):
        t_norm = normalize_side(raw_type)
        if t_norm in ("BUY", "SELL"):
            return t_norm
    if "isBuyer" in order and order.get("isBuyer") is not None:
        raw_buyer = order["isBuyer"]
        parsed_b = _parse_bool_flag(raw_buyer, None)
        if parsed_b is not None:
            return "BUY" if parsed_b else "SELL"
        if isinstance(raw_buyer, str):
            b_norm = raw_buyer.strip().lower()
            if b_norm in ("b", "buy"):
                return "BUY"
            if b_norm in ("s", "sell"):
                return "SELL"
    for sq_key in ("signed_qty", "orderedQuantity", "quantity"):
        sq = safe_float(order.get(sq_key), default=0.0)
        if sq != 0.0:
            if sq < 0.0:
                return "SELL"
            if sq_key in ("signed_qty", "orderedQuantity") or "ticker" in order or "limitPrice" in order:
                return "BUY"
    return ""


def extract_order_price(order: Any, default: float = 0.0) -> float:
    """Extract order/fill price from any venue shape."""
    if not isinstance(order, dict):
        return default
    for key in _PRICE_KEYS:
        if key in order and order[key] is not None:
            val = safe_pos_float(order[key], default=None)
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
            cost_val = abs(safe_float(order.get(cost_key), default=0.0))
            exec_val = abs(safe_float(order.get(exec_key), default=0.0))
            if cost_val > 0 and exec_val > 0:
                res = safe_pos_float(cost_val / exec_val, default=None)
                if res is not None:
                    return res
    fills = order.get("fills")
    if isinstance(fills, list) and fills:
        total_quote = 0.0
        total_qty = 0.0
        for f in fills:
            if not isinstance(f, dict):
                continue
            f_px = safe_pos_float(f.get("price", f.get("px", 0.0)), default=None)
            f_qty = abs(safe_float(f.get("qty", f.get("quantity", f.get("sz", 0.0))), default=0.0))
            if f_px is not None and f_qty > 0:
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
    if not isinstance(order, dict):
        return default
    for key in _QTY_KEYS:
        if key in order and order[key] is not None:
            val = abs(safe_float(order[key], default=0.0))
            if val > 0:
                return val
    return default


def extract_executed_qty(order: Any, default: float = 0.0) -> float:
    """Extract executed/filled quantity without falling back to origQty on unfilled/resting orders."""
    if not isinstance(order, dict):
        return default
    for key in _EXEC_QTY_KEYS:
        if key in order and order[key] is not None:
            val = safe_float(order[key], default=None)
            if val is not None:
                return abs(val)
    filled_sub = order.get("filled")
    if isinstance(filled_sub, dict):
        for key in ("totalSz", "sz", "qty", *_EXEC_QTY_KEYS):
            if key in filled_sub and filled_sub[key] is not None:
                val = safe_float(filled_sub[key], default=None)
                if val is not None:
                    return abs(val)
    if "origSz" in order and "sz" in order:
        orig_sz = safe_float(order.get("origSz"), default=None)
        rem_sz = safe_float(order.get("sz"), default=None)
        if orig_sz is not None and rem_sz is not None:
            return round(max(0.0, abs(orig_sz) - abs(rem_sz)), 8)
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
    if order is None or isinstance(order, bool):
        return default
    if hasattr(order, "native") and hasattr(order, "state"):
        order = getattr(order, "native", None)
        if order is None or isinstance(order, bool):
            return default
    is_dict = isinstance(order, dict)
    for key in (*_ORIG_QTY_KEYS, *_EXEC_QTY_KEYS):
        raw = order.get(key) if is_dict else getattr(order, key, None)
        if raw is not None:
            val = abs(safe_float(raw, default=0.0))
            if val > 0:
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
        sz_f = safe_float(order.get("sz"), default=None)
        if sz_f is not None:
            return abs(sz_f)
    orig = extract_orig_qty(order, default=0.0) or 0.0
    if orig > 0:
        exec_q = next(
            (
                abs(vf)
                for key in _EXEC_QTY_KEYS
                if key in order and (vf := safe_float(order.get(key), default=0.0)) != 0.0
            ),
            0.0,
        )
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
    fee = next(
        (
            cand_f
            for key in _FEE_KEYS
            if key in order and (cand_f := safe_nonneg_float(order[key], default=None)) is not None
        ),
        None,
    )
    if fee is None:
        return None

    fee_asset = ""
    has_asset_key = False
    for key in _FEE_ASSET_KEYS:
        if key in order:
            has_asset_key = True
            val = order[key]
            if isinstance(val, str) and val.strip():
                fee_asset = val.strip().upper()
                break

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
        "qty": extract_order_qty(order),
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
_CLIENT_ORDER_ID_KEYS = (
    "clientOrderId",
    "client_order_id",
    "cloid",
    "cl_ord_id",
    "userref",
)


def extract_client_order_id(order: Any, default: Optional[str] = None) -> Optional[str]:
    """Extract and normalize client order ID or user reference from any venue shape."""
    if not isinstance(order, dict):
        return default
    for key in _CLIENT_ORDER_ID_KEYS:
        val = order.get(key)
        cleaned = clean_order_id(val)
        if cleaned is not None:
            return cleaned
    return default


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


def resolve_order_origin(
    order: Any,
    known_ai_order_ids: Optional[set[str]] = None,
    audit_origin_map: Optional[dict[str, tuple[Optional[str], Optional[str]]]] = None,
) -> tuple[str, Optional[str], Optional[str]]:
    """Resolve ``(actor, module, rule)`` for an order or fill record."""
    actor = resolve_order_actor(order, known_ai_order_ids)
    if actor == "AI" or not isinstance(order, dict):
        return actor, None, None

    mod = str(order.get("module") or order.get("bot") or order.get("caller") or "").strip().lower() or None
    rule = (
        str(
            order.get("rule")
            or order.get("trigger_rule")
            or order.get("kind")
            or order.get("role")
            or order.get("motivation")
            or ""
        ).strip()
        or None
    )
    oid = extract_order_id(order, default="")
    raw_cid = _extract_order_cid(order)

    if isinstance(audit_origin_map, dict):
        for lookup_key in (str(oid) if oid else "", raw_cid):
            if lookup_key and lookup_key in audit_origin_map:
                a_mod, a_rule = audit_origin_map[lookup_key]
                if not mod and a_mod:
                    mod = a_mod
                if not rule and a_rule:
                    rule = a_rule

    if raw_cid:
        cid_up = raw_cid.upper()
        for pfx, bot_name in _BOT_CID_PREFIX_MODULES:
            if cid_up.startswith(f"{pfx}_") or cid_up == pfx or (pfx in ("AGS", "AGB") and cid_up.startswith(pfx)):
                if not mod:
                    mod = bot_name
                if not rule:
                    if pfx == "AGS":
                        rule = "SL_PROTECT"
                    elif pfx == "AGB":
                        rule = "BUY_TIER"
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
    parts: list[str] = []
    if module:
        parts.append(f"module={module}")
    if rule:
        parts.append(f"rule={rule}")
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
    if exec_f > 0 and rem_f > 0:
        norm_status = "PARTIALLY_FILLED"
    elif raw_status in ("OPEN", "PENDING", "WORKING", "RESTING", ""):
        norm_status = "NEW"
    else:
        norm_status = raw_status
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
        parent_oid = None
        for k in _PARENT_ORDER_ID_KEYS:
            cleaned_oid = _clean_single_order_id(r.get(k))
            if cleaned_oid is not None and cleaned_oid != "0":
                parent_oid = cleaned_oid
                break
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
    result: List[dict] = []
    for item in out:
        if "cost" in item:
            total_qty = item["qty"]
            cost_qty = item.get("cost_qty", total_qty)
            vwap = (item["cost"] / cost_qty) if cost_qty > 0 else item["fallback_price"]
            result.append({
                "orderId": item["orderId"],
                "side": item["side"],
                "price": vwap,
                "qty": total_qty,
                "timestamp": item["timestamp"],
            })
        else:
            result.append(item)
    return result


# Compatibility alias retained for existing internal and external imports.
_normalize_order = normalize_order

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
    raw = _clean_single_order_id(client_order_id) or ""
    if not raw or len(raw) > 128:
        raise ProviderError(f"{venue_label} client_order_id must contain 1-128 characters")
    try:
        encoded = raw.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ProviderError(f"{venue_label} client_order_id must be ASCII") from exc
    if exact_re.fullmatch(raw):
        return raw.lower()
    return hex_prefix + hashlib.sha256(domain_prefix + encoded).hexdigest()[:32]


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

    def round_quantity(self, symbol: str, qty: float) -> float:
        """Round or floor quantity to venue volume precision."""
        qty_f = safe_pos_float(qty, default=None)
        if qty_f is None:
            return 0.0
        try:
            pp = self.pair_precision(symbol) if hasattr(self, "pair_precision") else None
            dec = exact_nonneg_int(getattr(pp, "volume_decimals", None) if pp is not None else None, default=None)
            if dec is not None:
                factor = 10 ** min(dec, 18)
                return math.floor(qty_f * factor + 1e-12) / factor
        except Exception:
            pass
        return qty_f

    def round_amount(self, symbol: str, qty: float) -> float:
        """Backward-compatible alias for round_quantity."""
        return self.round_quantity(symbol, qty)

    def round_price(self, symbol: str, price: float) -> float:
        """Round price to venue price precision (tick size / price decimals)."""
        price_f = safe_pos_float(price, default=None)
        if price_f is None:
            return 0.0
        try:
            pp = self.pair_precision(symbol) if hasattr(self, "pair_precision") else None
            dec = exact_nonneg_int(getattr(pp, "price_decimals", None) if pp is not None else None, default=None)
            if dec is not None:
                return round(price_f, min(dec, 18))
        except Exception:
            pass
        return price_f

    def min_order_qty(self, symbol: str) -> float:
        """Return venue minimum order quantity; 0.0 means no known minimum."""
        try:
            pp = self.pair_precision(symbol) if hasattr(self, "pair_precision") else None
            if pp is not None:
                return safe_pos_float(getattr(pp, "order_min", 0.0), default=0.0)
        except Exception:
            pass
        return 0.0

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
        return order_guard.weight_limit(
            self, symbol, side, price, qty,
            available_qty=available_qty)

    def fee_cap_quantity(self, symbol: str, side: str, price: float,
                         available_qty: float) -> float:
        from providers.quantity import venue_fee_cap_quantity
        return venue_fee_cap_quantity(self.name, side, price, available_qty)

    def quantity_decision(self, symbol: str, side: str, price: float,
                          qty: float, **kwargs):
        from providers.quantity import decide_quantity
        return decide_quantity(self, symbol, side, price, qty, **kwargs)

    def adjust_order_price(self, symbol: str, side: str, price: float,
                           cancel_opposite: bool = True) -> float:
        return price

    def cancel_opposite_orders(self, symbol: str, side: str,
                               requested_price: float) -> None:
        """Cancel adverse opposing orders when the venue implements this policy."""
        return None

    def profit_guard_window_ref(self, symbol: str, side: str, safeback_sec, regime_context=None, reference_mode=None, module=None):
        import order_guard
        window_s = order_guard.get_window_cfg(
            self.name,
            symbol=symbol,
            order_type=side,
            regime_context=regime_context,
            module=module,
            reference_mode=reference_mode,
        )
        win_f = safe_pos_float(window_s, default=None)
        window_s = win_f if win_f is not None else safe_pos_float(safeback_sec, default=0.0)
        wr_kwargs = {}
        if reference_mode is not None:
            wr_kwargs["reference_mode"] = reference_mode
        if module is not None:
            wr_kwargs["module"] = module
        return order_guard.window_reference(self, symbol, side, window_s, **wr_kwargs)

    def last_opposite_fill(self, symbol: str, order_type: str,
                           since_s: float = 90 * 24 * 3600) -> Optional[float]:
        ot_norm = normalize_side(order_type)
        if ot_norm == "BUY":
            opposite = "SELL"
        elif ot_norm == "SELL":
            opposite = "BUY"
        else:
            return None
        eff_since_s = safe_nonneg_float(since_s, default=90.0 * 24.0 * 3600.0)
        fills = self.get_orders(symbol, opposite, eff_since_s)
        if fills is None:
            raise ProviderError(f"last_opposite_fill({symbol}): failed to read order history")
        valid_fills = [
            order for order in fills
            if isinstance(order, dict) and extract_order_price(order, default=0.0) > 0.0
        ]
        if not valid_fills:
            return None
        latest = max(valid_fills, key=lambda order: extract_timestamp_ms(order, default=0) or 0)
        price = extract_order_price(latest)
        return price or None

    def ohlc_closes(self, symbol: str, interval_min: int) -> list:
        """Return completed-bar closing prices from closed interval series."""
        if hasattr(self, "ohlc_series"):
            series = self.ohlc_series(symbol, interval_min)
            closes = getattr(series, "closes", None) if series is not None else None
            return list(closes) if closes is not None else []
        return []
