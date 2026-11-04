"""Provider primitives with no registry or venue-module dependencies."""

import contextlib
import hashlib
import math
import os
import re
import sys
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, List, Optional

from botcore import (
    _parse_bool_flag,
    bool_env,
    env_value,
    env_values_map,
    exact_nonneg_int,
    exact_pos_int,
    extract_current_price_item,
    obj_attr,
    resolve_now_ts,
    safe_float,
    safe_int,
    safe_nonneg_float,
    safe_pos_float,
)
from cachepaths import (
    cache_path,
    cache_symbol_variants,
    canonical_base_asset,
    canonical_symbol_key,
    clean_symbol_key,
    compatible_quote_suffixes,
    get_cachedb_dir,
    normalize_venue_name,
    resolve_cache_dir_path,
    resolve_cachedb_filepath,
    resolve_kraken_lookup_symbol,
    split_symbol_assets,
)
from instrument_registry import casefold_get
from state_io import read_json_dict_recovering, save_json_state


_LIVE_FETCH_LOCAL = threading.local()


@contextlib.contextmanager
def live_price_fetch_context():
    """Mark the current thread as performing a live venue price poll for cacheManager."""
    depth = int(getattr(_LIVE_FETCH_LOCAL, "depth", 0) or 0)
    _LIVE_FETCH_LOCAL.depth = depth + 1
    try:
        yield
    finally:
        _LIVE_FETCH_LOCAL.depth = max(0, depth)


def is_in_live_price_fetch() -> bool:
    """Return True when called inside a cacheManager remote price poll."""
    return int(getattr(_LIVE_FETCH_LOCAL, "depth", 0) or 0) > 0


_TIMESTAMP_KEYS = ("timestamp", "time", "updateTime", "transactTime", "opentm", "closetm", "workingTime", "T", "E", "ts")
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
    if numeric is not None:
        return numeric
    if isinstance(value, str):
        s = value.strip()
        if len(s) >= 10 and ("-" in s or "T" in s):
            try:
                iso_s = s[:-1] + "+00:00" if s.endswith(("Z", "z")) else s
                dt = datetime.fromisoformat(iso_s)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                ts_s = dt.timestamp()
                return ts_s if ts_s >= 0.0 else None
            except Exception:
                return None
    return None


def normalize_timestamp_ms(value: Any, default: Optional[int] = None) -> Optional[int]:
    """Normalize a raw timestamp (in seconds, milliseconds, or ISO-8601 string) to integer milliseconds.

    Epoch values in ``[1_000_000_000, 100_000_000_000)`` (all Unix seconds between
    2001 and 2286) and non-integer float seconds are converted to milliseconds.
    Millisecond values (``>= 100_000_000_000``) and small synthetic test integers
    (``< 1_000_000_000``) are preserved as-is.
    """
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
    return ts_ms / 1000.0 if ts_ms >= 100_000_000_000 else float(ts_ms)


def _is_fresh_obs_ts(obs_ts_s: Optional[float], now_s: float, max_age_s: float) -> bool:
    return obs_ts_s is not None and -5.0 <= (now_s - obs_ts_s) <= max_age_s


def extract_timestamp_ms(order: Any, default: Optional[int] = None) -> Optional[int]:
    """Extract and normalize an order/fill timestamp in milliseconds from any venue shape."""
    return _first_mapped(order, _TIMESTAMP_KEYS, lambda v: normalize_timestamp_ms(v, default=None), default)


def extract_timestamp_s(order: Any, default: Optional[float] = None) -> Optional[float]:
    """Extract and normalize an order/fill timestamp in seconds from any venue shape."""
    numeric = _first_mapped(order, _TIMESTAMP_KEYS, _parse_epoch_or_iso_s)
    return default if numeric is None else _ts_ms_to_s(numeric)


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
    for k in ("remainingQty", "remaining_qty", "leavesQty", "leaves_qty"):
        rem_f = safe_nonneg_float(order.get(k), default=None)
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
        _quote_aliases = {"ZUSD": "USD", "ZEUR": "EUR", "ZUSDT": "USDT"}
        fee_base_canon = canonical_base_asset(fee_asset) if fee_asset else ""
        base_canon = canonical_base_asset(base_u) if base_u else ""
        fee_quote_canon = _quote_aliases.get(fee_asset, fee_asset)
        quote_canon = _quote_aliases.get(quote_u, quote_u)
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
    ("MO_MN", "manual"),
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
    ("MN_OR", "manual"),
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
    ("MN", "manual"),
)
# Venue-native manual order client-ID prefixes (Binance app/web/API UIs); SSOT shared with order_id_context.
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
    if raw_cid.startswith("AI_"):
        return "AI"
    if raw_cid and known_ai_order_ids and raw_cid.lower() in known_ai_order_ids:
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
                        rem = raw_cid[len(pfx) :].lstrip("_-")
                        tokens = [t for t in re.split(r"[_-]+", rem) if t and t.upper() not in _GENERIC_CID_TOKENS]
                        if tokens:
                            cand = "_".join(tokens[:-1]) if len(tokens) > 1 and (tokens[-1].isdigit() or len(tokens[-1]) >= 8) else tokens[0]
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
        m_ts = re.search(r"(?:^|[_-])(1[789]\d{8}(?:\d{3})?)(?:$|[_-])", str(cid))
        if m_ts:
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
    if _parse_bool_flag(dry_run, True):
        return paper_label
    return real_label


from .quantity import (  # noqa: E402
    compute_fills_cost_basis,
    decide_quantity,
    remaining_average_cost,
    venue_fee_cap_quantity,
)
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


OHLC_INTERVALS_MIN_TO_STR: dict[int, str] = {
    1: "1m",
    3: "3m",
    5: "5m",
    15: "15m",
    30: "30m",
    60: "1h",
    120: "2h",
    240: "4h",
    360: "6h",
    480: "8h",
    720: "12h",
    1440: "1d",
    4320: "3d",
    10080: "1w",
    43200: "1M",
}

OHLC_INTERVALS_STR_TO_MIN: dict[str, int] = {
    v: k for k, v in OHLC_INTERVALS_MIN_TO_STR.items()
}

OHLC_CASE_INSENSITIVE_INTERVAL_ALIASES: dict[str, str] = {
    "1H": "1h",
    "2H": "2h",
    "4H": "4h",
    "6H": "6h",
    "8H": "8h",
    "12H": "12h",
    "1D": "1d",
    "3D": "3d",
    "1W": "1w",
}


def parse_ohlc_interval_minutes(
    interval: Any,
    *,
    str_to_min: Optional[dict[str, int]] = None,
    reject_float: bool = False,
    venue_label: str = "OHLC",
) -> int:
    """Parse an OHLC interval (integer minutes or string e.g. '1h', '60') into positive integer minutes."""
    if isinstance(interval, bool):
        raise ValueError(f"interval must be positive integer or string, not bool: {interval}")
    lookup = str_to_min if str_to_min is not None else OHLC_INTERVALS_STR_TO_MIN
    if isinstance(interval, str):
        cleaned = interval.strip()
        cleaned = OHLC_CASE_INSENSITIVE_INTERVAL_ALIASES.get(cleaned.upper(), cleaned)
        if cleaned not in lookup and cleaned.lower() in lookup and cleaned.upper() != "1M":
            cleaned = cleaned.lower()
        if cleaned in lookup:
            return lookup[cleaned]
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
        interval,
        str_to_min=s2m,
        reject_float=reject_float,
        venue_label=venue_label,
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
    now_ms = int(time.time() * 1000)
    timeout_f = safe_pos_float(timeout, default=None)
    if timeout_f is None or (strict_numeric_types and not isinstance(timeout, (int, float))):
        raise ValueError(f"invalid OHLC window parameters: timeout must be finite and positive: {timeout!r}")
    if end_ms is not None:
        end_f = safe_pos_float(end_ms, default=None)
        if end_f is None or int(end_f) <= 0 or (strict_numeric_types and not isinstance(end_ms, (int, float))):
            raise ValueError(f"invalid OHLC window parameters: end_ms must be finite and positive: {end_ms!r}")
        final_end = int(end_f)
    else:
        final_end = now_ms
    if final_end <= 0:
        raise ValueError(f"end_ms must be positive: {end_ms!r}")
    if start_ms is not None:
        start_f = safe_nonneg_float(start_ms, default=None)
        if start_f is None or (strict_numeric_types and not isinstance(start_ms, (int, float))):
            raise ValueError(f"invalid OHLC window parameters: start_ms must be finite and non-negative: {start_ms!r}")
        final_start = int(start_f)
    elif lookback_days is not None:
        lb_f = safe_pos_float(lookback_days, default=None)
        if lb_f is None or int(lb_f) <= 0 or (strict_numeric_types and not isinstance(lookback_days, (int, float))):
            raise ValueError(f"invalid OHLC window parameters: lookback_days must be finite and positive: {lookback_days!r}")
        final_start = final_end - int(lb_f) * 86400 * 1000
    elif lookback_hours is not None:
        lb_f = safe_pos_float(lookback_hours, default=None)
        if lb_f is None or (strict_numeric_types and not isinstance(lookback_hours, (int, float))):
            raise ValueError(f"invalid OHLC window parameters: lookback_hours must be finite and positive: {lookback_hours!r}")
        final_start = int(final_end - lb_f * 3600 * 1000)
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
        if drop_incomplete and now_ms is not None:
            if (close_i >= now_ms) if close_inclusive else (close_i > now_ms):
                return None
    if h_v < l_v or h_v < max(o_v, c_v) or l_v > min(o_v, c_v):
        return None
    ts_sec = open_i // 1000 if ts_in_ms else open_i
    if ts_sec <= 0:
        return None
    return {
        "timestamp": ts_sec,
        "open": o_v,
        "high": h_v,
        "low": l_v,
        "close": c_v,
    }


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

    _DEFAULT_FILLS_CACHE_FILENAME: Optional[str] = None
    _FILLS_CACHE_PATH_MODULE_ATTR: Optional[str] = None
    _FILLS_CACHE_DEFAULT_MODULE_ATTR: Optional[str] = None
    _FILLS_CACHE_MAX_STALE_S: float = 60.0
    _fills_match_fn: Any = None
    _fills_normalize_fn: Any = None

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

    def _fills_cache_file(self) -> Optional[str]:
        custom = getattr(self, "_ssot_fills_cache_path", None)
        if custom:
            return str(custom)
        mod = sys.modules.get(type(self).__module__)
        mod_path = (
            getattr(mod, self._FILLS_CACHE_PATH_MODULE_ATTR, None)
            if (mod is not None and self._FILLS_CACHE_PATH_MODULE_ATTR)
            else None
        )
        def_path = (
            getattr(mod, self._FILLS_CACHE_DEFAULT_MODULE_ATTR, None)
            if (mod is not None and self._FILLS_CACHE_DEFAULT_MODULE_ATTR)
            else None
        )
        if mod_path and def_path and mod_path != def_path:
            return str(mod_path)
        if self._allow_ssot_price_cache():
            if mod_path:
                return str(mod_path)
            if self._DEFAULT_FILLS_CACHE_FILENAME:
                return cache_path(self._DEFAULT_FILLS_CACHE_FILENAME)
        return None

    def _fills_from_cache(self, symbol: str) -> Optional[List[dict]]:
        """Read symbol fills from the shared SSOT trade cache, or return None for API fallback."""
        cache_file = self._fills_cache_file()
        if not cache_file:
            return None
        mod = sys.modules.get(type(self).__module__)
        stale_s = getattr(mod, "_CACHE_MAX_STALE_S", None) if mod is not None else None
        return self.read_ssot_fills_cache(
            symbol,
            cache_file,
            max_stale_sec=stale_s if stale_s is not None else self._FILLS_CACHE_MAX_STALE_S,
            match_fn=getattr(type(self), "_fills_match_fn", None),
            normalize_fn=getattr(type(self), "_fills_normalize_fn", None),
        )

    def _invalidate_fills_cache(self, symbol: str) -> None:
        """Expire ``fetchtime`` for ``symbol`` in this provider's configured SSOT trade cache."""
        cache_file = self._fills_cache_file()
        if cache_file:
            self.invalidate_ssot_fills_cache(
                symbol,
                cache_file,
                match_fn=getattr(type(self), "_fills_match_fn", None),
            )

    def get_orders(self, symbol: str, side: Optional[str], since_s: float) -> List[dict]:
        """Return normalized recent orders (fills aggregated by orderId), reading SSOT cache first and falling back to ``_fills_from_api``."""
        api_fn = getattr(self, "_fills_from_api", None)
        if not callable(api_fn):
            return []
        self.validate_symbol(symbol)
        since_val = safe_nonneg_float(since_s, default=None)
        if since_val is None:
            return []
        try:
            cache_fn = getattr(self, "_fills_from_cache", None)
            rows = cache_fn(symbol) if callable(cache_fn) else None
            if rows is None:
                rows = api_fn(symbol)
            if rows is None:
                raise ProviderError(f"get_orders({symbol}): fills history unavailable")
            cutoff_ms = (time.time() - since_val) * 1000.0
            return aggregate_fills_by_order_id(rows, cutoff_ms=cutoff_ms, side=side)
        except ProviderError:
            raise
        except Exception as e:  # noqa: BLE001
            print(f"[{self.name}] get_orders({symbol},{side}) failed: {e}")
            raise ProviderError(f"get_orders({symbol}): {e}") from e

    def get_trades(self, symbol: str, since_s: float) -> List[dict]:
        return self.get_orders(symbol, None, since_s)

    @staticmethod
    def resolve_venue_instrument_spec(venue: Any, symbol: str) -> Any:
        """Resolve ``InstrumentSpec`` for ``symbol`` on ``venue`` from ``instrument_registry``."""
        if isinstance(symbol, bool) or not isinstance(symbol, str) or not symbol.strip():
            return None
        sym_u = symbol.strip().upper()
        sym_canon = canonical_symbol_key(sym_u)
        try:
            import instrument_registry

            venue_slug = normalize_venue_name(getattr(venue, "name", venue) or "")
            return next(
                (
                    item
                    for item in instrument_registry.select_instruments(venue_slug).values()
                    if item.symbol.upper() == sym_u or (sym_canon and canonical_symbol_key(item.symbol) == sym_canon)
                ),
                None,
            )
        except Exception:
            return None

    def _resolve_venue_instrument_spec(self, symbol: str) -> Any:
        return self.resolve_venue_instrument_spec(getattr(self, "name", ""), symbol)

    def position_cost_basis(self, symbol: str, held_qty: float, since_s: float) -> Optional[float]:
        """Return inventory-reconciled acquisition cost from fresh SSOT fills, or None when unknown."""
        since_val = safe_pos_float(since_s, default=None)
        if since_val is None:
            raise ValueError("since_s must be finite and positive")
        if isinstance(symbol, bool) or not isinstance(symbol, str) or not symbol.strip():
            return None
        fills_fn = getattr(self, "_fills_from_cache", None)
        if not callable(fills_fn):
            return None
        rows = fills_fn(symbol)
        if rows is None:
            return None
        sym_u = symbol.strip().upper()
        spec = self._resolve_venue_instrument_spec(sym_u)
        base_asset = spec.base if spec is not None else ""
        quote_asset = spec.quote if spec is not None else ""
        if not base_asset:
            parts = split_symbol_assets(sym_u)
            if parts is not None:
                base_asset, quote_asset = parts
        return compute_fills_cost_basis(
            rows,
            held_qty,
            since_val,
            base_asset=base_asset,
            quote_asset=quote_asset,
        )

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

    def execution_enabled(self) -> bool:
        """Return whether this provider may create a real venue order now.

        Providers with an explicit dry/live switch override this hook. The shared
        placement pipeline checks it before creating durable retry state so a dry
        validation cannot become a live order after a later configuration change.
        """
        return True

    @staticmethod
    def resolve_strategy_dry_run(
        force_paper: bool, strategy_execute: bool, venue_live_enabled: bool
    ) -> bool:
        """Combine CLI `--paper`, strategy gate (`STRAT_EXECUTE`), and venue live gate (`*_LIVE_ORDERS`)."""
        return bool(
            _parse_bool_flag(force_paper, True)
            or not _parse_bool_flag(strategy_execute, False)
            or not _parse_bool_flag(venue_live_enabled, False)
        )

    def strategy_dry_run(self, force_paper: bool, strategy_execute: bool) -> bool:
        """Return whether a spot strategy should run in paper mode for this provider."""
        return self.resolve_strategy_dry_run(
            force_paper, strategy_execute, self.execution_enabled()
        )

    def _round_required_qty(self, symbol: str, qty: float, err_prefix: str = "submit_order") -> float:
        """Round ``qty`` using ``round_quantity`` and raise ``ProviderError`` if non-positive."""
        rounded = self.round_quantity(symbol, qty)
        if rounded <= 0:
            raise ProviderError(f"{err_prefix}({symbol}): invalid non-positive quantity qty={rounded}")
        return rounded

    @staticmethod
    def _execute_confirmed_cancel(order_id: Any, cancel_fn: Any) -> None:
        """Execute ``cancel_fn()`` and raise ``ProviderError`` unless cancellation is confirmed."""
        try:
            if not cancel_fn():
                raise ProviderError(
                    f"cancel_order({order_id}): the venue did not confirm the cancellation"
                )
        except ProviderError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"cancel_order({order_id}): {e}") from e

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

    # ── Agnostic SSOT cacheManager coupling (price, 24h history, short trend) ─

    def _ssot_symbol_candidates(self, symbol: str) -> tuple[str, ...]:
        """Return venue-scoped symbol keys to look up in SSOT cacheManager stores."""
        if isinstance(symbol, bool) or not isinstance(symbol, str) or not symbol.strip():
            return ()
        raw_u = symbol.strip().upper()
        cleaned = clean_symbol_key(raw_u)
        parts = split_symbol_assets(raw_u)
        canon = canonical_symbol_key(raw_u) if parts is not None else ""
        kr_lookup = resolve_kraken_lookup_symbol(raw_u) if (parts is not None and parts[1] == "USD") else ""
        return tuple(dict.fromkeys(c for c in (cleaned, raw_u, canon, kr_lookup) if c))

    _REAL_CLIENT_CLASS_NAMES = frozenset({"KrakenClient", "HLClient"})
    _DEFINITIVE_REFUSAL_TOKENS = (
        "insufficient",
        "minimum value",
        "order minimum",
        "volume minimum",
        "cost minimum",
        "invalid price",
        "invalid size",
        "invalid order",
        "invalid arguments",
        "positions limit exceeded",
        "tick size",
        "could not immediately match",
        "price must be",
        "too far from",
    )

    @classmethod
    def _is_definitive_refusal(cls, error: object) -> bool:
        """Return True when a venue error message indicates a permanent order rejection."""
        text = str(error or "").lower()
        return any(token in text for token in cls._DEFINITIVE_REFUSAL_TOKENS)

    @staticmethod
    def clamp_price_to_bbo_mid(
        price: Optional[float],
        bid: Any,
        ask: Any,
        *,
        max_spread_ratio: float = 0.05,
    ) -> Optional[float]:
        """Return ``price`` clamped to the BBO mid ``(bid + ask) / 2`` when ``price`` falls outside a tight ``[bid, ask]`` spread."""
        px_f = safe_pos_float(price, default=None) if not isinstance(price, bool) else None
        bid_f = safe_pos_float(bid, default=None) if not isinstance(bid, bool) else None
        ask_f = safe_pos_float(ask, default=None) if not isinstance(ask, bool) else None
        if (
            bid_f is not None
            and ask_f is not None
            and bid_f <= ask_f
            and (ask_f - bid_f) / bid_f <= max_spread_ratio
            and (px_f is None or px_f < bid_f or px_f > ask_f)
        ):
            return round((bid_f + ask_f) * 0.5, 8)
        return px_f

    def _allow_ssot_price_cache(self) -> bool:
        """Return True when this provider should consult cacheManager SSOT before venue HTTP."""
        if is_in_live_price_fetch():
            return False
        if getattr(self, "_ssot_cache_enabled", None) is False:
            return False
        has_explicit_ssot = bool(
            getattr(self, "_force_ssot_cache", False)
            or getattr(self, "_ssot_price_cache_path", None)
            or getattr(self, "_ssot_trend_cache_path", None)
            or getattr(self, "_ssot_24price_cache_dir", None)
            or getattr(self, "_ssot_fills_cache_path", None)
        )
        inj_client = getattr(self, "_cli", None) or getattr(self, "_client", None)
        if (
            inj_client is not None
            and type(inj_client).__name__ not in self._REAL_CLIENT_CLASS_NAMES
            and not has_explicit_ssot
        ):
            return False
        if has_explicit_ssot:
            return True
        if "pytest" in sys.modules:
            return False
        return True

    @staticmethod
    def _parse_ssot_price_point(entry: Any, fallback_ts: Optional[float] = None) -> Optional[tuple[int, float]]:
        if isinstance(entry, dict):
            ts_ms = extract_timestamp_ms(entry, default=None)
            if ts_ms is None and fallback_ts is not None:
                ts_ms = normalize_timestamp_ms(fallback_ts, default=None)
            px_f = extract_current_price_item(entry)
            if px_f is None:
                px_f = safe_pos_float(entry.get("price", entry.get("px")), default=None)
            if ts_ms is not None and ts_ms > 0 and px_f is not None:
                return ts_ms, px_f
            return None
        if isinstance(entry, (list, tuple)):
            if len(entry) >= 2 and not isinstance(entry[0], (list, tuple, dict)):
                ts_ms = normalize_timestamp_ms(entry[0], default=None)
                px_f = safe_pos_float(entry[1], default=None)
                if ts_ms is not None and ts_ms > 0 and px_f is not None:
                    return ts_ms, px_f
                return None
            for sub in reversed(entry):
                if not isinstance(sub, (list, tuple, dict)):
                    continue
                parsed = MarketDataProvider._parse_ssot_price_point(sub, fallback_ts=fallback_ts)
                if parsed is not None:
                    return parsed
            return None
        px_f = extract_current_price_item(entry)
        if px_f is not None and fallback_ts is not None:
            ts_ms = normalize_timestamp_ms(fallback_ts, default=None)
            if ts_ms is not None and ts_ms > 0:
                return ts_ms, px_f
        return None

    @staticmethod
    def _read_ssot_current_price_candidates(
        candidates: tuple[str, ...] | List[str],
        price_path: str,
        max_age_sec: float,
        *,
        now_s: Optional[float] = None,
        use_memory_manager: bool = True,
    ) -> Optional[float]:
        max_age = safe_pos_float(max_age_sec, default=12.0) or 12.0
        if use_memory_manager:
            cm = sys.modules.get("cacheManager")
            cpm = getattr(cm, "_current_price_instance", None) if cm is not None else None
            if cpm is not None and hasattr(cpm, "get_cached_price_observation"):
                for cand in candidates:
                    try:
                        obs = cpm.get_cached_price_observation(cand, max_age)
                        if obs is not None:
                            px = safe_pos_float(obs[1], default=None)
                            if px is not None:
                                return px
                    except Exception:
                        continue
        if not price_path or not os.path.exists(price_path):
            return None
        try:
            data = read_json_dict_recovering(price_path)
            if not isinstance(data, dict):
                return None
            items = data.get("items") if isinstance(data.get("items"), dict) else data
            fetchtime_map = data.get("fetchtime") if isinstance(data.get("fetchtime"), dict) else {}
            file_ts = extract_timestamp_s(data, default=None)
            eff_now = resolve_now_ts(now_s)
            for cand in candidates:
                raw_entry = casefold_get(items, cand)
                if raw_entry is None:
                    continue
                fb_ts = casefold_get(fetchtime_map, cand)
                if fb_ts is None:
                    fb_ts = file_ts
                parsed = MarketDataProvider._parse_ssot_price_point(raw_entry, fallback_ts=fb_ts)
                if parsed is None:
                    continue
                ts_ms, px = parsed
                if _is_fresh_obs_ts(_ts_ms_to_s(ts_ms), eff_now, max_age):
                    return px
        except Exception:
            pass
        return None

    def get_ssot_cached_price(self, symbol: str, max_age_sec: float = 12.0) -> Optional[float]:
        """Read fresh current price for ``symbol`` from CacheCurrentPriceManager / cache_current_price.json."""
        if not self._allow_ssot_price_cache():
            return None
        max_age = safe_pos_float(max_age_sec, default=12.0) or 12.0
        candidates = [c for c in self._ssot_symbol_candidates(symbol) if c]
        if not candidates:
            return None
        custom_path = getattr(self, "_ssot_price_cache_path", None)
        price_path = str(custom_path) if custom_path else cache_path("cache_current_price.json")
        return self._read_ssot_current_price_candidates(
            candidates,
            price_path,
            max_age,
            use_memory_manager=(custom_path is None),
        )

    def get_ssot_price_history(
        self,
        symbol: str,
        lookback_hours: float,
        max_latest_age_sec: float = 120.0,
    ) -> Optional[list[dict]]:
        """Read 24h tick history from Cache24PriceManager when fresh and covering ``lookback_hours``."""
        if not self._allow_ssot_price_cache():
            return None
        lh_f = safe_pos_float(lookback_hours, default=None)
        if lh_f is None or lh_f > 24.0:
            return None
        candidates = [c for c in self._ssot_symbol_candidates(symbol) if c]
        if not candidates:
            return None
        now_s = time.time()
        cutoff_ms = int((now_s - lh_f * 3600.0) * 1000)
        custom_dir = getattr(self, "_ssot_24price_cache_dir", None)
        for cand in candidates:
            c24_path = (
                os.path.join(str(custom_dir), f"cache_24_price_{cand}.json")
                if custom_dir
                else cache_path(f"cache_24_price_{cand}.json")
            )
            if not os.path.exists(c24_path):
                continue
            try:
                raw = read_json_dict_recovering(c24_path)
                if not isinstance(raw, dict):
                    continue
                if isinstance(raw.get("items"), list):
                    entries = raw.get("items")
                else:
                    items_map = raw.get("items") if isinstance(raw.get("items"), dict) else raw
                    entries = casefold_get(items_map, cand)
                if not isinstance(entries, list) or len(entries) < 2:
                    continue
                valid: list[tuple[int, float]] = []
                for entry in entries:
                    parsed = MarketDataProvider._parse_ssot_price_point(entry)
                    if parsed is not None:
                        valid.append(parsed)
                if len(valid) < 2:
                    continue
                valid.sort(key=lambda pt: pt[0])
                if not _is_fresh_obs_ts(valid[-1][0] / 1000.0, now_s, max_latest_age_sec):
                    continue
                out = [{"timestamp": ts, "price": px} for ts, px in valid if ts >= cutoff_ms]
                if len(out) < 2:
                    continue
                if custom_dir is None and lh_f > 0.25 and (out[-1]["timestamp"] - out[0]["timestamp"]) < (lh_f * 3600.0 * 1000.0 * 0.75):
                    continue
                return out
            except Exception:
                continue
        return None

    @staticmethod
    def read_ssot_short_trend_file(
        trend_path: str,
        candidates: tuple[str, ...] | List[str],
        max_age_sec: float = 15.0,
        *,
        now_s: Optional[float] = None,
    ) -> Optional[dict]:
        """Read a fresh short-trend snapshot for ``candidates`` from ``trend_path``."""
        if not trend_path:
            return None
        max_age = safe_pos_float(max_age_sec, default=15.0) or 15.0
        try:
            data = read_json_dict_recovering(trend_path)
            if not isinstance(data, dict):
                return None
            items = data.get("items") if isinstance(data.get("items"), dict) else data
            eff_now = resolve_now_ts(now_s)
            for cand in candidates:
                if not cand:
                    continue
                snap = casefold_get(items, cand)
                if not isinstance(snap, dict):
                    continue
                raw_ts = extract_timestamp_s(snap, default=None)
                if raw_ts is not None and raw_ts > 0 and _is_fresh_obs_ts(raw_ts, eff_now, max_age):
                    return dict(snap)
        except Exception:
            pass
        return None

    def trend_snapshot(self, symbol: str, max_age_sec: float = 15.0) -> Optional[dict]:
        """Return a fresh short-trend snapshot for ``symbol`` from CacheShortTrendManager / cache_short_trend.json."""
        if not self._allow_ssot_price_cache():
            return None
        max_age = safe_pos_float(max_age_sec, default=15.0) or 15.0
        candidates = [c for c in self._ssot_symbol_candidates(symbol) if c]
        if not candidates:
            return None
        custom_path = getattr(self, "_ssot_trend_cache_path", None)
        now_s = time.time()
        if custom_path is None:
            cm = sys.modules.get("cacheManager")
            stm = getattr(cm, "_short_trend_instance", None) if cm is not None else None
            if stm is not None:
                stm_mem = getattr(stm, "_mem", None)
                if isinstance(stm_mem, dict):
                    lock_ctx = getattr(stm, "_lock", None) or getattr(stm, "lock", None) or contextlib.nullcontext()
                    try:
                        with lock_ctx:
                            for cand in candidates:
                                snap = casefold_get(stm_mem, cand)
                                if not isinstance(snap, dict):
                                    continue
                                if _is_fresh_obs_ts(extract_timestamp_s(snap, default=None), now_s, max_age):
                                    return dict(snap)
                    except Exception:
                        pass
                elif hasattr(stm, "get_fresh_snapshot"):
                    for cand in candidates:
                        try:
                            snap = stm.get_fresh_snapshot(cand)
                            if isinstance(snap, dict):
                                raw_ts = extract_timestamp_s(snap, default=None)
                                if raw_ts is None or _is_fresh_obs_ts(raw_ts, now_s, max_age):
                                    return dict(snap)
                        except Exception:
                            continue
        trend_path = str(custom_path) if custom_path else cache_path("cache_short_trend.json")
        return self.read_ssot_short_trend_file(trend_path, candidates, max_age_sec=max_age, now_s=now_s)

    def trend(self, symbol: str, max_age_sec: float = 15.0) -> float:
        """Return instant-trend slope (>0 up, <0 down, 0.0 neutral/unknown) from cacheManager SSOT."""
        snap = self.trend_snapshot(symbol, max_age_sec=max_age_sec)
        if not isinstance(snap, dict):
            return 0.0
        try:
            from trailing_core import extract_short_trend_slope

            slope = safe_float(extract_short_trend_slope(snap, symbol), default=0.0)
            if slope != 0.0:
                return slope
        except Exception:
            pass
        for key in ("trend", "final_trend", "growth_coefficient", "gradient_recent", "slope_small"):
            val = safe_float(snap.get(key), default=None)
            if val is not None and val != 0.0:
                return val
        return 0.0

    def trend_direction(self, symbol: str, max_age_sec: float = 15.0) -> Optional[str]:
        """Return short-trend direction ('up', 'down', or 'flat') from cacheManager SSOT, or None if unavailable."""
        snap = self.trend_snapshot(symbol, max_age_sec=max_age_sec)
        if not isinstance(snap, dict):
            return None
        try:
            from cachetrend import CacheShortTrendManager

            if CacheShortTrendManager._eval_upward_formula(snap):
                return "up"
            inv_snap = dict(snap)
            for key in ("growth_coefficient", "gradient_recent", "slope_small", "final_trend"):
                val = safe_float(inv_snap.get(key), default=None)
                if val is not None:
                    inv_snap[key] = -val
            if CacheShortTrendManager._eval_upward_formula(inv_snap):
                return "down"
            return "flat"
        except Exception:
            return None

    @staticmethod
    def normalize_orderbook_payload(raw_data: Any) -> Optional[dict]:
        """Normalize Binance, Kraken, or Hyperliquid L2 orderbook payloads into ``{'bids': [(px, qty), ...], 'asks': [(px, qty), ...]}``."""
        if not isinstance(raw_data, dict):
            return None
        data = raw_data
        if "bids" not in data and isinstance(data.get("levels"), list) and len(data["levels"]) >= 2:
            hl_bids, hl_asks = data["levels"][0], data["levels"][1]
            if isinstance(hl_bids, list) and isinstance(hl_asks, list):
                data = {
                    "bids": [[lvl.get("px"), lvl.get("sz")] if isinstance(lvl, dict) else lvl for lvl in hl_bids],
                    "asks": [[lvl.get("px"), lvl.get("sz")] if isinstance(lvl, dict) else lvl for lvl in hl_asks],
                }
        elif "bids" not in data and isinstance(data.get("result"), dict):
            krk_book = next(
                (v for v in data["result"].values() if isinstance(v, dict) and "bids" in v),
                None,
            )
            if isinstance(krk_book, dict):
                data = krk_book
            else:
                return None

        def _parse_side(rows: Any) -> list[tuple[float, float]]:
            if not isinstance(rows, list):
                return []
            out_side: list[tuple[float, float]] = []
            for item in rows:
                if isinstance(item, dict):
                    pf = safe_pos_float(item.get("px", item.get("price")), default=None)
                    qf = safe_pos_float(item.get("sz", item.get("qty", item.get("volume"))), default=None)
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    pf = safe_pos_float(item[0], default=None)
                    qf = safe_pos_float(item[1], default=None)
                else:
                    continue
                if pf is not None and qf is not None:
                    out_side.append((pf, qf))
            return out_side

        bids = _parse_side(data.get("bids"))
        asks = _parse_side(data.get("asks"))
        if not bids or not asks:
            return None
        bids.sort(key=lambda x: x[0], reverse=True)
        asks.sort(key=lambda x: x[0])
        return {"bids": bids, "asks": asks}

    @staticmethod
    def clamp_orderbook_limit(
        limit: Any,
        *,
        default: int = 200,
        min_limit: int = 5,
        max_limit: int = 500,
    ) -> int:
        """Clamp an orderbook depth limit to ``[min_limit, max_limit]``."""
        raw = exact_pos_int(limit, default=None)
        if raw is None:
            raw_f = safe_pos_float(limit, default=None)
            raw = int(raw_f) if (raw_f is not None and int(raw_f) > 0) else default
        return max(min_limit, min(max_limit, raw))

    def get_orderbook(self, symbol: str, limit: int = 200) -> Optional[dict]:
        """Return normalized L2 orderbook ``{'bids': [(px, qty), ...], 'asks': [(px, qty), ...]}`` for ``symbol``."""
        return None

    def read_ssot_fills_cache(
        self,
        symbol: str,
        cache_file: str,
        *,
        max_stale_sec: float = 90.0,
        match_fn: Any = None,
        normalize_fn: Any = None,
    ) -> Optional[List[dict]]:
        """Read normalized fills for ``symbol`` from a shared SSOT trade cache file (`items` + `fetchtime`)."""
        if not symbol or isinstance(symbol, bool) or not str(symbol).strip() or not cache_file:
            return None
        data = read_json_dict_recovering(str(cache_file), default=None)
        if not isinstance(data, dict):
            return None
        items = data.get("items")
        ft = data.get("fetchtime")
        if not isinstance(items, dict) or not isinstance(ft, dict):
            return None
        if callable(match_fn):
            keys = [k for k in items if match_fn(symbol, k)]
        else:
            cands = {c.upper() for c in self._ssot_symbol_candidates(symbol) if c}
            keys = [k for k in items if isinstance(k, str) and clean_symbol_key(k) in cands]
        if not keys:
            return None
        now_s = time.time()
        max_stale_s = safe_pos_float(max_stale_sec, default=90.0) or 90.0
        for key in keys:
            if not isinstance(items.get(key), list):
                continue
            ft_val = safe_pos_float(casefold_get(ft, key), default=None)
            if ft_val is None or not _is_fresh_obs_ts(_ts_ms_to_s(ft_val), now_s, max_stale_s):
                continue
            if callable(normalize_fn):
                return [normalize_fn(t) for t in items[key] if isinstance(t, dict)]
            return [dict(t) for t in items[key] if isinstance(t, dict)]
        return None

    def write_ssot_fills_cache(self, symbol: str, cache_file: str, fills: List[dict]) -> bool:
        """Persist ``fills`` for ``symbol`` into the shared SSOT trade cache file (`items` + `fetchtime`)."""
        if not symbol or isinstance(symbol, bool) or not str(symbol).strip() or not cache_file:
            return False
        sym_key = clean_symbol_key(symbol) or str(symbol).strip().upper()
        try:
            data = read_json_dict_recovering(str(cache_file), default=None)
            if not isinstance(data, dict):
                data = {"items": {}, "fetchtime": {}}
            items = data.get("items") if isinstance(data.get("items"), dict) else {}
            ft = data.get("fetchtime") if isinstance(data.get("fetchtime"), dict) else {}
            items[sym_key] = [dict(f) for f in (fills or ()) if isinstance(f, dict)]
            ft[sym_key] = int(time.time() * 1000)
            data["items"] = items
            data["fetchtime"] = ft
            save_json_state(str(cache_file), data)
            return True
        except Exception:
            return False

    def invalidate_ssot_fills_cache(
        self, symbol: str, cache_file: str, *, match_fn: Any = None
    ) -> None:
        """Expire ``fetchtime`` for ``symbol`` in the shared SSOT trade cache after order execution."""
        if not symbol or isinstance(symbol, bool) or not str(symbol).strip() or not cache_file:
            return
        try:
            if not os.path.exists(str(cache_file)):
                return
            data = read_json_dict_recovering(str(cache_file), default=None)
            if not isinstance(data, dict) or not isinstance(data.get("fetchtime"), dict):
                return
            cands = {c.upper() for c in self._ssot_symbol_candidates(symbol) if c}
            changed = False
            for k in list(data["fetchtime"].keys()):
                if not isinstance(k, str):
                    continue
                matched = (
                    bool(match_fn(symbol, k))
                    if callable(match_fn)
                    else (k.upper() in cands or clean_symbol_key(k) in cands)
                )
                if matched:
                    data["fetchtime"].pop(k, None)
                    changed = True
            if changed:
                save_json_state(str(cache_file), data)
        except Exception:
            pass


def resolve_ssot_price_from_cache(
    symbol: str,
    cache_dir: Optional[str] = None,
    max_age_sec: float = 60.0,
    *,
    quote_isolated: bool = True,
    now_ts: Optional[float] = None,
) -> Optional[float]:
    """Resolve a fresh positive spot price for ``symbol`` from SSOT cacheManager stores (0 ms local)."""
    if not isinstance(symbol, str) or not symbol.strip():
        return None
    sym_u = symbol.strip().upper()
    sym_clean = clean_symbol_key(sym_u)
    if not sym_clean:
        return None
    max_age = safe_pos_float(max_age_sec, default=60.0) or 60.0
    now_s = resolve_now_ts(now_ts)
    quotes = (
        compatible_quote_suffixes(sym_clean, default_usd=("USDC", "USDT", "USD", "FDUSD"))
        if quote_isolated
        else ("USDC", "USDT", "USD", "EUR")
    )
    cands = list(
        dict.fromkeys(
            c
            for c in (sym_clean, sym_u, *cache_symbol_variants(sym_clean, quotes=quotes))
            if isinstance(c, str) and c and "/" not in c
        )
    )
    is_default_dir = not cache_dir or str(cache_dir).strip() in ("", "cachedb")
    eff_dir = resolve_cache_dir_path(cache_dir) or get_cachedb_dir()
    price_path = str(resolve_cachedb_filepath(os.path.join(eff_dir, "cache_current_price.json")))
    px = MarketDataProvider._read_ssot_current_price_candidates(
        cands,
        price_path,
        max_age,
        now_s=now_s,
        use_memory_manager=(is_default_dir and "pytest" not in sys.modules),
    )
    if px is not None:
        return px

    for cand in cands:
        c24_path = str(resolve_cachedb_filepath(os.path.join(eff_dir, f"cache_24_price_{cand}.json")))
        if not os.path.exists(c24_path):
            continue
        try:
            raw = read_json_dict_recovering(c24_path)
            if not isinstance(raw, dict):
                continue
            entries = raw.get("items") if isinstance(raw.get("items"), list) else casefold_get(raw.get("items") if isinstance(raw.get("items"), dict) else raw, cand)
            if not isinstance(entries, list) or not entries:
                continue
            parsed = MarketDataProvider._parse_ssot_price_point(entries[-1], fallback_ts=extract_timestamp_s(raw, default=None))
            if parsed is not None:
                ts_ms, px = parsed
                if _is_fresh_obs_ts(_ts_ms_to_s(ts_ms), now_s, max_age):
                    return px
        except Exception:
            continue
    return None
