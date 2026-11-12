"""Stateless order, fill, client-order-ID, and OHLC normalization utilities."""

import hashlib
import importlib
import math
import re
import time
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence

from botcore import (
    ClosedPriceSeries,
    _first_mapped,
    _nonneg_abs_float,
    _parse_bool_flag,
    _parse_epoch_or_iso_s,
    _pos_abs_float,
    _ts_ms_to_s,
    clean_order_id,
    exact_int,
    exact_pos_int,
    extract_order_id,
    extract_order_price,
    extract_order_qty,
    extract_order_side,
    extract_timestamp_ms,
    extract_timestamp_s,
    normalize_epoch_sec,
    normalize_scalar_order_id,
    normalize_side,
    normalize_timestamp_ms,
    obj_attr,
    require_finite_float,
    require_int,
    resolve_now_ts,
    safe_float,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_nonneg_ts,
    safe_pos_float,
    safe_pos_int,
)
from cachepaths import canonical_base_asset, normalize_venue_name


class ProviderError(Exception):
    """Venue-neutral error raised by strict strategy-execution adapters."""


def require_pos_ohlc_interval(symbol: object, interval_min: Any) -> int:
    """Validate and return a positive integer OHLC interval in minutes or raise ``ProviderError``."""
    interval_i = exact_pos_int(interval_min, default=None)
    if interval_i is None:
        raise ProviderError(f"ohlc_series({symbol}): invalid interval_min={interval_min!r}")
    return interval_i


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


def resolve_market_flag(force: Any = False, market: Any = False, **kwargs) -> bool:
    """Return True when any market/force flag in args or kwargs is truthy."""
    return any(_parse_bool_flag(v, False) for v in (force, market, kwargs.get("market"), kwargs.get("force")))


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
        age_m = round(max(0.0, (eff_now_s * 1000.0 - ts_ms) / 60000.0), 1)
        rec["created_ts"] = round(ts_ms / 1000.0, 1)
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


VENUE_PROVIDER_SPECS: dict[str, tuple[str, str]] = {
    "binance": ("providers.binance_provider", "BinanceProvider"),
    "hyperliquid": ("providers.hyperliquid_provider", "HyperliquidProvider"),
    "kraken": ("providers.kraken_provider", "KrakenProvider"),
    "t212": ("providers.t212_provider", "T212Provider"),
}


def resolve_venue_provider_class(
    venue: Any,
    *,
    provider: Any = None,
    base_cls: Any = None,
    specs: Optional[dict[str, tuple[str, str]]] = None,
) -> Any:
    """Resolve the concrete ``MarketDataProvider`` subclass for ``venue`` or ``provider``."""
    spec_map = specs if isinstance(specs, dict) else VENUE_PROVIDER_SPECS
    if not isinstance(venue, bool) and venue is not None:
        if isinstance(venue, type) and (
            (base_cls is not None and issubclass(venue, base_cls)) or hasattr(venue, "supports_symbol")
        ):
            return venue
        if not isinstance(venue, (str, bytes)) and (
            (base_cls is not None and isinstance(venue, base_cls)) or hasattr(type(venue), "supports_symbol")
        ):
            return type(venue)
        spec = spec_map.get(normalize_venue_name(venue))
        if spec is not None:
            mod_path, cls_name = spec
            resolved = getattr(importlib.import_module(mod_path), cls_name, None)
            if resolved is not None:
                return resolved
    if provider is not None and not isinstance(provider, (str, bytes, bool)):
        if isinstance(provider, type) and (
            (base_cls is not None and issubclass(provider, base_cls)) or hasattr(provider, "supports_symbol")
        ):
            return provider
        if (base_cls is not None and isinstance(provider, base_cls)) or hasattr(type(provider), "supports_symbol"):
            return type(provider)
    return None


def derive_venue_128bit_client_order_id(venue: Any, client_order_id: Any) -> str:
    """Map one durable client ID to the declarative 128-bit clientOrderId form for ``venue``."""
    prov_cls = resolve_venue_provider_class(venue)
    spec = getattr(prov_cls, "_CLIENT_ID_128BIT_SPEC", None) if prov_cls is not None else None
    if spec is None:
        raise ProviderError(f"Unsupported 128-bit client_order_id venue: {venue!r}")
    venue_label, domain_prefix, exact_re, hex_prefix = spec
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


def all_venue_native_client_order_ids(client_order_id: Any) -> set[str]:
    """Return the lowercase venue-native forms of one durable client ID across all registered venues."""
    raw = clean_order_id(client_order_id)
    if not raw:
        return set()
    out = {raw.lower()}
    for venue_key in VENUE_PROVIDER_SPECS:
        prov_cls = resolve_venue_provider_class(venue_key)
        if getattr(prov_cls, "_CLIENT_ID_128BIT_SPEC", None) is None:
            continue
        try:
            out.add(derive_venue_128bit_client_order_id(prov_cls, raw).lower())
        except Exception:
            pass
    return out


def format_venue_intent_client_order_id(venue: Any, token: str) -> Optional[str]:
    """Format a 32-hex intent token into the venue's clientOrderId format (or ``None`` when unsupported)."""
    if isinstance(token, bool) or not isinstance(token, str) or not token.strip():
        return None
    prov_cls = resolve_venue_provider_class(venue)
    fmt = getattr(prov_cls, "_INTENT_CLIENT_ID_FORMAT", None) if prov_cls is not None else None
    return fmt.format(token=token.strip()) if isinstance(fmt, str) and fmt else None


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
        scaled_ts = safe_pos_float(t_f * 1000.0 if ts_in_seconds else t_f, default=None)
        if scaled_ts is None:
            continue
        ts_ms = safe_pos_int(round(scaled_ts), default=None)
        if ts_ms is None:
            continue
        out.append({"timestamp": ts_ms, "price": p_f})
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
                minutes = (exact_int(num_part, default=0) or 0) * _OHLC_UNIT_MULTIPLIERS[unit]
                if minutes <= 0:
                    raise ValueError(
                        "the interval must be positive" if require_unit_suffix else f"interval must be positive: {interval}"
                    )
                return minutes
        if require_unit_suffix:
            raise ValueError(f"invalid {venue_label} interval: {interval!r}; examples: 5m, 1h, 1d")
        digits = cleaned[1:] if cleaned.startswith("+") else cleaned
        if digits.isascii() and digits.isdigit():
            val = exact_int(digits, default=0) or 0
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
        scaled_i = (
            (safe_pos_int if pos else safe_nonneg_int)(f * scale_ms, default=None)
            if f is not None and scale_ms >= 1.0 and name != "timeout"
            else 1
        )
        if f is None or scaled_i is None or (
            strict_numeric_types and not isinstance(val, (int, float))
        ):
            kind = "positive" if pos else "non-negative"
            raise ValueError(f"invalid OHLC window parameters: {name} must be finite and {kind}: {val!r}")
        return f

    now_ms = safe_pos_int(time.time() * 1000.0, default=1) or 1
    timeout_f = _valid_num(timeout, "timeout")
    final_end = safe_pos_int(_valid_num(end_ms, "end_ms"), default=0) if end_ms is not None else now_ms
    if final_end <= 0:
        raise ValueError(f"end_ms must be positive: {end_ms!r}")
    if start_ms is not None:
        final_start = safe_nonneg_int(_valid_num(start_ms, "start_ms", pos=False), default=0) or 0
    elif lookback_days is not None:
        span_ms = safe_pos_int(_valid_num(lookback_days, "lookback_days", scale_ms=86400 * 1000) * 86400 * 1000, default=0) or 0
        final_start = max(0, final_end - span_ms)
    elif lookback_hours is not None:
        span_ms = safe_pos_int(_valid_num(lookback_hours, "lookback_hours", scale_ms=3600 * 1000) * 3600 * 1000, default=0) or 0
        final_start = max(0, final_end - span_ms)
    else:
        final_start = max(0, final_end - default_lookback_days * 86400 * 1000)
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
    open_i = math.trunc(ot_f)
    if open_i <= 0:
        return None
    if close_ts is not _MISSING:
        ct_f = safe_pos_float(close_ts, default=None)
        if ct_f is None:
            return None
        close_i = math.trunc(ct_f)
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


def summarize_order_position(buy_orders: Any, sell_orders: Any) -> dict[str, Any]:
    """Aggregate BUY/SELL order lists into position statistics (SSOT for verify/watch tools)."""
    valid_buys = [
        (o, q, p)
        for o in (buy_orders or [])
        if isinstance(o, dict)
        and (q := extract_order_qty(o)) > 0.0
        and (p := extract_order_price(o)) > 0.0
    ]
    valid_sells = [
        (o, q)
        for o in (sell_orders or [])
        if isinstance(o, dict)
        and (q := extract_order_qty(o)) > 0.0
    ]
    buy_qty = sum(q for _, q, _ in valid_buys)
    buy_val = sum(q * p for _, q, p in valid_buys)
    sell_qty = sum(q for _, q in valid_sells)
    avg_buy = (buy_val / buy_qty) if buy_qty > 0.0 else 0.0
    last_buy_px = (
        extract_order_price(
            max((o for o, _, _ in valid_buys), key=lambda o: extract_timestamp_ms(o, default=0) or 0)
        )
        if valid_buys
        else 0.0
    )
    return {
        "buy_qty": buy_qty,
        "buy_val": buy_val,
        "sell_qty": sell_qty,
        "net_qty": buy_qty - sell_qty,
        "avg_buy": avg_buy,
        "sell_count": len(valid_sells),
        "last_buy_px": last_buy_px,
    }


def get_cached_pair_info(candidates: Any, cache_dict: dict | None = None) -> dict | None:
    """Return cached AssetPairs metadata for the first matching candidate without network I/O."""
    if not isinstance(cache_dict, dict):
        return None
    cands = tuple(c for c in (candidates or ()) if isinstance(c, str) and c)
    if not cands:
        return None

    def _cache_val(ckey: object) -> object:
        hit = cache_dict.get(ckey)
        return hit[1] if isinstance(hit, tuple) and len(hit) >= 2 else None

    p_meta = next((m for kc in cands if isinstance((m := _cache_val(f"pair:{kc}")), dict)), None)
    if p_meta is None:
        p_meta = next(
            (
                v
                for kc in cands
                if isinstance((ap_res := _cache_val(("AssetPairs", (("pair", kc),)))), dict)
                for v in ap_res.values()
                if isinstance(v, dict)
            ),
            None,
        )
    if p_meta is None and isinstance((ap_all_res := _cache_val(("AssetPairs", ()))), dict):
        p_meta = next((ap_all_res[kc] for kc in cands if isinstance(ap_all_res.get(kc), dict)), None)
    return p_meta if isinstance(p_meta, dict) else None


REQUIRED_FIELDS = ("timestamp", "open", "high", "low", "close")


def normalize_record(record: Mapping) -> dict:
    """Normalise a row into the common OHLC contract."""
    if not isinstance(record, Mapping):
        raise ValueError("record must be a mapping")
    for field in REQUIRED_FIELDS:
        if field not in record:
            raise ValueError(f"missing required field: {field}")
        if record[field] is None or isinstance(record[field], bool):
            raise ValueError(f"{field} cannot be None or boolean")
    raw_ts = record["timestamp"]
    timestamp = exact_int(raw_ts, default=None)
    if timestamp is None:
        raise ValueError(f"invalid timestamp: {raw_ts!r}")
    try:
        open_f = require_finite_float(record["open"], name="open")
        high_f = require_finite_float(record["high"], name="high")
        low_f = require_finite_float(record["low"], name="low")
        close_f = require_finite_float(record["close"], name="close")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid OHLC numeric value in record: {record!r}") from exc
    return {
        "timestamp": timestamp,
        "open": open_f,
        "high": high_f,
        "low": low_f,
        "close": close_f,
    }


def validate_dataset(records: Iterable[Mapping], *, interval_minutes: int | None = None) -> list[dict]:
    """Validate ordering, cadence and OHLC invariants; return the normalised form."""
    rows = [normalize_record(record) for record in records]
    if not rows:
        raise ValueError("OHLC dataset cannot be empty")
    interval_min = (
        require_int(
            interval_minutes,
            name="interval_minutes",
            min_value=1,
            error_msg="interval_minutes must be positive",
        )
        if interval_minutes is not None
        else None
    )

    expected_delta = interval_min * 60 if interval_min is not None else None
    previous_timestamp = None
    for index, row in enumerate(rows, start=2):
        timestamp = row["timestamp"]
        if previous_timestamp is not None:
            delta = timestamp - previous_timestamp
            if delta <= 0:
                raise ValueError(f"unordered or duplicate timestamp at line {index}: {timestamp}")
            if expected_delta is not None and delta != expected_delta:
                raise ValueError(
                    f"invalid cadence at line {index}: {delta}s, expected {expected_delta}s"
                )
        previous_timestamp = timestamp

        open_, high, low, close = row["open"], row["high"], row["low"], row["close"]
        if not all(safe_pos_float(v, default=None) is not None for v in (open_, high, low, close)):
            raise ValueError(f"non-positive price at line {index}")
        if high < max(open_, close, low) or low > min(open_, close, high):
            raise ValueError(f"invalid OHLC invariant at line {index}")
    return rows


def drop_incomplete_last_bar(
    records: Iterable[Mapping],
    *,
    interval_minutes: int,
    now_timestamp: int,
    regular_session_start: int | None = None,
    regular_session_end: int | None = None,
) -> list[dict]:
    """Drop only the tail of Yahoo candles that are still forming.

    For intraday the timestamp is the start of the bar. For daily bars Yahoo
    uses the session start, so during the session we compare against the
    ``regular`` end provided in the response metadata.
    """
    interval_i = require_int(
        interval_minutes,
        name="interval_minutes",
        min_value=1,
        error_msg="interval_minutes must be positive",
    )
    now_i = require_int(
        now_timestamp,
        name="now_timestamp",
        error_msg="now_timestamp must be an integer",
    )
    if isinstance(regular_session_start, bool) or isinstance(regular_session_end, bool):
        raise ValueError("regular session bounds cannot be boolean")
    sess_start_i = (
        require_int(
            regular_session_start,
            name="regular_session_start",
            error_msg="regular session bounds must be integers",
        )
        if regular_session_start is not None
        else None
    )
    sess_end_i = (
        require_int(
            regular_session_end,
            name="regular_session_end",
            error_msg="regular session bounds must be integers",
        )
        if regular_session_end is not None
        else None
    )
    rows = [normalize_record(record) for record in records]
    if not rows:
        return rows
    if interval_i >= 1440:
        while rows:
            last_timestamp = rows[-1]["timestamp"]
            in_open_session = (
                sess_start_i is not None
                and sess_end_i is not None
                and sess_start_i <= last_timestamp < sess_end_i
                and now_i < sess_end_i
            )
            if last_timestamp > now_i or in_open_session:
                rows.pop()
            else:
                break
        return rows
    interval_seconds = interval_i * 60
    while rows and rows[-1]["timestamp"] + interval_seconds > now_i:
        rows.pop()
    return rows


def validate_ohlc_bars(
    ohlc: Any,
    *,
    exact_len_4: bool = False,
    check_envelope: bool = True,
) -> None:
    """Validate a non-empty sequence of OHLC bars."""
    if not isinstance(ohlc, (list, tuple)) or not ohlc:
        raise ValueError("ohlc cannot be empty")
    for idx, row in enumerate(ohlc):
        if not isinstance(row, (list, tuple)) or (len(row) != 4 if exact_len_4 else len(row) < 4):
            raise ValueError(f"invalid OHLC bar at index {idx}")
        if any(isinstance(row[i], bool) for i in range(4)):
            raise ValueError(f"OHLC bar at index {idx} contains bool")
        vals = [safe_pos_float(row[i], default=None) for i in range(4)]
        if any(v is None for v in vals):
            if any(not isinstance(row[i], (int, float)) and safe_float(row[i], default=None) is None for i in range(4)):
                raise ValueError(f"OHLC bar at index {idx} has non-numeric prices")
            raise ValueError(f"OHLC bar at index {idx} has non-finite or invalid prices")
        o_f, h_f, l_f, c_f = vals[0], vals[1], vals[2], vals[3]
        if (
            h_f < l_f
            or (check_envelope and (h_f < max(o_f, c_f) or l_f > min(o_f, c_f)))
        ):
            raise ValueError(f"OHLC bar at index {idx} has non-finite or invalid prices")


def align_previous_values(
    target_timestamps: Sequence[int],
    source_records: Iterable[Mapping],
    *,
    value_field: str = "close",
    transform: Optional[Callable[[float], float]] = None,
) -> List[float]:
    """Align as-of without lookahead: the last value with timestamp <= target."""
    by_ts: dict[int, float] = {}
    for record in source_records:
        raw_ts = record["timestamp"]
        raw_val = record[value_field]
        if isinstance(raw_ts, bool) or isinstance(raw_val, bool) or raw_ts is None or raw_val is None:
            raise ValueError("source_records cannot contain None or boolean timestamp/value")
        ts_i = exact_int(raw_ts, default=None)
        val_f = safe_float(raw_val, default=None)
        if ts_i is None or val_f is None:
            raise ValueError("invalid or non-finite timestamp or value in source_records")
        by_ts[ts_i] = val_f
    source = sorted(by_ts.items(), key=lambda item: item[0])
    if not source:
        raise ValueError("the source series for alignment cannot be empty")
    convert = transform or (lambda value: value)
    result: List[float] = []
    source_index = 0
    current: Optional[float] = None
    prev_target: Optional[int] = None
    for timestamp in target_timestamps:
        if isinstance(timestamp, bool) or timestamp is None:
            raise ValueError("target_timestamps cannot contain None or boolean")
        target = exact_int(timestamp, default=None)
        if target is None:
            raise ValueError(f"invalid or non-finite target timestamp: {timestamp!r}")
        if prev_target is not None and target < prev_target:
            raise ValueError(f"target_timestamps must be sorted in non-decreasing order at {target}")
        prev_target = target
        while source_index < len(source) and source[source_index][0] <= target:
            current = convert(source[source_index][1])
            source_index += 1
        if current is None:
            raise ValueError(
                f"no known FX value at or before timestamp {target}; "
                "extend the FX dataset further back"
            )
        cur_f = safe_pos_float(current, default=None)
        if cur_f is None:
            raise ValueError(f"non-positive aligned value at timestamp {target}")
        result.append(cur_f)
    return result


def candles_to_historical_points(candles: Any) -> list[tuple[float, float]]:
    """Extract ``(epoch_sec, close_price)`` tuples from OHLC candle dictionaries."""
    if not isinstance(candles, Iterable) or isinstance(candles, (str, bytes, Mapping)):
        return []
    out: list[tuple[float, float]] = []
    for c in candles:
        if not isinstance(c, Mapping):
            continue
        tf = normalize_epoch_sec(c.get("timestamp"), default=None)
        pf = safe_pos_float(c.get("close"), default=None)
        if tf is not None and pf is not None:
            out.append((tf, pf))
    return out


def extract_yahoo_chart_result_and_quote(payload: Any) -> tuple[Optional[dict], dict, list, dict]:
    """Unpack a Yahoo v8 chart JSON payload into ``(result, chart_dict, timestamps, quote_dict)``."""
    chart = payload.get("chart") if isinstance(payload, Mapping) else None
    chart_d = dict(chart) if isinstance(chart, Mapping) else {}
    raw_results = chart_d.get("result")
    result = (
        raw_results[0]
        if isinstance(raw_results, Sequence) and not isinstance(raw_results, (str, bytes)) and raw_results
        else None
    )
    if not isinstance(result, dict) or not result:
        return None, chart_d, [], {}
    raw_ts = result.get("timestamp")
    timestamps = raw_ts if isinstance(raw_ts, list) else []
    indicators = result.get("indicators") if isinstance(result.get("indicators"), Mapping) else {}
    quote_list = indicators.get("quote") if isinstance(indicators.get("quote"), list) else []
    quote = (quote_list[0] if quote_list and isinstance(quote_list[0], dict) else {}) or {}
    return result, chart_d, timestamps, quote


def extract_yahoo_chart_latest_price(payload: Any, *, decimals: Optional[int] = None) -> Optional[float]:
    """Extract the latest positive intraday close (or ``meta.regularMarketPrice`` fallback) from a Yahoo chart payload."""
    result, _, _, quote = extract_yahoo_chart_result_and_quote(payload)
    if result is None:
        return None
    raw_closes = quote.get("close")
    if isinstance(raw_closes, Sequence) and not isinstance(raw_closes, (str, bytes)):
        for c in reversed(raw_closes):
            if (cf := safe_pos_float(c, default=None)) is not None:
                return round(cf, decimals) if decimals is not None else cf
    meta = result.get("meta")
    val = safe_pos_float(meta.get("regularMarketPrice") if isinstance(meta, Mapping) else None, default=None)
    if val is None:
        return None
    return round(val, decimals) if decimals is not None else val


def filter_and_normalize_open_orders(
    open_orders: Any,
    side: str,
    *,
    allowed_owners: Any = None,
    rc_mod: Any = None,
    owner_map_fn: Optional[Callable[[], Any]] = None,
    fallback_extract_oid: bool = False,
) -> dict:
    """Filter venue open orders by ``side`` and optional ``allowed_owners`` into a normalized dict keyed by scalar order ID."""
    want_side = normalize_side(side)
    if want_side not in ("BUY", "SELL"):
        return {}
    filtered_orders: dict = {}
    accepted_owner_map = None
    for order in (open_orders or ()):
        if not isinstance(order, dict) or extract_order_side(order) != want_side:
            continue
        cid = clean_order_id(order.get("clientOrderId")) or ""
        if (
            allowed_owners is not None
            and rc_mod is not None
            and not rc_mod.order_matches_owner(cid, allowed_owners)
        ):
            if not cid.startswith("OR_"):
                continue
            if accepted_owner_map is None:
                try:
                    accepted_owner_map = owner_map_fn() if callable(owner_map_fn) else {}
                except Exception:
                    accepted_owner_map = {}
            oid_key = (
                str(extract_order_id(order) or "")
                if fallback_extract_oid
                else str(order.get("orderId") or "")
            )
            inferred = accepted_owner_map.get(oid_key) or accepted_owner_map.get(cid)
            allowed_set = rc_mod._normalize_allowed_owners(allowed_owners)
            if not inferred or str(inferred).strip().lower() not in allowed_set:
                continue
        original_qty = safe_nonneg_float(extract_orig_qty(order, default=0.0), default=0.0) or 0.0
        raw_exec_qty = next(
            (order.get(k) for k in ("executedQty", "z", "vol_exec") if order.get(k) is not None),
            None,
        )
        executed_qty = safe_nonneg_float(raw_exec_qty, default=0.0)
        remaining_qty = extract_remaining_qty(order, default=max(0.0, original_qty - executed_qty))
        ts_ms = extract_timestamp_ms(order, default=0) or 0
        ts_sec = safe_nonneg_ts(ts_ms, default=0.0)
        raw_time = safe_nonneg_float(order.get("time"), default=None)
        if raw_time is not None:
            ts_sec = (
                raw_time / 1000.0
                if (raw_time < 1_000_000_000.0 or raw_time >= 100_000_000_000.0)
                else raw_time
            )
        raw_oid = normalize_scalar_order_id(order.get("orderId"))
        if raw_oid is None and fallback_extract_oid:
            raw_oid = normalize_scalar_order_id(extract_order_id(order))
        if raw_oid is None:
            continue
        filtered_orders[raw_oid] = {
            "price": extract_order_price(order),
            "quantity": original_qty,
            "origQty": original_qty,
            "qty": remaining_qty,
            "executedQty": executed_qty,
            "remainingQty": remaining_qty,
            "timestamp": ts_sec,
            "time": ts_ms,
            "clientOrderId": cid,
        }
    return filtered_orders


def iter_expired_open_order_ids(
    open_orders: Any,
    expire_sec: float,
    *,
    current_time_sec: float,
) -> list[Any]:
    """Return order IDs from ``open_orders`` mapping whose timestamp is older than ``expire_sec`` seconds."""
    exp_s = safe_pos_float(expire_sec, default=None)
    now_s = normalize_epoch_sec(current_time_sec, default=None)
    if not isinstance(open_orders, Mapping) or exp_s is None or now_s is None:
        return []
    expired_ids: list[Any] = []
    for order_id, order_details in open_orders.items():
        order_time = (extract_timestamp_ms(order_details, default=0) or 0) / 1000.0
        if order_time > 0 and (now_s - order_time) > exp_s:
            expired_ids.append(order_id)
    return expired_ids



