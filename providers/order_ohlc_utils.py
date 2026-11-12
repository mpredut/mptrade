"""Stateless order, fill, fee, post-only, and open-order normalization utilities."""

import re
from typing import Any, Callable, List, Mapping, Optional

from botcore import (
    ClosedPriceSeries,  # noqa: F401
    _first_mapped,
    _nonneg_abs_float,
    _parse_bool_flag,
    _parse_epoch_or_iso_s,  # noqa: F401
    _pos_abs_float,
    _ts_ms_to_s,  # noqa: F401
    clean_nonempty_str,
    clean_order_id,
    exact_pos_int,
    extract_order_id,
    extract_order_price,
    extract_order_qty,
    extract_order_side,
    extract_timestamp_ms,
    extract_timestamp_s,  # noqa: F401
    normalize_epoch_sec,
    normalize_scalar_order_id,
    normalize_side,
    normalize_timestamp_ms,
    obj_attr,
    resolve_now_ts,
    safe_float,
    safe_nonneg_float,
    safe_nonneg_ts,
    safe_pos_float,
)
from cachepaths import canonical_base_asset, normalize_venue_name, same_canonical_base_asset  # noqa: F401
from .broker_history import (  # noqa: F401
    aggregate_broker_history_fills,
    extract_broker_history_matches,
    extract_broker_order_ticker,
    iter_broker_history_order_fills,
    normalize_broker_history_items,
)
from .ohlc_utils import (  # noqa: F401
    OHLC_CASE_INSENSITIVE_INTERVAL_ALIASES,
    OHLC_INTERVALS_MIN_TO_STR,
    OHLC_INTERVALS_STR_TO_MIN,
    ProviderError,
    REQUIRED_FIELDS,
    _CANDLE_INTERVALS,
    align_previous_values,
    build_price_history_points,
    candle_interval,
    candles_to_historical_points,
    drop_incomplete_last_bar,
    extract_trading_quote_pairs,
    extract_yahoo_chart_latest_price,
    extract_yahoo_chart_result_and_quote,
    normalize_record,
    normalize_yahoo_5d_chart_bars,
    parse_ohlc_interval,
    parse_ohlc_interval_minutes,
    require_pos_ohlc_interval,
    resolve_ohlc_time_window_ms,
    series_to_historical_points,
    validate_dataset,
    validate_ohlc_bars,
    validate_ohlc_candle_bar,
)
from .order_id_context import (  # noqa: F401
    MANUAL_CID_PREFIXES,
    VENUE_PROVIDER_SPECS,
    all_venue_native_client_order_ids,
    derive_venue_128bit_client_order_id,
    extract_client_order_id,
    format_actor_tag,
    format_venue_intent_client_order_id,
    resolve_order_actor,
    resolve_order_origin,
    resolve_venue_provider_class,
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


def resolve_market_flag(force: Any = False, market: Any = False, **kwargs) -> bool:
    """Return True when any market/force flag in args or kwargs is truthy."""
    return any(_parse_bool_flag(v, False) for v in (force, market, kwargs.get("market"), kwargs.get("force")))


POST_ONLY_CROSS_TOKENS: tuple[str, ...] = (
    "would immediately match",
    "would have immediately matched",
    "post only order",
    "post-only order",
    "eorder:post only",
    "alo order",
    "badalopx",
)


def is_post_only_cross_refusal(error: object) -> bool:
    """Return True when a venue error indicates a post-only order would cross the book."""
    text = str(error or "").lower()
    return any(token in text for token in POST_ONLY_CROSS_TOKENS)


def compute_post_only_maker_price(
    side: object,
    price: object,
    *,
    current_price: object = None,
    fetch_current_price_fn: Optional[Callable[[], object]] = None,
    tick_size: object = 0.0,
    tick_size_fn: Optional[Callable[[float], object]] = None,
    round_price_fn: Optional[Callable[[float], object]] = None,
    step_bps_ratio: float = 0.0005,
) -> float:
    """Step ``price`` onto the passive Maker side of the spread after a post-only cross rejection."""
    px_f = safe_pos_float(price, default=None)
    s_norm = normalize_side(side)
    if px_f is None or s_norm not in ("BUY", "SELL"):
        return safe_pos_float(price, default=0.0) or 0.0
    cur_px = safe_pos_float(current_price, default=None)
    if cur_px is None and callable(fetch_current_price_fn):
        try:
            cur_px = safe_pos_float(fetch_current_price_fn(), default=None)
        except Exception:
            cur_px = None
    is_buy = s_norm == "BUY"
    anchor = (min(px_f, cur_px) if is_buy else max(px_f, cur_px)) if cur_px else px_f
    raw_tick = tick_size
    if callable(tick_size_fn):
        try:
            raw_tick = tick_size_fn(anchor)
        except Exception:
            raw_tick = tick_size
    tick = safe_pos_float(raw_tick, default=0.0) or 0.0
    ratio = safe_pos_float(step_bps_ratio, default=0.0005) or 0.0005
    cushion = max(tick, anchor * ratio)
    raw_stepped = (anchor - cushion) if is_buy else (anchor + cushion)
    if raw_stepped <= 0.0:
        return px_f

    def _round_cand(val: float) -> float:
        if round_price_fn is None:
            return val
        r_raw = round_price_fn(val)
        if isinstance(r_raw, (int, float)) and not isinstance(r_raw, bool) and r_raw <= 0.0:
            return 0.0
        return safe_pos_float(r_raw, default=val) or val

    rounded = _round_cand(raw_stepped)
    if rounded <= 0.0:
        return px_f
    if (rounded >= anchor) if is_buy else (rounded <= anchor):
        delta = tick if tick > 0.0 else anchor * ratio
        cand = (anchor - delta) if is_buy else (anchor + delta)
        if cand <= 0.0:
            return px_f
        rounded = _round_cand(cand)
        if rounded <= 0.0:
            return px_f
        if (rounded >= anchor) if is_buy else (rounded <= anchor):
            rounded = cand
    return rounded if rounded > 0.0 else px_f


def parse_nonneg_order_field(raw: Any, *, allow_empty_str: bool = False) -> Optional[float]:
    """Parse an optional venue order quantity/cost/fee field into a non-negative float (``None`` -> ``0.0``)."""
    if raw is None or (_parse_bool_flag(allow_empty_str, False) and isinstance(raw, str) and not raw.strip()):
        return 0.0
    return safe_nonneg_float(raw, default=None)


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
        fee_quote_canon = _QUOTE_FEE_ALIASES.get(fee_asset, fee_asset)
        quote_canon = _QUOTE_FEE_ALIASES.get(quote_u, quote_u)
        if base_u and (fee_asset == base_u or (fee_asset and same_canonical_base_asset(fee_asset, base_u))):
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
    cand_type = clean_nonempty_str(order.get("type"))
    raw_otype = (
        clean_nonempty_str(order.get("order_type"))
        or clean_nonempty_str(order.get("orderType"))
        or clean_nonempty_str(descr.get("ordertype"))
        or (cand_type if (cand_type and cand_type.upper() not in ("BUY", "SELL", "B", "S")) else None)
        or "LIMIT"
    )
    actor, mod, rule = resolve_order_origin(order, known_ai_order_ids=known_ai_order_ids, audit_origin_map=audit_origin_map)
    sym_out = clean_nonempty_str(symbol) or (clean_nonempty_str(order.get("symbol")) or "").upper()
    rec = {
        "venue": clean_nonempty_str(venue) or "",
        "symbol": sym_out,
        "side": extract_order_side(order) or normalize_side(order.get("side")),
        "order_type": raw_otype.upper(),
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
        cid = extract_client_order_id(order, default="") or ""
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
        executed_qty = _first_mapped(order, _EXEC_QTY_KEYS, _nonneg_abs_float, 0.0)
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
