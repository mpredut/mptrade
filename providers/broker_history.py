"""Stateless equity broker order-history matching, fill aggregation, and normalization utilities."""

import math
from typing import Any, Mapping, Optional

from botcore import (
    clean_nonempty_str,
    clean_order_id,
    extract_order_id,
    extract_order_side,
    extract_timestamp_ms,
    first_pos_float,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
)


def extract_broker_order_ticker(raw: Any) -> str:
    """Extract ticker string from a flat or nested ``{'instrument': {'ticker': ...}}`` broker order."""
    if not isinstance(raw, Mapping):
        return ""
    inst = raw.get("instrument")
    inst_ticker = inst.get("ticker") if isinstance(inst, Mapping) else None
    return clean_nonempty_str(raw.get("ticker")) or clean_nonempty_str(inst_ticker) or ""


def extract_broker_signed_qty_and_side(order: Any) -> tuple[float, str]:
    """Extract signed order quantity and canonical side (``'BUY'``/``'SELL'``) from a broker order dict."""
    if not isinstance(order, Mapping):
        return 0.0, "BUY"
    raw_qty = 0.0
    for key in ("orderedQuantity", "quantity", "filledQuantity"):
        val = safe_float(order.get(key), default=None)
        if val is not None and val != 0.0:
            raw_qty = val
            break
    order_dict = order if isinstance(order, dict) else dict(order)
    side = extract_order_side(order_dict) or ("SELL" if raw_qty < 0.0 else "BUY")
    return raw_qty, side


def iter_broker_history_order_fills(items: Any):
    """Yield ``(order_id_str, order_dict, fill_obj)`` from broker order-history items."""
    if not isinstance(items, (list, tuple)):
        return
    for item in items:
        if not isinstance(item, dict):
            continue
        order = item.get("order") if isinstance(item.get("order"), dict) else item
        if isinstance(order, dict) and (oid := extract_order_id(order)):
            yield str(oid), order, item.get("fill")


def extract_broker_history_matches(items: Any, target_oid: str) -> list[tuple[dict, Optional[dict]]]:
    """Return ``[(order, fill), ...]`` entries matching ``target_oid`` from broker order-history items."""
    clean_target = clean_order_id(target_oid)
    if not clean_target:
        return []
    return [
        (order, fill)
        for oid, order, fill in iter_broker_history_order_fills(items)
        if oid == clean_target
    ]


def aggregate_broker_history_fills(matches: list[tuple[dict, Optional[dict]]]) -> dict:
    """Aggregate one or more ``(order, fill)`` matches into a single enriched broker order dictionary."""
    if (
        not isinstance(matches, (list, tuple))
        or not matches
        or not isinstance(matches[0], (list, tuple))
        or not matches[0]
        or not isinstance(matches[0][0], Mapping)
    ):
        return {}
    result = dict(matches[0][0])
    total_qty = total_cost = total_fee = priced_qty = 0.0
    fee_currencies = set()
    latest_fill_ts_ms: Optional[int] = None
    seen_fill_ids: set[str] = set()
    for entry in matches:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        order, fill = entry[0], entry[1]
        cand_ts = extract_timestamp_ms(fill, default=None) or extract_timestamp_ms(order, default=None)
        if cand_ts is not None and (latest_fill_ts_ms is None or cand_ts > latest_fill_ts_ms):
            latest_fill_ts_ms = cand_ts
        if not isinstance(fill, dict):
            continue
        fill_id = extract_order_id(fill)
        if fill_id is not None:
            if fill_id in seen_fill_ids:
                continue
            seen_fill_ids.add(fill_id)
        qf = safe_float(fill.get("quantity"), default=None)
        pf = safe_nonneg_float(fill.get("price"), default=None)
        if qf is None or pf is None:
            continue
        qty = abs(qf)
        price = pf
        total_qty += qty
        if price > 0.0 and qty > 0.0:
            total_cost += qty * price
            priced_qty += qty
        wallet = fill.get("walletImpact")
        taxes = wallet.get("taxes") if isinstance(wallet, dict) else []
        for tax in taxes or []:
            if not isinstance(tax, dict):
                continue
            tqf = safe_float(tax.get("quantity"), default=None)
            if tqf is None:
                continue
            tax_amt = abs(tqf)
            if tax_amt <= 0.0:
                continue
            total_fee += tax_amt
            raw_cur = tax.get("currency")
            currency = raw_cur.strip().upper() if isinstance(raw_cur, str) else ""
            if currency:
                fee_currencies.add(currency)
    if total_qty > 0:
        result["filledQuantity"] = total_qty
        result["filledValue"] = total_cost
        if priced_qty > 0:
            result["fillPrice"] = total_cost / priced_qty
    result["fee"] = total_fee
    if fee_currencies:
        result["_feeCurrencies"] = sorted(fee_currencies)
    if latest_fill_ts_ms is not None:
        result["_timestamp_ms"] = latest_fill_ts_ms
    return result


def normalize_broker_history_items(items: Any) -> dict[str, list[dict]]:
    """Group FILLED broker order-history items by ticker and orderId into canonical fill records."""
    grouped: dict[str, list[tuple[dict, Optional[dict]]]] = {}
    for oid, order, fill in iter_broker_history_order_fills(items):
        if str(order.get("status") or "").strip().upper() == "FILLED":
            grouped.setdefault(oid, []).append((order, fill if isinstance(fill, dict) else None))
    by_symbol: dict[str, list[dict]] = {}
    for oid, matches in grouped.items():
        enriched = aggregate_broker_history_fills(matches)
        ticker = extract_broker_order_ticker(enriched).upper()
        if not ticker:
            continue
        raw_qty, side = extract_broker_signed_qty_and_side(enriched)
        qty = first_pos_float(enriched.get("filledQuantity"), abs(raw_qty))
        px = first_pos_float(enriched.get("fillPrice"), enriched.get("limitPrice"))
        ts_ms = enriched.get("_timestamp_ms")
        if qty is None or px is None or ts_ms is None:
            continue
        cost = safe_pos_float(enriched.get("filledValue"), default=px * qty) or (px * qty)
        fee = safe_nonneg_float(enriched.get("fee"), default=0.0) or 0.0
        by_symbol.setdefault(ticker, []).append({
            "symbol": ticker,
            "orderId": str(oid),
            "side": side,
            "type": str(clean_nonempty_str(enriched.get("type")) or "MARKET").upper(),
            "price": round(px, 6),
            "qty": round(qty, 6),
            "executedQty": round(qty, 6),
            "cummulativeQuoteQty": round(cost, 4),
            "fee": round(fee, 6),
            "timestamp": math.trunc(ts_ms),
            "time": math.trunc(ts_ms),
            "status": "FILLED",
        })
    for sym_list in by_symbol.values():
        sym_list.sort(key=lambda r: (r.get("timestamp") or 0, str(r.get("orderId") or "")))
    return by_symbol


__all__ = [
    "aggregate_broker_history_fills",
    "extract_broker_history_matches",
    "extract_broker_order_ticker",
    "extract_broker_signed_qty_and_side",
    "iter_broker_history_order_fills",
    "normalize_broker_history_items",
]
