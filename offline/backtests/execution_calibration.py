"""Pure aggregations for calibrating the execution model from the real audit."""

from __future__ import annotations

import math
import re
import statistics


def _f(val: object, default: float = 0.0) -> float:
    if val is None or isinstance(val, bool):
        return default
    try:
        res = float(val)
    except (TypeError, ValueError):
        return default
    return res if math.isfinite(res) else default


def _ts_sec(val: object, default: float = 0.0) -> float:
    res = _f(val, default)
    return res / 1000.0 if res > 1e11 else res


def _as_bool(val: object, default: bool = False) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return bool(val) if math.isfinite(float(val)) else default
    s = str(val).strip().lower()
    if s in ("1", "true", "yes", "on", "y", "t"):
        return True
    if s in ("0", "false", "no", "off", "n", "f", "none", "null", ""):
        return False
    return default


_CLIENT_ORDER_ID_PATTERNS = {
    "kraken": re.compile(r"[0-9a-f]{32}"),
    "binance": re.compile(r"SD_[0-9a-f]{32}"),
    "hyperliquid": re.compile(r"0x[0-9a-f]{32}"),
}


def _client_order_id_valid(venue: object, client_order_id: object) -> bool | None:
    """Validate only supported venues; T212/unknown do not enter the gate."""
    pattern = _CLIENT_ORDER_ID_PATTERNS.get(str(venue or "").strip().lower())
    if pattern is None:
        return None
    return bool(pattern.fullmatch(str(client_order_id or "")))


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values or isinstance(percentile, bool):
        return None
    try:
        pct = float(percentile)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(pct):
        return None
    pct = min(100.0, max(0.0, pct))
    clean: list[float] = []
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        try:
            vf = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(vf):
            clean.append(vf)
    if not clean:
        return None
    ordered = sorted(clean)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * pct / 100.0
    lower = max(0, min(int(position), len(ordered) - 1))
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def distribution(values: list[float]) -> dict:
    clean_values: list[float] = []
    for value in (values or ()):
        if value is None or isinstance(value, bool):
            continue
        try:
            vf = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(vf):
            clean_values.append(vf)
    return {
        "count": len(clean_values),
        "min": min(clean_values) if clean_values else None,
        "mean": statistics.fmean(clean_values) if clean_values else None,
        "p50": _percentile(clean_values, 50),
        "p90": _percentile(clean_values, 90),
        "p95": _percentile(clean_values, 95),
        "max": max(clean_values) if clean_values else None,
    }


def calibrate_execution_events(events: list[dict]) -> dict:
    """Correlate the events by ``intent_id`` without assuming missing fills."""
    intents: dict[str, list[dict]] = {}
    safe_events = events if isinstance(events, (list, tuple)) else []
    for event in sorted(safe_events, key=lambda item: _ts_sec(item.get("ts") if isinstance(item, dict) else 0.0, 0.0)):
        if not isinstance(event, dict):
            continue
        raw_intent_id = event.get("intent_id")
        intent_id = "" if isinstance(raw_intent_id, bool) or raw_intent_id is None else str(raw_intent_id).strip()
        if intent_id:
            intents.setdefault(intent_id, []).append(event)

    orders = []
    for intent_id, history in intents.items():
        requested = next(
            (event for event in history if event.get("event") == "submit_requested"),
            None,
        )
        if requested is None:
            continue
        accepted = next(
            (event for event in history if event.get("event") == "submit_accepted"),
            None,
        )
        STATUS_EVENTS = ("order_status", "order_terminal", "order_terminal_retry")
        statuses = [
            event for event in history if event.get("event") in STATUS_EVENTS
        ]
        qty = _f(requested.get("qty"), 0.0)
        status_rows = []
        for status in statuses:
            filled = _f(status.get("filled_qty"), 0.0)
            cost = _f(status.get("cost"), 0.0)
            fee = _f(status.get("fee"), 0.0)
            status_rows.append((status, filled, cost, fee))
        latest = max(
            status_rows,
            key=lambda row: (row[1], _ts_sec(row[0].get("ts"), 0.0)),
            default=None,
        )
        first_fill = next((row for row in status_rows if row[1] > 0), None)
        filled = latest[1] if latest else 0.0
        cost = latest[2] if latest else 0.0
        fee = latest[3] if latest else 0.0
        fill_ratios = [
            min(1.0, max(0.0, row[1] / qty))
            for row in status_rows if qty > 0 and row[1] > 0
        ]
        avg_fill = cost / filled if filled > 0 and cost > 0 else None
        raw_side = requested.get("side")
        side = "" if isinstance(raw_side, bool) or raw_side is None else str(raw_side).strip().lower()
        direction = 1.0 if side in ("buy", "long") else (-1.0 if side in ("sell", "short") else 0.0)
        requested_price = requested.get("price")
        deviation_bps = None
        if requested_price is not None and avg_fill is not None and direction != 0.0:
            reference = _f(requested_price, 0.0)
            if reference > 0:
                deviation_bps = direction * (avg_fill / reference - 1.0) * 10_000
        is_market = _as_bool(requested.get("market"), False)
        market_shortfall_bps = None
        reference_price = requested.get("reference_price")
        if (
            is_market
            and reference_price is not None
            and avg_fill is not None
            and direction != 0.0
        ):
            reference = _f(reference_price, 0.0)
            if reference > 0:
                market_shortfall_bps = (
                    direction * (avg_fill / reference - 1.0) * 10_000
                )
        latency = None
        if accepted is not None and first_fill is not None:
            latency = max(
                0.0,
                _ts_sec(first_fill[0].get("ts"), 0.0)
                - _ts_sec(accepted.get("ts"), 0.0),
            )
        client_order_id = None
        if accepted is not None:
            client_order_id = accepted.get("client_order_id")
        if not client_order_id:
            client_order_id = requested.get("client_order_id")
        client_order_id_valid = _client_order_id_valid(
            requested.get("venue"), client_order_id,
        )
        orders.append({
            "intent_id": intent_id,
            "venue": requested.get("venue"),
            "symbol": requested.get("symbol"),
            "market": is_market,
            "accepted": accepted is not None,
            "accepted_ts": accepted.get("ts") if accepted is not None else None,
            "order_id": accepted.get("order_id") if accepted is not None else None,
            "client_order_id": client_order_id,
            "client_order_id_valid": client_order_id_valid,
            "rejected": any(
                event.get("event") in ("submit_rejected", "submit_refused")
                for event in history
            ),
            "filled_qty": filled,
            "requested_qty": qty,
            "final_fill_ratio": min(1.0, filled / qty) if qty > 0 else None,
            "ever_partial": any(0.0 < ratio < 1.0 - 1e-9 for ratio in fill_ratios),
            "fee_bps": abs(fee) / cost * 10_000 if cost > 0 else None,
            "first_fill_latency_s": latency,
            "limit_fill_deviation_bps": deviation_bps,
            "market_execution_shortfall_bps": market_shortfall_bps,
        })

    def summarize(selected: list[dict]) -> dict:
        filled = [order for order in selected if order["filled_qty"] > 0]
        return {
            "orders": len(selected),
            "accepted": sum(order["accepted"] for order in selected),
            "rejected": sum(order["rejected"] for order in selected),
            "filled": len(filled),
            "ever_partial": sum(order["ever_partial"] for order in filled),
            "fee_bps": distribution([
                order["fee_bps"] for order in filled
                if order["fee_bps"] is not None
            ]),
            "first_fill_latency_s": distribution([
                order["first_fill_latency_s"] for order in filled
                if order["first_fill_latency_s"] is not None
            ]),
            "final_fill_ratio": distribution([
                order["final_fill_ratio"] for order in filled
                if order["final_fill_ratio"] is not None
            ]),
            "limit_fill_deviation_bps": distribution([
                order["limit_fill_deviation_bps"] for order in filled
                if order["limit_fill_deviation_bps"] is not None
            ]),
            "market_execution_shortfall_bps": distribution([
                order["market_execution_shortfall_bps"] for order in filled
                if order["market_execution_shortfall_bps"] is not None
            ]),
        }

    market = [order for order in orders if order["market"]]
    limit = [order for order in orders if not order["market"]]
    filled_count = sum(order["filled_qty"] > 0 for order in orders)
    supported_client_ids = [
        order for order in orders
        if order["accepted"] and order["client_order_id_valid"] is not None
    ]
    first_valid_by_venue = {}
    for order in sorted(
        supported_client_ids,
        key=lambda item: _f(item.get("accepted_ts"), 0.0),
    ):
        venue = str(order.get("venue") or "")
        if order["client_order_id_valid"] and venue not in first_valid_by_venue:
            first_valid_by_venue[venue] = {
                "intent_id": order["intent_id"],
                "symbol": order["symbol"],
                "client_order_id": order["client_order_id"],
                "order_id": order["order_id"],
                "accepted_ts": order["accepted_ts"],
            }
    return {
        "orders": orders,
        "summary": {
            "all": summarize(orders),
            "market": summarize(market),
            "limit": summarize(limit),
        },
        "calibration_readiness": {
            "minimum_filled_orders": 20,
            "filled_orders": filled_count,
            "enough_total_fills": filled_count >= 20,
            "has_market_fee_samples": any(
                order["market"] and order["fee_bps"] is not None for order in orders
            ),
            "has_limit_fee_samples": any(
                not order["market"] and order["fee_bps"] is not None
                for order in orders
            ),
            "has_market_execution_shortfall_samples": any(
                order["market_execution_shortfall_bps"] is not None
                for order in orders
            ),
            "can_calibrate_market_slippage": False,
            "market_slippage_blocker": (
                "the decision-to-fill shortfall includes market movement, spread and slippage; "
                "the audit cannot separate them without bid/ask/mid at submit time"
            ),
            "can_calibrate_spread": False,
            "spread_blocker": "the audit holds no bid/ask at decision time",
        },
        "client_order_id_validation": {
            "supported_accepted_orders": len(supported_client_ids),
            "with_client_order_id": sum(
                bool(order["client_order_id"]) for order in supported_client_ids
            ),
            "valid_client_order_ids": sum(
                order["client_order_id_valid"] is True
                for order in supported_client_ids
            ),
            "invalid_client_order_ids": sum(
                bool(order["client_order_id"])
                and order["client_order_id_valid"] is False
                for order in supported_client_ids
            ),
            "missing_client_order_ids": sum(
                not order["client_order_id"] for order in supported_client_ids
            ),
            "first_valid_by_venue": first_valid_by_venue,
        },
    }
