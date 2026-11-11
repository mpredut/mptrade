"""Pure aggregations for calibrating the execution model from the real audit."""

from __future__ import annotations

from functools import partial
import math
import re
import statistics

from botcore import (
    _parse_bool_flag,
    clean_finite_floats,
    clean_order_id_str,
    normalize_epoch_sec,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
)
from cachepaths import normalize_venue_name
from providers.base import normalize_side

_opt_ts_sec = partial(normalize_epoch_sec, default=None, allow_zero=True)


def _ts_sort_key(val: object) -> tuple[bool, float]:
    ts = _opt_ts_sec(val)
    return (ts is None, ts if ts is not None else 0.0)


def _has_cid(order: dict) -> bool:
    cid = order.get("client_order_id")
    return cid is not None and (isinstance(cid, bool) or bool(str(cid).strip()))


_CLIENT_ORDER_ID_PATTERNS = {
    "kraken": re.compile(r"[0-9a-f]{32}"),
    "binance": re.compile(r"SD_[0-9a-f]{32}"),
    "hyperliquid": re.compile(r"0x[0-9a-f]{32}"),
}


def _client_order_id_valid(venue: object, client_order_id: object) -> bool | None:
    """Validate only supported venues; T212/unknown do not enter the gate."""
    raw_v = "" if isinstance(venue, bool) else str(venue or "")
    norm_venue = normalize_venue_name(raw_v) or raw_v.strip().lower()
    pattern = _CLIENT_ORDER_ID_PATTERNS.get(norm_venue)
    if pattern is None:
        return None
    if isinstance(client_order_id, bool):
        return False
    return bool(pattern.fullmatch(str(client_order_id or "")))


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    pct = safe_float(percentile, default=None)
    if pct is None:
        return None
    pct = min(100.0, max(0.0, pct))
    clean = clean_finite_floats(values)
    if not clean:
        return None
    ordered = sorted(clean)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * pct / 100.0
    lower = max(0, min(math.floor(position), len(ordered) - 1))
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def distribution(values: list[float]) -> dict:
    clean_values = clean_finite_floats(values)
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
    keyed_events: list[tuple[tuple[bool, float, int], str, dict]] = []
    last_ts_by_intent: dict[str, float] = {}
    for idx, event in enumerate(safe_events):
        if not isinstance(event, dict):
            continue
        raw_intent_id = event.get("intent_id")
        intent_id = "" if isinstance(raw_intent_id, bool) or raw_intent_id is None else str(raw_intent_id).strip()
        if not intent_id:
            continue
        ev_ts = _opt_ts_sec(event.get("ts"))
        if ev_ts is not None:
            last_ts_by_intent[intent_id] = ev_ts
            sort_key = (False, ev_ts, idx)
        elif intent_id in last_ts_by_intent:
            sort_key = (False, last_ts_by_intent[intent_id], idx)
        else:
            sort_key = (True, 0.0, idx)
        keyed_events.append((sort_key, intent_id, event))
    for _, intent_id, event in sorted(keyed_events, key=lambda item: item[0]):
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
        qty = safe_float(requested.get("qty"), 0.0)
        status_rows = []
        by_order_id: dict[str, list[tuple[dict, float, float, float | None]]] = {}
        active_oid = clean_order_id_str(accepted.get("order_id")) if accepted is not None else ""
        for event in history:
            if event.get("event") == "submit_accepted":
                cleaned_oid = clean_order_id_str(event.get("order_id"))
                if cleaned_oid:
                    active_oid = cleaned_oid
                    if "" in by_order_id and cleaned_oid != "":
                        orphan_rows = by_order_id.pop("")
                        by_order_id.setdefault(cleaned_oid, []).extend(orphan_rows)
                        status_rows = [
                            (cleaned_oid if ok == "" else ok, rt)
                            for ok, rt in status_rows
                        ]
            elif event.get("event") in STATUS_EVENTS:
                filled_step = safe_nonneg_float(event.get("filled_qty"), 0.0)
                cost_step = safe_pos_float(event.get("cost"), default=None)
                raw_fee_val = safe_float(event.get("fee"), default=None)
                fee_step = abs(raw_fee_val) if raw_fee_val is not None else None
                cleaned_s_oid = clean_order_id_str(event.get("order_id"))
                oid_key = cleaned_s_oid if cleaned_s_oid else active_oid
                row_t = (event, filled_step, cost_step, fee_step)
                status_rows.append((oid_key, row_t))
                by_order_id.setdefault(oid_key, []).append(row_t)

        first_fill = next((row for _, row in status_rows if row[1] > 0), None)
        filled = 0.0
        cost = 0.0
        fee = 0.0
        all_costs_known = True
        fee_any_known = False
        all_fees_known = True
        for bucket in by_order_id.values():
            if not bucket:
                continue
            max_filled = max(row[1] for row in bucket)
            if max_filled <= 0.0:
                continue
            at_max = [row for row in bucket if math.isclose(row[1], max_filled, rel_tol=1e-12, abs_tol=1e-12)]
            known_costs = [row[2] for row in at_max if row[2] is not None and row[2] > 0.0]
            if known_costs:
                best_cost = max(known_costs)
            else:
                all_costs_known = False
                best_cost = 0.0
            known_fees = [row[3] for row in at_max if row[3] is not None]
            if known_fees:
                fee_any_known = True
                best_fee = max(known_fees)
            else:
                all_fees_known = False
                best_fee = 0.0
            filled += max_filled
            cost += best_cost
            fee += best_fee

        fill_ratios = []
        if qty > 0:
            running_by_oid: dict[str, float] = {}
            for oid_key, row in status_rows:
                running_by_oid[oid_key] = max(running_by_oid.get(oid_key, 0.0), row[1])
                cum_filled = sum(running_by_oid.values())
                if cum_filled > 0:
                    fill_ratios.append(min(1.0, max(0.0, cum_filled / qty)))
        avg_fill = cost / filled if (filled > 0 and cost > 0 and all_costs_known) else None
        side_norm = normalize_side(requested.get("side"))
        direction = 1.0 if side_norm == "BUY" else (-1.0 if side_norm == "SELL" else 0.0)
        requested_price = requested.get("price")
        deviation_bps = None
        if requested_price is not None and avg_fill is not None and direction != 0.0:
            reference = safe_float(requested_price, 0.0)
            if reference > 0:
                deviation_bps = direction * (avg_fill / reference - 1.0) * 10_000
        is_market = _parse_bool_flag(requested.get("market"), False)
        market_shortfall_bps = None
        reference_price = requested.get("reference_price")
        if (
            is_market
            and reference_price is not None
            and avg_fill is not None
            and direction != 0.0
        ):
            reference = safe_float(reference_price, 0.0)
            if reference > 0:
                market_shortfall_bps = (
                    direction * (avg_fill / reference - 1.0) * 10_000
                )
        latency = None
        if accepted is not None and first_fill is not None:
            acc_ts = _opt_ts_sec(accepted.get("ts"))
            fill_ts = _opt_ts_sec(first_fill[0].get("ts"))
            if acc_ts is not None and fill_ts is not None:
                latency = max(0.0, fill_ts - acc_ts)
        client_order_id = None
        if accepted is not None:
            raw_acc_cid = accepted.get("client_order_id")
            if raw_acc_cid is not None and (isinstance(raw_acc_cid, bool) or str(raw_acc_cid).strip()):
                client_order_id = raw_acc_cid
        if client_order_id is None:
            raw_req_cid = requested.get("client_order_id")
            if raw_req_cid is not None and (isinstance(raw_req_cid, bool) or str(raw_req_cid).strip()):
                client_order_id = raw_req_cid
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
            "rejected": accepted is None and any(
                event.get("event") in ("submit_rejected", "submit_refused")
                for event in history
            ),
            "filled_qty": filled,
            "requested_qty": qty,
            "final_fill_ratio": min(1.0, filled / qty) if qty > 0 else None,
            "ever_partial": any(0.0 < ratio < 1.0 - 1e-9 for ratio in fill_ratios),
            "fee_bps": (
                abs(fee) / cost * 10_000
                if (cost > 0 and all_costs_known and fee_any_known and all_fees_known)
                else None
            ),
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
                if not order["market"] and order["limit_fill_deviation_bps"] is not None
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
    seen_venues_norm: set[str] = set()
    for order in sorted(
        supported_client_ids,
        key=lambda item: _ts_sort_key(item.get("accepted_ts")),
    ):
        raw_venue = order.get("venue")
        venue = "" if isinstance(raw_venue, bool) else str(raw_venue or "").strip()
        venue_norm = normalize_venue_name(venue) or venue.lower()
        if order["client_order_id_valid"] and venue and venue_norm not in seen_venues_norm:
            seen_venues_norm.add(venue_norm)
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
                _has_cid(order) for order in supported_client_ids
            ),
            "valid_client_order_ids": sum(
                order["client_order_id_valid"] is True
                for order in supported_client_ids
            ),
            "invalid_client_order_ids": sum(
                _has_cid(order)
                and order["client_order_id_valid"] is False
                for order in supported_client_ids
            ),
            "missing_client_order_ids": sum(
                not _has_cid(order) for order in supported_client_ids
            ),
            "first_valid_by_venue": first_valid_by_venue,
        },
    }
