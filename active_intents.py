"""Read-only index of active financial intents across strategy-owned state files.

The index normalizes visibility only. It never writes state, submits orders, cancels
orders, or decides retry. Each strategy remains the authority for its campaign and
financial policy.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class StateSource:
    owner: str
    pattern: str
    json_lines: bool = False


DEFAULT_SOURCES = (
    StateSource("global_outbox", "cachedb/order_retry_queue.jsonl", True),
    StateSource("assetguardian", "cachedb/assetguardian_state.json"),
    StateSource("rtrade", "cachedb/rtrade_pairs.json"),
    StateSource("binance_trailing", "cachedb/trailing_state.json"),
    StateSource("kraken", "kraken/.state_*.json"),
    StateSource("kraken_trailing", "kraken/trailing_state.json"),
    StateSource("t212", "212trading/.state_*.json"),
    StateSource("t212_one_shot", "212trading/.t212_order.*.json"),
    StateSource("hyperliquid", "hyperliquid/.state_*.json"),
)

_TERMINAL = {"closed", "filled", "canceled", "cancelled", "expired", "rejected"}


def _is_backup(path: str) -> bool:
    name = os.path.basename(path)
    return any(token in name for token in (".pre_", ".bak", ".backup", ".tmp"))


def _first(row: dict, *names):
    for name in names:
        value = row.get(name)
        if value is not None and value != "":
            return value
    return None


def _intent_row(row: dict, *, owner: str, source: str, location: str,
                inherited: dict) -> dict | None:
    intent_id = _first(row, "intent_id", "client_order_id", "order_id", "id", "txid")
    side = _first(row, "side", "start_side")
    quantity = _first(row, "requested_qty", "qty", "quantity")
    if not intent_id or not side or quantity is None:
        return None
    side_str = str(side).upper()
    if side_str not in {"BUY", "SELL"}:
        return None
    status = str(_first(
        row, "submission_outcome", "lifecycle", "last_status", "status",
    ) or ("accepted" if _first(row, "order_id", "id", "txid") else "pending")).lower()
    terminal_payload = row.get("terminal_status")
    if isinstance(terminal_payload, dict):
        status = str(terminal_payload.get("status") or status).lower()
    if status in _TERMINAL:
        return None
    symbol = (_first(row, "symbol", "ticker", "pair")
              or _first(inherited, "symbol", "ticker", "pair"))
    venue = (_first(row, "provider_name", "venue") or inherited.get("venue")
             or owner.split("_", 1)[0])
    return {
        "intent_id": str(intent_id),
        "owner": owner,
        "venue": str(venue),
        "symbol": None if symbol is None else str(symbol),
        "side": side_str,
        "kind": _first(row, "kind", "motivation"),
        "status": status,
        "order_id": _first(row, "order_id", "id", "txid"),
        "client_order_id": row.get("client_order_id"),
        "requested_qty": quantity,
        "executed_qty": _first(row, "filled_qty", "executed_qty", "delivered_qty"),
        "requested_price": _first(row, "requested_price", "price", "limit"),
        "source": source,
        "location": location,
    }


def _walk(value, *, owner: str, source: str, location: str = "$",
          inherited: dict | None = None) -> Iterable[dict]:
    inherited = dict(inherited or {})
    if isinstance(value, dict):
        if value.get("terminal") is True:
            return
        next_context = dict(inherited)
        for key in ("symbol", "ticker", "pair", "venue", "provider_name"):
            if value.get(key) not in (None, ""):
                next_context["venue" if key == "provider_name" else key] = value[key]
        row = _intent_row(
            value, owner=owner, source=source, location=location,
            inherited=next_context,
        )
        if row is not None:
            yield row
        for key, child in value.items():
            if isinstance(child, (dict, list)):
                yield from _walk(
                    child, owner=owner, source=source,
                    location=f"{location}.{key}", inherited=next_context,
                )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(
                child, owner=owner, source=source,
                location=f"{location}[{index}]", inherited=inherited,
            )


def _read(path: str, *, json_lines: bool):
    with open(path, encoding="utf-8") as handle:
        if json_lines:
            return [json.loads(line) for line in handle if line.strip()]
        return json.load(handle)


def _path_context(path: str) -> dict:
    name = os.path.basename(path)
    if name.startswith(".state_") and name.endswith(".json"):
        return {"symbol": name[len(".state_"):-len(".json")]}
    return {}


def _extract_exposure(payload: dict, *, owner: str, source: str,
                      inherited: dict) -> dict | None:
    if not isinstance(payload, dict):
        return None
    qty = payload.get("qty")
    if qty is None:
        return None
    try:
        qty_val = float(qty)
    except (ValueError, TypeError):
        return None
    if abs(qty_val) < 1e-9:
        return None
    cost = _first(payload, "cost_usd", "cost", "spent_cash", "spent")
    cost_val = None
    if cost is not None:
        try:
            cost_val = float(cost)
        except (ValueError, TypeError):
            pass
    entry_price = payload.get("entry_price")
    entry_price_val = None
    if entry_price is not None:
        try:
            entry_price_val = float(entry_price)
        except (ValueError, TypeError):
            pass
    symbol = (_first(payload, "symbol", "ticker", "pair")
              or _first(inherited, "symbol", "ticker", "pair"))
    venue = (_first(payload, "provider_name", "venue") or inherited.get("venue")
             or owner.split("_", 1)[0])
    return {
        "owner": owner,
        "venue": str(venue),
        "symbol": None if symbol is None else str(symbol),
        "qty": qty_val,
        "cost": cost_val,
        "entry_price": entry_price_val,
        "entry_ts": payload.get("entry_ts"),
        "source": source,
    }


def _build_summary(intents: list[dict], exposures: list[dict]) -> dict:
    notional_by_sym: dict[str, dict[str, float]] = {}
    total_buy = 0.0
    total_sell = 0.0
    for it in intents:
        sym = it.get("symbol") or "UNKNOWN"
        side = (it.get("side") or "").upper()
        req_qty = float(it.get("requested_qty") or 0.0)
        exec_qty = float(it.get("executed_qty") or 0.0)
        unfilled = max(0.0, req_qty - exec_qty)
        price = it.get("requested_price")
        notional = (unfilled * float(price)) if price is not None else 0.0

        entry = notional_by_sym.setdefault(sym, {"buy": 0.0, "sell": 0.0})
        if side == "BUY":
            entry["buy"] += notional
            total_buy += notional
        elif side == "SELL":
            entry["sell"] += notional
            total_sell += notional

    for entry in notional_by_sym.values():
        entry["buy"] = round(entry["buy"], 2)
        entry["sell"] = round(entry["sell"], 2)

    exp_by_sym: dict[str, dict[str, float | None]] = {}
    total_cost = 0.0
    for exp in exposures:
        sym = exp.get("symbol") or "UNKNOWN"
        qty = exp.get("qty", 0.0)
        cost = exp.get("cost")
        if cost is not None:
            total_cost += cost
        entry = exp_by_sym.setdefault(sym, {"qty": 0.0, "cost": 0.0})
        entry["qty"] += qty
        if cost is not None:
            entry["cost"] = (entry["cost"] or 0.0) + cost

    for entry in exp_by_sym.values():
        entry["qty"] = round(entry["qty"], 8)
        if entry["cost"] is not None:
            entry["cost"] = round(entry["cost"], 2)

    return {
        "pending_notional": {
            "by_symbol": dict(sorted(notional_by_sym.items())),
            "total_buy": round(total_buy, 2),
            "total_sell": round(total_sell, 2),
        },
        "net_exposures": {
            "by_symbol": dict(sorted(exp_by_sym.items())),
            "positions": exposures,
            "total_cost": round(total_cost, 2),
        },
    }


def format_summary(result: dict) -> str:
    summary = result.get("summary", {})
    notional = summary.get("pending_notional", {})
    exposures = summary.get("net_exposures", {})
    lines = [
        "=" * 60,
        "ACTIVE INTENTS & EXPOSURE SUMMARY",
        "=" * 60,
        f"Total active intents:  {len(result.get('intents', []))}",
        f"Total open positions:  {len(exposures.get('positions', []))}",
        f"Total cost basis:      ${exposures.get('total_cost', 0.0):.2f}",
        f"Pending BUY notional:  ${notional.get('total_buy', 0.0):.2f}",
        f"Pending SELL notional: ${notional.get('total_sell', 0.0):.2f}",
        "",
        "Net Exposures:",
    ]
    positions = exposures.get("positions", [])
    if positions:
        for pos in positions:
            cost_str = f" (${pos['cost']:.2f})" if pos.get("cost") is not None else ""
            lines.append(
                f"  - {pos.get('symbol')}: {pos.get('qty')}{cost_str} [{pos.get('venue')}]"
            )
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append("Pending Notional by Symbol:")
    by_sym = notional.get("by_symbol", {})
    if by_sym:
        for sym, amounts in by_sym.items():
            parts = []
            if amounts.get("buy"):
                parts.append(f"BUY ${amounts['buy']:.2f}")
            if amounts.get("sell"):
                parts.append(f"SELL ${amounts['sell']:.2f}")
            lines.append(f"  - {sym}: {', '.join(parts)}")
    else:
        lines.append("  (none)")
    lines.append("=" * 60)
    return "\n".join(lines)


def build_active_intent_index(root: str, sources=DEFAULT_SOURCES) -> dict:
    """Return normalized active intents and read errors without mutating sources."""
    root = os.path.abspath(root)
    intents = []
    exposures = []
    errors = []
    files = []
    for spec in sources:
        for path in sorted(glob.glob(os.path.join(root, spec.pattern))):
            if _is_backup(path):
                continue
            relative = os.path.relpath(path, root).replace(os.sep, "/")
            files.append(relative)
            try:
                payload = _read(path, json_lines=spec.json_lines)
                inherited = _path_context(path)
                intents.extend(_walk(
                    payload, owner=spec.owner, source=relative,
                    inherited=inherited,
                ))
                if not spec.json_lines and isinstance(payload, dict):
                    exp = _extract_exposure(
                        payload, owner=spec.owner, source=relative,
                        inherited=inherited,
                    )
                    if exp is not None:
                        exposures.append(exp)
            except (OSError, ValueError, TypeError) as exc:
                errors.append({"source": relative, "error": f"{type(exc).__name__}: {exc}"})
    intents.sort(key=lambda row: (
        row["owner"], row.get("symbol") or "", row["side"], row["intent_id"],
        row["source"], row["location"],
    ))
    exposures.sort(key=lambda row: (
        row["owner"], row.get("symbol") or "", row["source"],
    ))
    return {
        "schema_version": 1,
        "read_only": True,
        "files": files,
        "intents": intents,
        "exposures": exposures,
        "summary": _build_summary(intents, exposures),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the read-only active-intent index.")
    parser.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--summary", action="store_true", help="Print human-readable text summary.")
    args = parser.parse_args()
    result = build_active_intent_index(args.root)
    if args.summary:
        print(format_summary(result))
    else:
        print(json.dumps(result, indent=None if args.compact else 2, sort_keys=True))
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

