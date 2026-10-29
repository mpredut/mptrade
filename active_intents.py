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

from botcore import _parse_bool_flag, safe_float, safe_nonneg_float, safe_pos_float
from cachepaths import normalize_venue_name
from providers.base import _clean_single_order_id, normalize_side


@dataclass(frozen=True)
class StateSource:
    owner: str
    pattern: str
    json_lines: bool = False


DEFAULT_SOURCES = (
    StateSource("global_outbox", "cachedb/order_retry_queue.jsonl", True),
    StateSource("assetguardian", "cachedb/assetguardian_state.json"),
    StateSource("rtrade", "cachedb/rtrade_pairs_state.json"),
    StateSource("binance_trailing", "cachedb/trailing_state.json"),
    StateSource("kraken", "kraken/.state_*.json"),
    StateSource("kraken_trailing", "kraken/trailing_state.json"),
    StateSource("t212", "212trading/.state_*.json"),
    StateSource("t212_one_shot", "212trading/.t212_order.*.json"),
    StateSource("hyperliquid", "hyperliquid/.state_*.json"),
)

_TERMINAL = {"closed", "filled", "canceled", "cancelled", "expired", "rejected"}
_INVALID_SCALAR_STRINGS = {"nan", "null", "none", "inf", "-inf", "+inf"}


def _is_backup(path: str) -> bool:
    name = os.path.basename(path)
    return any(token in name for token in (".pre_", ".bak", ".backup", ".tmp"))


def _clean_id(val) -> str | None:
    return _clean_single_order_id(val)


def _first(row: dict, *names):
    for name in names:
        value = row.get(name)
        if value is None or isinstance(value, (bool, dict, list, tuple, set, frozenset)):
            continue
        if isinstance(value, float) and safe_float(value, default=None) is None:
            continue
        if isinstance(value, str):
            s = value.strip()
            if not s or s.lower() in _INVALID_SCALAR_STRINGS:
                continue
        return value
    return None


def _resolve_symbol_and_venue(payload: dict, inherited: dict, owner: str) -> tuple[str | None, str]:
    symbol = (_first(payload, "symbol", "ticker", "pair")
              or _first(inherited, "symbol", "ticker", "pair"))
    raw_venue = (_first(payload, "provider_name", "venue") or _first(inherited, "venue")
                 or owner.split("_", 1)[0])
    venue = normalize_venue_name(str(raw_venue)) or str(raw_venue)
    return (None if symbol is None else str(symbol), str(venue))


def _intent_row(row: dict, *, owner: str, source: str, location: str,
                inherited: dict) -> dict | None:
    if _parse_bool_flag(row.get("active"), True) is False:
        return None
    place_kwargs = row.get("place_kwargs")
    nested_kwargs = place_kwargs if isinstance(place_kwargs, dict) else {}
    client_order_id = _clean_id(_first(row, "client_order_id")) or _clean_id(_first(nested_kwargs, "client_order_id"))
    order_id = _clean_id(_first(row, "order_id", "id", "txid"))
    intent_id = _clean_id(_first(row, "intent_id", "client_order_id", "order_id", "id", "txid")) or client_order_id or order_id
    side = _first(row, "side", "start_side")
    quantity = _first(row, "requested_qty", "qty", "quantity", "vol", "sz", "size")
    qty_f = safe_pos_float(quantity, default=None)
    if not intent_id or not side or isinstance(side, bool) or qty_f is None:
        return None
    side_str = normalize_side(side)
    if side_str not in {"BUY", "SELL"}:
        return None
    status = str(_first(
        row, "submission_outcome", "lifecycle", "last_status", "status",
    ) or ("accepted" if order_id else "pending")).strip().lower()
    terminal_payload = row.get("terminal_status")
    if isinstance(terminal_payload, dict):
        status = str(_first(terminal_payload, "status") or status).strip().lower()
    elif isinstance(terminal_payload, str):
        ts_val = _first(row, "terminal_status")
        if ts_val:
            status = str(ts_val).strip().lower()
    if status in _TERMINAL:
        return None
    symbol, venue = _resolve_symbol_and_venue(row, inherited, owner)
    return {
        "intent_id": str(intent_id),
        "owner": owner,
        "venue": venue,
        "symbol": symbol,
        "side": side_str,
        "kind": _first(row, "kind", "motivation"),
        "status": status,
        "order_id": order_id,
        "client_order_id": client_order_id,
        "requested_qty": quantity,
        "executed_qty": _first(row, "filled_qty", "executed_qty", "delivered_qty", "exec_vol", "applied_fill_qty", "filled_sz"),
        "requested_price": _first(row, "requested_price", "price", "limit", "limit_price", "ref_price", "px"),
        "source": source,
        "location": location,
    }


def _walk(value, *, owner: str, source: str, location: str = "$",
          inherited: dict | None = None) -> Iterable[dict]:
    inherited = dict(inherited or {})
    if isinstance(value, dict):
        if _parse_bool_flag(value.get("terminal"), False) is True:
            return
        next_context = dict(inherited)
        for key in ("symbol", "ticker", "pair", "venue", "provider_name"):
            val_k = _first(value, key)
            if val_k is not None:
                next_context["venue" if key == "provider_name" else key] = val_k
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
    from state_io import read_json_file, read_jsonl_records, read_recovering_previous

    if json_lines:
        return read_recovering_previous(path, lambda p: read_jsonl_records(p, strict=True))
    return read_json_file(path, recover_previous=True)


def _path_context(path: str) -> dict:
    name = os.path.basename(path)
    if name.startswith(".state_") and name.endswith(".json"):
        return {"symbol": name[len(".state_"):-len(".json")]}
    return {}


def _extract_exposure(payload: dict, *, owner: str, source: str,
                      inherited: dict) -> dict | None:
    if not isinstance(payload, dict):
        return None
    qty_val = safe_float(payload.get("qty"), default=None)
    if qty_val is None or abs(qty_val) < 1e-9:
        return None
    cost_val = safe_nonneg_float(_first(payload, "cost_usd", "cost", "spent_cash", "spent"), default=None)
    entry_price_val = safe_pos_float(payload.get("entry_price"), default=None)
    symbol, venue = _resolve_symbol_and_venue(payload, inherited, owner)
    return {
        "owner": owner,
        "venue": venue,
        "symbol": symbol,
        "qty": qty_val,
        "cost": cost_val,
        "entry_price": entry_price_val,
        "entry_ts": _first(payload, "entry_ts"),
        "source": source,
    }


def _build_summary(intents: list[dict], exposures: list[dict]) -> dict:
    notional_by_sym: dict[str, dict[str, float]] = {}
    total_buy = 0.0
    total_sell = 0.0
    for it in intents:
        sym = it.get("symbol") or "UNKNOWN"
        side = (it.get("side") or "").upper()
        req_qty = safe_float(it.get("requested_qty"), default=0.0)
        exec_qty = safe_float(it.get("executed_qty"), default=0.0)
        unfilled = max(0.0, req_qty - exec_qty)
        price_val = safe_pos_float(it.get("requested_price"), default=None)
        notional = (unfilled * price_val) if price_val is not None else 0.0

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
        if not isinstance(exp, dict):
            continue
        sym = exp.get("symbol") or "UNKNOWN"
        qty = safe_float(exp.get("qty"), default=0.0)
        cost = safe_nonneg_float(exp.get("cost"), default=None)
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
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    notional = summary.get("pending_notional") if isinstance(summary.get("pending_notional"), dict) else {}
    exposures = summary.get("net_exposures") if isinstance(summary.get("net_exposures"), dict) else {}
    total_cost_val = safe_float(exposures.get("total_cost"), default=0.0) or 0.0
    total_buy_val = safe_float(notional.get("total_buy"), default=0.0) or 0.0
    total_sell_val = safe_float(notional.get("total_sell"), default=0.0) or 0.0
    lines = [
        "=" * 60,
        "ACTIVE INTENTS & EXPOSURE SUMMARY",
        "=" * 60,
        f"Total active intents:  {len(result.get('intents') or [])}",
        f"Total open positions:  {len(exposures.get('positions') or [])}",
        f"Total cost basis:      ${total_cost_val:.2f}",
        f"Pending BUY notional:  ${total_buy_val:.2f}",
        f"Pending SELL notional: ${total_sell_val:.2f}",
        "",
        "Net Exposures:",
    ]
    positions = exposures.get("positions") or []
    if positions:
        for pos in positions:
            if not isinstance(pos, dict):
                continue
            pos_cost = safe_float(pos.get("cost"), default=None)
            cost_str = f" (${pos_cost:.2f})" if pos_cost is not None else ""
            lines.append(
                f"  - {pos.get('symbol')}: {pos.get('qty')}{cost_str} [{pos.get('venue')}]"
            )
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append("Pending Notional by Symbol:")
    by_sym = notional.get("by_symbol") if isinstance(notional.get("by_symbol"), dict) else {}
    if by_sym:
        for sym, amounts in by_sym.items():
            if not isinstance(amounts, dict):
                continue
            parts = []
            buy_v = safe_float(amounts.get("buy"), default=0.0) or 0.0
            sell_v = safe_float(amounts.get("sell"), default=0.0) or 0.0
            if buy_v:
                parts.append(f"BUY ${buy_v:.2f}")
            if sell_v:
                parts.append(f"SELL ${sell_v:.2f}")
            lines.append(f"  - {sym}: {', '.join(parts)}")
    else:
        lines.append("  (none)")
    lines.append("=" * 60)
    return "\n".join(lines)


def build_active_intent_index(root: str, sources=DEFAULT_SOURCES) -> dict:
    """Return normalized active intents and read errors without mutating sources."""
    from cachepaths import legacy_filenames_for, resolve_cachedb_filepath

    root = os.path.abspath(root)
    intents = []
    exposures = []
    errors = []
    files = []
    for spec in sources:
        matched_paths = sorted(glob.glob(os.path.join(glob.escape(root), spec.pattern)))
        if not matched_paths and spec.pattern.startswith("cachedb/"):
            resolved_pat = str(resolve_cachedb_filepath(os.path.join(root, spec.pattern)))
            matched_paths = sorted(glob.glob(resolved_pat))
            if not matched_paths:
                for leg_p in legacy_filenames_for(resolved_pat):
                    matched_paths = sorted(glob.glob(leg_p))
                    if matched_paths:
                        break
        for path in matched_paths:
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

