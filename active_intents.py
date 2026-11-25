"""Read-only index and end-to-end causal tracer of financial intents and orders across AI and bot state files.

The index normalizes visibility only. It never writes state, submits orders, cancels
orders, or decides retry. Each strategy remains the authority for its campaign and
financial policy.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import glob
import os
from typing import Any, Iterable

from botcore import (
    _parse_bool_flag,
    add_store_true_cli_args,
    clean_order_id,
    clean_scalar_str,
    clean_upper_scalar,
    coerce_dict_maps,
    extract_order_id,
    extract_order_price,
    first_pos_float,
    normalize_side,
    optional_env,
    resolve_now_ts,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
)
from cacheaccount import load_unified_order_book_evidence
from cachepaths import (
    ASSETGUARDIAN_STATE_FILENAME,
    AUTONOMOUS_TRADE_INTENTS_STATE_FILENAME,
    ORDER_RETRY_QUEUE_FILENAME,
    RTRADE_PAIRS_STATE_FILENAME,
    TRAILING_STATE_FILENAME,
    VENUE_BALANCES_STATE_FILENAME,
    cache_relpath,
    is_cachedb_relpath,
    is_shadow_or_paper_order_id,
    is_shadow_or_paper_state_filename,
    is_shadow_or_paper_state_payload,
    legacy_filenames_for,
    normalize_venue_name,
    resolve_cachedb_filepath,
)
from instrument_registry import resolve_symbol_identity
from providers.execution_audit import iter_audit_jsonl_paths
from providers.intent_state import (
    _extract_intent_id,
    _intent_order_ids,
    ai_intent_client_order_id,
    extract_intent_limit_price,
    extract_intent_notional,
    extract_intent_order_type,
    extract_intent_primary_order_id,
    extract_intent_side,
    extract_intent_timestamp_s,
)
from providers.order_id_context import (
    all_correlatable_client_order_ids,
    extract_client_order_id,
    resolve_order_actor,
    resolve_order_owner,
)
from providers.order_ledger import (
    OrderLedgerReader,
    OrderLedgerState,
    TERMINAL_LEDGER_STATES,
    classify_fill_evidence_state,
)
from providers.order_ohlc_utils import extract_executed_qty, extract_orig_qty
from providers.order_status_ssot import (
    OPEN_WORKING_OR_EMPTY_VENUE_STATUSES,
    TERMINAL_ACTIVE_INTENT_STATUSES,
    TERMINAL_AUDIT_VENUE_STATUSES,
    TERMINAL_CANCEL_VENUE_STATUSES,
    classify_order_lifecycle_state,
    is_open_working_status,
)
from state_io import (
    encode_json_payload,
    read_json_dict_recovering,
    read_json_file,
    read_jsonl_dicts,
    read_jsonl_records,
    read_recovering_previous,
)


@dataclass(frozen=True)
class StateSource:
    owner: str
    pattern: str
    json_lines: bool = False


DEFAULT_SOURCES = (
    StateSource("global_outbox", cache_relpath(ORDER_RETRY_QUEUE_FILENAME), True),
    StateSource("autonomous_ai", cache_relpath(AUTONOMOUS_TRADE_INTENTS_STATE_FILENAME)),
    StateSource("assetguardian", cache_relpath(ASSETGUARDIAN_STATE_FILENAME)),
    StateSource("rtrade", cache_relpath(RTRADE_PAIRS_STATE_FILENAME)),
    StateSource("binance_trailing", cache_relpath(TRAILING_STATE_FILENAME)),
    StateSource("kraken", "kraken/.state_*.json"),
    StateSource("kraken_trailing", f"kraken/{TRAILING_STATE_FILENAME}"),
    StateSource("t212", "212trading/.state_*.json"),
    StateSource("t212_one_shot", "212trading/.t212_order.*.json"),
    StateSource("hyperliquid", "hyperliquid/.state_*.json"),
)

_TERMINAL = TERMINAL_ACTIVE_INTENT_STATUSES
_STATUS_FIELDS = ("last_status", "status", "lifecycle", "submission_outcome", "submit_status")


def _first(row: dict, *names):
    for name in names:
        value = row.get(name)
        if value is None or isinstance(value, (bool, dict, list, tuple, set, frozenset)):
            continue
        if isinstance(value, float) and safe_float(value, default=None) is None:
            continue
        if isinstance(value, str):
            if clean_scalar_str(value) is None:
                continue
        return value
    return None


def _canonical_symbol(raw_sym: Any) -> str | None:
    if raw_sym is None:
        return None
    s = str(raw_sym).strip()
    if not s:
        return None
    ident = resolve_symbol_identity(s)
    return ident.storage_key if ident.is_tokenized_equity and ident.storage_key else s


def _autonomous_ai_intent_row(
    row: dict,
    *,
    source: str,
    location: str,
    include_terminal: bool = False,
) -> dict | None:
    intent_id = _extract_intent_id(row)
    side_str = extract_intent_side(row)
    if not intent_id or side_str not in {"BUY", "SELL"}:
        return None
    st_up = clean_upper_scalar(row.get("status")) or ""
    ep = row.get("execution_payload") if isinstance(row.get("execution_payload"), dict) else {}
    ep_st = clean_upper_scalar(ep.get("status")) or ""
    reconciled_term = _parse_bool_flag(row.get("reconciled_terminal"), False) or _parse_bool_flag(
        ep.get("reconciled_terminal"), False
    )
    order_type = extract_intent_order_type(row)
    exec_qty = extract_executed_qty(ep, default=0.0)
    is_active = False
    if st_up in {"PENDING", "PENDING_EXECUTION", "SUBMIT_UNKNOWN"}:
        is_active = True
    elif st_up == "EXECUTED" and not reconciled_term:
        if ep_st not in (TERMINAL_AUDIT_VENUE_STATUSES | TERMINAL_CANCEL_VENUE_STATUSES):
            if not (order_type == "MARKET" and not ep_st and exec_qty > 0.0):
                is_active = True
    if not is_active and not include_terminal:
        return None

    req_price = first_pos_float(
        extract_intent_limit_price(row, default=None),
        extract_order_price(ep, default=None),
        row.get("requested_price"),
        row.get("reference_price"),
    )
    orig_qty = (
        first_pos_float(
            extract_orig_qty(ep, default=None),
            row.get("requested_qty"),
            row.get("qty"),
            row.get("quantity"),
        )
        or 0.0
    )
    if orig_qty <= 0.0:
        notional = extract_intent_notional(row, default=None)
        if notional is not None and req_price is not None and req_price > 0.0:
            orig_qty = round(notional / req_price, 8)
    if orig_qty <= 0.0 and not include_terminal:
        return None

    order_id = clean_order_id(extract_intent_primary_order_id(row, default=None, include_top_level=True))
    client_order_id = clean_order_id(extract_client_order_id(ep, default=None)) or ai_intent_client_order_id(intent_id)
    raw_sym = _first(row, "symbol") or _first(ep, "symbol", "ticker", "pair")
    raw_venue = _first(row, "venue") or "autonomous_ai"
    eff_status = (
        ep_st.lower()
        if (st_up == "EXECUTED" and ep_st and ep_st in OPEN_WORKING_OR_EMPTY_VENUE_STATUSES)
        else (st_up.lower() or "pending")
    )
    _, created_ts_s = extract_intent_timestamp_s(row)
    out_row = {
        "intent_id": str(intent_id),
        "owner": "autonomous_ai",
        "venue": str(normalize_venue_name(str(raw_venue)) or raw_venue),
        "symbol": _canonical_symbol(raw_sym),
        "side": side_str,
        "kind": order_type.lower(),
        "status": eff_status,
        "order_id": order_id,
        "client_order_id": client_order_id,
        "requested_qty": orig_qty if orig_qty > 0.0 else None,
        "executed_qty": exec_qty if exec_qty > 0.0 else None,
        "requested_price": req_price,
        "source": source,
        "location": location,
    }
    if created_ts_s is not None and created_ts_s > 0.0 and not include_terminal:
        out_row["_created_ts_s"] = created_ts_s
    return out_row


def _intent_row(
    row: dict,
    *,
    owner: str,
    source: str,
    location: str,
    inherited: dict,
    include_terminal: bool = False,
) -> dict | None:
    if owner == "autonomous_ai":
        return _autonomous_ai_intent_row(
            row, source=source, location=location, include_terminal=include_terminal
        )
    if not include_terminal and (
        _parse_bool_flag(row.get("active"), True) is False or is_shadow_or_paper_order_id(row)
    ):
        return None
    place_kwargs = row.get("place_kwargs")
    nested_kwargs = place_kwargs if isinstance(place_kwargs, dict) else {}
    client_order_id = clean_order_id(_first(row, "client_order_id")) or clean_order_id(_first(nested_kwargs, "client_order_id"))
    order_id = clean_order_id(_first(row, "order_id", "id", "txid"))
    intent_id = clean_order_id(_first(row, "intent_id", "client_order_id", "order_id", "id", "txid")) or client_order_id or order_id
    side = _first(row, "side", "start_side")
    quantity = _first(row, "requested_qty", "qty", "quantity", "vol", "sz", "size")
    if not intent_id or not side or isinstance(side, bool) or safe_pos_float(quantity, default=None) is None:
        return None
    side_str = normalize_side(side)
    if side_str not in {"BUY", "SELL"}:
        return None
    first_status = None
    for status_field in _STATUS_FIELDS:
        field_val = _first(row, status_field)
        if field_val is not None:
            norm_st = str(field_val).strip().lower()
            if not include_terminal and norm_st in _TERMINAL:
                return None
            if first_status is None:
                first_status = norm_st
    status = first_status or ("accepted" if order_id else "pending")
    terminal_payload = row.get("terminal_status")
    if isinstance(terminal_payload, dict):
        status = str(_first(terminal_payload, "status") or status).strip().lower()
    elif isinstance(terminal_payload, str) and (ts_val := _first(row, "terminal_status")):
        status = str(ts_val).strip().lower()
    if not include_terminal and status in _TERMINAL:
        return None
    raw_sym = _first(row, "symbol", "ticker", "pair") or _first(inherited, "symbol", "ticker", "pair")
    raw_venue = _first(row, "provider_name", "venue") or _first(inherited, "venue") or owner.split("_", 1)[0]
    symbol, venue = _canonical_symbol(raw_sym), str(normalize_venue_name(str(raw_venue)) or raw_venue)
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


def _walk(
    value,
    *,
    owner: str,
    source: str,
    location: str = "$",
    inherited: dict | None = None,
    include_terminal: bool = False,
) -> Iterable[dict]:
    inherited = dict(inherited or {})
    if isinstance(value, dict):
        if not include_terminal and _parse_bool_flag(value.get("terminal"), False) is True:
            return
        next_context = dict(inherited)
        for key in ("symbol", "ticker", "pair", "venue", "provider_name"):
            if (val_k := _first(value, key)) is not None:
                next_context["venue" if key == "provider_name" else key] = val_k
        if (
            row := _intent_row(
                value,
                owner=owner,
                source=source,
                location=location,
                inherited=next_context,
                include_terminal=include_terminal,
            )
        ) is not None:
            yield row
        if owner == "autonomous_ai":
            return
        for key, child in value.items():
            if isinstance(child, (dict, list)):
                yield from _walk(
                    child,
                    owner=owner,
                    source=source,
                    location=f"{location}.{key}",
                    inherited=next_context,
                    include_terminal=include_terminal,
                )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(
                child,
                owner=owner,
                source=source,
                location=f"{location}[{index}]",
                inherited=inherited,
                include_terminal=include_terminal,
            )


def _read(path: str, *, json_lines: bool):
    if json_lines:
        return read_recovering_previous(path, lambda p: read_jsonl_records(p, strict=True))
    return read_json_file(path, recover_previous=True)


def _extract_exposure(payload: dict, *, owner: str, source: str, inherited: dict) -> dict | None:
    if not isinstance(payload, dict):
        return None
    qty_val = safe_float(payload.get("qty"), default=None)
    if qty_val is None or abs(qty_val) < 1e-9:
        return None
    raw_sym = _first(payload, "symbol", "ticker", "pair") or _first(inherited, "symbol", "ticker", "pair")
    raw_venue = _first(payload, "provider_name", "venue") or _first(inherited, "venue") or owner.split("_", 1)[0]
    symbol, venue = (None if raw_sym is None else str(raw_sym)), str(normalize_venue_name(str(raw_venue)) or raw_venue)
    return {
        "owner": owner,
        "venue": venue,
        "symbol": symbol,
        "qty": qty_val,
        "cost": safe_nonneg_float(_first(payload, "cost_usd", "cost", "spent_cash", "spent"), default=None),
        "entry_price": safe_pos_float(payload.get("entry_price"), default=None),
        "entry_ts": _first(payload, "entry_ts"),
        "source": source,
    }


def _build_summary(intents: list[dict], exposures: list[dict]) -> dict:
    notional_by_sym: dict[str, dict[str, float]] = {}
    total_buy = 0.0
    total_sell = 0.0
    seen_order_keys: set[tuple[str, str]] = set()
    seen_intent_ids: set[str] = set()
    for it in intents or ():
        if not isinstance(it, dict):
            continue
        venue_k = str(it.get("venue") or "").strip().lower()
        oid_k = clean_order_id(it.get("order_id"))
        iid_k = clean_order_id(it.get("intent_id"))
        if oid_k and (venue_k, oid_k) in seen_order_keys:
            continue
        if iid_k and iid_k in seen_intent_ids and it.get("owner") == "global_outbox":
            continue
        if oid_k:
            seen_order_keys.add((venue_k, oid_k))
        if iid_k:
            seen_intent_ids.add(iid_k)
        raw_sym = _canonical_symbol(it.get("symbol"))
        sym = raw_sym.strip() if isinstance(raw_sym, str) and raw_sym.strip() else "UNKNOWN"
        side = normalize_side(it.get("side"))
        req_qty = safe_nonneg_float(it.get("requested_qty"), default=0.0) or 0.0
        exec_qty = safe_nonneg_float(it.get("executed_qty"), default=0.0) or 0.0
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
    for exp in exposures or ():
        if not isinstance(exp, dict):
            continue
        raw_sym = exp.get("symbol")
        sym = raw_sym.strip() if isinstance(raw_sym, str) and raw_sym.strip() else "UNKNOWN"
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
    (res_d,) = coerce_dict_maps(result)
    (summary,) = coerce_dict_maps(res_d.get("summary"))
    notional, exposures = coerce_dict_maps(
        summary.get("pending_notional"), summary.get("net_exposures")
    )
    total_cost_val = safe_float(exposures.get("total_cost"), default=0.0) or 0.0
    total_buy_val = safe_float(notional.get("total_buy"), default=0.0) or 0.0
    total_sell_val = safe_float(notional.get("total_sell"), default=0.0) or 0.0
    lines = [
        "=" * 60,
        "ACTIVE INTENTS & EXPOSURE SUMMARY",
        "=" * 60,
        f"Total active intents:  {len(res_d.get('intents') or [])}",
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

    lines.extend(("", "Pending Notional by Symbol:"))
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
            if not parts:
                parts.append("$0.00")
            lines.append(f"  - {sym}: {', '.join(parts)}")
    else:
        lines.append("  (none)")
    lines.append("=" * 60)
    return "\n".join(lines)


def _iter_resolved_source_paths(root: str, spec: StateSource) -> list[str]:
    matched_paths = sorted(glob.glob(os.path.join(glob.escape(root), spec.pattern)))
    if not matched_paths and is_cachedb_relpath(spec.pattern):
        resolved_pat = str(resolve_cachedb_filepath(os.path.join(root, spec.pattern)))
        matched_paths = sorted(glob.glob(resolved_pat))
        if not matched_paths:
            for leg_p in legacy_filenames_for(resolved_pat):
                matched_paths = sorted(glob.glob(leg_p))
                if matched_paths:
                    break
    return [
        p
        for p in matched_paths
        if not any(token in os.path.basename(p) for token in (".pre_", ".bak", ".backup", ".tmp"))
        and not is_shadow_or_paper_state_filename(p)
    ]


def build_active_intent_index(root: str, sources=DEFAULT_SOURCES) -> dict:
    """Return normalized active intents and read errors without mutating sources."""
    root = os.path.abspath(root)
    intents = []
    exposures = []
    errors = []
    files = []
    for spec in sources:
        for path in _iter_resolved_source_paths(root, spec):
            name = os.path.basename(path)
            relative = os.path.relpath(path, root).replace(os.sep, "/")
            files.append(relative)
            try:
                payload = _read(path, json_lines=spec.json_lines)
                if not spec.json_lines and isinstance(payload, dict) and is_shadow_or_paper_state_payload(payload):
                    continue
                inherited = (
                    {"symbol": name[len(".state_"):-len(".json")]}
                    if name.startswith(".state_") and name.endswith(".json")
                    else {}
                )
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
    intents = _prune_and_deduplicate_active_intents(intents, root)
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


def _order_key_tuple(venue: Any, symbol: Any, oid: Any) -> tuple[str, str, str]:
    v_k = str(normalize_venue_name(str(venue)) or venue or "").strip().lower()
    sym_c = (_canonical_symbol(symbol) or "").lower()
    oid_c = clean_order_id(oid) or ""
    return (v_k, sym_c, oid_c)


def _load_cached_order_book_state(
    root: str,
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, str], float],
    set[tuple[str, str, str]],
    dict[tuple[str, str, str], dict[str, Any]],
]:
    """Collect read-only resting open orders, per-symbol verified open-book timestamps, and SSOT-classified fill evidence."""
    ev = load_unified_order_book_evidence(
        os.path.join(root, "cachedb"),
        symbol_normalizer=_canonical_symbol,
    )
    return ev.resting_open_orders, ev.symbol_verified_ts, ev.terminal_order_keys, ev.ambiguous_filled_by_key


def _load_cached_order_book_index(
    root: str,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], float], set[tuple[str, str, str]]]:
    """Collect read-only resting open orders, per-symbol verified open-book timestamps, and explicitly terminal venue order keys."""
    out, covered_symbol_ts, terminal_order_keys, _ = _load_cached_order_book_state(root)
    return out, covered_symbol_ts, terminal_order_keys


def _load_cached_open_orders_and_covered_venues(root: str) -> tuple[list[dict[str, Any]], set[str]]:
    """Collect read-only resting open orders and the set of venues with an on-disk open-order snapshot."""
    open_orders, covered_symbol_ts, _ = _load_cached_order_book_index(root)
    return open_orders, {v_k for v_k, _ in covered_symbol_ts}


def _load_cached_open_orders(root: str) -> list[dict[str, Any]]:
    """Collect read-only resting open orders across all venue order caches and venue_balances_state."""
    return _load_cached_order_book_index(root)[0]


def _collect_terminal_outbox_keys(root: str) -> tuple[set[str], set[tuple[str, str, str]]]:
    """Return terminal intent_ids and ``(venue, symbol, order_id)`` tuples recorded in order_retry_queue.jsonl."""
    term_iids: set[str] = set()
    term_oids: set[tuple[str, str, str]] = set()
    for path in _iter_resolved_source_paths(root, DEFAULT_SOURCES[0]):
        try:
            records = _read(path, json_lines=True)
        except (OSError, ValueError, TypeError):
            continue
        if not isinstance(records, list):
            continue
        for rec in records:
            if not isinstance(rec, dict):
                continue
            is_term = False
            for st_field in _STATUS_FIELDS:
                fv = _first(rec, st_field)
                if fv is not None and str(fv).strip().lower() in _TERMINAL:
                    is_term = True
                    break
            if not is_term:
                tp = rec.get("terminal_status")
                tp_st = _first(tp, "status") if isinstance(tp, dict) else (_first(rec, "terminal_status") if isinstance(tp, str) else None)
                if tp_st is not None and str(tp_st).strip().lower() in _TERMINAL:
                    is_term = True
            if not is_term:
                continue
            if iid := clean_order_id(_first(rec, "intent_id")):
                term_iids.add(iid)
            raw_v = _first(rec, "provider_name", "venue")
            raw_s = _first(rec, "symbol", "pair", "ticker")
            if raw_v and (oid := clean_order_id(_first(rec, "order_id", "id", "txid"))):
                term_oids.add(_order_key_tuple(raw_v, raw_s, oid))
    return term_iids, term_oids


def _prune_and_deduplicate_active_intents(intents: list[dict[str, Any]], root: str) -> list[dict[str, Any]]:
    """Drop stale unconfirmed/closed historical AI intents and deduplicate global_outbox mirrors."""
    if not intents:
        return []
    now_s = resolve_now_ts(None)
    open_orders, covered_symbol_ts, terminal_order_keys, ambiguous_filled_by_key = _load_cached_order_book_state(root)
    live_open_keys: set[tuple[str, str, str]] = set()
    for oo in open_orders:
        if oid_oo := clean_order_id(extract_order_id(oo, default=None)):
            live_open_keys.add(_order_key_tuple(oo.get("venue"), oo.get("symbol"), oid_oo))

    outbox_rows = [r for r in intents if r.get("owner") == "global_outbox"]
    primary_rows = [r for r in intents if r.get("owner") != "global_outbox"]
    outbox_iids = {r["intent_id"] for r in outbox_rows if r.get("intent_id")}
    outbox_oids = {
        _order_key_tuple(r.get("venue"), r.get("symbol"), oid_r)
        for r in outbox_rows
        if (oid_r := clean_order_id(r.get("order_id")))
    }
    term_outbox_iids, term_outbox_oids = _collect_terminal_outbox_keys(root)
    eff_terminal_oids = terminal_order_keys | (term_outbox_oids - outbox_oids)
    eff_terminal_iids = term_outbox_iids - outbox_iids

    kept_primary: list[dict[str, Any]] = []
    for row in primary_rows:
        created_ts_s = safe_pos_float(row.pop("_created_ts_s", None), default=None)
        if row.get("owner") == "autonomous_ai":
            v_k = str(row.get("venue") or "").strip().lower()
            oid_k = clean_order_id(row.get("order_id"))
            row_key = _order_key_tuple(v_k, row.get("symbol"), oid_k) if oid_k else None
            is_live_open = bool(row_key and row_key in live_open_keys)
            is_fill_terminal = False
            if row_key and (amb_fill := ambiguous_filled_by_key.get(row_key)) is not None:
                intent_req_qty = safe_pos_float(row.get("qty") or row.get("requested_qty"), default=None)
                is_fill_terminal = (
                    classify_fill_evidence_state(amb_fill, req_qty=intent_req_qty, resting_now=is_live_open)
                    in TERMINAL_LEDGER_STATES
                )
            if (row_key and (row_key in eff_terminal_oids or is_fill_terminal)) or (
                row.get("intent_id") in eff_terminal_iids and not is_live_open
            ):
                continue
            if created_ts_s is not None:
                age_s = max(0.0, now_s - created_ts_s)
                st_lc = str(row.get("status") or "").strip().lower()
                if st_lc in {"pending", "pending_execution", "submit_unknown"} and age_s > 3600.0:
                    if row["intent_id"] not in outbox_iids and not is_live_open:
                        continue
                elif age_s > 300.0 and row_key:
                    sym_ver_ts = covered_symbol_ts.get((row_key[0], row_key[1]), 0.0)
                    if sym_ver_ts > 0.0 and sym_ver_ts >= created_ts_s and not is_live_open and row_key not in outbox_oids:
                        continue
        kept_primary.append(row)

    primary_iids = {r["intent_id"] for r in kept_primary if r.get("intent_id")}
    primary_oids = {
        _order_key_tuple(r.get("venue"), r.get("symbol"), oid_p)
        for r in kept_primary
        if (oid_p := clean_order_id(r.get("order_id")))
    }
    kept_outbox = [
        r
        for r in outbox_rows
        if r.get("intent_id") not in primary_iids
        and not (
            (oid_o := clean_order_id(r.get("order_id")))
            and _order_key_tuple(r.get("venue"), r.get("symbol"), oid_o) in primary_oids
        )
    ]
    return kept_primary + kept_outbox


def trace_order_or_intent(
    query_id: Any,
    root: str = ".",
    *,
    sources=DEFAULT_SOURCES,
    audit_dir: str | None = None,
    max_audit_days: int = 14,
) -> dict[str, Any]:
    """Build a single read-only SSOT causal trace across AI intents, bot states, outbox, open orders, and audit JSONL."""
    clean_q = clean_order_id(query_id)
    abs_root = os.path.abspath(root)
    if not clean_q:
        return {
            "query_id": "" if query_id is None or isinstance(query_id, bool) else str(query_id),
            "found": False,
            "canonical_state": None,
            "actor": None,
            "correlated_ids": [],
            "ai_intents": [],
            "state_records": [],
            "outbox_records": [],
            "open_orders": [],
            "audit_events": [],
        }

    correlated_lc: set[str] = {clean_q.lower()} | all_correlatable_client_order_ids(clean_q)

    def _expand_ids(*candidates: Any) -> None:
        for cand in candidates:
            if c_str := clean_order_id(cand):
                correlated_lc.add(c_str.lower())
                correlated_lc.update(all_correlatable_client_order_ids(c_str))

    ai_matches: list[dict[str, Any]] = []
    state_matches: list[dict[str, Any]] = []
    outbox_matches: list[dict[str, Any]] = []

    for spec in sources:
        for path in _iter_resolved_source_paths(abs_root, spec):
            name = os.path.basename(path)
            relative = os.path.relpath(path, abs_root).replace(os.sep, "/")
            try:
                payload = _read(path, json_lines=spec.json_lines)
            except (OSError, ValueError, TypeError):
                continue

            if spec.owner == "autonomous_ai" and isinstance(payload, list):
                for item in payload:
                    if not isinstance(item, dict):
                        continue
                    item_ids = {s.lower() for s in _intent_order_ids(item) if s}
                    iid = _extract_intent_id(item)
                    if iid:
                        item_ids.add(iid.lower())
                        item_ids.update(all_correlatable_client_order_ids(iid))
                    for c_oid in (item.get("canceled_prior_open_orders") if isinstance(item.get("canceled_prior_open_orders"), list) else ()):
                        if c_clean := clean_order_id(c_oid):
                            item_ids.add(c_clean.lower())
                    if item_ids & correlated_lc:
                        ai_matches.append(item)
                        for s_id in item_ids:
                            _expand_ids(s_id)
                continue

            if spec.owner == "global_outbox" and isinstance(payload, list):
                for rec in payload:
                    if not isinstance(rec, dict):
                        continue
                    pkw = rec.get("place_kwargs") if isinstance(rec.get("place_kwargs"), dict) else {}
                    rec_cands = [
                        rec.get("id"),
                        rec.get("intent_id"),
                        rec.get("order_id"),
                        rec.get("replaces_order_id"),
                        rec.get("client_order_id"),
                        pkw.get("client_order_id"),
                        pkw.get("parent_client_order_id"),
                    ]
                    rec_lc = {c.lower() for r_c in rec_cands if (c := clean_order_id(r_c))}
                    if rec_lc & correlated_lc:
                        outbox_matches.append(rec)
                        _expand_ids(*rec_cands)
                continue

            inherited = (
                {"symbol": name[len(".state_"):-len(".json")]}
                if name.startswith(".state_") and name.endswith(".json")
                else {}
            )
            for row in _walk(
                payload,
                owner=spec.owner,
                source=relative,
                inherited=inherited,
                include_terminal=True,
            ):
                row_cands = [row.get("intent_id"), row.get("order_id"), row.get("client_order_id")]
                row_lc = set()
                for rc_val in row_cands:
                    if clean_rc := clean_order_id(rc_val):
                        row_lc.add(clean_rc.lower())
                        row_lc.update(all_correlatable_client_order_ids(clean_rc))
                if row_lc & correlated_lc:
                    state_matches.append(row)
                    _expand_ids(*row_cands)

    open_matches: list[dict[str, Any]] = []
    for oo in _load_cached_open_orders(abs_root):
        oo_oid = clean_order_id(extract_order_id(oo, default=None))
        oo_cid = clean_order_id(extract_client_order_id(oo, default=None))
        oo_lc = {x.lower() for x in (oo_oid, oo_cid) if x}
        if oo_lc & correlated_lc:
            open_matches.append(oo)
            _expand_ids(oo_oid, oo_cid)

    ledger_reader = OrderLedgerReader(cachedb_dir=os.path.join(abs_root, "cachedb"))
    eff_audit_dir = (
        clean_scalar_str(audit_dir)
        or clean_scalar_str(optional_env("EXECUTION_AUDIT_DIR"))
        or os.path.join(abs_root, "logger", "execution_audit")
    )
    audit_events: list[dict[str, Any]] = []
    # Skip legacy execution_audit JSONL scan in production once OrderLedger is populated (unless audit_dir is explicitly passed).
    should_scan_legacy_audit = bool(audit_dir) or not (ledger_reader.list_event_files() or ledger_reader.list_closed_files())
    if should_scan_legacy_audit and os.path.isdir(eff_audit_dir):
        for audit_path in iter_audit_jsonl_paths(eff_audit_dir, unbounded_limit=max_audit_days):
            for rec in read_jsonl_dicts(audit_path):
                rec_cands = (rec.get("intent_id"), rec.get("order_id"), rec.get("client_order_id"))
                rec_lc = {c.lower() for r_c in rec_cands if (c := clean_order_id(r_c))}
                if rec_lc & correlated_lc:
                    audit_events.append(rec)
                    _expand_ids(*rec_cands)
    audit_events.sort(key=lambda r: safe_float(r.get("ts"), default=0.0) or 0.0)

    venue_st: Any = None
    local_st: Any = None
    sub_st: Any = None
    filled_q: float = 0.0
    orig_q: float | None = None
    actor: str | None = None

    def _resolve_trace_actor(
        order_obj: dict[str, Any] | None = None,
        *,
        cid_val: Any = None,
        owner_val: Any = None,
    ) -> str | None:
        eff_cid = clean_order_id(cid_val) or (
            extract_client_order_id(order_obj, default=None) if isinstance(order_obj, dict) else None
        )
        eff_owner = (clean_scalar_str(owner_val) or "").lower() or (
            resolve_order_owner(eff_cid).lower() if eff_cid else ""
        )
        if eff_owner == "manual":
            return "MANUAL"
        if isinstance(order_obj, dict):
            return resolve_order_actor(order_obj)
        if eff_cid:
            return resolve_order_actor({"clientOrderId": eff_cid})
        return clean_scalar_str(owner_val)

    if ai_matches:
        actor = "AI"
        latest_ai = ai_matches[-1]
        ep = latest_ai.get("execution_payload") if isinstance(latest_ai.get("execution_payload"), dict) else {}
        venue_st = ep.get("status") or latest_ai.get("status")
        sub_st = ep.get("submission_state")
        filled_q = max(filled_q, extract_executed_qty(ep, default=0.0))
        if (oq_ai := extract_orig_qty(ep, default=0.0)) > 0.0:
            orig_q = oq_ai
        _, _ = extract_intent_timestamp_s(latest_ai)
    if state_matches:
        st_last = state_matches[-1]
        local_st = local_st or st_last.get("status")
        filled_q = max(filled_q, safe_pos_float(st_last.get("executed_qty"), default=0.0) or 0.0)
        orig_q = orig_q or safe_pos_float(st_last.get("requested_qty"), default=None)
        if actor is None:
            actor = _resolve_trace_actor(cid_val=st_last.get("client_order_id"), owner_val=st_last.get("owner"))
    if outbox_matches:
        ob_last = outbox_matches[-1]
        local_st = local_st or ob_last.get("lifecycle")
        sub_st = sub_st or ob_last.get("submission_state")
        orig_q = orig_q or safe_pos_float(ob_last.get("qty"), default=None)
        if actor is None:
            pkw = ob_last.get("place_kwargs") if isinstance(ob_last.get("place_kwargs"), dict) else {}
            actor = _resolve_trace_actor(cid_val=pkw.get("client_order_id"), owner_val=pkw.get("parent_owner"))
    if audit_events:
        au_last = audit_events[-1]
        venue_st = au_last.get("venue_status") or venue_st
        local_st = au_last.get("status") or local_st
        sub_st = au_last.get("submission_state") or sub_st
        filled_q = max(filled_q, safe_pos_float(au_last.get("filled_qty"), default=0.0) or 0.0)
        orig_q = orig_q or safe_pos_float(au_last.get("qty"), default=None)
        if actor is None and au_last.get("client_order_id"):
            actor = _resolve_trace_actor(cid_val=au_last.get("client_order_id"), owner_val=au_last.get("owner"))
    if open_matches:
        oo_last = open_matches[-1]
        venue_st = oo_last.get("status") or venue_st or "OPEN"
        filled_q = max(filled_q, extract_executed_qty(oo_last, default=0.0))
        if (oq_oo := extract_orig_qty(oo_last, default=0.0)) > 0.0:
            orig_q = orig_q or oq_oo
        if actor is None:
            actor = _resolve_trace_actor(oo_last)

    ledger_trace = ledger_reader.trace_order(
        clean_q, max_event_days=max_audit_days
    )
    ledger_order = ledger_trace.get("active_order") or (
        ledger_trace["closed_orders"][-1] if ledger_trace.get("closed_orders") else None
    )
    ledger_events = list(ledger_trace.get("events") or ())
    for cid_tok in ledger_trace.get("correlated_ids") or ():
        correlated_lc.add(cid_tok)
    if isinstance(ledger_order, dict):
        venue_st = ledger_order.get("vst") or venue_st
        local_st = local_st or ledger_order.get("state")
        filled_q = max(filled_q, safe_pos_float(ledger_order.get("fq"), default=0.0) or 0.0)
        orig_q = orig_q or safe_pos_float(ledger_order.get("qty"), default=None)
        if actor is None:
            actor = _resolve_trace_actor(cid_val=ledger_order.get("cid"), owner_val=ledger_order.get("owner"))

    found = bool(
        ai_matches or state_matches or outbox_matches or open_matches or audit_events or ledger_order or ledger_events
    )
    canonical_state = (
        str(
            classify_order_lifecycle_state(
                venue_st,
                local_st,
                submission_state=sub_st,
                filled_qty=filled_q,
                orig_qty=orig_q,
                is_resting_in_book=bool(open_matches),
            )
        )
        if found
        else None
    )
    return {
        "query_id": clean_q,
        "found": found,
        "canonical_state": canonical_state,
        "actor": actor,
        "is_resting_in_book": bool(open_matches),
        "filled_qty": round(filled_q, 8),
        "orig_qty": orig_q,
        "correlated_ids": sorted(correlated_lc),
        "ledger_order": ledger_order,
        "ledger_events": ledger_events,
        "ai_intents": ai_matches,
        "state_records": state_matches,
        "outbox_records": outbox_matches,
        "open_orders": open_matches,
        "audit_events": audit_events,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the read-only active-intent index or trace an order/intent ID.")
    parser.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--trace", default=None, help="Trace an intent_id, client_order_id, or venue order_id across all sources.")
    add_store_true_cli_args(parser, "--compact", ("--summary", "Print human-readable text summary."))
    args = parser.parse_args()
    indent = None if _parse_bool_flag(args.compact, False) else 2
    if clean_scalar_str(args.trace):
        trace_res = trace_order_or_intent(args.trace, args.root)
        print(encode_json_payload(trace_res, indent=indent, sort_keys=True))
        return 0 if trace_res["found"] else 1
    result = build_active_intent_index(args.root)
    if _parse_bool_flag(args.summary, False):
        print(format_summary(result))
    else:
        print(encode_json_payload(result, indent=indent, sort_keys=True))
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
