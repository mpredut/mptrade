#!/usr/bin/env python3
"""Idempotent one-shot backfill tool migrating legacy ``execution_audit_*.jsonl`` into Order Ledger SSoT files.

Populates historical ``cachedb/orders/order_events_YYYY-MM-DD.jsonl`` and
``cachedb/orders/orders_closed_YYYY-MM.jsonl`` strictly prior to the first live Order Ledger event,
updating ``orders_active.json`` byte offsets under ``generation_file_lock`` so the live
``order_ledger_reconciler_worker.py`` daemon stays synchronized with zero duplicate replays.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from botcore import (
    add_store_true_cli_args,
    clean_order_id,
    clean_scalar_str,
    clean_upper_scalar,
    extract_order_id,
    normalize_epoch_sec,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
)
from providers.execution_audit import iter_audit_jsonl_paths
from providers.order_id_context import all_correlatable_client_order_ids, extract_client_order_id
from providers.order_ohlc_utils import _EXEC_QTY_KEYS, extract_executed_qty
from providers.order_ledger import (
    OrderLedgerReader,
    _AUDIT_EVENT_TO_LEDGER_EVENT,
    apply_order_event,
    build_order_event,
    utc_day_for_ts,
    utc_month_for_ts,
)
from state_io import (
    atomic_write_bytes,
    atomic_write_jsonl,
    encode_json_payload,
    generation_file_lock,
    read_bytes_file,
    read_jsonl_dicts,
    save_json_state,
)


_EXTRA_BACKFILL_EVENT_MAP: dict[str, str] = {
    "order_terminal_retry": "STATUS",
    "submit_rejected": "SUBMIT_REFUSED",
}


def _convert_audit_row_to_ledger_event(rec: dict[str, Any]) -> Optional[dict[str, Any]]:
    if not isinstance(rec, dict):
        return None
    ev_name = str(rec.get("event") or "").strip().lower()
    mapped = _AUDIT_EVENT_TO_LEDGER_EVENT.get(ev_name) or _EXTRA_BACKFILL_EVENT_MAP.get(ev_name)
    if mapped is None:
        return None
    ts_v = normalize_epoch_sec(rec.get("ts"), default=None)
    if ts_v is None:
        return None
    venue = rec.get("venue")
    symbol = rec.get("symbol")
    if not clean_scalar_str(symbol):
        return None
    fq_val = safe_nonneg_float(rec.get("filled_qty"), default=None)
    if fq_val is None and (any(k in rec for k in _EXEC_QTY_KEYS) or ("origSz" in rec and "sz" in rec)):
        fq_val = safe_nonneg_float(extract_executed_qty(rec, default=None), default=None)
    elif fq_val is None:
        fq_val = safe_nonneg_float(rec.get("cumQty"), default=None)
    cost_val = safe_nonneg_float(rec.get("cost"), default=None)
    avg_px = (
        (cost_val / fq_val)
        if (cost_val and fq_val and fq_val > 0.0)
        else safe_pos_float(rec.get("avg_price"), default=None)
    )
    return build_order_event(
        mapped,
        venue=venue,
        symbol=symbol,
        ts=ts_v,
        owner=rec.get("owner"),
        side=rec.get("side"),
        order_type=rec.get("type") or rec.get("order_type"),
        market=rec.get("market"),
        qty=rec.get("qty"),
        px=rec.get("price"),
        fq=fq_val,
        avg_px=avg_px,
        fee=rec.get("fee"),
        fee_asset=rec.get("fee_asset") or rec.get("feeAsset"),
        scale=rec.get("scale"),
        iid=rec.get("intent_id"),
        cid=rec.get("client_order_id") or extract_client_order_id(rec, default=None),
        vid=rec.get("order_id") or extract_order_id(rec, default=None),
        kind=rec.get("kind"),
        vst=rec.get("venue_status") or rec.get("status"),
        reason=rec.get("reason") or rec.get("failure_reason") or rec.get("error"),
        src="audit_backfill",
    )


def _find_first_live_event_boundary(reader: OrderLedgerReader, *, now_ts: Optional[float] = None) -> tuple[str, float]:
    """Return ``(day_utc, ts)`` of the earliest non-backfilled live Order Ledger event, or ``('', 0.0)``."""
    eff_now = safe_pos_float(now_ts, default=None) or time.time()
    max_ts = eff_now + 60.0
    for fpath in reader.list_event_files():
        earliest_ts: Optional[float] = None
        for row in read_jsonl_dicts(fpath):
            if clean_upper_scalar(row.get("src")) == "AUDIT_BACKFILL":
                continue
            ts_v = safe_pos_float(row.get("ts"), default=None)
            if ts_v is not None and ts_v <= max_ts:
                if earliest_ts is None or ts_v < earliest_ts:
                    earliest_ts = ts_v
        if earliest_ts is not None:
            return utc_day_for_ts(earliest_ts), earliest_ts
    return "", 0.0


def _closed_dedup_key(row: dict[str, Any]) -> tuple[str, str]:
    v = str(row.get("v") or "").lower()
    if vid := clean_order_id(row.get("vid")):
        return v, f"vid:{vid.lower()}"
    return v, str(row.get("key") or "")


def _closed_row_richness(row: dict[str, Any]) -> tuple[float, int, int]:
    fq_v = safe_nonneg_float(row.get("fq"), default=0.0) or 0.0
    has_px = 1 if (safe_pos_float(row.get("avg_px"), default=None) or safe_pos_float(row.get("px"), default=None)) else 0
    ev_n = safe_nonneg_int(row.get("events"), default=0) or 0
    return (fq_v, has_px, ev_n)


def _merge_closed_summary(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    primary, secondary = (
        (incoming, existing) if _closed_row_richness(incoming) > _closed_row_richness(existing) else (existing, incoming)
    )
    merged = dict(primary)
    for field in ("owner", "side", "type", "kind", "iid", "cid", "vid", "qty", "px", "avg_px", "fee", "fee_asset", "vst", "reason"):
        if not merged.get(field) and secondary.get(field):
            merged[field] = secondary[field]
    return merged


def backfill_order_ledger_from_audit(
    audit_dir: str,
    cachedb_dir: str,
    *,
    dry_run: bool = False,
    now_ts: Optional[float] = None,
) -> dict[str, Any]:
    """Backfill legacy ``execution_audit_*.jsonl`` history into Order Ledger events & closed archives."""
    reader = OrderLedgerReader(cachedb_dir=cachedb_dir)
    now_s = safe_pos_float(now_ts, default=None) or time.time()
    live_day, live_ts = _find_first_live_event_boundary(reader, now_ts=now_s)

    audit_paths = iter_audit_jsonl_paths(audit_dir, max_day=live_day) if live_day else iter_audit_jsonl_paths(audit_dir)
    events_by_day: dict[str, list[dict[str, Any]]] = {}
    day_source_mtime: dict[str, float] = {}
    all_backfill_events: list[dict[str, Any]] = []
    source_rows = 0
    max_ts = now_s + 60.0
    for ap in audit_paths:
        base = os.path.basename(ap)
        fday = base[len("execution_audit_") : -len(".jsonl")] if base.startswith("execution_audit_") and base.endswith(".jsonl") else ""
        try:
            raw_mtime = os.path.getmtime(ap)
            ap_mtime = raw_mtime if (0.0 < raw_mtime <= max_ts) else 0.0
        except OSError:
            ap_mtime = 0.0
        for rec in read_jsonl_dicts(ap):
            source_rows += 1
            ts_r = normalize_epoch_sec(rec.get("ts"), default=None)
            if ts_r is None or ts_r > max_ts:
                continue
            if live_ts > 0.0 and ts_r >= live_ts:
                continue
            if live_day and fday and fday > live_day:
                continue
            if ev := _convert_audit_row_to_ledger_event(rec):
                day_k = utc_day_for_ts(ev["ts"])
                events_by_day.setdefault(day_k, []).append(ev)
                if ap_mtime > 0.0:
                    day_source_mtime[day_k] = max(day_source_mtime.get(day_k, 0.0), ap_mtime)
                all_backfill_events.append(ev)

    all_backfill_events.sort(key=lambda r: safe_pos_float(r.get("ts"), default=0.0) or 0.0)
    for ev_list in events_by_day.values():
        ev_list.sort(key=lambda r: safe_pos_float(r.get("ts"), default=0.0) or 0.0)

    # Materialize deduplicated terminal closed summaries in O(N) via O(1) ID indexes around apply_order_event.
    def _event_id_tokens(ev_row: dict[str, Any]) -> set[str]:
        toks: set[str] = set()
        for f_k in ("iid", "cid", "vid"):
            if clean_t := clean_order_id(ev_row.get(f_k)):
                toks.add(clean_t.lower())
                if f_k in {"iid", "cid"}:
                    toks.update(all_correlatable_client_order_ids(clean_t))
        return toks

    real_order_tokens: set[str] = set()
    for ev in all_backfill_events:
        if ev.get("vid") or ev.get("ev") not in {"SUBMITTED", "SUBMIT_REFUSED"}:
            real_order_tokens.update(_event_id_tokens(ev))

    sim_records: dict[str, dict[str, Any]] = {}
    idx_vid: dict[tuple[str, str, str], str] = {}
    idx_id: dict[tuple[str, str, str], str] = {}
    rec_tokens: dict[str, set[tuple[str, str, str]]] = {}
    sim_closed_raw: list[dict[str, Any]] = []

    def _drop_index(k: str) -> None:
        for ik in rec_tokens.pop(k, ()):
            if idx_id.get(ik) == k:
                idx_id.pop(ik, None)
            if idx_vid.get(ik) == k:
                idx_vid.pop(ik, None)

    def _index_record(k: str, rec: dict[str, Any]) -> None:
        _drop_index(k)
        v_n = str(rec.get("v") or "")
        s_n = str(rec.get("sym") or "")
        added: set[tuple[str, str, str]] = set()
        if rv := clean_order_id(rec.get("vid")):
            vk = (v_n, s_n, rv.lower())
            idx_vid[vk] = k
            added.add(vk)
        for tok in _event_id_tokens({"iid": rec.get("iid"), "cid": rec.get("cid")}):
            ik = (v_n, s_n, tok)
            idx_id[ik] = k
            added.add(ik)
        rec_tokens[k] = added

    for ev in all_backfill_events:
        if not ev.get("vid") and ev.get("ev") in {"SUBMITTED", "SUBMIT_REFUSED"}:
            iid_l = str(ev.get("iid") or "").lower()
            cid_l = str(ev.get("cid") or "").lower()
            if (not iid_l or iid_l not in real_order_tokens) and (not cid_l or cid_l not in real_order_tokens):
                continue
        v_n = str(ev.get("v") or "")
        s_n = str(ev.get("sym") or "")
        matched_keys: set[str] = set()
        if ev_vid := clean_order_id(ev.get("vid")):
            if mk := idx_vid.get((v_n, s_n, ev_vid.lower())):
                matched_keys.add(mk)
        for f_k in ("cid", "iid"):
            if fv := clean_order_id(ev.get(f_k)):
                if mk := idx_id.get((v_n, s_n, fv.lower())):
                    matched_keys.add(mk)
        mini: dict[str, dict[str, Any]] = {mk: sim_records[mk] for mk in matched_keys if mk in sim_records}
        apply_order_event(mini, ev, closed_out=sim_closed_raw)
        for mk in matched_keys:
            if mk not in mini:
                sim_records.pop(mk, None)
                _drop_index(mk)
        for nk, nrec in mini.items():
            sim_records[nk] = nrec
            _index_record(nk, nrec)

    # Archive only real venue-boundary historical orders (with venue order ID or FILLED) into orders_closed_*.jsonl;
    # pre-submit/venue-rejection retry loops remain traceable in order_events_*.jsonl without bloating monthly closed logs.
    dedup_closed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in sim_closed_raw:
        if not (clean_order_id(row.get("vid")) or str(row.get("state") or "").upper() == "FILLED"):
            continue
        dk = _closed_dedup_key(row)
        dedup_closed[dk] = _merge_closed_summary(dedup_closed[dk], row) if dk in dedup_closed else row
    sim_closed = sorted(
        dedup_closed.values(),
        key=lambda r: safe_pos_float(r.get("closed_ts"), default=0.0) or 0.0,
    )

    closed_by_month: dict[str, list[dict[str, Any]]] = {}
    for row in sim_closed:
        ts_c = safe_pos_float(row.get("closed_ts"), default=0.0) or 0.0
        if ts_c > 0.0:
            closed_by_month.setdefault(utc_month_for_ts(ts_c), []).append(row)

    stats: dict[str, Any] = {
        "live_boundary_day": live_day,
        "live_boundary_ts": live_ts,
        "audit_files_scanned": len(audit_paths),
        "audit_rows_scanned": source_rows,
        "events_converted": len(all_backfill_events),
        "event_days_written": 0,
        "boundary_day_prepended": 0,
        "closed_months_written": 0,
        "closed_rows_added": 0,
        "dry_run": bool(dry_run),
    }
    if dry_run:
        stats["event_days_planned"] = len(events_by_day)
        stats["closed_months_planned"] = len(closed_by_month)
        stats["closed_rows_planned"] = len(sim_closed)
        return stats

    active_p = reader.active_path()
    os.makedirs(os.path.dirname(active_p), exist_ok=True)
    with generation_file_lock(active_p):
        snap = reader.read_active_snapshot(repair_files=True)
        cursor = dict(snap.get("journal_cursor") or {})
        offsets = dict(cursor.get("offsets") or {}) if isinstance(cursor.get("offsets"), dict) else {}
        snap_dirty = False

        recent_backfill_days = set(sorted(events_by_day.keys())[-2:])
        for day_k, day_evs in sorted(events_by_day.items()):
            ev_path = reader.events_path(day_k)
            ev_fname = os.path.basename(ev_path)
            existing = read_jsonl_dicts(ev_path) if os.path.isfile(ev_path) else []
            already_backfilled = any(clean_upper_scalar(r.get("src")) == "AUDIT_BACKFILL" for r in existing)
            if already_backfilled:
                continue
            if existing:
                old_raw = read_bytes_file(ev_path)
                if old_raw is None:
                    old_raw = b""
                old_size = len(old_raw)
                prefix_bytes = "".join(
                    encode_json_payload(r, ensure_ascii=False, separators=(",", ":")) + "\n"
                    for r in day_evs
                ).encode("utf-8")
                merged_bytes = prefix_bytes + old_raw
                atomic_write_bytes(ev_path, merged_bytes)
                new_size = len(merged_bytes)
                delta = len(prefix_bytes)
                if ev_fname in offsets or cursor.get("file") == ev_fname:
                    prev_off = safe_nonneg_int(offsets.get(ev_fname), default=old_size)
                    shifted_off = min(new_size, (prev_off if prev_off is not None else old_size) + delta)
                    offsets[ev_fname] = shifted_off
                    if cursor.get("file") == ev_fname:
                        prev_cur = safe_nonneg_int(cursor.get("offset"), default=old_size)
                        cursor["offset"] = min(new_size, (prev_cur if prev_cur is not None else old_size) + delta)
                    snap_dirty = True
                stats["boundary_day_prepended"] += len(day_evs)
                stats["event_days_written"] += 1
            else:
                atomic_write_jsonl(ev_path, day_evs, ensure_ascii=False)
                if (src_mt := day_source_mtime.get(day_k, 0.0)) > 0.0:
                    try:
                        os.utime(ev_path, (src_mt, src_mt))
                    except OSError:
                        pass
                if live_day and day_k in recent_backfill_days:
                    offsets[ev_fname] = os.path.getsize(ev_path)
                    snap_dirty = True
                stats["event_days_written"] += 1

        for month_k, m_rows in sorted(closed_by_month.items()):
            cl_path = reader.closed_path(month_k)
            existing_cl = read_jsonl_dicts(cl_path) if os.path.isfile(cl_path) else []
            seen_keys = {_closed_dedup_key(r) for r in existing_cl}
            new_rows: list[dict[str, Any]] = []
            for row in m_rows:
                dk = _closed_dedup_key(row)
                if dk in seen_keys:
                    continue
                seen_keys.add(dk)
                new_rows.append(row)
            if not new_rows:
                continue
            merged_cl = sorted(
                [*new_rows, *existing_cl],
                key=lambda r: safe_pos_float(r.get("closed_ts"), default=0.0) or 0.0,
            )
            atomic_write_jsonl(cl_path, merged_cl, ensure_ascii=False)
            stats["closed_months_written"] += 1
            stats["closed_rows_added"] += len(new_rows)

        if snap_dirty:
            cursor["offsets"] = offsets
            snap["journal_cursor"] = cursor
            save_json_state(active_p, snap, indent=1, sort_keys=False)

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=_ROOT, help="Workspace root directory (default: repo root).")
    parser.add_argument("--audit-dir", default="", help="Override execution_audit directory.")
    parser.add_argument("--cachedb-dir", default="", help="Override cachedb directory.")
    add_store_true_cli_args(parser, ("--dry-run", "Inspect backfill counts without writing files."))
    args = parser.parse_args()

    abs_root = os.path.abspath(args.root)
    audit_dir = args.audit_dir or os.path.join(abs_root, "logger", "execution_audit")
    cachedb_dir = args.cachedb_dir or os.path.join(abs_root, "cachedb")
    stats = backfill_order_ledger_from_audit(audit_dir, cachedb_dir, dry_run=bool(args.dry_run))
    print(f"[backfill_order_ledger] {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
