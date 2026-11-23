#!/usr/bin/env python3
"""order_ledger_worker.py -- single-writer, network-free reconciler & fleet daemon for the Order Ledger SSoT.

Reconciles local bot/strategy/AI order events (``order_events_YYYY-MM-DD.jsonl``) against
live venue mirrors owned by ``cacheManager`` (``CacheOrderOpenManager`` and
``CacheOrderFilledManager``), converging ``orders_active.json`` toward 1:1 parity with
the venues and archiving terminal orders into ``orders_closed_YYYY-MM.jsonl``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Optional

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from botcore import (
    _parse_bool_flag,
    add_store_true_cli_args,
    clean_order_id,
    clean_upper_scalar,
    extract_order_id,
    extract_order_price,
    extract_order_side,
    extract_timestamp_ms,
    load_dotenv,
    required_float_env,
    resolve_now_ts,
    safe_nonneg_float,
    safe_pos_float,
    single_instance,
    touch_module_heartbeat,
)
from cacheaccount import (
    CacheOrderFilledManager,
    CacheOrderOpenManager,
    OPEN_ORDER_SYNC_INTERVAL_SEC,
    get_order_filled_manager,
    get_order_open_manager,
)
from cachepaths import clean_symbol_key, normalize_venue_name
from instrument_registry import canonical_display_symbol
from providers.order_id_context import (
    all_correlatable_client_order_ids,
    extract_client_order_id,
    resolve_order_owner,
)
from providers.order_ledger import (
    FileOrderLedgerBase,
    ORDER_LEDGER_SCHEMA_VERSION,
    OrderEventType,
    OrderLedgerState,
    TERMINAL_LEDGER_STATES,
    apply_order_event,
    build_order_event,
    make_closed_summary,
    match_active_order_key,
)
from providers.order_ohlc_utils import extract_executed_qty, extract_orig_qty
from state_io import generation_file_lock


def derive_reconcile_grace_sec(open_sync_interval_sec: float = OPEN_ORDER_SYNC_INTERVAL_SEC) -> float:
    """Derive absence-confirmation grace window formulaically from the open-order sync cadence."""
    base = safe_pos_float(open_sync_interval_sec, default=OPEN_ORDER_SYNC_INTERVAL_SEC) or OPEN_ORDER_SYNC_INTERVAL_SEC
    return round(3.0 * base, 3)


def derive_intent_stale_sec(open_sync_interval_sec: float = OPEN_ORDER_SYNC_INTERVAL_SEC) -> float:
    """Derive unsubmitted non-deferred intent expiry window formulaically from the open-order sync cadence."""
    base = safe_pos_float(open_sync_interval_sec, default=OPEN_ORDER_SYNC_INTERVAL_SEC) or OPEN_ORDER_SYNC_INTERVAL_SEC
    return round(10.0 * base, 3)


def resolve_worker_poll_sec() -> float:
    """Load and validate ``ORDER_LEDGER_POLL_SEC`` from ``config.env``."""
    load_dotenv()
    poll_sec = required_float_env("ORDER_LEDGER_POLL_SEC")
    if poll_sec <= 0:
        raise ValueError("ORDER_LEDGER_POLL_SEC must be finite and > 0")
    return poll_sec


class OrderLedgerReconciler(FileOrderLedgerBase):
    """Level 3 single-writer reconciler bridging local order events with venue cache snapshots."""

    def __init__(
        self,
        cachedb_dir: Optional[str] = None,
        *,
        use_singletons: Optional[bool] = None,
        reconcile_grace_sec: Optional[float] = None,
        intent_stale_sec: Optional[float] = None,
    ):
        super().__init__(cachedb_dir=cachedb_dir)
        self._use_singletons = (
            (self._explicit_cachedb_dir is None)
            if use_singletons is None
            else _parse_bool_flag(use_singletons, False)
        )
        self.reconcile_grace_sec = (
            safe_pos_float(reconcile_grace_sec, default=None)
            or derive_reconcile_grace_sec(OPEN_ORDER_SYNC_INTERVAL_SEC)
        )
        self.intent_stale_sec = (
            safe_pos_float(intent_stale_sec, default=None)
            or derive_intent_stale_sec(OPEN_ORDER_SYNC_INTERVAL_SEC)
        )
        self._file_sigs: dict[str, tuple[int, int]] = {}

    def _reload_singleton_if_changed(self, mgr: Any) -> None:
        fname = getattr(mgr, "filename", None)
        if not fname or not os.path.isfile(fname):
            return
        try:
            st = os.stat(fname)
            sig = (st.st_mtime_ns, st.st_size)
        except OSError:
            return
        if self._file_sigs.get(fname) == sig:
            return
        reload_fn = getattr(mgr, "_reload_snapshot_from_disk", None)
        if callable(reload_fn):
            reload_fn()
        self._file_sigs[fname] = sig

    def _read_venue_open_snapshot(
        self, venue: str
    ) -> tuple[dict[str, list[dict[str, Any]]], float]:
        if self._use_singletons and self._explicit_cachedb_dir is None:
            try:
                mgr = get_order_open_manager(venue)
                self._reload_singleton_if_changed(mgr)
                with mgr.lock:
                    items = {k: [dict(r) for r in v] for k, v in (mgr.cache or {}).items() if isinstance(v, list)}
                    ver_ts = safe_nonneg_float(getattr(mgr, "verified_ts", 0.0), default=0.0) or 0.0
                    if ver_ts <= 0.0 and getattr(mgr, "fetchtime_per_symbol", None):
                        max_ft = max(mgr.fetchtime_per_symbol.values(), default=0)
                        if max_ft > 0:
                            ver_ts = round(max_ft / 1000.0, 3)
                if items or ver_ts > 0.0:
                    return items, ver_ts
            except Exception:
                pass
        fname = CacheOrderOpenManager.disk_filename(venue)
        items, _, ver_ts = CacheOrderOpenManager.read_venue_open_orders_snapshot(
            os.path.join(self.cachedb_dir, fname)
        )
        return items, ver_ts

    def _read_venue_filled_snapshot(self, venue: str) -> dict[str, list[dict[str, Any]]]:
        if self._use_singletons and self._explicit_cachedb_dir is None:
            try:
                mgr = get_order_filled_manager(venue)
                self._reload_singleton_if_changed(mgr)
                with mgr.lock:
                    items = {k: [dict(r) for r in v] for k, v in (mgr.cache or {}).items() if isinstance(v, list)}
                if items:
                    return items
            except Exception:
                pass
        fname = CacheOrderFilledManager.disk_filename(venue)
        items, _ = CacheOrderFilledManager.read_venue_filled_orders_snapshot(
            os.path.join(self.cachedb_dir, fname),
            venue=venue,
            allow_legacy=True,
            allow_trade_fallback=False,
        )
        return items

    @staticmethod
    def _index_filled_rows(
        filled_by_venue: dict[str, dict[str, list[dict[str, Any]]]]
    ) -> tuple[
        dict[tuple[str, str, str], dict[str, Any]],
        dict[tuple[str, str], dict[str, Any]],
        dict[tuple[str, str], list[dict[str, Any]]],
    ]:
        by_sym_vid: dict[tuple[str, str, str], dict[str, Any]] = {}
        by_vid_any: dict[tuple[str, str], dict[str, Any]] = {}
        by_cid: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for venue, sym_map in filled_by_venue.items():
            for sym_k, rows in sym_map.items():
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    raw_sym = row.get("symbol") or sym_k
                    sym_norm = canonical_display_symbol(raw_sym, venue=venue) or clean_symbol_key(raw_sym)
                    sym_lc = sym_norm.lower()
                    enriched = {"symbol": sym_norm or raw_sym, **row}
                    if vid := clean_order_id(extract_order_id(enriched, default=None)):
                        if sym_lc:
                            by_sym_vid[(venue, sym_lc, vid.lower())] = enriched
                        by_vid_any[(venue, vid.lower())] = enriched
                    if cid := extract_client_order_id(enriched, default=None):
                        for tok in ({cid.lower()} | all_correlatable_client_order_ids(cid)):
                            by_cid.setdefault((venue, tok), []).append(enriched)
        return by_sym_vid, by_vid_any, by_cid

    @staticmethod
    def _find_matching_fill(
        rec: dict[str, Any],
        by_sym_vid: dict[tuple[str, str, str], dict[str, Any]],
        by_vid_any: dict[tuple[str, str], dict[str, Any]],
        by_cid: dict[tuple[str, str], list[dict[str, Any]]],
    ) -> Optional[dict[str, Any]]:
        venue = normalize_venue_name(rec.get("v")) or str(rec.get("v") or "").lower()
        sym_norm = canonical_display_symbol(rec.get("sym"), venue=venue) or clean_symbol_key(rec.get("sym") or "")
        sym_lc = sym_norm.lower()
        created_ts = safe_pos_float(rec.get("created_ts"), default=0.0) or 0.0
        if vid := clean_order_id(rec.get("vid")):
            if sym_lc:
                if match := by_sym_vid.get((venue, sym_lc, vid.lower())):
                    return match
            elif match := by_vid_any.get((venue, vid.lower())):
                return match
        cands: list[str] = []
        if cid := clean_order_id(rec.get("cid")):
            cands.append(cid)
        if iid := clean_order_id(rec.get("iid")):
            cands.append(iid)
        for cand in cands:
            for tok in ({cand.lower()} | all_correlatable_client_order_ids(cand)):
                for row in by_cid.get((venue, tok), ()):
                    r_sym_lc = (canonical_display_symbol(row.get("symbol"), venue=venue) or "").lower()
                    if sym_lc and r_sym_lc and sym_lc != r_sym_lc:
                        continue
                    ts_ms = extract_timestamp_ms(row, default=0) or 0
                    row_ts = ts_ms / 1000.0 if ts_ms > 0 else 0.0
                    if created_ts > 0.0 and row_ts > 0.0 and row_ts < (created_ts - 60.0):
                        continue
                    return row
        return None

    def reconcile_once(self, *, now_ts: Optional[float] = None) -> dict[str, int]:
        """Execute one locked, network-free reconciliation pass and persist updated state."""
        eff_now = resolve_now_ts(now_ts)
        active_file = self.active_path()
        stats = {
            "events_applied": 0,
            "active_count": 0,
            "reconciled_open": 0,
            "reconciled_filled": 0,
            "reconciled_closed": 0,
            "orphans_adopted": 0,
        }

        with generation_file_lock(active_file):
            snap = self.read_active_snapshot(repair_files=True)
            orders: dict[str, dict[str, Any]] = snap["orders"]
            cursor: dict[str, Any] = snap["journal_cursor"]
            venue_verified_ts: dict[str, float] = dict(snap["venue_verified_ts"])
            newly_closed: list[dict[str, Any]] = []
            reconciler_events: list[dict[str, Any]] = []

            # 1. Replay all new producer events strictly after journal_cursor.
            new_events, cursor = self.read_events_after_cursor(cursor)
            for ev in new_events:
                if clean_upper_scalar(ev.get("src")) == "RECONCILER":
                    continue
                apply_order_event(orders, ev, closed_out=newly_closed)
                stats["events_applied"] += 1

            # 2. Read live open and filled snapshots across all registered venues (100% network-free).
            open_by_venue: dict[str, dict[str, list[dict[str, Any]]]] = {}
            filled_by_venue: dict[str, dict[str, list[dict[str, Any]]]] = {}
            for venue_key in CacheOrderOpenManager.VENUE_FILENAMES:
                oo_items, ver_ts = self._read_venue_open_snapshot(venue_key)
                open_by_venue[venue_key] = oo_items
                if ver_ts > 0.0:
                    venue_verified_ts[venue_key] = max(venue_verified_ts.get(venue_key, 0.0), ver_ts)
                filled_by_venue[venue_key] = self._read_venue_filled_snapshot(venue_key)

            # 3. Reconcile resting venue open orders against active ledger records.
            seen_open_keys: set[str] = set()
            for venue_key, sym_map in open_by_venue.items():
                v_ver_ts = venue_verified_ts.get(venue_key, 0.0) or eff_now
                for sym_k, rows in sym_map.items():
                    for oo in rows:
                        if not isinstance(oo, dict):
                            continue
                        oid = clean_order_id(extract_order_id(oo, default=None))
                        if not oid:
                            continue
                        cid = extract_client_order_id(oo, default=None)
                        sym_norm = canonical_display_symbol(oo.get("symbol") or sym_k, venue=venue_key) or clean_symbol_key(oo.get("symbol") or sym_k)
                        exec_q = safe_nonneg_float(extract_executed_qty(oo, default=0.0), default=0.0) or 0.0
                        orig_q = safe_pos_float(extract_orig_qty(oo, default=None), default=None)
                        oo_px = safe_pos_float(extract_order_price(oo, default=None), default=None)
                        oo_side = extract_order_side(oo)
                        oo_type = clean_upper_scalar(oo.get("type"))
                        oo_vst = clean_upper_scalar(oo.get("status")) or "OPEN"

                        matched_key = match_active_order_key(orders, venue_key, symbol=sym_norm, cid=cid, vid=oid)
                        if matched_key is not None:
                            seen_open_keys.add(matched_key)
                            rec = orders[matched_key]
                            prev_state = str(rec.get("state") or "")
                            prev_fq = safe_nonneg_float(rec.get("fq"), default=0.0) or 0.0
                            prev_vid = clean_order_id(rec.get("vid"))
                            target_ev = (
                                OrderEventType.RECONCILED_PARTIAL.value
                                if exec_q > 0.0
                                else OrderEventType.RECONCILED_OPEN.value
                            )
                            target_state = (
                                prev_state
                                if prev_state == OrderLedgerState.CANCELING.value
                                else (OrderLedgerState.PARTIAL.value if exec_q > 0.0 else OrderLedgerState.OPEN.value)
                            )
                            rec["last_seen_open_ts"] = round(v_ver_ts, 3)
                            if (
                                prev_state != target_state
                                or prev_vid != oid
                                or exec_q > prev_fq + 1e-12
                            ):
                                rec_ev = build_order_event(
                                    target_ev,
                                    venue=venue_key,
                                    symbol=sym_norm or rec.get("sym"),
                                    ts=eff_now,
                                    owner=rec.get("owner"),
                                    side=oo_side or rec.get("side"),
                                    order_type=oo_type or rec.get("type"),
                                    qty=orig_q or rec.get("qty"),
                                    px=oo_px or rec.get("px"),
                                    fq=exec_q if exec_q > 0.0 else None,
                                    iid=rec.get("iid"),
                                    cid=cid or rec.get("cid"),
                                    vid=oid,
                                    vst=oo_vst,
                                    src="reconciler",
                                )
                                apply_order_event(orders, rec_ev, closed_out=newly_closed)
                                reconciler_events.append(rec_ev)
                                stats["reconciled_open"] += 1
                        else:
                            inferred_owner = (resolve_order_owner(cid) if cid else None)
                            if not inferred_owner or inferred_owner == "unspecified":
                                inferred_owner = "external"
                            adopt_ev = build_order_event(
                                OrderEventType.ORPHAN_ADOPTED.value,
                                venue=venue_key,
                                symbol=sym_norm,
                                ts=eff_now,
                                owner=inferred_owner,
                                side=oo_side,
                                order_type=oo_type or "LIMIT",
                                qty=orig_q,
                                px=oo_px,
                                fq=exec_q if exec_q > 0.0 else None,
                                cid=cid,
                                vid=oid,
                                vst=oo_vst,
                                src="reconciler",
                            )
                            adopted_key = apply_order_event(orders, adopt_ev, closed_out=newly_closed)
                            if adopted_key and adopted_key in orders:
                                orders[adopted_key]["last_seen_open_ts"] = round(v_ver_ts, 3)
                                seen_open_keys.add(adopted_key)
                            reconciler_events.append(adopt_ev)
                            stats["orphans_adopted"] += 1

            # 4. Reconcile active orders against CacheOrderFilledManager (strictly venue+symbol scoped).
            filled_by_sym_vid, filled_by_vid_any, filled_by_cid = self._index_filled_rows(filled_by_venue)
            for key in list(orders.keys()):
                rec = orders.get(key)
                if not isinstance(rec, dict):
                    continue
                fill_row = self._find_matching_fill(rec, filled_by_sym_vid, filled_by_vid_any, filled_by_cid)
                if fill_row is None:
                    continue
                fill_vid = clean_order_id(extract_order_id(fill_row, default=None))
                fill_cid = extract_client_order_id(fill_row, default=None)
                fill_qty = safe_pos_float(fill_row.get("quantity") or fill_row.get("qty"), default=0.0) or 0.0
                fill_px = safe_pos_float(fill_row.get("price"), default=None)
                fill_fee = safe_nonneg_float(fill_row.get("fee"), default=None)
                fill_fee_asset = clean_upper_scalar(fill_row.get("feeAsset") or fill_row.get("fee_asset"))
                fill_vst = clean_upper_scalar(fill_row.get("status"))
                req_qty = safe_pos_float(rec.get("qty"), default=None)
                resting_now = key in seen_open_keys
                is_complete = (
                    (not resting_now)
                    or fill_vst == "FILLED"
                    or (req_qty is not None and fill_qty >= req_qty * 0.9999)
                )
                if is_complete:
                    fill_ev = build_order_event(
                        OrderEventType.RECONCILED_FILLED.value,
                        venue=rec.get("v"),
                        symbol=rec.get("sym") or fill_row.get("symbol"),
                        ts=eff_now,
                        owner=rec.get("owner"),
                        side=rec.get("side") or extract_order_side(fill_row),
                        order_type=rec.get("type") or fill_row.get("type"),
                        qty=req_qty or fill_qty,
                        px=rec.get("px") or fill_px,
                        fq=fill_qty,
                        avg_px=fill_px,
                        fee=fill_fee,
                        fee_asset=fill_fee_asset,
                        iid=rec.get("iid"),
                        cid=fill_cid or rec.get("cid"),
                        vid=fill_vid or rec.get("vid"),
                        vst=fill_vst or "FILLED",
                        src="reconciler",
                    )
                    apply_order_event(orders, fill_ev, closed_out=newly_closed)
                    reconciler_events.append(fill_ev)
                    stats["reconciled_filled"] += 1
                else:
                    prev_fq = safe_nonneg_float(rec.get("fq"), default=0.0) or 0.0
                    if fill_qty > prev_fq + 1e-12:
                        part_ev = build_order_event(
                            OrderEventType.RECONCILED_PARTIAL.value,
                            venue=rec.get("v"),
                            symbol=rec.get("sym") or fill_row.get("symbol"),
                            ts=eff_now,
                            owner=rec.get("owner"),
                            side=rec.get("side"),
                            fq=fill_qty,
                            avg_px=fill_px,
                            fee=fill_fee,
                            fee_asset=fill_fee_asset,
                            iid=rec.get("iid"),
                            cid=fill_cid or rec.get("cid"),
                            vid=fill_vid or rec.get("vid"),
                            vst=fill_vst or "PARTIALLY_FILLED",
                            src="reconciler",
                        )
                        apply_order_event(orders, part_ev, closed_out=newly_closed)
                        reconciler_events.append(part_ev)

            # 5. Converge orders absent from freshly-verified venue open-order snapshots after grace window.
            for key in list(orders.keys()):
                rec = orders.get(key)
                if not isinstance(rec, dict) or key in seen_open_keys:
                    continue
                venue_key = normalize_venue_name(rec.get("v")) or str(rec.get("v") or "").lower()
                state = str(rec.get("state") or OrderLedgerState.INTENT.value)
                upd_ts = safe_pos_float(rec.get("updated_ts"), default=0.0) or 0.0
                last_open_ts = safe_pos_float(rec.get("last_seen_open_ts"), default=0.0) or 0.0
                ref_ts = max(upd_ts, last_open_ts)
                v_ver_ts = venue_verified_ts.get(venue_key, 0.0)

                close_ev_type: Optional[str] = None
                close_reason: Optional[str] = None
                if state in {
                    OrderLedgerState.OPEN.value,
                    OrderLedgerState.PARTIAL.value,
                    OrderLedgerState.CANCELING.value,
                }:
                    if v_ver_ts > 0.0 and (v_ver_ts - ref_ts) >= self.reconcile_grace_sec:
                        cur_fq = safe_nonneg_float(rec.get("fq"), default=0.0) or 0.0
                        req_qty = safe_pos_float(rec.get("qty"), default=None)
                        if cur_fq > 0.0 and req_qty is not None and cur_fq >= req_qty * 0.9999:
                            close_ev_type = OrderEventType.RECONCILED_FILLED.value
                            close_reason = "filled_and_absent_from_open"
                        else:
                            close_ev_type = OrderEventType.RECONCILED_CANCELED.value
                            close_reason = "absent_from_verified_open_book"
                elif state in {
                    OrderLedgerState.SUBMITTED.value,
                    OrderLedgerState.SUBMIT_UNKNOWN.value,
                }:
                    if v_ver_ts > 0.0 and (v_ver_ts - ref_ts) >= self.reconcile_grace_sec:
                        close_ev_type = OrderEventType.RECONCILED_FAILED.value
                        close_reason = "absent_after_verified_grace"
                elif state == OrderLedgerState.INTENT.value and not _parse_bool_flag(rec.get("deferred"), False):
                    if ref_ts > 0.0 and (eff_now - ref_ts) >= self.intent_stale_sec:
                        close_ev_type = OrderEventType.RECONCILED_EXPIRED.value
                        close_reason = "intent_stale"

                if close_ev_type:
                    c_ev = build_order_event(
                        close_ev_type,
                        venue=venue_key,
                        symbol=rec.get("sym"),
                        ts=eff_now,
                        owner=rec.get("owner"),
                        side=rec.get("side"),
                        iid=rec.get("iid"),
                        cid=rec.get("cid"),
                        vid=rec.get("vid"),
                        reason=close_reason,
                        src="reconciler",
                    )
                    apply_order_event(orders, c_ev, closed_out=newly_closed)
                    reconciler_events.append(c_ev)

            # 6. Persist reconciler transition events into the daily journal (skipped on replay via src="reconciler").
            for rev in reconciler_events:
                self.append_event(rev)

            # 7. Archive newly terminal orders to orders_closed_YYYY-MM.jsonl.
            for summary in newly_closed:
                self.append_closed_record(summary)
                stats["reconciled_closed"] += 1

            # 8. Ensure no terminal residue remains in orders_active.json and publish atomically.
            for k_rem in [k for k, v in orders.items() if str(v.get("state") or "") in TERMINAL_LEDGER_STATES]:
                rem = orders.pop(k_rem)
                self.append_closed_record(make_closed_summary(k_rem, rem, closed_ts=eff_now))
                stats["reconciled_closed"] += 1

            stats["active_count"] = len(orders)
            out_snap = {
                "schema_version": ORDER_LEDGER_SCHEMA_VERSION,
                "updated_ts": round(eff_now, 3),
                "journal_cursor": cursor,
                "venue_verified_ts": {k: round(v, 3) for k, v in sorted(venue_verified_ts.items())},
                "orders": orders,
            }
            self.write_active_snapshot(out_snap)

        return stats


def _touch_ledger_hb(status: dict, *, poll_sec: float, force: bool = False) -> None:
    touch_module_heartbeat(
        "order_ledger",
        interval_sec=30.0,
        force=force,
        status=status,
        intervals={"poll": poll_sec},
    )


def main() -> int:
    import log  # noqa: F401 -- ensure stdout timestamps and daily logger file from startup

    poll_sec = resolve_worker_poll_sec()
    parser = add_store_true_cli_args(
        argparse.ArgumentParser(description=__doc__),
        ("--once", "Run a single reconciliation pass and exit."),
    )
    args = parser.parse_args()
    reconciler = OrderLedgerReconciler()

    if args.once:
        stats = reconciler.reconcile_once()
        print(f"[order_ledger] single pass: {stats}")
        return 0

    single_instance("order_ledger_worker")
    print(
        f"[order_ledger] start (poll={poll_sec:g}s, "
        f"grace={reconciler.reconcile_grace_sec:g}s, intent_stale={reconciler.intent_stale_sec:g}s)"
    )
    _touch_ledger_hb({"active_count": 0}, poll_sec=poll_sec, force=True)
    while True:
        last_stats = None
        try:
            stats = reconciler.reconcile_once()
            last_stats = stats
            if (
                stats["events_applied"]
                or stats["reconciled_open"]
                or stats["reconciled_filled"]
                or stats["reconciled_closed"]
                or stats["orphans_adopted"]
            ):
                print(f"[order_ledger] {stats}")
        except Exception as exc:  # noqa: BLE001
            print(f"[order_ledger] pass error (continuing): {exc}")
        _touch_ledger_hb(last_stats if isinstance(last_stats, dict) else {"status": "ok"}, poll_sec=poll_sec)
        time.sleep(poll_sec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
