#!/usr/bin/env python3
"""order_ledger_reconciler_worker.py -- single-writer, network-free reconciler & fleet daemon for the Order Ledger SSoT.

Reconciles local bot/strategy/AI order events (``order_events_YYYY-MM-DD.jsonl``) against live venue
mirrors owned by ``cacheManager`` (``CacheOrderOpenManager`` and ``CacheOrderFilledManager``), converging
``orders_active.json`` toward 1:1 parity with venues and archiving terminal orders to ``orders_closed_YYYY-MM.jsonl``.
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
    safe_nonneg_int,
    safe_pos_float,
    single_instance,
    touch_module_heartbeat,
)
from cacheaccount import (
    CacheOrderFilledManager,
    CacheOrderOpenManager,
    OPEN_ORDER_SYNC_INTERVAL_SEC,
    UnifiedOrderBookEvidence,
    get_order_filled_manager,
    get_order_open_manager,
)
from cachepaths import (
    clean_symbol_key,
    normalize_venue_name,
    resolve_cachedb_filepath,
)
from instrument_registry import canonical_display_symbol
from providers.intent_state_store import (
    load_known_ai_order_id_to_intent_map_from_cachedb,
    load_known_ai_order_ids_from_cachedb,
)
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
    classify_fill_evidence_state,
    closed_order_tokens,
    make_closed_summary,
    match_active_order_key,
    replay_producer_events,
    update_recently_closed,
)
from providers.order_ohlc_utils import extract_executed_qty, extract_orig_qty
from state_io import encode_json_payload, generation_file_lock
from state_stores.order_retry_queue import OrderRetryQueueStore


def derive_reconcile_grace_sec(open_sync_interval_sec: float = OPEN_ORDER_SYNC_INTERVAL_SEC) -> float:
    """Derive absence-confirmation grace window formulaically from the open-order sync cadence."""
    base = safe_pos_float(open_sync_interval_sec, default=OPEN_ORDER_SYNC_INTERVAL_SEC) or OPEN_ORDER_SYNC_INTERVAL_SEC
    return round(3.0 * base, 3)


def derive_intent_stale_sec(open_sync_interval_sec: float = OPEN_ORDER_SYNC_INTERVAL_SEC) -> float:
    """Derive unsubmitted non-deferred intent expiry window formulaically from the open-order sync cadence."""
    base = safe_pos_float(open_sync_interval_sec, default=OPEN_ORDER_SYNC_INTERVAL_SEC) or OPEN_ORDER_SYNC_INTERVAL_SEC
    return round(10.0 * base, 3)


DEFAULT_DEFERRED_INTENT_STALE_SEC = 72.0 * 3600.0


def resolve_worker_poll_sec() -> float:
    """Load and validate ``ORDER_LEDGER_POLL_SEC`` from ``config.env``."""
    load_dotenv()
    poll_sec = required_float_env("ORDER_LEDGER_POLL_SEC")
    if poll_sec <= 0:
        raise ValueError("ORDER_LEDGER_POLL_SEC must be finite and > 0")
    return poll_sec


def _norm_sym_lc(raw_sym: Any, venue: str) -> tuple[str, str]:
    sym_norm = canonical_display_symbol(raw_sym, venue=venue) or clean_symbol_key(raw_sym or "")
    return sym_norm, sym_norm.lower()


class OrderLedgerReconciler(FileOrderLedgerBase):
    """Level 3 single-writer reconciler bridging local order events with venue cache snapshots."""

    def __init__(
        self,
        cachedb_dir: Optional[str] = None,
        *,
        use_singletons: Optional[bool] = None,
        reconcile_grace_sec: Optional[float] = None,
        intent_stale_sec: Optional[float] = None,
        deferred_intent_stale_sec: Optional[float] = None,
    ):
        super().__init__(cachedb_dir=cachedb_dir)
        self._use_singletons = (
            (self._explicit_cachedb_dir is None) if use_singletons is None else _parse_bool_flag(use_singletons, False)
        )
        self.reconcile_grace_sec = safe_pos_float(reconcile_grace_sec, default=None) or derive_reconcile_grace_sec()
        self.intent_stale_sec = safe_pos_float(intent_stale_sec, default=None) or derive_intent_stale_sec()
        self.deferred_intent_stale_sec = max(
            self.intent_stale_sec,
            safe_pos_float(deferred_intent_stale_sec, default=None) or DEFAULT_DEFERRED_INTENT_STALE_SEC,
        )
        self._file_sigs: dict[str, tuple[int, int]] = {}

    @staticmethod
    def _update_recently_closed(
        recently_closed: dict[str, float],
        summaries: list[dict[str, Any]],
        eff_now: float,
        *,
        ttl_sec: float = 21600.0,
        max_entries: int = 512,
    ) -> dict[str, float]:
        return update_recently_closed(
            recently_closed,
            summaries,
            eff_now,
            ttl_sec=ttl_sec,
            max_entries=max_entries,
            prune=True,
        )

    def _reload_singleton_if_changed(self, mgr: Any) -> bool:
        fname = getattr(mgr, "filename", None)
        if not fname or not os.path.isfile(fname):
            return True
        try:
            st = os.stat(fname)
            sig = (st.st_mtime_ns, st.st_size)
        except OSError:
            return False
        if self._file_sigs.get(fname) != sig:
            if callable(reload_fn := getattr(mgr, "_reload_snapshot_from_disk", None)):
                try:
                    ok = reload_fn()
                except Exception:
                    return False
                if ok is False:
                    return False
            self._file_sigs[fname] = sig
        return True

    def _read_venue_open_snapshot(self, venue: str) -> tuple[dict[str, list[dict[str, Any]]], float, dict[str, float]]:
        """Read ``(items_by_symbol, venue_verified_ts, symbol_verified_ts)`` for one venue.

        ``symbol_verified_ts`` contains ONLY timestamps of complete open-order snapshots covering each
        explicit symbol (never single-order WebSocket deltas and never omitted symbols).
        """
        if self._use_singletons and self._explicit_cachedb_dir is None:
            try:
                mgr = get_order_open_manager(venue)
                if self._reload_singleton_if_changed(mgr):
                    with mgr.lock:
                        items = {k: [dict(r) for r in v] for k, v in (mgr.cache or {}).items() if isinstance(v, list)}
                        ver_ts = safe_nonneg_float(getattr(mgr, "verified_ts", 0.0), default=0.0) or 0.0
                        raw_sym_ver = getattr(mgr, "symbol_verified_ts", None)
                        sym_ver = (
                            {
                                str(k): round(v_f, 3)
                                for k, v in raw_sym_ver.items()
                                if (v_f := safe_pos_float(v, default=None)) is not None
                            }
                            if isinstance(raw_sym_ver, dict)
                            else {}
                        )
                    if items or ver_ts > 0.0 or sym_ver:
                        return items, ver_ts, sym_ver
            except Exception:
                pass
        items, _, ver_ts, sym_ver = CacheOrderOpenManager.read_venue_open_orders_snapshot_detailed(
            os.path.join(self.cachedb_dir, CacheOrderOpenManager.disk_filename(venue))
        )
        return items, ver_ts, sym_ver

    def _read_venue_filled_snapshot(self, venue: str) -> dict[str, list[dict[str, Any]]]:
        if self._use_singletons and self._explicit_cachedb_dir is None:
            try:
                mgr = get_order_filled_manager(venue)
                if self._reload_singleton_if_changed(mgr):
                    with mgr.lock:
                        items = {k: [dict(r) for r in v] for k, v in (mgr.cache or {}).items() if isinstance(v, list)}
                    if items:
                        return items
            except Exception:
                pass
        items, _ = CacheOrderFilledManager.read_venue_filled_orders_snapshot(
            os.path.join(self.cachedb_dir, CacheOrderFilledManager.disk_filename(venue)),
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
                    sym_norm, sym_lc = _norm_sym_lc(row.get("symbol") or sym_k, venue)
                    enriched = {"symbol": sym_norm or sym_k, **row}
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
        _, sym_lc = _norm_sym_lc(rec.get("sym"), venue)
        created_ts = safe_pos_float(rec.get("created_ts"), default=0.0) or 0.0
        q_vid = clean_order_id(rec.get("vid"))
        if q_vid:
            if sym_lc:
                if match := by_sym_vid.get((venue, sym_lc, q_vid.lower())):
                    return match
            elif match := by_vid_any.get((venue, q_vid.lower())):
                return match
        for cand in (clean_order_id(rec.get("cid")), clean_order_id(rec.get("iid"))):
            if not cand:
                continue
            for tok in ({cand.lower()} | all_correlatable_client_order_ids(cand)):
                for row in by_cid.get((venue, tok), ()):
                    r_vid = clean_order_id(extract_order_id(row, default=None))
                    if q_vid and r_vid and q_vid.lower() != r_vid.lower():
                        continue
                    _, r_sym_lc = _norm_sym_lc(row.get("symbol"), venue)
                    if sym_lc and r_sym_lc and sym_lc != r_sym_lc:
                        continue
                    ts_ms = extract_timestamp_ms(row, default=0) or 0
                    row_ts = ts_ms / 1000.0 if ts_ms > 0 else 0.0
                    if created_ts > 0.0 and row_ts > 0.0 and row_ts < (created_ts - 60.0):
                        continue
                    return row
        return None

    @staticmethod
    def _classify_fill_outcome(
        fill_row: dict[str, Any],
        *,
        req_qty: Optional[float],
        resting_now: bool,
        order_created_ts: float = 0.0,
        snapshot_verified_ts: float = 0.0,
    ) -> str:
        """Delegate fill-cache classification to the centralized ``providers.order_ledger`` SSoT rule."""
        return classify_fill_evidence_state(
            fill_row,
            req_qty=req_qty,
            resting_now=resting_now,
            order_created_ts=order_created_ts,
            snapshot_verified_ts=snapshot_verified_ts,
        )

    def _is_fill_complete(
        self,
        fill_row: dict[str, Any],
        *,
        req_qty: Optional[float],
        resting_now: bool,
        order_created_ts: float = 0.0,
        snapshot_verified_ts: float = 0.0,
    ) -> bool:
        return (
            self._classify_fill_outcome(
                fill_row,
                req_qty=req_qty,
                resting_now=resting_now,
                order_created_ts=order_created_ts,
                snapshot_verified_ts=snapshot_verified_ts,
            )
            == OrderLedgerState.FILLED.value
        )

    _TERMINAL_OUTBOX_LIFECYCLES = frozenset({
        "succeeded",
        "abandoned",
        "canceled",
        "cancelled",
        "filled",
        "expired",
        "rejected",
        "terminal",
        "failed",
    })

    def _load_known_ai_order_ids_lc(self) -> set[str]:
        """Load lowercase venue order IDs and derived 128-bit cloids via ``providers.intent_state`` SSOT."""
        return load_known_ai_order_ids_from_cachedb(self.cachedb_dir)

    def _load_active_outbox_iids_lc(self) -> Optional[set[str]]:
        """Return lowercase ``outbox-<id>`` / ``intent_id`` tokens active in ``order_retry_queue.jsonl``, or ``None`` if absent."""
        rows = OrderRetryQueueStore.read_records(self.cachedb_dir)
        if rows is None:
            return None
        out: set[str] = set()
        for row in rows:
            lc_state = str(row.get("lifecycle") or row.get("status") or "").strip().lower()
            if lc_state in self._TERMINAL_OUTBOX_LIFECYCLES:
                continue
            if rid := clean_order_id(row.get("id")):
                rid_lc = rid.lower()
                out.add(rid_lc)
                out.add(f"outbox-{rid_lc}")
            if riid := clean_order_id(row.get("intent_id")):
                out.add(riid.lower())
        return out

    def reconcile_once(self, *, now_ts: Optional[float] = None) -> dict[str, int]:
        """Execute one locked, network-free reconciliation pass and persist updated state."""
        eff_now = resolve_now_ts(now_ts)
        stats = {"events_applied": 0, "active_count": 0, "reconciled_open": 0, "reconciled_filled": 0, "reconciled_closed": 0, "orphans_adopted": 0}

        with generation_file_lock(self.active_path()):
            snap = self.read_active_snapshot(repair_files=True)
            orders: dict[str, dict[str, Any]] = snap["orders"]
            cursor: dict[str, Any] = snap["journal_cursor"]
            venue_verified_ts: dict[str, float] = dict(snap["venue_verified_ts"])
            symbol_verified_ts: dict[str, float] = dict(snap.get("symbol_verified_ts") or {})
            recently_closed: dict[str, float] = dict(snap.get("recently_closed") or {})
            active_file_exists = (
                os.path.isfile(str(resolve_cachedb_filepath(self.active_path())))
                and (safe_pos_float(snap.get("updated_ts"), default=0.0) or 0.0) > 0.0
            )
            prev_core_sig = (
                encode_json_payload(
                    {
                        "schema_version": snap["schema_version"],
                        "journal_cursor": cursor,
                        "venue_verified_ts": {k: round(v, 3) for k, v in sorted(venue_verified_ts.items())},
                        "symbol_verified_ts": {k: round(v, 3) for k, v in sorted(symbol_verified_ts.items())},
                        "recently_closed": recently_closed,
                        "orders": orders,
                    },
                    sort_keys=True,
                )
                if active_file_exists
                else None
            )
            newly_closed: list[dict[str, Any]] = []
            reconciler_events: list[dict[str, Any]] = []

            def _emit(ev_type: str, **kw: Any) -> Optional[str]:
                ev_obj = build_order_event(ev_type, ts=eff_now, src="reconciler", **kw)
                reconciler_events.append(ev_obj)
                return apply_order_event(orders, ev_obj, closed_out=newly_closed, recently_closed=recently_closed)

            # Snapshot active retry-outbox tokens before replaying producer events so any order
            # removed from order_retry_queue.jsonl after submission already has its SUBMITTED event
            # visible in the journal pass below.
            active_outbox_iids = self._load_active_outbox_iids_lc()

            # 1. Replay producer events via the canonical shared helper (identical to OrderLedgerReader).
            new_events, cursor = self.read_events_after_cursor(cursor)
            applied_n, recently_closed = replay_producer_events(
                orders,
                new_events,
                recently_closed=recently_closed,
                closed_out=newly_closed,
                eff_now=eff_now,
                prune_recently_closed=True,
            )
            stats["events_applied"] += applied_n

            # 2. Read live open and filled snapshots across all venues (100% network-free) and index fills first.
            open_by_venue: dict[str, dict[str, list[dict[str, Any]]]] = {}
            filled_by_venue: dict[str, dict[str, list[dict[str, Any]]]] = {}
            current_sym_ver: dict[tuple[str, str], float] = {}
            for venue_key in CacheOrderOpenManager.VENUE_FILENAMES:
                oo_items, ver_ts, sym_ver_map = self._read_venue_open_snapshot(venue_key)
                open_by_venue[venue_key] = oo_items
                if ver_ts > 0.0:
                    venue_verified_ts[venue_key] = max(venue_verified_ts.get(venue_key, 0.0), ver_ts)
                for sym_k, s_ts in sym_ver_map.items():
                    _, sym_lc = _norm_sym_lc(sym_k, venue_key)
                    if sym_lc and s_ts > 0.0:
                        current_sym_ver[(venue_key, sym_lc)] = max(current_sym_ver.get((venue_key, sym_lc), 0.0), s_ts)
                        sk = f"{venue_key}:{sym_lc}"
                        symbol_verified_ts[sk] = max(symbol_verified_ts.get(sk, 0.0), s_ts)
                filled_by_venue[venue_key] = self._read_venue_filled_snapshot(venue_key)
            filled_by_sym_vid, filled_by_vid_any, filled_by_cid = self._index_filled_rows(filled_by_venue)

            # 3. Reconcile resting venue open orders against active ledger records.
            known_ai_ids_lc: Optional[set[str]] = None
            known_ai_iid_map: Optional[dict[str, str]] = None
            seen_open_keys: set[str] = set()
            for venue_key, sym_map in open_by_venue.items():
                for sym_k, rows in sym_map.items():
                    for oo in rows:
                        if not isinstance(oo, dict) or not (oid := clean_order_id(extract_order_id(oo, default=None))):
                            continue
                        cid = extract_client_order_id(oo, default=None)
                        sym_norm, sym_lc = _norm_sym_lc(oo.get("symbol") or sym_k, venue_key)
                        seen_ts = (
                            current_sym_ver.get((venue_key, sym_lc))
                            or symbol_verified_ts.get(f"{venue_key}:{sym_lc}", 0.0)
                            or eff_now
                        )
                        exec_q = safe_nonneg_float(extract_executed_qty(oo, default=0.0), default=0.0) or 0.0
                        orig_q = safe_pos_float(extract_orig_qty(oo, default=None), default=None)
                        oo_px = safe_pos_float(extract_order_price(oo, default=None), default=None)
                        oo_side, oo_type, oo_vst = extract_order_side(oo), clean_upper_scalar(oo.get("type")), clean_upper_scalar(oo.get("status")) or "OPEN"

                        matched_key = match_active_order_key(orders, venue_key, symbol=sym_norm, cid=cid, vid=oid)
                        if matched_key is not None:
                            seen_open_keys.add(matched_key)
                            rec = orders[matched_key]
                            prev_state = str(rec.get("state") or "")
                            prev_fq = safe_nonneg_float(rec.get("fq"), default=0.0) or 0.0
                            target_ev = OrderEventType.RECONCILED_PARTIAL.value if exec_q > 0.0 else OrderEventType.RECONCILED_OPEN.value
                            target_state = prev_state if prev_state == OrderLedgerState.CANCELING.value else (OrderLedgerState.PARTIAL.value if exec_q > 0.0 else OrderLedgerState.OPEN.value)
                            rec["last_seen_open_ts"] = round(seen_ts, 3)
                            if prev_state != target_state or clean_order_id(rec.get("vid")) != oid or exec_q > prev_fq + 1e-12:
                                _emit(
                                    target_ev,
                                    venue=venue_key,
                                    symbol=sym_norm or rec.get("sym"),
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
                                )
                                stats["reconciled_open"] += 1
                        else:
                            # Skip re-adopting stale open-cache entries that already completed or just closed (vid or cid/cloid).
                            if any(
                                tok in recently_closed
                                for tok in closed_order_tokens(venue_key, symbol=sym_norm, vid=oid, cid=cid)
                            ):
                                continue
                            prior_fill = self._find_matching_fill(
                                {"v": venue_key, "sym": sym_norm, "vid": oid, "cid": cid},
                                filled_by_sym_vid,
                                filled_by_vid_any,
                                filled_by_cid,
                            )
                            if prior_fill is not None and (
                                self._classify_fill_outcome(prior_fill, req_qty=orig_q, resting_now=True)
                                in TERMINAL_LEDGER_STATES
                            ):
                                continue
                            raw_owner = resolve_order_owner(cid) if cid else None
                            inferred_owner = raw_owner if (raw_owner and raw_owner not in ("unspecified", "unknown") and not raw_owner.startswith("alt:")) else "external"
                            if inferred_owner == "external":
                                if known_ai_ids_lc is None:
                                    known_ai_ids_lc = self._load_known_ai_order_ids_lc()
                                cand_ids = {oid.lower()}
                                if cid:
                                    cand_ids.update(tok.lower() for tok in all_correlatable_client_order_ids(cid))
                                if known_ai_ids_lc and (cand_ids & known_ai_ids_lc):
                                    inferred_owner = "ai_executor"
                            adopted_iid: Optional[str] = None
                            if inferred_owner == "ai_executor":
                                if known_ai_iid_map is None:
                                    known_ai_iid_map = load_known_ai_order_id_to_intent_map_from_cachedb(self.cachedb_dir)
                                cand_tokens = [oid.lower()]
                                if cid:
                                    cand_tokens.append(cid.lower())
                                    cand_tokens.extend(tok.lower() for tok in all_correlatable_client_order_ids(cid))
                                for tok in cand_tokens:
                                    if matched_iid := known_ai_iid_map.get(tok):
                                        adopted_iid = matched_iid
                                        break
                                if not adopted_iid and cid and cid.upper().startswith("AI_INTENT_"):
                                    adopted_iid = cid[3:]
                            adopted_key = _emit(
                                OrderEventType.ORPHAN_ADOPTED.value,
                                venue=venue_key,
                                symbol=sym_norm,
                                owner=inferred_owner,
                                side=oo_side,
                                order_type=oo_type or "LIMIT",
                                qty=orig_q,
                                px=oo_px,
                                fq=exec_q if exec_q > 0.0 else None,
                                iid=adopted_iid,
                                cid=cid,
                                vid=oid,
                                vst=oo_vst,
                            )
                            if adopted_key and adopted_key in orders:
                                orders[adopted_key]["last_seen_open_ts"] = round(seen_ts, 3)
                                seen_open_keys.add(adopted_key)
                            stats["orphans_adopted"] += 1

            # 4. Reconcile active orders against CacheOrderFilledManager (strictly venue+symbol scoped).
            for key in list(orders.keys()):
                rec = orders.get(key)
                if not isinstance(rec, dict):
                    continue
                fill_row = self._find_matching_fill(rec, filled_by_sym_vid, filled_by_vid_any, filled_by_cid)
                if fill_row is None:
                    continue
                fill_qty = safe_pos_float(fill_row.get("quantity") or fill_row.get("qty"), default=0.0) or 0.0
                fill_px = safe_pos_float(fill_row.get("price"), default=None)
                req_qty = safe_pos_float(rec.get("qty"), default=None)
                common_fill_kw = dict(
                    venue=rec.get("v"),
                    symbol=rec.get("sym") or fill_row.get("symbol"),
                    owner=rec.get("owner"),
                    side=rec.get("side") or extract_order_side(fill_row),
                    fq=fill_qty,
                    avg_px=fill_px,
                    fee=safe_nonneg_float(fill_row.get("fee"), default=None),
                    fee_asset=clean_upper_scalar(fill_row.get("feeAsset") or fill_row.get("fee_asset")),
                    iid=rec.get("iid"),
                    cid=extract_client_order_id(fill_row, default=None) or rec.get("cid"),
                    vid=clean_order_id(extract_order_id(fill_row, default=None)) or rec.get("vid"),
                )
                venue_key = normalize_venue_name(rec.get("v")) or str(rec.get("v") or "").lower()
                _, sym_lc = _norm_sym_lc(rec.get("sym") or fill_row.get("symbol"), venue_key)
                snap_ver_ts = current_sym_ver.get((venue_key, sym_lc), 0.0)
                order_cr_ts = safe_pos_float(rec.get("created_ts"), default=0.0) or 0.0
                outcome = self._classify_fill_outcome(
                    fill_row,
                    req_qty=req_qty,
                    resting_now=(key in seen_open_keys),
                    order_created_ts=order_cr_ts,
                    snapshot_verified_ts=snap_ver_ts,
                )
                if outcome == OrderLedgerState.FILLED.value:
                    _emit(
                        OrderEventType.RECONCILED_FILLED.value,
                        order_type=rec.get("type") or fill_row.get("type"),
                        qty=req_qty or fill_qty,
                        px=rec.get("px") or fill_px,
                        vst=clean_upper_scalar(fill_row.get("status")) or "FILLED",
                        **common_fill_kw,
                    )
                    stats["reconciled_filled"] += 1
                elif outcome == OrderLedgerState.CANCELED.value:
                    _emit(
                        OrderEventType.RECONCILED_CANCELED.value,
                        order_type=rec.get("type") or fill_row.get("type"),
                        qty=req_qty or fill_qty,
                        px=rec.get("px") or fill_px,
                        vst=clean_upper_scalar(fill_row.get("status")) or "CANCELED",
                        reason="venue_fill_status_canceled",
                        **common_fill_kw,
                    )
                elif outcome == OrderLedgerState.EXPIRED.value:
                    _emit(
                        OrderEventType.RECONCILED_EXPIRED.value,
                        order_type=rec.get("type") or fill_row.get("type"),
                        qty=req_qty or fill_qty,
                        px=rec.get("px") or fill_px,
                        vst=clean_upper_scalar(fill_row.get("status")) or "EXPIRED",
                        reason="venue_fill_status_expired",
                        **common_fill_kw,
                    )
                elif outcome == OrderLedgerState.PARTIAL.value and fill_qty > (safe_nonneg_float(rec.get("fq"), default=0.0) or 0.0) + 1e-12:
                    raw_fill_vst = clean_upper_scalar(fill_row.get("status"))
                    partial_vst = (
                        raw_fill_vst
                        if raw_fill_vst and raw_fill_vst not in {"FILLED", "CLOSED"}
                        else "PARTIALLY_FILLED"
                    )
                    _emit(
                        OrderEventType.RECONCILED_PARTIAL.value,
                        vst=partial_vst,
                        **common_fill_kw,
                    )

            # 5. Converge orders absent from freshly-verified complete open-order snapshots covering THAT symbol.
            for key in list(orders.keys()):
                rec = orders.get(key)
                if not isinstance(rec, dict) or key in seen_open_keys:
                    continue
                venue_key = normalize_venue_name(rec.get("v")) or str(rec.get("v") or "").lower()
                state = str(rec.get("state") or OrderLedgerState.INTENT.value)
                ref_ts = max(safe_pos_float(rec.get("updated_ts"), default=0.0) or 0.0, safe_pos_float(rec.get("last_seen_open_ts"), default=0.0) or 0.0)
                _, sym_lc = _norm_sym_lc(rec.get("sym"), venue_key)
                # Strictly require a complete open-order snapshot covering THIS order's symbol
                # via the canonical UnifiedOrderBookEvidence absence rule (shared with active_intents).
                symbol_absent_verified = UnifiedOrderBookEvidence.proves_symbol_absent(
                    symbol_verified_ts,
                    venue_key,
                    sym_lc,
                    ref_ts=ref_ts,
                    grace_sec=self.reconcile_grace_sec,
                )
                close_ev_type, close_reason = None, None
                if state in {OrderLedgerState.OPEN.value, OrderLedgerState.PARTIAL.value, OrderLedgerState.CANCELING.value}:
                    if symbol_absent_verified:
                        cur_fq, req_qty = safe_nonneg_float(rec.get("fq"), default=0.0) or 0.0, safe_pos_float(rec.get("qty"), default=None)
                        if cur_fq > 0.0 and (req_qty is None or cur_fq >= req_qty * 0.9999):
                            close_ev_type, close_reason = OrderEventType.RECONCILED_FILLED.value, "filled_and_absent_from_open"
                        else:
                            close_ev_type, close_reason = OrderEventType.RECONCILED_CANCELED.value, "absent_from_verified_open_book"
                elif state in {OrderLedgerState.SUBMITTED.value, OrderLedgerState.SUBMIT_UNKNOWN.value}:
                    if symbol_absent_verified:
                        close_ev_type, close_reason = OrderEventType.RECONCILED_FAILED.value, "absent_after_verified_grace"
                elif state == OrderLedgerState.INTENT.value:
                    is_deferred = _parse_bool_flag(rec.get("deferred"), False)
                    iid_clean = clean_order_id(rec.get("iid"))
                    has_corr_id = bool(
                        iid_clean
                        or clean_order_id(rec.get("cid"))
                        or clean_order_id(rec.get("vid"))
                    )
                    if not is_deferred:
                        if ref_ts > 0.0 and (eff_now - ref_ts) >= self.intent_stale_sec:
                            close_ev_type, close_reason = OrderEventType.RECONCILED_EXPIRED.value, "intent_stale"
                    elif not has_corr_id:
                        # Anonymous deferred intents (no iid/cid/vid) cannot be correlated to any future submit.
                        if ref_ts > 0.0 and (eff_now - ref_ts) >= self.intent_stale_sec:
                            close_ev_type, close_reason = (
                                OrderEventType.RECONCILED_EXPIRED.value,
                                "anonymous_deferred_intent_stale",
                            )
                    elif (
                        active_outbox_iids is not None
                        and iid_clean
                        and iid_clean.lower().startswith("outbox-")
                        and not rec.get("submitted")
                        and not clean_order_id(rec.get("vid"))
                        and ref_ts > 0.0
                        and (eff_now - ref_ts) >= self.reconcile_grace_sec
                        and iid_clean.lower() not in active_outbox_iids
                    ):
                        close_ev_type, close_reason = (
                            OrderEventType.RECONCILED_EXPIRED.value,
                            "absent_from_retry_outbox",
                        )
                    elif ref_ts > 0.0 and (eff_now - ref_ts) >= self.deferred_intent_stale_sec:
                        close_ev_type, close_reason = (
                            OrderEventType.RECONCILED_EXPIRED.value,
                            "deferred_intent_stale",
                        )
                if close_ev_type:
                    close_vst = {
                        OrderEventType.RECONCILED_FILLED.value: OrderLedgerState.FILLED.value,
                        OrderEventType.RECONCILED_CANCELED.value: OrderLedgerState.CANCELED.value,
                        OrderEventType.RECONCILED_EXPIRED.value: OrderLedgerState.EXPIRED.value,
                    }.get(close_ev_type)
                    _emit(
                        close_ev_type,
                        venue=venue_key,
                        symbol=rec.get("sym"),
                        owner=rec.get("owner"),
                        side=rec.get("side"),
                        iid=rec.get("iid"),
                        cid=rec.get("cid"),
                        vid=rec.get("vid"),
                        vst=close_vst,
                        key=key if not (rec.get("iid") or rec.get("cid") or rec.get("vid")) else None,
                        reason=close_reason,
                    )

            # 6. Persist reconciler events, archive terminal rows, persist recently_closed, and publish snapshot.
            for rev in reconciler_events:
                self.append_event(rev)
            for k_rem in [k for k, v in orders.items() if str(v.get("state") or "") in TERMINAL_LEDGER_STATES]:
                rem = orders.pop(k_rem)
                if rem.get("submitted") or rem.get("vid"):
                    newly_closed.append(make_closed_summary(k_rem, rem, closed_ts=eff_now))
            recently_closed = self._update_recently_closed(recently_closed, newly_closed, eff_now)
            for summary in newly_closed:
                self.append_closed_record(summary)
                stats["reconciled_closed"] += 1

            stats["active_count"] = len(orders)
            next_core = {
                "schema_version": ORDER_LEDGER_SCHEMA_VERSION,
                "journal_cursor": cursor,
                "venue_verified_ts": {k: round(v, 3) for k, v in sorted(venue_verified_ts.items())},
                "symbol_verified_ts": {k: round(v, 3) for k, v in sorted(symbol_verified_ts.items())},
                "recently_closed": recently_closed,
                "orders": orders,
            }
            next_core_sig = encode_json_payload(next_core, sort_keys=True) if active_file_exists else None
            if (not active_file_exists) or next_core_sig != prev_core_sig or reconciler_events or newly_closed:
                self.write_active_snapshot({
                    "schema_version": ORDER_LEDGER_SCHEMA_VERSION,
                    "updated_ts": round(eff_now, 3),
                    "journal_cursor": cursor,
                    "venue_verified_ts": next_core["venue_verified_ts"],
                    "symbol_verified_ts": next_core["symbol_verified_ts"],
                    "recently_closed": recently_closed,
                    "orders": orders,
                })

        return stats


def _touch_ledger_hb(status: dict, *, poll_sec: float, force: bool = False) -> None:
    touch_module_heartbeat("order_ledger", interval_sec=30.0, force=force, status=status, intervals={"poll": poll_sec})


def main() -> int:
    import log  # noqa: F401 -- ensure stdout timestamps and daily logger file from startup

    poll_sec = resolve_worker_poll_sec()
    args = add_store_true_cli_args(
        argparse.ArgumentParser(description=__doc__), ("--once", "Run a single reconciliation pass and exit.")
    ).parse_args()
    reconciler = OrderLedgerReconciler()
    if args.once:
        print(f"[order_ledger] single pass: {reconciler.reconcile_once()}")
        return 0

    single_instance("order_ledger_reconciler_worker")
    print(f"[order_ledger] start (poll={poll_sec:g}s, grace={reconciler.reconcile_grace_sec:g}s, intent_stale={reconciler.intent_stale_sec:g}s)")
    _touch_ledger_hb({"active_count": 0}, poll_sec=poll_sec, force=True)
    while True:
        last_stats = None
        last_error = None
        try:
            stats = reconciler.reconcile_once()
            last_stats = stats
            if any(stats[k] for k in ("events_applied", "reconciled_open", "reconciled_filled", "reconciled_closed", "orphans_adopted")):
                print(f"[order_ledger] {stats}")
        except Exception as exc:  # noqa: BLE001
            last_error = f"{exc.__class__.__name__}: {exc}"
            print(f"[order_ledger] pass error (continuing): {exc}")
        if isinstance(last_stats, dict):
            hb_status = last_stats
        elif last_error is not None:
            hb_status = {"status": "error", "error": last_error}
        else:
            hb_status = {"status": "ok"}
        _touch_ledger_hb(hb_status, poll_sec=poll_sec)
        time.sleep(poll_sec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
