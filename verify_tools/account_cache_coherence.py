"""Pure cross-cache event-reflection coherence evaluator and passive disk auditor.

Validates that multiplexed account events (order placement, partial/full fills, cancellations,
and balance updates) converge consistently across all six cache/ledger reflection surfaces per venue:
  1. ``cache_order_open_<venue>.json``
  2. ``cache_order_filled_<venue>.json``
  3. ``cache_trade_<venue>.json[l]``
  4. ``cache_balance_<venue>.json``
  5. ``orders_active.json`` (Order Ledger SSoT)
  6. ``cache_account_health.json``
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from account_cache_health import SUPPORTED_HEALTH_VENUES
from botcore import (
    clean_order_id,
    clean_scalar_str,
    clean_upper_scalar,
    extract_order_id,
    extract_order_side,
    extract_timestamp_s,
    normalize_epoch_sec,
    resolve_now_ts,
    safe_nonneg_float,
    safe_pos_float,
)
from cacheaccount import (
    BALANCE_SYNC_INTERVAL_SEC,
    CacheBalanceManager,
    CacheOrderFilledManager,
    CacheOrderOpenManager,
    CacheTradeManager,
    OPEN_ORDER_SYNC_INTERVAL_SEC,
    ORDER_SYNC_INTERVAL_SEC,
    TRADE_SYNC_INTERVAL_SEC,
    UnifiedOrderBookEvidence,
)
from cachepaths import (
    CACHE_ACCOUNT_HEALTH_FILENAME,
    ORDER_LEDGER_ACTIVE_FILENAME,
    get_cachedb_dir,
    is_equity_venue,
    is_primary_stream_venue,
    normalize_venue_name,
    resolve_cachedb_filepath,
    split_symbol_assets,
)
from instrument_registry import canonical_balance_asset_key, select_instruments
from order_ledger_reconciler_worker import derive_reconcile_grace_sec
from providers.base import MarketDataProvider
from providers.order_id_context import extract_client_order_id
from providers.order_ledger import OrderLedgerReader
from providers.order_ohlc_utils import extract_executed_qty, extract_orig_qty
from state_io import cachedb_exists, meta_file_path, read_json_dict, read_json_dict_recovering, read_jsonl_bucket_map

_QUOTE_CASH_ASSETS: frozenset[str] = frozenset({"USD", "USDC", "USDT", "USDG", "EUR"})
_TERMINAL_FILL_STATUSES: frozenset[str] = frozenset({"FILLED", "CLOSED"})
_ACTIVE_WORKING_LEDGER_STATES: frozenset[str] = frozenset({"OPEN", "PARTIAL"})
_UNCONFIRMED_LEDGER_STATES: frozenset[str] = frozenset({"INTENT", "DEFERRED"})


@dataclass(frozen=True, slots=True)
class CoherenceViolation:
    """One deterministic cross-cache coherence discrepancy."""

    venue: str
    rule: str
    detail: str
    severity: str = "error"

    def format_line(self) -> str:
        return f"[{self.severity.upper()}] {self.venue}:{self.rule}: {self.detail}"


@dataclass(frozen=True, slots=True)
class VenueCoherenceEvidence:
    """Passive, read-only reflection snapshot for a single venue across all account caches."""

    venue: str
    open_items: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    open_verified_ts: float = 0.0
    open_latest_ts: float = 0.0
    open_sym_verified_ts: dict[str, float] = field(default_factory=dict)
    open_file_exists: bool = False
    filled_items: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    filled_latest_ft_ts: float = 0.0
    filled_sym_ft_ts: dict[str, float] = field(default_factory=dict)
    filled_file_exists: bool = False
    trade_items: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    trade_latest_ft_ts: float = 0.0
    trade_sym_ft_ts: dict[str, float] = field(default_factory=dict)
    trade_file_exists: bool = False
    trade_has_fetchtime_surface: bool = False
    balance_items: dict[str, dict[str, Any]] = field(default_factory=dict)
    balance_verified_ts: float = 0.0
    balance_file_exists: bool = False
    health_sync_ts_by_stream: dict[str, float] = field(default_factory=dict)
    configured_base_assets: frozenset[str] = frozenset()
    configured_symbols: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class AccountCacheCoherenceSnapshot:
    """Complete read-only cross-cache snapshot consumed by the pure coherence evaluator."""

    now_s: float
    venues: dict[str, VenueCoherenceEvidence] = field(default_factory=dict)
    ledger_active_orders: dict[str, dict[str, Any]] = field(default_factory=dict)
    ledger_updated_ts: float = 0.0
    ledger_present: bool = False
    health_present: bool = False


def derive_cross_cache_sync_grace_sec(
    open_sync_interval_sec: float = OPEN_ORDER_SYNC_INTERVAL_SEC,
) -> float:
    """Derive cross-cache event propagation grace window formulaically from open-order sync cadence."""
    base = safe_pos_float(open_sync_interval_sec, default=OPEN_ORDER_SYNC_INTERVAL_SEC) or OPEN_ORDER_SYNC_INTERVAL_SEC
    return round(base, 3)


def _resolve_order_assets(venue: str, symbol: str, balance_items: Mapping[str, Any]) -> tuple[str, str]:
    """Resolve ``(base_asset, quote_asset)`` balance keys for one venue order without venue branches."""
    if is_equity_venue(venue):
        base_k = canonical_balance_asset_key(symbol, venue=venue) or str(symbol or "").strip().upper()
        quote_k = MarketDataProvider.resolve_venue_default_quote(venue, fallback="USD")
    else:
        b_raw, q_raw = split_symbol_assets(symbol) or (None, None)
        base_k = canonical_balance_asset_key(b_raw, venue=venue) if b_raw else ""
        quote_k = (
            canonical_balance_asset_key(q_raw, venue=venue)
            if q_raw
            else MarketDataProvider.resolve_venue_default_quote(venue, fallback="USD")
        )
    if quote_k in _QUOTE_CASH_ASSETS and quote_k not in balance_items and "USD" in balance_items:
        quote_k = "USD"
    return base_k, quote_k


def _order_remaining_qty(row: Mapping[str, Any]) -> float:
    rem_q = safe_pos_float(row.get("remaining_qty"), default=None)
    if rem_q is not None:
        return rem_q
    orig_q = safe_pos_float(extract_orig_qty(row, default=None), default=0.0) or 0.0
    exec_q = safe_nonneg_float(extract_executed_qty(row, default=None), default=0.0) or 0.0
    return max(0.0, orig_q - exec_q)


def _is_filled_row_terminal(
    fill_row: Mapping[str, Any],
    open_row: Optional[Mapping[str, Any]] = None,
    *,
    req_qty: Any = None,
) -> bool:
    st_u = clean_upper_scalar(fill_row.get("status")) or ""
    if st_u in _TERMINAL_FILL_STATUSES:
        return True
    filled_q = safe_pos_float(
        extract_executed_qty(fill_row, default=None) or fill_row.get("quantity"),
        default=None,
    )
    orig_q = safe_pos_float(
        req_qty
        if req_qty is not None
        else (extract_orig_qty(open_row, default=None) if open_row is not None else extract_orig_qty(fill_row, default=None)),
        default=None,
    )
    if orig_q is not None and filled_q is not None and filled_q >= orig_q * (1.0 - 1e-6):
        return True
    return False


def _audit_stream_per_symbol_fetchtimes(
    out: list[CoherenceViolation],
    *,
    venue: str,
    stream_name: str,
    health_ts: float,
    now_s: float,
    items_map: Mapping[str, Any],
    sym_ft_ts: Mapping[str, float],
    configured_symbols: frozenset[str],
    per_sym_lag_grace: float,
) -> None:
    """Verify every tracked symbol in a stream snapshot has a fresh, non-future per-symbol fetchtime."""
    items_cf = {
        str(k).strip().casefold(): str(k).strip()
        for k in (items_map or {})
        if str(k or "").strip()
    }
    ft_by_cf = {
        str(k).strip().casefold(): ft_v
        for k, ft_s in (sym_ft_ts or {}).items()
        if str(k or "").strip() and (ft_v := safe_pos_float(ft_s, default=None)) is not None
    }
    if configured_symbols:
        checked_syms = [
            sym_s for sym_s in sorted(configured_symbols) if sym_s.casefold() in items_cf
        ]
    elif ft_by_cf:
        checked_syms = sorted(items_cf.values())
    else:
        checked_syms = []
    for sym_s in checked_syms:
        sym_ft = ft_by_cf.get(sym_s.casefold(), 0.0)
        if sym_ft <= 0.0:
            out.append(
                CoherenceViolation(
                    venue=venue,
                    rule="health_symbol_fetchtime_missing",
                    detail=(
                        f"{sym_s} present in {stream_name} items but missing fetchtime while "
                        f"health {stream_name}_sync_ts={health_ts:.1f}"
                    ),
                )
            )
        elif now_s > 0.0 and sym_ft > now_s + 60.0:
            out.append(
                CoherenceViolation(
                    venue=venue,
                    rule="health_symbol_fetchtime_future",
                    detail=(
                        f"{sym_s} {stream_name} fetchtime={sym_ft:.1f} is {sym_ft - now_s:.1f}s "
                        f"in the future relative to now_s={now_s:.1f}"
                    ),
                )
            )
        elif (health_ts - sym_ft) > per_sym_lag_grace:
            out.append(
                CoherenceViolation(
                    venue=venue,
                    rule="health_symbol_fetchtime_lag",
                    detail=(
                        f"{sym_s} {stream_name} fetchtime={sym_ft:.1f} lags health {stream_name}_sync_ts={health_ts:.1f} "
                        f"by {health_ts - sym_ft:.1f}s (> {per_sym_lag_grace:.1f}s)"
                    ),
                )
            )


def _evaluate_venue_coherence(
    ev: VenueCoherenceEvidence,
    snapshot: AccountCacheCoherenceSnapshot,
    *,
    sync_grace: float,
    ledger_grace: float,
    balance_poll_sec: float,
    trade_lookback_sec: float,
    require_health_disk_files: bool = True,
) -> list[CoherenceViolation]:
    out: list[CoherenceViolation] = []
    now_s = snapshot.now_s
    v = ev.venue

    # Index filled orders by (sym_lc, oid) and collect fill IDs + latest fill timestamp.
    filled_by_sym_oid: dict[tuple[str, str], dict[str, Any]] = {}
    filled_oids: set[str] = set()
    filled_fids: set[str] = set()
    latest_fill_ts = 0.0
    for k_sym, rows in (ev.filled_items or {}).items():
        sym_lc = str(k_sym or "").strip().lower()
        for row in rows or ():
            if not isinstance(row, dict):
                continue
            r_ts = extract_timestamp_s(row) or 0.0
            if r_ts > latest_fill_ts:
                latest_fill_ts = r_ts
            oid = clean_order_id(extract_order_id(row, default=None))
            r_sym_lc = str(row.get("symbol") or k_sym or "").strip().lower() or sym_lc
            if oid:
                filled_oids.add(oid)
                prev = filled_by_sym_oid.get((r_sym_lc, oid))
                prev_ts = extract_timestamp_s(prev) or 0.0 if prev else -1.0
                if prev is None or r_ts >= prev_ts:
                    filled_by_sym_oid[(r_sym_lc, oid)] = row
            for fid in row.get("_fillIds") or ():
                if (fid_s := clean_scalar_str(fid)) is not None:
                    filled_fids.add(fid_s)

    # 1. Causal fill -> balance freshness correlation.
    bal_ts_future = now_s > 0.0 and ev.balance_verified_ts > now_s + 60.0
    if latest_fill_ts > 0.0 and ev.balance_file_exists and (now_s - latest_fill_ts) >= sync_grace:
        if ev.balance_verified_ts <= 0.0 or bal_ts_future or latest_fill_ts > ev.balance_verified_ts + 0.5:
            lag_s = max(0.0, latest_fill_ts - (0.0 if bal_ts_future else ev.balance_verified_ts))
            out.append(
                CoherenceViolation(
                    venue=v,
                    rule="causal_fill_to_balance",
                    detail=(
                        f"latest_fill_ts={latest_fill_ts:.1f} vs balance_verified_ts={ev.balance_verified_ts:.1f} "
                        f"({'future_ts' if bal_ts_future else f'lag={lag_s:.1f}s'}, fill_age={now_s - latest_fill_ts:.1f}s)"
                    ),
                )
            )

    # Index resting open orders + aggregate expected quote/base locks.
    open_rows_flat: list[tuple[str, str, dict[str, Any]]] = []
    open_oids: set[str] = set()
    expected_quote_locked: dict[str, float] = {}
    expected_base_locked: dict[str, float] = {}
    for k_sym, rows in (ev.open_items or {}).items():
        for row in rows or ():
            if not isinstance(row, dict):
                continue
            o_sym = str(row.get("symbol") or k_sym or "").strip()
            oid = clean_order_id(extract_order_id(row, default=None)) or ""
            if oid:
                open_oids.add(oid)
            open_rows_flat.append((o_sym, oid, row))
            rem_q = _order_remaining_qty(row)
            if rem_q <= 1e-12:
                continue
            side_u = extract_order_side(row)
            base_k, quote_k = _resolve_order_assets(v, o_sym, ev.balance_items)
            if side_u == "BUY":
                px = safe_pos_float(row.get("price"), default=0.0) or 0.0
                if px > 0.0 and quote_k:
                    expected_quote_locked[quote_k] = expected_quote_locked.get(quote_k, 0.0) + (rem_q * px)
            elif side_u == "SELL" and base_k:
                expected_base_locked[base_k] = expected_base_locked.get(base_k, 0.0) + rem_q

    # 2. Bidirectional open-orders <-> balance locked parity (+ explicit far-future snapshot timestamp checks).
    if now_s > 0.0 and ev.balance_file_exists and ev.balance_verified_ts > now_s + 60.0:
        out.append(
            CoherenceViolation(
                venue=v,
                rule="balance_verified_ts_future",
                detail=(
                    f"balance_verified_ts={ev.balance_verified_ts:.1f} is "
                    f"{ev.balance_verified_ts - now_s:.1f}s in the future (> 60s skew)"
                ),
                severity="error",
            )
        )
    if now_s > 0.0 and ev.open_file_exists and ev.open_latest_ts > now_s + 60.0:
        out.append(
            CoherenceViolation(
                venue=v,
                rule="open_orders_ts_future",
                detail=(
                    f"open_latest_ts={ev.open_latest_ts:.1f} is "
                    f"{ev.open_latest_ts - now_s:.1f}s in the future (> 60s skew)"
                ),
                severity="error",
            )
        )
    if (
        ev.open_file_exists
        and ev.balance_file_exists
        and ev.open_latest_ts > 0.0
        and ev.balance_verified_ts > 0.0
        and (now_s <= 0.0 or (ev.open_latest_ts <= now_s + 60.0 and ev.balance_verified_ts <= now_s + 60.0))
    ):
        bal_age = max(0.0, now_s - ev.balance_verified_ts)
        open_age = max(0.0, now_s - ev.open_latest_ts)
        both_aligned = abs(ev.open_latest_ts - ev.balance_verified_ts) <= sync_grace and max(bal_age, open_age) >= 2.0
        open_settled_newer = ev.open_latest_ts > ev.balance_verified_ts + 0.5 and (
            open_age >= sync_grace
            or (open_age >= 2.0 and (ev.open_latest_ts - ev.balance_verified_ts) >= sync_grace)
            or bal_age >= (balance_poll_sec + sync_grace)
        )
        bal_settled_newer = ev.balance_verified_ts > ev.open_latest_ts + 0.5 and (
            bal_age >= sync_grace
            or (bal_age >= 2.0 and (ev.balance_verified_ts - ev.open_latest_ts) >= sync_grace)
            or open_age >= (2.0 * sync_grace)
        )

        if both_aligned or open_settled_newer:
            for asset_k, b_row in ev.balance_items.items():
                if not isinstance(b_row, dict):
                    continue
                locked_v = safe_nonneg_float(b_row.get("locked"), default=None)
                if locked_v is None or locked_v <= 1e-6:
                    continue
                if asset_k in _QUOTE_CASH_ASSETS:
                    exp_q = expected_quote_locked.get(asset_k, 0.0)
                    tol_q = max(1.0, exp_q * 0.03)
                    if locked_v > exp_q + tol_q:
                        out.append(
                            CoherenceViolation(
                                venue=v,
                                rule="balance_phantom_lock",
                                detail=(
                                    f"quote asset {asset_k} balance_locked={locked_v:.2f} > "
                                    f"open_buy_notional={exp_q:.2f} (open_ts={ev.open_latest_ts:.1f}, "
                                    f"bal_ts={ev.balance_verified_ts:.1f})"
                                ),
                            )
                        )
                elif not ev.configured_base_assets or asset_k in ev.configured_base_assets:
                    exp_b = expected_base_locked.get(asset_k, 0.0)
                    tol_b = max(1e-6, exp_b * 0.01)
                    if locked_v > exp_b + tol_b:
                        out.append(
                            CoherenceViolation(
                                venue=v,
                                rule="balance_phantom_lock",
                                detail=(
                                    f"base asset {asset_k} balance_locked={locked_v:.6g} > "
                                    f"open_sell_qty={exp_b:.6g} (open_ts={ev.open_latest_ts:.1f}, "
                                    f"bal_ts={ev.balance_verified_ts:.1f})"
                                ),
                            )
                        )

        if (both_aligned or bal_settled_newer) and ev.balance_items:
            for q_k, exp_q in expected_quote_locked.items():
                tol_q = max(1.0, exp_q * 0.03)
                b_row = ev.balance_items.get(q_k)
                if isinstance(b_row, dict) and b_row.get("locked") is None:
                    continue
                act_q = safe_nonneg_float(b_row.get("locked") if isinstance(b_row, dict) else 0.0, default=0.0) or 0.0
                if act_q + tol_q < exp_q:
                    out.append(
                        CoherenceViolation(
                            venue=v,
                            rule="open_order_missing_balance_lock",
                            detail=(
                                f"quote asset {q_k} open_buy_notional={exp_q:.2f} > "
                                f"balance_locked={act_q:.2f} (open_ts={ev.open_latest_ts:.1f}, "
                                f"bal_ts={ev.balance_verified_ts:.1f})"
                            ),
                        )
                    )
            for b_k, exp_b in expected_base_locked.items():
                tol_b = max(1e-6, exp_b * 0.01)
                b_row = ev.balance_items.get(b_k)
                if isinstance(b_row, dict) and b_row.get("locked") is None:
                    continue
                act_b = safe_nonneg_float(b_row.get("locked") if isinstance(b_row, dict) else 0.0, default=0.0) or 0.0
                if act_b + tol_b < exp_b:
                    out.append(
                        CoherenceViolation(
                            venue=v,
                            rule="open_order_missing_balance_lock",
                            detail=(
                                f"base asset {b_k} open_sell_qty={exp_b:.6g} > "
                                f"balance_locked={act_b:.6g} (open_ts={ev.open_latest_ts:.1f}, "
                                f"bal_ts={ev.balance_verified_ts:.1f})"
                            ),
                        )
                    )

    # 3. Open-order vs terminal filled-order mutual exclusion.
    for o_sym, oid, o_row in open_rows_flat:
        if not oid:
            continue
        f_row = filled_by_sym_oid.get((o_sym.lower(), oid))
        if f_row is None or not _is_filled_row_terminal(f_row, open_row=o_row):
            continue
        f_ts = extract_timestamp_s(f_row) or 0.0
        if f_ts <= 0.0 or (now_s - f_ts) >= sync_grace:
            out.append(
                CoherenceViolation(
                    venue=v,
                    rule="open_order_already_filled",
                    detail=f"{o_sym} orderId={oid} is terminal in filled cache (ts={f_ts:.1f}) yet still in open_orders",
                )
            )

    # 4. Recent trade cache -> filled-order cache parity.
    if ev.trade_items and ev.filled_file_exists:
        missing_keys: dict[str, tuple[str, float]] = {}
        for k_sym, rows in ev.trade_items.items():
            for row in rows or ():
                if not isinstance(row, dict):
                    continue
                t_ts = extract_timestamp_s(row) or 0.0
                if t_ts <= 0.0 or (now_s - t_ts) > trade_lookback_sec or (now_s - t_ts) < sync_grace:
                    continue
                oid = clean_order_id(extract_order_id(row, default=None))
                tid = clean_scalar_str(row.get("id") if row.get("id") is not None else row.get("tid"))
                if not oid and not tid:
                    continue
                if (oid and oid in filled_oids) or (tid and tid in filled_fids):
                    continue
                dedup_k = oid or tid or ""
                if dedup_k not in missing_keys:
                    missing_keys[dedup_k] = (str(row.get("symbol") or k_sym or ""), t_ts)
        for miss_id, (m_sym, m_ts) in sorted(missing_keys.items()):
            out.append(
                CoherenceViolation(
                    venue=v,
                    rule="trade_missing_in_filled",
                    detail=f"{m_sym} id={miss_id} (age={now_s - m_ts:.1f}s) exists in trade cache but not filled cache",
                )
            )

    # 5. Account caches <-> Order Ledger (orders_active.json) parity.
    if snapshot.ledger_present:
        ledger_ids_for_venue: set[str] = set()
        venue_ledger_rows: list[tuple[str, dict[str, Any]]] = []
        for l_key, rec in (snapshot.ledger_active_orders or {}).items():
            if not isinstance(rec, dict):
                continue
            rec_v = normalize_venue_name(rec.get("v"))
            if rec_v != v:
                continue
            venue_ledger_rows.append((l_key, rec))
            for cand_id in (rec.get("vid"), rec.get("cid")):
                if (cid_c := clean_order_id(cand_id)) is not None:
                    ledger_ids_for_venue.add(cid_c)

        # 5a. Open order on venue missing from orders_active.json beyond ledger_grace.
        for o_sym, oid, o_row in open_rows_flat:
            if not oid:
                continue
            o_cid = clean_order_id(extract_client_order_id(o_row, default=None))
            if oid in ledger_ids_for_venue or (o_cid and o_cid in ledger_ids_for_venue):
                continue
            o_ts = extract_timestamp_s(o_row) or 0.0
            aged_out = (
                (o_ts > 0.0 and (now_s - o_ts) >= ledger_grace)
                or (
                    o_ts <= 0.0
                    and snapshot.ledger_updated_ts > 0.0
                    and ev.open_verified_ts > 0.0
                    and (snapshot.ledger_updated_ts - ev.open_verified_ts) >= ledger_grace
                )
            )
            if aged_out:
                out.append(
                    CoherenceViolation(
                        venue=v,
                        rule="open_order_missing_from_ledger",
                        detail=f"{o_sym} orderId={oid} rests in open_orders (age={now_s - o_ts if o_ts > 0 else -1:.1f}s) but missing from orders_active.json",
                    )
                )

        # 5b & 5c. Active ledger order already filled or proven absent from open_orders beyond ledger_grace.
        sym_ver_keyed = {(v, sym_l): ts_v for sym_l, ts_v in ev.open_sym_verified_ts.items()}
        for _l_key, rec in venue_ledger_rows:
            st_u = clean_upper_scalar(rec.get("state")) or ""
            if st_u in _UNCONFIRMED_LEDGER_STATES:
                continue
            vid = clean_order_id(rec.get("vid"))
            if not vid or vid in open_oids:
                continue
            sym_l = str(rec.get("sym") or "").strip().lower()
            rec_ts = (
                safe_pos_float(rec.get("updated_ts"), default=None)
                or safe_pos_float(rec.get("created_ts"), default=0.0)
                or 0.0
            )
            f_row = filled_by_sym_oid.get((sym_l, vid))
            if f_row is not None and _is_filled_row_terminal(f_row, req_qty=rec.get("qty")):
                f_ts = extract_timestamp_s(f_row) or rec_ts
                if f_ts <= 0.0 or (now_s - f_ts) >= ledger_grace:
                    out.append(
                        CoherenceViolation(
                            venue=v,
                            rule="ledger_stuck_filled_order",
                            detail=f"{rec.get('sym')} orderId={vid} state={st_u} is terminal in filled cache (ts={f_ts:.1f}) yet active in orders_active.json",
                        )
                    )
                continue
            if st_u in _ACTIVE_WORKING_LEDGER_STATES and rec_ts > 0.0 and UnifiedOrderBookEvidence.proves_symbol_absent(
                sym_ver_keyed,
                v,
                sym_l,
                ref_ts=rec_ts,
                grace_sec=ledger_grace,
            ):
                out.append(
                    CoherenceViolation(
                        venue=v,
                        rule="ledger_orphan_active_order",
                        detail=f"{rec.get('sym')} orderId={vid} state={st_u} absent from verified open_orders past grace ({ledger_grace:.0f}s)",
                    )
                )

    # 6. Health marker <-> disk file existence and timestamp parity.
    if snapshot.health_present:
        stream_disk_meta: tuple[tuple[str, bool, float], ...] = (
            ("balances", ev.balance_file_exists, ev.balance_verified_ts),
            ("open_orders", ev.open_file_exists, ev.open_latest_ts),
            ("filled_orders", ev.filled_file_exists, ev.filled_latest_ft_ts),
        )
        for stream_name, file_ok, disk_ts in stream_disk_meta:
            h_ts = ev.health_sync_ts_by_stream.get(stream_name, 0.0)
            if h_ts <= 0.0:
                continue
            if not file_ok:
                if require_health_disk_files:
                    out.append(
                        CoherenceViolation(
                            venue=v,
                            rule="health_disk_missing",
                            detail=f"health reports {stream_name}_sync_ts={h_ts:.1f} but disk snapshot file is missing",
                        )
                    )
            elif now_s > 0.0 and disk_ts > now_s + 60.0:
                out.append(
                    CoherenceViolation(
                        venue=v,
                        rule="health_disk_future_ts",
                        detail=f"{stream_name} disk_ts={disk_ts:.1f} is {disk_ts - now_s:.1f}s in the future relative to now_s={now_s:.1f}",
                    )
                )
            elif disk_ts <= 0.0 or (h_ts - disk_ts) > sync_grace:
                out.append(
                    CoherenceViolation(
                        venue=v,
                        rule="health_disk_lag",
                        detail=f"health reports {stream_name}_sync_ts={h_ts:.1f} ahead of disk_ts={disk_ts:.1f} by {h_ts - disk_ts:.1f}s",
                    )
                )

        # 6b. Stream-level lag on trades (when fetchtime surface exists) and per-symbol fetchtime parity
        # for both filled_orders and trades (prevents max() from masking one invalidated symbol).
        fo_h_ts = ev.health_sync_ts_by_stream.get("filled_orders", 0.0)
        if fo_h_ts > 0.0 and ev.filled_file_exists and ev.filled_latest_ft_ts > 0.0:
            _audit_stream_per_symbol_fetchtimes(
                out,
                venue=v,
                stream_name="filled_orders",
                health_ts=fo_h_ts,
                now_s=now_s,
                items_map=ev.filled_items,
                sym_ft_ts=ev.filled_sym_ft_ts,
                configured_symbols=ev.configured_symbols,
                per_sym_lag_grace=(
                    (ORDER_SYNC_INTERVAL_SEC + sync_grace)
                    if is_primary_stream_venue(v)
                    else sync_grace
                ),
            )

        tr_h_ts = ev.health_sync_ts_by_stream.get("trades", 0.0)
        if tr_h_ts > 0.0 and ev.trade_file_exists and ev.trade_has_fetchtime_surface:
            tr_stream_grace = (
                (TRADE_SYNC_INTERVAL_SEC + sync_grace)
                if is_primary_stream_venue(v)
                else sync_grace
            )
            tr_disk_lag = False
            if now_s > 0.0 and ev.trade_latest_ft_ts > now_s + 60.0:
                out.append(
                    CoherenceViolation(
                        venue=v,
                        rule="health_disk_future_ts",
                        detail=f"trades disk_ts={ev.trade_latest_ft_ts:.1f} is {ev.trade_latest_ft_ts - now_s:.1f}s in the future relative to now_s={now_s:.1f}",
                    )
                )
            elif ev.trade_latest_ft_ts <= 0.0 or (tr_h_ts - ev.trade_latest_ft_ts) > tr_stream_grace:
                tr_disk_lag = True
                out.append(
                    CoherenceViolation(
                        venue=v,
                        rule="health_disk_lag",
                        detail=f"health reports trades_sync_ts={tr_h_ts:.1f} ahead of disk_ts={ev.trade_latest_ft_ts:.1f} by {tr_h_ts - ev.trade_latest_ft_ts:.1f}s",
                    )
                )
            if not tr_disk_lag and ev.trade_latest_ft_ts > 0.0:
                _audit_stream_per_symbol_fetchtimes(
                    out,
                    venue=v,
                    stream_name="trades",
                    health_ts=tr_h_ts,
                    now_s=now_s,
                    items_map=ev.trade_items,
                    sym_ft_ts=ev.trade_sym_ft_ts,
                    configured_symbols=ev.configured_symbols,
                    per_sym_lag_grace=tr_stream_grace,
                )

    return out


def evaluate_account_cache_coherence(
    snapshot: AccountCacheCoherenceSnapshot,
    *,
    sync_grace_sec: Optional[float] = None,
    ledger_grace_sec: Optional[float] = None,
    balance_poll_sec: float = BALANCE_SYNC_INTERVAL_SEC,
    trade_lookback_sec: float = 86400.0,
    require_health_disk_files: bool = True,
) -> list[CoherenceViolation]:
    """Pure, deterministic cross-cache coherence evaluation across all venues in ``snapshot``."""
    sync_g = (
        safe_pos_float(sync_grace_sec, default=None)
        if sync_grace_sec is not None
        else derive_cross_cache_sync_grace_sec(OPEN_ORDER_SYNC_INTERVAL_SEC)
    ) or derive_cross_cache_sync_grace_sec(OPEN_ORDER_SYNC_INTERVAL_SEC)
    ledger_g = (
        safe_pos_float(ledger_grace_sec, default=None)
        if ledger_grace_sec is not None
        else derive_reconcile_grace_sec(OPEN_ORDER_SYNC_INTERVAL_SEC)
    ) or derive_reconcile_grace_sec(OPEN_ORDER_SYNC_INTERVAL_SEC)
    bal_poll = safe_pos_float(balance_poll_sec, default=BALANCE_SYNC_INTERVAL_SEC) or BALANCE_SYNC_INTERVAL_SEC
    lookback = safe_pos_float(trade_lookback_sec, default=86400.0) or 86400.0

    violations: list[CoherenceViolation] = []
    for v_key in sorted(snapshot.venues.keys()):
        violations.extend(
            _evaluate_venue_coherence(
                snapshot.venues[v_key],
                snapshot,
                sync_grace=sync_g,
                ledger_grace=ledger_g,
                balance_poll_sec=bal_poll,
                trade_lookback_sec=lookback,
                require_health_disk_files=require_health_disk_files,
            )
        )
    return violations


def _resolve_cache_file(cachedb_dir: str, filename: str) -> str:
    direct = os.path.join(cachedb_dir, "account", filename)
    if cachedb_exists(direct, include_previous=True):
        return direct
    flat = os.path.join(cachedb_dir, filename)
    if cachedb_exists(flat, include_previous=True):
        return flat
    return str(resolve_cachedb_filepath(flat))


def _configured_venue_metadata(venue: str) -> tuple[frozenset[str], frozenset[str]]:
    try:
        specs = select_instruments(venue)
    except Exception:
        return frozenset(), frozenset()
    bases: set[str] = set()
    syms: set[str] = set()
    for spec in specs.values():
        if (s_clean := clean_scalar_str(spec.symbol)) is not None:
            syms.add(s_clean)
        raw_part = spec.symbol if is_equity_venue(venue) else (spec.base or (split_symbol_assets(spec.symbol) or ("", ""))[0])
        if (b_k := canonical_balance_asset_key(raw_part, venue=venue)) is not None:
            bases.add(b_k)
    return frozenset(bases), frozenset(syms)


def _read_health_sync_map(
    cachedb_dir: str, health_path: Optional[str]
) -> tuple[bool, dict[str, dict[str, float]]]:
    candidates = [health_path] if health_path else [
        os.path.join(cachedb_dir, "account", CACHE_ACCOUNT_HEALTH_FILENAME),
        os.path.join(cachedb_dir, CACHE_ACCOUNT_HEALTH_FILENAME),
    ]
    raw_doc = None
    for cand in candidates:
        if cand and os.path.isfile(cand):
            raw_doc = read_json_dict(cand, default=None)
            break
    if not isinstance(raw_doc, dict):
        return False, {}
    out: dict[str, dict[str, float]] = {}
    venues_map = raw_doc.get("venues") if isinstance(raw_doc.get("venues"), dict) else {}
    for v_k, v_entry in venues_map.items():
        v_norm = normalize_venue_name(v_k)
        if not v_norm or not isinstance(v_entry, dict):
            continue
        s_map: dict[str, float] = {}
        for stream_name in ("balances", "open_orders", "filled_orders", "trades"):
            ms_v = safe_pos_float(v_entry.get(f"{stream_name}_sync_at_ms"), default=None)
            if ms_v is not None:
                s_map[stream_name] = ms_v / 1000.0
        out[v_norm] = s_map
    return True, out


def load_account_cache_coherence_snapshot(
    cachedb_dir: str,
    *,
    now_s: Optional[float] = None,
    venues: Iterable[str] = SUPPORTED_HEALTH_VENUES,
    health_path: Optional[str] = None,
    include_ledger: bool = True,
    explicit_ledger_dir: Optional[str] = None,
) -> AccountCacheCoherenceSnapshot:
    """Passively load all cross-cache reflection files under ``cachedb_dir`` without mutating disk."""
    eff_now = resolve_now_ts(now_s)
    health_present, health_by_venue = _read_health_sync_map(cachedb_dir, health_path)

    venue_evidence: dict[str, VenueCoherenceEvidence] = {}
    for raw_v in venues:
        v_norm = normalize_venue_name(raw_v)
        if not v_norm:
            continue

        # 1. Open orders
        oo_fname = CacheOrderOpenManager.VENUE_FILENAMES.get(v_norm, "")
        oo_path = _resolve_cache_file(cachedb_dir, oo_fname) if oo_fname else ""
        oo_exists = bool(oo_path and cachedb_exists(oo_path, include_previous=True))
        oo_items, oo_ft, oo_ver, oo_sver = (
            CacheOrderOpenManager.read_venue_open_orders_snapshot_detailed(oo_path)
            if oo_exists
            else ({}, {}, 0.0, {})
        )
        oo_latest = max(
            [
                oo_ver,
                *(oo_sver or {}).values(),
                *(normalize_epoch_sec(ft_v, default=0.0) or 0.0 for ft_v in (oo_ft or {}).values()),
            ],
            default=0.0,
        )
        oo_sym_ver_lc = {str(sk).strip().lower(): sv for sk, sv in (oo_sver or {}).items() if sv > 0.0}

        # 2. Filled orders
        fo_fname = CacheOrderFilledManager.VENUE_FILENAMES.get(v_norm, "")
        fo_path = _resolve_cache_file(cachedb_dir, fo_fname) if fo_fname else ""
        fo_exists = bool(fo_path and cachedb_exists(fo_path, include_previous=True))
        fo_items, fo_ft = (
            CacheOrderFilledManager.read_venue_filled_orders_snapshot(
                fo_path, venue=v_norm, allow_legacy=False, allow_trade_fallback=False
            )
            if fo_exists
            else ({}, {})
        )
        fo_sym_ft = {
            str(sk).strip(): ft_s
            for sk, ft_v in (fo_ft or {}).items()
            if str(sk or "").strip() and (ft_s := normalize_epoch_sec(ft_v, default=0.0) or 0.0) > 0.0
        }
        fo_latest_ft = max(fo_sym_ft.values(), default=0.0)

        # 3. Trade cache mirror
        tr_fname = CacheTradeManager.VENUE_FILENAMES.get(v_norm, "")
        tr_path = _resolve_cache_file(cachedb_dir, tr_fname) if tr_fname else ""
        tr_exists = bool(tr_path and cachedb_exists(tr_path, include_previous=True))
        tr_items: dict[str, list[dict[str, Any]]] = {}
        tr_ft_raw: Optional[dict[str, Any]] = None
        if tr_exists:
            if tr_fname.lower().endswith(".jsonl"):
                tr_raw, _ = read_jsonl_bucket_map(tr_path)
                tr_items = {
                    str(sk): [r for r in rows if isinstance(r, dict)]
                    for sk, rows in (tr_raw or {}).items()
                    if isinstance(rows, list)
                }
                tr_meta = read_json_dict(meta_file_path(tr_path), default=None)
                if isinstance(tr_meta, dict) and isinstance(tr_meta.get("fetchtime"), dict):
                    tr_ft_raw = tr_meta["fetchtime"]
            else:
                tr_doc = read_json_dict_recovering(tr_path, default=None, repair_files=False)
                raw_it = tr_doc.get("items") if isinstance(tr_doc, dict) and isinstance(tr_doc.get("items"), dict) else {}
                tr_items = {
                    str(sk): [r for r in rows if isinstance(r, dict)]
                    for sk, rows in raw_it.items()
                    if isinstance(rows, list)
                }
                if isinstance(tr_doc, dict) and isinstance(tr_doc.get("fetchtime"), dict):
                    tr_ft_raw = tr_doc["fetchtime"]
        tr_has_ft_surface = isinstance(tr_ft_raw, dict)
        tr_sym_ft = {
            str(sk).strip(): ft_s
            for sk, ft_v in (tr_ft_raw or {}).items()
            if str(sk or "").strip() and (ft_s := normalize_epoch_sec(ft_v, default=0.0) or 0.0) > 0.0
        }
        tr_latest_ft = max(tr_sym_ft.values(), default=0.0)

        # 4. Balances
        bal_fname = CacheBalanceManager.VENUE_FILENAMES.get(v_norm, "")
        bal_path = _resolve_cache_file(cachedb_dir, bal_fname) if bal_fname else ""
        bal_exists = bool(bal_path and cachedb_exists(bal_path, include_previous=True))
        bal_items, bal_ver, _ = (
            CacheBalanceManager.read_venue_balances_snapshot(bal_path)
            if bal_exists
            else ({}, 0.0, {})
        )

        cfg_bases, cfg_syms = _configured_venue_metadata(v_norm)
        venue_evidence[v_norm] = VenueCoherenceEvidence(
            venue=v_norm,
            open_items=oo_items,
            open_verified_ts=oo_ver,
            open_latest_ts=oo_latest,
            open_sym_verified_ts=oo_sym_ver_lc,
            open_file_exists=oo_exists,
            filled_items=fo_items,
            filled_latest_ft_ts=fo_latest_ft,
            filled_sym_ft_ts=fo_sym_ft,
            filled_file_exists=fo_exists,
            trade_items=tr_items,
            trade_latest_ft_ts=tr_latest_ft,
            trade_sym_ft_ts=tr_sym_ft,
            trade_file_exists=tr_exists,
            trade_has_fetchtime_surface=tr_has_ft_surface,
            balance_items=bal_items,
            balance_verified_ts=bal_ver,
            balance_file_exists=bal_exists,
            health_sync_ts_by_stream=health_by_venue.get(v_norm, {}),
            configured_base_assets=cfg_bases,
            configured_symbols=cfg_syms,
        )

    # 5. Order Ledger (orders_active.json)
    ledger_active: dict[str, dict[str, Any]] = {}
    ledger_updated_ts = 0.0
    ledger_present = False
    if include_ledger:
        is_default_dir = os.path.abspath(cachedb_dir) == os.path.abspath(get_cachedb_dir())
        if explicit_ledger_dir is not None:
            reader = OrderLedgerReader(cachedb_dir=explicit_ledger_dir)
        elif is_default_dir:
            reader = OrderLedgerReader()
        else:
            reader = OrderLedgerReader(cachedb_dir=cachedb_dir)
        active_p = reader.active_path()
        if os.path.isfile(active_p):
            ledger_present = True
            snap_l = reader.load_active_state(apply_tail=False)
            ledger_active = snap_l.get("orders") if isinstance(snap_l.get("orders"), dict) else {}
            ledger_updated_ts = safe_nonneg_float(snap_l.get("updated_ts"), default=0.0) or 0.0

    return AccountCacheCoherenceSnapshot(
        now_s=eff_now,
        venues=venue_evidence,
        ledger_active_orders=ledger_active,
        ledger_updated_ts=ledger_updated_ts,
        ledger_present=ledger_present,
        health_present=health_present,
    )


def audit_account_cache_coherence(
    cachedb_dir: str,
    *,
    now_s: Optional[float] = None,
    venues: Iterable[str] = SUPPORTED_HEALTH_VENUES,
    health_path: Optional[str] = None,
    include_ledger: bool = True,
    explicit_ledger_dir: Optional[str] = None,
    sync_grace_sec: Optional[float] = None,
    ledger_grace_sec: Optional[float] = None,
    balance_poll_sec: float = BALANCE_SYNC_INTERVAL_SEC,
    trade_lookback_sec: float = 86400.0,
    require_health_disk_files: bool = True,
) -> list[CoherenceViolation]:
    """Load passive account-cache reflections under ``cachedb_dir`` and return all coherence violations."""
    snap = load_account_cache_coherence_snapshot(
        cachedb_dir,
        now_s=now_s,
        venues=venues,
        health_path=health_path,
        include_ledger=include_ledger,
        explicit_ledger_dir=explicit_ledger_dir,
    )
    return evaluate_account_cache_coherence(
        snap,
        sync_grace_sec=sync_grace_sec,
        ledger_grace_sec=ledger_grace_sec,
        balance_poll_sec=balance_poll_sec,
        trade_lookback_sec=trade_lookback_sec,
        require_health_disk_files=require_health_disk_files,
    )
