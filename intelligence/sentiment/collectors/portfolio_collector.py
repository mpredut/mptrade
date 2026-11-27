"""Portfolio positions, venue DCA cycles, trailing peaks, recent fills, and runtime states collector."""
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
import logging
import math
import os
import re
import time as _time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from botcore import (
    TimedQuote,
    clean_order_id,
    coerce_dict_maps,
    extract_order_id,
    extract_order_side,
    extract_timestamp_s,
    first_epoch_sec,
    first_pos_float,
    is_live_network_backfill_enabled,
    load_key_value_conf,
    normalize_epoch_sec,
    normalize_timestamp_ms,
    quote_spread_bps,
    resolve_now_ts,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
    split_csv,
    _parse_bool_flag,
    _prices_diverged,
)
from cachepaths import (
    cache_symbol_variants,
    canonical_base_asset,
    canonical_symbol_key,
    clean_symbol_key,
    clean_upper_scalar,
    compatible_quote_suffixes,
    cycle_state_filename,
    ensure_equity_suffix,
    ensure_quote_suffix,
    ensure_xstock_pair,
    filter_asset_class_variants,
    has_equity_suffix,
    is_equity_venue,
    is_equity_venue_or_suffix,
    is_primary_stream_venue,
    is_shadow_or_paper_state_payload,
    iter_venue_cycle_state_files,
    normalize_venue_name,
    parse_cycle_state_symbol,
    resolve_cachedb_filepath,
    resolve_workspace_dir_from_cachedb,
    split_symbol_assets,
    symbol_quote_asset,
)
from state_stores.assetguardian_state import AssetguardianStateStore
from state_stores.mt_state import MtStateStore
from state_stores.smart_stop_loss_state import SmartStopLossStateStore
from state_stores.trailing_state import TrailingStateStore
from intelligence.macro_sentiment_ssot import (
    load_venue_balances_state,
    resolve_read_json_cached,
    resolve_save_json_state,
    scan_cached_price_archive_high_since,
)
from intelligence.external.collectors.base_collector import (
    TelemetryCacheCollectorInterface,
    TelemetryCollectorFactory,
)
import intelligence.sentiment.collectors.microstructure_collector as _mc
from intelligence.sentiment.collectors.microstructure_collector import (
    ensure_cacheprice_classes_bound,
    _iter_t212_state_files,
    _resolve_now_and_read_json,
)
from intelligence.sentiment.evaluators.microstructure_evaluator import (
    first_variant_dict,
)
from providers.order_ledger import OrderLedgerReader
from providers.order_status_ssot import is_terminal_audit_status
from providers.venue_fee_maintenance import (
    BinanceFeeMaintainer,
    build_persisted_venue_fee_resolver,
)
from intelligence.sentiment.evaluators.portfolio_evaluator import (
    apply_cycle_fallback_to_recent_fills,
    clip_open_lots_to_held_qty,
    enrich_positions_with_orders_and_cycles,
    extract_symbol_recent_fills,
    parse_venue_cycle_state,
    prune_superseded_open_lots_to_held_qty,
    reconcile_fresh_lots_vwap,
    reconcile_venue_cycles_with_recent_orders,
    select_fresh_open_lots,
)
from intelligence.sentiment.evaluators.price_series_evaluator import round_opt_px
from intelligence.sentiment.evaluators.tca_risk_evaluator import (
    resolve_symbol_price,
    resolve_venue_fee_and_limits,
)
from instrument_registry import (
    canonical_balance_asset_key,
    canonical_display_symbol,
    casefold_get,
    disabled_registry_symbols,
    observe_only_registry_symbols,
    resolve_kraken_tokenized_pair,
    resolve_registered_symbol_case,
    resolve_symbol_market_session,
    trailing_venues,
    venue_default_quote,
)
import order_retry as _oq
from providers.ai_circuit_breaker import (
    DEFAULT_XSTOCK_SELL_COOLDOWN_HOURS,
    DEFAULT_XSTOCK_SELL_CRASH_MIN_CONFIDENCE,
    DEFAULT_XSTOCK_SELL_CRASH_MIN_TARGET_DROP_PCT,
    DEFAULT_XSTOCK_SELL_MIN_CONFIDENCE,
    DEFAULT_XSTOCK_SELL_MIN_PROFIT_PCT,
    DEFAULT_XSTOCK_SELL_MIN_TARGET_DROP_PCT,
    parse_intent_circuit_breaker_config,
)
from providers.base import MarketDataProvider
from providers.execution_audit import _extract_audit_fill_metrics
from providers.order_id_context import derive_venue_128bit_client_order_id, resolve_order_origin
from providers.order_ohlc_utils import extract_executed_qty, extract_remaining_qty, normalize_open_order
from providers.intent_state import (
    collect_known_ai_order_ids,
    intent_venue_client_order_ids,
)
from providers.intent_state_store import load_durable_intents_from_cachedb
from providers.portfolio_state_store import (
    load_venue_balance_caches,
    resolve_kraken_trade_cache_position,
    resolve_portfolio_positions_state_path,
    resolve_venue_balance_cache_path,
    upsert_portfolio_positions_snapshot,
)
from providers.portfolio_valuation import (
    BalanceObservation,
    BalanceSnapshotRecord,
    build_cached_balance_observations,
    extract_balance_observations_from_caches,
    extract_position_held_qty,
    merge_balance_observation,
)
from state_stores.venue_balances_state import (
    INTERNAL_BALANCE_SCRATCH_KEYS,
    strip_internal_balance_scratch_keys,
)
from providers.quantity import resolve_assets, venue_fee_cap_quantity
from providers.venue_metadata import ProviderVenueMetadataMixin
from state_io import (
    generation_file_lock,
    read_json_dict,
    read_json_dict_recovering as _default_read_json_cached,
    read_json_file,
    read_json_recovering,
    read_tail_jsonl_dicts,
    save_json_state as _default_save_json_state,
)

read_json_cached = _default_read_json_cached
save_json_state = _default_save_json_state

logger = logging.getLogger("intelligence.sentiment.portfolio_collector")

_resolve_read_json_cached = resolve_read_json_cached
FRESH_BALANCE_CACHE_SYNC_MULTIPLIER: float = 2.5
FRESH_OPEN_ORDERS_CACHE_SYNC_MULTIPLIER: float = 4.0
FRESH_BALANCE_CACHE_MAX_AGE_SEC: Optional[float] = None
FRESH_OPEN_ORDERS_CACHE_MAX_AGE_SEC: Optional[float] = None


def resolve_fresh_balance_cache_max_age_sec() -> float:
    """Derive Layer 3 balance cache freshness ceiling formulaically from ``BALANCE_SYNC_INTERVAL_SEC``."""
    if FRESH_BALANCE_CACHE_MAX_AGE_SEC is not None:
        return safe_pos_float(FRESH_BALANCE_CACHE_MAX_AGE_SEC, default=None) or 0.0
    ensure_cacheprice_classes_bound()
    base = safe_pos_float(getattr(_mc, "BALANCE_SYNC_INTERVAL_SEC", None), default=None)
    if base is None:
        raise ValueError("BALANCE_SYNC_INTERVAL_SEC must be a positive finite number")
    return round(FRESH_BALANCE_CACHE_SYNC_MULTIPLIER * base, 3)


def resolve_fresh_open_orders_cache_max_age_sec() -> float:
    """Derive Layer 3 open-orders cache freshness ceiling formulaically from ``OPEN_ORDER_SYNC_INTERVAL_SEC``."""
    if FRESH_OPEN_ORDERS_CACHE_MAX_AGE_SEC is not None:
        return safe_pos_float(FRESH_OPEN_ORDERS_CACHE_MAX_AGE_SEC, default=None) or 0.0
    ensure_cacheprice_classes_bound()
    base = safe_pos_float(getattr(_mc, "OPEN_ORDER_SYNC_INTERVAL_SEC", None), default=None)
    if base is None:
        raise ValueError("OPEN_ORDER_SYNC_INTERVAL_SEC must be a positive finite number")
    return round(FRESH_OPEN_ORDERS_CACHE_SYNC_MULTIPLIER * base, 3)


def _resolve_order_cache_files() -> Tuple[Tuple[str, str], ...]:
    ensure_cacheprice_classes_bound()
    contracts_fn = getattr(_mc.CacheAccountManagerBase, "venue_order_and_trade_cache_contracts", None)
    if callable(contracts_fn):
        return tuple(contracts_fn())
    ord_cls = _mc.CacheOrderFilledManager
    return tuple((fname, v_key) for v_key, fname in ord_cls.VENUE_FILENAMES.items())


def _read_venue_filled_orders_doc(
    cachedb_dir: str,
    order_cache_name: str,
    cache_venue: str,
    read_json_fn: Any,
) -> Dict[str, Any]:
    """Read a venue's filled-order cache snapshot via ``CacheOrderFilledManager``."""
    ensure_cacheprice_classes_bound()
    ord_cls = _mc.CacheOrderFilledManager
    if ord_cls is None:
        return {}
    items, ft = ord_cls.read_venue_filled_orders_snapshot(
        os.path.join(cachedb_dir, order_cache_name),
        venue=cache_venue,
        read_json_fn=read_json_fn,
        allow_legacy=True,
        allow_trade_fallback=True,
    )
    return {"items": items, "fetchtime": ft} if (items or ft) else {}


def _read_cached_open_orders_by_venue_sym(
    cache_dir: Optional[str],
    read_json_fn: Any = None,
    *,
    out_latest_ts: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[Tuple[str, str], List[Dict[str, Any]]], Dict[str, float], Dict[Tuple[str, str], float]]:
    """Passively read open orders from ``cache_order_open_<venue>.json`` via ``UnifiedOrderBookEvidence`` SSOT."""
    if not cache_dir:
        return {}, {}, {}
    ensure_cacheprice_classes_bound()
    open_cls = _mc.CacheOrderOpenManager
    if open_cls is None:
        return {}, {}, {}
    _rj = _resolve_read_json_cached(read_json_fn) if read_json_fn is not None else None
    loader_fn = getattr(open_cls, "load_unified_evidence", None)
    if callable(loader_fn):
        ev = loader_fn(
            cache_dir,
            read_json_fn=_rj,
            symbol_normalizer=lambda raw_s: str(raw_s or "").strip().lower(),
            include_filled=False,
        )
        if isinstance(out_latest_ts, dict):
            out_latest_ts.update(getattr(ev, "open_latest_ts_by_venue", None) or {})
        return (
            ev.open_orders_by_venue_sym(),
            dict(ev.verified_by_venue),
            {(v_k, s_k.upper()): s_ts for (v_k, s_k), s_ts in ev.symbol_verified_ts.items()},
        )
    by_vs: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    verified_by_venue: Dict[str, float] = {}
    verified_by_vs: Dict[Tuple[str, str], float] = {}
    for venue_key, fname in open_cls.VENUE_FILENAMES.items():
        cand_path = os.path.join(cache_dir, fname)
        items, ft, ver_ts, sym_ver = open_cls.read_venue_open_orders_snapshot_detailed(cand_path, read_json_fn=_rj)
        latest_v = ver_ts if ver_ts > 0.0 else 0.0
        if ver_ts > 0.0:
            verified_by_venue[venue_key] = ver_ts
        for sym_k, s_ts in sym_ver.items():
            if s_ts > 0.0:
                verified_by_vs[(venue_key, str(sym_k).strip().upper())] = s_ts
                if s_ts > latest_v:
                    latest_v = s_ts
        for ft_v in (ft or {}).values():
            ft_s = normalize_epoch_sec(ft_v, default=0.0) or 0.0
            if ft_s > latest_v:
                latest_v = ft_s
        if isinstance(out_latest_ts, dict) and latest_v > 0.0:
            out_latest_ts[venue_key] = latest_v
        for sym_k, rows in items.items():
            if not isinstance(rows, list):
                continue
            by_vs[(venue_key, str(sym_k).strip().upper())] = [
                {"venue": venue_key, "symbol": str(r.get("symbol") or sym_k), **r}
                for r in rows
                if isinstance(r, dict)
            ]
    return by_vs, verified_by_venue, verified_by_vs


def _iter_audit_dir_file_batches(
    cache_dir: Optional[str], *, max_files: int = 2, newest_first: bool = True
) -> List[List[str]]:
    """Yield sorted batches of `execution_audit_*.jsonl` file paths from candidate logger/cache directories."""
    if not cache_dir:
        return []
    abs_cache = os.path.abspath(cache_dir)
    batches: List[List[str]] = []
    seen_dirs: set[str] = set()
    for d in (
        cache_dir,
        os.path.join(os.path.dirname(abs_cache), "logger", "execution_audit"),
        os.path.join(os.path.dirname(os.path.dirname(abs_cache)), "logger", "execution_audit"),
    ):
        abs_d = os.path.abspath(d)
        if abs_d in seen_dirs or not os.path.isdir(d):
            continue
        seen_dirs.add(abs_d)
        try:
            files = sorted(
                (os.path.join(d, f) for f in os.listdir(d) if f.startswith("execution_audit") and f.endswith(".jsonl")),
                reverse=newest_first,
            )
            batch = files[:max_files] if newest_first else files[-max_files:]
            if batch:
                batches.append(batch)
        except Exception:
            continue
    return batches


def _normalize_open_orders_list(
    raw_orders: Any,
    venue: str,
    symbol: str,
    known_ai_order_ids: Optional[set[str]] = None,
    *,
    audit_origin_map: Optional[Dict[str, tuple[Optional[str], Optional[str]]]] = None,
    now_ts: Optional[float] = None,
    order_ts_map: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    if not isinstance(raw_orders, list):
        return []
    eff_symbol = canonical_display_symbol(symbol, venue=venue) or symbol
    out: List[Dict[str, Any]] = []
    for oo in raw_orders:
        if not isinstance(oo, dict):
            continue
        oo_norm = normalize_open_order(
            oo,
            venue=venue,
            symbol=eff_symbol,
            known_ai_order_ids=known_ai_order_ids,
            audit_origin_map=audit_origin_map,
            now_ts=now_ts,
            order_ts_map=order_ts_map,
        )
        if isinstance(oo_norm, dict):
            if oo_s := canonical_display_symbol(oo_norm.get("symbol") or eff_symbol, venue=venue):
                oo_norm["symbol"] = oo_s
            out.append(oo_norm)
    return out


def _extract_cached_venue_held_qty(
    vb_doc: Any,
    venue: str,
    sym_upper: str,
    cand_syms: Sequence[str],
    base_upper: Optional[str] = None,
) -> Optional[float]:
    if not isinstance(vb_doc, dict) or not vb_doc or not venue:
        return None
    for cand in dict.fromkeys(str(c).strip().upper() for c in cand_syms if str(c).strip()):
        if isinstance(p_d := casefold_get(vb_doc, f"pos:{venue}:{cand}"), dict):
            if (held_q := extract_position_held_qty(p_d)) is not None:
                return held_q
    for fb_key in (sym_upper, base_upper or ""):
        if fb_key and (raw_b := safe_nonneg_float(vb_doc.get(f"{venue}:{fb_key}"), default=None)) is not None:
            return raw_b
    return None


def _load_cached_ai_intents_list(cache_dir: Optional[str], read_json_fn: Any = None) -> List[Dict[str, Any]]:
    """Load persisted AI trade intents via ``providers.intent_state`` SSOT across ``state/`` and root cache paths."""
    if not cache_dir:
        return []
    _rj = _resolve_read_json_cached(read_json_fn)
    custom_reader = _rj if _rj not in (_default_read_json_cached, read_json_recovering) else None
    return load_durable_intents_from_cachedb(cache_dir, read_json_fn=custom_reader)


def _resolve_ai_intents_and_known_order_ids(
    cache_dir: Optional[str],
    *,
    ai_intents: Optional[List[Dict[str, Any]]] = None,
    known_ai_order_ids: Optional[set[str]] = None,
    read_json_fn: Any = None,
    need_intents: bool = False,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[set[str]]]:
    """Resolve cached AI trade intents and ``known_ai_order_ids`` without redundant disk reads."""
    if ai_intents is None and (need_intents or known_ai_order_ids is None) and cache_dir:
        cached_intents = _load_cached_ai_intents_list(cache_dir, read_json_fn)
        if cached_intents:
            ai_intents = cached_intents
        if known_ai_order_ids is None:
            known_ai_order_ids = collect_known_ai_order_ids(ai_intents or cached_intents)
    if known_ai_order_ids is None and ai_intents is not None:
        known_ai_order_ids = collect_known_ai_order_ids(ai_intents)
    return ai_intents, known_ai_order_ids


def _load_open_order_timestamps(cache_dir: Optional[str], cached_bals: Dict[str, Any]) -> Dict[str, float]:
    """Build order_id / client_order_id -> created_ts (epoch seconds) lookup for resting open orders."""
    out: Dict[str, float] = {}

    def _record_open_rows(rows: Any) -> None:
        if not isinstance(rows, list):
            return
        for oo_prev in rows:
            if isinstance(oo_prev, dict):
                c_ts = first_epoch_sec(
                    oo_prev.get("created_ts"),
                    oo_prev.get("time"),
                    oo_prev.get("timestamp"),
                    extract_timestamp_s(oo_prev),
                    allow_iso=True,
                )
                if c_ts is not None and c_ts > 0.0:
                    for id_k in ("order_id", "orderId", "client_order_id", "clientOrderId"):
                        if kv := clean_order_id(oo_prev.get(id_k)):
                            out.setdefault(kv, c_ts)

    if cache_dir:
        by_vs, _, _ = _read_cached_open_orders_by_venue_sym(cache_dir)
        for oo_rows in by_vs.values():
            _record_open_rows(oo_rows)
    for k, v in (cached_bals or {}).items():
        if isinstance(k, str) and k.startswith("open_orders:") and isinstance(v, list):
            _record_open_rows(v)
    if not cache_dir:
        return out
    for it in _load_cached_ai_intents_list(cache_dir):
        it_ts = first_epoch_sec(
            it.get("executed_ts"),
            it.get("executed_at"),
            it.get("timestamp"),
            it.get("created_at"),
            it.get("ts"),
            allow_iso=True,
        )
        if it_ts is not None:
            exec_pay = it.get("execution_payload") if isinstance(it.get("execution_payload"), dict) else {}
            for src_d in (it, exec_pay):
                for id_k in ("order_id", "orderId", "client_order_id", "clientOrderId", "intent_id"):
                    if kv := clean_order_id(src_d.get(id_k)):
                        out.setdefault(kv, it_ts)
            if oid_pay := extract_order_id(exec_pay, default=""):
                out.setdefault(str(oid_pay), it_ts)
            for cid_v in intent_venue_client_order_ids(it):
                if cid_v:
                    out.setdefault(cid_v, it_ts)
    for files in _iter_audit_dir_file_batches(cache_dir, max_files=2, newest_first=False):
        found_in_batch = False
        for fpath in files:
            for rec in read_tail_jsonl_dicts(fpath, limit=None, max_bytes=512 * 1024):
                if (rec_ts := normalize_epoch_sec(rec.get("ts"), default=None, allow_iso=True)) is not None:
                    for id_k in ("order_id", "orderId", "client_order_id", "clientOrderId"):
                        if kv := clean_order_id(rec.get(id_k)):
                            out.setdefault(kv, rec_ts)
                            found_in_batch = True
        if found_in_batch:
            break
    return out


def _load_recent_audit_origins(cache_dir: str) -> Dict[str, tuple[Optional[str], Optional[str]]]:
    """Load recent orderId/clientOrderId -> (module, rule) mappings from OrderLedger SSoT (legacy fallback only when ledger absent)."""
    if cache_dir:
        ledger_reader = OrderLedgerReader(cachedb_dir=cache_dir)
        if ledger_reader.list_event_files() or ledger_reader.list_closed_files():
            return ledger_reader.load_recent_order_origins(max_days=3)
    out: Dict[str, tuple[Optional[str], Optional[str]]] = {}
    for files in _iter_audit_dir_file_batches(cache_dir, max_files=2, newest_first=True):
        found_legacy = False
        for fpath in files:
            for rec in read_tail_jsonl_dicts(fpath, limit=None, max_bytes=1024 * 1024):
                _, mod, rule = resolve_order_origin(rec)
                if mod or rule:
                    v_norm = normalize_venue_name(rec.get("venue"))
                    for k in ("order_id", "orderId", "client_order_id", "clientOrderId"):
                        if kv := str(rec.get(k) or "").strip():
                            prev_mod, prev_rule = out.get(kv, (None, None))
                            out[kv] = (prev_mod or mod, prev_rule or rule)
                            found_legacy = True
                            if k in ("client_order_id", "clientOrderId") and v_norm:
                                try:
                                    if nat_kv := derive_venue_128bit_client_order_id(v_norm, kv):
                                        p_m, p_r = out.get(nat_kv, (None, None))
                                        out[nat_kv] = (p_m or mod, p_r or rule)
                                except Exception:
                                    pass
        if found_legacy:
            break
    return out


def _load_audit_terminal_orders(cache_dir: str, venue: str) -> Dict[str, List[Dict[str, Any]]]:
    """Synthesize deduplicated per-symbol filled order lists from OrderLedger SSoT (legacy fallback only when ledger absent)."""
    by_sym: Dict[str, List[Dict[str, Any]]] = {}
    if not cache_dir:
        return by_sym
    ledger_reader = OrderLedgerReader(cachedb_dir=cache_dir)
    if ledger_reader.list_closed_files():
        return ledger_reader.load_terminal_filled_orders(venue)
    v_target = normalize_venue_name(venue)
    for audit_files in _iter_audit_dir_file_batches(cache_dir, max_files=10, newest_first=False):
        try:
            submit_meta_by_oid: Dict[str, Dict[str, Any]] = {}
            idx_by_oid: Dict[tuple[str, str], int] = {}
            for af in audit_files:
                for rec in read_tail_jsonl_dicts(af, limit=None, max_bytes=1024 * 1024):
                    if normalize_venue_name(rec.get("venue")) != v_target:
                        continue
                    ev_name = str(rec.get("event") or "")
                    oid_s = str(rec.get("order_id") or "").strip()
                    ts_s = normalize_epoch_sec(rec.get("ts"), default=0.0, allow_iso=True) or 0.0
                    if ev_name in ("submit_accepted", "order_recovered") and oid_s:
                        meta_d = submit_meta_by_oid.setdefault(oid_s, {})
                        if ts_s > 0.0:
                            meta_d.setdefault("ts", ts_s)
                        if side_sub := (extract_order_side(rec) or str(rec.get("side") or "").strip().upper()):
                            meta_d.setdefault("side", side_sub)
                        if cid_sub := rec.get("client_order_id"):
                            meta_d.setdefault("client_order_id", cid_sub)
                        if (px_sub := first_pos_float(rec.get("price"), rec.get("reference_price"), default=None)) is not None:
                            meta_d.setdefault("price", px_sub)
                        continue
                    if ev_name not in ("order_terminal", "order_terminal_retry", "order_status"):
                        continue
                    v_stat, l_stat, _, f_q_opt, _, cost_opt, fallback_px, is_fill = _extract_audit_fill_metrics(
                        rec, fallback_ref_price=True
                    )
                    if ev_name == "order_status" and not is_terminal_audit_status(v_stat, l_stat):
                        continue
                    f_q = safe_pos_float(f_q_opt, default=0.0) or 0.0
                    if not is_fill or f_q <= 0.0:
                        continue
                    raw_sym_t = str(rec.get("symbol") or "").strip()
                    sym_t = canonical_display_symbol(raw_sym_t, venue=venue) or raw_sym_t.upper()
                    meta = submit_meta_by_oid.get(oid_s, {}) if oid_s else {}
                    vwap_px = round(cost_opt / f_q, 6) if cost_opt is not None and cost_opt > 0.0 else None
                    px_f = vwap_px or fallback_px or meta.get("price")
                    eff_ts_s = meta.get("ts") or ts_s
                    side_t = extract_order_side(rec) or str(rec.get("side") or "").strip().upper() or meta.get("side", "")
                    cid_val = rec.get("client_order_id") or meta.get("client_order_id")
                    if sym_t and px_f is not None and px_f > 0 and eff_ts_s > 0 and side_t in ("BUY", "SELL"):
                        row_d = {
                            "symbol": sym_t,
                            "orderId": rec.get("order_id"),
                            "clientOrderId": cid_val,
                            "side": side_t,
                            "price": px_f,
                            "qty": f_q,
                            "executedQty": f_q,
                            "timestamp": round(eff_ts_s * 1000.0),
                            "status": "FILLED",
                        }
                        dedup_k = oid_s or str(cid_val or "").strip()
                        sym_list = by_sym.setdefault(sym_t, [])
                        if dedup_k and (sym_t, dedup_k) in idx_by_oid:
                            existing = sym_list[idx_by_oid[(sym_t, dedup_k)]]
                            new_q = max(safe_pos_float(existing.get("qty"), default=0.0) or 0.0, f_q)
                            existing["qty"] = new_q
                            existing["executedQty"] = new_q
                            existing["price"] = px_f
                            if cid_val and not existing.get("clientOrderId"):
                                existing["clientOrderId"] = cid_val
                        else:
                            if dedup_k:
                                idx_by_oid[(sym_t, dedup_k)] = len(sym_list)
                            sym_list.append(row_d)
            if by_sym:
                break
        except Exception:
            pass
    return by_sym


_QUOTE_CASH_LOCK_ASSETS: frozenset[str] = frozenset({"USD", "USDC", "USDT", "USDG", "EUR"})


def _read_latest_filled_ts_by_venue(
    cache_dir: Optional[str],
    read_json_fn: Any = None,
) -> Dict[str, float]:
    """Passively resolve the newest recorded fill timestamp (epoch seconds) per venue from ``cache_order_filled_<venue>.json``."""
    if not cache_dir:
        return {}
    ensure_cacheprice_classes_bound()
    filled_cls = _mc.CacheOrderFilledManager
    if filled_cls is None:
        return {}
    _rj = _resolve_read_json_cached(read_json_fn) if read_json_fn is not None else None
    out: Dict[str, float] = {}
    for v_key, fname in filled_cls.VENUE_FILENAMES.items():
        cand_path = str(resolve_cachedb_filepath(os.path.join(cache_dir, fname)))
        try:
            items, _ = filled_cls.read_venue_filled_orders_snapshot(cand_path, venue=v_key, read_json_fn=_rj)
        except Exception:
            continue
        latest_s = 0.0
        for sym_rows in (items or {}).values():
            if not isinstance(sym_rows, list):
                continue
            for row in sym_rows:
                if not isinstance(row, dict):
                    continue
                r_ts = extract_timestamp_s(row) or 0.0
                if r_ts > latest_s:
                    latest_s = r_ts
        if latest_s > 0.0:
            out[v_key] = latest_s
    return out


def _venue_balance_cache_has_stale_order_lock(
    venue_norm: str,
    bal_verified_ts: float,
    all_cache_obs: Dict[tuple[str, str], BalanceObservation],
    cached_open_by_vs: Dict[Tuple[str, str], List[Dict[str, Any]]],
    cached_open_verified_ts: Dict[str, float],
    cached_open_sym_verified_ts: Dict[Tuple[str, str], float],
    *,
    cached_open_latest_ts: Optional[Dict[str, float]] = None,
    now_ts: Optional[float] = None,
) -> bool:
    """Return True when ``open_orders`` is strictly newer than ``bal_verified_ts`` and shows fewer locked funds/coins than ``cache_balance_<venue>``."""
    eff_now = normalize_epoch_sec(now_ts, default=None) if now_ts is not None else None
    bal_ver_ts = normalize_epoch_sec(bal_verified_ts, default=None)
    if bal_ver_ts is None or (eff_now is not None and bal_ver_ts > eff_now + 60.0):
        return True

    def _valid_open_ts(raw_ts: Any) -> float:
        ts_s = normalize_epoch_sec(raw_ts, default=None)
        if ts_s is None or (eff_now is not None and ts_s > eff_now + 60.0):
            return 0.0
        return ts_s

    open_ver_ts = max(
        _valid_open_ts(cached_open_verified_ts.get(venue_norm)),
        _valid_open_ts((cached_open_latest_ts or {}).get(venue_norm)),
    )
    for (ov, _), sts in cached_open_sym_verified_ts.items():
        if ov == venue_norm and (sf := _valid_open_ts(sts)) > open_ver_ts:
            open_ver_ts = sf
    if open_ver_ts <= bal_ver_ts + 0.5:
        return False
    open_quote_locked: Dict[str, float] = {}
    open_base_locked: Dict[str, float] = {}
    for (ov, osym), rows in cached_open_by_vs.items():
        if ov != venue_norm or not rows:
            continue
        b_raw, q_raw = resolve_assets(osym, None, None)
        b_u = canonical_balance_asset_key(b_raw, venue=venue_norm)
        q_u = str(q_raw or "").strip().upper() or MarketDataProvider.resolve_venue_default_quote(venue_norm, fallback="USD")
        for row in rows:
            if not isinstance(row, dict):
                continue
            side_u = extract_order_side(row)
            rem_q = extract_remaining_qty(row, default=0.0)
            if rem_q <= 1e-12:
                continue
            if side_u == "BUY":
                px_o = safe_pos_float(row.get("price"), default=0.0) or 0.0
                if px_o > 0.0:
                    open_quote_locked[q_u] = open_quote_locked.get(q_u, 0.0) + (rem_q * px_o)
            elif side_u == "SELL" and b_u:
                open_base_locked[b_u] = open_base_locked.get(b_u, 0.0) + rem_q
    for (bv, asset_k), obs in all_cache_obs.items():
        if bv != venue_norm or obs.reflected_locked is None or obs.reflected_locked <= 1e-6:
            continue
        if asset_k in _QUOTE_CASH_LOCK_ASSETS:
            if obs.reflected_locked > open_quote_locked.get(asset_k, 0.0) + 1.0:
                return True
        elif obs.reflected_locked > open_base_locked.get(asset_k, 0.0) + 1e-6:
            return True
    return False


def _resolve_effective_trailing_raw_peak(
    t_state: Any,
    prev_cached_pos: Any,
    *,
    px_f: float = 0.0,
    min_notional: float = 1.0,
) -> Optional[float]:
    """Resolve raw trailing peak without allowing a warmup-frozen daemon peak to regress a persisted position peak."""
    t_dict = t_state if isinstance(t_state, dict) else {}
    t_peak = safe_pos_float(t_dict.get("peak"), default=None)
    is_warming_up = (
        safe_pos_float(t_dict.get("warmup_at"), default=None) is not None
        and not _parse_bool_flag(t_dict.get("is_flat"), default=False)
    )
    if is_warming_up and isinstance(prev_cached_pos, dict):
        prev_pk = safe_pos_float(prev_cached_pos.get("trailing_peak"), default=None)
        if prev_pk is not None:
            prev_fb = safe_nonneg_float(
                prev_cached_pos.get("free_base")
                if prev_cached_pos.get("free_base") is not None
                else prev_cached_pos.get("free"),
                default=0.0,
            ) or 0.0
            prev_lb = safe_nonneg_float(prev_cached_pos.get("locked_base_qty"), default=0.0) or 0.0
            prev_held = prev_fb + prev_lb
            if prev_held > 1e-8 and (px_f <= 0.0 or prev_held * px_f >= max(1.0, min_notional)):
                return max(t_peak or 0.0, prev_pk)
    return t_peak


def _attach_trailing_peak_to_position(
    pos_entry: Dict[str, Any],
    v_norm: str,
    sym: str,
    px_f: float,
    fb_val: Optional[float],
    raw_peak: Any,
    pending_peaks_by_vs: Dict[tuple[str, str], tuple[float, Optional[float]]],
    *,
    fallback_dd: Any = None,
) -> None:
    """Attach trailing peak and drawdown to `pos_entry` when held base exceeds min notional, or defer to `pending_peaks_by_vs`."""
    for k in ("trailing_peak", "dd_from_peak_pct"):
        pos_entry.pop(k, None)
    peak_f = safe_pos_float(raw_peak, default=0.0) or 0.0
    if raw_peak is None or peak_f <= 0.0:
        return
    eff_peak = max(peak_f, px_f) if px_f > 0.0 else peak_f
    dd_pct = (round(((px_f - eff_peak) / eff_peak) * 100.0, 2) + 0.0) if px_f > 0.0 else fallback_dd
    min_notional_f = max(1.0, first_pos_float(pos_entry.get("exchange_min_notional_usd"), pos_entry.get("min_notional_usd"), default=1.0) or 1.0)
    has_held_base = fb_val is None or (fb_val > 1e-8 and (px_f <= 0.0 or fb_val * px_f >= min_notional_f))
    if has_held_base:
        pos_entry["trailing_peak"] = round(eff_peak, 6)
        pos_entry["dd_from_peak_pct"] = dd_pct
    elif v_norm and sym:
        pending_peaks_by_vs[(v_norm, str(sym).upper())] = (round(eff_peak, 6), dd_pct)


def _iter_valid_tradable_positions(
    tradable_positions: List[Dict[str, Any]],
) -> List[tuple[Dict[str, Any], str, str]]:
    out: List[tuple[Dict[str, Any], str, str]] = []
    for pos_item in tradable_positions:
        if isinstance(pos_item, dict):
            pv = normalize_venue_name(pos_item.get("venue"))
            ps = str(pos_item.get("symbol") or "").strip().upper()
            if pv and ps:
                out.append((pos_item, pv, ps))
    return out


_INTERNAL_BALANCE_TS_KEYS: tuple[str, ...] = tuple(sorted(INTERNAL_BALANCE_SCRATCH_KEYS))
_strip_internal_balance_ts = strip_internal_balance_scratch_keys
_merge_balance_observation = merge_balance_observation


def _build_cached_balance_observations(
    cached_bals: Dict[str, Any],
) -> Dict[tuple[str, str], BalanceObservation]:
    """Build atomic ``(venue, asset) -> BalanceObservation`` records via ``providers.portfolio_valuation`` SSOT."""
    return build_cached_balance_observations(
        cached_bals,
        resolve_assets_fn=resolve_assets,
        canonical_symbol_fn=canonical_display_symbol,
    )


def _build_cached_balance_index(
    cached_bals: Dict[str, Any],
) -> Dict[tuple[str, str], tuple[float, Optional[float]]]:
    """Build legacy ``(venue, asset) -> (free, ts)`` view backed by ``build_cached_balance_observations`` SSOT."""
    return {
        va_k: (obs.free, obs.ts)
        for va_k, obs in _build_cached_balance_observations(cached_bals).items()
    }


def _resolve_non_tradable_registry_symbols(cache_dir: Optional[str]) -> frozenset[str]:
    """Return upper-cased disabled or observe-only symbols from the workspace instrument registry."""
    ws_dir = resolve_workspace_dir_from_cachedb(cache_dir) if cache_dir else ""
    if not ws_dir:
        return frozenset()
    return disabled_registry_symbols(ws_dir) | observe_only_registry_symbols(ws_dir)


def _filter_non_tradable_and_dedup_positions(
    tradable_positions: List[Dict[str, Any]],
    *,
    inst_by_sym: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
) -> None:
    """Single-path position hygiene: drop non-tradable registry ghosts (unless explicitly loaded) and dedup (venue, symbol)."""
    non_tradable_syms = _resolve_non_tradable_registry_symbols(cache_dir)
    explicit_vs: set[tuple[str, str]] = set()
    if isinstance(inst_by_sym, dict):
        for inst_obj in inst_by_sym.values():
            iv = normalize_venue_name(str(getattr(inst_obj, "provider_name", "") or "").strip().lower())
            isym = str(getattr(inst_obj, "symbol", "") or "").strip().upper()
            if iv and isym:
                explicit_vs.add((iv, isym))
    kept: List[Dict[str, Any]] = []
    seen_vs: set[tuple[str, str]] = set()
    for pos in tradable_positions:
        if not isinstance(pos, dict):
            continue
        pv = normalize_venue_name(pos.get("venue"))
        ps_raw = str(pos.get("symbol") or "").strip()
        ps_u = ps_raw.upper()
        if ps_u and non_tradable_syms and ps_u in non_tradable_syms and (pv, ps_u) not in explicit_vs:
            continue
        if ps_u and (pv, ps_u) in seen_vs:
            continue
        if ps_u:
            seen_vs.add((pv, ps_u))
        kept.append(pos)
    tradable_positions[:] = kept


def _reconcile_position_quote_breakdowns(
    tradable_positions: List[Dict[str, Any]],
    venue_quote_cash: Dict[str, Dict[str, Any]],
    *,
    fresh_bal_cache_bd: Optional[Dict[tuple[str, str], Dict[str, float]]] = None,
    cached_bals: Optional[Dict[str, Any]] = None,
) -> None:
    """Single-path quote breakdown normalizer shared by live harvest and cache recovery.

    Refreshes ``free_quote_breakdown`` from the freshest available breakdown source and drops any
    breakdown whose sum diverges >5% from the finalized runtime ``free_quote`` (e.g., stale cached
    sub-balances or synthetic open-order lock deductions).
    """
    fresh_bd_map = fresh_bal_cache_bd if isinstance(fresh_bal_cache_bd, dict) else {}
    cb_map = cached_bals if isinstance(cached_bals, dict) else {}
    cash_bd_by_vq: Dict[tuple[str, str], Dict[str, Any]] = {}
    for cash_val in venue_quote_cash.values():
        if isinstance(cash_val, dict) and isinstance(bd_c := cash_val.get("breakdown"), dict) and bd_c:
            cv_n = normalize_venue_name(cash_val.get("venue"))
            cq_u = str(cash_val.get("asset") or "").strip().upper()
            if cv_n and cq_u:
                cash_bd_by_vq[(cv_n, cq_u)] = bd_c
    def _resolve_clean_quote_bd(
        v_n: str, q_u: str, free_val: Optional[float], fallback_bd: Any
    ) -> Optional[Dict[str, float]]:
        bd_src = (
            fresh_bd_map.get((v_n, q_u))
            or cash_bd_by_vq.get((v_n, q_u))
            or (
                cb_map.get(f"breakdown:{v_n}:{q_u}")
                if isinstance(cb_map.get(f"breakdown:{v_n}:{q_u}"), dict)
                else None
            )
            or (
                cb_cash.get("breakdown")
                if isinstance(cb_cash := cb_map.get(f"cash:{v_n}:{q_u}"), dict)
                and isinstance(cb_cash.get("breakdown"), dict)
                else None
            )
            or (fallback_bd if isinstance(fallback_bd, dict) else None)
        )
        if not isinstance(bd_src, dict) or not bd_src:
            return None
        clean_bd = {
            str(bk): round(bv_f, 2)
            for bk, bv in bd_src.items()
            if (bv_f := safe_pos_float(bv, default=None)) is not None
        }
        if not clean_bd:
            return None
        if free_val is not None:
            bd_sum = sum(clean_bd.values())
            if bd_sum > 0.0 and abs(bd_sum - free_val) / max(bd_sum, free_val, 1.0) > 0.05:
                return None
        return clean_bd

    for cash_val in venue_quote_cash.values():
        if not isinstance(cash_val, dict):
            continue
        cv_n = normalize_venue_name(cash_val.get("venue"))
        cq_u = str(cash_val.get("asset") or "").strip().upper()
        if not (cv_n and cq_u):
            continue
        c_free = safe_nonneg_float(cash_val.get("free"), default=None)
        clean_c_bd = _resolve_clean_quote_bd(cv_n, cq_u, c_free, cash_val.get("breakdown"))
        if clean_c_bd:
            cash_val["breakdown"] = clean_c_bd
        else:
            cash_val.pop("breakdown", None)

    for pos_entry in tradable_positions:
        if not isinstance(pos_entry, dict):
            continue
        pv = normalize_venue_name(pos_entry.get("venue"))
        ps_raw = str(pos_entry.get("symbol") or "").strip()
        _, res_q = resolve_assets(ps_raw, pos_entry.get("base"), pos_entry.get("quote"))
        pq_u = str(res_q or pos_entry.get("quote_asset") or "").strip().upper()
        fq_val = safe_nonneg_float(pos_entry.get("free_quote"), default=None)
        clean_bd = _resolve_clean_quote_bd(pv, pq_u, fq_val, pos_entry.get("free_quote_breakdown"))
        if clean_bd:
            pos_entry["free_quote_breakdown"] = clean_bd
        else:
            pos_entry.pop("free_quote_breakdown", None)


def derive_runtime_portfolio_view(
    open_orders_list: List[Dict[str, Any]],
    tradable_positions: List[Dict[str, Any]],
    venue_quote_cash: Dict[str, Dict[str, Any]],
    inst_by_sym: Dict[str, Any],
    prices_and_trends: Dict[str, Any],
    updated_bals: Dict[str, Any],
    pending_peaks_by_vs: Dict[tuple[str, str], tuple[float, Optional[float]]],
    *,
    now_ts_oo: float,
    gateway_max_usd: float,
    gateway_min_usd: float = 0.0,
    cache_dir: Optional[str] = None,
    fresh_bal_cache_bd: Optional[Dict[tuple[str, str], Dict[str, float]]] = None,
    cached_bals: Optional[Dict[str, Any]] = None,
) -> None:
    """Derive runtime/prompt adjusted balances and limits from raw observations + open orders without mutating persisted raw free balances."""
    _filter_non_tradable_and_dedup_positions(
        tradable_positions, inst_by_sym=inst_by_sym, cache_dir=cache_dir
    )
    def _strip_all_internal_ts() -> None:
        for p_cl in tradable_positions:
            if isinstance(p_cl, dict):
                for ts_k in INTERNAL_BALANCE_SCRATCH_KEYS:
                    p_cl.pop(ts_k, None)
        for c_cl in venue_quote_cash.values():
            if isinstance(c_cl, dict):
                for ts_k in INTERNAL_BALANCE_SCRATCH_KEYS:
                    c_cl.pop(ts_k, None)

    pos_by_vs: Dict[tuple[str, str], Dict[str, Any]] = {
        (pv, ps): pos_item for pos_item, pv, ps in _iter_valid_tradable_positions(tradable_positions)
    }

    def _valid_obs_ts(raw_ts: Any) -> Optional[float]:
        ts_s = normalize_epoch_sec(raw_ts, default=None)
        return ts_s if (ts_s is not None and ts_s <= now_ts_oo + 60.0) else None

    shared_obs_by_va: Dict[tuple[str, str], BalanceObservation] = {}
    for cash_val in venue_quote_cash.values():
        if isinstance(cash_val, dict):
            cv_n = normalize_venue_name(cash_val.get("venue"))
            cq_u = str(cash_val.get("asset") or "").strip().upper()
            if cv_n and cq_u:
                c_free = safe_nonneg_float(cash_val.get("free"), default=None)
                c_ts = _valid_obs_ts(cash_val.get("_balance_ts"))
                c_prev_lq = safe_nonneg_float(cash_val.get("_prev_locked_quote_usd"), default=None)
                if (
                    m_c := merge_balance_observation(
                        shared_obs_by_va.get((cv_n, cq_u)), c_free, c_ts, 2, reflected_locked=c_prev_lq
                    )
                ) is not None:
                    shared_obs_by_va[(cv_n, cq_u)] = m_c
    for pos_item, pv, ps in _iter_valid_tradable_positions(tradable_positions):
        p_res_b, p_res_q = resolve_assets(ps, pos_item.get("base"), pos_item.get("quote"))
        p_b_u = canonical_balance_asset_key(p_res_b, venue=pv)
        p_q_u = str(p_res_q or pos_item.get("quote_asset") or "").strip().upper()
        fallback_ts = _valid_obs_ts(pos_item.get("_balance_ts"))
        if p_b_u:
            fb_v = safe_nonneg_float(pos_item.get("free_base"), default=None)
            b_ts = _valid_obs_ts(pos_item.get("_base_balance_ts")) or fallback_ts
            p_prev_lb = safe_nonneg_float(pos_item.get("_prev_locked_base_qty"), default=None)
            if (
                m_b := merge_balance_observation(
                    shared_obs_by_va.get((pv, p_b_u)), fb_v, b_ts, 1, reflected_locked=p_prev_lb
                )
            ) is not None:
                shared_obs_by_va[(pv, p_b_u)] = m_b
        if p_q_u:
            fq_v = safe_nonneg_float(pos_item.get("free_quote"), default=None)
            q_ts = _valid_obs_ts(pos_item.get("_quote_balance_ts")) or fallback_ts
            p_prev_lq = safe_nonneg_float(pos_item.get("_prev_locked_quote_usd"), default=None)
            if (
                m_q := merge_balance_observation(
                    shared_obs_by_va.get((pv, p_q_u)), fq_v, q_ts, 1, reflected_locked=p_prev_lq
                )
            ) is not None:
                shared_obs_by_va[(pv, p_q_u)] = m_q

    locked_quote_by_vq: Dict[tuple[str, str], float] = {}
    locked_base_by_vb: Dict[tuple[str, str], float] = {}
    locked_quote_by_vs: Dict[tuple[str, str], float] = {}
    locked_base_by_vs: Dict[tuple[str, str], float] = {}
    pre_bal_locked_by_va: Dict[tuple[str, str], float] = {}
    pre_bal_locked_quote_by_vq: Dict[tuple[str, str], float] = {}
    pre_bal_locked_quote_by_vs: Dict[tuple[str, str], float] = {}
    pre_bal_locked_base_by_vs: Dict[tuple[str, str], float] = {}
    for oo_item in open_orders_list or ():
        if not isinstance(oo_item, dict):
            continue
        oo_v = normalize_venue_name(oo_item.get("venue"))
        oo_sym = str(oo_item.get("symbol") or "").strip().upper()
        oo_side = str(oo_item.get("side") or "").strip().upper()
        oo_px = safe_pos_float(oo_item.get("price"), default=0.0) or 0.0
        oo_rem = safe_pos_float(oo_item.get("remaining_qty"), default=None) or safe_pos_float(oo_item.get("qty"), default=0.0) or 0.0
        explicit_oo_ts = _valid_obs_ts(oo_item.get("created_ts"))
        if oo_item.get("created_ts") is None:
            oo_item["created_ts"] = round(now_ts_oo, 3)
            oo_item["age_min"] = 0.0
            oo_item["age"] = "0.0m"
        if oo_px > 0.0 and oo_sym and oo_item.get("dist_from_spot_pct") is None:
            oo_lookup_sym = ensure_equity_suffix(oo_sym, is_equity=is_equity_venue(oo_v))
            spot_oo = safe_pos_float((prices_and_trends.get(oo_sym) or {}).get("price"), default=0.0) or resolve_symbol_price(oo_lookup_sym, prices_and_trends)
            if spot_oo > 0.0:
                oo_item["dist_from_spot_pct"] = round(((oo_px - spot_oo) / spot_oo) * 100.0, 2)
        if not oo_v or not oo_sym or oo_rem <= 0.0:
            continue
        inst_oo = inst_by_sym.get(f"{oo_v}:{oo_sym}") or inst_by_sym.get(oo_sym)
        matched_pos = pos_by_vs.get((oo_v, oo_sym))
        oo_pos_sym = oo_sym
        if matched_pos is None:
            oo_canon = canonical_symbol_key(oo_sym)
            oo_base = canonical_base_asset(oo_sym)
            for (pv, ps), p_obj in pos_by_vs.items():
                if pv != oo_v:
                    continue
                if (oo_canon and canonical_symbol_key(ps) == oo_canon) or (
                    MarketDataProvider.uses_bare_base_order_symbols(oo_v) and oo_base and canonical_base_asset(ps) == oo_base
                ):
                    matched_pos = p_obj
                    oo_pos_sym = ps
                    if inst_oo is None:
                        inst_oo = inst_by_sym.get(f"{oo_v}:{ps}") or inst_by_sym.get(ps)
                    break
        inst_b = canonical_balance_asset_key(getattr(inst_oo, "base", "") or getattr(inst_oo, "base_asset", ""), venue=oo_v) if inst_oo is not None else ""
        inst_q = str(getattr(inst_oo, "quote", "") or getattr(inst_oo, "quote_asset", "") or "").strip().upper() if inst_oo is not None else ""
        pos_b = canonical_balance_asset_key((matched_pos or {}).get("base") or (matched_pos or {}).get("base_asset"), venue=oo_v)
        pos_q = str((matched_pos or {}).get("quote") or (matched_pos or {}).get("quote_asset") or "").strip().upper()
        parsed_b, parsed_q = resolve_assets(oo_pos_sym, None, None)
        oo_b_u = inst_b or pos_b or canonical_balance_asset_key(parsed_b, venue=oo_v)
        oo_q_u = inst_q or pos_q or str(parsed_q or "").strip().upper() or MarketDataProvider.resolve_venue_default_quote(oo_v, fallback="USD")
        if oo_side == "BUY" and oo_px > 0.0:
            locked_notional = oo_rem * oo_px
            locked_quote_by_vq[(oo_v, oo_q_u)] = locked_quote_by_vq.get((oo_v, oo_q_u), 0.0) + locked_notional
            locked_quote_by_vs[(oo_v, oo_pos_sym)] = locked_quote_by_vs.get((oo_v, oo_pos_sym), 0.0) + locked_notional
            q_obs = shared_obs_by_va.get((oo_v, oo_q_u))
            q_ts = q_obs.ts if q_obs is not None else None
            if explicit_oo_ts is not None and q_ts is not None and explicit_oo_ts > q_ts + 0.5:
                pre_bal_locked_by_va[(oo_v, oo_q_u)] = pre_bal_locked_by_va.get((oo_v, oo_q_u), 0.0) + locked_notional
                pre_bal_locked_quote_by_vq[(oo_v, oo_q_u)] = pre_bal_locked_quote_by_vq.get((oo_v, oo_q_u), 0.0) + locked_notional
                pre_bal_locked_quote_by_vs[(oo_v, oo_pos_sym)] = pre_bal_locked_quote_by_vs.get((oo_v, oo_pos_sym), 0.0) + locked_notional
        elif oo_side == "SELL":
            locked_base_by_vs[(oo_v, oo_pos_sym)] = locked_base_by_vs.get((oo_v, oo_pos_sym), 0.0) + oo_rem
            if oo_b_u:
                locked_base_by_vb[(oo_v, oo_b_u)] = locked_base_by_vb.get((oo_v, oo_b_u), 0.0) + oo_rem
            b_obs = shared_obs_by_va.get((oo_v, oo_b_u)) if oo_b_u else None
            b_ts = b_obs.ts if b_obs is not None else None
            if explicit_oo_ts is not None and b_ts is not None and explicit_oo_ts > b_ts + 0.5:
                pre_bal_locked_by_va[(oo_v, oo_b_u)] = pre_bal_locked_by_va.get((oo_v, oo_b_u), 0.0) + oo_rem
                pre_bal_locked_base_by_vs[(oo_v, oo_pos_sym)] = pre_bal_locked_base_by_vs.get((oo_v, oo_pos_sym), 0.0) + oo_rem

    cur_locked_by_va: Dict[tuple[str, str], float] = dict(locked_quote_by_vq)
    for vb_k, lb_v in locked_base_by_vb.items():
        cur_locked_by_va[vb_k] = cur_locked_by_va.get(vb_k, 0.0) + lb_v
    balance_records_by_va: Dict[tuple[str, str], BalanceSnapshotRecord] = {}
    adjusted_free_by_va: Dict[tuple[str, str], float] = {}
    for va_key in set(shared_obs_by_va) | set(cur_locked_by_va) | set(pre_bal_locked_by_va):
        obs_va = shared_obs_by_va.get(va_key)
        bal_rec = BalanceSnapshotRecord(
            free_raw=obs_va.free if obs_va is not None else 0.0,
            total_locked=cur_locked_by_va.get(va_key, 0.0),
            pre_balance_locked=pre_bal_locked_by_va.get(va_key, 0.0),
            reflected_locked=obs_va.reflected_locked if obs_va is not None else None,
        )
        balance_records_by_va[va_key] = bal_rec
        if obs_va is not None and bal_rec.synthetic_deduct > 0.0:
            adjusted_free_by_va[va_key] = bal_rec.runtime_free

    # Apply synthetic pre-balance deductions and cross-position shared balance reconciliation to runtime/prompt views
    # via BalanceSnapshotRecord; persisted ``updated_bals`` keeps raw free balances + venue-reflected locks for NAV.
    for cash_val in venue_quote_cash.values():
        if isinstance(cash_val, dict):
            cv_norm, cq_u = normalize_venue_name(cash_val.get("venue")), str(cash_val.get("asset") or "").strip().upper()
            va_c = (cv_norm, cq_u)
            if va_c in shared_obs_by_va and (c_rec_free := balance_records_by_va.get(va_c)) is not None:
                cash_val["free"] = round(c_rec_free.runtime_free, 2)
            if (lq_val := locked_quote_by_vq.get(va_c, 0.0)) > 0.0:
                cash_val["locked_quote_usd"] = round(lq_val, 2)
                c_rec = balance_records_by_va.get(va_c) or BalanceSnapshotRecord(free_raw=0.0, total_locked=lq_val)
                if pers_cash_fields := c_rec.to_persisted_cash_fields(
                    decimals=2,
                    quote_locked=lq_val,
                    pre_quote_locked=pre_bal_locked_quote_by_vq.get(va_c, 0.0),
                ):
                    cash_bkey = f"cash:{cv_norm}:{cq_u}"
                    if cash_bkey in updated_bals and isinstance(updated_bals[cash_bkey], dict):
                        updated_bals[cash_bkey] = {**updated_bals[cash_bkey], **pers_cash_fields}

    fee_spec_resolver = build_persisted_venue_fee_resolver(cache_dir, now_ts=now_ts_oo)
    for pos_entry in tradable_positions:
        p_v_norm = normalize_venue_name(pos_entry.get("venue"))
        p_sym_raw = str(pos_entry.get("symbol") or "").strip()
        p_sym_u = p_sym_raw.upper()
        p_res_b, p_res_q = resolve_assets(p_sym_raw, pos_entry.get("base"), pos_entry.get("quote"))
        p_base_u = canonical_balance_asset_key(p_res_b, venue=p_v_norm)
        p_quote_u = str(p_res_q or pos_entry.get("quote_asset") or "").strip().upper()
        balance_changed = False
        va_q = (p_v_norm, p_quote_u)
        if p_quote_u and va_q in shared_obs_by_va and (q_rec := balance_records_by_va.get(va_q)) is not None:
            if va_q in adjusted_free_by_va or "free_quote" in pos_entry:
                new_fq = round(q_rec.runtime_free, 2)
                prev_fq = safe_nonneg_float(pos_entry.get("free_quote"), default=None)
                if va_q in adjusted_free_by_va or prev_fq is None or abs(new_fq - prev_fq) > 1e-6:
                    pos_entry["free_quote"] = new_fq
                    balance_changed = True
        va_b = (p_v_norm, p_base_u)
        if p_base_u and va_b in shared_obs_by_va and (b_rec := balance_records_by_va.get(va_b)) is not None:
            if va_b in adjusted_free_by_va or "free_base" in pos_entry:
                adj_fb = round(b_rec.runtime_free, 8 if va_b in adjusted_free_by_va else 6)
                prev_fb = safe_nonneg_float(pos_entry.get("free_base"), default=None)
                if va_b in adjusted_free_by_va or prev_fb is None or abs(adj_fb - prev_fb) > 1e-9:
                    pos_entry["free_base"] = adj_fb
                    px_adj = safe_pos_float(pos_entry.get("price"), default=0.0) or 0.0
                    if px_adj > 0.0:
                        pos_entry["value_usd"] = round(adj_fb * px_adj, 2)
                    balance_changed = True
        lq_pos = locked_quote_by_vs.get((p_v_norm, p_sym_u), 0.0)
        lb_pos = locked_base_by_vs.get((p_v_norm, p_sym_u), 0.0)
        if balance_changed or lq_pos > 0.0 or lb_pos > 0.0:
            inst_obj = (
                inst_by_sym.get(f"{p_v_norm}:{p_sym_raw}")
                or inst_by_sym.get(f"{p_v_norm}:{p_sym_u}")
                or inst_by_sym.get(p_sym_raw)
                or inst_by_sym.get(p_sym_u)
            )
            px_pos = safe_pos_float(pos_entry.get("price"), default=0.0) or 0.0
            fb_pos = safe_nonneg_float(pos_entry.get("free_base"), default=None)
            static_prec = ProviderVenueMetadataMixin.prefers_static_precision(p_v_norm)
            prev_limit_keys = (
                (
                    "fee_tier",
                    "maker_fee_bps",
                    "taker_fee_bps",
                    "maker_rebate_edge_bps",
                    "exchange_min_notional_usd",
                    "spread_bps",
                )
                if static_prec
                else (
                    "fee_tier",
                    "maker_fee_bps",
                    "taker_fee_bps",
                    "maker_rebate_edge_bps",
                    "exchange_min_notional_usd",
                    "min_qty",
                    "step_size",
                    "tick_size",
                    "spread_bps",
                )
            )
            prev_inst_limits = (
                {k: pos_entry[k] for k in prev_limit_keys if pos_entry.get(k) is not None}
                if inst_obj is None
                else {}
            )
            pos_entry.update(
                resolve_venue_fee_and_limits(
                    str(pos_entry.get("venue") or ""),
                    p_sym_raw or p_sym_u,
                    inst_obj,
                    price=px_pos,
                    free_quote=safe_nonneg_float(pos_entry.get("free_quote"), default=None),
                    free_base=fb_pos,
                    locked_quote_usd=lq_pos if lq_pos > 0.0 else None,
                    locked_base_qty=lb_pos if lb_pos > 0.0 else None,
                    gateway_max_usd=gateway_max_usd,
                    gateway_min_usd=gateway_min_usd,
                    fee_spec_resolver=fee_spec_resolver,
                    fallback_limits=prev_inst_limits,
                )
            )
            if prev_inst_limits:
                pos_entry.update(prev_inst_limits)
            if lb_pos > 1e-8 and "trailing_peak" not in pos_entry and (pend_peak := pending_peaks_by_vs.get((p_v_norm, p_sym_u))) is not None:
                min_not_pos = max(1.0, first_pos_float(pos_entry.get("exchange_min_notional_usd"), pos_entry.get("min_notional_usd"), default=1.0) or 1.0)
                held_total = (fb_pos or 0.0) + lb_pos
                if px_pos <= 0.0 or held_total * px_pos >= min_not_pos:
                    pos_entry["trailing_peak"] = pend_peak[0]
                    pos_entry["dd_from_peak_pct"] = pend_peak[1]
            pos_bkey = f"pos:{p_v_norm}:{p_sym_raw}" if f"pos:{p_v_norm}:{p_sym_raw}" in updated_bals else f"pos:{p_v_norm}:{p_sym_u}"
            if pos_bkey in updated_bals and isinstance(updated_bals[pos_bkey], dict) and (lq_pos > 0.0 or lb_pos > 0.0 or pos_entry.get("trailing_peak") is not None):
                pers_pos = dict(updated_bals[pos_bkey])
                base_bal_rec = (
                    balance_records_by_va.get((p_v_norm, p_base_u))
                    if p_base_u
                    else None
                ) or BalanceSnapshotRecord(free_raw=0.0, total_locked=lb_pos)
                quote_bal_rec = balance_records_by_va.get((p_v_norm, p_quote_u)) if p_quote_u else None
                pers_pos.update(
                    base_bal_rec.to_persisted_position_fields(
                        symbol_locked_quote=lq_pos,
                        symbol_pre_quote=pre_bal_locked_quote_by_vs.get((p_v_norm, p_sym_u), 0.0),
                        quote_record=quote_bal_rec,
                        symbol_locked_base=lb_pos,
                        symbol_pre_base=pre_bal_locked_base_by_vs.get((p_v_norm, p_sym_u), 0.0),
                        price=px_pos,
                        trailing_peak=pos_entry.get("trailing_peak"),
                        dd_from_peak_pct=pos_entry.get("dd_from_peak_pct"),
                    )
                )
                updated_bals[pos_bkey] = pers_pos
    _reconcile_position_quote_breakdowns(
        tradable_positions,
        venue_quote_cash,
        fresh_bal_cache_bd=fresh_bal_cache_bd,
        cached_bals=cached_bals,
    )
    _strip_all_internal_ts()


_apply_open_orders_locked_balances = derive_runtime_portfolio_view


def _merge_audit_orders(primary_list: Any, audit_list: Any) -> List[Dict[str, Any]]:
    """Merge cached venue orders with terminal execution_audit orders, deduplicating by orderId or side/qty/timestamp."""
    base_rows = [r for r in (primary_list if isinstance(primary_list, list) else ()) if isinstance(r, dict)]
    extra_rows = [r for r in (audit_list if isinstance(audit_list, list) else ()) if isinstance(r, dict)]
    if not base_rows:
        return extra_rows
    if not extra_rows:
        return base_rows
    seen_oids = {str(oid) for r in base_rows if (oid := extract_order_id(r, default=""))}
    merged = list(base_rows)
    for ar in extra_rows:
        a_oid = str(extract_order_id(ar, default="") or "")
        if a_oid and a_oid in seen_oids:
            continue
        a_side = extract_order_side(ar)
        a_ts = (
            normalize_timestamp_ms(ar.get("timestamp"), default=0)
            or normalize_timestamp_ms(ar.get("time"), default=0)
            or 0
        )
        a_qty = extract_executed_qty(ar, default=0.0) or 0.0
        if a_ts > 0 and a_qty > 0.0 and any(
            extract_order_side(br) == a_side
            and abs(
                (
                    normalize_timestamp_ms(br.get("timestamp"), default=0)
                    or normalize_timestamp_ms(br.get("time"), default=0)
                    or 0
                )
                - a_ts
            ) <= 60000
            and abs((extract_executed_qty(br, default=0.0) or 0.0) - a_qty) <= max(1e-5, a_qty * 0.01)
            for br in base_rows
        ):
            continue
        if a_oid:
            seen_oids.add(a_oid)
        merged.append(ar)
    return merged


class PortfolioCacheCollectorBase(TelemetryCacheCollectorInterface, ABC):
    """Abstract base contract (Level 2) for account/portfolio domain cache collectors.

    Modeled on ``CacheAccountManagerBase(CacheManagerInterface, ABC)``: encapsulates
    shared order/fill cache bindings (`CacheOrderFilledManager`, `CacheOrderOpenManager`, `CacheTradeManager`,
    `CacheAssetValueManager`, and execution_audit fallback) and portfolio state readers.
    """

    cls_name = "PortfolioCacheCollectorBase"
    _on_collector_init = staticmethod(ensure_cacheprice_classes_bound)


class VenueTrailingMapsCollector(PortfolioCacheCollectorBase):
    """Concrete collector (Level 3) for Binance and Kraken trailing peak state maps."""

    cls_name = "VenueTrailingMapsCollector"
    COLLECTOR_ID = "portfolio_trailing_maps"

    def collect(
        self,
        cachedb_dir: str,
        symbol: str = "*",
        *,
        workspace_dir: str = "",
        read_json_fn: Any = None,
        **kwargs: Any,
    ) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Dict[str, Any]]]]:
        _rj = _resolve_read_json_cached(read_json_fn)
        trailing_map: Dict[str, Dict[str, Any]] = {}
        venue_list: list[str] = []
        for v_name in (
            *trailing_venues(include_all_24x7=False),
            *TrailingStateStore.discover_venue_names(cache_dir=cachedb_dir, workspace_dir=workspace_dir),
        ):
            v_norm = str(normalize_venue_name(v_name) or v_name or "").strip().lower()
            if v_norm and v_norm not in venue_list:
                venue_list.append(v_norm)
        trailing_by_venue: Dict[str, Dict[str, Dict[str, Any]]] = {v: {} for v in venue_list}
        for trail_venue in venue_list:
            raw_trail = TrailingStateStore.read_venue_state(
                trail_venue,
                cache_dir=cachedb_dir,
                workspace_dir=workspace_dir,
                read_json_fn=_rj,
            )
            if not isinstance(raw_trail, dict):
                continue
            default_quote = venue_default_quote(trail_venue)
            for k_asset, k_st in raw_trail.items():
                if not (k_raw := str(k_asset).strip()):
                    continue
                k_quoted = ensure_quote_suffix(k_raw, default_quote) if default_quote else k_raw
                k_key = canonical_display_symbol(k_quoted, venue=trail_venue) or k_quoted.upper()
                norm_st = k_st if isinstance(k_st, dict) and k_st.get("peak") is not None else ({"peak": k_pk} if (k_pk := safe_pos_float(k_st, default=0.0) or 0.0) > 0 else None)
                if norm_st is not None:
                    trailing_map.setdefault(k_key, norm_st)
                    trailing_by_venue[trail_venue].setdefault(k_key, norm_st)
        return trailing_map, trailing_by_venue


class VenueCyclesCollector(PortfolioCacheCollectorBase):
    """Concrete collector (Level 3) for Kraken, Hyperliquid, and Trading212 DCA `.state_*.json` cycles."""

    cls_name = "VenueCyclesCollector"
    COLLECTOR_ID = "portfolio_venue_cycles"

    def collect(
        self,
        cachedb_dir: str,
        symbol: str = "*",
        *,
        workspace_dir: str = "",
        prices_and_trends: Optional[Dict[str, Any]] = None,
        fallback_prices: Optional[Dict[str, float]] = None,
        read_json_fn: Any = None,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        eff_workspace = workspace_dir or cachedb_dir
        eff_pt = prices_and_trends or {}
        eff_fb = fallback_prices or {}
        _rj = _resolve_read_json_cached(read_json_fn)

        def _record_cyc(
            target: Dict[str, Any], st_path: str, sym: str, dec: int,
            *, is_eq: bool = False, direction: Optional[str] = None,
        ) -> None:
            if isinstance((st_data := _rj(st_path)), dict):
                if is_shadow_or_paper_state_payload(st_data):
                    return
                lookup_sym = ensure_equity_suffix(sym, is_equity=is_eq)
                parsed = parse_venue_cycle_state(
                    st_data, lookup_sym, eff_pt,
                    avg_decimals=dec, fallback_prices=eff_fb,
                    include_meta=True, direction=direction,
                )
                prev = target.get(sym)
                if isinstance(prev, dict):
                    prev_active = (safe_nonneg_float(prev.get("qty"), default=0.0) or 0.0) > 1e-8 or bool(prev.get("resting_orders"))
                    new_active = (safe_nonneg_float(parsed.get("qty"), default=0.0) or 0.0) > 1e-8 or bool(parsed.get("resting_orders"))
                    if prev_active and not new_active:
                        return
                target[sym] = parsed

        kraken_cycles: Dict[str, Any] = {}
        for st_path in iter_venue_cycle_state_files("kraken", workspace_dir=eff_workspace):
            raw_pair = parse_cycle_state_symbol(st_path)
            pair_name = resolve_registered_symbol_case(raw_pair, "kraken") or raw_pair.upper()
            if pair_name.isalnum() and symbol_quote_asset(pair_name) in ("USD", "EUR", "USDC", "USDT"):
                _record_cyc(kraken_cycles, st_path, pair_name, 6)
        xst_alloc_path = os.path.join(eff_workspace, "kraken", "xstock_state.json")
        if os.path.exists(xst_alloc_path) and isinstance((xst_st := _rj(xst_alloc_path)), dict):
            alloc_doc = xst_st.get("allocated")
            if isinstance(alloc_doc, dict):
                a_qty = safe_pos_float(alloc_doc.get("qty"), default=0.0) or 0.0
                a_px = safe_pos_float(alloc_doc.get("alloc_price"), default=0.0) or 0.0
                a_asset = str(alloc_doc.get("asset") or "").strip()
                a_base = canonical_base_asset(a_asset)
                a_pair = resolve_registered_symbol_case(ensure_xstock_pair(a_base), "kraken") if a_base else None
                known_a = xst_st.get("known_assets") if isinstance(xst_st.get("known_assets"), dict) else None
                live_a_qty = (
                    safe_nonneg_float(casefold_get(known_a, a_asset), default=0.0) or 0.0
                    if (known_a and a_asset)
                    else a_qty
                )
                a_live_px = resolve_symbol_price(a_pair, eff_pt, eff_fb) if a_pair else 0.0
                if (
                    a_pair
                    and live_a_qty > 1e-4
                    and a_px > 0.0
                    and not (a_live_px > 0.0 and live_a_qty * a_live_px < 1.0)
                    and a_pair not in kraken_cycles
                ):
                    kraken_cycles[a_pair] = parse_venue_cycle_state(
                        {
                            "cycle": 1,
                            "qty": live_a_qty,
                            "cost": live_a_qty * a_px,
                            "entry_price": a_px,
                            "last_buy_price": a_px,
                            "adopted": True,
                        },
                        a_pair,
                        eff_pt,
                        avg_decimals=6,
                        fallback_prices=eff_fb,
                        include_meta=True,
                    )

        hyperliquid_cycles: Dict[str, Any] = {}
        for st_path in iter_venue_cycle_state_files("hyperliquid", workspace_dir=eff_workspace):
            raw_tok = parse_cycle_state_symbol(st_path).upper()
            if raw_tok.startswith("DN_") or raw_tok == "DN":
                continue
            hl_dir: Optional[str] = None
            for sfx in ("_LONG", "_SHORT"):
                if raw_tok.endswith(sfx):
                    raw_tok = raw_tok[: -len(sfx)]
                    hl_dir = sfx[1:]
                    break
            if raw_tok.isalnum():
                _record_cyc(hyperliquid_cycles, st_path, ensure_quote_suffix(raw_tok, "USDC"), 6, direction=hl_dir)

        all_t212_cycles: Dict[str, Any] = {}
        for st_path, ticker in _iter_t212_state_files(eff_workspace):
            _record_cyc(all_t212_cycles, st_path, ticker, 4, is_eq=True)

        return kraken_cycles, hyperliquid_cycles, all_t212_cycles


class KrakenXstocksCollector(PortfolioCacheCollectorBase):
    """Concrete collector (Level 3) for Kraken read-only xStocks holdings and T212 sibling linkage."""

    cls_name = "KrakenXstocksCollector"
    COLLECTOR_ID = "portfolio_kraken_xstocks"

    def collect(
        self,
        cachedb_dir: str,
        symbol: str = "*",
        *,
        kraken_dir: str = "",
        prices_and_trends: Optional[Dict[str, Any]] = None,
        all_t212_cycles: Optional[Dict[str, Any]] = None,
        kraken_cycles: Optional[Dict[str, Any]] = None,
        read_json_fn: Any = None,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        eff_k_dir = kraken_dir or cachedb_dir
        eff_pt = prices_and_trends or {}
        eff_t212 = all_t212_cycles or {}
        _rj = _resolve_read_json_cached(read_json_fn)
        eff_cachedb = cachedb_dir
        if eff_k_dir and eff_cachedb == eff_k_dir:
            sibling_cachedb = os.path.join(os.path.dirname(os.path.normpath(eff_k_dir)), "cachedb")
            if os.path.isdir(sibling_cachedb):
                eff_cachedb = sibling_cachedb
        live_bal_assets: Dict[str, float] = {}
        live_pos_pairs: Dict[str, str] = {}
        if eff_cachedb and os.path.isdir(eff_cachedb):
            _, vb_state, _ = load_venue_balances_state(eff_cachedb, _rj)
            if isinstance(vb_state, dict) and vb_state:
                for bk, bv in vb_state.items():
                    if bk.startswith("pos:kraken:") and isinstance(bv, dict):
                        b_asset = str(bv.get("base") or "").strip()
                        b_qty = extract_position_held_qty(bv)
                        b_sym = str(bv.get("symbol") or bk[len("pos:kraken:") :]).strip()
                        if b_asset and b_qty is not None:
                            live_bal_assets[b_asset] = b_qty
                            if b_sym:
                                norm_b, is_tok = resolve_kraken_tokenized_pair(b_sym)
                                if is_tok and norm_b:
                                    live_pos_pairs[canonical_base_asset(norm_b)] = norm_b
                    elif bk.startswith("kraken:") and not isinstance(bv, (dict, list, bool)):
                        b_asset = bk[len("kraken:") :].strip()
                        b_qty = safe_nonneg_float(bv, default=None)
                        if b_asset and b_qty is not None and b_asset not in live_bal_assets:
                            live_bal_assets[b_asset] = b_qty
        cand_paths = [
            p for p in (os.path.join(eff_k_dir, "xstock_state.json"), os.path.join(eff_k_dir, ".xstock_state.json"))
            if os.path.exists(p)
        ]
        xst_data: Dict[str, Any] = {}
        if cand_paths:
            xst_path = max(cand_paths, key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0.0)
            raw_xst = _rj(xst_path)
            if isinstance(raw_xst, dict):
                xst_data = raw_xst
        if not xst_data and not live_bal_assets:
            return None
        xstock_cfg = load_key_value_conf(eff_k_dir, root_dir=eff_k_dir)
        state_known_assets = xst_data.get("known_assets") if isinstance(xst_data.get("known_assets"), dict) else {}
        known_assets = live_bal_assets if live_bal_assets else state_known_assets
        x_regex = (xstock_cfg.get("xstock_regex") or "").strip()
        def _holding_underlying(token: str) -> str:
            tok_norm, _ = resolve_kraken_tokenized_pair(token.strip())
            return canonical_base_asset(tok_norm or token.strip())

        def _holding_unit_px(hk: str) -> float:
            hu = _holding_underlying(hk)
            return (
                resolve_symbol_price(ensure_xstock_pair(hu), eff_pt)
                or resolve_symbol_price(ensure_equity_suffix(hu), eff_pt)
                or 0.0
            )

        x_holdings: Dict[str, float] = {}
        for ak, av in known_assets.items():
            ak_str, av_f = str(ak).strip(), safe_float(av, 0.0) or 0.0
            if av_f < 1e-4:
                continue
            is_xstock_tok = resolve_kraken_tokenized_pair(ak_str)[1]
            if not is_xstock_tok and x_regex:
                try:
                    is_xstock_tok = bool(re.search(x_regex, ak_str, re.IGNORECASE))
                except re.error:
                    is_xstock_tok = False
            if is_xstock_tok:
                unit_px = _holding_unit_px(ak_str)
                if unit_px > 0.0 and av_f * unit_px < 1.0:
                    continue
                x_holdings[ak_str] = round(av_f, 6)

        active_pair_by_underlying: Dict[str, Tuple[str, Optional[float]]] = {}
        for raw_spec in split_csv(xstock_cfg.get("xstock_active_pairs") or ""):
            raw_p, _, raw_c = raw_spec.partition(":")
            norm_p = resolve_kraken_tokenized_pair(raw_p.strip())[0] if raw_p.strip() else ""
            u_p = canonical_base_asset(norm_p) if norm_p else ""
            if norm_p and u_p:
                active_pair_by_underlying[u_p] = (norm_p, safe_pos_float(raw_c, default=None))

        alloc_info = xst_data.get("allocated") if isinstance(xst_data.get("allocated"), dict) else None
        raw_listed_pair = clean_upper_scalar(xst_data.get("pair"))
        state_listed_pair = resolve_kraken_tokenized_pair(raw_listed_pair)[0] if raw_listed_pair else None
        if not (x_holdings or alloc_info or state_listed_pair or active_pair_by_underlying):
            return None

        def _holding_rank_usd(hk: str, qty: float) -> float:
            return qty * (_holding_unit_px(hk) or 1.0)

        x_holdings = dict(sorted(x_holdings.items(), key=lambda kv: _holding_rank_usd(kv[0], kv[1]), reverse=True))
        held_underlyings: List[str] = []
        holdings_by_underlying: Dict[str, str] = {}
        for hk in x_holdings:
            hu = _holding_underlying(hk)
            if hu:
                holdings_by_underlying[hk] = hu
                if hu not in held_underlyings:
                    held_underlyings.append(hu)
        if isinstance(alloc_info, dict) and known_assets:
            alloc_asset = str(alloc_info.get("asset") or "").strip()
            alloc_u = _holding_underlying(alloc_asset) if alloc_asset else ""
            if alloc_u and _holding_unit_px(alloc_asset) > 0.0 and alloc_u not in held_underlyings:
                alloc_info = None
                if not (x_holdings or state_listed_pair or active_pair_by_underlying):
                    return None
        cfg_underlying = clean_upper_scalar(xstock_cfg.get("xstock_yahoo")) or ""
        pair_underlying = _holding_underlying(state_listed_pair) if state_listed_pair else ""
        # BY DESIGN: Primary underlying/t212_sibling always tracks the largest held xStock by USD notional
        # so an unheld newly-listed IPO ticker never overwrites an active funded xStock position.
        if held_underlyings:
            underlying = held_underlyings[0]
        elif pair_underlying:
            underlying = pair_underlying
        elif active_pair_by_underlying:
            underlying = next(iter(active_pair_by_underlying))
        else:
            underlying = cfg_underlying
        if not underlying:
            return None
        t212_sib = ensure_equity_suffix(underlying)
        active_pair_entry = active_pair_by_underlying.get(underlying)
        default_xpair = ensure_xstock_pair(underlying)
        inferred_listed_pair = (
            live_pos_pairs.get(underlying)
            or (default_xpair if resolve_symbol_price(default_xpair, eff_pt) > 0.0 else None)
        )
        listed_pair = (
            state_listed_pair
            if (state_listed_pair and pair_underlying == underlying)
            else (
                active_pair_entry[0]
                if active_pair_entry
                else (inferred_listed_pair if held_underlyings else state_listed_pair)
            )
        )
        cfg_alloc_px = safe_pos_float(xstock_cfg.get("xstock_alloc_price"), default=None)
        state_alloc_px = safe_pos_float(alloc_info.get("alloc_price"), default=None) if isinstance(alloc_info, dict) else None
        active_alloc_px = active_pair_entry[1] if active_pair_entry else None
        static_alloc_px = (
            (active_alloc_px or cfg_alloc_px or state_alloc_px)
            if (cfg_underlying and cfg_underlying != underlying)
            else (cfg_alloc_px or active_alloc_px or state_alloc_px)
        )
        primary_xpair = listed_pair or default_xpair
        primary_hold_tok, primary_hold_qty = next(
            ((hk, hq) for hk, hq in x_holdings.items() if holdings_by_underlying.get(hk) == underlying),
            ("", 0.0),
        )
        active_cycle_cost_px: Optional[float] = None
        if primary_hold_qty > 1e-4:
            if eff_cachedb:
                cache_pos = resolve_kraken_trade_cache_position(
                    primary_xpair,
                    eff_cachedb,
                    held_qty=primary_hold_qty,
                    read_json_fn=_rj,
                )
                if isinstance(cache_pos, dict):
                    if listed_pair is None and (cache_pos.get("fills_count") or 0) > 0:
                        listed_pair = primary_xpair
                    active_cycle_cost_px = safe_pos_float(cache_pos.get("avg_cost_px"), default=None)
            if active_cycle_cost_px is None:
                cyc_map = kraken_cycles if isinstance(kraken_cycles, dict) else {}
                cyc_entry = casefold_get(cyc_map, primary_xpair) or casefold_get(cyc_map, default_xpair)
                if isinstance(cyc_entry, dict) and (safe_nonneg_float(cyc_entry.get("qty"), default=0.0) or 0.0) > 1e-4:
                    active_cycle_cost_px = first_pos_float(
                        cyc_entry.get("avg_cost_px"), cyc_entry.get("entry_price"), cyc_entry.get("last_buy_price")
                    )
            if active_cycle_cost_px is None and eff_k_dir:
                st_pair_path = os.path.join(eff_k_dir, cycle_state_filename(primary_xpair))
                if os.path.exists(st_pair_path) and isinstance(st_pair_d := _rj(st_pair_path), dict):
                    st_q = safe_nonneg_float(st_pair_d.get("qty"), default=0.0) or 0.0
                    st_c = safe_nonneg_float(st_pair_d.get("cost"), default=0.0) or 0.0
                    if st_q > 1e-4:
                        active_cycle_cost_px = first_pos_float(
                            (st_c / st_q) if st_c > 0.0 else None,
                            st_pair_d.get("entry_price"),
                            st_pair_d.get("last_buy_price"),
                        )
        alloc_px = active_cycle_cost_px or static_alloc_px
        xpair_px = (
            resolve_symbol_price(primary_xpair, eff_pt)
            or resolve_symbol_price(default_xpair, eff_pt)
        )
        sib_px = resolve_symbol_price(t212_sib, eff_pt)
        live_px = xpair_px or sib_px
        tp1_a = safe_pos_float(xstock_cfg.get("xstock_tp_alert_pct"), default=None)
        tp2_a = safe_pos_float(xstock_cfg.get("xstock_tp2_alert_pct"), default=None)
        sl_a = safe_pos_float(xstock_cfg.get("xstock_sl_alert_pct"), default=None)
        ipo_watch_active = bool(x_regex or cfg_underlying)
        if not ipo_watch_active and (x_holdings or active_pair_by_underlying):
            alloc_info = None
        has_distinct_watched_ipo = bool(cfg_underlying and cfg_underlying != underlying)
        ipo_alerts = {
            k: v
            for k, v in (
                ("tp1_alert_pct", tp1_a),
                ("tp2_alert_pct", tp2_a),
                ("sl_alert_pct", sl_a),
                ("listed_pair", state_listed_pair if (state_listed_pair and pair_underlying == cfg_underlying) else None),
            )
            if v is not None
        }
        bot_observe_only = _parse_bool_flag(xstock_cfg.get("xstock_bot_observe_only"), False)
        bot_tp = safe_pos_float(xstock_cfg.get("xstock_bot_tp_pct"), default=None)
        bot_trail = safe_pos_float(xstock_cfg.get("xstock_bot_tp_trail_pct"), default=None)
        max_dca_cfg = safe_nonneg_float(xstock_cfg.get("xstock_bot_max_dca_buys"), default=1.0) or 0.0
        bot_dca = safe_pos_float(xstock_cfg.get("xstock_bot_dca_drop_pct"), default=None) if max_dca_cfg > 0 else None
        ws_root_dir = os.path.dirname(eff_k_dir) if eff_k_dir else ""
        guard_conf = (
            load_key_value_conf(ws_root_dir, ws_root_dir)
            if ws_root_dir and os.path.isfile(os.path.join(ws_root_dir, "order_guard.conf"))
            else {}
        )
        xs_cb = parse_intent_circuit_breaker_config(guard_conf)
        xs_min_profit_pct = safe_nonneg_float(
            xs_cb.get("xstock_sell_min_profit_pct"),
            default=DEFAULT_XSTOCK_SELL_MIN_PROFIT_PCT,
        )
        xs_min_drop_pct = safe_nonneg_float(
            xs_cb.get("xstock_sell_min_target_drop_pct"),
            default=DEFAULT_XSTOCK_SELL_MIN_TARGET_DROP_PCT,
        )
        xs_min_conf = safe_nonneg_float(
            xs_cb.get("xstock_sell_min_confidence"),
            default=DEFAULT_XSTOCK_SELL_MIN_CONFIDENCE,
        )
        xs_cooldown_h = safe_nonneg_float(
            xs_cb.get("xstock_sell_cooldown_hours"),
            default=DEFAULT_XSTOCK_SELL_COOLDOWN_HOURS,
        )
        xs_crash_drop_pct = safe_nonneg_float(
            xs_cb.get("xstock_sell_crash_min_target_drop_pct"),
            default=DEFAULT_XSTOCK_SELL_CRASH_MIN_TARGET_DROP_PCT,
        )
        xs_crash_conf = safe_nonneg_float(
            xs_cb.get("xstock_sell_crash_min_confidence"),
            default=DEFAULT_XSTOCK_SELL_CRASH_MIN_CONFIDENCE,
        )
        hold_meta_parts = [p for p in (f"{primary_hold_qty:g} {primary_hold_tok}" if primary_hold_qty > 0 else "", f"cost ${alloc_px:g}" if alloc_px else "", "0bps maker/taker fee") if p]
        bot_rule_parts = [p for p in (f"TP +{bot_tp:g}%" if bot_tp else "", f"Trail {bot_trail:g}%" if bot_trail else "", f"DCA -{bot_dca:g}%" if bot_dca else "") if p]
        bot_rule_sfx = f" ({' / '.join(bot_rule_parts)})" if bot_rule_parts else ""
        bot_mode_desc = (
            "kraken_bot.py is OBSERVE-ONLY / retired (cost basis tracked from cacheManager fills; places NO automatic orders)"
            if bot_observe_only
            else f"kraken_bot.py (by=BOT{bot_rule_sfx})"
        )
        conf_clause = f", confidence >= {xs_min_conf:g}" if xs_min_conf > 0.0 else ""
        cd_clause = f"; {xs_cooldown_h:g}h sell cooldown" if xs_cooldown_h > 0.0 else ""
        crash_clause = (
            f"; OR (B) Imminent-crash capital preservation (allowed below +{xs_min_profit_pct:g}% / at a loss ONLY to stop severe crashes): "
            f"requires urgency=HIGH, confidence >= {xs_crash_conf:g}, and explicit target_px >= {xs_crash_drop_pct:g}% below sell price"
            if xs_crash_drop_pct > 0.0
            else ""
        )
        active_directive = (
            f"{primary_xpair} ({', '.join(hold_meta_parts)}): LONG-TERM INVESTMENT -- SELL-ONLY ON KRAKEN EU "
            f"(Spot API BUY on tokenized xStocks returns EGeneral:Permission denied; API SELL is 0bps, BUT manual rebuy on Kraken carries a HEAVY re-entry commission/spread and the owner cannot re-enter quickly; "
            f"{bot_mode_desc}). Default posture is HOLD. Sell RARELY via 0bps LIMIT (owner gets emailed automatically to rebuy manually) ONLY when: "
            f"(A) Profit-taking: unrealized profit > +{xs_min_profit_pct:g}% above active cost basis (gateway minimum >= +{xs_min_profit_pct:g}%) AND "
            f"HIGH-CONVICTION multi-day decline thesis (minimum 1-2+ days expected drop{conf_clause}, explicit target_px >= {xs_min_drop_pct:g}% below sell price) "
            f"large enough to clearly overcome the heavy manual rebuy fee/spread{crash_clause}{cd_clause}. NEVER scalp on 90s/5m kinematics or shallow dips, "
            f"and do NOT submit Kraken xStock BUY orders (use T212 {t212_sib} for US equity buys)."
        )
        ipo_sym_lbl = ensure_xstock_pair(cfg_underlying) if (ipo_watch_active and cfg_underlying) else ""
        xst_summary: Dict[str, Any] = {
            "underlying": underlying,
            "t212_sibling": t212_sib,
            "holdings": x_holdings,
            **({"holdings_underlying_map": holdings_by_underlying} if len(held_underlyings) > 1 else {}),
            **({"watched_ipo_underlying": cfg_underlying} if (ipo_watch_active and has_distinct_watched_ipo) else {}),
            **({"watched_ipo_alerts": ipo_alerts} if (ipo_watch_active and has_distinct_watched_ipo and ipo_alerts) else {}),
            "allocated": alloc_info,
            "listed_pair": listed_pair,
            "alloc_price": alloc_px,
            "fee_bps": 0,
            **(
                {"tp1_alert_pct": tp1_a, "tp2_alert_pct": tp2_a, "sl_alert_pct": sl_a}
                if (ipo_watch_active and not has_distinct_watched_ipo)
                else {}
            ),
            "active_swing_directive": active_directive,
            **(
                {
                    "watched_ipo_playbook": (
                        f"{ipo_sym_lbl} (0bps maker/taker fee, auto-adopted cost in .state_{ipo_sym_lbl}.json): "
                        "do NOT assume an explosive post-IPO moonshot; evaluate live L2 bid/ask walls, ob_imbalance, "
                        "and 90s/5m kinematics to harvest strength pragmatically in LIMIT tranches when upside stalls "
                        "into ask walls, avoid panic-selling opening wicks, and wait for post-IPO stabilization after full exit."
                    )
                }
                if ipo_sym_lbl
                else {}
            ),
        }
        if live_px > 0:
            xst_summary["live_price"] = round(live_px, 4)
            if alloc_px is not None:
                xst_summary["pnl_vs_alloc_pct"] = round(((live_px - alloc_px) / alloc_px) * 100.0, 2) + 0.0
        if xpair_px > 0 and sib_px > 0:
            sib_entry = casefold_get(eff_pt, t212_sib)
            sib_mstate = str(sib_entry.get("market_state") or "").strip().upper() if isinstance(sib_entry, dict) else ""
            sib_qsrc = str(sib_entry.get("quote_source") or "").strip().lower() if isinstance(sib_entry, dict) else ""
            if not sib_mstate:
                if sib_qsrc in ("yahoo_rth", "t212_live"):
                    sib_mstate = "REGULAR"
                elif sib_qsrc in ("t212_rth_close", "yahoo_rth_close"):
                    sib_mstate = "RTH_CLOSE"
                elif sib_qsrc == "yahoo_ext_cached":
                    sib_mstate = "EXT_CLOSE"
                else:
                    sib_mstate = resolve_symbol_market_session(t212_sib)
            xst_summary["underlying_live_price"] = round(sib_px, 4)
            if sib_mstate:
                xst_summary["underlying_session"] = sib_mstate
            xpair_q = TimedQuote.from_raw(xpair_px, source="kraken_xstock")
            sib_q = TimedQuote.from_raw(sib_px, source="underlying")
            if xpair_q is not None and sib_q is not None and not _prices_diverged(xpair_q.price, sib_q.price, threshold=0.35):
                xst_summary["xstock_vs_underlying_pct"] = round(quote_spread_bps(xpair_q, sib_q, decimals=4) / 100.0, 2) + 0.0
        if isinstance((sib_cyc := first_variant_dict(eff_t212, (t212_sib, underlying))), dict):
            xst_summary["t212_sibling_cycle"] = {k: v for k, v in sib_cyc.items() if not str(k).startswith("_")}
        return xst_summary


class TradablePositionsAndOrdersCollector(PortfolioCacheCollectorBase):
    """Concrete collector (Level 3) for spot wallet balances, trailing peaks, and resting open orders."""

    cls_name = "TradablePositionsAndOrdersCollector"
    COLLECTOR_ID = "portfolio_tradable_positions"

    @staticmethod
    def _dedup_open_orders(open_orders_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(open_orders_list) <= 1:
            return open_orders_list
        deduped_open: List[Dict[str, Any]] = []
        seen_open_keys: set[tuple[str, str, str]] = set()
        for oo_item in open_orders_list:
            if not isinstance(oo_item, dict):
                deduped_open.append(oo_item)
                continue
            oo_oid = extract_order_id(oo_item, default="") or ""
            if oo_oid:
                oo_k = (
                    normalize_venue_name(oo_item.get("venue")),
                    str(oo_item.get("symbol") or "").strip().upper(),
                    oo_oid,
                )
                if oo_k in seen_open_keys:
                    continue
                seen_open_keys.add(oo_k)
            deduped_open.append(oo_item)
        return deduped_open

    @staticmethod
    def _persist_venue_balances_locked(
        cache_dir: str,
        cached_bals: Dict[str, Any],
        updated_bals: Dict[str, Any],
        failed_open_venues: set[str],
        *,
        now_ts: Optional[float] = None,
        rj_bals: Any,
        skip_balance_cache_venues: Optional[set[str]] = None,
    ) -> None:
        try:
            eff_now = resolve_now_ts(now_ts)
            read_disk = rj_bals in (_default_read_json_cached, read_json_recovering, read_json_cached)

            def _bounded_ts(raw_ts: Any, fallback: float) -> float:
                ts_s = normalize_epoch_sec(raw_ts, default=None)
                return ts_s if (ts_s is not None and ts_s <= eff_now + 60.0) else fallback

            # 1. Layer 3 raw account balances -> cachedb/orders/cache_balance_<venue>.json (CacheBalanceManager)
            per_venue_assets: Dict[str, Dict[str, Dict[str, Any]]] = {}
            per_venue_bd: Dict[str, Dict[str, Dict[str, float]]] = {}
            for k, v in updated_bals.items():
                if not isinstance(k, str):
                    continue
                if k.startswith("cash:") and isinstance(v, dict):
                    parts = k.split(":", 2)
                    v_n = normalize_venue_name(v.get("venue") or (parts[1] if len(parts) > 1 else ""))
                    a_u = canonical_balance_asset_key(v.get("asset") or (parts[2] if len(parts) > 2 else ""), venue=v_n)
                    free_v = safe_nonneg_float(v.get("free"), default=None)
                    if v_n and a_u and free_v is not None:
                        locked_q = safe_nonneg_float(v.get("locked_quote_usd"), default=0.0) or 0.0
                        c_ts = _bounded_ts(v.get("ts"), eff_now)
                        per_venue_assets.setdefault(v_n, {})[a_u] = {
                            "free": free_v,
                            "locked": locked_q,
                            "ts": c_ts,
                        }
                elif k.startswith("breakdown:") and isinstance(v, dict):
                    parts = k.split(":", 2)
                    v_n = normalize_venue_name(parts[1] if len(parts) > 1 else "")
                    q_u = str(parts[2] if len(parts) > 2 else "USD").strip().upper() or "USD"
                    if v_n and v:
                        per_venue_bd.setdefault(v_n, {})[q_u] = dict(v)
                elif k.startswith("pos:") and isinstance(v, dict):
                    parts = k.split(":", 2)
                    v_n = normalize_venue_name(v.get("venue") or (parts[1] if len(parts) > 1 else ""))
                    if not v_n:
                        continue
                    p_sym = str(v.get("symbol") or (parts[2] if len(parts) > 2 else "")).strip()
                    res_b, res_q = resolve_assets(p_sym, v.get("base"), v.get("quote"))
                    b_k = canonical_balance_asset_key(res_b, venue=v_n)
                    q_u = str(res_q or "").strip().upper()
                    pos_ts = _bounded_ts(v.get("ts"), eff_now)
                    fb_v = safe_nonneg_float(v.get("free_base"), default=None)
                    if b_k and fb_v is not None:
                        lb_v = safe_nonneg_float(v.get("locked_base_qty"), default=0.0) or 0.0
                        b_ts = _bounded_ts(v.get("base_bal_ts"), pos_ts)
                        per_venue_assets.setdefault(v_n, {})[b_k] = {
                            "free": fb_v,
                            "locked": lb_v,
                            "ts": b_ts,
                        }
                    fq_v = safe_nonneg_float(v.get("free_quote"), default=None)
                    if q_u and fq_v is not None and q_u not in per_venue_assets.get(v_n, {}):
                        q_ts = _bounded_ts(v.get("quote_bal_ts"), pos_ts)
                        lq_v = safe_nonneg_float(v.get("locked_quote_usd"), default=None)
                        per_venue_assets.setdefault(v_n, {})[q_u] = {
                            "free": fq_v,
                            "locked": lq_v,
                            "ts": q_ts,
                        }
                elif ":" in k and not k.startswith(("open_orders:", "open_orders_verified_ts:")):
                    raw_v, raw_a = k.split(":", 1)
                    if ":" not in raw_a:
                        v_n = normalize_venue_name(raw_v)
                        a_k = canonical_balance_asset_key(raw_a, venue=v_n)
                        free_v = safe_nonneg_float(v, default=None)
                        if v_n and a_k and free_v is not None and a_k not in per_venue_assets.get(v_n, {}):
                            per_venue_assets.setdefault(v_n, {})[a_k] = {
                                "free": free_v,
                                "locked": None,
                                "ts": eff_now,
                            }

            skip_v_set = skip_balance_cache_venues or set()
            if per_venue_assets or per_venue_bd:
                ensure_cacheprice_classes_bound()
                bal_cls = _mc.CacheBalanceManager
                if bal_cls is not None:
                    for v_n in sorted(set(per_venue_assets) | set(per_venue_bd)):
                        if v_n in skip_v_set:
                            continue
                        if (bal_path := resolve_venue_balance_cache_path(cache_dir, v_n)) is None:
                            continue
                        try:
                            bal_cls.replace_venue_balances_snapshot(
                                bal_path,
                                per_venue_assets.get(v_n) or {},
                                venue=v_n,
                                now_s=eff_now,
                                breakdown=per_venue_bd.get(v_n),
                            )
                        except Exception:
                            pass

            # 2. Layer 5 derived position snapshots (pos:* ONLY) -> statedb/portfolio/portfolio_positions_state.json
            pos_path = resolve_portfolio_positions_state_path(cache_dir)
            upsert_portfolio_positions_snapshot(
                pos_path,
                updated_bals,
                fallback_state=cached_bals,
                drop_symbols=_resolve_non_tradable_registry_symbols(cache_dir),
                read_disk=read_disk,
                save_fn=resolve_save_json_state(local_fn=save_json_state),
            )
        except Exception:
            pass

    @staticmethod
    def _recover_from_cached_balances(
        cached_bals: Dict[str, Any],
        prices_and_trends: Dict[str, Any],
        trailing_map: Dict[str, Dict[str, Any]],
        trailing_by_venue: Dict[str, Dict[str, Dict[str, Any]]],
        *,
        known_ai_order_ids: set[str],
        audit_origin_map: Dict[str, Any],
        order_ts_map: Dict[str, Any],
        now_ts_oo: float,
        gateway_max_usd: float,
        tradable_positions: List[Dict[str, Any]],
        open_orders_list: List[Dict[str, Any]],
        venue_quote_cash: Dict[str, Dict[str, Any]],
        pending_peaks_by_vs: Dict[tuple[str, str], tuple[float, Optional[float]]],
        cache_dir: Optional[str] = None,
        gateway_min_usd: float = 0.0,
    ) -> None:
        gw_max_rec = safe_pos_float(gateway_max_usd, default=2000.0) or 2000.0
        gw_min_rec = safe_nonneg_float(gateway_min_usd, default=0.0) or 0.0
        seen_rec_vs: set[tuple[str, str]] = set()
        cached_obs_by_va = _build_cached_balance_observations(cached_bals)
        non_tradable_rec_syms = _resolve_non_tradable_registry_symbols(cache_dir)
        def _valid_rec_ts(raw_ts: Any) -> Optional[float]:
            ts_s = normalize_epoch_sec(raw_ts, default=None)
            return ts_s if (ts_s is not None and ts_s <= now_ts_oo + 60.0) else None

        for k, v in cached_bals.items():
            if k.startswith("pos:") and isinstance(v, dict):
                rec_pos = dict(v)
                pos_bal_ts = _valid_rec_ts(v.get("ts"))
                pos_base_ts = _valid_rec_ts(v.get("base_bal_ts")) or pos_bal_ts
                pos_quote_ts = _valid_rec_ts(v.get("quote_bal_ts")) or pos_bal_ts
                prev_pos_lb = safe_nonneg_float(v.get("locked_base_qty"), default=None)
                prev_pos_lq = safe_nonneg_float(v.get("locked_quote_usd"), default=None)
                for stale_k in ("locked_quote_usd", "replace_max_buy_usd", "locked_base_qty", "locked_base_usd", "replace_max_sell_usd", "ts", "base_bal_ts", "quote_bal_ts"):
                    rec_pos.pop(stale_k, None)
                v_rec_norm = normalize_venue_name(rec_pos.get("venue"))
                p_sym_raw = str(rec_pos.get("symbol") or "").strip()
                p_sym = canonical_display_symbol(p_sym_raw, venue=v_rec_norm) or p_sym_raw.upper()
                p_sym_u = p_sym.upper()
                if p_sym_u and p_sym_u in non_tradable_rec_syms:
                    continue
                if p_sym_u and (v_rec_norm, p_sym_u) in seen_rec_vs:
                    continue
                if p_sym_u:
                    seen_rec_vs.add((v_rec_norm, p_sym_u))
                if p_sym:
                    rec_pos["symbol"] = p_sym
                res_b_rec, res_q_rec = resolve_assets(p_sym, rec_pos.get("base"), rec_pos.get("quote"))
                p_base_u = canonical_balance_asset_key(res_b_rec, venue=v_rec_norm)
                p_quote = str(res_q_rec or rec_pos.get("quote") or "USD").strip().upper()
                is_eq_rec = is_equity_venue(v_rec_norm)
                base_ref_locked: Optional[float] = None
                if v_rec_norm and p_base_u and (base_obs := cached_obs_by_va.get((v_rec_norm, p_base_u))) is not None:
                    fb_val, base_bal_ts, base_ref_locked = base_obs.free, base_obs.ts, base_obs.reflected_locked
                else:
                    fb_val = safe_nonneg_float(rec_pos.get("free_base"), default=None)
                    base_bal_ts = pos_base_ts if fb_val is not None else None
                    base_ref_locked = prev_pos_lb
                if fb_val is not None:
                    rec_pos["free_base"] = round(fb_val, 6)
                quote_ref_locked: Optional[float] = None
                if v_rec_norm and p_quote and (quote_obs := cached_obs_by_va.get((v_rec_norm, p_quote))) is not None:
                    fq_val, quote_bal_ts, quote_ref_locked = quote_obs.free, quote_obs.ts, quote_obs.reflected_locked
                else:
                    fq_val = safe_nonneg_float(rec_pos.get("free_quote"), default=None)
                    quote_bal_ts = pos_quote_ts if fq_val is not None else None
                    quote_ref_locked = prev_pos_lq
                if fq_val is not None:
                    rec_pos["free_quote"] = round(fq_val, 2)
                if base_bal_ts is not None:
                    rec_pos["_base_balance_ts"] = base_bal_ts
                    if base_ref_locked is not None:
                        rec_pos["_prev_locked_base_qty"] = base_ref_locked
                if quote_bal_ts is not None:
                    rec_pos["_quote_balance_ts"] = quote_bal_ts
                    if quote_ref_locked is not None:
                        rec_pos["_prev_locked_quote_usd"] = quote_ref_locked
                px_rec = 0.0
                if p_sym:
                    exact_pt_px = safe_pos_float((casefold_get(prices_and_trends, p_sym) or {}).get("price"), default=0.0) or 0.0
                    cached_pos_px = safe_pos_float(rec_pos.get("price"), default=0.0) or 0.0
                    p_lookup_rec = ensure_equity_suffix(p_sym, is_equity=is_eq_rec)
                    sib_pt_px = (
                        resolve_symbol_price(p_lookup_rec, prices_and_trends, explicit_quote=None if is_eq_rec else p_quote)
                        if exact_pt_px <= 0.0
                        else exact_pt_px
                    )
                    if exact_pt_px > 0.0:
                        px_rec = exact_pt_px
                    elif cached_pos_px > 0.0 and (sib_pt_px <= 0.0 or not _prices_diverged(cached_pos_px, sib_pt_px, threshold=0.35 if is_eq_rec else 0.10)):
                        px_rec = cached_pos_px
                    else:
                        px_rec = sib_pt_px
                    if px_rec > 0:
                        rec_pos["price"] = round_opt_px(px_rec, 6)
                        if ProviderVenueMetadataMixin.prefers_static_precision(v_rec_norm):
                            _, _, _, _, def_step, def_tick, _ = (
                                ProviderVenueMetadataMixin.resolve_venue_default_fee_and_precision(v_rec_norm, px_rec)
                            )
                            rec_pos["min_qty"] = def_step
                            rec_pos["step_size"] = def_step
                            rec_pos["tick_size"] = def_tick
                        if fb_val is not None:
                            rec_pos["value_usd"] = round(fb_val * px_rec, 2)
                            sell_cap_rec = venue_fee_cap_quantity(v_rec_norm, "SELL", px_rec, fb_val)
                            raw_sell_rec = min(gw_max_rec, sell_cap_rec * px_rec)
                            rec_min_not = max(
                                safe_pos_float(rec_pos.get("exchange_min_notional_usd"), default=1.0) or 1.0,
                                gw_min_rec,
                            )
                            rec_min_qty = safe_pos_float(rec_pos.get("min_qty"), default=0.0) or 0.0
                            rec_pos["effective_max_sell_usd"] = (
                                0.0
                                if (raw_sell_rec < rec_min_not or (rec_min_qty > 0.0 and sell_cap_rec < rec_min_qty))
                                else round(raw_sell_rec, 2)
                            )
                        if (
                            gw_min_rec > 0.0
                            and rec_pos.get("effective_max_buy_usd") is not None
                            and (safe_nonneg_float(rec_pos.get("effective_max_buy_usd"), default=0.0) or 0.0) < gw_min_rec
                        ):
                            rec_pos["effective_max_buy_usd"] = 0.0
                        pt_key_rec = next((pk for pk in prices_and_trends if str(pk).upper() == p_sym_u), p_sym)
                        prices_and_trends.setdefault(pt_key_rec, {"price": px_rec, "trend": 0, "growth_coefficient": 0.0})
                    else:
                        px_rec = cached_pos_px
                rec_peak = safe_pos_float(rec_pos.get("trailing_peak"), default=None)
                if p_sym:
                    v_trail_map = trailing_by_venue.get(v_rec_norm, {}) if isinstance(trailing_by_venue, dict) else {}
                    t_cands = cache_symbol_variants(p_sym, include_original=True, include_base=True, quotes=compatible_quote_suffixes(p_sym))
                    t_dict = first_variant_dict(v_trail_map, t_cands) or (
                        first_variant_dict(trailing_map, t_cands) if MarketDataProvider.uses_shared_trailing_map(v_rec_norm) else None
                    )
                    if isinstance(t_dict, dict) and ("peak" in t_dict or rec_peak is None):
                        rec_min_not_pk = max(
                            1.0,
                            first_pos_float(rec_pos.get("exchange_min_notional_usd"), rec_pos.get("min_notional_usd"), default=1.0) or 1.0,
                        )
                        rec_peak = _resolve_effective_trailing_raw_peak(
                            t_dict, rec_pos, px_f=px_rec, min_notional=rec_min_not_pk
                        )
                _attach_trailing_peak_to_position(
                    rec_pos, v_rec_norm, p_sym, px_rec, fb_val, rec_peak, pending_peaks_by_vs,
                    fallback_dd=rec_pos.get("dd_from_peak_pct"),
                )
                tradable_positions.append(rec_pos)
            elif k.startswith("cash:") and isinstance(v, dict):
                c_venue = str(v.get("venue") or "").strip().lower()
                c_asset = canonical_balance_asset_key(v.get("asset"), venue=c_venue)
                if c_venue and c_asset:
                    c_v_norm = normalize_venue_name(c_venue)
                    rec_cash = dict(v)
                    prev_cash_lq = safe_nonneg_float(v.get("locked_quote_usd"), default=None)
                    raw_c_ts = _valid_rec_ts(v.get("ts"))
                    rec_cash.pop("locked_quote_usd", None)
                    rec_cash.pop("ts", None)
                    cash_ref_locked: Optional[float] = None
                    if (c_obs := cached_obs_by_va.get((c_v_norm, c_asset))) is not None:
                        rec_cash["free"] = round(c_obs.free, 2)
                        c_bal_ts = c_obs.ts
                        cash_ref_locked = c_obs.reflected_locked
                    else:
                        c_bal_ts = raw_c_ts
                        cash_ref_locked = prev_cash_lq
                    if c_bal_ts is not None:
                        rec_cash["_balance_ts"] = c_bal_ts
                        if cash_ref_locked is not None:
                            rec_cash["_prev_locked_quote_usd"] = cash_ref_locked
                    venue_quote_cash[f"{c_venue}:{c_asset}"] = rec_cash

    @staticmethod
    def _collect_from_instruments(
        inst_map: Dict[str, Any],
        prices_and_trends: Dict[str, Any],
        trailing_map: Dict[str, Dict[str, Any]],
        trailing_by_venue: Dict[str, Dict[str, Dict[str, Any]]],
        *,
        cached_bals: Dict[str, Any],
        updated_bals: Dict[str, Any],
        cached_open_by_vs: Dict[Tuple[str, str], List[Dict[str, Any]]],
        cached_open_verified_ts: Dict[str, float],
        cached_open_sym_verified_ts: Dict[Tuple[str, str], float],
        updated_open_by_venue: Dict[str, Dict[str, List[Dict[str, Any]]]],
        balance_memo: Dict[tuple[str, str], Optional[float]],
        known_ai_order_ids: set[str],
        audit_origin_map: Dict[str, Any],
        order_ts_map: Dict[str, Any],
        now_ts_oo: float,
        gateway_max_usd: float,
        gateway_min_usd: float = 0.0,
        cache_dir: Optional[str] = None,
        allow_network: bool,
        force_persist: bool,
        tradable_positions: List[Dict[str, Any]],
        open_orders_list: List[Dict[str, Any]],
        venue_quote_cash: Dict[str, Dict[str, Any]],
        inst_by_sym: Dict[str, Any],
        pending_peaks_by_vs: Dict[tuple[str, str], tuple[float, Optional[float]]],
        live_open_ok_venues: set[str],
        failed_open_venues: set[str],
        live_open_ok_symbols: set[Tuple[str, str]],
        failed_open_symbols: set[Tuple[str, str]],
        explicit_now_ts: bool = False,
        fresh_bal_cache_venues: Optional[set[str]] = None,
        fresh_bal_cache_obs: Optional[Dict[tuple[str, str], BalanceObservation]] = None,
        fresh_bal_cache_bd: Optional[Dict[tuple[str, str], Dict[str, float]]] = None,
        live_balance_venues: Optional[set[str]] = None,
    ) -> None:
        cached_obs_by_va = _build_cached_balance_observations(cached_bals)
        balance_obs_memo: Dict[tuple[str, str], Optional[BalanceObservation]] = {}
        l3_fresh_venues = fresh_bal_cache_venues or set()
        l3_fresh_obs = fresh_bal_cache_obs or {}
        l3_fresh_bd = fresh_bal_cache_bd or {}
        live_bal_v = live_balance_venues if isinstance(live_balance_venues, set) else set()

        def _query_free_bal(inst_obj: Any, asset: str) -> Optional[BalanceObservation]:
            if not asset:
                return None
            raw_v = str(getattr(inst_obj, "provider_name", "") or "").strip().lower()
            v_name = normalize_venue_name(raw_v) or raw_v
            eff_asset = canonical_balance_asset_key(asset, venue=v_name) or asset.upper()
            key = (v_name, eff_asset)
            if key in balance_obs_memo:
                return balance_obs_memo[key]
            if key in balance_memo and balance_memo[key] is not None:
                c_obs_pre = cached_obs_by_va.get(key)
                obs_mem = BalanceObservation(
                    free=balance_memo[key],
                    ts=c_obs_pre.ts if c_obs_pre is not None else None,
                    priority=1,
                    reflected_locked=c_obs_pre.reflected_locked if c_obs_pre is not None else None,
                )
                balance_obs_memo[key] = obs_mem
                return obs_mem
            prov = getattr(inst_obj, "_provider", None) if allow_network else None
            # Cache-first on Layer 3 cache_balance_<venue>.json for real providers when fresh (0 ms, zero REST).
            if (
                allow_network
                and isinstance(prov, MarketDataProvider)
                and v_name in l3_fresh_venues
                and (c_l3_obs := l3_fresh_obs.get(key)) is not None
            ):
                balance_memo[key] = c_l3_obs.free
                balance_obs_memo[key] = c_l3_obs
                return c_l3_obs
            prov_dict = getattr(prov, "__dict__", {}) if prov is not None else {}
            fl_fn = getattr(prov, "free_and_locked_balance", None) if prov is not None else None
            free_fn = getattr(prov, "free_balance", None) if prov is not None else None
            use_fl = (
                callable(fl_fn)
                and not hasattr(fl_fn, "_mock_name")
                and not ("free_balance" in prov_dict and "free_and_locked_balance" not in prov_dict)
            )
            val_f: Optional[float] = None
            live_locked: Optional[float] = None
            if use_fl:
                try:
                    raw_pair = fl_fn(eff_asset)
                    if isinstance(raw_pair, tuple) and len(raw_pair) == 2:
                        val_f = safe_nonneg_float(raw_pair[0], default=None)
                        live_locked = safe_nonneg_float(raw_pair[1], default=None)
                    else:
                        val_f = safe_nonneg_float(raw_pair, default=None)
                except Exception:
                    val_f, live_locked = None, None
            elif callable(free_fn):
                try:
                    val_f = safe_nonneg_float(free_fn(eff_asset), default=None)
                except Exception:
                    val_f = None
            bkey = f"{v_name}:{eff_asset}"
            raw_bkey = f"{raw_v}:{eff_asset}"
            obs_out: Optional[BalanceObservation] = None
            if val_f is not None:
                updated_bals[bkey] = round(val_f, 8)
                live_bal_v.add(v_name)
                bal_ts = round(now_ts_oo if explicit_now_ts else resolve_now_ts(None), 3)
                obs_out = BalanceObservation(free=val_f, ts=bal_ts, priority=2, reflected_locked=live_locked)
            elif (cached_obs := cached_obs_by_va.get(key)) is not None:
                obs_out = cached_obs
            elif raw_bkey in cached_bals and (raw_v_f := safe_nonneg_float(cached_bals.get(raw_bkey), default=None)) is not None:
                obs_out = BalanceObservation(free=raw_v_f, ts=None, priority=0, reflected_locked=None)
            balance_memo[key] = obs_out.free if obs_out is not None else None
            balance_obs_memo[key] = obs_out
            return obs_out

        merged_params_by_vs: Dict[tuple[str, str], Dict[str, Any]] = {}
        for _inst_cand in inst_map.values():
            _c_sym = str(getattr(_inst_cand, "symbol", "") or "").strip().upper()
            _c_ven = normalize_venue_name(str(getattr(_inst_cand, "provider_name", "") or "").strip().lower())
            if _c_sym and _c_ven and isinstance(_c_p := getattr(_inst_cand, "params", None), dict) and _c_p:
                merged_params_by_vs.setdefault((_c_ven, _c_sym), {}).update(_c_p)
        fee_spec_resolver = build_persisted_venue_fee_resolver(cache_dir, now_ts=now_ts_oo)
        seen_vs: set[tuple[str, str]] = set()
        for inst in inst_map.values():
            raw_sym = str(getattr(inst, "symbol", "") or "").strip()
            venue = str(getattr(inst, "provider_name", "") or "").strip().lower()
            if not raw_sym or not venue:
                continue
            v_norm = normalize_venue_name(venue)
            sym = canonical_display_symbol(raw_sym, venue=v_norm) or raw_sym.upper()
            sym_u = sym.upper()
            if (v_norm, sym_u) in seen_vs:
                continue
            seen_vs.add((v_norm, sym_u))
            is_eq_v = is_equity_venue(v_norm)
            pos_px_dec = MarketDataProvider.resolve_venue_position_price_decimals(v_norm)
            res_b, res_q = resolve_assets(sym, getattr(inst, "base", None), getattr(inst, "quote", None))
            base, quote = res_b or sym, res_q or "USD"
            if sym not in inst_by_sym or not MarketDataProvider.uses_bare_base_order_symbols(v_norm):
                inst_by_sym[sym] = inst
            inst_by_sym[f"{v_norm}:{sym}"] = inst
            if sym_u != sym:
                inst_by_sym.setdefault(sym_u, inst)
                inst_by_sym[f"{v_norm}:{sym_u}"] = inst

            base_obs, quote_obs = _query_free_bal(inst, base), _query_free_bal(inst, quote)
            base_free = base_obs.free if base_obs is not None else None
            base_bal_ts = base_obs.ts if base_obs is not None else None
            base_ref_locked = base_obs.reflected_locked if base_obs is not None else None
            quote_free = quote_obs.free if quote_obs is not None else None
            quote_bal_ts = quote_obs.ts if quote_obs is not None else None
            quote_ref_locked = quote_obs.reflected_locked if quote_obs is not None else None
            pt_entry_existing = casefold_get(prices_and_trends, sym)
            px = pt_entry_existing.get("price") if isinstance(pt_entry_existing, dict) else None
            px_f = safe_pos_float(px, default=0.0) or 0.0
            pos_bkey_cur = f"pos:{v_norm}:{sym}"
            raw_prev_pos = cached_bals.get(pos_bkey_cur)
            if not isinstance(raw_prev_pos, dict) and sym_u != sym:
                raw_prev_pos = cached_bals.get(f"pos:{v_norm}:{sym_u}")
            prev_cached_pos = raw_prev_pos if isinstance(raw_prev_pos, dict) else {}
            prev_pos_px = safe_pos_float(prev_cached_pos.get("price"), default=0.0) or 0.0
            prev_pos_ts = safe_pos_float(prev_cached_pos.get("ts"), default=None)
            if base_free is None and prev_cached_pos:
                base_free = safe_nonneg_float(
                    prev_cached_pos.get("free_base")
                    if prev_cached_pos.get("free_base") is not None
                    else prev_cached_pos.get("free"),
                    default=None,
                )
                if base_free is not None:
                    base_bal_ts = safe_pos_float(prev_cached_pos.get("base_bal_ts"), default=None) or prev_pos_ts
                    base_ref_locked = safe_nonneg_float(prev_cached_pos.get("locked_base_qty"), default=None)
            if quote_free is None:
                if prev_cached_pos and prev_cached_pos.get("free_quote") is not None:
                    quote_free = safe_nonneg_float(prev_cached_pos.get("free_quote"), default=None)
                    if quote_free is not None:
                        quote_bal_ts = safe_pos_float(prev_cached_pos.get("quote_bal_ts"), default=None) or prev_pos_ts
                        quote_ref_locked = safe_nonneg_float(prev_cached_pos.get("locked_quote_usd"), default=None)
                elif isinstance(c_cash := cached_bals.get(f"cash:{v_norm}:{quote.upper()}"), dict):
                    quote_free = safe_nonneg_float(c_cash.get("free"), default=None)
                    if quote_free is not None:
                        quote_bal_ts = safe_pos_float(c_cash.get("ts"), default=None)
                        quote_ref_locked = safe_nonneg_float(c_cash.get("locked_quote_usd"), default=None)
            px_from_cached_pos = False
            if allow_network and (
                px_f <= 0
                or (
                    is_eq_v
                    and (
                        _parse_bool_flag(force_persist, False)
                        or ((base_free or 0.0) > 0 and is_live_network_backfill_enabled())
                    )
                )
            ):
                try:
                    get_px_fn = getattr(inst._provider, "get_current_price", None)
                    if callable(get_px_fn) and (live_px := safe_pos_float(get_px_fn(sym), default=0.0) or 0.0) > 0:
                        px = px_f = round(live_px, pos_px_dec) if is_eq_v else live_px
                except Exception:
                    if px_f <= 0:
                        px, px_f = None, 0.0
            elif (
                not allow_network
                and is_eq_v
                and prev_pos_px > 0
                and prev_pos_ts is not None
                and 0.0 <= (now_ts_oo - prev_pos_ts) <= 300.0
            ):
                px = px_f = round(prev_pos_px, pos_px_dec)
                px_from_cached_pos = True
            lookup_sym_p = ensure_equity_suffix(sym, is_equity=is_eq_v)
            if px_f <= 0 and (px_f := resolve_symbol_price(lookup_sym_p, prices_and_trends, explicit_quote=None if is_eq_v else quote)) > 0:
                px = px_f
            if px_f <= 0 and prev_pos_px > 0:
                px = px_f = round(prev_pos_px, pos_px_dec) if is_eq_v else prev_pos_px
                px_from_cached_pos = True
            pt_key = next((k for k in prices_and_trends if str(k).upper() == sym_u), sym)
            sym_entry = prices_and_trends.setdefault(pt_key, {"price": px if px_f > 0 else None, "trend": 0, "growth_coefficient": 0.0})
            if px_f > 0 and ((safe_pos_float(sym_entry.get("price"), default=0.0) or 0.0) <= 0 or is_eq_v):
                sym_entry["price"] = px
            quote_breakdown: Optional[Dict[str, float]] = None
            bd_key = f"breakdown:{v_norm}:{quote.upper()}"
            prov_inst = getattr(inst, "_provider", None)
            if (
                allow_network
                and quote.upper() == "USD"
                and isinstance(prov_inst, MarketDataProvider)
                and v_norm in l3_fresh_venues
                and (v_norm, "USD") in l3_fresh_bd
            ):
                quote_breakdown = {
                    str(k): round(vf, 2)
                    for k, v in l3_fresh_bd[(v_norm, "USD")].items()
                    if (vf := safe_pos_float(v, default=None)) is not None
                } or None
            elif allow_network and quote.upper() == "USD" and callable(bd_fn := getattr(prov_inst, "usd_cash_breakdown", None)):
                try:
                    if isinstance(raw_bd := bd_fn(), dict) and raw_bd:
                        quote_breakdown = {str(k): round(vf, 2) for k, v in raw_bd.items() if (vf := safe_pos_float(v, default=None)) is not None} or None
                        if quote_breakdown:
                            updated_bals[bd_key] = quote_breakdown
                            live_bal_v.add(v_norm)
                except Exception:
                    quote_breakdown = None
            if quote_breakdown is None and isinstance(cached_bd := cached_bals.get(bd_key), dict) and cached_bd:
                quote_breakdown = {str(k): round(vf, 2) for k, v in cached_bd.items() if (vf := safe_pos_float(v, default=None)) is not None} or None
            if quote_free is not None:
                cash_entry: Dict[str, Any] = {"venue": venue, "asset": quote, "free": round(quote_free, 2)}
                if quote_breakdown:
                    cash_entry["breakdown"] = quote_breakdown
                persisted_cash = dict(cash_entry)
                if quote_bal_ts is not None:
                    persisted_cash["ts"] = round(quote_bal_ts, 3)
                    cash_entry["_balance_ts"] = quote_bal_ts
                    if quote_ref_locked is not None:
                        cash_entry["_prev_locked_quote_usd"] = quote_ref_locked
                updated_bals[f"cash:{v_norm}:{quote.upper()}"] = persisted_cash
                venue_quote_cash[f"{venue}:{quote}"] = cash_entry

            pos_val_usd = round(base_free * px_f, 2) if base_free is not None and px_f > 0 else None
            params = merged_params_by_vs.get((v_norm, sym_u)) or (
                getattr(inst, "params", {}) if isinstance(getattr(inst, "params", None), dict) else {}
            )
            sym_quotes = compatible_quote_suffixes(sym, quote)
            t_state = first_variant_dict(
                trailing_by_venue.get(v_norm, {}),
                cache_symbol_variants(sym, include_original=True, include_base="USD" in sym_quotes, quotes=sym_quotes),
                default={},
            )
            if not t_state and is_primary_stream_venue(v_norm) and isinstance((val_b := casefold_get(trailing_map, sym)), dict):
                t_state = val_b
            pos_entry: Dict[str, Any] = {
                "venue": venue, "symbol": sym, "base": base, "quote": quote,
                "price": round_opt_px(px, 6) if px is not None else None,
                "free_base": round(base_free, 6) if base_free is not None else None,
                "value_usd": pos_val_usd,
                "free_quote": round(quote_free, 2) if quote_free is not None else None,
            }
            if quote_breakdown:
                pos_entry["free_quote_breakdown"] = quote_breakdown
            pos_entry.update(
                resolve_venue_fee_and_limits(
                    venue,
                    sym,
                    inst,
                    price=px_f,
                    free_quote=quote_free,
                    free_base=base_free,
                    gateway_max_usd=gateway_max_usd,
                    gateway_min_usd=gateway_min_usd,
                    fee_spec_resolver=fee_spec_resolver,
                )
            )
            pos_min_not_pk = max(
                1.0,
                first_pos_float(pos_entry.get("exchange_min_notional_usd"), pos_entry.get("min_notional_usd"), default=1.0) or 1.0,
            )
            eff_raw_peak = _resolve_effective_trailing_raw_peak(
                t_state, prev_cached_pos, px_f=px_f, min_notional=pos_min_not_pk
            )
            _attach_trailing_peak_to_position(
                pos_entry, v_norm, sym, px_f, base_free, eff_raw_peak, pending_peaks_by_vs
            )
            if params.get("trailing.pct"):
                pos_entry["trailing_pct_cfg"] = params.get("trailing.pct")
            if params.get("mt.gain") or params.get("mt.lost"):
                pos_entry["mt_gain_pct"], pos_entry["mt_lost_pct"] = params.get("mt.gain"), params.get("mt.lost")
            if base_bal_ts is not None:
                pos_entry["_base_balance_ts"] = base_bal_ts
                if base_ref_locked is not None:
                    pos_entry["_prev_locked_base_qty"] = base_ref_locked
            if quote_bal_ts is not None:
                pos_entry["_quote_balance_ts"] = quote_bal_ts
                if quote_ref_locked is not None:
                    pos_entry["_prev_locked_quote_usd"] = quote_ref_locked
            tradable_positions.append(pos_entry)
            eff_pos_ts = (
                prev_pos_ts or round(now_ts_oo, 3)
                if (
                    px_from_cached_pos
                    or (
                        is_eq_v
                        and prev_pos_ts is not None
                        and prev_pos_px > 0.0
                        and px_f > 0.0
                        and abs(round(px_f, pos_px_dec) - round(prev_pos_px, pos_px_dec)) < 1e-4
                    )
                )
                else round(now_ts_oo, 3)
            )
            persisted_pos = {**_strip_internal_balance_ts(pos_entry), "ts": eff_pos_ts}
            if base_bal_ts is not None:
                persisted_pos["base_bal_ts"] = round(base_bal_ts, 3)
            if quote_bal_ts is not None:
                persisted_pos["quote_bal_ts"] = round(quote_bal_ts, 3)
            updated_bals[pos_bkey_cur] = persisted_pos

            vs_open_key = (v_norm, sym_u)
            cached_oo_list = cached_open_by_vs.get(vs_open_key)
            sym_oo_ver_ts = safe_pos_float(cached_open_sym_verified_ts.get(vs_open_key), default=0.0) or 0.0
            ven_oo_ver_ts = safe_pos_float(cached_open_verified_ts.get(v_norm), default=0.0) or 0.0
            eff_oo_ver_ts = max(sym_oo_ver_ts, ven_oo_ver_ts)
            # Cache-first on Layer 3 cache_order_open_<venue>.json for real providers when fresh (0 ms, zero REST).
            eff_oo_max_age = resolve_fresh_open_orders_cache_max_age_sec()
            fresh_l3_open = (
                allow_network
                and isinstance(prov_inst, MarketDataProvider)
                and eff_oo_ver_ts > 0.0
                and 0.0 <= (now_ts_oo - eff_oo_ver_ts) <= eff_oo_max_age
                and (vs_open_key in cached_open_by_vs or ven_oo_ver_ts > 0.0 or vs_open_key in cached_open_sym_verified_ts)
            )
            if fresh_l3_open:
                if cached_oo_list:
                    open_orders_list.extend(
                        _normalize_open_orders_list(
                            cached_oo_list,
                            venue,
                            sym,
                            known_ai_order_ids,
                            audit_origin_map=audit_origin_map,
                            now_ts=now_ts_oo,
                            order_ts_map=order_ts_map,
                        )
                    )
                live_open_ok_symbols.add(vs_open_key)
                if ven_oo_ver_ts > 0.0 and 0.0 <= (now_ts_oo - ven_oo_ver_ts) <= eff_oo_max_age:
                    live_open_ok_venues.add(v_norm)
            elif allow_network and callable(open_fn := getattr(inst, "open_orders", None)):
                sym_open_ok = False
                try:
                    if isinstance(raw_open := open_fn(), list):
                        sym_open_ok = True
                        sym_norm_orders = _normalize_open_orders_list(
                            raw_open,
                            venue,
                            sym,
                            known_ai_order_ids,
                            audit_origin_map=audit_origin_map,
                            now_ts=now_ts_oo,
                            order_ts_map=order_ts_map,
                        )
                        open_orders_list.extend(sym_norm_orders)
                        updated_open_by_venue.setdefault(v_norm, {})[sym] = list(raw_open)
                except Exception:
                    sym_open_ok = False
                if sym_open_ok:
                    live_open_ok_venues.add(v_norm)
                    live_open_ok_symbols.add(vs_open_key)
                else:
                    failed_open_venues.add(v_norm)
                    failed_open_symbols.add(vs_open_key)
                    open_orders_list.extend(
                        _normalize_open_orders_list(
                            cached_oo_list,
                            venue,
                            sym,
                            known_ai_order_ids,
                            audit_origin_map=audit_origin_map,
                            now_ts=now_ts_oo,
                            order_ts_map=order_ts_map,
                        )
                    )
            elif vs_open_key in cached_open_by_vs or vs_open_key in cached_open_sym_verified_ts:
                if vs_open_key in cached_open_by_vs:
                    open_orders_list.extend(
                        _normalize_open_orders_list(
                            cached_oo_list,
                            venue,
                            sym,
                            known_ai_order_ids,
                            audit_origin_map=audit_origin_map,
                            now_ts=now_ts_oo,
                            order_ts_map=order_ts_map,
                        )
                    )
                sym_ver_ts = safe_pos_float(cached_open_sym_verified_ts.get(vs_open_key), default=0.0) or 0.0
                if sym_ver_ts > 0.0 and 0.0 <= (now_ts_oo - sym_ver_ts) <= 300.0:
                    live_open_ok_symbols.add(vs_open_key)
                ver_ts = safe_pos_float(cached_open_verified_ts.get(v_norm), default=0.0) or 0.0
                if ver_ts > 0.0 and 0.0 <= (now_ts_oo - ver_ts) <= 300.0:
                    live_open_ok_venues.add(v_norm)

    def collect(
        self,
        cachedb_dir: str,
        symbol: str = "*",
        *,
        inst_map: Optional[Dict[str, Any]] = None,
        prices_and_trends: Optional[Dict[str, Any]] = None,
        trailing_map: Optional[Dict[str, Dict[str, Any]]] = None,
        trailing_by_venue: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
        known_ai_order_ids: Optional[set[str]] = None,
        gateway_max_usd: float = 2000.0,
        gateway_min_usd: float = 0.0,
        verified_open_venues: Optional[set[str]] = None,
        verified_open_symbols: Optional[set[Tuple[str, str]]] = None,
        allow_empty_inst_recovery: bool = False,
        allow_network: bool = True,
        persist: bool = True,
        force_persist: bool = False,
        now_ts: Optional[float] = None,
        **kwargs: Any,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
        inst_map = inst_map or {}
        prices_and_trends = prices_and_trends if isinstance(prices_and_trends, dict) else {}
        trailing_map = trailing_map or {}
        trailing_by_venue = trailing_by_venue or {}
        cache_dir = cachedb_dir or None

        tradable_positions, open_orders_list, venue_quote_cash, inst_by_sym = [], [], {}, {}
        balance_memo: Dict[tuple[str, str], Optional[float]] = {}
        _rj_bals = _resolve_read_json_cached()
        bcache_path, cached_bals, _ = load_venue_balances_state(cache_dir, _rj_bals)
        cached_open_latest_ts: Dict[str, float] = {}
        cached_open_by_vs, cached_open_verified_ts, cached_open_sym_verified_ts = _read_cached_open_orders_by_venue_sym(
            cache_dir, read_json_fn=_rj_bals, out_latest_ts=cached_open_latest_ts
        )
        latest_fill_ts_by_venue = _read_latest_filled_ts_by_venue(cache_dir, read_json_fn=_rj_bals) if allow_network else {}
        updated_bals: Dict[str, Any] = {}
        updated_open_by_venue: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        live_open_ok_venues: set[str] = set()
        failed_open_venues: set[str] = set()
        live_open_ok_symbols: set[Tuple[str, str]] = set()
        failed_open_symbols: set[Tuple[str, str]] = set()
        now_ts_oo = resolve_now_ts(now_ts)
        raw_bal_caches = load_venue_balance_caches(cache_dir, read_json_fn=_rj_bals) if cache_dir else {}
        fresh_bal_cache_venues: set[str] = set()
        fresh_bal_cache_obs: Dict[tuple[str, str], BalanceObservation] = {}
        fresh_bal_cache_bd: Dict[tuple[str, str], Dict[str, float]] = {}
        if raw_bal_caches:
            eff_bal_max_age = resolve_fresh_balance_cache_max_age_sec()
            _all_c_obs, _all_c_bd = extract_balance_observations_from_caches(raw_bal_caches)
            for _raw_v, _doc in raw_bal_caches.items():
                if not isinstance(_doc, dict):
                    continue
                _v_n = normalize_venue_name(_doc.get("venue") or _raw_v)
                _v_ts = normalize_epoch_sec(_doc.get("verified_ts"), default=None)
                if (
                    _v_n
                    and _v_ts is not None
                    and 0.0 <= (now_ts_oo - _v_ts) <= eff_bal_max_age
                    and isinstance(_doc.get("items"), dict)
                    and _doc.get("items")
                ):
                    if allow_network and (
                        latest_fill_ts_by_venue.get(_v_n, 0.0) > _v_ts + 0.5
                        or _venue_balance_cache_has_stale_order_lock(
                            _v_n,
                            _v_ts,
                            _all_c_obs,
                            cached_open_by_vs,
                            cached_open_verified_ts,
                            cached_open_sym_verified_ts,
                            cached_open_latest_ts=cached_open_latest_ts,
                            now_ts=now_ts_oo,
                        )
                    ):
                        continue
                    fresh_bal_cache_venues.add(_v_n)
            if fresh_bal_cache_venues:
                fresh_bal_cache_obs = {
                    k: obs for k, obs in _all_c_obs.items() if k[0] in fresh_bal_cache_venues
                }
                fresh_bal_cache_bd = {
                    k: bd for k, bd in _all_c_bd.items() if k[0] in fresh_bal_cache_venues
                }
        live_balance_venues: set[str] = set()
        order_ts_map = _load_open_order_timestamps(cache_dir, cached_bals)
        audit_origin_map = _load_recent_audit_origins(cache_dir) if cache_dir else {}
        _, known_ai_order_ids = _resolve_ai_intents_and_known_order_ids(
            cache_dir, known_ai_order_ids=known_ai_order_ids, read_json_fn=_rj_bals
        )
        pending_peaks_by_vs: Dict[tuple[str, str], tuple[float, Optional[float]]] = {}

        if isinstance(inst_map, dict) and inst_map:
            self._collect_from_instruments(
                inst_map,
                prices_and_trends,
                trailing_map,
                trailing_by_venue,
                cached_bals=cached_bals,
                updated_bals=updated_bals,
                cached_open_by_vs=cached_open_by_vs,
                cached_open_verified_ts=cached_open_verified_ts,
                cached_open_sym_verified_ts=cached_open_sym_verified_ts,
                updated_open_by_venue=updated_open_by_venue,
                balance_memo=balance_memo,
                known_ai_order_ids=known_ai_order_ids,
                audit_origin_map=audit_origin_map,
                order_ts_map=order_ts_map,
                now_ts_oo=now_ts_oo,
                gateway_max_usd=gateway_max_usd,
                gateway_min_usd=gateway_min_usd,
                cache_dir=cache_dir,
                allow_network=allow_network,
                force_persist=force_persist,
                tradable_positions=tradable_positions,
                open_orders_list=open_orders_list,
                venue_quote_cash=venue_quote_cash,
                inst_by_sym=inst_by_sym,
                pending_peaks_by_vs=pending_peaks_by_vs,
                live_open_ok_venues=live_open_ok_venues,
                failed_open_venues=failed_open_venues,
                live_open_ok_symbols=live_open_ok_symbols,
                failed_open_symbols=failed_open_symbols,
                explicit_now_ts=now_ts is not None,
                fresh_bal_cache_venues=fresh_bal_cache_venues,
                fresh_bal_cache_obs=fresh_bal_cache_obs,
                fresh_bal_cache_bd=fresh_bal_cache_bd,
                live_balance_venues=live_balance_venues,
            )
        elif (allow_empty_inst_recovery or not allow_network) and (
            cached_bals or cached_open_by_vs or cached_open_sym_verified_ts
        ):
            if cached_bals:
                self._recover_from_cached_balances(
                    cached_bals,
                    prices_and_trends,
                    trailing_map,
                    trailing_by_venue,
                    known_ai_order_ids=known_ai_order_ids,
                    audit_origin_map=audit_origin_map,
                    order_ts_map=order_ts_map,
                    now_ts_oo=now_ts_oo,
                    gateway_max_usd=gateway_max_usd,
                    gateway_min_usd=gateway_min_usd,
                    tradable_positions=tradable_positions,
                    open_orders_list=open_orders_list,
                    venue_quote_cash=venue_quote_cash,
                    pending_peaks_by_vs=pending_peaks_by_vs,
                    cache_dir=cache_dir,
                )
            for (v_k, sym_u_k), oo_rows in cached_open_by_vs.items():
                open_orders_list.extend(
                    _normalize_open_orders_list(
                        oo_rows,
                        v_k,
                        sym_u_k,
                        known_ai_order_ids,
                        audit_origin_map=audit_origin_map,
                        now_ts=now_ts_oo,
                        order_ts_map=order_ts_map,
                    )
                )
                if not allow_network:
                    ver_ts = safe_pos_float(cached_open_verified_ts.get(v_k), default=0.0) or 0.0
                    if ver_ts > 0.0 and 0.0 <= (now_ts_oo - ver_ts) <= 300.0:
                        live_open_ok_venues.add(v_k)
            if not allow_network:
                for (v_k, sym_u_k), sym_ver_ts in cached_open_sym_verified_ts.items():
                    if sym_ver_ts > 0.0 and 0.0 <= (now_ts_oo - sym_ver_ts) <= 300.0:
                        live_open_ok_symbols.add((v_k, sym_u_k))

        verified_now = live_open_ok_venues - failed_open_venues
        verified_symbols_now = live_open_ok_symbols - failed_open_symbols
        if isinstance(verified_open_venues, set):
            verified_open_venues.update(verified_now)
        if isinstance(verified_open_symbols, set):
            verified_open_symbols.update(verified_symbols_now)
        if allow_network and persist and cache_dir and updated_open_by_venue:
            ensure_cacheprice_classes_bound()
            open_cls = _mc.CacheOrderOpenManager
            if open_cls is not None:
                now_ms_oo = math.trunc(now_ts_oo * 1000)
                for v_k, sym_map in updated_open_by_venue.items():
                    try:
                        open_path = str(resolve_cachedb_filepath(os.path.join(cache_dir, open_cls.disk_filename(v_k))))
                        updated_syms_u = {str(s).strip().upper() for s in sym_map.keys()}
                        existing_syms_u = {
                            s_u for (ev, s_u) in cached_open_by_vs.keys() if ev == v_k
                        } | {
                            s_u for (ev, s_u) in cached_open_sym_verified_ts.keys() if ev == v_k
                        }
                        covers_known_venue_syms = not (existing_syms_u - updated_syms_u)
                        open_cls.replace_venue_open_orders_snapshot(
                            open_path,
                            sym_map,
                            now_ms=now_ms_oo,
                            verified_ts=round(now_ts_oo, 3) if (v_k in verified_now and covers_known_venue_syms) else None,
                        )
                    except Exception:
                        pass

        open_orders_list = self._dedup_open_orders(open_orders_list)
        derive_runtime_portfolio_view(
            open_orders_list,
            tradable_positions,
            venue_quote_cash,
            inst_by_sym,
            prices_and_trends,
            updated_bals,
            pending_peaks_by_vs,
            now_ts_oo=now_ts_oo,
            gateway_max_usd=gateway_max_usd,
            gateway_min_usd=gateway_min_usd,
            cache_dir=cache_dir,
            fresh_bal_cache_bd=fresh_bal_cache_bd,
            cached_bals=cached_bals,
        )

        if persist and allow_network and cache_dir and updated_bals and (
            _parse_bool_flag(force_persist, False)
            or is_live_network_backfill_enabled()
            or (bcache_path and os.path.exists(bcache_path))
        ):
            self._persist_venue_balances_locked(
                cache_dir,
                cached_bals,
                updated_bals,
                failed_open_venues,
                now_ts=now_ts_oo,
                rj_bals=_rj_bals,
                skip_balance_cache_venues=fresh_bal_cache_venues - live_balance_venues,
            )

        return tradable_positions, open_orders_list, venue_quote_cash, inst_by_sym


class SymbolBestRecentFillsCollector(PortfolioCacheCollectorBase):
    """Concrete collector (Level 3) for single-symbol recent fills across `CacheOrderFilledManager` and venue trade caches."""

    cls_name = "SymbolBestRecentFillsCollector"
    COLLECTOR_ID = "portfolio_symbol_recent_fills"

    def collect(
        self,
        cachedb_dir: str,
        symbol: str,
        *,
        quote_variants: Sequence[str] = (),
        read_json_fn: Any = None,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        _rj = _resolve_read_json_cached(read_json_fn)
        loaded_intents = _load_cached_ai_intents_list(cachedb_dir, _rj)
        known_ai_oids = collect_known_ai_order_ids(loaded_intents)
        ai_intents_list = loaded_intents or None
        audit_origin_map = _load_recent_audit_origins(cachedb_dir)
        fills: Optional[Dict[str, Any]] = None
        best_venue: Optional[str] = None
        best_fill_key: Tuple[int, int, float] = (-1, -1, -1.0)
        sym_upper = str(symbol or "").strip().upper()
        is_eq_sym = has_equity_suffix(sym_upper)
        is_crypto_pair = (not is_eq_sym) and (split_symbol_assets(sym_upper) is not None)
        scoped_variants = (
            list(filter_asset_class_variants(sym_upper, quote_variants, is_equity=is_eq_sym))
            if (is_eq_sym or is_crypto_pair)
            else list(quote_variants)
        )
        sym_quotes = compatible_quote_suffixes(sym_upper)
        base_upper = canonical_base_asset(sym_upper)
        for ord_file, cache_venue in _resolve_order_cache_files():
            is_eq_cache_v = is_equity_venue(cache_venue)
            if (is_eq_sym and not is_eq_cache_v) or (is_crypto_pair and is_eq_cache_v):
                continue
            ord_data = _read_venue_filled_orders_doc(cachedb_dir, ord_file, cache_venue, _rj)
            o_items = ord_data.get("items", {}) if isinstance(ord_data, dict) and isinstance(ord_data.get("items"), dict) else {}
            matched_cand = next((cand for cand in scoped_variants if casefold_get(o_items, cand)), None)
            orders_for_sym = casefold_get(o_items, matched_cand) if matched_cand else []
            if MarketDataProvider.merges_audit_terminal_orders(cache_venue):
                audit_items = _load_audit_terminal_orders(cachedb_dir, cache_venue)
                audit_cand = next((cand for cand in scoped_variants if casefold_get(audit_items, cand)), None)
                if audit_cand:
                    orders_for_sym = _merge_audit_orders(orders_for_sym, casefold_get(audit_items, audit_cand))
                    if not matched_cand:
                        matched_cand = audit_cand
            v_ai_intents = (
                [it for it in ai_intents_list if isinstance(it, dict) and normalize_venue_name(it.get("venue")) == cache_venue]
                if ai_intents_list is not None
                else None
            )
            if cand_fills := extract_symbol_recent_fills(
                orders_for_sym,
                known_ai_order_ids=known_ai_oids,
                audit_origin_map=audit_origin_map,
                ai_intents=v_ai_intents,
                symbol=symbol,
            ):
                cand_u = str(matched_cand or "").strip().upper()
                default_q_v = MarketDataProvider.resolve_venue_default_quote(cache_venue, fallback="USD")
                is_bare_base_cand = MarketDataProvider.uses_bare_base_order_symbols(cache_venue) and cand_u == base_upper
                default_pair_v = ensure_quote_suffix(base_upper, default_q_v)
                cand_quotes = (
                    compatible_quote_suffixes(default_pair_v)
                    if is_bare_base_cand
                    else compatible_quote_suffixes(cand_u)
                )
                same_quote_fam = 1 if (not sym_quotes or cand_quotes == sym_quotes) else 0
                sym_clean = clean_symbol_key(sym_upper)
                cand_clean = clean_symbol_key(cand_u)
                exact_sym = (
                    1
                    if (
                        cand_clean == sym_clean
                        or (is_bare_base_cand and sym_clean == default_pair_v)
                    )
                    else 0
                )
                cand_ts = max(
                    safe_nonneg_float((cand_fills.get("last_buy") or {}).get("ts_ms"), 0.0),
                    safe_nonneg_float((cand_fills.get("last_sell") or {}).get("ts_ms"), 0.0),
                )
                cand_key = (same_quote_fam, exact_sym, cand_ts)
                if fills is None or cand_key > best_fill_key:
                    cand_fills.setdefault("venue", cache_venue)
                    fills, best_venue, best_fill_key = cand_fills, cache_venue, cand_key
        if fills is None or fills.get("last_buy") is None or fills.get("last_sell") is None or is_equity_venue(best_venue):
            ws_dir = resolve_workspace_dir_from_cachedb(cachedb_dir)
            if ws_dir and os.path.isdir(ws_dir):
                k_cycs, hl_cycs, t212_cycs = harvest_venue_cycles(ws_dir, {}, {}, read_json_fn=_rj)
                cyc_maps = [
                    vc
                    for vc in (("t212", t212_cycs), ("kraken", k_cycs), ("hyperliquid", hl_cycs))
                    if not ((is_eq_sym and not is_equity_venue(vc[0])) or (is_crypto_pair and is_equity_venue(vc[0])))
                ]
                if best_venue:
                    cyc_maps.sort(key=lambda vc: 0 if vc[0] == best_venue else 1)
                cyc_cands = list(dict.fromkeys([sym_upper, *(str(c).strip().upper() for c in scoped_variants if str(c).strip())]))
                matched_cyc: Optional[Dict[str, Any]] = None
                matched_cyc_venue: Optional[str] = None
                for cand in cyc_cands:
                    if sym_quotes and compatible_quote_suffixes(cand) != sym_quotes:
                        continue
                    for c_venue, c_map in cyc_maps:
                        if isinstance((val := casefold_get(c_map, cand)), dict):
                            matched_cyc, matched_cyc_venue = val, c_venue
                            break
                    if matched_cyc is not None:
                        break
                if isinstance(matched_cyc, dict):
                    cyc_qty = safe_nonneg_float(matched_cyc.get("qty"), default=0.0) or 0.0
                    fills = apply_cycle_fallback_to_recent_fills(
                        matched_cyc_venue,
                        matched_cyc,
                        cyc_qty > 1e-8,
                        fills,
                        enrich_all_venues=True,
                        symbol=sym_upper,
                    )
        if isinstance(fills, dict) and isinstance(open_lots_raw := fills.get("open_lots"), list) and open_lots_raw:
            held_cur_f = -1.0
            if cachedb_dir and best_venue:
                _, vb_d_f, _ = load_venue_balances_state(cachedb_dir, _rj)
                if (q_cur := _extract_cached_venue_held_qty(
                    vb_d_f, best_venue, sym_upper, (sym_upper, *scoped_variants), base_upper
                )) is not None:
                    held_cur_f = q_cur
            if held_cur_f > 1e-8:
                fills["held_qty"] = round(held_cur_f, 6)
                if prune_superseded_open_lots_to_held_qty(fills, held_cur_f):
                    open_lots_raw = fills.get("open_lots") if isinstance(fills.get("open_lots"), list) else []
            fresh_lots, _ = select_fresh_open_lots(open_lots_raw, held_qty=held_cur_f)
            if fresh_lots:
                _, fresh_q = reconcile_fresh_lots_vwap(fills, open_lots_raw, fresh_lots, held_cur_f)
                fills["open_lots"] = clip_open_lots_to_held_qty(
                    fresh_lots, held_cur_f, total_qty=fresh_q
                )
        if isinstance(fills, dict):
            fills.pop("_lifo_sell_matches", None)
        return fills


class RecentOrdersAndPositionEnricherCollector(PortfolioCacheCollectorBase):
    """Concrete collector (Level 3) for fleet-wide recent fills and position cost-basis reconciliation."""

    cls_name = "RecentOrdersAndPositionEnricherCollector"
    COLLECTOR_ID = "portfolio_recent_orders"

    def collect(
        self,
        cachedb_dir: str,
        symbol: str = "*",
        *,
        inst_by_sym: Optional[Dict[str, Any]] = None,
        tradable_positions: Optional[List[Dict[str, Any]]] = None,
        prices_and_trends: Optional[Dict[str, Any]] = None,
        kraken_cycles: Optional[Dict[str, Dict[str, Any]]] = None,
        hyperliquid_cycles: Optional[Dict[str, Dict[str, Any]]] = None,
        t212_cycles: Optional[Dict[str, Dict[str, Any]]] = None,
        read_json_fn: Any = None,
        known_ai_order_ids: Optional[set[str]] = None,
        gateway_max_usd: float = 2000.0,
        gateway_min_usd: float = 0.0,
        ai_intents: Optional[List[Dict[str, Any]]] = None,
        allow_network: bool = True,
        now_ts: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        cache_dir = cachedb_dir
        inst_by_sym = inst_by_sym or {}
        tradable_positions = tradable_positions if isinstance(tradable_positions, list) else []
        prices_and_trends, kraken_cycles, hyperliquid_cycles, t212_cycles = coerce_dict_maps(
            prices_and_trends, kraken_cycles, hyperliquid_cycles, t212_cycles
        )

        _rj = _resolve_read_json_cached(read_json_fn)
        ai_intents, known_ai_order_ids = _resolve_ai_intents_and_known_order_ids(
            cache_dir,
            ai_intents=ai_intents,
            known_ai_order_ids=known_ai_order_ids,
            read_json_fn=_rj,
            need_intents=True,
        )
        audit_origin_map = _load_recent_audit_origins(cache_dir)
        recent_orders: Dict[str, Any] = {}
        venue_recent_orders: Dict[str, Dict[str, Any]] = {v: {} for v in MarketDataProvider.canonical_venues()}

        def _venue_intents(v_name: str) -> Optional[List[Dict[str, Any]]]:
            return [it for it in ai_intents if isinstance(it, dict) and normalize_venue_name(it.get("venue")) == v_name] if isinstance(ai_intents, list) else None

        def _record_venue_fills(c_venue: str, sym_k: str, fills_d: Dict[str, Any]) -> None:
            canon_k = canonical_display_symbol(sym_k, venue=c_venue) or sym_k
            fills_d.setdefault("venue", c_venue)
            venue_recent_orders[c_venue][canon_k] = fills_d
            if canon_k in recent_orders and not is_primary_stream_venue(c_venue):
                recent_orders[f"{c_venue}:{canon_k}"] = fills_d
            else:
                recent_orders.setdefault(canon_k, fills_d)

        def _scoped_sym_variants(raw_sym: Any, *, is_eq: bool) -> list[str]:
            s_u = str(raw_sym or "").strip().upper()
            if not s_u:
                return []
            return [
                str(v_s).strip().upper()
                for v_s in filter_asset_class_variants(
                    s_u,
                    (s_u, *cache_symbol_variants(s_u, include_base=True)),
                    is_equity=is_eq,
                )
            ]

        active_t212_syms: set[str] = set()
        for s_k in (t212_cycles or {}):
            active_t212_syms.update(_scoped_sym_variants(s_k, is_eq=True))
        for s_k, iv in (inst_by_sym or {}).items():
            if is_equity_venue(getattr(iv, "provider_name", None)):
                raw_s = str(getattr(iv, "symbol", "") or str(s_k).split(":", 1)[-1]).strip().upper()
                active_t212_syms.update(_scoped_sym_variants(raw_s, is_eq=True))
        for pos_it in (tradable_positions or ()):
            if isinstance(pos_it, dict) and is_equity_venue(pos_it.get("venue")):
                raw_s = str(pos_it.get("symbol") or "").strip().upper()
                active_t212_syms.update(_scoped_sym_variants(raw_s, is_eq=True))
        allowed_t212_syms = set(active_t212_syms)
        if allowed_t212_syms:
            for eq_v in MarketDataProvider.canonical_venues():
                if is_equity_venue(eq_v):
                    for it_t212 in (_venue_intents(eq_v) or ()):
                        raw_s = str(it_t212.get("symbol") or "").strip().upper()
                        allowed_t212_syms.update(_scoped_sym_variants(raw_s, is_eq=True))

        active_non_eq_syms: set[str] = set()
        for s_k, iv in (inst_by_sym or {}).items():
            if not is_equity_venue(getattr(iv, "provider_name", None)):
                raw_s = str(getattr(iv, "symbol", "") or str(s_k).split(":", 1)[-1]).strip().upper()
                if raw_s:
                    active_non_eq_syms.add(raw_s)
        for pos_it in (tradable_positions or ()):
            if isinstance(pos_it, dict) and not is_equity_venue(pos_it.get("venue")):
                if raw_s := str(pos_it.get("symbol") or "").strip().upper():
                    active_non_eq_syms.add(raw_s)
        for pt_k in (prices_and_trends or {}):
            if raw_s := str(pt_k or "").strip().upper():
                active_non_eq_syms.add(raw_s)
        for it_any in (ai_intents or ()):
            if isinstance(it_any, dict) and not is_equity_venue(it_any.get("venue")):
                if raw_s := str(it_any.get("symbol") or "").strip().upper():
                    active_non_eq_syms.add(raw_s)
        fee_reserve_sym = BinanceFeeMaintainer.BNB_SYMBOL.upper()
        ws_dir_ro = resolve_workspace_dir_from_cachedb(cache_dir) if cache_dir else ""
        disabled_ro_syms = disabled_registry_symbols(ws_dir_ro) if ws_dir_ro else frozenset()
        held_or_inst_syms: set[str] = {
            str(getattr(iv, "symbol", "") or str(s_k).split(":", 1)[-1]).strip().upper()
            for s_k, iv in (inst_by_sym or {}).items()
            if str(getattr(iv, "symbol", "") or str(s_k).split(":", 1)[-1]).strip()
        } | {
            str(pos_it.get("symbol") or "").strip().upper()
            for pos_it in (tradable_positions or ())
            if isinstance(pos_it, dict)
            and str(pos_it.get("symbol") or "").strip()
            and (
                (safe_nonneg_float(pos_it.get("free_base"), default=0.0) or 0.0)
                + (safe_nonneg_float(pos_it.get("locked_base_qty"), default=0.0) or 0.0)
            ) > 1e-8
        }

        audit_cache_by_venue: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for order_cache_name, cache_venue in _resolve_order_cache_files():
            orders_data = _read_venue_filled_orders_doc(cache_dir, order_cache_name, cache_venue, _rj)
            if isinstance(orders_data, dict) and isinstance(orders_data.get("items"), dict):
                v_ai_intents = _venue_intents(cache_venue)
                v_audit_map = (
                    audit_cache_by_venue.setdefault(cache_venue, _load_audit_terminal_orders(cache_dir, cache_venue))
                    if cache_dir and MarketDataProvider.merges_audit_terminal_orders(cache_venue)
                    else {}
                )
                for sym, o_list in orders_data["items"].items():
                    sym_u = str(sym or "").strip().upper()
                    if sym_u in disabled_ro_syms and sym_u not in held_or_inst_syms:
                        continue
                    if is_equity_venue(cache_venue) and allowed_t212_syms and sym_u not in allowed_t212_syms:
                        continue
                    if not is_equity_venue(cache_venue) and sym_u == fee_reserve_sym and sym_u not in active_non_eq_syms:
                        continue
                    merged_o_list = (
                        _merge_audit_orders(o_list, v_audit_map.get(sym_u) or v_audit_map.get(str(sym)) or [])
                        if v_audit_map and isinstance(o_list, list)
                        else o_list
                    )
                    if fills := extract_symbol_recent_fills(merged_o_list, known_ai_order_ids=known_ai_order_ids, audit_origin_map=audit_origin_map, ai_intents=v_ai_intents, symbol=sym):
                        _record_venue_fills(cache_venue, sym, fills)

        seen_live_order_vs: set[tuple[str, str]] = set()
        if allow_network:
            for inst in inst_by_sym.values():
                v_inst = normalize_venue_name(getattr(inst, "provider_name", ""))
                raw_inst_s = str(getattr(inst, "symbol", "") or "").strip()
                sym = canonical_display_symbol(raw_inst_s, venue=v_inst) or raw_inst_s.upper()
                if (
                    not sym
                    or not v_inst
                    or not MarketDataProvider.uses_bare_base_order_symbols(v_inst)
                    or (v_inst, sym.upper()) in seen_live_order_vs
                    or first_variant_dict(venue_recent_orders.get(v_inst, {}), cache_symbol_variants(sym, include_base=True)) is not None
                ):
                    continue
                seen_live_order_vs.add((v_inst, sym.upper()))
                try:
                    prov = getattr(inst, "_provider", None)
                    if prov is not None and hasattr(prov, "get_orders"):
                        if fills := extract_symbol_recent_fills(prov.get_orders(sym, None, 90 * 86400.0), known_ai_order_ids=known_ai_order_ids, audit_origin_map=audit_origin_map, ai_intents=_venue_intents(v_inst), symbol=sym):
                            _record_venue_fills(v_inst, sym, fills)
                except Exception:
                    pass
        for fallback_venue in MarketDataProvider.canonical_venues():
            fb_ai_intents = _venue_intents(fallback_venue)
            audit_by_sym = (
                audit_cache_by_venue.get(fallback_venue)
                if fallback_venue in audit_cache_by_venue
                else (
                    _load_audit_terminal_orders(cache_dir, fallback_venue)
                    if cache_dir and MarketDataProvider.merges_audit_terminal_orders(fallback_venue)
                    else {}
                )
            )
            intent_syms = {
                (canonical_display_symbol(it.get("symbol"), venue=fallback_venue) or str(it.get("symbol") or "").strip().upper())
                for it in (fb_ai_intents or ())
                if isinstance(it, dict) and str(it.get("symbol") or "").strip()
            }
            for sym_fb in sorted(set(audit_by_sym.keys()) | intent_syms):
                sym_fb_u = sym_fb.upper()
                if sym_fb_u in disabled_ro_syms and sym_fb_u not in held_or_inst_syms:
                    continue
                if not is_equity_venue(fallback_venue) and sym_fb_u == fee_reserve_sym and sym_fb_u not in active_non_eq_syms:
                    continue
                if first_variant_dict(venue_recent_orders[fallback_venue], cache_symbol_variants(sym_fb, include_base=True)) is not None:
                    continue
                o_list = audit_by_sym.get(sym_fb, [])
                if fills := extract_symbol_recent_fills(o_list, known_ai_order_ids=known_ai_order_ids, audit_origin_map=audit_origin_map, ai_intents=fb_ai_intents, symbol=sym_fb):
                    _record_venue_fills(fallback_venue, sym_fb, fills)
        live_held_by_venue_sym: Dict[tuple[str, str], float] = {}
        for pos_item, pv, ps in _iter_valid_tradable_positions(tradable_positions):
            fb = safe_float(pos_item.get("free_base"), -1.0)
            lb = safe_pos_float(pos_item.get("locked_base_qty"), default=0.0) or 0.0
            if fb >= 0.0 or lb > 0.0:
                held_q = (max(0.0, fb) + lb) if fb >= 0.0 else lb
                for sv in _scoped_sym_variants(ps, is_eq=is_equity_venue_or_suffix(ps, pv)):
                    sv_k = (pv, sv.upper())
                    if sv_k not in live_held_by_venue_sym or held_q > live_held_by_venue_sym[sv_k]:
                        live_held_by_venue_sym[sv_k] = held_q
        _, cached_bals_ro, _ = load_venue_balances_state(cache_dir, _rj)
        for c_venue_ro, cyc_map_ro in (("kraken", kraken_cycles), ("hyperliquid", hyperliquid_cycles), ("t212", t212_cycles)):
            for t_sym, t_cyc in (cyc_map_ro or {}).items():
                t_sym_u = str(t_sym).strip().upper()
                if not t_sym_u or (c_venue_ro, t_sym_u) in live_held_by_venue_sym or not isinstance(t_cyc, dict):
                    continue
                ro_cands = _scoped_sym_variants(t_sym_u, is_eq=is_equity_venue_or_suffix(t_sym_u, c_venue_ro))
                held_c = _extract_cached_venue_held_qty(
                    cached_bals_ro, c_venue_ro, t_sym_u, ro_cands, canonical_base_asset(t_sym_u)
                )
                if held_c is not None:
                    for sv_c in ro_cands:
                        live_held_by_venue_sym.setdefault((c_venue_ro, sv_c.upper()), held_c)
                    live_held_by_venue_sym[(c_venue_ro, t_sym_u)] = held_c
                    if is_equity_venue(c_venue_ro) and held_c > 1e-8:
                        t_cyc["qty"] = round(held_c, 6)
                        t_px = resolve_symbol_price(
                            ensure_equity_suffix(t_sym_u),
                            prices_and_trends,
                        )
                        if (t_avg := first_pos_float(t_cyc.get("avg_cost_px"), t_cyc.get("entry_price"))) is not None:
                            t_cyc["cost_usd"] = round(held_c * t_avg, 2)
                            if t_px > 0.0:
                                t_cyc["pnl_vs_avg_pct"] = t_cyc["dd_pct"] = round(((t_px - t_avg) / t_avg) * 100.0, 2) + 0.0
                        if t_px > 0.0 and (t_pk := safe_pos_float(t_cyc.get("pos_peak"), default=None)) is not None:
                            eff_t_pk = max(t_pk, t_px)
                            t_cyc["pos_peak"] = round(eff_t_pk, 6)
                            t_cyc["dd_from_peak_pct"] = round(((t_px - eff_t_pk) / eff_t_pk) * 100.0, 2) + 0.0
        cyc_sell_ts_map = reconcile_venue_cycles_with_recent_orders(
            kraken_cycles, hyperliquid_cycles, t212_cycles, venue_recent_orders, live_held_by_venue_sym, prices_and_trends
        )
        _archive_high_fn = (
            (lambda cands, oldest_ms, base_hi: scan_cached_price_archive_high_since(
                cache_dir, cands, oldest_ms, baseline_high=base_hi
            ))
            if cache_dir
            else None
        )
        _fee_spec_fn = build_persisted_venue_fee_resolver(cache_dir, now_ts=now_ts)
        enrich_positions_with_orders_and_cycles(
            tradable_positions,
            prices_and_trends,
            inst_by_sym,
            venue_recent_orders,
            recent_orders,
            kraken_cycles,
            hyperliquid_cycles,
            t212_cycles,
            cyc_sell_ts_map,
            gateway_max_usd=gateway_max_usd,
            gateway_min_usd=gateway_min_usd,
            now_ts=now_ts,
            archive_high_resolver=_archive_high_fn,
            fee_spec_resolver=_fee_spec_fn,
        )
        for ro_val in recent_orders.values():
            if isinstance(ro_val, dict):
                ro_val.pop("_lifo_sell_matches", None)
        return recent_orders


class BotRuntimeStatesCollector(PortfolioCacheCollectorBase):
    """Concrete collector (Level 3) for order_retry_queue, monitortrades, assetguardian, and smart_stop_loss states."""

    cls_name = "BotRuntimeStatesCollector"
    COLLECTOR_ID = "portfolio_bot_runtime_states"

    def collect(
        self,
        cachedb_dir: str,
        symbol: str = "*",
        *,
        now_ts: Any = None,
        read_json_fn: Any = None,
        tradable_positions: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        cache_dir = cachedb_dir
        eff_now, _rj = _resolve_now_and_read_json(now_ts, read_json_fn)
        out: Dict[str, Any] = {}
        retry_records = _oq.read_tail_queue_records(cache_dir, limit=50)
        if retry_records:
            terminal_ai_intent_ids: set[str] = set()
            for it in _load_cached_ai_intents_list(cache_dir, _rj):
                st_u = str(it.get("status") or "").strip().upper()
                if st_u in {
                    "EXECUTED",
                    "FAILED",
                    "CANCELED",
                    "CANCELLED",
                    "EXPIRED",
                    "REJECTED",
                    "BLOCKED_BY_GUARD",
                    "SKIPPED_DUPLICATE_OPEN_ORDER",
                    "OBSERVED_SHADOW",
                } or st_u.startswith("REJECTED_"):
                    if iid := str(it.get("intent_id") or "").strip():
                        terminal_ai_intent_ids.add(iid)
            active_retries: List[Dict[str, Any]] = []
            for rec in retry_records:
                rec_iid = str(rec.get("intent_id") or "").strip()
                if rec_iid and rec_iid in terminal_ai_intent_ids:
                    continue
                lifecycle = str(rec.get("lifecycle") or "submit_pending").lower()
                exp_ts = normalize_epoch_sec(rec.get("expires_at_ts"), default=0.0, allow_zero=True, allow_iso=True) or 0.0
                checked_oq_expiry = False
                if _oq is not None and (rec.get("created_ts") is not None or rec.get("ttl_started_ts") is not None):
                    try:
                        checked_oq_expiry = True
                        if _oq.is_expired(rec, eff_now):
                            continue
                    except Exception:
                        checked_oq_expiry = False
                if (
                    lifecycle == "submit_pending"
                    and str(rec.get("submission_state") or "").lower() == "refused"
                    and _parse_bool_flag(rec.get("safe_to_discard"), default=False)
                    and (safe_float(rec.get("attempts"), 0.0) or 0.0) >= 5.0
                    and str(rec.get("last_failure_reason") or "").lower()
                    in {"qty_zero_after_policy", "qty_zero_after_weight", "qty_zero_or_negative", "below_min_notional"}
                ):
                    continue
                if lifecycle in ("submit_pending", "awaiting_cancel", "pending", "active") and (
                    checked_oq_expiry or lifecycle == "awaiting_cancel" or exp_ts == 0.0 or exp_ts > eff_now
                ):
                    raw_rq_v = rec.get("provider_name") if rec.get("provider_name") is not None else rec.get("venue")
                    rq_item: Dict[str, Any] = {
                        "venue": normalize_venue_name(raw_rq_v) or str(raw_rq_v or "").strip().lower(),
                        **{k: rec.get(k) for k in ("symbol", "side", "qty", "requested_price", "kind", "attempts", "last_failure_reason")},
                    }
                    if (req_px := safe_pos_float(rec.get("requested_price"), default=None)) is not None:
                        rq_item["requested_price"] = round(req_px, 6)
                    c_ts = normalize_epoch_sec(rec.get("created_ts"), default=0.0, allow_zero=True, allow_iso=True) or 0.0
                    if c_ts > 0.0 and eff_now >= c_ts:
                        rq_item["age_min"] = round((eff_now - c_ts) / 60.0, 1)
                    active_retries.append(rq_item)
            if active_retries:
                out["active_retry_queue"] = active_retries

        if isinstance(mt_state := MtStateStore.read_from_cachedb(cache_dir, read_json_fn=_rj), dict) and mt_state:
            active_mt = {
                k: v for k in ("tp_ride_state", "sl_bounce_state", "sl_episode_state", "reentry_state", "pending_stop_exit")
                if isinstance(v := mt_state.get(k), dict) and v
            }
            if active_mt:
                out["monitortrades_active_state"] = active_mt

        ag_state = AssetguardianStateStore.read_from_cachedb(cache_dir, read_json_fn=_rj)
        if isinstance(ag_state, dict) and isinstance(ag_state.get("symbols"), dict) and ag_state["symbols"]:
            dust_binance_syms = {
                str(pos.get("symbol") or "").upper()
                for pos in (tradable_positions or ())
                if isinstance(pos, dict) and is_primary_stream_venue(pos.get("venue"))
                and (
                    (
                        0.0 <= safe_float(pos.get("value_usd"), -1.0)
                        and (safe_float(pos.get("value_usd"), 0.0) + max(0.0, safe_float(pos.get("locked_base_usd"), 0.0)))
                        < max(1.0, first_pos_float(pos.get("exchange_min_notional_usd"), pos.get("min_notional_usd"), default=5.0) or 5.0)
                    )
                    or (
                        0.0 <= safe_float(pos.get("free_base"), -1.0) <= 1e-8
                        and (safe_pos_float(pos.get("locked_base_qty"), default=0.0) or 0.0) <= 1e-8
                    )
                )
            }
            cleaned_ag_syms: Dict[str, Any] = {}
            for ag_sym, ag_val in ag_state["symbols"].items():
                if not isinstance(ag_val, dict):
                    continue
                sym_copy = dict(ag_val)
                if str(ag_sym).upper() in dust_binance_syms and "sell" in sym_copy:
                    if not isinstance(sell_st := sym_copy.get("sell"), dict) or not sell_st.get("pending"):
                        sym_copy.pop("sell", None)
                if isinstance(sell_d := sym_copy.get("sell"), dict):
                    sell_copy = dict(sell_d)
                    for qk in ("total_filled_qty", "initial_qty"):
                        if isinstance(sell_copy.get(qk), float):
                            sell_copy[qk] = round(sell_copy[qk], 8)
                    sym_copy["sell"] = sell_copy
                if sym_copy:
                    cleaned_ag_syms[ag_sym] = sym_copy
            if cleaned_ag_syms:
                out["assetguardian_state"] = cleaned_ag_syms

        if isinstance(ssl_state := SmartStopLossStateStore.read_from_cachedb(cache_dir, read_json_fn=_rj), dict) and ssl_state:
            out["smart_stop_loss_state"] = ssl_state
        return out


for _port_col_cls in (
    VenueTrailingMapsCollector,
    VenueCyclesCollector,
    KrakenXstocksCollector,
    TradablePositionsAndOrdersCollector,
    SymbolBestRecentFillsCollector,
    RecentOrdersAndPositionEnricherCollector,
    BotRuntimeStatesCollector,
):
    TelemetryCollectorFactory.register(_port_col_cls.COLLECTOR_ID, _port_col_cls)


def harvest_venue_cycles(
    workspace_dir: str,
    prices_and_trends: Dict[str, Any],
    fallback_prices: Dict[str, float],
    read_json_fn: Any = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Harvest normalized ``.state_*.json`` DCA cycle states for Kraken, Hyperliquid, and Trading212."""
    return TelemetryCollectorFactory.create("portfolio_venue_cycles").collect(
        workspace_dir,
        "*",
        workspace_dir=workspace_dir,
        prices_and_trends=prices_and_trends,
        fallback_prices=fallback_prices,
        read_json_fn=read_json_fn,
    )


def harvest_kraken_xstocks(
    kraken_dir: str,
    prices_and_trends: Dict[str, Any],
    all_t212_cycles: Dict[str, Any],
    read_json_fn: Any = None,
    *,
    kraken_cycles: Optional[Dict[str, Any]] = None,
    cachedb_dir: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Harvest Kraken read-only xStocks holdings and link to Trading212 sibling cycle."""
    return TelemetryCollectorFactory.create("portfolio_kraken_xstocks").collect(
        cachedb_dir or kraken_dir,
        "*",
        kraken_dir=kraken_dir,
        prices_and_trends=prices_and_trends,
        all_t212_cycles=all_t212_cycles,
        kraken_cycles=kraken_cycles,
        read_json_fn=read_json_fn,
    )


def harvest_tradable_positions_and_orders(
    inst_map: Dict[str, Any],
    prices_and_trends: Dict[str, Any],
    trailing_map: Dict[str, Dict[str, Any]],
    trailing_by_venue: Dict[str, Dict[str, Dict[str, Any]]],
    *,
    known_ai_order_ids: Optional[set[str]] = None,
    gateway_max_usd: float = 2000.0,
    gateway_min_usd: float = 0.0,
    cache_dir: Optional[str] = None,
    verified_open_venues: Optional[set[str]] = None,
    verified_open_symbols: Optional[set[Tuple[str, str]]] = None,
    allow_empty_inst_recovery: bool = False,
    allow_network: bool = True,
    persist: bool = True,
    force_persist: bool = False,
    now_ts: Optional[float] = None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Harvest spot wallet balances, trailing peaks, and open orders across tradable instruments."""
    return TelemetryCollectorFactory.create("portfolio_tradable_positions", start_sync=allow_network).collect(
        cache_dir or "",
        "*",
        inst_map=inst_map,
        prices_and_trends=prices_and_trends,
        trailing_map=trailing_map,
        trailing_by_venue=trailing_by_venue,
        known_ai_order_ids=known_ai_order_ids,
        gateway_max_usd=gateway_max_usd,
        gateway_min_usd=gateway_min_usd,
        verified_open_venues=verified_open_venues,
        verified_open_symbols=verified_open_symbols,
        allow_empty_inst_recovery=allow_empty_inst_recovery,
        allow_network=allow_network,
        persist=persist,
        force_persist=force_persist,
        now_ts=now_ts,
    )


def load_symbol_best_recent_fills(
    cachedb_dir: str,
    symbol: str,
    quote_variants: Sequence[str],
    read_json_fn: Any = None,
) -> Optional[Dict[str, Any]]:
    """Load the most recent venue fills dict for `symbol` across order caches and execution_audit fallback."""
    return TelemetryCollectorFactory.create("portfolio_symbol_recent_fills", start_sync=False).collect(
        cachedb_dir,
        symbol,
        quote_variants=quote_variants,
        read_json_fn=read_json_fn,
    )


def harvest_recent_orders_and_enrich_positions(
    cache_dir: str,
    inst_by_sym: Dict[str, Any],
    tradable_positions: List[Dict[str, Any]],
    prices_and_trends: Dict[str, Any],
    kraken_cycles: Dict[str, Dict[str, Any]],
    hyperliquid_cycles: Dict[str, Dict[str, Any]],
    t212_cycles: Dict[str, Dict[str, Any]],
    read_json_fn: Any = None,
    *,
    known_ai_order_ids: Optional[set[str]] = None,
    gateway_max_usd: float = 2000.0,
    gateway_min_usd: float = 0.0,
    ai_intents: Optional[List[Dict[str, Any]]] = None,
    allow_network: bool = True,
    now_ts: Optional[float] = None,
) -> Dict[str, Any]:
    """Load recent fills across Binance, Kraken, Hyperliquid, and T212 and enrich tradable_positions with cost-basis anchors."""
    return TelemetryCollectorFactory.create("portfolio_recent_orders", start_sync=allow_network).collect(
        cache_dir,
        "*",
        inst_by_sym=inst_by_sym,
        tradable_positions=tradable_positions,
        prices_and_trends=prices_and_trends,
        kraken_cycles=kraken_cycles,
        hyperliquid_cycles=hyperliquid_cycles,
        t212_cycles=t212_cycles,
        read_json_fn=read_json_fn,
        known_ai_order_ids=known_ai_order_ids,
        gateway_max_usd=gateway_max_usd,
        gateway_min_usd=gateway_min_usd,
        ai_intents=ai_intents,
        allow_network=allow_network,
        now_ts=now_ts,
    )


def harvest_bot_runtime_states(
    cache_dir: str,
    now_ts: Any = None,
    read_json_fn: Any = None,
    tradable_positions: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Harvest active retry queue, monitortrades, assetguardian, and smart_stop_loss states."""
    return TelemetryCollectorFactory.create("portfolio_bot_runtime_states").collect(
        cache_dir,
        "*",
        now_ts=now_ts,
        read_json_fn=read_json_fn,
        tradable_positions=tradable_positions,
    )


__all__ = [
    "BotRuntimeStatesCollector",
    "KrakenXstocksCollector",
    "PortfolioCacheCollectorBase",
    "RecentOrdersAndPositionEnricherCollector",
    "SymbolBestRecentFillsCollector",
    "TradablePositionsAndOrdersCollector",
    "VenueCyclesCollector",
    "VenueTrailingMapsCollector",
    "_iter_audit_dir_file_batches",
    "_iter_t212_state_files",
    "_load_audit_terminal_orders",
    "_load_open_order_timestamps",
    "_load_recent_audit_origins",
    "_normalize_open_orders_list",
    "extract_symbol_recent_fills",
    "harvest_bot_runtime_states",
    "harvest_kraken_xstocks",
    "harvest_recent_orders_and_enrich_positions",
    "harvest_tradable_positions_and_orders",
    "harvest_venue_cycles",
    "load_symbol_best_recent_fills",
    "parse_venue_cycle_state",
    "reconcile_venue_cycles_with_recent_orders",
]
