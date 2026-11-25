#!/usr/bin/env python3
"""Idempotent 1:1 historical sync & parity verifier from Stage-1 trade caches into CacheOrderFilledManager.

Usage:
    myenv/bin/python verify_tools/sync_order_filled_from_trade_cache.py [--check-only] [--cachedb-dir PATH]

Behavior:
  - For Kraken, Hyperliquid, and T212: loads ``cache_trade_<venue>.json`` and merges every
    historical fill into ``cache_order_filled_<venue>.json`` via
    ``CacheOrderFilledManager.merge_venue_filled_orders_snapshot`` (which automatically
    subsumes provisional Kraken Ledger rows and upgrades settled T212 fees).
  - For Binance: reads ``cache_trade_binance.jsonl`` via ``CacheTradeManager`` and enriches
    historical rows in ``cache_order_filled_binance.json`` (backfilling missing ``fee``,
    ``feeAsset``, ``clientOrderId`` and any missing historical orders) through
    ``CacheOrderFilledManager.save_state_to_file()`` so the schema-v2 ``.meta`` manifest
    and ``data_version`` remain certified.
  - Verifies 1:1 parity across all 4 venues (order counts, quantities, VWAP prices, fees,
    ``get_last_opposite_fill_price``, and Kraken ``summarize_fills_position_and_cost``).
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from botcore import (
    add_store_true_cli_args,
    clean_upper_scalar,
    extract_order_price,
    extract_order_qty,
    extract_order_side,
    extract_timestamp_ms,
    safe_nonneg_float,
    safe_pos_float,
)
from cacheaccount import (
    CacheOrderFilledManager,
    CacheTradeManager,
    _enrich_filled_row_metadata,
    _merge_optional_order_fee,
)
from cachepaths import (
    CACHE_BINANCE_ORDER_FILLED_FILENAME,
    CACHE_BINANCE_TRADE_FILENAME,
    CACHE_HYPERLIQUID_ORDER_FILLED_FILENAME,
    CACHE_HYPERLIQUID_TRADE_FILENAME,
    CACHE_KRAKEN_ORDER_FILLED_FILENAME,
    CACHE_KRAKEN_TRADE_FILENAME,
    CACHE_T212_ORDER_FILLED_FILENAME,
    CACHE_T212_TRADE_FILENAME,
    get_cachedb_dir,
    resolve_cachedb_filepath,
    split_symbol_assets,
)
from providers.order_id_context import extract_client_order_id
from providers.quantity import summarize_fills_position_and_cost
from state_io import read_json_dict_recovering


_JSON_VENUES = (
    ("kraken", CACHE_KRAKEN_TRADE_FILENAME, CACHE_KRAKEN_ORDER_FILLED_FILENAME),
    ("hyperliquid", CACHE_HYPERLIQUID_TRADE_FILENAME, CACHE_HYPERLIQUID_ORDER_FILLED_FILENAME),
    ("t212", CACHE_T212_TRADE_FILENAME, CACHE_T212_ORDER_FILLED_FILENAME),
)


def _sync_json_venue(
    cachedb_dir: str,
    venue: str,
    trade_fname: str,
    filled_fname: str,
    *,
    apply_changes: bool,
) -> dict[str, Any]:
    trade_path = str(resolve_cachedb_filepath(os.path.join(cachedb_dir, trade_fname)))
    filled_path = str(resolve_cachedb_filepath(os.path.join(cachedb_dir, filled_fname)))
    trade_doc = read_json_dict_recovering(trade_path, default={}, repair_files=False)
    trade_items = trade_doc.get("items") if isinstance(trade_doc, dict) and isinstance(trade_doc.get("items"), dict) else {}
    trade_ft = trade_doc.get("fetchtime") if isinstance(trade_doc, dict) and isinstance(trade_doc.get("fetchtime"), dict) else {}

    if apply_changes and trade_items:
        CacheOrderFilledManager.merge_venue_filled_orders_snapshot(
            filled_path,
            trade_items,
            venue=venue,
            is_snapshot=False,
            force_save=True,
        )

    filled_items, _ = CacheOrderFilledManager.read_venue_filled_orders_snapshot(
        filled_path, venue=venue, allow_legacy=False, allow_trade_fallback=False
    )
    diffs: list[str] = []
    for sym, t_rows in sorted(trade_items.items()):
        if not isinstance(t_rows, list) or not t_rows:
            continue
        expected_rows = CacheOrderFilledManager.aggregate_fills_to_order_rows(t_rows, venue=venue)
        actual_rows = filled_items.get(sym) if isinstance(filled_items.get(sym), list) else []
        actual_by_oid = {str(r.get("orderId")): r for r in actual_rows if isinstance(r, dict)}
        for exp in expected_rows:
            oid_s = str(exp.get("orderId"))
            act = actual_by_oid.get(oid_s)
            if act is None:
                diffs.append(f"{venue}:{sym}:missing_oid={oid_s}")
                continue
            exp_q = safe_pos_float(exp.get("quantity"), default=0.0) or 0.0
            act_q = safe_pos_float(act.get("quantity"), default=0.0) or 0.0
            if abs(exp_q - act_q) > 1e-6:
                diffs.append(f"{venue}:{sym}:oid={oid_s}:qty_diff({act_q}!={exp_q})")
            exp_px = safe_pos_float(exp.get("price"), default=0.0) or 0.0
            act_px = safe_pos_float(act.get("price"), default=0.0) or 0.0
            if abs(exp_px - act_px) > 1e-5:
                diffs.append(f"{venue}:{sym}:oid={oid_s}:px_diff({act_px}!={exp_px})")
            exp_fee = safe_nonneg_float(exp.get("fee"), default=0.0) or 0.0
            act_fee = safe_nonneg_float(act.get("fee"), default=0.0) or 0.0
            if exp_fee > 0.0 and abs(exp_fee - act_fee) > 1e-6:
                diffs.append(f"{venue}:{sym}:oid={oid_s}:fee_diff({act_fee}!={exp_fee})")
        base_a, quote_a = split_symbol_assets(sym) or ("", "")
        t_pos = summarize_fills_position_and_cost(t_rows, base_asset=base_a, quote_asset=quote_a)
        f_pos = summarize_fills_position_and_cost(actual_rows, base_asset=base_a, quote_asset=quote_a)
        if t_pos and f_pos:
            if abs((t_pos.get("qty") or 0.0) - (f_pos.get("qty") or 0.0)) > 1e-6:
                diffs.append(f"{venue}:{sym}:pos_qty_diff({f_pos.get('qty')}!={t_pos.get('qty')})")
            if abs((t_pos.get("cost") or 0.0) - (f_pos.get("cost") or 0.0)) > 1e-2:
                diffs.append(f"{venue}:{sym}:pos_cost_diff({f_pos.get('cost')}!={t_pos.get('cost')})")
    return {
        "venue": venue,
        "trade_symbols": len(trade_items),
        "filled_symbols": len(filled_items),
        "trade_fetchtimes": len(trade_ft),
        "diffs": diffs,
    }


def _sync_binance(cachedb_dir: str, *, apply_changes: bool) -> dict[str, Any]:
    trade_path = str(resolve_cachedb_filepath(os.path.join(cachedb_dir, CACHE_BINANCE_TRADE_FILENAME)))
    filled_path = str(resolve_cachedb_filepath(os.path.join(cachedb_dir, CACHE_BINANCE_ORDER_FILLED_FILENAME)))
    if not os.path.isfile(trade_path) or not os.path.isfile(filled_path):
        return {"venue": "binance", "skipped": True, "diffs": []}

    tm = CacheTradeManager(3600, [], trade_path, read_only=True, venue="binance")
    fm = CacheOrderFilledManager(3600, [], filled_path, read_only=not apply_changes, venue="binance")
    if apply_changes:
        fm.save_state = True
    enriched_count = 0
    added_count = 0
    if apply_changes:
        with tm.lock, fm.lock:
            for sym, raw_trades in tm.cache.items():
                if not isinstance(raw_trades, list) or not raw_trades:
                    continue
                by_oid_fills: dict[int, list[dict[str, Any]]] = {}
                for tr in raw_trades:
                    if not isinstance(tr, dict):
                        continue
                    oid_norm = fm._normalize_order_id(tr.get("orderId"), strict_int=True)
                    if oid_norm is None:
                        continue
                    by_oid_fills.setdefault(oid_norm, []).append(tr)
                existing_oids = {
                    fm._normalize_order_id(r.get("orderId"), strict_int=True)
                    for r in fm.cache.get(sym, [])
                    if isinstance(r, dict)
                }
                for oid_norm, fills in by_oid_fills.items():
                    if oid_norm in existing_oids:
                        continue
                    order_fills = []
                    for f in fills:
                        row: dict[str, Any] = {
                            "orderId": oid_norm,
                            "price": extract_order_price(f),
                            "quantity": extract_order_qty(f),
                            "timestamp": extract_timestamp_ms(f, default=0) or 0,
                            "side": extract_order_side(f),
                            "_fillId": str(f["id"]),
                        }
                        if f_cid := extract_client_order_id(f, default=""):
                            row["clientOrderId"] = f_cid
                        if (f_val := safe_nonneg_float(f.get("commission"), default=None)) is not None:
                            row["fee"] = f_val
                        if f_asset := clean_upper_scalar(f.get("commissionAsset")):
                            row["feeAsset"] = f_asset
                        order_fills.append(row)
                    if order_fills:
                        fm._persist_items(sym, order_fills)
                        existing_oids.add(oid_norm)
                        added_count += 1
                existing_by_oid = {
                    fm._normalize_order_id(r.get("orderId"), strict_int=True): r
                    for r in fm.cache.get(sym, [])
                    if isinstance(r, dict)
                }
                for oid_norm, fills in by_oid_fills.items():
                    ex_row = existing_by_oid.get(oid_norm)
                    if ex_row is None:
                        continue
                    agg_fee: dict[str, Any] = {}
                    cid = ""
                    for f in fills:
                        if not cid:
                            cid = extract_client_order_id(f, default="")
                        f_val = safe_nonneg_float(f.get("commission"), default=None)
                        f_asset = clean_upper_scalar(f.get("commissionAsset"))
                        if f_val is not None:
                            _merge_optional_order_fee(agg_fee, {"fee": f_val, "feeAsset": f_asset}, agg_fee)
                    meta_inc: dict[str, Any] = {}
                    if cid:
                        meta_inc["clientOrderId"] = cid
                    if "fee" in agg_fee:
                        meta_inc["fee"] = agg_fee["fee"]
                    if "feeAsset" in agg_fee:
                        meta_inc["feeAsset"] = agg_fee["feeAsset"]
                    if _enrich_filled_row_metadata(ex_row, meta_inc, allow_fee_upgrade=True):
                        enriched_count += 1
        if enriched_count > 0 or added_count > 0:
            saved = fm.save_state_to_file()
            if not saved:
                raise RuntimeError("Failed to persist certified BinanceOrderFilled snapshot")

    diffs: list[str] = []
    for sym in sorted(set(tm.cache.keys()) | set(fm.cache.keys())):
        for side in ("BUY", "SELL"):
            t_opp = tm.get_last_opposite_fill_price(sym, side)
            f_opp = fm.get_last_opposite_fill_price(sym, side)
            if (t_opp is None) != (f_opp is None):
                diffs.append(f"binance:{sym}:{side}:last_opp_presence({f_opp}!={t_opp})")
            elif t_opp is not None and f_opp is not None and abs(t_opp - f_opp) > 1e-4:
                # Allow minor VWAP vs last-partial-fill delta only if reported explicitly.
                diffs.append(f"binance:{sym}:{side}:last_opp_px({f_opp}!={t_opp})")
        t_by_oid: dict[int, list[dict[str, Any]]] = {}
        for r in (tm.cache.get(sym) or ()):
            if isinstance(r, dict) and (oid_n := fm._normalize_order_id(r.get("orderId"), strict_int=True)) is not None:
                t_by_oid.setdefault(oid_n, []).append(r)
        t_oids = set(t_by_oid.keys())
        f_rows = {
            fm._normalize_order_id(r.get("orderId"), strict_int=True): r
            for r in (fm.cache.get(sym) or ())
            if isinstance(r, dict)
        }
        missing_oids = sorted(t_oids - set(f_rows.keys()))
        if missing_oids:
            diffs.append(f"binance:{sym}:missing_oids={len(missing_oids)}")
        missing_fee = sum(
            1
            for oid, fs in t_by_oid.items()
            if oid in f_rows
            and any(clean_upper_scalar(f.get("commissionAsset")) for f in fs)
            and (f_rows[oid].get("fee") is None or not f_rows[oid].get("feeAsset"))
        )
        if missing_fee:
            diffs.append(f"binance:{sym}:missing_fee_metadata={missing_fee}")
    return {
        "venue": "binance",
        "enriched_rows": enriched_count,
        "added_orders": added_count,
        "diffs": diffs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill and verify 1:1 CacheOrderFilledManager parity.")
    parser.add_argument("--cachedb-dir", default=get_cachedb_dir(), help="Path to cachedb directory.")
    add_store_true_cli_args(parser, ("--check-only", "Verify parity without modifying files."))
    args = parser.parse_args(argv)
    apply_changes = not bool(args.check_only)

    all_diffs: list[str] = []
    for venue, trd_f, fld_f in _JSON_VENUES:
        res = _sync_json_venue(args.cachedb_dir, venue, trd_f, fld_f, apply_changes=apply_changes)
        print(f"[{venue.upper()}] trade_syms={res['trade_symbols']} filled_syms={res['filled_symbols']} diffs={len(res['diffs'])}")
        for d in res["diffs"][:20]:
            print(f"  DIFF: {d}")
        all_diffs.extend(res["diffs"])

    bn_res = _sync_binance(args.cachedb_dir, apply_changes=apply_changes)
    print(
        f"[BINANCE] enriched={bn_res.get('enriched_rows', 0)} added={bn_res.get('added_orders', 0)} "
        f"diffs={len(bn_res['diffs'])}"
    )
    for d in bn_res["diffs"][:20]:
        print(f"  DIFF: {d}")
    all_diffs.extend(bn_res["diffs"])
    return 1 if all_diffs else 0


if __name__ == "__main__":
    raise SystemExit(main())
