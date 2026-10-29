#!/usr/bin/env python3
# check_cache_coherence.py — checks the FRESHNESS and consistency of trend/price caches
# for ALL symbols (Binance and non-Binance entries from instruments.conf). Exits 1 if anything is stale.
# Run periodically (cron/loop) to detect a frozen cache.
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from botcore import extract_current_price_item, safe_pos_float  # noqa: E402
from cachepaths import cache_symbol_variants, resolve_cachedb_filepath  # noqa: E402
from state_io import previous_generation_path, read_json_dict  # noqa: E402

CACHEDB = os.path.join(ROOT, "cachedb")
STALE_S = 600.0  # More than 10 minutes without an update is STALE (instant trend refreshes every few seconds / 20s).


def load(name):
    p = str(resolve_cachedb_filepath(os.path.join(CACHEDB, name)))
    if not os.path.exists(p) and not os.path.exists(previous_generation_path(p)):
        return None
    try:
        return read_json_dict(p, recover_previous=True)
    except Exception as e:  # noqa: BLE001
        print(f"  [ERR] {name}: {e}")
        return None


def expected_symbols():
    """Use the registry without credentials; invalid configuration must fail the check."""
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from instrument_registry import select_instruments, symbols_for
    syms = symbols_for("binance")
    for inst in select_instruments(role="mt").values():
        if inst.provider != "binance" and inst.symbol not in syms:
            syms.append(inst.symbol)
    if not syms:
        raise ValueError("No configured symbols to verify")
    return syms


def main() -> int:
    now = time.time()
    problems = []
    print(f"=== cache coherence @ {time.strftime('%Y-%m-%d %H:%M:%S')} (stale>{int(STALE_S)}s) ===")
    exp = expected_symbols()
    print(f"expected symbols: {exp}")

    # 1. short trend
    raw_st = load("cache_short_trend.json") or {}
    it = raw_st.get("items") if isinstance(raw_st.get("items"), dict) else raw_st
    print(f"-- cache_short_trend.json: {sorted(it.keys())}")
    for s in exp:
        e = None
        for cand in cache_symbol_variants(s, include_original=True, include_base=True):
            cand_entry = it.get(cand)
            if isinstance(cand_entry, dict) and cand_entry:
                e = cand_entry
                break
        if not isinstance(e, dict) or not e:
            problems.append(f"short_trend MISSING {s}")
            print(f"   {s:10} IS MISSING")
            continue
        ts_val = safe_pos_float(e.get("ts"), default=0.0)
        age = now - ts_val if ts_val > 0 else 1e9
        px = e.get("current_price")
        px_ok = safe_pos_float(px, default=None) is not None
        ok = -5.0 <= age < STALE_S and px_ok
        if not ok:
            problems.append(f"short_trend {s} age={age:.0f}s px={px}")
        print(
            f"   {s:10} age={age:6.0f}s growth_coefficient={e.get('growth_coefficient')} "
            f"gradient_recent={e.get('gradient_recent')} price={px} -> {'OK' if ok else 'STALE/INVALID'}"
        )

    # 2. current price
    cp = load("cache_current_price.json") or {}
    # format: {'fetchtime': {...}, 'items': {symbol: price_or_rows}} (or directly per symbol)
    raw_items = cp.get("items", cp) if isinstance(cp, dict) else {}
    items = raw_items if isinstance(raw_items, dict) else {}
    raw_fetchtime = cp.get("fetchtime") if isinstance(cp, dict) else {}
    fetchtime_map = raw_fetchtime if isinstance(raw_fetchtime, dict) else {}
    print(f"-- cache_current_price.json: {sorted(k for k in items.keys() if k != 'fetchtime')}")
    for s in exp:
        entries = None
        matched_key = s
        for cand in cache_symbol_variants(s, include_original=True, include_base=True):
            if cand in items and items.get(cand) is not None:
                entries = items.get(cand)
                matched_key = cand
                break
        if entries is None or (isinstance(entries, (list, dict, str)) and not entries):
            problems.append(f"currentprice MISSING {s}")
            print(f"   {s:10} IS MISSING")
            continue

        px_val = extract_current_price_item(entries) or 0.0
        ts_raw = None
        if isinstance(entries, list):
            last_entry = entries[-1] if entries else None
            if not isinstance(last_entry, (list, tuple)) or len(last_entry) < 2:
                problems.append(f"currentprice INVALID {s}")
                print(f"   {s:10} IS INVALID")
                continue
            ts_raw = last_entry[0]
        elif isinstance(entries, dict):
            ts_raw = (
                entries.get("ts")
                or entries.get("timestamp")
                or entries.get("time")
                or fetchtime_map.get(matched_key)
                or fetchtime_map.get(s)
            )
        else:
            ts_raw = fetchtime_map.get(matched_key) or fetchtime_map.get(s)

        if px_val <= 0:
            problems.append(f"currentprice INVALID {s}")
            print(f"   {s:10} IS INVALID")
            continue

        ts_ms_val = safe_pos_float(ts_raw, default=0.0)
        ts_sec_val = (ts_ms_val / 1000.0 if ts_ms_val > 1e11 else ts_ms_val) if ts_ms_val > 0 else 0.0
        age = (now - ts_sec_val) if ts_sec_val > 0 else 1e9
        ok = -5.0 <= age < STALE_S and px_val > 0
        if not ok:
            problems.append(f"currentprice {s} age={age:.0f}s")
        print(f"   {s:10} age={age:6.0f}s price={px_val} -> {'OK' if ok else 'STALE'}")

    print()
    if problems:
        print(f"PROBLEME ({len(problems)}): {problems}")
        return 1
    print("OK — every cache is FRESH and coherent for all symbols")
    return 0


if __name__ == "__main__":
    sys.exit(main())
