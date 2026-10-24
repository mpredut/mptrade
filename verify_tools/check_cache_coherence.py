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
from state_io import previous_generation_path, read_recovering_previous  # noqa: E402

CACHEDB = os.path.join(ROOT, "cachedb")
STALE_S = 600.0  # More than 10 minutes without an update is STALE (instant trend refreshes every few seconds / 20s).


def load(name):
    p = os.path.join(CACHEDB, name)
    if not os.path.exists(p) and not os.path.exists(previous_generation_path(p)):
        return None
    try:
        def _read_json(candidate):
            with open(candidate, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("cache payload root must be a dict")
            return data

        return read_recovering_previous(p, _read_json)
    except Exception as e:  # noqa: BLE001
        print(f"  [ERR] {name}: {e}")
        return None


def expected_symbols():
    """Use the registry without credentials; invalid configuration must fail the check."""
    sys.path.insert(0, ROOT)
    from instrument_registry import select_instruments, symbols_for
    syms = symbols_for("binance")
    for inst in select_instruments(role="mt").values():
        if inst.provider != "binance" and inst.symbol not in syms:
            syms.append(inst.symbol)
    if not syms:
        raise ValueError("No configured symbols to verify")
    return syms


now = time.time()
problems = []
print(f"=== cache coherence @ {time.strftime('%Y-%m-%d %H:%M:%S')} (stale>{int(STALE_S)}s) ===")
exp = expected_symbols()
print(f"simboluri asteptate: {exp}")

# 1. short trend
it = load("cache_short_trend.json") or {}
print(f"-- cache_short_trend.json: {sorted(it.keys())}")
for s in exp:
    e = it.get(s)
    if not isinstance(e, dict) or not e:
        problems.append(f"short_trend MISSING {s}")
        print(f"   {s:10} IS MISSING")
        continue
    ts = e.get("ts", 0)
    try:
        ts_val = float(ts) if ts is not None and not isinstance(ts, bool) else 0.0
    except (TypeError, ValueError):
        ts_val = 0.0
    age = now - ts_val if ts_val > 0 else 1e9
    px = e.get("current_price")
    try:
        px_ok = px is not None and not isinstance(px, bool) and float(px) > 0
    except (TypeError, ValueError):
        px_ok = False
    ok = -5.0 <= age < STALE_S and px_ok
    if not ok:
        problems.append(f"short_trend {s} age={age:.0f}s px={px}")
    print(f"   {s:10} age={age:6.0f}s gradient_recent={e.get('gradient_recent')} price={px} -> {'OK' if ok else 'STALE/INVALID'}")

# 2. current price
cp = load("cache_current_price.json") or {}
# format: {'fetchtime': {...}, 'items': {symbol: [[ts_ms, price]]}} (or directly per symbol)
raw_items = cp.get("items", cp) if isinstance(cp, dict) else {}
items = raw_items if isinstance(raw_items, dict) else {}
print(f"-- cache_current_price.json: {sorted(k for k in items.keys() if k != 'fetchtime')}")
for s in exp:
    entries = items.get(s)
    if not entries or not isinstance(entries, list):
        problems.append(f"currentprice MISSING {s}")
        print(f"   {s:10} IS MISSING")
        continue
    last_entry = entries[-1]
    if not isinstance(last_entry, (list, tuple)) or len(last_entry) < 2:
        problems.append(f"currentprice INVALID {s}")
        print(f"   {s:10} IS INVALID")
        continue
    ts_ms, price = last_entry[0], last_entry[1]
    try:
        ts_ms_val = float(ts_ms) if ts_ms is not None and not isinstance(ts_ms, bool) else 0.0
        px_val = float(price) if price is not None and not isinstance(price, bool) else 0.0
    except (TypeError, ValueError):
        ts_ms_val, px_val = 0.0, 0.0
    age = (now - ts_ms_val / 1000.0) if ts_ms_val > 0 else 1e9
    ok = -5.0 <= age < STALE_S and px_val > 0
    if not ok:
        problems.append(f"currentprice {s} age={age:.0f}s")
    print(f"   {s:10} age={age:6.0f}s price={price} -> {'OK' if ok else 'STALE'}")

print()
if problems:
    print(f"PROBLEME ({len(problems)}): {problems}")
    sys.exit(1)
print("OK — every cache is FRESH and coherent for all symbols")
sys.exit(0)
