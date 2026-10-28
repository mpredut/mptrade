#!/usr/bin/env python3
"""
Cleans cache_order.json of the orders that are NOT completed trades.

Why: the WS path (_upsert_order_from_execution_report) stored ANY executionReport,
NEW/CANCELED/EXPIRED/REJECTED included, and the cancelled ones came with price=0 (the "L or p" bug:
L = "0.00000000" is a truthy string). Those polluted the profit guard (min(price) -> 0).
The code was fixed (the Order cache = FILLED/PARTIALLY_FILLED only), but the old file
mai contine gunoiul. Scriptul il scoate.

The REMOVAL criterion:  status in {CANCELED, EXPIRED, REJECTED, NEW}  OR  price <= 0.
It keeps: FILLED/PARTIALLY_FILLED and the REST entries (status None) with price > 0.

RUN IT ONLY with cacheManager STOPPED (otherwise it rewrites the file from the old memory).
  python3 offline/legacy_tools/clean_order_cache.py --dry
  python3 offline/legacy_tools/clean_order_cache.py
Idempotent.
"""
import os
import sys
import json
import math
import time

# offline/legacy_tools/ sits two levels below the repository root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from cachepaths import get_cachedb_dir

CACHE_DIR = get_cachedb_dir()
ORDER_FILE = os.path.join(CACHE_DIR, "cache_order.json")

BAD_STATUS = {"CANCELED", "EXPIRED", "REJECTED", "NEW"}


def is_garbage(order):
    if not isinstance(order, dict):
        return True
    status = str(order.get("status") or "").upper()
    raw_price = order.get("price")
    if isinstance(raw_price, bool) or raw_price is None:
        price = 0.0
    else:
        try:
            price = float(raw_price)
        except (TypeError, ValueError):
            price = 0.0
    return status in BAD_STATUS or not (math.isfinite(price) and price > 0)


def main():
    dry = "--dry" in sys.argv
    if not os.path.exists(ORDER_FILE):
        print(f"{ORDER_FILE} does not exist — nothing to do.")
        return

    with open(ORDER_FILE) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        print(f"{ORDER_FILE} does not contain a JSON object — nothing to do.")
        return
    items = data.get("items", {})
    if not isinstance(items, dict):
        print(f"{ORDER_FILE} items is not a dict — nothing to do.")
        return

    total = removed = 0
    cleaned = {}
    for sym, lst in items.items():
        if not isinstance(lst, list):
            continue
        keep = [o for o in lst if not is_garbage(o)]
        total += len(lst)
        removed += len(lst) - len(keep)
        cleaned[sym] = keep
        print(f"  {sym}: total={len(lst)} | scoase={len(lst) - len(keep)} | raman={len(keep)}")
    print(f"TOTAL: {total} | de scos={removed} | raman={total - removed}")

    if dry:
        print("[DRY] nothing was written.")
        return
    if removed == 0:
        print("Nothing to clean.")
        return

    bak = ORDER_FILE + f".bak.{int(time.time())}"
    os.rename(ORDER_FILE, bak)
    print(f"Backup: {bak}")
    data["items"] = cleaned
    tmp = ORDER_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, ORDER_FILE)
    print(f"Wrote a cleaned {ORDER_FILE} ({removed} removed).")


if __name__ == "__main__":
    main()
