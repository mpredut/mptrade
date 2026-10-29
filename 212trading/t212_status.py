#!/usr/bin/env python3
"""
t212_status.py — Trading 212 account report for free/invested/blocked cash,
positions, and PENDING orders. Explains invested, available, and blocked amounts.

  python3 t212_status.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.append(_HERE)
from ipo_common import (  # noqa: E402
    http_get, load_t212_environment, required_env, required_float_env, safe_float,
)
from t212_client import T212Client  # noqa: E402
from credentials import t212_credentials  # noqa: E402
from providers.base import extract_order_side  # noqa: E402


def _retry(fn, tries=4, pause=2):
    for i in range(tries):
        try:
            r = fn()
        except Exception:  # noqa: BLE001
            r = None
        if r is not None:
            return r
        if i < tries - 1:
            time.sleep(pause)
    return None


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    load_t212_environment(os.path.join(here, ".env"))
    credentials = t212_credentials()
    c = T212Client(
        credentials.key, credentials.secret, env=required_env("T212_ENV"),
        min_gap_sec=required_float_env("T212_MIN_GAP_SEC"),
        portfolio_ttl_sec=required_float_env("T212_PORTFOLIO_TTL_SEC"),
    )

    # --- CASH ---
    def get_cash():
        st, body = http_get(f"{c.base}/equity/account/cash", headers=c._headers())
        if st != 200 or not body:
            return None
        try:
            return json.loads(body)
        except ValueError:
            return None

    def _sf(obj: dict, key: str, default: float = 0.0) -> float:
        return safe_float(obj.get(key), default=default)

    cash = _retry(get_cash)
    print("============ TRADING 212 ACCOUNT ============")
    if isinstance(cash, dict) and cash:
        free = _sf(cash, "free"); inv = _sf(cash, "invested")
        blk = _sf(cash, "blocked"); tot = _sf(cash, "total")
        ppl = _sf(cash, "ppl"); pie = _sf(cash, "pieCash")
        print(f"  INVESTED (cost)   : {inv:>10.2f}")
        print(f"  AVAILABLE (free)  : {free:>10.2f}   <- this is what you can place new orders with")
        print(f"  BLOCKED           : {blk:>10.2f}   <- locked in pending or in-flight orders")
        print(f"  pie cash          : {pie:>10.2f}")
        print(f"  unrealized P&L    : {ppl:>+10.2f}")
        print(f"  TOTAL account     : {tot:>10.2f}")
    else:
        print("  ! cannot read the cash (rate limit or auth?)")

    # --- POSITIONS ---
    pf = [p for p in (_retry(c.get_portfolio) or []) if isinstance(p, dict)]
    print(f"\n  --- POSITIONS ({len(pf)}) ---")
    inv_sum = 0.0
    for p in pf:
        q = _sf(p, "quantity"); avg = _sf(p, "averagePrice")
        cur = _sf(p, "currentPrice"); ppl = _sf(p, "ppl")
        val = q * cur; inv_sum += q * avg
        print(f"    {str(p.get('ticker') or ''):<14} qty {q:<8.3f} avg {avg:<9.2f} price {cur:<9.2f} "
              f"value {val:>9.2f}  P&L {ppl:>+8.2f}")
    if pf:
        print(f"    (positions cost sum: {inv_sum:.2f})")

    # --- PENDING ORDERS that block cash ---
    orders = [o for o in (_retry(c.list_active_orders) or []) if isinstance(o, dict)]
    print(f"\n  --- PENDING ORDERS ({len(orders)}) ---  <- THESE lock up free cash")
    if not orders:
        print("    (none — the cash is not locked by orders)")
    for o in orders:
        q = _sf(o, "quantity")
        if q == 0.0:
            q = _sf(o, "orderedQuantity", _sf(o, "qty"))
        lp = o.get("limitPrice", o.get("price"))
        side = extract_order_side(o) or "UNKNOWN"
        inst = o.get("instrument") if isinstance(o.get("instrument"), dict) else {}
        ticker = o.get("ticker") or inst.get("ticker") or ""
        print(f"    {str(ticker):<14} {side} qty {q} @ {lp}  ({o.get('type','')}, {o.get('status','')}) id={o.get('id','')}")
    print("==========================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
