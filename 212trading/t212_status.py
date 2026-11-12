#!/usr/bin/env python3
"""
t212_status.py -- Trading 212 account report for free/invested/blocked cash,
positions, and PENDING orders. Explains invested, available, and blocked amounts.

  python3 t212_status.py
"""
from __future__ import annotations

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from botcore import (  # noqa: E402
    ensure_local_module_priority,
    required_env,
    required_float_env,
    safe_float,
)
ensure_local_module_priority(_HERE, "ipo_common", "t212_client")
from ipo_common import load_t212_environment  # noqa: E402
from t212_client import T212Client  # noqa: E402
from credentials import t212_credentials  # noqa: E402
from providers.base import (
    extract_order_id,
    extract_order_price,
)
from providers.order_ohlc_utils import (
    extract_broker_order_ticker as _order_ticker,
    extract_order_qty,
    extract_order_side,
)


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
    cash = _retry(c.get_cash)
    print("============ TRADING 212 ACCOUNT ============")
    if isinstance(cash, dict) and cash:
        free = safe_float(cash.get("free"), default=0.0)
        inv = safe_float(cash.get("invested"), default=0.0)
        blk = safe_float(cash.get("blocked"), default=0.0)
        tot = safe_float(cash.get("total"), default=0.0)
        ppl = safe_float(cash.get("ppl"), default=0.0)
        pie = safe_float(cash.get("pieCash"), default=0.0)
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
        q = safe_float(p.get("quantity"), default=0.0)
        avg = safe_float(p.get("averagePrice"), default=0.0)
        cur = safe_float(p.get("currentPrice"), default=0.0)
        ppl = safe_float(p.get("ppl"), default=0.0)
        val = q * cur; inv_sum += q * avg
        print(f"    {str(p.get('ticker') or ''):<14} qty {q:<8.3f} avg {avg:<9.2f} price {cur:<9.2f} "
              f"value {val:>9.2f}  P&L {ppl:>+8.2f}")
    if pf:
        print(f"    (positions cost sum: {inv_sum:.2f})")

    # --- PENDING ORDERS that block cash ---
    orders = [o for o in (_retry(c.list_active_orders) or []) if isinstance(o, dict)]
    print(f"\n  --- PENDING ORDERS ({len(orders)}) ---  <- THESE lock up free cash")
    if not orders:
        print("    (none -- the cash is not locked by orders)")
    for o in orders:
        q = extract_order_qty(o, default=0.0)
        lp = extract_order_price(o, default=None)
        side = extract_order_side(o) or "UNKNOWN"
        ticker = _order_ticker(o)
        oid = extract_order_id(o) or ""
        print(f"    {str(ticker):<14} {side} qty {q} @ {lp}  ({o.get('type','')}, {o.get('status','')}) id={oid}")
    print("==========================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
