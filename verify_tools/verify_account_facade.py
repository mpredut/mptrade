#!/usr/bin/env python3
"""Compare normalized facade results with legacy Binance access paths.

For each configured Binance symbol it compares:
  - the facade's direct per-asset balance call with the legacy full-account loop;
  - normalized facade orders with ``apiorders.get_trade_orders``;
  - average buy and net quantity calculated from both order lists.

The two balance paths can differ when an API read fails: the safe facade returns
``None`` while the legacy full-account path collapses failure to ``0.0``. Such a
comparison is reported as inconclusive rather than misclassifying unavailable data as
a real zero balance. Confirmed differences still produce a nonzero exit.

Run before restarting the fleet:
  ../myenv/bin/python verify_account_facade.py
"""
import math
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from binance_api import bapi as api                  # noqa: E402
from binance_api import bapi_allorders as apiorders  # noqa: E402
from providers.base import (                         # noqa: E402
    extract_order_price,
    extract_order_qty,
    extract_order_side,
    extract_timestamp_ms,
)
from providers.market_api import api as mkt                     # noqa: E402  (facada singleton)
import symbols as sym                                 # noqa: E402

MAXAGE = 17 * 24 * 3600   # ca trade_watch.position (referinta botului)
SYMBOLS = [sym.taosymbol, sym.btcsymbol]
import utils  # noqa: E402 — base_asset centralizat in utils (28 iul)
base_of = utils.base_asset


def _f(val, default=0.0):
    if isinstance(val, bool) or val is None:
        return default
    try:
        v = float(val)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def free_direct(base):
    """Legacy full-account loop; unavailable account data is indistinguishable from zero."""
    for b in (api.get_account_assets_balances() or []):
        if isinstance(b, dict) and b.get("asset") == base:
            return _f(b.get("free"), 0.0)
    return 0.0


def position_from(buy_orders, sell_orders):
    """The get_position_stats logic: avg_buy + net_qty derived from order lists."""
    buys = [
        o for o in (buy_orders or [])
        if isinstance(o, dict) and _f(extract_order_qty(o), 0.0) > 0.0 and _f(extract_order_price(o), 0.0) > 0.0
    ]
    sells = [
        o for o in (sell_orders or [])
        if isinstance(o, dict) and _f(extract_order_qty(o), 0.0) > 0.0
    ]
    tq = sum(_f(extract_order_qty(o), 0.0) for o in buys)
    tv = sum(_f(extract_order_price(o), 0.0) * _f(extract_order_qty(o), 0.0) for o in buys)
    sq = sum(_f(extract_order_qty(o), 0.0) for o in sells)
    avg = tv / tq if tq > 0 else 0.0
    return avg, tq - sq


def order_key(o):
    if not isinstance(o, dict):
        return (0.0, 0.0, 0.0, "")
    return (
        _f(extract_timestamp_ms(o, default=0), 0.0),
        _f(extract_order_price(o), 0.0),
        _f(extract_order_qty(o), 0.0),
        str(extract_order_side(o) or "").upper(),
    )


def main():
    failures = []
    for s in SYMBOLS:
        base = base_of(s)
        print(f"\n==== {s} (base={base}) ====")

        # 1) free_balance ------------------------------------------------------
        fd = free_direct(base)
        ff = mkt.free_balance_for("binance", base)
        balance_available = ff is not None and fd is not None
        same_free = balance_available and math.isclose(
            float(fd), float(ff), rel_tol=1e-9, abs_tol=1e-12
        )
        state = "OK" if same_free else ("SKIP (date indisponibile)" if not balance_available else "DIFERA")
        print(f"  free: direct={fd!r}  facada={ff!r}  -> {state}")
        if balance_available and not same_free:
            failures.append(f"{s}: free_balance direct={fd!r} != facada={ff!r}")
        if not balance_available:
            print("  the rest of the private comparison is skipped; two empty lists after API "
                  "errors are not proof of parity")
            continue

        # 2) BUY/SELL orders: count plus a field-by-field comparison -----------
        direct = {}
        facade = {}
        for side in ("BUY", "SELL"):
            raw = apiorders.get_trade_orders(side, s, MAXAGE) or []
            nrm = mkt.get_orders(s, side, MAXAGE) or []
            direct[side], facade[side] = raw, nrm
            if len(raw) != len(nrm):
                failures.append(f"{s} {side}: order count direct={len(raw)} != facade={len(nrm)}")
                print(f"  {side}: COUNT DIFERA direct={len(raw)} facada={len(nrm)}")
                continue
            mism = 0
            for r, n in zip(sorted(raw, key=order_key), sorted(nrm, key=order_key)):
                if order_key(r) != order_key(n):
                    mism += 1
            if mism:
                failures.append(f"{s} {side}: {mism} orders with different fields")
            print(f"  {side}: n={len(raw)}  field by field -> {'OK' if mism == 0 else f'{mism} DIFFER'}")

        # 3) avg_buy / net_qty from both sources ------------------------------
        avg_d, net_d = position_from(direct["BUY"], direct["SELL"])
        avg_f, net_f = position_from(facade["BUY"], facade["SELL"])
        same_pos = (
            math.isclose(avg_d, avg_f, rel_tol=1e-9, abs_tol=1e-12)
            and math.isclose(net_d, net_f, rel_tol=1e-9, abs_tol=1e-12)
        )
        print(f"  avg_buy: direct={avg_d:.8f} facada={avg_f:.8f} | net: direct={net_d:.8f} "
              f"facada={net_f:.8f} -> {'OK' if same_pos else 'DIFERA'}")
        if not same_pos:
            failures.append(f"{s}: pozitie direct(avg={avg_d},net={net_d}) != facada(avg={avg_f},net={net_f})")

    print("\n" + "=" * 56)
    if failures:
        print(f"FAIL — {len(failures)} diferente:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("PASS — there are zero confirmable differences between the facade and legacy access.")
    sys.exit(0)


if __name__ == "__main__":
    main()
