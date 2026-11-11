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
from providers.base import (
    extract_order_price,
    extract_timestamp_ms,
)
from providers.order_ohlc_utils import (
    extract_order_qty,
    extract_order_side,
    summarize_order_position,
)
from providers.market_api import api as mkt                     # noqa: E402  (singleton facade)
from providers.quantity import resolve_assets                   # noqa: E402
from instrument_registry import symbols_for                     # noqa: E402

MAXAGE = 17 * 24 * 3600   # like trade_watch.position (bot reference)
SYMBOLS = symbols_for("binance")


from botcore import safe_float


def free_direct(base):
    """Legacy full-account loop; unavailable account data is indistinguishable from zero."""
    base_up = str(base or "").strip().upper()
    for b in (api.get_account_assets_balances() or []):
        if isinstance(b, dict) and str(b.get("asset") or "").strip().upper() == base_up:
            return safe_float(b.get("free"), default=0.0)
    return 0.0


def position_from(buy_orders, sell_orders):
    """The get_position_stats logic: avg_buy + net_qty derived from order lists."""
    stats = summarize_order_position(buy_orders, sell_orders)
    return stats["avg_buy"], stats["net_qty"]


def order_key(o):
    if not isinstance(o, dict):
        return (0.0, 0.0, 0.0, "")
    return (
        float(extract_timestamp_ms(o, default=0) or 0),
        extract_order_price(o),
        extract_order_qty(o),
        str(extract_order_side(o) or "").upper(),
    )


def main():
    failures = []
    for s in SYMBOLS:
        base = resolve_assets(s)[0]
        print(f"\n==== {s} (base={base}) ====")

        # 1) free_balance ------------------------------------------------------
        fd = free_direct(base)
        ff = mkt.provider_by_name("binance").free_balance(base)
        fd_f = safe_float(fd, default=None)
        ff_f = safe_float(ff, default=None)
        balance_available = ff_f is not None and fd_f is not None
        same_free = balance_available and math.isclose(
            fd_f, ff_f, rel_tol=1e-9, abs_tol=1e-12
        )
        state = "OK" if same_free else ("SKIP (data unavailable)" if not balance_available else "DIFF")
        print(f"  free: direct={fd!r}  facade={ff!r}  -> {state}")
        if balance_available and not same_free:
            failures.append(f"{s}: free_balance direct={fd!r} != facade={ff!r}")
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
                print(f"  {side}: COUNT DIFF direct={len(raw)} facade={len(nrm)}")
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
        print(f"  avg_buy: direct={avg_d:.8f} facade={avg_f:.8f} | net: direct={net_d:.8f} "
              f"facade={net_f:.8f} -> {'OK' if same_pos else 'DIFF'}")
        if not same_pos:
            failures.append(f"{s}: direct position(avg={avg_d},net={net_d}) != facade(avg={avg_f},net={net_f})")

    print("\n" + "=" * 56)
    if failures:
        print(f"FAIL — {len(failures)} differences:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("PASS — there are zero confirmable differences between the facade and legacy access.")
    sys.exit(0)


if __name__ == "__main__":
    main()
