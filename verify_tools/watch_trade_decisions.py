#!/usr/bin/env python3
"""
trade_watch.py — quick view of the trade decision (TAO/BTC) for LIVE MONITORING.

Reads UN-buffered signals (the source of truth, not the buffered log):
  - short trend from the cache manager (gradient_recent, in sync with cache_short_trend.json)
  - the real free balance (api.get_account_assets_balances)
  - avg buy + net position from the Binance trades
It also shows the thresholds: HARD-TP (+18%) and the normal sell (gain>9.2% AND trend DOWN).

  ../myenv/bin/python trade_watch.py
"""
from __future__ import annotations

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from binance_api import bapi as api  # noqa: E402
from binance_api import bapi_allorders as apiorders  # noqa: E402
import cacheManager as cm  # noqa: E402
from providers.base import extract_order_price, extract_order_qty, extract_timestamp_ms  # noqa: E402
import symbols as sym  # noqa: E402

SYMBOLS = [sym.taosymbol, sym.btcsymbol]
HARD_TP_PCT = 17.0   # Keep in sync with monitortrades.conf (hard_tp_pct).
GAIN_PCT = 9.2


import utils  # noqa: E402 — base_asset centralizat in utils (28 iul)
_base = utils.base_asset


def _f(val, default=0.0):
    if val is None or isinstance(val, bool):
        return default
    try:
        res = float(val)
    except (TypeError, ValueError):
        return default
    return res if math.isfinite(res) else default


def free_qty(symbol):
    base = _base(symbol)
    for b in (api.get_account_assets_balances() or []):
        if isinstance(b, dict) and b.get("asset") == base:
            return _f(b.get("free", 0), 0.0)
    return 0.0


def position(symbol, maxage=17 * 24 * 3600):
    buys = [
        o for o in (apiorders.get_trade_orders("BUY", symbol, maxage) or [])
        if isinstance(o, dict) and extract_order_qty(o) > 0 and extract_order_price(o) > 0
    ]
    sells = [
        o for o in (apiorders.get_trade_orders("SELL", symbol, maxage) or [])
        if isinstance(o, dict) and extract_order_qty(o) > 0
    ]
    tq = sum(extract_order_qty(o) for o in buys)
    tv = sum(extract_order_price(o) * extract_order_qty(o) for o in buys)
    sq = sum(extract_order_qty(o) for o in sells)
    last_buy = (
        extract_order_price(max(buys, key=lambda o: extract_timestamp_ms(o)))
        if buys else 0.0
    )   # ca get_relevant_trade (referinta botului)
    return (tv / tq if tq else 0.0), tq - sq, len(sells), last_buy


def main():
    mgr = cm.get_short_trend_manager()
    print(f"==== TRADE WATCH {time.strftime('%Y-%m-%d %H:%M:%S')} ====")
    for s in SYMBOLS:
        snap = mgr.get_snapshot(s) or {}
        if not isinstance(snap, dict):
            snap = {}
        px = _f(snap.get("current_price"), 0.0) or _f(api.get_current_price(s), 0.0) or 0.0
        gr = _f(snap.get("gradient_recent", 0.0), 0.0)
        avg, net, n_sells, last_buy = position(s)
        free = free_qty(s)
        ref = last_buy or avg   # The bot uses the LAST buy as its reference (TP_REFERENCE="last").
        gain = (px - ref) / ref * 100 if ref else 0.0
        trend = "UP  " if gr > 0 else ("DOWN" if gr < 0 else "flat")
        hard_px = ref * (1 + HARD_TP_PCT / 100) if ref else 0.0
        print(f"  {s}: px={px:.2f}  ref=ULTIMUL_buy {ref:.2f} (avg {avg:.2f})  gain={gain:+.1f}%")
        print(f"      trend_instant={trend} ({gr:+.4f})  |  free={free:.4f}  net={net:.4f}  sells={n_sells}")
        flags = []
        if gain >= HARD_TP_PCT:
            flags.append(f"HARD-TP ARMED (it sells 50%={free*0.5:.3f})")
        elif gain >= GAIN_PCT and gr < 0:
            flags.append("A NORMAL SALE IS ARMED (gain>9.2% plus a DOWN trend -> it sells EVERYTHING)")
        elif gain >= GAIN_PCT and gr >= 0:
            flags.append(f"in profit but the trend is UP -> holding (HARD-TP at px {hard_px:.2f})")
        else:
            flags.append(f"below the thresholds (HARD-TP at px {hard_px:.2f})")
        print(f"      => {'; '.join(flags)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
