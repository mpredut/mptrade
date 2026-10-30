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

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from binance_api import bapi as api  # noqa: E402
from binance_api import bapi_allorders as apiorders  # noqa: E402
import cacheManager as cm  # noqa: E402
from providers.base import extract_order_price, extract_order_qty, extract_timestamp_ms  # noqa: E402
from botcore import safe_float, safe_pos_float  # noqa: E402
from instrument_registry import casefold_get, load_registry, symbols_for  # noqa: E402
import pricewindow as pw  # noqa: E402

SYMBOLS = symbols_for("binance") or ["TAOUSDC", "BTCUSDC"]


import utils  # noqa: E402 — base_asset centralizat in utils (28 iul)
_base = utils.base_asset


def free_qty(symbol):
    base_up = str(_base(symbol) or "").strip().upper()
    for b in (api.get_account_assets_balances() or []):
        if isinstance(b, dict) and str(b.get("asset") or "").strip().upper() == base_up:
            return safe_float(b.get("free", 0), default=0.0)
    return 0.0


def position(symbol, maxage=17 * 24 * 3600):
    buys = [
        (o, q, p)
        for o in (apiorders.get_trade_orders("BUY", symbol, maxage) or [])
        if isinstance(o, dict)
        and (q := safe_pos_float(extract_order_qty(o), default=0.0) or 0.0) > 0.0
        and (p := safe_pos_float(extract_order_price(o), default=0.0) or 0.0) > 0.0
    ]
    sells_q = [
        q
        for o in (apiorders.get_trade_orders("SELL", symbol, maxage) or [])
        if isinstance(o, dict)
        and (q := safe_pos_float(extract_order_qty(o), default=0.0) or 0.0) > 0.0
    ]
    tq = sum(q for _, q, _ in buys)
    tv = sum(q * p for _, q, p in buys)
    sq = sum(sells_q)
    last_buy = (
        safe_float(extract_order_price(max((o for o, _, _ in buys), key=lambda o: safe_float(extract_timestamp_ms(o, default=0), 0.0))), 0.0)
        if buys else 0.0
    )   # ca get_relevant_trade (referinta botului)
    return (tv / tq if tq else 0.0), tq - sq, len(sells_q), last_buy


def main():
    mgr = cm.get_short_trend_manager()
    try:
        reg = load_registry()
    except Exception:
        reg = {}
    specs_by_sym = {spec.symbol: spec for spec in reg.values()}

    print(f"==== TRADE WATCH {time.strftime('%Y-%m-%d %H:%M:%S')} ====")
    for s in SYMBOLS:
        spec = casefold_get(specs_by_sym, s)
        try:
            gain_pct = safe_pos_float(spec.number("mt.gain") if spec else None, default=9.2)
        except Exception:
            gain_pct = 9.2
        try:
            hard_tp_pct = safe_pos_float(spec.number("mt.hardtp") if spec else None, default=17.0)
        except Exception:
            hard_tp_pct = 17.0

        snap = mgr.get_snapshot(s) or {}
        if not isinstance(snap, dict):
            snap = {}
        px = safe_float(snap.get("current_price"), default=0.0) or safe_float(api.get_current_price(s), default=0.0) or 0.0
        target = snap
        fast_w = pw.fast_trend_window_for_symbol(s)
        dyn = snap.get("dynamic")
        if fast_w is not None and isinstance(dyn, dict):
            w_snap = dyn.get(f"{float(fast_w):g}")
            if isinstance(w_snap, dict):
                target = w_snap
        raw_gc = target.get("growth_coefficient")
        if raw_gc is None or isinstance(raw_gc, bool):
            raw_gc = target.get("gradient_recent", 0.0)
        gr = safe_float(raw_gc, default=0.0)
        avg, net, n_sells, last_buy = position(s)
        free = free_qty(s)
        ref = last_buy or avg   # The bot uses the LAST buy as its reference (TP_REFERENCE="last").
        gain = (px - ref) / ref * 100 if ref else 0.0
        trend = "UP  " if gr > 0 else ("DOWN" if gr < 0 else "flat")
        hard_px = ref * (1 + hard_tp_pct / 100) if ref else 0.0
        print(f"  {s}: px={px:.2f}  ref=ULTIMUL_buy {ref:.2f} (avg {avg:.2f})  gain={gain:+.1f}%")
        print(f"      trend_instant={trend} ({gr:+.4f})  |  free={free:.4f}  net={net:.4f}  sells={n_sells}")
        flags = []
        if gain >= hard_tp_pct:
            flags.append(f"HARD-TP ARMED (it sells 50%={free*0.5:.3f})")
        elif gain >= gain_pct and gr < 0:
            flags.append(f"A NORMAL SALE IS ARMED (gain>{gain_pct:.1f}% plus a DOWN trend -> it sells EVERYTHING)")
        elif gain >= gain_pct and gr >= 0:
            flags.append(f"in profit but the trend is UP -> holding (HARD-TP at px {hard_px:.2f})")
        else:
            flags.append(f"below the thresholds (HARD-TP at px {hard_px:.2f})")
        print(f"      => {'; '.join(flags)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
