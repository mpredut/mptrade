#!/usr/bin/env python3
"""LIVE fleet-wide P&L report: realized P&L (average cost from real fills) plus unrealized
P&L (current position at current price). Sources: Binance get_filled_orders (direct API,
not the incomplete cache) and Kraken/HL state. Binance uses a 120-day window, so older
cost basis is approximate. Run with: ./myenv/bin/python verify_tools/pnl_report.py"""
import sys, os, time, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from botcore import safe_float, safe_pos_float
from binance_api import bapi_allorders as ao
from binance_api import bapi
from instrument_registry import symbols_for
from providers.base import (
    extract_order_price,
    extract_order_qty,
    extract_order_side,
    extract_timestamp_ms,
)

WINDOW_D = 120


def binance_symbol(symbol):
    st = int((time.time() - WINDOW_D * 24 * 3600) * 1000)
    fills = [f for f in (ao.get_filled_orders(None, symbol, st) or []) if isinstance(f, dict)]
    fills.sort(key=lambda t: extract_timestamp_ms(t, default=0) or 0)
    qty = cost = realized = 0.0
    buys = sells = 0
    for t in fills:
        p = safe_pos_float(extract_order_price(t), default=0.0)
        q = safe_pos_float(extract_order_qty(t), default=0.0)
        if p <= 0 or q <= 0:
            continue
        side = extract_order_side(t)
        if side == "BUY":
            qty += q; cost += q * p; buys += 1
        elif side == "SELL":
            sells += 1
            if qty > 1e-12:
                m = min(q, qty); avg = cost / qty
                realized += (p - avg) * m; cost -= avg * m; qty -= m
    try:
        cur = safe_pos_float(bapi.get_current_price(symbol), default=0.0)
    except Exception:
        cur = 0.0
    unreal = (qty * cur - cost) if (qty > 1e-9 and cur > 0) else 0.0
    return dict(realized=realized, unreal=unreal, pos=qty, cur=cur, n=len(fills), buys=buys, sells=sells)


# ── PER-BOT ATTRIBUTION (from get_all_orders, by clientOrderId prefix) ──────────
# Separate PER-BOT activity (RT_/TA_/MT_/AG_) from MANUAL activity (and_/web_/x-).
# realized_own covers only each bot's own round trip: min(buy_qty, sell_qty) times
# (avg_sell - avg_buy). For bots that exchange inventory (tradeall buys TAO while
# rtrade sells it), net_qty shows which one accumulates versus distributes. The
# window contains roughly the latest 1,000 orders returned by get_all_orders.
from binance_api.order_id_context import resolve_order_owner


def _bot(cid):
    owner = resolve_order_owner(cid)
    if owner == "manual":
        return "MANUAL(app/web)"
    return owner


def per_bot(symbol):
    from collections import defaultdict
    try:
        orders = bapi.client.get_all_orders(symbol=symbol, limit=1000) or []
    except Exception as e:  # noqa: BLE001
        print(f"  {symbol}: EROARE {e}"); return
    g = defaultdict(lambda: {"bq": 0.0, "bv": 0.0, "sq": 0.0, "sv": 0.0})
    for o in orders:
        if not isinstance(o, dict):
            continue
        eq = safe_pos_float(o.get("executedQty", 0), default=0.0)
        if eq <= 0:
            continue
        cqq = safe_pos_float(o.get("cummulativeQuoteQty", 0), default=0.0)
        if cqq <= 0:
            op = safe_pos_float(o.get("price", 0), default=0.0)
            cqq = eq * op if op > 0 else 0.0
        if cqq <= 0:
            continue
        a = g[_bot(o.get("clientOrderId", ""))]
        side = extract_order_side(o)
        if side == "BUY":
            a["bq"] += eq; a["bv"] += cqq
        elif side == "SELL":
            a["sq"] += eq; a["sv"] += cqq
    print(f"\n  {symbol} (the last {len(orders)} orders):")
    print(f"    {'source':<16}{'buy$':>9}{'sell$':>9}{'net_qty':>12}{'realized_own':>14}")
    for b, a in sorted(g.items(), key=lambda x: -(x[1]["bq"] + x[1]["sq"])):
        avg_b = a["bv"] / a["bq"] if a["bq"] else 0.0
        avg_s = a["sv"] / a["sq"] if a["sq"] else 0.0
        matched = min(a["bq"], a["sq"])
        realized_own = matched * (avg_s - avg_b) if (avg_b and avg_s) else 0.0
        print(f"    {b:<16}{a['bv']:>9.0f}{a['sv']:>9.0f}{a['bq'] - a['sq']:>+12.4f}{realized_own:>+14.2f}")


def main():
    print(f"\n=== FLEET P&L REPORT ({time.strftime('%Y-%m-%d %H:%M')}) ===")
    print(f"{'source':<22}{'realized':>11}{'unrealized':>12}{'position':>12}")
    tot_r = tot_u = 0.0

    for sym in symbols_for("binance"):
        try:
            d = binance_symbol(sym)
            tot_r += d["realized"]; tot_u += d["unreal"]
            print(f"{'Binance '+sym:<22}{d['realized']:>+11.2f}{d['unreal']:>+12.2f}"
                  f"{d['pos']:>10.4f}  ({d['n']} fills 120d, {d['buys']}b/{d['sells']}s @ {d['cur']})")
        except Exception as e:
            print(f"{'Binance '+sym:<22} ERROR: {e}")

    # Kraken HYPE (kraken_bot)
    from state_io import read_json_dict
    try:
        k = read_json_dict(os.path.join(ROOT, "kraken/.state_HYPEUSD.json"), default=None, recover_previous=True)
        if not isinstance(k, dict):
            raise FileNotFoundError("kraken/.state_HYPEUSD.json missing or invalid")
        hy = 54.77  # Latest price from trail_k.
        k_qty = safe_float(k.get("qty"), 0.0)
        k_cost = safe_float(k.get("cost"), 0.0)
        kr = safe_float(k.get("realized_net"), 0.0)
        ku = (k_qty * hy - k_cost) if k_qty > 1e-9 else 0.0
        tot_r += kr; tot_u += ku
        print(f"{'Kraken HYPE (bot)':<22}{kr:>+11.2f}{ku:>+12.2f}{k_qty:>10.4f}  (avg cost {k_cost/k_qty if k_qty else 0:.2f}, cur ~{hy})")
    except Exception as e:
        print(f"{'Kraken HYPE':<22} ERROR: {e}")

    # Hyperliquid dn_bot (delta-neutral: profit = funding - fees, price-independent)
    try:
        h = read_json_dict(os.path.join(ROOT, "hyperliquid/.state_dn_HYPE.json"), default=None, recover_previous=True)
        if not isinstance(h, dict):
            raise FileNotFoundError("hyperliquid/.state_dn_HYPE.json missing or invalid")
        h_fund = safe_float(h.get("funding_accrued"), 0.0)
        h_fees = safe_float(h.get("fees_paid"), 0.0)
        h_spot = safe_float(h.get("spot_qty"), 0.0)
        hl = h_fund - h_fees
        tot_r += hl
        print(f"{'HL dn_bot (funding)':<22}{hl:>+11.2f}{0.0:>+12.2f}{h_spot:>10.4f}  (delta-neutral: funding {h_fund:.2f} - fee {h_fees:.2f})")
    except Exception as e:
        print(f"{'HL dn_bot':<22} ERROR: {e}")

    print("-" * 57)
    print(f"{'TOTAL':<22}{tot_r:>+11.2f}{tot_u:>+12.2f}   -> NET {tot_r+tot_u:+.2f} USD")
    print("\nNote: the Binance realised figure covers a 120-day window (a cost basis older than that is approximated).")
    print("No Binance fees in realized (the API fills do not expose them here); ~0.075%/leg with BNB.")

    print("\n=== ATTRIBUTION PER BOT (clientOrderId; realized_own = only its own round trips) ===")
    for _sym in symbols_for("binance"):
        per_bot(_sym)
    print("Note: bots that trade inventory between themselves (tradeall buys / rtrade sells the same TAO)")
    print("have no cleanly separable realized — net_qty shows who accumulates (+) versus who distributes (-).")


if __name__ == "__main__":
    main()
