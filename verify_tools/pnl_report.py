#!/usr/bin/env python3
"""LIVE fleet-wide P&L report: realized P&L (average cost from real fills) plus unrealized
P&L (current position at current price). Sources: Binance get_filled_orders (direct API,
not the incomplete cache) and Kraken/HL state. Binance uses a 120-day window, so older
cost basis is approximate. Run with: ./myenv/bin/python verify_tools/pnl_report.py"""
import sys, os, time, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from binance_api import bapi_allorders as ao
from binance_api import bapi
from instrument_registry import symbols_for

WINDOW_D = 120


def binance_symbol(symbol):
    st = int((time.time() - WINDOW_D * 24 * 3600) * 1000)
    fills = ao.get_filled_orders(None, symbol, st) or []
    fills.sort(key=lambda t: t.get("timestamp", 0))
    qty = cost = realized = 0.0
    buys = sells = 0
    for t in fills:
        p = float(t["price"]); q = float(t.get("quantity", 0)); side = (t.get("side") or "").upper()
        if side == "BUY":
            qty += q; cost += q * p; buys += 1
        elif side == "SELL":
            sells += 1
            if qty > 1e-12:
                m = min(q, qty); avg = cost / qty
                realized += (p - avg) * m; cost -= avg * m; qty -= m
    try:
        cur = float(bapi.get_current_price(symbol))
    except Exception:
        cur = 0.0
    unreal = qty * cur - cost if qty > 1e-9 else 0.0
    return dict(realized=realized, unreal=unreal, pos=qty, cur=cur, n=len(fills), buys=buys, sells=sells)


print(f"\n=== RAPORT P&L FLOTA ({time.strftime('%Y-%m-%d %H:%M')}) ===")
print(f"{'sursa':<22}{'realized':>11}{'unrealized':>12}{'pozitie':>12}")
tot_r = tot_u = 0.0

for sym in symbols_for("binance"):
    try:
        d = binance_symbol(sym)
        tot_r += d["realized"]; tot_u += d["unreal"]
        print(f"{'Binance '+sym:<22}{d['realized']:>+11.2f}{d['unreal']:>+12.2f}"
              f"{d['pos']:>10.4f}  ({d['n']} fills 120z, {d['buys']}b/{d['sells']}s @ {d['cur']})")
    except Exception as e:
        print(f"{'Binance '+sym:<22} EROARE: {e}")

# Kraken HYPE (kraken_bot)
try:
    k = json.load(open("kraken/.state_HYPEUSD.json"))
    hy = 54.77  # Latest price from trail_k.
    kr, ku = k["realized_net"], (k["qty"] * hy - k["cost"] if k["qty"] > 1e-9 else 0.0)
    tot_r += kr; tot_u += ku
    print(f"{'Kraken HYPE (bot)':<22}{kr:>+11.2f}{ku:>+12.2f}{k['qty']:>10.4f}  (avg cost {k['cost']/k['qty'] if k['qty'] else 0:.2f}, cur ~{hy})")
except Exception as e:
    print(f"{'Kraken HYPE':<22} EROARE: {e}")

# Hyperliquid dn_bot (delta-neutral: profit = funding - fees, price-independent)
try:
    h = json.load(open("hyperliquid/.state_dn_HYPE.json"))
    hl = h.get("funding_accrued", 0) - h.get("fees_paid", 0)
    tot_r += hl
    print(f"{'HL dn_bot (funding)':<22}{hl:>+11.2f}{0.0:>+12.2f}{h.get('spot_qty',0):>10.4f}  (delta-neutral: funding {h.get('funding_accrued',0):.2f} - fee {h.get('fees_paid',0):.2f})")
except Exception as e:
    print(f"{'HL dn_bot':<22} EROARE: {e}")

print("-" * 57)
print(f"{'TOTAL':<22}{tot_r:>+11.2f}{tot_u:>+12.2f}   -> NET {tot_r+tot_u:+.2f} USD")
print("\nNote: the Binance realised figure covers a 120-day window (a cost basis older than that is approximated).")
print("No Binance fees in realized (the API fills do not expose them here); ~0.075%/leg with BNB.")


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
        orders = bapi.client.get_all_orders(symbol=symbol, limit=1000)
    except Exception as e:  # noqa: BLE001
        print(f"  {symbol}: EROARE {e}"); return
    g = defaultdict(lambda: {"bq": 0.0, "bv": 0.0, "sq": 0.0, "sv": 0.0})
    for o in orders:
        eq = float(o.get("executedQty", 0) or 0)
        if eq <= 0:
            continue
        cqq = float(o.get("cummulativeQuoteQty", 0) or 0)
        a = g[_bot(o.get("clientOrderId", ""))]
        if o["side"] == "BUY":
            a["bq"] += eq; a["bv"] += cqq
        else:
            a["sq"] += eq; a["sv"] += cqq
    print(f"\n  {symbol} (the last {len(orders)} orders):")
    print(f"    {'sursa':<16}{'buy$':>9}{'sell$':>9}{'net_qty':>12}{'realized_own':>14}")
    for b, a in sorted(g.items(), key=lambda x: -(x[1]["bq"] + x[1]["sq"])):
        avg_b = a["bv"] / a["bq"] if a["bq"] else 0.0
        avg_s = a["sv"] / a["sq"] if a["sq"] else 0.0
        matched = min(a["bq"], a["sq"])
        realized_own = matched * (avg_s - avg_b) if (avg_b and avg_s) else 0.0
        print(f"    {b:<16}{a['bv']:>9.0f}{a['sv']:>9.0f}{a['bq'] - a['sq']:>+12.4f}{realized_own:>+14.2f}")


print("\n=== ATTRIBUTION PER BOT (clientOrderId; realized_own = only its own round trips) ===")
for _sym in symbols_for("binance"):
    per_bot(_sym)
print("Note: bots that trade inventory between themselves (tradeall buys / rtrade sells the same TAO)")
print("have no cleanly separable realized — net_qty shows who accumulates (+) versus who distributes (-).")
