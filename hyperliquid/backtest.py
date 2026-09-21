#!/usr/bin/env python3
"""
backtest.py — COMPLETE Hyperliquid DCA and take-profit strategy backtester.

Replay the strategy bar by bar over one-hour OHLC candles:
  * LIMIT entry at market-/+discount, filling when the bar touches price;
  * DCA after price moves against the position by drop%;
  * take profit at average_price*(1±tp), then restart the cycle;
  * include per-fill FEES AND FUNDING from real HL history;
  * optional signal gate allowing longs only on up and shorts only on down.

Modes:
  single — one run with a detailed report
  sweep  — test many TP/DROP combinations and show the best for tuning

Run with the virtual environment:
  python backtest.py --coin HYPE --days 45 --direction short
  python backtest.py --mode sweep --coin HYPE --days 45 --direction short --signal analysis
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import urllib.request

from common import log
from hl_client import HLClient


def _post(body):
    r = urllib.request.Request("https://api.hyperliquid.xyz/info",
                               data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(r, timeout=25) as x:
        return json.loads(x.read())


def fetch_funding(coin, start_ms):
    """Return hourly coin funding history sorted for time alignment."""
    try:
        data = _post({"type": "fundingHistory", "coin": coin, "startTime": start_ms})
        return sorted(((int(d["time"]), float(d["fundingRate"])) for d in data), key=lambda x: x[0])
    except Exception as e:  # noqa: BLE001
        log(f"  ! funding history indisponibil: {e}")
        return []


def funding_at(fund, t_ms):
    """Return the funding rate active at t_ms, using the most recent rate at or before t."""
    lo, hi = 0, len(fund)
    while lo < hi:
        mid = (lo + hi) // 2
        if fund[mid][0] <= t_ms:
            lo = mid + 1
        else:
            hi = mid
    return fund[lo-1][1] if lo > 0 else (fund[0][1] if fund else 0.0)


def precompute_signal(closes, ts, source, fast, slow, band, win_h, step_h):
    """Return a causal signal per bar using data available through that bar only."""
    out = ["neutral"] * len(closes)
    if source == "off":
        return ["__any__"] * len(closes)         # gate disabled
    for i in range(len(closes)):
        if source == "analysis":
            from price_analysis import detect_long_term_trend
            res = detect_long_term_trend(ts[:i+1], closes[:i+1], win_h, step_h)
            out[i] = res["direction"] if res else "neutral"
        else:  # builtin
            if i + 1 >= slow:
                mf = statistics.mean(closes[i+1-fast:i+1]); ms = statistics.mean(closes[i+1-slow:i+1])
                diff = (mf - ms) / ms if ms else 0
                out[i] = "up" if diff > band else "down" if diff < -band else "neutral"
    return out


def simulate(ohlc, ts, fund, sig, P, direction):
    """Replay the strategy over OHLC bars and millisecond timestamps; return metrics."""
    sign = 1 if direction == "long" else -1
    want = "up" if sign > 0 else "down"
    disc, drop, tp = P["disc"]/100, P["drop"]/100, P["tp"]/100
    qty = cost = spent = 0.0
    dca = 0
    last_open = None
    realized = fees = funding = 0.0
    eq = 0.0; peak = 0.0; maxdd = 0.0
    cycles = wins = 0
    rest_buy = None    # (price, size, kind) -> open
    rest_sell = None   # (px, sz)

    for i in range(len(ohlc)):
        o, h, l, c = ohlc[i]
        # Funding on held perpetuals: shorts receive positive funding; longs pay it.
        if qty > 1e-12:
            f = funding_at(fund, ts[i]) * qty * c
            funding += (f if sign < 0 else -f)
        # --- fills: check active orders against the bar range ---
        if rest_buy:
            px, sz, _ = rest_buy
            hit = (l <= px) if sign > 0 else (h >= px)
            if hit:
                qty += sz; cost += sz*px; spent += sz*px
                last_open = px
                if dca > 0 or qty > sz + 1e-9: dca += 1     # exclude the first entry
                fees += P["fee"]/100 * sz*px
                rest_buy = None; rest_sell = None            # average changed; replace TP
        if rest_sell and qty > 1e-12:
            px, sz = rest_sell
            hit = (h >= px) if sign > 0 else (l <= px)
            if hit:
                avg = cost/qty
                gross = sign*(px-avg)*sz
                fee = P["fee"]/100 * sz*px
                realized += gross - fee; fees += fee
                cycles += 1; wins += 1 if gross > 0 else 0
                qty = cost = spent = 0.0; dca = 0; last_open = None
                rest_sell = None; rest_buy = None
        # --- decisions at close ---
        if qty <= 1e-12:
            if rest_buy is None and (sig[i] == want or sig[i] == "__any__"):
                if spent + P["entry"] <= P["budget"]:
                    px = c*(1 - sign*disc); sz = round(P["entry"]/px, 4)
                    rest_buy = (px, sz, "ENTRY")
        else:
            avg = cost/qty
            target = avg*(1 + sign*tp)
            rest_sell = (target, qty)
            moved = sign*(c - last_open)/last_open if last_open else 0
            if (dca < P["maxdca"] and moved <= -drop and rest_buy is None
                    and spent + P["dca"] <= P["budget"]):
                px = c*(1 - sign*disc); sz = round(P["dca"]/px, 4)
                rest_buy = (px, sz, "DCA")
        # Equity curve includes realized and unrealized results.
        upnl = sign*(c - cost/qty)*qty if qty > 1e-12 else 0
        e = realized + funding + upnl
        eq = e; peak = max(peak, eq); maxdd = max(maxdd, peak - eq)

    final_upnl = (sign*(c - cost/qty)*qty) if qty > 1e-12 else 0.0
    return {"net": realized + funding, "realized": realized, "fees": fees, "funding": funding,
            "cycles": cycles, "wins": wins, "maxdd": maxdd, "open_qty": qty,
            "final_upnl": final_upnl, "total": realized + funding + final_upnl}


def run(coin, ohlc, ts, fund, sig, P, direction, budget):
    m = simulate(ohlc, ts, fund, sig, P, direction)
    total_pct = m["total"]/budget*100          # TRUE result: realized + funding + open position
    wr = 100*m["wins"]/m["cycles"] if m["cycles"] else 0
    return total_pct, m, wr


def main() -> int:
    ap = argparse.ArgumentParser(description="Backtester complet DCA+TP pe Hyperliquid.")
    ap.add_argument("--mode", choices=["single", "sweep"], default="single")
    ap.add_argument("--coin", default="HYPE")
    ap.add_argument("--days", type=int, default=45)
    ap.add_argument("--direction", choices=["long", "short"], default="long")
    ap.add_argument("--signal", choices=["off", "builtin", "analysis"], default="off")
    ap.add_argument("--entry", type=float, default=50); ap.add_argument("--dca", type=float, default=30)
    ap.add_argument("--disc", type=float, default=0.2); ap.add_argument("--drop", type=float, default=2.0)
    ap.add_argument("--tp", type=float, default=1.0); ap.add_argument("--maxdca", type=int, default=10)
    ap.add_argument("--budget", type=float, default=500); ap.add_argument("--fee", type=float, default=0.045)
    ap.add_argument("--fast", type=int, default=12); ap.add_argument("--slow", type=int, default=48)
    ap.add_argument("--band", type=float, default=0.3); ap.add_argument("--win", type=int, default=24)
    ap.add_argument("--step", type=int, default=8)
    args = ap.parse_args()

    c = HLClient.public()
    candles = c.candles(args.coin, "1h", lookback_hours=args.days*24 + 5)
    ohlc = [(float(x["o"]), float(x["h"]), float(x["l"]), float(x["c"])) for x in candles]
    ts = [int(x["t"]) for x in candles]
    closes = [x[3] for x in ohlc]
    if len(ohlc) < 50:
        log("! prea putine lumanari"); return 1
    fund = fetch_funding(args.coin, ts[0])
    sig = precompute_signal(closes, [t/1000 for t in ts], args.signal, args.fast, args.slow, args.band/100, args.win, args.step)
    bh = (closes[-1]-closes[0])/closes[0]*100

    base = dict(entry=args.entry, dca=args.dca, disc=args.disc, drop=args.drop, tp=args.tp,
                maxdca=args.maxdca, budget=args.budget, fee=args.fee)

    if args.mode == "single":
        total, m, wr = run(args.coin, ohlc, ts, fund, sig, base, args.direction, args.budget)
        log(f"=== BACKTEST COMPLET {args.coin} {args.days}z {args.direction} semnal={args.signal} ===")
        log(f"  params: entry={args.entry} dca={args.dca} disc={args.disc}% drop={args.drop}% tp={args.tp}% maxdca={args.maxdca}")
        log(f"  REAL TOTAL: {total:+.2f}% of the budget  ⇐ realised ${m['realized']:+.2f} + funding ${m['funding']:+.2f} + the open position ${m['final_upnl']:+.2f} - fees ${m['fees']:.2f}")
        log(f"  (the realised part alone: {m['net']/args.budget*100:+.2f}% — MISLEADING if a losing position remains)")
        log(f"  cycles  : {m['cycles']}  win rate {wr:.0f}%   max drawdown ${m['maxdd']:.2f}  ({m['maxdd']/args.budget*100:.0f}% of the budget)")
        log(f"  buy&hold: {bh:+.2f}%   pozitie deschisa la final: {m['open_qty']:.4f} ({'PIERZATOARE' if m['final_upnl']<-1 else 'ok'})")
        return 0

    # Sweep TP x DROP for tuning.
    log(f"=== SWEEP {args.coin} {args.days}z {args.direction} semnal={args.signal} (buy&hold {bh:+.1f}%) ===")
    results = []
    for tp in (0.5, 0.8, 1.0, 1.5, 2.0, 3.0):
        for drop in (1.0, 2.0, 3.0, 5.0):
            P = dict(base); P["tp"] = tp; P["drop"] = drop
            net, m, wr = run(args.coin, ohlc, ts, fund, sig, P, args.direction, args.budget)
            results.append((net, tp, drop, m["cycles"], wr, m["maxdd"]))
    results.sort(reverse=True)
    log("  top 8 combinatii (net% | tp | drop | cicluri | win% | maxDD$):")
    for net, tp, drop, cyc, wr, dd in results[:8]:
        log(f"    {net:+7.2f}%  tp={tp:<4} drop={drop:<4} cic={cyc:<3} win={wr:.0f}% dd=${dd:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
