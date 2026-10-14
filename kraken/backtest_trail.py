#!/usr/bin/env python3
"""
backtest_trail.py — compare fixed TP with trend-following trailing TP on Kraken OHLC.

The question is whether a fixed +5% TP sells too early and misses trends, while a
trailing TP (once profitable, sell only after price falls X% from its peak) captures
more. The comparison uses multiple pairs rather than only HYPE, plus sequential
walk-forward segments. TOTAL is realized P&L plus the open position, versus buy-and-hold.

  python3 backtest_trail.py                     # comparison table across several pairs
  python3 backtest_trail.py --wf HYPEUSD        # walk-forward on a single pair
  python3 backtest_trail.py --intraday HYPEUSD  # pattern by hour of day
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)
_ROOT = os.path.dirname(_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from kraken.backtest import fetch_candles, simulate
except ImportError:
    from backtest import fetch_candles, simulate  # reuse existing fetch and fixed-TP engine


def fetch_with_time(pair, interval):
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (bt)"})
    with urllib.request.urlopen(req, timeout=25) as r:
        data = json.loads(r.read())
    if data.get("error"):
        raise RuntimeError(", ".join(data["error"]))
    res = data.get("result", {})
    key = next((k for k in res if k != "last"), None)
    return [(int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4])) for x in res[key]] if key else []


def simulate_trail(ohlc, P):
    """Match ``backtest.simulate`` except for trailing exit behavior.

    Once price reaches the ``tp`` activation profit, arm the trailing exit and
    sell when the close falls ``trail`` percent below the post-activation peak,
    following the trend instead of selling immediately at +tp.
    """
    disc, drop, sl = P["disc"]/100, P["drop"]/100, P["sl"]/100
    act, trail = P["tp"]/100, P["trail"]/100
    fee = P["fee"]/100
    qty = cost = spent = 0.0
    dca = 0; last_open = None
    realized = fees = 0.0
    cycles = wins = 0
    peak = eq = 0.0; maxdd = 0.0
    rest_buy = None
    armed = False; peak_px = 0.0

    for (o, h, l, c) in ohlc:
        if rest_buy:                                        # 1) pending buy, filled on the low
            px, sz = rest_buy
            if l <= px:
                qty += sz; cost += sz*px; spent += sz*px; last_open = px
                if qty > sz + 1e-9:
                    dca += 1
                fees += fee*sz*px
                rest_buy = None
        if qty > 1e-9:                                      # 2) trailing exit
            avg = cost/qty
            if not armed and h >= avg*(1+act):
                armed = True; peak_px = h
            if armed:
                peak_px = max(peak_px, h)
                if c <= peak_px*(1-trail):
                    realized += (c-avg)*qty; fees += fee*qty*c
                    cycles += 1; wins += 1 if c > avg else 0
                    qty = cost = spent = 0.0; dca = 0; last_open = None
                    armed = False; peak_px = 0.0
        if qty > 1e-9 and sl > 0:                           # 3) stop-loss on the close
            avg = cost/qty
            if (avg - c)/avg >= sl:
                realized += (c-avg)*qty; fees += fee*qty*c
                cycles += 1
                qty = cost = spent = 0.0; dca = 0; last_open = None
                armed = False; peak_px = 0.0
        if qty <= 1e-9:                                     # 4) place the next order
            if rest_buy is None and spent + P["entry"] <= P["budget"]:
                px = c*(1-disc); rest_buy = (px, round(P["entry"]/px, 8))
        else:
            if (dca < P["maxdca"] and last_open and c <= last_open*(1-drop)
                    and spent + P["dca"] <= P["budget"] and rest_buy is None):
                px = c*(1-disc); rest_buy = (px, round(P["dca"]/px, 8))
        upnl = (c - cost/qty)*qty if qty > 1e-9 else 0
        eq = realized - fees + upnl; peak = max(peak, eq); maxdd = max(maxdd, peak - eq)

    final_upnl = (ohlc[-1][3] - cost/qty)*qty if qty > 1e-9 else 0.0
    return {"realized": realized, "fees": fees, "net": realized-fees,
            "total": realized-fees+final_upnl, "final_upnl": final_upnl,
            "cycles": cycles, "wins": wins, "maxdd": maxdd, "open_qty": qty}


def base_params(budget=3000):
    return dict(entry=300, dca=150, disc=0.8, drop=1.0, tp=5.0, maxdca=10,
                budget=budget, fee=0.25, sl=7.0, trail=3.0)


def bh(ohlc):
    return (ohlc[-1][3] - ohlc[0][3]) / ohlc[0][3] * 100


def run_compare(pairs, intervals):
    P = base_params()
    print(f"params: entry={P['entry']} dca={P['dca']} drop={P['drop']}% sl={P['sl']}% | "
          f"FIX tp={P['tp']}% | TRAIL act={P['tp']}% trail={P['trail']}%  (fee {P['fee']}%/leg)")
    print(f"{'pereche':<9} {'intv':<5} {'bare':<5} {'buy&hold':>9} {'TP-fix':>8} {'trailing':>9}  castigator")
    fixw = trailw = 0
    for pair in pairs:
        for iv in intervals:
            try:
                ohlc = fetch_candles(pair, iv)
            except Exception as e:  # noqa: BLE001
                print(f"{pair:<9} {iv:<5} EROARE: {e}"); continue
            if len(ohlc) < 30:
                print(f"{pair:<9} {iv:<5} not enough data ({len(ohlc)})"); continue
            h = bh(ohlc)
            f = simulate(ohlc, P)["total"]/P["budget"]*100
            t = simulate_trail(ohlc, P)["total"]/P["budget"]*100
            win = "trailing" if t > f else "TP-fix"
            (trailw, fixw) = (trailw+1, fixw) if t > f else (trailw, fixw+1)
            print(f"{pair:<9} {iv:<5} {len(ohlc):<5} {h:>+8.1f}% {f:>+7.2f}% {t:>+8.2f}%  {win}")
    print(f"\nTRAILING bate TP-fix in {trailw}/{trailw+fixw} cazuri")


def run_wf(pair, interval, segments=4):
    P = base_params()
    try:
        ohlc = fetch_candles(pair, interval)
    except Exception as e:  # noqa: BLE001
        print(f"EROARE: {e}"); return
    n = len(ohlc); seg = n // segments
    print(f"=== WALK-FORWARD {pair} interval={interval}m ({n} bare, {segments} segmente) ===")
    print(f"{'segment':<9} {'buy&hold':>9} {'TP-fix':>8} {'trailing':>9}  castigator")
    fixw = trailw = 0
    for i in range(segments):
        s = ohlc[i*seg:(i+1)*seg] if i < segments-1 else ohlc[i*seg:]
        if len(s) < 20:
            continue
        h = bh(s)
        f = simulate(s, P)["total"]/P["budget"]*100
        t = simulate_trail(s, P)["total"]/P["budget"]*100
        win = "trailing" if t > f else "TP-fix"
        (trailw, fixw) = (trailw+1, fixw) if t > f else (trailw, fixw+1)
        print(f"#{i+1:<8} {h:>+8.1f}% {f:>+7.2f}% {t:>+8.2f}%  {win}")
    print(f"\nTRAILING bate TP-fix in {trailw}/{trailw+fixw} segmente (out-of-sample secvential)")


def run_intraday(pair, tz_offset=3):
    """Report average (close-open)/open return by local hour.

    The sample is small (~30 days at 1h, or ~30 observations per hour), so the
    result should be treated skeptically.
    """
    try:
        rows = fetch_with_time(pair, 60)
    except Exception as e:  # noqa: BLE001
        print(f"EROARE: {e}"); return
    import collections
    by_hour = collections.defaultdict(list)
    for (ts, o, h, l, c) in rows:
        hr = (ts // 3600 + tz_offset) % 24
        if o > 0:
            by_hour[hr].append((c-o)/o*100)
    print(f"=== INTRADAY {pair} (ora LOCALA UTC+{tz_offset}, {len(rows)} bare orare) ===")
    print(f"{'hour':<5} {'n':<5} {'avg.return/hour':>14} {'% positive bars':>16}")
    cum = 0.0
    for hr in range(24):
        v = by_hour.get(hr, [])
        if not v:
            continue
        avg = sum(v)/len(v); pos = 100*sum(1 for x in v if x > 0)/len(v)
        bar = "#" * int(abs(avg)*20)
        sign = "+" if avg >= 0 else "-"
        print(f"{hr:>02}:00 {len(v):<5} {avg:>+13.3f}% {pos:>14.0f}%  {sign}{bar}")
    # Aggregate the 06:00-12:00 morning window.
    morn = [x for hr in range(6, 13) for x in by_hour.get(hr, [])]
    rest = [x for hr in list(range(0, 6)) + list(range(13, 24)) for x in by_hour.get(hr, [])]
    if morn and rest:
        print(f"\n  dimineata 06-12: medie {sum(morn)/len(morn):+.3f}%/ora ({len(morn)} obs)")
        print(f"  restul zilei   : medie {sum(rest)/len(rest):+.3f}%/ora ({len(rest)} obs)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wf", metavar="PAIR", help="walk-forward pe o pereche")
    ap.add_argument("--intraday", metavar="PAIR", help="the pattern by hour of day")
    ap.add_argument("--interval", type=int, default=60)
    args = ap.parse_args()
    if args.wf:
        run_wf(args.wf, args.interval)
    elif args.intraday:
        run_intraday(args.intraday)
    else:
        run_compare(["BTCUSD", "ETHUSD", "SOLUSD", "HYPEUSD", "XRPUSD"], [60, 240])
    return 0


if __name__ == "__main__":
    sys.exit(main())
