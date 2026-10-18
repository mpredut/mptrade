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
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)
_ROOT = os.path.dirname(_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from kraken.backtest import _bt_params, fetch_candles, simulate
    from kraken.market_data import fetch_ohlc_candles
    from kraken import replay as rp
except ImportError:
    from backtest import _bt_params, fetch_candles, simulate  # reuse existing fetch and fixed-TP engine
    from market_data import fetch_ohlc_candles
    import replay as rp


def fetch_with_time(pair, interval):
    """Fetch candles with timestamps for intraday distribution analysis."""
    records = fetch_ohlc_candles(pair, interval, drop_incomplete=False)
    return [(r["timestamp"], r["open"], r["high"], r["low"], r["close"]) for r in records]


def simulate_trail(ohlc, P):
    """Match ``backtest.simulate`` except for trailing exit behavior.

    Delegates to the faithful replay engine (rp.run_replay) which runs
    strategies.spot_engine.Strategy directly with trailing take-profit enabled,
    eliminating parity bugs and duplicate simulation drift.
    """
    over = {
        "tp_trend_hold": True,
        "tp_trail_pct": P.get("trail", 3.0),
        "tp_mdhtp_enabled": False,
        "reentry_drop_pct": P.get("reentry_drop", P.get("reentry", 0.0)),
    }
    params = _bt_params(P, over)
    return rp.run_replay(
        ohlc,
        params,
        fee_pct=P.get("fee", 0.25),
        bar_minutes=P.get("interval", 60),
        intrabar_high=True,
    )


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
            P_iv = {**P, "interval": iv}
            h = bh(ohlc)
            f = simulate(ohlc, P_iv)["total"]/P["budget"]*100
            t = simulate_trail(ohlc, P_iv)["total"]/P["budget"]*100
            win = "trailing" if t > f else "TP-fix"
            (trailw, fixw) = (trailw+1, fixw) if t > f else (trailw, fixw+1)
            print(f"{pair:<9} {iv:<5} {len(ohlc):<5} {h:>+8.1f}% {f:>+7.2f}% {t:>+8.2f}%  {win}")
    print(f"\nTRAILING bate TP-fix in {trailw}/{trailw+fixw} cazuri")


def run_wf(pair, interval, segments=4):
    P = {**base_params(), "interval": interval}
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
