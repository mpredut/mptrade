#!/usr/bin/env python3
"""
backtest.py — Trading 212 stock DCA and take-profit backtester using Yahoo data.

Replay the strategy bar by bar with discounted entry, DCA on declines, take profit,
and STOP-LOSS, including a 0.15% FX fee per leg (~0.30% round trip). The report's
TOTAL includes marked-to-market open positions, not only realized profit, avoiding a
misleading 100% win rate while a losing position remains trapped.

  python3 backtest.py --sym NVDA --range 1y --interval 1d
  python3 backtest.py --mode sweep --sym NVDA --range 6mo --interval 1d
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

from replay import run_replay
from strategy import StratParams


def fetch_candles(sym, rng, interval):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range={rng}&interval={interval}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (backtest)"})
    with urllib.request.urlopen(req, timeout=25) as r:
        data = json.loads(r.read())
    res = (data.get("chart", {}).get("result") or [None])[0]
    if not res:
        return []
    ts = res.get("timestamp") or []
    q = (res.get("indicators", {}).get("quote") or [{}])[0]
    out = []
    for i in range(len(ts)):
        o, h, l, c = q["open"][i], q["high"][i], q["low"][i], q["close"][i]
        if None in (o, h, l, c):
            continue
        out.append((o, h, l, c))
    return out


def simulate(ohlc, P):
    """CLI compatibility wrapper over the same ``Strategy.step`` used live."""
    budget = float(P["budget"])
    entry_pct = (float(P["entry"]) / budget * 100.0) if budget > 0 else 10.0
    dca_pct = (float(P["dca"]) / budget * 100.0) if budget > 0 else 10.0
    params = StratParams.from_env({
        "STRATEGY_MODE": "avg_tp",
        "STRAT_CURRENCY": "USD",
        "YAHOO_SYMBOL": str(P.get("sym") or "REPLAY"),
        "STRAT_ENTRY_PCT": str(entry_pct),
        "STRAT_DCA_PCT": str(dca_pct),
        "STRAT_ENTRY_DISCOUNT_PCT": str(P["disc"]),
        "STRAT_DCA_DROP_PCT": str(P["drop"]),
        "STRAT_TAKEPROFIT_PCT": str(P["tp"]),
        "STRAT_MAX_DCA_BUYS": str(P["maxdca"]),
        "STRAT_MAX_BUDGET": str(budget),
        "STRAT_FX_FEE_PCT": str(P["fee"]),
        "STRAT_STOP_LOSS_PCT": str(P["sl"]),
        "STRAT_CHECK_MINUTES": "1",
        "STRAT_ORDER_TTL_MIN": "10",
        "STRAT_REENTRY_DROP_PCT": "0",
        "STRAT_REENTRY_TOLERANCE_PCT": "0.05",
        "STRAT_LOSS_ALERT_STEP": "1",
        "STRAT_LADDER_MIN_FREE": "6",
        "STRAT_SL_REBUY_ENABLED": "false",
        "STRAT_SL_REBUY_BOUNCE_PCT": "1.2",
        "STRAT_DCA_TREND_GATE_PCT": "0",
        "STRAT_TRAIL_PCT": "0",
        "STRAT_TRAIL_MIN_PROFIT_PCT": "5",
    })
    return run_replay(
        ohlc, params, bar_minutes=P.get("bar_minutes"), fx_to_usd=1.0,
    )



def interval_minutes(value: str) -> int | None:
    units = {"m": 1, "h": 60, "d": 1440}
    try:
        return int(value[:-1]) * units[value[-1].lower()]
    except (KeyError, ValueError, IndexError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Backtester DCA+TP T212 (Yahoo).")
    ap.add_argument("--mode", choices=["single", "sweep"], default="single")
    ap.add_argument("--sym", default="NVDA")
    ap.add_argument("--range", default="1y", help="ex: 1mo,3mo,6mo,1y,2y")
    ap.add_argument("--interval", default="1d", help="ex: 1h,1d")
    ap.add_argument("--entry", type=float, default=100); ap.add_argument("--dca", type=float, default=80)
    ap.add_argument("--disc", type=float, default=0.2); ap.add_argument("--drop", type=float, default=2.0)
    ap.add_argument("--tp", type=float, default=1.5); ap.add_argument("--maxdca", type=int, default=10)
    ap.add_argument("--budget", type=float, default=2000); ap.add_argument("--fee", type=float, default=0.15)
    ap.add_argument("--sl", type=float, default=10.0, help="stop-loss %% (0=stopped)")
    args = ap.parse_args()

    ohlc = fetch_candles(args.sym, args.range, args.interval)
    if len(ohlc) < 20:
        print(f"! not enough data ({len(ohlc)})"); return 1
    closes = [x[3] for x in ohlc]
    bh = (closes[-1] - closes[0]) / closes[0] * 100
    base = dict(entry=args.entry, dca=args.dca, disc=args.disc, drop=args.drop,
                tp=args.tp, maxdca=args.maxdca, budget=args.budget, fee=args.fee, sl=args.sl,
                sym=args.sym, bar_minutes=interval_minutes(args.interval))

    if args.mode == "single":
        m = simulate(ohlc, base)
        tot = m["total"]/args.budget*100
        wr = 100*m["wins"]/m["cycles"] if m["cycles"] else 0
        print(f"=== BACKTEST T212 {args.sym} {args.range}/{args.interval} ({len(ohlc)} bare) ===")
        print(f"  params: entry={args.entry} dca={args.dca} drop={args.drop}% tp={args.tp}% sl={args.sl}% fee={args.fee}%/leg")
        print(f"  REAL TOTAL: {tot:+.2f}% of the budget  ⇐ realised ${m['realized']:+.2f} + the open position ${m['final_upnl']:+.2f} - fees ${m['fees']:.2f}")
        print(f"  (the realised part alone: {m['net']/args.budget*100:+.2f}% — misleading if a losing position remains)")
        print(f"  cicluri: {m['cycles']}  win-rate {wr:.0f}%   max drawdown ${m['maxdd']:.2f}")
        print(f"  buy&hold: {bh:+.2f}%   pozitie la final: {m['open_qty']:.4f}")
        return 0

    print(f"=== SWEEP {args.sym} {args.range}/{args.interval}  (buy&hold {bh:+.1f}%) ===")
    rows = []
    for tp in (1.0, 1.5, 2.0, 3.0, 5.0):
        for drop in (1.0, 2.0, 3.0, 5.0):
            for sl in (8.0, 15.0):
                P = dict(base); P["tp"] = tp; P["drop"] = drop; P["sl"] = sl
                m = simulate(ohlc, P)
                rows.append((m["total"]/args.budget*100, tp, drop, sl, m["cycles"], m["maxdd"]))
    rows.sort(reverse=True)
    print("  top 8 (total% | tp | drop | sl | cicluri | maxDD$):")
    for tot, tp, drop, sl, cyc, dd in rows[:8]:
        print(f"    {tot:+7.2f}%  tp={tp:<4} drop={drop:<4} sl={sl:<4} cic={cyc:<3} dd=${dd:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
