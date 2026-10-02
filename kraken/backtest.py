#!/usr/bin/env python3
"""
backtest.py — DCA + take-profit backtester for Kraken spot using Kraken OHLC data.

It replays the strategy bar by bar (market-discount entry, DCA on declines,
take-profit, and stop-loss), with Kraken fees (~0.25% per taker leg). The report
is honest: TOTAL includes the mark-to-market open position, not just realized profit.

  python3 backtest.py --pair HYPEUSD --interval 60
  python3 backtest.py --mode sweep --pair HYPEUSD --interval 240
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.request

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from strategies import spot_rules as sr
from strategies.spot_engine import StratParams

import dataclasses
import replay as rp        # Faithful engine: the live strategy over OHLC data.


def _bt_params(base: dict, over: dict | None = None) -> "StratParams":
    """Build ``StratParams`` from live configuration plus backtest overrides.

    Live ``STRAT_*`` values preserve production re-entry, tranche, and tolerance
    behavior, so the faithful engine tests the production strategy rather than
    defaults. This assumes ``kraken/config.env`` has already been loaded.
    """
    p = StratParams.from_env()
    fields = dict(entry_amount=base["entry"], dca_amount=base["dca"],
                  entry_discount_pct=base["disc"], dca_drop_pct=base["drop"],
                  takeprofit_pct=base["tp"], max_dca_buys=base["maxdca"],
                  max_budget=base["budget"], stop_loss_pct=base["sl"])
    if over:
        fields.update(over)
    return dataclasses.replace(p, **fields)


def fetch_candles(pair, interval):
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (backtest)"})
    with urllib.request.urlopen(req, timeout=25) as r:
        data = json.loads(r.read())
    if data.get("error"):
        raise RuntimeError(", ".join(data["error"]))
    res = data.get("result", {})
    key = next((k for k in res if k != "last"), None)
    if not key:
        return []
    # The final candle may still be forming; historical decisions use only closed
    # bars, matching the live signal and walk-forward runner.
    return [(float(x[1]), float(x[2]), float(x[3]), float(x[4])) for x in res[key][:-1]]


def simulate(ohlc, P, reentry_arr=None, sl_bounce_pct=None):
    """Run the DCA, take-profit, and stop-loss simulation.

    ``sl_bounce_pct`` is optional; ``None`` preserves the legacy behavior. When
    provided, re-entry after a stop-loss waits for a rebound of that percentage
    from the post-sale low instead of another decline below the sale price. This
    mirrors the August 4 strategy fix that avoids remaining sidelined during a
    recovery after a stop-loss.

    ``reentry_arr`` is also optional and ``None`` preserves legacy behavior.
    When supplied, it provides a per-bar threshold (NaN selects
    ``P["reentry_fallback"]``) for the re-entry barrier after a TP/SL close. The
    original simulator omitted the production strategy's explicit wait below
    ``last_sell_price * (1 - reentry_pct / 100)``. The deterministic
    ``P["reentry_tolerance_pct"]`` value (default zero) controls how close to the
    threshold counts as reached.
    """
    fee = P["fee"]/100
    reentry_tol = P.get("reentry_tolerance_pct", 0.0)
    qty = cost = spent = 0.0
    dca = 0; last_open = None
    realized = fees = 0.0
    cycles = wins = 0
    peak = eq = 0.0; maxdd = 0.0
    rest_buy = None; rest_sell = None
    last_sell_price = None
    last_exit_kind = None; sl_low = None   # stop-aware re-entry rebound reference
    blocked_ticks = 0

    for i, (o, h, l, c) in enumerate(ohlc):
        if rest_buy:
            px, sz = rest_buy
            if l <= px:
                qty += sz; cost += sz*px; spent += sz*px; last_open = px
                if qty > sz + 1e-9:
                    dca += 1
                fees += fee*sz*px
                rest_buy = None; rest_sell = None
        if rest_sell and qty > 1e-9:
            px, sz = rest_sell
            if h >= px:
                avg = cost/qty
                realized += (px-avg)*sz; fees += fee*sz*px
                cycles += 1; wins += 1 if px > avg else 0
                last_sell_price = px; last_exit_kind = "TP"
                qty = cost = spent = 0.0; dca = 0; last_open = None
                rest_sell = None; rest_buy = None
        if qty > 1e-9 and P["sl"] > 0:                  # STOP-LOSS pe close
            avg = cost/qty
            if sr.hit_stop(avg, c, P["sl"]):
                realized += (c-avg)*qty; fees += fee*qty*c
                cycles += 1
                last_sell_price = c; last_exit_kind = "STOP"; sl_low = c
                qty = cost = spent = 0.0; dca = 0; last_open = None
                rest_sell = None; rest_buy = None
        if qty <= 1e-9:
            if rest_buy is None and spent + P["entry"] <= P["budget"]:
                blocked = False
                if sl_bounce_pct and last_exit_kind == "STOP" and last_sell_price:
                    # Stop-aware re-entry waits for a rebound from the low, not a decline.
                    sl_low = c if sl_low is None else min(sl_low, c)
                    if sr.reentry_stop_blocked(c, sl_low, sl_bounce_pct, reentry_tol):
                        blocked = True
                        blocked_ticks += 1
                elif reentry_arr is not None and last_sell_price:
                    r = reentry_arr[i]
                    reentry_pct = P.get("reentry_fallback", 0.0) if (isinstance(r, float) and math.isnan(r)) else r
                    if sr.reentry_drop_blocked(c, last_sell_price, reentry_pct, reentry_tol):
                        blocked = True
                        blocked_ticks += 1
                if not blocked:
                    px = sr.entry_price(c, P["disc"]); rest_buy = (px, round(P["entry"]/px, 8))
        else:
            avg = cost/qty
            rest_sell = (sr.tp_price(avg, P["tp"]), qty)
            if (dca < P["maxdca"] and last_open and sr.dca_price_hit(c, last_open, P["drop"], 0.0)
                    and spent + P["dca"] <= P["budget"] and rest_buy is None):
                px = sr.entry_price(c, P["disc"]); rest_buy = (px, round(P["dca"]/px, 8))
        upnl = (c - cost/qty)*qty if qty > 1e-9 else 0
        eq = realized - fees + upnl; peak = max(peak, eq); maxdd = max(maxdd, peak - eq)

    final_upnl = (ohlc[-1][3] - cost/qty)*qty if qty > 1e-9 else 0.0
    return {"realized": realized, "fees": fees, "net": realized - fees,
            "total": realized - fees + final_upnl, "final_upnl": final_upnl,
            "cycles": cycles, "wins": wins, "maxdd": maxdd, "open_qty": qty,
            "blocked_ticks": blocked_ticks}


def main() -> int:
    ap = argparse.ArgumentParser(description="Backtester DCA+TP Kraken (OHLC).")
    ap.add_argument("--mode", choices=["single", "sweep"], default="single")
    ap.add_argument("--pair", default="HYPEUSD")
    ap.add_argument("--interval", type=int, default=60, help="minute: 60=1h, 240=4h, 1440=1z")
    ap.add_argument("--entry", type=float, default=100); ap.add_argument("--dca", type=float, default=50)
    ap.add_argument("--disc", type=float, default=0.2); ap.add_argument("--drop", type=float, default=2.0)
    ap.add_argument("--tp", type=float, default=1.9); ap.add_argument("--maxdca", type=int, default=10)
    ap.add_argument("--budget", type=float, default=1000); ap.add_argument("--fee", type=float, default=0.25)
    ap.add_argument("--sl", type=float, default=10.0, help="stop-loss %% (0=stopped)")
    ap.add_argument("--fast", action="store_true",
                    help="simulate() rapid (fill OHLC aproximativ, optimist pe stop) in loc de motorul LIVE faithful")
    args = ap.parse_args()

    # Load live STRAT_* configuration so the faithful engine uses production
    # re-entry, tranche, and tolerance values. This does not affect --fast.
    import os as _os
    from kraken_common import load_dotenv as _load_dotenv
    try:
        _load_dotenv(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "config.env"))
    except Exception:  # noqa: BLE001 — from_env falls back to defaults when absent
        pass

    try:
        ohlc = fetch_candles(args.pair, args.interval)
    except Exception as e:  # noqa: BLE001
        print(f"! eroare date: {e}"); return 1
    if len(ohlc) < 20:
        print(f"! not enough data ({len(ohlc)})"); return 1
    closes = [x[3] for x in ohlc]
    bh = (closes[-1] - closes[0]) / closes[0] * 100
    base = dict(entry=args.entry, dca=args.dca, disc=args.disc, drop=args.drop,
                tp=args.tp, maxdca=args.maxdca, budget=args.budget, fee=args.fee, sl=args.sl)

    engine = "simulate (rapid)" if args.fast else "motor LIVE (faithful)"
    if args.mode == "single":
        m = simulate(ohlc, base) if args.fast else rp.run_replay(
            ohlc, _bt_params(base), fee_pct=args.fee, bar_minutes=args.interval)
        tot = m["total"]/args.budget*100
        wr = 100*m["wins"]/m["cycles"] if m["cycles"] else 0
        print(f"=== BACKTEST KRAKEN {args.pair} interval={args.interval}m ({len(ohlc)} bare) [{engine}] ===")
        print(f"  params: entry={args.entry} dca={args.dca} drop={args.drop}% tp={args.tp}% sl={args.sl}% fee={args.fee}%/leg")
        print(f"  REAL TOTAL: {tot:+.2f}% of the budget  ⇐ realised ${m['realized']:+.2f} + the open position ${m['final_upnl']:+.2f} - fees ${m['fees']:.2f}")
        print(f"  (realizat singur: {m['net']/args.budget*100:+.2f}%)")
        print(f"  cicluri: {m['cycles']}  win-rate {wr:.0f}%   max drawdown ${m['maxdd']:.2f}")
        if not args.fast:
            def metric(name, digits=2):
                value = m.get(name)
                return "n/a" if value is None else f"{value:.{digits}f}"
            print(f"  RISC: maxDD {metric('max_drawdown_pct')}%  "
                  f"Sharpe {metric('sharpe')}  Sortino {metric('sortino')}  "
                  f"Calmar {metric('calmar')}  CVaR95 {metric('cvar_95_pct')}%")
            print(f"  EXECUTIE: expunere {metric('exposure_pct')}%  "
                  f"turnover {metric('turnover_pct')}%  fills {m['fills']}  "
                  f"profit-factor {metric('profit_factor')}  "
                  f"expectancy ${metric('expectancy')}")
        print(f"  buy&hold: {bh:+.2f}%   pozitie la final: {m['open_qty']:.6f}")
        return 0

    print(f"=== SWEEP {args.pair} interval={args.interval}m  (buy&hold {bh:+.1f}%) [{engine}] ===")
    rows = []
    for tp in (1.0, 1.5, 2.0, 3.0, 5.0):
        for drop in (1.0, 2.0, 3.0, 5.0):
            for sl in (8.0, 15.0):
                P = dict(base); P["tp"] = tp; P["drop"] = drop; P["sl"] = sl
                m = simulate(ohlc, P) if args.fast else rp.run_replay(
                    ohlc, _bt_params(P), fee_pct=args.fee, bar_minutes=args.interval)
                rows.append((m["total"]/args.budget*100, tp, drop, sl, m["cycles"], m["maxdd"]))
    rows.sort(reverse=True)
    print("  top 8 (total% | tp | drop | sl | cicluri | maxDD$):")
    for tot, tp, drop, sl, cyc, dd in rows[:8]:
        print(f"    {tot:+7.2f}%  tp={tp:<4} drop={drop:<4} sl={sl:<4} cic={cyc:<3} dd=${dd:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
