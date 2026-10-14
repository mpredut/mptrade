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
try:
    from kraken import replay as rp
except ImportError:
    import replay as rp        # Faithful engine: the live strategy over OHLC data.


def _bt_params(base: dict, over: dict | None = None) -> "StratParams":
    """Build ``StratParams`` from live configuration plus backtest overrides.

    Live ``STRAT_*`` values preserve production re-entry, tranche, and tolerance
    behavior, so the faithful engine tests the production strategy rather than
    defaults.
    """
    env = None
    if not os.environ.get("STRATEGY_MODE"):
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.env")
        if os.path.exists(cfg_path):
            from botcore import parse_dotenv
            env = dict(parse_dotenv(cfg_path))
            env.update(os.environ)
    p = StratParams.from_env(env=env)
    fields = dict(
        entry_amount=base.get("entry", p.entry_amount),
        dca_amount=base.get("dca", p.dca_amount),
        entry_discount_pct=base.get("disc", p.entry_discount_pct),
        dca_drop_pct=base.get("drop_fallback", base.get("drop", p.dca_drop_pct)),
        takeprofit_pct=base.get("tp", p.takeprofit_pct),
        max_dca_buys=base.get("maxdca", p.max_dca_buys),
        max_budget=base.get("budget", p.max_budget),
        stop_loss_pct=base.get("sl", p.stop_loss_pct),
        tp_trail_pct=base.get("trail", p.tp_trail_pct),
        reentry_drop_pct=base.get("reentry_fallback", base.get("reentry_drop", base.get("reentry", p.reentry_drop_pct))),
    )
    interval = base.get("interval", 60)
    if float(interval) != float(p.trend_interval):
        fields.update(
            trend_overlay=False,
            dca_trend_brake=False,
            reentry_hybrid_enabled=False,
            surge_guard=False,
            tp_dynamic_flat=False,
        )
    if float(interval) != float(p.tp_trail_vol_interval):
        fields.update(tp_trail_adaptive=False)
    if float(interval) != float(p.dca_vol_interval):
        fields.update(dca_vol_scale_k=0.0)
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
    """Run the DCA, take-profit, and stop-loss simulation using the production engine.

    Delegates to the faithful replay engine (rp.run_replay) which runs
    strategies.spot_engine.Strategy directly, eliminating drift between
    backtest and production.
    """
    over = {}
    if sl_bounce_pct is not None:
        over["reentry_sl_bounce_pct"] = sl_bounce_pct
    params = _bt_params(P, over if over else None)
    res = rp.run_replay(
        ohlc,
        params,
        fee_pct=P.get("fee", 0.25),
        bar_minutes=P.get("interval", 60),
        reentry_series=reentry_arr,
    )
    res.setdefault("blocked_ticks", 0)
    return res


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
                    help="deprecat; motorul LIVE faithful rp.run_replay este utilizat intotdeauna")
    args = ap.parse_args()

    # Load live STRAT_* configuration so the faithful engine uses production
    # re-entry, tranche, and tolerance values.
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

    engine = "motor LIVE (faithful)"
    if args.mode == "single":
        m = rp.run_replay(
            ohlc, _bt_params(base), fee_pct=args.fee, bar_minutes=args.interval)
        tot = m["total"]/args.budget*100
        wr = 100*m["wins"]/m["cycles"] if m["cycles"] else 0
        print(f"=== BACKTEST KRAKEN {args.pair} interval={args.interval}m ({len(ohlc)} bare) [{engine}] ===")
        print(f"  params: entry={args.entry} dca={args.dca} drop={args.drop}% tp={args.tp}% sl={args.sl}% fee={args.fee}%/leg")
        print(f"  REAL TOTAL: {tot:+.2f}% of the budget  ⇐ realised ${m['realized']:+.2f} + the open position ${m['final_upnl']:+.2f} - fees ${m['fees']:.2f}")
        print(f"  (realizat singur: {m['net']/args.budget*100:+.2f}%)")
        print(f"  cicluri: {m['cycles']}  win-rate {wr:.0f}%   max drawdown ${m['maxdd']:.2f}")
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
                m = rp.run_replay(
                    ohlc, _bt_params(P), fee_pct=args.fee, bar_minutes=args.interval)
                rows.append((m["total"]/args.budget*100, tp, drop, sl, m["cycles"], m["maxdd"]))
    rows.sort(reverse=True)
    print("  top 8 (total% | tp | drop | sl | cicluri | maxDD$):")
    for tot, tp, drop, sl, cyc, dd in rows[:8]:
        print(f"    {tot:+7.2f}%  tp={tp:<4} drop={drop:<4} sl={sl:<4} cic={cyc:<3} dd=${dd:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
