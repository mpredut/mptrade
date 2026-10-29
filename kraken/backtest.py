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
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.append(_HERE)

from strategies.spot_engine import StratParams
from offline.backtests.metrics import format_performance_summary
from botcore import exact_nonneg_int, load_dotenv, parse_dotenv, safe_float, safe_pos_float

import dataclasses
try:
    from kraken import replay as rp
    from kraken.market_data import fetch_ohlc_candles
except ImportError:
    import replay as rp        # Faithful engine: the live strategy over OHLC data.
    from market_data import fetch_ohlc_candles


def _bt_params(base: dict, over: dict | None = None) -> "StratParams":
    """Build ``StratParams`` from live configuration plus backtest overrides.

    Live ``STRAT_*`` values preserve production re-entry, tranche, and tolerance
    behavior, so the faithful engine tests the production strategy rather than
    defaults.
    """
    if not isinstance(base, dict):
        raise ValueError("_bt_params requires a dict for base")
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.env")
    env = dict(parse_dotenv(cfg_path))
    env.update(os.environ)
    p = StratParams.from_env(env=env)

    def _pick(keys: tuple[str, ...], default):
        for k in keys:
            if k in base and base[k] is not None:
                val = safe_float(base[k], default=None)
                if val is None:
                    raise ValueError(f"invalid value not allowed for {k}: {base[k]!r}")
                return val
        return default

    raw_maxdca = base.get("maxdca") if base.get("maxdca") is not None else p.max_dca_buys
    maxdca_i = exact_nonneg_int(raw_maxdca, default=None)
    if maxdca_i is None:
        raise ValueError(f"invalid value not allowed for maxdca: {base.get('maxdca')!r}")
    fields = dict(
        entry_amount=_pick(("entry",), p.entry_amount),
        dca_amount=_pick(("dca",), p.dca_amount),
        entry_discount_pct=_pick(("disc",), p.entry_discount_pct),
        dca_drop_pct=_pick(("drop_fallback", "drop"), p.dca_drop_pct),
        takeprofit_pct=_pick(("tp",), p.takeprofit_pct),
        max_dca_buys=maxdca_i,
        max_budget=_pick(("budget",), p.max_budget),
        stop_loss_pct=_pick(("sl",), p.stop_loss_pct),
        tp_trail_pct=_pick(("trail",), p.tp_trail_pct),
        reentry_drop_pct=_pick(("reentry_fallback", "reentry_drop", "reentry"), p.reentry_drop_pct),
    )
    raw_interval = base.get("interval") if base.get("interval") is not None else 60
    interval = safe_pos_float(raw_interval, default=None)
    if interval is None:
        raise ValueError(f"invalid interval: {raw_interval!r}")
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
    """Fetch completed Kraken candles as (open, high, low, close) tuples."""
    records = fetch_ohlc_candles(pair, interval, drop_incomplete=True)
    return [(r["open"], r["high"], r["low"], r["close"]) for r in records]


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
        fee_pct=P.get("fee") if P.get("fee") is not None else 0.25,
        bar_minutes=P.get("interval") if P.get("interval") is not None else 60,
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
                    help="deprecated; faithful live engine rp.run_replay is always used")
    args = ap.parse_args()

    # Load live STRAT_* configuration so the faithful engine uses production
    # re-entry, tranche, and tolerance values.
    load_dotenv(os.path.join(_HERE, "config.env"))

    try:
        ohlc = fetch_candles(args.pair, args.interval)
    except Exception as e:  # noqa: BLE001
        print(f"! data error: {e}"); return 1
    if len(ohlc) < 20:
        print(f"! not enough data ({len(ohlc)})"); return 1
    if args.budget <= 0:
        print("! budget must be > 0"); return 1
    closes = [x[3] for x in ohlc]
    bh = (closes[-1] - closes[0]) / closes[0] * 100 if closes[0] > 0 else 0.0
    base = dict(entry=args.entry, dca=args.dca, disc=args.disc, drop=args.drop,
                tp=args.tp, maxdca=args.maxdca, budget=args.budget, fee=args.fee,
                sl=args.sl, interval=args.interval)

    engine = "live engine (faithful)"
    if args.mode == "single":
        m = rp.run_replay(
            ohlc, _bt_params(base), fee_pct=args.fee, bar_minutes=args.interval)
        tot = m["total"]/args.budget*100
        wr = 100*m["wins"]/m["cycles"] if m["cycles"] else 0
        print(f"=== BACKTEST KRAKEN {args.pair} interval={args.interval}m ({len(ohlc)} bars) [{engine}] ===")
        print(f"  params: entry={args.entry} dca={args.dca} drop={args.drop}% tp={args.tp}% sl={args.sl}% fee={args.fee}%/leg")
        print(f"  REAL TOTAL: {tot:+.2f}% of the budget  ⇐ realised ${m['realized']:+.2f} + the open position ${m['final_upnl']:+.2f} - fees ${m['fees']:.2f}")
        print(f"  (realized alone: {m['net']/args.budget*100:+.2f}%)")
        print(f"  cycles: {m['cycles']}  win-rate {wr:.0f}%   max drawdown ${m['maxdd']:.2f}")
        for line in format_performance_summary(m, indent="  "):
            print(line)
        print(f"  buy&hold: {bh:+.2f}%   final position: {m['open_qty']:.6f}")
        return 0

    print(f"=== SWEEP {args.pair} interval={args.interval}m  (buy&hold {bh:+.1f}%) [{engine}] ===")
    if len(ohlc) > 40 and hasattr(rp, "prewarm_trend_series"):
        rp.prewarm_trend_series(ohlc, warmup=0, params=_bt_params(base), bar_minutes=args.interval)
    rows = []
    for tp in (1.0, 1.5, 2.0, 3.0, 5.0):
        for drop in (1.0, 2.0, 3.0, 5.0):
            for sl in (8.0, 15.0):
                P = dict(base); P["tp"] = tp; P["drop"] = drop; P["sl"] = sl
                m = rp.run_replay(
                    ohlc, _bt_params(P), fee_pct=args.fee, bar_minutes=args.interval)
                rows.append((m["total"]/args.budget*100, tp, drop, sl, m["cycles"], m["maxdd"]))
    rows.sort(reverse=True)
    print("  top 8 (total% | tp | drop | sl | cycles | maxDD$):")
    for tot, tp, drop, sl, cyc, dd in rows[:8]:
        print(f"    {tot:+7.2f}%  tp={tp:<4} drop={drop:<4} sl={sl:<4} cyc={cyc:<3} dd=${dd:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
