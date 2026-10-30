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
import os
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.append(_HERE)


from botcore import (
    float_env as _float_env,
    load_local_module,
    required_env as _required_env,
    required_float_env as _required_float_env,
    required_int_env as _required_int_env,
    safe_pos_float,
)
from offline.backtests.metrics import format_performance_summary


_common_mod = load_local_module(_HERE, "common")
_load_hl_local = _common_mod.load_hl_local
log = _common_mod.log
_load_dotenv = _common_mod.load_dotenv
_md_local = _load_hl_local("market_data")
fetch_ohlc_candles = _md_local.fetch_ohlc_candles
fetch_funding_history = _md_local.fetch_funding_history

_hl_mod = _load_hl_local("replay")
run_replay = _hl_mod.run_replay
funding_at = _hl_mod.funding_at


fetch_funding = fetch_funding_history


def precompute_signal(closes, ts, source, fast, slow, band, win_h, step_h):
    """Return a causal signal per bar using data available through that bar only."""
    if isinstance(source, bool):
        raise ValueError(f"Invalid signal source: {source!r}")
    src = str(source or "off").strip().lower()
    if src not in ("off", "builtin", "analysis"):
        raise ValueError(f"Invalid signal source: {source!r}")
    out = ["neutral"] * len(closes)
    if src == "off":
        return ["__any__"] * len(closes)         # gate disabled
    detect_fn = None
    if src == "analysis":
        detect_fn = _load_hl_local("price_analysis").detect_long_term_trend
    for i in range(len(closes)):
        if src == "analysis":
            res = detect_fn(ts[:i+1], closes[:i+1], win_h, step_h)
            out[i] = res["direction"] if res else "neutral"
        else:  # builtin
            if fast > 0 and slow > 0 and i + 1 >= max(fast, slow):
                mf = statistics.fmean(closes[i+1-fast:i+1]); ms = statistics.fmean(closes[i+1-slow:i+1])
                diff = (mf - ms) / ms if ms else 0
                out[i] = "up" if diff > band else "down" if diff < -band else "neutral"
    return out


def simulate(ohlc, ts, fund, sig, P, direction):
    """Replay the strategy over OHLC bars and millisecond timestamps; return metrics."""
    return run_replay(ohlc, ts, fund, sig, P, direction=direction)


def run(coin, ohlc, ts, fund, sig, P, direction, budget):
    m = simulate(ohlc, ts, fund, sig, P, direction)
    b_val = safe_pos_float(budget, default=None)
    total_pct = m["total"] / b_val * 100 if b_val is not None else 0.0  # TRUE result: realized + funding + open position
    wr = 100*m["wins"]/m["cycles"] if m["cycles"] else 0
    return total_pct, m, wr


def main() -> int:
    ap = argparse.ArgumentParser(description="Backtester complet DCA+TP pe Hyperliquid.")
    ap.add_argument("--mode", choices=["single", "sweep"], default="single")
    ap.add_argument("--coin", default=None)
    ap.add_argument("--days", type=int, default=45)
    ap.add_argument("--direction", choices=["long", "short"], default=None)
    ap.add_argument("--signal", choices=["off", "builtin", "analysis"], default="off")
    ap.add_argument("--entry", type=float, default=None); ap.add_argument("--dca", type=float, default=None)
    ap.add_argument("--disc", type=float, default=None); ap.add_argument("--drop", type=float, default=None)
    ap.add_argument("--tp", type=float, default=None); ap.add_argument("--maxdca", type=int, default=None)
    ap.add_argument("--budget", type=float, default=None); ap.add_argument("--fee", type=float, default=None)
    ap.add_argument("--sl", type=float, default=None, help="stop-loss %% (0=disabled)")
    ap.add_argument("--fast", type=int, default=12); ap.add_argument("--slow", type=int, default=48)
    ap.add_argument("--band", type=float, default=0.3); ap.add_argument("--win", type=int, default=24)
    ap.add_argument("--step", type=int, default=8)
    args = ap.parse_args()

    try:
        _common_mod.load_env_stack(os.path.join(_HERE, ".env"))
    except Exception:
        pass

    coin = args.coin or _required_env("HL_COIN")
    direction = args.direction or _required_env("HL_DIRECTION")
    entry = args.entry if args.entry is not None else _required_float_env("STRAT_ENTRY")
    dca = args.dca if args.dca is not None else _required_float_env("STRAT_DCA")
    disc = args.disc if args.disc is not None else _required_float_env("STRAT_ENTRY_DISCOUNT_PCT")
    drop = args.drop if args.drop is not None else _required_float_env("STRAT_DCA_DROP_PCT")
    tp = args.tp if args.tp is not None else _required_float_env("STRAT_TAKEPROFIT_PCT")
    maxdca = args.maxdca if args.maxdca is not None else _required_int_env("STRAT_MAX_DCA_BUYS")
    budget = args.budget if args.budget is not None else _required_float_env("STRAT_MAX_BUDGET")
    fee = args.fee if args.fee is not None else _required_float_env("HL_FEE_PCT")
    sl = args.sl if args.sl is not None else (_float_env("STRAT_STOP_LOSS_PCT") or 0.0)

    records = fetch_ohlc_candles(coin, "1h", lookback_hours=args.days * 24 + 5, drop_incomplete=True)
    ohlc = [(r["open"], r["high"], r["low"], r["close"]) for r in records]
    ts = [r["timestamp"] * 1000 for r in records]
    closes = [x[3] for x in ohlc]
    if len(ohlc) < 50:
        log("! prea putine lumanari"); return 1
    if budget <= 0:
        log("! budget must be > 0"); return 1
    fund = fetch_funding(coin, ts[0])
    sig = precompute_signal(closes, [t/1000 for t in ts], args.signal, args.fast, args.slow, args.band/100, args.win, args.step)
    bh = (closes[-1]-closes[0])/closes[0]*100 if closes[0] > 0 else 0.0

    base = dict(entry=entry, dca=dca, disc=disc, drop=drop, tp=tp,
                maxdca=maxdca, budget=budget, fee=fee, sl=sl)

    if args.mode == "single":
        total, m, wr = run(coin, ohlc, ts, fund, sig, base, direction, budget)
        log(f"=== BACKTEST COMPLET {coin} {args.days}z {direction} semnal={args.signal} ===")
        log(f"  params: entry={entry} dca={dca} disc={disc}% drop={drop}% tp={tp}% sl={sl}% maxdca={maxdca}")
        log(f"  REAL TOTAL: {total:+.2f}% of the budget  ⇐ realised ${m['realized']:+.2f} + funding ${m['funding']:+.2f} + the open position ${m['final_upnl']:+.2f} - fees ${m['fees']:.2f}")
        log(f"  (the realised part alone: {m['net']/budget*100:+.2f}% — MISLEADING if a losing position remains)")
        log(f"  cycles  : {m['cycles']}  win rate {wr:.0f}%   max drawdown ${m['maxdd']:.2f}  ({m['maxdd']/budget*100:.0f}% of the budget)")
        for line in format_performance_summary(m, indent="  "):
            log(line)
        log(f"  buy&hold: {bh:+.2f}%   pozitie deschisa la final: {m['open_qty']:.4f} ({'PIERZATOARE' if m['final_upnl']<-1 else 'ok'})")
        return 0

    # Sweep TP x DROP for tuning.
    log(f"=== SWEEP {coin} {args.days}z {direction} semnal={args.signal} (buy&hold {bh:+.1f}%) ===")
    results = []
    for tp_val in (0.5, 0.8, 1.0, 1.5, 2.0, 3.0):
        for drop_val in (1.0, 2.0, 3.0, 5.0):
            P = dict(base); P["tp"] = tp_val; P["drop"] = drop_val
            net, m, wr = run(coin, ohlc, ts, fund, sig, P, direction, budget)
            results.append((net, tp_val, drop_val, m["cycles"], wr, m["maxdd"]))
    results.sort(reverse=True)
    log("  top 8 combinatii (net% | tp | drop | cicluri | win% | maxDD$):")
    for net, tp, drop, cyc, wr, dd in results[:8]:
        log(f"    {net:+7.2f}%  tp={tp:<4} drop={drop:<4} cic={cyc:<3} win={wr:.0f}% dd=${dd:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
