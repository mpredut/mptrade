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
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T212_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (T212_DIR, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from offline.backtests.metrics import format_performance_summary

try:
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "t212_backtest_replay", os.path.join(T212_DIR, "replay.py")
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    run_replay = _mod.run_replay
    StratParams = _mod._strat.StratParams
except Exception:
    from replay import run_replay
    from strategy import StratParams


def fetch_candles(sym, rng, interval):
    """Fetch completed Yahoo candles as (open, high, low, close) tuples."""
    try:
        from market_data import fetch_yahoo_candles
    except ImportError:
        import importlib.util
        _md_spec = importlib.util.spec_from_file_location(
            "t212_backtest_md", os.path.join(T212_DIR, "market_data.py")
        )
        _md = importlib.util.module_from_spec(_md_spec)
        sys.modules[_md_spec.name] = _md
        _md_spec.loader.exec_module(_md)
        fetch_yahoo_candles = _md.fetch_yahoo_candles
    records = fetch_yahoo_candles(sym, rng, interval, drop_incomplete=True)
    return [(r["open"], r["high"], r["low"], r["close"]) for r in records]


def simulate(ohlc, P):
    """CLI compatibility wrapper over the same ``Strategy.step`` used live."""
    budget = float(P["budget"])
    entry_pct = float(
        P["entry_pct"]
        if P.get("entry_pct") is not None
        else ((float(P.get("entry", 0.0)) / budget * 100.0) if budget > 0 and P.get("entry") is not None else 10.0)
    )
    dca_pct = float(
        P["dca_pct"]
        if P.get("dca_pct") is not None
        else ((float(P.get("dca", 0.0)) / budget * 100.0) if budget > 0 and P.get("dca") is not None else 10.0)
    )

    env_overrides = {
        "STRATEGY_MODE": str(P.get("mode") or os.environ.get("STRATEGY_MODE") or "avg_tp"),
        "STRAT_CURRENCY": str(P.get("currency") or os.environ.get("STRAT_CURRENCY") or "USD"),
        "YAHOO_SYMBOL": str(P.get("sym") or os.environ.get("YAHOO_SYMBOL") or "REPLAY"),
        "STRAT_ENTRY_PCT": str(entry_pct),
        "STRAT_DCA_PCT": str(dca_pct),
        "STRAT_ENTRY_DISCOUNT_PCT": str(P["disc"]),
        "STRAT_DCA_DROP_PCT": str(P["drop"]),
        "STRAT_TAKEPROFIT_PCT": str(P["tp"]),
        "STRAT_MAX_DCA_BUYS": str(P["maxdca"]),
        "STRAT_MAX_BUDGET": str(budget),
        "STRAT_FX_FEE_PCT": str(P["fee"]),
        "STRAT_STOP_LOSS_PCT": str(P["sl"]),
        "STRAT_CHECK_MINUTES": str(P.get("check_min") or os.environ.get("STRAT_CHECK_MINUTES") or "1"),
        "STRAT_ORDER_TTL_MIN": str(P.get("ttl_min") or os.environ.get("STRAT_ORDER_TTL_MIN") or "10"),
        "STRAT_REENTRY_DROP_PCT": str(P.get("reentry_drop", os.environ.get("STRAT_REENTRY_DROP_PCT", "0"))),
        "STRAT_REENTRY_TOLERANCE_PCT": str(P.get("reentry_tol", os.environ.get("STRAT_REENTRY_TOLERANCE_PCT", "0.05"))),
        "STRAT_LOSS_ALERT_STEP": str(P.get("loss_alert_step", os.environ.get("STRAT_LOSS_ALERT_STEP", "1"))),
        "STRAT_LADDER_MIN_FREE": str(P.get("ladder_min_free", os.environ.get("STRAT_LADDER_MIN_FREE", "6"))),
        "STRAT_SL_REBUY_ENABLED": str(P.get("sl_rebuy_enabled", os.environ.get("STRAT_SL_REBUY_ENABLED", "false"))).lower(),
        "STRAT_SL_REBUY_BOUNCE_PCT": str(P.get("sl_rebuy_bounce", os.environ.get("STRAT_SL_REBUY_BOUNCE_PCT", "1.2"))),
        "STRAT_DCA_TREND_GATE_PCT": str(P.get("dca_trend_gate", os.environ.get("STRAT_DCA_TREND_GATE_PCT", "0"))),
        "STRAT_TRAIL_PCT": str(P.get("trail_pct", os.environ.get("STRAT_TRAIL_PCT", "0"))),
        "STRAT_TRAIL_MIN_PROFIT_PCT": str(P.get("trail_min_profit", os.environ.get("STRAT_TRAIL_MIN_PROFIT_PCT", "5"))),
        "STRAT_SL_MHDSL_ENABLED": str(P.get("sl_mhdsl_enabled", os.environ.get("STRAT_SL_MHDSL_ENABLED", "false"))).lower(),
        "STRAT_TP_MDHTP_ENABLED": str(P.get("tp_mdhtp_enabled", os.environ.get("STRAT_TP_MDHTP_ENABLED", "false"))).lower(),
        "STRAT_REENTRY_HYBRID_ENABLED": str(P.get("reentry_hybrid_enabled", os.environ.get("STRAT_REENTRY_HYBRID_ENABLED", "false"))).lower(),
        "STRAT_REENTRY_PEAK_RELATIVE": str(P.get("reentry_peak_relative", os.environ.get("STRAT_REENTRY_PEAK_RELATIVE", "false"))).lower(),
        "STRAT_REENTRY_PULLBACK_PCT": str(P.get("reentry_pullback_pct", os.environ.get("STRAT_REENTRY_PULLBACK_PCT", "1.5"))),
        "STRAT_REENTRY_PULLBACK_ADAPTIVE": str(P.get("reentry_pullback_adaptive", os.environ.get("STRAT_REENTRY_PULLBACK_ADAPTIVE", "false"))).lower(),
        "STRAT_REENTRY_PULLBACK_K": str(P.get("reentry_pullback_k", os.environ.get("STRAT_REENTRY_PULLBACK_K", "1.0"))),
        "STRAT_REENTRY_PULLBACK_MIN": str(P.get("reentry_pullback_min", os.environ.get("STRAT_REENTRY_PULLBACK_MIN", "0.8"))),
        "STRAT_REENTRY_PULLBACK_MAX": str(P.get("reentry_pullback_max", os.environ.get("STRAT_REENTRY_PULLBACK_MAX", "4.0"))),
        "STRAT_REENTRY_TTL_HOURS": str(P.get("reentry_ttl_hours", os.environ.get("STRAT_REENTRY_TTL_HOURS", "0.0"))),
        "STRAT_SL_DISASTER_BUFFER_RATIO": str(P.get("sl_disaster_buffer_ratio", os.environ.get("STRAT_SL_DISASTER_BUFFER_RATIO", "0.30"))),
        "STRAT_SL_DISASTER_BUFFER_MAX_PCT": str(P.get("sl_disaster_buffer_max_pct", os.environ.get("STRAT_SL_DISASTER_BUFFER_MAX_PCT", "15.0"))),
        "STRAT_SL_TREND_DEFER_MAX_SEC": str(P.get("sl_trend_defer_max_sec", os.environ.get("STRAT_SL_TREND_DEFER_MAX_SEC", "220.0"))),
        "STRAT_SL_BOUNCE_MIN_RECOVERY_PCT": str(P.get("sl_bounce_min_recovery_pct", os.environ.get("STRAT_SL_BOUNCE_MIN_RECOVERY_PCT", "0.40"))),
        "STRAT_SL_BOUNCE_PULLBACK_PCT": str(P.get("sl_bounce_pullback_pct", os.environ.get("STRAT_SL_BOUNCE_PULLBACK_PCT", "0.30"))),
        "STRAT_SL_STRUCTURAL_BREAK_ENABLED": str(P.get("sl_structural_break_enabled", os.environ.get("STRAT_SL_STRUCTURAL_BREAK_ENABLED", "true"))).lower(),
        "STRAT_TP_STAGNATION_MAX_SEC": str(P.get("tp_stagnation_max_sec", os.environ.get("STRAT_TP_STAGNATION_MAX_SEC", "1080.0"))),
        "STRAT_TP_LADDER": str(os.environ.get("STRAT_TP_LADDER", "")),
        "STRAT_TP_LADDER_CURVE": str(os.environ.get("STRAT_TP_LADDER_CURVE", "linear")),
    }
    for k, v in P.items():
        if k == "tp_ladder":
            if v is None:
                env_overrides["STRAT_TP_LADDER"] = ""
            elif isinstance(v, (list, tuple)):
                env_overrides["STRAT_TP_LADDER"] = ",".join(
                    f"{lvl}:{frac * 100.0 if frac <= 1.0 else frac}" for lvl, frac in v
                )
            else:
                env_overrides["STRAT_TP_LADDER"] = str(v)
        elif v is None or k in ("reentry_drop", "reentry_tol", "sl_rebuy_bounce"):
            continue
        elif k.startswith("STRAT_") or k.startswith("sl_") or k.startswith("tp_") or k.startswith("reentry_"):
            env_key = k if k.startswith("STRAT_") else f"STRAT_{k.upper()}"
            env_overrides[env_key] = str(v).lower() if isinstance(v, bool) else str(v)

    params = StratParams.from_env(env_overrides)
    return run_replay(
        ohlc, params, bar_minutes=P.get("bar_minutes"), fx_to_usd=1.0,
    )



def interval_minutes(value: str) -> int | None:
    try:
        try:
            from market_data import parse_interval_minutes
        except ImportError:
            import importlib.util
            _spec = importlib.util.spec_from_file_location(
                "t212_md_parse", os.path.join(T212_DIR, "market_data.py")
            )
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            parse_interval_minutes = _mod.parse_interval_minutes
        return parse_interval_minutes(value)
    except (KeyError, ValueError, IndexError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Backtester DCA+TP T212 (Yahoo).")
    ap.add_argument("--mode", choices=["single", "sweep"], default="single")
    ap.add_argument("--profile", "-p", default=None, help="config profile to load (e.g. nvda, spcx, rgnt)")
    ap.add_argument("--sym", default="NVDA")
    ap.add_argument("--range", default="1y", help="ex: 1mo,3mo,6mo,1y,2y")
    ap.add_argument("--interval", default="1d", help="ex: 1h,1d")
    ap.add_argument("--entry", type=float, default=None); ap.add_argument("--dca", type=float, default=None)
    ap.add_argument("--disc", type=float, default=None); ap.add_argument("--drop", type=float, default=None)
    ap.add_argument("--tp", type=float, default=None); ap.add_argument("--maxdca", type=int, default=None)
    ap.add_argument("--budget", type=float, default=None); ap.add_argument("--fee", type=float, default=None)
    ap.add_argument("--sl", type=float, default=None, help="stop-loss %% (0=stopped)")
    ap.add_argument("--mhdsl", action=argparse.BooleanOptionalAction, default=None, help="enable/disable MHDSL stop loss")
    ap.add_argument("--mdhtp", action=argparse.BooleanOptionalAction, default=None, help="enable/disable MDHTP take profit")
    ap.add_argument("--trail", type=float, default=None, help="trailing percent")
    ap.add_argument("--trail-min-profit", type=float, default=None, help="min profit %% to arm trailing")
    args = ap.parse_args()

    if args.profile:
        from ipo_common import load_t212_environment
        cfg_file = os.path.join(T212_DIR, f"config.{args.profile.lower()}.env")
        if not os.path.exists(cfg_file):
            print(f"! profile config not found: {cfg_file}")
            return 1
        load_t212_environment(os.path.join(T212_DIR, ".env"), profile_file=cfg_file)

    budget = args.budget if args.budget is not None else float(os.environ.get("STRAT_MAX_BUDGET", "2000"))
    entry_pct = float(os.environ.get("STRAT_ENTRY_PCT", "10"))
    dca_pct = float(os.environ.get("STRAT_DCA_PCT", "10"))
    entry = args.entry if args.entry is not None else (budget * entry_pct / 100.0)
    dca = args.dca if args.dca is not None else (budget * dca_pct / 100.0)
    disc = args.disc if args.disc is not None else float(os.environ.get("STRAT_ENTRY_DISCOUNT_PCT", "0.2"))
    drop = args.drop if args.drop is not None else float(os.environ.get("STRAT_DCA_DROP_PCT", "2.0"))
    tp = args.tp if args.tp is not None else float(os.environ.get("STRAT_TAKEPROFIT_PCT", "1.5"))
    sl = args.sl if args.sl is not None else float(os.environ.get("STRAT_STOP_LOSS_PCT", "10.0"))
    fee = args.fee if args.fee is not None else float(os.environ.get("STRAT_FX_FEE_PCT", "0.15"))
    sym = args.sym if (args.sym != "NVDA" or not os.environ.get("YAHOO_SYMBOL")) else os.environ.get("YAHOO_SYMBOL", "NVDA")

    mdb_raw = os.environ.get("STRAT_MAX_DCA_BUYS", "10").strip().lower()
    if args.maxdca is not None:
        maxdca = args.maxdca
    elif mdb_raw in ("auto", "buget", "budget"):
        maxdca = max(0, int((budget - entry) // dca)) if dca > 0 else 0
    else:
        try:
            maxdca = int(float(mdb_raw))
        except ValueError:
            maxdca = 10

    sl_mhdsl = args.mhdsl if args.mhdsl is not None else (
        os.environ.get("STRAT_SL_MHDSL_ENABLED", "false").lower() in ("true", "1", "yes")
    )
    tp_mdhtp = args.mdhtp if args.mdhtp is not None else (
        os.environ.get("STRAT_TP_MDHTP_ENABLED", "false").lower() in ("true", "1", "yes")
    )
    trail_pct = args.trail if args.trail is not None else float(os.environ.get("STRAT_TRAIL_PCT", "0"))
    trail_min_profit = args.trail_min_profit if args.trail_min_profit is not None else float(os.environ.get("STRAT_TRAIL_MIN_PROFIT_PCT", "5"))

    ohlc = fetch_candles(sym, args.range, args.interval)
    if len(ohlc) < 20:
        print(f"! not enough data ({len(ohlc)})"); return 1
    if budget <= 0:
        print("! budget must be > 0"); return 1
    closes = [x[3] for x in ohlc]
    bh = (closes[-1] - closes[0]) / closes[0] * 100 if closes[0] > 0 else 0.0
    base = dict(entry=entry, dca=dca, disc=disc, drop=drop,
                tp=tp, maxdca=maxdca, budget=budget, fee=fee, sl=sl,
                sym=sym, bar_minutes=interval_minutes(args.interval),
                sl_mhdsl_enabled=sl_mhdsl, tp_mdhtp_enabled=tp_mdhtp,
                trail_pct=trail_pct, trail_min_profit=trail_min_profit)

    if args.mode == "single":
        m = simulate(ohlc, base)
        tot = m["total"]/budget*100
        wr = 100*m["wins"]/m["cycles"] if m["cycles"] else 0
        prof_info = f" [profile={args.profile}]" if args.profile else ""
        print(f"=== BACKTEST T212 {sym} {args.range}/{args.interval} ({len(ohlc)} bare){prof_info} ===")
        print(f"  params: entry={entry:.1f} dca={dca:.1f} drop={drop}% tp={tp}% sl={sl}% fee={fee}%/leg | MHDSL={'ON' if sl_mhdsl else 'OFF'} MDHTP={'ON' if tp_mdhtp else 'OFF'}")
        print(f"  REAL TOTAL: {tot:+.2f}% of the budget  ⇐ realised ${m['realized']:+.2f} + the open position ${m['final_upnl']:+.2f} - fees ${m['fees']:.2f}")
        print(f"  (the realised part alone: {m['net']/budget*100:+.2f}% — misleading if a losing position remains)")
        print(f"  cicluri: {m['cycles']}  win-rate {wr:.0f}%   max drawdown ${m['maxdd']:.2f}")
        for line in format_performance_summary(m, indent="  "):
            print(line)
        print(f"  buy&hold: {bh:+.2f}%   pozitie la final: {m['open_qty']:.4f}")
        return 0

    print(f"=== SWEEP {sym} {args.range}/{args.interval}  (buy&hold {bh:+.1f}%) ===")
    rows = []
    for tp in (1.0, 1.5, 2.0, 3.0, 5.0):
        for drop in (1.0, 2.0, 3.0, 5.0):
            for sl in (8.0, 15.0):
                P = dict(base); P["tp"] = tp; P["drop"] = drop; P["sl"] = sl
                m = simulate(ohlc, P)
                rows.append((m["total"]/budget*100, tp, drop, sl, m["cycles"], m["maxdd"]))
    rows.sort(reverse=True)
    print("  top 8 (total% | tp | drop | sl | cicluri | maxDD$):")
    for tot, tp, drop, sl, cyc, dd in rows[:8]:
        print(f"    {tot:+7.2f}%  tp={tp:<4} drop={drop:<4} sl={sl:<4} cic={cyc:<3} dd=${dd:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
