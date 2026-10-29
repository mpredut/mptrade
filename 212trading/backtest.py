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
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if T212_DIR not in sys.path:
    sys.path.append(T212_DIR)

from offline.backtests.evaluation import load_venue_module
from offline.backtests.metrics import format_performance_summary
from botcore import safe_float, safe_pos_float

_replay_mod = load_venue_module("212trading", "replay.py", "t212_backtest_replay")
run_replay = _replay_mod.run_replay
StratParams = _replay_mod._strat.StratParams
_md_mod = _replay_mod._md


def fetch_candles(sym, rng, interval):
    """Fetch completed Yahoo candles as (open, high, low, close) tuples."""
    records = _md_mod.fetch_yahoo_candles(sym, rng, interval, drop_incomplete=True)
    return [(r["open"], r["high"], r["low"], r["close"]) for r in records]


_SIMULATE_ENV_DEFAULTS = {
    "STRAT_FX_FEE_PCT": "0.15",
    "STRAT_STOP_LOSS_PCT": "0.0",
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
    "STRAT_SL_MHDSL_ENABLED": "false",
    "STRAT_TP_MDHTP_ENABLED": "false",
    "STRAT_REENTRY_HYBRID_ENABLED": "false",
    "STRAT_REENTRY_PEAK_RELATIVE": "false",
    "STRAT_REENTRY_PULLBACK_PCT": "1.5",
    "STRAT_REENTRY_PULLBACK_ADAPTIVE": "false",
    "STRAT_REENTRY_PULLBACK_K": "1.0",
    "STRAT_REENTRY_PULLBACK_MIN": "0.8",
    "STRAT_REENTRY_PULLBACK_MAX": "4.0",
    "STRAT_REENTRY_TTL_HOURS": "0.0",
    "STRAT_SL_DISASTER_BUFFER_RATIO": "0.30",
    "STRAT_SL_DISASTER_BUFFER_MAX_PCT": "15.0",
    "STRAT_SL_TREND_DEFER_MAX_SEC": "220.0",
    "STRAT_SL_BOUNCE_MIN_RECOVERY_PCT": "0.40",
    "STRAT_SL_BOUNCE_PULLBACK_PCT": "0.30",
    "STRAT_SL_STRUCTURAL_BREAK_ENABLED": "true",
    "STRAT_TP_STAGNATION_MAX_SEC": "1080.0",
    "STRAT_TP_LADDER": "",
    "STRAT_TP_LADDER_CURVE": "linear",
}


def simulate(ohlc, P):
    """CLI compatibility wrapper over the same ``Strategy.step`` used live."""
    if not isinstance(P, dict):
        raise ValueError("simulate requires a parameter dict P")

    def _p_val(keys: tuple[str, ...], default: object) -> object:
        for k in keys:
            if k in P and P[k] is not None:
                return P[k]
        return default

    budget = safe_pos_float(_p_val(("budget", "max_budget"), 500.0), default=500.0)
    raw_entry = safe_pos_float(_p_val(("entry", "entry_amount"), None), default=None)
    raw_dca = safe_pos_float(_p_val(("dca", "dca_amount"), None), default=None)
    entry_pct = safe_float(
        P.get("entry_pct"),
        default=((raw_entry / budget * 100.0) if budget > 0 and raw_entry is not None else 10.0),
    )
    dca_pct = safe_float(
        P.get("dca_pct"),
        default=((raw_dca / budget * 100.0) if budget > 0 and raw_dca is not None else 10.0),
    )

    env_overrides = {k: str(os.environ.get(k) or v) if k in ("STRAT_CHECK_MINUTES", "STRAT_ORDER_TTL_MIN") else str(os.environ.get(k, v)) for k, v in _SIMULATE_ENV_DEFAULTS.items()}
    env_overrides.update({
        "STRATEGY_MODE": str(_p_val(("mode",), os.environ.get("STRATEGY_MODE") or "avg_tp")),
        "STRAT_CURRENCY": str(_p_val(("currency",), os.environ.get("STRAT_CURRENCY") or "USD")),
        "YAHOO_SYMBOL": str(_p_val(("sym", "yahoo_sym"), os.environ.get("YAHOO_SYMBOL") or "REPLAY")),
        "STRAT_ENTRY_PCT": str(entry_pct),
        "STRAT_DCA_PCT": str(dca_pct),
        "STRAT_ENTRY_DISCOUNT_PCT": str(_p_val(("disc", "entry_discount_pct"), 0.3)),
        "STRAT_DCA_DROP_PCT": str(_p_val(("drop", "dca_drop_pct"), 2.0)),
        "STRAT_TAKEPROFIT_PCT": str(_p_val(("tp", "takeprofit_pct"), 1.5)),
        "STRAT_MAX_DCA_BUYS": str(_p_val(("maxdca", "max_dca_buys"), 10)),
        "STRAT_MAX_BUDGET": str(budget),
    })
    for alias, env_key in (
        (("fee", "fx_fee_pct"), "STRAT_FX_FEE_PCT"),
        (("sl", "stop_loss_pct"), "STRAT_STOP_LOSS_PCT"),
        (("check_min",), "STRAT_CHECK_MINUTES"),
        (("ttl_min",), "STRAT_ORDER_TTL_MIN"),
        (("reentry_drop",), "STRAT_REENTRY_DROP_PCT"),
        (("reentry_tol",), "STRAT_REENTRY_TOLERANCE_PCT"),
        (("loss_alert_step",), "STRAT_LOSS_ALERT_STEP"),
        (("ladder_min_free",), "STRAT_LADDER_MIN_FREE"),
        (("sl_rebuy_bounce",), "STRAT_SL_REBUY_BOUNCE_PCT"),
        (("dca_trend_gate",), "STRAT_DCA_TREND_GATE_PCT"),
        (("trail_pct",), "STRAT_TRAIL_PCT"),
        (("trail_min_profit",), "STRAT_TRAIL_MIN_PROFIT_PCT"),
    ):
        val = _p_val(alias, None)
        if val is not None:
            env_overrides[env_key] = str(val)
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
        return _md_mod.parse_interval_minutes(value)
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

    profile = args.profile.lower() if args.profile else "nvda"
    from ipo_common import load_t212_environment
    cfg_file = os.path.join(T212_DIR, f"config.{profile}.env")
    if not os.path.exists(cfg_file):
        print(f"! profile config not found: {cfg_file}")
        return 1
    load_t212_environment(os.path.join(T212_DIR, ".env"), profile_file=cfg_file)

    from botcore import (
        _parse_bool_flag,
        float_env,
        required_env,
        required_float_env,
        required_int_env,
    )

    budget = args.budget if args.budget is not None else required_float_env("STRAT_MAX_BUDGET")
    entry_pct = float_env("STRAT_ENTRY_PCT")
    dca_pct = float_env("STRAT_DCA_PCT")
    entry = args.entry if args.entry is not None else ((budget * entry_pct / 100.0) if entry_pct is not None else required_float_env("STRAT_ENTRY"))
    dca = args.dca if args.dca is not None else ((budget * dca_pct / 100.0) if dca_pct is not None else required_float_env("STRAT_DCA"))
    disc = args.disc if args.disc is not None else required_float_env("STRAT_ENTRY_DISCOUNT_PCT")
    drop = args.drop if args.drop is not None else required_float_env("STRAT_DCA_DROP_PCT")
    tp = args.tp if args.tp is not None else required_float_env("STRAT_TAKEPROFIT_PCT")
    sl = args.sl if args.sl is not None else required_float_env("STRAT_STOP_LOSS_PCT")
    fee = args.fee if args.fee is not None else required_float_env("STRAT_FX_FEE_PCT")
    sym = args.sym if (args.sym != "NVDA" or not os.environ.get("YAHOO_SYMBOL")) else required_env("YAHOO_SYMBOL")

    mdb_raw = (os.environ.get("STRAT_MAX_DCA_BUYS") or "auto").strip().lower()
    if args.maxdca is not None:
        maxdca = args.maxdca
    elif mdb_raw in ("auto", "buget", "budget"):
        maxdca = max(0, int((budget - entry) // dca)) if dca > 0 else 0
    else:
        maxdca = required_int_env("STRAT_MAX_DCA_BUYS")

    sl_mhdsl = args.mhdsl if args.mhdsl is not None else (
        _parse_bool_flag(os.environ.get("STRAT_SL_MHDSL_ENABLED"), default=False)
    )
    tp_mdhtp = args.mdhtp if args.mdhtp is not None else (
        _parse_bool_flag(os.environ.get("STRAT_TP_MDHTP_ENABLED"), default=False)
    )
    trail_pct = args.trail if args.trail is not None else (float_env("STRAT_TRAIL_PCT") or 0.0)
    trail_min_profit = args.trail_min_profit if args.trail_min_profit is not None else (float_env("STRAT_TRAIL_MIN_PROFIT_PCT") or 0.0)

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
