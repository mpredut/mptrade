#!/usr/bin/env python3
"""
backtest.py -- COMPLETE Hyperliquid DCA and take-profit strategy backtester.

Replay the strategy bar by bar over one-hour OHLC candles:
  * LIMIT entry at market-/+discount, filling when the bar touches price;
  * DCA after price moves against the position by drop%;
  * take profit at average_price*(1+/-tp), then restart the cycle;
  * include per-fill FEES AND FUNDING from real HL history;
  * optional signal gate allowing longs only on up and shorts only on down.

Modes:
  single -- one run with a detailed report
  sweep  -- test many TP/DROP combinations and show the best for tuning

Run with the virtual environment:
  python backtest.py --coin HYPE --days 45 --direction short
  python backtest.py --mode sweep --coin HYPE --days 45 --direction short --signal analysis
"""

from __future__ import annotations

import bisect
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


from botcore import (
    exact_nonneg_int,
    exact_pos_int,
    load_local_module,
    log,
    normalize_epoch_sec,
    optional_env,
    required_env,
    required_float_env,
    required_int_env,
    safe_float,
    safe_int,
    safe_nonneg_float,
    safe_pos_float,
)
from strategies.replay_metrics import (
    add_common_backtest_cli_args,
    buy_hold_pct,
    create_backtest_cli_parser,
    run_cli_single_or_sweep_backtest,
)


_common_mod = load_local_module(_HERE, "common")
load_isolated_hl_cfg = _common_mod.load_isolated_hl_cfg
_md_local = load_local_module(_HERE, "market_data")
fetch_ohlc_candles = _md_local.fetch_ohlc_candles
fetch_funding_history = _md_local.fetch_funding_history

_hl_mod = load_local_module(_HERE, "replay")
run_replay = _hl_mod.run_replay


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
    fast_i = safe_int(fast, default=0)
    slow_i = safe_int(slow, default=0)
    band_f = safe_float(band, default=0.0)
    win_f = safe_pos_float(win_h, default=24.0)
    step_f = safe_pos_float(step_h, default=8.0)
    signals_mod = load_local_module(_HERE, "signals")
    detect_fn = signals_mod.detect_long_term_trend if src == "analysis" else None
    classify_fn = signals_mod.classify_fast_slow_closes
    ts_sec_norm = [normalize_epoch_sec(t, default=(idx * 3600) * 1.0) or 0.0 for idx, t in enumerate(ts)]
    horizon_sec = max((win_f + step_f * 14.0) * 3600.0, 256.0 * 3600.0)
    max_lookback = max(256, safe_int(math.ceil(win_f + step_f * 14.0) + 16, default=256))
    for i in range(len(closes)):
        if src == "analysis":
            if ts_sec_norm and ts_sec_norm[i] > 0:
                cutoff = ts_sec_norm[i] - horizon_sec
                lo_idx = max(0, bisect.bisect_left(ts_sec_norm, cutoff, 0, i + 1))
            else:
                lo_idx = max(0, i + 1 - max_lookback)
            res = detect_fn(ts[lo_idx:i + 1], closes[lo_idx:i + 1], win_f, step_f)
            out[i] = res["direction"] if res else "neutral"
        else:  # builtin
            if fast_i > 0 and slow_i > 0 and i + 1 >= max(fast_i, slow_i):
                win_fast = [cf for c in closes[i+1-fast_i:i+1] if (cf := safe_pos_float(c, default=None)) is not None]
                win_slow = [cs for c in closes[i+1-slow_i:i+1] if (cs := safe_pos_float(c, default=None)) is not None]
                if len(win_fast) == fast_i and len(win_slow) == slow_i:
                    _, _, _, out[i] = classify_fn(win_fast, win_slow, band_f)
    return out


def run(coin, ohlc, ts, fund, sig, P, direction, budget):
    m = run_replay(ohlc, ts, fund, sig, P, direction=direction, coin=coin)
    b_val = safe_pos_float(budget, default=None)
    total_pct = m["total"] / b_val * 100 if b_val is not None else 0.0  # TRUE result: realized + funding + open position
    wr = 100*m["wins"]/m["cycles"] if m["cycles"] else 0
    return total_pct, m, wr


def main(argv: list[str] | None = None) -> int:
    ap = create_backtest_cli_parser("Complete DCA+TP backtester on Hyperliquid.")
    ap.add_argument("--coin", default=None)
    ap.add_argument("--days", default=45)
    ap.add_argument("--direction", choices=["long", "short"], default=None)
    ap.add_argument("--signal", choices=["off", "builtin", "analysis"], default="off")
    add_common_backtest_cli_args(ap)
    ap.add_argument("--fast", default=None)
    ap.add_argument("--slow", default=None)
    ap.add_argument("--band", default=None)
    ap.add_argument("--win", default=None)
    ap.add_argument("--step", default=None)
    args = ap.parse_args(argv)
    days_i = exact_pos_int(args.days, default=None)
    if days_i is None:
        ap.error("--days must be a positive integer")

    cfg = load_isolated_hl_cfg(_HERE)

    coin = (args.coin or required_env("HL_COIN", env=cfg)).strip().upper()
    if (
        not coin
        or coin.startswith("-")
        or ".." in coin
        or "/" in coin
        or "\\" in coin
        or any(ch.isspace() for ch in coin)
    ):
        ap.error(f"invalid --coin: {coin!r}")
    direction = (args.direction or required_env("HL_DIRECTION", env=cfg)).strip().lower()
    if direction not in ("long", "short"):
        ap.error(f"invalid --direction: {direction!r}")
    entry = args.entry if args.entry is not None else required_float_env("STRAT_ENTRY", env=cfg)
    dca = args.dca if args.dca is not None else required_float_env("STRAT_DCA", env=cfg)
    disc = args.disc if args.disc is not None else required_float_env("STRAT_ENTRY_DISCOUNT_PCT", env=cfg)
    drop = args.drop if args.drop is not None else required_float_env("STRAT_DCA_DROP_PCT", env=cfg)
    tp = args.tp if args.tp is not None else required_float_env("STRAT_TAKEPROFIT_PCT", env=cfg)
    maxdca = args.maxdca if args.maxdca is not None else required_int_env("STRAT_MAX_DCA_BUYS", env=cfg)
    budget = args.budget if args.budget is not None else required_float_env("STRAT_MAX_BUDGET", env=cfg)
    fee = args.fee if args.fee is not None else required_float_env("HL_FEE_PCT", env=cfg)
    sl = args.sl if args.sl is not None else required_float_env("STRAT_STOP_LOSS_PCT", env=cfg)
    fast_h = (
        exact_pos_int(args.fast, default=None)
        if args.fast is not None
        else exact_pos_int(optional_env("SIGNAL_FAST_H", env=cfg), default=12)
    )
    slow_h = (
        exact_pos_int(args.slow, default=None)
        if args.slow is not None
        else exact_pos_int(optional_env("SIGNAL_SLOW_H", env=cfg), default=48)
    )
    if fast_h is None or slow_h is None or fast_h >= slow_h:
        ap.error("--fast and --slow must be positive integers with fast < slow")
    if args.band is not None:
        band_pct = safe_nonneg_float(args.band, default=None)
        if band_pct is None:
            ap.error("--band must be >= 0")
    else:
        raw_band = optional_env("SIGNAL_BAND_PCT", env=cfg)
        band_pct = safe_nonneg_float(raw_band, default=0.3) if raw_band is not None else 0.3
    win_h = (
        exact_pos_int(args.win, default=None)
        if args.win is not None
        else exact_pos_int(optional_env("ANALYSIS_WINDOW_H", env=cfg), default=24)
    )
    step_h = (
        exact_pos_int(args.step, default=None)
        if args.step is not None
        else exact_pos_int(optional_env("ANALYSIS_STEP_H", env=cfg), default=8)
    )
    if win_h is None or step_h is None:
        ap.error("--win and --step must be positive integers")
    ttl_min = safe_float(optional_env("STRAT_ORDER_TTL_MIN", env=cfg), 60.0)
    reentry_tol = safe_float(optional_env("STRAT_REENTRY_TOLERANCE_PCT", env=cfg), 0.0)
    currency = optional_env("STRAT_CURRENCY", "USDC", env=cfg).strip().upper() or "USDC"
    strat_mode = optional_env("STRATEGY_MODE", "avg_tp", env=cfg).strip().lower()
    enable_tp = strat_mode != "dca_only"

    if (budget := safe_pos_float(budget, default=None)) is None:
        ap.error("--budget must be > 0")
    if (entry := safe_pos_float(entry, default=None)) is None:
        ap.error("--entry must be > 0")
    if (maxdca := exact_nonneg_int(maxdca, default=None)) is None:
        ap.error("--maxdca must be a non-negative integer")
    if maxdca > 0:
        if (dca := safe_pos_float(dca, default=None)) is None:
            ap.error("--dca must be > 0 when maxdca > 0")
    elif (dca := safe_nonneg_float(dca, default=None)) is None:
        ap.error("--dca must be >= 0")
    if (drop := safe_pos_float(drop, default=None)) is None:
        ap.error("--drop must be > 0")
    if enable_tp:
        if (tp := safe_pos_float(tp, default=None)) is None:
            ap.error("--tp must be > 0 when take-profit is enabled")
    elif (tp := safe_nonneg_float(tp, default=None)) is None:
        ap.error("--tp must be >= 0")
    if (disc := safe_nonneg_float(disc, default=None)) is None or disc >= 100.0:
        ap.error("--disc must be in [0, 100)")
    if (fee := safe_nonneg_float(fee, default=None)) is None or fee >= 100.0:
        ap.error("--fee must be in [0, 100)")
    if (sl := safe_nonneg_float(sl, default=None)) is None:
        ap.error("--sl must be >= 0")

    warmup_h = max(slow_h + 6, math.ceil(win_h + step_h * 14), 48) if args.signal != "off" else 5
    records = fetch_ohlc_candles(coin, "1h", lookback_hours=days_i * 24 + warmup_h, drop_incomplete=True)
    if len(records) < 50:
        log("! too few candles"); return 1
    all_closes = [r["close"] for r in records]
    all_ts_sec = [r["timestamp"] for r in records]
    all_sig = precompute_signal(all_closes, all_ts_sec, args.signal, fast_h, slow_h, band_pct / 100.0, win_h, step_h)
    cutoff_ts = all_ts_sec[-1] - (days_i * 24 - 1) * 3600
    eval_start = bisect.bisect_left(all_ts_sec, cutoff_ts)
    if eval_start >= len(records):
        eval_start = 0
    eval_records = records[eval_start:]
    ohlc = [(r["open"], r["high"], r["low"], r["close"]) for r in eval_records]
    ts = [r["timestamp"] * 1000 for r in eval_records]
    sig = all_sig[eval_start:]
    closes = [x[3] for x in ohlc]
    fund = fetch_funding_history(coin, max(0, ts[0] - 3_600_000), end_ms=ts[-1])
    bh = buy_hold_pct(closes)

    base = dict(
        currency=currency,
        entry=entry,
        dca=dca,
        disc=disc,
        drop=drop,
        tp=tp,
        maxdca=maxdca,
        budget=budget,
        fee=fee,
        sl=sl,
        order_ttl_min=ttl_min if ttl_min is not None else 60.0,
        reentry_tolerance_pct=reentry_tol if reentry_tol is not None else 0.0,
        enable_takeprofit=enable_tp,
    )

    return run_cli_single_or_sweep_backtest(
        args.mode,
        base,
        lambda p: run_replay(ohlc, ts, fund, sig, p, direction=direction, coin=coin),
        budget=budget,
        bh=bh,
        single_header=f"=== COMPLETE BACKTEST {coin} {args.days}d {direction} signal={args.signal} ===",
        single_params_line=f"entry={entry} dca={dca} disc={disc}% drop={drop}% tp={tp}% sl={sl}% maxdca={maxdca}",
        sweep_header=f"=== SWEEP {coin} {args.days}d {direction} signal={args.signal} (buy&hold {bh:+.1f}%) ===",
        include_funding=True,
        open_qty_decimals=4,
        tp_values=(0.5, 0.8, 1.0, 1.5, 2.0, 3.0),
        drop_values=(1.0, 2.0, 3.0, 5.0),
        sl_values=None,
        emit=log,
    )


if __name__ == "__main__":
    sys.exit(main())
