#!/usr/bin/env python3
"""
backtest.py -- DCA + take-profit backtester for Kraken spot using Kraken OHLC data.

It replays the strategy bar by bar (market-discount entry, DCA on declines,
take-profit, and stop-loss), with Kraken fees (~0.25% per taker leg). The report
is honest: TOTAL includes the mark-to-market open position, not just realized profit.

  python3 backtest.py --pair HYPEUSD --interval 60
  python3 backtest.py --mode sweep --pair HYPEUSD --interval 240
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import dataclasses
from strategies.spot_engine import StratParams
from strategies.replay_metrics import (
    add_common_backtest_cli_args,
    create_backtest_cli_parser,
    parse_bt_cli_int,
    run_cli_single_or_sweep_backtest,
    validate_bt_cli_token,
)
from botcore import (
    add_store_true_cli_args,
    exact_nonneg_int,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
)
from strategies import spot_replay as rp
from kraken.kraken_common import load_isolated_kraken_cfg
from kraken.market_data import fetch_ohlc_candles


def _bt_params(base: dict, over: dict | None = None) -> "StratParams":
    """Build ``StratParams`` from live configuration plus backtest overrides.

    Live ``STRAT_*`` values preserve production re-entry, tranche, and tolerance
    behavior, so the faithful engine tests the production strategy rather than
    defaults.
    """
    if not isinstance(base, dict):
        raise ValueError("_bt_params requires a dict for base")
    env = load_isolated_kraken_cfg(_HERE)
    p = StratParams.from_env(env=env)

    def _pick(keys: tuple[str, ...], default, *, nonneg: bool = True, pos: bool = False):
        for k in keys:
            if k in base and base[k] is not None:
                val = (
                    safe_pos_float(base[k], default=None)
                    if pos
                    else (safe_nonneg_float(base[k], default=None) if nonneg else safe_float(base[k], default=None))
                )
                if val is None:
                    raise ValueError(f"invalid value not allowed for {k}: {base[k]!r}")
                return val
        return default

    raw_maxdca = base.get("maxdca") if base.get("maxdca") is not None else p.max_dca_buys
    maxdca_i = exact_nonneg_int(raw_maxdca, default=None)
    if maxdca_i is None:
        raise ValueError(f"invalid value not allowed for maxdca: {base.get('maxdca')!r}")
    fields = dict(
        entry_amount=_pick(("entry",), p.entry_amount, pos=True),
        dca_amount=_pick(("dca",), p.dca_amount, pos=True),
        entry_discount_pct=_pick(("disc",), p.entry_discount_pct),
        dca_drop_pct=_pick(("drop_fallback", "drop"), p.dca_drop_pct, pos=True),
        takeprofit_pct=_pick(("tp",), p.takeprofit_pct, pos=True),
        max_dca_buys=maxdca_i,
        max_budget=_pick(("budget",), p.max_budget, pos=True),
        stop_loss_pct=_pick(("sl",), p.stop_loss_pct),
        tp_trail_pct=_pick(("trail",), p.tp_trail_pct),
        reentry_drop_pct=_pick(("reentry_fallback", "reentry_drop", "reentry"), p.reentry_drop_pct),
    )
    if any(base.get(k) is not None for k in ("entry", "dca", "budget")):
        fields.update(total_budget=0.0, alloc_pct=0.0, entry_pct=0.0, dca_pct=0.0)
    if base.get("tp") is not None:
        fields.update(tp_tranches=[], tp_dynamic_flat=False)
    raw_interval = base.get("interval") if base.get("interval") is not None else 60
    interval = safe_pos_float(raw_interval, default=None)
    if interval is None:
        raise ValueError(f"invalid interval: {raw_interval!r}")
    if interval != safe_pos_float(p.trend_interval, interval):
        fields.update(
            trend_overlay=False,
            dca_trend_brake=False,
            reentry_hybrid_enabled=False,
            surge_guard=False,
            tp_dynamic_flat=False,
        )
    if interval != safe_pos_float(p.tp_trail_vol_interval, interval):
        fields.update(tp_trail_adaptive=False, surge_dynamic=False)
    if interval != safe_pos_float(p.dca_vol_interval, interval):
        fields.update(dca_vol_scale_k=0.0)
    if over:
        fields.update(over)
    return dataclasses.replace(p, **fields)


def fetch_candles(pair, interval):
    """Fetch completed Kraken candles as (open, high, low, close) tuples."""
    records = fetch_ohlc_candles(pair, interval, drop_incomplete=True)
    return [(r["open"], r["high"], r["low"], r["close"]) for r in records]


_DEFAULT_FETCH_CANDLES = fetch_candles


def simulate(ohlc, P, reentry_arr=None, sl_bounce_pct=None):
    """Run the DCA, take-profit, and stop-loss simulation using the production engine.

    Delegates to the faithful replay engine (rp.run_replay) which runs
    strategies.spot_engine.Strategy directly, eliminating drift between
    backtest and production.
    """
    over = {}
    if sl_bounce_pct is not None:
        over["reentry_sl_bounce_pct"] = sl_bounce_pct
    bar_min = P.get("interval") if P.get("interval") is not None else 60
    params = rp.scale_subbar_timeout_params(_bt_params(P, over if over else None), bar_min)
    res = rp.run_replay(
        ohlc,
        params,
        fee_pct=P.get("fee") if P.get("fee") is not None else 0.25,
        bar_minutes=bar_min,
        reentry_series=reentry_arr,
        timestamps=P.get("timestamps"),
    )
    res.setdefault("blocked_ticks", 0)
    return res


def main() -> int:
    ap = create_backtest_cli_parser("Backtester DCA+TP Kraken (OHLC).")
    ap.add_argument("--pair", default="HYPEUSD")
    ap.add_argument("--interval", type=parse_bt_cli_int, default=60, help="minutes: 60=1h, 240=4h, 1440=1d")
    add_common_backtest_cli_args(
        ap,
        defaults={
            "entry": 100, "dca": 50, "disc": 0.2, "drop": 2.0,
            "tp": 1.9, "maxdca": 10, "budget": 1000, "fee": 0.25, "sl": 10.0,
        },
    )
    args = add_store_true_cli_args(
        ap, ("--fast", "deprecated; faithful live engine rp.run_replay is always used")
    ).parse_args()

    try:
        pair_tok = validate_bt_cli_token(args.pair, "--pair", allow_slash=True).upper()
    except ValueError as exc:
        print(f"! {exc}"); return 1
    if args.interval <= 0:
        print("! interval must be > 0"); return 1
    if args.budget <= 0:
        print("! budget must be > 0"); return 1
    if args.entry <= 0 or args.dca <= 0:
        print("! entry and dca must be > 0"); return 1
    if args.maxdca < 0:
        print("! maxdca must be >= 0"); return 1
    if args.fee < 0 or args.sl < 0 or args.disc < 0:
        print("! fee/sl/disc must be >= 0"); return 1
    if args.fee >= 100.0 or args.sl >= 100.0 or args.disc >= 100.0:
        print("! fee/sl/disc must be < 100"); return 1
    if args.tp <= 0 or args.drop <= 0:
        print("! tp and drop must be > 0"); return 1
    base = dict(entry=args.entry, dca=args.dca, disc=args.disc, drop=args.drop,
                tp=args.tp, maxdca=args.maxdca, budget=args.budget, fee=args.fee,
                sl=args.sl, interval=args.interval)
    try:
        _bt_params(base)
    except (ValueError, KeyError, OSError) as exc:
        print(f"! config error: {exc}"); return 1
    try:
        if fetch_candles is _DEFAULT_FETCH_CANDLES:
            records = fetch_ohlc_candles(pair_tok, args.interval, drop_incomplete=True)
            ohlc = [(r["open"], r["high"], r["low"], r["close"]) for r in records]
            timestamps = [r["timestamp"] for r in records]
        else:
            ohlc = fetch_candles(pair_tok, args.interval)
            timestamps = None
    except Exception as e:  # noqa: BLE001
        print(f"! data error: {e}"); return 1

    engine = "live engine (faithful)"
    return run_cli_single_or_sweep_backtest(
        args.mode,
        base,
        lambda p: rp.run_replay(
            ohlc,
            rp.scale_subbar_timeout_params(_bt_params(p), args.interval),
            fee_pct=args.fee,
            bar_minutes=args.interval,
            timestamps=timestamps,
        ),
        budget=args.budget,
        ohlc=ohlc,
        single_header=f"=== BACKTEST KRAKEN {args.pair} interval={args.interval}m ({len(ohlc)} bars) [{engine}] ===",
        single_params_line=f"entry={args.entry} dca={args.dca} drop={args.drop}% tp={args.tp}% sl={args.sl}% fee={args.fee}%/leg",
        sweep_header=f"=== SWEEP {args.pair} interval={args.interval}m  (buy&hold {{bh:+.1f}}%) [{engine}] ===",
        open_qty_decimals=6,
        pre_sweep_fn=(
            lambda: rp.prewarm_trend_series(ohlc, warmup=0, params=_bt_params(base), bar_minutes=args.interval)
            if len(ohlc) > 40 and hasattr(rp, "prewarm_trend_series")
            else None
        ),
    )


if __name__ == "__main__":
    sys.exit(main())
