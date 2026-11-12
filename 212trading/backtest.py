#!/usr/bin/env python3
"""
backtest.py -- Trading 212 stock DCA and take-profit backtester using Yahoo data.

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

from strategies.replay_metrics import (
    add_common_backtest_cli_args,
    create_backtest_cli_parser,
    parse_bt_cli_float,
    print_bt_cli_error,
    resolve_cli_ohlc_and_timestamps,
    run_cli_single_or_sweep_backtest,
    validate_bt_cli_token,
    validate_common_backtest_cli_bounds,
)
from strategies.spot_replay import scale_subbar_timeout_params
from botcore import (
    _parse_bool_flag,
    float_env,
    load_local_module,
    required_env,
    required_float_env,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
)
from strategies.base_spot_mixin import parse_max_dca_buys_env

_replay_mod = load_local_module(T212_DIR, "replay", module_key="t212_backtest_replay")
run_replay = _replay_mod.run_replay
StratParams = _replay_mod._strat.StratParams
_md_mod = _replay_mod._md
_ipo_common_mod = load_local_module(T212_DIR, "ipo_common", module_key="t212_ipo_common")
_load_isolated_t212_cfg = _ipo_common_mod.load_isolated_t212_cfg


def fetch_candles(sym, rng, interval):
    """Fetch completed Yahoo candles as (open, high, low, close) tuples."""
    records = _md_mod.fetch_yahoo_candles(sym, rng, interval, drop_incomplete=True)
    return [(r["open"], r["high"], r["low"], r["close"]) for r in records]


_DEFAULT_FETCH_CANDLES = fetch_candles


_SIMULATE_ENV_DEFAULTS = {
    "STRAT_FX_FEE_PCT": "0.15", "STRAT_STOP_LOSS_PCT": "0.0",
    "STRAT_CHECK_MINUTES": "1", "STRAT_ORDER_TTL_MIN": "10",
    "STRAT_REENTRY_DROP_PCT": "0", "STRAT_REENTRY_TOLERANCE_PCT": "0.05",
    "STRAT_LOSS_ALERT_STEP": "1", "STRAT_LADDER_MIN_FREE": "6",
    "STRAT_SL_REBUY_ENABLED": "false", "STRAT_SL_REBUY_BOUNCE_PCT": "1.2",
    "STRAT_DCA_TREND_GATE_PCT": "0", "STRAT_TRAIL_PCT": "0",
    "STRAT_TRAIL_MIN_PROFIT_PCT": "5", "STRAT_SL_MHDSL_ENABLED": "false",
    "STRAT_TP_MDHTP_ENABLED": "false", "STRAT_REENTRY_HYBRID_ENABLED": "false",
    "STRAT_REENTRY_PEAK_RELATIVE": "false", "STRAT_REENTRY_PULLBACK_PCT": "1.5",
    "STRAT_REENTRY_PULLBACK_ADAPTIVE": "false", "STRAT_REENTRY_PULLBACK_K": "1.0",
    "STRAT_REENTRY_PULLBACK_MIN": "0.8", "STRAT_REENTRY_PULLBACK_MAX": "4.0",
    "STRAT_REENTRY_TTL_HOURS": "0.0", "STRAT_SL_DISASTER_BUFFER_RATIO": "0.30",
    "STRAT_SL_DISASTER_BUFFER_MAX_PCT": "15.0", "STRAT_SL_TREND_DEFER_MAX_SEC": "220.0",
    "STRAT_SL_BOUNCE_MIN_RECOVERY_PCT": "0.40", "STRAT_SL_BOUNCE_PULLBACK_PCT": "0.30",
    "STRAT_SL_STRUCTURAL_BREAK_ENABLED": "true", "STRAT_TP_STAGNATION_MAX_SEC": "1080.0",
    "STRAT_TP_LADDER": "", "STRAT_TP_LADDER_CURVE": "linear",
}
_SIMULATE_ALIAS_MAP = (
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
)


def simulate(ohlc, P):
    """CLI compatibility wrapper over the same ``Strategy.step`` used live."""
    if not isinstance(P, dict):
        raise ValueError("simulate requires a parameter dict P")

    def _p_val(keys: tuple[str, ...], default: object) -> object:
        for k in keys:
            if k in P and P[k] is not None:
                return P[k]
        return default

    base_env = P.get("_base_env") if isinstance(P.get("_base_env"), dict) else {}
    budget = safe_pos_float(_p_val(("budget", "max_budget"), 500.0), default=None)
    if budget is None:
        raise ValueError("budget must be finite and positive")
    explicit_entry = _p_val(("entry", "entry_amount"), None)
    if explicit_entry is not None and safe_pos_float(explicit_entry, default=None) is None:
        raise ValueError("entry must be finite and positive")
    explicit_dca = _p_val(("dca", "dca_amount"), None)
    if explicit_dca is not None and safe_pos_float(explicit_dca, default=None) is None:
        raise ValueError("dca must be finite and positive")
    raw_entry_pct = P.get("entry_pct")
    if raw_entry_pct is not None and safe_pos_float(raw_entry_pct, default=None) is None:
        raise ValueError("entry_pct must be finite and positive")
    raw_dca_pct = P.get("dca_pct")
    if raw_dca_pct is not None and safe_pos_float(raw_dca_pct, default=None) is None:
        raise ValueError("dca_pct must be finite and positive")
    raw_entry = safe_pos_float(explicit_entry, default=None)
    raw_dca = safe_pos_float(explicit_dca, default=None)
    entry_pct = safe_float(
        raw_entry_pct,
        default=((raw_entry / budget * 100.0) if budget > 0 and raw_entry is not None else 10.0),
    )
    dca_pct = safe_float(
        raw_dca_pct,
        default=((raw_dca / budget * 100.0) if budget > 0 and raw_dca is not None else 10.0),
    )

    env_overrides = {
        k: str(base_env.get(k) or v) if k in ("STRAT_CHECK_MINUTES", "STRAT_ORDER_TTL_MIN") else str(base_env.get(k, v))
        for k, v in _SIMULATE_ENV_DEFAULTS.items()
    }
    env_overrides.update({
        "STRATEGY_MODE": str(_p_val(("mode",), base_env.get("STRATEGY_MODE") or "avg_tp")),
        "STRAT_CURRENCY": str(_p_val(("currency",), base_env.get("STRAT_CURRENCY") or "USD")),
        "YAHOO_SYMBOL": str(_p_val(("sym", "yahoo_sym"), base_env.get("YAHOO_SYMBOL") or "REPLAY")),
        "STRAT_ENTRY_PCT": str(entry_pct),
        "STRAT_DCA_PCT": str(dca_pct),
        "STRAT_ENTRY_DISCOUNT_PCT": str(_p_val(("disc", "entry_discount_pct"), 0.3)),
        "STRAT_DCA_DROP_PCT": str(_p_val(("drop", "dca_drop_pct"), 2.0)),
        "STRAT_TAKEPROFIT_PCT": str(_p_val(("tp", "takeprofit_pct"), 1.5)),
        "STRAT_MAX_DCA_BUYS": str(
            _p_val(("maxdca", "max_dca_buys"), base_env.get("STRAT_MAX_DCA_BUYS") or "auto")
        ),
        "STRAT_MAX_BUDGET": str(budget),
    })
    bar_min = P.get("bar_minutes")
    if bar_min is not None and safe_pos_float(bar_min, default=None) != 5.0 and "dca_trend_gate" not in P:
        env_overrides["STRAT_DCA_TREND_GATE_PCT"] = "0"
    for alias, env_key in _SIMULATE_ALIAS_MAP:
        val = _p_val(alias, None)
        if val is not None:
            env_overrides[env_key] = str(val)
    for k, v in P.items():
        if k == "_base_env":
            continue
        if k == "tp_ladder":
            if v is None:
                env_overrides["STRAT_TP_LADDER"] = ""
            elif isinstance(v, (list, tuple)):
                all_fracs = (
                    bool(v)
                    and all(safe_nonneg_float(frac, 2.0) <= 1.0 + 1e-9 for _, frac in v)
                    and sum(safe_nonneg_float(frac, 0.0) for _, frac in v) <= 1.0 + 1e-6
                )
                env_overrides["STRAT_TP_LADDER"] = ",".join(
                    f"{lvl}:{frac * 100.0 if all_fracs else frac}" for lvl, frac in v
                )
            else:
                env_overrides["STRAT_TP_LADDER"] = str(v)
        elif v is None or k in ("reentry_drop", "reentry_tol", "sl_rebuy_bounce"):
            continue
        elif k.startswith(("STRAT_", "sl_", "tp_", "reentry_")):
            env_key = k if k.startswith("STRAT_") else f"STRAT_{k.upper()}"
            env_overrides[env_key] = str(v).lower() if isinstance(v, bool) else str(v)

    params = StratParams.from_env(env_overrides)
    if bar_min is not None:
        params = scale_subbar_timeout_params(params, bar_min)
    fx_arg = P.get("fx_to_usd")
    ts_arg = P.get("timestamps")
    return run_replay(
        ohlc,
        params,
        bar_minutes=bar_min,
        fx_to_usd=1.0 if fx_arg is None else fx_arg,
        timestamps=ts_arg if ts_arg is not None else (),
    )



def interval_minutes(value: str) -> int | None:
    try:
        return _md_mod.parse_interval_minutes(value)
    except (KeyError, ValueError, IndexError):
        return None


def main() -> int:
    ap = create_backtest_cli_parser("Backtester DCA+TP T212 (Yahoo).")
    ap.add_argument("--profile", "-p", default=None, help="config profile to load (e.g. nvda, spcx, rgnt)")
    ap.add_argument("--sym", default=None)
    ap.add_argument("--range", default="1y", help="ex: 1mo,3mo,6mo,1y,2y")
    ap.add_argument("--interval", default="1d", help="ex: 1h,1d")
    add_common_backtest_cli_args(ap)
    ap.add_argument("--mhdsl", action=argparse.BooleanOptionalAction, default=None, help="enable/disable MHDSL stop loss")
    ap.add_argument("--mdhtp", action=argparse.BooleanOptionalAction, default=None, help="enable/disable MDHTP take profit")
    ap.add_argument("--trail", type=parse_bt_cli_float, default=None, help="trailing percent")
    ap.add_argument("--trail-min-profit", type=parse_bt_cli_float, default=None, help="min profit %% to arm trailing")
    args = ap.parse_args()

    try:
        profile = validate_bt_cli_token(args.profile, "--profile").lower() if args.profile is not None else "nvda"
        range_tok = validate_bt_cli_token(args.range, "--range")
        interval_tok = validate_bt_cli_token(args.interval, "--interval")
    except ValueError as exc:
        return print_bt_cli_error(exc)
    cfg_file = os.path.join(T212_DIR, f"config.{profile}.env")
    if not os.path.exists(cfg_file):
        return print_bt_cli_error(f"profile config not found: {cfg_file}")
    try:
        cfg = _load_isolated_t212_cfg(os.path.join(T212_DIR, ".env"), profile_file=cfg_file)
        budget = args.budget if args.budget is not None else required_float_env("STRAT_MAX_BUDGET", cfg)
        entry_pct = float_env("STRAT_ENTRY_PCT", cfg)
        dca_pct = float_env("STRAT_DCA_PCT", cfg)
        entry = args.entry if args.entry is not None else ((budget * entry_pct / 100.0) if entry_pct is not None else required_float_env("STRAT_ENTRY", cfg))
        dca = args.dca if args.dca is not None else ((budget * dca_pct / 100.0) if dca_pct is not None else required_float_env("STRAT_DCA", cfg))
        disc = args.disc if args.disc is not None else required_float_env("STRAT_ENTRY_DISCOUNT_PCT", cfg)
        drop = args.drop if args.drop is not None else required_float_env("STRAT_DCA_DROP_PCT", cfg)
        tp = args.tp if args.tp is not None else required_float_env("STRAT_TAKEPROFIT_PCT", cfg)
        sl = args.sl if args.sl is not None else required_float_env("STRAT_STOP_LOSS_PCT", cfg)
        fee = args.fee if args.fee is not None else required_float_env("STRAT_FX_FEE_PCT", cfg)
        raw_sym = args.sym if args.sym is not None else (cfg.get("YAHOO_SYMBOL") or required_env("YAHOO_SYMBOL", cfg))
        sym = validate_bt_cli_token(raw_sym, "--sym")
        if args.maxdca is not None:
            maxdca = args.maxdca
        else:
            maxdca, entry = parse_max_dca_buys_env(cfg, budget, entry, dca)
    except (ValueError, KeyError, OSError) as exc:
        return print_bt_cli_error(f"config error: {exc}")

    sl_mhdsl = args.mhdsl if args.mhdsl is not None else (
        _parse_bool_flag(cfg.get("STRAT_SL_MHDSL_ENABLED"), default=False)
    )
    tp_mdhtp = args.mdhtp if args.mdhtp is not None else (
        _parse_bool_flag(cfg.get("STRAT_TP_MDHTP_ENABLED"), default=False)
    )
    trail_pct = args.trail if args.trail is not None else (float_env("STRAT_TRAIL_PCT", cfg) or 0.0)
    trail_min_profit = args.trail_min_profit if args.trail_min_profit is not None else (float_env("STRAT_TRAIL_MIN_PROFIT_PCT", cfg) or 0.0)

    bar_min = interval_minutes(interval_tok)
    if bar_min is None:
        return print_bt_cli_error(f"unsupported interval: {interval_tok}")
    if (
        bounds_err := validate_common_backtest_cli_bounds(
            budget=budget,
            entry=entry,
            dca=dca,
            maxdca=maxdca,
            fee=fee,
            sl=sl,
            disc=disc,
            tp=tp,
            drop=drop,
            pct_label="fee/sl/disc/trail parameters",
            extra_nonneg=(trail_pct, trail_min_profit),
            extra_pct_lt_100=(trail_pct,),
        )
    ) is not None:
        return print_bt_cli_error(bounds_err)
    try:
        ohlc, timestamps = resolve_cli_ohlc_and_timestamps(
            fetch_candles,
            _DEFAULT_FETCH_CANDLES,
            lambda: _md_mod.fetch_yahoo_candles(sym, range_tok, interval_tok, drop_incomplete=True),
            sym,
            range_tok,
            interval_tok,
        )
    except Exception as exc:  # noqa: BLE001
        return print_bt_cli_error(f"data error: {exc}")
    base = dict(entry=entry, dca=dca, disc=disc, drop=drop,
                tp=tp, maxdca=maxdca, budget=budget, fee=fee, sl=sl,
                sym=sym, bar_minutes=bar_min,
                sl_mhdsl_enabled=sl_mhdsl, tp_mdhtp_enabled=tp_mdhtp,
                trail_pct=trail_pct, trail_min_profit=trail_min_profit,
                timestamps=timestamps,
                _base_env=cfg)
    if args.tp is not None or args.mode == "sweep":
        base["tp_ladder"] = ""
        if args.mdhtp is None:
            base["tp_mdhtp_enabled"] = False
            tp_mdhtp = False

    prof_info = f" [profile={args.profile}]" if args.profile else ""
    return run_cli_single_or_sweep_backtest(
        args.mode,
        base,
        lambda p: simulate(ohlc, p),
        budget=budget,
        ohlc=ohlc,
        single_header=f"=== BACKTEST T212 {sym} {args.range}/{args.interval} ({len(ohlc)} bars){prof_info} ===",
        single_params_line=f"entry={entry:.1f} dca={dca:.1f} drop={drop}% tp={tp}% sl={sl}% fee={fee}%/leg | MHDSL={'ON' if sl_mhdsl else 'OFF'} MDHTP={'ON' if tp_mdhtp else 'OFF'}",
        sweep_header=f"=== SWEEP {sym} {args.range}/{args.interval}  (buy&hold {{bh:+.1f}}%) ===",
        open_qty_decimals=4,
    )


if __name__ == "__main__":
    sys.exit(main())
