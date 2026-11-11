"""Canonical pure financial metrics and CLI reporting helpers for strategy replay and backtests.

All functions are pure and read no data, config, or runtime state. Returns are computed
from mark-to-market equity curves rather than closed trades alone.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal, ROUND_CEILING
from typing import Any

from botcore import _parse_bool_flag, exact_int, safe_float, safe_nonneg_float, safe_nonneg_int, safe_pos_float


def round_metric(value: object, decimals: int = 10) -> object:
    """Round a finite numeric metric to ``decimals`` places, returning ``value`` unchanged otherwise."""
    vf = safe_float(value, default=None) if isinstance(value, (int, float)) else None
    return round(vf, decimals) if vf is not None else value


def calculate_performance_metrics(
    equity_curve: Sequence[float],
    *,
    initial_capital: float,
    periods_per_year: float | None = None,
    exposure: Iterable[bool] | None = None,
    trade_pnls: Iterable[float] | None = None,
    turnover_notional: float = 0.0,
    cvar_confidence: float = 0.95,
) -> dict:
    """Compute comparable metrics for an equity curve.

    ``equity_curve`` must include the initial capital as its first point.
    Sharpe/Sortino/Calmar are ``None`` without the real bar frequency; that choice
    prevents falsely comparing 1h series with 4h/1d ones.
    """
    initial_capital = safe_pos_float(initial_capital, default=None)
    if initial_capital is None:
        raise ValueError("initial_capital must be positive")
    cvar_confidence = safe_pos_float(cvar_confidence, default=None)
    if cvar_confidence is None or cvar_confidence >= 1:
        raise ValueError("cvar_confidence must be between 0 and 1")
    turnover_notional = safe_nonneg_float(turnover_notional, default=None)
    if turnover_notional is None:
        raise ValueError("turnover_notional must be non-negative")

    curve: list[float] = []
    if equity_curve is None or isinstance(equity_curve, (str, bytes, bool)):
        raise ValueError("equity_curve cannot be empty")
    for value in equity_curve:
        if isinstance(value, bool) or value is None:
            raise ValueError("equity_curve cannot contain booleans")
        fval = safe_float(value, default=None)
        if fval is None:
            raise ValueError("equity_curve must contain finite numbers")
        curve.append(fval)
    if not curve:
        raise ValueError("equity_curve cannot be empty")
    if not math.isclose(curve[0], initial_capital, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("the first point of equity_curve must be initial_capital")

    peak = curve[0]
    peak_index = 0
    max_drawdown_abs = 0.0
    max_drawdown_pct = 0.0
    max_underwater_periods = 0
    for index, equity in enumerate(curve[1:], start=1):
        if equity >= peak:
            peak = equity
            peak_index = index
            continue
        drawdown_abs = peak - equity
        drawdown_pct = drawdown_abs / peak if peak > 0 else 0.0
        max_drawdown_abs = max(max_drawdown_abs, drawdown_abs)
        max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)
        max_underwater_periods = max(max_underwater_periods, index - peak_index)

    periodic_returns = []
    returns_valid = True
    ruined = False
    for previous, current in zip(curve, curve[1:]):
        if ruined:
            if current > 0:
                returns_valid = False
                break
            periodic_returns.append(0.0)
            continue
        if previous <= 0:
            returns_valid = False
            break
        if current <= 0:
            periodic_returns.append(-1.0)
            ruined = True
        else:
            periodic_returns.append(current / previous - 1.0)

    annualized_return = sharpe = sortino = calmar = None
    if periods_per_year is not None:
        periods_per_year = safe_pos_float(periods_per_year, default=None)
        if periods_per_year is None:
            raise ValueError("periods_per_year must be positive")
    if periods_per_year and periodic_returns and returns_valid:
        periods = len(periodic_returns)
        growth = curve[-1] / initial_capital
        if growth <= 0:
            annualized_return = -1.0
        else:
            try:
                annualized_return = safe_float(growth ** (periods_per_year / periods) - 1.0, default=None)
            except OverflowError:
                annualized_return = None

        mean_return = statistics.fmean(periodic_returns)
        if len(periodic_returns) > 1:
            noise_floor = max(1e-12, abs(mean_return) * 1e-12)
            volatility = statistics.stdev(periodic_returns, xbar=mean_return)
            if volatility > noise_floor:
                sharpe = safe_float(mean_return / volatility * math.sqrt(periods_per_year), default=None)

            downside_deviation = math.sqrt(
                statistics.fmean(min(value, 0.0) ** 2 for value in periodic_returns)
            )
            if downside_deviation > noise_floor:
                sortino = safe_float(mean_return / downside_deviation * math.sqrt(periods_per_year), default=None)
        if annualized_return is not None and max_drawdown_pct > 0:
            calmar = safe_float(annualized_return / max_drawdown_pct, default=None)

    cvar = None
    if periodic_returns and returns_valid:
        raw_tail = (Decimal("1") - Decimal(str(cvar_confidence))) * Decimal(len(periodic_returns))
        ceil_tail = exact_int(raw_tail.to_integral_value(rounding=ROUND_CEILING), default=1) or 1
        tail_size = max(1, min(len(periodic_returns), ceil_tail))
        cvar = statistics.fmean(sorted(periodic_returns)[:tail_size])

    if exposure is not None and isinstance(exposure, (str, bytes, bool)):
        raise ValueError("exposure must be an iterable of booleans")
    exposure_flags: list[bool] = []
    for value in ([] if exposure is None else exposure):
        if isinstance(value, bool):
            exposure_flags.append(value)
        elif isinstance(value, str):
            parsed_exp = _parse_bool_flag(value, default=None)
            if parsed_exp is None:
                raise ValueError(f"invalid exposure flag: {value!r}")
            exposure_flags.append(parsed_exp)
        else:
            f_exp = safe_float(value, default=None)
            if f_exp is None:
                raise ValueError(f"invalid exposure value: {value!r}")
            exposure_flags.append(f_exp != 0.0)
    exposure_pct = (
        100.0 * statistics.fmean(exposure_flags)
        if exposure_flags else 0.0
    )

    if trade_pnls is not None and isinstance(trade_pnls, (str, bytes, bool)):
        raise ValueError("trade_pnls must be an iterable of finite numbers")
    pnls: list[float] = []
    for value in ([] if trade_pnls is None else trade_pnls):
        if isinstance(value, bool):
            raise ValueError("trade_pnls cannot contain booleans")
        fval = safe_float(value, default=None)
        if fval is None:
            raise ValueError("trade_pnls must contain finite numbers")
        pnls.append(fval)
    gross_profit = sum(value for value in pnls if value > 0)
    gross_loss = -sum(value for value in pnls if value < 0)
    winning_trades = sum(value > 0 for value in pnls)

    return {
        "net_pnl": curve[-1] - initial_capital,
        "return_pct": (curve[-1] / initial_capital - 1.0) * 100.0,
        "annualized_return_pct": (
            annualized_return * 100.0 if annualized_return is not None else None
        ),
        "max_drawdown_abs": max_drawdown_abs,
        "max_drawdown_pct": max_drawdown_pct * 100.0,
        "max_underwater_periods": max_underwater_periods,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "cvar_95_pct": cvar * 100.0 if cvar is not None else None,
        "exposure_pct": exposure_pct,
        "trade_count": len(pnls),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "win_rate_pct": 100.0 * winning_trades / len(pnls) if pnls else None,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else None,
        "expectancy": statistics.fmean(pnls) if pnls else None,
        "turnover_pct": turnover_notional / initial_capital * 100.0,
    }


def format_performance_summary(m: dict, indent: str = "  ") -> list[str]:
    """Format standardized RISK and EXECUTION summary lines across backtest harnesses."""
    def metric(name: str, digits: int = 2) -> str:
        vf = safe_float(m.get(name) if isinstance(m, dict) else None, default=None)
        return f"{vf:.{digits}f}" if vf is not None else "n/a"

    lines = [
        (f"{indent}RISK: maxDD {metric('max_drawdown_pct')}%  "
         f"Sharpe {metric('sharpe')}  Sortino {metric('sortino')}  "
         f"Calmar {metric('calmar')}  CVaR95 {metric('cvar_95_pct')}%"),
        (f"{indent}EXECUTION: exposure {metric('exposure_pct')}%  "
         f"turnover {metric('turnover_pct')}%  fills {m.get('fills', 'n/a') if isinstance(m, dict) else 'n/a'}  "
         f"profit-factor {metric('profit_factor')}  "
         f"expectancy ${metric('expectancy')}"),
    ]
    return lines


def buy_hold_pct(ohlc_or_closes: Sequence) -> float:
    """Compute percentage buy-and-hold return across an OHLC bar sequence or close series."""
    if not ohlc_or_closes or isinstance(ohlc_or_closes, (str, bytes, bool)):
        return 0.0
    first, last = ohlc_or_closes[0], ohlc_or_closes[-1]
    if isinstance(first, (list, tuple)) or isinstance(last, (list, tuple)):
        if not (isinstance(first, (list, tuple)) and isinstance(last, (list, tuple))):
            return 0.0
        if len(first) < 4 or len(last) < 4:
            return 0.0
        c0 = safe_pos_float(first[3], default=None)
        c1 = safe_nonneg_float(last[3], default=None)
    else:
        c0 = safe_pos_float(first, default=None)
        c1 = safe_nonneg_float(last, default=None)
    if c0 is None or c1 is None:
        return 0.0
    return (c1 - c0) / c0 * 100.0


def add_common_backtest_cli_args(
    ap: Any,
    *,
    defaults: Mapping[str, Any] | None = None,
) -> None:
    """Register the shared --entry/--dca/--disc/--drop/--tp/--maxdca/--budget/--fee/--sl CLI flags."""
    d = defaults or {}
    for flag in ("entry", "dca", "disc", "drop", "tp", "budget", "fee"):
        ap.add_argument(f"--{flag}", type=float, default=d.get(flag))
    ap.add_argument("--maxdca", type=int, default=d.get("maxdca"))
    ap.add_argument("--sl", type=float, default=d.get("sl"), help="stop-loss %% (0=disabled)")


def print_single_backtest_report(
    header: str,
    params_line: str,
    m: dict,
    *,
    budget: float,
    bh: float,
    include_funding: bool = False,
    open_qty_decimals: int = 4,
    emit=print,
) -> None:
    """Emit standardized single-run CLI backtest report across Kraken, Hyperliquid, and T212."""
    b_val = safe_pos_float(budget, default=None)
    if b_val is None:
        raise ValueError("budget must be positive")
    md = m if isinstance(m, dict) else {}
    bh_f = safe_float(bh, 0.0)
    qty_dec = safe_nonneg_int(open_qty_decimals, default=4)
    tot = safe_float(md.get("total"), 0.0) / b_val * 100.0
    net_pct = safe_float(md.get("net"), 0.0) / b_val * 100.0
    cycles = safe_nonneg_int(md.get("cycles"), default=0)
    wins = safe_nonneg_int(md.get("wins"), default=0)
    wr = 100.0 * wins / cycles if cycles else 0.0
    realized = safe_float(md.get("realized"), 0.0)
    upnl = safe_float(md.get("final_upnl"), 0.0)
    fees = safe_float(md.get("fees"), 0.0)
    maxdd = safe_float(md.get("maxdd"), 0.0)
    open_qty = safe_float(md.get("open_qty"), 0.0)
    emit(header)
    emit(f"  params: {params_line}")
    if include_funding:
        funding = safe_float(md.get("funding"), 0.0)
        emit(
            f"  REAL TOTAL: {tot:+.2f}% of the budget  <= realised ${realized:+.2f} "
            f"+ funding ${funding:+.2f} + the open position ${upnl:+.2f} - fees ${fees:.2f}"
        )
        emit(f"  (the realised part alone: {net_pct:+.2f}% -- MISLEADING if a losing position remains)")
        emit(
            f"  cycles  : {cycles}  win rate {wr:.0f}%   max drawdown ${maxdd:.2f}  "
            f"({maxdd / b_val * 100.0:.0f}% of the budget)"
        )
    else:
        emit(
            f"  REAL TOTAL: {tot:+.2f}% of the budget  <= realised ${realized:+.2f} "
            f"+ the open position ${upnl:+.2f} - fees ${fees:.2f}"
        )
        emit(f"  (realized alone: {net_pct:+.2f}%)")
        emit(f"  cycles: {cycles}  win-rate {wr:.0f}%   max drawdown ${maxdd:.2f}")
    for line in format_performance_summary(md, indent="  "):
        emit(line)
    if include_funding:
        status = "LOSING" if upnl < -1 else "ok"
        emit(f"  buy&hold: {bh_f:+.2f}%   final position: {open_qty:.{qty_dec}f} ({status})")
    else:
        emit(f"  buy&hold: {bh_f:+.2f}%   final position: {open_qty:.{qty_dec}f}")


def run_tp_drop_sl_sweep(
    base: dict,
    eval_fn,
    *,
    budget: float,
    tp_values: Sequence[float] = (1.0, 1.5, 2.0, 3.0, 5.0),
    drop_values: Sequence[float] = (1.0, 2.0, 3.0, 5.0),
    sl_values: Sequence[float] | None = (8.0, 15.0),
    top_n: int = 8,
    emit=print,
) -> list[tuple]:
    """Run a TP x DROP (x SL) parameter sweep and print the top N configurations."""
    b_val = safe_pos_float(budget, default=None)
    if b_val is None:
        raise ValueError("budget must be positive")
    limit_n = safe_nonneg_int(top_n, default=8)
    rows: list[tuple] = []
    if sl_values is None:
        for tp_val in tp_values:
            for drop_val in drop_values:
                p = dict(base, tp=tp_val, drop=drop_val)
                m = eval_fn(p)
                tot = safe_float(m.get("total"), 0.0) / b_val * 100.0
                cyc = safe_nonneg_int(m.get("cycles"), default=0)
                wins = safe_nonneg_int(m.get("wins"), default=0)
                wr = 100.0 * wins / cyc if cyc else 0.0
                rows.append((tot, tp_val, drop_val, cyc, wr, safe_float(m.get("maxdd"), 0.0)))
        rows.sort(key=lambda r: (r[0], -r[5], r[4], r[3], r[1], r[2]), reverse=True)
        emit(f"  top {limit_n} combinations (net% | tp | drop | cycles | win% | maxDD$):")
        for net, tp_v, drop_v, cyc, wr, dd in rows[:limit_n]:
            emit(f"    {net:+7.2f}%  tp={tp_v:<4} drop={drop_v:<4} cyc={cyc:<3} win={wr:.0f}% dd=${dd:.0f}")
        return rows

    for tp_val in tp_values:
        for drop_val in drop_values:
            for sl_val in sl_values:
                p = dict(base, tp=tp_val, drop=drop_val, sl=sl_val)
                m = eval_fn(p)
                tot = safe_float(m.get("total"), 0.0) / b_val * 100.0
                rows.append((tot, tp_val, drop_val, sl_val, safe_nonneg_int(m.get("cycles"), default=0), safe_float(m.get("maxdd"), 0.0)))
    rows.sort(key=lambda r: (r[0], -r[5], r[4], r[1], r[2], r[3]), reverse=True)
    emit(f"  top {limit_n} (total% | tp | drop | sl | cycles | maxDD$):")
    for tot, tp_v, drop_v, sl_v, cyc, dd in rows[:limit_n]:
        emit(f"    {tot:+7.2f}%  tp={tp_v:<4} drop={drop_v:<4} sl={sl_v:<4} cyc={cyc:<3} dd=${dd:.0f}")
    return rows


def run_cli_single_or_sweep_backtest(
    mode: str,
    base: dict,
    eval_fn,
    *,
    budget: float,
    bh: float | None = None,
    ohlc: Sequence | None = None,
    min_bars: int = 20,
    single_header: str,
    single_params_line: str,
    sweep_header: str,
    include_funding: bool = False,
    open_qty_decimals: int = 4,
    pre_sweep_fn=None,
    tp_values: Sequence[float] = (1.0, 1.5, 2.0, 3.0, 5.0),
    drop_values: Sequence[float] = (1.0, 2.0, 3.0, 5.0),
    sl_values: Sequence[float] | None = (8.0, 15.0),
    emit=print,
) -> int:
    """Execute either a single CLI backtest report or a TPxDROP(xSL) parameter sweep."""
    if ohlc is not None and len(ohlc) < min_bars:
        emit(f"not enough bars ({len(ohlc)}); aborting")
        return 1
    bh_val = float(bh) if bh is not None else (buy_hold_pct(ohlc) if ohlc is not None else 0.0)
    if mode == "single":
        m = eval_fn(base)
        print_single_backtest_report(
            single_header,
            single_params_line,
            m,
            budget=budget,
            bh=bh_val,
            include_funding=include_funding,
            open_qty_decimals=open_qty_decimals,
            emit=emit,
        )
        return 0
    emit(sweep_header.format(bh=bh_val) if "{bh" in sweep_header else sweep_header)
    if callable(pre_sweep_fn):
        pre_sweep_fn()
    run_tp_drop_sl_sweep(
        base,
        eval_fn,
        budget=budget,
        tp_values=tp_values,
        drop_values=drop_values,
        sl_values=sl_values,
        emit=emit,
    )
    return 0


def validate_replay_metrics(result: dict) -> None:
    """Assert that a replay result conforms to the canonical metrics schema."""
    if not isinstance(result, dict):
        raise AssertionError(f"Replay result must be a dictionary, got {type(result).__name__}")
    required_keys = (
        "return_pct",
        "max_drawdown_pct",
        "cycles",
        "fills",
        "initial_capital",
        "account_currency",
        "net_pnl",
        "realized",
        "total",
        "fees",
    )
    for key in required_keys:
        if key not in result:
            raise AssertionError(f"Missing required metric key: {key}")
        val = result[key]
        if key == "account_currency":
            if not isinstance(val, str) or not val:
                raise AssertionError(f"account_currency must be a non-empty string, got {val!r}")
        elif key in ("cycles", "fills"):
            if isinstance(val, bool) or not isinstance(val, int) or val < 0:
                raise AssertionError(f"{key} must be a non-negative int, got {val!r}")
        else:
            if safe_float(val, default=None) is None:
                raise AssertionError(f"{key} must be a finite number, got {val!r}")
    if "ambiguous_bars" in result:
        val = result["ambiguous_bars"]
        if isinstance(val, bool) or not isinstance(val, int) or val < 0:
            raise AssertionError(f"ambiguous_bars must be a non-negative int, got {val!r}")


__all__ = [
    "add_common_backtest_cli_args",
    "buy_hold_pct",
    "calculate_performance_metrics",
    "format_performance_summary",
    "print_single_backtest_report",
    "round_metric",
    "run_cli_single_or_sweep_backtest",
    "run_tp_drop_sl_sweep",
    "validate_replay_metrics",
]
