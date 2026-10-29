"""Common financial metrics for strategy comparisons.

The functions are pure and read no data, config or runtime state. All returns are
computed from mark-to-market equity, not only from closed trades.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence

from botcore import safe_float, safe_nonneg_float, safe_pos_float


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
    for previous, current in zip(curve, curve[1:]):
        if previous <= 0:
            returns_valid = False
            break
        periodic_returns.append(current / previous - 1.0)

    annualized_return = sharpe = sortino = calmar = None
    if periods_per_year is not None:
        periods_per_year = safe_pos_float(periods_per_year, default=None)
        if periods_per_year is None:
            raise ValueError("periods_per_year must be positive")
    if periods_per_year and periodic_returns and returns_valid:
        periods = len(periodic_returns)
        growth = curve[-1] / initial_capital
        if growth > 0:
            try:
                annualized_return = safe_float(growth ** (periods_per_year / periods) - 1.0, default=None)
            except OverflowError:
                annualized_return = None

        mean_return = statistics.fmean(periodic_returns)
        if len(periodic_returns) > 1:
            n_ret = len(periodic_returns)
            volatility = math.sqrt(
                sum((r - mean_return) ** 2 for r in periodic_returns) / (n_ret - 1)
            )
            if volatility > 0:
                sharpe = mean_return / volatility * math.sqrt(periods_per_year)

        downside_deviation = math.sqrt(
            statistics.fmean(min(value, 0.0) ** 2 for value in periodic_returns)
        )
        if downside_deviation > 0:
            sortino = mean_return / downside_deviation * math.sqrt(periods_per_year)
        if annualized_return is not None and max_drawdown_pct > 0:
            calmar = annualized_return / max_drawdown_pct

    cvar = None
    if periodic_returns and returns_valid:
        tail_size = max(1, math.ceil((1.0 - cvar_confidence) * len(periodic_returns)))
        cvar = statistics.fmean(sorted(periodic_returns)[:tail_size])

    exposure_values = [] if exposure is None else list(exposure)
    exposure_pct = (
        100.0 * sum(bool(value) for value in exposure_values) / len(exposure_values)
        if exposure_values else 0.0
    )

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
        "win_rate_pct": 100.0 * winning_trades / len(pnls) if pnls else None,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else None,
        "expectancy": statistics.fmean(pnls) if pnls else None,
        "turnover_pct": turnover_notional / initial_capital * 100.0,
    }


def format_performance_summary(m: dict, indent: str = "  ") -> list[str]:
    """Format standardized RISC and EXECUTIE summary lines across backtest harnesses."""
    def metric(name: str, digits: int = 2) -> str:
        vf = safe_float(m.get(name) if isinstance(m, dict) else None, default=None)
        return f"{vf:.{digits}f}" if vf is not None else "n/a"

    lines = [
        (f"{indent}RISC: maxDD {metric('max_drawdown_pct')}%  "
         f"Sharpe {metric('sharpe')}  Sortino {metric('sortino')}  "
         f"Calmar {metric('calmar')}  CVaR95 {metric('cvar_95_pct')}%"),
        (f"{indent}EXECUTIE: expunere {metric('exposure_pct')}%  "
         f"turnover {metric('turnover_pct')}%  fills {m.get('fills', 'n/a') if isinstance(m, dict) else 'n/a'}  "
         f"profit-factor {metric('profit_factor')}  "
         f"expectancy ${metric('expectancy')}"),
    ]
    return lines


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

