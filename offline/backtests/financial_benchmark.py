"""Scenarios and aggregations for reproducible financial benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import statistics
from typing import Iterable

from offline.backtests.execution import ExecutionModel, FeeModel
from offline.backtests.walk_forward import summarize_test_windows


@dataclass(frozen=True)
class BenchmarkScenario:
    """The cost/fill assumptions of a scenario, without strategy parameters."""

    name: str
    description: str
    fees: FeeModel
    execution: ExecutionModel
    calibrated: bool = False

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "fees": asdict(self.fees),
            "execution": asdict(self.execution),
            "calibrated_from_real_fills": self.calibrated,
        }


def default_scenarios(venue: str = "kraken") -> tuple[BenchmarkScenario, BenchmarkScenario]:
    """A provisional central case plus stress for a given venue (defaults to Kraken)."""
    venue_norm = (venue or "kraken").lower().strip()
    if venue_norm == "binance":
        return (
            BenchmarkScenario(
                name="central",
                description="Binance spot: 0.10% taker / 0.075% maker, moderate spread/slippage",
                fees=FeeModel(limit_fee_pct=0.075, market_fee_pct=0.10),
                execution=ExecutionModel(
                    spread_bps=5.0,
                    market_slippage_bps=10.0,
                    partial_fill_ratio=0.80,
                    intrabar_policy="worst_case",
                ),
            ),
            BenchmarkScenario(
                name="stress",
                description="Binance spot stress: 0.10% maker / 0.15% taker, adverse slippage",
                fees=FeeModel(limit_fee_pct=0.10, market_fee_pct=0.15),
                execution=ExecutionModel(
                    spread_bps=15.0,
                    market_slippage_bps=20.0,
                    partial_fill_ratio=0.50,
                    intrabar_policy="worst_case",
                ),
            ),
        )
    if venue_norm in ("hyperliquid", "hl"):
        return (
            BenchmarkScenario(
                name="central",
                description="Hyperliquid perps: 0.015% maker / 0.045% taker, tight spread",
                fees=FeeModel(limit_fee_pct=0.015, market_fee_pct=0.045),
                execution=ExecutionModel(
                    spread_bps=3.0,
                    market_slippage_bps=5.0,
                    partial_fill_ratio=0.85,
                    intrabar_policy="worst_case",
                ),
            ),
            BenchmarkScenario(
                name="stress",
                description="Hyperliquid perps stress: 0.03% maker / 0.07% taker, adverse slippage",
                fees=FeeModel(limit_fee_pct=0.030, market_fee_pct=0.070),
                execution=ExecutionModel(
                    spread_bps=10.0,
                    market_slippage_bps=15.0,
                    partial_fill_ratio=0.50,
                    intrabar_policy="worst_case",
                ),
            ),
        )
    if venue_norm in ("t212", "trading212"):
        return (
            BenchmarkScenario(
                name="central",
                description="Trading 212: 0% commission + 0.15% FX conversion, moderate spread",
                fees=FeeModel(limit_fee_pct=0.15, market_fee_pct=0.15),
                execution=ExecutionModel(
                    spread_bps=5.0,
                    market_slippage_bps=10.0,
                    partial_fill_ratio=0.75,
                    intrabar_policy="worst_case",
                ),
            ),
            BenchmarkScenario(
                name="stress",
                description="Trading 212 stress: 0.30% FX/slippage, adverse market conditions",
                fees=FeeModel(limit_fee_pct=0.30, market_fee_pct=0.30),
                execution=ExecutionModel(
                    spread_bps=15.0,
                    market_slippage_bps=20.0,
                    partial_fill_ratio=0.50,
                    intrabar_policy="worst_case",
                ),
            ),
        )
    return (
        BenchmarkScenario(
            name="central",
            description=(
                "maker/taker separat; costuri moderate, provizorii, necalibrate"
            ),
            fees=FeeModel(limit_fee_pct=0.16, market_fee_pct=0.26),
            execution=ExecutionModel(
                spread_bps=10.0,
                market_slippage_bps=15.0,
                partial_fill_ratio=0.75,
                intrabar_policy="worst_case",
            ),
        ),
        BenchmarkScenario(
            name="stress",
            description=(
                "adverse fee/spread/slippage and at most a 50% LIMIT fill per bar"
            ),
            fees=FeeModel(limit_fee_pct=0.26, market_fee_pct=0.40),
            execution=ExecutionModel(
                spread_bps=20.0,
                market_slippage_bps=30.0,
                partial_fill_ratio=0.50,
                intrabar_policy="worst_case",
            ),
        ),
    )


def _present(values: Iterable[float | int | None]) -> list[float]:
    return [float(value) for value in values if value is not None]


def _mean(values: Iterable[float | int | None]) -> float | None:
    present = _present(values)
    return statistics.fmean(present) if present else None


def _median(values: Iterable[float | int | None]) -> float | None:
    present = _present(values)
    return statistics.median(present) if present else None


def _minimum(values: Iterable[float | int | None]) -> float | None:
    present = _present(values)
    return min(present) if present else None


def _maximum(values: Iterable[float | int | None]) -> float | None:
    present = _present(values)
    return max(present) if present else None


def _regime(buy_hold_return_pct: float | None, threshold_pct: float) -> str:
    if buy_hold_return_pct is None:
        return "sideways"
    try:
        val = float(buy_hold_return_pct)
    except (TypeError, ValueError):
        return "sideways"
    if not math.isfinite(val):
        return "sideways"
    if val >= threshold_pct:
        return "bull"
    if val <= -threshold_pct:
        return "bear"
    return "sideways"


def aggregate_financial_windows(
    windows: list[dict],
    *,
    initial_capital: float,
    regime_threshold_pct: float = 3.0,
) -> dict:
    """Aggregate OOS TEST without turning the resets into guaranteed profit.

    ``sum_reset_net_pnl_usd`` adds up folds that each start with the same budget
    and the same empty state. It is not compounding and does not simulate position
    continuity across folds; the explicit name prevents the misreading.
    """
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    if regime_threshold_pct < 0:
        raise ValueError("regime_threshold_pct cannot be negative")
    if not windows:
        raise ValueError("the benchmark requires at least one TEST window")

    required = {
        "key", "return_pct", "max_drawdown_pct", "buy_hold_return_pct",
        "cycles", "fills", "metrics",
    }
    for window in windows:
        missing = required.difference(window)
        if missing:
            raise ValueError(f"incomplete financial window: {sorted(missing)}")

    summary = summarize_test_windows(windows)
    metrics = [window["metrics"] for window in windows]
    net_pnls = [float(item["net_pnl"]) for item in metrics]
    regimes = {}
    for name in ("bull", "bear", "sideways"):
        selected = [
            window for window in windows
            if _regime(window.get("buy_hold_return_pct"), regime_threshold_pct) == name
        ]
        regimes[name] = {
            "windows": len(selected),
            "mean_return_pct": _mean(window["return_pct"] for window in selected),
            "worst_return_pct": _minimum(window["return_pct"] for window in selected),
            "mean_buy_hold_return_pct": _mean(
                window.get("buy_hold_return_pct") for window in selected
            ),
        }

    return {
        **summary,
        "mean_buy_hold_return_pct": _mean(
            window.get("buy_hold_return_pct") for window in windows
        ),
        "initial_capital_usd": float(initial_capital),
        "mean_net_pnl_usd_per_window": statistics.fmean(net_pnls),
        "median_net_pnl_usd_per_window": statistics.median(net_pnls),
        "sum_reset_net_pnl_usd": sum(net_pnls),
        "sum_reset_return_on_initial_capital_pct": (
            sum(net_pnls) / float(initial_capital) * 100.0
        ),
        "mean_max_drawdown_abs_usd": _mean(
            item.get("max_drawdown_abs") for item in metrics
        ),
        "worst_max_drawdown_abs_usd": _maximum(
            item.get("max_drawdown_abs") for item in metrics
        ),
        "median_sortino": _median(item.get("sortino") for item in metrics),
        "median_calmar": _median(item.get("calmar") for item in metrics),
        "median_profit_factor": _median(
            item.get("profit_factor") for item in metrics
        ),
        "mean_expectancy_usd": _mean(item.get("expectancy") for item in metrics),
        "worst_cvar_95_pct": _minimum(item.get("cvar_95_pct") for item in metrics),
        "mean_exposure_pct": _mean(item.get("exposure_pct") for item in metrics),
        "mean_turnover_pct": _mean(item.get("turnover_pct") for item in metrics),
        "total_trade_count": sum(int(item.get("trade_count") or 0) for item in metrics),
        "worst_underwater_bars": int(max(
            (int(item.get("max_underwater_periods") or 0) for item in metrics),
            default=0,
        )),
        "total_test_bars": sum(int(window.get("bars") or 0) for window in windows),
        "regime_threshold_pct": float(regime_threshold_pct),
        "regimes": regimes,
    }
