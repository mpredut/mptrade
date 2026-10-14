#!/usr/bin/env python3
"""binance_api/replay.py — Replay adapter for Binance spot strategies.

Wraps the unified spot replay engine (strategies/spot_replay.py) with Binance
fee schedules and execution models for walk-forward baseline benchmarking.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from offline.backtests.execution import ExecutionModel, FeeModel
from providers.strategy_executor import PairPrecision
from strategies.spot_engine import StratParams, Strategy, notify
from strategies.spot_replay import (
    _run_once,
    _silent,
    _strat,
    _validate_replay,
    run_replay,
)

__all__ = [
    "ExecutionModel",
    "FeeModel",
    "MagicMock",
    "PairPrecision",
    "StratParams",
    "Strategy",
    "_run_once",
    "_silent",
    "_strat",
    "_validate_replay",
    "make_binance_replay_adapter",
    "notify",
    "run_replay",
]


def make_binance_replay_adapter(
    params: StratParams,
    fee_pct: float = 0.10,
    bar_minutes: float | None = 60.0,
    execution: ExecutionModel | None = None,
    fee_model: FeeModel | None = None,
    include_decision_trace: bool = False,
    initial_cash: float | None = None,
    dca_drop_series=None,
    reentry_series=None,
    intrabar_high: bool = False,
):
    """Return a generic ReplayFunction compatible with evaluation.py walk-forward benchmarks.

    Binance spot trading standard fee is 0.10% (0.0010) per leg (or 0.075% with BNB discount).
    """
    def _adapter(ohlc, warmup_ohlc, _context=None):
        return run_replay(
            ohlc,
            params,
            fee_pct=fee_pct,
            bar_minutes=bar_minutes,
            warmup_ohlc=warmup_ohlc,
            execution=execution,
            fee_model=fee_model,
            include_decision_trace=include_decision_trace,
            initial_cash=initial_cash,
            dca_drop_series=dca_drop_series,
            reentry_series=reentry_series,
            intrabar_high=intrabar_high,
        )
    return _adapter
