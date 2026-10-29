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
    make_spot_replay_adapter,
    prewarm_trend_series,
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
    "make_spot_replay_adapter",
    "notify",
    "prewarm_trend_series",
    "run_replay",
]


def make_binance_replay_adapter(params: StratParams, fee_pct: float = 0.10, **kwargs):
    """Return a generic ReplayFunction compatible with evaluation.py walk-forward benchmarks.

    Binance spot trading standard fee is 0.10% (0.0010) per leg (or 0.075% with BNB discount).
    """
    return make_spot_replay_adapter(params, fee_pct=fee_pct, **kwargs)
