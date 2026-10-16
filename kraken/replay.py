"""Replay the live Kraken strategy over historical OHLC.

The strategy decides orders. The harness separately models fills, costs, and intrabar
ambiguity without network, notifications, or persistent state.
Delegates core execution simulation to strategies/spot_replay.py (SSOT).
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
    "make_kraken_replay_adapter",
    "make_spot_replay_adapter",
    "notify",
    "run_replay",
]


def make_kraken_replay_adapter(params: StratParams, fee_pct: float = 0.25, **kwargs):
    """Return a generic ReplayFunction compatible with evaluation.py walk-forward benchmarks."""
    return make_spot_replay_adapter(params, fee_pct=fee_pct, **kwargs)
