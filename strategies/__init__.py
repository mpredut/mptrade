"""Venue-neutral strategy engines.

Each engine owns financial decisions while execution stays behind provider
contracts.  Venue-specific launchers remain responsible for configuration,
market selection, notifications, and live-order gates.
"""

from .spot_engine import StratParams, Strategy
from . import replay_execution, replay_metrics, spot_engine

__all__ = [
    "StratParams",
    "Strategy",
    "replay_execution",
    "replay_metrics",
    "spot_engine",
]

