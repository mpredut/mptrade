"""Venue-neutral strategy engines.

Each engine owns financial decisions while execution stays behind provider
contracts.  Venue-specific launchers remain responsible for configuration,
market selection, notifications, and live-order gates.
"""

from .spot_engine import StratParams, Strategy
from .position_exit_manager import (
    ExitAction,
    ExitDecision,
    PositionExitConfig,
    PositionExitManager,
    PositionSnapshot,
    ReentryAction,
    ReentryDecision,
    ReentrySnapshot,
)
from . import spot_engine
from . import spot_rules
from . import position_exit_manager
# Backward-compatibility aliases
from . import spot_dca
from . import spot_dca_rules

__all__ = [
    "StratParams",
    "Strategy",
    "ExitAction",
    "ExitDecision",
    "PositionExitConfig",
    "PositionExitManager",
    "PositionSnapshot",
    "ReentryAction",
    "ReentryDecision",
    "ReentrySnapshot",
    "spot_engine",
    "spot_rules",
    "position_exit_manager",
    "spot_dca",
    "spot_dca_rules",
]
