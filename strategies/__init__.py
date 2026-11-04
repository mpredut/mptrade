"""Venue-neutral strategy engines.

Each engine owns financial decisions while execution stays behind provider
contracts.  Venue-specific launchers remain responsible for configuration,
market selection, notifications, and live-order gates.
"""

import sys

from .base_spot_mixin import BaseSpotMixin
from .spot_engine import StratParams, Strategy
from .position_exit_manager import (
    ExitAction,
    ExitDecision,
    PositionExitConfig,
    PositionExitManager,
    PositionSnapshot,
    PriceGeometrySnapshot,
    ReentryAction,
    ReentryDecision,
    ReentrySnapshot,
)
from . import base_spot_mixin
from . import spot_engine
from . import spot_rules
from . import position_exit_manager

# Backward-compatibility aliases (zero-file shims)
spot_dca = spot_engine
spot_dca_rules = spot_rules
sys.modules.setdefault(f"{__name__}.spot_dca", spot_engine)
sys.modules.setdefault(f"{__name__}.spot_dca_rules", spot_rules)

__all__ = [
    "BaseSpotMixin",
    "StratParams",
    "Strategy",
    "ExitAction",
    "ExitDecision",
    "PositionExitConfig",
    "PositionExitManager",
    "PositionSnapshot",
    "PriceGeometrySnapshot",
    "ReentryAction",
    "ReentryDecision",
    "ReentrySnapshot",
    "base_spot_mixin",
    "spot_engine",
    "spot_rules",
    "position_exit_manager",
    "spot_dca",
    "spot_dca_rules",
]
