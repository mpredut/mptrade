"""Venue-neutral strategy engines.

Each engine owns financial decisions while execution stays behind provider
contracts.  Venue-specific launchers remain responsible for configuration,
market selection, notifications, and live-order gates.
"""

from .spot_engine import StratParams, Strategy
from . import spot_engine
from . import spot_rules
# Backward-compatibility aliases
from . import spot_dca
from . import spot_dca_rules

__all__ = ["StratParams", "Strategy", "spot_engine", "spot_rules", "spot_dca", "spot_dca_rules"]
