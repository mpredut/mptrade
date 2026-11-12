"""Backward-compatible re-exports from canonical SSOT ``strategies.replay_execution``."""

from providers.ohlc_utils import validate_ohlc_bars
from strategies.replay_execution import (
    choose_intrabar_scenario,
    ExecutionModel,
    FeeModel,
    select_eligible_intrabar_orders,
    split_order_fill,
    _POLICIES,
)

__all__ = [
    "_POLICIES",
    "ExecutionModel",
    "FeeModel",
    "choose_intrabar_scenario",
    "select_eligible_intrabar_orders",
    "split_order_fill",
    "validate_ohlc_bars",
]
