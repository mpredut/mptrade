"""Central SSOT registry mapping every critical ``mptrade`` state file 1:1 to its ``state_stores`` owner."""
from __future__ import annotations

from pathlib import Path

from cachepaths import (
    ASSETGUARDIAN_STATE_FILENAME,
    AUTONOMOUS_TRADE_INTENTS_STATE_FILENAME,
    MT_STATE_FILENAME,
    NAV_HISTORY_STATE_FILENAME,
    ORDER_RETRY_QUEUE_FILENAME,
    PORTFOLIO_POSITIONS_STATE_FILENAME,
    RTRADE_PAIRS_STATE_FILENAME,
    SMART_STOP_LOSS_STATE_FILENAME,
    TRAILING_STATE_FILENAME,
    VENUE_BALANCES_STATE_FILENAME,
    VENUE_FEE_MAINTENANCE_STATE_FILENAME,
)

from .base import StateFileSpec


def expected_store_class_name(filename: str) -> str:
    """Derive canonical 1:1 ``<PascalCase>Store`` class name from a state filename."""
    stem = Path(filename).stem
    return "".join(part.capitalize() for part in stem.split("_") if part) + "Store"


CRITICAL_STATE_SPECS: tuple[StateFileSpec, ...] = (
    StateFileSpec(
        filename=VENUE_BALANCES_STATE_FILENAME,
        const_name="VENUE_BALANCES_STATE_FILENAME",
        owner_module="state_stores/venue_balances_state.py",
        store_class="VenueBalancesStateStore",
        subdomain="state",
    ),
    StateFileSpec(
        filename=PORTFOLIO_POSITIONS_STATE_FILENAME,
        const_name="PORTFOLIO_POSITIONS_STATE_FILENAME",
        owner_module="state_stores/portfolio_positions_state.py",
        store_class="PortfolioPositionsStateStore",
        subdomain="state",
    ),
    StateFileSpec(
        filename=AUTONOMOUS_TRADE_INTENTS_STATE_FILENAME,
        const_name="AUTONOMOUS_TRADE_INTENTS_STATE_FILENAME",
        owner_module="state_stores/autonomous_trade_intents_state.py",
        store_class="AutonomousTradeIntentsStateStore",
        subdomain="state",
    ),
    StateFileSpec(
        filename=ORDER_RETRY_QUEUE_FILENAME,
        const_name="ORDER_RETRY_QUEUE_FILENAME",
        owner_module="state_stores/order_retry_queue.py",
        store_class="OrderRetryQueueStore",
        subdomain="orders",
        is_jsonl=True,
    ),
    StateFileSpec(
        filename=NAV_HISTORY_STATE_FILENAME,
        const_name="NAV_HISTORY_STATE_FILENAME",
        owner_module="state_stores/nav_history_state.py",
        store_class="NavHistoryStateStore",
        subdomain="state",
    ),
    StateFileSpec(
        filename=VENUE_FEE_MAINTENANCE_STATE_FILENAME,
        const_name="VENUE_FEE_MAINTENANCE_STATE_FILENAME",
        owner_module="state_stores/venue_fee_maintenance_state.py",
        store_class="VenueFeeMaintenanceStateStore",
        subdomain="state",
    ),
    StateFileSpec(
        filename=ASSETGUARDIAN_STATE_FILENAME,
        const_name="ASSETGUARDIAN_STATE_FILENAME",
        owner_module="state_stores/assetguardian_state.py",
        store_class="AssetguardianStateStore",
        subdomain="state",
    ),
    StateFileSpec(
        filename=SMART_STOP_LOSS_STATE_FILENAME,
        const_name="SMART_STOP_LOSS_STATE_FILENAME",
        owner_module="state_stores/smart_stop_loss_state.py",
        store_class="SmartStopLossStateStore",
        subdomain="state",
    ),
    StateFileSpec(
        filename=MT_STATE_FILENAME,
        const_name="MT_STATE_FILENAME",
        owner_module="state_stores/mt_state.py",
        store_class="MtStateStore",
        subdomain="state",
    ),
    StateFileSpec(
        filename=TRAILING_STATE_FILENAME,
        const_name="TRAILING_STATE_FILENAME",
        owner_module="state_stores/trailing_state.py",
        store_class="TrailingStateStore",
        subdomain="state",
    ),
    StateFileSpec(
        filename=RTRADE_PAIRS_STATE_FILENAME,
        const_name="RTRADE_PAIRS_STATE_FILENAME",
        owner_module="state_stores/rtrade_pairs_state.py",
        store_class="RtradePairsStateStore",
        subdomain="state",
    ),
)

STATE_SPEC_BY_CONST: dict[str, StateFileSpec] = {spec.const_name: spec for spec in CRITICAL_STATE_SPECS}
STATE_SPEC_BY_FILENAME: dict[str, StateFileSpec] = {spec.filename: spec for spec in CRITICAL_STATE_SPECS}
STATE_SPEC_BY_OWNER_MODULE: dict[str, StateFileSpec] = {spec.owner_module: spec for spec in CRITICAL_STATE_SPECS}

__all__ = [
    "CRITICAL_STATE_SPECS",
    "STATE_SPEC_BY_CONST",
    "STATE_SPEC_BY_FILENAME",
    "STATE_SPEC_BY_OWNER_MODULE",
    "StateFileSpec",
    "expected_store_class_name",
]
