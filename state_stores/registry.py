"""Central SSOT registry mapping every critical ``mptrade`` state file 1:1 to its ``state_stores`` owner."""
from __future__ import annotations

from pathlib import Path

from cachepaths import (
    ASSETGUARDIAN_STATE_FILENAME,
    AUTO_ORDER_INTENTS_STATE_FILENAME,
    CACHE_FEE_SCHEDULE_FILENAME,
    CACHE_MACRO_LEXICON_FILENAME,
    CACHE_T_TREND_FILENAME,
    LLM_QUOTA_STATUS_FILENAME,
    LLM_TOPIC_THREADS_FILENAME,
    MICROSTRUCTURE_CYCLE_HISTORY_FILENAME,
    MT_STATE_FILENAME,
    NAV_HISTORY_STATE_FILENAME,
    NOTIFY_QUEUE_FILENAME,
    ORDER_LEDGER_ACTIVE_FILENAME,
    ORDER_RETRY_QUARANTINE_ALERTS_FILENAME,
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
        subdomain="portfolio",
    ),
    StateFileSpec(
        filename=PORTFOLIO_POSITIONS_STATE_FILENAME,
        const_name="PORTFOLIO_POSITIONS_STATE_FILENAME",
        owner_module="state_stores/portfolio_positions_state.py",
        store_class="PortfolioPositionsStateStore",
        subdomain="portfolio",
    ),
    StateFileSpec(
        filename=AUTO_ORDER_INTENTS_STATE_FILENAME,
        const_name="AUTO_ORDER_INTENTS_STATE_FILENAME",
        owner_module="state_stores/auto_order_intents_state.py",
        store_class="AutoOrderIntentsStateStore",
        subdomain="orders",
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
        subdomain="portfolio",
    ),
    StateFileSpec(
        filename=VENUE_FEE_MAINTENANCE_STATE_FILENAME,
        const_name="VENUE_FEE_MAINTENANCE_STATE_FILENAME",
        owner_module="state_stores/venue_fee_maintenance_state.py",
        store_class="VenueFeeMaintenanceStateStore",
        subdomain="bots",
    ),
    StateFileSpec(
        filename=ASSETGUARDIAN_STATE_FILENAME,
        const_name="ASSETGUARDIAN_STATE_FILENAME",
        owner_module="state_stores/assetguardian_state.py",
        store_class="AssetguardianStateStore",
        subdomain="bots",
    ),
    StateFileSpec(
        filename=SMART_STOP_LOSS_STATE_FILENAME,
        const_name="SMART_STOP_LOSS_STATE_FILENAME",
        owner_module="state_stores/smart_stop_loss_state.py",
        store_class="SmartStopLossStateStore",
        subdomain="bots",
    ),
    StateFileSpec(
        filename=MT_STATE_FILENAME,
        const_name="MT_STATE_FILENAME",
        owner_module="state_stores/mt_state.py",
        store_class="MtStateStore",
        subdomain="bots",
    ),
    StateFileSpec(
        filename=TRAILING_STATE_FILENAME,
        const_name="TRAILING_STATE_FILENAME",
        owner_module="state_stores/trailing_state.py",
        store_class="TrailingStateStore",
        subdomain="bots",
    ),
    StateFileSpec(
        filename=RTRADE_PAIRS_STATE_FILENAME,
        const_name="RTRADE_PAIRS_STATE_FILENAME",
        owner_module="state_stores/rtrade_pairs_state.py",
        store_class="RtradePairsStateStore",
        subdomain="bots",
    ),
    StateFileSpec(
        filename=LLM_TOPIC_THREADS_FILENAME,
        const_name="LLM_TOPIC_THREADS_FILENAME",
        owner_module="state_stores/llm_topic_threads.py",
        store_class="LlmTopicThreadsStore",
        subdomain="llm",
    ),
    StateFileSpec(
        filename=NOTIFY_QUEUE_FILENAME,
        const_name="NOTIFY_QUEUE_FILENAME",
        owner_module="state_stores/notify_queue.py",
        store_class="NotifyQueueStore",
        subdomain="system",
        is_jsonl=True,
    ),
    StateFileSpec(
        filename=MICROSTRUCTURE_CYCLE_HISTORY_FILENAME,
        const_name="MICROSTRUCTURE_CYCLE_HISTORY_FILENAME",
        owner_module="state_stores/microstructure_cycle_history.py",
        store_class="MicrostructureCycleHistoryStore",
        subdomain="telemetry",
        storage_root="cachedb",
    ),
    StateFileSpec(
        filename=ORDER_RETRY_QUARANTINE_ALERTS_FILENAME,
        const_name="ORDER_RETRY_QUARANTINE_ALERTS_FILENAME",
        owner_module="state_stores/order_retry_quarantine_alerts.py",
        store_class="OrderRetryQuarantineAlertsStore",
        subdomain="orders",
        storage_root="statedb",
    ),
    StateFileSpec(
        filename=ORDER_LEDGER_ACTIVE_FILENAME,
        const_name="ORDER_LEDGER_ACTIVE_FILENAME",
        owner_module="state_stores/orders_active.py",
        store_class="OrdersActiveStore",
        subdomain="orders",
        storage_root="statedb",
    ),
    StateFileSpec(
        filename=CACHE_T_TREND_FILENAME,
        const_name="CACHE_T_TREND_FILENAME",
        owner_module="state_stores/cache_T_trend.py",
        store_class="CacheTTrendStore",
        subdomain="trends",
        storage_root="cachedb",
    ),
    StateFileSpec(
        filename=CACHE_FEE_SCHEDULE_FILENAME,
        const_name="CACHE_FEE_SCHEDULE_FILENAME",
        owner_module="state_stores/cache_fee_schedule.py",
        store_class="CacheFeeScheduleStore",
        subdomain="account",
        storage_root="cachedb",
    ),
    StateFileSpec(
        filename=CACHE_MACRO_LEXICON_FILENAME,
        const_name="CACHE_MACRO_LEXICON_FILENAME",
        owner_module="state_stores/macro_lexicon_collect.py",
        store_class="MacroLexiconCollectStore",
        subdomain="macro",
        storage_root="cachedb",
    ),
    StateFileSpec(
        filename=LLM_QUOTA_STATUS_FILENAME,
        const_name="LLM_QUOTA_STATUS_FILENAME",
        owner_module="state_stores/llm_quota_status.py",
        store_class="LlmQuotaStatusStore",
        subdomain="llm",
        storage_root="cachedb",
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
