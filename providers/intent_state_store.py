"""providers/intent_state_store.py -- Layer 3 persistence facade for durable autonomous AI trade intents.

Delegates locking, atomic reads, recovery, and upserts to Layer 2.5 ``AutonomousTradeIntentsStateStore``.
Pure schema normalization, ID correlation, and validation live exclusively in ``providers/intent_state.py``
(zero disk I/O).
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from state_stores.autonomous_trade_intents_state import AutonomousTradeIntentsStateStore

from .intent_state import _extract_intent_id, collect_known_ai_order_ids


def durable_intents_lock(intents_file: str):
    """Reentrant (per thread) cross-process lock serializing read-modify-write via 1:1 store."""
    return AutonomousTradeIntentsStateStore.lock(intents_file)


def load_durable_intents_state(
    intents_file: str,
    reconcile_fn: Optional[Callable[[list[dict[str, Any]]], Any]] = None,
    *,
    read_only: bool = False,
    fail_closed: Optional[bool] = None,
) -> list[dict[str, Any]]:
    """Load all trade intents from the durable JSON queue via ``AutonomousTradeIntentsStateStore``."""
    return AutonomousTradeIntentsStateStore.load_state(
        intents_file,
        reconcile_fn=reconcile_fn,
        read_only=read_only,
        fail_closed=fail_closed,
    )


def load_durable_intents_from_cachedb(
    cachedb_dir: Optional[str],
    *,
    read_json_fn: Optional[Callable[[str], Any]] = None,
) -> list[dict[str, Any]]:
    """Read-only SSOT loader for ``autonomous_trade_intents_state.json`` via ``AutonomousTradeIntentsStateStore``."""
    return AutonomousTradeIntentsStateStore.load_from_cachedb(
        cachedb_dir,
        read_json_fn=read_json_fn,
    )


def load_known_ai_order_ids_from_cachedb(cachedb_dir: Optional[str]) -> set[str]:
    """Return lowercase venue order IDs and derived venue-native cloids from durable AI intents."""
    intents = load_durable_intents_from_cachedb(cachedb_dir)
    if not intents:
        return set()
    return {tok.lower() for tok in collect_known_ai_order_ids(intents) if tok}


def save_durable_intents_state(
    intents_file: str,
    intents: list[dict[str, Any]],
    log: Optional[Any] = None,
) -> bool:
    """Transactionally persist all intents via ``AutonomousTradeIntentsStateStore``."""
    return AutonomousTradeIntentsStateStore.save_state(intents_file, intents, log=log)


def upsert_durable_intent(
    intents_file: str,
    intent: dict[str, Any],
    *,
    reconcile_fn: Optional[Callable[[list[dict[str, Any]]], Any]] = None,
    log: Optional[Any] = None,
) -> bool:
    """Atomically merge one intent (by ``intent_id``) via ``AutonomousTradeIntentsStateStore``."""
    return AutonomousTradeIntentsStateStore.upsert_intent(
        intents_file,
        intent,
        extract_id_fn=_extract_intent_id,
        reconcile_fn=reconcile_fn,
        log=log,
    )


__all__ = [
    "durable_intents_lock",
    "load_durable_intents_from_cachedb",
    "load_durable_intents_state",
    "load_known_ai_order_ids_from_cachedb",
    "save_durable_intents_state",
    "upsert_durable_intent",
]
