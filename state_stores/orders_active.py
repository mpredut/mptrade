"""1:1 SSOT store owner for ``orders_active.json`` (OrderLedger materialized active snapshot)."""
from __future__ import annotations

from typing import Any, Mapping, Optional

from cachepaths import ORDER_LEDGER_ACTIVE_FILENAME, resolve_cachedb_filepath
from state_io import cachedb_exists, read_json_dict_recovering, save_json_state

from .base import StateStoreBase


class OrdersActiveStore(StateStoreBase):
    """Exclusive 1:1 owner for reading and writing ``orders_active.json``."""

    FILENAME = ORDER_LEDGER_ACTIVE_FILENAME
    SUBDOMAIN = "orders"

    @classmethod
    def read_raw_snapshot(
        cls,
        active_path: str,
        *,
        repair_files: bool = False,
    ) -> Optional[dict[str, Any]]:
        """Read ``orders_active.json`` passively with ``.previous`` recovery."""
        if not active_path:
            return None
        path = str(resolve_cachedb_filepath(active_path))
        if not cachedb_exists(path, include_previous=True):
            return None
        raw = read_json_dict_recovering(path, repair_files=bool(repair_files))
        return raw if isinstance(raw, dict) else None

    @classmethod
    def write_snapshot(
        cls,
        active_path: str,
        snapshot: Mapping[str, Any],
        *,
        indent: int = 1,
    ) -> bool:
        """Atomically persist ``orders_active.json``."""
        if not active_path or not isinstance(snapshot, Mapping):
            return False
        save_json_state(active_path, dict(snapshot), indent=indent, sort_keys=False)
        return True
