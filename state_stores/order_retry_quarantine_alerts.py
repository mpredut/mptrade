"""1:1 SSOT store owner for ``order_retry_quarantine_alerts.json`` (locked per-key read-modify-write)."""
from __future__ import annotations

import os
from typing import Any, Iterable

from cachepaths import ORDER_RETRY_QUARANTINE_ALERTS_FILENAME
from state_io import generation_file_lock, load_json_state, save_json_state

from .base import StateStoreBase


class OrderRetryQuarantineAlertsStore(StateStoreBase):
    """Exclusive 1:1 owner for reading and locked-updating ``order_retry_quarantine_alerts.json``."""

    FILENAME = ORDER_RETRY_QUARANTINE_ALERTS_FILENAME
    SUBDOMAIN = "orders"

    @classmethod
    def load_key(cls, path: str, key: str) -> list[tuple[Any, ...]]:
        """Read persisted tuple items for ``key`` (empty list on missing/corrupt file)."""
        if not path or not key:
            return []
        try:
            data = load_json_state(path, default_factory=dict, fail_closed=False, label="quarantine alerts")
            if isinstance(data, dict):
                items = data.get(key, [])
                if isinstance(items, list):
                    return [tuple(item) for item in items if isinstance(item, list)]
        except Exception:
            pass
        return []

    @classmethod
    def save_key_locked(cls, path: str, key: str, items: Iterable[Any]) -> bool:
        """Merge and persist ``key`` items atomically under ``generation_file_lock`` without clobbering sibling keys."""
        if not path or not key:
            return False
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with generation_file_lock(path):
            loaded = load_json_state(path, default_factory=dict, fail_closed=False, label="quarantine alerts")
            data = dict(loaded) if isinstance(loaded, dict) else {}
            existing = {
                tuple(item)
                for item in (data.get(key) if isinstance(data.get(key), list) else ())
                if isinstance(item, list)
            }
            merged = existing | {
                x if isinstance(x, tuple) else (x,)
                for x in items
            }
            data[key] = [
                list(x)
                for x in sorted(
                    merged,
                    key=lambda item: tuple(str(e) for e in item),
                )
            ]
            save_json_state(path, data, separators=(",", ":"), skip_if_unchanged=True)
            return True

    @classmethod
    def clear_key_locked(cls, path: str, key: str) -> bool:
        """Remove ``key`` atomically under ``generation_file_lock`` while preserving sibling keys."""
        resolved_path = cls._resolve_disk_path(path)
        if not resolved_path or not key or not cls.has_disk_file(resolved_path):
            return False
        with generation_file_lock(resolved_path):
            loaded = load_json_state(resolved_path, default_factory=dict, fail_closed=False, label="quarantine alerts")
            if isinstance(loaded, dict) and key in loaded:
                data = dict(loaded)
                data.pop(key, None)
                save_json_state(resolved_path, data, separators=(",", ":"), skip_if_unchanged=True)
                return True
        return False
