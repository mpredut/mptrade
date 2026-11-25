"""1:1 SSOT store owner for ``llm_quota_status.json`` (locked multi-key LLM quota & latency-demotion cache)."""
from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

from cachepaths import LLM_QUOTA_STATUS_FILENAME
from state_io import read_json_dict, read_json_recovering

from .base import StateStoreBase


class LlmQuotaStatusStore(StateStoreBase):
    """Exclusive 1:1 owner for reading and locked-updating ``llm_quota_status.json``."""

    FILENAME = LLM_QUOTA_STATUS_FILENAME
    SUBDOMAIN = "llm"
    STORAGE_ROOT = "cachedb"

    @classmethod
    def read_dict_path(
        cls,
        state_path: str,
        *,
        default: Any = None,
        read_fn: Optional[Callable[..., Any]] = None,
    ) -> Any:
        """Read ``llm_quota_status.json`` from ``state_path``, honoring an optional injected test reader."""
        if not state_path:
            return default
        if callable(read_fn):
            return read_fn(state_path, default=default)
        return read_json_dict(state_path, default=default, recover_previous=True)

    @classmethod
    def read_recovering_path(
        cls,
        quota_path: str,
        *,
        default: Any = None,
    ) -> dict[str, Any]:
        """Passively read ``llm_quota_status.json`` with ``.previous`` recovery (returns dict or empty dict)."""
        if not quota_path or not cls.has_disk_file(quota_path):
            return dict(default) if isinstance(default, Mapping) else {}
        loaded = read_json_recovering(quota_path, default=default if default is not None else {})
        return dict(loaded) if isinstance(loaded, dict) else {}

    @classmethod
    def save_snapshot(
        cls,
        state_path: str,
        snapshot: Mapping[str, Any],
        *,
        save_fn: Optional[Callable[..., Any]] = None,
    ) -> bool:
        """Atomically persist ``llm_quota_status.json``, honoring an optional injected test writer."""
        if not state_path or not isinstance(snapshot, Mapping):
            return False
        if callable(save_fn):
            save_fn(state_path, dict(snapshot))
            return True
        cls.save_state_file(state_path, dict(snapshot), indent=None)
        return True
