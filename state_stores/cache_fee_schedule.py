"""1:1 SSOT store owner for ``cache_fee_schedule.json`` (per-venue exchange fee tiers & BNB reserve cache)."""
from __future__ import annotations

import os
from typing import Any, Callable, Mapping, Optional

from cachepaths import CACHE_FEE_SCHEDULE_FILENAME, cache_path, clean_scalar_str, resolve_cachedb_filepath
from state_io import read_json_dict

from .base import StateStoreBase


class CacheFeeScheduleStore(StateStoreBase):
    """Exclusive 1:1 owner for reading and writing ``cache_fee_schedule.json``."""

    FILENAME = CACHE_FEE_SCHEDULE_FILENAME
    SUBDOMAIN = "account"
    STORAGE_ROOT = "cachedb"

    @classmethod
    def resolve_schedule_path(cls, cache_dir: Optional[str] = None, *, migrate: bool = False) -> str:
        """Resolve canonical ``cachedb/account/cache_fee_schedule.json`` (with pre-cutover ``orders/`` fallback)."""
        if cache_dir and (clean_dir := clean_scalar_str(cache_dir)):
            resolved = str(
                resolve_cachedb_filepath(os.path.join(clean_dir, cls.FILENAME), migrate=migrate)
            )
            if os.path.exists(resolved) or os.path.abspath(os.path.dirname(resolved)) != os.path.abspath(clean_dir):
                return resolved
            return os.path.join(clean_dir, cls.SUBDOMAIN, cls.FILENAME)
        return str(resolve_cachedb_filepath(cache_path(cls.FILENAME), migrate=migrate))

    @classmethod
    def read_schedule(
        cls,
        cache_dir: Optional[str],
        *,
        read_json_fn: Optional[Callable[[str], Any]] = None,
    ) -> dict[str, Any]:
        """Read ``cache_fee_schedule.json`` across ``account/``, pre-cutover ``orders/``, and flat root layouts."""
        if not cache_dir or not (clean_dir := clean_scalar_str(cache_dir)):
            return {}
        candidates = list(
            dict.fromkeys((
                cls.resolve_schedule_path(clean_dir),
                os.path.join(clean_dir, cls.SUBDOMAIN, cls.FILENAME),
                os.path.join(clean_dir, "orders", cls.FILENAME),
                os.path.join(clean_dir, cls.FILENAME),
            ))
        )
        for cand in candidates:
            if callable(read_json_fn):
                raw = read_json_fn(cand)
            else:
                if not cls.has_disk_file(cand):
                    continue
                raw = read_json_dict(cand, default=None, recover_previous=True)
            if isinstance(raw, dict) and raw:
                return raw
        return {}

    @classmethod
    def save_schedule(
        cls,
        sched_path: str,
        payload: Mapping[str, Any],
        *,
        indent: int = 2,
    ) -> bool:
        """Atomically persist ``cache_fee_schedule.json`` (skipping unchanged writes)."""
        if not sched_path or not isinstance(payload, Mapping):
            return False
        cls.save_state_file(sched_path, dict(payload), indent=indent, skip_if_unchanged=True)
        return True
