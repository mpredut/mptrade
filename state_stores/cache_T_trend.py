"""1:1 SSOT store owner for ``cache_T_trend.json`` (empirical Weibull trend lifetime cache)."""
from __future__ import annotations

import os
from typing import Any, Callable, Mapping, Optional

from botcore import safe_pos_float
from cachepaths import CACHE_T_TREND_FILENAME
from state_io import (
    generation_file_lock,
    read_json_dict,
    read_json_dict_cached,
    save_json_state,
)

from .base import StateStoreBase


class CacheTTrendStore(StateStoreBase):
    """Exclusive 1:1 owner for reading and locked-merging ``cache_T_trend.json``."""

    FILENAME = CACHE_T_TREND_FILENAME
    SUBDOMAIN = "trends"
    STORAGE_ROOT = "cachedb"

    @classmethod
    def read_cached_dict(
        cls,
        t_cache_path: str,
        memo: Optional[dict[Any, Any]] = None,
        *,
        max_entries: int = 16,
    ) -> dict[str, Any]:
        """Read ``cache_T_trend.json`` via mtime/size memoized reader (0 ms)."""
        if not t_cache_path:
            return {}
        loaded = read_json_dict_cached(t_cache_path, memo, max_entries=max_entries)
        return dict(loaded) if isinstance(loaded, dict) else {}

    @classmethod
    def read_from_cachedb(
        cls,
        cachedb_dir: Optional[str],
        *,
        read_json_fn: Optional[Callable[[str], Any]] = None,
    ) -> dict[str, Any]:
        """Read ``cache_T_trend.json`` passively across ``trends/`` and legacy root layouts."""
        if not cachedb_dir:
            return {}
        return cls.read_dict(
            cachedb_dir,
            use_cache=False,
            read_json_fn=read_json_fn,
            require_exists=False,
            recover_previous=True,
        )

    @classmethod
    def upsert_entry_locked(
        cls,
        t_cache_path: str,
        cache_key: str,
        entry: Mapping[str, Any],
        *,
        fallback_cache: Optional[Mapping[str, Any]] = None,
        indent: int = 2,
    ) -> bool:
        """Atomically upsert one symbol entry under ``generation_file_lock``, preserving positive-``n`` entries."""
        if not t_cache_path or not cache_key or not isinstance(entry, Mapping):
            return False
        res_has_n = (safe_pos_float(entry.get("n"), default=0.0) or 0.0) > 0.0
        os.makedirs(os.path.dirname(os.path.abspath(t_cache_path)), exist_ok=True)
        with generation_file_lock(t_cache_path):
            disk_cache = read_json_dict(t_cache_path, default=None, recover_previous=True, repair_files=True)
            merged_cache = (
                dict(disk_cache)
                if isinstance(disk_cache, dict)
                else (dict(fallback_cache) if isinstance(fallback_cache, Mapping) else {})
            )
            disk_ent = merged_cache.get(cache_key)
            if not res_has_n and isinstance(disk_ent, Mapping) and (safe_pos_float(disk_ent.get("n"), default=0.0) or 0.0) > 0.0:
                return False
            merged_cache[cache_key] = dict(entry)
            save_json_state(t_cache_path, merged_cache, indent=indent)
            return True
