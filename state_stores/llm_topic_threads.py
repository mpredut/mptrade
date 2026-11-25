"""1:1 SSOT store owner for ``llm_topic_threads.json``."""
from __future__ import annotations

import os
from typing import Any, Callable, Optional

from cachepaths import (
    LLM_TOPIC_THREADS_FILENAME,
    resolve_statedb_dir_from_cachedb,
    resolve_statedb_filepath,
    state_path,
)

from .base import StateStoreBase


class LlmTopicThreadsStore(StateStoreBase):
    """Exclusive 1:1 owner for durable read/write access to ``llm_topic_threads.json``."""

    FILENAME = LLM_TOPIC_THREADS_FILENAME
    SUBDOMAIN = "llm"

    @classmethod
    def default_path(cls) -> str:
        """Return default canonical path for ``llm_topic_threads.json`` (``statedb/llm/`` with fallback)."""
        return state_path(cls.FILENAME)

    @classmethod
    def candidate_paths(cls, cachedb_dir: str) -> list[str]:
        """Return deduplicated candidate paths across ``statedb/llm/``, ``cachedb/llm/``, ``cachedb/system/``, and root."""
        statedb_dir = resolve_statedb_dir_from_cachedb(cachedb_dir)
        return list(
            dict.fromkeys((
                cls.resolve_path(cachedb_dir),
                os.path.join(statedb_dir, cls.SUBDOMAIN, cls.FILENAME),
                os.path.join(cachedb_dir, cls.SUBDOMAIN, cls.FILENAME),
                os.path.join(cachedb_dir, "system", cls.FILENAME),
                os.path.join(cachedb_dir, cls.FILENAME),
            ))
        )

    @classmethod
    def resolve_registry_path(
        cls,
        registry_path: Optional[str] = None,
        *,
        env_override: str = "",
        expand_fn: Optional[Callable[[str], str]] = None,
        fallback_root: str = "",
    ) -> str:
        """Resolve explicit override path or canonical ``statedb/llm/llm_topic_threads.json``."""
        raw = registry_path or env_override
        if raw:
            expanded = expand_fn(raw) if callable(expand_fn) else raw
            return str(resolve_statedb_filepath(expanded))
        try:
            return cls.default_path()
        except Exception:
            return os.path.join(fallback_root, "cachedb", cls.FILENAME) if fallback_root else cls.FILENAME

    @classmethod
    def read_registry(cls, path: str, *, repair_files: bool = False) -> dict[str, Any]:
        """Load the topic thread registry JSON dict with ``.previous`` crash recovery."""
        raw = cls.read_file_dict(path, default={}, recover_previous=True, repair_files=repair_files)
        return dict(raw) if isinstance(raw, dict) else {}

    @classmethod
    def save_registry(cls, path: str, registry: dict[str, Any]) -> None:
        """Atomically persist the topic thread registry under its cross-process generation lock."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with cls.generation_lock(path):
            cls.save_state_file(path, registry, indent=2)
