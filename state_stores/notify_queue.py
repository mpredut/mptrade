"""1:1 SSOT store owner for ``notify_queue.json``."""
from __future__ import annotations

import os
from typing import Sequence

from cachepaths import (
    NOTIFY_QUEUE_FILENAME,
    resolve_statedb_dir_from_cachedb,
    resolve_statedb_filepath,
    state_path,
)
from state_io import atomic_write_text, read_text_lines

from .base import StateStoreBase


class NotifyQueueStore(StateStoreBase):
    """Exclusive 1:1 owner for durable queued notifications in ``notify_queue.json``."""

    FILENAME = NOTIFY_QUEUE_FILENAME
    SUBDOMAIN = "system"
    IS_JSONL = True

    @classmethod
    def default_queue_file(cls) -> str:
        """Return default canonical path for ``notify_queue.json`` (``statedb/system/`` with fallback)."""
        return state_path(cls.FILENAME)

    @classmethod
    def candidate_paths(cls, cachedb_dir: str) -> list[str]:
        """Return deduplicated candidate paths across ``statedb/system/``, ``cachedb/system/``, and root."""
        statedb_dir = resolve_statedb_dir_from_cachedb(cachedb_dir)
        return list(
            dict.fromkeys((
                cls.resolve_path(cachedb_dir),
                os.path.join(statedb_dir, cls.SUBDOMAIN, cls.FILENAME),
                os.path.join(cachedb_dir, cls.SUBDOMAIN, cls.FILENAME),
                os.path.join(cachedb_dir, cls.FILENAME),
            ))
        )

    @classmethod
    def read_lines_nolock(cls, queue_path: str) -> list[str]:
        """Read raw lines (preserving line endings) from ``queue_path`` while caller holds lock."""
        resolved = str(resolve_statedb_filepath(queue_path))
        if not os.path.exists(resolved):
            return []
        return list(read_text_lines(resolved, keepends=True, errors="replace"))

    @classmethod
    def write_lines_nolock(cls, queue_path: str, lines: Sequence[str]) -> None:
        """Atomically replace ``queue_path`` with newline-delimited JSON lines while caller holds lock."""
        resolved = str(resolve_statedb_filepath(queue_path))
        content = "".join(f"{ln}\n" for ln in lines)
        atomic_write_text(resolved, content)
