"""1:1 SSOT store owner for ``order_retry_queue.jsonl``."""
from __future__ import annotations

import os
from typing import Any, Optional, Sequence

from botcore import FileLock
from cachepaths import ORDER_RETRY_QUEUE_FILENAME, cache_path, resolve_cachedb_filepath
from state_io import atomic_write_jsonl, read_tail_jsonl_dicts, read_text_lines

from .base import StateStoreBase


class OrderRetryQueueStore(StateStoreBase):
    """Exclusive 1:1 storage owner for ``order_retry_queue.jsonl``."""

    FILENAME = ORDER_RETRY_QUEUE_FILENAME
    SUBDOMAIN = "orders"
    IS_JSONL = True

    @classmethod
    def default_queue_file(cls) -> str:
        """Return default canonical path for ``order_retry_queue.jsonl``."""
        return cache_path(cls.FILENAME)

    @classmethod
    def default_lock_file(cls) -> str:
        """Return default canonical lock file path for ``order_retry_queue.jsonl``."""
        return cache_path("order_retry_queue.lock")

    @classmethod
    def resolve_queue_path(cls, path: Optional[str] = None) -> str:
        """Resolve an explicit queue path or default ``order_retry_queue.jsonl`` path."""
        return str(resolve_cachedb_filepath(cls.default_queue_file() if path is None else path))

    @classmethod
    def resolve_sidecar_path(cls, filename: str, *, queue_path: Optional[str] = None) -> str:
        """Return absolute path for a sidecar file next to the resolved queue file."""
        return os.path.join(os.path.dirname(os.path.abspath(cls.resolve_queue_path(queue_path))), filename)

    @classmethod
    def lock(cls, lock_file: Optional[str] = None, *, queue_path: Optional[str] = None) -> FileLock:
        """Ensure parent directory exists and return a ``FileLock`` guarding the queue."""
        os.makedirs(os.path.dirname(os.path.abspath(cls.resolve_queue_path(queue_path))), exist_ok=True)
        return FileLock(lock_file or cls.default_lock_file())

    @classmethod
    def read_lines_nolock(cls, queue_path: str) -> list[str]:
        """Read raw lines (preserving line endings) from ``queue_path`` while caller holds lock."""
        resolved = cls.resolve_queue_path(queue_path)
        if not os.path.exists(resolved):
            return []
        return list(read_text_lines(resolved, keepends=True))

    @classmethod
    def write_records_nolock(cls, queue_path: str, records: Sequence[Any]) -> None:
        """Atomically replace ``queue_path`` with JSONL ``records`` while caller holds lock."""
        atomic_write_jsonl(queue_path, list(records))

    @classmethod
    def read_tail_records(cls, cachedb_dir: Optional[str], *, limit: int = 50) -> list[dict[str, Any]]:
        """Read-only tail accessor returning up to the last ``limit`` dict records from ``order_retry_queue.jsonl``."""
        if not cachedb_dir:
            return []
        retry_q_path = cls.resolve_path(cachedb_dir)
        if not os.path.exists(retry_q_path):
            return []
        return list(read_tail_jsonl_dicts(retry_q_path, limit=limit))
