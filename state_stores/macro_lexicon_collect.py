"""1:1 SSOT store owner for ``macro_lexicon_collect.json`` (locked dynamic crisis lexicon cache)."""
from __future__ import annotations

import os
from typing import Any, Mapping, Optional

from cachepaths import CACHE_MACRO_LEXICON_FILENAME
from state_io import cachedb_exists, read_json_dict, read_json_dict_cached

from .base import StateStoreBase


class MacroLexiconCollectStore(StateStoreBase):
    """Exclusive 1:1 owner for reading and locked-updating ``macro_lexicon_collect.json``."""

    FILENAME = CACHE_MACRO_LEXICON_FILENAME
    SUBDOMAIN = "macro"
    STORAGE_ROOT = "cachedb"

    @classmethod
    def exists_on_disk(cls, path: str) -> bool:
        """Return True when ``path`` or its ``.previous`` backup exists."""
        return bool(path and cachedb_exists(path, include_previous=True))

    @classmethod
    def read_cached_dict(
        cls,
        path: str,
        memo: Optional[dict[Any, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        """Read ``macro_lexicon_collect.json`` via mtime/size memoized passive reader (0 ms)."""
        if not path:
            return None
        loaded = read_json_dict_cached(path, memo)
        return loaded if isinstance(loaded, dict) else None

    @classmethod
    def read_locked_dict(cls, path: str) -> Optional[dict[str, Any]]:
        """Read ``macro_lexicon_collect.json`` inside ``generation_lock`` with ``.previous`` repair."""
        if not cls.exists_on_disk(path):
            return None
        raw = read_json_dict(path, default=None, recover_previous=True, repair_files=True)
        return raw if isinstance(raw, dict) else None

    @classmethod
    def save_doc(
        cls,
        path: str,
        doc: Mapping[str, Any],
        *,
        indent: int = 2,
    ) -> bool:
        """Atomically persist ``macro_lexicon_collect.json``."""
        if not path or not isinstance(doc, Mapping):
            return False
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        cls.save_state_file(path, dict(doc), indent=indent)
        return True
