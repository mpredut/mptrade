"""Centralized Single Source of Truth (SSOT) for cachedb file path resolution."""
from __future__ import annotations

import os

_ROOT = os.path.dirname(os.path.abspath(__file__))


def get_cachedb_dir() -> str:
    """Return the canonical cachedb directory path honoring MPTRADE_CACHEDB_DIR and BINANCE_CACHE_DIR."""
    custom = os.environ.get("MPTRADE_CACHEDB_DIR") or os.environ.get("BINANCE_CACHE_DIR")
    if custom:
        return custom
    return os.path.join(_ROOT, "cachedb")


def cache_path(name: str) -> str:
    """Return the path to a cache file inside the canonical cachedb directory.

    Preserves name when it is already absolute or None/empty.
    """
    if not name:
        return name
    if os.path.isabs(name):
        return name
    norm = name.replace("\\", "/")
    if norm.startswith("cachedb/"):
        name = norm[len("cachedb/"):]
    d = get_cachedb_dir()
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return os.path.join(d, name)
