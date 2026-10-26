"""Centralized Single Source of Truth (SSOT) for cachedb file path resolution."""
from __future__ import annotations

import builtins
import glob
import os

_ROOT = os.path.dirname(os.path.abspath(__file__))

EXACT_LEGACY_FILENAMES = {
    "cache_current_price.json": ("cache_currentprice.json",),
    "cache_short_trend.json": ("cache_instant_trend.json",),
    "cache_long_trend.json": ("cache_price_long_trend.json",),
}
PREFIX_LEGACY_FILENAMES = (
    ("cache_sparse_price_", ("cache_price_sparse_", "cache_price_")),
    ("cache_24_price_", ("cache_24price_",)),
    ("cache_365_price_", ("cache_365price_", "cache_24price_long_")),
)


def legacy_filenames_for(filepath: str) -> tuple[str, ...]:
    """Return legacy pre-symmetry candidate paths for ``filepath``."""
    dirname, basename = os.path.split(filepath)
    candidates = EXACT_LEGACY_FILENAMES.get(basename)
    if candidates is None:
        for new_prefix, old_prefixes in PREFIX_LEGACY_FILENAMES:
            if basename.startswith(new_prefix):
                suffix = basename[len(new_prefix):]
                candidates = tuple(old_prefix + suffix for old_prefix in old_prefixes)
                break
    if not candidates:
        return ()
    return tuple(os.path.join(dirname, c) for c in candidates)


def migrate_legacy_filename_if_needed(filepath: str) -> None:
    """Rename legacy pre-symmetry cache files in place when the new path is absent."""
    if os.path.exists(filepath) or os.path.exists(filepath + ".previous"):
        return
    for legacy_path in legacy_filenames_for(filepath):
        if not os.path.exists(legacy_path) and not os.path.exists(legacy_path + ".previous"):
            continue
        try:
            if os.path.exists(legacy_path):
                os.replace(legacy_path, filepath)
            if (
                os.path.exists(legacy_path + ".meta")
                and not os.path.exists(filepath + ".meta")
            ):
                os.replace(legacy_path + ".meta", filepath + ".meta")
            if (
                os.path.exists(legacy_path + ".previous")
                and not os.path.exists(filepath + ".previous")
            ):
                os.replace(legacy_path + ".previous", filepath + ".previous")
            for old_archive in glob.glob(glob.escape(legacy_path) + ".*.archive*"):
                suffix = old_archive[len(legacy_path):]
                new_archive = filepath + suffix
                if not os.path.exists(new_archive):
                    os.replace(old_archive, new_archive)
            builtins.print(
                f"[cacheManager][migration] renamed {legacy_path} -> {filepath}"
            )
            return
        except OSError as exc:
            builtins.print(
                f"[cacheManager][migration] failed renaming "
                f"{legacy_path} -> {filepath}: {exc}"
            )


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
