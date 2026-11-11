"""Compatibility re-export shim over ``botcore`` and ``cachepaths``."""
from botcore import (
    _load_ed25519_signing_key,
    _sign_ed25519,
    gaussian_weights_from_idx,
    secondsToDays,
    secondsToHours,
    slope,
    timeToHMS,
    timestampToTime,
    value_diff_to_percent,
)
from cachepaths import cache_path, get_cachedb_dir

CACHE_DIR = get_cachedb_dir()

__all__ = [
    "CACHE_DIR",
    "_load_ed25519_signing_key",
    "_sign_ed25519",
    "cache_path",
    "gaussian_weights_from_idx",
    "secondsToDays",
    "secondsToHours",
    "slope",
    "timeToHMS",
    "timestampToTime",
    "value_diff_to_percent",
]

