import base64
from datetime import datetime
from functools import lru_cache
import os
import re as _re
import nacl.signing
try:
    import numpy as np
except Exception:
    np = None  # type: ignore[assignment]

import botcore
from cachepaths import (
    cache_path as _cp_cache_path,
    get_cachedb_dir,
    resolve_cachedb_filepath,
)
from state_io import read_text_file


def value_diff_to_percent(value1, value2):
    v1 = botcore.safe_float(value1, default=None)
    v2 = botcore.safe_float(value2, default=None)
    if v1 is None or v2 is None or v1 == 0.0:
        return 0.0
    return botcore.safe_float(((v1 - v2) / v1) * 100.0, default=0.0)


def slope(val1, idx1, val2, idx2):
    v1 = botcore.safe_float(val1, default=None)
    i1 = botcore.safe_float(idx1, default=None)
    v2 = botcore.safe_float(val2, default=None)
    i2 = botcore.safe_float(idx2, default=None)
    if v1 is None or i1 is None or v2 is None or i2 is None or i1 == i2:
        return 0.0
    return botcore.safe_float((v2 - v1) / (i2 - i1), default=0.0)


def timestampToTime(timestamp_ms):
    ts_ms = botcore.safe_nonneg_float(timestamp_ms, default=None)
    if ts_ms is None:
        return "N/A"
    try:
        return datetime.fromtimestamp(ts_ms / 1000.0).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError, OverflowError, OSError):
        return "N/A"


@lru_cache(maxsize=256)
def timeToHMS(timestamp_sec):
    ts_f = botcore.normalize_epoch_sec(timestamp_sec, default=None, allow_zero=True)
    if ts_f is None:
        return "N/A"
    try:
        return datetime.fromtimestamp(ts_f).strftime("%H:%M:%S")
    except (TypeError, ValueError, OverflowError, OSError):
        return "N/A"


def secondsToDays(max_age_seconds) -> float:
    return botcore.safe_nonneg_float(max_age_seconds, default=0.0) / 86400.0


def secondsToHours(max_age_seconds) -> float:
    return botcore.safe_nonneg_float(max_age_seconds, default=0.0) / 3600.0


def gaussian_weights_from_idx(T, idx: int):
    """Return ``t`` as the position axis ``[idx..T-1]`` and ``w`` as its Gaussian weights."""
    t_f = botcore.safe_pos_float(T, default=None)
    idx_f = botcore.safe_float(idx, default=None)
    if t_f is None or idx_f is None:
        return np.array([]), np.array([])
    t_steps = min(100000, max(1, int(round(t_f))))
    idx_int = max(0, int(idx_f))
    if idx_int >= t_steps:
        return np.array([]), np.array([])

    t_full = np.linspace(0, t_steps - 1, t_steps)
    mu = (t_steps - 1) / 2
    sigma = t_steps / 4
    w_full = np.exp(-0.5 * ((t_full - mu) / sigma) ** 2)
    w_full = w_full / w_full.sum()
    return t_full[idx_int:], w_full[idx_int:]


def _load_ed25519_signing_key():
    try:
        pem_data = read_text_file("keys/ed25519_private.pem")
        b64match = _re.search(r"-----BEGIN PRIVATE KEY-----(.*?)-----END PRIVATE KEY-----", pem_data, _re.DOTALL)
        if not b64match:
            raise ValueError("Invalid PEM")
        der_bytes = base64.b64decode(b64match.group(1).strip())
        if len(der_bytes) < 32:
            raise ValueError("DER too short")
        return nacl.signing.SigningKey(der_bytes[-32:])
    except Exception as e:
        print(f"Error loading the Ed25519 key: {e}")
        return None


def _sign_ed25519(signing_key, payload: str) -> str:
    if signing_key is None or payload is None or isinstance(payload, bool):
        raise ValueError("signing_key and payload are required")
    raw_bytes = payload if isinstance(payload, bytes) else str(payload).encode("utf-8")
    return base64.b64encode(signing_key.sign(raw_bytes).signature).decode("ascii")


CACHE_DIR = get_cachedb_dir()


def cache_path(name):
    """Return the cache path under ``cachedb``, preserving paths that already carry a directory separator."""
    if not name or isinstance(name, bool) or not isinstance(name, (str, os.PathLike)):
        return name
    name_str = str(os.fspath(name))
    if os.path.isabs(name_str) or os.sep in name_str or (os.altsep and os.altsep in name_str):
        return str(resolve_cachedb_filepath(name_str))
    return _cp_cache_path(name_str)
