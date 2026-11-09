import base64
from datetime import datetime
from functools import lru_cache
import math
import os
import platform
import random
import re as _re
import time
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


def base_asset(symbol: str) -> str:
    """Return a trading symbol's base asset by removing the first known quote suffix."""
    from providers.quantity import resolve_assets

    return resolve_assets(symbol)[0]


def beep(n=1):
    nf = botcore.safe_pos_float(n, default=None)
    if nf is None:
        return
    for _ in range(min(10, int(nf))):
        if platform.system() == "Windows":
            import winsound

            winsound.Beep(440, 500)
        else:
            os.system('echo "\007"')
        time.sleep(2)


def get_interval_time(valoare_prestabilita=97 * 79, marja_aleatoare=10):
    base_val = botcore.safe_float(valoare_prestabilita, default=float(97 * 79))
    margin_val = abs(botcore.safe_float(marja_aleatoare, default=10.0))
    return abs(base_val + random.uniform(-margin_val, margin_val))


calculate_difference_percent = botcore.diff_percent
are_close = botcore.are_close
format_price_log = botcore.format_price_log


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


def are_close_random(value1, value2, target_tolerance_percent=1.0):
    """Intentionally nondeterministic fuzz/jitter comparison; use ``botcore.are_close`` for trading logic."""
    tf = botcore.safe_float(target_tolerance_percent, default=1.0)
    tol = tf if tf >= 0.0 else 1.0
    upper = tol * (1.01 + random.random() * 0.49)
    return botcore.diff_percent(value1, value2) <= upper


def timestampToTime(timestamp_ms):
    ts_ms = botcore.safe_nonneg_float(timestamp_ms, default=None)
    if ts_ms is None:
        return "N/A"
    try:
        return datetime.fromtimestamp(ts_ms / 1000.0).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError, OverflowError, OSError):
        return "N/A"


def timeMsToHMS(timestamp_ms):
    ts_ms = botcore.safe_nonneg_float(timestamp_ms, default=None)
    return timeToHMS(ts_ms / 1000.0) if ts_ms is not None else "N/A"


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


def secondsToMinutes(max_age_seconds) -> float:
    return botcore.safe_nonneg_float(max_age_seconds, default=0.0) / 60.0


def _parse_decay_params(initial_procent, expired_duration, passs, half_life_duration):
    init_f = botcore.safe_float(initial_procent, default=None)
    exp_f = botcore.safe_float(expired_duration, default=None)
    pass_f = botcore.safe_float(passs, default=None)
    hl_f = botcore.safe_pos_float(half_life_duration, default=None)
    if init_f is None or exp_f is None or pass_f is None or hl_f is None:
        return None
    return init_f, exp_f, pass_f, hl_f


def asymptotic_decrease(initial_procent, expired_duration, passs, half_life_duration=24 * 60 * 60):
    """Gradually decrease ``initial_procent`` asymptotically as ``passs`` increases."""
    parsed = _parse_decay_params(initial_procent, expired_duration, passs, half_life_duration)
    if parsed is None:
        return 0.0
    init_f, exp_f, pass_f, hl_f = parsed
    if exp_f <= 0 or pass_f <= 0:
        return init_f
    return init_f / (1 + (exp_f / hl_f) * pass_f)


def exponential_decrease(initial_procent, expired_duration, passs, half_life_duration=24 * 60 * 60):
    """Decrease ``initial_procent`` exponentially as ``passs`` increases."""
    parsed = _parse_decay_params(initial_procent, expired_duration, passs, half_life_duration)
    if parsed is None:
        return 0.0
    init_f, exp_f, pass_f, hl_f = parsed
    if exp_f <= 0 or pass_f <= 0:
        return init_f
    return init_f * math.exp(-(exp_f * pass_f) / hl_f)


def decrese_value_by_increment_exp(increment_factor, value, coeficient=0.05):
    inc_f = botcore.safe_float(increment_factor, default=None)
    val_f = botcore.safe_float(value, default=None)
    coef_f = botcore.safe_float(coeficient, default=None)
    if inc_f is None or val_f is None or coef_f is None:
        return 0.0, 0.0
    denom = 1.0 + coef_f * (inc_f ** 2)
    adjustment_v1 = botcore.safe_float(val_f / denom, default=0.0) if denom != 0.0 else 0.0
    exp_arg = max(-700.0, min(700.0, -coef_f * inc_f))
    adjustment_v2 = botcore.safe_float(val_f * math.exp(exp_arg), default=0.0)
    return adjustment_v1, adjustment_v2


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
