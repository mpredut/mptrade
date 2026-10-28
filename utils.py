import os
import time
import math
import random
import platform
import numpy as np
from datetime import datetime, timedelta

import botcore   # single are_close/diff_percent source shared by fleet and bots


# Known quote suffixes in matching order; the first match wins. Centralized on
# July 28 after previously being copied across monitortrades, replay_provider,
# and verify_tools.
def base_asset(symbol: str) -> str:
    """Return a trading symbol's base asset by removing the first known quote suffix.
    For example BTCUSDC becomes BTC; an unknown suffix leaves the symbol unchanged."""
    from providers.quantity import resolve_assets
    return resolve_assets(symbol)[0]


def beep(n=1):
    nf = botcore.safe_pos_float(n, default=None)
    if nf is None:
        return
    count = min(10, int(nf))
    for _ in range(count):
        if platform.system() == 'Windows':
            import winsound
            winsound.Beep(440, 500)  # 440 Hz frequency for 500 milliseconds
        else:
            # Use a shell beep here; it does not work on every Android environment.
            os.system('echo "\007"')
        time.sleep(2)



def get_interval_time(valoare_prestabilita=97 * 79, marja_aleatoare=10):
    base_val = botcore.safe_float(valoare_prestabilita, default=float(97 * 79))
    margin_val = abs(botcore.safe_float(marja_aleatoare, default=10.0))
    # Generate a random value in ``[-random_margin, random_margin]``.
    valoare_aleatoare = random.uniform(-margin_val, margin_val)
    interval = abs(base_val + valoare_aleatoare)
    
    return interval
 
def calculate_difference_percent(val1, val2):
    """Delegate to the shared fleet/bot botcore.diff_percent formula."""
    return botcore.diff_percent(val1, val2)

    
def value_diff_to_percent(value1, value2):
    if any(v is None or isinstance(v, bool) for v in (value1, value2)):
        return 0.0
    try:
        v1 = float(value1)
        v2 = float(value2)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not (math.isfinite(v1) and math.isfinite(v2)):
        return 0.0
    if v1 == 0.0:
        return v2
    diff = v1 - v2
    percent = (diff / v1) * 100.0
    return percent if math.isfinite(percent) else 0.0

def slope(val1, idx1, val2, idx2):
    if any(v is None or isinstance(v, bool) for v in (val1, idx1, val2, idx2)):
        return 0.0
    try:
        v1 = float(val1)
        i1 = float(idx1)
        v2 = float(val2)
        i2 = float(idx2)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not all(math.isfinite(v) for v in (v1, i1, v2, i2)) or i1 == i2:
        return 0.0
    res = (v2 - v1) / (i2 - i1)
    return res if math.isfinite(res) else 0.0

# Values lie within an interval around the percentage.
def are_difference_equal_with_aprox_proc(value1, value2, target_percent = 10.0):
    """Deprecated July 16 with no repository callers; use botcore.diff_equals_percent.
    Semantics ask whether the difference is near target_percent, not whether values are
    close. The old randomized tolerance was removed. Retain the (bool, iteration,
    tolerance) tuple for signature compatibility. target/4 deterministically approximates
    the midpoint of the old random target*0.01 to target*0.5 band."""
    tpf = botcore.safe_float(target_percent, default=10.0)
    tp = tpf if tpf >= 0.0 else 10.0
    tolerance = tp * 0.25
    ok = botcore.diff_equals_percent(value1, value2, tp, tolerance)
    return ok, 0, tolerance


def are_close_random(value1, value2, target_tolerance_percent=1.0):
    """Intentionally nondeterministic as its explicit name indicates.
    The acceptance band varies randomly in [tol*1.01, tol*1.5], so identical input can
    return either result. Never use for trading decisions; retain only for intentional
    fuzz/jitter. Use botcore.are_close for predictable behavior."""
    tf = botcore.safe_float(target_tolerance_percent, default=1.0)
    tol = tf if tf >= 0.0 else 1.0
    upper = tol * (1.0 + 0.01 + random.random() * 0.49)
    return botcore.diff_percent(value1, value2) <= upper
   

# Values are approximately equal within the specified percentage.
def are_close(value1, value2, target_tolerance_percent=1.0):
    """Deterministically delegate to the shared fleet/bot botcore.are_close.
    The pre-July-16 implementation used random.randint and could return either result
    for identical input in [tol*1.01, tol*1.5], which is unacceptable for trading."""
    return botcore.are_close(value1, value2, target_tolerance_percent)


def timestampToTime(timestamp_ms):
    if timestamp_ms is None or isinstance(timestamp_ms, bool):
        return "N/A"
    try:
        timestamp_sec = float(timestamp_ms) / 1000.0
        if not math.isfinite(timestamp_sec) or timestamp_sec < 0:
            return "N/A"
        human_readable_time = datetime.fromtimestamp(timestamp_sec)
        return human_readable_time.strftime('%Y-%m-%d %H:%M:%S')
    except (TypeError, ValueError, OverflowError, OSError):
        return "N/A"


def timeMsToHMS(timestamp_ms):
    if timestamp_ms is None or isinstance(timestamp_ms, bool):
        return "N/A"
    try:
        ts_f = float(timestamp_ms)
        if not math.isfinite(ts_f) or ts_f < 0:
            return "N/A"
        return timeToHMS(ts_f / 1000.0)
    except (TypeError, ValueError, OverflowError):
        return "N/A"


from functools import lru_cache


@lru_cache(maxsize=256)
def timeToHMS(timestamp_sec):
    if timestamp_sec is None or isinstance(timestamp_sec, bool):
        return "N/A"
    try:
        ts_f = float(timestamp_sec)
        if not math.isfinite(ts_f) or ts_f < 0:
            return "N/A"
        human_readable_time = datetime.fromtimestamp(ts_f)
        return human_readable_time.strftime('%H:%M:%S')
    except (TypeError, ValueError, OverflowError, OSError):
        return "N/A"


def _safe_seconds(max_age_seconds) -> float:
    return botcore.safe_float(max_age_seconds, default=0.0)


# Convert seconds to days.
def secondsToDays(max_age_seconds):
    # Seconds per day.
    seconds_in_a_day = 86400  # 24 hours * 60 minutes * 60 seconds
    
    # Calculate the number of days.
    days = _safe_seconds(max_age_seconds) / seconds_in_a_day
    
    return days

# Convert seconds to hours.
def secondsToHours(max_age_seconds):
    # Seconds per hour.
    seconds_in_an_hour = 3600  # 60 minutes * 60 seconds
    
    # Calculate the number of hours.
    hours = _safe_seconds(max_age_seconds) / seconds_in_an_hour
    
    return hours

# Convert seconds to minutes.
def secondsToMinutes(max_age_seconds):
    # Seconds per minute.
    seconds_in_a_minute = 60  # 60 seconds
    
    # Calculate the number of minutes.
    minutes = _safe_seconds(max_age_seconds) / seconds_in_a_minute
    
    return minutes


def asymptotic_decrease(initial_procent, expired_duration, passs, half_life_duration=24*60*60):
    """
    Gradually decreases the percentage asymptotically as `passs` increases.
    Once `passs` reaches a point where expired_duration * passs > half_life_duration 
    (24 hours as the default constant), the percentage will decrease to half of its initial value.
    This decrease continues as `passs` increases, causing the percentage to approach zero
    asymptotically but never fully reach zero.

    :param initial_procent: The initial percentage (e.g., 0.7 for 7%)
    :param expired_duration: The duration in seconds for which the percentage should decrease
    :param passs: The variable that grows over time and influences the percentage decrease
    :param half_life_duration: The default value after which the percentage is halved (24 hours in seconds)
    :return: The adjusted percentage based on `passs`
    """
    if any(isinstance(v, bool) or v is None for v in (initial_procent, expired_duration, passs, half_life_duration)):
        return 0.0
    try:
        init_f = float(initial_procent)
        exp_f = float(expired_duration)
        pass_f = float(passs)
        hl_f = float(half_life_duration)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not all(math.isfinite(v) for v in (init_f, exp_f, pass_f, hl_f)) or hl_f <= 0:
        return 0.0
    if exp_f <= 0 or pass_f <= 0:
        return init_f
    k = exp_f / hl_f  # Calculate the constant k
    return init_f / (1 + k * pass_f)  # Asymptotic decrease formula


def exponential_decrease(initial_procent, expired_duration, passs, half_life_duration=24*60*60):
    """Decreases the percentage exponentially as `passs` increases."""
    if any(isinstance(v, bool) or v is None for v in (initial_procent, expired_duration, passs, half_life_duration)):
        return 0.0
    try:
        init_f = float(initial_procent)
        exp_f = float(expired_duration)
        pass_f = float(passs)
        hl_f = float(half_life_duration)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not all(math.isfinite(v) for v in (init_f, exp_f, pass_f, hl_f)) or hl_f <= 0:
        return 0.0
    if exp_f <= 0 or pass_f <= 0:
        return init_f
    return init_f * math.exp(-(exp_f * pass_f) / hl_f)
    
def decrese_value_by_increment_exp(increment_factor, value, coeficient=0.05):
    if any(v is None or isinstance(v, bool) for v in (increment_factor, value, coeficient)):
        return 0.0, 0.0
    try:
        inc_f = float(increment_factor)
        val_f = float(value)
        coef_f = float(coeficient)
    except (TypeError, ValueError, OverflowError):
        return 0.0, 0.0
    if not all(math.isfinite(v) for v in (inc_f, val_f, coef_f)):
        return 0.0, 0.0
    denom = 1.0 + coef_f * (inc_f ** 2)
    adjustment_v1 = val_f / denom if (math.isfinite(denom) and denom != 0.0) else 0.0
    exp_arg = max(-700.0, min(700.0, -coef_f * inc_f))
    adjustment_v2 = val_f * math.exp(exp_arg)
    if not math.isfinite(adjustment_v1):
        adjustment_v1 = 0.0
    if not math.isfinite(adjustment_v2):
        adjustment_v2 = 0.0
    return adjustment_v1, adjustment_v2


def gaussian_weights_from_idx(T, idx: int):
    """
    Return ``t`` as the position axis ``[idx..T-1]`` and ``w`` as its Gaussian weights.
    """
    if T is None or idx is None or isinstance(T, bool) or isinstance(idx, bool):
        return np.array([]), np.array([])
    try:
        t_f = float(T)
        idx_f = float(idx)
    except (TypeError, ValueError, OverflowError):
        return np.array([]), np.array([])
    if not (math.isfinite(t_f) and math.isfinite(idx_f)):
        return np.array([]), np.array([])
    t_steps = min(100000, max(1, int(round(t_f)))) if t_f > 0 else 0
    idx_int = max(0, int(idx_f))
    if idx_int >= t_steps or t_steps <= 0:
        return np.array([]), np.array([])

    # Gaussian over the complete T domain.
    t_full = np.linspace(0, t_steps - 1, t_steps)
    mu = (t_steps - 1) / 2
    sigma = t_steps / 4

    w_full = np.exp(-0.5 * ((t_full - mu) / sigma) ** 2)
    w_full = w_full / w_full.sum()  # normalize to sum 1
    return t_full[idx_int:], w_full[idx_int:]


# Canonical alias for full Gaussian domain
gaussian_weights = gaussian_weights_from_idx



import base64
import nacl.signing
import re as _re

# ── Ed25519 signing ──
def _load_ed25519_signing_key():
    try:
        with open("keys/ed25519_private.pem", "r") as f:
            pem_data = f.read()
        b64match = _re.search(
            r"-----BEGIN PRIVATE KEY-----(.*?)-----END PRIVATE KEY-----",
            pem_data,
            _re.DOTALL
        )
        if not b64match:
            raise ValueError("Invalid PEM")
        der_bytes = base64.b64decode(b64match.group(1).strip())
        if len(der_bytes) < 32:
            raise ValueError("DER too short")
        
        seed = der_bytes[-32:]
        return nacl.signing.SigningKey(seed)

    except Exception as e:
        print(f"Error loading the Ed25519 key: {e}")
        return None
    
def _sign_ed25519(signing_key, payload: str) -> str:
    if signing_key is None or payload is None or isinstance(payload, bool):
        raise ValueError("signing_key and payload are required")
    raw_bytes = payload if isinstance(payload, bytes) else str(payload).encode("utf-8")
    signed = signing_key.sign(raw_bytes)
    return base64.b64encode(signed.signature).decode("ascii")


"""
the single place that decides WHERE the cache files live.

All cache files (cache_*.json/.jsonl + .meta) live in the
`cachedb/` subfolder. `cache_path(name)` prefixes a plain name with that folder
(created on demand). Names that ALREADY carry a path (absolute or with a separator
— e.g. tests, migrations) are left untouched.

The folder can be overridden through the environment variable BINANCE_CACHE_DIR.
"""
from cachepaths import (
    cache_path as _cp_cache_path,
    get_cachedb_dir,
    resolve_cachedb_filepath,
)

CACHE_DIR = get_cachedb_dir()


def cache_path(name):
    """Return the cache path under ``cachedb``.

    Preserve ``name`` when it is absolute or already contains a directory separator,
    while resolving thematic cachedb subdomains when active on disk.
    """
    if not name or isinstance(name, bool) or not isinstance(name, (str, os.PathLike)):
        return name
    name_str = str(os.fspath(name))
    if os.path.isabs(name_str) or os.sep in name_str or (os.altsep and os.altsep in name_str):
        return str(resolve_cachedb_filepath(name_str))
    return _cp_cache_path(name_str)
