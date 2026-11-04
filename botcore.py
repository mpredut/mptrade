#!/usr/bin/env python3
"""
botcore.py — SHARED core for bot utilities: logging, .env, HTTP, and time.
No external dependencies; standard library only.

The SINGLE source for functions previously duplicated and beginning to diverge in
kraken/common.py, hyperliquid/common.py, and 212trading/ipo_common.py. Each module
re-exports these functions, preserving compatibility with `from common import log`.

`now_str()` is intentionally excluded because bots differ: T212 also includes ET,
while Kraken/HL use only Bucharest time. It remains provider-specific. HTTP transport
is shared, and venue shims only re-export the JSON/form helpers defined here.
"""
from __future__ import annotations

import configparser
import fcntl
import json
import math
import numbers
import os
import re
import sys
import threading
import time as _time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, time as dt_time, timezone
from zoneinfo import ZoneInfo

HTTP_TIMEOUT = 25
BUCHAREST = ZoneInfo("Europe/Bucharest")

_LOCKS: dict = globals().get("_LOCKS") or {}   # retain open lock fds for process lifetime; prevent garbage collection
_LOG_LOCK = globals().get("_LOG_LOCK") or threading.RLock()


def log(msg: str) -> None:
    with _LOG_LOCK:
        print(f"[{datetime.now(timezone.utc).astimezone(BUCHAREST):%H:%M:%S}] {msg}", flush=True)


def now_str() -> str:
    """Return current timestamp formatted in Europe/Bucharest timezone."""
    n = datetime.now(timezone.utc)
    return f"{n.astimezone(BUCHAREST):%Y-%m-%d %H:%M:%S} Bucuresti"


def single_instance(name: str, lockdir: str = "/tmp") -> None:
    """Enforce a single instance with an exclusive flock.
    The first process holds <lockdir>/binance_<name>.lock for its lifetime; a second exits
    successfully when it cannot acquire the lock. This prevents duplicate launches and
    trades regardless of whether bots_start, healthcheck, systemd, or a user starts it."""
    path = os.path.join(lockdir, f"binance_{name}.lock")
    fd = open(path, "w")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"[{name}] already running (lock held: {path}) — exiting.", flush=True)
        sys.exit(0)
    fd.write(str(os.getpid())); fd.flush()
    _LOCKS[name] = fd   # retain the reference so the lock lasts until process exit


def _strip_env_value_quotes_and_comment(val: str) -> str:
    """Strip inline `#` comments and outer quotes from the value portion of an env setting."""
    val = val.strip()
    if val.startswith(('"', "'")):
        q = val[0]
        close_idx = val.find(q, 1)
        if close_idx != -1:
            tail = val[close_idx + 1:].strip()
            if not tail or tail.startswith("#"):
                return val[1:close_idx]
            return val.strip('"').strip("'")
        return val.strip('"').strip("'")
    return val.split("#", 1)[0].strip()


def parse_dotenv_line(raw: str, *, strict: bool = False) -> tuple[str, str] | None:
    """Parse a single ``KEY=VALUE`` (or ``export KEY=VALUE``) line, stripping inline comments and outer quotes."""
    if not isinstance(raw, str):
        return None
    line = raw.strip()
    if not line or line.startswith("#"):
        return None
    eq_idx = line.find("=")
    hash_idx = line.find("#")
    if eq_idx == -1 or (hash_idx != -1 and hash_idx < eq_idx):
        if strict:
            raise ValueError(f"Malformed configuration line missing '=': {raw.rstrip()!r}")
        return None
    key, _, val = line.partition("=")
    key = key.strip()
    if key.startswith("export") and (len(key) == 6 or key[6].isspace()):
        key = key[6:].strip()
    if not key:
        if strict:
            raise ValueError(f"Empty configuration key in line: {raw.rstrip()!r}")
        return None
    return key, _strip_env_value_quotes_and_comment(val)


def _dotenv_pairs(path: str | os.PathLike, *, strict: bool = False) -> tuple[list[tuple[str, str]], bool]:
    """Parse shared dotenv syntax once and report whether the file was read successfully."""
    if not path or isinstance(path, (bool, int)):
        return [], False
    try:
        if not os.path.exists(path):
            return [], False
    except TypeError:
        return [], False
    try:
        with open(path, "r", encoding="utf-8") as f:
            pairs = []
            for raw in f:
                parsed = parse_dotenv_line(raw, strict=strict)
                if parsed is not None:
                    pairs.append(parsed)
        return pairs, True
    except OSError as e:
        log(f"  ! cannot read {path}: {e}")
        return [], False


_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_ROOT_CONFIG_PATH = os.path.abspath(os.path.join(_REPO_ROOT, "config.env"))
_ROOT_CONFIG_KEYS: set[str] = (
    globals()["_ROOT_CONFIG_KEYS"] if isinstance(globals().get("_ROOT_CONFIG_KEYS"), set) else set()
)
_ROOT_CONFIG_VALUES: dict[str, str] = (
    globals()["_ROOT_CONFIG_VALUES"] if isinstance(globals().get("_ROOT_CONFIG_VALUES"), dict) else {}
)
_LOADED_DOTENV_PATHS: set[str] = (
    globals()["_LOADED_DOTENV_PATHS"] if isinstance(globals().get("_LOADED_DOTENV_PATHS"), set) else set()
)


def _log_dotenv_loaded_once(path: str, *, override: bool = False) -> None:
    abs_path = os.path.abspath(path)
    if override or abs_path not in _LOADED_DOTENV_PATHS:
        _LOADED_DOTENV_PATHS.add(abs_path)
        log(f"  .env loaded from {path}")


def _is_unmodified_root_key(key: str) -> bool:
    if key not in _ROOT_CONFIG_KEYS:
        return False
    if key not in os.environ or os.environ.get(key) != _ROOT_CONFIG_VALUES.get(key):
        _ROOT_CONFIG_KEYS.discard(key)
        _ROOT_CONFIG_VALUES.pop(key, None)
        return False
    return True


def _parse_bool_flag(val, default: bool | None = False):
    if default is None:
        fb = None
    elif isinstance(default, bool):
        fb = default
    else:
        fb = _parse_bool_flag(default, False)
    if isinstance(val, bool):
        return val
    if val is None or isinstance(val, (dict, list, tuple, set, frozenset)):
        return fb
    if isinstance(val, (int, float)):
        if isinstance(val, float) and not math.isfinite(val):
            return fb
        return val != 0
    if isinstance(val, str):
        s = val.split("#", 1)[0].split(";", 1)[0].strip().strip('"').strip("'").strip().lower()
        if not s or s in {"none", "null"}:
            return fb
        if s == "nan":
            return False if fb is not None else None
        if s in {"1", "1.0", "true", "yes", "on", "y", "t", "da"}:
            return True
        if s in {"0", "0.0", "false", "no", "off", "n", "f"}:
            return False
        try:
            fv = float(s)
            if not math.isfinite(fv):
                return fb
            return fv != 0.0
        except (TypeError, ValueError):
            return fb
    return fb


def _set_dotenv_env_key(key: str, value: str, *, is_root: bool) -> None:
    os.environ[key] = value
    if is_root:
        _ROOT_CONFIG_KEYS.add(key)
        _ROOT_CONFIG_VALUES[key] = value
    else:
        _ROOT_CONFIG_KEYS.discard(key)
        _ROOT_CONFIG_VALUES.pop(key, None)


def load_dotenv(path: str | os.PathLike | None = None, override: bool = False) -> None:
    """Load KEY=VALUE entries from ``path`` (defaulting to root ``config.env`` when ``None``) without overriding existing env."""
    target = _ROOT_CONFIG_PATH if path is None else os.fspath(path)
    override = _parse_bool_flag(override, False)
    pairs, loaded = _dotenv_pairs(target)
    if not loaded:
        return
    is_root = os.path.abspath(target) == _ROOT_CONFIG_PATH
    for key, value in pairs:
        if key and (override or key not in os.environ or (not is_root and _is_unmodified_root_key(key))):
            _set_dotenv_env_key(key, value, is_root=is_root)
    _log_dotenv_loaded_once(target, override=bool(override))


def load_env_stack(
    env_file: str | os.PathLike | None = None,
    config_name: str | os.PathLike = "config.env",
    override: bool = False,
) -> None:
    """Load secrets/overrides and the adjacent versioned configuration.

    Local venue configurations take precedence over any globally loaded root config.env.
    Precedence (highest to lowest):
      1. Explicit process environment (unless override=True)
      2. Secrets/profile env file (env_file)
      3. Venue versioned configuration (config_name)
      4. Global root config.env defaults
    """
    override = _parse_bool_flag(override, False)
    eff_env = os.path.join(_REPO_ROOT, ".env") if env_file is None else os.fspath(env_file)
    env_path = os.path.abspath(eff_env)
    cfg_str = os.fspath(config_name) if config_name else "config.env"
    cfg_path = (
        os.path.abspath(cfg_str)
        if (os.path.isabs(cfg_str) or os.path.dirname(cfg_str))
        else os.path.join(os.path.dirname(env_path), cfg_str)
    )

    unmodified_root_keys = {k for k in tuple(_ROOT_CONFIG_KEYS) if _is_unmodified_root_key(k)}
    protected = set() if override else (set(os.environ.keys()) - unmodified_root_keys)

    for path, is_root_cfg in ((cfg_path, os.path.abspath(cfg_path) == _ROOT_CONFIG_PATH), (env_path, False)):
        pairs, loaded = _dotenv_pairs(path)
        if loaded:
            for key, value in pairs:
                if key and key not in protected:
                    _set_dotenv_env_key(key, value, is_root=is_root_cfg)
            _log_dotenv_loaded_once(path, override=bool(override))


def load_root_config(*, include_secrets: bool = False, override: bool = False) -> None:
    """Load repository root ``config.env`` (and root ``.env`` when ``include_secrets=True``) via the canonical SSOT."""
    if include_secrets:
        load_env_stack(os.path.join(_REPO_ROOT, ".env"), _ROOT_CONFIG_PATH, override=override)
    else:
        load_dotenv(_ROOT_CONFIG_PATH, override=override)


def parse_dotenv(
    path: str | os.PathLike,
    *,
    upper_keys: bool = False,
    lower_keys: bool = False,
    coerce_bools: bool = False,
    coerce_floats: bool = False,
    strict: bool = False,
) -> dict[str, object]:
    """Parse a ``KEY=VALUE`` dotenv/conf file into a dictionary without changing ``os.environ``."""
    pairs, _ = _dotenv_pairs(path, strict=strict)
    out: dict[str, object] = {}
    for k, v in pairs:
        if not k:
            continue
        key = k.upper() if upper_keys else (k.lower() if lower_keys else k)
        val: object = v
        if coerce_bools:
            v_low = v.strip().lower()
            bool_true = {"true", "yes", "on"} if coerce_floats else {"true", "1", "yes", "on"}
            bool_false = {"false", "no", "off"} if coerce_floats else {"false", "0", "no", "off"}
            if v_low in bool_true:
                val = True
            elif v_low in bool_false:
                val = False
        if coerce_floats and isinstance(val, str):
            fv = safe_float(val, default=None)
            if fv is not None:
                val = fv
        out[key] = val
    return out


def safe_float(val: object, default: float | None = 0.0) -> float | None:
    """Safely coerce ``val`` to finite float, rejecting booleans, None, non-finite values, and non-numeric objects."""
    if val is None or isinstance(val, bool):
        return default
    if (
        not isinstance(val, (int, float, str, bytes, bytearray, numbers.Real))
        and type(val).__name__ != "Decimal"
    ):
        return default
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError, OverflowError):
        return default


def clean_finite_floats(values: object) -> list[float]:
    """Filter an iterable into a list of finite floats, rejecting None, booleans, strings, and non-iterables."""
    if values is None or isinstance(values, (bool, str, bytes, bytearray)):
        return []
    try:
        return [vf for x in values if (vf := safe_float(x, default=None)) is not None]
    except TypeError:
        return []


def safe_pos_float(val: object, default: float | None = 0.0) -> float | None:
    """Safely coerce ``val`` to a strictly positive finite float, otherwise return ``default``."""
    f = safe_float(val, default=None)
    return f if (f is not None and f > 0.0) else default


def safe_int(val: object, default: int | None = 0) -> int | None:
    """Safely coerce ``val`` to an integer, rejecting booleans, None, non-finite values, and non-numeric objects."""
    if val is None or isinstance(val, bool):
        return default
    if isinstance(val, int):
        return val
    if isinstance(val, (str, bytes, bytearray)):
        try:
            return int(val, 10)
        except (TypeError, ValueError):
            pass
    f = safe_float(val, default=None)
    if f is None or not math.isfinite(f):
        return default
    try:
        return int(f)
    except (ValueError, OverflowError):
        return default


def safe_nonneg_float(val: object, default: float | None = 0.0) -> float | None:
    """Safely coerce ``val`` to a non-negative finite float (>= 0.0), otherwise return ``default``."""
    f = safe_float(val, default=None)
    return f if (f is not None and f >= 0.0) else default


def safe_pos_int(val: object, default: int | None = 1) -> int | None:
    """Safely coerce ``val`` to a strictly positive integer (>= 1), otherwise return ``default``."""
    i = safe_int(val, default=None)
    return i if (i is not None and i >= 1) else default


def safe_nonneg_int(val: object, default: int | None = 0) -> int | None:
    """Safely coerce ``val`` to a non-negative integer (>= 0), otherwise return ``default``."""
    i = safe_int(val, default=None)
    return i if (i is not None and i >= 0) else default


def exact_int(val: object, default: int | None = 0) -> int | None:
    """Safely coerce ``val`` to an exact integer without truncating fractional floats or strings."""
    if val is None or isinstance(val, bool):
        return default
    if isinstance(val, int):
        return val
    if isinstance(val, (str, bytes, bytearray)):
        try:
            return int(val, 10)
        except (TypeError, ValueError):
            pass
    f = safe_float(val, default=None)
    if f is None or not f.is_integer():
        return default
    try:
        return int(f)
    except (ValueError, OverflowError):
        return default


def exact_pos_int(val: object, default: int | None = 1) -> int | None:
    """Safely coerce ``val`` to a strictly positive exact integer (>= 1), otherwise return ``default``."""
    i = exact_int(val, default=None)
    return i if (i is not None and i >= 1) else default


def exact_nonneg_int(val: object, default: int | None = 0) -> int | None:
    """Safely coerce ``val`` to a non-negative exact integer (>= 0), otherwise return ``default``."""
    i = exact_int(val, default=None)
    return i if (i is not None and i >= 0) else default


def strict_ascii_int(
    val: object,
    default: int | None = None,
    *,
    min_value: int | None = None,
    allow_float_str: bool = False,
) -> int | None:
    """Coerce ``val`` (int, integral float, or ASCII digit string) to int, rejecting booleans and non-ASCII digits."""
    if isinstance(val, bool) or not isinstance(val, (int, float, str)):
        return default
    if isinstance(val, str):
        s = val.strip()
        if allow_float_str and s.endswith(".0"):
            s = s[:-2]
        cleaned = s.lstrip("+-")
        if not cleaned or not cleaned.isascii() or not cleaned.isdigit():
            return default
        parsed = exact_int(s, default=None)
    else:
        parsed = exact_int(val, default=None)
    if parsed is None:
        return default
    if min_value is not None and parsed < min_value:
        return default
    return parsed


def require_finite_float(
    val: object,
    name: str = "value",
    *,
    min_value: float | None = None,
    error_msg: str | None = None,
) -> float:
    """Coerce ``val`` to a finite float (optionally ``>= min_value``) or raise ``ValueError``."""
    f = safe_float(val, default=None)
    if f is None or (min_value is not None and f < min_value):
        raise ValueError(
            error_msg
            or (f"{name} must be finite and >= {min_value}" if min_value is not None else f"{name} must be finite")
        )
    return f


def require_pos_float(
    val: object,
    name: str = "value",
    *,
    min_value: float | None = None,
    error_msg: str | None = None,
) -> float:
    """Coerce ``val`` to a finite positive float (optionally ``>= min_value``) or raise ``ValueError``."""
    f = safe_pos_float(val, default=None)
    if f is None or (min_value is not None and f < min_value):
        raise ValueError(
            error_msg
            or (
                f"{name} must be finite and >= {min_value}"
                if min_value is not None
                else f"{name} must be finite and positive"
            )
        )
    return f


def require_int(
    val: object,
    name: str = "value",
    *,
    min_value: int | None = None,
    error_msg: str | None = None,
) -> int:
    """Coerce ``val`` to an exact integer (optionally ``>= min_value``) or raise ``ValueError``."""
    parsed = exact_int(val, default=None)
    if parsed is None or (min_value is not None and parsed < min_value):
        raise ValueError(error_msg or f"invalid {name}")
    return parsed


def obj_attr(obj: object, key: str, default: object = None) -> object:
    """Extract ``key`` uniformly from either a ``dict`` or an object attribute."""
    if obj is None or isinstance(obj, (str, bytes, bytearray)):
        return default
    val = obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)
    return default if val is None else val


def obj_float(obj: object, key: str, default: float = 0.0) -> float:
    """Extract ``key`` from ``obj`` (dict or object) and coerce to finite float."""
    return safe_float(obj_attr(obj, key, None), default)


def clean_json_value(val: object, *, stringify_keys: bool = True) -> object:
    """Recursively sanitize non-finite floats (NaN/Inf -> None) and optionally normalize dict keys to strings."""
    if isinstance(val, float):
        return safe_float(val, default=None)
    if isinstance(val, dict):
        return {
            (str(k) if stringify_keys else k): clean_json_value(v, stringify_keys=stringify_keys)
            for k, v in val.items()
        }
    if isinstance(val, (list, tuple, set, frozenset)):
        return [clean_json_value(v, stringify_keys=stringify_keys) for v in val]
    return val


def _clean_env_str(val: object) -> str:
    """Strip inline `#` comment and surrounding whitespace from a raw env value."""
    if val is None or isinstance(val, bool):
        return ""
    return _strip_env_value_quotes_and_comment(str(val))


def float_env(key: str, env: dict | float | None = None, default: float | None = None) -> float | None:
    """Read a float from os.environ or an injected dictionary, ignoring inline comments.
    Supports both float_env(key, env_dict) and float_env(key, default_val)."""
    if env is not None and not hasattr(env, "get"):
        default, env = env, None
    src = os.environ if env is None else env
    raw = _clean_env_str(src.get(key))
    return safe_float(raw, default=default) if raw else default


def pos_float_env(key: str, default: float, env: dict | None = None) -> float:
    """Read a strictly positive float from os.environ or an injected dictionary, falling back to default."""
    return safe_pos_float(float_env(key, env=env), default)


def int_env(key: str, env: dict | int | None = None, default: int | None = None) -> int | None:
    """Read an integer from os.environ or an injected dictionary, ignoring inline comments.
    Supports both int_env(key, env_dict) and int_env(key, default_val)."""
    if env is not None and not hasattr(env, "get"):
        default, env = env, None
    src = os.environ if env is None else env
    raw = _clean_env_str(src.get(key))
    return exact_int(raw, default=default) if raw else default


def required_env(key: str, env: dict | None = None) -> str:
    """Return a mandatory setting, rejecting missing, empty, or comment-only values."""
    src = os.environ if env is None else env
    val = src.get(key)
    if isinstance(val, bool):
        raise ValueError(f"Invalid mandatory setting: {key}={val!r}")
    raw = _clean_env_str(val)
    if not raw:
        raise ValueError(f"Missing or empty mandatory setting: {key}")
    return raw


def defined_env(key: str, env: dict | None = None) -> str:
    """Return a setting that must exist but may intentionally be empty."""
    src = os.environ if env is None else env
    if key not in src:
        raise ValueError(f"Missing mandatory setting: {key}")
    val = src[key]
    if isinstance(val, bool):
        raise ValueError(f"Invalid setting: {key}={val!r}")
    return _clean_env_str(val)


def required_float_env(key: str, env: dict | None = None) -> float:
    """Return a mandatory finite floating-point setting."""
    src = os.environ if env is None else env
    if isinstance(src.get(key), bool):
        raise ValueError(f"Invalid numeric setting: {key}={src.get(key)!r}")
    raw = required_env(key, env)
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid numeric setting: {key}={raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Non-finite numeric setting: {key}={raw!r}")
    return value


def required_int_env(key: str, env: dict | None = None) -> int:
    """Return a mandatory integer setting without silently truncating fractions."""
    src = os.environ if env is None else env
    raw_val = src.get(key)
    if isinstance(raw_val, bool):
        raise ValueError(f"Invalid numeric setting: {key}={raw_val!r}")
    if isinstance(raw_val, int):
        return raw_val
    raw = required_env(key, env)
    iv = exact_int(raw, default=None)
    if iv is not None:
        return iv
    value = required_float_env(key, env)
    if not value.is_integer():
        raise ValueError(f"Invalid integer setting: {key}={value!r}")
    return int(value)


def required_bool_env(key: str, env: dict | None = None) -> bool:
    """Return a mandatory strict boolean setting."""
    src = os.environ if env is None else env
    raw_val = src.get(key)
    if isinstance(raw_val, bool):
        return raw_val
    raw = required_env(key, env).lower()
    if raw not in {"true", "false"}:
        raise ValueError(f"Invalid boolean setting: {key}={raw!r}; expected true or false")
    return raw == "true"


def optional_env(key: str, default: str | dict | None = "", env: dict | None = None) -> str:
    """Read an optional string setting from os.environ or an injected dictionary, stripping inline comments."""
    if isinstance(default, dict) and env is None:
        env, default = default, ""
    default_str = "" if default is None else str(default)
    src = os.environ if env is None else env
    raw = _clean_env_str(src.get(key))
    return raw if raw else default_str


def env_value(folder: str | os.PathLike, key: str) -> str | None:
    """Read one key from ``.env``/``config.env``/``runtime.env`` in ``folder`` without mutating ``os.environ``."""
    if isinstance(folder, os.PathLike):
        folder = os.fspath(folder)
    if not isinstance(folder, str) or not folder.strip() or not isinstance(key, str) or not key.strip():
        return None
    for fname in (".env", "config.env", "runtime.env"):
        path = os.path.join(folder, fname)
        if not os.path.exists(path):
            continue
        try:
            parsed = parse_dotenv(path)
            if key in parsed:
                val = str(parsed[key]).strip()
                return val or None
        except Exception:
            continue
    return None


def env_values_map(folder: str | os.PathLike, names: tuple[str, ...] | list[str]) -> dict[str, str | None]:
    """Resolve each key from ``os.environ`` first, falling back to ``env_value(folder, name)``."""
    out: dict[str, str | None] = {}
    for name in names:
        raw_env = os.environ.get(name)
        clean_env = raw_env.strip() if isinstance(raw_env, str) else None
        out[name] = clean_env if clean_env else env_value(folder, name)
    return out


_UNSET = globals().get("_UNSET") or object()


def bool_env(
    key: str | os.PathLike,
    second: object = _UNSET,
    env: dict | object = _UNSET,
    *,
    default: object = _UNSET,
    folder: str | os.PathLike | None = None,
) -> bool:
    """Read an optional boolean setting from os.environ, an injected dict, or a folder's .env/config.env."""
    if isinstance(key, os.PathLike) and isinstance(second, str):
        folder, key, second = key, second, _UNSET
    eff_default: object = False if default is _UNSET else default
    eff_env: dict | None = env if isinstance(env, dict) else None
    if second is not _UNSET:
        if isinstance(second, dict) and eff_env is None:
            eff_env = second
        elif (
            second is None
            or isinstance(second, os.PathLike)
            or (isinstance(second, str) and _parse_bool_flag(second, None) is None)
        ):
            folder = second
            if default is _UNSET and env is not _UNSET and not isinstance(env, dict):
                eff_default = env
        elif default is _UNSET:
            eff_default = second
    def_flag = bool(_parse_bool_flag(eff_default, False))
    if not isinstance(key, str) or not key.strip():
        return def_flag
    src = os.environ if eff_env is None else eff_env
    val = src.get(key)
    if (val is None or (isinstance(val, str) and not val.strip())) and folder is not None:
        val = env_value(folder, key)
    return bool(_parse_bool_flag(val, def_flag))


def resolve_cli_arg(
    argv: list[str] | tuple[str, ...] | None,
    flags: str | tuple[str, ...],
    default: str | None = None,
) -> str | None:
    """Extract the last value of ``flags`` (e.g. ``'--env-file'``) from ``argv`` before full argparse."""
    args = sys.argv if argv is None else argv
    flag_set = (flags,) if isinstance(flags, str) else tuple(flags)
    result = default
    for i, arg in enumerate(args):
        if not isinstance(arg, str):
            continue
        if arg in flag_set and i + 1 < len(args) and isinstance(args[i + 1], str):
            result = args[i + 1]
        else:
            for fl in flag_set:
                prefix = f"{fl}="
                if arg.startswith(prefix):
                    result = arg[len(prefix):]
                    break
    return result


def http_request(
    method: str,
    url: str,
    headers: dict | None = None,
    payload: dict | None = None,
    *,
    form: dict | None = None,
) -> tuple[int, bytes]:
    """Shared standard-library HTTP transport with a ``(status, body)`` contract.

    Serialize ``payload`` as JSON and ``form`` as application/x-www-form-urlencoded;
    the two forms are mutually exclusive. HTTP errors retain status/body, while
    transport errors fail closed as ``(0, b"")``, matching the replaced venue helpers.
    """
    if payload is not None and form is not None:
        raise ValueError("payload and form are mutually exclusive")

    request_headers = dict(headers) if isinstance(headers, dict) else {}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")
    elif form is not None:
        data = urllib.parse.urlencode(form).encode()
        request_headers.setdefault(
            "Content-Type", "application/x-www-form-urlencoded"
        )

    if not method or isinstance(method, bool) or not isinstance(method, str) or not method.strip():
        return 0, b""
    if not url or isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        return 0, b""
    verb = method.strip().upper()
    try:
        req = urllib.request.Request(
            url.strip(), data=data, headers=request_headers, method=verb,
        )
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:  # noqa: BLE001
        log(f"  ! network error {verb}: {e}")
        return 0, b""


def http_get(url: str, headers: dict | None = None) -> tuple[int, bytes]:
    return http_request("GET", url, headers=headers)


def http_post_json(
    url: str, payload: dict, headers: dict | None = None,
) -> tuple[int, bytes]:
    return http_request("POST", url, headers=headers, payload=payload)


def http_post_form(
    url: str, data: dict, headers: dict | None = None,
) -> tuple[int, bytes]:
    return http_request("POST", url, headers=headers, form=data)


def _urlopen_json(req: urllib.request.Request, timeout: float) -> object:
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        return json.loads(raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw)


def _fetch_public_json_impl(
    url: str,
    *,
    timeout: float = 5.0,
    user_agent: str = "mptrade-intelligence/1.0",
    headers: dict | None = None,
) -> object:
    """Fetch and decode a JSON payload from a public HTTP GET endpoint."""
    req_headers = {"User-Agent": user_agent}
    if isinstance(headers, dict):
        req_headers.update(headers)
    return _urlopen_json(urllib.request.Request(url, headers=req_headers), timeout)


fetch_public_json = globals().get("fetch_public_json") or _fetch_public_json_impl


def fetch_json_post(
    url: str,
    payload: dict | bytes | None = None,
    *,
    timeout: float = 5.0,
    headers: dict | None = None,
) -> object:
    """Send an HTTP POST request with a JSON body and decode the JSON response."""
    req_headers = {"Content-Type": "application/json"}
    if isinstance(headers, dict):
        req_headers.update(headers)
    if payload is None:
        data = b"{}"
    elif isinstance(payload, (bytes, bytearray)):
        data = bytes(payload)
    else:
        data = json.dumps(payload).encode("utf-8")
    return _urlopen_json(urllib.request.Request(url, data=data, headers=req_headers), timeout)


# ── DETERMINISTIC percentage-based approximate comparisons ───────────────────
# Single source for the fleet and bots. Replaces utils.are_close, whose random.randint
# tolerance loop could return True OR False for the same input in [tol*1.01, tol*1.5],
# which is unacceptable for trading decisions.

def diff_percent(value1: float, value2: float) -> float:
    """Return symmetric percentage difference relative to the values' absolute mean."""
    v1 = safe_float(value1, default=None)
    v2 = safe_float(value2, default=None)
    if v1 is None or v2 is None:
        return float("nan")
    if v1 == v2:
        return 0.0
    denom = (abs(v1) + abs(v2)) / 2.0
    if denom == 0.0:
        return 0.0
    return abs(v1 - v2) / denom * 100.0


def are_close(value1: float, value2: float, tolerance_percent: float = 1.0) -> bool:
    """Return deterministically whether values differ by at most tolerance_percent.

    For price thresholds, are_close(price, threshold, 0.05) treats a price within
    0.05% as reached, avoiding missed entries by a few cents."""
    tol = safe_nonneg_float(tolerance_percent, default=None)
    if tol is None:
        return False
    diff_f = safe_nonneg_float(diff_percent(value1, value2), default=None)
    return diff_f is not None and diff_f <= tol


def diff_equals_percent(value1: float, value2: float, target_percent: float,
                        tolerance_percent: float = 1.0) -> bool:
    """Return whether the percentage DIFFERENCE is near target_percent using a
    deterministic two-sided band. Unlike are_close, this asks whether values differ by
    about X percent, e.g. whether price fell about 10%. Replaces the randomized
    utils.are_difference_equal_with_aprox_proc implementation."""
    target = safe_nonneg_float(target_percent, default=None)
    tol = safe_nonneg_float(tolerance_percent, default=None)
    if target is None or tol is None:
        return False
    diff_f = safe_nonneg_float(diff_percent(value1, value2), default=None)
    return diff_f is not None and abs(diff_f - target) <= tol


ET = ZoneInfo("America/New_York")


def is_us_rth_open(now_ts: float | None = None) -> bool:
    """Return True if US equities regular trading hours (09:30-16:00 America/New_York, Mon-Fri) are open."""
    ts = resolve_now_ts(now_ts)
    try:
        ny_dt = datetime.fromtimestamp(ts, tz=ET)
    except (ValueError, OverflowError, OSError):
        ny_dt = datetime.fromtimestamp(_time.time(), tz=ET)
    if ny_dt.weekday() >= 5:
        return False
    return dt_time(9, 30) <= ny_dt.time() < dt_time(16, 0)


_INI_CONFIG_CACHE: dict[str, tuple[tuple[int, int], configparser.ConfigParser]] = (
    globals().get("_INI_CONFIG_CACHE") or {}
)
_INI_CONFIG_LOCK = globals().get("_INI_CONFIG_LOCK") or threading.RLock()


def load_ini_config(
    path: str | os.PathLike,
    *,
    must_exist: bool = True,
    inline_comments: bool = True,
) -> configparser.ConfigParser:
    """Load and cache an INI ``.conf`` file by ``(mtime_ns, size)`` using ``ConfigParser(interpolation=None)``."""
    if isinstance(path, bool) or not isinstance(path, (str, os.PathLike)) or not str(path).strip():
        raise ValueError(f"Invalid INI configuration path: {path!r}")
    abs_path = os.path.abspath(os.fspath(path))
    if not os.path.exists(abs_path):
        if must_exist:
            raise FileNotFoundError(f"Missing mandatory configuration file: {path}")
        return configparser.ConfigParser(
            interpolation=None,
            inline_comment_prefixes=("#", ";") if inline_comments else (),
        )
    st = os.stat(abs_path)
    sig = (st.st_mtime_ns, st.st_size, bool(inline_comments))
    with _INI_CONFIG_LOCK:
        cached = _INI_CONFIG_CACHE.get(abs_path)
        if cached is not None and cached[0] == sig:
            return cached[1]
        cp = configparser.ConfigParser(
            interpolation=None,
            inline_comment_prefixes=("#", ";") if inline_comments else (),
        )
        with open(abs_path, "r", encoding="utf-8") as handle:
            cp.read_file(handle)
        lru_dict_put(_INI_CONFIG_CACHE, abs_path, (sig, cp), 64)
        return cp


def parse_env_file(env_path: str | os.PathLike) -> dict[str, str]:
    """Parse a key=value env/conf file into an uppercase-keyed dictionary without mutating os.environ."""
    return parse_dotenv(env_path, upper_keys=True)  # type: ignore[return-value]


def load_env_file_defaults(env_path: str | os.PathLike) -> dict[str, str]:
    """Parse an env file and set non-STRAT_* keys in os.environ via setdefault."""
    parsed = parse_env_file(env_path)
    for key_u, val in parsed.items():
        if not key_u.startswith("STRAT_"):
            os.environ.setdefault(key_u, val)
    return parsed


def load_key_value_conf(
    workspace_dir: str | os.PathLike = _REPO_ROOT,
    root_dir: str | os.PathLike = _REPO_ROOT,
    *,
    filenames: tuple[str, ...] = ("order_guard.conf", "config.env", ".env"),
    lower_values: bool = False,
) -> dict[str, str]:
    """Load key=value settings from ``filenames`` (``root_dir`` overlaid by ``workspace_dir``)."""
    conf: dict[str, str] = {}
    eff_root = os.fspath(root_dir) if root_dir else _REPO_ROOT
    ws_str = os.fspath(workspace_dir) if workspace_dir else ""
    search_dirs = [eff_root]
    if ws_str and os.path.abspath(ws_str) != os.path.abspath(eff_root):
        search_dirs.append(ws_str)
    for base_dir in search_dirs:
        for fname in filenames:
            for k_l, v_s in parse_dotenv(os.path.join(base_dir, fname), lower_keys=True).items():
                val_str = str(v_s).strip().strip("\"'")
                conf[k_l] = val_str.lower() if lower_values else str(v_s)
    return conf


def extract_current_price_item(item: object) -> float | None:
    """Extract the latest positive finite price from any ``cache_current_price.json`` item format."""
    if item is None or isinstance(item, bool):
        return None
    if isinstance(item, (int, float, str)):
        return safe_pos_float(item, None)
    if isinstance(item, dict):
        return safe_pos_float(item.get("price") if item.get("price") is not None else item.get("c"), None)
    if isinstance(item, (list, tuple)):
        for entry in reversed(item):
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                px_f = safe_pos_float(entry[1], None)
                if px_f is not None:
                    return px_f
            else:
                px_f = safe_pos_float(entry, None)
                if px_f is not None:
                    return px_f
    return None


def conf_or_override(conf: dict[str, str], key: str, override: object = None) -> object:
    """Return ``override`` when not None, else require ``conf[key]``."""
    if override is not None:
        return override
    if key not in conf:
        raise KeyError(f"Missing required '{key}' in order_guard.conf")
    return conf[key]


def parse_notify_enabled_flag(notify_enabled: object, conf: dict[str, str], conf_key: str) -> bool:
    """Validate and coerce an explicit ``notify_enabled`` override or ``conf[conf_key]`` into a bool."""
    if isinstance(notify_enabled, str) and not notify_enabled.strip():
        return False
    raw = conf_or_override(conf, conf_key, notify_enabled)
    res = _parse_bool_flag(raw, None)
    if res is not None:
        return res
    label = "notify_enabled" if notify_enabled is not None else conf_key
    raise ValueError(f"Invalid {label} {raw!r}")


_LOCAL_MODULE_LOCK = globals().get("_LOCAL_MODULE_LOCK") or threading.RLock()


def ensure_local_module_priority(directory: str, *module_names: str) -> None:
    """Ensure ``directory`` is first on ``sys.path`` (with repo root available) and evict foreign cached modules."""
    with _LOCAL_MODULE_LOCK:
        abs_dir = os.path.abspath(directory)
        if _REPO_ROOT not in sys.path:
            sys.path.insert(0, _REPO_ROOT)
        if sys.path[:1] != [abs_dir]:
            if abs_dir in sys.path:
                sys.path.remove(abs_dir)
            sys.path.insert(0, abs_dir)
        for mod_name in module_names:
            mod_file = getattr(sys.modules.get(mod_name), "__file__", None)
            if isinstance(mod_file, str) and mod_file and os.path.dirname(os.path.abspath(mod_file)) != abs_dir:
                sys.modules.pop(mod_name, None)


_AMBIGUOUS_LOCAL_MODULES = (
    "strategy",
    "market_data",
    "replay",
    "common",
    "hl_client",
    "t212_client",
    "ipo_common",
)


_LEGACY_ARCHIVE_DIR: str | None = None


def _resolve_archived_module_dir(abs_dir: str, stem: str) -> str:
    global _LEGACY_ARCHIVE_DIR
    if os.path.isfile(os.path.join(abs_dir, f"{stem}.py")):
        return abs_dir
    archive_tar = os.path.join(_REPO_ROOT, "archive", "legacy_archive.tar.gz")
    if not os.path.isfile(archive_tar):
        return abs_dir
    rel = os.path.relpath(abs_dir, _REPO_ROOT)
    if not (rel == "archive" or rel.startswith("archive" + os.sep)):
        return abs_dir
    if _LEGACY_ARCHIVE_DIR is None or not os.path.isdir(_LEGACY_ARCHIVE_DIR):
        import atexit
        import shutil
        import tarfile
        import tempfile
        _LEGACY_ARCHIVE_DIR = tempfile.mkdtemp(prefix="mptrade_legacy_archive_")
        atexit.register(shutil.rmtree, _LEGACY_ARCHIVE_DIR, True)
        with tarfile.open(archive_tar, "r:gz") as tf:
            tf.extractall(_LEGACY_ARCHIVE_DIR, filter="data")
    extracted_dir = os.path.join(_LEGACY_ARCHIVE_DIR, rel)
    return extracted_dir if os.path.isfile(os.path.join(extracted_dir, f"{stem}.py")) else abs_dir


def load_local_module(
    directory: str,
    name: str,
    *,
    module_key: str | None = None,
    register: bool = True,
    reload: bool = False,
):
    """Load a sibling module ``<name>.py`` from ``directory`` without colliding with other venue modules."""
    with _LOCAL_MODULE_LOCK:
        stem = name[:-3] if name.endswith(".py") else name
        abs_dir = _resolve_archived_module_dir(os.path.abspath(directory), stem)
        key = module_key or stem
        if register and not reload:
            mod = sys.modules.get(key)
            mod_file = getattr(mod, "__file__", None) if mod is not None else None
            if (
                mod is not None
                and getattr(mod, "__mptrade_exec_done__", True)
                and (
                    not isinstance(mod_file, str)
                    or not mod_file
                    or os.path.dirname(os.path.abspath(mod_file)) == abs_dir
                )
            ):
                return mod
        import importlib.util
        orig_sys_path = list(sys.path)
        saved_ambiguous: dict[str, object] = {}
        for amb_name in _AMBIGUOUS_LOCAL_MODULES:
            if amb_name in sys.modules:
                amb_file = getattr(sys.modules[amb_name], "__file__", None)
                if isinstance(amb_file, str) and amb_file and os.path.dirname(os.path.abspath(amb_file)) != abs_dir:
                    saved_ambiguous[amb_name] = sys.modules.pop(amb_name)
        succeeded = False
        try:
            while abs_dir in sys.path:
                sys.path.remove(abs_dir)
            while _REPO_ROOT in sys.path:
                sys.path.remove(_REPO_ROOT)
            sys.path.insert(0, _REPO_ROOT)
            if "hyperliquid" in abs_dir:
                hl_dir = os.path.join(_REPO_ROOT, "hyperliquid")
                if os.path.isdir(hl_dir) and hl_dir not in sys.path:
                    sys.path.insert(0, hl_dir)
            sys.path.insert(0, abs_dir)
            spec = importlib.util.spec_from_file_location(key, os.path.join(abs_dir, f"{stem}.py"))
            mod = importlib.util.module_from_spec(spec)
            setattr(mod, "__mptrade_exec_done__", False)
            if register:
                sys.modules[key] = mod
            try:
                spec.loader.exec_module(mod)
                setattr(mod, "__mptrade_exec_done__", True)
            except Exception:
                if register:
                    sys.modules.pop(key, None)
                raise
            succeeded = True
            return mod
        finally:
            sys.path[:] = orig_sys_path
            for amb_name, saved_mod in saved_ambiguous.items():
                if not (succeeded and register and amb_name == key):
                    sys.modules[amb_name] = saved_mod


def sanitize_heartbeat_status(status: object) -> tuple[float, dict]:
    """Sanitize a daemon/module cycle status dict, replacing non-finite floats with None."""
    raw = status if isinstance(status, dict) else {}
    hb_ts = safe_pos_float(raw.get("ts"), _time.time())
    safe_status = {
        k: (hb_ts if k == "ts" else (None if isinstance(v, float) and safe_float(v, default=None) is None else v))
        for k, v in raw.items()
    }
    return hb_ts, safe_status


def build_heartbeat_payload(
    status: object = None,
    *,
    symbols: list[str] | tuple[str, ...] | None = None,
    intervals: dict | None = None,
    extra: dict | None = None,
    pid: int | None = None,
) -> dict[str, object]:
    """Build the canonical module heartbeat payload used across all bots, daemons, and supervisors."""
    hb_ts, safe_status = sanitize_heartbeat_status(status)
    payload: dict[str, object] = {
        "pid": os.getpid() if pid is None else int(pid),
        "status": "alive",
        "ts": hb_ts,
    }
    if symbols is not None:
        payload["symbols"] = list(symbols)
    payload["intervals"] = clean_json_value(intervals) if isinstance(intervals, dict) else {}
    payload["last_run"] = safe_status
    if isinstance(extra, dict):
        cleaned_extra = clean_json_value(extra)
        if isinstance(cleaned_extra, dict):
            payload.update(cleaned_extra)
    return payload


def touch_heartbeat_file(
    path: str | os.PathLike,
    *,
    last_ts: float = float("-inf"),
    interval_sec: float = 30.0,
    force: object = False,
    now: float | None = None,
    status: object = None,
    symbols: list[str] | tuple[str, ...] | None = None,
    intervals: dict | None = None,
    extra: dict | None = None,
) -> float:
    """Write canonical heartbeat JSON and touch ``path`` if ``force`` is truthy or ``interval_sec`` has elapsed; return updated ``last_ts``."""
    force_flag = _parse_bool_flag(force, default=False)
    now_val = safe_float(now, default=None)
    now_m = _time.monotonic() if now_val is None else now_val
    if not force_flag and now_m - last_ts < interval_sec:
        return last_ts
    try:
        from state_io import atomic_write_json
        hb_path = str(path)
        payload = build_heartbeat_payload(
            status,
            symbols=symbols,
            intervals=intervals if intervals is not None else {"heartbeat": interval_sec},
            extra=extra,
        )
        atomic_write_json(hb_path, payload)
        os.utime(hb_path, None)
    except OSError:
        return last_ts
    return now_m


_MODULE_HEARTBEAT_LOCK = globals().get("_MODULE_HEARTBEAT_LOCK") or threading.Lock()
_MODULE_HEARTBEAT_LAST_TS: dict[str, float] = globals().get("_MODULE_HEARTBEAT_LAST_TS") or {}
_MODULE_HEARTBEAT_META: dict[str, dict[str, object]] = globals().get("_MODULE_HEARTBEAT_META") or {}


def touch_module_heartbeat(
    module_name: str,
    *,
    interval_sec: float = 30.0,
    force: object = False,
    now: float | None = None,
    status: object = None,
    symbols: list[str] | tuple[str, ...] | None = None,
    intervals: dict | None = None,
    extra: dict | None = None,
    path: str | os.PathLike | None = None,
) -> float:
    """Thread-safe shared heartbeat writer for any bot or fleet daemon.

    Resolves ``cachedb/system/heartbeat/<module_name>.heartbeat`` via ``cachepaths.cache_path``
    (unless ``path`` is explicitly provided), throttles writes to ``interval_sec``, and persists
    the canonical heartbeat JSON schema via ``touch_heartbeat_file``.
    """
    clean_name = str(module_name or "").strip() if not isinstance(module_name, bool) else ""
    if not clean_name and path is None:
        return float("-inf")
    if path is not None:
        hb_path = str(path)
    else:
        fname = clean_name if clean_name.endswith(".heartbeat") else f"{clean_name}.heartbeat"
        try:
            from cachepaths import cache_path as _cache_path, resolve_cachedb_filepath as _resolve_hb
            hb_path = str(_resolve_hb(_cache_path(fname)))
        except Exception:
            hb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cachedb", "system", "heartbeat", fname)
    with _MODULE_HEARTBEAT_LOCK:
        meta = _MODULE_HEARTBEAT_META.setdefault(hb_path, {})
        if status is not None:
            meta["status"] = status
        if symbols is not None:
            prev_syms = meta.get("symbols")
            if not (isinstance(prev_syms, (list, tuple)) and len(prev_syms) > 1 and len(symbols) == 1):
                meta["symbols"] = list(symbols)
        if intervals is not None:
            meta["intervals"] = intervals
        if extra is not None:
            meta["extra"] = extra
        prev_ts = _MODULE_HEARTBEAT_LAST_TS.get(hb_path, float("-inf"))
        updated_ts = touch_heartbeat_file(
            hb_path,
            last_ts=prev_ts,
            interval_sec=interval_sec,
            force=force,
            now=now,
            status=status if status is not None else meta.get("status"),
            symbols=meta.get("symbols") if symbols is not None or "symbols" in meta else None,  # type: ignore[arg-type]
            intervals=intervals if intervals is not None else meta.get("intervals"),  # type: ignore[arg-type]
            extra=extra if extra is not None else meta.get("extra"),  # type: ignore[arg-type]
        )
        _MODULE_HEARTBEAT_LAST_TS[hb_path] = updated_ts
        return updated_ts


def evaluate_heartbeat_files_freshness(
    hb_paths: list[str] | tuple[str, ...],
    stale_threshold_s: float,
    now_ts: float,
    *,
    fallback_start_ts: float | None = None,
) -> tuple[bool, float, float, str]:
    """Evaluate heartbeat freshness across ``hb_paths``.

    Returns ``(is_fresh, max_existing_age_sec, max_stale_duration_sec, stale_or_missing_path)``.
    """
    if not hb_paths:
        return True, 0.0, 0.0, ""
    try:
        from cachepaths import resolve_cachedb_filepath as _resolve_path
    except Exception:
        _resolve_path = str  # type: ignore[assignment]
    hb_only = [p for p in hb_paths if str(p).strip().endswith(".heartbeat")]
    effective_paths = hb_only if hb_only else hb_paths
    mtimes: list[float] = []
    missing_any, stale_file, max_stale_duration = False, "", 0.0
    start_ts = fallback_start_ts if fallback_start_ts is not None else now_ts

    for raw_hbp in effective_paths:
        hbp = str(_resolve_path(raw_hbp))
        if os.path.exists(hbp):
            try:
                mt = os.path.getmtime(hbp)
                mtimes.append(mt)
                dur = now_ts - mt
                if stale_threshold_s > 0 and dur > stale_threshold_s and dur > max_stale_duration:
                    max_stale_duration, stale_file = dur, hbp
            except Exception:
                missing_any = True
        else:
            missing_any = True
            dur = max(0.0, now_ts - start_ts)
            if stale_threshold_s > 0 and dur >= max_stale_duration:
                max_stale_duration, stale_file = dur, hbp

    age_sec = max(0.0, now_ts - min(mtimes)) if mtimes else 0.0
    fresh = not (stale_threshold_s > 0 and (missing_any or not mtimes or age_sec > stale_threshold_s))
    return fresh, age_sec, max_stale_duration, stale_file


def clamp_float(
    val: object,
    minimum: float = 0.0,
    maximum: float = 1.0,
    *,
    default: float = 0.0,
) -> float:
    """Safely coerce ``val`` to a float and clamp it to ``[minimum, maximum]``."""
    v = safe_float(val, default=default)
    if v is None:
        v = default
    return max(minimum, min(maximum, v))


def parse_ratio_or_pct(raw: object, pct_cutoff: float = 0.05) -> float:
    """Parse a numeric config value that may be expressed either as a ratio or as a percentage."""
    val = float(raw)
    return val / 100.0 if val > pct_cutoff else val


def safe_thresh(val: object, default: int) -> int:
    """Safely coerce sentiment index/threshold to an integer in [0, 100]."""
    f = safe_float(val, -1.0)
    return int(round(f)) if 0.0 <= f <= 100.0 else default


def resolve_order_notional(
    notional_eur: object,
    price: object,
    qty: object,
    *,
    default: float | None = None,
    fallback_on_nonfinite: bool = True,
) -> float | None:
    """Resolve positive order notional from explicit ``notional_eur`` or ``price * qty`` fallback."""
    n_clean = safe_pos_float(notional_eur, 0.0)
    if n_clean > 0.0:
        return n_clean
    if not fallback_on_nonfinite and notional_eur is not None and safe_float(notional_eur, None) is None:
        return default
    px_clean = safe_pos_float(price, 0.0)
    qty_clean = safe_pos_float(qty, 0.0)
    if px_clean > 0.0 and qty_clean > 0.0:
        return px_clean * qty_clean
    return default


def clean_scale(
    val: object,
    default: float = 1.0,
    *,
    allow_percent: bool = False,
    allow_percentage: bool = False,
    min_positive: bool = False,
    clamp: bool = False,
) -> float:
    """Normalize a scale factor into [0.0, 1.0], optionally converting [5.0, 100.0] or '%' percentages."""
    allow_pct = allow_percent or allow_percentage
    if allow_pct and isinstance(val, str) and val.strip().endswith("%"):
        pct = safe_nonneg_float(val.strip()[:-1].strip(), default=None)
        f = round(pct / 100.0, 4) if (pct is not None and pct <= 100.0) else -1.0
    else:
        f = safe_float(val, -1.0)
        if allow_pct and 5.0 <= f <= 100.0:
            f = round(f / 100.0, 4)
    if f < 0.0 or (min_positive and f <= 0.0):
        return default
    if clamp and f > 1.0:
        return 1.0
    return f if 0.0 <= f <= 1.0 else default


def normalize_epoch_sec(
    val: object,
    default: float | None = None,
    *,
    allow_zero: bool = False,
    allow_iso: bool = False,
) -> float | None:
    """Normalize a timestamp in seconds, milliseconds (>= 1e11), or optional ISO-8601 string into epoch seconds."""
    f = safe_nonneg_float(val, default=None) if allow_zero else safe_pos_float(val, default=None)
    if f is None and allow_iso and isinstance(val, str):
        s = val.strip()
        if len(s) >= 10 and ("-" in s or "T" in s):
            try:
                iso_s = s[:-1] + "+00:00" if s.endswith(("Z", "z")) else s
                dt = datetime.fromisoformat(iso_s)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                ts_s = dt.timestamp()
                if (ts_s >= 0.0) if allow_zero else (ts_s > 0.0):
                    f = ts_s
            except Exception:
                f = None
    if f is None:
        return default
    return f / 1000.0 if f >= 1e11 else f


def safe_nonneg_ts(val: object, default: float = 0.0) -> float:
    """Safely coerce a timestamp to non-negative epoch seconds, converting ms (>= 1e11)."""
    res = normalize_epoch_sec(val, default=default, allow_zero=True)
    return default if res is None else res


def resolve_now_ts(now_ts: object = None, *, allow_zero: bool = False) -> float:
    """Return ``now_ts`` normalized to epoch seconds (converting ms >= 1e11), or ``time.time()``."""
    f = normalize_epoch_sec(now_ts, default=None, allow_zero=allow_zero)
    return f if f is not None else _time.time()


def resolve_now_ms(now_ms: object = None) -> int:
    """Return ``now_ms`` as a positive integer millisecond timestamp, or ``int(time.time() * 1000)``."""
    parsed = safe_pos_int(now_ms, default=None)
    return parsed if parsed is not None else int(_time.time() * 1000)


def is_cache_fresh(
    last_ts: object,
    now_ts: float,
    ttl_sec: object,
    *,
    force_refresh: object = False,
) -> bool:
    """Return True if cached state at ``last_ts`` is still within ``ttl_sec`` at ``now_ts``."""
    if _parse_bool_flag(force_refresh, False):
        return False
    lt = safe_nonneg_float(last_ts, 0.0)
    ttl = safe_nonneg_float(ttl_sec, 0.0)
    return lt > 0.0 and 0.0 <= (now_ts - lt) < ttl


def is_interval_due(
    current_ts: float,
    last_ts: object,
    interval_sec: object,
    *,
    force: object = False,
) -> bool:
    """Return True if ``force`` is truthy or ``interval_sec`` has elapsed since ``last_ts``."""
    if _parse_bool_flag(force, False):
        return True
    lt = safe_nonneg_float(last_ts, 0.0)
    itv = safe_nonneg_float(interval_sec, 0.0)
    return not (0.0 <= (current_ts - lt) < itv)


def check_cooldown_gate(
    cooldown_map: dict,
    key: object,
    now_ts: float,
    cooldown_sec: float = 1800.0,
    max_entries: int = 1000,
) -> bool:
    """Return True and record ``now_ts`` if ``key`` is not within ``cooldown_sec``."""
    last = cooldown_map.get(key, -1e18)
    if 0.0 <= (now_ts - last) < cooldown_sec:
        return False
    cooldown_map[key] = now_ts
    while len(cooldown_map) > max_entries:
        cooldown_map.pop(next(iter(cooldown_map)), None)
    return True


_CSV_SPLIT_RE = re.compile(r"[,\n]+")


def split_csv(raw: str | None, *, lower: bool = False, upper: bool = False) -> tuple[str, ...]:
    """Split comma- or newline-delimited strings into a deduplicated tuple of stripped tokens."""
    if not raw:
        return ()
    seen: dict[str, None] = {}
    for part in _CSV_SPLIT_RE.split(raw):
        if cleaned := part.strip():
            cleaned = cleaned.lower() if lower else (cleaned.upper() if upper else cleaned)
            seen.setdefault(cleaned, None)
    return tuple(seen)


def parse_symbol_list(symbols: object) -> list[str]:
    """Normalize comma-separated string or iterable of symbols into uppercase list."""
    raw = (
        symbols.split(",")
        if isinstance(symbols, str)
        else (symbols if isinstance(symbols, (list, tuple, set, frozenset)) else [])
    )
    return [
        s.strip().upper()
        for s in raw
        if isinstance(s, str) and s.strip()
    ]


def validate_required_env_keys(
    keys: list[str] | tuple[str, ...],
    *,
    env: dict | None = None,
    label: str = "configuration",
) -> None:
    """Abort startup with the complete list of missing or empty mandatory environment keys."""
    src = os.environ if env is None else env
    missing = [name for name in keys if not _clean_env_str(src.get(name))]
    if missing:
        raise ValueError(f"Missing or empty mandatory {label} settings: " + ", ".join(missing))


def resolve_patched_attr(
    attr_name: str,
    default_val: object = None,
    *,
    local_val: object = _UNSET,
    defaults: tuple[object, ...] = (),
    modules: tuple[str, ...] | list[str] = (),
    require_callable: bool = False,
) -> object:
    """Resolve ``attr_name`` across ``local_val`` and ``sys.modules[modules]`` when monkeypatched away from ``defaults``."""
    unpatched = tuple(d for d in (defaults if defaults else (default_val,)) if d is not None)

    def _is_valid(cand: object) -> bool:
        if cand is _UNSET or cand is None:
            return False
        return callable(cand) if require_callable else True

    def _is_patched(cand: object) -> bool:
        return _is_valid(cand) and all(cand is not d for d in unpatched)

    if local_val is not _UNSET and _is_patched(local_val):
        return local_val
    for mod_name in modules:
        if not mod_name:
            continue
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, attr_name):
            cand = getattr(mod, attr_name, _UNSET)
            if _is_patched(cand):
                return cand
            if not unpatched and cand is not _UNSET and not require_callable:
                return cand
    if local_val is not _UNSET and _is_valid(local_val):
        return local_val
    return default_val


def bind_lazy_module_exports(
    module_globals: dict[str, object],
    lazy_imports: dict[str, str],
    submodules: tuple[str, ...] | list[str] | set[str] | frozenset[str] = (),
):
    """Bind PEP 562 ``__getattr__``, ``__dir__``, and ``__all__`` into ``module_globals``."""
    import importlib

    pkg_name = str(module_globals.get("__name__", ""))
    sub_set = frozenset(submodules)
    all_names = list(module_globals.get("__all__") or list(lazy_imports))
    module_globals["__all__"] = all_names

    def __getattr__(name: str):
        mod_name = lazy_imports.get(name)
        if mod_name is not None:
            val = getattr(importlib.import_module(mod_name), name)
            module_globals[name] = val
            return val
        if name in sub_set:
            mod = importlib.import_module(f"{pkg_name}.{name}")
            module_globals[name] = mod
            return mod
        raise AttributeError(f"module {pkg_name!r} has no attribute {name!r}")

    def __dir__() -> list[str]:
        return sorted(set(module_globals) | set(all_names))

    module_globals["__getattr__"] = __getattr__
    module_globals["__dir__"] = __dir__
    return __getattr__, __dir__


_SAFE_ID_DOT_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_SAFE_ID_NODOT_RE = re.compile(r"[^A-Za-z0-9_-]+")
_TREND_UP_TOKENS = frozenset(("up", "bull", "uptrend", "bullish"))
_TREND_DOWN_TOKENS = frozenset(("down", "bear", "downtrend", "bearish"))


def normalize_trend_direction(
    raw_dir: object,
    default: str | None = None,
    *,
    preserve_bull_bear: bool = False,
    allow_long_short: bool = False,
) -> str | None:
    """Normalize a raw trend token (``up``/``bull``/``uptrend``/``bullish``/``long``, etc.) into a canonical direction."""
    if not isinstance(raw_dir, str):
        return default
    d = raw_dir.strip().lower()
    if d in _TREND_UP_TOKENS or (allow_long_short and d == "long"):
        return ("up" if d == "up" else "bull") if preserve_bull_bear else "up"
    if d in _TREND_DOWN_TOKENS or (allow_long_short and d == "short"):
        return ("down" if d == "down" else "bear") if preserve_bull_bear else "down"
    return default


def sanitize_pipe_field(value: object) -> str:
    """Sanitize a field value for pipe-delimited logs by replacing ``|``, ``\\r``, and ``\\n``."""
    if value is None:
        return ""
    return str(value).replace("|", "/").replace("\r", " ").replace("\n", " ")


def sanitize_identifier(
    value: object,
    default: str = "unknown",
    *,
    sep: str = "_",
    allow_dots_hyphens: bool = True,
    strip_chars: str | None = None,
) -> str:
    """Sanitize an identifier/slug by replacing unsafe characters with ``sep`` and stripping edge separators."""
    if value is None or isinstance(value, bool):
        return default
    pattern = _SAFE_ID_DOT_RE if allow_dots_hyphens else _SAFE_ID_NODOT_RE
    cleaned = pattern.sub(sep, str(value).strip()).strip(strip_chars if strip_chars is not None else sep)
    return cleaned or default


def normalize_class_name(cls_or_name: object) -> str:
    """Return the class name for a ``type`` object or stripped string, else ``''``."""
    if isinstance(cls_or_name, type):
        return cls_or_name.__name__
    if isinstance(cls_or_name, str):
        return cls_or_name.strip()
    return ""


def lru_dict_put(cache: dict, key: object, value: object, max_entries: int) -> None:
    """Insert/refresh ``(key, value)`` at the MRU end of ``cache`` and evict oldest entries beyond ``max_entries``."""
    if hasattr(cache, "move_to_end"):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > max_entries:
            cache.popitem(last=False)
    else:
        if key in cache:
            cache.pop(key, None)
        cache[key] = value
        while len(cache) > max_entries:
            cache.pop(next(iter(cache)), None)


async def async_interruptible_sleep(
    stop_event: object,
    delay: object,
    step: object = 0.2,
    *,
    sleep_fn: object = None,
) -> None:
    """Sleep asynchronously up to ``delay`` seconds in ``step`` increments while ``stop_event`` is not set."""
    eff_delay = safe_pos_float(delay, default=None)
    if eff_delay is None:
        return
    eff_step = safe_pos_float(step, default=0.2) or 0.2
    if sleep_fn is None:
        import asyncio

        sleep_fn = asyncio.sleep
    is_stopped = getattr(stop_event, "is_set", None)
    elapsed = 0.0
    while elapsed < eff_delay and not (callable(is_stopped) and is_stopped()):
        await sleep_fn(min(eff_step, eff_delay - elapsed))
        elapsed += eff_step


def prepare_shadow_runtime(root: str, *module_dirs: str) -> None:
    """Make repository modules importable and disable live Binance WebSockets."""
    ensure_local_module_priority(root)
    for path in module_dirs:
        absolute = os.path.abspath(path)
        if absolute not in sys.path:
            sys.path.append(absolute)
    os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")


def load_shadow_environment(env_path: str, config_path: str | None = None) -> None:
    """Load a shadow runner with the same environment precedence as its live bot."""
    env_abs = os.path.abspath(env_path)
    cfg_abs = os.path.abspath(config_path or os.path.join(os.path.dirname(env_abs), "config.env"))
    load_env_stack(env_abs, cfg_abs)


def require_shadow_interval(interval: int, expected: int, label: str) -> None:
    """Reject a cadence for which a shadow strategy was not validated."""
    if (
        isinstance(interval, bool)
        or isinstance(expected, bool)
        or not isinstance(interval, int)
        or not isinstance(expected, int)
        or expected <= 0
        or interval != expected
    ):
        raise ValueError(f"{label} accepts only the native {expected}m interval")


# ─── Global trade_enabled Configuration Cache & Watcher (consolidated from config.py) ───

config_cache: dict = {}
_CONFIG_MTIME: tuple[float, float] = (-1.0, -1.0)
_CONFIG_LAST_CHECK: float = 0.0

CONFIG_ENV_PATH = os.path.join(_REPO_ROOT, "config.env")
CONFIG_TXT_PATH = os.path.join(_REPO_ROOT, "config.txt")

_parse_bool = _parse_bool_flag


def _file_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path) if path and os.path.exists(path) else -1.0
    except OSError:
        return -1.0


def _read_properties_file(path: str) -> dict:
    if not path or isinstance(path, (bool, int)):
        return {}
    return parse_dotenv(path, coerce_bools=True)


def load_config():
    """Load configuration from config.env (and legacy config.txt if present) and refresh cache."""
    global config_cache, _CONFIG_MTIME, _CONFIG_LAST_CHECK
    new_config = {}

    env_file_settings = _read_properties_file(CONFIG_ENV_PATH)
    for k, v in env_file_settings.items():
        if isinstance(k, str) and k.lower() == "trade_enabled" and str(v).strip() != "":
            new_config["trade_enabled"] = _parse_bool(v, default=True)

    txt_settings = _read_properties_file(CONFIG_TXT_PATH)
    for k, v in txt_settings.items():
        if k.lower() == "trade_enabled":
            if str(v).strip() != "":
                new_config["trade_enabled"] = _parse_bool(v, default=True)
        else:
            new_config[k.lower()] = v

    env_trade = os.environ.get("TRADE_ENABLED")
    if env_trade is not None and env_trade.strip() != "":
        new_config["trade_enabled"] = _parse_bool(env_trade, default=True)

    if "trade_enabled" not in new_config:
        new_config["trade_enabled"] = True

    _CONFIG_MTIME = (_file_mtime(CONFIG_ENV_PATH), _file_mtime(CONFIG_TXT_PATH))
    _CONFIG_LAST_CHECK = _time.time()
    config_cache = new_config


_watcher_stop_event = threading.Event()


def config_watcher(interval=5 * 60):
    """Periodically monitor configuration and reload the cache."""
    wait_sec = safe_pos_float(interval, default=5.0 * 60.0)
    while not _watcher_stop_event.is_set():
        load_config()
        if _watcher_stop_event.wait(wait_sec):
            break


def is_trade_enabled() -> bool:
    """Return whether cached ``trade_enabled`` is true (with on-demand mtime hot-reload every 5s)."""
    now_ts = _time.time()
    if not (0.0 <= (now_ts - _CONFIG_LAST_CHECK) < 5.0):
        cur_mtime = (_file_mtime(CONFIG_ENV_PATH), _file_mtime(CONFIG_TXT_PATH))
        if cur_mtime != _CONFIG_MTIME:
            load_config()
    return _parse_bool(config_cache.get("trade_enabled", True), default=True)


watcher_thread = None


def start_config_watcher():
    global watcher_thread
    if watcher_thread and watcher_thread.is_alive():
        return
    _watcher_stop_event.clear()
    watcher_thread = threading.Thread(
        target=config_watcher,
        name="start_config_watcher",
        daemon=True,
    )
    watcher_thread.start()
    print("Config watcher started.")


def stop_config_watcher():
    global watcher_thread
    _watcher_stop_event.set()
    if watcher_thread:
        watcher_thread.join(timeout=2.0)
        watcher_thread = None
        print("Config watcher stopped.")


load_config()


def register_module_aliases(module_name: str, *aliases: str) -> None:
    """Bind ``aliases`` in ``sys.modules`` (and parent package attributes) to ``sys.modules[module_name]``."""
    self_mod = sys.modules.get(module_name)
    if self_mod is None:
        return
    self_file = getattr(self_mod, "__file__", None)
    for alias in aliases:
        prev = sys.modules.get(alias)
        if prev is None or prev is self_mod or getattr(prev, "__file__", None) == self_file:
            sys.modules[alias] = self_mod
            if "." in alias:
                parent_name, child_attr = alias.rsplit(".", 1)
                parent_pkg = sys.modules.get(parent_name)
                if parent_pkg is not None:
                    setattr(parent_pkg, child_attr, self_mod)


register_module_aliases(__name__, "shadow_runtime")


_MERGED_MODULE_TARGETS = {
    "accepted_order_persistence": "order_lifecycle",
    "order_outcomes_log": "order_lifecycle",
    "order_rollback": "order_lifecycle",
    "shadow_runtime": "botcore",
    "liquidation_signal": "intelligence.external.collectors.bybit_liquidations_collector",
    "hyperliquid.signals": "hyperliquid.signals",
    "hyperliquid.price_analysis": "hyperliquid.signals",
    "price_analysis": "hyperliquid.signals",
    "keys": "credentials",
    "keys.apikeys": "credentials",
    "instruments_config": "instrument_registry",
    "binance_api.bapi_ws_health": "binance_api.bapi_ws_bridge",
    "lock.file_lock": "lock.cooldown",
    "lock.trade_cooldown": "lock.cooldown",
    "providers.binance_filters": "providers.quantity",
    "providers.http_resilience": "providers.base",
    "providers.replay_clock": "providers.replay_provider",
    "config": "botcore",
    "symbols": "instrument_registry",
}


class _MergedModuleAliasLoader:
    def __init__(self, alias_name: str, target_name: str) -> None:
        self.alias_name = alias_name
        self.target_name = target_name

    def create_module(self, spec):
        return None

    def exec_module(self, module) -> None:
        import importlib

        if self.target_name == "hyperliquid.signals":
            hl_dir = os.path.join(_REPO_ROOT, "hyperliquid")
            is_reload = (
                module is sys.modules.get("signals")
                and getattr(module, "__mptrade_exec_done__", False)
            )
            target_mod = load_local_module(hl_dir, "signals", reload=is_reload)
            for alias_k in ("signals", "hyperliquid.signals", "price_analysis", "hyperliquid.price_analysis"):
                sys.modules[alias_k] = target_mod
            hl_pkg = sys.modules.get("hyperliquid")
            if hl_pkg is not None:
                setattr(hl_pkg, "signals", target_mod)
                setattr(hl_pkg, "price_analysis", target_mod)
            sys.modules[self.alias_name] = target_mod
            return

        target_mod = importlib.import_module(self.target_name)
        if self.alias_name == "keys":
            keys_pkg = sys.modules.get("keys")
            if keys_pkg is None or keys_pkg is module:
                import types as _types

                keys_pkg = _types.ModuleType("keys")
                keys_pkg.__path__ = []
                keys_pkg._is_credentials_shim = True
            keys_pkg.apikeys = target_mod
            sys.modules["keys"] = keys_pkg
            sys.modules["keys.apikeys"] = target_mod
            return
        if self.alias_name == "keys.apikeys":
            keys_pkg = sys.modules.get("keys")
            if keys_pkg is not None:
                keys_pkg.apikeys = target_mod
        elif "." in self.alias_name:
            parent_name, child_attr = self.alias_name.rsplit(".", 1)
            parent_pkg = sys.modules.get(parent_name)
            if parent_pkg is not None:
                setattr(parent_pkg, child_attr, target_mod)
        sys.modules[self.alias_name] = target_mod


class _MergedModuleAliasFinder:
    """Resolve Package E merged module names even before host import or after sys.modules.pop."""

    @staticmethod
    def find_spec(fullname: str, path=None, target=None):
        target_name = _MERGED_MODULE_TARGETS.get(fullname)
        if target_name is None and fullname == "signals" and target is not None:
            target_file = getattr(target, "__file__", None)
            hl_signals_file = os.path.join(_REPO_ROOT, "hyperliquid", "signals.py")
            if isinstance(target_file, str) and os.path.abspath(target_file) == os.path.abspath(hl_signals_file):
                target_name = "hyperliquid.signals"
        if target_name is None and fullname.startswith("archive."):
            rel_parts = fullname.split(".")
            sub_dir = os.path.join(_REPO_ROOT, *rel_parts[:-1])
            stem = rel_parts[-1]
            resolved_dir = _resolve_archived_module_dir(sub_dir, stem)
            py_file = os.path.join(resolved_dir, f"{stem}.py")
            if os.path.isfile(py_file):
                import importlib.util

                return importlib.util.spec_from_file_location(fullname, py_file)
        if target_name is None:
            return None
        if fullname in ("keys", "keys.apikeys"):
            existing_keys = sys.modules.get("keys")
            if "keys" in sys.modules and existing_keys is None:
                return None
            if (
                existing_keys is not None
                and not hasattr(existing_keys, "__path__")
                and not getattr(existing_keys, "_is_credentials_shim", False)
            ):
                return None
        import importlib.util

        return importlib.util.spec_from_loader(
            fullname,
            _MergedModuleAliasLoader(fullname, target_name),
            is_package=(fullname == "keys"),
        )


if not any(getattr(f, "__name__", "") == "_MergedModuleAliasFinder" or isinstance(f, _MergedModuleAliasFinder) for f in sys.meta_path):
    sys.meta_path.append(_MergedModuleAliasFinder())


register_module_aliases(__name__, "config")




