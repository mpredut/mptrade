#!/usr/bin/env python3
"""
botcore.py — SHARED SSOT core for bot utilities: logging, .env, HTTP, numeric coercion, and time.
No external dependencies; standard library only.
"""
from __future__ import annotations

import asyncio
import atexit
import base64
import configparser
import contextlib
import fcntl
from functools import lru_cache
import importlib
import importlib.machinery
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import tarfile
import tempfile
import threading
import time as _time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, time as dt_time, timezone
from zoneinfo import ZoneInfo
from cachepaths import (
    _coerce_int,
    _parse_bool_flag,
    cache_path,
    cache_symbol_variants,
    canonical_base_asset,
    clean_finite_floats,
    compatible_quote_suffixes,
    exact_int,
    exact_nonneg_int,
    exact_pos_int,
    has_equity_suffix,
    is_equity_venue_or_suffix,
    normalize_venue_name,
    resolve_cachedb_filepath,
    safe_float,
    safe_int,
    safe_nonneg_float,
    safe_nonneg_int,
    safe_pos_float,
    safe_pos_int,
)
import state_io as _state_io

HTTP_TIMEOUT = 25
BUCHAREST = ZoneInfo("Europe/Bucharest")

_LOCKS: dict = globals().get("_LOCKS") or {}
_LOG_LOCK = globals().get("_LOG_LOCK") or threading.RLock()


def log(msg: str) -> None:
    with _LOG_LOCK:
        print(f"[{datetime.now(timezone.utc).astimezone(BUCHAREST):%H:%M:%S}] {msg}", flush=True)


def now_str() -> str:
    """Return current timestamp formatted in Europe/Bucharest timezone."""
    return f"{datetime.now(timezone.utc).astimezone(BUCHAREST):%Y-%m-%d %H:%M:%S} Bucharest"


def single_instance(name: str, lockdir: str = "/tmp") -> None:
    """Enforce a single instance with an exclusive flock."""
    path = os.path.join(lockdir, f"binance_{name}.lock")
    fd = open(path, "w")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"[{name}] already running (lock held: {path}) — exiting.", flush=True)
        sys.exit(0)
    fd.write(str(os.getpid()))
    fd.flush()
    _LOCKS[name] = fd


_HAVE_FCNTL = True


class FileLock:
    """Cross-process/thread lock using fcntl.flock(LOCK_EX) on an owned fd."""

    def __init__(self, path):

        self.path = resolve_cachedb_filepath(path)
        self._fd = None

    def __enter__(self):
        parent = os.path.dirname(os.fspath(self.path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._fd = open(self.path, "a+")
        if _HAVE_FCNTL:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX)
            except BaseException:
                self._fd.close()
                self._fd = None
                raise
        return self

    def __exit__(self, *exc):
        if self._fd is None:
            return
        try:
            if _HAVE_FCNTL:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            self._fd.close()
            self._fd = None


# ═════════════════════════════════════════════════════════════════════════════
# 1. NUMERIC COERCION & OBJECT SANITIZATION PRIMITIVES
# ═════════════════════════════════════════════════════════════════════════════



def strict_ascii_int(
    val: object, default: int | None = None, *, min_value: int | None = None, allow_float_str: bool = False
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
    if parsed is None or (min_value is not None and parsed < min_value):
        return default
    return parsed


def require_finite_float(
    val: object, name: str = "value", *, min_value: float | None = None, error_msg: str | None = None
) -> float:
    """Coerce ``val`` to a finite float (optionally ``>= min_value``) or raise ``ValueError``."""
    f = safe_float(val, default=None)
    if f is None or (min_value is not None and f < min_value):
        msg = f"{name} must be finite and >= {min_value}" if min_value is not None else f"{name} must be finite"
        raise ValueError(error_msg or msg)
    return f


def require_pos_float(
    val: object, name: str = "value", *, min_value: float | None = None, error_msg: str | None = None
) -> float:
    """Coerce ``val`` to a finite positive float (optionally ``>= min_value``) or raise ``ValueError``."""
    f = safe_pos_float(val, default=None)
    if f is None or (min_value is not None and f < min_value):
        msg = f"{name} must be finite and >= {min_value}" if min_value is not None else f"{name} must be finite and positive"
        raise ValueError(error_msg or msg)
    return f


def require_nonneg_float(
    val: object, name: str = "value", *, min_value: float | None = None, error_msg: str | None = None
) -> float:
    """Coerce ``val`` to a finite non-negative float (optionally ``>= min_value``) or raise ``ValueError``."""
    f = safe_nonneg_float(val, default=None)
    if f is None or (min_value is not None and f < min_value):
        msg = (
            f"{name} must be finite and >= {min_value}"
            if min_value is not None
            else f"{name} must be finite and non-negative"
        )
        raise ValueError(error_msg or msg)
    return f


def require_int(
    val: object, name: str = "value", *, min_value: int | None = None, error_msg: str | None = None
) -> int:
    """Coerce ``val`` to an exact integer (optionally ``>= min_value``) or raise ``ValueError``."""
    parsed = exact_int(val, default=None)
    if parsed is None or (min_value is not None and parsed < min_value):
        raise ValueError(error_msg or f"invalid {name}")
    return parsed


def clean_upper_token(val: object) -> str | None:
    """Return ``val.strip().upper()`` when ``val`` is a non-empty string, else ``None``."""
    if not isinstance(val, str) or not (s := val.strip()):
        return None
    return s.upper()


def obj_attr(obj: object, key: str, default: object = None) -> object:
    """Extract ``key`` uniformly from either a ``dict`` or an object attribute."""
    if obj is None or isinstance(obj, (str, bytes, bytearray)):
        return default
    val = obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)
    return default if val is None else val


def obj_float(obj: object, key: str, default: float = 0.0) -> float:
    """Extract ``key`` from ``obj`` (dict or object) and coerce to finite float."""
    return safe_float(obj_attr(obj, key, None), default)


def coerce_dict_maps(*maps: object) -> tuple[dict, ...]:
    """Return a tuple where each item is preserved if ``isinstance(item, dict)`` else ``{}``."""
    return tuple(m if isinstance(m, dict) else {} for m in maps)


def clean_json_value(val: object, *, stringify_keys: bool = True) -> object:
    """Recursively sanitize non-finite floats (NaN/Inf -> None) and optionally normalize dict keys to strings."""
    if isinstance(val, float):
        return safe_float(val, default=None)
    if isinstance(val, dict):
        return {(str(k) if stringify_keys else k): clean_json_value(v, stringify_keys=stringify_keys) for k, v in val.items()}
    if isinstance(val, (list, tuple, set, frozenset)):
        return [clean_json_value(v, stringify_keys=stringify_keys) for v in val]
    return val


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


class OrderedIdSet:
    """Insertion-ordered set of IDs with deterministic FIFO eviction."""

    def __init__(self):
        self._d = OrderedDict()

    def add(self, item: str) -> None:
        self._d[item] = None
        self._d.move_to_end(item)

    def clear(self) -> None:
        self._d.clear()

    def trim(self, keep: int = 5000) -> None:
        while len(self._d) > keep:
            self._d.popitem(last=False)

    def difference_update(self, items) -> None:
        for item in items:
            self._d.pop(item, None)

    def __contains__(self, item) -> bool:
        return item in self._d

    def __len__(self) -> int:
        return len(self._d)

    def __iter__(self):
        return iter(self._d)


def format_price_log(price: object) -> str:
    """Format a price with 2 decimals when >= 1.0 and up to 6 decimals for sub-dollar values."""
    px = safe_float(price, default=0.0) or 0.0
    return f"{px:.2f}" if abs(px) >= 1.0 else (f"{px:.6f}".rstrip("0").rstrip(".") or "0")


def format_notify_qty(qty: object) -> str:
    """Format a trade/order quantity without trailing zeros."""
    qf = safe_pos_float(qty, default=0.0) or 0.0
    return f"{qf:.8f}".rstrip("0").rstrip(".") or "0"


def format_notify_price(price: object, *, comma: bool = True) -> str:
    """Format a price with optional comma grouping and full sub-dollar precision."""
    pf = safe_pos_float(price, default=0.0) or 0.0
    sep = "," if comma else ""
    if pf >= 1.0 and round(pf, 2) == round(pf, 6):
        return f"{pf:{sep}.2f}"
    raw = f"{pf:{sep}.8f}".rstrip("0")
    if raw.endswith("."):
        return f"{raw}00"
    if len(raw.rsplit(".", 1)[-1]) == 1:
        return f"{raw}0"
    return raw


# ═════════════════════════════════════════════════════════════════════════════
# 2. CONFIGURATION & ENVIRONMENT SSOT (Loading, Parsing, Key-Value & Watcher)
# ═════════════════════════════════════════════════════════════════════════════

# ── 2.1. Dotenv / KEY=VALUE Parsing & Environment Stack Loading ──────────────


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
            pairs = [parsed for raw in f if (parsed := parse_dotenv_line(raw, strict=strict)) is not None]
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
    """Load secrets/overrides and the adjacent versioned configuration."""
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


def load_env_file_defaults(env_path: str | os.PathLike) -> dict[str, str]:
    """Parse an env file and set non-STRAT_* keys in os.environ via setdefault."""
    parsed: dict[str, str] = parse_dotenv(env_path, upper_keys=True)  # type: ignore[assignment]
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


def load_layered_env_config(
    config_file: str | os.PathLike,
    env_file: str | os.PathLike,
    *,
    override_prefixes: tuple[str, ...] | list[str] = (),
    fallback_prefixes: tuple[str, ...] | list[str] = (),
) -> dict[str, str]:
    """Load ``config_file`` then ``env_file``, applying ``os.environ`` overrides and fallbacks."""
    cfg: dict[str, str] = {}
    for path_val in (config_file, env_file):
        p_str = os.fspath(path_val)
        if p_str and os.path.exists(p_str):
            cfg.update(dict(parse_dotenv(p_str)))
    ov_tuple = tuple(override_prefixes)
    fb_tuple = tuple(fallback_prefixes)
    for k, v in os.environ.items():
        if (ov_tuple and k.startswith(ov_tuple)) or (fb_tuple and k.startswith(fb_tuple) and k not in cfg):
            cfg[k] = v
    return cfg


def load_isolated_venue_cfg(
    venue_dir: str | os.PathLike[str],
    *,
    config_path: str | os.PathLike[str] | None = None,
    env_path: str | os.PathLike[str] | None = None,
    override_prefixes: tuple[str, ...] | list[str] = (),
    fallback_prefixes: tuple[str, ...] | list[str] = (),
) -> dict[str, str]:
    """Load a venue folder's config.env + .env into an isolated dict without mutating os.environ."""
    base_dir = os.fspath(venue_dir)
    resolved_env = os.fspath(env_path) if env_path is not None else os.path.join(base_dir, ".env")
    resolved_cfg = (
        os.fspath(config_path)
        if config_path is not None
        else os.path.join(os.path.dirname(resolved_env) or base_dir, "config.env")
    )
    return load_layered_env_config(
        resolved_cfg,
        resolved_env,
        override_prefixes=override_prefixes,
        fallback_prefixes=fallback_prefixes,
    )


# ── 2.2. INI (.conf) Parsing, Mtime Caching & Override Resolution ────────────

_INI_CONFIG_CACHE: dict[str, tuple[tuple[int, int, bool], configparser.ConfigParser]] = (
    globals().get("_INI_CONFIG_CACHE") or {}
)
_INI_CONFIG_LOCK = globals().get("_INI_CONFIG_LOCK") or threading.RLock()


def load_ini_config(
    path: str | os.PathLike, *, must_exist: bool = True, inline_comments: bool = True
) -> configparser.ConfigParser:
    """Load and cache an INI ``.conf`` file by ``(mtime_ns, size)`` using ``ConfigParser(interpolation=None)``."""
    if isinstance(path, bool) or not isinstance(path, (str, os.PathLike)) or not str(path).strip():
        raise ValueError(f"Invalid INI configuration path: {path!r}")
    abs_path = os.path.abspath(os.fspath(path))
    prefixes = ("#", ";") if inline_comments else ()
    if not os.path.exists(abs_path):
        if must_exist:
            raise FileNotFoundError(f"Missing mandatory configuration file: {path}")
        return configparser.ConfigParser(interpolation=None, inline_comment_prefixes=prefixes)
    st = os.stat(abs_path)
    sig = (st.st_mtime_ns, st.st_size, bool(inline_comments))
    with _INI_CONFIG_LOCK:
        cached = _INI_CONFIG_CACHE.get(abs_path)
        if cached is not None and cached[0] == sig:
            return cached[1]
        cp = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=prefixes)
        with open(abs_path, "r", encoding="utf-8") as handle:
            cp.read_file(handle)
        lru_dict_put(_INI_CONFIG_CACHE, abs_path, (sig, cp), 64)
        return cp


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


# ── 2.3. Typed Environment & Key-Value Lookups ───────────────────────────────


def _clean_env_str(val: object) -> str:
    """Strip inline `#` comment and surrounding whitespace from a raw env value."""
    if val is None or isinstance(val, bool):
        return ""
    return _strip_env_value_quotes_and_comment(str(val))


def float_env(key: str, env: dict | float | None = None, default: float | None = None) -> float | None:
    """Read a float from os.environ or an injected dictionary, ignoring inline comments."""
    if env is not None and not hasattr(env, "get"):
        default, env = env, None
    src = os.environ if env is None else env
    raw = _clean_env_str(src.get(key))
    return safe_float(raw, default=default) if raw else default


def pos_float_env(key: str, default: float, env: dict | None = None) -> float:
    """Read a strictly positive float from os.environ or an injected dictionary, falling back to default."""
    return safe_pos_float(float_env(key, env=env), default)


def int_env(key: str, env: dict | int | None = None, default: int | None = None) -> int | None:
    """Read an integer from os.environ or an injected dictionary, ignoring inline comments."""
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
    raise ValueError(f"Invalid integer setting: {key}={value!r}")


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
                return str(parsed[key]).strip() or None
        except Exception:
            continue
    return None


def is_env_defined(key: str, env: dict | None = None) -> bool:
    """Return True if ``key`` exists in ``env`` (or ``os.environ`` when ``env`` is None)."""
    return key in (os.environ if env is None else env)


def has_explicit_nonroot_env(key: str) -> bool:
    """Return True when ``key`` has a non-empty ``os.environ`` value not solely loaded from root ``config.env``."""
    raw = os.environ.get(key)
    return bool(raw is not None and str(raw).strip() != "" and not _is_unmodified_root_key(key))


def secret_env(key: str, default: str | None = None, env: dict | None = None) -> str | None:
    """Read a secret/credential string from ``env``/``os.environ`` without stripping ``#`` characters."""
    src = os.environ if env is None else env
    raw = src.get(key)
    if isinstance(raw, str):
        clean = raw.strip()
        if clean:
            return clean
    return default


def folder_or_env_value(
    folder: str | os.PathLike | None,
    key: str,
    *,
    respect_empty_env: bool = False,
    strip_comments: bool = False,
    env: dict | None = None,
) -> str | None:
    """Resolve ``key`` from ``env``/``os.environ`` first, falling back to ``env_value(folder, key)``."""
    src = os.environ if env is None else env
    clean = optional_env(key, env=src) if strip_comments else (secret_env(key, env=src) or "")
    if respect_empty_env and key in src:
        return clean or None
    if clean:
        return clean
    return env_value(folder, key) if folder is not None else None


def env_values_map(folder: str | os.PathLike, names: tuple[str, ...] | list[str]) -> dict[str, str | None]:
    """Resolve each key from ``os.environ`` first, falling back to ``env_value(folder, name)``."""
    return {name: folder_or_env_value(folder, name) for name in names}


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


def any_bool_env(*keys: str, env: dict | None = None) -> bool:
    """Return True when at least one key in ``keys`` resolves to a truthy boolean flag."""
    return any(bool_env(k, env=env, default=False) for k in keys)


def resolve_cli_arg(
    argv: list[str] | tuple[str, ...] | None, flags: str | tuple[str, ...], default: str | None = None
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


@contextlib.contextmanager
def temporary_environ(overrides: Mapping[str, str | None] | None = None, **kw_overrides: str | None):
    """Temporarily set or unset ``os.environ`` keys and restore exact prior values on exit."""
    merged: dict[str, str | None] = {}
    if isinstance(overrides, Mapping):
        for k, v in overrides.items():
            if isinstance(k, str) and k:
                merged[k] = v
    for k, v in kw_overrides.items():
        if isinstance(k, str) and k:
            merged[k] = v
    saved = {k: os.environ.get(k) for k in merged}
    for k, v in merged.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)
    try:
        yield
    finally:
        for k, prev in saved.items():
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev


def resolve_cli_env_path(
    script_file: str | os.PathLike,
    argv: list[str] | tuple[str, ...] | None = None,
    *,
    env_flag: str = "--env-file",
    env_var: str = "ENV_FILE",
) -> str:
    """Resolve ``--env-file`` / ``ENV_FILE`` relative to ``script_file``, falling back on blank values."""
    default_env = optional_env(env_var, os.path.join(os.path.dirname(os.fspath(script_file)), ".env"))
    raw_arg = resolve_cli_arg(sys.argv if argv is None else argv, env_flag, default_env)
    return raw_arg.strip() if isinstance(raw_arg, str) and raw_arg.strip() else default_env


def load_cli_env_stack(
    script_file: str | os.PathLike,
    argv: list[str] | tuple[str, ...] | None = None,
    *,
    env_flag: str = "--env-file",
    env_var: str = "ENV_FILE",
    load_fn: object = None,
) -> str:
    """Resolve ``--env-file`` / ``ENV_FILE`` relative to ``script_file`` and load its environment stack."""
    env_file = resolve_cli_env_path(script_file, argv, env_flag=env_flag, env_var=env_var)
    eff_load = load_fn if callable(load_fn) else load_env_stack
    eff_load(env_file)
    return env_file


def add_paper_skip_wait_cli_args(
    parser: object,
    *,
    paper_help: str = "Force PAPER mode without real money",
    skip_wait_help: str = "Skip waiting for the launch",
) -> None:
    """Register standard ``--skip-wait`` and ``--paper`` CLI flags on ``parser``."""
    add_arg = getattr(parser, "add_argument", None)
    if callable(add_arg):
        add_arg("--skip-wait", action="store_true", help=skip_wait_help)
        add_arg("--paper", action="store_true", help=paper_help)


# ── 2.4. Global trade_enabled Configuration Cache & Hot-Reload Watcher ───────

config_cache: dict = {}
_CONFIG_MTIME: tuple[float, float] = (-1.0, -1.0)
_CONFIG_LAST_CHECK: float = 0.0

ROOT_DIR = _REPO_ROOT
CONFIG_ENV_PATH = os.path.join(_REPO_ROOT, "config.env")
CONFIG_TXT_PATH = os.path.join(_REPO_ROOT, "config.txt")


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
    new_config: dict = {}

    for path, allow_other in ((CONFIG_ENV_PATH, False), (CONFIG_TXT_PATH, True)):
        for k, v in _read_properties_file(path).items():
            if not isinstance(k, str):
                continue
            kl = k.lower()
            if kl == "trade_enabled":
                if str(v).strip() != "":
                    new_config["trade_enabled"] = _parse_bool_flag(v, default=True)
            elif allow_other:
                new_config[kl] = v

    env_trade = os.environ.get("TRADE_ENABLED")
    if env_trade is not None and env_trade.strip() != "":
        new_config["trade_enabled"] = _parse_bool_flag(env_trade, default=True)
    new_config.setdefault("trade_enabled", True)

    _CONFIG_MTIME = (_file_mtime(CONFIG_ENV_PATH), _file_mtime(CONFIG_TXT_PATH))
    _CONFIG_LAST_CHECK = _time.time()
    config_cache = new_config


def is_trade_enabled() -> bool:
    """Return whether cached ``trade_enabled`` is true (with on-demand mtime hot-reload every 5s)."""
    now_ts = _time.time()
    if not (0.0 <= (now_ts - _CONFIG_LAST_CHECK) < 5.0):
        cur_mtime = (_file_mtime(CONFIG_ENV_PATH), _file_mtime(CONFIG_TXT_PATH))
        if cur_mtime != _CONFIG_MTIME:
            load_config()
    return _parse_bool_flag(config_cache.get("trade_enabled", True), default=True)


load_config()


# ═════════════════════════════════════════════════════════════════════════════
# 3. HTTP TRANSPORT, APPROXIMATE MATH & MARKET HOURS
# ═════════════════════════════════════════════════════════════════════════════


def http_request(
    method: str,
    url: str,
    headers: dict | None = None,
    payload: dict | None = None,
    *,
    form: dict | None = None,
    body: bytes | None = None,
    timeout: float | None = None,
) -> tuple[int, bytes]:
    """Shared standard-library HTTP transport with a ``(status, body)`` contract."""
    if sum(x is not None for x in (payload, form, body)) > 1:
        raise ValueError("payload, form, and body are mutually exclusive")
    timeout_f = HTTP_TIMEOUT if timeout is None else require_pos_float(timeout, name="timeout")

    request_headers = dict(headers) if isinstance(headers, dict) else {}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")
    elif form is not None:
        data = urllib.parse.urlencode(form).encode()
        request_headers.setdefault("Content-Type", "application/x-www-form-urlencoded")
    elif body is not None:
        data = bytes(body)

    if not method or isinstance(method, bool) or not isinstance(method, str) or not method.strip():
        return 0, b""
    if not url or isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        return 0, b""
    verb = method.strip().upper()
    try:
        req = urllib.request.Request(url.strip(), data=data, headers=request_headers, method=verb)
        with urllib.request.urlopen(req, timeout=timeout_f) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:  # noqa: BLE001
        log(f"  ! network error {verb}: {e}")
        return 0, b""


def http_get(url: str, headers: dict | None = None) -> tuple[int, bytes]:
    return http_request("GET", url, headers=headers)


def http_post_json(url: str, payload: dict, headers: dict | None = None) -> tuple[int, bytes]:
    return http_request("POST", url, headers=headers, payload=payload)


def http_post_form(url: str, data: dict, headers: dict | None = None) -> tuple[int, bytes]:
    return http_request("POST", url, headers=headers, form=data)


def _validate_http_url(url: object) -> str:
    if isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        raise ValueError(f"invalid HTTP URL: {url!r}")
    u = url.strip()
    if not u.lower().startswith(("http://", "https://")):
        raise ValueError(f"unsupported non-HTTP URL scheme: {url!r}")
    return u


def _urlopen_bytes(req: urllib.request.Request, timeout: float) -> bytes:
    timeout_f = require_pos_float(timeout, name="timeout")
    with urllib.request.urlopen(req, timeout=timeout_f) as resp:
        return resp.read()


def _urlopen_json(req: urllib.request.Request, timeout: float) -> object:
    raw = _urlopen_bytes(req, timeout)
    return json.loads(raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw)


def fetch_public_bytes(
    url: str, *, timeout: float = 5.0, user_agent: str = "mptrade-intelligence/1.0", headers: dict | None = None
) -> bytes:
    """Fetch raw bytes from a public HTTP GET endpoint."""
    clean_url = _validate_http_url(url)
    req_headers = {"User-Agent": user_agent, **(headers if isinstance(headers, dict) else {})}
    return _urlopen_bytes(urllib.request.Request(clean_url, headers=req_headers), timeout)


def fetch_public_json(
    url: str, *, timeout: float = 5.0, user_agent: str = "mptrade-intelligence/1.0", headers: dict | None = None
) -> object:
    """Fetch and decode a JSON payload from a public HTTP GET endpoint."""
    clean_url = _validate_http_url(url)
    req_headers = {"User-Agent": user_agent, **(headers if isinstance(headers, dict) else {})}
    return _urlopen_json(urllib.request.Request(clean_url, headers=req_headers), timeout)


def fetch_json_post(
    url: str, payload: dict | bytes | None = None, *, timeout: float = 5.0, headers: dict | None = None
) -> object:
    """Send an HTTP POST request with a JSON body and decode the JSON response."""
    clean_url = _validate_http_url(url)
    req_headers = {"Content-Type": "application/json", **(headers if isinstance(headers, dict) else {})}
    if payload is None:
        data = b"{}"
    elif isinstance(payload, (bytes, bytearray)):
        data = bytes(payload)
    else:
        data = json.dumps(payload).encode("utf-8")
    return _urlopen_json(urllib.request.Request(clean_url, data=data, headers=req_headers), timeout)


# ── DETERMINISTIC percentage-based approximate comparisons ───────────────────


def diff_percent(value1: float, value2: float) -> float:
    """Return symmetric percentage difference relative to the values' absolute mean."""
    v1 = safe_float(value1, default=None)
    v2 = safe_float(value2, default=None)
    if v1 is None or v2 is None:
        return float("nan")
    if v1 == v2:
        return 0.0
    denom = (abs(v1) + abs(v2)) / 2.0
    return 0.0 if denom == 0.0 else abs(v1 - v2) / denom * 100.0


def are_close(value1: float, value2: float, tolerance_percent: float = 1.0) -> bool:
    """Return deterministically whether values differ by at most tolerance_percent."""
    tol = safe_nonneg_float(tolerance_percent, default=None)
    if tol is None:
        return False
    diff_f = safe_nonneg_float(diff_percent(value1, value2), default=None)
    return diff_f is not None and diff_f <= tol


ET = ZoneInfo("America/New_York")


def is_us_rth_open(now_ts: float | None = None) -> bool:
    """Return True if US equities regular trading hours (09:30-16:00 America/New_York, Mon-Fri) are open."""
    ts = resolve_now_ts(now_ts)
    try:
        ny_dt = datetime.fromtimestamp(ts, tz=ET)
    except (ValueError, OverflowError, OSError):
        ny_dt = datetime.fromtimestamp(_time.time(), tz=ET)
    return ny_dt.weekday() < 5 and dt_time(9, 30) <= ny_dt.time() < dt_time(16, 0)


# ═════════════════════════════════════════════════════════════════════════════
# 4. MODULE LOADING, HEARTBEATS, TIMESTAMPS & RUNTIME HELPERS
# ═════════════════════════════════════════════════════════════════════════════


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
            target = entry[1] if isinstance(entry, (list, tuple)) and len(entry) >= 2 else entry
            if (px_f := safe_pos_float(target, None)) is not None:
                return px_f
    return None


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
            mod_obj = sys.modules.get(mod_name)
            if mod_obj is None:
                continue
            mod_file = getattr(mod_obj, "__file__", None)
            if type(mod_obj) is object or (
                isinstance(mod_file, str) and mod_file and os.path.dirname(os.path.abspath(mod_file)) != abs_dir
            ):
                sys.modules.pop(mod_name, None)


_AMBIGUOUS_LOCAL_MODULES = (
    "strategy", "market_data", "replay", "common", "hl_client", "t212_client", "ipo_common",
    "notify", "ipo_notify", "signals",
)

_LEGACY_ARCHIVE_DIR: str | None = None


def _has_module_target(directory: str, stem: str) -> bool:
    return os.path.isfile(os.path.join(directory, f"{stem}.py")) or os.path.isdir(os.path.join(directory, stem))


def _resolve_archived_module_dir(abs_dir: str, stem: str) -> str:
    global _LEGACY_ARCHIVE_DIR
    if _has_module_target(abs_dir, stem):
        return abs_dir
    archive_tar = os.path.join(_REPO_ROOT, "archive", "legacy_archive.tar.gz")
    if not os.path.isfile(archive_tar):
        return abs_dir
    rel = os.path.relpath(abs_dir, _REPO_ROOT)
    if not (rel == "archive" or rel.startswith("archive" + os.sep)):
        return abs_dir
    if _LEGACY_ARCHIVE_DIR is None or not os.path.isdir(_LEGACY_ARCHIVE_DIR):
        _LEGACY_ARCHIVE_DIR = tempfile.mkdtemp(prefix="mptrade_legacy_archive_")
        atexit.register(shutil.rmtree, _LEGACY_ARCHIVE_DIR, True)
        with tarfile.open(archive_tar, "r:gz") as tf:
            tf.extractall(_LEGACY_ARCHIVE_DIR, filter="data")
    extracted_dir = os.path.join(_LEGACY_ARCHIVE_DIR, rel)
    return extracted_dir if _has_module_target(extracted_dir, stem) else abs_dir


def load_local_module(
    directory: str, name: str, *, module_key: str | None = None, register: bool = True, reload: bool = False
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
                and (not isinstance(mod_file, str) or not mod_file or os.path.dirname(os.path.abspath(mod_file)) == abs_dir)
            ):
                return mod

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
            repo_base = os.path.basename(_REPO_ROOT)
            for seg in abs_dir.split(os.sep):
                if not seg or seg == repo_base:
                    continue
                cand_dir = os.path.join(_REPO_ROOT, seg)
                if cand_dir != abs_dir and os.path.isdir(cand_dir) and cand_dir not in sys.path:
                    sys.path.insert(0, cand_dir)
            sys.path.insert(0, abs_dir)
            spec = importlib.util.spec_from_file_location(key, os.path.join(abs_dir, f"{stem}.py"))
            mod = importlib.util.module_from_spec(spec)
            setattr(mod, "__mptrade_exec_done__", False)
            prev_key_mod = sys.modules.get(key)
            sys.modules[key] = mod
            try:
                spec.loader.exec_module(mod)
                setattr(mod, "__mptrade_exec_done__", True)
            except Exception:
                if prev_key_mod is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = prev_key_mod
                raise
            if not register:
                if prev_key_mod is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = prev_key_mod
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
        hb_path = str(path)
        payload = build_heartbeat_payload(
            status,
            symbols=symbols,
            intervals=intervals if intervals is not None else {"heartbeat": interval_sec},
            extra=extra,
        )
        _state_io.atomic_write_json(hb_path, payload)
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
    """Thread-safe shared heartbeat writer for any bot or fleet daemon."""
    clean_name = str(module_name or "").strip() if not isinstance(module_name, bool) else ""
    if not clean_name and path is None:
        return float("-inf")
    if path is not None:
        hb_path = str(path)
    else:
        fname = clean_name if clean_name.endswith(".heartbeat") else f"{clean_name}.heartbeat"
        try:

            hb_path = str(resolve_cachedb_filepath(cache_path(fname)))
        except Exception:
            hb_path = os.path.join(_REPO_ROOT, "cachedb", "system", "heartbeat", fname)
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
    """Evaluate heartbeat freshness across ``hb_paths``."""
    if not hb_paths:
        return True, 0.0, 0.0, ""
    hb_only = [p for p in hb_paths if str(p).strip().endswith(".heartbeat")]
    effective_paths = hb_only if hb_only else hb_paths
    mtimes: list[float] = []
    missing_any, stale_file, max_stale_duration = False, "", 0.0
    start_ts = fallback_start_ts if fallback_start_ts is not None else now_ts

    for raw_hbp in effective_paths:
        hbp = str(resolve_cachedb_filepath(raw_hbp))
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


def resolve_heartbeat_paths(
    workspace_dir: str,
    proc_dir: str,
    name: str,
    pattern: str,
    hb_log: str,
    hb_stale_s: float,
    *,
    include_stdout_with_heartbeat_file: bool = False,
) -> tuple[list[str], str]:
    """Resolve canonical heartbeat file paths and stdout log file path for a procs.conf entry."""
    raw_stem = os.path.basename((name or pattern).rstrip("$"))
    raw_stem = raw_stem[:-3] if raw_stem.endswith(".py") else raw_stem
    clean_stem = "".join(c if (c.isalnum() or c in "-_") else "_" for c in raw_stem)
    stdout_log_file = os.path.join(workspace_dir, "logs", f"{clean_stem}.log")

    hb_paths: list[str] = []
    if hb_log:
        for part in split_csv(hb_log):
            if os.path.isabs(part):
                hb_paths.append(part)
            elif part.startswith("cachedb/"):
                hb_paths.append(str(resolve_cachedb_filepath(os.path.join(workspace_dir, part))))
            elif part.startswith("logs/"):
                hb_paths.append(os.path.join(workspace_dir, part))
            else:
                hb_paths.append(os.path.join(proc_dir, part))
        if (
            include_stdout_with_heartbeat_file
            and hb_paths
            and stdout_log_file not in hb_paths
            and any(p.endswith(".heartbeat") for p in hb_paths)
        ):
            hb_paths.append(stdout_log_file)
    elif hb_stale_s > 0:
        hb_paths = [stdout_log_file]

    return hb_paths, stdout_log_file


def parse_procs_manifest(
    workspace_dir: str,
    *,
    venv: str = "myenv",
    roles: tuple[str, ...] = ("bot", "fleet"),
    include_stdout_with_heartbeat_file: bool = False,
    require_cmd: bool = False,
) -> list[dict[str, object]]:
    """Parse ``procs.conf`` from ``workspace_dir`` as the single source of truth for bots and fleet modules."""
    procs: list[dict[str, object]] = []
    procs_conf = os.path.join(workspace_dir, "procs.conf")
    if not os.path.exists(procs_conf):
        return procs

    try:
        with open(procs_conf, "r", encoding="utf-8", errors="replace") as f:
            raw_lines = f.readlines()
        for raw_line in raw_lines:
            line = raw_line.strip()
            if not line or line.startswith("#") or len(parts := line.split("|")) < 7:
                continue
            pat, dr, cmd, label, hblog, hbstale, role = [p.strip() for p in parts[:7]]
            if role not in roles or (require_cmd and not cmd):
                continue
            hb_stale_s = max(0.0, safe_float(hbstale, default=0.0) or 0.0)
            resolved_dir = dr.replace("$ROOT", workspace_dir)
            resolved_cmd = cmd.replace("$ROOT", workspace_dir).replace("$VENV", venv)
            resolved_hblog = hblog.replace("$ROOT", workspace_dir) if hblog else ""
            name = label or pat
            hb_paths, log_file = resolve_heartbeat_paths(
                workspace_dir,
                resolved_dir,
                name,
                pat,
                resolved_hblog,
                hb_stale_s,
                include_stdout_with_heartbeat_file=include_stdout_with_heartbeat_file,
            )
            tokens: set[str] = {name.lower(), *([name[:-3].lower()] if name.lower().endswith(".py") else [])}
            if clean_pat := pat.rstrip("$").strip():
                pat_first = clean_pat.split()[0]
                base = os.path.basename(pat_first)
                tokens.update({clean_pat.lower(), pat_first.lower(), base.lower(), *([base[:-3].lower()] if base.endswith(".py") else [])})
            for tok in (resolved_cmd or "").split():
                if (cleaned := tok.strip("\"'&;")).endswith(".py"):
                    b_tok = os.path.basename(cleaned)
                    tokens.update({cleaned.lower(), b_tok.lower(), b_tok[:-3].lower()})
            aliases = {t for t in tokens if t}
            procs.append({
                "name": name,
                "pattern": pat,
                "clean_pattern": pat.rstrip("$"),
                "dir": resolved_dir,
                "cmd": resolved_cmd,
                "hb_log": resolved_hblog,
                "hb_paths": hb_paths,
                "hb_files": hb_paths,
                "hb_file": hb_paths[0] if hb_paths else "",
                "log_file": log_file,
                "hb_stale_s": hb_stale_s,
                "role": role,
                "aliases": sorted(aliases),
            })
    except Exception:
        pass

    return procs


def clamp_float(
    val: object, minimum: float = 0.0, maximum: float = 1.0, *, default: float = 0.0
) -> float:
    """Safely coerce ``val`` to a float and clamp it to ``[minimum, maximum]``."""
    v = safe_float(val, default=default)
    return max(minimum, min(maximum, default if v is None else v))


def value_diff_to_percent(value1, value2) -> float:
    v1 = safe_float(value1, default=None)
    v2 = safe_float(value2, default=None)
    if v1 is None or v2 is None or v1 == 0.0:
        return 0.0
    return safe_float(((v1 - v2) / v1) * 100.0, default=0.0)


def slope(val1, idx1, val2, idx2) -> float:
    v1 = safe_float(val1, default=None)
    i1 = safe_float(idx1, default=None)
    v2 = safe_float(val2, default=None)
    i2 = safe_float(idx2, default=None)
    if v1 is None or i1 is None or v2 is None or i2 is None or i1 == i2:
        return 0.0
    return safe_float((v2 - v1) / (i2 - i1), default=0.0)


def timestampToTime(timestamp_ms) -> str:
    ts_ms = safe_nonneg_float(timestamp_ms, default=None)
    if ts_ms is None:
        return "N/A"
    try:
        return datetime.fromtimestamp(ts_ms / 1000.0).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError, OverflowError, OSError):
        return "N/A"


@lru_cache(maxsize=256)
def timeToHMS(timestamp_sec) -> str:
    ts_f = normalize_epoch_sec(timestamp_sec, default=None, allow_zero=True)
    if ts_f is None:
        return "N/A"
    try:
        return datetime.fromtimestamp(ts_f).strftime("%H:%M:%S")
    except (TypeError, ValueError, OverflowError, OSError):
        return "N/A"


def secondsToDays(max_age_seconds) -> float:
    return safe_nonneg_float(max_age_seconds, default=0.0) / 86400.0


def secondsToHours(max_age_seconds) -> float:
    return safe_nonneg_float(max_age_seconds, default=0.0) / 3600.0


def gaussian_weights_from_idx(T, idx: int):
    """Return ``t`` as the position axis ``[idx..T-1]`` and ``w`` as its Gaussian weights."""
    import numpy as np

    t_f = safe_pos_float(T, default=None)
    idx_f = safe_float(idx, default=None)
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
        import nacl.signing

        pem_data = _state_io.read_text_file("keys/ed25519_private.pem")
        b64match = re.search(r"-----BEGIN PRIVATE KEY-----(.*?)-----END PRIVATE KEY-----", pem_data, re.DOTALL)
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




def parse_ratio_or_pct(raw: object, pct_cutoff: float = 0.05) -> float:
    """Parse a numeric config value that may be expressed either as a ratio or as a percentage."""
    if isinstance(raw, str) and raw.strip().endswith("%"):
        return require_nonneg_float(raw.strip()[:-1].strip(), name="ratio_or_pct") / 100.0
    val = require_nonneg_float(raw, name="ratio_or_pct")
    cutoff = require_nonneg_float(pct_cutoff, name="pct_cutoff")
    return val / 100.0 if val > cutoff else val


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
    return (px_clean * qty_clean) if (px_clean > 0.0 and qty_clean > 0.0) else default


def clean_scale(
    val: object,
    default: float = 1.0,
    *,
    allow_percent: bool = False,
    min_positive: bool = False,
    clamp: bool = False,
) -> float:
    """Normalize a scale factor into [0.0, 1.0], optionally converting [5.0, 100.0] or '%' percentages."""
    if allow_percent and isinstance(val, str) and val.strip().endswith("%"):
        pct = safe_nonneg_float(val.strip()[:-1].strip(), default=None)
        f = round(pct / 100.0, 4) if (pct is not None and pct <= 100.0) else -1.0
    else:
        f = safe_float(val, -1.0)
        if allow_percent and 5.0 <= f <= 100.0:
            f = round(f / 100.0, 4)
    if f < 0.0 or (min_positive and f <= 0.0):
        return default
    if clamp and f > 1.0:
        return 1.0
    return f if 0.0 <= f <= 1.0 else default


def normalize_epoch_sec(
    val: object, default: float | None = None, *, allow_zero: bool = False, allow_iso: bool = False
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
    """Return ``now_ts`` normalized to epoch seconds (converting ms >= 1e11), or the active runtime clock."""
    f = normalize_epoch_sec(now_ts, default=None, allow_zero=allow_zero)
    if f is not None:
        return f
    clk = getattr(globals().get("_RUNTIME_SERVICES"), "clock_fn", None)
    return float(clk()) if callable(clk) else _time.time()


def resolve_now_ms(now_ms: object = None) -> int:
    """Return ``now_ms`` as a positive integer millisecond timestamp, or ``int(resolve_now_ts() * 1000)``."""
    parsed = safe_pos_int(now_ms, default=None)
    return parsed if parsed is not None else int(resolve_now_ts() * 1000)


_TIMESTAMP_KEYS = (
    "timestamp", "time", "updateTime", "transactTime", "opentm", "closetm", "workingTime",
    "filledAt", "dateExecuted", "dateModified", "createdAt", "dateCreated", "T", "E", "ts", "ts_ms",
)


def _first_mapped(mapping: object, keys: tuple[str, ...], parser: object, default: object = None) -> object:
    """Return the first non-None parsed value from ``mapping`` across ``keys``."""
    if not isinstance(mapping, dict):
        return default
    for key in keys:
        if key in mapping and mapping[key] is not None:
            val = parser(mapping[key])
            if val is not None:
                return val
    return default


def _parse_epoch_or_iso_s(value: object) -> float | None:
    numeric = safe_nonneg_float(value, default=None)
    return numeric if numeric is not None else normalize_epoch_sec(value, default=None, allow_zero=True, allow_iso=True)


def normalize_timestamp_ms(value: object, default: int | None = None) -> int | None:
    """Normalize a raw timestamp (in seconds, milliseconds, or ISO-8601 string) to integer milliseconds."""
    numeric = safe_nonneg_float(value, default=None)
    if numeric is None:
        numeric = _parse_epoch_or_iso_s(value)
    if numeric is None:
        return default
    if numeric == 0.0:
        return 0
    if 1_000_000_000.0 <= numeric < 100_000_000_000.0 or (
        0.0 < numeric < 1_000_000_000.0 and isinstance(value, float) and not numeric.is_integer()
    ):
        return int(round(numeric * 1000.0))
    return int(round(numeric))


def _ts_ms_to_s(ts_ms: float | int) -> float:
    return normalize_epoch_sec(ts_ms, default=0.0, allow_zero=True) or 0.0


def _is_fresh_obs_ts(obs_ts_s: float | None, now_s: float, max_age_s: float) -> bool:
    return obs_ts_s is not None and -5.0 <= (now_s - obs_ts_s) <= max_age_s


def extract_timestamp_ms(order: object, default: int | None = None) -> int | None:
    """Extract and normalize an order/fill/tick timestamp in milliseconds from any venue shape."""
    if isinstance(order, (list, tuple)):
        return normalize_timestamp_ms(order[0], default=default) if order else default
    return _first_mapped(order, _TIMESTAMP_KEYS, lambda v: normalize_timestamp_ms(v, default=None), default)


def extract_timestamp_s(order: object, default: float | None = None) -> float | None:
    """Extract and normalize an order/fill/tick timestamp in seconds from any venue shape."""
    if isinstance(order, (list, tuple)):
        return normalize_epoch_sec(order[0], default=default, allow_zero=True, allow_iso=True) if order else default
    return _first_mapped(order, _TIMESTAMP_KEYS, lambda v: normalize_epoch_sec(v, allow_zero=True, allow_iso=True), default)


_ORDER_ID_KEYS = ("orderId", "order_id", "ordertxid", "txid", "oid", "i", "id")
_PRICE_KEYS = ("price", "px", "limitPx", "fillPrice", "limit", "limitPrice", "averagePrice", "avgPrice", "rate", "L", "p")
_QTY_KEYS = (
    "qty", "quantity", "executedQty", "executed_qty", "filledQuantity", "filledQty", "filled_qty", "vol_exec",
    "sz", "vol", "origQty", "origSz", "orig_qty", "requested_qty", "orderedQuantity", "amount_shares", "l", "z", "q",
)
_COST_EXEC_KEY_PAIRS = (
    ("cummulativeQuoteQty", "executedQty"),
    ("cummulativeQuoteQty", "qty"),
    ("cummulativeQuoteQty", "quantity"),
    ("quoteQty", "qty"),
    ("quoteQty", "quantity"),
    ("cost", "vol_exec"),
    ("cost", "vol"),
    ("filledValue", "filledQuantity"),
    ("Y", "l"),
    ("Z", "z"),
)


def _pos_abs_float(value: object) -> float | None:
    val = abs(safe_float(value, default=0.0))
    return val if val > 0.0 else None


def _nonneg_abs_float(value: object) -> float | None:
    val = safe_float(value, default=None)
    return abs(val) if val is not None else None


def extract_order_id(order: object, default: str | None = None) -> str | None:
    """Extract and normalize a venue order ID from any response shape."""
    if order is None or isinstance(order, bool):
        return default
    if isinstance(order, (list, tuple)):
        for item in order:
            if (cleaned := extract_order_id(item)) is not None:
                return cleaned
        return default
    if hasattr(order, "order_id") and hasattr(order, "state"):
        cleaned = clean_order_id(getattr(order, "order_id", None))
        return cleaned if cleaned is not None else default
    if isinstance(order, dict):
        return _first_mapped(order, _ORDER_ID_KEYS, extract_order_id, default)
    cleaned = clean_order_id(order)
    return cleaned if cleaned is not None else default


def extract_order_side(order: object) -> str:
    """Extract and normalize order side (``'BUY'`` or ``'SELL'``) from any venue shape."""
    if not isinstance(order, dict):
        return ""
    s_norm = _first_mapped(order, ("side", "S", "dir"), lambda v: normalize_side(v) or None)
    if s_norm:
        return s_norm
    descr = order.get("descr")
    for raw_type in (order.get("type"), descr.get("type") if isinstance(descr, dict) else None):
        if (t_norm := normalize_side(raw_type)) in ("BUY", "SELL"):
            return t_norm
    raw_buyer = order.get("isBuyer")
    if raw_buyer is not None:
        parsed_b = _parse_bool_flag(raw_buyer, None)
        if parsed_b is not None:
            return "BUY" if parsed_b else "SELL"
        if isinstance(raw_buyer, str) and (b_norm := raw_buyer.strip().lower()) in ("b", "buy", "s", "sell"):
            return "BUY" if b_norm.startswith("b") else "SELL"
    for sq_key in ("signed_qty", "orderedQuantity", "quantity"):
        sq = safe_float(order.get(sq_key), default=0.0)
        if sq < 0.0:
            return "SELL"
        if sq > 0.0 and (sq_key in ("signed_qty", "orderedQuantity") or "ticker" in order or "limitPrice" in order):
            return "BUY"
    return ""


def extract_order_price(order: object, default: float = 0.0) -> float:
    """Extract order/fill price from any venue shape."""
    if not isinstance(order, dict):
        return default
    val = _first_mapped(order, _PRICE_KEYS, lambda v: safe_pos_float(v, default=None))
    if val is not None:
        return val
    descr = order.get("descr")
    if isinstance(descr, dict) and (val := safe_pos_float(descr.get("price"), default=None)) is not None:
        return val
    for cost_key, exec_key in _COST_EXEC_KEY_PAIRS:
        if cost_key in order and exec_key in order:
            cost_val = _pos_abs_float(order.get(cost_key))
            exec_val = _pos_abs_float(order.get(exec_key))
            if cost_val and exec_val and (res := safe_pos_float(cost_val / exec_val, default=None)) is not None:
                return res
    fills = order.get("fills")
    if isinstance(fills, list) and fills:
        total_quote, total_qty = 0.0, 0.0
        for f in fills:
            if isinstance(f, dict):
                f_px = safe_pos_float(f.get("price", f.get("px", 0.0)), default=None)
                f_qty = _pos_abs_float(f.get("qty", f.get("quantity", f.get("sz", 0.0))))
                if f_px is not None and f_qty is not None:
                    total_quote += f_px * f_qty
                    total_qty += f_qty
        if total_qty > 0 and (res := safe_pos_float(total_quote / total_qty, default=None)) is not None:
            return res
    msg_str = order.get("msg")
    if isinstance(msg_str, str) and "@" in msg_str and msg_str.strip().lower().startswith("filled "):
        if (msg_px := safe_pos_float(msg_str.rsplit("@", 1)[1].strip(), default=None)) is not None:
            return msg_px
    return default


def extract_order_qty(order: object, default: float = 0.0) -> float:
    """Extract order/fill quantity from any venue shape."""
    return _first_mapped(order, _QTY_KEYS, _pos_abs_float, default)


def clamp_price_to_bbo_mid(
    price: float | None,
    bid: object,
    ask: object,
    *,
    max_spread_ratio: float = 0.05,
) -> float | None:
    """Return ``price`` clamped to the BBO mid ``(bid + ask) / 2`` when ``price`` falls outside a tight ``[bid, ask]`` spread."""
    px_f = safe_pos_float(price, default=None) if not isinstance(price, bool) else None
    bid_f = safe_pos_float(bid, default=None) if not isinstance(bid, bool) else None
    ask_f = safe_pos_float(ask, default=None) if not isinstance(ask, bool) else None
    if (
        bid_f is not None
        and ask_f is not None
        and bid_f <= ask_f
        and (ask_f - bid_f) / bid_f <= max_spread_ratio
        and (px_f is None or px_f < bid_f or px_f > ask_f)
    ):
        return round((bid_f + ask_f) * 0.5, 8)
    return px_f


def resolve_sustained_sample_rate(
    cpm: object, symbol: str, *, min_span_sec: float = 5.0
) -> float | None:
    """Resolve sustained sample rate from a current-price manager or compatible test double."""
    if cpm is None:
        return None
    if callable(getattr(type(cpm), "get_sustained_sample_rate", None)):
        return safe_pos_float(
            cpm.get_sustained_sample_rate(symbol, min_span_sec=min_span_sec),
            default=None,
        )
    lock_obj = getattr(cpm, "lock", None)
    with lock_obj if hasattr(lock_obj, "__enter__") else threading.RLock():
        ts_map = getattr(cpm, "_update_timestamps", None)
        dq = ts_map.get(symbol) if isinstance(ts_map, dict) else None
        samples = list(dq) if dq else []
    if len(samples) < 2 or (samples[-1] - samples[0]) < float(min_span_sec):
        return None
    sr_fn = getattr(cpm, "get_sample_rate", None)
    if callable(sr_fn):
        return safe_pos_float(sr_fn(symbol, fallback=None), default=None)
    return safe_pos_float(
        (samples[-1] - samples[0]) / (len(samples) - 1), default=None
    )


def is_cache_fresh(
    last_ts: object, now_ts: float, ttl_sec: object, *, force_refresh: object = False
) -> bool:
    """Return True if cached state at ``last_ts`` is still within ``ttl_sec`` at ``now_ts``."""
    if _parse_bool_flag(force_refresh, False):
        return False
    lt = safe_nonneg_float(last_ts, 0.0)
    ttl = safe_nonneg_float(ttl_sec, 0.0)
    return lt > 0.0 and 0.0 <= (now_ts - lt) < ttl


def is_interval_due(
    current_ts: float, last_ts: object, interval_sec: object, *, force: object = False
) -> bool:
    """Return True if ``force`` is truthy or ``interval_sec`` has elapsed since ``last_ts``."""
    if _parse_bool_flag(force, False):
        return True
    lt = safe_nonneg_float(last_ts, 0.0)
    itv = safe_nonneg_float(interval_sec, 0.0)
    return not (0.0 <= (current_ts - lt) < itv)


def check_cooldown_gate(
    cooldown_map: dict, key: object, now_ts: float, cooldown_sec: float = 1800.0, max_entries: int = 1000
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
    if isinstance(symbols, str):
        return list(split_csv(symbols, upper=True))
    raw = symbols if isinstance(symbols, (list, tuple, set, frozenset)) else ()
    return [s.strip().upper() for s in raw if isinstance(s, str) and s.strip()]


def validate_required_env_keys(
    keys: list[str] | tuple[str, ...], *, env: dict | None = None, label: str = "configuration"
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
    raw_unpatched = list(defaults if defaults else (default_val,))
    if any(
        getattr(d, "__module__", None) == __name__ and getattr(d, "__name__", None) == attr_name
        for d in raw_unpatched
        if d is not None
    ):
        cur_botcore_fn = globals().get(attr_name)
        if cur_botcore_fn is not None:
            raw_unpatched.append(cur_botcore_fn)
    unpatched = tuple(d for d in raw_unpatched if d is not None)

    def _is_valid(cand: object) -> bool:
        return False if (cand is _UNSET or cand is None) else (callable(cand) if require_callable else True)

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
_BUY_SIDE_TOKENS = frozenset(("B", "BUY", "BID", "LONG", "OPEN LONG", "CLOSE SHORT"))
_SELL_SIDE_TOKENS = frozenset(("A", "S", "SELL", "SOLD", "ASK", "SHORT", "CLOSE LONG", "OPEN SHORT"))
INVALID_SCALAR_STRINGS = frozenset((
    "nan", "inf", "-inf", "+inf", "infinity", "-infinity", "+infinity",
    "none", "null",
))
_INVALID_OID_STRINGS = INVALID_SCALAR_STRINGS | frozenset(("0", "0.0", "-0", "+0", "+0.0", "-0.0"))


def normalize_side(side: object, *, require_str: bool = False) -> str:
    """Normalize a scalar order side token to ``'BUY'`` or ``'SELL'`` (or ``''`` when unrecognized)."""
    if side is None or isinstance(side, bool) or (require_str and not isinstance(side, str)):
        return ""
    s = " ".join(str(side).strip().upper().replace("_", " ").replace("-", " ").split())
    if s in _BUY_SIDE_TOKENS:
        return "BUY"
    if s in _SELL_SIDE_TOKENS:
        return "SELL"
    return ""


def clean_order_id(val: object) -> str | None:
    """Validate and normalize a scalar venue order ID to a non-empty string (or ``None``)."""
    if val is None or isinstance(val, (bool, dict, list, tuple, set, frozenset)):
        return None
    if isinstance(val, (int, float)):
        val = exact_pos_int(val, default=None)
        if val is None:
            return None
    if not isinstance(val, (str, int)):
        return None
    text = str(val).strip()
    if not text or text.startswith("-") or text.lower() in _INVALID_OID_STRINGS:
        return None
    if text.endswith(".0") and text[:-2].lstrip("+").isdigit():
        text = text[:-2]
    if text in {"0", "+0"}:
        return None
    return text[1:] if text.startswith("+") and text[1:].isdigit() else text


def clean_order_id_str(val: object) -> str:
    """Return ``clean_order_id(val)`` or ``''`` when empty/invalid."""
    return clean_order_id(val) or ""


def normalize_scalar_order_id(val: object) -> int | str | None:
    """Normalize a scalar order ID, preserving positive integers as ``int`` and string IDs as ``str``."""
    cleaned = clean_order_id(val)
    if cleaned is None:
        return None
    return exact_pos_int(val, default=cleaned) if isinstance(val, (int, float)) else cleaned


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
    return "" if value is None else str(value).replace("|", "/").replace("\r", " ").replace("\n", " ")


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
    return cls_or_name.strip() if isinstance(cls_or_name, str) else ""


async def async_interruptible_sleep(
    stop_event: object, delay: object, step: object = 0.2, *, sleep_fn: object = None
) -> None:
    """Sleep asynchronously up to ``delay`` seconds in ``step`` increments while ``stop_event`` is not set."""
    eff_delay = safe_pos_float(delay, default=None)
    if eff_delay is None:
        return
    eff_step = safe_pos_float(step, default=0.2) or 0.2
    if sleep_fn is None:
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


class RuntimeServices:
    """Explicit dependency-injection container for clock, sleep, retry, snapshot, and external transport policies."""

    def __init__(
        self,
        *,
        clock_fn: object = None,
        sleep_fn: object = None,
        stale_cache_retry_delay_sec: float = 1.5,
        enforce_rth_gate: bool = True,
        allow_live_network_backfill: bool = True,
        allow_unmocked_llm_queries: bool = True,
        allow_shared_disk_cache: bool = True,
    ) -> None:
        self.clock_fn = clock_fn if callable(clock_fn) else None
        self.sleep_fn = sleep_fn if callable(sleep_fn) else None
        self.stale_cache_retry_delay_sec = safe_nonneg_float(stale_cache_retry_delay_sec, default=1.5)
        self.enforce_rth_gate = _parse_bool_flag(enforce_rth_gate, True)
        self.allow_live_network_backfill = _parse_bool_flag(allow_live_network_backfill, True)
        self.allow_unmocked_llm_queries = _parse_bool_flag(allow_unmocked_llm_queries, True)
        self.allow_shared_disk_cache = _parse_bool_flag(allow_shared_disk_cache, True)


_RUNTIME_SERVICES: RuntimeServices = (
    globals()["_RUNTIME_SERVICES"]
    if isinstance(globals().get("_RUNTIME_SERVICES"), RuntimeServices)
    else RuntimeServices()
)


def configure_runtime_services(**kwargs: object) -> RuntimeServices:
    """Update fields on the active ``RuntimeServices`` container and return it."""
    for k, v in kwargs.items():
        if not hasattr(_RUNTIME_SERVICES, k):
            raise AttributeError(f"Unknown RuntimeServices attribute: {k!r}")
        if k in ("clock_fn", "sleep_fn"):
            setattr(_RUNTIME_SERVICES, k, v if callable(v) else None)
        elif k == "stale_cache_retry_delay_sec":
            setattr(_RUNTIME_SERVICES, k, safe_nonneg_float(v, default=1.5))
        else:
            setattr(_RUNTIME_SERVICES, k, _parse_bool_flag(v, getattr(_RUNTIME_SERVICES, k, True)))
    return _RUNTIME_SERVICES


SIMULATED_PROVIDER_NAMES: frozenset[str] = frozenset(
    {"replay", "backtest", "sim", "mock", "paper_sim", "test"}
)


def is_simulated_provider_name(provider: object) -> bool:
    """Return True when ``provider`` identifies a simulated/replay/backtest/mock execution adapter."""
    return bool(
        provider
        and not isinstance(provider, bool)
        and str(provider).strip().lower() in SIMULATED_PROVIDER_NAMES
    )


def is_replay_or_backtest_env(*extra_keys: str, env: dict | None = None) -> bool:
    """Return True when ``BACKTEST_MODE``, ``REPLAY_MODE``, or any ``extra_keys`` flag is enabled."""
    return any_bool_env("BACKTEST_MODE", "REPLAY_MODE", *extra_keys, env=env)


def is_gemini_llm_disabled(env: dict | None = None) -> bool:
    """Return True when Gemini LLM queries are suppressed via ``DISABLE_GEMINI_LLM``, ``BACKTEST_MODE``, or ``REPLAY_MODE``."""
    return is_replay_or_backtest_env("DISABLE_GEMINI_LLM", env=env)


def is_offline_disk_cache_enabled(env: dict | None = None) -> bool:
    """Return True when ``MPTRADE_OFFLINE_DISK_CACHE`` is enabled for offline OHLC disk reads."""
    return bool_env("MPTRADE_OFFLINE_DISK_CACHE", env=env, default=False)


def is_auto_start_websockets_enabled(env: dict | None = None) -> bool:
    """Return True when ``BINANCE_AUTO_START_WEBSOCKETS`` permits background WS startup (default True)."""
    return bool_env("BINANCE_AUTO_START_WEBSOCKETS", env=env, default=True)


def is_rth_gate_enabled(override: bool | None = None, *, rth_fn: object = None) -> bool:
    """Return whether US equities RTH session gating is active under the current runtime policy."""
    if override is not None:
        return bool(override)
    if rth_fn is not None and rth_fn is not is_us_rth_open:
        return True
    if bool_env("MPTRADE_FORCE_RTH_GATE", default=False):
        return True
    return bool(_RUNTIME_SERVICES.enforce_rth_gate)


def is_live_network_backfill_enabled(override: bool | None = None) -> bool:
    """Return whether opportunistic on-demand live network backfills are enabled."""
    if override is not None:
        return bool(override)
    env_flag = os.environ.get("MPTRADE_ALLOW_LIVE_BACKFILL")
    if env_flag is not None and str(env_flag).strip() != "":
        return bool(_parse_bool_flag(env_flag, default=True))
    return bool(_RUNTIME_SERVICES.allow_live_network_backfill)


def is_unmocked_llm_query_allowed(override: bool | None = None) -> bool:
    """Return whether unmocked external LLM CLI/REST queries are permitted."""
    if override is not None:
        return bool(override)
    return bool(_RUNTIME_SERVICES.allow_unmocked_llm_queries)


def is_shared_disk_cache_enabled(override: bool | None = None) -> bool:
    """Return whether default cross-process disk caches are enabled when no explicit path is injected."""
    if override is not None:
        return bool(override)
    return bool(_RUNTIME_SERVICES.allow_shared_disk_cache)


def get_stale_cache_retry_delay_sec(override: float | None = None) -> float:
    """Return the retry delay in seconds for stale account cache order retries."""
    if override is not None:
        val = safe_nonneg_float(override, default=None)
        if val is not None:
            return val
    return float(_RUNTIME_SERVICES.stale_cache_retry_delay_sec)


# ═════════════════════════════════════════════════════════════════════════════
# 5. MERGED MODULE ALIAS REGISTRY (Backward-Compatible sys.modules Bindings)
# ═════════════════════════════════════════════════════════════════════════════


def register_module_aliases(module_name: str, *aliases: str) -> None:
    """Bind ``aliases`` in ``sys.modules`` (and parent package attributes) to ``sys.modules[module_name]``."""
    self_mod = sys.modules.get(module_name)
    if self_mod is None:
        return
    self_file = getattr(self_mod, "__file__", None)
    for alias in aliases:
        prev = sys.modules.get(alias)
        prev_loader = getattr(prev, "__loader__", None)
        if (
            prev is None
            or prev is self_mod
            or getattr(prev, "__file__", None) == self_file
            or getattr(prev_loader, "__class__", None).__name__ == "_MergedModuleAliasLoader"
        ):
            sys.modules[alias] = self_mod
            if "." in alias:
                parent_name, child_attr = alias.rsplit(".", 1)
                parent_pkg = sys.modules.get(parent_name)
                if parent_pkg is not None:
                    setattr(parent_pkg, child_attr, self_mod)


_MERGED_MODULE_TARGETS = {
    "hyperliquid.signals": "hyperliquid.signals",
    "symbols": "instrument_registry",
}


class _MergedModuleAliasLoader:
    def __init__(self, alias_name: str, target_name: str) -> None:
        self.alias_name = alias_name
        self.target_name = target_name

    def create_module(self, spec):
        return None

    def exec_module(self, module) -> None:
        if self.target_name == "hyperliquid.signals":
            hl_dir = os.path.join(_REPO_ROOT, "hyperliquid")
            is_reload = module is sys.modules.get("signals") and getattr(module, "__mptrade_exec_done__", False)
            target_mod = load_local_module(hl_dir, "signals", reload=is_reload)
            for alias_k in ("signals", "hyperliquid.signals"):
                sys.modules[alias_k] = target_mod
            hl_pkg = sys.modules.get("hyperliquid")
            if hl_pkg is not None:
                setattr(hl_pkg, "signals", target_mod)
            sys.modules[self.alias_name] = target_mod
            return

        target_mod = importlib.import_module(self.target_name)
        if "." in self.alias_name:
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
                return importlib.util.spec_from_file_location(fullname, py_file)
            pkg_dir = os.path.join(resolved_dir, stem)
            if os.path.isdir(pkg_dir):
                init_py = os.path.join(pkg_dir, "__init__.py")
                if os.path.isfile(init_py):
                    return importlib.util.spec_from_file_location(
                        fullname, init_py, submodule_search_locations=[pkg_dir]
                    )
                spec = importlib.machinery.ModuleSpec(fullname, None, is_package=True)
                spec.submodule_search_locations = [pkg_dir]
                return spec
        if target_name is None:
            return None

        return importlib.util.spec_from_loader(
            fullname,
            _MergedModuleAliasLoader(fullname, target_name),
        )


if not any(
    getattr(f, "__name__", "") == "_MergedModuleAliasFinder" or isinstance(f, _MergedModuleAliasFinder)
    for f in sys.meta_path
):
    sys.meta_path.append(_MergedModuleAliasFinder())


# ═════════════════════════════════════════════════════════════════════════════
# 10. CANONICAL LLM TOPIC IDENTITY REGISTRY
# ═════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class TopicSpec:
    """Permanent identity and capabilities of an LLM topic thread."""

    title: str
    can_trade: bool = False
    can_self_heal: bool = False
    asset_scope: str = "all"
    default_preferred_model: str | None = None
    default_timeout_sec: float = 120.0
    model_env_key: str | None = None

    @property
    def preferred_model(self) -> str | None:
        if self.model_env_key:
            raw = os.environ.get(self.model_env_key)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
        return self.default_preferred_model

    @property
    def bracket_tag(self) -> str:
        """Canonical '[auto_...]' tag from the thread title."""
        m = re.search(r"(\[[^\]]+\])", self.title)
        return m.group(1) if m else self.title

    @property
    def ntfy_prefix(self) -> str:
        """Canonical '<icon> [auto_...]' prefix from the beginning of the thread title."""
        m = re.match(r"^([^\[]*\[[^\]]+\])", self.title)
        return m.group(1).strip() if m else self.title


TOPIC_SPECS: dict[str, TopicSpec] = {
    "auto_technical_repair": TopicSpec(
        "🛠 [auto_technical_repair] System Audit & Code Self-Heal (03:35)",
        can_self_heal=True,
        default_preferred_model="gemini-4-argon-medium",
        default_timeout_sec=120.0,
        model_env_key="LLM_DEFAULT_MODEL",
    ),
    "auto_cryptotrade_executor": TopicSpec(
        "🎯 [auto_cryptotrade_executor] Autonomous Crypto Trade Decisions (Binance/Kraken/HL 6m)",
        can_trade=True,
        asset_scope="crypto",
        default_preferred_model="gemini-4-argon-medium",
        default_timeout_sec=90.0,
        model_env_key="LLM_DEFAULT_MODEL",
    ),
    "auto_sharetrade_executor": TopicSpec(
        "📈 [auto_sharetrade_executor] Autonomous Share Trade Decisions (Trading212 RTH 30m)",
        can_trade=True,
        asset_scope="equities",
        default_preferred_model="gemini-4-argon-medium",
        default_timeout_sec=90.0,
        model_env_key="LLM_DEFAULT_MODEL",
    ),
    "macro_geopolitical": TopicSpec(
        "🌍 [auto_macro_geopolitical] Geopolitical Crisis & Breaking News Brake",
        default_preferred_model="gpt-oss-120b-medium",
        default_timeout_sec=120.0,
        model_env_key="LLM_LOW_STAKE_MODEL",
    ),
    "macro_sentiment": TopicSpec(
        "🧭 [auto_macro_sentiment] Fear & Greed, Breadth & Market Bias (3h)",
        default_preferred_model="gemini-4-argon-medium",
        default_timeout_sec=120.0,
        model_env_key="LLM_DEFAULT_MODEL",
    ),
    "guard_high_stake": TopicSpec(
        "🛡 [auto_guard_large_order] Pre-Trade Veto for Large Orders (>=700 EUR)",
        default_preferred_model="gemini-4-argon-high",
        default_timeout_sec=120.0,
        model_env_key="LLM_HIGH_STAKE_MODEL",
    ),
    "guard_post_stop": TopicSpec(
        "🛡 [auto_guard_post_stop] Pre-Trade Veto for Post-Stop Re-Buy Bounce",
        default_preferred_model="gemini-4-argon-medium",
        default_timeout_sec=120.0,
        model_env_key="LLM_DEFAULT_MODEL",
    ),
}

TOPIC_ALIASES: dict[str, str] = {
    "auto_trade_executor": "auto_cryptotrade_executor",
}


def canonical_topic_key(topic_key: object) -> str:
    """Resolve legacy topic aliases (e.g. auto_trade_executor -> auto_cryptotrade_executor)."""
    if not isinstance(topic_key, str):
        return ""
    cleaned = topic_key.strip()
    return TOPIC_ALIASES.get(cleaned, cleaned)


def topic_spec(topic_key: str) -> TopicSpec:
    """Spec of a registered topic; unknown topics fail fast."""
    resolved = canonical_topic_key(topic_key)
    if resolved not in TOPIC_SPECS:
        raise KeyError(f"Unknown LLM topic '{topic_key}'; register it in TOPIC_SPECS")
    return TOPIC_SPECS[resolved]


def compute_venue_nav_breakdown(
    *,
    positions: object = None,
    quote_cash: object = None,
    venue_balances_state: object = None,
    live_prices: object = None,
    venue_overrides: object = None,
    exclude_venues: tuple[str, ...] = (),
    scope: str = "all",
) -> dict[str, object]:
    """Canonical SSOT multi-venue NAV calculator and position/cash deduplicator."""

    excl_set = {normalize_venue_name(v) or str(v).strip().lower() for v in (exclude_venues or ()) if v}
    scope_norm = str(scope or "all").strip().lower()
    live_src = (
        live_prices.get("items")
        if isinstance(live_prices, dict) and isinstance(live_prices.get("items"), dict)
        else (live_prices if isinstance(live_prices, dict) else {})
    )
    live_px_map = {
        raw_k.strip().upper(): px
        for raw_k, raw_v in live_src.items()
        if isinstance(raw_k, str) and raw_k.strip() and (px := extract_current_price_item(raw_v)) is not None
    }

    pos_rows: list[dict] = []
    cash_items: list[tuple[str, object]] = []

    if isinstance(venue_balances_state, dict):
        for key, entry in venue_balances_state.items():
            if not isinstance(key, str) or not isinstance(entry, dict):
                continue
            parts = key.split(":", 2)
            v_cand = entry.get("venue") or (parts[1] if len(parts) > 1 else "")
            if key.startswith("cash:"):
                a_cand = entry.get("asset") or (parts[2] if len(parts) > 2 else "USD")
                cash_items.append((f"{v_cand}:{a_cand}", {**entry, "venue": v_cand, "asset": a_cand}))
            elif key.startswith("pos:"):
                s_cand = entry.get("symbol") or (parts[2] if len(parts) > 2 else key)
                pos_rows.append({**entry, "venue": v_cand, "symbol": s_cand})

    if isinstance(positions, (list, tuple)):
        pos_rows.extend(p for p in positions if isinstance(p, dict))
    if isinstance(quote_cash, dict):
        cash_items.extend((str(ck or ""), cv) for ck, cv in quote_cash.items())

    def _matches_scope(v_name: str, sym_u: str = "") -> bool:
        if v_name in excl_set:
            return False
        if scope_norm not in ("equities", "crypto"):
            return True
        is_eq = is_equity_venue_or_suffix(sym_u, v_name)
        return is_eq if scope_norm == "equities" else (not is_eq)

    grouped_pos: dict[tuple[str, str], list[dict]] = {}
    for p in pos_rows:
        v_name = normalize_venue_name(p.get("venue")) or "unknown"
        sym_raw = str(p.get("symbol") or "").strip().upper()
        if not _matches_scope(v_name, sym_raw):
            continue
        is_eq_p = is_equity_venue_or_suffix(sym_raw, v_name)
        base_p = canonical_base_asset(p.get("base") or sym_raw) or sym_raw or f"IDX_{len(grouped_pos)}"
        grouped_pos.setdefault((v_name, f"EQ:{base_p}" if is_eq_p else base_p), []).append(p)

    def _lookup_live_px(sym_u: str) -> float | None:
        if not live_px_map or not sym_u:
            return None
        if sym_u in live_px_map:
            return live_px_map[sym_u]
        if has_equity_suffix(sym_u):
            return None
        for cand in cache_symbol_variants(sym_u, quotes=compatible_quote_suffixes(sym_u)):
            if isinstance(cand, str) and not has_equity_suffix(cand) and cand in live_px_map:
                return live_px_map[cand]
        return None

    def _resolve_pos_free_usd(gp: dict) -> float:
        sym_u = str(gp.get("symbol") or "").strip().upper()
        if (live_px := _lookup_live_px(sym_u)) is not None and (
            fb_live := safe_nonneg_float(gp.get("free_base"), None)
        ) is not None:
            return round(fb_live * live_px, 2)
        if (val_u := safe_nonneg_float(gp.get("value_usd"), None)) is not None:
            return val_u
        return round(
            (safe_nonneg_float(gp.get("free_base"), 0.0) or 0.0)
            * (safe_nonneg_float(gp.get("price"), 0.0) or 0.0),
            2,
        )

    def _resolve_pos_locked_usd(gp: dict) -> float:
        sym_u = str(gp.get("symbol") or "").strip().upper()
        if (live_px := _lookup_live_px(sym_u)) is not None and (
            lb_live := safe_pos_float(gp.get("locked_base_qty"), None)
        ) is not None:
            return round(lb_live * live_px, 2)
        return safe_nonneg_float(gp.get("locked_base_usd"), 0.0) or 0.0

    venue_pos_usd: dict[str, float] = {}
    grouped_positions_out: list[tuple[str, dict, float]] = []
    for (v_name, _), grp_list in grouped_pos.items():
        free_val_u = max((_resolve_pos_free_usd(gp) for gp in grp_list), default=0.0)
        locked_by_sym: dict[str, float] = {}
        for idx_g, gp in enumerate(grp_list):
            s_g = str(gp.get("symbol") or f"SYM_{idx_g}").strip().upper()
            locked_by_sym[s_g] = max(locked_by_sym.get(s_g, 0.0), _resolve_pos_locked_usd(gp))
        val_u = free_val_u + sum(locked_by_sym.values())
        venue_pos_usd[v_name] = venue_pos_usd.get(v_name, 0.0) + val_u
        grouped_positions_out.append((v_name, grp_list[0], val_u))

    dedup_cash: dict[tuple[str, str], tuple[float, float]] = {}
    for c_key, c in cash_items:
        k_parts = str(c_key or "").split(":", 1)
        v_fallback = normalize_venue_name(k_parts[0])
        if isinstance(c, dict):
            v_name = normalize_venue_name(c.get("venue")) or v_fallback or "unknown"
            c_asset_u = str(
                c.get("asset") or (k_parts[1] if len(k_parts) > 1 and k_parts[1].strip() else c_key or "USD")
            ).strip().upper() or "USD"
            f_u = (
                safe_pos_float(c.get("free"), default=None)
                or safe_pos_float(c.get("free_quote_usd"), default=None)
                or safe_pos_float(c.get("free_usd"), default=0.0)
                or 0.0
            )
            l_u = safe_pos_float(c.get("locked_quote_usd"), default=0.0) or 0.0
        else:
            f_u = safe_pos_float(c, default=0.0) or 0.0
            if f_u <= 0.0:
                continue
            v_name = v_fallback or "unknown"
            c_asset_u = k_parts[1].strip().upper() if len(k_parts) > 1 and k_parts[1].strip() else "USD"
            l_u = 0.0
        if not _matches_scope(v_name):
            continue
        prev_f, prev_l = dedup_cash.get((v_name, c_asset_u), (0.0, 0.0))
        dedup_cash[(v_name, c_asset_u)] = (max(prev_f, f_u), max(prev_l, l_u))

    venue_free_cash_usd: dict[str, float] = {}
    venue_locked_quote_usd: dict[str, float] = {}
    venue_cash_usd: dict[str, float] = {}
    for (v_name, _), (f_u, l_u) in dedup_cash.items():
        venue_free_cash_usd[v_name] = venue_free_cash_usd.get(v_name, 0.0) + f_u
        venue_locked_quote_usd[v_name] = venue_locked_quote_usd.get(v_name, 0.0) + l_u
        venue_cash_usd[v_name] = venue_cash_usd.get(v_name, 0.0) + f_u + l_u

    venues_usd: dict[str, float] = {}
    venue_breakdown: dict[str, dict[str, float]] = {}
    for v_k in sorted(set(venue_pos_usd) | set(venue_cash_usd)):
        p_u = round(venue_pos_usd.get(v_k, 0.0), 2)
        fc_u = round(venue_free_cash_usd.get(v_k, 0.0), 2)
        lq_u = round(venue_locked_quote_usd.get(v_k, 0.0), 2)
        c_u = round(venue_cash_usd.get(v_k, 0.0), 2)
        v_tot = round(p_u + c_u, 2)
        if v_tot > 0.0:
            venues_usd[v_k] = v_tot
            venue_breakdown[v_k] = {
                "pos_usd": p_u,
                "free_cash_usd": fc_u,
                "locked_quote_usd": lq_u,
                "cash_usd": c_u,
                "total_usd": v_tot,
            }

    if isinstance(venue_overrides, dict):
        for ov_k, ov_v in venue_overrides.items():
            v_norm = normalize_venue_name(ov_k) or str(ov_k or "").strip().lower()
            if not v_norm or not _matches_scope(v_norm):
                continue
            ov_f = safe_pos_float(ov_v, default=0.0) or 0.0
            if ov_f > venues_usd.get(v_norm, 0.0):
                venues_usd[v_norm] = round(ov_f, 2)
                if v_norm in venue_breakdown:
                    venue_breakdown[v_norm]["total_usd"] = round(ov_f, 2)

    venues_usd = {k: venues_usd[k] for k in sorted(venues_usd)}
    invested_usd = round(sum(venue_pos_usd.values()), 2)
    free_cash_usd = round(sum(venue_free_cash_usd.values()), 2)
    locked_quote_usd = round(sum(venue_locked_quote_usd.values()), 2)
    total_quote_reserve_usd = round(free_cash_usd + locked_quote_usd, 2)
    portfolio_total_usd = round(max(sum(venues_usd.values()), invested_usd + total_quote_reserve_usd), 2)

    return {
        "venues_usd": venues_usd,
        "venue_breakdown": venue_breakdown,
        "grouped_positions": grouped_positions_out,
        "venue_pos_usd": venue_pos_usd,
        "venue_free_cash_usd": venue_free_cash_usd,
        "venue_locked_quote_usd": venue_locked_quote_usd,
        "venue_cash_usd": venue_cash_usd,
        "invested_usd": invested_usd,
        "free_cash_usd": free_cash_usd,
        "locked_quote_usd": locked_quote_usd,
        "total_quote_reserve_usd": total_quote_reserve_usd,
        "portfolio_total_usd": portfolio_total_usd,
    }


@dataclass(frozen=True)
class ClosedPriceSeries:
    """Completed closes with the timestamp of the newest completed candle."""

    closes: tuple
    interval_min: int
    last_closed_at: float | None = None
    timestamps: tuple = ()
    volumes: tuple = ()

    @classmethod
    def from_raw_candles(
        cls,
        raw_candles: object,
        interval_min: int,
        close_getter: object,
        ts_ms_getter: object,
        vol_getter: object = None,
    ) -> "ClosedPriceSeries":
        completed = list(raw_candles or ())[:-1]  # type: ignore[arg-type]
        valid = [
            (item, c_f)
            for item in completed
            if (c_f := safe_pos_float(close_getter(item), default=None)) is not None  # type: ignore[operator]
        ]
        closes = tuple(c_f for _item, c_f in valid)
        ts_list = []
        ts_ok = bool(valid)
        for item, _ in valid:
            t_ms = safe_pos_float(ts_ms_getter(item), default=None)  # type: ignore[operator]
            if t_ms is None:
                ts_ok = False
                break
            ts_list.append(t_ms / 1000.0)
        timestamps = tuple(ts_list) if ts_ok else ()
        volumes = (
            tuple(safe_nonneg_float(vol_getter(item), default=0.0) or 0.0 for item, _ in valid)
            if callable(vol_getter) and valid
            else ()
        )
        return cls(
            closes,
            safe_pos_int(interval_min, 1),
            timestamps[-1] if timestamps else None,
            timestamps,
            volumes,
        )


def stop_monitoring_thread(
    instance: object, label: str, *, timeout: float = 5.0
) -> None:
    """Stop a background monitoring loop backed by ``_running`` and ``_thread``."""
    setattr(instance, "_running", False)
    thread = getattr(instance, "_thread", None)
    if thread and thread is not threading.current_thread():
        thread.join(timeout=safe_nonneg_float(timeout, default=5.0))
    print(f"[{label}] Monitoring stopped")


_REEXPORT_MODULE_NAMES = (
    "utils",
    "intelligence.macro_sentiment_ssot",
    "macro_sentiment_ssot",
    "offline.backtests.evaluation",
)


def _sync_reexport_modules() -> None:
    """Keep loaded re-export modules bound to the active ``botcore`` function objects."""
    for mod_name in _REEXPORT_MODULE_NAMES:
        reexport_mod = sys.modules.get(mod_name)
        if reexport_mod is None:
            continue
        for attr_name, attr_val in list(reexport_mod.__dict__.items()):
            if (
                not attr_name.startswith("__")
                and attr_name in globals()
                and callable(attr_val)
                and getattr(attr_val, "__module__", None) == __name__
            ):
                setattr(reexport_mod, attr_name, globals()[attr_name])


_sync_reexport_modules()




