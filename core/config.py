"""core/config.py -- Canonical SSOT configuration, dotenv/INI loading, CLI parsers, and trade gate."""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import configparser
import contextlib
import os
import sys
import threading
import time as _time

from cachepaths import (
    _parse_bool_flag,
    clean_nonempty_str,
    exact_int,
    exact_nonneg_int,
    exact_pos_int,
    safe_float,
    safe_nonneg_float,
    safe_pos_float,
)
from core.lock import log as _default_log
from core.math_coercion import _UNSET, lru_dict_put
import state_io as _state_io

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
_NONFINITE_FLOAT_TOKENS = frozenset(
    {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}
)


def _bc_attr(name: str, default: object) -> object:
    bc = sys.modules.get("botcore")
    if bc is not None and name in bc.__dict__:
        return bc.__dict__[name]
    return default


def _eff_root_config_path() -> str:
    return str(_bc_attr("_ROOT_CONFIG_PATH", _ROOT_CONFIG_PATH))


def _eff_log(msg: str) -> None:
    fn = _bc_attr("log", _default_log)
    if callable(fn):
        fn(msg)


def _strip_env_value_quotes_and_comment(val: str) -> str:
    """Strip inline `#` comments and outer quotes from the value portion of an env setting."""
    val = val.strip()
    if val.startswith(('"', "'")):
        q = val[0]
        close_idx = val.find(q, 1)
        if close_idx != -1:
            tail = val[close_idx + 1 :].strip()
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
    eq_idx, hash_idx = line.find("="), line.find("#")
    err_msg: str | None = None
    if eq_idx == -1 or (hash_idx != -1 and hash_idx < eq_idx):
        err_msg = f"Malformed configuration line missing '=': {raw.rstrip()!r}"
    else:
        key, _, val = line.partition("=")
        key = key.strip()
        if key.startswith("export") and (len(key) == 6 or key[6].isspace()):
            key = key[6:].strip()
        if not key:
            err_msg = f"Empty configuration key in line: {raw.rstrip()!r}"
    if err_msg is not None:
        if strict:
            raise ValueError(err_msg)
        return None
    return key, _strip_env_value_quotes_and_comment(val)


def _dotenv_pairs(path: str | os.PathLike, *, strict: bool = False) -> tuple[list[tuple[str, str]], bool]:
    """Parse shared dotenv syntax once and report whether the file was read successfully."""
    if not path or isinstance(path, bool) or not isinstance(path, (str, os.PathLike)):
        return [], False
    p_str = os.fspath(path)
    if not p_str or not os.path.exists(p_str):
        return [], False
    try:
        raw_text = _state_io.read_text_file(p_str)
        pairs = [parsed for raw in raw_text.splitlines() if (parsed := parse_dotenv_line(raw, strict=strict)) is not None]
        return pairs, True
    except OSError as e:
        _eff_log(f"  ! cannot read {path}: {e}")
        return [], False


def _log_dotenv_loaded_once(path: str, *, override: bool = False) -> None:
    abs_path = os.path.abspath(path)
    if override or abs_path not in _LOADED_DOTENV_PATHS:
        _LOADED_DOTENV_PATHS.add(abs_path)
        _eff_log(f"  .env loaded from {path}")


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
    (_ROOT_CONFIG_KEYS.add if is_root else _ROOT_CONFIG_KEYS.discard)(key)
    if is_root:
        _ROOT_CONFIG_VALUES[key] = value
    else:
        _ROOT_CONFIG_VALUES.pop(key, None)


def load_dotenv(path: str | os.PathLike | None = None, override: bool = False) -> None:
    """Load KEY=VALUE entries from ``path`` (defaulting to root ``config.env`` when ``None``) without overriding existing env."""
    root_cfg = _eff_root_config_path()
    target = root_cfg if path is None else os.fspath(path)
    override = _parse_bool_flag(override, False)
    pairs, loaded = _dotenv_pairs(target)
    if not loaded:
        return
    is_root = os.path.abspath(target) == os.path.abspath(root_cfg)
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
    root_cfg = _eff_root_config_path()
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

    for path, is_root_cfg in ((cfg_path, os.path.abspath(cfg_path) == os.path.abspath(root_cfg)), (env_path, False)):
        pairs, loaded = _dotenv_pairs(path)
        if loaded:
            for key, value in pairs:
                if key and key not in protected:
                    _set_dotenv_env_key(key, value, is_root=is_root_cfg)
            _log_dotenv_loaded_once(path, override=bool(override))


def load_root_config(*, include_secrets: bool = False, override: bool = False) -> None:
    """Load repository root ``config.env`` (and root ``.env`` when ``include_secrets=True``) via the canonical SSOT."""
    root_cfg = _eff_root_config_path()
    if include_secrets:
        load_env_stack(os.path.join(_REPO_ROOT, ".env"), root_cfg, override=override)
    else:
        load_dotenv(root_cfg, override=override)


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
    *env_files: str | os.PathLike,
    override_prefixes: tuple[str, ...] | list[str] = (),
    fallback_prefixes: tuple[str, ...] | list[str] = (),
    override_all_env: bool = False,
) -> dict[str, str]:
    """Load ``env_files`` in lowest-to-highest precedence order, applying ``os.environ`` overrides and fallbacks."""
    cfg_out: dict[str, str] = {}
    for path_val in env_files:
        if path_val is None:
            continue
        p_str = os.fspath(path_val)
        if p_str and os.path.exists(p_str):
            cfg_out.update(dict(parse_dotenv(p_str)))
    if override_all_env:
        cfg_out.update(os.environ)
        return cfg_out
    ov_tuple, fb_tuple = tuple(override_prefixes), tuple(fallback_prefixes)
    for k, v in os.environ.items():
        if (ov_tuple and k.startswith(ov_tuple)) or (fb_tuple and k.startswith(fb_tuple) and k not in cfg_out):
            cfg_out[k] = v
    return cfg_out


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


_VENUE_DIR_UNSET = globals().get("_VENUE_DIR_UNSET") or object()


def bind_isolated_venue_cfg_loader(
    default_dir: str | os.PathLike[str],
    override_prefixes: tuple[str, ...] | list[str] = (),
    fallback_prefixes: tuple[str, ...] | list[str] = (),
    *,
    dir_kwarg: str | None = None,
):
    """Return a venue-specific isolated config loader bound to ``default_dir`` and ``override_prefixes``."""
    ov_tuple, fb_tuple = tuple(override_prefixes), tuple(fallback_prefixes)

    def _loader(
        venue_dir: object = _VENUE_DIR_UNSET,
        *,
        config_path: str | os.PathLike[str] | None = None,
        env_path: str | os.PathLike[str] | None = None,
        **dir_kw: str | os.PathLike[str],
    ) -> dict[str, str]:
        eff_dir: object = venue_dir
        if dir_kw:
            if not dir_kwarg or set(dir_kw.keys()) != {dir_kwarg}:
                bad = next(k for k in dir_kw if k != dir_kwarg)
                raise TypeError(f"unexpected keyword argument {bad!r}")
            if venue_dir is not _VENUE_DIR_UNSET:
                raise TypeError(f"got multiple values for directory argument ({dir_kwarg!r})")
            eff_dir = dir_kw[dir_kwarg]
        resolved_dir = default_dir if (eff_dir is _VENUE_DIR_UNSET or eff_dir is None) else eff_dir
        return load_isolated_venue_cfg(
            resolved_dir,  # type: ignore[arg-type]
            config_path=config_path,
            env_path=env_path,
            override_prefixes=ov_tuple,
            fallback_prefixes=fb_tuple,
        )

    return _loader


_INI_CONFIG_CACHE: dict[str, tuple[tuple[int, int, bool], configparser.ConfigParser]] = (
    globals().get("_INI_CONFIG_CACHE") or {}
)
_INI_CONFIG_LOCK = globals().get("_INI_CONFIG_LOCK") or threading.RLock()


def empty_ini_config(*, inline_comments: bool = True) -> configparser.ConfigParser:
    """Create an empty ``ConfigParser(interpolation=None)`` with canonical inline comment handling."""
    prefixes = ("#", ";") if inline_comments else ()
    return configparser.ConfigParser(interpolation=None, inline_comment_prefixes=prefixes)


def read_config_text(path: str | os.PathLike) -> str:
    """Read a UTF-8 configuration file via the canonical ``state_io`` text reader."""
    if isinstance(path, bool) or not isinstance(path, (str, os.PathLike)) or not str(path).strip():
        raise ValueError(f"Invalid configuration path: {path!r}")
    return _state_io.read_text_file(os.path.abspath(os.fspath(path)))


def load_ini_config(
    path: str | os.PathLike, *, must_exist: bool = True, inline_comments: bool = True
) -> configparser.ConfigParser:
    """Load and cache an INI ``.conf`` file by ``(mtime_ns, size)`` using ``ConfigParser(interpolation=None)``."""
    if isinstance(path, bool) or not isinstance(path, (str, os.PathLike)) or not str(path).strip():
        raise ValueError(f"Invalid INI configuration path: {path!r}")
    abs_path = os.path.abspath(os.fspath(path))
    if not os.path.exists(abs_path):
        if must_exist:
            raise FileNotFoundError(f"Missing mandatory configuration file: {path}")
        return empty_ini_config(inline_comments=inline_comments)
    st = os.stat(abs_path)
    sig = (st.st_mtime_ns, st.st_size, bool(inline_comments))
    with _INI_CONFIG_LOCK:
        cached = _INI_CONFIG_CACHE.get(abs_path)
        if cached is not None and cached[0] == sig:
            return cached[1]
        cp = empty_ini_config(inline_comments=inline_comments)
        cp.read_string(_state_io.read_text_file(abs_path))
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


def _clean_env_str(val: object) -> str:
    """Strip inline `#` comment and surrounding whitespace from a raw env value."""
    if val is None or isinstance(val, bool):
        return ""
    return _strip_env_value_quotes_and_comment(str(val))


def _raw_env_val(key: str, env: object = None) -> object:
    return (os.environ if env is None else env).get(key)  # type: ignore[union-attr]


def _bind_coerced_env(name: str, coercer: object):
    def _fn(key: str, env: object = None, default: object = None):
        if env is not None and not hasattr(env, "get"):
            default, env = env, None
        raw = _clean_env_str(_raw_env_val(key, env))
        return coercer(raw, default=default) if raw else default  # type: ignore[operator]

    _fn.__name__ = _fn.__qualname__ = name
    return _fn


float_env = _bind_coerced_env("float_env", safe_float)


def pos_float_env(key: str, default: float, env: dict | None = None) -> float:
    """Read a strictly positive float from os.environ or an injected dictionary, falling back to default."""
    return safe_pos_float(float_env(key, env=env), default)


int_env = _bind_coerced_env("int_env", exact_int)


def required_env(key: str, env: dict | None = None) -> str:
    """Return a mandatory setting, rejecting missing, empty, or comment-only values."""
    val = _raw_env_val(key, env)
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


def _raw_nonbool_numeric_env(key: str, env: dict | None = None) -> object:
    raw_val = _raw_env_val(key, env)
    if isinstance(raw_val, bool):
        raise ValueError(f"Invalid numeric setting: {key}={raw_val!r}")
    return raw_val


def required_float_env(key: str, env: dict | None = None) -> float:
    """Return a mandatory finite floating-point setting."""
    _raw_nonbool_numeric_env(key, env)
    raw = required_env(key, env)
    if raw.strip().lower() in _NONFINITE_FLOAT_TOKENS:
        raise ValueError(f"Non-finite numeric setting: {key}={raw!r}")
    value = safe_float(raw, default=None)
    if value is None:
        raise ValueError(f"Invalid numeric setting: {key}={raw!r}")
    return value


def required_int_env(key: str, env: dict | None = None) -> int:
    """Return a mandatory integer setting without silently truncating fractions."""
    raw_val = _raw_nonbool_numeric_env(key, env)
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
    raw_val = _raw_env_val(key, env)
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
    raw = _clean_env_str(_raw_env_val(key, env))
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


def clone_process_env(overrides: Mapping[str, object] | None = None) -> dict[str, str]:
    """Return a shallow ``dict`` copy of ``os.environ`` overlaid with ``overrides``."""
    out = dict(os.environ)
    if isinstance(overrides, Mapping):
        for k, v in overrides.items():
            if isinstance(k, str) and k:
                if v is None:
                    out.pop(k, None)
                else:
                    out[k] = str(v)
    return out


def set_process_env(key: str, value: object) -> str:
    """Set ``os.environ[key] = str(value)`` after validating ``key`` and return the assigned string."""
    if isinstance(key, bool) or not isinstance(key, str) or not key.strip():
        raise ValueError(f"Invalid environment variable name: {key!r}")
    val_s = str(value)
    os.environ[key] = val_s
    return val_s


def env_items_with_prefix(prefix: str, env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return all ``(key, value)`` pairs from ``env``/``os.environ`` whose key starts with ``prefix``."""
    if isinstance(prefix, bool) or not isinstance(prefix, str) or not prefix:
        return {}
    src = os.environ if env is None else env
    return {k: str(v) for k, v in src.items() if isinstance(k, str) and k.startswith(prefix) and v is not None}


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
                    result = arg[len(prefix) :]
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


def _bind_cli_parser(name: str, coercer, label: str, doc: str):
    def _parse(raw: object):
        val = coercer(raw, default=None)
        if val is None:
            raise argparse.ArgumentTypeError(f"invalid {label} value: {raw!r}")
        return val

    _parse.__name__ = name
    _parse.__qualname__ = name
    _parse.__doc__ = doc
    return _parse


parse_cli_float = _bind_cli_parser(
    "parse_cli_float",
    safe_float,
    "finite float",
    "Parse a finite CLI float argument, rejecting NaN/Inf/booleans.",
)
parse_pos_cli_float = _bind_cli_parser(
    "parse_pos_cli_float",
    safe_pos_float,
    "positive float",
    "Parse a strictly positive finite CLI float argument, rejecting <= 0/NaN/Inf/booleans.",
)
parse_nonneg_cli_float = _bind_cli_parser(
    "parse_nonneg_cli_float",
    safe_nonneg_float,
    "non-negative float",
    "Parse a non-negative finite CLI float argument, rejecting < 0/NaN/Inf/booleans.",
)
parse_cli_int = _bind_cli_parser(
    "parse_cli_int",
    exact_int,
    "integer",
    "Parse an exact CLI integer argument, rejecting floats/NaN/Inf/booleans.",
)
parse_pos_cli_int = _bind_cli_parser(
    "parse_pos_cli_int",
    exact_pos_int,
    "positive integer",
    "Parse a strictly positive exact CLI integer argument, rejecting <= 0/floats/NaN/Inf/booleans.",
)
parse_nonneg_cli_int = _bind_cli_parser(
    "parse_nonneg_cli_int",
    exact_nonneg_int,
    "non-negative integer",
    "Parse a non-negative exact CLI integer argument, rejecting < 0/floats/NaN/Inf/booleans.",
)


def create_env_cli_parser(description: str, env_file: str) -> argparse.ArgumentParser:
    """Create an ``ArgumentParser`` pre-populated with ``--env-file``."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--env-file", default=env_file)
    return ap


def add_store_true_cli_args(parser: object, *specs: object) -> object:
    """Register one or more ``action='store_true'`` CLI flags on ``parser`` and return it."""
    add_arg = getattr(parser, "add_argument", None)
    if not callable(add_arg):
        return parser
    for spec in specs:
        if isinstance(spec, str):
            if flag_s := clean_nonempty_str(spec):
                add_arg(flag_s, action="store_true")
        elif isinstance(spec, (tuple, list)) and len(spec) >= 1:
            raw_flags = spec[0]
            flag_help = spec[1] if len(spec) >= 2 else ""
            if isinstance(raw_flags, (tuple, list)):
                flags = tuple(fs for f in raw_flags if (fs := clean_nonempty_str(f)) is not None)
            elif (fs := clean_nonempty_str(raw_flags)) is not None:
                flags = (fs,)
            else:
                flags = ()
            if not flags:
                continue
            help_s = str(flag_help) if flag_help else ""
            if help_s:
                add_arg(*flags, action="store_true", help=help_s)
            else:
                add_arg(*flags, action="store_true")
    return parser


def add_paper_skip_wait_cli_args(
    parser: object,
    *,
    paper_help: str = "Force PAPER mode without real money",
    skip_wait_help: str = "Skip waiting for the launch",
    include_skip_wait: bool = True,
    include_desktop: bool = False,
    test_strategy_metavar: str | None = None,
    test_strategy_help: str = "Run the strategy NOW",
    find_flag: str | None = None,
    find_metavar: str = "TERM",
    find_help: str = "",
    extra_store_true: tuple[tuple[str, str], ...] = (),
) -> object:
    """Register standard ``--skip-wait`` and ``--paper`` flags on ``parser`` and return it."""
    add_arg = getattr(parser, "add_argument", None)
    if callable(add_arg):
        if _parse_bool_flag(include_desktop, default=False):
            add_store_true_cli_args(parser, "--desktop")
        if _parse_bool_flag(include_skip_wait, default=True):
            add_store_true_cli_args(parser, ("--skip-wait", skip_wait_help))
        add_store_true_cli_args(parser, ("--paper", paper_help), *extra_store_true)
        if meta_s := clean_nonempty_str(test_strategy_metavar):
            add_arg("--test-strategy", metavar=meta_s, help=test_strategy_help)
        if find_s := clean_nonempty_str(find_flag):
            add_arg(find_s, metavar=find_metavar, help=find_help)
    return parser


def validate_required_env_keys(
    keys: list[str] | tuple[str, ...], *, env: dict | None = None, label: str = "configuration"
) -> None:
    """Abort startup with the complete list of missing or empty mandatory environment keys."""
    src = os.environ if env is None else env
    missing = [name for name in keys if not _clean_env_str(src.get(name))]
    if missing:
        raise ValueError(f"Missing or empty mandatory {label} settings: " + ", ".join(missing))


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


def _parse_strict_trade_enabled(val: Any) -> Optional[bool]:
    """Strictly parse trade_enabled flag; return None on invalid/unrecognized values."""
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return bool(val) if val in (0, 1) else None
    if isinstance(val, float):
        if val == 1.0:
            return True
        if val == 0.0:
            return False
        return None
    if isinstance(val, str):
        s = val.split("#", 1)[0].split(";", 1)[0].strip().strip("\"'").strip().lower()
        if s in {"1", "1.0", "true", "yes", "on", "t", "da"}:
            return True
        if s in {"0", "0.0", "false", "no", "off", "f", "nu"}:
            return False
    return None


def load_config():
    """Load configuration from config.env (and legacy config.txt if present) and refresh cache."""
    global config_cache, _CONFIG_MTIME, _CONFIG_LAST_CHECK
    env_p = str(_bc_attr("CONFIG_ENV_PATH", CONFIG_ENV_PATH))
    txt_p = str(_bc_attr("CONFIG_TXT_PATH", CONFIG_TXT_PATH))
    new_config: dict = {}

    prev_cache = _bc_attr("config_cache", config_cache)
    prev_trade_enabled = prev_cache.get("trade_enabled") if isinstance(prev_cache, dict) else None

    raw_trade_val = None
    has_trade_key = False
    for path, allow_other in ((env_p, False), (txt_p, True)):
        for k, v in _read_properties_file(path).items():
            if not isinstance(k, str):
                continue
            kl = k.lower()
            if kl == "trade_enabled":
                has_trade_key = True
                raw_trade_val = v
            elif allow_other:
                new_config[kl] = v

    if has_trade_key:
        parsed_trade = _parse_strict_trade_enabled(raw_trade_val)
        if parsed_trade is not None:
            new_config["trade_enabled"] = parsed_trade
        else:
            # On invalid reload (e.g. 'flase'), preserve previous disable if stopped or fail closed, report error
            print(f"[config] ERROR: Invalid trade_enabled value {raw_trade_val!r} in config file. Failing safe.", file=sys.stderr)
            new_config["trade_enabled"] = False if prev_trade_enabled is False else False
    elif prev_trade_enabled is False:
        # Key removed starting from stopped (false): keep stopped! Only re-enable on explicit valid value
        new_config["trade_enabled"] = False
    else:
        new_config.setdefault("trade_enabled", True)

    env_trade = os.environ.get("TRADE_ENABLED")
    if env_trade is not None and env_trade.strip() != "":
        if _is_unmodified_root_key("TRADE_ENABLED"):
            if "trade_enabled" in new_config:
                _set_dotenv_env_key(
                    "TRADE_ENABLED",
                    "true" if new_config["trade_enabled"] else "false",
                    is_root=True,
                )
        else:
            parsed_env_trade = _parse_strict_trade_enabled(env_trade)
            if parsed_env_trade is not None:
                new_config["trade_enabled"] = parsed_env_trade
            else:
                print(f"[config] ERROR: Invalid TRADE_ENABLED in env: {env_trade!r}. Failing safe.", file=sys.stderr)
                new_config["trade_enabled"] = False

    new_config.setdefault("trade_enabled", True)

    _CONFIG_MTIME = (_file_mtime(env_p), _file_mtime(txt_p))
    _CONFIG_LAST_CHECK = _time.time()
    config_cache = new_config
    bc = sys.modules.get("botcore")
    if bc is not None:
        bc.__dict__["config_cache"] = config_cache


def is_trade_enabled() -> bool:
    """Return whether cached ``trade_enabled`` is true (with on-demand mtime hot-reload every 5s)."""
    global _CONFIG_LAST_CHECK
    env_p = str(_bc_attr("CONFIG_ENV_PATH", CONFIG_ENV_PATH))
    txt_p = str(_bc_attr("CONFIG_TXT_PATH", CONFIG_TXT_PATH))
    now_ts = _time.time()
    if not (0.0 <= (now_ts - _CONFIG_LAST_CHECK) < 5.0):
        _CONFIG_LAST_CHECK = now_ts
        cur_mtime = (_file_mtime(env_p), _file_mtime(txt_p))
        if cur_mtime != _CONFIG_MTIME:
            load_config()
    cur_cache = _bc_attr("config_cache", config_cache)
    eff_cache = cur_cache if isinstance(cur_cache, dict) else config_cache
    if "trade_enabled" not in eff_cache:
        return True
    parsed = _parse_strict_trade_enabled(eff_cache.get("trade_enabled"))
    return bool(parsed) if parsed is not None else False


load_config()


class Config:
    """Canonical SSOT configuration namespace and hot-reloaded ``trade_enabled`` manager."""

    load_dotenv = staticmethod(load_dotenv)
    load_env_stack = staticmethod(load_env_stack)
    load_root_config = staticmethod(load_root_config)
    load_shadow_environment = staticmethod(load_shadow_environment)
    parse_dotenv = staticmethod(parse_dotenv)
    load_env_file_defaults = staticmethod(load_env_file_defaults)
    load_key_value_conf = staticmethod(load_key_value_conf)
    load_layered_env_config = staticmethod(load_layered_env_config)
    load_isolated_venue_cfg = staticmethod(load_isolated_venue_cfg)
    bind_isolated_venue_cfg_loader = staticmethod(bind_isolated_venue_cfg_loader)
    empty_ini_config = staticmethod(empty_ini_config)
    load_ini_config = staticmethod(load_ini_config)
    conf_or_override = staticmethod(conf_or_override)
    parse_notify_enabled_flag = staticmethod(parse_notify_enabled_flag)
    pos_float_env = staticmethod(pos_float_env)
    required_env = staticmethod(required_env)
    defined_env = staticmethod(defined_env)
    required_float_env = staticmethod(required_float_env)
    required_int_env = staticmethod(required_int_env)
    required_bool_env = staticmethod(required_bool_env)
    optional_env = staticmethod(optional_env)
    env_value = staticmethod(env_value)
    is_env_defined = staticmethod(is_env_defined)
    secret_env = staticmethod(secret_env)
    folder_or_env_value = staticmethod(folder_or_env_value)
    env_values_map = staticmethod(env_values_map)
    bool_env = staticmethod(bool_env)
    any_bool_env = staticmethod(any_bool_env)
    validate_required_env_keys = staticmethod(validate_required_env_keys)

    @property
    def cache(self) -> dict:
        cur_cache = _bc_attr("config_cache", config_cache)
        return cur_cache if isinstance(cur_cache, dict) else config_cache

    @staticmethod
    def reload() -> dict:
        load_config()
        return config_cache

    @staticmethod
    def is_trade_enabled() -> bool:
        return is_trade_enabled()


cfg = Config()
