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

import fcntl
import json
import math
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone, timedelta

HTTP_TIMEOUT = 25
BUCHAREST = timezone(timedelta(hours=3))   # EEST in summer

_LOCKS: dict = {}   # retain open lock fds for process lifetime; prevent garbage collection


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).astimezone(BUCHAREST):%H:%M:%S}] {msg}", flush=True)


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


def _dotenv_pairs(path: str) -> tuple[list[tuple[str, str]], bool]:
    """Parse shared dotenv syntax once and report whether the file was read successfully."""
    if not os.path.exists(path):
        return [], False
    try:
        with open(path, "r", encoding="utf-8") as f:
            pairs = []
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[len("export "):]
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip()
                # Strip inline comments from unquoted values (VALUE=x  # comment).
                if not (val.startswith('"') or val.startswith("'")):
                    val = val.split("#")[0].strip()
                val = val.strip('"').strip("'")
                pairs.append((key, val))
        return pairs, True
    except OSError as e:
        log(f"  ! cannot read {path}: {e}")
        return [], False


_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_ROOT_CONFIG_PATH = os.path.abspath(os.path.join(_REPO_ROOT, "config.env"))
_ROOT_CONFIG_KEYS: set[str] = set()


def load_dotenv(path: str = ".env", override: bool = False) -> None:
    """Load KEY=VALUE entries from .env without overriding the existing environment (unless override=True)."""
    pairs, loaded = _dotenv_pairs(path)
    if not loaded:
        return
    is_root = os.path.abspath(path) == _ROOT_CONFIG_PATH
    for key, value in pairs:
        if not key:
            continue
        if is_root:
            if key not in os.environ:
                os.environ[key] = value
                _ROOT_CONFIG_KEYS.add(key)
        else:
            if override or key not in os.environ or key in _ROOT_CONFIG_KEYS:
                os.environ[key] = value
                _ROOT_CONFIG_KEYS.discard(key)
    log(f"  .env loaded from {path}")


def load_env_stack(env_file: str, config_name: str = "config.env", override: bool = False) -> None:
    """Load secrets/overrides and the adjacent versioned configuration.

    Local venue configurations take precedence over any globally loaded root config.env.
    Precedence (highest to lowest):
      1. Explicit process environment (unless override=True)
      2. Secrets/profile env file (env_file)
      3. Venue versioned configuration (config_name)
      4. Global root config.env defaults
    """
    env_path = os.path.abspath(env_file)
    cfg_path = os.path.join(os.path.dirname(env_path), config_name)

    # Process keys are those in os.environ that were NOT set by root config.env
    protected = set() if override else ((set(os.environ.keys()) - _ROOT_CONFIG_KEYS))

    # 1. Load adjacent versioned configuration (venue defaults)
    cfg_pairs, cfg_loaded = _dotenv_pairs(cfg_path)
    if cfg_loaded:
        for key, value in cfg_pairs:
            if key and key not in protected:
                os.environ[key] = value
                _ROOT_CONFIG_KEYS.discard(key)
        log(f"  .env loaded from {cfg_path}")

    # 2. Load secrets/profile configuration (precedence over venue config, but below process env)
    env_pairs, env_loaded = _dotenv_pairs(env_path)
    if env_loaded:
        for key, value in env_pairs:
            if key and key not in protected:
                os.environ[key] = value
                _ROOT_CONFIG_KEYS.discard(key)
        log(f"  .env loaded from {env_path}")


def parse_dotenv(path: str) -> dict:
    """Parse dotenv into a dictionary without changing os.environ.
    This keeps configuration separate when multiple assets run in the same process."""
    pairs, _loaded = _dotenv_pairs(path)
    out: dict[str, str] = {}
    for key, value in pairs:
        out[key] = value
    return out


def float_env(key: str, env: dict | None = None) -> float | None:
    """Read a float from os.environ or an injected dictionary, ignoring inline comments.
    The optional `env` argument preserves compatibility with legacy float_env(key) calls."""
    src = os.environ if env is None else env
    raw = (src.get(key, "") or "").split("#")[0].strip()
    try:
        return float(raw) if raw else None
    except ValueError:
        return None


def required_env(key: str, env: dict | None = None) -> str:
    """Return a mandatory setting, rejecting missing, empty, or comment-only values."""
    src = os.environ if env is None else env
    raw = str(src.get(key, "") or "").split("#", 1)[0].strip()
    if not raw:
        raise ValueError(f"Missing or empty mandatory setting: {key}")
    return raw


def defined_env(key: str, env: dict | None = None) -> str:
    """Return a setting that must exist but may intentionally be empty."""
    src = os.environ if env is None else env
    if key not in src:
        raise ValueError(f"Missing mandatory setting: {key}")
    return str(src[key] or "").split("#", 1)[0].strip()


def required_float_env(key: str, env: dict | None = None) -> float:
    """Return a mandatory finite floating-point setting."""
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
    value = required_float_env(key, env)
    if not value.is_integer():
        raise ValueError(f"Invalid integer setting: {key}={value!r}")
    return int(value)


def required_bool_env(key: str, env: dict | None = None) -> bool:
    """Return a mandatory strict boolean setting."""
    raw = required_env(key, env).lower()
    if raw not in {"true", "false"}:
        raise ValueError(f"Invalid boolean setting: {key}={raw!r}; expected true or false")
    return raw == "true"


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

    request_headers = dict(headers or {})
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")
    elif form is not None:
        data = urllib.parse.urlencode(form).encode()
        request_headers.setdefault(
            "Content-Type", "application/x-www-form-urlencoded"
        )

    verb = method.upper()
    req = urllib.request.Request(
        url, data=data, headers=request_headers, method=verb,
    )
    try:
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


# ── DETERMINISTIC percentage-based approximate comparisons ───────────────────
# Single source for the fleet and bots. Replaces utils.are_close, whose random.randint
# tolerance loop could return True OR False for the same input in [tol*1.01, tol*1.5],
# which is unacceptable for trading decisions.

def diff_percent(value1: float, value2: float) -> float:
    """Return symmetric percentage difference relative to the values' absolute mean."""
    if value1 == 0 and value2 == 0:
        return 0.0
    return abs(value1 - value2) / ((abs(value1) + abs(value2)) / 2) * 100


def are_close(value1: float, value2: float, tolerance_percent: float = 1.0) -> bool:
    """Return deterministically whether values differ by at most tolerance_percent.

    For price thresholds, are_close(price, threshold, 0.05) treats a price within
    0.05% as reached, avoiding missed entries by a few cents."""
    return diff_percent(value1, value2) <= tolerance_percent


def diff_equals_percent(value1: float, value2: float, target_percent: float,
                        tolerance_percent: float = 1.0) -> bool:
    """Return whether the percentage DIFFERENCE is near target_percent using a
    deterministic two-sided band. Unlike are_close, this asks whether values differ by
    about X percent, e.g. whether price fell about 10%. Replaces the randomized
    utils.are_difference_equal_with_aprox_proc implementation."""
    return abs(diff_percent(value1, value2) - target_percent) <= tolerance_percent
