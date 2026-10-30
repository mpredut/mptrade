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
import time as _time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, time as dt_time, timezone
from zoneinfo import ZoneInfo

HTTP_TIMEOUT = 25
BUCHAREST = ZoneInfo("Europe/Bucharest")

_LOCKS: dict = {}   # retain open lock fds for process lifetime; prevent garbage collection


def log(msg: str) -> None:
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


def _dotenv_pairs(path: str) -> tuple[list[tuple[str, str]], bool]:
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
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[len("export "):]
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip()
                # Strip inline comments and outer quotes safely.
                if val.startswith(('"', "'")):
                    q = val[0]
                    close_idx = val.find(q, 1)
                    if close_idx != -1:
                        tail = val[close_idx + 1:].strip()
                        if not tail or tail.startswith("#"):
                            val = val[1:close_idx]
                        else:
                            val = val.strip('"').strip("'")
                    else:
                        val = val.strip('"').strip("'")
                else:
                    val = val.split("#")[0].strip()
                pairs.append((key, val))
        return pairs, True
    except OSError as e:
        log(f"  ! cannot read {path}: {e}")
        return [], False


_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_ROOT_CONFIG_PATH = os.path.abspath(os.path.join(_REPO_ROOT, "config.env"))
_ROOT_CONFIG_KEYS: set[str] = set()
_ROOT_CONFIG_VALUES: dict[str, str] = {}


def _is_unmodified_root_key(key: str) -> bool:
    return (
        key in _ROOT_CONFIG_KEYS
        and key in os.environ
        and os.environ.get(key) == _ROOT_CONFIG_VALUES.get(key)
    )


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


def load_dotenv(path: str = ".env", override: bool = False) -> None:
    """Load KEY=VALUE entries from .env without overriding the existing environment (unless override=True)."""
    override = _parse_bool_flag(override, False)
    pairs, loaded = _dotenv_pairs(path)
    if not loaded:
        return
    is_root = os.path.abspath(path) == _ROOT_CONFIG_PATH
    for key, value in pairs:
        if not key:
            continue
        if is_root:
            if override or key not in os.environ:
                os.environ[key] = value
                _ROOT_CONFIG_KEYS.add(key)
                _ROOT_CONFIG_VALUES[key] = value
        else:
            if override or key not in os.environ or _is_unmodified_root_key(key):
                os.environ[key] = value
                _ROOT_CONFIG_KEYS.discard(key)
                _ROOT_CONFIG_VALUES.pop(key, None)
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
    override = _parse_bool_flag(override, False)
    env_path = os.path.abspath(env_file)
    cfg_path = os.path.join(os.path.dirname(env_path), config_name)

    # Process keys are those in os.environ that were NOT set (or were modified since being set) by root config.env
    unmodified_root_keys = {k for k in _ROOT_CONFIG_KEYS if _is_unmodified_root_key(k)}
    protected = set() if override else (set(os.environ.keys()) - unmodified_root_keys)

    # 1. Load adjacent versioned configuration (venue defaults)
    cfg_pairs, cfg_loaded = _dotenv_pairs(cfg_path)
    if cfg_loaded:
        for key, value in cfg_pairs:
            if key and key not in protected:
                os.environ[key] = value
                _ROOT_CONFIG_KEYS.discard(key)
                _ROOT_CONFIG_VALUES.pop(key, None)
        log(f"  .env loaded from {cfg_path}")

    # 2. Load secrets/profile configuration (precedence over venue config, but below process env)
    env_pairs, env_loaded = _dotenv_pairs(env_path)
    if env_loaded:
        for key, value in env_pairs:
            if key and key not in protected:
                os.environ[key] = value
                _ROOT_CONFIG_KEYS.discard(key)
                _ROOT_CONFIG_VALUES.pop(key, None)
        log(f"  .env loaded from {env_path}")


def parse_dotenv(path: str) -> dict[str, str]:
    """Parse dotenv into a dictionary without changing os.environ.
    This keeps configuration separate when multiple assets run in the same process."""
    pairs, _ = _dotenv_pairs(path)
    return dict(pairs)


def safe_float(val: object, default: float | None = 0.0) -> float | None:
    """Safely coerce ``val`` to finite float, rejecting booleans, None, non-finite values, and non-numeric objects."""
    if val is None or isinstance(val, bool):
        return default
    if not isinstance(val, (int, float, str, bytes, bytearray)) and type(val).__name__ != "Decimal":
        return default
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError, OverflowError):
        return default


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


def obj_attr(obj: object, key: str, default: object = None) -> object:
    """Extract ``key`` uniformly from either a ``dict`` or an object attribute."""
    if obj is None:
        return default
    val = obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)
    return default if val is None else val


def float_env(key: str, env: dict | None = None) -> float | None:
    """Read a float from os.environ or an injected dictionary, ignoring inline comments.
    The optional `env` argument preserves compatibility with legacy float_env(key) calls."""
    src = os.environ if env is None else env
    val = src.get(key)
    if val is None or isinstance(val, bool):
        return None
    raw = str(val).split("#", 1)[0].strip()
    return safe_float(raw, default=None) if raw else None


def int_env(key: str, env: dict | None = None) -> int | None:
    """Read an integer from os.environ or an injected dictionary, ignoring inline comments."""
    src = os.environ if env is None else env
    val = src.get(key)
    if val is None or isinstance(val, bool):
        return None
    raw = str(val).split("#", 1)[0].strip()
    return exact_int(raw, default=None) if raw else None


def required_env(key: str, env: dict | None = None) -> str:
    """Return a mandatory setting, rejecting missing, empty, or comment-only values."""
    src = os.environ if env is None else env
    val = src.get(key)
    if isinstance(val, bool):
        raise ValueError(f"Invalid mandatory setting: {key}={val!r}")
    raw = "" if val is None else str(val).split("#", 1)[0].strip()
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
    return "" if val is None else str(val).split("#", 1)[0].strip()


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


def fetch_public_json(
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
    req = urllib.request.Request(url, headers=req_headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        return json.loads(raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw)


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
    ts = safe_pos_float(now_ts, _time.time()) or _time.time()
    try:
        ny_dt = datetime.fromtimestamp(ts, tz=ET)
    except (ValueError, OverflowError, OSError):
        ny_dt = datetime.fromtimestamp(_time.time(), tz=ET)
    if ny_dt.weekday() >= 5:
        return False
    return dt_time(9, 30) <= ny_dt.time() < dt_time(16, 0)


def parse_env_file(env_path: str) -> dict[str, str]:
    """Parse a key=value env/conf file into an uppercase-keyed dictionary without mutating os.environ."""
    if not env_path or not os.path.exists(env_path):
        return {}
    try:
        raw = parse_dotenv(env_path)
        return {k.upper(): v for k, v in raw.items() if k}
    except Exception:
        return {}


def load_env_file_defaults(env_path: str) -> dict[str, str]:
    """Parse an env file and set non-STRAT_* keys in os.environ via setdefault."""
    parsed = parse_env_file(env_path)
    for key_u, val in parsed.items():
        if not key_u.startswith("STRAT_"):
            os.environ.setdefault(key_u, val)
    return parsed


def load_key_value_conf(workspace_dir: str, root_dir: str = _REPO_ROOT) -> dict[str, str]:
    """Load key=value settings from order_guard.conf / config.env / .env (root_dir overlaid by workspace_dir)."""
    conf: dict[str, str] = {}
    eff_root = root_dir or _REPO_ROOT
    search_dirs = [eff_root]
    if workspace_dir and os.path.abspath(workspace_dir) != os.path.abspath(eff_root):
        search_dirs.append(workspace_dir)
    for base_dir in search_dirs:
        for fname in ("order_guard.conf", "config.env", ".env"):
            for k_u, v_s in parse_env_file(os.path.join(base_dir, fname)).items():
                conf[k_u.lower()] = v_s
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

