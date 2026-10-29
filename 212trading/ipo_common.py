#!/usr/bin/env python3
"""
ipo_common.py — utilities for the T212 watcher.
The shared log/.env/parse_dotenv/float_env/HTTP core comes from root botcore.py.
Only the ET timezone and T212 timestamp format remain here. Re-exports preserve
compatibility with `from ipo_common import log, http_get, float_env, ...`.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)  # repository root
from botcore import (  # noqa: E402,F401  (re-export: compat `from ipo_common import ...`)
    BUCHAREST, ET, HTTP_TIMEOUT, log, load_dotenv, parse_dotenv, float_env, int_env,
    http_get, http_post_json, http_request, single_instance, are_close, diff_percent,
    defined_env, required_env, required_float_env, required_int_env, required_bool_env,
    resolve_cli_arg,
    _parse_bool_flag, safe_float, safe_pos_float, safe_nonneg_float, safe_int, safe_pos_int, safe_nonneg_int,
    exact_int, exact_pos_int, exact_nonneg_int,
    is_us_rth_open,
)


def load_t212_market_data():
    """Load ``212trading/market_data.py`` without colliding with other venue ``market_data`` modules."""
    import importlib.util

    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("t212_market_data", os.path.join(here, "market_data.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_t212_environment(env_file: str, *, profile_file: str | None = None) -> None:
    """Load T212 configuration using one explicit precedence policy.

    Existing process values win, followed by T212 secrets, shared repository
    secrets, an optional instrument profile, and versioned runtime policy.
    ``load_dotenv`` keeps the first value, so later layers only fill gaps.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    load_dotenv(os.path.abspath(env_file))
    load_dotenv(os.path.join(root, ".env"))
    if profile_file:
        load_dotenv(os.path.abspath(profile_file))
    load_dotenv(os.path.join(here, "runtime.env"))


def now_str() -> str:
    """Return a clear timestamp in ET and Bucharest time."""
    n = datetime.now(timezone.utc)
    return (
        f"{n.astimezone(ET):%Y-%m-%d %H:%M:%S} ET  |  "
        f"{n.astimezone(BUCHAREST):%H:%M:%S} Bucuresti"
    )
