#!/usr/bin/env python3
"""
common.py — utilities for the Kraken bot.
The shared log/.env/float_env/HTTP core comes from root botcore.py. Only Kraken's
timestamp format remains here. Re-exports preserve compatibility with
`from common import log, http_get, ...`.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
from botcore import (  # noqa: E402,F401  (re-export: compat `from common import ...`)
    BUCHAREST, HTTP_TIMEOUT, log, load_dotenv, load_env_stack, parse_dotenv, float_env, http_get,
    http_post_form, http_request, single_instance, are_close, diff_percent,
    required_bool_env, required_env, required_float_env, required_int_env,
    _parse_bool_flag, safe_float,
)


def now_str() -> str:
    n = datetime.now(timezone.utc)
    return f"{n.astimezone(BUCHAREST):%Y-%m-%d %H:%M:%S} Bucuresti"
