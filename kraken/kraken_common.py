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

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from botcore import (  # noqa: E402,F401  (re-export: compat `from common import ...`)
    BUCHAREST, HTTP_TIMEOUT, log, load_dotenv, load_env_stack, parse_dotenv, float_env, int_env, http_get,
    http_post_form, http_request, fetch_public_json, single_instance, are_close, diff_percent,
    defined_env, required_bool_env, required_env, required_float_env, required_int_env,
    resolve_cli_arg,
    _parse_bool_flag, safe_float, safe_pos_float, safe_nonneg_float, safe_pos_int, safe_int, safe_nonneg_int,
    exact_int, exact_pos_int, exact_nonneg_int,
    now_str,
)
