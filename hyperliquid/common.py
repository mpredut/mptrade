#!/usr/bin/env python3
"""
common.py — utilities for the Hyperliquid bot.
The shared log/.env/float_env core comes from root botcore.py. Only HL-specific
global socket timeout and now_str remain here. Re-exports preserve compatibility
with `from common import log, load_dotenv, ...`.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import socket

# RESILIENCE: the Hyperliquid SDK makes requests WITHOUT a read timeout. A network
# failure during an open request would hang forever without error or heartbeat.
# Globally default sockets without explicit timeouts to 30 seconds.
socket.setdefaulttimeout(30)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)  # repository root
from botcore import (  # noqa: E402,F401  (re-export compatibility)
    BUCHAREST, log, load_dotenv, load_env_stack, float_env, int_env, fetch_public_json, single_instance, are_close,
    required_env, defined_env, required_float_env, required_int_env, required_bool_env,
    resolve_cli_arg,
    _parse_bool_flag, safe_float, safe_pos_float, safe_nonneg_float, safe_int, safe_pos_int, safe_nonneg_int,
    exact_int, exact_pos_int, exact_nonneg_int,
    now_str,
)

_HERE = os.path.dirname(os.path.abspath(__file__))


def load_hl_local(name: str):
    """Load a sibling module from ``hyperliquid/`` without colliding with other venue modules."""
    mod = sys.modules.get(name)
    mod_file = getattr(mod, "__file__", None) if mod is not None else None
    if mod is not None and (not isinstance(mod_file, str) or os.path.dirname(os.path.abspath(mod_file)) == _HERE):
        return mod
    orig_sys_path = list(sys.path)
    try:
        if _ROOT not in sys.path:
            sys.path.insert(0, _ROOT)
        if _HERE not in sys.path:
            sys.path.append(_HERE)
        spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path[:] = orig_sys_path
