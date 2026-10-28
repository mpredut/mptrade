#!/usr/bin/env python3
"""
common.py — utilities for the Hyperliquid bot.
The shared log/.env/float_env core comes from root botcore.py. Only HL-specific
global socket timeout and now_str remain here. Re-exports preserve compatibility
with `from common import log, load_dotenv, ...`.
"""
from __future__ import annotations

import os
import sys
import socket
from datetime import datetime, timezone

# RESILIENCE: the Hyperliquid SDK makes requests WITHOUT a read timeout. A network
# failure during an open request would hang forever without error or heartbeat.
# Globally default sockets without explicit timeouts to 30 seconds.
socket.setdefaulttimeout(30)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)  # repository root
from botcore import (  # noqa: E402,F401  (re-export compatibility)
    BUCHAREST, log, load_dotenv, load_env_stack, float_env, single_instance, are_close,
    required_env, defined_env, required_float_env, required_int_env, required_bool_env,
    _parse_bool_flag,
)


def now_str() -> str:
    n = datetime.now(timezone.utc)
    return f"{n.astimezone(BUCHAREST):%Y-%m-%d %H:%M:%S} Bucuresti"
