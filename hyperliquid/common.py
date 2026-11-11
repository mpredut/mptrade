#!/usr/bin/env python3
"""
common.py — Hyperliquid-specific runtime utilities (socket timeout, notify, local module loader).
Shared logging, .env, config, and numeric coercion primitives live in root botcore.py.
"""
from __future__ import annotations

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
from notify_engine.alertnotifiers import bind_notify  # noqa: E402

notify = bind_notify(("SYMBOL_LABEL", "HL_COIN"), "HL")
