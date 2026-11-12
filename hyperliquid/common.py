#!/usr/bin/env python3
"""
common.py -- Hyperliquid-specific runtime utilities (socket timeout, notify, local module loader).
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

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)  # repository root
from botcore import bind_isolated_venue_cfg_loader, load_layered_env_config  # noqa: E402
from notify_engine.alertnotifiers import bind_notify  # noqa: E402

notify = bind_notify(("SYMBOL_LABEL", "HL_COIN"), "HL")
load_isolated_hl_cfg = bind_isolated_venue_cfg_loader(_HERE, ("HL_",), dir_kwarg="hl_dir")


def load_hl_default_env(hl_dir: str | os.PathLike[str] = _HERE) -> dict[str, str]:
    """Load Hyperliquid config.env + .env layered with current os.environ overrides."""
    base_dir = os.fspath(hl_dir)
    base = load_layered_env_config(
        os.path.join(base_dir, "config.env"),
        os.path.join(base_dir, ".env"),
        fallback_prefixes=(),
    )
    base.update(os.environ)
    return base
