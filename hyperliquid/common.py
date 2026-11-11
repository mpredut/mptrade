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

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)  # repository root
from botcore import load_layered_env_config  # noqa: E402
from notify_engine.alertnotifiers import bind_notify  # noqa: E402

notify = bind_notify(("SYMBOL_LABEL", "HL_COIN"), "HL")


def load_isolated_hl_cfg(
    hl_dir: str | os.PathLike[str] = _HERE,
    *,
    config_path: str | os.PathLike[str] | None = None,
    env_path: str | os.PathLike[str] | None = None,
) -> dict[str, str]:
    """Load Hyperliquid config.env + .env into an isolated dict without mutating os.environ."""
    base_dir = os.fspath(hl_dir)
    resolved_env = os.fspath(env_path) if env_path is not None else os.path.join(base_dir, ".env")
    resolved_cfg = (
        os.fspath(config_path)
        if config_path is not None
        else os.path.join(os.path.dirname(resolved_env) or base_dir, "config.env")
    )
    return load_layered_env_config(
        resolved_cfg,
        resolved_env,
        override_prefixes=("HL_",),
        fallback_prefixes=(),
    )


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
