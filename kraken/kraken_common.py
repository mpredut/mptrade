#!/usr/bin/env python3
"""
kraken_common.py — Kraken-specific notification binding.
Shared logging, .env, config, HTTP, and numeric coercion primitives live in root botcore.py.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from botcore import load_layered_env_config  # noqa: E402
from notify_engine.alertnotifiers import bind_notify  # noqa: E402

notify = bind_notify(("SYMBOL_LABEL", "KRAKEN_PAIR"), "CRYPTO")


def load_isolated_kraken_cfg(
    kraken_dir: str | os.PathLike[str] = _HERE,
    *,
    config_path: str | os.PathLike[str] | None = None,
    env_path: str | os.PathLike[str] | None = None,
) -> dict[str, str]:
    """Load Kraken config.env + .env into an isolated dict without mutating os.environ."""
    base_dir = os.fspath(kraken_dir)
    resolved_env = os.fspath(env_path) if env_path is not None else os.path.join(base_dir, ".env")
    resolved_cfg = (
        os.fspath(config_path)
        if config_path is not None
        else os.path.join(os.path.dirname(resolved_env) or base_dir, "config.env")
    )
    return load_layered_env_config(
        resolved_cfg,
        resolved_env,
        override_prefixes=("KRAKEN_",),
        fallback_prefixes=(),
    )
