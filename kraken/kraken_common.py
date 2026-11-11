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
from botcore import load_isolated_venue_cfg  # noqa: E402
from notify_engine.alertnotifiers import bind_notify  # noqa: E402

notify = bind_notify(("SYMBOL_LABEL", "KRAKEN_PAIR"), "CRYPTO")


def load_isolated_kraken_cfg(
    kraken_dir: str | os.PathLike[str] = _HERE,
    *,
    config_path: str | os.PathLike[str] | None = None,
    env_path: str | os.PathLike[str] | None = None,
) -> dict[str, str]:
    """Load Kraken config.env + .env into an isolated dict without mutating os.environ."""
    return load_isolated_venue_cfg(
        kraken_dir,
        config_path=config_path,
        env_path=env_path,
        override_prefixes=("KRAKEN_",),
        fallback_prefixes=(),
    )
