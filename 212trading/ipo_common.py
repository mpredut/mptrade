#!/usr/bin/env python3
"""
ipo_common.py — T212-specific runtime utilities (notify, environment layering, dual ET/Bucharest timestamp).
Shared logging, .env, config, HTTP, and numeric coercion primitives live in root botcore.py.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)  # repository root
from botcore import BUCHAREST, ET, load_dotenv, load_local_module  # noqa: E402
from notify_engine.alertnotifiers import bind_notify  # noqa: E402

notify = bind_notify(("SYMBOL_LABEL", "YAHOO_SYMBOL"), "STOCK")


def load_t212_market_data():
    """Load ``212trading/market_data.py`` without colliding with other venue ``market_data`` modules."""
    here = os.path.dirname(os.path.abspath(__file__))
    return load_local_module(here, "market_data", module_key="t212_market_data", register=False)


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
        f"{n.astimezone(BUCHAREST):%H:%M:%S} Bucharest"
    )
