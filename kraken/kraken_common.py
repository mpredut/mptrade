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
from botcore import bind_isolated_venue_cfg_loader  # noqa: E402
from notify_engine.alertnotifiers import bind_notify  # noqa: E402

notify = bind_notify(("SYMBOL_LABEL", "KRAKEN_PAIR"), "CRYPTO")
load_isolated_kraken_cfg = bind_isolated_venue_cfg_loader(_HERE, ("KRAKEN_",))


def build_kraken_client(client_cls, creds_fn, role: str = "bot"):
    """Instantiate ``client_cls`` using ``creds_fn(role)``."""
    credentials = creds_fn(role)
    return client_cls(credentials.key, credentials.secret)
