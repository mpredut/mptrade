#!/usr/bin/env python3
"""
kraken_common.py — Kraken-specific notification binding.
Shared logging, .env, config, HTTP, and numeric coercion primitives live in root botcore.py.
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from notify_engine.alertnotifiers import bind_notify  # noqa: E402

notify = bind_notify(("SYMBOL_LABEL", "KRAKEN_PAIR"), "CRYPTO")
