#!/usr/bin/env python3
"""
notify.py — thin wrapper over the SHARED notify() in root alertnotifiers.py.
Resolve the Kraken-specific symbol and delegate all common logic.
"""
from __future__ import annotations

import os
import sys

# alertnotifiers.py is in the repository root, the parent of kraken/.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from alertnotifiers import bind_notify  # noqa: E402

notify = bind_notify(("SYMBOL_LABEL", "KRAKEN_PAIR"), "CRYPTO")
