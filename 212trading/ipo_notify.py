#!/usr/bin/env python3
"""
ipo_notify.py — thin wrapper over the SHARED notify() in root alertnotifiers.py.
It only resolves the symbol, supporting an explicit multi-asset parameter, and delegates.
"""
from __future__ import annotations

import os
import sys

# alertnotifiers.py is in the repository root, the parent of 212trading/.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from alertnotifiers import bind_notify  # noqa: E402

# An explicit ``symbol=`` remains authoritative for multi-asset T212 threads.
notify = bind_notify(("SYMBOL_LABEL", "YAHOO_SYMBOL"), "STOCK")
