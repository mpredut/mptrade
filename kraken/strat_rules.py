"""Compatibility import for the venue-neutral spot decision rules."""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from strategies.spot_rules import *  # noqa: E402,F401,F403
