#!/usr/bin/env python3
"""Compatibility import for the venue-neutral spot engine.

New code should import :mod:`strategies.spot_engine`.  This module remains so the
existing Kraken launch commands and external scripts keep working unchanged.
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from strategies.spot_engine import (  # noqa: E402,F401
    StratParams,
    Strategy,
    _new_state,
    _parse_tranches,
    notify,
    state_path_for,
)

__all__ = ["StratParams", "Strategy", "state_path_for"]
