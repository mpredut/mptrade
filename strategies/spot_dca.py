#!/usr/bin/env python3
"""Backward-compatibility shim for the venue-neutral spot engine.

The canonical engine has been renamed to :mod:`strategies.spot_engine`.
This module re-exports all symbols to maintain seamless compatibility for
running processes, external tools, and legacy scripts.
"""
from __future__ import annotations

import sys
from . import spot_engine

from .spot_engine import *  # noqa: F401,F403
from .spot_engine import (
    StratParams,
    Strategy,
    state_path_for,
    _new_state,
    _parse_tranches,
    notify,
)

# Synchronize any mock/patch operations on this compatibility module with spot_engine
class _CompatibilityModule(sys.modules[__name__].__class__):
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if hasattr(spot_engine, name):
            setattr(spot_engine, name, value)

sys.modules[__name__].__class__ = _CompatibilityModule
