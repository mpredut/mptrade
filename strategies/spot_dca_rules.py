"""Backward-compatibility shim for pure spot decision rules.

The canonical rules module has been renamed to :mod:`strategies.spot_rules`.
This module re-exports all symbols to maintain seamless compatibility for
running processes, external tools, and legacy scripts.
"""
from __future__ import annotations

import sys
from . import spot_rules

from .spot_rules import *  # noqa: F401,F403

# Synchronize any mock/patch operations on this compatibility module with spot_rules
class _CompatibilityRulesModule(sys.modules[__name__].__class__):
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if hasattr(spot_rules, name):
            setattr(spot_rules, name, value)

sys.modules[__name__].__class__ = _CompatibilityRulesModule
