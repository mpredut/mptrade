#!/usr/bin/env python3
"""Root executable wrapper for Macro Intelligence Analyzer Daemon."""
import os
import sys

# Ensure repository root is on sys.path
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from intelligence.macro_analyzer import main

if __name__ == "__main__":
    main()
