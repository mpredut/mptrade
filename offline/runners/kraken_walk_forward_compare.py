#!/usr/bin/env python3
"""Backward-compatibility shim for kraken_walk_forward_compare.

The canonical multi-venue compare runner is offline.runners.walk_forward_compare,
which supports Kraken, Binance, Hyperliquid, and Trading 212.
This shim re-exports all symbols to preserve full backward compatibility.
"""
from __future__ import annotations

from pathlib import Path
import sys

RUNNER_DIR = Path(__file__).resolve().parent
ROOT = RUNNER_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(RUNNER_DIR) not in sys.path:
    sys.path.append(str(RUNNER_DIR))

from offline.runners.walk_forward_compare import (
    DEFAULT_FEE_STRESS,
    _build_venue_evaluator,
    _candidate_windows,
    _detect_venue,
    _load_datasets,
    _load_report,
    _markdown,
    _resolve_candidate_overrides,
    _resolve_frozen_file,
    main,
)

__all__ = [
    "DEFAULT_FEE_STRESS",
    "_build_venue_evaluator",
    "_candidate_windows",
    "_detect_venue",
    "_load_datasets",
    "_load_report",
    "_markdown",
    "_resolve_candidate_overrides",
    "_resolve_frozen_file",
    "main",
]

if __name__ == "__main__":
    raise SystemExit(main())
