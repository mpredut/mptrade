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

from offline.runners.walk_forward_compare import *  # noqa: F401,F403
from offline.runners.walk_forward_compare import (
    Candidate,
    default_candidates,
    hype_240_candidates,
    main,
    summarize_test_windows,
    walk_forward_splits,
    _candidate_windows,
    _load_datasets,
    _load_report,
    _markdown,
    _segment_result,
)

if __name__ == "__main__":
    raise SystemExit(main())
