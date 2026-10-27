"""Shared initialization helpers for read-only shadow runners."""
from __future__ import annotations

import os
import sys

from botcore import load_dotenv, load_env_stack


def prepare_shadow_runtime(root: str, *module_dirs: str) -> None:
    """Make repository modules importable and disable live Binance WebSockets."""
    for path in (root, *module_dirs):
        absolute = os.path.abspath(path)
        if absolute not in sys.path:
            sys.path.insert(0, absolute)
    os.environ.setdefault("BINANCE_AUTO_START_WEBSOCKETS", "0")


def load_shadow_environment(env_path: str, config_path: str | None = None) -> None:
    """Load a shadow runner with the same environment precedence as its live bot."""
    env_path = os.path.abspath(env_path)
    config_path = os.path.abspath(
        config_path or os.path.join(os.path.dirname(env_path), "config.env")
    )
    if os.path.dirname(config_path) == os.path.dirname(env_path):
        load_env_stack(env_path, os.path.basename(config_path))
        return
    load_dotenv(env_path)
    load_dotenv(config_path)


def require_shadow_interval(interval: int, expected: int, label: str) -> None:
    """Reject a cadence for which a shadow strategy was not validated."""
    if (
        isinstance(interval, bool)
        or isinstance(expected, bool)
        or not isinstance(interval, int)
        or not isinstance(expected, int)
        or expected <= 0
        or interval != expected
    ):
        raise ValueError(f"{label} accepts only the native {expected}m interval")
