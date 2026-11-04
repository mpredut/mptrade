"""Reusable local synchronization primitives, not limited to trading.

FileLock  — cross-process and cross-thread mutex using fcntl.flock on its own fd.
Cooldown  — persisted gate that prevents repeating an operation for <key> more
            often than every <ttl>s, with atomic reservation and an RAII slot.
"""
import sys as _sys

from . import cooldown
from .cooldown import Cooldown, FileLock, Reservation

file_lock = cooldown
trade_cooldown = cooldown
_sys.modules[f"{__name__}.file_lock"] = cooldown
_sys.modules[f"{__name__}.trade_cooldown"] = cooldown


def __getattr__(name: str):
    if name in ("file_lock", "trade_cooldown"):
        mod = _sys.modules.get(f"{__name__}.cooldown", cooldown)
        _sys.modules[f"{__name__}.{name}"] = mod
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["FileLock", "Cooldown", "Reservation", "file_lock", "trade_cooldown"]

