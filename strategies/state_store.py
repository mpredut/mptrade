"""Shared JSON persistence for financial-engine state.

Writes are atomic within the same directory. Live runtimes may fail closed so
corrupt or unsavable state is not mistaken for empty state that could permit a
duplicate entry.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from botcore import _parse_bool_flag
from state_io import StateReadError, load_json_state, save_json_state


def safe_state_token(val: object, default: str = "unknown", *, lower: bool = False) -> str:
    """Sanitize a symbol/ticker/direction token for use in a ``.state_<token>.json`` filename."""
    if isinstance(val, bool) or not isinstance(val, str):
        return default
    s = val.strip().lower() if lower else val
    return "".join(c for c in s if c.isalnum() or c in "._-").strip("._-") or default


def state_path_for(
    symbol: str,
    state_dir: str | os.PathLike | None = None,
    *,
    default_dir: str = "",
) -> str:
    """Return ``<dir>/.state_<safe_symbol>.json`` using ``default_dir`` when ``state_dir`` is blank."""
    safe = safe_state_token(symbol, "unknown")
    if isinstance(state_dir, os.PathLike):
        directory = os.fspath(state_dir) or default_dir
    elif isinstance(state_dir, str) and state_dir.strip():
        directory = state_dir.strip()
    else:
        directory = default_dir
    return os.path.join(directory, f".state_{safe}.json")


class StatePersistenceError(RuntimeError):
    """Financial state cannot be read or persisted safely."""


class JsonStateStore:
    def __init__(self, path: str, default_factory: Callable[[], dict], *,
                 label: str, logger: Callable[[str], None], fail_closed: bool):
        self.path = path
        self.default_factory = default_factory
        self.label = label
        self.logger = logger
        self.fail_closed = False if fail_closed is None else _parse_bool_flag(fail_closed, True)

    def load(self) -> dict:
        try:
            loaded = load_json_state(
                self.path, default_factory=dict, fail_closed=True,
                label=self.label,
            )
            if not isinstance(loaded, dict):
                raise ValueError("loaded state is not a dict")
            merged = self.default_factory()
            if not isinstance(merged, dict):
                raise TypeError("default_factory did not return a dict")
            merged.update(loaded)
            self.logger(
                f"  [STRAT] state loaded (cycle {merged.get('cycle')}, "
                f"qty {merged.get('qty')})"
            )
            return merged
        except (OSError, StateReadError, TypeError, ValueError) as exc:
            message = f"invalid {self.label} state in {self.path}: {exc}"
            if self.fail_closed:
                raise StatePersistenceError(message) from exc
            self.logger(f"  ! [STRAT] {message}; reset allowed only in PAPER")
            return self.default_factory()

    def save(self, state: dict) -> bool:
        try:
            if not isinstance(state, dict):
                raise TypeError("state must be a dict")
            save_json_state(self.path, state, indent=2)
            return True
        except (OSError, TypeError, ValueError) as exc:
            self.logger(f"  ! [STRAT] cannot save the state: {exc}")
            if self.fail_closed:
                raise StatePersistenceError(
                    f"persisting the {self.label} state failed: {exc}"
                ) from exc
            return False
