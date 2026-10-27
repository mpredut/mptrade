"""Shared JSON persistence for financial-engine state.

Writes are atomic within the same directory. Live runtimes may fail closed so
corrupt or unsavable state is not mistaken for empty state that could permit a
duplicate entry.
"""

from __future__ import annotations

from collections.abc import Callable
from state_io import StateReadError, load_json_state, save_json_state


class StatePersistenceError(RuntimeError):
    """Financial state cannot be read or persisted safely."""


class JsonStateStore:
    def __init__(self, path: str, default_factory: Callable[[], dict], *,
                 label: str, logger: Callable[[str], None], fail_closed: bool):
        self.path = path
        self.default_factory = default_factory
        self.label = label
        self.logger = logger
        if isinstance(fail_closed, str):
            self.fail_closed = fail_closed.strip().lower() not in ("0", "0.0", "false", "no", "off", "")
        else:
            self.fail_closed = bool(fail_closed)

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
