# providers/replay_clock.py
"""Mutable clock shared by replay and backtest drivers.

Calling the object returns ``ts``. The driver is responsible for assigning the
timestamp of every replayed event; this class does not advance automatically or enforce
monotonic time. Its initial wall-clock value preserves the historical implementation,
but a correct replay replaces that value before strategy code reads the clock.
"""


import time

from botcore import safe_float


def _clean_ts(val: float) -> float:
    f = safe_float(val, default=None)
    if f is None:
        raise ValueError("SimClock timestamp must be a finite number, not bool/None")
    return f


class SimClock:
    """Callable simulated time source whose ``ts`` (or ``value``) is set by its driver."""

    def __init__(self, initial_ts: float | None = None):
        self.ts: float = time.time() if initial_ts is None else _clean_ts(initial_ts)

    @property
    def value(self) -> float:
        return self.ts

    @value.setter
    def value(self, new_val: float) -> None:
        self.ts = _clean_ts(new_val)

    def __call__(self) -> float:
        return self.ts
