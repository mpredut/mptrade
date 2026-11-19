"""core/lock.py -- Canonical logging and cross-process flock primitives."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import os
import sys
import threading
import time as _time
from zoneinfo import ZoneInfo

from cachepaths import (
    _parse_bool_flag,
    resolve_cachedb_filepath,
    safe_float,
    safe_pos_float,
)

BUCHAREST = ZoneInfo("Europe/Bucharest")

_LOCKS: dict = globals().get("_LOCKS") or {}
_LOG_LOCK = globals().get("_LOG_LOCK") or threading.RLock()
_HAVE_FCNTL = True


def log(msg: str) -> None:
    with _LOG_LOCK:
        print(f"[{datetime.now(timezone.utc).astimezone(BUCHAREST):%H:%M:%S}] {msg}", flush=True)


def now_str() -> str:
    """Return current timestamp formatted in Europe/Bucharest timezone."""
    return f"{datetime.now(timezone.utc).astimezone(BUCHAREST):%Y-%m-%d %H:%M:%S} Bucharest"


def single_instance(name: str, lockdir: str = "/tmp") -> None:
    """Enforce a single instance with an exclusive flock."""
    path = os.path.join(lockdir, f"binance_{name}.lock")
    fd = open(path, "w")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"[{name}] already running (lock held: {path}) \u2014 exiting.", flush=True)
        sys.exit(0)
    fd.write(str(os.getpid()))
    fd.flush()
    _LOCKS[name] = fd


class FileLock:
    """Cross-process/thread lock using fcntl.flock(LOCK_EX) on an owned fd, with optional bounded wait."""

    def __init__(
        self,
        path,
        timeout: float | None = None,
        *,
        poll_interval: float = 0.1,
        raise_on_timeout: bool = True,
    ):
        self.path = resolve_cachedb_filepath(path)
        if timeout is None:
            self._timeout: float | None = None
        else:
            if isinstance(timeout, bool):
                raise TypeError("timeout must be numeric or None")
            t_val = safe_float(timeout, default=None)
            if t_val is None:
                raise ValueError(f"invalid FileLock timeout: {timeout!r}")
            self._timeout = max(0.0, t_val)
        self._poll_interval = safe_pos_float(poll_interval, default=0.1) or 0.1
        self._raise_on_timeout = _parse_bool_flag(raise_on_timeout, True)
        self._fd = None
        self.acquired = False

    def __enter__(self):
        parent = os.path.dirname(os.fspath(self.path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._fd = open(self.path, "a+")
        self.acquired = False
        if _HAVE_FCNTL:
            try:
                if self._timeout is None:
                    fcntl.flock(self._fd, fcntl.LOCK_EX)
                else:
                    deadline = _time.monotonic() + self._timeout
                    while True:
                        try:
                            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            break
                        except (BlockingIOError, OSError):
                            if _time.monotonic() >= deadline:
                                self._fd, fd = None, self._fd
                                fd.close()
                                if self._raise_on_timeout:
                                    raise TimeoutError(f"timed out acquiring FileLock on {self.path}")
                                return self
                            _time.sleep(self._poll_interval)
            except BaseException:
                if self._fd is not None:
                    self._fd.close()
                    self._fd = None
                raise
        self.acquired = True
        return self

    def __exit__(self, *exc):
        if self._fd is None:
            return
        try:
            if _HAVE_FCNTL and self.acquired:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            self.acquired, self._fd, fd = False, None, self._fd
            fd.close()
