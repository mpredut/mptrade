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
_THREAD_FILELOCK_GUARD = globals().get("_THREAD_FILELOCK_GUARD") or threading.Lock()
_THREAD_FILELOCKS: dict[tuple[int, str], list] = globals().get("_THREAD_FILELOCKS") or {}
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
    """Cross-process and cross-thread lock using fcntl.flock(LOCK_EX), with optional per-thread re-entrancy.

    When ``reentrant=True``, nested acquisitions of the same canonical lock path within
    the same thread increment a per-(thread_id, realpath) depth counter instead of
    deadlocking on a second fd, while other threads and other OS processes remain
    strictly excluded via flock(LOCK_EX).
    """

    def __init__(
        self,
        path,
        timeout: float | None = None,
        *,
        poll_interval: float = 0.1,
        raise_on_timeout: bool = True,
        reentrant: bool = False,
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
        self._reentrant = _parse_bool_flag(reentrant, False)
        self._fd = None
        self._held_keys: list[tuple[int, str]] = []
        self.acquired = False

    def __enter__(self):
        parent = os.path.dirname(os.fspath(self.path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        lock_key: tuple[int, str] | None = None
        if self._reentrant:
            lock_key = (threading.get_ident(), os.path.realpath(os.fspath(self.path)))
            with _THREAD_FILELOCK_GUARD:
                existing = _THREAD_FILELOCKS.get(lock_key)
                if existing is not None:
                    existing[1] += 1
                    self._fd = existing[0]
                    self._held_keys.append(lock_key)
                    self.acquired = True
                    return self

        fd = open(self.path, "a+")
        self.acquired = False
        if _HAVE_FCNTL:
            try:
                if self._timeout is None:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                else:
                    deadline = _time.monotonic() + self._timeout
                    while True:
                        try:
                            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            break
                        except (BlockingIOError, OSError):
                            if _time.monotonic() >= deadline:
                                fd.close()
                                if self._raise_on_timeout:
                                    raise TimeoutError(f"timed out acquiring FileLock on {self.path}")
                                return self
                            _time.sleep(self._poll_interval)
            except BaseException:
                fd.close()
                raise
        if lock_key is not None:
            with _THREAD_FILELOCK_GUARD:
                _THREAD_FILELOCKS[lock_key] = [fd, 1]
            self._held_keys.append(lock_key)
        self._fd = fd
        self.acquired = True
        return self

    def __exit__(self, *exc):
        if self._reentrant:
            if not self._held_keys:
                return
            lock_key = self._held_keys.pop()
            release_fd = None
            with _THREAD_FILELOCK_GUARD:
                entry = _THREAD_FILELOCKS.get(lock_key)
                if entry is not None:
                    entry[1] -= 1
                    if entry[1] <= 0:
                        _THREAD_FILELOCKS.pop(lock_key, None)
                        release_fd = entry[0]
            if not self._held_keys:
                self.acquired = False
                self._fd = None
            if release_fd is None:
                return
            try:
                if _HAVE_FCNTL:
                    fcntl.flock(release_fd, fcntl.LOCK_UN)
            finally:
                release_fd.close()
            return
        if self._fd is None:
            return
        try:
            if _HAVE_FCNTL and self.acquired:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            self.acquired, self._fd, fd = False, None, self._fd
            fd.close()
