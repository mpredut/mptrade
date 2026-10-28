"""FileLock — cross-process and cross-thread mutex based on fcntl.flock.

    from lock import FileLock
    with FileLock("/path/to/.lock"):        # exclusive
        ...critical section...               # the next process/thread waits here

Each invocation opens its own file descriptor, so flock(LOCK_EX) is exclusive
across distinct file descriptions and serializes both processes and threads.
On Windows, where fcntl is unavailable, this is a no-op; the bot runs on Linux.
The lock is generic and can protect any operation, not just order placement.
"""

import os

try:
    import fcntl                            # Unix (Linux/WSL)
    _HAVE_FCNTL = True
except ImportError:                         # Windows: gate disabled (no-op)
    _HAVE_FCNTL = False


class FileLock:
    """Cross-process/thread lock using fcntl.flock(LOCK_EX) on an owned fd."""

    def __init__(self, path):
        from cachepaths import resolve_cachedb_filepath
        self.path = resolve_cachedb_filepath(path)
        self._fd = None

    def __enter__(self):
        parent = os.path.dirname(os.fspath(self.path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._fd = open(self.path, "a+")
        if _HAVE_FCNTL:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX)
            except BaseException:
                self._fd.close()
                self._fd = None
                raise
        return self

    def __exit__(self, *exc):
        if self._fd is None:
            return
        try:
            if _HAVE_FCNTL:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            self._fd.close()
            self._fd = None
