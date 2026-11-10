"""Generic cooldown gate: at most one operation per key within its TTL.

Atomic check-and-reserve uses ``FileLock``, disk state, and an RAII slot. The lock is
held only around reserve/release, not the protected operation, avoiding long I/O or
deadlocks under the lock. State persists; the lock does not.

    cd = Cooldown("trade", state_path=..., lock_path=...)
    with cd.slot("BTCUSDC", 180, side="BUY") as s:
        if not s.allowed:                 # Blocked by cooldown.
            return
        ...perform operation...
        if ok:
            s.commit(order_id=123)        # Success keeps the cooldown active.
        # Otherwise exit automatically releases the reservation.
"""
import contextlib
import multiprocessing
import os
import socket
import threading
import time

from botcore import (
    clean_json_value,
    load_ini_config,
    register_module_aliases,
    safe_float,
)
from cachepaths import resolve_cachedb_filepath
from state_io import read_json_dict, save_json_state

try:
    import fcntl                            # Unix (Linux/WSL)
    _HAVE_FCNTL = True
except ImportError:                         # Windows: gate disabled (no-op)
    _HAVE_FCNTL = False


class FileLock:
    """Cross-process/thread lock using fcntl.flock(LOCK_EX) on an owned fd."""

    def __init__(self, path):
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


def _is_active_entry(last, now: float, ttl: float) -> bool:
    if not isinstance(last, dict):
        return False
    ts = safe_float(last.get("timestamp"), None)
    ttl_f = safe_float(ttl, None)
    now_f = safe_float(now, None)
    return ts is not None and ttl_f is not None and now_f is not None and 0.0 <= (now_f - ts) < ttl_f


_INVALID_COOLDOWN_KEYS = frozenset({"", "none", "null", "nan", "inf", "-inf", "+inf"})


def _clean_cooldown_key(key) -> str:
    if key is None or isinstance(key, (bool, dict, list, tuple, set, frozenset)):
        return ""
    s = str(key).strip()
    return "" if s.lower() in _INVALID_COOLDOWN_KEYS else s


_RESERVED_ENTRY_KEYS = frozenset({
    "timestamp", "key", "group_id", "group_members", "group_pending", "group_committed", "group_results",
})


class Reservation:
    """RAII reservation guard returned by ``Cooldown.slot``.
    ``allowed=False`` means cooldown rejected the operation. Commit only after success;
    it retains the reservation and stores extra fields. Exiting without commit rolls back.
    """

    def __init__(self, cooldown, allowed, info, key):
        self._cd = cooldown
        self.allowed = allowed
        self.info = info
        self.key = key
        self._committed = False

    def commit(self, **fields):
        self._committed = True
        if fields:
            self._cd.update(self.key, **fields)


class Cooldown:
    """Generic disk-backed rapid-fire gate shared across processes and threads."""

    def __init__(self, name, state_path=None, lock_path=None, base_dir=None):
        self.name = name
        base = base_dir or os.getcwd()
        self.state_path = state_path or os.path.join(base, f"cooldown_{name}.json")
        self.lock_path = lock_path or os.path.join(base, f"cooldown_{name}.lock")

    # ── Storage ──────────────────────────────────────────────────────────────
    def _read(self):
        return read_json_dict(self.state_path, default={}, recover_previous=True)

    def _write(self, state):
        save_json_state(self.state_path, clean_json_value(state, stringify_keys=False), indent=2)

    @staticmethod
    def _build_reservation_entry(now, key, meta, **extra):
        entry = {
            "timestamp": now,
            "key": key,
            "pid": os.getpid(),
            "thread_id": threading.get_ident(),
            "process_name": multiprocessing.current_process().name,
            "hostname": socket.gethostname(),
            **extra,
        }
        for mk, mv in meta.items():
            if mk not in _RESERVED_ENTRY_KEYS:
                entry[mk] = mv
        return entry

    # ── API ──────────────────────────────────────────────────────────────────
    def reserve(self, key, ttl, **meta):
        """Atomically check and reserve the right to operate on ``key``.

        ``(True, entry)`` means newly reserved; ``(False, last)`` means blocked
        because the prior reservation is younger than the TTL.
        """
        key = _clean_cooldown_key(key)
        if not key:
            return False, {}
        with FileLock(self.lock_path):
            state = self._read()
            last = state.get(key)
            now = time.time()
            if _is_active_entry(last, now, ttl):
                return False, last
            entry = self._build_reservation_entry(now, key, meta)
            state[key] = entry
            self._write(state)
            return True, entry

    def release(self, key, *, expected_timestamp=None):
        """Release a failed reservation so it no longer blocks the TTL."""
        key = _clean_cooldown_key(key)
        if not key:
            return
        with FileLock(self.lock_path):
            state = self._read()
            if key in state:
                if expected_timestamp is not None:
                    cur_ts = state[key].get("timestamp") if isinstance(state[key], dict) else None
                    if cur_ts != expected_timestamp:
                        return
                del state[key]
                self._write(state)

    def update(self, key, **fields):
        """Add fields, such as a resulting ID, to an existing reservation."""
        key = _clean_cooldown_key(key)
        if not key:
            return
        safe_fields = {k: v for k, v in fields.items() if k not in _RESERVED_ENTRY_KEYS}
        if not safe_fields:
            return
        with FileLock(self.lock_path):
            state = self._read()
            if isinstance(state.get(key), dict):
                state[key].update(safe_fields)
                self._write(state)

    def get(self, key):
        key = _clean_cooldown_key(key)
        if not key:
            return None
        return self._read().get(key)

    @contextlib.contextmanager
    def slot(self, key, ttl, **meta):
        """Reserve on entry and roll back automatically when exiting without commit.

        The ``fcntl`` lock is held only during reserve/release, not across the body.
        """
        allowed, info = self.reserve(key, ttl, **meta)
        res = Reservation(self, allowed, info, key)
        try:
            yield res
        finally:
            if allowed and not res._committed:
                exp_ts = info.get("timestamp") if isinstance(info, dict) else None
                self.release(key, expected_timestamp=exp_ts)  # No completed operation, so roll back.


# ---------------------------------------------------------------------------
# Trade-specific rapid-fire order gate built on Cooldown
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # = lock/
STATE_FILE = os.path.join(BASE_DIR, "trade_cooldown.json")     # runtime state (gitignored)
LOCK_FILE = os.path.join(BASE_DIR, "trade_cooldown.lock")      # lock fcntl (gitignored)
CONF_FILE = os.path.join(BASE_DIR, "trade_cooldown.conf")      # tracked text/INI configuration


def _load_cooldown_sec() -> int:
    """Strictly read mandatory ``[cooldown] default_sec`` from ``trade_cooldown.conf``."""
    cp = load_ini_config(CONF_FILE, must_exist=True, inline_comments=True)
    if not cp.sections():
        raise ValueError(f"Failed to read mandatory configuration file: {CONF_FILE}")
    if not cp.has_option("cooldown", "default_sec"):
        raise KeyError(f"Missing required [cooldown] default_sec in {CONF_FILE}")
    val = cp.getint("cooldown", "default_sec")
    if val <= 0:
        raise ValueError(f"Invalid [cooldown] default_sec={val!r} in {CONF_FILE}; must be > 0")
    return val


DEFAULT_COOLDOWN_SEC = _load_cooldown_sec()


register_module_aliases(__name__, "lock.file_lock", "lock.trade_cooldown")

