from __future__ import annotations
import builtins
import os
import sys
import re
import atexit
import datetime
import logging
import shutil
import threading
from time import monotonic as _monotonic

from botcore import load_dotenv, required_float_env, required_int_env


# ── Versioned policy (overridable via configure()) ───────────────────────────

_ROOT                     = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_LOG_FOLDER       = os.path.join(_ROOT, "logger")
load_dotenv(os.path.join(_ROOT, "config.env"))
_DEFAULT_MAX_SIZE = required_int_env("LOG_MAX_FOLDER_BYTES")
_DEFAULT_CHECK_EVERY = required_int_env("LOG_CHECK_EVERY_WRITES")
_DEFAULT_MIN_FREE_PERCENT = required_float_env("LOG_MIN_FREE_PERCENT")
_FLUSH_EVERY_RECORDS = required_int_env("LOG_FLUSH_EVERY_RECORDS")
_FLUSH_INTERVAL_SEC = required_float_env("LOG_FLUSH_INTERVAL_SEC")

if _DEFAULT_MAX_SIZE <= 0 or _DEFAULT_CHECK_EVERY <= 0:
    raise ValueError("Log size and check interval must be positive")
if not 0 <= _DEFAULT_MIN_FREE_PERCENT <= 100:
    raise ValueError("LOG_MIN_FREE_PERCENT must be between 0 and 100")
if _FLUSH_EVERY_RECORDS <= 0 or _FLUSH_INTERVAL_SEC <= 0:
    raise ValueError("Log flush thresholds must be positive")


# ── ANSI ──────────────────────────────────────────────────────────────────────

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


# ── Module identity ───────────────────────────────────────────────────────────

_THIS_FILE = os.path.normpath(os.path.abspath(__file__))

_CALLER_PREFIXES = (
    "WARNING", "ERR", "CRITICAL", "DEBUG",
    "EXCEP", "TRACEBACK", "FAIL", "ALERT",
)
_CALLER_PREFIXES += tuple(prefix.lower() for prefix in _CALLER_PREFIXES)

def _needs_caller_info(message: str) -> bool:
    return message.startswith(_CALLER_PREFIXES)

_original_print        = builtins.print
_original_stderr_write = sys.stderr.write


# ── Filename cache ────────────────────────────────────────────────────────────

_FILENAME_CACHE_MAX = 512
_filename_cache: dict[str, tuple[str, str]] = {}

def _resolve_filename(raw: str) -> tuple[str, str]:
    entry = _filename_cache.get(raw)
    if entry is None:
        if len(_filename_cache) >= _FILENAME_CACHE_MAX:
            _filename_cache.pop(next(iter(_filename_cache)))
        norm  = os.path.normpath(os.path.abspath(raw))
        base  = os.path.basename(raw)
        entry = (norm, base)
        _filename_cache[raw] = entry
    return entry


def _get_caller_info() -> str:
    try:
        frame = sys._getframe(0)
        while frame is not None:
            norm, base = _resolve_filename(frame.f_code.co_filename)
            if norm != _THIS_FILE:
                return f"[{base}:{frame.f_lineno}]"
            frame = frame.f_back
    except (ValueError, AttributeError):
        pass
    return "[unknown]"


# ── Configuration state ───────────────────────────────────────────────────────

_config_lock      = threading.Lock()
_log_folder       = os.path.normpath(os.path.abspath(_DEFAULT_LOG_FOLDER))
_max_size         = _DEFAULT_MAX_SIZE
_check_every      = _DEFAULT_CHECK_EVERY
_min_free_percent = _DEFAULT_MIN_FREE_PERCENT


def configure(
    *,
    log_folder:       str   | None = None,
    max_size:         int   | None = None,
    check_every:      int   | None = None,
    min_free_percent: float | None = None,
) -> None:
    """Reconfigure the logger at runtime (all parameters are optional).

    Args:
        log_folder:        Directory for log files.  Resolved to an absolute
                           path immediately so later os.chdir() calls have no
                           effect.
        max_size:          Maximum total size in bytes of the log folder before
                           the oldest log is deleted.
        check_every:       Number of log writes between folder-size / disk
                           free-space checks.
        min_free_percent:  Minimum free disk space as a percentage of the
                           partition total (0-100).  When free space drops
                           below this threshold the oldest log file is deleted.
                           Set to 0.0 to disable the check.
    """
    global _log_folder, _max_size, _check_every, _min_free_percent

    with _config_lock:
        if log_folder is not None:
            resolved = os.path.normpath(os.path.abspath(log_folder))
            os.makedirs(resolved, exist_ok=True)
            _log_folder = resolved
        if max_size is not None:
            if max_size <= 0:
                raise ValueError("max_size must be > 0")
            _max_size = max_size
        if check_every is not None:
            if check_every <= 0:
                raise ValueError("check_every must be > 0")
            _check_every = check_every
        if min_free_percent is not None:
            if not (0.0 <= min_free_percent <= 100.0):
                raise ValueError("min_free_percent must be between 0 and 100")
            _min_free_percent = min_free_percent


# ── Background cleanup thread ─────────────────────────────────────────────────

_cleanup_event = threading.Event()
_cleanup_stop  = threading.Event()


def _get_folder_size(folder: str) -> int:
    total = 0
    try:
        for entry in os.scandir(folder):
            if entry.is_file(follow_symlinks=False):
                total += entry.stat().st_size
    except OSError:
        pass
    return total


def _get_disk_free_percent(folder: str) -> float | None:
    """Return free disk space as a percentage of partition total, or None on error."""
    try:
        usage = shutil.disk_usage(folder)
        if usage.total == 0:
            return None
        return usage.free / usage.total * 100.0
    except OSError:
        return None


def _delete_oldest_log(folder: str) -> None:
    oldest_path  = None
    oldest_mtime = float("inf")
    try:
        for entry in os.scandir(folder):
            if entry.is_file(follow_symlinks=False):
                mtime = entry.stat().st_mtime
                if mtime < oldest_mtime:
                    oldest_mtime = mtime
                    oldest_path  = entry.path
    except OSError:
        return

    if oldest_path is None:
        return

    try:
        os.remove(oldest_path)
        _original_print(f"[LOGGER] deleted oldest log: {os.path.basename(oldest_path)}")
    except OSError as exc:
        _original_print(f"[LOGGER] failed to delete log: {exc}")


def _cleanup_loop() -> None:
    while not _cleanup_stop.is_set():
        _cleanup_event.wait(timeout=5.0)
        _cleanup_event.clear()

        if _cleanup_stop.is_set():
            break

        with _config_lock:
            folder           = _log_folder
            max_size         = _max_size
            min_free_percent = _min_free_percent

        try:
            # ── Check 1: log folder total size ───────────────────────────────
            total = _get_folder_size(folder)
            while total >= max_size:
                mb     = total    / (1024 * 1024)
                max_mb = max_size / (1024 * 1024)
                _original_print(
                    f"[LOGGER] WARNING: log folder '{folder}' exceeds {max_mb:.0f} MB "
                    f"(current: {mb:.1f} MB) → deleting oldest log"
                )
                before = total
                _delete_oldest_log(folder)
                total  = _get_folder_size(folder)
                if total >= before:
                    _original_print(
                        "[LOGGER] WARNING: could not reduce log folder size, aborting cleanup"
                    )
                    break

            # ── Check 2: free disk space on partition ─────────────────────────
            if min_free_percent > 0.0:
                free_pct = _get_disk_free_percent(folder)
                while free_pct is not None and free_pct < min_free_percent:
                    _original_print(
                        f"[LOGGER] WARNING: disk free space {free_pct:.1f}% "
                        f"< threshold {min_free_percent:.1f}% → deleting oldest log"
                    )
                    before_pct = free_pct
                    _delete_oldest_log(folder)
                    free_pct = _get_disk_free_percent(folder)
                    if free_pct is None or free_pct <= before_pct:
                        # Nothing was freed, or we cannot measure it — stop here
                        _original_print(
                            "[LOGGER] WARNING: could not recover disk space, aborting cleanup"
                        )
                        break

        except Exception as exc:
            _original_print(f"[LOGGER] cleanup error: {exc}")


_cleanup_thread = threading.Thread(
    target=_cleanup_loop,
    name="logger-cleanup",
    daemon=True,
)
_cleanup_thread.start()


def _stop_cleanup_thread() -> None:
    _cleanup_stop.set()
    _cleanup_event.set()
    _cleanup_thread.join(timeout=5.0)


atexit.register(_stop_cleanup_thread)


# ── Write counter ─────────────────────────────────────────────────────────────

_write_counter = 0
_counter_lock  = threading.Lock()


def _bump_counter() -> bool:
    global _write_counter
    with _counter_lock:
        _write_counter += 1
        with _config_lock:
            threshold = _check_every
        if _write_counter >= threshold:
            _write_counter = 0
            return True
    return False


# ── Application name ──────────────────────────────────────────────────────────

def _resolve_app_name() -> str:
    entry = sys.argv[0] if sys.argv and sys.argv[0] else __file__
    name  = os.path.splitext(os.path.basename(entry))[0]
    return name or "app"


# ── File handler setup ────────────────────────────────────────────────────────

os.makedirs(_log_folder, exist_ok=True)

_app_name = _resolve_app_name()

_file_logger = logging.getLogger(f"app_file_logger.{_app_name}")
_file_logger.setLevel(logging.DEBUG)
_file_logger.propagate = False


class _DailyFileHandler(logging.Handler):
    """One open file handle per calendar day; rotates automatically at midnight.

    Thread-safe: the date check and file-handle swap are both inside the lock
    so midnight rotation is fully atomic.

    Also handles external deletion of the log folder or log file: before each
    write the handler checks whether the open fd still points to the same inode
    as the file on disk.  If the file (or folder) was deleted by another
    process the handler transparently recreates both and resumes logging.
    """

    def __init__(self, app_name: str) -> None:
        super().__init__()
        self._app_name      = app_name
        self._current_date: str | None = None
        self._current_path  = ""
        self._stream        = None
        self._lock          = threading.Lock()
        self._pending_records = 0
        self._last_flush = _monotonic()

    def _flush(self) -> None:
        if self._stream is not None:
            self._stream.flush()
        self._pending_records = 0
        self._last_flush = _monotonic()

    def _open_for_date(self, folder: str, date_str: str) -> None:
        if self._stream is not None:
            try:
                self._flush()
                self._stream.close()
            except OSError:
                pass
            self._stream = None

        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{self._app_name}_{date_str}.log")
        self._stream       = open(path, "a", encoding="utf-8")
        self._current_path = path
        self._current_date = date_str

    def _file_deleted(self) -> bool:
        """Return True if the log file or its folder was removed externally."""
        if self._stream is None:
            return True
        try:
            fd_stat   = os.fstat(self._stream.fileno())
            disk_stat = os.stat(self._current_path)
            return (fd_stat.st_ino != disk_stat.st_ino or
                    fd_stat.st_dev != disk_stat.st_dev)
        except OSError:
            return True

    def emit(self, record: logging.LogRecord) -> None:
        try:
            with self._lock:
                today = datetime.datetime.now().strftime("%Y-%m-%d")
                with _config_lock:
                    folder = _log_folder

                if today != self._current_date or self._stream is None or self._file_deleted():
                    self._open_for_date(folder, today)

                try:
                    self._stream.write(self.format(record) + "\n")
                    self._pending_records += 1
                    if (self._pending_records >= _FLUSH_EVERY_RECORDS
                            or _monotonic() - self._last_flush >= _FLUSH_INTERVAL_SEC):
                        self._flush()
                except (FileNotFoundError, OSError, ValueError):
                    self._open_for_date(folder, today)
                    self._stream.write(self.format(record) + "\n")
                    self._pending_records = 1
                    self._flush()

        except Exception:
            self.handleError(record)

    def close(self) -> None:
        with self._lock:
            if self._stream is not None:
                try:
                    self._flush()
                    self._stream.close()
                except OSError:
                    pass
                self._stream = None
        super().close()


_handler = _DailyFileHandler(_app_name)
_handler.setFormatter(
    logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
)
_file_logger.addHandler(_handler)

atexit.register(_handler.close)


# ── Print / stderr overrides ──────────────────────────────────────────────────

PRINT_CONTEXT = threading.local()


def _patched_print(*args, **kwargs) -> None:
    """Drop-in builtins.print replacement that mirrors output to the log file.

    Per-thread suppression : set PRINT_CONTEXT.enable_print = False
    Global suppression     : call disable_print()
    """
    if getattr(PRINT_CONTEXT, "enable_print", True) is False:
        return

    if _bump_counter():
        _cleanup_event.set()

    message      = " ".join(map(str, args))
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    clean        = (_ANSI_RE.sub("", message) if "\x1b" in message else message)

    if _needs_caller_info(clean):
        caller = _get_caller_info()
        _original_print(f"{current_time} {caller} {message}", **kwargs)
        _file_logger.info(f"{caller} {clean}")
    else:
        _original_print(f"{current_time} {message}", **kwargs)
        _file_logger.info(clean)


def _dummy_print(*args, **kwargs) -> None:
    pass


def disable_print() -> None:
    """Suppress all print output globally (including file logging)."""
    builtins.print = _dummy_print


def _patched_stderr_write(message: str) -> None:
    """Mirror stderr writes to the log file, stripping ANSI codes."""
    _original_stderr_write(message)

    if not message.strip():
        return

    if _bump_counter():
        _cleanup_event.set()

    clean = (_ANSI_RE.sub("", message) if "\x1b" in message else message).rstrip("\n")
    if not clean.strip():
        return

    caller = _get_caller_info()
    _file_logger.info(f"{caller} [STDERR] {clean}")


def _patched_stderr_writelines(lines) -> None:
    """Mirror stderr.writelines() to the log file."""
    for line in lines:
        _patched_stderr_write(line)


# ── One-time patch (double-import safe) ───────────────────────────────────────

_patch_lock = threading.Lock()

with _patch_lock:
    if not getattr(builtins, "_logger_patched", False):
        builtins.print            = _patched_print              # type: ignore[attr-defined]
        sys.stderr.write          = _patched_stderr_write       # type: ignore[method-assign]
        sys.stderr.writelines     = _patched_stderr_writelines  # type: ignore[method-assign]
        builtins._logger_patched  = True                        # type: ignore[attr-defined]
