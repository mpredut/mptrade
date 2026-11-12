from __future__ import annotations
import builtins
import os
import sys
import re
import atexit
import datetime
import logging
import math
import shutil
import threading
from time import monotonic as _monotonic

from botcore import (
    _parse_bool_flag,
    load_dotenv,
    lru_dict_put,
    required_float_env,
    required_int_env,
)


# -- Versioned policy (overridable via configure()) ---------------------------

_ROOT                     = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_LOG_FOLDER       = os.path.join(_ROOT, "logger")
load_dotenv()
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

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_THIS_FILE = os.path.normpath(os.path.abspath(__file__))

_CALLER_PREFIXES = ("WARNING", "ERR", "CRITICAL", "DEBUG", "EXCEP", "TRACEBACK", "FAIL", "ALERT")
_CALLER_PREFIXES += tuple(prefix.lower() for prefix in _CALLER_PREFIXES)


_original_print        = builtins.print
_original_stderr_write = sys.stderr.write

_FILENAME_CACHE_MAX = 512
_filename_cache: dict[str, tuple[str, str]] = {}


def _get_caller_info() -> str:
    try:
        frame = sys._getframe(0)
        while frame is not None:
            raw = frame.f_code.co_filename
            entry = _filename_cache.get(raw)
            if entry is None:
                entry = (os.path.normpath(os.path.abspath(raw)), os.path.basename(raw))
                lru_dict_put(_filename_cache, raw, entry, _FILENAME_CACHE_MAX)
            norm, base = entry
            if norm != _THIS_FILE:
                return f"[{base}:{frame.f_lineno}]"
            frame = frame.f_back
    except (ValueError, AttributeError):
        pass
    return "[unknown]"


# -- Configuration state -------------------------------------------------------

_config_lock      = threading.Lock()
_log_folder       = os.path.normpath(os.path.abspath(_DEFAULT_LOG_FOLDER))
_max_size         = _DEFAULT_MAX_SIZE
_check_every      = _DEFAULT_CHECK_EVERY
_min_free_percent = _DEFAULT_MIN_FREE_PERCENT
_suppress_default_folder_writes: bool = False


def set_suppress_default_folder_writes(suppress: bool) -> None:
    """Configure whether default-folder file logging is suppressed when no explicit log folder is set."""
    global _suppress_default_folder_writes
    with _config_lock:
        _suppress_default_folder_writes = bool(suppress)


# -- Background cleanup thread -------------------------------------------------

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
        return (usage.free / usage.total * 100.0) if usage.total else None
    except OSError:
        return None


def _delete_oldest_log(folder: str) -> None:
    oldest_path  = None
    oldest_mtime = math.inf
    try:
        for entry in os.scandir(folder):
            if entry.is_file(follow_symlinks=False):
                mtime = entry.stat().st_mtime
                if mtime < oldest_mtime:
                    oldest_mtime, oldest_path = mtime, entry.path
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
        if not _cleanup_event.wait(timeout=5.0):
            continue
        _cleanup_event.clear()
        if _cleanup_stop.is_set():
            break
        with _config_lock:
            folder, max_size, min_free_percent = _log_folder, _max_size, _min_free_percent

        try:
            total = _get_folder_size(folder)
            while total >= max_size:
                mb, max_mb = total / (1024 * 1024), max_size / (1024 * 1024)
                _original_print(
                    f"[LOGGER] WARNING: log folder '{folder}' exceeds {max_mb:.0f} MB "
                    f"(current: {mb:.1f} MB) -> deleting oldest log"
                )
                before = total
                _delete_oldest_log(folder)
                total = _get_folder_size(folder)
                if total >= before:
                    _original_print("[LOGGER] WARNING: could not reduce log folder size, aborting cleanup")
                    break

            free_pct = _get_disk_free_percent(folder) if min_free_percent > 0.0 else None
            while free_pct is not None and free_pct < min_free_percent:
                _original_print(
                    f"[LOGGER] WARNING: disk free space {free_pct:.1f}% "
                    f"< threshold {min_free_percent:.1f}% -> deleting oldest log"
                )
                before_pct = free_pct
                _delete_oldest_log(folder)
                free_pct = _get_disk_free_percent(folder)
                if free_pct is None or free_pct <= before_pct:
                    _original_print("[LOGGER] WARNING: could not recover disk space, aborting cleanup")
                    free_pct = None
        except Exception as exc:
            _original_print(f"[LOGGER] cleanup error: {exc}")


_cleanup_thread = threading.Thread(target=_cleanup_loop, name="logger-cleanup", daemon=True)
_cleanup_thread.start()
atexit.register(lambda: (_cleanup_stop.set(), _cleanup_event.set(), _cleanup_thread.join(timeout=5.0)))


# -- Write counter & application name ------------------------------------------

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


# -- File handler setup --------------------------------------------------------

os.makedirs(_log_folder, exist_ok=True)
_entry_script = sys.argv[0] if sys.argv and sys.argv[0] else __file__
_app_name = os.path.splitext(os.path.basename(_entry_script))[0].lstrip("-") or "app"
_file_logger = logging.getLogger(f"app_file_logger.{_app_name}")
_file_logger.setLevel(logging.DEBUG)
_file_logger.propagate = False


class _DailyFileHandler(logging.Handler):
    """One open file handle per calendar day; rotates automatically at midnight
    and recreates the file if deleted externally.
    """

    def __init__(self, app_name: str) -> None:
        super().__init__()
        self._app_name = app_name
        self._current_date: str | None = None
        self._current_folder: str | None = None
        self._current_path = ""
        self._stream = None
        self._lock = threading.Lock()
        self._flush()

    def _flush(self) -> None:
        if self._stream is not None:
            self._stream.flush()
        self._pending_records = 0
        self._last_flush = _monotonic()

    def _close_stream(self) -> None:
        if self._stream is not None:
            try:
                self._flush()
                self._stream.close()
            except OSError:
                pass
            self._stream = None

    def _open_for_date(self, folder: str, date_str: str) -> None:
        self._close_stream()
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{self._app_name}_{date_str}.log")
        self._stream = open(path, "a", encoding="utf-8")
        self._current_path = path
        self._current_folder = folder
        self._current_date = date_str

    def emit(self, record: logging.LogRecord) -> None:
        try:
            with self._lock:
                today = datetime.datetime.now().strftime("%Y-%m-%d")
                with _config_lock:
                    folder = _log_folder
                    suppress_default = _suppress_default_folder_writes
                if folder == os.path.normpath(os.path.abspath(_DEFAULT_LOG_FOLDER)) and (
                    suppress_default or self._app_name.lower() in {"__main__", "c", "app"}
                ):
                    return
                reopen = today != self._current_date or folder != self._current_folder or self._stream is None
                if not reopen:
                    try:
                        fd_stat = os.fstat(self._stream.fileno())
                        disk_stat = os.stat(self._current_path)
                        reopen = fd_stat.st_ino != disk_stat.st_ino or fd_stat.st_dev != disk_stat.st_dev
                    except OSError:
                        reopen = True
                if reopen:
                    self._open_for_date(folder, today)
                try:
                    self._stream.write(self.format(record) + "\n")
                    self._pending_records += 1
                    if (
                        self._pending_records >= _FLUSH_EVERY_RECORDS
                        or _monotonic() - self._last_flush >= _FLUSH_INTERVAL_SEC
                    ):
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
            self._close_stream()
        super().close()


_handler = _DailyFileHandler(_app_name)
_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
_file_logger.addHandler(_handler)
atexit.register(_handler.close)


# -- Print / stderr overrides --------------------------------------------------

PRINT_CONTEXT = threading.local()
_print_lock = threading.RLock()


def _patched_print(*args, **kwargs) -> None:
    """Drop-in builtins.print replacement that mirrors output to the log file."""
    if not _parse_bool_flag(getattr(PRINT_CONTEXT, "enable_print", True), True):
        return
    if _bump_counter():
        _cleanup_event.set()

    sep_val = kwargs.get("sep", " ")
    sep_str = " " if sep_val is None else str(sep_val)
    message = sep_str.join(map(str, args))
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    clean = _ANSI_RE.sub("", message) if "\x1b" in message else message

    if "__orchestrator_intent__" in message:
        with _print_lock:
            _original_print(message, **kwargs)
        _file_logger.info(clean)
        return

    has_leading_ts = (
        len(clean) >= 10
        and clean[0] == "["
        and clean[9] == "]"
        and clean[3] == ":"
        and clean[6] == ":"
        and clean[1:3].isdigit()
        and clean[4:6].isdigit()
        and clean[7:9].isdigit()
    )
    clean_body = clean[10:].lstrip(" ") if has_leading_ts else clean

    if clean.startswith(_CALLER_PREFIXES):
        caller = _get_caller_info()
        with _print_lock:
            _original_print(
                f"{caller} {message}" if has_leading_ts else f"{current_time} {caller} {message}",
                **kwargs,
            )
        _file_logger.info(f"{caller} {clean_body}")
    else:
        with _print_lock:
            _original_print(message if has_leading_ts else f"{current_time} {message}", **kwargs)
        _file_logger.info(clean_body)


def _dummy_print(*args, **kwargs) -> None:
    pass


def disable_print() -> None:
    """Suppress all print output globally (including file logging)."""
    builtins.print = _dummy_print


def enable_print() -> None:
    """Restore print output globally."""
    builtins.print = _patched_print


def _patched_stderr_write(message: str) -> int:
    """Mirror stderr writes to the log file, stripping ANSI codes."""
    if not isinstance(message, str):
        message = str(message)
    written = _original_stderr_write(message)
    ret = written if isinstance(written, int) else len(message)
    if not message.strip():
        return ret
    if _bump_counter():
        _cleanup_event.set()
    clean = (_ANSI_RE.sub("", message) if "\x1b" in message else message).rstrip("\n")
    if not clean.strip():
        return ret
    _file_logger.info(f"{_get_caller_info()} [STDERR] {clean}")
    return ret


# -- One-time patch (double-import safe) ---------------------------------------

_patch_lock = threading.Lock()

with _patch_lock:
    if not getattr(builtins, "_logger_patched", False):
        builtins.print            = _patched_print              # type: ignore[attr-defined]
        sys.stderr.write          = _patched_stderr_write       # type: ignore[method-assign]
        sys.stderr.writelines     = lambda lines: [_patched_stderr_write(line) for line in lines]  # type: ignore[method-assign]
        builtins._logger_patched  = True                        # type: ignore[attr-defined]
