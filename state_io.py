"""Shared durable file primitives for runtime and financial state.

Writers use a unique temporary file in the destination directory, flush file data,
atomically replace the destination, and clean up on failure. Locking and failure
policy remain the caller's responsibility because caches, live financial state,
and observational state intentionally have different concurrency semantics.
"""
from __future__ import annotations

import contextlib
import csv
import fcntl
from functools import partial
import glob
import gzip
import json
import math
import os
import re
import shutil
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any, TypeVar

try:
    import orjson as _orjson
except ImportError:  # pragma: no cover
    _orjson = None

from cachepaths import (
    STALE_STAGING_MAX_AGE_SEC,
    _parse_bool_flag,
    clean_scalar_str,
    exact_nonneg_int,
    exact_pos_int,
    resolve_cachedb_filepath,
    safe_pos_float,
)

T = TypeVar("T")
_MISSING = globals().get("_MISSING") or object()
JSONDecodeError = json.JSONDecodeError


def _resolve_target_path(path: Any, *, abspath: bool = False, migrate: bool = False) -> str:
    """Validate ``path`` and resolve its canonical ``cachedb/<subdomain>/`` location."""
    if path is None or isinstance(path, (bool, int, float, bytes)):
        raise TypeError("path must be a valid filesystem path")
    raw = os.fspath(path)
    if clean_scalar_str(raw) is None:
        raise ValueError("path must not be empty")
    resolved = str(
        resolve_cachedb_filepath(raw, migrate=True) if migrate else resolve_cachedb_filepath(raw)
    )
    return os.path.abspath(resolved) if abspath else resolved


def _json_dump_kwargs(
    *, indent: int | None = None, sort_keys: bool = False, separators: Any = None,
    ensure_ascii: bool = True, allow_nan: bool = False, default: Any = None,
    include_format_opts: bool = True,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"ensure_ascii": _parse_bool_flag(ensure_ascii, True)}
    if include_format_opts:
        kwargs["indent"] = exact_nonneg_int(indent, default=None)
        kwargs["sort_keys"] = _parse_bool_flag(sort_keys, False)
        kwargs["separators"] = separators
        kwargs["allow_nan"] = _parse_bool_flag(allow_nan, False)
    if default is not None:
        kwargs["default"] = default
    return kwargs


_PREV_STATE_READ_ERROR = globals().get("StateReadError")
if isinstance(_PREV_STATE_READ_ERROR, type) and issubclass(_PREV_STATE_READ_ERROR, RuntimeError):
    StateReadError = _PREV_STATE_READ_ERROR
else:
    class StateReadError(RuntimeError):
        """Persisted runtime state cannot be trusted."""


def _fsync_parent_directory(path: str) -> None:
    """Best-effort directory fsync after publishing a new directory entry."""
    directory = os.path.dirname(os.path.abspath(os.fspath(path)))
    try:
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def cleanup_stale_staging_siblings(
    directory: str, prefix: str, *, exclude_name: str = "", suffix: str = "",
    dirs_only: bool = False, max_age_sec: float = STALE_STAGING_MAX_AGE_SEC,
) -> None:
    """Best-effort removal of stale staging directories or temporary sibling files in ``directory``."""
    try:
        if not directory or not os.path.isdir(directory):
            return
        now_s = time.time()
        for entry in os.listdir(directory):
            if entry == exclude_name or not entry.startswith(prefix) or (suffix and not entry.endswith(suffix)):
                continue
            cand_path = os.path.join(directory, entry)
            try:
                if (now_s - os.path.getmtime(cand_path)) < max_age_sec:
                    continue
                if dirs_only:
                    if os.path.isdir(cand_path):
                        shutil.rmtree(cand_path, ignore_errors=True)
                elif not os.path.isdir(cand_path):
                    os.unlink(cand_path)
            except OSError:
                pass
    except OSError:
        pass


def durable_replace_file(source: str, destination: str, *, fsync: bool = True) -> None:
    """Atomically replace ``destination`` and optionally durably fsync the new entry."""
    source = os.path.abspath(os.fspath(source))
    destination = os.path.abspath(os.fspath(destination))
    do_fsync = _parse_bool_flag(fsync, True)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if do_fsync:
        with open(source, "rb") as handle:
            os.fsync(handle.fileno())
    os.replace(source, destination)
    if source != destination and os.path.exists(source):
        os.remove(source)
    if do_fsync:
        _fsync_parent_directory(destination)


def atomic_snapshot_file(source: str, destination: str, *, fsync: bool = True) -> None:
    """Atomically preserve one immutable file generation at ``destination``."""
    source = os.path.abspath(os.fspath(source))
    destination = os.path.abspath(os.fspath(destination))
    do_fsync = _parse_bool_flag(fsync, True)
    directory = os.path.dirname(destination)
    os.makedirs(directory, exist_ok=True)
    prefix = f".{os.path.basename(destination)}."
    staging_directory = tempfile.mkdtemp(dir=directory, prefix=prefix)
    staged = os.path.join(staging_directory, "generation")
    try:
        try:
            os.link(source, staged)
        except OSError:
            shutil.copyfile(source, staged)
            if do_fsync:
                with open(staged, "rb") as handle:
                    os.fsync(handle.fileno())
        durable_replace_file(staged, destination, fsync=do_fsync)
    finally:
        shutil.rmtree(staging_directory, ignore_errors=True)
        cleanup_stale_staging_siblings(
            directory, prefix, exclude_name=os.path.basename(staging_directory), dirs_only=True
        )


def _reject_json_constant(constant: str) -> Any:
    raise ValueError(f"non-standard JSON constant {constant!r}")


_ORIG_JSON_LOAD = json.load
_ORIG_JSON_LOADS = json.loads
_ORIG_JSON_DUMPS = json.dumps
_ORIG_OPEN = open


def _fast_json_loads(raw: str | bytes | bytearray | memoryview, *, strict_constants: bool = False) -> Any:
    """Parse JSON via ``orjson.loads`` on the fast path, falling back to ``json.loads``."""
    if _orjson is not None and json.loads is _ORIG_JSON_LOADS and isinstance(raw, (str, bytes, bytearray, memoryview)):
        try:
            return _orjson.loads(raw)
        except Exception:
            pass
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray, memoryview)) else raw
    return json.loads(text, parse_constant=_reject_json_constant if strict_constants else None)


def _read_json_candidate(candidate_path: str, **json_kwargs: Any) -> Any:
    """Read UTF-8 JSON while preserving patched ``open`` and ``json.load`` hooks."""
    if _orjson is not None and json.load is _ORIG_JSON_LOAD and open is _ORIG_OPEN:
        with open(candidate_path, "rb") as handle:
            raw_bytes = handle.read()
        try:
            return _orjson.loads(raw_bytes)
        except Exception:
            return json.loads(raw_bytes.decode("utf-8"), **json_kwargs)
    with open(candidate_path, "r", encoding="utf-8") as handle:
        return json.load(handle, **json_kwargs)


def load_json_state(path: str, *, default_factory, fail_closed: bool,
                    label: str, root_type: type = dict,
                    repair_files: bool = True):
    """Load JSON state with an explicit live-versus-observational failure policy."""
    fail_closed = _parse_bool_flag(fail_closed, True)
    repair_b = _parse_bool_flag(repair_files, True)

    def _read_and_validate(candidate_path: str):
        value = _read_json_candidate(
            candidate_path, parse_constant=_reject_json_constant if fail_closed else None
        )
        if not isinstance(value, root_type):
            raise ValueError(f"expected {root_type.__name__}, got {type(value).__name__}")
        return value

    try:
        target_path = _resolve_target_path(path, migrate=repair_b)
        backup_path = previous_generation_path(target_path)
        if not os.path.exists(target_path) and not os.path.exists(backup_path):
            return default_factory()
        return read_recovering_previous(target_path, _read_and_validate, repair_files=repair_b)
    except (OSError, TypeError, ValueError) as exc:
        if fail_closed:
            raise StateReadError(f"invalid {label} state in {path}: {exc}") from exc
        return default_factory()


def _prepare_target_and_dir(path: str) -> tuple[str, str]:
    destination = _resolve_target_path(path, abspath=True, migrate=True)
    directory = os.path.dirname(destination)
    if directory:
        os.makedirs(directory, exist_ok=True)
    return destination, directory


def _flush_and_fsync(handle: Any) -> None:
    handle.flush()
    os.fsync(handle.fileno())


@contextlib.contextmanager
def _atomic_stream_writer(
    path: str, mode: str, *, preserve_mode: bool = False, fsync: bool = True, **open_kwargs: Any,
) -> Iterator[Any]:
    destination, directory = _prepare_target_and_dir(path)
    do_fsync = _parse_bool_flag(fsync, True)
    prefix = f".{os.path.basename(destination)}."
    fd, temporary = tempfile.mkstemp(dir=directory, prefix=prefix, suffix=".tmp")
    handle = os.fdopen(fd, mode, **open_kwargs)
    try:
        yield handle
        if do_fsync:
            _flush_and_fsync(handle)
        else:
            handle.flush()
        handle.close()
        if preserve_mode and os.path.exists(destination):
            shutil.copymode(destination, temporary)
        os.replace(temporary, destination)
        if do_fsync:
            _fsync_parent_directory(destination)
    except BaseException as exc:
        handle.close()
        discard_generation_file(temporary)
        raise exc
    finally:
        cleanup_stale_staging_siblings(
            directory, prefix, exclude_name=os.path.basename(temporary), suffix=".tmp"
        )


def atomic_text_writer(path: str, *, encoding: str = "utf-8",
                       newline: str | None = None,
                       preserve_mode: bool = False,
                       fsync: bool = True) -> contextlib.AbstractContextManager[Any]:
    """Yield a text handle and atomically publish it only after a successful write."""
    return _atomic_stream_writer(
        path, "w", preserve_mode=_parse_bool_flag(preserve_mode, False),
        fsync=_parse_bool_flag(fsync, True),
        encoding=encoding, newline=newline,
    )


def open_append_text_file(
    path: str | os.PathLike[str],
    *,
    encoding: str = "utf-8",
    newline: str | None = None,
) -> Any:
    """Ensure parent directory exists and open ``path`` in append text mode."""
    destination, _ = _prepare_target_and_dir(path)  # type: ignore[arg-type]
    return open(destination, "a", encoding=encoding, newline=newline)


def open_binary_read_file(path: str | os.PathLike[str]) -> Any:
    """Resolve ``path`` and open it in binary read mode (``rb``)."""
    return open(_resolve_target_path(path), "rb")


def atomic_write_text(path: str, content: str, *, encoding: str = "utf-8",
                      newline: str | None = None, preserve_mode: bool = False,
                      fsync: bool = True) -> None:
    """Write text ``content`` to ``path`` atomically through ``atomic_text_writer``."""
    with atomic_text_writer(
        path, encoding=encoding, newline=newline, preserve_mode=preserve_mode, fsync=fsync
    ) as handle:
        handle.write(str(content))


def atomic_write_json(path: str, value: Any, *, indent: int | None = None,
                      sort_keys: bool = False, separators=None, ensure_ascii: bool = True,
                      allow_nan: bool = False, default: Any = None,
                      fsync: bool = True) -> None:
    """Serialize JSON through ``atomic_text_writer``."""
    kwargs = _json_dump_kwargs(
        indent=indent, sort_keys=sort_keys, separators=separators,
        ensure_ascii=ensure_ascii, allow_nan=allow_nan, default=default,
    )
    with atomic_text_writer(path, fsync=fsync) as handle:
        json.dump(value, handle, **kwargs)


def atomic_write_jsonl(path: str, items: Any, *, ensure_ascii: bool = True,
                       default: Any = None) -> None:
    """Serialize an iterable of JSON-encodable objects to a JSONL file through ``atomic_text_writer``."""
    if items is not None and (isinstance(items, (bool, str, bytes, bytearray, dict)) or not hasattr(items, "__iter__")):
        raise TypeError("items must be an iterable of JSON records")
    kwargs = _json_dump_kwargs(ensure_ascii=ensure_ascii, default=default, include_format_opts=False)
    with atomic_text_writer(path) as handle:
        for item in (items or ()):
            handle.write(json.dumps(item, **kwargs) + "\n")


def _rollback_generation(
    target_path: str, backup_path: str | None, had_prior: bool, *,
    on_rollback_error: Callable[[OSError], None] | None = None, suppress_error: bool = False,
) -> None:
    if backup_path is not None and os.path.exists(backup_path):
        try:
            restore_previous_generation(target_path, backup_path=backup_path)
        except OSError as exc:
            if on_rollback_error is not None:
                on_rollback_error(exc)
            elif not suppress_error:
                raise
    elif not had_prior:
        discard_generation_file(target_path)


def save_json_state(path: str, value: Any, *, indent: int | None = None,
                    sort_keys: bool = False, separators=None, ensure_ascii: bool = True,
                    allow_nan: bool = False, default: Any = None,
                    write_fn: Callable[..., Any] | None = None,
                    fsync: bool = True) -> Any:
    """Transactionally write JSON state while staging and cleaning up ``<path>.previous``."""
    target_path = _resolve_target_path(path, migrate=True)
    eff_fsync = _parse_bool_flag(fsync, True)
    backup_path = None
    had_prior = os.path.exists(target_path)
    if had_prior:
        try:
            backup_path = stage_previous_generation(target_path, fsync=eff_fsync)
        except OSError:
            backup_path = None
    try:
        if write_fn is None:
            res = atomic_write_json(
                target_path, value, indent=indent, sort_keys=sort_keys,
                separators=separators, ensure_ascii=ensure_ascii,
                allow_nan=allow_nan, default=default, fsync=eff_fsync,
            )
        elif (
            not eff_fsync
            and getattr(write_fn, "__module__", None) == __name__
            and getattr(write_fn, "__name__", None) == "atomic_write_json"
        ):
            res = write_fn(
                target_path, value, indent=indent, sort_keys=sort_keys,
                separators=separators, ensure_ascii=ensure_ascii,
                allow_nan=allow_nan, default=default, fsync=False,
            )
        else:
            res = write_fn(target_path, value, **({"indent": indent} if indent is not None else {}))
        if backup_path is not None:
            discard_generation_file(backup_path)
        return res
    except BaseException:
        _rollback_generation(target_path, backup_path, had_prior, suppress_error=True)
        raise


def atomic_write_bytes(
    path: str, data: bytes | bytearray | memoryview, *, preserve_mode: bool = False,
) -> None:
    """Durably and atomically write binary bytes to ``path``."""
    with _atomic_stream_writer(path, "wb", preserve_mode=_parse_bool_flag(preserve_mode, False)) as handle:
        handle.write(data)


def append_text_line(
    path: str, line: str, *, encoding: str = "utf-8", flock: bool = False,
) -> None:
    """Append a single newline-terminated text line to ``path``, creating parent directories."""
    destination, _ = _prepare_target_and_dir(path)
    text = str(line)
    if not text.endswith("\n"):
        text += "\n"
    use_flock = _parse_bool_flag(flock, False) and fcntl is not None
    with open(destination, "a", encoding=encoding) as handle:
        if use_flock:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(text)
            if use_flock:
                handle.flush()
        finally:
            if use_flock:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def append_jsonl_record(
    path: str, record: Any, *, ensure_ascii: bool = True, sort_keys: bool = False,
    separators=None, allow_nan: bool = False, default: Any = None, flock: bool = False,
) -> None:
    """Serialize one JSON record and append it as a single line to ``path``."""
    kwargs = _json_dump_kwargs(
        sort_keys=sort_keys, separators=separators,
        ensure_ascii=ensure_ascii, allow_nan=allow_nan, default=default,
    )
    kwargs.pop("indent", None)
    append_text_line(path, json.dumps(record, **kwargs), flock=flock)


@contextlib.contextmanager
def atomic_file_path(
    path: str, *, prefix: str | None = None, suffix: str = ".tmp",
) -> Iterator[str]:
    """Yield a temporary sibling path and durably publish it onto ``path`` on exit."""
    destination, directory = _prepare_target_and_dir(path)
    eff_prefix = prefix if prefix is not None else f".{os.path.basename(destination)}."
    fd, temporary = tempfile.mkstemp(dir=directory, prefix=eff_prefix, suffix=suffix)
    os.close(fd)
    try:
        yield temporary
        durable_replace_file(temporary, destination)
    except BaseException:
        discard_generation_file(temporary)
        raise
    finally:
        cleanup_stale_staging_siblings(
            directory, eff_prefix, exclude_name=os.path.basename(temporary), suffix=suffix
        )


def atomic_gzip_file(
    source: str, destination: str | None = None, *,
    remove_source: bool = False, chunk_size: int = 1024 * 1024,
) -> str:
    """Compress ``source`` into ``destination`` (default ``<source>.gz``) atomically."""
    remove_source = _parse_bool_flag(remove_source, False)
    eff_chunk = exact_pos_int(chunk_size, default=1024 * 1024)
    source_path = os.path.abspath(os.fspath(source))
    target_path = os.path.abspath(
        os.fspath(destination) if destination is not None else f"{source_path}.gz"
    )
    with atomic_file_path(target_path, suffix=".tmp") as temp_gz:
        with open(source_path, "rb") as src_handle, gzip.open(temp_gz, "wb") as gz_handle:
            shutil.copyfileobj(src_handle, gz_handle, length=eff_chunk)
    if remove_source:
        discard_generation_file(source_path)
    return target_path


def _extract_tail_non_empty(chunks: list[bytes], *, drop_partial_head: bool) -> list[bytes]:
    joined = b"".join(reversed(chunks))
    if drop_partial_head:
        joined = joined.split(b"\n", 1)[1] if b"\n" in joined else b""
    return [line for line in joined.splitlines() if line.strip()]


_JSONL_TAIL_ANCHOR_BYTES = 64


def read_tail_lines_with_cursor(
    path: str, limit: int | None = None, *, chunk_size: int = 65536, max_bytes: int | None = None,
) -> tuple[list[bytes], tuple[tuple[int, int], int, bytes] | None]:
    """Read up to the last ``limit`` lines and/or ``max_bytes`` from ``path`` plus an append-tail cursor."""
    try:
        resolved = _resolve_target_path(path)
    except (TypeError, ValueError):
        return [], None
    if not os.path.isfile(resolved):
        return [], None
    limit_i = exact_pos_int(limit, default=None) if limit is not None else None
    max_b = exact_pos_int(max_bytes, default=None) if max_bytes is not None else None
    if (limit is not None and limit_i is None) or (max_bytes is not None and max_b is None) or (limit_i is None and max_b is None):
        return [], None
    eff_chunk = max_b if limit_i is None else exact_pos_int(chunk_size, default=65536)
    chunks: list[bytes] = []
    newline_count = 0
    cursor: tuple[tuple[int, int], int, bytes] | None = None
    try:
        with open(resolved, "rb") as handle:
            st = os.fstat(handle.fileno())
            dev_ino = (st.st_dev, st.st_ino)
            end_pos = st.st_size
            position = end_pos
            min_pos = max(0, position - max_b) if max_b is not None else 0

            def _build_cursor() -> tuple[tuple[int, int], int, bytes] | None:
                if end_pos <= 0 or not chunks:
                    return None
                tail_chunk = chunks[0]
                if tail_chunk.endswith(b"\n"):
                    c_pos = end_pos
                else:
                    last_nl = tail_chunk.rfind(b"\n")
                    if last_nl < 0:
                        return None
                    c_pos = (end_pos - len(tail_chunk)) + last_nl + 1
                if c_pos <= 0:
                    return None
                a_len = min(_JSONL_TAIL_ANCHOR_BYTES, c_pos)
                handle.seek(c_pos - a_len)
                return (dev_ino, c_pos, handle.read(a_len))

            while position > min_pos:
                step = min(eff_chunk, position - min_pos)
                position -= step
                handle.seek(position)
                chunk = handle.read(step)
                chunks.append(chunk)
                newline_count += chunk.count(b"\n")
                if limit_i is not None and newline_count > limit_i and position > min_pos:
                    non_empty = _extract_tail_non_empty(chunks, drop_partial_head=True)
                    if len(non_empty) >= limit_i:
                        return non_empty[-limit_i:], _build_cursor()
                    newline_count = len(non_empty)
            cursor = _build_cursor()
    except OSError:
        return [], None
    lines = _extract_tail_non_empty(chunks, drop_partial_head=(position > 0 and b"\n" in b"".join(reversed(chunks))))
    return (lines[-limit_i:] if limit_i is not None else lines), cursor


def read_tail_lines(
    path: str, limit: int | None = None, *, chunk_size: int = 65536, max_bytes: int | None = None,
) -> list[bytes]:
    """Read up to the last ``limit`` lines and/or ``max_bytes`` from ``path`` with bounded memory."""
    lines, _ = read_tail_lines_with_cursor(path, limit, chunk_size=chunk_size, max_bytes=max_bytes)
    return lines


def _extract_jsonl_record_ts_ms(rec: Any) -> float | None:
    item = rec.get("i") if isinstance(rec, dict) and "i" in rec else rec
    if isinstance(item, (list, tuple)) and item:
        raw_t = item[0]
    elif isinstance(rec, dict):
        raw_t = rec.get("ts_ms") or rec.get("ts") or rec.get("timestamp")
    else:
        return None
    val = safe_pos_float(raw_t, default=None)
    if val is None:
        return None
    return val * 1000.0 if val < 1e11 else val


def read_jsonl_records_since_span(
    path: str,
    span_ms: float,
    *,
    max_records: int = 4000,
    chunk_size: int = 65536,
) -> list[Any]:
    """Binary-seek a chronologically ordered JSONL file to ``last_ts - span_ms`` and return strided records."""
    try:
        resolved = _resolve_target_path(path)
    except (TypeError, ValueError):
        return []
    if not os.path.isfile(resolved):
        return []
    span_f = safe_pos_float(span_ms, default=None)
    if span_f is None:
        return []
    max_rec_i = exact_pos_int(max_records, default=4000) or 4000
    eff_chunk = exact_pos_int(chunk_size, default=65536) or 65536
    try:
        file_size = os.path.getsize(resolved)
        if file_size <= 0:
            return []
        with open(resolved, "rb") as handle:
            handle.seek(max(0, file_size - min(file_size, 8192)))
            tail_lines = [ln for ln in handle.read().splitlines() if ln.strip()]
            last_ts: float | None = None
            for raw_ln in reversed(tail_lines):
                try:
                    last_ts = _extract_jsonl_record_ts_ms(_fast_json_loads(raw_ln))
                except ValueError:
                    continue
                if last_ts is not None:
                    break
            if last_ts is None:
                return []
            cutoff_ms = last_ts - span_f
            lo, hi = 0, file_size
            while hi - lo > eff_chunk:
                mid = (lo + hi) // 2
                handle.seek(mid)
                handle.readline()
                probe_ln = handle.readline()
                if not probe_ln:
                    hi = mid
                    continue
                try:
                    probe_ts = _extract_jsonl_record_ts_ms(_fast_json_loads(probe_ln))
                except ValueError:
                    hi = mid
                    continue
                if probe_ts is None or probe_ts < cutoff_ms:
                    lo = mid
                else:
                    hi = mid
            handle.seek(lo)
            if lo > 0:
                handle.readline()
            raw_lines = [ln for ln in handle.read().splitlines() if ln.strip()]
    except OSError:
        return []
    if not raw_lines:
        return []
    start_idx = len(raw_lines)
    for idx, raw_bytes in enumerate(raw_lines):
        try:
            ts_val = _extract_jsonl_record_ts_ms(_fast_json_loads(raw_bytes))
        except ValueError:
            continue
        if ts_val is not None and ts_val >= cutoff_ms:
            start_idx = idx
            break
    end_idx = len(raw_lines)
    while end_idx > start_idx:
        try:
            tail_ts = _extract_jsonl_record_ts_ms(_fast_json_loads(raw_lines[end_idx - 1]))
        except ValueError:
            end_idx -= 1
            continue
        if tail_ts is not None and tail_ts >= cutoff_ms:
            break
        end_idx -= 1
    window_lines = raw_lines[start_idx:end_idx]
    if not window_lines:
        return []
    n_win = len(window_lines)
    if n_win <= max_rec_i:
        selected = window_lines
    elif max_rec_i == 1:
        selected = [window_lines[-1]]
    else:
        selected = [
            window_lines[(i * (n_win - 1)) // (max_rec_i - 1)]
            for i in range(max_rec_i)
        ]
    out: list[Any] = []
    for raw_bytes in selected:
        try:
            rec = _fast_json_loads(raw_bytes)
        except ValueError:
            continue
        rec_ts = _extract_jsonl_record_ts_ms(rec)
        if rec_ts is not None and rec_ts < cutoff_ms:
            continue
        out.append(rec)
    return out


def read_tail_jsonl_records(
    path: str, limit: int | None = 50, *, chunk_size: int = 65536, max_bytes: int | None = None,
) -> list[Any]:
    """Read and parse JSON lines from the tail of a JSONL file."""
    out: list[Any] = []
    loads_orjson = _orjson.loads if (_orjson is not None and json.loads is _ORIG_JSON_LOADS) else None
    for raw_bytes in read_tail_lines(path, limit, chunk_size=chunk_size, max_bytes=max_bytes):
        try:
            if loads_orjson is not None:
                try:
                    out.append(loads_orjson(raw_bytes))
                    continue
                except Exception:
                    pass
            out.append(_fast_json_loads(raw_bytes))
        except ValueError:
            continue
    return out


def read_tail_jsonl_dicts(
    path: str, limit: int | None = 50, *, chunk_size: int = 65536, max_bytes: int | None = None,
) -> list[dict[str, Any]]:
    """Read and parse up to the last ``limit`` JSON object lines from a JSONL file."""
    return [
        rec for rec in read_tail_jsonl_records(path, limit=limit, chunk_size=chunk_size, max_bytes=max_bytes)
        if isinstance(rec, dict)
    ]


def read_last_jsonl_dict(path: str, tail_bytes: int = 4096) -> dict[str, Any] | None:
    """Read and parse the last JSON dict from a JSONL file using bounded tail reads."""
    items = read_tail_jsonl_dicts(path, limit=1, chunk_size=exact_pos_int(tail_bytes, default=4096))
    return items[-1] if items else None


def truncate_file_bytes(path: str, size: int) -> None:
    """Durably truncate ``path`` back to ``size`` bytes."""
    size_i = exact_nonneg_int(size, default=None)
    if size_i is None:
        raise ValueError("size must be a non-negative integer")
    with open(_resolve_target_path(path, migrate=True), "r+b") as handle:
        handle.truncate(size_i)
        _flush_and_fsync(handle)


def format_jsonl_bucket_line(symbol: str, item: Any) -> str:
    """Serialize one ``(symbol, item)`` pair as a compact JSONL line."""
    if _orjson is not None and json.dumps is _ORIG_JSON_DUMPS:
        try:
            return _orjson.dumps({"s": symbol, "i": item}, option=_orjson.OPT_APPEND_NEWLINE).decode("ascii")
        except Exception:
            pass
    return json.dumps({"s": symbol, "i": item}, separators=(",", ":")) + "\n"


def canonical_json_key(item: Any) -> str:
    """Return a deterministic sort-keyed JSON string for deduplicating cache items."""
    return json.dumps(item, sort_keys=True)


def parse_json_payload(raw: str | bytes, *, default: Any = _MISSING) -> Any:
    """Parse a JSON string or bytes payload, returning ``default`` (if provided) on decode failure."""
    try:
        return _fast_json_loads(raw)
    except Exception as exc:
        if default is _MISSING:
            raise exc
        return default


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```")


def extract_json_object(text: Any) -> dict[str, Any] | None:
    """Extract a JSON dict from pure JSON, markdown code fences, or hybrid prose + trailing JSON."""
    if not isinstance(text, str):
        return None
    cleaned = text.strip()
    if not cleaned:
        return None
    if "```" in cleaned:
        fences = _JSON_FENCE_RE.findall(cleaned)
        for block in reversed(fences):
            cand = block.strip()
            if not cand:
                continue
            try:
                obj = json.loads(cand, parse_constant=_reject_json_constant)
                if isinstance(obj, dict):
                    return obj
            except ValueError:
                pass
    try:
        obj = json.loads(cleaned, parse_constant=_reject_json_constant)
        if isinstance(obj, dict):
            return obj
    except ValueError:
        pass
    decoder = json.JSONDecoder(parse_constant=_reject_json_constant)
    scan_pos = 0
    last_obj: dict[str, Any] | None = None
    while True:
        idx = cleaned.find("{", scan_pos)
        if idx < 0:
            return last_obj
        try:
            obj, end_idx = decoder.raw_decode(cleaned, idx)
            if isinstance(obj, dict):
                last_obj = obj
                scan_pos = max(idx + 1, end_idx)
                continue
        except ValueError:
            pass
        scan_pos = idx + 1


def decode_embedded_json_dict(line: str, *, anchor: str = "") -> dict[str, Any] | None:
    """Decode a JSON dict embedded in ``line`` (optionally anchored before ``anchor``), raising ``ValueError`` if malformed."""
    if not isinstance(line, str):
        raise ValueError("line must be a string")
    decoder = json.JSONDecoder(parse_constant=_reject_json_constant)
    brace_idx = line.rfind("{", 0, line.find(anchor)) if anchor and anchor in line else -1
    if brace_idx != -1:
        try:
            payload, _ = decoder.raw_decode(line, brace_idx)
            if isinstance(payload, dict):
                return payload
        except ValueError:
            pass
    start_idx = line.find("{")
    end_idx = line.rfind("}")
    raw_slice = line[start_idx : end_idx + 1] if (start_idx != -1 and end_idx >= start_idx) else line
    try:
        payload = json.loads(raw_slice, parse_constant=_reject_json_constant)
        return payload if isinstance(payload, dict) else None
    except ValueError:
        extracted = extract_json_object(line)
        if extracted is not None:
            return extracted
        raise


def encode_json_payload(
    payload: Any,
    *,
    indent: int | None = None,
    sort_keys: bool = False,
    separators: Any = None,
    ensure_ascii: bool = True,
    default: Any = None,
) -> str:
    """Serialize ``payload`` to a JSON string."""
    kwargs: dict[str, Any] = {"ensure_ascii": _parse_bool_flag(ensure_ascii, True)}
    if indent is not None:
        kwargs["indent"] = exact_nonneg_int(indent, default=None)
    if sort_keys:
        kwargs["sort_keys"] = True
    if separators is not None:
        kwargs["separators"] = separators
    if default is not None:
        kwargs["default"] = default
    return json.dumps(payload, **kwargs)


def read_line_aligned_byte_range(
    path: str,
    start_byte: int,
    end_byte: int,
    *,
    short_file_error: str = "file is shorter than requested byte range",
    boundary_error: str = "byte range does not start at a newline boundary",
    partial_tail_error: str = "byte range ends with a partial line",
) -> bytes:
    """Read ``[start_byte, end_byte)`` from ``path`` and verify newline boundaries."""
    target_path = _resolve_target_path(path)
    with open(target_path, "rb") as handle:
        file_size = os.fstat(handle.fileno()).st_size
        if file_size < end_byte:
            raise ValueError(short_file_error)
        if start_byte > 0:
            handle.seek(start_byte - 1)
            if handle.read(1) != b"\n":
                raise ValueError(boundary_error)
        else:
            handle.seek(0)
        raw = handle.read(end_byte - start_byte)
    if raw and not raw.endswith(b"\n"):
        raise ValueError(partial_tail_error)
    return raw


def append_bytes_at_boundary(
    path: str,
    write_chunks_fn: Callable[[Callable[[bytes], Any]], T],
    *,
    previous_boundary: int | None = None,
    short_file_error: str = "file is shorter than its committed boundary",
    on_rollback_fn: Callable[[int], None] | None = None,
) -> tuple[int, int, T]:
    """Durably append bytes starting at ``previous_boundary`` (truncating uncommitted tail bytes first)."""
    target_path = _resolve_target_path(path, migrate=True)
    initial_size: int | None = None
    try:
        exists = os.path.exists(target_path)
        mode = "r+b" if exists else "w+b"
        with open(target_path, mode) as handle:
            actual_size = os.fstat(handle.fileno()).st_size
            if previous_boundary is not None:
                if actual_size < previous_boundary:
                    raise OSError(short_file_error)
                initial_size = previous_boundary
                if actual_size != previous_boundary:
                    handle.truncate(previous_boundary)
            else:
                initial_size = actual_size
            handle.seek(initial_size)
            result = write_chunks_fn(handle.write)
            _flush_and_fsync(handle)
            committed_bytes = handle.tell()
        return initial_size, committed_bytes, result
    except BaseException:
        if initial_size is not None and on_rollback_fn is not None:
            on_rollback_fn(initial_size)
        raise


def append_jsonl_bucket_delta(
    path: str,
    cache: dict[str, list[Any]],
    persisted_counts: dict[str, int],
    *,
    on_rollback_fn: Callable[[int], None] | None = None,
    fsync: bool = True,
) -> dict[str, int]:
    """Append unpersisted ``(symbol, item)`` rows to ``path`` and return updated counts."""
    target_path = _resolve_target_path(path, migrate=True)
    do_fsync = _parse_bool_flag(fsync, True)
    candidate_counts = dict(persisted_counts)
    initial_size: int | None = None
    try:
        with open(target_path, "a+", encoding="utf-8") as handle:
            initial_size = os.fstat(handle.fileno()).st_size
            for symbol, items in cache.items():
                start = persisted_counts.get(symbol, 0)
                if start > len(items):
                    start = 0
                for item in items[start:]:
                    handle.write(format_jsonl_bucket_line(symbol, item))
                candidate_counts[symbol] = len(items)
            if do_fsync:
                _flush_and_fsync(handle)
            else:
                handle.flush()
        return candidate_counts
    except BaseException:
        if initial_size is not None:
            if on_rollback_fn is not None:
                on_rollback_fn(initial_size)
            else:
                with contextlib.suppress(Exception):
                    truncate_file_bytes(target_path, initial_size)
        raise


def _parse_jsonl_bucket_record(
    raw_line: bytes | str,
    loads_orjson: Callable[[Any], Any] | None,
) -> tuple[Any, Any]:
    """Decode and validate one ``{'s': symbol, 'i': item}`` JSONL record."""
    if loads_orjson is not None:
        try:
            rec = loads_orjson(raw_line)
        except Exception:
            rec = _fast_json_loads(raw_line.decode("utf-8") if isinstance(raw_line, (bytes, bytearray)) else raw_line)
    else:
        rec = _fast_json_loads(raw_line.decode("utf-8") if isinstance(raw_line, (bytes, bytearray)) else raw_line)
    if type(rec) is not dict and not isinstance(rec, dict):
        raise ValueError("invalid JSONL record")
    raw_sym = rec.get("s")
    item = rec.get("i")
    if item is None:
        raise ValueError("invalid JSONL record")
    return raw_sym, item


def _scan_jsonl_bucket_lines(
    handle: Any,
    *,
    binary: bool,
    require_newline: bool,
    mark_unterminated_invalid: bool,
) -> tuple[dict[str, list[Any]], bool, int]:
    """Single canonical JSONL bucket-map parser for full and incremental byte/text streams."""
    cache: dict[str, list[Any]] = {}
    had_invalid_line = False
    consumed = 0
    last_raw_sym: str | None = None
    last_bucket: list[Any] | None = None
    loads_orjson = _orjson.loads if (_orjson is not None and json.loads is _ORIG_JSON_LOADS) else None
    nl: bytes | str = b"\n" if binary else "\n"
    open_brace: int | str = 123 if binary else "{"

    for raw_line in handle:
        line_len = len(raw_line)
        if require_newline and not raw_line.endswith(nl):
            # Unterminated EOF frame (even if valid JSON prior to '\n'): stop without advancing cursor.
            if mark_unterminated_invalid:
                had_invalid_line = True
            break
        if not line_len or (raw_line[0] != open_brace and not raw_line.strip()):
            consumed += line_len
            continue
        try:
            raw_sym, item = _parse_jsonl_bucket_record(raw_line, loads_orjson)
            if type(raw_sym) is str and raw_sym == last_raw_sym and last_bucket is not None:
                last_bucket.append(item)
            else:
                sym = clean_scalar_str(raw_sym)
                if sym is None:
                    raise ValueError("invalid JSONL record")
                bucket = cache.get(sym)
                if bucket is None:
                    bucket = []
                    cache[sym] = bucket
                bucket.append(item)
                if type(raw_sym) is str:
                    last_raw_sym = raw_sym
                    last_bucket = bucket
        except Exception:
            had_invalid_line = True
        consumed += line_len
    return cache, had_invalid_line, consumed


def read_jsonl_bucket_map(path: str) -> tuple[dict[str, list[Any]], bool]:
    """Read a JSONL file of ``{'s': symbol, 'i': item}`` records into ``(cache_map, had_invalid_line)``."""
    target_path = _resolve_target_path(path)
    binary = open is _ORIG_OPEN
    if binary:
        with open(target_path, "rb") as handle:
            cache, had_invalid_line, _ = _scan_jsonl_bucket_lines(
                handle, binary=True, require_newline=False, mark_unterminated_invalid=False
            )
    else:
        with open(target_path, "r", encoding="utf-8") as handle:
            cache, had_invalid_line, _ = _scan_jsonl_bucket_lines(
                handle, binary=False, require_newline=False, mark_unterminated_invalid=False
            )
    return cache, had_invalid_line


def read_jsonl_bucket_map_from_offset(
    path: str,
    *,
    start_offset: int = 0,
    expected_dev_ino: tuple[int, int] | None = None,
    expected_anchor: bytes = b"",
    allow_full_fallback: bool = True,
) -> tuple[dict[str, list[Any]], bool, int, tuple[int, int] | None, bytes, bool]:
    """Incrementally tail newline-terminated ``{'s': symbol, 'i': item}`` JSONL lines when inode and pre-offset anchor match.

    Returns ``(bucket_map, had_invalid_line, next_offset, dev_ino, new_anchor, incremental_used)``.
    When ``incremental_used`` is True, ``bucket_map`` contains only lines newly appended after ``start_offset``;
    otherwise, if ``allow_full_fallback`` is True, the entire file is parsed from offset 0.
    """
    target_path = _resolve_target_path(path)
    with open(target_path, "rb") as handle:
        st = os.fstat(handle.fileno())
        dev_ino = (st.st_dev, st.st_ino)
        incremental_used = False
        if (
            start_offset > 0
            and expected_dev_ino is not None
            and expected_dev_ino == dev_ino
            and st.st_size >= start_offset
        ):
            anchor_len = len(expected_anchor)
            if anchor_len > 0 and start_offset >= anchor_len:
                handle.seek(start_offset - anchor_len)
                if handle.read(anchor_len) == expected_anchor:
                    incremental_used = True
            elif anchor_len == 0:
                handle.seek(start_offset)
                incremental_used = True

        if not incremental_used:
            if not allow_full_fallback:
                return {}, False, 0, dev_ino, b"", False
            handle.seek(0)
            base_offset = 0
        else:
            base_offset = start_offset

        cache, had_invalid_line, consumed = _scan_jsonl_bucket_lines(
            handle,
            binary=True,
            require_newline=True,
            mark_unterminated_invalid=not incremental_used,
        )
        next_offset = base_offset + consumed

        new_anchor = b""
        if next_offset > 0:
            a_len = min(_JSONL_TAIL_ANCHOR_BYTES, next_offset)
            handle.seek(next_offset - a_len)
            new_anchor = handle.read(a_len)
    return cache, had_invalid_line, next_offset, dev_ino, new_anchor, incremental_used


def iter_jsonl_retention_lines(
    path: str,
    keep_record_fn: Callable[[dict[str, Any]], bool],
    *,
    yield_every: int = 4096,
) -> Iterator[tuple[str, bool]]:
    """Stream a JSONL file line by line, yielding ``(line_with_newline, keep_flag)``."""
    target_path = _resolve_target_path(path)
    with open(target_path, encoding="utf-8") as source:
        for line_idx, line in enumerate(source, 1):
            if yield_every > 0 and line_idx % yield_every == 0:
                time.sleep(0)
            try:
                rec = _fast_json_loads(line)
                keep = isinstance(rec, dict) and bool(keep_record_fn(rec))
            except (json.JSONDecodeError, TypeError, ValueError):
                keep = False
            yield (line if line.endswith("\n") else line + "\n"), keep


def read_tail_complete_lines_text(path: str, keep_bytes: int) -> str:
    """Read up to the last ``keep_bytes`` of ``path`` aligned to a complete line boundary."""
    target_path = _resolve_target_path(path)
    archive_size = os.path.getsize(target_path)
    keep_i = exact_pos_int(keep_bytes, default=None)
    if keep_i is None and isinstance(keep_bytes, float) and math.isfinite(keep_bytes) and keep_bytes >= 1.0:
        keep_i = math.trunc(keep_bytes)
    eff_keep = keep_i if keep_i is not None else 1
    with open(target_path, "rb") as source:
        source.seek(max(0, archive_size - eff_keep))
        if source.tell() > 0:
            source.readline()
        tail = source.read()
        if not tail.strip() and archive_size > 0:
            source.seek(0)
            lines = [
                line
                for line in source.read().splitlines(keepends=True)
                if line.strip()
            ]
            tail = lines[-1] if lines else b""
    decoded = tail.decode("utf-8", errors="ignore")
    if decoded and not decoded.endswith("\n"):
        decoded += "\n"
    return decoded


def prune_rotated_archives(base_filename: str, keep_count: int) -> list[str]:
    """Retain the newest ``keep_count`` ``<base_filename>.*.archive*`` files."""
    if not base_filename or isinstance(base_filename, (bool, int, float)):
        return []
    try:
        base_fspath = os.fspath(base_filename)
    except TypeError:
        return []
    candidates: list[tuple[float, str]] = []
    for path in glob.glob(f"{glob.escape(base_fspath)}.*.archive*"):
        if path.endswith(".tmp"):
            continue
        try:
            candidates.append((os.path.getmtime(path), path))
        except OSError:
            continue
    candidates.sort(key=lambda item: item[0], reverse=True)
    archives = [path for _, path in candidates]
    keep = exact_nonneg_int(keep_count, default=0) or 0
    for old_archive in archives[keep:]:
        discard_generation_file(old_archive)
    return archives[:keep]


# ---------------------------------------------------------------------------
# Transactional multi-file generation API (<file> + <file>.previous + <file>.meta)
# ---------------------------------------------------------------------------

def _sidecar_path(path: str, suffix: str) -> str:
    return f"{os.fspath(path)}.{suffix}"


previous_generation_path = partial(_sidecar_path, suffix="previous")
meta_file_path = partial(_sidecar_path, suffix="meta")
generation_lock_path = partial(_sidecar_path, suffix="generation.lock")

_GEN_THREAD_LOCKS_GUARD = threading.Lock()
_GEN_THREAD_LOCKS: dict[str, threading.RLock] = {}
_GEN_LOCK_TLS = threading.local()


@contextlib.contextmanager
def generation_file_lock(
    path: str | os.PathLike[str],
    *,
    lock_path: str | os.PathLike[str] | None = None,
) -> Iterator[str]:
    """Acquire a thread-reentrant and cross-process ``flock`` around a read-modify-write state transaction."""
    eff_lock = (
        _resolve_target_path(lock_path, abspath=True, migrate=True)
        if lock_path is not None
        else os.path.abspath(generation_lock_path(_resolve_target_path(path, migrate=True)))
    )
    parent = os.path.dirname(eff_lock)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with _GEN_THREAD_LOCKS_GUARD:
        tlock = _GEN_THREAD_LOCKS.get(eff_lock)
        if tlock is None:
            if len(_GEN_THREAD_LOCKS) > 512:
                _GEN_THREAD_LOCKS.clear()
            tlock = threading.RLock()
            _GEN_THREAD_LOCKS[eff_lock] = tlock
    held: set[tuple[int, str]] | None = getattr(_GEN_LOCK_TLS, "held", None)
    if held is None:
        held = set()
        _GEN_LOCK_TLS.held = held
    key = (os.getpid(), eff_lock)
    with tlock:
        if key in held:
            yield eff_lock
            return
        with open(eff_lock, "a+", encoding="utf-8") as lock_handle:
            if fcntl is not None:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            held.add(key)
            try:
                yield eff_lock
            finally:
                held.discard(key)
                if fcntl is not None:
                    with contextlib.suppress(OSError):
                        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def discard_generation_file(
    path: str | None, *, include_previous: bool = False, include_meta: bool = False,
) -> None:
    """Best-effort removal of a state/generation file (and optional ``.previous`` / ``.meta`` sidecars)."""
    if not path or isinstance(path, (bool, int, float)):
        return
    try:
        targets = [path]
        if include_previous:
            targets.append(previous_generation_path(path))
        if include_meta:
            targets.append(meta_file_path(path))
    except TypeError:
        return
    for target in targets:
        try:
            os.remove(target)
        except (OSError, TypeError):
            pass


def stage_previous_generation(
    path: str, *, backup_path: str | None = None, fsync: bool = True,
) -> str:
    """Snapshot the currently certified ``path`` at ``backup_path`` (default ``.previous``)."""
    target_backup = backup_path or previous_generation_path(path)
    atomic_snapshot_file(path, target_backup, fsync=fsync)
    return target_backup


def restore_previous_generation(path: str, *, backup_path: str | None = None) -> None:
    """Durably restore ``backup_path`` (default ``.previous``) onto ``path``."""
    durable_replace_file(backup_path or previous_generation_path(path), path)


def write_meta_json(path: str, payload: dict[str, Any], *, fsync: bool = True) -> None:
    """Atomically write ``payload`` to ``<path>.meta``."""
    atomic_write_json(meta_file_path(path), payload, fsync=fsync)


def commit_file_and_meta(
    path: str, write_data_fn: Callable[[], Any], write_meta_fn: Callable[[], bool], *,
    backup_path: str | None = None, had_prior_file: bool | None = None,
    on_rollback_error: Callable[[OSError], None] | None = None,
    fsync: bool = True,
) -> bool:
    """Transactionally replace ``path`` and commit its ``.meta`` manifest."""
    target_path = _resolve_target_path(path, migrate=True)
    do_fsync = _parse_bool_flag(fsync, True)
    if had_prior_file is None:
        had_prior_file = os.path.exists(target_path)
        if had_prior_file and backup_path is None:
            try:
                backup_path = (
                    stage_previous_generation(target_path)
                    if do_fsync
                    else stage_previous_generation(target_path, fsync=False)
                )
            except FileNotFoundError:
                backup_path = None
                had_prior_file = os.path.exists(target_path)
        elif not had_prior_file:
            discard_generation_file(previous_generation_path(target_path))
    else:
        had_prior_file = _parse_bool_flag(had_prior_file, False)
    data_replaced = False
    try:
        write_data_fn()
        data_replaced = True
        if not _parse_bool_flag(write_meta_fn(), False):
            _rollback_generation(target_path, backup_path, had_prior_file)
            return False
        if backup_path is not None:
            discard_generation_file(backup_path)
        return True
    except BaseException:
        if data_replaced:
            _rollback_generation(
                target_path, backup_path, had_prior_file,
                on_rollback_error=on_rollback_error, suppress_error=True,
            )
        raise
    finally:
        if backup_path is not None and os.path.exists(backup_path):
            discard_generation_file(backup_path)


def _read_json_file_strict(path: str) -> Any:
    return _read_json_candidate(_resolve_target_path(path))


def _read_json_dict_strict(path: str) -> dict[str, Any]:
    payload = _read_json_file_strict(path)
    if not isinstance(payload, dict):
        raise ValueError("JSON root must be a dict")
    return payload


def _read_with_policy(
    path: str, reader_fn: Callable[[str], T], *,
    default: Any = _MISSING, backup_path: str | None = None,
    recover_previous: bool = False, repair_files: bool = False,
) -> T | Any:
    if recover_previous:
        return read_recovering_previous(
            path, reader_fn, backup_path=backup_path, default=default, repair_files=repair_files
        )
    try:
        return reader_fn(_resolve_target_path(path, migrate=_parse_bool_flag(repair_files, False)))
    except Exception:
        if default is not _MISSING:
            return default
        raise


def read_json_file(
    path: str, *, default: Any = _MISSING, backup_path: str | None = None,
    recover_previous: bool = False, repair_files: bool = False,
) -> Any:
    """Read and parse any UTF-8 JSON file, with optional ``.previous`` fallback/recovery or ``default``."""
    return _read_with_policy(
        path, _read_json_file_strict, default=default, backup_path=backup_path,
        recover_previous=recover_previous, repair_files=repair_files,
    )


def read_json_dict(
    path: str, *, default: Any = _MISSING, backup_path: str | None = None,
    recover_previous: bool = False, repair_files: bool = False,
) -> dict[str, Any] | Any:
    """Read and parse a UTF-8 JSON file whose root value must be a ``dict``."""
    return _read_with_policy(
        path, _read_json_dict_strict, default=default, backup_path=backup_path,
        recover_previous=recover_previous, repair_files=repair_files,
    )


def iter_jsonl_records(path: str, *, strict: bool = False) -> Iterator[Any]:
    """Stream JSON records line by line from a UTF-8 JSONL file without loading the full file in RAM."""
    try:
        target = _resolve_target_path(path)
    except (TypeError, ValueError):
        if strict:
            raise
        return
    try:
        if not strict and not os.path.isfile(target):
            return
        with open(target, "r", encoding="utf-8") as handle:
            for raw in filter(None, map(str.strip, handle)):
                try:
                    yield _fast_json_loads(raw)
                except ValueError:
                    if strict:
                        raise
    except (OSError, TypeError):
        if strict:
            raise


def iter_jsonl_dicts(path: str, *, strict: bool = False) -> Iterator[dict[str, Any]]:
    """Stream JSON object records (``dict``) line by line from a UTF-8 JSONL file."""
    for rec in iter_jsonl_records(path, strict=strict):
        if isinstance(rec, dict):
            yield rec
        elif strict:
            raise ValueError("JSONL line must be a dict")


def read_jsonl_records(path: str, *, strict: bool = False) -> list[Any]:
    """Read JSON records from a UTF-8 JSONL file, optionally enforcing strict line parsing."""
    return list(iter_jsonl_records(path, strict=strict))


def read_jsonl_dicts(path: str, *, strict: bool = False) -> list[dict[str, Any]]:
    """Read JSON object records (``dict``) from a UTF-8 JSONL file."""
    return list(iter_jsonl_dicts(path, strict=strict))


def read_jsonl_dicts_from_offset(path: str, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
    """Read complete newline-terminated JSON object lines starting at byte ``offset``."""
    target = _resolve_target_path(path)
    if not os.path.isfile(target):
        return [], 0
    start_off = offset if (isinstance(offset, int) and not isinstance(offset, bool) and offset > 0) else 0
    records: list[dict[str, Any]] = []
    try:
        file_size = os.path.getsize(target)
        if start_off > file_size:
            start_off = 0
        with open(target, "rb") as handle:
            if start_off:
                handle.seek(start_off)
            chunk = handle.read()
    except OSError:
        return [], start_off
    if not chunk:
        return [], start_off
    last_nl = chunk.rfind(b"\n")
    if last_nl < 0:
        return [], start_off
    complete = chunk[: last_nl + 1]
    new_offset = start_off + len(complete)
    for raw_line in complete.splitlines():
        if not raw_line.strip():
            continue
        try:
            item = json.loads(raw_line.decode("utf-8", errors="replace"))
        except (ValueError, TypeError):
            continue
        if isinstance(item, dict):
            records.append(item)
    return records, new_offset


def read_text_file(
    path: str, *, default: Any = _MISSING, encoding: str = "utf-8", errors: str = "strict",
    newline: str | None = None, backup_path: str | None = None,
    recover_previous: bool = False, repair_files: bool = False,
) -> str | Any:
    """Read text from ``path`` (symmetric to ``atomic_write_text``), with optional fallback ``default``."""
    def _read_strict(candidate_path: str) -> str:
        with open(candidate_path, "r", encoding=encoding, errors=errors, newline=newline) as handle:
            return handle.read()

    return _read_with_policy(
        path, _read_strict, default=default, backup_path=backup_path,
        recover_previous=recover_previous, repair_files=repair_files,
    )


def iter_text_lines(
    path: str, *, encoding: str = "utf-8", errors: str = "strict",
) -> Iterator[str]:
    """Stream raw text lines (with trailing newlines preserved) from ``path`` in O(1) memory."""
    with open(_resolve_target_path(path), "r", encoding=encoding, errors=errors) as handle:
        yield from handle


def read_text_lines(
    path: str, *, strip: bool = False, skip_empty: bool = False, keepends: bool = False,
    encoding: str = "utf-8", errors: str = "strict",
) -> list[str]:
    """Read lines from a text file, returning ``[]`` when missing or unreadable."""
    try:
        with open(_resolve_target_path(path), "r", encoding=encoding, errors=errors) as handle:
            return [
                raw.strip() if strip else (raw if keepends else raw.rstrip("\r\n"))
                for raw in handle
                if not (skip_empty and not raw.strip())
            ]
    except (OSError, TypeError, ValueError):
        return []


def read_bytes_file(
    path: str, *, default: Any = _MISSING, backup_path: str | None = None,
    recover_previous: bool = False, repair_files: bool = False,
) -> bytes | Any:
    """Read binary bytes from ``path`` (symmetric to ``atomic_write_bytes``), with optional fallback ``default``."""
    def _read_strict(candidate_path: str) -> bytes:
        with open(candidate_path, "rb") as handle:
            return handle.read()

    return _read_with_policy(
        path, _read_strict, default=default, backup_path=backup_path,
        recover_previous=recover_previous, repair_files=repair_files,
    )


def read_csv_dicts(
    path: str, *, encoding: str = "utf-8", strict: bool = False, default: Any = _MISSING,
) -> list[dict[str, Any]]:
    """Read rows from a CSV file via ``csv.DictReader``."""
    def _read_strict(candidate_path: str) -> list[dict[str, Any]]:
        with open(candidate_path, "r", encoding=encoding, newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    eff_default = default if default is not _MISSING else (_MISSING if strict else [])
    return _read_with_policy(path, _read_strict, default=eff_default)


def read_recovering_previous(
    path: str, read_fn: Callable[[str], T] = json.loads, *,
    backup_path: str | None = None,
    error_mapper: Callable[[BaseException], BaseException] | None = None,
    default: Any = _MISSING,
    repair_files: bool = True,
) -> T:
    """Read and validate ``path`` via ``read_fn``, falling back to ``.previous`` (and repairing on disk only when ``repair_files=True``)."""
    repair = _parse_bool_flag(repair_files, True)
    if read_fn is json.loads:
        def _invoke(candidate_path: str) -> Any:
            with open(candidate_path, "r", encoding="utf-8") as handle:
                return json.loads(handle.read())
    elif read_fn is dict:
        _invoke = _read_json_dict_strict
    else:
        _invoke = read_fn
    target_backup: str | None = None
    try:
        target_path = _resolve_target_path(path, migrate=repair)
        target_backup = (
            _resolve_target_path(backup_path, migrate=repair)
            if backup_path
            else previous_generation_path(target_path)
        )
        result = _invoke(target_path)
        if repair and os.path.exists(target_backup):
            try:
                is_staged = os.path.samefile(target_path, target_backup)
            except OSError:
                is_staged = False
            if not is_staged:
                discard_generation_file(target_backup)
        return result
    except Exception as primary_exc:
        if not target_backup or not os.path.exists(target_backup):
            if default is _MISSING:
                raise
            return default
        try:
            backup_result = _invoke(target_backup)
            if not repair:
                return backup_result
            restore_previous_generation(target_path, backup_path=target_backup)
            return _invoke(target_path)
        except Exception as recovery_exc:
            if default is not _MISSING:
                return default
            if error_mapper is not None:
                mapped = error_mapper(recovery_exc)
                if mapped is not recovery_exc:
                    raise mapped from recovery_exc
            raise primary_exc from recovery_exc


_JSON_DICT_MEMO: dict[str, tuple[tuple, dict[str, Any]]] = {}
_ORIG_READ_JSON_DICT = read_json_dict
_ORIG_READ_RECOVERING_PREVIOUS = read_recovering_previous


def _stat_sig(path: str) -> tuple[int, int, int] | None:
    try:
        st = os.stat(path)
        return (st.st_mtime_ns, st.st_size, st.st_ino)
    except OSError:
        return None


def read_json_recovering(
    path: str, *, default: Any = None, repair_files: bool = False,
) -> Any:
    """Read and parse any UTF-8 JSON file with passive ``.previous`` fallback (or producer repair when ``repair_files=True``)."""
    return read_json_file(path, default=default, recover_previous=True, repair_files=repair_files)


def read_json_dict_recovering(
    path: str, *, default: Any = None, repair_files: bool = False,
) -> dict[str, Any] | Any:
    """Read and parse a UTF-8 JSON dict file with passive ``.previous`` fallback (or producer repair when ``repair_files=True``)."""
    return read_json_dict(path, default=default, recover_previous=True, repair_files=repair_files)


def read_json_dict_cached(
    path: str | os.PathLike[str],
    memo: dict[str, tuple[tuple, dict[str, Any]]] | None = None,
    *, max_entries: int = 32,
) -> dict[str, Any] | None:
    """Stat-memoized passive ``read_json_dict`` checking both ``path`` and ``path.previous`` without mutating disk."""
    try:
        resolved = _resolve_target_path(path)
    except Exception:
        return None
    backup_path = previous_generation_path(resolved)
    unpatched = is_state_io_dict_reader_unpatched()
    st_main, st_bak = _stat_sig(resolved), _stat_sig(backup_path)
    if st_main is None and st_bak is None and unpatched:
        return None
    if not unpatched:
        res_patched = (
            read_json_dict_recovering(resolved, default=None, repair_files=False)
            if read_json_dict_recovering is not _ORIG_READ_JSON_DICT_RECOVERING
            else read_json_dict(
                resolved, default=None, backup_path=backup_path, recover_previous=True, repair_files=False
            )
        )
        return res_patched if isinstance(res_patched, dict) else None

    cache_store = _JSON_DICT_MEMO if memo is None else memo
    sig = (st_main, st_bak, read_recovering_previous, json.load)
    cached = cache_store.get(resolved)
    if cached is not None and cached[0] == sig:
        return cached[1]

    res = read_json_dict(
        resolved, default=None, backup_path=backup_path, recover_previous=True, repair_files=False
    )
    if isinstance(res, dict):
        if st_bak is not None:
            sig = (_stat_sig(resolved), _stat_sig(backup_path), read_recovering_previous, json.load)
        if resolved in cache_store:
            cache_store.pop(resolved, None)
        cache_store[resolved] = (sig, res)
        for old_key in list(cache_store)[: max(0, len(cache_store) - max_entries)]:
            cache_store.pop(old_key, None)
        return res
    return None


def _looks_like_jsonl_bucket_file(path: str) -> bool:
    try:
        with open(path, "rb") as handle:
            for raw in handle:
                s = raw.strip()
                if s:
                    return s.startswith(b'{"s":') or s.startswith(b'{"s" :')
    except OSError:
        pass
    return False


_JSONL_CACHED_CURSOR_LOCK = threading.Lock()
_JSONL_CACHED_CURSORS: dict[tuple[int, str], tuple[int, tuple[int, int], int, bytes]] = {}


def read_jsonl_or_json_dict_cached(
    path: str | os.PathLike[str],
    memo: dict[str, tuple[tuple, dict[str, Any]]] | None = None,
    *, max_entries: int = 32,
) -> dict[str, Any] | None:
    """Stat-memoized reader supporting both JSONL bucket files (``{'s', 'i'}``) and legacy JSON dicts,
    with thread-safe incremental byte-offset tailing on appended JSONL files."""
    try:
        resolved = _resolve_target_path(path)
    except Exception:
        return None
    if not resolved.endswith(".jsonl") or not is_state_io_dict_reader_unpatched():
        return read_json_dict_cached(path, memo, max_entries=max_entries)
    backup_path = previous_generation_path(resolved)
    st_main, st_bak = _stat_sig(resolved), _stat_sig(backup_path)
    if st_main is None and st_bak is None:
        return None
    cache_store = _JSON_DICT_MEMO if memo is None else memo
    sig = (st_main, st_bak, read_recovering_previous, json.load, "jsonl")
    cursor_key = (id(cache_store), resolved)
    with _JSONL_CACHED_CURSOR_LOCK:
        cached = cache_store.get(resolved)
        if cached is not None and cached[0] == sig:
            return cached[1]
        prev_cursor = _JSONL_CACHED_CURSORS.get(cursor_key)
    main_is_jsonl = st_main is not None and _looks_like_jsonl_bucket_file(resolved)
    if not main_is_jsonl:
        bak_is_jsonl = st_bak is not None and _looks_like_jsonl_bucket_file(backup_path)
        if not bak_is_jsonl:
            return read_json_dict_cached(resolved, memo, max_entries=max_entries)
        if st_main is not None and isinstance(
            read_json_dict(resolved, default=None, recover_previous=False, repair_files=False),
            dict,
        ):
            return read_json_dict_cached(resolved, memo, max_entries=max_entries)

    prev_res = cached[1] if (cached is not None and isinstance(cached[1], dict)) else None
    prev_items = prev_res.get("items") if (prev_res is not None and isinstance(prev_res.get("items"), dict)) else None
    bucket_map: dict[str, list[Any]] = {}
    new_cursor: tuple[tuple[int, int], int, bytes] | None = None

    if st_main is not None and os.path.exists(resolved):
        try:
            can_tail = (
                prev_cursor is not None
                and prev_res is not None
                and prev_cursor[0] == id(prev_res)
                and bool(prev_items)
            )
            parsed_map, had_bad, next_off, dev_ino, new_anchor, inc_used = (
                read_jsonl_bucket_map_from_offset(
                    resolved,
                    start_offset=prev_cursor[2] if can_tail else 0,
                    expected_dev_ino=prev_cursor[1] if can_tail else None,
                    expected_anchor=prev_cursor[3] if can_tail else b"",
                    allow_full_fallback=True,
                )
            )
            if inc_used and isinstance(prev_items, dict):
                if parsed_map:
                    bucket_map = {k: list(v) for k, v in prev_items.items()}
                    for sym_k, rows_delta in parsed_map.items():
                        bucket_map.setdefault(sym_k, []).extend(rows_delta)
                else:
                    bucket_map = prev_items
            else:
                bucket_map = parsed_map
            if not had_bad and dev_ino is not None and next_off > 0 and bucket_map:
                new_cursor = (dev_ino, next_off, new_anchor)
        except Exception:
            bucket_map = {}

    if not bucket_map and backup_path and os.path.exists(backup_path):
        try:
            parsed_map, _ = read_jsonl_bucket_map(backup_path)
            if parsed_map:
                bucket_map = parsed_map
                new_cursor = None
        except Exception:
            pass

    if not bucket_map:
        with _JSONL_CACHED_CURSOR_LOCK:
            _JSONL_CACHED_CURSORS.pop(cursor_key, None)
        return None
    res: dict[str, Any] = {"items": bucket_map}
    with _JSONL_CACHED_CURSOR_LOCK:
        if resolved in cache_store:
            cache_store.pop(resolved, None)
        cache_store[resolved] = (sig, res)
        if new_cursor is not None:
            _JSONL_CACHED_CURSORS[cursor_key] = (id(res), new_cursor[0], new_cursor[1], new_cursor[2])
        else:
            _JSONL_CACHED_CURSORS.pop(cursor_key, None)
        for old_key in list(cache_store)[: max(0, len(cache_store) - max_entries)]:
            cache_store.pop(old_key, None)
            _JSONL_CACHED_CURSORS.pop((id(cache_store), old_key), None)
        if len(_JSONL_CACHED_CURSORS) > max(64, max_entries * 4):
            for stale_ck in list(_JSONL_CACHED_CURSORS)[: len(_JSONL_CACHED_CURSORS) - max(64, max_entries * 2)]:
                _JSONL_CACHED_CURSORS.pop(stale_ck, None)
    return res


_ORIG_READ_JSON_DICT_RECOVERING = read_json_dict_recovering
_ORIG_READ_JSON_DICT_CACHED = read_json_dict_cached


def is_state_io_dict_reader_unpatched() -> bool:
    """Return True when ``state_io`` dict/recovering readers are unpatched (survives ``importlib.reload(state_io)``)."""
    return (
        read_json_dict is _ORIG_READ_JSON_DICT
        and read_recovering_previous is _ORIG_READ_RECOVERING_PREVIOUS
        and read_json_dict_recovering is _ORIG_READ_JSON_DICT_RECOVERING
        and read_json_dict_cached is _ORIG_READ_JSON_DICT_CACHED
    )


def parse_proc_stat_fields(stat_text: str, *, min_fields: int = 20) -> tuple[str, list[str]]:
    """Parse ``(comm, fields_after_rparen)`` from ``/proc/<pid>/stat`` safely around parenthesized ``comm``."""
    if not isinstance(stat_text, str):
        raise ValueError("malformed /proc/stat line")
    lparen = stat_text.find("(")
    rparen = stat_text.rfind(")")
    if lparen < 0 or rparen <= lparen:
        raise ValueError("malformed /proc/stat line")
    comm = stat_text[lparen + 1:rparen]
    after = stat_text[rparen + 1:].split()
    if len(after) < min_fields:
        raise ValueError("malformed /proc/stat line")
    return comm, after
