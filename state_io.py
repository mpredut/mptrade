"""Shared durable file primitives for runtime and financial state.

Writers use a unique temporary file in the destination directory, flush file data,
atomically replace the destination, and clean up on failure. Locking and failure
policy remain the caller's responsibility because caches, live financial state,
and observational state intentionally have different concurrency semantics.
"""
from __future__ import annotations

import contextlib
import glob
import gzip
import json
import math
import os
import shutil
import tempfile
from collections.abc import Callable, Iterator
from typing import Any, TypeVar

T = TypeVar("T")


def _parse_bool_flag(val: Any, default: bool = False) -> bool:
    if isinstance(val, bool):
        return val
    if val is None or isinstance(val, (dict, list, tuple, set, frozenset)):
        return default
    if isinstance(val, (int, float)):
        if isinstance(val, float) and not math.isfinite(val):
            return default
        return val != 0
    if isinstance(val, str):
        s = val.strip().lower()
        if s in {"1", "true", "yes", "on", "y", "t"}:
            return True
        if s in {"0", "false", "no", "off", "n", "f", "", "none", "null", "nan"}:
            return False
    return default


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
        # Directory handles are not portable (notably on native Windows).
        pass


def durable_replace_file(source: str, destination: str) -> None:
    """Atomically replace ``destination`` and durably publish the new entry."""
    source = os.path.abspath(os.fspath(source))
    destination = os.path.abspath(os.fspath(destination))
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    os.replace(source, destination)
    if source != destination and os.path.exists(source):
        os.remove(source)
    with open(destination, "rb") as handle:
        os.fsync(handle.fileno())
    _fsync_parent_directory(destination)


def atomic_snapshot_file(source: str, destination: str) -> None:
    """Atomically preserve one immutable file generation at ``destination``.

    A hard link avoids copying a potentially large generation when supported;
    otherwise the bytes are copied and fsynced. Callers must publish subsequent
    source generations by replacement rather than mutating the source in place.
    """
    source = os.path.abspath(os.fspath(source))
    destination = os.path.abspath(os.fspath(destination))
    directory = os.path.dirname(destination)
    os.makedirs(directory, exist_ok=True)
    staging_directory = tempfile.mkdtemp(
        dir=directory, prefix=f".{os.path.basename(destination)}."
    )
    staged = os.path.join(staging_directory, "generation")
    try:
        try:
            os.link(source, staged)
        except OSError:
            shutil.copyfile(source, staged)
            with open(staged, "rb") as handle:
                os.fsync(handle.fileno())
        durable_replace_file(staged, destination)
    finally:
        try:
            os.unlink(staged)
        except OSError:
            pass
        try:
            os.rmdir(staging_directory)
        except OSError:
            pass


def _reject_json_constant(constant: str) -> Any:
    raise ValueError(f"non-standard JSON constant {constant!r}")


def load_json_state(path: str, *, default_factory, fail_closed: bool,
                    label: str, root_type: type = dict):
    """Load JSON state with an explicit live-versus-observational failure policy.

    A missing file represents first start and returns the factory default. Invalid,
    unreadable, or structurally wrong state heals from ``<path>.previous`` when
    present, or raises in fail-closed mode; paper, shadow, and monitoring callers
    may explicitly opt into a clean default.
    """
    fail_closed = _parse_bool_flag(fail_closed, True)
    target_path = os.fspath(path)
    backup_path = previous_generation_path(target_path)
    if not os.path.exists(target_path) and not os.path.exists(backup_path):
        return default_factory()

    def _read_and_validate(candidate_path: str):
        with open(candidate_path, "r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                parse_constant=_reject_json_constant if fail_closed else None,
            )
        if not isinstance(value, root_type):
            raise ValueError(
                f"expected {root_type.__name__}, got {type(value).__name__}"
            )
        return value

    try:
        return read_recovering_previous(
            target_path, _read_and_validate, backup_path=backup_path
        )
    except (OSError, TypeError, ValueError) as exc:
        if fail_closed:
            raise StateReadError(f"invalid {label} state in {path}: {exc}") from exc
        return default_factory()


@contextlib.contextmanager
def atomic_text_writer(path: str, *, encoding: str = "utf-8") -> Iterator[Any]:
    """Yield a text handle and atomically publish it only after a successful write."""
    destination = os.path.abspath(os.fspath(path))
    directory = os.path.dirname(destination)
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=directory, prefix=f".{os.path.basename(destination)}.", suffix=".tmp",
    )
    handle = os.fdopen(fd, "w", encoding=encoding)
    try:
        yield handle
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(temporary, destination)
        _fsync_parent_directory(destination)
    except BaseException:
        handle.close()
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def atomic_write_json(path: str, value: Any, *, indent: int | None = None,
                      sort_keys: bool = False, separators=None,
                      ensure_ascii: bool = True) -> None:
    """Serialize JSON through ``atomic_text_writer``."""
    sort_keys = _parse_bool_flag(sort_keys, False)
    ensure_ascii = _parse_bool_flag(ensure_ascii, True)
    clean_indent: int | None = None
    if indent is not None and not isinstance(indent, bool):
        try:
            ind_f = float(indent)
            if math.isfinite(ind_f) and ind_f.is_integer() and int(ind_f) >= 0:
                clean_indent = int(ind_f)
        except (TypeError, ValueError, OverflowError):
            clean_indent = None
    with atomic_text_writer(path) as handle:
        json.dump(
            value, handle, indent=clean_indent, sort_keys=sort_keys,
            separators=separators, ensure_ascii=ensure_ascii,
            allow_nan=False,
        )


def save_json_state(path: str, value: Any, *, indent: int | None = None,
                    sort_keys: bool = False, separators=None,
                    ensure_ascii: bool = True) -> None:
    """Transactionally write JSON state while staging and cleaning up ``<path>.previous``."""
    target_path = os.fspath(path)
    backup_path = None
    had_prior = os.path.exists(target_path)
    if had_prior:
        try:
            backup_path = stage_previous_generation(target_path)
        except OSError:
            backup_path = None
    try:
        atomic_write_json(
            target_path, value, indent=indent, sort_keys=sort_keys,
            separators=separators, ensure_ascii=ensure_ascii,
        )
        if backup_path is not None:
            discard_generation_file(backup_path)
    except BaseException:
        if backup_path is not None and os.path.exists(backup_path):
            try:
                restore_previous_generation(target_path, backup_path=backup_path)
            except OSError:
                pass
        elif not had_prior:
            discard_generation_file(target_path)
        raise


@contextlib.contextmanager
def atomic_binary_writer(path: str) -> Iterator[Any]:
    """Yield a binary handle and atomically publish it only after a durable write."""
    destination = os.path.abspath(os.fspath(path))
    directory = os.path.dirname(destination)
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=directory, prefix=f".{os.path.basename(destination)}.", suffix=".tmp",
    )
    handle = os.fdopen(fd, "wb")
    try:
        yield handle
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(temporary, destination)
        _fsync_parent_directory(destination)
    except BaseException:
        handle.close()
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def atomic_write_bytes(path: str, data: bytes | bytearray | memoryview) -> None:
    """Durably and atomically write binary bytes to ``path``."""
    with atomic_binary_writer(path) as handle:
        handle.write(data)


@contextlib.contextmanager
def atomic_file_path(
    path: str,
    *,
    prefix: str | None = None,
    suffix: str = ".tmp",
) -> Iterator[str]:
    """Yield a temporary sibling path and durably publish it onto ``path`` on exit."""
    destination = os.path.abspath(os.fspath(path))
    directory = os.path.dirname(destination)
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=directory,
        prefix=prefix if prefix is not None else f".{os.path.basename(destination)}.",
        suffix=suffix,
    )
    os.close(fd)
    try:
        yield temporary
        durable_replace_file(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _positive_chunk_size(chunk_size: Any, default: int) -> int:
    if chunk_size is None or isinstance(chunk_size, bool):
        return default
    try:
        cf = float(chunk_size)
        if not (cf == int(cf)):
            return default
        val = int(cf)
        return val if val > 0 else default
    except (TypeError, ValueError, OverflowError):
        return default


def atomic_gzip_file(
    source: str,
    destination: str | None = None,
    *,
    remove_source: bool = False,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Compress ``source`` into ``destination`` (default ``<source>.gz``) atomically."""
    remove_source = _parse_bool_flag(remove_source, False)
    eff_chunk = _positive_chunk_size(chunk_size, 1024 * 1024)
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


def read_tail_lines(
    path: str,
    limit: int,
    *,
    chunk_size: int = 65536,
) -> list[bytes]:
    """Read up to the last ``limit`` lines from ``path`` with bounded memory."""
    if not path or isinstance(path, (bool, int)):
        return []
    if limit is None or isinstance(limit, bool):
        return []
    try:
        lf = float(limit)
        if not (lf == int(lf)):
            return []
        limit = int(lf)
    except (TypeError, ValueError, OverflowError):
        return []
    if limit <= 0 or not os.path.isfile(path):
        return []
    eff_chunk = _positive_chunk_size(chunk_size, 65536)
    chunks: list[bytes] = []
    newline_count = 0
    with open(path, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        while position > 0:
            step = min(eff_chunk, position)
            position -= step
            handle.seek(position)
            chunk = handle.read(step)
            chunks.append(chunk)
            newline_count += chunk.count(b"\n")
            if newline_count > limit and position > 0:
                joined = b"".join(reversed(chunks))
                complete = joined.split(b"\n", 1)[1] if b"\n" in joined else b""
                non_empty = [line for line in complete.splitlines() if line.strip()]
                if len(non_empty) >= limit:
                    return non_empty[-limit:]
                newline_count = len(non_empty)
    joined = b"".join(reversed(chunks))
    if position > 0 and b"\n" in joined:
        joined = joined.split(b"\n", 1)[1]
    return [line for line in joined.splitlines() if line.strip()][-limit:]


def truncate_file_bytes(path: str, size: int) -> None:
    """Durably truncate ``path`` back to ``size`` bytes."""
    if isinstance(size, bool):
        raise ValueError("size must be a non-negative integer")
    try:
        size_f = float(size)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("size must be a non-negative integer") from exc
    if not (math.isfinite(size_f) and size_f.is_integer() and int(size_f) >= 0):
        raise ValueError("size must be a non-negative integer")
    size_i = int(size_f)
    with open(path, "r+b") as handle:
        handle.truncate(size_i)
        handle.flush()
        os.fsync(handle.fileno())


def prune_rotated_archives(
    base_filename: str,
    keep_count: int,
) -> list[str]:
    """Retain the newest ``keep_count`` ``<base_filename>.*.archive*`` files."""
    if not base_filename or isinstance(base_filename, (bool, int)):
        return []
    try:
        base_fspath = os.fspath(base_filename)
    except TypeError:
        return []
    pattern = f"{glob.escape(base_fspath)}.*.archive*"
    candidates: list[tuple[float, str]] = []
    for path in glob.glob(pattern):
        if path.endswith(".tmp"):
            continue
        try:
            candidates.append((os.path.getmtime(path), path))
        except OSError:
            continue
    candidates.sort(key=lambda item: item[0], reverse=True)
    archives = [path for _, path in candidates]
    if keep_count is None or isinstance(keep_count, bool):
        keep = 0
    else:
        try:
            kf = float(keep_count)
            keep = max(0, int(kf)) if (kf == int(kf)) else 0
        except (TypeError, ValueError, OverflowError):
            keep = 0
    for old_archive in archives[keep:]:
        discard_generation_file(old_archive)
    return archives[:keep]


# ---------------------------------------------------------------------------
# Transactional multi-file generation API (<file> + <file>.previous + <file>.meta)
# ---------------------------------------------------------------------------

def previous_generation_path(path: str) -> str:
    """Return the canonical ``.previous`` staging path for ``path``."""
    return f"{os.fspath(path)}.previous"


def meta_file_path(path: str) -> str:
    """Return the canonical ``.meta`` manifest path for ``path``."""
    return f"{os.fspath(path)}.meta"


def generation_lock_path(path: str) -> str:
    """Return the canonical cross-process generation lock path for ``path``."""
    return f"{os.fspath(path)}.generation.lock"


def discard_generation_file(path: str | None) -> None:
    """Best-effort removal of a staged or temporary generation file."""
    if not path or isinstance(path, (bool, int)):
        return
    try:
        os.remove(path)
    except (OSError, TypeError):
        pass


def stage_previous_generation(path: str, *, backup_path: str | None = None) -> str:
    """Snapshot the currently certified ``path`` at ``backup_path`` (default ``.previous``)."""
    target_backup = backup_path or previous_generation_path(path)
    atomic_snapshot_file(path, target_backup)
    return target_backup


def restore_previous_generation(path: str, *, backup_path: str | None = None) -> None:
    """Durably restore ``backup_path`` (default ``.previous``) onto ``path``."""
    durable_replace_file(backup_path or previous_generation_path(path), path)


def write_meta_json(path: str, payload: dict[str, Any]) -> None:
    """Atomically write ``payload`` to ``<path>.meta``."""
    atomic_write_json(meta_file_path(path), payload)


def commit_file_and_meta(
    path: str,
    write_data_fn: Callable[[], Any],
    write_meta_fn: Callable[[], bool],
    *,
    backup_path: str | None = None,
    had_prior_file: bool | None = None,
    on_rollback_error: Callable[[OSError], None] | None = None,
) -> bool:
    """Transactionally replace ``path`` and commit its ``.meta`` manifest.

    The currently certified generation of ``path`` is staged at ``<path>.previous``
    before ``write_data_fn`` replaces ``path``. ``write_meta_fn`` is the single
    atomic commit point: if it returns ``False`` or raises after ``path`` was
    replaced, ``<path>.previous`` is durably restored (or the newly created
    uncertified ``path`` is removed when no prior generation existed).
    """
    target_path = os.fspath(path)
    if had_prior_file is None:
        had_prior_file = os.path.exists(target_path)
        if had_prior_file and backup_path is None:
            backup_path = stage_previous_generation(target_path)
        elif not had_prior_file:
            discard_generation_file(previous_generation_path(target_path))
    else:
        had_prior_file = _parse_bool_flag(had_prior_file, False)
    data_replaced = False
    try:
        write_data_fn()
        data_replaced = True
        if not _parse_bool_flag(write_meta_fn(), False):
            if backup_path is not None and os.path.exists(backup_path):
                restore_previous_generation(target_path, backup_path=backup_path)
            elif not had_prior_file:
                discard_generation_file(target_path)
            return False
        if backup_path is not None:
            discard_generation_file(backup_path)
        return True
    except BaseException:
        if data_replaced:
            if backup_path is not None and os.path.exists(backup_path):
                try:
                    restore_previous_generation(target_path, backup_path=backup_path)
                except OSError as rollback_error:
                    if on_rollback_error is not None:
                        on_rollback_error(rollback_error)
            elif not had_prior_file:
                discard_generation_file(target_path)
        raise
    finally:
        if backup_path is not None and os.path.exists(backup_path):
            discard_generation_file(backup_path)


_MISSING = object()


def read_recovering_previous(
    path: str,
    read_fn: Callable[[str], T],
    *,
    backup_path: str | None = None,
    error_mapper: Callable[[BaseException], BaseException] | None = None,
    default: Any = _MISSING,
) -> T:
    """Read and validate ``path`` via ``read_fn``, recovering ``.previous`` if needed.

    If ``read_fn(path)`` succeeds, any leftover ``.previous`` file is discarded.
    If ``read_fn(path)`` fails and a valid ``.previous`` generation exists,
    ``.previous`` is validated with ``read_fn(backup_path)``, durably promoted
    onto ``path``, and re-read from ``path``.
    """
    target_path = os.fspath(path)
    target_backup = backup_path or previous_generation_path(target_path)
    if read_fn is json.loads:
        def _invoke(candidate_path: str) -> Any:
            with open(candidate_path, "r", encoding="utf-8") as handle:
                return json.loads(handle.read())
    else:
        _invoke = read_fn
    try:
        result = _invoke(target_path)
        if os.path.exists(target_backup):
            discard_generation_file(target_backup)
        return result
    except Exception as primary_exc:
        if not os.path.exists(target_backup):
            if default is not _MISSING:
                return default
            raise
        try:
            _invoke(target_backup)
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

