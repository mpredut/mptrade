"""Shared durable file primitives for runtime and financial state.

Writers use a unique temporary file in the destination directory, flush file data,
atomically replace the destination, and clean up on failure. Locking and failure
policy remain the caller's responsibility because caches, live financial state,
and observational state intentionally have different concurrency semantics.
"""
from __future__ import annotations

import contextlib
import csv
import glob
import gzip
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Iterator
from typing import Any, TypeVar

from botcore import _parse_bool_flag, exact_nonneg_int, exact_pos_int, fetch_public_json, safe_float
from cachepaths import resolve_cachedb_filepath

T = TypeVar("T")


def _exact_nonneg_int(val: Any) -> int | None:
    return exact_nonneg_int(val, default=None)


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
    try:
        if path is None or isinstance(path, bool):
            raise TypeError("path must be a valid filesystem path")
        target_path = os.fspath(path)
        if isinstance(target_path, str) and not target_path.strip():
            raise ValueError("path must not be empty")
        target_path = str(resolve_cachedb_filepath(target_path))
    except (TypeError, ValueError) as exc:
        if fail_closed:
            raise StateReadError(f"invalid {label} state in {path}: {exc}") from exc
        return default_factory()
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
def _atomic_stream_writer(path: str, mode: str, **open_kwargs: Any) -> Iterator[Any]:
    destination = os.path.abspath(str(resolve_cachedb_filepath(os.fspath(path))))
    directory = os.path.dirname(destination)
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=directory, prefix=f".{os.path.basename(destination)}.", suffix=".tmp",
    )
    handle = os.fdopen(fd, mode, **open_kwargs)
    try:
        yield handle
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(temporary, destination)
        _fsync_parent_directory(destination)
    except BaseException:
        handle.close()
        discard_generation_file(temporary)
        raise


def atomic_text_writer(path: str, *, encoding: str = "utf-8",
                       newline: str | None = None) -> contextlib.AbstractContextManager[Any]:
    """Yield a text handle and atomically publish it only after a successful write."""
    return _atomic_stream_writer(path, "w", encoding=encoding, newline=newline)


def atomic_write_text(path: str, content: str, *, encoding: str = "utf-8",
                      newline: str | None = None) -> None:
    """Write text ``content`` to ``path`` atomically through ``atomic_text_writer``."""
    with atomic_text_writer(path, encoding=encoding, newline=newline) as handle:
        handle.write(str(content))


def atomic_write_json(path: str, value: Any, *, indent: int | None = None,
                      sort_keys: bool = False, separators=None,
                      ensure_ascii: bool = True,
                      allow_nan: bool = False,
                      default: Any = None) -> None:
    """Serialize JSON through ``atomic_text_writer``."""
    dump_kwargs = {"default": default} if default is not None else {}
    with atomic_text_writer(path) as handle:
        json.dump(
            value, handle, indent=_exact_nonneg_int(indent),
            sort_keys=_parse_bool_flag(sort_keys, False),
            separators=separators,
            ensure_ascii=_parse_bool_flag(ensure_ascii, True),
            allow_nan=_parse_bool_flag(allow_nan, False),
            **dump_kwargs,
        )


def _rollback_generation(
    target_path: str,
    backup_path: str | None,
    had_prior: bool,
    *,
    on_rollback_error: Callable[[OSError], None] | None = None,
    suppress_error: bool = False,
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
                    sort_keys: bool = False, separators=None,
                    ensure_ascii: bool = True,
                    allow_nan: bool = False,
                    default: Any = None,
                    write_fn: Callable[..., Any] | None = None) -> Any:
    """Transactionally write JSON state while staging and cleaning up ``<path>.previous``."""
    target_path = str(resolve_cachedb_filepath(os.fspath(path)))
    backup_path = None
    had_prior = os.path.exists(target_path)
    if had_prior:
        try:
            backup_path = stage_previous_generation(target_path)
        except OSError:
            backup_path = None
    try:
        if write_fn is None:
            res = atomic_write_json(
                target_path, value, indent=indent, sort_keys=sort_keys,
                separators=separators, ensure_ascii=ensure_ascii,
                allow_nan=allow_nan, default=default,
            )
        else:
            res = write_fn(target_path, value, **({"indent": indent} if indent is not None else {}))
        if backup_path is not None:
            discard_generation_file(backup_path)
        return res
    except BaseException:
        _rollback_generation(target_path, backup_path, had_prior, suppress_error=True)
        raise


def atomic_binary_writer(path: str) -> contextlib.AbstractContextManager[Any]:
    """Yield a binary handle and atomically publish it only after a durable write."""
    return _atomic_stream_writer(path, "wb")


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
    destination = os.path.abspath(str(resolve_cachedb_filepath(os.fspath(path))))
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
    return exact_pos_int(chunk_size, default=default)


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
    limit: int | None = None,
    *,
    chunk_size: int = 65536,
    max_bytes: int | None = None,
) -> list[bytes]:
    """Read up to the last ``limit`` lines and/or ``max_bytes`` from ``path`` with bounded memory."""
    if not path or isinstance(path, (bool, int, float)):
        return []
    try:
        resolved = str(resolve_cachedb_filepath(os.fspath(path)))
    except TypeError:
        return []
    if not os.path.isfile(resolved):
        return []
    limit_i = exact_pos_int(limit, default=None) if limit is not None else None
    if limit is not None and limit_i is None:
        return []
    max_b = exact_pos_int(max_bytes, default=None) if max_bytes is not None else None
    if max_bytes is not None and max_b is None:
        return []
    if limit_i is None and max_b is None:
        return []
    eff_chunk = max_b if limit_i is None else _positive_chunk_size(chunk_size, 65536)
    chunks: list[bytes] = []
    newline_count = 0
    try:
        with open(resolved, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            position = handle.tell()
            min_pos = max(0, position - max_b) if max_b is not None else 0
            while position > min_pos:
                step = min(eff_chunk, position - min_pos)
                position -= step
                handle.seek(position)
                chunk = handle.read(step)
                chunks.append(chunk)
                newline_count += chunk.count(b"\n")
                if limit_i is not None and newline_count > limit_i and position > min_pos:
                    joined = b"".join(reversed(chunks))
                    complete = joined.split(b"\n", 1)[1] if b"\n" in joined else b""
                    non_empty = [line for line in complete.splitlines() if line.strip()]
                    if len(non_empty) >= limit_i:
                        return non_empty[-limit_i:]
                    newline_count = len(non_empty)
    except OSError:
        return []
    joined = b"".join(reversed(chunks))
    if position > 0 and b"\n" in joined:
        joined = joined.split(b"\n", 1)[1]
    lines = [line for line in joined.splitlines() if line.strip()]
    return lines[-limit_i:] if limit_i is not None else lines


def read_tail_jsonl_records(
    path: str,
    limit: int | None = 50,
    *,
    chunk_size: int = 65536,
    max_bytes: int | None = None,
) -> list[Any]:
    """Read and parse JSON lines from the tail of a JSONL file."""
    out: list[Any] = []
    for raw_bytes in read_tail_lines(
        path, limit, chunk_size=chunk_size, max_bytes=max_bytes
    ):
        try:
            out.append(json.loads(raw_bytes.decode("utf-8", errors="replace")))
        except ValueError:
            continue
    return out


def read_tail_jsonl_dicts(
    path: str,
    limit: int | None = 50,
    *,
    chunk_size: int = 65536,
    max_bytes: int | None = None,
) -> list[dict[str, Any]]:
    """Read and parse up to the last ``limit`` JSON object lines from a JSONL file."""
    return [
        rec
        for rec in read_tail_jsonl_records(
            path, limit=limit, chunk_size=chunk_size, max_bytes=max_bytes
        )
        if isinstance(rec, dict)
    ]


def read_last_jsonl_dict(
    path: str,
    tail_bytes: int = 4096,
) -> dict[str, Any] | None:
    """Read and parse the last JSON dict from a JSONL file using bounded tail reads."""
    items = read_tail_jsonl_dicts(path, limit=1, chunk_size=_positive_chunk_size(tail_bytes, 4096))
    return items[-1] if items else None


def truncate_file_bytes(path: str, size: int) -> None:
    """Durably truncate ``path`` back to ``size`` bytes."""
    size_i = _exact_nonneg_int(size)
    if size_i is None:
        raise ValueError("size must be a non-negative integer")
    with open(path, "r+b") as handle:
        handle.truncate(size_i)
        handle.flush()
        os.fsync(handle.fileno())


def prune_rotated_archives(
    base_filename: str,
    keep_count: int,
) -> list[str]:
    """Retain the newest ``keep_count`` ``<base_filename>.*.archive*`` files."""
    if not base_filename or isinstance(base_filename, (bool, int, float)):
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
    keep = _exact_nonneg_int(keep_count) or 0
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


def discard_generation_file(
    path: str | None,
    *,
    include_previous: bool = False,
    include_meta: bool = False,
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
    target_path = str(resolve_cachedb_filepath(os.fspath(path)))
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


_MISSING = object()


def _read_json_file_strict(path: str) -> Any:
    target = str(resolve_cachedb_filepath(os.fspath(path)))
    with open(target, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_json_dict_strict(path: str) -> dict[str, Any]:
    payload = _read_json_file_strict(path)
    if not isinstance(payload, dict):
        raise ValueError("JSON root must be a dict")
    return payload


def _read_with_policy(
    path: str,
    reader_fn: Callable[[str], T],
    *,
    default: Any = _MISSING,
    backup_path: str | None = None,
    recover_previous: bool = False,
) -> T | Any:
    if recover_previous:
        return read_recovering_previous(
            path, reader_fn, backup_path=backup_path, default=default
        )
    try:
        if path is None or isinstance(path, (bool, int, float)):
            raise TypeError("path must be a valid filesystem path")
        return reader_fn(path)
    except Exception:
        if default is not _MISSING:
            return default
        raise


def read_json_file(
    path: str,
    *,
    default: Any = _MISSING,
    backup_path: str | None = None,
    recover_previous: bool = False,
) -> Any:
    """Read and parse any UTF-8 JSON file, with optional ``.previous`` recovery or fallback ``default``."""
    return _read_with_policy(
        path, _read_json_file_strict, default=default, backup_path=backup_path, recover_previous=recover_previous
    )


def read_json_dict(
    path: str,
    *,
    default: Any = _MISSING,
    backup_path: str | None = None,
    recover_previous: bool = False,
) -> dict[str, Any] | Any:
    """Read and parse a UTF-8 JSON file whose root value must be a ``dict``."""
    return _read_with_policy(
        path, _read_json_dict_strict, default=default, backup_path=backup_path, recover_previous=recover_previous
    )


def read_jsonl_records(path: str, *, strict: bool = False) -> list[Any]:
    """Read JSON records from a UTF-8 JSONL file, optionally enforcing strict line parsing."""
    if not path or isinstance(path, (bool, int, float)):
        if strict:
            raise TypeError("path must be a valid filesystem path")
        return []
    records: list[Any] = []
    try:
        target = str(resolve_cachedb_filepath(os.fspath(path)))
        if not strict and not os.path.isfile(target):
            return []
        with open(target, "r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    records.append(json.loads(raw))
                except ValueError:
                    if strict:
                        raise
    except (OSError, TypeError):
        if strict:
            raise
        return []
    return records


def read_jsonl_dicts(path: str, *, strict: bool = False) -> list[dict[str, Any]]:
    """Read JSON object records (``dict``) from a UTF-8 JSONL file."""
    out: list[dict[str, Any]] = []
    for rec in read_jsonl_records(path, strict=strict):
        if isinstance(rec, dict):
            out.append(rec)
        elif strict:
            raise ValueError("JSONL line must be a dict")
    return out


def read_text_file(
    path: str,
    *,
    default: Any = _MISSING,
    encoding: str = "utf-8",
    errors: str = "strict",
    newline: str | None = None,
    backup_path: str | None = None,
    recover_previous: bool = False,
) -> str | Any:
    """Read text from ``path`` (symmetric to ``atomic_write_text``), with optional fallback ``default``."""
    def _read_strict(candidate_path: str) -> str:
        target = str(resolve_cachedb_filepath(os.fspath(candidate_path)))
        with open(target, "r", encoding=encoding, errors=errors, newline=newline) as handle:
            return handle.read()

    return _read_with_policy(
        path, _read_strict, default=default, backup_path=backup_path, recover_previous=recover_previous
    )


def read_text_lines(
    path: str,
    *,
    strip: bool = False,
    skip_empty: bool = False,
    keepends: bool = False,
    encoding: str = "utf-8",
    errors: str = "strict",
) -> list[str]:
    """Read lines from a text file, returning ``[]`` when missing or unreadable."""
    if not path or isinstance(path, (bool, int, float)):
        return []
    try:
        target = str(resolve_cachedb_filepath(os.fspath(path)))
        with open(target, "r", encoding=encoding, errors=errors) as handle:
            return [
                raw.strip() if strip else (raw if keepends else raw.rstrip("\r\n"))
                for raw in handle
                if not (skip_empty and not raw.strip())
            ]
    except (OSError, TypeError):
        return []


def read_bytes_file(
    path: str,
    *,
    default: Any = _MISSING,
    backup_path: str | None = None,
    recover_previous: bool = False,
) -> bytes | Any:
    """Read binary bytes from ``path`` (symmetric to ``atomic_write_bytes``), with optional fallback ``default``."""
    def _read_strict(candidate_path: str) -> bytes:
        target = str(resolve_cachedb_filepath(os.fspath(candidate_path)))
        with open(target, "rb") as handle:
            return handle.read()

    return _read_with_policy(
        path, _read_strict, default=default, backup_path=backup_path, recover_previous=recover_previous
    )


def read_csv_dicts(
    path: str,
    *,
    encoding: str = "utf-8",
    strict: bool = False,
    default: Any = _MISSING,
) -> list[dict[str, Any]]:
    """Read rows from a CSV file via ``csv.DictReader``."""
    def _read_strict(candidate_path: str) -> list[dict[str, Any]]:
        if not candidate_path:
            raise TypeError("path must be a valid filesystem path")
        target = str(resolve_cachedb_filepath(os.fspath(candidate_path)))
        with open(target, "r", encoding=encoding, newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    eff_default = default if default is not _MISSING else (_MISSING if strict else [])
    return _read_with_policy(path, _read_strict, default=eff_default)


def read_recovering_previous(
    path: str,
    read_fn: Callable[[str], T] = json.loads,
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
    try:
        if path is None or isinstance(path, (bool, int, float)):
            raise TypeError("path must be a valid filesystem path")
        target_path = str(resolve_cachedb_filepath(os.fspath(path)))
        target_backup = (
            str(resolve_cachedb_filepath(backup_path))
            if backup_path
            else previous_generation_path(target_path)
        )
    except Exception:
        if default is not _MISSING:
            return default
        raise
    if read_fn is json.loads:
        def _invoke(candidate_path: str) -> Any:
            with open(candidate_path, "r", encoding="utf-8") as handle:
                return json.loads(handle.read())
    elif read_fn is dict:
        _invoke = _read_json_dict_strict
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


_JSON_DICT_MEMO: dict[str, tuple[tuple, dict[str, Any]]] = {}
_ORIG_READ_RECOVERING_ID = id(read_recovering_previous)


def _stat_sig(path: str) -> tuple[int, int] | None:
    try:
        st = os.stat(path)
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def read_json_recovering(path: str, *, default: Any = None) -> Any:
    """Read and parse any UTF-8 JSON file with ``.previous`` recovery, returning ``default`` on error."""
    return read_json_file(path, default=default, recover_previous=True)


def read_json_dict_recovering(path: str, *, default: Any = None) -> dict[str, Any] | Any:
    """Read and parse a UTF-8 JSON dict file with ``.previous`` recovery, returning ``default`` on error."""
    return read_json_dict(path, default=default, recover_previous=True)


def read_json_dict_cached(
    path: str | os.PathLike[str],
    memo: dict[str, tuple[tuple, dict[str, Any]]] | None = None,
    *,
    max_entries: int = 32,
) -> dict[str, Any] | None:
    """Stat-memoized ``read_json_dict(..., default=None, recover_previous=True)`` checking both ``path`` and ``path.previous``."""
    if not path or isinstance(path, (bool, int, float, bytes)):
        return None
    try:
        resolved = str(resolve_cachedb_filepath(os.fspath(path)))
    except Exception:
        return None
    backup_path = previous_generation_path(resolved)
    st_main, st_bak = _stat_sig(resolved), _stat_sig(backup_path)
    if st_main is None and st_bak is None and id(read_recovering_previous) == _ORIG_READ_RECOVERING_ID:
        return None

    cache_store = _JSON_DICT_MEMO if memo is None else memo
    sig = (st_main, st_bak, id(read_recovering_previous), id(json.load))
    cached = cache_store.get(resolved)
    if cached is not None and cached[0] == sig:
        return cached[1]

    res = read_json_dict(resolved, default=None, backup_path=backup_path, recover_previous=True)
    if isinstance(res, dict):
        if st_bak is not None:
            sig = (_stat_sig(resolved), _stat_sig(backup_path), id(read_recovering_previous), id(json.load))
        if len(cache_store) > max_entries:
            cache_store.clear()
        cache_store[resolved] = (sig, res)
        return res
    return None

