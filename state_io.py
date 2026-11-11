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
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Iterator
from typing import Any, TypeVar
try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore[assignment]

from botcore import (
    _parse_bool_flag,
    exact_nonneg_int,
    exact_pos_int,
)
from cachepaths import STALE_STAGING_MAX_AGE_SEC, resolve_cachedb_filepath

T = TypeVar("T")
_MISSING = object()
JSONDecodeError = json.JSONDecodeError


def _resolve_target_path(path: Any, *, abspath: bool = False) -> str:
    """Validate ``path`` and resolve its canonical ``cachedb/<subdomain>/`` location."""
    if path is None or isinstance(path, (bool, int, float, bytes)):
        raise TypeError("path must be a valid filesystem path")
    raw = os.fspath(path)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("path must not be empty")
    resolved = str(resolve_cachedb_filepath(raw))
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


def durable_replace_file(source: str, destination: str) -> None:
    """Atomically replace ``destination`` and durably publish the new entry."""
    source = os.path.abspath(os.fspath(source))
    destination = os.path.abspath(os.fspath(destination))
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(source, "rb") as handle:
        os.fsync(handle.fileno())
    os.replace(source, destination)
    if source != destination and os.path.exists(source):
        os.remove(source)
    _fsync_parent_directory(destination)


def atomic_snapshot_file(source: str, destination: str) -> None:
    """Atomically preserve one immutable file generation at ``destination``."""
    source = os.path.abspath(os.fspath(source))
    destination = os.path.abspath(os.fspath(destination))
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
            with open(staged, "rb") as handle:
                os.fsync(handle.fileno())
        durable_replace_file(staged, destination)
    finally:
        shutil.rmtree(staging_directory, ignore_errors=True)
        cleanup_stale_staging_siblings(
            directory, prefix, exclude_name=os.path.basename(staging_directory), dirs_only=True
        )


def _reject_json_constant(constant: str) -> Any:
    raise ValueError(f"non-standard JSON constant {constant!r}")


def load_json_state(path: str, *, default_factory, fail_closed: bool,
                    label: str, root_type: type = dict):
    """Load JSON state with an explicit live-versus-observational failure policy."""
    fail_closed = _parse_bool_flag(fail_closed, True)
    try:
        target_path = _resolve_target_path(path)
    except (TypeError, ValueError) as exc:
        if fail_closed:
            raise StateReadError(f"invalid {label} state in {path}: {exc}") from exc
        return default_factory()
    backup_path = previous_generation_path(target_path)
    if not os.path.exists(target_path) and not os.path.exists(backup_path):
        return default_factory()

    def _read_and_validate(candidate_path: str):
        with open(candidate_path, "r", encoding="utf-8") as handle:
            value = json.load(handle, parse_constant=_reject_json_constant if fail_closed else None)
        if not isinstance(value, root_type):
            raise ValueError(f"expected {root_type.__name__}, got {type(value).__name__}")
        return value

    try:
        return read_recovering_previous(target_path, _read_and_validate, backup_path=backup_path)
    except (OSError, TypeError, ValueError) as exc:
        if fail_closed:
            raise StateReadError(f"invalid {label} state in {path}: {exc}") from exc
        return default_factory()


@contextlib.contextmanager
def _atomic_stream_writer(
    path: str, mode: str, *, preserve_mode: bool = False, **open_kwargs: Any,
) -> Iterator[Any]:
    destination = _resolve_target_path(path, abspath=True)
    directory = os.path.dirname(destination)
    os.makedirs(directory, exist_ok=True)
    prefix = f".{os.path.basename(destination)}."
    fd, temporary = tempfile.mkstemp(dir=directory, prefix=prefix, suffix=".tmp")
    handle = os.fdopen(fd, mode, **open_kwargs)
    try:
        yield handle
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        if preserve_mode and os.path.exists(destination):
            shutil.copymode(destination, temporary)
        os.replace(temporary, destination)
        _fsync_parent_directory(destination)
    except BaseException:
        handle.close()
        discard_generation_file(temporary)
        raise
    finally:
        cleanup_stale_staging_siblings(
            directory, prefix, exclude_name=os.path.basename(temporary), suffix=".tmp"
        )


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
                      sort_keys: bool = False, separators=None, ensure_ascii: bool = True,
                      allow_nan: bool = False, default: Any = None) -> None:
    """Serialize JSON through ``atomic_text_writer``."""
    kwargs = _json_dump_kwargs(
        indent=indent, sort_keys=sort_keys, separators=separators,
        ensure_ascii=ensure_ascii, allow_nan=allow_nan, default=default,
    )
    with atomic_text_writer(path) as handle:
        json.dump(value, handle, **kwargs)


def atomic_write_jsonl(path: str, items: Any, *, ensure_ascii: bool = True,
                       default: Any = None) -> None:
    """Serialize an iterable of JSON-encodable objects to a JSONL file through ``atomic_text_writer``."""
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
                    write_fn: Callable[..., Any] | None = None) -> Any:
    """Transactionally write JSON state while staging and cleaning up ``<path>.previous``."""
    target_path = _resolve_target_path(path)
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
    destination = _resolve_target_path(path, abspath=True)
    directory = os.path.dirname(destination)
    if directory:
        os.makedirs(directory, exist_ok=True)
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
    destination = _resolve_target_path(path, abspath=True)
    directory = os.path.dirname(destination)
    os.makedirs(directory, exist_ok=True)
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


def read_tail_lines(
    path: str, limit: int | None = None, *, chunk_size: int = 65536, max_bytes: int | None = None,
) -> list[bytes]:
    """Read up to the last ``limit`` lines and/or ``max_bytes`` from ``path`` with bounded memory."""
    try:
        resolved = _resolve_target_path(path)
    except (TypeError, ValueError):
        return []
    if not os.path.isfile(resolved):
        return []
    limit_i = exact_pos_int(limit, default=None) if limit is not None else None
    max_b = exact_pos_int(max_bytes, default=None) if max_bytes is not None else None
    if (limit is not None and limit_i is None) or (max_bytes is not None and max_b is None) or (limit_i is None and max_b is None):
        return []
    eff_chunk = max_b if limit_i is None else exact_pos_int(chunk_size, default=65536)
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
                    non_empty = _extract_tail_non_empty(chunks, drop_partial_head=True)
                    if len(non_empty) >= limit_i:
                        return non_empty[-limit_i:]
                    newline_count = len(non_empty)
    except OSError:
        return []
    lines = _extract_tail_non_empty(chunks, drop_partial_head=(position > 0 and b"\n" in b"".join(reversed(chunks))))
    return lines[-limit_i:] if limit_i is not None else lines


def read_tail_jsonl_records(
    path: str, limit: int | None = 50, *, chunk_size: int = 65536, max_bytes: int | None = None,
) -> list[Any]:
    """Read and parse JSON lines from the tail of a JSONL file."""
    out: list[Any] = []
    for raw_bytes in read_tail_lines(path, limit, chunk_size=chunk_size, max_bytes=max_bytes):
        try:
            out.append(json.loads(raw_bytes.decode("utf-8", errors="replace")))
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
    with open(_resolve_target_path(path), "r+b") as handle:
        handle.truncate(size_i)
        handle.flush()
        os.fsync(handle.fileno())


def format_jsonl_bucket_line(symbol: str, item: Any) -> str:
    """Serialize one ``(symbol, item)`` pair as a compact JSONL line."""
    return json.dumps({"s": symbol, "i": item}, separators=(",", ":")) + "\n"


def canonical_json_key(item: Any) -> str:
    """Return a deterministic sort-keyed JSON string for deduplicating cache items."""
    return json.dumps(item, sort_keys=True)


def parse_json_payload(raw: str | bytes, *, default: Any = _MISSING) -> Any:
    """Parse a JSON string or bytes payload, returning ``default`` (if provided) on decode failure."""
    try:
        text = raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray, memoryview)) else raw
        return json.loads(text)
    except Exception:
        if default is not _MISSING:
            return default
        raise


def extract_json_object(text: Any) -> dict[str, Any] | None:
    """Extract a JSON dict from pure JSON, markdown code fences, or hybrid prose + trailing JSON."""
    if not isinstance(text, str):
        return None
    cleaned = text.strip()
    if not cleaned:
        return None
    if "```" in cleaned:
        fences = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
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
    target_path = _resolve_target_path(path)
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
            handle.flush()
            os.fsync(handle.fileno())
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
) -> dict[str, int]:
    """Durably append unpersisted ``(symbol, item)`` rows to ``path`` and return updated counts."""
    target_path = _resolve_target_path(path)
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
            handle.flush()
            os.fsync(handle.fileno())
        return candidate_counts
    except BaseException:
        if initial_size is not None:
            if on_rollback_fn is not None:
                on_rollback_fn(initial_size)
            else:
                with contextlib.suppress(Exception):
                    truncate_file_bytes(target_path, initial_size)
        raise


def read_jsonl_bucket_map(path: str) -> tuple[dict[str, list[Any]], bool]:
    """Read a JSONL file of ``{'s': symbol, 'i': item}`` records into ``(cache_map, had_invalid_line)``."""
    target_path = _resolve_target_path(path)
    cache: dict[str, list[Any]] = {}
    had_invalid_line = False
    with open(target_path, "r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
                if (
                    not isinstance(rec, dict)
                    or not isinstance(rec.get("s"), str)
                    or rec.get("i") is None
                ):
                    raise ValueError("invalid JSONL record")
                cache.setdefault(rec["s"], []).append(rec["i"])
            except Exception:
                had_invalid_line = True
                continue
    return cache, had_invalid_line


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
                rec = json.loads(line)
                keep = isinstance(rec, dict) and bool(keep_record_fn(rec))
            except (json.JSONDecodeError, TypeError, ValueError):
                keep = False
            yield (line if line.endswith("\n") else line + "\n"), keep


def read_tail_complete_lines_text(path: str, keep_bytes: int) -> str:
    """Read up to the last ``keep_bytes`` of ``path`` aligned to a complete line boundary."""
    target_path = _resolve_target_path(path)
    archive_size = os.path.getsize(target_path)
    eff_keep = max(1, int(keep_bytes))
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
    path: str, write_data_fn: Callable[[], Any], write_meta_fn: Callable[[], bool], *,
    backup_path: str | None = None, had_prior_file: bool | None = None,
    on_rollback_error: Callable[[OSError], None] | None = None,
) -> bool:
    """Transactionally replace ``path`` and commit its ``.meta`` manifest."""
    target_path = _resolve_target_path(path)
    if had_prior_file is None:
        had_prior_file = os.path.exists(target_path)
        if had_prior_file and backup_path is None:
            try:
                backup_path = stage_previous_generation(target_path)
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
    with open(_resolve_target_path(path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_json_dict_strict(path: str) -> dict[str, Any]:
    payload = _read_json_file_strict(path)
    if not isinstance(payload, dict):
        raise ValueError("JSON root must be a dict")
    return payload


def _read_with_policy(
    path: str, reader_fn: Callable[[str], T], *,
    default: Any = _MISSING, backup_path: str | None = None, recover_previous: bool = False,
) -> T | Any:
    if recover_previous:
        return read_recovering_previous(path, reader_fn, backup_path=backup_path, default=default)
    try:
        return reader_fn(_resolve_target_path(path))
    except Exception:
        if default is not _MISSING:
            return default
        raise


def read_json_file(
    path: str, *, default: Any = _MISSING, backup_path: str | None = None, recover_previous: bool = False,
) -> Any:
    """Read and parse any UTF-8 JSON file, with optional ``.previous`` recovery or fallback ``default``."""
    return _read_with_policy(
        path, _read_json_file_strict, default=default, backup_path=backup_path, recover_previous=recover_previous
    )


def read_json_dict(
    path: str, *, default: Any = _MISSING, backup_path: str | None = None, recover_previous: bool = False,
) -> dict[str, Any] | Any:
    """Read and parse a UTF-8 JSON file whose root value must be a ``dict``."""
    return _read_with_policy(
        path, _read_json_dict_strict, default=default, backup_path=backup_path, recover_previous=recover_previous
    )


def read_jsonl_records(path: str, *, strict: bool = False) -> list[Any]:
    """Read JSON records from a UTF-8 JSONL file, optionally enforcing strict line parsing."""
    try:
        target = _resolve_target_path(path)
    except (TypeError, ValueError):
        if strict:
            raise
        return []
    records: list[Any] = []
    try:
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
    path: str, *, default: Any = _MISSING, encoding: str = "utf-8", errors: str = "strict",
    newline: str | None = None, backup_path: str | None = None, recover_previous: bool = False,
) -> str | Any:
    """Read text from ``path`` (symmetric to ``atomic_write_text``), with optional fallback ``default``."""
    def _read_strict(candidate_path: str) -> str:
        with open(candidate_path, "r", encoding=encoding, errors=errors, newline=newline) as handle:
            return handle.read()

    return _read_with_policy(
        path, _read_strict, default=default, backup_path=backup_path, recover_previous=recover_previous
    )


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
    path: str, *, default: Any = _MISSING, backup_path: str | None = None, recover_previous: bool = False,
) -> bytes | Any:
    """Read binary bytes from ``path`` (symmetric to ``atomic_write_bytes``), with optional fallback ``default``."""
    def _read_strict(candidate_path: str) -> bytes:
        with open(candidate_path, "rb") as handle:
            return handle.read()

    return _read_with_policy(
        path, _read_strict, default=default, backup_path=backup_path, recover_previous=recover_previous
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
) -> T:
    """Read and validate ``path`` via ``read_fn``, recovering ``.previous`` if needed."""
    try:
        target_path = _resolve_target_path(path)
        target_backup = (
            _resolve_target_path(backup_path) if backup_path else previous_generation_path(target_path)
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
            try:
                is_staged = os.path.samefile(target_path, target_backup)
            except OSError:
                is_staged = False
            if not is_staged:
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
    *, max_entries: int = 32,
) -> dict[str, Any] | None:
    """Stat-memoized ``read_json_dict(..., default=None, recover_previous=True)`` checking both ``path`` and ``path.previous``."""
    try:
        resolved = _resolve_target_path(path)
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
