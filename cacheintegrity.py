"""Cryptographic digest, semantic deduplication, and versioned manifest API for account caches."""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
import hashlib
import json
import os
from typing import Any, Callable, Iterable

import binance_cache_health as account_cache_health
from botcore import exact_int, safe_pos_int
from state_io import read_json_dict

CACHE_META_SCHEMA_VERSION = 2


def normalize_timestamp_map(
    mapping: Any,
    label: str = "timestamp map",
    *,
    error_message: str | None = None,
) -> dict[str, int]:
    """Validate and copy a ``{symbol: non_negative_int_ms}`` map."""
    msg = error_message or f"invalid {label}"
    if not isinstance(mapping, dict):
        raise ValueError(msg)
    normalized: dict[str, int] = {}
    for symbol, value in mapping.items():
        if (
            not isinstance(symbol, str)
            or not symbol.strip()
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise ValueError(error_message or f"invalid {label} entry")
        normalized[symbol] = value
    return normalized


def snapshot_data_version(items: dict[str, Any], fetchtime: dict[str, Any] | None = None) -> str:
    """Compute a deterministic Blake2s-128 content version for an Order items snapshot."""
    canonical = json.dumps(
        {"items": items},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "snapshot:" + hashlib.blake2s(canonical, digest_size=16).hexdigest()


def trade_data_version(stream_id: str, revision: int) -> str:
    """Format a monotonic ``<stream_id>:<revision>`` version string for Trade JSONL."""
    if (
        not isinstance(stream_id, str)
        or not stream_id.strip()
        or isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision < 0
    ):
        raise ValueError("invalid Trade stream revision")
    return f"{stream_id}:{revision}"


def trade_records_digest(
    records: Iterable[tuple[str, dict[str, Any]]],
    initial_digest: str = "",
) -> str:
    """Extend a deterministic Blake2s-128 digest chain with canonical Trade rows."""
    if initial_digest:
        if (
            not isinstance(initial_digest, str)
            or not initial_digest.startswith("chain:")
        ):
            raise ValueError("invalid Trade content digest")
        try:
            state = bytes.fromhex(initial_digest.removeprefix("chain:"))
        except ValueError as exc:
            raise ValueError("invalid Trade content digest") from exc
        if len(state) != 16:
            raise ValueError("invalid Trade content digest")
    else:
        state = bytes(16)
    for symbol, item in records:
        row = json.dumps(
            {"s": symbol, "i": item},
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
        state = hashlib.blake2s(state + row, digest_size=16).digest()
    return "chain:" + state.hex()


def _strict_int_field(val: Any) -> int:
    if not isinstance(val, (int, float, str)):
        raise ValueError("invalid Trade row field type")
    if isinstance(val, str):
        s = val.strip()
        if s.endswith(".0"):
            s = s[:-2]
        digits = s[1:] if s.startswith(("+", "-")) else s
        if not digits or not digits.isdigit():
            raise ValueError("invalid Trade row field type")
        val = s
    parsed = exact_int(val, default=None)
    if parsed is None:
        raise ValueError("invalid Trade row field type")
    return parsed


def trade_semantic_signature(item: dict[str, Any]) -> tuple[Any, ...]:
    """Normalize the immutable financial fields shared by REST and WebSocket."""
    if not isinstance(item, dict):
        raise ValueError("invalid Trade row field type")
    try:
        raw_sym = item["symbol"]
        raw_id = item["id"]
        raw_order_id = item["orderId"]
        raw_time = item["time"]
        raw_buyer = item["isBuyer"]
        raw_price = item["price"]
        raw_qty = item["qty"]
    except KeyError as exc:
        raise ValueError("invalid Trade row field type") from exc
    if (
        not isinstance(raw_sym, str)
        or not raw_sym.strip()
        or not isinstance(raw_buyer, bool)
        or isinstance(raw_price, bool)
        or isinstance(raw_qty, bool)
    ):
        raise ValueError("invalid Trade row field type")
    trade_id_int = _strict_int_field(raw_id)
    order_id_int = _strict_int_field(raw_order_id)
    time_int = _strict_int_field(raw_time)
    try:
        price_dec = Decimal(str(raw_price)).normalize()
        qty_dec = Decimal(str(raw_qty)).normalize()
    except ArithmeticError as exc:
        raise ValueError("invalid Trade price or qty") from exc
    if not (price_dec.is_finite() and qty_dec.is_finite()):
        raise ValueError("non-finite Trade price or qty")
    return (
        raw_sym.strip().upper(),
        trade_id_int,
        order_id_int,
        price_dec,
        qty_dec,
        time_int,
        raw_buyer,
    )


def dedupe_trade_items_by_id(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse equivalent Trade IDs and reject conflicting immutable fills."""
    seen: dict[str, tuple[Any, ...]] = {}
    deduped: list[dict[str, Any]] = []
    for item in items:
        signature = trade_semantic_signature(item)
        trade_id = str(signature[1])
        if trade_id in seen:
            if seen[trade_id] != signature:
                raise ValueError(
                    f"conflicting Trade rows for immutable ID {trade_id}"
                )
            continue
        seen[trade_id] = signature
        deduped.append(item)
    return deduped


def counts_for_records(records: Iterable[tuple[str, Any]]) -> dict[str, int]:
    """Count JSONL ``(symbol, item)`` records per symbol."""
    counts: defaultdict[str, int] = defaultdict(int)
    if records is None or isinstance(records, (str, bytes, dict, bool)):
        return {}
    try:
        iterator = iter(records)
    except TypeError:
        return {}
    for entry in iterator:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            continue
        symbol, _item = entry
        if isinstance(symbol, str) and symbol:
            counts[symbol] += 1
    return dict(counts)


def nonzero_counts(counts: dict[str, int]) -> dict[str, int]:
    """Filter ``counts`` to symbols with at least one record."""
    if not isinstance(counts, dict):
        return {}
    out: dict[str, int] = {}
    for symbol, count in counts.items():
        if not isinstance(symbol, str) or not symbol:
            continue
        c_int = safe_pos_int(count, default=0)
        if c_int > 0:
            out[symbol] = c_int
    return out


def build_account_meta_snapshot(
    *,
    data_version: str,
    fetchtime: dict[str, int],
    counts: dict[str, int],
    rest_sync_cursor: dict[str, int] | None = None,
    stream_id: str = "",
    revision: int = 0,
    committed_bytes: int | None = None,
    content_digest: str | None = None,
    schema_version: int = CACHE_META_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Build a schema v2 ``.meta`` manifest dictionary for an account cache."""
    normalized_fetchtime = normalize_timestamp_map(
        fetchtime,
        error_message="invalid cache manifest fetchtime",
    )
    normalized_rest_cursor = (
        dict(normalized_fetchtime)
        if rest_sync_cursor is None
        else normalize_timestamp_map(
            rest_sync_cursor,
            error_message="invalid cache manifest rest_sync_cursor",
        )
    )
    payload: dict[str, Any] = {
        "schema_version": schema_version,
        "data_version": data_version,
        "fetchtime": normalized_fetchtime,
        "rest_sync_cursor": normalized_rest_cursor,
        "counts": dict(counts),
    }
    if committed_bytes is not None:
        if (
            not isinstance(stream_id, str)
            or not stream_id.strip()
            or isinstance(revision, bool)
            or not isinstance(revision, int)
            or isinstance(committed_bytes, bool)
            or not isinstance(committed_bytes, int)
            or revision < 0
            or committed_bytes < 0
        ):
            raise ValueError("invalid Trade stream metadata")
        payload.update({
            "stream_id": stream_id,
            "revision": revision,
            "committed_bytes": committed_bytes,
            "content_digest": content_digest,
        })
    return payload


def load_versioned_account_meta_file(
    meta_path: str,
    *,
    is_trade_cache: bool,
    reader_reason: str,
    optional: bool = False,
    required_version: str | None = None,
    schema_version: int = CACHE_META_SCHEMA_VERSION,
) -> dict[str, Any] | None:
    """Load and strictly validate a versioned account-cache ``.meta`` manifest."""
    try:
        metadata = read_json_dict(meta_path)
    except (OSError, ValueError, TypeError) as exc:
        if optional:
            return None
        raise account_cache_health.AccountCacheNotReady(reader_reason) from exc

    if metadata.get("schema_version") != schema_version:
        if optional:
            return None
        raise account_cache_health.AccountCacheNotReady(reader_reason)
    if is_trade_cache:
        content_digest = metadata.get("content_digest")
        try:
            if (
                not isinstance(content_digest, str)
                or not content_digest.startswith("chain:")
            ):
                raise ValueError("missing Trade content digest")
            trade_records_digest([], content_digest)
        except ValueError as exc:
            if optional:
                return None
            raise account_cache_health.AccountCacheNotReady(reader_reason) from exc
    try:
        version = metadata["data_version"]
        fetchtime = metadata["fetchtime"]
        counts = metadata["counts"]
        if not isinstance(version, str) or not version:
            raise ValueError("missing data version")
        if required_version is not None and version != required_version:
            raise ValueError("cache marker and manifest versions differ")
        if not isinstance(fetchtime, dict) or not isinstance(counts, dict):
            raise ValueError("invalid cache manifest maps")
        normalized_fetchtime = normalize_timestamp_map(
            fetchtime,
            error_message="invalid cache manifest fetchtime",
        )
        raw_rest_cursor = metadata.get("rest_sync_cursor")
        if raw_rest_cursor is None:
            normalized_rest_cursor = dict(normalized_fetchtime)
        else:
            normalized_rest_cursor = normalize_timestamp_map(
                raw_rest_cursor,
                error_message="invalid cache manifest rest_sync_cursor",
            )
        normalized_counts = normalize_timestamp_map(
            counts,
            error_message="invalid cache manifest count",
        )
        result: dict[str, Any] = {
            "schema_version": schema_version,
            "data_version": version,
            "fetchtime": normalized_fetchtime,
            "rest_sync_cursor": normalized_rest_cursor,
            "counts": normalized_counts,
        }
        if is_trade_cache:
            stream_id = metadata["stream_id"]
            revision = metadata["revision"]
            committed_bytes = metadata["committed_bytes"]
            content_digest = metadata["content_digest"]
            if (
                not isinstance(stream_id, str)
                or not stream_id
                or isinstance(revision, bool)
                or not isinstance(revision, int)
                or revision <= 0
                or isinstance(committed_bytes, bool)
                or not isinstance(committed_bytes, int)
                or committed_bytes < 0
                or trade_data_version(stream_id, revision) != version
            ):
                raise ValueError("invalid Trade stream metadata")
            result.update({
                "stream_id": stream_id,
                "revision": revision,
                "committed_bytes": committed_bytes,
                "content_digest": content_digest,
            })
        return result
    except (KeyError, TypeError, ValueError) as exc:
        raise account_cache_health.AccountCacheNotReady(reader_reason) from exc


def read_trade_records_range_strict(
    path: str,
    start_byte: int,
    end_byte: int,
    *,
    is_valid_item_fn: Callable[[dict[str, Any]], bool],
    reader_reason: str,
) -> list[tuple[str, dict[str, Any]]]:
    """Read and validate Trade JSONL records strictly from ``[start_byte, end_byte)``."""
    try:
        if (
            isinstance(start_byte, bool)
            or isinstance(end_byte, bool)
            or not isinstance(start_byte, int)
            or not isinstance(end_byte, int)
            or start_byte < 0
            or end_byte < start_byte
        ):
            raise ValueError("Trade committed boundary moved backwards")
        with open(path, "rb") as handle:
            file_size = os.fstat(handle.fileno()).st_size
            if file_size < end_byte:
                raise ValueError("Trade cache is shorter than its manifest")
            if start_byte > 0:
                handle.seek(start_byte - 1)
                if handle.read(1) != b"\n":
                    raise ValueError("Trade suffix does not start at a newline boundary")
            else:
                handle.seek(0)
            raw = handle.read(end_byte - start_byte)
        if raw and not raw.endswith(b"\n"):
            raise ValueError("Trade committed data ends with a partial row")
        records: list[tuple[str, dict[str, Any]]] = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            record = json.loads(line.decode("utf-8"))
            if (
                not isinstance(record, dict)
                or not isinstance(record.get("s"), str)
                or not isinstance(record.get("i"), dict)
                or not is_valid_item_fn(record["i"])
                or str(record["i"]["symbol"]).upper() != str(record["s"]).upper()
            ):
                raise ValueError("invalid Trade cache row")
            records.append((record["s"], record["i"]))
        return records
    except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise account_cache_health.AccountCacheNotReady(reader_reason) from exc
