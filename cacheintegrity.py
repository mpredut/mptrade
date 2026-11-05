"""Cryptographic digest, semantic deduplication, and versioned manifest API for account caches."""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
import hashlib
import json
import os
from typing import Any, Callable, Iterable

import binance_cache_health as account_cache_health
from botcore import safe_pos_int, strict_ascii_int
from state_io import read_json_dict

CACHE_META_SCHEMA_VERSION = 2


def _is_strict_int(val: Any, min_val: int = 0) -> bool:
    return not isinstance(val, bool) and isinstance(val, int) and val >= min_val


def normalize_timestamp_map(
    mapping: Any, label: str = "timestamp map", *, error_message: str | None = None
) -> dict[str, int]:
    """Validate and copy a ``{symbol: non_negative_int_ms}`` map."""
    if not isinstance(mapping, dict):
        raise ValueError(error_message or f"invalid {label}")
    normalized: dict[str, int] = {}
    for symbol, value in mapping.items():
        if not isinstance(symbol, str) or not symbol.strip() or not _is_strict_int(value):
            raise ValueError(error_message or f"invalid {label} entry")
        normalized[symbol] = value
    return normalized


def snapshot_data_version(items: dict[str, Any], fetchtime: dict[str, Any] | None = None) -> str:
    """Compute a deterministic Blake2s-128 content version for an Order items snapshot."""
    canonical = json.dumps({"items": items}, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "snapshot:" + hashlib.blake2s(canonical, digest_size=16).hexdigest()


def trade_data_version(stream_id: str, revision: int) -> str:
    """Format a monotonic ``<stream_id>:<revision>`` version string for Trade JSONL."""
    if not isinstance(stream_id, str) or not stream_id.strip() or not _is_strict_int(revision):
        raise ValueError("invalid Trade stream revision")
    return f"{stream_id}:{revision}"


def trade_records_digest(records: Iterable[tuple[str, dict[str, Any]]], initial_digest: str = "") -> str:
    """Extend a deterministic Blake2s-128 digest chain with canonical Trade rows."""
    if initial_digest:
        if not isinstance(initial_digest, str) or not initial_digest.startswith("chain:"):
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
        row = json.dumps({"s": symbol, "i": item}, ensure_ascii=True, separators=(",", ":")).encode("utf-8") + b"\n"
        state = hashlib.blake2s(state + row, digest_size=16).digest()
    return "chain:" + state.hex()


def _strict_int_field(val: Any) -> int:
    parsed = strict_ascii_int(val, default=None, allow_float_str=True)
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
    return (raw_sym.strip().upper(), trade_id_int, order_id_int, price_dec, qty_dec, time_int, raw_buyer)


def dedupe_trade_items_by_id(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse equivalent Trade IDs and reject conflicting immutable fills."""
    seen: dict[str, tuple[Any, ...]] = {}
    deduped: list[dict[str, Any]] = []
    for item in items:
        signature = trade_semantic_signature(item)
        trade_id = str(signature[1])
        if trade_id in seen:
            if seen[trade_id] != signature:
                raise ValueError(f"conflicting Trade rows for immutable ID {trade_id}")
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
        if isinstance(entry, (tuple, list)) and len(entry) == 2 and isinstance(entry[0], str) and entry[0]:
            counts[entry[0]] += 1
    return dict(counts)


def nonzero_counts(counts: dict[str, int]) -> dict[str, int]:
    """Filter ``counts`` to symbols with at least one record."""
    if not isinstance(counts, dict):
        return {}
    return {
        symbol: c_int
        for symbol, count in counts.items()
        if isinstance(symbol, str) and symbol and (c_int := safe_pos_int(count, default=0)) > 0
    }


def _normalize_cursor_map(
    raw_cursor: Any,
    fallback_fetchtime: dict[str, int],
    label: str = "cache manifest rest_sync_cursor",
    *,
    error_message: str | None = None,
) -> dict[str, int]:
    if raw_cursor is None:
        return dict(fallback_fetchtime)
    return normalize_timestamp_map(raw_cursor, label, error_message=error_message)


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
    normalized_fetchtime = normalize_timestamp_map(fetchtime, error_message="invalid cache manifest fetchtime")
    normalized_rest_cursor = _normalize_cursor_map(
        rest_sync_cursor, normalized_fetchtime, error_message="invalid cache manifest rest_sync_cursor"
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
            or not _is_strict_int(revision)
            or not _is_strict_int(committed_bytes)
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
        if metadata.get("schema_version") != schema_version:
            raise ValueError("schema_version mismatch")
        if is_trade_cache:
            content_digest = metadata.get("content_digest")
            if not content_digest:
                raise ValueError("missing Trade content digest")
            trade_records_digest([], content_digest)
    except (OSError, ValueError, TypeError) as exc:
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
        normalized_fetchtime = normalize_timestamp_map(fetchtime, error_message="invalid cache manifest fetchtime")
        normalized_rest_cursor = _normalize_cursor_map(
            metadata.get("rest_sync_cursor"), normalized_fetchtime, error_message="invalid cache manifest rest_sync_cursor"
        )
        normalized_counts = normalize_timestamp_map(counts, error_message="invalid cache manifest count")
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
                or not _is_strict_int(revision, 1)
                or not _is_strict_int(committed_bytes, 0)
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
        if not _is_strict_int(start_byte) or not _is_strict_int(end_byte, start_byte):
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


def read_trade_snapshot_strict(
    path: str,
    metadata: dict[str, Any],
    *,
    is_valid_item_fn: Callable[[dict[str, Any]], bool],
    reader_reason: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Read and validate a complete committed Trade JSONL snapshot against its manifest."""
    records = read_trade_records_range_strict(
        path, 0, metadata["committed_bytes"], is_valid_item_fn=is_valid_item_fn, reader_reason=reader_reason
    )
    if trade_records_digest(records) != metadata["content_digest"]:
        raise account_cache_health.AccountCacheNotReady(reader_reason)
    cache: dict[str, list[dict[str, Any]]] = {}
    seen_ids: defaultdict[str, set[str]] = defaultdict(set)
    for symbol, item in records:
        normalized_symbol = str(symbol).upper()
        try:
            trade_id = str(_strict_int_field(item["id"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise account_cache_health.AccountCacheNotReady(reader_reason) from exc
        if trade_id in seen_ids[normalized_symbol]:
            raise account_cache_health.AccountCacheNotReady(reader_reason)
        seen_ids[normalized_symbol].add(trade_id)
        cache.setdefault(symbol, []).append(item)
    actual = {symbol: len(items) for symbol, items in cache.items()}
    if actual != nonzero_counts(metadata["counts"]):
        raise account_cache_health.AccountCacheNotReady(reader_reason)
    return cache, dict(metadata["counts"])


def _validate_order_items_map(
    items: Any, is_valid_item_fn: Callable[[dict[str, Any]], bool] | None = None
) -> dict[str, int]:
    for symbol, orders in items.items():
        if (
            not isinstance(symbol, str)
            or not isinstance(orders, list)
            or (is_valid_item_fn is not None and any(not is_valid_item_fn(o) for o in orders))
        ):
            raise ValueError("invalid Order cache items")
    return {s: len(v) for s, v in items.items()}


def read_legacy_order_snapshot_strict(
    path: str, legacy_meta: dict[str, Any] | None
) -> tuple[dict[str, list[Any]], dict[str, int], dict[str, int], dict[str, int]]:
    """Read and validate a pre-v2 Order snapshot against optional legacy metadata."""
    data = read_json_dict(path)
    if data.get("data_version") is not None:
        raise ValueError("versioned Order cache is missing its versioned manifest")
    items = data.get("items")
    fetchtime = data.get("fetchtime", {})
    if not isinstance(items, dict) or not isinstance(fetchtime, dict):
        raise ValueError("invalid Order cache maps")
    fetchtime = normalize_timestamp_map(fetchtime, "Order cache fetchtime")
    raw_cursor = data.get("rest_sync_cursor")
    if raw_cursor is None and legacy_meta is not None:
        raw_cursor = legacy_meta.get("rest_sync_cursor")
    rest_sync_cursor = _normalize_cursor_map(raw_cursor, fetchtime, "Order cache rest_sync_cursor")
    actual_counts = _validate_order_items_map(items)
    if legacy_meta is not None and "counts" in legacy_meta:
        expected_counts = legacy_meta["counts"]
        if not isinstance(expected_counts, dict) or any(
            not isinstance(s, str) or not _is_strict_int(c) for s, c in expected_counts.items()
        ):
            raise ValueError("invalid legacy Order manifest counts")
        if actual_counts != expected_counts:
            raise ValueError("Order cache counts differ from legacy manifest")
    return items, dict(fetchtime), dict(rest_sync_cursor), actual_counts


def read_order_snapshot_strict(
    path: str,
    metadata: dict[str, Any],
    *,
    is_valid_item_fn: Callable[[dict[str, Any]], bool],
    reader_reason: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int], dict[str, int]]:
    """Read and validate an Order snapshot against its durable manifest."""
    required_version = str(metadata["data_version"])
    try:
        payload = read_json_dict(path)
        items = payload.get("items")
        fetchtime = payload.get("fetchtime")
        if not isinstance(items, dict) or not isinstance(fetchtime, dict):
            raise ValueError("invalid Order cache maps")
        if payload.get("data_version") != required_version:
            raise ValueError("Order cache and marker versions differ")
        normalized_fetchtime = normalize_timestamp_map(fetchtime, error_message="invalid Order cache fetchtime")
        raw_rest_cursor = payload.get("rest_sync_cursor")
        if raw_rest_cursor is not None:
            normalize_timestamp_map(raw_rest_cursor, error_message="invalid Order cache rest_sync_cursor")
        actual_counts = _validate_order_items_map(items, is_valid_item_fn)
        if actual_counts != metadata["counts"]:
            raise ValueError("Order cache counts differ from its manifest")
        if normalized_fetchtime != metadata["fetchtime"]:
            raise ValueError("Order cache fetchtimes differ from its manifest")
        if snapshot_data_version(items, normalized_fetchtime) != required_version:
            raise ValueError("Order cache digest is invalid")
        return items, normalized_fetchtime, actual_counts
    except account_cache_health.AccountCacheNotReady:
        raise
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise account_cache_health.AccountCacheNotReady(reader_reason) from exc


def validate_account_cache_state(
    *,
    cls_name: str,
    is_trade_cache: bool,
    is_order_cache: bool,
    cache: dict[str, Any],
    fetchtime_per_symbol: dict[str, Any],
    rest_sync_cursor_per_symbol: dict[str, Any],
    is_valid_item_fn: Callable[[dict[str, Any]], bool],
) -> None:
    """Reject any in-memory account cache row or timestamp map that strict readers could not accept."""
    normalize_timestamp_map(fetchtime_per_symbol, error_message=f"invalid {cls_name} cache fetchtime")
    if rest_sync_cursor_per_symbol:
        normalize_timestamp_map(
            rest_sync_cursor_per_symbol, error_message=f"invalid {cls_name} cache rest_sync_cursor"
        )
    for symbol, items in cache.items():
        if not isinstance(symbol, str) or not isinstance(items, list):
            raise ValueError(f"invalid {cls_name} cache bucket")
        order_ids: set[str] = set()
        fill_owners: dict[str, str] = {}
        for item in items:
            if not is_valid_item_fn(item):
                raise ValueError(f"invalid {cls_name} cache row")
            if is_trade_cache and str(item["symbol"]).upper() != symbol.upper():
                raise ValueError("Trade row symbol differs from its bucket")
            if is_order_cache:
                if item.get("symbol") is not None and str(item["symbol"]).upper() != symbol.upper():
                    raise ValueError("Order row symbol differs from its bucket")
                order_id = str(_strict_int_field(item["orderId"]))
                if order_id in order_ids:
                    raise ValueError(f"duplicate Order aggregate ID {order_id}")
                order_ids.add(order_id)
                for fill_id in item.get("_fillIds", []):
                    fill_key = str(_strict_int_field(fill_id))
                    previous_order = fill_owners.get(fill_key)
                    if previous_order is not None:
                        raise ValueError(
                            f"immutable fill ID {fill_key} belongs to both orders {previous_order} and {order_id}"
                        )
                    fill_owners[fill_key] = order_id
        if is_trade_cache and len(dedupe_trade_items_by_id(items)) != len(items):
            raise ValueError("duplicate immutable Trade ID")
