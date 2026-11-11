"""The canonical OHLC dataset, shared by every offline backtester.

The module knows nothing about venues or strategies. It fixes only the data contract,
the reproducible hash, and the validations that must be identical for Kraken,
Trading212 and any future adapter.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from botcore import safe_float, safe_pos_float
from providers.order_ohlc_utils import (
    REQUIRED_FIELDS,
    drop_incomplete_last_bar,
    normalize_record,
    validate_dataset,
)
from state_io import atomic_write_bytes, read_csv_dicts, read_text_file


def canonical_bytes(records: Iterable[Mapping]) -> bytes:
    """Stable serialisation used for both the CSV and the SHA-256."""
    lines = [",".join(REQUIRED_FIELDS)]
    for source in records:
        row = normalize_record(source)
        lines.append(
            f"{row['timestamp']},{row['open']:.12g},{row['high']:.12g},"
            f"{row['low']:.12g},{row['close']:.12g}"
        )
    return ("\n".join(lines) + "\n").encode("ascii")


def dataset_sha256(records: Iterable[Mapping]) -> str:
    return hashlib.sha256(canonical_bytes(records)).hexdigest()


def save_dataset(records: Iterable[Mapping], destination: Path) -> None:
    atomic_write_bytes(destination, canonical_bytes(records))


def load_dataset(source: str | Path) -> list[dict]:
    path = Path(source)
    content = read_text_file(path)
    reader = csv.DictReader(io.StringIO(content))
    if tuple(reader.fieldnames or ()) != REQUIRED_FIELDS:
        raise ValueError(
            f"invalid OHLC header in {path}: {reader.fieldnames}; "
            f"expected {list(REQUIRED_FIELDS)}"
        )
    rows = [normalize_record(row) for row in reader]
    return validate_dataset(rows) if rows else []


def load_ohlc_tuples_csv(
    path: str | Path,
    *,
    strict: bool = False,
) -> list[tuple[int, float, float, float, float]]:
    """Load valid (timestamp, open, high, low, close) tuples from a CSV file via state_io.read_csv_dicts."""
    if isinstance(path, (bool, int, float)):
        return []
    rows: list[tuple[int, float, float, float, float]] = []
    for r in read_csv_dicts(path, strict=strict):
        ts_f = safe_pos_float(r.get("timestamp"), default=None)
        o = safe_pos_float(r.get("open"), default=None)
        h = safe_pos_float(r.get("high"), default=None)
        l = safe_pos_float(r.get("low"), default=None)
        c = safe_pos_float(r.get("close"), default=None)
        if (
            ts_f is not None
            and int(ts_f) > 0
            and o is not None
            and h is not None
            and l is not None
            and c is not None
            and h >= max(o, l, c)
            and l <= min(o, h, c)
        ):
            rows.append((int(ts_f), o, h, l, c))
    return rows


def merge_datasets(*datasets: Iterable[Mapping]) -> list[dict]:
    """Merge overlapping OHLC windows; the newer source wins at the same timestamp."""
    by_timestamp = {}
    for records in datasets:
        for record in records:
            normalized = normalize_record(record)
            by_timestamp[normalized["timestamp"]] = normalized
    return validate_dataset(by_timestamp[timestamp] for timestamp in sorted(by_timestamp))


def dataset_metadata(records: Iterable[Mapping], *, interval_minutes: int | None = None) -> dict:
    rows = validate_dataset(records, interval_minutes=interval_minutes)
    return {
        "sha256": dataset_sha256(rows),
        "bars": len(rows),
        "start_utc": dt.datetime.fromtimestamp(
            rows[0]["timestamp"], tz=dt.timezone.utc,
        ).isoformat(),
        "end_utc": dt.datetime.fromtimestamp(
            rows[-1]["timestamp"], tz=dt.timezone.utc,
        ).isoformat(),
    }


def align_previous_values(
    target_timestamps: Sequence[int],
    source_records: Iterable[Mapping],
    *,
    value_field: str = "close",
    transform: Callable[[float], float] | None = None,
) -> list[float]:
    """Align as-of without lookahead: the last value with timestamp <= target."""
    by_ts: dict[int, float] = {}
    for record in source_records:
        raw_ts = record["timestamp"]
        raw_val = record[value_field]
        if isinstance(raw_ts, bool) or isinstance(raw_val, bool) or raw_ts is None or raw_val is None:
            raise ValueError("source_records cannot contain None or boolean timestamp/value")
        ts_f = safe_float(raw_ts, default=None)
        val_f = safe_float(raw_val, default=None)
        if ts_f is None or val_f is None:
            raise ValueError("invalid or non-finite timestamp or value in source_records")
        ts_i = int(ts_f)
        by_ts[ts_i] = val_f
    source = sorted(by_ts.items(), key=lambda item: item[0])
    if not source:
        raise ValueError("the source series for alignment cannot be empty")
    convert = transform or (lambda value: value)
    result = []
    source_index = 0
    current: float | None = None
    for timestamp in target_timestamps:
        if isinstance(timestamp, bool) or timestamp is None:
            raise ValueError("target_timestamps cannot contain None or boolean")
        tf = safe_float(timestamp, default=None)
        if tf is None:
            raise ValueError(f"invalid or non-finite target timestamp: {timestamp!r}")
        target = int(tf)
        while source_index < len(source) and source[source_index][0] <= target:
            current = convert(source[source_index][1])
            source_index += 1
        if current is None:
            raise ValueError(
                f"no known FX value at or before timestamp {target}; "
                "extend the FX dataset further back"
            )
        cur_f = safe_pos_float(current, default=None)
        if cur_f is None:
            raise ValueError(f"non-positive aligned value at timestamp {target}")
        result.append(cur_f)
    return result
