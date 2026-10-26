"""The canonical OHLC dataset, shared by every offline backtester.

The module knows nothing about venues or strategies. It fixes only the data contract,
the reproducible hash, and the validations that must be identical for Kraken,
Trading212 and any future adapter.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import math
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence


REQUIRED_FIELDS = ("timestamp", "open", "high", "low", "close")


def normalize_record(record: Mapping) -> dict:
    """Normalise a row into the common OHLC contract."""
    for field in REQUIRED_FIELDS:
        if isinstance(record[field], bool):
            raise ValueError(f"{field} cannot be a boolean")
    raw_ts = record["timestamp"]
    if isinstance(raw_ts, int):
        timestamp = raw_ts
    else:
        try:
            timestamp = int(str(raw_ts).strip())
        except (TypeError, ValueError):
            ts_f = float(raw_ts)
            if not math.isfinite(ts_f) or not ts_f.is_integer():
                raise ValueError(f"invalid timestamp: {raw_ts!r}")
            timestamp = int(ts_f)
    return {
        "timestamp": timestamp,
        "open": float(record["open"]),
        "high": float(record["high"]),
        "low": float(record["low"]),
        "close": float(record["close"]),
    }


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
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(canonical_bytes(records))


def load_dataset(source: Path) -> list[dict]:
    with source.open("r", encoding="ascii", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != REQUIRED_FIELDS:
            raise ValueError(
                f"invalid OHLC header in {source}: {reader.fieldnames}; "
                f"expected {list(REQUIRED_FIELDS)}"
            )
        return [normalize_record(row) for row in reader]


def merge_datasets(*datasets: Iterable[Mapping]) -> list[dict]:
    """Merge overlapping OHLC windows; the newer source wins at the same timestamp."""
    by_timestamp = {}
    for records in datasets:
        for record in records:
            normalized = normalize_record(record)
            by_timestamp[normalized["timestamp"]] = normalized
    return validate_dataset(by_timestamp[timestamp] for timestamp in sorted(by_timestamp))


def validate_dataset(records: Iterable[Mapping], *, interval_minutes: int | None = None) -> list[dict]:
    """Validate ordering, cadence and OHLC invariants; return the normalised form."""
    rows = [normalize_record(record) for record in records]
    if not rows:
        raise ValueError("dataset-ul OHLC cannot be empty")
    if interval_minutes is not None and (isinstance(interval_minutes, bool) or interval_minutes <= 0):
        raise ValueError("interval_minutes must be positive")

    expected_delta = interval_minutes * 60 if interval_minutes is not None else None
    previous_timestamp = None
    for index, row in enumerate(rows, start=2):
        timestamp = row["timestamp"]
        if previous_timestamp is not None:
            delta = timestamp - previous_timestamp
            if delta <= 0:
                raise ValueError(f"timestamp neordonat/duplicat la linia {index}: {timestamp}")
            if expected_delta is not None and delta != expected_delta:
                raise ValueError(
                    f"invalid cadence at line {index}: {delta}s, expected {expected_delta}s"
                )
        previous_timestamp = timestamp

        open_, high, low, close = row["open"], row["high"], row["low"], row["close"]
        if not all(math.isfinite(v) for v in (open_, high, low, close)) or min(open_, high, low, close) <= 0:
            raise ValueError(f"non-positive price at line {index}")
        if high < max(open_, close, low) or low > min(open_, close, high):
            raise ValueError(f"invalid OHLC invariant at line {index}")
    return rows


def drop_incomplete_last_bar(
    records: Iterable[Mapping],
    *,
    interval_minutes: int,
    now_timestamp: int,
    regular_session_start: int | None = None,
    regular_session_end: int | None = None,
) -> list[dict]:
    """Drop only the tail of Yahoo candles that are still forming.

    For intraday the timestamp is the start of the bar. For daily bars Yahoo
    uses the session start, so during the session we compare against the
    ``regular`` end provided in the response metadata.
    """
    if isinstance(interval_minutes, bool) or interval_minutes <= 0:
        raise ValueError("interval_minutes must be positive")
    rows = [normalize_record(record) for record in records]
    if not rows:
        return rows
    if interval_minutes >= 1440:
        last_timestamp = int(rows[-1]["timestamp"])
        in_open_session = (
            regular_session_start is not None
            and regular_session_end is not None
            and int(regular_session_start) <= last_timestamp < int(regular_session_end)
            and int(now_timestamp) < int(regular_session_end)
        )
        return rows[:-1] if in_open_session else rows
    interval_seconds = interval_minutes * 60
    while rows and int(rows[-1]["timestamp"]) + interval_seconds > int(now_timestamp):
        rows.pop()
    return rows


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
    source = sorted(
        (
            int(record["timestamp"]),
            float(record[value_field]),
        )
        for record in source_records
    )
    if not source:
        raise ValueError("the source series for alignment cannot be empty")
    convert = transform or (lambda value: value)
    result = []
    source_index = 0
    current: float | None = None
    for timestamp in target_timestamps:
        target = int(timestamp)
        while source_index < len(source) and source[source_index][0] <= target:
            current = convert(source[source_index][1])
            source_index += 1
        if current is None:
            raise ValueError(
                f"no known FX value at or before timestamp {target}; "
                "extend the FX dataset further back"
            )
        if isinstance(current, bool) or not math.isfinite(float(current)) or float(current) <= 0:
            raise ValueError(f"non-positive aligned value at timestamp {target}")
        result.append(float(current))
    return result
