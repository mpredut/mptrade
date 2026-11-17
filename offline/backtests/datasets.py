"""The canonical OHLC dataset, shared by every offline backtester.

The module knows nothing about venues or strategies. It fixes only the data contract,
the reproducible hash, and the validations that must be identical for Kraken,
Trading212 and any future adapter.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from botcore import exact_pos_int, is_cache_fresh, parse_cli_float, resolve_now_ts, safe_pos_float
from providers.ohlc_utils import (
    REQUIRED_FIELDS,
    align_previous_values,
    drop_incomplete_last_bar,
    normalize_record,
    validate_dataset,
)
from state_io import atomic_write_bytes, read_csv_dicts, read_text_file

DAILY_REFRESH_CADENCE_SEC: float = 86400.0
DAILY_DATASET_MAX_AGE_SEC: float = DAILY_REFRESH_CADENCE_SEC * 0.75


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
    rows = list(records)
    validated = validate_dataset(rows) if rows else []
    atomic_write_bytes(destination, canonical_bytes(validated))


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
    by_ts: dict[int, tuple[int, float, float, float, float]] = {}
    for r in read_csv_dicts(path, strict=strict):
        ts_i = exact_pos_int(r.get("timestamp"), default=None)
        o = safe_pos_float(r.get("open"), default=None)
        h = safe_pos_float(r.get("high"), default=None)
        l = safe_pos_float(r.get("low"), default=None)
        c = safe_pos_float(r.get("close"), default=None)
        if (
            ts_i is not None
            and o is not None
            and h is not None
            and l is not None
            and c is not None
            and h >= max(o, l, c)
            and l <= min(o, h, c)
        ):
            by_ts[ts_i] = (ts_i, o, h, l, c)
    return [by_ts[k] for k in sorted(by_ts)]


def has_usable_ohlc_csv(
    path: str | Path,
    *,
    min_bars: int,
    strict: bool = True,
) -> bool:
    """Return True when ``path`` exists and parses to at least ``min_bars`` valid OHLC rows."""
    if isinstance(path, (bool, int, float)) or not os.path.isfile(path):
        return False
    try:
        rows = load_ohlc_tuples_csv(path, strict=strict)
    except (OSError, ValueError):
        return False
    return len(rows) >= min_bars


def should_skip_ohlc_csv_refresh(
    path: str | Path,
    *,
    min_bars: int,
    strict: bool = True,
    force: bool = False,
    missing_only: bool = False,
    max_age_sec: float = DAILY_DATASET_MAX_AGE_SEC,
    now_ts: float | None = None,
) -> tuple[bool, bool]:
    """Return ``(skip_refresh, has_usable_csv)`` for daily/on-demand OHLC CSV refreshers."""
    has_usable = has_usable_ohlc_csv(path, min_bars=min_bars, strict=strict)
    if not has_usable or force:
        return False, has_usable
    if missing_only:
        return True, True
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return False, True
    return is_cache_fresh(mtime, resolve_now_ts(now_ts), max_age_sec), True


def merge_datasets(*datasets: Iterable[Mapping]) -> list[dict]:
    """Merge overlapping OHLC windows; the newer source wins at the same timestamp."""
    by_timestamp = {}
    for records in datasets:
        for record in records:
            normalized = normalize_record(record)
            by_timestamp[normalized["timestamp"]] = normalized
    return validate_dataset(by_timestamp[timestamp] for timestamp in sorted(by_timestamp))


def ohlc_range_window_seconds(range_spec: object) -> int | None:
    """Convert an OHLC range token such as ``730d`` or ``10y`` into seconds."""
    if isinstance(range_spec, bool) or not isinstance(range_spec, str):
        return None
    token = range_spec.strip().lower()
    if len(token) < 2:
        return None
    unit = token[-1]
    qty = safe_pos_float(token[:-1], default=None)
    if qty is None:
        return None
    if unit == "d":
        return exact_pos_int(round(qty * 86400.0), default=None)
    if unit == "y":
        return exact_pos_int(round(qty * 365.25 * 86400.0), default=None)
    if unit == "h":
        return exact_pos_int(round(qty * 3600.0), default=None)
    return None


def _load_existing_ohlc_records(csv_path: str | Path) -> list[dict]:
    if isinstance(csv_path, (bool, int, float)) or not os.path.isfile(csv_path):
        return []
    try:
        return load_dataset(csv_path)
    except (OSError, ValueError):
        return [
            {"timestamp": ts_i, "open": o, "high": h, "low": l, "close": c}
            for ts_i, o, h, l, c in load_ohlc_tuples_csv(csv_path, strict=False)
        ]


def merge_with_existing_ohlc_csv(
    records: Iterable[Mapping],
    csv_path: str | Path,
    *,
    min_timestamp: int | None = None,
    window_seconds: int | None = None,
) -> list[dict]:
    """Merge newly fetched OHLC bars with an existing CSV and trim to the target horizon."""
    fresh_rows = list(records)
    existing = _load_existing_ohlc_records(csv_path)
    if existing and fresh_rows:
        merged = merge_datasets(existing, fresh_rows)
    elif fresh_rows:
        merged = validate_dataset(fresh_rows)
    else:
        merged = existing
    if not merged:
        return []
    floor_ts = exact_pos_int(min_timestamp, default=None)
    win_sec = exact_pos_int(window_seconds, default=None)
    if win_sec is not None:
        win_floor = merged[-1]["timestamp"] - win_sec
        floor_ts = win_floor if floor_ts is None else max(floor_ts, win_floor)
    if floor_ts is not None:
        windowed = [r for r in merged if r["timestamp"] >= floor_ts]
        if windowed:
            merged = windowed
    return merged


def format_utc_bar_timestamp(ts: object, *, include_time: bool = False) -> str | None:
    """Format an epoch-second OHLC bar timestamp in UTC."""
    ts_i = exact_pos_int(ts, default=None)
    if ts_i is None:
        return None
    fmt = "%Y-%m-%d %H:%M UTC" if include_time else "%Y-%m-%d"
    return dt.datetime.fromtimestamp(ts_i, tz=dt.timezone.utc).strftime(fmt)


def format_ohlc_bars_log_span(records: Sequence[Mapping]) -> str:
    """Format ``<N> bars <YYYY-MM-DD> -> <YYYY-MM-DD>`` for OHLC fetch progress logs."""
    if not records:
        return "0 bars"
    start_s = format_utc_bar_timestamp(records[0].get("timestamp"), include_time=False) or "?"
    end_s = format_utc_bar_timestamp(records[-1].get("timestamp"), include_time=False) or "?"
    return f"{len(records)} bars {start_s} -> {end_s}"


def parse_ohlc_refresh_cli_args(
    description: str | None = None,
    *,
    default_max_age_sec: float = DAILY_DATASET_MAX_AGE_SEC,
) -> tuple[bool, bool, float]:
    """Parse standard ``(--force, --missing-only, --max-age-hours)`` CLI flags and return ``(force, missing_only, max_age_sec)``."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--force", action="store_true", help="re-fetch all symbols regardless of existing CSV age")
    ap.add_argument("--missing-only", action="store_true", help="skip symbols with any usable existing CSV")
    ap.add_argument(
        "--max-age-hours",
        type=parse_cli_float,
        default=default_max_age_sec / 3600.0,
        help="maximum cached CSV age in hours before refreshing (default: 18h)",
    )
    cli_args = ap.parse_args()
    if cli_args.max_age_hours < 0.0:
        ap.error("--max-age-hours must be >= 0")
    force_refresh = bool(cli_args.force)
    missing_only = bool(cli_args.missing_only and not force_refresh)
    return force_refresh, missing_only, cli_args.max_age_hours * 3600.0


def summarize_ohlc_spans(
    spans: Iterable[tuple[int, int]],
    *,
    include_time: bool = True,
) -> str | None:
    """Format effective ``start -> end`` coverage and last-bar UTC timestamp(s) across assets."""
    valid: list[tuple[int, int]] = []
    for item in spans:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        s_i = exact_pos_int(item[0], default=None)
        e_i = exact_pos_int(item[1], default=None)
        if s_i is not None and e_i is not None and s_i <= e_i:
            valid.append((s_i, e_i))
    if not valid:
        return None
    min_start = min(s for s, _ in valid)
    min_end = min(e for _, e in valid)
    max_end = max(e for _, e in valid)
    start_s = format_utc_bar_timestamp(min_start, include_time=False)
    end_s = format_utc_bar_timestamp(max_end, include_time=False)
    last_max_s = format_utc_bar_timestamp(max_end, include_time=include_time)
    if not (start_s and end_s and last_max_s):
        return None
    if min_end != max_end:
        last_min_s = format_utc_bar_timestamp(min_end, include_time=include_time)
        last_desc = f"{last_min_s} .. {last_max_s}" if last_min_s else last_max_s
    else:
        last_desc = last_max_s
    return f"{start_s} -> {end_s} (last bar: {last_desc})"


def scan_ohlc_csv_spans(
    data_dir: str | Path,
    *,
    suffix: str = ".csv",
) -> list[tuple[int, int]]:
    """Scan OHLC CSV files in ``data_dir`` matching ``suffix`` and return ``(start_ts, end_ts)`` pairs."""
    if isinstance(data_dir, (bool, int, float)) or not os.path.isdir(data_dir):
        return []
    spans: list[tuple[int, int]] = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(suffix):
            continue
        rows = load_ohlc_tuples_csv(os.path.join(data_dir, fname), strict=False)
        if rows:
            spans.append((rows[0][0], rows[-1][0]))
    return spans


def summarize_ohlc_dir_coverage(
    data_dir: str | Path,
    *,
    suffix: str = ".csv",
    include_time: bool = True,
) -> str | None:
    """Scan OHLC CSVs in ``data_dir`` and return formatted ``start -> end (last bar: ...)`` coverage."""
    return summarize_ohlc_spans(scan_ohlc_csv_spans(data_dir, suffix=suffix), include_time=include_time)


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


def dataset_metadata_and_csv_path(
    records: Iterable[Mapping],
    output_dir: Path,
    stem: str,
    interval: int,
    *,
    tag: str = "",
) -> tuple[dict, Path]:
    """Compute validated dataset metadata and its content-addressed CSV path."""
    meta = dataset_metadata(records, interval_minutes=interval)
    digest = meta["sha256"]
    tag_part = f"_{tag}" if tag else ""
    return meta, Path(output_dir) / f"{stem}_{interval}m{tag_part}_{digest[:12]}.csv"
