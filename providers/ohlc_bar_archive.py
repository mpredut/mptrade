"""Persistent incremental OHLCV bar archive store (Layer 3, multi-timeframe ready).

Stores canonical ``[ts_ms, open, high, low, close, volume]`` bars per ``(timeframe, symbol)``
under ``cachedb/prices/cache_ohlc_bars_<TF>_<SYM>.json`` with ``ARCHIVE_DURABLE`` persistence
(``fsync=True``, ``.previous`` backup, compact JSON, serialized via ``generation_transaction``).

Stage 1 consumers use ``timeframe="1d"`` for multi-year long-horizon structure and chart zoom;
the file layout, bucketing, merge logic, and incremental fetch planner are timeframe-agnostic
so Stage 2 can plug in intraday intervals (e.g. ``4h``, ``1h``) without schema changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from botcore import (
    resolve_now_ts,
    safe_int,
    safe_nonneg_float,
    safe_pos_float,
    safe_pos_int,
)
from cachepaths import cache_ohlc_bars_filename, resolve_cachedb_filepath
from providers.ohlc_utils import parse_ohlc_interval
from state_io import (
    generation_transaction,
    read_json_dict_recovering,
    save_json_state,
)

OHLC_BAR_ARCHIVE_SCHEMA_VERSION = 1
_MS_PER_MINUTE = 60_000
_LOCAL_FALLBACK_SOURCE = "local_sparse"


@dataclass(frozen=True)
class BarFetchPlan:
    """Declarative incremental fetch plan computed from persisted archive coverage."""

    mode: str  # "full" | "backfill_extend" | "incremental"
    fetch_start_ms: int
    fetch_end_ms: int
    timeframe: str
    interval_min: int
    prefix_end_ms: Optional[int] = None


def resolve_archive_timeframe(timeframe: str = "1d") -> Tuple[int, str]:
    """Validate and canonicalize ``timeframe`` into ``(interval_min, interval_str)``."""
    return parse_ohlc_interval(timeframe, venue_label="OHLC archive")


def _bar_bucket_key(ts_ms: float, interval_min: int) -> Any:
    if interval_min >= 1440:
        return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
    interval_ms = max(1, interval_min) * _MS_PER_MINUTE
    return (safe_int(ts_ms, default=0) // interval_ms) * interval_ms


def _round_bar_row(ts_ms: float, o_v: float, h_v: float, l_v: float, c_v: float, v_v: float) -> List[float]:
    return [
        safe_int(round(ts_ms), default=0),
        round(o_v, 6),
        round(h_v, 6),
        round(l_v, 6),
        round(c_v, 6),
        round(max(0.0, v_v), 2),
    ]


def normalize_ohlc_archive_bars(
    raw_bars: Sequence[Sequence[Any]],
    *,
    interval_min: int = 1440,
) -> List[List[float]]:
    """Normalize and deduplicate OHLCV or tick rows into sorted ``[ts_ms, o, h, l, c, v]`` bars.

    Full OHLCV rows (len >= 5) on the same bucket replace earlier duplicates instead of
    double-counting volume, while tick-only rows (len < 5) aggregate into OHLC within each bucket.
    """
    eff_interval = max(1, safe_pos_int(interval_min, default=1440) or 1440)
    by_key: Dict[Any, Tuple[List[float], bool]] = {}
    for row in raw_bars or ():
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        ts_raw = safe_pos_float(row[0], default=None)
        if ts_raw is None:
            continue
        ts_ms = ts_raw if ts_raw > 1e11 else ts_raw * 1000.0
        is_ohlc = len(row) >= 5
        if is_ohlc:
            o_v = safe_pos_float(row[1], default=None)
            h_v = safe_pos_float(row[2], default=None)
            l_v = safe_pos_float(row[3], default=None)
            c_v = safe_pos_float(row[4], default=None)
            v_v = safe_nonneg_float(row[5], default=0.0) if len(row) >= 6 else 0.0
            if c_v is None:
                continue
            o_f = o_v if o_v is not None else c_v
            h_f = max(h_v if h_v is not None else c_v, o_f, c_v)
            l_f = min(l_v if l_v is not None else c_v, o_f, c_v)
        else:
            c_v = safe_pos_float(row[1], default=None)
            if c_v is None:
                continue
            o_f = h_f = l_f = c_v
            v_v = safe_nonneg_float(row[2], default=0.0) if len(row) >= 3 else 0.0

        b_key = _bar_bucket_key(ts_ms, eff_interval)
        prev = by_key.get(b_key)
        if prev is None:
            by_key[b_key] = (_round_bar_row(ts_ms, o_f, h_f, l_f, c_v, v_v), is_ohlc)
            continue
        prev_bar, prev_is_ohlc = prev
        if is_ohlc:
            # Authoritative OHLCV row replaces earlier duplicate in the same bucket.
            if not prev_is_ohlc or ts_ms >= prev_bar[0]:
                by_key[b_key] = (_round_bar_row(ts_ms, o_f, h_f, l_f, c_v, v_v), True)
        elif not prev_is_ohlc:
            # Aggregate multiple tick-only samples within the same bucket.
            if ts_ms < prev_bar[0]:
                o_new = o_f
                c_new = prev_bar[4]
                t_new = prev_bar[0]
            else:
                o_new = prev_bar[1]
                c_new = c_v
                t_new = ts_ms
            by_key[b_key] = (
                _round_bar_row(
                    t_new,
                    o_new,
                    max(prev_bar[2], h_f),
                    min(prev_bar[3], l_f),
                    c_new,
                    prev_bar[5] + v_v,
                ),
                False,
            )
    return [by_key[k][0] for k in sorted(by_key.keys())]


def merge_ohlc_archive_bars(
    existing_bars: Sequence[Sequence[Any]],
    incoming_bars: Sequence[Sequence[Any]],
    *,
    interval_min: int = 1440,
    incoming_is_fallback: bool = False,
) -> List[List[float]]:
    """Losslessly merge ``existing_bars`` with ``incoming_bars`` keyed by timeframe bucket.

    - All historical buckets from ``existing_bars`` are preserved.
    - Authoritative ``incoming_bars`` replace overlapping buckets (refreshing the tail bar
      without double-counting volume).
    - When ``incoming_is_fallback=True`` (e.g. ``local_sparse`` ticks during an outage),
      incoming rows only fill buckets absent from ``existing_bars`` so real exchange OHLCV
      is never downgraded.
    """
    eff_interval = max(1, safe_pos_int(interval_min, default=1440) or 1440)
    norm_existing = normalize_ohlc_archive_bars(existing_bars, interval_min=eff_interval)
    norm_incoming = normalize_ohlc_archive_bars(incoming_bars, interval_min=eff_interval)
    merged: Dict[Any, List[float]] = {
        _bar_bucket_key(b[0], eff_interval): b for b in norm_existing
    }
    for bar in norm_incoming:
        b_key = _bar_bucket_key(bar[0], eff_interval)
        if incoming_is_fallback and b_key in merged:
            continue
        merged[b_key] = bar
    return [merged[k] for k in sorted(merged.keys())]


def _is_authoritative_archive(doc: Optional[Mapping[str, Any]]) -> bool:
    if not isinstance(doc, Mapping):
        return False
    bars = doc.get("bars")
    if not isinstance(bars, list) or not bars:
        return False
    src = str(doc.get("source") or "").strip().lower()
    return src != _LOCAL_FALLBACK_SOURCE


def plan_incremental_bar_fetch(
    archive_doc: Optional[Mapping[str, Any]],
    *,
    needed_start_ms: int,
    now_ms: int,
    timeframe: str = "1d",
    overlap_bars: int = 5,
    start_tolerance_bars: int = 7,
) -> BarFetchPlan:
    """Compute minimal network fetch range given an existing persisted archive.

    Returns:
      - ``full`` when no authoritative archive exists on disk yet.
      - ``backfill_extend`` when the caller expanded lookback earlier than previously covered history.
      - ``incremental`` when history back to ``needed_start_ms`` is already covered (fetches only
        ``[last_ts_ms - overlap_bars * interval_ms .. now_ms]``).
    """
    interval_min, tf_str = resolve_archive_timeframe(timeframe)
    interval_ms = interval_min * _MS_PER_MINUTE
    eff_end = max(1, safe_int(now_ms, default=1))
    eff_need = max(0, min(eff_end, safe_int(needed_start_ms, default=0)))

    if not _is_authoritative_archive(archive_doc) or not isinstance(archive_doc, Mapping):
        return BarFetchPlan(
            mode="full",
            fetch_start_ms=eff_need,
            fetch_end_ms=eff_end,
            timeframe=tf_str,
            interval_min=interval_min,
        )

    bars = archive_doc.get("bars") or []
    first_ts = safe_pos_int(archive_doc.get("first_ts_ms"), default=None)
    if first_ts is None and bars and isinstance(bars[0], (list, tuple)) and bars[0]:
        first_ts = safe_pos_int(bars[0][0], default=None)
    last_ts = safe_pos_int(archive_doc.get("last_ts_ms"), default=None)
    if last_ts is None and bars and isinstance(bars[-1], (list, tuple)) and bars[-1]:
        last_ts = safe_pos_int(bars[-1][0], default=None)
    req_from = safe_pos_int(archive_doc.get("requested_from_ms"), default=None)

    if first_ts is None or last_ts is None:
        return BarFetchPlan(
            mode="full",
            fetch_start_ms=eff_need,
            fetch_end_ms=eff_end,
            timeframe=tf_str,
            interval_min=interval_min,
        )

    covered_candidates = [v for v in (first_ts, req_from) if v is not None and v > 0]
    covered_from_ms = min(covered_candidates) if covered_candidates else first_ts
    tol_ms = max(0, safe_int(start_tolerance_bars, default=7)) * interval_ms

    if eff_need + tol_ms < covered_from_ms:
        return BarFetchPlan(
            mode="backfill_extend",
            fetch_start_ms=eff_need,
            fetch_end_ms=eff_end,
            timeframe=tf_str,
            interval_min=interval_min,
            prefix_end_ms=first_ts,
        )

    eff_overlap = max(1, safe_int(overlap_bars, default=5))
    inc_start = max(0, last_ts - eff_overlap * interval_ms)
    return BarFetchPlan(
        mode="incremental",
        fetch_start_ms=min(inc_start, eff_end),
        fetch_end_ms=eff_end,
        timeframe=tf_str,
        interval_min=interval_min,
    )


def ohlc_bar_archive_path(cache_dir: str, symbol: str, timeframe: str = "1d") -> str:
    """Resolve canonical ``cachedb/prices/cache_ohlc_bars_<TF>_<SYM>.json`` path."""
    _, tf_str = resolve_archive_timeframe(timeframe)
    fname = cache_ohlc_bars_filename(symbol, timeframe=tf_str)
    return str(resolve_cachedb_filepath(os.path.join(cache_dir or "cachedb", fname)))


def load_ohlc_bar_archive(
    cache_dir: str,
    symbol: str,
    timeframe: str = "1d",
) -> Optional[Dict[str, Any]]:
    """Passively read persisted OHLCV bar archive for ``(symbol, timeframe)`` with ``.previous`` recovery."""
    path = ohlc_bar_archive_path(cache_dir, symbol, timeframe=timeframe)
    doc = read_json_dict_recovering(path, default=None, repair_files=False)
    if not isinstance(doc, dict):
        return None
    raw_bars = doc.get("bars")
    if not isinstance(raw_bars, list):
        return None
    return doc


def load_ohlc_archive_bars(
    cache_dir: str,
    symbol: str,
    timeframe: str = "1d",
) -> List[List[float]]:
    """Return normalized bars from the persisted OHLCV bar archive, or ``[]`` when absent."""
    interval_min, tf_str = resolve_archive_timeframe(timeframe)
    doc = load_ohlc_bar_archive(cache_dir, symbol, timeframe=tf_str)
    if not isinstance(doc, dict):
        return []
    return normalize_ohlc_archive_bars(doc.get("bars") or (), interval_min=interval_min)


def upsert_ohlc_bar_archive(
    cache_dir: str,
    symbol: str,
    incoming_bars: Sequence[Sequence[Any]],
    *,
    timeframe: str = "1d",
    source: str = "",
    requested_from_ms: Optional[int] = None,
    now_ts: Optional[float] = None,
    incoming_is_fallback: bool = False,
    persist: bool = True,
) -> Optional[Dict[str, Any]]:
    """Merge ``incoming_bars`` into the durable per-symbol OHLCV archive under ``generation_transaction``.

    Never discards older persisted bars; updates ``requested_from_ms`` to the earliest attempted
    network lookback so short-history or API-capped assets (e.g. Kraken 720-bar cap) do not
    re-trigger full backfills on every refresh cycle.
    """
    sym_s = str(symbol or "").strip()
    if not sym_s:
        return None
    interval_min, tf_str = resolve_archive_timeframe(timeframe)
    eff_now = resolve_now_ts(now_ts)
    path = ohlc_bar_archive_path(cache_dir, sym_s, timeframe=tf_str)

    def _merge_with_existing(existing: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        ex_bars = existing.get("bars") if isinstance(existing, dict) and isinstance(existing.get("bars"), list) else []
        merged_bars = merge_ohlc_archive_bars(
            ex_bars,
            incoming_bars,
            interval_min=interval_min,
            incoming_is_fallback=incoming_is_fallback,
        )
        if not merged_bars:
            return existing if isinstance(existing, dict) else None

        ex_source = str(existing.get("source") or "").strip() if isinstance(existing, dict) else ""
        inc_source = str(source or "").strip()
        if incoming_is_fallback and ex_source and ex_source.lower() != _LOCAL_FALLBACK_SOURCE:
            eff_source = ex_source
        else:
            eff_source = inc_source or ex_source

        prev_req = safe_pos_int(existing.get("requested_from_ms"), default=None) if isinstance(existing, dict) else None
        cand_req = None if incoming_is_fallback else safe_pos_int(requested_from_ms, default=None)
        if prev_req is not None and cand_req is not None:
            eff_req: Optional[int] = min(prev_req, cand_req)
        else:
            eff_req = cand_req if cand_req is not None else prev_req

        return {
            "schema_version": OHLC_BAR_ARCHIVE_SCHEMA_VERSION,
            "symbol": sym_s,
            "timeframe": tf_str,
            "interval_min": interval_min,
            "source": eff_source,
            "updated_ts": round(eff_now, 3),
            "first_ts_ms": safe_int(merged_bars[0][0], default=0),
            "last_ts_ms": safe_int(merged_bars[-1][0], default=0),
            "requested_from_ms": eff_req,
            "bar_count": len(merged_bars),
            "bars": merged_bars,
        }

    if not persist:
        return _merge_with_existing(load_ohlc_bar_archive(cache_dir, sym_s, timeframe=tf_str))

    with generation_transaction(path) as target_path:
        existing = read_json_dict_recovering(target_path, default=None, repair_files=True)
        doc = _merge_with_existing(existing if isinstance(existing, dict) else None)
        if doc is None:
            return None
        # Skip disk write + fsync when bars/source/coverage did not change.
        if (
            isinstance(existing, dict)
            and existing.get("bars") == doc.get("bars")
            and existing.get("source") == doc.get("source")
            and existing.get("requested_from_ms") == doc.get("requested_from_ms")
            and existing.get("schema_version") == doc.get("schema_version")
        ):
            return existing
        save_json_state(target_path, doc, skip_if_unchanged=True)
        return doc
