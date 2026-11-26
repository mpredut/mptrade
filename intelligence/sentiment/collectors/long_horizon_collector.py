"""Single-writer collector for persistent multi-month/annual daily bars and optional operator strategic notes.

Maintains durable per-symbol OHLCV archives in ``cachedb/prices/cache_ohlc_bars_1d_<SYM>.json``
(full backfill once, incremental tail refresh thereafter; never re-fetches persisted history)
and writes the lightweight derived prompt summary to ``cachedb/macro/long_horizon_collect.json``.
"""
from __future__ import annotations

import logging
import math
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from botcore import (
    fetch_json_post,
    fetch_public_json,
    is_cache_fresh,
    resolve_now_ts,
    safe_int,
    safe_nonneg_float,
    safe_pos_float,
    safe_pos_int,
)
from cachepaths import (
    CACHE_LONG_HORIZON_FILENAME,
    OPERATOR_REGIME_NOTE_FILENAME,
    cache_ohlc_bars_filename,
    cache_sparse_price_filename,
    canonical_base_asset,
    ensure_quote_suffix,
    is_kraken_tokenized_asset,
    resolve_cachedb_filepath,
    strip_equity_suffix,
)
from instrument_registry import (
    discover_tradable_specs,
    is_equity_symbol,
    venue_for_symbol,
)
from intelligence.external.collectors.base_collector import (
    TelemetryCacheCollectorInterface,
    TelemetryCollectorFactory,
)
from intelligence.macro_sentiment_ssot import (
    get_binance_daily_klines_url,
    get_hyperliquid_info_url,
    get_kraken_daily_ohlc_url,
    get_yahoo_daily_chart_url,
    load_cached_long_horizon_snapshot,
)
from intelligence.sentiment.evaluators.long_horizon_evaluator import (
    evaluate_symbol_long_horizon,
    format_horizon_label,
    parse_operator_regime_note_text,
    resolve_long_horizon_months,
)
from providers.ohlc_bar_archive import (
    BarFetchPlan,
    load_ohlc_bar_archive,
    plan_incremental_bar_fetch,
    upsert_ohlc_bar_archive,
)
from providers.ohlc_utils import extract_yahoo_chart_ohlcv_bars
from state_io import atomic_write_json, parse_json_payload, read_text_file

logger = logging.getLogger("intelligence.sentiment.long_horizon_collector")

_DAYS_PER_MONTH = 30.4375
_MS_PER_DAY = 86_400_000.0
_MA200_WARMUP_DAYS = 210
_INCREMENTAL_OVERLAP_DAYS = 5
_ARCHIVE_TIMEFRAME = "1d"
_BENCHMARK_SYMBOLS: Tuple[str, ...] = ("SPY", "QQQ")


def _yahoo_range_for_days(lookback_days: int) -> str:
    """Pick the smallest Yahoo daily range string covering ``lookback_days``."""
    days = max(1, safe_int(lookback_days, default=30))
    if days <= 25:
        return "1mo"
    if days <= 80:
        return "3mo"
    if days <= 170:
        return "6mo"
    if days <= 350:
        return "1y"
    if days <= 710:
        return "2y"
    if days <= 1800:
        return "5y"
    return "max"


def _fetch_binance_daily_bars(
    symbol: str,
    *,
    start_ms: int,
    end_ms: int,
    timeout_sec: float,
    fetch_fn: Callable[..., Any],
) -> List[List[float]]:
    eff_end = max(1, safe_int(end_ms, default=1))
    eff_start = max(0, min(eff_end, safe_int(start_ms, default=0)))
    span_days = max(1, math.ceil((eff_end - eff_start) / _MS_PER_DAY))
    remaining = span_days + 2
    cur_end = eff_end
    collected: List[List[float]] = []
    for _ in range(3):
        if remaining <= 0 or cur_end <= 0:
            break
        batch_lim = min(1000, remaining)
        url = get_binance_daily_klines_url(symbol, batch_lim, cur_end)
        raw = fetch_fn(url, timeout=timeout_sec)
        if not isinstance(raw, list) or not raw:
            break
        batch: List[List[float]] = []
        for k in raw:
            if not isinstance(k, (list, tuple)) or len(k) < 6:
                continue
            t_ms = safe_pos_float(k[0], default=None)
            o_v = safe_pos_float(k[1], default=None)
            h_v = safe_pos_float(k[2], default=None)
            l_v = safe_pos_float(k[3], default=None)
            c_v = safe_pos_float(k[4], default=None)
            v_v = safe_nonneg_float(k[5], default=0.0)
            if None in (t_ms, o_v, h_v, l_v, c_v):
                continue
            batch.append([t_ms, o_v, h_v, l_v, c_v, v_v])
        if not batch:
            break
        collected.extend(batch)
        oldest_ms = safe_int(batch[0][0], default=0)
        if len(batch) < batch_lim or oldest_ms <= max(1, eff_start):
            break
        cur_end = oldest_ms - 1
        remaining -= len(batch)
    collected.sort(key=lambda r: r[0])
    return collected


def _fetch_kraken_daily_bars(
    pair: str,
    *,
    start_ms: int,
    timeout_sec: float,
    fetch_fn: Callable[..., Any],
) -> List[List[float]]:
    since_sec = max(0, safe_int(start_ms // 1000, default=0))
    url = get_kraken_daily_ohlc_url(pair, since_sec)
    raw = fetch_fn(url, timeout=timeout_sec)
    if not isinstance(raw, dict):
        return []
    result = raw.get("result")
    if not isinstance(result, dict):
        return []
    rows_raw = None
    for k, v in result.items():
        if k != "last" and isinstance(v, list):
            rows_raw = v
            break
    if not rows_raw:
        return []
    out: List[List[float]] = []
    for k in rows_raw:
        if not isinstance(k, (list, tuple)) or len(k) < 7:
            continue
        t_s = safe_pos_float(k[0], default=None)
        o_v = safe_pos_float(k[1], default=None)
        h_v = safe_pos_float(k[2], default=None)
        l_v = safe_pos_float(k[3], default=None)
        c_v = safe_pos_float(k[4], default=None)
        v_v = safe_nonneg_float(k[6], default=0.0)
        if None in (t_s, o_v, h_v, l_v, c_v):
            continue
        out.append([t_s * 1000.0, o_v, h_v, l_v, c_v, v_v])
    return out


def _fetch_hyperliquid_daily_bars(
    coin: str,
    *,
    start_ms: int,
    end_ms: int,
    timeout_sec: float,
    post_fn: Callable[..., Any],
) -> List[List[float]]:
    eff_end = max(1, safe_int(end_ms, default=1))
    eff_start = max(0, min(eff_end, safe_int(start_ms, default=0)))
    payload = {
        "type": "candleSnapshot",
        "req": {"coin": str(coin).strip().upper(), "interval": "1d", "startTime": eff_start, "endTime": eff_end},
    }
    raw = post_fn(get_hyperliquid_info_url(), payload, timeout=timeout_sec)
    if not isinstance(raw, list):
        return []
    out: List[List[float]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        t_ms = safe_pos_float(item.get("t"), default=None)
        o_v = safe_pos_float(item.get("o"), default=None)
        h_v = safe_pos_float(item.get("h"), default=None)
        l_v = safe_pos_float(item.get("l"), default=None)
        c_v = safe_pos_float(item.get("c"), default=None)
        v_v = safe_nonneg_float(item.get("v"), default=0.0)
        if None in (t_ms, o_v, h_v, l_v, c_v):
            continue
        out.append([t_ms, o_v, h_v, l_v, c_v, v_v])
    return out


def _fetch_yahoo_daily_bars(
    ticker: str,
    *,
    lookback_days: int,
    timeout_sec: float,
    fetch_fn: Callable[..., Any],
) -> List[List[float]]:
    url = get_yahoo_daily_chart_url(ticker, _yahoo_range_for_days(lookback_days))
    raw = fetch_fn(url, timeout=timeout_sec)
    if not isinstance(raw, dict):
        return []
    _, bars = extract_yahoo_chart_ohlcv_bars(raw)
    out: List[List[float]] = []
    for b in bars or ():
        if isinstance(b, (list, tuple)) and len(b) >= 5:
            t_ms = safe_pos_float(b[0], default=None)
            o_v = safe_pos_float(b[1], default=None)
            h_v = safe_pos_float(b[2], default=None)
            l_v = safe_pos_float(b[3], default=None)
            c_v = safe_pos_float(b[4], default=None)
            v_v = safe_nonneg_float(b[5], default=0.0) if len(b) >= 6 else 0.0
            if None not in (t_ms, o_v, h_v, l_v, c_v):
                out.append([t_ms, o_v, h_v, l_v, c_v, v_v])
    return out


def _load_sparse_local_daily_bars(cache_dir: str, symbol: str) -> List[List[float]]:
    """Passive fallback reading local ``cache_sparse_price_<SYM>.jsonl`` via state_io when network is unavailable."""
    fpath = str(resolve_cachedb_filepath(os.path.join(cache_dir, cache_sparse_price_filename(symbol))))
    if not os.path.exists(fpath):
        return []
    raw_txt = read_text_file(fpath, default="")
    if not raw_txt:
        return []
    out: List[List[float]] = []
    for ln in raw_txt.splitlines():
        if not ln.strip():
            continue
        obj = parse_json_payload(ln, default=None)
        if not isinstance(obj, dict):
            continue
        item = obj.get("i")
        if isinstance(item, list) and len(item) >= 2:
            t_v = safe_pos_float(item[0], default=None)
            p_v = safe_pos_float(item[1], default=None)
            if t_v is not None and p_v is not None:
                out.append([t_v, p_v])
    return out


def _planned_windows(plan: BarFetchPlan, existing_doc: Optional[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """Return disjoint ``(start_ms, end_ms)`` fetch windows so ``backfill_extend`` never re-fetches the persisted middle."""
    if plan.mode != "backfill_extend" or plan.prefix_end_ms is None:
        return [(plan.fetch_start_ms, plan.fetch_end_ms)]
    windows: List[Tuple[int, int]] = []
    if plan.prefix_end_ms > plan.fetch_start_ms:
        windows.append((plan.fetch_start_ms, max(plan.fetch_start_ms + 1, plan.prefix_end_ms - 1)))
    last_ts = safe_pos_int(existing_doc.get("last_ts_ms"), default=None) if isinstance(existing_doc, dict) else None
    if last_ts is not None:
        tail_start = max(0, last_ts - _INCREMENTAL_OVERLAP_DAYS * safe_int(_MS_PER_DAY, default=86_400_000))
        if tail_start < plan.fetch_end_ms:
            windows.append((tail_start, plan.fetch_end_ms))
    return windows or [(plan.fetch_start_ms, plan.fetch_end_ms)]


def _fetch_hyperliquid_or_fallback_daily_bars(
    sym_s: str,
    *,
    plan: BarFetchPlan,
    existing_doc: Optional[Dict[str, Any]],
    timeout_sec: float,
    fetch_fn: Callable[..., Any],
    post_fn: Callable[..., Any],
) -> Tuple[List[List[float]], str]:
    base_c = canonical_base_asset(sym_s)
    usd_pair = ensure_quote_suffix(base_c, "USD")
    existing_src = str(existing_doc.get("source") or "") if isinstance(existing_doc, dict) else ""
    fallback_lbl = f"kraken:{usd_pair}"
    if plan.mode == "incremental" and existing_src == fallback_lbl:
        return (
            _fetch_kraken_daily_bars(usd_pair, start_ms=plan.fetch_start_ms, timeout_sec=timeout_sec, fetch_fn=fetch_fn),
            fallback_lbl,
        )

    bars: List[List[float]] = []
    for w_start, w_end in _planned_windows(plan, existing_doc):
        bars.extend(
            _fetch_hyperliquid_daily_bars(
                base_c, start_ms=w_start, end_ms=w_end, timeout_sec=timeout_sec, post_fn=post_fn
            )
        )
    src_lbl = f"hyperliquid:{base_c}"
    existing_cnt = safe_int(existing_doc.get("bar_count"), default=0) if isinstance(existing_doc, dict) else 0
    if plan.mode != "incremental" and max(len(bars), existing_cnt) < 30:
        k_bars = _fetch_kraken_daily_bars(
            usd_pair, start_ms=plan.fetch_start_ms, timeout_sec=timeout_sec, fetch_fn=fetch_fn
        )
        if len(k_bars) > len(bars):
            return k_bars, fallback_lbl
    return bars, src_lbl


def _fetch_kraken_venue_daily_bars(
    sym_s: str,
    *,
    plan: BarFetchPlan,
    existing_doc: Optional[Dict[str, Any]],
    timeout_sec: float,
    fetch_fn: Callable[..., Any],
    post_fn: Callable[..., Any],
) -> Tuple[List[List[float]], str]:
    _ = (existing_doc, post_fn)
    bars = _fetch_kraken_daily_bars(
        sym_s, start_ms=plan.fetch_start_ms, timeout_sec=timeout_sec, fetch_fn=fetch_fn
    )
    return bars, f"kraken:{sym_s}"


def _fetch_binance_venue_daily_bars(
    sym_s: str,
    *,
    plan: BarFetchPlan,
    existing_doc: Optional[Dict[str, Any]],
    timeout_sec: float,
    fetch_fn: Callable[..., Any],
    post_fn: Callable[..., Any],
) -> Tuple[List[List[float]], str]:
    _ = post_fn
    bars: List[List[float]] = []
    for w_start, w_end in _planned_windows(plan, existing_doc):
        bars.extend(
            _fetch_binance_daily_bars(
                sym_s, start_ms=w_start, end_ms=w_end, timeout_sec=timeout_sec, fetch_fn=fetch_fn
            )
        )
    return bars, f"binance:{sym_s}"


_CRYPTO_VENUE_DAILY_FETCHERS: Dict[str, Callable[..., Tuple[List[List[float]], str]]] = {
    "hyperliquid": _fetch_hyperliquid_or_fallback_daily_bars,
    "kraken": _fetch_kraken_venue_daily_bars,
    "binance": _fetch_binance_venue_daily_bars,
}


def _resolve_default_long_horizon_symbols(workspace_dir: str = ".") -> List[str]:
    """Discover configured tradable symbols from instruments.conf SSOT plus major benchmarks."""
    syms: List[str] = []
    seen: set[str] = set()
    try:
        specs = discover_tradable_specs(workspace_dir, asset_scope="all", include_closed_rth=True)
        for sp in specs.values():
            s = str(getattr(sp, "symbol", "") or "").strip()
            if s and s not in seen:
                # Deduplicate HYPEUSD into primary HYPEUSDC
                if s.upper() == "HYPEUSD" and "HYPEUSDC" in seen:
                    continue
                seen.add(s)
                syms.append(s)
    except Exception:
        pass
    for b_sym in _BENCHMARK_SYMBOLS:
        if b_sym not in seen:
            seen.add(b_sym)
            syms.append(b_sym)
    return syms


class LongHorizonStructureCollector(TelemetryCacheCollectorInterface):
    """Concrete collector (Level 3) maintaining durable 1d bar archives and ``long_horizon_collect.json``."""

    cls_name = "LongHorizonStructureCollector"
    COLLECTOR_ID = "long_horizon_structure"

    def collect(
        self,
        cachedb_dir: str = "cachedb",
        symbol: str = "*",
        *,
        months: Optional[int] = None,
        symbols: Optional[Sequence[str]] = None,
        now_ts: Optional[float] = None,
        timeout_sec: float = 5.0,
        allow_network: bool = True,
        persist: bool = True,
        skip_if_fresh_sec: float = 0.0,
        fetch_fn: Optional[Callable[..., Any]] = None,
        post_fn: Optional[Callable[..., Any]] = None,
        workspace_dir: str = ".",
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        _ = (symbol, kwargs)
        eff_now = resolve_now_ts(now_ts)
        eff_now_ms = safe_int(eff_now * 1000.0, default=1)
        eff_months = resolve_long_horizon_months(months)
        prev_snap = load_cached_long_horizon_snapshot(cachedb_dir)
        if not allow_network and fetch_fn is None and post_fn is None:
            return prev_snap
        if (
            prev_snap is not None
            and skip_if_fresh_sec > 0.0
            and safe_int(prev_snap.get("months"), default=0) == eff_months
            and isinstance(prev_snap.get("items"), dict)
            and bool(prev_snap["items"])
            and is_cache_fresh(prev_snap.get("ts"), eff_now, ttl_sec=skip_if_fresh_sec)
            and (
                not persist
                or all(
                    isinstance(it, dict)
                    and bool(it.get("bar_archive"))
                    and os.path.exists(
                        str(resolve_cachedb_filepath(os.path.join(cachedb_dir, str(it["bar_archive"]))))
                    )
                    for it in prev_snap["items"].values()
                )
            )
        ):
            return prev_snap

        eff_fetch = fetch_fn or fetch_public_json
        eff_post = post_fn or fetch_json_post
        target_syms = list(symbols) if symbols else _resolve_default_long_horizon_symbols(workspace_dir)
        lookback_days = round(eff_months * _DAYS_PER_MONTH)
        needed_days = lookback_days + _MA200_WARMUP_DAYS
        needed_start_ms = max(0, eff_now_ms - safe_int(needed_days * _MS_PER_DAY, default=0))
        items_out: Dict[str, Dict[str, Any]] = {}

        for sym in target_syms:
            sym_s = str(sym or "").strip()
            if not sym_s:
                continue
            existing_doc = load_ohlc_bar_archive(cachedb_dir, sym_s, timeframe=_ARCHIVE_TIMEFRAME)
            plan = plan_incremental_bar_fetch(
                existing_doc,
                needed_start_ms=needed_start_ms,
                now_ms=eff_now_ms,
                timeframe=_ARCHIVE_TIMEFRAME,
                overlap_bars=_INCREMENTAL_OVERLAP_DAYS,
            )
            plan_days = max(1, math.ceil((plan.fetch_end_ms - plan.fetch_start_ms) / _MS_PER_DAY))
            existing_src = str(existing_doc.get("source") or "") if isinstance(existing_doc, dict) else ""
            existing_cnt = safe_int(existing_doc.get("bar_count"), default=0) if isinstance(existing_doc, dict) else 0

            fetched_bars: List[List[float]] = []
            src_lbl = existing_src
            try:
                if sym_s.upper() in _BENCHMARK_SYMBOLS or is_equity_symbol(sym_s):
                    ticker = strip_equity_suffix(sym_s) if is_equity_symbol(sym_s) else sym_s.upper()
                    fetched_bars = _fetch_yahoo_daily_bars(
                        ticker, lookback_days=plan_days, timeout_sec=timeout_sec, fetch_fn=eff_fetch
                    )
                    src_lbl = f"yahoo:{ticker}"
                elif is_kraken_tokenized_asset(sym_s):
                    base_t = canonical_base_asset(sym_s)
                    yahoo_underlying_lbl = f"yahoo:{base_t}_underlying"
                    if plan.mode == "incremental" and base_t and existing_src == yahoo_underlying_lbl:
                        fetched_bars = _fetch_yahoo_daily_bars(
                            base_t, lookback_days=plan_days, timeout_sec=timeout_sec, fetch_fn=eff_fetch
                        )
                        src_lbl = yahoo_underlying_lbl
                    else:
                        fetched_bars = _fetch_kraken_daily_bars(
                            sym_s, start_ms=plan.fetch_start_ms, timeout_sec=timeout_sec, fetch_fn=eff_fetch
                        )
                        src_lbl = f"kraken:{sym_s}"
                        if plan.mode != "incremental" and max(len(fetched_bars), existing_cnt) < 30 and base_t:
                            y_bars = _fetch_yahoo_daily_bars(
                                base_t, lookback_days=plan_days, timeout_sec=timeout_sec, fetch_fn=eff_fetch
                            )
                            if len(y_bars) > len(fetched_bars):
                                fetched_bars = y_bars
                                src_lbl = yahoo_underlying_lbl
                else:
                    ven = (venue_for_symbol(sym_s) or "").strip().lower()
                    fetcher = _CRYPTO_VENUE_DAILY_FETCHERS.get(ven, _fetch_binance_venue_daily_bars)
                    fetched_bars, src_lbl = fetcher(
                        sym_s,
                        plan=plan,
                        existing_doc=existing_doc,
                        timeout_sec=timeout_sec,
                        fetch_fn=eff_fetch,
                        post_fn=eff_post,
                    )
            except Exception as exc:
                logger.debug("Long-horizon fetch failed for %s: %s", sym_s, exc)

            incoming_is_fallback = False
            if len(fetched_bars) < 5 and existing_cnt < 5:
                local_bars = _load_sparse_local_daily_bars(cachedb_dir, sym_s)
                if len(local_bars) > len(fetched_bars):
                    fetched_bars = local_bars
                    src_lbl = "local_sparse"
                    incoming_is_fallback = True

            if fetched_bars:
                archive_doc = upsert_ohlc_bar_archive(
                    cachedb_dir,
                    sym_s,
                    fetched_bars,
                    timeframe=_ARCHIVE_TIMEFRAME,
                    source=src_lbl,
                    requested_from_ms=None if incoming_is_fallback else needed_start_ms,
                    now_ts=eff_now,
                    incoming_is_fallback=incoming_is_fallback,
                    persist=persist,
                )
            else:
                archive_doc = existing_doc

            merged_bars = archive_doc.get("bars") if isinstance(archive_doc, dict) else []
            eff_source = str(archive_doc.get("source") or src_lbl) if isinstance(archive_doc, dict) else src_lbl

            if not merged_bars and isinstance(prev_snap, dict) and isinstance(prev_snap.get("items"), dict):
                prev_item = prev_snap["items"].get(sym_s)
                if isinstance(prev_item, dict):
                    items_out[sym_s] = dict(prev_item)
                    continue

            evaluated = evaluate_symbol_long_horizon(
                sym_s,
                merged_bars,
                months=eff_months,
                now_ts=eff_now,
                source=eff_source,
                include_bars=False,
            )
            if evaluated is not None:
                evaluated["bar_archive"] = cache_ohlc_bars_filename(sym_s, timeframe=_ARCHIVE_TIMEFRAME)
                items_out[sym_s] = evaluated

        if not items_out:
            return prev_snap

        payload: Dict[str, Any] = {
            "ts": round(eff_now, 3),
            "months": eff_months,
            "horizon_label": format_horizon_label(eff_months),
            "symbol_count": len(items_out),
            "items": items_out,
        }
        if persist:
            out_path = str(resolve_cachedb_filepath(os.path.join(cachedb_dir, CACHE_LONG_HORIZON_FILENAME)))
            atomic_write_json(out_path, payload)
        return payload


TelemetryCollectorFactory.register(LongHorizonStructureCollector.COLLECTOR_ID, LongHorizonStructureCollector)


def collect_long_horizon_snapshot(
    cache_dir: str = "cachedb",
    *,
    months: Optional[int] = None,
    symbols: Optional[Sequence[str]] = None,
    now_ts: Optional[float] = None,
    timeout_sec: float = 5.0,
    persist: bool = True,
    skip_if_fresh_sec: float = 0.0,
    fetch_fn: Optional[Callable[..., Any]] = None,
    post_fn: Optional[Callable[..., Any]] = None,
    workspace_dir: str = ".",
) -> Optional[Dict[str, Any]]:
    """Incrementally update durable 1d OHLCV archives and persist summary to ``cachedb/macro/long_horizon_collect.json``."""
    return TelemetryCollectorFactory.create("long_horizon_structure", start_sync=True).collect(
        cache_dir,
        "*",
        months=months,
        symbols=symbols,
        now_ts=now_ts,
        timeout_sec=timeout_sec,
        persist=persist,
        skip_if_fresh_sec=skip_if_fresh_sec,
        fetch_fn=fetch_fn,
        post_fn=post_fn,
        workspace_dir=workspace_dir,
    )


def load_active_operator_regime_note(
    workspace_dir: str = ".",
    *,
    now_ts: Optional[float] = None,
    default_ttl_days: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Read optional ``statedb/operator_regime_note.txt`` and return parsed payload only when non-empty and unexpired."""
    ws = str(workspace_dir or ".").strip() or "."
    note_path = os.path.join(ws, "statedb", OPERATOR_REGIME_NOTE_FILENAME)
    if not os.path.exists(note_path):
        return None
    try:
        mtime_ts = os.path.getmtime(note_path)
    except OSError:
        return None
    raw_txt = read_text_file(note_path, default="")
    if not raw_txt:
        return None
    parsed = parse_operator_regime_note_text(
        raw_txt,
        mtime_ts=mtime_ts,
        now_ts=now_ts,
        default_ttl_days=default_ttl_days,
    )
    if parsed is not None:
        parsed["path"] = note_path
    return parsed
