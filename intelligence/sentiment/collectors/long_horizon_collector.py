"""Single-writer collector for configurable multi-month/annual daily bars and reader for optional operator strategic notes.

Fetches 1d OHLCV bars over ``AI_LONG_HORIZON_MONTHS`` (default 36m) from public venue/Yahoo endpoints
configured in ``macro_intelligence.conf`` (with passive local ``cache_sparse_price_*.jsonl`` fallback),
computes exact anchors/swings/weekly bars via ``long_horizon_evaluator``, and persists atomically to
``cachedb/macro/long_horizon_collect.json``.
"""
from __future__ import annotations

import logging
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
)
from cachepaths import (
    CACHE_LONG_HORIZON_FILENAME,
    OPERATOR_REGIME_NOTE_FILENAME,
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
from providers.ohlc_utils import extract_yahoo_chart_ohlcv_bars
from state_io import atomic_write_json, parse_json_payload, read_text_file

logger = logging.getLogger("intelligence.sentiment.long_horizon_collector")

_DAYS_PER_MONTH = 30.4375
_MS_PER_DAY = 86_400_000.0
_BENCHMARK_SYMBOLS: Tuple[str, ...] = ("SPY", "QQQ")


def _yahoo_range_for_months(months: int) -> str:
    # Include >=200 trading-day warmup for MA200 even on shorter windows
    if months <= 12:
        return "2y"
    if months <= 24:
        return "2y"
    if months <= 60:
        return "5y"
    return "max"


def _fetch_binance_daily_bars(
    symbol: str,
    *,
    lookback_days: int,
    now_ts: float,
    timeout_sec: float,
    fetch_fn: Callable[..., Any],
) -> List[List[float]]:
    target_bars = max(60, lookback_days + 210)
    end_ms = safe_int(now_ts * 1000.0, default=1)
    collected: List[List[float]] = []
    remaining = target_bars
    for _ in range(3):
        if remaining <= 0 or end_ms <= 0:
            break
        batch_lim = min(1000, remaining)
        url = get_binance_daily_klines_url(symbol, batch_lim, end_ms)
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
        if len(batch) < batch_lim or oldest_ms <= 1:
            break
        end_ms = oldest_ms - 1
        remaining -= len(batch)
    collected.sort(key=lambda r: r[0])
    return collected


def _fetch_kraken_daily_bars(
    pair: str,
    *,
    lookback_days: int,
    now_ts: float,
    timeout_sec: float,
    fetch_fn: Callable[..., Any],
) -> List[List[float]]:
    since_sec = max(0, safe_int(now_ts - (lookback_days + 210) * 86400.0, default=0))
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
    lookback_days: int,
    now_ts: float,
    timeout_sec: float,
    post_fn: Callable[..., Any],
) -> List[List[float]]:
    end_ms = safe_int(now_ts * 1000.0, default=1)
    start_ms = max(0, end_ms - safe_int((lookback_days + 210) * _MS_PER_DAY, default=0))
    payload = {
        "type": "candleSnapshot",
        "req": {"coin": str(coin).strip().upper(), "interval": "1d", "startTime": start_ms, "endTime": end_ms},
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
    months: int,
    timeout_sec: float,
    fetch_fn: Callable[..., Any],
) -> List[List[float]]:
    url = get_yahoo_daily_chart_url(ticker, _yahoo_range_for_months(months))
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


def _fetch_hyperliquid_or_fallback_daily_bars(
    sym_s: str,
    *,
    lookback_days: int,
    now_ts: float,
    timeout_sec: float,
    fetch_fn: Callable[..., Any],
    post_fn: Callable[..., Any],
) -> Tuple[List[List[float]], str]:
    base_c = canonical_base_asset(sym_s)
    bars = _fetch_hyperliquid_daily_bars(
        base_c, lookback_days=lookback_days, now_ts=now_ts, timeout_sec=timeout_sec, post_fn=post_fn
    )
    src_lbl = f"hyperliquid:{base_c}"
    if len(bars) < 30:
        usd_pair = ensure_quote_suffix(base_c, "USD")
        k_bars = _fetch_kraken_daily_bars(
            usd_pair, lookback_days=lookback_days, now_ts=now_ts, timeout_sec=timeout_sec, fetch_fn=fetch_fn
        )
        if len(k_bars) > len(bars):
            return k_bars, f"kraken:{usd_pair}"
    return bars, src_lbl


def _fetch_kraken_venue_daily_bars(
    sym_s: str,
    *,
    lookback_days: int,
    now_ts: float,
    timeout_sec: float,
    fetch_fn: Callable[..., Any],
    post_fn: Callable[..., Any],
) -> Tuple[List[List[float]], str]:
    _ = post_fn
    bars = _fetch_kraken_daily_bars(
        sym_s, lookback_days=lookback_days, now_ts=now_ts, timeout_sec=timeout_sec, fetch_fn=fetch_fn
    )
    return bars, f"kraken:{sym_s}"


def _fetch_binance_venue_daily_bars(
    sym_s: str,
    *,
    lookback_days: int,
    now_ts: float,
    timeout_sec: float,
    fetch_fn: Callable[..., Any],
    post_fn: Callable[..., Any],
) -> Tuple[List[List[float]], str]:
    _ = post_fn
    bars = _fetch_binance_daily_bars(
        sym_s, lookback_days=lookback_days, now_ts=now_ts, timeout_sec=timeout_sec, fetch_fn=fetch_fn
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
    """Concrete collector (Level 3) building ``cachedb/macro/long_horizon_collect.json``."""

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
        ):
            return prev_snap

        eff_fetch = fetch_fn or fetch_public_json
        eff_post = post_fn or fetch_json_post
        target_syms = list(symbols) if symbols else _resolve_default_long_horizon_symbols(workspace_dir)
        lookback_days = round(eff_months * _DAYS_PER_MONTH)
        items_out: Dict[str, Dict[str, Any]] = {}

        for sym in target_syms:
            sym_s = str(sym or "").strip()
            if not sym_s:
                continue
            bars: List[List[float]] = []
            src_lbl = ""
            try:
                if sym_s.upper() in _BENCHMARK_SYMBOLS or is_equity_symbol(sym_s):
                    ticker = strip_equity_suffix(sym_s) if is_equity_symbol(sym_s) else sym_s.upper()
                    bars = _fetch_yahoo_daily_bars(ticker, months=eff_months, timeout_sec=timeout_sec, fetch_fn=eff_fetch)
                    src_lbl = f"yahoo:{ticker}"
                elif is_kraken_tokenized_asset(sym_s):
                    base_t = canonical_base_asset(sym_s)
                    bars = _fetch_kraken_daily_bars(sym_s, lookback_days=lookback_days, now_ts=eff_now, timeout_sec=timeout_sec, fetch_fn=eff_fetch)
                    src_lbl = f"kraken:{sym_s}"
                    if len(bars) < 30 and base_t:
                        y_bars = _fetch_yahoo_daily_bars(base_t, months=eff_months, timeout_sec=timeout_sec, fetch_fn=eff_fetch)
                        if len(y_bars) > len(bars):
                            bars = y_bars
                            src_lbl = f"yahoo:{base_t}_underlying"
                else:
                    ven = (venue_for_symbol(sym_s) or "").strip().lower()
                    fetcher = _CRYPTO_VENUE_DAILY_FETCHERS.get(ven, _fetch_binance_venue_daily_bars)
                    bars, src_lbl = fetcher(
                        sym_s,
                        lookback_days=lookback_days,
                        now_ts=eff_now,
                        timeout_sec=timeout_sec,
                        fetch_fn=eff_fetch,
                        post_fn=eff_post,
                    )
            except Exception as exc:
                logger.debug("Long-horizon fetch failed for %s: %s", sym_s, exc)

            if len(bars) < 5:
                local_bars = _load_sparse_local_daily_bars(cachedb_dir, sym_s)
                if len(local_bars) > len(bars):
                    bars = local_bars
                    src_lbl = "local_sparse"

            if not bars and isinstance(prev_snap, dict) and isinstance(prev_snap.get("items"), dict):
                prev_item = prev_snap["items"].get(sym_s)
                if isinstance(prev_item, dict):
                    items_out[sym_s] = dict(prev_item)
                    continue

            evaluated = evaluate_symbol_long_horizon(
                sym_s,
                bars,
                months=eff_months,
                now_ts=eff_now,
                source=src_lbl,
            )
            if evaluated is not None:
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
    """Fetch configurable multi-month/annual daily bars and persist to ``cachedb/macro/long_horizon_collect.json``."""
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
