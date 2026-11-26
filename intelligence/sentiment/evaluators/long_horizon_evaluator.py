"""Pure deterministic evaluator for configurable multi-month/annual structure, operator strategic notes, and ASCII chart zoom.

Zero disk or network I/O. All percentages and ranges can be recalculated live against spot prices
at prompt-render time so Section 2 long-horizon context always agrees mathematically with Section 3.
"""
from __future__ import annotations

import calendar
from datetime import datetime, timezone
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from botcore import (
    fmt_signed,
    optional_env,
    resolve_now_ts,
    safe_int,
    safe_nonneg_float,
    safe_pos_float,
)
from cachepaths import canonical_base_asset, is_kraken_tokenized_asset
from instrument_registry import is_equity_or_tokenized_symbol, is_equity_symbol
from intelligence.sentiment.evaluators.price_series_evaluator import TelemetryEvaluatorFactory
from intelligence.sentiment.evaluators.prompt_view_evaluator import PromptViewEvaluatorBase

_DAYS_PER_MONTH = 30.4375
_MS_PER_DAY = 86_400_000.0
_NOTE_TTL_RE = re.compile(r"^\s*TTL\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*([dhDH])?\s*$")
_NOTE_EXPIRES_RE = re.compile(r"^\s*EXPIRES\s*:\s*(\d{4}-\d{2}-\d{2})(?:[T\s](\d{2}:\d{2}))?\s*$", re.IGNORECASE)
_YM_RE = re.compile(r"^(\d{4})-(\d{2})(?:-(\d{2}))?$")


def resolve_long_horizon_months(months: Optional[int] = None) -> int:
    """Resolve configurable long-horizon lookback window in months from explicit arg or AI_LONG_HORIZON_MONTHS."""
    if months is not None:
        val = safe_int(months, default=0)
    else:
        raw = optional_env("AI_LONG_HORIZON_MONTHS", "")
        val = safe_int(raw, default=36) if str(raw or "").strip() else 36
    return max(1, min(60, val if val > 0 else 36))


def resolve_operator_note_ttl_days(ttl_days: Optional[float] = None) -> float:
    """Resolve default operator strategic note TTL in days from explicit arg or AI_OPERATOR_NOTE_TTL_DAYS."""
    if ttl_days is not None:
        val = safe_pos_float(ttl_days, default=None)
    else:
        raw = optional_env("AI_OPERATOR_NOTE_TTL_DAYS", "")
        val = safe_pos_float(raw, default=5.0) if str(raw or "").strip() else 5.0
    return max(0.25, min(90.0, val if val is not None else 5.0))


def format_horizon_label(months: int) -> str:
    """Format a compact human-readable horizon label (e.g. 36 -> '3y', 24 -> '2y', 18 -> '18m')."""
    m = max(1, safe_int(months, default=36))
    if m % 12 == 0:
        return f"{m // 12}y"
    return f"{m}m"


def _fmt_compact_px(px: Any) -> str:
    v = safe_pos_float(px, default=None)
    if v is None:
        return "-"
    if v >= 10_000.0:
        return f"{v / 1000.0:.1f}K"
    if v >= 1_000.0:
        return f"{v:,.0f}"
    if v >= 100.0:
        return f"{v:.1f}"
    if v >= 1.0:
        return f"{v:.2f}"
    return f"{v:.4f}"


def _fmt_range_px(px: Any) -> str:
    v = safe_pos_float(px, default=None)
    if v is None:
        return "-"
    if v >= 10_000.0:
        return f"${v / 1000.0:.1f}K"
    if v >= 1_000.0:
        return f"${v:,.0f}"
    if v >= 10.0:
        return f"${v:.2f}"
    if v >= 1.0:
        return f"${v:.3f}"
    return f"${v:.4f}"


def _normalize_daily_bars(raw_bars: Sequence[Sequence[Any]]) -> List[List[float]]:
    """Normalize and deduplicate daily bars into sorted ``[ts_ms, open, high, low, close, vol]`` rows."""
    by_day: Dict[str, List[float]] = {}
    for row in raw_bars or ():
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        ts_raw = safe_pos_float(row[0], default=None)
        if ts_raw is None:
            continue
        ts_ms = ts_raw if ts_raw > 1e11 else ts_raw * 1000.0
        if len(row) >= 5:
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
        day_key = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
        prev = by_day.get(day_key)
        if prev is None:
            by_day[day_key] = [round(ts_ms, 0), o_f, h_f, l_f, c_v, v_v]
        else:
            prev[0] = max(prev[0], round(ts_ms, 0))
            prev[2] = max(prev[2], h_f)
            prev[3] = min(prev[3], l_f)
            prev[4] = c_v
            prev[5] = prev[5] + v_v
    return [by_day[k] for k in sorted(by_day.keys())]


def compute_zigzag_swings(
    daily_bars: Sequence[Sequence[float]],
    *,
    threshold_pct: float = 15.0,
    max_pivots: int = 8,
    preserve_intraday: bool = False,
) -> List[Dict[str, Any]]:
    """Extract major alternating high/low ZigZag pivots from OHLC bars without temporal aliasing."""
    if preserve_intraday:
        bars = [
            [
                (t_v if t_v > 1e11 else t_v * 1000.0),
                safe_pos_float(b[1], 0.0) or 0.0,
                safe_pos_float(b[2], 0.0) or 0.0,
                safe_pos_float(b[3], 0.0) or 0.0,
                safe_pos_float(b[4], 0.0) or 0.0,
                safe_nonneg_float(b[5], 0.0) if len(b) > 5 else 0.0,
            ]
            for b in (daily_bars or ())
            if isinstance(b, (list, tuple))
            and len(b) >= 5
            and (t_v := safe_pos_float(b[0], default=None)) is not None
        ]
        bars.sort(key=lambda r: r[0])
    else:
        bars = _normalize_daily_bars(daily_bars)
    if len(bars) < 2:
        return []

    lbl_fmt = "%m-%d" if preserve_intraday else "%y-%m"

    def _run_zigzag(thr_pct: float) -> List[Dict[str, Any]]:
        thr = max(1.0, thr_pct) / 100.0
        pivots: List[Dict[str, Any]] = []
        trend = 0  # 1 = searching peak, -1 = searching trough
        ext_hi = bars[0][2]
        ext_hi_ts = bars[0][0]
        ext_lo = bars[0][3]
        ext_lo_ts = bars[0][0]

        def _ym(ts_ms: float) -> str:
            return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime(lbl_fmt)

        for bar in bars[1:]:
            ts_ms, _, h_v, l_v, _, _ = bar
            if trend == 0:
                if h_v > ext_hi:
                    ext_hi, ext_hi_ts = h_v, ts_ms
                if l_v < ext_lo:
                    ext_lo, ext_lo_ts = l_v, ts_ms
                if ext_lo > 0.0 and (h_v - ext_lo) / ext_lo >= thr:
                    pivots.append({"ym": _ym(ext_lo_ts), "px": round(ext_lo, 6), "kind": "L", "ts_ms": ext_lo_ts})
                    trend = 1
                    ext_hi, ext_hi_ts = h_v, ts_ms
                elif ext_hi > 0.0 and (ext_hi - l_v) / ext_hi >= thr:
                    pivots.append({"ym": _ym(ext_hi_ts), "px": round(ext_hi, 6), "kind": "H", "ts_ms": ext_hi_ts})
                    trend = -1
                    ext_lo, ext_lo_ts = l_v, ts_ms
            elif trend == 1:
                if h_v >= ext_hi:
                    ext_hi, ext_hi_ts = h_v, ts_ms
                elif ext_hi > 0.0 and (ext_hi - l_v) / ext_hi >= thr:
                    pivots.append({"ym": _ym(ext_hi_ts), "px": round(ext_hi, 6), "kind": "H", "ts_ms": ext_hi_ts})
                    trend = -1
                    ext_lo, ext_lo_ts = l_v, ts_ms
            else:
                if l_v <= ext_lo:
                    ext_lo, ext_lo_ts = l_v, ts_ms
                elif ext_lo > 0.0 and (h_v - ext_lo) / ext_lo >= thr:
                    pivots.append({"ym": _ym(ext_lo_ts), "px": round(ext_lo, 6), "kind": "L", "ts_ms": ext_lo_ts})
                    trend = 1
                    ext_hi, ext_hi_ts = h_v, ts_ms

        if trend == 1:
            pivots.append({"ym": _ym(ext_hi_ts), "px": round(ext_hi, 6), "kind": "H", "ts_ms": ext_hi_ts})
        elif trend == -1:
            pivots.append({"ym": _ym(ext_lo_ts), "px": round(ext_lo, 6), "kind": "L", "ts_ms": ext_lo_ts})
        else:
            pivots.append({"ym": _ym(bars[0][0]), "px": round(bars[0][4], 6), "kind": "S", "ts_ms": bars[0][0]})
        return pivots

    pivots = _run_zigzag(threshold_pct)
    if len(pivots) < 3 and threshold_pct > 8.0:
        pivots = _run_zigzag(8.0)
    if len(pivots) > max_pivots:
        for step_thr in (20.0, 25.0, 32.0, 42.0, 55.0, 75.0):
            if step_thr <= threshold_pct:
                continue
            cand = _run_zigzag(step_thr)
            if len(cand) >= 2:
                pivots = cand
            if len(pivots) <= max_pivots:
                break
    if len(pivots) > max_pivots:
        pivots = pivots[-max_pivots:]
    return [{"ym": p["ym"], "px": p["px"], "kind": p["kind"]} for p in pivots]


def aggregate_weekly_bars(daily_bars: Sequence[Sequence[float]]) -> List[List[float]]:
    """Aggregate normalized daily OHLCV bars into ISO-week ``[ts_ms, open, high, low, close, vol]`` bars."""
    bars = _normalize_daily_bars(daily_bars)
    weeks: Dict[Tuple[int, int], List[float]] = {}
    for ts_ms, o_v, h_v, l_v, c_v, v_v in bars:
        dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        iso = dt.isocalendar()
        wk_key = (iso[0], iso[1])
        cur = weeks.get(wk_key)
        if cur is None:
            weeks[wk_key] = [round(ts_ms, 0), round(o_v, 6), round(h_v, 6), round(l_v, 6), round(c_v, 6), round(v_v, 2)]
        else:
            cur[0] = round(ts_ms, 0)
            cur[2] = round(max(cur[2], h_v), 6)
            cur[3] = round(min(cur[3], l_v), 6)
            cur[4] = round(c_v, 6)
            cur[5] = round(cur[5] + v_v, 2)
    return [weeks[k] for k in sorted(weeks.keys())]


def _pct_chg(cur_px: float, ref_px: Optional[float]) -> Optional[float]:
    if ref_px is None or ref_px <= 0.0 or cur_px <= 0.0:
        return None
    return round((cur_px - ref_px) / ref_px * 100.0, 2)


def _range_pos(cur_px: float, lo_px: Optional[float], hi_px: Optional[float]) -> Optional[float]:
    if lo_px is None or hi_px is None or hi_px <= lo_px:
        return None
    return round(max(0.0, min(100.0, (cur_px - lo_px) / (hi_px - lo_px) * 100.0)), 1)


def _find_close_at_or_before(bars: Sequence[Sequence[float]], target_ts_ms: float) -> Optional[float]:
    cand: Optional[float] = None
    for b in bars:
        if b[0] <= target_ts_ms:
            cand = b[4]
        else:
            break
    return cand


def evaluate_symbol_long_horizon(
    symbol: str,
    daily_bars: Sequence[Sequence[Any]],
    *,
    months: Optional[int] = None,
    now_ts: Optional[float] = None,
    live_price: Optional[float] = None,
    source: str = "",
    include_bars: bool = True,
) -> Optional[Dict[str, Any]]:
    """Compute multi-month/annual structural anchors, ZigZag swings, and optional embedded zoom bars for one symbol."""
    all_bars = _normalize_daily_bars(daily_bars)
    if len(all_bars) < 2:
        return None
    eff_months = resolve_long_horizon_months(months)
    eff_now = resolve_now_ts(now_ts)
    last_ts_ms = all_bars[-1][0]
    ref_end_ms = max(last_ts_ms, eff_now * 1000.0)
    horizon_days = round(eff_months * _DAYS_PER_MONTH)
    horizon_cut_ms = ref_end_ms - horizon_days * _MS_PER_DAY
    window_bars = [b for b in all_bars if b[0] >= horizon_cut_ms]
    if len(window_bars) < 2:
        window_bars = list(all_bars)

    last_close = all_bars[-1][4]
    cur_px = safe_pos_float(live_price, default=None) or last_close
    ref_year = datetime.fromtimestamp(ref_end_ms / 1000.0, tz=timezone.utc).year
    jan1_ms = datetime(ref_year, 1, 1, tzinfo=timezone.utc).timestamp() * 1000.0

    # YTD reference: last close before Jan 1 of current year if available, else first bar open in current year
    ytd_ref = _find_close_at_or_before(all_bars, jan1_ms - 1.0)
    if ytd_ref is None:
        for b in all_bars:
            if b[0] >= jan1_ms:
                ytd_ref = b[1]
                break

    ref_30d = _find_close_at_or_before(all_bars, ref_end_ms - 30.0 * _MS_PER_DAY)
    ref_90d = _find_close_at_or_before(all_bars, ref_end_ms - 90.0 * _MS_PER_DAY)
    ref_1y = _find_close_at_or_before(all_bars, ref_end_ms - 365.0 * _MS_PER_DAY)
    ref_horizon = window_bars[0][1]

    bars_52w = [b for b in all_bars if b[0] >= ref_end_ms - 365.0 * _MS_PER_DAY] or list(window_bars)
    low_52w = min(b[3] for b in bars_52w)
    high_52w = max(b[2] for b in bars_52w)
    low_hor = min(b[3] for b in window_bars)
    high_hor = max(b[2] for b in window_bars)

    closes = [b[4] for b in all_bars]
    ma50 = round(sum(closes[-50:]) / 50.0, 6) if len(closes) >= 50 else None
    ma200 = round(sum(closes[-200:]) / 200.0, 6) if len(closes) >= 200 else None

    first_dt = datetime.fromtimestamp(window_bars[0][0] / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
    last_dt = datetime.fromtimestamp(all_bars[-1][0] / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
    span_days = max(1, round((all_bars[-1][0] - window_bars[0][0]) / _MS_PER_DAY))

    swings = compute_zigzag_swings(window_bars, threshold_pct=15.0 if eff_months >= 12 else 10.0, max_pivots=8)
    result: Dict[str, Any] = {
        "symbol": str(symbol).strip(),
        "source": str(source or ""),
        "months": eff_months,
        "horizon_label": format_horizon_label(eff_months),
        "history_days": span_days,
        "bar_count": len(window_bars),
        "first_date": first_dt,
        "last_date": last_dt,
        "last_close": round(last_close, 6),
        "ytd_ref": round(ytd_ref, 6) if ytd_ref is not None else None,
        "ytd_pct": _pct_chg(cur_px, ytd_ref),
        "ref_30d": round(ref_30d, 6) if ref_30d is not None else None,
        "chg_30d_pct": _pct_chg(cur_px, ref_30d),
        "ref_90d": round(ref_90d, 6) if ref_90d is not None else None,
        "chg_90d_pct": _pct_chg(cur_px, ref_90d),
        "ref_1y": round(ref_1y, 6) if ref_1y is not None else None,
        "chg_1y_pct": _pct_chg(cur_px, ref_1y),
        "ref_horizon": round(ref_horizon, 6) if ref_horizon is not None else None,
        "chg_horizon_pct": _pct_chg(cur_px, ref_horizon),
        "low_52w": round(low_52w, 6),
        "high_52w": round(high_52w, 6),
        "pos_52w_pct": _range_pos(cur_px, low_52w, high_52w),
        "dd_52w_high_pct": _pct_chg(cur_px, high_52w),
        "low_horizon": round(low_hor, 6),
        "high_horizon": round(high_hor, 6),
        "pos_horizon_pct": _range_pos(cur_px, low_hor, high_hor),
        "dd_horizon_high_pct": _pct_chg(cur_px, high_hor),
        "ma50": ma50,
        "vs_ma50_pct": _pct_chg(cur_px, ma50),
        "ma200": ma200,
        "vs_ma200_pct": _pct_chg(cur_px, ma200),
        "swings": swings,
    }
    if include_bars:
        result["weekly_bars"] = aggregate_weekly_bars(window_bars)
        result["recent_daily_bars"] = [
            [round(b[0], 0), round(b[1], 6), round(b[2], 6), round(b[3], 6), round(b[4], 6), round(b[5], 2)]
            for b in window_bars[-90:]
        ]
    return result


def recalculate_long_horizon_with_live_price(
    item: Mapping[str, Any],
    live_price: Optional[float],
) -> Dict[str, Any]:
    """Recalculate long-horizon percentages and range positions against live spot price without disk I/O."""
    out = dict(item)
    px = safe_pos_float(live_price, default=None)
    if px is None:
        return out
    out["live_price"] = round(px, 6)
    if (ytd_r := safe_pos_float(item.get("ytd_ref"), default=None)) is not None:
        out["ytd_pct"] = _pct_chg(px, ytd_r)
    if (r30 := safe_pos_float(item.get("ref_30d"), default=None)) is not None:
        out["chg_30d_pct"] = _pct_chg(px, r30)
    if (r90 := safe_pos_float(item.get("ref_90d"), default=None)) is not None:
        out["chg_90d_pct"] = _pct_chg(px, r90)
    if (r1y := safe_pos_float(item.get("ref_1y"), default=None)) is not None:
        out["chg_1y_pct"] = _pct_chg(px, r1y)
    if (rhor := safe_pos_float(item.get("ref_horizon"), default=None)) is not None:
        out["chg_horizon_pct"] = _pct_chg(px, rhor)
    if (l52 := safe_pos_float(item.get("low_52w"), default=None)) is not None and (
        h52 := safe_pos_float(item.get("high_52w"), default=None)
    ) is not None:
        eff_l52 = min(l52, px)
        eff_h52 = max(h52, px)
        out["low_52w"] = round(eff_l52, 6)
        out["high_52w"] = round(eff_h52, 6)
        out["pos_52w_pct"] = _range_pos(px, eff_l52, eff_h52)
        out["dd_52w_high_pct"] = _pct_chg(px, eff_h52)
    if (lhor := safe_pos_float(item.get("low_horizon"), default=None)) is not None and (
        hhor := safe_pos_float(item.get("high_horizon"), default=None)
    ) is not None:
        eff_lh = min(lhor, px)
        eff_hh = max(hhor, px)
        out["low_horizon"] = round(eff_lh, 6)
        out["high_horizon"] = round(eff_hh, 6)
        out["pos_horizon_pct"] = _range_pos(px, eff_lh, eff_hh)
        out["dd_horizon_high_pct"] = _pct_chg(px, eff_hh)
    if (ma50 := safe_pos_float(item.get("ma50"), default=None)) is not None:
        out["vs_ma50_pct"] = _pct_chg(px, ma50)
    if (ma200 := safe_pos_float(item.get("ma200"), default=None)) is not None:
        out["vs_ma200_pct"] = _pct_chg(px, ma200)
    return out


def format_long_horizon_section_lines(
    snapshot: Optional[Mapping[str, Any]],
    *,
    prices_and_trends: Optional[Mapping[str, Any]] = None,
    extra_live_prices: Optional[Mapping[str, Any]] = None,
    asset_scope: str = "all",
    tag: str = "",
) -> List[str]:
    """Render compact Multi-Month / Annual structure table and ZigZag swings for Section 2."""
    if not isinstance(snapshot, Mapping):
        return []
    raw_items = snapshot.get("items")
    if not isinstance(raw_items, Mapping) or not raw_items:
        return []
    scope_s = str(asset_scope or "all").strip().lower()
    is_eq_scope = scope_s in ("equities", "equity")
    pt_map = prices_and_trends if isinstance(prices_and_trends, Mapping) else {}
    ex_map = extra_live_prices if isinstance(extra_live_prices, Mapping) else {}
    eff_months = resolve_long_horizon_months(safe_int(snapshot.get("months"), default=None) if snapshot.get("months") else None)
    hor_lbl = str(snapshot.get("horizon_label") or format_horizon_label(eff_months))

    selected: List[Tuple[str, Dict[str, Any]]] = []
    for sym, item in raw_items.items():
        if not isinstance(item, Mapping):
            continue
        sym_s = str(sym).strip()
        if not sym_s:
            continue
        is_eq = is_equity_symbol(sym_s)
        is_tok = is_kraken_tokenized_asset(sym_s)
        is_bench = sym_s.upper() in ("SPY", "QQQ")
        if is_eq_scope:
            if not (is_eq or is_bench or canonical_base_asset(sym_s) == "BTC"):
                continue
        elif scope_s == "crypto":
            if is_eq or is_bench:
                continue
            if pt_map and not (sym_s in pt_map or canonical_base_asset(sym_s) == "BTC" or is_tok):
                continue
        live_px = None
        if isinstance(pt_map.get(sym_s), Mapping):
            live_px = safe_pos_float(pt_map[sym_s].get("price"), default=None)
        if live_px is None and ex_map:
            live_px = safe_pos_float(ex_map.get(sym_s), default=None) or safe_pos_float(
                ex_map.get(sym_s.upper()), default=None
            )
            if live_px is None and not is_eq and not is_tok:
                live_px = safe_pos_float(ex_map.get(canonical_base_asset(sym_s)), default=None)
        rec = recalculate_long_horizon_with_live_price(item, live_px)
        selected.append((sym_s, rec))

    if not selected:
        return []

    def _sort_key(pair: Tuple[str, Dict[str, Any]]) -> Tuple[int, str]:
        s = pair[0]
        if canonical_base_asset(s) == "BTC" and not is_equity_or_tokenized_symbol(s):
            return (0, s)
        if s.upper() in ("SPY", "QQQ"):
            return (1, s)
        return (2, s)

    selected.sort(key=_sort_key)
    show_hor_col = eff_months != 12
    hor_hdr = f" {hor_lbl} Range (Pos) |" if show_hor_col else ""
    hor_sep = "---|" if show_hor_col else ""
    hor_desc = f"{hor_lbl} / {eff_months}m" if hor_lbl != f"{eff_months}m" else f"{eff_months}m"
    out: List[str] = [
        f"  * Multi-Month & Annual Structure ({hor_desc} lookback; YTD, 52w, MA50/MA200 & Major Swings; zoom via verify_tools/chart_view.py){tag}:",
        f"    | Symbol | YTD | 30d | 90d | 52w Range (Pos52w) |{hor_hdr} vs MA50 | vs MA200 | Swings ({hor_lbl}, YY-MM:px) |",
        f"    |---|---|---|---|---|{hor_sep}---|---|---|",
    ]
    for sym_s, rec in selected:
        ytd_s = f"{fmt_signed(rec.get('ytd_pct'), 1)}%" if rec.get("ytd_pct") is not None else "-"
        c30_s = f"{fmt_signed(rec.get('chg_30d_pct'), 1)}%" if rec.get("chg_30d_pct") is not None else "-"
        c90_s = f"{fmt_signed(rec.get('chg_90d_pct'), 1)}%" if rec.get("chg_90d_pct") is not None else "-"
        if rec.get("low_52w") is not None and rec.get("high_52w") is not None:
            p52 = f" ({rec['pos_52w_pct']}%)" if rec.get("pos_52w_pct") is not None else ""
            r52_s = f"[{_fmt_range_px(rec['low_52w'])}..{_fmt_range_px(rec['high_52w'])}]{p52}"
        else:
            r52_s = "-"
        hor_cell = ""
        if show_hor_col:
            if rec.get("low_horizon") is not None and rec.get("high_horizon") is not None:
                ph = f" ({rec['pos_horizon_pct']}%)" if rec.get("pos_horizon_pct") is not None else ""
                hor_cell = f" [{_fmt_range_px(rec['low_horizon'])}..{_fmt_range_px(rec['high_horizon'])}]{ph} |"
            else:
                hor_cell = " - |"
        m50_s = f"{fmt_signed(rec.get('vs_ma50_pct'), 1)}%" if rec.get("vs_ma50_pct") is not None else "-"
        m200_s = f"{fmt_signed(rec.get('vs_ma200_pct'), 1)}%" if rec.get("vs_ma200_pct") is not None else "-"
        swings_raw = rec.get("swings") if isinstance(rec.get("swings"), list) else []
        sw_parts = [f"{sw['ym']}:{_fmt_compact_px(sw.get('px'))}" for sw in swings_raw if isinstance(sw, Mapping) and sw.get("ym")]
        cur_p = rec.get("live_price") or rec.get("last_close")
        if cur_p is not None and sw_parts:
            sw_parts.append(f"now:{_fmt_compact_px(cur_p)}")
        sw_s = "->".join(sw_parts) if sw_parts else "-"
        out.append(
            f"    | {sym_s} | {ytd_s} | {c30_s} | {c90_s} | {r52_s} |{hor_cell} {m50_s} | {m200_s} | {sw_s} |"
        )
    return out


def parse_operator_regime_note_text(
    raw_text: str,
    *,
    mtime_ts: float,
    now_ts: Optional[float] = None,
    default_ttl_days: Optional[float] = None,
    max_chars: int = 480,
) -> Optional[Dict[str, Any]]:
    """Parse optional statedb/operator_regime_note.txt with auto-expiry from mtime (or TTL:/EXPIRES: directive)."""
    if not raw_text or not str(raw_text).strip():
        return None
    eff_now = resolve_now_ts(now_ts)
    eff_mtime = safe_pos_float(mtime_ts, default=eff_now) or eff_now
    ttl_days = resolve_operator_note_ttl_days(default_ttl_days)
    expires_ts = eff_mtime + ttl_days * 86400.0

    body_lines: List[str] = []
    for raw_ln in str(raw_text).splitlines():
        ln = raw_ln.strip()
        if not ln or ln.startswith("#"):
            continue
        if not body_lines:
            if (m_ttl := _NOTE_TTL_RE.match(ln)) is not None:
                val = safe_pos_float(m_ttl.group(1), default=None)
                unit = (m_ttl.group(2) or "d").lower()
                if val is not None:
                    sec = val * 3600.0 if unit == "h" else val * 86400.0
                    expires_ts = eff_mtime + sec
                continue
            if (m_exp := _NOTE_EXPIRES_RE.match(ln)) is not None:
                d_str = m_exp.group(1)
                hm_str = m_exp.group(2) or "23:59"
                try:
                    dt_exp = datetime.strptime(f"{d_str} {hm_str}", "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
                    expires_ts = dt_exp.timestamp()
                    continue
                except ValueError:
                    pass
        body_lines.append(ln)

    if not body_lines:
        return None
    if eff_now > expires_ts:
        return None

    clean_text = " ".join(body_lines)
    if len(clean_text) > max_chars:
        clean_text = clean_text[: max(32, max_chars - 3)].rstrip() + "..."
    age_days = max(0.0, (eff_now - eff_mtime) / 86400.0)
    rem_days = max(0.0, (expires_ts - eff_now) / 86400.0)
    upd_iso = datetime.fromtimestamp(eff_mtime, tz=timezone.utc).strftime("%Y-%m-%d")
    return {
        "text": clean_text,
        "updated_date": upd_iso,
        "mtime_ts": round(eff_mtime, 3),
        "expires_ts": round(expires_ts, 3),
        "age_days": round(age_days, 2),
        "expires_in_days": round(rem_days, 2),
    }


def _parse_ym_bound_ms(raw: Optional[str], *, is_end: bool = False) -> Optional[float]:
    if not raw:
        return None
    m = _YM_RE.match(str(raw).strip())
    if not m:
        return None
    yr = safe_int(m.group(1), default=0)
    mo = safe_int(m.group(2), default=0)
    dy = safe_int(m.group(3), default=0) if m.group(3) else 0
    if yr < 2000 or mo < 1 or mo > 12:
        return None
    if dy > 0:
        max_dy = calendar.monthrange(yr, mo)[1]
        dt = datetime(
            yr,
            mo,
            min(dy, max_dy),
            23 if is_end else 0,
            59 if is_end else 0,
            59 if is_end else 0,
            tzinfo=timezone.utc,
        )
        return dt.timestamp() * 1000.0
    if is_end:
        if mo == 12:
            dt = datetime(yr + 1, 1, 1, tzinfo=timezone.utc)
        else:
            dt = datetime(yr, mo + 1, 1, tzinfo=timezone.utc)
        return dt.timestamp() * 1000.0 - 1.0
    return datetime(yr, mo, 1, tzinfo=timezone.utc).timestamp() * 1000.0


def render_ascii_chart_view(
    item: Mapping[str, Any],
    *,
    daily_bars: Optional[Sequence[Sequence[Any]]] = None,
    bars_override: Optional[Sequence[Sequence[Any]]] = None,
    start_ym: Optional[str] = None,
    end_ym: Optional[str] = None,
    months: Optional[int] = None,
    timeframe: str = "1w",
    width: int = 60,
    height: int = 12,
) -> str:
    """Render an exact ASCII price chart and OHLC summary for any chosen slice (auto, 1w, 1d, 4h, or 1h)."""
    tf = str(timeframe or "auto").strip().lower()
    if tf == "auto":
        has_daily_src = bool(daily_bars) or bool(item.get("recent_daily_bars"))
        req_end_ms = _parse_ym_bound_ms(end_ym, is_end=True)
        req_start_ms = _parse_ym_bound_ms(start_ym, is_end=False)
        short_range = (
            (months is not None and safe_int(months, default=12) <= 4)
            or (
                req_start_ms is not None
                and req_end_ms is not None
                and 0 <= (req_end_ms - req_start_ms) <= 125 * _MS_PER_DAY
            )
        )
        tf = "1d" if (has_daily_src and short_range) else "1w"
    is_intraday = tf in ("1h", "4h")
    if bars_override:
        bars = [
            [
                (t_v if t_v > 1e11 else t_v * 1000.0),
                safe_pos_float(b[1], 0.0) or 0.0,
                safe_pos_float(b[2], 0.0) or 0.0,
                safe_pos_float(b[3], 0.0) or 0.0,
                safe_pos_float(b[4], 0.0) or 0.0,
                safe_nonneg_float(b[5], 0.0) if len(b) > 5 else 0.0,
            ]
            for b in bars_override
            if isinstance(b, (list, tuple))
            and len(b) >= 5
            and (t_v := safe_pos_float(b[0], default=None)) is not None
        ]
        bars.sort(key=lambda r: r[0])
    elif daily_bars:
        norm_daily = _normalize_daily_bars(daily_bars)
        bars = norm_daily if tf == "1d" else aggregate_weekly_bars(norm_daily)
    else:
        raw_bars = item.get("recent_daily_bars") if tf == "1d" else item.get("weekly_bars")
        if not isinstance(raw_bars, list) or not raw_bars:
            raw_bars = item.get("weekly_bars") or item.get("recent_daily_bars") or []
        bars = _normalize_daily_bars(raw_bars) if tf == "1d" else [
            [safe_pos_float(b[0], 0.0) or 0.0, safe_pos_float(b[1], 0.0) or 0.0, safe_pos_float(b[2], 0.0) or 0.0,
             safe_pos_float(b[3], 0.0) or 0.0, safe_pos_float(b[4], 0.0) or 0.0, safe_nonneg_float(b[5], 0.0) if len(b) > 5 else 0.0]
            for b in raw_bars
            if isinstance(b, (list, tuple)) and len(b) >= 5
        ]
    if not bars:
        return f"No long-horizon bars available for {item.get('symbol', '?')}."

    end_ms = _parse_ym_bound_ms(end_ym, is_end=True) or bars[-1][0]
    start_ms = _parse_ym_bound_ms(start_ym, is_end=False)
    if start_ms is None and months is not None:
        m_val = max(1, safe_int(months, default=12))
        start_ms = end_ms - round(m_val * _DAYS_PER_MONTH) * _MS_PER_DAY
    sliced = [b for b in bars if (start_ms is None or b[0] >= start_ms) and b[0] <= end_ms]
    if not sliced:
        return f"No bars in requested window for {item.get('symbol', '?')}."

    date_fmt = "%Y-%m-%d %H:%M" if is_intraday else "%Y-%m-%d"
    d0 = datetime.fromtimestamp(sliced[0][0] / 1000.0, tz=timezone.utc).strftime(date_fmt)
    d1 = datetime.fromtimestamp(sliced[-1][0] / 1000.0, tz=timezone.utc).strftime(date_fmt)
    open_p, close_p = sliced[0][1], sliced[-1][4]
    lo_p = min(b[3] for b in sliced)
    hi_p = max(b[2] for b in sliced)
    chg = _pct_chg(close_p, open_p)
    pos = _range_pos(close_p, lo_p, hi_p)
    sw_thr = 5.0 if tf == "1h" else (8.0 if tf == "4h" else (12.0 if len(sliced) > 20 else 6.0))
    swings = compute_zigzag_swings(sliced, threshold_pct=sw_thr, max_pivots=10, preserve_intraday=is_intraday)
    sw_str = " -> ".join(f"{s['ym']}:{_fmt_compact_px(s['px'])}" for s in swings) if swings else "-"

    w = max(16, min(80, safe_int(width, default=60)))
    h = max(6, min(20, safe_int(height, default=12)))
    n = len(sliced)
    col_buckets: List[Tuple[float, float, float]] = []
    for col in range(w):
        if n > w:
            b0 = safe_int((col * n) / w, default=0)
            b1 = max(b0 + 1, min(n, safe_int(((col + 1) * n) / w, default=b0 + 1)))
            bucket = sliced[b0:b1]
        else:
            idx = min(n - 1, safe_int((col * (n - 1)) / max(1, w - 1), default=0)) if n > 1 else 0
            bucket = [sliced[idx]]
        col_buckets.append((max(b[2] for b in bucket), min(b[3] for b in bucket), bucket[-1][4]))

    span = max(hi_p - lo_p, 1e-12)
    grid = [[" " for _ in range(w)] for _ in range(h)]
    prev_close_row: Optional[int] = None
    for col, (b_hi, b_lo, b_close) in enumerate(col_buckets):
        r_hi = h - 1 - min(h - 1, max(0, safe_int(((b_hi - lo_p) / span) * (h - 1) + 0.5, default=0)))
        r_lo = h - 1 - min(h - 1, max(0, safe_int(((b_lo - lo_p) / span) * (h - 1) + 0.5, default=0)))
        for rr in range(min(r_hi, r_lo), max(r_hi, r_lo) + 1):
            if grid[rr][col] == " ":
                grid[rr][col] = ":"
        row = h - 1 - min(h - 1, max(0, safe_int(((b_close - lo_p) / span) * (h - 1) + 0.5, default=0)))
        grid[row][col] = "*"
        if prev_close_row is not None:
            c_lo, c_hi = min(row, prev_close_row), max(row, prev_close_row)
            for rr in range(c_lo + 1, c_hi):
                if grid[rr][col] in (" ", ":"):
                    grid[rr][col] = "|"
        prev_close_row = row

    chg_s = f"{fmt_signed(chg, 1)}%" if chg is not None else "-"
    ytd_s = f"{fmt_signed(item.get('ytd_pct'), 1)}%" if item.get("ytd_pct") is not None else "-"
    lines = [
        f"=== {item.get('symbol', '?')} ({tf}, {d0} .. {d1}, n={n} bars) ===",
        f"Open={_fmt_range_px(open_p)} Close={_fmt_range_px(close_p)} ({chg_s}) | "
        f"Range=[{_fmt_range_px(lo_p)}..{_fmt_range_px(hi_p)}] Pos={pos}% | "
        f"YTD={ytd_s} | "
        f"MA50={_fmt_range_px(item.get('ma50'))} MA200={_fmt_range_px(item.get('ma200'))}",
        f"Swings: {sw_str}",
        "",
    ]
    for r_idx in range(h):
        lvl = hi_p - (span * r_idx / max(1, h - 1))
        lines.append(f"{_fmt_range_px(lvl):>10s} |{''.join(grid[r_idx])}|")
    lines.append(f"{'':>10s} +{'-' * w}+")
    axis_fmt = "%m-%d %H:%M" if is_intraday else "%Y-%m-%d"
    a0 = datetime.fromtimestamp(sliced[0][0] / 1000.0, tz=timezone.utc).strftime(axis_fmt)
    mid_d = datetime.fromtimestamp(sliced[n // 2][0] / 1000.0, tz=timezone.utc).strftime(axis_fmt)
    a1 = datetime.fromtimestamp(sliced[-1][0] / 1000.0, tz=timezone.utc).strftime(axis_fmt)
    pad = max(1, (w - len(a0) - len(mid_d) - len(a1)) // 2)
    lines.append(f"{'':>11s}{a0}{' ' * pad}{mid_d}{' ' * pad}{a1}")
    return "\n".join(lines)


class LongHorizonStructureEvaluator(PromptViewEvaluatorBase):
    """Concrete evaluator (Level 3) computing configurable multi-month/annual structure."""

    cls_name = "LongHorizonStructureEvaluator"

    def evaluate(
        self,
        symbol: str,
        daily_bars: Sequence[Sequence[Any]],
        *,
        months: Optional[int] = None,
        now_ts: Optional[float] = None,
        live_price: Optional[float] = None,
        source: str = "",
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        _ = kwargs
        return evaluate_symbol_long_horizon(
            symbol,
            daily_bars,
            months=months,
            now_ts=now_ts,
            live_price=live_price,
            source=source,
        )


TelemetryEvaluatorFactory.register("long_horizon_structure", LongHorizonStructureEvaluator)
