
import os
import time
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np

# Local imports.
from botcore import (
    _parse_bool_flag,
    is_live_network_backfill_enabled,
    load_dotenv,
    lru_dict_put,
    normalize_epoch_sec,
    require_int,  # noqa: F401
    required_bool_env,
    required_int_env,
    safe_pos_float,
    safe_pos_int,
    touch_module_heartbeat,
)
from cachepaths import cache_path, cache_symbol_variants
import cacheManager as cm
from intelligence.internal.state.persistence import (
    MIN_POINTS_PER_WINDOW,
    _fast_linear_regression_1d,
    _trend_memo_cache,
    calculate_hurst_exponent,
    calculate_mann_kendall,
    classify_hurst_regime,
    detect_long_term_trend,
)
from intelligence.internal.state.survival import (
    estimate_T,
    estimate_remaining_trend_hours,
    fetch_klines,
    format_duration,
    format_timestamp,
    get_trade_weight,
    get_weight_for_cash_permission_at_quant_time,
    last_calc_time,
    last_duration,
    last_timestamp,
    last_w,
)
from instrument_registry import casefold_get, is_equity_symbol, tracked_price_instruments
from state_io import atomic_write_json, save_json_state

_CONFIG_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv()

# Intentionally retained dead code for reference: an alternative shared-memory
# IPC approach for trends instead of the current priceanalysis.json file.
#   from multiprocessing import shared_memory
#   import shmutils as shmu
#   shm = shmu.shmConnectForWrite(shmu.shmname)   # At startup in __main__.
#   shmu.shmWrite(shm, all_trends)                # In the loop after write_all_trends.
#   ... finally: shm.close(); shm.unlink()        # At shutdown.
# ------------------------------------------------------------------------------

# Debug tracing for weight zones and slope tuning is off by default to keep fleet
# logs clean. Enable PRICEANALYSIS_DEBUG when tuning.
_DEBUG = required_bool_env("PRICEANALYSIS_DEBUG")
PRICEANALYSIS_DRAW_MAX_POINTS = required_int_env("PRICEANALYSIS_DRAW_MAX_POINTS")


price_cache_manager = None

def build_price_cache_manager(symbols=None, provider_names=None):
    global price_cache_manager
    if provider_names:
        cm.get_current_price_manager(
            symbols=symbols,
            provider_names=provider_names,
        )
    price_cache_manager = cm.get_cache_manager(
        "SparsePrice",
        symbols=symbols,
    )  # Dictionary keyed by symbol.

def priceLstFor(symbol: str) -> List[Tuple[int, float]]:

    if price_cache_manager is None:
        raise RuntimeError("The price cache manager was not initialised. Run build_price_cache_manager() first.")

    manager = casefold_get(price_cache_manager, symbol)
    if manager is None:
        return []

    if hasattr(manager, "resync_mem_file"):
        manager.resync_mem_file()

    # Read the symbol's current cached series under the manager lock when present.
    lock = getattr(manager, "lock", None)
    if lock is not None:
        with lock:
            raw = list(casefold_get(manager.cache, symbol, []) or [])
    else:
        raw = list(casefold_get(manager.cache, symbol, []) or [])
    out: List[Tuple[int, float]] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            ts = safe_pos_int(item[0], default=None)
            p = safe_pos_float(item[1], default=None)
            if ts is not None and p is not None:
                out.append((ts, p))
    return out


_draw_fig = {}  # Reused symbol-to-(figure, axis) mapping; see drawPriceLst.


def _bounded_plot_indices(length: int, max_points: Optional[int] = None) -> np.ndarray:
    """Return uniform plot indices with retained endpoints and bounded memory."""
    if length <= 0:
        return np.empty(0, dtype=np.intp)
    limit = max(2, PRICEANALYSIS_DRAW_MAX_POINTS if max_points is None else int(max_points))
    if length <= limit:
        return np.arange(length, dtype=np.intp)
    return np.linspace(0, length - 1, num=limit, dtype=np.intp)


def drawPriceLst(timestamps, prices, trend_block_indices, symbol, trend_direction, duration_hours):
    """Draw using one persistent figure and axis per symbol.

    Creating and closing a figure every minute leaked about 1.17 MB RSS per call
    because Matplotlib/Agg font and rendering caches survive repeated figures.
    Clearing and redrawing one persistent figure avoids that allocator growth.

    Trend calculations use the complete series. Only the PNG renderer receives a
    bounded uniform sample; otherwise Matplotlib creates tens of thousands of datetime
    objects and array copies every minute, and the allocator retains their RSS."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    timestamps = np.asarray(timestamps)
    prices = np.asarray(prices)
    plot_idx = _bounded_plot_indices(len(timestamps))
    times = [datetime.fromtimestamp(float(timestamps[i])) for i in plot_idx]

    if symbol not in _draw_fig:
        _draw_fig[symbol] = plt.subplots(figsize=(12, 5))
    fig, ax = _draw_fig[symbol]
    ax.clear()

    ax.plot(times, prices[plot_idx], label='Price', color='blue')
    for start, end in trend_block_indices:
        # Reuse the global sample to bound the total number of plot objects even when
        # trend blocks overlap.
        block_idx = plot_idx[(plot_idx >= start) & (plot_idx < end)]
        if block_idx.size < 2 and end > start:
            block_idx = np.unique(np.array([start, end - 1], dtype=np.intp))
        if block_idx.size:
            block_times = [datetime.fromtimestamp(float(timestamps[i])) for i in block_idx]
            ax.plot(block_times, prices[block_idx], color='red', linewidth=2)

    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.set_title(f"{symbol} - Trend {trend_direction}, duration {duration_hours:.2f}h")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    fig.autofmt_xdate()

    out_dir = os.path.join(_CONFIG_ROOT, "logger", "plots")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"plot_{symbol}.png"))


_ORIG_PRICE_LST_FOR = priceLstFor
_OHLC_BACKFILL_MEMO: dict[str, Tuple[float, List[Tuple[float, float]]]] = {}
_OHLC_BACKFILL_TTL_SEC = 1800.0


def _interpolate_price_gaps(
    points: List[Tuple[float, float]],
    max_gap_s: float = 7200.0,
    step_s: float = 180.0,
) -> List[Tuple[float, float]]:
    """Linearly interpolate gaps in ``(step_s, max_gap_s]`` at ``step_s`` cadence."""
    if len(points) < 2:
        return list(points)
    out: List[Tuple[float, float]] = [points[0]]
    for t_cur, p_cur in points[1:]:
        t_prev, p_prev = out[-1]
        dt = t_cur - t_prev
        if step_s < dt <= max_gap_s:
            n_steps = int(round(dt / step_s))
            for step_k in range(1, n_steps):
                frac = step_k / n_steps
                out.append((t_prev + frac * dt, p_prev + frac * (p_cur - p_prev)))
        out.append((t_cur, p_cur))
    return out


def _backfill_ohlc_series(symbol: str, lookback_days: float = 30.0) -> List[Tuple[float, float]]:
    """Fetch and memoize 3-minute-resampled OHLC points for symbols whose local sparse price cache spans < 7 days."""
    sym_u = str(symbol or "").strip().upper()
    if not sym_u:
        return []
    now_s = time.time()
    cached = _OHLC_BACKFILL_MEMO.get(sym_u)
    if cached is not None:
        ttl_s = _OHLC_BACKFILL_TTL_SEC if cached[1] else 60.0
        if 0.0 <= (now_s - cached[0]) < ttl_s:
            return cached[1]
    days_int = max(30, int(round(safe_pos_float(lookback_days, 30.0) or 30.0)) + 3)
    is_eq = is_equity_symbol(sym_u)
    cands = (
        [sym_u]
        if is_eq
        else cache_symbol_variants(sym_u, include_original=True, include_base=False, quotes=("USDC", "USDT", "USD"))
    )
    raw_pts: List[Tuple[float, float]] = []
    for cand in cands:
        for iv in ("15m", "1h"):
            try:
                ts_arr, px_arr = fetch_klines(cand, days=days_int, interval=iv)
                if len(ts_arr) >= 48 and len(ts_arr) == len(px_arr):
                    raw_pts = [(float(t), float(p)) for t, p in zip(ts_arr, px_arr) if t > 0 and p > 0]
                    if raw_pts:
                        break
            except Exception:
                continue
        if raw_pts:
            break
    pts = _interpolate_price_gaps(raw_pts, max_gap_s=345600.0 if is_eq else 7200.0, step_s=180.0)
    lru_dict_put(_OHLC_BACKFILL_MEMO, sym_u, (now_s, pts), 64)
    return pts


def getTrendLongTerm_fixed(symbol: str, window_hours: int = 24, step_hours: int = 8,
                           min_consecutive_blocks: int = 3,
                           noise_tolerance: int = 2,  # Allow two noise blocks.
                           lookback_days: int = 30,
                           draw: bool = True,
                           min_points_per_window: int = MIN_POINTS_PER_WINDOW,
                           detection_lag_hours: float = 48.0,
                           mk_alpha: float = 0.05) -> Optional[dict]:
    # detection_lag_hours models the empirical delay between trend start and
    # detector confirmation. mk_alpha requires current-window Mann-Kendall
    # significance; None disables that filter.
    lookback_days = safe_pos_float(lookback_days, 30.0)
    data: List[Tuple[int, float]] = priceLstFor(symbol)
    norm_data = [
        (tf, pf)
        for ts, p in data
        if (tf := normalize_epoch_sec(ts, default=None)) is not None
        and (pf := safe_pos_float(p, default=None)) is not None
    ]
    norm_data.sort(key=lambda x: x[0])
    is_eq = is_equity_symbol(symbol)
    min_req_span_s = min((safe_pos_float(lookback_days, 30.0) or 30.0) * 0.5, 7.0) * 86400.0
    cur_span_s = (norm_data[-1][0] - norm_data[0][0]) if len(norm_data) >= 2 else 0.0
    if cur_span_s < min_req_span_s and priceLstFor is _ORIG_PRICE_LST_FOR and price_cache_manager is not None:
        oldest_ts = norm_data[0][0] if norm_data else float("inf")
        anchor_px = norm_data[0][1] if norm_data else None
        if not is_eq:
            for sib in cache_symbol_variants(symbol, include_original=False, include_base=False, quotes=("USDC", "USDT", "USD")):
                sib_raw = priceLstFor(sib)
                if len(sib_raw) >= 2:
                    sib_norm = sorted(
                        (
                            (tf, pf)
                            for ts, p in sib_raw
                            if (tf := normalize_epoch_sec(ts, default=None)) is not None
                            and (pf := safe_pos_float(p, default=None)) is not None
                        ),
                        key=lambda x: x[0],
                    )
                    older_pts = [pt for pt in sib_norm if pt[0] < oldest_ts - 1.0]
                    if older_pts and (
                        anchor_px is None
                        or abs(older_pts[-1][1] - anchor_px) / max(older_pts[-1][1], anchor_px) <= 0.10
                    ):
                        norm_data = older_pts + norm_data
                        oldest_ts = norm_data[0][0]
                        anchor_px = norm_data[0][1]
                        cur_span_s = norm_data[-1][0] - norm_data[0][0]
                        if cur_span_s >= min_req_span_s:
                            break
        if cur_span_s < min_req_span_s and is_live_network_backfill_enabled():
            ohlc_pts = _backfill_ohlc_series(symbol, lookback_days=safe_pos_float(lookback_days, 30.0) or 30.0)
            if ohlc_pts:
                older_ohlc = [pt for pt in ohlc_pts if pt[0] < oldest_ts - 1.0]
                if older_ohlc and (
                    anchor_px is None
                    or abs(older_ohlc[-1][1] - anchor_px) / max(older_ohlc[-1][1], anchor_px) <= 0.10
                ):
                    norm_data = older_ohlc + norm_data
    if (
        is_eq
        and len(norm_data) >= 2
        and priceLstFor is _ORIG_PRICE_LST_FOR
        and price_cache_manager is not None
    ):
        norm_data = _interpolate_price_gaps(norm_data, max_gap_s=345600.0, step_s=180.0)
    if len(norm_data) < 2:
        return None
    if lookback_days is not None:
        cutoff_timestamp = time.time() - (lookback_days * 86400)
        norm_data = [(ts, p) for ts, p in norm_data if ts > cutoff_timestamp]
        if len(norm_data) < 2:
            print(f"[{symbol}] Not enough data in the last {lookback_days} days")
            return None
    raw_ts, raw_px = zip(*norm_data)
    timestamps = np.array(raw_ts, dtype=float)
    prices = np.array(raw_px, dtype=float)

    # Use time-defined windows to tolerate uneven density while respecting gaps.
    res = detect_long_term_trend(
        timestamps, prices, window_hours=window_hours, step_hours=step_hours,
        min_consecutive_blocks=min_consecutive_blocks, noise_tolerance=noise_tolerance,
        min_points_per_window=min_points_per_window,
        detection_lag_hours=detection_lag_hours, mk_alpha=mk_alpha)

    if res is None:
        print(f"[{symbol}] Trend undecidable (not enough data, a gap, or MK-insignificant).")
        return None

    # Hurst regime is informational: persistence favors trend following, while
    # mean reversion suggests trends decay quickly.
    h = calculate_hurst_exponent(prices)
    res['hurst'] = h
    res['regime'] = classify_hurst_regime(h)
    print(f"[{symbol}] Hurst={h:.2f} ({res['regime']})" if h else f"[{symbol}] Hurst: series too short")

    direction = res['direction']
    emoji = "📈" if direction == 'up' else "📉"
    dur = res['duration_seconds']
    print(f"\n{'='*60}")
    print(f"[{symbol}] Trend {emoji} {direction.upper()} | slope/h={res['current_slope_h']:.4f}")
    print(f"  Points (last {lookback_days}d): {len(prices)} | window={window_hours}h "
          f"step={step_hours}h | blocks={len(res['blocks'])}")
    print(f"  Start: {format_timestamp(res['start_timestamp'])} | "
          f"Duration: {format_duration(dur)} ({dur/86400:.1f} days)")
    print(f"{'='*60}\n")

    if _parse_bool_flag(draw, True):
        drawPriceLst(timestamps, prices, res['blocks'], symbol, direction, dur / 3600.0)

    est_future_h = estimate_remaining_trend_hours(dur / 3600.0, symbol=symbol, allow_network=True)
    return {
        'timestamp': int(time.time()),
        'direction': direction,
        'start_timestamp': res['start_timestamp'],
        'duration_seconds': dur,
        'estimated_future_hours': est_future_h if est_future_h > 0.0 else res['estimated_future_hours'],
    }

# Persist all trend results with human-readable output.
def write_all_trends(all_trends, filename=None):
    """Atomically write trend results and display a human-readable summary."""
    filename = filename or cache_path("priceanalysis.json")
    if not isinstance(all_trends, dict):
        all_trends = {}

    print("\n" + "="*80)
    print("TREND SUMMARY".center(80))
    print("="*80)
    
    for symbol, trend_data in all_trends.items():
        if not isinstance(trend_data, dict):
            print(f"\n{symbol}: ❌ Not enough data")
            continue
            
        direction = str(trend_data.get('direction') or 'down').strip().lower()
        emoji = "📈" if direction == 'up' else "📉"
        
        start_str = format_timestamp(trend_data.get('start_timestamp', 0))
        dur_sec = safe_pos_float(trend_data.get('duration_seconds', 0), 0.0)
        duration_str = format_duration(dur_sec)
        duration_days = dur_sec / 86400
        
        future_hours = safe_pos_float(trend_data.get('estimated_future_hours', 0), 0.0)
        future_str = format_duration(future_hours * 3600)
        future_days = future_hours / 24
        
        print(f"\n{symbol}")
        print(f"  {emoji} {direction.upper()}")
        print(f"  Start:    {start_str}")
        print(f"  Duration: {duration_str} ({duration_days:.1f} days)")
        print(f"  Estimate: ~{future_str} ({future_days:.1f} days)")
    
    print("\n" + "="*80 + "\n")
    
    try:
        save_json_state(filename, all_trends, indent=2, write_fn=atomic_write_json)
        print(f"Results were written to {filename}")
    except Exception as exc:
        print(f"Could not write {filename}: {exc}")
    
    return all_trends


UPDATE_AND_REFRESH_TREND = 60 * 1  # One minute.
if __name__ == "__main__":
    import psutil
    symbols, _nb_syms, _nb_providers = tracked_price_instruments()
    build_price_cache_manager(symbols=symbols, provider_names=_nb_providers or None)
    touch_module_heartbeat(
        "priceAnalysis",
        interval_sec=30.0,
        force=True,
        symbols=list(symbols),
        intervals={"refresh_trend": UPDATE_AND_REFRESH_TREND},
    )
    try:
        while True:
            process = psutil.Process(os.getpid())
            print("Memory used (MB):", process.memory_info().rss / 1024**2)
            
            all_trends = {}
            for symbol in symbols:
                all_trends[symbol] = getTrendLongTerm_fixed(symbol,
                                            window_hours=16,
                                            step_hours=8,
                                            min_consecutive_blocks=3,
                                            noise_tolerance=2,  # Allow two UP blocks within a DOWN trend.
                                            lookback_days=30,
                                            draw=True)
            write_all_trends(all_trends);
            touch_module_heartbeat(
                "priceAnalysis",
                interval_sec=30.0,
                status={"trends": all_trends},
                symbols=list(symbols),
                intervals={"refresh_trend": UPDATE_AND_REFRESH_TREND},
            )

            print(f"write : {all_trends}")
            time.sleep(UPDATE_AND_REFRESH_TREND)
    except KeyboardInterrupt:
        print("Manual shutdown...")


######################
