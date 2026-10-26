
import os
import math
import time
import psutil

import numpy as np
from typing import List, Dict, Tuple, Optional

# Drawing dependencies.
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Local imports.
import utils as u
import symbols as sym
from botcore import (load_dotenv as _load_dotenv, required_bool_env,
                     required_int_env)
from state_io import (
    atomic_write_json,
    discard_generation_file,
    restore_previous_generation,
    stage_previous_generation,
)

_CONFIG_ROOT = os.path.dirname(os.path.abspath(__file__))
_load_dotenv(os.path.join(_CONFIG_ROOT, "config.env"))

# Intentionally retained dead code for reference: an alternative shared-memory
# IPC approach for trends instead of the current priceanalysis.json file.
#   from multiprocessing import shared_memory
#   import shmutils as shmu
#   shm = shmu.shmConnectForWrite(shmu.shmname)   # At startup in __main__.
#   shmu.shmWrite(shm, all_trends)                # In the loop after write_all_trends.
#   ... finally: shm.close(); shm.unlink()        # At shutdown.
# ──────────────────────────────────────────────────────────────────────────────

# Debug tracing for weight zones and slope tuning is off by default to keep fleet
# logs clean. Enable PRICEANALYSIS_DEBUG when tuning.
_DEBUG = required_bool_env("PRICEANALYSIS_DEBUG")
PRICEANALYSIS_DRAW_MAX_POINTS = required_int_env("PRICEANALYSIS_DRAW_MAX_POINTS")


def _dbg(*args, **kwargs):
    if _DEBUG:
        print(*args, **kwargs)


price_cache_manager = None

def build_price_cache_manager(symbols=None, provider_names=None):
    global price_cache_manager
    import cacheManager as cm
    if provider_names:
        cm.get_current_price_manager(
            symbols=symbols,
            provider_names=provider_names,
            start_sync=False,
        )
    price_cache_manager = cm.get_cache_manager("SparsePrice", symbols=symbols)  # Dictionary keyed by symbol.

def priceLstFor(symbol: str) -> List[Tuple[int, float]]:

    if price_cache_manager is None:
        raise RuntimeError("The price cache manager was not initialised. Run build_price_cache_manager() first.")

    manager = price_cache_manager.get(symbol)
    if manager is None:
        return []

    # Read the symbol's current cached series under the manager lock when present.
    lock = getattr(manager, "lock", None)
    if lock is not None:
        with lock:
            raw = list(manager.cache.get(symbol, []))
    else:
        raw = list(manager.cache.get(symbol, []))
    out: List[Tuple[int, float]] = []
    for item in raw:
        try:
            ts_raw, p_raw = item[0], item[1]
            if ts_raw is None or p_raw is None or isinstance(ts_raw, bool) or isinstance(p_raw, bool):
                continue
            ts_f = float(ts_raw)
            p_f = float(p_raw)
            if math.isfinite(ts_f) and ts_f > 0 and math.isfinite(p_f) and p_f > 0:
                out.append((int(ts_f), p_f))
        except (TypeError, ValueError, IndexError, OverflowError):
            continue
    return out


_draw_fig = {}  # Reused symbol-to-(figure, axis) mapping; see drawPriceLst.


def _draw_point_limit() -> int:
    """Return the renderer point limit; it does not affect trend calculations."""
    return max(2, PRICEANALYSIS_DRAW_MAX_POINTS)


def _bounded_plot_indices(length: int, max_points: Optional[int] = None) -> np.ndarray:
    """Return uniform plot indices with retained endpoints and bounded memory."""
    if length <= 0:
        return np.empty(0, dtype=np.intp)
    limit = _draw_point_limit() if max_points is None else max(2, int(max_points))
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
    ax.set_title(f"{symbol} - Trend {trend_direction}, durata {duration_hours:.2f}h")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    fig.autofmt_xdate()

    out_dir = os.path.join(_CONFIG_ROOT, "logger", "plots")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"plot_{symbol}.png"))



def weighted_moving_average(prices: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate the weighted moving average, giving newer values greater weight.
    """
    if len(prices) == 0 or window <= 0:
        return np.zeros_like(prices, dtype=float)
    window = min(int(window), len(prices))
    wma = np.zeros_like(prices, dtype=float)
    weights = np.arange(1, window + 1)
    weight_sum = float(np.sum(weights))
    for i in range(window - 1, len(prices)):
        wma[i] = np.sum(prices[i - window + 1:i + 1] * weights) / weight_sum
    return wma


# Weighted moving average (WMA).
def trend_wma(symbol: str, window_hours: int = 6):
    data = priceLstFor(symbol)
    if len(data) < 2:
        return None

    data = sorted(data, key=lambda x: x[0])
    timestamps, prices = zip(*data)
    timestamps = np.array(timestamps, dtype=float)
    if timestamps[-1] > 1e11:
        timestamps = timestamps / 1000.0
    prices = np.array(prices, dtype=float)

    delta = float(np.median(np.diff(timestamps)))
    if not np.isfinite(delta) or delta <= 0:
        return None
    points_per_hour = max(1, int(round(3600.0 / delta)))
    window = max(1, min(points_per_hour * window_hours, len(prices)))

    wma_prices = weighted_moving_average(prices, window)

    # Determine direction by comparing the newest WMA with its predecessor.
    if wma_prices[-1] > wma_prices[-2]:
        trend_direction = 'up'
    else:
        trend_direction = 'down'

    # Visualization.
    plt.figure(figsize=(12,5))
    plt.plot(timestamps, prices, label='Price', color='blue')
    plt.plot(timestamps, wma_prices, label=f'WMA {window_hours}h', color='red', linewidth=2)
    plt.xlabel('Timestamp')
    plt.ylabel('Price')
    plt.title(f"{symbol} - Trend WMA: {trend_direction}")
    plt.legend()
    plt.show()

    return {'direction': trend_direction}



# Intentionally retained reference implementation for Holt's Linear Trend. The
# active path uses detect_long_term_trend, but this may be resumed later.
# from statsmodels.tsa.holtwinters import Holt
# def trend_holt(symbol: str, smoothing_level: float = 0.3, smoothing_slope: float = 0.1, forecast_hours: int = 1):
    # data = priceLstFor(symbol)
    # if len(data) < 2:
        # return None
    # data = sorted(data, key=lambda x: x[0])
    # timestamps, prices = zip(*data)
    # timestamps = np.array(timestamps)
    # prices = np.array(prices)
    # delta = np.median(np.diff(timestamps))
    # points_per_hour = int(3600 / delta)
    # model = Holt(prices).fit(smoothing_level=smoothing_level, smoothing_slope=smoothing_slope, optimized=False)
    # fitted = model.fittedvalues
    # forecast_points = forecast_hours * points_per_hour   # Short trend forecast.
    # future = model.forecast(forecast_points)
    # trend_direction = 'up' if future[-1] > fitted[-1] else 'down'
    # return {'direction': trend_direction, 'fitted': fitted, 'forecast': future}
# ──────────────────────────────────────────────────────────────────────────────


def slope_tolerance_per_(symbol, price,
                              base_tolerance = 0.0015,
                              ):
    min_tol = 0.0005
    max_tol = 2000.0
    if price is None or isinstance(price, bool):
        return min_tol
    try:
        price_f = float(price)
    except (TypeError, ValueError, OverflowError):
        return min_tol
    if not math.isfinite(price_f) or price_f <= 0.0:
        return min_tol
    base_f = 0.0015
    if base_tolerance is not None and not isinstance(base_tolerance, bool):
        try:
            bf = float(base_tolerance)
            if math.isfinite(bf) and bf > 0.0:
                base_f = bf
        except (TypeError, ValueError, OverflowError):
            pass
    relative_tolerance = base_f * price_f
    adaptive_tolerance = min(max(relative_tolerance, min_tol), max_tol)
    return adaptive_tolerance



def _fast_linear_regression_1d(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Compute OLS linear slope and intercept in O(N) without LAPACK/SVD matrix overhead."""
    dx = x - x[0]
    mean_dx = float(np.mean(dx))
    mean_y = float(np.mean(y))
    dev_x = dx - mean_dx
    var_x = float(np.dot(dev_x, dev_x))
    if var_x == 0.0:
        return 0.0, mean_y
    slope = float(np.dot(dev_x, y) / var_x)
    intercept = mean_y - slope * mean_dx
    return slope, intercept


# Intentionally retained legacy trend detection based on block-to-block slopes,
# average reference slope, and relative tolerance. The active fixed path uses
# time-based windows, Mann-Kendall, and noise tolerance.
def getTrendLongTerm(symbol: str, window_hours: int = 24, step_hours: int = 8,
                                slope_tolerance: float = 0.0028, persistence_factor: float = 1.5,
                                lookback_days=30, draw: bool = True) -> Optional[dict]:

    data: List[Tuple[int, float]] = priceLstFor(symbol)
    if len(data) < 2:
        return None
    data = sorted(data, key=lambda x: x[0])
    
    # Select the last N days.
    cutoff_timestamp = time.time() - (lookback_days * 86400)
    data = [(ts, p) for ts, p in data if (ts / 1000.0 if ts > 1e11 else ts) > cutoff_timestamp]
    if len(data) < 2:
        return None
    
    timestamps, prices = zip(*data)
    timestamps = np.array(timestamps, dtype=float)
    if timestamps[-1] > 1e11:
        timestamps = timestamps / 1000.0  # Convert milliseconds to seconds.
    prices = np.array(prices, dtype=float)
    
    delta = float(np.median(np.diff(timestamps)))
    if not np.isfinite(delta) or delta <= 0:
        return None
    points_per_hour = max(1, int(round(3600.0 / delta))) # Price points per hour.
    window = max(1, min(points_per_hour * window_hours, len(prices))) # Points per window.
    step = max(1, points_per_hour * step_hours)     # Points per step.
    
    _dbg(f"[DEBUG] {symbol}: numar puncte={len(prices)}, window={window}, step={step}, delta(s)={delta}")
    _dbg(f"[DEBUG] {symbol}: window count={len(prices)/window}, step count in price {len(prices)/step}")
    _dbg(f"[DEBUG] {symbol}: slope_tolerance={slope_tolerance}")
 
    last_slope_h = None
    sum_slope = 0
    trend_start_ts = timestamps[-1]
    trend_ref_slope_h = None
    trend_ref_count = 1
    trend_block = 0
    trend_block_ups = 0
    trend_block_indices = []
    
    trend_block_indices_test=[]
    
    for start in range(len(prices) - window, -1, -step):
        _dbg(f"[DEBUG] start {start}")
        trend_block +=1
        end = start + window
        x_block = timestamps[start:end] - timestamps[start]
        y_block = prices[start:end]

        slope_s, intercept = _fast_linear_regression_1d(x_block, y_block)  # price increase per second
        
        trend_block_indices_test.append((0, window))

        slope_h = slope_s * 3600 # slope per h
        if slope_h > 0 :
            trend_block_ups +=1

        avg_price = prices[0]
        relative_tolerance = slope_tolerance_per_(symbol, avg_price, slope_tolerance) 

        _dbg(f"[DEBUG] {symbol}: relative_tolerance={relative_tolerance}, slope_h={slope_h}, last_slope_h={last_slope_h}")
        #drawPriceLst(x_block, y_block, trend_block_indices, symbol, "up", slope_h)
     
        if trend_ref_slope_h is None or last_slope_h is None:
            trend_ref_slope_h = slope_h
            trend_start_ts = timestamps[start]
            last_slope_h = slope_h

        continue_trend = True
                    
        if(trend_ref_slope_h * slope_h < 0):  # opposite trend sign
            if(len(trend_block_indices) == 0):
                continue
            avg_slope = sum_slope / len(trend_block_indices)
            _dbg(f"[DEBUG] the current trend differs {slope_h}. Compared against trend_ref_slope_h={trend_ref_slope_h} and avg_slope={avg_slope}")
            if abs(slope_h - trend_ref_slope_h) >= relative_tolerance: # Significant difference from the starting trend.
                continue_trend = False;
            if abs(slope_h - avg_slope) >= relative_tolerance:  # Large difference from the mean.
                continue_trend = False;
        else:
            continue_trend = True
                        
        if continue_trend:
            if (trend_ref_slope_h * slope_h < 0):  # sign changed
                trend_ref_slope_h = slope_h
                trend_ref_count = 1
                print(f"CONTINUE ... ")
            else:  # update the mean reference slope
                
                # w < 1 makes the previous mean count less than one new observation.
                #trend_ref_slope_h = (w * trend_ref_slope_h + slope_h) / (w + 1)              
                trend_ref_slope_h =  (trend_ref_slope_h * trend_ref_count + slope_h) / (trend_ref_count + 1);
                trend_ref_count += 1
            
            sum_slope += slope_h            
            trend_block_indices.append((start, end))
            last_slope_h = slope_h
        else:
            # The trend has broken.
            print(f"BREAK!")
            break
               
    duration_seconds = timestamps[-1] - trend_start_ts
    duration_hours = duration_seconds / 3600
    estimated_future_hours = duration_hours * persistence_factor
    
    if duration_seconds <= 0:
        print(f"[{symbol}] duration_seconds={duration_seconds}, not enough data for a trend.")
        return None
    
    if trend_ref_slope_h is None:
        print(f"[{symbol}] trend_ref_slope_h is None, the trend direction cannot be determined.")
        return None        # Not enough data to calculate slope
    
    print(f"trend_block {trend_block} and trend_block_ups {trend_block_ups}")
    trend_direction = 'up' if trend_ref_slope_h > 0 else 'down'

    if draw:
        drawPriceLst(timestamps, prices, trend_block_indices, symbol, trend_direction, duration_hours)

    return {
        'timestamp': int(time.time()),
        'direction': trend_direction,
        'start_timestamp': trend_start_ts,
        'duration_seconds': duration_seconds,
        'estimated_future_hours': estimated_future_hours
    }

            

def format_duration(seconds):
    """Convert seconds to the human-readable ``Xd Yh Zm`` format."""
    if seconds is None or isinstance(seconds, bool):
        return "0m"
    try:
        sec_f = float(seconds)
    except (TypeError, ValueError, OverflowError):
        return "0m"
    if not math.isfinite(sec_f) or sec_f <= 0:
        return "0m"
    days = int(sec_f // 86400)
    hours = int((sec_f % 86400) // 3600)
    minutes = int((sec_f % 3600) // 60)
    
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    
    return " ".join(parts) if parts else "0m"


def format_timestamp(ts):
    """Convert a timestamp to a human-readable representation."""
    if ts is None or isinstance(ts, bool):
        return datetime.fromtimestamp(0).strftime('%Y-%m-%d %H:%M:%S')
    try:
        ts_f = float(ts)
        if not math.isfinite(ts_f) or ts_f < 0:
            ts_f = 0.0
        return datetime.fromtimestamp(ts_f).strftime('%Y-%m-%d %H:%M:%S')
    except (TypeError, ValueError, OSError, OverflowError):
        return datetime.fromtimestamp(0).strftime('%Y-%m-%d %H:%M:%S')


# Minimum points required for a credible window slope; fewer points constitute a gap.
MIN_POINTS_PER_WINDOW = 4


_trend_memo_cache: dict = {}
_TREND_MEMO_MAX_ENTRIES = 128
_TREND_MEMO_NONE = object()


def _store_trend_memo(cache_key: tuple, value):
    if len(_trend_memo_cache) > _TREND_MEMO_MAX_ENTRIES:
        _trend_memo_cache.clear()
    _trend_memo_cache[cache_key] = _TREND_MEMO_NONE if value is None else value
    if value is None:
        return None
    out = dict(value)
    if "blocks" in out and isinstance(out["blocks"], list):
        out["blocks"] = list(out["blocks"])
    return out


def detect_long_term_trend(timestamps, prices, window_hours=24, step_hours=8,
                           min_consecutive_blocks=3, noise_tolerance=2,
                           min_points_per_window=MIN_POINTS_PER_WINDOW,
                           detection_lag_hours=0.0, mk_alpha=None):
    """Detect trends in time-defined windows rather than point-count windows.

    ``timestamps`` are ascending seconds. Each window covers ``[t, t+window_hours]``
    and advances by real ``step_hours``. A window below ``min_points_per_window`` is a
    gap and terminates the trend rather than inventing observations across the gap.

    Return ``{direction, start_timestamp, duration_seconds,
    estimated_future_hours, current_slope_h, blocks(index pairs for plotting)}``
    or ``None``.
    """
    if (
        timestamps is None
        or prices is None
        or isinstance(timestamps, (bool, str, bytes))
        or isinstance(prices, (bool, str, bytes))
    ):
        return None
    if any(
        isinstance(v, bool)
        for v in (
            window_hours,
            step_hours,
            min_consecutive_blocks,
            noise_tolerance,
            min_points_per_window,
            detection_lag_hours,
        )
    ):
        return None
    try:
        wh_f = float(window_hours)
        sh_f = float(step_hours)
        mcb_i = int(min_consecutive_blocks)
        nt_i = int(noise_tolerance)
        mpw_i = int(min_points_per_window)
        lag_f = float(detection_lag_hours)
    except (TypeError, ValueError, OverflowError):
        return None
    if not (math.isfinite(wh_f) and wh_f > 0.0 and math.isfinite(sh_f) and sh_f > 0.0 and math.isfinite(lag_f) and lag_f >= 0.0):
        return None
    if mcb_i < 1 or nt_i < 0 or mpw_i < 2:
        return None
    window_hours = wh_f
    step_hours = sh_f
    min_consecutive_blocks = mcb_i
    noise_tolerance = nt_i
    min_points_per_window = mpw_i
    detection_lag_hours = lag_f

    if isinstance(timestamps, np.ndarray) and timestamps.dtype == bool:
        return None
    if isinstance(prices, np.ndarray) and prices.dtype == bool:
        return None
    try:
        if not isinstance(timestamps, np.ndarray) and any(isinstance(x, bool) for x in timestamps):
            raise TypeError("bool element")
        if not isinstance(prices, np.ndarray) and any(isinstance(x, bool) for x in prices):
            raise TypeError("bool element")
        timestamps = np.asarray(timestamps, dtype=float)
        prices = np.asarray(prices, dtype=float)
    except (TypeError, ValueError, OverflowError):
        try:
            raw_ts = list(timestamps)
            raw_px = list(prices)
        except TypeError:
            return None
        if len(raw_ts) != len(raw_px) or len(raw_ts) < 2:
            return None
        clean_ts: list[float] = []
        clean_px: list[float] = []
        for t_item, p_item in zip(raw_ts, raw_px):
            if isinstance(t_item, bool) or isinstance(p_item, bool):
                continue
            try:
                tf = float(t_item)
                pf = float(p_item)
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(tf) and tf > 0.0 and math.isfinite(pf) and pf > 0.0:
                clean_ts.append(tf)
                clean_px.append(pf)
        timestamps = np.asarray(clean_ts, dtype=float)
        prices = np.asarray(clean_px, dtype=float)
    if timestamps.ndim != 1 or prices.ndim != 1 or len(prices) != len(timestamps):
        return None
    finite_mask = np.isfinite(timestamps) & (timestamps > 0) & np.isfinite(prices) & (prices > 0)
    if not np.all(finite_mask):
        timestamps = timestamps[finite_mask]
        prices = prices[finite_mask]
    n_pts = len(timestamps)
    if n_pts < 2:
        return None
    if np.any(np.diff(timestamps) < 0):
        order = np.argsort(timestamps, kind="mergesort")
        timestamps = timestamps[order]
        prices = prices[order]

    t_end, t_first = float(timestamps[-1]), float(timestamps[0])
    p_first = float(prices[0])
    p_mid = float(prices[n_pts // 2])
    p_last = float(prices[-1])
    cache_key = (
        n_pts,
        t_first,
        t_end,
        p_first,
        p_mid,
        p_last,
        window_hours,
        step_hours,
        min_consecutive_blocks,
        noise_tolerance,
        min_points_per_window,
        detection_lag_hours,
        mk_alpha,
    )
    cached = _trend_memo_cache.get(cache_key)
    if cached is _TREND_MEMO_NONE:
        return None
    if cached is not None:
        if isinstance(cached, dict):
            out = dict(cached)
            if "blocks" in out and isinstance(out["blocks"], list):
                out["blocks"] = list(out["blocks"])
            return out
        return cached

    window_sec = window_hours * 3600.0
    step_sec = step_hours * 3600.0

    def slope_h(t_lo, t_hi):
        """Return hourly slope and index bounds for ``[t_lo,t_hi)``, or None for a gap."""
        lo = int(np.searchsorted(timestamps, t_lo, "left"))
        hi = int(np.searchsorted(timestamps, t_hi, "left"))
        if hi - lo < min_points_per_window:
            return None, (lo, hi)
        x, y = timestamps[lo:hi], prices[lo:hi]
        s, _ = _fast_linear_regression_1d(x, y)
        if not math.isfinite(s):
            return None, (lo, hi)
        return s * 3600.0, (lo, hi)

    cur, cur_idx = slope_h(t_end - window_sec, t_end + 1.0)
    if cur is None or cur == 0.0:
        return _store_trend_memo(cache_key, None)    # Insufficient recent data or flat window.
    if mk_alpha:
        # Mann-Kendall requires the current window's slope to be statistically
        # significant rather than noise before reporting a direction.
        from intelligence.internal.state.persistence import calculate_mann_kendall as mann_kendall
        _, _, p_mk = mann_kendall(prices[cur_idx[0]:cur_idx[1]])
        if p_mk > mk_alpha:
            return _store_trend_memo(cache_key, None)
    current_sign = float(np.sign(cur))

    blocks = [cur_idx]
    consecutive, noise = 1, 0
    confirm_lo = cur_idx[0]      # Oldest point confirming the current direction.
    confirm_pos = 0              # Position of the last confirming block.
    t_ws = t_end - window_sec - step_sec
    while t_ws >= t_first:
        s, idx = slope_h(t_ws, t_ws + window_sec)
        if s is None:
            break                                    # Do not confirm a trend across a gap.
        if np.sign(s) == current_sign:
            blocks.append(idx); consecutive += 1; noise = 0
            confirm_lo = idx[0]; confirm_pos = len(blocks) - 1
        elif noise < noise_tolerance:
            noise += 1; blocks.append(idx)           # Tentatively continue across tolerated noise.
        else:
            break                                    # Excess noise ends the trend here.
        t_ws -= step_sec

    # Require the minimum confirmed blocks in the current direction. A one-day
    # bounce against a four-day decline is not a four-day upward trend.
    if consecutive < min_consecutive_blocks:
        return _store_trend_memo(cache_key, None)

    blocks = blocks[:confirm_pos + 1]                # Exclude unconfirmed trailing noise.
    # Duration includes the confirmed interval plus detection lag because the real
    # trend begins before confirmation. Clamp it to the available data span.
    confirmed_start = float(timestamps[confirm_lo])
    duration_seconds = (t_end - confirmed_start) + detection_lag_hours * 3600.0
    duration_seconds = min(duration_seconds, t_end - t_first)
    trend_start_ts = t_end - duration_seconds
    if duration_seconds <= 0:
        return _store_trend_memo(cache_key, None)
    res = {
        'direction': 'up' if current_sign > 0 else 'down',
        'start_timestamp': float(trend_start_ts),
        'duration_seconds': float(duration_seconds),
        'estimated_future_hours': float(duration_seconds / 3600.0 * 0.5),
        'current_slope_h': float(cur),
        'blocks': list(blocks),
    }
    return _store_trend_memo(cache_key, res)


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
    data: List[Tuple[int, float]] = priceLstFor(symbol)
    if len(data) < 2:
        return None

    data = sorted(data, key=lambda x: x[0])
    
    # Select the last N days.
    cutoff_timestamp = time.time() - (lookback_days * 86400)
    data = [(ts, p) for ts, p in data if ts/1000 > cutoff_timestamp]
    
    if len(data) < 2:
        print(f"[{symbol}] Not enough data in the last {lookback_days} days")
        return None
    
    timestamps, prices = zip(*data)
    timestamps = np.array(timestamps) / 1000      # Milliseconds to seconds.
    prices = np.array(prices)

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
    from intelligence.internal.state.persistence import calculate_hurst_exponent as hurst_rs, classify_hurst_regime as hurst_regime
    h = hurst_rs(prices)
    res['hurst'] = h
    res['regime'] = hurst_regime(h)
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

    if draw:
        drawPriceLst(timestamps, prices, res['blocks'], symbol, direction, dur / 3600.0)

    return {
        'timestamp': int(time.time()),
        'direction': direction,
        'start_timestamp': res['start_timestamp'],
        'duration_seconds': dur,
        'estimated_future_hours': res['estimated_future_hours'],
    }

# Persist all trend results with human-readable output.
def write_all_trends(all_trends, filename=None):
    """Atomically write trend results and display a human-readable summary."""
    filename = filename or os.path.join(_CONFIG_ROOT, "priceanalysis.json")

    print("\n" + "="*80)
    print("TREND SUMMARY".center(80))
    print("="*80)
    
    for symbol, trend_data in all_trends.items():
        if trend_data is None:
            print(f"\n{symbol}: ❌ Not enough data")
            continue
            
        direction = trend_data['direction']
        emoji = "📈" if direction == 'up' else "📉"
        
        start_str = format_timestamp(trend_data['start_timestamp'])
        duration_str = format_duration(trend_data['duration_seconds'])
        duration_days = trend_data['duration_seconds'] / 86400
        
        future_hours = trend_data['estimated_future_hours']
        future_str = format_duration(future_hours * 3600)
        future_days = future_hours / 24
        
        print(f"\n{symbol}")
        print(f"  {emoji} {direction.upper()}")
        print(f"  Start:    {start_str}")
        print(f"  Duration: {duration_str} ({duration_days:.1f} days)")
        print(f"  Estimate: ~{future_str} ({future_days:.1f} days)")
    
    print("\n" + "="*80 + "\n")
    
    backup_path = None
    had_prior = os.path.exists(filename)
    if had_prior:
        try:
            backup_path = stage_previous_generation(filename)
        except Exception:
            backup_path = None
    try:
        atomic_write_json(filename, all_trends, indent=2)
        if backup_path is not None:
            discard_generation_file(backup_path)
        print(f"Results were written to {filename}")
    except Exception as exc:
        if backup_path is not None and os.path.exists(backup_path):
            try:
                restore_previous_generation(filename, backup_path=backup_path)
            except Exception:
                pass
        elif not had_prior:
            discard_generation_file(filename)
        print(f"Could not write {filename}: {exc}")
    
    return all_trends
    
    
def get_weight_for_cash_permission_at_quant_time(symbol, order_type, T_quanta=None, quant_seconds=3600*24, draw=False):
    """Use an empirical per-symbol T when ``T_quanta`` is None.

    trend_survival.estimate_T blends history with a 14-day prior, favoring
    empirical data when enough episodes exist, and caches it on disk. Passing 14
    explicitly restores legacy behavior.
    """
    import cacheManager as cm
    global last_timestamp, last_w, last_calc_time, last_duration

    if not isinstance(symbol, str) or not symbol.strip():
        return None
    if not isinstance(order_type, str):
        return None
    side_u = order_type.strip().upper()
    if side_u == "B":
        side_u = "BUY"
    elif side_u == "S":
        side_u = "SELL"
    if side_u not in ("BUY", "SELL"):
        return None
    if isinstance(T_quanta, bool) or isinstance(quant_seconds, bool):
        return None

    if T_quanta is None:
        try:
            from intelligence.internal.state.survival import estimate_T
            est = estimate_T(symbol)
            T_quanta = est["T"]
            print(f"[{symbol}] T AUTO (empiric hibrid): {T_quanta} zile  "
                  f"(n={est['n']} episoade, w_empiric={est['w']}, "
                  f"mediana={est.get('median_d')}z, P90={est.get('p90_d')}z)")
        except Exception as e:
            T_quanta = 14
            print(f"[{symbol}] estimarea T failed ({e}) — using the prior T=14")

    all_trend_data = cm.get_cache_manager("LongTrend").cache
    sym_u = (symbol or "").strip().upper()
    sym_clean = sym_u.replace("/", "")
    items = (
        all_trend_data.get(symbol)
        or all_trend_data.get(sym_u)
        or all_trend_data.get(sym_clean)
    )
    if not items:
        # Fallback to normalized base asset match (e.g. ARBUSD -> ARBUSDC)
        from order_guard import _normalize_base_asset
        s_base = _normalize_base_asset(symbol)
        for k, v in all_trend_data.items():
            if v and _normalize_base_asset(k) == s_base:
                items = v
                break
    if not items:
        print(f"Symbol {symbol} is not present in the trends that were read.")
        return None

    # Retrieve latest trend record; an explicit trailing None is a neutral snapshot
    trend = None
    if isinstance(items, list):
        for entry in reversed(items):
            if entry is None:
                break
            if isinstance(entry, dict) and entry.get("duration_seconds") is not None:
                trend = entry
                break
    elif isinstance(items, dict):
        trend = items

    if trend is None:
        print(f" No trend in cache for symbol {symbol}.")
        return None

    raw_dur = trend.get("duration_seconds")
    if raw_dur is None or isinstance(raw_dur, bool):
        return None
    try:
        dur_sec = float(raw_dur)
        t_quanta_f = float(T_quanta)
        quant_sec_f = float(quant_seconds)
    except (TypeError, ValueError, OverflowError):
        return None
    if not (
        math.isfinite(dur_sec)
        and dur_sec > 0
        and math.isfinite(t_quanta_f)
        and t_quanta_f > 0
        and math.isfinite(quant_sec_f)
        and quant_sec_f > 0
    ):
        print(f"[{symbol}] duration_seconds/T_quanta invalid, return None")
        return None

    direction = str(trend.get("direction") or "").strip().lower()
    if direction == "bull":
        direction = "up"
    elif direction == "bear":
        direction = "down"
    if direction not in ("up", "down"):
        return None

    duration_days = dur_sec / 86400.0
    raw_start_ts = trend.get("start_timestamp")
    start_ts_val = 0.0
    if raw_start_ts is not None and not isinstance(raw_start_ts, bool):
        try:
            st_f = float(raw_start_ts)
            if math.isfinite(st_f):
                start_ts_val = st_f
        except (TypeError, ValueError, OverflowError):
            start_ts_val = 0.0
    print(f"Trend read from the cache manager for symbol {symbol}: {trend}")
    print(f"   Start trend:     {format_timestamp(start_ts_val)}")
    print(f"   Duration:        {format_duration(dur_sec)} ({duration_days:.1f} days)")
    timestamp = trend.get("timestamp")
    # Include order type, T, and direction in the memo key because their weights differ
    # and automatic T or direction may change after re-estimation.
    memo_key = (symbol, side_u, t_quanta_f, direction)
    curr_now = time.time()
    cache_age = curr_now - last_calc_time.get(memo_key, 0.0)
    dur_drift = abs(dur_sec - last_duration.get(memo_key, 0.0))

    # Recalculate if timestamp changed, cache age > 300s (5min fallback TTL) or < 0 (clock skew), or duration drifted > 300s
    if (
        timestamp is not None
        and timestamp == last_timestamp.get(memo_key)
        and 0.0 <= cache_age < 300.0
        and dur_drift < 300.0
    ):
        cached_w = last_w.get(memo_key)
        if cached_w is not None and len(cached_w) > 0 and not np.isnan(cached_w[0]):
            print(f"not new timestamp (age={cache_age:.1f}s), use weight from mem cache.")
            return float(cached_w[0])

    trend_len_quanta = dur_sec / quant_sec_f
    if not math.isfinite(trend_len_quanta) or trend_len_quanta <= 0:
        print(f"[{symbol}] duration_seconds invalid, return None")
        return None

    t, w = get_trade_weight(
        T=t_quanta_f,
        trend_len=trend_len_quanta,
        trend=direction,
        order_type=side_u
    )

    if len(w) == 0:
        print(f"[{symbol}] w is empty, returning None")
        return None

    current_weight = float(w[0])  # The slice's first value is the current weight.

    if np.isnan(current_weight) or current_weight <= 0:
        print(f"[{symbol}] Invalid weight: {current_weight}, returning None")
        return None

    print(f"[{symbol}] primele 5 ponderi: {w[:5]}")
    print(f"Suma tuturor {len(w)} ponderi = {w.sum():.4f}")

    if draw:
        plt.plot(t, w, label=symbol)
        plt.legend()
        plt.show()

    last_w[memo_key] = w        # w is already sliced from current position.
    last_timestamp[memo_key] = timestamp
    last_calc_time[memo_key] = curr_now
    last_duration[memo_key] = dur_sec
    return current_weight     # w[0] is the current weight.

last_timestamp = {}
last_w = {}
last_calc_time = {}
last_duration = {}


# Zone 1: 0 to T is Gaussian, confident in the middle and uncertain at endpoints.
# Zone 2: T to T*(1+percentage) is an over-age but persistent trend; high weight (0.86).
# Zone 3: above T*(1+percentage) is a very old trend; use a conservative weight (0.22).

# Aligned BUY+UP or SELL+DOWN scales the Gaussian peak to peak_weight.
# Middle -> 0.95 (maximum allocation); endpoints -> about 0.17 (small allocation).

# Countertrend SELL+UP or BUY+DOWN uses the inverse global Gaussian.
# Middle -> 0.02 (do not trade); both endpoints -> about 0.13 to 0.15.

def get_trade_weight(T, trend_len, trend, order_type,
                     exceed_percent=0.4, max_against_trend=0.15,
                     peak_weight=0.95, min_weight=0.02, lindy_plateau=True):
    if isinstance(T, bool) or isinstance(trend_len, bool) or not isinstance(order_type, str) or not isinstance(trend, str):
        return np.array([]), np.array([])
    try:
        T_f = float(T)
        trend_len_f = float(trend_len)
    except (TypeError, ValueError, OverflowError):
        return np.array([]), np.array([])
    if not (math.isfinite(T_f) and T_f > 0 and math.isfinite(trend_len_f) and trend_len_f >= 0):
        return np.array([]), np.array([])
    T = T_f
    trend_len = trend_len_f
    def _clean_param(val, default):
        if val is None or isinstance(val, bool):
            return default
        try:
            vf = float(val)
            return vf if math.isfinite(vf) else default
        except (TypeError, ValueError, OverflowError):
            return default
    exceed_percent = max(0.0, _clean_param(exceed_percent, 0.4))
    max_against_trend = max(0.0, _clean_param(max_against_trend, 0.15))
    peak_weight = max(0.0, _clean_param(peak_weight, 0.95))
    min_weight = max(0.0, _clean_param(min_weight, 0.02))
    side_u = order_type.strip().upper()
    if side_u == "B":
        side_u = "BUY"
    elif side_u == "S":
        side_u = "SELL"
    if side_u not in ("BUY", "SELL"):
        return np.array([]), np.array([])
    trend_norm = trend.strip().lower()
    if trend_norm == "bull":
        trend_norm = "up"
    elif trend_norm == "bear":
        trend_norm = "down"
    if trend_norm not in ("up", "down"):
        return np.array([]), np.array([])
    aligned = (
        (side_u == "BUY"  and trend_norm == "up") or
        (side_u == "SELL" and trend_norm == "down")
    )

    T_extended = T * (1 + exceed_percent)

    # Zone 2: an over-age but persistent trend has strong aligned momentum.
    if T < trend_len <= T_extended:
        w_val = 0.86 if aligned else max_against_trend
        _dbg(f"[DEBUG] Zona 2: trend_len={trend_len:.2f} exceeds T={T} but is below T_extended={T_extended}. Aligned={aligned}, return {w_val}  ")
        return np.array([0.0]), np.array([w_val])

    # Zone 3: use conservative weights in both directions for a very old trend.
    if trend_len > T_extended:
        w_val = 0.22 if aligned else max_against_trend
        _dbg(f"[DEBUG] Zone 3: trend_len={trend_len:.2f} is above T_extended={T_extended}. Aligned={aligned}, return {w_val} ")
        return np.array([0.0]), np.array([w_val])

    # Zone 1: Gaussian over the full T, sliced from the trend's current age.
    # Clamp index to t_steps-1 so trend_len==T leaves a nonempty slice at the zone seam.
    t_steps = max(1, int(round(T)))
    idx = max(0, min(int(trend_len), t_steps - 1))
    t_full, w_full = u.gaussian_weights_from_idx(T=t_steps, idx=0)
    if len(w_full) == 0:
        _dbg(f"[DEBUG] Zone 1: gaussian_weights_from_idx returned empty. return [0.05]")
        return np.array([0.0]), np.array([0.05])
    # utils normalizes a probability distribution. Rescale its peak to one for
    # trading weights, avoiding the former 8-40x zone-scale mismatch.
    w01_full = w_full / w_full.max()                  # 0..1 with a peak of one.
    if lindy_plateau:
        # Empirical BTC and TAO survival data supports a Lindy plateau: conditional
        # one-day survival remains roughly 0.65-0.75 after the midpoint. Hold the
        # curve at its peak rather than following the Gaussian's declining tail.
        peak_i = int(np.argmax(w01_full))
        w01_full = w01_full.copy()
        w01_full[peak_i:] = 1.0
    t_seq, w01 = t_full[idx:], w01_full[idx:]
    _dbg(f"[DEBUG] Zona 1: trend_len={trend_len:.2f}, slice from idx={idx} up to T={T}. Aligned={aligned}, gauss01[0]={w01[0]:.4f}")

    if aligned:
        w_seq = w01 * peak_weight
    else:
        # Invert the global curve, not its slice: midpoint maps to min_weight and
        # endpoints to max_against_trend.
        _dbg(f"[DEBUG] Order type {order_type} is not aligned with the trend {trend}, globally inverted, max_against_trend={max_against_trend}")
        w_seq = min_weight + (1.0 - w01) * (max_against_trend - min_weight)

    return t_seq, w_seq  # slice [idx..T-1]
    
    
    
    
UPDATE_AND_REFRESH_TREND = 60*1 # One minute.
if __name__ == "__main__":
    from instrument_registry import tracked_price_instruments
    symbols, _nb_syms, _nb_providers = tracked_price_instruments()
    build_price_cache_manager(symbols=symbols, provider_names=_nb_providers or None)
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

            print(f"write : {all_trends}")
            time.sleep(UPDATE_AND_REFRESH_TREND)
    except KeyboardInterrupt:
        print(f"Manual shutdown...")


######################
